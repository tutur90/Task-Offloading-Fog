"""
optuna_parallel.py
==================
Subclass of optuna.Study that replaces the default ThreadPoolExecutor-based
parallelism with true process-based parallelism (multiprocessing + spawn).

Public API
----------
ParallelStudy(Study)  — subclass with an extended optimize()
create_study(...)     — same signature as optuna.create_study, returns ParallelStudy
load_study(...)       — same signature as optuna.load_study,   returns ParallelStudy

New parameters in optimize()
-----------------------------
backend : "thread" | "process"   (default: "process")
    "thread"  → delegates to the original Study.optimize() (ThreadPoolExecutor,
                GIL-limited, Optuna default)
    "process" → spawns independent OS processes via multiprocessing/spawn
                (true CPU parallelism; objective must be picklable —
                 install cloudpickle to support closures)
worker_init : callable(worker_id: int) | None
    Called once at the start of each worker process before any trial runs.
    Useful for GPU pinning, logging setup, etc.

Usage
-----
    from utils.optuna_parallel import create_study
    import optuna

    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage("optuna.log")
    )
    study = create_study(
        study_name="my_study",
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    # thread backend (original Optuna behaviour, GIL-limited):
    study.optimize(objective, n_trials=100, n_jobs=4, backend="thread")

    # process backend (true CPU parallelism, spawn):
    study.optimize(objective, n_trials=100, n_jobs=4, backend="process")

    # with per-worker GPU pinning:
    def gpu_init(worker_id):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id % 4)

    study.optimize(objective, n_trials=100, n_jobs=4,
                   backend="process", worker_init=gpu_init)
"""

from __future__ import annotations

import os
import pickle
import multiprocessing as mp
from collections.abc import Callable, Iterable
from typing import Optional

import optuna
from optuna.study import Study
from optuna.trial import FrozenTrial


# ---------------------------------------------------------------------------
# Serialization: prefer cloudpickle (handles closures), fall back to pickle
# ---------------------------------------------------------------------------

def _dumps(obj: object) -> bytes:
    try:
        import cloudpickle
        return cloudpickle.dumps(obj)
    except ImportError:
        return pickle.dumps(obj)


def _loads(data: bytes) -> object:
    try:
        import cloudpickle
        return cloudpickle.loads(data)
    except ImportError:
        return pickle.loads(data)


# ---------------------------------------------------------------------------
# Sampler re-seeding: give each worker a different seed for diversity
# ---------------------------------------------------------------------------

def _reseed_sampler(sampler, worker_id: int):
    import copy

    base_seed = (
        getattr(sampler, "_seed", None)
        or getattr(sampler, "seed", None)
        or 0
    )
    new_seed = (base_seed + worker_id) if base_seed is not None else worker_id

    cls = type(sampler)
    try:
        if hasattr(sampler, "_param_grid"):          # GridSampler
            return cls(sampler._param_grid, seed=new_seed)
        return cls(seed=new_seed)
    except TypeError:
        return copy.deepcopy(sampler)


# ---------------------------------------------------------------------------
# Spawn-safe worker — MUST be a top-level function to be picklable
# ---------------------------------------------------------------------------

def _spawn_worker(
    worker_id: int,
    study_name: str,
    storage_pkl: bytes,
    sampler_pkl: bytes,
    objective_pkl: bytes,
    n_trials: int,
    init_pkl: Optional[bytes],
) -> None:
    """Entry point for each spawned worker process."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Optional per-worker setup (GPU pinning, etc.)
    if init_pkl is not None:
        _loads(init_pkl)(worker_id)

    storage   = _loads(storage_pkl)
    sampler   = _loads(sampler_pkl)
    objective = _loads(objective_pkl)

    # Load as a plain Study — no recursion into ParallelStudy needed
    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)


# ---------------------------------------------------------------------------
# QMC pre-sampling helpers
# ---------------------------------------------------------------------------

def _is_qmc_sampler(sampler) -> bool:
    """Return True if the sampler is QMC-based (generates a Sobol sequence)."""
    return isinstance(sampler, optuna.samplers.QMCSampler)


def _pre_enqueue_qmc(study: Study, n_trials: int) -> None:
    """
    Generate all QMC parameter sets in the current process and enqueue them
    as WAITING trials so that workers only evaluate, never suggest.

    This preserves the low-discrepancy property of the Sobol sequence, which
    would be broken if each worker generated its own independent subsequence.
    """
    # Use a temporary in-memory study with the same sampler to generate params
    # without polluting the real study with extra trials.
    tmp_study = optuna.create_study(sampler=study.sampler)

    print(f"  [optuna_parallel] Pre-sampling {n_trials} QMC points in main process...")
    for _ in range(n_trials):
        tmp_trial = tmp_study.ask()
        study.enqueue_trial(tmp_trial.params)

    print(f"  [optuna_parallel] {n_trials} trials enqueued.")


# ---------------------------------------------------------------------------
# ParallelStudy — proper subclass of optuna.Study
# ---------------------------------------------------------------------------

class ParallelStudy(Study):
    """
    Subclass of ``optuna.Study`` whose ``optimize()`` method accepts two extra
    keyword arguments — ``backend`` and ``worker_init`` — to switch between
    Optuna's default ``ThreadPoolExecutor`` and true process-based parallelism.

    Instantiate via ``create_study()`` or ``load_study()`` rather than directly.
    """

    def optimize(
        self,
        func: Callable,
        n_trials: int | None = None,
        timeout: float | None = None,
        n_jobs: int = 1,
        catch: Iterable[type[Exception]] | type[Exception] = (),
        callbacks: Iterable[Callable[[Study, FrozenTrial], None]] | None = None,
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
        backend: str = "process",
        worker_init: Callable[[int], None] | None = None,
    ) -> None:
        """Optimize an objective function.

        Identical to ``optuna.Study.optimize()`` with two additional arguments:

        Args:
            backend:
                ``'thread'`` — delegates entirely to ``Study.optimize()``
                (``ThreadPoolExecutor``, subject to the GIL).
                ``'process'`` — spawns independent OS processes via
                ``multiprocessing`` with the ``'spawn'`` start method,
                giving true CPU parallelism.
            worker_init:
                Optional ``callable(worker_id: int) -> None`` called once
                at the start of each worker process (process backend only).
        """
        if backend not in ("thread", "process"):
            raise ValueError(f"backend must be 'thread' or 'process', got {backend!r}")

        # ── thread backend: original Optuna behaviour ─────────────────────
        if n_jobs <= 1 or backend == "thread":
            super().optimize(
                func,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                catch=catch,
                callbacks=callbacks,
                gc_after_trial=gc_after_trial,
                show_progress_bar=show_progress_bar,
            )
            return

        # ── process backend: spawn OS processes ───────────────────────────
        if n_trials is None:
            raise ValueError("n_trials must be an integer when using backend='process'")

        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        # QMC samplers (Sobol sequences) must generate their full sequence in
        # a single process to preserve the low-discrepancy guarantee.
        # We pre-enqueue all n_trials parameter sets here before spawning, then
        # workers simply evaluate the waiting trials without suggesting new ones.
        if _is_qmc_sampler(self.sampler):
            _pre_enqueue_qmc(self, n_trials)

        # Distribute trials evenly across workers
        trials_per_worker = [n_trials // n_jobs] * n_jobs
        for i in range(n_trials % n_jobs):
            trials_per_worker[i] += 1

        # Serialize once — fail early with a clear error message
        try:
            storage_pkl   = _dumps(self._storage)
            objective_pkl = _dumps(func)
            init_pkl      = _dumps(worker_init) if worker_init is not None else None
        except Exception as exc:
            raise RuntimeError(
                f"Could not serialize for process backend: {exc}\n"
                "Install cloudpickle (`pip install cloudpickle`) to support "
                "closures and locally-defined objective functions."
            ) from exc

        ctx = mp.get_context("spawn")
        processes = []

        for i, n_t in enumerate(trials_per_worker):
            if n_t == 0:
                continue
            sampler_pkl = _dumps(_reseed_sampler(self.sampler, worker_id=i))
            p = ctx.Process(
                target=_spawn_worker,
                args=(i, self.study_name, storage_pkl, sampler_pkl,
                      objective_pkl, n_t, init_pkl),
                daemon=False,
            )
            p.start()
            print(f"  [optuna_parallel] Worker {i} started (PID {p.pid}, {n_t} trials)")
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                print(f"  [optuna_parallel] WARNING: worker PID {p.pid} "
                      f"exited with code {p.exitcode}")


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_journal_storage(path: str) -> optuna.storages.JournalStorage:
    """
    Create a JournalStorage backed by a local append-only log file.

    Safe for concurrent access by multiple processes.  The parent directory
    is created automatically if it does not exist.

    Args:
        path: Absolute or relative path to the ``.log`` file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(path)
    )


def create_study(
    study_name: str,
    storage,
    direction: str = "minimize",
    sampler=None,
    load_if_exists: bool = False,
) -> ParallelStudy:
    """
    Create (or load) an Optuna study and return it as a ``ParallelStudy``.

    Parameters are identical to ``optuna.create_study()``.
    """
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        load_if_exists=load_if_exists,
    )
    # Promote the Study instance to ParallelStudy in-place.
    # Safe because ParallelStudy adds no __slots__ and only overrides optimize().
    study.__class__ = ParallelStudy
    return study  # type: ignore[return-value]


def load_study(
    study_name: str,
    storage,
    sampler=None,
) -> ParallelStudy:
    """
    Load an existing Optuna study and return it as a ``ParallelStudy``.

    Parameters are identical to ``optuna.load_study()``.
    """
    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
    )
    study.__class__ = ParallelStudy
    return study  # type: ignore[return-value]
