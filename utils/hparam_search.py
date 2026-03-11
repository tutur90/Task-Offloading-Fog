"""
hparam_search.py
================
Lightweight hyperparameter search framework.

Compatible with Optuna's storage format and optuna-dashboard:

    optuna-dashboard --storage <storage_path>

Design principle — **centralized sampling**:
    All parameter combinations are generated once in the *main process*
    and enqueued as Optuna WAITING trials.  Worker processes simply pop
    and evaluate — they never run the sampler themselves.  This guarantees:

    - **Grid**   — no duplicate combinations across workers.
    - **QMC**    — the Sobol low-discrepancy property is preserved globally.
    - **Random** — reproducibility regardless of worker count.

Adding a custom sampler
-----------------------
Subclass :class:`Sampler`, implement :meth:`Sampler.sample`, and register
the new class in :data:`SAMPLERS`::

    class MySampler(Sampler):
        def sample(self, param_specs, n_trials, seed=42):
            ...
            return [{"lr": 0.01, "d_model": 128}, ...]

    SAMPLERS["mine"] = MySampler

Quick-start example
-------------------
::

    from utils.hparam_search import HparamSearch, GridSampler

    search = HparamSearch(
        param_specs={"model.d_model": [64, 128, 256], "training.lr": [1e-3, 5e-4]},
        sampler=GridSampler(),
        study_name="mlp_grid",
        storage_path="logs/Pakistan/Tuple50K/MLP/hparam_mlp.log",
    )

    def objective(params):
        # params → {"model.d_model": 128, "training.lr": 0.001}
        score = ...
        # Return a plain float, or a dict for richer dashboard metadata:
        return {"value": score, "val_metrics": [...], "test_metrics": [...]}

    best_params, best_value, study = search.run(objective)
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import pickle
from typing import Callable, Optional, Union

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Base Sampler
# ─────────────────────────────────────────────────────────────────────────────

class Sampler:
    """
    Base class for hyperparameter samplers.

    Subclasses must implement :meth:`sample`, which returns the *complete*
    list of parameter dicts to evaluate.  All sampling happens in the main
    process before any worker is spawned.

    To add a new sampler, subclass this class and register it in
    :data:`SAMPLERS` so it becomes available via the ``--sampler`` CLI flag.
    """

    def sample(
        self,
        param_specs: dict[str, list],
        n_trials: Optional[int],
        seed: int = 42,
    ) -> list[dict]:
        """
        Generate a list of parameter dicts.

        Parameters
        ----------
        param_specs:
            Mapping ``{param_name: [value1, value2, ...]}``.
        n_trials:
            Number of combinations to return.  ``None`` means "return all"
            (meaningful for :class:`GridSampler`; required for others).
        seed:
            Random seed for reproducibility.

        Returns
        -------
        list[dict]
            Each dict maps ``param_name`` to a chosen value.
        """
        raise NotImplementedError(f"{type(self).__name__}.sample() is not implemented.")


# ─────────────────────────────────────────────────────────────────────────────
# Built-in Samplers
# ─────────────────────────────────────────────────────────────────────────────

class GridSampler(Sampler):
    """
    Exhaustive Cartesian-product grid search.

    Generates every combination of the provided parameter values.
    When *n_trials* is given the list is (optionally shuffled and) truncated.

    Parameters
    ----------
    shuffle:
        If ``True`` (default) the combinations are shuffled with *seed*
        before truncation, giving diverse coverage when only a subset is run.
        Set to ``False`` for strict lexicographic order.
    """

    def __init__(self, shuffle: bool = True) -> None:
        self.shuffle = shuffle

    def sample(
        self,
        param_specs: dict[str, list],
        n_trials: Optional[int] = None,
        seed: int = 42,
    ) -> list[dict]:
        keys = list(param_specs.keys())
        combos: list[tuple] = list(itertools.product(*[param_specs[k] for k in keys]))
        if self.shuffle:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(combos))
            combos = [combos[i] for i in order]
        if n_trials is not None:
            combos = combos[:n_trials]
        return [dict(zip(keys, c)) for c in combos]


class RandomSampler(Sampler):
    """Uniform random sampling with replacement from each parameter's value list."""

    def sample(
        self,
        param_specs: dict[str, list],
        n_trials: int,
        seed: int = 42,
    ) -> list[dict]:
        rng = np.random.default_rng(seed)
        return [
            {k: rng.choice(v) for k, v in param_specs.items()}
            for _ in range(n_trials)
        ]


class QMCSampler(Sampler):
    """
    Quasi-Monte Carlo sampler using a scrambled Sobol sequence.

    Provides better space-filling coverage than pure random sampling.
    The *full* sequence is generated in one shot in the main process so
    that the low-discrepancy property is preserved across all workers.

    Requires ``scipy``.
    """

    def sample(
        self,
        param_specs: dict[str, list],
        n_trials: int,
        seed: int = 42,
    ) -> list[dict]:
        from scipy.stats import qmc as scipy_qmc

        keys = list(param_specs.keys())
        sobol = scipy_qmc.Sobol(d=len(keys), scramble=True, seed=seed)
        points = sobol.random(n_trials)  # (n_trials, n_dims), values in [0, 1)

        result: list[dict] = []
        for point in points:
            params: dict = {}
            for dim, key in enumerate(keys):
                values = param_specs[key]
                idx = min(int(point[dim] * len(values)), len(values) - 1)
                params[key] = values[idx]
            result.append(params)
        return result


#: Registry mapping CLI sampler names to sampler classes.
#: Add an entry here to expose a custom sampler via ``--sampler``.
SAMPLERS: dict[str, type[Sampler]] = {
    "grid":   GridSampler,
    "random": RandomSampler,
    "qmc":    QMCSampler,
}


# ─────────────────────────────────────────────────────────────────────────────
# Storage helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_journal_storage(path: str) -> optuna.storages.JournalStorage:
    """Create a JournalStorage backed by a local file, compatible with Optuna ≥ 4."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    abs_path = os.path.abspath(path)
    try:
        # Optuna ≥ 4.0: JournalFileBackend
        backend = optuna.storages.journal.JournalFileBackend(abs_path)
    except AttributeError:
        # Optuna < 4.0 fallback
        backend = optuna.storages.JournalFileStorage(abs_path)  # type: ignore[attr-defined]
    return optuna.storages.JournalStorage(backend)


# ─────────────────────────────────────────────────────────────────────────────
# Serialization helpers (prefer cloudpickle to support closures)
# ─────────────────────────────────────────────────────────────────────────────

def _dumps(obj) -> bytes:
    try:
        import cloudpickle
        return cloudpickle.dumps(obj)
    except ImportError:
        return pickle.dumps(obj)


def _loads(data: bytes):
    try:
        import cloudpickle
        return cloudpickle.loads(data)
    except ImportError:
        return pickle.loads(data)


# ─────────────────────────────────────────────────────────────────────────────
# Worker — top-level so it is picklable under the spawn context
# ─────────────────────────────────────────────────────────────────────────────

def _worker(
    worker_id: int,
    study_name: str,
    storage_path: str,
    param_specs: dict,
    objective_pkl: bytes,
    n_trials: int,
    gpu_id: Optional[int],
) -> None:
    """
    Entry point for each spawned worker process.

    Loads the shared study and evaluates *n_trials* pre-enqueued trials.
    Workers never run the sampler — the main process has already enqueued
    every trial in the queue.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"  [Worker {worker_id}] Using GPU {gpu_id}", flush=True)

    objective = _loads(objective_pkl)

    storage = _make_journal_storage(storage_path)
    # No sampler needed — workers only pop pre-enqueued WAITING trials.
    study = optuna.load_study(study_name=study_name, storage=storage)

    def _wrapped(trial: optuna.Trial) -> float:
        params = {k: trial.suggest_categorical(k, v) for k, v in param_specs.items()}
        result = objective(params, trial_number=trial.number)
        return _unpack_result(trial, result)

    study.optimize(_wrapped, n_trials=n_trials, n_jobs=1)


# ─────────────────────────────────────────────────────────────────────────────
# Result unpacking
# ─────────────────────────────────────────────────────────────────────────────

def _unpack_result(trial: optuna.Trial, result) -> float:
    """
    Unpack the user objective's return value.

    - **float** → used directly as the Optuna objective value.
    - **dict**  → must contain ``"value"`` (float, the optimization target);
                  all other keys are stored as Optuna *user attributes*
                  and are visible in ``optuna-dashboard``.
    """
    if isinstance(result, dict):
        value = float(result.pop("value"))
        for k, v in result.items():
            try:
                trial.set_user_attr(k, v)
            except Exception:
                pass
        return value
    return float(result)


# ─────────────────────────────────────────────────────────────────────────────
# HparamSearch
# ─────────────────────────────────────────────────────────────────────────────

class HparamSearch:
    """
    Hyperparameter search with centralized sampling and parallel evaluation.

    Parameters
    ----------
    param_specs:
        ``{param_name: [value1, value2, ...]}``.  Names may use dot notation
        for nested config keys (e.g. ``"model.d_model"``).
    sampler:
        A :class:`Sampler` instance (:class:`GridSampler`,
        :class:`RandomSampler`, :class:`QMCSampler`, or a custom subclass).
    study_name:
        Optuna study name shown in the dashboard.
    storage_path:
        Path to the ``.log`` file (JournalFileBackend).
    n_trials:
        Trial budget.  ``None`` means "all combinations" — only meaningful
        for :class:`GridSampler` (other samplers require an explicit count).
    num_workers:
        Number of parallel worker processes.  ``1`` (default) runs
        everything in the current process without spawning.
    direction:
        ``"minimize"`` or ``"maximize"``.
    seed:
        Global random seed passed to the sampler.
    num_gpus:
        Number of available GPUs.  Workers are assigned GPU IDs
        round-robin (``worker_id % num_gpus``).
    """

    def __init__(
        self,
        param_specs: dict[str, list],
        sampler: Sampler,
        study_name: str,
        storage_path: str,
        n_trials: Optional[int] = None,
        num_workers: int = 1,
        direction: str = "minimize",
        seed: int = 42,
        num_gpus: int = 0,
    ) -> None:
        self.param_specs  = param_specs
        self.sampler      = sampler
        self.study_name   = study_name
        self.storage_path = storage_path
        self.n_trials     = n_trials
        self.num_workers  = max(1, num_workers)
        self.direction    = direction
        self.seed         = seed
        self.num_gpus     = num_gpus

    # ── internal helpers ─────────────────────────────────────────────────────

    def _make_storage(self) -> optuna.storages.JournalStorage:
        return _make_journal_storage(self.storage_path)

    def _make_study(self, storage) -> optuna.Study:
        return optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            direction=self.direction,
            sampler=optuna.samplers.RandomSampler(seed=self.seed),
            load_if_exists=True,
        )

    # ── public API ───────────────────────────────────────────────────────────

    def run(
        self,
        objective: Callable[[dict], Union[float, dict]],
    ) -> tuple[dict, float, optuna.Study]:
        """
        Run the hyperparameter search.

        Parameters
        ----------
        objective:
            Callable that receives a ``params`` dict and returns either:

            - a **float** — used directly as the optimization target, or
            - a **dict** with key ``"value"`` (float, required) plus any
              additional keys stored as Optuna user attributes
              (e.g. ``"val_metrics"``, ``"test_metrics"``).

        Returns
        -------
        best_params : dict
        best_value  : float
        study       : optuna.Study
        """
        storage = self._make_storage()
        study   = self._make_study(storage)

        # Count already-completed trials for resume support.
        n_done = sum(1 for t in study.trials if t.state.name == "COMPLETE")

        # ── Centralized sampling (main process only) ──────────────────────
        all_trials = self.sampler.sample(
            self.param_specs,
            n_trials=self.n_trials,
            seed=self.seed,
        )
        remaining  = all_trials[n_done:]
        n_remaining = len(remaining)

        print(
            f"[HparamSearch] {type(self.sampler).__name__} | "
            f"params={list(self.param_specs.keys())} | "
            f"planned={len(all_trials)} | done={n_done} | "
            f"remaining={n_remaining} | workers={self.num_workers}"
        )

        if n_remaining == 0:
            print("[HparamSearch] All trials already complete.")
        else:
            # Enqueue every remaining trial in the main process before
            # spawning workers — this is the core of centralized sampling.
            print(f"[HparamSearch] Enqueueing {n_remaining} trials...")
            for params in remaining:
                params = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
                study.enqueue_trial(params)

            if self.num_workers == 1:
                self._run_sequential(study, objective, n_remaining)
            else:
                self._run_parallel(storage, objective, n_remaining)

            # Reload study to collect results from worker processes.
            study = optuna.load_study(study_name=self.study_name, storage=storage)

        best = study.best_trial
        print(
            f"\n[HparamSearch] Best trial #{best.number}: "
            f"value={best.value:.6f} | params={best.params}"
        )
        return best.params, best.value, study

    # ── execution backends ───────────────────────────────────────────────────

    def _run_sequential(
        self,
        study: optuna.Study,
        objective: Callable,
        n_trials: int,
    ) -> None:
        def _wrapped(trial: optuna.Trial) -> float:
            params = {
                k: trial.suggest_categorical(k, v)
                for k, v in self.param_specs.items()
            }
            result = objective(params, trial_number=trial.number)
            return _unpack_result(trial, result)

        study.optimize(_wrapped, n_trials=n_trials, n_jobs=1)

    def _run_parallel(
        self,
        storage,
        objective: Callable,
        n_remaining: int,
    ) -> None:
        n_jobs   = self.num_workers
        per_worker = [n_remaining // n_jobs] * n_jobs
        for i in range(n_remaining % n_jobs):
            per_worker[i] += 1

        try:
            objective_pkl = _dumps(objective)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot serialize the objective function for worker processes: {exc}\n"
                "Install cloudpickle (`pip install cloudpickle`) to support closures."
            ) from exc

        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []

        for i, n_t in enumerate(per_worker):
            if n_t == 0:
                continue
            gpu_id  = (i % self.num_gpus) if self.num_gpus > 0 else None
            gpu_str = f", GPU={gpu_id}" if gpu_id is not None else ""
            p = ctx.Process(
                target=_worker,
                args=(
                    i, self.study_name, self.storage_path,
                    self.param_specs, objective_pkl, n_t, gpu_id,
                ),
                daemon=False,
            )
            p.start()
            print(f"  [HparamSearch] Worker {i} started (PID={p.pid}, trials={n_t}{gpu_str})")
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                print(
                    f"  [HparamSearch] WARNING: worker PID {p.pid} "
                    f"exited with code {p.exitcode}"
                )

    # ── reporting ────────────────────────────────────────────────────────────

    def print_results(
        self,
        study: optuna.Study,
        top_k: int = 10,
        show_attrs: tuple[str, ...] = ("val_metrics", "test_metrics"),
    ) -> None:
        """Print the top-*k* completed trials sorted by objective value."""
        reverse = self.direction == "maximize"
        completed = sorted(
            [t for t in study.trials if t.state.name == "COMPLETE"],
            key=lambda t: t.value,
            reverse=reverse,
        )
        k = min(top_k, len(completed))
        print(f"\n{'='*70}")
        print(f"Top {k} trials (direction={self.direction}):")
        print(f"{'='*70}")
        for rank, t in enumerate(completed[:k], 1):
            params_str  = " | ".join(f"{pk}={pv}" for pk, pv in t.params.items())
            extra_parts = [
                f"{attr}={t.user_attrs[attr]}"
                for attr in show_attrs
                if attr in t.user_attrs
            ]
            extra = ("  ← " + " | ".join(extra_parts)) if extra_parts else ""
            print(f"  {rank:3d}. [{t.value:.6f}] {params_str}{extra}")

    def save_csv(self, study: optuna.Study, output_path: str) -> None:
        """Save all completed trials to a CSV file.

        Columns mirror the Optuna dashboard table:
            trial | State | value | <Param cols> | val_metrics | test_metrics | best_epoch
        val_metrics and test_metrics are kept as full lists.
        Sorted by value (best first, respecting direction).
        """
        import csv

        reverse = self.direction == "maximize"
        trials  = sorted(
            study.trials,
            key=lambda t: (t.value is None, t.value),
            reverse=reverse,
        )
        if not trials:
            print("[HparamSearch] No trials to export.")
            return

        param_cols = list(self.param_specs.keys())
        fieldnames = (
            ["trial", "State", "value"]
            + [f"Param {p}" for p in param_cols]
            + ["UserAttribute val_metrics", "UserAttribute test_metrics", "best_epoch"]
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in trials:
                row: dict = {
                    "trial":   t.number,
                    "State":   t.state.name.capitalize(),
                    "value":   t.value if t.value is not None else "",
                }
                for p in param_cols:
                    row[f"Param {p}"] = t.params.get(p, "")
                row["UserAttribute val_metrics"]  = t.user_attrs.get("val_metrics",  "")
                row["UserAttribute test_metrics"] = t.user_attrs.get("test_metrics", "")
                row["best_epoch"] = t.user_attrs.get("best_epoch", "")
                writer.writerow(row)

        print(f"[HparamSearch] Results saved to {output_path}  ({len(trials)} trials)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI: convert an existing .log file to CSV
#   python -m utils.hparam_search path/to/study.log [output.csv]
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m utils.hparam_search <study.log> [output.csv]")
        sys.exit(1)

    log_path = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else log_path.replace(".log", ".csv")

    _storage  = _make_journal_storage(log_path)
    _studies  = optuna.get_all_study_names(storage=_storage)
    if not _studies:
        print(f"No studies found in {log_path}")
        sys.exit(1)
    if len(_studies) > 1:
        print(f"Multiple studies found: {_studies}. Using first.")
    _study = optuna.load_study(study_name=_studies[0], storage=_storage)

    # Build a minimal HparamSearch just for the save_csv helper.
    _param_specs = {k: list({t.params[k] for t in _study.trials if k in t.params})
                    for k in (_study.trials[0].params if _study.trials else {})}
    _hs = HparamSearch(param_specs=_param_specs, sampler=GridSampler(),
                       study_name=_studies[0], storage_path=log_path)
    _hs.save_csv(_study, csv_path)
