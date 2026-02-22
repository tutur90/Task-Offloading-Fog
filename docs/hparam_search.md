# Hyperparameter Search Framework

`utils/hparam_search.py`

---

## How it works

### Centralized sampling

The key design principle is that **all parameter combinations are generated once in the main process** before any worker is spawned.
Each combination is immediately **enqueued** as an Optuna `WAITING` trial in the shared storage.
Workers only **pop and evaluate** — they never run the sampler themselves.

```
main process
  │
  ├─ Sampler.sample() → [params_0, params_1, ..., params_N]
  ├─ study.enqueue_trial(params_0)
  ├─ study.enqueue_trial(params_1)
  │   ...
  │
  ├─ spawn Worker 0 ──► pops & evaluates trials 0..k
  ├─ spawn Worker 1 ──► pops & evaluates trials k..2k
  │   ...
  └─ join all workers → read results from study
```

This matters because:

- **Grid** — no duplicate evaluations across workers.
- **QMC** — the Sobol low-discrepancy property is only preserved when the full sequence is generated in one shot.
- **Random** — reproducibility: a fixed seed always produces the same trial queue regardless of worker count.

### Storage & dashboard compatibility

Results are written to an Optuna `JournalFileBackend` log file (append-only, safe for concurrent writes).
This is the native format read by **optuna-dashboard**:

```bash
pip install optuna-dashboard
optuna-dashboard --storage logs/Pakistan/Tuple50K/MLP/hparam_d_model_lr.log
```

Every trial's objective value, parameters, and any extra metadata (val/test metrics, best epoch) are visible in the dashboard.

### Multiprocessing

Workers are spawned with Python's `spawn` start method — the safest option across platforms and CUDA.
GPU IDs are assigned round-robin: worker `i` gets GPU `i % num_gpus`.

---

## Samplers

| Name | Class | Description |
|---|---|---|
| `grid` | `GridSampler` | Cartesian product of all values. Optionally shuffled before truncation. |
| `random` | `RandomSampler` | Uniform random draw with replacement for each trial. |
| `qmc` | `QMCSampler` | Scrambled Sobol sequence. Better space-filling than random. Requires `scipy`. |

### Adding a custom sampler

Subclass `Sampler`, implement `sample()`, and register in `SAMPLERS`:

```python
# utils/hparam_search.py

class MyCustomSampler(Sampler):
    def sample(self, param_specs, n_trials, seed=42):
        # param_specs: {"model.d_model": [64, 128, 256], ...}
        # Return a list of dicts, one per trial.
        results = []
        ...
        return results

SAMPLERS["mine"] = MyCustomSampler
```

It is then available on the CLI via `--sampler mine`.

---

## CLI usage

```bash
# Single run (no search)
python main.py configs/Pakistan/Tuple50K/DQL/MLP.yaml

# Grid search — all combinations
python main.py configs/Pakistan/Tuple50K/DQL/MLP.yaml \
  --search "model.d_model=64,128,256" "model.n_layers=2,3,4" \
  --sampler grid

# Random search — 32 trials, 4 parallel workers
python main.py configs/Pakistan/Tuple50K/DQL/MLP.yaml \
  --search "model.d_model=64,128,256" "training.lr=1e-3,5e-4,1e-4" \
  --sampler random --n_samples 32 --num_workers 4

# QMC (Sobol) search — 64 trials, 8 workers
python main.py configs/Pakistan/Tuple50K/DQL/MLP.yaml \
  --search "model.d_model=64,128,256" "training.lr=1e-3,5e-4,1e-4" \
  --sampler qmc --n_samples 64 --num_workers 8
```

Parameter names use **dot notation** to address nested config keys:
`"training.lr=1e-3,5e-4"` sets `config["training"]["lr"]`.

`--n_samples` is ignored when `--sampler grid` is used (all combinations are always enumerated).

---

## Python API

```python
from utils.hparam_search import HparamSearch, GridSampler, RandomSampler, QMCSampler

search = HparamSearch(
    param_specs={
        "model.d_model":  [64, 128, 256],
        "training.lr":    [1e-3, 5e-4, 1e-4],
    },
    sampler=QMCSampler(),
    study_name="mlp_qmc",
    storage_path="logs/Pakistan/Tuple50K/MLP/hparam_mlp_qmc.log",
    n_trials=32,        # ignored by GridSampler
    num_workers=4,
    seed=42,
    num_gpus=2,         # workers get GPU 0, 1, 0, 1, ...
)
```

The objective receives a plain `dict` of parameters and returns either:

- a **`float`** — used directly as the optimization target, or
- a **`dict`** with key `"value"` (float, required) and any extra keys stored as Optuna user attributes:

```python
def objective(params):
    # params == {"model.d_model": 128, "training.lr": 0.001}
    ...
    return {
        "value":        val_score,          # minimized by default
        "val_metrics":  list(val_metrics),  # stored in dashboard
        "test_metrics": list(test_metrics),
        "best_epoch":   best_epoch,
    }

best_params, best_value, study = search.run(objective)
search.print_results(study, top_k=10)
```

### Resume support

If a search is interrupted, re-running the exact same command resumes automatically.
Already-completed trials (tracked by the log file) are skipped; only remaining trials are enqueued and evaluated.

---

## Output

| File | Description |
|---|---|
| `logs/{dataset}/{flag}/{policy}/hparam_{params}.log` | Optuna JournalFile — open with optuna-dashboard |
| `logs/{dataset}/{flag}/{policy}/hparam_{params}_contour.html` | Interactive contour plot (only when exactly 2 params are tuned) |
