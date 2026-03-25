"""
This script demonstrates how to run the DQRLPolicy.

Oh, wait a moment. It seems that extra effort is required to make this method work. The current version
is for reference only, and contributions are welcome.
"""

import os
import sys
import time

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
from tqdm import tqdm
import yaml

import numpy as np

from core.vis import *
from core.vis.vis_stats import VisStats

from eval.metrics.metrics import SuccessRate, AvgLatency
from policies import policies
from utils.dql import run_epoch
from utils.ppo import run_epoch_ppo
from utils.GA import run_generation

from utils.utils import create_env, error_handler, set_seed, update_metrics
from utils.utils import Logger, Checkpoint
from utils.grid_search import (
    apply_params_to_config, parse_grid_search_params,
    generate_probability_grid, load_grid_search_progress,
    save_grid_search_progress, lambda_to_key,
)

GA_ALGOS  = ["NPGA", "NSGA2"]
PPO_ALGOS = ["PPO"]


def train(config, policy, train_data, valid_data, logger, checkpoint):
    """ Train the policy using the provided training data and validate it using the validation data. """
    is_ga = config["algo"] in GA_ALGOS

    early_stop_patience = config["training"].get("early_stop_patience", None)
    epochs_without_improvement = 0
    best_val_metrics = None

    # Track fitness across generations for GA (avoids re-evaluating parents)
    cached_fitness = None

    if not is_ga and hasattr(policy, 'set_training_steps'):
        total_steps = config["training"]["num_epochs"] * len(train_data)
        policy.set_training_steps(total_steps)

    for epoch in range(config["training"]["num_epochs"]):

        logger.update_epoch(epoch)

        # Training phase.

        logger.update_mode('Training')

        epoch_start = time.time()

        if hasattr(policy, 'stats'):
            policy.stats = {k: (0.0 if isinstance(v, float) else 0) for k, v in policy.stats.items()}

        if is_ga:
            # Pass cached fitness to avoid re-evaluating parents (except first epoch)
            result = run_generation(config, policy, train_data, train=True,
                                    parent_fitness=cached_fitness)
            update_metrics(logger, None, config, metrics=tuple(result.best_metrics))
            # Cache fitness for next generation (these are the selected individuals)
            cached_fitness = result.fitness
            result.close()
        elif config["algo"] in PPO_ALGOS:
            env = run_epoch_ppo(config, policy, train_data, train=True)
            update_metrics(logger, env, config)
            env.close()
            logger.update_metric('AvgLoss', policy.avg_loss)
            logger.update_metric('AvgGradNorm', policy.avg_grad_norm)
            logger.update_metric('AvgReward', policy.avg_reward)
        else:
            env = run_epoch(config, policy, train_data, train=True)
            update_metrics(logger, env, config)
            env.close()
            logger.update_metric('AvgLoss', policy.avg_loss)
            logger.update_metric('AvgGradNorm', policy.avg_grad_norm)
            logger.update_metric('AvgReward', policy.avg_reward)

        if hasattr(policy, 'lstm_stats_summary'):
            s = policy.lstm_stats_summary()
            print(f"  [OPO epoch {epoch+1}] "
                  f"exploit={s['exploit_%']:.1f}% "
                  f"lstm_guided={s['lstm_guided_%']:.1f}% "
                  f"random={s['random_%']:.1f}% | "
                  f"load_loss={s['avg_load_lstm_loss']:.4f} "
                  f"task_loss={s['avg_task_lstm_loss']:.4f}")

        epoch_time = time.time() - epoch_start
        logger.update_metric('TimePerTask', epoch_time / len(train_data))

        # Validation phase.

        logger.update_mode('Validation')

        val_start = time.time()

        if is_ga:
            result = run_generation(config, policy, valid_data, train=False)
            score = update_metrics(logger, None, config, metrics=tuple(result.best_metrics))
            result.close()
        elif config["algo"] in PPO_ALGOS:
            env = run_epoch_ppo(config, policy, valid_data, train=False)
            score = update_metrics(logger, env, config)
            env.close()
        else:
            env = run_epoch(config, policy, valid_data, train=False)
            score = update_metrics(logger, env, config)
            env.close()

        val_time = time.time() - val_start
        logger.update_metric('TimePerTask', val_time / len(valid_data))

        if logger.is_best(score[3], epoch):
            checkpoint.save(policy, epoch)
            best_val_metrics = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stop_patience is not None and epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {early_stop_patience} epochs)")
            break


    return best_val_metrics


def main(config):

    set_seed(config.get("seed", 42))

    logger = Logger(config)

    env = create_env(config)

    if "training" in config.keys():

        keep_checkpoints = config["training"].get("keep_checkpoints", 1)
        checkpoint = Checkpoint(logger.log_dir, keep_last_n=keep_checkpoints)

        valid_size = config["training"].get("valid_size", 0.2)

        # Load train and test datasets.
        train_data = pd.read_csv(f"eval/benchmarks/{config['env']['dataset']}/data/{config['env']['flag']}/trainset.csv")
        train_data, valid_data = train_data.iloc[:int(len(train_data)*(1-valid_size))], train_data.iloc[int(len(train_data)*(1-valid_size)):]
        valid_data["GenerationTime"] = valid_data["GenerationTime"] - valid_data["GenerationTime"].min()  # Normalize generation time

        if "lambda" in config["training"]:

            config["training"]["lambda"] = (config["training"]["lambda"][0]/sum(config["training"]["lambda"]),
                                        config["training"]["lambda"][1]/sum(config["training"]["lambda"]),
                                        config["training"]["lambda"][2]/sum(config["training"]["lambda"]))
        policy = policies[config["policy"]](env, config, dataset=train_data)
        policy._train_logger = logger

    else:
        policy = policies[config["policy"]](env, config,)

    test_data = pd.read_csv(f"eval/benchmarks/{config['env']['dataset']}/data/{config['env']['flag']}/testset.csv")

    # Initialize the policy.

    val_metrics = None

    if "training" in config.keys():
        val_metrics = train(config, policy, train_data, valid_data, logger, checkpoint)
        checkpoint.load(policy, logger.best_epoch)

        if hasattr(policy, 'lstm_stats_summary'):
            import pprint
            print("\n── OPO LSTM stats (training) ──")
            pprint.pprint(policy.lstm_stats_summary())

    # Testing phase.

    logger.update_mode('Testing')

    if config["algo"] in GA_ALGOS:
        result = run_generation(config, policy, test_data, train=False)
        test_metrics = update_metrics(logger, None, config, metrics=tuple(result.best_metrics))
        env = result  # for close() compatibility below
    elif config["algo"] in PPO_ALGOS:
        env = run_epoch_ppo(config, policy, test_data, train=False)
        test_metrics = update_metrics(logger, env, config)
    else:
        env = run_epoch(config, policy, test_data, train=False)
        test_metrics = update_metrics(logger, env, config)

    if hasattr(policy, 'lstm_stats_summary'):
        import pprint
        print("\n── OPO LSTM stats (training + testing) ──")
        pprint.pprint(policy.lstm_stats_summary())

    logger.plot()
    logger.save_csv()

    best_epoch = logger.best_epoch

    logger.close()
    env.close()

    if config["algo"] not in GA_ALGOS:
        vis_stats = VisStats(save_path=logger.log_dir)
        vis_stats.vis(env)

    return val_metrics, test_metrics, best_epoch


def _lambda_search_worker(args):
    """Multiprocessing worker for lambda grid search."""
    i, params, config_path, num_gpus, device = args
    if num_gpus > 0:
        gpu_id = i % num_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        worker_device = "cuda"
        print(f"[Worker {i}] Assigned to GPU {gpu_id}")
    else:
        worker_device = device
    with open(config_path, "r") as f:
        worker_config = yaml.safe_load(f)
    worker_config["worker_id"] = i
    worker_config["device"] = worker_device
    worker_config["training"]["lambda"] = params.tolist()
    key = lambda_to_key(params)
    print(f"[Worker {i}] lambda={worker_config['training']['lambda']}")
    val_result, test_result, best_epoch = main(worker_config)
    return i, key, params, val_result, test_result, best_epoch


def run_lambda_search(config, config_path, args, n_steps):
    """Grid search over the lambda simplex [l0+l1+l2=1] using n_steps."""
    import multiprocessing
    from utils.plots import plot_ternary

    dataset     = config["env"]["dataset"]
    flag        = config["env"]["flag"]
    policy_name = config["policy"]
    results_dir = f"logs/{dataset}/{flag}/{policy_name}"
    results_file = f"{results_dir}/lambda_grid_search_progress.json"
    
    max_val = 40 if "Synthetic" in config["env"]["dataset"] else 0.8

    samples = generate_probability_grid(n_steps)
    progress = load_grid_search_progress(results_file)
    completed_count = len(progress["completed"])
    if completed_count > 0:
        print(f"Resuming lambda grid search: {completed_count}/{len(samples)} already completed")

    val_metrics  = np.zeros((len(samples), 4))
    test_metrics = np.zeros((len(samples), 4))

    num_gpus = get_num_gpus()
    if num_gpus > 0:
        print(f"Detected {num_gpus} GPU(s) — workers distributed round-robin")

    device = config["device"]
    work_items = []
    for i, params in enumerate(samples):
        key = lambda_to_key(params)
        if key in progress["completed"]:
            val_metrics[i]  = progress["val_metrics"][key]
            test_metrics[i] = progress["test_metrics"][key]
        else:
            work_items.append((i, params, config_path, num_gpus, device))

    max_workers = args.num_workers if args.num_workers else multiprocessing.cpu_count()
    num_workers = min(max_workers, len(work_items))

    def _save(i, key, params, val_result, test_result, best_epoch):
        nonlocal completed_count
        val_metrics[i]  = val_result
        test_metrics[i] = test_result
        print(f"Validation: {val_result}  Test: {test_result}")
        progress["completed"][key]     = True
        progress["val_metrics"][key]   = val_metrics[i].tolist()
        progress["test_metrics"][key]  = test_metrics[i].tolist()
        progress.setdefault("best_epoch", {})[key] = best_epoch
        save_grid_search_progress(results_file, progress)
        completed_count += 1
        print(f"Progress saved ({completed_count}/{len(samples)} completed, best epoch: {best_epoch})")

    if num_workers > 1:
        print(f"Parallel lambda search: {num_workers} workers, {len(work_items)} remaining")
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(num_workers) as pool:
            for result in pool.imap_unordered(_lambda_search_worker, work_items):
                _save(*result)
    else:
        print(f"Sequential lambda search: {len(work_items)} remaining")
        for item in work_items:
            _save(*_lambda_search_worker(item))

    k = min(20, len(samples))
    print_top_k_results(samples, val_metrics,  k=k, label="Validation Results")
    # print_top_k_results(samples, test_metrics, k=k, label="Test Results")
    plot_ternary(samples, values=test_metrics[:, 3], title="Test Score Lambda Grid",
                 labels=["λ0", "λ1", "λ2"],
                 output_path=f"{results_dir}/lambda_grid_search_test.png", max_value=max_val, dpi=400)
    plot_ternary(samples, values=val_metrics[:, 3],  title="Validation Score Lambda Grid",
                 labels=["λ0", "λ1", "λ2"],
                 output_path=f"{results_dir}/lambda_grid_search_val.png",  max_value=max_val, dpi=400)


def run_search(config, config_path, args):
    """Run hyperparameter search using the HparamSearch framework."""
    from utils.hparam_search import HparamSearch, SAMPLERS

    param_specs = parse_grid_search_params(args.search)

    sampler_cls = SAMPLERS.get(args.sampler)
    if sampler_cls is None:
        raise ValueError(
            f"Unknown sampler '{args.sampler}'. Available: {list(SAMPLERS.keys())}"
        )

    dataset     = config["env"]["dataset"]
    flag        = config["env"]["flag"]
    policy_name = config["policy"]
    search_name = "_".join(k.replace(".", "_") for k in param_specs.keys())
    results_dir = f"logs/{dataset}/{flag}/{policy_name}"
    storage_path = os.path.join(results_dir, f"hparam_{search_name}.log")

    num_gpus = get_num_gpus()
    if num_gpus > 0:
        print(f"Detected {num_gpus} GPU(s) — distributing workers round-robin.")

    # GridSampler enumerates all combinations by default; n_trials caps that.
    n_trials = None if args.sampler == "grid" else args.n_samples

    search = HparamSearch(
        param_specs=param_specs,
        sampler=sampler_cls(),
        study_name=search_name,
        storage_path=storage_path,
        n_trials=n_trials,
        num_workers=args.num_workers or 1,
        seed=config.get("seed", 42),
        num_gpus=num_gpus,
    )

    def objective(params, trial_number=None):
        worker_config = yaml.safe_load(open(config_path, "r"))
        apply_params_to_config(worker_config, params)
        worker_config["device"] = config["device"]  # propagate device from CLI args (round-robin GPU)
        worker_config["tuned_params"] = params  # tags log dir, e.g. 0216_143022_t3_dm128_nl3
        if trial_number is not None:
            worker_config["worker_id"] = trial_number
        val_metrics, test_metrics, best_epoch = main(worker_config)
        # Use val_metrics when available (training run), fall back to test.
        metrics = val_metrics if val_metrics is not None else test_metrics
        return {
            "value":        float(metrics[3]),
            "val_metrics":  [float(v) for v in val_metrics]  if val_metrics  is not None else [],
            "test_metrics": [float(v) for v in test_metrics] if test_metrics is not None else [],
            "best_epoch":   int(best_epoch) if best_epoch is not None else 0,
        }

    best_params, best_value, study = search.run(objective)
    search.print_results(study)
    search.save_csv(study, os.path.join(results_dir, f"hparam_{search_name}.csv"))

    # Generate an interactive contour plot when exactly 2 params are tuned.
    if len(param_specs) == 2:
        try:
            from optuna.visualization import plot_contour
            fig = plot_contour(study, params=list(param_specs.keys()))
            plot_path = os.path.join(results_dir, f"hparam_{search_name}_contour.html")
            fig.write_html(plot_path)
            print(f"Contour plot saved to {plot_path}")
        except Exception as e:
            print(f"Could not generate contour plot: {e}")


def run_multi_seed(config, config_path, seeds, args):
    """Train over multiple seeds and report aggregate statistics.

    Reuses HparamSearch (GridSampler on the seed axis) for parallelism,
    GPU round-robin, and Optuna-backed resumability.
    """
    from utils.hparam_search import HparamSearch, GridSampler

    dataset     = config["env"]["dataset"]
    flag        = config["env"]["flag"]
    policy_name = config["policy"]

    results_dir = os.path.join("logs", dataset, flag, policy_name)

    # Stable tag so different seed sets get separate studies/logs.
    if len(seeds) <= 10:
        seeds_tag = "_".join(str(s) for s in seeds)
    else:
        import hashlib
        h = hashlib.md5(str(seeds).encode()).hexdigest()[:8]
        seeds_tag = f"n{len(seeds)}_{h}"

    storage_path = os.path.join(results_dir, f"multi_seed_{seeds_tag}.log")
    study_name   = f"multi_seed_{seeds_tag}"

    num_gpus = get_num_gpus()
    if num_gpus > 0:
        print(f"Detected {num_gpus} GPU(s) — distributing workers round-robin.")

    search = HparamSearch(
        param_specs={"seed": seeds},
        sampler=GridSampler(shuffle=False),   # preserve seed order
        study_name=study_name,
        storage_path=storage_path,
        n_trials=None,                        # grid = all seeds
        num_workers=args.num_workers or 1,
        seed=0,
        num_gpus=num_gpus,
    )

    def objective(params, trial_number=None):
        seed_config = yaml.safe_load(open(config_path))
        apply_params_to_config(seed_config, params)   # sets seed_config["seed"]
        seed_config["device"] = config["device"]
        if trial_number is not None:
            seed_config["worker_id"] = trial_number
        val_metrics, test_metrics, best_epoch = main(seed_config)
        metrics = val_metrics if val_metrics is not None else test_metrics
        return {
            "value":        float(metrics[3]),
            "val_metrics":  [float(v) for v in val_metrics]  if val_metrics  is not None else [],
            "test_metrics": [float(v) for v in test_metrics] if test_metrics is not None else [],
            "best_epoch":   int(best_epoch) if best_epoch is not None else 0,
        }

    _, _, study = search.run(objective)

    # Aggregate stats across all completed trials.
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    all_val  = [t.user_attrs["val_metrics"]  for t in completed if t.user_attrs.get("val_metrics")]
    all_test = [t.user_attrs["test_metrics"] for t in completed if t.user_attrs.get("test_metrics")]

    summary: dict = {"seeds": seeds, "n_completed": len(completed)}

    print("\n" + "=" * 60)
    print(f"Multi-seed summary  —  {len(seeds)} seeds: {seeds}")
    for label, rows, key in [("Val", all_val, "val"), ("Test", all_test, "test")]:
        if not rows:
            continue
        arr   = np.array(rows)
        means = arr.mean(axis=0).tolist()
        stds  = arr.std(axis=0).tolist()
        summary[key] = {"mean": means, "std": stds, "n": len(rows)}
        print(f"\n{label} metrics (mean ± std over {len(rows)} runs):")
        for idx, (m, s) in enumerate(zip(means, stds)):
            print(f"  [{idx}] {m:.4f} ± {s:.4f}")
    print("=" * 60)

    import json
    summary_path = storage_path.replace(".log", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")
    print(f"Study log:     {storage_path}")


def print_top_k_results(samples, metrics, k=10, label="Results"):
    """Print the top-k lambda configurations sorted by score (descending)."""
    import numpy as np
    metrics = np.array(metrics)
    sorted_indices = np.argsort(metrics[:, 3])[:k]
    print(f"\n--- Top {k} {label} ---")
    print(f"{'Rank':<6} {'λ0':>6} {'λ1':>6} {'λ2':>6} {'TTR':>8} {'Latency':>10} {'Power':>10} {'Score':>8}")
    for rank, idx in enumerate(sorted_indices, 1):
        l = samples[idx]
        ttr, lat, pwr, score = metrics[idx]
        print(f"{rank:<6} {l[0]:>6.3f} {l[1]:>6.3f} {l[2]:>6.3f} {ttr:>8.4f} {lat:>10.4f} {pwr:>10.4f} {score:>8.4f}")
    print()


def get_num_gpus():
    """Detect the number of available CUDA GPUs without initializing CUDA."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return len(result.stdout.strip().split("\n"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def plot_lambda_search(config):
    """Re-generate ternary plots and top-k table from existing lambda grid search results."""
    from utils.plots import plot_ternary

    dataset     = config["env"]["dataset"]
    flag        = config["env"]["flag"]
    policy_name = config["policy"]
    results_dir = f"logs/{dataset}/{flag}/{policy_name}"
    results_file = f"{results_dir}/lambda_grid_search_progress.json"

    progress = load_grid_search_progress(results_file)
    if not progress["completed"]:
        print(f"No completed results found in {results_file}")
        return

    # Reconstruct arrays from stored keys (format: "l0,l1,l2" with 6 decimals).
    keys    = list(progress["completed"].keys())
    samples = np.array([[float(v) for v in k.split(",")] for k in keys])
    val_metrics  = np.array([progress["val_metrics"][k]  for k in keys])
    test_metrics = np.array([progress["test_metrics"][k] for k in keys])

    print(f"Loaded {len(keys)} completed lambda configurations from {results_file}")

    k = min(20, len(keys))
    print_top_k_results(samples, val_metrics,  k=k, label="Validation Results")
    # print_top_k_results(samples, test_metrics, k=k, label="Test Results")

    plot_ternary(samples, values=test_metrics[:, 3], title="Test Score Lambda Grid",
                 labels=["λ0", "λ1", "λ2"],
                 output_path=f"{results_dir}/lambda_grid_search_test.png", max_value=0.8)
    plot_ternary(samples, values=val_metrics[:, 3],  title="Validation Score Lambda Grid",
                 labels=["λ0", "λ1", "λ2"],
                 output_path=f"{results_dir}/lambda_grid_search_val.png",  max_value=0.8)
    print(f"Plots saved to {results_dir}/")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run Task Offloading Policy")
    parser.add_argument("config", type=str, help="Path to the YAML config file.")
    parser.add_argument(
        "--search", type=str, nargs="*", default=None,
        metavar="PARAM=v1,v2,...",
        help=(
            'Hyperparameter search. Specify each parameter as "section.key=v1,v2,v3". '
            'Example: --search "model.d_model=64,128,256" "training.lr=1e-3,5e-4"'
        ),
    )
    parser.add_argument(
        "--sampler", type=str, default="random",
        choices=["grid", "random", "qmc"],
        help=(
            "Sampler strategy (default: random). "
            "grid=all combinations, random=uniform, qmc=Sobol low-discrepancy."
        ),
    )
    parser.add_argument(
        "--n_samples", type=int, default=64,
        help="Number of trials (default: 64). Ignored when --sampler grid is used.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Number of parallel worker processes (default: 1, sequential).",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help=("Device to run on (e.g., 'cuda:0' or 'cpu'). By default, uses GPU if available, otherwise CPU. "
              "Note: For hyperparameter search with multiple workers, set this to 'cuda' to allow automatic GPU assignment."),
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        metavar="SEED",
        help=(
            "Run training over specific seeds and report aggregate stats (mean ± std). "
            "Example: --seeds 42 123 456. Resumable: already-completed seeds are skipped."
        ),
    )
    seed_group.add_argument(
        "--n_seeds", type=int, default=None,
        metavar="N",
        help="Shortcut for --seeds 0 1 ... N-1.",
    )
    parser.add_argument(
        "--keep_checkpoints", type=int, default=1,
        metavar="N",
        help="Keep only the last N checkpoints during training (default: 1).",
    )
    parser.add_argument(
        "--no_pareto_only", action="store_true", default=False,
        help="Save all individuals in GA checkpoints instead of Pareto-front only (default: Pareto-front only).",
    )
    parser.add_argument(
        "--plot", type=str, default=None,
        metavar="TYPE",
        help=(
            'Generate plots from existing results without re-running. '
            'Currently supported: "lambda" — ternary plots from lambda_grid_search_progress.json.'
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    config_path = args.config
    

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    config["device"] = args.device
    if "training" in config:
        config["training"]["keep_checkpoints"] = args.keep_checkpoints
        config["training"]["save_pareto_only"] = not args.no_pareto_only

    seeds = args.seeds
    if args.n_seeds is not None:
        seeds = list(range(args.n_seeds))

    if args.plot is not None:
        if args.plot == "lambda":
            plot_lambda_search(config)
        else:
            print(f"Unknown plot type '{args.plot}'. Supported: lambda")
    elif seeds is not None:
        run_multi_seed(config, config_path, seeds, args)
    elif args.search is not None:
        # Special case: --search "lambda=N" triggers ternary lambda grid search.
        if len(args.search) == 1 and args.search[0].startswith("lambda="):
            n_steps = int(args.search[0].split("=", 1)[1])
            run_lambda_search(config, config_path, args, n_steps=n_steps)
        else:
            run_search(config, config_path, args)
    else:
        val_metrics, test_metrics, best_epoch = main(config)
        print(f"Validation: {val_metrics} | Test: {test_metrics} | Best epoch: {best_epoch}")
