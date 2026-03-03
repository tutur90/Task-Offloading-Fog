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
from utils.grid_search import apply_params_to_config, parse_grid_search_params

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
        else:
            env = run_epoch(config, policy, train_data, train=True)
            update_metrics(logger, env, config)
            env.close()
            logger.update_metric('AvgLoss', policy.avg_loss)
            logger.update_metric('AvgGradNorm', policy.avg_grad_norm)

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

        if not is_ga:
            for param_group in policy.optimizer.param_groups:
                param_group['lr'] *= config["training"]["lr_decay"]
            if config["algo"] == "TaskFormer":
                for param_group in policy.optimizer.param_groups:
                    param_group['lr'] *= config["training"]["lr_decay"]

    return best_val_metrics


def main(config):

    set_seed(config.get("seed", 42))

    logger = Logger(config)

    env = create_env(config)

    if "training" in config.keys():

        checkpoint = Checkpoint(logger.log_dir)

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
    sorted_indices = np.argsort(metrics[:, 3])[::-1][:k]
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
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    config_path = args.config
    

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    config["device"] = args.device

    seeds = args.seeds
    if args.n_seeds is not None:
        seeds = list(range(args.n_seeds))

    if seeds is not None:
        run_multi_seed(config, config_path, seeds, args)
    elif args.search is not None:
        run_search(config, config_path, args)
    else:
        val_metrics, test_metrics, best_epoch = main(config)
        print(f"Validation: {val_metrics} | Test: {test_metrics} | Best epoch: {best_epoch}")
