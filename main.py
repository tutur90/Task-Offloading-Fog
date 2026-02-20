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
from utils.GA import run_generation


from utils.utils import create_env, error_handler, set_seed, update_metrics
from utils.utils import Logger, Checkpoint
from utils.grid_search import (
    apply_params_to_config, parse_grid_search_params
)

GA_ALGOS = ["NPGA", "NSGA2"]


def print_top_k_results(grid, metrics, k=10, label="Results"):
    """Print top k parameter combinations and metrics sorted by metrics[:, 3] (ascending)."""
    top_k_indices = np.argsort(metrics[:, 3])[:k]
    print(f"\n{'='*80}")
    print(f"Top {k} {label} (by score):")
    print(f"{'='*80}")
    for rank, idx in enumerate(top_k_indices, 1):
        params = grid[idx]
        if isinstance(params, dict):
            params_str = " | ".join(f"{k}={v}" for k, v in params.items())
        else:
            params_str = f"Lambda: {params}"
        print(f"{rank}. {params_str} | Metrics: {metrics[idx]}")


def _optuna_worker(worker_id, n_trials, config_path, param_specs, sampler_name, seed, results_dir, search_name, num_gpus=0):
    """Worker process for parallel Optuna optimization. Each process runs its own study.optimize()."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if num_gpus > 0:
        gpu_id = worker_id % num_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"  [Worker {worker_id}] Assigned to GPU {gpu_id}")

    samplers = {
        "tpe": lambda: optuna.samplers.TPESampler(seed=seed + worker_id),
        "random": lambda: optuna.samplers.RandomSampler(seed=seed + worker_id),
        "qmc": lambda: optuna.samplers.QMCSampler(scramble=True, seed=seed + worker_id),
        "grid": lambda: optuna.samplers.GridSampler(param_specs, seed=seed + worker_id),
        "cmaes": lambda: optuna.samplers.CmaEsSampler(seed=seed + worker_id),
    }
    sampler = samplers[sampler_name]()

    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(
            os.path.join(os.path.abspath(results_dir), f"optuna_{search_name}.log")
        )
    )

    study = optuna.load_study(
        study_name=search_name,
        storage=storage,
        sampler=sampler,
    )

    def objective(trial):
        params = {k: trial.suggest_categorical(k, v) for k, v in param_specs.items()}

        worker_config = yaml.safe_load(open(config_path, 'r'))
        worker_config["worker_id"] = trial.number
        worker_config["tuned_params"] = params
        apply_params_to_config(worker_config, params)

        val_result, test_result, best_epoch = main(worker_config)

        trial.set_user_attr("val_metrics", [float(v) for v in val_result])
        trial.set_user_attr("test_metrics", [float(v) for v in test_result])
        trial.set_user_attr("best_epoch", best_epoch)

        return float(val_result[3])

    study.optimize(objective, n_trials=n_trials, n_jobs=1)


def run_optuna_search(config, config_path, args):
    """Run Optuna hyperparameter search with true multiprocessing parallelism."""
    import optuna
    import multiprocessing as mp

    param_specs = parse_grid_search_params(args.optuna)
    search_name = "_".join(k.replace(".", "_") for k in param_specs.keys())
    seed = config.get("seed", 42)

    # Storage (JournalFile is safe for concurrent multi-process access)
    results_dir = f"logs/{config['env']['dataset']}/{config['env']['flag']}/{config['policy']}"
    os.makedirs(results_dir, exist_ok=True)
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(
            os.path.join(os.path.abspath(results_dir), f"optuna_{search_name}.log")
        )
    )

    # Sampler (for study creation only; workers create their own)
    samplers = {
        "tpe": lambda: optuna.samplers.TPESampler(seed=seed),
        "random": lambda: optuna.samplers.RandomSampler(seed=seed),
        "qmc": lambda: optuna.samplers.QMCSampler(scramble=True, seed=seed),
        "grid": lambda: optuna.samplers.GridSampler(param_specs, seed=seed),
        "cmaes": lambda: optuna.samplers.CmaEsSampler(seed=seed),
    }
    sampler = samplers[args.sampler]()

    # Create or load study
    study = optuna.create_study(
        study_name=search_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )

    # How many trials remain
    n_completed = len([t for t in study.trials if t.state.name == "COMPLETE"])
    n_remaining = max(0, args.n_samples - n_completed)
    n_jobs = args.num_workers or 1
    num_gpus = get_num_gpus()

    if num_gpus > 0:
        print(f"Detected {num_gpus} GPU(s) — workers will be distributed round-robin across them")

    print(f"Optuna search: {list(param_specs.keys())} | sampler={args.sampler} | "
          f"trials={n_completed}/{args.n_samples} done | n_jobs={n_jobs}")

    if n_remaining > 0:
        if n_jobs == 1:
            # Single process — run directly (no overhead)
            def objective(trial):
                params = {k: trial.suggest_categorical(k, v) for k, v in param_specs.items()}
                worker_config = yaml.safe_load(open(config_path, 'r'))
                worker_config["worker_id"] = trial.number
                worker_config["tuned_params"] = params
                apply_params_to_config(worker_config, params)
                val_result, test_result, best_epoch = main(worker_config)
                trial.set_user_attr("val_metrics", [float(v) for v in val_result])
                trial.set_user_attr("test_metrics", [float(v) for v in test_result])
                trial.set_user_attr("best_epoch", best_epoch)
                return float(val_result[3])

            study.optimize(objective, n_trials=n_remaining, n_jobs=1)
        else:
            # Multi-process: each worker gets a share of trials
            trials_per_worker = [n_remaining // n_jobs] * n_jobs
            for i in range(n_remaining % n_jobs):
                trials_per_worker[i] += 1

            processes = []
            ctx = mp.get_context("spawn")
            for i, n_t in enumerate(trials_per_worker):
                if n_t == 0:
                    continue
                p = ctx.Process(
                    target=_optuna_worker,
                    args=(i, n_t, config_path, param_specs, args.sampler, seed, results_dir, search_name, num_gpus),
                )
                p.start()
                processes.append(p)
                print(f"  Worker {i} started (PID {p.pid}, {n_t} trials)")

            for p in processes:
                p.join()

            # Reload study to get all results
            study = optuna.load_study(study_name=search_name, storage=storage)

    # Results
    trials = sorted(
        [t for t in study.trials if t.state.name == "COMPLETE"],
        key=lambda t: t.value
    )
    k = min(100, len(trials))
    for label, key in [("Validation", "val_metrics"), ("Test", "test_metrics")]:
        print(f"\n{'='*80}\nTop {k} {label} Results:\n{'='*80}")
        for rank, t in enumerate(trials[:k], 1):
            params_str = " | ".join(f"{k}={v}" for k, v in t.params.items())
            print(f"{rank}. {params_str} | {t.user_attrs.get(key, [])}")

    best = study.best_trial
    print(f"\nBest trial #{best.number}: score={best.value:.6f} | {best.params}")

    # Heatmap if exactly 2 tuned params
    if len(param_specs) == 2:
        from optuna.visualization import plot_contour
        param_names = list(param_specs.keys())
        fig = plot_contour(study, params=param_names)
        plot_path = os.path.join(results_dir, f"optuna_{search_name}_heatmap.html")
        fig.write_html(plot_path)
        print(f"Heatmap saved to {plot_path}")


def train(config, policy,  train_data, valid_data, logger, checkpoint):
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
        else:
            env = run_epoch(config, policy, train_data, train=True)
            update_metrics(logger, env, config)
            env.close()

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

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run DQRL Policy")
    parser.add_argument('config', type=str, default='configs/DQL/MLP.yaml', help='Path to the config file.')
    parser.add_argument('--optuna', type=str, nargs='*', default=None,
                        help='Optuna hyperparameter search. Params in format "section.param=val1,val2,val3". '
                             'E.g., --optuna "model.d_model=64,128,256" "model.n_layers=2,3,4"')
    parser.add_argument('--sampler', type=str, default='tpe', choices=['tpe', 'random', 'qmc', 'grid', 'cmaes'],
                        help='Optuna sampler to use (default: tpe).')
    parser.add_argument('--n_samples', type=int, default=64, help='Number of Optuna trials (default: 50).')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel Optuna jobs (default: 1).')
    args = parser.parse_args()
    return args

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


if __name__ == '__main__':

    args = parse_args()
    config_path = args.config

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if args.optuna is not None:
        run_optuna_search(config, config_path, args)
    else:
        val_metrics, test_metrics, best_epoch = main(config)
        print(f"Validation Metrics: {val_metrics}, Test Metrics: {test_metrics}, Best Epoch: {best_epoch}")
