"""
This script demonstrates how to run the DQRLPolicy.

Oh, wait a moment. It seems that extra effort is required to make this method work. The current version 
is for reference only, and contributions are welcome.
"""

import os
import sys
import time
import multiprocessing

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
from utils.plots import plot_ternary, plot_grid_search_heatmap
from utils.grid_search import (
    generate_probability_grid, load_grid_search_progress, save_grid_search_progress,
    lambda_to_key, params_to_key, generate_parameter_grid, generate_random_samples,
    apply_params_to_config, parse_grid_search_params, BayesianOptimizer
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


def train(config, policy,  train_data, valid_data, logger, checkpoint, max_total_energy=0, max_total_time=0):
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

        if is_ga:
            # Pass cached fitness to avoid re-evaluating parents (except first epoch)
            result = run_generation(config, policy, train_data, train=True,
                                    max_total_time=max_total_time, max_total_energy=max_total_energy,
                                    parent_fitness=cached_fitness)
            update_metrics(logger, None, config, metrics=tuple(result.best_metrics))
            max_total_time = result.max_total_time
            max_total_energy = result.max_total_energy
            # Cache fitness for next generation (these are the selected individuals)
            cached_fitness = result.fitness
            result.close()
        else:
            env = run_epoch(config, policy, train_data, train=True, lambda_=config["training"]["lambda"], max_total_time=max_total_time, max_total_energy=max_total_energy)
            update_metrics(logger, env, config)
            max_total_time = env.max_total_time
            max_total_energy = env.max_total_energy
            env.close()

        epoch_time = time.time() - epoch_start
        logger.update_metric('TimePerTask', epoch_time / len(train_data))

        # Validation phase.

        logger.update_mode('Validation')

        val_start = time.time()

        if is_ga:
            result = run_generation(config, policy, valid_data, train=False, max_total_time=max_total_time, max_total_energy=max_total_energy)
            score = update_metrics(logger, None, config, metrics=tuple(result.best_metrics))
            result.close()
        else:
            env = run_epoch(config, policy, valid_data, train=False)
            env.max_total_energy = max_total_energy
            env.max_total_time = max_total_time
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

    return max_total_energy, max_total_time, best_val_metrics

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run DQRL Policy")
    parser.add_argument('--config', type=str, default='configs/DQRL/MLP.yaml', help='Path to the config file.')
    parser.add_argument('--grid', type=str, nargs='*', default=None,
                        help='Grid search parameters in format "section.param=val1,val2,val3". '
                             'E.g., --grid "model.d_model=64,128,256" "model.n_layers=2,3,4"')
    parser.add_argument('--random', type=str, nargs='*', default=None,
                        help='Random search parameters in format "section.param=val1,val2,val3". '
                             'E.g., --random "model.d_model=64,128,256" "model.n_layers=2,3,4"')
    parser.add_argument('--bayesian', type=str, nargs='*', default=None,
                        help='Bayesian optimization parameters in format "section.param=val1,val2,val3". '
                             'E.g., --bayesian "model.d_model=64,128,256" "model.n_layers=2,3,4"')
    parser.add_argument('--n_samples', type=int, default=50, help='Number of samples for random/bayesian search (default: 50).')
    parser.add_argument('--method', type=str, default='lhs', choices=['random', 'lhs', 'sobol'],
                        help='Sampling method for random search: "random", "lhs" (Latin Hypercube), "sobol" (default: lhs).')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers for grid/random search (default: CPU count).')
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

        
    test_data = pd.read_csv(f"eval/benchmarks/{config['env']['dataset']}/data/{config['env']['flag']}/testset.csv")
    
    #         # Load train and test datasets.
    # train_data = pd.read_csv(f"eval/benchmarks/Topo4MEC/data/25N50E/trainset.csv")
    # test_data = pd.read_csv(f"eval/benchmarks/Topo4MEC/data/25N50E/testset.csv")

    # Initialize the policy.
    policy = policies[config["policy"]](env, config) 

    max_total_time = config.get("eval", {}).get("expected_max_latency", 0)
    max_total_energy = config.get("eval", {}).get("expected_max_energy", 0)

    val_metrics = None

    if "training" in config.keys():
        max_total_energy, max_total_time, val_metrics = train(config, policy, train_data, valid_data, logger, checkpoint, max_total_energy, max_total_time)
        checkpoint.load(policy, logger.best_epoch)
        
    print(f"Max total energy: {max_total_energy}, Max total time: {max_total_time}")
    
    # Testing phase.

    logger.update_mode('Testing')

    if config["algo"] in GA_ALGOS:
        result = run_generation(config, policy, test_data, train=False, max_total_time=max_total_time, max_total_energy=max_total_energy)
        test_metrics = update_metrics(logger, None, config, metrics=tuple(result.best_metrics))
        env = result  # for close() compatibility below
    else:
        env = run_epoch(config, policy, test_data, train=False, max_total_time=max_total_time, max_total_energy=max_total_energy)
        env.max_total_energy = max_total_energy
        env.max_total_time = max_total_time
        test_metrics = update_metrics(logger, env, config)


    logger.plot()
    logger.save_csv()

    best_epoch = logger.best_epoch

    logger.close()
    env.close()

    if config["algo"] not in GA_ALGOS:
        vis_stats = VisStats(save_path=logger.log_dir)
        vis_stats.vis(env)

    return val_metrics, test_metrics, best_epoch


def run_search_worker(args):
    """Worker function for parallel grid/random search."""
    i, params, config_path, search_type = args

    # Load fresh config for each worker
    with open(config_path, 'r') as file:
        worker_config = yaml.safe_load(file)

    # Add worker_id to config to create unique log directories
    worker_config["worker_id"] = i

    if search_type == "lambda":
        # Legacy lambda search
        worker_config["training"]["lambda"] = params.tolist()
        key = lambda_to_key(params)
        print(f"[Worker {i}] Running {search_type} search [{i+1}] with lambda: {worker_config['training']['lambda']}")
    else:
        # Generic parameter search (grid or random)
        apply_params_to_config(worker_config, params)
        key = params_to_key(params)
        print(f"[Worker {i}] Running {search_type} search [{i+1}] with params: {params}")

    val_result, test_result, best_epoch = main(worker_config)

    return i, key, params, val_result, test_result, best_epoch


if __name__ == '__main__':

    from itertools import combinations
    args = parse_args()
    config_path = args.config

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Determine search mode
    search_params = None
    if args.grid is not None:
        search_type = "grid"
        search_params = args.grid
    elif args.random is not None:
        search_type = "random"
        search_params = args.random
    elif args.bayesian is not None:
        search_type = "bayesian"
        search_params = args.bayesian

    if search_params is not None:
        # Check if lambda search (no params provided) or parameter search
        is_lambda_search = search_params == []

        if is_lambda_search:
            samples = generate_probability_grid(21)
            search_name = "lambda"
            param_specs = None
        else:
            param_specs = parse_grid_search_params(search_params)
            search_name = "_".join(k.replace(".", "_") for k in param_specs.keys())
            print(f"{search_type.capitalize()} search over parameters: {list(param_specs.keys())}")

            if search_type == "grid":
                samples = generate_parameter_grid(param_specs)
                print(f"Total combinations: {len(samples)}")
            elif search_type == "random":
                samples = generate_random_samples(param_specs, args.n_samples, method=args.method)
                print(f"Total samples: {len(samples)} (method: {args.method})")
            else:  # bayesian
                samples = None  # Bayesian generates samples iteratively
                print(f"Running {args.n_samples} iterations")

        # Setup results file for resumable search
        results_dir = f"logs/{config['env']['dataset']}/{config['env']['flag']}/{config['policy']}"
        results_file = f"{results_dir}/{search_type}_search_{search_name}_progress.json"

        # Load previous progress
        progress = load_grid_search_progress(results_file)
        completed_count = len(progress["completed"])

        if search_type == "bayesian" and not is_lambda_search:
            # Bayesian optimization (sequential)
            optimizer = BayesianOptimizer(param_specs, n_initial_points=min(5, args.n_samples))

            # Replay history to warm-start optimizer
            for key, score in progress.get("scores", {}).items():
                # Reconstruct params from key
                params = {}
                for part in key.split("|"):
                    k, v = part.split("=")
                    try:
                        params[k] = int(v)
                    except ValueError:
                        try:
                            params[k] = float(v)
                        except ValueError:
                            params[k] = v
                optimizer.tell(params, score)

            if completed_count > 0:
                print(f"Resuming bayesian search: {completed_count}/{args.n_samples} iterations already completed")

            all_samples = []
            val_metrics_list = []
            test_metrics_list = []

            for i in range(completed_count, args.n_samples):
                params = optimizer.ask()
                key = params_to_key(params)

                # Skip if already evaluated (shouldn't happen but just in case)
                if key in progress["completed"]:
                    continue

                print(f"[Bayesian {i+1}/{args.n_samples}] Trying params: {params}")

                # Run evaluation
                worker_config = yaml.safe_load(open(config_path, 'r'))
                worker_config["worker_id"] = i
                apply_params_to_config(worker_config, params)

                val_result, test_result, best_epoch = main(worker_config)

                # Use validation score for optimization (minimize)
                score = val_result[3]
                optimizer.tell(params, score)

                print(f"Validation Metrics: {val_result}, Test Metrics: {test_result}")

                # Save progress
                progress["completed"][key] = True
                progress["val_metrics"][key] = list(val_result)
                progress["test_metrics"][key] = list(test_result)
                progress["best_epoch"] = progress.get("best_epoch", {})
                progress["best_epoch"][key] = best_epoch
                progress["scores"] = progress.get("scores", {})
                progress["scores"][key] = score
                save_grid_search_progress(results_file, progress)
                completed_count += 1
                print(f"Progress saved ({completed_count}/{args.n_samples} completed, best epoch: {best_epoch})")

                all_samples.append(params)
                val_metrics_list.append(val_result)
                test_metrics_list.append(test_result)

            # Convert to arrays for printing
            samples = list(progress["completed"].keys())
            val_metrics = np.array([progress["val_metrics"][k] for k in samples])
            test_metrics = np.array([progress["test_metrics"][k] for k in samples])

            # Reconstruct samples as dicts for print_top_k_results
            samples_dicts = []
            for key in samples:
                params = {}
                for part in key.split("|"):
                    k, v = part.split("=")
                    try:
                        params[k] = int(v)
                    except ValueError:
                        try:
                            params[k] = float(v)
                        except ValueError:
                            params[k] = v
                samples_dicts.append(params)
            samples = samples_dicts

        else:
            # Grid or Random search (parallel)
            if completed_count > 0:
                print(f"Resuming {search_type} search: {completed_count}/{len(samples)} iterations already completed")

            val_metrics = np.zeros((len(samples), 4))
            test_metrics = np.zeros((len(samples), 4))

            # Prepare work items (skip already completed)
            work_items = []
            search_type_key = "lambda" if is_lambda_search else search_type
            for i, params in enumerate(samples):
                if is_lambda_search:
                    key = lambda_to_key(params)
                else:
                    key = params_to_key(params)

                if key in progress["completed"]:
                    val_metrics[i] = progress["val_metrics"][key]
                    test_metrics[i] = progress["test_metrics"][key]
                else:
                    work_items.append((i, params, config_path, search_type_key))

            # Run parallel search
            max_workers = args.num_workers if args.num_workers else multiprocessing.cpu_count()
            num_workers = min(max_workers, len(work_items))
            if num_workers > 1:
                print(f"Starting parallel {search_type} search with {num_workers} workers for {len(work_items)} remaining items")

                with multiprocessing.Pool(num_workers) as pool:
                    for result in pool.imap_unordered(run_search_worker, work_items):
                        i, key, params, val_result, test_result, best_epoch = result

                        val_metrics[i] = val_result
                        test_metrics[i] = test_result

                        print(f"Validation Metrics: {val_metrics[i]}, Test Metrics: {test_metrics[i]}")

                        # Save progress after each iteration
                        progress["completed"][key] = True
                        progress["val_metrics"][key] = val_metrics[i].tolist()
                        progress["test_metrics"][key] = test_metrics[i].tolist()
                        progress["best_epoch"] = progress.get("best_epoch", {})
                        progress["best_epoch"][key] = best_epoch
                        save_grid_search_progress(results_file, progress)
                        completed_count += 1
                        print(f"Progress saved ({completed_count}/{len(samples)} completed, best epoch: {best_epoch})")
            else:
                print(f"Running {search_type} search sequentially for {len(work_items)} items")
                for item in work_items:
                    result = run_search_worker(item)
                    i, key, params, val_result, test_result, best_epoch = result

                    val_metrics[i] = val_result
                    test_metrics[i] = test_result

                    print(f"Validation Metrics: {val_metrics[i]}, Test Metrics: {test_metrics[i]}")

                    # Save progress after each iteration
                    progress["completed"][key] = True
                    progress["val_metrics"][key] = val_metrics[i].tolist()
                    progress["test_metrics"][key] = test_metrics[i].tolist()
                    progress["best_epoch"] = progress.get("best_epoch", {})
                    progress["best_epoch"][key] = best_epoch
                    save_grid_search_progress(results_file, progress)
                    completed_count += 1
                    print(f"Progress saved ({completed_count}/{len(samples)} completed, best epoch: {best_epoch})")

        # Print top k results based on metrics[:, 3]
        k = min(100, len(samples))
        print_top_k_results(samples, val_metrics, k=k, label="Validation Results")
        print_top_k_results(samples, test_metrics, k=k, label="Test Results")

        # Only plot ternary for lambda search
        if is_lambda_search:
            plot_ternary(samples, values=test_metrics[:, 3], title='Test Score Lambda Grid', labels=['λ0', 'λ1', 'λ2'], output_path=f"{results_dir}/lambda_{search_type}_search_test.png", max_value=0.8)
            plot_ternary(samples, values=val_metrics[:, 3], title='Validation Score Lambda Grid', labels=['λ0', 'λ1', 'λ2'], output_path=f"{results_dir}/lambda_{search_type}_search_val.png", max_value=0.8)
        elif param_specs is not None and len(param_specs) == 2:
            plot_grid_search_heatmap(param_specs, progress, metric_idx=3, metric_name="Score",
                                     output_path=f"{results_dir}/{search_type}_search_{search_name}_heatmap.png")

    else:
        val_metrics, test_metrics, best_epoch = main(config)
        print(f"Validation Metrics: {val_metrics}, Test Metrics: {test_metrics}, Best Epoch: {best_epoch}")
            
            


