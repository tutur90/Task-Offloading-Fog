"""
This script demonstrates how to run the DQRLPolicy.

Oh, wait a moment. It seems that extra effort is required to make this method work. The current version 
is for reference only, and contributions are welcome.
"""

import os
import sys
import multiprocessing

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
from tqdm import tqdm
import yaml

import numpy as np

from core.task import Task
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
    lambda_to_key, params_to_key, generate_parameter_grid, apply_params_to_config, parse_grid_search_params
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

    for epoch in range(config["training"]["num_epochs"]):

        logger.update_epoch(epoch)

        # Training phase.

        logger.update_mode('Training')

        if is_ga:
            result = run_generation(config, policy, train_data, train=True,  max_total_time=max_total_time, max_total_energy=max_total_energy)
            update_metrics(logger, None, config, metrics=tuple(result.best_metrics))
            max_total_time = result.max_total_time
            max_total_energy = result.max_total_energy
            result.close()
        else:
            env = run_epoch(config, policy, train_data, train=True, lambda_=config["training"]["lambda"], max_total_time=max_total_time, max_total_energy=max_total_energy)
            update_metrics(logger, env, config)
            max_total_time = env.max_total_time
            max_total_energy = env.max_total_energy
            env.close()

        # Validation phase.

        logger.update_mode('Validation')

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

        if logger.is_best(score[3], epoch):
            checkpoint.save(policy, epoch)

        if not is_ga:
            policy.epsilon *= config["training"]["epsilon_decay"]

            for param_group in policy.optimizer.param_groups:
                param_group['lr'] *= config["training"]["lr_decay"]
            if config["algo"] == "TaskFormer":
                for param_group in policy.optimizer.param_groups:
                    param_group['lr'] *= config["training"]["lr_decay"]

    return max_total_energy, max_total_time, score

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run DQRL Policy")
    parser.add_argument('--config', type=str, default='configs/DQRL/MLP.yaml', help='Path to the config file.')
    parser.add_argument('--grid_search', action='store_true', help='Enable grid search mode.')
    parser.add_argument('--grid_params', type=str, nargs='+', default=None,
                        help='Grid search parameters in format "section.param=val1,val2,val3". '
                             'E.g., --grid_params "model.d_model=64,128,256" "model.n_layers=2,3,4"')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers for grid search (default: CPU count).')
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
        
            print(f"Normalized training lambda values: {config["training"]["lambda"][0]:.3f}, {config["training"]["lambda"][1]:.3f}, {config["training"]["lambda"][2]:.3f}")
        
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

    logger.close()
    env.close()

    if config["algo"] not in GA_ALGOS:
        vis_stats = VisStats(save_path=logger.log_dir)
        vis_stats.vis(env)

    return val_metrics, test_metrics


def run_grid_search_worker(args):
    """Worker function for parallel grid search."""
    i, params, config_path, is_lambda_search = args

    # Load fresh config for each worker
    with open(config_path, 'r') as file:
        worker_config = yaml.safe_load(file)

    # Add worker_id to config to create unique log directories
    worker_config["worker_id"] = i

    if is_lambda_search:
        # Legacy lambda search
        worker_config["training"]["lambda"] = params.tolist()
        key = lambda_to_key(params)
        print(f"[Worker {i}] Running grid search [{i+1}] with lambda: {worker_config['training']['lambda']}")
    else:
        # Generic parameter search
        apply_params_to_config(worker_config, params)
        key = params_to_key(params)
        print(f"[Worker {i}] Running grid search [{i+1}] with params: {params}")

    val_result, test_result = main(worker_config)

    return i, key, params, val_result, test_result


if __name__ == '__main__':

    from itertools import combinations
    args = parse_args()
    config_path = args.config

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)



    if args.grid_search:

        # Determine if using lambda search or generic parameter search
        is_lambda_search = args.grid_params is None

        if is_lambda_search:
            grid = generate_probability_grid(21)
            search_name = "lambda"
        else:
            param_specs = parse_grid_search_params(args.grid_params)
            grid = generate_parameter_grid(param_specs)
            search_name = "_".join(k.replace(".", "_") for k in param_specs.keys())
            print(f"Grid search over parameters: {list(param_specs.keys())}")
            print(f"Total combinations: {len(grid)}")

        # Setup results file for resumable grid search
        results_dir = f"logs/{config['env']['dataset']}/{config['env']['flag']}"
        results_file = f"{results_dir}/grid_search_{search_name}_progress.json"

        # Load previous progress
        progress = load_grid_search_progress(results_file)
        completed_count = len(progress["completed"])
        if completed_count > 0:
            print(f"Resuming grid search: {completed_count}/{len(grid)} iterations already completed")

        val_metrics = np.zeros((len(grid), 4))
        test_metrics = np.zeros((len(grid), 4))

        # Prepare work items (skip already completed)
        work_items = []
        for i, params in enumerate(grid):
            if is_lambda_search:
                key = lambda_to_key(params)
            else:
                key = params_to_key(params)

            if key in progress["completed"]:
                val_metrics[i] = progress["val_metrics"][key]
                test_metrics[i] = progress["test_metrics"][key]
            else:
                work_items.append((i, params, config_path, is_lambda_search))

        # Run parallel grid search
        max_workers = args.num_workers if args.num_workers else multiprocessing.cpu_count()
        num_workers = min(max_workers, len(work_items))
        if num_workers > 0:
            print(f"Starting parallel grid search with {num_workers} workers for {len(work_items)} remaining items")

            with multiprocessing.Pool(num_workers) as pool:
                for result in pool.imap_unordered(run_grid_search_worker, work_items):
                    i, key, params, val_result, test_result = result

                    val_metrics[i] = val_result
                    test_metrics[i] = test_result

                    print(f"Validation Metrics: {val_metrics[i]}, Test Metrics: {test_metrics[i]}")

                    # Save progress after each iteration
                    progress["completed"][key] = True
                    progress["val_metrics"][key] = val_metrics[i].tolist()
                    progress["test_metrics"][key] = test_metrics[i].tolist()
                    save_grid_search_progress(results_file, progress)
                    completed_count += 1
                    print(f"Progress saved ({completed_count}/{len(grid)} completed)")

        # Print top k results based on metrics[:, 3]
        k = min(100, len(grid))
        print_top_k_results(grid, val_metrics, k=k, label="Validation Results")
        print_top_k_results(grid, test_metrics, k=k, label="Test Results")

        # Only plot ternary for lambda search
        if is_lambda_search:
            plot_ternary(grid, values=test_metrics[:, 3], title='Test Score Lambda Grid', labels=['λ0', 'λ1', 'λ2'], output_path=f"{results_dir}/lambda_grid_search_test.png", max_value=0.8)
            plot_ternary(grid, values=val_metrics[:, 3], title='Validation Score Lambda Grid', labels=['λ0', 'λ1', 'λ2'], output_path=f"{results_dir}/lambda_grid_search_val.png", max_value=0.8)
        elif len(param_specs) == 2:
            plot_grid_search_heatmap(param_specs, progress, metric_idx=3, metric_name="Score",
                                     output_path=f"{results_dir}/grid_search_{search_name}_heatmap.png", max_value=0.8)

    else:
        val_metrics, test_metrics = main(config)
        print(f"Validation Metrics: {val_metrics}, Test Metrics: {test_metrics}")
            
            


