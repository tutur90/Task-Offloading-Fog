"""Lambda probability grid search over [l0, l1, l2] where l0 + l1 + l2 = 1."""

import os
import sys
import multiprocessing
import argparse

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import yaml
import numpy as np

from main import main, get_num_gpus, print_top_k_results
from utils.plots import plot_ternary
from utils.grid_search import (
    generate_probability_grid, load_grid_search_progress, save_grid_search_progress,
    lambda_to_key
)


def run_search_worker(args):
    """Worker function for parallel lambda grid search."""
    i, params, config_path, num_gpus = args

    if num_gpus > 0:
        gpu_id = i % num_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[Worker {i}] Assigned to GPU {gpu_id}")

    with open(config_path, 'r') as file:
        worker_config = yaml.safe_load(file)

    worker_config["worker_id"] = i
    worker_config["training"]["lambda"] = params.tolist()
    key = lambda_to_key(params)
    print(f"[Worker {i}] Running lambda search [{i+1}] with lambda: {worker_config['training']['lambda']}")

    val_result, test_result, best_epoch = main(worker_config)

    return i, key, params, val_result, test_result, best_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Lambda probability grid search")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--n_steps', type=int, default=21, help='Number of steps for the probability grid (default: 21).')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (default: CPU count).')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    config_path = args.config

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    samples = generate_probability_grid(args.n_steps)

    results_dir = f"logs/{config['env']['dataset']}/{config['env']['flag']}/{config['policy']}"
    results_file = f"{results_dir}/lambda_grid_search_progress.json"

    progress = load_grid_search_progress(results_file)
    completed_count = len(progress["completed"])

    if completed_count > 0:
        print(f"Resuming lambda grid search: {completed_count}/{len(samples)} iterations already completed")

    val_metrics = np.zeros((len(samples), 4))
    test_metrics = np.zeros((len(samples), 4))

    work_items = []
    num_gpus = get_num_gpus()
    if num_gpus > 0:
        print(f"Detected {num_gpus} GPU(s) — workers will be distributed round-robin across them")
    for i, params in enumerate(samples):
        key = lambda_to_key(params)
        if key in progress["completed"]:
            val_metrics[i] = progress["val_metrics"][key]
            test_metrics[i] = progress["test_metrics"][key]
        else:
            work_items.append((i, params, config_path, num_gpus))

    max_workers = args.num_workers if args.num_workers else multiprocessing.cpu_count()
    num_workers = min(max_workers, len(work_items))
    if num_workers > 1:
        print(f"Starting parallel lambda search with {num_workers} workers for {len(work_items)} remaining items")

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(num_workers) as pool:
            for result in pool.imap_unordered(run_search_worker, work_items):
                i, key, params, val_result, test_result, best_epoch = result

                val_metrics[i] = val_result
                test_metrics[i] = test_result

                print(f"Validation Metrics: {val_metrics[i]}, Test Metrics: {test_metrics[i]}")

                progress["completed"][key] = True
                progress["val_metrics"][key] = val_metrics[i].tolist()
                progress["test_metrics"][key] = test_metrics[i].tolist()
                progress["best_epoch"] = progress.get("best_epoch", {})
                progress["best_epoch"][key] = best_epoch
                save_grid_search_progress(results_file, progress)
                completed_count += 1
                print(f"Progress saved ({completed_count}/{len(samples)} completed, best epoch: {best_epoch})")
    else:
        print(f"Running lambda search sequentially for {len(work_items)} items")
        for item in work_items:
            result = run_search_worker(item)
            i, key, params, val_result, test_result, best_epoch = result

            val_metrics[i] = val_result
            test_metrics[i] = test_result

            print(f"Validation Metrics: {val_metrics[i]}, Test Metrics: {test_metrics[i]}")

            progress["completed"][key] = True
            progress["val_metrics"][key] = val_metrics[i].tolist()
            progress["test_metrics"][key] = test_metrics[i].tolist()
            progress["best_epoch"] = progress.get("best_epoch", {})
            progress["best_epoch"][key] = best_epoch
            save_grid_search_progress(results_file, progress)
            completed_count += 1
            print(f"Progress saved ({completed_count}/{len(samples)} completed, best epoch: {best_epoch})")

    k = min(100, len(samples))
    print_top_k_results(samples, val_metrics, k=k, label="Validation Results")
    print_top_k_results(samples, test_metrics, k=k, label="Test Results")

    plot_ternary(samples, values=test_metrics[:, 3], title='Test Score Lambda Grid', labels=['λ0', 'λ1', 'λ2'], output_path=f"{results_dir}/lambda_grid_search_test.png", max_value=0.8)
    plot_ternary(samples, values=val_metrics[:, 3], title='Validation Score Lambda Grid', labels=['λ0', 'λ1', 'λ2'], output_path=f"{results_dir}/lambda_grid_search_val.png", max_value=0.8)
