"""
This script demonstrates how to run the DQRLPolicy.

Oh, wait a moment. It seems that extra effort is required to make this method work. The current version 
is for reference only, and contributions are welcome.
"""

import os
import sys

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from networkx import config
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
from utils.plots import plot_ternary
from utils.grid_search import generate_probability_grid, load_grid_search_progress, save_grid_search_progress, lambda_to_key

GA_ALGOS = ["NPGA", "NSGA2"]    


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



if __name__ == '__main__':

    from itertools import combinations
    args = parse_args()
    config_path = args.config

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)



    if args.grid_search:

        grid = generate_probability_grid(21)

        # Setup results file for resumable grid search
        results_dir = f"logs/{config['env']['dataset']}/{config['env']['flag']}"
        results_file = f"{results_dir}/grid_search_progress.json"

        # Load previous progress
        progress = load_grid_search_progress(results_file)
        completed_count = len(progress["completed"])
        if completed_count > 0:
            print(f"Resuming grid search: {completed_count}/{len(grid)} iterations already completed")

        val_metrics = np.zeros((len(grid), 4))
        test_metrics = np.zeros((len(grid), 4))

        for i, lambda_ in enumerate(grid):
            key = lambda_to_key(lambda_)

            # Skip if already completed
            if key in progress["completed"]:
                val_metrics[i] = progress["val_metrics"][key]
                test_metrics[i] = progress["test_metrics"][key]
                continue

            config["training"]["lambda"] = lambda_.tolist()
            print(f"Running grid search [{i+1}/{len(grid)}] with lambda: {config['training']['lambda']}")
            val_metrics[i], test_metrics[i] = main(config)
            print(f"Validation Metrics: {val_metrics[i]}, Test Metrics: {test_metrics[i]}")

            # Save progress after each iteration
            progress["completed"][key] = True
            progress["val_metrics"][key] = val_metrics[i].tolist()
            progress["test_metrics"][key] = test_metrics[i].tolist()
            save_grid_search_progress(results_file, progress)
            print(f"Progress saved ({i+1}/{len(grid)} completed)")

        plot_ternary(grid, values=test_metrics[:, 3], title='Test Score Lambda Grid', labels=['λ0', 'λ1', 'λ2'], output_path=f"{results_dir}/lambda_grid_search_test.png")
        

        plot_ternary(grid, values=val_metrics[:, 3], title='Validation Score Lambda Grid', labels=['λ0', 'λ1', 'λ2'], output_path=f"{results_dir}/lambda_grid_search_val.png")

    else:
        val_metrics, test_metrics = main(config)
        print(f"Validation Metrics: {val_metrics}, Test Metrics: {test_metrics}")
            
            


