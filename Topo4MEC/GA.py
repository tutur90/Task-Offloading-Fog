"""
This script demonstrates how to run the NPGAPolicy.
"""

import os
import sys
from multiprocessing import Pool, cpu_count

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
from tqdm import tqdm
import yaml

from core.env import Env
from core.task import Task
from core.vis import *
from core.vis.vis_stats import VisStats
from core.vis.logger import Logger
from eval.benchmarks.Pakistan.scenario import Scenario
from eval.metrics.metrics import SuccessRate, AvgLatency
from policies.npga.npga_policy import Individual, NPGAPolicy
from policies.npga.nsga_policy import NSGA2Policy

import numpy as np
import matplotlib.pyplot as plt

def error_handler(error: Exception):
    """Customized error handler for different types of errors."""
    errors = ['DuplicateTaskIdError', 'NetworkXNoPathError', 'IsolatedWirelessNode', 'NetCongestionError', 'InsufficientBufferError']
    message = error.args[0][0]
    if message in errors:
        pass
    else:
        raise

def select_best_individual(fitness, lambda_=(1, 0.001, 0.00001)):
    """
    Select the best individual based on the fitness values.
    """
    scores = [lambda_[0]*f[0] + lambda_[1]*f[1] + lambda_[2]*f[2] for f in fitness]
    best_idx = np.argmax(scores)
    return best_idx

def evaluate_individual(args):
    """
    Evaluate an individual solution.
    """
    m1 = SuccessRate()
    m2 = AvgLatency()
    
    policy, data, config = args
    env = create_env(config)
    
    until = 0
    launched_task_cnt = 0
    iter_data = data.iterrows()
    
    for i, task_info in iter_data:
        generated_time = task_info['GenerationTime']
        task = Task(task_id=task_info['TaskID'],
                    task_size=task_info['TaskSize'],
                    cycles_per_bit=task_info['CyclesPerBit'],
                    trans_bit_rate=task_info['TransBitRate'],
                    ddl=task_info['DDL'] / 10,
                    src_name='e0',
                    task_name=task_info['TaskName'])
        
        while True:
            # Catch completed task information.
            while env.done_task_info:
                _ = env.done_task_info.pop(0)
            
            if env.now >= generated_time:
                dst_id, state = policy.act(env, task)  # offloading decision
                dst_name = env.scenario.node_id2name[dst_id]
                env.process(task=task, dst_name=dst_name)
                launched_task_cnt += 1
                break
            
            until += env.refresh_rate
            try:
                env.run(until=until)
            except Exception as e:
                error_handler(e)
    
    # Continue simulation until all launched tasks are completed.
    while env.task_count < launched_task_cnt:
        until += env.refresh_rate
        try:
            env.run(until=until)
        except Exception as e:
            pass
            
    success_rate = m1.eval(env.logger)
    avg_latency = m2.eval(env.logger) / (success_rate + 1e-6)
    avg_power = env.avg_node_power() / (success_rate + 1e-6)
    
    return success_rate, avg_latency, avg_power

def run_epoch(config, policy, data: pd.DataFrame, train=True):
    pool = Pool(processes=cpu_count())
    args = [(ind, data, config) for ind in policy.individuals()]
    fitness = pool.map(evaluate_individual, args)
    pool.close()
    pool.join()
    
    if train:
        policy.update(fitness)
    
    # Transform fitness if needed (e.g. weight latency and power by success_rate)
    fitness = [[f[0], f[1]*f[0], f[2]*f[0]] for f in fitness]
    return fitness

def create_env(config):
    """Create and return an environment instance."""
    dataset = config["env"]["dataset"]
    flag = config["env"]["flag"]
    scenario = Scenario(config_file=f"eval/benchmarks/{dataset}/data/{flag}/config.json", flag=flag)
    env = Env(scenario, config_file="core/configs/env_config_null.json", verbose=False)
    env.refresh_rate = config['env']['refresh_rate']
    return env

def pareto(points, maximize=(True, True)):
    """
    Compute the Pareto optimal mask for a set of 2D points.
    'points' is an array of shape (N,2).
    'maximize' is a tuple of booleans indicating whether to maximize each objective.
    Returns a boolean array where True indicates the point is Pareto optimal.
    """
    n_points = points.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        for j in range(n_points):
            if i == j:
                continue
            better0 = points[j,0] > points[i,0] if maximize[0] else points[j,0] < points[i,0]
            better1 = points[j,1] > points[i,1] if maximize[1] else points[j,1] < points[i,1]
            if better0 and better1:
                is_pareto[i] = False
                break
    return is_pareto

def plot_pareto(fitness, log_dir, epoch=None):
    """
    Plot Pareto frontiers for:
     - Success Rate vs. Latency (maximize success rate, minimize latency)
     - Success Rate vs. Energy (maximize success rate, minimize energy)
     - Latency vs. Energy (minimize both)
    
    If 'epoch' is provided, the plot is saved as 'pareto_frontiers_epoch_{epoch}.png'
    in the directory "{log_dir}/pareto". Otherwise, a default filename is used.
    """
    # Convert fitness list to numpy array.
    fitness_arr = np.array(fitness)
    success_rate = fitness_arr[:, 0]
    latency = fitness_arr[:, 1]
    energy = fitness_arr[:, 2]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Success Rate vs. Latency.
    ax = axes[0]
    ax.scatter(latency, success_rate, color='blue', label='Individuals')
    points = np.array(list(zip(success_rate, latency)))
    mask = pareto(points, maximize=(True, False))
    pareto_points = points[mask]
    idx_sort = np.argsort(pareto_points[:, 1])
    pareto_points = pareto_points[idx_sort]
    ax.plot(pareto_points[:, 1], pareto_points[:, 0], color='red', marker='o', label='Pareto Frontier')
    ax.set_xlabel('Latency')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate vs. Latency')
    ax.legend()
    
    # Success Rate vs. Energy.
    ax = axes[1]
    ax.scatter(energy, success_rate, color='blue', label='Individuals')
    points = np.array(list(zip(success_rate, energy)))
    mask = pareto(points, maximize=(True, False))
    pareto_points = points[mask]
    idx_sort = np.argsort(pareto_points[:, 1])
    pareto_points = pareto_points[idx_sort]
    ax.plot(pareto_points[:, 1], pareto_points[:, 0], color='red', marker='o', label='Pareto Frontier')
    ax.set_xlabel('Energy')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate vs. Energy')
    ax.legend()
    
    # Latency vs. Energy.
    ax = axes[2]
    ax.scatter(latency, energy, color='blue', label='Individuals')
    points = np.array(list(zip(latency, energy)))
    mask = pareto(points, maximize=(False, False))
    pareto_points = points[mask]
    idx_sort = np.argsort(pareto_points[:, 0])
    pareto_points = pareto_points[idx_sort]
    ax.plot(pareto_points[:, 0], pareto_points[:, 1], color='red', marker='o', label='Pareto Frontier')
    ax.set_xlabel('Latency')
    ax.set_ylabel('Energy')
    ax.set_title('Latency vs. Energy')
    ax.legend()
    
    plt.tight_layout()
    
    # Save the plot.
    save_dir = os.path.join(log_dir, 'pareto')
    os.makedirs(save_dir, exist_ok=True)
    if epoch is not None:
        save_path = os.path.join(save_dir, f'pareto_frontiers_epoch_{epoch}.png')
    else:
        save_path = os.path.join(save_dir, 'pareto_frontiers.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Pareto frontier plot saved to {save_path}")

def main():
    config_path = "main/configs/GA/NPGA.yaml"
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    logger = Logger(config)
    env = create_env(config)
    
    # Load train and test datasets.
    train_data = pd.read_csv(f"eval/benchmarks/Pakistan/data/{config['env']['flag']}/trainset.csv")
    test_data = pd.read_csv(f"eval/benchmarks/Pakistan/data/{config['env']['flag']}/testset.csv")

    if config["policy"] == "NPGA":
        policy = NPGAPolicy(env, config)
    if config["policy"] == "NSGA2":
        policy = NSGA2Policy(env, config)
    
    # Training and testing loop.
    for epoch in range(config["training"]["num_epochs"]):
        logger.update_epoch(epoch)
        
        # Training phase.
        logger.update_mode('Training')
        tr_fitness = run_epoch(config, policy, train_data, train=True)
        SR, L, E = tr_fitness[select_best_individual(tr_fitness)]
        logger.update_metric('SuccessRate', SR)
        logger.update_metric('AvgLatency', L)
        logger.update_metric("AvgPower", E)
        
        # Testing phase.
        logger.update_mode('Testing')
        fitness = run_epoch(config, policy, test_data, train=False)
        SR, L, E = fitness[select_best_individual(fitness)]
        logger.update_metric('SuccessRate', SR)
        logger.update_metric('AvgLatency', L)
        logger.update_metric("AvgPower", E)
        env.close()
        
        # Plot Pareto for this epoch.
        plot_pareto(fitness, logger.log_dir, epoch=epoch)
    
    logger.plot()
    logger.save_csv()
    
    vis_stats = VisStats(save_path=logger.log_dir)
    vis_stats.vis(env)
    
    # Final evaluation on test data.
    pool = Pool(processes=cpu_count())
    args = [(ind, test_data, config) for ind in policy.individuals()]
    final_fitness = pool.map(evaluate_individual, args)
    pool.close()
    pool.join()
    
    # Plot final Pareto frontiers.
    plot_pareto(final_fitness, logger.log_dir)
    
    logger.close()
    env.close()

if __name__ == '__main__':
    main()
