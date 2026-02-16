"""
This script demonstrates how to run the NPGAPolicy.
"""

import os
import sys
from multiprocessing import Pool, cpu_count

import torch

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
from tqdm import tqdm
import yaml

from core.env import Env
from core.task import Task
from eval.metrics.metrics import SuccessRate, AvgLatency


import numpy as np
import matplotlib.pyplot as plt

from utils.utils import create_env, get_metrics, update_metrics

def error_handler(error: Exception):
    """Customized error handler for different types of errors."""
    errors = ['DuplicateTaskIdError', 'NetworkXNoPathError', 'IsolatedWirelessNode', 'NetCongestionError', 'InsufficientBufferError']
    message = error.args[0][0]
    if message in errors:
        pass
    else:
        raise


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

    ttr, latency, energy, score = get_metrics(env, config)

    return ttr, latency, energy, score

def run_epoch(config, policy, data: pd.DataFrame, train=True, parent_fitness=None):
    """
    Run one epoch of GA training/evaluation.

    IMPORTANT: This now properly evaluates offspring fitness before selection,
    fixing the bug where fake/random fitness was assigned to offspring.

    Args:
        parent_fitness: Pre-computed fitness from previous generation (avoids re-evaluation)
    """
    n_processes = max(1, cpu_count() - 1)

    # Only evaluate parents if fitness not provided (first generation or validation)
    if parent_fitness is None:
        with Pool(processes=n_processes) as pool:
            args = [(ind, data, config) for ind in policy.individuals()]
            parent_results = pool.map(evaluate_individual, args)
        parent_fitness = np.array(parent_results)

    if train:
        # Step 1: Create offspring using parent fitness for selection
        offspring = policy.create_offspring(parent_fitness[:, :3])

        # Step 2: Evaluate offspring fitness (the fix - no more fake fitness!)
        offspring_individuals = policy.offspring_individuals(offspring)
        offspring_args = [(ind, data, config) for ind in offspring_individuals]

        with Pool(processes=n_processes) as pool:
            offspring_results = pool.map(evaluate_individual, offspring_args)

        offspring_fitness = np.array(offspring_results)

        # Step 3: Select next generation using real fitness values
        # Pass full fitness (all 4 columns) - selection uses first 3, returns all 4
        new_fitness = policy.select_from_combined(parent_fitness, offspring, offspring_fitness)
        fitness = np.array(new_fitness)
    else:
        fitness = parent_fitness

    return fitness


class GAResult:
    """
    Wrapper class to make GA results compatible with the train() interface in main.py.
    Mimics the env object returned by DQL's run_epoch.
    """
    def __init__(self, fitness, max_total_time, max_total_energy, logger=None):
        self.fitness = fitness
        self.max_total_time = max_total_time
        self.max_total_energy = max_total_energy
        self.logger = logger

        # Best individual metrics
        best_idx = np.argmin(fitness[:, 3])
        self.best_metrics = fitness[best_idx]

    def close(self):
        """No-op for compatibility with env.close()"""
        pass


def evaluate_individual_generation(args):
    """
    Evaluate an individual solution for a single generation with detailed metrics.

    Args:
        args: Tuple of (policy/individual, data, config, lambda_, max_total_time, max_total_energy)

    Returns:
        Tuple of (success_rate, latency, energy, score) for the individual
    """
    policy, data, config, lambda_, max_total_time, max_total_energy = args

    env = create_env(config)
    env.max_total_time = max_total_time
    env.max_total_energy = max_total_energy

    until = 0
    launched_task_cnt = 0

    for i, task_info in data.iterrows():
        generated_time = task_info['GenerationTime']
        task = Task(
            task_id=task_info['TaskID'],
            task_size=task_info['TaskSize'],
            cycles_per_bit=task_info['CyclesPerBit'],
            trans_bit_rate=task_info['TransBitRate'],
            ddl=task_info['DDL'],
            src_name=task_info['SrcName'] if 'SrcName' in task_info else 'e0',
            task_name=task_info['TaskName']
        )

        # Wait until the simulation reaches the task's generation time.
        while True:
            while env.done_task_info:
                _ = env.done_task_info.pop(0)

            if env.now >= generated_time:
                dst_id, state = policy.act(env, task)
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
            error_handler(e)

    ttr, latency, energy, score = get_metrics(env, config)

    return ttr, latency, energy, score


def run_generation(config, policy, data: pd.DataFrame, train=True,
                   lambda_=(1, 1, 1), max_total_time=1.0, max_total_energy=1.0,
                    parent_fitness=None):
    """
    Run one generation of the genetic algorithm over the provided task data.

    Evaluates all individuals in the population in parallel and optionally
    performs selection/crossover/mutation if train=True.

    IMPORTANT: This now properly evaluates offspring fitness before selection,
    fixing the bug where fake/random fitness was assigned to offspring.

    Args:
        config: Configuration dictionary
        policy: GA policy with population of individuals
        data: DataFrame containing task information
        train: Whether to update the population (selection, crossover, mutation)
        lambda_: Tuple of (fail_weight, time_weight, energy_weight) for fitness calculation
        max_total_time: Maximum total time for normalization
        max_total_energy: Maximum total energy for normalization
        n_processes: Number of parallel processes (defaults to cpu_count-1)
        parent_fitness: Pre-computed fitness from previous generation (avoids re-evaluation)

    Returns:
        GAResult object compatible with train() interface, containing:
        - fitness: numpy array with shape (n_individuals, 4) [ttr, latency, energy, score]
        - max_total_time, max_total_energy: normalization values
        - best_metrics: metrics of the best individual
        - close(): no-op method for compatibility
    """
    n_processes = config.get("training", {}).get("n_processes", None)
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    # Only evaluate parents if fitness not provided (first generation or validation)
    if parent_fitness is None:
        individuals = policy.individuals()
        args = [
            (ind, data, config, lambda_, max_total_time, max_total_energy)
            for ind in individuals
        ]

        with Pool(processes=n_processes) as pool:
            results = list(tqdm(
                pool.imap(evaluate_individual_generation, args),
                total=len(individuals),
                desc="Evaluating parents"
            ))

        parent_fitness = np.array(results)

    # Update the population if training
    if train:
        # Step 1: Create offspring using parent fitness for selection
        offspring = policy.create_offspring(parent_fitness[:, :3])

        # Step 2: Evaluate offspring fitness (the fix - no more fake fitness!)
        offspring_individuals = policy.offspring_individuals(offspring)
        offspring_args = [
            (ind, data, config, lambda_, max_total_time, max_total_energy)
            for ind in offspring_individuals
        ]

        with Pool(processes=n_processes) as pool:
            offspring_results = list(tqdm(
                pool.imap(evaluate_individual_generation, offspring_args),
                total=len(offspring_individuals),
                desc="Evaluating offspring"
            ))

        offspring_fitness = np.array(offspring_results)

        # Step 3: Select next generation using real fitness values
        # Pass full fitness (all 4 columns) - selection uses first 3, returns all 4
        new_fitness = policy.select_from_combined(parent_fitness, offspring, offspring_fitness)
        fitness = np.array(new_fitness)
    else:
        fitness = parent_fitness

    # Log generation statistics
    best_idx = np.argmin(fitness[:, 3])
    avg_fitness = np.mean(fitness, axis=0)
    best_fitness = fitness[best_idx]

    print(f"Generation stats - Best: SR={best_fitness[0]:.4f}, L={best_fitness[1]:.4f}, "
          f"E={best_fitness[2]:.4f}, Score={best_fitness[3]:.4f}")
    print(f"                   Avg:  SR={avg_fitness[0]:.4f}, L={avg_fitness[1]:.4f}, "
          f"E={avg_fitness[2]:.4f}, Score={avg_fitness[3]:.4f}")

    return GAResult(fitness, max_total_time, max_total_energy)



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
     - Task Throw Rate vs. Latency (maximize Task Throw Rate, minimize latency)
     - Task Throw Rate vs. Energy (maximize Task Throw Rate, minimize energy)
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
    
    # Task Throw Rate vs. Latency.
    ax = axes[0]
    ax.scatter(latency, success_rate, color='blue', label='Individuals')
    points = np.array(list(zip(success_rate, latency)))
    mask = pareto(points, maximize=(False, False))
    pareto_points = points[mask]
    idx_sort = np.argsort(pareto_points[:, 1])
    pareto_points = pareto_points[idx_sort]
    ax.plot(pareto_points[:, 1], pareto_points[:, 0], color='red', marker='o', label='Pareto Frontier')
    ax.set_xlabel('Latency')
    ax.set_ylabel('Task Throw Rate')
    ax.set_title('Task Throw Rate vs. Latency')
    ax.legend()
    
    # Task Throw Rate vs. Energy.
    ax = axes[1]
    ax.scatter(energy, success_rate, color='blue', label='Individuals')
    points = np.array(list(zip(success_rate, energy)))
    mask = pareto(points, maximize=(False, False))
    pareto_points = points[mask]
    idx_sort = np.argsort(pareto_points[:, 1])
    pareto_points = pareto_points[idx_sort]
    ax.plot(pareto_points[:, 1], pareto_points[:, 0], color='red', marker='o', label='Pareto Frontier')
    ax.set_xlabel('Energy')
    ax.set_ylabel('Task Throw Rate')
    ax.set_title('Task Throw Rate vs. Energy')
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


