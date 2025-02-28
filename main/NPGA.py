import os
import sys
import random
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

# Set up the module search path.
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import your environment, scenario, task, and metrics.
from core.env import Env
from core.task import Task
from eval.benchmarks.Pakistan.scenario import Scenario
from eval.metrics.metrics import SuccessRate, AvgLatency
from core.vis.plot_score import PlotScore
from core.utils import create_log_dir   

# --------------------------
# Candidate Policy Definition
# --------------------------
class GeneticPolicy:
    def __init__(self, weights):
        """
        Candidate solution: a weight matrix.
        weights: numpy array of shape (num_nodes, num_nodes)
        """
        self.weights = np.array(weights)  # Genome is now a matrix.

    def act(self, env, task):
        """
        Compute an observation vector (free CPU frequency per node), then use matrix multiplication
        (obs @ weights) to obtain scores for each node. Return the action (index with highest score)
        along with the observation.
        """
        nodes = env.scenario.get_nodes()
        obs = np.array([env.scenario.get_node(node).free_cpu_freq for node in nodes])
        scores = obs @ self.weights  # obs: (num_nodes,), weights: (num_nodes, num_nodes)
        action = int(np.argmax(scores))
        return action, obs

# --------------------------
# Evaluation Function
# --------------------------
def evaluate_individual(individual, scenario, data, refresh_rate=0.001):
    """
    Evaluate an individual's performance on a set of tasks.
    Returns a tuple of objectives:
      - Objective 1: -Success rate (minimizing this prioritizes a high success rate)
      - Objective 2: Average latency (to minimize)
      
    A new environment is created and tasks are processed from 'data' using the individual's policy.
    """
    env = Env(scenario, config_file="core/configs/env_config_null.json",
              refresh_rate=refresh_rate, verbose=False)
    m1 = SuccessRate()
    m2 = AvgLatency()
    until = 0

    for _, task_info in data.iterrows():
        generated_time = task_info['GenerationTime']
        task = Task(task_id=task_info['TaskID'],
                    task_size=task_info['TaskSize'],
                    cycles_per_bit=task_info['CyclesPerBit'],
                    trans_bit_rate=task_info['TransBitRate'],
                    ddl=task_info['DDL'],
                    src_name='e0',
                    task_name=task_info['TaskName'])
        while env.now < generated_time:
            try:
                env.run(until=until)
            except Exception:
                pass
            until += refresh_rate

        action, _ = individual.act(env, task)
        dst_name = env.scenario.node_id2name[action]
        env.process(task=task, dst_name=dst_name)

    while env.process_task_cnt < len(data):
        until += refresh_rate
        try:
            env.run(until=until)
        except Exception:
            pass

    success = m1.eval(env.logger.task_info)   # success in [0,1]
    avg_latency = m2.eval(env.logger.task_info)
    env.close()
    # Return objectives: minimizing -success (thus maximizing success) and latency.
    return (-success, avg_latency)

def evaluate_individual_wrapper(args):
    individual, scenario, data, refresh_rate = args
    return evaluate_individual(individual, scenario, data, refresh_rate)

# --------------------------
# Pareto Dominance and Niching Helpers
# --------------------------
def dominates(obj1, obj2):
    """
    Check if objective vector obj1 dominates obj2 (assuming minimization).
    """
    better_or_equal = all(a <= b for a, b in zip(obj1, obj2))
    strictly_better = any(a < b for a, b in zip(obj1, obj2))
    return better_or_equal and strictly_better

def crowding_distance(fitness_list):
    """
    Compute crowding distance for a list of objective vectors.
    """
    num_individuals = len(fitness_list)
    if num_individuals == 0:
        return []
    distances = [0.0] * num_individuals
    num_objectives = len(fitness_list[0])
    for m in range(num_objectives):
        values = [fit[m] for fit in fitness_list]
        sorted_indices = sorted(range(num_individuals), key=lambda i: values[i])
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        for i in range(1, num_individuals - 1):
            if max(values) - min(values) == 0:
                diff = 0
            else:
                diff = (values[sorted_indices[i+1]] - values[sorted_indices[i-1]]) / (max(values) - min(values))
            distances[sorted_indices[i]] += diff
    return distances

def non_dominated_sort(fitness):
    """
    Perform non-dominated sorting on the population.
    Returns a list of fronts (each a list of indices).
    """
    population_size = len(fitness)
    S = [[] for _ in range(population_size)]
    n = [0] * population_size
    fronts = [[]]
    for p in range(population_size):
        for q in range(population_size):
            if dominates(fitness[p], fitness[q]):
                S[p].append(q)
            elif dominates(fitness[q], fitness[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()  # Remove empty last front.
    return fronts

# --------------------------
# Genetic Operators
# --------------------------
def tournament_selection(population, fitness, niche_radius):
    """
    Perform a binary tournament selection using Pareto dominance.
    """
    i, j = random.sample(range(len(population)), 2)
    if dominates(fitness[i], fitness[j]):
        return population[i]
    elif dominates(fitness[j], fitness[i]):
        return population[j]
    else:
        return population[i] if random.random() < 0.5 else population[j]

def crossover(parent1, parent2):
    """
    Perform arithmetic crossover between two parent weight matrices.
    """
    alpha = random.random()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2

def mutate(weights, mutation_rate, sigma=0.1):
    """
    Apply Gaussian mutation elementwise to a weight matrix.
    """
    new_weights = np.copy(weights)
    rows, cols = new_weights.shape
    for i in range(rows):
        for j in range(cols):
            if random.random() < mutation_rate:
                new_weights[i, j] += np.random.normal(0, sigma)
    return np.clip(new_weights, 0, None)

# --------------------------
# Selection of Next Generation
# --------------------------
def select_next_generation(population, fitness, pop_size):
    """
    Use non-dominated sorting and crowding distance to select the next generation.
    """
    fronts = non_dominated_sort(fitness)
    new_population = []
    new_fitness = []
    for front in fronts:
        if len(new_population) + len(front) <= pop_size:
            for idx in front:
                new_population.append(population[idx])
                new_fitness.append(fitness[idx])
        else:
            front_fitness = [fitness[idx] for idx in front]
            distances = crowding_distance(front_fitness)
            sorted_front = sorted(list(zip(front, distances)), key=lambda x: -x[1])
            for idx, _ in sorted_front:
                if len(new_population) < pop_size:
                    new_population.append(population[idx])
                    new_fitness.append(fitness[idx])
                else:
                    break
            break
    return new_population, new_fitness

# --------------------------
# Niched Pareto Genetic Algorithm (NPGA)
# --------------------------
def niched_pareto_ga(scenario, data, population_size=20, generations=20,
                     crossover_rate=0.9, mutation_rate=0.1, refresh_rate=0.001,
                     plotter=None):
    """
    Evolve a population of GeneticPolicy individuals (weight matrices) to optimize:
      1. -Success rate (to prioritize high success rate)
      2. Average latency.
    
    Evaluations are performed in parallel. If a PlotScore instance is provided,
    the best individual's training metrics are appended at each generation.
    Returns the final Pareto front (population and associated objectives).
    """
    num_nodes = len(scenario.get_nodes())
    population = [GeneticPolicy(np.random.rand(num_nodes, num_nodes)) for _ in range(population_size)]
    
    pool = Pool(processes=cpu_count())
    args = [(ind, scenario, data, refresh_rate) for ind in population]
    fitness = pool.map(evaluate_individual_wrapper, args)
    
    # Print initial population results.
    for ind, fit in zip(population, fitness):
        print(f"Weights:\n{ind.weights}\n-> Objectives: -Success={fit[0]:.4f}, AvgLatency={fit[1]:.4f}")
    
    for gen in range(generations):
        print(f"\n--- Generation {gen + 1} ---")
        offspring = []
        while len(offspring) < population_size:
            parent1 = tournament_selection(population, fitness, niche_radius=0.1)
            parent2 = tournament_selection(population, fitness, niche_radius=0.1)
            if random.random() < crossover_rate:
                child1_weights, child2_weights = crossover(parent1.weights, parent2.weights)
            else:
                child1_weights = np.copy(parent1.weights)
                child2_weights = np.copy(parent2.weights)
            child1_weights = mutate(child1_weights, mutation_rate)
            child2_weights = mutate(child2_weights, mutation_rate)
            offspring.append(GeneticPolicy(child1_weights))
            if len(offspring) < population_size:
                offspring.append(GeneticPolicy(child2_weights))
        
        args = [(ind, scenario, data, refresh_rate) for ind in offspring]
        offspring_fitness = pool.map(evaluate_individual_wrapper, args)
        for ind, fit in zip(offspring, offspring_fitness):
            print(f"Weights:\n{ind.weights}\n-> Objectives: -Success={fit[0]:.4f}, AvgLatency={fit[1]:.4f}")
        
        combined_population = population + offspring
        combined_fitness = fitness + offspring_fitness
        population, fitness = select_next_generation(combined_population, combined_fitness, population_size)
        
        print("Selected individuals for next generation:")
        for ind, fit in zip(population, fitness):
            print(f"Weights:\n{ind.weights}\n-> Objectives: -Success={fit[0]:.4f}, AvgLatency={fit[1]:.4f}")
        
        # Identify the best individual (with highest success rate, i.e. smallest -success value).
        best_individual, best_obj = min(zip(population, fitness), key=lambda x: x[1][0])
        best_success = -best_obj[0]   # Convert back to positive success rate.
        best_latency = best_obj[1]
        print(f"Generation {gen+1} Best -> Success Rate: {best_success:.4f}, AvgLatency: {best_latency:.4f}")
        
        # Append training metrics for this generation.
        if plotter is not None:
            plotter.append(mode='Training', metric='SuccessRate', value=best_success)
            plotter.append(mode='Training', metric='AvgLatency', value=best_latency)
    
    pool.close()
    pool.join()
    pareto_front = list(zip(population, fitness))
    return pareto_front

# --------------------------
# Main Routine: Train on trainset, evaluate on testset, and plot results at each generation.
# --------------------------
def main():
    flag = 'Tuple30K'
    scenario = Scenario(config_file=f"eval/benchmarks/Pakistan/data/{flag}/config.json", flag=flag)
    train_data = pd.read_csv(f"eval/benchmarks/Pakistan/data/{flag}/trainset.csv")
    test_data = pd.read_csv(f"eval/benchmarks/Pakistan/data/{flag}/testset.csv")
    
    generations = 20
    log_dir = create_log_dir("NPGA", flag=flag, generations=generations)
    plotter = PlotScore(metrics=['SuccessRate', 'AvgLatency'], modes=['Training', 'Testing'], save_dir=log_dir)
    
    # Run NPGA on the trainset.
    final_pareto = niched_pareto_ga(scenario, train_data,
                                    population_size=16,
                                    generations=generations,
                                    crossover_rate=0.9,
                                    mutation_rate=0.1,
                                    refresh_rate=0.001,
                                    plotter=plotter)
    
    # Select best individual on the trainset (highest success rate).
    best_individual, best_obj = min(final_pareto, key=lambda x: x[1][0])
    best_success_train = -best_obj[0]
    best_latency_train = best_obj[1]
    print("\n=== Best Individual on Trainset ===")
    print(f"Weights:\n{best_individual.weights}\n-> Train Success Rate: {best_success_train:.4f}, AvgLatency: {best_latency_train:.4f}")
    
    # Evaluate best individual on the testset.
    test_obj = evaluate_individual(best_individual, scenario, test_data, refresh_rate=0.001)
    best_success_test = -test_obj[0]
    best_latency_test = test_obj[1]
    print("\n=== Evaluation on Testset ===")
    print(f"Test Success Rate: {best_success_test:.4f}, AvgLatency: {best_latency_test:.4f}")
    
    # Append test metrics.
    plotter.append(mode='Testing', metric='SuccessRate', value=best_success_test)
    plotter.append(mode='Testing', metric='AvgLatency', value=best_latency_test)
    
    # Plot the results.
    plotter.plot(generations)
    
if __name__ == '__main__':
    main()
