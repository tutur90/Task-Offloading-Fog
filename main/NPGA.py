import os
import sys
import random
import numpy as np
import pandas as pd

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import your environment, scenario, task, and metrics.
from core.env import Env
from core.task import Task
from eval.benchmarks.Pakistan.scenario import Scenario
from eval.metrics.metrics import SuccessRate, AvgLatency

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
        # Compute scores by matrix multiplication.
        scores = obs @ self.weights  # obs is (num_nodes,), weights is (num_nodes, num_nodes)
        action = int(np.argmax(scores))
        return action, obs

# --------------------------
# Evaluation Function
# --------------------------
def evaluate_individual(individual, scenario, data, refresh_rate=1):
    """
    Evaluate an individual's performance on a set of tasks.
    Returns a tuple of objective values:
      - avg_latency (to minimize)
      - -success_rate (to minimize, i.e. maximizing success_rate)
      
    This function creates a new environment, runs through all tasks in 'data'
    using the individual's policy, and then computes the metrics.
    """
    # Create a new environment instance.
    env = Env(scenario, config_file="core/configs/env_config_null.json",
              refresh_rate=1, verbose=False)
    m1 = SuccessRate()
    m2 = AvgLatency()
    until = 0

    # Process each task in the dataset.
    for _, task_info in data.iterrows():
        generated_time = task_info['GenerationTime']
        task = Task(task_id=task_info['TaskID'],
                    task_size=task_info['TaskSize'],
                    cycles_per_bit=task_info['CyclesPerBit'],
                    trans_bit_rate=task_info['TransBitRate'],
                    ddl=task_info['DDL'],
                    src_name='e0',
                    task_name=task_info['TaskName'])
        # Advance simulation until the task's generation time.
        while env.now < generated_time:
            try:
                env.run(until=until)
            except Exception:
                pass
            until += refresh_rate

        action, _ = individual.act(env, task)
        dst_name = env.scenario.node_id2name[action]
        env.process(task=task, dst_name=dst_name)

    # Run until all tasks are processed.
    while env.process_task_cnt < len(data):
        until += refresh_rate
        try:
            env.run(until=until)
        except Exception:
            pass

    # Compute objectives.
    success = m1.eval(env.logger.task_info)
    avg_latency = m2.eval(env.logger.task_info)
    env.close()
    # Convert success rate into a minimization objective.
    return avg_latency, -success

# --------------------------
# Pareto Dominance and Niching Helpers
# --------------------------
def dominates(obj1, obj2):
    """
    Check if objective vector obj1 dominates obj2.
    (Assumes minimization for both objectives.)
    """
    better_or_equal = all(a <= b for a, b in zip(obj1, obj2))
    strictly_better = any(a < b for a, b in zip(obj1, obj2))
    return better_or_equal and strictly_better

def crowding_distance(fitness_list):
    """
    Compute crowding distance for a list of objective vectors.
    Used as a niching measure to preserve diversity.
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
    Returns a list of fronts, each front is a list of indices.
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
    fronts.pop()  # Remove the last empty front.
    return fronts

# --------------------------
# Genetic Operators
# --------------------------
def tournament_selection(population, fitness, niche_radius):
    """
    Perform a binary tournament selection using Pareto dominance.
    In case of a tie, use the crowding (niching) measure.
    """
    i, j = random.sample(range(len(population)), 2)
    f1 = fitness[i]
    f2 = fitness[j]
    if dominates(f1, f2):
        return population[i]
    elif dominates(f2, f1):
        return population[j]
    else:
        # Tie-breaker: choose the one with lower local density (i.e. higher crowding distance).
        # Here, we approximate by simply randomizing.
        return population[i] if random.random() < 0.5 else population[j]

def crossover(parent1, parent2):
    """
    Perform arithmetic crossover between two parents.
    Both parents are weight matrices.
    """
    alpha = random.random()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2

def mutate(weights, mutation_rate, sigma=0.1):
    """
    Apply Gaussian mutation to a weight matrix.
    Mutation is applied elementwise.
    """
    new_weights = np.copy(weights)
    rows, cols = new_weights.shape
    for i in range(rows):
        for j in range(cols):
            if random.random() < mutation_rate:
                new_weights[i, j] += np.random.normal(0, sigma)
    new_weights = np.clip(new_weights, 0, None)
    return new_weights

# --------------------------
# Selection of Next Generation
# --------------------------
def select_next_generation(population, fitness, pop_size):
    """
    Combine non-dominated sorting and crowding distance to select the next generation.
    A simplified NSGA-II style selection is used.
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
# Niched Pareto Genetic Algorithm
# --------------------------
def niched_pareto_ga(scenario, data, population_size=20, generations=10,
                     crossover_rate=0.9, mutation_rate=0.1):
    """
    Evolve a population of GeneticPolicy individuals (with weight matrices)
    to optimize two objectives:
      1. Average latency (minimize)
      2. -Success rate (minimize, i.e. maximize success rate)
    
    Returns the final population approximating the Pareto front.
    """
    num_nodes = len(scenario.get_nodes())
    # Initialize population with random weight matrices.
    population = [GeneticPolicy(np.random.rand(num_nodes, num_nodes)) for _ in range(population_size)]
    fitness = []
    print("Evaluating initial population...")
    for individual in population:
        obj = evaluate_individual(individual, scenario, data)
        fitness.append(obj)
        print(f"Weights:\n{individual.weights}\n-> Objectives: Latency={obj[0]:.4f}, -Success={obj[1]:.4f}")

    for gen in range(generations):
        print(f"\n--- Generation {gen + 1} ---")
        offspring = []
        # Generate offspring until we have a full population.
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

        # Evaluate offspring.
        offspring_fitness = []
        print("Evaluating offspring...")
        for individual in offspring:
            obj = evaluate_individual(individual, scenario, data)
            offspring_fitness.append(obj)
            print(f"Weights:\n{individual.weights}\n-> Objectives: Latency={obj[0]:.4f}, -Success={obj[1]:.4f}")

        # Combine current population and offspring.
        combined_population = population + offspring
        combined_fitness = fitness + offspring_fitness

        # Select next generation using non-dominated sorting and crowding distance.
        population, fitness = select_next_generation(combined_population, combined_fitness, population_size)
        print("Selected individuals for next generation:")
        for ind, fit in zip(population, fitness):
            print(f"Weights:\n{ind.weights}\n-> Objectives: Latency={fit[0]:.4f}, -Success={fit[1]:.4f}")

    # Return final Pareto front (population and associated objectives).
    pareto_front = list(zip(population, fitness))
    return pareto_front

# --------------------------
# Main Routine
# --------------------------
def main():
    # Set up scenario and load a dataset (e.g., testset).
    flag = 'Tuple30K'
    scenario = Scenario(config_file=f"eval/benchmarks/Pakistan/data/{flag}/config.json", flag=flag)
    data = pd.read_csv(f"eval/benchmarks/Pakistan/data/{flag}/testset.csv")

    # Run the Niched Pareto GA.
    final_pareto = niched_pareto_ga(scenario, data,
                                    population_size=40,
                                    generations=200,
                                    crossover_rate=0.9,
                                    mutation_rate=0.1)
    print("\n=== Final Pareto Front ===")
    for idx, (ind, obj) in enumerate(final_pareto):
        print(f"Individual {idx + 1}:\nWeights:\n{ind.weights}\n-> Objectives: Latency={obj[0]:.4f}, -Success={obj[1]:.4f}")

if __name__ == '__main__':
    main()
