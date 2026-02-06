import random
import numpy as np
from core.env import Env
from core.task import Task

class Individual:
    def __init__(self, weights, biases, obs_type=["cpu", "buffer", "bw"]):
        self.weights = weights
        self.biases = biases
        self.obs_type = obs_type

    @staticmethod
    def ReLU(x):
        return np.maximum(0, x)
        
    def _make_observation(self, env: Env, task: Task, obs_type=["cpu", "buffer", "bw"]):
        """
        Returns a flat observation vector.
        For example, it concatenates free CPU, buffer, and bandwidth values.
        """
        if env is None:
            raise ValueError("Environment must be provided.")
        obs = []
        if "cpu" in obs_type:
            cpu_obs = [env.scenario.get_node(node_name).free_cpu_freq 
                       for node_name in env.scenario.get_nodes()]
            obs += cpu_obs
        if "buffer" in obs_type:
            buffer_obs = [env.scenario.get_node(node_name).buffer_free_size() 
                          for node_name in env.scenario.get_nodes()]
            obs += buffer_obs
        if "bw" in obs_type:
            bw_obs = [env.scenario.get_link(link_name[0], link_name[1]).free_bandwidth 
                      for link_name in env.scenario.get_links()]
            obs += bw_obs
        return np.array(obs)

    def act(self, env, task):
        """
        Compute an observation vector and forward-propagate it through
        the weight matrices and bias vectors (using dot products, bias addition, and ReLU activations)
        to generate scores. Returns the index of the highest score.
        """
        obs = self._make_observation(env, task, self.obs_type)
        for i in range(len(self.weights)):
            obs = np.dot(obs, self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                obs = Individual.ReLU(obs)
        return np.argmax(obs), obs


class NSGA2Policy:
    def __init__(self, env, config):
        self.config = config
        self.env = env

        self.obs_type = config["model"]["obs_type"]
        self.d_model = config["model"]["d_model"]
        self.n_layers = config["model"]["n_layers"]

        # Determine the observation dimension.
        self.n_observations = len(self._make_observation(self.env, None, self.obs_type))
        self.num_actions = len(self.env.scenario.node_id2name)

        # Initialize the population (each individual is a tuple of weight matrices and bias vectors).
        self.population = [self.genenerate_individual() 
                           for _ in range(config["training"]["pop_size"])]

    def _make_observation(self, env, task, obs_type):
        if env is None:
            raise ValueError("Environment must be provided to determine observation size.")
        obs = []
        if "cpu" in obs_type:
            cpu_obs = [env.scenario.get_node(node_name).free_cpu_freq 
                       for node_name in env.scenario.get_nodes()]
            obs += cpu_obs
        if "buffer" in obs_type:
            buffer_obs = [env.scenario.get_node(node_name).buffer_free_size() 
                          for node_name in env.scenario.get_nodes()]
            obs += buffer_obs
        if "bw" in obs_type:
            bw_obs = [env.scenario.get_link(link_name[0], link_name[1]).free_bandwidth 
                      for link_name in env.scenario.get_links()]
            obs += bw_obs
        return obs

    def genenerate_individual(self):
        """
        Generate a new individual with random weight matrices and bias vectors.
        Returns a tuple of (weights, biases).
        """
        if self.n_layers < 1:
            raise ValueError("The number of layers must be at least 1.")
        elif self.n_layers == 1:
            weights = [np.random.rand(self.n_observations, self.num_actions)]
            biases = [np.random.rand(self.num_actions)]
        elif self.n_layers == 2:
            weights = [np.random.rand(self.n_observations, self.d_model),
                       np.random.rand(self.d_model, self.num_actions)]
            biases = [np.random.rand(self.d_model),
                      np.random.rand(self.num_actions)]
        else:
            weights = [np.random.rand(self.n_observations, self.d_model)]
            biases = [np.random.rand(self.d_model)]
            for _ in range(self.n_layers - 2):
                weights.append(np.random.rand(self.d_model, self.d_model))
                biases.append(np.random.rand(self.d_model))
            weights.append(np.random.rand(self.d_model, self.num_actions))
            biases.append(np.random.rand(self.num_actions))
        return (weights, biases)

    def individuals(self):
        """
        Wrap the population's weight matrices and bias vectors into Individual objects.
        """
        return [Individual(weights, biases, self.obs_type) for weights, biases in self.population]

    def best_individual(self, fitness):
        """
        Select a single best individual via a weighted scalarization.
        Here, fitness is assumed to be tuples: (success_rate, avg_latency, avg_power),
        with success_rate to be maximized and latency/power to be minimized.
        """
        lambda_param = self.config["training"].get("latency_weight", 0.1)
        mu_param = self.config["training"].get("power_weight", 0.1)
        scores = [sr - lambda_param * l - mu_param * e for (sr, l, e) in fitness]
        best_idx = np.argmax(scores)
        return fitness[best_idx]

    # -------------------------------
    # NSGA-II Helper Functions
    # -------------------------------
    @staticmethod
    def dominates(obj1, obj2):
        """
        Check if objective vector obj1 dominates obj2 (assuming minimization).
        """
        better_or_equal = all(a <= b for a, b in zip(obj1, obj2))
        strictly_better = any(a < b for a, b in zip(obj1, obj2))
        return better_or_equal and strictly_better

    @staticmethod
    def crowding_distance(fitness_list):
        """
        Compute the crowding distance for each solution in a list.
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

    def non_dominated_sort(self, fitness):
        """
        Perform non-dominated sorting on the population.
        Returns a list of fronts (each front is a list of indices).
        """
        population_size = len(fitness)
        S = [[] for _ in range(population_size)]
        n = [0] * population_size
        fronts = [[]]
        for p in range(population_size):
            for q in range(population_size):
                if self.dominates(fitness[p], fitness[q]):
                    S[p].append(q)
                elif self.dominates(fitness[q], fitness[p]):
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
        fronts.pop()  # remove the last empty front.
        return fronts

    def select_next_generation(self, combined_population, combined_fitness, pop_size):
        """
        Use non-dominated sorting and crowding distance to select the next generation.
        """
        fronts = self.non_dominated_sort(combined_fitness)
        new_population = []
        new_fitness = []
        for front in fronts:
            if len(new_population) + len(front) <= pop_size:
                for idx in front:
                    new_population.append(combined_population[idx])
                    new_fitness.append(combined_fitness[idx])
            else:
                front_fitness = [combined_fitness[idx] for idx in front]
                distances = self.crowding_distance(front_fitness)
                # Sort the front based on descending crowding distance.
                sorted_front = sorted(list(zip(front, distances)), key=lambda x: -x[1])
                for idx, _ in sorted_front:
                    if len(new_population) < pop_size:
                        new_population.append(combined_population[idx])
                        new_fitness.append(combined_fitness[idx])
                    else:
                        break
                break
        return new_population, new_fitness

    def mutate_matrix(self, matrix, mutation_rate=None, sigma=0.1):
        """
        Apply Gaussian mutation to each element of the matrix.
        """
        if mutation_rate is None:
            mutation_rate = self.config["training"].get("mutation_rate", 0.1)
        new_matrix = np.copy(matrix)
        rows, cols = new_matrix.shape
        for i in range(rows):
            for j in range(cols):
                if random.random() < mutation_rate:
                    new_matrix[i, j] += np.random.normal(0, sigma)
        return new_matrix

    def mutate_vector(self, vector, mutation_rate=None, sigma=0.1):
        """
        Apply Gaussian mutation to each element of the bias vector.
        """
        if mutation_rate is None:
            mutation_rate = self.config["training"].get("mutation_rate", 0.1)
        new_vector = np.copy(vector)
        for i in range(len(new_vector)):
            if random.random() < mutation_rate:
                new_vector[i] += np.random.normal(0, sigma)
        return new_vector

    # -------------------------------
    # NSGA-II Update Routine
    # -------------------------------

    def tournament_selection(self, population_with_fitness, tournament_size=2):
        """
        Perform tournament selection to choose parents.
        
        Parameters:
          population_with_fitness: List of tuples (individual, fitness)
          tournament_size: Size of tournament
          
        Returns:
          Selected individual (weights, biases)
        """
        tournament = random.sample(population_with_fitness, tournament_size)
        # Select best individual from tournament (considering Pareto dominance)
        best = tournament[0]
        for candidate in tournament[1:]:
            if self.dominates(candidate[1], best[1]):
                best = candidate
        return best[0]

    def crossover(self, parent1, parent2, crossover_rate=0.8):
        """
        Perform crossover between two parents.
        
        Parameters:
          parent1, parent2: Tuples of (weights, biases)
          crossover_rate: Probability of performing crossover
          
        Returns:
          Two offspring as tuples of (weights, biases)
        """
        parent1_weights, parent1_biases = parent1
        parent2_weights, parent2_biases = parent2
        
        if random.random() > crossover_rate:
            # No crossover, return copies of parents
            return (
                ([w.copy() for w in parent1_weights], [b.copy() for b in parent1_biases]),
                ([w.copy() for w in parent2_weights], [b.copy() for b in parent2_biases])
            )
        
        # Arithmetic crossover
        alpha = random.random()
        
        # Crossover for weights
        child1_weights = []
        child2_weights = []
        for w1, w2 in zip(parent1_weights, parent2_weights):
            child1_w = alpha * w1 + (1 - alpha) * w2
            child2_w = (1 - alpha) * w1 + alpha * w2
            child1_weights.append(child1_w)
            child2_weights.append(child2_w)
        
        # Crossover for biases
        child1_biases = []
        child2_biases = []
        for b1, b2 in zip(parent1_biases, parent2_biases):
            child1_b = alpha * b1 + (1 - alpha) * b2
            child2_b = (1 - alpha) * b1 + alpha * b2
            child1_biases.append(child1_b)
            child2_biases.append(child2_b)
        
        return (child1_weights, child1_biases), (child2_weights, child2_biases)

    def update(self, fitness):
        """
        Update the population using NSGA-II selection.
        
        Parameters:
          fitness: A list of objective tuples for the current population.
                  If None, will evaluate current population.
          evaluate_offspring: Whether to evaluate offspring fitness (should be True for real runs)
          
        Returns:
          Updated fitness values for the new population
        """
        pop_size = len(self.population)
        
        # If no fitness provided, evaluate current population
        if fitness is None:
            fitness = []
            for weights, biases in self.population:
                individual = Individual(weights, biases, self.obs_type)
                fit = self.evaluate_individual(individual)
                fitness.append(fit)
        
        # Create offspring population
        offspring = []
        
        # Create population with fitness for tournament selection
        population_with_fitness = list(zip(self.population, fitness))
        
        # Generate offspring using tournament selection and crossover
        while len(offspring) < pop_size:
            # Select parents using tournament selection
            parent1 = self.tournament_selection(population_with_fitness)
            parent2 = self.tournament_selection(population_with_fitness)
            
            # Perform crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Apply mutation
            mutated_child1_weights = [self.mutate_matrix(w) for w in child1[0]]
            mutated_child1_biases = [self.mutate_vector(b) for b in child1[1]]
            
            mutated_child2_weights = [self.mutate_matrix(w) for w in child2[0]]
            mutated_child2_biases = [self.mutate_vector(b) for b in child2[1]]
            
            offspring.append((mutated_child1_weights, mutated_child1_biases))
            if len(offspring) < pop_size:
                offspring.append((mutated_child2_weights, mutated_child2_biases))
        
        # Trim offspring to exact population size
        offspring = offspring[:pop_size]
        
        # Evaluate offspring fitness
        offspring_fitness = []
        if evaluate_offspring:
            for weights, biases in offspring:
                individual = Individual(weights, biases, self.obs_type)
                fit = self.evaluate_individual(individual)
                offspring_fitness.append(fit)
        else:
            # For testing/debugging: simulate offspring fitness
            for _ in offspring:
                base_fit = random.choice(fitness)
                noise = tuple(random.uniform(-0.01, 0.01) for _ in range(len(base_fit)))
                offspring_fitness.append(tuple(b + n for b, n in zip(base_fit, noise)))
        
        # Combine current population and offspring
        combined_population = self.population + offspring
        combined_fitness = fitness + offspring_fitness
        
        # Select next generation using NSGA-II selection
        new_population, new_fitness = self.select_next_generation(
            combined_population, combined_fitness, pop_size
        )
        
        # Update population
        self.population = new_population
        
        return new_fitness