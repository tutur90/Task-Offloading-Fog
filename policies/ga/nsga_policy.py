import random
import numpy as np
from core.env import Env
from core.task import Task

class Individual:
    def __init__(self, weights, biases, obs_type=["cpu", "buffer", "bw"], norm=None):
        self.weights = weights
        self.biases = biases
        self.obs_type = obs_type
        self.norm = norm  # Normalization factor (max values per feature)

    @staticmethod
    def ReLU(x):
        return np.maximum(0, x)

    def _make_observation(self, env: Env, task: Task, obs_type=["cpu", "buffer", "bw"]):
        """
        Returns a flat observation vector with normalization.
        For example, it concatenates free CPU, buffer, and bandwidth values.
        """
        if env is None:
            raise ValueError("Environment must be provided.")

        n_nodes = len(env.scenario.get_nodes())
        n_features = len(obs_type)
        obs = np.zeros((n_nodes, n_features), dtype=np.float32)

        for node_name in env.scenario.get_nodes():
            node_id = env.scenario.node_name2id[node_name]
            if "cpu" in obs_type:
                obs[node_id, obs_type.index("cpu")] = env.scenario.get_node(node_name).free_cpu_freq
            if "buffer" in obs_type:
                obs[node_id, obs_type.index("buffer")] = env.scenario.get_node(node_name).buffer_free_size()
            if "bw" in obs_type:
                src_node = "e0"
                if node_name != src_node:
                    obs[node_id, obs_type.index("bw")] = min(
                        link.free_bandwidth for link in env.scenario.infrastructure.get_shortest_links(src_node, node_name)
                    )
                else:
                    obs[node_id, obs_type.index("bw")] = max(
                        link.free_bandwidth for link in env.scenario.infrastructure.get_links().values()
                    )

        # Apply normalization if norm is set
        if self.norm is not None:
            obs = obs / self.norm

        return obs.flatten()

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
    def __init__(self, env, config, dataset=None):
        self.config = config
        self.env = env

        self.obs_type = config["model"]["obs_type"]
        self.d_model = config["model"]["d_model"]
        self.n_layers = config["model"]["n_layers"]

        # Compute initial observation to determine dimensions and normalization
        initial_obs = self._make_observation(self.env, None, self.obs_type)

        # Store normalization factor (max values per feature, same as MLP policy)
        self.norm = initial_obs.max(axis=0, keepdims=True)
        # Avoid division by zero
        self.norm = np.where(self.norm == 0, 1.0, self.norm)

        # Determine the observation dimension (flattened size)
        self.n_observations = initial_obs.size
        self.num_actions = len(self.env.scenario.node_id2name)

        # Initialize the population (each individual is a tuple of weight matrices and bias vectors).
        self.population = [self.genenerate_individual()
                           for _ in range(config["training"]["pop_size"])]

    def _make_observation(self, env, task, obs_type):
        """
        Returns observation as a 2D array of shape (n_nodes, n_features).
        Same structure as MLP policy for consistent normalization.
        """
        if env is None:
            raise ValueError("Environment must be provided to determine observation size.")

        n_nodes = len(env.scenario.get_nodes())
        n_features = len(obs_type)
        obs = np.zeros((n_nodes, n_features), dtype=np.float32)

        for node_name in env.scenario.get_nodes():
            node_id = env.scenario.node_name2id[node_name]
            if "cpu" in obs_type:
                obs[node_id, obs_type.index("cpu")] = env.scenario.get_node(node_name).free_cpu_freq
            if "buffer" in obs_type:
                obs[node_id, obs_type.index("buffer")] = env.scenario.get_node(node_name).buffer_free_size()
            if "bw" in obs_type:
                src_node = "e0"
                if node_name != src_node:
                    obs[node_id, obs_type.index("bw")] = min(
                        link.free_bandwidth for link in env.scenario.infrastructure.get_shortest_links(src_node, node_name)
                    )
                else:
                    obs[node_id, obs_type.index("bw")] = max(
                        link.free_bandwidth for link in env.scenario.infrastructure.get_links().values()
                    )

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
        return [Individual(weights, biases, self.obs_type, self.norm) for weights, biases in self.population]

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

    def tournament_selection(self, population_with_rank_and_distance, tournament_size=2):
        """
        Perform tournament selection using NSGA-II's crowded comparison operator.

        In NSGA-II, selection is based on:
        1. Pareto rank (lower is better)
        2. Crowding distance (higher is better, when ranks are equal)

        Parameters:
          population_with_rank_and_distance: List of tuples (individual, fitness, rank, crowding_distance)
          tournament_size: Size of tournament

        Returns:
          Selected individual (weights, biases)
        """
        tournament = random.sample(population_with_rank_and_distance, tournament_size)

        # Select best individual using crowded comparison operator
        best = tournament[0]
        for candidate in tournament[1:]:
            # Compare by rank first (lower is better)
            if candidate[2] < best[2]:
                best = candidate
            # If same rank, compare by crowding distance (higher is better)
            elif candidate[2] == best[2] and candidate[3] > best[3]:
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
        return parent1, parent2 # Placeholder: no crossover for now
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

    def assign_rank_and_crowding(self, population, fitness):
        """
        Assign Pareto rank and crowding distance to each individual.

        Parameters:
          population: List of individuals (weights, biases)
          fitness: List of fitness tuples

        Returns:
          List of tuples (individual, fitness, rank, crowding_distance)
        """
        fronts = self.non_dominated_sort(fitness)
        ranks = [0] * len(population)
        crowding_distances = [0.0] * len(population)

        for rank, front in enumerate(fronts):
            # Assign rank to each individual in this front
            for idx in front:
                ranks[idx] = rank

            # Compute crowding distance for this front
            front_fitness = [fitness[idx] for idx in front]
            front_distances = self.crowding_distance(front_fitness)

            # Assign crowding distance to each individual
            for i, idx in enumerate(front):
                crowding_distances[idx] = front_distances[i]

        return [
            (population[i], fitness[i], ranks[i], crowding_distances[i])
            for i in range(len(population))
        ]

    def create_offspring(self, fitness):
        """
        Create offspring population using tournament selection, crossover, and mutation.

        Parameters:
          fitness: A list of objective tuples for the current population.
                   Each tuple contains objectives to be MINIMIZED.

        Returns:
          List of offspring individuals (weights, biases) that need to be evaluated.
        """
        pop_size = len(self.population)

        # Convert fitness to list of tuples if it's a numpy array
        if hasattr(fitness, 'tolist'):
            fitness = [tuple(f) for f in fitness]
        else:
            fitness = [tuple(f) for f in fitness]

        # Assign rank and crowding distance to current population
        population_with_rank_and_distance = self.assign_rank_and_crowding(
            self.population, fitness
        )

        # Create offspring population using tournament selection,
        # crossover, and mutation
        offspring = []

        while len(offspring) < pop_size:
            # Select parents using tournament selection with crowded comparison
            parent1 = self.tournament_selection(population_with_rank_and_distance)
            parent2 = self.tournament_selection(population_with_rank_and_distance)

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
        return offspring[:pop_size]

    def offspring_individuals(self, offspring):
        """
        Wrap offspring (weights, biases) tuples into Individual objects for evaluation.
        """
        return [Individual(weights, biases, self.obs_type, self.norm) for weights, biases in offspring]

    def _is_feasible(self, fitness_tuple):
        """
        Check if an individual meets the minimum score thresholds for all objectives.

        Thresholds act as upper bounds on each minimized objective: an individual is
        feasible only if all its objective values are <= the corresponding threshold.
        Set a threshold to None to disable that constraint.

        Parameters:
          fitness_tuple: Tuple of objective values (ttr, latency, energy, ...)

        Returns:
          True if feasible, False if any active threshold is exceeded.
        """
        min_scores = self.config.get("selection", {}).get("min_scores", None)
        
        # print(f"Checking feasibility for fitness {fitness_tuple} against thresholds {min_scores}")
        if not min_scores:
            return True
        for obj_val, threshold in zip(fitness_tuple[:3], min_scores):
            if threshold is not None and obj_val > threshold:
                return False
        return True

    def select_from_combined(self, parent_fitness, offspring, offspring_fitness):
        """
        Combine parents and offspring, then select next generation using
        non-dominated sorting and crowding distance.

        Individuals that exceed any min_scores threshold defined in config["selection"]
        are considered infeasible and are only used to fill remaining slots when there
        are not enough feasible individuals.

        Parameters:
          parent_fitness: Full fitness values for current population (parents) - can be 3 or 4 values
          offspring: List of offspring individuals (weights, biases)
          offspring_fitness: Full evaluated fitness values for offspring - can be 3 or 4 values

        Returns:
          Tuple of (selected_indices, full_fitness) where selected_indices maps to combined population
        """
        pop_size = len(self.population)

        # Convert fitness to list of tuples
        if hasattr(parent_fitness, 'tolist'):
            parent_fitness = [tuple(f) for f in parent_fitness]
        else:
            parent_fitness = [tuple(f) for f in parent_fitness]

        if hasattr(offspring_fitness, 'tolist'):
            offspring_fitness = [tuple(f) for f in offspring_fitness]
        else:
            offspring_fitness = [tuple(f) for f in offspring_fitness]

        # Store full fitness for later retrieval
        combined_full_fitness = parent_fitness + offspring_fitness

        # Use only first 3 objectives for selection (ttr, latency, energy)
        combined_selection_fitness = [f[:3] for f in combined_full_fitness]

        # Combine current population and offspring (size 2N)
        combined_population = self.population + offspring

        # Partition individuals into feasible and infeasible based on min_scores thresholds
        feasible_indices = [i for i, f in enumerate(combined_full_fitness) if self._is_feasible(f)]
        infeasible_indices = [i for i, f in enumerate(combined_full_fitness) if not self._is_feasible(f)]

        n_infeasible = len(infeasible_indices)
        if n_infeasible > 0:
            print(f"[NSGA-II] {n_infeasible}/{len(combined_population)} individuals disqualified "
                  f"by min_scores thresholds.")

        # Run NSGA-II selection on feasible individuals only
        feasible_population = [combined_population[i] for i in feasible_indices]
        feasible_selection_fitness = [combined_selection_fitness[i] for i in feasible_indices]
        feasible_full_fitness = [combined_full_fitness[i] for i in feasible_indices]

        new_population = []
        new_fitness = []

        if feasible_population:
            fronts = self.non_dominated_sort(feasible_selection_fitness)
            for front in fronts:
                if len(new_population) + len(front) <= pop_size:
                    for idx in front:
                        new_population.append(feasible_population[idx])
                        new_fitness.append(feasible_full_fitness[idx])
                else:
                    front_fitness = [feasible_selection_fitness[idx] for idx in front]
                    distances = self.crowding_distance(front_fitness)
                    sorted_front = sorted(list(zip(front, distances)), key=lambda x: -x[1])
                    for idx, _ in sorted_front:
                        if len(new_population) < pop_size:
                            new_population.append(feasible_population[idx])
                            new_fitness.append(feasible_full_fitness[idx])
                        else:
                            break
                    break

        # If not enough feasible individuals, fill remaining slots with the least-violating
        # infeasible individuals (sorted by sum of constraint violations)
        if len(new_population) < pop_size and infeasible_indices:
            min_scores = self.config.get("selection", {}).get("min_scores", None)

            def constraint_violation(idx):
                f = combined_full_fitness[idx]
                total = 0.0
                if min_scores:
                    for obj_val, threshold in zip(f[:3], min_scores):
                        if threshold is not None and obj_val > threshold:
                            total += obj_val - threshold
                return total

            sorted_infeasible = sorted(infeasible_indices, key=constraint_violation)
            for idx in sorted_infeasible:
                if len(new_population) >= pop_size:
                    break
                new_population.append(combined_population[idx])
                new_fitness.append(combined_full_fitness[idx])

        # Update population
        self.population = new_population

        return new_fitness

    def update(self, fitness):
        """
        Legacy update method - creates offspring and immediately selects.
        WARNING: This uses fake offspring fitness! Use create_offspring() and
        select_from_combined() for proper NSGA-II with real fitness evaluation.

        Parameters:
          fitness: A list of objective tuples for the current population.

        Returns:
          Updated fitness values for the new population
        """
        # Create offspring
        offspring = self.create_offspring(fitness)

        # WARNING: This assigns fake fitness - use select_from_combined() with
        # real evaluated fitness instead
        offspring_fitness = []
        for _ in offspring:
            base_fit = random.choice([tuple(f) for f in fitness])
            noise = tuple(random.uniform(-0.01, 0.01) for _ in range(len(base_fit)))
            offspring_fitness.append(tuple(b + n for b, n in zip(base_fit, noise)))

        return self.select_from_combined(fitness, offspring, offspring_fitness)
    
    def save(self, path):
        """Save the current population to a file."""
        # Convert .pt extension to .npz for numpy format
        if path.endswith('.pt'):
            path = path[:-3] + '.npz'
        # Flatten population into separate arrays for weights and biases
        save_dict = {
            'norm': self.norm,
            'n_individuals': len(self.population),
            'n_layers': self.n_layers,
        }
        for i, (weights, biases) in enumerate(self.population):
            for j, w in enumerate(weights):
                save_dict[f'ind_{i}_weight_{j}'] = w
            for j, b in enumerate(biases):
                save_dict[f'ind_{i}_bias_{j}'] = b
        np.savez_compressed(path, **save_dict)

    def load(self, path):
        """Load the population from a file."""
        # Convert .pt extension to .npz for numpy format
        if path.endswith('.pt'):
            path = path[:-3] + '.npz'
        data = np.load(path)
        self.norm = data['norm']
        n_individuals = int(data['n_individuals'])
        n_layers = int(data['n_layers'])

        self.population = []
        for i in range(n_individuals):
            weights = [data[f'ind_{i}_weight_{j}'] for j in range(n_layers)]
            biases = [data[f'ind_{i}_bias_{j}'] for j in range(n_layers)]
            self.population.append((weights, biases))