import random
import numpy as np
from core.env import Env
from core.task import Task

class Individual:
    def __init__(self, weights, obs_type=["cpu", "buffer", "bw"]):
        self.weights = weights
        self.obs_type = obs_type

    @staticmethod
    def ReLU(x):
        return np.maximum(0, x)
        
    def _make_observation(self, env: Env, task: Task, obs_type=["cpu", "buffer", "bw"]):
        """
        Returns a flat observation vector.
        For instance, returns free CPU frequency for each node combined with free bandwidth per link.
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
        Compute an observation vector and pass it through successive matrix multiplications
        (with ReLU nonlinearity between hidden layers) to produce scores for each node.
        Returns the action (index with the highest score) and the observation vector.
        """
        obs = self._make_observation(env, task, self.obs_type)
        for i in range(len(self.weights)):
            obs = np.dot(obs, self.weights[i])
            if i < len(self.weights) - 1:
                obs = self.ReLU(obs)
        return np.argmax(obs), obs


class NPGAPolicy:
    def __init__(self, env, config):
        self.config = config
        self.env = env

        self.obs_type = config["model"]["obs_type"]
        self.d_model = config["model"]["d_model"]
        self.n_layers = config["model"]["n_layers"]
        
        # Use a helper to determine observation size.
        self.n_observations = len(self._make_observation(self.env, None, self.obs_type))
        self.num_actions = len(self.env.scenario.node_id2name)
        
        # Initialize population: each individual is represented as a list of weight matrices.
        self.population = [self.genenerate_individual() 
                           for _ in range(config["training"]["pop_size"])]
    
    def _make_observation(self, env, task, obs_type):
        """
        Helper method to compute the observation vector.
        """
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
        Generate a new individual with random weights.
        """
        if self.n_layers < 1:
            raise ValueError("The number of layers must be at least 1.")
        elif self.n_layers == 1:
            weights = [np.random.rand(self.n_observations, self.num_actions)]
        elif self.n_layers == 2:
            weights = [np.random.rand(self.n_observations, self.d_model),
                       np.random.rand(self.d_model, self.num_actions)]
        else:
            weights = [np.random.rand(self.n_observations, self.d_model)]
            for _ in range(self.n_layers - 2):
                weights.append(np.random.rand(self.d_model, self.d_model))
            weights.append(np.random.rand(self.d_model, self.num_actions))
        return weights
    
    def individuals(self):
        """
        Wrap the current population into Individual objects.
        """
        return [Individual(weights, self.obs_type.copy()) for weights in self.population.copy()]
    
    
    # ---------------------------
    # NPGA Helper Functions
    # ---------------------------
    @staticmethod
    def dominates(obj1, obj2):
        """
        Check if objective vector obj1 dominates obj2 (assuming minimization).
        """
        better_or_equal = all(a <= b for a, b in zip(obj1, obj2))
        strictly_better = any(a < b for a, b in zip(obj1, obj2))
        return better_or_equal and strictly_better

    def npga_tournament_selection(self, population, fitness, niche_size):
        """
        Perform a NPGA-style binary tournament selection.
        Two individuals are randomly chosen and a niche is formed (a random subset of indices).
        The candidate with fewer dominations from the niche wins the tournament.
        """
        i, j = random.sample(range(len(population)), 2)
        candidate1 = population[i]
        candidate2 = population[j]
        
        # Create a niche: randomly select niche_size individuals from the population.
        niche_indices = random.sample(range(len(population)), min(niche_size, len(population)))
        
        def niche_domination_count(candidate_fit):
            count = 0
            for idx in niche_indices:
                if self.dominates(fitness[idx], candidate_fit):
                    count += 1
            return count
        
        count1 = niche_domination_count(fitness[i])
        count2 = niche_domination_count(fitness[j])
        
        if count1 < count2:
            return candidate1
        elif count2 < count1:
            return candidate2
        else:
            return candidate1 if random.random() < 0.5 else candidate2

    def mutate_matrix(self, matrix, mutation_rate=None, sigma=0.1):
        """
        Apply Gaussian mutation elementwise to a weight matrix.
        """
        if mutation_rate is None:
            mutation_rate = self.config["training"].get("mutation_rate", 0.1)
        new_matrix = np.copy(matrix)
        rows, cols = new_matrix.shape
        for i in range(rows):
            for j in range(cols):
                if random.random() < mutation_rate:
                    new_matrix[i, j] += np.random.normal(0, sigma)
        return np.clip(new_matrix, 0, None)
    
    def crossover(self, parent1, parent2):
        """
        Perform arithmetic crossover between two weight matrices.
        """
        alpha = random.random()
        child = []
        for w1, w2 in zip(parent1, parent2):
            child_w = alpha * w1 + (1 - alpha) * w2
            child.append(child_w)
        return child
    
    # ---------------------------
    # NPGA Update Routine
    # ---------------------------
    def create_offspring(self, fitness):
        """
        Create offspring population using NPGA tournament selection, crossover, and mutation.

        Parameters:
          fitness: List of objective tuples for the current population.

        Returns:
          List of offspring individuals (weights) that need to be evaluated.
        """
        pop_size = len(self.population)
        offspring = []
        niche_size = self.config["training"].get("niche_size", 5)

        # Generate offspring population.
        while len(offspring) < pop_size:
            parent1 = self.npga_tournament_selection(self.population, fitness, niche_size)
            parent2 = self.npga_tournament_selection(self.population, fitness, niche_size)
            child = []
            # Arithmetic crossover for each corresponding weight matrix.
            for w1, w2 in zip(parent1, parent2):
                child_w = self.crossover(w1, w2)
                child.append(child_w)
            # Mutation step.
            mutated_child = [self.mutate_matrix(w) for w in child]
            offspring.append(mutated_child)

        return offspring

    def offspring_individuals(self, offspring):
        """
        Wrap offspring weights into Individual objects for evaluation.
        """
        return [Individual(weights, self.obs_type.copy()) for weights in offspring]

    def select_from_combined(self, parent_fitness, offspring, offspring_fitness):
        """
        Select the next generation from combined parents and offspring based on
        Pareto dominance (NPGA-style selection).

        Parameters:
          parent_fitness: Full fitness values for current population (parents) - can be 3 or 4 values
          offspring: List of offspring individuals (weights)
          offspring_fitness: Full evaluated fitness values for offspring - can be 3 or 4 values

        Returns:
          Full fitness values for the new population (preserves all columns)
        """
        pop_size = len(self.population)

        # Convert to lists
        parent_fitness = [tuple(f) for f in parent_fitness]
        offspring_fitness = [tuple(f) for f in offspring_fitness]

        # Combine populations and full fitness
        combined_population = self.population + offspring
        combined_full_fitness = parent_fitness + offspring_fitness

        # Use only first 3 objectives for selection (ttr, latency, energy)
        combined_selection_fitness = [f[:3] for f in combined_full_fitness]

        # Use non-dominated sorting to select best individuals
        population_size = len(combined_selection_fitness)
        S = [[] for _ in range(population_size)]
        n = [0] * population_size

        for p in range(population_size):
            for q in range(population_size):
                if self.dominates(combined_selection_fitness[p], combined_selection_fitness[q]):
                    S[p].append(q)
                elif self.dominates(combined_selection_fitness[q], combined_selection_fitness[p]):
                    n[p] += 1

        # Assign fronts
        fronts = [[]]
        for p in range(population_size):
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
        fronts.pop()  # Remove empty front

        # Select top pop_size individuals from fronts
        new_population = []
        new_fitness = []
        for front in fronts:
            if len(new_population) + len(front) <= pop_size:
                for idx in front:
                    new_population.append(combined_population[idx])
                    new_fitness.append(combined_full_fitness[idx])
            else:
                # Fill remaining slots randomly from this front
                remaining = pop_size - len(new_population)
                selected = random.sample(front, remaining)
                for idx in selected:
                    new_population.append(combined_population[idx])
                    new_fitness.append(combined_full_fitness[idx])
                break

        self.population = new_population
        return new_fitness

    def update(self, fitness):
        """
        Legacy update method - creates offspring and replaces population.
        WARNING: This uses fake offspring fitness! Use create_offspring() and
        select_from_combined() for proper NPGA with real fitness evaluation.

        Parameters:
          fitness: List of objective tuples for the current population.

        Returns:
          Updated fitness values for the new population
        """
        # Create offspring
        offspring = self.create_offspring(fitness)

        # WARNING: This assigns fake fitness - use select_from_combined() with
        # real evaluated fitness instead
        offspring_fitness = []
        for _ in offspring:
            base_fit = random.choice(fitness)
            noise = (random.uniform(-0.01, 0.01),
                     random.uniform(-0.01, 0.01),
                     random.uniform(-0.01, 0.01))
            offspring_fitness.append(tuple(b + n for b, n in zip(base_fit, noise)))

        # Replace population with offspring (original behavior)
        self.population = offspring
        return offspring_fitness

    def save(self, path):
        """Save the current population to a file."""
        # Convert .pt extension to .npz for numpy format
        if path.endswith('.pt'):
            path = path[:-3] + '.npz'
        save_dict = {
            'n_individuals': len(self.population),
            'n_layers': self.n_layers,
        }
        for i, weights in enumerate(self.population):
            for j, w in enumerate(weights):
                save_dict[f'ind_{i}_weight_{j}'] = w
        np.savez_compressed(path, **save_dict)

    def load(self, path):
        """Load the population from a file."""
        # Convert .pt extension to .npz for numpy format
        if path.endswith('.pt'):
            path = path[:-3] + '.npz'
        data = np.load(path)
        n_individuals = int(data['n_individuals'])
        n_layers = int(data['n_layers'])

        self.population = []
        for i in range(n_individuals):
            weights = [data[f'ind_{i}_weight_{j}'] for j in range(n_layers)]
            self.population.append(weights)
