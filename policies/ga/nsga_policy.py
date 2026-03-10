import random
import numpy as np
from core.env import Env
from core.task import Task


# =============================================================================
# Individual — inference only, no training logic
# =============================================================================

class Individual:
    def __init__(self, weights, biases, obs_type=["cpu", "buffer", "bw"], norm=None, activation='relu'):
        self.weights = weights
        self.biases = biases
        self.obs_type = obs_type
        self.norm = norm
        self.activation = activation

    @staticmethod
    def relu(x):    return np.maximum(0, x)
    @staticmethod
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    @staticmethod
    def tanh(x):    return np.tanh(x)

    def _make_observation(self, env: Env, task: Task, obs_type):
        n_nodes = len(env.scenario.get_nodes())
        obs = np.zeros((n_nodes, len(obs_type)), dtype=np.float32)
        for node_name in env.scenario.get_nodes():
            node_id = env.scenario.node_name2id[node_name]
            if "cpu" in obs_type:
                obs[node_id, obs_type.index("cpu")] = env.scenario.get_node(node_name).free_cpu_freq
            if "buffer" in obs_type:
                obs[node_id, obs_type.index("buffer")] = env.scenario.get_node(node_name).buffer_free_size()
            if "bw" in obs_type:
                src = "e0"
                if node_name != src:
                    obs[node_id, obs_type.index("bw")] = min(
                        l.free_bandwidth for l in env.scenario.infrastructure.get_shortest_links(src, node_name)
                    )
                else:
                    obs[node_id, obs_type.index("bw")] = max(
                        l.free_bandwidth for l in env.scenario.infrastructure.get_links().values()
                    )
        if self.norm is not None:
            obs = obs / self.norm
        return obs.flatten()

    def act(self, env, task):
        obs = self._make_observation(env, task, self.obs_type)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            obs = np.dot(obs, w) + b
            if i < len(self.weights) - 1:
                if   self.activation == 'relu':    obs = self.relu(obs)
                elif self.activation == 'sigmoid': obs = self.sigmoid(obs)
                elif self.activation == 'tanh':    obs = self.tanh(obs)
        return np.argmax(obs), obs


# =============================================================================
# NSGA2Policy
# =============================================================================

class NSGA2Policy:

    def __init__(self, env, config, dataset=None):
        self.config = config
        self.env = env

        self.obs_type   = config["model"]["obs_type"]
        self.d_model    = config["model"]["d_model"]
        self.n_layers   = config["model"]["n_layers"]
        self.activation = config["model"].get("activation", "relu")

        # Compute normalisation from the initial environment state
        initial_obs     = self._make_observation(env, None, self.obs_type)
        self.norm       = np.where(initial_obs.max(axis=0, keepdims=True) == 0, 1.0,
                                   initial_obs.max(axis=0, keepdims=True))
        self.n_observations = initial_obs.size
        self.num_actions    = len(env.scenario.node_id2name)

        self.population = [self._generate_individual()
                           for _ in range(config["training"]["pop_size"])]

    # -------------------------------------------------------------------------
    # Observation helper (used for normalisation at init)
    # -------------------------------------------------------------------------

    def _make_observation(self, env, task, obs_type):
        n_nodes = len(env.scenario.get_nodes())
        obs = np.zeros((n_nodes, len(obs_type)), dtype=np.float32)
        for node_name in env.scenario.get_nodes():
            node_id = env.scenario.node_name2id[node_name]
            if "cpu" in obs_type:
                obs[node_id, obs_type.index("cpu")] = env.scenario.get_node(node_name).free_cpu_freq
            if "buffer" in obs_type:
                obs[node_id, obs_type.index("buffer")] = env.scenario.get_node(node_name).buffer_free_size()
            if "bw" in obs_type:
                src = "e0"
                if node_name != src:
                    obs[node_id, obs_type.index("bw")] = min(
                        l.free_bandwidth for l in env.scenario.infrastructure.get_shortest_links(src, node_name)
                    )
                else:
                    obs[node_id, obs_type.index("bw")] = max(
                        l.free_bandwidth for l in env.scenario.infrastructure.get_links().values()
                    )
        return obs

    # -------------------------------------------------------------------------
    # Individual initialisation  (He for ReLU, Xavier for tanh/sigmoid)
    # -------------------------------------------------------------------------

    def _init_weight(self, fan_in, fan_out):
        if self.activation == 'relu':
            return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))

    def _init_bias(self, size):
        return np.zeros(size)

    def _generate_individual(self):
        """Build layer dims then initialise weights and biases."""
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1.")

        # Build list of (fan_in, fan_out) for each layer
        if self.n_layers == 1:
            dims = [(self.n_observations, self.num_actions)]
        else:
            dims = ([(self.n_observations, self.d_model)]
                    + [(self.d_model, self.d_model)] * (self.n_layers - 2)
                    + [(self.d_model, self.num_actions)])

        weights = [self._init_weight(fi, fo) for fi, fo in dims]
        biases  = [self._init_bias(fo)       for _,  fo in dims]
        return weights, biases

    def individuals(self):
        return [Individual(w, b, self.obs_type, self.norm, self.activation)
                for w, b in self.population]

    # -------------------------------------------------------------------------
    # Mutation  (paper style: θ' = θ + σ·N(0,I), same σ for W and b per layer)
    # -------------------------------------------------------------------------

    def _mutation_sigma(self, fan_in):
        """Config sigma if set, else He std = sqrt(2/fan_in)."""
        cfg = self.config["training"].get("mutation_sigma", None)
        return cfg if cfg is not None else np.sqrt(2.0 / fan_in)

    def mutate_layer(self, weight, bias):
        """Mutate a weight matrix and its bias vector with the same sigma."""
        sigma = self._mutation_sigma(weight.shape[0])
        return (weight + np.random.randn(*weight.shape) * sigma,
                bias   + np.random.randn(*bias.shape)   * sigma)

    # -------------------------------------------------------------------------
    # Offspring generation
    # -------------------------------------------------------------------------

    def _tournament_select(self, ranked_pop, tournament_size=2):
        """NSGA-II crowded comparison: lower rank wins; ties broken by distance."""
        candidates = random.sample(ranked_pop, tournament_size)
        best = candidates[0]
        for c in candidates[1:]:
            if c[2] < best[2] or (c[2] == best[2] and c[3] > best[3]):
                best = c
        return best[0]   # return (weights, biases)

    def _crossover(self, parent1, parent2):
        """Placeholder — no crossover, offspring = copy of parent."""
        return parent1, parent2

    def _assign_rank_and_crowding(self, population, fitness):
        """Return list of (individual, fitness, rank, crowding_distance)."""
        fronts = self.non_dominated_sort(fitness)
        ranks = [0] * len(population)
        distances = [0.0] * len(population)
        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank
            front_dist = self.crowding_distance([fitness[i] for i in front])
            for i, idx in enumerate(front):
                distances[idx] = front_dist[i]
        return [(population[i], fitness[i], ranks[i], distances[i])
                for i in range(len(population))]

    def create_offspring(self, fitness):
        """
        Generate N offspring via tournament selection + mutation.
        Each layer's weight matrix and bias are mutated together with the same σ.
        """
        pop_size = len(self.population)
        fitness  = [tuple(f) for f in fitness]

        ranked = self._assign_rank_and_crowding(self.population, fitness)
        offspring = []

        while len(offspring) < pop_size:
            p1 = self.tournament_selection(ranked)
            p2 = self.tournament_selection(ranked)
            c1, c2 = self._crossover(p1, p2)

            # Mutate each layer (W and b together)
            c1_layers = [self.mutate_layer(w, b) for w, b in zip(c1[0], c1[1])]
            c2_layers = [self.mutate_layer(w, b) for w, b in zip(c2[0], c2[1])]

            c1_weights, c1_biases = zip(*c1_layers)
            c2_weights, c2_biases = zip(*c2_layers)

            offspring.append((list(c1_weights), list(c1_biases)))
            if len(offspring) < pop_size:
                offspring.append((list(c2_weights), list(c2_biases)))

        return offspring[:pop_size]

    def offspring_individuals(self, offspring):
        return [Individual(w, b, self.obs_type, self.norm, self.activation)
                for w, b in offspring]

    # -------------------------------------------------------------------------
    # NSGA-II core  (dominance, fronts, crowding distance)
    # -------------------------------------------------------------------------

    @staticmethod
    def dominates(obj1, obj2):
        """obj1 dominates obj2 (minimisation): ≤ on all, < on at least one."""
        return all(a <= b for a, b in zip(obj1, obj2)) and any(a < b for a, b in zip(obj1, obj2))

    @staticmethod
    def crowding_distance(fitness_list):
        n = len(fitness_list)
        if n == 0:
            return []
        distances = [0.0] * n
        for m in range(len(fitness_list[0])):
            vals = [f[m] for f in fitness_list]
            order = sorted(range(n), key=lambda i: vals[i])
            distances[order[0]] = distances[order[-1]] = float('inf')
            span = max(vals) - min(vals) or 1.0
            for i in range(1, n - 1):
                distances[order[i]] += (vals[order[i+1]] - vals[order[i-1]]) / span
        return distances

    def non_dominated_sort(self, fitness):
        n = len(fitness)
        dominates_set = [[] for _ in range(n)]
        dominated_by  = [0]  * n
        fronts = [[]]
        for p in range(n):
            for q in range(n):
                if self.dominates(fitness[p], fitness[q]):
                    dominates_set[p].append(q)
                elif self.dominates(fitness[q], fitness[p]):
                    dominated_by[p] += 1
            if dominated_by[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominates_set[p]:
                    dominated_by[q] -= 1
                    if dominated_by[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        fronts.pop()
        return fronts

    # kept for backward compat
    def tournament_selection(self, ranked_pop, tournament_size=2):
        return self._tournament_select(ranked_pop, tournament_size)

    def assign_rank_and_crowding(self, population, fitness):
        return self._assign_rank_and_crowding(population, fitness)

    # -------------------------------------------------------------------------
    # Feasibility filter  (min_scores in config["training"])
    # -------------------------------------------------------------------------

    def _is_feasible(self, fitness_tuple):
        """
        Return True if all active min_scores thresholds are satisfied.
        Objectives are minimised, so threshold = maximum allowed value.
        """
        min_scores = self.config["training"].get("min_scores", None)
        if not min_scores:
            return True
        return all(
            threshold is None or obj_val <= threshold
            for obj_val, threshold in zip(fitness_tuple[:3], min_scores)
        )

    def _constraint_violation(self, fitness_tuple):
        """Scaled sum of violations across objectives (0 if feasible)."""
        min_scores = self.config["training"].get("min_scores", None)
        if not min_scores:
            return 0.0
        total = 0.0
        for obj_val, threshold in zip(fitness_tuple[:3], min_scores):
            if threshold is not None and obj_val > threshold:
                total += (obj_val - threshold) / threshold  # normalised by threshold
        return total

    # -------------------------------------------------------------------------
    # Selection  (combine parents + offspring, filter, sort, trim to N)
    # -------------------------------------------------------------------------

    def select_from_combined(self, parent_fitness, offspring, offspring_fitness):
        """
        NSGA-II μ+λ selection:
          1. Combine parents and offspring into a pool of 2N.
          2. Discard infeasible individuals (those exceeding min_scores).
          3. Fill N slots via non-dominated sorting + crowding distance.
          4. If fewer than N feasible individuals exist, backfill with the
             least-violating infeasible ones.
        """
        pop_size = len(self.population)

        # -- Build combined pool -----------------------------------------------
        combined_fitness   = [tuple(f) for f in list(parent_fitness) + list(offspring_fitness)]
        combined_selection = [f[:3] for f in combined_fitness]   # first 3 objectives for sorting
        combined_pop       = self.population + offspring

        # -- Feasibility partition ---------------------------------------------
        feasible_idx   = [i for i, f in enumerate(combined_fitness) if     self._is_feasible(f)]
        infeasible_idx = [i for i, f in enumerate(combined_fitness) if not self._is_feasible(f)]

        if infeasible_idx:
            print(f"[NSGA-II] {len(infeasible_idx)}/{len(combined_pop)} individuals "
                  f"disqualified by min_scores.")

        # -- NSGA-II selection on feasible pool --------------------------------
        new_pop     = []
        new_fitness = []

        if feasible_idx:
            f_pop  = [combined_pop[i]       for i in feasible_idx]
            f_sel  = [combined_selection[i] for i in feasible_idx]
            f_full = [combined_fitness[i]   for i in feasible_idx]

            for front in self.non_dominated_sort(f_sel):
                if len(new_pop) + len(front) <= pop_size:
                    for idx in front:
                        new_pop.append(f_pop[idx])
                        new_fitness.append(f_full[idx])
                else:
                    distances   = self.crowding_distance([f_sel[i] for i in front])
                    sorted_front = sorted(zip(front, distances), key=lambda x: -x[1])
                    for idx, _ in sorted_front:
                        if len(new_pop) >= pop_size:
                            break
                        new_pop.append(f_pop[idx])
                        new_fitness.append(f_full[idx])
                    break

        # -- Backfill with least-violating infeasible individuals --------------
        if len(new_pop) < pop_size and infeasible_idx:
            sorted_inf = sorted(infeasible_idx,
                                key=lambda i: self._constraint_violation(combined_fitness[i]))
            for idx in sorted_inf:
                if len(new_pop) >= pop_size:
                    break
                new_pop.append(combined_pop[idx])
                new_fitness.append(combined_fitness[idx])

        # -- Update state ------------------------------------------------------
        self.population      = new_pop
        self._cached_fitness = new_fitness
        return new_fitness

    # -------------------------------------------------------------------------
    # Checkpoint  (save / load)
    # -------------------------------------------------------------------------

    def save(self, path):
        if path.endswith('.pt'):
            path = path[:-3] + '.npz'

        population = self.population
        cached     = getattr(self, '_cached_fitness', None)

        if self.config.get("training", {}).get("save_pareto_only", True) and cached is not None:
            fronts         = self.non_dominated_sort([tuple(f[:3]) for f in cached])
            pareto_idx     = fronts[0]
            population     = [self.population[i] for i in pareto_idx]
            cached         = [cached[i]          for i in pareto_idx]
            print(f"[Checkpoint] Saving {len(population)}/{len(self.population)} individuals (Pareto front only)")

        save_dict = {'norm': self.norm, 'n_individuals': len(population), 'n_layers': self.n_layers}
        for i, (weights, biases) in enumerate(population):
            for j, w in enumerate(weights): save_dict[f'ind_{i}_weight_{j}'] = w
            for j, b in enumerate(biases):  save_dict[f'ind_{i}_bias_{j}']   = b
        if cached is not None:
            save_dict['fitness'] = np.array(cached)
        np.savez_compressed(path, **save_dict)

    def load(self, path):
        if path.endswith('.pt'):
            path = path[:-3] + '.npz'
        data         = np.load(path)
        self.norm    = data['norm']
        n_ind        = int(data['n_individuals'])
        n_lay        = int(data['n_layers'])
        self.population = [
            ([data[f'ind_{i}_weight_{j}'] for j in range(n_lay)],
             [data[f'ind_{i}_bias_{j}']   for j in range(n_lay)])
            for i in range(n_ind)
        ]
        self._cached_fitness = data['fitness'].tolist() if 'fitness' in data else None
