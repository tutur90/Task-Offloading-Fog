import logging
import random
import numpy as np
from core.env import Env
from core.task import Task

logger = logging.getLogger(__name__)


# =============================================================================
# Individual — inference only, no training logic
# =============================================================================

class Individual:
    def __init__(self, weights, biases, obs_type=["cpu", "buffer", "bw"], norm=None, activation='relu'):
        self.weights    = weights
        self.biases     = biases
        self.obs_type   = obs_type
        self.norm       = norm
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
            if "cpu"    in obs_type: obs[node_id, obs_type.index("cpu")]    = env.scenario.get_node(node_name).free_cpu_freq
            if "buffer" in obs_type: obs[node_id, obs_type.index("buffer")] = env.scenario.get_node(node_name).buffer_free_size()
            if "bw"     in obs_type:
                src = "e0"
                if node_name != src:
                    obs[node_id, obs_type.index("bw")] = min(l.free_bandwidth for l in env.scenario.infrastructure.get_shortest_links(src, node_name))
                else:
                    obs[node_id, obs_type.index("bw")] = max(l.free_bandwidth for l in env.scenario.infrastructure.get_links().values())
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
    """
    NSGA-II policy with three mutation modes (config["training"]["mutation_mode"]):

      "fixed"        σ is fixed per layer: He std (sqrt(2/fan_in)) if mutation_sigma is null,
                     else the configured value.

      "proportional" σ scales with the current weight spread: σ = mutation_sigma * std(W).
                     mutation_sigma acts as a scaling factor (default 0.1).

      "self_adaptive" Each individual carries adaptive step-size scalars (sigma_w, sigma_b)
                     that evolve via log-normal mutation (ES-style):
                       sigma_w' = sigma_w * exp(tau * N(0,1))
                       sigma_b' = sigma_b * exp(tau * N(0,1))
                     mutation_tau controls the meta learning rate (default 0.1).

                     mutation_sigma_scope controls granularity:
                       "individual" (default) — one (sigma_w, sigma_b) pair shared across
                                                all layers; mutated once per individual.
                       "layer"                — one (sigma_w, sigma_b) pair per layer;
                                                each mutated independently.
    """

    MUTATION_MODES = ("fixed", "proportional", "self_adaptive")

    def __init__(self, env, config, dataset=None, logger=None):
        self.config = config
        self.env    = env
        self._train_logger = logger

        self.obs_type   = config["model"]["obs_type"]
        self.d_model    = config["model"]["d_model"]
        self.n_layers   = config["model"]["n_layers"]
        self.activation = config["model"].get("activation", "relu")

        self.mutation_mode  = config["training"].get("mutation_mode", "fixed")
        if self.mutation_mode not in self.MUTATION_MODES:
            raise ValueError(f"mutation_mode must be one of {self.MUTATION_MODES}, got '{self.mutation_mode}'")
        self.sigma_scope    = config["training"].get("mutation_sigma_scope", "individual")
        if self.sigma_scope not in ("individual", "layer"):
            raise ValueError(f"mutation_sigma_scope must be 'individual' or 'layer', got '{self.sigma_scope}'")
        self.mutation_he_scale = config["training"].get("mutation_he_scale", False)

        # Normalisation from initial environment state
        initial_obs         = self._make_observation(env, None, self.obs_type)
        self.norm           = np.where(initial_obs.max(axis=0, keepdims=True) == 0, 1.0,
                                       initial_obs.max(axis=0, keepdims=True))
        self.n_observations = initial_obs.size
        self.num_actions    = len(env.scenario.node_id2name)

        # Population: list of (weights, biases)
        # Sigma factors: parallel list of (sigma_w, sigma_b) per individual — only for "self_adaptive"
        pop_size = config["training"]["pop_size"]
        inds, sigmas        = zip(*[self._generate_individual() for _ in range(pop_size)])
        self.population     = list(inds)
        self._sigma_factors = list(sigmas)   # None per individual when not self_adaptive

    # =========================================================================
    # Observation (used at init for normalisation)
    # =========================================================================

    def _make_observation(self, env, task, obs_type):
        n_nodes = len(env.scenario.get_nodes())
        obs = np.zeros((n_nodes, len(obs_type)), dtype=np.float32)
        for node_name in env.scenario.get_nodes():
            node_id = env.scenario.node_name2id[node_name]
            if "cpu"    in obs_type: obs[node_id, obs_type.index("cpu")]    = env.scenario.get_node(node_name).free_cpu_freq
            if "buffer" in obs_type: obs[node_id, obs_type.index("buffer")] = env.scenario.get_node(node_name).buffer_free_size()
            if "bw"     in obs_type:
                src = "e0"
                if node_name != src:
                    obs[node_id, obs_type.index("bw")] = min(l.free_bandwidth for l in env.scenario.infrastructure.get_shortest_links(src, node_name))
                else:
                    obs[node_id, obs_type.index("bw")] = max(l.free_bandwidth for l in env.scenario.infrastructure.get_links().values())
        return obs

    # =========================================================================
    # Individual initialisation  (He for ReLU, Xavier for tanh/sigmoid)
    # =========================================================================

    def _init_weight(self, fan_in, fan_out):
        if self.activation == 'relu':
            return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))

    def _generate_individual(self):
        """
        Returns ((weights, biases), sigma_factors).
        sigma_factors:
          "individual" scope — (sigma_w, sigma_b) tuple
          "layer"      scope — list of (sigma_w, sigma_b) per layer
          non-self_adaptive  — None
        """
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1.")
        if self.n_layers == 1:
            dims = [(self.n_observations, self.num_actions)]
        else:
            dims = ([(self.n_observations, self.d_model)]
                    + [(self.d_model, self.d_model)] * (self.n_layers - 2)
                    + [(self.d_model, self.num_actions)])

        weights = [self._init_weight(fi, fo) for fi, fo in dims]
        biases  = [np.zeros(fo)              for _, fo  in dims]

        if self.mutation_mode == "self_adaptive":
            s0 = self.config["training"].get("mutation_sigma", 0.1) or 0.1
            if self.mutation_he_scale:
                # Divide out the mean He / bias scale so that sigma * scale ≈ s0 on average
                mean_he_w = float(np.mean([np.sqrt(2.0 / fi) for fi, _ in dims]))
                mean_he_b = float(np.mean([1.0 / np.sqrt(fo) for _, fo in dims]))
                s0_w = s0 / mean_he_w
                s0_b = s0 / mean_he_b
            else:
                s0_w = s0_b = s0
            if self.sigma_scope == "layer":
                sigmas = [(s0_w, s0_b)] * len(dims)
            else:
                sigmas = (s0_w, s0_b)
        else:
            sigmas = None
        return (weights, biases), sigmas

    def individuals(self):
        return [Individual(w, b, self.obs_type, self.norm, self.activation)
                for w, b in self.population]

    # =========================================================================
    # Mutation  —  three modes, dispatched per layer
    # =========================================================================

    def _evolve_layer(self, weight, bias, sigma_w=None, sigma_b=None):
        """
        Mutate one layer (W, b) according to mutation_mode.
        For self_adaptive, sigma_w / sigma_b are the already-mutated individual-level scalars.
        Returns (new_weight, new_bias).
        """
        mode = self.mutation_mode

        if mode == "fixed":
            cfg_sigma = self.config["training"].get("mutation_sigma", None)
            sigma     = cfg_sigma if cfg_sigma is not None else np.sqrt(2.0 / weight.shape[0])
            return (weight + np.random.randn(*weight.shape) * sigma,
                    bias   + np.random.randn(*bias.shape)   * sigma)

        elif mode == "proportional":
            scale   = self.config["training"].get("mutation_sigma", 0.1)
            sigma_w = scale * max(np.std(weight), 1e-8)
            sigma_b = sigma_w
            return (weight + np.random.randn(*weight.shape) * sigma_w,
                    bias   + np.random.randn(*bias.shape)   * sigma_b)

        elif mode == "self_adaptive":
            if self.mutation_he_scale:
                he_scale   = np.sqrt(2.0 / weight.shape[0])
                bias_scale = 1.0 / np.sqrt(weight.shape[1])
            else:
                he_scale = bias_scale = 1.0
            return (weight + np.random.randn(*weight.shape) * sigma_w * he_scale,
                    bias   + np.random.randn(*bias.shape)   * sigma_b * bias_scale)

    # =========================================================================
    # Offspring generation
    # =========================================================================

    def _assign_rank_and_crowding(self, population, fitness):
        """
        Returns list of (idx, individual, fitness, rank, crowding_distance).
        idx is the position in `population`, needed to look up _sigma_factors.
        """
        fronts    = self.non_dominated_sort(fitness)
        ranks     = [0]   * len(population)
        distances = [0.0] * len(population)
        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank
            front_dist = self.crowding_distance([fitness[i] for i in front])
            for i, idx in enumerate(front):
                distances[idx] = front_dist[i]
        return [(i, population[i], fitness[i], ranks[i], distances[i])
                for i in range(len(population))]

    def _tournament_select(self, ranked_pop, tournament_size=2):
        """NSGA-II crowded comparison. Returns (idx, individual)."""
        candidates = random.sample(ranked_pop, tournament_size)
        best = candidates[0]
        for c in candidates[1:]:
            if c[3] < best[3] or (c[3] == best[3] and c[4] > best[4]):
                best = c
        return best[0], best[1]   # (idx, individual)

    def _crossover(self, parent1, parent2):
        """Placeholder — no crossover."""
        return parent1, parent2

    def create_offspring(self, fitness):
        """
        Generate N offspring via tournament selection + _evolve_layer.
        For self_adaptive: sigma is mutated once per individual (log-normal ES step),
        then the same scalar pair is broadcast across all layers.
        Stores evolved offspring sigma factors in self._offspring_sigmas.
        """
        pop_size = len(self.population)
        fitness  = [tuple(f) for f in fitness]
        ranked   = self._assign_rank_and_crowding(self.population, fitness)

        tau = self.config["training"].get("mutation_tau", 0.1)

        offspring        = []
        offspring_sigmas = []

        def _mutate_individual(c, parent_sigma):
            if self.mutation_mode == "self_adaptive":
                if self.sigma_scope == "layer":
                    # Mutate each layer's pair independently
                    new_w, new_b, new_sig = [], [], []
                    for (sw, sb), w, b in zip(parent_sigma, c[0], c[1]):
                        sw_ = max(sw * np.exp(tau * np.random.randn()), 1e-8)
                        sb_ = max(sb * np.exp(tau * np.random.randn()), 1e-8)
                        nw, nb = self._evolve_layer(w, b, sigma_w=sw_, sigma_b=sb_)
                        new_w.append(nw); new_b.append(nb); new_sig.append((sw_, sb_))
                    return (new_w, new_b), new_sig
                else:
                    # Mutate the shared pair once, broadcast to all layers
                    sw = max(parent_sigma[0] * np.exp(tau * np.random.randn()), 1e-8)
                    sb = max(parent_sigma[1] * np.exp(tau * np.random.randn()), 1e-8)
                    pairs = [self._evolve_layer(w, b, sigma_w=sw, sigma_b=sb) for w, b in zip(c[0], c[1])]
                    new_w, new_b = zip(*pairs)
                    return (list(new_w), list(new_b)), (sw, sb)
            else:
                pairs = [self._evolve_layer(w, b) for w, b in zip(c[0], c[1])]
                new_w, new_b = zip(*pairs)
                return (list(new_w), list(new_b)), None

        while len(offspring) < pop_size:
            idx1, p1 = self._tournament_select(ranked)
            idx2, p2 = self._tournament_select(ranked)
            c1, c2   = self._crossover(p1, p2)

            child1, sig1 = _mutate_individual(c1, self._sigma_factors[idx1])
            child2, sig2 = _mutate_individual(c2, self._sigma_factors[idx2])

            offspring.append(child1)
            offspring_sigmas.append(sig1)

            if len(offspring) < pop_size:
                offspring.append(child2)
                offspring_sigmas.append(sig2)

        self._offspring_sigmas = offspring_sigmas[:pop_size]
        return offspring[:pop_size]

    def offspring_individuals(self, offspring):
        return [Individual(w, b, self.obs_type, self.norm, self.activation)
                for w, b in offspring]

    # =========================================================================
    # NSGA-II core  (dominance, fronts, crowding distance)
    # =========================================================================

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
            vals  = [f[m] for f in fitness_list]
            order = sorted(range(n), key=lambda i: vals[i])
            distances[order[0]] = distances[order[-1]] = float('inf')
            span = max(vals) - min(vals) or 1.0
            for i in range(1, n - 1):
                distances[order[i]] += (vals[order[i+1]] - vals[order[i-1]]) / span
        return distances

    def non_dominated_sort(self, fitness):
        n              = len(fitness)
        dominates_set  = [[] for _ in range(n)]
        dominated_by   = [0]  * n
        fronts         = [[]]
        for p in range(n):
            for q in range(n):
                if self.dominates(fitness[p], fitness[q]):   dominates_set[p].append(q)
                elif self.dominates(fitness[q], fitness[p]): dominated_by[p] += 1
            if dominated_by[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in dominates_set[p]:
                    dominated_by[q] -= 1
                    if dominated_by[q] == 0:
                        nxt.append(q)
            i += 1
            fronts.append(nxt)
        fronts.pop()
        return fronts

    # Backward-compat aliases
    def tournament_selection(self, ranked_pop, tournament_size=2):
        _, ind = self._tournament_select(ranked_pop, tournament_size)
        return ind

    def assign_rank_and_crowding(self, population, fitness):
        return self._assign_rank_and_crowding(population, fitness)

    # =========================================================================
    # Feasibility filter  (min_scores in config["training"])
    # =========================================================================

    def _is_feasible(self, fitness_tuple):
        """True if all objectives are within their max-allowed thresholds."""
        min_scores = self.config["training"].get("min_scores", None)
        if not min_scores:
            return True
        return all(t is None or v <= t for v, t in zip(fitness_tuple[:3], min_scores))

    def _constraint_violation(self, fitness_tuple):
        """Sum of normalised violations across objectives (0 if feasible)."""
        min_scores = self.config["training"].get("min_scores", None)
        if not min_scores:
            return 0.0
        return sum(
            (v - t) / t for v, t in zip(fitness_tuple[:3], min_scores)
            if t is not None and v > t
        )

    # =========================================================================
    # Selection  (combine parents + offspring, filter, sort, trim to N)
    # =========================================================================

    def select_from_combined(self, parent_fitness, offspring, offspring_fitness):
        """
        NSGA-II μ+λ selection with feasibility filter.

        Tracks combined pool indices so that _sigma_factors stays in sync
        with self.population after selection.
        """
        pop_size = len(self.population)

        # -- Build combined pool (parents first, then offspring) ---------------
        combined_fitness   = [tuple(f) for f in list(parent_fitness) + list(offspring_fitness)]
        combined_selection = [f[:3] for f in combined_fitness]
        combined_pop       = self.population + offspring
        offspring_sigmas   = getattr(self, '_offspring_sigmas', [None] * len(offspring))
        combined_sigmas    = self._sigma_factors + offspring_sigmas

        # -- Feasibility partition ---------------------------------------------
        feasible_idx   = [i for i, f in enumerate(combined_fitness) if     self._is_feasible(f)]
        infeasible_idx = [i for i, f in enumerate(combined_fitness) if not self._is_feasible(f)]



        # -- NSGA-II selection on feasible pool (track combined indices) -------
        selected_combined = []

        if feasible_idx:
            f_sel = [combined_selection[i] for i in feasible_idx]

            for front in self.non_dominated_sort(f_sel):
                remaining = pop_size - len(selected_combined)
                if len(front) <= remaining:
                    selected_combined.extend(feasible_idx[idx] for idx in front)
                else:
                    distances    = self.crowding_distance([f_sel[i] for i in front])
                    sorted_front = sorted(zip(front, distances), key=lambda x: -x[1])
                    for idx, _ in sorted_front:
                        if len(selected_combined) >= pop_size:
                            break
                        selected_combined.append(feasible_idx[idx])
                    break

        # -- Backfill with least-violating infeasible individuals --------------
        if len(selected_combined) < pop_size and infeasible_idx:
            sorted_inf = sorted(infeasible_idx,
                                key=lambda i: self._constraint_violation(combined_fitness[i]))
            for idx in sorted_inf:
                if len(selected_combined) >= pop_size:
                    break
                selected_combined.append(idx)

        # -- Update state (population and sigma factors stay in sync) ----------
        self.population      = [combined_pop[i]    for i in selected_combined]
        self._sigma_factors  = [combined_sigmas[i] for i in selected_combined]
        self._cached_fitness = [combined_fitness[i] for i in selected_combined]

        n_obj = len(self._cached_fitness[0]) if self._cached_fitness else 0
        avg = [np.mean([f[i] for f in self._cached_fitness]) for i in range(n_obj)]
        n_feasible = sum(1 for f in self._cached_fitness if self._is_feasible(f))
        logger.info(f"[NSGA-II] feasible: {n_feasible}/{len(self._cached_fitness)}")

        # Crowding distance on selected population (objectives 0-2)
        sel_obj = [f[:3] for f in self._cached_fitness]
        cd_all  = []
        for front in self.non_dominated_sort(sel_obj):
            cd_all.extend(self.crowding_distance([sel_obj[i] for i in front]))
        cd_finite = [d for d in cd_all if np.isfinite(d)]

        _metric_names = ['PopAvgDropRate', 'PopAvgLatency', 'PopAvgPower', 'PopAvgScore']
        if self._train_logger is not None:
            for i, val in enumerate(avg):
                name = _metric_names[i] if i < len(_metric_names) else f'PopAvgObj{i}'
                self._train_logger.update_metric(name, val)

            if cd_finite:
                cd_mean = float(np.mean(cd_finite))
                cd_std  = float(np.std(cd_finite))
                self._train_logger.update_metric('CDMean',        cd_mean)
                self._train_logger.update_metric('CDMean+Std',    cd_mean + cd_std)
                self._train_logger.update_metric('CDMean-Std',    cd_mean - cd_std)
                self._train_logger.update_metric('CDMin',         float(np.min(cd_finite)))
                self._train_logger.update_metric('CDStd',         cd_std)

            if self.mutation_mode == "self_adaptive":
                if self.sigma_scope == "layer":
                    all_sigma_w = [sw for s in self._sigma_factors if s is not None for sw, _ in s]
                    all_sigma_b = [sb for s in self._sigma_factors if s is not None for _, sb in s]
                else:
                    all_sigma_w = [s[0] for s in self._sigma_factors if s is not None]
                    all_sigma_b = [s[1] for s in self._sigma_factors if s is not None]
                if all_sigma_w:
                    self._train_logger.update_metric('AvgSigmaW', float(np.mean(all_sigma_w)))
                    self._train_logger.update_metric('AvgSigmaB', float(np.mean(all_sigma_b)))

        return self._cached_fitness

    # =========================================================================
    # Checkpoint  (save / load)
    # =========================================================================

    def save(self, path):
        if path.endswith('.pt'):
            path = path[:-3] + '.npz'

        population = self.population
        sigmas     = self._sigma_factors
        cached     = getattr(self, '_cached_fitness', None)

        if self.config.get("training", {}).get("save_pareto_only", True) and cached is not None:
            fronts     = self.non_dominated_sort([tuple(f[:3]) for f in cached])
            pareto_idx = fronts[0]
            population = [self.population[i]     for i in pareto_idx]
            sigmas     = [self._sigma_factors[i] for i in pareto_idx]
            cached     = [cached[i]              for i in pareto_idx]
            logger.info(f"[Checkpoint] Saving {len(population)}/{len(self.population)} individuals (Pareto front only)")

        save_dict = {
            'norm': self.norm, 'n_individuals': len(population), 'n_layers': self.n_layers,
            'mutation_mode': self.mutation_mode,
        }
        for i, (weights, biases) in enumerate(population):
            for j, w in enumerate(weights): save_dict[f'ind_{i}_w{j}'] = w
            for j, b in enumerate(biases):  save_dict[f'ind_{i}_b{j}'] = b
            if sigmas[i] is not None:
                if self.sigma_scope == "layer":
                    save_dict[f'ind_{i}_sigma_w'] = np.array([sw for sw, _ in sigmas[i]])
                    save_dict[f'ind_{i}_sigma_b'] = np.array([sb for _, sb in sigmas[i]])
                else:
                    save_dict[f'ind_{i}_sigma_w'] = np.float64(sigmas[i][0])
                    save_dict[f'ind_{i}_sigma_b'] = np.float64(sigmas[i][1])
        if cached is not None:
            save_dict['fitness'] = np.array(cached)
        np.savez_compressed(path, **save_dict)

    def load(self, path):
        if path.endswith('.pt'):
            path = path[:-3] + '.npz'
        data    = np.load(path, allow_pickle=True)
        self.norm = data['norm']
        n_ind   = int(data['n_individuals'])
        n_lay   = int(data['n_layers'])

        self.population     = []
        self._sigma_factors = []
        for i in range(n_ind):
            # Support both old key format (ind_i_weight_j) and new (ind_i_wj)
            try:
                weights = [data[f'ind_{i}_w{j}']      for j in range(n_lay)]
                biases  = [data[f'ind_{i}_b{j}']      for j in range(n_lay)]
            except KeyError:
                weights = [data[f'ind_{i}_weight_{j}'] for j in range(n_lay)]
                biases  = [data[f'ind_{i}_bias_{j}']   for j in range(n_lay)]
            self.population.append((weights, biases))

            if f'ind_{i}_sigma_w' in data:
                sw_raw = np.asarray(data[f'ind_{i}_sigma_w'])
                sb_raw = np.asarray(data[f'ind_{i}_sigma_b'])
                if self.sigma_scope == "layer" and sw_raw.ndim == 1:
                    # Per-layer array → list of tuples
                    self._sigma_factors.append(list(zip(sw_raw.tolist(), sb_raw.tolist())))
                elif self.sigma_scope == "layer" and sw_raw.ndim == 0:
                    # Checkpoint was individual-scope; broadcast scalar to all layers
                    s = (float(sw_raw), float(sb_raw))
                    self._sigma_factors.append([s] * n_lay)
                elif self.sigma_scope == "individual" and sw_raw.ndim == 1:
                    # Checkpoint was layer-scope; collapse to scalar, reinit sigma_b from sigma_w
                    sigma_w = float(np.mean(sw_raw))
                    self._sigma_factors.append((sigma_w, sigma_w))
                else:
                    # Both scalar — direct load
                    self._sigma_factors.append((float(sw_raw), float(sb_raw)))
            elif f'ind_{i}_sigma_fw' in data:
                # Legacy key names — collapse to scalar
                sigma_w = float(np.mean(data[f'ind_{i}_sigma_fw']))
                self._sigma_factors.append((sigma_w, sigma_w) if self.sigma_scope == "individual"
                                           else [(sigma_w, sigma_w)] * n_lay)
            else:
                self._sigma_factors.append(None)

        self._cached_fitness = data['fitness'].tolist() if 'fitness' in data else None
