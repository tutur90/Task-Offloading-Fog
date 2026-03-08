import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
from collections import deque

import numpy as np

from core.env import Env
from core.task import Task

from policies.model.base_model import BaseModel, scaled_lr



class DQNPolicy:
    def __init__(self, env: Env, config, dataset=None):
        """
        A simple deep Q-learning policy.

        Args:
            env: The simulation environment.
            config (dict): A configuration dictionary containing:
                - training: with keys 'lr', 'gamma', 'epsilon'
                - model: with key 'd_model' (used as the hidden size)
        """
        self.env = env
        self._init_obs_space(env, config)
        self._init_device(config)
        self._init_training_params(config)
        self._init_exploration(config)
        self._init_reward_params(config)
        self._init_model(env, config, dataset=dataset)

        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()

        self._init_optimizer(config)

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_obs_space(self, env: Env, config):
        """Initialize observation/action space dimensions."""
        self.obs_type = config["model"]["obs_type"]
        obs, _ = self._make_observation(env, None, self.obs_type)
        self.n_observations = len(obs)
        self.d_obs = len(obs[0])
        self.num_actions = len(env.scenario.node_id2name)

    def _init_device(self, config):
        """Initialize compute device and dtype."""
        if config["device"] == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config["device"])
        print(f"Using device: {self.device}")
        self.dtype = torch.float32

    def _init_training_params(self, config):
        """Initialize core training hyperparameters and replay buffer."""
        tr = config["training"]
        self.gamma = tr["gamma"]
        self.lr = scaled_lr(
            tr["lr"],
            d_model=config.get("model", {}).get("d_model", 1),
            num_layers=config.get("model", {}).get("n_layers", 1),
            ref_d_model=tr.get("ref_d_model"),
            ref_n_layers=tr.get("ref_n_layers"),
        )
        self.buffer_size = tr.get("buffer_size", 10000)
        self.batch_size = tr.get("batch_size", 64)
        self.target_update_freq = tr.get("target_update_freq", 1000)
        # target_update_freq >= 1 → hard update every N gradient steps
        # target_update_freq <  1 → treated as τ for soft (Polyak) update every gradient step
        self.soft_update = self.target_update_freq < 1
        self.tau = self.target_update_freq if self.soft_update else None
        self.update_freq = tr.get("update_freq", 1)
        self.learning_starts = tr.get("learning_starts", 0)
        self.warmup_ratio = tr.get("warmup", 0)
        self.warmup_steps = 0       # computed in set_training_steps()
        self.total_training_steps = None  # set by set_training_steps()
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.double_dqn = tr.get("double_dqn", False)
        self.update_count = 0
        self.total_steps = 0
        self.action_counts = np.zeros(self.num_actions, dtype=np.float32)  # for UCB
        self.clip_grad_norm = tr.get("clip_grad_norm", float('inf'))
        if self.clip_grad_norm <= 0:
            self.clip_grad_norm = float('inf')

    def _init_exploration(self, config):
        """Initialize exploration strategy parameters."""
        tr = config["training"]
        tr["exploration"] = tr.get("exploration", {})
        tr["exploration"].setdefault("strategy", "epsilon_greedy")
        self.exploration_strategy = tr["exploration"]["strategy"]

        _expl = tr["exploration"]
        _DEFAULTS = {
            "epsilon_greedy":    (1.0, 0.01, 0.3),
            "boltzmann":         (1.0, 0.1,  0.5),
            "boltzmann_gumbel":  (1.0, 0.1,  0.5),  # Gumbel-max trick ≡ Boltzmann sampling
            "parameter_noise":   (1.0, 0.01, 0.5),
            "ucb":               (1.0, None, 1.0),   # min=None → defaults to value (no decay)
        }

        if self.exploration_strategy == "thompson":
            self.thompson_n_samples = _expl.get("n_samples", 1)
        elif self.exploration_strategy == "noisy_net":
            self.noisy_sigma_init = _expl.get("sigma_init", 0.5)
        elif self.exploration_strategy in _DEFAULTS:
            _dv, _dmin, _ddecay = _DEFAULTS[self.exploration_strategy]
            self.explore_start = _expl.get("explore_value", _dv)
            self.explore_min   = _expl.get("explore_min",   _dmin if _dmin is not None else self.explore_start)
            self.explore_decay = _expl.get("explore_decay", _ddecay)
            if self.exploration_strategy == "ucb":
                self.ucb_count_decay = _expl.get("count_decay", 1.0)  # γ ∈ (0,1]; 1.0 = no decay
        else:
            raise ValueError(f"Unknown exploration strategy: {self.exploration_strategy}")

        if self.exploration_strategy not in ("thompson", "noisy_net"):
            self.explore_decay_type = _expl.get("decay_type", "linear")
            self.explore_value = self.explore_start
            self.explore_ema_alpha = None  # computed in set_training_steps(), only used for 'exp'

    def _init_reward_params(self, config):
        """Initialize reward normalization parameters."""
        _reward = config["training"].get("reward", {})
        self.reward_momentum = _reward.get("momentum") or 0.9
        self.reward_norm = _reward.get("norm", "standard")
        self.reward_eps = _reward.get("eps", 1e-6)
        self.reward_mean = config.get("eval", {}).get("lambda", [1.0] * 3)
        self.reward_var = [1.0] * 3
        self.reward_max = config.get("eval", {}).get("lambda", [1.0] * 3)
        self.avg_reward = 0
        self.reward_log1p = _reward.get("log1p", True)
        self.expected_reward = config.get("eval", {}).get("lambda", None)
        self.reward_patch = _reward.get("patch", 0.01)
        self.reward_clip = _reward.get("clip", 5.0)
        self.reward_storage_norm = _reward.get("storage_norm", False)

    def _get_param_groups(self, weight_decay):
        """Split parameters into decay / no-decay groups.

        Excluded from weight decay:
        - ndim == 1: RMSNorm/LayerNorm scale parameters
        - "bias" in name: regular biases AND ParallelLinear biases (2-D)
        - "norm" in name: explicit guard for any norm weight regardless of shape
        - "embedding" in name: positional / token embeddings (2-D, should not shrink)
        """
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or "bias" in name or "norm" in name or "embedding" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def _init_optimizer(self, config):
        """Initialize optimizer, loss criterion and lambda weights."""
        config["training"]["optimizer"] = config["training"].get("optimizer", {})
        opt_cfg = config["training"]["optimizer"]
        opt_type = opt_cfg.get("type", "Adam")
        weight_decay = opt_cfg.get("weight_decay", 0)

        param_groups = self._get_param_groups(weight_decay)

        if opt_type == "Adam":
            self.optimizer = optim.AdamW(param_groups, lr=self.lr, betas=(0.9, 0.999))
        elif opt_type == "SGD":
            self.optimizer = optim.SGD(param_groups, lr=self.lr)
        elif opt_type == "RMSprop":
            self.optimizer = optim.RMSprop(param_groups, lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

        self.criterion = nn.SmoothL1Loss()
        self.avg_loss = 0
        self.avg_grad_norm = 0

        _lambda = config.get("training", {}).get("lambda", [1.0, 1.0, 1.0])
        self._lambda = np.array(_lambda) / np.sum(_lambda)

    # ------------------------------------------------------------------

    def _init_model(self, env: Env, config, dataset=None):
        
        self.model = BaseModel()  # placeholder, should be overridden by subclass

        self.model.register_norm(self._make_observation(env, None, self.obs_type)[0], dataset=dataset)

    def norm_reward(self, reward, _lambda):
        reward = self._norm_reward(reward, _lambda)
        self.avg_reward = sum(self._lambda[i] * reward[i] if reward[i] is not None else 0 for i in range(3))
        return reward

    def adapt_coef(self, reward, _lambda):
        pass

    def _norm_reward_fn(self, reward):
        if self.reward_norm == "standard":
            r = [(reward[i] - self.reward_mean[i]) / (np.sqrt(self.reward_var[i]) + self.reward_eps) if reward[i] else 0 for i in range(3)]
        elif self.reward_norm == "partial_standard":
            r = [(reward[i] - self.reward_mean[i]) / (np.sqrt(self.reward_var[i]) + self.reward_eps) if reward[i] and i != 0 else 0 for i in range(3)]
        elif self.reward_norm == "mean":
            r = [reward[i] / (self.reward_mean[i] + self.reward_eps) if reward[i] else 0 for i in range(3)]
        elif self.reward_norm == "partial_mean":
            r = [reward[i] / (self.reward_mean[i] + self.reward_eps) if reward[i] and i != 0 else 0 for i in range(3)]
        elif self.reward_norm == "max":
            self.reward_max = [max(self.reward_max[i], reward[i]) if reward[i] else self.reward_max[i] for i in range(3)]
            r = [reward[i] / (self.reward_max[i] + self.reward_eps) if reward[i] else 0 for i in range(3)]
        elif self.reward_norm == "log1p_mean":
            r = [np.log1p(reward[i]) - (np.log1p(self.reward_mean[i]) + self.reward_eps) if reward[i] else 0 for i in range(3)]
        elif self.reward_norm == "expected":
            r = [reward[i] / (self.reward_mean[i] + self.reward_eps) if reward[i] else 0 for i in range(3)]
        elif self.reward_norm == "none":
            r = [reward[i] if reward[i] is not None else 0 for i in range(3)]
        else:
            raise ValueError(f"Unknown reward normalization method: {self.reward_norm}")

        if self.reward_patch is not None:
            patch = [self.reward_patch * max(reward[i] / (self.expected_reward[i] + 0.1) - 1, 0) if reward[i] else 0 for i in range(3)]
            r = [r[i] + patch[i] for i in range(3)]

        return r
    
    def _norm_reward(self, reward, _lambda):
        
        if self.reward_log1p:
            if reward[1] is not None:
                reward[1] = np.log1p(reward[1])
            if reward[2] is not None:
                reward[2] = np.log1p(reward[2])

        self.reward_mean = [self.reward_mean[i] * self.reward_momentum + reward[i] * (1 - self.reward_momentum) if reward[i] else self.reward_mean[i] for i in range(3)]
        self.reward_var  = [self.reward_var[i]  * self.reward_momentum + (reward[i] - self.reward_mean[i]) ** 2 * (1 - self.reward_momentum) if reward[i] else self.reward_var[i]  for i in range(3)]
        self.reward_max  = [max(self.reward_max[i], reward[i]) if reward[i] else self.reward_max[i] for i in range(3)]

        if self.reward_storage_norm:
            reward = self._norm_reward_fn(reward)

        return reward

    def _make_observation(self, env: Env, task: Task, obs_type=["cpu", "buffer", "bw"]):
        """Returns a flat observation vector."""
        obs_type = list(set(obs_type) & {"cpu", "buffer", "bw"})

        obs = np.zeros((len(env.scenario.get_nodes()), len(obs_type)), dtype=np.float32)

        for i, node_name in enumerate(env.scenario.get_nodes()):
            if "cpu" in obs_type:
                obs[env.scenario.node_name2id[node_name], obs_type.index("cpu")] = env.scenario.get_node(node_name).free_cpu_freq
            if "buffer" in obs_type:
                obs[env.scenario.node_name2id[node_name], obs_type.index("buffer")] = env.scenario.get_node(node_name).buffer_free_size()
            if "bw" in obs_type:
                src_node = "e0"
                if node_name != src_node:
                    obs[env.scenario.node_name2id[node_name], obs_type.index("bw")] = min(link.free_bandwidth for link in env.scenario.infrastructure.get_shortest_links(src_node, node_name))
                else:
                    obs[env.scenario.node_name2id[node_name], obs_type.index("bw")] = max(link.free_bandwidth for link in env.scenario.infrastructure.get_links().values())

        if task is None:
            task_obs = np.zeros(4, dtype=np.float32)
        else:
            task_obs = np.array([
                task.task_size,
                task.cycles_per_bit,
                task.trans_bit_rate,
                task.ddl,
            ], dtype=np.float32)

        return obs, task_obs

    def set_training_steps(self, total_steps):
        """Set total training steps for exploration schedule."""
        self.total_training_steps = total_steps
        self.warmup_steps = int(total_steps * self.warmup_ratio)
        if self.exploration_strategy not in ("thompson", "noisy_net") and self.explore_decay_type == "exp" and self.explore_min < self.explore_start:
            decay_steps = max(1, int(total_steps * self.explore_decay))
            # EMA alpha s.t. explore_start * alpha^decay_steps = explore_min
            self.explore_ema_alpha = (self.explore_min / self.explore_start) ** (1.0 / decay_steps)

    def _update_explore(self):
        """Decay the exploration parameter over training (linear or exp)."""
        if self.total_training_steps is None or self.exploration_strategy in ("thompson", "noisy_net"):
            return
        steps_since_learn = self.total_steps - self.learning_starts
        if steps_since_learn <= 0:
            self.explore_value = self.explore_start
        elif self.explore_decay_type == "exp" and self.explore_ema_alpha is not None:
            self.explore_value = max(self.explore_value * self.explore_ema_alpha, self.explore_min)
        else:
            decay_steps = int(self.total_training_steps * self.explore_decay)
            if steps_since_learn >= decay_steps:
                self.explore_value = self.explore_min
            else:
                self.explore_value = self.explore_start - (self.explore_start - self.explore_min) * (steps_since_learn / decay_steps)

    def _update_lr(self):
        """Linear LR warmup from 0 to base lr over warmup_steps steps after learning starts."""
        if self.warmup_steps == 0:
            return
        steps_since_learn = self.total_steps - self.learning_starts
        if steps_since_learn <= 0:
            lr = 0.0
        elif steps_since_learn < self.warmup_steps:
            lr = self.lr * steps_since_learn / self.warmup_steps
        else:
            return  # warmup complete
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def act(self, env, task, train=True):
        """
        Chooses an action using the configured exploration strategy and records the current state.
        Supports: "epsilon_greedy", "boltzmann", "ucb".
        """
        state = self._make_observation(env, task, self.obs_type)
        obs, task_obs = state
        obs_tensor  = torch.tensor(obs,      dtype=self.dtype, device=self.device).unsqueeze(0)
        task_tensor = torch.tensor(task_obs, dtype=self.dtype, device=self.device).unsqueeze(0)

        if train:
            self.total_steps += 1
            self._update_explore()

        # Random warm-up phase regardless of strategy
        if train and self.total_steps <= self.learning_starts:
            return random.randrange(self.num_actions), state

        if self.exploration_strategy == "epsilon_greedy":
            if train and random.random() < self.explore_value:
                action = random.randrange(self.num_actions)
            else:
                with torch.no_grad():
                    self.model.eval()
                    q_values = self.model(obs_tensor, task_tensor)
                    action = torch.argmax(q_values, dim=1).item()

        elif self.exploration_strategy == "boltzmann":
            with torch.no_grad():
                self.model.eval()
                q_values = self.model(obs_tensor, task_tensor).squeeze()
            if train:
                probs = torch.softmax(q_values / self.explore_value, dim=0).cpu().numpy()
                action = int(np.random.choice(self.num_actions, p=probs))
            else:
                action = int(torch.argmax(q_values).item())

        elif self.exploration_strategy == "boltzmann_gumbel":
            # Boltzmann-Gumbel Exploration (Ciosek & Whiteson, NeurIPS 2017).
            #
            # The Gumbel-max trick proves:
            #   argmax_a( Q(s,a)/T + G_a ),  G_a ~ Gumbel(0,1) i.i.d.
            # is *exactly* distributed as sampling from Boltzmann(Q, T).
            #
            # The asymmetric (right-skewed) Gumbel noise provides natural optimism:
            # actions whose Q-values are uncertain receive a positive bonus on average,
            # unlike symmetric noise (e.g. Gaussian parameter noise).
            #
            # Sampling Gumbel(0,1) via the inverse-CDF:
            #   G = -log(-log(U)),  U ~ Uniform(0,1)
            with torch.no_grad():
                self.model.eval()
                q_values = self.model(obs_tensor, task_tensor).squeeze()
            if train:
                u = torch.clamp(torch.rand_like(q_values), min=1e-20, max=1.0 - 1e-20)
                gumbel_noise = -torch.log(-torch.log(u))          # G ~ Gumbel(0,1)
                perturbed = q_values / self.explore_value + gumbel_noise
                action = int(torch.argmax(perturbed).item())
            else:
                action = int(torch.argmax(q_values).item())

        elif self.exploration_strategy == "thompson":
            if train:
                self.model.train()
                with torch.no_grad():
                    if self.thompson_n_samples > 1:
                        q_values = torch.stack([
                            self.model(obs_tensor, task_tensor).squeeze()
                            for _ in range(self.thompson_n_samples)
                        ]).mean(dim=0)
                    else:
                        q_values = self.model(obs_tensor, task_tensor).squeeze()
            else:
                with torch.no_grad():
                    self.model.eval()
                    q_values = self.model(obs_tensor, task_tensor).squeeze()
            action = int(torch.argmax(q_values).item())

        elif self.exploration_strategy == "noisy_net":
            if train:
                self.model.train()
                self.model.reset_noise()
                with torch.no_grad():
                    q_values = self.model(obs_tensor, task_tensor).squeeze()
            else:
                with torch.no_grad():
                    self.model.eval()
                    q_values = self.model(obs_tensor, task_tensor).squeeze()
            action = int(torch.argmax(q_values).item())

        elif self.exploration_strategy == "parameter_noise":
            with torch.no_grad():
                self.model.eval()
                q_values = self.model(obs_tensor, task_tensor).squeeze()
            if train:
                noise = torch.randn_like(q_values) * self.explore_value
                action = int(torch.argmax(q_values + noise).item())
            else:
                action = int(torch.argmax(q_values).item())

        elif self.exploration_strategy == "ucb":
            with torch.no_grad():
                self.model.eval()
                q_values = self.model(obs_tensor, task_tensor).squeeze().cpu().numpy()
            if train:
                self.action_counts *= self.ucb_count_decay
                bonus = self.explore_value * np.sqrt(np.log1p(np.sum(self.action_counts)) / (1 + self.action_counts))
                action = int(np.argmax(q_values + bonus))
                self.action_counts[action] += 1
            else:
                action = int(np.argmax(q_values))

        else:
            raise ValueError(f"Unknown exploration strategy: '{self.exploration_strategy}'. "
                             f"Choose from: 'epsilon_greedy', 'boltzmann', 'boltzmann_gumbel', 'thompson', 'parameter_noise', 'noisy_net', 'ucb'.")

        return action, state

    def store_transition(self, state, action, reward, next_state, done):
        """Stores a transition in the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def aggregate_reward(self, rewards):
        """Aggregates multiple reward components into a single scalar using lambda weights."""
        total_reward = []
        for r in rewards:
            if not self.reward_storage_norm:
                r = self._norm_reward_fn(r)
            if self.reward_clip is not None:
                r = np.clip(r, -self.reward_clip, self.reward_clip)
            total_reward.append(-sum(self._lambda[i] * r[i] for i in range(3)))
        return total_reward

    def _update(self):
        """
        Performs an update over a sampled batch of transitions using batched operations,
        moves tensors to the appropriate device and dtype.
        """
        self._update_lr()

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        obs_batch,      task_obs_batch      = zip(*states)
        next_obs_batch, next_task_obs_batch = zip(*next_states)

        rewards = self.aggregate_reward(rewards)

        obs_tensor       = torch.tensor(np.array(obs_batch),           dtype=self.dtype,       device=self.device)
        task_tensor      = torch.tensor(np.array(task_obs_batch),      dtype=self.dtype,       device=self.device)
        next_obs_tensor  = torch.tensor(np.array(next_obs_batch),      dtype=self.dtype,       device=self.device)
        next_task_tensor = torch.tensor(np.array(next_task_obs_batch), dtype=self.dtype,       device=self.device)
        actions_tensor   = torch.tensor(np.array(actions),             dtype=torch.int64,      device=self.device).unsqueeze(-1)
        rewards_tensor   = torch.tensor(rewards,                        dtype=self.dtype,       device=self.device)
        dones_tensor     = torch.tensor(dones,                          dtype=self.dtype,       device=self.device)

        self.optimizer.zero_grad()

        self.model.train()
        if self.exploration_strategy == "noisy_net":
            self.model.reset_noise()
        q_values    = self.model(obs_tensor, task_tensor).squeeze()
        predicted_q = q_values.gather(1, actions_tensor).squeeze()

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.model(next_obs_tensor, next_task_tensor).argmax(dim=1, keepdim=True)
                max_next_q   = self.target_model(next_obs_tensor, next_task_tensor).gather(1, next_actions).squeeze()
            else:
                next_q_values = self.target_model(next_obs_tensor, next_task_tensor).squeeze()
                max_next_q, _ = torch.max(next_q_values, dim=1)
            target_q = rewards_tensor if self.gamma == 0 else rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q

        loss = self.criterion(predicted_q, target_q)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
        self.optimizer.step()

        return loss.item(), grad_norm.item()

    def update(self, metric_momentum=0.99995):
        """
        Performs an update over a sampled batch of transitions using batched operations,
        moves tensors to the appropriate device and dtype.
        """
        self.update_count += 1
        if len(self.replay_buffer) < self.batch_size or self.total_steps <= self.learning_starts:
            return 0.0, None

        if not self.soft_update and self.update_count % self.target_update_freq == 0:
            self.update_target_network()

        if self.update_count % self.update_freq == 0:
            loss, grad_norm = self._update()
            if self.soft_update:
                self.update_target_network()
            self.avg_loss = self.avg_loss * metric_momentum + loss * (1 - metric_momentum)
            self.avg_grad_norm = self.avg_grad_norm * metric_momentum + grad_norm * (1 - metric_momentum) 
            return loss, grad_norm

    def update_target_network(self):
        """
        Hard update (target_update_freq >= 1): θ_target ← θ_online every N gradient steps.
        Soft update (target_update_freq < 1):  θ_target ← τ·θ_online + (1−τ)·θ_target every gradient step.
        """
        if self.soft_update:
            with torch.no_grad():
                for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                    target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
        else:
            self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path):
        """Saves the model to the specified path."""
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Loads the model from the specified path."""
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.model.state_dict())
        self.model.eval()
        self.target_model.eval()
