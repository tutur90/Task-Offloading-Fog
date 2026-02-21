import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from core.env import Env
from core.task import Task
from policies.model.tpto import TPTOModel


class TPTOPolicy:
    """
    PPO-based task offloading policy using a Transformer Actor-Critic.

    Adapted from:
        Gholipour et al., "TPTO: A Transformer-PPO based Task Offloading Solution
        for Edge Computing Environments", arXiv:2312.11739.

    Differences from the paper:
        - Action = selected node (n-way discrete) instead of binary local/offload.
        - Reward = existing multi-objective reward (TTR + latency + energy).
        - State  = existing node-resource observation [cpu, bw, buffer] per node.
    """

    def __init__(self, env: Env, config: dict, dataset=None, device="auto"):
        self.env = env

        # ---------- observation setup ----------
        self.obs_type = config["model"]["obs_type"]
        obs, task_obs = self._make_observation(env, None, self.obs_type)
        self.n_observations = len(obs)        # number of nodes
        self.d_obs = len(obs[0])              # features per node
        self.num_actions = len(env.scenario.node_id2name)

        # ---------- device ----------
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[TPTO] Using device: {self.device}")
        self.dtype = torch.float32

        # ---------- training hypers ----------
        tr = config["training"]
        self.gamma = tr.get("gamma", 0.99)
        self.gae_lambda = tr.get("gae_lambda", 0.95)
        self.clip_eps = tr.get("clip_eps", 0.2)
        self.entropy_coef = tr.get("entropy_coef", 0.01)
        self.value_loss_coef = tr.get("value_loss_coef", 0.5)
        self.clip_grad_norm = tr.get("clip_grad_norm", 0.5)
        if self.clip_grad_norm <= 0:
            self.clip_grad_norm = float("inf")
        self.ppo_epochs = tr.get("ppo_epochs", 4)
        self.mini_batch_size = tr.get("mini_batch_size", 64)
        self.lr = tr.get("lr", 3e-4)

        # ---------- reward normalization (mirrors DQNPolicy) ----------
        _reward = tr.get("reward", {})
        self.reward_momentum = _reward.get("momentum", 0.9)
        self.reward_norm = _reward.get("norm", "standard")
        self.reward_eps = _reward.get("eps", 1e-6)
        self.reward_mean = list(config.get("eval", {}).get("expected_values", [1.0] * 3))
        self.reward_var = [1.0] * 3
        self.reward_max = list(config.get("eval", {}).get("expected_values", [1.0] * 3))
        self.avg_reward = 0.0
        if "ln" in self.reward_norm:
            self.reward_mean = [np.log(v + self.reward_eps) for v in self.reward_mean]

        # ---------- model ----------
        self.model = TPTOModel(
            d_in=self.d_obs,
            d_pos=self.n_observations,
            d_task=4,
            **config["model"],
        )
        self.model.register_norm(obs, dataset=dataset)
        self.model.to(self.device).to(self.dtype)

        # ---------- optimizer ----------
        opt_cfg = tr.get("optimizer", {})
        opt_type = opt_cfg.get("type", "Adam")
        wd = opt_cfg.get("weight_decay", 0.0)
        if opt_type == "Adam":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=wd)
        elif opt_type == "Adagrad":
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.lr, weight_decay=wd)
        elif opt_type == "RMSprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        # ---------- stats ----------
        self.avg_loss = 0.0
        self.avg_grad_norm = 0.0
        self.total_steps = 0
        self.update_count = 0

        # For compatibility with main.py set_training_steps call
        self.total_training_steps = None

    def set_training_steps(self, total_steps: int):
        self.total_training_steps = total_steps

    # ------------------------------------------------------------------
    # Observation (identical to DQNPolicy._make_observation)
    # ------------------------------------------------------------------

    def _make_observation(self, env: Env, task: Task, obs_type=("cpu", "buffer", "bw")):
        obs_type = list(set(obs_type) & {"cpu", "buffer", "bw"})
        obs = np.zeros((len(env.scenario.get_nodes()), len(obs_type)), dtype=np.float32)

        for node_name in env.scenario.get_nodes():
            idx = env.scenario.node_name2id[node_name]
            if "cpu" in obs_type:
                obs[idx, obs_type.index("cpu")] = env.scenario.get_node(node_name).free_cpu_freq
            if "buffer" in obs_type:
                obs[idx, obs_type.index("buffer")] = env.scenario.get_node(node_name).buffer_free_size()
            if "bw" in obs_type:
                src_node = "e0"
                if node_name != src_node:
                    obs[idx, obs_type.index("bw")] = min(
                        link.free_bandwidth
                        for link in env.scenario.infrastructure.get_shortest_links(src_node, node_name)
                    )
                else:
                    obs[idx, obs_type.index("bw")] = max(
                        link.free_bandwidth
                        for link in env.scenario.infrastructure.get_links().values()
                    )

        if task is None:
            task_obs = np.zeros(4, dtype=np.float32)
        else:
            task_obs = np.array(
                [task.task_size, task.cycles_per_bit, task.trans_bit_rate, task.ddl],
                dtype=np.float32,
            )

        return obs, task_obs

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, env: Env, task: Task, train: bool = True):
        """
        Select a destination node.

        Returns:
            action     (int)  — selected node index
            log_prob   (float)
            value      (float)
            state      (tuple[np.ndarray, np.ndarray]) — (obs, task_obs) for storage
        """
        obs, task_obs = self._make_observation(env, task, self.obs_type)
        obs_t = torch.tensor(obs, dtype=self.dtype, device=self.device).unsqueeze(0)
        task_t = torch.tensor(task_obs, dtype=self.dtype, device=self.device).unsqueeze(0)

        if train:
            self.total_steps += 1

        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(obs_t, task_t)

        if train:
            dist = Categorical(logits=logits.squeeze(0))
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t).item()
        else:
            action_t = logits.squeeze(0).argmax()
            log_prob = 0.0

        action = action_t.item()
        value_scalar = value.item()

        return action, log_prob, value_scalar, (obs, task_obs)

    # ------------------------------------------------------------------
    # Reward normalization (mirrors DQNPolicy.norm_reward / _norm_reward)
    # ------------------------------------------------------------------

    def norm_reward(self, reward, _lambda):
        r = self._norm_reward(reward, _lambda)
        self.avg_reward = self.avg_reward * 0.999 + r * (1 - 0.999)
        return r

    def _norm_reward(self, reward, _lambda):
        eps = self.reward_eps
        mom = self.reward_momentum

        if reward[0] == 1:
            if self.reward_norm == "standard":
                self.reward_mean[0] = self.reward_mean[0] * mom + reward[0] * (1 - mom)
                self.reward_var[0] = self.reward_var[0] * mom + (reward[0] - self.reward_mean[0]) ** 2 * (1 - mom)
                reward[0] = (reward[0] - self.reward_mean[0]) / (np.sqrt(self.reward_var[0]) + eps)
            elif self.reward_norm == "ln_standard":
                reward[0] = np.log(reward[0] + eps)
                self.reward_mean[0] = self.reward_mean[0] * mom + reward[0] * (1 - mom)
                self.reward_var[0] = self.reward_var[0] * mom + (reward[0] - self.reward_mean[0]) ** 2 * (1 - mom)
                reward[0] = (reward[0] - self.reward_mean[0]) / (np.sqrt(self.reward_var[0]) + eps)
            elif self.reward_norm == "mean":
                self.reward_mean[0] = self.reward_mean[0] * mom + reward[0] * (1 - mom)
                reward[0] = reward[0] / (self.reward_mean[0] + eps)
            elif self.reward_norm == "ln_mean":
                self.reward_mean[0] = self.reward_mean[0] * mom + reward[0] * (1 - mom)
                reward[0] = np.log(reward[0] / (eps + self.reward_mean[0]))
            elif self.reward_norm == "partial_ln_mean":
                reward[0] = np.log(reward[0] + eps)
            return sum(_lambda[i] * reward[i] for i in range(3))
        else:
            if self.reward_norm == "max":
                self.reward_max = [max(self.reward_max[i], reward[i]) for i in range(3)]
                reward = [reward[i] / (self.reward_max[i] + eps) for i in range(3)]
            elif self.reward_norm == "mean":
                self.reward_mean = [self.reward_mean[i] * mom + reward[i] * (1 - mom) for i in range(3)]
                reward = [reward[i] / (self.reward_mean[i] + eps) for i in range(3)]
            elif self.reward_norm == "standard":
                self.reward_mean = [self.reward_mean[i] * mom + reward[i] * (1 - mom) for i in range(3)]
                self.reward_var = [self.reward_var[i] * mom + (reward[i] - self.reward_mean[i]) ** 2 * (1 - mom) for i in range(3)]
                reward = [(reward[i] - self.reward_mean[i]) / (np.sqrt(self.reward_var[i]) + eps) for i in range(3)]
            elif self.reward_norm == "ln_standard":
                reward = [np.log(reward[i] + eps) for i in range(3)]
                self.reward_mean = [self.reward_mean[i] * mom + reward[i] * (1 - mom) for i in range(3)]
                self.reward_var = [self.reward_var[i] * mom + (reward[i] - self.reward_mean[i]) ** 2 * (1 - mom) for i in range(3)]
                reward = [(reward[i] - self.reward_mean[i]) / (np.sqrt(self.reward_var[i]) + eps) for i in range(3)]
            elif self.reward_norm == "partial_mean":
                self.reward_mean = [self.reward_mean[i] * mom + reward[i] * (1 - mom) if i != 0 else reward[i] for i in range(3)]
                reward = [reward[i] / (self.reward_mean[i] + eps) if i != 0 else reward[i] for i in range(3)]
            elif self.reward_norm == "partial_ln_mean":
                self.reward_mean = [self.reward_mean[i] * mom + reward[i] * (1 - mom) if i != 0 else reward[i] for i in range(3)]
                reward = [np.log(reward[i] / (self.reward_mean[i] + eps) + eps) if i != 0 else reward[i] for i in range(3)]
            elif self.reward_norm == "ln":
                return sum(_lambda[i] * np.log(reward[i] + eps) for i in range(3))
            return sum(_lambda[i] * reward[i] for i in range(3))

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, rollout_buffer: list):
        """
        Perform PPO update from a collected rollout buffer.

        Each entry in rollout_buffer is a dict with keys:
            state      : (obs np.ndarray, task_obs np.ndarray)
            action     : int
            log_prob   : float
            value      : float
            reward     : float
            next_state : (obs np.ndarray, task_obs np.ndarray) or None
            done       : bool
        """
        if not rollout_buffer:
            return

        self.update_count += 1

        # ---- Unpack rollout ----
        obs_list        = [e["state"][0] for e in rollout_buffer]
        task_list       = [e["state"][1] for e in rollout_buffer]
        actions         = [e["action"]   for e in rollout_buffer]
        old_log_probs   = [e["log_prob"] for e in rollout_buffer]
        values          = [e["value"]    for e in rollout_buffer]
        rewards         = [e["reward"]   for e in rollout_buffer]
        dones           = [e["done"]     for e in rollout_buffer]
        next_obs_list   = [e["next_state"][0] if e["next_state"] is not None else np.zeros_like(e["state"][0]) for e in rollout_buffer]
        next_task_list  = [e["next_state"][1] if e["next_state"] is not None else np.zeros_like(e["state"][1]) for e in rollout_buffer]
        next_valid      = [e["next_state"] is not None for e in rollout_buffer]

        T = len(rollout_buffer)

        # ---- Bootstrap last next-value for GAE ----
        with torch.no_grad():
            next_obs_t  = torch.tensor(np.array(next_obs_list),  dtype=self.dtype, device=self.device)
            next_task_t = torch.tensor(np.array(next_task_list), dtype=self.dtype, device=self.device)
            self.model.eval()
            _, next_values_t = self.model(next_obs_t, next_task_t)
            next_values = next_values_t.squeeze(-1).cpu().numpy()

        # ---- GAE advantage computation ----
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            nv = next_values[t] if next_valid[t] else 0.0
            delta = rewards[t] + self.gamma * nv * (1.0 - float(dones[t])) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values, dtype=np.float32)

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # ---- Convert to tensors ----
        obs_t      = torch.tensor(np.array(obs_list),      dtype=self.dtype, device=self.device)
        task_t     = torch.tensor(np.array(task_list),     dtype=self.dtype, device=self.device)
        actions_t  = torch.tensor(actions, dtype=torch.int64, device=self.device)
        old_lp_t   = torch.tensor(old_log_probs, dtype=self.dtype, device=self.device)
        returns_t  = torch.tensor(returns,    dtype=self.dtype, device=self.device)
        adv_t      = torch.tensor(advantages, dtype=self.dtype, device=self.device)

        # ---- PPO epochs ----
        total_loss = 0.0
        total_grad_norm = 0.0
        n_updates = 0

        indices = np.arange(T)
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.mini_batch_size):
                batch_idx = indices[start : start + self.mini_batch_size]
                if len(batch_idx) < 2:
                    continue

                obs_b     = obs_t[batch_idx]
                task_b    = task_t[batch_idx]
                act_b     = actions_t[batch_idx]
                old_lp_b  = old_lp_t[batch_idx]
                ret_b     = returns_t[batch_idx]
                adv_b     = adv_t[batch_idx]

                self.model.train()
                logits, new_values = self.model(obs_b, task_b)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(act_b)
                entropy = dist.entropy().mean()

                # Clipped surrogate loss
                ratio = torch.exp(new_log_probs - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(new_values.squeeze(-1), ret_b)

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.clip_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_grad_norm += grad_norm.item()
                n_updates += 1

        if n_updates > 0:
            avg = total_loss / n_updates
            gn  = total_grad_norm / n_updates
            self.avg_loss      = self.avg_loss * 0.999      + avg * 0.001
            self.avg_grad_norm = self.avg_grad_norm * 0.999 + gn  * 0.001

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
