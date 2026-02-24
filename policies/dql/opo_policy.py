"""OPO — Online Predictive Offloading policy.

Based on:
    Tu et al., "Task Offloading Based on LSTM Prediction and Deep Reinforcement
    Learning for Efficient Edge Computing in IoT", Future Internet 2022, 14, 30.

Architecture:
    Phase 1 (training):  Double Dueling DQN + LSTM Load Predictor for
                         informed exploration (LSTM-guided ε-greedy).
    Phase 2 (inference): LSTM Task Predictor pre-computes actions; fallback
                         to DQN when prediction error exceeds threshold.
"""

import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.env import Env
from core.task import Task
from policies.dql.base_policy import DQNPolicy
from policies.model.mlp import DuelingMLP
from policies.model.lstm_opo import LSTMLoadPredictor, LSTMTaskPredictor


class OPOPolicy(DQNPolicy):
    """Double Dueling DQN with LSTM-augmented exploration and inference."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_model(self, env: Env, config: dict, dataset=None):
        opo_cfg = config.get("model", {})
        self.history_window: int = opo_cfg.get("history_window", 50)
        lstm_hidden: int = opo_cfg.get("lstm_hidden_dim", 64)
        lstm_layers: int = opo_cfg.get("lstm_n_layers", 1)
        lstm_lr: float = opo_cfg.get("lstm_lr", 1e-3)
        self.prediction_threshold: float = opo_cfg.get("prediction_threshold", 0.1)

        n_nodes = len(env.scenario.get_nodes())

        # ── Q-network (Double Dueling DQN backbone) ──────────────────
        self.model = DuelingMLP(
            d_in=self.d_obs,
            d_pos=self.n_observations,
            d_task=4,
            output_size=self.num_actions,
            **config["model"],
        )
        self.model.register_norm(
            self._make_observation(env, None, self.obs_type)[0],
            dataset=dataset,
        )
        self.model.to(self.device).to(self.dtype)

        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()

        # ── LSTM predictors ───────────────────────────────────────────
        self.load_lstm = LSTMLoadPredictor(
            n_nodes=n_nodes,
            hidden_dim=lstm_hidden,
            n_layers=lstm_layers,
        ).to(self.device).to(self.dtype)

        self.task_lstm = LSTMTaskPredictor(
            hidden_dim=lstm_hidden,
            n_layers=lstm_layers,
        ).to(self.device).to(self.dtype)

        self.load_lstm_optim = optim.Adam(self.load_lstm.parameters(), lr=lstm_lr)
        self.task_lstm_optim = optim.Adam(self.task_lstm.parameters(), lr=lstm_lr)
        self.lstm_criterion = nn.MSELoss()

        # ── History buffers ───────────────────────────────────────────
        # load_history stores idle CPU fraction for each node
        self.load_history: deque = deque(maxlen=self.history_window)
        # task_history stores normalised task features [size, cpb, tbr, ddl]
        self.task_history: deque = deque(maxlen=self.history_window)

        # Inference state: cached pre-computed action + predicted features
        self.precomputed_action: int | None = None
        self.last_predicted_task_feat: np.ndarray | None = None

        # Normalisers for task features (set lazily on first observation)
        self._task_norm: np.ndarray | None = None

        # ── Diagnostic counters ───────────────────────────────────────
        self.stats = {
            # Training action-selection breakdown
            "exploit": 0,          # argmax Q chosen
            "lstm_guided": 0,      # LSTM load predictor guided the action
            "random": 0,           # pure random exploration
            # Inference pre-computation breakdown
            "precomputed_hits": 0, # cached action used (prediction error < threshold)
            "precomputed_misses": 0,  # prediction error too high → DQN fallback
            # LSTM prediction losses (running sum + count for mean)
            "load_lstm_loss_sum": 0.0,
            "load_lstm_loss_n": 0,
            "task_lstm_loss_sum": 0.0,
            "task_lstm_loss_n": 0,
        }

    def lstm_stats_summary(self) -> dict:
        """Return a human-readable dict of LSTM diagnostic stats."""
        s = self.stats
        total_train = s["exploit"] + s["lstm_guided"] + s["random"] or 1
        total_infer = s["precomputed_hits"] + s["precomputed_misses"] or 1
        return {
            "exploit_%": 100 * s["exploit"] / total_train,
            "lstm_guided_%": 100 * s["lstm_guided"] / total_train,
            "random_%": 100 * s["random"] / total_train,
            "precomputed_hit_%": 100 * s["precomputed_hits"] / total_infer,
            "avg_load_lstm_loss": (
                s["load_lstm_loss_sum"] / s["load_lstm_loss_n"]
                if s["load_lstm_loss_n"] > 0 else float("nan")
            ),
            "avg_task_lstm_loss": (
                s["task_lstm_loss_sum"] / s["task_lstm_loss_n"]
                if s["task_lstm_loss_n"] > 0 else float("nan")
            ),
        }

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_load_obs(self, env: Env) -> np.ndarray:
        """Return idle CPU fraction per node (shape: [n_nodes])."""
        n = len(env.scenario.get_nodes())
        obs = np.zeros(n, dtype=np.float32)
        for node_name, node in env.scenario.get_nodes().items():
            idx = env.scenario.node_name2id[node_name]
            max_freq = node.max_cpu_freq if node.max_cpu_freq > 0 else 1.0
            obs[idx] = node.free_cpu_freq / max_freq
        return obs

    def _task_to_feat(self, task: Task) -> np.ndarray:
        raw = np.array(
            [task.task_size, task.cycles_per_bit, task.trans_bit_rate, task.ddl],
            dtype=np.float32,
        )
        t_min = self.model.task_min.cpu().numpy()
        t_max = self.model.task_max.cpu().numpy()
        return (raw - t_min) / (t_max - t_min + 1e-8)

    # ------------------------------------------------------------------
    # LSTM-guided action (informed exploration)
    # ------------------------------------------------------------------

    def _lstm_guided_action(self) -> int:
        """Use load LSTM to predict which server will have most idle CPU."""
        seq = np.array(self.load_history, dtype=np.float32)  # (T, n_nodes)
        x = torch.tensor(seq, dtype=self.dtype, device=self.device).unsqueeze(0)
        with torch.no_grad():
            self.load_lstm.eval()
            predicted_load = self.load_lstm(x).squeeze(0)  # (n_nodes,)
        return int(torch.argmax(predicted_load).item())

    # ------------------------------------------------------------------
    # Pre-compute inference action for next task
    # ------------------------------------------------------------------

    def _precompute_next_action(self, env: Env):
        """Predict next task features and compute its DQN action."""
        if len(self.task_history) < self.history_window:
            self.precomputed_action = None
            self.last_predicted_task_feat = None
            return

        seq = np.array(self.task_history, dtype=np.float32)  # (T, 4)
        x = torch.tensor(seq, dtype=self.dtype, device=self.device).unsqueeze(0)
        with torch.no_grad():
            self.task_lstm.eval()
            pred_feat = self.task_lstm(x).squeeze(0).cpu().numpy()  # (4,)

        self.last_predicted_task_feat = pred_feat

        # Build a fake observation using predicted task features
        obs, _ = self._make_observation(env, None, self.obs_type)
        obs_t = torch.tensor(obs, dtype=self.dtype, device=self.device).unsqueeze(0)
        pred_t = torch.tensor(pred_feat, dtype=self.dtype, device=self.device).unsqueeze(0)

        with torch.no_grad():
            self.model.eval()
            q_values = self.model(obs_t, pred_t)
            self.precomputed_action = int(torch.argmax(q_values, dim=1).item())

    # ------------------------------------------------------------------
    # act()
    # ------------------------------------------------------------------

    def act(self, env: Env, task: Task, train: bool = True):
        state = self._make_observation(env, task, self.obs_type)
        obs, task_obs = state

        # Update histories
        self.load_history.append(self._get_load_obs(env))
        self.task_history.append(self._task_to_feat(task))

        obs_tensor = torch.tensor(obs, dtype=self.dtype, device=self.device).unsqueeze(0)
        task_tensor = torch.tensor(task_obs, dtype=self.dtype, device=self.device).unsqueeze(0)

        if train:
            self.total_steps += 1
            self._update_epsilon()

        # ── Warm-up: pure random ──────────────────────────────────────
        if train and self.total_steps <= self.learning_starts:
            return random.randrange(self.num_actions), state

        # ── Training: modified ε-greedy ───────────────────────────────
        if train:
            if random.random() < self.epsilon:
                # Exploration branch
                sigma = random.random()
                history_ready = len(self.load_history) >= self.history_window
                if sigma < (1.0 - self.epsilon) and history_ready:
                    # Informed exploration: pick server predicted to be most idle
                    action = self._lstm_guided_action()
                    self.stats["lstm_guided"] += 1
                else:
                    # Pure random exploration
                    action = random.randrange(self.num_actions)
                    self.stats["random"] += 1
            else:
                # Exploitation
                with torch.no_grad():
                    self.model.eval()
                    q_values = self.model(obs_tensor, task_tensor)
                    action = int(torch.argmax(q_values, dim=1).item())
                self.stats["exploit"] += 1

            return action, state

        # ── Inference: use pre-computed action when possible ──────────
        action = None
        actual_feat = self._task_to_feat(task)

        if self.precomputed_action is not None and self.last_predicted_task_feat is not None:
            pred = self.last_predicted_task_feat
            # Normalised RMSE across the 4 task dimensions
            denom = np.abs(actual_feat) + 1e-8
            rel_error = float(np.mean(np.abs(pred - actual_feat) / denom))

            if rel_error < self.prediction_threshold:
                action = self.precomputed_action
                self.stats["precomputed_hits"] += 1
            else:
                # Fallback: run DQN on real task (normalised features already in history)
                with torch.no_grad():
                    self.model.eval()
                    q_values = self.model(obs_tensor, task_tensor)
                    action = int(torch.argmax(q_values, dim=1).item())
                self.stats["precomputed_misses"] += 1

        if action is None:
            with torch.no_grad():
                self.model.eval()
                q_values = self.model(obs_tensor, task_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

        # Pre-compute action for the NEXT task
        self._precompute_next_action(env)

        return action, state

    # ------------------------------------------------------------------
    # LSTM online training helpers
    # ------------------------------------------------------------------

    def _train_load_lstm(self):
        """One-step ahead prediction update for the load LSTM."""
        if len(self.load_history) < self.history_window:
            return
        seq = np.array(self.load_history, dtype=np.float32)   # (T, n_nodes)
        # Predict from [0 … T-2], target = T-1
        x = torch.tensor(seq[:-1], dtype=self.dtype, device=self.device).unsqueeze(0)
        y = torch.tensor(seq[-1], dtype=self.dtype, device=self.device).unsqueeze(0)
        self.load_lstm.train()
        self.load_lstm_optim.zero_grad()
        pred = self.load_lstm(x)
        loss = self.lstm_criterion(pred, y)
        loss.backward()
        self.load_lstm_optim.step()
        self.stats["load_lstm_loss_sum"] += loss.item()
        self.stats["load_lstm_loss_n"] += 1

    def _train_task_lstm(self):
        """One-step ahead prediction update for the task LSTM."""
        if len(self.task_history) < self.history_window:
            return
        seq = np.array(self.task_history, dtype=np.float32)   # (T, 4)
        x = torch.tensor(seq[:-1], dtype=self.dtype, device=self.device).unsqueeze(0)
        y = torch.tensor(seq[-1], dtype=self.dtype, device=self.device).unsqueeze(0)
        self.task_lstm.train()
        self.task_lstm_optim.zero_grad()
        pred = self.task_lstm(x)
        loss = self.lstm_criterion(pred, y)
        loss.backward()
        self.task_lstm_optim.step()
        self.stats["task_lstm_loss_sum"] += loss.item()
        self.stats["task_lstm_loss_n"] += 1

    # ------------------------------------------------------------------
    # _update() — Double Dueling DQN + LSTM training
    # ------------------------------------------------------------------

    def _update(self):
        self._update_lr()

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        obs_batch, task_obs_batch = zip(*states)
        next_obs_batch, next_task_obs_batch = zip(*next_states)
        
        rewards = self.aggregate_reward(rewards)  # Aggregate reward components into a single scalar for each transition

        obs_tensor = torch.tensor(np.array(obs_batch), dtype=self.dtype, device=self.device)
        task_tensor = torch.tensor(np.array(task_obs_batch), dtype=self.dtype, device=self.device)
        next_obs_tensor = torch.tensor(np.array(next_obs_batch), dtype=self.dtype, device=self.device)
        next_task_tensor = torch.tensor(np.array(next_task_obs_batch), dtype=self.dtype, device=self.device)

        actions_tensor = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_tensor = torch.tensor(rewards, dtype=self.dtype, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=self.dtype, device=self.device)

        # ── Double Dueling DQN update ─────────────────────────────────
        self.optimizer.zero_grad()
        self.model.train()
        q_values = self.model(obs_tensor, task_tensor)
        predicted_q = q_values.gather(1, actions_tensor).squeeze()

        with torch.no_grad():
            # Online network selects action; target network evaluates value
            next_actions = self.model(next_obs_tensor, next_task_tensor).argmax(dim=1, keepdim=True)
            next_q_values = self.target_model(next_obs_tensor, next_task_tensor)
            max_next_q = next_q_values.gather(1, next_actions).squeeze()
            target_q = (
                rewards_tensor
                if self.gamma == 0
                else rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q
            )

        loss = self.criterion(predicted_q, target_q)
        loss.backward()

        
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)

        self.optimizer.step()

        # ── LSTM online updates ───────────────────────────────────────
        self._train_load_lstm()
        self._train_task_lstm()


        return loss.item(), grad_norm.item()

    # ------------------------------------------------------------------
    # Checkpoint helpers (extend base class to include LSTMs)
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
                "model": self.model.state_dict(),
                "load_lstm": self.load_lstm.state_dict(),
                "task_lstm": self.task_lstm.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.target_model.load_state_dict(ckpt["model"])
        self.load_lstm.load_state_dict(ckpt["load_lstm"])
        self.task_lstm.load_state_dict(ckpt["task_lstm"])
        self.model.eval()
        self.target_model.eval()
        self.load_lstm.eval()
        self.task_lstm.eval()
