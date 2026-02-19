import copy
import torch
import torch.nn as nn
import torch.optim as optim
from policies.dql.base_policy import DQNPolicy
from policies.model.mlp import MLP, DuelingMLP


class MLPPolicy(DQNPolicy):

    def _init_model(self, env, config, dataset=None):
        self.model = MLP(d_in=self.d_obs, d_pos=self.n_observations, d_task=4, output_size=self.num_actions, **config["model"])
        self.model.register_norm(self._make_observation(env, None, self.obs_type)[0], dataset=dataset)
        self.model.to(self.device).to(self.dtype)


class DuelingMLPPolicy(DQNPolicy):

    def _init_model(self, env, config, dataset=None):
        self.model = DuelingMLP(d_in=self.d_obs, d_pos=self.n_observations, d_task=4, output_size=self.num_actions, **config["model"])
        self.model.register_norm(self._make_observation(env, None, self.obs_type)[0], dataset=dataset)
        self.model.to(self.device).to(self.dtype)

    def _update(self):
        import random
        import numpy as np
        self._update_lr()

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        obs_batch, task_obs_batch = zip(*states)
        next_obs_batch, next_task_obs_batch = zip(*next_states)

        obs_tensor      = torch.tensor(np.array(obs_batch),          dtype=self.dtype, device=self.device)
        task_tensor     = torch.tensor(np.array(task_obs_batch),      dtype=self.dtype, device=self.device)
        next_obs_tensor = torch.tensor(np.array(next_obs_batch),      dtype=self.dtype, device=self.device)
        next_task_tensor= torch.tensor(np.array(next_task_obs_batch), dtype=self.dtype, device=self.device)

        actions_tensor = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_tensor = torch.tensor(rewards, dtype=self.dtype, device=self.device)
        dones_tensor   = torch.tensor(dones,   dtype=self.dtype, device=self.device)

        self.optimizer.zero_grad()

        self.model.train()
        q_values    = self.model(obs_tensor, task_tensor)
        predicted_q = q_values.gather(1, actions_tensor).squeeze()

        with torch.no_grad():
            # Double DQN: online network selects action, target network evaluates it
            next_actions  = self.model(next_obs_tensor, next_task_tensor).argmax(dim=1, keepdim=True)
            next_q_values = self.target_model(next_obs_tensor, next_task_tensor)
            max_next_q    = next_q_values.gather(1, next_actions).squeeze()
            target_q = rewards_tensor if self.gamma == 0 else rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q

        loss = self.criterion(predicted_q, target_q)
        loss.backward()

        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)

        self.optimizer.step()
        return loss.item()


        


