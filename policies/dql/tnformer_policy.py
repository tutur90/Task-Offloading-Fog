import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.distributions import Categorical  # (optional for epsilon random selection)

from policies.model.TaskFormer import TaskFormer

import numpy as np

from core.env import Env
from core.task import Task

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
dtype = torch.float32

class TaskFormerPolicy:
    def __init__(self, env, config):
        """
        A simple deep Q-learning policy.

        Args:
            env: The simulation environment.
            config (dict): A configuration dictionary containing:
                - training: with keys 'lr', 'gamma', 'epsilon'
                - model: with key 'd_model' (used as the hidden size)
        """
        self.env = env
        
        self.n_observations = len(self._make_observation(env, None)[0])
        
        self.d_obs = len(self._make_observation(env, None)[0][0])
        
        
        
        self.num_actions = len(env.scenario.node_id2name)

        # Retrieve configuration parameters.
        self.gamma = config["training"]["gamma"]
        self.epsilon = config["training"]["epsilon"]
        self.lr = config["training"]["lr"]
        self.beta = config["training"]["beta"]
        
        
        
        d_model = config["model"]["d_model"]
        n_layers = config["model"]["n_layers"]
        n_heads = config["model"]["n_heads"]
        mlp_ratio = config["model"]["mlp_ratio"]
        n_layers_ratio = config["model"]["n_layers_ratio"]
        dropout = config["model"]["dropout"]
        mode = config["model"]["mode"]
        
        
        self.model = TaskFormer(d_in=self.d_obs, d_pos=self.n_observations, d_task=4, d_model=d_model, d_ff=d_model*mlp_ratio, n_heads=n_heads, n_layers=n_layers, dropout=dropout, mode=mode).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # Replay buffer for transitions.
        self.replay_buffer = []

    def _make_observation(self, env, task):
        """
        Returns a flat observation vector.
        For instance, we return the free CPU frequency for each node.
        """
        cpu_obs = {node_name: env.scenario.get_node(node_name).free_cpu_freq 
               for node_name in env.scenario.get_nodes()}
        
        buffer_obs = {node_name: env.scenario.get_node(node_name).buffer_free_size()
               for node_name in env.scenario.get_nodes()}
        # print(env.scenario.get_links())
        bw_obs = {link_name:env.scenario.get_link(link_name[0], link_name[1]).free_bandwidth for link_name in env.scenario.get_links()}
        
        obs = np.zeros((len(env.scenario.get_nodes()), 4))
        
        for i, node_name in enumerate(env.scenario.get_nodes()):
            obs[env.scenario.node_name2id[node_name], 0] = cpu_obs[node_name]
            obs[env.scenario.node_name2id[node_name], 1] = buffer_obs[node_name]

        for i, link_name in enumerate(env.scenario.get_links()):
            
            if link_name[0] == 'e0':
                obs[env.scenario.node_name2id[link_name[1]], 2] = bw_obs[link_name]
            else:
                obs[env.scenario.node_name2id[link_name[0]], 3] = bw_obs[link_name]
        
        if task is None:
            task_obs = [0, 0, 0, 0]
        else:
            task_obs = [
                task.task_size,
                task.cycles_per_bit,
                task.trans_bit_rate,
                task.ddl,
            ]
        
        return obs, task_obs

    def act(self, env, task, train=True):
        """
        Chooses an action using an ε-greedy strategy and records the current state.
        """
        state = self._make_observation(env, task)
        obs, task_obs = state
        obs_tensor = torch.tensor(obs, dtype=dtype).unsqueeze(0).to(device)
        task_tensor = torch.tensor(task_obs, dtype=dtype).unsqueeze(0).to(device)

        rand = random.random()
        
        if rand < self.epsilon and train:
            action = random.randrange(self.num_actions)
        elif rand - self.epsilon < self.beta * (1-self.epsilon) and train:
            with torch.no_grad():
                q_values = self.model(obs_tensor, task_tensor, False)
                action = torch.argmax(q_values, dim=1).item()
        else:
            with torch.no_grad():
                q_values = self.model(obs_tensor, task_tensor, True)
                action = torch.argmax(q_values, dim=1).item()


        # Return both the chosen action and the current state.
        return action, state
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay buffer.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))
        


    def update(self):
        """
        Performs an update over all stored transitions using batched operations,
        moves tensors to the appropriate device and dtype, and clears the replay buffer.
        """
        if not self.replay_buffer:
            return 0.0

        # Unpack transitions
        states, actions, rewards, next_states, dones = zip(*self.replay_buffer)
        obs_batch, task_obs_batch = zip(*states)
        next_obs_batch, next_task_obs_batch = zip(*next_states)

        # Convert lists to batched tensors and move them to the device with the appropriate dtype
        obs_tensor = torch.tensor(np.array(obs_batch), dtype=dtype, device=device)
        task_tensor = torch.tensor(np.array(task_obs_batch), dtype=dtype, device=device)
        next_obs_tensor = torch.tensor(np.array(next_obs_batch), dtype=dtype, device=device)
        next_task_tensor = torch.tensor(np.array(next_task_obs_batch), dtype=dtype, device=device)

        actions_tensor = torch.tensor(np.array(actions), dtype=torch.int64).to(device).unsqueeze(-1)  # Actions remain long dtype
        rewards_tensor = torch.tensor(rewards, dtype=dtype).to(device)
        dones_tensor = torch.tensor(dones, dtype=dtype).to(device)
        

        self.optimizer.zero_grad()

        # Compute Q-values for the current states
        q_values = self.model(obs_tensor, task_tensor).squeeze()  # Shape: [batch_size, num_actions]

        predicted_q = q_values.gather(1, actions_tensor).squeeze()


        # Compute target Q-values from next states
        with torch.no_grad():
            next_q_values = self.model(next_obs_tensor, next_task_tensor).squeeze()  # Shape: [batch_size, num_actions]
            max_next_q, _ = torch.max(next_q_values, dim=1)
            target_q = rewards_tensor if self.gamma == 0 else rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q

        # Compute loss over the batch
        loss = self.criterion(predicted_q, target_q)
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.clear()
        return loss.item()

