import torch
import torch.nn as nn
import torch.optim as optim
import random

import numpy as np

from core.env import Env
from core.task import Task

from policies.base_policy import BasePolicy


class MLP(nn.Module):   
    def __init__(self, d_in, d_pos,  d_model, output_size, n_layers=2,  bias=True, **kwargs):
        super(MLP, self).__init__()
        
        
        if n_layers < 2:
            raise ValueError("The number of layers must be at least 2.")
        layers = [nn.Linear(d_in*d_pos, d_model, bias=bias), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(d_model, d_model, bias=bias), nn.ReLU()]
        layers.append(nn.Linear(d_model, output_size))
        self.model = nn.Sequential(*layers)
        

    def forward(self, x, task):
        
        
        x = x / self.norm  # Apply normalization

        return self.model(x.view(x.size(0), -1))
    
    def register_norm(self, norm):
        self.register_buffer('norm', torch.tensor(norm).max(dim=0, keepdim=True).values.to(self.device))  # Register the normalization factor as a buffer
        print(self.norm)

class DQLPolicy(BasePolicy):
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
        
        self.obs_type = config["model"]["obs_type"]
        
        self.n_observations = len(self._make_observation(env, None, self.obs_type)[0])
        
        self.d_obs = len(self._make_observation(env, None, self.obs_type)[0][0])

        self.num_actions = len(env.scenario.node_id2name)

        # Retrieve configuration parameters.
        self.gamma = config["training"]["gamma"]
        self.epsilon = config["training"]["exploration"]["epsilon"]
        self.lr = config["training"]["lr"]
        
        # Replay buffer for transitions.
        self.replay_buffer = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else  "cpu")
        self.dtype = torch.float32
        
        self._init_model(env, config)
        
        

        
    def _init_model(self, env, config):
        self.model = MLP(d_in=self.d_obs, d_pos=self.n_observations, d_task=4, output_size=self.num_actions, **config["model"]).to(self.device).to(self.dtype)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        self.model.register_norm(self._make_observation(env, None, self.obs_type)[0])  # Register the normalization factor for latency


    def _make_observation(self, env: Env, task: Task, obs_type=["cpu", "buffer", "bw"]):
        """
        Returns a flat observation vector.
        For instance, we return the free CPU frequency for each node.
        """
        
        
        obs = np.zeros((len(env.scenario.get_nodes()), len(obs_type)), dtype=np.float32)
        
        for i, node_name in enumerate(env.scenario.get_nodes()):
            if "cpu" in obs_type:
                obs[env.scenario.node_name2id[node_name], obs_type.index("cpu")] = env.scenario.get_node(node_name).free_cpu_freq 
            if "buffer" in obs_type:
                obs[env.scenario.node_name2id[node_name], obs_type.index("buffer")] = env.scenario.get_node(node_name).buffer_free_size()
            if "bw" in obs_type:
                # Get the bandwidth for the link associated with the task
                src_node =  "e0"
                if node_name != src_node:
                    obs[env.scenario.node_name2id[node_name], obs_type.index("bw")] = min(link.free_bandwidth for link in env.scenario.infrastructure.get_shortest_links(src_node, node_name))
                else:
                    obs[env.scenario.node_name2id[node_name], obs_type.index("bw")] = max(link.free_bandwidth for link in env.scenario.infrastructure.get_links().values())


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
        state = self._make_observation(env, task, self.obs_type)
        obs, task_obs = state
        obs_tensor = torch.tensor(obs).unsqueeze(0).to(self.device).to(self.dtype)
        task_tensor = torch.tensor(task_obs).unsqueeze(0).to(self.device).to(self.dtype)
                
        if random.random() < self.epsilon and train:
            action = random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                self.model.eval()
                q_values = self.model(obs_tensor, task_tensor)
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
        obs_tensor = torch.tensor(np.array(obs_batch), device=self.device, dtype=self.dtype)
        task_tensor = torch.tensor(np.array(task_obs_batch), device=self.device, dtype=self.dtype)
        next_obs_tensor = torch.tensor(np.array(next_obs_batch), device=self.device, dtype=self.dtype)
        next_task_tensor = torch.tensor(np.array(next_task_obs_batch), device=self.device, dtype=self.dtype)

        actions_tensor = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(-1)  # Actions remain long dtype
        rewards_tensor = torch.tensor(rewards, device=self.device, dtype=self.dtype)
        dones_tensor = torch.tensor(dones, device=self.device, dtype=self.dtype)
        

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
    
    def save(self, path):
        """
        Saves the model to the specified path.
        """
        torch.save(self.model.state_dict(), path)   

    def load(self, path):
        """
        Loads the model from the specified path.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

