import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from collections import deque

import numpy as np

from core.env import Env
from core.task import Task



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
        self.register_buffer('norm', torch.tensor(norm, dtype=self.dtype).max(dim=0, keepdim=True).values)  # Register the normalization factor as a buffer
        # self.register_buffer('norm', torch.tensor(norm, dtype=dtype).to(device))  # Register the normalization factor as a buffer


class DQNPolicy:
    def __init__(self, env, config, device="auto"):
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
        self.epsilon_start = config["training"]["epsilon"]
        self.epsilon = self.epsilon_start
        self.epsilon_min = config["training"].get("epsilon_min", 0.01)
        self.epsilon_decay = config["training"].get("epsilon_decay", 0.9)
        self.lr = config["training"]["lr"]

        # Replay buffer for transitions.
        self.buffer_size = config["training"].get("buffer_size", 10000)
        self.batch_size = config["training"].get("batch_size", 64)
        self.target_update_freq = config["training"].get("target_update_freq", 100)
        self.update_freq = config["training"].get("update_freq", 1)
        self.learning_starts = config["training"].get("learning_starts", 0)
        self.total_training_steps = None  # set by set_training_steps()
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.update_count = 0
        self.total_steps = 0
        
        if device == "auto":

            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        self.dtype = torch.float32
        
        self._init_model(env, config)

    def _init_model(self, env, config):
        self.model = MLP(d_in=self.d_obs, d_pos=self.n_observations, d_task=4, output_size=self.num_actions, **config["model"]).to(self.device).to(self.dtype)
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.model.register_norm(self._make_observation(env, None, self.obs_type)[0])
        self.target_model.register_norm(self._make_observation(env, None, self.obs_type)[0])  # Register the normalization factor for latency
        



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
        """Set total training steps for linear epsilon schedule."""
        self.total_training_steps = total_steps

    def _update_epsilon(self):
        """Linearly decay epsilon from epsilon_start to epsilon_min over epsilon_decay fraction of training."""
        if self.total_training_steps is None:
            return
        decay_steps = int(self.total_training_steps * self.epsilon_decay)
        steps_since_learn = self.total_steps - self.learning_starts
        if steps_since_learn <= 0:
            self.epsilon = self.epsilon_start
        elif steps_since_learn >= decay_steps:
            self.epsilon = self.epsilon_min
        else:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_min) * (steps_since_learn / decay_steps)

    def act(self, env, task, train=True):
        """
        Chooses an action using an ε-greedy strategy and records the current state.
        """
        state = self._make_observation(env, task, self.obs_type)
        obs, task_obs = state
        obs_tensor = torch.tensor(obs, dtype=self.dtype, device=self.device).unsqueeze(0)
        task_tensor = torch.tensor(task_obs, dtype=self.dtype, device=self.device).unsqueeze(0)

        if train:
            self.total_steps += 1
            self._update_epsilon()

        if train and self.total_steps <= self.learning_starts:
            action = random.randrange(self.num_actions)
        elif random.random() < self.epsilon and train:
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
        
    def _update(self):
        """
        Performs an update over a sampled batch of transitions using batched operations,
        moves tensors to the appropriate device and dtype.
        """

        # Sample a batch from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        obs_batch, task_obs_batch = zip(*states)
        next_obs_batch, next_task_obs_batch = zip(*next_states)

        # Convert lists to batched tensors and move them to the device with the appropriate dtype
        obs_tensor = torch.tensor(np.array(obs_batch), dtype=self.dtype, device=self.device)
        task_tensor = torch.tensor(np.array(task_obs_batch), dtype=self.dtype, device=self.device)
        next_obs_tensor = torch.tensor(np.array(next_obs_batch), dtype=self.dtype, device=self.device)
        next_task_tensor = torch.tensor(np.array(next_task_obs_batch), dtype=self.dtype, device=self.device)

        actions_tensor = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_tensor = torch.tensor(rewards, dtype=self.dtype, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=self.dtype, device=self.device)


        self.optimizer.zero_grad()

        # Compute Q-values for the current states
        self.model.train()
        q_values = self.model(obs_tensor, task_tensor).squeeze()  # Shape: [batch_size, num_actions]

        predicted_q = q_values.gather(1, actions_tensor).squeeze()


        # Compute target Q-values from next states using target network
        with torch.no_grad():
            next_q_values = self.target_model(next_obs_tensor, next_task_tensor).squeeze()  # Shape: [batch_size, num_actions]
            max_next_q, _ = torch.max(next_q_values, dim=1)
            target_q = rewards_tensor if self.gamma == 0 else rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q

        # Compute loss over the batch
        loss = self.criterion(predicted_q, target_q)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update(self):
        """
        Performs an update over a sampled batch of transitions using batched operations,
        moves tensors to the appropriate device and dtype.
        """
        if len(self.replay_buffer) < self.batch_size or self.total_steps <= self.learning_starts:
            return 0.0
        
        if self.update_count % self.update_freq == 0:
            loss = self._update()

        # Update target network periodically (based on total task steps, not gradient steps)
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()


    def update_target_network(self):
        """
        Copies the weights from the main model to the target model.
        """
        self.target_model.load_state_dict(self.model.state_dict())
    
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
        self.target_model.load_state_dict(self.model.state_dict())
        self.model.eval()
        self.target_model.eval()

