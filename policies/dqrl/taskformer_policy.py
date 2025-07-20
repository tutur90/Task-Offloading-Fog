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
        self.beta = config.get("training", {}).get("beta", 0.5)
        self.beta_decay = config.get("training", {}).get("beta_decay", 1)
        
        
        
        d_model = config["model"]["d_model"]
        n_layers = config["model"]["n_layers"]
        n_heads = config["model"]["n_heads"]
        mlp_ratio = config["model"]["mlp_ratio"]
        dropout = config["model"]["dropout"]
        mode = config["model"]["mode"]
        
        
        self.model = TaskFormer(d_in=self.d_obs, d_pos=self.n_observations, d_task=4, d_model=d_model, d_ff=d_model*mlp_ratio, n_heads=n_heads, n_layers=n_layers, dropout=dropout, mode=mode).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # Replay buffer for transitions.
        self.replay_buffer = []

    def _make_observation(self, env: Env, task: Task, obs_type=["cpu", "buffer", "bw"]):
        """
        Returns a flat observation vector.
        For instance, we return the free CPU frequency for each node.
        """
        
        
        obs = np.zeros((len(env.scenario.get_nodes()), len(obs_type)))
        
        for i, node_name in enumerate(env.scenario.get_nodes()):
            if "cpu" in obs_type:
                obs[env.scenario.node_name2id[node_name], obs_type.index("cpu")] = env.scenario.get_node(node_name).free_cpu_freq 
            if "buffer" in obs_type:
                obs[env.scenario.node_name2id[node_name], obs_type.index("buffer")] = env.scenario.get_node(node_name).buffer_free_size()
            if "bw" in obs_type and task is not None:
                # Get the bandwidth for the link associated with the task
                src_node = task.src_name
                if node_name != src_node:
                    obs[env.scenario.node_name2id[node_name], obs_type.index("bw")] = min(link.free_bandwidth for link in env.scenario.infrastructure.get_shortest_links(src_node, node_name))
                else:
                    obs[env.scenario.node_name2id[node_name], obs_type.index("bw")] = max(link.free_bandwidth for link in env.scenario.infrastructure.get_links().values())

        # for i, link_name in enumerate(env.scenario.get_links()):
        #
        #     if link_name[0] == 'e0':
        #         obs[env.scenario.node_name2id[link_name[1]], 2] = bw_obs[link_name]
        #     else:
        #         obs[env.scenario.node_name2id[link_name[0]], 3] = bw_obs[link_name]
        

        
        if task is None:
            task_obs = [0, 0, 0, 0]
        else:
            task_obs = [
                task.task_size,
                task.cycles_per_bit,
                task.trans_bit_rate,
                task.ddl,
            ]
            # obs["bw"] = {}
            
            # src_node = task.src_name


            # for node_name in env.scenario.get_nodes():
            #     if node_name == src_node:
            #         obs["bw"][node_name] = 0
            #     for link in env.scenario.infrastructure.get_longest_shortest_path(src_node, dst_node):
            #         if link[0] == node_name or link[1] == node_name:
            #             obs["bw"][node_name] = min(env.scenario.get_node(node_name).free_bandwidth for link in env.scenario.infrastructure.get_longest_shortest_path(src_node, dst_node) if link[0] == node_name or link[1] == node_name)

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
        
        self.beta *= self.beta_decay
        
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

