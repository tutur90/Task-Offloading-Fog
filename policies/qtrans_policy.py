import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.distributions import Categorical  # (optional for epsilon random selection)

from policies.model.BaseMLP import BaseMLP
from policies.model.Transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QTransPolicy:
    def __init__(self, env, lr=1e-3, gamma=0.99, epsilon=0.1, d_model=16, nhead=2, n_layers=3, d_ff=16, dropout=0.1):
        """
        A simple deep Q-learning policy.

        Args:
            env: The simulation environment.
            lr: Learning rate.
            hidden_size: Size of hidden layers.
            gamma: Discount factor.
            epsilon: For ε-greedy exploration.
        """
        self.env = env
        # Use observation dimension based on node info. For instance, we use free CPU frequency per node.
        # self.n_observations = 3*len(env.scenario.get_nodes())-2
        self.n_observations = len(env.scenario.get_nodes())
        self.num_actions = len(env.scenario.node_id2name)
        self.gamma = gamma
        self.epsilon = epsilon

        self.model = Transformer(d_in=3, d_pos=self.n_observations, d_model=d_model, d_ff=d_model, n_heads=1, n_layers=3, dropout=dropout).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Replay buffer for transitions
        self.replay_buffer = []
        
    def _make_observation(self, env, task):
        """
        Returns a flat observation vector.
        For instance, we return the free CPU frequency for each node.
        """
        cpu_obs = {node_name: env.scenario.get_node(node_name).free_cpu_freq 
               for node_name in env.scenario.get_nodes()}
        # print(env.scenario.get_links())
        bw_obs ={link_name:env.scenario.get_link(link_name[0], link_name[1]).free_bandwidth for link_name in env.scenario.get_links()}
        
        # print(cpu_obs)
        # print(bw_obs)
        
        obs = np.zeros((len(env.scenario.get_nodes()), 3))
        
        for i, node_name in enumerate(env.scenario.get_nodes()):
            obs[env.scenario.node_name2id[node_name], 0] = cpu_obs[node_name]

        for i, link_name in enumerate(env.scenario.get_links()):
            
            if link_name[0] == 'e0':
                obs[env.scenario.node_name2id[link_name[1]], 1] = bw_obs[link_name]
            else:
                obs[env.scenario.node_name2id[link_name[0]], 2] = bw_obs[link_name]
            
        return obs

    def act(self, env, task, eval=False):
        """
        Chooses an action using ε-greedy strategy and records the current state.
        """
        state = self._make_observation(env, task)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        if random.random() < self.epsilon and not eval:
            action = random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                q_values = self.model(state_tensor)
                
                action = torch.argmax(q_values, dim=1).item()

        # Return both the action and the state (to form the transition later).
        return action, state

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay buffer.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def update(self):
        """
        Performs an update over all stored transitions and clears the buffer.
        """
        if not self.replay_buffer:
            return 0.0
        
        loss_total = 0.0
        self.optimizer.zero_grad()
        for state, action, reward, next_state, done in self.replay_buffer:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)
            done_tensor = torch.tensor([done], dtype=torch.float32).to(device)
            
            q_values = self.model(state_tensor)
            predicted_q = q_values[0, action]
            with torch.no_grad():
                next_q_values = self.model(next_state_tensor)
                max_next_q = torch.max(next_q_values)
                target_q = reward_tensor + (1 - done_tensor) * self.gamma * max_next_q
            loss = self.criterion(predicted_q, target_q)
            loss_total += loss
        loss_total.backward()
        self.optimizer.step()
        self.replay_buffer.clear()
        return loss_total.item()
    
    def update_epsilon(self, factor):
        self.epsilon *= factor
        print(f"New epsilon: {self.epsilon}")
