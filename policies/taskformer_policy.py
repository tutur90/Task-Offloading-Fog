import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.distributions import Categorical  # (optional for epsilon random selection)
from core.task import Task
import numpy as np


from policies.model.BaseMLP import BaseMLP
from policies.model.TaskFormer import TaskFormer

class TaskFormerPolicy:
    def __init__(self, env, d_model=64, n_head=8, n_layers=1, d_ff = 256, dropout=0.1, lr=1e-3, gamma=0.99, epsilon=0.1, mode="node"):
        """
        A simple deep Q-learning policy.

        Args:
            env: The simulation environment.
            lr: Learning rate.
            d_model: Size of hidden layers.
            gamma: Discount factor.
            epsilon: For ε-greedy exploration.
        """
        self.env = env
        # Use observation dimension based on node info. For instance, we use free CPU frequency per node.
        
        self.beta = 0.9

        self.d_pos = len(env.scenario.get_nodes())
        self.num_actions = len(env.scenario.node_id2name)
        self.gamma = gamma
        self.epsilon = epsilon

        self.model = TaskFormer(d_in=3, d_pos=self.d_pos, d_task=4, d_model=d_model, d_ff=d_ff, n_heads=n_head, n_layers=n_layers, dropout=dropout, mode=mode)
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
        
        obs = np.zeros((len(env.scenario.get_nodes()), 3))
        
        for i, node_name in enumerate(env.scenario.get_nodes()):
            obs[env.scenario.node_name2id[node_name], 0] = cpu_obs[node_name]

        for i, link_name in enumerate(env.scenario.get_links()):
            
            if link_name[0] == 'e0':
                obs[env.scenario.node_name2id[link_name[1]], 1] = bw_obs[link_name]
            else:
                obs[env.scenario.node_name2id[link_name[0]], 2] = bw_obs[link_name]
            
        task_obs = [
            task.task_size,
            task.cycles_per_bit,
            task.trans_bit_rate,
            task.ddl,
        ]
        
        return obs, task_obs



    def act(self, env, task, eval=False):
        """
        Chooses an action using an ε-greedy strategy. It uses both the environment
        observation and the task-specific observation to compute Q-values.
        """
        # Get the two-part observation.
        state, task_obs = self._make_observation(env, task)
        # Convert both parts into tensors.
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        task_tensor = torch.tensor(task_obs, dtype=torch.float32).unsqueeze(0)
        
        
        if random.random() < self.epsilon and not eval:
            action = random.randrange(self.num_actions)
            use_task = random.random() > self.beta
        else:
            with torch.no_grad():
                use_task = random.random() > self.beta or eval
                # The model now expects two inputs.
                q_values = self.model(state_tensor, task_tensor, use_task)
                action = torch.argmax(q_values, dim=1).item()

        # Return the chosen action along with the combined observation tuple.
        return action, (state, task_obs, use_task)

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay buffer.
        Here, both state and next_state are tuples: (env_observation, task_observation).
        """
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def update(self):
        """
        Processes all stored transitions, updates the model weights, and clears the replay buffer.
        """
        if not self.replay_buffer:
            return 0.0
        
        loss_total = 0.0
        self.optimizer.zero_grad()
        for state, action, reward, next_state, done in self.replay_buffer:
            # Unpack the current state tuple.
            obs, task_obs, use_task = state
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            task_tensor = torch.tensor(task_obs, dtype=torch.float32).unsqueeze(0)

            # Unpack the next state tuple.
            next_obs, next_task_obs, next_use_task = next_state
            next_state_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            next_task_tensor = torch.tensor(next_task_obs, dtype=torch.float32).unsqueeze(0)
            
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            done_tensor = torch.tensor([done], dtype=torch.float32)
            
            # Compute predicted Q-value for the taken action.
            q_values = self.model(state_tensor, task_tensor, use_task)
            predicted_q = q_values[0, action]
            with torch.no_grad():
                next_q_values = self.model(next_state_tensor, next_task_tensor, next_use_task)
                max_next_q = torch.max(next_q_values)
                # When done is False, (1-done_tensor)==1 and the future reward is considered.
                target_q = reward_tensor + (1 - done_tensor) * self.gamma * max_next_q
            loss = self.criterion(predicted_q, target_q)
            loss_total += loss
        loss_total.backward()
        self.optimizer.step()
        self.replay_buffer.clear()
        return loss_total.item()
    
    def update_epsilon(self, factor=0.99):
        """
        Applies a decay factor to the exploration rate ε.
        """
        self.epsilon *= factor
        self.beta *= 0.9
        return self.epsilon