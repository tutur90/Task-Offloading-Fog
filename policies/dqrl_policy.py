import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.distributions import Categorical  # (optional for epsilon random selection)
from core.task import Task

from policies.model.BaseMLP import BaseMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

class DQRLPolicy:
    def __init__(self, env, lr=1e-3, d_model=128, gamma=0.99, epsilon=0.1, incl_link_obs=True):
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
        if incl_link_obs:
            self.n_observations = 3*len(env.scenario.get_nodes())-2
        else:
            self.n_observations = len(env.scenario.get_nodes())
        self.num_actions = len(env.scenario.node_id2name)
        self.gamma = gamma
        self.epsilon = epsilon

        self.model = BaseMLP(dim_in=self.n_observations, dim_out=self.num_actions, hidden_size=d_model).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Replay buffer for transitions
        self.replay_buffer = []


    def _make_observation(self, env, task):
        """
        Returns a flat observation vector.
        For instance, we return the free CPU frequency for each node.
        """
        cpu_obs = [env.scenario.get_node(node_name).free_cpu_freq 
               for node_name in env.scenario.get_nodes()]
        # print(env.scenario.get_links())
        bw_obs = [env.scenario.get_link(link_name[0], link_name[1]).free_bandwidth
              for link_name in env.scenario.get_links()]
        
        # print(cpu_obs)
        # print(bw_obs)
        
        if self.n_observations == len(cpu_obs):
            obs = cpu_obs
        else:
            obs = cpu_obs + bw_obs
            
            
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
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        task_tensor = torch.tensor(task_obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        if random.random() < self.epsilon and not eval:
            action = random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                # The model now expects two inputs.
                q_values = self.model(state_tensor, task_tensor)
                action = torch.argmax(q_values, dim=1).item()

        # Return the chosen action along with the combined observation tuple.
        return action, (state, task_obs)

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
            obs, task_obs = state
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            task_tensor = torch.tensor(task_obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Unpack the next state tuple.
            next_obs, next_task_obs = next_state
            next_state_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
            next_task_tensor = torch.tensor(next_task_obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)
            done_tensor = torch.tensor([done], dtype=torch.float32).to(device)
            
            # Compute predicted Q-value for the taken action.
            q_values = self.model(state_tensor, task_tensor)
            predicted_q = q_values[0, action]
            with torch.no_grad():
                next_q_values = self.model(next_state_tensor, next_task_tensor)
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
        return self.epsilon