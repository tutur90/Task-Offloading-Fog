import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# Define an actor–critic network.
class PPOActorCritic(nn.Module):
    def __init__(self, dim_in, hidden_size, num_actions):
        super(PPOActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(dim_in, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.shared(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

class PPOPolicy:
    def __init__(self, env, lr=1e-3, hidden_size=128, gamma=0.99,
                 clip_param=0.2, lambda_=0.95, epochs=10, batch_size=256):
        """
        A Proximal Policy Optimization (PPO) policy.

        Args:
            env: The simulation environment.
            lr: Learning rate.
            hidden_size: Hidden layer size.
            gamma: Discount factor.
            clip_param: Clipping parameter for PPO.
            lambda_: GAE lambda parameter.
            epochs: Number of update epochs per PPO update.
            batch_size: Batch size (not used explicitly here; buffer size could be limited if desired).
        """
        self.env = env
        self.gamma = gamma
        self.clip_param = clip_param
        self.lambda_ = lambda_
        self.epochs = epochs
        self.batch_size = batch_size

        # Define observation and action dimensions based on the environment.
        self.n_observations = len(env.scenario.get_nodes())
        self.num_actions = len(env.scenario.node_id2name)
        
        self.network = PPOActorCritic(dim_in=self.n_observations,
                                      hidden_size=hidden_size,
                                      num_actions=self.num_actions)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Buffer to store transitions: (state, action, reward, done, log_prob, value)
        self.buffer = []
        
    def _make_observation(self, env, task):
        """
        Returns a flat observation vector.
        Here, we use the free CPU frequency for each node.
        """
        cpu_obs = [env.scenario.get_node(node_name).free_cpu_freq
                   for node_name in env.scenario.get_nodes()]
        # Additional features (e.g., bandwidth) can be added if needed.
        obs = cpu_obs
        return obs
    
    def act(self, env, task):
        """
        Chooses an action by sampling from the policy distribution.
        Returns:
            action: Chosen action index.
            state: Current observation.
            log_prob: Log probability of the selected action.
            value: Value estimate for the state.
        """
        state = self._make_observation(env, task)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, value = self.network(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action))
        return action, state, log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """
        Stores a transition for later PPO updates.
        """
        self.buffer.append((state, action, reward, done, log_prob, value))
    
    def update(self):
        """
        Updates the policy network using PPO's clipped objective.
        """
        if len(self.buffer) == 0:
            return 0.0
        
        # Unpack buffer into separate lists.
        states, actions, rewards, dones, old_log_probs, values = zip(*self.buffer)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = list(rewards)
        dones = list(dones)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        
        # Compute discounted returns.
        returns = []
        G = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Compute advantages.
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update loop.
        for _ in range(self.epochs):
            logits, new_values = self.network(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate objective.
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss.
            value_loss = (new_values.squeeze() - returns).pow(2).mean()
            
            loss = policy_loss + 0.5 * value_loss  # 0.5 is a common coefficient for value loss.
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.buffer = []  # Clear buffer after update.
        return loss.item()
