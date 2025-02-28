import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Define the shared actor-critic network.
class A2CActorCritic(nn.Module):
    def __init__(self, dim_in, hidden_size, num_actions):
        super(A2CActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(dim_in, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.shared(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

# A2C policy implementation.
class A2CPolicy:
    def __init__(self, env, lr=1e-3, hidden_size=128, gamma=0.99):
        """
        Args:
            env: The simulation environment.
            lr: Learning rate.
            hidden_size: Hidden layer size.
            gamma: Discount factor.
        """
        self.env = env
        self.gamma = gamma
        
        # Define observation and action dimensions based on the environment.
        self.n_observations = len(env.scenario.get_nodes())
        self.num_actions = len(env.scenario.node_id2name)
        
        # Initialize the actor-critic network and optimizer.
        self.network = A2CActorCritic(dim_in=self.n_observations, 
                                      hidden_size=hidden_size, 
                                      num_actions=self.num_actions)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Buffer to store transitions: (state, action, reward, next_state, done, log_prob, value)
        self.buffer = []
    
    def _make_observation(self, env, task):
        """
        Construct a flat observation vector (e.g., free CPU frequency for each node).
        """
        obs = [env.scenario.get_node(node).free_cpu_freq for node in env.scenario.get_nodes()]
        return obs

    def act(self, env, task):
        """
        Select an action based on the current observation using the actor-critic network.
        
        Returns:
            action: Chosen action index.
            state: Current observation.
            log_prob: Log probability of the selected action.
            value: State value estimate.
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
        Store a transition in the replay buffer.
        """
        self.buffer.append((state, action, reward, next_state, done, log_prob, value))
    
    def update(self):
        """
        Update the actor-critic network using the collected transitions.
        Computes discounted returns and advantages, then applies gradient descent.
        """
        if not self.buffer:
            return 0.0
        
        # Unpack buffer.
        states, actions, rewards, next_states, dones, log_probs, values = zip(*self.buffer)
        
        # Convert to tensors.
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        
        # Compute discounted returns.
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Advantage is the difference between returns and estimated values.
        advantages = returns - values
        
        # Forward pass on all states.
        logits, value_preds = self.network(states)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        
        # Calculate the policy loss (using the advantage as a weight).
        policy_loss = -(new_log_probs * advantages.detach()).mean()
        # Value loss as mean squared error.
        value_loss = (returns - value_preds.squeeze()).pow(2).mean()
        # Optional entropy bonus to encourage exploration.
        entropy_loss = -dist.entropy().mean()
        
        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.buffer.clear()
        return loss.item()
