import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class DRLPolicy:
    def __init__(self,  env, lr=1e-3, hidden_size=128*2):
        """
        A simple REINFORCE-style policy for demonstration.

        Args:
            n_observations (int): Dimension of the observation vector
            env (Env): Custom environment that contains scenario and other metadata
            lr (float): Learning rate for the optimizer
        """
        self.n_observations = len(env.scenario.get_nodes())
        self.env = env
        # Assume each node in the environment is a valid action choice.
        self.num_actions = len(env.scenario.node_id2name)

        # Simple feed-forward network:
        # Input -> Hidden (128 units) -> Output (num_actions)
        self.model = nn.Sequential(
            nn.Linear(self.n_observations, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, self.num_actions),
            nn.Softmax(dim=-1)
        )
        # self.model = nn.Sequential(
        #     nn.Linear(n_observations, self.num_actions),
        #     nn.Softmax(dim=-1)
        # )

        # Adam optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        # We will store log probabilities of each action taken
        self.log_probs = []
        
        self.probs = []

    def _make_observation(self, env, task):
        """
        Extract the relevant fields from `task` into a flat observation vector.

        For example, if your 'header' is:
           [TaskSize, CyclesPerBit, TransBitRate, DDL]
        then just return them in a Python list (convert to tensor inside `act`).

        Args:
            task (Task): A task object containing task-specific properties.

        Returns:
            list: A list of floats that represent the observation.
        """
        
        task_obs = [
            float(task.task_size),
            float(task.cycles_per_bit),
            float(task.trans_bit_rate),
            float(task.ddl)
        ]
        
        node_task_ids = [env.scenario.get_node(node_id).active_task_ids[:] for node_id in env.scenario.get_nodes()]
        node_buffer_task_ids = [env.scenario.get_node(node_id).task_buffer.task_ids for node_id in env.scenario.get_nodes()]
        node_obs = [len(node_task_ids[i]) + len(node_buffer_task_ids[i]) for i in range(len(node_task_ids))]
        
        # node_obs = [self.env.scenario.get_node(node_name).free_cpu_freq for node_name in self.env.scenario.get_nodes()]
            
        
        return node_obs

    def act(self, env, task):
        """
        Choose an action based on the current policy (forward pass through the net).

        Args:
            env (Env): The environment (not always needed directly, but shown for reference).
            task (Task): The current task to be scheduled/offloaded.

        Returns:
            int: The selected action (index of the chosen node).
        """
        # Convert observation to a PyTorch tensor
        obs = torch.tensor(self._make_observation(env, task), dtype=torch.float32).unsqueeze(0)
        
        # print(obs)

        
        # print(obs)
        
        # Get action probabilities from the policy network
        probs = self.model(obs)
        
        self.probs.append(probs)
        
        # print(probs)
        
        
        distribution = Categorical(probs)
        

        # Sample an action
        action = distribution.sample()
        
        
        # print(distribution.log_prob(action))

        # Store the log probability for the policy gradient update
        self.log_probs.append(distribution.log_prob(action))

        # Convert the sampled action to a Python integer
        return action.item()

    def backward(self, reward):
        """
        Update the policy parameters given the reward signal.
        Here we do a very naive REINFORCE update: multiply each log_prob by the reward.

        Args:
            reward (float): The reward signal from the episode/evaluation.
                           In your example, this is the average latency (or any metric).
        """
        # Usually you want to *maximize* reward, but if your metric is something like
        # "average latency" (lower is better), you might invert it or adjust the sign.
        # For simplicity, we assume higher reward is better here.
        
        # Combine all log probabilities from the tasks in the current "episode"
        policy_loss = []
        for log_prob in self.log_probs:
            # REINFORCE uses: -log_prob * reward (we do negative because we want to maximize)
            policy_loss.append(log_prob * reward)

        self.optimizer.zero_grad()
        # print(torch.cat(policy_loss))
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        # Clear the log probabilities after each epoch (or episode)
        self.log_probs = []
