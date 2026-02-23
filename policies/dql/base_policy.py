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
    def __init__(self, env: Env, config, dataset=None):
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

        self.lr = config["training"]["lr"]

        config["training"]["exploration"]["strategy"] = config["training"].get("exploration", {}).get("strategy", "epsilon_greedy")
        
        self.exploration_strategy = config["training"]["exploration"]["strategy"]
        
        if config["training"]["exploration"]["strategy"] == "epsilon_greedy":
            self.epsilon_start = config["training"]["exploration"].get("epsilon", 1.0)
            self.epsilon = self.epsilon_start
            self.epsilon_min = config["training"]["exploration"].get("epsilon_min", 0.01)
            self.epsilon_decay = config["training"]["exploration"].get("epsilon_decay", 0.3)
        elif config["training"]["exploration"]["strategy"] == "boltzmann":
            self.temperature = config["training"]["exploration"].get("temperature", 1.0)
            self.temperature_start = self.temperature
            self.temperature_min = config["training"]["exploration"].get("temperature_min", 0.1)
            self.temperature_decay = config["training"]["exploration"].get("temperature_decay", 0.5)
        elif config["training"]["exploration"]["strategy"] == "ucb":
            self.ucb_c = config["training"]["exploration"].get("ucb_c", 1.0)
        else: 
            raise ValueError(f"Unknown exploration strategy: {config['training']['exploration']['strategy']}")
        # Replay buffer for transitions.
        self.buffer_size = config["training"].get("buffer_size", 10000)
        self.batch_size = config["training"].get("batch_size", 64)
        self.target_update_freq = config["training"].get("target_update_freq", 1000)
        # target_update_freq >= 1 → hard update every N gradient steps
        # target_update_freq <  1 → treated as τ for soft (Polyak) update every gradient step
        self.soft_update = self.target_update_freq < 1
        self.tau = self.target_update_freq if self.soft_update else None
        self.update_freq = config["training"].get("update_freq", 1)
        self.learning_starts = config["training"].get("learning_starts", 0)
        self.warmup_ratio = config["training"].get("warmup", 0)
        self.warmup_steps = 0  # computed in set_training_steps()
        self.total_training_steps = None  # set by set_training_steps()
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.update_count = 0
        self.total_steps = 0
        self.action_counts = np.zeros(self.num_actions, dtype=np.float32)  # for UCB
        
        if config["device"] == "auto":

            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config["device"])
            
        print(f"Using device: {self.device}")
            
        self.dtype = torch.float32
        
        self.clip_grad_norm = config["training"].get("clip_grad_norm", float('inf'))
        
        if self.clip_grad_norm <= 0:
            self.clip_grad_norm = float('inf')
        
        _reward = config["training"].get("reward", {})
        self.reward_momentum = _reward.get("momentum") or 0.9
        self.reward_norm = _reward.get("norm", "standard")
        self.reward_eps = _reward.get("eps", 1e-6)
        self.reward_mean = config.get("eval", {}).get("lambda", [1.0] * 3)
        self.reward_var = [1.0] * 3
        self.reward_max = config.get("eval", {}).get("expected_values", [1.0] * 3)
        self.avg_reward = 0
        
        self.reward_clip = _reward.get("clip", 5.0)


        self._init_model(env, config, dataset=dataset)
        
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        
        config["training"]["optimizer"] = config["training"].get("optimizer", {})
        

        opt_type = config["training"]["optimizer"].get("type", "Adam")
        
        if opt_type == "Adam":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=config["training"]["optimizer"].get("weight_decay", 0), betas=(0.9, 0.999))
        elif opt_type == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=config["training"]["optimizer"].get("weight_decay", 0))
        elif opt_type == "RMSprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=config["training"]["optimizer"].get("weight_decay", 0))
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
        
        self.criterion = nn.MSELoss()
        
        self.avg_loss = 0
        self.avg_grad_norm = 0


    def _init_model(self, env: Env, config, dataset=None):
        self.model = MLP(d_in=self.d_obs, d_pos=self.n_observations, d_task=4, output_size=self.num_actions, **config["model"])
        self.model.register_norm(self._make_observation(env, None, self.obs_type)[0], dataset=dataset)  # Register the normalization factor for latency
        
    def norm_reward(self, reward, _lambda):
        # Normalize reward components based on the specified method
        
        reward = self._norm_reward(reward, _lambda)
        self.avg_reward = self.avg_reward * 0.999 + reward * (1 - 0.999)
        # print(f"Raw reward: {reward}, Avg reward: {self.avg_reward}")
        return reward 
    
    def adapt_coef(self, reward, _lambda):
        # Adaptively adjust lambda coefficients based on reward trends
        # For example, if reward is consistently low, increase the weight on TDR
        # This is a placeholder for a more sophisticated adaptation mechanism
        pass
    
    def _norm_reward(self, reward, _lambda):


        self.reward_mean = [self.reward_mean[i] * self.reward_momentum + reward[i] * (1 - self.reward_momentum) if reward[i] else self.reward_mean[i] for i in range(3)]  
        self.reward_var = [self.reward_var[i] * self.reward_momentum + (reward[i] - self.reward_mean[i]) ** 2 * (1 - self.reward_momentum) if reward[i] else self.reward_var[i] for i in range(3)]
        self.reward_max = [max(self.reward_max[i], reward[i]) if reward[i] else self.reward_max[i] for i in range(3)]
        
        if self.reward_norm == "standard":
            reward = [(reward[i] - self.reward_mean[i]) / (np.sqrt(self.reward_var[i]) + self.reward_eps) if reward[i] else 0 for i in range(3)]
        elif self.reward_norm == "mean":
            reward = [reward[i] / (self.reward_mean[i] + self.reward_eps) if reward[i] else 0 for i in range(3)]
        elif self.reward_norm == "partial_mean":
            reward = [reward[i] / (self.reward_mean[i] + self.reward_eps) if reward[i] and i != 0 else 0 for i in range(3)]
        elif self.reward_norm == "max":
            self.reward_max = [max(self.reward_max[i], reward[i]) for i in range(3)]
            reward = [reward[i] / (self.reward_max[i] + self.reward_eps) if reward[i] else 0 for i in range(3)]
        elif self.reward_norm == "log1p":
            reward = [np.log1p(reward[i]) if reward[i] else 0 for i in range(3)]
        elif self.reward_norm == "log1p_mean":
            reward = [np.log1p(reward[i]) - (np.log1p(self.reward_mean[i]) + self.reward_eps) if reward[i] else 0 for i in range(3)]
        else:
            reward = [reward[i] if reward[i] is not None else 0 for i in range(3)]
            
        if self.reward_clip is not None:
            reward = [np.clip(r, -self.reward_clip, self.reward_clip) for r in reward]
            
        return sum(_lambda[i] * reward[i] for i in range(3))
            

    def _make_observation(self, env: Env, task: Task, obs_type=["cpu", "buffer", "bw"]):
        """
        Returns a flat observation vector.
        For instance, we return the free CPU frequency for each node.
        """
        
        obs_type = list(set(obs_type) & set(["cpu", "buffer", "bw"]))  # Ensure obs_type is a list and remove duplicates
        
        
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
        self.warmup_steps = int(total_steps * self.warmup_ratio)

    def _update_epsilon(self):
        """Linearly decay exploration parameter over training."""
        if self.total_training_steps is None:
            return
        steps_since_learn = self.total_steps - self.learning_starts

        if self.exploration_strategy == "epsilon_greedy":
            # ε-greedy decay
            decay_steps = int(self.total_training_steps * self.epsilon_decay)
            if steps_since_learn <= 0:
                self.epsilon = self.epsilon_start
            elif steps_since_learn >= decay_steps:
                self.epsilon = self.epsilon_min
            else:
                self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_min) * (steps_since_learn / decay_steps)

        # Boltzmann temperature decay
        if self.exploration_strategy == "boltzmann":
            t_decay_steps = int(self.total_training_steps * self.temperature_decay)
            if steps_since_learn <= 0:
                self.temperature = self.temperature_start
            elif steps_since_learn >= t_decay_steps:
                self.temperature = self.temperature_min
            else:
                self.temperature = self.temperature_start - (self.temperature_start - self.temperature_min) * (steps_since_learn / t_decay_steps)

    def _update_lr(self):
        """Linear LR warmup from 0 to base lr over warmup_steps steps after learning starts."""
        if self.warmup_steps == 0:
            return
        steps_since_learn = self.total_steps - self.learning_starts
        if steps_since_learn <= 0:
            lr = 0.0
        elif steps_since_learn < self.warmup_steps:
            lr = self.lr * steps_since_learn / self.warmup_steps
        else:
            return  # warmup complete
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def act(self, env, task, train=True):
        """
        Chooses an action using the configured exploration strategy and records the current state.
        Supports: "epsilon_greedy", "boltzmann", "ucb".
        """
        state = self._make_observation(env, task, self.obs_type)
        obs, task_obs = state
        obs_tensor = torch.tensor(obs, dtype=self.dtype, device=self.device).unsqueeze(0)
        task_tensor = torch.tensor(task_obs, dtype=self.dtype, device=self.device).unsqueeze(0)

        if train:
            self.total_steps += 1
            self._update_epsilon()

        # Random warm-up phase regardless of strategy
        if train and self.total_steps <= self.learning_starts:
            action = random.randrange(self.num_actions)
            return action, state

        if self.exploration_strategy == "epsilon_greedy":
            if train and random.random() < self.epsilon:
                action = random.randrange(self.num_actions)
            else:
                with torch.no_grad():
                    self.model.eval()
                    q_values = self.model(obs_tensor, task_tensor)
                    action = torch.argmax(q_values, dim=1).item()

        elif self.exploration_strategy == "boltzmann":
            with torch.no_grad():
                self.model.eval()
                q_values = self.model(obs_tensor, task_tensor).squeeze()
            if train:
                # Sample from softmax(Q / temperature)
                probs = torch.softmax(q_values / self.temperature, dim=0).cpu().numpy()
                action = int(np.random.choice(self.num_actions, p=probs))
            else:
                action = int(torch.argmax(q_values).item())

        elif self.exploration_strategy == "ucb":
            with torch.no_grad():
                self.model.eval()
                q_values = self.model(obs_tensor, task_tensor).squeeze().cpu().numpy()
            if train:
                # UCB bonus: c * sqrt(log(t) / (1 + N(a)))
                bonus = self.ucb_c * np.sqrt(np.log(self.total_steps + 1) / (1 + self.action_counts))
                action = int(np.argmax(q_values + bonus))
                self.action_counts[action] += 1
            else:
                action = int(np.argmax(q_values))

        else:
            raise ValueError(f"Unknown exploration strategy: '{self.exploration_strategy}'. "
                             f"Choose from: 'epsilon_greedy', 'boltzmann', 'ucb'.")

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

        self._update_lr()

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

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
        
        self.optimizer.step()

        return loss.item(), grad_norm.item()    

    def update(self):
        """
        Performs an update over a sampled batch of transitions using batched operations,
        moves tensors to the appropriate device and dtype.
        """
        self.update_count += 1
        if len(self.replay_buffer) < self.batch_size or self.total_steps <= self.learning_starts:
            return 0.0, None
        
        if not self.soft_update and self.update_count % self.target_update_freq == 0:
            self.update_target_network()
            
        if self.update_count % self.update_freq == 0:
            loss, grad_norm = self._update()
            if self.soft_update:
                self.update_target_network()
            self.avg_loss = self.avg_loss * 0.999 + loss * 0.001
            self.avg_grad_norm = self.avg_grad_norm * 0.999 + grad_norm * 0.001 
            return loss, grad_norm


        


    def update_target_network(self):
        """
        Hard update (target_update_freq >= 1): θ_target ← θ_online every N gradient steps.
        Soft update (target_update_freq < 1):  θ_target ← τ·θ_online + (1−τ)·θ_target every gradient step,
                                               where τ = target_update_freq.
        """
        if self.soft_update:
            with torch.no_grad():
                for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                    target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
        else:
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

