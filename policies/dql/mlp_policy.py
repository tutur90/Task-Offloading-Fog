import copy
import torch
import torch.nn as nn
import torch.optim as optim
from policies.dql.base_policy import DQNPolicy
from policies.model.mlp import MLP


class MLPPolicy(DQNPolicy):

    def _init_model(self, env, config, dataset=None):
        self.model = MLP(d_in=self.d_obs, d_pos=self.n_observations, d_task=4, output_size=self.num_actions, **config["model"])
        self.model.register_norm(self._make_observation(env, None, self.obs_type)[0], dataset=dataset, device=self.device)
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.model.to(self.device).to(self.dtype)
        self.target_model.to(self.device).to(self.dtype)


