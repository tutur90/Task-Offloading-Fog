
import torch
from policies.dql.base_policy import DQNPolicy
from policies.model.NOTE import NOTE, DuelingNOTE
import copy
import torch.nn as nn
import torch.optim as optim


class NOTEPolicy(DQNPolicy):

    def _init_model(self, env, config, dataset=None):
        self.model = NOTE(d_in=self.d_obs, d_pos=self.n_observations, d_task=4, output_size=self.num_actions, **config["model"])
        self.model.register_norm(self._make_observation(env, None, self.obs_type)[0], dataset=dataset)
        self.model.to(self.device).to(self.dtype)

        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=config["training"].get("weight_decay", 0.01))
        self.criterion = nn.MSELoss()


class DuelingNOTEPolicy(DQNPolicy):

    def _init_model(self, env, config, dataset=None):
        self.model = DuelingNOTE(d_in=self.d_obs, d_pos=self.n_observations, d_task=4, output_size=self.num_actions, **config["model"])
        self.model.register_norm(self._make_observation(env, None, self.obs_type)[0], dataset=dataset)
        self.model.to(self.device).to(self.dtype)

        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=config["training"].get("weight_decay", 0.01))
        self.criterion = nn.MSELoss()




        
    