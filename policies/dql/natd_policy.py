
import torch
from policies.dql.base_policy import DQNPolicy
from policies.model.natd import NATD
from policies.model.t_nate import TNATE
import copy
import torch.nn as nn
import torch.optim as optim


class NATDPolicy(DQNPolicy):

    def _init_model(self, env, config, dataset=None):

        self.model = NATD(d_in=self.d_obs, d_pos=self.n_observations, d_task=4, output_size=self.num_actions, **config["model"])

        
        self.model.register_norm(self._make_observation(env, None, self.obs_type)[0], dataset=dataset)
        self.model.to(self.device).to(self.dtype)






        
    