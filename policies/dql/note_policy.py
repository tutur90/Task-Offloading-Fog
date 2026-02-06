
import torch
from policies.dql_policy import DQLPolicy
from policies.dql.mlp_policy import MLPPolicy
from policies.model.NOTE import NOTE


class NOTEPolicy(DQLPolicy):
    def _init_model(self, env, config):
        self.model = NOTE(d_in=self.d_obs, d_pos=self.n_observations, d_task=4, output_size=self.num_actions, **config["model"]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()
        
        self.model.register_norm(self._make_observation(env, None, self.obs_type)[0])  # Register the normalization factor for latency
        
    