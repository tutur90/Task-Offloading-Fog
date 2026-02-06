import torch
import torch.nn as nn
import torch.optim as optim
from policies.dql_policy import DQLPolicy

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
        self.register_buffer('norm', torch.tensor(norm).max(dim=0, keepdim=True).values.to(self.device))  # Register the normalization factor as a buffer
        print(self.norm)


class MLPPolicy(DQLPolicy):
        
    def _init_model(self, env, config):
        self.model = MLP(d_in=self.d_obs, d_pos=self.n_observations, d_task=4, **config["model"]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        self.model.register_norm(self._make_observation(env, None, self.obs_type)[0])  # Register the normalization factor for latency


