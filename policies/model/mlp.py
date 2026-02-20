import torch
from torch import nn
from policies.model.base_model import BaseModel

class MLP(BaseModel):
    def __init__(self, d_in, d_pos,  d_model, output_size, n_layers=2, dropout=0.2,  bias=True, obs_type=None, **kwargs):
        super(MLP, self).__init__()

        input_size = d_in * d_pos if "task" not in obs_type else d_in * d_pos + 4

        self.obs_type = obs_type

        if n_layers < 2:
            raise ValueError("The number of layers must be at least 2.")
        layers = [nn.Linear(input_size, d_model, bias=bias), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(d_model, d_model, bias=bias), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(d_model, output_size))
        self.model = nn.Sequential(*layers)


    def _forward(self, x, task):

        x = x.view(x.size(0), -1)


        if task is not None and "task" in self.obs_type:
            task = task.view(task.size(0), -1)
            x = torch.cat([x, task], dim=1)


        return self.model(x)


class DuelingMLP(BaseModel):
    """Dueling Network architecture: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))."""

    def __init__(self, d_in, d_pos, d_model, output_size, n_layers=2, dropout=0.2, bias=True, obs_type=None, **kwargs):
        super(DuelingMLP, self).__init__()

        input_size = d_in * d_pos if "task" not in obs_type else d_in * d_pos + 4

        self.obs_type = obs_type

        if n_layers < 2:
            raise ValueError("The number of layers must be at least 2.")

        shared_layers = [nn.Linear(input_size, d_model, bias=bias), nn.ReLU()]
        for _ in range(n_layers - 2):
            shared_layers += [nn.Linear(d_model, d_model, bias=bias), nn.ReLU(), nn.Dropout(dropout)]
        self.shared = nn.Sequential(*shared_layers)

        self.value_stream = nn.Linear(d_model, 1, bias=bias)
        self.advantage_stream = nn.Linear(d_model, output_size, bias=bias)

    def _forward(self, x, task):
        x = x.view(x.size(0), -1)

        if task is not None and "task" in self.obs_type:
            task = task.view(task.size(0), -1)
            x = torch.cat([x, task], dim=1)

        features = self.shared(x)
        value = self.value_stream(features)           # [batch, 1]
        advantage = self.advantage_stream(features)   # [batch, num_actions]
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
