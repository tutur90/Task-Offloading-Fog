import torch
from torch import nn
from policies.model.base_model import BaseModel

class MLP(BaseModel):   
    def __init__(self, d_in, d_pos,  d_model, output_size, n_layers=2, dropout=0.2,  bias=True, **kwargs):
        super(MLP, self).__init__()
        
        
        if n_layers < 2:
            raise ValueError("The number of layers must be at least 2.")
        layers = [nn.Linear(d_in*d_pos, d_model, bias=bias), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(d_model, d_model, bias=bias), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(d_model, output_size))
        self.model = nn.Sequential(*layers)
        

    def _forward(self, x, task):
        
        return self.model(x.view(x.size(0), -1))
