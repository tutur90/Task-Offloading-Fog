from collections import OrderedDict

import torch
from torch import nn
import numpy as np
from policies.model.modules.mlp import MLP


class BaseMLP(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_size):
        super(BaseMLP, self).__init__()
        
        self.mlp = MLP(in_features=dim_in, hidden_features=hidden_size, out_features=dim_out, num_layers=3)

    def forward(self, x):
        return self.mlp(x)
    

def _test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dim_in = 10
    dim_out = 5
    hidden_size = 32
    batch_size = 4
    
    model = BaseMLP(dim_in, dim_out, hidden_size).to(device)
    model.train()
    
    x = torch.randn(batch_size, dim_in).to(device)
    y = model(x)
    
    assert y.shape == (batch_size, dim_out)
    print("Test passed.")


if __name__ == '__main__':
    _test()
