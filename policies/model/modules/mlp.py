import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.4, num_layers=2):
        """
        :param channel_first: if True, during forward the tensor shape is [B, C, T, J] and fc layers are performed with
                              1x1 convolutions.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        
        if num_layers <= 2:
            raise ValueError("num_layers must be greater than 2")
        else:
            self.first_layer = nn.Linear(in_features, hidden_features)
            self.hidden_layers = nn.ModuleList([nn.Linear(hidden_features, hidden_features) for _ in range(num_layers - 2)])
            self.last_layer = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.act(self.first_layer(x))
        
        for layer in self.hidden_layers:
            x = self.act(layer(x))
            x = self.drop(x)
            
        x = self.last_layer(x)
        
        return x
