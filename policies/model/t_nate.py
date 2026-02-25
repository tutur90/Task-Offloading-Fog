
import torch
import torch.nn as nn
import torch.nn.functional as F
from policies.model.base_model import BaseModel
from policies.model.modules.transformer import TransformerEncoderLayer, LearnedPositionalEncoding

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional



    
class TNATE(BaseModel):
    def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None,  n_heads=4, n_layers=3, dropout=0.1, **kwargs):
        super().__init__()

        self.embedding = nn.Linear(d_in + d_task, d_model)
        self.pos_nodes_embed = LearnedPositionalEncoding(max_seq_len=d_pos, d_model=d_model)


        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff if d_ff is not None else d_model*mlp_ratio,
                dropout=dropout,
                norm_first=True,
                batch_first=True,
                activation="gelu"
                ),
            num_layers=n_layers,
            mask_check=False,
            enable_nested_tensor=False
        )
        self.fc = nn.Linear(d_model, 1)


    def _forward(self, nodes, task):

        x = self.embedding(torch.cat([nodes, task], dim=-1))

        x = self.pos_nodes_embed(x)

        x = self.transformer_encoder(x, is_causal=False)

        x = self.fc(x)
        return x

