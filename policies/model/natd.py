
import torch
import torch.nn as nn
import torch.nn.functional as F
from policies.model.base_model import BaseModel
from policies.model.modules.transformer import TransformerEncoderLayer, LearnedPositionalEncoding

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional



    
class NATD(BaseModel):
    def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None,  n_heads=4, n_layers=3, dropout=0.1, mode="mixed", **kwargs):
        super().__init__()

        self.nodes_embed = nn.Linear(d_in, d_model)
        self.task_embed = nn.Linear(d_task, d_model)
        self.pos_nodes_embed = LearnedPositionalEncoding(max_seq_len=d_pos, d_model=d_model)
        # self.trasformer_encoder = TransformerEncoder(d_model=d_model, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout)

        self.transformer_encoder = nn.TransformerDecoder(
            encoder_layer=nn.TransformerDecoderLayer(
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
        x = self.nodes_embed(nodes)
        x = self.pos_nodes_embed(x)

        x = self.transformer_encoder(x, self.task_embed(task), tgt_is_causal=False)

        x = self.fc(x)
        return x


