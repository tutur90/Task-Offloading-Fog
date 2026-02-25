import torch
import torch.nn as nn
from policies.model.base_model import BaseModel
from policies.model.modules.transformer import LearnedPositionalEncoding
from policies.model.modules.noebert import NeoBERT, NeoBERTConfig


class NATE(BaseModel):
    def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None, n_heads=4, n_layers=3, dropout=0.1, mode="mixed", **kwargs):
        super().__init__()

        self.nodes_embed = nn.Linear(d_in, d_model)
        self.pos_nodes_embed = LearnedPositionalEncoding(max_seq_len=d_pos, d_model=d_model)
        self.transformer_encoder = NeoBERT(NeoBERTConfig(
            hidden_size=d_model,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=d_ff if d_ff is not None else d_model * mlp_ratio,
            dropout=dropout,
        ))
        self.fc = nn.Linear(d_model, 1)


    def _forward(self, nodes, task):
        x = self.nodes_embed(nodes)
        x = self.pos_nodes_embed(x)
        x, _, _ = self.transformer_encoder(inputs_embeds=x)
        x = self.fc(x)
        return x


class TNATE(NATE):
    def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None, n_heads=4, n_layers=3, dropout=0.1, **kwargs):
        super().__init__(d_in=d_in+d_task, d_pos=d_pos, d_task=d_task, d_model=d_model, mlp_ratio=mlp_ratio, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout, mode="task", **kwargs)
        
    def _forward(self, nodes, task):
        x = self.nodes_embed(torch.cat([nodes, task], dim=-1))
        x = self.pos_nodes_embed(x)
        x, _, _ = self.transformer_encoder(inputs_embeds=x)
        x = self.fc(x)
        return x
