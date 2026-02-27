import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from policies.model.base_model import BaseModel
from policies.model.modules.transformer import LearnedPositionalEncoding
from policies.model.modules.noebert import NeoBERT, NeoBERTConfig


class ParallelLinear(nn.Module):
    def __init__(self, n, d_in, d_out, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(n, d_out)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # x: (B, N, d_in) -> (B, N, d_out)
        x = torch.einsum("bni,nio->bno", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

class NATE(BaseModel):
    def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None, n_heads=4, n_layers=3, dropout=0.1, qk_norm=True, learnable_qk_norm=True, embed="regular", **kwargs):
        super().__init__()

        self.embed = embed
        if embed == "regular":
            self.nodes_embed = nn.Linear(d_in, d_model)
            self.pos_nodes_embed = LearnedPositionalEncoding(max_seq_len=d_pos, d_model=d_model)
        elif embed == "parallel_linear":
            # Parallel linear: one weight matrix per node, batched as (d_pos, d_in, d_model)
            self.nodes_embed = ParallelLinear(d_pos, d_in, d_model)
        elif embed == "gelu":
            self.nodes_embed = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.GELU(),
                # nn.Linear(d_model, d_model),
            )
            self.pos_nodes_embed = LearnedPositionalEncoding(max_seq_len=d_pos, d_model=d_model)

        self.transformer_encoder = NeoBERT(NeoBERTConfig(
            hidden_size=d_model,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=d_ff if d_ff is not None else d_model * mlp_ratio,
            dropout=dropout,
            qk_norm=qk_norm,
            learnable_qk_norm=learnable_qk_norm,
        ))
        self.fc = nn.Linear(d_model, 1)
        
    def _emded_forward(self, x):
        if self.embed == "regular":
            x = self.pos_nodes_embed(self.nodes_embed(x))
        elif self.embed == "gelu":
            x = self.pos_nodes_embed(self.nodes_embed(x))

        else:   
            # nodes: (B, N, d_in), parallel_weight: (N, d_in, d_model) -> (B, N, d_model)

            x = self.nodes_embed(x)
        return x

    def _forward(self, nodes, task=None):
        x = self._emded_forward(nodes)
        x, _, _ = self.transformer_encoder(inputs_embeds=x)
        x = self.fc(x)
        return x


class TNATE(NATE):
    def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None, n_heads=4, n_layers=3, dropout=0.1, **kwargs):
        super().__init__(d_in=d_in+d_task, d_pos=d_pos, d_task=d_task, d_model=d_model, mlp_ratio=mlp_ratio, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout, **kwargs)
        
    def _forward(self, nodes, task):
        x = torch.cat([nodes, task.unsqueeze(1).repeat(1, nodes.shape[1], 1)], dim=-1)
        x = self._emded_forward(x)
        x, _, _ = self.transformer_encoder(inputs_embeds=x)
        x = self.fc(x)
        return x

        
# class TNATE(NATE):
#     def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None, n_heads=4, n_layers=3, dropout=0.1, **kwargs):
#         super().__init__(d_in=d_in, d_pos=d_pos+1, d_task=d_task, d_model=d_model, mlp_ratio=mlp_ratio, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout, **kwargs)
#         self.nodes_embed = nn.Linear(d_in, d_model)
#         self.task_embed = nn.Linear(d_task, d_model)
        
#     def _forward(self, nodes, task):
#         x = torch.cat([self.nodes_embed(nodes), self.task_embed(task.unsqueeze(1))], dim=1)
#         x = self.pos_nodes_embed(x)
#         x, _, _ = self.transformer_encoder(inputs_embeds=x)
#         x = self.fc(x[:, :-1, :])
#         return x
