import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from policies.model.base_model import BaseModel
from policies.model.modules.transformer import LearnedPositionalEncoding
from policies.model.modules.noebert import NeoBERT, NeoBERTConfig


class RelativeNodeEncoder(nn.Module):
    def __init__(self, d_in, d_model):
        super().__init__()
        # absolute + deviation + rank + distance to max + distance to min
        self.proj = nn.Linear(d_in * 5, d_model)
    
    def forward(self, nodes):
        # nodes: (B, N, d_in)
        mean = nodes.mean(dim=1, keepdim=True)
        diff = nodes - mean
        
        rank = nodes.argsort(dim=1).argsort(dim=1).float()
        rank = rank / (nodes.size(1) - 1)
        
        dist_to_max = nodes.max(dim=1, keepdim=True).values - nodes
        dist_to_min = nodes - nodes.min(dim=1, keepdim=True).values
        
        x = torch.cat([nodes, diff, rank, dist_to_max, dist_to_min], dim=-1)
        return self.proj(x)

class NATE(BaseModel):
    def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None, n_heads=4, n_layers=3, dropout=0.1, qk_norm=True, learnable_qk_norm=True, embed="regular", **kwargs):
        super().__init__()

        if embed == "regular":
            self.nodes_embed = nn.Linear(d_in, d_model)
        elif embed == "gelu":
            self.nodes_embed = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )
        elif embed == "relative":
            self.nodes_embed = RelativeNodeEncoder(d_in, d_model)
        elif embed == "no_bias":
            self.nodes_embed = nn.Linear(d_in, d_model, bias=False)
        else:
            raise ValueError(f"Unknown embed type: {embed}")

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
        

    def _forward(self, nodes, task=None):
        x = self.pos_nodes_embed(self.nodes_embed(nodes))
        x, _, _ = self.transformer_encoder(inputs_embeds=x)
        x = self.fc(x)
        return x
    

class FiLMConditioner(nn.Module):
    """Task modulates node representations via learned scale + shift."""
    def __init__(self, d_task, d_model):
        super().__init__()
        self.gamma = nn.Linear(d_task, d_model)  # scale
        self.beta = nn.Linear(d_task, d_model)   # shift
        
    def forward(self, node_embeds, task_features):
        # task_features: (B, d_task)
        gamma = self.gamma(task_features).unsqueeze(1)  # (B, 1, d_model)
        beta = self.beta(task_features).unsqueeze(1)
        return gamma * node_embeds + beta
    
class AdditiveConditioner(nn.Module):
    """Task modulates node representations via learned additive embedding."""
    def __init__(self, d_task, d_model):
        super().__init__()
        self.task_embed = nn.Linear(d_task, d_model, bias=False)
        
    def forward(self, node_embeds, task_features):
        # task_features: (B, d_task)
        task_emb = self.task_embed(task_features).unsqueeze(1)  # (B, 1, d_model)
        return node_embeds + task_emb


class TNATE(NATE):
    def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None, n_heads=4, n_layers=3, dropout=0.1, conditioning="film", **kwargs):
        super().__init__(d_in=d_in, d_pos=d_pos, d_task=d_task, d_model=d_model, mlp_ratio=mlp_ratio, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout, **kwargs)
        
        if conditioning == "film":
            self.conditioner = FiLMConditioner(d_task, d_model)
        elif conditioning == "add":
            self.conditioner = AdditiveConditioner(d_task, d_model)
        else:
            raise ValueError(f"Unknown conditioning type: {conditioning}")

    def _forward(self, nodes, task):
        x = torch.cat([nodes, task.unsqueeze(1).repeat(1, nodes.shape[1], 1)], dim=-1)
        x = self.pos_nodes_embed(self.nodes_embed(x))
        x = self.conditioner(x, task)
        x, _, _ = self.transformer_encoder(inputs_embeds=x)
        x = self.fc(x)
        return x

