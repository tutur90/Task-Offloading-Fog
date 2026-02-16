from policies.model.transformer_encoder.encoder import TransformerEncoder
from policies.model.transformer_encoder.multi_head_attention import MultiHeadAttention
import torch
import torch.nn as nn
import torch.functional as F


import random
import math


class AttentionWeights(nn.Module):
    def __init__(self, d_model=8, d_proj=8, dropout=0.1, bias=True):
        super().__init__()
        
        self.query = nn.Linear(d_model, d_proj, bias=bias)
        self.key = nn.Linear(d_model, d_proj, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key):
        """
        Args:
            `query`: shape (batch_size, max_len, d_model)
            `key`: shape (batch_size, max_len, d_model)
        """
        q = self.dropout(self.query(query))
        
        k = self.dropout(self.key(key))
        x = self.scale_dot_product_attention(q, k)
        
        return x
    
    def scale_dot_product_attention(self, query, key):
        """
        Args:
            `query`: shape (batch_size,  max_len, d_q)
            `key`: shape (batch_size, max_len, d_k)

        Returns:
            `p_attn`: shape (batch_size, max_len)

        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = self.softmax(scores)
        return p_attn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.embedding(positions)
    
    
class NOTE(nn.Module):
    def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None,  n_heads=4, n_layers=3, dropout=0.1, mode="mixed", **kwargs):
        super().__init__()

        
        self.nodes_embed = nn.Linear(d_in, d_model)
        self.task_embed = nn.Linear(d_task, d_model, bias=False)
        self.pos_nodes_embed = LearnedPositionalEncoding(max_seq_len=d_pos, d_model=d_model)
        # self.trasformer_encoder = TransformerEncoder(d_model=d_model, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout)

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
        self.softmax = nn.Softmax(dim=1)


        self.mode = mode
        
        
    def forward(self, nodes, task, use_task=True):

        nodes = nodes / self.norm
        
        x = self.nodes_embed(nodes)
        
        x = self.pos_nodes_embed(x)
        
        if (use_task and not self.mode == "node") or self.mode == "task":
            task = self.task_embed(task)
            x = x + task.unsqueeze(1).repeat(1, nodes.size(1), 1)
        
        x = self.transformer_encoder(x, is_causal=False)

        x = self.fc(x)
        return x
        
    def register_norm(self, norm, device, epsilon=1e-8):
        self.register_buffer('norm', torch.tensor(norm, dtype=torch.float32, device=device).max(dim=0, keepdim=True).values + epsilon)  # Register the normalization factor as a buffer
        self.norm = self.norm.to(device)




