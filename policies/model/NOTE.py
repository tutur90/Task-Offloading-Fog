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



class NOTE(nn.Module):
    def __init__(self, d_in, d_pos, d_task, d_model=8, d_ff=8, n_heads=1, n_layers=1, dropout=0.1, mode="mixed", **kwargs):
        super().__init__()

        
        self.nodes_embed = nn.Linear(d_in, d_model)
        self.task_embed = nn.Linear(d_task, d_model, bias=False)
        self.pos_nodes_embed = nn.Parameter(torch.zeros(d_pos, d_model))
        self.trasformer_encoder = TransformerEncoder(d_model=d_model, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        self.fc = nn.Linear(d_model, 1)
        self.softmax = nn.Softmax(dim=1)


        self.mode = mode
        
        
    def forward(self, nodes, task, use_task=True):
        
        
        # nodes = nodes / self.norm
        
        
        nodes = self.nodes_embed(nodes)
        x = nodes + self.pos_nodes_embed 
        
        if (use_task and not self.mode == "node") or self.mode == "task":
            task = self.task_embed(task)
            x = x + task.unsqueeze(1).repeat(1, nodes.size(1), 1)
        
        x = self.trasformer_encoder(x, None)

        x = self.fc(x)
        return x
        
    def register_norm(self, norm, epsilon=1e-8):
        self.register_buffer('norm', torch.tensor(norm, dtype=torch.float32).max(dim=0, keepdim=True).values + epsilon)  # Register the normalization factor as a buffer
        print(self.norm)



