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
        
        self.query = nn.Linear(d_model, d_proj, bias=bias, dropout=dropout)
        self.key = nn.Linear(d_model, d_proj, bias=bias, dropout=dropout)

    def forward(self, query, key):
        """
        Args:
            `query`: shape (batch_size, max_len, d_model)
            `key`: shape (batch_size, max_len, d_model)
        """
        q = self.query(query)
        k = self.key(key)
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
        p_attn = F.softmax(scores, dim=-1)
        return p_attn


class TaskFormer(nn.Module):
    def __init__(self, d_in, d_pos, d_task, d_model=8, d_ff=8, n_heads=1, n_layers=1, dropout=0.1):
        super().__init__()

        
        self.nodes_embed = nn.Linear(d_in, d_model)
        self.task_embed = nn.Linear(d_task, d_model)
        self.pos_nodes_embed = nn.Parameter(torch.zeros(d_pos, d_model))
        self.trasformer_encoder = TransformerEncoder(d_model=d_model, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        # self.softmax = nn.Softmax(dim=1)
        self.attention = AttentionWeights(d_model=d_model, d_proj=d_model, dropout=dropout)
        
        
    def forward(self, nodes, task):
        task = self.task_embed(task)
        nodes = self.nodes_embed(nodes)
        nodes = nodes + self.pos_nodes_embed
        nodes = self.trasformer_encoder(nodes, None)
        
        x = self.attention(nodes, task)

        return x
        
    



