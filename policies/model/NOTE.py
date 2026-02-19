
import torch
import torch.nn as nn
import torch.nn.functional as F
from policies.model.base_model import BaseModel

from torch import Tensor
from torch.nn.modules.activation import _is_make_fx_tracing, _check_arg_device, _arg_requires_grad
from typing import Callable, Optional, Tuple

import random
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        factory_kwargs = {"device": device, "dtype": dtype}
        
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.batch_first = batch_first
        self.dropout = dropout
        self.scale = self.head_dim ** -0.5
        self.norm_eps = norm_eps

        self.q_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)


        self.attn_dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        # Normalize to (batch, seq, d_model)
        if not self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

        B, S, _ = query.shape
        T = key.shape[1]

        def project_and_split(proj, x, seq_len):
            # (B, S, d_model) -> (B, nhead, S, head_dim)
            return proj(x).view(B, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        q = project_and_split(self.q_proj, query, S)
        k = project_and_split(self.k_proj, key, T)
        v = project_and_split(self.v_proj, value, T)

        q = q / torch.sqrt(torch.sum(q ** 2 , dim=-1, keepdim=True) + self.norm_eps)
        k = k / torch.sqrt(torch.sum(k ** 2 , dim=-1, keepdim=True) + self.norm_eps)

        # Scaled dot-product attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # (B, nhead, S, T)

        if attn_mask is not None:
            # attn_mask: (S, T) or (B*nhead, S, T) or (B, nhead, S, T)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            # (B, T) -> (B, 1, 1, T)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        return (out, attn_weights.mean(dim=1)) if need_weights else (out, None)


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        
        super().__init__(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
        batch_first=batch_first,
        norm_first=norm_first,
        bias=bias,
        device=device,
        dtype=dtype, 
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )




class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.embedding(positions)
    
    
class NOTE(BaseModel):
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


    def _forward(self, nodes, task):

        x = self.nodes_embed(nodes)

        x = self.pos_nodes_embed(x)

        if self.mode == "task":
            task = self.task_embed(task)
            x = x + task.unsqueeze(1).repeat(1, nodes.size(1), 1)

        x = self.transformer_encoder(x, is_causal=False)

        x = self.fc(x)
        return x


class DuelingNOTE(BaseModel):
    """Dueling Network version of NOTE: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))."""

    def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None, n_heads=4, n_layers=3, dropout=0.1, mode="mixed", **kwargs):
        super().__init__()

        self.nodes_embed = nn.Linear(d_in, d_model)
        self.task_embed = nn.Linear(d_task, d_model, bias=False)
        self.pos_nodes_embed = LearnedPositionalEncoding(max_seq_len=d_pos, d_model=d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff if d_ff is not None else d_model * mlp_ratio,
                dropout=dropout,
                norm_first=True,
                batch_first=True,
                activation="gelu"
            ),
            num_layers=n_layers,
            mask_check=False,
            enable_nested_tensor=False
        )

        # Value stream: mean-pool over nodes -> scalar V(s)
        self.value_stream = nn.Linear(d_model, 1)
        # Advantage stream: per-node scalar A(s,a)
        self.advantage_stream = nn.Linear(d_model, 1)

        self.mode = mode

    def _forward(self, nodes, task):
        x = self.nodes_embed(nodes)
        x = self.pos_nodes_embed(x)

        if self.mode == "task":
            task = self.task_embed(task)
            x = x + task.unsqueeze(1).repeat(1, nodes.size(1), 1)

        x = self.transformer_encoder(x, is_causal=False)  # (batch, d_pos, d_model)

        value = self.value_stream(x.mean(dim=1, keepdim=True))  # (batch, 1, 1)
        advantage = self.advantage_stream(x)                     # (batch, d_pos, 1)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))



