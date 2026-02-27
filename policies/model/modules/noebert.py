import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


# --- SwiGLU ---

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=True):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# --- Config ---

@dataclass
class NeoBERTConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    embedding_init_range: float = 0.02
    decoder_init_range: float = 0.02
    norm_eps: float = 1e-6
    dropout: float = 0.0
    qk_norm: bool = True
    learnable_qk_norm: bool = True

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.dim_head = self.hidden_size // self.num_attention_heads

def rmsnorm(x, eps):
    def _norm(y):
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)

    return _norm(x.float()).type_as(x)
# --- Encoder Block ---

class EncoderBlock(nn.Module):
    def __init__(self, config: NeoBERTConfig):
        super().__init__()
        self.config = config

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        multiple_of = 8
        intermediate_size = int(2 * config.intermediate_size / 3)
        intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
        self.ffn = SwiGLU(config.hidden_size, intermediate_size, config.hidden_size, bias=False)

        self.attention_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, attention_mask, output_attentions, max_seqlen=None, cu_seqlens=None):
        attn_output, attn_weights = self._att_block(
            self.attention_norm(x), attention_mask, output_attentions, max_seqlen, cu_seqlens
        )
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, attn_weights

    def _att_block(self, x, attention_mask, output_attentions, max_seqlen=None, cu_seqlens=None):
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = (
            self.qkv(x)
            .view(batch_size, seq_len, self.config.num_attention_heads, self.config.dim_head * 3)
            .chunk(3, dim=-1)
        )

        # QK norm
        
        if self.config.qk_norm:
            xq = rmsnorm(xq, self.config.norm_eps)
            xk = rmsnorm(xk, self.config.norm_eps)


        attn_weights = None
        if cu_seqlens is not None:
            attn = flash_attn_varlen_func(
                q=xq.squeeze(0), k=xk.squeeze(0), v=xv.squeeze(0),
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                dropout_p=0.0, causal=False,
            )
        elif output_attentions:
            scale = xq.size(-1) ** -0.5
            attn_weights = xq.permute(0, 2, 1, 3) @ xk.permute(0, 2, 3, 1) * scale
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = attn_weights.softmax(-1)
            attn = (attn_weights @ xv.permute(0, 2, 1, 3)).transpose(1, 2)
        else:
            attn = F.scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=attention_mask,
                dropout_p=0.0,
            ).transpose(1, 2)

        return self.wo(attn.reshape(batch_size, seq_len, self.config.hidden_size)), attn_weights


# --- NeoBERT ---

class NeoBERT(nn.Module):
    def __init__(self, config: NeoBERTConfig):
        super().__init__()
        self.config = config

        self.transformer_encoder = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.embedding_init_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Scale output projections by depth (GPT-2 / NeoBERT style)
        std = self.config.decoder_init_range / (2 * self.config.num_hidden_layers) ** 0.5
        for layer in self.transformer_encoder:
            nn.init.normal_(layer.wo.weight, mean=0.0, std=std)
            nn.init.normal_(layer.ffn.w3.weight, mean=0.0, std=std)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        x = inputs_embeds
        

        hidden_states, attentions = [], []
        for layer in self.transformer_encoder:
            x, attn = layer(x, attention_mask, output_attentions)
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions:
                attentions.append(attn)

        x = self.layer_norm(x)
        return x, hidden_states or None, attentions or None
