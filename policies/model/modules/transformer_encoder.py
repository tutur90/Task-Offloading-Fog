import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# Transformer Encoder
#
# - Pre-norm with nn.RMSNorm
# - Residual connection type controlled by `residual_type`:
#     "gru"     — GRU-style gated residual (GTrXL), gate biased near 0 at init
#     "sigmoid" — simple sigmoid gating
#     "plain"   — standard additive residual
# - Optional QK normalization (learnable or fixed)
# - Optional attention bypass (FFN-only blocks)
# - GELU activations throughout
# =============================================================================

class GRUGating(nn.Module):
    """GRU-style gated residual (GTrXL formulation).
    Fused r/z projections: 3 matmuls instead of 6.
    z-gate bias init ensures output ≈ residual at start.
    ReLU applied to sublayer output before gating (Eq. 6, 8)."""
    def __init__(self, d_model, init_bias=-2.0):
        super().__init__()
        self.W_rz = nn.Linear(2 * d_model, 2 * d_model)
        self.W_h  = nn.Linear(d_model, d_model, bias=False)
        self.U_h  = nn.Linear(d_model, d_model)
        nn.init.constant_(self.W_rz.bias[d_model:], init_bias)

    def forward(self, x, y):
        """x: residual (hidden state), y: sublayer output."""
        y = F.relu(y)
        rz = self.W_rz(torch.cat([y, x], dim=-1))
        r, z = rz.chunk(2, dim=-1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        h_tilde = torch.tanh(self.W_h(y) + self.U_h(r * x))
        return (1 - z) * x + z * h_tilde


class PlainResidual(nn.Module):
    """Standard additive residual connection: output = x + y."""
    def forward(self, x, y):
        return x + y

class SigmoidGating(nn.Module):
    def __init__(self, d_model, bg=2.0):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)
        nn.init.constant_(self.gate.bias, -bg)

    def forward(self, x, y):
        z = torch.sigmoid(self.gate(torch.cat([y, x], dim=-1)))
        return (1 - z) * x + z * y

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0,
                 qk_norm=True, learnable_qk_norm=True, softplus_attn=None):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.softplus_attn = softplus_attn

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = dropout

        # QK normalization (per-head RMSNorm on q and k)
        affine = learnable_qk_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.d_head, elementwise_affine=affine)
            self.k_norm = nn.RMSNorm(self.d_head, elementwise_affine=affine)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.qkv_proj(x).reshape(B, S, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)  # each (B, S, H, d_head)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # (B, H, S, d_head) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.softplus_attn:
            # Softplus attention: A = softplus(QKᵀ/√d, β), row-normalized
            scores = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
            scores = F.softplus(scores, beta=self.softplus_attn)
            scores = scores / scores.sum(dim=-1, keepdim=True) + 1e-6
            if self.training and self.attn_drop > 0.0:
                scores = F.dropout(scores, p=self.attn_drop)
            out = scores @ v
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop if self.training else 0.0,
            )
        out = out.transpose(1, 2).reshape(B, S, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """GELU FFN."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.down(self.drop(F.gelu(self.up(x))))


def _build_gate(residual_type: str, d_model: int) -> nn.Module:
    if residual_type == "gru":
        return GRUGating(d_model)
    elif residual_type == "sigmoid":
        return SigmoidGating(d_model)
    elif residual_type == "plain":
        return PlainResidual()
    else:
        raise ValueError(f"Unknown residual_type '{residual_type}'. Choose 'gru', 'sigmoid', or 'plain'.")


class TransformerEncoderBlock(nn.Module):
    """Single Transformer Encoder block: pre-RMSNorm, optional attention, FFN,
    and configurable residual connections (gru / sigmoid / plain)."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0,
                 qk_norm=True, learnable_qk_norm=True,
                 use_attention=True, residual_type="gru", softplus_attn=None):
        super().__init__()
        self.use_attention = use_attention

        if use_attention:
            self.attn_norm = nn.RMSNorm(d_model)
            self.attn = MultiHeadSelfAttention(
                d_model, n_heads, dropout,
                qk_norm=qk_norm, learnable_qk_norm=learnable_qk_norm,
                softplus_attn=softplus_attn
            )
            self.attn_gate = _build_gate(residual_type, d_model)

        self.ff_norm = nn.RMSNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ff_gate = _build_gate(residual_type, d_model)

    def forward(self, x):
        if self.use_attention:
            y = self.attn(self.attn_norm(x))
            x = self.attn_gate(x, y)

        y = self.ff(self.ff_norm(x))
        x = self.ff_gate(x, y)
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer Encoder blocks with final RMSNorm."""
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1,
                 qk_norm=True, learnable_qk_norm=True,
                 use_attention=True, residual_type="gru", softplus_attn=None):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model, n_heads, d_ff, dropout,
                qk_norm=qk_norm, learnable_qk_norm=learnable_qk_norm,
                use_attention=use_attention, residual_type=residual_type, softplus_attn=softplus_attn,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.RMSNorm(d_model)

    def forward(self, inputs_embeds):
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
