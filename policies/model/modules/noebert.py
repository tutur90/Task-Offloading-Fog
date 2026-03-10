import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


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
    hard_scale_qk: bool = False
    sink_attn: bool = True
    use_attention: bool = True

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.dim_head = self.hidden_size // self.num_attention_heads


def rmsnorm(x, eps):
    def _norm(y):
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return _norm(x.float()).type_as(x)


# =============================================================================
# Per-layer conditioners
#
# Recommendations (for task-conditioned shallow encoders):
#
#   adaRMS      Best default. Natural with RMSNorm (no redundant shift),
#               fewer params than adaLN-Zero. 4 * d_model params/layer.
#
#   adaLN-Zero  Most expressive (scale + shift + gate). 6 * d_model params/layer.
#               Preferred for deeper models (8+) or complex task signals.
#
#   prefix      Lets the model "attend to" task info. Good for longer sequences
#               where the overhead is small. Adds to sequence length.
#
#   add         Least recommended. Modulates the full residual stream, unstable
#               for deeper models. Only viable for very shallow (1-2 layers).
# =============================================================================

class AdaLNZeroConditioner(nn.Module):
    """DiT-style: adaptive norm (scale + shift) + output gate per sub-layer.
    Produces (gamma1, beta1, alpha1, gamma2, beta2, alpha2).
    All init to zero -> each layer is identity at start of training."""
    def __init__(self, d_task, d_model):
        super().__init__()
        self.proj = nn.Linear(d_task, 6 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, task):
        return self.proj(task).unsqueeze(1).chunk(6, dim=-1)


class AdaRMSConditioner(nn.Module):
    """Adaptive RMSNorm: scale + gate per sub-layer, no shift.
    Natural complement to RMSNorm (which doesn't center).
    Produces (gamma1, alpha1, gamma2, alpha2).
    All init to zero -> identity at start."""
    def __init__(self, d_task, d_model):
        super().__init__()
        self.proj = nn.Linear(d_task, 4 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, task):
        return self.proj(task).unsqueeze(1).chunk(4, dim=-1)


class PerLayerAddConditioner(nn.Module):
    """Simple additive bias per layer. Init to zero -> no-op at start."""
    def __init__(self, d_task, d_model):
        super().__init__()
        self.proj = nn.Linear(d_task, d_model, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, task):
        return self.proj(task).unsqueeze(1)  # (B, 1, d_model)


class PrefixConditioner(nn.Module):
    """Prefix conditioning: project task features into prefix tokens
    prepended to the sequence. The model attends to them naturally."""
    def __init__(self, d_task, d_model, n_prefix=4):
        super().__init__()
        self.n_prefix = n_prefix
        self.d_model = d_model
        self.proj = nn.Linear(d_task, n_prefix * d_model)

    def forward(self, task):
        B = task.size(0)
        return self.proj(task).view(B, self.n_prefix, self.d_model)


PER_LAYER_CONDITIONER_REGISTRY = {
    "adaln_zero": AdaLNZeroConditioner,
    "adarms": AdaRMSConditioner,
    "add": PerLayerAddConditioner,
}


# --- Encoder Block ---

class EncoderBlock(nn.Module):
    def __init__(self, config: NeoBERTConfig):
        super().__init__()
        self.config = config

        if config.use_attention:
            self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
            self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.attention_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)
            if config.sink_attn:
                self.sink_logit = nn.Parameter(torch.zeros(1, config.num_attention_heads, 1, 1))

        # intermediate_size is used directly — LLaMA-style 2/3 scaling done once upstream
        self.ffn = SwiGLU(config.hidden_size, config.intermediate_size, config.hidden_size, bias=False)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None, output_attentions=False, modulation=None):
        """Forward with optional per-layer modulation.
        modulation: tuple from AdaLNZeroConditioner (len 6) or AdaRMSConditioner (len 4), or None."""
        if modulation is not None and len(modulation) == 6:
            return self._forward_adaln_zero(x, attention_mask, output_attentions, modulation)
        elif modulation is not None and len(modulation) == 4:
            return self._forward_adarms(x, attention_mask, output_attentions, modulation)
        else:
            return self._forward_standard(x, attention_mask, output_attentions)

    def _forward_standard(self, x, attention_mask, output_attentions):
        attn_w = None
        if self.config.use_attention:
            attn_out, attn_w = self._att_block(self.attention_norm(x), attention_mask, output_attentions)
            x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, attn_w

    def _forward_adaln_zero(self, x, attention_mask, output_attentions, modulation):
        g1, b1, a1, g2, b2, a2 = modulation
        attn_w = None
        if self.config.use_attention:
            attn_out, attn_w = self._att_block(
                g1 * self.attention_norm(x) + b1, attention_mask, output_attentions
            )
            x = x + a1 * self.dropout(attn_out)
        x = x + a2 * self.dropout(self.ffn(g2 * self.ffn_norm(x) + b2))
        return x, attn_w

    def _forward_adarms(self, x, attention_mask, output_attentions, modulation):
        g1, a1, g2, a2 = modulation
        attn_w = None
        if self.config.use_attention:
            attn_out, attn_w = self._att_block(
                g1 * self.attention_norm(x), attention_mask, output_attentions
            )
            x = x + a1 * self.dropout(attn_out)
        x = x + a2 * self.dropout(self.ffn(g2 * self.ffn_norm(x)))
        return x, attn_w

    def _att_block(self, x, attention_mask, output_attentions):
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = (
            self.qkv(x)
            .view(batch_size, seq_len, self.config.num_attention_heads, self.config.dim_head * 3)
            .chunk(3, dim=-1)
        )

        if self.config.qk_norm:
            xq = rmsnorm(xq, self.config.norm_eps)
            xk = rmsnorm(xk, self.config.norm_eps)
            
        scale = self.config.dim_head ** -1 if self.config.hard_scale_qk else (self.config.dim_head ** -0.5)

        xq_t = xq.transpose(1, 2)  # (B, H, S, D)
        xk_t = xk.transpose(1, 2)
        xv_t = xv.transpose(1, 2)

        scores = torch.matmul(xq_t, xk_t.transpose(-2, -1)) * scale

        if attention_mask is not None:
            scores = scores + attention_mask

        if self.config.sink_attn:
            sink = self.sink_logit.expand(scores.shape[0], -1, scores.shape[2], 1)
            scores = torch.cat([scores, sink], dim=-1)  # (B, H, S, S+1)

        attn_weights = F.softmax(scores, dim=-1)

        if self.config.sink_attn:
            attn_weights = attn_weights[..., :-1]  # drop sink column (B, H, S, S)

        attn = torch.matmul(attn_weights, xv_t).transpose(1, 2)

        if not output_attentions:
            attn_weights = None

        return self.wo(attn.reshape(batch_size, seq_len, self.config.hidden_size)), attn_weights


# --- NeoBERT (unconditioned) ---

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
            if self.config.use_attention:
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
        return x


# --- CNeoBERT (conditioned) ---

class CNeoBERT(NeoBERT):
    """NeoBERT with per-layer conditioning.

    Supports: adaln_zero, adarms, add, prefix.
    Conditioners handle their own zero/identity init, so NeoBERT._init_weights
    running first is fine — they re-init themselves in their __init__."""

    def __init__(self, config: NeoBERTConfig, d_condition: int,
                 per_layer_type: str = "adarms", n_prefix: int = 4):
        super().__init__(config)
        self.per_layer_type = per_layer_type

        if per_layer_type == "prefix":
            self.prefix_conditioner = PrefixConditioner(d_condition, config.hidden_size, n_prefix)
            self.layer_conditioners = None
        elif per_layer_type in PER_LAYER_CONDITIONER_REGISTRY:
            cls = PER_LAYER_CONDITIONER_REGISTRY[per_layer_type]
            self.layer_conditioners = nn.ModuleList([
                cls(d_condition, config.hidden_size)
                for _ in range(config.num_hidden_layers)
            ])
            self.prefix_conditioner = None
        else:
            raise ValueError(
                f"Unknown per_layer_type: '{per_layer_type}'. "
                f"Choose from: {list(PER_LAYER_CONDITIONER_REGISTRY.keys()) + ['prefix']}"
            )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        condition: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        x = inputs_embeds

        # Prefix: prepend task-derived tokens to sequence
        if self.per_layer_type == "prefix":
            prefix = self.prefix_conditioner(condition)
            x = torch.cat([prefix, x], dim=1)

        hidden_states, attentions = [], []
        for i, layer in enumerate(self.transformer_encoder):
            modulation = None

            if self.per_layer_type == "add":
                x = x + self.layer_conditioners[i](condition)
            elif self.per_layer_type in ("adaln_zero", "adarms"):
                modulation = self.layer_conditioners[i](condition)

            x, attn = layer(x, attention_mask, output_attentions, modulation=modulation)
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions:
                attentions.append(attn)

        x = self.layer_norm(x)

        # Prefix: strip prefix tokens from output
        if self.per_layer_type == "prefix":
            x = x[:, self.prefix_conditioner.n_prefix:]

        return x