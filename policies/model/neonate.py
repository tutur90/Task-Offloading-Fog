import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from policies.model.base_model import BaseModel
from policies.model.modules.transformer import LearnedPositionalEncoding
from policies.model.modules.noebert import NeoBERT, NeoBERTConfig, CNeoBERT, SwiGLU


torch.backends.cuda.matmul.allow_tf32 = True


# =============================================================================
# Utilities
# =============================================================================

def compute_intermediate_size(d_model, mlp_ratio, multiple_of=8):
    """LLaMA-style SwiGLU sizing: 2/3 expansion, rounded to multiple_of."""
    raw = int(2 * d_model * mlp_ratio / 3)
    return multiple_of * ((raw + multiple_of - 1) // multiple_of)


# =============================================================================
# Node Encoder
# =============================================================================

class RelativeNodeEncoder(nn.Module):
    def __init__(self, d_in, d_model):
        super().__init__()
        # absolute + deviation + rank + distance to max + distance to min
        self.proj = nn.Linear(d_in * 5, d_model)

    def forward(self, nodes):
        # nodes: (B, N, d_in)
        N = nodes.size(1)
        mean = nodes.mean(dim=1, keepdim=True)
        diff = nodes - mean

        rank = nodes.argsort(dim=1).argsort(dim=1).float()
        rank = rank / max(N - 1, 1)  # safe when N=1

        dist_to_max = nodes.max(dim=1, keepdim=True).values - nodes
        dist_to_min = nodes - nodes.min(dim=1, keepdim=True).values

        x = torch.cat([nodes, diff, rank, dist_to_max, dist_to_min], dim=-1)
        return self.proj(x)


# =============================================================================
# Pre-encoder conditioners (applied once to embeddings before the transformer)
#
# Recommendations:
#   film : best default — expressive (scale + shift), identity at init.
#   add  : simpler, fewer params, fine for shallow models.
# =============================================================================

class FiLMConditioner(nn.Module):
    """Scale + shift embeddings from task features. Identity init (gamma=1, beta=0)."""
    def __init__(self, d_task, d_model):
        super().__init__()
        self.gamma = nn.Linear(d_task, d_model)
        self.beta = nn.Linear(d_task, d_model)
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, node_embeds, task_features):
        gamma = self.gamma(task_features).unsqueeze(1)
        beta = self.beta(task_features).unsqueeze(1)
        return gamma * node_embeds + beta


class PreAdditiveConditioner(nn.Module):
    """Add task embedding to node embeddings. Zero init -> no-op at start."""
    def __init__(self, d_task, d_model):
        super().__init__()
        self.task_embed = nn.Linear(d_task, d_model, bias=False)
        nn.init.zeros_(self.task_embed.weight)

    def forward(self, node_embeds, task_features):
        task_emb = self.task_embed(task_features).unsqueeze(1)
        return node_embeds + task_emb


PRE_CONDITIONER_REGISTRY = {
    "film": FiLMConditioner,
    "add": PreAdditiveConditioner,
}


# =============================================================================
# MLPConditioner — fallback for n_layers=0 in TNATE
# =============================================================================

class MLPConditioner(nn.Module):
    """Lightweight conditioner wrapper for the n_layers=0 case."""
    def __init__(self, d_task, d_model, d_ff=None, pre_conditioning="film"):
        super().__init__()
        if pre_conditioning in PRE_CONDITIONER_REGISTRY:
            self.conditioner = PRE_CONDITIONER_REGISTRY[pre_conditioning](d_task, d_model)
        else:
            raise ValueError(f"Unknown pre_conditioning for MLPConditioner: {pre_conditioning}")

    def forward(self, inputs_embeds, condition):
        return self.conditioner(inputs_embeds, condition)


# =============================================================================
# NATE (unconditioned)
# =============================================================================

class NATE(BaseModel):
    def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None,
                 n_heads=4, n_layers=3, dropout=0.1, qk_norm=True,
                 learnable_qk_norm=True, embed="regular", d_head=None,
                 use_attention=True, sink_attn=True, **kwargs):
        super().__init__()
        if d_head is not None:
            n_heads = d_model // d_head

        self.nodes_embed = self._build_embed(embed, d_in, d_model, mlp_ratio)
        self.pos_nodes_embed = LearnedPositionalEncoding(max_seq_len=d_pos, d_model=d_model)
        self.fc = nn.Linear(d_model, 1)

        config = NeoBERTConfig(
            hidden_size=d_model,
            intermediate_size=d_ff if d_ff is not None else compute_intermediate_size(d_model, mlp_ratio),
            num_attention_heads=n_heads,
            num_hidden_layers=n_layers,
            dropout=dropout,
            qk_norm=qk_norm,
            use_attention=use_attention,
            sink_attn=sink_attn,
        )

        self._init_encoder(config, d_task=d_task)
        self._init_non_encoder_weights()

    @staticmethod
    def _build_embed(embed, d_in, d_model, mlp_ratio):
        if embed == "regular":
            return nn.Linear(d_in, d_model)
        elif embed == "gelu":
            return nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        elif embed == "relative":
            return RelativeNodeEncoder(d_in, d_model)
        elif embed == "no_bias":
            return nn.Linear(d_in, d_model, bias=False)
        elif embed == "ff":
            return SwiGLU(d_in, d_model * mlp_ratio, d_model)
        else:
            raise ValueError(f"Unknown embed type: {embed}")

    def _init_non_encoder_weights(self):
        """Init all modules NOT inside the transformer encoder.
        Covers nodes_embed, pos_nodes_embed, fc, and n_layers=0 fallback.
        Modules that self-init (conditioners inside CNeoBERT/MLPConditioner)
        are excluded to preserve their zero/identity init."""
        init_std = 0.02
        encoder_params = (
            set(self.transformer_encoder.parameters())
            if isinstance(self.transformer_encoder, (NeoBERT, CNeoBERT, MLPConditioner))
            else set()
        )
        for p in self.parameters():
            if p in encoder_params:
                continue
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=init_std)
            elif p.dim() == 1:
                nn.init.zeros_(p)

    def _init_encoder(self, config, d_task=None):
        if config.num_hidden_layers == 0:
            self.transformer_encoder = nn.Sequential(
                nn.RMSNorm(config.hidden_size, eps=1e-6),
                SwiGLU(config.hidden_size, config.intermediate_size, config.hidden_size),
                nn.RMSNorm(config.hidden_size, eps=1e-6),
            )
            return

        self.transformer_encoder = NeoBERT(config)

    def _forward(self, nodes, task=None):
        x = self.pos_nodes_embed(self.nodes_embed(nodes))
        x = self.transformer_encoder(inputs_embeds=x)
        x = self.fc(x)
        return x


# =============================================================================
# TNATE (task-conditioned)
#
# Two-stage conditioning:
#   1. Pre-encoder  (pre_conditioning):       "film", "add", "none"
#   2. Per-layer    (per_layer_conditioning):  "none", "adarms", "adaln_zero", "add", "prefix"
#
# Recommended combos:
#   Shallow (2-4 layers): pre=film, per_layer=none         — simplest, sufficient
#   Medium  (4-8 layers): pre=film, per_layer=adarms       — best balance
#   Deep    (8+ layers):  pre=add,  per_layer=adaln_zero   — most expressive
# =============================================================================

class TNATE(NATE):
    def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None,
                 n_heads=4, n_layers=3, dropout=0.1,
                 pre_conditioning="film", per_layer_conditioning="none",
                 n_prefix=4, d_head=None, use_attention=True, sink_attn=True, **kwargs):
        # Store before super().__init__ because _init_encoder reads them
        self.pre_conditioning = pre_conditioning
        self.per_layer_conditioning = per_layer_conditioning
        self.n_prefix = n_prefix

        if d_head is not None:
            n_heads = d_model // d_head

        super().__init__(
            d_in=d_in, d_pos=d_pos, d_task=d_task, d_model=d_model,
            mlp_ratio=mlp_ratio, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers,
            dropout=dropout, use_attention=use_attention, sink_attn=sink_attn, **kwargs,
        )

        # Pre-encoder conditioner (separate from transformer encoder)
        if pre_conditioning in PRE_CONDITIONER_REGISTRY:
            self.pre_conditioner = PRE_CONDITIONER_REGISTRY[pre_conditioning](d_task, d_model)
        elif pre_conditioning == "none":
            self.pre_conditioner = None
        else:
            raise ValueError(
                f"Unknown pre_conditioning: '{pre_conditioning}'. "
                f"Choose from: {list(PRE_CONDITIONER_REGISTRY.keys()) + ['none']}"
            )

    def _init_encoder(self, config, d_task=None):
        if config.num_hidden_layers == 0:
            self.transformer_encoder = MLPConditioner(d_task, config.hidden_size, config.intermediate_size, self.pre_conditioning)
            return

        if self.per_layer_conditioning == "none":
            self.transformer_encoder = NeoBERT(config)
        else:
            self.transformer_encoder = CNeoBERT(
                config,
                d_condition=d_task,
                per_layer_type=self.per_layer_conditioning,
                n_prefix=self.n_prefix,
            )

    def _forward(self, nodes, task):
        x = self.pos_nodes_embed(self.nodes_embed(nodes))

        # Stage 1: pre-encoder conditioning
        if self.pre_conditioner is not None:
            x = self.pre_conditioner(x, task)

        # Stage 2: encoder (with optional per-layer conditioning)
        if isinstance(self.transformer_encoder, CNeoBERT):
            x = self.transformer_encoder(inputs_embeds=x, condition=task)
        elif isinstance(self.transformer_encoder, MLPConditioner):
            x = self.transformer_encoder(inputs_embeds=x, condition=task)
        else:
            x = self.transformer_encoder(inputs_embeds=x)

        return self.fc(x)