import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from policies.model.base_model import BaseModel
from policies.model.modules.transformer import LearnedPositionalEncoding
from policies.model.modules.noebert import NeoBERT, NeoBERTConfig, CNeoBERT, SwiGLU


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
        rank = rank / max(N - 1, 1)  # avoid division by zero when N=1

        dist_to_max = nodes.max(dim=1, keepdim=True).values - nodes
        dist_to_min = nodes - nodes.min(dim=1, keepdim=True).values

        x = torch.cat([nodes, diff, rank, dist_to_max, dist_to_min], dim=-1)
        return self.proj(x)


# --- Conditioners ---

class FiLMConditioner(nn.Module):
    """Task modulates node representations via learned scale + shift.
    Initialized to identity: gamma=1, beta=0."""
    def __init__(self, d_task, d_model):
        super().__init__()
        self.gamma = nn.Linear(d_task, d_model)
        self.beta = nn.Linear(d_task, d_model)
        # Identity init so conditioning is a no-op at start of training
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, node_embeds, task_features):
        gamma = self.gamma(task_features).unsqueeze(1)  # (B, 1, d_model)
        beta = self.beta(task_features).unsqueeze(1)
        return gamma * node_embeds + beta


class AdditiveConditioner(nn.Module):
    """Task modulates node representations via learned additive embedding."""
    def __init__(self, d_task, d_model):
        super().__init__()
        self.task_embed = nn.Linear(d_task, d_model, bias=False)
        nn.init.zeros_(self.task_embed.weight)  # no-op at init

    def forward(self, node_embeds, task_features):
        task_emb = self.task_embed(task_features).unsqueeze(1)  # (B, 1, d_model)
        return node_embeds + task_emb


class MLPConditioner(nn.Module):
    """Lightweight conditioner wrapper for the n_layers=0 fallback."""
    def __init__(self, d_task, d_model, d_hidden=None, conditioning="film"):
        super().__init__()
        conditioner_cls = FiLMConditioner if conditioning == "film" else AdditiveConditioner
        self.conditioner = conditioner_cls(d_task, d_model)

    def forward(self, inputs_embeds, condition):
        return self.conditioner(inputs_embeds, condition)


# --- Compute LLaMA-style intermediate size once ---

def compute_intermediate_size(d_model, mlp_ratio, multiple_of=8):
    """LLaMA-style SwiGLU sizing: 2/3 expansion, rounded to multiple_of."""
    raw = int(2 * d_model * mlp_ratio / 3)
    return multiple_of * ((raw + multiple_of - 1) // multiple_of)


# --- NATE ---

class NATE(BaseModel):
    def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None,
                 n_heads=4, n_layers=3, dropout=0.1, qk_norm=True,
                 learnable_qk_norm=True, embed="regular", **kwargs):
        super().__init__()

        self.nodes_embed = self._build_embed(embed, d_in, d_model, mlp_ratio)
        self.pos_nodes_embed = LearnedPositionalEncoding(max_seq_len=d_pos, d_model=d_model)
        self.fc = nn.Linear(d_model, 1)

        self._init_encoder(d_model, mlp_ratio, d_ff, n_heads, n_layers, dropout, qk_norm, d_task=d_task)
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
        """Init weights for modules outside the transformer encoder."""
        init_std = 0.02
        for module in [self.nodes_embed, self.pos_nodes_embed, self.fc]:
            for p in module.parameters():
                if p.dim() >= 2:
                    nn.init.normal_(p, mean=0.0, std=init_std)
                elif p.dim() == 1:
                    nn.init.zeros_(p)

    def _init_encoder(self, d_model, mlp_ratio, d_ff, n_heads, n_layers, dropout, qk_norm, d_task=None):
        if d_ff is None:
            d_ff = compute_intermediate_size(d_model, mlp_ratio)

        if n_layers == 0:
            self.transformer_encoder = nn.Sequential(
                nn.RMSNorm(d_model, eps=1e-6),
                SwiGLU(d_model, d_ff, d_model),
                nn.RMSNorm(d_model, eps=1e-6),
            )
            return

        encoder_config = NeoBERTConfig(
            hidden_size=d_model,
            intermediate_size=d_ff,
            num_attention_heads=n_heads,
            num_hidden_layers=n_layers,
            dropout=dropout,
            qk_norm=qk_norm,
        )
        self.transformer_encoder = NeoBERT(encoder_config)

    def _forward(self, nodes, task=None):
        x = self.pos_nodes_embed(self.nodes_embed(nodes))
        x = self.transformer_encoder(inputs_embeds=x)
        x = self.fc(x)
        return x


# --- TNATE ---

class TNATE(NATE):
    def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None,
                 n_heads=4, n_layers=3, dropout=0.1, conditioning="film",
                 condition_each_layer=True, **kwargs):
        # Store before super().__init__ because _init_encoder needs them
        self.conditioning = conditioning
        self.condition_each_layer = condition_each_layer
        super().__init__(
            d_in=d_in, d_pos=d_pos, d_task=d_task, d_model=d_model,
            mlp_ratio=mlp_ratio, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers,
            dropout=dropout, **kwargs,
        )

    def _init_encoder(self, d_model, mlp_ratio, d_ff, n_heads, n_layers, dropout, qk_norm, d_task=None):
        if d_ff is None:
            d_ff = compute_intermediate_size(d_model, mlp_ratio)

        if n_layers == 0:
            self.transformer_encoder = MLPConditioner(d_task, d_model, d_ff, self.conditioning)
            return

        encoder_config = NeoBERTConfig(
            hidden_size=d_model,
            intermediate_size=d_ff,
            num_attention_heads=n_heads,
            num_hidden_layers=n_layers,
            dropout=dropout,
            qk_norm=qk_norm,
            condition_each_layer=self.condition_each_layer,
        )

        conditioner_cls = FiLMConditioner if self.conditioning == "film" else AdditiveConditioner
        self.transformer_encoder = CNeoBERT(encoder_config, conditioner_cls, d_task)

    def _forward(self, nodes, task):
        x = self.pos_nodes_embed(self.nodes_embed(nodes))
        x = self.transformer_encoder(inputs_embeds=x, condition=task)
        x = self.fc(x)
        return x