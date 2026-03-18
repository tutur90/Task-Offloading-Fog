import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from policies.model.base_model import BaseModel
from policies.model.modules.transformer import LearnedPositionalEncoding
from policies.model.modules.transformer_encoder import TransformerEncoder
from policies.model.modules.conditional_transformer import TransformerEncoder as ConditionalTransformerEncoder


torch.backends.cuda.matmul.allow_tf32 = True


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
# NATE (unconditioned)
# =============================================================================

class NATE(BaseModel):
    def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None,
                 n_heads=4, n_layers=3, dropout=0.1, qk_norm=True,
                 learnable_qk_norm=True, embed="regular", d_heads=None, softplus_attn=None,
                 use_attention=True, residual_type="gru", output_size=None, obs_type=None):
        super().__init__()
        if d_heads is not None:
            n_heads = d_model // d_heads

        # Store encoder hyperparams (read by _init_encoder)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff if d_ff is not None else d_model * mlp_ratio
        self.dropout = dropout
        self.qk_norm = qk_norm
        self.learnable_qk_norm = learnable_qk_norm
        self.use_attention = use_attention
        self.residual_type = residual_type
        self.softplus_attn = softplus_attn

        self.nodes_embed = self._build_embed(embed, d_in, d_model, mlp_ratio)
        self.pos_nodes_embed = LearnedPositionalEncoding(max_seq_len=d_pos, d_model=d_model)
        self.fc = nn.Linear(d_model, 1)

        self._init_encoder()
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
        elif embed == "no_bias":
            return nn.Linear(d_in, d_model, bias=False)
        else:
            raise ValueError(f"Unknown embed type: {embed}")

    def _init_non_encoder_weights(self):
        """Init all modules NOT inside the transformer encoder.
        Covers nodes_embed, pos_nodes_embed, fc, and n_layers=0 fallback.
        Modules that self-init (conditioners, GRU gates) are excluded
        to preserve their zero/identity init."""
        init_std = 0.02
        encoder_params = set(self.transformer_encoder.parameters())
        for p in self.parameters():
            if p in encoder_params:
                continue
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=init_std)
            elif p.dim() == 1:
                nn.init.zeros_(p)

    def _init_encoder(self):
        self.transformer_encoder = TransformerEncoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            n_layers=self.n_layers,
            dropout=self.dropout,
            qk_norm=self.qk_norm,
            learnable_qk_norm=self.learnable_qk_norm,
            use_attention=self.use_attention,
            residual_type=self.residual_type,
            softplus_attn=self.softplus_attn,
        )

    def _forward(self, nodes, task=None):
        x = self.pos_nodes_embed(self.nodes_embed(nodes))
        x = self.transformer_encoder(inputs_embeds=x)
        x = self.fc(x)
        return x


# =============================================================================
# T-NATE (task-conditioned with prefix tokens + pre-encoder conditioning)
# =============================================================================

class TNATE(NATE):
    def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None,
                 n_heads=4, n_layers=3, dropout=0.1,
                 pre_conditioning="add", qk_norm=True, learnable_qk_norm=True,
                 n_prefix=4, d_heads=None, embed="regular", softplus_attn=None,
                 use_attention=True, residual_type="gru", output_size=None, obs_type=None):
        # Store before super().__init__ because _init_encoder reads them
        self.pre_conditioning = pre_conditioning
        self.n_prefix = n_prefix
        self._d_task = d_task

        super().__init__(
            d_in=d_in, d_pos=d_pos, d_task=d_task, d_model=d_model,
            embed=embed,
            mlp_ratio=mlp_ratio, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers,
            dropout=dropout, qk_norm=qk_norm, learnable_qk_norm=learnable_qk_norm,
            d_heads=d_heads, softplus_attn=softplus_attn,
            use_attention=use_attention, residual_type=residual_type, output_size=output_size,
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

    def _init_encoder(self):
        """Called from NATE.__init__; also builds prefix embedding if needed."""
        super()._init_encoder()

        if self.n_prefix > 0:
            # Project task features -> n_prefix tokens of size d_model
            self.prefix_embed = nn.Linear(self._d_task, self.d_model * self.n_prefix)
        else:
            self.prefix_embed = None

    def _forward(self, nodes, task):
        x = self.pos_nodes_embed(self.nodes_embed(nodes))

        # Stage 1: pre-encoder conditioning (FiLM / additive / none)
        if self.pre_conditioner is not None:
            x = self.pre_conditioner(x, task)

        # Stage 2: prepend learned prefix tokens derived from task features
        if self.prefix_embed is not None:
            B = x.size(0)
            prefix = self.prefix_embed(task).reshape(B, self.n_prefix, self.d_model)
            x = torch.cat([prefix, x], dim=1)  # (B, n_prefix + S, d_model)

        # Stage 3: transformer encoder
        x = self.transformer_encoder(inputs_embeds=x)

        # Strip prefix tokens, keep only node positions
        if self.prefix_embed is not None:
            x = x[:, self.n_prefix:, :]

        return self.fc(x)


# =============================================================================
# CT-NATE (condition injected via AdaRMSNorm at every layer norm)
# =============================================================================

class CTNATE(NATE):
    """CT-NATE: task condition injected into every norm via AdaRMSNorm.
    No prefix tokens; conditioning is purely through the norm layers."""
    def __init__(self, d_in, d_pos, d_task, d_model=64, mlp_ratio=4, d_ff=None,
                 n_heads=4, n_layers=3, dropout=0.1, qk_norm=True,
                 learnable_qk_norm=True, embed="regular", d_heads=None, softplus_attn=None,
                 use_attention=True, residual_type="gru", modulation=True,
                 output_size=None, obs_type=None):
        self._d_task = d_task
        self._modulation = modulation
        super().__init__(
            d_in=d_in, d_pos=d_pos, d_task=d_task, d_model=d_model,
            mlp_ratio=mlp_ratio, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers,
            dropout=dropout, qk_norm=qk_norm, learnable_qk_norm=learnable_qk_norm,
            embed=embed, d_heads=d_heads, softplus_attn=softplus_attn,
            use_attention=use_attention, residual_type=residual_type, output_size=output_size,
        )

    def _init_encoder(self):
        self.transformer_encoder = ConditionalTransformerEncoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            n_layers=self.n_layers,
            dropout=self.dropout,
            qk_norm=self.qk_norm,
            learnable_qk_norm=self.learnable_qk_norm,
            use_attention=self.use_attention,
            residual_type=self.residual_type,
            softplus_attn=self.softplus_attn,
            d_cond=self._d_task,
            modulation=self._modulation,
        )

    def _forward(self, nodes, task):
        x = self.pos_nodes_embed(self.nodes_embed(nodes))
        x = self.transformer_encoder(inputs_embeds=x, condition=task)
        return self.fc(x)