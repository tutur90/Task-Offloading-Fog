import math
import torch
import torch.nn as nn
from policies.model.base_model import BaseModel
from policies.model.NOTE import LearnedPositionalEncoding


class TPTOModel(BaseModel):
    """
    Transformer Actor-Critic model for PPO-based node selection.

    Adapted from the TPTO paper (Gholipour et al., arXiv:2312.11739).
    Key difference: action = selected node (n-way discrete) instead of binary local/offload.

    Architecture:
        Input: node resource obs (batch, n_nodes, d_obs) + task features (batch, d_task)
        ↓  nodes_embed  (Linear d_obs → d_model)  + LearnedPositionalEncoding
        ↓  task_embed   (Linear d_task → d_model)  added to every node position
        ↓  TransformerEncoder  (n_layers × BERT-style encoder layer)
        ↓  actor_head:  Linear(d_model, 1) → squeeze → (batch, n_nodes) logits
        ↓  critic_head: mean-pool over nodes → Linear(d_model, 1) → (batch, 1) value
    """

    def __init__(
        self,
        d_in: int,
        d_pos: int,
        d_task: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        mlp_ratio: int = 4,
        d_ff: int = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.nodes_embed = nn.Linear(d_in, d_model)
        self.task_embed = nn.Linear(d_task, d_model, bias=False)
        self.pos_embed = LearnedPositionalEncoding(max_seq_len=d_pos, d_model=d_model)

        dim_feedforward = d_ff if d_ff is not None else d_model * mlp_ratio

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,   # pre-LN (more stable)
                batch_first=True,
                activation="relu",
            ),
            num_layers=n_layers,
            mask_check=False,
            enable_nested_tensor=False,
        )

        # Actor: per-node scalar logit → (batch, n_nodes)
        self.actor_head = nn.Linear(d_model, 1)

        # Critic: global value estimate → (batch, 1)
        self.critic_head = nn.Linear(d_model, 1)

    # Override BaseModel.forward to return (logits, value) tuple.
    def forward(self, nodes, task):
        nodes, task = self.normalize(nodes, task)
        return self._forward(nodes, task)

    def _forward(self, nodes, task):
        x = self.nodes_embed(nodes)          # (batch, n_nodes, d_model)
        x = self.pos_embed(x)

        task_emb = self.task_embed(task)     # (batch, d_model)
        x = x + task_emb.unsqueeze(1)       # broadcast over n_nodes

        x = self.transformer(x, is_causal=False)  # (batch, n_nodes, d_model)

        logits = self.actor_head(x).squeeze(-1)    # (batch, n_nodes)
        value = self.critic_head(x.mean(dim=1))    # (batch, 1)

        return logits, value
