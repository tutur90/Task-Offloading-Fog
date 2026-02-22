import torch
import torch.nn as nn
from policies.model.base_model import BaseModel


class TPTOModel(BaseModel):
    """
    Transformer Actor-Critic following the TPTO paper architecture exactly
    (Gholipour et al., arXiv:2312.11739), minus num_actions-related aspects.

    Paper architecture (kept exactly):
      - 3 Transformer encoder layers (BERT-style, no causal mask)
      - 8 attention heads
      - Hidden size / d_model = 512
      - FFN dimension = 512
      - Dropout = 0.4, ReLU activation
      - Post-LN (LayerNorm after attention and after FFN, not before)
      - No explicit positional encoding (implicit from input ordering)
      - Task context injected as a prepended token at position 0 (not added globally)
      - Critic uses the task-token representation (position 0)
      - Actor uses node-token representations (positions 1..n_nodes)

    Adaptation (num_actions-related, kept from our framework):
      - Actor outputs one logit per node → (batch, n_nodes), not 2-way softmax
    """

    def __init__(
        self,
        d_in: int,       # node feature dimension (e.g. 3 for cpu/bw/buffer)
        d_pos: int,      # number of nodes (sequence length)
        d_task: int,     # task feature dimension (4: size, cycles, rate, ddl)
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.4,
        **kwargs,        # absorb unused config keys (mlp_ratio, obs_type, etc.)
    ):
        super().__init__()

        # Input projections
        self.nodes_embed = nn.Linear(d_in, d_model)
        # Task token: projected to d_model, prepended to the node sequence
        self.task_embed = nn.Linear(d_task, d_model, bias=False)

        # No positional encoding — paper relies on implicit ordering (priority rank)

        # Transformer encoder: post-LN (norm_first=False), ReLU FFN
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                norm_first=False,      # post-LN: Add & Norm after attention/FFN
                batch_first=True,
                activation="relu",
            ),
            num_layers=n_layers,
            mask_check=False,
            enable_nested_tensor=False,
        )

        # Actor head: one logit per node token → (batch, n_nodes)
        self.actor_head = nn.Linear(d_model, 1)

        # Critic head: scalar value from task token (position 0)
        self.critic_head = nn.Linear(d_model, 1)

    def forward(self, nodes, task):
        nodes, task = self.normalize(nodes, task)
        return self._forward(nodes, task)

    def _forward(self, nodes, task):
        # Embed nodes: (batch, n_nodes, d_in) → (batch, n_nodes, d_model)
        node_tokens = self.nodes_embed(nodes)

        # Embed task and prepend as position-0 token: (batch, 1, d_model)
        task_token = self.task_embed(task).unsqueeze(1)

        # Full sequence: [task_token | node_tokens] → (batch, 1+n_nodes, d_model)
        x = torch.cat([task_token, node_tokens], dim=1)

        # Transformer (post-LN, no causal mask, no positional encoding)
        x = self.transformer(x, is_causal=False)  # (batch, 1+n_nodes, d_model)

        # Actor: node representations at positions 1..n_nodes
        logits = self.actor_head(x[:, 1:, :]).squeeze(-1)  # (batch, n_nodes)

        # Critic: task-token representation at position 0 (paper §4.3)
        value = self.critic_head(x[:, 0, :])               # (batch, 1)

        return logits, value
