# TPTO — Transformer-PPO Task Offloading

Adaptation of Gholipour et al., *"TPTO: A Transformer-PPO based Task Offloading Solution for Edge Computing Environments"*, arXiv:2312.11739.

**Adaptation**: action = selected node (n-way discrete) instead of binary local/offload.
Everything else follows the paper exactly.

---

## Architecture (`TPTOModel`) — paper §4

```
obs (batch, n_nodes, d_obs)       task (batch, 4)
        │                                │
  nodes_embed (Linear)           task_embed (Linear, no bias)
        │                                │
        │                         unsqueeze(1)
        │                         → (batch, 1, d_model)   ← task token (position 0)
        │
  [task_token | node_tokens]  →  (batch, 1+n_nodes, d_model)
        │
  TransformerEncoder  ×3 layers
    └─ MultiHeadAttention (8 heads)
    └─ Add & LayerNorm             ← post-LN (after attention)
    └─ FFN (d_ff=512, ReLU)
    └─ Add & LayerNorm             ← post-LN (after FFN)
        │
  ┌─────┴──────┐
  position 0   positions 1..n_nodes
  (task token) (node tokens)
       │              │
  critic_head    actor_head
  Linear→1       Linear→1 per node → squeeze
  V(s)           → (batch, n_nodes) logits
```

**No positional encoding** — paper relies on implicit ordering (priority rank).
Normalization inherited from `BaseModel.register_norm` (max-norm for nodes, min-max for task).

### Paper hyperparameters (§4.1)

| Parameter | Value |
|---|---|
| Transformer layers | 3 |
| Attention heads | 8 |
| Hidden size (d_model) | 512 |
| FFN dimension (d_ff) | 512 |
| Dropout | 0.4 |
| Activation | ReLU |
| LayerNorm | Post-attention, post-FFN |

---

## MDP

| Element | Definition |
|---|---|
| **State** | Node resource obs `[cpu, bw, buffer]` per node |
| **Action** | Selected destination node index (0 … n_nodes−1) |
| **Reward** | `−norm_reward([TTR, latency, energy], λ)` |
| **Episode** | Each task is one step; consecutive tasks form a continuing MDP (done=False, γ-bootstrapped) |

---

## PPO Update — paper §5

Rollout buffer collects `rollout_size` transitions, then `policy.update()` runs:

1. **Bootstrap** — compute V(s') for all next states with one batched forward pass.
2. **GAE** — δ_t = r_t + γ V(s_{t+1})(1−done) − V(s_t), accumulated backward with λ=0.95.
3. **Normalize** advantages: (A − mean) / std.
4. **PPO epochs** over mini-batches:
   - ratio r = exp(log π_new − log π_old)
   - L_clip = −min(r·A, clip(r, 1−ε, 1+ε)·A)  with ε=0.2
   - L_vf = MSE(V_new, returns)
   - L_entropy = −entropy   (coefficient c2=0.5 from paper)
   - loss = L_clip + c1·L_vf − c2·L_entropy
5. Gradient clip → **Adagrad** optimizer step (lr=0.1, from paper).

### Paper PPO hyperparameters (§5.3)

| Parameter | Paper value | Config key |
|---|---|---|
| Policy LR | 0.1 | `lr` |
| Optimizer | Adagrad | `optimizer.type` |
| Batch size | 100 | `rollout_size` |
| Clip ratio ε | 0.2 | `clip_eps` |
| Discount γ | 0.99 | `gamma` |
| Entropy coef c2 | 0.5 | `entropy_coef` |

---

## Differences from current NOTE/DQN policies

| Aspect | DQL policies (NOTE, MLP…) | TPTO |
|---|---|---|
| Algorithm | Off-policy DQN, replay buffer | On-policy PPO, rollout buffer |
| Network output | Q-values `(batch, n_nodes)` | Logits + value `(batch, n_nodes), (batch,1)` |
| Task context | Added globally to all node positions | Prepended as dedicated token at position 0 |
| Positional encoding | Learned | None |
| LayerNorm | Pre-LN | Post-LN |
| Critic | Mean-pool → FC | Task-token → FC |
| Optimizer | AdamW | Adagrad |

---

## Usage

```bash
# Train
python main.py configs/Pakistan/Tuple100k/PPO/TPTO.yaml

# Hyperparameter search
python main.py configs/Pakistan/Tuple100k/PPO/TPTO.yaml \
  --search "model.d_model=256,512" "training.lr=0.1,0.01" \
  --sampler grid
```
