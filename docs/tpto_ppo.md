# TPTO — Transformer-PPO Task Offloading

Adaptation of Gholipour et al., *"TPTO: A Transformer-PPO based Task Offloading Solution for Edge Computing Environments"*, arXiv:2312.11739.

**Key difference from the paper**: action = selected node (n-way discrete) instead of binary local/offload.

---

## Files

| File | Role |
|---|---|
| `policies/model/tpto.py` | `TPTOModel` — Transformer Actor-Critic network |
| `policies/ppo/tpto_policy.py` | `TPTOPolicy` — PPO policy (act, update, norm_reward) |
| `utils/ppo.py` | `run_epoch_ppo` — on-policy training/eval loop |
| `configs/Pakistan/Tuple100k/PPO/TPTO.yaml` | Reference config |

---

## Architecture (`TPTOModel`)

```
obs (batch, n_nodes, d_obs)    task (batch, 4)
       │                               │
  nodes_embed (Linear)            task_embed (Linear, no bias)
       │                               │
  LearnedPositionalEncoding           ──► added to every node position
       │
  TransformerEncoder
    └─ n_layers × (pre-LN, batch_first, ReLU FFN)
       │
  ┌────┴────┐
  actor     critic
  Linear    mean-pool → Linear
  (d_model→1) per node   (d_model→1) scalar
  squeeze → (batch, n_nodes)    (batch, 1)
  logits                        value V(s)
```

Normalization (min-max for tasks, max-norm for node obs) is inherited from `BaseModel.register_norm`.

---

## MDP

| Element | Definition |
|---|---|
| **State** | Node resource obs `[cpu, bw, buffer]` per node |
| **Action** | Selected destination node index (0 … n_nodes−1) |
| **Reward** | `−norm_reward([TTR, latency, energy], λ)` — identical to DQN policies |
| **Episode** | Each task is one step; consecutive tasks form a continuing MDP (done=False, γ-bootstrapped) |

---

## PPO Update

Rollout buffer is collected online during `run_epoch_ppo`. When `rollout_size` transitions are ready, `policy.update()` runs:

1. **Bootstrap** — compute V(s') for all next states in one batched forward pass.
2. **GAE** — δ_t = r_t + γ V(s_{t+1})(1−done) − V(s_t), accumulated backward.
3. **Normalize** advantages: (A − mean) / std.
4. **PPO epochs** — `ppo_epochs` passes over shuffled mini-batches:
   - ratio r = exp(log π_new − log π_old)
   - L_clip = −min(r·A, clip(r, 1−ε, 1+ε)·A)
   - L_vf = MSE(V_new, returns)
   - L_entropy = −entropy
   - loss = L_clip + c1·L_vf + c2·L_entropy
5. Gradient clip → optimizer step.

---

## Config keys (PPO-specific)

```yaml
training:
  gae_lambda: 0.95       # GAE trace-decay λ
  clip_eps: 0.2          # PPO clip ratio ε
  entropy_coef: 0.01     # c2 — encourages exploration
  value_loss_coef: 0.5   # c1
  clip_grad_norm: 0.5
  ppo_epochs: 4          # gradient epochs per rollout
  rollout_size: 512      # transitions collected before each update
  mini_batch_size: 64
```

All other keys (`lr`, `gamma`, `reward`, `lambda`, `model`, etc.) are shared with DQL policies.

---

## Usage

```bash
# Train
python main.py configs/Pakistan/Tuple100k/PPO/TPTO.yaml

# Hyperparameter search
python main.py configs/Pakistan/Tuple100k/PPO/TPTO.yaml \
  --search "model.d_model=64,128,256" "training.lr=3e-4,1e-4" \
  --sampler random --n_samples 20
```
