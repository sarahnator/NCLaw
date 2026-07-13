# Stage 3 through the Warp sim (your machine)

An encoder predicts each material's latent from its **motion trajectory** in one pass;
encoder + both conditioned nets are trained by BPTT through the differentiable MPM on
motion only, plus the contrastive term. Faithful to `experiments/train.py`; unverified
here (no Warp) — check the first epoch's loss is finite before a long run.

## Files
- `conditioned_materials.py` — the two conditioned nets (Warp SVD, `self.z`), same as Stage 2.
- `traj_encoder.py` — `TrajEncoderSim` (DeepSets over per-particle motion) + `contrastive_loss`.
  **Hassan's GNS goes here**: replace the `phi` MLP with a message-passing block over a
  radius graph on the particles; everything downstream (pool → `rho` → z) is unchanged.
- `train_encoder_sim.py` — the through-sim joint training loop.

## Setup & run
1. Install the materials + encoder into `nclaw/material/` (expose the class names).
2. Generate the `(ν, τY)` family datasets (see Stage-2 `README_warp.md`) and `family.json`.
3. Run:
```bash
uv run python experiments/train_encoder_sim.py \
    env=jelly sim=low \
    env/blob/material/elasticity=cond_invariant_full \
    env/blob/material/plasticity=cond_invariant_full \
    +z_dim=4 +family_spec=$PWD/family.json +lam=0.3 \
    name=cond/stage3_sim
```

## What to expect vs the labeled homework
- Harder and slower: you're differentiating the encoder AND the two nets through a
  1000-step rollout. Watch `|grad|`; `error_if_nonfinite=True` will hard-stop on a NaN.
- The encoder sees only motion (no stress labels) — the whole NCLaw premise.
- The amortization gap on unseen materials is typically larger than in the labeled case;
  the GNS encoder is the main lever for closing it.
- Inference is one forward pass: `encoder(trajectory)` gives z for a brand-new material,
  no optimization — the payoff over Stage 2's inversion loop.

Two simplifications to fix for a real run: this loop uses a single encoder view (the
labeled homework uses two disjoint views for the contrastive term — add view sampling
over particle subsets or trajectory windows), and it re-simulates every material every
epoch (sample a subset per epoch if that's too slow).
