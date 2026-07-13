# Conditioning NCLaw — Homework (Stage 3: the feed-forward encoder)

## Purpose — the actual contribution

Stage 2 could adapt to a new material, but only by **optimizing** a fresh latent
through inversion — a per-material gradient-descent loop at test time. **Stage 3
amortizes that away.** A feed-forward encoder maps a material's observations straight
to its latent `z` in a **single forward pass**: no inner optimization, no retraining.
This is Hassan's contribution and the novel niche relative to UniPhy (which infers the
latent by test-time optimization) and PC-NCLaws (which needs the true parameters).

It also fixes Stage 2's other weakness. The autodecoder's latent geometry was
*emergent* (structure score ~0.6, seed-dependent) because the loss never asked for it.
Stage 3 adds a **contrastive term** that shapes the geometry *on purpose*, using the
true material parameters as a graded similarity signal — answering the "how do we gauge
similarity?" question directly. In this reference that lifts the structure score to
~0.94.

## What to expect

- **Amortized inference works, with a gap.** One forward pass on a material's
  observations yields a `z` that reconstructs it. On *training* materials this is tight
  (~5% elastic, ~8% plastic-correction). On **unseen** materials — the real test — the
  one-pass `z` reconstructs to ~20%: clearly working, but looser than training. That
  gap is the honest amortization cost, and it shrinks with more training materials and a
  stronger encoder (see the GNS note below). Compare it to Stage 2, which reached a few
  % on unseen materials but *only after* an optimization loop; Stage 3 trades some
  accuracy for a single forward pass.
- **Structure is now enforced, not hoped for.** The contrastive term drives
  `spearman(latent, params)` to ~0.9+, versus the autodecoder's ~0.6. This is the
  quantity you set out to beat.
- **Through the sim is harder still.** The `warp/` version encodes *motion* trajectories
  and trains through the differentiable MPM — messier and slower, and the encoder there
  is the natural place for Hassan's GNS.

## Implementation details

- **Encoder (E1).** A DeepSets set-encoder over a material's `(F, stress)` samples:
  a per-sample MLP `φ`, a permutation-invariant **mean pool** over samples, then an MLP
  `ρ` to `z`. Permutation-invariance is essential — the material's identity can't depend
  on the order you list its observations.
- **Contrastive loss (E2).** CLIP-style with a **soft** target: two views (disjoint
  observation subsets) of each material in the batch, and a target affinity matrix
  `softmax(−‖Δparams‖² / 2σ²)` — diagonal (same material) highest, off-diagonal graded by
  true-parameter distance. Symmetric soft cross-entropy. This is what makes the latent
  geometry smooth and ordered rather than merely separated.
- **Joint training (E3).** Encoder + both conditioned nets, optimized together on
  `reconstruction + λ·contrastive`. Reconstruction ties `z` to the decoder (the two
  nets must reproduce the material from the encoded `z`); the contrastive term shapes the
  geometry. Two views per material per step.
- **Amortized inference (E4).** Literally `encoder(F, tau)` under `no_grad` — the whole
  point is that it's *just a forward pass*, in deliberate contrast to Stage 2's inversion
  loop. Frame it that way when you read the code.
- **Family & the two nets.** Same `(ν, τY)` grid and the same elastic+plastic pair from
  Stage 2. Equilibrium at z=0 is inherited from the conditioned material.

## Layout & workflow

```
homework/
  hw/          YOU edit -- E1-E4 in encoder.py (materials.py is given)
  solutions/   reference
  grade.py     PASS / TODO / FAIL   (E3 trains a small model, ~30s)
```
```bash
pip install torch && cd homework && python grade.py
```

## The exercises

| # | Location | What you implement | Why it matters |
|---|---|---|---|
| **E1** | `encoder.TrajEncoder.forward` | DeepSets: per-sample `φ`, mean-pool, `ρ` → z. | The amortized encoder — permutation-invariant material fingerprint. |
| **E2** | `encoder.contrastive_loss` | CLIP-style soft contrastive with param-distance targets. | Shapes the latent geometry Stage 2 only stumbled into. |
| **E3** | `encoder.train_encoder` | Joint training: two views, reconstruction + λ·contrastive. | Ties `z` to the decoder AND orders the space. |
| **E4** | `encoder.amortized_infer` | One forward pass, no grad. | The contribution: inference without an optimization loop. |

Reference: train recon ~5%/8%, unseen one-pass ~20%/17%, permutation-invariant, and
structure ~0.94 (vs Stage-2 ~0.6).

## Where this leaves the project

You now have the full three-stage arc: oracle conditioning (Stage 1) → discovered latent
by inversion (Stage 2) → amortized, structured latent by a feed-forward encoder
(Stage 3). The open threads are the ones the smell test flagged: the unseen-material
amortization gap (more data, a GNS encoder), and the move from a within-form continuum to
genuinely heterogeneous materials, where a latent finally beats explicit parameters.
`warp/README_warp.md` covers running the through-sim encoder on your machine.
