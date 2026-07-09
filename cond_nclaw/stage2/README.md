# Conditioning NCLaw — Homework (Stage 2: autodecoder, two networks)

## Purpose — where this sits

Stage 1 conditioned the network on the material's *true* parameters (an oracle).
**Stage 2 throws the true parameters away.** Each material now gets a *free latent
vector the model must discover on its own*, learned jointly with the network by
optimizing the loss — the UniPhy **autodecoder**. A *new* material's latent is then
recovered by **test-time inversion**: freeze the trained network and optimize a fresh
latent to fit observations, with no retraining.

This is the last rung before Stage 3 (the feed-forward encoder). Its jobs are to
(1) prove a latent *can* carry material identity when it isn't handed the parameters,
(2) reproduce UniPhy's inference procedure, and (3) give you a concrete baseline —
especially a **latent-structure** number — that the Stage-3 encoder has to beat.

Two things extend the earlier Stage-2 sketch, both by request:

- **Two conditioned networks**, elastic *and* plastic, exactly like NCLaw's pair,
  sharing **one** latent per material (material identity is a single code that
  conditions both constitutive laws).
- **Equilibrium checked at z=0** — see implementation notes below.

The homework itself is the *labeled* version (supervise on stress + plastic
projection) so it runs on CPU in seconds and isolates the latent question. The
`warp/` folder holds the faithful **through-the-simulator** version for your Warp
machine (trained on motion only); it is given, not graded.

## What to expect

- **Reconstruction and inversion are reliable.** One shared network + learned latents
  reproduce every training material (~1–2% elastic, a few % plastic-correction), and
  inversion fits an unseen material to a few %. These pass every run.
- **Global latent structure is emergent, not guaranteed.** The `spearman(latent
  distances, (ν, τY) distances)` readout wanders (~0.4–0.85) across seeds. The
  autodecoder loss only asks each material's latent to *decode correctly*; it never
  asks the latents to be *arranged* by similarity, so any structure is a byproduct of
  the network's smoothness, not the objective. That gap is exactly what Stage 3's
  contrastive term is for — beat this number there.
- **Through the sim is messier than labeled.** When you run `warp/`, expect slower,
  noisier convergence (long-horizon BPTT + SVD-adjoint-at-rest), no need for stress
  labels, and typically *weaker* latent structure than the labeled upper bound here.

## Implementation details

- **Invariants (given).** Both nets consume the repo's rotation invariants
  `[σ−1, FᵀF−I, detF−1]` (width 13); the latent `z` is concatenated on (width 13+z_dim).
  The elastic head returns `R·T·Fᵀ`; the plastic head returns `F + α·R·T` (a small
  return-map nudge). `T` is the symmetrized network output in the invariant frame.
- **The equilibrium fork (this is the "z=0" point).** With **plain concatenation**
  (`rest_correct=False`, the default and the UniPhy convention), zero-stress-at-rest
  holds **only at z=0**: at F=I the invariants vanish, so only a *zero* latent leaves
  the no-bias net with all-zero input. A nonzero latent re-introduces rest stress —
  so you check equilibrium *at z=0*. Setting `rest_correct=True` subtracts the net's
  output at the rest input, restoring equilibrium for **every** z (the Stage-1 trick).
  Both are implemented; T1 wires the switch.
- **The autodecoder recipe.** One optimizer with two param groups — the networks at
  `lr`, the latents at `10·lr` (latents must move faster to specialize) — plus a
  std-1.0 latent init and a cosine schedule. Stress targets are normalized by a global
  scale; plastic targets (F′) are O(1) and left unscaled.
- **Material family.** A `(ν, τY)` grid at fixed E: Poisson ratio ν drives the elastic
  *shape* (so the latent must carry more than scale), and von Mises yield τY gives the
  plastic net something to condition on. A held-out set tests inversion.

## Layout & workflow

```
homework/
  hw/            YOU edit -- T1, T2 in materials.py; T3, T4, T5 in autodecoder.py
  solutions/     reference
  grade.py       PASS / TODO / FAIL
```
```bash
pip install torch && cd homework && python grade.py
```

## The exercises

| # | Location | What you implement | Why it matters |
|---|---|---|---|
| **T1** | `materials._CondInvariantFull._frame` | Concatenate `z`, run the MLP, symmetrize; optional rest-state subtraction. | The conditioning injection shared by both heads, and the equilibrium switch. |
| **T2** | `CondElasticity.forward`, `CondPlasticity.forward` | The two heads: `R·T·Fᵀ` and `F+α·R·T`. Graded together with **equilibrium at z=0**. | Conditioning NCLaw's elastic/plastic pair on one latent. |
| **T3** | `autodecoder.LatentBook` | A learnable table of per-material latents (std=1.0), one shared code feeding both nets. | The autodecoder store; std-1.0 init lets latents separate. |
| **T4** | `autodecoder.train_autodecoder` | Jointly optimize both nets + latents: dual-LR Adam, cosine, elastic + plastic loss. | The autodecoder recipe over two networks at once. |
| **T5** | `autodecoder.invert_latent` | Freeze both nets; optimize a fresh latent to fit an unseen material. | Test-time adaptation with no retraining — UniPhy's inference, and the thing Stage 3 amortizes. |

Reference behaviour: elastic ~1.7%, plastic-correction ~2.9%, equilibrium exactly 0
at z=0 for both nets, inversion ~3–6%, structure ~0.5–0.6 (emergent).

## Next

Stage 3 replaces the per-material optimization of T4/T5 with a feed-forward encoder
(trajectory → latent in one pass) and adds a contrastive objective to *enforce* the
latent geometry this stage only stumbles into. `warp/README_warp.md` covers running
the through-sim version on your machine.
