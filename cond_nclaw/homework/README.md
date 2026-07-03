# Conditioning NCLaw — Homework (Stage 1, 3D)

This mirrors the original repo's 3D `InvariantFullMetaElasticity` and its training
loop, extended with a **material latent `z`**. You condition one network on the
*true* material parameters (an oracle) and prove it can span a whole family of
materials — including ones it never trained on. This is the cheapest, most
fundamental test of Hassan's idea: if conditioning fails here, no encoder saves it.

## Layout

```
homework/
  hw/          <- YOU edit these (4 exercises live here)
    materials.py   C1, C2
    pipeline.py    C3, C4
  solutions/   <- complete reference (the grader checks against it; peek when stuck)
  grade.py     <- autograder: PASS / TODO / FAIL per exercise
```

## Workflow

```bash
pip install torch          # CPU is fine; the homework uses torch.linalg.svd
cd homework
python grade.py            # all 4 show TODO at the start
# ... fill in one exercise in hw/ ...
python grade.py            # it flips to PASS (or FAIL with a hint)
```

Everything runs on CPU in float64 in a few seconds — no Warp, no GPU. (The real
repo keeps its Warp SVD; see `PORT_TO_REPO.md` for the exact edits to the actual
`nclaw/material/meta.py` and `experiments/train.py`.)

## The exercises

Ordered from the constitutive law outward to training, matching the repo's
`InvariantFullMetaElasticity`.

| # | Location | What you implement | Why it matters |
|---|---|---|---|
| **C1** | `materials.CondInvariantFullElasticity._frame_stress` | Concatenate `z` onto the rotation invariants, run the MLP, symmetrize. | The conditioning injection itself. The invariants are `[σ−1, FᵀF−I, detF−1]` (width 13, exactly the repo); `z` is appended, and the first-layer width already accounts for it. |
| **C2** | `materials.CondInvariantFullElasticity.forward` | Rest-state subtraction, then `R @ T @ Fᵀ`. | **The subtle one.** Plain concatenation breaks NCLaw's zero-stress-at-rest prior: at `F=I` the invariants vanish but `z` does not, so a no-bias net still emits stress. `NN(inv,z) − NN(0,z)` restores zero-at-rest for *every* `z`. |
| **C3** | `pipeline.build_dataset` | Sample a material continuum, compute ground-truth stress, standardize the oracle `z`, per material. | Builds the family with a held-out split. Generalizing to unseen materials is the one thing conditioning can do that per-material NCLaw cannot. |
| **C4** | `pipeline.train_conditioned` | One conditioned network across all materials (Adam + cosine, normalized-stress loss). | Mirrors `experiments/train.py`'s optimizer/scheduler. The payoff check: it must reach the unseen materials at < 0.15 relative error. |

## What a pass means

When `grade.py` reads 4/4, run `python demo.py` (one level up) for the full
diagnostics. The reference reaches ~3% error on training materials, ~5% on unseen
ones, with wrong-`z` ~10× worse and both physics priors exact. That is Stage 1
succeeding — the conditioning mechanism works and interpolates. **Next:** Stage 2
replaces the oracle `z` with a latent optimized through the sim; Stage 3 trains
the feed-forward encoder (Hassan's contribution).
