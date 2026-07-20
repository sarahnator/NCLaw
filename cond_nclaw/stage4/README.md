# Conditioning NCLaw — Homework (Stage 4: heterogeneous materials)

## Purpose — the claim the whole project was building toward

Stages 1-3 validated the conditioning machinery on a **within-form continuum**
(materials of one constitutive form, varying only in `(ν, τY)`). But on a continuum
you could have conditioned on the explicit parameters and skipped the latent entirely —
so the latent never had to prove its worth. **Stage 4 crosses into heterogeneous
materials**: distinct constitutive *forms* — elastic, von Mises plastic, and fluid —
that have **no shared parameter vector**. A fluid's bulk modulus, a plastic's yield
stress, and an elastic's Poisson ratio don't live on a common axis; you cannot hand the
network one parameter vector that means the same thing across them. This is the first
setting where a **learned latent genuinely beats explicit parameters**, and it is the
untested core claim of the project.

Two design choices change because of this, and they are the exercises:
- **No graded similarity signal exists**, so the contrastive loss switches from
  parameter-distance (Stage 3) to a **type-based supervised** loss: same-form positives,
  different-form negatives.
- **Structure is measured by clustering**, not correlation: can you read a material's
  *form* off its latent?

Guardrail (from the representational-ceiling discussion): all three forms are **inside
NCLaw's representable class**. Something like snow (hardening → history-dependent) is out
of class, so a failure there wouldn't tell you if the encoder or the physics broke —
good as a deliberate negative control later, not part of this first test.

## What to expect

- **One conditioned pair spans all three forms.** The single elastic/plastic network
  pair reproduces jelly, plasticine, *and* water from their latents (~6% elastic/plastic,
  ~12% fluid). This is the real test of NCLaw's "one architecture, many materials" claim
  under conditioning — and it holds.
- **The latent clusters by form.** With the type-contrastive term, training materials
  separate *perfectly* by type (nearest-centroid accuracy 1.0).
- **Generalization is the frontier.** Amortized one-pass reconstruction of *unseen*
  materials works (~0.25-0.30 per form), but unseen **type placement** only partially
  generalizes (~0.6, versus chance 0.33): the encoder learns to separate the *specific*
  training materials more than a transferable form boundary. That gap is the honest
  Stage-4 result and the thing a stronger encoder (a GNS) and more data would close.

## Implementation details

- **The fluid form (H1).** Weakly-compressible fluid = corotated elasticity with shear
  modulus `μ = 0` (no resistance to shape change, only to volume change), and a
  volume-preserving return map `Fp = J^(1/3) I`. Reuses the existing corotated code.
- **Heterogeneous dispatch (H2).** Each material carries a *type id*; the dataset builder
  routes it to the right constitutive law. There is deliberately no shared parameter
  tensor — the material's identity is its form plus per-form parameters.
- **Type-based contrastive (H3).** Supervised contrastive with same-*type* positives.
  The target affinity is uniform over materials of the same form (there's no smooth
  parameter distance to grade by), replacing Stage 3's parameter-graded target.
- **Clustering metric (H4).** Nearest-type-centroid accuracy: build a centroid per form
  from training latents, classify held-out materials by nearest centroid. This is the
  heterogeneous analogue of Stage 3's structure score.
- **Given from earlier stages:** the two conditioned nets, the DeepSets encoder, and the
  joint training loop (now using the type-contrastive term).

## Layout & workflow

```
homework/
  hw/          YOU edit -- H1-H4 in hetero.py (materials.py is given)
  solutions/   reference
  grade.py     PASS / TODO / FAIL, plus a short integration train (~30s)
```
```bash
pip install torch && cd homework && python grade.py
```

## The exercises

| # | Location | What you implement | Why it matters |
|---|---|---|---|
| **H1** | `hetero.fluid_cauchy`, `fluid_return` | The fluid constitutive form (μ=0 stress, `J^(1/3) I` return). | Adds a genuinely different form, making the family heterogeneous. |
| **H2** | `hetero.build_grouped` | Dispatch each material to its form to build `(F, τ, Fp)`. | Encodes that there is no shared parameter vector — only forms + per-form params. |
| **H3** | `hetero.type_contrastive` | Same-type-positives supervised contrastive. | The similarity signal when parameters can't be compared across forms. |
| **H4** | `hetero.cluster_accuracy` | Nearest-centroid type classification of latents. | Measures whether the latent organizes by material form. |

Reference: train recon ~6%/6%/12% across the three forms, train type-accuracy 1.0,
unseen recon ~0.25-0.30, unseen type-accuracy ~0.6 (the frontier).

## Where this leaves you

You've now tested the project's central claim — a single learned latent space that
unifies materials with no common parameterization — and found the frontier: *conditioning
and one-pass inference span heterogeneous forms, but generalizing the form boundary to
unseen materials is imperfect.* The concrete levers from here are the ones flagged all
along: a **GNS encoder** and more data to close the generalization gap; **through-the-sim**
training on motion (the Warp path) to confirm it outside the labeled setting; and
**multi-material trajectories** (per-particle latents) for scenes that mix forms. And the
honest boundary to state in any writeup: none of this reaches materials *outside* NCLaw's
F-only class (rate/history-dependent), which no amount of conditioning can fix.
