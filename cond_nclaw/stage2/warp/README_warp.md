# Stage 2 through the Warp sim (your machine)

Trains the conditioned elastic+plastic pair across a material family on MOTION only,
by back-propagating through the differentiable MPM. I could not run this here (no
Warp), so it's faithful to `experiments/train.py` but unverified — sanity-check the
first epoch.

## 1. Install the conditioned materials
Copy `conditioned_materials.py` into `nclaw/material/`, then make the classes
importable, e.g. add to `nclaw/material/__init__.py`:
```python
from .conditioned_materials import CondInvariantFullElasticity, CondInvariantFullPlasticity
```
Add material configs (`configs/env/blob/material/{elasticity,plasticity}/cond_invariant_full.yaml`)
that mirror the `invariant_full_meta` ones but with `cls: CondInvariantFull...` and a
`z_dim: 4` field.

## 2. Generate the family datasets
Each family material is one analytical trajectory sharing the SAME geometry + initial
velocity (only the material differs). Vary Poisson ratio `nu` (elastic shape) and von
Mises yield `tauY` (plastic behaviour) — both networks then have something to condition
on. For each `(nu, tauY)`, run the stock dataset generator with those analytical
parameters and save to `log/<name>/dataset` (reuse the `jelly`/`plasticine` presets,
overriding `nu` and the yield stress). Record each material's params in `family.json`:
```json
[{"name": "fam_00", "params": [0.10, 80.0]},
 {"name": "fam_01", "params": [0.25, 150.0]},
 {"name": "fam_02", "params": [0.40, 250.0]}]
```
Keep the stiffness (E) fixed; hold out a few `(nu, tauY)` for a test set.

## 3. Train
```bash
uv run python experiments/train_conditioned_sim.py \
    env=jelly sim=low \
    env/blob/material/elasticity=cond_invariant_full \
    env/blob/material/plasticity=cond_invariant_full \
    +z_dim=4 +family_spec=$PWD/family.json \
    name=cond/stage2_sim
```

## What to watch (labeled vs through-sim)
The labeled version (`../labeled/`) is the clean upper bound — direct stress
supervision, ~1–3% reconstruction. Through the sim you're optimizing PARTICLE
POSITIONS after a 1000-step BPTT rollout, so expect it to be **messier**:
- slower and noisier convergence; watch `|grad|` — the SVD-adjoint-at-rest and
  long-horizon gradients make this fussy (`error_if_nonfinite=True` will hard-stop on
  a NaN, which is a signal, not a nuisance);
- it needs NO stress labels — only motion — which is the whole point of NCLaw;
- the `struct` readout each epoch is the emergent latent structure (spearman of latent
  distances vs `(nu, tauY)` distances). Compare its final value to the labeled version:
  through-sim structure is typically **weaker and noisier**, which reinforces why Stage
  3's contrastive loss is worth adding.

Equilibrium: these use PLAIN concatenation, so zero-stress-at-rest holds at z=0. Check
it by evaluating a material with its latent zeroed. (For equilibrium at every z,
subtract the net's rest output — the `rest_correct` path in the labeled module.)
