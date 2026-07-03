# Porting the conditioning into the real `PingchuanMa/NCLaw` repo

The homework module is a faithful port of `nclaw.material.InvariantFullMetaElasticity`;
the only backend difference is `torch.linalg.svd` in place of the repo's Warp SVD.
The conditioning edits below are identical in both. (Reminder: the repo is pinned to
Warp 0.6.1 / CUDA 11.7 and won't build on Blackwell, so validate the logic here first.)

## 1. `nclaw/material/meta.py` — `InvariantFullMetaElasticity`

**`__init__`:** widen the first layer for the latent.

```python
# was:  width = self.dim + self.dim * self.dim + 1
z_dim = cfg.get('z_dim', 0)
width = self.dim + self.dim * self.dim + 1 + z_dim
self.register_buffer('z', torch.zeros(z_dim))   # set per rollout; see train.py
```

**`forward`:** after building `x = torch.cat([I1, I2, I3], dim=1)`, concatenate the
per-particle latent and apply the rest-state subtraction. The cleanest minimal change
keeps the `forward(F)` signature by reading `self.z`:

```python
z = self.z.unsqueeze(0).expand(x.shape[0], -1)   # broadcast to all particles
x = torch.cat([x, z], dim=1)                      # <-- C1

def _run(feat):                                   # the existing layer stack
    h = feat
    for layer in self.layers:
        h = layer(h)
    h = self.final_layer(h)
    h = self.unflatten(h)
    return 0.5 * (self.transpose(h) + h)

zeros = torch.cat([torch.zeros_like(invariants), z], dim=1)  # invariants at F=I are 0
T = _run(x) - _run(zeros)                          # <-- C2 rest-state subtraction
P = torch.matmul(R, T)
cauchy = torch.matmul(P, Ft)
```

Do the identical edit to `InvariantFullMetaPlasticity` (its rest-state prior is
"preserve `F` at rest", which the same subtraction restores).

## 2. `experiments/train.py` — one conditioned model across materials

The stock loop trains one network on one dataset (`log/<material.name>/dataset`).
For conditioning:

1. **Generate a continuum.** In the dataset generator, sample `(E, ν)` across a grid,
   write one trajectory per material tagged with its params, and reserve a held-out set.
2. **Loop over materials.** Wrap the epoch body so it iterates (or samples) over the
   training materials' datasets. Before each rollout, set the latent:
   ```python
   elasticity.z = standardize(torch.tensor([math.log(E), nu], device=torch_device))
   plasticity.z = elasticity.z
   ```
3. **Everything else is unchanged** — the BPTT rollout, teacher forcing, `loss_factor`,
   `clip_grad_norm_`, Adam + cosine. One shared `elasticity`/`plasticity`, all materials.

`eval.py` calls `elasticity(F)` too, so the `self.z` (buffer) approach means you only
set `z` before a rollout and touch no call sites. Save/restore `z` alongside the
checkpoint so evaluation on a chosen material is reproducible.

## Two cautions carried over from the homework

- Training **through the sim** re-introduces BPTT and the SVD-adjoint-at-rest issue that
  the labeled homework sidesteps (there `F` is data, so no gradient flows through the
  SVD). Expect it to be fussier to optimize — that's gap 6, not a conditioning failure.
- Keep the stiffness range narrow at first (a ~4× span), or the loss is dominated by the
  stiffest materials. This is the same reason the repo carries a hand-tuned `loss_factor`.
