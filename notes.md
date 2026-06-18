
### `nclaw/assets/` — 3D mesh files

These are the `.obj` files for all the objects the paper simulates: `bunny.obj`, `dragon.obj`, `spot.obj`, `blub.obj`, `armadillo.obj`, `sphere.obj`, `cube.obj`, `cylinder.obj`, etc. They're the input geometry that gets **seeded with material points** at the start of each simulation. You've seen these objects in the paper's figures — the jelly bunny, the sand dragon, the elastic spot (the cow). They are purely static assets; no equations live here.

---

### `nclaw/extern/VolumeSampling` — external binary for particle seeding

This is a **compiled C++ binary** (not Python), and `sph.py` is the thin Python wrapper that calls it via `subprocess`. "SPH" here is a misnomer in a sense — it's not smoothed-particle hydrodynamics the fluid solver; it's a volume sampling algorithm that uses an SPH-like spacing approach to scatter material points uniformly inside a 3D mesh. When you run `dataset/main.py`, this binary takes a `.obj` file and places particles inside it at a given radius/spacing, outputting their initial positions. It's a preprocessing step, entirely upstream of any simulation or learning.

So `sph.py` does exactly this:

```python
# Call the external binary to fill a mesh with particles
volume_sampling(input_path="bunny.obj", output_path="bunny_particles.npz",
                radius=0.01, ...)
```

That's its entire job. You can treat it as a black box.

---

### `nclaw/render.py` — camera position math, nothing more

This is not a rendering engine. It's just two functions, `get_camera()` and `get_slope_camera()`, that compute the 3D camera origin/target coordinates for PyVista (the visualization library). The actual rendering is done by PyVista externally — `render.py` just hardcodes the viewpoint angles used in the paper's figures. You can completely ignore this file; it has zero scientific content.

---

### `nclaw/warp/` — the differentiability bridge (the most important "infrastructure" file)

This is **not** the Warp library itself. It's NCLaw's custom wrapper around two Warp features:

**`warp/svd.py` — differentiable SVD**
This is genuinely important. It implements `SVD` as a PyTorch `autograd.Function`, meaning gradients can flow through it during backprop. Internally it uses Warp's `wp.svd3` GPU kernel to decompose each deformation gradient F = UΣVᵀ. The determinant sign-fixing logic (the `if U_p_det < 0` blocks) ensures U and V are proper rotation matrices (det = +1), which is the standard "polar" convention used in MPM. Every constitutive network passes F through this SVD first — this is the mechanism that enforces rotation equivariance described in Section 3.2 of the paper.

**`warp/tape.py` — custom gradient tape**
A thin subclass of Warp's `wp.Tape` that handles gradient bookkeeping across the Warp/PyTorch boundary. When NCLaw runs the MPM time step inside a `with tape:` block, all intermediate GPU computations are recorded so that `.backward()` can propagate gradients back through the physics. This is the implementation backbone of differentiable simulation — Eq. (4)'s gradient `∂L/∂θ` flows through here.

---

### `nclaw/material/` — the heart of the paper

This folder contains both the **classic** and **neural** constitutive laws, and is directly the implementation of Equation (3) and Section 3.2.

**`material/abstract.py`** — defines the base classes:
- `Material(nn.Module)` — base; owns the `SVD` module
- `Elasticity(Material)` — interface for 𝓔: F → P
- `Plasticity(Material)` — interface for 𝓟: F_trial → F_new

**`material/preset.py`** — classic (non-neural) constitutive laws. These are the **baselines** and the **Appendix B** examples. Key ones:
- `CorotatedElasticity` — the standard MPM elastic model; computes P from F using U, Σ, Vᵀ
- `StVKElasticity` — Saint Venant–Kirchhoff elasticity
- `SigmaElasticity` — Hencky (log-strain) elasticity, operates on log(σ) of the SVD singular values
- `VonMisesPlasticity` — yield-surface plasticity for plasticine-like materials; projects σ onto the von Mises yield cone
- `DruckerPragerPlasticity` — granular plasticity for sand; uses friction angle
- `IdentityPlasticity` — no plasticity (purely elastic materials like jelly/water)

Notice that even the classic laws use `self.svd(F)` — they go through the same differentiable SVD. This is what makes system identification (finding E and ν from motion data) possible even with classic models.

**`material/meta.py`** — the **neural** constitutive laws. These are the paper's contribution. Key classes:
- `PlainMetaElasticity` — naive MLP baseline: flattens F to 9 numbers, runs through MLP, outputs P. No physics priors. Used as an ablation.
- `PolarMetaElasticity` — uses polar decomposition (R and S from SVD) as input. Partially equivariant.
- `InvariantMetaElasticity` (likely the main one, referenced in `invariant_full_meta` configs) — uses the **invariants of F** (the singular values σ₁, σ₂, σ₃ from SVD) as input to the MLP, outputs stress in the rotated frame, then rotates back via U and Vᵀ. This is the architecture of Section 3.2. The "full" variant handles both elasticity and plasticity in this form.

The naming in the configs — `invariant_full_meta` — tells you: `invariant` = SVD-based rotation-equivariant architecture, `full` = both elastic and plastic NNs are neural (not one neural + one classic), `meta` = trained with meta-learning.

---

### `third_party/warp/` — NVIDIA Warp (the physics engine backend)

This is a vendored (bundled) copy of NVIDIA's Warp library, included because NCLaw uses a specific patched version. It's a complete GPU-accelerated differentiable simulation framework. The key subfolders inside it:

- `warp/native/` — C++/CUDA source: `svd.h` (GPU SVD kernel), `mat33.h`, `warp.cu`, etc. This is where the actual math runs on GPU.
- `warp/sim/integrator_xpbd.py` — bundled XPBD integrator (not used by NCLaw, but relevant to your own research)
- `warp/tape.py` — Warp's base tape for autodiff (NCLaw's `nclaw/warp/tape.py` subclasses this)
- `examples/example_sph.py` — Warp's own SPH fluid demo (completely separate from NCLaw's `sph.py`)

You don't need to read the Warp internals to understand NCLaw. Just know: when you see `wp.launch(...)`, that's running a CUDA kernel. When you see `with tape:`, that's recording operations for reverse-mode autodiff through those CUDA kernels.

---

### How it all fits together — the complete picture

```
PREPROCESSING
  assets/*.obj  →  extern/VolumeSampling (via sph.py)  →  particle positions (npz)

SIMULATION + TRAINING
  particle positions
      ↓
  sim/mpm.py  (MPM time-stepper, Warp kernels, with tape)
      calls → material/meta.py  (neural 𝓔 and 𝓟, using warp/svd.py)
      ↓
  particle positions at t+1
      ↓
  train/loss.py  (Eq. 4: compare to ground-truth positions)
      ↓
  train/teacher.py  (meta-learning outer/inner loop schedule)
      ↓
  backprop through tape → gradients → update neural net weights

VISUALIZATION
  render.py  (just camera math)  →  PyVista renders particle clouds
```

The `data/dataset.py` stores and loads the ground-truth trajectories. The `constants.py` holds physical constants (gravity, etc.). `utils.py` has miscellaneous helpers. `ffmpeg.py` is for stitching frames into videos.

---

The folder you should read first is `nclaw/material/` — specifically `abstract.py`, then `preset.py` (to understand the classic laws the neural ones are replacing), then `meta.py` (to see the actual neural architectures). After that, `nclaw/warp/svd.py` to understand how rotation equivariance is enforced mechanically. Everything else is support infrastructure.