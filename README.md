# Neural Constitutive Laws

[![Website](https://img.shields.io/badge/Project%20Page-NCLaw-yellowgreen.svg)](https://sites.google.com/view/nclaw) [![arXiv](https://img.shields.io/badge/arXiv-2304.14369-b31b1b.svg)](https://arxiv.org/abs/2304.14369)

Learning Neural Constitutive Laws from Motion Observations for Generalizable PDE Dynamics

ICML 2023 / [Website](https://sites.google.com/view/nclaw) / [arXiv](https://arxiv.org/abs/2304.14369)

https://github.com/PingchuanMa/NCLaw/assets/16499005/e1fed91f-da58-4a79-a130-5ccd054091fa

```
@inproceedings{ma2023learning,
    title={Learning Neural Constitutive Laws from Motion Observations for Generalizable PDE Dynamics},
    author={Ma, Pingchuan and Chen, Peter Yichen and Deng, Bolei and Tenenbaum, Joshua B and Du, Tao and Gan, Chuang and Matusik, Wojciech},
    booktitle={International Conference on Machine Learning},
    year={2023},
    organization={PMLR}
}
```

## Blackwell RTX 5090 Optimized

This codebase has been updated and optimized for **NVIDIA Blackwell Architecture (RTX 5090 / sm_120)**:

- Ubuntu 20.04 / 22.04
- CUDA Driver 12.8+
- Python 3.10
- PyTorch 2.x (CUDA 12.8 Nightly Build)
- Warp 0.10.1
- NumPy < 2.0

Key Takeaways:
* **`PYTHONNOUSERSITE=1`**: the global user directory (`~/.local`) contains package versions that conflict with the sandbox versions required by Warp.
* **`PYTHONPATH=.`**: Explicitly tells Python to look inside the root directory of `NCLaw` for module paths, removing any structural confusion.

## Installation

1. Prepare the isolated conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate nclaw
```

2. Link NCLaw into the local active environment sandbox:
```bash
pip install -e .
```


## Experiments
When running experiments on this architecture, you must prepend the runtime execution flags. This guarantees that Python isolates its modules to your local sandbox environment and explicitly maps your repository structure.
Generate dataset:

```bash
export CUDA_PATH=$CONDA_PREFIX
PYTHONPATH=. PYTHONNOUSERSITE=1 python experiments/scripts/dataset/main.py
```

Train NCLaw:

```bash
export CUDA_PATH=$CONDA_PREFIX
PYTHONPATH=. PYTHONNOUSERSITE=1 python experiments/scripts/train/invariant_full_meta-invariant_full_meta.py
```

Evaluate NCLaw:

```bash
# Reconstruction
export CUDA_PATH=$CONDA_PREFIX
PYTHONPATH=. PYTHONNOUSERSITE=1 python experiments/scripts/eval/dataset.py --gt

# Generalization
export CUDA_PATH=$CONDA_PREFIX
PYTHONPATH=. PYTHONNOUSERSITE=1 python experiments/scripts/eval/time.py --gt  # (a) time
export CUDA_PATH=$CONDA_PREFIX
PYTHONPATH=. PYTHONNOUSERSITE=1 python experiments/scripts/eval/vel.py --gt   # (b) velocity
export CUDA_PATH=$CONDA_PREFIX
PYTHONPATH=. PYTHONNOUSERSITE=1 python experiments/scripts/eval/shape.py --gt # (c) geometry
export CUDA_PATH=$CONDA_PREFIX
PYTHONPATH=. PYTHONNOUSERSITE=1 python experiments/scripts/eval/slope.py      # (d) boundary

# Extreme
export CUDA_PATH=$CONDA_PREFIX
PYTHONPATH=. PYTHONNOUSERSITE=1 python experiments/scripts/eval/large.py   # (a) one-million
export CUDA_PATH=$CONDA_PREFIX
PYTHONPATH=. PYTHONNOUSERSITE=1 python experiments/scripts/eval/contact.py # (b) collision

# Multi-physics
export CUDA_PATH=$CONDA_PREFIX
PYTHONPATH=. PYTHONNOUSERSITE=1 python experiments/scripts/eval/pool.py    # (a) coupled-physics
export CUDA_PATH=$CONDA_PREFIX
PYTHONPATH=. PYTHONNOUSERSITE=1 python experiments/scripts/eval/melting.py # (b) phase-transition
```

## Results

https://github.com/PingchuanMa/NCLaw/assets/16499005/4860bce0-8d20-4641-9052-ea7430361bca

https://github.com/PingchuanMa/NCLaw/assets/16499005/7ea44615-564a-4fa1-bfd8-0ce5cfdecc31

https://github.com/PingchuanMa/NCLaw/assets/16499005/e1fed91f-da58-4a79-a130-5ccd054091fa

https://github.com/PingchuanMa/NCLaw/assets/16499005/37917f41-522c-4b3b-a845-cb2055b78c92
