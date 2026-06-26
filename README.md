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

## Installation
Run
```sh
uv syc
```

Install `nclaw`:

```bash
uv pip install -e . -v
```

## Experiments

Generate dataset:

```bash
uv run experiments/scripts/dataset/main.py
```

Train NCLaw:

```bash
uv run experiments/scripts/train/invariant_full_meta-invariant_full_meta.py 
```

Evaluate NCLaw:

```bash
# Reconstruction
uv run experiments/scripts/eval/dataset.py --gt

# Generalization
uv run experiments/scripts/eval/time.py --gt  # (a) time
uv run experiments/scripts/eval/vel.py --gt   # (b) velocity
uv run experiments/scripts/eval/shape.py --gt # (c) geometry
uv run experiments/scripts/eval/slope.py      # (d) boundary

# Extreme
uv run experiments/scripts/eval/large.py   # (a) one-million
uv run experiments/scripts/eval/contact.py # (b) collision

# Multi-physics
uv run experiments/scripts/eval/pool.py    # (a) coupled-physics
uv run experiments/scripts/eval/melting.py # (b) phase-transition
```

## Results

https://github.com/PingchuanMa/NCLaw/assets/16499005/4860bce0-8d20-4641-9052-ea7430361bca

https://github.com/PingchuanMa/NCLaw/assets/16499005/7ea44615-564a-4fa1-bfd8-0ce5cfdecc31

https://github.com/PingchuanMa/NCLaw/assets/16499005/e1fed91f-da58-4a79-a130-5ccd054091fa

https://github.com/PingchuanMa/NCLaw/assets/16499005/37917f41-522c-4b3b-a845-cb2055b78c92
