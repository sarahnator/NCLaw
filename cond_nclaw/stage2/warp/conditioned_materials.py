"""
Conditioned elastic/plastic laws for the ORIGINAL repo (Warp SVD).

Append these to nclaw/material/meta.py (or a new file) and make them importable as
nclaw.material.CondInvariantFullElasticity / ...Plasticity (add to __init__.py or
just `from .conditioned_materials import *`). Reference them in the material config
by `cls:` name, and add a `z_dim:` field.

Conditioning = concatenate a per-particle latent `z` (set once per rollout via
set_latent) onto the [sigma-1, FtF-I, detF-1] invariants. PLAIN concatenation, so
zero-stress-at-rest holds at z=0 -- check equilibrium there. (For equilibrium at all
z, subtract the network's output at the rest input, as in the labeled version.)
"""
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .meta import MLPBlock, MetaElasticity, MetaPlasticity
from .utils import init_weight


class CondInvariantFullElasticity(MetaElasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.z_dim = int(cfg.z_dim)
        self.z = None                                    # set per rollout (plain attr)
        self.layers = nn.ModuleList()
        width = self.dim + self.dim * self.dim + 1 + self.z_dim
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width
        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)
        for m in self.modules():
            init_weight(m)

    def set_latent(self, z: torch.Tensor) -> None:
        self.z = z                                       # (z_dim,), may require grad

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device)
        U, sigma, Vh = self.svd(F)                        # repo's Warp SVD
        R = torch.matmul(U, Vh)
        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)
        I1 = sigma - 1.0
        I2 = self.flatten(FtF - I)
        I3 = torch.linalg.det(F).unsqueeze(dim=1) - 1.0
        z = self.z.unsqueeze(0).expand(F.shape[0], -1)    # broadcast to all particles
        x = torch.cat([I1, I2, I3, z], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        x = self.unflatten(x)
        x = 0.5 * (self.transpose(x) + x)
        P = torch.matmul(R, x)
        return torch.matmul(P, Ft)


class CondInvariantFullPlasticity(MetaPlasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.z_dim = int(cfg.z_dim)
        self.z = None
        self.layers = nn.ModuleList()
        width = self.dim + self.dim * self.dim + 1 + self.z_dim
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width
        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)
        for m in self.modules():
            init_weight(m)

    def set_latent(self, z: torch.Tensor) -> None:
        self.z = z

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device)
        U, sigma, Vh = self.svd(F)
        R = torch.matmul(U, Vh)
        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)
        I1 = sigma - 1.0
        I2 = self.flatten(FtF - I)
        I3 = torch.linalg.det(F).unsqueeze(dim=1) - 1.0
        z = self.z.unsqueeze(0).expand(F.shape[0], -1)
        x = torch.cat([I1, I2, I3, z], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        x = self.unflatten(x)
        x = 0.5 * (self.transpose(x) + x)
        delta_Fp = self.alpha * torch.matmul(R, x)
        return delta_Fp + F


class LatentBook(nn.Module):
    """One latent per family material; std=1.0 init."""
    def __init__(self, n_materials: int, z_dim: int, std: float = 1.0) -> None:
        super().__init__()
        self.z = nn.Parameter(torch.randn(n_materials, z_dim) * std)

    def forward(self, idx: int) -> torch.Tensor:
        return self.z[idx]
