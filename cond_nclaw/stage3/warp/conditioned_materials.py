"""
Repo-faithful conditioned elastic/plastic laws (Warp SVD) + a latent codebook.
Same as the Stage-2 warp module. Drop into nclaw/material/ and expose the classes as
nclaw.material.CondInvariantFull{Elasticity,Plasticity}. Plain concat -> equilibrium at z=0.
Set the per-rollout latent via set_latent(z) (z may require grad -> flows to the encoder).
"""
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .meta import MLPBlock, MetaElasticity, MetaPlasticity
from .utils import init_weight


class _CondBase:
    def _build(self, cfg):
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

    def set_latent(self, z):
        self.z = z

    def _frame(self, F):
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
        x = self.unflatten(self.final_layer(x))
        return R, Ft, 0.5 * (self.transpose(x) + x)


class CondInvariantFullElasticity(MetaElasticity, _CondBase):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg); self._build(cfg)

    def forward(self, F):
        R, Ft, T = self._frame(F)
        return torch.matmul(torch.matmul(R, T), Ft)


class CondInvariantFullPlasticity(MetaPlasticity, _CondBase):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg); self._build(cfg)

    def forward(self, F):
        R, Ft, T = self._frame(F)
        return F + self.alpha * torch.matmul(R, T)
