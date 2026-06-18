# elastoplastic base class for neural constitutive models and classical constitutive models
# Elastoplastic means the material combines both behaviors:
# - Elastic: reversible deformation — stretch it, it springs back (stores elastic energy)
# - Plastic: irreversible deformation — stress past the yield surface permanently changes shape (dissipates energy)
# The transition between the two is governed by a yield condition. Most real solids are elastoplastic: metals bend and stay bent (plastic), but first flex slightly (elastic). Clay deforms permanently, but still has some springback.



import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from einops.layers.torch import Rearrange
from nclaw.warp import SVD


class Material(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.dim = 3 # 3D space

        # SVD and transpose are GPU-accelerated primitives for learned and classical materials
        # Deformation gradient F is decomposed into rotation and stretch via SVD, which is used in many constitutive models, e.g. corotated elasticity, to compute stress from deformation
        # 1. Many models in meta.py are isotropic materials that use only the stretch (singular values) for computing stress, so we can precompute the SVD and pass the singular values to the material model, which is more efficient than computing SVD multiple times in the same forward pass
        # 2. SVD is also used in the Polar Decomposition to extract pure rotation (R = U * V^T) and symmetric stretch (S = V * Sigma * V^T) from F
        self.svd = SVD()

        # batched 3x3 matrix transpose for computing: 
        # 1. Cauchy stress from first Piola-Kirchhoff stress: sigma = P * F^T
        # 2. Symmetrizing the stress for isotropic materials: sigma = 0.5 * (sigma + sigma^T)
        # 3. Right Cauchy-Green deformation tensor: C = F^T @ F, which is used in StVK elasticity and strain energy invariant, e.g. InvariantMetaElasticity computations (Invariants are functions of C, not F, for isotropic materials)
        self.transpose = Rearrange('b d1 d2 -> b d2 d1', d1=self.dim, d2=self.dim)

    def forward(self, F: Tensor) -> Tensor:
        raise NotImplementedError

# abstract subclasses for elasticity and plasticity, for the role of the material model in the constitutive update of MPM
class Elasticity(Material):
    def forward(self, F: Tensor) -> Tensor:
        # F -> P: maps deformation gradient to first Piola-Kirchhoff stress (What force does this deformation cause?)
        raise NotImplementedError


class Plasticity(Material):
    def forward(self, F: Tensor) -> Tensor:
        # F -> F: plasticity correction maps deformation gradient to updated deformation gradient (projects deformation onto yield surface, a feasible elastic state)
        raise NotImplementedError
