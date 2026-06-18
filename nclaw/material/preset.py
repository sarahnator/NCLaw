from typing import Sequence
import math

import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from .abstract import Elasticity, Plasticity, Material


class ComposeMaterial(Material):
    # splits a batch of particles by sections  and routes each section to a different material model, then concatenates the results. 
    # This is useful for simulating multiple interacting materials in one MPM forward pass, e.g. a solid and a fluid, without having to run two separate MPM forward passes and coupling them via boundary conditions, which can be less efficient and more complex to implement.

    def __init__(self, materials: list[Material], sections: Sequence[int]) -> None:
        super().__init__(None)
        self.materials = nn.ModuleList(materials)
        self.sections = sections

    # enables section boundaries to change dynamically (e.g. adding or removing materials mid-simulation)
    def update_sections(self, sections: Sequence[int]) -> None:
        self.sections = sections

    def forward(self, F: Tensor) -> Tensor:
        outs = []
        for m, f in zip(self.materials, torch.split(F, self.sections, dim=0)):
            if f.numel() == 0:
                continue
            outs.append(m(f))
        return torch.cat(outs, dim=0)


class CorotatedElasticity(Elasticity):
# corotated_stress = 2μ * (F - U@Vh) @ Fᵀ
# volume_stress    = λ * J * (J-1) * I
# Intuition: The SVD gives you R = U@Vh, the pure rotation component of F.
# The corotated term F - R measures how far the deformation deviates from a pure rotation — 
# i.e., how much it actually stretches the material. 
# The volume term penalizes J = det(F) deviating from 1 (incompressibility).

# Think of it like a rubber band: it springs back from both squishing and stretching. 
# The name "fixed corotated" comes from the fact that the reference frame is co-rotated with the material, 
# which makes it stable under large rotations (unlike naive linear elasticity, which breaks when things rotate a lot). 
# This is why the paper uses it for weakly compressible fluids — it resists volume change but 
# tolerates large deformations.

# Eqns. 12, 13 in the paper to model purely elastic materials (rubber, biological tissues, and jelly)
# P = 2 * mu * (F - U @ Vh) + lambda * J * (J - 1) * F^{-T}, where R is the rotation matrix from polar decompoistion of F
# Plastic return mapping is the identity (ie no plasticity): \mathcal{P} = I, which means the plasticity correction does nothing and the material is purely elastic. This is a common choice for modeling materials that don't exhibit plastic behavior, like rubber or soft tissues, where we only care about the elastic response to deformation.

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.log_E = nn.Parameter(torch.Tensor([cfg.E]).log())
        self.register_buffer('nu', torch.Tensor([cfg.nu]))

        if cfg.random:
            self.log_E.data.mul_(0.8)

    def forward(self, F: Tensor) -> Tensor:
        E = self.log_E.exp()
        nu = self.nu

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        # warp svd
        U, sigma, Vh = self.svd(F)

        corotated_stress = 2 * mu * torch.matmul(F - torch.matmul(U, Vh), F.transpose(1, 2))

        J = torch.prod(sigma, dim=1).view(-1, 1, 1)
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)
        volume_stress = la * J * (J - 1) * I

        stress = corotated_stress + volume_stress

        return stress


class StVKElasticity(Elasticity):
    # Saint Venant-Kirchhoff elasticity: Elastic with the drucker-praeger yield condition
    # E_strain = 0.5 * (Fᵀ@F - I)   # Green-Lagrange strain
    # stress    = 2μ * F @ E_strain + volume_stress
    # Intuition: Fᵀ@F - I is the Green-Lagrange strain tensor — the classic large-deformation strain 
    # measure from continuum mechanics. It's zero when F is a pure rotation (since Rᵀ@R = I), so it 
    # correctly measures only the actual stretching.
    #
    # StVK is essentially "Hooke's law for large deformations." It works well for moderate deformations
    # of stiff objects but has a known flaw: under extreme compression it can invert (pass through itself), 
    # producing unphysical negative stiffness. The paper pairs this with Drucker-Prager plasticity 
    # for granular materials like sand, where the Drucker-Prager yield surface prevents extreme
    # compression from occurring in the first place.

    # (eq. 14) P = U(2 * mu * epsilon + lambda * trace(epsilon) * I) Vh, where epsilon = log(sigma) is the logarithmic strain (also called Hencky strain), which is a more accurate measure of strain for large deformations than the Green-Lagrange strain used in the corotated model. The logarithmic strain captures the true geometric nonlinearity of deformation, making it more suitable for materials that undergo significant stretching or compression.
    # Plastic return mapping is the Drucker-Prager yield condition, eqns. 15, 16

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.log_E = nn.Parameter(torch.Tensor([cfg.E]).log())
        self.register_buffer('nu', torch.Tensor([cfg.nu]))

        if cfg.random:
            self.log_E.data.mul_(0.8)

    def forward(self, F: Tensor) -> Tensor:
        E = self.log_E.exp()
        nu = self.nu

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        # warp svd
        U, sigma, Vh = self.svd(F)

        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)
        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)

        E = 0.5 * (FtF - I)

        stvk_stress = 2 * mu * torch.matmul(F, E)

        J = torch.prod(sigma, dim=1).view(-1, 1, 1)
        volume_stress = la * J * (J - 1) * I

        stress = stvk_stress + volume_stress

        return stress


class VolumeElasticity(Elasticity):
    # pressure-only / equation of state
     
    # J = det(F)
    # Ziran mode: bulk modulus equation of state
    # stress = κ * (J - J^(1-γ)) * I
    # Taichi mode (simpler):
    # stress = λ * J * (J-1) * I (eq. 20)
    # Intuition: Only the volume change (J = det(F)) drives stress — there's no resistance to shear at all. 
    # This models an ideal fluid: fluids flow freely (no shear resistance) but resist compression. 
    # The "Ziran" mode uses a proper equation of state from fluid mechanics with a bulk modulus κ; 
    # "Taichi" is a simpler approximation.

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.log_E = nn.Parameter(torch.Tensor([cfg.E]).log())
        self.register_buffer('nu', torch.Tensor([cfg.nu]))

        if cfg.random:
            self.log_E.data.mul_(0.8)

        self.mode = cfg.mode

    def forward(self, F: Tensor) -> Tensor:
        E = self.log_E.exp()
        nu = self.nu

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        J = torch.det(F).view(-1, 1, 1)
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)

        if self.mode.casefold() == 'ziran':

            #  https://en.wikipedia.org/wiki/Bulk_modulus
            kappa = 2 / 3 * mu + la

            # https://github.com/penn-graphics-research/ziran2020/blob/master/Lib/Ziran/Physics/ConstitutiveModel/EquationOfState.h
            # using gamma = 7 would have gradient issue, fix later
            gamma = 2

            stress = kappa * (J - 1 / torch.pow(J, gamma-1)) * I

        elif self.mode.casefold() == 'taichi':

            stress = la * J * (J - 1) * I

        else:
            raise ValueError('invalid mode for volume plasticity: {}'.format(self.mode))

        return stress


class SigmaElasticity(Elasticity):
        # Hencky / neo-Hookean in log space

# ε = log(σ)                    # Hencky (true) strain — log of singular values
# τ = 2μ*ε + λ*sum(ε)*I         # Kirchhoff stress in principal frame
# stress = U @ diag(τ) @ Uᵀ 
# Intuition: Instead of measuring stretch linearly, this measures it logarithmically.
#  That's physically motivated: doubling the stretch from 1→2 should feel the same as doubling it 
# from 2→4 (a relative change). Log-strain also has the property that compression and extension 
# are symmetric, and it's much more stable under large deformations than StVK.

# This is essentially a neo-Hookean model reformulated in principal-stretch space. 
# Working in the singular-value frame means the stress is automatically frame-invariant.
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.log_E = nn.Parameter(torch.Tensor([cfg.E]).log())
        self.register_buffer('nu', torch.Tensor([cfg.nu]))

        if cfg.random:
            self.log_E.data.mul_(0.8)

    def forward(self, F: Tensor) -> Tensor:
        E = self.log_E.exp()
        nu = self.nu

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        # warp svd
        U, sigma, Vh = self.svd(F)

        epsilon = sigma.log()
        trace = epsilon.sum(dim=1, keepdim=True)
        tau = 2 * mu * epsilon + la * trace

        # eq. 14
        stress = torch.matmul(torch.matmul(U, torch.diag_embed(tau)), self.transpose(U))

        return stress



class IdentityPlasticity(Plasticity):
    def forward(self, F: Tensor) -> Tensor:
        return F


class SigmaPlasticity(Plasticity):

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def forward(self, F: Tensor) -> Tensor:
        J = torch.det(F)

        # unilateral incompressibility: https://github.com/penn-graphics-research/ziran2020/blob/master/Lib/Ziran/Physics/PlasticityApplier.cpp#L1084
        # J = torch.clamp(J, min=0.05, max=1.2)

        # eq 21?
        Je_1_3 = torch.pow(J, 1.0 / 3.0).view(-1, 1).expand(-1, 3)
        F = torch.diag_embed(Je_1_3)
        return F


class VonMisesPlasticity(Plasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.log_E = nn.Parameter(torch.Tensor([cfg.E]).log())
        self.register_buffer('nu', torch.Tensor([cfg.nu]))
        self.sigma_y = nn.Parameter(torch.Tensor([cfg.sigma_y]))

        if cfg.random:
            self.log_E.data.mul_(0.8)
            self.sigma_y.data.mul_(0.8)

    def forward(self, F: Tensor) -> Tensor:

        E = self.log_E.exp()
        nu = self.nu
        sigma_y = self.sigma_y

        mu = E / (2 * (1 + nu))

        # warp svd
        U, sigma, Vh = self.svd(F)

        # prevent NaN
        thredhold = 0.05
        sigma = torch.clamp_min(sigma, thredhold)

        epsilon = torch.log(sigma)
        trace = epsilon.sum(dim=1, keepdim=True)
        epsilon_hat = epsilon - trace / self.dim
        epsilon_hat_norm = torch.linalg.norm(epsilon_hat, dim=1, keepdim=True)

        # eq 18
        delta_gamma = epsilon_hat_norm - sigma_y / (2 * mu)
        cond_yield = (delta_gamma > 0).view(-1, 1, 1)

        #  eq 19
        yield_epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        yield_F = torch.matmul(torch.matmul(U, torch.diag_embed(yield_epsilon.exp())), Vh)

        F = torch.where(cond_yield, yield_F, F)

        return F


class DruckerPragerPlasticity(Plasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.log_E = nn.Parameter(torch.Tensor([cfg.E]).log())
        self.register_buffer('nu', torch.Tensor([cfg.nu]))
        self.friction_angle = nn.Parameter(torch.Tensor([cfg.friction_angle]))
        self.register_buffer('cohesion', torch.Tensor([cfg.cohesion]))

        if cfg.random:
            self.log_E.data.mul_(0.8)
            self.friction_angle.data.mul_(0.8)

    def forward(self, F: Tensor) -> Tensor:

        E = self.log_E.exp()
        nu = self.nu
        friction_angle = self.friction_angle
        sin_phi = torch.sin(torch.deg2rad(friction_angle))
        alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
        cohesion = self.cohesion

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        # warp svd
        U, sigma, Vh = self.svd(F)

        # prevent NaN
        thredhold = 0.05
        sigma = torch.clamp_min(sigma, thredhold)

        epsilon = torch.log(sigma)
        trace = epsilon.sum(dim=1, keepdim=True)
        epsilon_hat = epsilon - trace / self.dim
        epsilon_hat_norm = torch.linalg.norm(epsilon_hat, dim=1, keepdim=True)

        expand_epsilon = torch.ones_like(epsilon) * cohesion

        # eq. 16
        shifted_trace = trace - cohesion * self.dim
        cond_yield = (shifted_trace < 0).view(-1, 1)

        delta_gamma = epsilon_hat_norm + (self.dim * la + 2 * mu) / (2 * mu) * shifted_trace * alpha
        compress_epsilon = epsilon - (torch.clamp_min(delta_gamma, 0.0) / epsilon_hat_norm) * epsilon_hat

        epsilon = torch.where(cond_yield, compress_epsilon, expand_epsilon)

        F = torch.matmul(torch.matmul(U, torch.diag_embed(epsilon.exp())), Vh)

        return F
