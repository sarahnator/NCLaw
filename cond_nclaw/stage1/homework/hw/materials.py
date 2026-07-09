"""
3D conditioned elasticity -- a faithful port of the original repo's
`nclaw.material.InvariantFullMetaElasticity`, extended with a material latent z.

Differences from the repo class (and ONLY these):
  * z_dim is added to the first-layer input width and z is concatenated onto the
    rotation invariants before the MLP;
  * a rest-state subtraction makes stress(F=I, z) = 0 for EVERY z (plain
    concatenation would break the repo's zero-stress-at-rest prior);
  * torch.linalg.svd stands in for the repo's Warp SVD (identical math, but runs
    anywhere -- the repo keeps its Warp SVD).

Everything else -- the invariant set (sigma-1, FtF-I, detF-1, width 13), the
symmetrize-then-rotate-R-back construction, cauchy = P @ F^T -- matches the repo.
"""
import torch
import torch.nn as nn


def proper_svd(F):
    """SVD with a proper-rotation fix so det(U @ Vh) = +1 (mirrors Warp's svd3)."""
    U, S, Vh = torch.linalg.svd(F)
    sign = torch.sign(torch.linalg.det(torch.matmul(U, Vh)))
    U = U.clone(); U[..., :, -1] = U[..., :, -1] * sign.unsqueeze(-1)
    S = S.clone(); S[..., -1] = S[..., -1] * sign
    return U, S, Vh


def lame(E, nu):
    return E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))


def corotated_cauchy(F, mu, lam):
    """Ground-truth 3D fixed-corotated Kirchhoff stress tau = P F^T (the quantity
    the repo's network outputs as `cauchy`)."""
    U, S, Vh = proper_svd(F)
    R = torch.matmul(U, Vh)
    J = torch.linalg.det(F)
    FinvT = torch.linalg.inv(F).transpose(-1, -2)
    P = (2 * mu[..., None, None] * (F - R)
         + lam[..., None, None] * ((J - 1) * J)[..., None, None] * FinvT)
    return torch.matmul(P, F.transpose(-1, -2))


class MLPBlock(nn.Module):
    def __init__(self, i, o, no_bias, act):
        super().__init__()
        self.fc = nn.Linear(i, o, bias=not no_bias)
        self.act = act

    def forward(self, x):
        x = self.fc(x)
        return self.act(x) if self.act is not None else x


class CondInvariantFullElasticity(nn.Module):
    def __init__(self, layer_widths=(64, 64), z_dim=2, no_bias=True, normalize_input=True):
        super().__init__()
        self.dim = 3
        self.z_dim = z_dim
        self.normalize_input = normalize_input
        act = nn.GELU()
        # repo width = dim + dim*dim + 1 = 13; conditioning adds z_dim.  [C1]
        width = self.dim + self.dim * self.dim + 1 + z_dim
        layers = []
        for nw in layer_widths:
            layers.append(MLPBlock(width, nw, no_bias, act)); width = nw
        self.layers = nn.ModuleList(layers)
        self.final_layer = MLPBlock(width, self.dim * self.dim, no_bias, None)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

    def _invariants(self, F):
        """Rotation R (to rotate the answer back) and the 13-dim invariant vector."""
        U, sigma, Vh = proper_svd(F)
        R = torch.matmul(U, Vh)
        Ft = F.transpose(-1, -2)
        FtF = torch.matmul(Ft, F)
        I = torch.eye(3, dtype=F.dtype, device=F.device)
        if self.normalize_input:
            I1 = sigma - 1.0
            I2 = (FtF - I).reshape(*F.shape[:-2], 9)
            I3 = torch.linalg.det(F).unsqueeze(-1) - 1.0
        else:
            I1 = sigma
            I2 = FtF.reshape(*F.shape[:-2], 9)
            I3 = torch.linalg.det(F).unsqueeze(-1)
        return R, Ft, torch.cat([I1, I2, I3], -1)

    def _frame_stress(self, inv, z):                       # [C1]: concatenate z, run MLP, symmetrize")
        
        # EXERCISE C1 -- the conditioning injection.
        #   x = cat([inv, z], -1)              # (..., 13 + z_dim)
        #   x -> self.layers -> self.final_layer -> reshape (..., 3, 3)
        #   return 0.5 * (x + x^T)             # symmetrize in the invariant frame

        x = torch.cat([inv, z], dim = -1)

        for layer in self.layers:
            x = layer(x)
        T = self.final_layer(x).reshape(-1, self.dim, self.dim)
        return 0.5 * (T + T.mT)

    def forward(self, F, z):                               # [C2]
        # EXERCISE C2 -- rest-state subtraction, then assemble.
        # 1. R, Ft, inv = self._invariants(F)
        # 2. Plain concat breaks zero-stress-at-rest (inv=0 at F=I but z!=0), so
        #    subtract the frame stress at the REST input for the SAME z:
        #        T = self._frame_stress(inv, z) - self._frame_stress(0*inv, z)
        # 3. P = R @ T ;  return P @ Ft        # cauchy = P F^T, as in the repo

        R, Ft, inv = self._invariants(F)
        T = self._frame_stress(inv, z) - self._frame_stress(0 * inv, z)
        P = R @ T
        cauchy = P @ Ft
        return cauchy