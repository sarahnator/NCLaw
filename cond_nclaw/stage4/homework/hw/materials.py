import torch
import torch.nn as nn


def proper_svd(F):
    U, S, Vh = torch.linalg.svd(F)
    sign = torch.sign(torch.linalg.det(torch.matmul(U, Vh)))
    U = U.clone(); U[..., :, -1] = U[..., :, -1] * sign.unsqueeze(-1)
    S = S.clone(); S[..., -1] = S[..., -1] * sign
    return U, S, Vh


def lame(E, nu):
    return E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))


def corotated_cauchy(F, mu, lam):
    U, S, Vh = proper_svd(F)
    R = torch.matmul(U, Vh)
    J = torch.linalg.det(F)
    FinvT = torch.linalg.inv(F).transpose(-1, -2)
    P = (2 * mu[..., None, None] * (F - R)
         + lam[..., None, None] * ((J - 1) * J)[..., None, None] * FinvT)
    return torch.matmul(P, F.transpose(-1, -2))


def vonmises_return(F, mu, tauY):
    U, S, Vh = proper_svd(F)
    eps = torch.log(S.clamp_min(1e-6))
    eps_hat = eps - eps.sum(-1, keepdim=True) / 3.0
    norm = eps_hat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    dgamma = norm.squeeze(-1) - tauY / (2 * mu)
    yield_mask = (dgamma > 0).unsqueeze(-1)
    eps_new = eps - (dgamma / norm.squeeze(-1)).unsqueeze(-1) * eps_hat
    S_new = torch.where(yield_mask, torch.exp(eps_new), S)
    Fp = torch.matmul(torch.matmul(U, torch.diag_embed(S_new)), Vh)
    return torch.where(yield_mask.unsqueeze(-1), Fp, F)


class MLPBlock(nn.Module):
    def __init__(self, i, o, act):
        super().__init__(); self.fc = nn.Linear(i, o, bias=False); self.act = act
    def forward(self, x):
        x = self.fc(x); return self.act(x) if self.act is not None else x


class _CondInvariantFull(nn.Module):
    def __init__(self, layer_widths=(64, 64), z_dim=4, rest_correct=False):
        super().__init__()
        self.dim = 3; self.rest_correct = rest_correct; act = nn.GELU()
        width = self.dim + self.dim * self.dim + 1 + z_dim
        layers = []
        for nw in layer_widths:
            layers.append(MLPBlock(width, nw, act)); width = nw
        self.layers = nn.ModuleList(layers)
        self.final_layer = MLPBlock(width, self.dim * self.dim, None)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

    def _invariants(self, F):
        U, sigma, Vh = proper_svd(F); R = torch.matmul(U, Vh)
        Ft = F.transpose(-1, -2); FtF = torch.matmul(Ft, F)
        I = torch.eye(3, dtype=F.dtype, device=F.device)
        I1 = sigma - 1.0; I2 = (FtF - I).reshape(*F.shape[:-2], 9)
        I3 = torch.linalg.det(F).unsqueeze(-1) - 1.0
        return R, Ft, torch.cat([I1, I2, I3], -1)

    def _frame(self, inv, z):
        x = torch.cat([inv, z], -1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x).reshape(*inv.shape[:-1], 3, 3)
        return 0.5 * (x + x.transpose(-1, -2))


class CondElasticity(_CondInvariantFull):
    def forward(self, F, z):
        R, Ft, inv = self._invariants(F)
        return torch.matmul(torch.matmul(R, self._frame(inv, z)), Ft)


class CondPlasticity(_CondInvariantFull):
    def __init__(self, layer_widths=(64, 64), z_dim=4, rest_correct=False, alpha=0.1):
        super().__init__(layer_widths, z_dim, rest_correct); self.alpha = alpha
    def forward(self, F, z):
        R, Ft, inv = self._invariants(F)
        return F + self.alpha * torch.matmul(R, self._frame(inv, z))
