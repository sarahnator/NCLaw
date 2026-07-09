"""
Material continuum + oracle-conditioned training, mirroring the structure of the
repo's experiments/train.py (Adam + cosine schedule + per-material loss) but in
the LABELED setting (supervise on stress), so it runs without the Warp MPM.
"""
import math
import torch

from .materials import lame, corotated_cauchy


def sample_materials(n, seed, E_range=(1e3, 4e3), nu_range=(0.1, 0.4)):
    g = torch.Generator().manual_seed(seed)
    E = torch.exp(torch.empty(n, dtype=torch.float64).uniform_(
        math.log(E_range[0]), math.log(E_range[1]), generator=g))
    nu = torch.empty(n, dtype=torch.float64).uniform_(nu_range[0], nu_range[1], generator=g)
    return E, nu


def z_stats(E, nu):
    zraw = torch.stack([torch.log(E), nu], -1)
    return zraw.mean(0), zraw.std(0)


def build_dataset(E, nu, n_def, zmean, zstd, seed):                     # [C3]
    """Return (F, z, tau) flattened over materials x deformations.
       z is the standardized (log E, nu) -- the ORACLE latent for Stage 1."""
    g = torch.Generator().manual_seed(seed)
    mu, lam = lame(E, nu)
    n = E.shape[0]
    F = torch.eye(3, dtype=torch.float64).expand(n, n_def, 3, 3) \
        + 0.07 * torch.randn(n, n_def, 3, 3, generator=g, dtype=torch.float64)
    tau = corotated_cauchy(F, mu[:, None].expand(n, n_def), lam[:, None].expand(n, n_def))
    zraw = torch.stack([torch.log(E), nu], -1)
    z = ((zraw - zmean) / zstd)[:, None, :].expand(n, n_def, 2)
    return F.reshape(-1, 3, 3), z.reshape(-1, 2), tau.reshape(-1, 3, 3)


def train_conditioned(model, F, z, tau, scale, steps=2500, lr=3e-3):    # [C4]
    """One conditioned network across ALL training materials.
       Mirrors train.py: Adam + CosineAnnealingLR, normalized-stress MSE."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        opt.zero_grad()
        loss = (model(F, z) - tau / scale).pow(2).mean()
        loss.backward()
        opt.step()
        sched.step()
    return model


def rel_err(model, F, z, tau, scale):
    with torch.no_grad():
        return ((model(F, z) - tau / scale).pow(2).mean().sqrt()
                / (tau / scale).pow(2).mean().sqrt()).item()
