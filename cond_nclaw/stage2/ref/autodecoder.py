"""Autodecoder over a (nu, tauY) family: one shared latent conditions BOTH the
elastic and plastic networks. Labeled setting (stress + plastic-projection targets)."""
import torch
import torch.nn as nn

from .materials import lame, corotated_cauchy, vonmises_return

Z_DIM = 4


def sample_family(n, seed, E=2000.0, nu_range=(0.1, 0.4), tauY_range=(50.0, 300.0)):
    g = torch.Generator().manual_seed(seed)
    nu = torch.empty(n, dtype=torch.float64).uniform_(*nu_range, generator=g)
    tauY = torch.empty(n, dtype=torch.float64).uniform_(*tauY_range, generator=g)
    E_t = torch.full((n,), float(E), dtype=torch.float64)
    params = torch.stack([nu, tauY], -1)
    return E_t, nu, tauY, params


def param_stats(params):
    return params.mean(0), params.std(0)


def build_grouped(E, nu, tauY, n_def, seed):
    g = torch.Generator().manual_seed(seed)
    mu, lam = lame(E, nu)
    n = E.shape[0]
    F = torch.eye(3, dtype=torch.float64).expand(n, n_def, 3, 3) \
        + 0.15 * torch.randn(n, n_def, 3, 3, generator=g, dtype=torch.float64)
    mu_b = mu[:, None].expand(n, n_def)
    tau = corotated_cauchy(F, mu_b, lam[:, None].expand(n, n_def))
    Fp = vonmises_return(F, mu_b, tauY[:, None].expand(n, n_def))
    return F, tau, Fp


class LatentBook(nn.Module):
    def __init__(self, n, z_dim=Z_DIM, std=1.0):
        super().__init__()
        self.z = nn.Parameter(torch.randn(n, z_dim, dtype=torch.float64) * std)

    def forward(self, idx):
        return self.z[idx]


def train_autodecoder(elastic, plastic, book, F, tau, Fp, tau_scale,
                      steps=3000, lr=3e-3, latent_lr_mult=10.0):
    n_mat, n_def = F.shape[0], F.shape[1]
    Ff, tf, Fpf = F.reshape(-1, 3, 3), tau.reshape(-1, 3, 3), Fp.reshape(-1, 3, 3)
    idx = torch.arange(n_mat).repeat_interleave(n_def)
    opt = torch.optim.Adam([
        {'params': list(elastic.parameters()) + list(plastic.parameters()), 'lr': lr},
        {'params': book.parameters(), 'lr': lr * latent_lr_mult},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        opt.zero_grad()
        z = book(idx)
        loss_e = (elastic(Ff, z) - tf / tau_scale).pow(2).mean()
        loss_p = (plastic(Ff, z) - Fpf).pow(2).mean()          # Fp is O(1), no scaling
        (loss_e + loss_p).backward()
        opt.step(); sched.step()
    return elastic, plastic, book


def rel_err_elastic(elastic, book, F, tau, tau_scale):
    n_mat, n_def = F.shape[0], F.shape[1]
    idx = torch.arange(n_mat).repeat_interleave(n_def)
    Ff, tf = F.reshape(-1, 3, 3), tau.reshape(-1, 3, 3)
    with torch.no_grad():
        return ((elastic(Ff, book(idx)) - tf / tau_scale).pow(2).mean().sqrt()
                / (tf / tau_scale).pow(2).mean().sqrt()).item()


def rel_err_plastic(plastic, book, F, Fp):
    n_mat, n_def = F.shape[0], F.shape[1]
    idx = torch.arange(n_mat).repeat_interleave(n_def)
    Ff, Fpf = F.reshape(-1, 3, 3), Fp.reshape(-1, 3, 3)
    with torch.no_grad():
        # measure the plastic CORRECTION, not F itself (most of Fp is just F)
        num = (plastic(Ff, book(idx)) - Fpf).pow(2).mean().sqrt()
        den = (Fpf - Ff).pow(2).mean().sqrt().clamp_min(1e-9)
        return (num / den).item()


def _spearman(a, b):
    ra = a.argsort().argsort().double(); rb = b.argsort().argsort().double()
    ra = (ra - ra.mean()) / ra.std(); rb = (rb - rb.mean()) / rb.std()
    return (ra * rb).mean().item()


def structure_score(latents, params, pmean, pstd):
    p = (params - pmean) / pstd
    n = latents.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    ld = torch.cdist(latents, latents)[iu[0], iu[1]]
    pd = torch.cdist(p, p)[iu[0], iu[1]]
    return _spearman(ld, pd)


def invert_latent(elastic, plastic, F, tau, Fp, tau_scale,
                  z_dim=Z_DIM, steps=1500, lr=3e-2, std=1.0):
    """Freeze BOTH trained nets; optimize a fresh latent to fit a new material's
    observations (elastic stress + plastic projection)."""
    for p in list(elastic.parameters()) + list(plastic.parameters()):
        p.requires_grad_(False)
    Ff, tf, Fpf = F.reshape(-1, 3, 3), tau.reshape(-1, 3, 3), Fp.reshape(-1, 3, 3)
    z = torch.nn.Parameter(torch.randn(z_dim, dtype=torch.float64) * std)
    opt = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        opt.zero_grad()
        zb = z.unsqueeze(0).expand(Ff.shape[0], -1)
        loss = (elastic(Ff, zb) - tf / tau_scale).pow(2).mean() + (plastic(Ff, zb) - Fpf).pow(2).mean()
        loss.backward(); opt.step(); sched.step()
    for p in list(elastic.parameters()) + list(plastic.parameters()):
        p.requires_grad_(True)
    return z.detach()
