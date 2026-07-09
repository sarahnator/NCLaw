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
        # EXERCISE T3 -- the shared codebook (ONE latent per material, feeds both nets).
        #   self.z = nn.Parameter(randn(n, z_dim, float64) * std)   # std=1.0 matters
        raise NotImplementedError("T3: make self.z a learnable (n, z_dim) table")

    def forward(self, idx):
        raise NotImplementedError("T3: look up self.z[idx]")


def train_autodecoder(elastic, plastic, book, F, tau, Fp, tau_scale,
                      steps=3000, lr=3e-3, latent_lr_mult=10.0):
    n_mat, n_def = F.shape[0], F.shape[1]
    Ff, tf, Fpf = F.reshape(-1, 3, 3), tau.reshape(-1, 3, 3), Fp.reshape(-1, 3, 3)
    idx = torch.arange(n_mat).repeat_interleave(n_def)   # material index per sample
    # EXERCISE T4 -- joint autodecoder training (BOTH nets + the shared latents).
    # 1. One Adam, TWO param groups: (elastic + plastic) params at lr, book params
    #    at lr * latent_lr_mult (latents ~10x faster). CosineAnnealingLR.
    # 2. Each step: z = book(idx);
    #      loss = MSE(elastic(Ff, z), tf / tau_scale)   # normalized stress
    #           + MSE(plastic(Ff, z), Fpf)              # Fp is O(1), unscaled
    #      backward; opt.step(); sched.step()
    # 3. return elastic, plastic, book
    raise NotImplementedError("T4: dual-LR Adam, joint elastic+plastic loss")


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
    Ff, tf, Fpf = F.reshape(-1, 3, 3), tau.reshape(-1, 3, 3), Fp.reshape(-1, 3, 3)
    # EXERCISE T5 -- test-time inversion (adapt to a new material, no retraining).
    # 1. FREEZE both nets: for p in elastic.parameters()+plastic.parameters(): requires_grad_(False)
    # 2. z = nn.Parameter(randn(z_dim, float64) * std); Adam([z], lr) + cosine.
    # 3. Each step: broadcast z to (batch, z_dim);
    #      loss = MSE(elastic(...), tf/tau_scale) + MSE(plastic(...), Fpf); backward; step; sched.step()
    # 4. Unfreeze the nets; return z.detach().
    raise NotImplementedError("T5: freeze both nets, optimize a fresh latent")
