"""
Stage 3: a FEED-FORWARD encoder that maps a material's observations to its latent z
in one pass -- amortizing the per-material optimization that Stage 2 did by inversion.

Encoder: DeepSets over a set of (F, stress) samples (permutation-invariant).
Training: joint (encoder + the two conditioned nets) with
    reconstruction loss   (predicted z must make both nets reproduce the material)
  + contrastive loss      (CLIP-style, using TRUE params as a SOFT similarity label
                           -- Hassan's "gauge similarity with the material parameters")
The contrastive term shapes the latent geometry the Stage-2 autodecoder only stumbled
into. At test time, a new material's trajectory -> z in a single forward pass.
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .materials import lame, corotated_cauchy, vonmises_return

Z_DIM = 4


# ------------------------------- data --------------------------------------
def sample_family(n, seed, E=2000.0, nu_range=(0.1, 0.4), tauY_range=(50.0, 300.0)):
    g = torch.Generator().manual_seed(seed)
    nu = torch.empty(n, dtype=torch.float64).uniform_(*nu_range, generator=g)
    tauY = torch.empty(n, dtype=torch.float64).uniform_(*tauY_range, generator=g)
    E_t = torch.full((n,), float(E), dtype=torch.float64)
    return E_t, nu, tauY, torch.stack([nu, tauY], -1)


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


# ----------------------------- the encoder ---------------------------------
class TrajEncoder(nn.Module):                                # [E1]
    """DeepSets: per-sample feature -> mean pool over samples -> latent."""
    def __init__(self, z_dim=Z_DIM, hidden=128):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(18, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        self.rho = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, z_dim))

    def forward(self, F, tau):
        """F, tau: (M, K, 3, 3). Returns z: (M, z_dim)."""
        # EXERCISE E1 -- DeepSets encoder (permutation-invariant over samples).
        #   x = cat([F.flatten to 9, tau.flatten to 9], -1)   # (M, K, 18)
        #   h = self.phi(x).mean(dim=-2)                        # per-sample feat, POOL over K
        #   return self.rho(h)                                 # (M, z_dim)
        raise NotImplementedError("E1: per-sample phi, mean-pool over samples, rho")


# --------------------------- contrastive loss ------------------------------
def contrastive_loss(za, zb, params_std, temp=0.2, sigma=1.0):   # [E2]
    """CLIP-style soft contrastive. Two views (za, zb) of the M materials in the
    batch. Target affinity comes from TRUE params: same material highest, others
    graded by parameter distance -> enforces smooth, ordered latent geometry."""
    # EXERCISE E2 -- CLIP-style soft contrastive using TRUE params as the label.
    # 1. z1, z2 = normalize(za), normalize(zb);  logits = z1 @ z2.T / temp
    # 2. soft target (NO grad): pd = cdist(params_std, params_std);
    #        target = softmax(-pd^2 / (2 sigma^2), dim=-1)   # diagonal highest, graded off-diag
    # 3. symmetric soft cross-entropy over the two views:
    #        -0.5 * [ (target * log_softmax(logits)).sum(-1).mean()
    #               + (target * log_softmax(logits.T)).sum(-1).mean() ]
    raise NotImplementedError("E2: CLIP-style soft contrastive with param-distance targets")


# ------------------------------ training -----------------------------------
def train_encoder(encoder, elastic, plastic, F, tau, Fp, tau_scale, params_std,   # [E3]
                  steps=3000, lr=2e-3, K=24, lam=1.0):
    M, N = F.shape[0], F.shape[1]
    # EXERCISE E3 -- joint training: encoder + BOTH conditioned nets.
    # Adam over encoder + elastic + plastic params; CosineAnnealingLR.
    # Each step:
    #   perm = randperm(N); i1, i2 = perm[:K], perm[K:2K]       # two disjoint views
    #   za = encoder(F[:,i1], tau[:,i1]); zb = encoder(F[:,i2], tau[:,i2])
    #   reconstruct the i1 samples with za (expand za over the K samples):
    #     loss_recon = MSE(elastic(F[i1], za), tau[i1]/tau_scale)
    #                + MSE(plastic(F[i1], za), Fp[i1])
    #   loss_con = contrastive_loss(za, zb, params_std)
    #   (loss_recon + lam * loss_con).backward(); opt.step(); sched.step()
    # return encoder, elastic, plastic
    raise NotImplementedError("E3: two-view joint training, reconstruction + contrastive")


# --------------------------- amortized inference ---------------------------
def amortized_infer(encoder, F, tau):                        # [E4]
    """A material's observations -> its latent, in ONE forward pass (no optimization)."""
    # EXERCISE E4 -- one forward pass, NO optimization (contrast with Stage-2 inversion).
    #   with torch.no_grad(): return encoder(F, tau)
    raise NotImplementedError("E4: encode observations -> z in a single pass")


# ------------------------------ diagnostics --------------------------------
def recon_with_z(elastic, plastic, z, F, tau, Fp, tau_scale):
    M, N = F.shape[0], F.shape[1]
    Ff, tf, Fpf = F.reshape(-1, 3, 3), tau.reshape(-1, 3, 3), Fp.reshape(-1, 3, 3)
    zf = z[:, None, :].expand(M, N, z.shape[-1]).reshape(-1, z.shape[-1])
    with torch.no_grad():
        e = ((elastic(Ff, zf) - tf / tau_scale).pow(2).mean().sqrt()
             / (tf / tau_scale).pow(2).mean().sqrt()).item()
        num = (plastic(Ff, zf) - Fpf).pow(2).mean().sqrt()
        den = (Fpf - Ff).pow(2).mean().sqrt().clamp_min(1e-9)
        p = (num / den).item()
    return e, p


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
