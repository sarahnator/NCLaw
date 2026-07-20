"""
Stage 4: HETEROGENEOUS materials -- distinct constitutive FORMS with no shared
parameter vector (elastic / von Mises plastic / fluid). This is where a learned
latent beats explicit parameters, because you can't standardize the parameters
across forms. The similarity signal becomes the material TYPE (discrete), and the
latent should form CLUSTERS by type.
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .materials import lame, corotated_cauchy, vonmises_return

Z_DIM = 4
TYPES = ['elastic', 'plastic', 'fluid']       # type ids 0, 1, 2


# ----------------------- a new form: fluid (H1) ----------------------------
def fluid_cauchy(F, lam):                                   # [H1]
    """Weakly-compressible fluid: corotated with mu = 0 (no shear resistance)."""
    zero = torch.zeros(F.shape[:-2], dtype=F.dtype, device=F.device)
    return corotated_cauchy(F, zero, lam)


def fluid_return(F):                                        # [H1]
    """Volume-preserving projection onto the hydrostatic axis: Fp = J^(1/3) I."""
    J = torch.linalg.det(F)
    I = torch.eye(3, dtype=F.dtype, device=F.device)
    return J.clamp_min(1e-6).pow(1.0 / 3.0)[..., None, None] * I


# ------------------- the heterogeneous family (H2) -------------------------
def sample_hetero(n_per, seed):
    """n_per materials of EACH type. Returns per-material (type, mu, lam, tauY)."""
    g = torch.Generator().manual_seed(seed)
    types, mu, lam, tauY = [], [], [], []
    for t in range(3):
        for _ in range(n_per):
            E = torch.empty((), dtype=torch.float64).uniform_(1500, 3000, generator=g)
            nu = torch.empty((), dtype=torch.float64).uniform_(0.15, 0.4, generator=g)
            m, l = lame(E, nu)
            ty = torch.empty((), dtype=torch.float64).uniform_(80, 250, generator=g)
            types.append(t); mu.append(m); lam.append(l); tauY.append(ty)
    return (torch.tensor(types), torch.stack(mu), torch.stack(lam), torch.stack(tauY))


def build_grouped(types, mu, lam, tauY, n_def, seed):      # [H2]
    """Dispatch each material to its constitutive FORM to make (F, tau, Fp)."""
    g = torch.Generator().manual_seed(seed)
    M = types.shape[0]
    F = torch.eye(3, dtype=torch.float64).expand(M, n_def, 3, 3) \
        + 0.15 * torch.randn(M, n_def, 3, 3, generator=g, dtype=torch.float64)
    tau = torch.empty(M, n_def, 3, 3, dtype=torch.float64)
    Fp = torch.empty(M, n_def, 3, 3, dtype=torch.float64)
    for i in range(M):
        Fi = F[i]; mub = mu[i].expand(n_def); lamb = lam[i].expand(n_def)
        if types[i] == 0:      # elastic: full corotated, no plasticity
            tau[i] = corotated_cauchy(Fi, mub, lamb); Fp[i] = Fi
        elif types[i] == 1:    # von Mises plastic
            tau[i] = corotated_cauchy(Fi, mub, lamb); Fp[i] = vonmises_return(Fi, mub, tauY[i].expand(n_def))
        else:                  # fluid: mu=0 stress, volume projection
            tau[i] = fluid_cauchy(Fi, lamb); Fp[i] = fluid_return(Fi)
    return F, tau, Fp


# ------------------------------ encoder (given) ----------------------------
class TrajEncoder(nn.Module):
    def __init__(self, z_dim=Z_DIM, hidden=128):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(18, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, z_dim))

    def forward(self, F, tau):
        x = torch.cat([F.reshape(*F.shape[:-2], 9), tau.reshape(*tau.shape[:-2], 9)], -1)
        return self.rho(self.phi(x).mean(dim=-2))


# --------------------- type-based contrastive (H3) -------------------------
def type_contrastive(za, zb, types, temp=0.2):             # [H3]
    """Supervised contrastive: same-TYPE materials are positives (no graded param
    signal exists across forms). Target puts equal mass on all same-type materials."""
    z1, z2 = Fn.normalize(za, dim=-1), Fn.normalize(zb, dim=-1)
    logits = z1 @ z2.t() / temp
    same = (types[:, None] == types[None, :]).double()
    target = same / same.sum(-1, keepdim=True)
    lp1 = Fn.log_softmax(logits, dim=-1); lp2 = Fn.log_softmax(logits.t(), dim=-1)
    return -0.5 * ((target * lp1).sum(-1).mean() + (target * lp2).sum(-1).mean())


# ------------------------------ training (given) ---------------------------
def train_encoder(encoder, elastic, plastic, F, tau, Fp, tau_scale, types,
                  steps=2500, lr=2e-3, K=20, lam=0.5):
    M, N = F.shape[0], F.shape[1]
    opt = torch.optim.Adam(list(encoder.parameters()) + list(elastic.parameters())
                           + list(plastic.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        opt.zero_grad()
        perm = torch.randperm(N); i1, i2 = perm[:K], perm[K:2 * K]
        za = encoder(F[:, i1], tau[:, i1]); zb = encoder(F[:, i2], tau[:, i2])
        Fr = F[:, i1].reshape(-1, 3, 3); tr = tau[:, i1].reshape(-1, 3, 3); Fpr = Fp[:, i1].reshape(-1, 3, 3)
        zr = za[:, None, :].expand(M, K, za.shape[-1]).reshape(-1, za.shape[-1])
        loss_recon = ((elastic(Fr, zr) - tr / tau_scale).pow(2).mean()
                      + (plastic(Fr, zr) - Fpr).pow(2).mean())
        (loss_recon + lam * type_contrastive(za, zb, types)).backward()
        opt.step(); sched.step()
    return encoder, elastic, plastic


def amortized_infer(encoder, F, tau):
    with torch.no_grad():
        return encoder(F, tau)


# ------------------------------ metrics ------------------------------------
def per_type_recon(elastic, plastic, z, F, tau, Fp, tau_scale, types):
    M, N = F.shape[0], F.shape[1]
    out = {}
    with torch.no_grad():
        for t in range(3):
            m = types == t
            Ff = F[m].reshape(-1, 3, 3); tf = tau[m].reshape(-1, 3, 3); Fpf = Fp[m].reshape(-1, 3, 3)
            zf = z[m][:, None, :].expand(m.sum(), N, z.shape[-1]).reshape(-1, z.shape[-1])
            e = ((elastic(Ff, zf) - tf / tau_scale).pow(2).mean().sqrt()
                 / (tf / tau_scale).pow(2).mean().sqrt()).item()
            out[TYPES[t]] = e
    return out


def cluster_accuracy(z_tr, types_tr, z_te, types_te):      # [H4]
    """Nearest-type-centroid classification in latent space: can you read a
    material's FORM off its latent? High accuracy = the latent clusters by type."""
    cents = torch.stack([z_tr[types_tr == t].mean(0) for t in range(3)])
    pred = torch.cdist(z_te, cents).argmin(dim=1)
    return (pred == types_te).double().mean().item()
