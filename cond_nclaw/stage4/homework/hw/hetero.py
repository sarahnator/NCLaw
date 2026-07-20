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
    # EXERCISE H1a -- reuse corotated_cauchy with a ZERO shear modulus.
    #   zero = torch.zeros(F.shape[:-2], ...); return corotated_cauchy(F, zero, lam)
    zero = torch.zeros(F.shape[:-2], dtype=F.dtype, device=F.device)
    return corotated_cauchy(F, zero, lam)


def fluid_return(F):                                        # [H1]
    """Volume-preserving projection onto the hydrostatic axis: Fp = J^(1/3) I."""
    # EXERCISE H1b -- Fp = det(F)^(1/3) * I  (project onto the hydrostatic axis)
    J = torch.linalg.det(F)
    I = torch.eye(3, dtype=F.dtype, device=F.device)

    # If F is (M, n_def, 3, 3), scalars becomes (M, n_def)
    scalars = J.clamp_min(1e-6).pow(1.0 / 3.0)
   
    # i, j = spatial matrix dimensions (3, 3)
    # '... , ij -> ...ij' keeps ALL leading batch dimensions intact (M, n_def)
    # and scales the trailing 3x3 identity matrix across them 
    # as opposed to hardcoding a batch dim (b) and doing
    # 'b, ij -> bij' to multiply each batch scalar 'b' across every 'i, j' pair
    Fp = torch.einsum('..., ij -> ...ij', scalars, I)
    
    return Fp

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
    # EXERCISE H2 -- dispatch each material to its constitutive FORM:
    #   type 0 elastic : tau = corotated_cauchy(Fi, mu, lam);  Fp = Fi  (no plasticity)
    #   type 1 plastic : tau = corotated_cauchy(Fi, mu, lam);  Fp = vonmises_return(Fi, mu, tauY[i])
    #   type 2 fluid   : tau = fluid_cauchy(Fi, lam);          Fp = fluid_return(Fi)
    # (mu[i], lam[i], tauY[i] each .expand(n_def) as needed)

    # 1. CREATE BOOLEAN MASKS
    # These create 1D boolean vectors of length M (e.g., [True, False, True, ...])
    mask0 = (types == 0)
    mask1 = (types == 1)
    mask2 = (types == 2)

    # 2. VECTORIZED ALIGNMENT FOR MATERIAL PARAMETERS
    # Use [..., None] to expand 1D material property arrays (M,) 
    # to perfectly match the (M, n_def) indexing layout
    mu_expanded = mu[..., None]
    lam_expanded = lam[..., None]
    tauY_expanded = tauY[..., None]

    # --- TYPE 0: PURE ELASTIC ---
    if mask0.any():
        F_0 = F[mask0]  # Extracts a batch shape of (M_type0, n_def, 3, 3)
        tau[mask0] = corotated_cauchy(F_0, mu_expanded[mask0], lam_expanded[mask0])
        Fp[mask0] = F_0

    # --- TYPE 1: VON MISES PLASTIC ---
    if mask1.any():
        F_1 = F[mask1]  # Extracts a batch shape of (M_type1, n_def, 3, 3)
        tau[mask1] = corotated_cauchy(F_1, mu_expanded[mask1], lam_expanded[mask1])
        Fp[mask1] = vonmises_return(F_1, mu_expanded[mask1], tauY_expanded[mask1])

    # --- TYPE 2: WEAKLY COMPRESSIBLE FLUID ---
    if mask2.any():
        F_2 = F[mask2]  # Extracts a batch shape of (M_type2, n_def, 3, 3)
        tau[mask2] = fluid_cauchy(F_2, lam_expanded[mask2])
        Fp[mask2] = fluid_return(F_2)

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
    # EXERCISE H3 -- supervised (type-based) contrastive.
    # No graded param signal exists across forms, so POSITIVES are same-TYPE materials.
    # 1. z1,z2 = normalize(za),normalize(zb); logits = z1 @ z2.T / temp
    # 2. same = (types[:,None]==types[None,:]).double();  target = same / same.sum(-1,keepdim=True)
    # 3. symmetric soft cross-entropy: -0.5*[(target*log_softmax(logits)).sum(-1).mean()
    #                                        +(target*log_softmax(logits.T)).sum(-1).mean()]
    z1, z2 = torch.nn.functional.normalize(za, dim=-1), torch.nn.functional.normalize(zb, dim=-1)
    logits = z1 @ z2.mT / temp
    same = (types[:, None] == types[None, :]).double()
    target = same / same.sum(-1, keepdim=True)

    # log probabilities
    lp1, lp2 = torch.log_softmax(logits, dim=-1), torch.log_softmax(logits.mT, dim=-1)

    # cross entropy
    l1 = -1 * (target * lp1).sum(dim=-1).mean()
    l2 = -1 * (target * lp2).sum(dim=-1).mean()

    loss = 0.5 * (l1 + l2)
    return loss


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
    # EXERCISE H4 -- nearest-type-centroid classification (does z cluster by form?).
    # 1. centroids: mean latent of each type over the TRAIN materials -> (3, z_dim)
    # 2. assign each test latent to the nearest centroid (cdist -> argmin)
    # 3. return accuracy vs the true test types
    raise NotImplementedError("H4: nearest-centroid type accuracy")
