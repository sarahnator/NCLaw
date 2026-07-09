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
    # EXERCISE C3 -- the material continuum with an ORACLE latent.
    # 1. mu, lam = lame(E, nu)
    # 2. F = I + 0.07 * randn(n_mat, n_def, 3, 3)          (use generator g)
    # 3. tau = corotated_cauchy(F, mu, lam)            (broadcast mu, lam to (n, n_def))
    # 4. z = standardized (log E, nu):
    #        ((stack([log E, nu], -1) - zmean) / zstd) broadcast to (n_mat, n_def, 2)
    # 5. return F, z, tau each flattened over (n * n_def)

    g = torch.Generator().manual_seed(seed)
    n_mat = E.shape[0]
    dtype, device = E.dtype, E.device
    
    # 1. Compute Lamé parameters
    mu, lam = lame(E, nu) # Shapes: [n_mat]
    
    # 2. Generate random deformation gradients around identity matrix
    I = torch.eye(3, dtype=dtype, device=device).view(1, 1, 3, 3) # broadcasting dim 
    noise = torch.randn(n_mat, n_def, 3, 3, dtype=dtype, device=device, generator=g)
    F = I + 0.07 * noise # Shape: [n_mat, n_def, 3, 3]
    
    # 3. Compute ground-truth Kirchhoff stress (tau)
    # Broadcast mu and lam to [n_mat, n_def] to match the batch dimensions
    mu_bc = mu.unsqueeze(1).expand(n_mat, n_def)
    lam_bc = lam.unsqueeze(1).expand(n_mat, n_def)
    tau = corotated_cauchy(F, mu_bc, lam_bc) # Shape: [n_mat, n_def, 3, 3]

    # 4. Generate standardized material embedding z
    z_raw = torch.stack([torch.log(E), nu], dim=-1) # Shape: [n_mat, 2]
    z_std = (z_raw - zmean) / zstd # Shape: [n_mat, 2]
    # Broadcast z to cover every deformation scenario
    z = z_std.unsqueeze(1).expand(n_mat, n_def, 2) # Shape: [n_mat, n_def, 2]
    
    # 5. Flatten arrays over (n_mat * n_def) for training consumption
    F_flat = F.reshape(-1, 3, 3)
    z_flat = z.reshape(-1, 2)
    tau_flat = tau.reshape(-1, 3, 3)
    
    return F_flat, z_flat, tau_flat

def train_conditioned(model, F, z, tau, scale, steps=2500, lr=3e-3):    # [C4]
    """One conditioned network across ALL training materials.
       Mirrors train.py: Adam + CosineAnnealingLR, normalized-stress MSE."""
    # EXERCISE C4 -- train ONE conditioned network across all materials.
    # Mirror experiments/train.py: Adam(model.parameters(), lr) + CosineAnnealingLR.
    # Each step:  pred = model(F, z)
    #             loss = MSE(pred, tau / scale)      # normalized-stress loss
    #             loss.backward(); opt.step(); sched.step()

    opt = torch.optim.Adam([
        {"params": model.parameters(), "lr": lr},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    criterion = torch.nn.MSELoss()

    for step in range(steps):
        opt.zero_grad()

        Y = model(F, z)
        loss = criterion(Y, tau / scale)
        loss.backward()
        opt.step()
        sched.step()

def rel_err(model, F, z, tau, scale):
    with torch.no_grad():
        return ((model(F, z) - tau / scale).pow(2).mean().sqrt()
                / (tau / scale).pow(2).mean().sqrt()).item()
