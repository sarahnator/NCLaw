"""
Trajectory encoder for the through-sim setting: a material's MOTION (particle
positions over time) -> its latent z, permutation-invariant over particles.

This default is a DeepSets over per-particle motion fingerprints -- enough to get the
pipeline running. Hassan's proposal is a GNS: build a radius graph on the particles and
run message-passing layers per step, then pool. Swap `phi` for a GNS block where marked;
the rest of the pipeline (pool over particles -> rho -> z, contrastive + reconstruction)
is unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn


def particle_fingerprint(traj):
    """traj: (T, P, 3) positions over time -> per-particle features (P, D).
    Compact motion descriptor: start, end, mean velocity, total displacement."""
    start, end = traj[0], traj[-1]
    vel = (traj[1:] - traj[:-1]).mean(0)
    disp = (traj[-1] - traj[0])
    return torch.cat([start, end, vel, disp], dim=-1)     # (P, 12)


class TrajEncoderSim(nn.Module):
    def __init__(self, z_dim=4, hidden=128, in_dim=12):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),   # <-- swap for a GNS block
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, z_dim))

    def forward(self, traj):
        feat = particle_fingerprint(traj)                 # (P, 12)
        h = self.phi(feat).mean(dim=0)                     # pool over particles
        return self.rho(h)                                 # (z_dim,)


def contrastive_loss(za, zb, params_std, temp=0.2, sigma=1.0):
    z1, z2 = Fn.normalize(za, dim=-1), Fn.normalize(zb, dim=-1)
    logits = z1 @ z2.t() / temp
    with torch.no_grad():
        pd = torch.cdist(params_std, params_std)
        target = torch.softmax(-pd.pow(2) / (2 * sigma ** 2), dim=-1)
    lp1 = Fn.log_softmax(logits, dim=-1); lp2 = Fn.log_softmax(logits.t(), dim=-1)
    return -0.5 * ((target * lp1).sum(-1).mean() + (target * lp2).sum(-1).mean())
