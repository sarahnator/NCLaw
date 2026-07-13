"""
Stage 3 through the sim: an encoder predicts each material's latent from its motion
trajectory (one forward pass), and encoder + both conditioned nets are trained by
BPTT through the differentiable MPM on MOTION only, plus a contrastive term.

Mirrors experiments/train.py; conditioning/encoder additions marked  # <-- S3.
Runs in YOUR Warp env (unverified here) -- check the first epoch's loss is finite.

    uv run python experiments/train_encoder_sim.py \
        env=jelly sim=low \
        env/blob/material/elasticity=cond_invariant_full \
        env/blob/material/plasticity=cond_invariant_full \
        +z_dim=4 +family_spec=$PWD/family.json +lam=0.3 \
        name=cond/stage3_sim
"""
from pathlib import Path
import json
from tqdm.autonotebook import tqdm, trange

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import warp as wp

import nclaw
from nclaw.train import Teacher
from nclaw.data import MPMDataset
from nclaw.sim import MPMModelBuilder, MPMCacheDiffSim, MPMStaticsInitializer, MPMInitData
from nclaw.utils import get_root
from nclaw.material.traj_encoder import TrajEncoderSim, contrastive_loss   # <-- S3

root: Path = get_root(__file__)


def load_traj(ds, num_steps):
    """Stack a dataset's ground-truth positions into (T, P, 3) for the encoder."""
    return torch.stack([ds[s][0] for s in range(num_steps + 1)], dim=0)


@hydra.main(version_base='1.2', config_path=str(root / 'configs'), config_name='train')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    wp.init(); wp_device = wp.get_device(f'cuda:{cfg.gpu}')
    wp.set_module_options({'fast_math': False})
    torch_device = torch.device(f'cuda:{cfg.gpu}')

    log_root = root / 'log'; exp_root = log_root / cfg.name
    nclaw.utils.mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume)
    writer = SummaryWriter(exp_root, purge_step=0)
    ckpt_root = exp_root / 'ckpt'; ckpt_root.mkdir(parents=True, exist_ok=True)

    # family + oracle params (contrastive target)  # <-- S3
    family = json.loads(Path(cfg.family_spec).read_text())
    names = [m['name'] for m in family]
    params = torch.tensor([m['params'] for m in family], dtype=torch.float32, device=torch_device)
    params_std = (params - params.mean(0)) / params.std(0)
    datasets = [MPMDataset(log_root / n / 'dataset', torch_device) for n in names]
    trajs = [load_traj(d, cfg.sim.num_steps) for d in datasets]      # encoder inputs
    n_mat = len(family)

    model = MPMModelBuilder().parse_cfg(cfg.sim).finalize(wp_device, requires_grad=True)
    sim = MPMCacheDiffSim(model, cfg.sim.num_steps)
    si = MPMStaticsInitializer(model); si.add_group(MPMInitData.get(cfg.env.blob))
    statics = si.finalize()

    elasticity = getattr(nclaw.material, cfg.env.blob.material.elasticity.cls)(cfg.env.blob.material.elasticity).to(torch_device)
    plasticity = getattr(nclaw.material, cfg.env.blob.material.plasticity.cls)(cfg.env.blob.material.plasticity).to(torch_device)
    encoder = TrajEncoderSim(z_dim=int(cfg.z_dim)).to(torch_device)   # <-- S3

    opt = torch.optim.Adam(list(encoder.parameters()) + list(elasticity.parameters())
                           + list(plasticity.parameters()), lr=cfg.train.elasticity_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.num_epochs)
    teacher = Teacher(cfg); criterion = nn.MSELoss().to(torch_device)
    lam = float(cfg.get('lam', 0.3))

    for epoch in trange(cfg.train.num_epochs):
        opt.zero_grad()
        loss_factor = 1e4 * teacher.loss_factor(cfg.sim.num_steps)

        zs = torch.stack([encoder(trajs[m]) for m in range(n_mat)], dim=0)   # <-- S3 one pass each
        recon = 0.0
        for m in range(n_mat):
            elasticity.set_latent(zs[m]); plasticity.set_latent(zs[m])
            xt, vt, Ct, Ft, _ = datasets[m][0]
            x, v, C, F = xt, vt, Ct, Ft
            for step, is_teacher in enumerate(teacher(cfg.sim.num_steps)):
                if is_teacher:
                    x, v, C, F = xt, vt, Ct, Ft
                stress = elasticity(F)
                x, v, C, F = sim(statics, step, x, v, C, F, stress)
                F = plasticity(F)
                xt, vt, Ct, Ft, _ = datasets[m][step + 1]
                recon = recon + criterion(x, xt) * loss_factor

        con = contrastive_loss(zs, zs.detach() * 0 + zs, params_std)         # single view here
        loss = recon + lam * con
        loss.backward()
        clip_grad_norm_(elasticity.parameters(), cfg.train.elasticity_grad_max_norm, error_if_nonfinite=True)
        clip_grad_norm_(plasticity.parameters(), cfg.train.plasticity_grad_max_norm, error_if_nonfinite=True)
        opt.step(); sched.step()

        tqdm.write(f"[{cfg.name}] epoch {epoch+1} recon {recon.item():.4f} con {con.item():.4f}")
        writer.add_scalar('loss/recon', recon.item(), epoch + 1)
        writer.add_scalar('loss/contrastive', con.item(), epoch + 1)
        torch.save({'encoder': encoder.state_dict(), 'elasticity': elasticity.state_dict(),
                    'plasticity': plasticity.state_dict()}, ckpt_root / f'{epoch+1:04d}.pt')
    writer.close()


if __name__ == '__main__':
    main()
