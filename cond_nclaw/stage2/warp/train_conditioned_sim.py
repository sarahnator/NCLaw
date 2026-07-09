"""
Through-the-sim Stage 2: condition NCLaw on a learned latent and train ONE
elastic+plastic pair across a family of materials, supervised on MOTION only
(particle positions), by back-propagating through the differentiable MPM.

This mirrors experiments/train.py; the conditioning additions are marked  # <-- COND.
Runs in YOUR Warp environment (I could not execute it here) -- sanity-check the
first epoch's loss before launching a long run.

Setup
-----
1. Put warp/conditioned_materials.py into nclaw/material/ and make the two classes
   importable as nclaw.material.CondInvariantFull{Elasticity,Plasticity}.
2. Generate one dataset per family material (see README_warp.md), all sharing the
   same geometry + initial conditions, at log/<name>/dataset.
3. Write family.json:  [{"name": "fam_nu10_ty80", "params": [0.10, 80.0]}, ...]
   ("params" is only used for the structure-score readout / oracle comparison.)
4. Run like train.py (hydra), pointing the material configs at the Cond* classes
   with an added z_dim, and set +family_spec=/abs/path/family.json  +z_dim=4

    uv run python experiments/train_conditioned_sim.py \
        env=jelly sim=low \
        env/blob/material/elasticity=cond_invariant_full \
        env/blob/material/plasticity=cond_invariant_full \
        +z_dim=4 +family_spec=$PWD/family.json \
        name=cond/stage2_sim
"""
from pathlib import Path
import json
import random
import time
from collections import defaultdict
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
from nclaw.material.conditioned_materials import LatentBook   # <-- COND

root: Path = get_root(__file__)


def _spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / ra.std(); rb = (rb - rb.mean()) / rb.std()
    return (ra * rb).mean().item()


def structure_score(latents, params):
    n = latents.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    p = (params - params.mean(0)) / params.std(0)
    ld = torch.cdist(latents, latents)[iu[0], iu[1]]
    pd = torch.cdist(p, p)[iu[0], iu[1]]
    return _spearman(ld.detach().cpu(), pd.detach().cpu())


@hydra.main(version_base='1.2', config_path=str(root / 'configs'), config_name='train')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    seed = cfg.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    wp.init(); wp_device = wp.get_device(f'cuda:{cfg.gpu}')
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})
    torch_device = torch.device(f'cuda:{cfg.gpu}')

    log_root: Path = root / 'log'
    exp_root: Path = log_root / cfg.name
    nclaw.utils.mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume)
    writer = SummaryWriter(exp_root, purge_step=0)
    ckpt_root = exp_root / 'ckpt'; ckpt_root.mkdir(parents=True, exist_ok=True)

    # ---- family: datasets + oracle params (structure readout only) ----  # <-- COND
    family = json.loads(Path(cfg.family_spec).read_text())
    names = [m['name'] for m in family]
    params = torch.tensor([m['params'] for m in family], dtype=torch.float32, device=torch_device)
    datasets = [MPMDataset(log_root / n / 'dataset', torch_device) for n in names]
    n_mat = len(family)

    # ---- sim (shared geometry/IC across the family) ----
    model = MPMModelBuilder().parse_cfg(cfg.sim).finalize(wp_device, requires_grad=True)
    sim = MPMCacheDiffSim(model, cfg.sim.num_steps)
    statics_initializer = MPMStaticsInitializer(model)
    statics_initializer.add_group(MPMInitData.get(cfg.env.blob))
    statics = statics_initializer.finalize()

    # ---- conditioned materials + latent codebook ----  # <-- COND
    elasticity = getattr(nclaw.material, cfg.env.blob.material.elasticity.cls)(cfg.env.blob.material.elasticity)
    plasticity = getattr(nclaw.material, cfg.env.blob.material.plasticity.cls)(cfg.env.blob.material.plasticity)
    elasticity.to(torch_device).train(True)
    plasticity.to(torch_device).train(True)
    book = LatentBook(n_mat, int(cfg.z_dim)).to(torch_device)

    opt = torch.optim.Adam([                                   # <-- COND dual LR
        {'params': list(elasticity.parameters()) + list(plasticity.parameters()),
         'lr': cfg.train.elasticity_lr},
        {'params': book.parameters(), 'lr': cfg.train.elasticity_lr * 10.0},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.num_epochs)
    teacher = Teacher(cfg)
    criterion = nn.MSELoss().to(torch_device)

    for epoch in trange(cfg.train.num_epochs, position=1):
        opt.zero_grad()
        num_teachers = teacher.num_teachers(cfg.sim.num_steps)
        loss_factor = 1e4 * teacher.loss_factor(cfg.sim.num_steps)

        losses = defaultdict(float)
        for m in range(n_mat):                                # <-- COND loop over family
            z = book(m)                                        # <-- COND per-material latent
            elasticity.set_latent(z); plasticity.set_latent(z)

            xt, vt, Ct, Ft, _ = datasets[m][0]
            x, v, C, F = xt, vt, Ct, Ft
            for step, is_teacher in enumerate(teacher(cfg.sim.num_steps)):
                if is_teacher:
                    x, v, C, F = xt, vt, Ct, Ft
                stress = elasticity(F)                         # conditioned on z
                x, v, C, F = sim(statics, step, x, v, C, F, stress)
                F = plasticity(F)                              # conditioned on z
                xt, vt, Ct, Ft, _ = datasets[m][step + 1]
                losses['acc'] = losses['acc'] + criterion(x, xt) * loss_factor

        loss = losses['acc']
        loss.backward()
        e_gn = clip_grad_norm_(elasticity.parameters(), cfg.train.elasticity_grad_max_norm, error_if_nonfinite=True)
        p_gn = clip_grad_norm_(plasticity.parameters(), cfg.train.plasticity_grad_max_norm, error_if_nonfinite=True)
        opt.step(); sched.step()

        struct = structure_score(book.z.detach(), params)      # <-- COND readout
        tqdm.write(f"[{cfg.name}] epoch {epoch+1}/{cfg.train.num_epochs} "
                   f"loss {loss.item():.4f} |e-grad| {e_gn:.3f} |p-grad| {p_gn:.3f} "
                   f"struct {struct:.2f}")
        writer.add_scalar('loss/acc', loss.item(), epoch + 1)
        writer.add_scalar('latent/structure', struct, epoch + 1)

        torch.save({'elasticity': elasticity.state_dict(),
                    'plasticity': plasticity.state_dict(),
                    'book': book.state_dict()}, ckpt_root / f'{epoch+1:04d}.pt')

    writer.close()


if __name__ == '__main__':
    main()
