"""
Experiment 1 (robust version): cross-material transfer matrix for NCLaw,
aggregated over training seeds, with a saved heatmap figure.

What this demonstrates
----------------------
NCLaw's constitutive network takes only the deformation F as input -- there is
no material-identity input anywhere in its forward signature. So a network
trained on material m encodes exactly that one material and has no way to
represent another. This script measures that as a matrix instead of a single
jelly->sand pair, and repeats every cell across several training seeds so each
number carries a mean +/- standard deviation rather than being a single run.

Why only (N x seeds) rollouts, not (N x N x seeds)
--------------------------------------------------
All four training environments share IDENTICAL initial conditions (same cube
geometry, same preset velocity, same density). So a trained network's rollout
depends ONLY on its weights, never on the target we later compare against. We
therefore roll out each (material, seed) network once, then compare that single
trajectory against all four ground-truth datasets.

The matrix is indexed [trained_on][evaluated_against]:
  * diagonal  M[m][m]  = reconstruction error  (net reproducing its own material)
  * off-diag  M[m][e]  = cross-material error   (net m asked to mimic material e)

Seeds
-----
A seed only matters at TRAINING time (it changes weight init and the training
trajectory), so "across seeds" means: one trained checkpoint per (material,
seed). Checkpoints are expected at a seed-namespaced path:

    log/<env>/train_s<seed>/invariant_full_meta-invariant_full_meta/ckpt/<epoch>.pt

Run with --train to produce any that are missing (slow: one full training run
per material per seed). The ground-truth datasets are seed-independent (preset
velocity is deterministic), so all seeds train against the same dataset.

Prerequisites (under experiments/log/)
--------------------------------------
  * ground-truth datasets:  <env>/dataset/state/*.pt     for every env in ENVS
        (produced by experiments/scripts/dataset/main.py)
  * trained checkpoints:     the seed-namespaced paths above
        (produced here with --train, or by training manually)

Usage
-----
    # train the 3 seeds x 4 materials, then build the matrix and figure:
    python experiments/scripts/eval/material_matrix.py --train -g 0

    # checkpoints already exist -> just roll out, score, and plot:
    python experiments/scripts/eval/material_matrix.py -g 0

    # iterate on the figure without re-running rollouts:
    python experiments/scripts/eval/material_matrix.py --skip-rollout -g 0

Outputs (under log/probe/matrix/)
---------------------------------
    results.json, results.csv, heatmap.png, heatmap.pdf
"""

from pathlib import Path
from argparse import ArgumentParser
import json
import subprocess

import numpy as np

from nclaw.constants import ENVS, SEEDS, RENDER
from nclaw.utils import get_root, get_script_parser, dict_to_hydra, diff_mse


# ----------------------------- configuration -----------------------------

ELASTICITY = 'invariant_full_meta'
PLASTICITY = 'invariant_full_meta'
EPOCH = 300
QUALITY = 'low'   # MUST match the sim preset used to build the datasets

# Order used for the FIGURE only (puts the sand/plasticine cluster adjacent and
# jelly/water at the extremes). Console tables stay in ENVS order.
DISPLAY_ORDER = ['jelly', 'plasticine', 'sand', 'water']


def train_name(env: str, seed: int) -> Path:
    return Path(env) / f'train_s{seed}' / f'{ELASTICITY}-{PLASTICITY}'


def ckpt_rel(env: str, seed: int) -> str:
    """Checkpoint path relative to experiments/log/."""
    return str(train_name(env, seed) / 'ckpt' / f'{EPOCH:04d}.pt')


# ------------------------------- helpers ---------------------------------

def _preflight(root: Path, will_train: bool) -> None:
    log_root = root / 'log'
    problems = []

    missing_ds = [f'  log/{e}/dataset/state/*.pt' for e in ENVS
                  if not (log_root / e / 'dataset' / 'state').is_dir()
                  or not any((log_root / e / 'dataset' / 'state').glob('*.pt'))]
    if missing_ds:
        problems.append('ground-truth datasets missing '
                        '(run experiments/scripts/dataset/main.py):\n' + '\n'.join(missing_ds))

    if not will_train:
        missing_ckpt = [f'  log/{ckpt_rel(e, s)}' for e in ENVS for s in SEEDS
                        if not (log_root / ckpt_rel(e, s)).is_file()]
        if missing_ckpt:
            problems.append('trained checkpoints missing -- re-run with --train, '
                            'or train them manually:\n' + '\n'.join(missing_ckpt))

    if problems:
        raise SystemExit('\n\n'.join(problems))


def _train_one(root: Path, base_args: dict, env: str, seed: int) -> None:
    """One full training run (mirrors the repo's train wrapper + seed + name)."""
    args = base_args | {
        'env': env,
        'env/blob/material/elasticity': ELASTICITY,
        'env/blob/material/plasticity': PLASTICITY,
        'env.blob.material.elasticity.requires_grad': True,
        'env.blob.material.plasticity.requires_grad': True,
        'render': RENDER,
        'sim': QUALITY,
        'seed': seed,
        'name': train_name(env, seed),
        'overwrite': True,
    }
    cmds = [str(root / 'train.py')] + dict_to_hydra(args)
    subprocess.run(['python', *cmds], shell=False, check=True)


def _train_missing(root: Path, base_args: dict, force: bool) -> None:
    for env in ENVS:
        for seed in SEEDS:
            if (root / 'log' / ckpt_rel(env, seed)).is_file() and not force:
                print(f'[train] skip {env} seed {seed} (checkpoint exists)')
                continue
            print(f'[train] {env} seed {seed} -> log/{train_name(env, seed)}  (slow)')
            _train_one(root, base_args, env, seed)


def _rollout(root: Path, base_args: dict, env: str, seed: int) -> Path:
    """Roll out the (env, seed) network once from the shared initial conditions."""
    name = Path('probe') / 'matrix' / f'{env}_s{seed}'
    args = base_args | {
        'env': env,
        'sim': QUALITY,
        'env/blob/material/elasticity': ELASTICITY,
        'env/blob/material/plasticity': PLASTICITY,
        'env.blob.material.ckpt': ckpt_rel(env, seed),
        'seed': seed,
        'name': name,
        'overwrite': True,
    }
    cmds = [str(root / 'eval.py')] + dict_to_hydra(args)
    subprocess.run(['python', *cmds], shell=False, check=True)
    return root / 'log' / name


def _fmt(x: float) -> str:
    return 'nan' if x != x else f'{x:.2e}'


def _print_tables(mean, std, ratio) -> None:
    multi = len(SEEDS) > 1
    w = 17 if multi else 11

    def header(title):
        print(f'\n{title}')
        print('  (rows = trained on,  cols = evaluated against ground truth)')
        print(' ' * 13 + ''.join(f'{c:>{w}}' for c in ENVS))

    header('Raw position MSE' + (' (mean +/- std over seeds)' if multi else ''))
    for m in ENVS:
        cells = []
        for e in ENVS:
            c = _fmt(mean[m][e]) + (f'+-{_fmt(std[m][e])}' if multi else '')
            cells.append(f'{c:>{w}}')
        print(f'{m:>12} ' + ''.join(cells))

    header('Error relative to own reconstruction (cell / diagonal)')
    for m in ENVS:
        cells = []
        for e in ENVS:
            r = ratio[m][e]
            txt = '1.0' if (r == r and abs(r - 1) < 1e-9) else (f'{r:,.0f}x' if r == r else 'nan')
            cells.append(f'{txt:>{w}}')
        print(f'{m:>12} ' + ''.join(cells))

    diag = [mean[m][m] for m in ENVS]
    off = [mean[m][e] for m in ENVS for e in ENVS if e != m]
    off_r = [ratio[m][e] for m in ENVS for e in ENVS if e != m and ratio[m][e] == ratio[m][e]]
    print('\nSummary')
    print(f'  seeds: {SEEDS}')
    print(f'  mean reconstruction error (diagonal):     {_fmt(float(np.mean(diag)))}')
    print(f'  mean cross-material error (off-diagonal):  {_fmt(float(np.mean(off)))}')
    if off_r:
        print(f'  cross/recon ratio -- mean {np.mean(off_r):,.0f}x   '
              f'min {min(off_r):,.0f}x   max {max(off_r):,.0f}x')
        print('  (min ratio is the weakest demonstration -- report it honestly)')


def _save_outputs(root: Path, mean, std, ratio, raw) -> None:
    out_dir = root / 'log' / 'probe' / 'matrix'
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'results.json').open('w') as f:
        json.dump({'envs': ENVS, 'seeds': SEEDS, 'mean': mean, 'std': std,
                   'ratio': ratio, 'raw': raw}, f, indent=2)
    with (out_dir / 'results.csv').open('w') as f:
        f.write('trained_on,' + ','.join(ENVS) + '\n')
        for m in ENVS:
            f.write(m + ',' + ','.join(f'{mean[m][e]:.6e}' for e in ENVS) + '\n')
    print('\nSaved: log/probe/matrix/results.json and results.csv')


def _save_heatmap(root: Path, mean, std) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Rectangle
    except ImportError:
        print('\n[viz] matplotlib not installed -- skipping figure (pip install matplotlib)')
        return

    order = [m for m in DISPLAY_ORDER if m in ENVS] + [m for m in ENVS if m not in DISPLAY_ORDER]
    M = np.array([[mean[m][e] for e in order] for m in order])
    S = np.array([[std[m][e] for e in order] for m in order])

    vmin = max(M[M > 0].min(), 1e-12)
    vmax = M.max()
    thresh = np.sqrt(vmin * vmax)  # geometric midpoint -> text color switch
    multi = len(SEEDS) > 1

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(M, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='Blues')

    ax.set_xticks(range(len(order)), labels=order)
    ax.set_yticks(range(len(order)), labels=order)
    ax.set_xlabel('evaluated against ground truth')
    ax.set_ylabel('network trained on')
    ax.set_title('Cross-material position MSE\n(diagonal, boxed = reconstruction)')

    for i in range(len(order)):
        for j in range(len(order)):
            txt = f'{M[i, j]:.1e}'
            if multi and S[i, j] > 0:
                txt += f'\n$\\pm${S[i, j]:.0e}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                    color='white' if M[i, j] > thresh else '#0C447C')
            if i == j:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor='#0C447C', lw=2.2))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('position MSE (log scale)')
    fig.tight_layout()

    out = root / 'log' / 'probe' / 'matrix' / 'heatmap.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)
    print(f'[viz] saved log/probe/matrix/heatmap.png and .pdf')


# -------------------------------- main -----------------------------------

def main() -> None:
    root = get_root(Path(__file__))
    base_args, unknown = get_script_parser().parse_known_args()
    base_args = vars(base_args)

    extra = ArgumentParser()
    extra.add_argument('--train', action='store_true',
                       help='train any missing (env, seed) checkpoints first (slow)')
    extra.add_argument('--retrain', action='store_true',
                       help='retrain every (env, seed) even if a checkpoint exists')
    extra.add_argument('--skip-rollout', dest='skip_rollout', action='store_true',
                       help='reuse existing rollouts; just recompute the matrix and figure')
    opts, _ = extra.parse_known_args(unknown)

    _preflight(root, will_train=opts.train or opts.retrain)

    if opts.train or opts.retrain:
        _train_missing(root, base_args, force=opts.retrain)

    # ---- rollouts: one per (material, seed) ----
    rollout_dirs = {(env, s): root / 'log' / 'probe' / 'matrix' / f'{env}_s{s}'
                    for env in ENVS for s in SEEDS}
    if not opts.skip_rollout:
        for env in ENVS:
            for s in SEEDS:
                print(f'[rollout] net trained on "{env}" seed {s}')
                rollout_dirs[(env, s)] = _rollout(root, base_args, env, s)

    # ---- comparisons: each rollout vs each ground-truth dataset ----
    raw = {m: {e: [] for e in ENVS} for m in ENVS}
    for m in ENVS:
        for s in SEEDS:
            rdir = rollout_dirs[(m, s)]
            for e in ENVS:
                raw[m][e].append(diff_mse(src=rdir, tar=root / 'log' / e / 'dataset')['mse'])

    mean = {m: {e: float(np.mean(raw[m][e])) for e in ENVS} for m in ENVS}
    std = {m: {e: float(np.std(raw[m][e])) for e in ENVS} for m in ENVS}
    ratio = {m: {e: (mean[m][e] / mean[m][m] if mean[m][m] else float('nan'))
                 for e in ENVS} for m in ENVS}

    _print_tables(mean, std, ratio)
    _save_outputs(root, mean, std, ratio, raw)
    _save_heatmap(root, mean, std)


if __name__ == '__main__':
    main()