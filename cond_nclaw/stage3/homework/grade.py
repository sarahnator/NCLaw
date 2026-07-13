"""
Autograder for the Stage 3 encoder homework.

  cd homework && python grade.py      (E3 trains a small model -> ~30s)

Your edits go in hw/encoder.py (E1-E4); solutions/ is the reference. Each exercise
-> PASS / TODO (stub) / FAIL (attempted, wrong).
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
import hw.encoder as HX                      # noqa: E402
import solutions.encoder as SX               # noqa: E402
from solutions.materials import CondElasticity, CondPlasticity  # noqa: E402

torch.manual_seed(0)


def rep(name, state, detail=""):
    print(f"[{state}] {name}" + (f": {detail}" if detail else ""))
    return state


def grade():
    r = {}
    g = torch.Generator().manual_seed(7)
    Fb = torch.eye(3, dtype=torch.float64).expand(6, 20, 3, 3) \
        + 0.15 * torch.randn(6, 20, 3, 3, generator=g, dtype=torch.float64)
    tb = torch.randn(6, 20, 3, 3, generator=g, dtype=torch.float64)

    torch.manual_seed(3)
    s_enc = SX.TrajEncoder(z_dim=4).double()
    h_enc = HX.TrajEncoder(z_dim=4).double(); h_enc.load_state_dict(s_enc.state_dict())

    # E1 -- encoder ----------------------------------------------------
    try:
        zh, zs = h_enc(Fb, tb), s_enc(Fb, tb)
        perm = torch.randperm(20)
        zp = h_enc(Fb[:, perm], tb[:, perm])
        ok = tuple(zh.shape) == (6, 4) and torch.allclose(zh, zs, atol=1e-9) \
            and (zh - zp).abs().max().item() < 1e-9
        r['E1'] = rep("E1 DeepSets encoder (permutation-invariant)", 'PASS' if ok else 'FAIL',
                      f"shape {tuple(zh.shape)}, perm-gap ok")
    except NotImplementedError:
        r['E1'] = rep("E1 DeepSets encoder (permutation-invariant)", 'TODO')

    # E2 -- contrastive loss -------------------------------------------
    try:
        za = torch.randn(6, 4, generator=g, dtype=torch.float64)
        zb = torch.randn(6, 4, generator=g, dtype=torch.float64)
        ps = torch.randn(6, 2, generator=g, dtype=torch.float64)
        lh, ls = HX.contrastive_loss(za, zb, ps), SX.contrastive_loss(za, zb, ps)
        ok = torch.isfinite(lh) and abs(lh.item() - ls.item()) < 1e-9
        r['E2'] = rep("E2 CLIP-style contrastive loss", 'PASS' if ok else 'FAIL',
                      f"value {lh.item():.4f} (ref {ls.item():.4f})")
    except NotImplementedError:
        r['E2'] = rep("E2 CLIP-style contrastive loss", 'TODO')

    # E4 -- amortized inference (test before E3, it only needs E1) ------
    try:
        zi = HX.amortized_infer(h_enc, Fb, tb)
        ok = torch.allclose(zi, s_enc(Fb, tb), atol=1e-9) and not zi.requires_grad
        r['E4'] = rep("E4 amortized inference (one pass, no grad)", 'PASS' if ok else 'FAIL',
                      "matches encoder, detached")
    except NotImplementedError:
        r['E4'] = rep("E4 amortized inference (one pass, no grad)", 'TODO', "(needs E1)")

    # E3 -- joint training (behavioral) --------------------------------
    try:
        if 'TODO' in (r.get('E1'), r.get('E2')):
            raise NotImplementedError
        E, nu, tY, params = SX.sample_family(24, seed=1)
        pmean, pstd = SX.param_stats(params)
        F, tau, Fp = SX.build_grouped(E, nu, tY, 32, seed=10)
        scale = tau.pow(2).mean().sqrt(); pz = (params - pmean) / pstd
        torch.manual_seed(0)
        enc = HX.TrajEncoder(z_dim=4).double()
        el = CondElasticity(z_dim=4).double(); pl = CondPlasticity(z_dim=4).double()
        HX.train_encoder(enc, el, pl, F, tau, Fp, scale, pz, steps=1000, K=16, lam=0.3)
        z_tr = SX.amortized_infer(enc, F, tau)
        e_tr, p_tr = SX.recon_with_z(el, pl, z_tr, F, tau, Fp, scale)
        struct = SX.structure_score(z_tr, params, pmean, pstd)
        ok = e_tr < 0.15 and p_tr < 0.25
        r['E3'] = rep("E3 joint training (recon + contrastive)", 'PASS' if ok else 'FAIL',
                      f"train {e_tr:.3f}/{p_tr:.3f}, structure {struct:.2f}")
    except NotImplementedError:
        r['E3'] = rep("E3 joint training (recon + contrastive)", 'TODO', "(needs E1, E2)")

    order = ['E1', 'E2', 'E3', 'E4']
    print(f"\n{sum(r.get(k) == 'PASS' for k in order)}/4 passing")
    return r


if __name__ == '__main__':
    grade()
