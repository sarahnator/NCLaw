"""
Autograder for the Stage 4 (heterogeneous materials) homework.

  cd homework && python grade.py     (includes a short integration train, ~30s)

Edits go in hw/hetero.py (H1-H4); the encoder + training loop are given. Each
exercise -> PASS / TODO / FAIL. A final integration check trains briefly to confirm
one conditioned pair spans all three forms and the latent clusters by type.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
import hw.hetero as HH                       # noqa: E402
import solutions.hetero as SH                # noqa: E402
from solutions.materials import CondElasticity, CondPlasticity  # noqa: E402

torch.manual_seed(0)


def rep(name, state, detail=""):
    print(f"[{state}] {name}" + (f": {detail}" if detail else ""))
    return state


def grade():
    r = {}
    g = torch.Generator().manual_seed(7)
    F = torch.eye(3, dtype=torch.float64).expand(10, 3, 3) \
        + 0.15 * torch.randn(10, 3, 3, generator=g, dtype=torch.float64)
    lam = torch.full((10,), 5000.0, dtype=torch.float64)

    # H1 -- fluid law ---------------------------------------------------
    try:
        c_h, c_s = HH.fluid_cauchy(F, lam), SH.fluid_cauchy(F, lam)
        fr_h, fr_s = HH.fluid_return(F), SH.fluid_return(F)
        J = torch.linalg.det(F)
        vol_ok = torch.allclose(torch.linalg.det(fr_h), J, atol=1e-6)      # volume-preserving map
        ok = torch.allclose(c_h, c_s, atol=1e-9) and torch.allclose(fr_h, fr_s, atol=1e-9) and vol_ok
        r['H1'] = rep("H1 fluid law (mu=0 stress, J^1/3 return)", 'PASS' if ok else 'FAIL',
                      "matches ref, return is volume-preserving")
    except NotImplementedError:
        r['H1'] = rep("H1 fluid law (mu=0 stress, J^1/3 return)", 'TODO')

    # H2 -- heterogeneous dispatch --------------------------------------
    try:
        ty, mu, la, tY = SH.sample_hetero(2, seed=1)
        Fh, th, Fph = HH.build_grouped(ty, mu, la, tY, 6, seed=3)
        Fs, ts, Fps = SH.build_grouped(ty, mu, la, tY, 6, seed=3)
        ok = torch.allclose(Fh, Fs) and torch.allclose(th, ts) and torch.allclose(Fph, Fps)
        r['H2'] = rep("H2 heterogeneous dispatch (build_grouped)", 'PASS' if ok else 'FAIL',
                      f"3 forms assembled, shapes {tuple(th.shape)}")
    except NotImplementedError:
        r['H2'] = rep("H2 heterogeneous dispatch (build_grouped)", 'TODO', "(needs H1)")

    # H3 -- type contrastive --------------------------------------------
    try:
        za = torch.randn(9, 4, generator=g, dtype=torch.float64)
        zb = torch.randn(9, 4, generator=g, dtype=torch.float64)
        types = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        lh, ls = HH.type_contrastive(za, zb, types), SH.type_contrastive(za, zb, types)
        ok = torch.isfinite(lh) and abs(lh.item() - ls.item()) < 1e-9
        r['H3'] = rep("H3 type-based supervised contrastive", 'PASS' if ok else 'FAIL',
                      f"value {lh.item():.4f}")
    except NotImplementedError:
        r['H3'] = rep("H3 type-based supervised contrastive", 'TODO')

    # H4 -- cluster metric ----------------------------------------------
    try:
        # perfectly separated latents -> accuracy 1.0
        ztr = torch.tensor([[0., 0], [0, 0.1], [5, 5], [5, 5.1], [-5, 5], [-5, 5.1]])
        ttr = torch.tensor([0, 0, 1, 1, 2, 2])
        zte = torch.tensor([[0., 0.2], [5, 4.9], [-5, 5.2]]); tte = torch.tensor([0, 1, 2])
        acc = HH.cluster_accuracy(ztr, ttr, zte, tte)
        acc_ref = SH.cluster_accuracy(ztr, ttr, zte, tte)
        ok = abs(acc - acc_ref) < 1e-9 and acc > 0.99
        r['H4'] = rep("H4 nearest-centroid type accuracy", 'PASS' if ok else 'FAIL',
                      f"separated-clusters acc {acc:.2f}")
    except NotImplementedError:
        r['H4'] = rep("H4 nearest-centroid type accuracy", 'TODO')

    # integration -- short train (needs H1-H4) --------------------------
    try:
        if 'TODO' in (r.get('H1'), r.get('H2'), r.get('H3'), r.get('H4')):
            raise NotImplementedError
        types, mu, la, tY = SH.sample_hetero(6, seed=1)
        F2, tau, Fp = HH.build_grouped(types, mu, la, tY, 32, seed=10)
        scale = tau.pow(2).mean().sqrt()
        torch.manual_seed(0)
        enc = SH.TrajEncoder(z_dim=4).double()
        el = CondElasticity(z_dim=4).double(); pl = CondPlasticity(z_dim=4).double()
        SH.train_encoder(enc, el, pl, F2, tau, Fp, scale, types, steps=1000, K=16, lam=0.5)
        z = SH.amortized_infer(enc, F2, tau)
        rec = SH.per_type_recon(el, pl, z, F2, tau, Fp, scale, types)
        acc = SH.cluster_accuracy(z, types, z, types)
        ok = max(rec.values()) < 0.20 and acc > 0.9
        r['INT'] = rep("integration: one pair spans forms + clusters by type",
                       'PASS' if ok else 'FAIL',
                       f"recon {[round(v,2) for v in rec.values()]}, type acc {acc:.2f}")
    except NotImplementedError:
        r['INT'] = rep("integration: one pair spans forms + clusters by type", 'TODO', "(needs H1-H4)")

    core = ['H1', 'H2', 'H3', 'H4']
    print(f"\n{sum(r.get(k) == 'PASS' for k in core)}/4 exercises + "
          f"{'integration PASS' if r.get('INT') == 'PASS' else 'integration ' + str(r.get('INT'))}")
    return r


if __name__ == '__main__':
    grade()
