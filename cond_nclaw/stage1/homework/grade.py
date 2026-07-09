"""
Autograder for the conditioned-NCLaw homework (Stage 1, 3D).

  cd homework
  python grade.py

Each exercise reports PASS (matches the reference and satisfies its structural
invariant), TODO (still a NotImplementedError stub), or FAIL (attempted but
wrong -- the message says what mismatched). Your edits go in hw/; solutions/ is
the reference the grader checks against.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
import hw.materials as HWM, hw.pipeline as HWP          # noqa: E402
import solutions.materials as SOLM, solutions.pipeline as SOLP  # noqa: E402

torch.manual_seed(0)


def report(name, state, detail=""):
    print(f"[{state}] {name}" + (f": {detail}" if detail else ""))
    return state


def grade():
    res = {}
    g = torch.Generator().manual_seed(7)
    inv = torch.randn(16, 13, generator=g, dtype=torch.float64)
    z = torch.randn(16, 2, generator=g, dtype=torch.float64)
    F = torch.eye(3, dtype=torch.float64).expand(16, 3, 3) \
        + 0.07 * torch.randn(16, 3, 3, generator=g, dtype=torch.float64)

    torch.manual_seed(3)
    sol = SOLM.CondInvariantFullElasticity(z_dim=2).double()
    hw = HWM.CondInvariantFullElasticity(z_dim=2).double()
    hw.load_state_dict(sol.state_dict())                # identical weights

    # C1 -----------------------------------------------------------------
    try:
        oh, os_ = hw._frame_stress(inv, z), sol._frame_stress(inv, z)
        sym = (oh - oh.transpose(-1, -2)).abs().max().item()
        ok = torch.allclose(oh, os_, atol=1e-9) and sym < 1e-9
        res['C1'] = report("C1 conditioning injection (_frame_stress)",
                           'PASS' if ok else 'FAIL',
                           "matches reference, symmetric" if ok else f"sym residual {sym:.1e}")
    except NotImplementedError:
        res['C1'] = report("C1 conditioning injection (_frame_stress)", 'TODO')

    # C2 -----------------------------------------------------------------
    try:
        fh, fs = hw(F, z), sol(F, z)
        I8 = torch.eye(3, dtype=torch.float64).expand(8, 3, 3)
        z8 = torch.randn(8, 2, dtype=torch.float64)
        rest = hw(I8, z8).abs().max().item()
        th = torch.tensor(0.5, dtype=torch.float64); c, s = torch.cos(th), torch.sin(th)
        Rs = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)
        equiv = (hw(Rs @ F, z) - Rs @ hw(F, z) @ Rs.T).abs().max().item()
        ok = torch.allclose(fh, fs, atol=1e-9) and rest < 1e-6 and equiv < 1e-8
        res['C2'] = report("C2 rest-state subtraction + assemble (forward)",
                           'PASS' if ok else 'FAIL', f"rest {rest:.1e}, equiv {equiv:.1e}")
    except NotImplementedError:
        res['C2'] = report("C2 rest-state subtraction + assemble (forward)", 'TODO')

    # C3 -----------------------------------------------------------------
    try:
        E, nu = SOLP.sample_materials(6, seed=1)
        zm, zs = SOLP.z_stats(E, nu)
        Fh, zh, th_ = HWP.build_dataset(E, nu, 8, zm, zs, seed=5)
        Fs, zs_, ts = SOLP.build_dataset(E, nu, 8, zm, zs, seed=5)
        ok = (torch.allclose(Fh, Fs) and torch.allclose(zh, zs_) and torch.allclose(th_, ts))
        res['C3'] = report("C3 material continuum + oracle z (build_dataset)",
                           'PASS' if ok else 'FAIL', f"shapes {tuple(Fh.shape)}, {tuple(zh.shape)}")
    except NotImplementedError:
        res['C3'] = report("C3 material continuum + oracle z (build_dataset)", 'TODO')

    # C4 -----------------------------------------------------------------
    try:
        if res.get('C1') == 'TODO' or res.get('C2') == 'TODO':
            raise NotImplementedError
        E_tr, nu_tr = SOLP.sample_materials(24, seed=1)
        E_te, nu_te = SOLP.sample_materials(10, seed=2)
        zm, zs = SOLP.z_stats(E_tr, nu_tr)
        Ftr, ztr, ttr = SOLP.build_dataset(E_tr, nu_tr, 32, zm, zs, seed=10)
        Fte, zte, tte = SOLP.build_dataset(E_te, nu_te, 32, zm, zs, seed=20)
        sc = ttr.pow(2).mean().sqrt()
        torch.manual_seed(0)
        m = HWM.CondInvariantFullElasticity(z_dim=2).double()
        HWP.train_conditioned(m, Ftr, ztr, ttr, sc, steps=1200, lr=3e-3)
        e = SOLP.rel_err(m, Fte, zte, tte, sc)
        res['C4'] = report("C4 conditioned training (generalizes to unseen)",
                           'PASS' if e < 0.15 else 'FAIL', f"unseen rel.err {e:.3f} (< 0.15)")
    except NotImplementedError:
        res['C4'] = report("C4 conditioned training (generalizes to unseen)", 'TODO',
                           "(finish C1, C2, C4)")

    print(f"\n{sum(v == 'PASS' for v in res.values())}/4 passing")
    return res


if __name__ == '__main__':
    grade()
