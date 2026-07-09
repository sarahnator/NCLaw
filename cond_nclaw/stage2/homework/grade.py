"""
Autograder for the Stage 2 two-network homework.

  cd homework && python grade.py

Your edits go in hw/ (T1-T5); solutions/ is the reference the grader checks against.
Each exercise -> PASS / TODO (still a stub) / FAIL (attempted, wrong -- message says why).
Equilibrium is checked AT z=0 (plain concat), as in UniPhy.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
import hw.materials as HWM, hw.autodecoder as HWA           # noqa: E402
import solutions.materials as SM, solutions.autodecoder as SA  # noqa: E402

torch.manual_seed(0)


def rep(name, state, detail=""):
    print(f"[{state}] {name}" + (f": {detail}" if detail else ""))
    return state


def grade():
    r = {}
    g = torch.Generator().manual_seed(7)
    inv = torch.randn(12, 13, generator=g, dtype=torch.float64)
    z = torch.randn(12, 4, generator=g, dtype=torch.float64)
    F = torch.eye(3, dtype=torch.float64).expand(12, 3, 3) \
        + 0.12 * torch.randn(12, 3, 3, generator=g, dtype=torch.float64)

    torch.manual_seed(3)
    s_el = SM.CondElasticity(z_dim=4, rest_correct=False).double()
    s_pl = SM.CondPlasticity(z_dim=4, rest_correct=False).double()
    h_el = HWM.CondElasticity(z_dim=4, rest_correct=False).double(); h_el.load_state_dict(s_el.state_dict())
    h_pl = HWM.CondPlasticity(z_dim=4, rest_correct=False).double(); h_pl.load_state_dict(s_pl.state_dict())

    # T1 -- _frame -----------------------------------------------------
    try:
        oh, os_ = h_el._frame(inv, z), s_el._frame(inv, z)
        sym = (oh - oh.transpose(-1, -2)).abs().max().item()
        rest0 = h_el._frame(torch.zeros_like(inv), torch.zeros_like(z)).abs().max().item()
        ok = torch.allclose(oh, os_, atol=1e-9) and sym < 1e-9 and rest0 < 1e-9
        r['T1'] = rep("T1 conditioned core (_frame)", 'PASS' if ok else 'FAIL',
                      f"symmetric, matches ref, frame(0,0)={rest0:.1e}")
    except NotImplementedError:
        r['T1'] = rep("T1 conditioned core (_frame)", 'TODO')

    # T2 -- two heads + equilibrium at z=0 -----------------------------
    try:
        eh, ph = h_el(F, z), h_pl(F, z)
        okmatch = torch.allclose(eh, s_el(F, z), atol=1e-9) and torch.allclose(ph, s_pl(F, z), atol=1e-9)
        I = torch.eye(3, dtype=torch.float64).expand(6, 3, 3)
        z0 = torch.zeros(6, 4, dtype=torch.float64)
        e0 = h_el(I, z0).abs().max().item()
        p0 = (h_pl(I, z0) - I).abs().max().item()
        ok = okmatch and e0 < 1e-6 and p0 < 1e-6
        r['T2'] = rep("T2 elastic + plastic heads (equilibrium at z=0)",
                      'PASS' if ok else 'FAIL', f"stress(I,0)={e0:.1e}, Fp(I,0)-I={p0:.1e}")
    except NotImplementedError:
        r['T2'] = rep("T2 elastic + plastic heads (equilibrium at z=0)", 'TODO', "(needs T1)")

    # T3 -- LatentBook -------------------------------------------------
    try:
        book = HWA.LatentBook(8, z_dim=4).double()
        ok = tuple(book.z.shape) == (8, 4) and book.z.requires_grad \
            and 0.5 < book.z.detach().std().item() < 1.6 \
            and torch.allclose(book(torch.tensor([1, 4])), book.z[[1, 4]])
        r['T3'] = rep("T3 shared latent codebook (LatentBook)", 'PASS' if ok else 'FAIL',
                      f"shape {tuple(book.z.shape)}, std {book.z.detach().std():.2f}")
    except NotImplementedError:
        r['T3'] = rep("T3 shared latent codebook (LatentBook)", 'TODO')

    # data for behavioral checks
    E, nu, tauY, params = SA.sample_family(20, seed=1)
    Fd, taud, Fpd = SA.build_grouped(E, nu, tauY, 32, seed=10)
    scale = taud.pow(2).mean().sqrt()

    # T4 -- joint training --------------------------------------------
    el = SM.CondElasticity(z_dim=4, rest_correct=False).double()
    pl = SM.CondPlasticity(z_dim=4, rest_correct=False).double()
    book = None
    try:
        if 'TODO' in (r.get('T1'), r.get('T2'), r.get('T3')):
            raise NotImplementedError
        torch.manual_seed(0)
        book = HWA.LatentBook(20, z_dim=4).double()
        HWA.train_autodecoder(el, pl, book, Fd, taud, Fpd, scale, steps=1500)
        e_el = SA.rel_err_elastic(el, book, Fd, taud, scale)
        e_pl = SA.rel_err_plastic(pl, book, Fd, Fpd)
        ok = e_el < 0.12 and e_pl < 0.30
        r['T4'] = rep("T4 joint autodecoder training", 'PASS' if ok else 'FAIL',
                      f"elastic {e_el:.3f}, plastic {e_pl:.3f}")
    except NotImplementedError:
        r['T4'] = rep("T4 joint autodecoder training", 'TODO', "(needs T1-T3)")

    # T5 -- inversion --------------------------------------------------
    try:
        if r.get('T4') != 'PASS':
            raise NotImplementedError
        Eu, nuu, tyu, _ = SA.sample_family(1, seed=99)
        Fu, tu, Fpu = SA.build_grouped(Eu, nuu, tyu, 32, seed=77)
        zi = HWA.invert_latent(el, pl, Fu, tu, Fpu, scale, z_dim=4, steps=1000)
        zb = zi.unsqueeze(0).expand(Fu.reshape(-1, 3, 3).shape[0], -1)
        with torch.no_grad():
            fit = ((el(Fu.reshape(-1, 3, 3), zb) - tu.reshape(-1, 3, 3) / scale).pow(2).mean().sqrt()
                   / (tu.reshape(-1, 3, 3) / scale).pow(2).mean().sqrt()).item()
        r['T5'] = rep("T5 latent inversion (fits unseen material)",
                      'PASS' if fit < 0.15 else 'FAIL', f"fit {fit:.3f}")
    except NotImplementedError:
        r['T5'] = rep("T5 latent inversion (fits unseen material)", 'TODO', "(needs T1-T4)")

    print(f"\n{sum(v == 'PASS' for v in r.values())}/5 passing")
    return r


if __name__ == '__main__':
    grade()
