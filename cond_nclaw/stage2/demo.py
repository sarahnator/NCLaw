import torch
from ref.materials import CondElasticity, CondPlasticity
from ref import autodecoder as A

torch.manual_seed(0)
REST_CORRECT = False

E, nu, tauY, params = A.sample_family(24, seed=1)
pmean, pstd = A.param_stats(params)
F, tau, Fp = A.build_grouped(E, nu, tauY, 64, seed=10)
scale = tau.pow(2).mean().sqrt()

el = CondElasticity(z_dim=A.Z_DIM, rest_correct=REST_CORRECT).double()
pl = CondPlasticity(z_dim=A.Z_DIM, rest_correct=REST_CORRECT).double()
book = A.LatentBook(24, z_dim=A.Z_DIM).double()
A.train_autodecoder(el, pl, book, F, tau, Fp, scale, steps=3000)

e_el = A.rel_err_elastic(el, book, F, tau, scale)
e_pl = A.rel_err_plastic(pl, book, F, Fp)
struct = A.structure_score(book.z.detach(), params, pmean, pstd)

# invert an unseen material
Eu, nuu, tyu, pu = A.sample_family(1, seed=99)
Fu, tu, Fpu = A.build_grouped(Eu, nuu, tyu, 64, seed=77)
z_inv = A.invert_latent(el, pl, Fu, tu, Fpu, scale)
zb = z_inv.unsqueeze(0).expand(Fu.reshape(-1, 3, 3).shape[0], -1)
with torch.no_grad():
    inv_fit = ((el(Fu.reshape(-1, 3, 3), zb) - tu.reshape(-1, 3, 3) / scale).pow(2).mean().sqrt()
               / (tu.reshape(-1, 3, 3) / scale).pow(2).mean().sqrt()).item()

I = torch.eye(3, dtype=torch.float64).expand(5, 3, 3)
z0 = torch.zeros(5, A.Z_DIM, dtype=torch.float64); zr = torch.randn(5, A.Z_DIM, dtype=torch.float64)
with torch.no_grad():
    e0 = el(I, z0).abs().max().item(); p0 = (pl(I, z0) - I).abs().max().item()
    ez = el(I, zr).abs().max().item()


def ck(n, ok, d): print(f"[{'PASS' if ok else 'FAIL'}] {n}: {d}")

print(f"\n=== Stage 2 two-network reference (rest_correct={REST_CORRECT}) ===")
ck("elastic reconstructs family", e_el < 0.10, f"rel.err {e_el:.3f}")
ck("plastic reconstructs family", e_pl < 0.25, f"correction rel.err {e_pl:.3f}")
ck("elastic equilibrium at z=0", e0 < 1e-6, f"|stress(I,z=0)| {e0:.1e}")
ck("plastic equilibrium at z=0", p0 < 1e-6, f"|Fp(I,z=0)-I| {p0:.1e}")
ck("invert unseen material", inv_fit < 0.15, f"rel.err {inv_fit:.3f}")
print(f"[INFO] elastic |stress(I,z!=0)| = {ez:.2e} (plain concat breaks away from z=0)")
print(f"[INFO] latent structure vs (nu,tauY): spearman = {struct:.2f} (emergent)")
