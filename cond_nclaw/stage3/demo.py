import torch
from ref.materials import CondElasticity, CondPlasticity
from ref import encoder as X

torch.manual_seed(0)
E, nu, tY, params = X.sample_family(48, seed=1)
Eu, nuu, tYu, pu = X.sample_family(8, seed=2)
pmean, pstd = X.param_stats(params)
F, tau, Fp = X.build_grouped(E, nu, tY, 48, seed=10)
Fu, tauu, Fpu = X.build_grouped(Eu, nuu, tYu, 48, seed=20)
scale = tau.pow(2).mean().sqrt(); pz = (params - pmean) / pstd

enc = X.TrajEncoder(z_dim=X.Z_DIM).double()
el = CondElasticity(z_dim=X.Z_DIM).double(); pl = CondPlasticity(z_dim=X.Z_DIM).double()
X.train_encoder(enc, el, pl, F, tau, Fp, scale, pz, steps=2500, K=20, lam=0.3)

z_tr = X.amortized_infer(enc, F, tau)
z_te = X.amortized_infer(enc, Fu, tauu)                 # UNSEEN, one forward pass
e_tr, p_tr = X.recon_with_z(el, pl, z_tr, F, tau, Fp, scale)
e_te, p_te = X.recon_with_z(el, pl, z_te, Fu, tauu, Fpu, scale)
struct = X.structure_score(z_tr, params, pmean, pstd)
perm = torch.randperm(48)
with torch.no_grad():
    perm_gap = (enc(F, tau) - enc(F[:, perm], tauu[:, perm] if False else tau[:, perm])).abs().max().item()


def ck(n, ok, d): print(f"[{'PASS' if ok else 'FAIL'}] {n}: {d}")

print("\n=== Stage 3 (feed-forward encoder) ===")
ck("train recon via encoded z (elastic)", e_tr < 0.15, f"{e_tr:.3f}")
ck("train recon via encoded z (plastic)", p_tr < 0.20, f"{p_tr:.3f}")
ck("UNSEEN one-pass recon (elastic)", e_te < 0.30, f"{e_te:.3f}  <- amortized, no optimization")
ck("UNSEEN one-pass recon (plastic)", p_te < 0.35, f"{p_te:.3f}")
ck("encoder permutation-invariant", perm_gap < 1e-9, f"gap {perm_gap:.1e}")
ck("structure beats Stage-2 baseline (~0.6)", struct > 0.75, f"spearman {struct:.2f}")
print(f"\nsummary: train {e_tr:.3f}/{p_tr:.3f} | unseen {e_te:.3f}/{p_te:.3f} | struct {struct:.2f}")
