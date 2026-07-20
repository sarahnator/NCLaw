import torch
from ref.materials import CondElasticity, CondPlasticity
from ref import hetero as H

torch.manual_seed(0)
types, mu, lam, tY = H.sample_hetero(8, seed=1)          # 24 train (8 x 3 forms)
tyu, muu, lamu, tYu = H.sample_hetero(6, seed=2)         # 18 unseen (6 x 3 forms)
F, tau, Fp = H.build_grouped(types, mu, lam, tY, 48, seed=10)
Fu, tauu, Fpu = H.build_grouped(tyu, muu, lamu, tYu, 48, seed=20)
scale = tau.pow(2).mean().sqrt()

enc = H.TrajEncoder(z_dim=H.Z_DIM).double()
el = CondElasticity(z_dim=H.Z_DIM).double(); pl = CondPlasticity(z_dim=H.Z_DIM).double()
H.train_encoder(enc, el, pl, F, tau, Fp, scale, types, steps=2500, K=20, lam=0.5)

z_tr = H.amortized_infer(enc, F, tau); z_te = H.amortized_infer(enc, Fu, tauu)
rec = H.per_type_recon(el, pl, z_tr, F, tau, Fp, scale, types)
rec_u = H.per_type_recon(el, pl, z_te, Fu, tauu, Fpu, scale, tyu)
acc_tr = H.cluster_accuracy(z_tr, types, z_tr, types)
acc_te = H.cluster_accuracy(z_tr, types, z_te, tyu)


def ck(n, ok, d): print(f"[{'PASS' if ok else 'FAIL'}] {n}: {d}")

print("\n=== Stage 4 (heterogeneous forms) ===")
ck("ONE conditioned pair spans all 3 forms (train recon)",
   max(rec.values()) < 0.15, " ".join(f"{k}={v:.3f}" for k, v in rec.items()))
ck("latent clusters by type (train)", acc_tr > 0.95, f"type acc {acc_tr:.2f}")
ck("amortized recon of unseen materials spans forms",
   max(rec_u.values()) < 0.35, " ".join(f"{k}={v:.3f}" for k, v in rec_u.items()))
ck("unseen type placement above chance (0.33)", acc_te > 0.5,
   f"type acc {acc_te:.2f}  <- the frontier: perfect on train, partial on unseen")
print(f"\nsummary: train recon {[round(v,2) for v in rec.values()]} | "
      f"unseen recon {[round(v,2) for v in rec_u.values()]} | unseen type-acc {acc_te:.2f}")
