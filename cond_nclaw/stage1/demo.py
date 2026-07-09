"""Run the reference end-to-end and print the Stage-1 diagnostics."""
import torch
from cond_ref.materials import CondInvariantFullElasticity
from cond_ref.pipeline import (sample_materials, z_stats, build_dataset,
                               train_conditioned, rel_err)

torch.manual_seed(0)

E_tr, nu_tr = sample_materials(40, seed=1)
E_te, nu_te = sample_materials(15, seed=2)
zmean, zstd = z_stats(E_tr, nu_tr)
F_tr, z_tr, t_tr = build_dataset(E_tr, nu_tr, 48, zmean, zstd, seed=10)
F_te, z_te, t_te = build_dataset(E_te, nu_te, 48, zmean, zstd, seed=20)
scale = t_tr.pow(2).mean().sqrt()

model = CondInvariantFullElasticity(z_dim=2).double()
train_conditioned(model, F_tr, z_tr, t_tr, scale, steps=2500)


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

e_tr = rel_err(model, F_tr, z_tr, t_tr, scale)
e_te = rel_err(model, F_te, z_te, t_te, scale)
z_shuf = z_te[torch.randperm(z_te.shape[0])]
e_shuf = rel_err(model, F_te, z_shuf, t_te, scale)
e_mean = rel_err(model, F_te, torch.zeros_like(z_te), t_te, scale)

I = torch.eye(3, dtype=torch.float64).expand(z_te.shape[0], 3, 3)
with torch.no_grad():
    rest = model(I, z_te).abs().max().item()

th = torch.tensor(0.6, dtype=torch.float64)
c, s = torch.cos(th), torch.sin(th)
Rstar = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)
with torch.no_grad():
    lhs = model(Rstar @ F_te, z_te)
    rhs = Rstar @ model(F_te, z_te) @ Rstar.T
    equiv = (lhs - rhs).abs().max().item()

print("\n=== Stage 1 (3D) diagnostics ===")
check("fit on training materials", e_tr < 0.10, f"rel.err {e_tr:.3f} (< 0.10)")
check("generalize to UNSEEN materials", e_te < 0.15, f"rel.err {e_te:.3f} (< 0.15)")
check("uses z (shuffled-z ablation)", e_shuf > 3 * e_te, f"wrong-z {e_shuf:.3f} vs {e_te:.3f}")
check("z carries info (mean-z ablation)", e_mean > 3 * e_te, f"mean-z {e_mean:.3f} vs {e_te:.3f}")
check("rest-state equilibrium at F=I", rest < 1e-6, f"max|stress| {rest:.1e}")
check("rotation equivariance", equiv < 1e-8, f"max|tau(R*F)-R* tau R*^T| {equiv:.1e}")
print(f"\nsummary: train {e_tr:.3f} | unseen {e_te:.3f} | wrong-z {e_shuf:.3f} | mean-z {e_mean:.3f}")
