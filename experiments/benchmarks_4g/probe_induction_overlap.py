"""SPEC 26 on frozen weights: size of the one-body E1 and pair Xi corrections, by carrier charge class (2026-10-18).

Usage: python probe_induction_overlap.py MODEL.pt [XYZ]
Prints, for the first Al-doped structure, E1 per (element, state) and |Xi| by the larger |Q| of the pair, with the
Boltzmann-relevant states (|Q| <= 1) separated from the exotic ones."""
import dataclasses, sys, warnings
import torch
warnings.simplefilter("ignore"); torch.set_default_dtype(torch.float64)
from ase.io import read
from flashace.calculator import TransformersACECalculator
import trace_redox.induction as I

model = sys.argv[1]; xyz = sys.argv[2] if len(sys.argv) > 2 else "experiments/benchmarks_4g/data/AuMgO_test.xyz"
calc = TransformersACECalculator(model_path=model, device="cpu"); torch.set_num_threads(2)
mod = calc.model.electrostatics; mod.exclusion = None
mod.redox = dataclasses.replace(mod.redox, carrier_penetration=True, carrier_overlap="gordon_kim", exclusion=False,
                                carrier_induction=True)
store = {}
orig = I.overlap_induction_terms
def wrapped(V, M, N, y0, f, Q):
    out = orig(V, M, N, y0, f, Q); store.update(Q=Q.detach(), E1=out[0].detach(), pairs=tuple(x.detach() for x in out[1]))
    return out
I.overlap_induction_terms = wrapped
a = [x for x in read(xyz, ":") if (x.numbers == 13).any()][0]
a.calc = calc; a.get_potential_energy()
Z = torch.as_tensor(a.numbers); Q, E1 = store["Q"], store["E1"]; iu, iw, Xi = store["pairs"]
print("E1 by element and carrier charge Q (eV): median / max |.|")
for z in sorted(set(Z.tolist())):
    for q in sorted(set(Q[Z == z].flatten().tolist())):
        if q == 0: continue
        m = (Q[Z == z] == q); v = E1[Z == z][m]
        print(f"  Z={z:2d} Q={q:+.0f}: median {float(v.median()):+.4f}  max|.| {float(v.abs().max()):.4f}  (n={int(m.sum())})")
Qa = Q[iu].abs().unsqueeze(2).expand_as(Xi); Qb = Q[iw].abs().unsqueeze(1).expand_as(Xi)
qmax = torch.maximum(Qa, Qb)
for lo, hi in ((0.5, 1.5), (1.5, 2.5), (2.5, 9)):
    m = (qmax > lo) & (qmax < hi) & (Qa > 0) & (Qb > 0)
    if int(m.sum()):
        k = Xi.abs()[m].argmax(); 
        print(f"pairs with max|Q| in ({lo},{hi}): {int(m.sum())} entries, max |Xi| {float(Xi.abs()[m].max()):.4f} eV, "
              f"99th pct {float(Xi.abs()[m].quantile(0.99)):.4f}")
# the largest entry
flat = Xi.abs().argmax(); p, s, t = torch.unravel_index(flat, Xi.shape)
u, w = int(iu[p]), int(iw[p]); R = float(torch.linalg.norm(torch.as_tensor(a.get_distance(u, w, mic=True))))
print(f"largest |Xi| {float(Xi[p, s, t]):+.4f} eV: Z {int(Z[u])} Q {float(Q[u, s]):+.0f} / Z {int(Z[w])} Q {float(Q[w, t]):+.0f}, R {R:.3f} A")
