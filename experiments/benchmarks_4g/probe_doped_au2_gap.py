# Diagnostic probe (2026-10-16; reported).  Usage (repo root): python experiments/benchmarks_4g/probe_doped_au2_gap.py CKPT [N]
"""Doped test structures: in the model's configuration window, the lowest configuration with two electrons on Au2
(Au2(2-): both Au at -1) against the lowest with exactly one (Au2(-)).  Gap = E(best Au2-) - E(best Au2(2-)), eV
(positive: Au2(2-) is lower).  Also the Au-Au distance."""
import sys, warnings; sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np, torch; torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
from ase.io import read
from flashace.calculator import TransformersACECalculator
import trace_redox.module as M
test = [a for a in read("experiments/benchmarks_4g/data/AuMgO_test.xyz", ":") if (a.numbers == 13).any()]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu"); torch.set_num_threads(2)
cap = {}
orig = M.search_configurations
def hook(*a, **k):
    out = orig(*a, **k); cap.update(c=out[0], E=out[1]); return out
M.search_configurations = hook
rows = []
for a0 in test[:: max(1, len(test) // n)][:n]:
    a = a0.copy(); a.info = dict(a0.info); a.calc = calc; a.get_potential_energy()
    au = np.where(a.numbers == 79)[0]
    c, E = cap["c"], cap["E"]
    two = (c[:, au] == -1).all(dim=1); one = ((c[:, au] == -1).sum(dim=1) == 1)
    e2 = float(E[two].min()) if bool(two.any()) else float("inf")
    e1 = float(E[one].min()) if bool(one.any()) else float("inf")
    d = float(np.linalg.norm(a.positions[au[0]] - a.positions[au[1]]))
    rows.append((e1 - e2, d))
    print(f"  Au-Au {d:.3f} A  E(best Au2-) - E(best Au2(2-)) = {e1 - e2:+.3f} eV")
g = np.array([r[0] for r in rows if np.isfinite(r[0])])
print(f"=== {sys.argv[1].split('/')[-2]}: gap mean {g.mean():+.3f}, min {g.min():+.3f}, max {g.max():+.3f} eV over {len(g)} doped structures")
