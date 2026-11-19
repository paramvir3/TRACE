# Diagnostic probe (2026-10-10; reported, not a criterion).  Usage (repo root): python experiments/benchmarks_4g/probe_au_pattern.py CKPT [...]
"""Per-atom Au charges by geometry class: model (free and total) against DFT Hirshfeld, all 400 test structures.
'near' is the Au nearer the slab (larger z: the slab lies above the Au in these cells)."""
import sys, warnings; sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np, torch; torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from ase.io import read
from flashace.calculator import TransformersACECalculator
test = read("experiments/benchmarks_4g/data/AuMgO_test.xyz", ":")
def geometry(a):
    p = a.positions[a.numbers == 79]; v = p[1] - p[0]
    return "flat" if np.degrees(np.arccos(abs(v[2]) / np.linalg.norm(v))) > 45 else "upright"
for ck in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=ck, device="cpu"); torch.set_num_threads(3)
    cap = []; calc.model.electrostatics.register_forward_hook(lambda m, i, o: cap.append(o[1]))
    rows = {}
    for a0 in test:
        a = a0.copy(); a.info = dict(a0.info); a.calc = calc; cap.clear(); a.get_potential_energy(); r = cap[-1]
        au = np.where(a.numbers == 79)[0]; near = au[int(np.argmax(a.positions[au, 2]))]; far = au[au != near][0]
        key = ("doped" if (a.numbers == 13).any() else "undoped", geometry(a))
        f, q, h = r.free_charge.detach().numpy(), r.q.detach().numpy(), a0.arrays["hirshfeld_charges"]
        rows.setdefault(key, []).append([f[near], f[far], q[near], q[far], h[near], h[far]])
    print(f"=== {ck.split('/')[-2]}")
    print(f"{'class':18s} {'n':>4s} | {'free near':>9s} {'free far':>9s} | {'total near':>10s} {'total far':>10s} | {'DFT near':>8s} {'DFT far':>8s}")
    allm, alld = [], []
    for k in sorted(rows):
        x = np.array(rows[k]); m = x.mean(0)
        print(f"{k[0]+' '+k[1]:18s} {len(x):4d} | {m[0]:+9.3f} {m[1]:+9.3f} | {m[2]:+10.3f} {m[3]:+10.3f} | {m[4]:+8.3f} {m[5]:+8.3f}")
        allm += list(x[:, 2]) + list(x[:, 3]); alld += list(x[:, 4]) + list(x[:, 5])
    print(f"per-ATOM Pearson r(model total q_Au, DFT Hirshfeld q_Au) over all {len(allm)} Au atoms: {np.corrcoef(allm, alld)[0, 1]:+.3f}")
