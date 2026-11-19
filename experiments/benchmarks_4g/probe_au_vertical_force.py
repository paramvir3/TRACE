# Diagnostic probe (2026-10-15; reported).  Usage (repo root): python experiments/benchmarks_4g/probe_au_vertical_force.py CKPT
"""Is the Au-surface energy slope wrong AT FIXED SUBSTRATE?  Signed error of the force on each Au along the surface
normal (towards the slab is +z here: the slab lies above the Au), model - DFT, in undoped test structures, by class
and against the Au's height.  A real missing attraction of ~0.2 eV/A shows as a systematic signed error; accurate
vertical forces mean the energy-height correlation across structures is a substrate confound."""
import sys, warnings; sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np, torch; torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from ase.io import read
from flashace.calculator import TransformersACECalculator
test = [a for a in read("experiments/benchmarks_4g/data/AuMgO_test.xyz", ":") if not (a.numbers == 13).any()]
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu"); torch.set_num_threads(3)
rows = []
for a0 in test:
    a = a0.copy(); a.info = dict(a0.info); a.calc = calc
    dF = a.get_forces() - a0.get_forces()
    p = a0.positions; au = np.where(a0.numbers == 79)[0]; v = p[au[1]] - p[au[0]]
    flat = np.degrees(np.arccos(abs(v[2]) / np.linalg.norm(v))) >= 45
    top_o = p[a0.numbers == 8, 2].min()
    for i in au:
        rows.append((flat, top_o - p[i, 2], dF[i, 2], a0.get_forces()[i, 2]))
x = np.array(rows, dtype=float)
print(f"=== {sys.argv[1].split('/')[-2]}  (dF_z = F_model - F_DFT on Au, eV/A; +z points towards the slab)")
for label, sel in (("upright", x[:, 0] < 0.5), ("flat", x[:, 0] > 0.5)):
    y = x[sel]
    near = y[:, 1] < np.median(y[:, 1])
    r = np.corrcoef(y[:, 1], y[:, 2])[0, 1]
    print(f"  {label:8s} Au atoms n={len(y)}: mean signed dF_z {y[:, 2].mean():+.4f} +- {y[:, 2].std(ddof=1) / np.sqrt(len(y)):.4f}; "
          f"closer half {y[near, 2].mean():+.4f}, farther half {y[~near, 2].mean():+.4f}; r(height, dF_z) {r:+.2f}; "
          f"rms DFT F_z {np.sqrt((y[:, 3] ** 2).mean()):.3f}")
