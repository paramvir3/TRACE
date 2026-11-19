# Diagnostic probe (2026-10-15; reported).  Usage (repo root): python experiments/benchmarks_4g/probe_residual_geometry.py CKPT
"""Which geometric feature does a model's energy error follow?  Undoped test structures: residual r = E_model - E_DFT
(meV per structure, mean removed) against the Au2 tilt angle (0 upright, 90 flat), the lower Au's height above the top
O layer, the nearest Au-O and Au-Mg distances, and the Au-Au bond.  Pearson r and the least-squares slope of r on each
(single regressions), plus the residual binned by tilt."""
import sys, warnings; sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np, torch; torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
from ase.io import read
from flashace.calculator import TransformersACECalculator
test = [a for a in read("experiments/benchmarks_4g/data/AuMgO_test.xyz", ":") if not (a.numbers == 13).any()]
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu"); torch.set_num_threads(2)
rows = []
for a0 in test:
    a = a0.copy(); a.info = dict(a0.info); a.calc = calc
    res = 1e3 * (a.get_potential_energy() - a0.get_potential_energy())
    z = a0.numbers; p = a0.positions; au = np.where(z == 79)[0]
    v = p[au[1]] - p[au[0]]; tilt = np.degrees(np.arccos(abs(v[2]) / np.linalg.norm(v)))
    o = np.where(z == 8)[0]; mg = np.where(z == 12)[0]
    top_o = p[o, 2].min()                                   # the surface facing Au (Au lie below the slab: smaller z)
    low = au[int(np.argmax(p[au, 2]))]                      # the Au nearer the slab
    height = top_o - p[low, 2]
    d_o = min(a0.get_distances(i, o, mic=True).min() for i in au)
    d_mg = min(a0.get_distances(i, mg, mic=True).min() for i in au)
    rows.append([res, tilt, height, d_o, d_mg, a0.get_distance(au[0], au[1], mic=True)])
x = np.array(rows); x[:, 0] -= x[:, 0].mean()
names = ["tilt (deg)", "lower-Au height (A)", "min Au-O (A)", "min Au-Mg (A)", "Au-Au (A)"]
print(f"=== {sys.argv[1].split('/')[-2]}  undoped n={len(x)}; residual std {x[:, 0].std():.1f} meV")
for k, name in enumerate(names, start=1):
    r = np.corrcoef(x[:, 0], x[:, k])[0, 1]; slope = np.polyfit(x[:, k], x[:, 0], 1)[0]
    print(f"  {name:22s} range {x[:, k].min():7.2f} .. {x[:, k].max():7.2f}   Pearson r {r:+.3f}   slope {slope:+9.2f} meV/unit")
bins = [0, 15, 30, 45, 60, 75, 90.01]
print("  residual by tilt: " + "; ".join(f"[{bins[i]:.0f},{bins[i+1]:.0f}) {x[(x[:,1]>=bins[i])&(x[:,1]<bins[i+1]),0].mean():+.0f} (n={int(((x[:,1]>=bins[i])&(x[:,1]<bins[i+1])).sum())})"
                                    for i in range(len(bins) - 1) if ((x[:,1]>=bins[i])&(x[:,1]<bins[i+1])).any()))
# within each class (tilt below / above 45 deg): the dependence that the class confound cannot produce
for label, sel in (("upright", x[:, 1] < 45), ("flat", x[:, 1] >= 45)):
    y = x[sel, 0] - x[sel, 0].mean()
    parts = []
    for k, name in enumerate(names, start=1):
        if x[sel, k].std() > 1e-9:
            r = np.corrcoef(y, x[sel, k])[0, 1]; slope = np.polyfit(x[sel, k], y, 1)[0]
            parts.append(f"{name} r {r:+.2f} slope {slope:+.0f} (range {x[sel, k].min():.2f}..{x[sel, k].max():.2f})")
    print(f"  within {label} (n={int(sel.sum())}, residual std {y.std():.1f} meV): " + "; ".join(parts))
