"""Model - DFT energy residual per structure, averaged by (doping, geometry class), on all 400 test structures
(CLASS_SPLIT=train|valid and CLASS_SUBSET=N for a seeded random subset of another split).
If mean_res(flat) - mean_res(upright) ~ 0 the model reproduces DFT's flat-upright energy difference at the DFT
geometries, and a T5 error must come from the relaxed PES (outside the data); if it is large, the energies
themselves are wrong in the data region."""
import sys, warnings; sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np, torch; torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from ase.io import read
from flashace.calculator import TransformersACECalculator
import os
SPLIT = os.environ.get("CLASS_SPLIT", "test")                      # test (default) | train | valid
test = read(f"experiments/benchmarks_4g/data/AuMgO_{SPLIT}.xyz", ":")
if int(os.environ.get("CLASS_SUBSET", "0")) > 0:                    # a seeded random subset (reported as such)
    pick = np.random.default_rng(0).choice(len(test), int(os.environ["CLASS_SUBSET"]), replace=False)
    test = [test[i] for i in sorted(pick)]
def geom(a):
    p = a.positions[a.numbers == 79]; v = p[1] - p[0]
    return "flat" if np.degrees(np.arccos(abs(v[2]) / np.linalg.norm(v))) > 45 else "upright"
for ck in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=ck, device="cpu"); torch.set_num_threads(3)
    res = {}
    for a0 in test:
        a = a0.copy(); a.info = dict(a0.info); a.calc = calc
        key = ("doped" if (a0.numbers == 13).any() else "undoped", geom(a0))
        res.setdefault(key, []).append(1e3 * (a.get_potential_energy() - a0.get_potential_energy()))
    print(f"=== {ck.split('/')[-2]}   ({SPLIT}, {len(test)} structures; residual = model - DFT, meV per structure)")
    for d in ("doped", "undoped"):
        f, u = np.array(res[(d, "flat")]), np.array(res[(d, "upright")])
        se = np.sqrt(f.var(ddof=1) / len(f) + u.var(ddof=1) / len(u))
        print(f"  {d:8s} mean residual flat {f.mean():+8.1f} (n={len(f)})  upright {u.mean():+8.1f} (n={len(u)})  "
              f"flat - upright {f.mean() - u.mean():+8.1f} +- {se:.1f} meV")
