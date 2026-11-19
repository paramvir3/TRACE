# Diagnostic probe (2026-10-16; reported).  Usage (repo root): python experiments/benchmarks_4g/probe_compressed_dimer.py CKPT [...]
"""Force errors on Au in the compressed-Au2 undoped test structures (Au-Au < 2.21 A; training has 1 such structure,
test 7).  'Structure 87' is undoped test structure 87 (Au-Au 2.089 A), the pre-registered C1 case."""
import sys, warnings; sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np, torch; torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
from ase.io import read
from flashace.calculator import TransformersACECalculator
test = read("experiments/benchmarks_4g/data/AuMgO_test.xyz", ":")
undoped = [a for a in test if not (a.numbers == 13).any()]
def d_auau(a):
    p = a.positions[a.numbers == 79]; return float(np.linalg.norm(p[1] - p[0]))
picked = [i for i, a in enumerate(undoped) if d_auau(a) < 2.21]
for ck in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=ck, device="cpu"); torch.set_num_threads(2)
    print(f"=== {ck.split('/')[-2]}")
    for i in picked:
        a0 = undoped[i]; a = a0.copy(); a.info = dict(a0.info); a.calc = calc
        dF = a.get_forces() - a0.get_forces()
        au = a0.numbers == 79
        print(f"  undoped {i:3d}  Au-Au {d_auau(a0):.3f} A  max |dF| on Au {np.linalg.norm(dF[au], axis=1).max():6.2f} eV/A"
              f"  (DFT |F| on Au {np.linalg.norm(a0.get_forces()[au], axis=1).max():6.2f})")
