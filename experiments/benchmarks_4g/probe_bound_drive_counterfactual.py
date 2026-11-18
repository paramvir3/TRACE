# Diagnostic probe (2026-10-15; reported).  Usage (repo root): python experiments/benchmarks_4g/probe_bound_drive_counterfactual.py CKPT
"""Counterfactual on a trained FORMAL checkpoint: the same weights with the bound sector's own electronegativity drive
removed (RedoxConfig bound_electronegativity='none', SPEC 17.8; exactly E_learned(n) - E_none(n) = E_b0 + b.n).
Per undoped test structure: residual (model - DFT) with and without the drive, and the near/far Au total charges.
Not a trained model: it measures what the drive contributes to the flat - upright offset and to the Au charge sign."""
import sys, warnings, dataclasses; sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np, torch; torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from ase.io import read
from flashace.calculator import TransformersACECalculator
test = [a for a in read("experiments/benchmarks_4g/data/AuMgO_test.xyz", ":") if not (a.numbers == 13).any()]
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu"); torch.set_num_threads(3)
mod = calc.model.electrostatics
cap = []; mod.register_forward_hook(lambda m, i, o: cap.append(o[1]))
learned = mod.redox
rows = {"learned": [], "none": []}
for mode in ("learned", "none"):
    mod.redox = dataclasses.replace(learned, bound_electronegativity=mode)
    for a0 in test:
        a = a0.copy(); a.info = dict(a0.info); a.calc = calc; calc.reset(); cap.clear()
        res = 1e3 * (a.get_potential_energy() - a0.get_potential_energy())
        p = a0.positions; au = np.where(a0.numbers == 79)[0]; v = p[au[1]] - p[au[0]]
        tilt = np.degrees(np.arccos(abs(v[2]) / np.linalg.norm(v)))
        near = au[int(np.argmax(p[au, 2]))]; far = au[au != near][0]
        q = cap[-1].q.detach().numpy()
        rows[mode].append((tilt >= 45, res, q[near], q[far], a0.arrays["hirshfeld_charges"][near], a0.arrays["hirshfeld_charges"][far]))
mod.redox = learned
for mode in ("learned", "none"):
    x = np.array(rows[mode], dtype=float); flat = x[:, 0] > 0.5
    off = x[flat, 1].mean() - x[~flat, 1].mean()
    print(f"[{mode:7s}] flat - upright residual {off:+7.1f} meV | upright Au total q near/far {x[~flat, 2].mean():+.3f}/{x[~flat, 3].mean():+.3f}"
          f" flat {x[flat, 2].mean():+.3f}/{x[flat, 3].mean():+.3f} | DFT upright {x[~flat, 4].mean():+.3f}/{x[~flat, 5].mean():+.3f} flat {x[flat, 4].mean():+.3f}/{x[flat, 5].mean():+.3f}")
