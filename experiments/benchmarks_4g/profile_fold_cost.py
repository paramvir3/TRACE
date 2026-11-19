# Written on 2026-10-30 (the diagnosis of step 22's first training step: CHANGELOG Round 28, theory note 4c); run from the repo root.
"""Quiet cost of the fold: energy+forces per structure, per_state against exact (Pauli, c = 0.85), on several training
structures (undoped and doped), nothing else running."""
import sys, time, dataclasses, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
from flashace.calculator import TransformersACECalculator
from ase.io import read
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
mod = calc.model.electrostatics
frames = read("experiments/benchmarks_4g/data/AuMgO_train.xyz", ":")
idx = [int(i) for i in sys.argv[2].split(",")]
for i in idx:
    at = frames[i]
    row = [f"{i:4d} {at.get_chemical_formula():>14s}"]
    for blk in ("per_state", "exact"):
        mod.redox = dataclasses.replace(mod.redox, downfold_transfer=True, downfold_hop_ordering="pauli", downfold_block=blk,
                                        downfold_scale=0.85, downfold_free_axes=False, downfold_anisotropy=0.0, downfold_crystal_field=0.0)
        a = at.copy(); a.calc = calc; calc.reset()
        t0 = time.time(); a.get_forces(); dt = time.time() - t0
        rec = getattr(mod, "_downfold_last", {}) or {}
        row.append(f"{blk} {dt:5.2f} s (configs {int(mod._ensemble_last['configs'].shape[0])}, leaves {rec.get('leaves')}, clusters {rec.get('clusters')})")
    print("  ".join(row), flush=True)
