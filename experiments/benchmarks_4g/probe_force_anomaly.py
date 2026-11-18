# Written on 2026-10-30 (the diagnosis of step 22's first training step: CHANGELOG Round 28, theory note 4c); run from the repo root.
"""Structure 76 of the training set under autograd anomaly detection: energy and forces with the fold (per_state and
exact), step-20 weights.  A 0/0 in any backward node raises with the forward call that made it."""
import sys, dataclasses, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
from flashace.calculator import TransformersACECalculator
from ase.io import read
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
mod = calc.model.electrostatics
at = read("experiments/benchmarks_4g/data/AuMgO_train.xyz", "76")
for blk in ("per_state", "exact"):
    mod.redox = dataclasses.replace(mod.redox, downfold_transfer=True, downfold_hop_ordering="pauli", downfold_block=blk,
                                    downfold_scale=0.85, downfold_free_axes=False, downfold_anisotropy=0.0, downfold_crystal_field=0.0)
    a = at.copy(); a.calc = calc; calc.reset()
    try:
        with torch.autograd.set_detect_anomaly(True):
            f = a.get_forces(); e = a.get_potential_energy()
        print(f"[{blk}] E {e:.4f} eV, |F|max {abs(f).max():.3f}, finite {bool(__import__('numpy').isfinite(f).all())}", flush=True)
    except Exception as exc:
        print(f"[{blk}] ANOMALY: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
