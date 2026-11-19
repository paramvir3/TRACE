# Written on 2026-10-30 (the epoch-1 outliers of step 22, benchmarks README); run from the repo root.
"""The training outliers of step 22's first epoch (seed 42): energy and force errors against the DFT labels with the
fold on (warm-start weights = step 20's), the fold record (promotions, explicit weights, gaps) and the geometry."""
import sys, dataclasses, numpy as np, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
from flashace.calculator import TransformersACECalculator
from ase.io import read
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
mod = calc.model.electrostatics
frames = read("experiments/benchmarks_4g/data/AuMgO_train.xyz", ":")
for i in [int(x) for x in sys.argv[2].split(",")]:
    at = frames[i]; sym = at.get_chemical_symbols()
    AU = [k for k, s_ in enumerate(sym) if s_ == "Au"]; O_ = [k for k, s_ in enumerate(sym) if s_ == "O"]
    dAuO = [float(at.get_distances(g, O_, mic=True).min()) for g in AU]; dAuAu = float(at.get_distance(AU[0], AU[1], mic=True))
    E_ref = at.get_potential_energy() if at.calc is not None else at.info.get("energy", float("nan"))
    F_ref = at.get_forces() if at.calc is not None else at.arrays.get("forces")
    out = {}
    for label, on in (("fold off", False), ("fold on", True)):
        mod.redox = dataclasses.replace(mod.redox, downfold_transfer=on, downfold_hop_ordering="pauli", downfold_block="exact",
                                        downfold_scale=0.85, downfold_free_axes=False, downfold_anisotropy=0.0, downfold_crystal_field=0.0)
        a = at.copy(); a.calc = calc; calc.reset()
        E = a.get_potential_energy(); F = a.get_forces()
        rec = getattr(mod, "_downfold_last", {}) or {}
        dF = F - F_ref if F_ref is not None else np.full_like(F, np.nan)
        out[label] = (E, float(np.abs(dF).max()), float(np.sqrt((dF ** 2).mean())), int(np.abs(dF).max(axis=1).argmax()),
                      len(rec.get("promoted", []) or []), [round(w, 3) for w in rec.get("explicit_weight", [])], rec.get("mu_min"))
    print(f"[{i} {at.get_chemical_formula()}] dAuO {dAuO[0]:.2f}/{dAuO[1]:.2f} dAuAu {dAuAu:.2f}; E_ref {E_ref:.3f}", flush=True)
    for label, (E, fmax, frms, worst, nprom, w, gap) in out.items():
        print(f"    {label:9s}: dE {E - E_ref:+.3f} eV; |dF|max {fmax:.3f} eV/A on {sym[worst]}{worst}; dF rms {frms:.3f}; promoted {nprom}; explicit weights {w}; gap {gap}", flush=True)
