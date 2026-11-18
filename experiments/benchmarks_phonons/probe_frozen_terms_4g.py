"""Frozen test (2026-11-07; approved and pre-registered in the README): which of SPEC 35 (frame-polarizability anchor)
and SPEC 34 (frame-pair overlap damping) costs the 4G slab accuracy that step 28b lost (undoped F RMSE 0.07 -> 0.16 eV/A).
Step 20's weights, frozen, on a fixed subset of the 4G test set (every 4th structure: 50 undoped + 50 doped), with
(a) neither, (b) the anchor alone (g = 1, as 28b trained it; and g = 0.92, the PBE-eps_inf value), (c) the damping alone,
(d) both.  Force RMSE per class against DFT, and the energy RMSE about each class's own mean (an offset-free number).
Usage: python experiments/benchmarks_phonons/probe_frozen_terms_4g.py MODEL.pt"""
import dataclasses, math, sys, warnings
sys.path.insert(0, "."); warnings.simplefilter("ignore")
import numpy as np, torch
torch.set_default_dtype(torch.float64); torch.set_num_threads(8)
from ase.io import read
from flashace.calculator import TransformersACECalculator
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu"); torch.set_num_threads(8)
mod = calc.model.electrostatics
test = read("experiments/benchmarks_4g/data/AuMgO_test.xyz", ":")[::4]
classes = {"undoped": [a for a in test if 13 not in a.numbers], "doped": [a for a in test if 13 in a.numbers]}
base = mod.redox
settings = [("neither (step 20 as trained)", dict(), None), ("anchor, g = 1", dict(frame_polarizability_anchor=True), 1.0),
            ("anchor, g = 0.92", dict(frame_polarizability_anchor=True), 0.92), ("damping only", dict(frame_induction_damping=True), None),
            ("anchor g = 1 + damping (28b's start)", dict(frame_polarizability_anchor=True, frame_induction_damping=True), 1.0)]
print(f"=== {sys.argv[1].split('/')[-2]}: {len(classes['undoped'])} undoped + {len(classes['doped'])} doped test structures", flush=True)
for label, flags, g in settings:
    mod.redox = dataclasses.replace(base, **flags)
    if g is not None:
        mod.log_alpha_scale = torch.nn.Parameter(torch.tensor(math.log(g), dtype=torch.float64))
    out = []
    for cls, atoms in classes.items():
        dF, dE = [], []
        for a in atoms:
            b = a.copy(); b.info = {"charge": float(a.info.get("charge", 0.0))}; b.calc = calc; calc.reset()
            try:
                F = b.get_forces(); E = b.get_potential_energy()
            except Exception as err:
                print(f"   {label} {cls}: refused ({type(err).__name__}: {str(err)[:80]})", flush=True); continue
            dF.append(F - a.get_forces()); dE.append((E - a.get_potential_energy()) / len(a))
        dF = np.concatenate(dF); dE = np.array(dE)
        out.append(f"{cls} F RMSE {np.sqrt((dF ** 2).mean()):.4f} eV/A, E RMSE about the class mean {1000 * dE.std():.2f} meV/atom (mean offset {1000 * dE.mean():+.1f})")
    print(f"{label}: " + " | ".join(out), flush=True)
mod.redox = base
