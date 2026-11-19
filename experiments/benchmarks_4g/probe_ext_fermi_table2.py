"""C4 stage 2, reported numbers (not criteria): Table II on frozen weights with the Fermi band (SPEC 38).

wet = E(wetting) - E(non-wetting), Au relaxed on the mean substrate (eval_aumgo.relax, fmax 0.005 eV/A),
on the 1x1 cell, at the unit mesh and at a Born-von Karman mesh (SPEC 37); T5 = wet_d - wet_u.
Settings: 'onsite' stiffness (SPEC 28.6), 'clusters' space (SPEC 36, closure on), band_form 'fermi'.
Nothing is written into a run directory.

    PYTHONPATH=. python experiments/benchmarks_4g/probe_ext_fermi_table2.py MODEL.pt [doped] [undoped] [mesh=2,2,1 ...] [cap=N]

cap=N sets RedoxConfig.fragment_max_configurations for this measurement only (the cap refuses, never truncates).
"""
import dataclasses
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np
import torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references, relax, au_o_bond

model_path = sys.argv[1]
args = sys.argv[2:]
dopings = [d for d in ("undoped", "doped") if d in args] or ["undoped", "doped"]
meshes = [tuple(int(x) for x in a.split("=")[1].split(",")) for a in args if a.startswith("mesh=")] or [(1, 1, 1), (2, 2, 1)]
form = next((a.split("=")[1] for a in args if a.startswith("band_form=")), "fermi")
calc = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
base = dataclasses.replace(mod.redox, state_polarizability_stiffness="onsite", diabatic_space="clusters", band_form=form)
cap = next((int(a.split("=")[1]) for a in args if a.startswith("cap=")), None)
if cap is not None:
    base = dataclasses.replace(base, fragment_max_configurations=cap)
SEEN = {"mesh": None, "calls": 0}


def record(module, inputs, output):                               # the mesh the module used, read off its own record
    SEEN["mesh"] = tuple(output[1].bvk_mesh); SEEN["calls"] += 1


mod.register_forward_hook(record)
refs = references()
DFT = {"undoped": 928.9, "doped": -2.7}
print(f"model {model_path.split('/')[-2]}; 'onsite' stiffness; clusters space; band_form '{form}'; frozen weights; "
      f"cap {base.fragment_max_configurations}")
wet = {}
for mesh in meshes:
    mod.redox = dataclasses.replace(base, bvk_mesh=mesh)
    for doping in dopings:
        out = {}
        for geometry in ("wetting", "non-wetting"):
            calc.reset(); SEEN["calls"] = 0; t0 = time.time()
            a, e = relax(calc, refs[(doping, geometry)])
            assert len(a) == 110 and SEEN["mesh"] == mesh, (len(a), SEEN["mesh"])
            out[geometry] = e
            print(f"   bvk_mesh {mesh} {doping:7s} {geometry:11s}: E/cell {e:.6f} eV  "
                  f"Au-Au {np.linalg.norm(a.positions[109] - a.positions[108]):.4f} A  Au-O {au_o_bond(a):.4f} A  "
                  f"max|F(Au)| {np.abs(a.get_forces()[108:]).max():.1e} eV/A  module calls {SEEN['calls']}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
        wet[(mesh, doping)] = 1e3 * (out["wetting"] - out["non-wetting"])
        print(f"== bvk_mesh {mesh} {doping}: E(wetting) - E(non-wetting) = {wet[(mesh, doping)]:+.4f} meV "
              f"(DFT {DFT[doping]:+.1f})", flush=True)
    if (mesh, "doped") in wet and (mesh, "undoped") in wet:
        print(f"== bvk_mesh {mesh}: T5 = wet_d - wet_u = {wet[(mesh, 'doped')] - wet[(mesh, 'undoped')]:+.4f} meV (DFT -931.6)", flush=True)
print(time.strftime("done %Y-%m-%d %H:%M %Z"))
