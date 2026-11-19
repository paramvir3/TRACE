"""C4 stage 2, where the doped supercell puts its electrons: the band's filling and the Au charges at Gamma on the
1x1 cell and on its 2x2 supercell (what bvk_mesh = (2, 2, 1) evaluates, SPEC 37), per seed.  Frozen weights;
'onsite' stiffness, clusters space (closure on), band_form 'fermi'; the doped reference geometries (not relaxed).
Nothing in the package or a run directory changes.

    PYTHONPATH=. python experiments/benchmarks_4g/probe_ext_fermi_filling.py MODEL.pt [cap=N]

cap=N sets RedoxConfig.fragment_max_configurations for this measurement only (the cap refuses, never truncates).
"""
import dataclasses
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references

model_path = sys.argv[1]
calc = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
mod.redox = dataclasses.replace(mod.redox, state_polarizability_stiffness="onsite", diabatic_space="clusters", band_form="fermi")
cap = next((int(x.split("=")[1]) for x in sys.argv[2:] if x.startswith("cap=")), None)
if cap is not None:
    mod.redox = dataclasses.replace(mod.redox, fragment_max_configurations=cap)
REC = {}
mod.register_forward_hook(lambda module, inputs, output: REC.update(record=output[1]))
refs = references()
print(f"model {model_path.split('/')[-2]}; doped reference geometries; 'onsite', clusters, 'fermi'; frozen weights; "
      f"cap {mod.redox.fragment_max_configurations}")
for geometry in ("wetting", "non-wetting"):
    for rep in ((1, 1, 1), (2, 2, 1)):
        a = refs[("doped", geometry)].copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
        a.calc = calc; calc.reset(); REC.clear(); t0 = time.time()
        e = a.get_potential_energy()
        rec, band, last = REC["record"], mod._band_last, mod._ensemble_last
        z = a.get_atomic_numbers(); copies = rep[0] * rep[1]
        n0 = last["reference"]
        gold = [i for i in range(len(a)) if z[i] == 79]
        free = " ".join(f"{float(rec.free_charge[i]):+.3f}" for i in gold)
        total = sum(float(rec.free_charge[i]) for i in gold)
        pops = " ".join(f"{float(x):.4f}" for x in band["populations"])
        moved = [(int(u), int(n0[u])) for u in range(int(n0.numel()) - 1) if int(n0[u]) != 0]
        print(f"   {geometry:11s} {rep[0]}x{rep[1]}: E/cell {e / copies:.6f} eV; band holds {-float(rec.unit_charge[-1]):.4f} e "
              f"({-float(rec.unit_charge[-1]) / copies:.4f} per cell); N0 {band['reference']} in {band['passes']} pass(es); "
              f"band-state populations {pops}; Au free charges {free} (sum per cell {total / copies:+.4f}); "
              f"clusters {[int(H.shape[0]) for H in last['hamiltonians']]}; units of n0 off state 0, (unit, state): {moved[:16]}"
              f"{' ...' if len(moved) > 16 else ''} ({len(moved)} in all)  ({time.time() - t0:.0f} s)", flush=True)
print(time.strftime("done %Y-%m-%d %H:%M %Z"))
