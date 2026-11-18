# Written on 2026-10-30 (the diagnosis of step 22's first training step: CHANGELOG Round 28, theory note 4c); run from the repo root.
"""A forward with seed 42's initial weights (the nanguard dump's state) on training structures: time, the explicit
configuration count, leaves, folded states, and the fold's pieces timed.  Usage: python profile_init42.py MODEL DUMP IDX,IDX"""
import sys, time, dataclasses, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
from flashace.calculator import TransformersACECalculator
from ase.io import read
import trace_redox.diabatic as D, trace_redox.downfold as F, trace_redox.polarizability as P
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
model = calc.model; mod = model.electrostatics
dump = torch.load(sys.argv[2], map_location="cpu", weights_only=False); model.load_state_dict(dump["state"], strict=False)
frames = read("experiments/benchmarks_4g/data/AuMgO_train.xyz", ":")
counts = {}
def wrap(obj, name):
    f = getattr(obj, name)
    def g(*a, **k):
        t0 = time.time(); out = f(*a, **k); c = counts.setdefault(name, [0, 0.0]); c[0] += 1; c[1] += time.time() - t0; return out
    setattr(obj, name, g)
for obj, name in ((F, "dressing"), (F, "dressing_exact"), (F, "_leaves"), (F, "block_ground"), (D, "search_configurations"),
                  (D, "coupled_closure"), (P, "state_polarizability_corrections"), (D, "configuration_energies")):
    wrap(obj, name)
mod.redox = dataclasses.replace(mod.redox, downfold_transfer=True, downfold_hop_ordering="pauli", downfold_block="exact",
                                downfold_scale=0.85, downfold_free_axes=False, downfold_anisotropy=0.0, downfold_crystal_field=0.0)
for i in [int(x) for x in sys.argv[3].split(",")]:
    counts.clear(); a = frames[i].copy(); a.calc = calc; calc.reset()
    t0 = time.time(); a.get_potential_energy(); dt = time.time() - t0
    rec = getattr(mod, "_downfold_last", {}) or {}
    C = int(mod._ensemble_last["configs"].shape[0])
    print(f"[{i} {a.get_chemical_formula()}] forward {dt:.2f} s; configs {C}; leaves {rec.get('leaves')}; clusters {rec.get('clusters')}; "
          + "; ".join(f"{k} {v[0]}x {v[1]:.2f}s" for k, v in counts.items()), flush=True)
