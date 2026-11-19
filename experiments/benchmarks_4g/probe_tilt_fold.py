"""The fold's O-top dressing on the tilted dimer against the upright one on the same O (2026-11-04; theory note §4c):
the model's counterpart of almo_tilt.py's R.  Geometries in the 4G undoped cell: the path's lambda = 1 state (the
contact Au straight above an O) and the upright dimer on that O (2.26 / 4.805 A below it: the slab lies above).
Per geometry: the total energy, the ensemble, each configuration's fold dressing and the weighted dressing sum p.dressing.
Usage (repo root): python experiments/benchmarks_4g/probe_tilt_fold.py PATH.json MODEL.pt [...]"""
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
warnings.simplefilter("ignore")
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
from flashace.calculator import TransformersACECalculator  # noqa: E402
from eval_aumgo import AU, references  # noqa: E402

P = json.load(open(sys.argv[1]))
ref = references()[("undoped", "wetting")]
tilted = np.array(P["cell_au"]["1.0"])
low = int(np.argmax(tilted[:, 2]))                         # the Au nearest the slab (larger z: the slab is above)
O = [i for i in range(108) if ref.get_chemical_symbols()[i] == "O"]
o = O[int(np.argmin([np.linalg.norm(ref.positions[i] - tilted[low]) for i in O]))]
upright = np.array([ref.positions[o] - [0, 0, 2.26], ref.positions[o] - [0, 0, 4.805]])
geoms = {"flat": ref.positions[AU].copy(), "tilted": tilted, "upright on the same O": upright}
print(f"contact O {o}: tilted contact Au at {np.linalg.norm(tilted[low] - ref.positions[o]):.3f} A, its in-plane offset "
      f"{np.linalg.norm((tilted[low] - ref.positions[o])[:2]):.3f} A")
for model in sys.argv[2:]:
    calc = TransformersACECalculator(model_path=model, device="cpu")
    torch.set_num_threads(3)
    mod = calc.model.electrostatics
    print(f"=== {os.path.basename(os.path.dirname(model))[-40:]} (fold {'on' if mod.redox.downfold_transfer else 'off'})")
    E = {}
    for name, pos in geoms.items():
        a = ref.copy(); a.positions[AU] = pos; a.info = {"charge": 0.0}; a.calc = calc; calc.reset()
        mod._downfold_last = None
        E[name] = a.get_potential_energy()
        H = mod._ensemble_last["H"].detach().numpy(); cfg = mod._ensemble_last["configs"]
        w, V = np.linalg.eigh(H); p = V[:, 0] ** 2
        rec = mod._downfold_last or {}
        dress = np.array(rec.get("dressing", [0.0] * len(p)))
        au = [[int(c[k]) for k in AU] for c in cfg]
        print(f"  {name:22s} E - E(flat) {E[name] - E['flat']:+.3f} eV; configurations (Au charges, p, dressing) "
              + "; ".join(f"{a_} {pi:.2f} {d:+.3f}" for a_, pi, d in zip(au, p, dress))
              + f"; p.dressing {float(p @ dress) if len(dress) == len(p) else float('nan'):+.3f} eV", flush=True)
