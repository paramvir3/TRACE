"""Step 34's pre-flight (2): the same model and structures on the Mac's CPU and on the GPU host's GPU (README, "Pre-
registered: step 34"; 2026-11-15).  The final step 33 model of seed 43 with frame_pair, spin_pairing and two_site
switched on (frozen; a check of the code path, not of accuracy).

    PYTHONPATH=../.. python gpu_check_step34.py cpu  OUT.json          (the Mac)
    PYTHONPATH=../.. python gpu_check_step34.py cuda OUT.json          (the GPU host)
    python gpu_check_step34.py compare CPU.json GPU.json               (the Mac)

Structures: two training slabs (one doped), one bulk cell, the eps prior cell, test 167 (a compressed dimer), and gas-
phase Au2+ at 2.47 A.  Energies and forces in float64; the comparison's tolerance is 1e-8 eV per atom and 1e-7 eV/A.
"""
import dataclasses
import json
import os
import sys
import time
import warnings

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if sys.argv[1] == "compare":
    a, b = (json.load(open(p)) for p in sys.argv[2:4])
    worst = 0
    for name in a:
        dE = abs(a[name]["E"] - b[name]["E"]) / a[name]["n"]
        dF = max(abs(x - y) for x, y in zip(a[name]["F"], b[name]["F"]))
        ok = dE <= 1e-8 and dF <= 1e-7
        worst += int(not ok)
        print(f"{name:22s} |dE|/atom {dE:.2e} eV, largest |dF| {dF:.2e} eV/A, time CPU {a[name]['s']:.1f} s / GPU {b[name]['s']:.1f} s  {'OK' if ok else 'DIFFERENT'}")
    sys.exit(1 if worst else 0)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.io import read  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

device, out = sys.argv[1], sys.argv[2]
torch.set_num_threads(4)
calc = TransformersACECalculator(model_path=os.path.join(HERE, "runs", "AuMgO_single_all_e1kw_bondlaw_r5_s43", "model.pt"), device=device)
mod = calc.model.electrostatics
mod.redox = dataclasses.replace(mod.redox, frame_pair=True, spin_pairing=True, two_site=True)
train = read(os.path.join(HERE, "..", "benchmarks_phonons", "data", "AuMgO_bulkaugw_train.xyz"), ":")
slabs = [a for a in train if len(a) == 110]
pick = {"slab undoped": next(a for a in slabs if 13 not in a.numbers), "slab doped": next(a for a in slabs if 13 in a.numbers),
        "bulk cell": next(a for a in train if len(a) == 64),
        "eps prior cell": read(os.path.join(HERE, "..", "benchmarks_phonons", "data", "MgO_bulk_eps_prior_pbe.xyz")),
        "test 167": read(os.path.join(HERE, "data", "AuMgO_test.xyz"), 167),
        "Au2+ gas 2.47": Atoms("Au2", positions=[[0, 0, 0], [0, 0, 2.47]], pbc=False)}
pick["Au2+ gas 2.47"].info = {"charge": 1.0}
res = {}
for name, atoms in pick.items():
    a = atoms.copy()
    a.set_constraint()
    a.info = {"charge": float(atoms.info.get("charge", 0.0))}
    a.calc = calc
    calc.reset()
    t0 = time.time()
    E = float(a.get_potential_energy())
    F = a.get_forces().reshape(-1).tolist()
    res[name] = {"E": E, "F": F, "n": len(a), "s": time.time() - t0}
    print(f"{name:22s} E {E:+.10f} eV  ({res[name]['s']:.1f} s)", flush=True)
json.dump(res, open(out, "w"))
