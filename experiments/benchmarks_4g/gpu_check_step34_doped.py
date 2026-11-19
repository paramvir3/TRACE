"""Pre-flight (2) follow-up (2026-11-15): the doped training slab's CPU-GPU force difference (1.3e-7 eV/A) with the new
terms ON and OFF, to tell the new code from the existing solver.  Same model and structure as gpu_check_step34.py.

    PYTHONPATH=../.. python gpu_check_step34_doped.py DEVICE OUT.json
    python gpu_check_step34_doped.py compare CPU.json GPU.json
"""
import dataclasses, json, os, sys, warnings
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if sys.argv[1] == "compare":
    import numpy as np
    a, b = (json.load(open(p)) for p in sys.argv[2:4])
    for name in a:
        Fa, Fb = np.array(a[name]["F"]).reshape(-1, 3), np.array(b[name]["F"]).reshape(-1, 3)
        dF = np.abs(Fa - Fb).max(axis=1)
        k = int(np.argmax(dF))
        print(f"{name:12s}: |dE| {abs(a[name]['E'] - b[name]['E']):.2e} eV; largest |dF| {dF[k]:.2e} eV/A on atom {k} (Z {a[name]['Z'][k]}), "
              f"its force {np.linalg.norm(Fa[k]):.3f} eV/A; RMS |dF| {np.sqrt((dF ** 2).mean()):.2e}; largest |F| {np.abs(Fa).max():.3f}")
    sys.exit(0)
import torch
from ase.io import read
from flashace.calculator import TransformersACECalculator
device, out = sys.argv[1], sys.argv[2]
torch.set_num_threads(4)
calc = TransformersACECalculator(model_path=os.path.join(HERE, "runs", "AuMgO_single_all_e1kw_bondlaw_r5_s43", "model.pt"), device=device)
mod = calc.model.electrostatics
base = mod.redox
train = read(os.path.join(HERE, "..", "benchmarks_phonons", "data", "AuMgO_bulkaugw_train.xyz"), ":")
slab = next(a for a in train if len(a) == 110 and 13 in a.numbers)
res = {}
for name, ch in (("terms ON", dict(frame_pair=True, spin_pairing=True, two_site=True)), ("terms OFF", {})):
    mod.redox = dataclasses.replace(base, **ch)
    a = slab.copy(); a.set_constraint(); a.info = {"charge": 0.0}; a.calc = calc; calc.reset()
    res[name] = {"E": float(a.get_potential_energy()), "F": a.get_forces().reshape(-1).tolist(), "Z": a.numbers.tolist()}
    print(name, res[name]["E"], flush=True)
json.dump(res, open(out, "w"))
