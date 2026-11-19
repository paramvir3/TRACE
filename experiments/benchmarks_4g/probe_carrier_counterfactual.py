# Diagnostic probe (2026-10-16; reported, not a criterion).  Usage (repo root):
#   python experiments/benchmarks_4g/probe_carrier_counterfactual.py CKPT [N_PER_CLASS]
"""A trained checkpoint re-evaluated with the free-carrier partition (trace_redox SPEC 19), untrained for it:
'as trained' (carriers on atoms), 'covalent units' (sign-definite carriers per covalent unit; Au2's Au+Au- pair
becomes a bound fluctuation) and 'units + exclusion anchors' (SPEC 19.2).  Per (doping, geometry) class on the test
structures: mean total and free charges on the near / far Au against DFT Hirshfeld, and the class residual."""
import dataclasses, sys, warnings; sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np, torch; torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
from ase.io import read
from flashace.calculator import TransformersACECalculator

test = read("experiments/benchmarks_4g/data/AuMgO_test.xyz", ":")
ck = sys.argv[1]
per_class = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 6


def geometry(a):
    p = a.positions[a.numbers == 79]; v = p[1] - p[0]
    return "flat" if np.degrees(np.arccos(abs(v[2]) / np.linalg.norm(v))) > 45 else "upright"


picked, count = [], {}
for a0 in test:
    key = ("doped" if (a0.numbers == 13).any() else "undoped", geometry(a0))
    if count.get(key, 0) < per_class:
        picked.append((key, a0)); count[key] = count.get(key, 0) + 1
calc = TransformersACECalculator(model_path=ck, device="cpu"); torch.set_num_threads(2)
module = calc.model.electrostatics
trained = module.redox
cap = []
module.register_forward_hook(lambda m, i, o: cap.append(o[1]))
print(f"=== {ck.split('/')[-2]}: {len(picked)} test structures {dict(count)}")
modes = {"as trained": {}, "covalent units": {"carriers": "covalent_units"},
         "units + exclusion anchors": {"carriers": "covalent_units", "exclusion_bound": True}}
for mode, change in modes.items():
    if change.get("exclusion_bound") and not trained.exclusion:
        continue
    module.redox = dataclasses.replace(trained, **change)
    rows = {}
    for key, a0 in picked:
        a = a0.copy(); a.info = dict(a0.info); a.calc = calc; cap.clear()
        e = a.get_potential_energy(); r = cap[-1]
        au = np.where(a.numbers == 79)[0]; near = au[int(np.argmax(a.positions[au, 2]))]; far = au[au != near][0]
        f, q, h = r.free_charge.detach().numpy(), r.q.detach().numpy(), a0.arrays["hirshfeld_charges"]
        rows.setdefault(key, []).append([f[near], f[far], q[near], q[far], h[near], h[far],
                                         1e3 * (e - a0.get_potential_energy())])
    print(f"--- {mode}")
    for k in sorted(rows):
        x = np.array(rows[k]); m = x.mean(0)
        print(f"  {k[0] + ' ' + k[1]:16s} {len(x):3d} | free {m[0]:+.3f} {m[1]:+.3f} | total {m[2]:+.3f} {m[3]:+.3f} | "
              f"DFT {m[4]:+.3f} {m[5]:+.3f} | residual {m[6]:+9.1f}")
    for d in ("doped", "undoped"):
        if (d, "flat") in rows and (d, "upright") in rows:
            fl, up = np.array(rows[(d, "flat")])[:, 6], np.array(rows[(d, "upright")])[:, 6]
            print(f"  {d} flat - upright residual {fl.mean() - up.mean():+8.1f} meV")
module.redox = trained
