# Diagnostic probe (2026-10-15; reported, not a criterion).  Usage (repo root):
#   python experiments/benchmarks_4g/probe_exclusion_counterfactual.py CKPT [N_PER_CLASS]
"""A trained checkpoint's weights, re-evaluated with the exclusion term switched on, untrained (trace_redox SPEC 18):
'as trained' (none), 'frozen' (SPEC 18 and 18.3, K = 10 Ha, I_eff(O) = 9 eV) and 'relaxed' (SPEC 18.4, the same K and
I_eff, each state's frontier density relaxing within the bound sector's own alpha_u).  A checkpoint trained with
exclusion keeps its learned K and I_eff in both modes (then 'frozen' equals 'as trained').  On the test structures, per
(doping, geometry) class: mean free and total charges on the near / far Au against DFT Hirshfeld, the mean Al free
charge, and the class residual (model - DFT energy; the flat - upright offset is the H1 quantity).  'near' is the Au
nearer the slab (larger z: the slab lies above the Au in these cells).  Not retrained: a counterfactual, not a
prediction of a trained model."""
import dataclasses, sys, warnings; sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np, torch; torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
from ase.io import read
from flashace.calculator import TransformersACECalculator
from trace_redox.exclusion import Exclusion

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
trained = (module.redox, module.exclusion)
cap = []
module.register_forward_hook(lambda m, i, o: cap.append(o[1]))
print(f"=== {ck.split('/')[-2]}: {len(picked)} test structures {dict(count)}")
if trained[1] is not None:
    print(f"trained exclusion: K = {float(trained[1].K) / 27.211386245988:.3f} Ha, "
          f"I_eff(O) = {float(trained[1].i_eff(torch.tensor([8]))[0]):.3f} eV")
for mode in ("as trained", "frozen", "relaxed"):
    if mode == "as trained":
        module.redox, module.exclusion = trained
    else:
        module.redox = dataclasses.replace(trained[0], exclusion=True, exclusion_relaxation=(mode == "relaxed"))
        module.exclusion = Exclusion(r_cut=module.redox.exclusion_cutoff, relax=(mode == "relaxed"))
        if trained[1] is not None:                     # a checkpoint trained with exclusion keeps its K and I_eff
            module.exclusion.load_state_dict(trained[1].state_dict())
    rows = {}
    for key, a0 in picked:
        a = a0.copy(); a.info = dict(a0.info); a.calc = calc; cap.clear()
        e = a.get_potential_energy(); r = cap[-1]
        au = np.where(a.numbers == 79)[0]; near = au[int(np.argmax(a.positions[au, 2]))]; far = au[au != near][0]
        f, q, h = r.free_charge.detach().numpy(), r.q.detach().numpy(), a0.arrays["hirshfeld_charges"]
        al = f[a.numbers == 13].mean() if (a.numbers == 13).any() else np.nan
        rows.setdefault(key, []).append([f[near], f[far], q[near], q[far], h[near], h[far], al,
                                         1e3 * (e - a0.get_potential_energy())])
    print(f"--- {mode}")
    print(f"  {'class':16s} {'n':>3s} | {'free near':>9s} {'far':>7s} | {'total near':>10s} {'far':>7s} | "
          f"{'DFT near':>8s} {'far':>7s} | {'Al free':>7s} | {'residual meV':>12s}")
    qm, qd = [], []
    for k in sorted(rows):
        x = np.array(rows[k]); m = x.mean(0)
        print(f"  {k[0] + ' ' + k[1]:16s} {len(x):3d} | {m[0]:+9.3f} {m[1]:+7.3f} | {m[2]:+10.3f} {m[3]:+7.3f} | "
              f"{m[4]:+8.3f} {m[5]:+7.3f} | {m[6]:+7.3f} | {m[7]:+12.1f}")
        qm += list(x[:, 2]) + list(x[:, 3]); qd += list(x[:, 4]) + list(x[:, 5])
    for d in ("doped", "undoped"):
        if (d, "flat") in rows and (d, "upright") in rows:
            fl, up = np.array(rows[(d, "flat")])[:, 7], np.array(rows[(d, "upright")])[:, 7]
            print(f"  {d} flat - upright residual {fl.mean() - up.mean():+8.1f} meV")
    print(f"  per-atom r(total q_Au, Hirshfeld) {np.corrcoef(qm, qd)[0, 1]:+.3f}")
