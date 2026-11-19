# Written on 2026-10-30 (the learned couplings against Harrison; theory note 4c); run from the repo root.
"""The learned couplings T of step 20 (fold off) on the undoped references, by element pair and distance shell,
against Harrison's universal two-centre values (V_ss sigma -1.40, V_sp sigma 1.84, V_pp sigma 3.24, V_pp pi -0.81,
times hbar^2/m d^2 = 7.62/d^2 eV)."""
import sys, dataclasses, collections, numpy as np, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
import probe_t5_first_order as Q
from flashace.calculator import TransformersACECalculator
Q._install()
refs = Q.references()
H2M = 7.62
for model in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=model, device="cpu"); torch.set_num_threads(2)
    mod = calc.model.electrostatics; mod.redox = dataclasses.replace(mod.redox, downfold_transfer=False)
    at = refs[("undoped", "wetting")]
    s = Q.evaluate(calc, at, 0.0)
    T = s["args"].get("T") if isinstance(s.get("args"), dict) else None
    if T is None:
        print(model.split('/')[-2][-3:], "no T in the probe's args; keys:", list(s["args"].keys())[:20]); continue
    T = T.detach(); z = np.array(at.get_atomic_numbers()); n = len(z)
    D = at.get_all_distances(mic=True)
    rows = collections.defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] > 4.0 or abs(float(T[i, j])) < 1e-9: continue
            key = tuple(sorted((int(z[i]), int(z[j])))); shell = round(float(D[i, j]), 1)
            rows[(key, shell)].append(abs(float(T[i, j])))
    print(f"=== {model.split('/')[-2][-3:]}: |T| by pair and distance (mean, max, count)", flush=True)
    for (key, shell), v in sorted(rows.items()):
        harr = {(8, 8): 3.24, (79, 79): 1.40, (8, 79): 1.84, (8, 12): 1.84, (8, 13): 1.84}.get(key, float("nan")) * H2M / shell ** 2
        print(f"   {key} at {shell:.1f} A: mean {np.mean(v):.3f}, max {np.max(v):.3f} eV (n {len(v)}); Harrison {harr:.3f}", flush=True)
