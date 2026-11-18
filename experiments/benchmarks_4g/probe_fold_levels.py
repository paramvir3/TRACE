# Written on 2026-10-29 (the measurement scripts of the theory note's section 4c); run from the repo root.
"""The model's own charge-transfer levels at the O-top contact: the fold's lowest leaves (Delta = E(q) - E(n) with
the inner correction, the fold's v) on the undoped upright and flat references, against the ALMO-state Delta_sigma*
(upright 2.92 eV, flat 1.20; free pair 1.50).  Usage: python otop_delta.py MODEL.pt"""
import sys, dataclasses, numpy as np, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
import probe_t5_first_order as Q
from flashace.calculator import TransformersACECalculator
Q._install()
refs = Q.references()
for model in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=model, device="cpu"); torch.set_num_threads(2)
    mod = calc.model.electrostatics
    mod.redox = dataclasses.replace(mod.redox, fragment_max_configurations=20000, downfold_transfer=True,
                                    downfold_hop_ordering="pauli", downfold_scale=0.85, downfold_anisotropy=0.0,
                                    downfold_crystal_field=0.0, downfold_free_axes=False)
    for key in (("undoped", "non-wetting"), ("undoped", "wetting")):
        at = refs[key]
        s = Q.evaluate(calc, at, 0.0)
        rec = mod._downfold_last
        sym = at.get_chemical_symbols()
        print(f"=== {model.split('/')[-2][-3:]} {key[1]}: {rec['leaves']} leaves, {rec['clusters']} clusters, mu_min {rec['mu_min']:+.3f}, lambda_min {rec['lambda_min']:+.3f} eV")
        for c, o, g, k, d, v in sorted(rec["lowest_leaves"]):
            r = at.get_distance(o, g, mic=True)
            print(f"   config {c}  {sym[o]}{o} -> {sym[g]}{g}  r {r:.2f} A  kind {k}  Delta {d:+.3f} eV  v {v:+.3f} eV")
