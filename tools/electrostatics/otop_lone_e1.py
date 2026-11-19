"""E1(R) of the lone Au0 straight above the O-top O of MgO(100) (2026-11-05): the frame pair's frozen interaction
(penetration electrostatics + Pauli) that SPEC 33's law is measured relative to, on almo_tilt.py's 6-ion cluster at the
heights R given (frz_nonadditivity.first_order).  Usage: otop_lone_e1.py PATH.json OUT.json R [R ...]"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cluster_interaction as CI, frz_nonadditivity as FN, wetting_eda as W  # noqa: E402
P = json.load(open(sys.argv[1])); out_json = sys.argv[2]
tilted = [np.array(p) for p in P["cluster_au"]["1.0"]]; upright = [np.array(p) for p in P["cluster_au"]["upright"]]
sites = W.quantum_sites(tilted + upright, r_au=3.6)
surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf); lattice = CI.rocksalt_lattice(W.D, 5, True, exclude=sites)
res = json.load(open(out_json)) if os.path.exists(out_json) else {}
for R in map(float, sys.argv[3:]):
    key = f"{R:.2f}"
    if key not in res:
        res[key] = FN.first_order([np.array([upright[0][0], upright[0][1], R])], surf, qB, lattice)
        json.dump(res, open(out_json, "w"), indent=1)
    print(f"  lone Au0 at {R:.2f} A over the O-top: E1 {res[key]:+.4f} eV", flush=True)
