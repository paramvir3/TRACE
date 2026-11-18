"""PBE's vertical approach of the upright Au2 on the O-top O (2026-11-05): dE_SCF(h) = E_AB - E_A - E_B (counterpoise
basis, the dimer's grid) for the contact Au at height h and the far Au 2.545 A above it, on almo_tilt.py's 6-ion
cluster; the target for the isotropic wall + SPEC 33 at theta = 0.  Usage: otop_upright_scan.py PATH.json OUT.json h [h ...]"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cluster_interaction as CI, wetting_eda as W  # noqa: E402
from pyscf import dft  # noqa: E402
P = json.load(open(sys.argv[1])); out_json = sys.argv[2]
tilted = [np.array(p) for p in P["cluster_au"]["1.0"]]; upright = [np.array(p) for p in P["cluster_au"]["upright"]]
sites = W.quantum_sites(tilted + upright, r_au=3.6)
surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf); lattice = CI.rocksalt_lattice(W.D, 5, True, exclude=sites)
res = json.load(open(out_json)) if os.path.exists(out_json) else {}
for h in map(float, sys.argv[3:]):
    key = f"{h:.2f}"
    if key not in res:
        au = [("Au", np.array([upright[0][0], upright[0][1], h])), ("Au", np.array([upright[0][0], upright[0][1], h + 2.545]))]
        dimer = W._mol(au + surf, qB); grids = dft.gen_grid.Grids(dimer); grids.level = 3; grids.build()
        _, eAB = W._solve(W._rks(dimer, grids, lattice))
        _, eA = W._solve(W._rks(W._mol(au, 0, ghosts=surf), grids))
        _, eB = W._solve(W._rks(W._mol_order_B(surf, qB, au), grids, lattice))
        res[key] = (eAB - eA - eB) * W.HARTREE
        json.dump(res, open(out_json, "w"), indent=1)
    print(f"  upright dimer, contact at {h:.2f} A: dE_SCF(cp) {res[key]:+.4f} eV", flush=True)
