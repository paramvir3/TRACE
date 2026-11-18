# One of the measurement scripts of the theory note's section 4c (2026-10-29); run from the repo root.
"""A vertical Au2 (axis along the normal, bond 2.545 A) with its lower Au at height h = 2.89 A above the surface, at a
site: 'mgtop' (above Mg0: the flat dimer's site, the acceptor now vertical) or 'hollow' (above the centre of the
O1-Mg0-O(0,-1)-Mg(1,-1) square: no cation below, O1 at the same 3.6 A).  Usage: python site_direction.py SITE OUT"""
import sys, json, os
import numpy as np
sys.path.insert(0, "tools/electrostatics")
import almo_split as AS, wetting_eda as W, cluster_interaction as CI
site, out = sys.argv[1], sys.argv[2]
G = W.geometries()
sites = W.quantum_sites(G["flat"] + G["upright"], r_au=3.6)
lattice = CI.rocksalt_lattice(W.D, 5, True, exclude=sites)
d = W.D; h = 2.89
# 'hollow' (the O1-Mg0-O(0,-1)-Mg(1,-1) square) is legacy: in this 10-ion cluster O(0,-1) is a bare point ion
# 3.26 A from the Au, inside the cluster's 3.6 A design rule (stopped 2026-10-29 12:30).  'hollow_pp' is the
# O(1,0)-Mg0-O(0,1)-Mg(1,1) square: both O's quantum, no point ion within 3.6 A, the same embedding as 'mgtop'.
# 'flat_ref' is the flat reference dimer (h = 2.89) in this same cluster: reading (A)'s denominator.
base = {"mgtop": np.array([0.0, 0.0, h]), "hollow": np.array([0.5 * d, -0.5 * d, h]),
        "hollow_pp": np.array([0.5 * d, 0.5 * d, h])}.get(site)
au = G["flat"] if site == "flat_ref" else [base, base + np.array([0.0, 0.0, 2.545])]
res = AS.run_direction(f"vertical_{site}", au, sites, lattice)
json.dump(res, open(os.path.join(out, f"vertical_{site}.json"), "w"), indent=1)
