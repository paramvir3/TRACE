"""PBE binding curve of an Au2 dimer standing over a surface Mg (Mg0 of the ten-ion embedded cluster), lower Au at
height h: Delta E_cp(h) = E_AB - E_A(ghost B) - E_B(ghost A) (def2-TZVP, def2 ECP on Au, the point-ion lattice as in
wetting_eda / almo_split).  The reference for the model's short-range Au - Mg2+ repulsion (2026-10-30: the training set
holds no Au - Mg contact below 2.43 A and the Pauli table's Au - Mg window starts at 2.66 A).
Usage: python mgtop_scan.py H OUT_DIR"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import almo_split as AS, wetting_eda as W, cluster_interaction as CI
from pyscf import dft
h, out = float(sys.argv[1]), sys.argv[2]
G = W.geometries()
sites = W.quantum_sites(G["flat"] + G["upright"], r_au=3.6)
lattice = CI.rocksalt_lattice(W.D, 5, True, exclude=sites)
surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf)
A_atoms = [("Au", np.array([0.0, 0.0, h])), ("Au", np.array([0.0, 0.0, h + 2.545]))]
dimer = W._mol(A_atoms + surf, qB)
grids = dft.gen_grid.Grids(dimer); grids.level = 3; grids.build()
_, eAg = W._solve(W._rks(W._mol(A_atoms, 0, ghosts=surf), grids))
_, eBg = W._solve(W._rks(W._mol_order_B(surf, qB, A_atoms), grids, lattice))
mfAB, eAB = W._solve(W._rks(dimer, grids, lattice))
res = {"h": h, "E_AB": eAB * W.HARTREE, "E_A_ghost": eAg * W.HARTREE, "E_B_ghost": eBg * W.HARTREE,
       "dE_cp_eV": (eAB - eAg - eBg) * W.HARTREE, "converged": bool(mfAB.converged)}
json.dump(res, open(os.path.join(out, f"mgtop_h{h:.2f}.json"), "w"), indent=1)
print(f"mgtop h {h:.2f}: dE_cp {res['dE_cp_eV']:+.4f} eV (converged {res['converged']})", flush=True)
