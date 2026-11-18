"""Is the tilted dimer's frozen penalty pairwise in the cluster's own densities? (2026-11-04; theory note §4c)

    OMP_NUM_THREADS=4 nice python tools/electrostatics/frz_nonadditivity.py PATH.json OUT.json [R_AU]
    (python: an interpreter with PySCF installed)

E1 = E_AB[P_HL] - E_A - E_B (the antisymmetrised product of the SCF monomers, both in the AB basis, per spin; the route of
wetting_eda.run_geometry_doped) for A = the dimer (closed shell) and A = each Au atom alone (doublet) at its position in
the tilted (path lambda = 1) and the upright dimer on the same O, B = almo_tilt.py's 6-ion cluster in its point-ion lattice
(undoped, bare boundary). The non-additivity D = E1(dimer) - E1(atom 1) - E1(atom 2) per geometry; (n1)-(n3) read
D(tilted) - D(upright). Also reported: E1(dimer) against ALMO's FRZ (the same quantity in the own-basis convention).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cluster_interaction as CI  # noqa: E402
import wetting_eda as W  # noqa: E402

HARTREE = W.HARTREE


def first_order(au, surf, qB, lattice):
    """E1 (eV) of the Au atoms ``au`` (list of positions) over the cluster ``surf``."""
    from pyscf import dft, gto
    A_atoms = [("Au", p) for p in au]
    spin = len(au) % 2

    def mol(atoms, charge, sp, ghosts=()):
        spec = [[sym, tuple(p)] for sym, p in atoms] + [["ghost-" + sym, tuple(p)] for sym, p in ghosts]
        basis = {sym: W.BASIS for sym, _ in list(atoms)} | {"ghost-" + sym: W.BASIS for sym, _ in ghosts}
        ecp = {"Au": W.BASIS} if any(sym == "Au" for sym, _ in atoms) else {}
        return gto.M(atom=spec, basis=basis, ecp=ecp, charge=charge, spin=sp, unit="Angstrom", verbose=0)

    dimer = mol(A_atoms + surf, qB, spin)
    grids = dft.gen_grid.Grids(dimer)
    grids.level = 3
    grids.build()
    molA = mol(A_atoms, 0, spin, ghosts=surf)
    molB = W._mol_order_B(surf, qB, A_atoms)
    n, nA = dimer.nao_nr(), molA.nao_nr()
    assert molB.nao_nr() == n and nA == n, "A and B must both be in the AB basis"
    mfA, eA = W._solve(W._uks(molA, grids))
    mfB, eB = W._solve(W._rks(molB, grids, lattice))
    mfAB = W._uks(dimer, grids, lattice)                  # its Hamiltonian only: E1 needs no AB SCF
    S = dimer.intor_symmetric("int1e_ovlp")
    CB = mfB.mo_coeff[:, mfB.mo_occ > 0]
    P = []
    for sp in (0, 1):
        CA = mfA.mo_coeff[sp][:, mfA.mo_occ[sp] > 0]
        C = np.hstack([CA, CB])
        P.append(C @ np.linalg.solve(C.T @ S @ C, C.T))
    E_hl = float(mfAB.energy_tot(dm=np.array(P)))
    return (E_hl - eA - eB) * HARTREE


def main(path_json, out_json, r_au=3.6):
    P = json.load(open(path_json))
    G = {"tilted": [np.array(p) for p in P["cluster_au"]["1.0"]], "upright": [np.array(p) for p in P["cluster_au"]["upright"]]}
    sites = W.quantum_sites(G["tilted"] + G["upright"], r_au=r_au)
    surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
    qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf)
    lattice = CI.rocksalt_lattice(W.D, 5, True, exclude=sites)
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for name, au in G.items():
        for label, atoms in (("dimer", au), ("atom 0", au[:1]), ("atom 1", au[1:])):
            key = f"{name} {label}"
            if key not in res:
                res[key] = first_order(atoms, surf, qB, lattice)
                json.dump(res, open(out_json, "w"), indent=1)
            print(f"  {key}: E1 {res[key]:+.4f} eV (Au heights {[round(float(p[2]), 3) for p in atoms]})", flush=True)
    D = {g: res[f"{g} dimer"] - res[f"{g} atom 0"] - res[f"{g} atom 1"] for g in G}
    for g in G:
        print(f"  {g}: E1(dimer) {res[g + ' dimer']:+.4f} = atoms {res[g + ' atom 0']:+.4f} + {res[g + ' atom 1']:+.4f} "
              f"+ non-additivity {D[g]:+.4f} eV", flush=True)
    d = D["tilted"] - D["upright"]
    e1 = res["tilted dimer"] - res["upright dimer"]
    print(f"  tilted - upright: E1(dimer) {e1:+.4f} (ALMO FRZ +1.402), atoms {e1 - d:+.4f}, non-additivity {d:+.4f} eV -> "
          f"{'(n1)' if d >= 0.4 else '(n2)' if abs(d) <= 0.2 else '(n3)'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 3.6)
