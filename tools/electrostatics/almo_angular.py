"""The bond-oriented repulsion at the ALMO level (2026-11-04; theory note §4c, pre-registered 22:04).

    OMP_NUM_THREADS=4 nice python tools/electrostatics/almo_angular.py PATH.json OUT.json [H]
    (python: an interpreter with PySCF installed)

The dimer of frz_angular.py (Au-Au 2.545 A rotated about its contact Au held H = 2.26 A straight above the O-top O; the
6-ion cluster of almo_tilt.py) at theta = 0, 20, 40, 60, 70, 90 deg. Interaction energies with charge transfer
forbidden, monomers in their own basis on the dimer's grid:
  I_FRZ = E_FRZ - E_A - E_B,  I_ALMO = E_ALMO - E_A - E_B,
for the dimer (closed shell; almo_split.run_geometry, its checks a1-a3) and for each Au atom alone (a doublet;
almo_split_open's unrestricted ALMO: E_FRZ at zero rotation, E_ALMO its minimum). D_X(theta) = I_X(dimer) -
I_X(contact) - I_X(far), X = FRZ, ALMO; and the cos / P2 fit of each."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import almo_split as AS  # noqa: E402
import almo_split_open as AO  # noqa: E402
import cluster_interaction as CI  # noqa: E402
import wetting_eda as W  # noqa: E402

HARTREE = W.HARTREE
THETAS = [0, 20, 40, 60, 70, 90]


def atom_almo(p, surf, qB, lattice):
    """(I_FRZ, I_ALMO) in eV for one Au atom (doublet) at p over the cluster."""
    from pyscf import dft, gto
    A_atoms = [("Au", p)]

    def mol(atoms, charge, spin):
        spec = [[sym, tuple(x)] for sym, x in atoms]
        basis = {sym: W.BASIS for sym, _ in atoms}
        ecp = {"Au": W.BASIS} if any(sym == "Au" for sym, _ in atoms) else {}
        return gto.M(atom=spec, basis=basis, ecp=ecp, charge=charge, spin=spin, unit="Angstrom", verbose=0)

    dimer = mol(A_atoms + surf, qB, 1)
    grids = dft.gen_grid.Grids(dimer)
    grids.level = 3
    grids.build()
    molA, molB = mol(A_atoms, 0, 1), mol(surf, qB, 0)
    nA = molA.nao_nr()
    assert nA + molB.nao_nr() == dimer.nao_nr()
    mfA, eA = W._solve(W._uks(molA, AS._grid_like(molA, grids)))
    mfB, eB = W._solve(W._rks(molB, AS._grid_like(molB, grids), lattice))
    mfAB = W._uks(dimer, grids, lattice)                      # its Hamiltonian only
    de = AO.DirectionalEnergyU(AO.SpanEnergyU(mfAB), mfA, mfB, nA, {"A": ("A",), "B": ("B",)})
    e_frz, e_almo, res = AS.minimise_rot(de, label=" atom ALMO")
    assert res.success, "the atom's ALMO minimisation did not converge"
    return (e_frz - eA - eB) * HARTREE, (e_almo - eA - eB) * HARTREE


def main(path_json, out_json, h=2.26):
    P = json.load(open(path_json))
    tilted = [np.array(p) for p in P["cluster_au"]["1.0"]]
    upright = [np.array(p) for p in P["cluster_au"]["upright"]]
    sites = W.quantum_sites(tilted + upright, r_au=3.6)
    surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
    qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf)
    lattice = CI.rocksalt_lattice(W.D, 5, True, exclude=sites)
    contact = np.array([upright[0][0], upright[0][1], h])
    low = int(np.argmin([p[2] for p in tilted]))
    e = tilted[1 - low][:2] - tilted[low][:2]
    e = np.array([*(e / np.linalg.norm(e)), 0.0])
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}
    if "contact" not in res:
        res["contact"] = atom_almo(contact, surf, qB, lattice)
        json.dump(res, open(out_json, "w"), indent=1)
    print(f"contact atom: I_FRZ {res['contact'][0]:+.4f}, I_ALMO {res['contact'][1]:+.4f} eV", flush=True)
    for th in THETAS:
        far = contact + 2.545 * (np.sin(np.radians(th)) * e + np.cos(np.radians(th)) * np.array([0, 0, 1.0]))
        if f"far {th}" not in res:
            res[f"far {th}"] = atom_almo(far, surf, qB, lattice)
            json.dump(res, open(out_json, "w"), indent=1)
        if f"dimer {th}" not in res:
            o = AS.run_geometry(f"theta {th}", [contact, far], sites, lattice)
            res[f"dimer {th}"] = (o["FRZ"], o["FRZ"] + o["POL"])
            json.dump(res, open(out_json, "w"), indent=1)
        D = [res[f"dimer {th}"][k] - res["contact"][k] - res[f"far {th}"][k] for k in (0, 1)]
        res[f"D {th}"] = D
        print(f"  theta {th:3d}: dimer I_FRZ {res[f'dimer {th}'][0]:+.4f}, I_ALMO {res[f'dimer {th}'][1]:+.4f}; far atom "
              f"{res[f'far {th}'][0]:+.4f} / {res[f'far {th}'][1]:+.4f} -> D_FRZ {D[0]:+.4f}, D_ALMO {D[1]:+.4f} eV", flush=True)
    json.dump(res, open(out_json, "w"), indent=1)
    c = np.cos(np.radians(THETAS))
    A = np.stack([np.ones_like(c), c, 0.5 * (3 * c * c - 1)], 1)
    for k, name in ((0, "FRZ"), (1, "ALMO")):
        y = np.array([res[f"D {th}"][k] for th in THETAS])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        print(f"  D_{name} = {coef[0]:+.4f} + {coef[1]:+.4f} cos + {coef[2]:+.4f} P2 (residual {np.abs(A @ coef - y).max():.4f} eV); "
              f"D(70) - D(0) {y[4] - y[0]:+.4f} eV", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 2.26)
