"""Where does the doped cell's extra electron sit in the embedded-cluster model?  (2026-10-26; a diagnostic.)

    OMP_NUM_THREADS=2 nice python tools/electrostatics/doped_spin.py plain R_AU   # bare point ions
    OMP_NUM_THREADS=2 nice python tools/electrostatics/doped_spin.py tip R_AU     # TIP-capped
    (python: an interpreter with PySCF installed)

Upright Au2- (doublet UKS PBE, def2-TZVP, Au def2 ECP) on the wetting_eda quantum cluster (quantum_sites(r_au)),
embedded in the doped point lattice (doped_lattice: one Al3+).  'plain' is the first check (a one-off script of
2026-10-26: spin on the cluster's Mg +1.090, on Au2 -0.102).  'tip' replaces every point cation within 3.5 A of a
quantum ion by a Mg2+ total-ion pseudopotential: the Stuttgart 10-electron-core ECP with no valence electrons
(ESTABLISHED: Winter, Pitzer & Klein 1987; Sushko, Shluger & Catlow 2000; Nasluzov et al. 2001).  PySCF refuses an
atom with no basis, so each TIP carries one tight s function (exponent 1e4 bohr^-2, kinetic energy 1.5e4 Ha), which no
electron can occupy; its population is checked.

Checks: (t1) the electron count equals the plain run's and the total charge (quantum nuclei - electrons + points) is
unchanged; (t2) the summed population of the TIP dummy functions is below 1e-4 e.  Pre-registered reading (theory note
2026-10-18 section 4c): hypothesis H1 (leakage toward the bare point cations) holds if the spin on Au2 exceeds 0.8 with
the TIPs, and is falsified if the spin on the cluster's Mg stays above 0.8.
"""
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import wetting_eda as W  # noqa: E402

R_TIP = 3.5
TIGHT = [[0, [1.0e4, 1.0]]]


def build(mode, r_au):
    from pyscf import gto
    G = W.geometries()
    sites = W.quantum_sites(G["flat"] + G["upright"], r_au=r_au)
    (coords, q), _ = W.doped_lattice(sites, "upright")
    surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
    qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf)
    A = [("Au", p) for p in G["upright"]]
    spec = [[s, tuple(p)] for s, p in A + surf]
    basis = {"Au": W.BASIS, "Mg": W.BASIS, "O": W.BASIS}
    ecp = {"Au": W.BASIS}
    charge = qB - 1
    tips = np.zeros(0, dtype=int)
    if mode == "tip":
        dist = np.linalg.norm(coords[:, None, :] - np.array([p for _, p in surf])[None], axis=2).min(1)
        tips = np.nonzero((dist < R_TIP) & (q > 0))[0]
        assert np.allclose(q[tips], 2.0), "only Mg2+ sites are capped (the Al3+ lies far below)"
        spec += [["Mg1", tuple(coords[j])] for j in tips]
        basis["Mg1"] = TIGHT
        ecp["Mg1"] = "stuttgart"
        charge += 2 * len(tips)                       # each TIP nucleus is Z_eff = 2 and brings no electron
        keep = np.ones(len(q), dtype=bool)
        keep[tips] = False
        coords, q = coords[keep], q[keep]
    mol = gto.M(atom=spec, basis=basis, ecp=ecp, charge=charge, spin=1, unit="Angstrom", verbose=0)
    return mol, (coords, q), A, surf, tips


def main(mode, r_au):
    from pyscf import dft
    mol, lat, A, surf, tips = build(mode, r_au)
    ref, ref_lat, _, _, _ = build("plain", r_au)
    total = float(mol.atom_charges().sum()) - mol.nelectron + float(lat[1].sum())
    ref_total = float(ref.atom_charges().sum()) - ref.nelectron + float(ref_lat[1].sum())
    ok1 = mol.nelectron == ref.nelectron and abs(total - ref_total) < 1e-9
    print(f"(t1) electrons {mol.nelectron} (plain {ref.nelectron}), total charge {total:+.6f} (plain {ref_total:+.6f}); "
          f"{len(tips)} TIPs -> {'PASS' if ok1 else 'FAIL'}", flush=True)
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build()
    mf, e = W._solve(W._uks(mol, grids, lat))
    dm = mf.make_rdm1()
    S = mol.intor_symmetric("int1e_ovlp")
    slices = mol.aoslice_by_atom()

    def pop(m, a):
        a0, a1 = slices[a][2], slices[a][3]
        return float(np.einsum("ij,ji->", m[:, a0:a1], S[a0:a1, :]))

    nq = len(A) + len(surf)
    spin = [pop(dm[0] - dm[1], a) for a in range(mol.natm)]
    chg = [pop(dm[0] + dm[1], a) for a in range(mol.natm)]
    dummy = sum(chg[nq:])
    if mode == "tip":
        print(f"(t2) population of the TIP dummy functions {dummy:.2e} e -> {'PASS' if abs(dummy) < 1e-4 else 'FAIL'}",
              flush=True)
    au_q = sum(mol.atom_charge(a) - chg[a] for a in range(len(A)))
    print(f"[{mode}] E {e:.10f} Ha; Au2 Mulliken charge {au_q:+.3f} e; spin on Au2 {sum(spin[:len(A)]):+.3f}, on the "
          f"cluster's Mg {sum(s for (n, _), s in zip(surf, spin[len(A):nq]) if n == 'Mg'):+.3f}, on its O "
          f"{sum(s for (n, _), s in zip(surf, spin[len(A):nq]) if n == 'O'):+.3f}, on the TIPs {sum(spin[nq:]):+.2e}",
          flush=True)
    for (n, p), s in zip(surf, spin[len(A):nq]):
        print(f"    {n:2s} at {np.round(p / W.D).astype(int)}: spin {s:+.3f}", flush=True)
    homo = mf.mo_energy[0][mf.mo_occ[0] > 0].max() * W.HARTREE
    print(f"alpha SOMO energy {homo:+.3f} eV", flush=True)
    s_au = sum(spin[:len(A)])
    s_mg = sum(s for (n, _), s in zip(surf, spin[len(A):nq]) if n == "Mg")
    if mode == "tip":
        verdict = "H1 holds" if s_au > 0.8 else ("H1 falsified" if s_mg > 0.8 else "neither threshold met")
        print(f"pre-registered reading: {verdict}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))
