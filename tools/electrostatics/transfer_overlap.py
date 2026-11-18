"""The in-crystal O 2p <-> Au 6s overlap against position: the first-principles form of SPEC 32's coupling (2026-10-28).

    OMP_NUM_THREADS=2 nice python tools/electrostatics/transfer_overlap.py OUT_DIR
    (python: an interpreter with PySCF installed)

Why (measured, benchmarks README): PBE's directional transfer at the flat site implies an O -> Au coupling of 0.26-0.28
of Harrison's 1/(r^2 + r0^2) at every height from 3.6 to 3.0 A, while the vertical O-top contact (2.26 A) implies
0.78: not a radial decay.  Hypothesis: the surface O 2p is compressed toward its Mg2+ neighbours (Pauli) and free
along the surface normal, which the ion library's free O2- in a Watson sphere (whose overlap agreed with 1/r^2 to
3.6 A) cannot show.

Method (ESTABLISHED: Wolfsberg-Helmholz t = K S (e_1 + e_2) / 2, J. Chem. Phys. 20, 837 (1952); natural orbitals of a
density matrix, Loewdin 1955):
1. the surface cluster B of wetting_eda (10 ions within 3.6 A of either dimer, PBE/def2-TZVP, in the point-ion
   lattice), solved once;
2. the cluster's O 2p band: its occupied Kohn-Sham orbitals phi_i above the O 2s and the Mg cores (the six highest of
   the ten-ion cluster), with their eigenvalues e_i and their Mulliken p_z / in-plane character on O(1,0,0);
3. the Au 6s of the ion library's free atom (PBE0, X2C, ANO-RCC; frozen_interaction._solve_monomer) placed at R, and
      S(R)^2 = sum_i <6s(R) | phi_i>^2 w_i,     w_i = the Mulliken 2p weight of phi_i on O(1,0,0),
   the overlap of the Au 6s with the O atom's share of the 2p band, against R: vertical above the O (the O-top
   contact; r = 2.0 .. 4.5 A), above the neighbouring Mg0 (the flat site; heights 2.1 .. 3.2 A), and a (r, theta) grid
   in the plane through O, Mg0 and the normal.  Reported: S(R) / S(vertical 2.26 A), the same ratio for the free O2-
   (Watson sphere; the library monomer), and Harrison's 1/(r^2 + r0^2) ratio.  Pre-registered reading: the anisotropy
   hypothesis holds if the in-crystal ratio at the flat reference contact (3.5 A, over Mg0 at h = 2.89) is below 0.5
   of the vertical ratio at the same r; it is falsified if the two agree within 20 %.

A first version (2026-10-28, kept as ``o2p_natural_orbitals``, legacy) took the O's 2p natural orbitals from the
Loewdin-orthogonalised density matrix; back-transformed through S^-1/2 in the overcomplete triple-zeta basis they
carry tails on the neighbouring atoms, and their vertical overlap was non-monotonic (0.085 at 3.0 A, 0.65 at 4.0 A):
an artifact, not read.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cluster_interaction as CI  # noqa: E402
import frozen_interaction as FI  # noqa: E402
import wetting_eda as W  # noqa: E402

R0 = 0.5
O_SITE = np.array([1.0, 0.0, 0.0]) * W.D
MG_SITE = np.zeros(3)


def cluster_B(chk=None):
    from pyscf import dft, lib
    G = W.geometries()
    sites = W.quantum_sites(G["flat"] + G["upright"], r_au=3.6)
    lattice = CI.rocksalt_lattice(W.D, 5, True, exclude=sites)
    surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
    qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf)
    mol = W._mol(surf, qB)
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build()
    mf = W._rks(mol, grids, lattice)
    if chk is not None:
        mf.chkfile = chk
        if os.path.exists(chk):
            mf.init_guess = "chkfile"
            mf.__dict__.update(lib.chkfile.load(chk, "scf"))
            mf.converged = True
    if chk is None or not getattr(mf, "converged", False):
        mf, e = W._solve(mf)
    i_o = [i for i, (s, p) in enumerate(surf) if np.linalg.norm(p - O_SITE) < 1e-6][0]
    return mol, mf, i_o


def o2p_band(mol, mf, i_o, n_band=6):
    """The cluster's O 2p band: the ``n_band`` highest occupied orbitals ``[nao, n_band]``, their eigenvalues (eV),
    the Mulliken 2p weight of each on atom ``i_o`` and that weight's p_z / in-plane split."""
    occ = mf.mo_occ > 0
    C = mf.mo_coeff[:, occ][:, -n_band:]
    eps = mf.mo_energy[occ][-n_band:] * W.HARTREE
    S = mol.intor_symmetric("int1e_ovlp")
    labels = mol.ao_labels(fmt=False)
    p_idx = [k for k, (ia, sym, nl, m) in enumerate(labels) if ia == i_o and "p" in nl]
    z_idx = [k for k in p_idx if labels[k][3].endswith("z")]
    SC = S @ C
    w_p = np.array([float(np.sum(C[p_idx, i] * SC[p_idx, i])) for i in range(n_band)])
    w_z = np.array([float(np.sum(C[z_idx, i] * SC[z_idx, i])) for i in range(n_band)])
    return C, eps, w_p, w_z


def o2p_natural_orbitals(mol, mf, i_o):
    """The in-crystal 2p natural orbitals of atom ``i_o``: [nao, 3] coefficients in the cluster's AO basis, their
    occupations, their Cartesian composition [3, 3] (x, y, z weight of each) and their orbital energies
    <p|F|p> (eV): the orbital-resolved hole energies."""
    from scipy.linalg import sqrtm
    S = mol.intor_symmetric("int1e_ovlp")
    D = mf.make_rdm1()
    Sh = np.real(sqrtm(S))
    Dl = Sh @ D @ Sh                                               # Loewdin populations
    labels = mol.ao_labels(fmt=False)
    idx = [k for k, (ia, sym, nl, m) in enumerate(labels) if ia == i_o and "p" in nl]    # every p shell of the O
    block = Dl[np.ix_(idx, idx)]
    occ, vec = np.linalg.eigh(block)
    top = np.argsort(occ)[-3:]
    C_l = np.zeros((mol.nao_nr(), 3))
    C_l[idx] = vec[:, top]
    C = np.linalg.solve(Sh, C_l)                                   # back to the AO basis
    cart = np.zeros((3, 3))
    for j, k in enumerate(idx):
        m = labels[k][3]
        ax = {"x": 0, "y": 1, "z": 2}[m[-1]]
        cart[:, ax] += vec[j, top] ** 2
    F = mf.get_fock()
    eps = np.array([float(C[:, m] @ F @ C[:, m] / (C[:, m] @ S @ C[:, m])) for m in range(3)]) * W.HARTREE
    return C, occ[top], cart, eps


def au_6s():
    occ, spin = FI._solve_monomer("Au", 0, 1.0)
    mol_au = FI._mol([("Au", np.zeros(3))], 0, spin)
    return mol_au, occ[0][:, -1]                                   # the alpha SOMO


def free_o2p():
    occ, spin = FI._solve_monomer("O", -2, 1.0)
    mol_o = FI._mol([("O", np.zeros(3))], -2, spin)
    return mol_o, occ[0][:, -3:]


def overlap_sq(mol_au, c_au, R, mol_b, C_b, w=None):
    from pyscf import gto
    m = FI._mol([("Au", R)], 0, 1)
    S = gto.intor_cross("int1e_ovlp", m, mol_b)
    s = c_au @ S @ C_b
    return float(np.sum(s * s)) if w is None else float(np.sum(s * s * w))


def main(out):
    os.makedirs(out, exist_ok=True)
    mol_b, mf, i_o = cluster_B(chk=os.path.join(out, "cluster_B.chk"))
    C_p, eps, w_p, w_z = o2p_band(mol_b, mf, i_o)
    print("the O 2p band of the cluster (the six highest occupied orbitals): eigenvalue, Mulliken 2p weight on "
          "O(1,0,0), of it p_z (the surface normal)", flush=True)
    for i in range(len(eps)):
        print(f"  orbital {i}: e {eps[i]:+.3f} eV, weight on O(1,0,0) {w_p[i]:.3f}, p_z share "
              f"{(w_z[i] / w_p[i] if w_p[i] > 1e-6 else 0.0):.2f}", flush=True)
    w = np.clip(w_p, 0.0, None)
    mol_au, c_au = au_6s()
    mol_o, c_o = free_o2p()
    ref = overlap_sq(mol_au, c_au, O_SITE + np.array([0, 0, 2.26]), mol_b, C_p, w)
    ref_free = overlap_sq(mol_au, c_au, np.array([0, 0, 2.26]), mol_o, c_o)
    rows = []
    print("vertical above the O (the O-top contact)")
    for r in (2.0, 2.26, 2.6, 3.0, 3.4, 3.5, 3.8, 4.0, 4.5):
        s2 = overlap_sq(mol_au, c_au, O_SITE + np.array([0, 0, r]), mol_b, C_p, w) / ref
        f2 = overlap_sq(mol_au, c_au, np.array([0, 0, r]), mol_o, c_o) / ref_free
        H = (2.26 ** 2 + R0 ** 2) / (r * r + R0 ** 2)
        rows.append({"site": "O-top", "r": r, "h": r, "crystal": s2 ** 0.5, "free": f2 ** 0.5, "harrison": H})
        print(f"  r {r:4.2f}: S/S0 crystal {s2 ** 0.5:.3f}, free ion {f2 ** 0.5:.3f}, Harrison {H:.3f}", flush=True)
    print("above the neighbouring Mg0 (the flat site; r = O-Au distance)")
    for h in (2.10, 2.37, 2.60, 2.89, 3.2):
        R = MG_SITE + np.array([0, 0, h])
        r = float(np.linalg.norm(R - O_SITE))
        s2 = overlap_sq(mol_au, c_au, R, mol_b, C_p, w) / ref
        f2 = overlap_sq(mol_au, c_au, np.array([-W.D, 0, h]), mol_o, c_o) / ref_free
        H = (2.26 ** 2 + R0 ** 2) / (r * r + R0 ** 2)
        rows.append({"site": "Mg-top", "r": r, "h": h, "crystal": s2 ** 0.5, "free": f2 ** 0.5, "harrison": H})
        print(f"  h {h:4.2f} (r {r:4.2f}): S/S0 crystal {s2 ** 0.5:.3f}, free ion {f2 ** 0.5:.3f}, Harrison {H:.3f}",
              flush=True)
    print("the (r, theta) grid in the O-Mg0-normal plane (theta from the normal toward Mg0)")
    for r in (2.6, 3.0, 3.5):
        for th in (0, 15, 30, 45, 60, 75, 90):
            t = np.radians(th)
            R = O_SITE + r * np.array([-np.sin(t), 0.0, np.cos(t)])
            s2 = overlap_sq(mol_au, c_au, R, mol_b, C_p, w) / ref
            f2 = overlap_sq(mol_au, c_au, r * np.array([-np.sin(t), 0.0, np.cos(t)]), mol_o, c_o) / ref_free
            rows.append({"site": "grid", "r": r, "theta": th, "crystal": s2 ** 0.5, "free": f2 ** 0.5})
            print(f"  r {r:3.1f} theta {th:2d}: crystal {s2 ** 0.5:.3f}, free {f2 ** 0.5:.3f}", flush=True)
    json.dump(rows, open(os.path.join(out, "transfer_overlap.json"), "w"), indent=1)
    vert = [x for x in rows if x["site"] == "O-top" and abs(x["r"] - 3.5) < 1e-6][0]["crystal"]
    flat = [x for x in rows if x["site"] == "Mg-top" and abs(x["h"] - 2.89) < 1e-6][0]["crystal"]
    verdict = "holds" if flat < 0.5 * vert else ("falsified" if abs(flat - vert) <= 0.2 * vert else "neither threshold")
    print(f"flat-site ratio {flat:.3f} against vertical {vert:.3f} at r = 3.5 A -> anisotropy hypothesis {verdict}",
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
