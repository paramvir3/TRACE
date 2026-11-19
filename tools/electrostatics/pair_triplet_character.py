"""Which triplet is the unrestricted SCF triplet of a pair of s-class atoms?  (2026-11-13; a check of
pair_no_hop_curve.py's "relaxed triplet", asked for by the design review of that day.)

The no-hop state of the two-site model is the sigma sigma* triplet: both 6s electrons unpaired, one in chi_A + chi_B and
one in chi_A - chi_B, the d shells full.  An aufbau triplet may instead be a d-hole state (a 5d -> sigma* excitation with
the 6s sigma_g pair intact), which still contains the bond.  Printed per R: the d population per atom (Mulliken), and
for the two singly occupied natural orbitals of the spin density their weights on the atomic s combinations
g = chi_A + chi_B and u = chi_A - chi_B (chi the atom's own valence s orbital, as in au2_hubbard.sigma_vectors).

    python pair_triplet_character.py SYMBOL R [R ...]         (PBE, def2-TZVP with the def2 ECP)
"""
import sys

import numpy as np
from pyscf import dft, gto

HARTREE = 27.211386245988


def main():
    symbol = sys.argv[1]
    atom = gto.M(atom=[[symbol, (0, 0, 0)]], basis="def2-tzvp", ecp="def2-tzvp", spin=1, verbose=0)
    ma = dft.UKS(atom); ma.xc = "pbe"; ma.conv_tol = 1e-10; ma.kernel()
    c_s = ma.mo_coeff[0][:, ma.mo_occ[0] > 0][:, -1]                  # the atom's singly occupied valence s orbital
    n1 = atom.nao_nr()
    for R in (float(x) for x in sys.argv[2:]):
        mol = gto.M(atom=[[symbol, (0, 0, 0)], [symbol, (0, 0, R)]], basis="def2-tzvp", ecp="def2-tzvp", unit="Angstrom", spin=2, verbose=0)
        mf = dft.UKS(mol); mf.xc = "pbe"; mf.conv_tol = 1e-10; mf.kernel()
        S = mol.intor_symmetric("int1e_ovlp")
        Pa, Pb = mf.make_rdm1()
        labels = mol.ao_labels(fmt=False)
        d_idx = [k for k, l in enumerate(labels) if l[2][-1] == "d" and l[0] == 0]
        pop = np.diag((Pa + Pb) @ S)
        d_pop = float(pop[d_idx].sum())
        # natural orbitals of the spin density in the Loewdin basis
        w, V = np.linalg.eigh(S)
        Sh, Smh = (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T
        occ, U = np.linalg.eigh(Sh @ (Pa - Pb) @ Sh)
        top = np.argsort(-occ)[:2]
        chiA, chiB = np.zeros(mol.nao_nr()), np.zeros(mol.nao_nr())
        chiA[:n1], chiB[n1:] = c_s, c_s
        out = []
        for k in top:
            c = Smh @ U[:, k]                                        # the natural orbital in the AO basis, normalised
            wg = (c @ S @ (chiA + chiB)) ** 2 / ((chiA + chiB) @ S @ (chiA + chiB))
            wu = (c @ S @ (chiA - chiB)) ** 2 / ((chiA - chiB) @ S @ (chiA - chiB))
            dw = float((c * (S @ c))[[i for i, l in enumerate(labels) if l[2][-1] == "d"]].sum())
            out.append((float(occ[k]), float(wg), float(wu), dw))
        stable = "n/a"
        try:
            _, _, stab_i, _ = mf.stability(return_status=True)
            stable = str(bool(stab_i))
        except Exception as error:
            stable = f"not run ({type(error).__name__})"
        print(f"{symbol}2 triplet, R {R:.2f} A: d population per atom {d_pop:.3f} (10 = full shell); singly occupied natural orbitals "
              + "; ".join(f"occ {o:.3f}: weight on s_g {g:.2f}, on s_u {u:.2f}, d character {d:.2f}" for o, g, u, d in out)
              + f"; SCF internally stable: {stable}; <S^2> {float(mf.spin_square()[0]):.3f}", flush=True)


if __name__ == "__main__":
    main()
