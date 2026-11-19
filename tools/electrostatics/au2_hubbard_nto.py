"""Artifact check on the Au2 Hubbard anchor (2026-11-01): is the low sigma-sigma* weight of the picked singlet roots
(0.36-0.67 against 0.76-0.92 for the triplets) state mixing, or a single excitation into a 6s-6p hybridised sigma_u
orbital (the ionic state's own polarisation)?  For each picked root: the natural transition orbitals (SVD of the
occupied x virtual amplitudes; ESTABLISHED: Martin, J. Chem. Phys. 118, 4775 (2003)), the leading NTO pair's share of
the excitation, and the Mulliken 6s / 6p / 5d composition of its hole and particle orbitals.  A single dominant pair
(share > 0.9) whose particle is a sigma_u s-p hybrid means the anchor's adiabatic root is the polarised ionic state
(the counterpart of the model's Delta_ion, which carries its bound-sector polarisation); a split share means mixing.
Usage: python au2_hubbard_nto.py [--ccsd] R [R ...]"""
import sys
import numpy as np
from pyscf import gto, scf, dft, cc, tdscf
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from au2_hubbard import sigma_vectors, pick, HARTREE  # noqa: E402


def composition(mol, S, c):
    """Mulliken shares by angular momentum (s, p, d, f; both atoms): the ECP basis's shell labels are not principal
    quantum numbers (the first version grouped by '6s' / '6p' labels and summed to ~0.3, 2026-11-01)."""
    ls = [mol.bas_angular(b) for b in range(mol.nbas) for _ in range((2 * mol.bas_angular(b) + 1) * mol.bas_nctr(b))]
    pop = c * (S @ c)
    pop = pop / pop.sum()
    return {"spdf"[l]: round(float(sum(pop[k] for k in range(len(ls)) if ls[k] == l)), 2) for l in range(4)}


def nto(mol, mf, r):
    nocc = mol.nelectron // 2
    U, s, Vt = np.linalg.svd(r, full_matrices=False)
    share = s ** 2 / (s ** 2).sum()
    S = mol.intor_symmetric("int1e_ovlp")
    hole = mf.mo_coeff[:, :nocc] @ U[:, 0]
    part = mf.mo_coeff[:, nocc:] @ Vt[0]
    return float(share[0]), composition(mol, S, hole), composition(mol, S, part)


args = [a for a in sys.argv[1:] if not a.startswith("--")]
for R in map(float, args):
    mol = gto.M(atom=[["Au", (0, 0, 0)], ["Au", (0, 0, R)]], basis="def2-tzvp", ecp="def2-tzvp", unit="Angstrom", verbose=0)
    ks = dft.RKS(mol); ks.xc = "pbe"; ks.conv_tol = 1e-10; ks.kernel()
    o, v = sigma_vectors(mol, ks, R)
    for singlet in (True, False):
        td = tdscf.TDA(ks); td.nstates = 12; td.singlet = singlet; td.kernel()
        xs = [x[0] for x in td.xy]
        E, w, i, _ = pick(td.e, xs, o, v)
        share, h, p = nto(mol, ks, xs[i])
        print(f"R {R:.2f} TDA-PBE {'singlet' if singlet else 'triplet'} root {i} {E:.3f} eV: sigma-sigma*(atomic 6s) weight "
              f"{w:.2f}; leading NTO pair share {share:.3f}; hole {h}; particle {p}", flush=True)
    if "--ccsd" in sys.argv:
        from pyscf.cc import eom_rccsd
        mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.kernel()
        mycc = cc.CCSD(mf); mycc.conv_tol = 1e-8; mycc.kernel()
        o2, v2 = sigma_vectors(mol, mf, R)
        for singlet in (True, False):
            e, c = (mycc.eomee_ccsd_singlet if singlet else mycc.eomee_ccsd_triplet)(nroots=10)
            eom = eom_rccsd.EOMEESinglet(mycc) if singlet else eom_rccsd.EOMEETriplet(mycc)
            r1 = [eom.vector_to_amplitudes(x)[0] for x in c]
            E, w, i, _ = pick(e, r1, o2, v2)
            norm1 = float((r1[i] ** 2).sum()) / float((c[i] ** 2).sum())
            share, h, p = nto(mol, mf, r1[i])
            print(f"R {R:.2f} EOM-CCSD {'singlet' if singlet else 'triplet'} root {i} {E:.3f} eV: weight {w:.2f}; r1 share of "
                  f"the vector {norm1:.3f}; leading NTO pair share (of r1) {share:.3f}; hole {h}; particle {p}", flush=True)
