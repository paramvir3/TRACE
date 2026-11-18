"""The no-hop state of a bonded pair of s-class atoms from first principles (2026-11-13; for trace_redox SPEC 39 / the hop law).

TRACE's ensemble for two s-class atoms is a two-site Hubbard model (au2_hubbard.py): the covalent frame X0X0, the ionic
X+X- and X-X+ at the gap U, the hop t.  Its sigma sigma* triplet is the state in which no electron can hop, at the
frame's own energy; the singlet ground state lies below it by Delta_ST = (W - U) / 2, W = sqrt(U^2 + 16 t^2)
(ESTABLISHED: the two-electron two-orbital Hubbard / valence-bond mapping).  So the frame pair's interaction of the
"repulsion + hopping" decomposition is, at the RELAXED level, the triplet's interaction energy

    E_nohop(R) = E[X2, triplet](R) - 2 E[X]        (unrestricted PBE, counterpoise-corrected),

to be set against SPEC 39's frozen-density value Phi(R) (the high-spin Heitler-London product of two frozen atoms).

    python pair_no_hop_curve.py SYMBOL R [R ...]        (PBE, def2-TZVP with the def2 ECP, as au2_hubbard.py)

Printed per R: the ground singlet's interaction energy, the TDA-PBE sigma sigma* triplet and singlet excitations with U
and t, the VERTICAL no-hop level E_cov = E_gs + E_T, and t over Harrison's s-s form.

2026-11-13, later the same day: the unrestricted SCF triplet this tool first printed as "the relaxed no-hop state" is
a 5d-hole state for Au2 below 2.6 A (pair_triplet_character.py: d population 9.4 per atom, the 6s sigma_g pair
intact), and the level that belongs to the two-site model's U and t is the vertical one by identity.  The SCF triplet
is no longer computed here; its values are in runs/general_laws_2026-11-13/au2_no_hop_curve.log.
"""
import sys

import numpy as np
from pyscf import dft, gto, tdscf

import au2_hubbard as H

HARTREE = 27.211386245988


ECP = {"Ag": "def2-tzvp", "Au": "def2-tzvp"}                      # def2 ECPs exist from Rb on; lighter atoms are all-electron


def _mol(atoms, spin):
    symbols = {a[0].replace("ghost-", "") for a in atoms}
    ecp = {x: ECP[x] for x in symbols if x in ECP}
    return gto.M(atom=atoms, basis="def2-tzvp", ecp=ecp, unit="Angstrom", spin=spin, verbose=0)


def atom_energy(symbol, R):
    """One atom in the pair's basis (the partner a ghost), UKS doublet; also its singly occupied valence s orbital."""
    mol = _mol([[symbol, (0, 0, 0)], ["ghost-" + symbol, (0, 0, R)]], 1)
    mf = dft.UKS(mol)
    mf.xc = "pbe"
    mf.conv_tol = 1e-10
    mf.kernel()
    return float(mf.e_tot), bool(mf.converged)


def sigma_vectors(symbol, mol, mf, R):
    """au2_hubbard.sigma_vectors for any s-class atom: o, v = the occupied / virtual MO components of chi_A + chi_B and
    chi_A - chi_B, chi the atom's own singly occupied valence orbital (PBE, UKS doublet)."""
    atom = _mol([[symbol, (0, 0, 0)]], 1)
    ma = dft.UKS(atom); ma.xc = "pbe"; ma.conv_tol = 1e-10; ma.kernel()
    c = ma.mo_coeff[0][:, ma.mo_occ[0] > 0][:, -1]
    n1 = atom.nao_nr()
    chiA, chiB = np.zeros(mol.nao_nr()), np.zeros(mol.nao_nr())
    chiA[:n1], chiB[n1:] = c, c
    S = mol.intor_symmetric("int1e_ovlp")
    nocc = mol.nelectron // 2
    return mf.mo_coeff[:, :nocc].T @ S @ (chiA + chiB), mf.mo_coeff[:, nocc:].T @ S @ (chiA - chiB)


HBAR2_OVER_M, ETA_SS, R0 = 7.619964, 1.40, 0.5                    # eV A^2; Harrison's s-s coefficient; trace_redox's r0 (A)


def main():
    symbol = sys.argv[1]
    for R in (float(x) for x in sys.argv[2:]):
        mol = _mol([[symbol, (0, 0, 0)], [symbol, (0, 0, R)]], 0)
        ks = dft.RKS(mol)
        ks.xc = "pbe"
        ks.conv_tol = 1e-10
        ks.kernel()
        e_atom, ok = atom_energy(symbol, R)
        e_gs = (float(ks.e_tot) - 2.0 * e_atom) * HARTREE
        line = f"{symbol}2 R {R:.3f} A: ground singlet {e_gs:+.4f} eV (converged {bool(ks.converged)}/{ok})"
        td_s = tdscf.TDA(ks); td_s.nstates = 12; td_s.singlet = True; td_s.kernel()
        td_t = tdscf.TDA(ks); td_t.nstates = 12; td_t.singlet = False; td_t.kernel()
        o, v = sigma_vectors(symbol, mol, ks, R)
        ES, wS, _, _ = H.pick(td_s.e, [x[0] for x in td_s.xy], o, v)
        ET, wT, _, _ = H.pick(td_t.e, [x[0] for x in td_t.xy], o, v)
        U, t = H.hubbard(ET, ES)
        base = ETA_SS * HBAR2_OVER_M / (R * R + R0 * R0)
        line += (f"; TDA sigma sigma*: triplet {ET:.3f} (weight {wT:.2f}), singlet {ES:.3f} (weight {wS:.2f}) -> U {U:+.3f}, t {t:.3f} eV; "
                 f"no-hop level E_cov = E_gs + E_T {e_gs + ET:+.4f} eV; t / Harrison {t / base:.3f} (r0 = 0: {t * R * R / (ETA_SS * HBAR2_OVER_M):.3f})")
        print(line, flush=True)


if __name__ == "__main__":
    main()
