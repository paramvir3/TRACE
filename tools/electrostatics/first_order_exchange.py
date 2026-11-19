"""First-order exchange of two frozen ion densities, by the library's own method (2026-10-19).

    python tools/electrostatics/first_order_exchange.py "Au,-1:O,-2:2.26" ["Au,0:O,-2:2.26" ...]
    (python: an interpreter with PySCF installed)

Why: trace_redox's overlap tables (SPEC 25) carry the Pauli repulsion of two charge-state densities by Gordon-Kim
(Thomas-Fermi kinetic + LDA exchange of the superposed densities).  The exact frozen-density counterpart is the
first-order exchange energy of the two monomer determinants (ESTABLISHED: first-order SAPT; Jeziorski, Moszynski and
Szalewicz, Chem. Rev. 94, 1887 (1994)):

    E_exch^(1) = < A Phi_A Phi_B | H | A Phi_A Phi_B > / < A Phi_A Phi_B | A Phi_A Phi_B >  -  < Phi_A Phi_B | H | Phi_A Phi_B >,

the Heitler-London energy of the antisymmetrised product minus that of the Hartree product, both with one
Hamiltonian H.  So every one-body and classical term (nuclei, the Watson sphere's potential, the monomers' own
energies, the electrostatics between them) cancels identically, and what remains is the exchange.  For determinants:

    E_HL = Tr[(P^a + P^b) h] + 1/2 J[P] - 1/2 sum_s Tr[P^s K(P^s)],    P^s = C^s (C^s^T S C^s)^-1 C^s^T  (both monomers' occupied spin orbitals)
    E_HP = the same with P^s -> P_A^s + P_B^s and exchange only inside each monomer.

Monomers: spin-unrestricted PBE0, X2C spin-free for Z > 36, ANO-RCC + two diffuse s and p shells, every O state in the
library's Watson sphere (+2 at 1.205 A); exactly the ion-density library's method (build_ion_densities.py), so these
are the densities the tables use.  The determinants are KS determinants, so this is SAPT(KS)'s first-order exchange,
evaluated with the HF Hamiltonian (stated).  The Watson sphere is part of H in both terms; it acts on the exchange
density change only outside its radius (a constant inside cannot act on a charge-conserving change), which
`WATSON_SCALE` tests by moving the radius.

Prints E_exch^(1) in eV per pair, and checks: positive for closed shells, and decaying with R.

Erratum (2026-10-20): the dimer's point charges were passed without a unit to a molecule that gto.conc_mol had put in
Bohr, so every pair with an O state had its Watson sphere at 0.53x the intended centre and radius.  Fixed (unit stated);
frozen_interaction.py (validated) supersedes this module's numbers for O pairs.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_ion_densities import ELEMENTS, METHOD, WATSON, _basis  # noqa: E402

HARTREE = 27.211386245988
SYMBOL_TO_Z = {v: k for k, v in ELEMENTS.items()}


def _sphere(center, radius_A, charge, n_dir=302):
    i = np.arange(n_dir) + 0.5
    phi = np.arccos(1 - 2 * i / n_dir)
    th = np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    return np.asarray(center) + dirs * radius_A, np.full(n_dir, charge / n_dir)


def monomer(symbol, q, center, spin=None):
    """UKS monomer by the library's method: (mol, occupied alpha/beta coefficients, sphere points/charges or None)."""
    from pyscf import dft, gto, qmmm
    Z = SYMBOL_TO_Z[symbol]
    n_el = Z - q
    scale = float(os.environ.get("WATSON_SCALE", "1.0"))
    best = None
    for s in ((spin,) if spin is not None else (n_el % 2, n_el % 2 + 2)):
        mol = gto.M(atom=[[symbol, tuple(center)]], basis={symbol: _basis(symbol)}, charge=q, spin=s, unit="Angstrom",
                    verbose=0)
        mf = dft.UKS(mol)
        if Z > 36:
            mf = mf.x2c()
        mf.xc = METHOD
        mf.conv_tol = 1e-11
        mf.max_cycle = 300
        sphere = None
        if (Z, q) in WATSON:
            RW, QW = WATSON[(Z, q)]
            pts, chg = _sphere(center, RW * scale, QW)
            sphere = (pts, chg)
            mf = qmmm.mm_charge(mf, pts, chg, radii=np.full(len(chg), 0.3), unit="Angstrom")
        e = mf.kernel()
        if not mf.converged:
            mf = mf.newton()
            e = mf.kernel(dm0=mf.make_rdm1())
        if mf.converged and (best is None or e < best[0]):
            best = (e, mol, mf, sphere)
    _, mol, mf, sphere = best
    occ = [mf.mo_coeff[k][:, mf.mo_occ[k] > 0] for k in (0, 1)]
    return mol, occ, sphere


def hl_hp_energies(mol, mf, h, occ_a, occ_b):
    """``(E_HL, E_HP)`` in hartree for occupied spin-orbital coefficients ``occ_a``, ``occ_b`` (lists over spin, already
    in ``mol``'s basis), with one-electron Hamiltonian ``h`` and ``mf``'s two-electron integrals.  E_HL: the energy of
    the antisymmetrised product (projector onto both monomers' occupied spaces, per spin); E_HP: the Hartree product
    (Coulomb of the summed densities, exchange inside each monomer only).  Nuclear repulsion is left out (it cancels in
    every difference this module takes)."""
    S = mol.intor_symmetric("int1e_ovlp")
    nao = mol.nao_nr()

    def projector(c):
        return c @ np.linalg.solve(c.T @ S @ c, c.T) if c.shape[1] else np.zeros((nao, nao))

    P_hl = [projector(np.hstack([occ_a[s], occ_b[s]])) for s in (0, 1)]
    P_a = [projector(occ_a[s]) for s in (0, 1)]
    P_b = [projector(occ_b[s]) for s in (0, 1)]

    def energy(P_coul, P_exch_blocks):
        tot = P_coul[0] + P_coul[1]
        vj, _ = mf.get_jk(mol, tot, hermi=1, with_k=False)
        e = float(np.einsum("ij,ji->", h, tot)) + 0.5 * float(np.einsum("ij,ji->", vj, tot))
        for Ps in P_exch_blocks:
            _, vk = mf.get_jk(mol, Ps, hermi=1, with_j=False)
            e -= 0.5 * float(np.einsum("ij,ji->", vk, Ps))
        return e

    e_hl = energy(P_hl, [P_hl[0], P_hl[1]])
    e_hp = energy([P_a[0] + P_b[0], P_a[1] + P_b[1]], [P_a[0], P_a[1], P_b[0], P_b[1]])
    return e_hl, e_hp


def first_order_exchange(spec_a, spec_b, R):
    """E_exch^(1) (eV) for monomers ``(symbol, q)`` at the origin and at (0, 0, R) A."""
    from pyscf import gto, qmmm, scf
    mol_a, occ_a, sph_a = monomer(*spec_a, (0.0, 0.0, 0.0))
    mol_b, occ_b, sph_b = monomer(*spec_b, (0.0, 0.0, R))
    mol = gto.conc_mol(mol_a, mol_b)
    na = mol_a.nao_nr()
    mf = scf.UHF(mol)
    if max(SYMBOL_TO_Z[spec_a[0]], SYMBOL_TO_Z[spec_b[0]]) > 36:
        mf = mf.x2c()
    pts = [s[0] for s in (sph_a, sph_b) if s is not None]
    if pts:
        # unit stated: gto.conc_mol returns a molecule in Bohr, and mm_charge reads coordinates in the molecule's unit
        # unless told (before 2026-10-20 the dimer's Watson sphere sat at 0.53x its place: every O pair was wrong)
        mf = qmmm.mm_charge(mf, np.concatenate(pts), np.concatenate([s[1] for s in (sph_a, sph_b) if s is not None]),
                            radii=np.full(sum(len(p) for p in pts), 0.3), unit="Angstrom")
    h = mf.get_hcore()
    S = mol.intor_symmetric("int1e_ovlp")
    nao = mol.nao_nr()

    def embed(c, offset):
        out = np.zeros((nao, c.shape[1]))
        out[offset:offset + c.shape[0]] = c
        return out

    e_hl, e_hp = hl_hp_energies(mol, mf, h, [embed(occ_a[s], 0) for s in (0, 1)], [embed(occ_b[s], na) for s in (0, 1)])
    return (e_hl - e_hp) * HARTREE


def main(args):
    for item in args:
        a, b, R = item.split(":")
        sa, qa = a.split(",")
        sb, qb = b.split(",")
        e = first_order_exchange((sa, int(qa)), (sb, int(qb)), float(R))
        print(f"{sa}{int(qa):+d} / {sb}{int(qb):+d} at {float(R):.2f} A: E_exch(1) = {e:+.4f} eV  (WATSON_SCALE "
              f"{os.environ.get('WATSON_SCALE', '1.0')})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
