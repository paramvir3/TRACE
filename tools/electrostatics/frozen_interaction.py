"""Frozen-density interaction of two charge-state ions, for any functional (2026-10-20).

    python tools/electrostatics/frozen_interaction.py "Mg,1:O,-2:2.106" ...
    (python: an interpreter with PySCF installed)

The overlap term of trace_redox's configuration energies (SPEC 23 penetration + SPEC 25 Pauli) is a frozen-density
interaction: two ions keep their isolated densities and only the interaction is added.  Its first-principles
definition for a functional xc (HF, PBE, PBE0) is the energy of the antisymmetrised product of the two monomer
determinants minus the monomers (ESTABLISHED: Heitler-London / first-order SAPT; the frozen-density "promolecule"
energy):

    E_int^(1)[xc] = E_xc^AB[P_HL] - E_xc^A[P_A] - E_xc^B[P_B],        P_HL = projector onto both occupied spaces, per spin
    E_elst^(1)    = E_HP - E_A - E_B           (classical: the Hartree product, exchange only inside each monomer)
    E_nonelst[xc] = E_int^(1)[xc] - E_elst^(1)  (kinetic orthogonality cost + the functional's exchange-correlation)

With xc = HF, E_nonelst is the first-order exchange E_exch^(1).  For PBE it is what the data's functional assigns to
the overlap beyond electrostatics, which is the quantity Gordon-Kim's electron-gas term approximates.

Erratum (2026-10-20): for pairs with Au (X2C) the monomer of the light partner was given a non-relativistic
Hamiltonian, so `elst` and `int_*` of frozen_interaction carry that partner's own scalar-relativistic one-electron
energy (hypothesis, tested by decomposed_interaction's 'vac_elst_h_mixed').  exch_HL and nonelst_* use the dimer's
Hamiltonian only and are unaffected; decomposed_interaction's 'vac_elst' is the Coulomb energy of the two charge
distributions (SPEC 23's definition).

Monomers are the ion-density library's (build_ion_densities.py): UKS PBE0, X2C for Z > 36, ANO-RCC + 2 diffuse s, p;
O states in the Watson sphere.  All energies use one convention: A's and B's Hamiltonians are built in the union basis
with the partner as ghost functions (the same X2C and sphere treatment), so every classical and one-body term cancels
in the differences.  Self-check: with xc = 'hf', E_int^(1) equals the Heitler-London E^(1) of first_order_exchange.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_ion_densities import METHOD, WATSON, _basis  # noqa: E402
from first_order_exchange import SYMBOL_TO_Z, _sphere, hl_hp_energies  # noqa: E402

HARTREE = 27.211386245988


def _mol(atoms, charge, spin):
    from pyscf import gto
    basis = {}
    for sym, _ in atoms:
        basis[sym] = _basis(sym.replace("ghost-", ""))
    return gto.M(atom=[[s, tuple(p)] for s, p in atoms], basis=basis, charge=charge, spin=spin, unit="Angstrom",
                 verbose=0)


def _decorate(mf, Zs, spheres):
    from pyscf import qmmm
    if max(Zs) > 36:
        mf = mf.x2c()
    pts = [s for s in spheres if s is not None]
    if pts:
        coords = np.concatenate([p for p, _ in pts])
        mf = qmmm.mm_charge(mf, coords, np.concatenate([c for _, c in pts]),
                            radii=np.full(sum(len(c) for _, c in pts), 0.3), unit="Angstrom")
        # every sphere must be centred on an O nucleus of this molecule (Angstrom): a unit slip moves it
        nuclei = mf.mol.atom_coords(unit="Angstrom")
        for p, _ in pts:
            centre = p.mean(axis=0)
            assert np.min(np.linalg.norm(nuclei - centre, axis=1)) < 1e-3, f"Watson sphere off its nucleus: {centre}"
    return mf


_SOLVED = {}


def _monomer(symbol, q, center, partner, partner_center):
    """The library's monomer, solved in its own basis; returns (occupied coefficients in the monomer's own basis,
    spin, sphere at ``center``).  The basis and the sphere move with the atom, so the coefficients do not depend on
    ``center``: each (symbol, q, Watson scale) is solved once at the origin and reused, which also keeps one
    orientation of an open-shell state (O-) for every bond axis of an orientation average."""
    Z = SYMBOL_TO_Z[symbol]
    scale = float(os.environ.get("WATSON_SCALE", "1.0"))
    sphere = None
    if (Z, q) in WATSON:
        RW, QW = WATSON[(Z, q)]
        sphere = _sphere(center, RW * scale, QW)
    key = (symbol, q, scale)
    if key not in _SOLVED:
        _SOLVED[key] = _solve_monomer(symbol, q, scale)
    occ, spin = _SOLVED[key]
    return occ, spin, sphere


def _solve_monomer(symbol, q, scale):
    from pyscf import dft
    Z = SYMBOL_TO_Z[symbol]
    n_el = Z - q
    center = np.zeros(3)
    sphere = None
    if (Z, q) in WATSON:
        RW, QW = WATSON[(Z, q)]
        sphere = _sphere(center, RW * scale, QW)
    best = None
    for s in (n_el % 2, n_el % 2 + 2):
        if s > n_el:
            continue
        mol = _mol([(symbol, center)], q, s)
        mf = _decorate(dft.UKS(mol), [Z], [sphere])
        mf.xc = METHOD
        mf.conv_tol = 1e-11
        mf.max_cycle = 300
        e = mf.kernel()
        if not mf.converged:
            mf = mf.newton()
            e = mf.kernel(dm0=mf.make_rdm1())
        if mf.converged and (best is None or e < best[0]):
            best = (e, s, mf)
    _, spin, mf = best
    occ = [mf.mo_coeff[k][:, mf.mo_occ[k] > 0] for k in (0, 1)]
    return occ, spin


def frozen_interaction(spec_a, spec_b, R, axis=(0.0, 0.0, 1.0), xcs=("hf", "pbe", "pbe0")):
    """eV: {'elst', 'exch_HL', 'nonelst_<xc>'...} for monomers (symbol, q) at the origin and at R * axis (A)."""
    from pyscf import dft, scf
    ca, cb = np.zeros(3), np.asarray(axis, float) / np.linalg.norm(axis) * R
    (sa, qa), (sb, qb) = spec_a, spec_b
    Za, Zb = SYMBOL_TO_Z[sa], SYMBOL_TO_Z[sb]
    occ_a, spin_a, sph_a = _monomer(sa, qa, ca, sb, cb)
    occ_b, spin_b, sph_b = _monomer(sb, qb, cb, sa, ca)
    dimer = _mol([(sa, ca), (sb, cb)], qa + qb, spin_a + spin_b)
    mono_a = _mol([(sa, ca), ("ghost-" + sb, cb)], qa, spin_a)
    mono_b = _mol([("ghost-" + sa, ca), (sb, cb)], qb, spin_b)
    na = mono_a.nao_nr() - _mol([(sb, cb)], qb, spin_b).nao_nr()
    nao = dimer.nao_nr()

    def embed(c, offset):
        out = np.zeros((nao, c.shape[1]))
        out[offset:offset + c.shape[0]] = c
        return out

    A = [embed(occ_a[s], 0) for s in (0, 1)]
    B = [embed(occ_b[s], na) for s in (0, 1)]
    S = dimer.intor_symmetric("int1e_ovlp")

    def proj(c):
        return c @ np.linalg.solve(c.T @ S @ c, c.T) if c.shape[1] else np.zeros((nao, nao))

    P_hl = np.array([proj(np.hstack([A[s], B[s]])) for s in (0, 1)])
    P_a = np.array([proj(A[s]) for s in (0, 1)])
    P_b = np.array([proj(B[s]) for s in (0, 1)])
    # HF-expression energies with each Hamiltonian (one convention; nuclear terms from each mol + its spheres)
    mf_ab = _decorate(scf.UHF(dimer), [Za, Zb], [sph_a, sph_b])
    mf_a = _decorate(scf.UHF(mono_a), [Za], [sph_a])
    mf_b = _decorate(scf.UHF(mono_b), [Zb], [sph_b])
    e_hl, e_hp = hl_hp_energies(dimer, mf_ab, mf_ab.get_hcore(), A, B)
    e_hl, e_hp = e_hl + mf_ab.energy_nuc(), e_hp + mf_ab.energy_nuc()
    e_a = mf_a.energy_tot(dm=P_a)
    e_b = mf_b.energy_tot(dm=P_b)
    out = {"elst": (e_hp - e_a - e_b) * HARTREE, "exch_HL": (e_hl - e_hp) * HARTREE,
           "int_HL": (e_hl - e_a - e_b) * HARTREE}
    for xc in xcs:
        def ks(mol, Zs, spheres):
            mf = _decorate(dft.UKS(mol), Zs, spheres)
            mf.xc = xc
            return mf
        e_int = (ks(dimer, [Za, Zb], [sph_a, sph_b]).energy_tot(dm=P_hl)
                 - ks(mono_a, [Za], [sph_a]).energy_tot(dm=P_a) - ks(mono_b, [Zb], [sph_b]).energy_tot(dm=P_b))
        out[f"int_{xc}"] = e_int * HARTREE
        out[f"nonelst_{xc}"] = e_int * HARTREE - out["elst"]
    return out


def decomposed_interaction(spec_a, spec_b, R, axis=(0.0, 0.0, 1.0), xcs=("hf", "pbe", "pbe0"), grid_level=5):
    """eV.  The frozen-density interaction of :func:`frozen_interaction` split into exact pieces, in two pair
    Hamiltonians: 'vac' (the two ions alone) and 'sph' (each O state's Watson sphere also in the pair Hamiltonian, as
    :func:`frozen_interaction`).  The orbitals are the library's in both.  With dP = P_HL - P_A - P_B (spin summed;
    it holds zero electrons), for convention c and functional xc (exact-exchange fraction a_xc):

        elst         = Coulomb energy of the two frozen charge distributions ('vac_elst'; SPEC 23's definition);
        elst_h_c     = E_HP - E_A - E_B from the Hamiltonians (differs from vac_elst by the X2C non-additivity only,
                       for 'vac'; includes the spheres' fields for 'sph')
        onebody_c    = Tr[dP h_c]
        coul         = (dP | rho_A + rho_B) + 1/2 (dP | dP)
        exx_xc       = -(a_xc / 2) sum_s { Tr[P_HL^s K P_HL^s] - Tr[P_A^s K P_A^s] - Tr[P_B^s K P_B^s] }
        dfa_xc       = E_dfa[rho_HL] - E_dfa[rho_A] - E_dfa[rho_B]          semilocal part, all on the dimer's grid
        nonelst_c_xc = onebody_c + coul + exx_xc + dfa_xc                   (= E_int - elst, exactly)

    Diagnostics: kin = Tr[dP T] (non-relativistic T; for Z > 36 the X2C picture change sits in onebody - kin),
    dfa_sum_xc = E_dfa[rho_A + rho_B] - E_dfa[rho_A] - E_dfa[rho_B] (the functional's own non-additive xc of the
    superposed densities: what Gordon-Kim's LDA exchange + Wigner correlation approximate), number = Tr[dP S]."""
    from pyscf import dft, scf
    ca, cb = np.zeros(3), np.asarray(axis, float) / np.linalg.norm(axis) * R
    (sa, qa), (sb, qb) = spec_a, spec_b
    Za, Zb = SYMBOL_TO_Z[sa], SYMBOL_TO_Z[sb]
    occ_a, spin_a, sph_a = _monomer(sa, qa, ca, sb, cb)
    occ_b, spin_b, sph_b = _monomer(sb, qb, cb, sa, ca)
    dimer = _mol([(sa, ca), (sb, cb)], qa + qb, spin_a + spin_b)
    mono_a = _mol([(sa, ca), ("ghost-" + sb, cb)], qa, spin_a)
    mono_b = _mol([("ghost-" + sa, ca), (sb, cb)], qb, spin_b)
    na = mono_a.nao_nr() - _mol([(sb, cb)], qb, spin_b).nao_nr()
    nao = dimer.nao_nr()

    def embed(c, offset):
        out = np.zeros((nao, c.shape[1]))
        out[offset:offset + c.shape[0]] = c
        return out

    A = [embed(occ_a[s], 0) for s in (0, 1)]
    B = [embed(occ_b[s], na) for s in (0, 1)]
    S = dimer.intor_symmetric("int1e_ovlp")

    def proj(c):
        return c @ np.linalg.solve(c.T @ S @ c, c.T) if c.shape[1] else np.zeros((nao, nao))

    P_hl = np.array([proj(np.hstack([A[s], B[s]])) for s in (0, 1)])
    P_a = np.array([proj(A[s]) for s in (0, 1)])
    P_b = np.array([proj(B[s]) for s in (0, 1)])
    dP = (P_hl - P_a - P_b).sum(0)
    tA, tB = P_a.sum(0), P_b.sum(0)
    jk = scf.UHF(dimer)

    def vj(D):
        return jk.get_jk(dimer, D, hermi=1, with_k=False)[0]

    def kx(P):
        return sum(float(np.einsum("ij,ji->", jk.get_jk(dimer, P[s], hermi=1, with_j=False)[1], P[s])) for s in (0, 1))

    out = {"number": float(np.einsum("ij,ji->", dP, S)),
           "kin": float(np.einsum("ij,ji->", dP, dimer.intor_symmetric("int1e_kin"))) * HARTREE}
    out["coul"] = (float(np.einsum("ij,ji->", vj(tA + tB), dP)) + 0.5 * float(np.einsum("ij,ji->", vj(dP), dP))) * HARTREE
    j_ab = float(np.einsum("ij,ji->", vj(tA), tB))
    # elst, three ways.  'vac_elst': the Coulomb energy of the two charge distributions (nuclei + frozen densities;
    # 1/|r - R| operators), which is SPEC 23's definition.  '<conv>_elst_h': from the Hamiltonians, h - h_a, with the
    # monomers' Hamiltonians relativistic exactly when the dimer's is (X2C for max Z > 36).  'vac_elst_h_mixed': the
    # earlier form, a monomer's own Hamiltonian relativistic only for its own Z > 36 (falsifying check of the
    # 2026-10-20 Au-O discrepancy).
    R_bohr = np.linalg.norm(cb - ca) / 0.529177210903
    with dimer.with_rinv_origin(dimer.atom_coord(1)):
        V_b = -Zb * dimer.intor_symmetric("int1e_rinv")
    with dimer.with_rinv_origin(dimer.atom_coord(0)):
        V_a = -Za * dimer.intor_symmetric("int1e_rinv")
    out["vac_elst"] = (float(np.einsum("ij,ji->", tA, V_b)) + float(np.einsum("ij,ji->", tB, V_a)) + j_ab
                       + Za * Zb / R_bohr) * HARTREE
    for conv, (ha_s, hb_s) in (("vac", (None, None)), ("sph", (sph_a, sph_b))):
        m_ab = _decorate(scf.UHF(dimer), [Za, Zb], [ha_s, hb_s])
        m_a = _decorate(scf.UHF(mono_a), [Za, Zb], [ha_s])
        m_b = _decorate(scf.UHF(mono_b), [Za, Zb], [hb_s])
        h, h_a, h_b = m_ab.get_hcore(), m_a.get_hcore(), m_b.get_hcore()
        out[f"{conv}_elst_h"] = (float(np.einsum("ij,ji->", tA, h - h_a)) + float(np.einsum("ij,ji->", tB, h - h_b))
                                 + j_ab + m_ab.energy_nuc() - m_a.energy_nuc() - m_b.energy_nuc()) * HARTREE
        out[f"{conv}_onebody"] = float(np.einsum("ij,ji->", dP, h)) * HARTREE
        if conv == "vac":
            h_a_own = _decorate(scf.UHF(mono_a), [Za], [None]).get_hcore()
            h_b_own = _decorate(scf.UHF(mono_b), [Zb], [None]).get_hcore()
            out["vac_elst_h_mixed"] = (float(np.einsum("ij,ji->", tA, h - h_a_own))
                                       + float(np.einsum("ij,ji->", tB, h - h_b_own)) + j_ab + Za * Zb / R_bohr) * HARTREE
    out["sph_elst"] = out["sph_elst_h"]
    grids = dft.gen_grid.Grids(dimer)
    grids.level = grid_level
    grids.build()
    ni = dft.numint.NumInt()
    k_hl, k_a, k_b = kx(P_hl), kx(P_a), kx(P_b)
    for xc in xcs:
        a = 1.0 if xc == "hf" else ni.hybrid_coeff(xc)
        if xc != "hf":
            omega, _, _ = ni.rsh_coeff(xc)
            assert omega == 0, "range-separated functionals are not handled"

        def dfa(P):
            return 0.0 if xc == "hf" else float(ni.nr_uks(dimer, grids, xc, P)[1])

        e_a, e_b = dfa(P_a), dfa(P_b)
        out[f"exx_{xc}"] = -0.5 * a * (k_hl - k_a - k_b) * HARTREE
        out[f"dfa_{xc}"] = (dfa(P_hl) - e_a - e_b) * HARTREE
        out[f"dfa_sum_{xc}"] = (dfa(P_a + P_b) - e_a - e_b) * HARTREE
        for conv in ("vac", "sph"):
            out[f"{conv}_nonelst_{xc}"] = out[f"{conv}_onebody"] + out["coul"] + out[f"exx_{xc}"] + out[f"dfa_{xc}"]
    return out


def main(args):
    for item in args:
        a, b, R = item.split(":")
        sa, qa = a.split(",")
        sb, qb = b.split(",")
        axes = [(0, 0, 1)] if os.environ.get("ORIENT_AVERAGE") != "1" else [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        rows = [frozen_interaction((sa, int(qa)), (sb, int(qb)), float(R), axis=ax) for ax in axes]
        res = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        spread = {k: float(np.ptp([r[k] for r in rows])) for k in rows[0]}
        print(f"{sa}{int(qa):+d} / {sb}{int(qb):+d} at {float(R):.3f} A ({len(axes)} orientation(s)): "
              + "  ".join(f"{k} {v:+.4f}" + (f" (spread {spread[k]:.4f})" if len(axes) > 1 else "")
                          for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
