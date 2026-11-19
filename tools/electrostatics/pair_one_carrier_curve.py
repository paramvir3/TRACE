"""The one-carrier pair of two s-class atoms from first principles (2026-11-14; for trace_redox SPEC 40.6 iii).

For X2^q, q = +1 or -1, the two-site model has the configurations a^q b0 and a0 b^q at one level W_q(R), coupled by t:
the doublets lie at W_q -+ t (ESTABLISHED: the one- and three-electron resonance, as for H2+).  With the pair's two
resonance states, the carrier in sigma_g and in sigma_u,

    E_int^q(R) = E[X2^q, ground] - E[X^q] - E[X]     (counterpoise: each fragment in the pair's basis)
    dE(R)      = |E[sigma_u state] - E[sigma_g state]|
    W_q(R)     = E_int^q + dE / 2                     the no-hop level of the one-carrier pair
    t_q(R)     = dE / 2

    python pair_one_carrier_curve.py SYMBOL CHARGE XC R [R ...] [basis=NAME]
        XC: pbe, pbe0, hf, ...;  basis: def2-tzvp for the cation, def2-tzvpd for the anion unless given

Each state is a self-consistent unrestricted state with its electrons counted per symmetry species (D2h, the bond
on z: sigma_g in Ag, sigma_u in B1u), starting from the neutral dimer's closed-shell count:
    cation: sigma_g state  Ag (n, n-1);            sigma_u state  Ag (n-1, n-1), B1u (m+1, m)
    anion:  sigma_u state  B1u (m+1, m) [ground];  sigma_g state  Ag (n, n-1),  B1u (m+1, m+1)
(first tried with TDA on the ground doublet; for a one-electron system PySCF's HF diagonalises h and TDA on it is
not valid, so that route had no exact test; README, 2026-11-14).  Ag also holds the d_z2 and d_x2-y2 combinations,
so each state is checked by its spin population on the s combination it should carry: sigma_g and sigma_u are the
sum and difference of the atoms' own singly occupied valence s orbitals.
Printed per R: E_int, <S^2> and the spin populations of both states, the d population per atom, dE, W_q, t_q; the
neutral dimer in the same basis and the vertical EA or IP, as computed and "on the exact atomic values"; VALID or
FLAGGED by the rules of the README's pre-registration.
"""
import sys

import numpy as np
from pyscf import dft, gto, scf

HARTREE = 27.211386245988
ECP = {"Ag": "def2-tzvp", "Au": "def2-tzvp"}
EXACT = {"H": (0.754195, 13.598434), "Ag": (1.30447, 7.576234), "Au": (2.30861, 9.225554)}     # atomic EA, IP (eV)


def _mol(atoms, charge, spin, basis, symmetry=False):
    symbols = {a[0].replace("ghost-", "") for a in atoms}
    ecp = {x: ECP[x] for x in symbols if x in ECP}
    return gto.M(atom=atoms, basis=basis, ecp=ecp, unit="Angstrom", charge=charge, spin=spin, verbose=0,
                 symmetry="D2h" if symmetry else False)


def _solve(mol, xc, irrep_nelec=None):
    """Unrestricted SCF (symmetry-adapted when the molecule carries a point group), with a second-order fallback."""
    if mol.nelectron == 0:
        return None
    if mol.symmetry:
        mf = scf.uhf_symm.UHF(mol) if xc == "hf" else dft.uks_symm.UKS(mol)
    else:
        mf = scf.uhf.UHF(mol) if xc == "hf" else dft.uks.UKS(mol)
    if xc != "hf":
        mf.xc = xc
        mf.grids.level = 4
    if irrep_nelec is not None:
        mf.irrep_nelec = irrep_nelec
    mf.conv_tol = 1e-10
    mf.max_cycle = 300
    mf.kernel()
    if not mf.converged:
        mf.level_shift = 0.3
        mf.kernel(mf.make_rdm1())
        mf.level_shift = 0.0
        mf.kernel(mf.make_rdm1())
    return mf


def _energy(mf):
    return 0.0 if mf is None else float(mf.e_tot)


def _ok(mf):
    return True if mf is None else bool(mf.converged)


def run(symbol, q, xc, R, basis):
    pair = [[symbol, (0, 0, -R / 2)], [symbol, (0, 0, R / 2)]]
    ghost = [[symbol, (0, 0, -R / 2)], ["ghost-" + symbol, (0, 0, R / 2)]]
    atom = _solve(_mol(ghost, 0, 1, basis), xc)
    ion = _solve(_mol(ghost, q, 0, basis), xc)
    neutral = _solve(_mol(pair, 0, 0, basis, symmetry=True), xc)
    base = neutral.get_irrep_nelec()
    n, m = base.get("Ag", (0, 0))[0], base.get("B1u", (0, 0))[0]
    if q > 0:
        counts = {"g": {"Ag": (n, n - 1)}, "u": {"Ag": (n - 1, n - 1), "B1u": (m + 1, m)}}
    else:
        counts = {"u": {"B1u": (m + 1, m)}, "g": {"Ag": (n, n - 1), "B1u": (m + 1, m + 1)}}
    mol = _mol(pair, q, 1, basis, symmetry=True)
    states = {k: _solve(mol, xc, dict(base, **v)) for k, v in counts.items()}
    # the atoms' own singly occupied valence s orbital on each centre (the atom alone in its own basis)
    own = _solve(_mol([[symbol, (0, 0, 0)]], 0, 1, basis), xc)
    c = own.mo_coeff[0][:, own.mo_occ[0] > 0][:, -1]
    S = mol.intor_symmetric("int1e_ovlp")
    n1 = mol.nao_nr() // 2
    chiA, chiB = np.zeros(mol.nao_nr()), np.zeros(mol.nao_nr())
    chiA[:n1], chiB[n1:] = c, c
    if float(chiA @ S @ chiB) < 0:
        chiB = -chiB
    plus, minus = chiA + chiB, chiA - chiB
    plus, minus = plus / np.sqrt(plus @ S @ plus), minus / np.sqrt(minus @ S @ minus)
    labels = mol.ao_labels(fmt=False)

    def character(mf):
        Ca, Cb = mf.mo_coeff
        oa, ob = mf.mo_occ[0] > 0, mf.mo_occ[1] > 0
        pop = lambda C, occ, x: float(((C[:, occ].T @ S @ x) ** 2).sum())
        dm = mf.make_rdm1()
        P = (dm[0] + dm[1]) @ S
        d = sum(float(P[k, k]) for k, lab in enumerate(labels) if lab[0] == 0 and "d" in lab[2]) if symbol != "H" else 0.0
        return pop(Ca, oa, plus) - pop(Cb, ob, plus), pop(Ca, oa, minus) - pop(Cb, ob, minus), float(mf.spin_square()[0]), d

    ch = {k: character(mf) for k, mf in states.items()}
    ground = min(states, key=lambda k: _energy(states[k]))
    e_int = (_energy(states[ground]) - _energy(ion) - _energy(atom)) * HARTREE
    e_int0 = (_energy(neutral) - 2.0 * _energy(atom)) * HARTREE
    dE = abs(_energy(states["u"]) - _energy(states["g"])) * HARTREE
    expected_ground = "g" if q > 0 else "u"
    valid = bool(all(_ok(x) for x in (atom, ion, neutral, states["g"], states["u"]))
                 and ground == expected_ground
                 and all(abs(ch[k][2] - 0.75) <= 0.02 for k in ch)
                 and ch["g"][0] >= 0.5 and ch["u"][1] >= 0.5)
    ea_exp, ip_exp = EXACT[symbol]
    if q < 0:
        dimer_calc = (_energy(neutral) - _energy(states[ground])) * HARTREE
        atom_calc = (_energy(atom) - _energy(ion)) * HARTREE
        vertical = f"EA(X2) {dimer_calc:+.3f}, atom {atom_calc:+.3f}, on the exact atom {ea_exp + dimer_calc - atom_calc:+.3f}"
    else:
        dimer_calc = (_energy(states[ground]) - _energy(neutral)) * HARTREE
        atom_calc = (_energy(ion) - _energy(atom)) * HARTREE
        vertical = f"IP(X2) {dimer_calc:+.3f}, atom {atom_calc:+.3f}, on the exact atom {ip_exp + dimer_calc - atom_calc:+.3f}"
    show = lambda k: f"sigma_{k} state: <S^2> {ch[k][2]:.3f}, spin on sigma_g {ch[k][0]:+.2f} / sigma_u {ch[k][1]:+.2f}, d per atom {ch[k][3]:.2f}, converged {_ok(states[k])}"
    print(f"{symbol}2 q {q:+d} {xc} {basis} R {R:.4f} A: E_int {e_int:+.4f} eV (ground: sigma_{ground}); {show('g')}; {show('u')}; "
          f"dE {dE:.4f} -> W {e_int + dE / 2:+.4f}, t {dE / 2:.4f} eV; neutral E_int {e_int0:+.4f}; {vertical}; "
          f"{'VALID' if valid else 'FLAGGED'}", flush=True)


def main():
    symbol, q, xc = sys.argv[1], int(sys.argv[2]), sys.argv[3].lower()
    opts = dict(a.split("=", 1) for a in sys.argv[4:] if "=" in a)
    basis = opts.get("basis", "def2-tzvpd" if q < 0 else "def2-tzvp")
    for R in (float(x) for x in sys.argv[4:] if "=" not in x):
        run(symbol, q, xc, R, basis)


if __name__ == "__main__":
    main()
