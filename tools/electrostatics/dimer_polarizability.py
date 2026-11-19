"""Static polarisability tensor of a homonuclear dimer by the ion-density library's method (2026-10-18).

    python tools/electrostatics/dimer_polarizability.py SYMBOL D_ANGSTROM [OUT.json]
    (python: an interpreter with PySCF installed)
    (D <= 0: the atom, same method, for the dimer / atom ratio)

Why: the coherent charge-transfer response of a metal dimer (the ionic-covalent mixing: Delta_ion and tau) sets its
parallel polarisability; unlike a charge partition (Hirshfeld), alpha is an observable.  Method: closed-shell PBE0
(UKS, singlet), X2C spin-free for Z > 36, ANO-RCC plus two diffuse s, p shells (build_ion_densities._basis), a
uniform field F along the bond (z) and across it (x) in the one-electron Hamiltonian; alpha = d mu / dF by central
differences at F and F / 2 (DIMER_FIELDS, default 2e-3 and 1e-3 au), Richardson-extrapolated.  bohr^3 and A^3.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_ion_densities import METHOD, _basis  # noqa: E402

BOHR3_TO_A3 = 0.529177210903 ** 3


#: DIMER_BASIS=library (X2C, ANO-RCC + diffuse: the ion-density library's, ~hours per Au2 SCF), def2-tzvppd
#: (scalar-relativistic def2 ECP for Z > 36), or def2-tzvppd+2d (that plus two diffuse s, p, d shells: needed for
#: anions and anionic character).  Atom and dimer must use the same one.
BASIS = os.environ.get("DIMER_BASIS", "def2-tzvppd")
#: DIMER_CHARGE: the total charge (e.g. -1 for the anion dimer)
CHARGE = int(os.environ.get("DIMER_CHARGE", "0"))
#: DIMER_FIELDS: the two field strengths (au) of the Richardson pair, the larger first; weaker for diffuse anions,
#: whose response turns nonlinear at 2e-3 au (Au2-: the two strengths differed by 663 bohr^3)
FIELDS = tuple(float(x) for x in os.environ.get("DIMER_FIELDS", "2e-3,1e-3").split(","))


def run(sym: str, d: float):
    from pyscf import dft, gto
    Z = gto.charge(sym)
    geometry = f"{sym} 0 0 0" if d <= 0 else f"{sym} 0 0 {-d / 2}; {sym} 0 0 {d / 2}"
    n_el = (Z if d <= 0 else 2 * Z) - CHARGE
    spin = n_el % 2                                       # d <= 0: the atom (doublet for odd Z)
    if BASIS == "library":
        mol = gto.M(atom=geometry, basis={sym: _basis(sym)}, charge=CHARGE, spin=spin, verbose=0)
    elif BASIS == "def2-tzvppd+2d":
        # def2-TZVPPD alone is not converged for anions (Au-: PBE0 HOMO +0.26 eV, alpha 77 against 122 bohr^3 with
        # two more even-tempered diffuse s, p, d shells, ratio 3, where PBE0 and wB97X agree): augment it
        b = gto.basis.load("def2-tzvppd", sym)
        for l in (0, 1, 2):
            exps = [p[0] for sh in b if sh[0] == l for p in sh[1:]]
            for k in (1, 2):
                b.append([l, [min(exps) / 3.0 ** k, 1.0]])
        mol = gto.M(atom=geometry, basis={sym: b}, ecp="def2-tzvppd" if Z > 36 else None, charge=CHARGE, spin=spin,
                    verbose=0)
    else:
        mol = gto.M(atom=geometry, basis=BASIS, ecp=BASIS if Z > 36 else None, charge=CHARGE, spin=spin, verbose=0)
    mol.set_common_orig([0.0, 0.0, 0.0])
    r = mol.intor_symmetric("int1e_r", comp=3)

    def solve(axis, F, dm0=None):
        mf = dft.UKS(mol) if spin else dft.RKS(mol)
        if Z > 36 and BASIS == "library":
            mf = mf.x2c()
        mf.xc = METHOD
        mf.conv_tol = 1e-12
        mf.max_cycle = 300
        if F != 0.0:
            h0 = mf.get_hcore()
            mf.get_hcore = lambda *a, _h=h0 + F * r[axis]: _h
        t = time.time()
        e = mf.kernel(dm0=dm0)
        if not mf.converged:
            mf = mf.newton()
            e = mf.kernel(dm0=mf.make_rdm1())
        dm = mf.make_rdm1()
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        print(f"  axis {axis} F {F:+.0e}: E {e:.8f} Ha, converged {mf.converged}, {time.time() - t:.0f} s", flush=True)
        return -float(np.einsum("ij,ji->", dm, r[axis])), dm

    _, dm0 = solve(2, 0.0)
    out = {}
    for axis, name in (((2, "parallel"), (0, "perpendicular")) if d > 0 else ((2, "atom"),)):
        a = {}
        F1, F2 = FIELDS                                   # F2 = F1 / 2: Richardson (4 a(F2) - a(F1)) / 3
        for F in (F1, F2):
            mu = [solve(axis, s * F, dm0)[0] for s in (1.0, -1.0)]
            a[F] = (mu[0] - mu[1]) / (2.0 * F)
        best = (4.0 * a[F2] - a[F1]) / 3.0
        out[name] = dict(alpha_bohr3=best, alpha_A3=best * BOHR3_TO_A3, richardson_step_bohr3=abs(a[F2] - a[F1]),
                         fields_au=[F1, F2])
        print(f"{sym}2 d = {d} A  alpha_{name} = {best:.3f} bohr^3 = {best * BOHR3_TO_A3:.4f} A^3 "
              f"(F-step change {abs(a[F2] - a[F1]):.3f})", flush=True)
    if d > 0:
        out["mean_A3"] = (out["parallel"]["alpha_A3"] + 2 * out["perpendicular"]["alpha_A3"]) / 3
    return out


if __name__ == "__main__":
    sym, d = sys.argv[1], float(sys.argv[2])
    res = dict(symbol=sym, d_A=d, charge=CHARGE, method=METHOD, basis=BASIS, **run(sym, d))
    tag = "" if CHARGE == 0 else f"_q{CHARGE:+d}"
    path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.environ.get("TRACE_FP_CACHE", "fp_cache"), f"dimer_alpha_{sym}2_{d}_{BASIS}{tag}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print("written", path)
