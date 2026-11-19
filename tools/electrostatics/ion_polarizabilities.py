"""Static dipole polarisabilities of every charge state, by the method of the ion-density library (2026-10-18).

    python tools/electrostatics/ion_polarizabilities.py WORK [SYMBOL[q] ...]
    (python: an interpreter with PySCF installed)

Why: the bound sector carries one polarisability per atom, the same in every charge state (trained to the neutral
atom's, e.g. Au 5.4 A^3), while a charge state's own polarisability differs by a large factor (an anion's diffuse
frontier shell against a cation's compact core).  This measures alpha(Z, q) before anything is built.

Method (the library's, tools/electrostatics/build_ion_densities.py): spin-unrestricted PBE0, X2C spin-free for Z > 36,
ANO-RCC plus two diffuse s and p shells, the spin state stored with the density, every O state in the Watson sphere.
A uniform field F along an axis enters the one-electron Hamiltonian as F z (no picture-change correction of the
dipole operator under X2C: a valence-shell response, stated).  alpha = d mu / dF from the central difference of the
electronic dipole at +-F, Richardson-extrapolated over F = 2e-3 and 1e-3 au (the O(F^2) hyperpolarisability term
cancels); S states use one axis, every other state the mean over x, y, z.  Results: WORK/polarizabilities.json, A^3.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_ion_densities import METHOD, _basis  # noqa: E402

BOHR3_TO_A3 = 0.529177210903 ** 3
#: atoms whose ground state is not an S state in some charge state: their polarisability is the mean over three axes
S_STATES = {(1, 0), (1, -1), (1, 1), (11, -1), (11, 0), (11, 1), (12, 0), (12, 1), (12, 2), (13, 3), (17, -1),
            (8, -2), (47, -1), (47, 0), (47, 1), (79, -1), (79, 0), (79, 1), (6, 4)}


def _mf(mol, Z, watson, field_axis, F, dm0=None):
    from pyscf import dft, qmmm
    mf = dft.UKS(mol)
    if Z > 36:
        mf = mf.x2c()
    mf.xc = METHOD
    mf.conv_tol = 1e-11
    mf.max_cycle = 300
    if watson is not None:
        n_dir = 302
        i = np.arange(n_dir) + 0.5
        phi = np.arccos(1 - 2 * i / n_dir)
        th = np.pi * (1 + 5 ** 0.5) * i
        dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
        RW, QW = watson
        mf = qmmm.mm_charge(mf, dirs * RW, np.full(n_dir, QW / n_dir), radii=np.full(n_dir, 0.3))
    if F != 0.0:
        mol.set_common_orig([0.0, 0.0, 0.0])
        r = mol.intor_symmetric("int1e_r", comp=3)
        h0 = mf.get_hcore()
        mf.get_hcore = lambda *args, _h=h0 + F * r[field_axis]: _h
    e = mf.kernel(dm0=dm0)
    if not mf.converged:
        mf = mf.newton()
        e = mf.kernel(dm0=mf.make_rdm1())
    return mf, e


def _dipole(mol, mf, axis):
    mol.set_common_orig([0.0, 0.0, 0.0])
    r = mol.intor_symmetric("int1e_r", comp=3)[axis]
    dm = mf.make_rdm1()
    return -float(np.einsum("ij,ji->", dm[0] + dm[1], r))         # electronic dipole (nucleus at the origin)


def polarizability(entry, spin):
    """``spin < 0`` (unknown: densities restored from the library after the 2026-10-18 reboot): the lower of the two
    lowest spin states, as the density stage chooses."""
    from pyscf import gto
    Z, q, sym = entry["Z"], entry["q"], entry["symbol"]
    n_el = Z - q
    best = None
    for s in ((spin,) if spin >= 0 else (n_el % 2, n_el % 2 + 2)):
        if s > n_el:
            continue
        m = gto.M(atom=f"{sym} 0 0 0", basis={sym: _basis(sym)}, charge=q, spin=s, verbose=0)
        mf, e = _mf(m, Z, entry["watson"], 2, 0.0)
        if mf.converged and (best is None or e < best[0]):
            best = (e, s, m, mf)
    _, spin, mol, mf0 = best
    dm0 = mf0.make_rdm1()
    axes = (2,) if (Z, q) in S_STATES else (0, 1, 2)
    per_axis = []
    for axis in axes:
        a = {}
        for F in (2e-3, 1e-3):
            mu = [_dipole(mol, _mf(mol, Z, entry["watson"], axis, s * F, dm0)[0], axis) for s in (1.0, -1.0)]
            a[F] = (mu[0] - mu[1]) / (2.0 * F)
        per_axis.append(((4.0 * a[1e-3] - a[2e-3]) / 3.0, a[1e-3], a[2e-3]))
    alpha = float(np.mean([p[0] for p in per_axis]))
    spread = float(np.mean([abs(p[1] - p[2]) for p in per_axis]))
    return alpha, spread, [p[0] for p in per_axis], spin


def main(work, wanted):
    with open(os.path.join(work, "spec.json")) as fh:
        spec = json.load(fh)
    out_path = os.path.join(work, "polarizabilities.json")
    done = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for entry in spec:
        Z, q, sym = entry["Z"], entry["q"], entry["symbol"]
        key = f"{sym}{q:+d}"
        if wanted and sym not in wanted and key not in wanted:
            continue
        if key in done or Z - q == 0:
            continue
        path = os.path.join(work, f"{Z}_{q:+d}.npz")
        if not os.path.exists(path):
            print(f"{key}: no density (left out)", flush=True)
            continue
        spin = int(np.load(path)["spin"])
        try:
            a, spread, axes, spin = polarizability(entry, spin)
        except Exception as err:                                       # noqa: BLE001
            print(f"{key}: {type(err).__name__} {str(err)[:100]}", flush=True)
            continue
        done[key] = dict(Z=Z, q=q, spin=spin, alpha_bohr3=a, alpha_A3=a * BOHR3_TO_A3, axes_bohr3=axes,
                         richardson_step_bohr3=spread, method=METHOD, watson=entry["watson"])
        with open(out_path, "w") as fh:
            json.dump(done, fh, indent=1)
        print(f"{key}: alpha = {a:.3f} bohr^3 = {a * BOHR3_TO_A3:.4f} A^3 (F-step change {spread:.3f} bohr^3; "
              f"axes {[round(x, 3) for x in axes]})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], set(sys.argv[2:]))
