"""Gordon-Kim's electron-gas overlap energy split into its four functionals (2026-10-20).

trace_redox.radial.gordon_kim_difference integrates f[rho_a + rho_b] - f[rho_a] - f[rho_b] (relative to the frames) with
f = Thomas-Fermi + (1/9) von Weizsaecker + LDA exchange + Wigner correlation (SPEC 25).  This module integrates the
same non-additive differences one functional at a time, on the same prolate-spheroidal grid, so each can be set
against its first-principles counterpart (frozen_interaction.decomposed_interaction):

    tf + vw   (electron-gas kinetic)   against   the exact orbital kinetic cost of antisymmetrisation, with the
                                                 Coulomb and xc of the orthogonalisation density change
    x + c     (LDA + Wigner)           against   dfa_sum_<xc>: the functional's own non-additive xc of rho_a + rho_b

gk_parts(r, rho_a, rho_b, R) is absolute (no frame subtraction); the sum of its four parts equals
gordon_kim_difference(r, 0, rho_a, 0, rho_b, R), which `check` asserts.
"""
import os
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _radial():
    """trace_redox/radial.py by its path: it is numpy only, and the package itself imports torch (absent in the PySCF
    environment)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("trace_redox_radial", os.path.join(REPO, "trace_redox", "radial.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


radial = _radial()
C_F, C_X, VW_FRACTION = radial.C_F, radial.C_X, radial.VW_FRACTION
RadialProfile, gordon_kim_difference = radial.RadialProfile, radial.gordon_kim_difference

HARTREE = 27.211386245988


def _parts(rho, grad2):
    rho = np.maximum(rho, 0.0)
    safe = np.maximum(rho, 1e-300)
    live = rho > 1e-300
    rs = (3.0 / (4.0 * np.pi * safe)) ** (1.0 / 3.0)
    return {"tf": np.where(live, C_F * rho ** (5.0 / 3.0), 0.0),
            "vw": np.where(live, VW_FRACTION * grad2 / (8.0 * safe), 0.0),
            "x": np.where(live, -C_X * rho ** (4.0 / 3.0), 0.0),
            "c": np.where(live, -0.44 * rho / (7.8 + rs), 0.0)}


def gk_parts(r, rho_a, rho_b, R, n_mu=450, n_nu=200):
    """Hartree; R in bohr; the grid of gordon_kim_difference."""
    t = np.exp(np.linspace(np.log(1e-7), np.log(60.0 / R), n_mu))
    w_t = np.gradient(t)
    x, wx = np.polynomial.legendre.leggauss(n_nu)
    MU, NU = np.meshgrid(1.0 + t, x, indexing="ij")
    ra = np.maximum(R * (MU + NU) / 2.0, 1e-12)
    rb = np.maximum(R * (MU - NU) / 2.0, 1e-12)
    cos = np.clip((ra * ra + rb * rb - R * R) / (2.0 * ra * rb), -1.0, 1.0)
    dV = 2.0 * np.pi * (R / 2.0) ** 3 * (MU * MU - NU * NU) * w_t[:, None] * wx[None, :]
    A, dA = RadialProfile(r, rho_a)(ra)
    B, dB = RadialProfile(r, rho_b)(rb)
    tot, pa, pb = (_parts(A + B, dA * dA + dB * dB + 2.0 * dA * dB * cos), _parts(A, dA * dA), _parts(B, dB * dB))
    return {k: float(np.sum((tot[k] - pa[k] - pb[k]) * dV)) for k in tot}


def check(r, rho_a, rho_b, R):
    parts = gk_parts(r, rho_a, rho_b, R)
    zero = np.zeros_like(r)
    ref = gordon_kim_difference(r, zero, rho_a, zero, rho_b, R)
    s = sum(parts.values())
    assert abs(s - ref) <= 1e-12 * max(1.0, abs(ref)) + 1e-14, (s, ref)
    return parts
