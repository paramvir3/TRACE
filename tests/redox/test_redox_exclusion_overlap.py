"""The density-overlap integral of exclusion (trace_redox SPEC 18; theory note 2026-10-15).

Frontier density of a charge state, exact in its asymptotic tail (Katriel & Davidson, PNAS 77, 4403 (1980)) and
finite at the nucleus:

    rho(r) = (r^2 + kappa^-2)^beta exp(-2 kappa r) / Z(kappa, beta),   int rho d^3r = 1      (atomic units)

and the overlap S(R) = int rho_a(|r|) rho_b(|r - R|) d^3r, computed in prolate spheroidal coordinates with fixed
Gauss-Legendre x Gauss-Laguerre nodes (differentiable in R).  Reference: scipy's adaptive 2-D quadrature in
cylindrical coordinates, an independent route.
"""
import math

import pytest
import torch
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

BOHR = 0.529177210903


def _reference(ka, ba, kb, bb, R_bohr):
    import numpy as np
    from scipy import integrate

    def radial_norm(k, b):
        return integrate.quad(lambda r: 4 * np.pi * r * r * (r * r + k ** -2) ** b * np.exp(-2 * k * r), 0, np.inf,
                              limit=400, epsabs=0, epsrel=1e-12)[0]

    za, zb = radial_norm(ka, ba), radial_norm(kb, bb)

    def f(rho, z):
        ra2, rb2 = rho * rho + z * z, rho * rho + (z - R_bohr) ** 2
        return (2 * np.pi * rho * (ra2 + ka ** -2) ** ba * np.exp(-2 * ka * np.sqrt(ra2))
                * (rb2 + kb ** -2) ** bb * np.exp(-2 * kb * np.sqrt(rb2)) / (za * zb))

    span = 40.0 / min(ka, kb)
    return integrate.dblquad(f, -span, R_bohr + span, 0, span, epsabs=0, epsrel=1e-10)[0]


CASES = [  # (kappa_a, beta_a, kappa_b, beta_b): Au- 6s (EA), Au0 6s, Al2+ 3s, O2- valence (I ~ 9 eV), a cation core
    (0.4119, -1.0, 0.8133, 0.0),
    (0.8234, 0.2145, 0.8133, 0.0),
    (1.4460, 1.0747, 0.8133, 0.0),
    (2.4271, 1.8853, 0.8133, 0.0),
    (0.4119, -1.0, 0.4119, -1.0),
]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("R_angstrom", [1.8, 2.2, 3.0, 4.7, 6.0])
def test_the_overlap_matches_an_independent_adaptive_integral(case, R_angstrom):
    from trace_redox.exclusion import density_overlap
    ka, ba, kb, bb = case
    R = R_angstrom / BOHR
    got = float(density_overlap(torch.tensor(ka), torch.tensor(ba), torch.tensor(kb), torch.tensor(bb), torch.tensor(R)))
    ref = _reference(ka, ba, kb, bb, R)
    assert abs(got - ref) <= 1e-6 * ref + 1e-14, (case, R_angstrom, got, ref)


def test_the_density_is_normalised():
    from trace_redox.exclusion import radial_normalisation
    import numpy as np
    from scipy import integrate
    for ka, ba, _, _ in CASES:
        ref = integrate.quad(lambda r: 4 * np.pi * r * r * (r * r + ka ** -2) ** ba * np.exp(-2 * ka * r), 0, np.inf,
                             limit=400, epsabs=0, epsrel=1e-12)[0]
        got = float(radial_normalisation(torch.tensor(ka), torch.tensor(ba)))
        assert abs(got - ref) <= 1e-9 * ref


def test_the_overlap_is_symmetric_and_decays_monotonically():
    from trace_redox.exclusion import density_overlap
    ka, ba, kb, bb = (torch.tensor(x) for x in CASES[0])
    R = torch.linspace(2.0, 12.0, 21)
    s_ab = density_overlap(ka, ba, kb, bb, R)
    s_ba = density_overlap(kb, bb, ka, ba, R)
    assert torch.allclose(s_ab, s_ba, rtol=1e-10, atol=0)
    assert bool((s_ab[1:] < s_ab[:-1]).all())


def test_the_overlap_derivative_matches_finite_differences():
    from trace_redox.exclusion import density_overlap
    ka, ba, kb, bb = (torch.tensor(x) for x in CASES[2])
    R = torch.tensor(4.0, requires_grad=True)
    (g,) = torch.autograd.grad(density_overlap(ka, ba, kb, bb, R), R)
    h = 1e-5
    fd = (float(density_overlap(ka, ba, kb, bb, torch.tensor(4.0 + h)))
          - float(density_overlap(ka, ba, kb, bb, torch.tensor(4.0 - h)))) / (2 * h)
    assert abs(float(g) - fd) <= 1e-7 * abs(fd)
