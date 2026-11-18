"""The Mermin ensemble of one unit (trace_redox.ensemble; SPEC 25.3, Lemma 1).

Every expected value is an identity of the Legendre transform or a bound that
follows from it; no number is taken from the implementation under test.
"""

import math

import pytest
import torch
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

from trace_redox import ensemble as ens

STATES = torch.tensor([-1.0, 0.0, 1.0])
EPS = torch.finfo(torch.float64).eps


def _units(seed, U=5, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    e = torch.randn(U, 3, generator=g) * scale
    e[:, 1] = 0.0
    return e


def test_legendre_pair_is_the_supremum():
    """Phi(c) = sup_nu' [nu' c + Omega(nu')] is attained at the nu with N(nu) = c."""
    e = _units(1)
    kT = 0.1
    nu = torch.linspace(-1.5, 1.5, 5)
    c, phi = ens.conjugate_at(e, STATES, nu, kT)
    grid = torch.linspace(-40.0, 40.0, 4001)
    for u in range(e.shape[0]):
        values = grid * c[u] + ens.grand_potential(e[u].expand(grid.numel(), 3), STATES, grid, kT)
        # the supremum over a grid can only be lower; equality where the grid hits nu
        assert float(values.max()) <= float(phi[u]) + 64 * EPS * (1 + float(values.abs().max()))


def test_slope_of_phi_is_the_potential():
    """Phi'(c) = nu, by central differences of Phi at N(nu +- h), with no inversion."""
    e = _units(2, scale=0.5)
    kT = 0.1
    nu = torch.tensor([-0.4, -0.1, 0.0, 0.2, 0.5])
    h = 1e-4
    c_p, phi_p = ens.conjugate_at(e, STATES, nu + h, kT)
    c_m, phi_m = ens.conjugate_at(e, STATES, nu - h, kT)
    slope = (phi_p - phi_m) / (c_p - c_m)
    # chord slope of a C^inf convex function between N(nu-h), N(nu+h): nu + O(h^2)
    assert torch.allclose(slope, nu, atol=1e-7, rtol=0.0)


def test_susceptibility_is_the_derivative_of_the_mean_charge():
    e = _units(3)
    kT = 0.07
    nu = torch.linspace(-2.0, 2.0, 5).requires_grad_(True)
    N = ens.mean_charge(e, STATES, nu, kT)
    (dN,) = torch.autograd.grad(N.sum(), nu)
    X = ens.susceptibility(e, STATES, nu.detach(), kT)
    assert torch.allclose(dN, X, rtol=1e-12, atol=1e-14)
    assert bool((X >= 0).all())


def test_phi_is_convex_even_for_non_convex_state_energies():
    """A state above the chord of its neighbours does not break convexity (Lemma 1)."""
    e = torch.tensor([[0.0, 1.0, 0.0], [2.0, 0.0, -3.0], [-1.0, 0.0, 4.0]])
    kT = 0.05
    nu = torch.linspace(-12.0, 12.0, 6001)
    c, phi = ens.conjugate_at(e.repeat_interleave(nu.numel(), 0), STATES, nu.repeat(3), kT)
    m = nu.numel()
    for u in range(3):
        cu, pu = c[u * m:(u + 1) * m], phi[u * m:(u + 1) * m]
        # a subsequence of points at least 1e-6 e apart, so a chord slope carries
        # at most ~eps |Phi| / 1e-6 ~ 1e-9 of round-off
        picked, last = [0], float(cu[0])
        for k in range(1, m):
            if float(cu[k]) - last >= 1e-6:
                picked.append(k)
                last = float(cu[k])
        idx = torch.tensor(picked)
        slopes = torch.diff(pu[idx]) / torch.diff(cu[idx])
        assert len(picked) > 100
        assert bool((torch.diff(slopes) >= -1e-8).all())


@pytest.mark.parametrize("kT", [0.2, 0.05, 0.01, 0.002])
def test_zero_temperature_limit_is_the_convex_envelope(kT):
    """env(c) - kT ln S <= Phi_T(c) <= env(c): the entropy term lies in [-kT ln S, 0]."""
    e = torch.tensor([[0.0, 1.0, 0.0], [-2.0, 0.0, 3.0], [1.5, 0.0, -0.5], [4.0, 0.0, 4.0]])
    nu = torch.linspace(-8.0, 8.0, 801)
    U = e.shape[0]
    c, phi = ens.conjugate_at(e.repeat_interleave(nu.numel(), 0), STATES, nu.repeat(U), kT)
    env = ens.convex_envelope(e.repeat_interleave(nu.numel(), 0), STATES, c)
    slack = 64 * EPS * (1.0 + float(phi.abs().max()))
    assert bool((phi <= env + slack).all())
    assert bool((phi >= env - kT * math.log(3.0) - slack).all())


def test_a_state_above_the_envelope_is_never_populated():
    """Disproportionation: E_0 above the (-1, +1) chord by D -> p_0 = e^{-D/kT} / (2 + e^{-D/kT}) at c = 0."""
    D = 0.7
    e = torch.tensor([[0.0, D, 0.0]])
    for kT in (0.1, 0.03):
        p = ens.occupations(e, STATES, torch.zeros(1), kT)
        x = math.exp(-D / kT)
        assert abs(float(p[0, 1]) - x / (2.0 + x)) <= 16 * EPS
        assert abs(float(ens.mean_charge(e, STATES, torch.zeros(1), kT))) <= 16 * EPS


def test_mean_charge_monotone_inside_the_open_interval_and_inverse():
    e = _units(4)
    kT = 0.1
    nu = torch.linspace(-30.0, 30.0, 2001)
    for u in range(e.shape[0]):
        N = ens.mean_charge(e[u].expand(nu.numel(), 3), STATES, nu, kT)
        assert bool((torch.diff(N) >= 0).all())
        # open interval mathematically; at |nu| = 30 eV, kT = 0.1 the distance to
        # +-1 is ~e^-300, below float64 resolution, so only |nu| <= 2 eV is strict
        assert bool(((N >= -1.0) & (N <= 1.0)).all())
        moderate = nu.abs() <= 2.0
        assert bool(((N[moderate] > -1.0) & (N[moderate] < 1.0)).all())
    c = torch.tensor([-0.9, -0.3, 0.0, 0.4, 0.95])
    nu_c = ens.potential_for_charge(e, STATES, c, kT)
    assert torch.allclose(ens.mean_charge(e, STATES, nu_c, kT), c, atol=1e-12, rtol=0.0)
    with pytest.raises(ValueError):
        ens.potential_for_charge(e, STATES, torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]), kT)


def test_the_suite_runs_in_float64():
    """The autouse fixture must win over tests/conftest.py's per-test dtype reset."""
    assert torch.get_default_dtype() == torch.float64
    assert torch.zeros(1).dtype == torch.float64
