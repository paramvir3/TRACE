"""SPEC 38, stage 1 (C4, 2026-11-10): the Fermi band's functional, alone.

    F_b(N) = A(N_up) + A(N_dn) + 1/2 sum_{u != v} V_uv [rho_u rho_v - sum_s (P^s_uv)^2] + sum_u U_u rho^up_u rho^dn_u,
    A(n) = mu n - kT tr ln(1 + exp(-(h - mu)/kT)),   P = [1 + exp((h - mu)/kT)]^-1,   tr P = n,
    N_up = ceil(N/2), N_dn = floor(N/2),   U = diag V.

The criteria F1 - F8 are pre-registered in experiments/benchmarks_4g/README.md ("Pre-registered: C4").  The reference
below is an eigendecomposition in numpy, independent of the module (which uses no eigenvectors in its graph).
"""
import math
import os

import numpy as np
import pytest
import torch
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

KT = 0.1
EPS = torch.finfo(torch.float64).eps
ARRAYS = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "benchmarks_4g", "runs",
                      "extensivity_2026-11-10", "band_arrays_s43.npz")


def _reference_channel(lam, vec, n, kT):
    """(P, A, mu) of one spin channel holding n electrons, by bisection on the eigenvalues."""
    if n == 0:
        return np.zeros((lam.size, lam.size)), 0.0, -math.inf
    if n == lam.size:
        return np.eye(lam.size), float(lam.sum()), math.inf
    occupation = lambda mu: 0.5 * (1.0 - np.tanh(0.5 * (lam - mu) / kT))
    lo, hi = lam[0] - 60.0 * kT, lam[-1] + 60.0 * kT
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        lo, hi = (mu, hi) if occupation(mu).sum() < n else (lo, mu)
    f = occupation(mu)
    omega = -kT * np.logaddexp(0.0, -(lam - mu) / kT).sum()
    return (vec * f) @ vec.T, mu * n + omega, mu


def _reference_energy(h, V, N, kT):
    lam, vec = np.linalg.eigh(h)
    Pu, Au, _ = _reference_channel(lam, vec, (N + 1) // 2, kT)
    Pd, Ad, _ = _reference_channel(lam, vec, N // 2, kT)
    ru, rd = np.diag(Pu), np.diag(Pd)
    off = V - np.diag(np.diag(V))
    pair = 0.5 * float((np.outer(ru + rd, ru + rd) * off).sum() - ((Pu * Pu + Pd * Pd) * off).sum())
    return Au + Ad + pair + (float((np.diag(V) * ru * rd).sum()) if N // 2 else 0.0)


def _random_band(B, seed, width=3.0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(B, B, generator=g, dtype=torch.float64)
    h = width * (a + a.T) / (2.0 * math.sqrt(B))
    d = torch.rand(B, B, generator=g, dtype=torch.float64)
    V = 0.5 * (d + d.T) + torch.diag(2.0 + torch.rand(B, generator=g, dtype=torch.float64))
    return h, V


def _ring(B, t=1.0):
    """A ring of B sites with uniform hopping: levels -2t cos(2 pi k / B), exactly degenerate in +-k."""
    h = torch.zeros(B, B, dtype=torch.float64)
    for u in range(B):
        h[u, (u + 1) % B] = h[(u + 1) % B, u] = -t
    return h


def _symmetric(B, seed):
    a = torch.randn(B, B, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return 0.5 * (a + a.T)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("differentiable", [True, False])
def test_the_projector_holds_n_electrons_and_matches_the_eigendecomposition(seed, differentiable):    # F1
    """Both routes: the polynomial in h (taken when a derivative can be asked for) and the eigenvectors (otherwise)."""
    from trace_redox.fermi import fermi_operator
    h, _ = _random_band(12, seed)
    out = fermi_operator(h.clone().requires_grad_(differentiable), list(range(13)), KT)
    out = out._replace(projector={n: P.detach() for n, P in out.projector.items()})
    lam, vec = np.linalg.eigh(h.numpy())
    for n in range(13):
        P, A, mu = out.projector[n], out.free_energy[n], out.mu[n]
        Pr, Ar, mur = _reference_channel(lam, vec, n, KT)
        assert abs(float(torch.trace(P)) - n) <= 64 * out.order * EPS
        assert np.abs(P.numpy() - Pr).max() <= 64 * out.order * EPS
        assert abs(float(A) - Ar) <= 64 * out.order * EPS * max(1.0, abs(Ar))
        if 0 < n < 12:
            assert abs(float(mu) - mur) <= 1e-9


def test_a_gapped_band_is_spec_20():                                                        # F2
    from trace_redox.fermi import band_free_energy
    h, V = _random_band(10, 3)
    lam, vec = torch.linalg.eigh(h)
    lam = lam.clone(); lam[0] = lam[1] - 12.0                     # gap 12 eV: exp(-gap / 2 kT) = 9e-27
    h = (vec * lam) @ vec.T
    w = vec[:, 0] ** 2
    j11 = float((torch.outer(w, w) * V).sum())
    F, rho = band_free_energy(h, V, [0, 1, 2], KT)
    assert float(F[0]) == 0.0 and float(rho[0].abs().max()) == 0.0
    assert abs(float(F[1]) - float(lam[0])) <= 1e-11
    assert abs(float(F[2]) - (2.0 * float(lam[0]) + j11)) <= 1e-11
    assert float((rho[1] - w).abs().max()) <= 1e-11 and float((rho[2] - 2.0 * w).abs().max()) <= 1e-11


def test_the_full_band():                                                                   # F3
    from trace_redox.fermi import band_free_energy
    h, V = _random_band(7, 4)
    F, rho = band_free_energy(h, V, [14], KT)
    off = V - torch.diag(torch.diagonal(V))
    exact = 2.0 * float(torch.trace(h)) + 2.0 * float(off.sum()) + float(torch.diagonal(V).sum())
    assert abs(float(F[0]) - exact) <= 64 * EPS * abs(exact)
    assert float((rho[0] - 2.0).abs().max()) == 0.0


def _stencil(function, step):
    """Fourth-order central difference of a scalar function of one real variable at 0."""
    return (-function(2.0 * step) + 8.0 * function(step) - 8.0 * function(-step) + function(-2.0 * step)) / (12.0 * step)


def _degenerate_band():
    """A six-ring (levels -2, -1, -1, 1, 1, 2 eV) with a generic interaction; N = 4 puts two electrons in each spin
    channel, so the Fermi level lies inside the degenerate shell at -1 eV; N = 3 does it for the up channel only."""
    return _ring(6), _random_band(6, 5)[1]


@pytest.mark.parametrize("N", [3, 4])
def test_first_derivatives_at_a_degenerate_fermi_level(N):                                   # F4
    from trace_redox.fermi import band_free_energy, fermi_operator
    h0, V0 = _degenerate_band()
    lam = np.linalg.eigvalsh(h0.numpy())
    # the shell is half filled up to the lowest level's hole, exp(-1 eV / kT): mu sits 4 kT * 2.3e-5 above it
    assert abs(lam[1] - lam[2]) <= 8 * EPS and abs(float(fermi_operator(h0, [2], KT).mu[2]) - lam[1]) <= 1e-3 * KT
    X, Y = _symmetric(6, 11), _symmetric(6, 12)
    h, V = h0.clone().requires_grad_(True), V0.clone().requires_grad_(True)
    F, rho = band_free_energy(h, V, [N], KT)
    gh, gV = torch.autograd.grad(F[0], (h, V), retain_graph=True)
    energy = lambda s, dh, dV: float(band_free_energy(h0 + s * dh, V0 + s * dV, [N], KT)[0][0])
    for analytic, dh, dV in ((float((gh * X).sum()), X, 0.0 * Y), (float((gV * Y).sum()), 0.0 * X, Y)):
        numeric = _stencil(lambda s: energy(s, dh, dV), 1e-3)
        assert abs(analytic - numeric) <= 1e-7 * max(1.0, abs(numeric)), (analytic, numeric)
    # the charge shape is the derivative of nothing here: check it directly, in the same way
    shape = lambda s: band_free_energy(h0 + s * X, V0, [N], KT)[1][0]
    (grho,) = torch.autograd.grad((rho[0] * torch.arange(1.0, 7.0)).sum(), h)
    numeric = float((_stencil(shape, 1e-3) * torch.arange(1.0, 7.0)).sum())
    assert abs(float((grho * X).sum()) - numeric) <= 1e-7 * max(1.0, abs(numeric))


@pytest.mark.parametrize("N", [3, 4])
def test_second_derivatives_at_a_degenerate_fermi_level(N):                                  # F5
    from trace_redox.fermi import band_free_energy
    h0, V0 = _degenerate_band()
    X, Y, Z = _symmetric(6, 11), _symmetric(6, 12), _symmetric(6, 13)

    def slope(s, dh, dV):
        """d/dt F(h + t X) at h = h0 + s dh, V = V0 + s dV, with its graph."""
        h, V = (h0 + s * dh).requires_grad_(True), (V0 + s * dV).requires_grad_(True)
        F, _ = band_free_energy(h, V, [N], KT)
        (g,) = torch.autograd.grad(F[0], h, create_graph=True)
        return (g * X).sum(), h, V

    g, h, V = slope(0.0, Y, Z)
    ghh, ghV = torch.autograd.grad(g, (h, V))
    for analytic, dh, dV in ((float((ghh * Y).sum()), Y, 0.0 * Z), (float((ghV * Z).sum()), 0.0 * Y, Z)):
        numeric = _stencil(lambda s: float(slope(s, dh, dV)[0]), 1e-3)
        assert abs(analytic - numeric) <= 1e-7 * max(1.0, abs(numeric)), (analytic, numeric)


@pytest.mark.parametrize("N", [2, 4, 6])
def test_two_uncoupled_copies_at_twice_an_even_filling(N):                                   # F6
    from trace_redox.fermi import band_free_energy, fermi_operator
    h, V = _random_band(8, 7)
    zero = torch.zeros_like(h)
    h2 = torch.cat([torch.cat([h, zero], 1), torch.cat([zero, h], 1)])
    V2 = torch.cat([torch.cat([V, zero], 1), torch.cat([zero, V], 1)])
    one, two = band_free_energy(h, V, [N], KT), band_free_energy(h2, V2, [2 * N], KT)
    bound = 64 * fermi_operator(h, [1], KT).order * EPS
    assert abs(float(two[0][0]) - 2.0 * float(one[0][0])) <= bound * max(1.0, abs(float(two[0][0])))
    assert float((two[1][0] - torch.cat([one[1][0], one[1][0]])).abs().max()) <= bound


def test_an_infinite_second_electron_cost_admits_one_electron():                            # F7
    from trace_redox.fermi import band_free_energy
    h, V = _random_band(6, 8)
    V[2, 2] = float("inf")
    F, rho = band_free_energy(h, V, [0, 1], KT)
    assert bool(torch.isfinite(F).all()) and bool(torch.isfinite(rho).all())
    finite = V.clone(); finite[2, 2] = 0.0
    assert float(F[1]) == float(band_free_energy(h, finite, [1], KT)[0][0])
    with pytest.raises(ValueError, match="at most one electron"):
        band_free_energy(h, V, [2], KT)


@pytest.mark.parametrize("tag", ["doped_wetting_1x1", "doped_wetting_2x1", "doped_non-wetting_2x2", "undoped_wetting_1x1"])
def test_the_benchmark_bands_against_the_reference(tag):                                     # F8
    from trace_redox.fermi import band_free_energy
    if not os.path.exists(ARRAYS):
        pytest.skip("the saved band arrays of the Au2/MgO cells are not on this machine (a gitignored run directory)")
    data = np.load(ARRAYS)
    h = np.diag(data[tag + "/eps"]) - data[tag + "/tau"]
    V = data[tag + "/V"]
    copies = int(tag[-3]) * int(tag[-1])
    fillings = list(range(0, 4 * copies + 1))
    reference = [_reference_energy(h, V, N, KT) for N in fillings]
    for differentiable in (True, False):                 # the polynomial in h, then the eigenvectors
        F, _ = band_free_energy(torch.as_tensor(h).requires_grad_(differentiable), torch.as_tensor(V), fillings, KT)
        for N, value, exact in zip(fillings, F.tolist(), reference):
            assert abs(value - exact) <= 1e-9, (tag, N, differentiable)


@pytest.mark.parametrize("N", [1, 2, 3])
def test_derivatives_of_a_gapped_band(N):
    """Not a pre-registered criterion: where the count does not respond to mu (a 12 eV gap at kT = 0.1 eV) mu is held
    constant, and the first and second derivatives must still be those of the energy."""
    from trace_redox.fermi import band_free_energy
    h0, V0 = _random_band(8, 9)
    lam, vec = torch.linalg.eigh(h0)
    lam = lam.clone(); lam[0] = lam[1] - 12.0
    h0 = (vec * lam) @ vec.T
    X, Y = _symmetric(8, 14), _symmetric(8, 15)

    def slope(s):
        h = (h0 + s * Y).requires_grad_(True)
        F, _ = band_free_energy(h, V0, [N], KT)
        (g,) = torch.autograd.grad(F[0], h, create_graph=True)
        return (g * X).sum(), h

    g, h = slope(0.0)
    (gh,) = torch.autograd.grad(g, h)
    first = _stencil(lambda s: float(band_free_energy(h0 + s * X, V0, [N], KT)[0][0]), 1e-3)
    second = _stencil(lambda s: float(slope(s)[0]), 1e-3)
    assert abs(float(g) - first) <= 1e-7 * max(1.0, abs(first)), (float(g), first)
    assert abs(float((gh * Y).sum()) - second) <= 1e-7 * max(1.0, abs(second)), (float((gh * Y).sum()), second)
