"""Exact half-integer regularized incomplete gamma for the periodic real-space damping (speed-up A, B).

P(a, x) = gamma(a, x) / Gamma(a).  For a = n + 1/2:
    P(1/2, x) = erf(sqrt x),   P(a + 1, x) = P(a, x) - x^a e^{-x} / Gamma(a + 1)     (exact recursion)
    P(a, x)   = x^a e^{-x} sum_k x^k / Gamma(a + k + 1)                             (exact series)
Upward recursion from erf cancels at small x, so x < 3 sums the series for the top order and
recurses DOWNWARD (adding positive terms); x >= 3 recurses upward (amplification <= ~5).

Reference: 50-digit mpmath.  torch.special.gammainc itself is off by up to 1.8e-14 relative
(measured, 2026-10-09); the new function is held to 32 eps.
"""
import math

import pytest
import torch
from conftest import require

mpmath = pytest.importorskip("mpmath")
EPS = torch.finfo(torch.float64).eps
XS = torch.cat((torch.zeros(1, dtype=torch.float64), torch.logspace(-12, -1, 23, dtype=torch.float64),
                torch.linspace(0.1, 6.0, 119, dtype=torch.float64), torch.tensor([2.999999, 3.0, 3.000001]),
                torch.logspace(math.log10(6.0), 3.0, 30, dtype=torch.float64)))


def _ref(a, x):
    mpmath.mp.dps = 50
    return float(mpmath.gammainc(a, 0, x, regularized=True))


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_matches_fifty_digit_reference(n):
    f = require("special", "half_integer_regularized_gamma")
    (got,) = f(XS, orders=(n,))
    for x, value in zip(XS.tolist(), got.tolist()):
        ref = _ref(n + 0.5, x)
        if ref == 0.0:
            assert value == 0.0
        elif ref > 1e-290:
            assert abs(value - ref) <= 32 * EPS * ref, (n, x, value, ref)


def test_orders_are_returned_in_the_requested_order_and_agree():
    f = require("special", "half_integer_regularized_gamma")
    all_four = f(XS, orders=(1, 2, 3, 4))
    picked, ordered = f(XS, orders=(3, 1)), f(XS, orders=(1, 3))       # same set: identical arithmetic
    assert torch.equal(picked[0], ordered[1]) and torch.equal(picked[1], ordered[0])
    # A different set of orders changes the recursion path (downward from the highest order), so the
    # same order can differ in the last bits: each value is within 32 eps of the truth, hence the two
    # within 64 eps of each other.
    for n in (1, 2, 3, 4):
        alone = f(XS, orders=(n,))[0]
        assert bool(((all_four[n - 1] - alone).abs() <= 64 * EPS * alone.abs() + 1e-300).all())


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_derivative_is_exact_and_finite_at_zero(n):
    f = require("special", "half_integer_regularized_gamma")
    x = XS.clone().requires_grad_(True)
    (p,) = f(x, orders=(n,))
    (g,) = torch.autograd.grad(p.sum(), x)
    assert bool(torch.isfinite(g).all())
    a = n + 0.5
    for xv, gv in zip(XS.tolist(), g.tolist()):
        mpmath.mp.dps = 50
        ref = float(mpmath.power(xv, a - 1) * mpmath.exp(-xv) / mpmath.gamma(a)) if xv > 0 else 0.0
        if ref > 1e-290 or ref == 0.0:
            assert abs(gv - ref) <= 64 * EPS * max(ref, 1e-300) or abs(gv - ref) <= 1e-300, (n, xv, gv, ref)


def _old_damping(r, width, alpha):
    smeared, screened = (r / width) ** 2, (alpha * r) ** 2
    return tuple(torch.special.gammainc(torch.full_like(smeared, o + 0.5), smeared)
                 - torch.special.gammainc(torch.full_like(screened, o + 0.5), screened) for o in (1, 2, 3, 4))


def test_real_space_damping_agrees_with_the_gammainc_form():
    real_space_damping = require("lattice", "real_space_damping")
    g = torch.Generator().manual_seed(1)
    r = 12.0 * torch.rand(4, 30, 30, generator=g, dtype=torch.float64)
    width = 0.6 + 2.0 * torch.rand(1, 30, 30, generator=g, dtype=torch.float64)
    alpha = 0.31
    old = _old_damping(r, width, alpha)
    new = real_space_damping(r, width, alpha)
    scale = max(float(o.abs().max()) for o in old)
    for o, nw in zip(old, new):
        assert float((o - nw).abs().max()) <= 1e-13 * scale
    sub = real_space_damping(r, width, alpha, orders=(2, 3))
    for a, b in ((sub[0], new[1]), (sub[1], new[2])):            # see test_orders_...: 64 eps per term
        assert float((a - b).abs().max()) <= 2 * 64 * EPS


def test_masked_damping_equals_the_full_damping_on_the_mask():
    """Speed-up B: computing only where the real-space cutoff keeps a pair changes nothing."""
    real_space_damping = require("lattice", "real_space_damping")
    g = torch.Generator().manual_seed(2)
    r = 12.0 * torch.rand(3, 20, 20, generator=g, dtype=torch.float64)
    width = 0.6 + 2.0 * torch.rand(1, 20, 20, generator=g, dtype=torch.float64)
    mask = r < 8.0
    full = real_space_damping(r, width, 0.3)
    masked = real_space_damping(r, width, 0.3, mask=mask)
    for f, m in zip(full, masked):
        assert torch.equal(m[mask], f[mask]) and torch.equal(m[~mask], torch.zeros_like(m[~mask]))


@pytest.mark.parametrize("case", ["rocksalt", "triclinic"])
def test_lattice_blocks_unchanged_to_round_off(case, monkeypatch):
    """Every periodic Gram block with the new damping equals the gammainc one to 1e-13 relative."""
    lattice = require("lattice")
    EwaldParams = require("ewald", "EwaldParams")
    width = require("kernels", "gaussian_multipole_width")
    g = torch.Generator().manual_seed(5)
    if case == "rocksalt":
        a = 4.212
        frac = torch.tensor([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                             [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]], dtype=torch.float64)
        cell = a * torch.eye(3, dtype=torch.float64)
        pos = frac @ cell
    else:
        cell = torch.tensor([[7.0, 0.3, 0.0], [0.2, 6.5, 0.4], [0.0, 0.5, 7.5]], dtype=torch.float64)
        pos = torch.rand(6, 3, generator=g, dtype=torch.float64) @ cell
    n = len(pos)
    sigma = 0.8 + 0.4 * torch.rand(n, generator=g, dtype=torch.float64)
    ell = 0.6 + 0.6 * torch.rand(n, generator=g, dtype=torch.float64)
    A = torch.randn(n, 3, 3, generator=g, dtype=torch.float64)
    A_mu = A @ A.transpose(1, 2) + 3.0 * torch.eye(3, dtype=torch.float64)
    sigma_mu = width(A_mu, ell, 1, 0.5)
    ewald = EwaldParams.auto_tune(cell, 12.0, accuracy=1e-12)
    new = lattice.gaussian_lattice_blocks(pos, cell, sigma, sigma_mu, ell, ewald)
    monkeypatch.setattr(lattice, "real_space_damping", lambda r, w, al, orders=(1, 2, 3, 4), mask=None:
                        tuple(_old_damping(r, w, al)[o - 1] for o in orders))
    old = lattice.gaussian_lattice_blocks(pos, cell, sigma, sigma_mu, ell, ewald)
    # All blocks are parts of one operator K; a block that vanishes by symmetry (rocksalt t01) has
    # only rounding noise as its own maximum, so the scale is K's largest pair entry.
    scale = max(float(block.abs().max()) for block in old.values())
    for key in old:
        assert float((new[key] - old[key]).abs().max()) <= 1e-13 * scale, key


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_second_derivative_is_exact(n):
    """Force training differentiates the forces: d2P/dx2 = x^(a-2) e^(-x) ((a - 1) - x) / Gamma(a)."""
    f = require("special", "half_integer_regularized_gamma")
    xs = torch.cat((torch.logspace(-3, 0, 10, dtype=torch.float64), torch.linspace(1.5, 40.0, 25, dtype=torch.float64)))
    x = xs.clone().requires_grad_(True)
    (p,) = f(x, orders=(n,))
    (g,) = torch.autograd.grad(p.sum(), x, create_graph=True)
    (h,) = torch.autograd.grad(g.sum(), x)
    a = n + 0.5
    for xv, hv in zip(xs.tolist(), h.tolist()):
        mpmath.mp.dps = 50
        ref = float(mpmath.power(xv, a - 2) * mpmath.exp(-xv) * ((a - 1) - xv) / mpmath.gamma(a))
        assert abs(hv - ref) <= 64 * EPS * abs(ref) + 1e-300, (n, xv, hv, ref)
