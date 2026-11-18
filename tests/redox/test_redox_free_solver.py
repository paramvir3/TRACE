"""The free-charge solve (trace_redox.free_solver; SPEC 25.5 T1, T4, T5).

The optimum is checked by routes that share nothing with the dual Newton
iteration: the primal KKT conditions with Phi' from bisection, a brute-force
golden-section minimisation of the primal for two units, the closed-form
linear response (T5) for the implicit derivatives, and the rigorous
zero-temperature bound for the two-site transfer of SPEC 25.4 (25.3).
"""

import math

import pytest
import torch
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

from trace_redox import ensemble as ens
from trace_redox.free_solver import (
    FreeChargeNotConvergedError,
    differentiable_solution,
    primal_value,
    solve_free_charge,
)

STATES = torch.tensor([-1.0, 0.0, 1.0])
EPS = torch.finfo(torch.float64).eps


def problem(U, seed, gap_scale=2.0):
    """A random convex problem: W positive definite (eV/e^2), gaps G > 0, chi, b."""
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(U, U, generator=g)
    W = 0.4 * (M @ M.T) / U + 0.5 * torch.eye(U)
    chi = torch.randn(U, generator=g) * 1.5
    G = gap_scale * torch.rand(U, generator=g) + 0.3
    e_hat = torch.stack((-chi + 0.5 * G, torch.zeros(U), chi + 0.5 * G), dim=1)
    b = torch.randn(U, generator=g) * 0.5
    return b, W, e_hat


def primal_at(c, b, W, e_hat, kT):
    """P(c) with Phi from bisection: independent of the solver's dual variables."""
    nu = ens.potential_for_charge(e_hat, STATES, c, kT)
    _, phi = ens.conjugate_at(e_hat, STATES, nu, kT)
    return phi.sum() + b @ c + 0.5 * c @ (W @ c), nu


@pytest.mark.parametrize("U,Q,seed", [(4, 0.0, 1), (6, 1.0, 2), (7, -2.0, 3), (5, 0.37, 4)])
def test_kkt_conditions_hold_with_an_independent_phi_prime(U, Q, seed):
    b, W, e_hat = problem(U, seed)
    kT = 0.1
    sol = differentiable_solution(b, W, e_hat, STATES, Q, kT)
    c = sol.charge.detach()
    assert abs(float(c.sum()) - Q) <= 1e-11
    nu = ens.potential_for_charge(e_hat, STATES, c, kT)        # Phi'(c), by bisection
    stationarity = nu + b + W @ c - sol.mu_e.detach()
    # |dnu| <= |dc| / X: the solve's charge residual over the smallest susceptibility
    X = ens.susceptibility(e_hat, STATES, nu, kT)
    tol = 10.0 * (1e-11 / float(X.min()) + 1e-12)
    assert float(stationarity.abs().max()) <= tol
    assert abs(sol.duality_gap) <= 1e-10 * (1.0 + abs(float(sol.energy)))
    value, _ = primal_at(c, b, W, e_hat, kT)
    assert abs(float(value) - float(sol.energy)) <= 1e-10 * (1.0 + abs(float(sol.energy)))


def test_the_optimum_is_the_global_minimum_on_the_constraint_plane():
    b, W, e_hat = problem(5, 7)
    kT = 0.1
    sol = differentiable_solution(b, W, e_hat, STATES, 0.5, kT)
    c = sol.charge.detach()
    best, _ = primal_at(c, b, W, e_hat, kT)
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        d = torch.randn(5, generator=g)
        d = d - d.mean()
        for eps in (1e-1, 1e-2, 1e-3):
            trial = c + eps * d / d.norm()
            if bool((trial.abs() < 1.0).all()):
                value, _ = primal_at(trial, b, W, e_hat, kT)
                assert float(value) >= float(best) - 1e-12


def test_envelope_derivatives_are_exact():
    """dE/db = c, dE/dW = c c^T / 2, dE/dEhat = p: the derivative of a minimum is explicit (T4)."""
    b, W, e_hat = problem(5, 11)
    b.requires_grad_(True)
    W.requires_grad_(True)
    e_hat.requires_grad_(True)
    sol = differentiable_solution(b, W, e_hat, STATES, -0.6, 0.1)
    gb, gW, ge = torch.autograd.grad(sol.energy, (b, W, e_hat))
    c, p = sol.charge.detach(), sol.populations.detach()
    assert torch.allclose(gb, c, atol=1e-12, rtol=0.0)
    assert torch.allclose(gW, 0.5 * torch.outer(c, c), atol=1e-12, rtol=0.0)
    assert torch.allclose(ge, p, atol=1e-12, rtol=0.0)


def test_implicit_jacobian_is_the_linear_response():
    """dc/db = -(M^-1 - M^-1 1 1^T M^-1 / 1^T M^-1 1),  M = W + diag(1/X)  (T5)."""
    b, W, e_hat = problem(6, 13)
    kT = 0.1
    b = b.requires_grad_(True)
    sol = differentiable_solution(b, W, e_hat, STATES, 0.8, kT)
    jac = torch.stack([torch.autograd.grad(sol.charge[u], b, retain_graph=True)[0] for u in range(6)])
    nu = ens.potential_for_charge(e_hat, STATES, sol.charge.detach(), kT)
    X = ens.susceptibility(e_hat, STATES, nu, kT)
    Minv = torch.linalg.inv(W + torch.diag(1.0 / X))
    one = torch.ones(6)
    v = Minv @ one
    expected = -(Minv - torch.outer(v, v) / (one @ v))
    assert torch.allclose(jac, expected, atol=1e-9, rtol=1e-9)
    # and the Hessian of the optimal value in b is the same matrix (second order)
    (gb,) = torch.autograd.grad(sol.energy, b, create_graph=True)
    hess = torch.stack([torch.autograd.grad(gb[u], b, retain_graph=True)[0] for u in range(6)])
    assert torch.allclose(hess, expected, atol=1e-9, rtol=1e-9)


def _golden(f, lo, hi, iters=200):
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
        if b - a < 1e-13:
            break
    return 0.5 * (a + b)


def test_two_units_against_a_brute_force_primal_minimisation():
    b, W, e_hat = problem(2, 17)
    kT = 0.08
    sol = differentiable_solution(b, W, e_hat, STATES, 0.0, kT)

    def F(delta):
        c = torch.tensor([delta, -delta])
        return float(primal_at(c, b, W, e_hat, kT)[0])

    delta = _golden(F, -1.0 + 1e-9, 1.0 - 1e-9)
    assert abs(float(sol.charge[0]) - delta) <= 1e-6
    assert abs(float(sol.energy) - F(delta)) <= 1e-11


@pytest.mark.parametrize("bias", [0.0, 2.0, 3.6, 5.0])
def test_two_site_transfer_obeys_the_zero_temperature_formula(bias):
    """SPEC 25.4 (25.3): delta_0 = clip(-(Ehat_A+ + Ehat_B- + b_A - b_B) / Jt, 0, 1).

    With Phi_T within kT ln 3 of the envelope for each unit, the finite-T
    optimum obeys |delta_T - delta_0| <= sqrt(2 eps / Jt), eps = 2 kT ln 3
    (strong convexity of F_0 with curvature Jt).
    """
    W = torch.tensor([[6.0, 1.5], [1.5, 5.0]])
    jt = float(W[0, 0] + W[1, 1] - 2 * W[0, 1])
    # A gives an electron to B; the integer transfer costs Ehat_A+ + Ehat_B- + b_A - b_B
    e_hat = torch.tensor([[3.0, 0.0, 1.0], [-2.0, 0.0, 4.0]])  # Ehat_A+ = 1, Ehat_B- = -2
    b = torch.tensor([bias, 0.0]) * 0.5 - 0.5 * bias
    slope0 = float(e_hat[0, 2] + e_hat[1, 0] + b[0] - b[1])
    delta0 = min(max(-slope0 / jt, 0.0), 1.0)
    for kT in (0.02, 0.005, 0.001):
        sol = differentiable_solution(b, W, e_hat, STATES, 0.0, kT)
        bound = math.sqrt(2.0 * 2.0 * kT * math.log(3.0) / jt)
        assert abs(float(sol.charge[0]) - delta0) <= bound


def test_refusals():
    b, W, e_hat = problem(3, 19)
    with pytest.raises(ValueError):
        solve_free_charge(b, W, e_hat, STATES, 3.0, 0.1)          # needs every unit at +1: not interior
    with pytest.raises(ValueError):
        solve_free_charge(b, W, e_hat, STATES, 0.0, 0.0)
    with pytest.raises(FreeChargeNotConvergedError):
        solve_free_charge(b, W, e_hat, STATES, 0.3, 0.1, max_iter=1, tol=1e-15)


def test_primal_value_helper_matches_the_dual_at_the_optimum():
    b, W, e_hat = problem(4, 23)
    sol = differentiable_solution(b, W, e_hat, STATES, 1.2, 0.1)
    p = primal_value(sol.charge, sol.nu, b, W, e_hat, STATES, 0.1)
    assert abs(float(p - sol.energy)) <= 1e-10 * (1 + abs(float(sol.energy)))


@pytest.mark.parametrize("kT", [0.1, 0.001])
def test_dead_zone_with_underflowed_susceptibilities_is_finite_and_exact(kT):
    """Gaps of 100-400 eV: every x_u underflows to 0 at kT = 1e-3 (e^-50000).

    The mu row of the Newton matrix is then exactly singular; the solve must
    hold mu, return c = 0, and give finite first and second derivatives.  The
    energy lies in [-U kT ln 3, 0] (entropy only) -- and is 0 to round-off
    once the excited populations underflow.
    """
    U = 4
    b, W, _ = problem(U, 29)
    G = torch.tensor([200.0, 300.0, 400.0, 250.0])
    chi = torch.tensor([10.0, -20.0, 5.0, 0.0])
    e_hat = torch.stack((-chi + 0.5 * G, torch.zeros(U), chi + 0.5 * G), dim=1).requires_grad_(True)
    b = b.requires_grad_(True)
    sol = differentiable_solution(b, W, e_hat, STATES, 0.0, kT)
    assert float(sol.charge.abs().max()) <= 1e-12
    assert -U * kT * math.log(3.0) - 1e-12 <= float(sol.energy) <= 1e-12
    (gb,) = torch.autograd.grad(sol.energy, b, create_graph=True)
    hess = torch.stack([torch.autograd.grad(gb[u], b, retain_graph=True)[0] for u in range(U)])
    assert bool(torch.isfinite(gb).all()) and bool(torch.isfinite(hess).all())
    assert float(hess.abs().max()) <= 1e-12


def _identical_molecules(seed, units=3, singular=False):
    """Two identical, uncoupled molecules: every eigenvalue of the Newton operators is double.

    ``singular=True`` makes each block of W rank one, so A = W + W X W is singular
    and the single-constraint step takes its pseudo-inverse branch.
    """
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(units, units, generator=g)
    if singular:
        w = torch.randn(units, generator=g)
        block = torch.outer(w, w)
    else:
        block = M @ M.T / units + 0.5 * torch.eye(units)
    W = torch.block_diag(block, block)
    b = (0.5 * torch.randn(units, generator=g)).repeat(2)
    chi = 1.5 * torch.randn(units, generator=g)
    G = 2.0 * torch.rand(units, generator=g) + 0.3
    e_hat = torch.stack((-chi + 0.5 * G, torch.zeros(units), chi + 0.5 * G), dim=1).repeat(2, 1)
    group = torch.tensor([0] * units + [1] * units)
    return b, W, e_hat, group


def _constrained_response(W, X, G):
    """dc/db of the molecule-constrained optimum: -(M - M G^T (G M G^T)^+ G M), M = (W + X^-1)^-1.

    Closed form, independent of the solver; written with X^1/2 so that an underflowed
    susceptibility gives 0, not 0 * inf.
    """
    r = X.sqrt()
    M = r.unsqueeze(1) * torch.linalg.solve(torch.eye(len(X)) + r.unsqueeze(1) * W * r.unsqueeze(0), torch.diag(r))
    M = 0.5 * (M + M.T)
    C = G @ M @ G.T
    return -(M - M @ G.T @ torch.linalg.pinv(C, hermitian=True) @ G @ M)


@pytest.mark.parametrize("N", [(0.0, 0.0), (1.0, 1.0)])
@pytest.mark.parametrize("kT", [0.05, 0.1])
def test_identical_molecules_give_exact_finite_derivatives(N, kT):
    """Two identical molecules make the Schur complement on mu exactly degenerate.

    The energy is smooth there, so its derivatives must be finite and exact: the envelope
    identities dE/db = c, dE/dW = c c^T / 2, dE/dEhat = p (the constraints do not depend
    on b, W, Ehat), and the Hessian in b equal to the closed-form constrained response.
    An eigh backward through the degenerate spectrum gives 0/0 here instead.
    """
    from trace_redox.free_solver import _membership, differentiable_constrained_solution

    b, W, e_hat, group = _identical_molecules(5)
    b, W, e_hat = b.requires_grad_(True), W.requires_grad_(True), e_hat.requires_grad_(True)
    sol = differentiable_constrained_solution(b, W, e_hat, STATES, group, torch.tensor(N), kT)
    gb, gW, ge = torch.autograd.grad(sol.energy, (b, W, e_hat), create_graph=True)
    c, p = sol.charge.detach(), sol.populations.detach()
    tol = 1e4 * EPS
    assert torch.allclose(gb.detach(), c, atol=tol, rtol=0.0)
    assert torch.allclose(gW.detach(), 0.5 * torch.outer(c, c), atol=tol, rtol=0.0)
    assert torch.allclose(ge.detach(), p, atol=tol, rtol=0.0)
    hess = torch.stack([torch.autograd.grad(gb[u], b, retain_graph=True)[0] for u in range(len(b))])
    nu = sol.nu.detach()
    X = ens.susceptibility(e_hat.detach(), STATES, nu, kT)
    expected = _constrained_response(W.detach(), X, _membership(group, 2, torch.float64, "cpu"))
    assert torch.allclose(hess, expected, atol=1e-9, rtol=1e-9)


def test_singular_w_with_identical_blocks_gives_exact_finite_derivatives():
    """The pseudo-inverse branch of the single-constraint step, at a doubly degenerate spectrum."""
    b, W, e_hat, _ = _identical_molecules(7, singular=True)
    kT = 0.1
    b, W, e_hat = b.requires_grad_(True), W.requires_grad_(True), e_hat.requires_grad_(True)
    sol = differentiable_solution(b, W, e_hat, STATES, 0.0, kT)
    X = ens.susceptibility(e_hat.detach(), STATES, sol.nu.detach(), kT)
    A = W.detach() + W.detach() @ (X.unsqueeze(1) * W.detach())
    assert int(torch.linalg.cholesky_ex(A).info) != 0, "the test must exercise the pseudo-inverse branch"
    gb, gW, ge = torch.autograd.grad(sol.energy, (b, W, e_hat))
    c, p = sol.charge.detach(), sol.populations.detach()
    tol = 1e4 * EPS
    assert torch.allclose(gb, c, atol=tol, rtol=0.0)
    assert torch.allclose(gW, 0.5 * torch.outer(c, c), atol=tol, rtol=0.0)
    assert torch.allclose(ge, p, atol=tol, rtol=0.0)
