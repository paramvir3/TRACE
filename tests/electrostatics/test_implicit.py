"""The converged, implicitly differentiated solve (SPEC 17, ``implicit.py``).

Checked against a closed form that autograd differentiates on its own -- the
constrained minimiser written with ``torch.linalg.solve`` on the free block --
for the value, and the first, second and THIRD derivatives, with a mask, with a
preconditioner, and for a matrix-free factory that reads several tensors.
"""

from __future__ import annotations

import pytest
import torch

from conftest import require

torch.set_default_dtype(torch.float64)


def _problem_inputs(seed=0, n=9):
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(4, generator=g, dtype=torch.float64)
    base = torch.randn(n, n, generator=g, dtype=torch.float64)
    bend = torch.randn(n, n, generator=g, dtype=torch.float64)
    c0 = torch.randn(n, generator=g, dtype=torch.float64)
    c1 = torch.randn(n, generator=g, dtype=torch.float64)
    mask = torch.ones(n, dtype=torch.float64)
    mask[[2, 5]] = 0.0
    return theta, base, bend, c0, c1, mask


def _operator(theta, base, bend):
    """A smooth SPD family K(theta)."""
    a = base + torch.tanh(theta[0]) * bend + theta[1] * theta[2] * torch.eye(base.shape[0])
    return a @ a.T + (2.0 + theta[3] ** 2) * torch.eye(base.shape[0])


def _linear(theta, c0, c1):
    return c0 * torch.cos(theta[0]) + c1 * theta[1] ** 2 + theta[2] * theta[3]


def _closed_form(theta, base, bend, c0, c1, mask):
    K = _operator(theta, base, bend)
    c = _linear(theta, c0, c1)
    free = mask.nonzero(as_tuple=True)[0]
    x_free = -torch.linalg.solve(K[free][:, free], c[free])
    x = torch.zeros_like(c).index_put((free,), x_free)
    return x, 0.5 * x @ K @ x + c @ x


def _implicit(theta, base, bend, c0, c1, mask, precondition=None):
    QuadraticProblem, minimise = require("implicit", "QuadraticProblem", "minimise")
    K = _operator(theta, base, bend)
    c = _linear(theta, c0, c1)
    problem = QuadraticProblem(
        lambda T: (lambda v: T[0] @ v), (K,), mask=mask, precondition=precondition, rtol=1e-13
    )
    return minimise(problem, c)


@pytest.mark.parametrize("preconditioned", [False, True])
def test_value_and_derivatives_to_third_order(preconditioned):
    theta, base, bend, c0, c1, mask = _problem_inputs()
    precondition = None
    if preconditioned:
        diagonal = torch.diagonal(_operator(theta, base, bend)).detach()
        precondition = lambda r: r / diagonal  # noqa: E731

    def both(fn):
        t = theta.clone().requires_grad_(True)
        x, energy = fn(t, base, bend, c0, c1, mask) if fn is _closed_form else fn(
            t, base, bend, c0, c1, mask, precondition
        )
        g1 = torch.autograd.grad(energy, t, create_graph=True)[0]
        hess = torch.stack([torch.autograd.grad(g1[i], t, create_graph=True)[0] for i in range(4)])
        third = torch.autograd.grad(hess.sum(), t, retain_graph=True)[0]
        jac = torch.stack([torch.autograd.grad(x[i], t, create_graph=True)[0] for i in range(x.numel())])
        jac2 = torch.autograd.grad((jac ** 2).sum(), t)[0]
        return x.detach(), energy.detach(), g1.detach(), hess.detach(), third, jac.detach(), jac2

    reference = both(_closed_form)
    implicit = both(_implicit)
    names = ("x", "energy", "dE", "d2E", "d3E", "dx", "d2x")
    for name, a, b in zip(names, implicit, reference):
        scale = float(b.abs().max()) + 1e-30
        assert float((a - b).abs().max()) < 1e-9 * scale, (name, float((a - b).abs().max()), scale)
    assert float(implicit[0][mask == 0].abs().max()) == 0.0


def test_matrix_free_factory_reading_several_tensors():
    """A factory whose product reads two tensors -- the lattice-operator pattern."""
    QuadraticProblem, minimise = require("implicit", "QuadraticProblem", "minimise")
    theta, base, bend, c0, c1, mask = _problem_inputs(seed=3)

    def via_factory(t):
        left = base + torch.tanh(t[0]) * bend
        shift = (2.0 + t[3] ** 2) * torch.ones(base.shape[0])

        def factory(T):
            L, s = T
            return lambda v: L @ (L.T @ v) + s * v

        problem = QuadraticProblem(factory, (left, shift), mask=mask, rtol=1e-13)
        return minimise(problem, _linear(t, c0, c1))

    def reference(t):
        left = base + torch.tanh(t[0]) * bend
        K = left @ left.T + torch.diag((2.0 + t[3] ** 2) * torch.ones(base.shape[0]))
        c = _linear(t, c0, c1)
        free = mask.nonzero(as_tuple=True)[0]
        x_free = -torch.linalg.solve(K[free][:, free], c[free])
        x = torch.zeros_like(c).index_put((free,), x_free)
        return x, 0.5 * x @ K @ x + c @ x

    for fn_a, fn_b in ((via_factory, reference),):
        ta = theta.clone().requires_grad_(True)
        tb = theta.clone().requires_grad_(True)
        xa, ea = fn_a(ta)
        xb, eb = fn_b(tb)
        ga = torch.autograd.grad(ea, ta, create_graph=True)[0]
        gb = torch.autograd.grad(eb, tb, create_graph=True)[0]
        ha = torch.autograd.grad(ga.sum() + (xa ** 2).sum(), ta)[0]
        hb = torch.autograd.grad(gb.sum() + (xb ** 2).sum(), tb)[0]
        assert torch.allclose(ea, eb, atol=1e-11, rtol=0)
        assert torch.allclose(ga, gb, atol=1e-9, rtol=1e-9)
        assert torch.allclose(ha, hb, atol=1e-8, rtol=1e-8)


def test_indefinite_operator_raises_instead_of_returning_a_saddle():
    QuadraticProblem, minimise, IndefiniteOperatorError = require(
        "implicit", "QuadraticProblem", "minimise", "IndefiniteOperatorError"
    )
    K = torch.diag(torch.tensor([2.0, 1.0, -0.5, 3.0]))
    problem = QuadraticProblem(lambda T: (lambda v: T[0] @ v), (K,))
    with pytest.raises(IndefiniteOperatorError):
        minimise(problem, torch.tensor([1.0, 1.0, 1.0, 1.0]))


def test_unconverged_solve_raises():
    QuadraticProblem, minimise, SolverNotConvergedError = require(
        "implicit", "QuadraticProblem", "minimise", "SolverNotConvergedError"
    )
    g = torch.Generator().manual_seed(1)
    a = torch.randn(40, 40, generator=g)
    K = a @ a.T + 1e-3 * torch.eye(40)
    problem = QuadraticProblem(lambda T: (lambda v: T[0] @ v), (K,), rtol=1e-12, max_iter=3)
    with pytest.raises(SolverNotConvergedError):
        minimise(problem, torch.randn(40, generator=g))


def test_zero_linear_term_gives_zero_and_finite_derivatives():
    QuadraticProblem, minimise = require("implicit", "QuadraticProblem", "minimise")
    t = torch.tensor([0.3], requires_grad=True)
    K = torch.eye(3) * (2.0 + t)
    problem = QuadraticProblem(lambda T: (lambda v: T[0] @ v), (K,))
    x, energy = minimise(problem, torch.zeros(3) * t)
    grad = torch.autograd.grad(energy, t, create_graph=True)[0]
    assert float(energy) == 0.0 and float(x.abs().max()) == 0.0
    assert torch.isfinite(grad).all()


def test_converged_forces_are_conservative_around_a_closed_loop():
    """SPEC 17.2: the work of the forces around a closed loop vanishes.

    Integrator-free: trapezoid rule on a smooth periodic path (spectrally
    accurate), 8-site cluster with a real charge contrast. The fixed-k energy
    is exactly conservative, so its loop work is the quadrature floor; the
    converged forces must sit at that floor (measured -1.2e-9 vs -1.3e-9 eV on
    16 sites at 400 points; 1.2e-7 at rtol 1e-6, 2.2e-5 at rtol 1e-3).
    """
    import math

    solver = require("solver", "ElectrostaticParams", "Geometry", "FragmentConstraints",
                     "SolverOptions", "solve_charges", "GeometrySeed")
    ElectrostaticParams, Geometry, FragmentConstraints, SolverOptions, solve_charges, GeometrySeed = solver
    build_incidence = require("topology", "build_incidence")
    hardness_floor = require("kernels", "hardness_floor")
    bond_envelope = require("energy", "bond_envelope")
    g = torch.Generator().manual_seed(11)
    n = 8
    R0 = torch.tensor([[x, y, z] for x in range(2) for y in range(2) for z in range(2)],
                      dtype=torch.float64) * 2.2 + 0.1 * torch.randn(n, 3, generator=g)
    A, B = 0.25 * torch.randn(n, 3, generator=g), 0.25 * torch.randn(n, 3, generator=g)
    sigma = torch.full((n,), 1.2)
    rows, cols = torch.triu_indices(n, n, 1)
    incidence, bonds = build_incidence(torch.stack([rows, cols]), n)

    def force(p, mode, rtol):
        p = p.clone().requires_grad_(True)
        f = bond_envelope((p[bonds[0]] - p[bonds[1]]).norm(dim=-1), 4.0, 0.5)
        params = ElectrostaticParams(
            chi=torch.tensor([2.0 if i % 2 else -2.0 for i in range(n)]),
            eta=hardness_floor(sigma) + 2.0, sigma=sigma, ell=torch.ones(n),
            kappa=torch.ones(bonds.shape[1]), A_mu=torch.eye(3).repeat(n, 1, 1) * 20.0,
        )
        result = solve_charges(
            params, Geometry(positions=p, incidence=incidence * f, bond_index=bonds),
            FragmentConstraints(matrix=torch.ones(1, n), targets=torch.zeros(1)), 6,
            seed=GeometrySeed.zeros(int(bonds.shape[1]), n),
            options=SolverOptions(solve_mode=mode, solve_rtol=rtol),
        )
        return -torch.autograd.grad(result.energy, p)[0]

    def loop_work(mode, rtol, points=120):
        work = 0.0
        for m in range(points):
            s = 2.0 * math.pi * m / points
            R = R0 + A * math.sin(s) + B * (1.0 - math.cos(s))
            dR = (A * math.cos(s) + B * math.sin(s)) * 2.0 * math.pi
            work += float((force(R, mode, rtol) * dR).sum()) / points
        return work

    floor = abs(loop_work("fixed_k", 1e-10))
    converged = abs(loop_work("converged", 1e-10))
    loose = abs(loop_work("converged", 1e-3))
    assert converged < 10.0 * floor + 1e-8, (converged, floor)
    assert loose > 10.0 * converged, "the loop work must resolve a loose tolerance"
