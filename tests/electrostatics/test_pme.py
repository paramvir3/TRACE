"""Smooth particle-mesh Ewald for the reciprocal part of the lattice operator.

``trace_electrostatics.pme`` replaces the chunked direct k-sum of
``LatticeOperator.apply`` by SPME with multipoles spread through B-spline
derivatives.  Asserted here:

* the reference used below -- an independent full-space direct sum written in
  this file -- IS what ``_reciprocal_chunk`` accumulates over the operator's own
  k-set, and PME equals that operator sum to the mesh accuracy (1:1 swap);
* relative RMS error of phi, field and grad (Cartesian and projected on the 2e
  basis) below the requested accuracy for 1e-4, 1e-6, 1e-8, with the mesh from
  ``pme_mesh``, for q only, mu only, Theta only and all together;
* convergence with mesh (rate p - 2 for quadrupoles) and with order;
* the operator is exactly symmetric and ``energy`` is its quadratic form;
* forces, stress and double backward are exact derivatives (finite differences,
  ``gradcheck`` / ``gradgradcheck``);
* invariance: exact under a mesh-vector translation and a lattice wrap of one
  atom, to the accuracy under a general translation;
* charged cells: k = 0 excluded exactly (zero mean of the mesh potential), and
  the outputs match the exact sum, which excludes k = 0 the same way.

Every tensor is created in float64 explicitly: module-level caches run before
the suite's autouse dtype fixture would.
"""

from __future__ import annotations

import functools
import math

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from conftest import require

F64 = torch.float64
K_E = 14.399645


# ---------------------------------------------------------------------------
# Independent reference: the full-space direct sum, complex arithmetic
# ---------------------------------------------------------------------------
def exact_reciprocal(pos, cell, q, mu, theta, alpha, tail=1e-20, chunk=2000):
    """``(phi, field, grad)`` of the reciprocal sum over every k != 0 with
    ``exp(-k^2/4 alpha^2) > tail`` -- untruncated to double precision."""
    recip = 2.0 * math.pi * torch.linalg.inv(cell).T
    k_cut = 2.0 * alpha * math.sqrt(-math.log(tail))
    n_max = [math.ceil(k_cut * float(cell[a].norm()) / (2.0 * math.pi)) for a in range(3)]
    grid = torch.cartesian_prod(*[torch.arange(-m, m + 1, dtype=F64) for m in n_max])
    k = grid @ recip
    k2 = (k * k).sum(-1)
    keep = (grid.abs().sum(-1) > 0) & (k2 <= k_cut * k_cut)
    k, k2 = k[keep], k2[keep]
    g = 4.0 * math.pi * K_E / torch.det(cell).abs() * torch.exp(-k2 / (4.0 * alpha**2)) / k2
    n = pos.shape[0]
    phi = torch.zeros(n, dtype=F64)
    field = torch.zeros(n, 3, dtype=F64)
    grad = torch.zeros(n, 3, 3, dtype=F64)
    for s in range(0, k.shape[0], chunk):
        kk, gg = k[s : s + chunk], g[s : s + chunk]
        moment = (q.unsqueeze(-1).to(torch.complex128) - 1j * (mu @ kk.T)
                  - 0.5 * torch.einsum("ka,iab,kb->ik", kk, theta, kk))
        phase = torch.exp(1j * (pos @ kk.T))                       # e^{ik.r_i}
        z = phase * (phase.conj() * moment).sum(0)                 # e^{ik.r_i} S(k)
        phi = phi + (z.real * gg).sum(-1)
        field = field + ((1j * z).real * gg) @ kk
        grad = grad - 0.5 * torch.einsum("ik,ka,kb->iab", z.real * gg, kk, kk)
    return phi, field, grad


def _traceless(a):
    s = a + a.transpose(-1, -2)
    return s - torch.eye(3, dtype=F64) * s.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] / 3.0


def _system(n=30, seed=0, cell=None, neutral=True):
    g = torch.Generator().manual_seed(seed)
    if cell is None:
        cell = torch.tensor([[9.0, 0.0, 0.0], [1.6, 8.4, 0.0], [-1.1, 1.3, 9.6]], dtype=F64)
    pos = torch.rand(n, 3, generator=g, dtype=F64) @ cell
    q = torch.randn(n, generator=g, dtype=F64)
    if neutral:
        q = q - q.mean()
    mu = 0.3 * torch.randn(n, 3, generator=g, dtype=F64)
    theta = _traceless(0.2 * torch.randn(n, 3, 3, generator=g, dtype=F64))
    return pos, cell, q, mu, theta


ALPHA = 0.5


@functools.lru_cache(maxsize=None)
def _reference(source: str):
    pos, cell, q, mu, theta = _system()
    zq, zm, zt = torch.zeros_like(q), torch.zeros_like(mu), torch.zeros_like(theta)
    moments = {"q": (q, zm, zt), "mu": (zq, mu, zt), "theta": (zq, zm, theta), "all": (q, mu, theta)}[source]
    return (pos, cell) + moments, exact_reciprocal(pos, cell, *moments, ALPHA)


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def _basis():
    return require("kernels", "quadrupole_cartesian_basis")().to(F64)


def _energy(outputs, q, mu, theta):
    phi, field, grad = outputs
    return 0.5 * ((q * phi).sum() + (mu * field).sum() + (theta * grad).sum())


# ---------------------------------------------------------------------------
# The reference is the operator's own sum, and PME replaces it 1:1
# ---------------------------------------------------------------------------
def test_pme_replaces_the_lattice_operator_reciprocal_sum():
    """Measured: reference vs ``_reciprocal_chunk`` over the operator's k-set
    5e-15 / 6e-14 / 3.6e-13 (phi / field / grad); PME with a 1e-10 mesh vs the
    same sum 6e-13 (phi), 2e-12 (field), 1.7e-11 (projected grad)."""
    LatticeOperator, lattice_pairs, chunk = require(
        "lattice_operator", "LatticeOperator", "lattice_pairs", "_reciprocal_chunk")
    EwaldParams = require("ewald", "EwaldParams")
    PMEReciprocal, pme_mesh = require("pme", "PMEReciprocal", "pme_mesh")
    pos, cell, q, mu, theta = _system(n=12, seed=3)
    n = pos.shape[0]
    ewald = EwaldParams.auto_tune(cell, 9.0, accuracy=1e-14)
    widths = torch.full((n,), 0.9, dtype=F64)
    op = LatticeOperator(pos, cell, ewald, lattice_pairs(pos, cell, 9.0), sigma=widths,
                         kernel="gaussian", sigma_mu=widths, sigma_theta=widths)
    basis = _basis()
    theta5 = torch.einsum("iab,mab->im", theta, basis)          # the solver's variables
    th_c = torch.einsum("im,mab->iab", theta5, basis)
    direct = chunk(op.pos, q, mu, th_c, op.k, op.weight)
    reference = exact_reciprocal(pos, cell, q, mu, th_c, ewald.alpha)
    # The operator truncates at exp(-k^2/4 alpha^2) = 1e-14; grad weights that
    # tail by k^2 (measured 1e-15, 1e-14, 3.6e-13 for phi, field, grad).
    for got, want in zip(reference, direct):
        assert _rel(got, want) < 1e-12

    pme = PMEReciprocal(pos, cell, ewald.alpha, pme_mesh(cell, ewald.alpha, 1e-10, 10), 10)
    phi, field, grad = pme.apply(q, mu, th_c)
    assert _rel(phi, direct[0]) < 1e-10
    assert _rel(field, direct[1]) < 1e-10
    project = lambda m: torch.einsum("iab,mab->im", m, basis)    # noqa: E731  (what apply does)
    assert _rel(project(grad), project(direct[2])) < 1e-10


# ---------------------------------------------------------------------------
# Accuracy against the exact sum, per moment type
# ---------------------------------------------------------------------------
CASES = [(1e-4, 6), (1e-4, 8), (1e-6, 6), (1e-6, 8), (1e-6, 10), (1e-8, 8), (1e-8, 10)]


@pytest.mark.parametrize("accuracy,order", CASES)
@pytest.mark.parametrize("source", ["q", "mu", "theta", "all"])
def test_relative_rms_error_meets_the_requested_accuracy(accuracy, order, source):
    """Worst measured error / accuracy over these 28 cases: 0.38 (rank-2 rule)."""
    PMEReciprocal, pme_mesh = require("pme", "PMEReciprocal", "pme_mesh")
    (pos, cell, q, mu, theta), exact = _reference(source)
    mesh = pme_mesh(cell, ALPHA, accuracy, order)
    phi, field, grad = PMEReciprocal(pos, cell, ALPHA, mesh, order).apply(q, mu, theta)
    basis = _basis()
    project = lambda m: torch.einsum("iab,mab->im", m, basis)    # noqa: E731
    errors = [_rel(phi, exact[0]), _rel(field, exact[1]), _rel(grad, exact[2]),
              _rel(project(grad), project(exact[2]))]
    assert max(errors) < accuracy, (mesh, errors)


def test_rank_one_mesh_is_enough_without_quadrupoles():
    """``max_rank=1`` meshes are coarser and still meet the accuracy for q, mu."""
    PMEReciprocal, pme_mesh = require("pme", "PMEReciprocal", "pme_mesh")
    (pos, cell, q, mu, _), _ = _reference("all")
    exact = exact_reciprocal(pos, cell, q, mu, torch.zeros(len(q), 3, 3, dtype=F64), ALPHA)
    for accuracy in (1e-6, 1e-8):
        mesh1, mesh2 = pme_mesh(cell, ALPHA, accuracy, 8, max_rank=1), pme_mesh(cell, ALPHA, accuracy, 8)
        assert all(a <= b for a, b in zip(mesh1, mesh2)) and mesh1 != mesh2
        phi, field, grad = PMEReciprocal(pos, cell, ALPHA, mesh1, 8, quadrupoles=False).apply(q, mu)
        assert grad is None
        assert max(_rel(phi, exact[0]), _rel(field, exact[1])) < accuracy


def test_convergence_with_mesh_and_order():
    """Quadrupolar outputs converge as h^(p-2): measured local rates 6.9, 6.9,
    6.4 at p = 8 (K = 16..48); at K = 24 every order 5..10 improves on the one
    before (3.2e-4 ... 1.6e-8)."""
    PMEReciprocal = require("pme", "PMEReciprocal")
    (pos, cell, q, mu, theta), exact = _reference("all")

    def worst(mesh, order):
        out = PMEReciprocal(pos, cell, ALPHA, mesh, order).apply(q, mu, theta)
        return max(_rel(a, b) for a, b in zip(out, exact))

    by_mesh = [worst(k, 8) for k in (12, 16, 24, 32, 48)]
    assert all(b < a for a, b in zip(by_mesh, by_mesh[1:]))
    assert by_mesh[1] / by_mesh[3] > 2.0 ** (8 - 2 - 0.5)        # 16 -> 32
    by_order = [worst(24, p) for p in range(5, 11)]
    assert all(b < a for a, b in zip(by_order, by_order[1:]))
    assert by_order[-1] < 1e-7 < by_order[0]


@pytest.mark.parametrize("order", [4, 5, 7, 9])
def test_low_and_odd_orders(order):
    """Odd orders (spline modulus zero at Nyquist) and p = 4 work; p = 4 with
    quadrupoles is refused -- its energy has kinks at the mesh planes."""
    PMEReciprocal = require("pme", "PMEReciprocal")
    (pos, cell, q, mu, theta), exact = _reference("all")
    if order == 4:
        with pytest.raises(ValueError):
            PMEReciprocal(pos, cell, ALPHA, 32, order)
        ex = exact_reciprocal(pos, cell, q, mu, torch.zeros_like(theta), ALPHA)
        phi, field, _ = PMEReciprocal(pos, cell, ALPHA, 48, order, quadrupoles=False).apply(q, mu)
        assert max(_rel(phi, ex[0]), _rel(field, ex[1])) < 1e-3
        return
    out = PMEReciprocal(pos, cell, ALPHA, 32, order).apply(q, mu, theta)
    assert max(_rel(a, b) for a, b in zip(out, exact)) < {5: 1e-3, 7: 1e-5, 9: 1e-7}[order]


# ---------------------------------------------------------------------------
# Structure: symmetry, energy
# ---------------------------------------------------------------------------
def test_operator_is_symmetric_and_energy_is_its_quadratic_form():
    PMEReciprocal = require("pme", "PMEReciprocal")
    pos, cell, q1, mu1, th1 = _system(seed=4)
    _, _, q2, mu2, th2 = _system(seed=5)
    op = PMEReciprocal(pos, cell, ALPHA, (20, 24, 21), 7)
    a = op.apply(q1, mu1, th1)
    b = op.apply(q2, mu2, th2)
    ab = (q2 * a[0]).sum() + (mu2 * a[1]).sum() + (th2 * a[2]).sum()
    ba = (q1 * b[0]).sum() + (mu1 * b[1]).sum() + (th1 * b[2]).sum()
    assert abs(float(ab - ba)) < 1e-13 * abs(float(ab))
    energy = op.energy(q1, mu1, th1)
    assert abs(float(energy - _energy(a, q1, mu1, th1))) < 1e-13 * abs(float(energy))


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------
def test_forces_match_finite_differences_and_the_exact_forces():
    """Autograd forces vs central differences: 7e-10 relative.  Vs the exact
    sum's forces at a 1e-8 mesh: 1.2e-8 relative."""
    PMEReciprocal, pme_mesh = require("pme", "PMEReciprocal", "pme_mesh")
    pos, cell, q, mu, theta = _system(n=10, seed=6)
    mesh = pme_mesh(cell, ALPHA, 1e-8, 8)

    def energy(p):
        return PMEReciprocal(p, cell, ALPHA, mesh, 8).energy(q, mu, theta)

    p0 = pos.clone().requires_grad_(True)
    force = -torch.autograd.grad(energy(p0), p0)[0]
    h = 1e-5
    numeric = torch.zeros_like(pos)
    for i in range(pos.shape[0]):
        for a in range(3):
            step = torch.zeros_like(pos)
            step[i, a] = h
            numeric[i, a] = -(energy(pos + step) - energy(pos - step)) / (2 * h)
    assert _rel(force, numeric) < 1e-8

    p1 = pos.clone().requires_grad_(True)
    exact = -torch.autograd.grad(_energy(exact_reciprocal(p1, cell, q, mu, theta, ALPHA), q, mu, theta), p1)[0]
    assert _rel(force, exact) < 1e-7


def test_gradcheck_and_gradgradcheck_in_every_input():
    """Positions, cell (stress), q, mu and Theta; first and second derivatives
    against finite differences (torch's own checkers)."""
    PMEReciprocal = require("pme", "PMEReciprocal")
    pos, cell, q, mu, theta = _system(n=3, seed=7, cell=torch.tensor(
        [[5.0, 0.0, 0.0], [0.7, 4.6, 0.0], [-0.4, 0.5, 5.3]], dtype=F64))

    def energy(p, c, qq, mm, tt):
        return PMEReciprocal(p, c, ALPHA, (9, 10, 11), 6).energy(qq, mm, tt)

    inputs = tuple(x.clone().requires_grad_(True) for x in (pos, cell, q, mu, theta))
    assert torch.autograd.gradcheck(energy, inputs)
    assert torch.autograd.gradgradcheck(energy, inputs)


def test_training_double_backward_matches_finite_differences():
    """d/d(lambda) of a force loss, moments = m0 + lambda dm: the path training
    takes (forces differentiated w.r.t. parameters).  Through ``apply``."""
    PMEReciprocal = require("pme", "PMEReciprocal")
    pos, cell, q, mu, theta = _system(n=8, seed=8)
    _, _, dq, dmu, dth = _system(n=8, seed=9)
    weights = torch.randn(8, 3, generator=torch.Generator().manual_seed(1), dtype=F64)

    def loss(lam, create_graph):
        p = pos.clone().requires_grad_(True)
        qq, mm, tt = q + lam * dq, mu + lam * dmu, theta + lam * dth
        out = PMEReciprocal(p, cell, ALPHA, 24, 8).apply(qq, mm, tt)
        force = -torch.autograd.grad(_energy(out, qq, mm, tt), p, create_graph=create_graph)[0]
        return (weights * force).sum()

    lam = torch.tensor(0.3, dtype=F64, requires_grad=True)
    analytic = torch.autograd.grad(loss(lam, True), lam)[0]
    h = 1e-5
    numeric = (loss(torch.tensor(0.3 + h, dtype=F64), False) - loss(torch.tensor(0.3 - h, dtype=F64), False)) / (2 * h)
    assert abs(float(analytic - numeric)) < 1e-7 * abs(float(numeric))


def test_checkpointed_apply_is_identical():
    """For the CG wiring: ``checkpoint(op.apply, ...)`` gives the same values
    and the same gradients."""
    PMEReciprocal = require("pme", "PMEReciprocal")
    pos, cell, q, mu, theta = _system(n=8, seed=10)
    grads = []
    for wrapped in (False, True):
        p = pos.clone().requires_grad_(True)
        op = PMEReciprocal(p, cell, ALPHA, 20, 8)
        out = checkpoint(op.apply, q, mu, theta, use_reentrant=False) if wrapped else op.apply(q, mu, theta)
        grads.append((_energy(out, q, mu, theta), torch.autograd.grad(_energy(out, q, mu, theta), p)[0]))
    assert float((grads[0][0] - grads[1][0]).abs()) == 0.0
    assert torch.equal(grads[0][1], grads[1][1])


# ---------------------------------------------------------------------------
# Invariances
# ---------------------------------------------------------------------------
def test_translation_and_wrapping():
    """Mesh-vector translation and a lattice wrap of one atom: round-off
    (measured <= 4e-15).  A general translation: within the accuracy (measured
    energy 1e-8, net force 3.6e-7 of the force norm at a 1e-6 mesh)."""
    PMEReciprocal, pme_mesh = require("pme", "PMEReciprocal", "pme_mesh")
    pos, cell, q, mu, theta = _system(seed=11)
    mesh = pme_mesh(cell, ALPHA, 1e-6, 8)

    def run(p):
        op = PMEReciprocal(p, cell, ALPHA, mesh, 8)
        return op.energy(q, mu, theta), op.apply(q, mu, theta)

    e0, out0 = run(pos)
    mesh_vector = cell[0] / mesh[0] + 2.0 * cell[1] / mesh[1] - cell[2] / mesh[2]
    wrapped = pos.clone()
    wrapped[4] = wrapped[4] + cell[1] - 2.0 * cell[2]
    for p in (pos + mesh_vector, wrapped):
        e1, out1 = run(p)
        assert abs(float(e1 - e0)) < 1e-13 * abs(float(e0))
        for a, b in zip(out1, out0):
            assert _rel(a, b) < 1e-12

    e2, _ = run(pos + torch.tensor([0.123, -0.311, 0.207], dtype=F64))
    assert abs(float(e2 - e0)) < 1e-6 * abs(float(e0))
    p = pos.clone().requires_grad_(True)
    force = -torch.autograd.grad(PMEReciprocal(p, cell, ALPHA, mesh, 8).energy(q, mu, theta), p)[0]
    assert float(force.sum(0).norm()) < 1e-6 * float(force.norm())


# ---------------------------------------------------------------------------
# Charged cells: k = 0
# ---------------------------------------------------------------------------
def test_charged_cell_excludes_k_zero_exactly():
    """With a net charge of 3 e: the outputs match the exact sum (which drops
    k = 0) to the accuracy, and the mesh potential has zero mean to round-off,
    probed at every mesh point by zero-moment sites (sum over mesh-point
    probes of M_p = 1, so their mean potential is the m = 0 mode, G(0) = 0).
    Measured: 1.1e-9 worst output error at a 1e-8 mesh; probe mean 5e-18 of
    the largest probe potential."""
    PMEReciprocal, pme_mesh = require("pme", "PMEReciprocal", "pme_mesh")
    pos, cell, q, mu, theta = _system(n=20, seed=12)
    q = q + 3.0 / len(q)
    assert abs(float(q.sum()) - 3.0) < 1e-12
    exact = exact_reciprocal(pos, cell, q, mu, theta, ALPHA)
    mesh = pme_mesh(cell, ALPHA, 1e-8, 8)
    out = PMEReciprocal(pos, cell, ALPHA, mesh, 8).apply(q, mu, theta)
    assert max(_rel(a, b) for a, b in zip(out, exact)) < 1e-8

    probe_mesh = (12, 10, 14)
    grid = torch.cartesian_prod(*[torch.arange(k, dtype=F64) / k for k in probe_mesh]) @ cell
    n_probe = grid.shape[0]
    all_pos = torch.cat([pos, grid])
    zeros = torch.zeros(n_probe, dtype=F64)
    phi, _, _ = PMEReciprocal(all_pos, cell, ALPHA, probe_mesh, 6).apply(
        torch.cat([q, zeros]), torch.cat([mu, torch.zeros(n_probe, 3, dtype=F64)]),
        torch.cat([theta, torch.zeros(n_probe, 3, 3, dtype=F64)]))
    probe = phi[len(q):]
    assert abs(float(probe.mean())) < 1e-13 * float(probe.abs().max())


# ---------------------------------------------------------------------------
# Mesh rule
# ---------------------------------------------------------------------------
def test_mesh_rule():
    pme_mesh, fft_friendly_size = require("pme", "pme_mesh", "fft_friendly_size")
    cell = torch.tensor([[8.0, 0.0, 0.0], [2.0, 16.0, 0.0], [0.0, 0.0, 24.0]], dtype=F64)
    previous = None
    for accuracy in (1e-4, 1e-6, 1e-8, 1e-10):
        mesh = pme_mesh(cell, 0.4, accuracy, 8)
        assert all(fft_friendly_size(k) == k and k >= 8 for k in mesh)
        assert mesh[0] < mesh[1] < mesh[2]
        if previous is not None:
            assert all(b >= a for a, b in zip(previous, mesh)) and mesh != previous
        previous = mesh
    assert pme_mesh(cell, 0.8, 1e-6, 8)[2] > pme_mesh(cell, 0.4, 1e-6, 8)[2]
    assert fft_friendly_size(97) == 98 and fft_friendly_size(121) == 125
