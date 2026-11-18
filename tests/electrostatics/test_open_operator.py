"""The matrix-free open-boundary operator (``open_operator.OpenOperator``).

Each claim the module rests on is asserted here:

* ``K v`` equals the dense open-boundary blocks (``energy._direct_pair_blocks``,
  ``_direct_quadrupole_blocks``; for Thole quadrupoles the kernels.py tensors)
  to round-off -- both kernels, with and without quadrupoles, both the
  precomputed and the memory-bounded mode, several chunkings, on clusters with
  a near-coincident, an exactly coincident and a sub-shield pair;
* K is symmetric: ``v.(K w) == w.(K v)``;
* first derivatives of the pair energy (positions and every parameter) equal
  autograd of the dense construction, and second derivatives through the
  checkpointed chunks equal both the dense ones and finite differences;
* the pair energy is invariant under rotation and translation;
* the far-field short cut is exact: ``P(n + 1/2, x) == 1.0`` beyond it;
* the operator is a drop-in for ``solve_lattice``: the whole solve equals the
  dense ``solve_charges`` (fixed-k to round-off, converged to its tolerance).
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from conftest import require

K_E = 14.399645


def _spd(n, d, floor, generator, scale=0.6):
    factor = scale * torch.randn(n, d, d, generator=generator)
    return factor @ factor.transpose(-1, -2) + floor * torch.eye(d)


def _params(n, generator):
    ElectrostaticParams = require("solver", "ElectrostaticParams")
    kernel_self = require("kernels", "kernel_self")
    sigma = 0.8 + 0.5 * torch.rand(n, generator=generator)
    return ElectrostaticParams(
        chi=torch.zeros(n), eta=kernel_self(sigma) + 1.0, sigma=sigma,
        ell=0.6 + 0.6 * torch.rand(n, generator=generator), kappa=torch.ones(0),
        A_mu=_spd(n, 3, 2.0, generator), A_Theta=_spd(n, 5, 1.0, generator),
    )


def _cluster(n=23, seed=0, near=1e-7, spread=2.0, degenerate=True):
    """Random cluster with a near-coincident, a coincident and a sub-shield pair."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g) * spread
    direction = torch.randn(3, generator=g)
    pos[3] = pos[2] + near * direction / direction.norm()
    if degenerate:
        pos[5] = pos[4]                                      # coincident
        pos[7] = pos[6] + torch.tensor([3e-13, 0.0, 0.0])    # below DEGENERATE_SEPARATION
    params = _params(n, g)
    moments = (torch.randn(n, generator=g), torch.randn(n, 3, generator=g), torch.randn(n, 5, generator=g))
    return pos, params, moments


def _operator(pos, params, kernel, quadrupoles, **kwargs):
    OpenOperator = require("open_operator", "OpenOperator")
    gaussian_widths, alpha_volume = require("solver", "gaussian_widths", "isotropic_polarizability_volume")
    if kernel == "gaussian":
        sigma_mu, sigma_theta = gaussian_widths(params, 0.5)
        return OpenOperator(pos, sigma=params.sigma, kernel="gaussian", sigma_mu=sigma_mu,
                            sigma_theta=sigma_theta if quadrupoles else None, **kwargs)
    return OpenOperator(pos, sigma=params.sigma, kernel="thole",
                        alpha_volume=alpha_volume(params.A_mu, params.ell), thole_a=0.39,
                        quadrupoles=quadrupoles, **kwargs)


def _thole_quadrupole_blocks(pos, params):
    """Reduced Thole T02/T12/T22 from the kernels.py tensors (no dense package route exists)."""
    kernels = require("kernels", "thole_damping_quadrupole", "charge_quadrupole_tensor",
                      "dipole_quadrupole_tensor", "quadrupole_interaction_tensor",
                      "quadrupole_cartesian_basis")
    damping, t02_fn, t12_fn, t22_fn, basis_fn = kernels
    alpha = require("solver", "isotropic_polarizability_volume")(params.A_mu, params.ell)
    W = basis_fn()
    n = pos.shape[0]
    r_vec = pos[:, None] - pos[None]
    squared = (r_vec * r_vec).sum(-1)
    separated = squared > 1e-24
    r = torch.sqrt(torch.where(separated, squared, torch.ones_like(squared)))
    keep = separated.to(pos.dtype) * (1.0 - torch.eye(n))
    l3, l5, l7, l9 = damping(r, alpha[:, None], alpha[None], 0.39)
    ell = params.ell
    t02 = torch.einsum("ijab,mab->ijm", t02_fn(r_vec, r, l3, l5), W) * (keep * ell[None] ** 2)[..., None]
    t12 = torch.einsum("ijabc,mbc->ijam", t12_fn(r_vec, r, l5, l7), W) * (
        keep * ell[:, None] * ell[None] ** 2)[..., None, None]
    t22 = torch.einsum("mab,ijabcd,ncd->ijmn", W, t22_fn(r_vec, r, l5, l7, l9), W) * (
        keep * ell[:, None] ** 2 * ell[None] ** 2)[..., None, None]
    return t02, t12, t22


def _dense_blocks(pos, params, kernel, quadrupoles):
    direct, direct_quadrupole = require("energy", "_direct_pair_blocks", "_direct_quadrupole_blocks")
    alpha = require("solver", "isotropic_polarizability_volume")(params.A_mu, params.ell)
    j, t01, t2 = direct(pos, params, alpha, 0.39, kernel=kernel, gaussian_a=0.5)
    blocks = dict(j=j, t01=t01, t2=t2)
    if quadrupoles:
        quad = direct_quadrupole(pos, params, 0.5) if kernel == "gaussian" else _thole_quadrupole_blocks(pos, params)
        blocks.update(t02=quad[0], t12=quad[1], t22=quad[2])
    return blocks


def _dense_product(b, q, mh, th):
    phi = b["j"] @ q + torch.einsum("ija,ja->i", b["t01"], mh)
    field = torch.einsum("jia,j->ia", b["t01"], q) + torch.einsum("ijab,jb->ia", b["t2"], mh)
    if "t02" not in b:
        return phi, field, None
    phi = phi + torch.einsum("ijm,jm->i", b["t02"], th)
    field = field + torch.einsum("ijam,jm->ia", b["t12"], th)
    grad = (torch.einsum("jim,j->im", b["t02"], q) + torch.einsum("jiam,ja->im", b["t12"], mh)
            + torch.einsum("ijmn,jn->im", b["t22"], th))
    return phi, field, grad


def _energy(op, params, q, mh, th):
    phi, field, grad = op.apply_reduced(q, mh, params.ell, th if op.quadrupolar else None)
    energy = 0.5 * (q @ phi + (mh * field).sum())
    return energy if grad is None else energy + 0.5 * (th * grad).sum()


def _dense_energy(pos, params, kernel, quadrupoles, q, mh, th):
    phi, field, grad = _dense_product(_dense_blocks(pos, params, kernel, quadrupoles), q, mh, th)
    energy = 0.5 * (q @ phi + (mh * field).sum())
    return energy if grad is None else energy + 0.5 * (th * grad).sum()


def _relative(got, want):
    return float((got - want).abs().max()) / float(want.abs().max())


CASES = [("gaussian", False), ("gaussian", True), ("thole", False), ("thole", True)]


# ===========================================================================
# the product
# ===========================================================================
@pytest.mark.parametrize("kernel, quadrupoles", CASES)
def test_product_equals_the_dense_blocks(kernel, quadrupoles):
    """Measured 2e-16 .. 1.1e-15 relative for every case, mode and chunking."""
    pos, params, (q, mh, th) = _cluster()
    want = _dense_product(_dense_blocks(pos, params, kernel, quadrupoles), q, mh, th)
    for precompute in (True, False):
        for chunk in (1 << 18, 37, 1):
            op = _operator(pos, params, kernel, quadrupoles, precompute=precompute, chunk=chunk)
            got = op.apply_reduced(q, mh, params.ell, th if quadrupoles else None)
            for g, w in zip(got, want):
                if w is None:
                    assert g is None
                else:
                    assert _relative(g, w) < 1e-12, (precompute, chunk)


def test_quadrupolar_operator_without_theta_is_its_charge_dipole_part():
    pos, params, (q, mh, th) = _cluster(n=9)
    full = _operator(pos, params, "gaussian", True)
    reduced = _operator(pos, params, "gaussian", False)
    phi, field, grad = full.apply_reduced(q, mh, params.ell)
    phi_d, field_d, _ = reduced.apply_reduced(q, mh, params.ell)
    assert grad is None
    assert torch.equal(phi, phi_d) and torch.equal(field, field_d)
    with pytest.raises(ValueError, match="quadrupole"):
        reduced.apply_reduced(q, mh, params.ell, th)


@pytest.mark.parametrize("kernel, quadrupoles", CASES)
def test_operator_is_symmetric(kernel, quadrupoles):
    """v.(K w) == w.(K v): the ordered-width Gaussian blocks enter with their transposes."""
    pos, params, _ = _cluster(n=17, seed=4)
    g = torch.Generator().manual_seed(9)
    op = _operator(pos, params, kernel, quadrupoles, chunk=40)
    n = pos.shape[0]

    def moments():
        return (torch.randn(n, generator=g), torch.randn(n, 3, generator=g),
                torch.randn(n, 5, generator=g) if quadrupoles else None)

    def dot(a, b):
        return sum((x * y).sum() for x, y in zip(a, b) if x is not None)

    v, w = moments(), moments()
    Kv, Kw = op.apply(*v), op.apply(*w)
    left, right = dot(v, Kw), dot(w, Kv)
    assert abs(float(left - right)) < 1e-13 * float(dot(v, Kv).abs() + dot(w, Kw).abs())


def test_tiny_systems():
    OpenOperator = require("open_operator", "OpenOperator")
    for n in (0, 1, 2):
        pos = torch.arange(3 * n, dtype=torch.float64).reshape(n, 3)
        op = OpenOperator(pos, sigma=torch.ones(n), kernel="gaussian", sigma_mu=torch.ones(n))
        phi, field, grad = op.apply(torch.ones(n), torch.ones(n, 3))
        assert phi.shape == (n,) and field.shape == (n, 3) and grad is None
    # the n = 2 operator, two charges and no dipoles: phi_i = J(r) q_j, r = sqrt(27)
    j = require("kernels", "coulomb_kernel")(torch.tensor(math.sqrt(27.0)), torch.tensor(math.sqrt(2.0)))
    phi, _, _ = op.apply(torch.tensor([1.0, 2.0]), torch.zeros(2, 3))
    assert torch.allclose(phi, torch.stack([2.0 * j, j]), rtol=1e-15, atol=0.0)


def test_row_blocks_bound_the_tile_size():
    """No tile ever exceeds max(chunk, n) entries: nothing n x n is formed."""
    row_blocks = require("open_operator", "_row_blocks")
    for n, chunk in ((1000, 4096), (1000, 10), (37, 1 << 18), (5000, 1 << 18)):
        blocks = row_blocks(n, chunk)
        assert blocks[0][0] == 0 and blocks[-1][1] == n - 1
        assert all(b0[1] == b1[0] for b0, b1 in zip(blocks, blocks[1:]))
        assert all((b - a) * (n - a) <= max(chunk, n) for a, b in blocks)


def test_precompute_modes():
    pos, params, _ = _cluster(n=12)
    assert _operator(pos, params, "gaussian", True, precompute=False)._tiles is None
    auto = _operator(pos, params, "gaussian", True)
    assert auto.precomputed and auto.precompute_bytes == 14 * 8 * sum(
        (b - a) * (12 - a) for a, b in auto.blocks)
    assert not _operator(pos, params, "gaussian", True, max_precompute_bytes=1000).precomputed
    # a rebuild on graph-carrying inputs is applied once (implicit derivative): no tiles
    rebuilt = auto.rebuilt(pos.clone().requires_grad_(True), *auto.inputs()[1:])
    assert not rebuilt.precomputed
    assert auto.rebuilt(*auto.inputs()).precomputed


def test_far_field_short_cut_is_exact():
    """torch.special.gammainc(n + 1/2, x) is exactly 1.0 beyond the saturation point.

    If a torch update breaks this, the operator's far pairs would differ from
    the dense blocks by at most 2^-53 relative; this test says so first.
    """
    saturation = require("open_operator", "_GAMMA_SATURATION")
    x = torch.cat([torch.linspace(saturation, 200.0, 200001), torch.logspace(2.31, 8.0, 1000)])
    for n in range(5):
        assert bool((torch.special.gammainc(torch.full_like(x, n + 0.5), x) == 1.0).all())


# ===========================================================================
# derivatives
# ===========================================================================
@pytest.mark.parametrize("kernel, quadrupoles", CASES[:3])
@pytest.mark.parametrize("precompute", [True, False])
def test_energy_derivatives_equal_the_dense_construction(kernel, quadrupoles, precompute):
    """First and second derivatives, positions and parameters.

    Coincident and sub-shield pairs are included; the near-coincident pair sits
    at 0.3 A.  Closer, autograd of l_n/r^n is round-off dominated in BOTH
    constructions (terms ~ l_n/r^(n+2) cancel): at 1e-7 A the dense second
    derivative itself moves by 76 % (Gaussian), 17x (quadrupoles), 4 % (Thole)
    under a 3 A translation.  Measured here: E 4e-16, first 1e-15,
    second 1e-14.
    """
    pos0, params0, (q, mh, th) = _cluster(n=11, seed=1, near=0.3)
    leaves = {name: getattr(params0, name).clone().requires_grad_(True)
              for name in ("sigma", "ell", "A_mu", "A_Theta")}
    pos = pos0.clone().requires_grad_(True)
    params = dataclasses.replace(params0, **leaves)
    wrt = [pos] + list(leaves.values())
    dense = _dense_energy(pos, params, kernel, quadrupoles, q, mh, th)
    free = _energy(_operator(pos, params, kernel, quadrupoles, precompute=precompute, chunk=13),
                   params, q, mh, th)
    assert abs(float(dense - free)) < 1e-14 * abs(float(dense))
    g_dense = torch.autograd.grad(dense, wrt, create_graph=True, allow_unused=True)
    g_free = torch.autograd.grad(free, wrt, create_graph=True, allow_unused=True)
    for a, b in zip(g_dense, g_free):
        assert (a is None) == (b is None)
        if a is not None:
            assert _relative(b, a) < 1e-12
    h_dense = torch.autograd.grad((g_dense[0] ** 2).sum(), wrt, allow_unused=True)
    h_free = torch.autograd.grad((g_free[0] ** 2).sum(), wrt, allow_unused=True)
    for a, b in zip(h_dense, h_free):
        if a is not None:
            assert _relative(b, a) < 1e-11


@pytest.mark.parametrize("precompute", [True, False])
def test_double_backward_matches_finite_differences(precompute):
    """H v through the checkpointed chunks against central differences of the forces.

    No coincident pair here: the DEGENERATE_SEPARATION shield is a jump by
    design (kernels.dipole_interaction_tensor), which a displaced stencil
    would straddle.
    """
    pos0, params, (q, mh, th) = _cluster(n=8, seed=2, near=0.4, degenerate=False)
    direction = torch.randn(pos0.shape, generator=torch.Generator().manual_seed(1))

    def force(p, create_graph):
        p = p.clone().requires_grad_(True)
        energy = _energy(_operator(p, params, "gaussian", True, precompute=precompute, chunk=5),
                         params, q, mh, th)
        return p, torch.autograd.grad(energy, p, create_graph=create_graph)[0]

    p, f = force(pos0, True)
    (hv,) = torch.autograd.grad((f * direction).sum(), p)
    h = 1e-5
    fd = (force(pos0 + h * direction, False)[1] - force(pos0 - h * direction, False)[1]) / (2 * h)
    assert _relative(hv, fd) < 1e-7


@pytest.mark.parametrize("kernel, quadrupoles", CASES)
def test_pair_energy_is_rotation_and_translation_invariant(kernel, quadrupoles):
    basis = require("kernels", "quadrupole_cartesian_basis")()
    pos, params, (q, mh, th) = _cluster(n=15, seed=6)
    angle = torch.tensor(0.7)
    axis = torch.tensor([0.3, -0.5, 0.8])
    axis = axis / axis.norm()
    cross = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = torch.eye(3) + torch.sin(angle) * cross + (1 - torch.cos(angle)) * cross @ cross
    th_cart = torch.einsum("im,mab->iab", th, basis)
    th_rot = torch.einsum("iab,mab->im", R @ th_cart @ R.T, basis)
    reference = _energy(_operator(pos, params, kernel, quadrupoles), params, q, mh, th)
    moved = _energy(_operator(pos @ R.T + torch.tensor([3.0, -7.0, 11.0]), params, kernel, quadrupoles),
                    params, q, mh @ R.T, th_rot)
    assert abs(float(moved - reference)) < 1e-13 * abs(float(reference))


# ===========================================================================
# drop-in for the matrix-free solve
# ===========================================================================
def _open_system(n=9, seed=3, quadrupoles=True):
    solver = require("solver", "ElectrostaticParams", "FragmentConstraints")
    ElectrostaticParams, FragmentConstraints = solver
    build_incidence = require("topology", "build_incidence")
    kernel_self = require("kernels", "kernel_self")
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g) * 1.4
    distance = torch.cdist(pos, pos)
    rows, cols = torch.triu_indices(n, n, 1)
    bonded = distance[rows, cols] < 2.4
    incidence, bonds = build_incidence(torch.stack([rows[bonded], cols[bonded]]), n)
    sigma = 1.0 + torch.rand(n, generator=g)
    params = ElectrostaticParams(
        chi=torch.randn(n, generator=g), eta=kernel_self(sigma) + 0.5, sigma=sigma,
        ell=0.6 + 0.8 * torch.rand(n, generator=g), kappa=0.5 + torch.rand(bonds.shape[1], generator=g),
        A_mu=_spd(n, 3, 2.0, g), A_Theta=_spd(n, 5, 1.0, g) if quadrupoles else None,
        chi_theta=torch.randn(n, 5, generator=g) if quadrupoles else None,
    )
    fragments = torch.zeros(2, n)
    fragments[0, : n // 2] = 1.0
    fragments[1, n // 2:] = 1.0
    constraints = FragmentConstraints(matrix=fragments, targets=torch.tensor([0.4, -0.4]))
    return pos, incidence, bonds, params, constraints


@pytest.mark.parametrize("kernel, quadrupoles", CASES[:3])
@pytest.mark.parametrize("mode", ["fixed_k", "converged"])
def test_solve_lattice_with_the_open_operator_equals_the_dense_solve(kernel, quadrupoles, mode):
    """Energy, charges, dipoles, forces and d|F|^2/dr, two charged fragments, a field.

    The field couples through ``bond_dipole = f (r_lo - r_hi)``, which under
    open boundaries IS the dense route's ``chi -> chi - E.r`` (SPEC 15).
    Measured: fixed-k 1.5e-14; converged at rtol 1e-13, 9e-13 (at the default
    1e-9 the two independently preconditioned solves differ by ~1e-9).
    """
    solver = require("solver", "SolverOptions", "GeometrySeed", "Geometry", "solve_charges",
                     "gaussian_widths", "isotropic_polarizability_volume")
    SolverOptions, GeometrySeed, Geometry, solve_charges, gaussian_widths, alpha_volume = solver
    OpenOperator = require("open_operator", "OpenOperator")
    solve_lattice = require("lattice_operator", "solve_lattice")
    pos0, incidence, bonds, params, constraints = _open_system(quadrupoles=quadrupoles)
    n, n_bonds = pos0.shape[0], bonds.shape[1]
    options = SolverOptions(multipole_kernel=kernel, solve_mode=mode, solve_rtol=1e-13, solve_max_iter=2000)
    field = torch.tensor([0.05, -0.02, 0.03])
    results = {}
    for route in ("dense", "open"):
        pos = pos0.clone().requires_grad_(True)
        if route == "dense":
            result = solve_charges(params, Geometry(positions=pos, incidence=incidence, bond_index=bonds),
                                   constraints, 6, seed=GeometrySeed.zeros(n_bonds, n), options=options,
                                   external_field=field)
        else:
            if kernel == "gaussian":
                sigma_mu, sigma_theta = gaussian_widths(params, options.gaussian_a)
                op = OpenOperator(pos, sigma=params.sigma, kernel="gaussian", sigma_mu=sigma_mu,
                                  sigma_theta=sigma_theta if quadrupoles else None)
            else:
                op = OpenOperator(pos, sigma=params.sigma, kernel="thole",
                                  alpha_volume=alpha_volume(params.A_mu, params.ell), thole_a=options.thole_a)
            envelope = torch.ones(n_bonds)
            result = solve_lattice(op, params, bonds, envelope, constraints, 6, tau_seed=torch.zeros(n_bonds),
                                   options=options, external_field=field,
                                   bond_dipole=envelope.unsqueeze(-1) * (pos[bonds[0]] - pos[bonds[1]]))
        (force,) = torch.autograd.grad(result.energy, pos, create_graph=True)
        (curvature,) = torch.autograd.grad((force * force).sum(), pos)
        results[route] = (result.energy.detach(), result.q.detach(), result.mu.detach(), force.detach(), curvature)
    tolerance = 1e-12 if mode == "fixed_k" else 1e-10
    for a, b in zip(results["dense"], results["open"]):
        assert _relative(b, a) < tolerance
