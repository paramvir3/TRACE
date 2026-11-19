"""Quadrupoles and dispersion in the master equation (SPEC 10, 12, 13).

Every claim the extension rests on is asserted here:

SPEC 12 -- the Gaussian (Gram) multipole kernel
* lambda_{2n+1} = P(n + 1/2, r^2/s^2) turns every existing tensor into the exact
  derivative of K_E erf(r/s)/r;
* the width rule bounds the Gram self term by a * lambda_min(A);
* K with charges, dipoles AND quadrupoles is positive definite for every
  geometry, including coincident atoms and on-site parameters at their floors;
* the Thole-coupled three-rank Hessian is NOT -- the measured counterexample
  that makes the Gaussian kernel mandatory for quadrupoles is pinned here.

SPEC 10/12 -- the quadrupole sector
* the converged solve is the exact minimiser; forces are exact;
* the permanent source chi_Theta lands where the physics says;
* two permanent quadrupoles interact through the bare T22 at long range;
* the pipeline's two constructions agree, the energy is rotation invariant, the
  moments rotate as a vector and a rank-2 tensor, alpha stays reciprocal.

SPEC 13 -- dispersion as the zero-point energy of the same moment block
* the log-det frequency integral IS the eigenvalue zero-point energy;
* the pair limit is the London / Slater-Kirkwood C6; the TRK inertia gives the
  hydrogen numbers;
* second derivatives stay finite on exactly degenerate spectra, where eigh fails;
* the subtractive range separation leaves the long-range C6 untouched.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import warnings

import pytest
import torch

from conftest import require

K_E = 14.399645


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _spd(n, d, floor, generator, scale=0.6):
    factor = scale * torch.randn(n, d, d, generator=generator)
    return factor @ factor.transpose(-1, -2) + floor * torch.eye(d)


def _pair_geometry(pos):
    n = pos.shape[0]
    r_vec = pos[:, None] - pos[None]
    squared = (r_vec * r_vec).sum(-1)
    eye = torch.eye(n, dtype=torch.bool)
    r = torch.sqrt(torch.where(eye, torch.ones_like(squared), squared))
    return r_vec, r, (~eye).to(pos.dtype)


def _solver_system(pos, *, quadrupoles=True, chi_theta=True, seed=3, cutoff=2.2,
                   eta_margin=0.5, a_mu_floor=2.0, a_theta_floor=1.0):
    solver = require(
        "solver", "ElectrostaticParams", "Geometry", "FragmentConstraints", "SolverOptions"
    )
    ElectrostaticParams, Geometry, FragmentConstraints, _ = solver
    build_incidence = require("topology", "build_incidence")
    kernel_self = require("kernels", "kernel_self")
    g = torch.Generator().manual_seed(seed)
    n = pos.shape[0]
    distance = torch.cdist(pos, pos)
    rows, cols = torch.triu_indices(n, n, 1)
    bonded = distance[rows, cols] < cutoff
    incidence, bonds = build_incidence(torch.stack([rows[bonded], cols[bonded]]), n)
    sigma = 1.0 + torch.rand(n, generator=g)
    params = ElectrostaticParams(
        chi=torch.randn(n, generator=g),
        eta=kernel_self(sigma) + eta_margin,
        sigma=sigma,
        ell=0.6 + 0.8 * torch.rand(n, generator=g),
        kappa=torch.ones(bonds.shape[1]),
        A_mu=_spd(n, 3, a_mu_floor, g),
        A_Theta=_spd(n, 5, a_theta_floor, g) if quadrupoles else None,
        chi_theta=torch.randn(n, 5, generator=g) if (quadrupoles and chi_theta) else None,
    )
    geometry = Geometry(positions=pos, incidence=incidence, bond_index=bonds)
    constraints = FragmentConstraints(matrix=torch.ones(1, n), targets=torch.zeros(1))
    return params, geometry, constraints, int(bonds.shape[1])


def _gaussian_options(**kwargs):
    return require("solver", "SolverOptions")(multipole_kernel="gaussian", **kwargs)


def _solve(params, geometry, constraints, n_bonds, n_iter, options=None):
    solve_charges, GeometrySeed = require("solver", "solve_charges", "GeometrySeed")
    return solve_charges(
        params, geometry, constraints, n_iter,
        seed=GeometrySeed.zeros(n_bonds, params.n_atoms),
        options=options or _gaussian_options(),
    )


def _heads(seed=0, **kwargs):
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return require("heads", "ElectrostaticHeads")(**kwargs)


PIPELINE_POSITIONS = torch.tensor(
    [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [1.9, 1.0, 0.3],
     [-0.8, 0.9, 0.1], [0.3, -1.2, 0.8], [2.9, 1.3, 1.0]],
    dtype=torch.float64,
)


def _complete_graph(n):
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    return torch.tensor([[i for i, _ in pairs], [j for _, j in pairs]])


def _pipeline(heads, h, pos, config, quadrupoles=True):
    electrostatic_state = require("energy", "electrostatic_state")
    edge_index = _complete_graph(pos.shape[0])
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    return electrostatic_state(
        h, pos, edge_index, edge_vec, edge_vec.norm(dim=-1), heads, config,
        quadrupoles=quadrupoles,
    )


def _quadrupole_heads(**kwargs):
    return _heads(
        quadrupole_blocks=("A_Theta", "chi_theta"), allow_truncated_a_theta=True, **kwargs
    )


def _config(**kwargs):
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    return ElectrostaticConfig(multipole_kernel="gaussian", **kwargs)


# ===========================================================================
# SPEC 12: the Gaussian multipole kernel
# ===========================================================================
def test_gaussian_damping_gives_exact_derivatives_of_the_erf_potential():
    """Each tensor with Gaussian lambdas is the exact derivative of K_E erf(r/s)/r."""
    kernels = require(
        "kernels", "gaussian_damping", "charge_dipole_tensor", "dipole_interaction_tensor",
        "charge_quadrupole_tensor", "dipole_quadrupole_tensor", "quadrupole_interaction_tensor",
    )
    damping, t01, t11, t02, t12, t22 = kernels
    jac = torch.autograd.functional.jacobian
    g = torch.Generator().manual_seed(11)
    for _ in range(8):
        r_vec = torch.randn(3, generator=g) * (0.3 + 3.0 * torch.rand(1, generator=g))
        s = 0.3 + 2.5 * float(torch.rand(1, generator=g))
        r = r_vec.norm()
        l3, l5, l7, l9 = damping(r, torch.tensor(s))

        def phi(x):
            return torch.erf(x.norm() / s) / x.norm()

        d1 = jac(phi, r_vec, create_graph=True)
        d2 = jac(lambda x: jac(phi, x, create_graph=True), r_vec, create_graph=True)
        d3 = jac(lambda x: jac(lambda y: jac(phi, y, create_graph=True), x,
                               create_graph=True), r_vec, create_graph=True)
        d4 = jac(lambda x: jac(lambda y: jac(lambda z: jac(phi, z, create_graph=True), y,
                                             create_graph=True), x, create_graph=True), r_vec)
        cases = [
            (t01(r_vec, r, l3), -K_E * d1),
            (t11(r_vec, r, l3, l5), -K_E * d2),
            (t02(r_vec, r, l3, l5), 0.5 * K_E * d2),
            (t12(r_vec, r, l5, l7), 0.5 * K_E * d3),
            (t22(r_vec, r, l5, l7, l9), 0.25 * K_E * d4),
        ]
        for got, want in cases:
            assert float((got - want).abs().max()) <= 1e-10 * float(want.abs().max())


@pytest.mark.parametrize("rank, dim", [(1, 3), (2, 5)])
@pytest.mark.parametrize("a", [0.2, 0.5, 0.9])
def test_width_rule_bounds_the_gram_self_term(rank, dim, a):
    """ell^(2l) S_l(sigma_l) <= a lambda_min(A): the on-site residual stays positive."""
    width, self_stiffness = require(
        "kernels", "gaussian_multipole_width", "gaussian_self_stiffness"
    )
    g = torch.Generator().manual_seed(rank * 10 + int(10 * a))
    stiffness = _spd(20, dim, 0.05, g)
    ell = 0.25 + 1.75 * torch.rand(20, generator=g)
    sigma = width(stiffness, ell, rank, a)
    ratio = ell ** (2 * rank) * self_stiffness(sigma, rank) / torch.linalg.eigvalsh(stiffness)[:, 0]
    assert float(ratio.max()) <= a * (1.0 + 1e-12)


def test_gaussian_width_rule_rejects_a_at_or_above_one():
    width = require("kernels", "gaussian_multipole_width")
    with pytest.raises(ValueError, match="0 < a < 1"):
        width(torch.eye(3)[None], torch.ones(1), 1, 1.0)


def test_three_rank_gaussian_hessian_is_positive_definite_for_every_geometry():
    """The Gram theorem: K > 0 at the parameter floors, near-coincident atoms included."""
    assemble_system = require("solver", "assemble_system")
    g = torch.Generator().manual_seed(5)
    worst = math.inf
    for trial in range(30):
        n = 3 + trial % 5
        pos = torch.randn(n, 3, generator=g) * (0.3 + 2.0 * float(torch.rand(1, generator=g)))
        if trial % 3 == 0:
            pos[1] = pos[0] + 1e-4 * torch.randn(3, generator=g)
        # eta at the hardness floor, A_mu and A_Theta at tiny floors: the worst
        # case the heads can emit.
        params, geometry, constraints, _ = _solver_system(
            pos, seed=trial, eta_margin=0.0, a_mu_floor=0.05, a_theta_floor=0.05
        )
        K = assemble_system(params, geometry, constraints, _gaussian_options()).K
        eigenvalues = torch.linalg.eigvalsh(0.5 * (K + K.T))
        worst = min(worst, float(eigenvalues.min() / eigenvalues.max()))
    assert worst > 0.0


def test_thole_coupled_three_rank_hessian_can_be_indefinite():
    """Why the Gaussian kernel is mandatory for Theta: a pinned Thole counterexample.

    Six atoms at the heads' initial values (alpha = 1.5 A^3, quadrupole
    polarizability 6 A^5): every diagonal block and every pairwise sub-block is
    positive definite, the full (q, mu, Theta) Hessian is not.
    """
    kernels = require(
        "kernels", "thole_damping_quadrupole", "coulomb_kernel", "combine_sigma",
        "kernel_self", "charge_dipole_tensor", "dipole_interaction_tensor",
        "charge_quadrupole_tensor", "dipole_quadrupole_tensor",
        "quadrupole_interaction_tensor", "quadrupole_cartesian_basis",
    )
    (thole, coulomb, combine, kself, t01, t11, t02, t12, t22, basis_fn) = kernels
    W = basis_fn()
    alpha, c_theta = 1.5, 6.0
    torch.manual_seed(0)
    found = False
    for _ in range(200):
        n = 6
        pos = torch.randn(n, 3) * 1.2
        if torch.pdist(pos).min() < 0.9:
            continue
        sigma = 1.0 + torch.rand(n)
        r_vec, r, keep = _pair_geometry(pos)
        l3, l5, l7, l9 = thole(r, torch.full((n, n), alpha), torch.full((n, n), alpha))
        J = coulomb(r, combine(sigma[:, None], sigma[None])) * keep + torch.diag(kself(sigma) + 0.5)
        Tqm = (t01(r_vec, r, l3) * keep[..., None]).reshape(n, 3 * n)
        Tmm = (t11(r_vec, r, l3, l5) * keep[..., None, None]).permute(0, 2, 1, 3).reshape(3 * n, 3 * n)
        Tqt = (torch.einsum("ijab,mab->ijm", t02(r_vec, r, l3, l5), W) * keep[..., None]).reshape(n, 5 * n)
        Tmt = (torch.einsum("ijabc,mbc->ijam", t12(r_vec, r, l5, l7), W)
               * keep[..., None, None]).permute(0, 2, 1, 3).reshape(3 * n, 5 * n)
        Ttt = (torch.einsum("mab,ijabcd,ncd->ijmn", W, t22(r_vec, r, l5, l7, l9), W)
               * keep[..., None, None]).permute(0, 2, 1, 3).reshape(5 * n, 5 * n)
        Kmm = Tmm + (K_E / alpha) * torch.eye(3 * n)
        Ktt = Ttt + (K_E / c_theta) * torch.eye(5 * n)
        K = torch.cat([torch.cat([J, Tqm, Tqt], 1),
                       torch.cat([Tqm.T, Kmm, Tmt], 1),
                       torch.cat([Tqt.T, Tmt.T, Ktt], 1)], 0)
        pairs = [
            torch.cat([torch.cat([J, Tqm], 1), torch.cat([Tqm.T, Kmm], 1)], 0),
            torch.cat([torch.cat([J, Tqt], 1), torch.cat([Tqt.T, Ktt], 1)], 0),
            torch.cat([torch.cat([Kmm, Tmt], 1), torch.cat([Tmt.T, Ktt], 1)], 0),
        ]
        if (torch.linalg.eigvalsh(K).min() < 0
                and all(torch.linalg.eigvalsh(block).min() > 0 for block in pairs)):
            found = True
            break
    assert found, "no Thole counterexample found -- the SPEC 12 justification is stale"


def test_thole_kernel_with_quadrupoles_is_refused():
    assemble_system, SolverOptions = require("solver", "assemble_system", "SolverOptions")
    params, geometry, constraints, _ = _solver_system(torch.randn(4, 3) * 1.3)
    with pytest.raises(ValueError, match="gaussian"):
        assemble_system(params, geometry, constraints, SolverOptions(multipole_kernel="thole"))


def test_quadrupole_sector_needs_its_stiffness():
    ElectrostaticParams = require("solver", "ElectrostaticParams")
    params, *_ = _solver_system(torch.randn(3, 3) * 1.3)
    with pytest.raises(ValueError, match="unbounded"):
        dataclasses.replace(params, A_Theta=None)


# ===========================================================================
# SPEC 10/12: the quadrupole sector of the solve
# ===========================================================================
def test_converged_solve_is_the_exact_minimiser_with_quadrupoles():
    assemble_system = require("solver", "assemble_system")
    torch.manual_seed(3)
    pos = torch.randn(7, 3) * 1.3
    params, geometry, constraints, n_bonds = _solver_system(pos)
    system = assemble_system(params, geometry, constraints, _gaussian_options())
    assert system.K.shape[0] == n_bonds + 8 * 7
    m_star = -torch.linalg.solve(system.K, system.c)
    result = _solve(params, geometry, constraints, n_bonds, 400)
    assert abs(float(result.energy - system.energy(m_star))) < 1e-12
    assert torch.allclose(result.theta_hat.reshape(-1), m_star[-35:], atol=1e-11)


def test_theta_is_the_traceless_second_moment_of_its_coefficients():
    basis_fn = require("kernels", "quadrupole_cartesian_basis")
    pos = torch.randn(5, 3, generator=torch.Generator().manual_seed(1)) * 1.4
    params, geometry, constraints, n_bonds = _solver_system(pos)
    result = _solve(params, geometry, constraints, n_bonds, 6)
    trace = torch.diagonal(result.theta, dim1=1, dim2=2).sum(-1)
    assert float(trace.abs().max()) < 1e-14
    assert torch.allclose(result.theta, result.theta.transpose(1, 2))
    # Theta = ell^2 sum_m Theta_hat_m W_m, and the orthonormal basis inverts it.
    recovered = torch.einsum("iab,mab->im", result.theta, basis_fn()) / params.ell[:, None] ** 2
    assert torch.allclose(recovered, result.theta_hat, atol=1e-14)


def test_permanent_quadrupole_of_an_isolated_atom():
    """chi_Theta alone: Theta_hat = -A_Theta^-1 chi_Theta, reached by the seed step."""
    params, geometry, constraints, n_bonds = _solver_system(torch.zeros(1, 3))
    result = _solve(params, geometry, constraints, n_bonds, 1)
    exact = -torch.linalg.solve(params.A_Theta[0], params.chi_theta[0])
    assert torch.allclose(result.theta_hat[0], exact, atol=1e-14)


def test_quadrupole_seed_step_never_raises_the_energy():
    assemble_system = require("solver", "assemble_system")
    relax = require("solver", "_relax_quadrupole_seed")
    g = torch.Generator().manual_seed(9)
    for trial in range(10):
        pos = torch.randn(5, 3, generator=g) * 1.4
        params, geometry, constraints, _ = _solver_system(pos, seed=trial)
        system = assemble_system(params, geometry, constraints, _gaussian_options())
        m0 = torch.randn(system.K.shape[0], generator=g)
        m1 = relax(system, m0, system.quadrupole_offset, 5, torch.finfo(torch.float64).tiny,
                   params.A_Theta)
        assert float(system.energy(m1)) <= float(system.energy(m0)) + 1e-12
        assert torch.equal(m1[: system.quadrupole_offset], m0[: system.quadrupole_offset])


def test_forces_are_exact_with_quadrupoles():
    torch.manual_seed(3)
    pos = torch.randn(6, 3) * 1.3

    def energy(x):
        params, geometry, constraints, n_bonds = _solver_system(x)
        return _solve(params, geometry, constraints, n_bonds, 6).energy

    x = pos.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(x), x)
    for atom, axis in [(0, 0), (2, 1), (5, 2)]:
        step = torch.zeros_like(pos)
        step[atom, axis] = 1e-4
        fd = (energy(pos + step) - energy(pos - step)) / 2e-4
        assert abs(float(fd) - float(grad[atom, axis])) < 1e-6 * abs(float(grad[atom, axis]))


def test_two_permanent_quadrupoles_interact_through_bare_t22_at_long_range():
    """E_int -> Theta_1 : T22 : Theta_2 (bare), with an R^-3 relative correction.

    The correction is induction and nothing else: the quadrupole field polarises
    the other atom's dipole, -alpha E^2/2 ~ R^-8 against the R^-5 leading term.
    Measured: the deviation shrinks x8.33, x8.08, x7.99 per doubling of R from
    15 A to 120 A (x32 with the dipoles frozen: Theta-Theta induction, R^-10).
    A convention error -- Buckingham's Theta, a missing 1/4 -- would instead
    leave a CONSTANT offset, which is what this test excludes.
    """
    t22_fn, basis_fn = require("kernels", "quadrupole_interaction_tensor", "quadrupole_cartesian_basis")
    W = basis_fn()
    single, geometry1, constraints1, _ = _solver_system(torch.zeros(1, 3), seed=4)
    other, geometry2, constraints2, _ = _solver_system(torch.zeros(1, 3), seed=8)
    ElectrostaticParams, Geometry, FragmentConstraints = require(
        "solver", "ElectrostaticParams", "Geometry", "FragmentConstraints"
    )
    build_incidence = require("topology", "build_incidence")

    def monomer(params):
        return _solve(dataclasses.replace(params, chi=torch.zeros(1)), geometry1, constraints1, 0, 50)

    theta_1 = monomer(single).theta[0]
    theta_2 = monomer(other).theta[0]
    ratios = []
    for R in (30.0, 60.0):
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.3 * R, 0.4 * R, 0.866 * R]])
        pos = pos / pos[1].norm() * R
        pair = ElectrostaticParams(**{
            field.name: (None if getattr(single, field.name) is None
                         else torch.cat([getattr(single, field.name), getattr(other, field.name)]))
            for field in dataclasses.fields(single)
            if field.name not in ("kappa",)
        }, kappa=torch.ones(0))
        pair = dataclasses.replace(pair, chi=torch.zeros(2))
        incidence, bonds = build_incidence(torch.zeros(2, 0, dtype=torch.long), 2)
        geometry = Geometry(positions=pos, incidence=incidence, bond_index=bonds)
        constraints = FragmentConstraints(matrix=torch.eye(2), targets=torch.zeros(2))
        e_pair = _solve(pair, geometry, constraints, 0, 200).energy
        e_apart = (monomer(single).energy + monomer(other).energy)
        r_vec = pos[0] - pos[1]
        one = torch.ones(())
        bare = torch.einsum("ab,abcd,cd->", theta_1, t22_fn(r_vec, r_vec.norm(), one, one, one), theta_2)
        ratios.append(float((e_pair - e_apart) / bare))
    shrink = abs(ratios[0] - 1.0) / abs(ratios[1] - 1.0)
    assert 7.5 < shrink < 8.5          # R^-3: induction, not a prefactor error
    assert abs(ratios[1] - 1.0) < 1e-4


# ===========================================================================
# the pipeline: heads -> assemble -> solve -> energy
# ===========================================================================
@pytest.fixture(scope="module")
def pipeline_case():
    heads = _quadrupole_heads(dispersion=True)
    torch.manual_seed(0)
    h = torch.randn(6, heads.irreps_in.dim) * 2.0
    return heads, h


def test_pipeline_constructions_agree_with_quadrupoles(pipeline_case):
    heads, h = pipeline_case
    state = _pipeline(heads, h, PIPELINE_POSITIONS, _config(dispersion=True))
    assembly = state.assembly
    assert assembly.quadrupoles and assembly.params.chi_theta is not None
    assert assembly.moment_dim == assembly.n_bonds + 9 * 6
    assert abs(float(state.energy - state.solve_result.energy)) < 1e-13
    dense, curvature = assembly.dense_hessian(), assembly.curvature()
    assert float((0.5 * (dense + dense.T) - curvature).abs().max()) < 1e-12
    assert state.dispersion is not None and float(state.dispersion) < 0.0


def test_pipeline_is_rotation_invariant_and_moments_rotate(pipeline_case):
    o3 = pytest.importorskip("e3nn.o3")
    heads, h = pipeline_case
    config = _config(dispersion=True)
    rotation = o3.rand_matrix()
    wigner = heads.irreps_in.D_from_matrix(rotation)
    state = _pipeline(heads, h, PIPELINE_POSITIONS, config)
    turned = _pipeline(heads, h @ wigner.T, PIPELINE_POSITIONS @ rotation.T, config)
    assert abs(float(turned.energy - state.energy)) < 1e-12
    assert abs(float(turned.dispersion - state.dispersion)) < 1e-11
    mu, theta = state.solve_result.mu, state.solve_result.theta
    assert torch.allclose(turned.solve_result.mu, mu @ rotation.T, atol=1e-13)
    assert torch.allclose(turned.solve_result.theta, rotation @ theta @ rotation.T, atol=1e-13)


def test_polarizability_stays_reciprocal_and_feels_the_quadrupoles(pipeline_case):
    polarizability = require("observables", "polarizability")
    heads, h = pipeline_case
    with_theta = _pipeline(heads, h, PIPELINE_POSITIONS, _config(n_cg_iter=300))
    without = _pipeline(heads, h, PIPELINE_POSITIONS, _config(n_cg_iter=300), quadrupoles=False)
    alpha = polarizability(with_theta.solve_result, with_theta.assembly)
    alpha_0 = polarizability(without.solve_result, without.assembly)
    # Default route (SPEC 17): each column from its own adjoint solve at
    # solve_rtol = 1e-10, so reciprocity holds to that tolerance (measured
    # 7e-12 on O(1) entries); the dense minimiser route is exact.
    assert float((alpha - alpha.T).abs().max()) < 1e-9 * float(alpha.abs().max())
    alpha_min = polarizability(with_theta.solve_result, with_theta.assembly, route="minimiser")
    assert float((alpha_min - alpha_min.T).abs().max()) < 1e-12
    assert float((alpha - alpha_0).abs().max()) > 1e-6


def test_forces_include_dispersion_and_quadrupoles(pipeline_case):
    electrostatic_energy = require("energy", "electrostatic_energy")
    heads, h = pipeline_case
    config = _config(dispersion=True)
    x = PIPELINE_POSITIONS.clone().requires_grad_(True)
    edge_index = _complete_graph(6)
    edge_vec = x[edge_index[0]] - x[edge_index[1]]
    energy, _ = electrostatic_energy(
        h, x, edge_index, edge_vec, edge_vec.norm(dim=-1), heads, config, quadrupoles=True
    )
    (grad,) = torch.autograd.grad(energy, x)

    def total(pos):
        state = _pipeline(heads, h, pos, config)
        return state.energy + state.dispersion

    for atom, axis in [(0, 0), (3, 2), (5, 1)]:
        step = torch.zeros_like(PIPELINE_POSITIONS)
        step[atom, axis] = 1e-4
        fd = (total(PIPELINE_POSITIONS + step) - total(PIPELINE_POSITIONS - step)) / 2e-4
        assert abs(float(fd) - float(grad[atom, axis])) < 1e-6 * abs(float(grad[atom, axis]))


def test_heads_with_blocks_but_quadrupoles_off_leave_the_sector_off(pipeline_case):
    heads, h = pipeline_case
    state = _pipeline(heads, h, PIPELINE_POSITIONS, _config(), quadrupoles=False)
    assert not state.assembly.quadrupoles
    assert state.assembly.params.A_Theta is None
    assert state.solve_result.theta is None
    assert state.assembly.moment_dim == state.assembly.n_bonds + 4 * 6


def test_unwired_onsite_couplings_are_refused():
    heads = _heads(quadrupole_blocks=("A_Theta", "q_theta"), allow_truncated_a_theta=True)
    h = torch.randn(4, heads.irreps_in.dim)
    with pytest.raises(NotImplementedError, match="q_theta"):
        _pipeline(heads, h, PIPELINE_POSITIONS[:4], _config())


def test_quadrupoles_without_a_stiffness_head_are_refused():
    heads = _heads()
    h = torch.randn(4, heads.irreps_in.dim)
    with pytest.raises(ValueError, match="A_Theta"):
        _pipeline(heads, h, PIPELINE_POSITIONS[:4], _config())


def test_quadrupoles_require_the_gaussian_kernel_in_the_pipeline():
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    heads = _quadrupole_heads()
    h = torch.randn(4, heads.irreps_in.dim)
    with pytest.raises(ValueError, match="gaussian"):
        _pipeline(heads, h, PIPELINE_POSITIONS[:4], ElectrostaticConfig(multipole_kernel="thole"))


# ===========================================================================
# SPEC 13: dispersion
# ===========================================================================
def _dipole_coupling(pos, A_mu, ell, sigma, width_extra=0.0, a=0.5):
    kernels = require("kernels", "gaussian_multipole_width", "gaussian_damping", "dipole_interaction_tensor")
    width, damping, tensor = kernels
    n = pos.shape[0]
    r_vec, r, keep = _pair_geometry(pos)
    s_mu = width(A_mu, ell, 1, a)
    s = torch.sqrt(s_mu[:, None] ** 2 + s_mu[None] ** 2 + width_extra ** 2)
    l3, l5, _, _ = damping(r, s)
    return (tensor(r_vec, r, l3, l5) * (keep * ell[:, None] * ell[None])[..., None, None]
            ).permute(0, 2, 1, 3).reshape(3 * n, 3 * n)


def _eigen_zero_point(K, D, M):
    inverse = torch.linalg.inv(torch.linalg.cholesky(M))
    return 0.5 * (torch.sqrt(torch.linalg.eigvalsh(inverse @ K @ inverse.T)).sum()
                  - torch.sqrt(torch.linalg.eigvalsh(inverse @ D @ inverse.T)).sum())


def test_logdet_integral_is_the_zero_point_energy():
    dispersion_energy, inertia_fn = require("dispersion", "dispersion_energy", "dipole_inertia")
    quadrupole_pair_tensors, gaussian_widths_fn = require(
        "solver", "quadrupole_pair_tensors", "gaussian_widths"
    )
    width = require("kernels", "gaussian_multipole_width")
    g = torch.Generator().manual_seed(7)
    n = 6
    pos = torch.randn(n, 3, generator=g) * 1.8
    A_mu, A_theta = _spd(n, 3, 3.0, g, 0.5), _spd(n, 5, 2.0, g, 0.5)
    ell = 0.7 + 0.6 * torch.rand(n, generator=g)
    electrons = 1.0 + 2.0 * torch.rand(n, generator=g)
    sigma = 1.0 + torch.rand(n, generator=g)
    T = _dipole_coupling(pos, A_mu, ell, sigma)
    m = inertia_fn(electrons, ell)
    D = torch.block_diag(*A_mu)
    M = torch.block_diag(*[m[i] * torch.eye(3) for i in range(n)])
    reference = _eigen_zero_point(D + T, D, M)
    assert abs(float(dispersion_energy(T, A_mu, ell, electrons) / reference) - 1.0) < 1e-9

    r_vec, r, keep = _pair_geometry(pos)
    _, t12, t22 = quadrupole_pair_tensors(
        r_vec, r, keep, sigma, width(A_mu, ell, 1, 0.5), width(A_theta, ell, 2, 0.5), ell
    )
    Tmq = t12.permute(0, 2, 1, 3).reshape(3 * n, 5 * n)
    Tqq = t22.permute(0, 2, 1, 3).reshape(5 * n, 5 * n)
    omega_bar_sq = torch.diagonal(A_mu, dim1=-2, dim2=-1).sum(-1) / (3 * m)
    Dq = torch.block_diag(*A_theta)
    Mq = torch.block_diag(*[A_theta[i] / (4 * omega_bar_sq[i]) for i in range(n)])
    K_full = torch.cat([torch.cat([D + T, Tmq], 1), torch.cat([Tmq.T, Dq + Tqq], 1)], 0)
    reference_q = _eigen_zero_point(K_full, torch.block_diag(D, Dq), torch.block_diag(M, Mq))
    got_q = dispersion_energy(
        T, A_mu, ell, electrons, A_Theta=A_theta, dipole_quadrupole_pair=Tmq, quadrupole_pair=Tqq
    )
    assert abs(float(got_q / reference_q) - 1.0) < 1e-9


def test_pair_limit_is_the_london_slater_kirkwood_c6():
    """-E R^6 -> (3/2) a1 a2 w1 w2 / (w1 + w2), with w_i = sqrt(N_i hbar^2/m_e / alpha_i)."""
    dispersion_energy = require("dispersion", "dispersion_energy")
    HBAR2_OVER_ME = require("constants", "HBAR2_OVER_ME")
    alpha_volume = torch.tensor([1.2, 2.3])
    electrons = torch.tensor([1.3, 2.4])
    A_mu = (K_E / alpha_volume)[:, None, None] * torch.eye(3)
    ell = torch.ones(2)
    omega = torch.sqrt(electrons * HBAR2_OVER_ME * K_E / alpha_volume)
    c6 = 1.5 * alpha_volume.prod() * omega.prod() / omega.sum()
    R = 24.0
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, R]])
    energy = dispersion_energy(
        _dipole_coupling(pos, A_mu, ell, torch.ones(2)), A_mu, ell, electrons, n_quadrature=32
    )
    assert abs(float(-energy * R ** 6 / c6) - 1.0) < 1e-5


def test_trk_inertia_gives_the_hydrogen_excitation_energy():
    frequencies = require("dispersion", "isolated_frequencies")
    alpha_response = 0.6668 / K_E        # free H atom, 4.5 bohr^3
    omega = frequencies(torch.eye(3)[None] / alpha_response, torch.ones(1), torch.ones(1))
    assert torch.allclose(omega, torch.full((1, 3), 12.828), atol=2e-3)


def test_isolated_atom_has_no_dispersion():
    dispersion_energy = require("dispersion", "dispersion_energy")
    energy = dispersion_energy(torch.zeros(3, 3), 5.0 * torch.eye(3)[None], torch.ones(1), torch.ones(1))
    assert float(energy) == 0.0


@pytest.mark.parametrize("structure", ["cube", "octahedron", "fcc"])
def test_second_derivatives_are_finite_on_degenerate_spectra(structure):
    """Where eigh's eigenvector derivative divides by zero gaps, log-det stays finite."""
    dispersion_energy = require("dispersion", "dispersion_energy")
    if structure == "cube":
        base = torch.tensor(list(itertools.product([0.0, 3.0], repeat=3)))
    elif structure == "octahedron":
        base = 3.0 * torch.cat([torch.eye(3), -torch.eye(3)])
    else:
        cell = torch.tensor([[0.0, 0, 0], [0, 2.0, 2], [2, 0, 2], [2, 2, 0]])
        base = torch.cat([cell + 4.0 * torch.tensor(s, dtype=torch.float64)
                          for s in itertools.product([0, 1], repeat=3)])
    n = base.shape[0]
    A_mu = (K_E / 1.5) * torch.eye(3).repeat(n, 1, 1)
    pos = base.clone().requires_grad_(True)
    energy = dispersion_energy(_dipole_coupling(pos, A_mu, torch.ones(n), torch.ones(n)),
                               A_mu, torch.ones(n), torch.full((n,), 2.0))
    (grad,) = torch.autograd.grad(energy, pos, create_graph=True)
    (second,) = torch.autograd.grad((grad[0] ** 2).sum() + grad.sum(), pos)
    assert torch.isfinite(grad).all() and torch.isfinite(second).all()


def test_subtractive_range_separation_keeps_the_long_range_c6():
    """E = ZPE[A+T] - ZPE[A+T-T_w]: monomer short range removed, dimer C6 untouched."""
    dispersion_energy = require("dispersion", "dispersion_energy")
    unit = torch.tensor([[0.0, 0, 0], [1.1, 0, 0], [-0.4, 1.0, 0]])
    A_mu = (K_E / 1.5) * torch.eye(3).repeat(6, 1, 1)
    ell, electrons, sigma = torch.ones(6), torch.full((6,), 2.0), torch.ones(6)

    def energy(pos, lam):
        n = pos.shape[0]
        T = _dipole_coupling(pos, A_mu[:n], ell[:n], sigma[:n])
        full = dispersion_energy(T, A_mu[:n], ell[:n], electrons[:n])
        if lam == 0.0:
            return full
        wide = _dipole_coupling(pos, A_mu[:n], ell[:n], sigma[:n], width_extra=lam)
        return full - dispersion_energy(T - wide, A_mu[:n], ell[:n], electrons[:n])

    R = 16.0
    dimer = torch.cat([unit, unit + torch.tensor([0.0, 0, R])])
    c6_full = (energy(dimer, 0.0) - 2 * energy(unit, 0.0)) * R ** 6
    c6_separated = (energy(dimer, 2.0) - 2 * energy(unit, 2.0)) * R ** 6
    assert abs(float(c6_separated / c6_full) - 1.0) < 1e-6
    assert abs(float(energy(unit, 2.0))) < 0.5 * abs(float(energy(unit, 0.0)))


def test_pipeline_range_separation_matches_the_direct_form(pipeline_case):
    """assembly_dispersion_energy with dispersion_range reproduces the subtraction."""
    heads, h = pipeline_case
    full = _pipeline(heads, h, PIPELINE_POSITIONS, _config(dispersion=True))
    separated = _pipeline(heads, h, PIPELINE_POSITIONS, _config(dispersion=True, dispersion_range=2.0))
    assert float(separated.dispersion) < 0.0
    assert abs(float(separated.dispersion)) < abs(float(full.dispersion))
    assert abs(float(separated.energy - full.energy)) < 1e-14


def test_dispersion_needs_the_electron_count_head():
    heads = _heads()
    h = torch.randn(4, heads.irreps_in.dim)
    with pytest.raises(ValueError, match="dispersion=True"):
        _pipeline(heads, h, PIPELINE_POSITIONS[:4], _config(dispersion=True), quadrupoles=False)


def test_dispersion_range_needs_the_gaussian_kernel():
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    with pytest.raises(ValueError, match="gaussian"):
        ElectrostaticConfig(dispersion=True, dispersion_range=2.0, multipole_kernel="thole")
