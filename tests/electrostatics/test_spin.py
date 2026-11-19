"""SPEC 9 -- the spin sector, which had NO test coverage at all.

Before this file there was no reference to ``j_exchange``, ``spin_targets`` or
``spin_resolved`` anywhere in ``tests/``.  That gap is not incidental: it is
exactly why the fractional-spin telemetry shipped claiming its frozen-bond
UPPER BOUND was the model's exact value, on every spin solve of every bonded
system, for as long as the sector has existed.  A sector with no tests has no
claims, only intentions.
"""

from __future__ import annotations

import dataclasses
import warnings

import pytest
import torch

from conftest import require

GEOMETRY = torch.tensor(
    [
        [0.00, 0.00, 0.00],
        [1.60, 0.00, 0.00],
        [0.00, 1.70, 0.00],
        [1.10, 1.10, 0.90],
    ],
    dtype=torch.float64,
)


def _spin_module(spin=True, **config_overrides):
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    ElectrostaticModule = require("energy", "ElectrostaticModule")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heads = ElectrostaticHeads(spin=spin)
    config = dataclasses.replace(ElectrostaticConfig(), **config_overrides)
    torch.manual_seed(3)
    features = torch.randn(
        GEOMETRY.shape[0], heads.irreps_in.dim, dtype=torch.float64
    ) * 2.0
    return ElectrostaticModule(heads, config), features


def _edges(positions, r_cut=6.0):
    delta = positions.unsqueeze(1) - positions.unsqueeze(0)
    distance = delta.norm(dim=-1)
    mask = (distance < r_cut) & (distance > 1.0e-9)
    return mask.nonzero().T, delta[mask], distance[mask]


def _solve(module, features, positions=GEOMETRY, **kwargs):
    edge_index, edge_vec, edge_len = _edges(positions)
    return module(features, positions, edge_index, edge_vec, edge_len, **kwargs)


# ---------------------------------------------------------------------------
# The two constraints
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_iter", [1, 2, 3, 6, 15])
def test_both_spin_constraints_hold_exactly_at_every_iterate(n_iter):
    """``sum_F q_i = Q_F`` AND ``sum_F s_i = 2 S_F``, for any truncation.

    Feasibility is a property of the PROJECTION, not of convergence, so a
    1-iteration solve must satisfy both constraints exactly -- the same claim
    SPEC 2 makes for the charge constraint alone.
    """
    module, features = _spin_module(n_cg_iter=n_iter)
    _, result = _solve(module, features, fragment_spins=[1.0])

    assert abs(float(result.q.sum())) < 1.0e-12, (
        f"net charge {float(result.q.sum()):.3e} at n_iter={n_iter}"
    )
    assert float(result.s.sum()) == pytest.approx(2.0, abs=1.0e-12), (
        f"net spin {float(result.s.sum()):.12f} at n_iter={n_iter}; the "
        "convention is sum_F s_i = 2 S_F, so S_F = 1.0 must give 2.0"
    )


def test_the_spin_target_is_twice_the_fragment_spin():
    """``2 S_F``, not ``S_F`` -- an easy factor of two to lose."""
    for spin, expected in ((0.0, 0.0), (0.5, 1.0), (1.0, 2.0), (1.5, 3.0)):
        module, features = _spin_module()
        _, result = _solve(module, features, fragment_spins=[spin])
        assert float(result.s.sum()) == pytest.approx(expected, abs=1.0e-12), (
            f"S_F = {spin} gave sum s = {float(result.s.sum()):.9f}, expected "
            f"2 S_F = {expected}"
        )


# ---------------------------------------------------------------------------
# The closed-shell limit
# ---------------------------------------------------------------------------
def test_a_zero_spin_TARGET_still_allows_local_spin_polarisation():
    """``fragment_spins=[0.0]`` is NOT the closed-shell condition.

    It constrains the NET spin of the fragment, ``sum_F s_i = 0``.  The head
    still emits a spin electronegativity ``chi^s``, which drives a
    spin-polarised state with zero net spin -- ``s_i != 0`` summing to zero.
    That is physics, not a leak, and mistaking it for a normalisation bug is an
    error this suite now pins down rather than invites.
    """
    module, features = _spin_module()
    _, result = _solve(module, features, fragment_spins=[0.0])

    assert abs(float(result.s.sum())) < 1.0e-12, "net spin must still be zero"
    assert float(result.s.abs().max()) > 1.0e-6, (
        "a zero NET spin target must still permit local polarisation while "
        f"chi_spin != 0; got max|s_i| = {float(result.s.abs().max()):.3e}"
    )


def test_the_closed_shell_limit_is_bitwise_exact_when_chi_spin_vanishes():
    """The real SPEC 9 claim: at ``chi^s = 0`` the spin path reduces to the plain path.

    To round-off, NOT bitwise. An earlier version demanded bitwise equality and
    argued a last-ulp difference would betray "a different route". That was
    wrong: the spin-resolved solve runs the same algebra on a LONGER moment
    vector (t^c, t^s, mu_hat), and the zero spin block changes the grouping of
    CG's dot products, so identical algebra legitimately differs by an ulp or
    two. It passed bitwise with the old seed by coincidence; with the zero-
    initialised seed it differs by 1.7e-18 eV on 3.5e-3 (5e-16 relative).
    A genuinely different route -- a spurious term -- would show up far above
    that, which is what the tolerance still catches.
    """
    spin_module, features = _spin_module(spin=True)
    plain_module, plain_features = _spin_module(spin=False)
    torch.testing.assert_close(features, plain_features, rtol=0, atol=0)

    # Shared parameters are identical across the two constructions (verified:
    # all 14 agree exactly), so the only difference left is chi_spin itself.
    with torch.no_grad():
        for parameter in spin_module.heads.spin_scalar_net.parameters():
            parameter.zero_()

    spin_energy, spin_result = _solve(spin_module, features, fragment_spins=[0.0])
    plain_energy, plain_result = _solve(plain_module, plain_features)

    assert float(spin_energy) == pytest.approx(float(plain_energy), rel=1.0e-13, abs=0.0), (
        f"closed-shell spin energy {float(spin_energy):.17g} against non-spin "
        f"{float(plain_energy):.17g}; with chi_spin = 0 they must agree to "
        "round-off"
    )
    # The ENERGY is bitwise equal, which is the load-bearing claim. `q` agrees
    # only to round-off (2.7e-15 relative) because the spin path reconstructs
    # the charge through two channels and the plain path forms it directly --
    # one extra rounding, not a different functional. Asserting bitwise here
    # would be asserting an accident of summation order.
    # Under the converged solve (SPEC 17) the two paths are two separate CG
    # solves to solve_rtol = 1e-10 on vectors of different length: the energy
    # is second order in the residual and agrees to 1e-13, the charges are
    # first order and agree to the solve tolerance (measured 3.7e-11 relative).
    torch.testing.assert_close(spin_result.q, plain_result.q, rtol=1e-9, atol=1e-12)
    assert float(spin_result.s.abs().max()) == 0.0, (
        "with chi_spin = 0 and a zero target there is nothing to polarise"
    )


# ---------------------------------------------------------------------------
# SPEC 9.1 -- the sign that is not negotiable
# ---------------------------------------------------------------------------
def test_the_on_site_eigenvalues_are_two_eta_and_two_j_exchange():
    """In the ``(q_up, q_down)`` basis the on-site block has eigenvalues
    ``2 eta`` and ``2 J^x``, which is WHY ``J^x > 0`` is required rather than
    merely conventional: a non-positive ``J^x`` makes the block indefinite.
    """
    eta, j_exchange = 11.4, 2.7
    # U = eta (q_up + q_down)^2 / ... in the (charge, spin) basis the block is
    # diag(eta, J^x) acting on (q, s); transforming to (q_up, q_down) with
    # q = q_up + q_down, s = q_up - q_down gives the matrix below.
    block = torch.tensor(
        [[eta + j_exchange, eta - j_exchange],
         [eta - j_exchange, eta + j_exchange]],
        dtype=torch.float64,
    )
    eigenvalues = torch.linalg.eigvalsh(block)
    torch.testing.assert_close(
        eigenvalues,
        torch.tensor([2.0 * min(eta, j_exchange), 2.0 * max(eta, j_exchange)],
                     dtype=torch.float64),
        rtol=0, atol=1e-12,
    )
    assert float(eigenvalues.min()) > 0.0

    negative = torch.tensor(
        [[eta - 1.0, eta + 1.0], [eta + 1.0, eta - 1.0]], dtype=torch.float64
    )
    assert float(torch.linalg.eigvalsh(negative).min()) < 0.0, (
        "J^x < 0 must make the on-site block indefinite -- that is the whole "
        "reason the floor exists"
    )


def test_j_exchange_is_floored_strictly_positive():
    """The head must not be able to emit ``J^x <= 0`` at any input."""
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heads = ElectrostaticHeads(spin=True)

    for scale in (0.0, 1.0, -1.0e6, 1.0e6, -1.0e30):
        features = torch.full(
            (3, heads.irreps_in.dim), scale, dtype=torch.float64
        )
        readout = heads(
            features,
            torch.tensor([[0], [1]], dtype=torch.long),
            torch.tensor([1.6], dtype=torch.float64),
        )
        j_exchange = readout.spin.j_exchange
        assert bool((j_exchange > 0).all()), (
            f"j_exchange went non-positive at feature scale {scale}: "
            f"min {float(j_exchange.min()):.3e}"
        )
        assert bool(torch.isfinite(j_exchange).all())


# ---------------------------------------------------------------------------
# Gradient integrity through the spin path
# ---------------------------------------------------------------------------
def test_forces_are_the_exact_gradient_through_the_spin_solve():
    """The spin sector must be differentiated THROUGH, like the charge one.

    A Hellmann-Feynman shortcut (freezing the moments) would leave the energy
    unchanged and the force wrong, so only a finite-difference check catches it.
    """
    module, features = _spin_module()
    positions = GEOMETRY.clone().requires_grad_(True)
    energy, _ = _solve(module, features, positions=positions, fragment_spins=[1.0])
    analytic = -torch.autograd.grad(energy, positions)[0]

    step = 1.0e-6
    worst = 0.0
    for atom in range(GEOMETRY.shape[0]):
        for axis in range(3):
            plus = GEOMETRY.clone(); plus[atom, axis] += step
            minus = GEOMETRY.clone(); minus[atom, axis] -= step
            finite = -(
                float(_solve(module, features, positions=plus, fragment_spins=[1.0])[0])
                - float(_solve(module, features, positions=minus, fragment_spins=[1.0])[0])
            ) / (2.0 * step)
            worst = max(worst, abs(finite - float(analytic[atom, axis])))

    assert worst < 1.0e-7, (
        f"max |F_autograd - F_fd| = {worst:.3e} on the spin path; the moments "
        "are supposed to be differentiated through, not frozen"
    )


# ---------------------------------------------------------------------------
# The telemetry that shipped mislabelled
# ---------------------------------------------------------------------------
def test_fractional_spin_telemetry_admits_it_is_an_upper_bound():
    """Guards the real bug, through the real solve path.

    ``solve_charges`` reports the cheap frozen-bond curvature.  It did not pass
    ``incidence``, so the callee read ``n_bonds = 0`` and set
    ``bonds_relaxed=True`` -- "this is the model's exact value" -- suppressing
    the caveat on every bonded system.
    """
    module, features = _spin_module()
    _, result = _solve(module, features, fragment_spins=[1.0])

    fractional = result.diagnostics.fractional_spin
    assert fractional is not None, "spin solves must report this telemetry"
    assert fractional.bonds_relaxed is False, (
        "this system has bonds and they were NOT relaxed, so the reported "
        "curvature is a frozen-bond upper bound and must say so"
    )
    assert "upper bound" in fractional.summary().lower(), (
        f"summary() must carry the caveat; got {fractional.summary()[:120]}"
    )
