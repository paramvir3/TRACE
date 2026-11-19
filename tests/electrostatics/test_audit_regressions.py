"""Guards for three defects found by the module audit.

Each is the same shape of failure: a safety mechanism that exists, is correct,
and is not reached.  None of them changes a number in a passing run, which is
exactly why none of them was caught by the 557 tests that were already green.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from conftest import require


# ---------------------------------------------------------------------------
# 1. SPEC 2.1 items 3 and 4 were inert on every production path
# ---------------------------------------------------------------------------
def test_the_spec_2_1_gates_actually_reach_the_solver():
    """``check_positive_definite`` / ``check_is_minimum`` must reach SolverOptions.

    ``energy._solver_options`` forwarded only ``thole_a`` and
    ``fragment_charge_tol``, so ``SolverOptions.validate`` stayed ``None``.
    ``None`` defers to the process-global validation mode, and
    ``diagnostics.set_validation_mode`` is never called anywhere in this package
    -- only defined.  Setting either flag therefore did nothing, silently, and
    ``assert_is_minimum`` had no call site in ``energy.py`` at all.
    """
    energy = require("energy")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")

    on = energy._solver_options(
        dataclasses.replace(
            ElectrostaticConfig(),
            check_positive_definite=True,
            check_is_minimum=True,
        )
    )
    assert on.validate is True, (
        "check_positive_definite=True did not set SolverOptions.validate; the "
        "SPEC 2.1 item 3 check is inert"
    )
    assert on.check_is_minimum is True, (
        "check_is_minimum=True did not reach SolverOptions; SPEC 2.1 item 4 is "
        "inert. SolverOptions' own docstring warns about exactly this: 'A "
        "caller that exposes item 3 through a config flag and leaves this one "
        "bundled has item 4 silently off.'"
    )

    off = energy._solver_options(ElectrostaticConfig())
    assert off.validate is False and off.check_is_minimum is False, (
        "the default config must leave both O(n^3) checks off, explicitly "
        "rather than by deferring to a global nobody sets"
    )


def test_the_canonical_saddle_is_caught_when_the_gate_is_on():
    """SPEC's own trap: both blocks positive, coupled Schur complement negative.

    With ``enabled`` left to default this returned without raising, which is how
    the check came to be documented as the headline safety property while never
    running.
    """
    diagnostics = require("diagnostics")
    K_qq = torch.tensor([[0.58]], dtype=torch.float64)
    A_mu = torch.tensor([[16.45]], dtype=torch.float64)
    T01 = torch.tensor([[3.6]], dtype=torch.float64)

    schur = float(K_qq - T01.T @ torch.linalg.inv(A_mu) @ T01)
    assert schur < 0.0, f"fixture is not a saddle (Schur = {schur:.4f})"

    with pytest.raises(Exception):
        diagnostics.assert_positive_definite_at_solution(
            K_qq, T01, A_mu, enabled=True
        )


# ---------------------------------------------------------------------------
# 2. Fractional-spin telemetry claimed to be exact when it was a bound
# ---------------------------------------------------------------------------
def test_frozen_bond_curvature_is_not_labelled_exact_when_bonds_exist():
    """``bonds_relaxed`` must distinguish "no bonds" from "bonds not relaxed".

    Without ``incidence`` the callee computes ``n_bonds = 0`` and returns
    ``bonds_relaxed=True`` -- "this is the model's exact value" -- for every
    system that has bonds, suppressing the ``summary()`` caveat.  Measured 40%
    high against the relaxed value on a 6-atom ring with non-uniform ``J^x``.
    """
    fractional_spin_curvature = require("diagnostics", "fractional_spin_curvature")

    n_atoms = 6
    j_exchange = torch.tensor(
        [0.5, 3.5, 1.0, 2.5, 0.8, 1.7], dtype=torch.float64
    )
    reference = torch.ones(n_atoms, 1, dtype=torch.float64)
    incidence = torch.zeros(n_atoms, n_atoms, dtype=torch.float64)
    for bond in range(n_atoms):
        incidence[bond, bond] = 1.0
        incidence[(bond + 1) % n_atoms, bond] = -1.0

    frozen = fractional_spin_curvature(
        j_exchange, reference, incidence=incidence, relax_bonds=False
    )
    assert frozen.bonds_relaxed is False, (
        "a frozen-bond curvature on a system WITH bonds is an upper bound, not "
        "the exact value; bonds_relaxed=True suppresses the caveat that says so"
    )
    assert "upper bound" in frozen.summary().lower(), (
        f"summary() must carry the caveat; got: {frozen.summary()[:120]}"
    )

    bondless = fractional_spin_curvature(
        j_exchange, reference, incidence=None, relax_bonds=False
    )
    assert bondless.bonds_relaxed is True, (
        "with no bonds there is nothing to relax, so the frozen value IS exact "
        "and claiming otherwise would understate what is known"
    )
    assert float(frozen.curvature) == float(bondless.curvature), (
        "relax_bonds=False uses neither incidence nor kappa_spin, so passing "
        "incidence must change only the LABEL, never the number"
    )


# ---------------------------------------------------------------------------
# 3. The octupole channel bypassed the shape check built for (alpha, d)
# ---------------------------------------------------------------------------
def test_a_system_octupole_cannot_broadcast_onto_a_per_atom_beta():
    """The failure ``_require_beta_shapes`` exists to stop, on the third input.

    A ``[3, 3, 3]`` system octupole added to a per-atom beta broadcasts to a
    well-shaped, correctly equivariant, entirely meaningless result in which
    every atom carries the whole system's octupole.  The equivalent
    ``(alpha, d)`` mismatch has always raised; this one did not.
    """
    observables = require("observables")
    hyperpolarizability = observables.hyperpolarizability
    beta_full = observables.BETA_FULL

    torch.manual_seed(0)
    alpha = torch.randn(2, 3, 3, dtype=torch.float64)
    alpha = alpha + alpha.transpose(-1, -2)
    d = torch.zeros(2, 3, dtype=torch.float64)

    with pytest.raises(ValueError, match="octupole"):
        hyperpolarizability(
            alpha, d, octupole=torch.randn(3, 3, 3, dtype=torch.float64),
            form=beta_full,
        )

    # the control: the (alpha, d) path has always caught its own version
    with pytest.raises(ValueError):
        hyperpolarizability(alpha, torch.randn(3, dtype=torch.float64), form=beta_full)

    # and a correctly shaped octupole must still be accepted
    good = hyperpolarizability(
        alpha, d, octupole=torch.randn(2, 3, 3, 3, dtype=torch.float64),
        form=beta_full,
    )
    assert good.shape == (2, 3, 3, 3)
