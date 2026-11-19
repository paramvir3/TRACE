"""Runtime assertions -- SPEC sections 1.2, 1.4, 2.1 and 6.

Every assertion in SPEC 2.1 came from a measured failure, so every test here
uses the measured numbers rather than made-up ones.
"""

import math

import numpy as np
import pytest
import torch

from conftest import require


K_E = 14.399645  # SPEC 0


def _floor(sigma):
    """S_0 = sqrt(2) K_E / (sigma sqrt(pi))   (SPEC 1.2 / 17.3), computed here.

    The tight Gram bound: the Coulomb self-energy of one Gaussian charge of
    width sigma, i.e. the pair kernel at the self width sqrt(2) sigma.  (The
    package used to demand 2 K_E / (sigma sqrt(pi)) = sqrt(2) S_0; retracted.)
    """
    return math.sqrt(2.0) * K_E / (sigma * math.sqrt(math.pi))


def _coincident_min_eig(eta, sigma):
    """Smallest eigenvalue of [[eta, J(0; sqrt2 sigma)], [J, eta]] -- two sites at one point."""
    j0 = math.sqrt(2.0) * K_E / (sigma * math.sqrt(math.pi))
    return eta - j0


# ---------------------------------------------------------------------------
# SPEC 2.1 item 1 / SPEC 6: "eta < J(0) raises"
# ---------------------------------------------------------------------------
def test_mulliken_carbon_hardness_clears_the_gram_bound_at_sigma_1p5():
    """SPEC 17.3 retracts SPEC 1.2's old table row 1.

    At sigma = 1.5 A the tight bound is S_0 = sqrt(2) K_E / (1.5 sqrt(pi)) =
    7.659503 eV, so the Mulliken eta_C = 9.997 eV is ADMISSIBLE: the charge
    block is positive definite for every geometry.  The bound is tight: two
    coincident sites have min eig eta - S_0, negative just below it.
    """
    assert_hardness_floor, error = require(
        "diagnostics", "assert_hardness_floor", "ElectrostaticsAssertionError"
    )
    assert _floor(1.5) == pytest.approx(7.659503, abs=1e-6)
    margin = assert_hardness_floor(torch.tensor([9.997]), torch.tensor([1.5]))
    assert float(margin) == pytest.approx(9.997 - _floor(1.5), abs=1e-9)
    assert _coincident_min_eig(_floor(1.5) - 1e-3, 1.5) < 0.0 < _coincident_min_eig(9.997, 1.5)
    with pytest.raises(error) as excinfo:
        assert_hardness_floor(torch.tensor([7.0]), torch.tensor([1.5]))
    assert "7.659" in str(excinfo.value)


def test_hardness_floor_accepts_eta_at_or_above_j_zero():
    """SPEC 1.2 says ``eta_i >= J(0)``: the boundary itself is admissible."""
    assert_hardness_floor = require("diagnostics", "assert_hardness_floor")
    sigma = torch.tensor([1.5, 1.5, 2.0])
    eta = torch.tensor([_floor(1.5), 10.9, _floor(2.0) + 1.0e-9])
    margin = assert_hardness_floor(eta, sigma)
    assert torch.all(margin >= -1.0e-12)


def test_a_one_nanoelectronvolt_deficit_is_rejected():
    """SPEC 1.2 makes the floor an inequality, not a guideline.

    Below ``J(0)`` the energy is unbounded below and the failure is silent, so
    there is no "close enough": the assertion has to fire on the smallest
    representable violation, not merely on chemically-sized ones like the
    9.997 eV above.  1e-9 eV against a 10.83 eV floor is 9e-11 relative, which
    is where a tolerance-based check would wave it through.
    """
    assert_hardness_floor, error = require(
        "diagnostics", "assert_hardness_floor", "ElectrostaticsAssertionError"
    )
    floor = _floor(1.5)
    with pytest.raises(error):
        assert_hardness_floor(
            torch.tensor([floor - 1.0e-9]), torch.tensor([1.5])
        )
    # The boundary itself is admissible (SPEC 1.2 writes ``eta_i >= J(0)``),
    # so the check must be one-sided and not a band around the floor.
    assert_hardness_floor(torch.tensor([floor]), torch.tensor([1.5]))
    assert_hardness_floor(torch.tensor([floor + 1.0e-9]), torch.tensor([1.5]))


def test_the_floor_moves_with_sigma_not_with_a_hard_coded_constant():
    """``S_0 = sqrt(2) K_E / (sigma sqrt(pi))``: 7.66 eV is the value at 1.5 A only.

    A check that hard-coded 7.66 would accept an indefinite system at
    sigma = 1.0 (bound 11.489 eV) and reject a sound one at sigma = 2.0 (bound
    5.745 eV).  Both directions are tested.
    """
    assert_hardness_floor, error = require(
        "diagnostics", "assert_hardness_floor", "ElectrostaticsAssertionError"
    )
    assert _floor(1.0) == pytest.approx(11.4892542, abs=1e-6)
    with pytest.raises(error):
        assert_hardness_floor(torch.tensor([10.0]), torch.tensor([1.0]))
    assert _floor(2.0) == pytest.approx(5.7446271, abs=1e-6)
    assert_hardness_floor(torch.tensor([6.0]), torch.tensor([2.0]))


@pytest.mark.parametrize(
    "element,eta",
    [("Na", 4.6), ("Al", 6.0), ("Cu", 6.5), ("Pt", 7.4)],
)
def test_every_common_metal_hardness_is_rejected(element, eta):
    """SPEC 1.2, table row 3: "common metals at real bond length | all below
    threshold: Na 4.6, Al 6.0, Cu 6.5, Pt 7.4 eV"."""
    assert_hardness_floor, error = require(
        "diagnostics", "assert_hardness_floor", "ElectrostaticsAssertionError"
    )
    with pytest.raises(error):
        assert_hardness_floor(torch.tensor([eta]), torch.tensor([1.5]))


def test_hardness_floor_reports_the_worst_offender_among_many():
    assert_hardness_floor, error = require(
        "diagnostics", "assert_hardness_floor", "ElectrostaticsAssertionError"
    )
    eta = torch.tensor([12.0, 9.997, 13.0, 4.6])
    sigma = torch.full((4,), 1.5)
    with pytest.raises(error) as excinfo:
        assert_hardness_floor(eta, sigma)
    assert "atom 3" in str(excinfo.value)  # 4.6 eV is the worst margin


# ---------------------------------------------------------------------------
# SPEC 2.1 item 2 / SPEC 6: "asymmetric kappa raises"
# ---------------------------------------------------------------------------
def test_asymmetric_kappa_is_rejected():
    """SPEC 1.4: ``Q_i . K_j != Q_j . K_i``, so an unsymmetrised attention logit
    yields kappa_ij != kappa_ji and the energy 0.5 kappa_ij t_ij^2 depends on
    which direction of the pair the code visited first."""
    assert_symmetric_kappa, error = require(
        "diagnostics", "assert_symmetric_kappa", "ElectrostaticsAssertionError"
    )
    # Directed edge list: (0,1) and (1,0) are the same physical bond.
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    asymmetric = torch.tensor([1.5, 1.9, 2.0, 2.0])  # bond (0,1) disagrees by 0.4
    with pytest.raises(error) as excinfo:
        assert_symmetric_kappa(asymmetric, edge_index)
    assert "(0, 1)" in str(excinfo.value)


def test_symmetric_kappa_is_accepted():
    assert_symmetric_kappa = require("diagnostics", "assert_symmetric_kappa")
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    symmetric = torch.tensor([1.7, 1.7, 2.0, 2.0])
    worst = assert_symmetric_kappa(symmetric, edge_index)
    assert float(worst) == 0.0


def test_non_positive_kappa_is_rejected():
    """SPEC 2.1 item 2: ``kappa_ij > 0`` for all bonds, at construction."""
    assert_positive_kappa, error = require(
        "diagnostics", "assert_positive_kappa", "ElectrostaticsAssertionError"
    )
    for bad in (0.0, -1.0e-12, -3.0):
        with pytest.raises(error):
            assert_positive_kappa(torch.tensor([1.0, bad, 2.0]))


# ---------------------------------------------------------------------------
# SPEC 2.1 item 3 / SPEC 6: the Schur-complement trap
# ---------------------------------------------------------------------------
#
# Deliberately constructed 1x1 blocks that reproduce the audit numbers exactly.
#
#   charge block      H_qq   = 0.58     (positive definite on its own)
#   dipole block      A_mu   = 16.45    (positive definite on its own)
#   coupled K = [[0.58, c], [c, 16.45]] must have min eigenvalue -1.05
#
# min_eig = (tr - sqrt((a-d)^2 + 4 c^2)) / 2 = -1.05
#   =>  sqrt(15.87^2 + 4 c^2) = 17.03 + 2.10 = 19.13
#   =>  c^2 = (19.13^2 - 15.87^2) / 4 = 28.5250...
#   =>  c   = 5.340880077290633
#
# and the Schur complement is 0.58 - c^2 / 16.45 = -1.15404... < 0.
Q_BLOCK = 0.58
MU_BLOCK = 16.45
COUPLED_MIN_EIG = -1.05
COUPLING = math.sqrt(((Q_BLOCK + MU_BLOCK - 2.0 * COUPLED_MIN_EIG) ** 2
                      - (MU_BLOCK - Q_BLOCK) ** 2) / 4.0)


def _audit_blocks():
    return (
        torch.tensor([[Q_BLOCK]]),
        torch.tensor([[COUPLING]]),
        torch.tensor([[MU_BLOCK]]),
    )


def _blocks_only_check(K_qq, A_dipole, tol=0.0):
    """The WRONG cheap check: are the two diagonal blocks each PD?

    Kept in the suite as an executable statement of what SPEC 2.1 item 3 calls
    "THE trap", so the next test can show it passing on a system that is not
    positive definite.
    """
    return (
        float(torch.linalg.eigvalsh(K_qq).min()) > tol
        and float(torch.linalg.eigvalsh(A_dipole).min()) > tol
    )


def test_audit_construction_really_has_positive_blocks_and_a_negative_coupled_system():
    """Validate the test bed before using it: +0.58, +16.45, coupled -1.05."""
    K_qq, T01, A_mu = _audit_blocks()
    coupled = torch.tensor(
        [[Q_BLOCK, COUPLING], [COUPLING, MU_BLOCK]]
    )
    assert float(torch.linalg.eigvalsh(K_qq).min()) == pytest.approx(0.58, abs=1e-12)
    assert float(torch.linalg.eigvalsh(A_mu).min()) == pytest.approx(16.45, abs=1e-12)
    # SPEC 2.1 item 3: the audit's full coupled minimum eigenvalue.
    assert float(torch.linalg.eigvalsh(coupled).min()) == pytest.approx(-1.05, abs=1e-10)


def test_a_blocks_only_check_fails_to_catch_the_indefinite_coupled_system():
    """SPEC 2.1 item 3: "Checking the two blocks separately is THE trap".

    This test asserts the trap: the cheap check *passes* on a system that is
    not positive definite.  It is the control for the next test.
    """
    K_qq, _, A_mu = _audit_blocks()
    assert _blocks_only_check(K_qq, A_mu) is True


def test_schur_complement_catches_what_the_blocks_miss():
    """SPEC 2.1 item 3: S = H_qq - T01^T (A_mu + T_Thole)^-1 T01 is negative."""
    schur_complement = require("diagnostics", "schur_complement")
    K_qq, T01, A_mu = _audit_blocks()
    schur = schur_complement(K_qq, T01, A_mu)
    # 0.58 - 5.340880077^2 / 16.45 = -1.1540425531914894, derived above.
    assert float(schur) == pytest.approx(-1.1540425531914894, abs=1e-12)
    assert float(torch.linalg.eigvalsh(schur).min()) < 0.0


def test_positive_definite_assertion_rejects_the_audit_case():
    """The runtime assertion itself must fire on the +0.58 / +16.45 / -1.05 case."""
    assert_pd, error = require(
        "diagnostics",
        "assert_positive_definite_at_solution",
        "ElectrostaticsAssertionError",
    )
    K_qq, T01, A_mu = _audit_blocks()
    with pytest.raises(error):
        assert_pd(K_qq, T01, A_mu, enabled=True)


def test_positive_definite_assertion_accepts_a_genuinely_stable_system():
    """Guard against an assertion that rejects everything."""
    assert_pd = require("diagnostics", "assert_positive_definite_at_solution")
    K_qq = torch.tensor([[4.0, 0.3], [0.3, 5.0]])
    T01 = torch.tensor([[0.5, 0.1], [0.0, 0.4], [0.2, 0.2]])
    A_mu = torch.eye(3) * 12.0
    result = assert_pd(K_qq, T01, A_mu, enabled=True)
    assert result is not None


def test_schur_trap_also_appears_in_a_multi_mode_system():
    """The same trap with 3 charge modes and 6 dipole modes, so the 1x1 case
    cannot be dismissed as a special case."""
    schur_complement, assert_pd, error = require(
        "diagnostics",
        "schur_complement",
        "assert_positive_definite_at_solution",
        "ElectrostaticsAssertionError",
    )
    torch.manual_seed(3)
    K_qq = torch.diag(torch.tensor([0.60, 1.20, 2.40]))
    A_mu = torch.diag(torch.tensor([16.45, 18.0, 20.0, 22.0, 24.0, 26.0]))
    # Coupling large enough that T01^T A^-1 T01 overwhelms K_qq in one direction.
    T01 = torch.zeros(6, 3)
    T01[0, 0] = 5.5
    T01[1, 0] = 4.0
    T01[2, 1] = 2.0

    assert float(torch.linalg.eigvalsh(K_qq).min()) > 0.0
    assert float(torch.linalg.eigvalsh(A_mu).min()) > 0.0
    assert _blocks_only_check(K_qq, A_mu) is True  # the cheap check still passes

    schur = schur_complement(K_qq, T01, A_mu)
    assert float(torch.linalg.eigvalsh(schur).min()) < 0.0
    with pytest.raises(error):
        assert_pd(K_qq, T01, A_mu, enabled=True)


def test_constraint_projection_ignores_directions_the_constraint_forbids():
    """SPEC 2.1 item 3 projects onto the constraints before taking eigenvalues.

    A direction that violates the charge constraint cannot be explored, so
    negative curvature along it is not a physical instability.
    """
    assert_pd = require("diagnostics", "assert_positive_definite_at_solution")
    # Curvature -1 along (1, 1)/sqrt(2) and +3 along (1, -1)/sqrt(2).
    K_qq = torch.tensor([[1.0, -2.0], [-2.0, 1.0]])
    A_mu = torch.eye(1) * 10.0
    T01 = torch.zeros(1, 2)
    assert float(torch.linalg.eigvalsh(K_qq).min()) < 0.0
    # The constraint q0 + q1 = const removes exactly the unstable direction.
    constraint = torch.tensor([[1.0, 1.0]])
    result = assert_pd(K_qq, T01, A_mu, constraint, enabled=True)
    assert result is not None


def test_null_space_basis_handles_the_rank_deficient_bond_space_constraint():
    """SPEC/diagnostics: ``C @ B`` is always rank deficient because 1^T B = 0."""
    null_space_basis = require("diagnostics", "null_space_basis")
    build_incidence = require("topology", "build_incidence")
    incidence, _ = build_incidence(torch.tensor([[0, 1], [1, 2]]), 3)
    global_constraint = torch.ones(1, 3)
    bond_space = global_constraint @ incidence  # identically zero: 1^T B = 0
    assert float(bond_space.abs().max()) == 0.0
    basis = null_space_basis(bond_space, incidence.shape[1])
    # A zero constraint removes no directions.
    assert basis.shape == (2, 2)
    torch.testing.assert_close(basis @ basis.T, torch.eye(2), atol=1e-13, rtol=0)


# ---------------------------------------------------------------------------
# SPEC 2.1 items 4 and 5
# ---------------------------------------------------------------------------
def test_a_saddle_with_zero_gradient_is_not_accepted_as_a_minimum():
    """SPEC 2.1 item 4: "Newton converges silently onto saddles and returns
    negative polarizability"; stationarity alone is not enough."""
    assert_is_minimum, error = require(
        "diagnostics", "assert_is_minimum", "ElectrostaticsAssertionError"
    )
    saddle = torch.tensor([[2.0, 0.0], [0.0, -3.0]])
    with pytest.raises(error):
        assert_is_minimum(saddle, gradient=torch.zeros(2), enabled=True)
    minimum = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    assert assert_is_minimum(minimum, gradient=torch.zeros(2), enabled=True) is not None


def test_fragment_charge_constraint_is_checked_to_1e_minus_10():
    """SPEC 2.1 item 5: ``|sum_{i in F} q_i - Q_F| < 1e-10`` per fragment."""
    assert_fragment_charges, error = require(
        "diagnostics", "assert_fragment_charges", "ElectrostaticsAssertionError"
    )
    constraint = torch.tensor([[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    targets = torch.tensor([1.0, 0.0])
    good = torch.tensor([0.5, 0.25, 0.25, 0.0])
    assert_fragment_charges(good, constraint, targets)

    bad = torch.tensor([0.5, 0.25, 0.25 + 1.0e-8, 0.0])
    with pytest.raises(error):
        assert_fragment_charges(bad, constraint, targets)


def test_validation_mode_context_manager_toggles_the_expensive_checks():
    """SPEC 2.1 item 3 is O(n^3); it is opt-in but must be reachable from CI."""
    validation_mode, validation_enabled, assert_pd = require(
        "diagnostics",
        "validation_mode",
        "validation_enabled",
        "assert_positive_definite_at_solution",
    )
    K_qq, T01, A_mu = _audit_blocks()
    before = validation_enabled()
    try:
        with validation_mode(False):
            assert assert_pd(K_qq, T01, A_mu) is None  # skipped, not raised
        with validation_mode(True):
            with pytest.raises(Exception):
                assert_pd(K_qq, T01, A_mu)
    finally:
        assert validation_enabled() == before
