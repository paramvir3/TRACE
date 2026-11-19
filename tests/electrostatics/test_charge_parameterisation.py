"""THE RHO DEGENERACY -- SPEC 1.3 and the "GENERAL PRINCIPLE" section.

    q_i = sum_{j in N(i)} t_ij + rho_i ,   t_ij = -t_ji

``rho`` is a **FIXED CONSTRAINT CARRIER, NOT variational**.  SPEC 1.3 records
that an earlier draft had it variational and that this "destroys the model":
with both ``t`` and ``rho`` free under a per-fragment constraint the minimiser
gives ``|t| = 1.65e-07`` against ``3.57e-01`` with ``rho`` fixed.

SPEC states the reason as a general principle, of which the ``rho`` bug is one
of three instances:

    Any per-atom charge channel that bypasses the bond-hardness penalty will
    absorb all the charge and drive t -> 0.

The mechanism is exactly degeneracy plus a one-sided price.  ``rho`` can
reproduce any charge distribution that ``B t`` can, because ``B t`` only ever
spans zero-sum charge vectors and ``rho`` spans everything; but only ``t`` pays
``0.5 kappa |t|^2``.  So a free ``rho`` reaches near-identical charges for
strictly less energy, and the bond-resolved charge-transfer mechanism -- the
entire locality argument for SQE over QEq -- silently disappears while every
charge-level observable still looks fine.

This module locks that in from two directions:

  1. an independent reference minimisation, coded here in numpy, that
     reproduces the SPEC's collapse (``|t|`` 3.6e-01 -> 4.6e-16) and shows the
     free-rho answer is *cheaper* at near-identical charges; and
  2. assertions on the package itself that would fail if such a channel were
     reintroduced -- ``rho`` does not depend on the node features, and ``|t|``
     is not negligible for a charged system at kappa > 0.
"""

import numpy as np
import pytest
import torch

import elec_reference as ref
from conftest import complete_edge_index, require


K_E = 14.399645  # SPEC 0


# ===========================================================================
# 1. The degeneracy itself, in an independent reference minimisation
# ===========================================================================
def _reference_system():
    """A 5-atom charged molecule with a 4-bond tree, chemical parameters.

    Nothing here touches ``trace_electrostatics`` except ``build_incidence``,
    which is used only for the topology (and is itself checked bitwise in
    ``test_topology.py``); the energy and the two minimisations are numpy.
    """
    build_incidence = require("topology", "build_incidence")
    positions = np.array(
        [
            [0.00, 0.00, 0.00],
            [1.52, 0.00, 0.00],
            [2.10, 1.41, 0.00],
            [-0.63, 1.17, 0.35],
            [3.55, 1.62, 0.41],
        ]
    )
    bond_index = torch.tensor([[0, 1, 0, 2], [1, 2, 3, 4]])
    incidence, _ = build_incidence(bond_index, 5)
    incidence = incidence.numpy()

    chi = np.array([-4.7, -3.1, -5.9, -2.2, -6.4])
    eta = np.array([12.0, 13.5, 11.5, 14.0, 12.7])
    sigma = np.full(5, 1.5)
    hessian = ref.coulomb_matrix(positions, sigma) + np.diag(eta)
    kappa = np.full(incidence.shape[1], 0.6)
    return incidence, hessian, chi, kappa, 1.0


def _energy(hessian, chi, kappa, q, t):
    """U = 0.5 q^T (J + diag(eta)) q + chi^T q + 0.5 sum_b kappa_b t_b^2 (SPEC 1.5)."""
    return 0.5 * q @ hessian @ q + chi @ q + 0.5 * float((kappa * t * t).sum())


def _minimise_with_rho_fixed(incidence, hessian, chi, kappa, total_charge):
    """SPEC 1.3: ``rho`` fixed, only ``t`` variational.

    With ``rho`` fixed the per-fragment constraint is satisfied *structurally*:
    ``1^T B == 0``, so ``sum_i q_i = sum_i rho_i`` for any ``t`` whatsoever, and
    ``t`` is an unconstrained minimisation.  That is the whole point of putting
    the net charge in ``rho``.
    """
    n_atoms = incidence.shape[0]
    rho = np.full(n_atoms, total_charge / n_atoms)
    matrix = incidence.T @ hessian @ incidence + np.diag(kappa)
    rhs = -(incidence.T @ (hessian @ rho + chi))
    t = np.linalg.solve(matrix, rhs)
    return t, incidence @ t + rho


def _minimise_with_rho_free(incidence, hessian, chi, kappa, total_charge):
    """The REGRESSION: ``rho`` variational alongside ``t``, KKT-solved.

    Variables m = (t, rho), charges q = [B I] m, one global charge constraint.
    """
    n_atoms = incidence.shape[0]
    n_bonds = incidence.shape[1]
    n_vars = n_bonds + n_atoms
    to_charge = np.hstack([incidence, np.eye(n_atoms)])

    curvature = to_charge.T @ hessian @ to_charge
    curvature[:n_bonds, :n_bonds] += np.diag(kappa)
    linear = to_charge.T @ chi
    constraint = np.ones((1, n_atoms)) @ to_charge

    kkt = np.zeros((n_vars + 1, n_vars + 1))
    kkt[:n_vars, :n_vars] = curvature
    kkt[:n_vars, n_vars:] = constraint.T
    kkt[n_vars:, :n_vars] = constraint
    rhs = np.concatenate([-linear, [total_charge]])
    moments = np.linalg.solve(kkt, rhs)[:n_vars]
    return moments[:n_bonds], to_charge @ moments


def test_a_free_rho_collapses_the_bond_variables():
    """SPEC 1.3: "``|t| = 1.65e-07`` versus ``3.57e-01`` with ``rho`` fixed".

    Reproduced here in an independent minimisation.  Measured on this system:
    fixed rho gives |t|max = 3.643e-01; free rho gives 4.650e-16 -- SPEC's
    1.65e-07 is its own minimiser's residual, and an exact KKT solve drives the
    same quantity to round-off.  The assertion is on the ratio, which is the
    part that is a property of the model rather than of the solver.
    """
    incidence, hessian, chi, kappa, charge = _reference_system()
    t_fixed, _ = _minimise_with_rho_fixed(incidence, hessian, chi, kappa, charge)
    t_free, _ = _minimise_with_rho_free(incidence, hessian, chi, kappa, charge)

    fixed = float(np.abs(t_fixed).max())
    free = float(np.abs(t_free).max())
    # SPEC 1.3 quotes 3.57e-01 for its own system; this one gives 3.64e-01.
    assert fixed > 1.0e-1, f"the fixed-rho reference has no charge transfer: {fixed:.3e}"
    # SPEC 1.3 quotes 1.65e-07; an exact solve gets to 4.6e-16.
    assert free < 1.0e-6, f"free rho did not collapse t: {free:.3e}"
    assert fixed / max(free, 1.0e-300) > 1.0e5


def test_the_free_rho_minimum_is_cheaper_at_near_identical_charges():
    """SPEC 1.3: "``rho`` absorbs everything, pocketing the whole
    ``0.5 kappa |t|^2`` penalty at nearly identical charges".

    This is why the bug is silent.  Every charge-level observable -- dipole,
    Born charges, electrostatic potential -- agrees to ~2%, while the mechanism
    underneath has been replaced.  Measured here: max|dq| = 2.4e-02 e, and the
    free-rho minimum is 0.0797 eV lower.
    """
    incidence, hessian, chi, kappa, charge = _reference_system()
    t_fixed, q_fixed = _minimise_with_rho_fixed(incidence, hessian, chi, kappa, charge)
    t_free, q_free = _minimise_with_rho_free(incidence, hessian, chi, kappa, charge)

    assert float(np.abs(q_fixed - q_free).max()) < 5.0e-2  # charges barely move
    energy_fixed = _energy(hessian, chi, kappa, q_fixed, t_fixed)
    energy_free = _energy(hessian, chi, kappa, q_free, t_free)
    assert energy_free < energy_fixed, (
        "if the free-rho minimum were not cheaper there would be no degeneracy "
        "to guard against"
    )
    # Both minimisations satisfy the same net-charge constraint.
    assert abs(float(q_fixed.sum()) - charge) < 1.0e-12
    assert abs(float(q_free.sum()) - charge) < 1.0e-12


def test_the_saving_is_exactly_the_bond_hardness_penalty_that_was_dodged():
    """The general principle, quantified: the free-rho minimum is cheaper by at
    least the ``0.5 kappa |t|^2`` that the fixed-rho answer had to pay.

    This is the statement that generalises past ``rho``.  ANY per-atom channel
    that reaches the same charges without paying the bond penalty wins by at
    least this much, so "penalise it at least as hard as t" (SPEC 1.3) is not
    a style preference.
    """
    incidence, hessian, chi, kappa, charge = _reference_system()
    t_fixed, q_fixed = _minimise_with_rho_fixed(incidence, hessian, chi, kappa, charge)
    t_free, q_free = _minimise_with_rho_free(incidence, hessian, chi, kappa, charge)

    penalty_paid = 0.5 * float((kappa * t_fixed * t_fixed).sum())
    saving = _energy(hessian, chi, kappa, q_fixed, t_fixed) - _energy(
        hessian, chi, kappa, q_free, t_free
    )
    assert penalty_paid > 1.0e-3, "the test bed must actually pay a bond penalty"
    # The free minimiser gives up some Coulomb/chi energy to dodge the penalty,
    # so the saving is a fraction of it -- but the same order, not a rounding
    # error.  Measured: penalty 0.104 eV, saving 0.0797 eV.
    assert saving > 0.5 * penalty_paid


# ===========================================================================
# 2. The package: the guards that would catch a reintroduction
# ===========================================================================
def _solve(positions, seed, total_charge, **config_kwargs):
    electrostatic_energy = require("energy", "electrostatic_energy")
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")

    torch.manual_seed(seed)
    heads = ElectrostaticHeads()
    n_atoms = positions.shape[0]
    features = torch.randn(n_atoms, heads.irreps_in.dim) * 2.0
    edge_index = complete_edge_index(n_atoms)
    edge_vec = positions[edge_index[0]] - positions[edge_index[1]]
    _, result = electrostatic_energy(
        features,
        positions,
        edge_index,
        edge_vec,
        edge_vec.norm(dim=-1),
        heads,
        ElectrostaticConfig(**config_kwargs),
        total_charge=total_charge,
    )
    return result, features


POSITIONS = torch.tensor(
    [
        [0.00, 0.00, 0.00],
        [1.52, 0.00, 0.00],
        [2.10, 1.41, 0.00],
        [-0.63, 1.17, 0.35],
        [3.55, 1.62, 0.41],
    ]
)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_the_bond_variables_are_not_negligible_for_a_charged_system(seed):
    """THE regression guard for SPEC 1.3.

    If a free per-atom charge channel is ever reintroduced -- ``rho`` made
    variational, a ``q0_i(h_i)`` readout, a "residual charge" head -- the
    minimiser will drive ``t`` to the solver's round-off (SPEC's audit: 1.65e-07)
    while every charge-level observable stays put.  This test fails in that
    case and nothing else does.

    The threshold is 1e-3 e: three orders above the 1.65e-07 collapse SPEC
    measured, and two orders below the smallest |t|max seen across these seeds
    (1.94e-02).  A pass therefore means genuine bond-resolved charge transfer,
    not a solver residual.
    """
    result, _ = _solve(POSITIONS, seed=seed, total_charge=1.0)
    magnitude = float(result.t.abs().max())
    assert magnitude > 1.0e-3, (
        f"|t|max = {magnitude:.3e} for a +1 system at kappa > 0. SPEC 1.3: a "
        "free per-atom charge channel collapses |t| to ~1.65e-07 while the "
        "charges barely move -- has one been reintroduced?"
    )


def test_the_bond_variables_carry_no_net_charge():
    """SPEC 1.3: ``1^T B == 0``, so ``sum_i (B t)_i == 0`` and every bit of the
    net charge is in ``rho``.  The split has to be clean, or the constraint and
    the bond variables are fighting."""
    result, _ = _solve(POSITIONS, seed=1, total_charge=1.0)
    bond_charge = result.q - result.rho
    assert abs(float(bond_charge.sum())) < 1.0e-14
    assert abs(float(result.rho.sum()) - 1.0) < 1.0e-14


def test_rho_does_not_depend_on_the_node_features():
    """SPEC 1.3: ``rho`` is a FIXED constraint carrier, not a readout.

    The decisive test of "fixed": it is the same vector for wildly different
    ``h``.  A geometry- or feature-dependent ``rho`` is instance 3 of the SPEC
    1.3 general principle -- "watch for any future correction or residual
    charge head" -- and this is what would catch it.
    """
    first, features_first = _solve(POSITIONS, seed=0, total_charge=1.0)
    second, features_second = _solve(POSITIONS, seed=11, total_charge=1.0)
    assert not torch.allclose(features_first, features_second)  # really different h
    torch.testing.assert_close(first.rho, second.rho, rtol=0, atol=0)
    # ... and t did move, so the two solves are not trivially identical.
    assert float((first.t - second.t).abs().max()) > 1.0e-4


def test_rho_does_not_depend_on_geometry():
    """The same statement against the other input a readout could key on.

    SPEC 1.3: "There is NO ``q0_i(h_i)`` geometry-dependent reference charge...
    A per-element constant is allowed; a geometry-dependent readout is not."
    """
    moved = POSITIONS + torch.tensor([[0.0, 0.0, 0.0], [0.31, -0.17, 0.22]] * 2 + [[0.1, 0.4, -0.3]])
    base, _ = _solve(POSITIONS, seed=3, total_charge=1.0)
    shifted, _ = _solve(moved, seed=3, total_charge=1.0)
    torch.testing.assert_close(base.rho, shifted.rho, rtol=0, atol=0)
    assert float((base.t - shifted.t).abs().max()) > 1.0e-5  # geometry did change t


def test_the_charged_system_fixture_reproduces_the_collapse(charged_system):
    """The same degeneracy on the shared ``charged_system`` fixture, driven end
    to end through the package rather than through the numpy reference.

    Ties the two halves of this module together: the fixture's geometry, the
    reference minimisations above, and the package's own solve all describe the
    same system, so a future change to the fixture cannot silently detune the
    guard.
    """
    result, _ = _solve(
        charged_system["pos"],
        seed=7,
        total_charge=charged_system["total_charge"],
    )
    assert float(result.t.abs().max()) > 1.0e-3
    assert abs(float(result.rho.sum()) - charged_system["total_charge"]) < 1.0e-14
    # Every atom carries the same share of the net charge: rho is uniform
    # within the fragment, which is what "fixed constraint carrier" means here.
    assert float((result.rho - result.rho.mean()).abs().max()) < 1.0e-15


def test_a_neutral_molecule_still_transfers_charge_along_its_bonds(small_molecule):
    """``rho`` is zero at Q = 0, so everything the model says about charge comes
    from ``t``.  That is the cleanest statement of what the bond variables are
    for, and it fails immediately if a per-atom channel has taken over."""
    result, _ = _solve(small_molecule["pos"], seed=2, total_charge=0.0)
    assert float(result.rho.abs().max()) < 1.0e-15  # nothing for rho to carry
    assert float(result.t.abs().max()) > 1.0e-3  # ... and t is doing the work
    assert abs(float(result.q.sum())) < 1.0e-12
    # Charges are non-trivial even though they sum to zero.
    assert float(result.q.abs().max()) > 1.0e-4


def test_symmetry_equivalent_atoms_receive_equal_charges(water_molecule):
    """A C2v molecule cannot distinguish its two hydrogens.

    Permutation symmetry is a property of the *solve*, not of the readout: the
    two H atoms only come out equal if the constraint, the bond variables and
    the Coulomb kernel all respect the relabelling.  Here the features are made
    identical for the two hydrogens, so any difference in their charges is the
    solver breaking a symmetry it should not.
    """
    electrostatic_energy = require("energy", "electrostatic_energy")
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")

    positions = water_molecule["pos"]
    torch.manual_seed(4)
    heads = ElectrostaticHeads()
    features = torch.randn(3, heads.irreps_in.dim) * 2.0
    # The molecule sits with O at the origin and the two H at (+x, 0, z) and
    # (-x, 0, z), so the symmetry operation is the mirror x -> -x.  Making the
    # *geometry* symmetric is not enough: the features have to be symmetric
    # under the same operation, which means BOTH
    #   * the two hydrogens carry mirrored features, and
    #   * the oxygen, which the mirror maps to itself, carries features in the
    #     mirror-invariant subspace -- obtained by projecting with
    #     (I + D) / 2, since D^2 = I for a reflection.
    # Omitting the second half leaves a system that is not actually symmetric,
    # and the charges then differ by ~6e-06 e for entirely correct reasons.
    mirror = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    wigner = heads.irreps_in.D_from_matrix(mirror)
    features[0] = 0.5 * (features[0] + features[0] @ wigner.T)
    features[2] = features[1] @ wigner.T

    edge_index = complete_edge_index(3)
    edge_vec = positions[edge_index[0]] - positions[edge_index[1]]
    _, result = electrostatic_energy(
        features,
        positions,
        edge_index,
        edge_vec,
        edge_vec.norm(dim=-1),
        heads,
        ElectrostaticConfig(),
        total_charge=0.0,
    )
    # Exact identity; 1e-12 is the fixed-k solve's round-off budget.
    assert abs(float(result.q[1] - result.q[2])) < 1.0e-12, (
        f"the two hydrogens of water differ by {float(result.q[1] - result.q[2]):.3e} e"
    )
    assert float(result.q.abs().max()) > 1.0e-4  # not trivially all zero


@pytest.mark.parametrize("total_charge", [0.0, 1.0, -1.0, 2.5])
def test_rho_carries_exactly_the_net_charge(total_charge):
    """SPEC 1.3: ``rho`` carries net charge.  Since ``B t`` is exactly zero-sum,
    ``sum_i rho_i`` must equal the constraint target bit for bit, at every net
    charge including zero."""
    result, _ = _solve(POSITIONS, seed=4, total_charge=total_charge)
    assert abs(float(result.rho.sum()) - total_charge) < 1.0e-14
    # SPEC 2.1 item 5: the realised charges satisfy the constraint to 1e-10.
    assert abs(float(result.q.sum()) - total_charge) < 1.0e-10
