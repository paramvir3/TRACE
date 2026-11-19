"""Observables -- SPEC sections 0, 4 and 6.

    P     = sum_i (q_i r_i + mu_i)
    alpha = D^T K^-1 D
    Z*_i  = dP / dr_i ,     sum_i Z*_i = Q_total  (NOT zero)

SPEC section 6 requires Maxwell reciprocity ``alpha == alpha^T`` to 1e-10 and
insists it be measured on a *close-contact, polarizability-contrasted* system,
because a real Thole bug showed 0.005% asymmetry on a sparse geometry and 14.9%
at 1.2 A contact.  Both halves of that statement get a test here: the package is
checked at contact, and the claim about sparse test beds is demonstrated against
an independently coded broken relay.
"""

import math

import numpy as np
import pytest
import torch

import elec_reference as ref
from conftest import complete_edge_index, require


K_E = 14.399645  # SPEC 0


def _state(positions, seed=0, total_charge=0.0, feature_scale=2.0, **config_kwargs):
    """Solve a small open-boundary system and return (state, heads, h)."""
    electrostatic_state = require("energy", "electrostatic_state")
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")

    torch.manual_seed(seed)
    heads = ElectrostaticHeads()
    n_atoms = positions.shape[0]
    h = torch.randn(n_atoms, heads.irreps_in.dim) * feature_scale
    edge_index = complete_edge_index(n_atoms)
    edge_vec = positions[edge_index[0]] - positions[edge_index[1]]
    edge_len = edge_vec.norm(dim=-1)
    config = ElectrostaticConfig(**config_kwargs)
    state = electrostatic_state(
        h,
        positions,
        edge_index,
        edge_vec,
        edge_len,
        heads,
        config,
        total_charge=total_charge,
    )
    return state, heads, h


# ---------------------------------------------------------------------------
# SPEC 4: P = sum_i (q_i r_i + mu_i)
# ---------------------------------------------------------------------------
def test_polarization_is_the_sum_of_charge_and_dipole_terms():
    polarization = require("observables", "polarization")
    q = torch.tensor([0.5, -0.2, -0.3])
    pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    mu = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]])
    expected = torch.tensor([0.5 * 1.0 + 0.1, -0.2 * 2.0 + 0.2, -0.3 * 3.0 + 0.3])
    torch.testing.assert_close(polarization(q, pos, mu), expected, rtol=0, atol=1e-15)


def test_polarization_of_a_neutral_system_is_origin_independent():
    """Textbook: only for Q_total = 0 is the dipole moment origin-free."""
    polarization = require("observables", "polarization")
    q = torch.tensor([0.5, -0.2, -0.3])  # sums to zero
    pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    shift = torch.tensor([3.1, -0.7, 2.2])
    torch.testing.assert_close(
        polarization(q, pos), polarization(q, pos + shift), rtol=0, atol=1e-14
    )


# ---------------------------------------------------------------------------
# SPEC 6: Maxwell reciprocity
# ---------------------------------------------------------------------------
def _contact_geometry():
    """Four atoms with a 1.19 A closest contact and no symmetry to hide behind."""
    return torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [1.19, 0.00, 0.00],
            [0.37, 1.00, 0.53],
            [1.55, 0.88, -0.61],
        ]
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_polarizability_is_symmetric_at_close_contact(seed):
    """SPEC 6: "Maxwell reciprocity ``alpha == alpha^T`` | 1e-10", measured on a
    close-contact, polarizability-contrasted system."""
    polarizability = require("observables", "polarizability")
    positions = _contact_geometry()
    state, _, _ = _state(positions, seed=seed)

    # The test bed has to have contrast, or it cannot expose a damping bug keyed
    # on the wrong atom of a pair.  SPEC 6 makes this a precondition, so assert it.
    alpha_volume = state.assembly.alpha_volume
    contrast = float(alpha_volume.max() / alpha_volume.min())
    assert contrast > 1.2, f"insufficient polarizability contrast ({contrast:.2f}x)"

    separations = torch.cdist(positions, positions)
    closest = float(separations[separations > 0].min())
    assert closest < 1.25  # SPEC 6 names 1.2 A as the distance the bug shows up at

    alpha = polarizability(state.solve_result, state.assembly)
    asymmetry = float((alpha - alpha.T).abs().max() / alpha.abs().max())
    assert asymmetry < 1.0e-10, (
        f"alpha is asymmetric by {asymmetry:.3e} relative at {closest:.2f} A "
        f"contact with {contrast:.1f}x polarizability contrast"
    )


def test_a_sparse_geometry_is_not_a_valid_reciprocity_test_bed(
    contrasted_close_contact, sparse_triple
):
    """SPEC 6: "On a sparse test system a known Thole bug shows only 0.005%
    asymmetry and passes silently; at 1.2 A contact the same bug gives 14.9%."

    Demonstrated against an independently coded relay with a deliberately
    injected, realistic Thole bug: the damping length uses only the source
    polarizability instead of ``(alpha_i alpha_j)^(1/6)``.  The absolute
    percentages depend on the bug and on the parameters -- the audit's were
    0.005% and 14.9% -- but the *amplification* is the point, and it reproduces:
    measured here 2.8e-05 % sparse against 0.70 % at contact, a factor of ~2.5e4.

    Without this test, a suite that checked reciprocity only on a sparse system
    would look green while the relay was wrong.
    """
    measured = {}
    for label, geometry in (
        ("sparse", sparse_triple),
        ("contact", contrasted_close_contact),
    ):
        relay = ref.thole_relay_matrix(
            geometry["pos"].numpy(),
            geometry["alpha_volume"].numpy(),
            a=2.089,
            asymmetric=True,
        )
        measured[label] = ref.relative_asymmetry(ref.relay_polarizability(relay))

    # A sparse test bed hides it: 2.8e-7 relative would pass any threshold a
    # reviewer would plausibly write (1%, 0.1%, even 0.001%).
    assert measured["sparse"] < 1.0e-4
    # At contact the same bug is unmissable.
    assert measured["contact"] > 1.0e-3
    assert measured["contact"] / measured["sparse"] > 100.0, (
        f"sparse {measured['sparse']:.3e} vs contact {measured['contact']:.3e}"
    )


def test_a_symmetric_reference_relay_is_symmetric_at_both_separations(
    contrasted_close_contact, sparse_triple
):
    """Control for the test above: the amplification comes from the bug, not
    from the close-contact geometry itself.

    The same two geometries and the same polarizabilities, with the symmetric
    damping length ``(alpha_i alpha_j)^(1/6)``: both stay symmetric to 1e-14.
    """
    for geometry in (sparse_triple, contrasted_close_contact):
        relay = ref.thole_relay_matrix(
            geometry["pos"].numpy(),
            geometry["alpha_volume"].numpy(),
            a=2.089,
            asymmetric=False,
        )
        assert ref.relative_asymmetry(ref.relay_polarizability(relay)) < 1.0e-14


def test_polarizability_is_positive_definite_for_a_stable_system():
    """SPEC 2.1 item 4: a negative alpha is the signature of a saddle."""
    polarizability = require("observables", "polarizability")
    state, _, _ = _state(_contact_geometry(), seed=7)
    alpha = polarizability(state.solve_result, state.assembly)
    eigenvalues = torch.linalg.eigvalsh(0.5 * (alpha + alpha.T))
    assert float(eigenvalues.min()) > 0.0, f"alpha eigenvalues {eigenvalues.tolist()}"


def test_polarizability_volume_conversion_is_the_k_e_factor():
    """SPEC 0: ``alpha_volume [A^3] = K_E * alpha_response [e^2 A^2 / eV]``.

    Reporting the raw response as a volume is a 14x error, not a subtle one.
    """
    polarizability = require("observables", "polarizability")
    state, _, _ = _state(_contact_geometry(), seed=2)
    response = polarizability(state.solve_result, state.assembly)
    volume = polarizability(state.solve_result, state.assembly, as_volume=True)
    torch.testing.assert_close(volume, K_E * response, rtol=1e-13, atol=0.0)


def test_charge_transfer_and_induced_contributions_sum_to_alpha():
    """SPEC 4: alpha "contains BOTH charge-transfer and induced parts"."""
    polarizability, contributions = require(
        "observables", "polarizability", "polarizability_contributions"
    )
    state, _, _ = _state(_contact_geometry(), seed=1)
    alpha = polarizability(state.solve_result, state.assembly)
    charge_part, dipole_part = contributions(state.solve_result, state.assembly)
    # Default route (SPEC 17): each channel's second derivative comes from its
    # own adjoint solve at solve_rtol = 1e-10, so the sum is exact to that
    # tolerance (measured 2.2e-10 relative), not to round-off.
    torch.testing.assert_close(charge_part + dipole_part, alpha, rtol=1e-8, atol=1e-10)
    # The dense minimiser route splits ONE factorisation: exact to round-off.
    alpha_min = polarizability(state.solve_result, state.assembly, route="minimiser")
    charge_min, dipole_min = contributions(state.solve_result, state.assembly, route="minimiser")
    torch.testing.assert_close(charge_min + dipole_min, alpha_min, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(alpha, alpha_min, rtol=1e-8, atol=1e-10)
    # Neither half is separately zero: both channels are live at contact.
    assert float(charge_part.abs().max()) > 0.0
    assert float(dipole_part.abs().max()) > 0.0


# ---------------------------------------------------------------------------
# SPEC 4 / 6: Born charges of a charged system sum to Q_total
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("total_charge", [1.0, -1.0, 2.0, 0.0])
def test_born_charges_sum_to_the_total_charge_not_to_zero(total_charge):
    """SPEC 4: "For a charged system ``sum_i Z*_i = Q_tot``, NOT zero".

    Derivation: a rigid translation r_i -> r_i + u leaves every q_i and mu_i
    unchanged (they depend on relative geometry only), so P -> P + Q_tot u and
    hence sum_i dP_a/dr_{i,b} = Q_tot delta_ab.
    """
    polarization, born_charges = require(
        "observables", "polarization", "born_charges"
    )
    positions = _contact_geometry().clone().requires_grad_(True)
    state, _, _ = _state(positions, seed=3, total_charge=total_charge)
    result = state.solve_result
    p_vector = polarization(result.q, positions, result.mu)
    born = born_charges(p_vector, positions)
    assert born.shape == (positions.shape[0], 3, 3)

    total = born.sum(dim=0)
    expected = torch.eye(3) * total_charge
    error = float((total - expected).abs().max())
    # Exact identity; 1e-9 is the autograd/solve round-off budget, not physics.
    assert error < 1.0e-9, (
        f"sum_i Z*_i = {total.diagonal().tolist()} for Q_total = {total_charge}"
    )


def test_acoustic_sum_rule_residual_measures_against_q_total():
    """SPEC 4: the helper checks the identity, it must never impose it."""
    polarization, born_charges, residual = require(
        "observables", "polarization", "born_charges", "acoustic_sum_rule_residual"
    )
    positions = _contact_geometry().clone().requires_grad_(True)
    state, _, _ = _state(positions, seed=4, total_charge=1.0)
    p_vector = polarization(state.solve_result.q, positions, state.solve_result.mu)
    born = born_charges(p_vector, positions)
    assert float(residual(born, 1.0).abs().max()) < 1.0e-9
    # Against the wrong reference (the "force-symmetrise to zero" mistake) the
    # residual is O(1), i.e. the helper really is measuring something.
    assert float(residual(born, 0.0).abs().max()) > 0.5


def test_hyperpolarizability_raises_rather_than_returning_zero():
    """SPEC 4: "Hyperpolarizability beta is out of scope.  Deferred, not
    omitted by oversight"."""
    hyperpolarizability = require("observables", "hyperpolarizability")
    with pytest.raises(NotImplementedError):
        hyperpolarizability()


# ---------------------------------------------------------------------------
# O(3) behaviour of the polarization vector
# ---------------------------------------------------------------------------
def test_polarization_rotates_as_a_vector_and_alpha_as_a_rank_two_tensor():
    """SPEC 3/4: P is 1o.  Under R, P -> R P and alpha -> R alpha R^T.

    The node features are rotated with e3nn's own Wigner D matrix for the input
    irreps, which is what makes this a test of the readout-plus-solve chain
    rather than of an accidental isotropy.
    """
    o3 = pytest.importorskip("e3nn.o3")
    polarization, polarizability = require(
        "observables", "polarization", "polarizability"
    )
    electrostatic_state = require("energy", "electrostatic_state")
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")

    positions = _contact_geometry()
    torch.manual_seed(8)
    heads = ElectrostaticHeads()
    h = torch.randn(positions.shape[0], heads.irreps_in.dim) * 2.0
    config = ElectrostaticConfig()
    edge_index = complete_edge_index(positions.shape[0])

    def run(pos, features):
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        return electrostatic_state(
            features,
            pos,
            edge_index,
            edge_vec,
            edge_vec.norm(dim=-1),
            heads,
            config,
            total_charge=1.0,
        )

    base = run(positions, h)
    rotation = o3.rand_matrix()
    wigner = heads.irreps_in.D_from_matrix(rotation)
    rotated = run(positions @ rotation.T, h @ wigner.T)

    # Energy is a scalar.
    torch.testing.assert_close(rotated.energy, base.energy, rtol=1e-10, atol=1e-10)
    # Charges are scalars.
    torch.testing.assert_close(
        rotated.solve_result.q, base.solve_result.q, rtol=1e-9, atol=1e-10
    )

    p_base = polarization(base.solve_result.q, positions, base.solve_result.mu)
    p_rotated = polarization(
        rotated.solve_result.q, positions @ rotation.T, rotated.solve_result.mu
    )
    torch.testing.assert_close(p_rotated, rotation @ p_base, rtol=1e-9, atol=1e-9)

    alpha_base = polarizability(base.solve_result, base.assembly)
    alpha_rotated = polarizability(rotated.solve_result, rotated.assembly)
    torch.testing.assert_close(
        alpha_rotated, rotation @ alpha_base @ rotation.T, rtol=1e-8, atol=1e-9
    )


def test_polarization_is_inverted_by_a_parity_operation():
    """P is odd (1o): under r -> -r with odd features, P -> -P."""
    o3 = pytest.importorskip("e3nn.o3")
    polarization = require("observables", "polarization")
    electrostatic_state = require("energy", "electrostatic_state")
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")

    positions = _contact_geometry()
    torch.manual_seed(12)
    heads = ElectrostaticHeads()
    h = torch.randn(positions.shape[0], heads.irreps_in.dim) * 2.0
    config = ElectrostaticConfig()
    edge_index = complete_edge_index(positions.shape[0])

    def run(pos, features):
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        return electrostatic_state(
            features, pos, edge_index, edge_vec, edge_vec.norm(dim=-1), heads, config
        )

    inversion = -torch.eye(3)
    parity = heads.irreps_in.D_from_matrix(inversion)
    base = run(positions, h)
    flipped = run(-positions, h @ parity.T)

    torch.testing.assert_close(flipped.energy, base.energy, rtol=1e-10, atol=1e-10)
    p_base = polarization(base.solve_result.q, positions, base.solve_result.mu)
    p_flipped = polarization(flipped.solve_result.q, -positions, flipped.solve_result.mu)
    torch.testing.assert_close(p_flipped, -p_base, rtol=1e-9, atol=1e-9)
