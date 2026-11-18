"""Periodic electrostatics -- SPEC section 6 (NaCl Madelung constant, alpha independence).

The NaCl test geometry is derived here, not taken from the implementation:
rock-salt is the cubic cell with Na at the FCC sites and Cl at the octahedral
sites, four formula units per conventional cell, nearest-neighbour distance
r0 = a/2.  The Madelung energy of one ion pair is -M K_E q^2 / r0, so the total
energy of the cell is -4 M K_E / r0 and

    M = -E_cell r0 / (4 K_E).

``elec_reference.ewald_reference`` is an independently written textbook Ewald
sum; the first test below uses it to confirm that this geometry really does
carry M = 1.747564595 before the package is asked to reproduce the same number.
"""

import math

import numpy as np
import pytest
import torch

import elec_reference as ref
from conftest import full_neighbour_list, require


MADELUNG_NACL = 1.747564595  # SPEC section 6, required value


@pytest.mark.slow
def test_reference_ewald_confirms_the_nacl_test_geometry(nacl_crystal):
    """Test-bed validation: an independent Ewald sum on the derived geometry.

    This does not touch ``trace_electrostatics`` at all.  If it ever fails, the
    geometry in ``conftest.nacl_crystal`` is wrong and every Madelung assertion
    below it is meaningless.
    """
    energy = ref.ewald_reference(
        nacl_crystal["pos"].numpy(),
        nacl_crystal["q"].numpy(),
        nacl_crystal["cell"].numpy(),
        alpha=0.35,
        n_real=5,
        k_max=9,
    )
    madelung = ref.madelung_from_energy(
        energy, nacl_crystal["nearest_neighbour"], nacl_crystal["n_pairs"]
    )
    # SPEC section 6 quotes 1.747564595 (9 significant figures); the converged
    # value is 1.7475645946..., so the rounding gap alone is ~4e-10.
    assert abs(madelung - MADELUNG_NACL) < 5.0e-10


def test_package_ewald_reproduces_the_madelung_constant(nacl_crystal):
    """SPEC section 6: "Ewald reproduces NaCl Madelung constant | 1.747564595"."""
    ewald_energy, EwaldParams = require("ewald", "ewald_energy", "EwaldParams")

    r_cut = 12.0
    edge_index, edge_shift = full_neighbour_list(
        nacl_crystal["pos"], nacl_crystal["cell"], r_cut
    )
    # alpha r_cut = 5.4 gives erfc(5.4) ~ 8e-14 in the real-space tail, and
    # k_max = 10 on a 5.64 A cube reaches |k| = 11.1 1/A, where
    # exp(-k^2/4 alpha^2) ~ e^-153.  Both truncations are far below the 5e-10
    # accuracy the SPEC value demands.
    params = EwaldParams(alpha=0.45, k_max=10, r_cut=r_cut)
    energy = ewald_energy(
        nacl_crystal["q"],
        nacl_crystal["pos"],
        nacl_crystal["cell"],
        edge_index,
        edge_shift,
        params,
    )
    madelung = ref.madelung_from_energy(
        energy, nacl_crystal["nearest_neighbour"], nacl_crystal["n_pairs"]
    )
    assert abs(madelung - MADELUNG_NACL) < 5.0e-10, (
        f"Madelung constant {madelung:.12f}, SPEC requires {MADELUNG_NACL} "
        f"(>= 8 digits); error {madelung - MADELUNG_NACL:.3e}"
    )


def test_madelung_constant_is_independent_of_the_lattice_constant(nacl_crystal):
    """M is dimensionless: scaling the cell must scale E as 1/a and leave M fixed."""
    ewald_energy, EwaldParams = require("ewald", "ewald_energy", "EwaldParams")
    for a in (4.0, 5.64, 7.5):
        cell = torch.eye(3) * a
        pos = nacl_crystal["pos"] / nacl_crystal["lattice_constant"] * a
        r_cut = 2.2 * a
        edge_index, edge_shift = full_neighbour_list(pos, cell, r_cut)
        params = EwaldParams(alpha=5.4 / r_cut, k_max=10, r_cut=r_cut)
        energy = ewald_energy(nacl_crystal["q"], pos, cell, edge_index, edge_shift, params)
        madelung = ref.madelung_from_energy(energy, a / 2.0, 4)
        assert abs(madelung - MADELUNG_NACL) < 5.0e-9  # looser: k_max fixed, cell varies


def test_ewald_total_energy_is_independent_of_alpha(nacl_crystal):
    """SPEC 6 / ewald docstring: alpha splits the work, it does not change U.

    The four contributions individually move by many eV across this range; the
    total must not.
    """
    ewald_terms, EwaldParams = require("ewald", "ewald_terms", "EwaldParams")
    r_cut = 12.0
    edge_index, edge_shift = full_neighbour_list(
        nacl_crystal["pos"], nacl_crystal["cell"], r_cut
    )

    totals, reals, selfs = [], [], []
    for alpha in (0.40, 0.45, 0.50, 0.55, 0.60):
        params = EwaldParams(alpha=alpha, k_max=12, r_cut=r_cut)
        terms = ewald_terms(
            nacl_crystal["q"],
            nacl_crystal["pos"],
            nacl_crystal["cell"],
            edge_index,
            edge_shift,
            params,
        )
        totals.append(float(terms.total))
        reals.append(float(terms.real))
        selfs.append(float(terms.self_energy))

    spread = max(totals) - min(totals)
    # Tolerance from the truncation budget above (both tails < 1e-12 relative on
    # a ~36 eV energy), not from the SPEC; SPEC only states that the total is
    # alpha-independent.
    assert spread < 1.0e-9, f"total moved by {spread:.3e} eV across alpha"
    # Guard against a vacuous pass: the split must actually have moved.
    assert max(selfs) - min(selfs) > 1.0
    assert max(reals) - min(reals) > 1.0


def test_madelung_constant_from_the_non_orthogonal_primitive_cell(nacl_primitive):
    """SPEC section 6, on a cell with no right angles.

    The conventional cubic cell is a weak test of a reciprocal-space sum: its
    reciprocal cell is diagonal, so transposing the cell matrix, or inverting it
    on the wrong side, changes nothing.  The FCC primitive cell of the *same*
    crystal has 60-degree angles, one formula unit and a volume of a^3/4, and it
    must give the same dimensionless Madelung constant.

    Measured: M = 1.747564594633, i.e. 3.7e-10 from the value SPEC quotes --
    which is itself the rounding of 1.7475645946... to nine figures.
    """
    ewald_energy, EwaldParams = require("ewald", "ewald_energy", "EwaldParams")
    r_cut = 14.0
    edge_index, edge_shift = full_neighbour_list(
        nacl_primitive["pos"], nacl_primitive["cell"], r_cut
    )
    # alpha r_cut = 5.6 leaves erfc(5.6) ~ 8e-15 in the real tail; k_max = 16 on
    # a cell of width ~4 A reaches exp(-k^2/4 alpha^2) ~ e^-250.
    params = EwaldParams(alpha=5.6 / r_cut, k_max=16, r_cut=r_cut)
    energy = ewald_energy(
        nacl_primitive["q"],
        nacl_primitive["pos"],
        nacl_primitive["cell"],
        edge_index,
        edge_shift,
        params,
    )
    madelung = ref.madelung_from_energy(
        energy, nacl_primitive["nearest_neighbour"], nacl_primitive["n_pairs"]
    )
    assert abs(madelung - MADELUNG_NACL) < 5.0e-10, (
        f"primitive (non-orthogonal) cell gives M = {madelung:.12f}, "
        f"SPEC requires {MADELUNG_NACL}"
    )


def test_the_two_nacl_cells_agree_on_the_energy_per_formula_unit(
    nacl_crystal, nacl_primitive
):
    """The conventional and primitive cells are the same crystal.

    Four formula units versus one, orthogonal versus rhombohedral, eight atoms
    versus two -- the energy per formula unit is the same number or one of the
    two cells is not the crystal it claims to be.
    """
    ewald_energy, EwaldParams = require("ewald", "ewald_energy", "EwaldParams")
    r_cut = 14.0
    energies = []
    for crystal in (nacl_crystal, nacl_primitive):
        edge_index, edge_shift = full_neighbour_list(
            crystal["pos"], crystal["cell"], r_cut
        )
        params = EwaldParams(alpha=5.6 / r_cut, k_max=16, r_cut=r_cut)
        energy = float(
            ewald_energy(
                crystal["q"],
                crystal["pos"],
                crystal["cell"],
                edge_index,
                edge_shift,
                params,
            )
        )
        energies.append(energy / crystal["n_pairs"])
    # Joint truncation budget of the two cells, not a physical tolerance.
    assert abs(energies[0] - energies[1]) < 1.0e-8, (
        f"cubic {energies[0]:.12f} eV/formula unit vs primitive "
        f"{energies[1]:.12f}"
    )


def test_ewald_total_energy_is_independent_of_alpha_for_a_CHARGED_cell():
    """The jellium guard.

    For a neutral cell alpha-independence holds term by term and says nothing
    about the k = 0 background.  For a cell with ``sum_i q_i != 0`` the
    conditionally convergent sum only closes against a uniform neutralising
    background, and the background term carries an explicit ``1/alpha^2``.  So
    dropping it -- the single most common Ewald omission, and invisible on every
    neutral test -- leaves an energy that still looks finite and still looks
    smooth, but moves with the splitting parameter.

    Measured on this +1.5 e cell over alpha in [0.40, 0.70]: the total moves by
    1.6e-11 eV WITH the jellium term and by 9.9e-01 eV without it.
    """
    ewald_terms, EwaldParams = require("ewald", "ewald_terms", "EwaldParams")
    cell = torch.eye(3) * 6.0
    pos = torch.tensor(
        [[0.4, 0.5, 0.6], [3.1, 0.8, 1.9], [1.4, 3.6, 4.9], [5.0, 4.1, 2.2]]
    )
    q = torch.tensor([0.9, -0.3, 0.5, 0.4])
    assert abs(float(q.sum()) - 1.5) < 1.0e-12  # the cell really is charged

    r_cut = 12.0
    edge_index, edge_shift = full_neighbour_list(pos, cell, r_cut)
    with_jellium, without_jellium = [], []
    for alpha in (0.40, 0.45, 0.50, 0.55, 0.60, 0.70):
        params = EwaldParams(alpha=alpha, k_max=16, r_cut=r_cut)
        terms = ewald_terms(q, pos, cell, edge_index, edge_shift, params)
        base = float(terms.real + terms.recip + terms.self_energy)
        without_jellium.append(base)
        with_jellium.append(base + float(terms.jellium))
        torch.testing.assert_close(
            terms.total,
            terms.real + terms.recip + terms.self_energy + terms.jellium,
            rtol=1e-14,
            atol=0.0,
        )

    spread = max(with_jellium) - min(with_jellium)
    # Truncation budget: erfc(alpha r_cut) < erfc(4.8) ~ 8e-12 and the k-space
    # tail is far below that, on an energy of ~5 eV.
    assert spread < 1.0e-9, f"charged-cell total moved by {spread:.3e} eV"
    # Guard against a vacuous pass: without the background it moves by ~1 eV.
    naive_spread = max(without_jellium) - min(without_jellium)
    assert naive_spread > 1.0e-1, (
        "dropping the jellium term did not change the alpha dependence, so "
        f"this test does not guard it (spread {naive_spread:.3e} eV)"
    )


def test_the_jellium_term_is_the_only_alpha_dependent_piece_left_over():
    """The same statement, checked analytically rather than by spread.

    U_bg = -K_E pi Q^2 / (2 V alpha^2).  If the omitted piece really is the
    background, then adding that closed form to the naive (real + recip + self)
    sum must restore alpha independence -- which is a stronger claim than
    "the package's own jellium term happens to work".
    """
    ewald_terms, EwaldParams = require("ewald", "ewald_terms", "EwaldParams")
    cell = torch.eye(3) * 6.0
    volume = float(torch.det(cell))
    pos = torch.tensor(
        [[0.4, 0.5, 0.6], [3.1, 0.8, 1.9], [1.4, 3.6, 4.9], [5.0, 4.1, 2.2]]
    )
    q = torch.tensor([0.9, -0.3, 0.5, 0.4])
    total_charge = float(q.sum())
    r_cut = 12.0
    edge_index, edge_shift = full_neighbour_list(pos, cell, r_cut)

    repaired = []
    for alpha in (0.40, 0.50, 0.60, 0.70):
        params = EwaldParams(alpha=alpha, k_max=16, r_cut=r_cut)
        terms = ewald_terms(q, pos, cell, edge_index, edge_shift, params)
        background = (
            -ref.K_E * math.pi * total_charge**2 / (2.0 * volume * alpha**2)
        )
        # The package's own term must equal the closed form, to round-off.
        assert float(terms.jellium) == pytest.approx(background, rel=1.0e-12)
        repaired.append(
            float(terms.real + terms.recip + terms.self_energy) + background
        )
    assert max(repaired) - min(repaired) < 1.0e-9


def test_package_ewald_matches_an_independent_ewald_on_a_random_cell():
    """Cross-check on a geometry with no symmetry to hide behind."""
    ewald_energy, EwaldParams = require("ewald", "ewald_energy", "EwaldParams")
    torch.manual_seed(11)
    cell = torch.tensor([[6.0, 0.0, 0.0], [0.8, 5.5, 0.0], [-0.4, 0.6, 6.7]])
    pos = torch.rand(6, 3) @ cell
    q = torch.tensor([0.7, -0.3, 0.25, -0.9, 0.55, -0.3])
    q = q - q.mean()  # neutral cell: no jellium term to disagree about

    r_cut = 14.0
    edge_index, edge_shift = full_neighbour_list(pos, cell, r_cut)
    params = EwaldParams(alpha=5.4 / r_cut, k_max=14, r_cut=r_cut)
    energy = float(ewald_energy(q, pos, cell, edge_index, edge_shift, params))

    reference = ref.ewald_reference(
        pos.numpy(), q.numpy(), cell.numpy(), alpha=0.42, n_real=4, k_max=10
    )
    # Two independent truncations of the same conditionally convergent sum;
    # 1e-8 eV is the joint truncation budget, not a physical constant.
    assert abs(energy - reference) < 1.0e-8, (
        f"package {energy:.12f} eV vs independent reference {reference:.12f} eV"
    )


def test_ewald_energy_is_invariant_under_rigid_translation(nacl_crystal):
    """A periodic energy cannot depend on the origin."""
    ewald_energy, EwaldParams = require("ewald", "ewald_energy", "EwaldParams")
    r_cut = 12.0
    shift_vector = torch.tensor([0.37, -1.21, 2.04])
    params = EwaldParams(alpha=0.45, k_max=10, r_cut=r_cut)

    energies = []
    for offset in (torch.zeros(3), shift_vector):
        pos = nacl_crystal["pos"] + offset
        edge_index, edge_shift = full_neighbour_list(pos, nacl_crystal["cell"], r_cut)
        energies.append(
            float(
                ewald_energy(
                    nacl_crystal["q"],
                    pos,
                    nacl_crystal["cell"],
                    edge_index,
                    edge_shift,
                    params,
                )
            )
        )
    assert abs(energies[0] - energies[1]) < 1.0e-10  # exact invariance; 1e-10 is round-off


def test_ewald_position_gradient_matches_finite_difference():
    """SPEC section 6: "analytic gradient vs finite difference | < 1e-10"."""
    ewald_energy, EwaldParams = require("ewald", "ewald_energy", "EwaldParams")
    cell = torch.eye(3) * 7.0
    pos = torch.tensor(
        [[0.4, 0.6, 0.3], [3.1, 0.9, 2.2], [1.5, 3.4, 5.1], [5.2, 4.4, 1.1]]
    )
    q = torch.tensor([0.8, -0.2, 0.4, -1.0])
    r_cut = 12.0
    edge_index, edge_shift = full_neighbour_list(pos, cell, r_cut)
    params = EwaldParams(alpha=5.4 / r_cut, k_max=10, r_cut=r_cut)

    def energy_of(positions):
        return ewald_energy(q, positions, cell, edge_index, edge_shift, params)

    x = pos.clone().requires_grad_(True)
    (analytic,) = torch.autograd.grad(energy_of(x), x)
    numeric = ref.fd_gradient(lambda p: energy_of(p).detach(), pos, step=1.0e-4)
    error = float((analytic - numeric).abs().max())
    assert error < 1.0e-10, f"max |autograd - FD| = {error:.3e} eV/A"


def test_jellium_term_vanishes_for_a_neutral_cell_and_not_for_a_charged_one():
    """The k = 0 background exists only when sum_i q_i != 0."""
    ewald_jellium, EwaldParams = require("ewald", "ewald_jellium", "EwaldParams")
    params = EwaldParams(alpha=0.45, k_max=8, r_cut=12.0)
    volume = torch.tensor(512.0)
    assert float(ewald_jellium(torch.tensor(0.0), volume, params)) == 0.0
    charged = float(ewald_jellium(torch.tensor(1.0), volume, params))
    # U_bg = -K_E pi Q^2 / (2 V alpha^2)  -- derived from the k -> 0 limit of the
    # reciprocal sum against a uniform neutralising background.
    expected = -ref.K_E * math.pi / (2.0 * 512.0 * 0.45**2)
    assert abs(charged - expected) < 1.0e-12
