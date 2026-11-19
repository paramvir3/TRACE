"""EXTERNAL validation of the Makov-Payne finite-size correction.

A charged periodic cell does not have the energy of the isolated charged defect
it is supposed to represent.  ``ewald_jellium`` makes the lattice sum converge,
but what is left is the defect interacting with its own periodic images and with
the neutralising background, and that decays only as ``1/L``.  The Makov-Payne
correction (Makov and Payne, *Phys. Rev. B* **51**, 4014 (1995)) removes it in
closed form:

    E_corr = K_E q^2 xi / (2 epsilon V^(1/3))
             - 2 pi K_E q Q_r / (3 epsilon V)      + O(L^-5)

THE REFERENCE IS EXACT, AND IT IS EXTERNAL
------------------------------------------
For the test bed below -- ONE ``+1 e`` point charge, nothing else in the cell,
``epsilon = 1`` -- the isolated system is a single point charge alone in vacuum,
whose energy in this convention is **exactly zero**: there is no second charge
to interact with and this module sums no self energy of a point.  So the target
is not another number produced by the package.  It is 0, and every eV the
corrected sum reports is error.

That makes this test bed unusually strong.  ``xi`` is the whole correction (the
``Q_r`` term vanishes for a point charge), the isolated answer is zero, and the
uncorrected answer is the Madelung energy of a charge in jellium, which is
itself a tabulated constant.  Nothing here is a comparison of the package with
itself.

THE MEASURED TABLE, which this file locks in as a regression
------------------------------------------------------------
``MakovPayneParams(madelung=2.8373, dielectric=1.0)``, cubic cells,
``r_cut = min(6.0, L/2 - 0.1)``, ``accuracy = 1e-8``:

    L (A)    uncorrected E        corrected E
     8.0      -2.55350478          0.00000227
    10.0      -2.04280382          0.00000181
    12.0      -1.70233652          0.00000151
    16.0      -1.27675239          0.00000113
    20.0      -1.02140191          0.00000091

The uncorrected energy spans 1.5321029 eV over that range and the corrected one
spans 1.3611e-06 eV: a reduction of 1.126e+06.

AND THE 1e-06 IS NOT A RESIDUAL FINITE-SIZE ERROR
-------------------------------------------------
It is the five-digit rounding of ``xi``.  ``2.8373 - 2.837297479481 =
2.5205e-06``, and a wrong ``xi`` leaves exactly ``K_E dxi / (2 L)`` behind, which
is 2.268e-06 eV at L = 8 A -- the whole of the corrected column, to four digits,
at every L.  Feeding the package's own twelve-digit
``ewald.MADELUNG_SIMPLE_CUBIC`` instead takes the corrected energy to 7.0e-11 eV
at ``accuracy = 1e-8`` and to 3.5e-13 eV at ``accuracy = 1e-14``, i.e. to the
truncation floor of the Ewald sum itself.  The correction is EXACT for this
system; the 1e-06 measures the input, not the physics.  Both statements are
tested below rather than asserted in prose.

``xi`` itself is derived here from an independent Ewald sum written in this
file, sharing no line of code with the package, and cross-checked against the
textbook 2.8372974795 and against the ``(9/5)(4 pi/3)^(1/3)`` uniform-sphere
bound that no lattice may exceed.
"""

import itertools
import math

import numpy
import pytest
import torch

from conftest import full_neighbour_list, require


K_E = 14.399645  # SPEC 0

# The rounded constant the regression table above was measured with.
TABLE_MADELUNG = 2.8373

# Simple-cubic Madelung constant of a point charge in jellium, in the
# L = V^(1/3) convention.  Quoted from the solid-state literature (the value is
# usually written 2.8372974795; it is the "Madelung constant of the jellium
# lattice" and appears in every treatment of the electron-gas Madelung energy,
# e.g. Fraser et al., Phys. Rev. B 53, 1814 (1996), Table I).  It is DERIVED
# independently below as well, so nothing depends on this line being right.
TEXTBOOK_MADELUNG_SIMPLE_CUBIC = 2.8372974795

# The regression table.  (L, uncorrected eV, corrected eV) -- measured with the
# parameters named in the module docstring.
MEASURED_TABLE = (
    (8.0, -2.55350478, 0.00000227),
    (10.0, -2.04280382, 0.00000181),
    (12.0, -1.70233652, 0.00000151),
    (16.0, -1.27675239, 0.00000113),
    (20.0, -1.02140191, 0.00000091),
)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
def _neighbour_list(pos, cell, cutoff):
    """``conftest.full_neighbour_list`` with a shift tensor that is always [n, 3].

    A one-atom cell whose nearest image sits beyond ``cutoff`` has an EMPTY
    real-space list -- which is the correct list, and is what happens at
    ``L = 8 A`` with ``r_cut = 3.9 A``: ``erfc(alpha * 8) / 8`` is 1e-30 there,
    so the real-space sum genuinely contributes nothing.  The helper builds the
    shift tensor from a Python list, so an empty list comes back with shape
    ``[0]`` rather than ``[0, 3]`` and the ``edge_shift @ cell`` inside
    ``ewald_real`` fails on the shape.  Reshaping is the whole fix; it is done
    here rather than in ``conftest`` so that this file owns it.
    """
    edge_index, edge_shift = full_neighbour_list(pos, cell, cutoff)
    return edge_index, edge_shift.reshape(-1, 3)


def _point_defect_energies(length, madelung, accuracy=1.0e-8):
    """``(uncorrected, corrected)`` energy of one ``+1 e`` charge in a cubic cell.

    The conditions of the module docstring's table, verbatim: cubic cell of side
    ``length``, ``r_cut = min(6.0, L/2 - 0.1)``, ``MakovPayneParams(madelung,
    dielectric=1.0)``.  The ``Q_r`` term is not passed, because a point charge
    has ``Q_r = 0`` exactly and the monopole term is then the complete
    correction.
    """
    ewald_energy, EwaldParams, makov_payne_correction, MakovPayneParams = require(
        "ewald",
        "ewald_energy",
        "EwaldParams",
        "makov_payne_correction",
        "MakovPayneParams",
    )
    cell = torch.eye(3) * length
    r_cut = min(6.0, length / 2.0 - 0.1)
    params = EwaldParams.auto_tune(cell, r_cut, accuracy)
    pos = torch.zeros(1, 3)
    charges = torch.ones(1)
    edge_index, edge_shift = _neighbour_list(pos, cell, r_cut)

    uncorrected = ewald_energy(charges, pos, cell, edge_index, edge_shift, params)
    correction = makov_payne_correction(
        charges.sum(), cell, MakovPayneParams(madelung=madelung, dielectric=1.0)
    )
    return float(uncorrected), float(uncorrected + correction)


def _independent_jellium_madelung(alpha, n_real=8, n_recip=12):
    """``xi`` for the simple cubic lattice, from an Ewald sum written HERE.

    One unit point charge at the origin of a unit cube, neutralised by a uniform
    background, in units ``K_E = 1``:

        E = 1/2 [ sum_{n != 0} erfc(alpha |n|) / |n|
                  + (4 pi / V) sum_{k != 0} exp(-k^2 / 4 alpha^2) / k^2
                  - 2 alpha / sqrt(pi)
                  - pi / (alpha^2 V) ]

    and ``xi = -2 E V^(1/3) = -2 E`` at ``V = 1``.  Plain Python floats and
    ``math.erfc``; no tensor, no import from the package.  The four terms move
    strongly with ``alpha`` and the total does not, which is the standard
    correctness check and is exercised by the parametrisation.
    """
    real = 0.0
    for image in itertools.product(range(-n_real, n_real + 1), repeat=3):
        if image == (0, 0, 0):
            continue
        distance = math.sqrt(sum(component * component for component in image))
        real += math.erfc(alpha * distance) / distance

    recip = 0.0
    for wave in itertools.product(range(-n_recip, n_recip + 1), repeat=3):
        if wave == (0, 0, 0):
            continue
        k_squared = (2.0 * math.pi) ** 2 * sum(
            component * component for component in wave
        )
        recip += math.exp(-k_squared / (4.0 * alpha * alpha)) / k_squared
    recip *= 4.0 * math.pi

    self_term = -2.0 * alpha / math.sqrt(math.pi)
    jellium = -math.pi / (alpha * alpha)
    return -(real + recip + self_term + jellium)


def _power_law_exponent(lengths, values):
    """Least-squares slope of ``log |value|`` against ``log L``."""
    return float(
        numpy.polyfit(
            numpy.log(numpy.asarray(lengths, dtype=float)),
            numpy.log(numpy.abs(numpy.asarray(values, dtype=float))),
            1,
        )[0]
    )


# ===========================================================================
# 1.  xi = 2.8373, derived rather than believed
# ===========================================================================
@pytest.mark.parametrize("alpha", [2.0, 2.5, math.sqrt(math.pi), 3.0, 3.5])
def test_the_independent_jellium_sum_is_alpha_independent(alpha):
    """The reference this file uses for ``xi`` must be a converged Ewald sum.

    Before it can be used to judge the package, the hand-written sum in
    ``_independent_jellium_madelung`` has to be shown converged: its four terms
    each depend strongly on the splitting parameter and only the total does not.
    Five splittings spanning the self-dual ``sqrt(pi)``, all twelve digits
    identical.  Measured: 2.837297479481 at every alpha.
    """
    value = _independent_jellium_madelung(alpha)
    assert value == pytest.approx(2.837297479481, abs=1.0e-12), (
        f"the independent jellium sum gives xi = {value:.12f} at alpha = "
        f"{alpha:.6f}; it is not converged, so nothing below can lean on it"
    )


def test_the_package_simple_cubic_madelung_constant_matches_an_independent_sum():
    """``ewald.MADELUNG_SIMPLE_CUBIC`` and ``madelung_constant`` against that sum.

    The correction's whole size is set by ``xi``, so taking it on faith would
    make the rest of this file a comparison of the package with itself.  Three
    values are compared: the package's stored constant, what
    ``ewald.madelung_constant`` returns for a cubic cell, and the sum written in
    this file.  The textbook 2.8372974795 is checked alongside as a fourth,
    published, witness.

    Measured: independent sum 2.837297479481, ``MADELUNG_SIMPLE_CUBIC``
    2.837297479481, ``madelung_constant(eye(3) * L)`` 2.837297479481 at every L
    tried -- agreement 0.0, and 1.9e-11 against the six-digit textbook value,
    which is that value's own rounding.
    """
    madelung_constant, stored = require(
        "ewald", "madelung_constant", "MADELUNG_SIMPLE_CUBIC"
    )
    independent = _independent_jellium_madelung(math.sqrt(math.pi))

    assert float(stored) == pytest.approx(independent, abs=1.0e-11), (
        f"ewald.MADELUNG_SIMPLE_CUBIC = {float(stored):.12f} against an "
        f"independently summed {independent:.12f}"
    )
    assert independent == pytest.approx(
        TEXTBOOK_MADELUNG_SIMPLE_CUBIC, abs=1.0e-9
    ), f"independent sum {independent:.12f} vs textbook 2.8372974795"

    for length in (1.0, 5.3, 12.0):
        computed = float(madelung_constant(torch.eye(3) * length))
        assert computed == pytest.approx(independent, abs=1.0e-11), (
            f"madelung_constant(eye(3) * {length}) = {computed:.12f} against "
            f"{independent:.12f}; xi is scale free and must not move with L"
        )


def test_the_table_constant_2p8373_is_that_number_rounded_to_five_digits():
    """``2.8373`` is ``xi`` rounded, and the rounding error is 2.52e-06.

    This is the number the regression table was measured with, and the size of
    its error is what the corrected column of that table turns out to be.  It is
    pinned here so the explanation in
    ``test_the_surviving_trend_is_the_rounding_of_xi`` has something to point at.
    """
    independent = _independent_jellium_madelung(math.sqrt(math.pi))
    rounding = TABLE_MADELUNG - independent
    assert rounding == pytest.approx(2.520519e-06, rel=1.0e-5), (
        f"2.8373 - xi = {rounding:.6e}, expected 2.520519e-06"
    )


def test_the_simple_cubic_constant_sits_below_the_uniform_sphere_bound():
    """``xi_sc < (9/5) (4 pi / 3)^(1/3) = 2.9015855``, an independent closed form.

    A unit point charge at the centre of a uniform neutralising sphere of radius
    ``R`` has energy ``-(9/10) q^2 / R`` (the classic self-energy of a uniform
    ball plus the charge-background term; Ashcroft and Mermin, chapter 17,
    equation 17.19).  Writing ``R = (3 V / 4 pi)^(1/3)`` and comparing with
    ``E = -K_E q^2 xi / (2 V^(1/3))`` gives ``xi_sphere = (9/5)(4 pi/3)^(1/3)``.
    The sphere is the shape that maximises the Madelung energy, so no lattice
    can reach it -- this is an inequality a wrong ``xi`` can violate, and it
    catches the conventional-cube convention error (fcc's tabulated 4.584862)
    immediately.
    """
    bound = 1.8 * (4.0 * math.pi / 3.0) ** (1.0 / 3.0)
    package_bound = require("ewald", "MADELUNG_UNIFORM_SPHERE")
    independent = _independent_jellium_madelung(math.sqrt(math.pi))

    assert float(package_bound) == pytest.approx(bound, rel=1.0e-15)
    assert independent < bound, (
        f"xi_sc = {independent:.6f} is above the uniform-sphere bound "
        f"{bound:.6f}, which no lattice can exceed"
    )
    assert bound - independent == pytest.approx(0.064288, abs=1.0e-5), (
        f"the sc lattice sits {bound - independent:.6f} below the sphere bound"
    )


# ===========================================================================
# 2.  The measured table
# ===========================================================================
@pytest.mark.parametrize("length, uncorrected, corrected", MEASURED_TABLE)
def test_the_measured_finite_size_table_is_reproduced(length, uncorrected, corrected):
    """Regression on the numbers quoted in the module docstring.

    Measured directly, under the parameters named there.  The uncorrected column
    is also the Madelung energy ``-K_E xi / (2 L)`` of a charge in jellium, so it
    is checked against that closed form at the same time: agreement 6.4e-11 eV,
    which is the ``accuracy = 1e-8`` Ewald truncation of this cell and not a
    disagreement.
    """
    stored = float(require("ewald", "MADELUNG_SIMPLE_CUBIC"))
    measured_uncorrected, measured_corrected = _point_defect_energies(
        length, TABLE_MADELUNG
    )

    assert measured_uncorrected == pytest.approx(uncorrected, abs=5.0e-9), (
        f"L = {length} A: uncorrected {measured_uncorrected:.10f} eV against the "
        f"recorded {uncorrected:.8f} eV"
    )
    assert measured_corrected == pytest.approx(corrected, abs=5.0e-9), (
        f"L = {length} A: corrected {measured_corrected:.10e} eV against the "
        f"recorded {corrected:.8f} eV"
    )

    madelung_energy = -K_E * stored / (2.0 * length)
    assert measured_uncorrected == pytest.approx(madelung_energy, abs=1.0e-9), (
        f"L = {length} A: the uncorrected energy {measured_uncorrected:.12f} eV "
        f"is not the jellium Madelung energy -K_E xi / 2L = "
        f"{madelung_energy:.12f} eV, so the test bed is not what it claims"
    )


def test_the_uncorrected_energy_goes_as_one_over_L_and_the_corrected_one_is_flat():
    """The two trends, fitted.  This is the whole point of the correction.

    ``E_uncorrected ~ L^-1`` is the finite-size error of a charged supercell; if
    the fitted exponent were not -1 the test bed would not be exhibiting the
    error Makov-Payne exists to remove.  Measured exponent **-1.0000000000**
    over L = 8 to 20 A, and a span of **1.5321029 eV**.

    After the correction the same five cells span **1.3611e-06 eV**, a reduction
    of **1.126e+06**, against a physical target of exactly zero (an isolated
    point charge in vacuum has no self energy in this convention).  "Flat" is
    asserted as a bound on the span rather than as a fitted exponent, because a
    residual of constant SHAPE and negligible SIZE still fits a power law -- and
    the next test identifies what that residual is.
    """
    lengths = [row[0] for row in MEASURED_TABLE]
    uncorrected = []
    corrected = []
    for length in lengths:
        raw, fixed = _point_defect_energies(length, TABLE_MADELUNG)
        uncorrected.append(raw)
        corrected.append(fixed)

    exponent = _power_law_exponent(lengths, uncorrected)
    assert exponent == pytest.approx(-1.0, abs=1.0e-6), (
        f"the uncorrected charged-cell energy fits L^{exponent:.6f}, not L^-1; "
        "the test bed is not showing the 1/L finite-size error"
    )

    raw_span = max(uncorrected) - min(uncorrected)
    fixed_span = max(corrected) - min(corrected)
    assert raw_span == pytest.approx(1.5321029, abs=1.0e-6), (
        f"uncorrected span {raw_span:.7f} eV, recorded 1.5321029 eV"
    )
    assert fixed_span < 2.0e-06, (
        f"corrected span {fixed_span:.6e} eV, recorded 1.3611e-06 eV"
    )
    assert raw_span / fixed_span > 1.0e6, (
        f"the correction flattens the trend by only {raw_span / fixed_span:.3e}, "
        "recorded 1.126e+06"
    )
    assert max(abs(value) for value in corrected) < 3.0e-06, (
        "the corrected energies are not converging on the isolated-charge "
        f"answer of zero: largest |E| = {max(abs(v) for v in corrected):.3e} eV"
    )


def test_the_surviving_trend_is_the_rounding_of_xi_not_a_finite_size_error():
    """The residual 1e-06 is the INPUT constant, and this proves it two ways.

    A wrong ``xi`` leaves ``K_E (xi_used - xi_true) / (2 L)`` behind, exactly --
    a term with the same ``1/L`` shape as the error it was meant to remove, which
    is why a fitted exponent cannot distinguish the two.  Two checks that do:

    1.  Rescaling the residual by ``2 L / K_E`` must return the constant
        ``2.8373 - xi`` at every L.  Measured 2.5204e-06 against an exact
        2.5205e-06, at all five cells -- the fourth digit is the Ewald
        truncation at ``accuracy = 1e-8``.
    2.  Re-running with the package's own twelve-digit constant must make the
        residual vanish.  Measured max ``|E|`` = 7.0e-11 eV at
        ``accuracy = 1e-8``, and 3.5e-13 eV at ``accuracy = 1e-14`` -- i.e. the
        floor moves with the Ewald truncation, not with the cell size, which is
        what "the correction is exact here" means.
    """
    stored = float(require("ewald", "MADELUNG_SIMPLE_CUBIC"))
    expected_rounding = TABLE_MADELUNG - stored

    for length, _raw, _fixed in MEASURED_TABLE:
        _, corrected = _point_defect_energies(length, TABLE_MADELUNG)
        implied = corrected * 2.0 * length / K_E
        assert implied == pytest.approx(expected_rounding, rel=1.0e-3), (
            f"L = {length} A: the corrected residual implies a constant error of "
            f"{implied:.6e} in xi, but 2.8373 - xi = {expected_rounding:.6e}. "
            "The residual is then NOT the rounding of xi and something else "
            "survives the correction"
        )

    loose = [_point_defect_energies(row[0], stored)[1] for row in MEASURED_TABLE]
    tight = [
        _point_defect_energies(row[0], stored, accuracy=1.0e-14)[1]
        for row in MEASURED_TABLE
    ]
    assert max(abs(value) for value in loose) < 1.0e-09, (
        f"with the exact xi the corrected energy is still "
        f"{max(abs(v) for v in loose):.3e} eV from zero at accuracy 1e-8"
    )
    assert max(abs(value) for value in tight) < 1.0e-11, (
        f"with the exact xi and accuracy 1e-14 the corrected energy is still "
        f"{max(abs(v) for v in tight):.3e} eV from zero; the residual is then "
        "not the Ewald truncation"
    )
    assert max(abs(value) for value in tight) < max(abs(value) for value in loose), (
        "tightening the Ewald accuracy did not shrink the residual, so the "
        "residual is not the Ewald truncation after all"
    )


# ===========================================================================
# 3.  It cannot touch a neutral cell
# ===========================================================================
@pytest.mark.parametrize(
    "madelung, dielectric",
    [(2.8373, 1.0), (2.8373, 12.0), (2.2752, 1.0), (-0.458005, 3.5)],
)
@pytest.mark.parametrize("second_moment", [None, 3.7, -12.0])
def test_the_correction_is_identically_zero_for_a_neutral_cell(
    madelung, dielectric, second_moment
):
    """Both terms carry a factor of ``q``, so a neutral cell gets exactly ``0.0``.

    Not "small": zero to the last bit, for every ``xi`` (including the negative
    one a ``c/a = 4`` tetragonal cell has), every ``epsilon``, every ``Q_r`` and
    every cell shape -- and with an exactly zero gradient with respect to the
    cell, so the correction cannot leak into the stress of a neutral system
    either.  That is what makes it safe to leave enabled: it is opt-in
    structurally, and inert where it does not apply.
    """
    makov_payne_terms, MakovPayneParams = require(
        "ewald", "makov_payne_terms", "MakovPayneParams"
    )
    params = MakovPayneParams(madelung=madelung, dielectric=dielectric)
    moment = None if second_moment is None else torch.tensor(second_moment)

    shapes = {
        "cubic": torch.eye(3) * 11.3,
        "triclinic": torch.tensor(
            [[9.0, 0.0, 0.0], [1.3, 8.1, 0.0], [0.7, -2.1, 10.4]]
        ),
    }
    for name, shape in shapes.items():
        cell = shape.clone().requires_grad_(True)
        terms = makov_payne_terms(
            torch.tensor(0.0), cell, params, second_moment=moment
        )
        assert float(terms.monopole) == 0.0, (
            f"{name}: monopole term {float(terms.monopole)!r} on a neutral cell"
        )
        assert float(terms.quadrupole) == 0.0, (
            f"{name}: Q_r term {float(terms.quadrupole)!r} on a neutral cell"
        )
        (gradient,) = torch.autograd.grad(terms.total, cell)
        assert float(gradient.abs().max()) == 0.0, (
            f"{name}: a neutral cell picked up a stress contribution of "
            f"{float(gradient.abs().max()):.3e} eV per unit cell derivative"
        )


def test_a_neutral_cell_energy_is_bit_identical_with_the_correction_added():
    """The operational form: adding it changes nothing a neutral system reports.

    A neutral four-site cell with a genuinely non-zero second moment
    (``Q_r = -0.5 e A^2``), run through ``defect_moments`` exactly as a caller
    would, then through the correction.  The Ewald energy must come back
    unchanged to the last bit -- not to 1e-15.  Charges are chosen to sum to
    zero exactly in binary, so ``q`` really is ``0.0`` and not ``5.6e-17``.
    """
    (
        ewald_energy,
        EwaldParams,
        defect_moments,
        makov_payne_correction,
        MakovPayneParams,
    ) = require(
        "ewald",
        "ewald_energy",
        "EwaldParams",
        "defect_moments",
        "makov_payne_correction",
        "MakovPayneParams",
    )
    cell = torch.eye(3) * 10.0
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ) + 5.0
    charges = torch.tensor([0.5, 0.25, -0.25, -0.5])
    assert float(charges.sum()) == 0.0

    moments = defect_moments(charges, pos, cell)
    assert float(moments.second_moment) != 0.0, "the test bed has no Q_r to ignore"

    params = EwaldParams.auto_tune(cell, 4.9, 1.0e-10)
    edge_index, edge_shift = _neighbour_list(pos, cell, 4.9)
    bare = ewald_energy(charges, pos, cell, edge_index, edge_shift, params)
    corrected = bare + makov_payne_correction(
        moments.charge,
        cell,
        MakovPayneParams(madelung=TABLE_MADELUNG),
        second_moment=moments.second_moment,
    )
    assert float(bare) == float(corrected), (
        f"a neutral cell moved by {float(corrected) - float(bare):.3e} eV when "
        "the Makov-Payne correction was added"
    )


# ===========================================================================
# 4.  It contributes to the stress, and that contribution is differentiable
# ===========================================================================
def _charged_defect():
    """A ``+1 e`` seven-point defect in a 12 A cubic cell.

    Seven charges on a small octahedron about the cell centre, summing to +1 e
    with ``Q_r = 0.6 e A^2`` and ``r_q / L = 0.065`` -- well inside the
    ``DEFAULT_MAX_RADIUS_RATIO`` of 0.25, so the multipole expansion is in its
    regime and ``check_defect_localisation`` accepts it.
    """
    length = 12.0
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    ) + length / 2.0
    charges = torch.tensor([0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    return length, pos, charges


def _strained_energy(with_correction=True):
    """``U(strain)`` for the charged defect: Ewald, optionally plus Makov-Payne.

    The strain convention of ``test_gradients._periodic_case`` and of
    ``flashace.model``: one deformation applied to both the positions and the
    cell, with the image shifts held as integers so that ``edge_shift @ cell``
    carries the cell dependence.  ``Q_r`` and ``q`` are detached constants here,
    so what is differentiated is the correction's explicit dependence on ``V``
    and nothing else -- which is exactly the claim under test.
    """
    ewald_energy, EwaldParams, makov_payne_correction, MakovPayneParams = require(
        "ewald",
        "ewald_energy",
        "EwaldParams",
        "makov_payne_correction",
        "MakovPayneParams",
    )
    defect_moments, check_defect_localisation = require(
        "ewald", "defect_moments", "check_defect_localisation"
    )
    length, pos, charges = _charged_defect()
    reference_cell = torch.eye(3) * length
    moments = defect_moments(charges, pos, reference_cell)
    check_defect_localisation(moments)

    r_cut = 5.5
    params = EwaldParams.auto_tune(reference_cell, r_cut, 1.0e-10)
    edge_index, edge_shift = _neighbour_list(pos, reference_cell, r_cut)
    mp_params = MakovPayneParams(
        madelung=float(require("ewald", "MADELUNG_SIMPLE_CUBIC")), dielectric=1.0
    )
    charge = moments.charge.detach().clone()
    second_moment = moments.second_moment.detach().clone()

    def energy_of(strain):
        deformation = torch.eye(3, dtype=strain.dtype) + strain
        strained_pos = pos @ deformation.transpose(0, 1)
        strained_cell = reference_cell @ deformation.transpose(0, 1)
        energy = ewald_energy(
            charges, strained_pos, strained_cell, edge_index, edge_shift, params
        )
        if with_correction:
            energy = energy + makov_payne_correction(
                charge, strained_cell, mp_params, second_moment=second_moment
            )
        return energy

    return energy_of


def _numeric_strain_matrix(function, stencil, step):
    """Finite-difference ``dU/d eps`` for a 3x3 strain, as in test_gradients."""
    numeric = torch.zeros(3, 3)
    for a in range(3):
        for b in range(3):

            def shifted(amount):
                strain = torch.zeros(3, 3)
                strain[a, b] = amount
                return float(function(strain).detach())

            if stencil == "two-point":
                numeric[a, b] = (shifted(step) - shifted(-step)) / (2.0 * step)
            else:  # five-point, O(h^4) truncation
                numeric[a, b] = (
                    shifted(-2.0 * step)
                    - 8.0 * shifted(-step)
                    + 8.0 * shifted(step)
                    - shifted(2.0 * step)
                ) / (12.0 * step)
    return numeric


def test_the_correction_actually_changes_the_stress():
    """Before checking the derivative, check there is something to check.

    ``E_corr`` depends on ``V``, so it must contribute to the stress; a test of
    ``autograd == FD`` on a term that happened to contribute nothing would pass
    for the wrong reason.  Measured on the ``+1 e`` defect at L = 12 A: the
    diagonal of ``dU/d eps`` moves from -1.0673 eV to -1.6243 eV, a contribution
    of **0.5570 eV per unit strain** -- 52 % of the uncorrected value.
    """
    strain = torch.zeros(3, 3, requires_grad=True)
    (with_correction,) = torch.autograd.grad(_strained_energy(True)(strain), strain)
    strain = torch.zeros(3, 3, requires_grad=True)
    (without,) = torch.autograd.grad(_strained_energy(False)(strain), strain)

    contribution = float((with_correction - without).abs().max())
    assert contribution == pytest.approx(0.5570, abs=1.0e-3), (
        f"the Makov-Payne stress contribution is {contribution:.6f} eV per unit "
        "strain, recorded 0.5570"
    )
    assert float(without.diagonal().max()) == pytest.approx(-1.0673, abs=1.0e-3)
    assert float(with_correction.diagonal().max()) == pytest.approx(
        -1.6243, abs=1.0e-3
    )


@pytest.mark.parametrize(
    "stencil, step, tolerance, recorded",
    [
        ("two-point", 1.0e-5, 4.6e-10, 2.76e-10),
        # None: the budget is the stencil's round-off floor, computed below.
        ("five-point", 1.0e-4, None, 9.20e-12),
    ],
)
def test_the_corrected_stress_matches_finite_differences(
    stencil, step, tolerance, recorded
):
    """``d/dcell`` of ``Ewald + Makov-Payne`` against FD, at two stencil qualities.

    SPEC section 6 quotes 4.6e-10 as the FD ROUNDOFF FLOOR of a two-point
    stencil at ``h = 1e-5``, not as an implementation defect, and the falsifiable
    form of that claim is that a better stencil must shrink the disagreement.
    Both are run here on the charged-defect energy WITH the correction included:

        two-point, h = 1e-5    2.76e-10 eV per unit strain
        five-point, h = 1e-4   9.20e-12

    a factor of 30, and the five-point result is below the 1e-10 SPEC asks of
    every other derivative.  A wrong cell derivative in the correction would
    survive both.

    The five-point budget is its ROUND-OFF FLOOR, not a recorded number: each
    energy carries a rounding error of about one ulp, ``u |E|``, and the stencil
    weights (1, 8, 8, 1) / 12 amplify it to ``1.5 u |E| / h`` -- 1.63e-11 at
    ``|E| = 4.894`` eV, ``h = 1e-4``.  A fixed 1e-11 budget sat BELOW that floor
    and passed on one machine by the luck of its rounding (9.2e-12 on macOS /
    torch 2.2; 1.09e-11 on Linux / torch 2.11).  The budget is the floor at four
    ulps, 6.5e-11: still far below the 0.557 eV contribution of the correction
    it guards and the 1e-10 SPEC asks of every derivative.
    """
    energy_of = _strained_energy(True)
    strain = torch.zeros(3, 3, requires_grad=True)
    energy = energy_of(strain)
    (analytic,) = torch.autograd.grad(energy, strain)
    numeric = _numeric_strain_matrix(energy_of, stencil, step)
    if tolerance is None:
        unit_roundoff = torch.finfo(torch.float64).eps
        tolerance = 4.0 * 1.5 * unit_roundoff * abs(float(energy)) / step

    error = float((analytic - numeric).abs().max())
    assert error < tolerance, (
        f"{stencil} at h = {step:g}: max |autograd - FD| = {error:.3e} eV per "
        f"unit strain, recorded {recorded:.3g}, budget {tolerance:.1e}"
    )


def test_the_correction_cell_derivative_matches_its_closed_form():
    """``dE_corr/dcell`` against the derivative written out by hand.

    Finite differences bound the error at 1e-11; the closed form pins it at
    round-off.  With ``V = |det cell|`` and ``dV/dcell = V (cell^-1)^T``,

        E_corr = K_E q^2 xi / (2 eps V^(1/3)) - 2 pi K_E q Q_r / (3 eps V)
        dE_corr/dV = -E_mono / (3 V) - E_quad / V

    so the whole cell derivative is ``(-E_mono/3 - E_quad) (cell^-1)^T``.
    Measured agreement: **2.1e-17 eV**, i.e. exact.  This also fixes the
    EXPONENTS: a term written as ``V^(-1/2)`` or the ``Q_r`` term written as
    ``V^(-2/3)`` would reproduce the energy at one cell and fail here.
    """
    makov_payne_terms, MakovPayneParams = require(
        "ewald", "makov_payne_terms", "MakovPayneParams"
    )
    length = 12.0
    params = MakovPayneParams(
        madelung=float(require("ewald", "MADELUNG_SIMPLE_CUBIC")), dielectric=1.0
    )
    charge = torch.tensor(1.0)
    second_moment = torch.tensor(0.6)

    cell = (torch.eye(3) * length).clone().requires_grad_(True)
    terms = makov_payne_terms(charge, cell, params, second_moment=second_moment)
    (analytic,) = torch.autograd.grad(terms.total, cell)

    volume = length ** 3
    monopole = float(terms.monopole)
    quadrupole = float(terms.quadrupole)
    d_energy_d_volume = -monopole / (3.0 * volume) - quadrupole / volume
    closed_form = d_energy_d_volume * volume * torch.linalg.inv(
        torch.eye(3) * length
    ).transpose(0, 1)

    error = float((analytic - closed_form).abs().max())
    assert error < 1.0e-15, (
        f"dE_corr/dcell differs from the closed form by {error:.3e} eV; "
        f"autograd\n{analytic}\nclosed form\n{closed_form}"
    )
