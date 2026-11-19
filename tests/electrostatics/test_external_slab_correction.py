"""EXTERNAL validation of the 2-D slab (Yeh-Berkowitz) correction -- SPEC 7.

``ewald.py`` carries ``SLAB_CORRECTION_COEFFICIENT = 2.0`` and, in its own
docstring, four claims about it:

1. the coefficient is ``2``, not ``4``;
2. the sign of ``U = C pi K_E M_n^2 / V`` is POSITIVE;
3. it was derived from a parallel-plate capacitor, where both the isolated and
   the tinfoil energies are closed forms;
4. it was checked against a direct 2-D lattice sum to 1.3e-08 eV on a 4.47 eV
   energy.

None of that is a test.  This file reproduces all four independently, and the
central reference -- the energy of a genuinely ISOLATED 2-D periodic slab -- is
computed here by a real-space sum over a finite array of in-plane images that
shares no line of code with the package.

The test bed
------------
A polar bilayer in an ``8 x 8 A`` square in-plane cell:

    +1 e at (0, 0, 0)     and (4, 4, 0)
    -1 e at (0, 4, 2.5)   and (4, 0, 2.5)

Neutral, with ``M_x = M_y = 0`` exactly by the in-plane checkerboard symmetry
and ``M_z = -5 e A`` exactly.  The in-plane components are made to vanish on
purpose: the cell-cell interaction at large in-plane separation ``T`` is
``K_E [M.M - 3 (M.T_hat)^2] / |T|^3``, whose angular average over the plane is
``K_E [M_z^2 - (M_x^2 + M_y^2)/2] / |T|^3``, so a stray in-plane dipole would
change the tail of the reference sum by a factor that has nothing to do with the
slab physics (measured on a deliberately asymmetric variant: the fitted tail
coefficient came out -14.49 against the -17.67 of ``M_z`` alone, exactly the
``-0.5 (M_x^2 + M_y^2)`` term).

The reference, and how it is converged
--------------------------------------
    U_iso = (1/2) sum_T sum_{i,j}' K_E q_i q_j / |r_ij + T|

summed over the 2-D lattice translations ``T`` with ``|T| <= R``, whole cells at
a time (never truncated per pair -- the cell must stay neutral or the sum does
not converge).  Truncating at ``R`` leaves the dipole-dipole tail

    sum_{|T| > R} (1/2) K_E M_z^2 / |T|^3

which is ADDED BACK EXACTLY rather than extrapolated, using the closed form for
the square lattice's Epstein zeta function,

    sum_{(m,n) != 0} (m^2 + n^2)^(-s) = 4 zeta(s) beta(s)

(the standard factorisation over the Gaussian integers; Borwein and Borwein,
*Pi and the AGM*, and every treatment of 2-D lattice sums).  What remains after
that is the next multipole order, ``O(R^-3)`` -- measured, see
``test_the_extrapolation_is_third_order`` -- and one Richardson step in
``R^-3`` removes it.

Measured:

    R (A)      U_iso estimate        successive delta   ratio
      200      -4.7414134088626
      400      -4.7413865306506      2.6878e-05         8.70
      800      -4.7413831074421      3.4232e-06         7.85
     1600      -4.7413826779518      4.2949e-07         7.97
     3200      -4.7413826240792      5.3873e-08         7.97

    Richardson (R^-3) on the last pair    -4.7413826163831
    package ewald_slab_energy             -4.7413826163826

    agreement                              5.0e-13 eV on a 4.74 eV energy

which is five decades better than the 1.3e-08 the module docstring claims, and
the ratio settling on 8 = 2^3 is what identifies the extrapolation order.
"""

import functools
import math

import numpy
import pytest
import torch

from conftest import full_neighbour_list, require


K_E = 14.399645  # SPEC 0

IN_PLANE = 8.0
LAYER_SEPARATION = 2.5
AREA = IN_PLANE * IN_PLANE

SLAB_PBC = (True, True, False)

# The polar bilayer of the module docstring.  M_z = -5 e A exactly.
POLAR_POS = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [4.0, 4.0, 0.0],
        [0.0, 4.0, LAYER_SEPARATION],
        [4.0, 0.0, LAYER_SEPARATION],
    ]
)
POLAR_CHARGES = torch.tensor([1.0, 1.0, -1.0, -1.0])
POLAR_MZ = -2.0 * LAYER_SEPARATION

# A NON-polar slab: the same lateral arrangement, mirror-symmetric about
# z = 2.5, so sum_i q_i z_i = 0 - 2.5 - 2.5 + 5.0 = 0 in exact binary.
NONPOLAR_POS = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [0.0, 4.0, 2.5],
        [4.0, 0.0, 2.5],
        [4.0, 4.0, 5.0],
    ]
)
NONPOLAR_CHARGES = torch.tensor([1.0, -1.0, -1.0, 1.0])

# Converged package value for the polar bilayer; stable to 1e-13 over
# L_z in {40, 60, 80} A, r_cut in {7.5, 9.0} A and accuracy in {1e-12, 1e-14}.
PACKAGE_ISOLATED_SLAB_ENERGY = -4.7413826163826

# Published constants, used only to check the Euler-Maclaurin helper below.
ZETA_THREE_HALVES = 2.612375348685488
BETA_THREE_HALVES = 0.864502653461202


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
def _neighbour_list(pos, cell, cutoff):
    """``conftest.full_neighbour_list`` with a shift tensor that is always [n, 3].

    The helper builds its shifts from a Python list, so an empty list comes back
    with shape ``[0]`` and ``edge_shift @ cell`` fails on the shape.  Reshaping
    here keeps that detail out of every call site.
    """
    edge_index, edge_shift = full_neighbour_list(pos, cell, cutoff)
    return edge_index, edge_shift.reshape(-1, 3)


def _padded_cell(height, pbc=SLAB_PBC):
    """An ``8 x 8 x height`` orthogonal cell with the rows cyclically permuted.

    The non-periodic lattice vector is placed at the index ``pbc`` says is
    non-periodic, by a CYCLIC permutation of the rows -- which is even, so the
    determinant keeps its sign and the cell stays right handed.  The Cartesian
    geometry is identical in all three cases; only which row is the vacuum one
    changes, which is precisely what ``non_periodic_axis`` has to get right.
    """
    vectors = [
        torch.tensor([IN_PLANE, 0.0, 0.0]),
        torch.tensor([0.0, IN_PLANE, 0.0]),
        torch.tensor([0.0, 0.0, height]),
    ]
    axis = require("ewald", "non_periodic_axis")(pbc)
    order = [(axis + 1) % 3, (axis + 2) % 3, axis]
    rows = [None, None, None]
    rows[order[0]] = vectors[0]
    rows[order[1]] = vectors[1]
    rows[order[2]] = vectors[2]
    return torch.stack(rows)


def _slab_energies(
    pos,
    charges,
    height,
    pbc=SLAB_PBC,
    cell=None,
    r_cut=7.5,
    accuracy=1.0e-12,
    validate=False,
):
    """``(U_3D, correction, U_slab)`` for a slab at a given vacuum padding.

    ``validate`` defaults to False because the padding scans below deliberately
    go BELOW the padding ``check_slab_geometry`` requires -- that is the point of
    a padding scan.  The tests that stay in the converged regime turn it on.
    """
    (
        ewald_energy,
        ewald_slab_energy,
        slab_correction,
        EwaldParams,
    ) = require(
        "ewald",
        "ewald_energy",
        "ewald_slab_energy",
        "slab_correction",
        "EwaldParams",
    )
    if cell is None:
        cell = _padded_cell(height, pbc)
    params = EwaldParams.auto_tune_slab(cell, r_cut, pbc, accuracy)
    edge_index, edge_shift = _neighbour_list(pos, cell, r_cut)

    bulk = float(ewald_energy(charges, pos, cell, edge_index, edge_shift, params))
    correction = float(slab_correction(charges, None, pos, cell, pbc))
    slab = float(
        ewald_slab_energy(
            charges,
            pos,
            cell,
            edge_index,
            edge_shift,
            params,
            pbc,
            validate=validate,
        )
    )
    return bulk, correction, slab


# ---------------------------------------------------------------------------
# The independent reference: a direct real-space 2-D lattice sum
# ---------------------------------------------------------------------------
def _hurwitz_zeta(s, a, n_terms=40):
    """``zeta(s, a)`` by Euler-Maclaurin, in plain Python floats.

        zeta(s, a) = sum_{k<N} (a+k)^-s + x^(1-s)/(s-1) + x^-s/2
                     + s x^(-s-1)/12 - s(s+1)(s+2) x^(-s-3)/720
                     + s(s+1)(s+2)(s+3)(s+4) x^(-s-5)/30240 ,   x = a + N

    Written here so that the Epstein zeta the reference sum needs is DERIVED
    rather than pasted in.  At ``N = 40`` and ``s = 3/2`` the first neglected
    Bernoulli term is ~1e-19, and the helper is checked against published
    values in ``test_the_lattice_zeta_helper_reproduces_published_constants``.
    """
    total = sum((a + k) ** (-s) for k in range(n_terms))
    x = a + n_terms
    total += x ** (1.0 - s) / (s - 1.0) + 0.5 * x ** (-s)
    total += s * x ** (-s - 1.0) / 12.0
    total -= s * (s + 1.0) * (s + 2.0) * x ** (-s - 3.0) / 720.0
    total += (
        s * (s + 1.0) * (s + 2.0) * (s + 3.0) * (s + 4.0) * x ** (-s - 5.0) / 30240.0
    )
    return total


def _square_lattice_zeta(s):
    """``sum_{(m,n) != (0,0)} (m^2 + n^2)^(-s) = 4 zeta(s) beta(s)``.

    The Epstein zeta of the square lattice factorises over the Gaussian
    integers; ``beta`` is the Dirichlet beta function, obtained here from the
    Hurwitz zeta as ``beta(s) = 4^-s [zeta(s, 1/4) - zeta(s, 3/4)]``.
    """
    zeta = _hurwitz_zeta(s, 1.0)
    beta = 4.0 ** (-s) * (_hurwitz_zeta(s, 0.25) - _hurwitz_zeta(s, 0.75))
    return 4.0 * zeta * beta


@functools.lru_cache(maxsize=None)
def _isolated_slab_estimate(radius):
    """Energy per cell of the ISOLATED 2-D periodic polar bilayer, in eV.

        U(R) = (1/2) K_E sum_{|T| <= R} sum_{i,j}' q_i q_j / |r_ij + T|
               + (1/2) K_E M_z^2 [ Z / a^3 - sum_{0 < |T| <= R} |T|^-3 ]

    The first line is the brute-force sum over whole image cells; the second is
    the exact dipole-dipole tail beyond ``R``, with ``Z = 4 zeta(3/2) beta(3/2)``
    the square-lattice Epstein zeta.  Perpendicular dipoles have
    ``M . T_hat = 0``, so the tail term is ``+K_E M_z^2 / |T|^3`` per pair with
    no angular factor, and ``M_x = M_y = 0`` for this bilayer by construction.

    Nothing from ``trace_electrostatics`` is used.
    """
    first = torch.tensor([IN_PLANE, 0.0, 0.0])
    second = torch.tensor([0.0, IN_PLANE, 0.0])
    half_width = int(math.ceil(radius / IN_PLANE)) + 1
    steps = torch.arange(-half_width, half_width + 1, dtype=torch.float64)
    rows, columns = torch.meshgrid(steps, steps, indexing="ij")
    translations = rows.reshape(-1, 1) * first + columns.reshape(-1, 1) * second
    lengths = translations.norm(dim=-1)
    inside = lengths <= radius + 1.0e-9
    translations, lengths = translations[inside], lengths[inside]

    separations = POLAR_POS.unsqueeze(1) - POLAR_POS.unsqueeze(0)
    products = POLAR_CHARGES.unsqueeze(1) * POLAR_CHARGES.unsqueeze(0)
    pair_sum = 0.0
    for start in range(0, translations.shape[0], 20000):
        block = translations[start : start + 20000]
        distance = (separations.unsqueeze(0) + block.view(-1, 1, 1, 3)).norm(dim=-1)
        inverse = torch.where(
            distance > 1.0e-9, 1.0 / distance, torch.zeros_like(distance)
        )
        pair_sum += float((products.unsqueeze(0) * inverse).sum())

    summed_inside = float((1.0 / lengths[lengths > 1.0e-9] ** 3).sum())
    tail = _square_lattice_zeta(1.5) / IN_PLANE ** 3 - summed_inside
    return 0.5 * K_E * pair_sum + 0.5 * K_E * POLAR_MZ ** 2 * tail


REFERENCE_RADII = (200.0, 400.0, 800.0, 1600.0, 3200.0)


def _richardson_isolated_slab_energy():
    """The reference, one Richardson step in ``R^-3`` on the last two radii."""
    near = _isolated_slab_estimate(REFERENCE_RADII[-2])
    far = _isolated_slab_estimate(REFERENCE_RADII[-1])
    return far + (far - near) / 7.0


# ===========================================================================
# 1.  The reference sum: is it what it claims to be?
# ===========================================================================
def test_the_lattice_zeta_helper_reproduces_published_constants():
    """``zeta(3/2)``, ``beta(3/2)`` and the square-lattice Epstein zeta.

    The reference sum leans on ``Z = 4 zeta(3/2) beta(3/2) = 9.033621683101``
    for its exact tail, so the helper that produces it is checked first --
    against the published ``zeta(3/2) = 2.612375348685488`` and
    ``beta(3/2) = 0.864502653461202``, and against a brute-force sum over the
    lattice itself, which tests the FACTORISATION rather than the two constants.

    Measured: the Euler-Maclaurin helper agrees with both published constants to
    3e-16, and the brute-force lattice sum (radius 500, plus its ``2 pi / R``
    continuum tail) agrees with ``4 zeta beta`` to 3.2e-08.
    """
    assert _hurwitz_zeta(1.5, 1.0) == pytest.approx(ZETA_THREE_HALVES, abs=1.0e-14)
    beta = 4.0 ** (-1.5) * (_hurwitz_zeta(1.5, 0.25) - _hurwitz_zeta(1.5, 0.75))
    assert beta == pytest.approx(BETA_THREE_HALVES, abs=1.0e-14)

    closed_form = _square_lattice_zeta(1.5)
    assert closed_form == pytest.approx(9.033621683100950, abs=1.0e-12)

    radius = 500
    steps = torch.arange(-radius - 1, radius + 2, dtype=torch.float64)
    rows, columns = torch.meshgrid(steps, steps, indexing="ij")
    lengths = torch.sqrt(rows * rows + columns * columns).reshape(-1)
    lengths = lengths[(lengths > 0.5) & (lengths <= radius + 1.0e-9)]
    brute = float((1.0 / lengths ** 3).sum()) + 2.0 * math.pi / radius

    assert brute == pytest.approx(closed_form, abs=1.0e-6), (
        f"the square-lattice Epstein zeta by brute force is {brute:.10f} but "
        f"4 zeta(3/2) beta(3/2) = {closed_form:.10f}; the factorisation, not "
        "the constants, is what this disagrees about"
    )


def test_the_extrapolation_is_third_order():
    """The residual after the exact dipole tail falls as ``R^-3``, measured.

    This is what licenses the single Richardson step with a factor of 1/7.  The
    successive differences of ``U(R)`` over a doubling ladder must fall by
    ``2^3 = 8``; if the remainder were the dipole term itself they would fall by
    2, and if it were the next even multipole they would fall by 4.

    Measured ratios over R = 200, 400, 800, 1600, 3200 A:
    **8.70, 7.85, 7.97, 7.97** -- settling on 8, so the leading neglected
    cell-cell term goes as ``|T|^-5`` and the ``|T|^-4`` term is killed by the
    bilayer's in-plane symmetry.
    """
    values = [_isolated_slab_estimate(radius) for radius in REFERENCE_RADII]
    deltas = [later - earlier for earlier, later in zip(values, values[1:])]
    ratios = [
        earlier / later for earlier, later in zip(deltas, deltas[1:])
    ]
    assert ratios[-1] == pytest.approx(8.0, abs=0.1), (
        f"successive-difference ratios {['%.2f' % r for r in ratios]}; the last "
        "one should be 8 if the remainder is O(R^-3)"
    )
    assert ratios[-2] == pytest.approx(8.0, abs=0.2), (
        f"successive-difference ratios {['%.2f' % r for r in ratios]}"
    )


def test_the_slab_energy_matches_an_independent_two_dimensional_lattice_sum():
    """THE reference test: ``ewald_slab_energy`` against a direct 2-D sum.

    The package computes the isolated slab as 3-D Ewald in a padded cell plus
    ``2 pi K_E M_n^2 / V``.  The sum in ``_isolated_slab_estimate`` computes the
    same quantity with no Ewald splitting, no reciprocal space, no padding and
    no correction at all -- just Coulomb's law over a large finite array of
    in-plane images, with the analytically known dipole tail added back.  They
    have no machinery in common, so agreement tests the correction itself.

    Measured:

        independent sum, Richardson (R^-3)   -4.7413826163831 eV
        ewald_slab_energy (L_z = 60 A)       -4.7413826163826 eV
        agreement                             5.0e-13 eV

    on a 4.74 eV energy -- a relative 1.1e-13, and five decades better than the
    1.3e-08 the ``ewald`` module docstring records.  ``validate=True`` here, so
    the padding check runs on the real thing as well.
    """
    reference = _richardson_isolated_slab_energy()
    _bulk, _correction, package = _slab_energies(
        POLAR_POS, POLAR_CHARGES, 60.0, validate=True
    )

    assert package == pytest.approx(PACKAGE_ISOLATED_SLAB_ENERGY, abs=1.0e-12), (
        f"ewald_slab_energy = {package:.13f} eV against the recorded "
        f"{PACKAGE_ISOLATED_SLAB_ENERGY:.13f} eV"
    )
    error = abs(package - reference)
    assert error < 1.0e-9, (
        f"the package's isolated-slab energy {package:.13f} eV disagrees with an "
        f"independent 2-D lattice sum {reference:.13f} eV by {error:.3e} eV "
        "(recorded 5.0e-13)"
    )


# ===========================================================================
# 2.  The decisive test: padding does NOT fix plain 3-D Ewald
# ===========================================================================
PADDINGS = (20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0, 240.0, 320.0)


@functools.lru_cache(maxsize=None)
def _padding_scan():
    """``(L_z, U_3D, U_slab)`` over the padding ladder, computed once."""
    return tuple(
        (height,) + _slab_energies(POLAR_POS, POLAR_CHARGES, height)[::2]
        for height in PADDINGS
    )


def _sufficiently_padded(height, accuracy=1.0e-12, pbc=SLAB_PBC):
    """Does ``check_slab_geometry`` accept this padding at this accuracy?

    The converged set below is defined by the package's OWN criterion rather
    than by a hand-picked ``L_z``, so that if the criterion is ever retuned the
    test follows it instead of silently drifting out of agreement with it.
    """
    slab_vacuum_report = require("ewald", "slab_vacuum_report")
    return slab_vacuum_report(
        POLAR_POS, _padded_cell(height, pbc), pbc, POLAR_CHARGES, accuracy
    ).sufficient


@pytest.mark.slow
def test_plain_three_d_ewald_does_not_converge_to_the_isolated_slab():
    """Vacuum padding does NOT remove the spurious term.  It halves it.

    Under tinfoil boundaries the ``k = 0`` term is dropped, which screens the
    depolarizing field of the stack of replica slabs; an isolated slab keeps
    that field.  The difference is ``-2 pi K_E M_z^2 / V``, and with
    ``V = A L_z`` it falls only as ``1/L_z``.  So this is the claim that makes
    the correction necessary rather than cosmetic, and it is a POSITIVE test:
    the uncorrected error must fit ``L^-1``, not merely be non-zero.

    Measured against the independent 2-D sum, over L_z = 20 to 320 A:

        fitted exponent   -1.00000
        error at L_z = 20 A    -1.767 eV
        error at L_z = 320 A   -0.110 eV

    A 16x increase in padding bought one decade.  Reaching 1e-06 eV this way
    needs ``L_z`` of order 1e+07 A.
    """
    reference = _richardson_isolated_slab_energy()
    rows = _padding_scan()
    heights = [row[0] for row in rows]
    errors = [row[1] - reference for row in rows]

    exponent = float(
        numpy.polyfit(numpy.log(heights), numpy.log(numpy.abs(errors)), 1)[0]
    )
    assert exponent == pytest.approx(-1.0, abs=1.0e-4), (
        f"the uncorrected slab error fits L_z^{exponent:.5f}; it must be L_z^-1, "
        "which is the whole reason the correction exists"
    )
    assert errors[0] == pytest.approx(-1.767, abs=1.0e-3), (
        f"error at L_z = {heights[0]} A is {errors[0]:.4f} eV, recorded -1.767"
    )
    assert abs(errors[-1]) > 0.1, (
        f"at L_z = {heights[-1]} A the uncorrected error has fallen to "
        f"{errors[-1]:.4f} eV; it is supposed to still be 0.110 eV"
    )


@pytest.mark.slow
def test_the_corrected_slab_energy_is_padding_independent():
    """The other half of the same trend: with the correction, padding stops mattering.

    Over L_z = 30 to 320 A -- an 11x range over which the uncorrected energy
    moves by 1.18 eV -- the corrected energy has a spread of **1.8e-14 eV**, and
    every value sits 2.1e-10 eV from the independent 2-D sum (a constant offset,
    which is that reference's own extrapolation error at these radii, not a
    drift).  The spread is taken over exactly the paddings ``check_slab_geometry``
    accepts at ``accuracy = 1e-12``, which wants a 36.3 A vacuum gap:

        L_z = 20 A   gap 17.5 A   rejected   residual 8.3e-08 eV
        L_z = 30 A   gap 27.5 A   rejected   residual 1.3e-12 eV
        L_z >= 40 A  gap 37.5 A   accepted   bit-identical, -4.741382616383 eV

    Both rejected paddings keep separate assertions below, so the corrugation
    trend stays locked in rather than merely skipped.  What they show is the
    limit of the claim: Yeh-Berkowitz removes the DIPOLE term of the replica
    interaction, and the ``exp(-|G|_min h)`` corrugation underneath it is left
    behind, which is why the criterion asks for a gap in the first place.
    """
    reference = _richardson_isolated_slab_energy()
    rows = _padding_scan()
    converged = [row[2] for row in rows if _sufficiently_padded(row[0])]
    assert len(converged) == 7, (
        f"{len(converged)} of {len(rows)} paddings clear check_slab_geometry at "
        "accuracy = 1e-12; recorded 7, L_z = 40 A upwards"
    )

    spread = max(converged) - min(converged)
    assert spread < 1.0e-12, (
        f"the corrected slab energy spans {spread:.3e} eV over L_z = 40-320 A "
        "(recorded 1.8e-14); it is supposed to be padding independent"
    )
    for height, _bulk, slab in rows:
        if height < 30.0:
            continue
        assert slab == pytest.approx(reference, abs=1.0e-9), (
            f"L_z = {height} A: corrected energy {slab:.13f} eV against the "
            f"independent sum {reference:.13f} eV"
        )

    # L_z = 30 A: rejected by the criterion, but only just.  Its residual has to
    # sit BETWEEN the converged spread and the L_z = 20 A corrugation -- that
    # ordering is the evidence the leftover really is the exp(-|G|_min h) term
    # and not a constant the correction is missing.
    marginal = [row for row in rows if row[0] == 30.0][0][2]
    assert not _sufficiently_padded(30.0), (
        "L_z = 30 A has a 27.5 A gap against the 36.3 A required at 1e-12; "
        "check_slab_geometry is supposed to reject it"
    )
    assert spread < abs(marginal - min(converged)) < 1.0e-09, (
        f"at L_z = 30 A the corrected energy is "
        f"{marginal - min(converged):.3e} eV from the converged plateau; it is "
        "supposed to be above the 1.8e-14 plateau spread and below the 1e-09 "
        "the criterion's own error estimate allows at that gap"
    )

    under_padded = [row for row in rows if row[0] == 20.0][0]
    residual = under_padded[2] - reference
    assert 1.0e-09 < abs(residual) < 1.0e-06, (
        f"at L_z = 20 A the corrected energy is {residual:.3e} eV from the "
        "reference; recorded 8.3e-08, the exp(-|G| h) replica corrugation"
    )


# ===========================================================================
# 3.  The coefficient, derived
# ===========================================================================
@pytest.mark.slow
@pytest.mark.parametrize("near, far", [(40.0, 320.0), (60.0, 240.0), (80.0, 160.0)])
def test_the_slab_coefficient_is_two_not_four(near, far):
    """``C = 2`` read off the package's own 3-D sum, without touching the constant.

    Everything in ``U_3D(L_z)`` except the spurious macroscopic term is
    independent of the padding (to the exponentially small replica corrugation),
    so with ``U_3D(L) = U_iso - C pi K_E M_z^2 / (A L)``,

        C = [U_3D(L_far) - U_3D(L_near)] * A / (pi K_E M_z^2 (1/L_near - 1/L_far))

    ``M_z = -5 e A`` and ``V = A L_z`` are exactly known from the geometry, so
    this is a clean measurement of the coefficient with NO reference to
    ``SLAB_CORRECTION_COEFFICIENT``, to ``slab_correction``, or to the isolated
    answer.  Measured: **2.000000000000** on every pair of well-padded cells.
    A coefficient of 4 -- the other value the literature quotes -- is out by a
    factor of two and could not survive this to twelve digits.
    """
    rows = {row[0]: row[1] for row in _padding_scan()}
    difference = rows[far] - rows[near]
    coefficient = (
        difference
        * AREA
        / (math.pi * K_E * POLAR_MZ ** 2 * (1.0 / near - 1.0 / far))
    )
    assert coefficient == pytest.approx(2.0, abs=1.0e-9), (
        f"the padding dependence of plain 3-D Ewald between L_z = {near} and "
        f"{far} A implies a slab coefficient of {coefficient:.12f}, not 2"
    )

    stored = float(require("ewald", "SLAB_CORRECTION_COEFFICIENT"))
    assert stored == pytest.approx(coefficient, abs=1.0e-9), (
        f"ewald.SLAB_CORRECTION_COEFFICIENT = {stored} against a measured "
        f"{coefficient:.12f}"
    )


def test_the_sign_of_the_correction_is_positive():
    """Tinfoil screening LOWERS the energy, so the isolated slab must come back up.

    Two separate statements, both checked: the correction the package adds is
    ``> 0``, and the uncorrected 3-D energy sits BELOW the isolated-slab answer
    at every padding.  A sign error would show up as the second failing while
    the first still passed, because ``M_n`` enters squared.

    Measured at L_z = 60 A: ``U_3D = -5.330417`` eV, correction ``+0.589034`` eV,
    ``U_slab = -4.741383`` eV, isolated reference ``-4.741383`` eV.
    """
    reference = _richardson_isolated_slab_energy()
    bulk, correction, slab = _slab_energies(
        POLAR_POS, POLAR_CHARGES, 60.0, validate=True
    )
    assert correction > 0.0, f"the slab correction is {correction:.6f} eV, not positive"
    assert bulk < reference, (
        f"plain 3-D Ewald gives {bulk:.6f} eV, which is ABOVE the isolated slab "
        f"{reference:.6f} eV; tinfoil screening must lower the energy"
    )
    assert correction == pytest.approx(0.5890341009, abs=1.0e-9)
    assert slab == pytest.approx(reference, abs=1.0e-9)

    expected = (
        2.0 * math.pi * K_E * POLAR_MZ ** 2 / (AREA * 60.0)
    )
    assert correction == pytest.approx(expected, rel=1.0e-14), (
        f"the correction {correction:.12f} eV is not 2 pi K_E M_z^2 / V = "
        f"{expected:.12f} eV"
    )


# ===========================================================================
# 4.  A non-polar slab: the correction must be inert
# ===========================================================================
def test_the_correction_vanishes_identically_for_a_non_polar_slab():
    """``M_n = 0`` gives exactly ``0.0``, and the slab energy is BIT-identical.

    ``M_n`` enters squared and nothing else in the expression survives at
    ``M_n = 0``, so this is an exact-zero claim, not a tolerance.  Checked at
    four paddings, and with point dipoles present as well: a slab of neutral
    polar molecules contributes to ``M_n`` through ``sum_i mu_i . n``, so the
    dipole channel is arranged to cancel here too and the correction must still
    be exactly zero rather than merely small.
    """
    slab_correction, slab_dipole, slab_geometry = require(
        "ewald", "slab_correction", "slab_dipole", "slab_geometry"
    )
    for height in (20.0, 40.0, 80.0, 160.0):
        cell = _padded_cell(height)
        moment = slab_dipole(
            NONPOLAR_CHARGES, None, NONPOLAR_POS, slab_geometry(cell, SLAB_PBC).normal
        )
        assert float(moment) == 0.0, (
            f"L_z = {height} A: the test bed is not non-polar, M_n = {float(moment)!r}"
        )
        correction = slab_correction(
            NONPOLAR_CHARGES, None, NONPOLAR_POS, cell, SLAB_PBC
        )
        assert float(correction) == 0.0, (
            f"L_z = {height} A: a non-polar slab picked up a correction of "
            f"{float(correction)!r} eV"
        )
        bulk, _correction, slab = _slab_energies(
            NONPOLAR_POS, NONPOLAR_CHARGES, height
        )
        assert bulk == slab, (
            f"L_z = {height} A: the slab route moved a non-polar slab by "
            f"{slab - bulk:.3e} eV"
        )

    cell = _padded_cell(60.0)
    dipoles = torch.tensor(
        [[0.0, 0.0, 0.3], [0.0, 0.0, -0.3], [0.0, 0.0, 0.3], [0.0, 0.0, -0.3]]
    )
    with_dipoles = slab_correction(
        NONPOLAR_CHARGES, dipoles, NONPOLAR_POS, cell, SLAB_PBC
    )
    assert float(with_dipoles) == 0.0, (
        f"a non-polar slab with cancelling dipoles picked up "
        f"{float(with_dipoles)!r} eV"
    )


@pytest.mark.slow
def test_a_non_polar_slab_converges_with_padding_on_its_own():
    """The contrast that makes the polar result mean something.

    With ``M_z = 0`` there is no spurious ``1/L`` term, so plain 3-D Ewald on
    this slab converges with padding EXPONENTIALLY -- through the replica
    corrugation alone, ``exp(-|G|_min h)``.  Measured change in ``U_3D`` on
    doubling the padding:

        20 -> 40 A     5.25e-05 eV
        40 -> 80 A     7.97e-12 eV
        80 -> 160 A    8.9e-15 eV

    against the polar slab's 1/L, which is still 0.11 eV at 320 A.  Same code
    path, same cell, same charges -- only ``M_z`` differs -- so this isolates
    the spurious term as a property of the cell dipole and nothing else.
    """
    energies = [
        _slab_energies(NONPOLAR_POS, NONPOLAR_CHARGES, height)[0]
        for height in (20.0, 40.0, 80.0, 160.0)
    ]
    changes = [
        abs(later - earlier) for earlier, later in zip(energies, energies[1:])
    ]
    assert changes[0] == pytest.approx(5.25e-05, rel=0.05), (
        f"20 -> 40 A moved the non-polar slab by {changes[0]:.3e} eV"
    )
    assert changes[1] < 1.0e-10, (
        f"40 -> 80 A moved the non-polar slab by {changes[1]:.3e} eV; an "
        "exponential convergence should already be at 8e-12"
    )
    assert changes[2] < 1.0e-13, (
        f"80 -> 160 A moved the non-polar slab by {changes[2]:.3e} eV"
    )
    assert changes[0] / max(changes[1], 1.0e-300) > 1.0e05, (
        "the non-polar slab is not converging exponentially with padding"
    )


# ===========================================================================
# 5.  z is not assumed
# ===========================================================================
@pytest.mark.parametrize(
    "pbc", [(True, True, False), (True, False, True), (False, True, True)]
)
def test_every_slab_axis_permutation_gives_the_same_energy(pbc):
    """``(T,T,F)``, ``(T,F,T)`` and ``(F,T,T)`` are the same physical slab.

    The Cartesian geometry is identical in all three; only WHICH ROW of the cell
    is the vacuum one changes, by a cyclic (hence orientation-preserving)
    permutation.  A route that assumed the third lattice vector is the
    non-periodic one would take the wrong cross product and project ``M_n`` onto
    the wrong direction.

    Measured: -4.741382616382648, -4.741382616382644, -4.741382616382644 --
    identical to 4e-15 eV, with ``validate=True`` so ``check_slab_geometry``,
    ``non_periodic_axis`` and ``slab_geometry`` all run on each permutation.
    """
    non_periodic_axis, slab_geometry = require(
        "ewald", "non_periodic_axis", "slab_geometry"
    )
    cell = _padded_cell(60.0, pbc)
    axis = non_periodic_axis(pbc)
    assert axis == list(pbc).index(False)
    geometry = slab_geometry(cell, pbc)
    assert float(geometry.spacing) == pytest.approx(60.0, abs=1.0e-12)
    assert float(geometry.area) == pytest.approx(AREA, abs=1.0e-12)

    _bulk, correction, slab = _slab_energies(
        POLAR_POS, POLAR_CHARGES, 60.0, pbc=pbc, validate=True
    )
    assert correction == pytest.approx(0.5890341009, abs=1.0e-9)
    assert slab == pytest.approx(PACKAGE_ISOLATED_SLAB_ENERGY, abs=1.0e-12), (
        f"pbc = {pbc}: slab energy {slab:.15f} eV against the (T,T,F) value "
        f"{PACKAGE_ISOLATED_SLAB_ENERGY:.15f} eV"
    )


def test_a_slab_whose_normal_points_along_x():
    """The complementary check: same row order, different normal DIRECTION.

    The three permutations above all leave the slab normal along Cartesian z and
    move only the lattice-vector index.  Here the whole system is rigidly
    rotated by ``(x, y, z) -> (z, x, y)`` -- positions and cell rows together --
    so ``pbc`` stays ``(T, T, F)`` while the normal becomes ``(1, 0, 0)``.  A
    projection that read a hardened z component instead of
    ``slab_geometry(...).normal`` would give ``M_n = 0`` here and lose the
    correction entirely.

    Measured: -4.741382616382648 eV, bit-identical to the unrotated cell.
    """
    slab_geometry, slab_dipole = require("ewald", "slab_geometry", "slab_dipole")

    def rotate(vectors):
        return torch.stack(
            [vectors[..., 2], vectors[..., 0], vectors[..., 1]], dim=-1
        )

    rotated_pos = rotate(POLAR_POS)
    rotated_cell = rotate(_padded_cell(60.0))
    geometry = slab_geometry(rotated_cell, SLAB_PBC)
    assert geometry.normal.tolist() == [1.0, 0.0, 0.0], (
        f"the rotated slab's normal is {geometry.normal.tolist()}, not x"
    )
    moment = float(slab_dipole(POLAR_CHARGES, None, rotated_pos, geometry.normal))
    assert moment == pytest.approx(POLAR_MZ, abs=1.0e-14)

    _bulk, _correction, slab = _slab_energies(
        POLAR_CHARGES.new_zeros(0) if False else rotated_pos,
        POLAR_CHARGES,
        60.0,
        cell=rotated_cell,
        validate=True,
    )
    assert slab == pytest.approx(PACKAGE_ISOLATED_SLAB_ENERGY, abs=1.0e-12), (
        f"the x-normal slab gives {slab:.15f} eV against {PACKAGE_ISOLATED_SLAB_ENERGY:.15f}"
    )


# ===========================================================================
# 6.  The parallel-plate capacitor the derivation is built on
# ===========================================================================
@pytest.mark.slow
def test_the_parallel_plate_capacitor_closed_form_fixes_the_two_pi():
    """The isolated slab must obey ``dU/dd = 2 pi K_E sigma^2 A``, Gauss's law.

    The ``ewald`` docstring derives the coefficient from a parallel-plate
    capacitor: ``+sigma`` at ``z = 0``, ``-sigma`` at ``z = d``, field energy
    ``(1/8 pi) (4 pi sigma)^2 A d = 2 pi sigma^2 A d``.  The plates attract with
    a constant force, so the ISOLATED energy grows linearly in the plate
    separation with slope ``2 pi K_E sigma^2 A`` -- a statement about the real
    world, not about this package.

    The bilayer's plates carry 2 e per ``64 A^2`` cell, so
    ``sigma = 0.03125 e/A^2`` and the predicted slope is **5.654727368 eV/A**.
    Point charges are not a uniform sheet, and the difference is the corrugation
    of the ``8/sqrt(2) = 5.657 A`` sublattice, which decays as
    ``exp(-2 pi d / 5.657) = exp(-1.1107 d)``.  Measured slope of
    ``ewald_slab_energy``, as a central difference over a 2 A window:

        d (A)    slope (eV/A)    deviation
         7        5.561088       -9.36e-02
         9        5.643709       -1.10e-02
        11        5.653496       -1.23e-03
        13        5.654592       -1.35e-04
        15        5.654726       -1.60e-06

    a decade per 2 A, i.e. exactly ``exp(-1.1107 d)``, converging on the closed
    form.  A coefficient of ``4 pi`` instead of ``2 pi`` in the correction would
    add ``2 pi K_E (2d)^2 / (A L_z)`` to the energy and push the slope at
    d = 15 A to 5.867 eV/A -- 0.21 eV/A out, five decades above the measured
    1.6e-06 deviation.
    """
    ewald_slab_energy, EwaldParams = require(
        "ewald", "ewald_slab_energy", "EwaldParams"
    )
    surface_charge = 2.0 / AREA
    predicted = 2.0 * math.pi * K_E * surface_charge ** 2 * AREA
    assert predicted == pytest.approx(5.654727368, abs=1.0e-8)

    height = 80.0
    cell = _padded_cell(height)
    params = EwaldParams.auto_tune_slab(cell, 7.5, SLAB_PBC, 1.0e-12)

    def energy_at(separation):
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [4.0, 4.0, 0.0],
                [0.0, 4.0, separation],
                [4.0, 0.0, separation],
            ]
        )
        edge_index, edge_shift = _neighbour_list(pos, cell, 7.5)
        return float(
            ewald_slab_energy(
                POLAR_CHARGES,
                pos,
                cell,
                edge_index,
                edge_shift,
                params,
                SLAB_PBC,
                validate=False,
            )
        )

    decay = 2.0 * math.pi / (IN_PLANE / math.sqrt(2.0))
    assert decay == pytest.approx(1.1107207345, abs=1.0e-9)

    deviations = []
    for separation in (7.0, 9.0, 11.0, 13.0, 15.0):
        slope = (energy_at(separation + 1.0) - energy_at(separation - 1.0)) / 2.0
        deviations.append(slope - predicted)

    assert abs(deviations[-1]) < 1.0e-05, (
        f"at d = 15 A the isolated-slab energy grows at "
        f"{predicted + deviations[-1]:.9f} eV/A against the capacitor's "
        f"{predicted:.9f} eV/A; a 4 pi coefficient would put it at 5.867"
    )
    for near, far in zip(deviations, deviations[1:]):
        ratio = abs(near) / abs(far)
        assert ratio == pytest.approx(math.exp(2.0 * decay), rel=0.25), (
            f"the departure from the capacitor law falls by {ratio:.2f} per 2 A; "
            f"the sublattice corrugation predicts exp(2 * {decay:.4f}) = "
            f"{math.exp(2.0 * decay):.2f}"
        )
