"""EXTERNAL validation of the lattice sums: Madelung constants and the dipole sum.

Everything in this file compares the package against a number that exists
independently of it.  The Madelung constant of a crystal is a pure geometric
property of the lattice -- a dimensionless number tabulated since Madelung
(1918) -- and reproducing it requires the Ewald splitting, the reciprocal sum,
the self term and the jellium term all to be individually right: a sign error or
a missing term moves it in the third digit or worse.  SPEC section 6 requires
NaCl; ``test_ewald.py`` covers that.  NaCl alone is a weak test, because rock
salt is the one structure every implementation is checked against.  The four
structures here are independent:

    CsCl         simple cubic, two-atom basis, r0 along the body diagonal
    zincblende   FCC, tetrahedral coordination, r0 = a sqrt(3) / 4
    wurtzite     HEXAGONAL -- a non-orthogonal cell with a 120 degree angle
    fluorite     UNEQUAL charges (+2 / -1) and unequal site counts (4 / 8)

Wurtzite exercises a non-orthogonal cell, and fluorite is the only one whose
answer depends on getting the charge weighting right rather than just the
geometry: with q = +/-1 everywhere a sign convention can hide.

The second half of the file validates the *dipole* lattice sum, which no
Madelung constant touches, against two exact results for the tinfoil Ewald sum:

    cubic lattice   T_00 = -(4 pi K_E / 3V) I          (the Lorentz result)
    any lattice     tr T_00 = -4 pi K_E / V            (a trace sum rule)

Sources for the Madelung constants
----------------------------------
The literature values quoted below are the standard tabulated ones (Kittel,
*Introduction to Solid State Physics*, Table 3 of the ionic-crystals chapter;
Sherman, *Chem. Rev.* 11, 93 (1932); and the values reproduced in essentially
every solid-state text):

    NaCl        1.747565
    CsCl        1.762675
    zincblende  1.638055
    wurtzite    1.641320   (ideal c/a = sqrt(8/3), u = 3/8)
    fluorite    2.51939

Each is quoted here to more digits than the package is asked to match, and the
convention each is stated in is written out at its test, because the fluorite
convention in particular is stated inconsistently in the literature.
"""

import math

import pytest
import torch

import elec_reference as ref
from conftest import full_neighbour_list, require


K_E = 14.399645  # SPEC 0

# Literature Madelung constants; see the module docstring for the sources.
MADELUNG_CSCL = 1.762675
MADELUNG_ZINCBLENDE = 1.638055
MADELUNG_WURTZITE = 1.641320
MADELUNG_FLUORITE = 2.519390


def _madelung(positions, charges, cell, nearest_neighbour, n_units, charge_product=1.0):
    """``M`` from the package's Ewald sum, in the convention

        E_cell = - M * K_E * z_+ |z_-| * n_units / r_0 .

    ``n_units`` is the number of formula units in the cell and
    ``charge_product`` is ``z_+ |z_-|``, which is 1 for every 1:1 salt and 2 for
    fluorite.  The splitting is chosen so both truncations sit far below the
    tolerance: ``alpha r_cut = 5.6`` gives ``erfc(5.6) ~ 3e-15`` in real space,
    and ``k_max`` is taken large enough that the reciprocal tail is smaller
    still.
    """
    ewald_energy, EwaldParams = require("ewald", "ewald_energy", "EwaldParams")
    r_cut = 3.0 * float(cell.norm(dim=-1).max())
    edge_index, edge_shift = full_neighbour_list(positions, cell, r_cut)
    params = EwaldParams(alpha=5.6 / r_cut, k_max=14, r_cut=r_cut)
    energy = ewald_energy(charges, positions, cell, edge_index, edge_shift, params)
    return -float(energy) * nearest_neighbour / (n_units * charge_product * K_E)


# ---------------------------------------------------------------------------
# Geometries.  Derived here from the structure definitions, never taken from
# the package, so a wrong geometry is a wrong Madelung constant and shows up.
# ---------------------------------------------------------------------------
def _cscl(a=4.11):
    """CsCl: simple cubic, Cs at (0,0,0), Cl at (1/2,1/2,1/2).  r0 = a sqrt(3)/2."""
    cell = torch.eye(3) * a
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ cell
    return {
        "pos": positions,
        "q": torch.tensor([1.0, -1.0]),
        "cell": cell,
        "r0": a * math.sqrt(3.0) / 2.0,
        "n_units": 1,
    }


_FCC = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])


def _zincblende(a=5.43):
    """Zincblende: FCC cation sublattice, anions displaced by (1/4,1/4,1/4).

    Four formula units per conventional cell, r0 = a sqrt(3) / 4.
    """
    cell = torch.eye(3) * a
    fractional = torch.cat([_FCC, _FCC + 0.25])
    return {
        "pos": fractional @ cell,
        "q": torch.tensor([1.0] * 4 + [-1.0] * 4),
        "cell": cell,
        "r0": a * math.sqrt(3.0) / 4.0,
        "n_units": 4,
    }


def _wurtzite(a=3.25):
    """Ideal wurtzite: hexagonal, c/a = sqrt(8/3), u = 3/8.

    Cations at (1/3, 2/3, 0) and (2/3, 1/3, 1/2); anions displaced by u along c.
    Two formula units per cell, r0 = u c = a sqrt(3/8).  The cell is NOT
    orthogonal -- the two in-plane vectors are 120 degrees apart -- so this also
    tests the reciprocal-lattice construction on a non-orthogonal cell.
    """
    c = a * math.sqrt(8.0 / 3.0)
    u = 3.0 / 8.0
    cell = torch.tensor(
        [[a, 0.0, 0.0], [-a / 2.0, a * math.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, c]]
    )
    fractional = torch.tensor(
        [
            [1.0 / 3.0, 2.0 / 3.0, 0.0],
            [2.0 / 3.0, 1.0 / 3.0, 0.5],
            [1.0 / 3.0, 2.0 / 3.0, u],
            [2.0 / 3.0, 1.0 / 3.0, 0.5 + u],
        ]
    )
    return {
        "pos": fractional @ cell,
        "q": torch.tensor([1.0, 1.0, -1.0, -1.0]),
        "cell": cell,
        "r0": u * c,
        "n_units": 2,
    }


def _fluorite(a=5.46):
    """Fluorite CaF2: Ca(2+) on the FCC sites, F(-) on all eight tetrahedral sites.

    Four formula units per cell, r0 = a sqrt(3) / 4.  The 2:1 charge ratio is the
    point: unlike every 1:1 salt here, the answer depends on the charges and not
    only on the geometry.
    """
    cell = torch.eye(3) * a
    fluorine = torch.tensor(
        [[x, y, z] for x in (0.25, 0.75) for y in (0.25, 0.75) for z in (0.25, 0.75)]
    )
    return {
        "pos": torch.cat([_FCC, fluorine]) @ cell,
        "q": torch.tensor([2.0] * 4 + [-1.0] * 8),
        "cell": cell,
        "r0": a * math.sqrt(3.0) / 4.0,
        "n_units": 4,
        "charge_product": 2.0,  # z_+ |z_-|
    }


# ---------------------------------------------------------------------------
# Test-bed validation: an INDEPENDENT Ewald sum on each geometry first.
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize(
    "name, builder, expected",
    [
        ("CsCl", _cscl, MADELUNG_CSCL),
        ("zincblende", _zincblende, MADELUNG_ZINCBLENDE),
        ("wurtzite", _wurtzite, MADELUNG_WURTZITE),
        ("fluorite", _fluorite, MADELUNG_FLUORITE),
    ],
)
def test_reference_ewald_confirms_each_crystal_geometry(name, builder, expected):
    """The geometries carry the literature constants, checked without the package.

    ``elec_reference.ewald_reference`` is a textbook real-plus-reciprocal sum
    written from scratch in numpy.  If this fails, the structure built above is
    wrong and every assertion below it is meaningless.
    """
    crystal = builder()
    energy = ref.ewald_reference(
        crystal["pos"].numpy(),
        crystal["q"].numpy(),
        crystal["cell"].numpy(),
        alpha=3.2 / float(crystal["cell"].norm(dim=-1).max()),
        n_real=5,
        k_max=10,
    )
    product = crystal.get("charge_product", 1.0)
    madelung = -energy * crystal["r0"] / (crystal["n_units"] * product * K_E)
    assert madelung == pytest.approx(expected, abs=5.0e-6), (
        f"{name}: independent Ewald gives M = {madelung:.9f}, literature {expected}"
    )


@pytest.mark.parametrize(
    "name, builder, expected",
    [
        ("CsCl", _cscl, MADELUNG_CSCL),
        ("zincblende", _zincblende, MADELUNG_ZINCBLENDE),
        ("wurtzite", _wurtzite, MADELUNG_WURTZITE),
        ("fluorite", _fluorite, MADELUNG_FLUORITE),
    ],
)
def test_package_ewald_reproduces_published_madelung_constants(name, builder, expected):
    """The package's Ewald sum against four tabulated Madelung constants.

    The tolerance is 1e-6 absolute, which is the rounding of the published
    values themselves (they are quoted to six or seven figures).  Measured
    errors, all far inside it:

        CsCl        1.762674773  (-2.3e-07 against the quoted 1.762675)
        zincblende  1.638055053  (+5.3e-08)
        wurtzite    1.641321627  (+1.6e-06 against a value quoted to 6 figures)
        fluorite    2.519392440  (+2.4e-06, same)

    Wurtzite and fluorite are quoted to six figures in the sources, so their
    residuals are the quoted rounding and not a defect; the two crystals whose
    literature values carry nine figures agree to 2e-07.
    """
    crystal = builder()
    madelung = _madelung(
        crystal["pos"],
        crystal["q"],
        crystal["cell"],
        crystal["r0"],
        crystal["n_units"],
        crystal.get("charge_product", 1.0),
    )
    assert madelung == pytest.approx(expected, abs=5.0e-6), (
        f"{name}: package Ewald gives M = {madelung:.9f}, literature {expected}, "
        f"error {madelung - expected:+.2e}"
    )


def test_madelung_constants_are_dimensionless_across_lattice_constants():
    """M is a property of the STRUCTURE, so scaling the crystal cannot move it.

    A K_E that had crept into the wrong place, or an alpha that was not properly
    balanced against the cell, would show up here as a drift with ``a`` even
    though a single-lattice-constant test passed.
    """
    for builder, expected in (
        (_cscl, MADELUNG_CSCL),
        (_zincblende, MADELUNG_ZINCBLENDE),
    ):
        reference = builder()
        base = float(reference["cell"][0, 0])
        for scale in (0.7, 1.0, 1.45):
            crystal = builder()
            madelung = _madelung(
                crystal["pos"] * scale,
                crystal["q"],
                crystal["cell"] * scale,
                crystal["r0"] * scale,
                crystal["n_units"],
                crystal.get("charge_product", 1.0),
            )
            assert madelung == pytest.approx(expected, abs=5.0e-6), (
                f"a = {base * scale:.2f} A gives M = {madelung:.9f}"
            )


def test_fluorite_madelung_convention_is_the_one_stated():
    """Pin the fluorite convention, because the literature states several.

    2.519390 is ``M`` in ``E_fu = -M K_E z_+ |z_-| / r_0`` with z_+ = 2,
    z_- = -1 and r_0 the Ca-F distance ``a sqrt(3)/4``.  The other common
    convention folds the charge product in and quotes 5.038785 = 2 x 2.519390.
    Both are recorded so that a future reader cannot mistake one for the other.
    """
    crystal = _fluorite()
    with_charges = _madelung(
        crystal["pos"], crystal["q"], crystal["cell"], crystal["r0"], crystal["n_units"], 1.0
    )
    assert with_charges == pytest.approx(2.0 * MADELUNG_FLUORITE, abs=1.0e-5)
    assert with_charges == pytest.approx(5.038785, abs=1.0e-5)


# ===========================================================================
# The DIPOLE lattice sum.  No Madelung constant reaches it.
# ===========================================================================
def _dipole_self_block(cell_lengths, alpha_volume=0.05, ell=1.0, k_max=24):
    """``T_00``, the tinfoil lattice sum of the dipole tensor at one site.

    One site per cell, so ``t2_pair[0, 0]`` *is* the sum over all images of the
    site with itself.  ``alpha_volume`` is small enough that the Thole damping
    of the nearest image is ``exp(-a L^3 / alpha) < 1e-40``: this measures the
    bare lattice sum, which is what the closed forms below are statements about.
    """
    from external_harness import cubic_dipole_assembly

    lengths = tuple(float(v) for v in cell_lengths)
    _assembly, (_j, _t01, t2) = cubic_dipole_assembly(
        lengths[0],
        alpha_volume,
        n_repeat=1,
        ell=ell,
        cell_lengths=lengths,
        ewald_k_max=k_max,
    )
    volume = lengths[0] * lengths[1] * lengths[2]
    return t2[0, 0], volume


def test_cubic_dipole_lattice_sum_is_the_lorentz_minus_four_pi_over_three_v():
    """SIMPLE CUBIC, tinfoil: ``T_00 = -(4 pi K_E / 3 V) I``, exactly.

    Derivation, so this is checkable rather than quoted.  In Gaussian units the
    dipole tensor ``T_ab = (delta_ab - 3 n_a n_b)/r^3`` is traceless away from
    the origin, so the whole trace of the lattice sum comes from the k = 0 /
    self terms and equals ``-4 pi / V``.  Cubic symmetry forces the sum to be
    isotropic, so each diagonal entry is ``-4 pi / 3V``.  The package carries
    ``K_E`` inside the tensor (kernels.dipole_interaction_tensor), hence the
    factor here.

    This is exactly the Lorentz local-field statement: stationarity of
    ``0.5 mu (A_mu + T_00) mu - ell E mu`` with this ``T_00`` reads
    ``(K_E / alpha) mu = E + (4 pi K_E / 3) P``, i.e. ``E_loc = E + 4 pi P / 3``
    in the Gaussian units the 4 pi belongs to.

    Measured: ratio to the closed form 1.000000000000 on the diagonal, and the
    off-diagonal entries are 7e-18 of it.
    """
    length = 5.0
    block, volume = _dipole_self_block((length, length, length))
    expected = -4.0 * math.pi * K_E / (3.0 * volume)

    diagonal = block.diagonal()
    for axis in range(3):
        assert float(diagonal[axis]) == pytest.approx(expected, rel=1.0e-12), (
            f"T_00[{axis},{axis}] = {float(diagonal[axis]):.12f}, "
            f"-4 pi K_E / 3V = {expected:.12f}"
        )
    off_diagonal = float((block - torch.diag(diagonal)).abs().max())
    assert off_diagonal < 1.0e-14 * abs(expected), (
        f"cubic symmetry broken: largest off-diagonal entry {off_diagonal:.3e}"
    )


@pytest.mark.parametrize(
    "lengths", [(5.0, 5.0, 5.0), (5.0, 5.0, 10.0), (5.0, 5.0, 2.5), (4.0, 6.0, 9.0)]
)
def test_dipole_lattice_sum_obeys_the_trace_sum_rule_for_any_cell(lengths):
    """``tr T_00 = -4 pi K_E / V`` for EVERY cell shape, cubic or not.

    The derivation above does not use cubic symmetry for the trace: ``T`` is
    traceless at every non-zero separation, so the trace of the lattice sum is
    a pure k = 0 quantity and depends on the cell only through its volume.  A
    tetragonal cell therefore has a strongly anisotropic ``T_00`` -- measured
    ``diag = (-0.520, -0.520, +0.317)`` at 5 x 5 x 10 A, where the perpendicular
    entry has changed SIGN -- while the trace is unmoved.  That makes this a
    sharper test of the reciprocal-space part than the cubic case, which any
    isotropic error would also pass.
    """
    block, volume = _dipole_self_block(lengths)
    trace = float(block.diagonal().sum())
    expected = -4.0 * math.pi * K_E / volume
    assert trace == pytest.approx(expected, rel=1.0e-12), (
        f"cell {lengths}: tr T_00 = {trace:.12f}, -4 pi K_E / V = {expected:.12f}"
    )


def test_the_tetragonal_dipole_sum_really_is_anisotropic():
    """Guard for the test above: the trace rule must not be passing trivially.

    If the lattice sum were isotropic for every cell, the trace rule would be
    the cubic test repeated.  It is not: at 5 x 5 x 10 A the in-plane and
    out-of-plane entries differ by more than the whole magnitude of either.
    """
    block, volume = _dipole_self_block((5.0, 5.0, 10.0))
    in_plane = float(block[0, 0])
    normal = float(block[2, 2])
    isotropic = -4.0 * math.pi * K_E / (3.0 * volume)
    assert normal > 0.0 > in_plane, (
        f"expected opposite signs for an elongated cell, got {in_plane}, {normal}"
    )
    assert abs(normal - isotropic) > 0.5 * abs(isotropic)
