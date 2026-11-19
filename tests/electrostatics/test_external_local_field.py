"""EXTERNAL validation of the local field: Clausius-Mossotti and Lorentz.

This is the single most decisive external test in the suite, and it is here for
a reason the rest of the regression suite cannot supply.  Every other file
checks INTERNAL consistency -- that the code implements its own spec, that
gradients match finite differences, that symmetries hold.  A model can be
flawlessly self-consistent and still be wrong physics.  The Clausius-Mossotti
relation is not a property of this package: it is a nineteenth-century result
about a cubic lattice of polarizable points, and reproducing it requires the
Ewald dipole sum, the Thole damping, the ``K_E`` unit conversion and the
self-consistent induction to be *simultaneously* right.  A sign error in the
reciprocal-space dipole term, a factor of ``4 pi``, a ``K_E`` in the wrong
place, or a self-term double count all break it, and none of them is visible in
a single-component test.

The reference
-------------
For a cubic lattice of point-polarizable sites of atomic polarizability
``alpha_at`` (in A^3) at number density ``n`` (in A^-3),

    (eps - 1) / (eps + 2) = 4 pi n alpha_at / 3                          (CM)

with ``eps = 1 + 4 pi chi`` and ``chi = P / E``.  Clausius (1879) / Mossotti
(1850); the standard derivation is in Kittel, *Introduction to Solid State
Physics*, and in Jackson, *Classical Electrodynamics*, section 4.5, where it is
obtained from the Lorentz local field

    E_loc = E + 4 pi P / 3                                           (Lorentz)

which holds for a cubic lattice because the dipole lattice sum at a site is
isotropic there, ``T_00 = -(4 pi / 3V) I``.

Derivation of what the package must return, so that "CM holds" is checkable
--------------------------------------------------------------------------
For a one-site cubic cell of volume ``V``, all sites equivalent by translation,
the moment energy of SPEC 1.5 is exactly

    U(mu_hat) = 0.5 mu_hat . (A_mu + T_00) . mu_hat

with ``A_mu = K_E ell^2 / alpha_at`` (the isotropic inverse of the atomic
polarizability, external_harness.isotropic_A_mu) and ``T_00`` the tinfoil
lattice sum of the dipole tensor.  ``test_external_lattice_sums.py`` has already
established ``T_00 = -(4 pi K_E / 3V) I`` against its own closed form, so

    chi = P / E = (1/K_E) alpha_at / (V - 4 pi alpha_at / 3)
    eps - 1 = 4 pi K_E chi = 4 pi n alpha_at / (1 - 4 pi n alpha_at / 3)

and with ``x = 4 pi n alpha_at / 3`` that is ``eps - 1 = 3x/(1-x)``,
``eps + 2 = 3/(1-x)``, hence ``(eps-1)/(eps+2) = x``.  Exactly CM.  The factor
``K_E`` appears because this package works in eV / A / e rather than Gaussian
units, so the dimensionless susceptibility is ``K_E chi`` (SPEC 0).

Damping, and why the sweep is run twice
---------------------------------------
CM is a statement about BARE point dipoles.  The package damps (SPEC 1.6), and
the damping is keyed on ``v = a r^3 / alpha`` at the nearest image.  Both
regimes are tested and they say different things:

* at the production damping ``a = 0.39`` and a dilute lattice (``v >> 1``) the
  damping factors are 1.0 to the last bit, and CM must hold to round-off -- this
  is the test of the production operator;
* with the damping switched off exactly (``damping_alpha_volume = 1e-12``, see
  the external_harness docstring) the density can be pushed to within 5 % of the
  polarization catastrophe, ``eps = 58``, where CM is a strong constraint rather
  than a first-order one.

And the departure at high physical density is measured rather than tolerated:
it is the damping doing its job, and it goes the right way (down).
"""

import math

import pytest
import torch

from external_harness import cubic_dipole_assembly, uniform_dipole_moment
from external_harness import uniform_dipole_susceptibility


K_E = 14.399645  # SPEC 0


def _dielectric(lattice_constant, alpha_volume, *, damping_alpha=None,
                cell_lengths=None, k_max=28):
    """``(eps, n, alpha)`` of a one-site lattice, from the package's own energy.

    ``eps = 1 + 4 pi K_E chi`` with ``chi`` from
    ``external_harness.uniform_dipole_susceptibility``, which minimises
    ``ElectrostaticAssembly.energy`` -- the energy the model actually reports --
    over the uniform moment.  ``k_max = 28`` puts the reciprocal-space
    truncation of the dipole sum far below the tolerances used here.
    """
    assembly, _blocks = cubic_dipole_assembly(
        lattice_constant,
        alpha_volume,
        n_repeat=1,
        cell_lengths=cell_lengths,
        ewald_k_max=k_max,
        damping_alpha_volume=damping_alpha,
    )
    susceptibility, _curvature = uniform_dipole_susceptibility(assembly)
    volume = float(torch.det(assembly.cell).abs())
    return 1.0 + 4.0 * math.pi * K_E * susceptibility, 1.0 / volume, assembly


# ===========================================================================
# 1.  Clausius-Mossotti
# ===========================================================================
@pytest.mark.parametrize("lattice_constant", [2.0, 2.5, 3.0, 4.0, 6.0])
def test_clausius_mossotti_holds_across_densities_at_production_damping(
    lattice_constant,
):
    """CM at ``thole_a = 0.39``, fixed ``alpha_at``, five number densities.

    The relation has to hold across densities, not at one tuned point -- a
    wrong constant folded into the dipole lattice sum would still let a
    single-density comparison be matched by choosing ``alpha_at``.  The density
    spans a factor of 27 here (a = 2 A to 6 A).

    ``alpha_at = 0.05 A^3`` keeps the nearest-image Thole argument
    ``v = 0.39 a^3 / alpha`` between 62 (a = 2 A) and 1685 (a = 6 A), so
    ``lambda_3 = lambda_5 = 1.0`` exactly and this is the bare lattice sum that
    CM is about.  Measured relative error on ``(eps-1)/(eps+2)`` versus
    ``4 pi n alpha / 3``, worst case over the five:  7.2e-15.
    """
    alpha_volume = 0.05
    eps, density, _assembly = _dielectric(lattice_constant, alpha_volume)

    left = (eps - 1.0) / (eps + 2.0)
    right = 4.0 * math.pi * density * alpha_volume / 3.0
    assert left == pytest.approx(right, rel=1.0e-12), (
        f"a = {lattice_constant} A, n = {density:.6f} A^-3, eps = {eps:.9f}: "
        f"(eps-1)/(eps+2) = {left:.12f} against 4 pi n alpha / 3 = {right:.12f}"
    )


# ``x = 4 pi n alpha / 3``, the right-hand side of CM.  ``x -> 1`` is the
# polarization catastrophe (``eps -> inf``), so 0.95 is as hard as this
# reference gets: eps = 58.
@pytest.mark.parametrize(
    "target_x, expected_eps",
    [
        (0.10, 1.333333333333),   # eps = 1 + 3x/(1-x), evaluated exactly
        (0.30, 2.285714285714),
        (0.50, 4.0),
        (0.70, 8.0),
        (0.90, 28.0),
        (0.95, 58.0),
    ],
)
def test_clausius_mossotti_holds_up_to_the_polarization_catastrophe(
    target_x, expected_eps
):
    """Undamped point dipoles, ``eps`` from 1.33 to 58.  CM to round-off.

    ``eps = 1 + 3x/(1 - x)`` is the same relation rearranged, and it is quoted
    here as a second, independent form of the check: at ``x = 0.95`` a 1 %
    error in the lattice sum moves ``eps`` by 20, so this is a far sharper
    constraint than the dilute sweep above, where ``eps - 1`` is 0.02.

    The damping is switched off exactly rather than made small (see the
    external_harness module docstring): at ``x = 0.95`` the production damping
    would be a 30 % effect and CM would not apply.

    Measured: relative error on ``(eps-1)/(eps+2)`` at most 6.2e-16 across the
    six, and ``eps`` itself matches the closed form to 1e-12 relative.
    """
    lattice_constant = 3.0
    volume = lattice_constant**3
    alpha_volume = target_x * 3.0 / (4.0 * math.pi) * volume

    eps, density, _assembly = _dielectric(
        lattice_constant, alpha_volume, damping_alpha=1.0e-12
    )

    left = (eps - 1.0) / (eps + 2.0)
    right = 4.0 * math.pi * density * alpha_volume / 3.0
    assert right == pytest.approx(target_x, rel=1.0e-14)  # the sweep is what it says
    assert left == pytest.approx(right, rel=1.0e-12), (
        f"x = {target_x}: (eps-1)/(eps+2) = {left:.14f}, CM says {right:.14f}"
    )
    assert eps == pytest.approx(expected_eps, rel=1.0e-12), (
        f"x = {target_x}: eps = {eps:.10f}, 1 + 3x/(1-x) = {expected_eps}"
    )


def test_the_dielectric_constant_sweep_is_not_trivially_near_one():
    """Guard: the sweep above must actually reach a large ``eps``.

    CM is nearly unfalsifiable at ``eps -> 1``, where both sides collapse to
    ``4 pi n alpha / 3`` at first order regardless of what the lattice sum
    did.  This asserts that the parameterisation really does drive the model
    into the strongly coupled regime, so the test above is doing work.
    """
    lattice_constant = 3.0
    alpha_volume = 0.95 * 3.0 / (4.0 * math.pi) * lattice_constant**3
    eps, _n, _assembly = _dielectric(
        lattice_constant, alpha_volume, damping_alpha=1.0e-12
    )
    assert eps > 50.0, f"eps = {eps}, expected the sweep to reach eps ~ 58"


@pytest.mark.parametrize(
    "lattice_constant, max_relative_departure",
    [
        # v = 0.39 a^3 / alpha at the nearest image, alpha = 0.05 A^3.
        (1.20, 1.0e-5),   # v = 13.5  -- measured -3.3e-06
        (1.00, 3.0e-3),   # v =  7.8  -- measured -9.6e-04
        (0.90, 2.0e-2),   # v =  5.7  -- measured -7.9e-03
    ],
)
def test_thole_damping_bends_clausius_mossotti_down_at_high_density(
    lattice_constant, max_relative_departure
):
    """Where the damping bites, CM must fail -- DOWNWARD, and by this much.

    This is not a tolerance being loosened; it is the measurement of a physical
    effect.  Thole damping screens the dipole-dipole interaction at short range
    (SPEC 1.6), so the induced dipoles help each other LESS than bare point
    dipoles do and the dielectric constant must come out BELOW Clausius-
    Mossotti.  The sign is the content of the test: a damping implemented with
    the wrong sign, or applied to the wrong factor, would push it up.

    Measured departures, relative, at alpha = 0.05 A^3:

        a = 1.20 A   v = 13.5   -3.278e-06
        a = 1.00 A   v =  7.8   -9.579e-04
        a = 0.90 A   v =  5.7   -7.876e-03

    which is the ``exp(-v)`` tail of ``lambda_3`` arriving exactly where
    ``kernels.thole_damping`` says it should.
    """
    alpha_volume = 0.05
    eps, density, _assembly = _dielectric(lattice_constant, alpha_volume)
    left = (eps - 1.0) / (eps + 2.0)
    right = 4.0 * math.pi * density * alpha_volume / 3.0
    departure = (left - right) / right

    assert departure < 0.0, (
        f"a = {lattice_constant} A: damping moved the dielectric constant UP "
        f"({departure:+.3e}); screening can only reduce the mutual induction"
    )
    assert abs(departure) < max_relative_departure, (
        f"a = {lattice_constant} A: departure {departure:+.3e} exceeds the "
        f"measured {max_relative_departure:.0e}"
    )


def test_undamped_and_damped_agree_where_the_damping_is_off():
    """Cross-check of the damping switch itself, so it cannot be hiding a bug.

    ``damping_alpha_volume = 1e-12`` is the route the high-density CM test uses,
    and it has to be the *same operator* as the production one wherever the
    production damping is already saturated.  At a = 3 A, alpha = 0.05 A^3 the
    nearest-image argument is v = 211, so the two must agree bit for bit.
    """
    damped, _n, _a = _dielectric(3.0, 0.05)
    undamped, _n2, _a2 = _dielectric(3.0, 0.05, damping_alpha=1.0e-12)
    assert damped == pytest.approx(undamped, rel=1.0e-15), (
        f"damped eps = {damped:.15f}, undamped eps = {undamped:.15f}"
    )


# ===========================================================================
# 2.  The Lorentz local field -- the microscopic statement underneath CM
# ===========================================================================
@pytest.mark.parametrize("lattice_constant, alpha_volume",
                         [(3.0, 0.05), (4.0, 0.30), (2.5, 0.02)])
def test_lorentz_local_field_is_the_applied_field_plus_four_pi_p_over_three(
    lattice_constant, alpha_volume
):
    """``E_loc = E + 4 pi P / 3`` on the SOLVED dipole, not on the lattice sum.

    ``test_external_lattice_sums.py`` already checks the tensor identity
    ``T_00 = -(4 pi K_E / 3V) I``.  This is the operational form of the same
    statement and it is a different assertion: it goes through the induced
    moment the model actually settles on, so it also tests that ``A_mu`` is the
    inverse of the atomic polarizability the caller asked for, and that the
    ``K_E`` converting a volume in A^3 to a response in e^2 A^2 / eV is applied
    once and in the right direction (SPEC 0's explicit trap).

    Construction: solve the lattice in a uniform field ``E``, read the per-site
    moment ``mu``, and ask what field an ISOLATED atom of this polarizability
    would have needed to acquire it, ``E_loc = K_E mu / alpha_at``.  Lorentz
    says that is ``E + 4 pi K_E P / 3`` with ``P = mu / V``.  Measured
    agreement: exact to 2.2e-16 relative on all three lattices.
    """
    field = 1.0
    _eps, _density, assembly = _dielectric(lattice_constant, alpha_volume)
    moment, polarization = uniform_dipole_moment(assembly, field)

    local_from_model = K_E * moment / alpha_volume
    local_from_lorentz = field + 4.0 * math.pi * K_E * polarization / 3.0

    assert local_from_model == pytest.approx(local_from_lorentz, rel=1.0e-12), (
        f"a = {lattice_constant} A, alpha = {alpha_volume} A^3: "
        f"K_E mu / alpha = {local_from_model:.14f} but "
        f"E + 4 pi K_E P / 3 = {local_from_lorentz:.14f}"
    )


@pytest.mark.parametrize("lattice_constant, alpha_volume",
                         [(3.0, 0.05), (4.0, 0.30), (2.5, 0.02)])
def test_the_lorentz_enhancement_factor_is_eps_plus_two_over_three(
    lattice_constant, alpha_volume
):
    """``E_loc / E = (eps + 2) / 3`` -- the published form of the same relation.

    Jackson section 4.5.  It ties the MICROSCOPIC quantity (the field at a site)
    to the MACROSCOPIC one (the dielectric constant of the same lattice), so it
    fails if either the local-field sum or the susceptibility is wrong, and it
    cannot be passed by an error common to both.  Measured: exact to 1.1e-16.
    """
    field = 1.0
    eps, _density, assembly = _dielectric(lattice_constant, alpha_volume)
    moment, _polarization = uniform_dipole_moment(assembly, field)
    local = K_E * moment / alpha_volume

    assert local / field == pytest.approx((eps + 2.0) / 3.0, rel=1.0e-12), (
        f"E_loc/E = {local / field:.14f}, (eps+2)/3 = {(eps + 2.0) / 3.0:.14f}, "
        f"eps = {eps:.10f}"
    )


def test_the_lorentz_factor_does_not_hold_for_a_non_cubic_lattice():
    """Guard: ``4 pi P / 3`` is a CUBIC result, and the package must know it.

    The derivation needs ``T_00`` isotropic, which only cubic symmetry gives.
    On a 4 x 4 x 8 A tetragonal cell of the same sites the true local field is
    measurably different -- measured ``K_E mu / alpha = 0.9873`` against
    ``E + 4 pi K_E P / 3 = 1.0097``, a 2.2 % gap that lands on the OTHER SIDE of
    the applied field (the depolarizing lattice sum has changed sign along the
    long axis, exactly as ``test_external_lattice_sums`` measures for the
    tetragonal ``T_00``).

    Without this the cubic tests above could be passing on an implementation
    that ignored the cell shape and hard-coded ``-4 pi / 3V``.
    """
    assembly, _blocks = cubic_dipole_assembly(
        4.0, 0.30, n_repeat=1, cell_lengths=(4.0, 4.0, 8.0), ewald_k_max=28
    )
    moment, polarization = uniform_dipole_moment(assembly, 1.0)
    local_from_model = K_E * moment / 0.30
    local_from_lorentz = 1.0 + 4.0 * math.pi * K_E * polarization / 3.0

    assert local_from_model < 1.0 < local_from_lorentz, (
        "a tetragonal cell must depolarize along its long axis: got "
        f"{local_from_model:.6f} and {local_from_lorentz:.6f}"
    )
    relative_gap = abs(local_from_model / local_from_lorentz - 1.0)
    assert relative_gap > 0.01, (
        f"the tetragonal local field is within {relative_gap:.2e} of the cubic "
        "Lorentz value, so the cubic tests are not testing the cell shape"
    )
