"""EXTERNAL validation of the dipole relay: closed forms, a conductor, molecules.

Four independent references, in increasing order of how much of the package they
touch:

1. ``energy._direct_pair_blocks``, the private entry point ``external_harness``
   drives, reconstructed from the PUBLIC ``kernels`` API.  Not physics -- a pin,
   so the rest of this file is known to be running on the production operator.
2. the **Applequist two-atom relay**, which has a closed form in every
   orientation and a genuine divergence, so it tests the propagator including
   its sign, its ``3 r_a r_b / r^5`` anisotropy and its self-consistency.
3. a **discretised conducting sphere**, which must give ``alpha = a^3``.  This
   is the only test here that exercises the CHARGE-TRANSFER sector against an
   external number, and it tests the whole field -> dipole -> volume chain
   including the ``K_E`` of SPEC 0.  It is scale invariant, so a unit error that
   a single-size test would absorb into the discretisation shows up.
4. **published molecular polarizabilities** of water and benzene from literature
   ATOMIC polarizabilities.  The package predicts its parameters from learned
   features, so there is no end-to-end number without a checkpoint; what CAN be
   tested is the physics engine, handed the atomic values the Thole model was
   fitted to.

Sources for every number used below
-----------------------------------
ATOMIC polarizabilities (A^3), and the damping they belong to:

* Thole, *Chem. Phys.* **59**, 341 (1981) -- the damped-interaction model.  The
  smeared-charge form this package implements is ``rho(u) = (3a/4 pi)
  exp(-a u^3)``, ``u = r / (alpha_i alpha_j)^(1/6)``, for which Thole's fitted
  screening parameter is ``a = 0.572``.
* Ren and Ponder, *J. Phys. Chem. B* **107**, 5933 (2003), and Ponder et al.,
  *J. Phys. Chem. B* **114**, 2549 (2010) (AMOEBA): the SAME functional form
  with a universal ``a = 0.39`` -- which is this package's default
  (``kernels.DEFAULT_THOLE_A``, SPEC 1.6) -- and Thole's atomic values
  ``alpha(C) = 1.334``, ``alpha(H) = 0.496``, ``alpha(O) = 0.837``,
  ``alpha(N) = 1.073``.  AMOEBA's own documentation records that 0.39 "leads to
  stronger damping and less short-range polarization than the original value of
  0.572 suggested by Thole"; that difference is measured below.
* van Duijnen and Swart, *J. Phys. Chem. A* **102**, 2399 (1998), refit Thole
  with a DIFFERENT smearing, ``rho(u) propto exp(-a u)`` with ``a = 2.1304`` and
  ``alpha(C) = 1.405``, ``alpha(H) = 0.514``, ``alpha(O) = 0.862``.  That form
  is **not** what this package implements and its ``a`` is not comparable with
  this one's -- 2.1304 would be refused by ``kernels._validate_thole_a``, and
  correctly so, because the SPEC 1.6 bound ``sup_r alpha|T| = a < 1`` is a
  statement about the ``u^3`` form.  It is carried here only in the independent
  reference relay, to locate the benzene discrepancy.

MOLECULAR polarizabilities, experimental:

* water, mean ``alpha = 1.45 A^3`` (the value quoted in the task statement and
  the standard gas-phase number, e.g. CRC Handbook of Chemistry and Physics,
  "Dipole polarizabilities of atoms and molecules"; Murphy, *J. Chem. Phys.*
  **67**, 5877 (1977) gives 1.45);
* benzene, in-plane ``alpha_xx = alpha_yy = 12.3 A^3``, out-of-plane
  ``alpha_zz = 6.3-6.7 A^3``, mean ``10.3 A^3`` (the task statement's values;
  they are the standard Kerr-effect / depolarised-Rayleigh tensor, and the mean
  is quoted as 10.32 A^3 in the CRC tables).  The out-of-plane component is
  given as a RANGE in the source, so it is used as a range here.

GEOMETRIES, experimental, derived in this file and not taken from the package:

* water r(OH) = 0.9572 A, angle(HOH) = 104.52 deg (the ``water_molecule``
  fixture's geometry, restated locally so this file is self-contained);
* benzene D6h, r(CC) = 1.397 A, r(CH) = 1.087 A.

Which response route, and why
-----------------------------
Everything in this file uses ``polarizability(route="minimiser")``.  That is not
a convenience: Applequist's closed forms, Clausius-Mossotti, the conducting
sphere and the Thole molecular polarizabilities are all statements about the
EXACT stationary point of the quadratic form, which is the ``k -> infinity``
limit of SPEC 8.  The fixed-k route answers a different (and, for the model's
forces, the more important) question, and on a 12-atom relay at k = 6 it has not
converged.  ``test_observables.py`` owns the comparison between the two.
"""

import math

import numpy as np
import pytest
import torch

import external_harness as harness
from conftest import require
from external_harness import fibonacci_sphere, open_system


K_E = 14.399645  # SPEC 0

# --- Thole / AMOEBA atomic polarizability volumes, A^3.  Sources in the module
# --- docstring.  These belong to the exp(-a u^3) smearing this package uses.
ALPHA_C_THOLE = 1.334
ALPHA_H_THOLE = 0.496
ALPHA_O_THOLE = 0.837

# --- van Duijnen & Swart, for the exp(-a u) smearing the package does NOT use.
ALPHA_C_VDS = 1.405
ALPHA_H_VDS = 0.514
ALPHA_O_VDS = 0.862
THOLE_A_VDS = 2.1304

THOLE_A_AMOEBA = 0.39   # this package's default, SPEC 1.6
THOLE_A_ORIGINAL = 0.572  # Thole 1981, same functional form

# --- experimental molecular polarizabilities, A^3.  Sources in the docstring.
WATER_ALPHA_EXPERIMENT = 1.45
BENZENE_ALPHA_MEAN_EXPERIMENT = 10.3
BENZENE_ALPHA_IN_PLANE_EXPERIMENT = 12.3
BENZENE_ALPHA_OUT_OF_PLANE_RANGE = (6.3, 6.7)


def _polarizability(positions, alpha_volume, *, thole_a=THOLE_A_AMOEBA,
                    damping_alpha=None, **kwargs):
    """``alpha`` in A^3 of an open-boundary system, exact-minimiser route."""
    polarizability = require("observables", "polarizability")
    assembly, solve_result = open_system(
        positions,
        alpha_volume,
        thole_a=thole_a,
        damping_alpha_volume=damping_alpha,
        **kwargs,
    )
    return polarizability(
        solve_result, assembly, route="minimiser", as_volume=True
    ), assembly


def _dipole_block(assembly):
    """The ``A_mu + T_Thole`` block of the dense operator, ``[3n, 3n]``."""
    hessian = assembly.dense_hessian()
    offset = assembly.n_bonds + assembly.n_atoms
    return hessian[offset:, offset:]


# ===========================================================================
# 0.  Pin the private entry point the harness drives.
# ===========================================================================
def test_direct_pair_blocks_match_a_reconstruction_from_the_public_kernels():
    """``energy._direct_pair_blocks`` against the public ``kernels`` API.

    ``external_harness`` builds its assemblies through this private function so
    that the external-physics tests run on the production operator rather than
    on a re-derivation of it.  That is only safe if the private entry point is
    itself pinned -- otherwise a change to it would silently redefine what every
    test in this file is measuring.  Rebuilt here from ``combine_sigma``,
    ``coulomb_kernel``, ``thole_damping``, ``charge_dipole_tensor`` and
    ``dipole_interaction_tensor``, all public, with the reduced-dipole ``ell``
    factors of SPEC 1.5 applied by hand.

    Measured agreement: exact, 0.0 on all three blocks.
    """
    combine_sigma, coulomb_kernel, thole_damping = require(
        "kernels", "combine_sigma", "coulomb_kernel", "thole_damping"
    )
    charge_dipole_tensor, dipole_interaction_tensor = require(
        "kernels", "charge_dipole_tensor", "dipole_interaction_tensor"
    )
    direct_pair_blocks = require("energy", "_direct_pair_blocks")

    positions = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [1.43, 0.11, -0.22],
            [2.31, 1.30, 0.41],
            [-0.72, 1.09, 0.63],
        ]
    )
    alpha_volume = torch.tensor([0.30, 1.60, 11.0, 0.90])
    assembly, _result = open_system(positions, alpha_volume, ell=0.8)
    params = assembly.params
    thole_a = assembly.config.thole_a

    j_pair, t01_pair, t2_pair = direct_pair_blocks(
        positions, params, assembly.alpha_volume, thole_a
    )

    n_atoms = positions.shape[0]
    vector = positions.unsqueeze(1) - positions.unsqueeze(0)
    distance = vector.norm(dim=-1)
    off_diagonal = 1.0 - torch.eye(n_atoms, dtype=torch.float64)
    safe = torch.where(distance > 0.0, distance, torch.ones_like(distance))

    sigma_pair = combine_sigma(params.sigma.unsqueeze(1), params.sigma.unsqueeze(0))
    expected_j = coulomb_kernel(safe, sigma_pair) * off_diagonal
    # Same formula, different evaluation order: agreement to a few ulps, not
    # bitwise (torch 2.11 differs from 2.2 by exactly one ulp, 8.9e-16 at 6.6).
    torch.testing.assert_close(
        j_pair, expected_j, rtol=4.0 * torch.finfo(torch.float64).eps, atol=0.0
    )

    lam3, lam5 = thole_damping(
        safe,
        assembly.alpha_volume.unsqueeze(1),
        assembly.alpha_volume.unsqueeze(0),
        thole_a,
    )
    ell = params.ell
    expected_t01 = (
        charge_dipole_tensor(vector, safe, lam3)
        * ell.view(1, n_atoms, 1)
        * off_diagonal.unsqueeze(-1)
    )
    torch.testing.assert_close(
        t01_pair, expected_t01, rtol=4.0 * torch.finfo(torch.float64).eps,
        atol=4.0 * torch.finfo(torch.float64).eps * float(expected_t01.abs().max()),
    )

    expected_t2 = (
        dipole_interaction_tensor(vector, safe, lam3, lam5)
        * (ell.view(n_atoms, 1, 1, 1) * ell.view(1, n_atoms, 1, 1))
        * off_diagonal.view(n_atoms, n_atoms, 1, 1)
    )
    torch.testing.assert_close(
        t2_pair, expected_t2, rtol=4.0 * torch.finfo(torch.float64).eps,
        atol=4.0 * torch.finfo(torch.float64).eps * float(expected_t2.abs().max()),
    )


# ===========================================================================
# 1.  The Applequist two-atom relay.
# ===========================================================================
#
# Two isotropic point-polarizable sites of polarizability ``alpha`` separated by
# ``r`` along z.  Applequist, Carl and Fung, *J. Am. Chem. Soc.* **94**, 2952
# (1972); the two closed forms are standard and re-derived here so they are
# checkable:
#
#   A uniform field along z induces mu on both.  Each sees the applied field
#   plus the other's dipole field, ``+2 mu / r^3`` for a dipole ALONG the axis,
#   so ``mu = alpha (E + 2 mu / r^3)`` and ``mu = alpha E / (1 - 2 alpha/r^3)``.
#   With ``alpha_zz = 2 mu / E`` (two sites),
#
#       alpha_zz = 2 alpha r^3 / (r^3 - 2 alpha) .
#
#   Perpendicular, the field of a transverse dipole at the other site is
#   ``-mu / r^3``, giving
#
#       alpha_xx = 2 alpha r^3 / (r^3 + alpha) .
#
# The four eigenmodes of the undamped relay and where each goes soft:
#
#       symmetric parallel        1/alpha - 2/r^3    soft at alpha = r^3 / 2
#       antisymmetric parallel    1/alpha + 2/r^3
#       symmetric transverse      1/alpha + 1/r^3
#       antisymmetric transverse  1/alpha - 1/r^3    soft at alpha = r^3
#
# The two orientations therefore give genuinely different conditions, and the
# transverse instability lives in the ANTISYMMETRIC mode, which a uniform field
# cannot excite -- so ``alpha_xx`` stays on its closed form straight through it.
# Both facts are tested below.


def _two_atoms(separation):
    return torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, float(separation)]])


@pytest.mark.parametrize("alpha_volume", [0.05, 0.2, 0.5, 1.0, 3.0])
@pytest.mark.parametrize("separation", [3.0, 4.5])
def test_two_atom_relay_matches_the_applequist_closed_forms(alpha_volume, separation):
    """``alpha_zz`` and ``alpha_xx`` against the closed forms, undamped.

    The damping is switched off exactly (external_harness module docstring), so
    this is the bare point-dipole relay the closed forms describe.  Measured
    worst case over the ten combinations: 2.1e-16 relative on both components,
    i.e. exact.
    """
    alpha, _assembly = _polarizability(
        _two_atoms(separation), alpha_volume, damping_alpha=1.0e-10
    )
    cube = separation**3
    expected_zz = 2.0 * alpha_volume * cube / (cube - 2.0 * alpha_volume)
    expected_xx = 2.0 * alpha_volume * cube / (cube + alpha_volume)

    assert float(alpha[2, 2]) == pytest.approx(expected_zz, rel=1.0e-12), (
        f"r = {separation}, alpha = {alpha_volume}: alpha_zz = "
        f"{float(alpha[2, 2]):.12f}, closed form {expected_zz:.12f}"
    )
    for axis in (0, 1):
        assert float(alpha[axis, axis]) == pytest.approx(expected_xx, rel=1.0e-12), (
            f"r = {separation}, alpha = {alpha_volume}: alpha_{axis}{axis} = "
            f"{float(alpha[axis, axis]):.12f}, closed form {expected_xx:.12f}"
        )
    off_diagonal = float((alpha - torch.diag(alpha.diagonal())).abs().max())
    assert off_diagonal < 1.0e-12 * float(alpha[2, 2])


@pytest.mark.parametrize("fraction", [0.5, 0.9, 0.99, 0.999, 0.9999])
def test_the_parallel_relay_diverges_at_alpha_equals_r_cubed_over_two(fraction):
    """The BINDING divergence: ``alpha_zz -> inf`` as ``alpha -> r^3 / 2``.

    This is the one the uniform-field response can see, and following it to
    within 1e-4 of the pole is a far stronger test of the propagator than any
    value away from it: at ``fraction = 0.9999`` the closed form is 269973,
    four orders of magnitude above ``2 alpha``, and it is reached only if the
    ``+2/r^3`` axial coupling is right to the last digit.

    Measured, ``r = 3 A`` so the pole is at ``alpha = 13.5 A^3``:

        fraction   alpha_zz        relative error
        0.5             27.0000    1.3e-16
        0.9            243.0000    2.1e-15
        0.99          2673.0000    1.0e-14
        0.999        26973.0000    6.3e-14
        0.9999      269973.0000    2.6e-12

    the error growing exactly as the conditioning of the operator does.
    """
    separation = 3.0
    critical = separation**3 / 2.0
    alpha_volume = fraction * critical
    alpha, _assembly = _polarizability(
        _two_atoms(separation), alpha_volume, damping_alpha=1.0e-10
    )
    cube = separation**3
    expected = 2.0 * alpha_volume * cube / (cube - 2.0 * alpha_volume)
    assert float(alpha[2, 2]) == pytest.approx(expected, rel=1.0e-10), (
        f"fraction {fraction}: alpha_zz = {float(alpha[2, 2]):.6f}, "
        f"closed form {expected:.6f}"
    )
    # ... and it really is diverging, not merely large.
    assert float(alpha[2, 2]) > 2.0 * alpha_volume / (1.0 - fraction) * 0.9


def test_the_undamped_operator_is_singular_exactly_at_the_binding_divergence():
    """The divergence is an eigenvalue of the relay crossing zero, not a blow-up.

    Below ``alpha = r^3/2`` the dipole block is positive definite; above it the
    smallest eigenvalue is NEGATIVE and ``alpha_zz`` comes out negative -- a
    stationary point that is a saddle, which is the failure SPEC 2.1 item 4
    exists to catch.  Measured at ``r = 3 A``:

        alpha / (r^3/2)   min eig        alpha_zz
        0.900             +1.185e-01       +243.00
        0.999             +1.068e-03     +26973.00
        1.000              0.000e+00      singular
        1.001             -1.066e-03     -27027.00
        1.100             -9.697e-02       -297.00

    A negative polarizability is unphysical and it is what an undamped relay
    returns past the pole; SPEC 1.6's ``a < 1`` is what makes it unreachable in
    the damped model (next test).
    """
    separation = 3.0
    critical = separation**3 / 2.0

    below = _polarizability(
        _two_atoms(separation), 0.999 * critical, damping_alpha=1.0e-10
    )
    eigenvalues = torch.linalg.eigvalsh(_dipole_block(below[1]))
    assert float(eigenvalues.min()) > 0.0
    assert float(below[0][2, 2]) > 0.0

    above_alpha, above_assembly = _polarizability(
        _two_atoms(separation), 1.001 * critical, damping_alpha=1.0e-10
    )
    eigenvalues = torch.linalg.eigvalsh(_dipole_block(above_assembly))
    assert float(eigenvalues.min()) < 0.0, (
        "past the Applequist pole the undamped relay must lose positive "
        f"definiteness; smallest eigenvalue {float(eigenvalues.min()):+.3e}"
    )
    assert float(above_alpha[2, 2]) < 0.0, (
        "past the pole the uniform-field response must come out NEGATIVE; got "
        f"{float(above_alpha[2, 2]):.4f}"
    )


@pytest.mark.parametrize("fraction", [0.9, 1.0, 1.5, 3.0, 10.0, 100.0])
def test_thole_damping_removes_the_applequist_divergence(fraction):
    """SPEC 1.6, against the external reference it is a claim about.

    "``sup_r alpha |T| = a`` exactly, so the relay is positive definite for all
    r and all alpha iff ``a < 1``."  The Applequist pole is precisely where an
    undamped relay stops being positive definite, so running the damped operator
    straight through it -- and a hundred times past it -- is the external test
    of that claim.  Measured at ``r = 3 A``, ``a = 0.39``:

        alpha/(r^3/2)   alpha_zz (damped)   undamped closed form   min eig
        0.9                     25.048                 243.000    +8.76e-01
        1.0                     27.143                     inf    +7.78e-01
        1.5                     37.246                 -81.000    +4.95e-01
        3.0                     66.651                 -40.500    +2.33e-01
        10.0                   202.725                 -30.000    +6.66e-02
        100.0                 1950.947                 -27.273    +6.52e-03

    ``alpha_zz`` stays finite, positive and MONOTONE in ``alpha`` throughout,
    while the closed form has passed through infinity and come back negative.
    """
    separation = 3.0
    alpha_volume = fraction * separation**3 / 2.0
    alpha, assembly = _polarizability(_two_atoms(separation), alpha_volume)

    assert math.isfinite(float(alpha[2, 2]))
    assert float(alpha[2, 2]) > 0.0, (
        f"damped alpha_zz = {float(alpha[2, 2])} at alpha = {alpha_volume} A^3"
    )
    eigenvalues = torch.linalg.eigvalsh(_dipole_block(assembly))
    assert float(eigenvalues.min()) > 0.0, (
        "SPEC 1.6 requires the damped relay to stay positive definite for every "
        f"alpha; smallest eigenvalue {float(eigenvalues.min()):+.3e} at "
        f"alpha/(r^3/2) = {fraction}"
    )


def test_the_damped_relay_is_monotone_in_the_atomic_polarizability():
    """Companion to the test above: finiteness alone is not enough.

    A damping that had merely clamped the response would also be finite.  The
    physical requirement is that a more polarizable atom gives a more
    polarizable pair, monotonically, with no pole anywhere -- which is what
    positive definiteness at every alpha buys.
    """
    separation = 3.0
    values = [
        float(_polarizability(_two_atoms(separation), fraction * separation**3 / 2.0)[0][2, 2])
        for fraction in (0.9, 1.0, 1.5, 3.0, 10.0, 100.0)
    ]
    assert values == sorted(values)
    assert values[-1] > 10.0 * values[0]


def test_the_perpendicular_instability_is_invisible_to_a_uniform_field():
    """``alpha = r^3`` softens the ANTISYMMETRIC transverse mode, and only that.

    The two orientations give genuinely different conditions, and this is the
    one a polarizability measurement cannot see: at ``alpha = 1.1 r^3`` the
    undamped operator has TWO negative eigenvalues (the degenerate x and y
    antisymmetric modes, measured -0.0485 each) while ``alpha_xx`` still sits
    exactly on the closed form ``2 alpha r^3 / (r^3 + alpha) = 28.28571``,
    because a uniform field couples only to the symmetric combination.

    So ``alpha > 0`` is NOT evidence that the relay is stable, which is exactly
    why SPEC 2.1 item 4 requires the minimum to be verified rather than
    inferred from the reported response.  Measured spectrum at
    ``alpha = 1.1 r^3``, r = 3 A:
    ``[-0.5818, -0.0485, -0.0485, +1.0182, +1.0182, +1.5515]``.
    """
    separation = 3.0
    alpha_volume = 1.1 * separation**3
    alpha, assembly = _polarizability(
        _two_atoms(separation), alpha_volume, damping_alpha=1.0e-10
    )

    eigenvalues = torch.linalg.eigvalsh(_dipole_block(assembly))
    negative = int((eigenvalues < 0.0).sum())
    assert negative == 3, (
        "past both poles the undamped relay should have the symmetric-parallel "
        "and the two antisymmetric-transverse modes negative; got "
        f"{negative} negative eigenvalues: {[round(float(v), 4) for v in eigenvalues]}"
    )

    cube = separation**3
    expected_xx = 2.0 * alpha_volume * cube / (cube + alpha_volume)
    assert float(alpha[0, 0]) == pytest.approx(expected_xx, rel=1.0e-12), (
        "the transverse response must still follow its closed form through the "
        f"antisymmetric instability: {float(alpha[0, 0]):.6f} vs {expected_xx:.6f}"
    )
    assert float(alpha[0, 0]) > 0.0
    assert float(alpha[2, 2]) < 0.0, (
        "meanwhile the PARALLEL response, whose mode a uniform field does "
        "excite, is negative -- which is what makes the transverse silence "
        "worth recording"
    )


# ===========================================================================
# 2.  A discretised conducting sphere: alpha = a^3.
# ===========================================================================
#
# The polarizability of an isolated conducting sphere of radius ``a`` is
# ``alpha = a^3`` in Gaussian units -- exactly, with no adjustable constant.
# (Jackson, *Classical Electrodynamics*, section 2.5: a grounded/isolated
# conducting sphere in a uniform field acquires the dipole ``p = a^3 E``.)
#
# The model's conductor is the CHARGE-TRANSFER sector with the hardness at its
# SPEC 1.2 floor: sites on the sphere, connected by a spanning tree with
# ``kappa -> 0`` so charge equilibrates globally (SPEC 1.4: kappa -> 0 is global
# QEq), no induced dipoles.  Two discretisation errors compete and both vanish
# together:
#
#   * the Gaussian width ``sigma`` smears the surface, which INCREASES the
#     effective radius;
#   * the on-site hardness ``eta = 2 K_E / (sigma sqrt(pi))`` costs energy,
#     which DECREASES the response.
#
# The second is the interesting one, because the SPEC hardness floor is
# mandatory and it is *not* the value a set of Gaussian clouds would have
# (``kernel_self`` documents the tight bound as ``sqrt(2)`` times smaller).  It
# nevertheless drops out in the continuum limit: a conductor's induced charge
# per site scales as ``1/N``, so the on-site term ``N eta q^2`` scales as
# ``eta/N ~ 1/(sigma N)``, while the Coulomb term is O(1).  Taking
# ``sigma = c d`` with ``d = a sqrt(4 pi / N)`` the nearest-neighbour spacing
# makes the whole construction scale free and the error O(1/sqrt(N)).
#
# That is what is measured: the error extrapolates to zero for FOUR independent
# choices of ``c``, which is what distinguishes "converges to a^3" from "happens
# to cross a^3 at one discretisation".


def _conducting_sphere(n_points, radius, sigma_fraction):
    """``alpha`` of a discretised conductor, in A^3, and the geometry it used."""
    spacing = radius * math.sqrt(4.0 * math.pi / n_points)
    sigma = sigma_fraction * spacing
    # The floor is a strict lower bound in exact arithmetic; nudge it so float
    # round-off cannot put eta an ulp below it (the same guard the harness's
    # periodic builder uses).
    eta = 2.0 * K_E / (sigma * math.sqrt(math.pi)) * (1.0 + 1.0e-12)
    bonds = torch.tensor(
        [[0] * (n_points - 1), list(range(1, n_points))], dtype=torch.long
    )
    alpha, _assembly = _polarizability(
        fibonacci_sphere(n_points, radius),
        1.0e-10,          # A_mu enormous: no induced-dipole channel
        bond_index=bonds,
        sigma=sigma,
        eta=eta,
        kappa=1.0e-12,    # SPEC 1.4: kappa -> 0 is global QEq
    )
    return alpha, spacing


@pytest.mark.parametrize("sigma_fraction", [0.20, 0.30, 0.40, 0.55])
def test_discretised_conducting_sphere_extrapolates_to_alpha_equals_a_cubed(
    sigma_fraction,
):
    """``alpha / a^3 -> 1`` as the discretisation is refined, for every sigma/d.

    Richardson extrapolation in ``1 / sqrt(N)``, which is the measured order of
    the error.  The point of running four values of ``sigma / d`` is that they
    approach the limit from OPPOSITE SIDES -- ``c = 0.20`` from below (too much
    on-site hardness), ``c = 0.55`` from above (too much smearing) -- so a
    common scale error could not be absorbed by any of them.

    Measured ``alpha / a^3`` at ``a = 4 A`` (the value is independent of ``a``;
    see the scale-invariance test below):

        sigma/d   N=252    N=492    N=1002   2-pt Richardson
        0.20      0.91456  0.93749  0.95543    0.99866  (from below)
        0.30      1.00709  1.00525  1.00374    1.00223
        0.40      1.06349  1.04483  1.03107    0.99941
        0.55      1.12871  1.08919  1.06099    0.99755  (from above)

    all four within 0.3 % of the exact ``a^3``.
    """
    radius = 4.0
    counts = (492, 1002)
    values = [
        float(_conducting_sphere(n, radius, sigma_fraction)[0].diagonal().mean())
        / radius**3
        for n in counts
    ]
    # alpha/a^3 = 1 + C/sqrt(N): eliminate C from the two finest grids.
    weights = [1.0 / math.sqrt(n) for n in counts]
    extrapolated = values[-1] + (values[-1] - values[-2]) * weights[-1] / (
        weights[-2] - weights[-1]
    )
    assert extrapolated == pytest.approx(1.0, abs=0.01), (
        f"sigma/d = {sigma_fraction}: alpha/a^3 = {values} at N = {counts}, "
        f"extrapolates to {extrapolated:.6f}, exact conductor is 1"
    )


def test_the_conducting_sphere_error_is_actually_converging():
    """Guard: the extrapolation above must be extrapolating something.

    A sequence that had already converged, or one that was not converging at
    all, would both pass a Richardson step.  This asserts the raw error falls
    monotonically and by the expected factor -- ``1/sqrt(N)``, so doubling N
    should remove about 30 % of it -- at the two sigma/d that bracket the limit.
    """
    radius = 4.0
    for sigma_fraction in (0.20, 0.55):
        errors = [
            abs(
                float(_conducting_sphere(n, radius, sigma_fraction)[0].diagonal().mean())
                / radius**3
                - 1.0
            )
            for n in (252, 492, 1002)
        ]
        assert errors[0] > errors[1] > errors[2], (
            f"sigma/d = {sigma_fraction}: errors {errors} are not decreasing"
        )
        # sqrt(492/252) = 1.397, sqrt(1002/492) = 1.427: expect ratios near those.
        assert 1.2 < errors[0] / errors[1] < 1.7
        assert 1.2 < errors[1] / errors[2] < 1.7


@pytest.mark.parametrize("radius", [1.7, 4.0, 11.3])
def test_the_conducting_sphere_polarizability_is_scale_invariant(radius):
    """``alpha`` must scale exactly as ``a^3`` -- the unit test a single size hides.

    Everything in the construction scales with ``a``: the positions, the
    Gaussian width, and hence ``eta ~ 1/sigma``.  The response
    ``D^T K^-1 D`` then scales as ``a^2 * a = a^3`` identically, so the ratio
    ``alpha / a^3`` is a pure number independent of the radius.  A stray ``K_E``
    or a length that had been hard-coded in A would break this even though the
    extrapolation test above -- run at a single radius -- would still pass.

    Measured at N = 492, sigma/d = 0.30: ``alpha`` = 4.938771, 64.335704 and
    1450.465527 A^3 at a = 1.7, 4.0 and 11.3 A, i.e. ``alpha / a^3`` =
    1.005245375657, 1.005245369342 and 1.005245368840 -- a spread of 6.8e-09
    relative across a 300-fold range in ``alpha``.  That residual is the
    conditioning of the 1967-dimensional dense constrained solve, not a scale
    dependence: it does not grow with the radius, and it shrinks with N.
    """
    reference = 1.00524537  # alpha/a^3 at N = 492, sigma/d = 0.30, measured
    alpha, _spacing = _conducting_sphere(492, radius, 0.30)
    ratio = float(alpha.diagonal().mean()) / radius**3
    assert ratio == pytest.approx(reference, rel=1.0e-7), (
        f"a = {radius} A: alpha/a^3 = {ratio:.12f}, expected {reference}"
    )
    # Isotropic, as a sphere must be, and symmetric (Maxwell reciprocity).
    diagonal = alpha.diagonal()
    assert float(diagonal.max() - diagonal.min()) / float(diagonal.mean()) < 1.0e-3
    assert float((alpha - alpha.transpose(0, 1)).abs().max()) < 1.0e-10 * float(
        diagonal.mean()
    )


@pytest.mark.slow
def test_the_conducting_sphere_converges_from_both_sides_at_high_resolution():
    """The full convergence study, N = 92 to 2002, recorded once.

    Slow because the dense constrained solve is O((4N)^3).  It is here because
    the two-point extrapolation above is a summary of it, and a summary is worth
    less if the sequence behind it is never asserted in full.

    Measured ``alpha / a^3`` at a = 4 A:

        sigma/d    N=92     N=252    N=492    N=1002   N=2002
        0.20      0.86503  0.91456  0.93749  0.95543  0.96807
        0.30      1.01035  1.00709  1.00525  1.00374  1.00267
        0.40      1.10785  1.06349  1.04483  1.03107  1.02181
        0.55      1.22915  1.12871  1.08919  1.06099  1.04243

    Least-squares intercept in ``1/sqrt(N)`` over N = 252..2002:
    0.997158, 1.000283, 0.998739, 0.994240.
    """
    radius = 4.0
    counts = (252, 492, 1002, 2002)
    weights = np.array([1.0 / math.sqrt(n) for n in counts])
    design = np.vstack([np.ones(len(counts)), weights]).T
    for sigma_fraction in (0.20, 0.30, 0.40, 0.55):
        values = np.array(
            [
                float(_conducting_sphere(n, radius, sigma_fraction)[0].diagonal().mean())
                / radius**3
                for n in counts
            ]
        )
        intercept = float(np.linalg.lstsq(design, values, rcond=None)[0][0])
        assert intercept == pytest.approx(1.0, abs=0.01), (
            f"sigma/d = {sigma_fraction}: {values} extrapolates to {intercept:.6f}"
        )


# ===========================================================================
# 3.  Published molecular polarizabilities.
# ===========================================================================
def _water_geometry():
    """Experimental water: r(OH) = 0.9572 A, angle(HOH) = 104.52 deg, C2v.

    In the C2v frame with z the two-fold axis and the molecule in the xz plane,
    so ``alpha`` is diagonal and its three components are the three symmetry
    channels: in-plane along the bisector, in-plane perpendicular to it, and
    out of plane.
    """
    half_angle = math.radians(104.52) / 2.0
    bond = 0.9572
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [bond * math.sin(half_angle), 0.0, bond * math.cos(half_angle)],
            [-bond * math.sin(half_angle), 0.0, bond * math.cos(half_angle)],
        ]
    )


def _benzene_geometry():
    """Experimental benzene D6h: r(CC) = 1.397 A, r(CH) = 1.087 A, ring in xy.

    The six carbons on a regular hexagon of circumradius r(CC) -- which for a
    regular hexagon is also the C-C bond length -- and each hydrogen radially
    outward at r(CC) + r(CH).  z is the C6 axis, so ``alpha_zz`` is the
    out-of-plane component and ``alpha_xx = alpha_yy`` the in-plane one by
    symmetry.
    """
    cc, ch = 1.397, 1.087
    carbons = [
        [cc * math.cos(k * math.pi / 3.0), cc * math.sin(k * math.pi / 3.0), 0.0]
        for k in range(6)
    ]
    hydrogens = [
        [
            (cc + ch) * math.cos(k * math.pi / 3.0),
            (cc + ch) * math.sin(k * math.pi / 3.0),
            0.0,
        ]
        for k in range(6)
    ]
    return torch.tensor(carbons + hydrogens)


def _reference_thole_relay(positions, alpha_volume, a, form):
    """An INDEPENDENT Thole relay in numpy: ``alpha = sum_ij (A^-1)_ij``.

    Imports nothing from ``trace_electrostatics``.  ``A_ii = I / alpha_i`` and
    ``A_ij = -T_ij`` with the damped propagator
    ``T = lam5 3 r r / r^5 - lam3 I / r^3``, which is the standard Applequist
    matrix with Thole smearing.

    ``form="exp3"`` is ``rho(u) propto exp(-a u^3)``, this package's damping;
    ``form="exp1"`` is ``rho(u) propto exp(-a u)``, van Duijnen and Swart's.
    The second is carried only so this file can say *which* of the two the
    benzene discrepancy belongs to.
    """
    positions = np.asarray(positions, dtype=float)
    alpha_volume = np.asarray(alpha_volume, dtype=float)
    n_atoms = len(alpha_volume)
    matrix = np.zeros((3 * n_atoms, 3 * n_atoms))
    for i in range(n_atoms):
        matrix[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = np.eye(3) / alpha_volume[i]
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            vector = positions[i] - positions[j]
            distance = float(np.linalg.norm(vector))
            if form == "exp3":
                v = a * distance**3 / math.sqrt(alpha_volume[i] * alpha_volume[j])
                lam3 = 1.0 - math.exp(-v)
                lam5 = 1.0 - (1.0 + v) * math.exp(-v)
            else:
                u = distance / (alpha_volume[i] * alpha_volume[j]) ** (1.0 / 6.0)
                au = a * u
                decay = math.exp(-au)
                lam3 = 1.0 - (1.0 + au + au * au / 2.0) * decay
                lam5 = 1.0 - (1.0 + au + au * au / 2.0 + au**3 / 6.0) * decay
            propagator = (
                lam5 * 3.0 * np.outer(vector, vector) / distance**5
                - lam3 * np.eye(3) / distance**3
            )
            matrix[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = -propagator
    inverse = np.linalg.inv(matrix)
    total = np.zeros((3, 3))
    for i in range(n_atoms):
        for j in range(n_atoms):
            total += inverse[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
    return total


@pytest.mark.parametrize(
    "name, geometry, alpha_volume",
    [
        ("water", _water_geometry, [ALPHA_O_THOLE, ALPHA_H_THOLE, ALPHA_H_THOLE]),
        ("benzene", _benzene_geometry, [ALPHA_C_THOLE] * 6 + [ALPHA_H_THOLE] * 6),
    ],
)
def test_the_package_relay_matches_an_independent_thole_relay(
    name, geometry, alpha_volume
):
    """Separate the IMPLEMENTATION question from the PARAMETERISATION question.

    The two molecular tests below compare against experiment and one of them
    misses by 19 %.  That number is only interpretable if the relay algebra is
    known to be right, so it is checked first, against a from-scratch numpy
    Applequist-Thole matrix inversion that imports nothing from the package.

    Measured agreement: 4e-16 relative on both molecules, every tensor
    component.  So anything left over is the parameterisation, not the code.
    """
    positions = geometry()
    alpha, _assembly = _polarizability(positions, alpha_volume)
    reference = _reference_thole_relay(
        positions.numpy(), alpha_volume, THOLE_A_AMOEBA, "exp3"
    )
    largest = float(np.abs(reference).max())
    difference = float(np.abs(alpha.numpy() - reference).max())
    assert difference < 1.0e-12 * largest, (
        f"{name}: package alpha differs from an independent Thole relay by "
        f"{difference:.3e} (largest component {largest:.4f} A^3)"
    )


def test_water_molecular_polarizability_from_atomic_values_matches_experiment():
    """Water from Thole/AMOEBA atomic values against the experimental 1.45 A^3.

    ``alpha(O) = 0.837``, ``alpha(H) = 0.496`` A^3 (Thole 1981 / AMOEBA; sources
    in the module docstring) sum to 1.829 A^3 -- 26 % above the molecule -- so
    the relay has real work to do, and getting 1.45 out of 1.829 is a test of
    the damped propagator and not of the atomic values.

    Measured at the package's default ``a = 0.39``: mean ``alpha = 1.4052 A^3``,
    3.1 % below the experimental 1.45.  At Thole's own ``a = 0.572`` for the
    same functional form: 1.4144 A^3, 2.5 % below.  Both are inside the 1-2 %
    -ish accuracy the Thole model is quoted at, once the difference between the
    two screening parameters is allowed for.
    """
    positions = _water_geometry()
    alpha, _assembly = _polarizability(
        positions, [ALPHA_O_THOLE, ALPHA_H_THOLE, ALPHA_H_THOLE]
    )
    mean = float(alpha.diagonal().mean())

    assert mean == pytest.approx(1.4052, abs=1.0e-3), (
        f"regression: mean alpha = {mean:.4f} A^3, previously measured 1.4052"
    )
    relative = (mean - WATER_ALPHA_EXPERIMENT) / WATER_ALPHA_EXPERIMENT
    assert abs(relative) < 0.05, (
        f"mean alpha = {mean:.4f} A^3 against experiment "
        f"{WATER_ALPHA_EXPERIMENT} A^3, a {relative:+.1%} error"
    )

    # The additive (no-relay) sum is much worse, so the relay is what is being
    # tested rather than the atomic values.
    additive = ALPHA_O_THOLE + 2.0 * ALPHA_H_THOLE
    assert abs(mean - WATER_ALPHA_EXPERIMENT) < 0.25 * abs(
        additive - WATER_ALPHA_EXPERIMENT
    )

    # C2v: the two hydrogens are equivalent, so alpha is diagonal in this frame.
    off_diagonal = float((alpha - torch.diag(alpha.diagonal())).abs().max())
    assert off_diagonal < 1.0e-12 * mean


def test_benzene_polarizability_anisotropy_matches_experiment():
    """Benzene is the sharp case: the ANISOTROPY, not only the mean.

    An isotropic model gets benzene's large in-plane / out-of-plane contrast
    wrong, so that contrast is the thing to report.  Experiment (module
    docstring): ``alpha_xx = alpha_yy = 12.3``, ``alpha_zz = 6.3-6.7``, mean
    ``10.3 A^3``, hence an anisotropy ratio ``alpha_xx / alpha_zz`` of
    1.84-1.95.

    Measured, Thole/AMOEBA atomic values at the package's default a = 0.39:

        alpha_xx = alpha_yy = 9.8111 A^3   (-20 % against 12.3)
        alpha_zz            = 5.3915 A^3   (-15 % against 6.35)
        mean                = 8.3379 A^3   (-19 % against 10.3)
        alpha_xx/alpha_zz   = 1.8197       ( -6 % against 1.94)

    So the model reproduces the SHAPE of benzene's response -- D6h symmetry
    exact, in-plane well above out-of-plane, the ratio within 6 % -- while
    underestimating its SIZE by 19 %.  The gap is the parameterisation, not this
    implementation; ``test_the_benzene_gap_is_the_damping_parameterisation``
    below locates it.
    """
    positions = _benzene_geometry()
    alpha, _assembly = _polarizability(
        positions, [ALPHA_C_THOLE] * 6 + [ALPHA_H_THOLE] * 6
    )
    diagonal = alpha.diagonal()
    in_plane = 0.5 * (float(diagonal[0]) + float(diagonal[1]))
    out_of_plane = float(diagonal[2])
    mean = float(diagonal.mean())

    # D6h: the two in-plane components must be equal exactly, and alpha diagonal.
    assert float(diagonal[0]) == pytest.approx(float(diagonal[1]), rel=1.0e-12)
    off_diagonal = float((alpha - torch.diag(diagonal)).abs().max())
    assert off_diagonal < 1.0e-12 * mean

    # The qualitative statement experiment makes, which an isotropic model fails.
    assert in_plane > 1.5 * out_of_plane, (
        f"benzene must be strongly oblate: alpha_xx = {in_plane:.4f}, "
        f"alpha_zz = {out_of_plane:.4f}"
    )

    # The quantitative one: the ANISOTROPY RATIO, against 12.3 / 6.3-6.7.
    ratio = in_plane / out_of_plane
    low = BENZENE_ALPHA_IN_PLANE_EXPERIMENT / BENZENE_ALPHA_OUT_OF_PLANE_RANGE[1]
    high = BENZENE_ALPHA_IN_PLANE_EXPERIMENT / BENZENE_ALPHA_OUT_OF_PLANE_RANGE[0]
    assert abs(ratio - 0.5 * (low + high)) / (0.5 * (low + high)) < 0.10, (
        f"alpha_xx/alpha_zz = {ratio:.4f}, experiment {low:.3f}-{high:.3f}"
    )

    # And the measured magnitudes, pinned so the 19 % gap cannot drift silently.
    assert in_plane == pytest.approx(9.8111, abs=2.0e-3)
    assert out_of_plane == pytest.approx(5.3915, abs=2.0e-3)
    assert mean == pytest.approx(8.3379, abs=2.0e-3)
    shortfall = (mean - BENZENE_ALPHA_MEAN_EXPERIMENT) / BENZENE_ALPHA_MEAN_EXPERIMENT
    assert -0.22 < shortfall < -0.15, (
        f"the benzene shortfall has moved: {shortfall:+.1%}, previously -19.0 %"
    )


def test_the_benzene_gap_is_the_damping_parameterisation_not_the_implementation():
    """Locate the 19 %: it is the SMEARING FUNCTION, and the fit that goes with it.

    Three relays, all from the same independent numpy Applequist-Thole matrix
    (``_reference_thole_relay``), differing only in the smearing and the atomic
    values fitted to it.  Measured mean polarizability of benzene against the
    experimental 10.3 A^3:

        rho ~ exp(-a u^3), a = 0.39  (AMOEBA; THIS PACKAGE)    8.338   -19 %
        rho ~ exp(-a u^3), a = 0.572 (Thole 1981)              9.018   -13 %
        rho ~ exp(-a u),   a = 2.1304 (van Duijnen & Swart)   10.190    -1.3 %

    The third reproduces benzene to the 1-2 % the Thole model is quoted at, and
    it differs from the first ONLY in the functional form of the smeared charge
    density and the atomic polarizabilities refitted to it.  The package's own
    result sits on the first line and agrees with the independent relay to
    4e-16, so the package is computing the right thing for the damping it
    declares.

    This is a FINDING, recorded rather than tolerated:

    * the ``exp(-a u^3)`` form is what SPEC 1.6 specifies, and the bound
      ``sup_r alpha|T| = a < 1`` is a property of that form.  van Duijnen and
      Swart's ``a = 2.1304`` is not a value this package could accept, and
      ``kernels._validate_thole_a`` is right to refuse it -- the two ``a`` are
      not the same parameter;
    * so a model that has to reproduce aromatic polarizabilities from fixed
      isotropic atomic values would need either the ``exp(-a u)`` smearing or
      an element-resolved refit (an aromatic-carbon ``alpha`` near 1.75 A^3
      brings the mean to 9.59 and the in-plane component to 11.29);
    * for THIS package that is a training-time question, since ``A_mu`` is a
      learned readout (SPEC 3) rather than a table.  The relevant conclusion is
      that the engine is faithful and the shortfall is not an engine defect.
    """
    positions = _benzene_geometry().numpy()
    package_form = _reference_thole_relay(
        positions,
        [ALPHA_C_THOLE] * 6 + [ALPHA_H_THOLE] * 6,
        THOLE_A_AMOEBA,
        "exp3",
    )
    thole_original = _reference_thole_relay(
        positions,
        [ALPHA_C_THOLE] * 6 + [ALPHA_H_THOLE] * 6,
        THOLE_A_ORIGINAL,
        "exp3",
    )
    van_duijnen = _reference_thole_relay(
        positions, [ALPHA_C_VDS] * 6 + [ALPHA_H_VDS] * 6, THOLE_A_VDS, "exp1"
    )

    means = [float(np.trace(m) / 3.0) for m in (package_form, thole_original, van_duijnen)]
    assert means[0] == pytest.approx(8.3379, abs=2.0e-3)
    assert means[1] == pytest.approx(9.0176, abs=2.0e-3)
    assert means[2] == pytest.approx(10.1900, abs=2.0e-3)

    # The alternative smearing lands inside 2 % of experiment; this one does not.
    assert abs(means[2] - BENZENE_ALPHA_MEAN_EXPERIMENT) < 0.02 * (
        BENZENE_ALPHA_MEAN_EXPERIMENT
    )
    assert abs(means[0] - BENZENE_ALPHA_MEAN_EXPERIMENT) > 0.15 * (
        BENZENE_ALPHA_MEAN_EXPERIMENT
    )
    # ... and the in-plane component with it: 12.398 against the measured 12.3.
    assert float(van_duijnen[0, 0]) == pytest.approx(
        BENZENE_ALPHA_IN_PLANE_EXPERIMENT, rel=0.02
    )


def test_the_van_duijnen_screening_parameter_is_refused_by_this_package():
    """The two ``a`` are different parameters, and the package must not accept both.

    ``kernels._validate_thole_a`` enforces ``0 < a < 1`` because for the
    ``exp(-a u^3)`` smearing ``sup_r alpha |T| = a``, so ``a >= 1`` is the
    polarization catastrophe (SPEC 1.6).  van Duijnen and Swart's ``a = 2.1304``
    belongs to a different smearing with a different bound, and feeding it in
    here would be a silent category error.  It raises, which is the correct
    behaviour and is recorded so the module docstring's warning has teeth.
    """
    thole_damping = require("kernels", "thole_damping")
    with pytest.raises(ValueError, match="strictly less than"):
        thole_damping(
            torch.tensor([1.4]),
            torch.tensor([1.405]),
            torch.tensor([1.405]),
            THOLE_A_VDS,
        )


def test_thole_original_screening_gives_a_larger_molecular_polarizability():
    """``a = 0.572`` must give MORE polarization than ``a = 0.39``, in the package.

    AMOEBA's documentation states that its universal 0.39 "leads to stronger
    damping and less short-range polarization than the original value of 0.572
    suggested by Thole".  That is a monotonicity statement about this exact
    functional form, so it is checkable here: larger ``a`` means larger ``v``
    means ``lambda_3, lambda_5`` closer to 1 means a less screened relay.

    Measured on benzene: mean 8.3379 A^3 at a = 0.39, 9.0176 at a = 0.572
    (+8.2 %); on water 1.4052 and 1.4144 (+0.7 %, the relay being far weaker at
    a 0.96 A bond because the atoms are small).
    """
    positions = _benzene_geometry()
    weak, _a = _polarizability(
        positions, [ALPHA_C_THOLE] * 6 + [ALPHA_H_THOLE] * 6, thole_a=THOLE_A_AMOEBA
    )
    strong, _b = _polarizability(
        positions, [ALPHA_C_THOLE] * 6 + [ALPHA_H_THOLE] * 6, thole_a=THOLE_A_ORIGINAL
    )
    assert float(strong.diagonal().mean()) > float(weak.diagonal().mean())
    assert float(strong.diagonal().mean()) == pytest.approx(9.0176, abs=2.0e-3)


def test_the_relay_is_not_merely_the_additive_atomic_sum():
    """Guard: the molecular tests must be testing the relay.

    If the induced-dipole coupling were dead -- a ``T`` block that came out
    identically zero, which SPEC 3 warns is exactly what a mis-declared head
    produces -- every molecular polarizability would be the plain sum of the
    atomic values and the tests above would still be within their tolerances on
    water.  The relay must move it, and by a lot: benzene's additive sum is
    10.980 A^3 and the relay gives 8.338, a 24 % reduction in the mean and a
    82 % in-plane/out-of-plane split out of an isotropic input.
    """
    positions = _benzene_geometry()
    alpha_volume = [ALPHA_C_THOLE] * 6 + [ALPHA_H_THOLE] * 6
    alpha, _assembly = _polarizability(positions, alpha_volume)
    additive = sum(alpha_volume)

    assert additive == pytest.approx(10.98, abs=1.0e-6)
    assert abs(float(alpha.diagonal().mean()) - additive) > 0.2 * additive, (
        "the relay barely moved the additive sum -- is the dipole-dipole block "
        "alive?"
    )
    # The input is isotropic; every bit of the anisotropy is the relay's.
    diagonal = alpha.diagonal()
    assert float(diagonal.max() / diagonal.min()) > 1.5
