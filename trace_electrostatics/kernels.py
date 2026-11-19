"""Damped electrostatic kernels: Gaussian charge clouds and Thole dipoles.

Every function here is a pure function of torch tensors, is fully
differentiable (including at coincident sites), and carries ``K_E`` already --
the returned quantities are in eV per unit of the moments that multiply them,
so nothing downstream may apply ``K_E`` a second time:

    J(r)        eV / e^2            charge-charge
    T01(r)      eV / (e * (e A))    charge-dipole
    T(r)        eV / (e A)^2        dipole-dipole
    T02(r)      eV / (e * (e A^2))  charge-quadrupole
    T12(r)      eV / ((e A) (e A^2))        dipole-quadrupole
    T22(r)      eV / (e A^2)^2      quadrupole-quadrupole

Charge-charge: Gaussian/erf ONLY
--------------------------------
A charge ``q_i`` is smeared into a spherical Gaussian of width ``sigma_i``.
The interaction of two such clouds is the erf-damped Coulomb kernel

    J(r_ij) = K_E * erf(r_ij / sigma_ij) / r_ij ,  sigma_ij = sqrt(sigma_i^2 + sigma_j^2)
    J(0)    = 2 K_E / (sigma sqrt(pi))                     (removable singularity)

The Ohno/Klopman form ``1 / sqrt(r^2 + s^2)`` is FORBIDDEN by SPEC 1.1: it
produced an indefinite charge block in audit.  It is not implemented here and
must not be added.

The hardness floor (SPEC 1.2, corrected in SPEC 17.3)
-----------------------------------------------------
``J`` INCLUDING its true diagonal ``J_ii = J(0; sigma_ii)`` is the Gram matrix
of the smeared charge densities in the Coulomb metric, so it is positive
SEMI-definite for every geometry, and zero-eigenvalued when two sites
coincide.  Writing the charge block as ``J + diag(eta_i - J_ii)`` gives,
exactly,

    diag(eta) + J_off  >=  min_i (eta_i - S_0(sigma_i))   for EVERY geometry,
    S_0(sigma) = J(0; sqrt(2) sigma) = sqrt(2) K_E / (sigma sqrt(pi)),

because the self width is ``combine_sigma(sigma, sigma) = sqrt(2) sigma``.
``S_0`` is :func:`gram_self_energy`, 7.659503 eV at sigma = 1.5 A, and the
bound is tight: two coincident sites give min eig +2.1e-15 at eta = 7.659503
and -1.0e-02 at eta = 7.649503.

The eta head emits ``softplus(raw) + S_0(sigma) + margin`` (:func:`hardness_floor`,
margin :data:`DEFAULT_HARDNESS_MARGIN` = 1 eV/e^2), so the charge block's
smallest eigenvalue is at least the margin whatever the geometry -- a stated,
geometry-independent distance from round-off.

The package used to put the floor at ``kernel_self(sigma) = 2 K_E / (sigma
sqrt(pi)) = sqrt(2) S_0`` and to describe everything below it as "unbounded
below".  That was wrong in both directions: eta = 8 eV at sigma = 1.5 A (below
the old 10.83 eV "floor") is positive definite over any cluster (measured min
eig +0.34 over 200 dense 30-atom clusters), and the multiplicative sqrt(2)
forced ``eta sigma >= 16.25 eV A`` -- hydrogen at its Mulliken hardness of
~12.8 eV could only be represented with sigma >= 1.27 A.  :func:`kernel_self`
keeps its other meaning, the ``r -> 0`` limit of :func:`coulomb_kernel` at a
PAIR width.

Dipole-dipole: Thole exponential damping
----------------------------------------
Bare point dipoles suffer the polarization catastrophe at short range.  Thole's
exponential smearing replaces the point dipole by

    rho(u) = (3 a / 4 pi) exp(-a u^3) ,   u = r_ij / (alpha_i alpha_j)^(1/6)

giving the damping factors ``lambda_3, lambda_5`` of :func:`thole_damping`.
For this form ``sup_r alpha |T| = a`` exactly, so the relay matrix
``alpha^-1 - T`` is positive definite for all separations and all
polarizabilities if and only if ``a < 1``.  ``a >= 1`` raises; the recommended
value is ``a = 0.39`` (2.5x margin, the AMOEBA value).

NOTE on the units of ``alpha`` in the Thole reduced distance: ``u`` is
dimensionless only if ``alpha`` is a polarizability VOLUME in A^3.  A response
polarizability in e^2 A^2 / eV must be converted first -- see
:func:`trace_electrostatics.constants.to_polarizability_volume` and SPEC 0.

Sign convention (read before wiring these into the energy)
----------------------------------------------------------
:func:`charge_dipole_tensor` and :func:`dipole_interaction_tensor` return the
tensors **as they enter the energy** ``U = 0.5 m^T K m``, not the field-gradient
tensors.  With ``r_vec = r_i - r_j``:

    U(q_i, mu_j) = q_i * T01(r_vec) . mu_j
    U(mu_i, mu_j) = mu_i . T(r_vec) . mu_j          (unordered pair, counted once)

``T`` is therefore minus the conventional propagator ``grad grad (1/r)``; it is
the block that sits inside ``A_mu + T_Thole`` in SPEC 1.5 with a plus sign.

Quadrupoles: one generating identity, and one convention
--------------------------------------------------------
Everything in this module is a derivative of the same damped potential.  Write
``phi_d(r_vec)`` for the Thole-damped ``1/r`` and ``d_a`` for ``d/d r_vec_a``
with ``r_vec = r_i - r_j``; then the entire interaction of two multipole sets is

    U / K_E = [ q_i + mu_i.d + (1/2) Theta_i:dd ]
              [ q_j - mu_j.d + (1/2) Theta_j:dd ] phi_d

-- the sign asymmetry is real and is just ``d/dr_i = -d/dr_j`` acting on the
site whose moments are being expanded.  Reading the six blocks off that product
gives every tensor here, including the two that already existed, and fixes all
of their signs at once:

    J   = K_E phi_d                         T01_a  = -K_E d_a phi_d
    T_ab = -K_E d_a d_b phi_d               T02_ab = +(K_E/2) d_a d_b phi_d
    T12_abc = +(K_E/2) d_a d_b d_c phi_d    T22_abcd = +(K_E/4) d_a d_b d_c d_d phi_d

so ``T02 = -T/2`` exactly, with the same damping -- a relation the tests assert
rather than a coincidence (measured: difference exactly 0.0).  The odd-rank
tensors change sign when ``i`` and ``j`` are exchanged and the even-rank ones do
not, which is what makes ``K`` symmetric.

ONE IDENTITY, TWO DAMPED POTENTIALS.  ``phi_d`` above is not a single function
across the whole table: the charge-charge entry is the Gaussian-smeared
potential ``erf(r/sigma_ij)/r`` and every entry from ``T01`` down is the
Thole-smeared one of :func:`thole_damping_quadrupole`.  The generating identity
holds *within each sector*, with that sector's own ``phi_d``; it is what fixes
the relative prefactors and signs, not a claim that one smeared density
produces all six.  That split is the standard AMOEBA-style arrangement -- charge
penetration is a property of the charge cloud, the polarization catastrophe a
property of the induced dipoles -- and mixing the two would give ``T01`` a
damping inconsistent with the dipole block it multiplies.

Every derivative above was checked numerically, not just derived.  Against
50-digit derivatives of the closed-form damped potential
``phi_d(r) = 1/r - (c^(1/3)/3) Gamma(-1/3, c r^3)``, ``c = a/sqrt(alpha_i
alpha_j)``, over three geometries times three polarizability pairs (``v = 0.069``
to ``v = 80.7``, spanning both series cutovers and the saturation), every
Cartesian component of all five tensors agrees to

    T01 5.7e-16   T 3.1e-14   T02 3.1e-14   T12 3.0e-14   T22 3.3e-14

relative, which is float64 round-off on the tensors themselves.  The same check
rejects every misassignment of a damping factor by 16 % to 3000 % -- see
:func:`thole_damping_quadrupole`, where the assignment is derived.

THE QUADRUPOLE CONVENTION, stated once because the literature does not agree.
``Theta`` here is the **traceless second moment**

    Theta_ab = sum_k q_k ( r_a r_b - (1/3) |r|^2 delta_ab )        [e A^2]

and the energy of such a quadrupole in an external potential is
``U = (1/2) Theta_ab d_a d_b phi``.  Buckingham's ``Theta^B``, which is what
most tables of molecular quadrupole moments report, is **3/2 times this one**
(``Theta^B = (1/2) sum q (3 r_a r_b - r^2 delta_ab)``), so a literature value in
Buckingham convention must be multiplied by 2/3 before it is contracted with
anything in this module.  A factor of 3/2 in a quadrupole is invisible to every
symmetry, equivariance and reciprocity test there is; only a dimensioned
comparison catches it, which is why the definition is written out here and
verified against explicit point-charge clusters.

How the convention was verified, and how loud the wrong one is.  Take clusters
whose lower moments vanish by symmetry, so that exactly one tensor is the
leading term of the exact Coulomb sum, and watch the residual's power law:

| clusters                                    | tensor | leading | residual |
|---|---|---|---|
| point charge vs (+d, -2d, +d) linear quadrupole | T02 | R^-3 | **R^-5.000** |
| (+q, -q) pair vs linear quadrupole              | T12 | R^-4 | **R^-6.001** |
| two linear quadrupoles, random orientations     | T22 | R^-5 | **R^-6.997** |

Each residual is exactly the first omitted rank pair -- ``(0,4)``, ``(1,4)``,
``(2,4)``; the odd moments of a centrosymmetric cluster vanish -- so each tensor
is the exact leading term with the right prefactor, the right sign and a
complete permutation list.  Substituting ``Theta^B`` for ``Theta`` instead gives
(measured, dipole vs quadrupole at R = 10 A) ``+1.633e-05 eV`` against an exact
``+1.085e-05 eV``: a **50 % energy error** that no rotation, parity, trace or
reciprocity test sees.

Because ``Theta`` is traceless, adding any multiple of ``delta_ab f(r)`` to
``T02`` -- or of ``delta_ab r_c`` to ``T12`` -- would not change a single
energy.  Measured: adding ``137 * delta_ab`` to ``T02`` moves a ``q : Theta``
energy of ``5.480535e-01 eV`` by ``6.4e-15 eV``.  The functions below
nevertheless return the exact derivatives above, traces and all, so that they
remain the derivatives they claim to be and can be differentiated, contracted or
compared term by term without a hidden gauge.

THE DAMPED T02 IS NOT TRACELESS, AND MUST NOT BE "FIXED".  In the undamped
limit ``tr T02 = 0``, because ``1/r`` is harmonic away from the origin.  The
damped one is not, and its trace is not an error term -- it is Poisson's
equation.  ``lambda_3`` sits on the ``delta`` term and ``lambda_5`` on the
``r_a r_b`` term, so

    tr T02 = (K_E/2) * 3 (lambda_5 - lambda_3) / r^3
           = -(3 K_E / 2) v e^-v / r^3
           = (K_E/2) * grad^2 phi_d = -(K_E/2) * 4 pi rho(r) ,

exactly the smeared Thole density evaluated at ``r``, because the field point is
*inside* the other site's cloud.  Verified to 7.9e-15 relative against
``-4 pi rho`` at ``v = 4.54``, and the same identity one and two derivatives
further up holds for the traces of ``T12`` (``tr_bc T12_abc = grad_a grad^2
phi_d``, dev 4.4e-16) and ``T22`` (dev 2.1e-16).  A "traceless" version of these
tensors would no longer be the derivative of any potential, would break the
finite-difference identity the tests rest on, and would change no energy
whatsoever -- because ``Theta`` is traceless and contracts the trace part away.
There is nothing to gain and a derivation to lose.
"""

from __future__ import annotations

import math

from math import factorial
from typing import Final

import torch
from torch import Tensor

from .constants import K_E, SQRT_PI

__all__ = [
    "DEFAULT_THOLE_A",
    "THOLE_A_MAX",
    "combine_sigma",
    "kernel_self",
    "gram_self_energy",
    "hardness_floor",
    "DEFAULT_HARDNESS_MARGIN",
    "coulomb_kernel",
    "thole_damping",
    "thole_damping_quadrupole",
    "dipole_interaction_tensor",
    "charge_dipole_tensor",
    "charge_quadrupole_tensor",
    "dipole_quadrupole_tensor",
    "quadrupole_interaction_tensor",
    "DEFAULT_GAUSSIAN_A",
    "GAUSSIAN_A_MAX",
    "GAUSSIAN_DIPOLE_SELF_VOLUME",
    "GAUSSIAN_QUADRUPOLE_SELF_VOLUME",
    "gaussian_damping",
    "gaussian_self_stiffness",
    "gaussian_multipole_width",
    "quadrupole_cartesian_basis",
]

#: Recommended Thole screening parameter (AMOEBA value).  Note the bound
#: ``thole_a < 1`` guarantees a positive definite relay for a PAIR of sites
#: only; three or more close sites can be indefinite below it (SPEC 1.6).
DEFAULT_THOLE_A: Final[float] = 0.39

#: Guaranteed excess of the on-site hardness over the tight Gram bound
#: ``S_0(sigma)``, in eV/e^2 (SPEC 17.3): the charge block's smallest
#: eigenvalue is at least this for every geometry.
DEFAULT_HARDNESS_MARGIN: Final[float] = 1.0

#: Strict upper bound on the Thole parameter: at a = 1 the relay catastrophes.
THOLE_A_MAX: Final[float] = 1.0

#: Below this reduced distance x = r / sigma the Taylor series for erf(x) / x is
#: both more accurate and better conditioned than the direct quotient.  With the
#: coefficients below the truncation error at the cutover is ~1e-16 relative.
ERF_SERIES_CUTOVER: Final[float] = 0.1

#: erf(x) / x = (2 / sqrt(pi)) * sum_n (-1)^n x^(2n) / (n! (2n + 1)).
#: Coefficients of x^0, x^2, ... x^10, evaluated by Horner in x^2.
ERF_OVER_X_COEFFS: Final[tuple[float, ...]] = (
    1.0,
    -1.0 / 3.0,
    1.0 / 10.0,
    -1.0 / 42.0,
    1.0 / 216.0,
    -1.0 / 1320.0,
)

#: Below this ``v = a u^3`` the direct form of lambda_5 loses digits to
#: cancellation (1 - (1 + v) e^-v is O(v^2) built from terms of O(1)), so the
#: series is used instead.  Worst-case relative error at the cutover ~7e-14.
THOLE_SERIES_CUTOVER: Final[float] = 0.05

#: 1 - (1 + v) exp(-v) = sum_{n>=2} (-1)^n (n - 1) v^n / n!.
#: Coefficients of v^2 ... v^8, evaluated by Horner in v.
THOLE_LAMBDA5_COEFFS: Final[tuple[float, ...]] = (
    1.0 / 2.0,
    -1.0 / 3.0,
    1.0 / 8.0,
    -1.0 / 30.0,
    1.0 / 144.0,
    -1.0 / 840.0,
    1.0 / 5760.0,
)

#: Below this ``v`` the direct forms of lambda_7 and lambda_9 lose too much to
#: cancellation and the series below is used instead.  They cancel far harder
#: than lambda_5 does -- lambda_5 starts at ``v^2 / 2``, lambda_7 at
#: ``-v^2 / 10`` and lambda_9 at ``-v^2 / 70``, all assembled from terms of
#: O(1) -- so the cutover sits four times higher than ``THOLE_SERIES_CUTOVER``
#: and the series runs to ``v^12``.  Measured worst-case relative error against
#: 50-digit arithmetic: lambda_7 5.2e-15 and lambda_9 4.8e-14 at the cutover
#: from below, 4.6e-15 and 1.4e-13 from above (the direct form's own
#: cancellation, which is what the cutover trades against).
THOLE_HIGH_SERIES_CUTOVER: Final[float] = 0.2

#: Highest power of ``v`` kept in the lambda_7 / lambda_9 series.
THOLE_HIGH_SERIES_ORDER: Final[int] = 12

#: 1 - (1 + v + (3/5) v^2) e^-v = sum_{n>=2} (-1)^(n+1) (n-1)(3n-5) v^n / (5 n!).
#: Collect the coefficient of v^n in ``(1 + v + (3/5) v^2) e^-v`` over the common
#: denominator ``5 n!``: ``5 - 5n + 3n(n-1) = (n-1)(3n-5)``.  Coefficients of
#: v^2 ... v^12, evaluated by Horner in v.
THOLE_LAMBDA7_COEFFS: Final[tuple[float, ...]] = tuple(
    (-1.0) ** (n + 1) * (n - 1) * (3 * n - 5) / (5.0 * factorial(n))
    for n in range(2, THOLE_HIGH_SERIES_ORDER + 1)
)

#: 1 - (1 + v + (18/35) v^2 + (9/35) v^3) e^-v
#:     = sum_{n>=2} (-1)^(n+1) (-9n^3 + 45n^2 - 71n + 35) v^n / (35 n!),
#: by the same collection over the common denominator ``35 n!``.  Coefficients
#: of v^2 ... v^12.
THOLE_LAMBDA9_COEFFS: Final[tuple[float, ...]] = tuple(
    (-1.0) ** (n + 1)
    * (-9 * n**3 + 45 * n**2 - 71 * n + 35)
    / (35.0 * factorial(n))
    for n in range(2, THOLE_HIGH_SERIES_ORDER + 1)
)

#: Separations below this (in A) are treated as coincident sites.  The dipole
#: tensors are set to zero there so that no NaN can escape; physically this is a
#: degenerate configuration that a neighbour list never produces.  It is a NaN
#: shield, not a physical regularisation -- it is TWELVE orders of magnitude
#: below any distance the model will ever see (bond lengths are ~1 A).  The
#: shield is a discontinuity, not a limit; see :func:`dipole_interaction_tensor`.
DEGENERATE_SEPARATION: Final[float] = 1e-12

#: ``v = a u^3`` above which ``1 - (1 + v) exp(-v)`` is 1.0 to the last bit of a
#: float64, so :func:`thole_damping` saturates ``v`` here.  Verified
#: bit-identical to the unclamped form for every finite ``v`` (max difference
#: exactly 0.0 over v in [0.05, 80] plus 1e3 and 1e10); what it buys is that
#: ``v = inf``, which ``alpha_i alpha_j = 0`` produces, yields lambda_5 = 1.0
#: instead of ``1 - inf * 0 = NaN``.
THOLE_V_SATURATION: Final[float] = 64.0


def _erf_over_x_normalised(x_squared: Tensor) -> Tensor:
    """``P(x^2) = (sqrt(pi) / 2) erf(x) / x``, the series normalised to ``P(0) = 1``.

    Factoring out the ``x -> 0`` limit is what lets :func:`coulomb_kernel`
    return a value bit-identical to :func:`kernel_self` at ``r = 0``: the
    polynomial is exactly 1.0 there, so no rounding of a different but
    algebraically equal expression can creep in.
    """
    series = torch.full_like(x_squared, ERF_OVER_X_COEFFS[-1])
    for coefficient in reversed(ERF_OVER_X_COEFFS[:-1]):
        series = series * x_squared + coefficient
    return series


def combine_sigma(sigma_i: Tensor, sigma_j: Tensor) -> Tensor:
    """Width of the overlap of two Gaussian charge clouds.

        sigma_ij = sqrt(sigma_i^2 + sigma_j^2)

    The convolution of two normalised Gaussians of widths ``sigma_i`` and
    ``sigma_j`` is a Gaussian of width ``sigma_ij``, which is why the pair
    kernel :func:`coulomb_kernel` depends on the two widths only through this
    combination.

    Note ``combine_sigma(sigma, sigma) = sqrt(2) sigma``: the *self* width of an
    atom is not its own ``sigma``, which is the whole origin of the sqrt(2) in
    the module docstring's hardness-floor discussion.

    PRECONDITION -- both widths strictly positive.  It is a caller obligation,
    not a checked one: ``sigma`` comes from a softplus head and is positive by
    construction, and a ``(sigma > 0).all()`` test here would be a host sync and
    a data-dependent branch, which forbids ``torch.compile(fullgraph=True)`` and
    with it the AOT path of SPEC section 2 (measured: such a test raises
    ``Unsupported: call_function BuiltinVariable(bool)``).

    Violating it is therefore handled the way :func:`thole_damping` handles a
    zero polarizability -- the same validation policy, for the same reason, and
    this function used to be the one place that disagreed with it.  ``a`` is a
    Python float and raises; a tensor cannot, so the degenerate input is made
    non-poisoning instead.  ``combine_sigma(0, 0)`` returns 0.0 with a **zero**
    gradient, not the ``NaN`` that ``d sqrt(x)/dx`` at zero produces, and a NaN
    can no longer appear in the sigma head's gradient from here.

    The zero is a shield, not a limit: the true one-sided derivative at
    ``sigma = 0`` is ``+inf``, and no finite number is the right answer for an
    input the model cannot physically produce.  What the shield buys is that a
    single degenerate site cannot silently NaN the gradient of every other
    parameter in the batch, which is what a ``NaN`` in a summed loss does.
    Sanitising the *input* of the sqrt rather than masking its output is the
    same rule as in :func:`coulomb_kernel`: ``torch.where`` sends a zero
    cotangent into the untaken branch and ``0 * inf = NaN``, so masking the
    output would leave the NaN exactly where it was.

    Verified: ``combine_sigma(1.5, 1.5) = 2.1213203435596424`` with gradient
    ``1.414213562373095`` (bit-identical to ``sqrt(2)``), so the guard costs
    nothing on any physical input.
    """
    sum_of_squares = sigma_i * sigma_i + sigma_j * sigma_j
    positive = sum_of_squares > 0.0
    safe = torch.where(positive, sum_of_squares, torch.ones_like(sum_of_squares))
    return torch.where(positive, torch.sqrt(safe), torch.zeros_like(sum_of_squares))


def gram_self_energy(sigma: Tensor) -> Tensor:
    """``S_0 = sqrt(2) K_E / (sigma sqrt(pi))``, the tight hardness bound (SPEC 17.3).

    The Coulomb self-energy of one Gaussian charge cloud of width ``sigma``:
    the diagonal of the charge Gram matrix, ``J(0; sqrt(2) sigma)``.  The
    charge block ``diag(eta) + J_off`` is positive semi-definite for every
    geometry iff ``eta_i >= S_0(sigma_i)`` (necessary: two coincident sites).
    7.659503 eV at sigma = 1.5 A.
    """
    return (math.sqrt(2.0) * K_E) / (sigma * SQRT_PI)


def hardness_floor(sigma: Tensor, margin: float = DEFAULT_HARDNESS_MARGIN) -> Tensor:
    """``S_0(sigma) + margin``: what the eta head adds to ``softplus(raw)``."""
    return gram_self_energy(sigma) + margin


def kernel_self(sigma: Tensor) -> Tensor:
    """``r -> 0`` limit of the pair kernel at width ``sigma``, and the SPEC 1.2 floor.

        kernel_self(sigma) = 2 K_E / (sigma sqrt(pi))     [eV / e^2]
                           = 10.832173 eV at sigma = 1.5 A

    The expression has two distinct jobs in this package, and which one is meant
    is decided entirely by the ARGUMENT:

    * ``kernel_self(sigma_ij)`` -- a *pair* width -- is the removable-singularity
      limit of :func:`coulomb_kernel`, which is how that function uses it.
    * ``kernel_self(sigma_i)`` -- an *atom's own* width -- is the hardness floor
      the eta head must add to ``softplus(raw)`` (SPEC 1.2, mandatory).

    It is **not** the self-repulsion of one Gaussian cloud of width ``sigma``,
    and it is no longer the hardness floor (SPEC 17.3).  The self-repulsion is
    the pair kernel at the self width ``sqrt(2) sigma``,
    :func:`gram_self_energy` = 7.659503 eV at 1.5 A, which is also the tight
    positive-definiteness bound; the eta head adds that plus an explicit margin
    (:func:`hardness_floor`).  This function is the ``r -> 0`` limit of
    :func:`coulomb_kernel` at a PAIR width ``sigma_ij``.
    """
    return (2.0 * K_E) / (sigma * SQRT_PI)


def coulomb_kernel(r: Tensor, sigma_ij: Tensor) -> Tensor:
    """Gaussian-damped Coulomb kernel between two smeared charges.

        J(r) = K_E * erf(r / sigma_ij) / r       [eV / e^2]

    Limits, both of which this implementation reproduces exactly:

        r -> 0    J -> 2 K_E / (sigma_ij sqrt(pi)) = kernel_self(sigma_ij)
        r -> inf  J -> K_E / r, approached as K_E erfc(r / sigma_ij) / r

    The value and the gradient are finite at ``r = 0``; ``dJ/dr = 0`` there,
    because ``J`` is an even, smooth function of ``r``.  ``sigma_ij`` must be
    strictly positive.

    The removable singularity is handled by a Taylor series below
    ``x = r / sigma_ij = ERF_SERIES_CUTOVER``.  Masking the *output* of the
    quotient would not be enough: ``torch.where`` sends a zero cotangent into
    the untaken branch, and ``0 * inf = NaN``, so a division by zero hiding in
    the discarded branch still poisons the gradient.  Both branches are
    therefore given sanitised *inputs* -- the quotient never sees a small ``x``,
    the series never sees a large one -- and only then combined.

    Args:
        r: Separations, any shape.
        sigma_ij: Pair widths from :func:`combine_sigma`, broadcastable to ``r``.
    """
    x = r / sigma_ij
    small = x < ERF_SERIES_CUTOVER
    unit = torch.ones_like(x)

    # Near branch: J(0) * P(x^2), which is exactly kernel_self(sigma_ij) at r = 0.
    # x^2 is zeroed outside the series region so that a large argument cannot
    # overflow x^10 in a term that is discarded anyway.
    x_squared = torch.where(small, x * x, torch.zeros_like(x))
    near = kernel_self(sigma_ij) * _erf_over_x_normalised(x_squared)

    # Far branch: the plain quotient, with both r and x replaced by constants
    # where the series is used, so neither its value nor its derivative diverges.
    x_far = torch.where(small, unit, x)
    r_far = torch.where(small, unit, r)
    far = K_E * torch.erf(x_far) / r_far

    return torch.where(small, near, far)


def thole_damping(
    r: Tensor,
    alpha_i: Tensor,
    alpha_j: Tensor,
    a: float = DEFAULT_THOLE_A,
) -> tuple[Tensor, Tensor]:
    """Thole exponential damping factors for the dipole field tensors.

    Smearing each point dipole over ``rho(u) = (3 a / 4 pi) exp(-a u^3)`` with
    the reduced distance ``u = r / (alpha_i alpha_j)^(1/6)`` replaces the ``1/r^3``
    and ``1/r^5`` terms of the propagator by damped versions.  Writing
    ``v = a u^3 = a r^3 / sqrt(alpha_i alpha_j)``:

        lambda_3 = 1 - exp(-v)
        lambda_5 = 1 - (1 + v) exp(-v)

    Short range behaviour is the whole point: ``lambda_3 = O(r^3)`` and
    ``lambda_5 = O(r^6)`` kill the ``1/r^3`` and ``1/r^5`` divergences, so the
    damped tensors stay finite as ``r -> 0`` and the polarization catastrophe
    cannot occur.  Quantitatively ``sup_r alpha |T| = a`` exactly, which makes
    the relay matrix positive definite for every separation and every
    polarizability iff ``a < 1`` (SPEC 1.6).

    Args:
        r: Separations in A.
        alpha_i, alpha_j: Polarizability **volumes in A^3** (not response
            polarizabilities in e^2 A^2 / eV -- see SPEC 0 and
            :func:`trace_electrostatics.constants.to_polarizability_volume`).
            Strictly positive; see the validation policy below.
        a: Screening parameter, ``0 < a < 1``.  Default 0.39.

    Returns:
        ``(lambda_3, lambda_5)``, broadcast to the shape of the inputs.

    Raises:
        ValueError: if ``a >= 1`` (the relay loses positive definiteness) or
            ``a <= 0`` (no damping / unphysical sign).

    Validation policy -- why only ``a`` raises
    ------------------------------------------
    ``a`` is a Python float, so checking it is free and it raises.
    ``alpha_i`` and ``alpha_j`` are tensors, and a ``(alpha > 0).all()`` test is
    a host sync and a data-dependent branch: it breaks
    ``torch.compile(fullgraph=True)`` (measured: ``Unsupported: call_function
    BuiltinVariable(bool)``) and with it the AOT path SPEC section 2 requires.
    They are therefore a caller obligation -- the polarizability head emits a
    softplus, so they are positive by construction -- and the degenerate input
    is made to return the correct limit instead of raising:
    ``alpha_i alpha_j -> 0`` sends ``v -> inf``, which is the completely
    undamped limit, and both factors saturate at 1.0.  Before
    ``THOLE_V_SATURATION`` was introduced ``lambda_3`` returned 1.0 there while
    ``lambda_5`` returned ``1 - inf * 0 = NaN``, which is the inconsistency this
    note exists to record.  The *gradient* with respect to a polarizability that
    is exactly zero is still NaN, in both factors; that input is unphysical and
    is not made differentiable.
    """
    v = _thole_reduced(r, alpha_i, alpha_j, a)
    return _thole_lambda_3_5(v)


def _validate_thole_a(a: float) -> None:
    """SPEC 1.6: ``0 < a < 1``, strictly.  Shared by every Thole entry point."""
    if not a < THOLE_A_MAX:
        raise ValueError(
            f"Thole screening parameter a must be strictly less than {THOLE_A_MAX} "
            f"(got {a}): sup_r alpha|T| = a, so at a >= 1 the dipole relay is not "
            "positive definite and the polarization catastrophe returns."
        )
    if not a > 0.0:
        raise ValueError(f"Thole screening parameter a must be strictly positive (got {a}).")


def _thole_reduced(r: Tensor, alpha_i: Tensor, alpha_j: Tensor, a: float) -> Tensor:
    """``v = a u^3 = a r^3 / sqrt(alpha_i alpha_j)``, saturated.

    v = a u^3 with u = r / (alpha_i alpha_j)^(1/6); the sixth root and the cube
    collapse into a square root, which is cheaper and better conditioned.

    The saturation is the top end of the "sanitise the input" discipline used
    throughout this module: every lambda is already 1.0 to the last bit at
    v = 64 (the largest correction is lambda_9's, ``1.1e-23``, which is 1.0 in
    float64), so clamping there is bit-identical for every finite v -- verified,
    max difference exactly 0.0.

    Clamping alone is NOT enough for ``alpha_i alpha_j = 0``, and this is the
    module's own trap turned on itself.  The clamp does give the right VALUE
    there -- ``v = inf`` saturates to 64, hence ``lambda = 1.0``, the correct
    undamped limit rather than ``1 - inf * 0 = NaN`` -- but it does nothing for
    the derivative: ``clamp`` passes a zero cotangent above its maximum, and
    ``0 * inf = NaN``.  Measured at ``alpha_i = alpha_j = 0``, ``r = 1.5``,
    ``a = 0.39``: ``lambda_3 = lambda_5 = 1.0`` with ``d/d alpha``, ``d/dr`` and
    ``d2/dr2`` all ``nan``.  So the degenerate DENOMINATOR is substituted before
    the square root, exactly as :func:`combine_sigma` substitutes before its
    own, and the saturated value is then selected by branch rather than reached
    through an infinity.  All three derivatives are 0.0 afterwards.

    Bit-identical for every ``alpha_i alpha_j > 0``: the product passes through
    ``torch.where`` unchanged, so the division and the clamp are the ones above.
    Production never reaches the degenerate branch anyway --
    :func:`~trace_electrostatics.solver.isotropic_polarizability_volume` and
    ``energy._direct_pair_blocks`` both floor the volume at
    ``MIN_POLARIZABILITY_VOLUME = 1e-12``, where the derivatives are already
    0.0 -- but this is a public entry point and a caller may pass zero.
    """
    _validate_thole_a(a)
    product = alpha_i * alpha_j
    positive = product > 0.0
    safe = torch.where(positive, product, torch.ones_like(product))
    saturation = torch.full_like(safe, THOLE_V_SATURATION)
    reduced = torch.where(positive, a * (r * r * r) / torch.sqrt(safe), saturation)
    return torch.clamp(reduced, max=THOLE_V_SATURATION)


def _damped_series(
    v: Tensor, small: Tensor, coefficients: tuple[float, ...]
) -> Tensor:
    """``sum_n c_n v^n`` starting at ``v^2``, evaluated only where ``small``.

    ``v`` is zeroed outside the series region first.  That is the same
    sanitise-the-input rule as everywhere else in this file and it is not
    optional: the caller combines this with a direct form through
    ``torch.where``, which sends a zero cotangent into the untaken branch, and a
    high power of a large ``v`` there would contribute a finite value to a
    discarded branch and an enormous derivative to the kept one.
    """
    v_series = torch.where(small, v, torch.zeros_like(v))
    series = torch.full_like(v, coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        series = series * v_series + coefficient
    return series * v_series * v_series


def _thole_lambda_3_5(v: Tensor) -> tuple[Tensor, Tensor]:
    """``(lambda_3, lambda_5)`` from the reduced distance ``v``."""
    # expm1 keeps the small-v end of lambda_3 accurate to full precision.
    lambda_3 = -torch.expm1(-v)

    # lambda_5 = 1 - (1 + v) e^-v is O(v^2) assembled from O(1) terms, so the
    # direct form loses all significance for small v.  Both branches are given
    # sanitised inputs before being combined, for the same gradient reason as in
    # `_erf_over_x_normalised`.
    small = v < THOLE_SERIES_CUTOVER
    v_direct = torch.where(small, torch.zeros_like(v), v)
    lambda_5_direct = 1.0 - (1.0 + v_direct) * torch.exp(-v_direct)

    lambda_5_series = _damped_series(v, small, THOLE_LAMBDA5_COEFFS)

    lambda_5 = torch.where(small, lambda_5_series, lambda_5_direct)
    return lambda_3, lambda_5


def thole_damping_quadrupole(
    r: Tensor,
    alpha_i: Tensor,
    alpha_j: Tensor,
    a: float = DEFAULT_THOLE_A,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``(lambda_3, lambda_5, lambda_7, lambda_9)`` -- Thole damping to 4th order.

    A superset of :func:`thole_damping`: the first two return values are
    bit-identical to that function's, from the same code.  The higher two are
    what the quadrupole tensors need, and they come from the *same* smeared
    density -- they are not a second model bolted on:

        rho(u) = (3 a / 4 pi) exp(-a u^3),   u = r / (alpha_i alpha_j)^(1/6)

    Derivation, because the factors 3/5, 18/35 and 9/35 are not guessable.
    Gauss's law on that density gives the enclosed charge
    ``Q(r) = 1 - e^-v`` with ``v = a u^3``, hence ``-d phi_d / dr = Q(r) / r^2``.
    Differentiating ``phi_d`` and using ``dv/dr = 3v/r`` repeatedly:

        d_a phi_d         = -lambda_3 r_a / r^3
        d_a d_b phi_d     =  3 lambda_5 r_a r_b / r^5 - lambda_3 delta_ab / r^3
        d_a d_b d_c phi_d = -15 lambda_7 r_a r_b r_c / r^7
                            + 3 lambda_5 (delta_ab r_c + 2 perms) / r^5
        d_a..d_d phi_d    =  105 lambda_9 r_a r_b r_c r_d / r^9
                            - 15 lambda_7 (delta_ab r_c r_d + 5 perms) / r^7
                            + 3 lambda_5 (delta_ab delta_cd + 2 perms) / r^5

    where each new lambda is forced by requiring the result to be symmetric in
    all its indices -- the ``delta`` terms of one order and the ``r r`` terms of
    the next have to share a coefficient, which is exactly the recursion

        lambda_5 = lambda_3 - v e^-v
        lambda_7 = lambda_5 - (3/5) v^2 e^-v
        lambda_9 = lambda_7 + (3/35) v^2 e^-v - (9/35) v^3 e^-v

    giving

        lambda_3 = 1 - e^-v
        lambda_5 = 1 - (1 + v) e^-v
        lambda_7 = 1 - (1 + v + (3/5) v^2) e^-v
        lambda_9 = 1 - (1 + v + (18/35) v^2 + (9/35) v^3) e^-v

    All four were checked against 30-digit derivatives of the closed form
    ``phi_d(r) = 1/r - (c^(1/3)/3) Gamma(-1/3, c r^3)``, ``c = a / sqrt(alpha_i
    alpha_j)`` (substitute ``t = c s^3`` in the Gauss-law integral), at three
    separations spanning ``v = 0.012`` to ``v = 12``: worst relative deviation
    over ten independent mixed partials **3.2e-26**.

    Short range.  ``lambda_7`` and ``lambda_9`` are both ``O(v^2) = O(r^6)`` and
    both start out **negative** (``-v^2/10`` and ``-v^2/70``), which is not a
    sign error: the damped potential has a source inside the cloud, so its
    higher derivatives are not the traceless ``1/r`` ones.  Combined with the
    ``1/r^7`` and ``1/r^9`` prefactors they leave ``T12`` and ``T22`` vanishing
    linearly in ``r``, so both tensors are continuous at contact -- unlike the
    dipole-dipole tensor, whose limit is finite and non-zero.

    Args:
        r: Separations in A.
        alpha_i, alpha_j: Polarizability **volumes in A^3** (SPEC 0), strictly
            positive; the validation policy is :func:`thole_damping`'s.
        a: Screening parameter, ``0 < a < 1``.  Default 0.39.

    Raises:
        ValueError: if ``a >= 1`` or ``a <= 0``, as :func:`thole_damping`.
    """
    v = _thole_reduced(r, alpha_i, alpha_j, a)
    lambda_3, lambda_5 = _thole_lambda_3_5(v)

    # Same structure as lambda_5's, with a cutover four times higher because the
    # cancellation is four to thirty times worse; see THOLE_HIGH_SERIES_CUTOVER.
    small = v < THOLE_HIGH_SERIES_CUTOVER
    v_direct = torch.where(small, torch.zeros_like(v), v)
    decay = torch.exp(-v_direct)
    v_direct_squared = v_direct * v_direct

    lambda_7_direct = 1.0 - (1.0 + v_direct + 0.6 * v_direct_squared) * decay
    lambda_9_direct = 1.0 - (
        1.0
        + v_direct
        + (18.0 / 35.0) * v_direct_squared
        + (9.0 / 35.0) * v_direct_squared * v_direct
    ) * decay

    lambda_7 = torch.where(
        small, _damped_series(v, small, THOLE_LAMBDA7_COEFFS), lambda_7_direct
    )
    lambda_9 = torch.where(
        small, _damped_series(v, small, THOLE_LAMBDA9_COEFFS), lambda_9_direct
    )
    return lambda_3, lambda_5, lambda_7, lambda_9


def _safe_inverse_powers(r: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(finite, 1/r^3, 1/r^5)`` with no NaN at coincident sites.

    ``finite`` is the boolean mask of separations the caller may trust.  As in
    :func:`_erf_over_x_normalised`, the division is protected by substituting the
    *input*, so the untaken branch contributes neither an infinite value nor an
    infinite derivative.
    """
    finite = r > DEGENERATE_SEPARATION
    r_safe = torch.where(finite, r, torch.ones_like(r))
    inv_r = 1.0 / r_safe
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r * inv_r2
    return finite, inv_r3, inv_r3 * inv_r2


def dipole_interaction_tensor(
    r_vec: Tensor,
    r: Tensor,
    lam3: Tensor,
    lam5: Tensor,
) -> Tensor:
    """Thole-damped dipole-dipole tensor, in the energy convention.

        T_ab(r_vec) = K_E * ( lambda_3 delta_ab / r^3
                              - 3 lambda_5 r_a r_b / r^5 )   [eV / (e A)^2]

    so that the interaction of two dipoles (unordered pair, counted once) is

        U = mu_i . T . mu_j

    This is minus the usual propagator ``grad grad (1/r)``: it is the block
    that enters ``A_mu + T_Thole`` of ``U = 0.5 m^T K m`` with a plus sign
    (SPEC 1.5), so no sign flip is needed when assembling ``K``.  ``T`` is
    symmetric in its two Cartesian indices and invariant under ``r_vec -> -r_vec``,
    as it must be for ``K`` to be symmetric.

    With ``lam3, lam5`` from :func:`thole_damping` the ``r -> 0`` limit is
    finite and non-zero: ``lam3 = O(r^3)`` and ``lam5 = O(r^6)`` leave
    ``T_ab -> K_E a delta_ab / sqrt(alpha_i alpha_j)``, the anisotropic part
    vanishing.

    The ``DEGENERATE_SEPARATION`` shield does NOT take that limit -- it returns
    zero, so it is a JUMP of the full limiting value, not a continuous
    extension.  Measured at ``alpha_i = alpha_j = 1 A^3``, ``a = 0.39``, where
    the limit is ``K_E a = 5.615862``: ``T_yy = +5.615862`` at
    ``r = 1.0000001e-12 A`` and ``0.0`` at ``r = 1.0e-12 A``.  This is
    deliberate and is why the threshold sits twelve orders of magnitude below
    any physical separation: a discontinuity that is never crossed costs
    nothing, whereas a continuous extension would have to evaluate the very
    ``0/0`` the shield exists to avoid.

    Args:
        r_vec: Separation vectors ``r_i - r_j``, shape ``[..., 3]``.
        r: ``|r_vec|``, shape ``[...]``.  Passed in rather than recomputed so
            that a caller which already has a sanitised distance reuses it.
        lam3, lam5: Damping factors, shape ``[...]``.

    Returns:
        Tensor of shape ``[..., 3, 3]``.
    """
    finite, inv_r3, inv_r5 = _safe_inverse_powers(r)

    identity = torch.eye(3, dtype=r_vec.dtype, device=r_vec.device)
    outer = r_vec.unsqueeze(-1) * r_vec.unsqueeze(-2)

    isotropic = (lam3 * inv_r3).unsqueeze(-1).unsqueeze(-1) * identity
    anisotropic = (3.0 * lam5 * inv_r5).unsqueeze(-1).unsqueeze(-1) * outer
    tensor = K_E * (isotropic - anisotropic)

    return torch.where(finite.unsqueeze(-1).unsqueeze(-1), tensor, torch.zeros_like(tensor))


def charge_dipole_tensor(r_vec: Tensor, r: Tensor, lam3: Tensor) -> Tensor:
    """Thole-damped charge-dipole tensor, in the energy convention.

        T01_a(r_vec) = K_E * lambda_3 r_a / r^3      [eV / (e * (e A))]

    so that the interaction of a charge on ``i`` with a dipole on ``j`` is

        U = q_i * T01(r_i - r_j) . mu_j

    i.e. this is the ``B^T T01`` block of ``K`` (SPEC 1.5), already carrying
    ``K_E``.  It is odd under ``r_vec -> -r_vec``, which is what makes the
    charge-dipole coupling change sign when the roles of ``i`` and ``j`` are
    exchanged, as it must.

    Only ``lambda_3`` appears: the charge-dipole term is a first derivative of
    the Coulomb potential, so it is damped by the same factor as the isotropic
    part of the dipole-dipole tensor.

    Args:
        r_vec: Separation vectors ``r_i - r_j``, shape ``[..., 3]``.
        r: ``|r_vec|``, shape ``[...]``.
        lam3: Damping factor from :func:`thole_damping`, shape ``[...]``.

    Returns:
        Tensor of shape ``[..., 3]``.
    """
    finite, inv_r3, _ = _safe_inverse_powers(r)
    tensor = K_E * (lam3 * inv_r3).unsqueeze(-1) * r_vec
    return torch.where(finite.unsqueeze(-1), tensor, torch.zeros_like(tensor))


# ---------------------------------------------------------------------------
# Quadrupole tensors
#
# Theta is the TRACELESS SECOND MOMENT, Theta_ab = sum_k q_k (r_a r_b -
# |r|^2 delta_ab / 3), and the energy of a quadrupole in an external potential
# is U = (1/2) Theta:grad grad phi.  Buckingham's Theta is 3/2 of this one.  See
# the module docstring; the convention is not negotiable after the fact because
# a factor of 3/2 passes every symmetry test there is.
# ---------------------------------------------------------------------------


def _safe_inverse_powers_quadrupole(
    r: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``(finite, 1/r^3, 1/r^5, 1/r^7, 1/r^9)`` with no NaN at coincident sites.

    :func:`_safe_inverse_powers` continued two orders up, and separate from it
    rather than a superset so that the dipole path -- which is inside the CG
    iteration -- pays for nothing it does not use.  Same discipline: the
    division is protected by substituting the *input*, so the untaken branch
    contributes neither an infinite value nor an infinite derivative.

    Range note: ``1/r^9`` at the shield threshold ``r = 1e-12`` is ``1e108``,
    eleven orders inside the float64 ceiling, and every tensor below multiplies
    it by an ``O(r^4)`` outer product, so no intermediate overflows.
    """
    finite = r > DEGENERATE_SEPARATION
    r_safe = torch.where(finite, r, torch.ones_like(r))
    inv_r = 1.0 / r_safe
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r * inv_r2
    inv_r5 = inv_r3 * inv_r2
    inv_r7 = inv_r5 * inv_r2
    return finite, inv_r3, inv_r5, inv_r7, inv_r7 * inv_r2


def _identity(r_vec: Tensor) -> Tensor:
    return torch.eye(3, dtype=r_vec.dtype, device=r_vec.device)


def charge_quadrupole_tensor(
    r_vec: Tensor, r: Tensor, lam3: Tensor, lam5: Tensor
) -> Tensor:
    """Thole-damped charge-quadrupole tensor, in the energy convention.

        T02_ab(r_vec) = (K_E / 2) * ( 3 lambda_5 r_a r_b / r^5
                                      - lambda_3 delta_ab / r^3 )

    with units eV / (e * (e A^2)), so that the interaction of a charge on ``i``
    with a quadrupole on ``j`` is

        U = q_i * sum_ab T02_ab(r_i - r_j) Theta_j,ab

    ``T02`` is EVEN under ``r_vec -> -r_vec``, so the same tensor also gives
    ``U = q_j * T02 : Theta_i`` -- charge-quadrupole does not change sign when
    the two sites swap roles, unlike charge-dipole.

    It is exactly ``-1/2`` times :func:`dipole_interaction_tensor` evaluated at
    the same damping.  That is not a coincidence to be exploited by calling the
    other function: both are ``d_a d_b`` of the same damped potential, and they
    differ by the ``-1`` of ``-mu_j . d`` against the ``+1/2`` of
    ``(1/2) Theta_j : dd`` (module docstring).  It is kept as its own function
    because it carries its own units and its own contraction, and asserted as an
    identity in the tests.

    Limits.  ``lambda_3 = O(r^3)`` and ``lambda_5 = O(r^6)`` leave
    ``T02 -> -(K_E a / 2) delta_ab / sqrt(alpha_i alpha_j)`` as ``r -> 0``,
    finite and isotropic.  As in :func:`dipole_interaction_tensor`, the
    ``DEGENERATE_SEPARATION`` shield returns zero rather than that limit; it is
    a jump, twelve orders below any physical separation.

    Args:
        r_vec: Separation vectors ``r_i - r_j``, shape ``[..., 3]``.
        r: ``|r_vec|``, shape ``[...]``.
        lam3, lam5: Damping factors from :func:`thole_damping` or
            :func:`thole_damping_quadrupole`, shape ``[...]``.

    Returns:
        Tensor of shape ``[..., 3, 3]``, symmetric in its two indices.
    """
    finite, inv_r3, inv_r5, _, _ = _safe_inverse_powers_quadrupole(r)

    outer = r_vec.unsqueeze(-1) * r_vec.unsqueeze(-2)
    anisotropic = (3.0 * lam5 * inv_r5).unsqueeze(-1).unsqueeze(-1) * outer
    isotropic = (lam3 * inv_r3).unsqueeze(-1).unsqueeze(-1) * _identity(r_vec)
    tensor = (0.5 * K_E) * (anisotropic - isotropic)

    return torch.where(finite.unsqueeze(-1).unsqueeze(-1), tensor, torch.zeros_like(tensor))


def dipole_quadrupole_tensor(
    r_vec: Tensor, r: Tensor, lam5: Tensor, lam7: Tensor
) -> Tensor:
    """Thole-damped dipole-quadrupole tensor, in the energy convention.

        T12_abc(r_vec) = (K_E / 2) * ( -15 lambda_7 r_a r_b r_c / r^7
                                       + 3 lambda_5 (delta_ab r_c
                                                     + delta_ac r_b
                                                     + delta_bc r_a) / r^5 )

    with units eV / ((e A) * (e A^2)), so that the interaction of a dipole on
    ``i`` with a quadrupole on ``j`` is

        U = sum_abc mu_i,a T12_abc(r_i - r_j) Theta_j,bc

    ``T12`` is ODD under ``r_vec -> -r_vec``, so the opposite pairing carries a
    minus sign:

        U(Theta_i, mu_j) = - sum_abc mu_j,a T12_abc(r_i - r_j) Theta_i,bc

    exactly as the charge-dipole tensor's oddness makes ``K`` antisymmetric
    between those two blocks.  The returned tensor is symmetric under any
    permutation of its three indices -- it is a third derivative -- so the
    "dipole index first" reading above is a convention for the caller, not a
    property of the storage.

    ``lambda_7 = O(r^6)`` and ``lambda_5 = O(r^6)`` make every term ``O(r^2)``,
    so ``T12 -> 0`` at contact, continuously.

    Args:
        r_vec: Separation vectors ``r_i - r_j``, shape ``[..., 3]``.
        r: ``|r_vec|``, shape ``[...]``.
        lam5, lam7: Damping factors from :func:`thole_damping_quadrupole`,
            shape ``[...]``.

    Returns:
        Tensor of shape ``[..., 3, 3, 3]``.
    """
    finite, _, inv_r5, inv_r7, _ = _safe_inverse_powers_quadrupole(r)

    identity = _identity(r_vec)
    a_axis = r_vec[..., :, None, None]
    b_axis = r_vec[..., None, :, None]
    c_axis = r_vec[..., None, None, :]

    triple = a_axis * b_axis * c_axis
    delta_r = (
        identity[:, :, None] * c_axis
        + identity[:, None, :] * b_axis
        + identity[None, :, :] * a_axis
    )

    tensor = (0.5 * K_E) * (
        (-15.0 * lam7 * inv_r7)[..., None, None, None] * triple
        + (3.0 * lam5 * inv_r5)[..., None, None, None] * delta_r
    )

    return torch.where(
        finite[..., None, None, None], tensor, torch.zeros_like(tensor)
    )


def quadrupole_interaction_tensor(
    r_vec: Tensor, r: Tensor, lam5: Tensor, lam7: Tensor, lam9: Tensor
) -> Tensor:
    """Thole-damped quadrupole-quadrupole tensor, in the energy convention.

        T22_abcd(r_vec) = (K_E / 4) * (
              105 lambda_9 r_a r_b r_c r_d / r^9
            -  15 lambda_7 (delta_ab r_c r_d + 5 permutations) / r^7
            +   3 lambda_5 (delta_ab delta_cd + 2 permutations) / r^5 )

    with units eV / (e A^2)^2, so that the interaction of two quadrupoles
    (unordered pair, counted once) is

        U = sum_abcd Theta_i,ab T22_abcd(r_i - r_j) Theta_j,cd

    ``T22`` is EVEN under ``r_vec -> -r_vec`` and symmetric under any
    permutation of its four indices, so it needs no sign bookkeeping when the
    two sites swap roles and the block it forms in ``K`` is symmetric.

    The six ``delta r r`` permutations are the six ways of choosing the index
    pair that carries the delta; the three ``delta delta`` terms are the three
    pairings of four indices.  Both lists are complete here -- a missing
    permutation is the one error in this function that leaves it symmetric under
    rotation and still wrong, which is why the tests finite-difference it rather
    than only rotating it.

    Every term is ``O(r)`` at contact (``lambda_9, lambda_7, lambda_5`` are all
    ``O(r^6)``), so ``T22 -> 0`` continuously.

    Args:
        r_vec: Separation vectors ``r_i - r_j``, shape ``[..., 3]``.
        r: ``|r_vec|``, shape ``[...]``.
        lam5, lam7, lam9: Damping factors from
            :func:`thole_damping_quadrupole`, shape ``[...]``.

    Returns:
        Tensor of shape ``[..., 3, 3, 3, 3]``.
    """
    finite, _, inv_r5, inv_r7, inv_r9 = _safe_inverse_powers_quadrupole(r)

    identity = _identity(r_vec)
    a_axis = r_vec[..., :, None, None, None]
    b_axis = r_vec[..., None, :, None, None]
    c_axis = r_vec[..., None, None, :, None]
    d_axis = r_vec[..., None, None, None, :]

    quartic = a_axis * b_axis * c_axis * d_axis
    delta_pair = (
        identity[:, :, None, None] * c_axis * d_axis
        + identity[:, None, :, None] * b_axis * d_axis
        + identity[:, None, None, :] * b_axis * c_axis
        + identity[None, :, :, None] * a_axis * d_axis
        + identity[None, :, None, :] * a_axis * c_axis
        + identity[None, None, :, :] * a_axis * b_axis
    )
    delta_delta = (
        identity[:, :, None, None] * identity[None, None, :, :]
        + identity[:, None, :, None] * identity[None, :, None, :]
        + identity[:, None, None, :] * identity[None, :, :, None]
    )

    tensor = (0.25 * K_E) * (
        (105.0 * lam9 * inv_r9)[..., None, None, None, None] * quartic
        - (15.0 * lam7 * inv_r7)[..., None, None, None, None] * delta_pair
        + (3.0 * lam5 * inv_r5)[..., None, None, None, None] * delta_delta
    )

    return torch.where(
        finite[..., None, None, None, None], tensor, torch.zeros_like(tensor)
    )


# ---------------------------------------------------------------------------
# Gaussian multipoles -- the Gram construction (SPEC 12)
# ---------------------------------------------------------------------------
#
# Every tensor above is a derivative of ONE damped potential phi_d (module
# docstring).  Feeding them the damping factors of the Gaussian-smeared
# potential erf(r/s)/r instead of Thole's turns every block of K into the
# Coulomb inner product of two Gaussian multipole densities:
#
#     <D^a g_{sigma_a}(. - r_i), D^b g_{sigma_b}(. - r_j)>_Coulomb
#         = the same derivative of K_E erf(r_ij / s_ab) / r_ij ,
#     s_ab = sqrt(sigma_a,i^2 + sigma_b,j^2)
#
# with each site carrying its own width PER RANK (charge, dipole, quadrupole).
# The full (q, mu, Theta) interaction matrix, self blocks included, is then a
# Gram matrix and positive semidefinite for EVERY geometry.  That is a theorem,
# not a damping heuristic, and it is what the Thole route lacks: measured over
# 121 six-atom clusters at the heads' initial values (alpha 1.5 A^3, C 6 A^5),
# four Thole-coupled (q, mu, Theta) Hessians were indefinite (min -0.135 eV)
# although EVERY diagonal block and EVERY pairwise sub-block was positive
# definite -- the three-way coupling alone broke it.  The Gaussian construction
# measured min/max eigenvalue >= +1.0e-10 over 300 clusters, 60 of them with a
# near-coincident pair.


#: Gaussian counterpart of Thole's ``a``: the fraction of the on-site stiffness
#: the Gram self term may use.  PD of K is guaranteed for 0 < a < 1 (SPEC 12).
DEFAULT_GAUSSIAN_A: Final[float] = 0.5

#: Strict upper bound on the Gaussian ``a``: at a = 1 the on-site residual of the
#: stiffest-response direction reaches zero and K is only semidefinite.
GAUSSIAN_A_MAX: Final[float] = 1.0

#: ``K_E / S_1(sigma) = GAUSSIAN_DIPOLE_SELF_VOLUME * sigma^3`` [A^3], where
#: ``S_1 = K_E B_1(0; sqrt(2) sigma) = K_E 4 / (3 sqrt(pi) (sqrt(2) sigma)^3)`` is
#: the Coulomb self energy of a unit Gaussian dipole.  3 sqrt(pi) / sqrt(2).
GAUSSIAN_DIPOLE_SELF_VOLUME: Final[float] = 3.0 * SQRT_PI / 1.4142135623730951

#: ``K_E / S_2(sigma) = GAUSSIAN_QUADRUPOLE_SELF_VOLUME * sigma^5`` [A^5], where
#: ``S_2 = (K_E / 2) B_2(0; sqrt(2) sigma)`` with ``B_2(0; s) = 8 / (5 sqrt(pi) s^5)``
#: is the self energy of a unit traceless Gaussian quadrupole in the orthonormal
#: ``2e`` basis (T22(0) = (K_E/4) B_2(0) (dd + dd + dd), and on a traceless
#: symmetric Theta the three pairings give 0 + 2 Theta:Theta).  5 sqrt(2 pi).
GAUSSIAN_QUADRUPOLE_SELF_VOLUME: Final[float] = 5.0 * 1.4142135623730951 * SQRT_PI

_GAUSSIAN_SELF_VOLUME = {1: GAUSSIAN_DIPOLE_SELF_VOLUME, 2: GAUSSIAN_QUADRUPOLE_SELF_VOLUME}


def gaussian_damping(r: Tensor, width: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``(lambda_3, lambda_5, lambda_7, lambda_9)`` of the Gaussian-smeared potential.

    For ``phi(r) = erf(r / s) / r`` the radial functions ``B_n = (-1/r d/dr)^n phi``
    are ``B_n = (2n - 1)!! lambda_{2n+1} / r^{2n+1}`` with

        lambda_{2n+1}(r) = P(n + 1/2, r^2 / s^2)

    the regularised lower incomplete gamma function -- ``P(1/2, x^2) = erf(x)`` is
    the charge kernel itself, and one integration by parts per order gives the
    rest.  These are drop-in replacements for :func:`thole_damping_quadrupole`'s
    factors: every tensor in this module takes them unchanged and returns the
    exact derivative of ``K_E erf(r/s)/r``.  Measured against autograd
    derivatives of that potential over 40 random separations and widths, worst
    relative deviation T01 1.1e-14, T 1.7e-14, T02 1.7e-14, T12 2.3e-13,
    T22 5.2e-13 (fourth-order autograd round-off).

    ``torch.special.gammainc`` evaluates each factor directly, so the small-``r``
    end -- where ``lambda_9 = O(r^9)`` -- keeps full relative precision without
    the cancellation an upward recursion from ``erf`` would suffer.  Its
    derivative in ``x`` is ``x^(n-1/2) e^-x / Gamma(n+1/2)``, finite at ``x = 0``
    for every ``n >= 1``, and twice differentiable (force training).

    Args:
        r: Separations in A (already sanitised by the caller at coincidence).
        width: Combined width ``s_ab = sqrt(sigma_a,i^2 + sigma_b,j^2)`` in A,
            broadcastable to ``r``.
    """
    x_squared = (r / width) ** 2
    return tuple(
        torch.special.gammainc(torch.full_like(x_squared, order + 0.5), x_squared)
        for order in (1, 2, 3, 4)
    )


def gaussian_self_stiffness(width: Tensor, rank: int) -> Tensor:
    """``S_l(sigma)``: Coulomb self energy of a unit Gaussian multipole of rank ``l``.

    ``S_1 = K_E / (GAUSSIAN_DIPOLE_SELF_VOLUME sigma^3)`` in eV / (e A)^2 and
    ``S_2 = K_E / (GAUSSIAN_QUADRUPOLE_SELF_VOLUME sigma^5)`` in eV / (e A^2)^2.
    These are the diagonal blocks of the Gram matrix; the on-site stiffness must
    dominate them (SPEC 12), which :func:`gaussian_multipole_width` guarantees.
    Rank 0 is :func:`kernel_self`'s tight bound ``sqrt(2) K_E / (sqrt(pi) sigma)``,
    which the hardness floor already exceeds by sqrt(2) and is not repeated here.
    """
    if rank not in _GAUSSIAN_SELF_VOLUME:
        raise ValueError(f"rank must be 1 (dipole) or 2 (quadrupole), got {rank}")
    return K_E / (_GAUSSIAN_SELF_VOLUME[rank] * width ** (2 * rank + 1))


def gaussian_multipole_width(
    stiffness: Tensor, ell: Tensor, rank: int, a: float = DEFAULT_GAUSSIAN_A
) -> Tensor:
    """Per-site Gaussian width for the rank-``l`` moment, derived from its stiffness.

    The Gaussian analogue of Thole's ``u = r / alpha^(1/6)``: the width grows with
    the response, so damping never caps the polarizability.  With the reduced
    moment ``m_hat = m / (e ell^l)`` the on-site block is ``A`` and the Gram self
    block is ``ell^(2l) S_l(sigma)``; K is positive definite for every geometry
    when ``A - ell^(2l) S_l(sigma) I`` is (SPEC 12).  Choosing

        sigma^(2l+1) = K_E ell^(2l) ||A^-2||_F^(1/2) / (a V_l)

    gives ``ell^(2l) S_l = a / ||A^-2||_F^(1/2) <= a lambda_min(A)``, because
    ``||A^-2||_F^(1/2) = (sum_k lambda_k^-4)^(1/4) >= 1/lambda_min``.  So the
    residual is at least ``(1 - a) lambda_min(A) > 0`` for ``0 < a < 1``.

    The Schatten-4 norm is used instead of ``1/lambda_min`` because an
    eigenvalue is not differentiable where eigenvalues cross, and an isotropic
    atom sits exactly there; it overestimates the largest response by at most
    ``d^(1/4)`` (1.32 for a dipole, 1.50 for a quadrupole), which only widens the
    damping, never breaks the bound.

    Args:
        stiffness: ``[n, d, d]`` symmetric positive definite on-site block
            (``A_mu`` with d = 3, or ``A_Theta`` with d = 5), eV.
        ell: ``[n]`` reduction length, A.
        rank: 1 or 2.
        a: ``0 < a < 1``.
    """
    if not 0.0 < a < GAUSSIAN_A_MAX:
        raise ValueError(
            f"gaussian a must satisfy 0 < a < {GAUSSIAN_A_MAX} (got {a}); at a >= 1 "
            "the on-site residual can vanish and K is no longer positive definite."
        )
    if rank not in _GAUSSIAN_SELF_VOLUME:
        raise ValueError(f"rank must be 1 (dipole) or 2 (quadrupole), got {rank}")
    inverse = torch.linalg.inv(stiffness)
    schatten4 = torch.sqrt(torch.linalg.matrix_norm(inverse @ inverse))
    bound = K_E * ell ** (2 * rank) * schatten4
    return (bound / (a * _GAUSSIAN_SELF_VOLUME[rank])) ** (1.0 / (2 * rank + 1))


def quadrupole_cartesian_basis() -> Tensor:
    """``W[m, a, b]``: the ``2e`` coefficient basis for symmetric traceless 3x3.

    A quadrupole is a ``2e`` object, so everything in the quadrupole sector is
    carried as a 5-component coefficient vector ``theta``.  This is the map back
    to cartesian, ``Theta_ab = sum_m theta_m W[m, a, b]``, and it is the *same*
    tensor the dipole stiffness uses for its anisotropy - deliberately, so the
    two sectors cannot end up in different conventions.

    Measured on e3nn 0.4.4, in float64: the five slices are orthonormal to
    1.78e-15 (``|<W_m, W_n> - delta_mn|``) and traceless to 1.11e-16
    (``|tr W_m|``).  Orthonormality is what makes the 5-dimensional forms in
    :class:`QuadrupoleReadout` equal to their cartesian counterparts with no
    Jacobian: ``theta^T A theta`` is ``A_abcd Theta_ab Theta_cd`` for
    ``A_abcd = W[m,a,b] A_mn W[n,c,d]``, and that rank-4 object obeys
    ``A_abcd -> R_ai R_bj R_ck R_dl A_ijkl``, verified to 2.9e-15 relative
    against proper rotations, mirrors and inversion.
    """
    from e3nn import o3  # local: the only e3nn use in this module

    return (o3.wigner_3j(2, 1, 1, dtype=torch.float64) * 5.0**0.5).contiguous()
