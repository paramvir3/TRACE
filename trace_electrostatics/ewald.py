"""Ewald summation for the periodic point-charge and point-dipole energy.

The conditionally convergent lattice sum

    U = (K_E / 2) sum_n sum_ij' q_i q_j / |r_ij + n L|

is split by the identity ``1 = erfc(alpha r) + erf(alpha r)`` into a
short-ranged real-space part, a smooth part summed exactly in reciprocal space,
the self interaction that the Gaussian screening introduced, and -- for a cell
with net charge -- the neutralising background:

    U = U_real + U_recip + U_self + U_jellium

    U_real    = (K_E / 2) sum_{i,j,n}' q_i q_j erfc(alpha r_ijn) / r_ijn
    U_recip   = (2 pi K_E / V) sum_{k != 0} exp(-k^2 / 4 alpha^2) |S(k)|^2 / k^2
    U_self    = -(K_E alpha / sqrt(pi)) sum_i q_i^2
    U_jellium = -K_E pi Q^2 / (2 alpha^2 V)

with ``S(k) = sum_j q_j exp(i k . r_j)`` and ``Q = sum_i q_i``.

``alpha`` is the splitting parameter in 1/A and is a pure convenience: the total
is **independent of alpha** once ``r_cut`` and ``k_max`` are converged.  That
invariance is the standard correctness check for an Ewald implementation and
:func:`ewald_terms` exists to make it easy to run -- the four terms move
strongly with alpha while their sum does not.

The jellium term
----------------
Dropping ``k = 0`` from the reciprocal sum is equivalent to adding a uniform
background of charge ``-Q`` that makes the cell neutral.  The interaction of
the point charges with that background, plus the background's self energy, is
the constant ``-K_E pi Q^2 / (2 alpha^2 V)``.  It does not depend on the atomic
positions and, being quadratic in ``Q`` with ``Q`` fixed by the per-fragment
charge constraints, it does not bias the charges either.  It DOES contribute to
the energy and -- through ``V`` -- to the stress of any charged cell, and it is
what makes the total alpha-independent for ``Q != 0``.  It is included here;
omitting it is a common and silent bug.

A charged cell is finite-size wrong, and the jellium term does not fix it
------------------------------------------------------------------------
The jellium term above makes the sum CONVERGE.  It does not make it the energy
of an isolated charged defect, and the difference is large.  What survives is
the interaction of the defect with its own periodic images and with the
background, and it decays only as ``1/L = V^(-1/3)``.  Measured on a ``+1 e``
seven-point defect in a cubic cell, against its exactly-known isolated energy:

    L (A)          8        10        12        14        17        20        24
    E_per - E_iso  -2.614   -2.073    -1.720    -1.470    -1.208    -1.025   -0.853

a fitted ``L^-1.02``.  At ``-2 e`` it is -10.1 eV at 8 A.  Divided by a
semiconductor's ``epsilon ~ 12`` the 12 A number is still 0.14 eV, which is the
several-tenths-of-an-eV scale that makes an uncorrected charged supercell
unusable for defect formation energies.

:func:`makov_payne_correction` removes it, and :func:`madelung_constant` supplies
the one lattice constant it needs.  Adding the monopole term takes the fitted
exponent to ``L^-3.02``; adding the ``Q_r`` term as well takes it to ``L^-5.00``,
and the error at L = 24 A from 0.853 eV to 5.6e-06 eV.  See
:func:`makov_payne_terms` for the derivation, for the sign of the ``Q_r`` term
(which is the opposite of the form usually quoted, because of a convention on
what ``rho`` means), and for the three assumptions it makes.

*It is opt-in structurally, not by promise.*  Nothing in this module calls it.
Every existing charged and neutral result is bit-identical without it.

*And it does not apply to this package's own charges.*  Makov-Payne needs the
defect charge localised well inside the cell.  For a homogeneous periodic
crystal, SPEC 1.3's ``rho`` spreads the net charge uniformly over the fragment
by construction and the SQE relaxation barely moves it.  Measured on ``n x n x n``
simple-cubic cells at ``+1 e``, with the per-atom excess ``dq_i = q_i(+1) -
q_i(0)``:

    atoms                8        27        64       125
    L (A)             5.20      7.80     10.40     13.00
    max |dq_i| * N    1.160     1.203     1.180     1.195
    Q_r / L^2         0.382     0.221     0.282     0.240

``max|dq_i|`` falls exactly as ``1/N``: the excess charge is uniform to about
20 %.  ``Q_r / L^2`` sits on 0.25, which is what a charge spread uniformly over a
cubic cell gives exactly.  Making one site chemically distinct changes nothing
(the same table to three digits, at feature perturbations of 3x and 12x) -- that
site ends up with *less* than its share.  So ``Q_r`` scales as ``L^2``, the
``Q_r`` term scales as ``L^2 / V = 1/L``, and it is the same order as the
monopole term it is supposed to be a correction to: the expansion has no small
parameter.  :func:`check_defect_localisation` measures exactly this and raises.
The correction is therefore for a defect density the CALLER constructs -- the
difference between a defective and a perfect cell -- and
:func:`defect_moments` takes that density as an argument with no default.

Differentiability
-----------------
Both the positions and the cell enter differentiably: the real-space term
through ``edge_shift @ cell``, the reciprocal term through the reciprocal
vectors ``2 pi (cell^-1)^T``, the volume ``|det cell|`` and the phases, and the
jellium term through the volume.  This is what the strain-deformation route in
``flashace`` needs: deform ``pos`` and ``cell`` by ``1 + epsilon``, add the
electrostatic energy, and the stress follows from the same autograd call.

Gaussian-smeared charges
------------------------
This module sums **point** charges.  The variational model uses the smeared
kernel ``J = K_E erf(r / sigma_ij) / r``; the difference from the point-charge
kernel is ``-K_E erfc(r / sigma_ij) / r``, which is short-ranged and therefore
belongs in the real-space neighbour sum:

    U_smeared = U_ewald(point charges)
                - (K_E / 2) sum_{i,j,n}' q_i q_j erfc(r_ijn / sigma_ij) / r_ijn
                + on-site term 0.5 sum_i q_i^2 J(0; sigma_ii)

where the primed sum excludes the ``i == j``, zero-shift pair.  The on-site
term is the pair kernel at the atom's own SELF width
``sigma_ii = sqrt(2) sigma_i``,

    J(0; sigma_ii) = sqrt(2) K_E / (sigma_i sqrt(pi))    = 7.6595 eV at 1.5 A

and it is what the eta head carries.  Note it is NOT the same number as
``kernels.kernel_self(sigma_i) = 2 K_E / (sigma_i sqrt(pi))`` = 10.8322 eV:
SPEC 1.2 mandates that larger value as the hardness FLOOR, deliberately
sqrt(2) = 41 % above the on-site term written here.  The discrepancy is the
margin, not a bug; see the ``kernels`` module docstring before reconciling the
two files.
Keeping the split explicit is deliberate: the Ewald machinery stays independent
of the charge model, and the smearing correction stays a plain neighbour-list
sum over :func:`trace_electrostatics.kernels.coulomb_kernel`.

Point dipoles
=============
:func:`ewald_multipole_energy` sums the same three blocks the open-boundary
route of ``energy.ElectrostaticAssembly._direct_pair_energy`` computes --
charge-charge, charge-dipole, dipole-dipole -- over the lattice.  The charge
and dipole entry points are kept separate: :func:`ewald_energy` is the plain
point-charge sum over a caller-supplied neighbour list and is unchanged.

Reciprocal space.  The charge density of a point dipole ``mu_j`` at ``r_j`` is
``-mu_j . grad delta(r - r_j)``, whose transform is ``i (k . mu_j) exp(i k . r_j)``,
so the structure factor generalises to

    S(k) = sum_j ( q_j + i k . mu_j ) exp(i k . r_j)

and the reciprocal energy keeps its form, ``(2 pi K_E / V) sum_{k != 0}
exp(-k^2/4 alpha^2) |S(k)|^2 / k^2``.  Setting ``mu = 0`` returns the charge
expression term for term.

Real space.  The screened kernel ``B_0(r) = erfc(alpha r) / r`` and its radial
derivatives ``B_{n+1} = [(2n+1) B_n + (2 alpha^2)^n (2 alpha / sqrt(pi))
exp(-alpha^2 r^2)] / r^2`` give

    U_qq   = (K_E/2) q_i q_j B_0
    U_qmu  =  K_E q_i (mu_j . r) B_1                 (ordered pairs, not halved)
    U_mumu = (K_E/2) [ (mu_i . mu_j) B_1 - (mu_i . r) (mu_j . r) B_2 ]

with ``r = r_i - r_j + n L``.  ``B_1 -> 1/r^3`` and ``B_2 -> 3/r^5`` as
``alpha -> 0``, which is the sign and normalisation convention of
:func:`trace_electrostatics.kernels.dipole_interaction_tensor`.

The dipole self term
--------------------
The reciprocal sum contains every site interacting with its own screening
cloud, and for dipoles that artefact is NOT the charge one.  With the smooth
kernel ``G(r) = erf(alpha r) / r = (2 alpha / sqrt(pi)) (1 - (alpha r)^2 / 3 + ...)``
the spurious term is ``(1/2) (-K_E) mu_a mu_b d_a d_b G |_{r -> 0}``; only the
``r^2`` coefficient survives the two derivatives, ``d_a d_b G|_0 =
-(4 alpha^3 / (3 sqrt(pi))) delta_ab``, so

    U_self = -(K_E alpha / sqrt(pi)) sum_i q_i^2
             -(2 K_E alpha^3 / (3 sqrt(pi))) sum_i |mu_i|^2

The dipole piece scales as ``alpha^3``, has no charge analogue, and is the term
whose omission leaves an energy that is merely shifted by a constant at fixed
alpha -- invisible until someone sweeps alpha, or differentiates with respect
to the cell at fixed alpha in a code that retunes alpha with the volume.

Boundary condition: TINFOIL (conducting)
----------------------------------------
Dropping ``k = 0`` is the conducting ("tinfoil", ``eps' = infinity``) boundary
condition, which is what the charge code already assumes and what this module
keeps.  The ``k -> 0`` limit of the summand is
``(2 pi K_E / V) [ Q^2 / k^2 + (k_hat . P)^2 + ... ]`` with
``P = sum_i (q_i r_i + mu_i)`` the cell dipole: the divergent monopole piece is
cancelled by the neutralising background (:func:`ewald_jellium`), and the
finite, direction-dependent remainder is exactly what an enclosing conductor
screens away.  For vacuum boundaries (``eps' = 1``) the caller would have to add

    U_surface = (2 pi K_E / (3 V)) |P|^2

which is NOT included here.  The choice is observable, and the sign of it is
worth stating in a case with a closed form: for a **cubic** lattice (sc, bcc or
fcc) of identical aligned dipoles every term of the bare dipole tensor is
traceless and a cubic region of images is isotropic, so the vacuum sum over
such a region is exactly zero -- and this module therefore returns exactly

    U = -(2 pi K_E / (3 V)) |P|^2 ,      P = sum_i mu_i

the conductor screening the macroscopic polarization and lowering the energy.
(Measured: 5.6e-17 to 8.9e-16 eV from that closed form, for sc, bcc and fcc and
for two splitting parameters.  In a tetragonal cell cubic symmetry no longer
forces the lattice part to vanish and the closed form stops holding: at
``a = 4 A`` with one dipole of ``|mu| = 0.5 e A`` along z, ``alpha = 0.9``,
the departure is +3.0e-2 eV at ``c/a = 1.1``, +6.2e-2 at ``c/a = 1.25`` and
+1.4e-1 at ``c/a = 2``.  A bare "4.6e-2 eV" with no cell attached would be
unfalsifiable -- the number is a function of ``c/a``.)

Two-dimensional (slab) boundaries -- SPEC section 7
---------------------------------------------------
Periodicity is an INPUT (a ``pbc`` triple, as in ASE) and never inferred from
the cell.  Four regimes, three of which are different physics rather than a
flag; :func:`boundary_regime` is the one place the triple is interpreted:

===============  ===========================================================
``(F, F, F)``    open boundary, :func:`direct_multipole_energy`.  **A supplied
                 cell is ignored** -- structurally, not by promise: that
                 function has no ``cell`` parameter at all.
``(T, T, T)``    3-D Ewald, tinfoil -- everything above, unchanged.
two True         **2-D slab**, below.  Any permutation; the non-periodic
                 lattice vector is found by :func:`non_periodic_axis` and the
                 correction is projected onto the normal of the periodic
                 plane, so ``(T, F, T)`` and ``(F, T, T)`` need no relabelling.
one True         a **wire**.  :func:`boundary_regime` raises; 3-D Ewald on it
                 would be wrong in the same way and worse.
===============  ===========================================================

*Which slab method.*  **Yeh-Berkowitz**: 3-D Ewald in a cell padded with vacuum
along the non-periodic direction, plus a closed-form correction.  Not a genuine
2-D (Parry / Heyes-van Swol) Ewald, deliberately.  The reason is not only the
implementation cost: a 2-D Ewald's reciprocal sum carries a per-pair
``exp(+k z) erfc(...)`` structure whose cost is ``O(n_atoms^2 n_k)`` rather than
``O(n_atoms n_k)``, it needs its own splitting and truncation parameters, and it
would be a second, independently-derived lattice sum sitting next to this one --
the duplication this module already refuses elsewhere.  The correction below
reuses every line of the 3-D machinery and adds one scalar.

*Why plain 3-D Ewald on a slab is wrong.*  Under tinfoil boundaries the ``k = 0``
term is dropped, which sets the macroscopic field in the cell to zero: the
conductor screens the depolarizing field of the stack of replica slabs.  An
isolated slab in vacuum carries that depolarizing field.  The difference is
computable in closed form on a case with an exact answer -- a parallel-plate
capacitor, ``+sigma`` at ``z = 0`` and ``-sigma`` at ``z = d`` in a cell of area
``A`` and height ``L``, for which ``M_z = -sigma A d`` and ``V = A L``:

    isolated   U = (1/8 pi) (4 pi sigma)^2 A d          = 2 pi sigma^2 A d
    tinfoil    U = 2 pi sigma^2 A d (1 - d / L)         (uniform field removed
                                                         so that <E> = 0)
    difference = -2 pi sigma^2 A d^2 / L = -2 pi M_z^2 / V

So the spurious term the 3-D sum adds to a slab is ``-2 pi K_E M_z^2 / V``, and
the correction that removes it is

    U_slab = U_3D(tinfoil) + 2 pi K_E M_n^2 / V ,    M_n = sum_i q_i (r_i . n)
                                                           + sum_i mu_i . n

with ``n`` the unit normal of the plane spanned by the two PERIODIC lattice
vectors (:func:`slab_geometry`).  It is ``2 pi``, not ``4 pi``; it carries one
factor of ``K_E`` exactly as every other term here does; and the sign is
**positive**, because tinfoil screening lowers the energy and the isolated slab
does not get to keep that.  The general statement is the depolarization tensor
of the macroscopic shape, ``(2 pi K_E / V) sum_a n_a M_a^2``: ``n_a = 1/3`` for
the sphere gives the familiar ``(2 pi K_E / 3 V) |M|^2`` vacuum surface term of
the previous section, and a plate normal to ``n`` gives ``n_a = n_a n_b``, i.e.
this.  The two are the same formula at two shapes, which is the check that the
factor is right.

*"It converges away with more vacuum" -- it does not, in any useful sense.*
The spurious term is ``-2 pi K_E M_n^2 / (A L)``: it falls only as ``1/L``, so
doubling the vacuum halves it and nothing more.  Measured on a polar bilayer
(8 x 8 A cell, ``M_z = -5 e A``, isolated-slab energy 4.4713445 eV from the
direct 2-D lattice sum of the next paragraph):

    L_z (A)      5        10        20        40       80      160      320
    U_3D - U_iso  -6.61    -3.53     -1.77    -0.884   -0.442   -0.221   -0.110
    U_YB - U_iso  +0.454   +6.7e-3   +2.6e-6  +3.2e-8  3.2e-8   3.2e-8   3.2e-8

At 40x padding the uncorrected error is still 0.11 eV; reaching 1e-6 eV that way
needs ``L_z`` of order ``10^7 A``.  The corrected sum is at its floor by 5x.

*Convergence condition, and it is NOT about the slab thickness.*  What is left
after the correction is the interaction of the slab with its replicas through
the CORRUGATION of its charge density.  The potential of a 2-D periodic density
decays as ``exp(-|G| h)`` above the plane, where ``G`` runs over the non-zero
2-D reciprocal lattice, so the residual is set by the smallest such ``|G|`` and
by the vacuum GAP ``h`` between the slab and its image:

    residual ~ U_scale * exp(-|G|_min h) ,    |G|_min = 2 pi / lateral

Measured on the same bilayer (``lateral = 8 A``), residual versus ``h``:

    h / lateral   0.31    0.44    0.69     0.94     1.44     2.19     3.44
    residual     4.5e-1  1.8e-1  3.3e-2   6.7e-3   2.9e-4   2.6e-6   1.2e-9
    exp(-2 pi x) 1.4e-1  6.4e-2  1.3e-2   2.8e-3   1.2e-4   1.1e-6   4.2e-10

i.e. ``residual = 2.4 eV * exp(-2 pi h / lateral)`` to within a factor 2.4 over
nine decades, and 2.4 eV is about half the total energy of that system -- the
prefactor is the energy scale, not a universal number.  So the requirement is on
``h`` measured in units of the LONGEST IN-PLANE REPEAT, not in units of the slab
thickness and not as a ratio ``L_z / L_xy``:

    h >= ln(U_scale / accuracy) / |G|_min    (1e-6:  h >= 2.34 lateral;
                                              1e-10: h >= 3.80 lateral)

A 30 A wide surface supercell therefore wants ~114 A of vacuum for 1e-10, which
is the fact that makes this a runtime check and not a docstring remark:
:func:`slab_vacuum_report` measures ``h`` and ``|G|_min`` from the actual cell
and positions, and :func:`check_slab_geometry` raises with both numbers.

*Neutral cells only.*  ``M_n`` is origin-independent only when ``sum_i q_i = 0``,
so for a charged cell the correction above is not even well defined -- it changes
if the caller wraps the coordinates.  The charged-slab generalisation
(Ballenegger, Arnold and Cerda) adds ``-Q sum_i q_i z_i^2 - Q^2 L^2 / 12`` inside
the bracket, but it fixes a 2-D neutralising-background convention that this
package does not otherwise define, so it is out of scope and the slab route
raises on a charged cell rather than returning an origin-dependent number.

*Cost.*  Padding a cell makes the reciprocal lattice dense along the padded
direction, and :meth:`EwaldParams.auto_tune` sizes ``k_max`` from the LONGEST
lattice vector for all three directions -- correct, but ``(2 k + 1)^3`` with
``k`` set by ``L_z`` is unusable at real paddings.  :meth:`EwaldParams.auto_tune_slab`
fills in ``k_max_per_axis`` instead, one half-width per direction; on a
``8 x 8 x 60 A`` cell at ``accuracy = 1e-6`` that is ``(4, 4, 25)`` -- 4131 wave
vectors against 132651 for the isotropic grid, a 32x saving that grows linearly
with the padding.

Three different "alphas"
------------------------
They are unrelated and must never be conflated:

* ``params.alpha``   -- the Ewald splitting parameter, 1/A.  A convenience; the
  total does not depend on it.
* ``sigma_i``        -- the physical Gaussian width of the charge cloud, A.  It
  is part of the model and changes the energy.
* ``alpha_volume_i`` -- the polarizability VOLUME in A^3 that sets the Thole
  reduced distance (SPEC 0: ``K_E`` times the response polarizability).

Damping is real space only
--------------------------
The lattice sum above is over *bare* point multipoles.  The model's damped
kernels differ from the bare ones by ``erf(r/sigma_ij) - 1`` and by
``lambda_3 - 1``, ``lambda_5 - 1``, all of which vanish exponentially, so the
whole difference is a short-ranged real-space correction added to the same
image sum (:func:`ewald_real_multipole` returns it separately).  The reciprocal
sum never sees the Thole factors, so they cannot be double counted; equally,
they must not be applied there, because at ``k`` small the damped and bare
kernels agree and any attempt to damp the smooth part would remove a term that
the real-space sum has already been charged for.

Conventions
-----------
``cell`` has the three lattice vectors as its ROWS (``edge_shift @ cell`` is a
Cartesian displacement), matching ``flashace.model``.  A single periodic cell
per call; batched systems are looped or blocked by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final, NamedTuple, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .constants import K_E, SQRT_PI
from .kernels import (
    DEFAULT_THOLE_A,
    DEGENERATE_SEPARATION,
    charge_dipole_tensor,
    combine_sigma,
    coulomb_kernel,
    dipole_interaction_tensor,
    kernel_self,
    thole_damping,
)

__all__ = [
    "EwaldParams",
    "EwaldTerms",
    "EwaldMultipoleTerms",
    "ewald_real",
    "ewald_recip",
    "ewald_self",
    "ewald_self_dipole",
    "ewald_jellium",
    "ewald_terms",
    "ewald_energy",
    "ewald_real_multipole",
    "ewald_recip_multipole",
    "ewald_multipole_terms",
    "ewald_multipole_energy",
    # finite-size correction for a charged cell -- Makov-Payne
    "MADELUNG_SIMPLE_CUBIC",
    "MADELUNG_BCC",
    "MADELUNG_FCC",
    "MADELUNG_UNIFORM_SPHERE",
    "UNIFORM_CELL_RADIUS_RATIO",
    "madelung_constant",
    "DefectMoments",
    "defect_moments",
    "MakovPayneParams",
    "MakovPayneTerms",
    "makov_payne_terms",
    "makov_payne_correction",
    "check_defect_localisation",
    # boundary conditions -- SPEC section 7
    "BoundaryRegime",
    "boundary_regime",
    "non_periodic_axis",
    "SlabGeometry",
    "slab_geometry",
    "slab_dipole",
    "slab_correction",
    "slab_correction_blocks",
    "SlabVacuumReport",
    "slab_vacuum_report",
    "check_slab_geometry",
    "required_vacuum_gap",
    "EwaldSlabTerms",
    "EwaldMultipoleSlabTerms",
    "ewald_slab_terms",
    "ewald_slab_energy",
    "ewald_multipole_slab_terms",
    "ewald_multipole_slab_energy",
    "direct_multipole_energy",
    "energy_for_pbc",
]

#: Default target for the neglected tails of both sums, relative to K_E e^2 / A.
DEFAULT_EWALD_ACCURACY: Final[float] = 1.0e-10

#: ``DEGENERATE_SEPARATION`` (separations below this, in A, are treated as
#: coincident and dropped from the real-space sum) is imported from
#: :mod:`trace_electrostatics.kernels` rather than restated here.  It used to be
#: defined in both modules with the same literal, which is exactly the kind of
#: pair that drifts.  A NaN shield only: a neighbour list never contains a pair
#: at zero separation, and the ``i == j`` self interaction is handled
#: analytically by :func:`ewald_self`.

TWO_PI: Final[float] = 2.0 * math.pi

#: Coefficient of the dipole self energy, ``U = -(K_E alpha / sqrt(pi)) *
#: DIPOLE_SELF_COEFFICIENT * alpha^2 * sum_i |mu_i|^2``.  Named rather than
#: inlined because it is the one number in this module that cannot be checked
#: by inspection against a charge-only reference: see the module docstring for
#: the derivation (it is the ``r^2`` coefficient of ``erf(alpha r) / r``,
#: twice differentiated).
DIPOLE_SELF_COEFFICIENT: Final[float] = 2.0 / 3.0

#: Extra half shell added when sizing the real-space image box.  Displacements
#: are reduced to the minimum image first, which can still be half a cell out
#: along every lattice direction, so a box of ``r_cut / spacing + 0.5`` shells
#: is what guarantees that every image within ``r_cut`` is present.
MINIMUM_IMAGE_MARGIN: Final[float] = 0.5

#: Coefficient of the slab (Yeh-Berkowitz) correction,
#: ``U = SLAB_CORRECTION_COEFFICIENT * pi * K_E * M_n^2 / V``.  Named because it
#: is the number the literature is inconsistent about: it is 2, not 4, and the
#: sign of the whole term is POSITIVE.  Derived in the module docstring from a
#: parallel-plate capacitor, where both the isolated and the tinfoil energies
#: are available in closed form, and checked numerically against a direct 2-D
#: lattice sum (agreement 1.3e-08 eV on a 4.47 eV energy).
SLAB_CORRECTION_COEFFICIENT: Final[float] = 2.0

#: Largest net cell charge, in e, for which the slab correction is accepted.
#: ``M_n`` is origin-dependent when ``sum_i q_i != 0``, so beyond this the
#: correction is not merely inaccurate but ill-defined; see
#: :func:`check_slab_geometry`.  Sits two decades above the ``1e-10`` per-fragment
#: charge residual SPEC 2.1.5 requires of the solve, so a converged solve never
#: trips it.
SLAB_NEUTRALITY_TOLERANCE: Final[float] = 1.0e-8

#: Energy scale, in eV, used to turn the dimensionless replica suppression
#: ``exp(-|G|_min h)`` into an absolute error estimate in
#: :func:`slab_vacuum_report`.  Measured, not assumed: on the polar bilayer of
#: the module docstring the residual is ``2.4 eV * exp(-|G|_min h)`` over nine
#: decades of ``h``, and 2.4 eV is about half that system's total electrostatic
#: energy.  It is therefore an order-of-magnitude stand-in for "the
#: electrostatic energy scale of the system", and a system an order of magnitude
#: more strongly bound wants an order of magnitude more suppression.
SLAB_RESIDUAL_ENERGY_SCALE: Final[float] = 2.4


def _balanced_alpha(r_cut: float, accuracy: float) -> float:
    """Splitting parameter that balances the two truncation errors.

    The real-space tail is governed by ``erfc(alpha r_cut) ~ exp(-(alpha r_cut)^2)``
    and the reciprocal tail by ``exp(-k_cut^2 / 4 alpha^2)``.  Demanding both be
    ``accuracy`` gives ``alpha = sqrt(p) / r_cut`` with ``p = -ln(accuracy)``,
    and then ``k_cut = 2 alpha sqrt(p)``.
    """
    return math.sqrt(-math.log(accuracy)) / r_cut


@dataclass(frozen=True)
class EwaldParams:
    """Splitting and truncation parameters for the Ewald sum.

    Attributes:
        alpha: Splitting parameter in 1/A.  Large alpha moves work into the
            reciprocal sum, small alpha into the real-space sum.  The total
            energy does not depend on it; the cost and the truncation errors do.
            ``None`` means "choose it from ``r_cut`` and ``accuracy``" via the
            balanced choice ``alpha = sqrt(-ln(accuracy)) / r_cut``.  It is
            resolved, and coerced to ``float``, at construction: **after
            ``__init__`` the attribute is always a real float and never None**,
            so consumers must neither test it for None nor wrap it in
            ``float()``.  The annotation stays ``float | None`` only because
            that is what the constructor accepts.
        k_max: Half-width of the integer grid of reciprocal vectors.  All
            ``n in [-k_max, k_max]^3`` except the origin are summed, i.e.
            ``(2 k_max + 1)^3 - 1`` wave vectors.  :meth:`auto_tune` picks it
            from the cell; given explicitly it is the caller's responsibility.
            Coerced to ``int`` at construction, and a non-integral value is
            REJECTED rather than truncated -- ``k_max = 1.9`` used to be stored
            unconverted while every consumer silently summed a grid of 1.
        r_cut: Real-space cutoff in A.  It sizes the image box of the multipole
            route (:func:`_image_offsets`) and sets the target the splitting is
            balanced against.  For the neighbour-list route
            (:func:`ewald_real`) it is *not* applied as a second cut: that
            function sums whatever the caller's list contains, which must cover
            at least ``r_cut``.
        accuracy: Target size of the neglected tails, used only to derive
            ``alpha`` when it is not given.  Kept on the object so that the
            choice is inspectable after the fact.
        image_shells: ``(n_x, n_y, n_z)`` half-widths of the real-space image
            box of the multipole route, or None.  This is the real-space twin of
            ``k_max`` and exists for the same reason: it makes the summation
            shape a plain Python constant instead of something derived from the
            cell tensor at call time.  ``None`` keeps the old behaviour --
            derive it from the cell on every call -- which costs a
            device-to-host synchronisation and makes
            :func:`ewald_multipole_energy` impossible to
            ``torch.compile(fullgraph=True)`` (measured: ``Unsupported: Input
            tensor for tolist must be an integer tensor``).  :meth:`auto_tune`
            fills it in, and the AOT path of SPEC section 2 requires it.
        k_max_per_axis: ``(n_1, n_2, n_3)`` half-widths of the reciprocal grid,
            one per lattice direction, or None for the isotropic
            ``[-k_max, k_max]^3``.  It exists for the slab route: padding a cell
            with vacuum makes the reciprocal lattice dense along the padded
            direction only, and a single ``k_max`` then has to be large in all
            three (see the module docstring -- 4131 wave vectors against 132651
            on one real cell).  ``k_max`` is NOT superseded by it and must stay
            at least ``max(k_max_per_axis)``: a consumer that reads only
            ``k_max`` -- ``energy._periodic_pair_blocks`` is one -- then sums a
            grid that is too large rather than too small, which costs time and
            not accuracy.  :meth:`auto_tune_slab` maintains that relation and
            ``__post_init__`` enforces it.

    The parameters are plain Python numbers, never tensors: they are
    hyperparameters, and nothing differentiates through them.
    """

    alpha: float | None
    k_max: int
    r_cut: float
    accuracy: float = DEFAULT_EWALD_ACCURACY
    image_shells: Optional[Tuple[int, int, int]] = None
    k_max_per_axis: Optional[Tuple[int, int, int]] = None

    def __post_init__(self) -> None:
        if not self.r_cut > 0.0:
            raise ValueError(f"Ewald r_cut must be strictly positive (got {self.r_cut}).")
        if not 0.0 < self.accuracy < 1.0:
            raise ValueError(
                f"Ewald accuracy must lie strictly between 0 and 1 (got {self.accuracy}); "
                "it is the target relative size of the neglected sum tails."
            )
        if self.alpha is None:
            # Frozen dataclass: resolving the default in place keeps `alpha` a
            # plain float everywhere downstream, so no consumer has to handle None.
            object.__setattr__(self, "alpha", _balanced_alpha(self.r_cut, self.accuracy))
        if not self.alpha > 0.0:
            raise ValueError(f"Ewald alpha must be strictly positive (got {self.alpha}).")
        # Resolved AND coerced, so `alpha` is a plain float on every path. One
        # call site used to write `float(params.alpha)` and five did not, which
        # is the tell that the declared `float | None` was never really either.
        object.__setattr__(self, "alpha", float(self.alpha))

        # Coerce, do not merely check: `k_max` used to be validated with
        # `int(self.k_max)` and then stored unconverted, so `k_max = 1.9` passed
        # validation, was stored as 1.9, and every consumer's `int(params.k_max)`
        # quietly summed a grid of half-width 1.
        if float(self.k_max) != int(self.k_max):
            raise ValueError(
                f"Ewald k_max must be a whole number (got {self.k_max!r}); it is the "
                "half-width of an integer grid of reciprocal vectors, and silently "
                "truncating it would change the sum without changing the recorded "
                "parameters."
            )
        object.__setattr__(self, "k_max", int(self.k_max))
        if self.k_max < 1:
            raise ValueError(f"Ewald k_max must be at least 1 (got {self.k_max}).")
        if self.image_shells is not None:
            shells = tuple(self.image_shells)
            if len(shells) != 3 or any(
                float(n) != int(n) or int(n) < 1 for n in shells
            ):
                raise ValueError(
                    "Ewald image_shells must be three whole numbers >= 1, the "
                    f"half-widths of the real-space image box (got {self.image_shells!r}). "
                    "Use EwaldParams.auto_tune(cell, r_cut) to derive them from a cell."
                )
            object.__setattr__(self, "image_shells", tuple(int(n) for n in shells))
        if self.k_max_per_axis is not None:
            per_axis = tuple(self.k_max_per_axis)
            if len(per_axis) != 3 or any(
                float(n) != int(n) or int(n) < 1 for n in per_axis
            ):
                raise ValueError(
                    "Ewald k_max_per_axis must be three whole numbers >= 1, the "
                    "half-widths of the reciprocal grid along each lattice "
                    f"direction (got {self.k_max_per_axis!r}). Use "
                    "EwaldParams.auto_tune_slab(cell, r_cut, pbc) to derive them."
                )
            per_axis = tuple(int(n) for n in per_axis)
            object.__setattr__(self, "k_max_per_axis", per_axis)
            if max(per_axis) > self.k_max:
                raise ValueError(
                    f"Ewald k_max ({self.k_max}) is smaller than "
                    f"max(k_max_per_axis) = {max(per_axis)}. k_max is the "
                    "fallback for every consumer that does not read the "
                    "per-axis grid, so it has to be the CONSERVATIVE one; "
                    "leaving it small would let such a consumer silently sum "
                    "too few wave vectors. Set k_max = max(k_max_per_axis)."
                )

    @property
    def k_grid(self) -> Union[int, Tuple[int, int, int]]:
        """The reciprocal grid half-widths actually summed: per-axis, else ``k_max``.

        One accessor rather than a conditional at each call site, so that the
        isotropic path is literally unchanged when ``k_max_per_axis`` is None --
        :func:`_wave_vectors` then receives the same ``int`` it always did and
        builds the same grid in the same order, which is what keeps every
        ``(T, T, T)`` result bit-identical across this addition.
        """
        return self.k_max if self.k_max_per_axis is None else self.k_max_per_axis

    @classmethod
    def auto_tune(
        cls,
        cell: Tensor,
        r_cut: float,
        accuracy: float = DEFAULT_EWALD_ACCURACY,
        alpha: Optional[float] = None,
    ) -> "EwaldParams":
        """Pick alpha and k_max for a target truncation error at fixed ``r_cut``.

        The real-space tail is governed by ``erfc(alpha r_cut) < exp(-p)`` and
        the reciprocal tail by ``exp(-k_cut^2 / 4 alpha^2) < exp(-p)`` with
        ``p = -ln(accuracy)``, which gives the textbook balanced choice

            alpha = sqrt(p) / r_cut ,   k_cut = 2 alpha sqrt(p) = 2 p / r_cut

        ``k_max`` is then the number of integer steps needed to reach ``k_cut``
        along the *shortest* reciprocal lattice vector, so that every direction
        is converged even for an elongated cell.

        ``image_shells``, the real-space twin of ``k_max``, is filled in here
        too.  Doing it once, at tune time, is what lets the multipole route
        compile with static shapes: the alternative is to re-derive it from the
        cell tensor inside every call, which forces a device-to-host sync and
        blocks ``fullgraph=True``.

        The parameters are plain Python numbers: they are hyperparameters chosen
        from the cell, never differentiated through.  Re-tune if the cell
        changes shape substantially (constant-volume MD does not need it).

        Args:
            cell: ``[3, 3]`` lattice vectors as rows.
            r_cut: Real-space cutoff in A, normally the neighbour-list cutoff.
            accuracy: Target relative size of the neglected tails.

        Raises:
            ValueError: on a non-positive ``r_cut`` or an ``accuracy`` outside
                ``(0, 1)`` -- both surface from the constructor.
        """
        # ``alpha`` overrides the balanced sqrt(p)/r_cut, and the reciprocal grid
        # is then sized for the alpha ACTUALLY USED: k_cut = 2 alpha sqrt(p) is a
        # property of the splitting, so a k_max derived from one alpha and paired
        # with another under-sums the reciprocal tail (measured 0.22 eV at
        # alpha = 1.6 A^-1 with the grid sized for the balanced ~0.5). With a
        # pinned alpha the REAL-space tail erfc(alpha r_cut) is the caller's
        # responsibility: below the balanced alpha it exceeds ``accuracy``.
        params = cls(alpha=alpha, k_max=1, r_cut=r_cut, accuracy=accuracy)

        # k_cut = 2 alpha sqrt(p): the reciprocal tail matches the real-space one.
        k_cut = 2.0 * params.alpha * math.sqrt(-math.log(accuracy))
        # The grid is summed over INTEGER coefficients n in [-k_max, k_max]^3 of
        # the reciprocal basis, so the requirement is on those integers, not on
        # |b|.  For a wanted |k| = k_cut, n_j = k . a_j / (2 pi), hence
        #
        #     k_max >= k_cut * max_j |a_j| / (2 pi)          (tight)
        #
        # Using k_cut / min_j |b_j| instead is never conservative: a_j . b_j =
        # 2 pi forces |b_j| >= 2 pi / |a_j|, so that form is always <= the tight
        # bound, with equality only for orthorhombic cells.  On a strongly
        # sheared cell it returned k_max = 6 where 42 was needed, giving a
        # 1.455 eV error against a requested accuracy of 1e-8, silently.
        #
        # This matches the real-space half of the module, which already uses the
        # tight bound r_cut / spacing + 1/2 in _image_shell_counts.
        #
        # The same bound holds per direction, |n_j| <= k_cut |a_j| / (2 pi), so the box with
        # those half-widths holds the whole k_cut sphere for ANY cell (speed-up R2, 2026-10-09):
        # the longest vector is needed only along its own direction.  On the 9.05 x 9.05 x 26.46 A
        # Au2/MgO cell that is (4, 4, 12) and 2 025 wave vectors, where max_j for all three summed
        # 15 625.  Only vectors with |k| > k_cut, beyond the accuracy target, are no longer
        # summed (tests/electrostatics/test_ewald_per_axis.py bounds the change by their sum).
        # k_max keeps the longest-vector value, so a consumer that reads only k_max is unchanged,
        # and a cell whose three counts agree keeps the scalar grid, bit for bit.
        lengths = cell.detach().to(dtype=torch.float64).norm(dim=-1)
        per_axis = tuple(
            max(1, int(math.ceil(k_cut * float(length) / (2.0 * math.pi)))) for length in lengths
        )
        k_max = max(per_axis)

        return cls(
            alpha=params.alpha,
            k_max=k_max,
            r_cut=r_cut,
            accuracy=accuracy,
            image_shells=_image_shell_counts(cell, r_cut),
            k_max_per_axis=None if len(set(per_axis)) == 1 else per_axis,
        )

    @classmethod
    def auto_tune_slab(
        cls,
        cell: Tensor,
        r_cut: float,
        pbc: Sequence[bool] = (True, True, False),
        accuracy: float = DEFAULT_EWALD_ACCURACY,
        alpha: Optional[float] = None,
    ) -> "EwaldParams":
        """:meth:`auto_tune` with a per-direction reciprocal grid, for a padded cell.

        Same alpha and the same balanced ``k_cut = 2 alpha sqrt(p)``; the only
        difference is that the number of integer steps needed to reach ``k_cut``
        is computed per lattice direction,

            k_max_d = ceil( k_cut |a_d| / (2 pi) )

        rather than once from the longest vector.  On a slab the two differ by
        the padding ratio and nothing else, because the padded direction is the
        long one: ``8 x 8 x 60 A`` at ``accuracy = 1e-6`` and ``r_cut = 8 A``
        gives ``(4, 4, 25)`` where :meth:`auto_tune` gives 25 in all three,
        4131 wave vectors against 132651.  The saving is linear in the padding,
        which is what makes a convergence study in the padding affordable.

        ``k_max`` is set to ``max(k_max_per_axis)``, i.e. to exactly what
        :meth:`auto_tune` would have chosen, so a consumer that reads only
        ``k_max`` is slow rather than wrong.  **The reciprocal sum is over the
        full 3-D lattice either way**: this is a cost parameter, not a change of
        physics, and the slab physics is entirely in
        :func:`slab_correction`.

        ``pbc`` is accepted and validated -- a wire or an all-open triple is
        rejected here rather than several calls later -- but does not otherwise
        enter: ``k_cut`` is isotropic in Cartesian space and the non-periodic
        direction still needs its reciprocal sum, since the 3-D Ewald sum
        underneath is a genuine 3-D sum.

        Args:
            cell: ``[3, 3]`` lattice vectors as rows, INCLUDING the vacuum
                padding along the non-periodic direction.
            r_cut: Real-space cutoff in A.
            pbc: Periodicity triple; must select the slab regime.
            accuracy: Target relative size of the neglected tails.

        Raises:
            ValueError: if ``pbc`` is not a slab (see :func:`boundary_regime`).
        """
        regime = boundary_regime(pbc)
        if regime is not BoundaryRegime.SLAB:
            raise ValueError(
                f"EwaldParams.auto_tune_slab needs a 2-D slab pbc triple, got "
                f"{tuple(bool(flag) for flag in pbc)!r} ({regime.value}). Use "
                "EwaldParams.auto_tune for (True, True, True); the open "
                "boundary needs no Ewald parameters at all."
            )
        # See auto_tune: an explicit alpha sizes the grid for itself.
        params = cls(alpha=alpha, k_max=1, r_cut=r_cut, accuracy=accuracy)
        k_cut = 2.0 * params.alpha * math.sqrt(-math.log(accuracy))
        lengths = cell.detach().to(dtype=torch.float64).norm(dim=-1)
        per_axis = tuple(
            max(1, int(math.ceil(k_cut * float(length) / (2.0 * math.pi))))
            for length in lengths
        )
        return cls(
            alpha=params.alpha,
            k_max=max(per_axis),
            r_cut=r_cut,
            accuracy=accuracy,
            image_shells=_image_shell_counts(cell, r_cut),
            k_max_per_axis=per_axis,
        )


class EwaldTerms(NamedTuple):
    """The four contributions to the Ewald energy, in eV.

    Each term depends strongly on ``EwaldParams.alpha``; ``total`` does not.
    Comparing ``total`` across a range of alpha is the standard correctness
    check, and comparing the individual terms shows where a converged sum is
    spending its accuracy.

    A ``NamedTuple`` and not a frozen dataclass, deliberately: Dynamo cannot
    construct a user-defined frozen dataclass inside a graph, so returning one
    made every function that produced it impossible to
    ``torch.compile(fullgraph=True)``.  That took :func:`ewald_energy` down with
    it, not only the multipole route, even though every leaf function compiles
    cleanly.  ``NamedTuple`` is supported, including the ``total`` property.
    """

    real: Tensor
    recip: Tensor
    self_energy: Tensor
    jellium: Tensor

    @property
    def total(self) -> Tensor:
        return self.real + self.recip + self.self_energy + self.jellium


def _require_matching_dtype(
    tensor: Tensor, name: str, reference: Tensor, reference_name: str
) -> None:
    """Raise a message that names the fix when two inputs disagree on dtype.

    Mixed dtypes RAISE here rather than being promoted, and that is on purpose:
    a float32 dipole silently upcast against a float64 cell would cost six
    digits of a quantity whose whole job is to be accurate to 1e-10, and the
    loss would show up only as a mysteriously loose gradient test.  What was
    wrong before is not the failure but the message -- torch's own
    ``expected scalar type Double but found Float`` names neither the argument
    nor the remedy.  No ``.to()`` is inserted; the caller's dtypes are the
    caller's to fix.

    Compares ``dtype`` only, which is static metadata: no host sync, no
    data-dependent branch, and nothing that blocks ``fullgraph=True``.
    """
    if tensor.dtype is not reference.dtype:
        raise TypeError(
            f"{name} has dtype {tensor.dtype} but {reference_name} has "
            f"{reference.dtype}; trace_electrostatics.ewald does not promote "
            "silently, because an upcast float32 multipole would quietly cost six "
            "digits of a sum that is required to be accurate to 1e-10. Fix it at "
            "the source: call torch.set_default_dtype(torch.float64) before "
            "constructing the model (SPEC 3.1), or pass "
            f"{name}.to({reference_name}.dtype) if the low precision is deliberate."
        )


def _reciprocal_vectors(cell: Tensor) -> Tensor:
    """Rows ``b_j`` of the reciprocal lattice, ``a_i . b_j = 2 pi delta_ij``.

    With the direct lattice vectors as rows of ``cell``, ``b = 2 pi (cell^-1)^T``.
    Differentiable in ``cell``, which is what carries the reciprocal-space part
    of the stress.
    """
    return TWO_PI * torch.linalg.inv(cell).transpose(-1, -2)


def _wave_vectors(
    k_max: Union[int, Tuple[int, int, int]], cell: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Build the reciprocal grid and its Gaussian weights.

    Returns ``(k, k_squared, nonzero)``: the Cartesian wave vectors, their
    squared lengths, and the mask that removes the origin.  The origin is kept
    in the grid -- so the shape is a static function of ``k_max`` alone, which
    the AOT export path needs -- and excluded by the mask rather than by
    data-dependent indexing.  The Gaussian weight is applied by the caller.

    ``k_max`` is either one half-width for all three directions or a triple, one
    per direction (:attr:`EwaldParams.k_max_per_axis`, for the padded cell of
    the slab route).  The scalar branch is kept verbatim rather than folded into
    the general one: it is the ``(T, T, T)`` path, and ``cartesian_prod`` of one
    tensor with itself three times is the expression whose output order every
    existing result was produced with.
    """
    if isinstance(k_max, (tuple, list)):
        ranges = [
            torch.arange(-n, n + 1, device=cell.device, dtype=cell.dtype)
            for n in k_max
        ]
        grid = torch.cartesian_prod(*ranges)
    else:
        index_range = torch.arange(
            -k_max, k_max + 1, device=cell.device, dtype=cell.dtype
        )
        grid = torch.cartesian_prod(index_range, index_range, index_range)
    k = grid @ _reciprocal_vectors(cell)
    k_squared = (k * k).sum(dim=-1)
    nonzero = grid.abs().sum(dim=-1) > 0
    return k, k_squared, nonzero


def _half_wave_vectors(
    k_max: Union[int, Tuple[int, int, int]], cell: Tensor
) -> tuple[Tensor, Tensor]:
    """One wave vector of each pair ``{k, -k}``, ``k != 0``: ``(k, k_squared)``.

    The integer coordinates ``n`` of the grid of :func:`_wave_vectors` whose first non-zero entry
    is positive: ``n_1 > 0``; or ``n_1 = 0, n_2 > 0``; or ``n_1 = n_2 = 0, n_3 > 0``.  Exactly one
    of ``n`` and ``-n`` qualifies and the origin does not, so the size is the static
    ``((2a+1)(2b+1)(2c+1) - 1) / 2``.  A lattice sum whose summand is invariant under
    ``k -> -k`` -- every real one: ``w(-k) = w(k)`` times an even factor with ``cos`` or an odd
    one with ``sin`` -- is exactly twice the sum over these
    (tests/electrostatics/test_lattice_half_space.py).
    """
    a, b, c = (k_max,) * 3 if isinstance(k_max, int) else tuple(int(n) for n in k_max)

    def span(lo: int, hi: int) -> Tensor:
        return torch.arange(lo, hi + 1, device=cell.device, dtype=cell.dtype)

    grid = torch.cat(
        [
            torch.cartesian_prod(span(1, a), span(-b, b), span(-c, c)),
            torch.cartesian_prod(span(0, 0), span(1, b), span(-c, c)),
            torch.cartesian_prod(span(0, 0), span(0, 0), span(1, c)),
        ]
    )
    k = grid @ _reciprocal_vectors(cell)
    return k, (k * k).sum(dim=-1)


def _gaussian_weights(k_squared: Tensor, nonzero: Tensor, alpha: float) -> Tensor:
    """``exp(-k^2 / 4 alpha^2) / k^2``, zero at the origin.

    ``k^2`` is only ever zero at the origin, which ``nonzero`` removes;
    substituting the *input* keeps both the value and the gradient of the
    discarded term finite, where masking the output would send ``0 * inf`` into
    the cotangent.

    These four lines used to be copy-pasted into both :func:`ewald_recip` and
    :func:`ewald_recip_multipole` -- a place where the Gaussian-weight
    convention could drift between the two routes unnoticed.  They now live
    here, and the charge-only sum reaches them by delegating; this is the single
    point where ``k = 0`` is excluded.
    """
    k_squared_safe = torch.where(nonzero, k_squared, torch.ones_like(k_squared))
    gaussian = torch.exp(-k_squared_safe / (4.0 * alpha * alpha))
    return torch.where(nonzero, gaussian / k_squared_safe, torch.zeros_like(k_squared))


def ewald_real(
    q: Tensor,
    pos: Tensor,
    cell: Tensor,
    edge_index: Tensor,
    edge_shift: Tensor | None,
    params: EwaldParams,
    *,
    full_neighbor_list: bool = True,
) -> Tensor:
    """Real-space (screened) part of the Ewald sum.

        U_real = (K_E / 2) sum_{i,j,n}' q_i q_j erfc(alpha r_ijn) / r_ijn

    The complementary error function makes the summand negligible beyond
    ``r_cut``, so the sum runs over a neighbour list.  Self-image pairs
    (``i == j`` with a non-zero shift) are legitimate members of that list and
    are included; the ``i == j``, zero-shift term is not part of this sum and is
    accounted for by :func:`ewald_self`.

    NO DISTANCE MASK.  Every pair the caller's list contains is summed, however
    far away -- ``params.r_cut`` is the distance the list must *cover*, not a
    second cut applied on top of it.  This used to apply a hard
    ``r <= params.r_cut`` mask, which put a step of ``K_E q_i q_j erfc(alpha
    r_cut) / r_cut`` into the energy and a matching delta into the force at
    exactly ``r_cut``: measured 7.150544e-03 eV per unit ``q_i q_j`` at
    ``alpha = 0.35, r_cut = 6``, 3.523243e-07 eV at the ``accuracy = 1e-6``
    default alpha, and 9.5e-01 eV at a hand-supplied ``alpha = 0.1`` -- the
    error is unbounded as alpha falls, because nothing ties a caller's alpha to
    ``r_cut``.  Dropping the mask is both smoother and strictly more accurate,
    and it is the same argument :func:`_pair_geometry` already makes in full;
    the multipole route never had the mask.  Whatever step remains belongs to
    the neighbour list's own radius, exactly as it does for any short-ranged
    potential, and is the caller's to choose.

    Args:
        q: Charges in e, shape ``[n_atoms]``.
        pos: Cartesian positions in A, shape ``[n_atoms, 3]``.
        cell: ``[3, 3]`` lattice vectors as rows.
        edge_index: ``[2, n_edges]``; the pair energy is symmetric in the two
            rows, so the sender/receiver convention does not matter here.
        edge_shift: ``[n_edges, 3]`` integer image offsets, applied as
            ``edge_shift @ cell`` exactly as in ``flashace.model``.  ``None``
            means every edge is inside the primary cell.
        params: Splitting parameters.  Only ``alpha`` is read; ``r_cut`` is a
            statement about what the list must cover and is not applied here.
        full_neighbor_list: True if the list contains every ordered pair, i.e.
            both ``(i, j, n)`` and ``(j, i, -n)``; then the sum is halved.  Set
            False for a half list that stores each pair once.

    Returns:
        Scalar energy in eV.
    """
    r_vec = pos[edge_index[0]] - pos[edge_index[1]]
    if edge_shift is not None:
        r_vec = r_vec + edge_shift.to(dtype=pos.dtype, device=pos.device) @ cell
    r = r_vec.norm(dim=-1)

    # The only mask is the NaN shield; see the docstring for why there is no
    # `r <= params.r_cut` cut here.  Sanitise the input to the division, not
    # just the output: masking the output still lets an infinite derivative from
    # the untaken branch reach the gradient as 0 * inf = NaN.
    inside = r > DEGENERATE_SEPARATION
    r_safe = torch.where(inside, r, torch.ones_like(r))

    pair = q[edge_index[0]] * q[edge_index[1]] * torch.erfc(params.alpha * r_safe) / r_safe
    pair = torch.where(inside, pair, torch.zeros_like(pair))

    prefactor = 0.5 * K_E if full_neighbor_list else K_E
    return prefactor * pair.sum()


def ewald_recip(q: Tensor, pos: Tensor, cell: Tensor, params: EwaldParams) -> Tensor:
    """Reciprocal-space part of the Ewald sum.

        U_recip = (2 pi K_E / V) sum_{k != 0} exp(-k^2 / 4 alpha^2) |S(k)|^2 / k^2
        S(k)    = sum_j q_j exp(i k . r_j)

    This is the energy of the smooth Gaussian-screened charge density, summed
    exactly over the reciprocal lattice.  ``k = 0`` is excluded; its would-be
    divergence for a charged cell is cancelled by the neutralising background of
    :func:`ewald_jellium`.

    Both sign pairs ``+k`` and ``-k`` are summed, which is what the ``2 pi / V``
    prefactor assumes.

    Cost and memory are ``O(n_atoms * n_k)`` with ``n_k = (2 k_max + 1)^3``; the
    structure factor is formed as a single matmul, and autograd keeps the phase
    matrix alive, so very large ``k_max`` with very many atoms should be blocked
    by the caller.

    IMPLEMENTED BY DELEGATION to :func:`ewald_recip_multipole` with
    ``dipoles=None``, which is term for term this expression -- the two used to
    be written out separately, and two hand-maintained copies of a reciprocal
    sum is precisely the pair that drifts.  The delegation is bit-identical: the
    dipole branch is skipped entirely, and the remaining operations are the same
    ones in the same order.

    Args:
        q: Charges in e, shape ``[n_atoms]``.
        pos: Cartesian positions in A, shape ``[n_atoms, 3]``.
        cell: ``[3, 3]`` lattice vectors as rows.
        params: Splitting parameters.

    Returns:
        Scalar energy in eV.
    """
    return ewald_recip_multipole(q, None, pos, cell, params)


def ewald_self(q: Tensor, params: EwaldParams) -> Tensor:
    """Self-interaction correction.

        U_self = -(K_E alpha / sqrt(pi)) sum_i q_i^2

    The reciprocal sum includes the interaction of every screening Gaussian
    with the point charge at its own centre.  That is an artefact of the
    splitting and is removed here.  It is independent of geometry and of the
    cell, so it contributes to neither forces nor stress -- but it is very much
    part of the energy, and of the derivative with respect to the charges.

    Args:
        q: Charges in e, shape ``[n_atoms]``.
        params: Splitting parameters.

    Returns:
        Scalar energy in eV.
    """
    return -(K_E * params.alpha / SQRT_PI) * (q * q).sum()


def ewald_jellium(q_total: Tensor, volume: Tensor, params: EwaldParams) -> Tensor:
    """Neutralising-background ("jellium") correction for a charged cell.

        U_jellium = -K_E pi Q^2 / (2 alpha^2 V)

    Omitting ``k = 0`` from the reciprocal sum implicitly immerses the cell in a
    uniform background of charge ``-Q``; this is the energy of that background
    interacting with the charges and with itself.  It vanishes for a neutral
    cell.

    For a charged cell it is *not* optional.  It is constant in the charge
    distribution -- ``Q`` is fixed by the per-fragment constraints, so it adds
    nothing to the gradient that drives the charges -- but it contributes
    energy, and through ``V`` it contributes an isotropic pressure
    ``dU/dV = +K_E pi Q^2 / (2 alpha^2 V^2)``.  It is also what makes the total
    independent of ``alpha`` when ``Q != 0``.

    **It makes the sum converge; it does not make it physical.**  The result is
    the energy of the defect plus its own periodic images plus the background,
    and that is not the energy of an isolated charged defect: the difference
    decays only as ``1/L`` and is 1.7 eV at a 12 A cell for ``Q = +1 e``.
    :func:`makov_payne_correction` is the opt-in estimate of it.

    Args:
        q_total: Net charge ``Q`` of the cell in e (scalar tensor).
        volume: Cell volume in A^3 (scalar tensor); keep it as
            ``torch.det(cell).abs()`` so the stress contribution survives.
        params: Splitting parameters.

    Returns:
        Scalar energy in eV.
    """
    return -(K_E * math.pi) * (q_total * q_total) / (2.0 * params.alpha * params.alpha * volume)


def ewald_terms(
    q: Tensor,
    pos: Tensor,
    cell: Tensor,
    edge_index: Tensor,
    edge_shift: Tensor | None,
    params: EwaldParams,
    *,
    full_neighbor_list: bool = True,
) -> EwaldTerms:
    """Compute the four Ewald contributions separately.

    Useful for the alpha-independence check (the terms move, the sum does not)
    and for diagnosing a mis-tuned splitting.  :func:`ewald_energy` is the same
    computation with the terms added up.
    """
    real = ewald_real(
        q, pos, cell, edge_index, edge_shift, params, full_neighbor_list=full_neighbor_list
    )
    recip = ewald_recip(q, pos, cell, params)
    self_energy = ewald_self(q, params)
    jellium = ewald_jellium(q.sum(), torch.det(cell).abs(), params)
    return EwaldTerms(real=real, recip=recip, self_energy=self_energy, jellium=jellium)


def ewald_energy(
    q: Tensor,
    pos: Tensor,
    cell: Tensor,
    edge_index: Tensor,
    edge_shift: Tensor | None,
    params: EwaldParams,
    *,
    full_neighbor_list: bool = True,
) -> Tensor:
    """Total periodic electrostatic energy of a set of point charges, in eV.

        U = U_real + U_recip + U_self + U_jellium

    Differentiable with respect to ``q``, ``pos`` and ``cell``.  The result is
    independent of ``params.alpha`` to the truncation accuracy of ``r_cut`` and
    ``k_max``; see :func:`ewald_terms`.

    Args:
        q: Charges in e, shape ``[n_atoms]``.
        pos: Cartesian positions in A, shape ``[n_atoms, 3]``.
        cell: ``[3, 3]`` lattice vectors as rows.
        edge_index: ``[2, n_edges]`` neighbour list covering ``params.r_cut``.
        edge_shift: ``[n_edges, 3]`` integer image offsets.
        params: Splitting parameters, e.g. from :meth:`EwaldParams.auto_tune`.
        full_neighbor_list: See :func:`ewald_real`.
    """
    return ewald_terms(
        q, pos, cell, edge_index, edge_shift, params, full_neighbor_list=full_neighbor_list
    ).total


# ---------------------------------------------------------------------------
# Point charges AND point dipoles
# ---------------------------------------------------------------------------


class EwaldMultipoleTerms(NamedTuple):
    """The five contributions to the damped multipole lattice sum, in eV.

    The first four are the bare point-multipole Ewald sum and each depends
    strongly on ``EwaldParams.alpha``; their sum does not.  ``damping`` is the
    model's short-ranged correction -- Gaussian smearing of the charges, Thole
    smearing of the dipoles -- and contains no alpha at all, which is what makes
    the alpha-independence check still bite once the dipoles are switched on.

    A ``NamedTuple`` rather than a frozen dataclass, for the compile reason
    given on :class:`EwaldTerms`.
    """

    real: Tensor
    recip: Tensor
    self_energy: Tensor
    jellium: Tensor
    damping: Tensor

    @property
    def bare(self) -> Tensor:
        """The point-multipole lattice sum, without the model's damping."""
        return self.real + self.recip + self.self_energy + self.jellium

    @property
    def total(self) -> Tensor:
        # Spelled out rather than `self.bare + self.damping`: Dynamo cannot read
        # one property from inside another (`call_function BuiltinVariable(add)
        # [GetAttrVariable(NamedTupleVariable(), bare), ...]`), and that one line
        # was enough to block fullgraph compilation of every caller.
        return self.real + self.recip + self.self_energy + self.jellium + self.damping


def _lattice_spacings(cell: Tensor) -> Tensor:
    """Perpendicular spacing between successive lattice planes, in A.

    ``d_i = |det cell| / |a_j x a_k| = 2 pi / |b_i|``: the distance one has to
    travel along the normal of the plane spanned by the other two lattice
    vectors to reach the next image.  This, not ``|a_i|``, is what decides how
    many image shells a given real-space cutoff needs -- on a strongly sheared
    cell the two differ by a large factor.
    """
    return TWO_PI / _reciprocal_vectors(cell).norm(dim=-1)


def _image_shell_counts(cell: Tensor, r_cut: float) -> Tuple[int, int, int]:
    """Per-direction half-widths of the real-space image box, as Python ints.

        n_max_d = ceil(r_cut / spacing_d + MINIMUM_IMAGE_MARGIN)

    so that the box contains every periodic image lying within ``r_cut`` of the
    minimum image of any pair.

    THIS IS A HOST SYNCHRONISATION and it belongs at tune time, not in the inner
    loop.  It reads values out of the cell tensor, so calling it inside a
    compiled region forces a device-to-host copy per call and makes the
    summation shape data-dependent, which ``torch.compile(fullgraph=True)``
    rejects outright.  :meth:`EwaldParams.auto_tune` calls it once and stores
    the answer in ``EwaldParams.image_shells``; :func:`_pair_geometry` falls
    back to it only when a hand-built ``EwaldParams`` left that field None.
    """
    spacings = _lattice_spacings(cell.detach().to(dtype=torch.float64))
    counts = [
        max(1, int(math.ceil(r_cut / float(spacing) + MINIMUM_IMAGE_MARGIN)))
        for spacing in spacings
    ]
    return (counts[0], counts[1], counts[2])


def _image_offsets(cell: Tensor, shells: Tuple[int, int, int]) -> Tensor:
    """Integer lattice offsets of the real-space image box, ``[n_images, 3]``.

    Every ``n`` in ``[-shells[d], shells[d]]`` per direction ``d`` is returned,
    including the origin.  ``shells`` is a tuple of plain Python ints, so the
    returned shape is a compile-time constant: nothing is read out of ``cell``
    here, and nothing differentiates through the integers -- exactly the status
    ``k_max`` already has in :func:`_wave_vectors`.

    The shape is static **per shell count**, not per cell, and that distinction
    matters.  The count is a step function of the cell, so a cell change that
    crosses a step changes the shape and, under ``torch.compile``, triggers a
    recompilation: at ``r_cut = 6`` a cubic cell of ``L = 3.9`` needs 343
    images, ``L = 4.0`` needs 125 and ``L = 12.0`` needs 27.  Pinning the count
    on :class:`EwaldParams` is what makes the shape static across a strain
    trajectory rather than merely static within one call.

    The energy step across such a flip is the size of the neglected tail, in the
    same way and for the same reason that rebuilding a neighbour list steps it.
    At the cutoffs this module is used with it is unmeasurable.  Isolated by
    holding the cell fixed and varying only the count, at ``alpha = 0.8`` and
    ``L`` on either side of 12: ``1^3`` versus ``2^3`` versus ``3^3`` shells
    move the energy by at most 1.8e-15 eV, i.e. by round-off.  The physics
    across the flip is fine; what the flip costs is a recompilation.
    """
    ranges = [
        torch.arange(-n, n + 1, device=cell.device, dtype=cell.dtype) for n in shells
    ]
    return torch.cartesian_prod(*ranges)


def _pair_geometry(
    pos: Tensor, cell: Tensor, params: EwaldParams
) -> tuple[Tensor, Tensor, Tensor]:
    """Displacements of every ordered pair in every image of the box.

    Returns ``(r_vec, r, included)`` of shapes ``[n_images, n, n, 3]``,
    ``[n_images, n, n]`` and ``[n_images, n, n]``:

        r_vec[c, i, j] = r_i - r_j + (n_c - round(f_ij)) @ cell

    Two deliberate choices:

    * **Minimum image first.**  ``round(f_ij)``, the nearest integer of the
      fractional displacement, is taken from the detached positions and is a
      constant -- shifting a pair by a lattice vector cannot change the energy,
      so this only recentres the window of images on the closest one.  It makes
      the box size a function of the cell alone and keeps unwrapped
      trajectories correct.
    * **No distance mask.**  Every member of the box is summed, however far
      away, rather than being cut at ``r_cut``.  The summand is exponentially
      small out there, and a distance mask would put a step discontinuity into
      the energy at exactly the radius where a strain derivative is most likely
      to cross it.  Keeping all of them is both smoother and strictly more
      accurate.

    ``included`` removes the ``i == j`` zero-shift self pair, which
    :func:`ewald_self` and :func:`ewald_self_dipole` account for analytically,
    and any exactly coincident pair as a NaN shield.  Where it is False the
    distance is set to 1 A **before** the square root, so neither the value nor
    the derivative of a discarded term can be infinite.

    Cost is ``O(n_atoms^2 * n_images)`` in time and memory -- this is the
    all-pairs route of :func:`trace_electrostatics.energy._direct_pair_blocks`
    repeated over the image box, not a neighbour-list sum.  It is intended for
    the cell sizes a periodic training example uses; shrinking
    ``EwaldParams.r_cut`` (and raising ``k_max`` to match) is the lever.

    The box extent comes from ``params.image_shells`` when it is set -- a plain
    Python tuple, so the shape is a compile-time constant -- and otherwise from
    a per-call :func:`_image_shell_counts`, which synchronises with the host and
    blocks ``fullgraph=True``.  Use :meth:`EwaldParams.auto_tune` to get the
    former.
    """
    n_atoms = int(pos.shape[0])
    shells = params.image_shells
    if shells is None:
        shells = _image_shell_counts(cell, params.r_cut)
    offsets = _image_offsets(cell, shells)                    # [n_images, 3]

    # One inverse, reused: this used to invert the cell here and again inside
    # _lattice_spacings on every call.
    delta = pos.unsqueeze(1) - pos.unsqueeze(0)               # [n, n, 3], r_i - r_j
    fractional = delta.detach() @ torch.linalg.inv(cell.detach())
    minimum_image = torch.round(fractional)                   # integer, constant

    shifts = offsets.view(-1, 1, 1, 3) - minimum_image.unsqueeze(0)
    r_vec = delta.unsqueeze(0) + shifts @ cell                # [n_images, n, n, 3]

    at_origin = (offsets.abs().sum(dim=-1) == 0).view(-1, 1, 1)
    same_atom = torch.eye(n_atoms, dtype=torch.bool, device=pos.device).view(
        1, n_atoms, n_atoms
    )
    squared = (r_vec * r_vec).sum(dim=-1)
    included = ~(at_origin & same_atom) & (
        squared > DEGENERATE_SEPARATION * DEGENERATE_SEPARATION
    )

    # Substitute the *input* of the square root, not its output: a masked-away
    # branch still propagates d sqrt(x) / dx = inf into the cotangent otherwise.
    r = torch.sqrt(torch.where(included, squared, torch.ones_like(squared)))
    return r_vec, r, included


def _screened_kernels(r: Tensor, alpha: float) -> tuple[Tensor, Tensor, Tensor]:
    """``(B_0, B_1, B_2)`` of the real-space Ewald kernel ``erfc(alpha r) / r``.

        B_0   = erfc(alpha r) / r
        B_n+1 = [ (2n + 1) B_n
                  + (2 alpha^2)^n (2 alpha / sqrt(pi)) e^{-alpha^2 r^2} ] / r^2

    with ``B_n = (-1/r d/dr)^n B_0``, so that the gradient and Hessian of the
    screened potential are ``grad_a = -r_a B_1`` and
    ``grad_a grad_b = -delta_ab B_1 + r_a r_b B_2``.  In the unscreened limit
    ``alpha -> 0`` these become ``1/r``, ``1/r^3`` and ``3/r^5``.

    ``r`` must already be sanitised (see :func:`_pair_geometry`): the recurrence
    divides by ``r^2``.
    """
    inverse_r2 = 1.0 / (r * r)
    gaussian = (2.0 * alpha / SQRT_PI) * torch.exp(-((alpha * r) ** 2))

    b0 = torch.erfc(alpha * r) / r
    b1 = (b0 + gaussian) * inverse_r2
    b2 = (3.0 * b1 + (2.0 * alpha * alpha) * gaussian) * inverse_r2
    return b0, b1, b2


def ewald_real_multipole(
    charges: Tensor,
    dipoles: Tensor | None,
    pos: Tensor,
    cell: Tensor,
    sigma: Tensor,
    alpha_volume: Tensor | None,
    params: EwaldParams,
    *,
    thole_a: float = DEFAULT_THOLE_A,
) -> tuple[Tensor, Tensor]:
    """Real-space part of the multipole lattice sum, and its damping correction.

    Returns ``(screened, damping)``:

    * ``screened`` is the short-ranged half of the *bare* point-multipole
      lattice sum, the partner of :func:`ewald_recip_multipole`::

          (K_E/2) q_i q_j B_0
          + K_E q_i (mu_j . r) B_1
          + (K_E/2) [ (mu_i . mu_j) B_1 - (mu_i . r)(mu_j . r) B_2 ]

      summed over every ordered pair and every image except the ``i == j``
      zero-shift one.  The charge-charge and dipole-dipole blocks are halved
      because the ordered sum visits each pair twice; the charge-dipole block is
      not, because ``(q_i, mu_j)`` and ``(q_j, mu_i)`` are physically distinct
      terms.  This is the convention of
      ``ElectrostaticAssembly._direct_pair_energy`` and the two must agree.

    * ``damping`` is the model's short-ranged correction, the damped kernels
      minus the bare ones::

          (K_E/2) q_i q_j [ erf(r/sigma_ij) - 1 ] / r
          + K_E q_i (mu_j . r) (lambda_3 - 1) / r^3
          + (K_E/2) [ (mu_i . mu_j)(lambda_3 - 1) / r^3
                      - 3 (mu_i . r)(mu_j . r)(lambda_5 - 1) / r^5 ]

      Each bracket vanishes exponentially -- ``erf - 1 = -erfc(r/sigma_ij)`` and
      ``lambda_3 - 1 = -exp(-a r^3 / sqrt(alpha_i alpha_j))`` -- so this is a
      genuinely short-ranged sum that needs no reciprocal-space partner.  It is
      built from the very kernels the open-boundary route uses
      (:func:`trace_electrostatics.kernels.coulomb_kernel`,
      :func:`~trace_electrostatics.kernels.thole_damping`) so that the two
      routes cannot drift apart.

    The two are returned separately rather than added because the split is the
    only way to check the dipolar sum against the charge-only implementation:
    with ``dipoles = 0`` the four *bare* terms must reproduce
    :func:`ewald_energy`, while ``damping`` is a property of the charge model
    rather than of the lattice sum.

    How closely they must agree is a TRUNCATION question, not a round-off one,
    and the two routes truncate differently: this one sums a whole image box,
    :func:`ewald_energy` sums the caller's neighbour list.  Measured on a 6 A
    cubic cell with four charges, ``r_cut = 12`` and a list reaching 16 A:

        alpha    gap
        0.30     8.8e-12 eV
        0.35     8.9e-16 eV
        0.45     8.9e-16 eV

    i.e. round-off once alpha is high enough that ``erfc(alpha r)`` has died
    inside the list, and an honest ``erfc`` tail below that.  This docstring
    used to call the residual "round-off" unconditionally, which hid an
    alpha-dependent truncation behind a word that implies 1e-16.  It also used
    to be much worse than the table above: the same three cases gave 9.9e-07,
    7.0e-09 and 3.8e-14 eV while :func:`ewald_real` still applied a hard
    ``r <= params.r_cut`` mask and threw away the part of the caller's list that
    reached past ``r_cut``.  Tighten alpha or lengthen the list to close what
    remains.

    Args:
        charges: ``[n_atoms]`` charges in e.
        dipoles: ``[n_atoms, 3]`` dipoles in e*A, or None for charges only.
        pos: ``[n_atoms, 3]`` Cartesian positions in A.  They need not be
            wrapped into the cell.
        cell: ``[3, 3]`` lattice vectors as rows.
        sigma: ``[n_atoms]`` Gaussian charge widths in A.
        alpha_volume: ``[n_atoms]`` polarizability volumes in A^3 for the Thole
            reduced distance (SPEC 0 -- a volume, not a response
            polarizability).  Required when ``dipoles`` is given.
        params: splitting parameters; ``r_cut`` sizes the image box.
        thole_a: Thole screening parameter, ``0 < a < 1``.

    Returns:
        ``(screened, damping)``, both scalar tensors in eV.
    """
    r_vec, r, included = _pair_geometry(pos, cell, params)
    n_atoms = int(pos.shape[0])

    keep = included.to(dtype=pos.dtype)

    charge_i = charges.view(1, n_atoms, 1)
    charge_j = charges.view(1, 1, n_atoms)
    charge_pair = charge_i * charge_j

    b0, b1, b2 = _screened_kernels(r, params.alpha)

    # -- charge-charge -----------------------------------------------------
    screened = 0.5 * K_E * (keep * charge_pair * b0).sum()

    sigma_pair = combine_sigma(sigma.view(1, n_atoms, 1), sigma.view(1, 1, n_atoms))
    # coulomb_kernel(r) - K_E / r = -K_E erfc(r / sigma_ij) / r, written as the
    # difference so that the damped kernel is literally the one the
    # open-boundary route evaluates.
    smearing = coulomb_kernel(r, sigma_pair) - K_E / r
    damping = 0.5 * (keep * charge_pair * smearing).sum()

    if dipoles is None:
        return screened, damping

    if alpha_volume is None:
        raise ValueError(
            "ewald_real_multipole needs alpha_volume (polarizability volumes in "
            "A^3) whenever dipoles are supplied: it sets the Thole reduced "
            "distance, and there is no sensible default for it."
        )

    _require_matching_dtype(dipoles, "dipoles", pos, "pos")

    dipole_i = dipoles.view(1, n_atoms, 1, 3)
    dipole_j = dipoles.view(1, 1, n_atoms, 3)
    mu_i_dot_r = (dipole_i * r_vec).sum(dim=-1)
    mu_j_dot_r = (dipole_j * r_vec).sum(dim=-1)
    mu_dot_mu = (dipole_i * dipole_j).sum(dim=-1)

    # -- charge-dipole and dipole-dipole, bare ------------------------------
    cross = K_E * (keep * charge_i * mu_j_dot_r * b1).sum()
    relay = 0.5 * K_E * (keep * (mu_dot_mu * b1 - mu_i_dot_r * mu_j_dot_r * b2)).sum()
    screened = screened + cross + relay

    # -- the same two blocks, damped minus bare -----------------------------
    lam3, lam5 = thole_damping(
        r,
        alpha_volume.view(1, n_atoms, 1),
        alpha_volume.view(1, 1, n_atoms),
        thole_a,
    )
    inverse_r3 = 1.0 / (r * r * r)
    inverse_r5 = inverse_r3 / (r * r)
    delta_3 = (lam3 - 1.0) * inverse_r3
    delta_5 = (lam5 - 1.0) * inverse_r5

    cross_damping = K_E * (keep * charge_i * mu_j_dot_r * delta_3).sum()
    relay_damping = (
        0.5
        * K_E
        * (keep * (mu_dot_mu * delta_3 - 3.0 * mu_i_dot_r * mu_j_dot_r * delta_5)).sum()
    )
    return screened, damping + cross_damping + relay_damping


def ewald_recip_multipole(
    charges: Tensor,
    dipoles: Tensor | None,
    pos: Tensor,
    cell: Tensor,
    params: EwaldParams,
) -> Tensor:
    """Reciprocal-space part of the point-charge plus point-dipole sum.

        U_recip = (2 pi K_E / V) sum_{k != 0} exp(-k^2 / 4 alpha^2) |S(k)|^2 / k^2
        S(k)    = sum_j ( q_j + i k . mu_j ) exp(i k . r_j)

    Written out over the real and imaginary parts, with ``phi_j = k . r_j``,

        Re S = sum_j [ q_j cos phi_j - (k . mu_j) sin phi_j ]
        Im S = sum_j [ q_j sin phi_j + (k . mu_j) cos phi_j ]

    so a zero dipole recovers :func:`ewald_recip` expression by expression, and
    not merely up to algebra.  ``k = 0`` is excluded: see the tinfoil discussion
    in the module docstring, and :func:`ewald_jellium` for the monopole part of
    what that omission implies.

    The dipoles enter through ``k . mu_j`` only, so the cost is one extra
    ``[n_atoms, n_k]`` matmul over the charge-only sum.

    Args:
        charges: ``[n_atoms]`` charges in e.
        dipoles: ``[n_atoms, 3]`` dipoles in e*A, or None for charges only.
        pos: ``[n_atoms, 3]`` positions in A.
        cell: ``[3, 3]`` lattice vectors as rows.
        params: splitting parameters.

    Returns:
        Scalar energy in eV.
    """
    k, k_squared, nonzero = _wave_vectors(params.k_grid, cell)
    weight = _gaussian_weights(k_squared, nonzero, params.alpha)

    phase = pos @ k.transpose(-1, -2)                         # [n_atoms, n_k]
    cosine, sine = torch.cos(phase), torch.sin(phase)
    charge = charges.unsqueeze(-1)

    real_part = (charge * cosine).sum(dim=0)
    imaginary_part = (charge * sine).sum(dim=0)

    if dipoles is not None:
        _require_matching_dtype(dipoles, "dipoles", cell, "cell")
        k_dot_mu = dipoles @ k.transpose(-1, -2)              # [n_atoms, n_k]
        real_part = real_part - (k_dot_mu * sine).sum(dim=0)
        imaginary_part = imaginary_part + (k_dot_mu * cosine).sum(dim=0)

    structure = real_part * real_part + imaginary_part * imaginary_part
    volume = torch.det(cell).abs()
    return (TWO_PI * K_E / volume) * (weight * structure).sum()


def ewald_self_dipole(dipoles: Tensor, params: EwaldParams) -> Tensor:
    """Self-interaction correction for the dipoles.

        U_self_dipole = -(2 K_E alpha^3 / (3 sqrt(pi))) sum_i |mu_i|^2

    The reciprocal sum lets every dipole interact with its own screening cloud.
    Taking the smooth kernel ``G(r) = erf(alpha r) / r`` and expanding it about
    the origin,

        G(r) = (2 alpha / sqrt(pi)) [ 1 - (alpha r)^2 / 3 + O(r^4) ]

    only the ``r^2`` term survives two derivatives at ``r = 0``, giving
    ``d_a d_b G|_0 = -(4 alpha^3 / (3 sqrt(pi))) delta_ab``.  The spurious
    energy is ``(1/2) (-K_E) mu_a mu_b d_a d_b G|_0 = +2 K_E alpha^3 |mu|^2 /
    (3 sqrt(pi))`` per site, and this function returns minus that.

    Note the ``alpha^3``: this term is NOT the charge self energy with a
    different prefactor, and there is no charge-dipole self term at all
    (``grad G|_0 = 0``, because ``G`` is even).  Like :func:`ewald_self` it is
    independent of geometry and of the cell, so it contributes to neither forces
    nor stress -- but it is part of the energy and of the derivative with
    respect to the dipoles, which is what the variational solve sees.

    Args:
        dipoles: ``[n_atoms, 3]`` dipoles in e*A.
        params: splitting parameters.

    Returns:
        Scalar energy in eV.
    """
    alpha = params.alpha
    coefficient = DIPOLE_SELF_COEFFICIENT * K_E * alpha * alpha * alpha / SQRT_PI
    return -coefficient * (dipoles * dipoles).sum()


def ewald_multipole_terms(
    charges: Tensor,
    dipoles: Tensor | None,
    pos: Tensor,
    cell: Tensor,
    sigma: Tensor,
    alpha_volume: Tensor | None,
    params: EwaldParams,
    *,
    thole_a: float = DEFAULT_THOLE_A,
) -> EwaldMultipoleTerms:
    """Compute the five contributions to the damped multipole sum separately.

    Useful for the alpha-independence check -- ``real``, ``recip`` and
    ``self_energy`` move by many eV over a sweep while ``bare`` does not, and
    ``damping`` does not contain alpha at all -- and for comparing the
    zero-dipole limit against :func:`ewald_energy`.
    :func:`ewald_multipole_energy` is the same computation, summed.

    NOT merged with :func:`ewald_terms`, despite the parallel shape.  The two
    are different algorithms over the same physics: :func:`ewald_terms` sums a
    caller-supplied neighbour list at ``O(n_edges)``, this one sums every
    ordered pair in an image box at ``O(n_atoms^2 n_images)``.  Collapsing them
    would force the cheap route to build the expensive geometry.  What they DO
    share -- the reciprocal sum, the self energy, the jellium term -- is already
    shared by call, not by copy, which is where drift would actually start.
    """
    screened, damping = ewald_real_multipole(
        charges,
        dipoles,
        pos,
        cell,
        sigma,
        alpha_volume,
        params,
        thole_a=thole_a,
    )
    recip = ewald_recip_multipole(charges, dipoles, pos, cell, params)
    self_energy = ewald_self(charges, params)
    if dipoles is not None:
        self_energy = self_energy + ewald_self_dipole(dipoles, params)
    jellium = ewald_jellium(charges.sum(), torch.det(cell).abs(), params)
    return EwaldMultipoleTerms(
        real=screened,
        recip=recip,
        self_energy=self_energy,
        jellium=jellium,
        damping=damping,
    )


def ewald_multipole_energy(
    charges: Tensor,
    dipoles: Tensor | None,
    pos: Tensor,
    cell: Tensor,
    sigma: Tensor,
    alpha_volume: Tensor | None,
    params: EwaldParams,
    *,
    thole_a: float = DEFAULT_THOLE_A,
) -> Tensor:
    """Periodic energy of Gaussian-smeared charges and Thole-damped dipoles, in eV.

        U = U_real + U_recip + U_self + U_jellium + U_damping

    This is the lattice sum of exactly the three blocks
    ``ElectrostaticAssembly._direct_pair_energy`` sums over all pairs at open
    boundaries -- charge-charge, charge-dipole, dipole-dipole -- with the
    ``i == j`` on-site terms excluded, because SPEC 1.2 and 1.5 fold those into
    ``eta_i`` and ``A_mu_i``.

    Conducting (tinfoil) boundary conditions; no surface term.  See the module
    docstring for the boundary condition, for the derivation of the
    ``alpha^3 |mu|^2`` dipole self energy, and for why the Thole and Gaussian
    damping live in the real-space sum only.

    Differentiable with respect to ``charges``, ``dipoles``, ``pos`` **and**
    ``cell``.  The cell enters through the image displacements
    ``shift @ cell``, the reciprocal vectors ``2 pi (cell^-1)^T``, the phases,
    and the volume; nothing about it is detached except the integer image
    offsets and the integer minimum-image shift, which cannot contribute a
    derivative because the energy is invariant under a change of either.

    The result is independent of ``params.alpha`` to the truncation accuracy of
    the image box and ``k_max``; see :func:`ewald_multipole_terms`.

    Args:
        charges: ``[n_atoms]`` charges in e.
        dipoles: ``[n_atoms, 3]`` dipoles in e*A (physical units, not the
            reduced ``mu_hat`` of SPEC 1.5), or None for a charge-only sum.
        pos: ``[n_atoms, 3]`` positions in A.
        cell: ``[3, 3]`` lattice vectors as rows.
        sigma: ``[n_atoms]`` Gaussian charge widths in A.  The image box is
            sized by ``params.r_cut`` alone -- a hyperparameter, so that nothing
            has to synchronise on a head output -- and ``r_cut`` therefore has
            to cover the ``erfc(r / sigma_ij)`` tail of the smearing correction
            as well as the ``erfc(alpha r)`` tail of the splitting.  The two
            decay as ``exp(-r^2 / sigma_ij^2)`` and ``exp(-alpha^2 r^2)``, so
            the SMEARING tail is the slower one exactly when
            ``alpha > 1 / sigma_ij``; with the head bound ``sigma <= 2 A``,
            i.e. ``sigma_ij <= 2.828 A``, that threshold is
            ``alpha > 0.354 1/A``.  **Below** it the splitting tail is the
            binding one, which is the side a user with a large ``r_cut`` lands
            on: at ``r = 15 A`` and ``sigma_ij = 2.828 A``, ``alpha = 0.30``
            gives ``erfc(alpha r) = 1.97e-10`` against
            ``erfc(r / sigma_ij) = 6.38e-14``, so the splitting tail is four
            orders of magnitude larger.  (An earlier version of this note had
            the inequality the wrong way round.)

            What is actually dropped is smaller than either bound, because no
            distance mask is applied and the box over-covers -- but the size of
            it depends on alpha and quoting a number without one says nothing.
            On a 6 A cubic cell with four charges and ``sigma = 2 A``,
            ``r_cut = 5 A`` (a box reaching past 15 A) differs from
            ``r_cut = 26 A`` by 3.6e-15 eV at ``alpha = 0.5`` and 0.8, and by
            1.0e-12 eV at ``alpha = 0.3``.
        alpha_volume: ``[n_atoms]`` polarizability volumes in A^3, required
            whenever ``dipoles`` is given.
        params: splitting parameters, e.g. from :meth:`EwaldParams.auto_tune`.
        thole_a: Thole screening parameter, ``0 < a < 1``.
    """
    return ewald_multipole_terms(
        charges,
        dipoles,
        pos,
        cell,
        sigma,
        alpha_volume,
        params,
        thole_a=thole_a,
    ).total


# ---------------------------------------------------------------------------
# Finite-size correction for a CHARGED cell -- Makov-Payne
# ---------------------------------------------------------------------------

#: Madelung constant of a point charge in a neutralising background on a SIMPLE
#: CUBIC lattice, in the ``L = V^(1/3)`` convention of :func:`madelung_constant`.
#: DERIVED, not quoted: it is what :func:`madelung_constant` returns for
#: ``torch.eye(3) * L`` at every ``L`` tried (1.0, 3.0, 6.0, 11.7 A -- all twelve
#: digits identical, as they must be, since the constant is scale free) and for
#: every splitting parameter tried (alpha = 0.30, 0.55, 0.80, 1.10 1/A -- again
#: identical to 2.837297479481).  Confirmed against an independent numpy Ewald
#: sum at the self-dual splitting ``alpha = sqrt(pi)`` on the unit cell, which
#: shares no line of code with this module: 2.837297479481, and against the
#: textbook 2.8372974795.
MADELUNG_SIMPLE_CUBIC: Final[float] = 2.837297479481

#: The same constant for body- and face-centred cubic lattices, in the SAME
#: ``L = V^(1/3)`` convention -- i.e. ``L`` is the cube root of the volume per
#: lattice point, NOT the side of the conventional cubic cell.  The usual tables
#: use the conventional cell, whose cube holds 2 (bcc) or 4 (fcc) lattice points,
#: so they quote ``xi * 2^(1/3) = 3.639233`` and ``xi * 4^(1/3) = 4.584862``
#: against the literature 3.639231 and 4.584862.  **That factor is the single
#: most common way to get this wrong**, and it is a 26 % error on fcc.
MADELUNG_BCC: Final[float] = 2.888461503054
MADELUNG_FCC: Final[float] = 2.888282119020

#: ``xi`` for a uniform sphere of background around the charge,
#: ``(9/5) (4 pi / 3)^(1/3)``.  Not a lattice: it is the shape bound.  A point
#: charge in a neutralising background has ``|E| = K_E xi / (2 V^(1/3))``, and the
#: sphere is the shape that maximises it, so every lattice value must come in
#: BELOW this.  Measured: sc 2.8373 < fcc 2.8883 < bcc 2.8885 < 2.9016, in the
#: order of how nearly spherical the Wigner-Seitz cell is.  A cell whose
#: ``madelung_constant`` exceeds this is a bug, not a lattice.
MADELUNG_UNIFORM_SPHERE: Final[float] = 1.8 * (4.0 * math.pi / 3.0) ** (1.0 / 3.0)

#: ``sqrt(Q_r / q) / V^(1/3)`` for a charge spread UNIFORMLY over a cubic cell:
#: each Cartesian coordinate has variance ``L^2 / 12``, so ``<r^2> = L^2 / 4``
#: and the ratio is exactly ``1/2``.  This is the reference point for
#: "completely delocalised" in :class:`DefectMoments`, and it is not
#: hypothetical -- see the module docstring for the measurement on this
#: package's own charges, which land on 0.47-0.62 against this 0.5.
UNIFORM_CELL_RADIUS_RATIO: Final[float] = 0.5

#: Largest ``DefectMoments.radius_ratio`` at which :func:`check_defect_localisation`
#: still accepts the multipole expansion.  The expansion parameter is
#: ``(r_q / L)^2``: the ratio of the ``Q_r`` term to the monopole term is
#: ``(4 pi / 3 xi) (r_q / L)^2``, which at this threshold and cubic ``xi`` is
#: 9.2 %, and the neglected ``O(L^-5)`` remainder is smaller again by roughly the
#: same factor.  Measured on a 7-point defect with ``r_q = 1 A``: at ``L = 8 A``
#: (``r_q/L = 0.125``) the corrected residual is 2.3 % of the ``Q_r`` term it just
#: removed.  Beyond 0.25 the "correction" is no longer small compared with what
#: it corrects, and past 0.5 it is the uniform-charge limit, where there is no
#: expansion left at all.
DEFAULT_MAX_RADIUS_RATIO: Final[float] = 0.25


def madelung_constant(
    cell: Tensor,
    *,
    r_cut: Optional[float] = None,
    accuracy: float = 1.0e-14,
) -> float:
    """``xi`` for this cell: the Madelung constant of a point charge in jellium.

    Defined by the Madelung energy of ONE unit point charge alone in the cell,
    neutralised by a uniform background -- which is exactly what the four terms
    ``real + recip + self + jellium`` of this module already compute:

        E_M = -K_E xi q^2 / (2 L) ,      L = V^(1/3)

    so ``xi = -2 E_M V^(1/3) / (K_E q^2)``.  Nothing new is summed here; the
    lattice sum is the module's own, which is the point -- ``xi`` is then
    guaranteed to be the constant that belongs to the *same* Ewald convention
    (tinfoil, ``k = 0`` dropped, jellium included) that
    :func:`makov_payne_terms` is asked to undo.

    **The convention, which is where non-cubic lattices go wrong.**  ``L`` here
    is ``V^(1/3)``, the cube root of the volume PER LATTICE POINT.  The tables
    in the literature usually use the side of the CONVENTIONAL cubic cell
    instead, which holds 2 points for bcc and 4 for fcc, so they quote
    ``xi_conv = xi * n^(1/3)``: 3.639231 for bcc and 4.584862 for fcc against
    the 2.888462 and 2.888282 of this convention.  Mixing the two is a 26 %
    error on fcc, and it is silent.  The convention here was chosen because it
    is the only one that extends to a cell with no conventional cube at all.

    **And for a non-cubic cell there is no constant to look up.**  ``xi`` is a
    function of the cell SHAPE, and a strongly anisotropic cell is nothing like
    the cubic number.  Measured by this function:

        simple cubic                   2.837297
        bcc / fcc (primitive)          2.888462 / 2.888282
        hexagonal, c/a = 1.63          2.515402
        tetragonal, c/a = 2            2.275218
        tetragonal, c/a = 4           -0.458005
        triclinic, strongly sheared    2.851602
        uniform sphere (upper bound)   2.901586

    At ``c/a = 4`` it has changed SIGN: the in-plane images at 2.5 A dominate
    and the Madelung energy of a charged cell is positive.  Passing
    :data:`MADELUNG_SIMPLE_CUBIC` for such a cell would not merely mis-size the
    correction, it would apply it the wrong way round.  Call this function.

    Cost and accuracy.  One single-atom Ewald sum, so it is cheap, and it is a
    HOST SYNCHRONISATION returning a plain Python float -- a hyperparameter, in
    the same class as ``EwaldParams.alpha``, computed once at setup and never
    differentiated through.  ``xi`` is scale free (it depends on the shape of
    the cell, not its size), so it survives any isotropic strain exactly; see
    :func:`makov_payne_terms` for what holding it fixed costs the deviatoric
    stress.  Converged: over ``r_cut`` from ``1.5 V^(1/3)`` to ``4 V^(1/3)`` the
    returned value moves by at most 2e-15 on every cell in the table above.

    Args:
        cell: ``[3, 3]`` lattice vectors as rows.  Detached; no gradient flows.
        r_cut: Real-space cutoff for the internal sum, in A.  ``None`` uses
            ``2 V^(1/3)``, which is converged for every shape tested.
        accuracy: Target size of the neglected tails of the internal sum.

    Returns:
        ``xi``, dimensionless.

    Raises:
        ValueError: on a singular cell -- ``xi`` is undefined without a volume.
    """
    work = cell.detach().to(dtype=torch.float64, device="cpu")
    volume = float(torch.det(work).abs())
    if not volume > 0.0:
        raise ValueError(
            "madelung_constant needs a non-singular cell (got |det cell| = "
            f"{volume}). A charged cell has no Madelung constant without a "
            "volume, and an all-zero cell means open boundaries (SPEC 7), "
            "where there is no finite-size error to correct in the first place."
        )
    if r_cut is None:
        r_cut = 2.0 * volume ** (1.0 / 3.0)

    params = EwaldParams.auto_tune(work, r_cut, accuracy)
    charges = torch.ones(1, dtype=torch.float64)
    pos = torch.zeros(1, 3, dtype=torch.float64)

    # The charge-charge real-space block, built from the same image geometry and
    # the same screened kernel as ewald_real_multipole -- shared by call, not by
    # copy.  Only B_0 is wanted, and `q_i q_j = 1`.
    _, r, included = _pair_geometry(pos, work, params)
    b0, _, _ = _screened_kernels(r, params.alpha)
    real = 0.5 * K_E * (included.to(dtype=r.dtype) * b0).sum()

    madelung_energy = (
        real
        + ewald_recip_multipole(charges, None, pos, work, params)
        + ewald_self(charges, params)
        + ewald_jellium(charges.sum(), torch.det(work).abs(), params)
    )
    return -2.0 * float(madelung_energy) * volume ** (1.0 / 3.0) / K_E


class DefectMoments(NamedTuple):
    """The moments of a defect charge density, and how localised it is.

    Every field is a tensor so that gradients reach the charges and positions
    they were built from -- except that the centre and the minimum-image shift
    used to find it are detached integers, exactly as in :func:`_pair_geometry`.

    Attributes:
        charge: ``q``, the net defect charge in e.
        centre: ``[3]`` the charge centroid ``sum_i q_i r_i / q``, detached.
        dipole: ``[3]`` ``sum_i q_i (r_i - centre)``.  Zero to round-off when
            the centre is the centroid, which is the default and the reason for
            it; see below.
        second_moment: ``Q_r = sum_i q_i |r_i - centre|^2`` in e A^2.  Signed:
            a defect whose compensating charge sits further out than its core
            has ``Q_r`` of the opposite sign to ``q``.
        charge_radius: ``r_q = sqrt(|Q_r / q|)`` in A, the RMS radius of the
            defect charge.  Undefined (returned as zero) for a neutral defect.
        radius_ratio: ``r_q / V^(1/3)``, dimensionless.  This is the number to
            look at: the multipole expansion is in ``(r_q / L)^2``, and
            :data:`UNIFORM_CELL_RADIUS_RATIO` = 0.5 is what a charge spread
            uniformly over the whole cell gives -- the point at which there is
            no expansion left.

    WHY THE CENTROID.  ``Q_r`` and ``dipole`` are each origin-dependent when
    ``q != 0``: moving the centre by ``a`` sends ``Q_r -> Q_r - 2 a.d + q a^2``
    and ``d -> d - q a``.  The combination that appears in the energy,
    ``q Q_r - |d|^2``, is invariant -- verified to ten digits across three
    centres 4 A apart.  Choosing the centroid makes ``d = 0``, so that invariant
    collapses onto ``q Q_r`` and the Makov-Payne pair of terms is COMPLETE at
    ``O(L^-3)`` with no separate dipole term.  Measured on a deliberately
    non-centrosymmetric defect (``|d| = 0.49 e A``): about an arbitrary origin
    the corrected error still falls only as ``L^-3.09`` (5.3e-4 eV at L = 24 A),
    about the centroid it falls as ``L^-5.00`` (7.5e-6 eV).  A caller who
    supplies their own ``centre`` is choosing the first of those.
    """

    charge: Tensor
    centre: Tensor
    dipole: Tensor
    second_moment: Tensor
    charge_radius: Tensor
    radius_ratio: Tensor


def defect_moments(
    defect_charges: Tensor,
    pos: Tensor,
    cell: Tensor,
    *,
    centre: Tensor | None = None,
) -> DefectMoments:
    """Net charge, dipole and second radial moment of a DEFECT charge density.

    ``defect_charges`` is **not** the cell's charges.  It is the defect charge
    density ``rho_d = rho(defective cell) - rho(perfect cell)``, sampled on the
    atoms -- the difference that carries the net charge ``q`` and nothing else.
    Handing this function the raw output of the solve instead is the mistake
    that makes the whole correction meaningless, because in a perfect crystal
    every site already carries a charge and their second moment is a property
    of the lattice, not of the defect.  There is no default for this argument
    for that reason.

    Wrapped coordinates are handled: positions are recentred on the largest
    ``|q_i|`` by the minimum image, the centroid is taken there, and they are
    recentred again on it.  Both shifts are integer lattice vectors taken from
    detached positions, so they change no gradient -- and without them a defect
    that straddles a cell face would report ``Q_r ~ L^2`` and look completely
    delocalised when it is not.

    Args:
        defect_charges: ``[n_atoms]`` defect charge density in e.
        pos: ``[n_atoms, 3]`` positions in A.
        cell: ``[3, 3]`` lattice vectors as rows, used for the minimum image and
            for the volume that ``radius_ratio`` is measured against.
        centre: ``[3]`` defect centre in A, or None for the charge centroid.
            None is strongly preferred: see :class:`DefectMoments`.

    Returns:
        :class:`DefectMoments`.
    """
    charge = defect_charges.sum()

    lattice = cell.detach().to(dtype=pos.dtype, device=pos.device)
    inverse = torch.linalg.inv(lattice)

    def minimum_image_about(reference: Tensor, positions: Tensor) -> Tensor:
        shift = torch.round((positions.detach() - reference) @ inverse)
        return positions - shift @ lattice

    anchor = pos.detach()[int(defect_charges.detach().abs().argmax())]
    unwrapped = minimum_image_about(anchor, pos)

    if centre is None:
        # A centroid needs a non-zero charge to divide by.  A neutral "defect"
        # has no centroid and no Makov-Payne correction either; the anchor keeps
        # every downstream number finite and the correction is zero regardless.
        safe_charge = torch.where(
            charge.abs() > 0.0, charge, torch.ones_like(charge)
        )
        raw = (defect_charges.unsqueeze(-1) * unwrapped).sum(dim=0) / safe_charge
        located = torch.where(charge.abs() > 0.0, raw, anchor)
        centre = located.detach()
    else:
        centre = centre.detach().to(dtype=pos.dtype, device=pos.device)

    relative = minimum_image_about(centre, unwrapped) - centre
    dipole = (defect_charges.unsqueeze(-1) * relative).sum(dim=0)
    second_moment = (defect_charges * (relative * relative).sum(dim=-1)).sum()

    safe_charge = torch.where(charge.abs() > 0.0, charge, torch.ones_like(charge))
    radius_squared = torch.where(
        charge.abs() > 0.0,
        (second_moment / safe_charge).abs(),
        torch.zeros_like(second_moment),
    )
    charge_radius = torch.sqrt(radius_squared)
    length = torch.det(cell).abs() ** (1.0 / 3.0)
    return DefectMoments(
        charge=charge,
        centre=centre,
        dipole=dipole,
        second_moment=second_moment,
        charge_radius=charge_radius,
        radius_ratio=charge_radius / length,
    )


@dataclass(frozen=True)
class MakovPayneParams:
    """The two hyperparameters of the Makov-Payne correction.

    Attributes:
        madelung: ``xi``, the Madelung constant of the lattice of point charges
            in a neutralising background, in the ``L = V^(1/3)`` convention.
            :func:`madelung_constant` computes it from the cell; do that rather
            than reaching for :data:`MADELUNG_SIMPLE_CUBIC` unless the cell
            really is cubic.  No default, because there is no shape-independent
            one -- it ranges from 2.9016 down through zero on the cells in the
            :func:`madelung_constant` table.
        dielectric: ``epsilon``, the STATIC DIELECTRIC CONSTANT OF THE HOST
            MATERIAL.  Both terms scale as ``1 / epsilon``, so this factor sets
            the whole size of the correction, and **supplying the wrong one is
            the usual way this goes wrong** -- see :func:`makov_payne_terms`.

    Plain Python numbers, never tensors, for the reason :class:`EwaldParams`
    gives: they are hyperparameters and nothing differentiates through them.
    """

    madelung: float
    dielectric: float = 1.0

    def __post_init__(self) -> None:
        madelung = float(self.madelung)
        if not math.isfinite(madelung):
            raise ValueError(
                f"Makov-Payne madelung constant must be finite (got {self.madelung!r}). "
                "Use ewald.madelung_constant(cell) to compute it for this cell."
            )
        if madelung > MADELUNG_UNIFORM_SPHERE * (1.0 + 1.0e-9):
            raise ValueError(
                f"Makov-Payne madelung constant {madelung} exceeds the shape "
                f"bound {MADELUNG_UNIFORM_SPHERE:.6f} = (9/5)(4 pi/3)^(1/3), the "
                "value for a spherical neutralising background, which no lattice "
                "can beat. The usual cause is the CONVENTIONAL-CUBE convention: "
                "the tabulated 3.639231 (bcc) and 4.584862 (fcc) are xi * 2^(1/3) "
                "and xi * 4^(1/3), and this module's convention is L = V^(1/3). "
                "Use ewald.madelung_constant(cell)."
            )
        object.__setattr__(self, "madelung", madelung)

        dielectric = float(self.dielectric)
        if not dielectric >= 1.0:
            raise ValueError(
                f"Makov-Payne dielectric constant must be at least 1 (got "
                f"{self.dielectric!r}). It is the static dielectric constant of "
                "the HOST, which screens the defect from its images; 1 is vacuum "
                "and nothing screens less than vacuum. A value below 1 would make "
                "the correction larger than the error it removes."
            )
        object.__setattr__(self, "dielectric", dielectric)

    @classmethod
    def for_cell(
        cls,
        cell: Tensor,
        dielectric: float = 1.0,
        *,
        r_cut: Optional[float] = None,
        accuracy: float = 1.0e-14,
    ) -> "MakovPayneParams":
        """Take ``madelung`` from the cell itself, via :func:`madelung_constant`.

        The counterpart of :meth:`EwaldParams.auto_tune`, and the way to get
        ``xi`` right on a cell that is not cubic.  Re-derive it if the cell
        changes SHAPE; an isotropic rescaling leaves ``xi`` exactly unchanged.
        """
        return cls(
            madelung=madelung_constant(cell, r_cut=r_cut, accuracy=accuracy),
            dielectric=dielectric,
        )


class MakovPayneTerms(NamedTuple):
    """The two Makov-Payne terms, in eV.

    Both carry an explicit factor of the net charge, so both are identically
    zero -- not small, zero -- for a neutral cell.

    A ``NamedTuple`` with ``total`` spelled out rather than delegating to
    another property, for the Dynamo reason recorded on :class:`EwaldTerms` and
    :class:`EwaldMultipoleTerms`.
    """

    monopole: Tensor
    quadrupole: Tensor

    @property
    def total(self) -> Tensor:
        return self.monopole + self.quadrupole


def makov_payne_terms(
    charge: Tensor,
    cell: Tensor,
    params: MakovPayneParams,
    *,
    second_moment: Tensor | None = None,
) -> MakovPayneTerms:
    """Finite-size correction to a charged periodic cell -- the two terms apart.

        E_corr = K_E q^2 xi / (2 epsilon V^(1/3))
                 - 2 pi K_E q Q_r / (3 epsilon V)      + O(L^-5)

    ADD this to the Ewald energy of a charged cell to estimate the energy of the
    ISOLATED charged defect.  It is opt-in, structurally: no function in this
    module calls it, so every existing result is untouched unless a caller asks
    for it by name.

    Why a charged cell needs it
    ---------------------------
    :func:`ewald_jellium` already removes the divergence of a net-charged
    lattice by adding a neutralising background, and the total is then finite
    and alpha-independent.  It is NOT the energy of an isolated charged defect.
    What is left is the interaction of the defect with its own periodic images
    and with that background, and it decays only as ``1/L``.  Measured here on a
    ``+1 e`` defect in a cubic cell, against the exactly-known isolated energy:
    2.61 eV at L = 8 A, 1.72 eV at 12 A, 0.85 eV at 24 A.  At ``-2 e`` it is
    10.1 eV at 8 A.  Even divided by a semiconductor's ``epsilon ~ 12`` that is
    0.14 eV at a 12 A cell -- the several-tenths-of-an-eV scale that makes
    charged point defects unusable uncorrected.

    Derivation -- the monopole term
    -------------------------------
    Under tinfoil boundaries with ``k = 0`` dropped, the periodic Green's
    function is ``G(k) = 4 pi / (V k^2)`` for ``k != 0`` and zero at the origin,
    so in real space ``lap G = -4 pi [delta(r) - 1/V]``.  Subtracting the free
    solution, ``G(r) = 1/r + phi(r)`` with ``lap phi = +4 pi / V``, whose
    regular solution about the origin is

        phi(r) = phi(0) + (2 pi / 3 V) r^2 + O(r^4)

    (``lap r^2 = 6``, and ``6 * 2 pi / 3 V = 4 pi / V``).  For a point charge,
    ``E = (1/2) q^2 phi(0)``, and that is by definition the Madelung energy of a
    charge in jellium, ``-K_E q^2 xi / (2 L)``.  So ``phi(0) = -xi / L``: the
    Madelung constant IS the self-potential the periodic boundary conditions
    manufacture, which is why :func:`madelung_constant` can read it straight off
    this module's own lattice sum.  The leading correction is minus that term,
    ``+K_E q^2 xi / (2 L)``, screened by the host's ``epsilon``.

    Derivation -- the ``Q_r`` term, and its SIGN
    --------------------------------------------
    For a density rather than a point charge, ``E = (1/2) int int rho(r)
    G(r-r') rho(r')``.  Expanding ``phi`` inside,

        E = E_isolated + (1/2) q^2 phi(0)
            + (1/2)(2 pi / 3 V) int int rho rho' |r - r'|^2
        int int rho rho' |r - r'|^2 = 2 q Q_r - 2 |d|^2

    with ``Q_r = int r^2 rho(r)`` and ``d = int r rho(r)``.  So

        E_periodic = E_isolated - K_E q^2 xi / (2 L)
                     + (2 pi K_E / 3 V) ( q Q_r - |d|^2 )

    and the correction is minus the second line.  **Note the sign of the ``Q_r``
    term: it is NEGATIVE**, where the form usually quoted from Makov and Payne
    is ``+2 pi q Q / (3 V)``.  That is a convention on ``Q``, not a disagreement
    about physics -- the DFT literature routinely calls the electron *number*
    density ``rho``, which is minus the charge density.  This module's ``Q_r``
    is unambiguously the second moment of the CHARGE density in e A^2, so the
    minus sign is the one that belongs to it, and the measurement settles it:
    with the sign written here the residual falls as ``L^-5.00``; with the sign
    flipped it stays at ``L^-3`` and the error at L = 8 A DOUBLES, from 1.4e-3 eV
    to 1.2e-1 eV.

    ``|d|^2`` does not appear above because :func:`defect_moments` centres the
    moments on the charge centroid, where ``d = 0``.  The combination
    ``q Q_r - |d|^2`` is centre-invariant while each piece is not, so that
    choice is what makes these two terms complete rather than merely
    conventional; :class:`DefectMoments` has the numbers.

    What it does NOT do
    -------------------
    * **It assumes the defect charge is LOCALISED** well inside the cell.  The
      expansion parameter is ``(r_q / L)^2``; at ``r_q / L = 0.5`` the charge is
      spread uniformly over the cell and there is no expansion left.
      :func:`check_defect_localisation` measures it and raises.  This module's
      own charge model lands on that limit for a homogeneous periodic crystal --
      see the module docstring, where it is measured -- so the correction is for
      a defect density the caller constructs, not for the cell's own charges.
    * **It assumes an ISOTROPIC dielectric response**, a single scalar
      ``epsilon``.  For an anisotropic host the image interaction is screened
      differently along different directions and no scalar is right.
    * **Freysoldt-Neugebauer-Van de Walle (FNV) supersedes it** when the charge
      is not localised, or when the defect sits at a surface or interface.  FNV
      aligns the electrostatic potential far from the defect against a reference
      calculation instead of assuming a multipole expansion, so it needs no
      ``Q_r`` and no assumption about the shape of the density.  Makov-Payne is
      the cheap, closed-form case; it is not the general one.
    * It is a correction to an ENERGY, not to a charge distribution.  It does
      not change the charges the solve returns, and it must not: ``q`` is fixed
      by the per-fragment constraints (SPEC 1.3), so this term adds nothing to
      the gradient that drives them -- exactly as :func:`ewald_jellium` does not.

    Gradients and the stress
    ------------------------
    Both terms go through ``V = |det cell|``, so both contribute to the stress;
    keep ``cell`` in the graph and it follows from the same autograd call the
    rest of the module uses.  ``xi`` is held fixed, which is exact for any
    ISOTROPIC strain (``xi`` is scale free) and, at a cubic cell, exact to first
    order in a deviatoric one as well: ``xi`` is stationary there, measured
    2.837297 at zero strain against 2.836426 and 2.836440 at uniaxial strain of
    -2 % and +2 %, i.e. ``d xi / d epsilon = 3e-4`` -- quadratic, not linear.
    Re-derive ``xi`` with :meth:`MakovPayneParams.for_cell` if the cell is
    driven far from its reference shape.

    Args:
        charge: ``q``, net defect charge in e (scalar tensor).  Take it from
            :attr:`DefectMoments.charge`.
        cell: ``[3, 3]`` lattice vectors as rows; the volume is
            ``|det cell|``, differentiably.
        params: ``xi`` and ``epsilon``.
        second_moment: ``Q_r`` in e A^2, from :attr:`DefectMoments.second_moment`,
            or None for the monopole term alone.  None is a defensible choice --
            it is the ``L^-3``-accurate correction, it needs no defect density at
            all, and it removes the entire ``1/L`` error -- but say so, because
            it converges as ``L^-3.02`` where the pair converges as ``L^-5.00``.

    Returns:
        :class:`MakovPayneTerms`, both zero for a neutral cell.
    """
    volume = torch.det(cell).abs()
    length = volume ** (1.0 / 3.0)

    monopole = (K_E * params.madelung / (2.0 * params.dielectric)) * (
        charge * charge
    ) / length

    if second_moment is None:
        quadrupole = torch.zeros_like(monopole)
    else:
        quadrupole = -(TWO_PI * K_E / (3.0 * params.dielectric)) * (
            charge * second_moment
        ) / volume

    return MakovPayneTerms(monopole=monopole, quadrupole=quadrupole)


def makov_payne_correction(
    charge: Tensor,
    cell: Tensor,
    params: MakovPayneParams,
    *,
    second_moment: Tensor | None = None,
) -> Tensor:
    """:func:`makov_payne_terms`, summed.  Add it to the energy of a charged cell.

    Identically zero for a neutral cell -- both terms carry a factor of
    ``charge``, so this returns exactly ``0.0`` and exactly zero gradient with
    respect to the cell, and cannot perturb a system it should not touch.
    """
    return makov_payne_terms(
        charge, cell, params, second_moment=second_moment
    ).total


def check_defect_localisation(
    moments: DefectMoments,
    *,
    max_radius_ratio: float = DEFAULT_MAX_RADIUS_RATIO,
) -> None:
    """Raise when the defect charge is too delocalised for Makov-Payne.

    The runtime counterpart of :func:`check_slab_geometry`, and there for the
    same reason: the assumption that fails is invisible in the answer.  A
    Makov-Payne correction applied to a delocalised charge returns a plausible
    number of the right sign and magnitude that converges to the wrong limit.

    The test is on :attr:`DefectMoments.radius_ratio`, ``r_q / V^(1/3)``, because
    the expansion is in ``(r_q / L)^2``: the ``Q_r`` term is
    ``(4 pi / 3 xi) (r_q / L)^2`` times the monopole term, 9.2 % at the default
    threshold of 0.25 and cubic ``xi``, and the neglected remainder smaller
    again.  At :data:`UNIFORM_CELL_RADIUS_RATIO` = 0.5 the charge is uniform over
    the cell, ``Q_r`` scales as ``L^2``, and the ``Q_r`` term scales as ``1/L``
    -- the same order as the monopole term it is a correction to.  There is
    nothing to salvage at that point, which is why this raises rather than warns.

    A neutral defect passes trivially: the correction is identically zero.

    Args:
        moments: from :func:`defect_moments`.
        max_radius_ratio: threshold on ``r_q / V^(1/3)``.

    Raises:
        ValueError: naming the measured ratio, the threshold, and the
            uniform-charge limit it is being compared against.
    """
    if float(moments.charge.detach().abs()) == 0.0:
        return
    ratio = float(moments.radius_ratio.detach())
    if ratio <= max_radius_ratio:
        return
    raise ValueError(
        f"the defect charge has RMS radius {float(moments.charge_radius.detach()):.3f} A "
        f"in a cell of V^(1/3) = "
        f"{float(moments.charge_radius.detach() / moments.radius_ratio.detach()):.3f} A, "
        f"a ratio of {ratio:.3f} against a limit of {max_radius_ratio}. "
        f"Makov-Payne expands in (r_q / L)^2 and is meaningless here: at a ratio "
        f"of {UNIFORM_CELL_RADIUS_RATIO} the charge is spread uniformly over the "
        "whole cell, Q_r grows as L^2, and the Q_r term falls as 1/L -- the same "
        "order as the monopole term it is supposed to correct. Either use a "
        "larger cell, or pass a genuinely localised defect density (rho of the "
        "defective cell minus rho of the perfect one, NOT the cell's own "
        "charges), or use the Freysoldt (FNV) potential-alignment scheme, which "
        "assumes no multipole expansion. Raise max_radius_ratio only with a "
        "convergence study in the cell size to show what it costs."
    )


# ---------------------------------------------------------------------------
# Boundary conditions -- SPEC section 7
# ---------------------------------------------------------------------------


class BoundaryRegime(Enum):
    """Which lattice sum a ``pbc`` triple selects.

    Three regimes and they are different physics, not a flag (SPEC 7).  The
    value is the word used in error messages.

    Classification is host side, on a plain Python triple: it decides which
    *function* runs, never which branch a tensor takes, so it happens once at
    assembly time and nothing about it reaches a traced graph.
    """

    OPEN = "open"
    BULK = "3-D periodic"
    SLAB = "2-D slab"


def _normalised_pbc(pbc: Sequence[bool]) -> Tuple[bool, bool, bool]:
    """Coerce a periodicity triple to three plain bools, or raise.

    Accepts anything with three elements -- an ASE ``atoms.pbc`` array, a list,
    a tuple, a 3-element tensor -- because the caller's source is ASE and the
    dtype there is numpy's.  The elements are taken through ``bool()``, so a
    0/1 array works and a float 0.5 becomes True rather than being quietly
    rounded somewhere later.
    """
    flags = tuple(bool(flag) for flag in pbc)
    if len(flags) != 3:
        raise ValueError(
            f"pbc must have exactly three entries, one per lattice vector, got "
            f"{len(flags)} ({pbc!r}). It is an INPUT (SPEC 7) and matches ASE's "
            "atoms.pbc; it is never inferred from the cell, because a gas-phase "
            "ASE frame carries cell = zeros(3, 3) and inferring 'periodic' from "
            "'a cell was supplied' routes a singular lattice into linalg.inv."
        )
    return flags


def boundary_regime(pbc: Sequence[bool]) -> BoundaryRegime:
    """Classify a periodicity triple into one of the three regimes.

    ======================  ==========================================
    ``(F, F, F)``           :attr:`BoundaryRegime.OPEN`
    exactly two True        :attr:`BoundaryRegime.SLAB`
    ``(T, T, T)``           :attr:`BoundaryRegime.BULK`
    exactly one True        raises -- a wire
    ======================  ==========================================

    The wire case raises rather than falling back on 3-D Ewald, which is the
    point of having this function at all: a 1-D periodic system has TWO
    non-periodic directions, so the shape-dependent term is a line charge's, not
    a plate's, and the slab correction of :func:`slab_correction` would be
    quietly wrong rather than absent.  Say so; do not silently do something
    else (SPEC 7).

    Args:
        pbc: Three flags, one per lattice vector, in ASE's convention.

    Returns:
        The regime.

    Raises:
        ValueError: for a wire (exactly one periodic direction), or a triple
            that is not three elements long.
    """
    flags = _normalised_pbc(pbc)
    count = sum(flags)
    if count == 3:
        return BoundaryRegime.BULK
    if count == 2:
        return BoundaryRegime.SLAB
    if count == 0:
        return BoundaryRegime.OPEN
    axis = flags.index(True)
    raise ValueError(
        f"pbc = {flags!r} is a WIRE: periodic along lattice vector {axis} only. "
        "trace_electrostatics.ewald does not implement 1-D periodic "
        "electrostatics. This raises rather than falling back on the 3-D sum on "
        "purpose: a wire has two non-periodic directions, so the conditionally "
        "convergent sum closes against a line-shaped boundary and neither the "
        "3-D tinfoil result nor the 2 pi M_n^2 / V slab correction applies -- "
        "using either would be wrong by a term that does not vanish with more "
        "vacuum. Supported: (False, False, False) open, two True for a slab, "
        "(True, True, True) for the bulk."
    )


def non_periodic_axis(pbc: Sequence[bool]) -> int:
    """Index of the non-periodic lattice vector of a slab.

    ``(T, T, F) -> 2``, ``(T, F, T) -> 1``, ``(F, T, T) -> 0``.  **z is not
    assumed.**  Everything downstream reaches the right direction through this
    index and through the normal :func:`slab_geometry` builds from it, so no
    permutation of the input is rotated, relabelled or copied: an axis index and
    a cross product cost nothing and cannot get out of step with the cell the
    way a rotated copy of the positions could.

    Raises:
        ValueError: if ``pbc`` is not a slab triple.
    """
    regime = boundary_regime(pbc)
    if regime is not BoundaryRegime.SLAB:
        raise ValueError(
            f"non_periodic_axis needs a 2-D slab pbc triple, got "
            f"{_normalised_pbc(pbc)!r} ({regime.value}), which has no single "
            "non-periodic axis."
        )
    return _normalised_pbc(pbc).index(False)


class SlabGeometry(NamedTuple):
    """The directions and measures the slab correction needs, from the cell.

    Attributes:
        axis: Index of the non-periodic lattice vector, a plain Python int --
            structure, not data, so it never enters a tensor operation.
        normal: ``[3]`` unit normal of the plane spanned by the two PERIODIC
            lattice vectors, from their cross product taken in cyclic order
            (``axis = 2`` gives ``a_0 x a_1``), so it points along ``+a_axis``
            for a right-handed cell.  Differentiable in ``cell``.
        area: Area of that plane's cell, ``|a_p x a_q|``, in A^2.
        spacing: Perpendicular distance to the next replica slab,
            ``volume / area``, in A.  This is ``|a_axis . normal|`` and NOT
            ``|a_axis|``: on a cell with a tilted third vector the two differ,
            and it is the perpendicular one the replica separation depends on.
        volume: ``|det cell|`` in A^3, the ``V`` of ``2 pi K_E M_n^2 / V``.

    ``normal`` is the direction the correction projects the cell dipole onto,
    and it is the normal of the PERIODIC PLANE rather than the direction of the
    non-periodic lattice vector.  For an orthogonal cell the two coincide and
    the distinction is invisible; for a cell whose third vector is tilted they
    do not, and it is the plane's normal that is right, because the macroscopic
    shape whose depolarization factor the correction carries is the plate --
    which is defined by the periodic directions.

    A ``NamedTuple``, for the ``torch.compile`` reason on :class:`EwaldTerms`.
    """

    axis: int
    normal: Tensor
    area: Tensor
    spacing: Tensor
    volume: Tensor


def slab_geometry(cell: Tensor, pbc: Sequence[bool] = (True, True, False)) -> SlabGeometry:
    """Normal, area, spacing and volume of a slab cell.

    Handles every permutation of the slab triple through
    :func:`non_periodic_axis`; nothing here assumes the third vector is the
    non-periodic one or that it points along z.

    Differentiable in ``cell`` -- which it must be, since the correction's
    ``1 / V`` and the projection onto ``normal`` both carry stress.

    Args:
        cell: ``[3, 3]`` lattice vectors as rows.
        pbc: Periodicity triple selecting the slab regime.

    Returns:
        A :class:`SlabGeometry`.

    Raises:
        ValueError: if ``pbc`` is not a slab triple.
    """
    axis = non_periodic_axis(pbc)
    first, second = (axis + 1) % 3, (axis + 2) % 3
    cross = torch.linalg.cross(cell[first], cell[second])
    area = cross.norm()
    volume = torch.det(cell).abs()
    return SlabGeometry(
        axis=axis,
        normal=cross / area,
        area=area,
        spacing=volume / area,
        volume=volume,
    )


def slab_dipole(
    charges: Tensor,
    dipoles: Optional[Tensor],
    pos: Tensor,
    normal: Tensor,
) -> Tensor:
    """Component of the cell dipole along the slab normal, in e*A.

        M_n = sum_i q_i (r_i . n) + sum_i mu_i . n

    Both terms, because a point dipole contributes to the cell dipole exactly as
    a pair of displaced charges would and the correction cannot tell them apart.
    Omitting the ``mu`` half would leave a slab of neutral polar molecules --
    water on a surface, the case this whole route exists for -- uncorrected.

    ORIGIN DEPENDENCE.  ``M_n`` shifts by ``(sum_i q_i) (d . n)`` under a rigid
    translation ``d``, so it is a property of the system only for a NEUTRAL
    cell.  That is why :func:`check_slab_geometry` refuses a charged one; see
    the module docstring.

    Args:
        charges: ``[n_atoms]`` charges in e.
        dipoles: ``[n_atoms, 3]`` dipoles in e*A, or None.
        pos: ``[n_atoms, 3]`` positions in A.
        normal: ``[3]`` unit normal, from :func:`slab_geometry`.

    Returns:
        Scalar tensor in e*A.
    """
    moment = torch.dot(charges, pos @ normal)
    if dipoles is not None:
        moment = moment + (dipoles @ normal).sum()
    return moment


def slab_correction(
    charges: Tensor,
    dipoles: Optional[Tensor],
    pos: Tensor,
    cell: Tensor,
    pbc: Sequence[bool] = (True, True, False),
) -> Tensor:
    """The Yeh-Berkowitz slab correction, in eV.

        U_slab = + 2 pi K_E M_n^2 / V

    added to the 3-D tinfoil Ewald energy to recover the energy of a single
    isolated slab.  ``M_n`` is :func:`slab_dipole`, ``V = |det cell|``.

    SIGN AND FACTOR, both of which the literature states inconsistently: it is
    ``2 pi``, not ``4 pi``, and it is ADDED.  The derivation is in the module
    docstring (a parallel-plate capacitor, where the isolated and the tinfoil
    energies are both closed forms and differ by exactly ``-2 pi M_z^2 / V``);
    the numerical confirmation is against a direct 2-D lattice sum.  Physically:
    dropping ``k = 0`` is a conducting boundary, the conductor screens the
    macroscopic polarization and LOWERS the energy, and an isolated slab does
    not have a conductor, so the energy must come back up.

    It vanishes identically -- ``0.0``, not ``1e-17`` -- for a non-polar slab,
    because ``M_n`` enters squared and nothing else in the expression can be
    non-zero at ``M_n = 0``.  So the correction cannot perturb a system it
    should not touch, however large the padding or the cell.

    Differentiable in ``charges``, ``dipoles``, ``pos`` and ``cell``: through
    ``M_n`` (hence a force ``-4 pi K_E M_n q_i n / V`` on every charge, the
    depolarizing field of the isolated slab), and through ``V`` and ``n``, hence
    stress.  It is the derivative of the same scalar the energy reports, so
    forces and stress follow from the flashace autograd block unchanged.

    Args:
        charges: ``[n_atoms]`` charges in e.
        dipoles: ``[n_atoms, 3]`` dipoles in e*A, or None.
        pos: ``[n_atoms, 3]`` positions in A.
        cell: ``[3, 3]`` lattice vectors as rows, INCLUDING the vacuum padding.
        pbc: Periodicity triple selecting the slab regime.

    Returns:
        Scalar energy in eV, always >= 0.
    """
    geometry = slab_geometry(cell, pbc)
    moment = slab_dipole(charges, dipoles, pos, geometry.normal)
    return (
        SLAB_CORRECTION_COEFFICIENT * math.pi * K_E * moment * moment / geometry.volume
    )


def slab_correction_blocks(
    pos: Tensor,
    cell: Tensor,
    pbc: Sequence[bool] = (True, True, False),
    ell: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """The slab correction as pair blocks, to be ADDED to the lattice-summed ones.

    Returns ``(j_slab, t01_slab, t2_slab)`` with the shapes and conventions of
    ``energy._periodic_pair_blocks`` -- ``[n, n]``, ``[n, n, 3]``,
    ``[n, n, 3, 3]`` -- so that

        0.5 q^T (J + J_slab) q + q^T (T01 + T01_slab) mu_hat
            + 0.5 mu_hat^T (T2 + T2_slab) mu_hat

    is the slab energy, operator and all.  This exists because SPEC 8 requires
    the reported response to be the derivative of the energy the model actually
    evaluates: adding the correction to the energy alone would leave the solve
    minimising the uncorrected operator, and the polarizability would then be
    the bulk one while the energy was the slab one.

    With ``C = 2 pi K_E / V`` and ``z_i = r_i . n``, expanding
    ``C (sum_i q_i z_i + sum_i mu_i . n)^2`` gives

        J_slab[i, j]        = 2 C z_i z_j
        T01_slab[i, j, a]   = 2 C z_i n_a
        T2_slab[i, j, a, b] = 2 C n_a n_b

    the factor 2 being the one that turns ``0.5 x^T K x`` back into the square.
    Every block is rank one in the site indices -- the correction couples every
    atom to every other through a single number, which is what a macroscopic
    depolarizing field IS -- so this is ``O(n^2)`` memory but carries no new
    physics per pair.

    Args:
        pos: ``[n_atoms, 3]`` positions in A.
        cell: ``[3, 3]`` lattice vectors as rows.
        pbc: Periodicity triple selecting the slab regime.
        ell: ``[n_atoms]`` dipole reduction lengths in A (SPEC 1.5,
            ``mu_hat = mu / (e ell)``), or None for blocks that act on physical
            dipoles.  Applied exactly as ``energy._direct_pair_blocks`` applies
            it: once on the dipole index of ``t01``, twice on ``t2``.

    Returns:
        ``(j_slab, t01_slab, t2_slab)``.
    """
    geometry = slab_geometry(cell, pbc)
    n_atoms = int(pos.shape[0])
    coefficient = (
        SLAB_CORRECTION_COEFFICIENT * math.pi * K_E / geometry.volume
    )
    projection = pos @ geometry.normal                             # [n], z_i
    normal = geometry.normal

    j_slab = (2.0 * coefficient) * projection.unsqueeze(1) * projection.unsqueeze(0)
    t01_slab = (2.0 * coefficient) * projection.view(n_atoms, 1, 1) * normal.view(
        1, 1, 3
    ).expand(n_atoms, n_atoms, 3)
    t2_slab = (2.0 * coefficient) * (
        normal.view(1, 1, 3, 1) * normal.view(1, 1, 1, 3)
    ).expand(n_atoms, n_atoms, 3, 3)

    if ell is not None:
        reduction = ell.reshape(-1)
        t01_slab = t01_slab * reduction.view(1, n_atoms, 1)
        t2_slab = t2_slab * (
            reduction.view(n_atoms, 1, 1, 1) * reduction.view(1, n_atoms, 1, 1)
        )
    return j_slab, t01_slab, t2_slab


# -- the runtime check on the padding ---------------------------------------


class SlabVacuumReport(NamedTuple):
    """What :func:`slab_vacuum_report` measured, all plain Python floats.

    Attributes:
        axis: Non-periodic lattice vector index.
        span: Extent of the POINT charge distribution along the normal, in A.
            The Gaussian clouds of SPEC 1.1 reach a few ``sigma`` further; a
            caller who wants that margin should compare ``gap`` against its own
            ``3 sigma_max`` as well.
        spacing: Perpendicular distance between replica slabs, in A.
        gap: ``spacing - span``, the vacuum between a slab and its image.
            Negative means the replicas overlap.
        lateral: Longest in-plane repeat, ``2 pi / |G|_min``, in A -- the length
            that actually sets the convergence, and usually NOT the same as the
            in-plane cell edge for a non-orthogonal or centred lattice.
        gap_ratio: ``gap / lateral``, the dimensionless quantity the exponential
            law is written in.
        required_gap: Gap needed for ``accuracy``, in A.
        estimated_error: ``SLAB_RESIDUAL_ENERGY_SCALE * exp(-|G|_min gap)`` in
            eV -- an order-of-magnitude estimate with a measured prefactor, not
            a bound.
        net_charge: ``sum_i q_i`` in e, or None if no charges were supplied.
        sufficient: Whether ``gap >= required_gap`` AND the cell is neutral.
    """

    axis: int
    span: float
    spacing: float
    gap: float
    lateral: float
    gap_ratio: float
    required_gap: float
    estimated_error: float
    net_charge: Optional[float]
    sufficient: bool


def required_vacuum_gap(
    lateral: float, accuracy: float = DEFAULT_EWALD_ACCURACY
) -> float:
    """Vacuum gap, in A, at which the replica interaction falls to ``accuracy``.

        gap >= lateral * ln(U_scale / accuracy) / (2 pi)

    from ``residual ~ U_scale exp(-|G|_min gap)`` with ``|G|_min = 2 pi / lateral``
    (module docstring, measured over nine decades).  ``U_scale`` is
    :data:`SLAB_RESIDUAL_ENERGY_SCALE`.

    In ratios of the LONGEST IN-PLANE REPEAT -- not of the slab thickness, and
    not as ``L_z / L_xy``:

        accuracy    1e-4    1e-6    1e-8    1e-10    1e-12
        gap/lateral  1.61    2.34    3.07     3.80     4.53

    which is the fact worth carrying away: a 30 A surface supercell wants ~114 A
    of vacuum for 1e-10 eV, while a 5 A one wants 19 A. Widening the supercell
    laterally makes the padding requirement WORSE, and it is the only thing that
    does.
    """
    if not 0.0 < accuracy < 1.0:
        raise ValueError(
            f"accuracy must lie strictly between 0 and 1 (got {accuracy}); it is "
            "the target size in eV of the neglected slab-replica interaction."
        )
    return lateral * math.log(SLAB_RESIDUAL_ENERGY_SCALE / accuracy) / TWO_PI


def _minimum_in_plane_reciprocal(cell: Tensor, geometry: SlabGeometry) -> float:
    """``|G|_min`` of the 2-D reciprocal lattice of the periodic plane, in 1/A.

    The 2-D reciprocal basis is NOT two rows of ``2 pi (cell^-1)^T``: those are
    perpendicular to the non-periodic vector, which for a tilted cell is not the
    same as lying IN the periodic plane.  Built from the plane's own vectors,

        g_p = 2 pi (a_q x n) / (a_p . (a_q x n)) ,   g_q = 2 pi (n x a_p) / (a_q . (n x a_p))

    which satisfy ``g . a = 2 pi delta`` and ``g . n = 0`` by construction, and
    then minimised over a small window of integer combinations, because the
    shortest vector of an oblique lattice need not be a basis vector (a 60-degree
    lattice's ``g_p + g_q`` is shorter than either).

    Host side: this reads the cell and returns a float, so it belongs where
    :func:`_image_shell_counts` belongs -- at tune or check time, never in the
    inner loop.
    """
    lattice = cell.detach().to(dtype=torch.float64)
    normal = geometry.normal.detach().to(dtype=torch.float64)
    first, second = (geometry.axis + 1) % 3, (geometry.axis + 2) % 3
    a_p, a_q = lattice[first], lattice[second]

    g_p = TWO_PI * torch.linalg.cross(a_q, normal) / torch.dot(
        a_p, torch.linalg.cross(a_q, normal)
    )
    g_q = TWO_PI * torch.linalg.cross(normal, a_p) / torch.dot(
        a_q, torch.linalg.cross(normal, a_p)
    )

    shortest = math.inf
    for m in range(-3, 4):
        for n in range(-3, 4):
            if m == 0 and n == 0:
                continue
            shortest = min(shortest, float((m * g_p + n * g_q).norm()))
    return shortest


def slab_vacuum_report(
    pos: Tensor,
    cell: Tensor,
    pbc: Sequence[bool] = (True, True, False),
    charges: Optional[Tensor] = None,
    accuracy: float = DEFAULT_EWALD_ACCURACY,
) -> SlabVacuumReport:
    """Measure the vacuum padding of a slab cell against the convergence condition.

    See :func:`required_vacuum_gap` for the condition and the module docstring
    for where it comes from.  This is the diagnostic; :func:`check_slab_geometry`
    is the same measurement with a raise on it.

    THIS SYNCHRONISES WITH THE HOST.  It reads positions and charges out of
    tensors to produce Python floats, so it belongs at setup time next to
    :meth:`EwaldParams.auto_tune`, and the energy routes take it on a flag
    (``validate=False`` inside a compiled region) for exactly that reason.

    Args:
        pos: ``[n_atoms, 3]`` positions in A.
        cell: ``[3, 3]`` lattice vectors as rows.
        pbc: Periodicity triple selecting the slab regime.
        charges: ``[n_atoms]`` charges in e, for the neutrality field.  Optional
            because the padding question is geometric and can be asked before
            any charge exists.
        accuracy: Target size in eV of the neglected replica interaction.

    Returns:
        A :class:`SlabVacuumReport`.
    """
    geometry = slab_geometry(cell, pbc)
    projection = pos.detach().to(dtype=torch.float64) @ geometry.normal.detach().to(
        dtype=torch.float64
    )
    span = float(projection.max() - projection.min())
    spacing = float(geometry.spacing.detach())
    gap = spacing - span

    g_min = _minimum_in_plane_reciprocal(cell, geometry)
    lateral = TWO_PI / g_min
    required = required_vacuum_gap(lateral, accuracy)
    estimated = SLAB_RESIDUAL_ENERGY_SCALE * math.exp(-g_min * gap)

    net_charge = None if charges is None else float(charges.detach().sum())
    neutral = net_charge is None or abs(net_charge) <= SLAB_NEUTRALITY_TOLERANCE

    return SlabVacuumReport(
        axis=geometry.axis,
        span=span,
        spacing=spacing,
        gap=gap,
        lateral=lateral,
        gap_ratio=gap / lateral,
        required_gap=required,
        estimated_error=estimated,
        net_charge=net_charge,
        sufficient=bool(gap >= required and neutral),
    )


def check_slab_geometry(
    pos: Tensor,
    cell: Tensor,
    pbc: Sequence[bool] = (True, True, False),
    charges: Optional[Tensor] = None,
    accuracy: float = DEFAULT_EWALD_ACCURACY,
) -> SlabVacuumReport:
    """Raise unless a slab cell is neutral and has enough vacuum.

    Two separate failures, with separate messages, because the fixes are
    different:

    1. **Charged cell.**  ``M_n`` is origin-dependent for ``sum_i q_i != 0``, so
       the correction is not merely inaccurate, it has no value: wrapping the
       coordinates changes the answer.  Not repairable by padding.
    2. **Too little vacuum.**  The replica interaction is above the requested
       accuracy.  Repairable, by padding -- the message says by how much.

    Both raise rather than warn.  A warning on an energy that is wrong by
    0.1 eV, in a package whose gradient tests are quoted at 1e-10, would be
    swallowed by any training loop that has ever seen a deprecation notice.

    Returns the report on success, so a caller can log the measured numbers.

    Raises:
        ValueError: on either failure, quoting the measured numbers.
    """
    report = slab_vacuum_report(pos, cell, pbc, charges, accuracy)

    if report.net_charge is not None and abs(report.net_charge) > SLAB_NEUTRALITY_TOLERANCE:
        raise ValueError(
            f"the slab correction needs a NEUTRAL cell; this one carries "
            f"sum_i q_i = {report.net_charge:.6e} e (tolerance "
            f"{SLAB_NEUTRALITY_TOLERANCE:.1e}). M_n = sum_i q_i (r_i . n) shifts "
            f"by Q (d . n) under a rigid translation d, so with Q != 0 the "
            "correction 2 pi K_E M_n^2 / V is not a property of the system at "
            "all -- re-wrapping the coordinates would change the energy. More "
            "vacuum does not help. The charged-slab generalisation (Ballenegger, "
            "Arnold and Cerda) adds -Q sum_i q_i z_i^2 - Q^2 L^2 / 12 inside the "
            "bracket but fixes a 2-D neutralising-background convention this "
            "package does not define, and is deliberately out of scope. Neutralise "
            "the cell, or use pbc = (True, True, True) and accept the 3-D physics."
        )

    if report.gap < report.required_gap:
        raise ValueError(
            f"not enough vacuum along lattice vector {report.axis} for the slab "
            f"correction to reach accuracy {accuracy:.1e} eV: the gap between the "
            f"slab and its image is {report.gap:.3f} A "
            f"({report.gap_ratio:.3f} x the longest in-plane repeat of "
            f"{report.lateral:.3f} A) and {report.required_gap:.3f} A "
            f"({report.required_gap / report.lateral:.3f} x) is needed. Estimated "
            f"residual replica interaction {report.estimated_error:.3e} eV. "
            "The residual decays as exp(-2 pi gap / lateral), so the requirement "
            "is set by the LATERAL cell size, not by the slab thickness: widening "
            "the surface supercell makes it worse. Extend lattice vector "
            f"{report.axis} by {report.required_gap - report.gap:.3f} A, or pass a "
            "looser accuracy if the estimate above is acceptable."
        )
    return report


# -- the slab energy: 3-D Ewald plus the correction --------------------------


class EwaldSlabTerms(NamedTuple):
    """The four 3-D point-charge contributions plus the slab correction, in eV.

    ``real``, ``recip``, ``self_energy`` and ``jellium`` are exactly
    :class:`EwaldTerms`; ``slab`` is :func:`slab_correction`, which contains no
    ``alpha`` at all.  So the alpha-independence check still bites: the four
    3-D terms move by many eV over a sweep, ``slab`` does not move at all, and
    ``total`` does not move.

    A ``NamedTuple``, for the ``torch.compile`` reason on :class:`EwaldTerms`,
    and flat rather than wrapping an :class:`EwaldTerms` for the same reason --
    Dynamo cannot read one NamedTuple property from inside another.
    """

    real: Tensor
    recip: Tensor
    self_energy: Tensor
    jellium: Tensor
    slab: Tensor

    @property
    def bulk_3d(self) -> Tensor:
        """What plain 3-D Ewald would have returned: the WRONG answer for a slab."""
        return self.real + self.recip + self.self_energy + self.jellium

    @property
    def total(self) -> Tensor:
        return self.real + self.recip + self.self_energy + self.jellium + self.slab


class EwaldMultipoleSlabTerms(NamedTuple):
    """:class:`EwaldMultipoleTerms` plus the slab correction, in eV.

    ``damping`` is the model's short-ranged Gaussian/Thole correction and
    ``slab`` the Yeh-Berkowitz term; neither contains ``alpha``.
    """

    real: Tensor
    recip: Tensor
    self_energy: Tensor
    jellium: Tensor
    damping: Tensor
    slab: Tensor

    @property
    def bulk_3d(self) -> Tensor:
        """What plain 3-D Ewald would have returned: the WRONG answer for a slab."""
        return self.real + self.recip + self.self_energy + self.jellium + self.damping

    @property
    def total(self) -> Tensor:
        # Spelled out, not `self.bulk_3d + self.slab`: Dynamo cannot read one
        # property from inside another (see EwaldMultipoleTerms.total).
        return (
            self.real
            + self.recip
            + self.self_energy
            + self.jellium
            + self.damping
            + self.slab
        )


def ewald_slab_terms(
    q: Tensor,
    pos: Tensor,
    cell: Tensor,
    edge_index: Tensor,
    edge_shift: Tensor | None,
    params: EwaldParams,
    pbc: Sequence[bool] = (True, True, False),
    *,
    full_neighbor_list: bool = True,
    validate: bool = True,
) -> EwaldSlabTerms:
    """Point-charge slab energy: the 3-D terms and the correction, separately.

    The neighbour-list route of :func:`ewald_terms` with
    :func:`slab_correction` alongside.  :func:`ewald_slab_energy` sums them.

    Args:
        q, pos, cell, edge_index, edge_shift, params, full_neighbor_list: as
            :func:`ewald_terms`.  ``cell`` must include the vacuum padding and
            the neighbour list must be built from the same padded cell.
        pbc: Periodicity triple selecting the slab regime.
        validate: Run :func:`check_slab_geometry` first.  **True costs a
            host synchronisation** -- it reads ``pos`` and ``q`` to floats -- so
            set it False inside a ``torch.compile(fullgraph=True)`` region or an
            AOT export, having checked once at setup.  The check is the padding
            and neutrality contract, not a NaN shield; turning it off warrants
            it rather than skipping it.

    Returns:
        An :class:`EwaldSlabTerms`.
    """
    if validate:
        check_slab_geometry(pos, cell, pbc, q, params.accuracy)
    bulk = ewald_terms(
        q, pos, cell, edge_index, edge_shift, params, full_neighbor_list=full_neighbor_list
    )
    return EwaldSlabTerms(
        real=bulk.real,
        recip=bulk.recip,
        self_energy=bulk.self_energy,
        jellium=bulk.jellium,
        slab=slab_correction(q, None, pos, cell, pbc),
    )


def ewald_slab_energy(
    q: Tensor,
    pos: Tensor,
    cell: Tensor,
    edge_index: Tensor,
    edge_shift: Tensor | None,
    params: EwaldParams,
    pbc: Sequence[bool] = (True, True, False),
    *,
    full_neighbor_list: bool = True,
    validate: bool = True,
) -> Tensor:
    """Electrostatic energy of a 2-D periodic slab of point charges, in eV.

        U = U_real + U_recip + U_self + U_jellium + 2 pi K_E M_n^2 / V

    i.e. 3-D Ewald in the padded cell, plus the Yeh-Berkowitz correction that
    removes the interaction of the replica slabs through their dipoles.  See
    :func:`slab_correction` for the sign and the factor, and the module
    docstring for the padding requirement -- which :func:`check_slab_geometry`
    enforces unless ``validate=False``.

    Differentiable with respect to ``q``, ``pos`` and ``cell``, and independent
    of ``params.alpha`` to the truncation accuracy, exactly as the 3-D route is:
    the correction contains no alpha.
    """
    return ewald_slab_terms(
        q,
        pos,
        cell,
        edge_index,
        edge_shift,
        params,
        pbc,
        full_neighbor_list=full_neighbor_list,
        validate=validate,
    ).total


def ewald_multipole_slab_terms(
    charges: Tensor,
    dipoles: Tensor | None,
    pos: Tensor,
    cell: Tensor,
    sigma: Tensor,
    alpha_volume: Tensor | None,
    params: EwaldParams,
    pbc: Sequence[bool] = (True, True, False),
    *,
    thole_a: float = DEFAULT_THOLE_A,
    validate: bool = True,
) -> EwaldMultipoleSlabTerms:
    """:func:`ewald_multipole_terms` with the slab correction alongside.

    The correction sees the dipoles as well as the charges
    (:func:`slab_dipole`), which is the difference that matters for a slab of
    neutral polar molecules: its charges may contribute little to ``M_n`` while
    its induced dipoles contribute all of it.

    ``validate`` is :func:`ewald_slab_terms`' flag, with the same host-sync
    warning.
    """
    if validate:
        check_slab_geometry(pos, cell, pbc, charges, params.accuracy)
    bulk = ewald_multipole_terms(
        charges, dipoles, pos, cell, sigma, alpha_volume, params, thole_a=thole_a
    )
    return EwaldMultipoleSlabTerms(
        real=bulk.real,
        recip=bulk.recip,
        self_energy=bulk.self_energy,
        jellium=bulk.jellium,
        damping=bulk.damping,
        slab=slab_correction(charges, dipoles, pos, cell, pbc),
    )


def ewald_multipole_slab_energy(
    charges: Tensor,
    dipoles: Tensor | None,
    pos: Tensor,
    cell: Tensor,
    sigma: Tensor,
    alpha_volume: Tensor | None,
    params: EwaldParams,
    pbc: Sequence[bool] = (True, True, False),
    *,
    thole_a: float = DEFAULT_THOLE_A,
    validate: bool = True,
) -> Tensor:
    """Slab energy of Gaussian-smeared charges and Thole-damped dipoles, in eV.

        U = U_ewald_3D(padded cell) + 2 pi K_E M_n^2 / V

    :func:`ewald_multipole_energy` plus :func:`slab_correction`.  Use
    :meth:`EwaldParams.auto_tune_slab` for the parameters: a padded cell wants a
    per-direction reciprocal grid, and the isotropic one is unusable at real
    paddings.
    """
    return ewald_multipole_slab_terms(
        charges,
        dipoles,
        pos,
        cell,
        sigma,
        alpha_volume,
        params,
        pbc,
        thole_a=thole_a,
        validate=validate,
    ).total


# -- the open boundary, and the dispatch -------------------------------------


def direct_multipole_energy(
    charges: Tensor,
    dipoles: Tensor | None,
    pos: Tensor,
    sigma: Tensor,
    alpha_volume: Tensor | None,
    *,
    thole_a: float = DEFAULT_THOLE_A,
) -> Tensor:
    """Open-boundary energy of the same three damped blocks, in eV.

        U = 0.5 sum_{i != j} q_i J(r_ij) q_j
            + sum_{i != j} q_i T01(r_i - r_j) . mu_j
            + 0.5 sum_{i != j} mu_i . T2(r_i - r_j) . mu_j

    The ``pbc = (False, False, False)`` branch of :func:`energy_for_pbc`.  **It
    takes no cell**, which is the SPEC 7 requirement "a supplied cell is
    IGNORED, not used" made structural rather than promised: there is no
    argument for the dispatcher to pass, so no path exists by which a gas-phase
    ``cell = zeros(3, 3)`` could reach ``linalg.inv``.

    The on-site ``i == j`` terms are excluded, as everywhere in this module:
    SPEC 1.2 and 1.5 carry them in ``eta_i`` and ``A_mu_i``.

    Relation to ``energy.ElectrostaticAssembly._direct_pair_energy``.  That is
    the same sum, contracted out of stored blocks because the solver needs the
    operator and not only the scalar.  The two are separate expressions of one
    physics, so the kernels are shared rather than restated -- ``coulomb_kernel``,
    ``kernel_self``, ``charge_dipole_tensor``, ``dipole_interaction_tensor`` and
    ``thole_damping`` are imported, and only the contraction is written twice.
    Measured agreement on 6 atoms with random charges and dipoles: **0.0 eV,
    exactly**, on an energy of order 10 eV.

    Args:
        charges: ``[n_atoms]`` charges in e.
        dipoles: ``[n_atoms, 3]`` dipoles in e*A (physical, not reduced), or
            None.
        pos: ``[n_atoms, 3]`` positions in A.
        sigma: ``[n_atoms]`` Gaussian charge widths in A.
        alpha_volume: ``[n_atoms]`` polarizability volumes in A^3, required
            whenever ``dipoles`` is given.
        thole_a: Thole screening parameter, ``0 < a < 1``.

    Returns:
        Scalar energy in eV.
    """
    n_atoms = int(pos.shape[0])
    r_vec = pos.unsqueeze(1) - pos.unsqueeze(0)                   # [n, n, 3]
    squared = (r_vec * r_vec).sum(dim=-1)

    # Substitute the input of the square root, not its output; see
    # _pair_geometry for why masking afterwards is not enough.
    separated = squared > DEGENERATE_SEPARATION * DEGENERATE_SEPARATION
    r = torch.sqrt(torch.where(separated, squared, torch.ones_like(squared)))
    off_diagonal = 1.0 - torch.eye(n_atoms, dtype=pos.dtype, device=pos.device)
    keep = separated.to(pos.dtype) * off_diagonal

    sigma_pair = combine_sigma(sigma.unsqueeze(1), sigma.unsqueeze(0))
    # Two coincident but DISTINCT smeared charges have a finite, well defined
    # interaction and it is the r -> 0 limit of the kernel, not zero; the dipole
    # tensors have no such limit and stay masked.  This is
    # energy._direct_pair_blocks' rule, restated so the two cannot diverge.
    j_pair = (
        torch.where(separated, coulomb_kernel(r, sigma_pair), kernel_self(sigma_pair))
        * off_diagonal
    )
    energy = 0.5 * torch.dot(charges, j_pair @ charges)

    if dipoles is None:
        return energy

    if alpha_volume is None:
        raise ValueError(
            "direct_multipole_energy needs alpha_volume (polarizability volumes "
            "in A^3) whenever dipoles are supplied: it sets the Thole reduced "
            "distance, and there is no sensible default for it."
        )
    _require_matching_dtype(dipoles, "dipoles", pos, "pos")

    lam3, lam5 = thole_damping(
        r, alpha_volume.unsqueeze(1), alpha_volume.unsqueeze(0), thole_a
    )
    t01 = charge_dipole_tensor(r_vec, r, lam3) * keep.unsqueeze(-1)
    t2 = dipole_interaction_tensor(r_vec, r, lam3, lam5) * keep.unsqueeze(-1).unsqueeze(-1)

    energy = energy + torch.einsum("i,ija,ja->", charges, t01, dipoles)
    return energy + 0.5 * torch.einsum("ia,ijab,jb->", dipoles, t2, dipoles)


def energy_for_pbc(
    charges: Tensor,
    dipoles: Tensor | None,
    pos: Tensor,
    cell: Tensor | None,
    sigma: Tensor,
    alpha_volume: Tensor | None,
    params: Optional[EwaldParams],
    pbc: Sequence[bool],
    *,
    thole_a: float = DEFAULT_THOLE_A,
    validate: bool = True,
) -> Tensor:
    """Electrostatic energy under the boundary conditions ``pbc`` says, in eV.

    The single place a periodicity triple is turned into a choice of physics
    (SPEC 7).  One function so that the four regimes cannot be spelled
    differently in four call sites:

    ======================  =============================================
    ``(F, F, F)``           :func:`direct_multipole_energy`.  ``cell`` is
                            not read, not validated, and not passed on --
                            it may be ``None``, ``zeros(3, 3)``, or a
                            perfectly good lattice; all three behave the
                            same, which is the SPEC 7 requirement.
                            ``params`` may be None for the same reason.
    ``(T, T, T)``           :func:`ewald_multipole_energy`, unchanged.
    two True                :func:`ewald_multipole_slab_energy`.
    one True                raises (:func:`boundary_regime`) -- a wire.
    ======================  =============================================

    The dispatch is a Python branch on a Python tuple, so it is resolved at
    trace time and costs nothing in a compiled graph; what it must not become is
    a branch on a tensor.

    Args:
        charges: ``[n_atoms]`` charges in e.
        dipoles: ``[n_atoms, 3]`` dipoles in e*A, or None.
        pos: ``[n_atoms, 3]`` positions in A.
        cell: ``[3, 3]`` lattice vectors as rows; required by the periodic
            regimes, ignored entirely by the open one.
        sigma: ``[n_atoms]`` Gaussian charge widths in A.
        alpha_volume: ``[n_atoms]`` polarizability volumes in A^3, required
            whenever ``dipoles`` is given.
        params: Splitting parameters; required by the periodic regimes.  Use
            :meth:`EwaldParams.auto_tune` for the bulk and
            :meth:`EwaldParams.auto_tune_slab` for a slab.
        pbc: Three flags, one per lattice vector, ASE's convention.
        thole_a: Thole screening parameter, ``0 < a < 1``.
        validate: Passed to the slab route; see :func:`ewald_slab_terms`.

    Raises:
        ValueError: for a wire, for a missing ``cell`` or ``params`` under a
            periodic regime, and from :func:`check_slab_geometry`.
    """
    regime = boundary_regime(pbc)

    if regime is BoundaryRegime.OPEN:
        return direct_multipole_energy(
            charges, dipoles, pos, sigma, alpha_volume, thole_a=thole_a
        )

    if cell is None or params is None:
        missing = "cell" if cell is None else "params"
        raise ValueError(
            f"pbc = {_normalised_pbc(pbc)!r} selects the {regime.value} regime, "
            f"which needs a {missing}. Only the open boundary "
            "(False, False, False) runs without one."
        )

    if regime is BoundaryRegime.SLAB:
        return ewald_multipole_slab_energy(
            charges,
            dipoles,
            pos,
            cell,
            sigma,
            alpha_volume,
            params,
            pbc,
            thole_a=thole_a,
            validate=validate,
        )

    return ewald_multipole_energy(
        charges, dipoles, pos, cell, sigma, alpha_volume, params, thole_a=thole_a
    )
