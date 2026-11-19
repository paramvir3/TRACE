"""Equivariant readout heads mapping TRACE node features to electrostatic parameters.

The charge-equilibration model in this package is fully specified by six learned
fields.  This module is the only place they are produced, so every physical
constraint they must satisfy is enforced here, by construction, rather than being
checked downstream and raised as a runtime error::

    chi_i    0e          electronegativity                    unconstrained
    eta_i    0e          on-site hardness                     >= J(0; sigma_i)
    sigma_i  0e          Gaussian charge width                bounded, ~1-2 A
    ell_i    0e          moment length scale                  > 0
    A_mu_i   0e + 2e     dipole stiffness (3x3)               positive definite
    kappa_ij edge        bond hardness                        symmetric, > 0

Three optional sectors extend that set, and **all are off by default**, so the
closed-shell dipole-only model above is bit-for-bit what it always was unless
they are explicitly switched on (``spin=True``, ``quadrupole_blocks=(...)``,
``hyperpolarizability=True``)::

    chi^s_i   0e         spin field, (chi_up - chi_down)/2    unconstrained
    J^x_i     0e         on-site exchange                     > 0, STRICTLY
    kappa^s   edge       per-spin bond hardness split         symmetric, > 0
    A_Theta_i 0e+2e+4e   quadrupole stiffness (5x5)           positive definite
    C^qT_i    2e         charge-quadrupole coupling           unconstrained
    C^muT_i   1o+2o+3o   dipole-quadrupole coupling           unconstrained
    d_i       1o         hyperpolarizability direction        unconstrained

Three of the closed-shell constraints are load-bearing, and each one is the fix
for a measured failure recorded in ``SPEC.md``.

**The hardness floor.**  The electrostatic energy is unbounded below whenever
``eta_i < J(0) = 2 K_E / (sigma_i sqrt(pi))``, and a direct solve of an unbounded
quadratic form does not fail loudly - it silently returns a saddle point, which
then yields negative polarizabilities.  The threshold is not academic: Mulliken's
carbon hardness, 9.997 eV, sits *below* the 10.36-10.70 eV that graphene and
diamond require, and every common metal (Na 4.6, Al 6.0, Cu 6.5, Pt 7.4 eV) is
below threshold at its real bond length.  A raw linear ``eta`` head is therefore a
latent catastrophic bug, not a modelling preference, so this module emits
``eta_i = softplus(raw_i) + J(0; sigma_i)``.  Because ``sigma`` is itself a
readout, the floor is per-atom and is evaluated from the *predicted* sigma of the
same forward pass - it can never be precomputed or hard-coded.  The floor is
taken from ``kernels.kernel_self`` rather than rewritten here, so that the
solver's ``eta_i >= J(0)`` construction assertion compares two values produced by
the same expression from the same ``sigma`` and cannot fail by one ulp.  The
inequality is exact in floating point as well as in exact arithmetic: softplus is
non-negative, and round-to-nearest addition of a non-negative quantity never
returns less than the original.

**Bond-hardness symmetry.**  Bond charge transfer is antisymmetric by
construction, ``t_ij = -t_ji``, so the energy contribution ``0.5 kappa_ij t_ij^2``
is well defined only if ``kappa_ij = kappa_ji``.  Attention logits do not have
that property: ``Q_i . K_j != Q_j . K_i`` in general.  The logit is therefore
explicitly symmetrised before the softplus, by :func:`symmetrise_bond_logits`,
and ``kappa_0 > 0`` strictly - a vanishing bond hardness destroys the positive
definiteness of the SQE Hessian.

**The dipole stiffness must be positive definite.**  ``A_mu`` is built as
``L L^T + eps I`` from a symmetric rank-2 tensor ``L``, never read out freely.
``L`` is assembled from a ``0e`` trace channel and a ``2e`` traceless channel,
which is exactly the symmetric part of ``1o (x) 1o``, and ``L L^T`` is symmetric
positive semidefinite for *any* ``L``, symmetric or not; the ``eps I`` makes it
strictly definite.  The code writes ``L L^T`` and not ``L @ L`` on purpose: the
Clebsch-Gordan slices are symmetric only to double-precision round-off
(``max |W - W^T| = 1.55e-15``), so ``L L^T = L^2`` holds to that tolerance and
not exactly, and only the ``L L^T`` form makes definiteness independent of it.
The parameterisation is surjective onto the positive semidefinite cone (every
such matrix is the square of a symmetric matrix), so nothing is lost.

Spin channels (SPEC 9)
----------------------

``spin=True`` splits the charge sector into ``(q^up, q^down)`` and emits three
extra fields.  The induced dipole still responds to the *total* density, so
``A_mu``, ``ell`` and ``sigma`` are untouched.

**``J^x_i > 0`` is not a preference, it is the definiteness condition.**  In the
``(q_up, q_down)`` basis the on-site form ``0.5 eta q^2 + 0.5 J^x s^2`` has
Hessian ``[[eta + J^x, eta - J^x], [eta - J^x, eta + J^x]]``, whose eigenvalues
are ``2 eta`` (charge mode) and ``2 J^x`` (spin mode).  ``J^x <= 0`` therefore
makes the on-site block indefinite and the energy unbounded below in exactly the
way ``eta < J(0)`` does - and with the same silent failure, a solve that returns
a saddle rather than raising.  ``J^x`` is emitted as ``softplus(raw) + floor``
with ``floor > 0``, the same construction ``eta`` uses, and the inequality is
exact in IEEE-754 for every finite ``raw``: ``softplus`` never returns a negative
double (it underflows to exactly ``0.0``), and round-to-nearest addition of a
non-negative quantity to a positive one is monotone, so the sum can never fall
below the floor.  Measured over ``raw`` from ``-1e308`` to ``+700``, the minimum
returned ``J^x`` is exactly the floor, bitwise.

**So where does spin polarisation come from?**  Not from ``J^x``.  A positive
``J^x`` *penalises* ``s_i``, which is the opposite of what Hund's rule and the
Stoner criterion describe - and that opposite sign is precisely the non-definite,
unbounded case.  In this model spin polarisation is produced by the
spin-dependent **linear** term: ``chi_up`` and ``chi_down`` differ, and the
minimiser trades ``s`` against the ``J^x`` penalty.  The head therefore emits a
spin field ``chi^s_i`` and forms ``chi^{up,down}_i = chi_i +/- chi^s_i``
(the sign convention follows ``U = ... + chi^T m``, so a *lower* ``chi`` for a
channel attracts charge into it).  ``chi^s`` is unconstrained in sign; it is the
only channel in the spin sector that is.

**The error this inherits, stated up front** (SPEC 9.2).  A quadratic-in-``s``
on-site energy violates the fractional-spin (flat-plane) condition.  For a
one-electron system whose exact energy is FLAT in spin polarisation, a quadratic
term at ``J^x = 2 eV`` manufactures a 1.0 eV splitting at ``m = 1``, against real
Fe(II) high-spin/low-spin gaps of 0.1-0.4 eV - an artifact 100-400% of the
signal.  AIMNet2-NSE and MACE-POLAR-1 use the same functional form.  It is not a
reason to refuse the feature, but **the spin-state splitting this model produces
is a FITTED quantity, not a derived one.**

Quadrupoles (SPEC 10) - detected, not excluded
----------------------------------------------

Each quadrupole coupling coefficient is a tensor in its own right, and can only
be read out if the node features carry the irreps it decomposes into::

    block                                  irreps          minimum l_max
    A_mu      sym. part of 1o (x) 1o       0e + 2e         2
    q-Theta                                2e              2
    mu-Theta                               1o + 2o + 3o    3   (see below)
    A_Theta   sym. part of 2e (x) 2e       0e + 2e + 4e    4

The ``A_Theta`` row is the one worth deriving rather than quoting.  ``2e (x) 2e``
is ``0e + 1e + 2e + 3e + 4e``; of those, the ``1e`` and ``3e`` Clebsch-Gordan
tensors are *antisymmetric* in their two ``2e`` indices and so contribute
nothing to a symmetric bilinear form, leaving ``0e + 2e + 4e``.  Verified with
e3nn 0.4.4 by splitting each ``wigner_3j(l, 2, 2)`` into its symmetric and
antisymmetric parts: ``l = 0, 2, 4`` give ``|sym| = 1`` and ``|asym| <= 6.6e-16``,
``l = 1, 3`` give ``|asym| = 1`` and ``|sym| <= 5.0e-16``.  The dimensions close:
``5 * 6 / 2 = 15 = 1 + 5 + 9``, and the fifteen normalised basis matrices have
numerical rank 15 over the 25-dimensional space of 5x5 matrices.

**TRACE's ``l_max = 2`` features cannot carry a general ``A_Theta``.**  They have
no ``4e`` block, so the ``4e`` part of the stiffness would be identically zero -
the silent-dead-block failure SPEC 3 exists to reject.  What remains, ``0e + 2e``,
is a real and useful object but it is *not* a general quadrupole stiffness: it is
an isotropic response plus the lowest anisotropy, and it cannot represent, for
example, a stiffness that distinguishes two different traceless deformations
lying in the same ``2e`` shell.  This module therefore refuses to build it by
accident: ``A_Theta`` without ``4e`` features raises
:class:`QuadrupoleIrrepsError` unless the caller passes
``allow_truncated_a_theta=True``, which additionally emits a
:class:`QuadrupoleTruncationWarning` **at construction** and records the choice
in the ``physical_bounds`` buffer so a checkpoint cannot later be reinterpreted
as a full one.  The restriction is stated when the head is built, never
discovered downstream.

**``mu-Theta`` needs odd parity at l = 2, which l_max alone does not give you.**
``mu`` is ``1o`` and ``Theta`` is ``2e``, so their product is parity-odd and the
coupling coefficient must be too: ``1o (x) 2e = 1o + 2o + 3o``.  A network with
the usual natural parity ``(-1)^l`` supplies ``0e + 1o + 2e + 3o`` at
``l_max = 3`` - which has ``1o`` and ``3o`` but **no ``2o``**.  ``l_max >= 3`` is
necessary and not sufficient, which is exactly why this module tests the
*irreps* and not ``l_max``.  The requirement is checked with the same
``output_mask`` reachability logic that guards every other head here, through
:func:`unreachable_output_irreps`.

Hyperpolarizability (``hyperpolarizability=True``) - and what it cannot know
---------------------------------------------------------------------------

The first hyperpolarizability is the third-order response,
``beta = -d3U/dE3`` under ``U = U0 - mu.E - (1/2) alpha:EE - (1/6) beta:EEE``.
It is rank 3 and **odd** under inversion, so it vanishes identically in a
centrosymmetric environment - which is why second-harmonic and sum-frequency
generation are surface-specific probes in the first place.

This sector emits one thing: a ``1o`` vector ``d_i``, from which
:func:`~trace_electrostatics.observables.hyperpolarizability` builds
``beta_i = alpha_i (x) d_i``.  Three facts make that the right construction,
and each was measured rather than assumed:

* **The irreps close on the full tensor.**  ``alpha`` is ``0e + 2e`` and ``d``
  is ``1o``, so ``alpha (x) d = 2x1o + 1x2o + 1x3o``, of dimension **18** -
  exactly the non-Kleinman SHG tensor, a rank-3 object symmetric in its two
  field indices and nothing more (3 x 6 = 18).  Verified with e3nn, and
  independently as a cartesian rank: the span of ``{d (x) alpha}`` over random
  ``d`` and symmetric ``alpha`` has singular values ``sigma_18 = 1.54e+01`` and
  ``sigma_19 = 6.3e-15``, i.e. numerical rank 18.  The commonly used
  ``1o + 3o`` assignment has dimension **10** and silently imposes **Kleinman
  symmetry**, which holds only far from resonance - that is, precisely not
  where an SHG or SFG experiment is done.
* **The selection rule is structural.**  ``d`` is ``1o``, so it can only be
  produced from the ``1o`` block of the features; at a centrosymmetric site
  that block is identically zero, ``d`` is zero, and ``beta`` vanishes without
  anything having to check for it.  Measured on an explicitly centrosymmetric
  neighbour set: ``|d| = 2.9e-17``, against ``1.6e-01`` at a polar site whose
  ``alpha`` is the same size.
* **It costs 3 parameters, not 18.**  ``alpha`` is already computed by the
  model; only the direction is new.

**What this construction cannot represent, measured.**  ``beta = alpha (x) d``
is non-zero only where the site's point group admits an invariant *vector*.
That is a stronger condition than "not centrosymmetric", and the gap is real.
Projecting onto the invariant subspace of each irrep of ``beta``
(``P = (1/|G|) sum_g D(g)``, rank taken numerically)::

    site group                       1o   2o   3o   beta dof   reached by alpha (x) d
    Td   (diamond, zincblende)        0    0    1     1 of 18          0
    Oh   (rocksalt, has i)            0    0    0     0 of 18          0
    C3v  (polar)                      1    0    2     4 of 18          1
    C2v  (polar)                      1    1    2     5 of 18          1
    Ci   (inversion only)             0    0    0     0 of 18          0
    C1   (no symmetry)                3    5    7    18 of 18          3

The ``Ci`` row is the selection rule and it is exact: at an inversion centre
*no* component of ``beta`` is allowed, and this construction returns zero.  The
``Td`` row is the blind spot: Td's vector representation is ``T2``, irreducible
and not the trivial representation, so an equivariant ``1o`` readout **must**
vanish at a tetrahedral site - yet Td does admit one invariant in ``3o``, the
``xyz`` function, which is precisely the ``d14`` that makes zincblende
SHG-active.  Measured on an ideal diamond/zincblende neighbour shell, the
``1o`` block of the node features is ``0.000e+00`` (the four tetrahedral unit
vectors sum to zero) and so is ``d``.  Tying the ``3o`` of ``beta`` to ``d``
means that where symmetry kills ``d`` it kills every component of ``beta``,
including ones symmetry would have allowed.  A free ``3o`` readout would reach
that component - but TRACE's ``l_max = 2`` features carry no ``3o`` at all, so
at these irreps nothing in this network can produce it either.  The honest
statement is that ``alpha (x) d`` covers polar sites - surfaces, interfaces,
defects, molecules, the SFG and SHG targets this package exists for - and is
identically zero at a tetrahedral one.

**The site rule and the crystal rule are different statements.**  ``d`` is a
per-atom readout, so the selection rule enforced here is a *site* rule.  A
centrosymmetric crystal need not have centrosymmetric sites, and usually does
not, so per-site ``beta`` is non-zero there and only the cell sum vanishes.
It vanishes structurally, for the same reason and with no extra machinery:
inversion maps site ``i`` to site ``i'`` with an environment that is the exact
negative of ``i``'s, so ``d_i' = -d_i`` (``1o``, odd) while ``alpha_i' =
alpha_i`` (``0e + 2e``, even), giving ``beta_i' = -beta_i``.  Measured on a
centrosymmetric simple-cubic cell with a two-atom basis at ``+-u``, both sites
polar: ``|d_0| = |d_1| = 3.056827e-01`` with ``|d_0 + d_1| = 8.1e-17``,
``|alpha_0 - alpha_1|_F = 1.9e-16``, per-site ``|beta_i|_F = 2.873804e-01`` and
``|sum_i beta_i|_F = 1.110340e-16``.  On a generic inversion-symmetric cluster
``S u -S`` with ``C1`` sites, per-site ``|beta_i|_F`` runs 1.747839e-01 to
2.609384e-01 against ``|sum_i beta_i|_F = 8.062661e-16``.  Diamond is the one
case where even the per-site tensor is zero, and that is the Td row of the table
above rather than the inversion rule: ``|d| = 2.6e-16`` and ``6.9e-16`` on the
two basis atoms.

**Why not an anharmonic on-site energy.**  A cubic-in-moments term
``(c/3) mu^3`` was tried and rejected.  It forces ``beta = -2 c alpha^3``
exactly - Miller's rule as an *identity* the model cannot violate and therefore
cannot be tested against (measured: ``beta/alpha^3 = -1.8000000000`` at three
different stiffnesses, agreeing with ``-2c`` to 1.8e-15).  Worse, it is false.
An exact two-level charge-transfer model, diagonalised in a field, gives at
fixed gap and fixed transition dipole, varying only the ground/excited dipole
difference ``Delta-mu``::

    Delta-mu     0.000   0.111   0.250   0.500   1.000   1.500   2.000
    alpha        1.3333  1.3333  1.3333  1.3333  1.3333  1.3333  1.3333
    beta         0.0000  0.2963  0.6667  1.3333  2.6667  4.0000  5.3333

``alpha`` does not move at all (spread 0.00%) while ``beta`` runs from 0 to
5.333, exactly linearly in ``Delta-mu``; ``beta/alpha^3`` varies by 18x.  So
``beta`` is **not a function of alpha**, and the degree of freedom that carries
it has to be an independent channel.  That channel is ``d``.

**``d`` IS UNIDENTIFIABLE FROM ENERGIES, FORCES AND POLARIZABILITIES.**  This is
the most important sentence in this section.  ``d`` appears in no energy, no
force, no stress, no charge, no dipole and no ``alpha``: it is read off the
features and handed straight to the beta constructor.  Nothing in a training
run that fits energies, forces, dipoles or polarizabilities produces any
gradient on it at all.  **With no beta labels, ``d`` is exactly what its
initialisation made it, and the predicted beta is noise wearing a
physics-shaped hat** - correctly equivariant noise, obeying the right selection
rule, with an entirely arbitrary magnitude and sign.  The sector is off by
default and warns at construction (:class:`HyperpolarizabilityUnidentifiableWarning`)
for that reason and no other.

Scope and traps
---------------

**The quartic remains out of scope, and beta does not reintroduce it.**  SPEC
1.5 admits a quadratic form only.  The quadrupole sector extends the *rank* of
the moments and the ``d`` sector adds a pure readout; neither changes the
*order* of the energy.  ``beta`` here is a readout, not a term in ``U``, and
the model's energy is still exactly quadratic in the moments.

**The e3nn dtype trap.**  ``torch.set_default_dtype(torch.float64)`` must be
called *before* any e3nn module in this file is constructed.  e3nn generates and
caches TensorProduct code under the default dtype in effect at first
construction; calling ``.double()`` on the module afterwards is a silent no-op
for that cached kernel and costs roughly seven orders of magnitude of precision
(SPEC 3: 2.54e-08 against 1.94e-15; measured here on ``A_mu``, 6.16e-09 for a
float32-built module later promoted with ``.double()`` against 3.17e-16 for the
same module built under float64 - a factor of 1.9e7).  There is no warning and
the model still trains, so this can only be caught by looking for it.

``ElectrostaticConfig.validate_dtype`` cannot catch it: it runs when the
electrostatics are *attached*, which is strictly after the heads were
constructed and the kernels were baked, and it passes cleanly on an already
ruined module.  :class:`ElectrostaticHeads` therefore repeats the check in its
own ``__init__``, before the first ``o3.Linear`` is built, which is the last
moment at which the mistake is still fixable.  Pass ``require_float64=False``
for a deliberate single-precision experiment.

**The zero-instruction trap.**  An :class:`e3nn.o3.Linear` whose output irreps
cannot be produced from its input irreps - a parity mismatch such as ``0e -> 1o``
is the usual cause - constructs successfully with zero instructions and zero
parameters, returns identically zero forever, and emits no warning.  Every
``o3.Linear`` here is built through :func:`checked_o3_linear`, which raises with
the offending head named.  The check is a raise rather than an ``assert`` so that
``python -O`` cannot silently reinstate the bug.  Counting instructions, or even
counting which output irreps an instruction *mentions*, is not enough: see
:func:`checked_o3_linear` for the zero-multiplicity hole that ``output_mask``
closes.

Conventions
-----------

Energies are in eV, lengths in angstrom, charges in ``e`` (see ``constants.py``).

``A_mu`` is returned in the same cartesian frame as the positions fed to the
network, so it can be contracted directly with physical dipole vectors.  This is
worth stating because e3nn's ``1o`` components are conventionally described as
ordered ``(y, z, x)``: that naming refers to e3nn's internal polar axis, and
e3nn is self-consistent about it, with degree-1 spherical harmonics acting as the
identity and ``o3.Irrep("1o").D_from_matrix(R)`` returning ``R`` itself (measured
on e3nn 0.4.4: ``max |D - R| = 1.10e-15`` over 2000 random rotations, median
1.11e-16; over the six axis permutations, mean 8.16e-17 and max 1.22e-16 - i.e.
identity to double-precision round-off, not to some smaller special number).
The ordering is unchanged on e3nn 0.6.0, and so are the Clebsch-Gordan tensors
themselves: ``wigner_3j`` norms are 1.0 and the ``(2,1,1)`` entries agree
entry-for-entry, signs included.  What DOES change is the accuracy of
``D_from_matrix``: 0.4.4 reads its Wigner matrices out of a float64 ``_Jd``
cache, while 0.6.0 rebuilds them through ``so3_generators``, which takes no
dtype and so builds at ``torch.get_default_dtype()``.  At the float32 default
that gives ``max |D - R| = 7.26e-07``, and 2.13e-10 even with the default set
to float64.  ``D_from_matrix`` is used only by the rotation tests, never in this
package's forward path, so it costs accuracy in the harness rather than in the
physics - but a suite quoting equivariance at 1e-18 will appear to fail at 1e-07
on 0.6.x, and the fix is the tolerance, not the model.

The same dtype hazard DOES reach the physics through ``wigner_3j``, which also
builds at the default dtype.  ``o3.wigner_3j(2, 1, 1).to(torch.float64)`` is
therefore NOT a double-precision tensor - it widens the storage of a value that
has already been rounded to float32, and the ~6e-09 error is baked in.  Every
basis below passes ``dtype=torch.float64`` into the call instead.  The property
this damages is orthonormality (``|W W^T - I|``: 3.4e-08 built-then-cast against
1.8e-15 built wide); the trace and the symmetry are exactly zero either way,
because those are structural zeros that survive rounding, which is why the
obvious checks miss it.  ``tests/electrostatics/test_e3nn_conventions.py``
asserts all of this, including a static check on the call pattern.
Since TRACE feeds plain cartesian ``edge_vec`` to
``o3.SphericalHarmonics``, index ``a`` of any ``1o`` quantity in this network *is*
cartesian axis ``a``, and the Clebsch-Gordan basis used below needs no relabelling.
The rotation test in the audit checks exactly this, by rotating the input features
with the Wigner D matrices and requiring ``A_mu -> R A_mu R^T`` for the cartesian
``R``.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Iterable, NamedTuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from torch import Tensor

from .constants import K_E
from .kernels import DEFAULT_HARDNESS_MARGIN, hardness_floor, kernel_self, quadrupole_cartesian_basis
from .solver import DEFAULT_EXCHANGE_FLOOR, ElectrostaticParams

__all__ = [
    "ElectrostaticHeads",
    "ElectrostaticReadout",
    "SpinReadout",
    "QuadrupoleReadout",
    "DispersionReadout",
    "DispersionElectrons",
    "QuadrupoleIrrepsError",
    "QuadrupoleTruncationWarning",
    "HyperpolarizabilityReadout",
    "HyperpolarizabilityIrrepsError",
    "HyperpolarizabilityUnidentifiableWarning",
    "symmetrise_bond_logits",
    "antisymmetrise_bond_logits",
    "bond_hardness",
    "exchange_coupling",
    "checked_o3_linear",
    "unreachable_output_irreps",
    "a_mu_floor_from_polarizability_ceiling",
    "polarizability_ceiling_from_a_mu_floor",
    "a_mu_trace_init_from_polarizability",
    "a_theta_floor_from_quadrupole_ceiling",
    "quadrupole_ceiling_from_a_theta_floor",
    "a_theta_trace_init_from_quadrupole_polarizability",
    "quadrupole_cartesian_basis",
    "DEFAULT_KAPPA_FLOOR",
    "DEFAULT_A_THETA_FLOOR",
    "DEFAULT_HYPER_D_INIT_GAIN",
    "HYPER_D_IRREPS",
    "QUADRUPOLE_BLOCKS",
    "QUADRUPOLE_BLOCK_IRREPS",
    "TRACE_NODE_IRREPS",
]


# --- The polarizability ceiling, both directions ---------------------------
#
# Defined above the defaults because ``DEFAULT_A_MU_FLOOR`` is *derived* from a
# stated ceiling rather than asserted to match one.


def a_mu_floor_from_polarizability_ceiling(alpha_max: float, ell: float = 1.0) -> float:
    """Convert a polarizability ceiling in A^3 to the ``A_mu`` floor in eV.

    Minimising ``0.5 A_mu mu_hat^2 - E . mu`` with ``mu = e * ell * mu_hat`` gives
    ``alpha = e^2 ell^2 / A_mu``, which is in ``e^2 A^2 / eV`` and is **not** a
    volume; multiplying by ``K_E`` converts it to A^3 (SPEC section 0).  Inverting
    that relation gives the floor that caps the induced polarizability.

    The ceiling is only a ceiling if ``ell`` is bounded above as well, because
    ``alpha_max`` grows as ``ell^2``; that is why :class:`ElectrostaticHeads`
    emits ``ell`` through a two-sided sigmoid rather than a softplus.  Pass the
    module's ``ell_max``, not a typical ``ell``.
    """
    if alpha_max <= 0.0:
        raise ValueError(f"alpha_max must be positive, got {alpha_max}")
    if ell <= 0.0:
        raise ValueError(f"ell must be positive, got {ell}")
    return K_E * ell * ell / alpha_max


def polarizability_ceiling_from_a_mu_floor(a_mu_floor: float, ell_max: float) -> float:
    """Inverse of :func:`a_mu_floor_from_polarizability_ceiling`, in A^3.

    ``alpha_max = K_E * ell_max^2 / a_mu_floor`` is the largest induced atomic
    polarizability the parameterisation can reach: ``A_mu`` cannot fall below
    ``a_mu_floor`` and ``ell`` cannot exceed ``ell_max``, so the product is
    bounded by construction rather than by hope.
    """
    if a_mu_floor <= 0.0:
        raise ValueError(f"a_mu_floor must be positive, got {a_mu_floor}")
    if ell_max <= 0.0:
        raise ValueError(f"ell_max must be positive, got {ell_max}")
    return K_E * ell_max * ell_max / a_mu_floor


def a_mu_trace_init_from_polarizability(
    alpha_init: float, a_mu_floor: float, ell: float = 1.0
) -> float:
    """Isotropic ``L`` that starts ``A_mu`` at the stated polarizability, in A^3.

    ``A_mu = L L^T + a_mu_floor I``, so an isotropic ``L = c I`` gives
    ``A_mu = (c^2 + a_mu_floor) I`` and ``alpha = K_E ell^2 / (c^2 +
    a_mu_floor)``.  Inverting for ``c`` is what lets the initial dipole stiffness
    be stated as a *polarizability* - the thing that has a physical value one can
    look up - instead of as an eV number whose consequences have to be worked out
    by hand every time it changes.
    """
    if alpha_init <= 0.0:
        raise ValueError(f"alpha_init must be positive, got {alpha_init}")
    stiffness = a_mu_floor_from_polarizability_ceiling(alpha_init, ell=ell)
    if stiffness <= a_mu_floor:
        raise ValueError(
            f"alpha_init={alpha_init} A^3 needs A_mu = {stiffness} eV, which is at "
            f"or below the floor a_mu_floor={a_mu_floor}: the floor alone already "
            "caps the polarizability below the requested initial value."
        )
    return math.sqrt(stiffness - a_mu_floor)


# --- The same pair of conversions, one rank up ------------------------------
#
# The quadrupole sector inherits SPEC 0's unit trap intact, and inherits it
# worse: the reduced quadrupole is ``Theta_hat = Theta / (e * ell^2)``, so
# minimising ``0.5 A_Theta Theta_hat^2 - (grad E) . Theta`` gives a quadrupole
# polarizability ``C = e^2 ell^4 / A_Theta`` in ``e^2 A^4 / eV``, which is NOT an
# A^5 volume until it is multiplied by K_E. The ell^4 is the part that bites:
# a factor of two in ell moves the ceiling by sixteen, not by four, so an ell
# that is merely "bounded" is not enough - it has to be bounded at a value whose
# fourth power was looked at.


def a_theta_floor_from_quadrupole_ceiling(c_max: float, ell: float = 1.0) -> float:
    """Convert a quadrupole-polarizability ceiling in A^5 to the ``A_Theta`` floor.

    ``A_Theta >= K_E * ell^4 / c_max``.  Pass the module's ``ell_max``, not a
    typical ``ell``: the ceiling is only a ceiling if ``ell`` is bounded above,
    and here it scales as the *fourth* power.
    """
    if c_max <= 0.0:
        raise ValueError(f"c_max must be positive, got {c_max}")
    if ell <= 0.0:
        raise ValueError(f"ell must be positive, got {ell}")
    return K_E * ell**4 / c_max


def quadrupole_ceiling_from_a_theta_floor(a_theta_floor: float, ell_max: float) -> float:
    """Inverse of :func:`a_theta_floor_from_quadrupole_ceiling`, in A^5."""
    if a_theta_floor <= 0.0:
        raise ValueError(f"a_theta_floor must be positive, got {a_theta_floor}")
    if ell_max <= 0.0:
        raise ValueError(f"ell_max must be positive, got {ell_max}")
    return K_E * ell_max**4 / a_theta_floor


def a_theta_trace_init_from_quadrupole_polarizability(
    c_init: float, a_theta_floor: float, ell: float = 1.0
) -> float:
    """Isotropic ``L`` that starts ``A_Theta`` at the stated quadrupole
    polarizability, in A^5.

    ``A_Theta = L L^T + a_theta_floor I_5``, so an isotropic ``L = c I_5`` gives
    ``A_Theta = (c^2 + a_theta_floor) I_5``.  Stating the initialisation as a
    polarizability rather than as a bare eV number is the same discipline the
    dipole sector uses, and for the same reason: ``a_theta_trace_init`` on its own
    has no interpretation anyone can check.
    """
    if c_init <= 0.0:
        raise ValueError(f"c_init must be positive, got {c_init}")
    stiffness = a_theta_floor_from_quadrupole_ceiling(c_init, ell=ell)
    if stiffness <= a_theta_floor:
        raise ValueError(
            f"c_init={c_init} A^5 needs A_Theta = {stiffness} eV, which is at or "
            f"below the floor a_theta_floor={a_theta_floor}: the floor alone "
            "already caps the quadrupole polarizability below the requested "
            "initial value."
        )
    return math.sqrt(stiffness - a_theta_floor)


# --- Defaults -------------------------------------------------------------
#
# Node irreps produced by TransformersACEV3 at hidden_dim = 64, l_max = 2.
TRACE_NODE_IRREPS = "64x0e + 32x1o + 16x2e"

# Gaussian charge width. J(0) = 2 K_E / (sigma sqrt(pi)) is 10.832 eV at the
# 1.5 A midpoint, bracketing the 8.1-16.3 eV hardness floors that the audited
# molecular and solid-state systems need. Bounding sigma bounds the floor.
DEFAULT_SIGMA_MIN = 1.0
DEFAULT_SIGMA_MAX = 2.0

# Moment length scale, used to reduce dipoles as mu_hat = mu / (e * ell).
# Bounded on BOTH sides, like sigma and for the same kind of reason: the induced
# polarizability is alpha = K_E * ell^2 / A_mu, so an unbounded ell means an
# unbounded polarizability no matter what floor A_mu carries. An earlier version
# emitted ell = ell_min + softplus(raw), which has no upper bound at all: nothing
# stopped training from walking ell to 5.37 A (ceiling 4152 A^3) or 20.4 A
# (ceiling 59926 A^3) while the constant below still advertised 144 A^3.
DEFAULT_ELL_MIN = 0.25
DEFAULT_ELL_MAX = 2.0
DEFAULT_ELL_INIT = 1.0

# Strictly positive bond-hardness floor: kappa -> 0 destroys positive
# definiteness of the SQE Hessian. This is the ONLY definition of the floor in
# the package -- it is a property of the head that emits kappa, and it is baked
# into that head at construction, so there is deliberately no second copy of it
# on ElectrostaticConfig for a later assignment to silently disagree with.
DEFAULT_KAPPA_FLOOR = 0.1

#: SPEC 17.4: fraction of r_max below which the dissociation envelope is exactly
#: 1.  None selects the legacy full-range cosine envelope.  0.75: first-shell
#: bonds (the dominant charge-transfer channels) must lie in the flat region,
#: or the switch's fixed-sign d ln t/dr contaminates every Born charge --
#: measured on rocksalt (NN 2.80 A, r_max 5 A): q = +1.45, Z* = +1.40 at 0.75,
#: +1.05 at 0.5, -0.54 with the cosine.
DEFAULT_ENVELOPE_ONSET = 0.75

# Hard ceiling on the induced atomic polarizability, in A^3, and the dipole
# stiffness floor derived from it. Caesium, the most polarizable neutral atom at
# 59.4 A^3, leaves an order of magnitude of headroom. The floor is *computed*
# from the ceiling rather than asserted to match it, so the two cannot drift:
# with ell bounded above by DEFAULT_ELL_MAX the bound
# alpha <= K_E * ell_max^2 / a_mu_floor is enforced by construction
# (see the unit trap in SPEC section 0: e^2 A^2 / eV is not a volume until
# multiplied by K_E).
DEFAULT_ALPHA_MAX = 576.0
DEFAULT_A_MU_FLOOR = a_mu_floor_from_polarizability_ceiling(
    DEFAULT_ALPHA_MAX, ell=DEFAULT_ELL_MAX
)

# Initial isotropic part of L, stated as the polarizability it produces rather
# than as a bare eV number. A_mu = L L^T + a_mu_floor I, so an isotropic
# L = c I starts every atom at alpha = K_E ell^2 / (c^2 + a_mu_floor).
#
# 1.5 A^3 sits between the experimental atomic polarizabilities of the elements
# this model is trained on -- H 0.67, O 0.80, C 1.76 A^3 -- so the dipole relay
# starts at roughly the right strength. The previous default of
# a_mu_trace_init = 1.0 meant A_mu ~ 1 eV, i.e. alpha ~ 14 A^3: every atom an
# order of magnitude more polarizable than any real one, which is a soft,
# strongly coupled dipole block, not a neutral starting point. Measured on real
# TransformersACEV3 features over six seeds at the default init gain, the two
# choices give alpha in [7.44, 55.2] A^3 and in [1.18, 2.30] A^3 respectively.
DEFAULT_ALPHA_INIT = 1.5
DEFAULT_A_MU_TRACE_INIT = a_mu_trace_init_from_polarizability(
    DEFAULT_ALPHA_INIT, DEFAULT_A_MU_FLOOR
)

# Gain applied to the A_mu readout weights at initialisation, and the reason it
# is not 1.0. The o3.Linear emits a 0e coefficient c0 and five 2e coefficients
# c2, all of unit scale for unit-scale node features, and
# L = (c0 + a_mu_trace_init) I + sum_m c2_m W_m. The five W_m are unit Frobenius
# norm each, so the anisotropy enters with norm |c2| ~ sqrt(5) = 2.24 against
# ||I||_F = sqrt(3) = 1.73 for the offset. Nothing scaled that down, so at gain
# 1.0 the anisotropy was comparable to or larger than the isotropic part it was
# supposed to perturb, L came out indefinite, and A_mu = L L^T sat on its own
# floor. Expected relative deviation of L from a_mu_trace_init * I is
# gain * sqrt(3 + 5) / (a_mu_trace_init * sqrt(3)) = 1.633 * gain / 3.082,
# i.e. 10.6% at 0.2.
#
# Measured on real TransformersACEV3 features, six seeds, six atoms, at the
# default a_mu_trace_init:
#
#     gain 1.0   min eig 0.165 (floor is 0.100)  cond <= 294   alpha 0.57-29.4 A^3
#     gain 0.2   min eig 3.26                    cond <=  3.68 alpha 1.18-2.30 A^3
#     gain 0.05  min eig 7.70                    cond <=  1.33 alpha 1.38-1.60 A^3
#
# The first row is the soft, ill-conditioned, order-of-magnitude-spread state
# a_mu_trace_init exists to avoid. 0.2 rather than 0.05 because the anisotropy
# must start SMALL, not negligible: at 0.05 the 2e channel's influence on the
# energy is so weak that the fixed-feature strain gradient comes out very nearly
# symmetric (antisymmetry 7.8e-03 against the 1e-02 the regression suite
# requires as evidence that the 1o/2e blocks reach the readout at all).
# It is an INITIALISATION, not a constraint: training is free to grow the
# anisotropy from there.
DEFAULT_A_MU_INIT_GAIN = 0.2

# Initial gain on the geometry-determined CG warm start for bond charge transfer.
# Partial charges are O(0.3 e) and raw attention logits are O(1), so a modest gain
# keeps the seed a small perturbation of zero at initialisation.
DEFAULT_T_SEED_SCALE = 0.1

_N_SCALAR_OUTPUTS = 4  # chi, eta_raw, sigma_raw, ell_raw
_CHI, _ETA_RAW, _SIGMA_RAW, _ELL_RAW = range(_N_SCALAR_OUTPUTS)

# --- Spin sector defaults (SPEC 9), used only when spin=True ---------------
#
# The strictly positive on-site exchange floor is NOT defined here. It is
# solver.DEFAULT_EXCHANGE_FLOOR (1e-3 eV/e^2), imported above, for the same
# reason the hardness floor is kernels.kernel_self(sigma) rather than a second
# copy of 2 K_E / (sigma sqrt(pi)): the head that emits the value and the
# assertion that checks it must be looking at one expression, or they can
# disagree by an ulp -- or, far worse, by a factor, the way config.kappa_0 once
# silently disagreed with heads.DEFAULT_KAPPA_FLOOR by 9.5x while a comment
# asserted they matched.
#
# Its value matters to the physics and is worth restating: 1e-3 eV/e^2 is four
# orders of magnitude BELOW the smallest exchange splitting anyone fits (Fe(II)
# high-spin/low-spin gaps are 0.1-0.4 eV, SPEC 9.2), which is the point. The
# floor exists to keep the spin mode strictly definite, not to model exchange. A
# floor at, say, 0.1 eV would sit inside the range of real splittings and would
# be doing chemistry nobody asked it to do; it would also close off the
# soft-exchange limit, which SPEC 9 keeps reachable the way SPEC 1.4 keeps
# kappa -> 0 reachable.

# Gain on the spin heads at initialisation. The spin sector must start as a
# small perturbation of the closed-shell model, not as an equal partner: at
# chi^s = 0 and equal per-spin kappa the whole extension reduces to the
# closed-shell one *exactly*, and that is the state training should begin near.
DEFAULT_SPIN_INIT_GAIN = 0.1

_N_SPIN_SCALAR_OUTPUTS = 2  # chi_spin, j_exchange_raw
_CHI_SPIN, _J_EXCHANGE_RAW = range(_N_SPIN_SCALAR_OUTPUTS)

# --- Quadrupole sector defaults (SPEC 10), used only when enabled ----------
#
# The irreps each coupling coefficient decomposes into. This is the contract
# from SPEC 10, held as DATA so the construction check and the error message
# cannot drift apart, and so a caller can interrogate it without constructing a
# head. Derived in the module docstring and verified against e3nn; see
# tests and the docstring of _quadrupole_stiffness_basis.
#
# A_mu is listed for completeness even though it is not optional: it is the
# l = 1 member of the same family and makes the pattern legible.
QUADRUPOLE_BLOCK_IRREPS = {
    "A_mu": "1x0e + 1x2e",
    # The permanent quadrupole source of SPEC 12: a linear term chi_Theta .
    # Theta_hat, one 2e object per atom. Representable at l_max = 2.
    "chi_theta": "1x2e",
    "q_theta": "1x2e",
    "mu_theta": "1x1o + 1x2o + 1x3o",
    "A_Theta": "1x0e + 1x2e + 1x4e",
}

#: The blocks ``quadrupole_blocks=`` accepts, in the order they are built.
# chi_theta is LAST so that every block list that existed before it keeps its
# construction order, and therefore its RNG stream.
QUADRUPOLE_BLOCKS = ("A_Theta", "q_theta", "mu_theta", "chi_theta")

#: ``A_Theta`` with the ``4e`` part dropped. Accepted ONLY behind
#: ``allow_truncated_a_theta=True``, and never selected silently.
_A_THETA_TRUNCATED_IRREPS = "1x0e + 1x2e"

# Hard ceiling on the induced atomic QUADRUPOLE polarizability, in A^5, and the
# A_Theta floor derived from it -- the same construction as DEFAULT_ALPHA_MAX and
# DEFAULT_A_MU_FLOOR, one rank up, so the two cannot drift.
#
# 2304 A^5 is DEFAULT_ALPHA_MAX * ell_max^2: it is the dipole ceiling carried up
# a rank at the same ell bound, which keeps the derived floor at the same 0.1 eV
# as a_mu_floor and kappa_floor rather than introducing a third unrelated scale.
# Atomic quadrupole polarizabilities are far smaller than this -- tens of A^5 for
# main-group atoms -- so like the dipole ceiling it is a guard rail, not a fit.
DEFAULT_C_THETA_MAX = 2304.0
DEFAULT_A_THETA_FLOOR = a_theta_floor_from_quadrupole_ceiling(
    DEFAULT_C_THETA_MAX, ell=DEFAULT_ELL_MAX
)

# Initial isotropic part of L for A_Theta, stated as the quadrupole
# polarizability it produces. 6.0 A^5 is 1.5 A^3 (DEFAULT_ALPHA_INIT) times
# ell_max^2 -- the dipole starting point carried up a rank on the same bound --
# so the quadrupole relay starts at the same relative strength as the dipole one
# rather than at an unrelated number.
DEFAULT_C_THETA_INIT = 6.0
DEFAULT_A_THETA_TRACE_INIT = a_theta_trace_init_from_quadrupole_polarizability(
    DEFAULT_C_THETA_INIT, DEFAULT_A_THETA_FLOOR
)

# Gain on the A_Theta readout weights at initialisation, and why it is smaller
# than DEFAULT_A_MU_INIT_GAIN. The argument is the one recorded for A_mu, rerun
# in five dimensions: the isotropic offset carries ||I_5||_F = sqrt(5) = 2.236,
# while unit-scale anisotropy coefficients contribute five unit-norm 2e slices
# and nine unit-norm 4e slices, i.e. sqrt(5 + 9) = 3.742. The anisotropy/isotropy
# ratio at unit gain is therefore 3.742 / 2.236 = 1.674, worse than the 1.291 of
# the dipole sector, and it grows with l. 0.15 puts the expected relative
# deviation of L from a_theta_trace_init * I_5 at
# gain * sqrt(5 + 5 + 9) / (a_theta_trace_init * sqrt(5)).
DEFAULT_A_THETA_INIT_GAIN = 0.15

# Gain on the q-Theta and mu-Theta coupling readouts at initialisation. These
# are OFF-DIAGONAL blocks of K with no definiteness constraint of their own: a
# large one can drive the full coupled Hessian indefinite even with every
# diagonal block comfortably positive, which is precisely the trap SPEC 2.1.3
# records (q block +0.58, dipole block +16.45, full coupled -1.05). Starting them
# small is the only thing this module can do about it; the Schur-complement check
# at the solution is what actually catches it.
DEFAULT_QUADRUPOLE_COUPLING_INIT_GAIN = 0.05

_QUADRUPOLE_DIM = 5  # components of a symmetric traceless cartesian 3x3 tensor
_SPATIAL_DIM = 3

# --- Hyperpolarizability sector, used only when hyperpolarizability=True ----
#
# beta_i = alpha_i (x) d_i, so the only thing this sector reads out is the 1o
# vector d. Its irreps are held as a module constant for the same reason
# QUADRUPOLE_BLOCK_IRREPS is: the construction check, the error message and any
# caller that wants to ask "can these features carry it?" must all be looking at
# one string.
#
# Why 1o and not something else: alpha is the symmetric part of 1o (x) 1o, i.e.
# 0e + 2e, and (0e + 2e) (x) 1o = 2x1o + 1x2o + 1x3o, of dimension 18 -- the
# full non-Kleinman SHG tensor. Contracting with a 1o vector is what makes the
# product parity-ODD, which is the whole selection rule: beta vanishes at a
# centrosymmetric site because the 1o block of the features does.
HYPER_D_IRREPS = "1x1o"

#: Irreps of the octupolar hyperpolarizability channel: ``3o``, the J=3 part of
#: the Zyss dipolar/octupolar decomposition of beta.
#:
#: This exists because ``beta = alpha (x) d`` alone has a BLIND SPOT at a
#: tetrahedral site.  At Td the vector representation is T2, which contains no
#: A1, so there is no invariant vector and ``d`` vanishes IDENTICALLY (measured
#: 0.000e+00 on an ideal tetrahedral shell).  But a totally symmetric rank-3
#: tensor DOES contain A1 -- the ``beta_xyz`` component -- which is precisely why
#: zincblende crystals are nonlinear optical materials.  Without this channel the
#: model predicts exactly zero second-harmonic response for GaAs, ZnS and ZnSe at
#: their equilibrium geometry: qualitatively wrong, not a small error.
#:
#: The ``3o`` channel is non-zero at Td (measured 7.888 on the same shell) and
#: still vanishes under inversion (4.4e-16), so it closes the blind spot WITHOUT
#: breaking the surface-selectivity rule that motivates beta in the first place.
HYPER_OCTUPOLE_IRREPS = "1x3o"

# Gain on the d readout at initialisation, and the honest reason it is 1.0.
#
# Every other gain in this file is set by an argument: a_mu_init_gain keeps L
# from going indefinite, a_theta_init_gain the same one rank up, and
# quadrupole_coupling_init_gain starts an off-diagonal block of K small because
# a large one can take the coupled Hessian indefinite while every diagonal block
# looks healthy. NONE of those arguments is available here, because d enters no
# energy at all: there is no definiteness to protect, no conditioning to set,
# and no physical scale to match -- the magnitude of d is fixed entirely by the
# units and size of the beta labels it is trained against, and until such labels
# exist it means nothing.
#
# 1.0, i.e. e3nn's own initialisation, left deliberately alone. A smaller gain
# would suggest a scale argument that does not exist, and a zero gain would be
# worse still: it would return beta = 0 for every system, which reads as the
# centrosymmetric selection rule firing when in fact it is an untrained readout
# saying nothing. An arbitrary non-zero d is at least visibly arbitrary.
DEFAULT_HYPER_D_INIT_GAIN = 1.0

# Non-learned attributes that nevertheless decide what the learned parameters
# MEAN. They are persisted in the state dict under ``physical_bounds`` so that
# loading a checkpoint into a differently-constructed module is an error instead
# of a silent reinterpretation; see ElectrostaticHeads._load_from_state_dict.
_PHYSICAL_BOUND_FIELDS = (
    "sigma_min",
    "sigma_span",
    "ell_min",
    "ell_span",
    "kappa_floor",
    "a_mu_floor",
    "a_mu_trace_init",
    "a_mu_init_gain",
    "r_max",
    "rbf_width",
    "key_dim",
)

# Bounds contributed by the optional sectors, appended only when the sector is
# switched on. Appending conditionally rather than always is deliberate: with
# both sectors off the buffer is bit-for-bit the one this module has always
# written, so every checkpoint trained before they existed still loads. A
# checkpoint that *does* carry them cannot be loaded into a module without them
# anyway -- the parameter keys differ and load_state_dict fails first -- so
# nothing is lost by the length changing.
_SPIN_BOUND_FIELDS = ("j_exchange_floor",)
_QUADRUPOLE_BOUND_FIELDS = (
    "a_theta_floor",
    "a_theta_trace_init",
    "a_theta_init_gain",
    "quadrupole_coupling_init_gain",
    # 1.0 if the 4e part of A_Theta was dropped, 0.0 if it is fully represented.
    # A truncated stiffness is a different physical object from a full one, so a
    # checkpoint trained under one must not load into the other.
    "_a_theta_truncated_flag",
)
# The hyperpolarizability sector contributes one bound. It is carried for
# consistency rather than because it constrains anything -- d is unbounded and
# unconstrained -- but the gain still decides what a loaded d MEANS relative to
# the run it was trained in, and the length of the buffer is what tells a
# checkpoint that this sector was on at all.
_HYPER_BOUND_FIELDS = ("hyper_d_init_gain",)
_DISPERSION_BOUND_FIELDS = ("dispersion_electrons_floor", "dispersion_electrons_init")

#: SPEC 13.  Initial effective polarizable-electron count N_i.  Free first-row
#: atoms need 2.3-2.7 to reproduce their C6 in the Slater-Kirkwood relation and
#: hydrogen ~0.8; 2.0 is a neutral start, and N_i is learned per environment.
DEFAULT_DISPERSION_ELECTRONS_INIT = 2.0
#: Strictly positive floor on N_i: N -> 0 is an infinitely heavy oscillator, a
#: dipole that never fluctuates, and the inertia 1/N would diverge.
DEFAULT_DISPERSION_ELECTRONS_FLOOR = 0.05

# Which constructor arguments set each of those, for the error message: two of
# them are derived, so naming the field alone would not tell the caller what to
# change.
_PHYSICAL_BOUND_SOURCE = {
    "sigma_span": "sigma_max - sigma_min",
    "ell_span": "ell_max - ell_min",
    "rbf_width": "r_max / (bond_radial_basis - 1)",
    "_a_theta_truncated_flag": "allow_truncated_a_theta (1.0 means the 4e part "
    "of A_Theta was dropped)",
}

# Relative tolerance for that comparison. Bounds constructed identically match
# bitwise; the tolerance exists only so that a module round-tripped through
# float32 does not trip on 0.1 not being representable. Every mismatch this is
# meant to catch is a factor, not an ulp -- the measured failures were 6.1x in
# sigma and 9.5x in kappa.
_BOUND_MATCH_RTOL = 1e-6


# --- The seam with solver.ElectrostaticParams ------------------------------
#
# solver.py owns ElectrostaticParams; this module owns what is emitted into it.
# The two optional sectors are being developed against SPEC 9 and SPEC 10 rather
# than against each other, so the set of optional fields the dataclass declares
# can lag or lead the set this head produces.
#
# Resolving that by introspection, once, at import, rather than by passing the
# keywords literally: a field this head emits and the dataclass does not declare
# stays reachable on `readout.spin` / `readout.quadrupole` instead of raising
# TypeError from a keyword nobody has heard of, and a field the dataclass
# declares and this head does not emit keeps its own default. The mapping is
# from THIS module's name for a quantity to the dataclass field it feeds, so a
# rename on either side is one line here and not a silent mismatch.
#
# The value side is checked against the real dataclass, so a typo in this table
# fails at import rather than dropping a field on the floor.
_PARAMS_FIELD_TARGETS = {
    "chi_spin": "chi_spin",
    "j_exchange": "j_exchange",
    "kappa_spin": "kappa_spin",
    "A_Theta": "A_Theta",
    "chi_theta": "chi_theta",
    "q_theta": "q_theta",
    "mu_theta": "mu_theta",
}

_DECLARED_PARAMS_FIELDS = frozenset(
    field.name for field in dataclasses.fields(ElectrostaticParams)
)

#: Only the entries ``ElectrostaticParams`` actually declares today.
_OPTIONAL_PARAM_FIELDS = {
    source: target
    for source, target in _PARAMS_FIELD_TARGETS.items()
    if target in _DECLARED_PARAMS_FIELDS
}


class ElectrostaticReadout(NamedTuple):
    """Everything :class:`ElectrostaticHeads` emits for one configuration.

    ``params`` holds the six physical fields consumed by the solver.  ``t_seed``
    is the antisymmetrised attention logit, the warm start for the fixed-k
    conjugate-gradient solve.

    **The seed is part of the definition of the energy.**  It is tempting to
    describe a warm start as an optimisation detail that cannot change the
    answer, and that is true only of a *converged* solve.  This solve is
    deliberately truncated at ``n_cg_iter`` steps (SPEC 2), so the returned
    ``m`` - and therefore ``U`` - is a function of where the iteration started.
    Measured on a 5-atom system at the default ``n_cg_iter = 6``, varying only
    ``t_seed_scale``::

        scale 0.0  ->  U = -0.00648982353 eV
        scale 0.1  ->  U = -0.00633144835 eV      dU/d(scale) ~ 1.6e-03 eV
        scale 5.0  ->  U = +0.14154213857 eV

    At ``n_cg_iter = 200`` the same three seeds agree to 8.7e-19 eV, which is
    the point: the dependence is exactly the residual non-convergence.  Between
    ``scale = 0`` and ``scale = 0.1`` the energy moves by 1.58e-04 eV at k = 6
    and by 8.7e-19 eV at k = 200 - fourteen orders of magnitude apart, so at the
    iteration count this model actually uses the seed is not a rounding detail.

    That is *why* the seed must stay geometry-determined and must never be taken
    from the previous MD step.  A history-dependent seed would not merely slow
    convergence, it would give the same configuration two different energies and
    forfeit conservative forces - the whole point of fixing the iteration count.
    ``t_seed`` is kept separate from ``params`` because it is not a physical
    parameter; it is not, however, physically inert.

    ``spin`` and ``quadrupole`` are ``None`` unless their sector was switched on,
    and what they carry also reaches the solver through ``params``.  ``hyper`` is
    different in kind: it is ``None`` unless ``hyperpolarizability=True``, and
    when it is present it reaches ``params`` **not at all**.  ``d`` is a pure
    readout - see :class:`HyperpolarizabilityReadout` - and this object is the
    only route by which it leaves this module.
    """

    params: ElectrostaticParams
    t_seed: Tensor
    spin: Optional["SpinReadout"] = None
    quadrupole: Optional["QuadrupoleReadout"] = None
    hyper: Optional["HyperpolarizabilityReadout"] = None
    dispersion: Optional["DispersionReadout"] = None


class SpinReadout(NamedTuple):
    """The spin sector of SPEC 9.  ``None`` on :class:`ElectrostaticReadout`
    unless the head was built with ``spin=True``.

    Everything here is in the ``(charge, spin)`` basis that
    :class:`~trace_electrostatics.solver.ElectrostaticParams` consumes, **not**
    the ``(up, down)`` basis SPEC 9 writes the energy in.  That is the basis in
    which the closed-shell limit is exact rather than approximate, and the two
    differ by factors of two that are invisible downstream if they are got wrong;
    see ``ElectrostaticParams`` and
    :func:`~trace_electrostatics.solver.split_spin_potentials`.

    Attributes
    ----------
    chi_spin:
        ``[n_atoms]`` spin field ``chi^s_i = (chi^up_i - chi^down_i) / 2``, eV/e,
        coupling as ``chi_spin . s``.  Unconstrained in sign, and **this is the
        only thing in the model that produces spin polarisation**: ``J^x > 0``
        penalises ``s_i``, so nothing else can.  See the module docstring.
    chi_up, chi_down:
        ``[n_atoms]`` per-spin electronegativities, ``chi + chi_spin`` and
        ``chi - chi_spin``.  Provided as well as ``chi_spin`` because SPEC 9
        writes the model in the ``(up, down)`` basis while the solver consumes
        the ``(charge, spin)`` one, and doing that conversion by hand at each
        call site - with its factor of 1/2 - is how sign conventions get lost.
        ``split_spin_potentials(chi_up, chi_down)`` returns ``(chi, chi_spin)``
        and round-trips these to round-off; when ``chi_spin`` is exactly zero
        both equal ``chi`` bitwise.
    j_exchange:
        ``[n_atoms]`` on-site exchange ``J^x_i``, eV/e^2, the coefficient of
        ``0.5 J^x_i s_i^2``.  **Strictly positive**, by
        ``softplus(raw) + floor``: the spin-mode eigenvalue of the on-site
        Hessian is ``2 J^x`` (SPEC 9.1), so a non-positive value makes the energy
        unbounded below.
    kappa_spin:
        ``[n_bonds]`` spin-channel bond hardness, eV/e^2, the coefficient of
        ``0.5 kappa^s_b (t^s_b)^2`` with ``t^s = t^up - t^down``.  Strictly above
        ``kappa_floor``, and bitwise invariant under exchanging the bond
        endpoints for exactly the reason ``kappa`` is - it is a softplus of a
        symmetrised logit.  When the spin split is exactly zero it equals
        ``params.kappa`` bitwise, which is the closed-shell fixed point.

        Note what is deliberately *not* emitted: a per-channel
        ``kappa^up != kappa^down``.  That would add a ``t^c t^s`` cross term and
        label a bond as passing one spin more easily than the other, which is
        only meaningful against a global quantisation axis this model does not
        carry.  ``ElectrostaticParams`` puts it out of scope; this head does not
        produce it.
    t_seed_spin:
        ``[n_bonds]`` geometry-determined seed for the spin-channel transfer
        ``t^s``, the same antisymmetrised-logit construction as ``t_seed``
        (SPEC 2) on the spin-channel logit.  Goes to
        :class:`~trace_electrostatics.solver.GeometrySeed` as ``t_spin``.  Like
        ``t_seed`` it is part of the *definition* of the energy at finite ``k``,
        not an optimisation detail, and like ``t_seed`` it must never come from
        the previous MD step.
    """

    chi_spin: Tensor
    chi_up: Tensor
    chi_down: Tensor
    j_exchange: Tensor
    kappa_spin: Tensor
    t_seed_spin: Tensor


class QuadrupoleReadout(NamedTuple):
    """The quadrupole sector of SPEC 10.  ``None`` unless blocks were requested.

    A block that was not requested is ``None`` here.  Nothing is ever silently
    zero: a requested block whose irreps the features cannot carry raises
    :class:`QuadrupoleIrrepsError` at construction, long before this object
    exists.

    Everything is expressed in the ``2e`` basis, i.e. a quadrupole is a
    5-component coefficient vector ``theta`` and the cartesian symmetric
    traceless tensor is ``Theta_ab = sum_m theta_m W[m, a, b]`` with ``W`` from
    :func:`quadrupole_cartesian_basis`.  That basis is orthonormal, so the
    5-dimensional forms below are the cartesian ones with no Jacobian to carry.

    Attributes
    ----------
    A_Theta:
        ``[n_atoms, 5, 5]`` quadrupole stiffness, eV, for the reduced quadrupole
        ``Theta_hat = Theta / (e ell^2)``.  Symmetric positive definite by
        construction (``L L^T + floor * I_5``), eigenvalues ``>= a_theta_floor``.
        Transforms as ``D^2 A (D^2)^T``, which is the rank-4 cartesian law
        ``A_abcd -> R_ai R_bj R_ck R_dl A_ijkl``.
    a_theta_truncated:
        ``True`` when the ``4e`` part was dropped because the features do not
        carry it.  Then ``A_Theta`` is an isotropic-plus-lowest-anisotropy
        response, **not** a general quadrupole stiffness.  Carried on the readout
        as well as raised about at construction so that a consumer which never
        saw the constructor can still tell.
    q_theta:
        ``[n_atoms, 5]`` charge-quadrupole coupling coefficient, a ``2e``
        object.  ``None`` unless requested.
    mu_theta:
        ``[n_atoms, 3, 5]`` dipole-quadrupole coupling coefficient, decomposing
        as ``1o + 2o + 3o``.  ``None`` unless requested.

    Neither coupling is constrained: they are off-diagonal blocks of ``K`` and
    have no definiteness of their own.  They can nevertheless destroy the
    definiteness of the *coupled* system while every diagonal block looks
    healthy - SPEC 2.1.3 measured exactly that, +0.58 and +16.45 for two blocks
    with the full coupled Schur complement at -1.05 - so the solver's
    constrained eigenvalue check at the solution is not optional once these are
    switched on.
    """

    A_Theta: Optional[Tensor] = None
    a_theta_truncated: bool = False
    q_theta: Optional[Tensor] = None
    mu_theta: Optional[Tensor] = None
    #: ``[n_atoms, 5]`` quadrupolar electronegativity (SPEC 12), the linear term
    #: ``chi_Theta . Theta_hat``: the permanent quadrupole source.
    chi_theta: Optional[Tensor] = None


class DispersionReadout(NamedTuple):
    """SPEC 13.  ``electrons``: ``[n_atoms]`` effective polarizable electron count.

    ``N_i`` sets the inertia of each induced dipole through the Thomas-Reiche-
    Kuhn sum rule, ``m_i = ell_i^2 / (N_i hbar^2/m_e)``; with the learned
    polarizability it fixes the atom's excitation energy and so its C6.  It
    enters no block of K -- it is read only by the dispersion energy.
    """

    electrons: Tensor


class HyperpolarizabilityReadout(NamedTuple):
    """The ``1o`` direction ``d_i`` from which ``beta_i = alpha_i (x) d_i`` is built.

    ``None`` on :class:`ElectrostaticReadout` unless the head was constructed
    with ``hyperpolarizability=True``.

    Attributes
    ----------
    d:
        ``[n_atoms, 3]`` per-atom ``1o`` vector, in the same cartesian frame as
        everything else this module emits (see the frame note in the module
        docstring - e3nn's degree-1 D matrix is the cartesian rotation itself,
        so no relabelling is correct here).  Unconstrained in magnitude and
        sign.  Pass it to
        :func:`~trace_electrostatics.observables.hyperpolarizability` together
        with a polarizability; that function, not this one, owns the
        18-component / 10-component distinction.

    What ``d`` is, and what it is not
    ---------------------------------
    ``d`` is odd under inversion, which is the entire physical content of the
    channel.  ``beta`` is a rank-3 odd tensor and must vanish at a
    centrosymmetric site; here it does so structurally, because a ``1o`` output
    can only be produced from the ``1o`` block of the features and that block is
    identically zero at such a site (measured ``|d| = 2.9e-17``, against
    ``1.6e-01`` at a polar site of comparable ``alpha``).  Nothing downstream has
    to test for centrosymmetry, and nothing downstream *can* reinstate a beta
    that symmetry forbids.

    The converse does **not** hold, and the module docstring tabulates it: a
    non-centrosymmetric site does not guarantee a non-zero ``d``.  A tetrahedral
    (Td) site admits no invariant vector at all, so ``d`` vanishes there even
    though Td does allow one component of ``beta`` - the ``xyz`` term that makes
    zincblende SHG-active.  ``d = 0`` means "this site admits no polar axis",
    which is not the same statement as "beta vanishes here".

    Nor is it the same statement as "beta vanishes in this crystal".  ``d`` is
    per atom, so what this rule constrains is each site; a centrosymmetric cell
    made of non-centrosymmetric sites has a non-zero ``d_i`` everywhere and a
    vanishing sum, because inversion pairs ``i`` with ``i'`` and ``d_i' = -d_i``.
    The module docstring carries the measured numbers.  Anyone reading a per-atom
    ``beta`` out of a centrosymmetric crystal and expecting zero site by site is
    applying the wrong one of the two rules.

    **It is unidentifiable from anything this model is normally trained on.**
    ``d`` does not enter the energy, the forces, the stress, the charges, the
    dipoles or ``alpha``.  It has no gradient under an energy, force, dipole or
    polarizability loss - not a small one, exactly none - so a model fitted to
    those quantities and then asked for a hyperpolarizability returns the
    initialisation, correctly equivariant and otherwise meaningless.  ``d`` is
    determined by beta training data and by nothing else.  The module docstring
    states this at greater length, the constructor warns about it, and it is
    repeated here because this object is where somebody will first meet a
    number they are tempted to believe.
    """

    d: Tensor


class QuadrupoleIrrepsError(ValueError):
    """A requested quadrupole block cannot be read out from these features.

    Raised at construction, never at call time, and never downgraded to a
    silently zero block: a dead anisotropy is the failure SPEC 3's reachability
    rule exists to prevent, and a dead *quadrupole* block is worse than a dead
    dipole one because the energy it contributes to still looks plausible.

    A subclass of ``ValueError`` so that callers catching the general
    construction error keep working.
    """


class QuadrupoleTruncationWarning(UserWarning):
    """``A_Theta`` was built without its ``4e`` part, on explicit request.

    Issued **at construction**, which is the point of it: SPEC 10 requires the
    restriction be stated when the head is built rather than discovered when a
    quadrupole response comes out wrong.  The resulting stiffness is isotropic
    plus the lowest anisotropy, not a general quadrupole stiffness.
    """


class HyperpolarizabilityIrrepsError(ValueError):
    """These features cannot carry the ``1o`` direction ``d``.

    Raised at construction, like :class:`QuadrupoleIrrepsError` and for the same
    reason: a head whose output block is unreachable returns identically zero for
    the lifetime of the model without warning, and a zero ``beta`` is especially
    treacherous here because zero is also the *correct* answer at a
    centrosymmetric site.  A dead ``d`` and a working ``d`` at an inversion
    centre are indistinguishable downstream, so the difference has to be settled
    when the module is built.

    A subclass of ``ValueError`` so that callers catching the general
    construction error keep working.
    """


class HyperpolarizabilityUnidentifiableWarning(UserWarning):
    """``d`` was switched on; it is trainable only against beta labels.

    Issued **at construction**, which is the point of it.  ``d`` enters no
    energy, force, stress, charge, dipole or polarizability, so an energy- or
    force-trained model has exactly zero gradient on it and the beta it produces
    is its initialisation.  That is not a defect of the parameterisation - it is
    what makes ``beta`` an independent observable rather than Miller's rule in
    disguise - but it does mean a number will come out of
    :func:`~trace_electrostatics.observables.hyperpolarizability` whether or not
    anything ever constrained it, and the warning is the only moment at which
    that can be said before the number exists.

    **This is the only place it can be said, and it covers every path.**
    :func:`~trace_electrostatics.observables.hyperpolarizability` is handed a
    bare ``[n, 3]`` tensor and cannot tell a fitted ``d`` from an unfitted one -
    "never trained" is not a property of a float array - so a call-site check is
    not available, and adding a per-call warning would only add noise to a path
    this one already covers.  It covers it completely: ``d`` cannot be produced
    except by a module constructed with ``hyperpolarizability=True``, and that
    includes the process that merely loads a checkpoint, since the sector has to
    be switched on *before* ``load_state_dict`` will take one (otherwise the
    ``d_linear`` keys are unexpected and loading raises).  So every process that
    can obtain a ``d`` has raised this warning first.  Do not delete it on the
    grounds that the consumer could warn instead; the consumer cannot.

    Suppress it with ``warnings.filterwarnings`` if beta labels are in the loss.
    """


def symmetrise_bond_logits(forward_logit: Tensor, reverse_logit: Tensor) -> Tensor:
    """Return ``0.5 * (s_ij + s_ji)``, the part of the logit kappa may depend on.

    Attention scores are not symmetric - ``Q_i . K_j != Q_j . K_i`` - but bond
    charge transfer is antisymmetric, ``t_ij = -t_ji``, so an asymmetric bond
    hardness makes ``0.5 kappa_ij t_ij^2`` depend on which endpoint happened to
    be listed first, and the energy is ill-defined.

    The result is symmetric to the last bit, not merely to tolerance: IEEE-754
    addition is exactly commutative, so swapping the arguments returns a
    bitwise-identical tensor.
    """
    return 0.5 * (forward_logit + reverse_logit)


def antisymmetrise_bond_logits(forward_logit: Tensor, reverse_logit: Tensor) -> Tensor:
    """Return ``0.5 * (s_ij - s_ji)``, the geometry-determined solver seed.

    This is the complement of :func:`symmetrise_bond_logits`: it carries exactly
    the information that the symmetric bond hardness discards, and it changes
    sign when the bond endpoints are swapped, which is the transformation law
    that ``t_ij`` itself obeys.
    """
    return 0.5 * (forward_logit - reverse_logit)


def bond_hardness(
    forward_logit: Tensor,
    reverse_logit: Tensor,
    kappa_floor: float = DEFAULT_KAPPA_FLOOR,
) -> Tensor:
    """``kappa_ij = softplus(0.5 * (s_ij + s_ji)) + kappa_0``, with ``kappa_0 > 0``.

    Softplus keeps the learned part non-negative and the strictly positive floor
    keeps the SQE Hessian's bond block away from the singular ``kappa = 0`` limit,
    where the block-diagonal part of ``K`` loses rank.  Exact symmetry under
    exchange of the two arguments follows from :func:`symmetrise_bond_logits`.
    """
    if kappa_floor <= 0.0:
        raise ValueError(
            f"kappa_floor must be strictly positive (got {kappa_floor}); kappa -> 0 "
            "destroys the positive definiteness of the SQE Hessian"
        )
    return F.softplus(symmetrise_bond_logits(forward_logit, reverse_logit)) + kappa_floor


def exchange_coupling(
    raw: Tensor, j_exchange_floor: float = DEFAULT_EXCHANGE_FLOOR
) -> Tensor:
    """``J^x_i = softplus(raw_i) + floor``, with ``floor > 0``  (SPEC 9.1).

    The same construction ``eta`` uses, for the same reason.  In the
    ``(q_up, q_down)`` basis the on-site form ``0.5 eta q^2 + 0.5 J^x s^2`` has
    Hessian ``[[eta + J^x, eta - J^x], [eta - J^x, eta + J^x]]``, eigenvalues
    ``2 eta`` and ``2 J^x``; ``J^x <= 0`` therefore makes the on-site block
    indefinite and the energy unbounded below, and the solve does not fail, it
    returns a saddle.

    **The inequality is exact in IEEE-754, not merely true in exact
    arithmetic.**  ``softplus`` returns a non-negative double for every finite
    input - it underflows to exactly ``0.0`` far into the left tail and is the
    identity far into the right one - and correctly-rounded addition is
    monotone, so ``fl(softplus(raw) + floor) >= fl(0 + floor) = floor > 0`` for
    every finite ``raw``.  Measured over ``raw`` spanning ``-1e308 .. +700``
    including the softplus linearisation threshold, the minimum returned value
    is bitwise equal to ``floor``.

    Note what this does *not* buy.  A positive ``J^x`` penalises spin
    polarisation, so it cannot be the source of one; that is ``chi^sigma``'s job
    (module docstring, SPEC 9.1).  And a quadratic-in-``s`` on-site term violates
    the flat-plane condition regardless of sign (SPEC 9.2), so the spin-state
    splitting it produces is fitted, not derived.
    """
    if j_exchange_floor <= 0.0:
        raise ValueError(
            f"j_exchange_floor must be strictly positive (got {j_exchange_floor}); "
            "the spin-mode eigenvalue of the on-site Hessian is 2 J^x, so J^x <= 0 "
            "makes the on-site form indefinite and the energy unbounded below "
            "(SPEC 9.1) -- the same failure eta < J(0) produces, with the same "
            "silent symptom of a solve that returns a saddle"
        )
    return F.softplus(raw) + j_exchange_floor


def _inverse_sigmoid(fraction: float, name: str) -> float:
    """Inverse of ``sigmoid``, used to place an initial value at ``raw = 0``.

    ``name`` is the constructor argument whose value produced ``fraction``, so
    an out-of-range initial value is reported as a named, fixable mistake rather
    than as a bare arithmetic error from three frames down.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            f"{name} must lie strictly inside its bounds; the sigmoid argument "
            f"resolved to {fraction}, which is not in (0, 1)."
        )
    return math.log(fraction / (1.0 - fraction))


def checked_o3_linear(irreps_in, irreps_out, name: str) -> o3.Linear:
    """Build an ``o3.Linear`` and refuse to return a silently dead head.

    A parity- or degree-mismatched ``o3.Linear`` is constructed happily by e3nn
    with zero instructions and zero parameters.  It then returns identically zero
    for the lifetime of the model, with no warning, so a mis-declared head trains
    to nothing while every loss curve and shape check looks healthy.

    ``SPEC.md`` requires ``len(lin.instructions) > 0``.  That is necessary but not
    sufficient: a head declaring ``0e + 2e`` outputs from a ``0e``-only input has
    one instruction and still returns an identically zero ``2e`` block, so every
    output irrep is additionally required to be *reachable*.

    Reachability is tested with e3nn's own ``output_mask`` and not by collecting
    the ``i_out`` of each instruction.  The instruction list has a hole:
    a zero-multiplicity input irrep still produces an instruction, with
    ``path_shape = (0, 1)`` and no weights, so its ``i_out`` appears "reached"
    while the block it feeds is identically zero.  Measured::

        checked_o3_linear("64x0e + 0x2e", "1x0e + 1x2e")   # accepted, wrongly
        instructions -> [(0, 0, (64, 1)), (1, 1, (0, 1))]  # i_out 1 "reached"
        output_mask  -> [1, 0, 0, 0, 0, 0]                 # ... but dead

    and a healthy head over ``64x0e + 32x1o + 16x2e`` gives an all-ones mask.
    ``ElectrostaticHeads(irreps_in="64x0e + 32x1o + 0x2e")`` then built happily
    and returned an ``A_mu`` isotropic to 8.9e-16 - the dead anisotropy block
    SPEC 3 requires be rejected.  Summing the mask over each output slice has no
    such hole, because it is derived from the weight paths that actually exist.

    Both checks raise rather than assert, because ``python -O`` strips asserts
    and would reinstate exactly the bug this guards against.
    """
    irreps_in = o3.Irreps(irreps_in)
    irreps_out = o3.Irreps(irreps_out)
    lin = o3.Linear(irreps_in, irreps_out)

    if len(lin.instructions) == 0:
        raise ValueError(
            f"o3.Linear head {name!r} has zero instructions: none of "
            f"irreps_out={irreps_out} can be produced from irreps_in={irreps_in} "
            "(a parity mismatch is the usual cause). Such a head has no "
            "parameters and returns identically zero without warning."
        )

    # Pass the module we already built: unreachable_output_irreps would otherwise
    # construct a second one, and an o3.Linear draws its weights from the global
    # RNG. See that function for why a probe that shifts the stream is a bug.
    unreachable = unreachable_output_irreps(irreps_in, irreps_out, linear=lin)
    if unreachable:
        raise ValueError(
            f"o3.Linear head {name!r} cannot produce {', '.join(unreachable)} from "
            f"irreps_in={irreps_in}. Those output blocks would be identically zero "
            "for the lifetime of the model. Check both the parity and the "
            "MULTIPLICITY of the matching input irrep: a 0x2e input contributes a "
            "weightless instruction that looks like a path and carries nothing."
        )
    return lin


def unreachable_output_irreps(
    irreps_in, irreps_out, linear: Optional[o3.Linear] = None
) -> list[str]:
    """Which of ``irreps_out`` an ``o3.Linear`` from ``irreps_in`` cannot produce.

    The reachability half of :func:`checked_o3_linear`, split out so that it can
    be asked as a *question* rather than only enforced as a raise.  The
    quadrupole sector needs the question: SPEC 10 says "detect, do not
    hard-exclude", so it has to be able to establish which blocks these features
    support before deciding what to build and what to refuse.

    Returns the offending irreps as strings, empty when every output block is
    reachable.  The test is e3nn's ``output_mask`` for the reason documented at
    length in :func:`checked_o3_linear`: counting instructions, or collecting
    their ``i_out``, misses a zero-multiplicity input, which produces an
    instruction with ``path_shape = (0, 1)``, no weights, and a block that is
    identically zero while looking reached.

    **The probe must not touch the global RNG.**  ``o3.Linear.__init__`` draws
    its weights from it, so a throwaway module built here would advance the
    stream and every parameter initialised afterwards would depend on how many
    blocks happened to be probed - which is how asking a question changes the
    answer.  Measured: routing :func:`checked_o3_linear`'s own check through a
    second construction left every state-dict entry bitwise identical (the
    probe's weights are discarded) while shifting every *downstream* draw, so
    the module looked unchanged and the model around it was not.  The probe is
    therefore built inside ``torch.random.fork_rng``, and callers that already
    hold the module pass it as ``linear`` and skip the construction entirely.
    """
    irreps_in = o3.Irreps(irreps_in)
    irreps_out = o3.Irreps(irreps_out)
    if linear is None:
        with torch.random.fork_rng(devices=[]):
            output_mask = o3.Linear(irreps_in, irreps_out).output_mask
    else:
        output_mask = linear.output_mask
    return [
        str(irreps_out[i])
        for i, block in enumerate(irreps_out.slices())
        if float(output_mask[block].sum()) == 0.0
    ]


def _symmetric_tensor_basis() -> tuple[Tensor, Tensor]:
    """Cartesian bases for the ``0e`` and ``2e`` parts of a symmetric 3x3 tensor.

    A symmetric rank-2 cartesian tensor decomposes as ``0e + 2e``: an isotropic
    trace and a traceless anisotropy.  The ``2e`` basis is the Wigner 3j symbol
    ``W[m, a, b] = <2 m | 1 a, 1 b>``, whose defining invariance
    ``W = D^2 D^1 D^1 W`` is exactly the statement that contracting a ``2e``
    coefficient vector ``c`` with it produces ``M -> D^1 M (D^1)^T`` whenever
    ``c -> D^2 c``.  Since ``D^1`` from ``D_from_matrix`` is the cartesian
    rotation matrix itself (see the frame note in the module docstring), that is
    the rank-2 cartesian transformation law, with no relabelling of axes.

    Taking the basis from e3nn's own Clebsch-Gordan table rather than writing one
    out by hand means the component ordering and phase conventions cannot drift
    out of sync with the rest of the network.  Both bases are built in float64
    regardless of the default dtype, so the basis itself contributes no error.

    The two bases are deliberately *not* on the same scale, and the difference is
    accounted for where it matters rather than normalised away here.  The
    identity carries ``||I||_F = sqrt(3) = 1.732`` because a unit ``0e``
    coefficient has to mean exactly ``L = I``; each of the five ``2e`` slices is
    scaled to unit Frobenius norm, so unit ``2e`` coefficients contribute
    ``sqrt(5) = 2.236``.  Unit-scale readout coefficients therefore give an
    anisotropy 1.29x the isotropic part, not an equal one, which is why the
    ``A_mu`` readout is initialised with a gain rather than at unit scale (see
    ``DEFAULT_A_MU_INIT_GAIN``).
    """
    # wigner_3j(0, 1, 1) is delta_ab / sqrt(3); use the plain identity instead so
    # that a unit 0e coefficient means exactly L = I.
    identity = torch.eye(3, dtype=torch.float64)

    # Each of the five 2e slices has Frobenius norm 1/sqrt(5); the common factor
    # sqrt(5) makes them individually unit-norm and mutually orthogonal. A single
    # overall factor preserves equivariance.
    traceless = o3.wigner_3j(2, 1, 1, dtype=torch.float64) * math.sqrt(5.0)
    return identity, traceless.contiguous()


# quadrupole_cartesian_basis is defined in kernels.py (the home of the
# quadrupole convention) so the solver can use it without importing this
# module; it is re-exported here unchanged.
def _quadrupole_stiffness_basis(
    include_4e: bool,
) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    """Bases for the ``0e``, ``2e`` and ``4e`` parts of a symmetric 5x5 tensor.

    ``A_Theta`` is a bilinear form on the five-dimensional ``2e`` space, so it is
    a symmetric 5x5 matrix, and the space of those decomposes as the symmetric
    part of ``2e (x) 2e``.  ``2e (x) 2e = 0e + 1e + 2e + 3e + 4e``; the ``1e`` and
    ``3e`` Clebsch-Gordan tensors are antisymmetric in the two ``2e`` indices, so
    a symmetric form has no component along them and the answer is
    ``0e + 2e + 4e``.  Dimensions: ``5 * 6 / 2 = 15 = 1 + 5 + 9``.

    Verified against e3nn 0.4.4 rather than asserted.  Splitting each
    ``wigner_3j(l, 2, 2)`` into symmetric and antisymmetric parts in its last two
    indices gives, for ``l = 0, 2, 4``, ``|sym| = 1.000000`` and
    ``|asym| <= 6.56e-16``; for ``l = 1, 3``, ``|asym| = 1.000000`` and
    ``|sym| <= 4.99e-16``.  The fifteen normalised basis matrices below have
    numerical rank 15 in the 25-dimensional space of 5x5 matrices, and each is
    symmetric to ``1.23e-15``.

    Normalisation follows :func:`_symmetric_tensor_basis` exactly: the ``0e``
    slice is the plain identity, so a unit ``0e`` coefficient means ``L = I_5``,
    and the ``2e`` and ``4e`` slices are scaled to unit Frobenius norm
    (``wigner_3j`` gives them ``1/sqrt(5)`` and ``1/3``).  The two scales are
    again not equal, and again that is accounted for in the init gain rather than
    normalised away here: see ``DEFAULT_A_THETA_INIT_GAIN``.  Each anisotropy
    slice is orthogonal to the identity to 8.9e-16, so the ``0e`` channel really
    is the trace and the others really are traceless.

    ``include_4e=False`` returns the truncated basis SPEC 10 permits at
    ``l_max = 2``.  It is still a valid positive-definite parameterisation - it is
    just not a *general* quadrupole stiffness, and the caller has been told so.
    """
    identity = torch.eye(_QUADRUPOLE_DIM, dtype=torch.float64)
    rank_2 = o3.wigner_3j(2, 2, 2, dtype=torch.float64) * math.sqrt(5.0)
    rank_4 = (
        (o3.wigner_3j(4, 2, 2, dtype=torch.float64) * 3.0).contiguous()
        if include_4e
        else None
    )
    return identity, rank_2.contiguous(), rank_4


def _mu_theta_basis() -> tuple[Tensor, Tensor, Tensor]:
    """Bases for the ``1o``, ``2o`` and ``3o`` parts of a 3x5 coupling matrix.

    The dipole-quadrupole energy is ``mu^T C Theta`` with ``mu`` a ``1o`` vector
    and ``Theta`` a ``2e`` coefficient vector, so ``C`` is a general 3x5 matrix
    and the space of those is ``1o (x) 2e = 1o + 2o + 3o`` (3 + 5 + 7 = 15,
    numerical rank 15 as a set of 3x5 matrices).  The parity is the point: the
    product of an odd and an even object is odd, so the *coefficient* has to be
    odd for the energy to be a true scalar.  A network with natural parity
    ``(-1)^l`` has no ``2o`` block at any ``l_max``, which is why this module
    tests irreps rather than ``l_max``.

    Each ``wigner_3j(l, 1, 2)`` has uniform slice norms (``1/sqrt(3)``,
    ``1/sqrt(5)``, ``1/sqrt(7)``, uniform to 8.3e-16), rescaled here to unit
    Frobenius norm per slice for the same reason as everywhere else in this file.
    """
    return tuple(  # type: ignore[return-value]
        (o3.wigner_3j(l, 1, 2, dtype=torch.float64) * math.sqrt(2 * l + 1)).contiguous()
        for l in (1, 2, 3)
    )


def _octupole_cartesian_basis(dtype: torch.dtype, device: torch.device) -> Tensor:
    """``[7, 3, 3, 3]`` basis mapping a ``3o`` irrep to a cartesian rank-3 tensor.

    The image is the space of **totally symmetric, traceless** rank-3 tensors,
    which has exactly 7 independent components -- the J=3 (octupolar) part of the
    Zyss decomposition of beta.

    Built by chaining Wigner 3j symbols ``1 (x) 1 -> 2`` and ``2 (x) 1 -> 3``,
    then symmetrising over the three cartesian indices and removing the trace.
    Verified: symmetric to 4.2e-17, traceless to 6.9e-18, rank 7 of 7, l=3
    equivariant to 3.4e-16 over random proper rotations, and ODD under inversion
    to 6.9e-18 -- that last is what keeps the surface-selectivity rule intact.
    """
    w211 = o3.wigner_3j(2, 1, 1, dtype=torch.float64).to(dtype=dtype, device=device)
    w312 = o3.wigner_3j(3, 1, 2, dtype=torch.float64).to(dtype=dtype, device=device)
    basis = torch.einsum("mck,kab->mabc", w312, w211)
    basis = _symmetrise_rank3(basis)
    return basis - _dipolar_rank3(basis)


def _symmetrise_rank3(tensor: Tensor) -> Tensor:
    """Totally symmetric part of a ``[3, 3, 3]`` cartesian tensor."""
    rank = tensor.dim()
    lead = list(range(rank - 3))
    i, j, k = rank - 3, rank - 2, rank - 1
    orders = ((i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i))
    total = tensor.new_zeros(tensor.shape)
    for order in orders:
        total = total + tensor.permute(*lead, *order)
    return total / 6.0


def _dipolar_rank3(tensor: Tensor) -> Tensor:
    """J=1 part of a symmetric rank-3 tensor, reconstructed from its trace vector.

    With ``v_a = S_abb``, the dipolar piece is
    ``(1/5)(delta_ab v_c + delta_ac v_b + delta_bc v_a)``.  The ``1/5`` is fixed
    by requiring the reconstruction to carry the whole trace: verified, its own
    trace reproduces ``v`` to 4.4e-16 and the remainder is traceless to 5.0e-16.
    """
    eye = torch.eye(3, dtype=tensor.dtype, device=tensor.device)
    v = torch.einsum("...abb->...a", tensor)
    return (
        torch.einsum("ab,...c->...abc", eye, v)
        + torch.einsum("ac,...b->...abc", eye, v)
        + torch.einsum("bc,...a->...abc", eye, v)
    ) / 5.0


class SolverSeedPotential(nn.Module):
    """Per-atom seed potential ``u_hat`` for the solver's starting point (SPEC 2.0.1).

    Why a potential, not a bond charge
    ----------------------------------
    At the solution, stationarity of the bond block gives EXACTLY
    ``tau* = -kappa^-1 F B^T u*`` with ``u* = dU/dq`` the potential on each atom
    (verified to 1.9e-16 on a 64-atom solve): the optimal split charge is Ohm's
    law, bond conductance times the potential difference across it. So the seed
    predicts one number per atom and the physics supplies kappa, the envelope F
    and the incidence B exactly -- N outputs instead of n_bonds (~10x fewer), in
    the functional form the minimiser actually has.

    Why its own parameters
    ----------------------
    The previous seed was the antisymmetric half of the SAME attention logit
    whose symmetric half is kappa, so any loss on the seed moved kappa -- the
    physics. It was also untrained, and measured ~10x WORSE than starting CG from
    zero (relative k = 6 error 3.4e-3 - 7.3e-3 against 1.9e-4 - 8.4e-4). This
    module owns every parameter it uses, is zero-initialised (an untrained seed
    is exactly zero, CG's natural origin), and is trained only by
    :func:`trace_electrostatics.energy.solver_seed_loss` -- the variational
    energy after k steps with the physics detached -- never by the data loss.
    """

    def __init__(self, invariant_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(invariant_dim)
        self.net = nn.Sequential(
            nn.Linear(invariant_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # u = w * chi + net(h): chi is the zeroth-order potential, so the net
        # learns the screening correction rather than the whole potential.
        self.chi_weight = nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, invariants: Tensor, chi: Tensor) -> Tensor:
        return self.chi_weight * chi + self.net(self.norm(invariants)).squeeze(-1)


class DispersionElectrons(nn.Module):
    """``N_i = floor + softplus(net(h_i))``, the effective polarizable electron count.

    SPEC 13.  Starts at exactly ``init`` for every atom (zero-initialised last
    weight, bias at the inverse softplus), so an untrained head is a uniform
    Slater-Kirkwood model on top of the learned polarizabilities, and learns its
    environment dependence from the data.  Positive by construction; the floor
    keeps the inertia ``1/N`` finite.
    """

    def __init__(self, invariant_dim: int, hidden_dim: int, init: float, floor: float) -> None:
        super().__init__()
        if not floor > 0.0:
            raise ValueError(f"dispersion_electrons_floor must be > 0, got {floor}")
        if not init > floor:
            raise ValueError(
                f"dispersion_electrons_init={init} must exceed the floor {floor}"
            )
        self.floor = float(floor)
        self.norm = nn.LayerNorm(invariant_dim)
        self.net = nn.Sequential(
            nn.Linear(invariant_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            # softplus^-1(y) = y + log(1 - exp(-y)), exact for y > 0.
            excess = float(init) - self.floor
            self.net[-1].bias.fill_(excess + math.log(-math.expm1(-excess)))

    def forward(self, invariants: Tensor) -> Tensor:
        return self.floor + F.softplus(self.net(self.norm(invariants)).squeeze(-1))


class ElectrostaticHeads(nn.Module):
    """Read charge-equilibration parameters off TRACE node features.

    Scalar fields are predicted from the rotation- and inversion-invariant part
    of the node features - the ``0e`` block together with the squared norms of
    every higher-l block, which is the same invariant construction TRACE's own
    energy readout uses - so ``chi``, ``eta``, ``sigma`` and ``ell`` are true
    scalars rather than pseudoscalars.  The dipole stiffness needs genuine
    directional information and comes from an equivariant ``o3.Linear``.

    Parameters
    ----------
    irreps_in:
        Node feature irreps, ``64x0e + 32x1o + 16x2e`` for TRACE v3 at
        ``hidden_dim = 64``.  The ``0e`` block must come first, and a ``2e``
        block must be present: without one the anisotropic part of ``A_mu``
        would be identically zero, so construction raises rather than quietly
        producing an isotropic-only dipole stiffness.
    hidden_dim:
        Width of the invariant MLP that produces the scalar fields.
    key_dim:
        Width of the query/key projections whose inner product forms the raw
        bond logit.
    sigma_min, sigma_max:
        Bounds for the Gaussian charge width, in angstrom.  ``sigma`` is emitted
        through a sigmoid, so it is bounded on both sides; the upper bound also
        bounds the hardness floor ``J(0)`` from below, keeping ``eta`` finite and
        the solve well conditioned.
    ell_min, ell_max, ell_init:
        Bounds and initial value of the moment length scale, in angstrom.
        ``ell`` is emitted through a sigmoid, so it is bounded on both sides for
        the same reason ``sigma`` is: the induced polarizability
        ``alpha = K_E ell^2 / A_mu`` is only capped by ``a_mu_floor`` if ``ell``
        is capped too.  At the defaults the hard ceiling is
        ``K_E * 2.0^2 / a_mu_floor = 576 A^3``, an order of magnitude above
        caesium's 59.4 A^3.
    kappa_floor:
        Strictly positive bond-hardness floor.  This is the *only* place the
        floor is set; ``ElectrostaticConfig`` deliberately carries no copy of it.
    a_mu_floor:
        Strictly positive eigenvalue floor of the dipole stiffness, in eV.  With
        ``ell_max`` it fixes the polarizability ceiling; see
        :func:`polarizability_ceiling_from_a_mu_floor`.
    a_mu_trace_init:
        Initial isotropic part of ``L``, so ``A_mu`` starts near
        ``(a_mu_trace_init**2 + a_mu_floor) * I``, i.e. at a polarizability of
        ``K_E ell^2 / (a_mu_trace_init**2 + a_mu_floor)``.  Set it from the
        polarizability you want with
        :func:`a_mu_trace_init_from_polarizability` rather than by hand; the
        default puts every atom at 1.5 A^3, between H (0.67) and C (1.76).
    a_mu_init_gain:
        Gain on the ``A_mu`` readout weights at initialisation.  Without it the
        ``2e`` anisotropy swamps that isotropic offset and ``A_mu`` starts soft
        and ill-conditioned; see ``DEFAULT_A_MU_INIT_GAIN`` for the measurement.
    t_seed_scale:
        Initial gain applied to the antisymmetrised logit that seeds the solver.
        Not an optimisation detail - at a fixed iteration count it changes the
        energy; see :class:`ElectrostaticReadout`.
    bond_radial_basis:
        Number of Gaussian radial features used to let ``kappa`` depend on bond
        length.  A function of ``|r_ij|`` alone is automatically symmetric under
        exchange of the endpoints, so it cannot break the symmetry constraint.
        Set to 0 to make ``kappa`` depend on the node features only; negative
        values are rejected rather than silently treated as 0.
    r_max:
        Range of that radial basis, in angstrom.
    require_float64:
        Refuse to construct unless ``torch.get_default_dtype()`` is float64.
        See Notes; set it to ``False`` only for a deliberate single-precision
        experiment.
    spin:
        Switch on the SPEC 9 spin sector.  **Default off**, and off means the
        head is bit-for-bit the closed-shell one: no extra module is
        constructed, so the RNG stream is unchanged, and ``forward`` skips the
        branch entirely.  On, it additionally emits ``chi_spin``, ``j_exchange``
        and ``kappa_spin`` into ``params``, and the full
        :class:`SpinReadout` on the readout.
    j_exchange_floor:
        Strictly positive floor on ``J^x``, eV/e^2.  Defaults to
        ``solver.DEFAULT_EXCHANGE_FLOOR`` (1e-3), which is four orders of
        magnitude below any exchange splitting a fit would target: it is there
        to keep the spin mode definite (SPEC 9.1), not to model exchange.
    spin_init_gain:
        Gain on the spin heads at initialisation, so the sector starts as a
        small perturbation of the closed-shell model rather than as an equal
        partner.
    quadrupole_blocks:
        Which SPEC 10 quadrupole blocks to build, a subsequence of
        :data:`QUADRUPOLE_BLOCKS`.  ``None`` (the default) means the sector is
        off.  Each requested block is checked against ``irreps_in`` **at
        construction** and raises :class:`QuadrupoleIrrepsError`, naming the
        block and the unreachable irreps, if the features cannot carry it -
        never a silently dead block.  An unknown name is an error, not a no-op.
    allow_truncated_a_theta:
        Permit ``A_Theta`` to be built from ``0e + 2e`` when the features carry
        no ``4e``.  Off by default, because the result is an
        isotropic-plus-lowest-anisotropy response and **not** a general
        quadrupole stiffness; on, it emits a
        :class:`QuadrupoleTruncationWarning` at construction and records the
        choice in ``physical_bounds`` so a checkpoint cannot be reinterpreted as
        a full one.  This is the ``l_max = 2`` route SPEC 10 allows, and the
        restriction is stated when the head is built, never discovered later.
    a_theta_floor:
        Strictly positive eigenvalue floor of ``A_Theta``, eV.  With ``ell_max``
        it fixes the quadrupole-polarizability ceiling; see
        :func:`quadrupole_ceiling_from_a_theta_floor`.  Note the ``ell^4``.
    a_theta_trace_init, a_theta_init_gain:
        Initial isotropic part of ``L`` and the gain on the ``A_Theta`` readout
        weights, exactly as ``a_mu_trace_init`` and ``a_mu_init_gain`` are for
        the dipole block.  Set the first from the quadrupole polarizability you
        want with :func:`a_theta_trace_init_from_quadrupole_polarizability`.
    quadrupole_coupling_init_gain:
        Gain on the ``chi_theta``, ``q-Theta`` and ``mu-Theta`` readouts.  These are
        off-diagonal blocks of ``K`` with no definiteness of their own and are
        started small deliberately; see
        ``DEFAULT_QUADRUPOLE_COUPLING_INIT_GAIN``.
    hyperpolarizability:
        Switch on the ``1o`` direction ``d_i`` from which
        :func:`~trace_electrostatics.observables.hyperpolarizability` builds
        ``beta_i = alpha_i (x) d_i``.  **Default off**, and off means bit-for-bit
        the head without it: the readout is constructed last, so the RNG stream
        is untouched, and ``forward`` skips the branch.  On, it emits
        :class:`HyperpolarizabilityReadout` on the readout and **nothing into**
        ``params`` - ``d`` is not a parameter of the energy and must never
        become one.  Raises :class:`HyperpolarizabilityIrrepsError` if the
        features cannot carry ``1o``, and warns
        (:class:`HyperpolarizabilityUnidentifiableWarning`) that ``d`` has no
        gradient under any energy, force, dipole or polarizability loss: it is
        trainable against beta labels and against nothing else.
    dispersion, dispersion_electrons_init, dispersion_electrons_floor:
        SPEC 13.  Switch on the per-atom effective electron count ``N_i``
        (:class:`DispersionElectrons`) that the dispersion energy needs.
        **Default off**, built last, so the RNG stream is untouched.  It enters
        no block of K; it only sets the inertia of the fluctuating dipoles.
    hyper_d_init_gain:
        Gain on the ``d`` readout weights at initialisation.  Unlike every other
        gain here it rests on no definiteness or conditioning argument, because
        ``d`` enters no energy; see ``DEFAULT_HYPER_D_INIT_GAIN`` for why it is
        left at 1.0 and why 0.0 would be actively misleading.

    Notes
    -----
    **Construction time is the deadline for the dtype.**  e3nn bakes the default
    dtype into the generated tensor-product code of the first module built, so a
    later ``.double()`` is a silent no-op on that kernel and costs seven orders
    of magnitude of equivariance (measured on ``A_mu``: 6.16e-09 against
    3.17e-16).  ``ElectrostaticConfig.validate_dtype`` runs at *attach* time,
    which is far too late - it passes cleanly on a module that was already
    ruined - so the check is repeated here, before the first ``o3.Linear``.

    **The physical bounds are part of the state dict.**  ``sigma_min``,
    ``ell_min``, ``kappa_floor``, ``a_mu_floor`` and friends are not learned, so
    they used to live only as Python attributes and a checkpoint said nothing
    about them.  Loading such a checkpoint into a module constructed with
    different bounds reported "All keys matched successfully" while sigma moved
    1.588 -> 0.259 A and kappa 1.046 -> 9.946 eV - silently discarding the
    positive-definiteness and hardness-floor guarantees the bounds encode.  They
    are now carried in the ``physical_bounds`` buffer and a mismatch on load is
    an error that names the offending bound.
    """

    def __init__(
        self,
        irreps_in=TRACE_NODE_IRREPS,
        hidden_dim: int = 64,
        key_dim: int = 32,
        sigma_min: float = DEFAULT_SIGMA_MIN,
        sigma_max: float = DEFAULT_SIGMA_MAX,
        ell_min: float = DEFAULT_ELL_MIN,
        ell_max: float = DEFAULT_ELL_MAX,
        ell_init: float = DEFAULT_ELL_INIT,
        kappa_floor: float = DEFAULT_KAPPA_FLOOR,
        a_mu_floor: float = DEFAULT_A_MU_FLOOR,
        a_mu_trace_init: float = DEFAULT_A_MU_TRACE_INIT,
        a_mu_init_gain: float = DEFAULT_A_MU_INIT_GAIN,
        t_seed_scale: float = DEFAULT_T_SEED_SCALE,
        bond_radial_basis: int = 8,
        r_max: float = 5.0,
        require_float64: bool = True,
        spin: bool = False,
        j_exchange_floor: float = DEFAULT_EXCHANGE_FLOOR,
        spin_init_gain: float = DEFAULT_SPIN_INIT_GAIN,
        quadrupole_blocks: Optional[Sequence[str]] = None,
        allow_truncated_a_theta: bool = False,
        a_theta_floor: float = DEFAULT_A_THETA_FLOOR,
        a_theta_trace_init: float = DEFAULT_A_THETA_TRACE_INIT,
        a_theta_init_gain: float = DEFAULT_A_THETA_INIT_GAIN,
        quadrupole_coupling_init_gain: float = DEFAULT_QUADRUPOLE_COUPLING_INIT_GAIN,
        hyperpolarizability: bool = False,
        hyper_d_init_gain: float = DEFAULT_HYPER_D_INIT_GAIN,
        dispersion: bool = False,
        dispersion_electrons_init: float = DEFAULT_DISPERSION_ELECTRONS_INIT,
        dispersion_electrons_floor: float = DEFAULT_DISPERSION_ELECTRONS_FLOOR,
        hardness_margin: float = DEFAULT_HARDNESS_MARGIN,
        envelope_onset: Optional[float] = DEFAULT_ENVELOPE_ONSET,
        charge_transfer_decay: Optional[float] = None,
        charge_transfer_jump: float = 1.0e3,
        charge_transfer_onset: float = 0.5,
        charge_transfer_width: float = 0.1,
        bond_hardness_law: str = "free",
        bond_hardness_range: float = 10.0,
    ) -> None:
        super().__init__()
        #: Bond-hardness law (docs/ELECTROSTATICS_AUDIT_2026-10-09.md, step 1): "free" (default:
        #: kappa = softplus(s) + kappa_floor, every earlier model bit for bit) or "two_level":
        #: kappa = kappa_anchor(r, Z_i, Z_j) exp(L tanh(s / L)) + kappa_floor, L = ln(range),
        #: with kappa_anchor = (Delta chi^2 + 4 h^2)^(3/2) / (4 h^2) (bond_law.py).  Unidentified
        #: bonds then keep a physical hardness instead of their initial value.
        if bond_hardness_law not in ("free", "two_level"):
            raise ValueError(f"bond_hardness_law must be 'free' or 'two_level', got {bond_hardness_law!r}")
        if not float(bond_hardness_range) > 1.0:
            raise ValueError(f"bond_hardness_range must be > 1, got {bond_hardness_range}")
        self.bond_hardness_law = bond_hardness_law
        self.bond_hardness_range = float(bond_hardness_range)
        #: SPEC 23: optional contact law of the bond hardness,
        #: kappa *= exp(ln(K) sigmoid(x) + beta w softplus(x)),
        #: x = (r - R_i - R_j - delta) / w, R the covalent radii.  None
        #: (default): the bond hardness of every earlier model, bit for bit.
        if charge_transfer_decay is not None and not float(charge_transfer_decay) > 0.0:
            raise ValueError(
                f"charge_transfer_decay must be > 0 (1/A) or None, got {charge_transfer_decay}"
            )
        if not float(charge_transfer_jump) >= 1.0:
            raise ValueError(f"charge_transfer_jump must be >= 1, got {charge_transfer_jump}")
        if not float(charge_transfer_onset) >= 0.0 or not float(charge_transfer_width) > 0.0:
            raise ValueError(
                "charge_transfer_onset must be >= 0 A and charge_transfer_width > 0 A"
            )
        self.charge_transfer_decay = (
            None if charge_transfer_decay is None else float(charge_transfer_decay)
        )
        self.charge_transfer_jump = float(charge_transfer_jump)
        self.charge_transfer_onset = float(charge_transfer_onset)
        self.charge_transfer_width = float(charge_transfer_width)
        #: SPEC 17.3: eta = softplus(raw) + S_0(sigma) + hardness_margin, or
        #: "legacy": eta = softplus(raw) + sqrt(2) S_0(sigma), the old floor.
        #: The tight floor is safe under the GAUSSIAN kernel, whose coupled K is
        #: a Gram form plus the on-site residuals. The Thole charge-dipole
        #: coupling is not, and relied on the old floor's slack: a 14-atom
        #: cluster measured min eig K = -1.0 at a 1 eV margin (both diagonal
        #: blocks positive definite) and +0.69 under the legacy floor. Use
        #: "legacy" (or a margin of several eV) with multipole_kernel="thole" --
        #: necessary, not sufficient: at 0.14 A contact Thole was still
        #: indefinite under it. Only the Gaussian kernel is bounded by theorem.
        if isinstance(hardness_margin, str):
            if hardness_margin != "legacy":
                raise ValueError(
                    f"hardness_margin must be a float >= 0 or 'legacy', got {hardness_margin!r}"
                )
            self.hardness_margin = "legacy"
        else:
            if not float(hardness_margin) >= 0.0:
                raise ValueError(f"hardness_margin must be >= 0 eV, got {hardness_margin}")
            self.hardness_margin = float(hardness_margin)
        if envelope_onset is not None and not 0.0 <= float(envelope_onset) < 1.0:
            raise ValueError(
                f"envelope_onset must lie in [0, 1) (a fraction of r_max) or be None "
                f"for the legacy full-range cosine, got {envelope_onset}"
            )
        #: SPEC 17.4: the dissociation envelope is 1 below envelope_onset * r_max
        #: and switches off smoothly (C2) between there and r_max.
        self.envelope_onset = None if envelope_onset is None else float(envelope_onset)

        # FIRST, before any e3nn module exists. e3nn generates and caches its
        # TensorProduct code under the default dtype in force at construction, so
        # by the time ElectrostaticConfig.validate_dtype() runs -- at attach time,
        # strictly later -- the kernels are already baked and it passes cleanly on
        # a module that is quietly wrong at 1e-8. This is the last line at which
        # the mistake is still fixable.
        if require_float64 and torch.get_default_dtype() is not torch.float64:
            raise RuntimeError(
                "ElectrostaticHeads must be constructed with float64 as the "
                "default dtype: call torch.set_default_dtype(torch.float64) "
                "BEFORE this constructor, not after. e3nn caches generated "
                "TensorProduct code under the default dtype at first "
                "construction, so a later .double() is a silent no-op for that "
                "kernel -- measured on A_mu, 6.16e-09 relative equivariance "
                "error against 3.17e-16 for the same module built under float64. "
                "Pass require_float64=False only for a deliberate "
                "single-precision experiment. Current default dtype: "
                f"{torch.get_default_dtype()}."
            )

        self.irreps_in = o3.Irreps(irreps_in)
        if self.irreps_in[0].ir != o3.Irrep("0e"):
            raise ValueError(
                f"irreps_in must begin with a 0e block, got {self.irreps_in}; the "
                "scalar heads read that block directly"
            )
        if not sigma_max > sigma_min > 0.0:
            raise ValueError(
                f"require 0 < sigma_min < sigma_max, got {sigma_min}, {sigma_max}"
            )
        if not ell_max > ell_init > ell_min > 0.0:
            raise ValueError(
                f"require 0 < ell_min < ell_init < ell_max, got {ell_min}, "
                f"{ell_init}, {ell_max}; ell is emitted through a sigmoid so that "
                "the polarizability ceiling K_E * ell_max^2 / a_mu_floor is a real "
                "bound, which means ell_init has to lie strictly inside the bounds"
            )
        if kappa_floor <= 0.0:
            raise ValueError(
                f"kappa_floor must be strictly positive, got {kappa_floor}; a "
                "vanishing bond hardness makes the SQE Hessian singular"
            )
        if a_mu_floor <= 0.0:
            raise ValueError(
                f"a_mu_floor must be strictly positive, got {a_mu_floor}; it is "
                "what makes A_mu positive *definite* rather than semidefinite"
            )
        if a_mu_init_gain <= 0.0:
            raise ValueError(
                f"a_mu_init_gain must be strictly positive, got {a_mu_init_gain}; "
                "zero would start A_mu exactly isotropic with no symmetry to break"
            )
        if int(bond_radial_basis) < 0:
            raise ValueError(
                f"bond_radial_basis must be >= 0, got {bond_radial_basis}. A "
                "negative value used to be accepted and silently treated as 0, "
                "which drops the radial term from kappa without a word: kappa then "
                "loses its length dependence entirely. Pass 0 if that is what you "
                "want."
            )

        self.scalar_dim = int(self.irreps_in[0].mul)
        self.hidden_dim = int(hidden_dim)
        self.key_dim = int(key_dim)
        self.sigma_min = float(sigma_min)
        self.sigma_span = float(sigma_max) - float(sigma_min)
        self.ell_min = float(ell_min)
        self.ell_span = float(ell_max) - float(ell_min)
        self.kappa_floor = float(kappa_floor)
        self.a_mu_floor = float(a_mu_floor)
        self.a_mu_init_gain = float(a_mu_init_gain)
        self.r_max = float(r_max)

        # torch.export and torch.compile(fullgraph=True) both die on the FIRST
        # `self.irreps_in.dim` in forward(): Irreps is a Python object and the
        # property iterates it, which Dynamo reports as
        # "Unsupported call_function BuiltinVariable(iter)
        #  [UserDefinedObjectVariable(Irreps)]". The width is a static shape, so
        # it is resolved once, here, and forward() reads a plain int.
        self._irreps_in_dim = int(self.irreps_in.dim)

        # Squared norms are even under inversion for every l, so the invariant
        # vector is a genuine 0e feature and the scalar fields cannot leak a
        # pseudoscalar.
        self.non_scalar_irreps = o3.Irreps(self.irreps_in[1:])
        self.non_scalar_norm = o3.Norm(self.non_scalar_irreps, squared=True)
        invariant_dim = self.scalar_dim + self.non_scalar_norm.irreps_out.dim

        self.invariant_norm = nn.LayerNorm(invariant_dim)
        self.scalar_net = nn.Sequential(
            nn.Linear(invariant_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, _N_SCALAR_OUTPUTS),
        )
        # ell_min + ell_span * sigmoid(raw + ell_shift) = ell_init at raw = 0.
        # sigma needs no such shift: a zero sigmoid argument already lands on the
        # midpoint of [sigma_min, sigma_max], whereas ell_init is deliberately not
        # the midpoint of [ell_min, ell_max].
        self.register_buffer(
            "ell_shift",
            torch.tensor(
                _inverse_sigmoid(
                    (float(ell_init) - self.ell_min) / self.ell_span, "ell_init"
                )
            ),
        )

        # Raw bond logit, in TRACE's attention style: s_ij = (Q_i . K_j)/sqrt(d).
        # Deliberately not symmetric here - the symmetric and antisymmetric parts
        # are separated in forward(), and both are used.
        self.q_proj = nn.Linear(invariant_dim, self.key_dim, bias=False)
        self.k_proj = nn.Linear(invariant_dim, self.key_dim, bias=False)

        # The solver's starting point, with its OWN parameters (SPEC 2.0.1).
        self.solver_seed = SolverSeedPotential(invariant_dim, self.hidden_dim)

        # kappa(|r_ij|): a function of bond length alone is exchange-symmetric by
        # construction, so this term cannot violate kappa_ij = kappa_ji.
        self.bond_radial_basis = int(bond_radial_basis)
        if self.bond_radial_basis > 0:
            centers = torch.linspace(
                0.0, self.r_max, self.bond_radial_basis, dtype=torch.get_default_dtype()
            )
            self.register_buffer("rbf_centers", centers)
            self.rbf_width = self.r_max / max(self.bond_radial_basis - 1, 1)
            radial_hidden = max(16, self.bond_radial_basis * 2)
            self.radial_bias = nn.Sequential(
                nn.Linear(self.bond_radial_basis, radial_hidden),
                nn.SiLU(),
                nn.Linear(radial_hidden, 1),
            )
        else:
            self.register_buffer("rbf_centers", None)
            self.rbf_width = 1.0
            self.radial_bias = None

        # The only genuinely equivariant head. 0e + 2e is the symmetric part of
        # 1o (x) 1o, which is precisely a symmetric 3x3 tensor; see the module
        # docstring for why the antisymmetric 1e part is neither available at
        # l_max = 2 nor wanted.
        self.a_mu_linear = checked_o3_linear(
            self.irreps_in, "1x0e + 1x2e", name="A_mu (dipole stiffness, 0e + 2e)"
        )
        self.a_mu_trace_init = float(a_mu_trace_init)

        # Shrink the readout so L really does start near a_mu_trace_init * I. At
        # unit gain the five unit-norm 2e slices contribute sqrt(5) against the
        # identity's sqrt(3) and L comes out indefinite; see
        # DEFAULT_A_MU_INIT_GAIN for the measured consequences.
        with torch.no_grad():
            for parameter in self.a_mu_linear.parameters():
                parameter.mul_(self.a_mu_init_gain)

        identity, traceless = _symmetric_tensor_basis()
        default_dtype = torch.get_default_dtype()
        self.register_buffer("a_mu_identity", identity.to(default_dtype))
        self.register_buffer("a_mu_traceless_basis", traceless.to(default_dtype))

        self.t_seed_scale = nn.Parameter(torch.tensor(float(t_seed_scale)))

        # -- optional sectors, built LAST ----------------------------------
        #
        # Order matters and not only for readability. Everything above is the
        # closed-shell head, and every module above draws from the global RNG in
        # a fixed sequence. Constructing the optional heads after all of them
        # means that with the sectors OFF the stream is byte-for-byte the one
        # this module has always produced, so `spin=False, quadrupole_blocks=None`
        # is not merely "equivalent to before", it is bitwise identical for the
        # same manual_seed -- which is the property the regression suite pins.
        self._bound_fields = tuple(_PHYSICAL_BOUND_FIELDS)
        self.spin = bool(spin)
        if self.spin:
            self._build_spin_heads(invariant_dim, j_exchange_floor, spin_init_gain)

        self.quadrupole_blocks = self._resolve_quadrupole_blocks(quadrupole_blocks)
        self.a_theta_truncated = False
        if self.quadrupole_blocks:
            self._build_quadrupole_heads(
                allow_truncated_a_theta=allow_truncated_a_theta,
                a_theta_floor=a_theta_floor,
                a_theta_trace_init=a_theta_trace_init,
                a_theta_init_gain=a_theta_init_gain,
                coupling_init_gain=quadrupole_coupling_init_gain,
            )

        # Last of the three optional sectors, and last for the same reason: with
        # it off, every draw above it is the one this module has always made.
        self.hyperpolarizability = bool(hyperpolarizability)
        if self.hyperpolarizability:
            self._build_hyperpolarizability_head(hyper_d_init_gain)

        # SPEC 13, after every other sector so their RNG draws are unchanged.
        self.dispersion = bool(dispersion)
        if self.dispersion:
            self.dispersion_electrons_init = float(dispersion_electrons_init)
            self.dispersion_electrons_floor = float(dispersion_electrons_floor)
            self.dispersion_electrons = DispersionElectrons(
                invariant_dim,
                self.hidden_dim,
                self.dispersion_electrons_init,
                self.dispersion_electrons_floor,
            )
            self._bound_fields = self._bound_fields + _DISPERSION_BOUND_FIELDS

        # The non-learned physical bounds, carried in the state dict so that a
        # checkpoint cannot be loaded into a module that quietly disagrees about
        # them. See the class docstring: this used to report "All keys matched
        # successfully" while sigma moved 1.588 -> 0.259 A.
        self.register_buffer(
            "physical_bounds",
            torch.tensor(
                [float(getattr(self, field)) for field in self._bound_fields],
                dtype=torch.float64,
            ),
        )

    # -- optional sector construction --------------------------------------

    def _build_spin_heads(
        self, invariant_dim: int, j_exchange_floor: float, spin_init_gain: float
    ) -> None:
        """Construct the SPEC 9 spin sector.  Called only when ``spin=True``.

        Two invariant scalar outputs (``chi^s`` and the raw ``J^x``) and one
        extra query/key pair for the spin-channel bond logit.  Nothing
        equivariant: the spin sector splits the *charge* sector only, and the
        induced dipole responds to the total density (SPEC 9), so ``A_mu``,
        ``sigma`` and ``ell`` are untouched.

        The gain is what makes the closed-shell limit the *starting point*
        rather than merely a reachable point: at ``chi^s = 0`` and a zero spin
        bond logit, ``chi_up == chi_down`` and ``kappa_spin == kappa`` bitwise,
        and the spin extension reproduces the closed-shell model exactly.
        """
        if j_exchange_floor <= 0.0:
            raise ValueError(
                f"j_exchange_floor must be strictly positive, got {j_exchange_floor}; "
                "the spin mode of the on-site Hessian has curvature 2 J^x, so a "
                "non-positive floor admits an indefinite block and an energy "
                "unbounded below along spin polarisation (SPEC 9.1)"
            )
        if spin_init_gain <= 0.0:
            raise ValueError(
                f"spin_init_gain must be strictly positive, got {spin_init_gain}; "
                "zero would start the spin sector exactly closed-shell with no "
                "symmetry to break, the same defect a_mu_init_gain=0 has"
            )
        self.j_exchange_floor = float(j_exchange_floor)
        self.spin_init_gain = float(spin_init_gain)

        self.spin_scalar_net = nn.Sequential(
            nn.Linear(invariant_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, _N_SPIN_SCALAR_OUTPUTS),
        )
        self.spin_q_proj = nn.Linear(invariant_dim, self.key_dim, bias=False)
        self.spin_k_proj = nn.Linear(invariant_dim, self.key_dim, bias=False)
        with torch.no_grad():
            for module in (self.spin_scalar_net, self.spin_q_proj, self.spin_k_proj):
                for parameter in module.parameters():
                    parameter.mul_(self.spin_init_gain)

        self._bound_fields = self._bound_fields + _SPIN_BOUND_FIELDS

    def _resolve_quadrupole_blocks(
        self, quadrupole_blocks: Optional[Sequence[str]]
    ) -> tuple[str, ...]:
        """Normalise and validate the requested block list.

        ``None`` means the sector is off, which is the default and the only
        setting under which this module behaves exactly as it did before the
        quadrupole sector existed.  An empty sequence is treated as off too, but
        an *unknown* block name is an error rather than a silent no-op: a typo
        like ``("A_theta",)`` would otherwise switch nothing on and report
        nothing, which is the same class of failure as a dead readout block.
        """
        if quadrupole_blocks is None:
            return ()
        if isinstance(quadrupole_blocks, str):
            raise TypeError(
                "quadrupole_blocks must be a sequence of block names, not a bare "
                f"string {quadrupole_blocks!r} (which would be read character by "
                f"character). Pass a tuple, e.g. {(quadrupole_blocks,)!r}."
            )
        requested = tuple(quadrupole_blocks)
        unknown = [name for name in requested if name not in QUADRUPOLE_BLOCKS]
        if unknown:
            raise ValueError(
                f"unknown quadrupole block(s) {unknown}; the blocks this head can "
                f"build are {list(QUADRUPOLE_BLOCKS)}. Names are case-sensitive and "
                "match SPEC 10's table ('A_Theta', not 'a_theta' or 'A_theta'): an "
                "unrecognised name is rejected rather than ignored, because a "
                "silently unbuilt block is indistinguishable from a dead one."
            )
        # De-duplicate while keeping QUADRUPOLE_BLOCKS' order, so construction
        # order (and therefore the RNG stream) depends on the set requested and
        # not on the order the caller happened to type.
        return tuple(name for name in QUADRUPOLE_BLOCKS if name in requested)

    def _require_block_irreps(self, block: str, irreps_out: str) -> None:
        """Raise :class:`QuadrupoleIrrepsError` unless every output irrep is live.

        SPEC 10: "Enable quadrupoles when the feature irreps carry what the block
        needs, and raise a NAMED error when they do not -- never construct a head
        whose anisotropic block is silently dead."  This is that check, and it
        uses :func:`unreachable_output_irreps`, i.e. e3nn's ``output_mask``, so
        that a zero-multiplicity input irrep (``16x2e`` written as ``0x2e``) is
        caught as well as an absent one.  Counting instructions is not enough;
        see :func:`checked_o3_linear`.
        """
        missing = unreachable_output_irreps(self.irreps_in, irreps_out)
        if not missing:
            return
        hint = ""
        if block == "A_Theta" and irreps_out == _A_THETA_TRUNCATED_IRREPS:
            hint = (
                " This is already the TRUNCATED 0e + 2e form requested with "
                "allow_truncated_a_theta=True, so there is nothing further to fall "
                "back to: features with no live 2e block cannot carry any "
                "anisotropic quadrupole stiffness at all."
            )
        elif block == "A_Theta":
            hint = (
                " A_Theta is the symmetric part of 2e (x) 2e = 0e + 2e + 4e (the 1e "
                "and 3e are antisymmetric and cannot enter a symmetric bilinear "
                "form), so it needs 4e features, i.e. l_max >= 4. TRACE's l_max = 2 "
                "features do not have them. Pass allow_truncated_a_theta=True to "
                "build the 0e + 2e part instead, understanding that it is an "
                "isotropic-plus-lowest-anisotropy response and NOT a general "
                "quadrupole stiffness."
            )
        elif block == "mu_theta":
            hint = (
                " mu is 1o and Theta is 2e, so the coupling coefficient is "
                "parity-ODD: 1o (x) 2e = 1o + 2o + 3o. Note the 2o. A network with "
                "the usual natural parity (-1)^l supplies 0e + 1o + 2e + 3o at "
                "l_max = 3 and has no 2o block at any l_max, so 'raise l_max' is "
                "not necessarily the fix -- the features have to carry odd parity "
                "at l = 2."
            )
        elif block == "q_theta":
            hint = (
                " The charge-quadrupole coefficient is a single 2e object, so this "
                "needs l = 2 features of EVEN parity with non-zero multiplicity."
            )
        raise QuadrupoleIrrepsError(
            f"quadrupole block {block!r} cannot be read out from "
            f"irreps_in={self.irreps_in}: {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} unreachable, so "
            f"{'that block' if len(missing) == 1 else 'those blocks'} would be "
            f"identically zero for the lifetime of the model. This head was asked "
            f"for {irreps_out}; SPEC 10 requires {block!r} to carry "
            f"{QUADRUPOLE_BLOCK_IRREPS[block]}.{hint}"
        )

    def _build_quadrupole_heads(
        self,
        allow_truncated_a_theta: bool,
        a_theta_floor: float,
        a_theta_trace_init: float,
        a_theta_init_gain: float,
        coupling_init_gain: float,
    ) -> None:
        """Construct the SPEC 10 quadrupole sector for the requested blocks."""
        if a_theta_floor <= 0.0:
            raise ValueError(
                f"a_theta_floor must be strictly positive, got {a_theta_floor}; it "
                "is what makes A_Theta positive *definite* rather than "
                "semidefinite, exactly as a_mu_floor does for the dipole block"
            )
        if a_theta_init_gain <= 0.0:
            raise ValueError(
                f"a_theta_init_gain must be strictly positive, got "
                f"{a_theta_init_gain}; zero would start A_Theta exactly isotropic "
                "with no anisotropy to grow from"
            )
        if coupling_init_gain <= 0.0:
            raise ValueError(
                f"quadrupole_coupling_init_gain must be strictly positive, got "
                f"{coupling_init_gain}; zero would start the q-Theta and mu-Theta "
                "couplings identically dead, which is the state SPEC 3's "
                "reachability rule exists to prevent"
            )
        self.a_theta_floor = float(a_theta_floor)
        self.a_theta_trace_init = float(a_theta_trace_init)
        self.a_theta_init_gain = float(a_theta_init_gain)
        self.quadrupole_coupling_init_gain = float(coupling_init_gain)

        default_dtype = torch.get_default_dtype()
        self.register_buffer(
            "quadrupole_cartesian", quadrupole_cartesian_basis().to(default_dtype)
        )

        if "A_Theta" in self.quadrupole_blocks:
            irreps_out = QUADRUPOLE_BLOCK_IRREPS["A_Theta"]
            if unreachable_output_irreps(self.irreps_in, irreps_out):
                # Only NOW consider the truncation, and only on explicit request.
                # Checking the full irreps first means the fallback can never be
                # taken by a head that could have had the real thing.
                if not allow_truncated_a_theta:
                    self._require_block_irreps("A_Theta", irreps_out)
                irreps_out = _A_THETA_TRUNCATED_IRREPS
                # The truncated form still has to be reachable -- l_max = 1
                # features have no 2e either, and would give a dead anisotropy.
                self._require_block_irreps("A_Theta", irreps_out)
                self.a_theta_truncated = True
                warnings.warn(
                    "A_Theta is being built WITHOUT its 4e part, because "
                    f"irreps_in={self.irreps_in} does not carry 4e. The symmetric "
                    "part of 2e (x) 2e is 0e + 2e + 4e; what this head will emit "
                    "is the 0e + 2e part, an isotropic response plus the lowest "
                    "anisotropy. It is positive definite and equivariant, but it "
                    "is NOT a general quadrupole stiffness -- it cannot "
                    "distinguish two traceless deformations lying in the same 2e "
                    "shell. Quadrupole responses computed from it are an "
                    "approximation of stated form, not the full SPEC 10 object. "
                    "Use l_max >= 4 features for the general case.",
                    QuadrupoleTruncationWarning,
                    stacklevel=3,
                )
            self.a_theta_linear = checked_o3_linear(
                self.irreps_in,
                irreps_out,
                name=f"A_Theta (quadrupole stiffness, {irreps_out})",
            )
            with torch.no_grad():
                for parameter in self.a_theta_linear.parameters():
                    parameter.mul_(self.a_theta_init_gain)

            identity, rank_2, rank_4 = _quadrupole_stiffness_basis(
                include_4e=not self.a_theta_truncated
            )
            self.register_buffer("a_theta_identity", identity.to(default_dtype))
            self.register_buffer("a_theta_rank2_basis", rank_2.to(default_dtype))
            self.register_buffer(
                "a_theta_rank4_basis",
                None if rank_4 is None else rank_4.to(default_dtype),
            )

        if "chi_theta" in self.quadrupole_blocks:
            self._require_block_irreps("chi_theta", QUADRUPOLE_BLOCK_IRREPS["chi_theta"])
            self.chi_theta_linear = checked_o3_linear(
                self.irreps_in,
                QUADRUPOLE_BLOCK_IRREPS["chi_theta"],
                name="chi-Theta (permanent quadrupole source, 2e)",
            )
            with torch.no_grad():
                for parameter in self.chi_theta_linear.parameters():
                    parameter.mul_(self.quadrupole_coupling_init_gain)

        if "q_theta" in self.quadrupole_blocks:
            self._require_block_irreps("q_theta", QUADRUPOLE_BLOCK_IRREPS["q_theta"])
            self.q_theta_linear = checked_o3_linear(
                self.irreps_in,
                QUADRUPOLE_BLOCK_IRREPS["q_theta"],
                name="q-Theta (charge-quadrupole coupling, 2e)",
            )
            with torch.no_grad():
                for parameter in self.q_theta_linear.parameters():
                    parameter.mul_(self.quadrupole_coupling_init_gain)

        if "mu_theta" in self.quadrupole_blocks:
            self._require_block_irreps("mu_theta", QUADRUPOLE_BLOCK_IRREPS["mu_theta"])
            self.mu_theta_linear = checked_o3_linear(
                self.irreps_in,
                QUADRUPOLE_BLOCK_IRREPS["mu_theta"],
                name="mu-Theta (dipole-quadrupole coupling, 1o + 2o + 3o)",
            )
            with torch.no_grad():
                for parameter in self.mu_theta_linear.parameters():
                    parameter.mul_(self.quadrupole_coupling_init_gain)
            for order, basis in zip((1, 2, 3), _mu_theta_basis()):
                self.register_buffer(
                    f"mu_theta_basis_{order}", basis.to(default_dtype)
                )

        self._a_theta_truncated_flag = 1.0 if self.a_theta_truncated else 0.0
        self._bound_fields = self._bound_fields + _QUADRUPOLE_BOUND_FIELDS

    def _build_hyperpolarizability_head(self, hyper_d_init_gain: float) -> None:
        """Construct the ``1o`` direction ``d``.  Called only when switched on.

        One :func:`checked_o3_linear`, and the check is the whole safety
        argument.  ``1o`` is reachable *only* from a ``1o`` input block, so a
        feature set with no odd-parity l = 1 - or with one of zero multiplicity,
        which is the hole ``output_mask`` closes and instruction counting does
        not - gives a head that returns identically zero forever.  That failure
        is worse here than anywhere else in this file, because zero is also the
        physically correct answer at a centrosymmetric site: a dead readout and a
        working selection rule look the same in every downstream number.  So the
        reachability question is asked explicitly first, with a named error, and
        then asked again by ``checked_o3_linear`` when the module is actually
        built.

        No positive-definiteness, no floor, no basis: ``d`` is an unconstrained
        vector.  That is not an oversight - a hyperpolarizability direction has
        no definiteness to preserve, since it multiplies a tensor rather than
        forming a quadratic form.
        """
        if hyper_d_init_gain <= 0.0:
            raise ValueError(
                f"hyper_d_init_gain must be strictly positive, got "
                f"{hyper_d_init_gain}; a zero gain starts d identically zero, and "
                "beta would then be zero for every system -- which reads exactly "
                "like the centrosymmetric selection rule firing, while in fact "
                "being an untrained readout saying nothing. See "
                "DEFAULT_HYPER_D_INIT_GAIN."
            )

        missing = unreachable_output_irreps(self.irreps_in, HYPER_D_IRREPS)
        if missing:
            raise HyperpolarizabilityIrrepsError(
                f"the hyperpolarizability direction d cannot be read out from "
                f"irreps_in={self.irreps_in}: {', '.join(missing)} is unreachable, "
                "so d -- and therefore every beta built from it -- would be "
                "identically zero for the lifetime of the model. d is 1o because "
                "beta = alpha (x) d must be parity-ODD, which is what makes it "
                "vanish at a centrosymmetric site; 1o can only be produced from a "
                "1o input block, so these features need l = 1 of ODD parity and "
                "NON-ZERO multiplicity (a '0x1o' block contributes a weightless "
                "instruction that looks like a path and carries nothing). Note "
                "that a zero d is also the correct answer at an inversion centre, "
                "so this cannot be detected downstream -- which is why it is "
                "refused here."
            )

        self.hyper_d_init_gain = float(hyper_d_init_gain)
        self.d_linear = checked_o3_linear(
            self.irreps_in,
            HYPER_D_IRREPS,
            name="d (hyperpolarizability direction, 1o)",
        )
        with torch.no_grad():
            for parameter in self.d_linear.parameters():
                parameter.mul_(self.hyper_d_init_gain)

        # A fingerprint of d at INITIALISATION, so the head can answer a
        # question a bare [n, 3] tensor cannot: was this ever trained?
        # HyperpolarizabilityUnidentifiableWarning says the call-site check is
        # impossible because '"never trained" is not a property of a float
        # array'. That is true of the ARRAY and false of the HEAD: d receives
        # exactly zero gradient from energy, force, dipole or alpha (measured --
        # d_linear.weight.grad is None after a backward through all four, while
        # 14 of 14 other head parameters receive one), so an untrained d is
        # BITWISE equal to its initialisation, which is decidable.
        # Non-persistent: it describes THIS construction, so a checkpoint whose
        # d differs from it is correctly read as trained.
        self.register_buffer(
            "_d_init_fingerprint",
            torch.cat(
                [p.detach().reshape(-1) for p in self.d_linear.parameters()]
            ).clone(),
            persistent=False,
        )

        # The octupolar (J=3) channel needs a 3o block in the features.  It is
        # OPTIONAL rather than required, because forbidding beta entirely on an
        # l_max=2 model would be worse than offering it with a stated blind spot:
        # alpha (x) d is still correct wherever d != 0, which is everywhere
        # except the high-symmetry sites Td and above.
        has_3o = any(ir.l == 3 and ir.p == -1 and mul > 0 for mul, ir in self.irreps_in)
        if has_3o:
            self.octupole_linear = checked_o3_linear(
                self.irreps_in,
                HYPER_OCTUPOLE_IRREPS,
                name="octupolar beta (3o, the J=3 Zyss component)",
            )
            with torch.no_grad():
                for parameter in self.octupole_linear.parameters():
                    parameter.mul_(self.hyper_d_init_gain)
        else:
            self.octupole_linear = None
            warnings.warn(
                "The hyperpolarizability channel is on but the features carry no "
                f"3o block ({self.irreps_in}), so beta is the dipolar part alone, "
                "beta = alpha (x) d. That is exact wherever d != 0, but it is "
                "IDENTICALLY ZERO at a tetrahedral site: at Td the vector "
                "representation is T2, which contains no A1, so no invariant "
                "vector exists and d vanishes by symmetry (measured 0.000e+00 on "
                "an ideal tetrahedral shell). A totally symmetric rank-3 tensor "
                "does contain A1 -- the beta_xyz component -- which is why "
                "zincblende crystals are nonlinear optical materials, so this "
                "model will predict exactly zero second-harmonic response for "
                "GaAs, ZnS and ZnSe at equilibrium. Supply features with l_max "
                ">= 3 to close that blind spot; the 3o channel is non-zero at Td "
                "(7.888 on the same shell) and still vanishes under inversion "
                "(4.4e-16), so it costs nothing in surface selectivity.",
                HyperpolarizabilityUnidentifiableWarning,
                stacklevel=3,
            )

        self._bound_fields = self._bound_fields + _HYPER_BOUND_FIELDS

        warnings.warn(
            "The hyperpolarizability channel is on. d_i is a PURE READOUT: it "
            "enters no energy, force, stress, charge, dipole or polarizability, "
            "so it has exactly zero gradient under an energy, force, dipole or "
            "polarizability loss. It is identifiable ONLY from beta training "
            "labels. Trained without them, hyperpolarizability(alpha, d) will "
            "still return a correctly equivariant tensor that obeys the "
            "centrosymmetric selection rule and is otherwise this module's "
            "initialisation -- an arbitrary magnitude and sign. Nothing "
            "downstream can detect that, which is why it is said here.",
            HyperpolarizabilityUnidentifiableWarning,
            stacklevel=3,
        )

    def seed_potential(
        self, node_features: Tensor, chi: Tensor, *, trainable: bool = False
    ) -> Tensor:
        """The per-atom seed potential ``u_hat`` (see :class:`SolverSeedPotential`).

        ``trainable=False`` (the energy path) evaluates the seed with its
        parameters DETACHED but its inputs attached: forces see the seed's
        geometry dependence, the data loss sees no handle on the seed's
        weights. ``trainable=True`` is for ``solver_seed_loss`` only.
        """
        invariants = self.node_invariants(node_features)
        if trainable:
            return self.solver_seed(invariants, chi)
        frozen = {
            name: parameter.detach()
            for name, parameter in self.solver_seed.named_parameters()
        }
        return torch.func.functional_call(self.solver_seed, frozen, (invariants, chi))

    def charge_transfer_law(
        self, bond_index: Tensor, bond_length: Tensor | None, species: Tensor | None
    ) -> Tensor | None:
        """SPEC 23: the factor ``>= 1`` by which charge transfer stiffens out of contact.

            law = exp(ln(K) sigmoid(x) + beta w softplus(x)),
            x   = (r - R_i - R_j - delta) / w,

        ``R`` the covalent radii (``constants.COVALENT_RADII``).  Inside contact
        ``law -> 1``; across the onset the hardness jumps smoothly by ``K``; beyond
        it, it grows as ``exp(beta r)``.  Smooth (C-infinity) everywhere.  ``None``
        when the law is off.

        The physics: a charge-transfer channel between sites coupled by ``h``
        across a gap ``Delta`` has susceptibility ``2 h^2 / Delta^3`` (second-order
        perturbation theory), so ``kappa = Delta^3 / (2 h^2)``.  Breaking orbital
        contact drops ``h`` by over an order of magnitude (about 2.5 eV along a
        covalent bond, about 0.1 eV across a hydrogen bond), i.e. ``kappa`` by
        ``K ~ (25)^2 ~ 10^3``.  Past that, ``h`` decays as a tunnelling tail and
        ``kappa`` grows as ``exp(beta r)``, ``beta`` ~ 3/A (the through-space
        decay of electron-transfer rates).

        What it cures, measured: water models trained on energies and forces
        alone learned a hardness flat in distance (0.25 eV/e^2 on O-H bonds,
        0.21 between molecules 4.5-6 A apart). Their polarizability per
        molecule grew from 2.5 to 6.2 A^3 between the monomer and the tetramer,
        against PBE0's 1.35-1.44 A^3. That is the charge-transfer polarizability
        catastrophe: charge moved across a hydrogen bond costs only kappa plus
        the on-site hardnesses, about 15-20 eV/e^2, over a 2 A lever arm.

        Defaults K = 1e3, beta = (given), delta = 0.5 A, w = 0.1 A give these
        hardness factors: O-H 0.96 A, 1.04; C-C 1.54 A, 1.06; Si-O, 1.01;
        Na-Cl 2.82 A, 1.21 (ionic contact kept); C-C stretched to 2.0 A, 27;
        O...H 1.9 A, 3.3e3; O...O 2.8 A, 1.9e4.
        """
        if self.charge_transfer_decay is None:
            return None
        if species is None or bond_length is None:
            raise ValueError(
                "the charge-transfer contact law needs species (atomic numbers) and "
                "bond_length"
            )
        from .constants import COVALENT_RADII

        species = species.reshape(-1).long().to(bond_length.device)
        if bool((species < 1).any()) or bool((species >= len(COVALENT_RADII)).any()):
            raise ValueError(
                f"atomic numbers must lie in 1..{len(COVALENT_RADII) - 1} for the "
                "charge-transfer contact law (Cordero covalent radii)"
            )
        radii = torch.tensor(COVALENT_RADII, dtype=bond_length.dtype, device=bond_length.device)
        contact = radii[species[bond_index[0]]] + radii[species[bond_index[1]]]
        w = self.charge_transfer_width
        x = (bond_length - contact - self.charge_transfer_onset) / w
        # softplus as logaddexp(x, 0): exact and smooth for every x.  (F.softplus
        # switches to x above its threshold of 20, a 2e-9 step in the exponent at
        # r = contact + 2.5 A -- a region this law reaches routinely.)
        exponent = math.log(self.charge_transfer_jump) * torch.sigmoid(x) + (
            self.charge_transfer_decay * w * torch.logaddexp(x, torch.zeros_like(x))
        )
        return torch.exp(exponent)

    def node_invariants(self, node_features: Tensor) -> Tensor:
        """Rotation- and inversion-invariant summary of the node features.

        Concatenates the ``0e`` block with the squared norms of every higher-l
        block, matching ``InvariantIrrepReadout`` in ``flashace.model``.
        """
        scalars = node_features[:, : self.scalar_dim]
        higher = node_features[:, self.scalar_dim :]
        return torch.cat((scalars, self.non_scalar_norm(higher)), dim=-1)

    def _radial_logit(self, bond_length: Tensor) -> Tensor:
        """Learned, exchange-symmetric contribution of bond length to ``kappa``."""
        scaled = (bond_length[:, None] - self.rbf_centers) / self.rbf_width
        return self.radial_bias(torch.exp(-scaled * scaled)).squeeze(-1)

    def dipole_stiffness(self, node_features: Tensor) -> Tensor:
        """Positive-definite dipole stiffness ``A_mu``, shape ``[n_atoms, 3, 3]``.

        ``L = c0 * I + sum_m c2_m W_m`` is the general symmetric 3x3 tensor built
        from the ``0e`` and ``2e`` readout channels, and transforms as
        ``L -> R L R^T``.  Then ``A = L L^T + eps I`` transforms the same way, is
        symmetric, and has eigenvalues ``lambda^2 + eps >= eps > 0``, so the dipole
        block of ``K`` can never go indefinite - which a free 0e + 2e readout
        would allow, and which breaks the solve.

        That bound is exact in exact arithmetic and remains TRUE but becomes
        UNRESOLVABLE in IEEE-754 once ``|L|`` is large enough that ``eps`` sinks
        below the rounding of ``L L^T``.  Feeding a single ``2e`` feature
        component of magnitude ``1e11`` already gives ``||A|| ~ 1e18``, where
        ``eps = 0.1`` is 3 orders below ``eps_machine * ||A||``.  ``eigvalsh``
        then reports a smallest eigenvalue of 0, and at ``1e30`` it reports
        ``-2.9e+39`` -- but ``||A|| = 1.4e+56`` there, so that number is itself
        an order of magnitude BELOW the eigensolver's own error floor
        ``eps_machine * ||A|| = 3.2e+40``.  It is noise in the eigensolver, not
        an indefinite matrix: ``torch.linalg.cholesky(A)`` SUCCEEDS at every
        magnitude tested, up to and including ``1e30``.

        So this is a resolution limit of any ``L L^T + eps I`` in floating point,
        not a property of this construction, and nothing downstream breaks.  It
        is recorded because the claim used to read "for every input", which
        invites a reader to verify it with ``eigvalsh`` and conclude there is a
        bug.  Node features of ``1e11`` are not reachable from a trained network
        in any case, and by ``1e150`` the ``sigma > 0`` assertion fires loudly.  ``L L^T`` is
        written out rather than squared: ``L`` is symmetric only to round-off
        (the Clebsch-Gordan slices give ``max |W - W^T| = 1.55e-15``), and
        ``L L^T`` is positive semidefinite whether or not that holds.
        """
        coefficients = self.a_mu_linear(node_features)
        trace_coefficient = coefficients[:, :1] + self.a_mu_trace_init
        traceless_coefficients = coefficients[:, 1:]

        l_tensor = trace_coefficient[:, :, None] * self.a_mu_identity
        l_tensor = l_tensor + torch.einsum(
            "nm,mab->nab", traceless_coefficients, self.a_mu_traceless_basis
        )
        return torch.matmul(l_tensor, l_tensor.transpose(-1, -2)) + (
            self.a_mu_floor * self.a_mu_identity
        )

    # -- quadrupole sector (SPEC 10) ---------------------------------------

    def quadrupole_stiffness(self, node_features: Tensor) -> Tensor:
        """Positive-definite quadrupole stiffness ``A_Theta``, ``[n_atoms, 5, 5]``.

        The dipole construction one rank up, and for the same reasons.
        ``L = c0 I_5 + sum_m c2_m B2_m + sum_m c4_m B4_m`` is the general
        symmetric 5x5 tensor built from the ``0e``, ``2e`` and ``4e`` readout
        channels; it transforms as ``L -> D^2 L (D^2)^T``, so
        ``A = L L^T + eps I_5`` transforms the same way, is symmetric, and has
        eigenvalues ``lambda^2 + eps >= eps > 0`` for every input.  A free
        ``0e + 2e + 4e`` readout would allow an indefinite quadrupole block,
        which breaks the solve exactly as an indefinite dipole block does.

        ``L L^T`` rather than ``L @ L``: the Clebsch-Gordan slices are symmetric
        only to double round-off (measured ``max |B2 - B2^T| = 8.88e-16``,
        ``max |B4 - B4^T| = 1.22e-15``), so the two agree only to that tolerance
        and only ``L L^T`` makes definiteness independent of it.

        In cartesian terms this is a rank-4 object,
        ``A_abcd = W[m,a,b] A_mn W[n,c,d]`` with ``W`` from
        :func:`quadrupole_cartesian_basis`, obeying
        ``A_abcd -> R_ai R_bj R_ck R_dl A_ijkl``; the 5x5 form carries the same
        information with no redundancy and no traceless constraint to maintain.

        When ``self.a_theta_truncated`` the ``4e`` term is absent.  The result is
        still symmetric, still positive definite and still equivariant - it is
        simply a restricted stiffness, and the caller was told so at
        construction.
        """
        coefficients = self.a_theta_linear(node_features)
        trace_coefficient = coefficients[:, :1] + self.a_theta_trace_init
        l_tensor = trace_coefficient[:, :, None] * self.a_theta_identity
        l_tensor = l_tensor + torch.einsum(
            "nm,mab->nab", coefficients[:, 1:6], self.a_theta_rank2_basis
        )
        if self.a_theta_rank4_basis is not None:
            l_tensor = l_tensor + torch.einsum(
                "nm,mab->nab", coefficients[:, 6:15], self.a_theta_rank4_basis
            )
        return torch.matmul(l_tensor, l_tensor.transpose(-1, -2)) + (
            self.a_theta_floor * self.a_theta_identity
        )

    def charge_quadrupole_coupling(self, node_features: Tensor) -> Tensor:
        """``C^qT_i``, ``[n_atoms, 5]``: the charge-quadrupole coefficient.

        A single ``2e`` object, contracting as ``q_i * (C^qT_i . Theta_hat_i)``.
        Unconstrained: it is an off-diagonal block of ``K`` and has no
        definiteness of its own.  It can nevertheless take the *coupled* Hessian
        indefinite while every diagonal block looks healthy, which is the trap
        SPEC 2.1.3 records - so the constrained Schur-complement check at the
        solution is not optional once this is switched on.
        """
        return self.q_theta_linear(node_features)

    def dipole_quadrupole_coupling(self, node_features: Tensor) -> Tensor:
        """``C^muT_i``, ``[n_atoms, 3, 5]``: the dipole-quadrupole coefficient.

        Contracts as ``mu_hat_i^T C^muT_i Theta_hat_i``.  ``mu`` is ``1o`` and
        ``Theta`` is ``2e``, so the product is parity-odd and the coefficient
        must be too: ``1o (x) 2e = 1o + 2o + 3o``, assembled here from the three
        ``wigner_3j(l, 1, 2)`` bases.  The ``2o`` is the part a natural-parity
        network does not have at any ``l_max``; construction has already refused
        if these features cannot carry it.

        Unconstrained, with the same caveat as
        :meth:`charge_quadrupole_coupling`.
        """
        coefficients = self.mu_theta_linear(node_features)
        coupling = torch.einsum(
            "nk,kab->nab", coefficients[:, 0:3], self.mu_theta_basis_1
        )
        coupling = coupling + torch.einsum(
            "nk,kab->nab", coefficients[:, 3:8], self.mu_theta_basis_2
        )
        return coupling + torch.einsum(
            "nk,kab->nab", coefficients[:, 8:15], self.mu_theta_basis_3
        )

    # -- hyperpolarizability sector ----------------------------------------

    def hyperpolarizability_octupole(self, node_features: Tensor) -> Tensor | None:
        """``beta^(3)_i``, ``[n_atoms, 3, 3, 3]``: the octupolar (J=3) part of beta.

        ``None`` when the feature irreps carry no ``3o`` block, in which case beta
        is the dipolar part alone and is identically zero at a tetrahedral site --
        see :data:`HYPER_OCTUPOLE_IRREPS` for why that matters and what it costs.

        Symmetric and traceless by construction, and ODD under inversion, so it
        vanishes at an inversion centre exactly as ``d`` does and the surface
        selectivity that motivates beta is untouched.
        """
        if self.octupole_linear is None:
            return None
        coefficients = self.octupole_linear(node_features)
        basis = _octupole_cartesian_basis(coefficients.dtype, coefficients.device)
        return torch.einsum("...m,mabc->...abc", coefficients, basis)

    def hyperpolarizability_direction_is_untrained(self) -> bool:
        """Whether ``d`` is still bit-identical to its initialisation.

        ``d`` enters no energy, force, stress, charge, dipole or polarizability,
        so a model trained on any of those has *exactly* zero gradient on it --
        measured: ``d_linear.weight.grad is None`` after a backward through a
        loss containing all four, while 14 of 14 other head parameters receive
        one.  An untrained ``d`` is therefore not merely near its
        initialisation, it is bitwise equal to it, and that is decidable.

        :class:`HyperpolarizabilityUnidentifiableWarning` used to say a
        call-site check was impossible because "never trained" is not a property
        of a float array.  True of the ARRAY handed to
        :func:`~trace_electrostatics.observables.hyperpolarizability`; false of
        the HEAD, which can remember what it started from.

        False when the head has no ``d``, and after any ``load_state_dict`` that
        moved the weights -- the fingerprint is non-persistent precisely so a
        loaded checkpoint differing from this construction reads as trained.
        """
        fingerprint = getattr(self, "_d_init_fingerprint", None)
        if fingerprint is None or not getattr(self, "hyperpolarizability", False):
            return False
        current = torch.cat(
            [p.detach().reshape(-1) for p in self.d_linear.parameters()]
        )
        if current.shape != fingerprint.shape:
            return False
        return bool(torch.equal(current, fingerprint))

    def hyperpolarizability_direction(self, node_features: Tensor) -> Tensor:
        """``d_i``, ``[n_atoms, 3]``: the ``1o`` direction that carries beta.

        A plain equivariant linear readout, transforming as ``d -> R d`` for
        every ``R`` in O(3) - proper rotations, mirrors and inversion alike, with
        no relabelling of axes (SPEC 3.1: e3nn's degree-1 D matrix *is* the
        cartesian rotation).  Under inversion ``R = -I`` it changes sign, so
        ``beta = alpha (x) d`` changes sign as a rank-3 odd tensor must, and at a
        site whose environment is invariant under inversion the two statements
        ``d = -d`` and ``d = d`` force ``d = 0``.

        Nothing here is constrained or floored, and nothing here is consumed by
        the solver.  The result is emitted on
        :class:`HyperpolarizabilityReadout` and used only by
        :func:`~trace_electrostatics.observables.hyperpolarizability`; see that
        class for why that isolation is the point and what it costs.
        """
        if self.hyperpolarizability_direction_is_untrained():
            warnings.warn(
                "hyperpolarizability direction `d` is bit-identical to its "
                "initialisation, so nothing has ever constrained it and any "
                "beta built from it is a random draw, not a prediction. `d` "
                "takes zero gradient from energy, force, dipole or alpha -- it "
                "is identifiable ONLY against beta/SHG labels -- so this is the "
                "expected state of any model trained without them. Train it "
                "against beta labels, or do not report beta.",
                HyperpolarizabilityUnidentifiableWarning,
                stacklevel=2,
            )
        return self.d_linear(node_features)

    def forward(
        self,
        node_features: Tensor,
        bond_index: Tensor,
        bond_length: Tensor | None = None,
        species: Tensor | None = None,
    ) -> ElectrostaticReadout:
        """Emit the electrostatic parameters for one configuration.

        Parameters
        ----------
        node_features:
            ``[n_atoms, irreps_in.dim]`` TRACE node features.
        bond_index:
            ``[2, n_bonds]`` endpoint pairs, in the same order and orientation as
            the incidence matrix ``B`` from ``topology.py``, whose convention is
            ``B[bond_index[0], b] = +1`` and ``B[bond_index[1], b] = -1``: a
            positive ``t_b`` puts ``+t_b`` of charge **onto** ``bond_index[0]``
            and takes it from ``bond_index[1]``.  Nothing here depends on the
            sign - the antisymmetrised logit that seeds ``t`` learns whichever
            sign the incidence matrix implies - but the two modules must describe
            the same convention, and an earlier version of this docstring
            described the opposite one.
        bond_length:
            Optional ``[n_bonds]`` bond lengths, in angstrom.  Required when the
            module was built with ``bond_radial_basis > 0``.  It is validated
            whenever it is passed, including when ``bond_radial_basis == 0`` and
            it does not enter ``kappa`` at all, so that a length that is the
            wrong shape or negative is never accepted on the quiet path either.

        Returns
        -------
        ElectrostaticReadout
            ``params`` for the energy and ``t_seed`` for the solver warm start.
        """
        if node_features.shape[-1] != self._irreps_in_dim:
            raise ValueError(
                f"node_features has width {node_features.shape[-1]}, expected "
                f"{self._irreps_in_dim} for irreps_in={self.irreps_in}"
            )
        if bond_index.shape[0] != 2:
            raise ValueError(
                f"bond_index must have shape [2, n_bonds], got {tuple(bond_index.shape)}"
            )
        if bond_length is not None:
            if bond_length.shape[0] != bond_index.shape[1]:
                raise ValueError(
                    f"bond_length has {bond_length.shape[0]} entries but there are "
                    f"{bond_index.shape[1]} bonds"
                )
            # A length is non-negative by definition, and a negative one used to
            # be fed straight into the Gaussian radial basis, which is perfectly
            # happy to evaluate it and return a kappa from a bond that does not
            # exist. Asserted asynchronously, as in energy.py: a `.any()` here
            # would be a host synchronisation on every forward pass.
            torch._assert_async(
                torch.all(bond_length >= 0.0),
                "bond_length contains a negative entry; |r_ij| cannot be negative "
                "and the Gaussian radial basis will happily evaluate it anyway",
            )

        invariants = self.invariant_norm(self.node_invariants(node_features))
        raw = self.scalar_net(invariants)

        chi = raw[:, _CHI]

        # Bounded on both sides: a sigmoid cannot drift to an unphysical width,
        # and it keeps the per-atom hardness floor J(0) inside a known window.
        sigma = self.sigma_min + self.sigma_span * torch.sigmoid(raw[:, _SIGMA_RAW])

        # Also bounded on both sides, and for a reason that is easy to miss: the
        # induced polarizability is alpha = K_E ell^2 / A_mu, so the a_mu_floor
        # only caps alpha if ell is capped as well. A softplus here left the
        # advertised 144 A^3 ceiling unenforced -- ell = 5.37 A would have meant
        # 4152 A^3 and ell = 20.4 A, 59926 A^3.
        ell = self.ell_min + self.ell_span * torch.sigmoid(
            raw[:, _ELL_RAW] + self.ell_shift
        )

        # THE HARDNESS FLOOR (SPEC 17.3): the tight Gram bound S_0(sigma) plus
        # an explicit margin, so the charge block's smallest eigenvalue is at
        # least the margin for every geometry. Evaluated from the sigma
        # predicted in this same pass, because the threshold moves with it.
        margin = getattr(self, "hardness_margin", DEFAULT_HARDNESS_MARGIN)
        floor = kernel_self(sigma) if margin == "legacy" else hardness_floor(sigma, margin)
        eta = F.softplus(raw[:, _ETA_RAW]) + floor

        a_mu = self.dipole_stiffness(node_features)

        # Raw, deliberately asymmetric attention logit, then split into its two
        # parts: the symmetric half sets the bond hardness, the antisymmetric
        # half seeds the solver.
        queries = self.q_proj(invariants)
        keys = self.k_proj(invariants)
        head, tail = bond_index[0], bond_index[1]
        scale = math.sqrt(self.key_dim)
        forward_logit = (queries[head] * keys[tail]).sum(dim=-1) / scale
        reverse_logit = (queries[tail] * keys[head]).sum(dim=-1) / scale

        symmetric_logit = symmetrise_bond_logits(forward_logit, reverse_logit)
        if self.radial_bias is not None:
            if bond_length is None:
                raise ValueError(
                    "bond_length is required when bond_radial_basis > 0; kappa "
                    "depends on |r_ij|"
                )
            symmetric_logit = symmetric_logit + self._radial_logit(bond_length)

        if getattr(self, "bond_hardness_law", "free") == "two_level":
            if species is None or bond_length is None:
                raise ValueError("bond_hardness_law='two_level' needs the bond lengths and the atomic numbers")
            from .bond_law import two_level_bond_hardness

            anchor = two_level_bond_hardness(bond_length, species[head], species[tail])
            span = math.log(self.bond_hardness_range)
            kappa = anchor * torch.exp(span * torch.tanh(symmetric_logit / span)) + self.kappa_floor
        else:
            kappa = F.softplus(symmetric_logit) + self.kappa_floor
        contact_law = self.charge_transfer_law(bond_index, bond_length, species)
        if contact_law is not None:
            kappa = kappa * contact_law
        # Ohm's-law seed: tau0 = -(u_head - u_tail) / kappa, which after the
        # envelope energy.py applies is exactly -kappa^-1 F B^T u_hat (bond_index
        # is (lo, hi) and B[lo, b] = +1). Antisymmetric under a bond flip because
        # kappa is symmetric. The seed parameters are FROZEN here: the data loss
        # must never tune the solver's starting point into a fitting knob. Only
        # solver_seed_loss trains them. The inputs stay attached, so the seed's
        # dependence on geometry is differentiated and forces stay exact.
        seed_potential = self.seed_potential(node_features, chi)
        t_seed = -(seed_potential[head] - seed_potential[tail]) / kappa

        # -- the two optional sectors, each fully gated ---------------------
        #
        # Both branches are skipped entirely when the sector is off, so the
        # closed-shell path executes exactly the operations it always did, in
        # the same order, on the same tensors.
        spin_readout: Optional[SpinReadout] = None
        extra: dict = {}
        if self.spin:
            spin_readout = self._spin_readout(
                invariants, chi, symmetric_logit, head, tail, scale
            )
            if contact_law is not None:
                spin_readout = spin_readout._replace(
                    kappa_spin=spin_readout.kappa_spin * contact_law
                )
            extra["chi_spin"] = spin_readout.chi_spin
            extra["j_exchange"] = spin_readout.j_exchange
            extra["kappa_spin"] = spin_readout.kappa_spin

        quadrupole_readout: Optional[QuadrupoleReadout] = None
        if self.quadrupole_blocks:
            quadrupole_readout = QuadrupoleReadout(
                A_Theta=(
                    self.quadrupole_stiffness(node_features)
                    if "A_Theta" in self.quadrupole_blocks
                    else None
                ),
                a_theta_truncated=self.a_theta_truncated,
                q_theta=(
                    self.charge_quadrupole_coupling(node_features)
                    if "q_theta" in self.quadrupole_blocks
                    else None
                ),
                mu_theta=(
                    self.dipole_quadrupole_coupling(node_features)
                    if "mu_theta" in self.quadrupole_blocks
                    else None
                ),
                chi_theta=(
                    self.chi_theta_linear(node_features)
                    if "chi_theta" in self.quadrupole_blocks
                    else None
                ),
            )
            extra.update(
                {
                    name: value
                    for name, value in (
                        ("A_Theta", quadrupole_readout.A_Theta),
                        ("chi_theta", quadrupole_readout.chi_theta),
                        ("q_theta", quadrupole_readout.q_theta),
                        ("mu_theta", quadrupole_readout.mu_theta),
                    )
                    if value is not None
                }
            )

        # The third sector, and the one that deliberately contributes NOTHING to
        # `extra`. d is not an electrostatic parameter: it appears in no block of
        # K, no linear term, no constraint. Routing it into ElectrostaticParams
        # would make an unidentifiable readout look like part of the energy, so
        # it leaves only on the readout, and _PARAMS_FIELD_TARGETS has no entry
        # for it on purpose.
        hyper_readout: Optional[HyperpolarizabilityReadout] = None
        if self.hyperpolarizability:
            hyper_readout = HyperpolarizabilityReadout(
                d=self.hyperpolarizability_direction(node_features)
            )

        # AOT note, re-measured. An earlier version of this comment said this
        # construction is "the remaining obstacle to torch.export /
        # torch.compile(fullgraph=True) of this module", because
        # ElectrostaticParams.__post_init__ (solver.py) runs
        # assert_hardness_floor and assert_positive_kappa unconditionally and
        # each evaluates bool(tensor.any()). The synchronisation is real in
        # EAGER mode -- 3 bool(tensor) calls per construction, counted by
        # instrumenting Tensor.__bool__ -- but it is already gated: with
        # diagnostics._is_tracing() true the count is 0, because
        # _needs_host_report queues torch._assert_async instead. So the gate IS
        # in __post_init__ and this is not the obstacle.
        #
        # The two obstacles that are real, both measured on torch 2.2.0:
        #
        # 1. e3nn 0.4.4 builds every TensorProduct with RecursiveScriptModule
        #    children (`tp._compiled_main_left_right`), and dynamo refuses:
        #    "ScriptModules aren't supported in UnspecializedNNModuleVariable".
        #    forward() hits this on its FIRST line, at self.node_invariants,
        #    long before reaching this construction.
        # 2. dynamo cannot construct a FROZEN dataclass at all, regardless of
        #    what its __post_init__ does. Control: a bare
        #    @dataclass(frozen=True) holding one tensor field fails with the
        #    same "Unsupported: call_function UserDefinedClassVariable()",
        #    while the identical mutable dataclass compiles. That is a torch
        #    limitation, not something to "fix" by unfreezing
        #    ElectrostaticParams -- immutability is what lets the solver mark
        #    and reject its own output as a seed.
        #
        # Neither is reachable from this side of the call.
        #
        # Optional fields go through _OPTIONAL_PARAM_FIELDS rather than being
        # passed literally: solver.py owns ElectrostaticParams and is free to
        # carry a sector this head does not emit, or not yet carry one it does.
        # Filtering by the fields the dataclass actually declares means a version
        # skew is a missing field on the readout -- visible, and still available
        # via `readout.spin` / `readout.quadrupole` -- rather than a TypeError
        # from a keyword the dataclass has never heard of.
        params = ElectrostaticParams(
            chi=chi,
            eta=eta,
            sigma=sigma,
            ell=ell,
            kappa=kappa,
            A_mu=a_mu,
            **{
                _OPTIONAL_PARAM_FIELDS[name]: value
                for name, value in extra.items()
                if name in _OPTIONAL_PARAM_FIELDS
            },
        )
        dispersion_readout: Optional[DispersionReadout] = None
        if self.dispersion:
            dispersion_readout = DispersionReadout(
                electrons=self.dispersion_electrons(invariants)
            )
        return ElectrostaticReadout(
            params=params,
            t_seed=t_seed,
            spin=spin_readout,
            quadrupole=quadrupole_readout,
            hyper=hyper_readout,
            dispersion=dispersion_readout,
        )

    def _spin_readout(
        self,
        invariants: Tensor,
        chi: Tensor,
        symmetric_logit: Tensor,
        head: Tensor,
        tail: Tensor,
        scale: float,
    ) -> SpinReadout:
        """The SPEC 9 spin sector for one configuration.

        The spin bond logit is formed from its own query/key projections and then
        split the same way the charge one is: the symmetric half sets
        ``kappa_spin``, the antisymmetric half seeds ``t^s``.  Both halves inherit
        their exactness from :func:`symmetrise_bond_logits` - IEEE-754 addition
        is commutative to the last bit - so ``kappa_spin`` is bitwise invariant
        under exchanging the bond endpoints and ``t_seed_spin`` changes sign
        exactly.

        ``kappa_spin`` reuses the *charge* symmetric logit as its base and adds
        the spin one, so a zero spin logit gives ``kappa_spin == kappa``
        bitwise.  That is the closed-shell fixed point of SPEC 9's bond sector:
        ``kappa^up = kappa^down = 2 kappa`` in the (up, down) basis, i.e. a
        spin-unpolarised system reproduces the closed-shell answer with the same
        ``kappa`` rather than with a silently halved one.
        """
        raw = self.spin_scalar_net(invariants)
        chi_spin = raw[:, _CHI_SPIN]

        # J^x > 0 STRICTLY (SPEC 9.1). softplus(raw) + floor, the construction
        # eta uses, and exact in IEEE-754 for every finite raw: softplus never
        # returns a negative double and correctly-rounded addition is monotone.
        j_exchange = exchange_coupling(raw[:, _J_EXCHANGE_RAW], self.j_exchange_floor)

        spin_queries = self.spin_q_proj(invariants)
        spin_keys = self.spin_k_proj(invariants)
        spin_forward = (spin_queries[head] * spin_keys[tail]).sum(dim=-1) / scale
        spin_reverse = (spin_queries[tail] * spin_keys[head]).sum(dim=-1) / scale

        kappa_spin = (
            F.softplus(
                symmetric_logit + symmetrise_bond_logits(spin_forward, spin_reverse)
            )
            + self.kappa_floor
        )
        t_seed_spin = self.t_seed_scale * antisymmetrise_bond_logits(
            spin_forward, spin_reverse
        )
        return SpinReadout(
            chi_spin=chi_spin,
            chi_up=chi + chi_spin,
            chi_down=chi - chi_spin,
            j_exchange=j_exchange,
            kappa_spin=kappa_spin,
            t_seed_spin=t_seed_spin,
        )

    # -- checkpoint safety --------------------------------------------------

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Refuse a checkpoint whose physical bounds differ from this module's.

        The learned parameters are meaningless without the bounds they were
        trained under: ``sigma`` is ``sigma_min + sigma_span * sigmoid(raw)`` and
        ``kappa`` is ``softplus(logit) + kappa_floor``, so loading weights into a
        module built with different bounds does not fail, it *reinterprets* them.
        Measured before this hook existed: ``load_state_dict`` reported "All keys
        matched successfully" while sigma went 1.588 -> 0.259 A and kappa
        1.046 -> 9.946 eV, i.e. the hardness floor J(0) moved by a factor of 6
        and the SQE bond block by a factor of 10, silently.

        Appending to ``error_msgs`` rather than raising directly is deliberate:
        it is the mechanism ``nn.Module.load_state_dict`` already uses, it
        reports every problem at once, and it raises whether or not ``strict``
        is set.
        """
        key = prefix + "physical_bounds"
        if key in state_dict:
            incoming = state_dict[key].reshape(-1).to(torch.float64)
            current = self.physical_bounds.reshape(-1).to(torch.float64)
            if incoming.numel() != current.numel():
                error_msgs.append(
                    f"size mismatch for {key}: checkpoint has {incoming.numel()} "
                    f"physical bounds, this module has {current.numel()} "
                    f"({', '.join(self._bound_fields)}). The optional spin, "
                    "quadrupole and hyperpolarizability sectors each append "
                    "their own bounds, so a length mismatch usually means the "
                    "checkpoint and this module disagree about which sectors are "
                    "switched on."
                )
            else:
                for field, want, have in zip(
                    self._bound_fields, incoming.tolist(), current.tolist()
                ):
                    if abs(have - want) > _BOUND_MATCH_RTOL * max(abs(want), 1.0):
                        knob = _PHYSICAL_BOUND_SOURCE.get(field, field)
                        error_msgs.append(
                            f"physical bound {field!r} differs: the checkpoint was "
                            f"trained with {want!r}, this module was constructed "
                            f"with {have!r}. The learned parameters mean different "
                            "physical values under the two, so loading would be a "
                            "silent reinterpretation. Construct ElectrostaticHeads "
                            f"so that {knob} == {want!r} (or retrain)."
                        )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
