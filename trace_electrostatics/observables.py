"""Electrical observables derived from the solved charge-equilibration system.

    P      = sum_i (q_i r_i + mu_i)                  polarization,  e*A
    alpha  = -d2U/dE2                                polarizability
    Z*_i   = dP / dr_i                               Born effective charges, e

Four things about this module are load-bearing.

**alpha is computed, never read out.**  There is no polarizability head.  The
polarizability is the linear response of the *same* quadratic form the energy
uses: a uniform field enters as ``-E . P = -E . D^T x``, which shifts the linear
term of ``U = 0.5 x^T K x + g^T x`` and moves the solution, so alpha is built
from the operator the solver was handed rather than from a second guess at it.
A negative alpha is then a reliable alarm that the solve landed on a saddle
rather than a minimum (SPEC 2.1.4) instead of a fitted number that cannot tell.

**alpha is the second derivative of the energy the model actually evaluates.**
The default route (named ``"fixed_k"`` for compatibility) threads the field
through the production solve and differentiates the returned energy twice, so
``alpha = -d2U/dE2`` of the model's own energy by construction.  Under the
default converged solve (SPEC 17) that energy IS the constrained minimum and
its implicit derivatives are exact, so this route equals the minimiser route
``D^T K^-1 D`` to the solve tolerance (measured ~5e-11 relative) -- at
symmetric structures too, where the legacy fixed-k response was arbitrary
(SPEC 15.2, 17.1).

Under the legacy ``solve_mode="fixed_k"`` the energy is the ``k``-th CG
iterate and its response is NOT the minimiser's.  Measured on an eight-atom
cluster, isotropic average in e^2 A^2 / eV:

    k                  1        2        3        6       12       30
    -d2U/dE2      0.1274   0.4906   0.4119   0.3689   0.4616   0.4616
    D^T K^-1 D    0.4616   0.4616   0.4616   0.4616   0.4616   0.4616
    error          72.4%     6.3%    10.8%    20.1%  4.8e-05  1.5e-07

The ``"minimiser"`` route returns ``D^T K^-1 D`` from the dense operator; it is
kept because it inverts the UNSYMMETRISED operator (next paragraph) and needs
no solve.  :func:`polarizability_gap` reports the distance between the routes.

**Maxwell reciprocity has to be tested on the minimiser route.**  A second
derivative of a scalar is symmetric by Clairaut's theorem, so the fixed-k alpha
is symmetric to round-off (measured 1.1e-16 relative) *whatever* the operator
does, and a reciprocity assertion on it is vacuous -- it would no longer catch
the Thole bug it was written for (a damping length keyed on one atom of the
pair, 14.9% asymmetry at close contact, 0.005% on a sparse geometry).  The
minimiser route is a genuine test, because ``D^T K^-1 D`` inverts the
*unsymmetrised* operator from :meth:`ElectrostaticAssembly.dense_hessian`:
injecting exactly that bug here moves its asymmetry from 2.7e-16 to 1.3e-02 at
1.19 A contact, while the fixed-k alpha stays symmetric at 1.5e-16 and the
*value* of alpha moves by 4.5%.  :func:`reciprocity_residual` is the check, and
it uses the minimiser route for this reason.  What each route can and cannot
see, stated plainly:

===================  ==========================  ==========================
detects              fixed_k (``-d2U/dE2``)      minimiser (``D^T K^-1 D``)
===================  ==========================  ==========================
operator asymmetry   in the VALUE only           as ``alpha != alpha^T``
a saddle             negative eigenvalue         negative eigenvalue
truncation of the    yes -- it *is* the          no -- k-independent
solve                truncated response
===================  ==========================  ==========================

**alpha contains the charge-transfer and the induced-dipole response together.**
They are different *blocks of the same inverse*, not two models added up: the
charge response already knows about the induced dipoles through the ``B^T T01``
block, and vice versa.  A long molecule gets most of its longitudinal alpha from
charge transfer and most of its transverse alpha from induced dipoles, and no
fixed partition would reproduce that.  :func:`polarizability_contributions`
splits the response by the *channel the polarization comes out of* -- charge
times position, versus induced dipole -- and the two halves sum to the total
exactly, on both routes.  On the fixed-k route the split is taken by giving the
two channels their own fields ``E_ct`` and ``E_ind`` and differentiating once
with respect to each and once along the diagonal; on the minimiser route it is
the corresponding split of the rows of ``D``.  Neither half is separately
symmetric, and neither is a polarizability of any physical subsystem.

**For a charged system the Born charges sum to Q_total, not to zero.**  Under a
rigid translation of every atom, ``q_i`` and ``mu_i`` are unchanged -- they
depend only on relative geometry -- so ``P -> P + Q_total d`` and therefore

    sum_i Z*_i = Q_total * identity

For a neutral system this is the familiar acoustic sum rule ``sum_i Z*_i = 0``.
For a charged fragment it is not, and force-symmetrising ``Z*`` to zero is simply
wrong: it discards the real translational dipole of a charged system.  Nothing in
this module enforces an acoustic sum rule.  :func:`acoustic_sum_rule_residual`
is provided to *check* the identity against ``Q_total``, never to impose it.

``Z*`` inherits the same consistency question as alpha, and the same answer.
The infrared intensity a user wants is ``dP/dR`` where ``P`` is the polarization
*the energy defines*, ``P = -dU/dE``; at finite ``k`` that is not the moment sum
``sum_i (q_i r_i + mu_i)``, because the iterate is not stationary and the
envelope theorem does not apply.  :func:`born_charges` differentiates the moment
sum, which is the right thing to do when the moments themselves are the
observable (a fitted charge model, a reported dipole); :func:`born_charges_from_energy`
returns ``Z* = -d2U/dE dR``, which is the mixed derivative the forces and the
field agree on and is what an intensity computed from this model's dynamics
will show.  Measured on the four-atom contact geometry at ``k = 6``, the two
differ by 7.4e-02 e (14.9% of the largest component) and converge onto each
other by ``k = 30`` (3.0e-07).  Both satisfy ``sum_i Z*_i = Q_total`` exactly --
the sum rule is a translation identity and holds iterate by iterate.

Hyperpolarizability -- a readout, and never a derivative of this energy
----------------------------------------------------------------------
``beta = -d3U/dE3`` under ``U = U0 - mu.E - (1/2) alpha:EE - (1/6) beta:EEE``.
It is rank 3 and **odd** under inversion, so it vanishes in any centrosymmetric
environment; that is what makes SHG and SFG surface-specific.

**It is not taken from this model's energy, and it must not be.**  The energy is
quadratic in the moments by construction (SPEC 1.5), so the minimiser's third
derivative is identically zero.  The fixed-k energy's third derivative is not
exactly zero -- CG step lengths depend on the right-hand side -- but that residue
is truncation, not physics, and reporting it as a hyperpolarizability would be
worse than reporting nothing.  The anharmonic term that would supply a genuine
beta failed audit on eleven counts, one decisive: a spontaneous polar ordering
produced a spurious beta up to 300x a real chemical beta in a nominally
centrosymmetric system, where symmetry forces zero.

``beta`` is therefore a **readout**: :func:`hyperpolarizability` contracts the
computed ``alpha`` with a learned ``1o`` direction ``d`` from
``heads.ElectrostaticHeads(hyperpolarizability=True)``::

    beta_abc = d_a alpha_bc

``alpha`` is ``0e + 2e``, so ``alpha (x) d`` is ``2x1o + 1x2o + 1x3o``, of
dimension **18** -- the full non-Kleinman SHG tensor, symmetric in the two field
indices ``(b, c)`` and in nothing else.  The habitual ``1o + 3o`` assignment has
dimension 10 and silently imposes **Kleinman symmetry**, which is a
far-from-resonance approximation, i.e. exactly not the regime an SHG experiment
works in.

**The 18 and the 10 are different objects and this module keeps them apart.**
Only the *fully symmetric* part of a rank-3 tensor survives contraction with
``E (x) E (x) E``, because that product is symmetric; so a static-field energy
can only ever see 10 of the 18 numbers, and the remaining 8 (the second ``1o``
and the ``2o``) are meaningful only for frequency-resolved response with
``omega_1 != omega_2``, which a ground-state energy functional does not have.
Hence two forms, named, with no default that silently picks for you in the case
that matters:

    ``form="full"``       all 18 components -- the spectroscopic readout
    ``form="kleinman"``   the fully symmetric part, 10 -- the only part an
                          energy can carry

Conflating them is the easiest way for a beta to be quietly wrong: the full
tensor contracted into a static-field energy silently *is* its symmetric part,
so nothing complains and 8 components vanish from the answer.

**Where the construction is zero, and where it is zero but should not be.**
``alpha (x) d`` is non-zero only where the site's point group admits an
invariant *vector*, which is a stronger condition than "not centrosymmetric".
At a centrosymmetric site that is exactly right: no component of ``beta`` is
allowed and none is produced.  At a **tetrahedral (Td)** site it is not: Td
admits no invariant vector, so ``d`` vanishes, while Td does admit one invariant
in ``3o`` - the ``xyz`` component, which is the ``d14`` of zincblende and the
reason GaAs is SHG-active.  Measured on an ideal diamond shell, the ``1o`` block
of the features is ``0.000e+00`` and so is ``d``.  The per-group counts are
tabulated in ``heads.py``.  The construction covers polar sites - surfaces,
interfaces, defects, molecules, the SFG and SHG targets of this package - and is
identically zero at a tetrahedral one.

**``beta`` is per site, and the cell sum is where the crystal's symmetry acts.**
``d`` is a per-atom readout, so :func:`hyperpolarizability` returns a per-atom
``beta_i``.  A *global* selection rule is then a statement about the sum, not
about any one site, and the two are routinely different: a centrosymmetric
crystal need not have centrosymmetric sites, and generally does not.  What
inversion symmetry of the cell guarantees is a pairing ``i -> i'`` with
``d_i' = -d_i`` (``1o``, odd) and ``alpha_i' = alpha_i`` (``0e + 2e``, even),
hence ``beta_i' = -beta_i`` and ``sum_i beta_i = 0``, while each ``beta_i``
is individually non-zero.  Measured on a centrosymmetric simple-cubic cell with
a two-atom basis at ``+-u`` (inversion centre at the origin, both sites polar):
per-site ``|beta_i|_F = 2.873804e-01``, ``|d_0 + d_1| = 8.1e-17``,
``|alpha_0 - alpha_1|_F = 1.9e-16``, and ``|beta_0 + beta_1|_F = 1.110340e-16``.
On a generic inversion-symmetric cluster ``S u -S`` whose sites have no symmetry
at all, per-site ``|beta_i|_F`` runs 1.747839e-01 to 2.609384e-01 and
``|sum_i beta_i|_F = 8.062661e-16``.  So a non-zero per-site ``beta`` in a
centrosymmetric crystal is the correct answer and not a broken selection rule;
it is the sum that has to vanish, and it does, to rounding.

That also fixes how a cell-level number must be formed: it is ``sum_i beta_i``,
**not** ``alpha_total (x) sum_i d_i``.  The shape check below rejects the
accidental pairing of a system ``[3, 3]`` alpha with a per-atom ``[n, 3]`` d,
but it cannot reject a deliberate one in which ``d`` has already been summed,
and that object is not the sum of the site tensors.  Sum the betas.

**And ``d`` is unidentifiable from anything else this package computes.**  It
enters no energy, force, stress, charge, dipole or ``alpha``, so an energy- or
force-trained model has exactly zero gradient on it.  Without beta labels the
number this function returns is the head's initialisation: correctly
equivariant, correctly vanishing at a centrosymmetric site, and otherwise
arbitrary in magnitude and sign.  The construction is still the right one -- an
anharmonic energy would instead force ``beta = -2 c alpha^3``, Miller's rule as
an untestable identity, and an exact two-level model shows beta is not a
function of alpha at all (alpha fixed to 0.00% while beta runs 0 -> 5.333,
``beta/alpha^3`` varying 18x) -- but the honest statement about an
energy-trained checkpoint is that its beta is noise with the right symmetry.

**Why that is said at construction and not here.**  :func:`hyperpolarizability`
takes a bare tensor and has no way to know whether the ``d`` it was handed was
ever fitted; "untrained" is not a property of a ``[n, 3]`` float array.  The
head is the only object that knows, and the guarantee it provides is stronger
than a call-site check could be: ``d`` cannot be obtained from this package
except from a module constructed with ``hyperpolarizability=True``, and that
constructor raises the warning every time, in every process, *before* any number
exists -- including the process that merely loads a checkpoint, since the module
must be built with the sector on before ``load_state_dict`` will accept one
(a hyper checkpoint loaded into a module without the sector is a ``RuntimeError``
naming the missing ``d_linear`` keys, not a silent success).  So the warning is
not a best effort that this function should back up; it is unavoidable, and
repeating it per call would add noise to a path that is already covered.  Do not
"fix" this by warning here, and do not remove the construction warning on the
grounds that the call site could do it -- the call site genuinely cannot.

Open versus periodic boundaries
-------------------------------
``P`` as ``sum_i q_i r_i`` is the open-boundary (cluster) definition.  Under
periodic boundary conditions it is branch dependent, and -- worse -- it cannot
see the divergence-free bond currents that carry a crystal's dielectric
response (a monatomic crystal has ``q = 0`` and still polarizes).
:func:`bond_polarization` is the periodic definition (SPEC 15):
``Omega P = sum_i rho_i r_i + sum_b f_b tau_b d_b + sum_i mu_i``, with ``d_b``
the image-resolved bond vector.  It equals ``sum_i (q_i r_i + mu_i)`` under
open boundaries, is single valued under PBC, and is what the field couples to
in the electric enthalpy the solver minimises.  Both polarizability routes,
the Born charges from the enthalpy, :func:`dielectric_tensor` (``eps_inf``)
and :func:`static_dielectric_tensor` (``eps_0``) all work on periodic
assemblies through it.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import NamedTuple, Tuple

import torch
from torch import Tensor

import math

from .constants import K_E, to_polarizability_volume
from .diagnostics import null_space_basis
from .heads import _dipolar_rank3, _symmetrise_rank3
from .energy import (
    ElectrostaticAssembly,
    _fragment_constraints,
    _solve_matrix_free,
    _moments,
    _solver_geometry,
    _solver_options,
)
from .solver import (
    FragmentConstraints,
    GeometrySeed,
    SolveResult,
    assemble_system,
    solve_charges,
)

#: ``alpha = -d2U/dE2`` of the fixed-k energy the model actually evaluates.
#: The default, because it is the response the model's own forces produce.
FIXED_K = "fixed_k"

#: ``alpha = D^T K^-1 D``, the response of the exact minimiser.  The
#: ``k -> infinity`` limit of :data:`FIXED_K`, and the route on which Maxwell
#: reciprocity is a genuine test of the assembled operator.
MINIMISER = "minimiser"

#: The route taken when a caller does not name one.
DEFAULT_ROUTE = FIXED_K

_ROUTES = (FIXED_K, MINIMISER)

#: All 18 components of ``beta``: ``2x1o + 1x2o + 1x3o``, symmetric in the two
#: field indices and in nothing else.  The spectroscopic readout.
BETA_FULL = "full"

#: The fully symmetric part of ``beta``, ``1o + 3o``, 10 components.  The only
#: part that can enter an energy, because ``E (x) E (x) E`` is symmetric.
BETA_KLEINMAN = "kleinman"

#: The form :func:`hyperpolarizability` returns when a caller does not name one.
#: ``BETA_FULL``, because "the hyperpolarizability" in a spectroscopic context
#: means the whole tensor; the energy-facing form has to be asked for, since
#: using it is a statement about what the number is for.
DEFAULT_BETA_FORM = BETA_FULL

_BETA_FORMS = (BETA_FULL, BETA_KLEINMAN)

__all__ = [
    "BETA_FULL",
    "BETA_KLEINMAN",
    "DEFAULT_BETA_FORM",
    "DEFAULT_ROUTE",
    "FIXED_K",
    "MINIMISER",
    "PolarizabilityGap",
    "acoustic_sum_rule_residual",
    "bond_polarization",
    "dielectric_tensor",
    "static_dielectric_tensor",
    "nonanalytic_force_constants",
    "born_charges",
    "born_charges_from_energy",
    "born_charges_minimiser",
    "FixedKDegenerateError",
    "FixedKDegenerateWarning",
    "fixed_k_residual",
    "hyperpolarizability",
    "hyperpolarizability_energy",
    "hyperpolarizability_readout",
    "polarizability",
    "polarizability_contributions",
    "polarizability_fixed_k",
    "polarizability_gap",
    "polarizability_minimiser",
    "polarization",
    "polarization_from_energy",
    "reciprocity_residual",
]

#: Where :func:`_field_response` parks its result on the assembly it was built
#: from.  See that function for why the cache exists and why it is safe.
_RESPONSE_CACHE_ATTR = "_cached_field_response"

#: The same, for the fixed-k response, which is the more expensive of the two to
#: recompute because it re-runs the solve.
_FIXED_K_CACHE_ATTR = "_cached_fixed_k_response"


class FixedKDegenerateWarning(UserWarning):
    """The fixed-k response was replaced by the exact one at a degenerate point."""


class FixedKDegenerateError(ValueError):
    """A fixed-k derivative was asked for where the fixed-k map has none."""


_DEGENERATE_EXPLANATION = (
    "the zero-field projected residual of this solve is EXACTLY zero -- the "
    "seed already solves the system, which happens whenever the configuration "
    "is symmetric enough that chi drives no charge transfer (a homonuclear "
    "diatomic, a monatomic crystal, any site-equivalent structure). CG's step "
    "sizes are ratios of residual norms, so the k-step map m_k(c) is "
    "homogeneous but NOT differentiable at c = 0, and autograd returns 0 for "
    "every response derivative there. Measured on a homonuclear diatomic: "
    "fixed-k alpha exactly 0 at k = 6 and at k = 50, true alpha "
    "(0.270, 0.217, 0.205); with a 1e-12 asymmetry the fixed-k value is the "
    "true one, so the defect is a removable discontinuity at the symmetric "
    "point. Forces are unaffected (the solve is exact there)."
)


def _base_residual_vanishes(assembly: ElectrostaticAssembly) -> bool:
    """Whether the production solve starts at an exact solution (SPEC 15.2).

    One CG step of the real solve, at the assembly's own field, read for its
    initial residual -- which ``solve_charges`` clamps at ``sqrt(tiny)`` so an
    exact zero reads exactly that.  A host read; observables only.
    """
    if assembly.config.solve_mode != "fixed_k":
        # SPEC 17: the converged energy is a smooth function of the field; the
        # degenerate point is a property of the truncated Krylov map only.
        return False
    with torch.no_grad():
        if assembly.lattice_operator is not None:
            result = _solve_matrix_free(assembly, 1, assembly.external_field)
            floor = torch.finfo(result.q.dtype).tiny ** 0.5
            return float(result.diagnostics.initial_residual_norm) <= floor * (1.0 + 1e-6)
        geometry = _solver_geometry(assembly)
        seed = GeometrySeed.from_bond_logits(
            assembly.t_seed.detach(), assembly.n_atoms, geometry=geometry
        )
        result = solve_charges(
            assembly.params,
            geometry,
            _fragment_constraints(assembly),
            1,
            seed=seed,
            options=_solver_options(assembly.config),
            external_field=assembly.external_field,
        )
    floor = torch.finfo(result.q.dtype).tiny ** 0.5
    return float(result.diagnostics.initial_residual_norm) <= floor * (1.0 + 1e-6)

#: Cartesian dimension.  Named so that the shape checks in the beta constructor
#: read as a statement about space rather than as a bare 3.
_SPATIAL = 3


class PolarizabilityGap(NamedTuple):
    """The two routes side by side, and the distance between them.

    Attributes
    ----------
    fixed_k:
        ``[3, 3]`` ``-d2U/dE2`` of the energy this package returns.
    minimiser:
        ``[3, 3]`` ``D^T K^-1 D``, the ``k -> infinity`` limit.
    difference:
        ``fixed_k - minimiser``, in the same units as both.
    relative:
        Scalar, ``max|difference| / max|minimiser|``.  This is the number to
        watch: it is the fraction of the converged response that the truncated
        solve has not yet picked up, and it is *not* bounded by the CG residual
        ratio (see :func:`fixed_k_residual`).
    residual_ratio:
        ``||r_k|| / ||r_0||`` from the solve, for comparison.  Correlated with
        ``relative`` but not a bound on it.
    """

    fixed_k: Tensor
    minimiser: Tensor
    difference: Tensor
    relative: Tensor
    residual_ratio: Tensor


class _FixedKResponse(NamedTuple):
    """``-d2U/dE2`` and its two channels, from one differentiated solve."""

    total: Tensor
    charge_transfer: Tensor
    induced: Tensor


def polarization(q: Tensor, pos: Tensor, mu: Tensor | None = None) -> Tensor:
    """Total dipole moment ``P = sum_i (q_i r_i + mu_i)``, in e*A.

    The *moment* definition: the polarization of the charges and dipoles the
    solve returned.  For the polarization the energy defines, ``P = -dU/dE`` --
    which is the same thing only at convergence -- see
    :func:`polarization_from_energy`.

    Args:
        q: ``[n_atoms]`` atomic charges in e, from the solve.
        pos: ``[n_atoms, 3]`` positions in A.  The *same* tensor the energy was
            built from, so that ``P`` is part of the same autograd graph and
            :func:`born_charges` can differentiate it.
        mu: ``[n_atoms, 3]`` induced point dipoles in e*A (physical, not the
            reduced ``mu_hat``).  ``None`` for a charge-only model.

    Returns:
        ``[3]`` polarization vector.

    Note:
        For a charged system ``P`` depends on the choice of origin -- shifting
        every position by ``d`` shifts ``P`` by ``Q_total d``.  That is physics,
        not a defect: only for a neutral system is the dipole moment
        origin-independent.  Both derived quantities in this module are origin
        independent regardless of the net charge.
    """
    total = torch.einsum("i,ia->a", q, pos)
    if mu is not None:
        total = total + mu.sum(dim=0)
    return total


def polarization_from_energy(
    assembly: ElectrostaticAssembly, *, create_graph: bool = False
) -> Tensor:
    """``P = -dU/dE``, the polarization the model's energy defines, in e*A.

    Differentiated through the fixed-k solve, so this is the field derivative of
    the number the package actually returns as the energy.  It equals the moment
    sum :func:`polarization` only when the solve has converged: at finite ``k``
    the iterate is not stationary, the envelope theorem does not apply, and the
    difference is the term ``(dU/dx) . (dx_k/dE)`` that a converged solve kills.
    Measured on the four-atom contact geometry, ``|P_energy - P_moments|``:
    1.2e-01 e*A at ``k = 1``, 3.4e-02 at ``k = 6``, 5.6e-08 at ``k = 30``.

    Use the moment sum when the moments are the observable (a reported dipole, a
    charge-model fit); use this when the quantity has to be consistent with the
    forces -- an infrared intensity, a field-dependent energy, anything compared
    against ``-d2U/dE2``.

    Args:
        assembly: the assembly to evaluate.  Open or periodic (under PBC the
            bond-current polarization of SPEC 15).
        create_graph: keep the graph, so the result can be differentiated again
            (for training against a measured dipole, say).

    Returns:
        ``[3]`` polarization vector in e*A.
    """
    base = _base_field(assembly)
    energy = _fixed_k_energy(assembly, base, base)
    gradient = _grad(energy, base, create_graph=create_graph)
    return -gradient


def polarizability(
    solve_result: SolveResult,
    assembly: ElectrostaticAssembly | None = None,
    *,
    route: str = DEFAULT_ROUTE,
    as_volume: bool = False,
    create_graph: bool = False,
) -> Tensor:
    """Linear polarizability of the solved system, ``[3, 3]``.

    Two routes, and the choice is physics rather than numerics:

    ``route="fixed_k"`` (the default)
        ``alpha = -d2U/dE2`` where ``U`` is the fixed-k energy of SPEC 2 -- the
        one this package returns, the one the forces are the gradient of.  The
        uniform field is threaded through the production solve and the result
        differentiated twice, so the identity holds by construction at every
        ``k``, not merely in the converged limit.

    ``route="minimiser"``
        ``alpha = D^T K^-1 D``, the response of the exact minimiser, taken
        subject to the per-fragment charge constraints.  This is the
        ``k -> infinity`` limit.  It is the right number to quote for the
        *quadratic form*, it is what Maxwell reciprocity has to be tested on
        (see :func:`reciprocity_residual`), and at the shipped ``k = 6`` it can
        be a factor of two away from what the model's forces produce.  See the
        module docstring for the measured table.

    :func:`polarizability_gap` returns both and the distance between them, which
    is the honest convergence diagnostic for a given solve.

    Args:
        solve_result: the result of the solve.  Required, and checked: on the
            fixed-k route the iteration count it carries must match the config
            the assembly was built with, since a response taken at a different
            ``k`` from the reported energy is exactly the defect this route
            exists to remove.
        assembly: the assembly that produced ``solve_result``.  Required in
            practice: a ``SolveResult`` does not carry one.
        route: ``"fixed_k"`` or ``"minimiser"``; see above.
        as_volume: return a polarizability volume in A^3 instead of the raw
            response in e^2 A^2 / eV.  The two differ by ``K_E = 14.4``, so
            comparing the raw response against a literature volume is a 14x
            error, not a subtle one (SPEC 0).
        create_graph: fixed-k route only -- keep the graph so alpha can be
            differentiated (training against a measured polarizability).  The
            minimiser route is differentiable either way, being a plain
            ``linalg.solve`` on tensors that already carry the graph.

    Returns:
        ``[3, 3]`` polarizability tensor.

    Periodic assemblies (SPEC 15): alpha is the CELL polarizability at fixed
    macroscopic field -- the electric enthalpy through the bond-current
    polarization on the fixed-k route, ``D^T K^-1 D`` with bond-current rows of
    ``D`` on the minimiser route.  :func:`dielectric_tensor` turns it into
    ``eps_inf``.

    Raises:
        ValueError: for an unknown ``route``, or a solve whose iteration count
            disagrees with the assembly's config.
    """
    assembly = _require_assembly(solve_result, assembly)
    if _require_route(route) == FIXED_K and _base_residual_vanishes(assembly):
        warnings.warn(
            "fixed-k polarizability replaced by the minimiser route: "
            + _DEGENERATE_EXPLANATION,
            FixedKDegenerateWarning,
            stacklevel=2,
        )
        route = MINIMISER
    if route == MINIMISER:
        response = _field_response(assembly)
        alpha = assembly.dipole_derivative().transpose(0, 1) @ response
    else:
        _require_matching_iterations(solve_result, assembly)
        alpha = _fixed_k_response(assembly, create_graph=create_graph).total
    return to_polarizability_volume(alpha) if as_volume else alpha


def polarizability_fixed_k(
    solve_result: SolveResult,
    assembly: ElectrostaticAssembly | None = None,
    *,
    as_volume: bool = False,
    create_graph: bool = False,
) -> Tensor:
    """``alpha = -d2U/dE2`` of the fixed-k energy.  The default route, by name.

    Identical to ``polarizability(..., route=FIXED_K)``.  It exists so that code
    which cares *which* response it is using can say so at the call site instead
    of relying on a default that a reader has to look up.
    """
    return polarizability(
        solve_result,
        assembly,
        route=FIXED_K,
        as_volume=as_volume,
        create_graph=create_graph,
    )


def polarizability_minimiser(
    solve_result: SolveResult,
    assembly: ElectrostaticAssembly | None = None,
    *,
    as_volume: bool = False,
) -> Tensor:
    """``alpha = D^T K^-1 D``, the response of the exact minimiser.

    The ``k -> infinity`` limit of :func:`polarizability_fixed_k`, and a
    legitimate diagnostic in its own right: it is the polarizability of the
    quadratic form, independent of how many CG steps were taken, so the distance
    between it and the fixed-k response measures the truncation directly
    (:func:`polarizability_gap`).

    The inverse is taken subject to the per-fragment charge constraints, since
    an applied field may not change the net charge of a fragment.  Concretely,
    the response solves the saddle-point system

        [ K   A^T ] [ dx/dE   ]   [ D ]
        [ A    0  ] [ dlam/dE ] = [ 0 ]

    and ``alpha = D^T dx/dE``.  The operator is
    :meth:`ElectrostaticAssembly.dense_hessian`, assembled from the stored pair
    blocks and deliberately **not** symmetrised: a Thole damping factor keyed on
    the wrong atom of a pair leaves ``K != K^T`` and shows up here as an
    asymmetric alpha.  That is the one place in this module where such a bug is
    visible as an asymmetry, so this is the route
    :func:`reciprocity_residual` uses.
    """
    return polarizability(solve_result, assembly, route=MINIMISER, as_volume=as_volume)


def polarizability_contributions(
    solve_result: SolveResult,
    assembly: ElectrostaticAssembly | None = None,
    *,
    route: str = DEFAULT_ROUTE,
    as_volume: bool = False,
    create_graph: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Split alpha into the charge-transfer and induced-dipole channels.

    Returns ``(alpha_charge_transfer, alpha_induced_dipole)``, which sum exactly
    to :func:`polarizability` **on the same route** -- measured agreement
    1.1e-16 relative on the minimiser route (one factorisation), 2.2e-16 on
    the legacy fixed-k route (one graph), and to the solve tolerance (~5e-11)
    on the default converged route, where each channel is its own adjoint
    solve.  Ask
    for the same ``route`` in both calls or the two will not add up, and the
    default is the same for both so that they do by default.

    The split is over the channel the polarization comes out of, never over the
    response itself: both terms are computed with the fully coupled solve, so
    each already contains the effect of the other channel.  Neither half is
    separately symmetric -- only the sum is -- and neither is separately a
    polarizability of any physical subsystem.  Use this for analysis, never as
    two models to be reweighted.

    On the minimiser route this is the split of the rows of ``D`` into charge
    rows (``dP/dt``, ``dP/drho``) and dipole rows (``dP/dmu_hat``).  On the
    fixed-k route the same split is realised by giving the two channels their
    own fields: the charge channel couples through ``chi_i -> chi_i - E_ct . r_i``
    and the dipole channel through the reduced-moment linear term ``-ell_i
    E_ind``, both threaded through the real solve, and then

        alpha_ct  = -d2U / dE_ct dE ,    alpha_ind = -d2U / dE_ind dE

    with the second derivative taken along ``E_ct = E_ind = E``.  The chain rule
    makes the two sum to ``-d2U/dE2`` identically, which is why the sum is exact
    rather than merely close, and in the converged limit each reduces to the
    corresponding row split above.

    Calling this and :func:`polarizability` on the same assembly costs one solve
    (fixed-k) or one dense factorisation (minimiser), not two -- both are
    memoised on the assembly.
    """
    assembly = _require_assembly(solve_result, assembly)
    if _require_route(route) == FIXED_K and _base_residual_vanishes(assembly):
        warnings.warn(
            "fixed-k polarizability split replaced by the minimiser route: "
            + _DEGENERATE_EXPLANATION,
            FixedKDegenerateWarning,
            stacklevel=2,
        )
        route = MINIMISER
    if route == MINIMISER:
        response = _field_response(assembly)
        dipole_derivative = assembly.dipole_derivative()
        n_charge_rows = assembly.n_bonds + assembly.n_atoms
        charge = (
            dipole_derivative[:n_charge_rows].transpose(0, 1)
            @ response[:n_charge_rows]
        )
        induced = (
            dipole_derivative[n_charge_rows:].transpose(0, 1)
            @ response[n_charge_rows:]
        )
    else:
        _require_matching_iterations(solve_result, assembly)
        fixed_k = _fixed_k_response(assembly, create_graph=create_graph)
        charge, induced = fixed_k.charge_transfer, fixed_k.induced
    if as_volume:
        return to_polarizability_volume(charge), to_polarizability_volume(induced)
    return charge, induced


def polarizability_gap(
    solve_result: SolveResult,
    assembly: ElectrostaticAssembly | None = None,
    *,
    as_volume: bool = False,
) -> PolarizabilityGap:
    """Both routes and the distance between them -- the convergence telemetry.

    The gap is not an inconsistency to be tolerated; it is a measurement of how
    far the truncated solve is from the minimiser, expressed in the observable
    the user cares about rather than in the CG residual.  That matters because
    the residual ratio is *not* a bound on it: on an eight-atom cluster the
    residual ratio falls monotonically with ``k`` while the relative gap goes
    72.4%, 6.3%, 10.8%, 3.5%, 20.1% at ``k = 1, 2, 3, 4, 6``.  If a workflow
    needs the converged response, raise ``k`` until this number is small and
    accept the cost -- do not report the minimiser route as though the dynamics
    produced it.

    Both tensors are in the same units, selected by ``as_volume`` (SPEC 0:
    e^2 A^2 / eV against A^3, a factor of ``K_E = 14.4``).  ``relative`` is
    dimensionless and unaffected.

    Returns:
        :class:`PolarizabilityGap`.
    """
    assembly = _require_assembly(solve_result, assembly)
    fixed_k = polarizability(
        solve_result, assembly, route=FIXED_K, as_volume=as_volume
    )
    minimiser = polarizability(
        solve_result, assembly, route=MINIMISER, as_volume=as_volume
    )
    difference = fixed_k - minimiser
    scale = minimiser.abs().amax()
    relative = difference.abs().amax() / torch.where(
        scale > 0, scale, torch.ones_like(scale)
    )
    return PolarizabilityGap(
        fixed_k=fixed_k,
        minimiser=minimiser,
        difference=difference,
        relative=relative,
        residual_ratio=fixed_k_residual(solve_result),
    )


def reciprocity_residual(
    solve_result: SolveResult,
    assembly: ElectrostaticAssembly | None = None,
    *,
    as_volume: bool = False,
) -> Tensor:
    """``alpha - alpha^T`` on the minimiser route -- the Maxwell reciprocity check.

    SPEC 6 requires ``alpha == alpha^T`` to 1e-10 and requires it to be measured
    on a close-contact, polarizability-contrasted system, because it is there to
    catch an asymmetric operator: a Thole damping length keyed on the source
    atom instead of the symmetric ``(alpha_i alpha_j)^(1/6)`` gives 14.9%
    asymmetry at 1.2 A contact and 0.005% on a sparse geometry, which would pass
    silently.

    **This deliberately does not use the default route.**  The fixed-k alpha is
    a second derivative of a scalar, so it is symmetric by Clairaut's theorem no
    matter what the operator does -- measured 1.1e-16 relative asymmetry with
    the Thole bug injected, against 1.3e-02 here.  A reciprocity assertion on it
    would be an assertion about calculus, not about the model.  The minimiser
    route inverts the unsymmetrised
    :meth:`ElectrostaticAssembly.dense_hessian`, so the asymmetry survives into
    the answer and the test keeps its teeth.

    What this cannot see, and nothing else in the suite can either: an
    asymmetric operator still changes the *value* of both alphas (measured 4.5%
    on the injected bug), so a wrong-but-symmetric operator is invisible to a
    reciprocity test by construction.  Reciprocity is a necessary condition, not
    a sufficient one.

    Returns:
        ``[3, 3]``, zero to round-off for a symmetric operator.
    """
    alpha = polarizability(solve_result, assembly, route=MINIMISER, as_volume=as_volume)
    return alpha - alpha.transpose(0, 1)


def born_charges(
    polarization_vector: Tensor,
    pos: Tensor,
    *,
    create_graph: bool = False,
) -> Tensor:
    """Born effective charges ``Z*_i = dP / dr_i``, in e, from the moment sum.

    Taken by automatic differentiation of the polarization through the whole
    solve, so ``Z*`` includes the response of the charges and of the induced
    dipoles to the displacement, not only the static term ``q_i delta_ab``.  The
    difference is the entire physics of an infrared intensity.

    This differentiates whatever ``P`` it is handed, which is normally the moment
    sum ``sum_i (q_i r_i + mu_i)`` from :func:`polarization`.  That is the right
    ``P`` when the moments are the observable.  It is *not* the ``P`` the energy
    defines at finite ``k`` -- see :func:`born_charges_from_energy`, which
    returns the mixed derivative ``-d2U/dE dR`` instead and is the one to use
    for an intensity that has to agree with the forces.  The two differ by
    14.9% of the largest component at ``k = 6`` on the four-atom contact
    geometry and agree to 3.0e-07 by ``k = 30``.

    Args:
        polarization_vector: ``[3]`` from :func:`polarization`, built from the
            charges and positions of a graph that was not detached.
        pos: ``[n_atoms, 3]`` the positions that graph depends on, with
            ``requires_grad``.
        create_graph: keep the graph, for a second derivative or for training
            against measured infrared intensities.

    Returns:
        ``[n_atoms, 3, 3]`` with ``Z*[i, a, b] = dP_a / dr_{i,b}``.

    Note:
        Three backward passes, one per Cartesian component of ``P``.  This is an
        observable, not a hot path; do not put it inside an MD step.

        The graph is retained for the first two passes because the third still
        needs it, and released by the third unless ``create_graph`` asks for it.
        Passing ``retain_graph=True`` on the final axis as well -- which is what
        this used to do -- keeps the whole solve's saved tensors alive for as
        long as the caller holds any reference into the graph, which for a
        differentiated fixed-k solve is every intermediate of every CG
        iteration.  Nothing failed; the memory simply was not returned.

        The sum rule is ``sum_i Z*_i = Q_total * identity`` -- see the module
        docstring.  It is *not* zero for a charged system, and this function
        does not enforce it.  Use :func:`acoustic_sum_rule_residual` to check it.
    """
    if polarization_vector.shape != (3,):
        raise ValueError(
            f"polarization_vector must have shape (3,), got {tuple(polarization_vector.shape)}"
        )
    # retain_graph on every axis: the graph belongs to the caller's state (the
    # assembly's pair blocks, positions, heads), not to this function. Freeing
    # it on the last axis made a second observable call on the same state -- a
    # second born_charges_minimiser, say -- die with "backward through the
    # graph a second time".
    rows = [
        torch.autograd.grad(
            polarization_vector[axis],
            pos,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=False,
        )[0]
        for axis in range(3)
    ]
    return torch.stack(rows, dim=1)


def born_charges_from_energy(
    assembly: ElectrostaticAssembly,
    *,
    create_graph: bool = False,
) -> Tensor:
    """``Z*_i = -d2U / dE dr_i``, in e -- the Born charges the energy defines.

    The mixed second derivative of the fixed-k energy: the field derivative of
    the force, equivalently the position derivative of ``P = -dU/dE``.  This is
    the quantity an infrared intensity must be built from if it is to be
    consistent with the dynamics, for exactly the reason alpha is
    ``-d2U/dE2``: at finite ``k`` the iterate is not stationary, so the envelope
    theorem fails and ``dP_moments/dR`` is a different number.  Measured on the
    four-atom contact geometry, ``max|Z*_energy - Z*_moments|``: 1.6e-01 e at
    ``k = 1``, 7.4e-02 at ``k = 6``, 3.0e-07 at ``k = 30``.

    The translational sum rule survives the truncation exactly.  A rigid shift
    of every atom by ``d`` shifts ``chi_i`` by the constant ``-E . d``, and a
    constant shift of ``chi`` leaves every CG iterate untouched (``1^T B = 0``,
    so it cancels out of the bond-sector linear term) while moving ``U`` by
    ``-Q_total E . d``.  Hence ``sum_i Z*_i = Q_total * identity`` at every
    ``k``, not only at convergence -- measured 1.1e-15 e.

    Args:
        assembly: the assembly to differentiate.  Its ``pos`` must carry
            ``requires_grad`` (or descend from a tensor that does), which means
            the caller has to have built the state from a differentiable
            position tensor.  Open or periodic (SPEC 15).
        create_graph: keep the graph, for a second derivative or for training
            against measured infrared intensities.

    Returns:
        ``[n_atoms, 3, 3]`` with ``Z*[i, a, b] = -d2U / dE_a dr_{i,b}``.

    Note:
        Four backward passes over the solve (one for the field, three for the
        Cartesian components) plus the solve itself.  An observable, not a hot
        path.
    """
    pos = assembly.pos
    if not pos.requires_grad:
        raise ValueError(
            "born_charges_from_energy needs an assembly whose positions carry "
            "requires_grad: Z* is a derivative with respect to them. Build the "
            "state from `positions.clone().requires_grad_(True)` and pass its "
            "assembly here."
        )
    if _base_residual_vanishes(assembly):
        raise FixedKDegenerateError(
            "born_charges_from_energy: " + _DEGENERATE_EXPLANATION + " Use "
            "born_charges_minimiser(assembly), the exact response of the "
            "quadratic form, which the fixed-k value tends to."
        )
    base = _base_field(assembly)
    energy = _fixed_k_energy(assembly, base, base)
    gradient = _grad(energy, base, create_graph=True)
    rows = [
        _grad(gradient[axis], pos, retain_graph=True, create_graph=create_graph)
        for axis in range(3)
    ]
    return -torch.stack(rows, dim=1)


def fixed_k_residual(solve_result: SolveResult) -> Tensor:
    """How far the fixed-k iterate is from the minimiser, ``||r_k|| / ||r_0||``.

    The solver's own convergence telemetry, read rather than recomputed: both
    norms are already in :class:`SolveDiagnostics`.

    It used to be documented as the number that says whether
    :func:`polarizability` and the curvature of the reported energy are the same
    quantity.  They now are, by construction, so this measures something
    narrower and more honest: how far the reported response is from the
    ``k -> infinity`` limit, i.e. how much of the *converged* polarizability the
    truncation is leaving on the table.

    Read it as a guide and not a bound.  Measured on an eight-atom cluster, the
    residual ratio against the relative gap between the two routes
    (:func:`polarizability_gap`):

        k                1       2       3       4       6      12      30
        ratio       3.9e-01 1.2e-01 6.9e-02 4.1e-02 1.4e-02 1.5e-04 2.0e-06
        gap           72.4%    6.3%   10.8%    3.5%   20.1% 4.8e-05 1.5e-07

    The ratio falls monotonically; the gap does not, because a CG iterate is not
    a monotone approximation of the minimiser in this norm.  A small ratio does
    imply a small gap; an intermediate one implies very little.

    Returns:
        Scalar in [0, 1]-ish; exactly zero only if the seed was already the
        solution.
    """
    diagnostics = solve_result.diagnostics
    initial = diagnostics.initial_residual_norm
    return diagnostics.residual_norm / torch.where(
        initial > 0, initial, torch.ones_like(initial)
    )


def acoustic_sum_rule_residual(born: Tensor, total_charge: Tensor | float) -> Tensor:
    """``sum_i Z*_i - Q_total * identity``, the quantity that should vanish.

    This is a *check*, and deliberately not a correction.  The usual acoustic
    sum rule ``sum_i Z*_i = 0`` is the neutral special case of
    ``sum_i Z*_i = Q_total * identity``; imposing the neutral form on a charged
    fragment would remove a real translational dipole and corrupt every
    infrared intensity derived from it.  If this residual is large, the fault is
    upstream -- a solve that is not translationally invariant, or a detached
    graph -- and must be fixed there.

    Applies to both Born-charge routes: :func:`born_charges` and
    :func:`born_charges_from_energy` satisfy the identity separately and
    exactly, at every ``k``.

    Returns:
        ``[3, 3]`` residual, which is zero to machine precision when the solve
        is translationally invariant.
    """
    identity = torch.eye(3, dtype=born.dtype, device=born.device)
    charge = (
        total_charge
        if isinstance(total_charge, Tensor)
        else torch.tensor(total_charge, dtype=born.dtype, device=born.device)
    )
    return born.sum(dim=0) - charge * identity


def hyperpolarizability(
    alpha: Tensor | None = None,
    d: Tensor | None = None,
    *,
    octupole: Tensor | None = None,
    form: str = DEFAULT_BETA_FORM,
) -> Tensor:
    """First hyperpolarizability ``beta_abc = d_a alpha_bc``, ``[..., 3, 3, 3]``.

    Not a derivative of this package's energy, and deliberately so: the energy is
    quadratic in the moments (SPEC 1.5), so its exact third derivative is zero
    and its fixed-k third derivative is truncation residue.  ``beta`` is instead
    *constructed*, from the polarizability the model computes and a learned
    ``1o`` direction ``d`` read off the node features by
    ``heads.ElectrostaticHeads(hyperpolarizability=True)``.  See the module
    docstring for the full argument; the three load-bearing points are repeated
    here because this is where somebody will read them.

    **The irreps close on the whole tensor.**  ``alpha`` is ``0e + 2e``, ``d`` is
    ``1o``, and ``(0e + 2e) (x) 1o = 2x1o + 1x2o + 1x3o``: dimension 18, which is
    the full non-Kleinman SHG tensor, symmetric in the two field indices
    ``(b, c)`` and in nothing else (3 x 6 = 18).  The usual ``1o + 3o``
    assignment is dimension 10 and quietly assumes Kleinman symmetry, a
    far-from-resonance approximation -- that is, not the regime an SHG or SFG
    measurement is made in.

    **18 for spectroscopy, 10 for an energy, and never the two confused.**
    ``E_a E_b E_c`` is symmetric, so only the fully symmetric part of ``beta``
    survives contraction with it.  A static-field energy can therefore access at
    most the Kleinman 10; the other 8 components (the second ``1o`` and the
    ``2o``) exist only in frequency-resolved response with ``omega_1 !=
    omega_2``, which a ground-state energy functional does not provide.  Hence:

    ``form="full"`` (:data:`BETA_FULL`, the default)
        All 18 components.  This is beta *as a readout* -- what to compare
        against a measured SHG or SFG tensor, and what to train against
        frequency-resolved labels.

    ``form="kleinman"`` (:data:`BETA_KLEINMAN`)
        The fully symmetric part, ``(1/3)(d_a alpha_bc + d_b alpha_ca + d_c
        alpha_ab)``, 10 independent components.  This is the only part of beta an
        energy can carry, so it is the form to use in any third-order term of a
        static-field energy expansion.  Passing the full tensor there is not an
        error that anything can detect: the contraction silently symmetrises it
        and 8 components disappear from the answer.

    **Zero here means "no polar axis", not always "beta is forbidden".**  This
    construction is non-zero only where the site group admits an invariant
    vector.  At a centrosymmetric site that is exactly the selection rule and
    the result is correct (measured ``|beta| = 3.5e-18`` against ``1.7e-02`` at
    a polar site of comparable ``alpha``).  At a *tetrahedral* site it is not:
    Td admits no invariant vector, so ``d`` and hence ``beta`` vanish, while Td
    does allow one component of ``beta`` - the ``xyz`` term responsible for
    zincblende SHG.  ``heads.py`` tabulates the per-group counts.

    **A non-zero per-site beta in a centrosymmetric crystal is not a bug.**  The
    result is per site, because ``d`` is; the global selection rule is a
    statement about ``sum_i beta_i``, and a centrosymmetric cell generally has
    non-centrosymmetric sites.  Inversion pairs ``i`` with ``i'`` such that
    ``d_i' = -d_i`` and ``alpha_i' = alpha_i``, so ``beta_i' = -beta_i`` and the
    sum vanishes while each term does not (measured on a centrosymmetric cubic
    cell with a polar two-atom basis: per-site ``|beta_i|_F = 2.873804e-01``,
    ``|beta_0 + beta_1|_F = 1.110340e-16``).  A cell-level tensor is therefore
    ``sum_i beta_i`` and never ``alpha_total (x) sum_i d_i`` -- the shape check
    below catches the accidental version of that pairing but cannot catch a
    deliberate one.

    **``d`` is unidentifiable from energies, forces and polarizabilities.**  It
    appears in no energy, force, stress, charge, dipole or ``alpha``, so a model
    fitted to any of those has *exactly zero* gradient on it.  With no beta
    labels in the loss, what comes back here is the head's initialisation:
    correctly equivariant, correctly zero at a centrosymmetric site, and
    otherwise arbitrary in magnitude and sign.  The head warns about this at
    construction, which is the only place it can be said: this function receives
    a bare tensor, and "never fitted" is not a property of a float array.  The
    warning is not weaker for being raised there -- ``d`` cannot be obtained
    except from a head built with ``hyperpolarizability=True``, and that
    constructor warns in every process, loading a checkpoint included.  It is
    still the right construction -- an anharmonic on-site
    energy would instead force ``beta = -2 c alpha^3``, i.e. Miller's rule as an
    identity the model cannot violate and therefore cannot be tested against,
    while an exact two-level model shows beta is not a function of alpha at all.

    Args:
        alpha: ``[..., 3, 3]`` polarizability, from :func:`polarizability`, or
            a per-atom stack of them.  Symmetrised in its two indices here (see
            Note); an asymmetric ``alpha`` is a real defect and
            :func:`reciprocity_residual` is what detects it.
        d: ``[..., 3]`` hyperpolarizability direction, ``readout.hyper.d``.  Its
            leading (batch) shape must match ``alpha``'s exactly.
        form: :data:`BETA_FULL` or :data:`BETA_KLEINMAN`; see above.

    Returns:
        ``[..., 3, 3, 3]``.  Index ``a`` is the emitted (harmonic) direction and
        ``(b, c)`` are the two field indices: ``P_a = beta_abc E_b E_c``.

    Raises:
        NotImplementedError: when ``alpha`` or ``d`` is missing.  There is no
            beta to be had from this model without a ``d``, and in particular
            none to be derived from ``alpha`` alone.
        ValueError: for an unknown ``form``, a wrong trailing shape, or leading
            shapes that do not match.

    Note:
        **Units are whatever the labels made them.**  ``beta`` comes back in the
        units of ``alpha`` times those of ``d``, and ``d`` has no calibrated
        scale of its own -- its magnitude is fixed entirely by the beta data it
        was trained against.  No ``as_volume`` conversion is offered here for
        that reason: unlike ``alpha``, where the ``K_E`` factor between
        ``e^2 A^2 / eV`` and ``A^3`` is a definite 14.4 (SPEC 0), there is no
        definite conversion to offer and inventing one would be a fiction.  Feed
        ``alpha`` in whatever units the answer is wanted in.

        ``alpha`` is symmetrised in ``(b, c)`` before the product.  That is not
        cosmetic: the 18-dimensional count *is* the count of rank-3 tensors
        symmetric in their last two indices, so an asymmetric ``alpha`` would put
        the result outside the space beta lives in.  The minimiser route returns
        a genuinely unsymmetrised ``alpha`` on purpose -- it is the one place a
        Thole bug is visible -- so the symmetrisation is done here rather than
        assumed, and :func:`reciprocity_residual` remains the place to look.
    """
    if alpha is None or d is None:
        raise NotImplementedError(
            "hyperpolarizability(alpha, d) needs BOTH a polarizability and the "
            "learned 1o direction d. beta is not a derivative of this model's "
            "energy -- the energy is quadratic in the moments (SPEC 1.5), so its "
            "exact third derivative is zero and its fixed-k third derivative is "
            "truncation residue, not physics. Nor can beta be derived from alpha "
            "alone: an on-site energy anharmonic in the moments would force "
            "beta = -2 c alpha^3 (Miller's rule as an identity, untestable), and "
            "an exact two-level model refutes it -- at fixed gap and transition "
            "dipole, varying the ground/excited dipole difference leaves alpha "
            "constant to 0.00% while beta runs 0 -> 5.333, with beta/alpha^3 "
            "varying 18x. Build the head with "
            "ElectrostaticHeads(hyperpolarizability=True), take d from "
            "readout.hyper.d, and pass it here with alpha from polarizability(). "
            "Note that d is identifiable ONLY from beta training labels: it "
            "enters no energy, force or polarizability, so without such labels "
            "the result is the head's initialisation."
        )

    form = _require_beta_form(form)
    alpha, d = _require_beta_shapes(alpha, d)

    # beta must be symmetric in its two FIELD indices; alpha is symmetric in
    # exact arithmetic and on the minimiser route deliberately not symmetrised,
    # so take the symmetric part rather than trusting it.
    alpha = 0.5 * (alpha + alpha.transpose(-1, -2))

    # beta_abc = d_a alpha_bc. The vector index leads, so that the result reads
    # as P_a = beta_abc E_b E_c in the SHG convention rather than needing a
    # transpose at every call site.
    beta = torch.einsum("...a,...bc->...abc", d, alpha)

    if octupole is not None:
        # The same check `_require_beta_shapes` makes for (alpha, d), for the
        # same reason and against the same failure. A system-level [3,3,3]
        # octupole added to a per-atom beta broadcasts to a well-shaped,
        # correctly equivariant, entirely meaningless result in which every
        # atom carries the whole system's octupole -- verified: two atoms came
        # back bit-identical to the supplied system tensor, with no raise,
        # while the equivalent (alpha, d) mismatch raises ValueError. Requiring
        # equality rather than broadcastability is what makes it a raise.
        expected = tuple(alpha.shape[:-2]) + (_SPATIAL, _SPATIAL, _SPATIAL)
        if tuple(octupole.shape) != expected:
            raise ValueError(
                f"octupole has shape {tuple(octupole.shape)}, expected "
                f"{expected} to match alpha {tuple(alpha.shape)}. The batch "
                "dimensions must be EQUAL, not merely broadcastable: a system "
                "[3, 3, 3] octupole against a per-atom alpha would broadcast "
                "to a correctly-shaped, correctly-equivariant result in which "
                "every atom carries the whole system's octupole, and nothing "
                "downstream would catch it."
            )
        # Zyss dipolar/octupolar split.  `alpha (x) d` already spans a J=3
        # component, so simply ADDING an independent octupolar channel would
        # parameterise that sector twice and leave flat directions in the loss.
        # Remove the J=3 part of the product first, then substitute.
        #
        # What this buys: `d` is a 1o object and vanishes IDENTICALLY at a
        # tetrahedral site (Td's vector representation is T2, which contains no
        # A1), so `alpha (x) d` alone predicts exactly zero second-harmonic
        # response for zincblende -- GaAs, ZnS, ZnSe -- at equilibrium. The
        # octupolar channel is non-zero there and still odd under inversion, so
        # it closes that blind spot without touching surface selectivity.
        symmetric = _symmetrise_rank3(beta)
        beta = beta - (symmetric - _dipolar_rank3(symmetric)) + octupole
    if form == BETA_FULL:
        return beta

    # The fully symmetric part, written out as the three cyclic terms rather
    # than as permutations of `beta`, because a permute() of a batched rank-3
    # tensor is exactly the kind of index bookkeeping that is wrong once and
    # then wrong forever. The average over all six elements of S3 collapses to
    # these three: beta is already symmetric in (b, c), so each of the other
    # three permutations equals one of these term for term.
    return (
        beta
        + torch.einsum("...b,...ca->...abc", d, alpha)
        + torch.einsum("...c,...ab->...abc", d, alpha)
    ) / 3.0


def hyperpolarizability_readout(alpha: Tensor, d: Tensor) -> Tensor:
    """All 18 components of ``beta``.  The spectroscopic tensor, by name.

    Identical to ``hyperpolarizability(alpha, d, form=BETA_FULL)``.  It exists so
    that code which means the full non-Kleinman tensor says so at the call site,
    rather than relying on a default a reader has to look up -- the same reason
    :func:`polarizability_fixed_k` exists.
    """
    return hyperpolarizability(alpha, d, form=BETA_FULL)


def hyperpolarizability_energy(alpha: Tensor, d: Tensor) -> Tensor:
    """The fully symmetric (Kleinman) part of ``beta``: 10 of the 18 components.

    Identical to ``hyperpolarizability(alpha, d, form=BETA_KLEINMAN)``, and the
    form to use in any third-order term of a static-field energy expansion --
    ``U ->  U - (1/6) beta_abc E_a E_b E_c``.  Because ``E_a E_b E_c`` is
    symmetric, that contraction sees only this part whatever it is handed, so
    using the full tensor there is a silent 8-component discrepancy between the
    beta a model reports and the beta its energy implements, not an error
    anything raises.

    This is also the reason a ground-state energy functional cannot supply the
    other 8 components at all: they live in frequency-resolved response with
    ``omega_1 != omega_2``, which a static field does not have.
    """
    return hyperpolarizability(alpha, d, form=BETA_KLEINMAN)


# ---------------------------------------------------------------------------
# Internals -- the fixed-k route
# ---------------------------------------------------------------------------


def _base_field(assembly: ElectrostaticAssembly) -> Tensor:
    """The field to differentiate about, as a fresh leaf, ``[3]``.

    The assembly's own field when it has one -- the response of a system already
    sitting in a finite field is the response *at that field*, and at finite
    ``k`` the energy is not exactly quadratic in ``E``, so the two are genuinely
    different numbers -- and zero otherwise.

    Detached on purpose.  A caller who built the state from a field tensor that
    carries a graph wants the derivative with respect to the field itself, not
    with respect to whatever produced it.
    """
    field = assembly.external_field
    zero = torch.zeros(3, dtype=assembly.pos.dtype, device=assembly.pos.device)
    if field is not None:
        zero = field.reshape(-1).detach().to(zero.dtype).clone()
    return zero.requires_grad_(True)


def _fixed_k_energy(
    assembly: ElectrostaticAssembly, field_charge: Tensor, field_dipole: Tensor
) -> Tensor:
    """``U`` of the production path, with the two field channels separated.

    Re-runs *the real solve* -- ``solve_charges`` on the geometry, constraints
    and seed this assembly carries, at the assembly's own ``n_cg_iter`` -- and
    evaluates :meth:`ElectrostaticAssembly.energy` at the moments it returns.
    Nothing here re-derives the CG recurrence or the field coupling; that is the
    point.  A second implementation of either would be free to drift away from
    the energy whose derivative this is supposed to be, which is the precise
    failure the fixed-k route exists to fix.

    ``field_charge`` couples to ``sum_i q_i r_i`` and ``field_dipole`` to
    ``sum_i mu_i``.  Passing the same tensor twice is the physical case, a
    uniform field on everything; passing two independent tensors is what
    :func:`polarizability_contributions` differentiates separately, and it is
    realised without a second energy expression:

    * ``assemble_system`` couples a supplied ``external_field`` to *both*
      sectors, as ``chi_i -> chi_i - E . r_i`` and as a ``-ell_i E`` term on the
      reduced dipole.  So ``external_field = field_dipole`` gives the dipole
      channel what it needs and over-shifts ``chi`` by ``-field_dipole . r_i``;
    * pre-shifting ``chi`` by ``-(field_charge - field_dipole) . r_i`` cancels
      that exactly, leaving ``chi_i - field_charge . r_i``.

    The same pre-shifted ``chi`` then makes :meth:`ElectrostaticAssembly.energy`
    report ``U_0 - field_charge . P_charge - field_dipole . P_dipole`` with no
    hand-written coupling term, because its ``chi . q`` and its ``-P . E`` term
    do the arithmetic.  When the two fields are equal the shift is exactly zero
    and this is bit-for-bit the ordinary single-field path.
    """
    # `cell is not None` and not only `periodic`: an assembly may not carry an
    # external field together with a cell at all -- ElectrostaticAssembly refuses
    # the combination in __post_init__ -- so threading a field through one is not
    # merely wrong physics here, it is unconstructible. Refusing in this module's
    # own words beats letting `dataclasses.replace` raise a ValueError about an
    # argument the caller never passed.
    if assembly.periodic or assembly.lattice_operator is not None:
        # Matrix-free assemblies (periodic, or open through OpenOperator) carry
        # the bond-current coupling, which under open boundaries IS the
        # chi -> chi - E.r shift written through B (SPEC 15).
        return _fixed_k_enthalpy_periodic(assembly, field_charge, field_dipole)
    params = assembly.params
    offset = field_charge - field_dipole
    shifted = dataclasses.replace(
        assembly,
        params=dataclasses.replace(params, chi=params.chi - assembly.pos @ offset),
        external_field=field_dipole,
    )

    geometry = _solver_geometry(shifted)
    constraints = FragmentConstraints(
        matrix=shifted.constraint, targets=shifted.fragment_charges
    )
    # SPEC 2: geometry-determined seed, bound to this geometry, exactly as
    # energy.electrostatic_state builds it. The field does not enter the seed --
    # it must not, or U would stop being a function of the geometry alone.
    seed = GeometrySeed.from_bond_logits(
        shifted.t_seed, shifted.n_atoms, geometry=geometry
    )
    result = solve_charges(
        shifted.params,
        geometry,
        constraints,
        shifted.config.n_cg_iter,
        seed=seed,
        options=_solver_options(shifted.config),
        external_field=shifted.external_field,
    )
    if shifted.config.solve_mode != "fixed_k":
        # SPEC 17: the solver's minimum-energy node -- the same functional as
        # shifted.energy(moments), with the envelope-theorem first derivative.
        return result.energy
    return shifted.energy(*_moments(result, shifted))


def _fixed_k_enthalpy_periodic(
    assembly: ElectrostaticAssembly, field_charge: Tensor, field_dipole: Tensor
) -> Tensor:
    """``U - E_c . Omega P_charge - E_d . Omega P_dipole`` for a periodic cell (SPEC 15).

    The Nunes-Gonze electric enthalpy of this model: the production solve at
    the assembly's own ``n_cg_iter``, with the charge sector coupled through the
    image-resolved bond currents (``charge_field``) and the dipole sector
    through ``-ell_i E``.  Tinfoil boundaries hold the macroscopic field at the
    applied one, which is exactly the fixed-E condition the enthalpy needs.
    """
    if assembly.lattice_operator is not None:
        result = _solve_matrix_free(
            assembly, assembly.config.n_cg_iter, field_dipole, charge_field=field_charge
        )
    else:
        geometry = _solver_geometry(assembly)
        seed = GeometrySeed.from_bond_logits(assembly.t_seed, assembly.n_atoms, geometry=geometry)
        result = solve_charges(
            assembly.params,
            geometry,
            _fragment_constraints(assembly),
            assembly.config.n_cg_iter,
            seed=seed,
            options=_solver_options(assembly.config),
            external_field=field_dipole,
            charge_field=field_charge,
        )
    if assembly.config.solve_mode != "fixed_k":
        # SPEC 17: the solve's own energy IS this enthalpy -- its linear term
        # carries -E_c . (rho r + tau d) and -E_d . ell, its constant
        # -E_c . rho r -- with the envelope-theorem first derivative.
        return result.energy
    moments = _moments(result, assembly)
    unfielded = dataclasses.replace(assembly, external_field=None)
    energy = unfielded.energy(*moments)
    t, rho, mu_hat = moments[0], moments[1], moments[2]
    charge_dipole = rho @ assembly.pos + t @ assembly.bond_dipole
    return (
        energy
        - torch.dot(field_charge, charge_dipole)
        - torch.dot(field_dipole, assembly.dipoles(mu_hat).sum(dim=0))
    )


def _fixed_k_response(
    assembly: ElectrostaticAssembly, *, create_graph: bool = False
) -> _FixedKResponse:
    """``-d2U/dE2`` and its two channels, from one differentiated solve.

    One forward solve with three independent field leaves -- the physical field
    ``E`` and a zero offset on each channel -- then nine backward passes: three
    for the total and three for each channel.  The channel derivatives are taken
    with respect to the offsets and then along the diagonal, so the chain rule
    guarantees ``charge_transfer + induced == total`` identically rather than
    approximately (measured 2.2e-16 relative).

    Memoised on the assembly when no graph is kept, for the same reason
    :func:`_field_response` is: the assembly is frozen, immutable and created
    fresh per evaluation, so a value cached on it cannot go stale.  A
    ``create_graph`` result is never cached and never served from the cache --
    it holds the whole CG graph alive, and a caller who did not ask for that
    must not be handed it.
    """
    cached = getattr(assembly, _FIXED_K_CACHE_ATTR, None)
    if cached is not None and not create_graph:
        return cached

    base = _base_field(assembly)
    offset_charge = torch.zeros_like(base).requires_grad_(True)
    offset_dipole = torch.zeros_like(base).requires_grad_(True)
    energy = _fixed_k_energy(assembly, base + offset_charge, base + offset_dipole)

    # dU/dE along the diagonal, and the two channel derivatives at zero offset.
    # The second and third sum to the first, which is what makes the split exact.
    gradients = torch.autograd.grad(
        energy,
        (base, offset_charge, offset_dipole),
        create_graph=True,
        allow_unused=True,
    )
    gradients = tuple(
        torch.zeros_like(base) if g is None else g for g in gradients
    )

    def jacobian(gradient: Tensor) -> Tensor:
        """``-d/dE`` of a 3-vector, row by row, in the ``[a, b]`` convention."""
        rows = [
            _grad(gradient[axis], base, retain_graph=True, create_graph=create_graph)
            for axis in range(3)
        ]
        return -torch.stack(rows, dim=0)

    response = _FixedKResponse(
        total=jacobian(gradients[0]),
        charge_transfer=jacobian(gradients[1]),
        induced=jacobian(gradients[2]),
    )
    if not create_graph:
        # The dataclass is frozen; this is the documented way past that, and the
        # attribute is private and read only through this function.
        object.__setattr__(assembly, _FIXED_K_CACHE_ATTR, response)
    return response


def _grad(
    output: Tensor,
    inputs: Tensor,
    *,
    retain_graph: bool = True,
    create_graph: bool = False,
) -> Tensor:
    """``torch.autograd.grad`` with an unused input materialised as zeros.

    A quantity that does not depend on the field at all -- a system with no
    charges and no dipoles, which is degenerate but constructible -- makes
    ``grad`` return ``None`` rather than a zero tensor.  Zero is the right
    answer there and ``allow_unused=False`` would raise instead.
    """
    result = torch.autograd.grad(
        output,
        inputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )[0]
    return torch.zeros_like(inputs) if result is None else result


# ---------------------------------------------------------------------------
# Internals -- the minimiser route
# ---------------------------------------------------------------------------


def _field_response(assembly: ElectrostaticAssembly) -> Tensor:
    """``dx/dE``, the moment response to a uniform field, ``[moment_dim, 3]``.

    Solves the constrained saddle-point system in one dense factorisation.  The
    operator comes from :meth:`ElectrostaticAssembly.dense_hessian`, which is
    assembled from the stored pair blocks rather than from autograd on the
    energy -- an autograd Hessian is symmetric by construction and would hide
    the asymmetry this is meant to expose.

    Memoised on the assembly.  :func:`polarizability` and
    :func:`polarizability_contributions` are two views of one response, and both
    used to build the dense Hessian and factorise it from scratch, so asking for
    the total and the split -- which any analysis of the two channels does --
    cost two cubic solves for one answer.  The assembly is frozen, immutable and
    created fresh per evaluation, so a value cached on it cannot go stale: the
    only way to change the operator is to build a new assembly, which carries no
    cache.  The stored tensor keeps the same autograd graph the assembly already
    holds through ``pos``, so it adds no new class of retention.
    """
    cached = getattr(assembly, _RESPONSE_CACHE_ATTR, None)
    if cached is not None:
        return cached
    response = _solve_field_response(assembly)
    # The dataclass is frozen; this is the documented way past that, and the
    # attribute is private and read only through this function.
    object.__setattr__(assembly, _RESPONSE_CACHE_ATTR, response)
    return response


def _solve_field_response(assembly: ElectrostaticAssembly) -> Tensor:
    """The factorisation itself.  See :func:`_field_response` for the caching."""
    hessian = assembly.dense_hessian()
    constraint = assembly.constraint_operator()
    dipole_derivative = assembly.dipole_derivative()

    # rho is a FIXED constraint carrier, not a variational moment (SPEC 1.3):
    # the solver varies only (t, mu_hat).  Letting rho relax here would report a
    # response through a channel the model does not have -- measured as a stable
    # 1.0e-03 relative disagreement with the finite-field alpha, which is the
    # truthful number because it is what the solve actually does.  Pin it.
    n_bonds, n_atoms = assembly.n_bonds, assembly.n_atoms
    free = torch.cat(
        (
            torch.arange(n_bonds, device=hessian.device),
            torch.arange(n_bonds + n_atoms, hessian.shape[0], device=hessian.device),
        )
    )
    full_dim = hessian.shape[0]
    hessian = hessian.index_select(0, free).index_select(1, free)
    constraint = constraint.index_select(1, free)
    dipole_derivative_full = dipole_derivative
    dipole_derivative = dipole_derivative.index_select(0, free)

    # The constraint restricted to the variational sector is ALWAYS rank
    # deficient: 1^T B == 0 makes its rows sum to zero on the bond block.  A
    # magnitude threshold cannot handle that -- dropping rows by
    # ``abs().amax() > 0`` discarded every constraint in every system (the bond
    # block used to be hard-coded to zero), which reported the fully
    # unconstrained polarizability: measured 69% high on a 6-atom two-fragment
    # chain and 309% high on a 10-atom five-fragment chain.  Keeping the rows
    # instead makes the saddle singular (smallest singular value 2.19e-34).
    #
    # So reduce onto an orthonormal basis of the constraint null space, which is
    # rank revealing by construction and is what the solver's own projector does.
    basis = null_space_basis(constraint, hessian.shape[0], dtype=hessian.dtype,
                             device=hessian.device)
    reduced_hessian = basis.transpose(0, 1) @ hessian @ basis
    reduced_rhs = basis.transpose(0, 1) @ dipole_derivative
    reduced = basis @ torch.linalg.solve(reduced_hessian, reduced_rhs)

    # Scatter back into the full moment layout, with drho/dE = 0 by construction.
    response = dipole_derivative_full.new_zeros((full_dim, 3))
    response.index_copy_(0, free, reduced)
    return response


# ---------------------------------------------------------------------------
# Internals -- argument checking
# ---------------------------------------------------------------------------


def _require_route(route: str) -> str:
    """Validate the route name and return it."""
    if route not in _ROUTES:
        raise ValueError(
            f"unknown polarizability route {route!r}; expected one of "
            f"{list(_ROUTES)}. {FIXED_K!r} is -d2U/dE2 of the fixed-k energy "
            "this package returns (the default, and the response the model's "
            f"own forces produce); {MINIMISER!r} is D^T K^-1 D, the response of "
            "the exact minimiser, which is the k -> infinity limit and the "
            "route Maxwell reciprocity has to be tested on."
        )
    return route


def _require_beta_form(form: str) -> str:
    """Validate the hyperpolarizability form and return it."""
    if form not in _BETA_FORMS:
        raise ValueError(
            f"unknown hyperpolarizability form {form!r}; expected one of "
            f"{list(_BETA_FORMS)}. {BETA_FULL!r} is all 18 components "
            "(2x1o + 1x2o + 1x3o), symmetric in the two field indices only -- "
            f"the spectroscopic SHG tensor. {BETA_KLEINMAN!r} is the fully "
            "symmetric part, 10 components (1o + 3o), which is the only part a "
            "static-field energy can carry, because E (x) E (x) E is symmetric. "
            "The remaining 8 exist only in frequency-resolved response with "
            "omega_1 != omega_2."
        )
    return form


def _require_beta_shapes(alpha: Tensor, d: Tensor) -> Tuple[Tensor, Tensor]:
    """Check that ``alpha`` and ``d`` describe the same thing, and return them.

    The trailing shapes are the easy half.  The leading shapes are the half that
    matters: ``einsum`` would happily broadcast a single ``[3, 3]`` system
    polarizability against a ``[n_atoms, 3]`` per-atom ``d`` and return an
    ``[n_atoms, 3, 3, 3]`` stack in which every atom carries the *whole
    system's* alpha.  That is a physically meaningless object which is
    nonetheless the right shape, the right dtype and correctly equivariant, so
    nothing downstream would catch it.  Requiring the batch shapes to be equal
    rather than broadcastable is what makes the mistake a raise.
    """
    if alpha.shape[-2:] != (_SPATIAL, _SPATIAL):
        raise ValueError(
            f"alpha must have trailing shape [3, 3], got {tuple(alpha.shape)}"
        )
    if alpha.dim() < 2 or d.dim() < 1 or d.shape[-1] != _SPATIAL:
        raise ValueError(
            f"d must have trailing shape [3], got {tuple(d.shape)}"
        )
    if alpha.shape[:-2] != d.shape[:-1]:
        raise ValueError(
            f"alpha and d describe different things: alpha has leading shape "
            f"{tuple(alpha.shape[:-2])} and d has {tuple(d.shape[:-1])}. These "
            "must match exactly, not merely broadcast -- pairing one system "
            "alpha [3, 3] with a per-atom d [n_atoms, 3] would broadcast to a "
            "well-shaped, correctly equivariant and entirely meaningless "
            "[n_atoms, 3, 3, 3] in which every atom carries the whole system's "
            "polarizability. Use a per-atom alpha with a per-atom d, or reduce "
            "d to one vector for one alpha."
        )
    return alpha, d


def _require_matching_iterations(
    solve_result: SolveResult, assembly: ElectrostaticAssembly
) -> None:
    """Refuse a solve taken at a different ``k`` from the assembly's config.

    The fixed-k route re-runs the solve at ``assembly.config.n_cg_iter``,
    because that is what the energy of this assembly was evaluated at.  If the
    supplied ``SolveResult`` was produced at some other iteration count then the
    caller is holding an energy from one ``k`` and a response from another --
    which is, in miniature, exactly the inconsistency this route exists to
    remove, so it raises rather than quietly answering the wrong question.
    """
    if assembly.config.solve_mode != "fixed_k":
        return  # SPEC 17: the converged energy has no iteration count to match.
    solved = solve_result.diagnostics.n_iter
    configured = assembly.config.n_cg_iter
    if solved != configured:
        raise ValueError(
            f"this solve ran {solved} CG iterations but the assembly's config "
            f"says {configured}. The fixed-k response is -d2U/dE2 of the energy "
            "at the assembly's own iteration count, so the two must agree or "
            "the reported alpha belongs to a different energy than the reported "
            "U. Rebuild the state with one config, or ask for "
            f"route={MINIMISER!r}, which is k-independent."
        )


def _require_assembly(
    solve_result: SolveResult, assembly: ElectrostaticAssembly | None
) -> ElectrostaticAssembly:
    """Check both arguments and return the assembly.

    Two things used to be wrong here and they compounded.  The docstring of
    ``polarizability`` said ``assembly`` was "optional only if the solve result
    carries one", and this function looked for ``solve_result.assembly`` --
    which ``SolveResult`` does not have and never had, so the lookup was dead
    and ``assembly`` was in fact mandatory.  Meanwhile ``solve_result`` itself
    was never read, so ``polarizability(None, assembly)`` returned a perfectly
    good tensor from a call that names no solve at all.  Between them, the
    signature described the opposite of the code in both arguments.

    Both are required now, and both are checked here so that the two public
    entry points cannot drift apart on it.  On the default route the solve
    result is load-bearing as well as checked: its iteration count has to match
    the assembly's (see :func:`_require_matching_iterations`).
    """
    if assembly is None:
        raise ValueError(
            "polarizability needs the assembly that produced this solve result: "
            "it is built from the operator K, which the moments alone do not "
            "determine, and SolveResult does not carry one. Call "
            "trace_electrostatics.electrostatic_state(...) and pass its "
            "`assembly` field as the second argument."
        )
    if not isinstance(solve_result, SolveResult):
        raise TypeError(
            "the first argument must be the SolveResult this assembly was "
            f"solved to, got {type(solve_result).__name__}. The default route "
            "re-runs that solve with a field on it, so the iteration count the "
            "result carries has to match the assembly's config; a call that "
            "cannot name its solve cannot be checked for that, and on the "
            "minimiser route it would be reporting the k -> infinity limit of a "
            "solve it never saw."
        )
    return assembly


# ---------------------------------------------------------------------------
# SPEC 15 -- the crystal's dielectric response
# ---------------------------------------------------------------------------


def bond_polarization(solve_result: SolveResult, assembly: ElectrostaticAssembly) -> Tensor:
    """The cell dipole ``Omega P``, ``[3]`` in e*A, single valued under PBC.

    ``Omega P = sum_i rho_i r_i + sum_b f_b tau_b d_b + sum_i mu_i`` with
    ``d_b`` the image-resolved bond vector (SPEC 15).  Under open boundaries
    ``f_b d_b = (B^T r)_b`` and this is exactly ``sum_i (q_i r_i + mu_i)``.
    Under PBC it is unchanged when an atom is wrapped into another image, while
    ``sum_i q_i r_i`` jumps by ``q_i R``; it also records the divergence-free
    bond currents -- self-image bonds included -- that no atomic charge sees.
    Differentiable: :func:`born_charges` of it gives the periodic ``Z*``.

    The ``rho`` term is the fixed per-fragment carrier.  It vanishes for
    neutral fragments; for charged ones (free ions under PBC) it is the ionic
    part of the itinerant polarization and is only as continuous as the
    positions the caller supplies.
    """
    assembly = _require_assembly(solve_result, assembly)
    t, rho, mu_hat = _moments(solve_result, assembly)[:3]
    if assembly.bond_dipole is None:
        return polarization(assembly.charges(t, rho), assembly.pos, assembly.dipoles(mu_hat))
    return rho @ assembly.pos + t @ assembly.bond_dipole + assembly.dipoles(mu_hat).sum(dim=0)


def _cell_volume(assembly: ElectrostaticAssembly) -> Tensor:
    if not assembly.periodic or sum(assembly.pbc) != 3:
        raise ValueError(
            "a dielectric tensor needs a 3-D periodic cell: it is a response per "
            "unit volume at fixed macroscopic field, which a cluster or a slab "
            f"does not define (pbc={assembly.pbc})."
        )
    return torch.det(assembly.cell).abs()


def dielectric_tensor(
    solve_result: SolveResult,
    assembly: ElectrostaticAssembly,
    *,
    route: str = DEFAULT_ROUTE,
    create_graph: bool = False,
) -> Tensor:
    """Clamped-ion (electronic) dielectric tensor ``eps_inf``, ``[3, 3]``.

    ``eps_inf = I + 4 pi K_E alpha_cell / Omega`` with ``alpha_cell`` the
    cell's polarizability at fixed macroscopic field from
    :func:`polarizability` (response units, e^2 A^2 / eV; ``4 pi K_E`` is
    ``1/eps_0`` in this unit system).  SQE with positive bond hardness is a
    dielectric, never a metal (Nistor and Muser, Phys. Rev. B 79, 104303
    (2009)); here the finite value comes out of the model's own enthalpy.
    """
    volume = _cell_volume(assembly)
    alpha = polarizability(solve_result, assembly, route=route, create_graph=create_graph)
    identity = torch.eye(3, dtype=alpha.dtype, device=alpha.device)
    return identity + (4.0 * math.pi * K_E / volume) * alpha


def static_dielectric_tensor(
    epsilon_inf: Tensor, born: Tensor, force_constants: Tensor, volume: Tensor
) -> Tensor:
    """``eps_0 = eps_inf + (4 pi K_E / Omega) Z^T Phi^+ Z``, ``[3, 3]``.

    The relaxed-ion response: ions displace against the force constants
    ``Phi = d2E/dr2`` (``[3n, 3n]``, eV/A^2, of the FULL model energy) under the
    force ``Z* E``.  ``Phi^+`` is the pseudo-inverse on the complement of the
    three uniform translations, which carry no restoring force and, by the
    acoustic sum rule, no force from the field either.

    Args:
        epsilon_inf: ``[3, 3]`` from :func:`dielectric_tensor`.
        born: ``[n, 3, 3]`` Born charges, ``Z[i, a, b] = d(Omega P_a)/dr_{i,b}``.
        force_constants: ``[3n, 3n]``.
        volume: cell volume in A^3.
    """
    n_atoms = int(born.shape[0])
    dtype, device = force_constants.dtype, force_constants.device
    # Project the three translations out before inverting.
    translations = torch.eye(3, dtype=dtype, device=device).repeat(n_atoms, 1) / math.sqrt(n_atoms)
    projector = torch.eye(3 * n_atoms, dtype=dtype, device=device) - translations @ translations.T
    stiffness = projector @ force_constants @ projector
    coupling = born.permute(0, 2, 1).reshape(3 * n_atoms, 3)      # [3n (atom, b), 3 (a)]
    response = coupling.T @ torch.linalg.pinv(stiffness, hermitian=True) @ coupling
    return epsilon_inf + (4.0 * math.pi * K_E / volume) * response


def nonanalytic_force_constants(
    born: Tensor, epsilon_inf: Tensor, volume: Tensor, q_direction: Tensor
) -> Tensor:
    """The q -> 0 non-analytic (LO-TO) force-constant term, ``[3n, 3n]`` in eV/A^2.

    ``Phi^NA_{ia,jb} = (4 pi K_E / Omega) (q.Z_i)_a (q.Z_j)_b / (q . eps_inf . q)``
    (Cochran and Cowley; Gonze and Lee, Phys. Rev. B 55, 10355 (1997)), with
    ``(q.Z_i)_a = sum_c q_c Z[i, c, a]``.  Added to the analytic force constants
    along ``q_direction`` it splits the polar LO modes from the TO ones.
    """
    q = q_direction / q_direction.norm()
    projected = torch.einsum("c,ica->ia", q, born).reshape(-1)
    return (4.0 * math.pi * K_E / volume) * torch.outer(projected, projected) / (q @ epsilon_inf @ q)


def born_charges_minimiser(assembly: ElectrostaticAssembly, *, create_graph: bool = False) -> Tensor:
    """Born charges of the EXACT minimiser, ``Z*[i, a, b] = d(Omega P_a)/dr_{i,b}``.

    ``m* = argmin U`` on the feasible set, by one dense, differentiable solve of
    the same system the production solve iterates on, and ``Omega P`` the
    bond-current polarization of SPEC 15 (``sum_i (q_i r_i + mu_i)`` under open
    boundaries).  Differentiating ``Omega P(m*(r), r)`` through the solve is the
    implicit-function derivative, so this is well defined at symmetric points,
    where the fixed-k map is not (see :class:`FixedKDegenerateError`).  It is
    the ``k -> infinity`` limit of :func:`born_charges_from_energy`.  Periodic
    and open boundaries alike; ``sum_i Z*_i = Q I`` (:func:`acoustic_sum_rule_residual`).
    """
    if assembly.lattice_operator is not None:
        raise NotImplementedError(
            "born_charges_minimiser factorises the dense operator, which a "
            "solver='matrix_free' assembly does not hold (SPEC 16). Use "
            "born_charges_from_energy, or born_charges(bond_polarization(...)) of a "
            "converged solve."
        )
    pos = assembly.pos
    if not pos.requires_grad:
        raise ValueError(
            "born_charges_minimiser needs an assembly whose positions carry "
            "requires_grad: Z* is a derivative with respect to them."
        )
    system = assemble_system(
        assembly.params,
        _solver_geometry(assembly),
        _fragment_constraints(assembly),
        _solver_options(assembly.config),
    )
    width = system.n_bond_variables
    n_m = int(system.K.shape[0])
    constraint = system.constraint_t
    # Projector onto the feasible set, the way the production solve builds it:
    # pinv of the bond block, NOT an SVD null-space basis. Through the
    # dissociation envelope the block C B diag(f) carries a position gradient,
    # and for a single neutral fragment it is identically zero -- every singular
    # value degenerate -- where the SVD backward divides by zero (measured: NaN
    # Born charges on a rocksalt cell). When it is exactly zero there is nothing
    # to project at all.
    if int(constraint.shape[0]) == 0 or not bool((constraint != 0).any()):
        m_star = -torch.linalg.solve(system.K, system.c)
    else:
        bond_projector = torch.eye(width, dtype=system.K.dtype, device=system.K.device) - (
            torch.linalg.pinv(constraint) @ constraint
        )
        projector = torch.block_diag(
            bond_projector, torch.eye(n_m - width, dtype=system.K.dtype, device=system.K.device)
        )
        complement = torch.eye(n_m, dtype=system.K.dtype, device=system.K.device) - projector
        m_star = -projector @ torch.linalg.solve(
            projector @ system.K @ projector + complement, projector @ system.c
        )
    n_bonds, n_atoms = system.n_bonds, system.n_atoms
    t = m_star[:n_bonds]
    offset = system.dipole_offset
    mu = assembly.params.ell.unsqueeze(-1) * m_star[offset : offset + 3 * n_atoms].reshape(n_atoms, 3)
    if assembly.bond_dipole is None:
        q = assembly.incidence @ t + system.rho
        cell_dipole = q @ pos + mu.sum(dim=0)
    else:
        cell_dipole = system.rho @ pos + t @ assembly.bond_dipole + mu.sum(dim=0)
    return born_charges(cell_dipole, pos, create_graph=create_graph)

