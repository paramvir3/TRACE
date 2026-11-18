"""Variational charge/dipole solver for TRACE electrostatics.

The model minimises the quadratic electrostatic energy (SPEC 1.5)

    U_elec = 0.5 * m^T K m + chi^T m ,      m = ( {t_ij}, {mu_hat_i} )

subject to one net-charge constraint per fragment, where the atomic charges are
the split-charge sum ``q = B t + rho`` (SPEC 1.3) and the dipoles are carried in
*reduced* form ``mu_hat_i = mu_i / (e * ell_i)``.  The reduction is not
cosmetic: ``m`` mixes ranks, and a monopole-dipole cross term in a quadratic
form built from bare ``e`` and ``e*A`` moments is dimensionally meaningless.  In
reduced variables every block of ``K`` carries the same units (eV) and the
cross-rank blocks are homogeneous.

Three properties of this solver are load-bearing, and each was established by
measuring the alternative:

1. **Fixed iteration count, differentiated through.**  The energy is *defined*
   as the value after exactly ``n_iter`` conjugate-gradient steps -- not as the
   exact minimum.  Forces are then the true gradient of that definition, which
   is what makes them conservative.  Treating the truncated solve as if it were
   converged (Hellmann-Feynman, i.e. ignoring ``dm/dR``) drifts by
   **2348 K/ns/atom** at one iteration; differentiating through the iterations
   gives **0.14 K/ns/atom** -- a factor of 12000 (SPEC 2).  Consequently this
   module contains no ``torch.no_grad``, no ``detach`` inside the loop, and no
   custom ``autograd.Function`` that assumes convergence.

2. **Geometry-determined seed.**  The starting point is supplied by the caller
   as a :class:`GeometrySeed` -- or as a pure ``f(geometry, params)`` the solver
   evaluates itself -- and must be a function of the current configuration alone
   (SPEC 2 prescribes the antisymmetrised attention logit).  Seeding from the
   previous MD step makes ``U`` history-dependent -- the same configuration then
   has two different energies, measured 3.346e-02 eV apart -- and forfeits
   conservative forces, which is the entire reason for the fixed-iteration
   design.  A fingerprint binds the seed to the configuration it was computed
   at, and the solver re-checks it.  See :class:`GeometrySeed`.

3. **Per-fragment constraints.**  One Lagrange constraint per fragment, applied
   by projection.  A single global constraint is size-inconsistent: a +1 ion
   surrounded by distant spectators kept 1.000, 0.242, 0.102, 0.052, 0.016 of
   its charge for 0/5/20/50/200 spectators at 1000 A.  Per-fragment gives
   1.000000 at every spectator count (SPEC 1.3).

Shapes are static throughout (SPEC 2): the CG loop is a Python ``for`` over a
fixed integer, and there is no convergence test that branches.

Spin channels -- opt-in, and a sign that is not negotiable
----------------------------------------------------------
SPEC 9 splits the charge sector (and only the charge sector -- the induced
dipole responds to the total density) into

    q_i^sigma = q_i^(0)sigma + sum_j t_ij^sigma + rho_i^sigma ,  sigma in {up, down}
    q_i = q_i^up + q_i^down  (charge) ,   s_i = q_i^up - q_i^down  (spin)

with **two** constraints per fragment, ``sum_F q_i = Q_F`` and
``sum_F s_i = 2 S_F``.  It is requested by giving
:class:`ElectrostaticParams` a ``j_exchange`` and :class:`FragmentConstraints`
a ``spin_targets``; with both absent nothing below changes, and the
closed-shell path is the same code it always was.

**The Coulomb kernel couples TOTAL charges.**  ``0.5 sum_ij J_ij q_i q_j``, not
one copy per spin channel: an up electron and a down electron on different
atoms repel exactly as any two charges do.  This module carries the bond
variables in the ``(t^c, t^s) = (t^up + t^down, t^up - t^down)`` basis, where
that statement is structural -- ``J`` reaches ``t^c`` and nothing else, and the
``t^s`` block of ``K`` is ``B^T diag(J^x) B + diag(kappa^s)`` with no ``J`` in
it anywhere.

**J^x > 0, and the reason it cannot be otherwise.**  In the ``(q_up, q_down)``
basis the on-site form ``0.5 eta q^2 + 0.5 J^x s^2`` has Hessian

    [[eta + J^x, eta - J^x], [eta - J^x, eta + J^x]]

whose eigenvalues are ``2 eta`` (charge mode) and ``2 J^x`` (spin mode) --
verified numerically to 1.6e-15 over eta in {1, 10.832, 12} and J^x in
[-3, 3].  Positive definiteness therefore requires ``J^x > 0``, exactly as it
requires ``eta > 0``, and :class:`ElectrostaticParams` asserts it at
construction.  At ``J^x = -0.5 eV`` the spin mode measures ``-1.0 eV/e^2`` and
the energy runs to ``-2.5e5 eV`` at ``|s| = 1000``: unbounded below, the same
structure as the polaron instability this project audited earlier.

``J^x > 0`` **penalises** spin polarisation.  Hund's rule and Stoner
ferromagnetism want the opposite sign, and that sign *is* the unbounded,
non-PD case.  Consequently:

    **Spin polarisation in this model comes from the spin-dependent linear term
    chi_i^sigma, never from a Stoner instability.**

A learned ``J^x`` must be emitted as ``softplus(raw) + floor``, like ``eta``.

**The error this inherits, stated where a user will see it.**  A quadratic-in-s
on-site energy violates the fractional-spin (flat-plane) condition.  For a
one-electron system whose exact energy is FLAT in spin polarisation,
``0.5 J^x s^2`` with ``J^x = 2 eV`` manufactures a **1.0 eV** splitting at
``s = 1`` (SPEC 9.2).  Real Fe(II) high-spin/low-spin gaps are **0.1-0.4 eV**,
so the artifact is 100-400% of the signal.  This is the same functional form
AIMNet2-NSE and MACE-POLAR-1 use; it is a known limitation and not a reason to
refuse the feature, but **the spin-state splitting this model produces is a
FITTED quantity, not a derived one**, and no spin-state energetics from it
should be read as a prediction of the underlying physics.

That error is **measured, not just declared**.  :func:`fractional_spin_error`
returns the curvature this model imposes along ``M_s`` at fixed ``N`` -- the
quantity the flat-plane condition requires to vanish for a degenerate open shell
-- in eV per unpaired electron squared, and
:meth:`~trace_electrostatics.diagnostics.FractionalSpinError.splitting` turns it
into the fraction of any predicted spin-state splitting that the artifact
supplied.  :func:`degenerate_doublet_reference` is the calibration point where
the exact answer is provably zero.  The cheap (``O(n_atoms)``) form of the
curvature runs as solver telemetry on every spin-resolved solve; see
:class:`SolverOptions`.

Host synchronisation
--------------------
Nothing in the solve reads a tensor on the host.  Every Python ``if`` below is
on structure known at trace time -- a shape, a dtype, whether an override is
``None`` -- never on a tensor value, so the graph is unbroken and the device
queue never stalls.  The cheap physics assertions are the one place this is not
free, and they go through
:func:`trace_electrostatics.diagnostics._needs_host_report`, which issues
``torch._assert_async`` under tracing and only reads the flag on the host in
eager mode, where there is no graph to break.  Construction-time checks
(:class:`ElectrostaticParams`, :class:`FragmentConstraints`, an override's
symmetry) run once per parameter set and are deliberately outside that budget.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor

from .constants import to_polarizability_volume
from .implicit import (
    DEFAULT_SOLVE_MAX_ITER,
    DEFAULT_SOLVE_RTOL,
    QuadraticProblem,
    minimise,
)
from .diagnostics import (
    DEFAULT_OPERATOR_SYMMETRY_RTOL,
    ElectrostaticsAssertionError,
    FractionalSpinError,
    assert_fragment_charges,
    assert_hardness_floor,
    assert_is_minimum,
    assert_pair_operator_symmetry,
    assert_positive_definite_at_solution,
    assert_positive_kappa,
    fractional_spin_curvature,
    validation_enabled,
    verify_spin_mode_eigenvalue,
    _needs_host_report,
)
from .kernels import (
    DEFAULT_GAUSSIAN_A,
    DEFAULT_THOLE_A,
    DEGENERATE_SEPARATION,
    GAUSSIAN_A_MAX,
    THOLE_A_MAX,
    charge_dipole_tensor,
    charge_quadrupole_tensor,
    combine_sigma,
    coulomb_kernel,
    dipole_interaction_tensor,
    dipole_quadrupole_tensor,
    gaussian_damping,
    gaussian_multipole_width,
    kernel_self,
    quadrupole_cartesian_basis,
    quadrupole_interaction_tensor,
    thole_damping,
)
from .topology import (
    build_incidence,
    charge_spin_to_spin_channels,
    distribute_fragment_charges,
    distribute_fragment_spins,
    fragment_constraint_matrix,
    spin_constraint_block,
)

__all__ = [
    "DEFAULT_CG_ITERATIONS",
    "DEFAULT_EXCHANGE_FLOOR",
    "DEFAULT_THOLE_A",
    "ElectrostaticParams",
    "Geometry",
    "FragmentConstraints",
    "GeometrySeed",
    "SolverOptions",
    "ElectrostaticSystem",
    "SolveDiagnostics",
    "SolveResult",
    "assemble_system",
    "crossing_bond_selector",
    "assert_positive_exchange",
    "solve_charges",
    "split_spin_potentials",
    "isotropic_polarizability_volume",
    # The fractional-spin (flat-plane) error, SPEC 9.2
    "fractional_spin_error",
    "degenerate_doublet_reference",
]


# Number of conjugate-gradient iterations of the LEGACY fixed-k solve
# (solve_mode="fixed_k", SPEC 2). The default solve is converged (SPEC 17): the
# energy is the minimum itself, so it does not depend on an iteration count.
DEFAULT_CG_ITERATIONS: int = 6

#: SPEC 17: the energy is the constrained minimum of U, solved to a tolerance
#: and differentiated implicitly. SPEC 2: k CG steps differentiated through.
SOLVE_CONVERGED = "converged"
SOLVE_FIXED_K = "fixed_k"
SOLVE_MODES = (SOLVE_CONVERGED, SOLVE_FIXED_K)
DEFAULT_SOLVE_MODE = SOLVE_CONVERGED

# Floor on the polarizability volume fed to the Thole width
# ``(alpha_i alpha_j)^(1/6)``. A genuinely zero-polarizability atom would make
# the width zero and the damping 0/0; this keeps the limit well defined.
MIN_POLARIZABILITY_VOLUME: float = 1e-12

# Cartesian dimensionality; named so the reshapes below are readable.
_SPATIAL_DIM: int = 3
_QUADRUPOLE_DIM: int = 5  # symmetric traceless 3x3, the 2e irrep

#: The smeared densities the multipole blocks can be derived from (SPEC 12).
MULTIPOLE_KERNELS = ("thole", "gaussian")

# Placeholder separation, in A, substituted into the square root wherever the
# true one is degenerate (the i == i diagonal, and any two distinct atoms at
# coincident positions). Every entry it reaches is masked out of all three pair
# blocks afterwards -- except the charge-charge block, which takes its true
# r -> 0 limit there -- so the value never enters the energy; what it buys is a
# finite derivative for the square root. Same value and same threshold as
# ``energy._DIAGONAL_SHIFT``, which guards the reported energy the same way.
_DIAGONAL_SHIFT: float = 1.0

# Suggested floor for a learned on-site exchange ``J^x``, eV/e^2.
#
# There is no *physical* lower bound on J^x the way J(0) bounds eta: the spin
# sector has no off-diagonal Coulomb block for the on-site term to dominate, so
# the positive-definiteness requirement is the bare ``J^x > 0`` of SPEC 9.1.
# What this number is for is conditioning. The spin mode's curvature is 2 J^x
# (verified to 1.6e-15 below), so a J^x at 1e-12 leaves a mode 13 orders of
# magnitude softer than the charge mode and the fixed-k CG spends its whole
# budget on it. 1e-3 eV/e^2 is four orders below the smallest exchange splitting
# anyone fits (Fe(II) high-spin/low-spin gaps are 0.1-0.4 eV, SPEC 9.2) and
# eleven above the point where the block stops being numerically definite.
#
# It is a DEFAULT, not a floor that is enforced: SPEC 9.1 requires J^x > 0 and
# that is all assert_positive_exchange checks, so the soft-exchange limit stays
# reachable the way the kappa -> 0 limit of SPEC 1.4 stays reachable.
DEFAULT_EXCHANGE_FLOOR: float = 1e-3

# Attribute stamped on every tensor :func:`solve_charges` returns, so that a
# seed built out of one can be recognised and refused. An attribute on the
# tensor object rather than a module-level registry: it is garbage collected
# with the tensor, needs no locking, and cannot leak between threads or tests.
_SOLVER_OUTPUT_ATTR = "_trace_electrostatics_solver_output"

# Words a provenance string may not contain. A label is the weakest of the
# three seed guards and this list is not a security boundary -- it catches the
# honest caller who writes down what they are actually doing, which is how the
# hole was found ("previous MD step" and "warm start from step 41" were both
# accepted verbatim).
_HISTORY_WORDS = (
    "previous",
    "prev ",
    "warm start",
    "warm-start",
    "warmstart",
    "last step",
    "prior step",
    "earlier step",
    "restart",
    "cached",
    "cache",
    "history",
    "carried over",
)


def _visible(text: str) -> bool:
    """Whether a string has any non-whitespace character.

    ``str.strip()`` removes ASCII whitespace only, so ``"\\xa0a"`` -- and a bare
    ``"\\xa0"`` -- survived it. ``str.split()`` splits on Unicode whitespace.
    """
    return bool(text.split())


def _names_history(provenance: str) -> Optional[str]:
    """The first history word a provenance string contains, if any."""
    lowered = " ".join(provenance.lower().split())
    for word in _HISTORY_WORDS:
        if word in lowered:
            return word.strip()
    return None


def _mark_solver_output(*tensors: Tensor) -> None:
    """Stamp tensors as having come out of a solve."""
    for tensor in tensors:
        try:
            setattr(tensor, _SOLVER_OUTPUT_ATTR, True)
        except (AttributeError, RuntimeError):  # pragma: no cover - exotic subclasses
            pass


def _is_solver_output(tensor: Tensor) -> bool:
    """Whether a tensor is, or is a view of, the output of a previous solve.

    The stamp is read through ``_base`` as well, because the obvious way to pass
    a previous result on is ``previous.t.reshape(-1)``: torch returns a fresh
    tensor *object*, which carries no attribute of its own but whose ``_base``
    is the solution vector.  ``clone()`` and ``detach()`` do break the link; they
    copy or re-root, and nothing short of tracking storage would see through
    them.  That is why the geometry fingerprint, and not this, is the barrier
    for the failure mode that actually happens -- a seed carried across an MD
    step, where the positions have moved.
    """
    if getattr(tensor, _SOLVER_OUTPUT_ATTR, False):
        return True
    base = getattr(tensor, "_base", None)
    return bool(base is not None and getattr(base, _SOLVER_OUTPUT_ATTR, False))


def _reject_solver_output(tensor: Tensor, name: str) -> None:
    """Refuse a seed component that is the output of a previous solve."""
    if _is_solver_output(tensor):
        raise ValueError(
            f"GeometrySeed.{name} is a tensor returned by a previous solve_charges "
            "call. Seeding from a previous solve -- the previous MD step, or the same "
            "configuration solved twice -- makes U history-dependent: the same "
            "configuration then has two different energies (measured: 3.346e-02 eV "
            "apart on identical geometry), the force stops being the gradient of "
            "anything, and the conservative forces the fixed-iteration design exists "
            "to buy are gone (SPEC 2). Recompute the seed from the current "
            "configuration and bind it with GeometrySeed.for_geometry."
        )


def isotropic_polarizability_volume(A_mu: Tensor, ell: Tensor) -> Tensor:
    """Per-atom isolated-atom polarizability volume in A^3.

    With ``U = 0.5 mu_hat^T A_mu mu_hat - mu . E`` and ``mu = ell * mu_hat``,
    the stationary condition gives ``mu = ell^2 A_mu^-1 E``, so the
    polarizability tensor is ``ell^2 A_mu^-1`` in ``e^2 A^2 / eV``.  That is a
    *response* polarizability and **not** a volume -- SPEC 0 flags exactly this
    trap -- so it is converted with :func:`to_polarizability_volume` before it
    reaches the Thole width, which is dimensionless only for a volume in A^3.
    The isotropic average ``tr(alpha)/3`` is what ``(alpha_i alpha_j)^(1/6)``
    expects.
    """
    inverse = torch.linalg.inv(A_mu)
    trace = inverse.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    response = ell.reshape(-1) ** 2 * trace / _SPATIAL_DIM
    return to_polarizability_volume(response).clamp_min(MIN_POLARIZABILITY_VOLUME)


def assert_positive_exchange(
    j_exchange: Tensor, *, tol: float = 0.0, sync: Optional[bool] = None
) -> Tensor:
    """Assert ``J^x_i > 0`` on every atom (SPEC 9.1).

    **Derivation, so the sign is checkable rather than asserted.**  In the
    ``(q_up, q_down)`` basis the on-site energy ``0.5 eta q^2 + 0.5 J^x s^2``
    with ``q = q_up + q_down`` and ``s = q_up - q_down`` has Hessian

        [[eta + J^x, eta - J^x], [eta - J^x, eta + J^x]]

    whose eigenvectors are ``(1, 1)/sqrt(2)`` and ``(1, -1)/sqrt(2)`` -- the
    charge mode and the spin mode -- with eigenvalues ``2 eta`` and ``2 J^x``.
    Positive definiteness of the on-site block is therefore ``eta > 0`` **and**
    ``J^x > 0``, with no interaction between the two conditions.

    That derivation is **checked rather than trusted**:
    :func:`~trace_electrostatics.diagnostics.verify_spin_mode_eigenvalue` runs
    it numerically -- the Hessian by the chain rule against the matrix written
    above, the eigenvalues against ``eigvalsh``, and the eigenvectors as a
    residual so the degenerate ``J^x == eta`` case cannot break the test.
    Measured over ``eta`` in {1, 10.832, 12, 25} eV/e^2 and ``J^x`` in [-3, 3]
    eV/e^2: eigenvalues to **1.78e-15**, eigenvector residuals to **2.06e-15**,
    Hessian to **0.0** exactly.

    Below zero the energy is unbounded along the spin mode, exactly the way it
    is unbounded below the hardness floor along the charge mode: at
    ``eta = 12``, ``J^x = -0.5`` the measured energy at ``|s| = 10, 100, 1000``
    is ``-25``, ``-2500``, ``-250000`` eV.

    **What this costs, and why it is still the right sign.**  ``J^x > 0``
    penalises spin polarisation.  Hund's rule and Stoner ferromagnetism need
    ``J^x < 0``, and that is precisely the non-PD case above -- the same
    structure as the polaron instability audited earlier in this project.  Spin
    polarisation in this model therefore comes from the spin-dependent linear
    term ``chi^sigma``, never from a Stoner instability.  A learned ``J^x``
    must be ``softplus(raw) + floor``; see :data:`DEFAULT_EXCHANGE_FLOOR`.

    Strict positivity in exact arithmetic and nothing more, matching
    :func:`~trace_electrostatics.diagnostics.assert_positive_kappa`: the
    smallest subnormal passes, and no caller may infer a numerically definite
    spin block from this check passing.  The SPEC 9 soft-exchange limit has to
    stay reachable.

    A NaN counts as a violation -- the predicate is negated rather than written
    as ``j_exchange <= 0``, because ``NaN > 0`` is ``False`` and an untrained
    head would otherwise put a NaN straight through the one check that exists
    to catch it.

    Returns
    -------
    Tensor
        ``j_exchange``, unchanged, so this can be used inline.
    """
    j_exchange = j_exchange.reshape(-1)
    violated = ~(j_exchange > abs(tol))
    if _needs_host_report(
        violated.any(),
        "J^x must be strictly positive (and not NaN) on every atom: the spin mode "
        "of the on-site Hessian has curvature 2 J^x, so J^x <= 0 makes the energy "
        "unbounded below along spin polarisation (SPEC 9.1). Use "
        "softplus(raw) + floor.",
        sync=sync,
    ):
        index = torch.nonzero(violated).reshape(-1)
        worst = int(index[int(torch.argmin(j_exchange[index]))])
        raise ElectrostaticsAssertionError(
            f"on-site exchange J^x must be strictly positive: {int(violated.sum())} "
            f"atom(s) violate it; worst is atom {worst} with "
            f"J^x={float(j_exchange[worst]):.6e} eV/e^2. In the (q_up, q_down) basis "
            "the on-site Hessian [[eta+J^x, eta-J^x], [eta-J^x, eta+J^x]] has "
            "eigenvalues 2 eta (charge mode) and 2 J^x (spin mode), so J^x <= 0 is "
            "an indefinite block and the energy is unbounded below along spin "
            "polarisation -- at J^x = -0.5 eV/e^2 the measured energy reaches "
            "-2.5e5 eV at |s| = 1000 (SPEC 9.1). Hund's rule and Stoner "
            "ferromagnetism want this sign; they are not available here. Spin "
            "polarisation in this model comes from the spin-dependent linear term "
            "chi^sigma. Emit J^x as softplus(raw) + floor."
        )
    return j_exchange


def split_spin_potentials(chi_up: Tensor, chi_down: Tensor) -> tuple[Tensor, Tensor]:
    """``(chi^up, chi^down) -> (chi, chi_spin)`` for :class:`ElectrostaticParams`.

    A head that emits one electronegativity per spin channel writes the linear
    term as ``chi^up . q^up + chi^down . q^down``.  With ``q^up = (q + s)/2``
    and ``q^down = (q - s)/2`` that is

        chi . q + chi_spin . s ,
        chi      = (chi^up + chi^down) / 2 ,
        chi_spin = (chi^up - chi^down) / 2 ,

    which is what this solver consumes.  **Note the factor of 1/2**: the
    potentials transform with the inverse-transpose of the map that takes the
    populations to ``(q, s)``, not with the map itself, so
    :func:`~trace_electrostatics.topology.spin_channels_to_charge_spin` is the
    wrong function here even though the words match.  Getting it wrong doubles
    or halves every spin splitting the model produces and is invisible
    downstream, because both forms are perfectly well-behaved quadratics.

    Closed shell is the fixed point: ``chi^up == chi^down`` gives
    ``chi = chi^up`` and ``chi_spin = 0``, so the charge sector is untouched.

    ``chi_spin`` is the **only** source of spin polarisation in this model; see
    :func:`assert_positive_exchange` for why a Stoner-type instability is not
    available.
    """
    return 0.5 * (chi_up + chi_down), 0.5 * (chi_up - chi_down)


@dataclass(frozen=True)
class ElectrostaticParams:
    """Per-atom and per-bond electrostatic parameters emitted by the heads.

    Attributes
    ----------
    chi:
        ``[n_atoms]`` electronegativity, eV/e.  The linear term of ``U``.
    eta:
        ``[n_atoms]`` on-site hardness, eV/e^2.  Must satisfy the hardness floor
        ``eta >= J(0)``; the head is required to emit ``softplus(raw) + J(0)``.
    sigma:
        ``[n_atoms]`` Gaussian charge width, A.
    ell:
        ``[n_atoms]`` dipole length scale, A.  Sets the reduction
        ``mu_hat = mu / (e * ell)``.
    kappa:
        ``[n_bonds]`` bond hardness, eV/e^2, strictly positive, one entry per
        *undirected* bond in the same order as ``bond_index``.
    A_mu:
        ``[n_atoms, 3, 3]`` reduced dipole stiffness, eV.  Symmetric positive
        definite; its inverse times ``ell^2`` is the isolated-atom
        polarizability.

    The cheap physics assertions (hardness floor, ``kappa > 0``) run at
    construction, which is where SPEC 2.1 places them -- they are properties of
    the parameters, so checking them once per parameter set costs nothing and
    keeps the solve itself free of host synchronisation.

    Spin channels (SPEC 9) -- all optional, all default ``None``
    ------------------------------------------------------------
    Supplying ``j_exchange`` is what turns the spin sector on; leaving it
    ``None`` leaves every field above meaning exactly what it meant before and
    every code path the one it followed before.

    j_exchange:
        ``[n_atoms]`` on-site exchange ``J^x_i``, eV/e^2, the coefficient of
        ``0.5 J^x_i s_i^2`` with ``s_i = q_i^up - q_i^down``.  **Strictly
        positive** -- the spin mode of the on-site Hessian has curvature
        ``2 J^x`` and ``J^x <= 0`` is unbounded below (SPEC 9.1).  A learned
        head must emit ``softplus(raw) + floor``; see
        :data:`DEFAULT_EXCHANGE_FLOOR` and :func:`assert_positive_exchange`,
        which carries the derivation and the measured numbers.

        **This term is also the model's fractional-spin error.**  Being
        quadratic in ``s`` it violates the flat-plane condition by construction:
        for a one-electron system, whose exact energy is degenerate in spin
        polarisation, ``J^x = 2 eV`` manufactures a 1.0 eV splitting where the
        exact answer is 0 (SPEC 9.2).  ``J^x`` is therefore two things at once --
        the parameter that makes the spin sector bounded below, and the size of
        the artifact.  :func:`fractional_spin_error` measures the second;
        :func:`degenerate_doublet_reference` is the case where the exact answer
        is known.  Any spin-state splitting this model produces is a FITTED
        quantity, not a derived one, and it is the same functional form
        AIMNet2-NSE and MACE-POLAR-1 use, so this is a limitation of the class
        of model rather than of this implementation.
    chi_spin:
        ``[n_atoms]`` spin-channel linear term, eV/e, coupling as
        ``chi_spin . s``.  Equals ``(chi^up - chi^down)/2``; see
        :func:`split_spin_potentials`, and note the factor of 1/2.  ``None``
        means zero, i.e. no spin polarisation except what the constraints
        force.  **This is the only source of spin polarisation in the model.**
    kappa_spin:
        ``[n_bonds]`` spin-channel bond hardness, eV/e^2, strictly positive:
        the coefficient of ``0.5 kappa^s_b (t^s_b)^2`` with
        ``t^s = t^up - t^down``.  ``None`` means ``kappa``.

    THE FACTOR OF TWO IN THE BOND HARDNESS -- read before emitting kappa_spin
    ------------------------------------------------------------------------
    SPEC 9 writes the bond hardness per spin, ``0.5 kappa^sigma (t^sigma)^2``.
    This class carries the ``(charge, spin)`` basis instead, because that is
    the basis in which the closed-shell limit is exact, and the two are related
    by a factor of two that is silent if it is got wrong::

        0.5 kappa^up (t^up)^2 + 0.5 kappa^down (t^down)^2
          =  0.5 * (kappa^up + kappa^down)/4 * ((t^c)^2 + (t^s)^2)
           +       (kappa^up - kappa^down)/4 * t^c t^s

    so ``kappa`` (the coefficient of ``0.5 (t^c)^2``, and the hardness of the
    **total** transfer, which is what the closed-shell model has always meant)
    corresponds to ``kappa^up = kappa^down = 2 kappa``.  Handing the
    closed-shell ``kappa`` to both spin channels literally would halve the
    charge-channel hardness, making charge transfer twice as easy the moment
    spin is switched on -- a discontinuity in the model, not in the numerics,
    and one that no test of the spin sector would notice.  With the convention
    here, a spin-unpolarised system gives the closed-shell answer with the same
    ``kappa``; that is verified, and it is the strongest consistency check
    available on this feature.

    ``kappa^up != kappa^down`` -- a bond that passes up-spin more easily than
    down-spin -- would add the cross term above, coupling ``t^c`` to ``t^s``.
    It is deliberately **out of scope**: it breaks up/down symmetry at the level
    of a single bond, which is only meaningful against a global spin
    quantisation axis this model does not carry.  ``kappa_spin`` is the
    scope that is in: a bond can be made stiff against spin flow and soft
    against charge flow, or the reverse, without either channel being labelled.
    """

    chi: Tensor
    eta: Tensor
    sigma: Tensor
    ell: Tensor
    kappa: Tensor
    A_mu: Tensor
    j_exchange: Optional[Tensor] = None
    chi_spin: Optional[Tensor] = None
    kappa_spin: Optional[Tensor] = None
    #: SPEC 10 quadrupole sector, opt-in.  Supplying ``A_Theta`` turns it on.
    #: ``[n_atoms, 5, 5]`` on-site stiffness of the reduced quadrupole
    #: ``Theta_hat = Theta / (e ell^2)`` in the orthonormal ``2e`` basis of
    #: :func:`~trace_electrostatics.kernels.quadrupole_cartesian_basis`, eV,
    #: symmetric positive definite (the heads emit ``L L^T + floor I_5``).
    A_Theta: Optional[Tensor] = None
    #: ``[n_atoms, 5]`` quadrupolar electronegativity, eV: the linear term
    #: ``chi_Theta . Theta_hat``.  The PERMANENT source of an atomic quadrupole --
    #: sigma-holes, lone pairs, pi clouds -- which induction by the other
    #: moments alone would never produce.  ``None`` means zero (purely induced).
    chi_theta: Optional[Tensor] = None
    #: ``[n_bonds]`` per-bond linear term, eV/e: ``U += bond_bias . tau``.  Zero
    #: on physical bonds; the external-circuit bonds of a constant-potential
    #: electrode carry ``-(V_e - V_ref)`` (SPEC 20).  ``None`` means zero.
    bond_bias: Optional[Tensor] = None

    def __post_init__(self) -> None:
        n_atoms = int(self.chi.reshape(-1).numel())
        if self.bond_bias is not None and self.bond_bias.reshape(-1).numel() != self.kappa.numel():
            raise ValueError(
                f"bond_bias must have one entry per bond ({self.kappa.numel()}), got "
                f"{self.bond_bias.reshape(-1).numel()}"
            )
        for name in ("eta", "sigma", "ell"):
            value = getattr(self, name).reshape(-1)
            if value.numel() != n_atoms:
                raise ValueError(
                    f"{name} must have {n_atoms} entries to match chi, got {value.numel()}"
                )
        if self.A_mu.shape != (n_atoms, _SPATIAL_DIM, _SPATIAL_DIM):
            raise ValueError(
                f"A_mu must have shape [{n_atoms}, 3, 3], got {tuple(self.A_mu.shape)}"
            )
        if self.kappa.dim() != 1:
            raise ValueError(
                f"kappa must be a 1-D per-bond tensor, got {tuple(self.kappa.shape)}"
            )
        assert_hardness_floor(self.eta, self.sigma)
        assert_positive_kappa(self.kappa)

        # --- spin sector (SPEC 9). Every check below is skipped entirely when
        # the sector is off, so the closed-shell construction path is the one
        # it always was, down to the sequence of tensor ops.
        if self.chi_spin is not None and self.j_exchange is None:
            raise ValueError(
                "chi_spin was supplied without j_exchange. chi_spin is a linear term "
                "on the spin population s, and with no on-site exchange J^x and no "
                "spin-channel bond hardness there is nothing quadratic to balance it "
                "against: the spin sector would be unbounded below along any "
                "direction the constraints leave free. Supply j_exchange "
                "(softplus(raw) + floor) to turn the spin sector on."
            )
        if self.kappa_spin is not None and self.j_exchange is None:
            raise ValueError(
                "kappa_spin was supplied without j_exchange, so the spin sector is "
                "off and it would be silently ignored. Supply j_exchange to turn the "
                "spin sector on (SPEC 9)."
            )
        if self.j_exchange is not None:
            exchange = self.j_exchange.reshape(-1)
            if exchange.numel() != n_atoms:
                raise ValueError(
                    f"j_exchange must have {n_atoms} entries to match chi, got "
                    f"{exchange.numel()}"
                )
            if self.chi_spin is not None and self.chi_spin.reshape(-1).numel() != n_atoms:
                raise ValueError(
                    f"chi_spin must have {n_atoms} entries to match chi, got "
                    f"{self.chi_spin.reshape(-1).numel()}"
                )
            if self.kappa_spin is not None:
                if self.kappa_spin.dim() != 1:
                    raise ValueError(
                        "kappa_spin must be a 1-D per-bond tensor, got "
                        f"{tuple(self.kappa_spin.shape)}"
                    )
                if self.kappa_spin.numel() != self.kappa.numel():
                    raise ValueError(
                        f"kappa_spin must have {int(self.kappa.numel())} entries to "
                        f"match kappa, got {int(self.kappa_spin.numel())}"
                    )
                assert_positive_kappa(self.kappa_spin)
            assert_positive_exchange(exchange)

        # --- quadrupole sector (SPEC 10/12), skipped entirely when off.
        if self.chi_theta is not None and self.A_Theta is None:
            raise ValueError(
                "chi_theta was supplied without A_Theta. chi_theta is a linear term "
                "on the quadrupole, and with no on-site stiffness the quadrupole "
                "sector has nothing quadratic to balance it: the energy is "
                "unbounded below along chi_theta (SPEC 10.1)."
            )
        if self.A_Theta is not None:
            if self.A_Theta.shape != (n_atoms, _QUADRUPOLE_DIM, _QUADRUPOLE_DIM):
                raise ValueError(
                    f"A_Theta must have shape [{n_atoms}, 5, 5], got "
                    f"{tuple(self.A_Theta.shape)}"
                )
            if self.chi_theta is not None and self.chi_theta.shape != (
                n_atoms,
                _QUADRUPOLE_DIM,
            ):
                raise ValueError(
                    f"chi_theta must have shape [{n_atoms}, 5], got "
                    f"{tuple(self.chi_theta.shape)}"
                )

    @property
    def n_atoms(self) -> int:
        return int(self.chi.reshape(-1).numel())

    @property
    def n_bonds(self) -> int:
        return int(self.kappa.numel())

    @property
    def spin_resolved(self) -> bool:
        """Whether the spin sector of SPEC 9 is switched on for these parameters."""
        return self.j_exchange is not None

    @property
    def quadrupole_resolved(self) -> bool:
        """Whether the variational quadrupole sector of SPEC 10/12 is switched on."""
        return self.A_Theta is not None

    def effective_kappa_spin(self) -> Tensor:
        """``kappa_spin``, defaulting to ``kappa``.  Raises when spin is off."""
        if self.j_exchange is None:
            raise ValueError(
                "these parameters are closed shell (j_exchange is None); there is no "
                "spin-channel bond hardness"
            )
        return self.kappa if self.kappa_spin is None else self.kappa_spin


@dataclass(frozen=True)
class Geometry:
    """Positions plus the bond topology the split charges live on.

    Attributes
    ----------
    positions:
        ``[n_atoms, 3]`` Cartesian coordinates in A.  Gradients of the returned
        energy with respect to this tensor are the electrostatic forces.
    incidence:
        ``[n_atoms, n_bonds]`` signed incidence matrix ``B`` from
        :func:`trace_electrostatics.topology.build_incidence`.
    bond_index:
        ``[2, n_bonds]`` undirected bond list, same column order as ``B``.

    The three pair-operator overrides
    --------------------------------
    ``coulomb_matrix``, ``charge_dipole`` and ``dipole_block`` are the seam for
    periodic boundary conditions: an Ewald sum supplies the lattice-summed pair
    operators and everything downstream is unchanged.  They must be supplied
    **together or not at all**.  Overriding only the charge-charge block is
    worse than overriding none, because the solve then minimises a functional in
    which the charges see the lattice sum and the dipoles see raw, unwrapped
    open-boundary coordinates -- measured 0.513 eV away from the energy the
    assembly reports, on four atoms in an 8 A cell where one pair 0.616 A apart
    by minimum image is 7.401 A apart in raw coordinates.

    Each override is the complete **pair** operator, *including its diagonal*:
    under a lattice sum an atom interacts with its own periodic images, and that
    self term is a real contribution of the same order as the on-site
    parameters.  It is not masked out, and it is not folded into ``eta`` or
    ``A_mu`` -- ``eta`` has to keep satisfying the hardness floor ``eta >= J(0)``
    (SPEC 1.2), which a negative lattice self term would destroy.  What
    :func:`assemble_system` still adds is exactly the on-site parameters:

    ================  ===================  ==========================================
    field             shape                what is added to it
    ================  ===================  ==========================================
    ``coulomb_matrix``  ``[n, n]``           ``diag(eta)``
    ``charge_dipole``   ``[n, 3 n]``         nothing
    ``dipole_block``    ``[3 n, 3 n]``       ``A_mu`` on the diagonal 3x3 blocks
    ================  ===================  ==========================================

    Reduced-dipole convention, matching the fields of the same name on
    :class:`ElectrostaticSystem`: ``charge_dipole[i, 3j + b]`` already carries
    the factor ``ell_j`` and ``dipole_block[3i + a, 3j + b]`` already carries
    ``ell_i ell_j``, because ``K`` acts on ``mu_hat = mu / (e ell)``.  For open
    boundaries these are ``ell_j T01[i, j, b]`` and ``ell_i ell_j T[i, j, a, b]``
    with zero diagonals; passing zeros for all three reproduces a bare
    charge-only model.

    Every supplied operator is checked for the exchange symmetry of the term it
    multiplies (:func:`~trace_electrostatics.diagnostics.assert_pair_operator_symmetry`).
    Nothing downstream notices an asymmetric one on its own: only the symmetric
    part of ``K`` enters ``0.5 m^T K m``, so the projected CG runs happily on an
    operator that describes no energy and merely stops converging -- measured,
    at ``|K - K^T| = 4.96``, as a residual stalling at 0.203 from 3.48 after six
    iterations, with no complaint.
    """

    positions: Tensor
    incidence: Tensor
    bond_index: Tensor
    coulomb_matrix: Optional[Tensor] = None
    charge_dipole: Optional[Tensor] = None
    dipole_block: Optional[Tensor] = None
    #: Whether ``charge_dipole`` is a pure pair interaction.  ``True`` for a
    #: lattice sum of ``T01`` (its ``+n`` and ``-n`` images cancel, so
    #: ``W_ij == -W_ji``).  ``False`` for the 2-D slab route, whose correction
    #: ``U = 2 pi K_E M_n^2 / V`` is a rank-1 GLOBAL term in the total dipole:
    #: its charge-dipole block goes as ``z_i`` alone, independent of ``j``.
    #: ``K`` stays symmetric either way -- the off-diagonal blocks are built as
    #: transposes -- so this waives a bug detector, not a correctness condition.
    charge_dipole_is_pair: bool = True
    #: SPEC 14.  Which smeared density built the overrides: ``"thole"`` (the
    #: Thole Ewald of ``energy._periodic_pair_blocks``) or ``"gaussian"`` (the
    #: Gram lattice operator of :mod:`trace_electrostatics.lattice`).  Must
    #: match ``SolverOptions.multipole_kernel``: a Gaussian on-site treatment
    #: over Thole lattice blocks is not the functional either route reports.
    pair_kernel: str = "thole"
    #: SPEC 14, the quadrupole pair overrides, all three or none, and only with
    #: the dipole overrides: ``charge_quadrupole [n, 5n]`` (x ell_j^2),
    #: ``dipole_quadrupole [3n, 5n]`` (x ell_i ell_j^2), ``quadrupole_pair
    #: [5n, 5n]`` (x ell_i^2 ell_j^2, pair part only -- ``A_Theta`` is added).
    charge_quadrupole: Optional[Tensor] = None
    dipole_quadrupole: Optional[Tensor] = None
    quadrupole_pair: Optional[Tensor] = None
    #: SPEC 15.  ``[n_bonds, 3]`` bond-current dipole ``d(Omega P)/d tau_b``:
    #: ``f_b d_b`` with ``d_b`` the image-resolved bond vector.  When given, a
    #: uniform field couples to the charge sector through the bonds,
    #: ``-E . sum_b f_b tau_b d_b``, which is single valued under periodic
    #: boundaries and bounded by the bond hardness on every bond graph.
    #: ``None`` keeps the open-boundary coupling ``chi_i -> chi_i - E . r_i``,
    #: which is the same number there, since ``B^T r`` IS the bond vector.
    bond_dipole: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if self.positions.dim() != 2 or self.positions.shape[1] != _SPATIAL_DIM:
            raise ValueError(
                f"positions must have shape [n_atoms, 3], got {tuple(self.positions.shape)}"
            )
        n_atoms = int(self.positions.shape[0])
        if self.incidence.dim() != 2 or self.incidence.shape[0] != n_atoms:
            raise ValueError(
                f"incidence must have shape [{n_atoms}, n_bonds], got "
                f"{tuple(self.incidence.shape)}"
            )
        if self.bond_index.shape != (2, self.incidence.shape[1]):
            raise ValueError(
                f"bond_index must have shape [2, {int(self.incidence.shape[1])}], got "
                f"{tuple(self.bond_index.shape)}"
            )
        expected = {
            "coulomb_matrix": (n_atoms, n_atoms),
            "charge_dipole": (n_atoms, n_atoms * _SPATIAL_DIM),
            "dipole_block": (n_atoms * _SPATIAL_DIM, n_atoms * _SPATIAL_DIM),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value is not None and tuple(value.shape) != shape:
                raise ValueError(
                    f"{name} must have shape {list(shape)}, got {tuple(value.shape)}"
                )

        dipole_sector = ("charge_dipole", "dipole_block")
        if any(getattr(self, name) is not None for name in dipole_sector):
            missing = [name for name in expected if getattr(self, name) is None]
            if missing:
                raise ValueError(
                    f"the pair-operator overrides {missing} are missing. Overriding "
                    "any of the two dipole blocks means the dipole sector is no "
                    "longer the one this module would build from `positions`, so "
                    "all three blocks have to come from the same source -- "
                    "otherwise the solve minimises a functional in which the charges "
                    "see one geometry and the dipoles see another. Measured 0.513 eV "
                    "apart on four atoms in an 8 A cell, where one pair 0.616 A apart "
                    "by minimum image is 7.401 A apart in raw coordinates.\n"
                    "\n"
                    "`coulomb_matrix` on its own is allowed, and means something "
                    "narrower: the charge-charge kernel is replaced while the dipole "
                    "sector stays the open-boundary one built from `positions`. That "
                    "is consistent only when the override really is the "
                    "open-boundary kernel for these positions (or the dipole sector "
                    "is inert). It is NOT a periodic seam."
                )
        if self.pair_kernel not in MULTIPOLE_KERNELS:
            raise ValueError(
                f"pair_kernel must be one of {MULTIPOLE_KERNELS}, got {self.pair_kernel!r}"
            )
        quadrupole_fields = ("charge_quadrupole", "dipole_quadrupole", "quadrupole_pair")
        present = [name for name in quadrupole_fields if getattr(self, name) is not None]
        if present:
            if len(present) != 3 or self.dipole_block is None:
                raise ValueError(
                    "the quadrupole pair overrides go together, and with the three "
                    f"dipole-sector overrides; got {present} with dipole overrides "
                    f"{'present' if self.dipole_block is not None else 'absent'}."
                )
            quadrupole_shapes = {
                "charge_quadrupole": (n_atoms, n_atoms * _QUADRUPOLE_DIM),
                "dipole_quadrupole": (n_atoms * _SPATIAL_DIM, n_atoms * _QUADRUPOLE_DIM),
                "quadrupole_pair": (n_atoms * _QUADRUPOLE_DIM, n_atoms * _QUADRUPOLE_DIM),
            }
            for name, shape in quadrupole_shapes.items():
                if tuple(getattr(self, name).shape) != shape:
                    raise ValueError(
                        f"{name} must have shape {list(shape)}, got "
                        f"{tuple(getattr(self, name).shape)}"
                    )
        if self.bond_dipole is not None and tuple(self.bond_dipole.shape) != (
            int(self.incidence.shape[1]),
            _SPATIAL_DIM,
        ):
            raise ValueError(
                f"bond_dipole must have shape [{int(self.incidence.shape[1])}, 3], got "
                f"{tuple(self.bond_dipole.shape)}"
            )

    @property
    def has_pair_overrides(self) -> bool:
        """Whether the caller supplied the full set of lattice-summed operators."""
        return self.dipole_block is not None

    @property
    def n_atoms(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_bonds(self) -> int:
        return int(self.incidence.shape[1])

    def fingerprint(self) -> Tensor:
        """The configuration a :class:`GeometrySeed` must have been computed at.

        The detached positions themselves: exact, so there are no collisions and
        no false alarms, and ``O(3n)`` beside the ``O(n^2)`` blocks around it.
        Detached because a fingerprint is an identity check, not a term in the
        energy -- it must never contribute a gradient.
        """
        return self.positions.detach()


@dataclass(frozen=True)
class FragmentConstraints:
    """Per-fragment net-charge constraints ``C q = Q``.

    One row per fragment.  The rows are what make the model size consistent:
    they pin the charge of each chemical object separately, so adding a distant
    spectator cannot drain a charged fragment (SPEC 1.3).

    Attributes
    ----------
    matrix:
        ``[n_fragments, n_atoms]`` fragment membership matrix ``C``.
    targets:
        ``[n_fragments]`` net charges ``Q_F`` in units of ``e``.
    reference_weights:
        Optional ``[n_atoms]`` weights for spreading ``Q_F`` into the reference
        charge ``rho``.  Uniform within each fragment by default.
    spin_targets:
        Optional ``[n_fragments]`` total spin quantum numbers ``S_F`` --
        **not** ``2 S_F``.  Present only for a spin-resolved solve (SPEC 9),
        where it adds the second constraint per fragment,

            sum_{i in F} s_i = 2 S_F ,      s_i = q_i^up - q_i^down .

        ``S_F`` is 0 for a closed shell, 1/2 for a doublet, 1 for a triplet;
        ``2 S_F`` is the number of unpaired electrons, ``multiplicity - 1``.
        The factor of two lives in :attr:`spin_rhs` and in
        :func:`~trace_electrostatics.topology.distribute_fragment_spins` and
        nowhere else, because passing ``2 S_F`` where ``S_F`` belongs simply
        doubles every moment in the system and nothing downstream can tell.

        Like the fragmentation itself, the spin of a fragment is a **design
        input**: the model cannot infer whether a transition-metal centre is
        high spin or low spin, and SPEC 9.2 is explicit that the splitting this
        model would produce is a fitted quantity, not a derived one.  There is
        therefore no default -- a spin-resolved solve without ``spin_targets``
        raises rather than quietly assuming every fragment is a singlet.

    Every column of ``matrix`` must sum to one.  That is checked, and what it
    rules out is the failure it was written for: an atom in no fragment is not a
    milder version of the constraint, it is an unconstrained charge -- nothing
    pins it, and ``rho`` places none of the net charge on it, so it is free to
    drain the rest of the system exactly the way the distant spectators of
    SPEC 1.3 do.  That used to be accepted silently.  An atom counted in more
    than one fragment is rejected by the same test.

    Note the check is on the column SUM, not on the entries: a fractional
    membership such as ``[0.5, 0.5]`` sums to one and is accepted.  That is a
    weaker statement than "every atom belongs to exactly one fragment", and it
    is deliberate -- a fractional column is still a well-posed linear
    constraint, ``C rho = Q`` still holds because
    :func:`~trace_electrostatics.topology.distribute_fragment_charges` uses the
    same matrix, and nothing downstream requires the entries to be 0/1.  Every
    matrix this package builds itself
    (:func:`~trace_electrostatics.topology.fragment_constraint_matrix`,
    :meth:`from_fragment_ids`) is a genuine 0/1 partition.
    """

    matrix: Tensor
    targets: Tensor
    reference_weights: Optional[Tensor] = None
    spin_targets: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if self.matrix.dim() != 2:
            raise ValueError(f"matrix must be 2-D, got {tuple(self.matrix.shape)}")
        if self.targets.reshape(-1).numel() != self.matrix.shape[0]:
            raise ValueError(
                f"targets must have {int(self.matrix.shape[0])} entries, got "
                f"{self.targets.reshape(-1).numel()}"
            )
        if (
            self.spin_targets is not None
            and self.spin_targets.reshape(-1).numel() != self.matrix.shape[0]
        ):
            raise ValueError(
                f"spin_targets must have {int(self.matrix.shape[0])} entries (one "
                f"S_F per fragment), got {self.spin_targets.reshape(-1).numel()}"
            )
        if self.matrix.numel():
            membership = self.matrix.sum(dim=0)
            if not bool(torch.isclose(membership, torch.ones_like(membership)).all()):
                orphan = int((membership == 0).sum())
                shared = int((membership > 1).sum())
                raise ValueError(
                    "the fragment matrix must be a partition: every column must sum "
                    f"to 1. Got {orphan} atom(s) in no fragment and {shared} atom(s) "
                    "in more than one. An atom in no fragment carries no constraint "
                    "and no share of rho, so its charge is free to run away into the "
                    "rest of the system (SPEC 1.3)."
                )

    @classmethod
    def from_fragment_ids(
        cls,
        fragment_id: Tensor,
        targets: Tensor,
        *,
        num_fragments: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        reference_weights: Optional[Tensor] = None,
        spin_targets: Optional[Tensor] = None,
    ) -> "FragmentConstraints":
        """Build the constraints from per-atom fragment labels.

        ``spin_targets`` is ``S_F``, not ``2 S_F``; see the class docstring.
        """
        matrix = fragment_constraint_matrix(fragment_id, num_fragments, dtype=dtype)
        return cls(
            matrix=matrix,
            targets=targets.reshape(-1).to(matrix.dtype),
            reference_weights=reference_weights,
            spin_targets=(
                None if spin_targets is None else spin_targets.reshape(-1).to(matrix.dtype)
            ),
        )

    @property
    def n_fragments(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def spin_resolved(self) -> bool:
        """Whether per-fragment spin constraints were supplied."""
        return self.spin_targets is not None

    @property
    def spin_rhs(self) -> Tensor:
        """``2 S_F``, the right-hand side actually applied to ``sum_F s_i``.

        The one place the SPEC 9 factor of two is applied on this class.
        """
        if self.spin_targets is None:
            raise ValueError(
                "these constraints carry no spin targets; the solve is closed shell"
            )
        return 2.0 * self.spin_targets.reshape(-1)


@dataclass(frozen=True)
class GeometrySeed:
    """Starting point for the fixed-iteration solve. **Geometry-determined.**

    The seed is part of the *definition* of the energy: with a fixed, truncated
    iteration count, ``U`` depends on where the iteration started.  It must
    therefore be a pure function of the current configuration -- SPEC 2
    prescribes the antisymmetrised attention logit, which is exactly that.

    Seeding from the previous MD step (a "warm start") is the failure this type
    exists to prevent.  It would make ``U`` history-dependent: the same
    configuration reached from two trajectories would have two different
    energies, the force would no longer be the gradient of anything, and the
    conservativeness that the whole fixed-iteration design buys would be gone.

    Three things enforce that, because the first two on their own do not
    ---------------------------------------------------------------------
    1. **A geometry fingerprint.**  :meth:`for_geometry` stamps the seed with the
       configuration it was computed at, and :func:`solve_charges` re-checks it
       against the geometry actually being solved.  This is the barrier that
       catches the real failure mode: in MD the geometry moves every step, so a
       seed carried over from step ``k-1`` is stamped with step ``k-1``'s
       positions and is rejected at step ``k``.  Bind every seed this way.
    2. **Rejection of solver output.**  :func:`solve_charges` marks the tensors
       it returns, and refuses a seed built from one of them.  This catches the
       same-geometry case a fingerprint cannot see -- re-solving one
       configuration from its own previous answer, which measurably shifts the
       energy by 3.346e-02 eV -- and it catches it whether the tensor arrives
       through the constructor or through :meth:`from_bond_logits`.
    3. **Provenance.**  A non-empty string naming what the seed was computed
       from, rejected if it names history.  This one is only a label and is the
       weakest of the three: a wrong seed with an honest string is caught by
       (1) and (2), which is the point of having them.

    Attributes
    ----------
    t:
        ``[n_bonds]`` initial split charges.  In a spin-resolved solve this is
        the **charge channel** ``t^c = t^up + t^down``, which is the same
        quantity it is in a closed-shell solve -- so one seed head serves both,
        and an unpolarised spin solve starts where the closed-shell solve
        starts.
    mu_hat:
        ``[n_atoms, 3]`` initial reduced dipoles.
    provenance:
        Non-empty description of the geometry-determined quantity used, e.g.
        ``"antisymmetrised attention logit"`` or ``"zero"``.
    geometry_fingerprint:
        The configuration the seed was computed at, from
        :meth:`Geometry.fingerprint`.  ``None`` means unbound: the seed is then
        accepted (a constant seed genuinely has no geometry to bind to) but
        nothing can check where it came from, so bind it whenever the seed is
        computed from anything at all.
    t_spin:
        Optional ``[n_bonds]`` initial spin-channel transfer
        ``t^s = t^up - t^down`` (SPEC 9).  ``None`` means zero, which is the
        right default: it is geometry-independent, and it is what makes an
        unpolarised spin solve start from exactly the closed-shell starting
        point.  A learned spin seed must be geometry-determined like ``t`` --
        the antisymmetrised edge logit of a spin-channel head -- and is bound
        and screened by the same three barriers.
    """

    t: Tensor
    mu_hat: Tensor
    provenance: str
    geometry_fingerprint: Optional[Tensor] = None
    t_spin: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if self.t.dim() != 1:
            raise ValueError(f"seed t must be 1-D [n_bonds], got {tuple(self.t.shape)}")
        if self.mu_hat.dim() != 2 or self.mu_hat.shape[1] != _SPATIAL_DIM:
            raise ValueError(
                f"seed mu_hat must be [n_atoms, 3], got {tuple(self.mu_hat.shape)}"
            )
        if self.t_spin is not None:
            if self.t_spin.dim() != 1:
                raise ValueError(
                    f"seed t_spin must be 1-D [n_bonds], got {tuple(self.t_spin.shape)}"
                )
            if self.t_spin.numel() != self.t.numel():
                raise ValueError(
                    f"seed t_spin must have {self.t.numel()} entries to match t, got "
                    f"{self.t_spin.numel()}"
                )
            _reject_solver_output(self.t_spin, "t_spin")
        _reject_solver_output(self.t, "t")
        _reject_solver_output(self.mu_hat, "mu_hat")
        if not isinstance(self.provenance, str) or not _visible(self.provenance):
            raise ValueError(
                "GeometrySeed.provenance must be a non-empty string naming the "
                "geometry-determined quantity the seed was computed from. The seed may "
                "not come from a previous MD step: that makes U history-dependent and "
                "forfeits conservative forces."
            )
        named = _names_history(self.provenance)
        if named is not None:
            raise ValueError(
                f"GeometrySeed.provenance says {self.provenance!r}, which names "
                f"history ({named!r}). A seed carried over from an earlier step, an "
                "earlier solve or a cache makes U history-dependent: the same "
                "configuration then has two different energies, the force stops being "
                "the gradient of anything, and the conservativeness the "
                "fixed-iteration design exists to buy is gone (SPEC 2). Compute the "
                "seed from the current configuration -- the antisymmetrised attention "
                "logit -- and bind it with GeometrySeed.for_geometry."
            )

    @classmethod
    def for_geometry(
        cls,
        geometry: "Geometry",
        *,
        t: Tensor,
        mu_hat: Optional[Tensor] = None,
        provenance: str,
        t_spin: Optional[Tensor] = None,
    ) -> "GeometrySeed":
        """Bind a seed to the configuration it was computed at.

        This is the constructor to use.  ``solve_charges`` re-checks the stamp
        against the geometry it is handed, so a seed that travelled from another
        step, another trajectory or another system fails loudly instead of
        quietly redefining the energy.
        """
        if mu_hat is None:
            mu_hat = torch.zeros(
                geometry.n_atoms,
                _SPATIAL_DIM,
                dtype=t.dtype,
                device=t.device,
            )
        return cls(
            t=t.reshape(-1),
            mu_hat=mu_hat,
            provenance=provenance,
            geometry_fingerprint=geometry.fingerprint(),
            t_spin=None if t_spin is None else t_spin.reshape(-1),
        )

    @classmethod
    def zeros(
        cls,
        n_bonds: int,
        n_atoms: int,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        geometry: Optional["Geometry"] = None,
    ) -> "GeometrySeed":
        """The trivially geometry-independent seed ``m_0 = 0``.

        Always admissible (a constant is a function of geometry, if a dull one)
        and the right choice when no learned seed is available; a learned
        geometry-determined seed simply reaches a lower energy in the same
        number of iterations.  It is the one seed that needs no fingerprint --
        there is no configuration it could have come from -- though passing
        ``geometry`` binds it anyway, which also pins the shapes.
        """
        if geometry is not None:
            n_bonds, n_atoms = geometry.n_bonds, geometry.n_atoms
            if dtype is None:
                dtype = geometry.positions.dtype
            if device is None:
                device = geometry.positions.device
        dtype = torch.get_default_dtype() if dtype is None else dtype
        return cls(
            t=torch.zeros(n_bonds, dtype=dtype, device=device),
            mu_hat=torch.zeros(n_atoms, _SPATIAL_DIM, dtype=dtype, device=device),
            provenance="zero",
            geometry_fingerprint=None if geometry is None else geometry.fingerprint(),
        )

    @classmethod
    def from_bond_logits(
        cls,
        bond_logits: Tensor,
        n_atoms: int,
        *,
        mu_hat: Optional[Tensor] = None,
        geometry: Optional["Geometry"] = None,
        spin_bond_logits: Optional[Tensor] = None,
    ) -> "GeometrySeed":
        """Seed from per-bond logits, e.g. the antisymmetrised attention logit.

        Use :func:`trace_electrostatics.topology.antisymmetrise_edge_scalar` to
        reduce directed edge logits ``s_ij`` to the per-bond antisymmetric part
        ``0.5 * (s_ij - s_ji)`` first.

        ``spin_bond_logits`` seeds the spin channel ``t^s`` of a spin-resolved
        solve (SPEC 9) the same way, and is subject to the same requirement: it
        must be a function of the current configuration alone.  Omit it and the
        spin channel starts at zero.

        The provenance this stamps is a claim about the caller's input, not
        something this method can verify -- ``from_bond_logits`` cannot tell an
        attention logit from last step's ``t``.  Pass ``geometry`` so the seed
        is bound and the claim becomes checkable; the tensors are screened
        against solver output either way.
        """
        bond_logits = bond_logits.reshape(-1)
        spin_logits = None if spin_bond_logits is None else spin_bond_logits.reshape(-1)
        if geometry is not None and mu_hat is None:
            mu_hat = torch.zeros(
                geometry.n_atoms,
                _SPATIAL_DIM,
                dtype=bond_logits.dtype,
                device=bond_logits.device,
            )
        if mu_hat is None:
            mu_hat = torch.zeros(
                n_atoms, _SPATIAL_DIM, dtype=bond_logits.dtype, device=bond_logits.device
            )
        return cls(
            t=bond_logits,
            mu_hat=mu_hat,
            provenance="antisymmetrised edge logit (geometry-determined)",
            geometry_fingerprint=None if geometry is None else geometry.fingerprint(),
            t_spin=spin_logits,
        )


@dataclass(frozen=True)
class SolverOptions:
    """Knobs that do not belong to the physics.

    Attributes
    ----------
    thole_a:
        Thole damping parameter; must be ``< 1`` strictly (SPEC 1.6).
    validate:
        ``None`` defers to the global validation mode in
        :mod:`trace_electrostatics.diagnostics`; ``True``/``False`` force the
        expensive O(n^3) checks on or off for this call.
    check_is_minimum:
        SPEC 2.1 item 4 -- verify the returned point is a *minimum*, not merely
        stationary.  ``None`` follows ``validate``; ``True``/``False`` decide it
        on its own.  It is separate from ``validate`` because it answers a
        different question from item 3: item 3 asks whether the *charge sector*
        is stable once the dipoles have relaxed, item 4 asks whether the full
        coupled point is a minimum at all, and Newton-type solvers converge
        silently onto saddles and then report a negative polarizability.  A
        caller that exposes item 3 through a config flag and leaves this one
        bundled has item 4 silently off.
    expect_converged:
        When ``True`` the validation path also tests stationarity.  Leave it
        ``False`` for the production fixed-iteration solve, where a non-zero
        projected gradient is the intended behaviour, not an error.
    eigenvalue_tol, gradient_tol, fragment_charge_tol:
        Tolerances handed to the corresponding assertions.
    operator_symmetry_rtol:
        Relative tolerance for the exchange symmetry of a supplied pair operator
        (see :class:`Geometry`).
    multipole_kernel:
        ``"thole"`` (default) or ``"gaussian"``.  Which smeared density the
        dipole (and quadrupole) blocks are derivatives of.  ``"gaussian"`` makes
        every block of K a Coulomb inner product of Gaussian multipoles, so K is
        positive definite for EVERY geometry by construction (SPEC 12); it is
        REQUIRED once the quadrupole sector is on, because the Thole-coupled
        (q, mu, Theta) Hessian was measured indefinite (-0.135 eV) with every
        block and every pairwise sub-block positive definite.
    gaussian_a:
        ``0 < a < 1``: the fraction of each on-site stiffness the Gaussian self
        energy may use (SPEC 12).  The Gaussian counterpart of ``thole_a``.
    relax_quadrupole_seed:
        One exact-line-search block-Jacobi step on the quadrupoles after the
        dipole step, when the quadrupole sector is on.  It lands a permanent
        quadrupole (``chi_theta``) at its on-site answer before CG starts, and
        like the dipole step it can only lower ``U(m_0)``.
    measure_fractional_spin_error:
        Report the model's own fractional-spin (flat-plane) error on every
        spin-resolved solve, as
        :attr:`SolveDiagnostics.fractional_spin` (SPEC 9.2).  ``None``, the
        default, means "on whenever the spin sector is on"; ``False`` switches it
        off, ``True`` is the same as ``None`` because there is nothing to measure
        on a closed-shell solve.

        Default ON, unlike the O(n^3) checks, for two reasons.  It costs
        ``O(n_atoms)`` -- the *frozen-bond* form, which is a rigorous upper bound
        on the relaxed curvature, not an approximation to it -- and no host
        synchronisation, so it is affordable as training telemetry.  And it
        measures an error that is present in **every** parameterisation, correct
        ones included: a user who turns the spin sector on has, by that act,
        taken on the artifact, and should not have to ask a second time to see
        how big it is.  :func:`fractional_spin_error` gives the exact relaxed
        number when the extra factorisation is affordable.
    """

    thole_a: float = DEFAULT_THOLE_A
    validate: Optional[bool] = None
    check_is_minimum: Optional[bool] = None
    expect_converged: bool = False
    #: One exact-line-search block-Jacobi step on the dipoles from the seed
    #: charges before CG (SPEC 2.0.1). Physics, not learning: see
    #: _relax_dipole_seed. On by default; it can only lower U(m_0).
    relax_dipole_seed: bool = True
    relax_quadrupole_seed: bool = True
    multipole_kernel: str = "gaussian"
    gaussian_a: float = DEFAULT_GAUSSIAN_A
    eigenvalue_tol: float = 1e-8
    gradient_tol: float = 1e-6
    fragment_charge_tol: float = 1e-10
    operator_symmetry_rtol: float = DEFAULT_OPERATOR_SYMMETRY_RTOL
    measure_fractional_spin_error: Optional[bool] = None
    #: SPEC 17. ``"converged"`` (default): the energy is the constrained
    #: minimum, solved to ``solve_rtol`` and differentiated implicitly, exact to
    #: every order. ``"fixed_k"``: the legacy SPEC 2 energy, ``U`` after
    #: exactly ``n_iter`` CG steps differentiated through.
    solve_mode: str = DEFAULT_SOLVE_MODE
    solve_rtol: float = DEFAULT_SOLVE_RTOL
    solve_max_iter: int = DEFAULT_SOLVE_MAX_ITER

    def __post_init__(self) -> None:
        if self.solve_mode not in SOLVE_MODES:
            raise ValueError(
                f"solve_mode must be one of {SOLVE_MODES}, got {self.solve_mode!r}"
            )
        if not 0.0 < float(self.solve_rtol) < 1.0:
            raise ValueError(f"solve_rtol must lie in (0, 1), got {self.solve_rtol}")
        if isinstance(self.solve_max_iter, bool) or not isinstance(self.solve_max_iter, int) \
                or self.solve_max_iter < 1:
            raise ValueError(f"solve_max_iter must be a positive int, got {self.solve_max_iter!r}")
        if not self.thole_a < THOLE_A_MAX:
            raise ValueError(
                f"Thole damping requires a < {THOLE_A_MAX} strictly, got a={self.thole_a}. "
                "sup_r alpha |T| = a exactly, so a PAIR of sites loses positive "
                "definiteness at a = 1. (a < 1 does not protect three or more close "
                "sites; the converged solve detects that case at run time.)"
            )
        if self.thole_a <= 0.0:
            raise ValueError(f"Thole damping requires a > 0, got a={self.thole_a}")
        if self.multipole_kernel not in MULTIPOLE_KERNELS:
            raise ValueError(
                f"multipole_kernel must be one of {MULTIPOLE_KERNELS}, got "
                f"{self.multipole_kernel!r}"
            )
        if not 0.0 < self.gaussian_a < GAUSSIAN_A_MAX:
            raise ValueError(
                f"gaussian_a must satisfy 0 < a < {GAUSSIAN_A_MAX}, got "
                f"{self.gaussian_a}: at a >= 1 the on-site residual of the Gram "
                "construction can vanish and K is no longer positive definite."
            )


@dataclass(frozen=True)
class ElectrostaticSystem:
    """The assembled quadratic form ``U(m) = 0.5 m^T K m + c^T m + u_const``.

    Holding the assembled system as a value (rather than hiding it inside the
    solve) is what lets the observables module compute ``alpha = D^T K^-1 D``
    and the diagnostics module inspect the *coupled* curvature at the solution.

    Attributes
    ----------
    K:
        ``[n_m, n_m]`` Hessian, ``n_m = n_bonds + 3 n_atoms``, ordered
        ``(t, mu_hat)`` -- or ``n_m = 2 n_bonds + 3 n_atoms`` ordered
        ``(t^c, t^s, mu_hat)`` when the spin sector of SPEC 9 is on.  In the
        spin case the layout is::

            K = [ B^T J B + diag(kappa)          0                  B^T T01        ]
                [          0            B^T diag(J^x) B + diag(k^s)     0          ]
                [       T01^T B                   0              A_mu + T_Thole    ]

        The two structural zeros are the physics: **the Coulomb kernel couples
        TOTAL charges**, so ``J`` appears once, in the ``t^c`` block, and not
        once per spin channel; and the induced dipole responds to the total
        density, so it does not see ``t^s`` at all.  A ``J`` anywhere in the
        middle row would be the characteristic error of this feature.
    c:
        ``[n_m]`` linear term.
    u_const:
        Scalar part, ``0.5 rho^T J rho + chi^T rho`` -- with ``chi`` already
        carrying the applied-field shift ``chi_i -> chi_i - E . r_i`` when an
        ``external_field`` was supplied, so the field contributes here too.
        Independent of ``m`` but **not** of geometry, so it contributes to the
        forces and must be kept.
    rho:
        ``[n_atoms]`` constraint-carrying reference charge.
    constraint_t:
        ``[n_fragments, n_bonds]`` constraint ``C B`` acting on the bond
        variables -- or the ``[2 n_fragments, 2 n_bonds]`` block diagonal
        ``diag(C B, C B)`` of SPEC 9 when the spin sector is on, so that it
        always lives in the same space as the rows of :meth:`charge_block` and
        can be handed straight to the SPEC 2.1 curvature assertions.  Zero rows
        are normal (and expected) whenever no bond crosses a fragment boundary.
    coulomb:
        ``[n_atoms, n_atoms]`` charge kernel with ``eta`` on the diagonal.
        Acts on **total** charges in both the closed-shell and the spin case.
    charge_dipole:
        ``[n_atoms, 3 n_atoms]`` block coupling atomic charges to reduced
        dipoles.  Total charges, as above.
    dipole_block:
        ``[3 n_atoms, 3 n_atoms]`` block ``A_mu + T_Thole``.
    operator_asymmetry:
        Measured exchange asymmetry of each pair operator the caller supplied,
        or ``None`` when the kernels were built here.
    spin_resolved:
        Whether the spin sector of SPEC 9 is present.  ``False`` for every
        closed-shell system, and then nothing else on this class differs in
        shape or meaning from what it was before spin existed.
    rho_spin:
        ``[n_atoms]`` constraint-carrying reference spin population, satisfying
        ``C rho^s = 2 S``.  ``None`` when closed shell.
    exchange:
        ``[n_atoms]`` on-site exchange ``J^x``.  ``None`` when closed shell.
    """

    K: Tensor
    c: Tensor
    u_const: Tensor
    rho: Tensor
    constraint_t: Tensor
    coulomb: Tensor
    charge_dipole: Tensor
    dipole_block: Tensor
    operator_asymmetry: Optional[Dict[str, Tensor]] = None
    spin_resolved: bool = False
    rho_spin: Optional[Tensor] = None
    exchange: Optional[Tensor] = None
    constraint_bond: Optional[Tensor] = None
    quadrupole_resolved: bool = False
    charge_quadrupole: Optional[Tensor] = None
    dipole_quadrupole: Optional[Tensor] = None
    quadrupole_block: Optional[Tensor] = None

    @property
    def bond_constraint(self) -> Tensor:
        """The single ``[n_fragments, n_bonds]`` block ``C B``.

        Identical to :attr:`constraint_t` for a closed-shell system.  With spin
        on, ``constraint_t`` is ``diag(C B, C B)`` and this is the block it is
        built from, kept because the projector onto the feasible set is then
        ``diag(P, P)`` with ``P = I - (C B)^+ (C B)``: one pseudo-inverse of the
        narrow matrix, applied to both halves, instead of one of a matrix twice
        as wide.  That is exact -- the pseudo-inverse of a block diagonal is the
        block diagonal of the pseudo-inverses -- and it is the difference
        between one SVD and one eight times the cost, on every solve.
        """
        return self.constraint_t if self.constraint_bond is None else self.constraint_bond

    @property
    def n_bonds(self) -> int:
        """Number of *bonds*, which is half the bond-sector width when spin is on."""
        columns = int(self.constraint_t.shape[1])
        return columns // 2 if self.spin_resolved else columns

    @property
    def n_bond_variables(self) -> int:
        """Width of the bond sector of ``K``: ``n_bonds``, or ``2 n_bonds`` with spin."""
        return int(self.constraint_t.shape[1])

    @property
    def n_atoms(self) -> int:
        return int(self.rho.numel())

    def energy(self, m: Tensor) -> Tensor:
        """``U(m) = 0.5 m^T K m + c^T m + u_const``."""
        return 0.5 * (m * (self.K @ m)).sum() + (self.c * m).sum() + self.u_const

    def gradient(self, m: Tensor) -> Tensor:
        """``dU/dm = K m + c`` (unprojected)."""
        return self.K @ m + self.c

    @property
    def dipole_offset(self) -> int:
        """Index of the first ``mu_hat`` entry in ``m``."""
        return self.n_bond_variables

    @property
    def quadrupole_offset(self) -> int:
        """Index of the first ``Theta_hat`` entry in ``m`` (== ``len(m)`` when off)."""
        return self.n_bond_variables + _SPATIAL_DIM * self.n_atoms

    def multipole_block(self) -> Tensor:
        """``K`` over the atomic moments: ``(mu_hat)``, or ``(mu_hat, Theta_hat)``.

        The block the SPEC 2.1 Schur-complement check has to see whole: once the
        quadrupole sector is on, checking the dipole block alone would leave the
        Theta modes -- and the three-way coupling that the Thole route measured
        indefinite -- outside the check.
        """
        width = self.n_bond_variables
        return self.K[width:, width:]

    def charge_block(self) -> Tensor:
        """Bond-space charge block ``B^T J B + diag(kappa)``.

        With the spin sector on this is the whole ``[2 n_bonds, 2 n_bonds]``
        non-dipole block, ``diag(B^T J B + diag(kappa), B^T diag(J^x) B +
        diag(kappa^s))`` -- the ``H_qq`` that SPEC 2.1 item 3 forms the Schur
        complement of.  Returning only the charge channel would leave the spin
        mode out of the one check that would see it soften.
        """
        width = self.n_bond_variables
        return self.K[:width, :width]

    def coupling_block(self) -> Tensor:
        """Coupling block as ``[3 n_atoms, n_bond_variables]``, the ``T01`` of SPEC 2.1.

        Its spin columns are exactly zero: the induced dipole responds to the
        total density, not to the spin density (SPEC 9).
        """
        width = self.n_bond_variables
        return self.K[width:, :width]


@dataclass
class SolveDiagnostics:
    """Solver telemetry.  Tensors, so nothing here forces a host sync.

    Attributes
    ----------
    energy_terms:
        The additive decomposition ``energy = quadratic + linear + reference``,
        plus ``bond_hardness_in_quadratic``, which is a **component of**
        ``quadratic`` and not a fourth term.  It used to be called
        ``bond_hardness``, sitting in the same flat dict as the three terms that
        do add up, so summing the four double counted it.
    operator_asymmetry:
        Measured exchange asymmetry of each supplied pair operator, empty when
        the solve ran on its own kernels.
    constraint_violation:
        Largest per-fragment violation of ``sum_F q_i = Q_F``.
    spin_constraint_violation:
        Largest per-fragment violation of ``sum_F s_i = 2 S_F`` (SPEC 9), or
        ``None`` for a closed-shell solve.  Reported separately from the charge
        violation rather than maxed into it, because the two answer different
        questions and a spin constraint can be violated while the charge one is
        exact.
    fractional_spin:
        The model's own fractional-spin (flat-plane) error at this
        parameterisation, or ``None`` for a closed-shell solve or when
        ``SolverOptions.measure_fractional_spin_error`` is off.  This is
        **telemetry about a known defect of the functional form**, not an
        assertion: ``0.5 J^x s^2`` violates the flat-plane condition at every
        parameterisation and nothing here can make it stop (SPEC 9.2).  It is
        the cheap frozen-bond form, so
        :attr:`~trace_electrostatics.diagnostics.FractionalSpinError.curvature`
        is an upper bound; :func:`fractional_spin_error` relaxes the bond
        transfer for the exact value.
    """

    n_iter: int
    seed_provenance: str
    initial_residual_norm: Tensor
    residual_norm: Tensor
    constraint_violation: Tensor
    energy_terms: Dict[str, Tensor]
    system: ElectrostaticSystem
    operator_asymmetry: Optional[Dict[str, Tensor]] = None
    curvature: Optional[Dict[str, Tensor]] = None
    minimum: Optional[Dict[str, Tensor]] = None
    spin_constraint_violation: Optional[Tensor] = None
    fractional_spin: Optional[FractionalSpinError] = None


@dataclass
class SolveResult:
    """Result of :func:`solve_charges`.

    Attributes
    ----------
    t:
        ``[n_bonds]`` bond variables ``tau``, in the ``lo -> hi`` orientation of
        ``bond_index``.  The PHYSICAL split charge is ``f(r_b) tau_b`` with ``f``
        the dissociation envelope (SPEC 1.4.1 / 17.4); ``q = B diag(f) tau +
        rho`` below already includes it.  In a spin-resolved solve this is the charge channel
        ``t^c = t^up + t^down``, i.e. the *total* transfer along each bond --
        the same quantity a closed-shell solve returns, so downstream code that
        reads ``t`` keeps meaning what it meant.
    rho:
        ``[n_atoms]`` constraint-carrying reference charge.
    mu:
        ``[n_atoms, 3]`` dipoles in ``e*A`` (``mu = ell * mu_hat``).
    q:
        ``[n_atoms]`` atomic charges, ``q = B diag(f) t + rho``.
    energy:
        Scalar electrostatic energy in eV.  Differentiable with respect to
        positions and to every parameter tensor: by default the converged
        minimum with implicit derivatives of every order (SPEC 17); under
        ``solve_mode="fixed_k"`` the CG iterations are part of the graph.
    diagnostics:
        :class:`SolveDiagnostics`.

    Spin sector (SPEC 9), all ``None`` for a closed-shell solve
    -----------------------------------------------------------
    t_spin:
        ``[n_bonds]`` spin-channel transfer ``t^s = t^up - t^down``.
    rho_spin:
        ``[n_atoms]`` reference spin population, ``C rho^s = 2 S``.
    s:
        ``[n_atoms]`` spin populations ``s = q^up - q^down = B t^s + rho^s``.
        Summed over a fragment this is ``2 S_F``, the number of unpaired
        electrons, not ``S_F``.

    Use :meth:`spin_channels` for the per-spin charges and transfers.
    """

    t: Tensor
    rho: Tensor
    mu: Tensor
    q: Tensor
    energy: Tensor
    diagnostics: SolveDiagnostics
    t_spin: Optional[Tensor] = None
    rho_spin: Optional[Tensor] = None
    s: Optional[Tensor] = None
    #: Quadrupole sector (SPEC 10/12), ``None`` when off. ``theta_hat`` is the
    #: reduced 2e coefficient vector ``[n_atoms, 5]``; ``theta`` the cartesian
    #: traceless second moment ``[n_atoms, 3, 3]`` in e*A^2 (NOT Buckingham's,
    #: which is 3/2 of it).
    theta_hat: Optional[Tensor] = None
    theta: Optional[Tensor] = None

    @property
    def spin_resolved(self) -> bool:
        """Whether this solve carried the spin sector of SPEC 9."""
        return self.s is not None

    def spin_channels(self) -> Dict[str, Tensor]:
        """The per-spin view: ``q_up``, ``q_down``, ``t_up``, ``t_down``.

        ``q^up = (q + s)/2``, ``q^down = (q - s)/2`` and likewise for the bond
        transfers -- the exact inverse of the basis this solver works in.  The
        solve itself stays in ``(charge, spin)``, where the Coulomb kernel
        touches one block and the closed-shell limit is exact; this is the
        reporting view.
        """
        if self.s is None or self.t_spin is None:
            raise ValueError(
                "this solve was closed shell; there are no spin channels. Pass "
                "j_exchange on ElectrostaticParams and spin_targets on "
                "FragmentConstraints for a spin-resolved solve (SPEC 9)."
            )
        q_up, q_down = charge_spin_to_spin_channels(self.q, self.s)
        t_up, t_down = charge_spin_to_spin_channels(self.t, self.t_spin)
        return {"q_up": q_up, "q_down": q_down, "t_up": t_up, "t_down": t_down}


def gaussian_widths(
    params: ElectrostaticParams, gaussian_a: float
) -> tuple[Tensor, Optional[Tensor]]:
    """Per-site Gaussian widths ``(sigma_mu, sigma_Theta)`` of SPEC 12.

    Derived from the on-site stiffnesses by
    :func:`~trace_electrostatics.kernels.gaussian_multipole_width`, so that the
    Gram self term takes at most the fraction ``gaussian_a`` of the stiffest
    direction and K is positive definite for every geometry.  The charge width
    is ``params.sigma`` itself: the hardness floor ``eta >= 2 K_E/(sqrt(pi)
    sigma)`` already exceeds the Gram self energy ``sqrt(2) K_E/(sqrt(pi) sigma)``
    by sqrt(2).  ``sigma_Theta`` is ``None`` when the quadrupole sector is off.

    The single construction site: :func:`assemble_system` and
    ``energy._direct_pair_blocks`` both call it, so the operator the solve
    minimises and the energy that is reported cannot drift apart.
    """
    ell = params.ell.reshape(-1)
    sigma_mu = gaussian_multipole_width(params.A_mu, ell, 1, gaussian_a)
    sigma_theta = (
        None
        if params.A_Theta is None
        else gaussian_multipole_width(params.A_Theta, ell, 2, gaussian_a)
    )
    return sigma_mu, sigma_theta


def _combined_width(sigma_row: Tensor, sigma_column: Tensor) -> Tensor:
    """``s_ij = sqrt(sigma_row_i^2 + sigma_column_j^2)``, ``[n, n]``."""
    return torch.sqrt(sigma_row.unsqueeze(1) ** 2 + sigma_column.unsqueeze(0) ** 2)


def gaussian_dipole_damping(
    r: Tensor, sigma: Tensor, sigma_mu: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """``(lambda_3 of q_i-mu_j, lambda_3 and lambda_5 of mu_i-mu_j)``, each ``[n, n]``.

    Two different combined widths, and the asymmetry is the point: the charge
    of ``i`` is a Gaussian of width ``sigma_i`` and the dipole of ``j`` one of
    width ``sigma_mu_j``, so their Coulomb overlap is damped on
    ``sqrt(sigma_i^2 + sigma_mu_j^2)``.  Using one width for both ranks would
    still be a valid Gram matrix only if the two widths were equal.
    """
    lambda_3_charge_dipole = gaussian_damping(r, _combined_width(sigma, sigma_mu))[0]
    lambda_3, lambda_5, _, _ = gaussian_damping(r, _combined_width(sigma_mu, sigma_mu))
    return lambda_3_charge_dipole, lambda_3, lambda_5


def quadrupole_pair_tensors(
    r_vec: Tensor,
    r: Tensor,
    keep: Tensor,
    sigma: Tensor,
    sigma_mu: Tensor,
    sigma_theta: Tensor,
    ell: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """The three Gaussian pair tensors of the quadrupole sector, in REDUCED moments.

    With ``r_vec[i, j] = r_i - r_j`` and ``W`` the orthonormal ``2e`` basis:

    * ``t02[i, j, m]``      coefficient of ``q_i Theta_hat_{j,m}``,
      ``ell_j^2 T02(r_ij) : W_m``, width ``sqrt(sigma_i^2 + sigma_Theta_j^2)``;
    * ``t12[i, j, a, m]``   coefficient of ``mu_hat_{i,a} Theta_hat_{j,m}``,
      ``ell_i ell_j^2 T12(r_ij)_abc W_m,bc``, width ``sqrt(sigma_mu_i^2 +
      sigma_Theta_j^2)``;
    * ``t22[i, j, m, n]``   coefficient of ``Theta_hat_{i,m} Theta_hat_{j,n}``
      (unordered pair counted once), ``ell_i^2 ell_j^2 W_m : T22(r_ij) : W_n``,
      width ``sqrt(sigma_Theta_i^2 + sigma_Theta_j^2)``.

    Each ordered pair ``(i, j)`` of the two cross-rank tensors is its own
    physical term -- ``q_i Theta_j`` and ``q_j Theta_i`` are different
    interactions -- so neither is symmetrised; ``K`` is symmetric because each
    enters it together with its transpose.  ``keep`` zeroes the diagonal and any
    coincident pair (the kernels' ``DEGENERATE_SEPARATION`` shield).

    Memory: ``T22`` is materialised as ``[n, n, 3, 3, 3, 3]`` before it is
    contracted, ~2.6 kB per pair with its temporaries; ``energy.assemble``
    bounds it with ``max_operator_bytes``.
    """
    basis = quadrupole_cartesian_basis().to(dtype=r_vec.dtype, device=r_vec.device)
    ell_squared = ell * ell

    lambda_3, lambda_5, _, _ = gaussian_damping(r, _combined_width(sigma, sigma_theta))
    t02 = torch.einsum(
        "ijab,mab->ijm", charge_quadrupole_tensor(r_vec, r, lambda_3, lambda_5), basis
    )
    t02 = t02 * (keep * ell_squared.unsqueeze(0)).unsqueeze(-1)

    _, lambda_5, lambda_7, _ = gaussian_damping(r, _combined_width(sigma_mu, sigma_theta))
    t12 = torch.einsum(
        "ijabc,mbc->ijam", dipole_quadrupole_tensor(r_vec, r, lambda_5, lambda_7), basis
    )
    t12 = t12 * (keep * ell.unsqueeze(1) * ell_squared.unsqueeze(0))[..., None, None]

    _, lambda_5, lambda_7, lambda_9 = gaussian_damping(
        r, _combined_width(sigma_theta, sigma_theta)
    )
    t22 = torch.einsum(
        "mab,ijabcd,ncd->ijmn",
        basis,
        quadrupole_interaction_tensor(r_vec, r, lambda_5, lambda_7, lambda_9),
        basis,
    )
    t22 = t22 * (keep * ell_squared.unsqueeze(1) * ell_squared.unsqueeze(0))[
        ..., None, None
    ]
    return t02, t12, t22


def crossing_bond_selector(
    constraint_matrix: Tensor, bond_index: Tensor, n_bonds: int
) -> Tensor:
    """``[n_crossing, n_bonds]`` rows ``e_b^T``, one per bond joining two fragments.

    SPEC 17.2 -- the fragment constraint on the bond variables.  A fragment is
    a set of atoms whose net charge is FIXED (SPEC 1.3).  Charge flows only
    through bonds, so a bond inside a fragment can never change it, and a bond
    that joins two fragments must carry no transfer at all: ``tau_b = 0``.

    The package used to impose only the weaker ``sum_{b crossing F} f_b tau_b =
    0`` per fragment, which kept each net charge but let current enter a
    fragment through one crossing bond and leave through another -- charge
    passing from one neighbouring fragment to another THROUGH this one.  An
    explicit fragment partition is the statement that these fragments do not
    exchange electrons (an ion and its solvent, a redox pair held at fixed
    oxidation states), and that statement is per bond.  It also makes the
    feasible set independent of the geometry: the old constraint carried the
    dissociation envelope ``f_b``, so its projector moved with the positions
    and its pseudo-inverse needed an SVD derivative that is singular at every
    degenerate configuration (it was NaN on a rocksalt cell).  The selector has
    no gradient.

    Connected-component fragments (the default) have no crossing bonds, so for
    them nothing changes.  Self-image bonds (zero incidence column) join an
    atom to itself and never cross.
    """
    n_fragments = int(constraint_matrix.shape[0])
    if n_bonds == 0 or n_fragments <= 1:
        return constraint_matrix.new_zeros((0, n_bonds))
    fragment_of = constraint_matrix.argmax(dim=0)
    crossing = (
        fragment_of.index_select(0, bond_index[0]) != fragment_of.index_select(0, bond_index[1])
    ).nonzero(as_tuple=True)[0]
    selector = constraint_matrix.new_zeros((int(crossing.numel()), n_bonds))
    if crossing.numel():
        selector[torch.arange(int(crossing.numel()), device=selector.device), crossing] = 1.0
    return selector


def assemble_system(
    params: ElectrostaticParams,
    geometry: Geometry,
    constraints: FragmentConstraints,
    options: SolverOptions = SolverOptions(),
    *,
    external_field: Tensor | None = None,
    charge_field: Tensor | None = None,
) -> ElectrostaticSystem:
    """Assemble ``K``, ``c`` and the constraint operator for the quadratic form.

    The blocks follow SPEC 1.5 exactly::

        K = [ B^T J B + diag(kappa)     B^T T01        ]
            [ (B^T T01)^T               A_mu + T_Thole ]

    with ``J_ii = eta_i`` (the on-site hardness *is* the diagonal of the charge
    kernel -- which is why the hardness floor ``eta >= J(0)`` is exactly the
    condition for the charge block to stay positive definite as two atoms
    approach contact) and ``J_ij = K_E erf(r_ij/sigma_ij)/r_ij`` off the
    diagonal.  Gaussian/erf only: the Ohno/Klopman form produced an indefinite
    charge block in audit and is forbidden (SPEC 1.1).

    Sign conventions follow :mod:`trace_electrostatics.kernels`, which returns
    both dipole tensors *in the energy convention*: with ``r_vec = r_i - r_j``,

        U(q_i, mu_j)  = q_i * T01(r_vec) . mu_j       (each ordered pair once)
        U(mu_i, mu_j) = mu_i . T(r_vec) . mu_j        (each unordered pair once)

    so both blocks enter ``K`` with a plus sign and no factor is applied twice.
    The ``0.5`` in ``0.5 m^T K m`` together with the two symmetric off-diagonal
    blocks reproduces exactly one count of each ordered charge-dipole pair and
    one count of each unordered dipole-dipole pair.

    All three pair operators can be replaced by a caller -- see
    :class:`Geometry` -- which is the seam a lattice sum plugs into.  The
    on-site parameters ``eta`` and ``A_mu`` are added on top of whatever comes
    in, and the supplied diagonals (an atom's interaction with its own periodic
    images) survive.
    """
    n_atoms = geometry.n_atoms
    n_bonds = geometry.n_bonds
    if params.n_atoms != n_atoms:
        raise ValueError(
            f"params describe {params.n_atoms} atoms but geometry has {n_atoms}"
        )
    if params.n_bonds != n_bonds:
        raise ValueError(
            f"params describe {params.n_bonds} bonds but geometry has {n_bonds}"
        )
    if constraints.matrix.shape[1] != n_atoms:
        raise ValueError(
            f"constraint matrix has {int(constraints.matrix.shape[1])} columns but the "
            f"system has {n_atoms} atoms"
        )

    positions = geometry.positions
    dtype = positions.dtype
    device = positions.device
    _check_parameter_precision(params, positions)
    incidence = geometry.incidence.to(dtype)
    chi = params.chi.reshape(-1)
    eta = params.eta.reshape(-1)
    sigma = params.sigma.reshape(-1)
    ell = params.ell.reshape(-1)

    eye = torch.eye(n_atoms, dtype=dtype, device=device)
    off_diagonal = 1.0 - eye
    on_site_dipole = _block_diagonal_dipole(params.A_mu.to(dtype), eye)

    if external_field is not None and geometry.bond_dipole is None and (
        geometry.has_pair_overrides or geometry.coulomb_matrix is not None
    ):
        raise ValueError(
            "external_field was supplied together with lattice-summed pair "
            "operators. The overrides are the periodic seam, and under periodic "
            "boundary conditions the position operator is not well defined: a "
            "uniform field acting on free charges is unbounded below, the charges "
            "simply run to the cell edge. That is the electric-enthalpy / "
            "Berry-phase problem, which this package does not implement. Use open "
            "boundaries for finite-field work."
        )

    kernel = options.multipole_kernel
    quadrupolar = params.quadrupole_resolved
    if quadrupolar and kernel != "gaussian":
        raise ValueError(
            "the quadrupole sector requires multipole_kernel='gaussian' (SPEC 12). "
            "With Thole damping the coupled (q, mu, Theta) Hessian is not "
            "positive definite by construction, and it fails in practice: over "
            "121 six-atom clusters at the heads' initial values (alpha 1.5 A^3, "
            "quadrupole polarizability 6 A^5) four were indefinite, min eigenvalue "
            "-0.135 eV, while every diagonal block AND every pairwise sub-block "
            "was positive definite. The Gaussian construction makes K a Coulomb "
            "Gram matrix plus positive on-site residuals, positive definite for "
            "every geometry."
        )
    overridden = geometry.has_pair_overrides or geometry.coulomb_matrix is not None
    # Only the multipole overrides carry a kernel: the charge block J is the
    # erf Coulomb kernel under both, so a coulomb_matrix-only override is
    # compatible with either.
    if geometry.has_pair_overrides and geometry.pair_kernel != kernel:
        raise ValueError(
            f"the pair-operator overrides were built with the {geometry.pair_kernel!r} "
            f"kernel but the solve asks for {kernel!r}. The on-site treatment and "
            "the lattice blocks have to come from the same smeared density, or the "
            "solve minimises a functional neither route reports (SPEC 14)."
        )
    if quadrupolar and overridden and geometry.quadrupole_pair is None:
        raise ValueError(
            "the quadrupole sector is on and the pair operators are overridden, but "
            "the Geometry carries no quadrupole overrides: the Theta blocks would "
            "be built from raw open-boundary positions while q and mu see the "
            "lattice sum. Pass charge_quadrupole, dipole_quadrupole and "
            "quadrupole_pair from trace_electrostatics.lattice (SPEC 14)."
        )

    # Each pair block is either supplied or built here. A supplied one is the
    # complete pair operator, diagonal INCLUDED: under a lattice sum an atom
    # interacts with its own periodic images, and that self term is real. It is
    # added to the on-site parameter, never folded into it -- eta has to keep
    # satisfying the hardness floor (SPEC 1.2), which a negative lattice self
    # term of the same magnitude would destroy.
    needs_pair_geometry = (
        geometry.coulomb_matrix is None
        or geometry.charge_dipole is None
        or geometry.dipole_block is None
    )
    if needs_pair_geometry:
        # d|r|/dr is 0/0 at zero separation, so the square root is taken of a
        # separation that has been SUBSTITUTED wherever it is degenerate --
        # every degenerate entry, not only the i == j diagonal.
        #
        # Shifting the diagonal alone, `sqrt(|r_vec|^2 + delta_ij)`, is what
        # this used to do, on the reasoning that only a site with itself can be
        # at zero separation. Two DISTINCT atoms at coincident positions break
        # that: their entry took sqrt(0), whose derivative is infinite, and
        # although the energy stayed finite the forces did not. Measured
        # through `electrostatic_state` on four atoms with two of them
        # superposed: E = -0.0037534496 (finite) with nan force components on
        # both, spreading to the whole array on the next optimiser step.
        # Masking the output is not enough -- torch.where sends a zero cotangent
        # into the untaken branch and 0 * inf = NaN -- so the substitution is on
        # the INPUT of the square root, exactly as in kernels.combine_sigma.
        #
        # `energy._direct_pair_blocks` carries the same guard on the same
        # threshold for the energy it reports; this is the copy the SOLVE uses,
        # and on the open-boundary route the solver builds these blocks itself
        # (no pair overrides), so both are needed and the two now agree.
        #
        # Where the pair is genuinely separated this is the old expression
        # bit-for-bit: `squared` passes through torch.where unchanged and the
        # old diagonal shift was multiplied by an eye that is zero there.
        r_vec = positions.unsqueeze(1) - positions.unsqueeze(0)
        squared = (r_vec * r_vec).sum(dim=-1)
        separated = squared > DEGENERATE_SEPARATION * DEGENERATE_SEPARATION
        r_safe = torch.sqrt(
            torch.where(separated, squared, torch.full_like(squared, _DIAGONAL_SHIFT))
        )
        # Pairs whose geometry is a substituted placeholder, and the diagonal,
        # in one mask. The dipole tensors' own DEGENERATE_SEPARATION shield
        # cannot do this job: it sees the substituted 1 A, not the zero.
        keep = separated.to(dtype) * off_diagonal
        if kernel == "thole":
            alpha_volume = isotropic_polarizability_volume(params.A_mu, ell)
            lam3, lam5 = thole_damping(
                r_safe, alpha_volume.unsqueeze(1), alpha_volume.unsqueeze(0), options.thole_a
            )
            # One damping for both blocks: Thole damps a PAIR, not a density.
            lam3_charge_dipole = lam3
        else:
            # SPEC 12: every block is a derivative of K_E erf(r/s)/r with s the
            # combined width of the two moments it couples.
            sigma_mu, sigma_theta = gaussian_widths(params, options.gaussian_a)
            lam3_charge_dipole, lam3, lam5 = gaussian_dipole_damping(
                r_safe, sigma, sigma_mu
            )

    # --- charge-charge block -------------------------------------------------
    if geometry.coulomb_matrix is None:
        sigma_ij = combine_sigma(sigma.unsqueeze(1), sigma.unsqueeze(0))
        # The one substituted entry that is a LIMIT rather than a mask: two
        # coincident Gaussian clouds have a perfectly well defined finite
        # interaction, and it is the value coulomb_kernel itself returns at
        # zero. The dipole tensors have no such limit (lambda_3 / r^3 is finite
        # but direction-dependent), so they are zeroed instead.
        coulomb_pair = off_diagonal * torch.where(
            separated, coulomb_kernel(r_safe, sigma_ij), kernel_self(sigma_ij)
        )
    else:
        coulomb_pair = geometry.coulomb_matrix.to(dtype)
    coulomb = coulomb_pair + torch.diag(eta)

    # --- charge-dipole block -------------------------------------------------
    # U_cross = sum_ij q_i t01_ij . mu_j with mu_j = ell_j mu_hat_j, written as
    # q^T G mu_hat with G[i, 3j + b] = ell_j t01[i, j, b]. The ell factor sits on
    # the dipole index only: the block is deliberately not symmetric in (i, j),
    # because it couples two different kinds of moment.
    if geometry.charge_dipole is None:
        t01 = keep.unsqueeze(-1) * charge_dipole_tensor(r_vec, r_safe, lam3_charge_dipole)
        charge_dipole = (ell.reshape(1, n_atoms, 1) * t01).reshape(
            n_atoms, n_atoms * _SPATIAL_DIM
        )
    else:
        charge_dipole = geometry.charge_dipole.to(dtype)

    # --- dipole-dipole block -------------------------------------------------
    if geometry.dipole_block is None:
        relay = dipole_interaction_tensor(r_vec, r_safe, lam3, lam5)
        ell_pair = ell.unsqueeze(1) * ell.unsqueeze(0)
        dipole_pair = (
            keep.unsqueeze(-1).unsqueeze(-1)
            * (ell_pair.unsqueeze(-1).unsqueeze(-1) * relay)
        ).permute(0, 2, 1, 3).reshape(n_atoms * _SPATIAL_DIM, n_atoms * _SPATIAL_DIM)
    else:
        dipole_pair = geometry.dipole_block.to(dtype)
    dipole_block = dipole_pair + on_site_dipole

    # --- quadrupole sector (SPEC 10/12), open boundaries, Gaussian kernel ------
    # Theta_hat is appended AFTER mu_hat, so every consumer that reads
    # K[n_bond_variables:, ...] as "the moments" sees it without a layout change,
    # and every consumer that reads the bond sector is untouched.
    charge_quadrupole = dipole_quadrupole = quadrupole_block = None
    if quadrupolar and geometry.quadrupole_pair is not None:
        # SPEC 14: the lattice-summed Gram blocks, pair part only.
        charge_quadrupole = geometry.charge_quadrupole.to(dtype)
        dipole_quadrupole = geometry.dipole_quadrupole.to(dtype)
        quadrupole_block = geometry.quadrupole_pair.to(dtype) + torch.block_diag(
            *params.A_Theta.to(dtype).unbind(0)
        )
    elif quadrupolar:
        t02, t12, t22 = quadrupole_pair_tensors(
            r_vec, r_safe, keep, sigma, sigma_mu, sigma_theta, ell
        )
        charge_quadrupole = t02.reshape(n_atoms, n_atoms * _QUADRUPOLE_DIM)
        dipole_quadrupole = t12.permute(0, 2, 1, 3).reshape(
            n_atoms * _SPATIAL_DIM, n_atoms * _QUADRUPOLE_DIM
        )
        quadrupole_block = t22.permute(0, 2, 1, 3).reshape(
            n_atoms * _QUADRUPOLE_DIM, n_atoms * _QUADRUPOLE_DIM
        ) + torch.block_diag(*params.A_Theta.to(dtype).unbind(0))

    operator_asymmetry = None
    if geometry.coulomb_matrix is not None or geometry.has_pair_overrides:
        operator_asymmetry = assert_pair_operator_symmetry(
            charge_dipole_is_pair=geometry.charge_dipole_is_pair,
            coulomb=None if geometry.coulomb_matrix is None else coulomb_pair,
            # ell_i ell_j T01_ij: the ell-symmetrised form, whose antisymmetry
            # under i <-> j is a plain index exchange. See the assertion itself.
            charge_dipole=(
                None
                if geometry.charge_dipole is None
                else ell.reshape(n_atoms, 1, 1)
                * charge_dipole.reshape(n_atoms, n_atoms, _SPATIAL_DIM)
            ),
            dipole=None if geometry.dipole_block is None else dipole_pair,
            rtol=options.operator_symmetry_rtol,
        )

    # --- spin sector (SPEC 9): on iff the params carry an exchange ------------
    # The two flags must agree. A spin-resolved parameter set with no spin
    # targets would silently assume every fragment is a singlet, which is a
    # physical assertion the model is not entitled to make (SPEC 9.2: the
    # spin-state splitting is FITTED, not derived); spin targets with
    # closed-shell parameters would be silently ignored.
    spin_resolved = params.spin_resolved
    if spin_resolved and not constraints.spin_resolved:
        raise ValueError(
            "params carry j_exchange (spin-resolved, SPEC 9) but the constraints "
            "carry no spin_targets. The second constraint per fragment, "
            "sum_{i in F} s_i = 2 S_F, has no right-hand side. The total spin of a "
            "fragment is a design input in exactly the way its net charge is -- the "
            "model cannot infer whether a centre is high spin or low spin, and SPEC "
            "9.2 records that the splitting it would produce is a fitted quantity, "
            "not a derived one. Pass spin_targets explicitly, "
            "torch.zeros(n_fragments) for an all-singlet system."
        )
    if constraints.spin_resolved and not spin_resolved:
        raise ValueError(
            "the constraints carry spin_targets but the params carry no j_exchange, "
            "so the spin sector is off and the spin constraints would be silently "
            "ignored. Emit J^x as softplus(raw) + floor (SPEC 9.1) to turn it on."
        )

    # --- reference charge and constraints ------------------------------------
    constraint_matrix = constraints.matrix.to(dtype)
    reference_weights = (
        None
        if constraints.reference_weights is None
        else constraints.reference_weights.to(dtype)
    )
    rho = distribute_fragment_charges(
        constraint_matrix,
        constraints.targets.reshape(-1).to(dtype),
        weights=reference_weights,
    )
    constraint_t = crossing_bond_selector(
        constraint_matrix, geometry.bond_index, int(incidence.shape[1])
    ).to(dtype)

    # --- assemble ------------------------------------------------------------
    k_tt = incidence.transpose(0, 1) @ coulomb @ incidence + torch.diag(params.kappa)
    k_tm = incidence.transpose(0, 1) @ charge_dipole
    if quadrupolar:
        k_tq = incidence.transpose(0, 1) @ charge_quadrupole
    if not spin_resolved and not quadrupolar:
        top = torch.cat([k_tt, k_tm], dim=1)
        bottom = torch.cat([k_tm.transpose(0, 1), dipole_block], dim=1)
        K = torch.cat([top, bottom], dim=0)
    elif not spin_resolved:
        K = torch.cat(
            [
                torch.cat([k_tt, k_tm, k_tq], dim=1),
                torch.cat([k_tm.transpose(0, 1), dipole_block, dipole_quadrupole], dim=1),
                torch.cat(
                    [k_tq.transpose(0, 1), dipole_quadrupole.transpose(0, 1), quadrupole_block],
                    dim=1,
                ),
            ],
            dim=0,
        )
    else:
        # SPEC 9. Variables (t^c, t^s, mu_hat) with t^c = t^up + t^down and
        # t^s = t^up - t^down, so q = B t^c + rho and s = B t^s + rho^s.
        #
        # The two zero blocks below are the physics content of this branch and
        # not an optimisation:
        #
        #   * ``coulomb`` (which carries eta on its diagonal) appears ONCE, in
        #     k_tt, acting on the TOTAL charge. The Coulomb kernel does not
        #     resolve spin -- an up electron and a down electron on different
        #     atoms repel exactly as any two charges do -- so a copy of J in the
        #     t^s block, which is what running the two spin channels separately
        #     through J produces, would be wrong. There is no J below;
        #   * the induced dipole responds to the total density, so the
        #     charge-dipole coupling reaches t^c only and the (t^s, mu_hat)
        #     block is exactly zero.
        #
        # What the spin sector does carry is the on-site exchange
        # 0.5 J^x_i s_i^2 -- a DIAGONAL on-site form, with no off-diagonal
        # partner. That is why J^x > 0 alone makes the block positive definite
        # (with kappa^s > 0), where the charge sector needs the stronger
        # eta >= J(0) to dominate its off-diagonal kernel at contact.
        exchange = params.j_exchange.reshape(-1).to(dtype)
        kappa_spin = params.effective_kappa_spin().to(dtype)
        k_ss = incidence.transpose(0, 1) @ (
            exchange.unsqueeze(1) * incidence
        ) + torch.diag(kappa_spin)
        zero_tt = torch.zeros(n_bonds, n_bonds, dtype=dtype, device=device)
        zero_tm = torch.zeros(
            n_bonds, n_atoms * _SPATIAL_DIM, dtype=dtype, device=device
        )
        if not quadrupolar:
            K = torch.cat(
                [
                    torch.cat([k_tt, zero_tt, k_tm], dim=1),
                    torch.cat([zero_tt, k_ss, zero_tm], dim=1),
                    torch.cat(
                        [k_tm.transpose(0, 1), zero_tm.transpose(0, 1), dipole_block],
                        dim=1,
                    ),
                ],
                dim=0,
            )
        else:
            # The quadrupole, like the dipole, responds to the TOTAL density:
            # its spin-bond block is exactly zero.
            zero_tq = torch.zeros(
                n_bonds, n_atoms * _QUADRUPOLE_DIM, dtype=dtype, device=device
            )
            K = torch.cat(
                [
                    torch.cat([k_tt, zero_tt, k_tm, k_tq], dim=1),
                    torch.cat([zero_tt, k_ss, zero_tm, zero_tq], dim=1),
                    torch.cat(
                        [
                            k_tm.transpose(0, 1),
                            zero_tm.transpose(0, 1),
                            dipole_block,
                            dipole_quadrupole,
                        ],
                        dim=1,
                    ),
                    torch.cat(
                        [
                            k_tq.transpose(0, 1),
                            zero_tq.transpose(0, 1),
                            dipole_quadrupole.transpose(0, 1),
                            quadrupole_block,
                        ],
                        dim=1,
                    ),
                ],
                dim=0,
            )

    # q = B t + rho, so rho contributes a linear term in both sectors and a
    # constant. The constant depends on geometry through J and must be kept:
    # dropping it would silently break the forces.
    # A uniform applied field E couples as -P.E with P = sum_i (q_i r_i + mu_i).
    # The charge half is exactly a shift of the electronegativity, chi_i -> chi_i
    # - E.r_i, so it flows into c_t and u_const through the existing path.  The
    # dipole half is a new linear term, -ell_i E, on the reduced moment.
    # Both must enter HERE, in the term the solver minimises: adding -P.E only
    # to the reported energy leaves the moments unpolarised, U linear in E, and
    # d2U/dE2 identically zero.
    if external_field is not None:
        field = external_field.reshape(-1)
        if field.numel() != _SPATIAL_DIM:
            raise ValueError(
                "external_field must be a uniform field with 3 components, got shape "
                f"{tuple(external_field.shape)}. A per-atom field is not what -P.E "
                "describes: P = sum_i (q_i r_i + mu_i) is one vector for the whole "
                "system."
            )
        field = field.to(dtype)
        if geometry.bond_dipole is None:
            chi = chi - positions @ field
        field_mu = -(ell.reshape(n_atoms, 1) * field).reshape(-1)
    else:
        field_mu = None

    c_t = incidence.transpose(0, 1) @ (coulomb @ rho + chi)
    if params.bond_bias is not None:
        # SPEC 20: the external circuit's work on the electrode bonds.
        c_t = c_t + params.bond_bias.reshape(-1).to(c_t.dtype)
    field_constant = None
    if charge_field is not None and geometry.bond_dipole is None:
        raise ValueError(
            "charge_field separates the charge-sector field from the dipole one "
            "and acts through the bond-current coupling (SPEC 15); this Geometry "
            "carries no bond_dipole. Under open boundaries shift chi instead."
        )
    if charge_field is not None and external_field is None:
        raise ValueError("charge_field needs external_field (the dipole-sector field).")
    if external_field is not None and geometry.bond_dipole is not None:
        # SPEC 15, the bond-current electric enthalpy:
        #     -E . Omega P,  Omega P = sum_i rho_i r_i + sum_b f_b tau_b d_b + sum_i mu_i
        # The tau part is a linear term on the bond variables, single valued
        # because d_b is the image-resolved bond vector; the rho part is a
        # constant in m (rho is fixed), kept because it carries the forces
        # rho_i E on charged fragments.
        charge_sector_field = (
            field if charge_field is None else charge_field.reshape(-1).to(dtype)
        )
        c_t = c_t - geometry.bond_dipole.to(dtype) @ charge_sector_field
        field_constant = -(rho @ positions) @ charge_sector_field
    c_mu = charge_dipole.transpose(0, 1) @ rho
    if field_mu is not None:
        c_mu = c_mu + field_mu
    # A uniform field has no gradient, so it does not couple to Theta; the
    # quadrupole sees rho through T02 and its own permanent source chi_Theta.
    c_theta = None
    if quadrupolar:
        c_theta = charge_quadrupole.transpose(0, 1) @ rho
        if params.chi_theta is not None:
            c_theta = c_theta + params.chi_theta.to(dtype).reshape(-1)
    if not spin_resolved:
        c = torch.cat([c_t, c_mu] if c_theta is None else [c_t, c_mu, c_theta], dim=0)
        u_const = 0.5 * (rho * (coulomb @ rho)).sum() + (chi * rho).sum()
        if field_constant is not None:
            u_const = u_const + field_constant
        rho_spin = None
        exchange = None
    else:
        # The spin half of q = B t^c + rho, s = B t^s + rho^s. rho^s carries the
        # 2 S_F of the second constraint and is FIXED, exactly as rho is: a free
        # per-atom spin channel would absorb everything and drive t^s -> 0, the
        # third instance of the SPEC 1.3 general principle.
        rho_spin = distribute_fragment_spins(
            constraint_matrix,
            constraints.spin_targets.reshape(-1).to(dtype),
            weights=reference_weights,
        )
        # chi_spin = (chi^up - chi^down)/2 is the ONLY source of spin
        # polarisation in this model: J^x > 0 penalises it, so there is no
        # Stoner instability available (SPEC 9.1). Absent, it is zero, and the
        # spin sector then does nothing but satisfy its constraint.
        chi_spin = (
            torch.zeros(n_atoms, dtype=dtype, device=device)
            if params.chi_spin is None
            else params.chi_spin.reshape(-1).to(dtype)
        )
        # NOTE the applied field does NOT appear here. A uniform electric field
        # couples as -P.E with P = sum_i (q_i r_i + mu_i), which is built from
        # the TOTAL charge; it shifts chi and leaves chi_spin alone. A field
        # term in the spin sector would be a Zeeman coupling, which is a
        # different field and is not implemented.
        c_s = incidence.transpose(0, 1) @ (exchange * rho_spin + chi_spin)
        c = torch.cat(
            [c_t, c_s, c_mu] if c_theta is None else [c_t, c_s, c_mu, c_theta], dim=0
        )
        u_const = (
            0.5 * (rho * (coulomb @ rho)).sum()
            + (chi * rho).sum()
            + 0.5 * (rho_spin * exchange * rho_spin).sum()
            + (chi_spin * rho_spin).sum()
        )
        if field_constant is not None:
            u_const = u_const + field_constant
        constraint_bond = constraint_t
        constraint_t = spin_constraint_block(constraint_t)

    return ElectrostaticSystem(
        K=K,
        c=c,
        u_const=u_const,
        rho=rho,
        constraint_t=constraint_t,
        constraint_bond=None if not spin_resolved else constraint_bond,
        coulomb=coulomb,
        charge_dipole=charge_dipole,
        dipole_block=dipole_block,
        operator_asymmetry=operator_asymmetry,
        spin_resolved=spin_resolved,
        rho_spin=rho_spin,
        exchange=exchange,
        quadrupole_resolved=quadrupolar,
        charge_quadrupole=charge_quadrupole,
        dipole_quadrupole=dipole_quadrupole,
        quadrupole_block=quadrupole_block,
    )


def _block_diagonal_dipole(A_mu: Tensor, eye: Tensor) -> Tensor:
    """``A_mu`` as a ``[3n, 3n]`` block-diagonal, in the ``(site, cartesian)`` layout."""
    n_atoms = int(eye.shape[0])
    return (
        (eye.unsqueeze(-1).unsqueeze(-1) * A_mu.unsqueeze(1))
        .permute(0, 2, 1, 3)
        .reshape(n_atoms * _SPATIAL_DIM, n_atoms * _SPATIAL_DIM)
    )


def _check_parameter_precision(params: ElectrostaticParams, positions: Tensor) -> None:
    """Refuse parameters that do not live where the positions do.

    Mixing dtypes does not fail: torch promotes, so float32 parameters with
    float64 positions give a float64 ``K`` that carries float32 accuracy and
    looks entirely healthy -- measured 4.67e-07 off the all-float64 answer,
    which is six orders of magnitude past the 1e-13 the gradient regressions
    assert and would be read as a broken implementation rather than a broken
    dtype.  SPEC 3 requires float64 throughout; this says so where it is
    checkable.

    ``incidence`` and the constraint matrix are exempt: they hold only 0 and
    +-1, which every float dtype represents exactly, and they are cast
    explicitly a few lines below.
    """
    reference_dtype = positions.dtype
    reference_device = positions.device
    for name in (
        "chi",
        "eta",
        "sigma",
        "ell",
        "kappa",
        "A_mu",
        # SPEC 9, absent on a closed-shell parameter set. A float32 J^x is
        # exactly as damaging as a float32 eta and for the same reason.
        "j_exchange",
        "chi_spin",
        "kappa_spin",
        # SPEC 10/12, absent unless the quadrupole sector is on.
        "A_Theta",
        "chi_theta",
    ):
        value = getattr(params, name)
        if value is None:
            continue
        if value.dtype != reference_dtype:
            raise ValueError(
                f"params.{name} is {value.dtype} but positions are {reference_dtype}. "
                "Mixing precisions silently promotes to the wider dtype while keeping "
                "the narrower accuracy (measured 4.67e-07 error on K), so it reads as "
                "a wrong implementation rather than a wrong dtype. SPEC 3 requires "
                "float64 for the whole pipeline."
            )
        if value.device != reference_device:
            raise ValueError(
                f"params.{name} is on {value.device} but positions are on "
                f"{reference_device}."
            )


def _check_seed_binding(seed: GeometrySeed, geometry: Geometry) -> None:
    """Re-check a bound seed against the configuration actually being solved.

    With a fixed, truncated iteration count the seed is part of the *definition*
    of the energy, so a seed computed at one configuration and used at another
    silently redefines it: the same configuration reached from two trajectories
    gets two different energies and the force stops being the gradient of
    anything (SPEC 2).  In MD every step moves the atoms, so a seed carried over
    from the previous step is stamped with the previous step's positions and is
    caught here.

    Bit-exact comparison, because "approximately the same geometry" is precisely
    the thing that must not be accepted -- a warm start from a nearby
    configuration is still a warm start.
    """
    stamp = seed.geometry_fingerprint
    if stamp is None:
        # An UNBOUND seed used to pass silently, which made the whole barrier
        # opt-in: constructing GeometrySeed(...) directly, without going through
        # for_geometry(), bypassed the check entirely.  Measured: a seed carrying
        # a previous solution's moments was accepted.
        #
        # Only a provably geometry-INDEPENDENT seed may go unbound, and the one
        # such seed is m_0 = 0.  Verify that by VALUE, not by trusting
        # provenance == "zero" -- a string can lie, an all-zero tensor cannot.
        # t_spin is screened here too: a spin seed carried over from the
        # previous step redefines the energy exactly as a charge seed does, and
        # leaving it out of this test would have made the barrier bypassable by
        # putting the history in the spin channel.
        spin_nonzero = seed.t_spin is not None and bool(seed.t_spin.any())
        if bool(seed.t.any()) or bool(seed.mu_hat.any()) or spin_nonzero:
            raise ValueError(
                "the GeometrySeed carries non-zero moments but is not bound to a "
                "geometry, so it cannot be checked against the configuration being "
                "solved. With a fixed, truncated iteration count the seed is part of "
                "the DEFINITION of the energy (SPEC 2), so an unverifiable seed is "
                "indistinguishable from a warm start. Build it with "
                "GeometrySeed.for_geometry(...) or from_bond_logits(..., "
                "geometry=...) so it carries a fingerprint, or pass "
                "GeometrySeed.zeros(...) if you genuinely want the trivial seed."
            )
        return
    current = geometry.fingerprint()
    if stamp.shape != current.shape or not torch.equal(
        stamp, current.to(stamp.dtype).to(stamp.device)
    ):
        raise ValueError(
            "the GeometrySeed was computed at a different configuration from the one "
            "being solved: the seed is bound to positions that do not match "
            "geometry.positions. That is a warm start. With a fixed, truncated "
            "iteration count the seed is part of the DEFINITION of the energy, so "
            "carrying one over from the previous MD step makes U history-dependent "
            "and forfeits conservative forces -- which is the entire reason for the "
            "fixed-iteration design (SPEC 2). Recompute the seed from the current "
            "positions."
        )


def _relax_dipole_seed(
    system,
    m: Tensor,
    n_bonds: int,
    offset: int,
    n_atoms: int,
    tiny: float,
    on_site: Tensor | None = None,
) -> Tensor:
    """One block-Jacobi step on the dipoles from the seed, with exact line search.

    A learned 1o dipole seed recovered nothing (its error equalled the whole
    dipole magnitude, 0.41-0.45 of |m*|), while 20-38% of the k = 6 energy
    error sat in the dipole sector. The physics does it without parameters: at
    the seed charges the dipole gradient is ``T01^T q_0 + (A_mu + T2) mu``, and
    a block-Jacobi step with the on-site 3x3 blocks of K, followed by an EXACT
    line search along it, cut the prototype's dipole error from 0.41-0.45 to
    0.14-0.24. Measured in this package, on top of the trained seed, on held-out
    27-216 atom clusters with TRACE-trunk features: k = 6 relative energy error
    1.7-2.6x lower (e.g. 7.7e-5 -> 3.0e-5 at 125 atoms). With a zero charge seed
    on a neutral fragment it does nothing, correctly: the dipole gradient is then
    exactly zero.

    Guarantees: the direction has no bond component, so feasibility is
    untouched; the preconditioner is positive definite, so the direction is a
    descent direction; the exact line search makes U(m_0') <= U(m_0) always.
    Static shapes, no host sync; a zero gradient gives a zero step through safe
    inputs rather than 0/0.

    ``on_site`` (SPEC 14.1): the preconditioner, ``A_mu``.  Under PBC the
    diagonal 3x3 blocks of ``K`` also hold each atom's interaction with its own
    periodic images, which depends on the SIMULATION CELL -- a primitive cell
    and its supercell have different ones -- so preconditioning with them made
    the seed, and the fixed-k energy, cell dependent.  ``A_mu`` is the atom's
    own and is the same in every cell; under open boundaries those diagonal
    blocks ARE ``A_mu`` bit for bit, so nothing changes there.  The exact line
    search below still uses the full ``K``.  ``None`` keeps the diagonal-block
    extraction, for callers that do not have the parameters.
    """
    dtype, device = m.dtype, m.device
    # Only the blocks this step touches, so the arithmetic is IDENTICAL with and
    # without spin channels. The dipoles couple to the CHARGE bonds (T01) and to
    # each other, and not at all to the spin bonds -- K[dipole, spin] is exactly
    # zero (SPEC 9: the induced dipole responds to the total density). Forming
    # the gradient over the full vector instead summed extra zeros in a
    # different grouping and broke the closed-shell limit's bitwise identity by
    # one ulp. It is also cheaper: no full K @ v product.
    # With the quadrupole sector on, Theta_hat sits after mu_hat (SPEC 12); the
    # slices stop there and the mu-Theta coupling is added on its own, so the
    # arithmetic without that sector is exactly what it always was.
    stop = offset + _SPATIAL_DIM * n_atoms
    dipole_rows = system.K[offset:stop]
    dipole_block = dipole_rows[:, offset:stop]
    dipole_gradient = (
        dipole_rows[:, :n_bonds] @ m[:n_bonds]
        + dipole_block @ m[offset:stop]
        + system.c[offset:stop]
    )
    tail = int(m.numel()) - stop
    if tail:
        dipole_gradient = dipole_gradient + dipole_rows[:, stop:] @ m[stop:]
    if on_site is None:
        on_site = torch.diagonal(
            dipole_block.reshape(n_atoms, 3, n_atoms, 3), dim1=0, dim2=2
        ).permute(2, 0, 1)
    dipole_step = -torch.linalg.solve(
        on_site, dipole_gradient.reshape(n_atoms, 3, 1)
    ).reshape(-1)
    slope = dipole_gradient @ dipole_step
    curvature = dipole_step @ (dipole_block @ dipole_step)
    positive = curvature > tiny
    unit = torch.ones_like(curvature)
    step = torch.where(
        positive, -slope / torch.where(positive, curvature, unit), torch.zeros_like(unit)
    )
    pieces = [torch.zeros(offset, dtype=dtype, device=device), step * dipole_step]
    if tail:
        pieces.append(torch.zeros(tail, dtype=dtype, device=device))
    return m + torch.cat(pieces)


def _relax_quadrupole_seed(
    system, m: Tensor, offset: int, n_atoms: int, tiny: float, on_site: Tensor
) -> Tensor:
    """One block-Jacobi step on ``Theta_hat`` with exact line search (SPEC 12).

    The quadrupole twin of :func:`_relax_dipole_seed`, run after it.  Its main
    job is the PERMANENT quadrupole: with ``chi_Theta`` on, the isolated-atom
    answer ``Theta_hat = -A_Theta^-1 (chi_Theta + T02^T q + T12^T mu)`` is one
    5x5 solve per atom away, and the fixed-k CG should not spend its iterations
    finding it.  Same guarantees: the direction touches only the quadrupole
    block, which the constraints do not act on; the 5x5 on-site blocks are
    principal submatrices of a positive definite K, so it is a descent direction;
    the exact line search gives ``U(m_0') <= U(m_0)``.

    Preconditioned with ``A_Theta`` itself, not the diagonal blocks of K, for the
    reason :func:`_relax_dipole_seed` gives: under PBC those blocks carry the
    cell-dependent self-image sum (measured 3.8e-7 eV per cell between a
    2-atom cell and its 2x2x2 supercell after one CG step).
    """
    dtype, device = m.dtype, m.device
    rows = system.K[offset:]
    gradient = rows @ m + system.c[offset:]
    block = rows[:, offset:]
    step_direction = -torch.linalg.solve(
        on_site, gradient.reshape(n_atoms, _QUADRUPOLE_DIM, 1)
    ).reshape(-1)
    slope = gradient @ step_direction
    curvature = step_direction @ (block @ step_direction)
    positive = curvature > tiny
    unit = torch.ones_like(curvature)
    step = torch.where(
        positive, -slope / torch.where(positive, curvature, unit), torch.zeros_like(unit)
    )
    return m + torch.cat(
        [torch.zeros(offset, dtype=dtype, device=device), step * step_direction]
    )


def _fixed_k_cg(
    system: ElectrostaticSystem, m: Tensor, project, n_iter: int
) -> tuple:
    """The legacy SPEC 2 energy: ``n_iter`` projected CG steps, differentiated through.

    Returns ``(m_k, ||r_0||, ||r_k||)``.  Kept bit for bit as it was; selected
    by ``SolverOptions(solve_mode="fixed_k")``.  See SPEC 17 for why it is no
    longer the default: its response to a perturbation the zero-field residual
    does not excite is arbitrary, and it depends on the gauge ``ell``.
    """
    dtype = system.K.dtype
    residual = project(-system.gradient(m))
    direction = residual
    residual_sq = residual @ residual

    # Denominator guard: the smallest normal number of the working dtype.
    #
    # It is NOT true that a positive kappa makes a vanishing denominator
    # impossible outside exact convergence. assert_positive_kappa is strict
    # positivity in exact arithmetic and nothing more: it accepts the smallest
    # subnormal, kappa = 5e-324, and on a 6-atom ring that parameterisation
    # measures min eig(K_tt) = -2.397e-15 -- numerically indefinite, so
    # d^T K d can reach zero, or cross it, with the residual still large. The
    # SPEC 1.4 kappa -> 0 limit has to stay reachable, so the guard is what
    # keeps that regime finite rather than a floor on kappa. No branch, no sync.
    tiny = torch.finfo(dtype).tiny

    # sqrt(0) has an infinite derivative, so a zero RHS -- a system already at
    # its solution, or with no degrees of freedom -- puts inf/NaN into the
    # gradient of anything that reads this telemetry. The energy path never
    # touches it, but a loss term regularising on the residual would. clamp_min
    # passes zero gradient below the bound and is exact above it, at the cost of
    # reporting 1.49e-154 rather than 0.
    initial_residual_norm = torch.sqrt(residual_sq.clamp_min(tiny))

    # Converged-iteration freeze. Once the residual is below machine precision
    # RELATIVE to the starting residual, a further CG step moves m by less than
    # one ulp, so it contributes nothing to the energy -- but it still sits in
    # the autograd graph, and its divisions have denominators of order the
    # squared residual. At k ~ 250 on a 6-atom chain the residual reaches
    # 5e-111, its square 3e-221, and the SECOND derivative of `a / b` carries
    # 1/b^2 = 1/1e-442, which underflows to 0 and turns into 0/0. Measured: the
    # fixed-k polarizability (a second derivative of this energy) was exact
    # through k = 150 (0.50827461, equal to the minimiser route) and NaN at
    # k = 250 and 400, while the energy and forces stayed finite. The SPEC 2.0
    # diagnostic of re-running at large k is exactly what exposed it.
    #
    # Frozen iterations take step 0 and beta 0, with SAFE inputs substituted in
    # the untaken branch (torch.where sends a zero cotangent there, and 0 * inf
    # is NaN, so the guard has to be on the input, not the output -- the same
    # rule as coulomb_kernel). While active, every expression below is the one
    # it replaced, so any run that never reaches the floor -- the default k = 6
    # always -- is bitwise unchanged. No branch, no host sync, static shapes.
    freeze_floor = (torch.finfo(dtype).eps ** 2) * residual_sq.detach()
    unit = torch.ones_like(residual_sq)
    for _ in range(n_iter):
        active = residual_sq > freeze_floor
        safe_residual_sq = torch.where(active, residual_sq, unit)
        curvature_vector = system.K @ direction
        curvature = direction @ curvature_vector
        safe_curvature = torch.where(active, curvature + tiny, unit)
        step = torch.where(
            active, safe_residual_sq / safe_curvature, torch.zeros_like(unit)
        )
        m = m + step * direction
        residual = project(residual - step * curvature_vector)
        residual_sq_new = residual @ residual
        beta = torch.where(
            active,
            residual_sq_new / (safe_residual_sq + tiny),
            torch.zeros_like(unit),
        )
        direction = residual + beta * direction
        residual_sq = residual_sq_new

    residual_norm = torch.sqrt(residual_sq.clamp_min(tiny))
    return m, initial_residual_norm, residual_norm



def _woodbury_bond_inverse(incidence: Tensor, kappa: Tensor, onsite: Tensor):
    """``r -> (diag(kappa) + B^T C B)^-1 r`` by the Woodbury identity, detached.

    ``C`` is ``[n, n]`` (a dense charge kernel) or ``[n]`` (a diagonal):

        (kappa + B^T C B)^-1 = kappa^-1 - kappa^-1 B^T (C^-1 + B kappa^-1 B^T)^-1 B kappa^-1

    One ``[n, n]`` Cholesky per solve.  ``B kappa^-1 B^T`` is a weighted bond-
    graph Laplacian: this is the structure that makes the bond sector
    ill-conditioned and that no diagonal preconditioner sees (block-Jacobi was
    measured SLOWER than none: 109 against 67 iterations on 216-atom rocksalt).
    """
    B = incidence.detach()
    if B.is_sparse:
        B = B.to_dense()
    k = kappa.detach()
    c_inverse = None
    if onsite.dim() == 2:
        # Only C on range(B) enters M = kappa + B^T C B, and 1^T B = 0 (every
        # column is +f, -f, or zero), so C + s 11^T gives the SAME M for any s.
        # C itself need not be definite along the uniform charge direction (a
        # point-charge kernel, a lattice sum with k = 0 removed); the shift
        # makes it invertible without touching the preconditioner.
        C = onsite.detach()
        C = 0.5 * (C + C.transpose(0, 1))
        n = int(C.shape[0])
        shift = float(torch.diagonal(C).abs().max()) + 1.0
        for trial in (C, C + (shift / n) * torch.ones_like(C)):
            factor_c, info = torch.linalg.cholesky_ex(trial)
            if int(info) == 0:
                c_inverse = torch.cholesky_inverse(factor_c)
                break
        if c_inverse is None:
            # Not definite on the transfer subspace either: fall back to the
            # on-site diagonal, still an SPD preconditioner.
            onsite = torch.diagonal(C).clamp_min(torch.finfo(C.dtype).tiny)
    if c_inverse is None:
        c_inverse = torch.diag(1.0 / onsite.detach().reshape(-1))
    atom_space = c_inverse + (B / k) @ B.transpose(0, 1)
    factor, info = torch.linalg.cholesky_ex(0.5 * (atom_space + atom_space.transpose(0, 1)))
    if int(info) != 0:
        return lambda r: r / k

    def apply(r: Tensor) -> Tensor:
        scaled = r / k
        correction = torch.cholesky_solve((B @ scaled).unsqueeze(-1), factor).squeeze(-1)
        return scaled - (B.transpose(0, 1) @ correction) / k

    return apply


def _block_jacobi_preconditioner(
    system: ElectrostaticSystem, options: SolverOptions, params: ElectrostaticParams,
    incidence: Tensor,
):
    """``r -> M^-1 r`` for the converged solve (SPEC 17), detached.

    Bond sector: the EXACT inverse of ``kappa + B^T (eta + J) B`` (and of the
    spin twin ``kappa_s + B^T J^x B``) by the Woodbury identity -- one atom-space
    Cholesky.  Moments: each atom's own 3x3 / 5x5 diagonal block of ``K``.
    Measured on rocksalt supercells to solve_rtol = 1e-10: 22 / 25 iterations
    at 64 / 216 atoms, against 53 / 67 unpreconditioned and 85 / 109 with the
    bond diagonal.  Symmetric positive definite whenever ``K`` is, so it
    changes the iteration count only, never the answer.
    """
    K = system.K.detach()
    n_atoms = system.n_atoms
    n_bonds = system.n_bonds
    width = system.n_bond_variables
    dip0, quad0 = system.dipole_offset, system.quadrupole_offset
    atoms = torch.arange(n_atoms, device=K.device)
    kappa = params.kappa.to(K.dtype)
    if n_bonds:
        charge_inverse = _woodbury_bond_inverse(incidence.to(K.dtype), kappa, system.coulomb)
        spin_inverse = (
            _woodbury_bond_inverse(
                incidence.to(K.dtype),
                params.effective_kappa_spin().to(K.dtype),
                system.exchange.reshape(-1),
            )
            if system.spin_resolved
            else None
        )

    def blocks(start: int, size: int) -> Tensor:
        index = start + size * atoms.unsqueeze(-1) + torch.arange(size, device=K.device)
        return K[index.unsqueeze(-1), index.unsqueeze(-2)]

    dipole_inverse = torch.linalg.inv(blocks(dip0, _SPATIAL_DIM)) if n_atoms else None
    quadrupole_inverse = (
        torch.linalg.inv(blocks(quad0, _QUADRUPOLE_DIM))
        if system.quadrupole_resolved and n_atoms
        else None
    )

    def apply(vector: Tensor) -> Tensor:
        pieces = []
        if n_bonds:
            pieces.append(charge_inverse(vector[:n_bonds]))
            if spin_inverse is not None:
                pieces.append(spin_inverse(vector[n_bonds:width]))
        if n_atoms:
            dip = vector[dip0:quad0].reshape(n_atoms, _SPATIAL_DIM, 1)
            pieces.append((dipole_inverse @ dip).reshape(-1))
        if quadrupole_inverse is not None:
            quad = vector[quad0:].reshape(n_atoms, _QUADRUPOLE_DIM, 1)
            pieces.append((quadrupole_inverse @ quad).reshape(-1))
        return torch.cat(pieces)

    return apply


def feasibility_mask(system: ElectrostaticSystem) -> Tensor:
    """``[n_m]`` 0/1: zero on every bond variable pinned by SPEC 17.2."""
    mask = torch.ones(int(system.K.shape[0]), dtype=system.K.dtype, device=system.K.device)
    selector = system.constraint_t
    if int(selector.shape[0]):
        pinned = selector.detach().abs().sum(dim=0) > 0
        mask[: int(selector.shape[1])] = torch.where(
            pinned, torch.zeros_like(mask[: int(selector.shape[1])]), mask[: int(selector.shape[1])]
        )
    return mask


def _solve_converged(
    system: ElectrostaticSystem,
    x0: Tensor,
    options: SolverOptions,
    params: ElectrostaticParams,
    n_bonds: int,
    spin_resolved: bool,
    quadrupolar: bool,
    n_atoms: int,
    incidence: Tensor,
) -> tuple:
    """SPEC 17: ``x* = argmin_V U`` to ``solve_rtol``, with exact derivatives.

    Returns ``(x*, U(x*), ||P c||, ||P(K x* + c)||, iterations)``; ``U``
    includes ``u_const``.  ``x*`` and ``U`` carry the implicit derivatives of
    :mod:`trace_electrostatics.implicit`, so forces, responses and every higher
    derivative are those of the minimiser.
    """
    problem = QuadraticProblem(
        lambda tensors: (lambda vector: tensors[0] @ vector),
        (system.K,),
        mask=feasibility_mask(system),
        precondition=_block_jacobi_preconditioner(system, options, params, incidence),
        rtol=options.solve_rtol,
        max_iter=options.solve_max_iter,
        name="solve_charges",
    )
    x, quadratic = minimise(problem, system.c, x0)
    return (
        x,
        quadratic + system.u_const,
        problem.rhs_norm,
        problem.residual_norm,
        int(problem.iterations),
    )


def solve_charges(
    params: ElectrostaticParams,
    geometry: Geometry,
    constraints: FragmentConstraints,
    n_iter: int = DEFAULT_CG_ITERATIONS,
    *,
    seed: GeometrySeed,
    options: SolverOptions = SolverOptions(),
    external_field: Tensor | None = None,
    charge_field: Tensor | None = None,
) -> SolveResult:
    """Run ``n_iter`` projected conjugate-gradient steps and return the state.

    The returned energy is *defined* as ``U(m_k)`` after exactly ``n_iter``
    iterations from ``seed``; it is not an approximation to the exact minimum
    that happens to be truncated.  Because every operation below is ordinary
    differentiable torch -- no ``no_grad``, no ``detach``, no custom
    ``autograd.Function`` -- ``autograd`` traverses the iterations and
    ``dU/dR`` is the exact gradient of that definition, including the
    ``dm_k/dR`` term.  That term is what the Hellmann-Feynman shortcut drops,
    and dropping it cost a factor of 12000 in energy drift (SPEC 2).

    The iteration is *projected* CG: the search directions are confined to the
    null space of the per-fragment constraints, and the starting point is made
    feasible once, so every iterate satisfies ``C q = Q`` exactly -- including
    the truncated one.  Feasibility is therefore independent of convergence.

    Spin (SPEC 9) is opt-in and is requested by ``params.j_exchange`` together
    with ``constraints.spin_targets``; supplying one without the other raises.
    With it on, the variable vector is ``(t^c, t^s, mu_hat)``, the projector
    becomes ``diag(P, P, I)`` with the same bond projector on both channels,
    and **both** constraints hold exactly at every iterate --
    ``sum_F q_i = Q_F`` and ``sum_F s_i = 2 S_F`` -- for the same reason the
    charge one does when spin is off.  With it off, every line below is the one
    it was before spin existed.

    Parameters
    ----------
    params:
        Head outputs; see :class:`ElectrostaticParams`.
    geometry:
        Positions and bond topology; see :class:`Geometry`.
    constraints:
        Per-fragment net charges; see :class:`FragmentConstraints`.
    n_iter:
        Fixed number of CG iterations, default 6.  A Python ``int``, so the
        loop unrolls to a static graph for AOT compilation.
    seed:
        Keyword-only :class:`GeometrySeed`, or a pure
        ``f(geometry, params) -> GeometrySeed`` which is evaluated here at the
        configuration being solved.  Must be geometry-determined; see that class
        for why a warm start from the previous MD step is forbidden and for the
        three things that enforce it.
    options:
        See :class:`SolverOptions`.
    external_field:
        Optional uniform field ``[3]`` in V/A, coupling as ``-P . E``.  Open
        boundaries only -- it is rejected together with the periodic pair
        operators, for the same reason it is rejected together with a cell
        upstream: under periodic boundary conditions the position operator is
        not well defined and a uniform field on free charges is unbounded below.

    Returns
    -------
    SolveResult
    """
    if not isinstance(n_iter, int) or isinstance(n_iter, bool) or n_iter < 1:
        raise ValueError(f"n_iter must be a positive int, got {n_iter!r}")
    if isinstance(seed, (SolveResult, SolveDiagnostics)):
        raise TypeError(
            "the solver seed may not be a previous SolveResult. The seed must be "
            "geometry-determined: seeding from the previous MD step makes U "
            "history-dependent and forfeits conservative forces, which is the entire "
            "reason for the fixed-iteration design (SPEC 2)."
        )
    if not isinstance(seed, GeometrySeed) and callable(seed):
        # A pure f(geometry, params) -> GeometrySeed, evaluated HERE, at the
        # configuration actually being solved. The result is then a function of
        # the current geometry by construction rather than by assertion, which
        # is the strongest form of the SPEC 2 seed requirement.
        seed = seed(geometry, params)
    if not isinstance(seed, GeometrySeed):
        raise TypeError(
            "seed must be a GeometrySeed (or a callable f(geometry, params) returning "
            "one), not a bare tensor, so that the geometry provenance of the starting "
            f"point is explicit; got {type(seed).__name__}"
        )

    n_atoms = geometry.n_atoms
    n_bonds = geometry.n_bonds
    if seed.t.numel() != n_bonds or seed.mu_hat.shape[0] != n_atoms:
        raise ValueError(
            f"seed has {seed.t.numel()} bonds and {int(seed.mu_hat.shape[0])} atoms, "
            f"geometry has {n_bonds} bonds and {n_atoms} atoms"
        )
    _check_seed_binding(seed, geometry)

    system = assemble_system(
        params,
        geometry,
        constraints,
        options,
        external_field=external_field,
        charge_field=charge_field,
    )
    dtype = system.K.dtype

    # --- constraint projector -------------------------------------------------
    # Constraints act only on the bond sector: C q = C B t + C rho = Q, and rho
    # is built to satisfy C rho = Q exactly, so the constraint on t is the
    # homogeneous A t = 0 with A = C B. A is always rank deficient (its rows sum
    # to 1^T B = 0), hence a pseudo-inverse rather than an inverse.
    #
    # P = I - A^+ A, from pinv(A) directly and NOT from pinv(A A^T): forming the
    # Gram matrix squares the condition number, and the projector then stops
    # projecting. Measured at cond(A) = 1e7, |A P| is 1.43e-03 through the Gram
    # route against 4.76e-11 through pinv(A) -- eight orders of magnitude, on a
    # conditioning that an explicit multi-fragment labelling reaches easily.
    #
    # A is structural: C is a 0/1 membership matrix and B is the signed
    # incidence, neither carries a gradient, so the SVD inside pinv is never
    # differentiated. The only short-circuits taken are the ones visible in the
    # shapes -- no bonds, or no constraints. Whether A is numerically zero is a
    # question about tensor VALUES, and asking it would cost the host
    # synchronisation this module goes to some length to avoid; it is also no
    # longer the common case, since the constraint operator reports the true
    # C B block and an explicit fragment_id makes it genuinely non-zero.
    #
    # SPEC 9 adds the second constraint sum_F s_i = 2 S_F.  Because q and s are
    # built from the SAME incidence matrix, and rho^s is built to satisfy the
    # spin constraint exactly the way rho satisfies the charge one, the combined
    # constraint on the bond variables (t^c, t^s) is the BLOCK DIAGONAL
    # diag(A, A) with the same A = C B -- not a single wide row block [A, A],
    # which would pin only t^c + t^s = 2 t^up and leave the down channel free.
    # The projector is therefore diag(P, P), and one factorisation of A serves
    # both halves: pinv of a block diagonal is the block diagonal of the pinvs,
    # so this is exact, and it costs one SVD of A rather than one of a matrix
    # twice as wide in each dimension.
    spin_resolved = system.spin_resolved
    if seed.t_spin is not None and not spin_resolved:
        raise ValueError(
            "the seed carries t_spin but this solve is closed shell (params have no "
            "j_exchange), so the spin seed would be silently dropped and the energy "
            "would not be the one the seed describes. Either pass spin-resolved "
            "params (SPEC 9) or drop t_spin from the seed."
        )
    constraint_t = system.bond_constraint
    trivial = n_bonds == 0 or int(constraint_t.shape[0]) == 0

    if trivial:
        def project(vector: Tensor) -> Tensor:
            """No constraint acts on the bond sector; the projector is the identity."""
            return vector
    elif not spin_resolved:
        constraint_pinv = torch.linalg.pinv(constraint_t)

        def project(vector: Tensor) -> Tensor:
            """Orthogonal projection onto the feasible subspace of ``m``."""
            bond_part = vector[:n_bonds]
            rest = vector[n_bonds:]
            correction = constraint_pinv @ (constraint_t @ bond_part)
            return torch.cat([bond_part - correction, rest])
    else:
        constraint_pinv = torch.linalg.pinv(constraint_t)

        def project(vector: Tensor) -> Tensor:
            """``diag(P, P, I)``: the same bond projector on both spin channels."""
            charge_part = vector[:n_bonds]
            spin_part = vector[n_bonds : 2 * n_bonds]
            rest = vector[2 * n_bonds :]
            charge_correction = constraint_pinv @ (constraint_t @ charge_part)
            spin_correction = constraint_pinv @ (constraint_t @ spin_part)
            return torch.cat(
                [
                    charge_part - charge_correction,
                    spin_part - spin_correction,
                    rest,
                ]
            )

    # --- starting point -------------------------------------------------------
    # The seed's t is the CHARGE channel t^c = t^up + t^down in both cases, so a
    # single seed head serves the closed-shell and the spin-resolved solve and
    # an unpolarised spin solve starts exactly where the closed-shell one does.
    #
    # SPEC 17: under the converged solve the seed and the two physics steps are
    # only a starting point -- they change the iteration count, never the
    # answer -- so they are built without a graph.
    converged = options.solve_mode == SOLVE_CONVERGED
    seed_context = torch.no_grad() if converged else contextlib.nullcontext()
    seed_context.__enter__()
    quadrupolar = system.quadrupole_resolved
    theta_seed = (
        [torch.zeros(n_atoms * _QUADRUPOLE_DIM, dtype=dtype, device=system.K.device)]
        if quadrupolar
        else []
    )
    if not spin_resolved:
        m = project(
            torch.cat(
                [seed.t.reshape(-1).to(dtype), seed.mu_hat.reshape(-1).to(dtype)]
                + theta_seed
            )
        )
    else:
        seed_spin = (
            torch.zeros(n_bonds, dtype=dtype, device=system.K.device)
            if seed.t_spin is None
            else seed.t_spin.reshape(-1).to(dtype)
        )
        m = project(
            torch.cat(
                [
                    seed.t.reshape(-1).to(dtype),
                    seed_spin,
                    seed.mu_hat.reshape(-1).to(dtype),
                ]
                + theta_seed
            )
        )
    if options.relax_dipole_seed and n_atoms > 0:
        m = _relax_dipole_seed(
            system,
            m,
            n_bonds,
            2 * n_bonds if spin_resolved else n_bonds,
            n_atoms,
            torch.finfo(dtype).tiny,
            on_site=params.A_mu.to(dtype),
        )
    if quadrupolar and options.relax_quadrupole_seed and n_atoms > 0:
        m = _relax_quadrupole_seed(
            system,
            m,
            system.quadrupole_offset,
            n_atoms,
            torch.finfo(dtype).tiny,
            params.A_Theta.to(dtype),
        )
    seed_context.__exit__(None, None, None)

    if converged:
        m, energy, initial_residual_norm, residual_norm, n_iter = _solve_converged(
            system, m.detach(), options, params, n_bonds, spin_resolved, quadrupolar, n_atoms,
            geometry.incidence,
        )
    else:
        m, initial_residual_norm, residual_norm = _fixed_k_cg(system, m, project, n_iter)
        energy = system.energy(m)

    # --- unpack ---------------------------------------------------------------
    incidence_f = geometry.incidence.to(dtype)
    t = m[:n_bonds]
    dipole_start, dipole_stop = system.dipole_offset, system.quadrupole_offset
    if not spin_resolved:
        t_spin = None
        spin_population = None
    else:
        t_spin = m[n_bonds : 2 * n_bonds]
        spin_population = incidence_f @ t_spin + system.rho_spin
    mu_hat = m[dipole_start:dipole_stop].reshape(n_atoms, _SPATIAL_DIM)
    mu = params.ell.reshape(n_atoms, 1) * mu_hat
    theta_hat = theta = None
    if quadrupolar:
        # Theta = e ell^2 sum_m Theta_hat_m W_m: the traceless second moment of
        # kernels.py (Buckingham's Theta is 3/2 of this one).
        theta_hat = m[dipole_stop:].reshape(n_atoms, _QUADRUPOLE_DIM)
        basis = quadrupole_cartesian_basis().to(dtype=dtype, device=m.device)
        theta = (params.ell.reshape(n_atoms, 1) ** 2 * theta_hat) @ basis.reshape(
            _QUADRUPOLE_DIM, _SPATIAL_DIM * _SPATIAL_DIM
        )
        theta = theta.reshape(n_atoms, _SPATIAL_DIM, _SPATIAL_DIM)
    q = incidence_f @ t + system.rho

    # energy = quadratic + linear + reference. bond_hardness is a COMPONENT of
    # quadratic, not a fourth term; the name says so because a flat dict of four
    # keys invites a four-way sum that double counts it.
    energy_terms = {
        "quadratic": 0.5 * (m * (system.K @ m)).sum(),
        "linear": (system.c * m).sum(),
        "reference": system.u_const,
        "bond_hardness_in_quadratic": 0.5 * (params.kappa * t * t).sum(),
    }
    if spin_resolved:
        # Both are COMPONENTS of "quadratic", like bond_hardness_in_quadratic,
        # and the names say so for the same reason.
        energy_terms["spin_bond_hardness_in_quadratic"] = 0.5 * (
            params.effective_kappa_spin().to(dtype) * t_spin * t_spin
        ).sum()
        energy_terms["on_site_exchange_in_quadratic"] = 0.5 * (
            system.exchange * spin_population * spin_population
        ).sum()

    # Cheap and always on: a violated charge constraint invalidates every
    # downstream observable.
    constraint_violation = assert_fragment_charges(
        q,
        constraints.matrix.to(dtype),
        constraints.targets.reshape(-1).to(dtype),
        tol=options.fragment_charge_tol,
    )
    spin_constraint_violation = None
    fractional_spin = None
    if spin_resolved:
        spin_constraint_violation = _assert_fragment_spins(
            spin_population,
            constraints.matrix.to(dtype),
            constraints.spin_rhs.to(dtype),
            tol=options.fragment_charge_tol,
        )
        # Telemetry, not an assertion: the flat-plane violation of 0.5 J^x s^2 is
        # present at every parameterisation and cannot be asserted away (SPEC
        # 9.2). The frozen-bond form is O(n_atoms), allocates two small matrices
        # and reads nothing on the host, which is what makes it affordable on
        # every solve; it is a rigorous upper bound on the relaxed curvature, so
        # a user acting on it is never acting on an understatement.
        if options.measure_fractional_spin_error is not False:
            fractional_spin = fractional_spin_curvature(
                params.j_exchange,
                _spin_reference_map(constraints, dtype, system.K.device),
                chi_spin=params.chi_spin,
                # `incidence` is passed even though `relax_bonds=False` never
                # USES it, because it is the only way the callee can tell "this
                # system has no bonds" from "this system has bonds that were not
                # relaxed". Without it `n_bonds` reads 0 and the returned
                # FractionalSpinError claims `bonds_relaxed=True`, i.e. "this is
                # the model's exact value", when it is actually the cheap frozen
                # upper bound -- measured 40% high on a 6-atom ring with
                # non-uniform J^x (0.2778 frozen against 0.1982 relaxed). The
                # `summary()` caveat that exists to say so was being suppressed
                # on every spin solve of every system that has any bonds.
                incidence=geometry.incidence.to(dtype=dtype, device=system.K.device),
                relax_bonds=False,
            )

    diagnostics = SolveDiagnostics(
        n_iter=n_iter,
        seed_provenance=seed.provenance,
        initial_residual_norm=initial_residual_norm,
        residual_norm=residual_norm,
        constraint_violation=constraint_violation,
        energy_terms=energy_terms,
        system=system,
        operator_asymmetry=system.operator_asymmetry,
        spin_constraint_violation=spin_constraint_violation,
        fractional_spin=fractional_spin,
    )

    validate = validation_enabled() if options.validate is None else bool(options.validate)
    check_minimum = validate if options.check_is_minimum is None else bool(options.check_is_minimum)
    if validate:
        # system.constraint_t spans the whole bond sector -- diag(C B, C B) with
        # spin on -- which is the space charge_block() lives in. The projected
        # Schur complement then carries the spin mode, whose eigenvalue is the
        # one that softens as J^x is driven down.
        diagnostics.curvature = assert_positive_definite_at_solution(
            system.charge_block(),
            system.coupling_block(),
            # (mu_hat) or (mu_hat, Theta_hat): coupling_block() spans the same
            # rows, and the Theta modes must be inside the Schur complement.
            system.multipole_block(),
            system.constraint_t,
            tol=options.eigenvalue_tol,
            enabled=True,
        )
    if check_minimum:
        # SPEC 2.1 item 4, over the FULL coupled Hessian: block-wise positivity
        # proves nothing (audit: +0.58, +16.45, coupled -1.05).
        #
        # system.constraint_t, not the single C B block the projector uses: with
        # spin on it is diag(C B, C B) and spans the whole bond sector, so the
        # spin mode is inside the projection rather than outside it. Padding the
        # narrow block instead would leave t^s unconstrained in this check and a
        # soft spin mode would be attributed to a direction the physics has
        # already removed.
        bond_constraint = system.constraint_t
        full_constraint = torch.cat(
            [
                bond_constraint,
                torch.zeros(
                    bond_constraint.shape[0],
                    int(system.K.shape[1]) - int(bond_constraint.shape[1]),
                    dtype=dtype,
                    device=bond_constraint.device,
                ),
            ],
            dim=1,
        )
        diagnostics.minimum = assert_is_minimum(
            system.K,
            system.gradient(m) if options.expect_converged else None,
            full_constraint,
            eig_tol=options.eigenvalue_tol,
            grad_tol=options.gradient_tol,
            enabled=True,
        )

    # Stamp the moments so a later GeometrySeed built out of them is recognised
    # as a warm start and refused. `m` is stamped too because `t` and `mu_hat`
    # are views of it, and a reshape of a view re-roots on `m`. See GeometrySeed.
    _mark_solver_output(m, t, mu_hat, mu, q)
    if quadrupolar:
        _mark_solver_output(theta_hat, theta)
    if spin_resolved:
        # The spin moments get the same stamp. A warm start hidden in the spin
        # channel redefines U exactly as one in the charge channel does.
        _mark_solver_output(t_spin, spin_population)

    return SolveResult(
        t=t,
        rho=system.rho,
        mu=mu,
        q=q,
        energy=energy,
        diagnostics=diagnostics,
        t_spin=t_spin,
        rho_spin=system.rho_spin,
        s=spin_population,
        theta_hat=theta_hat,
        theta=theta,
    )


def _assert_fragment_spins(
    spin_population: Tensor,
    constraint_matrix: Tensor,
    spin_rhs: Tensor,
    tol: float,
) -> Tensor:
    """Assert ``|sum_{i in F} s_i - 2 S_F| < tol`` for every fragment (SPEC 9).

    The spin twin of
    :func:`~trace_electrostatics.diagnostics.assert_fragment_charges`, kept
    separate rather than folded into it because the two failures mean different
    things and must be distinguishable in the message: a solve can satisfy every
    net charge exactly while putting the spin on the wrong fragment, and being
    told "charge constraint violated" would send the reader to the wrong half of
    the model.  ``spin_rhs`` is ``2 S_F``, already doubled.

    NaN counts as a violation, for the same reason it does in the charge check:
    the predicate is negated rather than written as ``worst > tol``.
    """
    residual = constraint_matrix @ spin_population.reshape(-1) - spin_rhs.reshape(-1)
    if residual.numel() == 0:
        return torch.zeros(
            (), dtype=spin_population.dtype, device=spin_population.device
        )
    worst = residual.abs().max()
    if _needs_host_report(
        ~(worst <= tol),
        "per-fragment spin constraint violated beyond tol: sum_{i in F} s_i = 2 S_F "
        "(SPEC 9)",
    ):
        index = int(torch.argmax(residual.abs()))
        actual = float((constraint_matrix @ spin_population.reshape(-1))[index])
        target = float(spin_rhs.reshape(-1)[index])
        raise ElectrostaticsAssertionError(
            f"per-fragment spin constraint violated: fragment {index} carries "
            f"s = {actual:.12f} (i.e. S = {0.5 * actual:.12f}) but was constrained "
            f"to 2 S_F = {target:.12f} (S_F = {0.5 * target:.12f}); violation "
            f"{float(worst):.3e} > {tol:g}. Note the factor of two: s_i = "
            "q_i^up - q_i^down counts electrons, so the constrained quantity is "
            "2 S_F, the number of unpaired electrons, not S_F."
        )
    return worst


# ---------------------------------------------------------------------------
# The fractional-spin (flat-plane) error -- SPEC 9.2
# ---------------------------------------------------------------------------


def _spin_reference_map(
    constraints: FragmentConstraints,
    dtype: torch.dtype,
    device: Optional[torch.device],
) -> Tensor:
    """``R = d rho^s / d m``, ``[n_atoms, n_fragments]``.

    The Jacobian of
    :func:`~trace_electrostatics.topology.distribute_fragment_spins` with respect
    to the per-fragment unpaired-electron count ``m_F = 2 S_F``: that function is
    linear in its targets, so ``rho^s = R m`` exactly and ``R`` is the whole of
    its content.  Written out here rather than obtained by calling it once per
    fragment because this runs as per-solve telemetry and a loop over fragments
    would cost ``n_fragments`` kernel launches for a single matrix.

    The weights and their normalisation must match
    :func:`~trace_electrostatics.topology.distribute_fragment_charges` exactly,
    since that is what actually builds ``rho^s``; the SPEC 9 factor of two is
    *not* applied here, because this differentiates with respect to ``2 S_F``
    and not with respect to ``S_F``.

    No validation: a negative weight or a fragment with zero total weight is
    rejected by ``distribute_fragment_charges`` itself, which
    :func:`assemble_system` calls on the same weights on every solve, and
    repeating the check here would cost a host synchronisation for nothing.
    """
    matrix = constraints.matrix.to(dtype=dtype, device=device)
    n_atoms = int(matrix.shape[1])
    if constraints.reference_weights is None:
        weights = torch.ones(n_atoms, dtype=dtype, device=device)
    else:
        weights = constraints.reference_weights.reshape(-1).to(dtype=dtype, device=device)
    weighted = matrix * weights.unsqueeze(0)
    return (weighted / weighted.sum(dim=1, keepdim=True)).transpose(0, 1)


def fractional_spin_error(
    params: ElectrostaticParams,
    geometry: Geometry,
    constraints: FragmentConstraints,
    *,
    relax_bonds: bool = True,
) -> FractionalSpinError:
    """The fractional-spin (flat-plane) error of this parameterisation, in eV.

    **This measures a known defect, it does not check for one.**  The exact
    energy of a many-electron system is piecewise planar in ``(N, M_s)``
    (Mori-Sanchez, Cohen and Yang, PRL 102, 066403, 2009): at fixed electron
    number it is piecewise *linear* in the spin polarisation, and for a
    degenerate open shell it is exactly *flat*.  The on-site term
    ``0.5 J^x_i s_i^2`` of SPEC 9 is quadratic in ``s``, so it has curvature
    along ``M_s`` at every parameterisation -- including a perfectly fitted one.
    What this function returns is how much.

    The number is curvature per **unpaired electron squared**,
    ``d^2 E / d m_F d m_G`` with ``m_F = 2 S_F = N_up - N_down``; see
    :class:`~trace_electrostatics.diagnostics.FractionalSpinError`, which also
    carries :meth:`~trace_electrostatics.diagnostics.FractionalSpinError.splitting`
    -- the bound that says what fraction of a predicted spin-state splitting the
    artifact supplied.

    Why the charge sector does not appear
    -------------------------------------
    "Curvature along ``M_s`` at fixed ``N``" is exact here rather than
    approximate.  SPEC 9's ``K`` has structural zeros between the charge and spin
    bond sectors -- the Coulomb kernel couples TOTAL charges, so ``J`` appears
    once and not once per channel, and the induced dipole responds to the total
    density -- so the spin curvature does not involve ``J``, ``eta``, ``kappa``,
    the positions or the dipoles at all.  Only ``J^x``, ``kappa^s``, the bond
    graph and the fragment partition enter.  A consequence worth stating: the
    artifact is a property of the *parameters*, so it can be read off a model
    without running molecular dynamics on it.

    **The splitting this model produces is a FITTED quantity, not a derived
    one.**  It inherits the fractional-spin error of the functional form, and
    that is the same form AIMNet2-NSE and MACE-POLAR-1 use -- a limitation of
    the class, not of this implementation.

    Parameters
    ----------
    params, geometry, constraints:
        As for :func:`solve_charges`.  ``params`` must carry ``j_exchange``
        (SPEC 9); a closed-shell parameter set has no spin sector and raises.
    relax_bonds:
        ``True`` (default) relaxes the spin-channel bond transfer ``t^s`` and
        returns the model's own curvature, at the cost of a null-space basis and
        a solve in bond space.  ``False`` freezes ``t^s`` at zero and returns a
        rigorous ``O(n_atoms)`` upper bound -- that is the form the solver runs
        as telemetry.

    Returns
    -------
    FractionalSpinError
    """
    if not params.spin_resolved:
        raise ValueError(
            "these params are closed shell (j_exchange is None), so there is no "
            "spin sector and no fractional-spin error to measure. The error being "
            "measured is the curvature the on-site term 0.5 J^x s^2 imposes along "
            "M_s (SPEC 9.2); with no J^x that term is not in the energy at all."
        )
    if params.n_atoms != geometry.n_atoms:
        raise ValueError(
            f"params describe {params.n_atoms} atoms but geometry has "
            f"{geometry.n_atoms}"
        )
    if int(constraints.matrix.shape[1]) != geometry.n_atoms:
        raise ValueError(
            f"constraint matrix has {int(constraints.matrix.shape[1])} columns but the "
            f"system has {geometry.n_atoms} atoms"
        )
    dtype = params.j_exchange.dtype
    device = params.j_exchange.device
    reference = _spin_reference_map(constraints, dtype, device)
    return fractional_spin_curvature(
        params.j_exchange,
        reference,
        chi_spin=params.chi_spin,
        incidence=geometry.incidence.to(dtype=dtype, device=device),
        kappa_spin=params.effective_kappa_spin(),
        # C B: the constraint as it acts on the bond variables. Both spin
        # channels carry the same one (SPEC 9), so the single block is right.
        bond_constraint=constraints.matrix.to(dtype=dtype, device=device)
        @ geometry.incidence.to(dtype=dtype, device=device),
        relax_bonds=relax_bonds,
    )


def degenerate_doublet_reference(
    j_exchange: float = 2.0,
    *,
    n_atoms: int = 1,
    n_iter: int = DEFAULT_CG_ITERATIONS,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Tensor]:
    """The calibration case: a one-electron system, where the exact E is FLAT.

    A diagnostic that only reports a relative number is hard to act on, so this
    is the absolute anchor.  A one-electron system has a **degenerate doublet**:
    the up and down states have the same energy, and so does every fractional
    spin polarisation between them.  The exact curvature along ``M_s`` is
    therefore **provably zero**, with no reference calculation and no fitting
    involved -- it is a degeneracy, not an approximation.

    The model cannot reproduce it.  With the spin localised on one atom the
    curvature comes out at exactly ``J^x``, so

        E(m) - E(0)  =  0.5 J^x m^2 ,

    which at ``J^x = 2 eV`` and ``m = 1`` is **1.0 eV of splitting where the
    exact answer is 0** -- the number SPEC 9.2 quotes.  Real Fe(II)
    high-spin/low-spin gaps are 0.1-0.4 eV, so that artifact is 100-400% of the
    signal it would be used to predict.

    Everything is measured through the ordinary code path -- :func:`solve_charges`
    at two spin states and :func:`fractional_spin_error` -- rather than
    hand-derived, so this doubles as an end-to-end test of the spin sector
    against an analytic answer.  Measured at the defaults: curvature 2.0 eV/e^2
    against an exact ``J^x`` of 2.0 (error **exactly 0.0**), a splitting of
    **1.0 eV** between two states the exact theory makes degenerate (error
    against the closed-form prediction **exactly 0.0**), and an artifact fraction
    of **1.0** -- every last electronvolt of it is the functional form.

    Parameters
    ----------
    j_exchange:
        The on-site exchange to calibrate at, eV/e^2.  2.0 reproduces SPEC 9.2.
    n_atoms:
        How many atoms the one electron's spin is spread over, in one fragment
        joined by a chain of bonds.  The default 1 is the sharp case.  Larger
        values show the other half of the story: with a uniform ``J^x`` and
        uniform reference weights the curvature is exactly ``J^x / n_atoms``
        *independent of the bonds, the geometry and* ``kappa^s`` -- the spin
        relaxation drive vanishes identically because ``1^T B = 0``.  A localised
        spin carries the full artifact; a delocalised one divides it.
    n_iter:
        CG iterations for the two solves.  The curvature itself is a property of
        the energy functional and does not depend on it.

    Returns
    -------
    dict
        ``curvature`` and ``curvature_exact`` (``J^x / n_atoms``),
        ``curvature_error``; ``splitting`` (measured, ``E(m=1) - E(m=0)``, from
        two real solves), ``splitting_exact`` (**0**, the degeneracy),
        ``splitting_predicted`` (from the curvature alone) and
        ``splitting_error``; ``artifact_fraction``, which is **1.0**: all of it
        is artifact; and the SPEC 9.1 check --
        ``spin_mode_eigenvalue`` (``2 J^x``) with the three residuals
        :func:`~trace_electrostatics.diagnostics.verify_spin_mode_eigenvalue`
        measures.
    """
    dtype = torch.get_default_dtype() if dtype is None else dtype
    if n_atoms < 1:
        raise ValueError(f"n_atoms must be >= 1, got {n_atoms}")

    # A chain of n_atoms atoms in one fragment. The positions and the Coulomb
    # parameters below are ordinary and deliberately uninteresting: the spin
    # curvature does not depend on them (SPEC 9's structural zeros), and having
    # them in the system is what makes this an end-to-end check rather than an
    # evaluation of a formula.
    positions = torch.zeros(n_atoms, _SPATIAL_DIM, dtype=dtype, device=device)
    positions[:, 0] = torch.arange(n_atoms, dtype=dtype, device=device) * 2.0
    if n_atoms == 1:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
    else:
        chain = torch.arange(n_atoms - 1, device=device)
        edge_index = torch.stack(
            [torch.cat([chain, chain + 1]), torch.cat([chain + 1, chain])]
        )
    incidence, bond_index = build_incidence(
        edge_index, n_atoms, dtype=dtype, allow_no_bonds=True
    )
    incidence = incidence.to(device=device)
    n_bonds = int(incidence.shape[1])

    sigma = torch.full((n_atoms,), 1.5, dtype=dtype, device=device)
    params = ElectrostaticParams(
        # chi = 0 and chi_spin = 0: up and down are equivalent for a
        # one-electron system with no applied field, so the exact energy is even
        # in m and the model's linear term must be zero for the comparison to be
        # the one SPEC 9.2 makes. Every eV of splitting is then curvature.
        chi=torch.zeros(n_atoms, dtype=dtype, device=device),
        eta=kernel_self(sigma) + 1.0,
        sigma=sigma,
        ell=torch.ones(n_atoms, dtype=dtype, device=device),
        kappa=torch.full((n_bonds,), 2.0, dtype=dtype, device=device),
        A_mu=torch.eye(_SPATIAL_DIM, dtype=dtype, device=device).expand(
            n_atoms, _SPATIAL_DIM, _SPATIAL_DIM
        ).contiguous()
        * 12.0,
        j_exchange=torch.full((n_atoms,), float(j_exchange), dtype=dtype, device=device),
        chi_spin=torch.zeros(n_atoms, dtype=dtype, device=device),
        kappa_spin=torch.full((n_bonds,), 1.0, dtype=dtype, device=device),
    )
    geometry = Geometry(
        positions=positions, incidence=incidence, bond_index=bond_index.to(device=device)
    )
    fragment_id = torch.zeros(n_atoms, dtype=torch.long, device=device)
    net_charge = torch.zeros(1, dtype=dtype, device=device)

    def solve_at(total_spin: float) -> Tensor:
        constraints = FragmentConstraints.from_fragment_ids(
            fragment_id,
            net_charge,
            dtype=dtype,
            spin_targets=torch.full((1,), total_spin, dtype=dtype, device=device),
        )
        seed = GeometrySeed.zeros(n_bonds, n_atoms, geometry=geometry)
        return solve_charges(params, geometry, constraints, n_iter, seed=seed).energy

    # S = 0 (the unpolarised state, m = 0) and S = 1/2 (the doublet, m = 1).
    # These are the two states whose exact energies are equal.
    energy_unpolarised = solve_at(0.0)
    energy_doublet = solve_at(0.5)
    measured_splitting = energy_doublet - energy_unpolarised

    doublet_constraints = FragmentConstraints.from_fragment_ids(
        fragment_id,
        net_charge,
        dtype=dtype,
        spin_targets=torch.full((1,), 0.5, dtype=dtype, device=device),
    )
    error = fractional_spin_error(params, geometry, doublet_constraints)
    splitting = error.splitting(0.0, 1.0)

    # The other half of the calibration. The curvature above is only meaningful
    # if the spin sector is a minimum rather than a saddle, and that rests
    # entirely on SPEC 9.1's claim that the spin mode of the on-site Hessian has
    # eigenvalue 2 J^x. Check it here rather than cite it: this is the one
    # function in the package whose job is to establish that the spin sector
    # behaves the way the derivation says, so it is the right place for the
    # derivation to be re-derived numerically on every run.
    eigenvalues = verify_spin_mode_eigenvalue(params.eta, params.j_exchange)
    exact = torch.full((), float(j_exchange) / n_atoms, dtype=dtype, device=device)
    zero = torch.zeros((), dtype=dtype, device=device)
    return {
        "curvature": error.worst_curvature,
        "curvature_exact": exact,
        "curvature_error": (error.worst_curvature - exact).abs(),
        "splitting": measured_splitting,
        # The flat-plane condition, evaluated on the one case where it is a
        # degeneracy rather than an approximation.
        "splitting_exact": zero,
        "splitting_predicted": splitting.splitting,
        "splitting_error": (splitting.splitting - measured_splitting).abs(),
        "artifact": splitting.artifact,
        "artifact_fraction": splitting.artifact_fraction,
        "nonlinearity": splitting.nonlinearity,
        # SPEC 9.1, re-derived numerically rather than cited.
        "spin_mode_eigenvalue": 2.0 * params.j_exchange[0],
        "spin_mode_eigenvalue_error": eigenvalues["eigenvalue_error"],
        "on_site_hessian_error": eigenvalues["hessian_error"],
        "spin_mode_eigenvector_error": eigenvalues["eigenvector_error"],
    }
