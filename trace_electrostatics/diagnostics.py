"""Runtime assertions for the TRACE electrostatics solver.

Every check in this module exists because a *measured* failure got past the
checks that were already there (SPEC 1.2, 2.1).  They are written as raising
functions rather than bare ``assert`` statements so that ``python -O`` cannot
silently delete the safety net.

Cost model
----------
The checks split cleanly into two classes:

* **Cheap, always on** -- elementwise comparisons whose cost is O(n): the
  hardness floor, ``kappa`` symmetry and positivity, and the per-fragment
  charge residual.  These run on every solve.
* **Expensive, opt-in** -- eigenvalue decompositions and dense solves, O(n^3):
  the projected Schur complement and the minimum test.  These are gated on a
  validation flag which is OFF by default (production) and turned ON for
  validation runs, regression tests and any new parameterisation.

Use :func:`set_validation_mode` or the :func:`validation_mode` context manager
to switch the expensive checks on globally; individual calls accept an explicit
``enabled=`` override.

Host synchronisation
--------------------
The cheap checks are reached from the solver on *every* solve, so how they read
their own result matters.  ``bool(tensor)`` copies a device scalar to the host:
on a GPU that serialises the pipeline, and under ``torch.compile`` it is a hard
graph break (measured: ``torch._dynamo.explain`` reports
``call_function BuiltinVariable(bool) [TensorVariable()]`` and
``fullgraph=True`` fails outright).  Every cheap check therefore goes through
:func:`_needs_host_report`, which

* under tracing (``torch.compile``/dynamo) issues :func:`torch._assert_async`
  and returns ``False`` -- no sync and no graph break, measured as 0 breaks and
  a working ``fullgraph=True`` where both previously failed;
* in eager mode reads the flag on the host and lets the caller build the full
  diagnostic message, which is where the value of these assertions lies.

Callers that must not sync even in eager mode (a GPU MD loop) pass
``sync=False`` explicitly and get the async form.

**Known limitation, measured on torch 2.2.0.**  ``torch._assert_async`` fires
under the ``eager`` and ``aot_eager`` backends but is eliminated by inductor's
dead-code pass, because it has no users: compiling the hot path with the default
backend therefore keeps the graph and loses the assertion.  That is a property
of torch, not of this module -- ``energy.py``'s own ``_assert_async`` calls
behave identically -- and it is the trade a fully compiled hot path currently
comes with.  Validate a new parameterisation in eager, where the checks are
real, before compiling it.

Exception type
--------------
All failures raise :class:`ElectrostaticsAssertionError`, which derives from
both :class:`AssertionError` and :class:`ValueError` so that callers (and
tests) can catch whichever they consider natural for an assertion that is also
a rejected input.

NaN
---
Every violation predicate is written as ``~(ok)`` rather than ``(not ok)``, so
that a NaN parameter is a *violation* rather than a silent pass.  ``NaN < x`` and
``NaN > x`` are both ``False``, so the natural spelling
``violated = margin < -tol`` let a NaN ``eta`` through the tripwire for SPEC's
single most important constraint without a word.

A third class of check: measuring a known error
-----------------------------------------------
:func:`fractional_spin_curvature` and what hangs off it are not assertions.  The
on-site exchange term ``0.5 J^x_i s_i^2`` violates the flat-plane condition of
Mori-Sanchez, Cohen and Yang (PRL 102, 066403, 2009) *by construction*, and that
cannot be asserted away.  The exact energy of a many-electron system is
**piecewise planar** in ``(N, M_s)``; curvature along ``N`` is the
delocalisation error and curvature along ``M_s`` is the fractional-spin (static
correlation) error.  A term quadratic in ``s`` has non-zero curvature along
``M_s`` everywhere, so this model has a fractional-spin error at every
parameterisation, including the correct one.

Do **not** try to remove it by imposing piecewise linearity per atom.  That was
tried on the charge axis earlier in this project and it is a category error:
piecewise linearity (PPLB) is a condition on a whole system or on a separated
fragment, never on an atom-in-a-molecule.  Imposed per atom it pins ordinary
partial charges to zero -- measured, with a properly damped kernel, C-H and O-H
both going to exactly 0.0000 at bonding distance where the physical values are
+0.1 and +0.35.  Atomic charges are not observables and no exact piecewise
linearity condition applies to them; the same objection applies verbatim to the
spin axis, where ``s_i`` is a population analysis and not an observable either.

The fractional-spin error is an open problem in density functional theory
itself.  The goal of these functions is therefore not to fix it but to make it
**visible, quantified and bounded**, so that a user reading a spin-state
splitting out of this model knows how much of it is artifact.
"""

from __future__ import annotations

import importlib
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence, Union

import torch
from torch import Tensor

from .kernels import gram_self_energy

__all__ = [
    "ElectrostaticsAssertionError",
    "set_validation_mode",
    "validation_enabled",
    "validation_mode",
    "assert_hardness_floor",
    "assert_positive_kappa",
    "assert_symmetric_kappa",
    "assert_pair_operator_symmetry",
    "schur_complement",
    "assert_positive_definite_at_solution",
    "assert_is_minimum",
    "assert_fragment_charges",
    "null_space_basis",
    # The fractional-spin (flat-plane) error, SPEC 9.2
    "DEFAULT_ARTIFACT_FRACTION_LIMIT",
    "FractionalSpinError",
    "SpinStateSplitting",
    "fractional_spin_curvature",
    "assert_spin_splitting_is_prediction",
    "verify_spin_mode_eigenvalue",
]


class ElectrostaticsAssertionError(AssertionError, ValueError):
    """Raised when a physics invariant of the electrostatics model is violated.

    Inherits from both :class:`AssertionError` (it is an assertion about the
    model) and :class:`ValueError` (it is also a rejected input), so callers
    need not guess which one to catch.
    """


# Default tolerance for "is this eigenvalue positive" questions, in eV/e^2.
# Chosen well above float64 round-off on a typical Hessian norm (~10 eV/e^2)
# and well below any physically meaningful curvature.
DEFAULT_EIGENVALUE_TOL: float = 1e-8

# Default tolerance for the per-fragment charge residual (SPEC 2.1 item 5).
DEFAULT_CHARGE_TOL: float = 1e-10

# Relative singular-value cutoff used when extracting a constraint null space.
DEFAULT_RANK_RTOL: float = 1e-10

# Default tolerance for the symmetry of a supplied pair operator, relative to
# the largest entry of that operator.
DEFAULT_OPERATOR_SYMMETRY_RTOL: float = 1e-10

# Whether the expensive checks are on.  The process-wide value is the default
# every thread starts from; ``_validation_state.enabled``, when set, overrides
# it for the current thread only.  Two threads validating different models (a
# validation run beside a production inference worker) must not toggle each
# other's checks, and a :func:`validation_mode` block must not leak out of the
# thread that opened it -- including when it exits by exception.
_VALIDATION_DEFAULT: bool = False
_validation_state = threading.local()


def set_validation_mode(enabled: bool) -> None:
    """Enable/disable the O(n^3) validation checks.

    Production MD wants this OFF: an eigendecomposition per step would dominate
    the cost of the model.  Validation runs, regression tests and every newly
    trained parameterisation want it ON, because the failures it catches
    (saddle points returned as minima, negative polarizability) are silent.

    Sets both the process-wide default (so threads started later inherit it) and
    the calling thread's own setting.  It is a deliberate global switch and it
    stays set until changed: use :func:`validation_mode` where a scoped,
    thread-local, exception-safe change is wanted instead.
    """
    global _VALIDATION_DEFAULT
    _VALIDATION_DEFAULT = bool(enabled)
    _validation_state.enabled = bool(enabled)


def validation_enabled() -> bool:
    """Whether the expensive validation checks are currently enabled here."""
    return bool(getattr(_validation_state, "enabled", _VALIDATION_DEFAULT))


@contextmanager
def validation_mode(enabled: bool = True) -> Iterator[None]:
    """Scoped, thread-local form of :func:`set_validation_mode`.

    Restores the previous setting of *this thread* on exit, including on an
    exception, and never touches any other thread.
    """
    sentinel = object()
    previous = getattr(_validation_state, "enabled", sentinel)
    _validation_state.enabled = bool(enabled)
    try:
        yield
    finally:
        if previous is sentinel:
            del _validation_state.enabled
        else:
            _validation_state.enabled = previous  # type: ignore[assignment]


def _resolve(enabled: Optional[bool]) -> bool:
    return validation_enabled() if enabled is None else bool(enabled)


def _never() -> bool:
    return False


def _resolve_tracing_probe():
    """Bind ``is_compiling`` **once, at import**, to a plain function object.

    It has to be resolved here and not at the call site: a ``getattr`` or
    ``hasattr`` on a module *inside* a traced function is itself a graph break
    (``Unsupported: hasattr PythonModuleVariable is_compiling``), so a
    lazily-resolved probe would break exactly the graph it exists to keep whole.
    ``torch._dynamo`` is imported for the same reason -- the probe must already
    be the real one the first time a compiled function runs.

    The attribute moved between torch releases (2.2 has
    ``torch._dynamo.is_compiling`` but no ``torch.compiler.is_compiling``), and
    a missing one degrades to "never tracing", never to an import error.
    """
    probe = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if probe is not None:
        return probe
    try:
        dynamo = importlib.import_module("torch._dynamo")
    except Exception:  # pragma: no cover - torch built without dynamo
        return _never
    return getattr(dynamo, "is_compiling", None) or _never


_IS_TRACING = _resolve_tracing_probe()


def _is_tracing() -> bool:
    """Whether we are inside ``torch.compile``/dynamo tracing."""
    return bool(_IS_TRACING())


def _needs_host_report(
    violated: Tensor, message: str, *, sync: Optional[bool] = None
) -> bool:
    """Decide how to report a violation flag without gratuitously syncing.

    ``violated`` is a 0-d boolean tensor.  Returns ``True`` only when the caller
    should read it on the host and raise a fully detailed
    :class:`ElectrostaticsAssertionError`; otherwise the check has already been
    queued device-side with :func:`torch._assert_async` and the caller must
    simply carry on.

    ``sync=None`` (the default) means "sync in eager, do not sync while
    tracing", which keeps the rich messages everywhere they are readable and
    keeps the compiled graph unbroken.  ``sync=False`` forces the async form,
    for a GPU hot loop that must not stall even in eager mode; ``sync=True``
    forces the host read.

    See the module docstring for the inductor DCE limitation that applies to the
    async form on torch 2.2.
    """
    if sync is None:
        sync = not _is_tracing()
    if not sync:
        torch._assert_async(~violated, message)
        return False
    return bool(violated)


def assert_hardness_floor(
    eta: Tensor, sigma: Tensor, *, tol: float = 0.0, sync: Optional[bool] = None
) -> Tensor:
    """Assert ``eta_i >= S_0(sigma_i) = sqrt(2) K_E / (sigma_i sqrt(pi))`` for every atom.

    SPEC 1.2 as corrected in SPEC 17.3.  ``S_0`` is the Coulomb self-energy of
    the atom's own Gaussian -- the diagonal of the charge Gram matrix -- so
    ``diag(eta) + J_off = Gram + diag(eta - S_0)`` is positive semi-definite for
    EVERY geometry exactly when this holds.  It is also necessary: two sites
    brought into coincidence make the block indefinite for any ``eta_i < S_0``,
    and the energy is then unbounded below along that mode.

    The eta head emits ``softplus(raw) + S_0 + margin`` and satisfies this by
    construction; this check is the tripwire for parameters that did not come
    from it.  (The package used to demand ``eta >= 2 K_E / (sigma sqrt(pi)) =
    sqrt(2) S_0`` and to call everything below it unbounded; that was a
    sufficient condition presented as a necessary one.)

    A NaN ``eta`` or ``sigma`` counts as a violation: the predicate is negated
    so that NaN cannot pass.

    ``sync`` controls whether the flag is read on the host (see the module
    docstring); it is called once per parameter set from the solver's hot path.

    Returns
    -------
    Tensor
        The per-atom margin ``eta - S_0``; non-negative on success.
    """
    eta = eta.reshape(-1)
    sigma = sigma.reshape(-1)
    if eta.shape != sigma.shape:
        raise ValueError(
            f"eta and sigma must have the same shape, got {tuple(eta.shape)} and "
            f"{tuple(sigma.shape)}"
        )
    bad_sigma = ~(sigma > 0)
    if _needs_host_report(
        bad_sigma.any(), "sigma must be strictly positive (and not NaN)", sync=sync
    ):
        raise ElectrostaticsAssertionError("sigma must be strictly positive")

    floor = gram_self_energy(sigma)
    margin = eta - floor
    violated = ~(margin >= -abs(tol))
    if _needs_host_report(
        violated.any(),
        "hardness floor violated: eta_i >= S_0 = sqrt(2) K_E / (sigma_i sqrt(pi)) "
        "fails for at least one atom (or eta/sigma is NaN). Below S_0 the charge "
        "block is indefinite for close enough sites and the energy is unbounded "
        "below. The eta head must emit softplus(raw) + S_0 + margin.",
        sync=sync,
    ):
        index = torch.nonzero(violated).reshape(-1)
        worst = int(index[int(torch.argmin(margin[index]))])
        raise ElectrostaticsAssertionError(
            "hardness floor violated: eta_i >= S_0 = sqrt(2) K_E / (sigma_i sqrt(pi)) "
            f"fails for {int(violated.sum())} atom(s); worst is atom {worst} with "
            f"eta={float(eta[worst]):.6f} eV < S_0={float(floor[worst]):.6f} eV "
            f"(sigma={float(sigma[worst]):.6f} A). Below S_0 the charge block is "
            "indefinite for close enough sites and the energy is unbounded below. "
            "The eta head must emit softplus(raw) + S_0 + margin."
        )
    return margin


def assert_positive_kappa(
    kappa: Tensor, *, tol: float = 0.0, sync: Optional[bool] = None
) -> Tensor:
    """Assert ``kappa_ij > 0`` on every bond (SPEC 1.4, 2.1 item 2).

    The bond hardness is what makes charge transfer local and what keeps the
    SQE Hessian positive definite; a non-positive entry removes the penalty on
    unbounded charge flow along that bond.  ``kappa -> 0`` is a legitimate
    *limit* (it must reproduce global QEq to 1e-15, SPEC 1.4) but is reached
    from above, via ``softplus(...) + kappa_0`` with ``kappa_0 > 0``.

    **This is strict positivity in exact arithmetic, not a guarantee of
    numerical positive definiteness.**  At the default ``tol = 0`` the smallest
    subnormal, ``kappa = 5e-324``, passes, and on a 6-atom ring that
    parameterisation measures ``min eig(K_tt) = -2.397e-15``: the bond block is
    numerically indefinite even though every ``kappa`` is "positive".  That is
    the intended behaviour -- SPEC 1.4 requires the ``kappa -> 0`` limit to stay
    reachable, and the regression suite drives it to 1e-16 -- but no caller may
    infer a positive definite Hessian from this check passing.  Raise ``tol`` to
    the curvature scale actually needed if that inference matters.
    """
    kappa = kappa.reshape(-1)
    violated = ~(kappa > abs(tol))  # negated so NaN counts as a violation
    if _needs_host_report(
        violated.any(),
        "kappa must be strictly positive (and not NaN) on every bond; "
        "use softplus(raw) + kappa_0 with kappa_0 > 0",
        sync=sync,
    ):
        index = torch.nonzero(violated).reshape(-1)
        worst = int(index[int(torch.argmin(kappa[index]))])
        raise ElectrostaticsAssertionError(
            f"kappa must be strictly positive: {int(violated.sum())} bond(s) violate it; "
            f"worst is bond {worst} with kappa={float(kappa[worst]):.6e} eV/e^2. "
            "Use softplus(raw) + kappa_0 with kappa_0 > 0."
        )
    return kappa


def assert_symmetric_kappa(
    kappa: Tensor,
    bond_index: Tensor,
    *,
    num_atoms: Optional[int] = None,
    tol: float = 1e-12,
    require_positive: bool = True,
    directed: Optional[bool] = None,
) -> Tensor:
    """Assert that the bond hardness is symmetric under ``i <-> j``.

    ``kappa_ij`` multiplies ``t_ij^2`` and ``t_ij = -t_ji``, so an asymmetric
    ``kappa`` makes the energy depend on an arbitrary bookkeeping choice --
    which direction of the pair the code happened to visit -- and the energy is
    then simply ill-defined (SPEC 1.4).  Attention logits do **not** come out
    symmetric on their own: ``Q_i . K_j != Q_j . K_i``.  The head must emit
    ``softplus(0.5 * (s_ij + s_ji)) + kappa_0``.

    Accepts either form of ``kappa``:

    * one entry per *undirected* bond, matching ``bond_index`` column-for-column,
      in which case symmetry is structural and only positivity is checked;
    * one entry per *directed* edge, in which case the two directions of each
      pair are compared.

    ``directed`` says which.  **The two cannot be told apart from the data**, and
    leaving it at ``None`` (auto) is only as safe as the list handed in: auto
    treats "every pair appears exactly once" as the per-bond form and returns
    zero without comparing anything.  A *directed* edge list that happens to
    carry only one direction per pair looks identical -- ``(0,1)`` and ``(0,2)``
    with ``kappa = [1.0, 9.0]`` returns ``0.0`` unchecked -- and that is exactly
    the list a head emits when the reverse edges were dropped upstream.  Pass
    ``directed=True`` where the input is meant to be a directed edge list and
    every pair is then *required* to appear in both directions.

    This function is the SPEC 1.4 / SPEC 6 "asymmetric kappa raises" tripwire and
    belongs at the head, on the raw per-edge logit *before* symmetrisation --
    once :func:`trace_electrostatics.topology.symmetrise_edge_scalar` has run,
    symmetry is structural and there is nothing left to catch.

    **Nothing inside this package calls it, and that is correct, not an
    oversight.**  Two reasons, both checkable:

    * :class:`~trace_electrostatics.heads.ElectrostaticHeads` builds ``kappa``
      as ``softplus(0.5 * (s_ij + s_ji)) + kappa_0`` from logits it computes in
      both directions itself, and IEEE-754 addition is exactly commutative, so
      the result is BITWISE invariant under swapping the rows of ``bond_index``
      -- verified on a 4-bond chain, all four entries identical to the last bit,
      with ``t_seed`` exactly negated.  There is no asymmetry for this check to
      find in a ``kappa`` this package produced;
    * the solver cannot call it either: by the time ``kappa`` reaches
      :class:`~trace_electrostatics.solver.ElectrostaticParams` it is already one
      entry per undirected bond, and the ``torch.unique`` this check needs is
      data-dependent, so calling it there would break the AOT graph for no
      information gained.

    It is exported for a **third-party** head, where neither guarantee holds.
    Do not wire it into the hot path to make it look used.

    Returns
    -------
    Tensor
        The largest observed ``|kappa_ij - kappa_ji|``.
    """
    kappa = kappa.reshape(-1)
    if bond_index.dim() != 2 or bond_index.shape[0] != 2:
        raise ValueError(
            f"bond_index must have shape [2, n_edges], got {tuple(bond_index.shape)}"
        )
    if kappa.numel() != bond_index.shape[1]:
        raise ValueError(
            f"kappa has {kappa.numel()} entries but bond_index has "
            f"{int(bond_index.shape[1])} columns"
        )

    if require_positive:
        assert_positive_kappa(kappa)

    if kappa.numel() == 0:
        return torch.zeros((), dtype=kappa.dtype, device=kappa.device)

    src = bond_index[0].long()
    dst = bond_index[1].long()
    if num_atoms is None:
        num_atoms = int(torch.maximum(src.max(), dst.max())) + 1
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    key = lo * num_atoms + hi

    unique_key, inverse = torch.unique(key, sorted=True, return_inverse=True)
    n_groups = int(unique_key.numel())

    if directed:
        # A pair that appears in only one direction cannot be checked for
        # symmetry at all, and the group spread below would report it as
        # perfectly symmetric. Reject it instead of pretending.
        counts = torch.bincount(inverse, minlength=n_groups)
        if bool((counts < 2).any()):
            lonely = int((counts < 2).sum())
            worst = int(torch.argmin(counts))
            pair = (
                int(torch.div(unique_key[worst], num_atoms, rounding_mode="floor")),
                int(unique_key[worst] % num_atoms),
            )
            raise ElectrostaticsAssertionError(
                f"kappa was declared to be one entry per DIRECTED edge, but {lonely} "
                f"pair(s) appear in only one direction (e.g. {pair}). Symmetry under "
                "i <-> j cannot be checked against an edge list that carries only one "
                "direction, and it would be reported as symmetric. Supply both "
                "directions, or pass directed=False if this is one entry per "
                "undirected bond."
            )
    elif n_groups == kappa.numel():
        # Every pair occurs exactly once, so there is only one variable per pair
        # and t_ij = -t_ji makes the symmetry structural. See the docstring for
        # why this shortcut is only sound when the caller has not declared the
        # list directed.
        return torch.zeros((), dtype=kappa.dtype, device=kappa.device)

    big = torch.full((n_groups,), float("-inf"), dtype=kappa.dtype, device=kappa.device)
    small = torch.full((n_groups,), float("inf"), dtype=kappa.dtype, device=kappa.device)
    group_max = big.scatter_reduce(0, inverse, kappa, reduce="amax", include_self=True)
    group_min = small.scatter_reduce(0, inverse, kappa, reduce="amin", include_self=True)
    spread = group_max - group_min
    worst_value = torch.max(spread)
    if bool((~(spread <= tol)).any()):  # negated so NaN counts as a violation
        worst = int(torch.argmax(spread))
        pair = (
            int(torch.div(unique_key[worst], num_atoms, rounding_mode="floor")),
            int(unique_key[worst] % num_atoms),
        )
        raise ElectrostaticsAssertionError(
            f"kappa is not symmetric: pair {pair} differs by "
            f"{float(worst_value):.3e} eV/e^2 between its two directions (tol={tol:g}). "
            "Since t_ij = -t_ji, an asymmetric kappa makes the energy ill-defined. "
            "Symmetrise the attention logit: softplus(0.5 * (s_ij + s_ji)) + kappa_0."
        )
    return worst_value


def assert_pair_operator_symmetry(
    *,
    coulomb: Optional[Tensor] = None,
    charge_dipole: Optional[Tensor] = None,
    dipole: Optional[Tensor] = None,
    rtol: float = DEFAULT_OPERATOR_SYMMETRY_RTOL,
    sync: Optional[bool] = None,
    charge_dipole_is_pair: bool = True,
) -> Dict[str, Tensor]:
    """Assert that supplied pair operators have the exchange symmetry of ``K``.

    ``U = 0.5 m^T K m + c^T m`` is a quadratic form: only the symmetric part of
    ``K`` contributes to it, so an asymmetric operator does not describe *any*
    energy.  The solver still runs -- projected CG on a non-symmetric operator
    neither diverges nor complains, it merely stops converging (measured: the
    residual stalls at 0.203 from 3.48 after 6 iterations, with
    ``|K - K^T| = 4.96``) -- and the returned energy is then a number with no
    variational meaning.  These are the operators that come in from outside
    (an Ewald lattice sum), so they are the ones that have to be checked.

    Each block carries the exchange symmetry of the term it multiplies, with
    ``r_vec = r_i - r_j``:

    * ``coulomb`` ``[n, n]``: ``J_ij == J_ji``.  ``J`` is even in ``r_vec``.
    * ``charge_dipole`` ``[n, n, 3]``: ``W_ij == -W_ji``.  ``T01`` is *odd* in
      ``r_vec``, and under a lattice sum the ``+n`` and ``-n`` images cancel, so
      this also forces the self-image diagonal to vanish.  Pass the
      ``ell``-symmetrised form ``W_ij = ell_i ell_j T01_ij``, which is the one
      whose antisymmetry is a plain index exchange -- the block as it enters
      ``K`` carries ``ell_j`` on one index only, deliberately, because it
      couples two different kinds of moment.
    ``charge_dipole_is_pair=False`` waives only the ``W_ij == -W_ji`` test, for
    an operator that legitimately contains a term which is not a pair
    interaction.  The slab (2-D Ewald) correction is the case this exists for:
    ``U = 2 pi K_E M_n^2 / V`` is a rank-1 global term in the total dipole, so
    its charge-dipole block is proportional to ``z_i`` alone, independent of
    ``j`` -- measured ``max|W_ij + W_ji| = 4.36`` where a Coulomb tensor gives
    exactly ``0.0``.

    Waiving it costs nothing that matters, and it is important to see why.  The
    assembled ``K`` takes its off-diagonal blocks as transposes of each other by
    construction, so ``K`` is symmetric -- and therefore ``U`` is well defined --
    whether or not ``W`` is pair-antisymmetric (verified: ``max|K - K^T| = 0.0``
    for both a Coulomb ``W`` and a rank-1 slab ``W``).  Pair antisymmetry is a
    *bug detector* for a tensor that claims to be a lattice sum of ``T01``, not a
    condition for the energy to exist.  Keep it on for those, off for a term that
    never claimed to be one.

    * ``dipole``: ``[3n, 3n]`` ``D == D^T``, or ``[n, n, 3, 3]``
      ``T_ij[a, b] == T_ji[b, a]``.  Site and Cartesian indices exchange
      *together*; this is the symmetry Maxwell reciprocity
      (``alpha == alpha^T``, SPEC 6) tests, and a Thole damping factor keyed on
      the wrong atom of the pair breaks exactly this.

    The tolerance is relative to the largest entry of each operator, since the
    blocks span many orders of magnitude between a contact pair and a distant
    one.  A converged Ewald sum satisfies all three to round-off.

    Returns
    -------
    dict
        The measured asymmetry of each operator that was supplied.
    """
    measured: Dict[str, Tensor] = {}
    checks = []

    if coulomb is not None:
        measured["coulomb"] = (coulomb - coulomb.transpose(-1, -2)).abs().max()
        checks.append(("coulomb", coulomb, measured["coulomb"], "J_ij == J_ji"))
    if charge_dipole is not None:
        # Odd under i <-> j: T01[i, j, :] == -T01[j, i, :].
        measured["charge_dipole"] = (
            (charge_dipole + charge_dipole.transpose(0, 1)).abs().max()
        )
        # Always MEASURED (it is useful telemetry either way), but only ENFORCED
        # for an operator claiming to be a lattice sum of pair tensors.  See the
        # docstring: K is symmetric by construction regardless, so this is a bug
        # detector, not a condition for U to exist.
        if charge_dipole_is_pair:
            checks.append(
            (
                "charge_dipole",
                charge_dipole,
                measured["charge_dipole"],
                "ell_i ell_j T01_ij == -ell_j ell_i T01_ji (odd in r_i - r_j, so the "
                "self-image diagonal vanishes)",
            )
        )
    if dipole is not None:
        # Site and Cartesian indices exchange together.
        if dipole.dim() == 4:
            transposed = dipole.permute(1, 0, 3, 2)
            rule = "T_ij[a, b] == T_ji[b, a]"
        elif dipole.dim() == 2:
            transposed = dipole.transpose(-1, -2)
            rule = "D[3i+a, 3j+b] == D[3j+b, 3i+a]"
        else:
            raise ValueError(
                "dipole operator must be [3n, 3n] or [n, n, 3, 3], got "
                f"{tuple(dipole.shape)}"
            )
        measured["dipole"] = (dipole - transposed).abs().max()
        checks.append(("dipole", dipole, measured["dipole"], rule))

    for name, operator, asymmetry, rule in checks:
        if operator.numel() == 0:
            continue
        scale = operator.abs().max().clamp_min(1.0)
        violated = ~(asymmetry <= rtol * scale)  # negated so NaN is a violation
        if _needs_host_report(
            violated,
            f"the supplied {name} operator does not satisfy {rule}; an asymmetric "
            "operator describes no energy at all, and the projected CG will run on "
            "it without complaining",
            sync=sync,
        ):
            raise ElectrostaticsAssertionError(
                f"the supplied {name} operator is not symmetric: {rule} is violated "
                f"by {float(asymmetry):.6e} against a largest entry of "
                f"{float(scale):.6e} (rtol={rtol:g}). Only the symmetric part of K "
                "enters U = 0.5 m^T K m, so an asymmetric operator does not describe "
                "an energy; the projected CG runs on it silently and simply stops "
                "converging."
            )
    return measured


def schur_complement(K_qq: Tensor, T01: Tensor, A_dipole: Tensor) -> Tensor:
    """Schur complement of the dipole block, ``S = K_qq - T01^T A_dipole^-1 T01``.

    ``S`` is the *effective* charge-sector curvature after the dipoles have
    relaxed: it is the Hessian of ``U`` restricted to the charge degrees of
    freedom with ``mu_hat`` at its conditional minimum.  Only ``S`` says
    anything about the stability of the coupled system; the raw ``K_qq`` block
    describes a fictitious system with the dipoles frozen.

    Parameters
    ----------
    K_qq:
        ``[n_q, n_q]`` charge-sector block (for the SQE parameterisation this is
        ``B^T J B + diag(kappa)`` in bond space).
    T01:
        ``[n_mu, n_q]`` charge-dipole coupling block, mapping a charge degree of
        freedom to the dipole sector.
    A_dipole:
        ``[n_mu, n_mu]`` dipole block ``A_mu + T_Thole``.
    """
    if K_qq.dim() != 2 or K_qq.shape[0] != K_qq.shape[1]:
        raise ValueError(f"K_qq must be square, got {tuple(K_qq.shape)}")
    if A_dipole.dim() != 2 or A_dipole.shape[0] != A_dipole.shape[1]:
        raise ValueError(f"A_dipole must be square, got {tuple(A_dipole.shape)}")
    if T01.dim() != 2 or T01.shape != (A_dipole.shape[0], K_qq.shape[0]):
        raise ValueError(
            "T01 must have shape [n_mu, n_q] = "
            f"({int(A_dipole.shape[0])}, {int(K_qq.shape[0])}), got {tuple(T01.shape)}"
        )
    relaxed = torch.linalg.solve(A_dipole, T01)
    return K_qq - T01.transpose(-1, -2) @ relaxed


def null_space_basis(
    constraint_matrix: Optional[Tensor],
    dimension: int,
    *,
    rtol: float = DEFAULT_RANK_RTOL,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Orthonormal basis ``Z`` of ``null(constraint_matrix)``, ``[dimension, k]``.

    ``None`` or an all-zero constraint matrix yields the identity: no
    constraint removes no directions.  Note that the per-fragment constraint
    matrix acting on bond variables is *always* rank deficient -- summing its
    rows gives ``1^T B = 0`` -- so a plain inverse of ``A A^T`` is never
    available and the rank must be determined from the singular values.

    The column count is validated *before* the empty-matrix shortcut.  A
    ``[0, 5]`` matrix handed in for a 3-dimensional space is a caller bug --
    the constraint was built over the wrong variable vector -- and returning
    ``eye(3)`` for it hid that mismatch behind a silently unconstrained
    projection.
    """
    if dtype is None:
        dtype = constraint_matrix.dtype if constraint_matrix is not None else torch.get_default_dtype()
    if device is None and constraint_matrix is not None:
        device = constraint_matrix.device

    if constraint_matrix is not None:
        if constraint_matrix.dim() != 2:
            raise ValueError(
                f"constraint matrix must be 2-D, got {tuple(constraint_matrix.shape)}"
            )
        if constraint_matrix.shape[1] != dimension:
            raise ValueError(
                f"constraint matrix has {int(constraint_matrix.shape[1])} columns but the "
                f"space has dimension {dimension}"
            )

    if constraint_matrix is None or constraint_matrix.numel() == 0:
        return torch.eye(dimension, dtype=dtype, device=device)

    _, singular, vh = torch.linalg.svd(constraint_matrix, full_matrices=True)
    if singular.numel() == 0:
        return torch.eye(dimension, dtype=dtype, device=device)
    rank = int((singular > rtol * singular.max()).sum())
    return vh[rank:].transpose(-1, -2).contiguous()


def assert_positive_definite_at_solution(
    K_qq: Tensor,
    T01: Tensor,
    A_dipole: Tensor,
    constraint_matrix: Optional[Tensor] = None,
    *,
    tol: float = DEFAULT_EIGENVALUE_TOL,
    enabled: Optional[bool] = None,
) -> Optional[Dict[str, Tensor]]:
    """Assert the constraint-projected Schur complement is PD at the solution.

    This is SPEC 2.1 item 3, and the one check whose *obvious* cheaper version
    is wrong.  Checking the two blocks separately is **the trap**: the audit
    measured a charge block of ``+0.58``, a dipole block of ``+16.45``, and a
    full coupled system of ``-1.05``.  Charge transfer and induced dipoles
    destabilise each other through ``T01``; only the Schur complement sees it.

    The projection matters too: the physical problem is constrained, so
    curvature along directions that violate the per-fragment charge constraints
    is irrelevant, and including them reports spurious negative eigenvalues
    (the constraint matrix in bond space is always rank deficient).

    "At the solution" means the blocks handed in must be the ones actually
    assembled at the returned configuration -- not construction-time estimates,
    and not a different geometry.

    Parameters
    ----------
    K_qq, T01, A_dipole:
        Blocks as in :func:`schur_complement`.
    constraint_matrix:
        ``[n_constraints, n_q]`` constraints acting on the charge sector; the
        rows must live in the same space as the rows of ``K_qq`` (bond space for
        the SQE parameterisation, i.e. ``C @ B``).
    tol:
        Minimum acceptable eigenvalue.
    enabled:
        ``None`` defers to the global validation mode; ``True``/``False``
        override it.  The check costs an eigendecomposition, so it is off in
        production.

    Returns
    -------
    dict or None
        ``None`` when skipped, otherwise the measured eigenvalues.
    """
    if not _resolve(enabled):
        return None

    dipole_eigs = torch.linalg.eigvalsh(_symmetrised(A_dipole))
    min_dipole = dipole_eigs.min()
    if bool(min_dipole <= tol):
        raise ElectrostaticsAssertionError(
            f"dipole block A_mu + T_Thole is not positive definite: min eigenvalue "
            f"{float(min_dipole):.6e} <= {tol:g}. The Thole relay is positive definite "
            "for all r and all alpha only while a < 1 (SPEC 1.6); check the damping "
            "parameter and the dipole stiffness head."
        )

    schur = schur_complement(K_qq, T01, A_dipole)
    basis = null_space_basis(constraint_matrix, int(schur.shape[0]))
    projected = basis.transpose(-1, -2) @ schur @ basis
    if projected.numel() == 0:
        min_schur = torch.full((), float("inf"), dtype=schur.dtype, device=schur.device)
    else:
        min_schur = torch.linalg.eigvalsh(_symmetrised(projected)).min()

    if bool(min_schur <= tol):
        raw_eigs = torch.linalg.eigvalsh(_symmetrised(K_qq))
        raise ElectrostaticsAssertionError(
            "the constraint-projected Schur complement is not positive definite at the "
            f"solution: min eigenvalue {float(min_schur):.6e} <= {tol:g}. The charge "
            f"block alone measures {float(raw_eigs.min()):.6e} and the dipole block "
            f"{float(min_dipole):.6e}, so checking the blocks separately would have "
            "passed. The coupled system is a saddle: charge transfer and induced "
            "dipoles are destabilising each other through T01."
        )

    return {
        "min_eig_schur_projected": min_schur,
        "min_eig_dipole": min_dipole,
        "schur_asymmetry": _asymmetry(schur),
    }


def assert_is_minimum(
    hessian: Tensor,
    gradient: Optional[Tensor] = None,
    constraint_matrix: Optional[Tensor] = None,
    *,
    eig_tol: float = DEFAULT_EIGENVALUE_TOL,
    grad_tol: float = 1e-6,
    enabled: Optional[bool] = None,
) -> Optional[Dict[str, Tensor]]:
    """Assert the returned point is a *minimum*, not merely stationary.

    SPEC 2.1 item 4.  Newton-type solvers converge silently onto saddle points
    and then report a **negative** polarizability, because
    ``alpha = D^T K^-1 D`` inherits the sign of the curvature it was evaluated
    at.  Stationarity alone (``grad = 0``) does not distinguish a minimum from a
    saddle; the projected Hessian has to be positive definite as well.

    The Hessian is projected onto the constraint null space for the same reason
    as in :func:`assert_positive_definite_at_solution`, and it must be the
    **full coupled** Hessian over ``m = (t, mu_hat)``; block-wise positivity
    proves nothing (audit: ``+0.58``, ``+16.45``, coupled ``-1.05``).

    Parameters
    ----------
    hessian:
        ``[n_m, n_m]`` full Hessian ``K`` of ``U = 0.5 m^T K m + chi^T m``.
    gradient:
        Optional ``[n_m]`` gradient ``K m + c`` at the returned point.  Pass it
        only when the solve is expected to have **converged**: the production
        model deliberately truncates the solve at a fixed iteration count, so a
        non-zero projected gradient is the normal, correct state of affairs
        there and is not an error.
    constraint_matrix:
        ``[n_constraints, n_m]`` constraints over the full variable vector.
    eig_tol, grad_tol:
        Tolerances for the curvature and stationarity tests.
    enabled:
        ``None`` defers to the global validation mode.

    Returns
    -------
    dict or None
        ``None`` when skipped, otherwise the measured curvature and gradient
        norm.
    """
    if not _resolve(enabled):
        return None

    dimension = int(hessian.shape[0])
    basis = null_space_basis(constraint_matrix, dimension)
    projected = basis.transpose(-1, -2) @ hessian @ basis
    if projected.numel() == 0:
        min_eig = torch.full((), float("inf"), dtype=hessian.dtype, device=hessian.device)
    else:
        min_eig = torch.linalg.eigvalsh(_symmetrised(projected)).min()

    if bool(min_eig <= eig_tol):
        raise ElectrostaticsAssertionError(
            "the returned point is not a minimum: the constraint-projected Hessian has "
            f"min eigenvalue {float(min_eig):.6e} <= {eig_tol:g}. A stationary point "
            "with negative curvature is a saddle, and the polarizability computed "
            "there comes out negative."
        )

    result: Dict[str, Tensor] = {
        "min_eig_projected_hessian": min_eig,
        "hessian_asymmetry": _asymmetry(hessian),
    }

    if gradient is not None:
        projected_gradient = basis.transpose(-1, -2) @ gradient.reshape(-1)
        gradient_norm = torch.linalg.vector_norm(projected_gradient)
        result["projected_gradient_norm"] = gradient_norm
        if bool(gradient_norm > grad_tol):
            raise ElectrostaticsAssertionError(
                "the returned point is not stationary: projected gradient norm "
                f"{float(gradient_norm):.6e} > {grad_tol:g}. Increase the iteration "
                "count, or do not request the stationarity test for a deliberately "
                "truncated fixed-iteration solve."
            )
    return result


def assert_fragment_charges(
    q: Tensor,
    constraint_matrix: Tensor,
    targets: Tensor,
    tol: float = DEFAULT_CHARGE_TOL,
    *,
    sync: Optional[bool] = None,
) -> Tensor:
    """Assert ``|sum_{i in F} q_i - Q_F| < tol`` for every fragment.

    SPEC 2.1 item 5.  Cheap (one mat-vec) and always on: a violated charge
    constraint means the solve drifted off the feasible set, and every
    downstream observable -- dipole, polarizability, Born charges -- is then
    computed for a system with the wrong net charge.

    This is the assertion the solver reaches on *every* solve, so it reads its
    flag through :func:`_needs_host_report`: async device-side while tracing (no
    graph break), a host read with the full message in eager.  ``sync=False``
    forces the async form for a GPU loop that must not stall.

    Returns
    -------
    Tensor
        The largest per-fragment violation.
    """
    q = q.reshape(-1)
    targets = targets.reshape(-1)
    if constraint_matrix.dim() != 2:
        raise ValueError(
            f"constraint_matrix must be 2-D, got {tuple(constraint_matrix.shape)}"
        )
    if constraint_matrix.shape[1] != q.numel():
        raise ValueError(
            f"constraint_matrix has {int(constraint_matrix.shape[1])} columns but q has "
            f"{q.numel()} entries"
        )
    if constraint_matrix.shape[0] != targets.numel():
        raise ValueError(
            f"constraint_matrix has {int(constraint_matrix.shape[0])} rows but targets "
            f"has {targets.numel()} entries"
        )

    residual = constraint_matrix @ q - targets
    if residual.numel() == 0:
        return torch.zeros((), dtype=q.dtype, device=q.device)
    worst = residual.abs().max()
    if _needs_host_report(
        ~(worst <= tol),  # negated so a NaN charge counts as a violation
        "per-fragment charge constraint violated beyond tol (SPEC 2.1 item 5)",
        sync=sync,
    ):
        index = int(torch.argmax(residual.abs()))
        raise ElectrostaticsAssertionError(
            f"per-fragment charge constraint violated: fragment {index} carries "
            f"{float((constraint_matrix @ q)[index]):.12f} e but was constrained to "
            f"{float(targets[index]):.12f} e (violation {float(worst):.3e} > {tol:g})."
        )
    return worst


# ---------------------------------------------------------------------------
# The fractional-spin (flat-plane) error -- SPEC 9.2
#
# THE AXIS, ONCE, SO NOTHING BELOW IS AMBIGUOUS
# ---------------------------------------------
# Everything in this section measures curvature along
#
#     m_F  =  2 S_F  =  sum_{i in F} s_i  =  N_up - N_down        [electrons]
#
# the number of unpaired electrons in fragment F -- NOT the spin projection
# M_s = m / 2, and NOT the spin quantum number S = m / 2.  ``m`` is the axis
# because it is the quantity the model's own constraint pins
# (``sum_{i in F} s_i = 2 S_F``, SPEC 9) and the one SPEC 9.2's measured numbers
# are quoted on: "a quadratic term with J = 2 eV manufactures a 1.0 eV splitting
# at m = 1" is 0.5 * J^x * m^2 at m = 1.  A curvature reported per M_s^2 is FOUR
# TIMES the number reported here, and mixing the two is a factor of four in a
# quantity whose whole purpose is to be compared against a 0.1-0.4 eV signal --
# so the per-M_s form is offered as a named property and never as a bare float.
#
#     curvature   d^2 E / d m^2       eV / electron^2
#     slope       d E / d m  at m=0   eV / electron
#     artifact    0.5 m^T H m         eV
#
# The charge axis is held fixed throughout: ``N`` (every fragment's ``Q_F``) does
# not move.  In this model that is exact rather than approximate -- SPEC 9 puts
# structural zeros between the charge and spin bond sectors of ``K``, so the
# charge sector does not enter the spin curvature at all.
# ---------------------------------------------------------------------------

# Above this fraction, a predicted spin-state splitting is mostly a property of
# the functional form rather than of the fit.  0.5 is not a physical constant:
# it is the point past which the quadratic-in-s term supplies more of the number
# than everything else put together, so the splitting can no longer be read as a
# prediction.  SPEC 9.2's own example is far past it -- a 1.0 eV artifact against
# a real Fe(II) high-spin/low-spin gap of 0.1-0.4 eV is a fraction of 2.5 to 10.
DEFAULT_ARTIFACT_FRACTION_LIMIT: float = 0.5


def _as_spin_vector(
    m: Union[Tensor, Sequence[float], float, int],
    n_fragments: int,
    *,
    dtype: torch.dtype,
    device: Optional[torch.device],
    name: str,
) -> Tensor:
    """Coerce a spin configuration to a ``[n_fragments]`` tensor of ``m_F = 2 S_F``."""
    if not isinstance(m, Tensor):
        m = torch.as_tensor(m, dtype=dtype, device=device)
    m = m.reshape(-1).to(dtype=dtype, device=device)
    if m.numel() == 1 and n_fragments != 1:
        raise ValueError(
            f"{name} was given as a single number but the system has {n_fragments} "
            "fragments. Broadcasting it would silently put the same number of "
            "unpaired electrons on every fragment, which is a physical claim about "
            "the system rather than a convenience. Pass one m_F = 2 S_F per fragment."
        )
    if m.numel() != n_fragments:
        raise ValueError(
            f"{name} must have {n_fragments} entries (one m_F = 2 S_F per fragment), "
            f"got {m.numel()}"
        )
    return m


@dataclass(frozen=True)
class SpinStateSplitting:
    """How much of a predicted spin-state splitting is the quadratic artifact.

    Produced by :meth:`FractionalSpinError.splitting`.  All energies in eV; the
    spin configurations ``m_low`` and ``m_high`` are in unpaired electrons
    (``m_F = 2 S_F``), one entry per fragment.

    The decomposition, and why it is not arbitrary
    ----------------------------------------------
    The model's spin-sector energy is exactly quadratic in ``m``::

        E(m) = 0.5 m^T H m  +  g^T m  +  const

    so a splitting between two spin states splits into two additive pieces:

    * :attr:`linear_part` ``= g . (m_high - m_low)``.  A linear dependence on
      ``M_s`` is **admissible** under the flat-plane condition -- the exact
      energy is piecewise *linear*, not constant -- and in this model it is
      carried by ``chi_spin``, which is precisely the fitted degree of freedom
      SPEC 9.1 says spin polarisation must come from.
    * :attr:`artifact` ``= 0.5 (m_high^T H m_high - m_low^T H m_low)``.  This is
      the contribution of the quadratic-in-``s`` term, the term that violates the
      flat-plane condition.  It is the number that is there because of the shape
      of the functional and not because of what was fitted.

    The split is anchored at ``m = 0`` because the model anchors it there: the
    on-site term is ``0.5 J^x_i s_i^2``, whose zero is the unpolarised state
    ``q^up = q^down``.  That is a physical point, not a gauge choice, so the
    attribution is well posed.

    :attr:`nonlinearity` is the same information with no anchor at all: the
    largest vertical gap between ``E(m)`` and the straight line joining the two
    end states, ``|d^T H d| / 8`` with ``d = m_high - m_low``, attained at the
    midpoint.  The flat-plane condition forbids exactly this bow.  Quote it when
    a reader objects that two end points can always be joined by a line, because
    that objection is correct and this number answers it.

    **The splitting is FITTED, not derived.**  See :class:`FractionalSpinError`.
    """

    splitting: Tensor
    artifact: Tensor
    linear_part: Tensor
    nonlinearity: Tensor
    artifact_fraction: Tensor
    m_low: Tensor
    m_high: Tensor
    bonds_relaxed: bool

    def is_artifact_dominated(
        self, limit: float = DEFAULT_ARTIFACT_FRACTION_LIMIT
    ) -> bool:
        """Whether more than ``limit`` of the splitting comes from the artifact.

        Reads the fraction on the host, so it synchronises; this is a reporting
        call, not a hot-path one.  A NaN fraction counts as dominated -- the
        predicate is negated for the same reason every other predicate in this
        module is.
        """
        return not bool(self.artifact_fraction <= limit)

    def summary(self) -> str:
        """One-paragraph plain-language verdict, for a log or a report."""
        fraction = float(self.artifact_fraction)
        verdict = (
            "NOT a prediction: most of this number is the functional form"
            if self.is_artifact_dominated()
            else "the artifact is a minority of the number, but it is still there"
        )
        return (
            f"predicted splitting {float(self.splitting):+.4f} eV = "
            f"{float(self.linear_part):+.4f} eV fitted (linear in M_s, admissible) "
            f"{float(self.artifact):+.4f} eV quadratic artifact; artifact fraction "
            f"{fraction:.1%} -- {verdict}. The energy bows {float(self.nonlinearity):.4f} "
            "eV off the straight line joining the two spin states, where the "
            "flat-plane condition requires no bow at all. Spin-state splittings "
            "from this model are FITTED quantities, not derived ones (SPEC 9.2)."
        )


@dataclass(frozen=True)
class FractionalSpinError:
    """The model's own fractional-spin error, measured in eV.

    What this is
    ------------
    The exact energy of a many-electron system is piecewise planar in
    ``(N, M_s)``: at fixed electron number it is piecewise **linear** in the spin
    polarisation, and for a degenerate open shell -- the one-electron doublet
    being the cleanest case -- it is exactly **flat**.  Curvature along ``M_s``
    is the fractional-spin error, also called the static-correlation error.

    This model's on-site exchange term ``0.5 J^x_i s_i^2`` is quadratic in ``s``,
    so it has such curvature by construction, at every parameterisation.
    :attr:`curvature` is that curvature: ``d^2 E / d m_F d m_G`` in
    **eV per unpaired electron squared**, with ``m_F = 2 S_F`` the number of
    unpaired electrons on fragment F.  **It should be zero for a degenerate open
    shell and never is.**

    **The spin-state splitting this model produces is a FITTED quantity, not a
    derived one.**  It inherits the fractional-spin error of the functional form,
    which no amount of training removes: training can only choose where along the
    parabola the fit sits.  This is the same functional form AIMNet2-NSE and
    MACE-POLAR-1 use, so it is a known limitation of the whole class of models,
    not of this implementation -- and it is not a reason to refuse the feature,
    only a reason to report it.

    Attributes
    ----------
    curvature:
        ``[n_fragments, n_fragments]`` ``d^2 E / d m_F d m_G``, eV/e^2.  Positive
        semi-definite: it is a Schur complement of a positive definite block.
    curvature_frozen:
        The same quantity with the bond transfer ``t^s`` held at zero instead of
        relaxed.  Because relaxation can only lower a minimum, this is a rigorous
        **upper bound** on :attr:`curvature`, and it costs ``O(n_atoms)`` rather
        than a factorisation -- which is what makes it usable as training
        telemetry.  Equal to :attr:`curvature` when ``bonds_relaxed`` is False.
    slope:
        ``[n_fragments]`` ``dE/dm_F`` at ``m = 0``, eV/e.  Reported because it is
        the *admissible* part: a linear dependence on ``M_s`` does not violate
        the flat-plane condition, and it is what ``chi_spin`` fits.
    bonds_relaxed:
        Whether ``t^s`` was relaxed.  ``False`` means every number here is an
        upper bound rather than the model's own value.

    What the curvature does NOT depend on
    -------------------------------------
    For a single fragment with a uniform ``J^x`` and uniform reference weights
    the curvature is exactly ``J^x / n_atoms``, independent of the geometry, the
    bond graph and ``kappa^s``: the spin-transfer relaxation contributes nothing
    because ``1^T B = 0`` makes its driving term vanish identically.  Measured
    through the full assemble-and-solve path at ``J^x = 2 eV`` for ``n_atoms`` of
    1, 2, 3 and 6: 2, 1, 0.666666666667 and 0.333333333333 eV/e^2, each with an
    error of **exactly 0.0** against ``J^x / n``.  Two consequences:

    * the artifact is inversely proportional to how far the spin is spread.  One
      unpaired electron pinned on a single metal centre carries the full
      ``0.5 J^x``; the same electron smeared over six atoms carries a sixth of
      it.  ``reference_weights`` is therefore a physically load-bearing input,
      not a bookkeeping default -- but note that the *relaxation* also spreads
      the spin, and by a lot: a 7-atom centre-plus-ligands complex with the
      reference weight entirely on the metal and ``J^x = 2 eV`` has a frozen
      curvature of 2.0 and a relaxed one of 0.4 eV/e^2, because ``t^s`` carries
      the spin out onto the ligands;
    * a non-uniform ``J^x`` makes the relaxation bite in a second way.  Measured
      on a 6-atom chain with ``J^x`` drawn from [0.5, 3.5] eV: frozen bound
      0.235664, relaxed (exact) 0.195099, and the shipped fixed-6-iteration
      solve 0.195247 eV/e^2.  The fixed-k solve sits **within 0.08%** of the
      exact relaxed value, which is what makes the relaxed number the right one
      to report; the frozen bound is 21% above it and is the cheap one.
    """

    curvature: Tensor
    curvature_frozen: Tensor
    slope: Tensor
    bonds_relaxed: bool

    @property
    def n_fragments(self) -> int:
        return int(self.slope.numel())

    @property
    def worst_curvature(self) -> Tensor:
        """The largest per-fragment curvature, eV/e^2.  The telemetry scalar.

        The diagonal rather than the largest eigenvalue: a diagonal entry is the
        curvature of one chemical object, which is the thing a user can act on.
        Zero fragments gives zero -- there is nothing to be wrong about.
        """
        if self.curvature.numel() == 0:
            return torch.zeros(
                (), dtype=self.curvature.dtype, device=self.curvature.device
            )
        return self.curvature.diagonal().max()

    @property
    def curvature_per_ms_squared(self) -> Tensor:
        """:attr:`curvature` re-expressed per ``M_s^2`` rather than per electron^2.

        ``m = 2 M_s``, so this is ``4 x`` :attr:`curvature`.  Offered as a named
        property precisely so that the factor of four is never applied by
        accident: everything else in this module, and every number in SPEC 9.2,
        is per unpaired electron squared.
        """
        return 4.0 * self.curvature

    def artifact_energy(
        self, m: Union[Tensor, Sequence[float], float, int]
    ) -> Tensor:
        """``0.5 m^T H m``: the energy the quadratic term manufactures at ``m``.

        Measured from the unpolarised state ``m = 0``, which is where the model's
        own ``0.5 J^x s^2`` is anchored.  For the SPEC 9.2 calibration case --
        one electron, ``J^x = 2 eV``, ``m = 1`` -- this returns **1.0 eV**, where
        the exact answer is a degenerate doublet and therefore **0**.
        """
        vector = _as_spin_vector(
            m,
            self.n_fragments,
            dtype=self.curvature.dtype,
            device=self.curvature.device,
            name="m",
        )
        return 0.5 * (vector @ (self.curvature @ vector))

    def splitting(
        self,
        m_low: Union[Tensor, Sequence[float], float, int],
        m_high: Union[Tensor, Sequence[float], float, int],
    ) -> SpinStateSplitting:
        """Decompose the predicted splitting between two spin states.

        Both arguments are in unpaired electrons, ``m_F = 2 S_F``, one per
        fragment: ``(0, 4)`` for an Fe(II) low-spin (S=0) to high-spin (S=2)
        gap, ``(0, 1)`` for the SPEC 9.2 doublet calibration.

        Returns
        -------
        SpinStateSplitting
            Which carries the artifact fraction and a plain-language verdict.
        """
        low = _as_spin_vector(
            m_low,
            self.n_fragments,
            dtype=self.curvature.dtype,
            device=self.curvature.device,
            name="m_low",
        )
        high = _as_spin_vector(
            m_high,
            self.n_fragments,
            dtype=self.curvature.dtype,
            device=self.curvature.device,
            name="m_high",
        )
        artifact = self.artifact_energy(high) - self.artifact_energy(low)
        linear = self.slope @ (high - low)
        total = artifact + linear
        difference = high - low
        nonlinearity = (difference @ (self.curvature @ difference)).abs() / 8.0
        # A zero splitting with a non-zero artifact is infinitely
        # artifact-dominated. clamp_min keeps that finite and keeps the gradient
        # defined; 0/tiny is still exactly 0, so a genuinely artifact-free
        # splitting still reports 0.
        tiny = torch.finfo(self.curvature.dtype).tiny
        fraction = artifact.abs() / total.abs().clamp_min(tiny)
        return SpinStateSplitting(
            splitting=total,
            artifact=artifact,
            linear_part=linear,
            nonlinearity=nonlinearity,
            artifact_fraction=fraction,
            m_low=low,
            m_high=high,
            bonds_relaxed=self.bonds_relaxed,
        )

    def summary(self) -> str:
        """One-line plain-language report, for a training log."""
        bound = "" if self.bonds_relaxed else " (frozen-bond upper bound)"
        return (
            f"fractional-spin curvature{bound}: worst fragment "
            f"{float(self.worst_curvature):.6f} eV per unpaired electron^2, which "
            f"manufactures {0.5 * float(self.worst_curvature):.6f} eV at one "
            "unpaired electron where a degenerate open shell is exactly flat "
            "(SPEC 9.2). Spin-state splittings from this model are FITTED, not "
            "derived."
        )


def fractional_spin_curvature(
    j_exchange: Tensor,
    spin_reference_map: Tensor,
    *,
    chi_spin: Optional[Tensor] = None,
    incidence: Optional[Tensor] = None,
    kappa_spin: Optional[Tensor] = None,
    bond_constraint: Optional[Tensor] = None,
    relax_bonds: bool = True,
    rank_rtol: float = DEFAULT_RANK_RTOL,
) -> FractionalSpinError:
    """Measure the curvature this model imposes along ``M_s`` at fixed ``N``.

    This is the quantity the flat-plane condition requires to vanish for a
    degenerate open shell, and which ``0.5 J^x_i s_i^2`` makes non-zero by
    construction (SPEC 9.2).  See :class:`FractionalSpinError` for what the
    number means and :mod:`~trace_electrostatics.diagnostics`'s module docstring
    for why the answer is to measure it rather than to remove it.

    The derivation, so the closed form is checkable
    -----------------------------------------------
    The spin sector of SPEC 9 is, with ``J = diag(J^x)``, ``s = B t^s + rho^s``
    and ``rho^s = R m`` linear in the per-fragment unpaired-electron counts,

        U(t^s; m) = 0.5 s^T J s + 0.5 sum_b kappa^s_b (t^s_b)^2 + chi^s . s
                  = 0.5 t^T K_ss t + t^T (G m + h) + 0.5 m^T R^T J R m + chi^s . R m

    with ``K_ss = B^T J B + diag(kappa^s)``, ``G = B^T J R`` and
    ``h = B^T chi^s``, subject to the homogeneous ``A t^s = 0``, ``A = C B``.
    Minimising over ``t^s = Z y`` in the constraint null space ``Z`` gives

        H  =  R^T J R  -  (Z^T G)^T (Z^T K_ss Z)^-1 (Z^T G)
        g  =  R^T chi^s  -  (Z^T G)^T (Z^T K_ss Z)^-1 (Z^T h)

    Both terms subtracted are formed from a positive definite ``Z^T K_ss Z``
    (positive definite exactly because ``J^x > 0`` and ``kappa^s > 0``, SPEC
    9.1/9.0), so ``H`` is positive semi-definite and bounded above by
    ``R^T J R``.  That upper bound is :attr:`~FractionalSpinError.curvature_frozen`
    and costs no factorisation.

    Verified against a central-difference second derivative of the exactly
    relaxed energy on 1-, 2-, 4- and 5-atom systems with one and two fragments:
    agreement to **2.3e-09** on the curvature and **5.9e-12** on the slope, which
    is the finite-difference floor at ``h = 1e-4``, not an implementation gap
    (**1.3e-11** on a 6-atom chain at ``h = 1e-3``, where the floor is lower).
    Cross-checked a third way, by differentiating the *fixed-6-iteration* solve
    twice with autograd -- the curvature of the energy the model actually
    evaluates, rather than of its ``k -> infinity`` limit: 0.195247 against the
    0.195099 returned here, a gap of 0.08% (SPEC 8's k-dependence, an order
    smaller here than the factor of 2.2 it costs the polarizability, because the
    spin sector's CG residual is small by the sixth iteration).

    Parameters
    ----------
    j_exchange:
        ``[n_atoms]`` on-site exchange ``J^x_i``, eV/e^2.  Strictly positive
        (SPEC 9.1).
    spin_reference_map:
        ``[n_atoms, n_fragments]`` ``R = d rho^s / d m``, the derivative of the
        reference spin population with respect to the per-fragment unpaired
        electron count.  This is the Jacobian of
        :func:`~trace_electrostatics.topology.distribute_fragment_spins`; build
        it with ``trace_electrostatics.solver.fractional_spin_error`` rather
        than by hand unless there is a reason not to.
    chi_spin:
        ``[n_atoms]`` spin-channel linear term, eV/e.  ``None`` means zero, which
        makes :attr:`~FractionalSpinError.slope` zero and the whole splitting
        artifact.
    incidence, kappa_spin, bond_constraint:
        ``B``, ``kappa^s`` and ``C B``.  Required when ``relax_bonds`` is True
        and the system has bonds; ignored otherwise.
    relax_bonds:
        Whether to relax the spin-channel bond transfer ``t^s``.  ``True`` gives
        the model's own curvature and costs a null-space basis and a solve in
        bond space.  ``False`` gives the rigorous upper bound in ``O(n_atoms)``
        and is what belongs in a training loop.
    rank_rtol:
        Relative singular-value cutoff for the constraint null space; ``A = C B``
        is always rank deficient because ``1^T B = 0``.
    """
    j_exchange = j_exchange.reshape(-1)
    n_atoms = int(j_exchange.numel())
    if spin_reference_map.dim() != 2 or int(spin_reference_map.shape[0]) != n_atoms:
        raise ValueError(
            "spin_reference_map must have shape [n_atoms, n_fragments] = "
            f"[{n_atoms}, ...], got {tuple(spin_reference_map.shape)}"
        )
    dtype = spin_reference_map.dtype
    device = spin_reference_map.device
    n_fragments = int(spin_reference_map.shape[1])
    reference = spin_reference_map

    # A non-positive J^x is not merely a bad number here: the spin block stops
    # being positive definite, the "minimum" the curvature is defined from is a
    # saddle, and the Schur complement below is meaningless. SPEC 9.1.
    #
    # Through _needs_host_report rather than a bare bool(): the frozen branch of
    # this function runs as solver telemetry on every spin-resolved solve, and a
    # bare host read there would stall the device queue and break the compiled
    # graph -- exactly the cost the rest of this module is written to avoid.
    violated = ~(j_exchange > 0)
    if _needs_host_report(
        violated.any(),
        "fractional_spin_curvature needs J^x > 0 on every atom (SPEC 9.1): at "
        "J^x <= 0 the spin sector is unbounded below and the curvature measured "
        "here would be the curvature of a saddle",
    ):
        raise ElectrostaticsAssertionError(
            "fractional_spin_curvature needs J^x > 0 on every atom: "
            f"{int(violated.sum())} atom(s) violate it. The spin mode of the on-site "
            "Hessian has curvature 2 J^x (SPEC 9.1), so at J^x <= 0 the spin sector "
            "is unbounded below, the relaxed energy has no minimum, and the curvature "
            "reported here would be the curvature of a saddle."
        )

    frozen = reference.transpose(0, 1) @ (j_exchange.reshape(n_atoms, 1) * reference)
    if chi_spin is None:
        spin_potential = torch.zeros(n_atoms, dtype=dtype, device=device)
    else:
        spin_potential = chi_spin.reshape(-1).to(dtype=dtype, device=device)
        if spin_potential.numel() != n_atoms:
            raise ValueError(
                f"chi_spin must have {n_atoms} entries, got {spin_potential.numel()}"
            )
    slope_frozen = reference.transpose(0, 1) @ spin_potential

    n_bonds = 0 if incidence is None else int(incidence.shape[1])
    if not relax_bonds or n_bonds == 0:
        return FractionalSpinError(
            curvature=frozen,
            curvature_frozen=frozen,
            slope=slope_frozen,
            # With no bonds there is nothing to relax, so the frozen value IS the
            # relaxed one and saying otherwise would understate what is known.
            bonds_relaxed=(n_bonds == 0),
        )

    if kappa_spin is None:
        raise ValueError(
            "relax_bonds=True needs kappa_spin: the spin-transfer hardness is what "
            "makes the relaxed spin block positive definite (SPEC 9.0). Pass "
            "relax_bonds=False for the frozen-bond upper bound, which needs neither "
            "the bond graph nor kappa^s."
        )
    if incidence is None or int(incidence.shape[0]) != n_atoms:
        raise ValueError(
            "incidence must have shape [n_atoms, n_bonds] = "
            f"[{n_atoms}, {n_bonds}], got "
            f"{None if incidence is None else tuple(incidence.shape)}"
        )
    incidence = incidence.to(dtype=dtype, device=device)
    kappa_spin = kappa_spin.reshape(-1).to(dtype=dtype, device=device)
    if kappa_spin.numel() != n_bonds:
        raise ValueError(
            f"kappa_spin must have {n_bonds} entries to match the bond count, got "
            f"{kappa_spin.numel()}"
        )
    assert_positive_kappa(kappa_spin)

    k_ss = incidence.transpose(0, 1) @ (
        j_exchange.reshape(n_atoms, 1) * incidence
    ) + torch.diag(kappa_spin)
    basis = null_space_basis(
        None if bond_constraint is None else bond_constraint.to(dtype=dtype, device=device),
        n_bonds,
        rtol=rank_rtol,
        dtype=dtype,
        device=device,
    )
    if int(basis.shape[1]) == 0:
        # Every bond direction is pinned by the constraints, so there is nothing
        # to relax and the frozen value is exact.
        return FractionalSpinError(
            curvature=frozen,
            curvature_frozen=frozen,
            slope=slope_frozen,
            bonds_relaxed=True,
        )

    projected_k = basis.transpose(0, 1) @ k_ss @ basis
    drive = basis.transpose(0, 1) @ (
        incidence.transpose(0, 1) @ (j_exchange.reshape(n_atoms, 1) * reference)
    )
    potential_drive = basis.transpose(0, 1) @ (incidence.transpose(0, 1) @ spin_potential)
    relaxed = frozen - drive.transpose(0, 1) @ torch.linalg.solve(projected_k, drive)
    slope = slope_frozen - drive.transpose(0, 1) @ torch.linalg.solve(
        projected_k, potential_drive.reshape(-1, 1)
    ).reshape(-1)
    return FractionalSpinError(
        curvature=_symmetrised(relaxed),
        curvature_frozen=frozen,
        slope=slope,
        bonds_relaxed=True,
    )


def assert_spin_splitting_is_prediction(
    splitting: SpinStateSplitting,
    *,
    limit: float = DEFAULT_ARTIFACT_FRACTION_LIMIT,
) -> SpinStateSplitting:
    """Raise when a predicted spin-state splitting is mostly the artifact.

    A splitting that is 90% artifact is not a prediction, and this is the call
    that says so out loud.  It is deliberately a *raise* and not a warning: a
    spin-state energetics number read out of a model whose functional form
    supplied most of it will otherwise be quoted as if it were physics.

    Use it at the point where a number leaves the model for a user -- a report,
    a paper table, a ranking of spin states -- not inside the solve.  Reading
    the fraction synchronises with the device.

    ``limit`` is the artifact fraction above which the number is refused; see
    :data:`DEFAULT_ARTIFACT_FRACTION_LIMIT`.  Passing ``limit=float("inf")``
    turns it into a no-op for a caller who wants the numbers without the
    judgement.

    Returns
    -------
    SpinStateSplitting
        The argument, unchanged, so this can be used inline.
    """
    if splitting.is_artifact_dominated(limit):
        raise ElectrostaticsAssertionError(
            "this spin-state splitting is not a prediction: "
            f"{float(splitting.artifact_fraction):.1%} of it "
            f"({float(splitting.artifact):+.4f} eV of {float(splitting.splitting):+.4f} "
            f"eV) comes from the on-site term 0.5 J^x s^2, which violates the "
            "flat-plane condition by construction and would be exactly zero for a "
            "degenerate open shell (SPEC 9.2). The energy bows "
            f"{float(splitting.nonlinearity):.4f} eV off the straight line joining the "
            "two spin states, where the exact energy is piecewise linear. The "
            "spin-state splitting this model produces is a FITTED quantity, not a "
            "derived one, and it inherits the fractional-spin error of its functional "
            "form -- the same form AIMNet2-NSE and MACE-POLAR-1 use, so this is a "
            "limitation of the class of model, not of this implementation. Reduce it "
            "by lowering J^x or by spreading the spin reference over more atoms "
            f"(reference_weights), or raise the limit above {limit:g} deliberately."
            + ("" if splitting.bonds_relaxed else " NOTE: measured with the bond "
               "transfer frozen, so the artifact quoted is an upper bound.")
        )
    return splitting


def verify_spin_mode_eigenvalue(
    eta: Tensor,
    j_exchange: Tensor,
    *,
    tol: float = 1e-12,
) -> Dict[str, Tensor]:
    """Verify SPEC 9.1's ``(2 eta, 2 J^x)`` derivation numerically.

    SPEC 9.1 states that in the ``(q_up, q_down)`` basis the on-site energy
    ``0.5 eta q^2 + 0.5 J^x s^2``, with ``q = q_up + q_down`` and
    ``s = q_up - q_down``, has Hessian ``[[eta+J^x, eta-J^x], [eta-J^x, eta+J^x]]``
    with eigenvalues ``2 eta`` (charge mode) and ``2 J^x`` (spin mode), and hence
    that positive definiteness requires ``J^x > 0``.  The whole sign constraint
    rests on that, so it is checked rather than trusted.

    Three independent things are checked, at every supplied ``(eta, J^x)``:

    1. **The Hessian is the Hessian of that energy.**  With
       ``T = [[1, 1], [1, -1]]`` mapping ``(q_up, q_down)`` to ``(q, s)``, the
       energy is ``0.5 x^T T^T diag(eta, J^x) T x``, so its Hessian is
       ``T^T diag(eta, J^x) T``.  That product is formed here and compared with
       the matrix SPEC 9.1 writes down -- the chain rule, evaluated numerically.
    2. **The eigenvalues are ``2 eta`` and ``2 J^x``.**  Against
       :func:`torch.linalg.eigvalsh`, sorted, so the comparison survives the
       ``J^x > eta`` case and the degenerate ``J^x == eta`` case.
    3. **The eigenvectors are ``(1, 1)`` and ``(1, -1)``.**  Checked as
       ``H v - lambda v == 0`` rather than against the vectors ``eigh`` returns,
       because at ``J^x == eta`` the eigenvectors are not unique and a direct
       comparison would fail on a correct implementation.

    Measured over ``eta`` in {1, 10.832, 12, 25} eV/e^2 and ``J^x`` in
    [-3, 3] eV/e^2: eigenvalues to **1.78e-15**, eigenvector residuals to
    **2.06e-15**, Hessian to **0.0** (exactly -- the chain rule and the stated
    matrix agree bit for bit).  At ``eta = 12``, ``J^x = -0.5`` the spin mode is
    ``-1.0 eV/e^2`` and the energy reaches ``-25``, ``-2500``, ``-250000`` eV at
    ``|s| = 10, 100, 1000``: unbounded below, which is what the sign constraint
    exists to exclude.

    Parameters
    ----------
    eta, j_exchange:
        Broadcastable tensors of on-site hardness and on-site exchange, eV/e^2.
        ``J^x <= 0`` is *allowed here* -- the derivation holds for either sign
        and this function is how the ``J^x <= 0`` case is shown to be indefinite.
    tol:
        Absolute tolerance on each of the three residuals.

    Returns
    -------
    dict
        ``eigenvalue_error``, ``eigenvector_error``, ``hessian_error`` and
        ``min_eigenvalue`` (the smallest eigenvalue seen, negative exactly when
        some ``J^x <= 0``).
    """
    dtype = torch.promote_types(eta.dtype, j_exchange.dtype)
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    eta, j_exchange = torch.broadcast_tensors(
        eta.reshape(-1).to(dtype), j_exchange.reshape(-1).to(dtype)
    )

    # (1) the Hessian, by the chain rule rather than by assertion.
    transform = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=dtype, device=eta.device)
    on_site = torch.zeros(eta.numel(), 2, 2, dtype=dtype, device=eta.device)
    on_site[:, 0, 0] = eta
    on_site[:, 1, 1] = j_exchange
    derived = transform.transpose(0, 1) @ on_site @ transform
    stated = torch.stack(
        [
            torch.stack([eta + j_exchange, eta - j_exchange], dim=-1),
            torch.stack([eta - j_exchange, eta + j_exchange], dim=-1),
        ],
        dim=-2,
    )
    hessian_error = (derived - stated).abs().max()

    # (2) the eigenvalues.
    measured = torch.linalg.eigvalsh(stated)
    expected, _ = torch.sort(torch.stack([2.0 * eta, 2.0 * j_exchange], dim=-1), dim=-1)
    eigenvalue_error = (measured - expected).abs().max()

    # (3) the eigenvectors, as a residual so degeneracy cannot break the test.
    charge_mode = torch.tensor([1.0, 1.0], dtype=dtype, device=eta.device)
    spin_mode = torch.tensor([1.0, -1.0], dtype=dtype, device=eta.device)
    charge_residual = stated @ charge_mode - (2.0 * eta).reshape(-1, 1) * charge_mode
    spin_residual = stated @ spin_mode - (2.0 * j_exchange).reshape(-1, 1) * spin_mode
    eigenvector_error = torch.maximum(
        charge_residual.abs().max(), spin_residual.abs().max()
    )

    worst = torch.maximum(
        torch.maximum(hessian_error, eigenvalue_error), eigenvector_error
    )
    if not bool(worst <= tol):  # negated so NaN is a failure
        raise ElectrostaticsAssertionError(
            "the SPEC 9.1 spin-mode derivation does not hold numerically: worst "
            f"residual {float(worst):.6e} > {tol:g} (Hessian {float(hessian_error):.3e}, "
            f"eigenvalues {float(eigenvalue_error):.3e}, eigenvectors "
            f"{float(eigenvector_error):.3e}). The on-site Hessian in the "
            "(q_up, q_down) basis must be [[eta+J^x, eta-J^x], [eta-J^x, eta+J^x]] "
            "with eigenvalues 2 eta and 2 J^x; if that is false, the J^x > 0 "
            "constraint the whole spin sector rests on is not the right constraint."
        )
    return {
        "hessian_error": hessian_error,
        "eigenvalue_error": eigenvalue_error,
        "eigenvector_error": eigenvector_error,
        "min_eigenvalue": measured.min(),
    }


def _symmetrised(matrix: Tensor) -> Tensor:
    """Symmetric part, so eigenvalue routines see exactly what they assume."""
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _asymmetry(matrix: Tensor) -> Tensor:
    """Largest ``|M_ij - M_ji|``; reported, never silently removed.

    Silently symmetrising a matrix would hide a broken Thole implementation:
    the Maxwell reciprocity test ``alpha == alpha^T`` is precisely how such a
    bug is detected, and it only shows up at close contact (SPEC 6).
    """
    if matrix.numel() == 0:
        return torch.zeros((), dtype=matrix.dtype, device=matrix.device)
    return (matrix - matrix.transpose(-1, -2)).abs().max()
