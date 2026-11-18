"""The converged solve of the master equation, differentiated implicitly (SPEC 17).

The model's electrostatic energy is the constrained minimum of a quadratic form,

    E(theta) = min_{x in V} U(x; theta),   U = 0.5 x^T K x + c^T x,

with ``K`` and ``c`` built from the positions, the cell and the head outputs
(collectively ``theta``) and ``V`` the feasible subspace (bonds that cross an
explicit fragment boundary pinned to zero).  ``K`` is positive definite on
``V`` -- by theorem under the Gaussian kernel (SPEC 12, 14) -- so ``x*`` is
unique and ``E`` is a smooth function of ``theta``.

Why not "k CG steps, differentiated through" (the fixed-k design of SPEC 2)
-----------------------------------------------------------------------------
A Krylov iterate ``x_k = p_k(K) c`` uses a polynomial fitted to the spectrum
that ``c`` happens to excite.  Its response to a perturbation that excites
other modes -- a field on a symmetric molecule, a polar phonon of an ideal
crystal -- is therefore arbitrary and does not converge with ``k`` (measured:
alpha of an Oh cluster 0.399 at k = 6 and 12 against 0.730; a D6h ring -2.25
and +3.82 against 1.46).  And CG is not invariant under rescaling of the
variables, so the per-site length ``ell`` -- a pure gauge of the physics --
moved the k = 6 energy by 0.11 eV.  Both defects are properties of the
truncated map, not of the physics; the minimiser has neither.

How it is differentiated
------------------------
The solve itself runs WITHOUT autograd (preconditioned projected CG to a
relative residual ``rtol``).  Two ``autograd.Function``\\ s then attach exact
derivatives of every order:

* :class:`_MinimumEnergy` -- ``E``.  Its backward is the envelope theorem,
  ``dE = x*^T dc + 0.5 x*^T dK x*``: forces cost one extra operator
  application and NO linear solve.
* :class:`_Minimiser` -- ``x*`` itself.  Its backward is the adjoint solve,
  ``g^T dx = -s^T (dK x* + dc)`` with ``K s = g`` on ``V``; the adjoint is
  produced by the same Function, so it is differentiable again.

Both backwards are built from the forward ``x*`` re-attached through
:class:`_Minimiser`, so second derivatives (response, force training) and
third derivatives (training against a polarizability or Born charges) are the
exact derivatives of the minimiser, to the solve tolerance.

The operator's explicit dependence on theta is taken through a FACTORY: a pure
function ``factory(tensors) -> (v -> K v)`` of the tensors the operator is
built from.  The explicit partial ``d/dT <a, K_T b>`` is evaluated on fresh
leaf copies of ``T`` re-joined to the originals by a zero-valued difference
(``T_eff = T_leaf + (T - T.detach())``), so that ``autograd.grad`` isolates the
explicit dependence while the returned graph still reaches ``T``, ``a`` and
``b`` for the next order.  Every tensor the product reads must be among the
factory's inputs; a tensor captured by closure would silently lose its
gradient, which the regression suite checks against finite differences.

Energy conservation.  Forces are the exact gradient of ``U(x*(theta))`` up to
the solve tolerance: the residual ``r`` enters the force as ``O(|r|)`` and the
energy as ``O(|r|^2)``.  At the default ``rtol = 1e-10`` that is far below the
integrator's own error; it is measured in the regression suite.

Established ingredients, cited: implicit differentiation of an optimisation
layer through its optimality conditions (the adjoint method); conjugate
gradients with a symmetric positive definite preconditioner.  TRACE-native: the
envelope-theorem energy node whose higher derivatives are exact, and the
factory/leaf construction that makes one implementation serve the dense and
the matrix-free operators.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import math

import torch
from torch import Tensor

__all__ = [
    "DEFAULT_SOLVE_MAX_ITER",
    "DEFAULT_SOLVE_RTOL",
    "IndefiniteOperatorError",
    "QuadraticProblem",
    "SolverNotConvergedError",
    "minimise",
]

#: Relative residual ``||P(K x + c)|| / ||P c||`` at which the solve stops.
#: The energy error is second order in it, the force error first order.
DEFAULT_SOLVE_RTOL = 1e-10

#: Iteration ceiling.  Exceeding it raises: a silently truncated "converged"
#: solve would be the fixed-k model under another name.
DEFAULT_SOLVE_MAX_ITER = 1000

#: Residual replacements (true residual recomputed, CG restarted from the
#: current iterate) before a stagnating solve is declared unconverged.
_MAX_RESTARTS = 3

Product = Callable[[Tensor], Tensor]
Factory = Callable[[Sequence[Tensor]], Product]


class SolverNotConvergedError(RuntimeError):
    """The converged solve did not reach its tolerance within ``max_iter``."""


class IndefiniteOperatorError(RuntimeError):
    """CG met a direction of non-positive curvature: U is unbounded below."""


class QuadraticProblem:
    """``min_{x in V} 0.5 x^T K x + c^T x`` with ``K = factory(tensors)``.

    Holds everything the solve needs that is NOT differentiated: the factory,
    the feasibility mask, the preconditioner and the tolerances.  The detached
    operator is built once, lazily, and reused by every solve on this problem
    (the forward solve and every adjoint solve of every backward).

    Args:
        factory: pure ``factory(tensors) -> product``; ``product(v) = K v``.
        tensors: the tensors ``K`` is built from (differentiable inputs).
        mask: ``[n]`` 0/1 vector selecting the feasible coordinates, or
            ``None`` when every coordinate is free.
        precondition: ``r -> M^-1 r`` with ``M`` symmetric positive definite,
            applied to detached vectors.  ``None`` for the identity.
        rtol, max_iter: stopping rule; see the module constants.
        name: what is being solved, for error messages.
    """

    def __init__(
        self,
        factory: Factory,
        tensors: Sequence[Tensor],
        *,
        mask: Optional[Tensor] = None,
        precondition: Optional[Callable[[Tensor], Tensor]] = None,
        rtol: float = DEFAULT_SOLVE_RTOL,
        max_iter: int = DEFAULT_SOLVE_MAX_ITER,
        name: str = "electrostatic solve",
    ) -> None:
        if not 0.0 < float(rtol) < 1.0:
            raise ValueError(f"rtol must lie in (0, 1), got {rtol}")
        if int(max_iter) < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        self.factory = factory
        self.tensors = tuple(tensors)
        self.mask = None if mask is None else mask.detach()
        self.precondition = precondition
        self.rtol = float(rtol)
        self.max_iter = int(max_iter)
        self.name = name
        self._product: Optional[Product] = None
        #: Iterations of the most recent forward solve (diagnostics).
        self.iterations = 0
        self.residual_norm: Optional[Tensor] = None
        self.rhs_norm: Optional[Tensor] = None

    # -- helpers ---------------------------------------------------------------

    def masked(self, vector: Tensor) -> Tensor:
        return vector if self.mask is None else vector * self.mask

    def detached_product(self) -> Product:
        if self._product is None:
            with torch.no_grad():
                self._product = self.factory(tuple(t.detach() for t in self.tensors))
        return self._product

    def solve(self, rhs: Tensor, x0: Optional[Tensor] = None) -> Tuple[Tensor, int, Tensor, Tensor]:
        """``K x = rhs`` on ``V`` by preconditioned projected CG, no autograd.

        Returns ``(x, iterations, ||r||, ||rhs||)`` with ``r`` the TRUE
        residual (recomputed, not the recurrence's).
        """
        with torch.no_grad():
            return _pcg(
                self.detached_product(),
                self.masked(rhs.detach()),
                self.mask,
                self.precondition,
                None if x0 is None else self.masked(x0.detach()),
                self.rtol,
                self.max_iter,
                self.name,
            )


def _pcg(
    product: Product,
    rhs: Tensor,
    mask: Optional[Tensor],
    precondition: Optional[Callable[[Tensor], Tensor]],
    x0: Optional[Tensor],
    rtol: float,
    max_iter: int,
    name: str,
) -> Tuple[Tensor, int, Tensor, Tensor]:
    """Projected preconditioned CG with residual replacement.

    ``V`` is the range of the diagonal 0/1 ``mask``; every vector the
    recurrence forms is kept inside it, and the preconditioner is applied as
    ``mask M^-1 mask`` -- symmetric positive definite on ``V`` -- so this is
    ordinary PCG on the restriction of ``K`` to ``V``.
    """

    def keep(vector: Tensor) -> Tensor:
        return vector if mask is None else vector * mask

    def apply_preconditioner(vector: Tensor) -> Tensor:
        return keep(vector if precondition is None else precondition(vector))

    rhs_norm = torch.linalg.vector_norm(rhs)
    zero = torch.zeros_like(rhs)
    if float(rhs_norm) == 0.0:
        return zero, 0, rhs_norm, rhs_norm
    target = rtol * float(rhs_norm)
    x = zero if x0 is None else x0.clone()
    iterations = 0
    residual_norm = rhs_norm
    for _ in range(_MAX_RESTARTS + 1):
        residual = rhs - keep(product(x)) if iterations or x0 is not None else rhs.clone()
        residual_norm = torch.linalg.vector_norm(residual)
        if float(residual_norm) <= target:
            return x, iterations, residual_norm, rhs_norm
        z = apply_preconditioner(residual)
        direction = z.clone()
        rz = residual @ z
        while iterations < max_iter:
            curvature_vector = keep(product(direction))
            curvature = direction @ curvature_vector
            step = rz / curvature
            x_next = x + step * direction
            residual_next = residual - step * curvature_vector
            residual_norm_next = torch.linalg.vector_norm(residual_next)
            # ONE host read per iteration for the three decisions below (finite
            # curvature, positive curvature, convergence) instead of three: each
            # read stalls a GPU until its queue drains.  The update above is only
            # kept if the curvature checks pass, so the result and the errors are
            # exactly those of checking first.
            curvature_value, residual_value = torch.stack(
                [curvature.detach(), residual_norm_next.detach()]
            ).tolist()
            if not math.isfinite(curvature_value):
                raise FloatingPointError(
                    f"{name}: non-finite curvature p^T K p = {curvature_value} at "
                    f"iteration {iterations}: the operator, the linear term or the "
                    "preconditioner produced NaN/inf. This is a numerical defect in "
                    "the inputs, not an indefinite energy."
                )
            if not curvature_value > 0.0:
                raise IndefiniteOperatorError(
                    f"{name}: conjugate gradients met a direction with curvature "
                    f"p^T K p = {curvature_value:.3e} <= 0 at iteration {iterations}. "
                    "The quadratic form is not positive definite on the feasible "
                    "set, so the energy is UNBOUNDED BELOW and there is no minimum "
                    "to report. Under the Thole kernel this is possible for three "
                    "or more close sites even with thole_a < 1 (the thole_a < 1 "
                    "bound is a PAIR bound; SPEC 1.6), and its charge-dipole "
                    "coupling needs more on-site hardness than the tight floor: "
                    "ElectrostaticHeads(hardness_margin='legacy') is NECESSARY with "
                    "Thole but not sufficient -- nothing bounds Thole at close "
                    "contact. The Gaussian kernel (multipole_kernel='gaussian') is "
                    "positive definite by theorem at the default floor for every "
                    "geometry (SPEC 12, 14, 17.3). Check also kappa > 0."
                )
            x, residual, residual_norm = x_next, residual_next, residual_norm_next
            iterations += 1
            if residual_value <= target:
                break
            z = apply_preconditioner(residual)
            rz_new = residual @ z
            direction = z + (rz_new / rz) * direction
            rz = rz_new
        # Residual replacement: the recurrence residual drifts from the true one
        # by rounding; only the true one may certify convergence.
        true_residual = rhs - keep(product(x))
        residual_norm = torch.linalg.vector_norm(true_residual)
        if float(residual_norm) <= target:
            return x, iterations, residual_norm, rhs_norm
        if iterations >= max_iter:
            break
    raise SolverNotConvergedError(
        f"{name}: relative residual {float(residual_norm) / float(rhs_norm):.3e} after "
        f"{iterations} iterations, above the tolerance {rtol:.1e}. The converged "
        "solve refuses to return a truncated iterate as if it were the minimum. "
        "Raise ElectrostaticConfig.solve_max_iter, loosen solve_rtol, or check the "
        "conditioning of K (very small kappa or on-site stiffness)."
    )


# ---------------------------------------------------------------------------
# The explicit partial derivative d/dT <a, K_T b>
# ---------------------------------------------------------------------------


def _explicit_partial(
    factory: Factory,
    tensors: Sequence[Tensor],
    scalar: Callable[[Product], Tensor],
    create_graph: bool,
) -> List[Optional[Tensor]]:
    """``d scalar(K_T) / dT`` with the moments inside ``scalar`` held fixed.

    ``scalar`` reads the operator through the product it is handed and closes
    over the (tracked) moments.  The derivative is taken with respect to fresh
    leaves, so ``autograd.grad`` cannot wander into the moments' own graph
    (which leads back to ``T`` through the solve); with ``create_graph`` each
    leaf is re-joined to its original by a zero-valued difference, so the
    returned gradient is a differentiable function of ``T`` and of the moments.
    """
    differentiable = [bool(t.requires_grad) for t in tensors]
    if not any(differentiable):
        return [None] * len(tensors)
    with torch.enable_grad():
        leaves: List[Tensor] = []
        effective: List[Tensor] = []
        for tensor, needs in zip(tensors, differentiable):
            if not needs:
                effective.append(tensor.detach())
                continue
            leaf = tensor.detach().requires_grad_(True)
            leaves.append(leaf)
            effective.append(leaf + (tensor - tensor.detach()) if create_graph else leaf)
        value = scalar(factory(effective))
        grads = torch.autograd.grad(
            value, leaves, create_graph=create_graph, allow_unused=True
        )
    result: List[Optional[Tensor]] = []
    position = 0
    for needs in differentiable:
        if needs:
            result.append(grads[position])
            position += 1
        else:
            result.append(None)
    return result


# ---------------------------------------------------------------------------
# The two autograd nodes
# ---------------------------------------------------------------------------


class _Minimiser(torch.autograd.Function):
    """``x* = argmin_{x in V} 0.5 x^T K x + c^T x``, value supplied, derivatives exact.

    Inputs ``(problem, x_value, c, *tensors)``; ``x_value`` is the detached
    solution (the forward does no work beyond returning it).  Backward, for a
    cotangent ``g``: ``s = S g`` (``S`` the inverse of ``K`` on ``V``), then

        grad_c = -s ,   grad_T = -d/dT <s, K_T x*>

    and ``s`` is itself produced by this Function, so the backward is
    differentiable to any order.
    """

    @staticmethod
    def forward(ctx, problem: QuadraticProblem, x_value: Tensor, c: Tensor, *tensors: Tensor):
        out = x_value.detach().clone()
        ctx.problem = problem
        ctx.save_for_backward(out, *tensors)
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        problem: QuadraticProblem = ctx.problem
        out, *tensors = ctx.saved_tensors
        n_tensors = len(tensors)
        if grad_output is None:
            return (None, None, None) + (None,) * n_tensors
        create_graph = torch.is_grad_enabled()
        g = problem.masked(grad_output)
        s_value, _, _, _ = problem.solve(g)
        s = _Minimiser.apply(problem, s_value, -g, *tensors)
        grad_c = -s
        grads = _explicit_partial(
            problem.factory,
            tensors,
            lambda product: -(s * product(out)).sum(),
            create_graph,
        )
        return (None, None, grad_c, *grads)


class _MinimumEnergy(torch.autograd.Function):
    """``E = min_{x in V} 0.5 x^T K x + c^T x``, backward by the envelope theorem.

    Inputs ``(problem, x_value, c, *tensors)``.  Backward, for a scalar
    cotangent ``gE``: ``grad_c = gE x*``, ``grad_T = gE d/dT <x*, K_T x*>/2``,
    with ``x*`` re-attached through :class:`_Minimiser` so the result is exact
    at every higher order too.  No linear solve at first order.
    """

    @staticmethod
    def forward(ctx, problem: QuadraticProblem, x_value: Tensor, c: Tensor, *tensors: Tensor):
        x = x_value.detach()
        with torch.no_grad():
            value = 0.5 * (x * problem.detached_product()(x)).sum() + (c.detach() * x).sum()
        ctx.problem = problem
        ctx.save_for_backward(x, c, *tensors)
        return value

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        problem: QuadraticProblem = ctx.problem
        x_value, c, *tensors = ctx.saved_tensors
        n_tensors = len(tensors)
        if grad_output is None:
            return (None, None, None) + (None,) * n_tensors
        create_graph = torch.is_grad_enabled()
        x = _Minimiser.apply(problem, x_value, c, *tensors)
        grad_c = grad_output * x
        grads = _explicit_partial(
            problem.factory,
            tensors,
            lambda product: 0.5 * (x * product(x)).sum(),
            create_graph,
        )
        grads = [None if grad is None else grad_output * grad for grad in grads]
        return (None, None, grad_c, *grads)


def minimise(
    problem: QuadraticProblem, c: Tensor, x0: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """Solve and attach: ``(x*, U(x*))`` with exact derivatives of every order.

    ``U(x*) = 0.5 x*^T K x* + c^T x*`` (no constant term).  Records the
    iteration count and the true residual on ``problem`` for diagnostics.
    """
    x_value, iterations, residual_norm, rhs_norm = problem.solve(-c, x0)
    problem.iterations = iterations
    problem.residual_norm = residual_norm
    problem.rhs_norm = rhs_norm
    x = _Minimiser.apply(problem, x_value, c, *problem.tensors)
    energy = _MinimumEnergy.apply(problem, x_value, c, *problem.tensors)
    return x, energy
