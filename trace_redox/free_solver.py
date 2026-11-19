"""The free-charge problem of SPEC 25, solved exactly through its concave dual.

Primal (convex; ``W`` positive definite, ``Phi_u`` from :mod:`.ensemble`):

    P = min_c  sum_u Phi_u(c_u) + b^T c + 1/2 c^T W c      s.t.  sum_u c_u = Q

Dual (Fenchel; ``Phi_u^* = -Omega_u``, ``1/2 c^T W c = max_z (z^T W c - 1/2 z^T W z)``):

    D(z, mu) = -1/2 z^T W z + mu Q + sum_u Omega_u(nu_u),   nu = mu 1 - b - W z

* ``D`` is concave and smooth on all of ``R^{U+1}``: no domain, no barrier,
  no inverse of ``W`` and no division by a susceptibility.
* Stationarity: ``grad_z D = W (N(nu) - z) = 0``, ``dD/dmu = Q - sum_u N_u = 0``.
  Hence ``z* = c*``, charge is conserved, and ``nu* = Phi'(c*)`` -- the primal
  KKT condition ``Phi'(c) + b + W c = mu``.  ``mu*`` is the Fermi level
  (electrochemical potential of the free charge).
* Strong duality (Slater: an interior feasible ``c`` exists whenever
  ``U min n < Q < U max n``): ``P* = D*``; the gap is reported.
* ``-Hess D = [[W + W X W, -W x], [-x^T W, sum x]]``, ``X = diag(x)``,
  ``x_u = X_u(nu_u)``.  It is positive definite: for ``(a, beta)``,
  ``a^T W a + sum_u x_u ((W a)_u - beta)^2 > 0`` unless ``a = 0`` and then
  ``beta^2 sum x > 0`` (every state has non-zero weight at ``kT > 0``).

Newton's method with Armijo backtracking on ``D`` converges globally (concave
objective) and quadratically near the optimum.  The iterations run detached;
:func:`differentiable_solution` then takes ONE Newton step with the autograd
graph of ``(b, W, Ehat)`` from the converged point.  Its value equals the
optimum (the step is ``O(|grad D|)``, zero at convergence), and its derivative
is ``-H^{-1} d(grad D)/dtheta`` -- the implicit-function-theorem derivative of
the optimiser.  Because ``D`` is stationary there, first AND second
derivatives of the returned energy are exact (forces, and the parameter
gradient of a force loss), the same construction as the converged solve of
trace_electrostatics SPEC 17.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from . import ensemble


class FreeChargeNotConvergedError(RuntimeError):
    """The dual Newton iteration did not reach the requested tolerance."""


@dataclasses.dataclass
class FreeChargeSolution:
    """Optimum of the free-charge problem (tensors carry the graph when requested)."""

    z: Tensor            #: [U] dual variable = free charge at the optimum (e)
    mu_e: Tensor         #: [] Fermi level / electrochemical potential (eV)
    nu: Tensor           #: [U] unit potentials nu_u = mu_e - b_u - (W z)_u (eV)
    charge: Tensor       #: [U] free charges c_u = N_u(nu_u) (e)
    populations: Tensor  #: [U, S] state populations p_un
    energy: Tensor       #: [] optimal value P* = D* (eV)
    iterations: int
    residual_charge: float   #: max_u |(W (N - z))_u| (eV): the KKT residual
    residual_total: float    #: |Q - sum_u N_u| (e)
    duality_gap: float       #: P(c) - D(z, mu) (eV), >= 0, ~ 0 at convergence
    expectation: Tensor | None = None   #: [D] ensemble average of an observable of the configurations (SPEC 38)


def dual_value(z: Tensor, mu: Tensor, b: Tensor, W: Tensor, e_hat: Tensor, states: Tensor,
               Q: float, kT: float) -> Tensor:
    """``D(z, mu)``."""
    nu = mu - b - W @ z
    return -0.5 * (z @ (W @ z)) + mu * Q + ensemble.grand_potential(e_hat, states, nu, kT).sum()


def primal_value(c: Tensor, nu: Tensor, b: Tensor, W: Tensor, e_hat: Tensor, states: Tensor,
                 kT: float) -> Tensor:
    """``P(c)`` with ``c = N(nu)`` (``Phi(c) = nu c + Omega(nu)``; no inversion)."""
    phi = nu * c + ensemble.grand_potential(e_hat, states, nu, kT)
    return phi.sum() + b @ c + 0.5 * (c @ (W @ c))


def schur_floor(n_units: int, states: Tensor, kT: float) -> float:
    """Schur curvature ``s`` (e/eV) below which the ``mu`` direction is numerically flat.

    ``g_mu = Q - sum_u N_u`` carries round-off ``~ eps U max|n|``; a Newton step
    ``g_mu / s`` must stay well below ``kT`` for its value to mean anything, so
    ``s`` must exceed ``1e4 eps U max|n| / kT`` (``s`` is then also far above the
    underflow of a saturated unit's susceptibility).  Below it every unit is
    saturated, ``N`` depends on ``mu`` by less than ``~1e-12 e``, ``mu`` is held
    fixed (re-bracketed by bisection if the total charge is off), and the
    dropped coupling changes second derivatives by ``O(s)``.
    """
    eps = torch.finfo(states.dtype).eps
    return max(1e-14, 1e4 * eps * n_units * float(states.abs().max()) / kT)


def _gradient(z, mu, b, W, e_hat, states, Q, kT):
    """``(grad_z D, dD/dmu, N, x)`` at ``(z, mu)``."""
    nu = mu - b - W @ z
    N = ensemble.mean_charge(e_hat, states, nu, kT)
    x = ensemble.susceptibility(e_hat, states, nu, kT)
    return W @ (N - z), Q - N.sum(), N, x


def _newton_step(g_z, g_mu, W, x, floor):
    """Solve ``-Hess D [dz; dmu] = [g_z; g_mu]`` by the Schur complement on ``mu``.

    ``-Hess D = [[A, -W x], [-x^T W, 1^T x]]`` with ``A = W + W X W`` positive
    definite for any ``x >= 0``.  With ``a = A^-1 g_z``, ``v = A^-1 W x`` and
    ``s = 1^T x - (W x)^T v`` (``> 0`` for ``x > 0``):
    ``dmu = (g_mu + (W x)^T a) / s``, ``dz = a + v dmu``.  Returns
    ``(dz, dmu, flat)``; ``flat`` (``s <= floor``, :func:`schur_floor`) means
    ``dmu`` is set to 0.
    """
    A = W + W @ (x.unsqueeze(1) * W)
    A = 0.5 * (A + A.transpose(0, 1))
    wx = W @ x
    factor, info = torch.linalg.cholesky_ex(A.detach())
    if int(info) == 0:
        factor = torch.linalg.cholesky(A) if A.requires_grad else factor
        a = torch.cholesky_solve(g_z.unsqueeze(1), factor).squeeze(1)
        v = torch.cholesky_solve(wx.unsqueeze(1), factor).squeeze(1)
    else:
        # W numerically singular (e.g. free Gaussians far wider than a crystal's
        # period: W_uu ~ 1e-20): z is undetermined along null(W), where D does
        # not depend on it; the pseudo-inverse leaves that component alone.
        evals, evecs = torch.linalg.eigh(A)
        keep = evals > 1e-12 * float(evals.detach().abs().max())
        inv = torch.where(keep, 1.0 / torch.where(keep, evals, torch.ones_like(evals)), torch.zeros_like(evals))
        pinv = evecs @ (inv.unsqueeze(1) * evecs.transpose(0, 1))
        a, v = pinv @ g_z, pinv @ wx
    s = x.sum() - wx @ v
    if float(s) <= floor:
        return a, torch.zeros_like(s), True
    dmu = (g_mu + wx @ a) / s
    return a + v * dmu, dmu, False


def _bracket_mu(z, b, W, e_hat, states, Q, kT):
    """``mu`` with ``sum_u N_u(mu - b_u - (W z)_u) = Q`` by bisection (``N`` is monotone in ``mu``)."""
    shift = b + W @ z
    finite = e_hat[torch.isfinite(e_hat)]                 # +inf marks an absent state
    span = float(finite.abs().max() + shift.abs().max()) + 60.0 * kT + 1.0
    lo, hi = -span, span

    def total(m: float) -> float:
        return float(ensemble.mean_charge(e_hat, states, m - shift, kT).sum())

    while total(lo) > Q:
        lo *= 2.0
    while total(hi) < Q:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < Q:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 4.0 * torch.finfo(b.dtype).eps * max(1.0, abs(mid)):
            break
    return torch.tensor(0.5 * (lo + hi), dtype=b.dtype, device=b.device)


def solve_free_charge(
    b: Tensor,
    W: Tensor,
    e_hat: Tensor,
    states: Tensor,
    Q: float,
    kT: float,
    *,
    tol: float = 1e-11,
    max_iter: int = 200,
) -> tuple[Tensor, Tensor, int, float, float]:
    """Maximise ``D`` (detached).  Returns ``(z*, mu*, iterations, r_charge, r_total)``.

    Refuses an infeasible total charge (outside ``(U min n, U max n)``) and a
    run that does not converge -- a truncated iterate is never returned as if
    it were the optimum.
    """
    if kT <= 0.0:
        raise ValueError(f"kT must be > 0 (SPEC 25.3 uses the Mermin ensemble), got {kT}")
    n_units = int(b.shape[0])
    present = torch.isfinite(e_hat)                       # +inf marks an absent state
    lo = float(torch.where(present, states, torch.full_like(e_hat, float("inf"))).min(dim=1).values.sum())
    hi = float(torch.where(present, states, torch.full_like(e_hat, float("-inf"))).max(dim=1).values.sum())
    if not (lo < Q < hi):
        raise ValueError(
            f"total free charge {Q} is outside the open interval ({lo:g}, {hi:g}) of what these "
            f"{n_units} units can carry"
        )
    with torch.no_grad():
        b, W, e_hat = b.detach(), W.detach(), e_hat.detach()
        floor = schur_floor(n_units, states, kT)
        z = torch.zeros_like(b)
        mu = _bracket_mu(z, b, W, e_hat, states, Q, kT)
        value = dual_value(z, mu, b, W, e_hat, states, Q, kT)
        for iteration in range(1, max_iter + 1):
            g_z, g_mu, N, x = _gradient(z, mu, b, W, e_hat, states, Q, kT)
            # KKT residual: g_z = W (N - z) in eV (z is undetermined along null(W), where
            # nothing depends on it; the physical charges are N), and |Q - sum N| in e.
            r_charge = float(g_z.abs().max())
            r_total = abs(float(g_mu))
            if r_charge < tol and r_total < tol:
                return z, mu, iteration - 1, r_charge, r_total
            dz, dmu, flat = _newton_step(g_z, g_mu, W, x, floor)
            if flat and r_total >= tol:
                # mu is numerically flat and the total charge is off: put mu back
                # on the constraint for this z (exact 1-D root), then continue.
                mu = _bracket_mu(z, b, W, e_hat, states, Q, kT)
                value = dual_value(z, mu, b, W, e_hat, states, Q, kT)
                continue
            slope = float(g_z @ dz + g_mu * dmu)   # Newton decrement^2 (twice the predicted increase)
            if slope <= 0.0:
                raise FreeChargeNotConvergedError(
                    f"non-ascent Newton direction (slope {slope:.3e}) at iteration {iteration}"
                )
            # Pure-Newton phase: once the predicted increase is below the
            # resolution of D itself, an Armijo comparison of two values of D is
            # round-off noise (measured: step halved at slope 9e-21, |D| = 1.4 eV,
            # stalling at |N - z| = 7e-11).  The decrement is then tiny, the
            # iterate is inside the quadratic-convergence region, and the full
            # step is taken (Boyd & Vandenberghe, Convex Optimization, 9.5).
            resolution = 1e4 * torch.finfo(b.dtype).eps * (1.0 + abs(float(value)))
            alpha = 1.0
            if slope > resolution:
                for _ in range(60):
                    z_new, mu_new = z + alpha * dz, mu + alpha * dmu
                    new_value = dual_value(z_new, mu_new, b, W, e_hat, states, Q, kT)
                    if float(new_value) >= float(value) + 1e-4 * alpha * slope:
                        break
                    alpha *= 0.5
                else:
                    raise FreeChargeNotConvergedError(
                        f"line search failed at iteration {iteration} (slope {slope:.3e})"
                    )
            else:
                z_new, mu_new = z + dz, mu + dmu
                new_value = dual_value(z_new, mu_new, b, W, e_hat, states, Q, kT)
            z, mu, value = z_new, mu_new, new_value
        g_z, g_mu, N, _ = _gradient(z, mu, b, W, e_hat, states, Q, kT)
        raise FreeChargeNotConvergedError(
            f"free-charge Newton did not converge in {max_iter} iterations: "
            f"max |W (N - z)| = {float(g_z.abs().max()):.3e} eV, "
            f"|Q - sum N| = {abs(float(g_mu)):.3e} e (tol {tol:.1e})"
        )


def differentiable_solution(
    b: Tensor,
    W: Tensor,
    e_hat: Tensor,
    states: Tensor,
    Q: float,
    kT: float,
    *,
    tol: float = 1e-11,
    max_iter: int = 200,
) -> FreeChargeSolution:
    """Solve, then re-attach the optimum to the graph of ``(b, W, e_hat)``.

    See the module docstring: one Newton step with the graph from the
    detached optimum gives the exact implicit derivatives of every quantity
    returned, to the order the energy needs.  When the ``mu`` direction is
    numerically flat (every unit saturated, ``s <=`` :func:`schur_floor`) ``mu``
    is held fixed; its coupling to the rest is ``O(s)``.
    """
    z0, mu0, iterations, r_charge, r_total = solve_free_charge(
        b, W, e_hat, states, Q, kT, tol=tol, max_iter=max_iter
    )
    floor = schur_floor(int(b.shape[0]), states, kT)
    # z is fixed only modulo null(W) (W(N - z) = 0 at the optimum); D does not depend on that
    # component, but dD/dW at z = c + n is c c^T/2 - n n^T/2.  Start the step from z = c,
    # the primal charge, whose envelope derivative is the primal one, c c^T/2.  It moves
    # nu by W(c - z0) = the residual: a no-op to O(tol^2) when W is positive definite.
    with torch.no_grad():
        z0 = _gradient(z0, mu0, b, W, e_hat, states, Q, kT)[2]
    g_z, g_mu, _, x = _gradient(z0, mu0, b, W, e_hat, states, Q, kT)
    # The Newton operator is detached; only the residual (g_z, g_mu) carries the graph.
    # With s = s0 - P g(s0; theta), P = (-Hess D)^+ at s0 and g(s0; theta0) = 0:
    # dE/dtheta = D_theta (envelope) and d2E/dtheta2 = D_tt - g_t^T P g_t, the exact
    # Schur complement; every derivative of P would multiply g = 0.  Differentiating P
    # instead runs eigh's backward, whose 1/(lambda_i - lambda_j) is 0/0 at a degenerate
    # spectrum (two identical molecules): NaN gradients for a smooth energy.
    dz, dmu, _ = _newton_step(g_z, g_mu, W.detach(), x.detach(), floor)
    z = z0 + dz
    mu = mu0 + dmu
    nu = mu - b - W @ z
    populations = ensemble.occupations(e_hat, states, nu, kT)
    charge = populations @ states
    energy = dual_value(z, mu, b, W, e_hat, states, Q, kT)
    with torch.no_grad():
        gap = float(primal_value(charge, nu, b, W, e_hat, states, kT) - energy)
    return FreeChargeSolution(
        z=z, mu_e=mu, nu=nu, charge=charge, populations=populations, energy=energy,
        iterations=iterations, residual_charge=r_charge, residual_total=r_total, duality_gap=gap,
    )


# ---------------------------------------------------------------------------
# one constraint per molecule (the exact canonical configuration energies, SPEC §15)
# ---------------------------------------------------------------------------
#
# Primal for a configuration N (one integer total per molecule F):
#
#     E(N) = min_c  sum_u Phi_u(c_u) + b^T c + 1/2 c^T W c      s.t.  G c = N
#
# with G [m, U] the 0/1 molecule membership (every unit in exactly one molecule).
# Dual (the same Fenchel steps as above, one multiplier per molecule):
#
#     D(z, mu) = -1/2 z^T W z + mu^T N + sum_u Omega_u(nu_u),   nu = G^T mu - b - W z
#     grad_z D = W (N(nu) - z),   grad_mu D = N - G N(nu)
#     -Hess D  = [[A, -B], [-B^T, C]],  A = W + W X W,  B = W X G^T,  C = diag(G x)
#
# Newton by the Schur complement on mu: a = A^-1 g_z, V = A^-1 B,
# S_mu = C - B^T V (m x m, positive semidefinite), dmu = S_mu^+ (g_mu + B^T a),
# dz = a + V dmu.  A molecule whose units all sit deep in their dead zone has
# S_mu ~ exp(-gap/kT) along its mu: that direction is dropped (its charge is
# pinned at the integer and does not depend on mu_F there), and mu_F is put
# back on its constraint by a 1-D bisection whenever its charge is off.


def _membership(group: Tensor, m: int, dtype, device) -> Tensor:
    G = torch.zeros(m, int(group.shape[0]), dtype=dtype, device=device)
    G[group, torch.arange(int(group.shape[0]), device=device)] = 1.0
    return G


def constrained_dual_value(z, mu, b, W, e_hat, states, G, N_target, kT) -> Tensor:
    nu = G.transpose(0, 1) @ mu - b - W @ z
    return -0.5 * (z @ (W @ z)) + mu @ N_target + ensemble.grand_potential(e_hat, states, nu, kT).sum()


def _constrained_gradient(z, mu, b, W, e_hat, states, G, N_target, kT):
    nu = G.transpose(0, 1) @ mu - b - W @ z
    N = ensemble.mean_charge(e_hat, states, nu, kT)
    x = ensemble.susceptibility(e_hat, states, nu, kT)
    return W @ (N - z), N_target - G @ N, N, x


def _constrained_step(g_z, g_mu, W, x, G, floor):
    """``(dz, dmu, flat)``: Newton step with the flat molecular directions dropped."""
    A = W + W @ (x.unsqueeze(1) * W)
    A = 0.5 * (A + A.transpose(0, 1))
    B = W @ (x.unsqueeze(1) * G.transpose(0, 1))                       # [U, m]
    # The residual g_z and the operator columns B are solved separately, so that V and S
    # depend on (W, x, G) alone: with those detached (differentiable_constrained_solution)
    # no eigh below is ever differentiated.
    factor, info = torch.linalg.cholesky_ex(A.detach())
    if int(info) == 0:
        factor = torch.linalg.cholesky(A) if A.requires_grad else factor
        a = torch.cholesky_solve(g_z.unsqueeze(1), factor).squeeze(1)
        V = torch.cholesky_solve(B, factor)
    else:
        evals, evecs = torch.linalg.eigh(A)
        keep = evals > 1e-12 * float(evals.detach().abs().max())
        inv = torch.where(keep, 1.0 / torch.where(keep, evals, torch.ones_like(evals)), torch.zeros_like(evals))
        pinv = evecs @ (inv.unsqueeze(1) * evecs.transpose(0, 1))
        a, V = pinv @ g_z, pinv @ B
    S = torch.diag(G @ x) - B.transpose(0, 1) @ V
    S = 0.5 * (S + S.transpose(0, 1))
    evals, evecs = torch.linalg.eigh(S)
    keep = evals > floor
    inv = torch.where(keep, 1.0 / torch.where(keep, evals, torch.ones_like(evals)), torch.zeros_like(evals))
    dmu = evecs @ (inv * (evecs.transpose(0, 1) @ (g_mu + B.transpose(0, 1) @ a)))
    flat = bool((~keep).any())
    return a + V @ dmu, dmu, flat


def _bracket_molecules(z, b, W, e_hat, states, G, N_target, kT):
    """``mu_F`` with ``sum_{u in F} N_u(mu_F - b_u - (W z)_u) = N_F`` for every F (independent 1-D roots)."""
    shift = b + W @ z
    finite = e_hat[torch.isfinite(e_hat)]
    span = float(finite.abs().max() + shift.abs().max()) + 60.0 * kT + 1.0
    mu = torch.zeros(int(G.shape[0]), dtype=b.dtype, device=b.device)
    for F in range(int(G.shape[0])):
        members = G[F] > 0
        sub_e, sub_shift, target = e_hat[members], shift[members], float(N_target[F])

        def total(m: float) -> float:
            return float(ensemble.mean_charge(sub_e, states, m - sub_shift, kT).sum())

        lo, hi = -span, span
        while total(lo) > target:
            lo *= 2.0
        while total(hi) < target:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if total(mid) < target:
                lo = mid
            else:
                hi = mid
            if hi - lo <= 4.0 * torch.finfo(b.dtype).eps * max(1.0, abs(mid)):
                break
        mu[F] = 0.5 * (lo + hi)
    return mu


def solve_constrained(b, W, e_hat, states, group, N_target, kT, *, tol=1e-11, max_iter=200):
    """Maximise the multi-constraint dual (detached).  Returns ``(z*, mu*, iterations)``.

    Every molecule's target must lie strictly inside what its units can carry
    (a molecule at a capacity bound is fixed exactly by the caller).
    """
    if kT <= 0.0:
        raise ValueError(f"kT must be > 0, got {kT}")
    m = int(N_target.shape[0])
    with torch.no_grad():
        b, W, e_hat, N_target = b.detach(), W.detach(), e_hat.detach(), N_target.detach().to(b.dtype)
        G = _membership(group, m, b.dtype, b.device)
        present = torch.isfinite(e_hat)
        lo = G @ torch.where(present, states, torch.full_like(e_hat, float("inf"))).min(dim=1).values
        hi = G @ torch.where(present, states, torch.full_like(e_hat, float("-inf"))).max(dim=1).values
        if bool(((N_target <= lo) | (N_target >= hi)).any()):
            raise ValueError("a molecular charge target is not strictly inside its capacity; fix that molecule exactly")
        floor = schur_floor(int(b.shape[0]), states, kT)
        z = torch.zeros_like(b)
        mu = _bracket_molecules(z, b, W, e_hat, states, G, N_target, kT)
        value = constrained_dual_value(z, mu, b, W, e_hat, states, G, N_target, kT)
        for iteration in range(1, max_iter + 1):
            g_z, g_mu, N, x = _constrained_gradient(z, mu, b, W, e_hat, states, G, N_target, kT)
            r_charge, r_total = float(g_z.abs().max()), float(g_mu.abs().max())
            if r_charge < tol and r_total < tol:
                return z, mu, iteration - 1
            dz, dmu, flat = _constrained_step(g_z, g_mu, W, x, G, floor)
            slope = float(g_z @ dz + g_mu @ dmu)
            if slope <= 0.0:
                if flat and r_total >= tol:
                    # the residual lies along a dropped (dead-zone) direction: put every
                    # mu_F back on its own constraint for this z (exact 1-D roots)
                    mu = _bracket_molecules(z, b, W, e_hat, states, G, N_target, kT)
                    value = constrained_dual_value(z, mu, b, W, e_hat, states, G, N_target, kT)
                    continue
                raise FreeChargeNotConvergedError(f"non-ascent Newton direction (slope {slope:.3e}) at iteration {iteration}")
            resolution = 1e4 * torch.finfo(b.dtype).eps * (1.0 + abs(float(value)))
            alpha = 1.0
            if slope > resolution:
                for _ in range(60):
                    z_new, mu_new = z + alpha * dz, mu + alpha * dmu
                    new_value = constrained_dual_value(z_new, mu_new, b, W, e_hat, states, G, N_target, kT)
                    if float(new_value) >= float(value) + 1e-4 * alpha * slope:
                        break
                    alpha *= 0.5
                else:
                    raise FreeChargeNotConvergedError(f"line search failed at iteration {iteration} (slope {slope:.3e})")
            else:
                z_new, mu_new = z + dz, mu + dmu
                new_value = constrained_dual_value(z_new, mu_new, b, W, e_hat, states, G, N_target, kT)
            z, mu, value = z_new, mu_new, new_value
            if flat:
                _, g_check, _, _ = _constrained_gradient(z, mu, b, W, e_hat, states, G, N_target, kT)
                if float(g_check.abs().max()) >= tol:
                    mu = _bracket_molecules(z, b, W, e_hat, states, G, N_target, kT)
                    value = constrained_dual_value(z, mu, b, W, e_hat, states, G, N_target, kT)
        g_z, g_mu, _, _ = _constrained_gradient(z, mu, b, W, e_hat, states, G, N_target, kT)
        raise FreeChargeNotConvergedError(
            f"constrained Newton did not converge in {max_iter} iterations: max |W (N - z)| = "
            f"{float(g_z.abs().max()):.3e} eV, max |N_F - G N| = {float(g_mu.abs().max()):.3e} e"
        )


def differentiable_constrained_solution(b, W, e_hat, states, group, N_target, kT, *, tol=1e-11, max_iter=200):
    """Solve, then one graph-attached Newton step: exact first and second derivatives of the
    stationary value E(N) (module docstring), with one multiplier per molecule."""
    N_target = N_target.to(b.dtype)
    z0, mu0, iterations = solve_constrained(b, W, e_hat, states, group, N_target, kT, tol=tol, max_iter=max_iter)
    m = int(N_target.shape[0])
    G = _membership(group, m, b.dtype, b.device)
    floor = schur_floor(int(b.shape[0]), states, kT)
    with torch.no_grad():                    # z = c, not c + n, n in null(W): see differentiable_solution
        z0 = _constrained_gradient(z0, mu0, b, W, e_hat, states, G, N_target, kT)[2]
    g_z, g_mu, _, x = _constrained_gradient(z0, mu0, b, W, e_hat, states, G, N_target, kT)
    # Detached operator, graph on the residual only: exact to second order, and no eigh
    # backward through a degenerate Schur complement (see differentiable_solution).
    dz, dmu, _ = _constrained_step(g_z, g_mu, W.detach(), x.detach(), G, floor)
    z, mu = z0 + dz, mu0 + dmu
    nu = G.transpose(0, 1) @ mu - b - W @ z
    populations = ensemble.occupations(e_hat, states, nu, kT)
    charge = populations @ states
    energy = constrained_dual_value(z, mu, b, W, e_hat, states, G, N_target, kT)
    with torch.no_grad():
        gap = float(primal_value(charge, nu, b, W, e_hat, states, kT) - energy)
    return FreeChargeSolution(
        z=z, mu_e=mu, nu=nu, charge=charge, populations=populations, energy=energy, iterations=iterations,
        residual_charge=float(g_z.detach().abs().max()), residual_total=float(g_mu.detach().abs().max()),
        duality_gap=gap,
    )
