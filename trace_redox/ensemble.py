"""The finite-temperature ensemble of one redox unit over its integer charge states.

Notation (SPEC 25.3).  Unit ``u`` has integer charge states ``n`` (electrons
REMOVED: ``n = +1`` a cation, ``n = -1`` an anion) with on-site energies
``Ehat_un`` -- the learned state energies with the unit's own Coulomb
self-energy removed (``Ehat = E - J_uu n^2 / 2``; the functional adds it back).
At an electrochemical potential ``nu`` (conjugate to the free charge ``c``):

    Omega_u(nu) = -kT log sum_n exp(-(Ehat_un - nu n) / kT)          grand potential
    N_u(nu)     = -dOmega/dnu = sum_n p_un n,   p_un = softmax_n(-(Ehat_un - nu n)/kT)
    X_u(nu)     =  dN/dnu    = Var_p(n) / kT  >= 0                    susceptibility
    Phi_u(c)    = sup_nu [ nu c + Omega_u(nu) ]                       Legendre conjugate

* ``Omega_u`` is concave in ``nu`` (minus a log-sum-exp of affine functions), so
  ``Phi_u = (-Omega_u)^*`` is convex in ``c``, with ``Phi_u'(N_u(nu)) = nu`` and
  ``Phi_u'' = 1 / X_u`` (Mermin, Phys. Rev. 137, A1441 (1965)).
* As ``kT -> 0``, ``Phi_u`` tends to the lower convex envelope of the points
  ``(n, Ehat_un)`` -- the ensemble ground state, whose kinks at the integers are
  the derivative discontinuity of Perdew, Parr, Levy, Balduz (PRL 49, 1691
  (1982)); :func:`convex_envelope` evaluates that limit for tests.
* The domain of ``Phi_u`` is the open interval ``(min n, max n)``: the barrier
  keeps every free charge inside its physical range automatically.

Everything is batched over units: ``Ehat`` is ``[U, S]``, ``n`` is ``[S]``.
Numerically stable (log-sum-exp with the maximum removed); no division by a
susceptibility anywhere (the free-charge solver works in the dual, where only
``X`` itself appears).
"""

from __future__ import annotations

import torch
from torch import Tensor


def _logits(e_hat: Tensor, states: Tensor, nu: Tensor, kT: float) -> Tensor:
    """``-(Ehat_un - nu_u n) / kT``, shape ``[U, S]``."""
    return -(e_hat - nu.unsqueeze(-1) * states) / kT


def grand_potential(e_hat: Tensor, states: Tensor, nu: Tensor, kT: float) -> Tensor:
    """``Omega_u(nu_u)``, shape ``[U]`` (eV)."""
    return -kT * torch.logsumexp(_logits(e_hat, states, nu, kT), dim=-1)


def occupations(e_hat: Tensor, states: Tensor, nu: Tensor, kT: float) -> Tensor:
    """State populations ``p_un``, shape ``[U, S]`` (rows sum to 1)."""
    return torch.softmax(_logits(e_hat, states, nu, kT), dim=-1)


def mean_charge(e_hat: Tensor, states: Tensor, nu: Tensor, kT: float) -> Tensor:
    """``N_u(nu) = sum_n p_un n``, shape ``[U]`` (e)."""
    return occupations(e_hat, states, nu, kT) @ states


def susceptibility(e_hat: Tensor, states: Tensor, nu: Tensor, kT: float) -> Tensor:
    """``X_u = dN_u/dnu = Var_p(n) / kT``, shape ``[U]`` (e / eV), never negative."""
    p = occupations(e_hat, states, nu, kT)
    mean = p @ states
    second = p @ (states * states)
    return (second - mean * mean).clamp_min(0.0) / kT


def conjugate_at(e_hat: Tensor, states: Tensor, nu: Tensor, kT: float) -> tuple[Tensor, Tensor]:
    """``(c, Phi(c))`` at ``c = N(nu)``: the Legendre pair without any inversion.

    ``Phi(N(nu)) = nu N(nu) + Omega(nu)`` because ``nu`` is the maximiser of
    ``nu' c + Omega(nu')`` exactly when ``c = N(nu)``.
    """
    c = mean_charge(e_hat, states, nu, kT)
    return c, nu * c + grand_potential(e_hat, states, nu, kT)


def potential_for_charge(
    e_hat: Tensor, states: Tensor, c: Tensor, kT: float, *, tol: float = 1e-13, max_iter: int = 400
) -> Tensor:
    """``nu`` with ``N_u(nu) = c_u`` (the inverse map), by bisection; detached.

    ``N_u`` is strictly increasing from ``min n`` to ``max n``; a target outside
    that open interval has no preimage and is refused.  Used by tests and for
    reporting ``Phi(c)`` at an arbitrary ``c``; the solver never needs it.
    """
    with torch.no_grad():
        lo_n, hi_n = float(states.min()), float(states.max())
        if bool(((c <= lo_n) | (c >= hi_n)).any()):
            raise ValueError(
                f"free charge outside the open interval ({lo_n}, {hi_n}) of the charge states"
            )
        span = float((e_hat.max() - e_hat.min()).abs()) + 50.0 * kT + 1.0
        lo = torch.full_like(c, -span)
        hi = torch.full_like(c, span)
        while bool((mean_charge(e_hat, states, lo, kT) > c).any()):
            lo = lo * 2.0
        while bool((mean_charge(e_hat, states, hi, kT) < c).any()):
            hi = hi * 2.0
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            too_low = mean_charge(e_hat, states, mid, kT) < c
            lo = torch.where(too_low, mid, lo)
            hi = torch.where(too_low, hi, mid)
            if float((hi - lo).max()) < tol:
                break
        return 0.5 * (lo + hi)


def convex_envelope(e_hat: Tensor, states: Tensor, c: Tensor) -> Tensor:
    """``kT -> 0`` limit of ``Phi_u(c)``: the lower convex envelope of ``(n, Ehat_un)``.

    ``max_nu [nu c + min_n (Ehat_n - nu n)]``, evaluated as the minimum over
    pairs of states that bracket ``c`` of the chord between them (the envelope
    of finitely many points is the pointwise minimum of their bracketing
    chords).  Shapes: ``e_hat [U, S]``, ``c [U]``.
    """
    n = states.to(e_hat.dtype)
    best = torch.full_like(c, float("inf"))
    for a in range(n.numel()):
        for b in range(n.numel()):
            na, nb = float(n[a]), float(n[b])
            if not na <= nb:
                continue
            inside = (c >= na) & (c <= nb)
            if na == nb:
                value = e_hat[:, a]
            else:
                weight = (c - na) / (nb - na)
                value = (1.0 - weight) * e_hat[:, a] + weight * e_hat[:, b]
            best = torch.where(inside, torch.minimum(best, value), best)
    return best
