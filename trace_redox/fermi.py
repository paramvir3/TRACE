"""SPEC 38, stage 1: the Fermi band's functional (C4, 2026-11-10).

SPEC 20's band holds at most two electrons in the lowest orbital of h = diag(eps) - tau.  A replicated doped cell has
more excess electrons than that (experiments/benchmarks_4g/README.md, R17), and its lowest levels are degenerate by
symmetry.  This module gives the band any number N of electrons:

    F_b(N) = A(N_up) + A(N_dn) + 1/2 sum_{u != v} V_uv [rho_u rho_v - sum_s (P^s_uv)^2] + sum_u U_u rho^up_u rho^dn_u,
    A(n)   = mu n - kT tr ln(1 + exp(-(h - mu)/kT)),   P = [1 + exp((h - mu)/kT)]^-1,   tr P = n,
    N_up = ceil(N/2),  N_dn = floor(N/2),   rho^s = diag P^s,   rho = rho^up + rho^dn,   U = diag V.

A(n) is the free energy of n non-interacting electrons at the model's kT with Fermi-Dirac occupations (ESTABLISHED:
Mermin, Phys. Rev. 137, A1441 (1965)); the interaction is its first-order (Hartree-Fock) average in that state, with
the orbitals of h kept (as in SPEC 20).  For a gap far above kT, F_b(1) = lambda_1 and F_b(2) = 2 lambda_1 + J_11:
SPEC 20, up to exp(-gap / 2 kT).

Derivatives.  An eigendecomposition's derivative divides by level differences, and the levels of a replicated cell are
degenerate.  P is therefore a polynomial in h (ESTABLISHED: the Chebyshev expansion of the Fermi operator, Goedecker
and Colombo, Phys. Rev. Lett. 73, 122 (1994)): with the spectrum mapped into [-1, 1] and x_j Chebyshev-Gauss nodes,

    P = sum_k c_k(mu) T_k(h~),   c_k(mu) = sum_j C_kj f(x_j; mu),   tr P = sum_j w_j f(x_j; mu),   w_j = sum_k C_kj tr T_k(h~).

The eigenvalues enter only detached: for the interval, for the starting value of mu and for the value of A.  Two
constructions carry the derivatives (NEW-TO-TRACE as constructions):
  * mu(h) follows two Newton steps on the count tr P = n from its detached root, which makes its derivatives exact to
    third order (the device of ``band.lowest_eigenpair``).  Where the count does not respond to mu (a gap far above
    kT: sum f (1 - f) < 1e-12) mu is a constant: everything it multiplies is smaller than that number.
  * A(n) is given its exact gradient, dA/dh = P (A is stationary in mu at fixed n), so the second derivative of the
    energy is the first of the polynomial.  Each derivative of the polynomial costs about a factor of its order in
    round-off, because the coefficients carry an absolute error near 1e-17 at every k.
"""
import math
from typing import Dict, NamedTuple, Sequence

import torch

SPECTRUM_MARGIN = 5.0e-2          # relative widening of the detached spectral interval: T_k'' grows as k^4 at its ends
ORDER_PAD = 8
MU_RESPONSE_FLOOR = 1.0e-12       # sum_k f_k (1 - f_k) below which mu is held constant


class FermiOperator(NamedTuple):
    projector: Dict[int, torch.Tensor]       # n -> P [B, B], tr P = n
    free_energy: Dict[int, torch.Tensor]     # n -> A(n) = sum f lambda - kT S
    mu: Dict[int, torch.Tensor]              # n -> chemical potential (-inf / +inf for the empty / full channel)
    order: int                               # number of Chebyshev terms


def _fermi(y: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 - torch.tanh(0.5 * y))


def chebyshev_order(half_width: float, kT: float, eps: float) -> int:
    """Terms needed for the Fermi function on an interval of the given half-width: its poles nearest the real axis lie
    at mu +- i pi kT, on or outside the Bernstein ellipse of parameter rho = d + sqrt(1 + d^2), d = pi kT / half_width,
    and the expansion converges as rho^-k."""
    return int(math.ceil(math.log(1.0 / eps) / math.asinh(math.pi * kT / half_width))) + ORDER_PAD


def _detached_mu(lam: torch.Tensor, fillings: torch.Tensor, kT: float) -> torch.Tensor:
    """mu with sum_k f((lam_k - mu)/kT) = n for every n in ``fillings`` (0 < n < B), by bisection."""
    pad = kT * (math.log(float(lam.numel())) + 40.0)
    lo = (lam[0] - pad).expand(fillings.shape).clone()
    hi = (lam[-1] + pad).expand(fillings.shape).clone()
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        low = _fermi((lam[None, :] - mid[:, None]) / kT).sum(-1) < fillings
        lo, hi = torch.where(low, mid, lo), torch.where(low, hi, mid)
    return 0.5 * (lo + hi)


class _ValueWithGradient(torch.autograd.Function):
    """A scalar function of h given by its value and by its gradient: out = value, d out / d h = gradient.

    A(n) has the gradient P exactly (it is stationary in mu at fixed n).  Differentiating its own Chebyshev series
    instead costs a factor of the order per derivative in round-off (measured: 2e-10 in the first and 6e-8 in the second
    derivative at order 1300); P is that derivative taken once analytically, and higher derivatives flow through P."""

    @staticmethod
    def forward(ctx, h, value, gradient):
        ctx.save_for_backward(gradient)
        return value.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (gradient,) = ctx.saved_tensors
        return grad_output * gradient, None, None


def fermi_operator(h: torch.Tensor, fillings: Sequence[int], kT: float) -> FermiOperator:
    """Fermi-Dirac projectors of the symmetric matrix ``h`` [B, B] at fixed electron numbers.

    For every integer n in ``fillings`` (0 <= n <= B): P with tr P = n, the free energy A(n) and mu.  Differentiable
    in h to any order, at degenerate spectra too (module docstring).  When no derivative with respect to h is asked for
    (``h`` does not require grad, or grad is disabled) P is taken from the eigenvectors directly; the two routes agree
    to the expansion's round-off."""
    if h.dtype != torch.float64:
        raise TypeError("fermi_operator needs float64")
    B = int(h.shape[0])
    wanted = sorted({int(n) for n in fillings})
    if not wanted or wanted[0] < 0 or wanted[-1] > B:
        raise ValueError(f"a spin channel of {B} sites holds 0 .. {B} electrons; asked for {wanted}")
    h = 0.5 * (h + h.T)
    differentiable = torch.is_grad_enabled() and h.requires_grad
    lam, vec = torch.linalg.eigh(h.detach())
    centre = 0.5 * float(lam[-1] + lam[0])
    half = max(0.5 * float(lam[-1] - lam[0]) * (1.0 + SPECTRUM_MARGIN), float(kT))
    order = chebyshev_order(half, float(kT), torch.finfo(h.dtype).eps)
    eye = torch.eye(B, dtype=h.dtype, device=h.device)
    projector = {n: torch.zeros_like(h) if n == 0 else eye for n in wanted if n in (0, B)}
    free = {n: h.new_zeros(()) if n == 0 else torch.trace(h) for n in projector}
    mus = {n: h.new_full((), -math.inf if n == 0 else math.inf) for n in projector}
    inner = [n for n in wanted if 0 < n < B]
    if not inner:
        return FermiOperator(projector, free, mus, order)
    target = torch.tensor(inner, dtype=h.dtype, device=h.device)
    mu = _detached_mu(lam, target, float(kT))
    level = (lam[None, :] - mu[:, None]) / kT
    # ln(1 + e^-y) by logaddexp: torch's softplus is linear above its threshold (20) and drops e^-20 = 2e-9 there
    value = mu * target - kT * torch.logaddexp(torch.zeros_like(level), -level).sum(-1)
    if not differentiable:
        P = torch.einsum("uk,nk,vk->nuv", vec, _fermi(level), vec)
        for i, n in enumerate(inner):
            projector[n], free[n], mus[n] = P[i], value[i], mu[i]
        return FermiOperator(projector, free, mus, order)

    nodes = 2 * order
    theta = math.pi * (torch.arange(nodes, dtype=h.dtype, device=h.device) + 0.5) / nodes
    C = (2.0 / nodes) * torch.cos(torch.arange(order, dtype=h.dtype, device=h.device)[:, None] * theta[None, :])
    C[0] = 0.5 * C[0]
    x = centre + half * torch.cos(theta)                                       # the nodes, in eV
    scaled = (h - centre * eye) / half
    T = [eye, scaled]
    for _ in range(order - 2):
        T.append(2.0 * (scaled @ T[-1]) - T[-2])
    T = T[:order]
    w = torch.stack([torch.trace(t) for t in T]) @ C                           # tr G(h) = sum_j w_j G(x_j)

    def count(m):
        f = _fermi((x[None, :] - m[:, None]) / kT)
        return f @ w, (f * (1.0 - f)) @ w / kT

    n0, d0 = count(mu)
    shift = n0.detach()                                                        # the count's round-off at mu
    live = (d0.detach() * kT) > MU_RESPONSE_FLOOR
    one, zero = torch.ones_like(d0), torch.zeros_like(d0)
    mu = mu - torch.where(live, (n0 - shift) / torch.where(live, d0, one), zero)
    n1, d1 = count(mu)
    mu = mu - torch.where(live, (n1 - shift) / torch.where(live, d1, one), zero)

    coefficient = _fermi((x[None, :] - mu[:, None]) / kT) @ C.T                # [fillings, order]
    P = coefficient[:, 0, None, None] * eye
    for k in range(1, order):
        P = P + coefficient[:, k, None, None] * T[k]
    for i, n in enumerate(inner):
        projector[n], free[n], mus[n] = P[i], _ValueWithGradient.apply(h, value[i], P[i]), mu[i]
    return FermiOperator(projector, free, mus, order)


def band_free_energy(h: torch.Tensor, V: torch.Tensor, fillings: Sequence[int], kT: float):
    """F_b(N) [len(fillings)] and the charge shapes rho_N [len(fillings), B] of the band (module docstring).

    ``V`` [B, B] is SPEC 20's interaction between band sites; its diagonal U is the cost of the second electron on a
    site and may be infinite (the site has no such state), which admits at most one electron in the band."""
    B = int(h.shape[0])
    fillings = [int(N) for N in fillings]
    if any(N < 0 or N > 2 * B for N in fillings):
        raise ValueError(f"a band of {B} sites holds 0 .. {2 * B} electrons; asked for {fillings}")
    diagonal = torch.eye(B, dtype=torch.bool, device=V.device)
    U, off = torch.diagonal(V), V.masked_fill(diagonal, 0.0)
    if not bool(torch.isfinite(off).all()):
        raise ValueError("the interaction between different band sites is not finite")
    if any(N // 2 for N in fillings) and not bool(torch.isfinite(U).all()):
        raise ValueError("a band site without a second-electron state (infinite U) admits at most one electron in "
                         f"the band; asked for {fillings}")
    op = fermi_operator(h, [(N + 1) // 2 for N in fillings] + [N // 2 for N in fillings], kT)
    energies, shapes = [], []
    for N in fillings:
        up, dn = (N + 1) // 2, N // 2
        Pu, Pd = op.projector[up], op.projector[dn]
        ru, rd = torch.diagonal(Pu), torch.diagonal(Pd)
        rho = ru + rd
        energy = op.free_energy[up] + op.free_energy[dn] + 0.5 * (rho @ off @ rho - ((Pu * Pu + Pd * Pd) * off).sum())
        if dn:
            energy = energy + (U * ru * rd).sum()
        energies.append(energy)
        shapes.append(rho)
    return torch.stack(energies), torch.stack(shapes)
