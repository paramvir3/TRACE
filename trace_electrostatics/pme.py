"""Smooth particle-mesh Ewald for the reciprocal part of the lattice operator (SPEC 16).

:class:`~trace_electrostatics.lattice_operator.LatticeOperator` evaluates the
reciprocal half of the periodic multipole operator as a direct sum over the
wave vectors inside the cutoff sphere, ``O(n n_k)`` per product.  This module
evaluates the same functional on a mesh -- smooth particle-mesh Ewald (Essmann,
Perera, Berkowitz, Darden, Lee, Pedersen, J. Chem. Phys. 103, 8577 (1995)),
with the multipoles spread and gathered through derivatives of the B-splines
(Sagui, Pedersen, Darden, J. Chem. Phys. 120, 73 (2004)) -- in
``O(n p^3 + M log M)`` for a mesh of ``M`` points and B-splines of order ``p``.

Conventions -- identical to ``lattice_operator._reciprocal_chunk``
------------------------------------------------------------------
Charge ``q``, dipole ``mu`` and CARTESIAN quadrupole ``Th`` (the caller maps the
five ``2e`` coefficients through ``quadrupole_cartesian_basis``) enter

    S(k) = sum_j e^{-ik.r_j} (q_j - i k.mu_j - (1/2) k.Th_j.k)
    U    = (1/2) (4 pi K_E / V) sum_{k != 0} g(k) |S(k)|^2 ,  g = exp(-k^2/4 alpha^2) / k^2

and the outputs are the derivatives of U with respect to the moments,

    phi_i   = dU/dq_i   = phi(r_i)
    field_i = dU/dmu_i  = + grad phi (r_i)            (the potential gradient, -E)
    grad_i  = dU/dTh_i  = (1/2) grad grad phi (r_i)   (Cartesian 3x3, NOT projected)

with ``phi(r) = (4 pi K_E / V) sum_{k != 0} g(k) Re[e^{ik.r} S(k)]``.  That is
what the chunked loop of ``LatticeOperator.apply`` accumulates (half of each
+-k pair with a doubled weight is the same sum), point multipoles with the
alpha Gaussian only: the site widths live in the real-space part.  k = 0 is
excluded, as there; the background ``-(pi K_E / V alpha^2) Q`` and the self
terms stay with the caller.  ``U = (1/2) sum_i (q_i phi_i + mu_i . field_i +
Th_i : grad_i)`` (U is homogeneous quadratic in the moments).

Derivation
----------
Mesh coordinates ``u_ia = K_a (r_i A^{-1})_a`` for the cell ``A`` (rows = lattice
vectors) and the mesh ``(K_1, K_2, K_3)``; with ``k = 2 pi A^{-1} m`` for integer
``m``, ``k . r = 2 pi sum_a m_a u_a / K_a``.  The Euler exponential spline
(Essmann et al. sec. 3) gives, per axis,

    exp(2 pi i m u / K) ~= b(m) sum_l M_p(u - l) exp(2 pi i m l / K)
    b(m) = exp(2 pi i (p-1) m / K) / sum_{k=0}^{p-2} M_p(k+1) exp(2 pi i m k / K)

with ``M_p`` the cardinal B-spline of order p (support [0, p]).  A multipole is
a differential operator on its source position,
``(q - i k.mu - (1/2) k.Th.k) e^{-ik.r} = (q + mu.D_r + (1/2) Th:D_r D_r) e^{-ik.r}``,
and ``D_r = J D_u`` with ``J[b, a] = du_a/dr_b = K_a (A^{-1})[b, a]``.  The mesh
therefore sees the transformed moments ``mu' = J^T mu``, ``Th' = J^T Th J``, and
the charge array is

    Q(l) = sum_j (q_j + mu'_j . D_u + (1/2) Th'_j : D_u D_u) prod_a M_p(u_ja - l_a),

so that ``S(k_m) ~= conj(b(m)) Qhat(m)`` (``Qhat`` the unnormalised DFT) and

    U_pme = (1/2) sum_{m != 0} G(m) |Qhat(m)|^2 ,  G(m) = (4 pi K_E / V) g(k_m) prod_a |b_a(m_a)|^2
          = (1/2) sum_l Q(l) phi_mesh(l) ,           phi_mesh = K_tot irfftn(G rfftn(Q)).

The outputs are the moment derivatives of ``U_pme``: ``phi_mesh`` gathered back
through the SAME stencils (the adjoint of spreading), ``phi_i = sum_l M
phi_mesh``, ``field_i = J sum_l D_u M phi_mesh``, ``grad_i = (1/2) J [sum_l
D_u D_u M phi_mesh] J^T``.  Consequences, all tested:

* the approximate operator is exactly SYMMETRIC (what CG needs), and every
  autograd derivative (forces, stress, double backward) is the exact
  derivative of ``U_pme``: energy conserving;
* ``U_pme`` is invariant, to round-off, under a lattice translation of one atom
  and under a translation of all atoms by a MESH vector; under a general
  translation only to the mesh accuracy, so ``sum_i F_i`` is small but not
  zero (intrinsic to SPME with analytic forces, Essmann et al. sec. 4);
* ``Qhat(0) = sum_j q_j`` exactly (the B-splines are a partition of unity and
  their derivatives sum to zero) and ``G(0) = 0``: k = 0 is excluded exactly
  as the direct sum excludes it, neutral cell or not.

An even axis' Nyquist index ``K/2`` stands for both ``+K/2`` and ``-K/2``, whose
k differ in a triclinic cell, so ``G`` is set to 0 on the Nyquist planes; the
mesh rule makes their weight negligible (and the calibration includes it).

Accuracy
--------
Each B-spline derivative, in spreading or in gathering, costs one order:
measured on random multipoles, the relative RMS error of output rank ``t`` from
source rank ``s`` falls as ``h^(p - max(s, t))`` (charge -> phi as ``h^p``,
anything touching a quadrupole as ``h^(p-2)``).  :func:`pme_mesh` sizes the
mesh from a model of that error, calibrated per rank (see there).

Fourier-space differentiation -- spreading every Cartesian moment component
with the plain spline and applying ``-ik``, ``-(1/2)kk`` exactly in k space --
was implemented and measured as the alternative.  It keeps the full ``h^p``
rate for every output (20-100x smaller error at equal mesh, and an energy
``C^{p-2}`` in the positions whatever the rank), but needs ten FFT channels
each way.  At EQUAL accuracy it was 3-4x slower and far heavier in memory
(1000 atoms, p = 10, measured error 1.5e-9 at K = 48 against 2.1e-9 for this
route at K = 60: 0.20 s against 0.064 s for setup + energy + forces; the same
meshes scaled to 8000 atoms: 2.65 s against 0.60 s), so it is not used.

Smoothness: ``M_p`` is ``C^{p-2}``; a quadrupole is spread with ``M_p''``, so the
energy is ``C^{p-2-r}`` in the positions for highest rank ``r``.  Quadrupoles
need ``p >= 5`` for continuous forces (enforced) and ``p >= 6`` for continuous
position Hessians (the double-backward path).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .constants import K_E

__all__ = [
    "DEFAULT_PME_ORDER",
    "PMEReciprocal",
    "pme_reciprocal",
    "pme_mesh",
    "pme_error_estimate",
    "fft_friendly_size",
]

#: Default B-spline order (see :func:`pme_mesh` for the measured trade-off).
DEFAULT_PME_ORDER = 8

MeshLike = Union[int, Sequence[int]]


def fft_friendly_size(n: int, primes: Tuple[int, ...] = (2, 3, 5, 7)) -> int:
    """Smallest integer ``>= n`` whose prime factors all lie in ``primes``."""
    m = max(int(n), 1)
    while True:
        rest = m
        for p in primes:
            while rest % p == 0:
                rest //= p
        if rest == 1:
            return m
        m += 1


# ---------------------------------------------------------------------------
# Cardinal B-splines
# ---------------------------------------------------------------------------
def _bspline_tables(w: Tensor, order: int, n_derivatives: int) -> Tensor:
    """``d^d/du^d M_p(w + j)`` for ``j = 0..p-1``, ``d = 0..n_derivatives``.

    ``w`` in ``[0, 1)`` is the fractional part of the mesh coordinate; the p
    mesh points an atom touches are ``floor(u) - j`` with weight
    ``M_p(u - floor(u) + j)``.  Recursion (Essmann et al. eq. 4.1)

        M_n(x) = [x M_{n-1}(x) + (n - x) M_{n-1}(x - 1)] / (n - 1),  M_1 = 1 on [0, 1)

    and derivatives ``M_n^(d)(x) = sum_i (-1)^i C(d, i) M_{n-d}(x - i)``.  Every
    entry is a polynomial in ``w`` built from differentiable torch ops, so
    autograd returns the true higher derivatives of the spline (the piece index
    ``floor(u)`` is constant within a piece).  Returns ``[..., d+1, p]``.
    """
    if order - n_derivatives < 1:
        raise ValueError(f"B-spline order {order} has no derivative of order {n_derivatives}")
    keep = {}
    table = torch.ones_like(w).unsqueeze(-1)                      # M_1(w + 0)
    if order - n_derivatives <= 1:
        keep[1] = table
    for n in range(2, order + 1):
        x = w.unsqueeze(-1) + torch.arange(n, dtype=w.dtype, device=w.device)
        zero = torch.zeros_like(table[..., :1])
        here = torch.cat([table, zero], dim=-1)                   # M_{n-1}(w + j)
        below = torch.cat([zero, table], dim=-1)                  # M_{n-1}(w + j - 1)
        table = (x * here + (n - x) * below) / (n - 1)
        if n >= order - n_derivatives:
            keep[n] = table
    out = []
    for d in range(n_derivatives + 1):
        t = keep[order - d]
        for _ in range(d):
            zero = torch.zeros_like(t[..., :1])
            t = torch.cat([t, zero], dim=-1) - torch.cat([zero, t], dim=-1)
        out.append(t)
    return torch.stack(out, dim=-2)


def _bspline_moduli(size: int, order: int) -> Tensor:
    """``|b(m)|^2`` for ``m = 0..size-1`` (Essmann et al. eq. 4.4), float64 constants.

    ``|b(m)|^-2 = |sum_{k=0}^{p-2} M_p(k+1) e^{2 pi i m k/K}|^2``.  For odd p this
    vanishes at ``m = K/2`` (``M_p`` is symmetric about p/2, the terms cancel in
    pairs); the Nyquist planes are dropped from ``G`` anyway, so it is set to 0.
    """
    knots = _bspline_tables(torch.zeros(1, dtype=torch.float64), order, 0)[0, 0, 1:]  # M_p(1..p-1)
    k = torch.arange(order - 1, dtype=torch.float64)
    m = torch.arange(size, dtype=torch.float64)
    arg = (2.0 * math.pi / size) * m.unsqueeze(-1) * k
    den = (knots * torch.cos(arg)).sum(-1) ** 2 + (knots * torch.sin(arg)).sum(-1) ** 2
    return torch.where(den > 1e-10, 1.0 / den.clamp_min(1e-10), torch.zeros_like(den))


# ---------------------------------------------------------------------------
# Influence function
# ---------------------------------------------------------------------------
def _signed_modes(size: int, half: bool) -> Tuple[Tensor, Tensor]:
    """Signed wave numbers of one mesh axis and the mask of the kept modes.

    Full axis in fftfreq order ``0, 1, .., -1``; half (rfft) axis ``0..K//2``.
    An even axis' Nyquist ``K/2`` is masked out.
    """
    if half:
        m = torch.arange(size // 2 + 1, dtype=torch.float64)
    else:
        m = torch.fft.fftfreq(size, 1.0 / size).round().to(torch.float64)
    keep = torch.ones_like(m, dtype=torch.bool)
    if size % 2 == 0:
        keep = m.abs() != size // 2
    return m, keep


def _influence(cell: Tensor, alpha: float, mesh: Tuple[int, int, int], order: int) -> Tensor:
    """``G(m) = (4 pi K_E / V) g(k_m) prod_a |b_a(m_a)|^2`` on the rfft half mesh.

    ``[K_1, K_2, K_3 // 2 + 1]``, zero at m = 0 and on the Nyquist planes;
    differentiable in ``cell`` through ``k = sum_a m_a b_a`` (``b = 2 pi A^{-T}``,
    rows) and ``V = |det A|``.
    """
    dtype, device = cell.dtype, cell.device
    recip = 2.0 * math.pi * torch.linalg.inv(cell).transpose(-1, -2)
    modes, masks = zip(*(_signed_modes(size, half=(axis == 2)) for axis, size in enumerate(mesh)))
    m1, m2, m3 = (m.to(dtype=dtype, device=device) for m in modes)
    k = (
        m1.view(1, -1, 1, 1) * recip[0].view(3, 1, 1, 1)
        + m2.view(1, 1, -1, 1) * recip[1].view(3, 1, 1, 1)
        + m3.view(1, 1, 1, -1) * recip[2].view(3, 1, 1, 1)
    )
    k2 = (k * k).sum(0)
    keep = (masks[0].view(-1, 1, 1) & masks[1].view(1, -1, 1) & masks[2].view(1, 1, -1)).to(device)
    origin = (m1.view(-1, 1, 1) == 0) & (m2.view(1, -1, 1) == 0) & (m3.view(1, 1, -1) == 0)
    keep = keep & ~origin
    safe = torch.where(keep, k2, torch.ones_like(k2))
    g = torch.where(keep, torch.exp(-safe / (4.0 * alpha * alpha)) / safe, torch.zeros_like(k2))
    b1 = _bspline_moduli(mesh[0], order).to(dtype=dtype, device=device).view(-1, 1, 1)
    b2 = _bspline_moduli(mesh[1], order).to(dtype=dtype, device=device).view(1, -1, 1)
    b3 = _bspline_moduli(mesh[2], order)[: mesh[2] // 2 + 1].to(dtype=dtype, device=device).view(1, 1, -1)
    volume = torch.det(cell).abs()
    return (4.0 * math.pi * K_E / volume) * g * (b1 * b2 * b3)


def _half_weights(mesh: Tuple[int, int, int], dtype, device) -> Tensor:
    """Multiplicity of each rfft column in the full spectrum: 1 at m_3 = 0 (and K_3/2), else 2."""
    n3 = mesh[2] // 2 + 1
    w = [2.0] * n3
    w[0] = 1.0
    if mesh[2] % 2 == 0:
        w[-1] = 1.0
    return torch.tensor(w, dtype=dtype, device=device)


def _selectors(n_derivatives: int, dtype, device) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Maps between the moments and the flattened ``(d1, d2, d3)`` derivative grid.

    ``f = (d1 D + d2) D + d3`` with ``D = n_derivatives + 1``.  ``S0[f]`` picks
    (0,0,0), ``S1[f, a]`` the first derivative along ``a``, ``S2[f, 3a+b]`` the
    second derivative along ``a`` and ``b`` (both orders of a mixed pair hit the
    same f, which is what symmetrises ``Th'``).
    """
    dims = n_derivatives + 1

    def flat(d):
        return (d[0] * dims + d[1]) * dims + d[2]

    size = dims ** 3
    s0 = torch.zeros(size, dtype=dtype, device=device)
    s0[0] = 1.0
    s1 = torch.zeros(size, 3, dtype=dtype, device=device)
    for a in range(3):
        d = [0, 0, 0]
        d[a] += 1
        s1[flat(d), a] = 1.0
    s2 = None
    if n_derivatives >= 2:
        s2 = torch.zeros(size, 9, dtype=dtype, device=device)
        for a in range(3):
            for b in range(3):
                d = [0, 0, 0]
                d[a] += 1
                d[b] += 1
                s2[flat(d), 3 * a + b] = 1.0
    return s0, s1, s2


# ---------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------
class PMEReciprocal:
    """SPME reciprocal-space product for fixed positions and cell.

    The geometry -- B-spline tables with their first and second derivatives,
    stencil indices, influence function -- is built once here, differentiably
    in ``pos`` and ``cell``, and reused by every :meth:`apply`: per product a CG
    solve pays one spread, one ``rfftn``/``irfftn`` pair and one gather.

    Args:
        pos: ``[n, 3]`` positions (any image; the mesh wraps them).
        cell: ``[3, 3]`` lattice vectors as rows.
        alpha: Ewald splitting parameter (1/A), the one the real space uses.
        mesh: ``(K_1, K_2, K_3)`` points along the three lattice vectors (an
            int for all three); :func:`pme_mesh` chooses it.
        order: B-spline order ``p >= 4``; ``>= 5`` with quadrupoles (continuous
            forces), ``>= 6`` for continuous position Hessians.  Tested 4..10.
        quadrupoles: build second-derivative stencils, accept ``theta`` and
            return ``grad``.  Without it ``grad`` is None and ``theta`` must be.

    Memory ``O(n p^3 + M)``: the stencil index (int32) and, under autograd, one
    ``[n, p^3]`` gather per :meth:`apply`.
    """

    def __init__(
        self,
        pos: Tensor,
        cell: Tensor,
        alpha: float,
        mesh: MeshLike,
        order: int = DEFAULT_PME_ORDER,
        *,
        quadrupoles: bool = True,
    ) -> None:
        mesh = (int(mesh),) * 3 if isinstance(mesh, int) else tuple(int(k) for k in mesh)
        if len(mesh) != 3 or min(mesh) < 1:
            raise ValueError(f"mesh must be three positive integers, got {mesh!r}")
        order = int(order)
        if order < 4 or (quadrupoles and order < 5):
            raise ValueError(
                f"PME B-spline order {order} is too low: quadrupoles are spread with "
                "M_p'' and need p >= 5 for continuous forces (p >= 4 otherwise)."
            )
        dtype, device = pos.dtype, pos.device
        self.mesh, self.order, self.alpha = mesh, order, float(alpha)
        self.quadrupoles = bool(quadrupoles)
        self.n_derivatives = 2 if quadrupoles else 1
        self.n_atoms = int(pos.shape[0])
        self.size = mesh[0] * mesh[1] * mesh[2]

        inverse = torch.linalg.inv(cell)
        counts = torch.tensor(mesh, dtype=dtype, device=device)
        u = (pos @ inverse) * counts                                   # mesh coordinates
        base = torch.floor(u.detach())
        tables = _bspline_tables(u - base, order, self.n_derivatives)  # [n, 3, d+1, p]
        self.t1, self.t2, self.t3 = tables[:, 0], tables[:, 1], tables[:, 2]
        offsets = torch.arange(order, device=device)
        sizes = torch.tensor(mesh, device=device).view(1, 3, 1)
        points = torch.remainder(base.long().unsqueeze(-1) - offsets, sizes)   # [n, 3, p]
        index = (
            (points[:, 0, :, None, None] * mesh[1] + points[:, 1, None, :, None]) * mesh[2]
            + points[:, 2, None, None, :]
        ).reshape(-1)
        self.index = index.to(torch.int32) if self.size < 2**31 else index
        self.jacobian = inverse * counts                               # J[b, a] = du_a/dr_b
        self.influence = float(self.size) * _influence(cell, self.alpha, mesh, order)
        self.half_weights = _half_weights(mesh, dtype, device)
        self.s0, self.s1, self.s2 = _selectors(self.n_derivatives, dtype, device)

    # ------------------------------------------------------------------
    def spread(self, q: Tensor, mu: Optional[Tensor] = None, theta: Optional[Tensor] = None) -> Tensor:
        """The charge mesh ``Q(l)``, flattened ``[K_1 K_2 K_3]``; ``theta`` Cartesian."""
        n, dims, j = self.n_atoms, self.n_derivatives + 1, self.jacobian
        coeff = q.unsqueeze(-1) * self.s0                             # [n, dims^3]
        if mu is not None:
            coeff = coeff + (mu @ j) @ self.s1.T                      # mu' = J^T mu
        if theta is not None:
            if not self.quadrupoles:
                raise ValueError("theta given to a PMEReciprocal built with quadrupoles=False")
            th_mesh = torch.einsum("ba,nbc,cd->nad", j, theta, j).reshape(n, 9)   # J^T Th J
            coeff = coeff + 0.5 * (th_mesh @ self.s2.T)
        # values[n, j1, j2, j3] = sum_{d} coeff[n, d1, d2, d3] T1[d1, j1] T2[d2, j2] T3[d3, j3]
        c = coeff.view(n, dims, dims, dims)
        a = torch.einsum("nabc,naj->nbcj", c, self.t1)
        b = torch.einsum("nbcj,nbk->ncjk", a, self.t2)
        values = torch.einsum("ncjk,ncl->njkl", b, self.t3)
        mesh = torch.zeros(self.size, dtype=values.dtype, device=values.device)
        return mesh.index_add(0, self.index, values.reshape(-1))

    def potential(self, charge_mesh: Tensor) -> Tensor:
        """``phi_mesh = K_tot irfftn(G rfftn(Q)) = dU_pme/dQ``, flattened."""
        spectrum = torch.fft.rfftn(charge_mesh.view(self.mesh)) * self.influence
        return torch.fft.irfftn(spectrum, s=self.mesh).reshape(-1)

    def gather(self, phi_mesh: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Adjoint of :meth:`spread`: ``(phi, field, grad)`` at the sites."""
        n, p, j = self.n_atoms, self.order, self.jacobian
        local = phi_mesh.index_select(0, self.index).view(n, p, p, p)
        x = torch.einsum("njkl,ncl->njkc", local, self.t3)
        y = torch.einsum("njkc,nbk->njbc", x, self.t2)
        r = torch.einsum("njbc,naj->nabc", y, self.t1).reshape(n, -1)
        phi = r @ self.s0
        field = (r @ self.s1) @ j.T                                    # J D_u phi
        grad = None
        if self.quadrupoles:
            hessian = (r @ self.s2).view(n, 3, 3)
            grad = 0.5 * torch.einsum("ba,nad,cd->nbc", j, hessian, j)   # (1/2) J H J^T
        return phi, field, grad

    def apply(
        self, q: Tensor, mu: Optional[Tensor] = None, theta: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """``(phi [n], field [n, 3], grad [n, 3, 3] or None)``, module conventions.

        ``theta`` is the CARTESIAN quadrupole; None means zero moments of that
        rank.  ``grad`` is returned whenever the object was built with
        ``quadrupoles=True`` (e.g. the gradient that charges exert on sites
        whose quadrupole is still zero).
        """
        return self.gather(self.potential(self.spread(q, mu, theta)))

    def energy(self, q: Tensor, mu: Optional[Tensor] = None, theta: Optional[Tensor] = None) -> Tensor:
        """``U_pme = (1/2) sum_m G |Qhat|^2`` from the spectrum alone (no gather).

        Equal to ``(1/2) sum_i (q phi + mu . field + Th : grad)`` of :meth:`apply`
        to round-off (tested).
        """
        spectrum = torch.fft.rfftn(self.spread(q, mu, theta).view(self.mesh))
        power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        return (0.5 / self.size) * (self.influence * power * self.half_weights).sum()


def pme_reciprocal(
    pos: Tensor,
    cell: Tensor,
    q: Tensor,
    mu: Optional[Tensor],
    theta: Optional[Tensor],
    alpha: float,
    mesh: MeshLike,
    order: int = DEFAULT_PME_ORDER,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """One-shot SPME reciprocal product: :class:`PMEReciprocal` built and applied.

    Drop-in for the whole chunked k-sum of ``LatticeOperator.apply``: returns
    ``(phi, field, grad)`` with the conventions and normalisation of
    ``_reciprocal_chunk`` summed over every ``k != 0``.  ``theta`` is the
    CARTESIAN ``[n, 3, 3]`` quadrupole, or None (no second-derivative stencils;
    ``grad`` is then None).  Project ``grad`` onto the ``2e`` basis as ``apply``
    does.
    """
    operator = PMEReciprocal(pos, cell, alpha, mesh, order, quadrupoles=theta is not None)
    return operator.apply(q, mu, theta)


# ---------------------------------------------------------------------------
# Mesh sizing
# ---------------------------------------------------------------------------
#: Largest measured ratio of the actual error to the model was 3.6 at rank 2
#: and 6.75 at ranks 0-1 (5 cells, alpha 0.3-0.8, p 5-10, x 0.06-0.45; see
#: :func:`pme_mesh`); these carry a ~1.4x margin.
_ERROR_CONSTANT = {0: 10.0, 1: 10.0, 2: 5.0}


def pme_error_estimate(x: float, order: int, max_rank: int = 2) -> float:
    """Model of the worst relative RMS error at mesh spacing ``x = alpha |a| / K``.

    For a mode with mesh frequency ``xi_a = m_a / K_a = k . a_a / (2 pi K_a)``
    along axis a, the B-spline interpolation leaves aliases at ``m_a +- K_a``
    of relative power ``(xi/(1 -+ xi))^(2 p_eff)`` (the ratio of the spline's
    Fourier transforms ``sinc^p``), and a mode past Nyquist is lost (power 1).
    The model averages that over directions (``xi_a = kappa x |cos| / 2 pi``,
    ``kappa = k / alpha``) and over the spectrum of the worst source-output
    pair, ``kappa^(2w) exp(-kappa^2/2)`` with ``w = s + t - 1`` (amplitude
    ``g k^(s+t)``, squared, times the ``k^2 dk`` shell):

        err^2 = A^2 < 3 sum_{+-} (xi/(1 -+ xi))^(2 p_eff) >_{kappa, cos}

    ``p_eff = p - max(1, R)`` (each spline derivative costs one order, and the
    field is always needed), ``w = max(1, 2R - 1)``.  ``A`` is the calibrated
    constant.  Monotone in ``x``.
    """
    if max_rank not in _ERROR_CONSTANT:
        raise ValueError(f"max_rank must be 0, 1 or 2, got {max_rank!r}")
    p_eff = order - max(1, max_rank)
    w = max(1, 2 * max_rank - 1)
    kappa = torch.linspace(1e-3, 14.0, 1400, dtype=torch.float64)
    cosine = (torch.arange(64, dtype=torch.float64) + 0.5) / 64.0
    xi = (kappa * (x / (2.0 * math.pi))).unsqueeze(-1) * cosine
    inside = xi < 0.5
    t = torch.where(inside, xi, torch.zeros_like(xi))
    alias = (t / (1.0 - t)) ** (2 * p_eff) + (t / (1.0 + t)) ** (2 * p_eff)
    per_mode = 3.0 * torch.where(inside, alias, torch.ones_like(xi)).mean(-1)
    weight = kappa ** (2 * w) * torch.exp(-0.5 * kappa * kappa)
    return _ERROR_CONSTANT[max_rank] * float(((weight * per_mode).sum() / weight.sum()).sqrt())


def pme_mesh(
    cell: Tensor,
    alpha: float,
    accuracy: float,
    order: int = DEFAULT_PME_ORDER,
    max_rank: int = 2,
) -> Tuple[int, int, int]:
    """Mesh ``(K_1, K_2, K_3)`` for a target relative RMS error of the outputs.

    Guarantees (to the calibration) that ``phi``, ``field`` and -- for
    ``max_rank = 2`` -- ``grad`` from sources of rank ``<= max_rank`` each have
    relative RMS error ``<= accuracy`` against the exact (untruncated)
    reciprocal sum.  Rule: ``K_a = fft_friendly(max(p, ceil(alpha |a_a| / x)))``
    with ``x`` the largest spacing at which :func:`pme_error_estimate` meets
    ``accuracy`` (bisection).  ``|a_a|``, not the cell width, because the
    largest mesh frequency a wave vector reaches along axis a is
    ``k |a_a| / (2 pi K_a)`` -- the same argument that sizes ``k_max`` in
    ``EwaldParams.auto_tune``.  Sizes have prime factors 2, 3, 5, 7 only.

    Calibration (``tests``-style exact sum at ``exp(-k^2/4 alpha^2) < 1e-20``):
    random multipoles in a triclinic 8 A, a cubic 12 A, a strongly sheared 9 A
    and a triclinic 20 A (300 atoms) cell, and a jittered 216-atom rocksalt
    crystal; ``alpha`` 0.3-0.8 1/A (``alpha L`` 2.4-17); ``p`` 5-10; ``x``
    0.06-0.45; 1960 (cell, alpha, p, x, source) cases, each output and source
    rank separately.  Actual / model error: ``[0.85, 3.6]`` at rank 2,
    ``<= 6.75`` at ranks 0-1 -- one constant per rank across 10 decades of
    error.  Spacing ``x`` chosen by the rule (``max_rank = 2``):

        accuracy     p=6     p=8     p=10
        1e-4       0.138   0.280   0.360
        1e-6       0.045   0.144   0.232
        1e-8       0.014   0.069   0.141

    (``max_rank <= 1``: 0.216 / 0.119 at p = 8 for 1e-6 / 1e-8.)  Out of
    sample -- two other triclinic cells, alpha 0.37 and 0.62, new seeds -- the
    worst error / accuracy with these meshes was 0.40 (rank 2), 0.21 (rank 1),
    0.09 (rank 0).  Below x = 0.06 (e.g. p = 6 at 1e-8) the model is used
    outside the calibrated range, in its asymptotic ``x^(p-R)`` regime.

    Cost is ``~ (alpha L / x)^3`` mesh points against ``n p^3`` stencil work;
    measured, p = 8 is fastest at 1e-6 and p = 10 at 1e-8 (see the module
    benchmark).  ``cell`` is read detached: the mesh is a hyperparameter,
    chosen once, never differentiated.
    """
    if not 0.0 < accuracy < 0.1:
        raise ValueError(f"accuracy must lie in (0, 0.1), got {accuracy}")
    lo, hi = 1e-4, 3.0
    for _ in range(50):
        mid = math.sqrt(lo * hi)
        if pme_error_estimate(mid, order, max_rank) <= accuracy:
            lo = mid
        else:
            hi = mid
    lengths = cell.detach().to(torch.float64).norm(dim=-1).tolist()
    return tuple(fft_friendly_size(max(order, math.ceil(alpha * length / lo))) for length in lengths)
