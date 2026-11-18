"""Matrix-free multipole operator for OPEN boundaries (SPEC 12, 16 without a cell).

The open-boundary route materialises dense ``[n, n]``, ``[n, n, 3]``,
``[n, n, 3, 3]`` (and, with quadrupoles, ``[n, n, 5]``, ``[n, n, 3, 5]``,
``[n, n, 5, 5]`` via ``[n, n, 3, 3, 3, 3]``) pair blocks and then a dense K.
The fixed-k CG of SPEC 2 needs only ``K v``, so this module provides that
product for molecules, clusters and proteins -- no cell, no images -- without
forming any ``n x n`` object.  It is the non-periodic sibling of
:class:`~trace_electrostatics.lattice_operator.LatticeOperator` with the same
interface, so :func:`~trace_electrostatics.lattice_operator.solve_lattice`
accepts it unchanged.

Kernel
------
The FULL (unscreened) kernel: no reciprocal part, no Ewald self term, no
background term.  Gaussian (SPEC 12): every block is a derivative of
``K_E erf(r/s)/r`` at the combined width ``s`` of the two moments it couples,
with damping ``lambda_{2n+1} = P(n + 1/2, r^2/s^2)``; the q_i-mu_j and q_j-mu_i
widths differ, so both orderings of every pair are evaluated.  Thole: one
pair damping for every rank (:func:`~trace_electrostatics.kernels.thole_damping`,
or :func:`~trace_electrostatics.kernels.thole_damping_quadrupole` with the
quadrupole sector).  Charge-charge is Gaussian/erf under both (SPEC 1.1).

Pair enumeration
----------------
Every unordered pair ``i < j`` exactly once, in row blocks: rows ``[a, b)``
against columns ``[a, n)`` as dense ``[b - a, n - a]`` tiles with the strict
upper triangle kept, so no index list is stored and every reduction is a
contiguous row or column sum.  One pair feeds both of its sites: with
``r = r_i - r_j`` the site ``j`` sees the same terms with ``r -> -r`` and the
roles of the two moment sets exchanged.  ``O(n^2)`` time, ``O(chunk)`` memory.

Closed-form contractions (``r = r_i - r_j``, ``Th`` traceless cartesian)
-----------------------------------------------------------------------
From the energy-convention tensors of :mod:`~trace_electrostatics.kernels`
(``T01 = K_E l3 r/r^3``, ``T = K_E(l3 I/r^3 - 3 l5 rr/r^5)``, ``T02``, ``T12``,
``T22``), the pair energy of one unordered pair is

    U_ij = J q_i q_j + g_a q_i (r.mu_j) - g_b q_j (r.mu_i)
           + a (mu_i.mu_j) - b (r.mu_i)(r.mu_j)
           + h_a q_i (r Th_j r) + h_b q_j (r Th_i r)
           - A1 (r.mu_i)(r Th_j r) + B1 mu_i.(Th_j r)
           + A2 (r.mu_j)(r Th_i r) - B2 mu_j.(Th_i r)
           + Th_i : T22 : Th_j

with, at the combined width of the two moments named (sigma, sigma_mu,
sigma_Theta of sites i, j):

    g_a = K_E l3/r^3 (q_i, mu_j)          g_b = K_E l3/r^3 (q_j, mu_i)
    a   = K_E l3/r^3 (mu, mu)             b   = 3 K_E l5/r^5 (mu, mu)
    h_a = (3/2) K_E l5/r^5 (q_i, Th_j)    h_b = (3/2) K_E l5/r^5 (q_j, Th_i)
    A1  = (15/2) K_E l7/r^7 (mu_i, Th_j)  B1  = 3 K_E l5/r^5 (mu_i, Th_j)
    A2  = (15/2) K_E l7/r^7 (mu_j, Th_i)  B2  = 3 K_E l5/r^5 (mu_j, Th_i)

The T02 and T12 lines: ``T02 : Th = (K_E/2)(3 l5 rThr/r^5 - l3 tr Th/r^3)`` and
``T12_abc Th_bc = (K_E/2)(-15 l7 r_a rThr/r^7 + 3 l5 (2 (Th r)_a + r_a tr Th)/r^5)``,
and ``tr Th = 0``; ``U(Th_i, mu_j) = -mu_j . T12(r) : Th_i`` because T12 is odd.
The T22 contraction (the six ``delta r r`` terms give ``delta rThr + 2 (Thr) r
+ 2 r (Thr) + rr tr Th``, the three ``delta delta`` terms ``delta tr Th + 2 Th``):

    dU/dTh_i = (K_E/4)[105 l9/r^9 rr (r Th_j r) - 30 l7/r^7 ((Th_j r) r + r (Th_j r))
                       + 6 l5/r^5 Th_j]  + (multiple of delta, projected away)

The derivatives of ``U = sum_{i<j} U_ij`` are then (``phi = dU/dq``,
``F = dU/dmu``, ``G = dU/dTheta`` in the orthonormal ``2e`` basis ``W``; every
cartesian rank-2 sum is projected with ``W``, which drops its trace):

    phi_i += J q_j + g_a r.mu_j + h_a rTh_jr        phi_j += J q_i - g_b r.mu_i + h_b rTh_ir
    F_i   += (-g_b q_j - b r.mu_j - A1 rTh_jr) r + a mu_j + B1 Th_j r
    F_j   += ( g_a q_i - b r.mu_i + A2 rTh_ir) r + a mu_i - B2 Th_i r
    G_i   += W:[(h_b q_j + A2 r.mu_j + a22 rTh_jr) rr + (w r^T + r w^T)] + c22 th_j,
             w = -(B2/2) mu_j - 2 b22 Th_j r
    G_j   += W:[(h_a q_i - A1 r.mu_i + a22 rTh_ir) rr + (w r^T + r w^T)] + c22 th_i,
             w =  (B1/2) mu_i - 2 b22 Th_i r

with ``a22, b22, c22 = (K_E/4)(105 l9/r^9, 15 l7/r^7, 6 l5/r^5)``.  Writing
``u = (s/2) r + w`` the rank-2 part is ``W:(Q + Q^T) = 2 W:Q``, ``Q = sum_j u r^T``:
nine row sums, no per-pair 3x3 matrix.  Every line is the one
:class:`~trace_electrostatics.lattice_operator.LatticeOperator` applies to an
ordered pair, and the whole product equals the dense blocks of
``energy._direct_pair_blocks`` / ``_direct_quadrupole_blocks`` to round-off
(tests/electrostatics/test_open_operator.py).

Coincident sites follow the dense construction exactly: the separation is
substituted (``solver._DIAGONAL_SHIFT``) before the square root wherever
``|r|^2 <= DEGENERATE_SEPARATION^2``; the charge-charge entry takes its limit
``kernel_self(sigma_ij)`` there and every multipole coefficient is zero.

Far-field short cut
-------------------
``torch.special.gammainc(n + 1/2, x)`` returns exactly 1.0 for ``x >= 48.65``
(n <= 4, measured on a 1e-4 grid; monotone), so pairs with ``r^2/s^2 >=
_GAMMA_SATURATION = 50`` skip the evaluation -- bit-identical values, and the
dropped derivative is below ``x^3.5 e^-x / Gamma(4.5) < 2e-17``.  On a
protein-like cluster most pairs are far: the incomplete gamma is what costs
(18-27 ns per element, 40x a multiply).

Precompute or recompute (``precompute``)
----------------------------------------
The per-pair coefficients depend on geometry and widths only, and a solve
applies K 8-10 times, so they are computed once per operator when they fit in
``max_precompute_bytes`` (``precompute=None``, default).  Their construction is
checkpointed, so what stays resident is the coefficient tiles themselves
(5 per pair Gaussian, 14 with quadrupoles, 8 bytes each), plus the same again
for their gradient during backward.  ``precompute=False`` is the
memory-bounded mode: nothing ``O(n^2)`` is kept, every product rebuilds each
tile's coefficients inside its checkpoint (and again in backward).

Measured (Gaussian kernel, jittered cubic cluster at 0.1 atoms/A^3, CPU,
10 threads, other jobs running), seconds per product, setup in brackets:

    atoms                      500            2000           8000
    precompute, dipoles        0.004 (0.06)   0.062 (0.37)   0.46 (2.7)
    recompute,  dipoles        0.066          0.51           4.4
    precompute, quadrupoles    0.039 (0.27)   0.24 (0.81)    -- (3.6 GB of tiles)
    recompute,  quadrupoles    0.18           1.00           10.5

Eight products + backward (a fixed-k solve's graph), dipoles: 2000 atoms
2.6 s / 0.34 GB precomputed against 11.6 s / 0.25 GB recomputed; 8000 atoms
precomputed 24 s / 2.9 GB.  Memory-bounded forward + backward of one product
at 8000 atoms: 0.28 GB (dipoles), 0.51 GB (quadrupoles).

Autograd
--------
Every product chunk is a ``torch.utils.checkpoint`` (``use_reentrant=False``):
backward keeps only the chunk inputs, not its ``O(chunk)`` temporaries.
Double backward (force training) works through it -- tested against finite
differences -- but a create_graph backward retains the recomputed chunk
graphs, so second-order memory is ``O(n^2)`` per product whatever the mode.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .constants import K_E
from .kernels import (
    DEGENERATE_SEPARATION,
    combine_sigma,
    coulomb_kernel,
    kernel_self,
    quadrupole_cartesian_basis,
    thole_damping,
    thole_damping_quadrupole,
)
from .solver import _DIAGONAL_SHIFT, _combined_width

__all__ = ["OpenOperator", "DEFAULT_PAIR_CHUNK", "DEFAULT_MAX_PRECOMPUTE_BYTES"]


#: Pair-tile entries per chunk.  A chunk's working set is ~40 float64 tiles of
#: this size (~80 MB at 2^18).  The product time is flat in it (2000 atoms,
#: 2^14 .. 2^22: 0.050-0.064 s per product without quadrupoles, 0.20-0.27 s
#: with), so it only sets the working memory.
DEFAULT_PAIR_CHUNK = 1 << 18

#: Resident budget for the precomputed coefficient tiles (values only; the
#: backward pass needs about the same again for their gradient).
DEFAULT_MAX_PRECOMPUTE_BYTES = 2_000_000_000

#: ``P(n + 1/2, x) == 1.0`` in float64 for ``x >= 48.65`` and every ``n <= 4``
#: (measured with torch.special.gammainc on a 1e-4 grid; P is monotone in x).
_GAMMA_SATURATION = 50.0

_QUADRUPOLE_DIM = 5


def _regularised_gamma(x_squared: Tensor, orders: Sequence[int]) -> List[Tensor]:
    """``P(n + 1/2, x^2)`` for each ``n``, evaluated only where it is not 1.0.

    Bit-identical to ``torch.special.gammainc`` on the whole tile (which returns
    exactly 1.0 beyond ``_GAMMA_SATURATION``); the far entries are constants, so
    they cost nothing and carry a zero derivative (true one < 2e-17).
    """
    flat = x_squared.reshape(-1)
    near = (flat < _GAMMA_SATURATION).nonzero(as_tuple=False).squeeze(1)
    values = flat.index_select(0, near)
    unit = torch.ones_like(flat)
    return [
        unit.index_put((near,), torch.special.gammainc(torch.full_like(values, n + 0.5), values))
        .view_as(x_squared)
        for n in orders
    ]


def _gaussian_factors(r: Tensor, width: Tensor, orders: Sequence[int]) -> List[Tensor]:
    """``kernels.gaussian_damping``'s factors, only the orders asked for."""
    return _regularised_gamma((r / width) ** 2, orders)


def _row_blocks(n_atoms: int, chunk: int) -> List[Tuple[int, int]]:
    """Row ranges ``[a, b)`` whose ``(b - a) x (n - a)`` tiles hold <= ``chunk`` entries.

    Rows are also capped at ``max(16, ceil(n/16))`` so the wasted lower triangle
    of each diagonal sub-block stays a few per cent of the work for small n.
    """
    cap = max(16, -(-n_atoms // 16))
    blocks = []
    start = 0
    while start < n_atoms - 1:
        rows = max(1, min(cap, chunk // (n_atoms - start)))
        stop = min(n_atoms - 1, start + rows)
        blocks.append((start, stop))
        start = stop
    return blocks


def _tile_geometry(pos: Tensor, a: int, b: int):
    """``(x, y, z, r, upper, separated, keep)`` of the tile rows ``[a, b)`` x cols ``[a, n)``.

    The coincidence guard is ``energy._direct_pair_blocks``': substitute the
    squared separation BEFORE the square root, never mask its output.
    ``keep`` also carries the kernels' own ``r > DEGENERATE_SEPARATION`` shield,
    which the dense tensors apply to the substituted distance.
    """
    rows, cols = pos[a:b], pos[a:]
    x = rows[:, 0:1] - cols[:, 0]
    y = rows[:, 1:2] - cols[:, 1]
    z = rows[:, 2:3] - cols[:, 2]
    squared = x * x + y * y + z * z
    upper = torch.ones(squared.shape, dtype=torch.bool, device=pos.device).triu(1)
    separated = squared > DEGENERATE_SEPARATION * DEGENERATE_SEPARATION
    r = torch.sqrt(torch.where(separated, squared, torch.full_like(squared, _DIAGONAL_SHIFT)))
    keep = upper & separated & (r > DEGENERATE_SEPARATION)
    return x, y, z, r, upper, separated, keep


def _tile_coefficients(
    pos: Tensor,
    a: int,
    b: int,
    sigma: Tensor,
    sigma_mu: Optional[Tensor],
    sigma_theta: Optional[Tensor],
    alpha_volume: Optional[Tensor],
    kernel: str,
    thole_a: Optional[float],
    quadrupolar: bool,
) -> Tuple[Tensor, ...]:
    """Per-pair scalars of one tile, in the order :func:`_tile_product` reads them.

    ``(J, g_a, g_b, a, b)`` and with quadrupoles ``(h_a, h_b, A1, B1, A2, B2/2,
    a22, b22, c22)`` -- module docstring for the definitions.  Every entry is
    zero outside the strict upper triangle and at coincident pairs, except
    ``J``, which takes ``kernel_self(sigma_ij)`` at coincidence exactly as the
    dense block does.
    """
    _, _, _, r, upper, separated, keep = _tile_geometry(pos, a, b)
    dtype = pos.dtype
    upper_f = upper.to(dtype)
    keep_f = keep.to(dtype)
    s_row, s_col = sigma[a:b], sigma[a:]
    sigma_pair = combine_sigma(s_row.unsqueeze(1), s_col.unsqueeze(0))
    coulomb = torch.where(separated, coulomb_kernel(r, sigma_pair), kernel_self(sigma_pair)) * upper_f

    inv_r = 1.0 / r
    inv_r2 = inv_r * inv_r
    inv_r3 = inv_r * inv_r2
    inv_r5 = inv_r3 * inv_r2
    scale = K_E * keep_f
    if kernel == "gaussian":
        mu_row, mu_col = sigma_mu[a:b], sigma_mu[a:]
        (l3_a,) = _gaussian_factors(r, _combined_width(s_row, mu_col), (1,))
        (l3_b,) = _gaussian_factors(r, _combined_width(mu_row, s_col), (1,))
        l3_mm, l5_mm = _gaussian_factors(r, _combined_width(mu_row, mu_col), (1, 2))
    else:
        alpha_row = alpha_volume[a:b].unsqueeze(1)
        alpha_col = alpha_volume[a:].unsqueeze(0)
        if quadrupolar:
            l3_mm, l5_mm, l7_t, l9_t = thole_damping_quadrupole(r, alpha_row, alpha_col, thole_a)
        else:
            l3_mm, l5_mm = thole_damping(r, alpha_row, alpha_col, thole_a)
        l3_a = l3_b = l3_mm
    g_a = scale * l3_a * inv_r3
    g_b = g_a if l3_b is l3_a else scale * l3_b * inv_r3
    a_mm = scale * l3_mm * inv_r3
    b_mm = 3.0 * scale * l5_mm * inv_r5
    coefficients = (coulomb, g_a, g_b, a_mm, b_mm)
    if not quadrupolar:
        return coefficients

    inv_r7 = inv_r5 * inv_r2
    inv_r9 = inv_r7 * inv_r2
    if kernel == "gaussian":
        th_row, th_col = sigma_theta[a:b], sigma_theta[a:]
        (l5_qa,) = _gaussian_factors(r, _combined_width(s_row, th_col), (2,))
        (l5_qb,) = _gaussian_factors(r, _combined_width(th_row, s_col), (2,))
        l5_m1, l7_m1 = _gaussian_factors(r, _combined_width(mu_row, th_col), (2, 3))
        l5_m2, l7_m2 = _gaussian_factors(r, _combined_width(th_row, mu_col), (2, 3))
        l5_22, l7_22, l9_22 = _gaussian_factors(r, _combined_width(th_row, th_col), (2, 3, 4))
        h_a = 1.5 * scale * l5_qa * inv_r5
        h_b = 1.5 * scale * l5_qb * inv_r5
        a_1 = 7.5 * scale * l7_m1 * inv_r7
        b_1 = 3.0 * scale * l5_m1 * inv_r5
        a_2 = 7.5 * scale * l7_m2 * inv_r7
        b_2_half = 1.5 * scale * l5_m2 * inv_r5
    else:
        # Thole damps a PAIR: one set of factors for every rank pair.
        l5_22, l7_22, l9_22 = l5_mm, l7_t, l9_t
        h_a = h_b = 1.5 * scale * l5_mm * inv_r5
        a_1 = a_2 = 7.5 * scale * l7_t * inv_r7
        b_2_half = 0.5 * b_mm
        b_1 = b_mm
    a_22 = 26.25 * scale * l9_22 * inv_r9
    b_22 = 3.75 * scale * l7_22 * inv_r7
    c_22 = 1.5 * scale * l5_22 * inv_r5
    return coefficients + (h_a, h_b, a_1, b_1, a_2, b_2_half, a_22, b_22, c_22)


def _tile_product(
    pos: Tensor,
    a: int,
    b: int,
    q: Tensor,
    mu: Tensor,
    theta_cart: Optional[Tensor],
    theta: Optional[Tensor],
    basis: Optional[Tensor],
    *coefficients: Tensor,
) -> Tuple[Tensor, ...]:
    """``(phi, F, G)`` contributions of one tile to its rows and to its columns."""
    x = pos[a:b, 0:1] - pos[a:, 0]
    y = pos[a:b, 1:2] - pos[a:, 1]
    z = pos[a:b, 2:3] - pos[a:, 2]
    r = (x, y, z)
    coulomb, g_a, g_b, a_mm, b_mm = coefficients[:5]
    q_row, q_col = q[a:b], q[a:]
    mu_row, mu_col = mu[a:b], mu[a:]

    r_mu_col = x * mu_col[:, 0] + y * mu_col[:, 1] + z * mu_col[:, 2]          # r . mu_j
    r_mu_row = x * mu_row[:, 0:1] + y * mu_row[:, 1:2] + z * mu_row[:, 2:3]    # r . mu_i
    phi_row = coulomb @ q_col + (g_a * r_mu_col).sum(1)
    phi_col = q_row @ coulomb - (g_b * r_mu_row).sum(0)
    f_row = -(g_b * q_col) - b_mm * r_mu_col                                   # F_i along r
    f_col = g_a * q_row.unsqueeze(1) - b_mm * r_mu_row                         # F_j along r
    field_row = a_mm @ mu_col
    field_col = a_mm.transpose(0, 1) @ mu_row
    if theta is None:
        field_row = field_row + torch.stack([(f_row * c).sum(1) for c in r], dim=1)
        field_col = field_col + torch.stack([(f_col * c).sum(0) for c in r], dim=1)
        return phi_row, phi_col, field_row, field_col, None, None

    h_a, h_b, a_1, b_1, a_2, b_2_half, a_22, b_22, c_22 = coefficients[5:]
    th_row, th_col = theta_cart[a:b], theta_cart[a:]
    # Th_j r and Th_i r, component by component; Theta is symmetric.
    th_r_col = [th_col[:, k, 0] * x + th_col[:, k, 1] * y + th_col[:, k, 2] * z for k in range(3)]
    th_r_row = [th_row[:, k, 0:1] * x + th_row[:, k, 1:2] * y + th_row[:, k, 2:3] * z for k in range(3)]
    r_th_r_col = x * th_r_col[0] + y * th_r_col[1] + z * th_r_col[2]
    r_th_r_row = x * th_r_row[0] + y * th_r_row[1] + z * th_r_row[2]

    phi_row = phi_row + (h_a * r_th_r_col).sum(1)
    phi_col = phi_col + (h_b * r_th_r_row).sum(0)
    f_row = f_row - a_1 * r_th_r_col
    f_col = f_col + a_2 * r_th_r_row
    b_2 = 2.0 * b_2_half
    field_row = field_row + torch.stack(
        [(f_row * r[k] + b_1 * th_r_col[k]).sum(1) for k in range(3)], dim=1
    )
    field_col = field_col + torch.stack(
        [(f_col * r[k] - b_2 * th_r_row[k]).sum(0) for k in range(3)], dim=1
    )

    # G: W : sum_j [s rr + w r^T + r w^T] = 2 W : sum_j u r^T,  u = (s/2) r + w.
    half_s_row = 0.5 * (h_b * q_col + a_2 * r_mu_col + a_22 * r_th_r_col)
    half_s_col = 0.5 * (h_a * q_row.unsqueeze(1) - a_1 * r_mu_row + a_22 * r_th_r_row)
    two_b22 = 2.0 * b_22
    half_b1 = 0.5 * b_1
    u_row = [half_s_row * r[k] - b_2_half * mu_col[:, k] - two_b22 * th_r_col[k] for k in range(3)]
    u_col = [half_s_col * r[k] + half_b1 * mu_row[:, k : k + 1] - two_b22 * th_r_row[k] for k in range(3)]
    q_rows = torch.stack(
        [torch.stack([(u_row[k] * r[m]).sum(1) for m in range(3)], dim=1) for k in range(3)], dim=1
    )
    q_cols = torch.stack(
        [torch.stack([(u_col[k] * r[m]).sum(0) for m in range(3)], dim=1) for k in range(3)], dim=1
    )
    grad_row = 2.0 * torch.einsum("iab,mab->im", q_rows, basis) + c_22 @ theta[a:]
    grad_col = 2.0 * torch.einsum("iab,mab->im", q_cols, basis) + c_22.transpose(0, 1) @ theta[a:b]
    return phi_row, phi_col, field_row, field_col, grad_row, grad_col


def _recomputed_tile(
    pos: Tensor,
    a: int,
    b: int,
    sigma: Tensor,
    sigma_mu: Optional[Tensor],
    sigma_theta: Optional[Tensor],
    alpha_volume: Optional[Tensor],
    kernel: str,
    thole_a: Optional[float],
    quadrupolar: bool,
    q: Tensor,
    mu: Tensor,
    theta_cart: Optional[Tensor],
    theta: Optional[Tensor],
    basis: Optional[Tensor],
) -> Tuple[Tensor, ...]:
    """Memory-bounded mode: the tile's coefficients and its product in one checkpoint.

    Only the coefficients this product reads are built: without ``theta`` the
    quadrupole ones are skipped (Thole's ``lambda_3, lambda_5`` are the same
    bits from either damping function).
    """
    coefficients = _tile_coefficients(
        pos, a, b, sigma, sigma_mu, sigma_theta, alpha_volume, kernel, thole_a,
        quadrupolar and theta is not None,
    )
    return _tile_product(pos, a, b, q, mu, theta_cart, theta, basis, *coefficients)


def _needs_graph(*tensors: Optional[Tensor]) -> bool:
    return torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors)


class OpenOperator:
    """``K_pair v`` for the open-boundary multipole operator, without any ``n x n`` object.

    Same contract as :class:`~trace_electrostatics.lattice_operator.LatticeOperator`:
    acts on PHYSICAL moments ``(q, mu, Theta)`` -- ``Theta`` as its orthonormal
    ``2e`` coefficients -- and returns ``dU_pair/dq``, ``dU_pair/dmu``,
    ``dU_pair/dTheta``; :meth:`apply_reduced` wraps it in the solver's reduced
    moments (``mu = ell mu_hat``, ``Theta = ell^2 Theta_hat``).  The pair part
    only: the on-site ``eta``, ``A_mu``, ``A_Theta`` are added by the caller.
    It equals the dense open-boundary blocks (``energy._direct_pair_blocks``,
    ``_direct_quadrupole_blocks``, ``solver.assemble_system``) to round-off.

    Args:
        pos: ``[n, 3]`` positions, A.
        sigma: ``[n]`` Gaussian charge widths, A (both kernels).
        kernel: ``"gaussian"`` (SPEC 12) or ``"thole"`` (SPEC 1.6).
        sigma_mu: ``[n]`` dipole widths, Gaussian kernel only
            (``solver.gaussian_widths``).
        sigma_theta: ``[n]`` quadrupole widths, Gaussian kernel only; supplying
            it turns the quadrupole sector on.
        alpha_volume: ``[n]`` polarizability volumes in A^3, Thole kernel only
            (``solver.isotropic_polarizability_volume``).
        thole_a: Thole ``a``, Thole kernel only.
        quadrupoles: whether the quadrupole sector is on; default
            ``sigma_theta is not None``.  ``True`` with the Thole kernel builds
            the Thole-damped T02/T12/T22 of :mod:`~trace_electrostatics.kernels`
            -- which the package's solve refuses (SPEC 12: not positive definite
            by construction), so it exists for diagnostics only.
        chunk: pair-tile entries per chunk; working memory is ~40 x 8 bytes
            times this.
        precompute: keep the per-pair coefficients resident (fast) or rebuild
            them in every product (memory-bounded).  ``None`` precomputes when
            :attr:`precompute_bytes` fits in ``max_precompute_bytes``.
        max_precompute_bytes: budget for ``precompute=None``.
    """

    def __init__(
        self,
        pos: Tensor,
        *,
        sigma: Tensor,
        kernel: str,
        sigma_mu: Optional[Tensor] = None,
        sigma_theta: Optional[Tensor] = None,
        alpha_volume: Optional[Tensor] = None,
        thole_a: Optional[float] = None,
        quadrupoles: Optional[bool] = None,
        chunk: int = DEFAULT_PAIR_CHUNK,
        precompute: Optional[bool] = None,
        max_precompute_bytes: int = DEFAULT_MAX_PRECOMPUTE_BYTES,
    ) -> None:
        if kernel not in ("gaussian", "thole"):
            raise ValueError(f"kernel must be 'gaussian' or 'thole', got {kernel!r}")
        if quadrupoles is None:
            quadrupoles = sigma_theta is not None
        if kernel == "gaussian":
            if sigma_mu is None:
                raise ValueError("the Gaussian kernel needs sigma_mu")
            if quadrupoles and sigma_theta is None:
                raise ValueError("Gaussian quadrupoles need sigma_theta (SPEC 12)")
            if sigma_theta is not None and not quadrupoles:
                raise ValueError("sigma_theta was supplied with quadrupoles=False")
        else:
            if alpha_volume is None or thole_a is None:
                raise ValueError("the Thole kernel needs alpha_volume and thole_a")
            if sigma_mu is not None or sigma_theta is not None:
                raise ValueError(
                    "the Thole kernel damps a pair, not a density: it takes no "
                    "sigma_mu / sigma_theta (use quadrupoles=True for its Theta sector)"
                )
        if pos.dim() != 2 or pos.shape[-1] != 3:
            raise ValueError(f"pos must be [n, 3], got {tuple(pos.shape)}")
        if chunk < 1:
            raise ValueError(f"chunk must be positive, got {chunk}")

        self.pos = pos
        self.kernel = kernel
        self.quadrupolar = bool(quadrupoles)
        self.n_atoms = int(pos.shape[0])
        self.sigma = sigma.reshape(-1)
        self.sigma_mu = None if sigma_mu is None else sigma_mu.reshape(-1)
        self.sigma_theta = None if sigma_theta is None else sigma_theta.reshape(-1)
        self.alpha_volume = None if alpha_volume is None else alpha_volume.reshape(-1)
        self.thole_a = None if thole_a is None else float(thole_a)
        self.chunk = int(chunk)
        self.blocks = _row_blocks(self.n_atoms, self.chunk)
        self.basis = (
            quadrupole_cartesian_basis().to(dtype=pos.dtype, device=pos.device)
            if self.quadrupolar
            else None
        )

        self._settings = dict(
            kernel=kernel, thole_a=thole_a, quadrupoles=self.quadrupolar, chunk=self.chunk,
            precompute=precompute, max_precompute_bytes=max_precompute_bytes,
        )
        n_coefficients = 5 + (9 if self.quadrupolar else 0)
        tile_entries = sum((b - a) * (self.n_atoms - a) for a, b in self.blocks)
        #: Resident bytes of the precomputed coefficient tiles (values only).
        self.precompute_bytes = n_coefficients * tile_entries * torch.finfo(pos.dtype).bits // 8
        if precompute is None:
            precompute = self.precompute_bytes <= max_precompute_bytes
        self.precomputed = bool(precompute)
        self._tiles: Optional[List[Tuple[Tensor, ...]]] = None
        if self.precomputed:
            self._tiles = [self._coefficients(a, b) for a, b in self.blocks]

    # ------------------------------------------------------------------------
    def inputs(self) -> Tuple[Optional[Tensor], ...]:
        """The tensors this operator is a function of, in :meth:`rebuilt` order."""
        return (self.pos, self.sigma, self.sigma_mu, self.sigma_theta, self.alpha_volume)

    def rebuilt(self, pos, sigma, sigma_mu, sigma_theta, alpha_volume) -> "OpenOperator":
        """The same operator (kernel, chunking, settings) on new input tensors.

        The converged solve (SPEC 17, ``implicit.QuadraticProblem``) builds one
        detached operator and applies it every CG iteration -- precomputing
        pays there -- and, per derivative order, one operator on tensors that
        carry a graph, applied exactly ONCE (``implicit._explicit_partial``).
        Resident tiles buy that one product little and cost ``O(n^2)`` memory,
        so with ``precompute=None`` a rebuild on graph-carrying inputs uses the
        memory-bounded mode.  Measured, one product forward + backward,
        Gaussian, recompute vs precompute: 2000 atoms 1.54 vs 1.44 s, 0.20 vs
        0.31 GB; 8000 atoms 14.1 vs 11.2 s, 0.28 vs 2.50 GB.
        """
        settings = dict(self._settings)
        if settings["precompute"] is None and _needs_graph(pos, sigma, sigma_mu, sigma_theta, alpha_volume):
            settings["precompute"] = False
        return OpenOperator(
            pos, sigma=sigma, sigma_mu=sigma_mu, sigma_theta=sigma_theta,
            alpha_volume=alpha_volume, **settings,
        )

    def _geometry_inputs(self) -> Tuple:
        return (
            self.pos, self.sigma, self.sigma_mu, self.sigma_theta, self.alpha_volume,
        )

    def _coefficients(self, a: int, b: int) -> Tuple[Tensor, ...]:
        # Checkpointed: the tiles stay resident, their construction graph (the
        # incomplete gammas and everything under them) does not; backward
        # rebuilds it once, after every product has added its gradient.
        args = (
            self.pos, a, b, self.sigma, self.sigma_mu, self.sigma_theta, self.alpha_volume,
            self.kernel, self.thole_a, self.quadrupolar,
        )
        if _needs_graph(*self._geometry_inputs()):
            return checkpoint(_tile_coefficients, *args, use_reentrant=False)
        return _tile_coefficients(*args)

    def _cartesian(self, theta: Tensor) -> Tensor:
        return torch.einsum("im,mab->iab", theta, self.basis)

    def apply(
        self, q: Tensor, mu: Tensor, theta: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """``(dU/dq, dU/dmu, dU/dTheta)`` of the pair energy, physical moments.

        ``theta`` is ``[n, 5]`` in the ``2e`` basis; ``None`` (or an operator
        built without the quadrupole sector) returns ``None`` for the third.
        """
        if theta is not None and not self.quadrupolar:
            raise ValueError("theta supplied to an operator built without the quadrupole sector")
        n = self.n_atoms
        dtype, device = q.dtype, q.device
        phi = torch.zeros(n, dtype=dtype, device=device)
        field = torch.zeros(n, 3, dtype=dtype, device=device)
        quadrupolar = theta is not None
        grad = torch.zeros(n, _QUADRUPOLE_DIM, dtype=dtype, device=device) if quadrupolar else None
        theta_cart = self._cartesian(theta) if quadrupolar else None
        basis = self.basis if quadrupolar else None
        if n < 2:
            return phi, field, grad

        row_phi, row_field, row_grad = [], [], []
        for index, (a, b) in enumerate(self.blocks):
            if self.precomputed:
                coefficients = self._tiles[index]
                if not quadrupolar:
                    coefficients = coefficients[:5]
                args = (self.pos, a, b, q, mu, theta_cart, theta, basis) + tuple(coefficients)
                function = _tile_product
                graph = _needs_graph(self.pos, q, mu, theta, *coefficients)
            else:
                args = (
                    self.pos, a, b, self.sigma, self.sigma_mu, self.sigma_theta,
                    self.alpha_volume, self.kernel, self.thole_a, self.quadrupolar,
                    q, mu, theta_cart, theta, basis,
                )
                function = _recomputed_tile
                graph = _needs_graph(q, mu, theta, *self._geometry_inputs())
            if graph:
                out = checkpoint(function, *args, use_reentrant=False)
            else:
                out = function(*args)
            phi_row, phi_col, field_row, field_col, grad_row, grad_col = out
            row_phi.append(phi_row)
            row_field.append(field_row)
            phi = phi + F.pad(phi_col, (a, 0))
            field = field + F.pad(field_col, (0, 0, a, 0))
            if quadrupolar:
                row_grad.append(grad_row)
                grad = grad + F.pad(grad_col, (0, 0, a, 0))
        # The row blocks partition [0, n - 1); the last atom has no j > i.
        last = n - self.blocks[-1][1]
        phi = phi + F.pad(torch.cat(row_phi), (0, last))
        field = field + F.pad(torch.cat(row_field), (0, 0, 0, last))
        if quadrupolar:
            grad = grad + F.pad(torch.cat(row_grad), (0, 0, 0, last))
        return phi, field, grad

    def apply_reduced(
        self, q: Tensor, mu_hat: Tensor, ell: Tensor, theta_hat: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """The same product in the solver's reduced moments (LatticeOperator's convention)."""
        mu = ell.unsqueeze(-1) * mu_hat
        theta = None if theta_hat is None else (ell * ell).unsqueeze(-1) * theta_hat
        phi, field, grad = self.apply(q, mu, theta)
        field_hat = ell.unsqueeze(-1) * field
        grad_hat = None if grad is None else (ell * ell).unsqueeze(-1) * grad
        return phi, field_hat, grad_hat
