"""Matrix-free lattice operator for large periodic cells (SPEC 16).

The dense periodic route materialises ``[n_images, n, n]`` real-space tensors,
dense ``[n, n]`` reciprocal blocks and a dense K, which caps it near a thousand
atoms.  The fixed-k CG of SPEC 2 never needs K itself -- only ``K v`` -- so this
module provides that product without forming any ``n x n`` object:

* **real space**: an explicit list of ordered pairs ``(i, j, n)`` within the
  real-space cutoff, with a handful of per-pair scalars (the damping factors of
  SPEC 14 at each combined width) and the multipole contractions written out
  in closed form -- ``T22 : Theta`` for instance is

      (K_E/4)[105 l9/r^9 r r (r.Th.r) - 15 l7/r^7 (2 r (Th r) + 2 (Th r) r)
              + 6 l5/r^5 Th]                        (traceless part)

  so no rank-4 tensor is ever built.  Memory O(pairs), time O(pairs).
* **reciprocal space**: the multipole structure factor
  ``S(k) = sum_j e^{-ik.r_j} (q_j - i k.mu_j - (1/2) k.Th_j.k)`` and its
  back-projection, chunked over k.  Time O(n n_k), memory O(n chunk).
* self and background terms: O(n).

With the Ewald parameter balanced for the cell the cost per product is
``O(n^(3/2))``; the ``K v`` it returns is the dense lattice operator's to
round-off (tested).  It is the same functional, not an approximation of it.

The pair list is built by a chunked, image-resolved search (``O(n^2 n_images)``
distance evaluations, ``O(pairs)`` memory); a linked-cell search would make that
preprocessing linear too and is the next step for ~1e5 atoms.
"""

from __future__ import annotations

import contextlib

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .constants import K_E, SQRT_PI
from .ewald import EwaldParams, _gaussian_weights, _image_shell_counts, _wave_vectors
from .kernels import DEGENERATE_SEPARATION, quadrupole_cartesian_basis

__all__ = ["PairList", "lattice_pairs", "LatticeOperator", "balanced_real_cutoff", "solve_lattice"]


#: Relative cost of one reciprocal (atom, k) term to one real-space pair, for a
#: FULL energy + forces evaluation (CPU, 4 threads).  The forward product alone
#: measured 0.55, which put the balance at 17 A for 2000 atoms -- but backward
#: through the per-pair real-space terms (not checkpointed; the reciprocal
#: chunks are) is far costlier, and the measured optimum of energy + forces at
#: 2000 atoms is 12 A (8/10/12/15 A: 23.4/21.9/14.4/23.2 s, 1.9/2.1/2.8/6.3 GB).
#: 0.06 reproduces it (r ~ ratio^(1/6)).  It only sets where the cutoff
#: balances; the operator is the same either way.
DEFAULT_EWALD_COST_RATIO = 0.06


def balanced_real_cutoff(
    n_atoms: int, volume: float, accuracy: float, cost_ratio: float = DEFAULT_EWALD_COST_RATIO
) -> float:
    """Real-space cutoff that minimises the cost of one ``K v`` product.

    Real space costs ``N (4 pi/3) rho r^3`` pairs, reciprocal space
    ``N (1/2)(4 pi/3) k_c^3 V/(2 pi)^3`` terms with ``k_c = 2p/r``,
    ``p = -ln(accuracy)`` (the grid :meth:`EwaldParams.auto_tune` builds for
    that r).  Minimising the sum gives

        r^6 = cost_ratio (2p)^3 V^2 / (16 pi^3 N)

    so ``r ~ N^(1/6)`` at fixed density and the product costs ``O(N^(3/2))``
    instead of the ``O(N^2)`` a fixed cutoff gives.  Measured at 2000 atoms,
    forward product only: 9 A -> 0.64 s, 15 A -> 0.19 s; energy + forces:
    optimum 12 A (see ``DEFAULT_EWALD_COST_RATIO``).
    """
    p = -math.log(accuracy)
    return (cost_ratio * (2.0 * p) ** 3 * volume**2 / (16.0 * math.pi**3 * n_atoms)) ** (1.0 / 6.0)

_QUADRUPOLE_DIM = 5


@dataclass(frozen=True)
class PairList:
    """Ordered image-resolved pairs ``(first, second, shift)`` within a cutoff.

    ``vector = r_first - r_second + shift @ cell`` is differentiable in the
    positions and the cell; the integer ``shift`` and the index arrays are not.
    Both orders of every pair are present; ``(i, i, 0)`` is not.
    """

    first: Tensor
    second: Tensor
    shift: Tensor

    def vectors(self, pos: Tensor, cell: Tensor) -> Tensor:
        return pos.index_select(0, self.first) - pos.index_select(0, self.second) + self.shift @ cell


def _linked_cell_pairs(
    pos: Tensor, cell: Tensor, r_cut: float, bins: Tuple[int, int, int], chunk: int = 512
) -> PairList:
    """Linked-cell search, ``O(n)``: every periodic axis has at least 3 bins.

    Atoms are binned by wrapped fractional coordinate; each is compared with
    the atoms of its own and the 26 neighbouring bins.  With >= 3 bins per
    axis every ``(i, j, n)`` within ``r_cut`` is found exactly once.  The image
    shift of a pair is ``w_j - w_i - c`` with ``w`` the atoms' own wrap counts
    and ``c`` the bin wrap.
    """
    device, dtype = pos.device, pos.dtype
    n_atoms = int(pos.shape[0])
    fractional = pos @ torch.linalg.inv(cell)
    wrap = torch.floor(fractional)
    wrapped = fractional - wrap
    nb = torch.tensor(bins, device=device)
    # min() guards the wrapped coordinate landing on exactly 1.0 by rounding.
    bin3 = torch.minimum((wrapped * nb.to(dtype)).long(), nb - 1)
    flat = (bin3[:, 0] * bins[1] + bin3[:, 1]) * bins[2] + bin3[:, 2]
    n_bins = bins[0] * bins[1] * bins[2]
    order = torch.argsort(flat)
    counts = torch.bincount(flat, minlength=n_bins)
    width = int(counts.max())
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.arange(n_atoms, device=device) - starts[flat[order]]
    table = torch.full((n_bins, width), -1, dtype=torch.long, device=device)
    table[flat[order], slot] = order
    offsets = torch.cartesian_prod(*[torch.arange(-1, 2, device=device)] * 3)   # [27, 3]
    u = wrapped @ cell                                                          # wrapped Cartesian
    firsts, seconds, shifts = [], [], []
    for start in range(0, n_atoms, chunk):
        rows = torch.arange(start, min(start + chunk, n_atoms), device=device)
        neighbour = bin3[rows].unsqueeze(1) + offsets.unsqueeze(0)             # [r, 27, 3]
        bin_wrap = torch.div(neighbour, nb, rounding_mode="floor")             # c
        neighbour = neighbour - bin_wrap * nb
        nflat = (neighbour[..., 0] * bins[1] + neighbour[..., 1]) * bins[2] + neighbour[..., 2]
        cand = table[nflat]                                                    # [r, 27, width]
        valid = cand >= 0
        cand_safe = torch.where(valid, cand, torch.zeros_like(cand))
        image = bin_wrap.to(dtype).unsqueeze(2).expand(-1, -1, width, -1)      # [r, 27, w, 3]
        vector = u[rows].view(-1, 1, 1, 3) - (u[cand_safe] + image @ cell)
        inside = valid & ((vector * vector).sum(-1) < r_cut * r_cut)
        shift = wrap[cand_safe] - wrap[rows].view(-1, 1, 1, 3) - image         # n = w_j - w_i - c
        same = (cand_safe == rows.view(-1, 1, 1)) & (shift.abs().sum(-1) == 0)
        inside = inside & ~same
        index = inside.nonzero(as_tuple=False)
        firsts.append(rows[index[:, 0]])
        seconds.append(cand_safe[index[:, 0], index[:, 1], index[:, 2]])
        shifts.append(shift[index[:, 0], index[:, 1], index[:, 2]])
    return PairList(
        first=torch.cat(firsts), second=torch.cat(seconds), shift=torch.cat(shifts).to(dtype)
    )


def lattice_pairs(
    pos: Tensor, cell: Tensor, r_cut: float, pbc: Tuple[bool, bool, bool] = (True, True, True),
    *, chunk: int = 256,
) -> PairList:
    """Every ordered ``(i, j, n)`` with ``|r_i - r_j + n cell| < r_cut``, ``(i,i,0)`` excluded.

    Minimum image first, then a box of images wide enough for ``r_cut`` along
    each periodic axis.  Selection happens on detached positions -- a cutoff is
    a set, not a function -- and ``PairList.vectors`` re-derives the vectors
    with gradients.
    """
    detached = pos.detach()
    cell_d = cell.detach()
    n_atoms = int(pos.shape[0])
    # Linked cells when every axis is periodic and wide enough for 3 bins of
    # size >= r_cut; the image-box search below otherwise (small cells, where
    # it is cheap anyway).
    spacing = (2.0 * math.pi) / (2.0 * math.pi * torch.linalg.inv(cell_d).T).norm(dim=-1)
    bins = tuple(int(float(d) // r_cut) for d in spacing)
    if all(pbc) and min(bins) >= 3:
        return _linked_cell_pairs(detached, cell_d, float(r_cut), bins)
    shells = _image_shell_counts(cell_d, float(r_cut))
    ranges = [
        torch.arange(-s, s + 1, dtype=pos.dtype, device=pos.device) if periodic
        else torch.zeros(1, dtype=pos.dtype, device=pos.device)
        for s, periodic in zip(shells, pbc)
    ]
    offsets = torch.cartesian_prod(*ranges)                           # [c, 3]
    inverse = torch.linalg.inv(cell_d)
    firsts, seconds, shifts = [], [], []
    for start in range(0, n_atoms, chunk):
        rows = torch.arange(start, min(start + chunk, n_atoms), device=pos.device)
        delta = detached[rows].unsqueeze(1) - detached.unsqueeze(0)    # [r, n, 3]
        minimum = torch.round(delta @ inverse)
        periodic_mask = torch.tensor(pbc, dtype=pos.dtype, device=pos.device)
        minimum = minimum * periodic_mask
        shift = offsets.view(1, 1, -1, 3) - minimum.unsqueeze(2)       # [r, n, c, 3]
        vector = delta.unsqueeze(2) + shift @ cell_d
        inside = (vector * vector).sum(-1) < r_cut * r_cut
        same = (rows.view(-1, 1, 1) == torch.arange(n_atoms, device=pos.device).view(1, -1, 1)) & (
            shift.abs().sum(-1) == 0
        )
        inside = inside & ~same
        index = inside.nonzero(as_tuple=False)
        firsts.append(rows[index[:, 0]])
        seconds.append(index[:, 1])
        shifts.append(shift[index[:, 0], index[:, 1], index[:, 2]])
    return PairList(
        first=torch.cat(firsts), second=torch.cat(seconds), shift=torch.cat(shifts).to(pos.dtype)
    )


def _gamma_factors(r: Tensor, width: Tensor, alpha: float, orders) -> list:
    """``P(n+1/2, r^2/s^2) - P(n+1/2, alpha^2 r^2)`` for each n in ``orders``."""
    smeared = (r / width) ** 2
    screened = (alpha * r) ** 2
    return [
        torch.special.gammainc(torch.full_like(r, n + 0.5), smeared)
        - torch.special.gammainc(torch.full_like(r, n + 0.5), screened)
        for n in orders
    ]


def _reciprocal_chunk(
    pos: Tensor, q: Tensor, mu: Tensor, theta: Tensor, k: Tensor, w: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """``(phi, field, grad)`` from one chunk of wave vectors (SPEC 16).

    ``S(k) = sum_j e^{-ik.r_j} (q_j - i k.mu_j - (1/2) k.Th_j.k)``;
    ``phi_i = sum_k w Re[e^{ik.r_i} S]``, ``F_i = sum_k w Re[i k e^{ik.r_i} S]``,
    ``G_i = sum_k w Re[-(1/2) k k e^{ik.r_i} S]`` (cartesian; projected by the
    caller).
    """
    phase = pos @ k.T
    cos, sin = torch.cos(phase), torch.sin(phase)
    kthk = torch.einsum("ka,iab,kb->ik", k, theta, k)
    real = q.unsqueeze(-1) - 0.5 * kthk
    imag = -(mu @ k.T)
    s_real = (cos * real + sin * imag).sum(0)
    s_imag = (cos * imag - sin * real).sum(0)
    re = cos * s_real - sin * s_imag
    im = sin * s_real + cos * s_imag
    phi = re @ w
    field = (-(im * w)) @ k
    grad = -0.5 * torch.einsum("ik,ka,kb->iab", re * w, k, k)
    return phi, field, grad


def _thole_minus_screen(r: Tensor, damping: Tensor, alpha: float, order: int) -> Tensor:
    return damping - torch.special.gammainc(torch.full_like(r, order + 0.5), (alpha * r) ** 2)


class LatticeOperator:
    """``K_pair v`` for the periodic multipole operator, without any ``n x n`` object.

    Acts on PHYSICAL moments ``(q, mu, Theta)`` -- ``Theta`` as its orthonormal
    ``2e`` coefficients -- and returns the potentials ``dU_pair/dq``,
    ``dU_pair/dmu``, ``dU_pair/dTheta``.  :meth:`apply_reduced` wraps it in the
    reduced-moment convention of the solver (``mu = ell mu_hat``,
    ``Theta = ell^2 Theta_hat``).

    The pair part only, like every other pair operator in the package: the
    on-site ``eta``, ``A_mu``, ``A_Theta`` are added by the caller.

    Kernels: ``"gaussian"`` (SPEC 12/14, widths per site and rank) or
    ``"thole"`` (the Thole Ewald of the dense periodic route; ``alpha_volume``
    and ``thole_a`` required, quadrupoles not supported).
    """

    def __init__(
        self,
        pos: Tensor,
        cell: Tensor,
        ewald: EwaldParams,
        pairs: PairList,
        *,
        sigma: Tensor,
        kernel: str,
        sigma_mu: Optional[Tensor] = None,
        sigma_theta: Optional[Tensor] = None,
        alpha_volume: Optional[Tensor] = None,
        thole_a: Optional[float] = None,
        k_chunk: int = 4096,
        gram_background: bool = True,
        reciprocal: str = "ewald",
        pme_order: int = 8,
    ) -> None:
        if reciprocal not in ("ewald", "pme"):
            raise ValueError(f"reciprocal must be 'ewald' or 'pme', got {reciprocal!r}")
        if kernel not in ("gaussian", "thole"):
            raise ValueError(f"kernel must be 'gaussian' or 'thole', got {kernel!r}")
        if kernel == "gaussian" and sigma_mu is None:
            raise ValueError("the Gaussian kernel needs sigma_mu")
        if kernel == "thole" and (alpha_volume is None or thole_a is None):
            raise ValueError("the Thole kernel needs alpha_volume and thole_a")
        if kernel == "thole" and sigma_theta is not None:
            raise ValueError("quadrupoles need the Gaussian kernel (SPEC 12)")
        self.pos, self.cell, self.ewald, self.kernel = pos, cell, ewald, kernel
        # Everything needed to rebuild this operator from new tensors (SPEC 17:
        # the implicit derivative takes the operator's explicit dependence on
        # its inputs through a pure factory, see :meth:`rebuilt`).
        self._pairs = pairs
        self._settings = dict(kernel=kernel, thole_a=thole_a, k_chunk=k_chunk,
                              gram_background=gram_background, reciprocal=reciprocal,
                              pme_order=pme_order)
        self.reciprocal = reciprocal
        self.sigma_mu, self.sigma_theta, self.alpha_volume = sigma_mu, sigma_theta, alpha_volume
        self.alpha = float(ewald.alpha)
        self.quadrupolar = sigma_theta is not None
        self.sigma = sigma
        self.gram_background = gram_background and kernel == "gaussian"
        self.k_chunk = int(k_chunk)
        dtype = pos.dtype
        self.n_atoms = int(pos.shape[0])
        self.volume = torch.det(cell).abs()
        alpha = self.alpha

        # ---- real space: per-pair scalars ----------------------------------
        i, j = pairs.first, pairs.second
        r_vec = pairs.vectors(pos, cell)
        squared = (r_vec * r_vec).sum(-1)
        keep = squared > DEGENERATE_SEPARATION * DEGENERATE_SEPARATION
        r = torch.sqrt(torch.where(keep, squared, torch.ones_like(squared)))
        keep_f = keep.to(dtype)
        self.i, self.j, self.r_vec = i, j, r_vec
        inv_r3 = 1.0 / (r * r * r)
        inv_r5 = inv_r3 / (r * r)
        inv_r7 = inv_r5 / (r * r)
        inv_r9 = inv_r7 / (r * r)

        def width(a: Tensor, b: Tensor) -> Tensor:
            return torch.sqrt(a * a + b * b)

        s00 = width(sigma[i], sigma[j])
        screen = torch.erf(alpha * r) / r
        self.c_qq = keep_f * K_E * (torch.erf(r / s00) / r - screen)

        if kernel == "gaussian":
            l3_ij = _gamma_factors(r, width(sigma[i], sigma_mu[j]), alpha, (1,))[0]
            l3_ji = _gamma_factors(r, width(sigma[j], sigma_mu[i]), alpha, (1,))[0]
            l3_mm, l5_mm = _gamma_factors(r, width(sigma_mu[i], sigma_mu[j]), alpha, (1, 2))
        else:
            from .kernels import thole_damping

            t3, t5 = thole_damping(r, alpha_volume[i], alpha_volume[j], thole_a)
            l3_ij = l3_ji = l3_mm = _thole_minus_screen(r, t3, alpha, 1)
            l5_mm = _thole_minus_screen(r, t5, alpha, 2)
        # q_i <- mu_j : phi_i += K_E l3/r^3 (r . mu_j)
        self.c_qm = keep_f * K_E * l3_ij * inv_r3
        # mu_i <- q_j : F_i += -K_E l3'/r^3 r q_j   (width of q_j with mu_i)
        self.c_mq = keep_f * K_E * l3_ji * inv_r3
        # mu_i <- mu_j : F_i += K_E (l3/r^3 mu_j - 3 l5/r^5 r (r.mu_j))
        self.a_mm = keep_f * K_E * l3_mm * inv_r3
        self.b_mm = keep_f * 3.0 * K_E * l5_mm * inv_r5

        if self.quadrupolar:
            l5_q2 = _gamma_factors(r, width(sigma[i], sigma_theta[j]), alpha, (2,))[0]
            l5_2q = _gamma_factors(r, width(sigma[j], sigma_theta[i]), alpha, (2,))[0]
            _, l5_m2, l7_m2 = _gamma_factors(r, width(sigma_mu[i], sigma_theta[j]), alpha, (1, 2, 3))
            _, l5_2m, l7_2m = _gamma_factors(r, width(sigma_mu[j], sigma_theta[i]), alpha, (1, 2, 3))
            _, l5_22, l7_22, l9_22 = _gamma_factors(
                r, width(sigma_theta[i], sigma_theta[j]), alpha, (1, 2, 3, 4)
            )
            half = 0.5 * K_E
            self.c_q2 = keep_f * 3.0 * half * l5_q2 * inv_r5          # phi_i <- Th_j
            self.c_2q = keep_f * 3.0 * half * l5_2q * inv_r5          # G_i <- q_j
            self.a_m2 = keep_f * half * 15.0 * l7_m2 * inv_r7         # F_i <- Th_j
            self.b_m2 = keep_f * half * 6.0 * l5_m2 * inv_r5
            self.a_2m = keep_f * half * 15.0 * l7_2m * inv_r7         # G_i <- mu_j
            self.b_2m = keep_f * half * 3.0 * l5_2m * inv_r5
            quarter = 0.25 * K_E
            self.a_22 = keep_f * quarter * 105.0 * l9_22 * inv_r9     # G_i <- Th_j
            self.b_22 = keep_f * quarter * 15.0 * l7_22 * inv_r7
            self.c_22 = keep_f * quarter * 6.0 * l5_22 * inv_r5
            self.basis = quadrupole_cartesian_basis().to(dtype=dtype, device=pos.device)

        # ---- reciprocal space ------------------------------------------------
        self._pme = None
        if reciprocal == "pme":
            # SPEC 19: smooth particle-mesh Ewald, the same point-multipole
            # reciprocal sum on a mesh sized for the requested accuracy.
            from .pme import PMEReciprocal, pme_mesh

            mesh = pme_mesh(
                cell.detach(), alpha, float(ewald.accuracy), int(pme_order),
                max_rank=2 if self.quadrupolar else 1,
            )
            self._pme = PMEReciprocal(
                pos, cell, alpha, mesh, int(pme_order), quadrupoles=self.quadrupolar
            )
            self.pme_mesh = mesh
        # Half of each +-k pair (the sum is real: the -k term is the complex
        # conjugate), doubled, and only inside the cutoff sphere k <= k_cut =
        # 2 alpha sqrt(-ln accuracy) that the grid was tuned for: the corners of
        # the integer cube carry exp(-k^2/4 alpha^2) < accuracy.  ~4x fewer
        # wave vectors than the dense route's full cube, same sum to accuracy.
        k, k_squared, nonzero = _wave_vectors(ewald.k_grid, cell)
        weight = _gaussian_weights(k_squared, nonzero, alpha)
        integer = torch.round(k.detach() @ cell.detach().T / (2.0 * math.pi))
        half = (integer[:, 0] > 0) | (
            (integer[:, 0] == 0) & ((integer[:, 1] > 0) | ((integer[:, 1] == 0) & (integer[:, 2] > 0)))
        )
        k_cut = 2.0 * alpha * math.sqrt(-math.log(float(ewald.accuracy)))
        inside = k_squared.detach() <= k_cut * k_cut * (1.0 + 1e-12)
        keep_k = nonzero & half & inside
        self.k = k[keep_k]
        self.weight = 2.0 * (4.0 * math.pi * K_E / self.volume) * weight[keep_k]

        # ---- self terms ------------------------------------------------------
        self.self_charge = -2.0 * K_E * alpha / SQRT_PI
        self.self_dipole = -4.0 * K_E * alpha**3 / (3.0 * SQRT_PI)
        self.self_quadrupole = -4.0 * K_E * alpha**5 / (5.0 * SQRT_PI)

    # ------------------------------------------------------------------------
    def inputs(self) -> Tuple[Optional[Tensor], ...]:
        """The tensors this operator is a function of, in :meth:`rebuilt` order."""
        return (self.pos, self.cell, self.sigma, self.sigma_mu, self.sigma_theta, self.alpha_volume)

    def rebuilt(self, pos, cell, sigma, sigma_mu, sigma_theta, alpha_volume) -> "LatticeOperator":
        """The same operator (pairs, grid, kernel, settings) on new input tensors."""
        return LatticeOperator(
            pos, cell, self.ewald, self._pairs, sigma=sigma, sigma_mu=sigma_mu,
            sigma_theta=sigma_theta, alpha_volume=alpha_volume, **self._settings,
        )

    def _cartesian(self, theta: Tensor) -> Tensor:
        return torch.einsum("im,mab->iab", theta, self.basis)

    def _project(self, matrix: Tensor) -> Tensor:
        return torch.einsum("iab,mab->im", matrix, self.basis)

    def apply(
        self, q: Tensor, mu: Tensor, theta: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """``(dU/dq, dU/dmu, dU/dTheta)`` of the pair energy, physical moments."""
        n = self.n_atoms
        i, j, rv = self.i, self.j, self.r_vec
        q_j, mu_j = q.index_select(0, j), mu.index_select(0, j)
        r_dot_mu = (rv * mu_j).sum(-1)

        phi_pairs = self.c_qq * q_j + self.c_qm * r_dot_mu
        field_pairs = (
            (-self.c_mq * q_j + (-self.b_mm) * r_dot_mu).unsqueeze(-1) * rv
            + self.a_mm.unsqueeze(-1) * mu_j
        )
        grad_pairs = None
        if self.quadrupolar and theta is not None:
            th_j = self._cartesian(theta).index_select(0, j)                 # [P,3,3]
            th_r = torch.einsum("pab,pb->pa", th_j, rv)
            r_th_r = (rv * th_r).sum(-1)
            phi_pairs = phi_pairs + self.c_q2 * r_th_r
            field_pairs = field_pairs + (-self.a_m2 * r_th_r).unsqueeze(-1) * rv + self.b_m2.unsqueeze(-1) * th_r
            outer = rv.unsqueeze(-1) * rv.unsqueeze(-2)
            grad_matrix = (self.c_2q * q_j).view(-1, 1, 1) * outer
            # G_i <- mu_j: -(K_E/2)[-15 l7/r^7 (mu.r) r r + 3 l5/r^5 (mu r + r mu)]
            mu_r = mu_j.unsqueeze(-1) * rv.unsqueeze(-2)
            grad_matrix = grad_matrix + (self.a_2m * r_dot_mu).view(-1, 1, 1) * outer
            grad_matrix = grad_matrix - self.b_2m.view(-1, 1, 1) * (mu_r + mu_r.transpose(-1, -2))
            # G_i <- Th_j (traceless part)
            thr_r = th_r.unsqueeze(-1) * rv.unsqueeze(-2)
            grad_matrix = grad_matrix + (self.a_22 * r_th_r).view(-1, 1, 1) * outer
            grad_matrix = grad_matrix - 2.0 * self.b_22.view(-1, 1, 1) * (thr_r + thr_r.transpose(-1, -2))
            grad_matrix = grad_matrix + self.c_22.view(-1, 1, 1) * th_j
            grad_pairs = grad_matrix

        phi = torch.zeros(n, dtype=q.dtype, device=q.device).index_add(0, i, phi_pairs)
        field = torch.zeros(n, 3, dtype=q.dtype, device=q.device).index_add(0, i, field_pairs)
        grad = None
        if grad_pairs is not None:
            grad = self._project(
                torch.zeros(n, 3, 3, dtype=q.dtype, device=q.device).index_add(0, i, grad_pairs)
            )

        # reciprocal space, chunked over k and recomputed in backward: autograd
        # would otherwise keep every chunk's [n, chunk] phases for every product
        # of the CG, which is what dominated memory (5 GB at 432 atoms).
        th_c = (
            self._cartesian(theta)
            if (self.quadrupolar and theta is not None)
            else torch.zeros(n, 3, 3, dtype=q.dtype, device=q.device)
        )
        with_grad = grad is not None
        if self._pme is not None:
            d_phi, d_field, d_grad = self._pme.apply(
                q, mu, th_c if self.quadrupolar else None
            )
            phi = phi + d_phi
            field = field + d_field
            if with_grad:
                grad = grad + self._project(d_grad)
        for start in range(0, 0 if self._pme is not None else int(self.k.shape[0]), self.k_chunk):
            k = self.k[start : start + self.k_chunk]
            w = self.weight[start : start + self.k_chunk]
            d_phi, d_field, d_grad = checkpoint(
                _reciprocal_chunk, self.pos, q, mu, th_c, k, w, use_reentrant=False
            )
            phi = phi + d_phi
            field = field + d_field
            if with_grad:
                grad = grad + self._project(d_grad)

        # self and background
        phi = phi + self.self_charge * q
        total = q.sum()
        phi = phi - (math.pi * K_E / self.volume) * (1.0 / self.alpha**2) * total
        if self.gram_background:
            sigma_sq = self.sigma * self.sigma
            phi = phi + (math.pi * K_E / self.volume) * (sigma_sq * total + (sigma_sq * q).sum())
        field = field + self.self_dipole * mu
        if grad is not None:
            grad = grad + self.self_quadrupole * theta
        return phi, field, grad

    def apply_reduced(
        self, q: Tensor, mu_hat: Tensor, ell: Tensor, theta_hat: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """The same product in the solver's reduced moments."""
        mu = ell.unsqueeze(-1) * mu_hat
        theta = None if theta_hat is None else (ell * ell).unsqueeze(-1) * theta_hat
        phi, field, grad = self.apply(q, mu, theta)
        field_hat = ell.unsqueeze(-1) * field
        grad_hat = None if grad is None else (ell * ell).unsqueeze(-1) * grad
        return phi, field_hat, grad_hat


# ---------------------------------------------------------------------------
# The matrix-free fixed-k solve (SPEC 16)
# ---------------------------------------------------------------------------


class _BondMaps:
    """``B diag(f)`` and its transpose as scatter / gather over ``(lo, hi, f)``.

    Replaces the dense ``[n_atoms, n_bonds]`` incidence: a self-image bond has
    ``lo == hi`` and its two contributions cancel exactly, which is its zero
    column.
    """

    def __init__(self, bond_index: Tensor, envelope: Tensor, n_atoms: int) -> None:
        self.lo, self.hi, self.f, self.n_atoms = bond_index[0], bond_index[1], envelope, n_atoms

    def charges(self, tau: Tensor) -> Tensor:
        flow = self.f * tau
        out = torch.zeros(self.n_atoms, dtype=tau.dtype, device=tau.device)
        return out.index_add(0, self.lo, flow).index_add(0, self.hi, -flow)

    def transpose(self, potential: Tensor) -> Tensor:
        return self.f * (potential.index_select(0, self.lo) - potential.index_select(0, self.hi))


#: Largest atom count for which the matrix-free preconditioner factorises its
#: dense ``[n, n]`` atom-space matrix (8000 atoms: 0.5 GB, one Cholesky per
#: solve).  Above it the bond sector is left unpreconditioned, which measured
#: better than the bond diagonal (SPEC 17.6).
WOODBURY_MAX_ATOMS = 8000

#: Largest atom count for which the near-field real-space Coulomb kernel is
#: added to the matrix-free preconditioner's atom-space factor (SPEC 17.6).
NEAR_FIELD_MAX_ATOMS = 2048


def _woodbury_bond_inverse(maps, bond_index, envelope, kappa, onsite, n_atoms, near_field=None):
    """``r -> (kappa + B^T C B)^-1 r`` by Woodbury, matrix free, detached.

    ``C = diag(onsite) [+ near_field]``: with ``near_field`` (a dense ``[n, n]``
    real-space charge kernel from the operator's pair list) the charge sector's
    short-range Coulomb coupling is in the preconditioner too.  ``B diag(f)``
    enters only through the scatter/gather maps; the atom-space matrix
    ``C^-1 + B kappa^-1 B^T`` (self-image bonds have zero columns and drop out)
    is factorised once.  ``1^T B = 0``, so ``C + s 11^T`` gives the same
    preconditioner and makes a non-definite ``C`` invertible; failing that the
    diagonal is used.
    """
    if int(bond_index.shape[1]) == 0 or n_atoms > WOODBURY_MAX_ATOMS:
        return lambda r: r
    lo, hi = bond_index[0], bond_index[1]
    weight = torch.where(lo == hi, torch.zeros_like(envelope), envelope * envelope / kappa)
    c_inverse = None
    if near_field is not None:
        C = torch.diag(onsite) + 0.5 * (near_field + near_field.transpose(0, 1))
        shift = float(torch.diagonal(C).abs().max()) + 1.0
        for trial in (C, C + (shift / n_atoms) * torch.ones_like(C)):
            factor_c, info = torch.linalg.cholesky_ex(trial)
            if int(info) == 0:
                c_inverse = torch.cholesky_inverse(factor_c)
                break
    atom_space = torch.diag(1.0 / onsite) if c_inverse is None else c_inverse
    atom_space = atom_space.index_put((lo, lo), weight, accumulate=True)
    atom_space = atom_space.index_put((hi, hi), weight, accumulate=True)
    atom_space = atom_space.index_put((lo, hi), -weight, accumulate=True)
    atom_space = atom_space.index_put((hi, lo), -weight, accumulate=True)
    factor, info = torch.linalg.cholesky_ex(0.5 * (atom_space + atom_space.transpose(0, 1)))
    if int(info) != 0:
        return lambda r: r / kappa

    def apply(r: Tensor) -> Tensor:
        scaled = r / kappa
        correction = torch.cholesky_solve(maps.charges(scaled).unsqueeze(-1), factor).squeeze(-1)
        return scaled - maps.transpose(correction) / kappa

    return apply


def _lattice_product(
    op: "LatticeOperator",
    maps: "_BondMaps",
    ell: Tensor,
    eta: Tensor,
    kappa: Tensor,
    A_mu: Tensor,
    A_theta: Optional[Tensor],
    j_exchange: Optional[Tensor],
    kappa_spin: Optional[Tensor],
    n_bonds: int,
    n_atoms: int,
):
    """``v -> K v`` of the full master-equation operator, matrix free.

    ``v = (tau, [tau_s], mu_hat, [Theta_hat])``; ``K`` = bond sector
    ``diag(f) B^T (J + eta) B diag(f) + kappa`` (and its spin twin), the pair
    operator between all moments, and the on-site ``A_mu`` / ``A_Theta``.
    """
    spin = j_exchange is not None
    quad = A_theta is not None
    width = 2 * n_bonds if spin else n_bonds
    dip0, quad0 = width, width + 3 * n_atoms
    ell, eta = ell.reshape(-1), eta.reshape(-1)

    def product(m: Tensor) -> Tensor:
        tau = m[:n_bonds]
        mu_hat = m[dip0:quad0].reshape(n_atoms, 3)
        theta_hat = m[quad0:].reshape(n_atoms, 5) if quad else None
        q = maps.charges(tau)
        phi, field, grad = op.apply_reduced(q, mu_hat, ell, theta_hat)
        parts = [maps.transpose(phi + eta * q) + kappa * tau]
        if spin:
            tau_s = m[n_bonds:width]
            s = maps.charges(tau_s)
            parts.append(maps.transpose(j_exchange.reshape(-1) * s) + kappa_spin * tau_s)
        parts.append((field + torch.einsum("iab,ib->ia", A_mu, mu_hat)).reshape(-1))
        if quad:
            parts.append((grad + torch.einsum("imn,in->im", A_theta, theta_hat)).reshape(-1))
        return torch.cat(parts)

    return product


def solve_lattice(
    operator: LatticeOperator,
    params,
    bond_index: Tensor,
    envelope: Tensor,
    constraints,
    n_iter: int,
    *,
    tau_seed: Tensor,
    options,
    bond_dipole: Tensor,
    external_field: Optional[Tensor] = None,
    charge_field: Optional[Tensor] = None,
):
    """``solver.solve_charges``, matrix free.

    Same variables ``(tau, [tau_s], mu_hat, [Theta_hat])``, same feasible set
    (bonds crossing an explicit fragment boundary pinned, SPEC 17.2), same
    seed steps (preconditioned with ``A_mu`` / ``A_Theta``, SPEC 14.1) and the
    same two energy definitions: the converged minimum with implicit
    derivatives (``options.solve_mode == "converged"``, SPEC 17, default) or
    the legacy ``U(m_k)`` after ``n_iter`` CG steps.  ``K v`` comes from
    :class:`LatticeOperator` and the bond maps.  Periodic assemblies only (the
    field couples through ``bond_dipole``, SPEC 15).
    """
    from .diagnostics import assert_fragment_charges
    from .implicit import QuadraticProblem, minimise
    from .solver import (
        SOLVE_CONVERGED,
        SolveDiagnostics,
        SolveResult,
        _mark_solver_output,
        distribute_fragment_charges,
        distribute_fragment_spins,
    )

    dtype = operator.pos.dtype
    device = operator.pos.device
    n = operator.n_atoms
    n_bonds = int(bond_index.shape[1])
    spin = params.spin_resolved
    quad = params.quadrupole_resolved
    if quad and not operator.quadrupolar:
        raise ValueError("quadrupole parameters with an operator built without sigma_theta")
    converged = options.solve_mode == SOLVE_CONVERGED
    ell = params.ell.reshape(-1)
    eta = params.eta.reshape(-1)
    chi = params.chi.reshape(-1)
    kappa = params.kappa
    width = 2 * n_bonds if spin else n_bonds
    dip0, quad0 = width, width + 3 * n
    total = quad0 + (5 * n if quad else 0)
    j_exchange = params.j_exchange.reshape(-1) if spin else None
    kappa_spin = params.effective_kappa_spin() if spin else None
    A_theta = params.A_Theta if quad else None

    constraint_matrix = constraints.matrix.to(dtype)
    weights = None if constraints.reference_weights is None else constraints.reference_weights.to(dtype)
    rho = distribute_fragment_charges(constraint_matrix, constraints.targets.reshape(-1).to(dtype), weights=weights)

    maps = _BondMaps(bond_index, envelope, n)
    product = _lattice_product(
        operator, maps, ell, eta, kappa, params.A_mu, A_theta, j_exchange, kappa_spin, n_bonds, n
    )

    # ---- linear term and constant, from rho ----------------------------------
    zeros_mu = torch.zeros(n, 3, dtype=dtype, device=device)
    zeros_th = torch.zeros(n, 5, dtype=dtype, device=device) if quad else None
    phi_rho, field_rho, grad_rho = operator.apply_reduced(rho, zeros_mu, ell, zeros_th)
    c_tau = maps.transpose(phi_rho + eta * rho + chi)
    if params.bond_bias is not None:
        c_tau = c_tau + params.bond_bias.reshape(-1).to(dtype)     # SPEC 20
    c_mu = field_rho
    u_const = 0.5 * (rho * (phi_rho + eta * rho)).sum() + (chi * rho).sum()
    if external_field is not None:
        field_vec = external_field.reshape(-1).to(dtype)
        charge_vec = field_vec if charge_field is None else charge_field.reshape(-1).to(dtype)
        c_tau = c_tau - bond_dipole.to(dtype) @ charge_vec
        c_mu = c_mu - ell.unsqueeze(-1) * field_vec
        u_const = u_const - (rho @ operator.pos) @ charge_vec
    parts = [c_tau]
    rho_spin = None
    if spin:
        rho_spin = distribute_fragment_spins(constraint_matrix, constraints.spin_targets.reshape(-1).to(dtype), weights=weights)
        exchange = params.j_exchange.reshape(-1)
        chi_spin = torch.zeros(n, dtype=dtype, device=device) if params.chi_spin is None else params.chi_spin.reshape(-1)
        parts.append(maps.transpose(exchange * rho_spin + chi_spin))
        u_const = u_const + 0.5 * (rho_spin * exchange * rho_spin).sum() + (chi_spin * rho_spin).sum()
    parts.append(c_mu.reshape(-1))
    if quad:
        c_th = grad_rho if params.chi_theta is None else grad_rho + params.chi_theta
        parts.append(c_th.reshape(-1))
    c = torch.cat(parts)

    # ---- feasible set: bonds joining two fragments carry nothing (SPEC 17.2) --
    mask = torch.ones(total, dtype=dtype, device=device)
    if int(constraints.matrix.shape[0]) > 1 and n_bonds:
        fragment_of = constraints.matrix.argmax(dim=0)
        crossing = (fragment_of[bond_index[0]] != fragment_of[bond_index[1]]).nonzero(as_tuple=True)[0]
        mask[crossing] = 0.0
        if spin:
            mask[n_bonds + crossing] = 0.0

    def project(v: Tensor) -> Tensor:
        return v * mask

    # ---- seed and the two physics steps -------------------------------------
    tiny = torch.finfo(dtype).tiny
    seed_context = torch.no_grad() if converged else contextlib.nullcontext()
    with seed_context:
        seed_parts = [tau_seed.reshape(-1).to(dtype)]
        if spin:
            seed_parts.append(torch.zeros(n_bonds, dtype=dtype, device=device))
        seed_parts.append(torch.zeros(3 * n, dtype=dtype, device=device))
        if quad:
            seed_parts.append(torch.zeros(5 * n, dtype=dtype, device=device))
        m = project(torch.cat(seed_parts))

        def block_step(m: Tensor, start: int, stop: int, stiffness: Tensor, size: int) -> Tensor:
            gradient = (product(m) + c)[start:stop]
            direction = -torch.linalg.solve(stiffness, gradient.reshape(n, size, 1)).reshape(-1)
            full = torch.zeros(total, dtype=dtype, device=device)
            full = torch.cat([full[:start], direction, full[stop:]])
            slope = gradient @ direction
            curvature = full @ product(full)
            positive = curvature > tiny
            unit = torch.ones_like(curvature)
            step = torch.where(positive, -slope / torch.where(positive, curvature, unit), torch.zeros_like(unit))
            return m + step * full

        if options.relax_dipole_seed and n > 0:
            m = block_step(m, dip0, quad0, params.A_mu.to(dtype), 3)
        if quad and options.relax_quadrupole_seed and n > 0:
            m = block_step(m, quad0, total, params.A_Theta.to(dtype), 5)

    if converged:
        # The operator as a pure function of every tensor it reads, so the
        # implicit derivative sees each explicit dependence (SPEC 17).
        base_inputs = operator.inputs()
        present = [index for index, tensor in enumerate(base_inputs) if tensor is not None]
        site = [ell, eta, kappa, envelope, params.A_mu]
        if quad:
            site.append(A_theta)
        if spin:
            site.extend([j_exchange, kappa_spin])
        tensors = tuple(base_inputs[index] for index in present) + tuple(site)

        def factory(T):
            T = list(T)
            rebuilt_inputs = [None] * len(base_inputs)
            for index in present:
                rebuilt_inputs[index] = T.pop(0)
            op = operator.rebuilt(*rebuilt_inputs)
            ell_, eta_, kappa_, envelope_, a_mu_ = T[:5]
            rest = T[5:]
            a_theta_ = rest.pop(0) if quad else None
            jx_, kappa_s_ = (rest[0], rest[1]) if spin else (None, None)
            return _lattice_product(
                op, _BondMaps(bond_index, envelope_, n), ell_, eta_, kappa_, a_mu_,
                a_theta_, jx_, kappa_s_, n_bonds, n,
            )

        with torch.no_grad():
            detached_maps = _BondMaps(bond_index, envelope.detach(), n)
            near_field = None
            # Near-field Coulomb in the factor: 13-16 % fewer iterations
            # (512 / 1728 / 4096 atoms: 57 -> 48, 80 -> 69, 98 -> 85) for one
            # extra n^3 inverse -- time-neutral at 4096 atoms and a net loss
            # above, so only where the inverse is cheap.
            if n <= NEAR_FIELD_MAX_ATOMS and hasattr(operator, "c_qq"):
                # The real-space charge kernel the operator already holds, on
                # its directed pair list (self-image pairs on the diagonal).
                near_field = torch.zeros(n, n, dtype=dtype, device=device).index_put(
                    (operator.i, operator.j), operator.c_qq.detach(), accumulate=True
                )
            bond_inverse = [
                _woodbury_bond_inverse(detached_maps, bond_index, envelope.detach(), kappa.detach(),
                                       eta.detach(), n, near_field)
            ]
            if spin:
                bond_inverse.append(
                    _woodbury_bond_inverse(
                        detached_maps, bond_index, envelope.detach(), kappa_spin.detach(),
                        j_exchange.detach(), n,
                    )
                )
            a_mu_inv = torch.linalg.inv(params.A_mu.detach())
            a_th_inv = torch.linalg.inv(A_theta.detach()) if quad else None

        def precondition(v: Tensor) -> Tensor:
            pieces = [inverse(v[k * n_bonds:(k + 1) * n_bonds]) for k, inverse in enumerate(bond_inverse)]
            pieces.append(torch.einsum("iab,ib->ia", a_mu_inv, v[dip0:quad0].reshape(n, 3)).reshape(-1))
            if quad:
                pieces.append(torch.einsum("imn,in->im", a_th_inv, v[quad0:].reshape(n, 5)).reshape(-1))
            return torch.cat(pieces)

        problem = QuadraticProblem(
            factory, tensors, mask=mask, precondition=precondition,
            rtol=options.solve_rtol, max_iter=options.solve_max_iter, name="solve_lattice",
        )
        m, quadratic = minimise(problem, c, m.detach())
        energy = quadratic + u_const
        linear = (c * m).sum()
        quadratic_part = quadratic - linear
        n_done = int(problem.iterations)
        initial_residual_norm = problem.rhs_norm
        residual_norm = problem.residual_norm
    else:
        # ---- legacy fixed-k projected CG (SPEC 2) ----------------------------
        residual = project(-(product(m) + c))
        direction = residual
        residual_sq = residual @ residual
        initial_residual_norm = torch.sqrt(residual_sq.clamp_min(tiny))
        freeze_floor = (torch.finfo(dtype).eps ** 2) * residual_sq.detach()
        unit = torch.ones_like(residual_sq)
        for _ in range(n_iter):
            active = residual_sq > freeze_floor
            safe_residual_sq = torch.where(active, residual_sq, unit)
            curvature_vector = product(direction)
            curvature = direction @ curvature_vector
            safe_curvature = torch.where(active, curvature + tiny, unit)
            step = torch.where(active, safe_residual_sq / safe_curvature, torch.zeros_like(unit))
            m = m + step * direction
            residual = project(residual - step * curvature_vector)
            residual_sq_new = residual @ residual
            beta = torch.where(active, residual_sq_new / (safe_residual_sq + tiny), torch.zeros_like(unit))
            direction = residual + beta * direction
            residual_sq = residual_sq_new
        Km = product(m)
        quadratic_part = 0.5 * (m * Km).sum()
        linear = (c * m).sum()
        energy = quadratic_part + linear + u_const
        n_done = n_iter
        residual_norm = torch.sqrt(residual_sq.clamp_min(tiny))

    tau = m[:n_bonds]
    tau_s = m[n_bonds:width] if spin else None
    mu_hat = m[dip0:quad0].reshape(n, 3)
    theta_hat = m[quad0:].reshape(n, 5) if quad else None
    q = maps.charges(tau) + rho
    mu = ell.unsqueeze(-1) * mu_hat
    spin_population = None if not spin else maps.charges(tau_s) + rho_spin
    theta = None
    if quad:
        basis = quadrupole_cartesian_basis().to(dtype=dtype, device=device)
        theta = torch.einsum("im,mab->iab", (ell * ell).unsqueeze(-1) * theta_hat, basis)
    violation = assert_fragment_charges(q, constraint_matrix, constraints.targets.reshape(-1).to(dtype),
                                        tol=options.fragment_charge_tol)
    diagnostics = SolveDiagnostics(
        n_iter=n_done,
        seed_provenance="matrix-free lattice solve (geometry-determined seed)",
        initial_residual_norm=initial_residual_norm,
        residual_norm=residual_norm,
        constraint_violation=violation,
        energy_terms={
            "quadratic": quadratic_part,
            "linear": linear,
            "reference": u_const,
            "bond_hardness_in_quadratic": 0.5 * (kappa * tau * tau).sum(),
        },
        system=None,
    )
    _mark_solver_output(m, tau, mu_hat, mu, q)
    return SolveResult(
        t=tau, rho=rho, mu=mu, q=q, energy=energy, diagnostics=diagnostics,
        t_spin=tau_s, rho_spin=rho_spin, s=spin_population,
        theta_hat=theta_hat, theta=theta,
    )
