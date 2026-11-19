"""The periodic Gram operator: Gaussian multipoles summed over the lattice (SPEC 14).

SPEC 12 made every block of K the Coulomb inner product of two Gaussian
multipole densities, so that K is positive definite for every open-boundary
geometry.  This module is the same statement for a crystal.

The lattice Gram form
---------------------
Give site ``i`` a Gaussian of width ``sigma_{a,i}`` for each rank ``a`` (charge,
dipole, quadrupole).  The cell's charge density has the Fourier coefficients

    rho(k) = sum_j exp(-i k.r_j) sum_a g_{a,j}(k) c_a(k) . m_{a,j},
    g_{a,j}(k) = exp(-k^2 sigma_{a,j}^2 / 4),
    c_0 = 1,   c_1 = -i k,   c_2 = -(1/2) k k          (Theta the traceless
                                                        second moment, kernels.py)

and, under tinfoil boundaries with the k = 0 component removed (the
neutralising background; the plane-wave convention of every DFT code),

    U = (K_E / 2) (4 pi / Omega) sum_{k != 0} |rho(k)|^2 / k^2   >=  0.

A sum of non-negative terms: the lattice interaction matrix, self blocks
included, is positive SEMIdefinite for every crystal.  Subtracting the
same-image self block ``S_a`` and adding the on-site stiffnesses gives

    K_lattice = M^T (Gram_lattice + D) M + diag(kappa, 0, 0),
    D = blockdiag(eta - S_0, A_mu - ell^2 S_1, A_Theta - ell^4 S_2)

with EXACTLY the floors of SPEC 12 -- nothing about them depends on the
boundary condition.  The PD theorem survives the lattice sum unchanged.

The Ewald split (standard, point-multipole reciprocal space)
-------------------------------------------------------------
With the splitting parameter ``alpha``, per rank pair ``(a, b)`` of combined
width ``s = sqrt(sigma_{a,i}^2 + sigma_{b,j}^2)``:

* real space, over every image except the same-site zero shift, the kernel
  ``K_E [erf(r/s) - erf(alpha r)] / r``; its damping factors are differences
  of regularised incomplete gammas,

      lambda_{2n+1}(r) = P(n + 1/2, r^2/s^2) - P(n + 1/2, alpha^2 r^2),

  which drop into the existing tensors of ``kernels.py`` unchanged;
* reciprocal space, the point-multipole Ewald sum with weight
  ``(4 pi K_E / Omega) exp(-k^2/4 alpha^2) / k^2`` and the block factors

      (0,0) cos,  (0,1) k_b sin,  (1,1) k_a k_b cos,
      (0,2) -(1/2)(k.W_m.k) cos,  (1,2) (1/2) k_a (k.W_m.k) sin,
      (2,2) (1/4)(k.W_m.k)(k.W_n.k) cos          (arguments k.(r_i - r_j));

* the self term of the alpha-Gaussian, removed: ``-2 K_E alpha/sqrt(pi)``
  (charge), ``-(4/3) K_E alpha^3/sqrt(pi)`` (dipole),
  ``-(4/5) K_E alpha^5/sqrt(pi)`` (quadrupole, orthonormal 2e basis);
* the k = 0 Fourier component of the real-space kernel, removed:
  ``-(pi K_E / Omega)(1/alpha^2 - s_ij^2)`` on the charge block.  The
  ``1/alpha^2`` part is the usual jellium term; the ``s_ij^2`` part completes
  the Gram (plane-wave) convention and sums to zero for a neutral cell,
  ``sum_ij q_i q_j (sigma_i^2 + sigma_j^2) = 2 Q sum_i q_i sigma_i^2``.

Established, cited: Ewald summation of Gaussian multipoles is the pGM model's
(Wei, Qi, Wang, Cieplak, Duan, Luo, J. Chem. Phys. 153, 114116 (2020)); point
multipole Ewald is textbook.  TRACE-native: per-rank widths DERIVED from the
learned stiffnesses (SPEC 12), so the lattice Hessian of the bond-SQE +
dipole + quadrupole model is positive definite by construction, for every
crystal, with no runtime check.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor

from .special import half_integer_regularized_gamma
from .constants import K_E, SQRT_PI
from .ewald import EwaldParams
from .ewald import _image_offsets, _image_shell_counts
from .ewald import _pair_geometry as _ewald_pair_geometry
from .ewald import _half_wave_vectors as _ewald_half_wave_vectors
from .ewald import _gaussian_weights as _ewald_gaussian_weights
from .kernels import (
    charge_dipole_tensor,
    charge_quadrupole_tensor,
    coulomb_kernel,
    dipole_interaction_tensor,
    dipole_quadrupole_tensor,
    quadrupole_cartesian_basis,
    quadrupole_interaction_tensor,
)

__all__ = ["LatticeWidthError", "free_charge_lattice_blocks", "gaussian_lattice_blocks", "real_space_damping"]

_QUADRUPOLE_DIM = 5


class LatticeWidthError(ValueError):
    """A Gaussian is too wide for the real-space image box to converge its tail."""


def real_space_damping(
    r: Tensor, width: Tensor, alpha: float, orders: tuple = (1, 2, 3, 4), mask: Optional[Tensor] = None
) -> tuple:
    """``lambda_{2n+1} = P(n+1/2, r^2/s^2) - P(n+1/2, alpha^2 r^2)`` for n in ``orders`` (default 1..4).

    The damping factors of ``K_E [erf(r/s) - erf(alpha r)]/r``: each term is the
    Gaussian factor of :func:`~trace_electrostatics.kernels.gaussian_damping`
    at its own width, so the difference is the exact real-space part of the
    split.  Both terms are ``O(r^(2n+1))`` at small ``r``, so the kernels'
    ``lambda / r^(2n+1)`` stays finite.

    P is :func:`~trace_electrostatics.special.half_integer_regularized_gamma`
    (closed form, within 32 eps of 50-digit mpmath), not
    ``torch.special.gammainc`` (up to 1.8e-14 off, and several times slower).
    Only the requested ``orders`` are computed.  ``mask`` (broadcastable to ``r``)
    restricts the work to the pair-images it selects; the others are returned
    as exact zeros, which their consumers multiply by ``keep = 0`` anyway.
    """
    smeared = (r / width) ** 2
    screened = (alpha * r) ** 2
    if mask is None:
        p_s = half_integer_regularized_gamma(smeared, orders)
        p_c = half_integer_regularized_gamma(screened, orders)
        return tuple(a - b for a, b in zip(p_s, p_c))
    mask = mask.expand_as(smeared)
    p_s = half_integer_regularized_gamma(smeared[mask], orders)
    p_c = half_integer_regularized_gamma(screened.expand_as(smeared)[mask], orders)
    zero = torch.zeros_like(smeared)
    return tuple(zero.masked_scatter(mask, a - b) for a, b in zip(p_s, p_c))


def _reciprocal_terms(k_grid, cell: Tensor):
    """``(k, k_squared, nonzero, multiplicity)`` of the reciprocal sums: one of each pair ``{k, -k}``, twice.

    Every reciprocal block of :func:`gaussian_lattice_blocks` sums ``w(k) F(k) g(k . r_ij)`` with
    ``w(-k) = w(k)`` and an even ``F`` (1, k k, k.W.k, (k.W.k)(k.W.k)) under ``cos`` or an odd one
    (k, k (k.W.k)) under ``sin``, so the summand is invariant under ``k -> -k`` and the sum over
    all ``k != 0`` is exactly twice the sum over the half-space (speed-up R1, 2026-10-09;
    tests/electrostatics/test_lattice_half_space.py).  Half the work in the forward pass, the
    backward pass and the second-order backward of force training.
    """
    k, k_squared = _ewald_half_wave_vectors(k_grid, cell)
    return k, k_squared, torch.ones_like(k_squared, dtype=torch.bool), 2.0


#: Every real-space term is dropped below this, relative to K_E: 2^-70 K_E ~ 1.2e-20 eV per block
#: entry, summed over all dropped images -- far below one unit in the last place of any entry.
_ROUND_OFF_TARGET = 2.0 ** -70
#: Rows of a pair list processed at once by the [m, 3, 3, 3, 3] quadrupole-quadrupole kernel.
_QUADRUPOLE_CHUNK = 16384


def _upper_gamma_tail_bound(a: float, x: float) -> float:
    """Upper bound on the regularized upper incomplete gamma ``Q(a, x)``, for ``x > max(a - 1, 0)``.

    ``Gamma(a, x) = x^(a-1) e^(-x) int_0^inf (1 + t/x)^(a-1) e^(-t) dt``.  For ``a <= 1`` the
    factor ``(1 + t/x)^(a-1) <= 1``; for ``a > 1`` it is ``<= e^(t (a-1)/x)``.  Hence
    ``Q(a, x) <= x^(a-1) e^(-x) / (Gamma(a) (1 - max(a - 1, 0) / x))`` (checked against mpmath in
    tests/electrostatics/test_lattice_pair_list.py; it is +inf for x <= a - 1, within 6 % of Q at
    x >= 10 and within 0.5 % at x >= 30 -- the radius is found near x ~ 50).
    """
    shrink = 1.0 - max(a - 1.0, 0.0) / x
    if shrink <= 0.0:
        return math.inf
    return math.exp((a - 1.0) * math.log(x) - x - math.lgamma(a)) / shrink


def _KERNEL_BOUND(quadrupoles: bool) -> float:
    """``C``: bounds every real-space kernel entry by ``C K_E |lambda| / r^p`` (``p >= 1``).

    Charge, charge-dipole and dipole-dipole entries carry coefficients ``<= 1 + 3 = 4``.  The
    quadrupole tensors carry up to ``105`` (the ``lambda_9`` term) on four unit vectors, and a
    contraction with the five-function basis on each side adds at most a factor 9 per side.
    """
    return 105.0 * 81.0 * 10.0 if quadrupoles else 4.0


def _round_off_radius(alpha: float, widest: float, quadrupoles: bool, n_images: int) -> float:
    """The separation ``R0`` beyond which every real-space term together is below one ulp.

    Every damping factor is ``lambda_{2n+1}(r) = Q(n+1/2, alpha^2 r^2) - Q(n+1/2, r^2/s^2)`` (the
    charge kernel is ``(erfc(alpha r) - erfc(r/s)) / r``, i.e. ``n = -1``): both terms lie in
    ``[0, Q(n+1/2, kappa^2 r^2)]`` with ``kappa = min(alpha, 1/widest)``, because ``Q`` is
    decreasing in its second argument.  ``Q`` grows with the order at large ``x``, so the highest
    order present (5/2, or 9/2 with quadrupoles) bounds them all.  ``R0`` is the smallest ``r``
    (>= 1 A, so every ``1/r^p <= 1``) with ``n_images C Q_bound(a, kappa^2 r^2) <= 2^-70``, by
    bisection: a block entry gains at most ``2^-70 K_E`` from all the pair-images beyond it.
    """
    a = 4.5 if quadrupoles else 2.5
    kappa = min(float(alpha), 1.0 / float(widest))
    budget = _ROUND_OFF_TARGET / (n_images * _KERNEL_BOUND(quadrupoles))
    lo, hi = a + 1.0, 4096.0
    if _upper_gamma_tail_bound(a, lo) <= budget:
        return max(1.0, math.sqrt(lo) / kappa)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _upper_gamma_tail_bound(a, mid) <= budget:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-12 * hi:
            break
    return max(1.0, math.sqrt(hi) / kappa)


def _near_pair_geometry(pos: Tensor, cell: Tensor, ewald: EwaldParams, radius: float):
    """The pair-images of the image box within ``radius``, as a list (speed-up R3).

    The box, the minimum-image recentring and the exclusion of the self pair are those of
    :func:`~trace_electrostatics.ewald._pair_geometry`; the selection ``r <= radius`` is made on
    detached positions and is a constant.  Returns ``(i, j, r_vec, r)``: atom indices ``[m]``,
    ``r_vec = r_i - r_j + shift @ cell`` ``[m, 3]`` -- differentiable in ``pos`` and ``cell``,
    so forces and stress follow -- and its length ``[m]``.
    """
    with torch.no_grad():
        detached_pos, detached_cell = pos.detach(), cell.detach()
        _, r_box, included = _ewald_pair_geometry(detached_pos, detached_cell, ewald)
        near = included & (r_box <= radius)
        image, i, j = near.nonzero(as_tuple=True)
        shells = ewald.image_shells
        if shells is None:
            shells = _image_shell_counts(detached_cell, ewald.r_cut)
        offsets = _image_offsets(detached_cell, shells)
        delta = detached_pos.unsqueeze(1) - detached_pos.unsqueeze(0)
        minimum_image = torch.round(delta @ torch.linalg.inv(detached_cell))
        shift = offsets[image] - minimum_image[i, j]
    r_vec = (pos[i] - pos[j]) + shift @ cell
    r = torch.sqrt((r_vec * r_vec).sum(dim=-1))
    return i, j, r_vec, r


def _check_widest(widest: float, ewald: EwaldParams) -> None:
    """Refuse a Gaussian pair too wide for the image box (its tail erfc(r_cut / s) > 10 x accuracy)."""
    limit = 10.0 * float(ewald.accuracy)
    if math.erfc(float(ewald.r_cut) / widest) > limit:
        # the smallest sufficient cutoff, s erfc^-1(limit), by bisection on the monotone erfc
        lo, hi = 0.0, 40.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if math.erfc(mid) > limit else (lo, mid)
        raise LatticeWidthError(
            f"the widest Gaussian pair (s = {widest:.3g} A) leaves a real-space "
            f"tail erfc(r_cut/s) = {math.erfc(float(ewald.r_cut) / widest):.2e} at "
            f"r_cut = {float(ewald.r_cut):.3g} A, above 10 x accuracy = "
            f"{limit:.1e}. The Gaussian itself is that "
            "wide, so no alpha can move it into reciprocal space; raise "
            "ElectrostaticConfig.ewald_real_cutoff to at least "
            f"{math.ceil(100.0 * widest * hi) / 100.0:.2f} A."
        )


def _real_space(pos: Tensor, cell: Tensor, ewald: EwaldParams, widest: float, quadrupoles: bool):
    """The round-off pair list (R3) and its scatter: ``(pair_i, pair_j, pair, r_vec, r, scatter)``."""
    n_atoms = int(pos.shape[0])
    shells = ewald.image_shells if ewald.image_shells is not None else _image_shell_counts(cell, ewald.r_cut)
    n_images = math.prod(2 * s + 1 for s in shells)
    radius = _round_off_radius(float(ewald.alpha), widest, quadrupoles, n_images)
    pair_i, pair_j, r_vec, r = _near_pair_geometry(pos, cell, ewald, radius)    # [m], [m], [m, 3], [m]
    pair = pair_i * n_atoms + pair_j

    def scatter(values: Tensor) -> Tensor:
        """Sum pair-image ``values`` ``[m, ...]`` into the ``[n, n, ...]`` block."""
        flat = values.new_zeros((n_atoms * n_atoms,) + tuple(values.shape[1:]))
        return flat.index_add(0, pair, values).view(n_atoms, n_atoms, *values.shape[1:])

    return pair_i, pair_j, pair, r_vec, r, scatter


def _reciprocal(pos: Tensor, cell: Tensor, ewald: EwaldParams):
    """``(k, k_squared, volume, even, odd)``: the half-space reciprocal sums (R1, R2) as contractions.

    ``even(F) = sum_k w F cos(k.(r_i - r_j))`` and ``odd(F) = sum_k w F sin(k.(r_i - r_j))`` for a
    ``[n_k, ...]`` factor ``F``, ``w = 4 pi K_E / V exp(-k^2 / 4 alpha^2) / k^2`` (twice, for the
    half-space).
    """
    k, k_squared, nonzero, multiplicity = _reciprocal_terms(ewald.k_grid, cell)
    volume = torch.det(cell).abs()
    weight = (multiplicity * 4.0 * math.pi * K_E / volume) * _ewald_gaussian_weights(
        k_squared, nonzero, float(ewald.alpha)
    )
    phase = pos @ k.transpose(0, 1)                                   # [n, n_k]
    cosine, sine = torch.cos(phase), torch.sin(phase)

    def even(factor: Tensor) -> Tensor:
        """``sum_k weight * factor * cos(k.(r_i - r_j))`` for a ``[n_k, ...]`` factor."""
        f = (weight.view(-1, *([1] * (factor.dim() - 1))) * factor)
        return torch.einsum("ik,k...,jk->ij...", cosine, f, cosine) + torch.einsum(
            "ik,k...,jk->ij...", sine, f, sine
        )

    def odd(factor: Tensor) -> Tensor:
        """``sum_k weight * factor * sin(k.(r_i - r_j))``."""
        f = (weight.view(-1, *([1] * (factor.dim() - 1))) * factor)
        return torch.einsum("ik,k...,jk->ij...", sine, f, cosine) - torch.einsum(
            "ik,k...,jk->ij...", cosine, f, sine
        )

    return k, k_squared, volume, even, odd


def _combined(row: Tensor, column: Tensor) -> Tensor:
    """``[1, n, n]`` combined width ``sqrt(row_i^2 + column_j^2)``."""
    return torch.sqrt(row.view(1, -1, 1) ** 2 + column.view(1, 1, -1) ** 2)


def gaussian_lattice_blocks(
    pos: Tensor,
    cell: Tensor,
    sigma: Tensor,
    sigma_mu: Tensor,
    ell: Tensor,
    ewald: EwaldParams,
    *,
    sigma_theta: Optional[Tensor] = None,
    include_charge: bool = True,
) -> Dict[str, Tensor]:
    """Lattice-summed Gaussian multipole pair blocks in REDUCED moments.

    The periodic twin of ``energy._direct_pair_blocks`` (Gaussian kernel) and
    ``solver.quadrupole_pair_tensors``, with the same shapes and ``ell``
    conventions, so every consumer contracts them unchanged:

    ``j [n, n]``, ``t01 [n, n, 3]`` (x ell_j), ``t2 [n, n, 3, 3]``
    (x ell_i ell_j), and with ``sigma_theta``: ``t02 [n, n, 5]`` (x ell_j^2),
    ``t12 [n, n, 3, 5]`` (x ell_i ell_j^2), ``t22 [n, n, 5, 5]``
    (x ell_i^2 ell_j^2).

    As under the Thole route, the DIAGONALS are the interaction of a site with
    its own images, its own screening cloud and the background -- the pair part
    only.  The on-site ``eta``, ``A_mu`` and ``A_Theta`` are added by the
    consumer, exactly as for the open-boundary blocks.

    Raises:
        LatticeWidthError: when the widest Gaussian pair is too wide for the
            image box: its real-space tail ``erfc(r_cut / s_max)`` would exceed
            ten times the requested accuracy.
    """
    dtype, device = pos.dtype, pos.device
    n_atoms = int(pos.shape[0])
    alpha = float(ewald.alpha)
    widths = [sigma, sigma_mu] + ([] if sigma_theta is None else [sigma_theta])
    widest = math.sqrt(2.0) * max(float(w.detach().max()) for w in widths)
    _check_widest(widest, ewald)

    # Real space over the pair-images within the round-off radius only (speed-up R3): the box sum
    # to round-off, at a fraction of the work (tests/electrostatics/test_lattice_pair_list.py).
    pair_i, pair_j, pair, r_vec, r, scatter = _real_space(pos, cell, ewald, widest, sigma_theta is not None)
    screen_width = torch.full_like(r, 1.0 / alpha)

    blocks: Dict[str, Tensor] = {}

    # -- reciprocal space: phases and weights shared by every block ---------
    k, k_squared, volume, even, odd = _reciprocal(pos, cell, ewald)

    identity3 = torch.eye(3, dtype=dtype, device=device)
    eye_n = torch.eye(n_atoms, dtype=dtype, device=device)

    # -- charge-charge -------------------------------------------------------
    if include_charge:
        s00 = _combined(sigma, sigma)
        j_real = scatter(coulomb_kernel(r, s00[0][pair_i, pair_j]) - coulomb_kernel(r, screen_width))
        ones = torch.ones_like(k_squared)
        j = j_real + even(ones)
        j = j - (2.0 * K_E * alpha / SQRT_PI) * eye_n
        j = j - (math.pi * K_E / volume) * (1.0 / alpha**2 - s00[0] ** 2)
        blocks["j"] = j

        s01 = _combined(sigma, sigma_mu)
        (lam3,) = real_space_damping(r, s01[0][pair_i, pair_j], alpha, orders=(1,))
        t01 = scatter(charge_dipole_tensor(r_vec, r, lam3))
        blocks["t01"] = (t01 + odd(k)) * ell.view(1, n_atoms, 1)

    # -- dipole-dipole -------------------------------------------------------
    s11 = _combined(sigma_mu, sigma_mu)
    lam3, lam5 = real_space_damping(r, s11[0][pair_i, pair_j], alpha, orders=(1, 2))
    t2 = scatter(dipole_interaction_tensor(r_vec, r, lam3, lam5))
    t2 = t2 + even(k.unsqueeze(-1) * k.unsqueeze(-2))
    t2 = t2 - (4.0 * K_E * alpha**3 / (3.0 * SQRT_PI)) * (
        eye_n.view(n_atoms, n_atoms, 1, 1) * identity3
    )
    blocks["t2"] = t2 * (ell.view(-1, 1, 1, 1) * ell.view(1, -1, 1, 1))

    if sigma_theta is None:
        return blocks

    # -- quadrupole sector ---------------------------------------------------
    basis = quadrupole_cartesian_basis().to(dtype=dtype, device=device)
    kk = torch.einsum("ka,mab,kb->km", k, basis, k)                  # k.W_m.k, [n_k, 5]
    ell_sq = ell * ell

    if include_charge:
        s02 = _combined(sigma, sigma_theta)
        lam3, lam5 = real_space_damping(r, s02[0][pair_i, pair_j], alpha, orders=(1, 2))
        t02 = scatter(torch.einsum("pab,mab->pm", charge_quadrupole_tensor(r_vec, r, lam3, lam5), basis))
        blocks["t02"] = (t02 - 0.5 * even(kk)) * ell_sq.view(1, n_atoms, 1)

    s12 = _combined(sigma_mu, sigma_theta)
    lam5, lam7 = real_space_damping(r, s12[0][pair_i, pair_j], alpha, orders=(2, 3))
    t12 = scatter(torch.einsum("pabc,mbc->pam", dipole_quadrupole_tensor(r_vec, r, lam5, lam7), basis))
    t12 = t12 + 0.5 * odd(k.unsqueeze(-1) * kk.unsqueeze(1))
    blocks["t12"] = t12 * (ell.view(-1, 1, 1, 1) * ell_sq.view(1, -1, 1, 1))

    # T22 is [m, 3, 3, 3, 3] before contraction: build it in chunks of the pair list so the
    # peak is one chunk's worth.
    s22 = _combined(sigma_theta, sigma_theta)[0][pair_i, pair_j]
    t22 = r.new_zeros(n_atoms * n_atoms, _QUADRUPOLE_DIM, _QUADRUPOLE_DIM)
    for start in range(0, int(r.shape[0]), _QUADRUPOLE_CHUNK):
        rows = slice(start, start + _QUADRUPOLE_CHUNK)
        lam5, lam7, lam9 = real_space_damping(r[rows], s22[rows], alpha, orders=(2, 3, 4))
        tensor = quadrupole_interaction_tensor(r_vec[rows], r[rows], lam5, lam7, lam9)
        t22 = t22.index_add(0, pair[rows], torch.einsum("mab,pabcd,ncd->pmn", basis, tensor, basis))
    t22 = t22.view(n_atoms, n_atoms, _QUADRUPOLE_DIM, _QUADRUPOLE_DIM)
    t22 = t22 + 0.25 * even(kk.unsqueeze(-1) * kk.unsqueeze(1))
    t22 = t22 - (4.0 * K_E * alpha**5 / (5.0 * SQRT_PI)) * (
        eye_n.view(n_atoms, n_atoms, 1, 1)
        * torch.eye(_QUADRUPOLE_DIM, dtype=dtype, device=device)
    )
    blocks["t22"] = t22 * (ell_sq.view(-1, 1, 1, 1) * ell_sq.view(1, -1, 1, 1))
    return blocks


def free_charge_lattice_blocks(
    pos: Tensor,
    cell: Tensor,
    sigma_free: Tensor,
    sigma: Tensor,
    sigma_mu: Tensor,
    ell: Tensor,
    ewald: EwaldParams,
    *,
    sigma_theta: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Lattice-summed blocks between free Gaussian charges and the bound sector, PAIR PARTS.

    For trace_redox's free charges (widths ``sigma_free``) under periodic boundaries, by the Ewald
    split, so the reciprocal sum depends on ``alpha`` alone and compact (ionic) widths cost nothing
    extra -- the plane-wave Gram form needs k up to 12 / s_min (143 687 half-space vectors for
    ionic widths on the Au2/MgO cell, against 1 654 for the hardness widths; measured).

    Returns, in the conventions of :func:`gaussian_lattice_blocks` (reduced moments):

    * ``j`` ``[n, n]``: free-free charge, pair width ``sqrt(sf_i^2 + sf_j^2)``;
    * ``j_fb`` ``[n, n]``: free charge - bound charge, ``sqrt(sf_i^2 + sigma_j^2)``;
    * ``t01`` ``[n, n, 3]``: free charge - bound dipole, ``x ell_j``;
    * ``t02`` ``[n, n, 5]`` (with ``sigma_theta``): free charge - bound quadrupole, ``x ell_j^2``.

    PAIR PARTS: every interaction except the same-image coincident one (``i = j``, zero shift), as
    for ``gaussian_lattice_blocks``'s ``j``: the real-space list excludes it and ``-2 K_E alpha /
    sqrt(pi)`` removes its screened part from the reciprocal sum.  The caller adds
    ``2 K_E / (sqrt(pi) s)`` for the on-site width it wants (zero for ``t01``, ``t02``: an odd
    kernel, and a traceless basis on an isotropic one).  The background term
    ``-(pi K_E / V)(1/alpha^2 - s_ij^2)`` is the k = 0 component removed with the neutralising
    background.  Checked against the plane-wave Gram sum, an independent route, to 1e-10
    (tests/electrostatics/test_free_charge_lattice.py).
    """
    dtype, device = pos.dtype, pos.device
    n_atoms = int(pos.shape[0])
    alpha = float(ewald.alpha)
    widths = [sigma_free, sigma, sigma_mu] + ([] if sigma_theta is None else [sigma_theta])
    widest = math.sqrt(2.0) * max(float(w.detach().max()) for w in widths)
    _check_widest(widest, ewald)
    pair_i, pair_j, pair, r_vec, r, scatter = _real_space(pos, cell, ewald, widest, sigma_theta is not None)
    screen_width = torch.full_like(r, 1.0 / alpha)
    k, k_squared, volume, even, odd = _reciprocal(pos, cell, ewald)
    eye_n = torch.eye(n_atoms, dtype=dtype, device=device)
    screened = even(torch.ones_like(k_squared))          # shared by both charge blocks

    def charge_block(column: Tensor) -> Tensor:
        s = _combined(sigma_free, column)[0]
        real = scatter(coulomb_kernel(r, s[pair_i, pair_j]) - coulomb_kernel(r, screen_width))
        block = real + screened - (2.0 * K_E * alpha / SQRT_PI) * eye_n
        return block - (math.pi * K_E / volume) * (1.0 / alpha**2 - s**2)

    blocks: Dict[str, Tensor] = {"j": charge_block(sigma_free), "j_fb": charge_block(sigma)}
    s01 = _combined(sigma_free, sigma_mu)[0]
    (lam3,) = real_space_damping(r, s01[pair_i, pair_j], alpha, orders=(1,))
    t01 = scatter(charge_dipole_tensor(r_vec, r, lam3)) + odd(k)
    blocks["t01"] = t01 * ell.view(1, n_atoms, 1)
    if sigma_theta is not None:
        basis = quadrupole_cartesian_basis().to(dtype=dtype, device=device)
        kk = torch.einsum("ka,mab,kb->km", k, basis, k)
        s02 = _combined(sigma_free, sigma_theta)[0]
        lam3, lam5 = real_space_damping(r, s02[pair_i, pair_j], alpha, orders=(1, 2))
        t02 = scatter(torch.einsum("pab,mab->pm", charge_quadrupole_tensor(r_vec, r, lam3, lam5), basis))
        blocks["t02"] = (t02 - 0.5 * even(kk)) * (ell * ell).view(1, n_atoms, 1)
    return blocks
