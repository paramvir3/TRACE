"""Certified self-interaction removal (NEW-TO-TRACE; trace_redox/SPEC.md §8a).

The mean-field free sector writes the Coulomb energy of the free charges as
``1/2 c^T W c``.  The exact ensemble energy of integer configurations differs
from it on site: ``1/2 W_uu <n_u^2>`` versus ``1/2 W_uu c_u^2``, so a fractional
charge interacts with itself -- the delocalization error (SPEC §8).  Moving an
amount ``r_u`` of the on-site Coulomb from the quadratic form into the
populations,

    1/2 c^T W c  ->  1/2 c^T (W - diag r) c + 1/2 sum_u r_u <n_u^2>,

is exact at every integer configuration (``<n^2> = c^2`` there) and removes
``1/2 r_u Var_u`` of self-interaction at fractional ones.  The functional stays
convex iff ``W - diag(r) >= 0``.  This module gives a closed-form, smooth ``r``
that certifies it:

**Theorem.**  Let ``W > 0``, ``M = W^-1``, ``s_u = sum_v |M_uv|``.  Then
``W - diag(r) >= 0`` for every ``0 < r_u <= 1 / s_u``.

*Proof.*  With ``R = diag(r)``, ``W >= R  <=>  lambda_max(R^1/2 M R^1/2) <= 1``.
For the symmetric ``B = R^1/2 M R^1/2``, ``lambda_max(B) <= rho(|B|)``, and by the
Collatz-Wielandt bound with ``x_u = r_u^-1/2 > 0``,
``rho(|B|) <= max_u (|B| x)_u / x_u = max_u r_u s_u <= 1``.  []

Properties (tested): ``r_u <= 1/M_uu <= W_uu``; an isolated unit gets
``r_u = W_uu`` (its full on-site Coulomb: the exact PPLB discontinuity); for two
units the bound is tight (``det(W - R) = 0``) and the curvature left along the
transfer direction is ``|w| (a - d)^2 / ((a + |w|)(d + |w|))``, zero for equal
units; an ill-conditioned (dense, strongly screened) ``W`` gives ``r -> 0``,
the v1 mean field.

``|M_uv|`` is replaced by the smooth upper bound ``sqrt(M_uv^2 + eps_uv^2)`` so
that ``r`` (hence the force) is smooth where an off-diagonal ``M_uv`` changes
sign.  Any ``eps >= 0`` keeps the certificate valid (it only lowers ``r``);
``eps_uv = eps0 sqrt(M_uu M_vv) rho_uv^2``, with ``rho_uv = J_uv / sqrt(J_uu J_vv)``
the normalised bare coupling, keeps the total loosening small for distant units.
``theta < 1`` makes ``W - diag(r)`` strictly positive definite.
"""

from __future__ import annotations

import torch
from torch import Tensor


def certified_removal(W: Tensor, J: Tensor, *, theta: float = 0.999, smoothing: float = 0.01) -> Tensor:
    """``r_u = theta / sum_v sabs(M_uv)``, ``M = W^-1`` (see the module docstring).

    ``W`` is the (unit-level, screened) free-free operator, ``J`` the bare one
    (only its normalised couplings set the smoothing scale).  Differentiable.
    """
    if not 0.0 < theta < 1.0:
        raise ValueError(f"theta must lie in (0, 1), got {theta}")
    if smoothing < 0.0:
        raise ValueError(f"smoothing must be >= 0, got {smoothing}")
    factor = torch.linalg.cholesky(0.5 * (W + W.transpose(0, 1)))
    M = torch.cholesky_inverse(factor)
    M = 0.5 * (M + M.transpose(0, 1))
    diag = torch.diagonal(M)                                   # > 0 for W > 0
    n = int(W.shape[0])
    off = 1.0 - torch.eye(n, dtype=W.dtype, device=W.device)
    j_diag = torch.diagonal(J)
    rho = J / torch.sqrt(j_diag.unsqueeze(1) * j_diag.unsqueeze(0))
    scale = torch.sqrt(diag.unsqueeze(1) * diag.unsqueeze(0))
    # the eps_machine floor keeps the square root (and its gradient) away from 0
    eps = (smoothing * rho * rho + torch.finfo(W.dtype).eps) * scale
    s = diag + (torch.sqrt(M * M + eps * eps) * off).sum(dim=1)
    return theta / s
