"""Pauli screening of the split-charge (bond) channel.

A split charge on bond (i, j) describes charge flowing through the direct i-j orbital overlap.
When a third atom k sits between i and j, its occupied orbitals block that overlap (Pauli), and
charge moves through k's own bonds instead: two nearest-neighbour bonds in series, both already in
the bond list.  Rocksalt's second shell is the measured failure this removes.  There, the Mg-Mg
"bonds" across a shared O carried ~97 % of a trained model's eps_inf (1281 of 1316 for MgO).
docs/ELECTROSTATICS_AUDIT_2026-10-09.md, defect A2.

The screening factor is the many-body one of MEAM (Baskes; ESTABLISHED) and of environment-dependent
tight binding (Tang, Wang, Chan & Ho, PRB 53, 979 (1996); ESTABLISHED):

    X_ik = (r_ik / r_ij)^2,   X_kj = (r_kj / r_ij)^2,
    C_ikj = [2 (X_ik + X_kj) - (X_ik - X_kj)^2 - 1] / [1 - (X_ik - X_kj)^2],
    S_ij  = prod_k s(C_ikj),   s(C) = q((C - C_min) / (C_max - C_min)),

with q the quintic C^2 switch (0 for C <= C_min, 1 for C >= C_max).  In coordinates along the bond
(x from i at 0 to j at 1) and perpendicular to it (y, in units of r_ij), C_ikj = y^2 / (x (1 - x)).
So only atoms strictly between the planes x = 0 and x = 1 can screen (elsewhere
(X_ik - X_kj)^2 >= 1, and s = 1), and they lie inside the ellipse y^2 < C_max x (1 - x).

Exact values (C_min = 2.0, C_max = 2.8):
* rocksalt and fcc second neighbours, bridged at x = 1/2, y^2 = 1/4: C = 1, so S = 0;
* fcc nearest neighbours, whose common neighbours sit at x = 1/2, y^2 = 3/4: C = 3, so S = 1;
* a linear chain's 1-3 pair, middle atom at y = 0: C = 0, so S = 0.

Neighbour lists are built from the bonds themselves.  Each bond gives two half-edges, (lo, +d) and
(hi, -d), so the list is complete within the bond cutoff and free of duplicates, whether or not the
caller's edges are symmetric.  A screening atom's distance to its nearer endpoint is at most
sqrt(1/4 + C_max/4) r_ij = 0.975 r_ij, so it is always in the list.  Each atom contributes
log s from both ends, weighted 1 - w(x) and w(x), with w a C^2 switch over x in [0.45, 0.55].  That
is exact in exact arithmetic (the weights sum to 1), and it counts an atom on the bisector exactly
once, whatever rounding each end sees.
"""
from __future__ import annotations

import torch
from torch import Tensor

#: Defaults when screening is switched on (MEAM-style; see the module docstring for what they give).
DEFAULT_SCREENING_C_MIN = 2.0
DEFAULT_SCREENING_C_MAX = 2.8
#: Factors below this are treated as exact zeros (their derivative is below 1e-160).
_ZERO = 1e-250


def _quintic(x: Tensor) -> Tensor:
    """C^2 switch from 0 (x <= 0) to 1 (x >= 1)."""
    x = x.clamp(0.0, 1.0)
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def bond_screening(
    bond_index: Tensor,
    bond_vector: Tensor,
    n_atoms: int,
    *,
    c_min: float = DEFAULT_SCREENING_C_MIN,
    c_max: float = DEFAULT_SCREENING_C_MAX,
) -> Tensor:
    """``S_b`` in [0, 1] for every bond ``b = (lo, hi)`` with image-resolved vector ``lo -> hi``.

    Differentiable in ``bond_vector``, so forces include the screening.  ``S_b = S`` of the
    reversed bond, and it is invariant under rotation and translation.
    """
    if not 0.0 <= float(c_min) < float(c_max):
        raise ValueError(f"need 0 <= c_min < c_max, got c_min={c_min}, c_max={c_max}")
    n_bonds = int(bond_index.shape[1])
    if n_bonds == 0:
        return bond_vector.new_ones(0)
    device = bond_vector.device
    lo, hi = bond_index[0].to(device), bond_index[1].to(device)
    arange = torch.arange(n_bonds, device=device)

    # half-edges: 2b = (lo_b, +d_b), 2b + 1 = (hi_b, -d_b), sorted by source atom (CSR)
    source = torch.stack((lo, hi), dim=1).reshape(-1)
    vectors = torch.stack((bond_vector, -bond_vector), dim=1).reshape(-1, 3)
    order = torch.argsort(source, stable=True)
    vectors_sorted, ids_sorted = vectors[order], order
    counts = torch.bincount(source, minlength=int(n_atoms))
    start = torch.cumsum(counts, 0) - counts

    def leaving(atom: Tensor, own: Tensor):
        """For every bond: (bond id, vector atom -> k) over the half-edges leaving ``atom``, minus its own."""
        degree = counts[atom]
        rep = torch.repeat_interleave(arange, degree)
        offset = torch.arange(int(degree.sum()), device=device) - torch.repeat_interleave(
            torch.cumsum(degree, 0) - degree, degree)
        slot = torch.repeat_interleave(start[atom], degree) + offset
        keep = ids_sorted[slot] != own[rep]
        return rep[keep], vectors_sorted[slot[keep]]

    d = bond_vector
    d2 = (d * d).sum(dim=1)
    b_i, v_i = leaving(lo, 2 * arange)                 # v_i = r_k - r_i
    b_j, v_j = leaving(hi, 2 * arange + 1)             # v_j = r_k - r_j

    def factors(b: Tensor, r_ik: Tensor, r_kj: Tensor, x: Tensor):
        Xik = (r_ik * r_ik).sum(dim=1) / d2[b]
        Xkj = (r_kj * r_kj).sum(dim=1) / d2[b]
        D = (Xik - Xkj) ** 2
        inside = D < 1.0
        denom = torch.where(inside, 1.0 - D, torch.ones_like(D))
        C = (2.0 * (Xik + Xkj) - D - 1.0) / denom
        s = torch.where(inside, _quintic((C - c_min) / (c_max - c_min)), torch.ones_like(C))
        return s

    x_i = (v_i * d[b_i]).sum(dim=1) / d2[b_i]
    s_i = factors(b_i, v_i, d[b_i] - v_i, x_i)
    x_j = 1.0 + (v_j * d[b_j]).sum(dim=1) / d2[b_j]
    s_j = factors(b_j, d[b_j] + v_j, -v_j, x_j)
    w_i = 1.0 - _quintic((x_i - 0.45) / 0.1)           # end i owns x < 0.45, end j owns x > 0.55
    w_j = _quintic((x_j - 0.45) / 0.1)

    bond = torch.cat((b_i, b_j))
    s = torch.cat((s_i, s_j))
    w = torch.cat((w_i, w_j))
    counted = w > 0.0
    zero = counted & (s < _ZERO)
    safe = torch.where(zero | ~counted, torch.ones_like(s), s)
    log_s = torch.where(counted, w * torch.log(safe), torch.zeros_like(s))
    log_S = bond_vector.new_zeros(n_bonds).index_add(0, bond, log_s)
    screened = torch.zeros(n_bonds, dtype=torch.bool, device=device).index_put(
        (bond[zero],), torch.ones(int(zero.sum()), dtype=torch.bool, device=device), accumulate=True)
    return torch.where(screened, torch.zeros_like(log_S), torch.exp(log_S))
