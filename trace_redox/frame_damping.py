"""SPEC 34 (2026-11-07): the overlap damping of the frame-charge -> bound-dipole coupling — NEW-TO-TRACE.

Why (measured; theory note 2026-10-18 section 4e).  A lattice of Gaussian-smeared induced dipoles driven by the frame
charges' Coulomb fields carries a Lorentz-type enhancement of the Born charge: on step 20's bulk MgO, Z*(Mg) = 2.43
against PBE's 1.99 at the right eps_inf (3.0), and no width, species split or short-range pair term moves it (every
scan of 2026-11-07).  The embedded Mg2+-O2- pair (ALMO, PBE / def2-tzvp) measured what is missing: the O2- dipole's
response to its own cation is damped by overlap to s(R) = 0.61 + 0.37 (R - 2.128 A) of the point-field value over
R = 1.93-2.33 A — the shell model's short-range shell-shell coupling (Dick and Overhauser 1958; Sangster et al. 1970;
ESTABLISHED) in first-principles form.  The slope is the physics: with it, the first-shell derivative of the frame's
drive on the bound dipoles falls from A' ~ -0.94 A / A to s A' + s' A ~ -0.20 A / A, and the frozen Z*(Mg) goes from
2.43 to 1.93 with eps_inf untouched (pre-registered, passed).

Construction (SPEC 26's on the frame; exact in the configuration energies).  The frame charge f_i drives the bound
moments through A + Delta A with, on the dipole rows of atom j,

    Delta A[i, mu_j] = (s(R_ij) - 1) K lambda_3(R_ij) (r_i - r_j)_c / R_ij^3 ell_j,   lambda_3 = P(3/2, R^2 / (sigma_f,i^2 + w_j^2)),

for the tabulated cation-anion frame pairs (periodic images summed), s(R) C^2-switched to 1 over the first-shell
window.  With v = Delta A^T f the bound drive gains v, so y0 -> y0 - K^-1 v, u0 -> u0 - A K^-1 v and the free-sector
constant gains v . y0 - 1/2 v . K^-1 v (the quadratic form of the drive).  The bound sector's own response to a field
(K, eps_inf) is untouched; the carriers' couplings are untouched (SPEC 26 corrects those separately).

Limitations.  One tabulated pair, (Mg2+, O2-), from one ALMO scan; untabulated frame pairs are left alone (s = 1).
The law's slope carries +-0.1 /A from the five-point fit against the Watson-sphere alpha; dZ*/ds' ~ -3.5 e per 1/A.
"""
from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from .penetration import K_E, pairs_within

#: (s0, s1, R0 [A], r_in [A], r_cut [A]) per (Z_cation, Z_anion): s(R) = clamp(s0 + s1 (R - R0), S_MIN, 1), switched to 1
#: between r_in and r_cut (C^2).  Mg2+-O2-: measured 2026-11-07 (tools/electrostatics/deformation_dipole.py, the ALMO
#: polarisation dipole against alpha(O2-) E_point with alpha = 2.44 A^3, R = 1.928-2.328 A).
LAWS = {(12, 8): (0.61, 0.37, 2.128, 2.6, 3.0)}
S_MIN = 0.2


def _window(R: Tensor, r_in: float, r_cut: float) -> Tensor:
    """1 for R <= r_in, the quintic C^2 switch to 0 at r_cut."""
    t = ((R - r_in) / (r_cut - r_in)).clamp(0.0, 1.0)
    return 1.0 - t ** 3 * (10.0 - 15.0 * t + 6.0 * t * t)


def damping_factor(R: Tensor, z_cation: int, z_anion: int, laws=None) -> Tensor:
    """``s(R)`` of the pair, 1 for an untabulated pair and beyond the window."""
    laws = LAWS if laws is None else laws
    law = laws.get((int(z_cation), int(z_anion)))
    if law is None:
        return torch.ones_like(R)
    s0, s1, R0, r_in, r_cut = law
    s = torch.clamp(s0 + s1 * (R - R0), S_MIN, 1.0)
    return 1.0 + (s - 1.0) * _window(R, r_in, r_cut)


def damping_vector(elim, species: Tensor, frames: Tensor, sigma_free: Tensor, w_mu: Tensor, ell: Tensor, pos: Tensor,
                   cell: Tensor | None, pbc, laws=None) -> Tensor:
    """``v = Delta A^T f`` [m] in the moment basis (dipole rows n_bonds + 3 j + c; every other row zero)."""
    from torch.special import gammainc
    laws = LAWS if laws is None else laws
    dtype, device = pos.dtype, pos.device
    n, nb = int(species.shape[0]), int(elim.n_bonds)
    m = int(elim.A.shape[1])
    v = torch.zeros(m, dtype=dtype, device=device)
    r_max = max([law[4] for law in laws.values()], default=0.0)
    if r_max <= 0.0:
        return v
    i, j, R, vec = pairs_within(pos, cell, pbc, r_max, return_vectors=True, allow_self=True)[:4]   # vec = r_j + T - r_i
    if int(i.numel()) == 0:
        return v
    f = frames.to(dtype)
    z = species.to(torch.long)
    for charge, moment, sep in ((i, j, -vec), (j, i, vec)):                       # sep = r_charge - r_moment
        is_pair = (f[charge] > 0) & (f[moment] < 0) | (f[charge] < 0) & (f[moment] > 0)
        if not bool(is_pair.any()):
            continue
        zc = torch.where(f[charge] > 0, z[charge], z[moment]); za = torch.where(f[charge] > 0, z[moment], z[charge])
        s = torch.ones_like(R)
        for (zc_k, za_k) in laws:
            sel = (zc == zc_k) & (za == za_k) & is_pair
            if bool(sel.any()):
                s = torch.where(sel, damping_factor(R, zc_k, za_k, laws), s)
        active = is_pair & (s < 1.0)
        if not bool(active.any()):
            continue
        a = active.nonzero().flatten()
        s01 = torch.sqrt(sigma_free[charge[a]] ** 2 + w_mu[moment[a]] ** 2)
        lam = gammainc(torch.tensor(1.5, dtype=dtype, device=device), (R[a] / s01) ** 2)
        geometry = K_E * sep[a] / (R[a] ** 3).unsqueeze(1) * ell[moment[a]].unsqueeze(1)            # [p, 3]
        value = ((s[a] - 1.0) * lam * f[charge[a]]).unsqueeze(1) * geometry
        rows = nb + 3 * moment[a].unsqueeze(1) + torch.arange(3, device=device)
        v = v.index_put((rows.reshape(-1),), value.reshape(-1), accumulate=True)
    return v


def frame_induction_damping(elim, species: Tensor, frames: Tensor, sigma_free: Tensor, w_mu: Tensor, ell: Tensor,
                            pos: Tensor, cell: Tensor | None, pbc, laws=None):
    """The elimination with the frame's damped drive: y0 - K^-1 v, u0 - A K^-1 v, constant + v.y0 - 1/2 v.K^-1 v."""
    v = damping_vector(elim, species, frames, sigma_free, w_mu, ell, pos, cell, pbc, laws=laws)
    if not bool((v != 0).any()):
        return elim
    Kinv_v = elim.operator.solve(v.unsqueeze(1)).squeeze(1)
    constant = elim.constant + (v @ elim.y0) - 0.5 * (v @ Kinv_v)
    return dataclasses.replace(elim, y0=elim.y0 - Kinv_v, u0=elim.u0 - elim.A @ Kinv_v, constant=constant)
