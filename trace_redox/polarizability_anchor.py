"""SPEC 35 (2026-11-07): the frame-polarizability anchor — NEW-TO-TRACE.

Why (measured 2026-11-07, theory note 2026-10-18 section 4e).  Step 20's bound sector carries alpha_O = 0.205 and
alpha_Mg = 1.916 A^3: the eps_inf prior fixed the sum of the two and nothing fixed the split, so the model put the
polarizability on the cation.  The first-principles table of SPEC 28 (data/ion_polarizabilities.json; X2C-PBE0 finite
field, the O states in the Watson sphere; ESTABLISHED as the free-ion values) has O2- 16.44 bohr^3 (2.44 A^3) and
Mg2+ 0.50 bohr^3 (0.074 A^3).  The inverted split is harmless to eps_inf and fatal in combination with SPEC 34 (the
damped frame-driven Mg dipoles were what held the surface O2-); with the physical split SPEC 34 is benign on the
slabs and gives Z*(Mg) 1.93 at eps_inf 3.235.

Construction.  For every frame ion whose alpha_Z(f) is reliably tabulated, the dipole stiffness becomes

    A_mu,i = K_E ell_i^2 / (g alpha_Z(f_i)) I,       g = exp(log_alpha_scale),

one learned crystal factor g for all of them (the in-crystal reduction of the free-ion polarizability, which the
eps_inf prior and the forces set; init g = 1), the species ratios first principles.

SPEC 35.1 (2026-11-12).  g multiplies the CHARGED frame ions only (``scope='ions'``, the default): it is the in-crystal
reduction of an ion's free polarizability (ESTABLISHED as physics: Fowler and Madden, Phys. Rev. B 29, 1035 (1984),
cited from memory), and a neutral frame atom such as Au0 is its own free atom.  Measured before the change: with one g
on every tabulated species (``scope='all'``, legacy, kept) the polarizability prior on the isolated Au atom held g at
0.997 against the eps_inf prior, the two gradients on log g cancelling to 0.04 % (benchmarks_phonons README).  Untabulated species keep the heads'
A_mu.  Applied at the heads' output — before the assembly builds its operator — in forward and bound_state alike
(:class:`AnchoredHeads`).  chi, eta, sigma, ell, kappa and the seed are untouched.
"""
from __future__ import annotations

import dataclasses
import math

import torch
from torch import Tensor

from trace_electrostatics.constants import K_E

BOHR3 = 0.529177210903 ** 3


def frame_alpha(species: Tensor) -> Tensor:
    """``[n]`` the tabulated free-ion polarizability of each atom's formal frame state (A^3; nan where not tabulated)."""
    from .constants import formal_frame
    from .polarizability import _alphas
    al = _alphas()
    frames = formal_frame(species).tolist()
    out = torch.full((int(species.numel()),), float("nan"), dtype=torch.float64)
    for i, (z, f) in enumerate(zip(species.tolist(), frames)):
        a = al.get((int(z), int(f)))
        if a is not None:
            out[i] = a * BOHR3
    return out


#: SPEC 35.1: which tabulated frame species the crystal factor g multiplies
SCOPES = ("ions", "all")


def anchored_params(params, species: Tensor, log_scale: Tensor, scope: str = "ions"):
    """``params`` with ``A_mu`` replaced by ``K_E ell^2 / (g alpha_Z) I`` on the tabulated frame ions.

    ``scope`` (SPEC 35.1): ``'ions'`` (default) applies g to the charged frame ions only; a neutral frame atom keeps its
    tabulated free-atom polarizability (g = 1).  ``'all'`` is SPEC 35 as first built: one g on every tabulated species."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    alpha = frame_alpha(species).to(params.ell.dtype).to(params.ell.device)
    have = torch.isfinite(alpha)
    if not bool(have.any()):
        return params
    g = torch.exp(log_scale.to(params.ell.dtype))
    if scope == "ions":                                            # a neutral atom is not an ion in a crystal
        from .constants import formal_frame
        charged = formal_frame(species).to(params.ell.device) != 0
        g = torch.where(charged, g, torch.ones_like(alpha))
    eye = torch.eye(3, dtype=params.ell.dtype, device=params.ell.device)
    safe_alpha = torch.where(have, alpha, torch.ones_like(alpha))
    anchored = (K_E * params.ell.reshape(-1) ** 2 / (g * safe_alpha)).view(-1, 1, 1) * eye
    A_mu = torch.where(have.view(-1, 1, 1), anchored, params.A_mu)
    return dataclasses.replace(params, A_mu=A_mu)


class AnchoredHeads:
    """The heads with SPEC 35 applied to their readout; every other attribute is the wrapped heads' own."""

    def __init__(self, heads, species: Tensor, log_scale: Tensor, scope: str = "ions") -> None:
        self._heads, self._species, self._log_scale, self._scope = heads, species, log_scale, scope

    def __call__(self, *args, **kwargs):
        out = self._heads(*args, **kwargs)
        species = kwargs.get("species", self._species)
        return out._replace(params=anchored_params(out.params, species, self._log_scale, self._scope))

    def __getattr__(self, name):
        return getattr(self._heads, name)
