"""The two-level anchor of the bond (split-charge) hardness.

One occupied orbital on one end of a bond, one empty orbital on the other, level offset ``Delta``,
hopping ``h``; the bonding orbital holds two electrons.  Exactly,

    N_B = 1 - Delta / G,   G = sqrt(Delta^2 + 4 h^2),   dN_B / dDelta = -4 h^2 / G^3,

so the split-charge hardness, the inverse of the bond's charge-transfer compliance at fixed site
potentials, is

    kappa = G^3 / (4 h^2)                      (2026-10-09 band note, Theorem 6; ESTABLISHED as
                                                second-order perturbation theory and Harrison's
                                                bond-orbital polarizability)

and, for identical atoms (Delta = 0), kappa = 2 h.  This is the hardness a bond has with no data
at all.  Learned models keep it where energies and forces are silent: a bond with t* ~ 0 has
dE*/dkappa = 1/2 t*^2 ~ 0 (docs/ELECTROSTATICS_AUDIT_2026-10-09.md, defect A2), so a free head
leaves it wherever it was initialised.  A trained MgO kept its second-shell Mg-Mg bonds at the
0.134 eV/e^2 floor, giving eps_inf = 1316.

The anchor:
* Delta = chi_i - chi_j, with chi the Mulliken electronegativity (IE + EA) / 2 from the NIST
  tables in :mod:`trace_electrostatics.constants`;
* h = 1.42 hbar^2 / (m_e r^2), the universal sp-sigma coefficient (Froyen & Harrison,
  PRB 20, 2420 (1979)).

The heads multiply it by a bounded learned factor (``bond_hardness_law='two_level'``).  Beyond
covalent contact, the optional SPEC 23 law (``charge_transfer_decay``) stiffens it further; that
is the closed-shell and nearsighted regime.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .constants import ATOMIC_ELECTRON_AFFINITY_EV, ATOMIC_IONIZATION_EV

#: hbar^2 / m_e in eV A^2.
HBAR2_OVER_ME = 7.619964
#: Froyen-Harrison universal sp-sigma coefficient.
HOPPING_COEFFICIENT = 1.42

_MULLIKEN = torch.tensor(
    [0.5 * (ie + ea) for ie, ea in zip(ATOMIC_IONIZATION_EV, ATOMIC_ELECTRON_AFFINITY_EV)], dtype=torch.float64
)


def two_level_bond_hardness(bond_length: Tensor, z_i: Tensor, z_j: Tensor) -> Tensor:
    """``kappa_anchor = (Delta^2 + 4 h^2)^(3/2) / (4 h^2)`` in eV/e^2, per bond.  Symmetric in i <-> j."""
    chi = _MULLIKEN.to(dtype=bond_length.dtype, device=bond_length.device)
    delta = chi[z_i.long()] - chi[z_j.long()]
    h = HOPPING_COEFFICIENT * HBAR2_OVER_ME / (bond_length * bond_length)
    four_h2 = 4.0 * h * h
    return (delta * delta + four_h2) ** 1.5 / four_h2
