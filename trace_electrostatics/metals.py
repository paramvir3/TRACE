"""Metals and constant-potential electrodes in split charge equilibration (SPEC 20).

Why metals were unreachable
---------------------------
SQE with a positive bond hardness is a dielectric, never a metal (Nistor and
Muser, Phys. Rev. B 79, 104303 (2009)): the hardness ``kappa`` is the price of
moving charge across a bond, and a finite price means finite screening.  A metal
is the ``kappa -> 0`` limit -- but on the PHYSICAL bond graph that limit is
singular: every closed loop of bonds carries a divergence-free current that no
atomic charge records, so with ``kappa = 0`` those currents cost nothing and
``K`` loses its definiteness.

The construction
----------------
A connected metallic region (a *domain*: metallic atoms joined by bonds inside
one fragment) is made equipotential by a SPANNING STAR of zero-hardness bonds,
one from the domain's hub (its lowest atom index) to each other member.  A star
is a tree -- it has no loops -- so its incidence columns are independent and
span exactly the charge redistributions that keep the domain's total fixed.
With (numerically) zero hardness on them the minimiser equalises the
electrochemical potential ``u_i = dU/dq_i`` over the domain:

    u_i = u_hub   for every i in the domain,

which is the QEq condition inside a conductor.  Because the star spans the
whole redistribution space, the minimum does not depend on which tree was
chosen.  ``K`` stays positive definite: on the star the bond block is
``B^T (eta + J) B``, and ``eta + J`` is a Gram matrix plus the positive on-site
residuals ``eta - S_0`` (SPEC 17.3), so it is definite on the span of any set of
independent columns.  The physical metal-metal bonds are kept; at the minimum
they carry no current, since the star already equalises the potentials.

Electrodes at fixed potential DIFFERENCES
-----------------------------------------
A domain given an electrode potential ``V_e`` (in V = eV/e, the electrochemical
potential of a positive charge) is joined to a reference electrode through an
external-circuit bond, hub to hub, with the linear term ``-(V_e - V_ref) tau``:
the work the external source does moving ``tau`` from the reference electrode
to electrode e.  Stationarity then gives

    u_e - u_ref = V_e - V_ref ,

the constant-potential condition of the Siepmann-Sprik electrode (J. Chem.
Phys. 102, 511 (1995); Reed, Lanning and Madden, J. Chem. Phys. 126, 084704
(2007)), with the total charge conserved -- only DIFFERENCES of potential are
imposed, which is also the only thing that is defined under periodic
boundaries.  All electrodes of one circuit must lie in one fragment: the
circuit connects them, so their charges are not separately conserved.

Established, cited: QEq inside a conductor; the constant-potential electrode.
TRACE-native: metallic domains as zero-hardness spanning stars in the SQE bond
graph, and the external circuit as a biased bond, so that metals, electrodes,
dielectrics and the bond-current polarization share one operator, one solver
and the implicit derivatives of SPEC 17.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

__all__ = ["METAL_BOND_HARDNESS", "MetalBonds", "metal_bonds"]

#: Hardness of the star and circuit bonds, eV/e^2.  Zero in the physics; 1e-8
#: keeps the parameter validation (kappa > 0) and moves potentials by
#: kappa * tau ~ 1e-8 V -- far below anything a potential energy resolves.
METAL_BOND_HARDNESS = 1e-8


@dataclass(frozen=True)
class MetalBonds:
    """Extra bonds that make metallic domains equipotential and close circuits.

    Attributes:
        bond_index: ``[2, m]`` ``(lo, hi)``; the column is ``+1`` at lo, ``-1``
            at hi, so ``tau > 0`` moves positive charge from hi to lo.
        vector: ``[m, 3]`` ``r_lo - r_hi`` along the path the charge takes
            (through the domain's bond graph for star bonds, so it is image-
            resolved under PBC), for the bond-current polarization.
        bias: ``[m]`` linear term per bond, eV/e: 0 on star bonds,
            ``-(V_e - V_ref)`` on circuit bonds.
        domain: ``[n_atoms]`` domain id per atom, ``-1`` for non-metallic atoms.
        n_star, n_circuit: counts.
    """

    bond_index: Tensor
    vector: Tensor
    bias: Tensor
    domain: Tensor
    n_star: int
    n_circuit: int


def metal_bonds(
    metallic: Tensor,
    bond_index: Tensor,
    bond_vector: Tensor,
    fragment_of: Tensor,
    positions: Tensor,
    electrode_potential: Optional[Tensor] = None,
) -> Optional[MetalBonds]:
    """Star bonds per metallic domain and circuit bonds between electrodes.

    Args:
        metallic: ``[n_atoms]`` bool.
        bond_index: ``[2, n_bonds]`` physical bonds (self-image bonds have
            ``lo == hi`` and are ignored).
        bond_vector: ``[n_bonds, 3]`` image-resolved ``r_lo - r_hi``.
        fragment_of: ``[n_atoms]`` fragment label per atom.
        positions: ``[n_atoms, 3]``; used only for circuit-bond vectors.
        electrode_potential: optional ``[n_atoms]`` in V; NaN for atoms that are
            not on a constant-potential electrode.  Must be uniform within a
            metallic domain.

    Returns:
        :class:`MetalBonds`, or ``None`` when no atom is metallic.
    """
    metallic = metallic.reshape(-1).to(torch.bool)
    n_atoms = int(metallic.numel())
    if not bool(metallic.any()):
        if electrode_potential is not None and bool(torch.isfinite(electrode_potential).any()):
            raise ValueError(
                "electrode_potential is set on atoms that are not metallic: an "
                "electrode is a metallic domain (pass metallic=True for its atoms)."
            )
        return None
    lo_all, hi_all = bond_index[0].tolist(), bond_index[1].tolist()
    is_metal = metallic.tolist()
    frag = fragment_of.reshape(-1).tolist()

    # --- domains: connected components of metal-metal bonds within a fragment -
    adjacency = [[] for _ in range(n_atoms)]
    for b, (i, j) in enumerate(zip(lo_all, hi_all)):
        if i != j and is_metal[i] and is_metal[j] and frag[i] == frag[j]:
            adjacency[i].append((j, b, -1.0))      # r_j = r_i - vector_b
            adjacency[j].append((i, b, +1.0))      # r_i = r_j + vector_b
    domain = [-1] * n_atoms
    star_lo, star_hi, star_vectors, hubs = [], [], [], []
    dtype, device = bond_vector.dtype, bond_vector.device
    zero = torch.zeros(3, dtype=dtype, device=device)
    for hub in range(n_atoms):
        if not is_metal[hub] or domain[hub] != -1:
            continue
        d = len(hubs)
        hubs.append(hub)
        domain[hub] = d
        # BFS: displacement of each member from the hub along the tree path,
        # built from the (differentiable, image-resolved) bond vectors.
        displacement = {hub: zero}
        queue = deque([hub])
        while queue:
            i = queue.popleft()
            for j, b, sign in adjacency[i]:
                if domain[j] == -1:
                    domain[j] = d
                    displacement[j] = displacement[i] + sign * bond_vector[b]
                    queue.append(j)
        for member, disp in displacement.items():
            if member != hub:
                star_lo.append(hub)
                star_hi.append(member)
                star_vectors.append(-disp)          # r_hub - r_member
    n_star = len(star_lo)

    # --- electrodes and the external circuit ---------------------------------
    circuit_lo, circuit_hi, circuit_vectors, circuit_bias = [], [], [], []
    if electrode_potential is not None:
        potential = electrode_potential.reshape(-1).to(dtype)
        finite = torch.isfinite(potential)
        if bool((finite & ~metallic).any()):
            raise ValueError(
                "electrode_potential is set on non-metallic atoms: an electrode is a "
                "metallic domain."
            )
        electrodes = []
        for d, hub in enumerate(hubs):
            members = [i for i in range(n_atoms) if domain[i] == d]
            values = potential[members]
            if not bool(torch.isfinite(values).any()):
                continue
            if not bool(torch.isfinite(values).all()) or float(values.max() - values.min()) > 1e-12:
                raise ValueError(
                    f"metallic domain {d} (hub atom {hub}) carries more than one "
                    "electrode potential: a connected conductor is equipotential, so "
                    "its atoms must share one value (or all be NaN)."
                )
            electrodes.append((hub, float(values[0])))
        if len(electrodes) == 1:
            raise ValueError(
                "one constant-potential electrode alone: only potential DIFFERENCES "
                "are imposed (total charge is conserved), so an electrode needs a "
                "counter-electrode in the same fragment."
            )
        if electrodes:
            fragments = {frag[hub] for hub, _ in electrodes}
            if len(fragments) != 1:
                raise ValueError(
                    "the electrodes of one circuit lie in different fragments "
                    f"({sorted(fragments)}). The external circuit connects them, so "
                    "their charges are not separately conserved: give all electrodes "
                    "one fragment label (fragment_id) with their combined net charge."
                )
            ref_hub, ref_v = electrodes[0]
            for hub, value in electrodes[1:]:
                circuit_lo.append(hub)
                circuit_hi.append(ref_hub)
                circuit_vectors.append(positions[hub] - positions[ref_hub])
                circuit_bias.append(-(value - ref_v))
    n_circuit = len(circuit_lo)

    lo = star_lo + circuit_lo
    hi = star_hi + circuit_hi
    index = torch.tensor([lo, hi], dtype=bond_index.dtype, device=bond_index.device).reshape(2, -1)
    vectors = star_vectors + [v.to(dtype) for v in circuit_vectors]
    vector = torch.stack(vectors) if vectors else torch.zeros(0, 3, dtype=dtype, device=device)
    bias = torch.tensor([0.0] * n_star + circuit_bias, dtype=dtype, device=device)
    return MetalBonds(
        bond_index=index,
        vector=vector,
        bias=bias,
        domain=torch.tensor(domain, dtype=torch.long, device=bond_index.device),
        n_star=n_star,
        n_circuit=n_circuit,
    )
