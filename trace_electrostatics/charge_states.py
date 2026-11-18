"""Integer charge localisation through coupled diabatic charge states (SPEC 21).

The problem
-----------
Charge equilibration cannot localise an integer charge: when a charged species
dissociates, or an electron transfers between a donor and an acceptor, the
minimiser spreads the charge fractionally (the delocalisation error shared by
every QEq/SQE model; SPEC 17.4 and the calculator's ``fragment_policy``).
Which fragment carries the charge is a DISCRETE choice -- a diabatic state --
and the physics of the transfer is the coupling between those states.

The construction
----------------
Each diabatic state ``a`` fixes integer charges on explicit fragments
(``fragment_id``, ``fragment_charges``) and is a full electrostatic solve,
``E_a``.  The local (short-range) energy of the potential does not depend on
the charge assignment, so it shifts every diagonal element equally and drops
out of the mixing.  The adiabatic electrostatic energy is the lowest eigenvalue
of

    H = [ E_1   V_12  ... ]
        [ V_12  E_2   ... ]        (empirical valence bond; Warshel and Weiss,
        [ ...              ]         J. Am. Chem. Soc. 102, 6218 (1980))

For two states, ``E = (E_1 + E_2)/2 - sqrt(((E_1 - E_2)/2)^2 + V^2)``: with
``V != 0`` the ground state is non-degenerate at every geometry, so the energy
and forces are smooth through the crossing, and far from it the ground state
is the lower diabatic state -- integer charges -- to O(V^2 / dE).  At the
crossing (``E_1 = E_2``) the gap is ``2|V|``, the Marcus adiabatic splitting.

The coupling ``V`` is an input: a constant, or any differentiable function of
the positions the caller computes (typically ``V_0 exp(-beta (r_DA - r_0))``
over a donor-acceptor distance).  Nothing here learns it.

Observables: the ground-state weights ``c_a^2`` (sum to 1) and the
Hellmann-Feynman charges ``sum_a c_a^2 q_a`` -- the charge the adiabatic energy
responds with, continuous through the transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence

import torch
from torch import Tensor

__all__ = ["ChargeStateResult", "adiabatic_charge_states"]


@dataclass
class ChargeStateResult:
    """Adiabatic energy of coupled diabatic charge states.

    Attributes:
        energy: scalar adiabatic electrostatic energy, eV (differentiable).
        diabatic_energies: ``[n_states]``.
        weights: ``[n_states]`` ground-state populations ``c_a^2``.
        charges: ``[n_atoms]`` population-weighted charges.
        states: the per-state :class:`~trace_electrostatics.energy.ElectrostaticState`.
    """

    energy: Tensor
    diabatic_energies: Tensor
    weights: Tensor
    charges: Tensor
    states: List[Any]


def adiabatic_charge_states(
    state_fn,
    states: Sequence[Mapping[str, Any]],
    coupling,
) -> ChargeStateResult:
    """Lowest eigenvalue of the diabatic charge-state Hamiltonian.

    Args:
        state_fn: ``state_fn(**spec) -> ElectrostaticState``, typically a
            ``functools.partial`` of :func:`~trace_electrostatics.energy.electrostatic_state`
            with everything but the charge specification bound.
        states: one mapping per diabatic state, e.g.
            ``{"fragment_id": ids, "fragment_charges": [1.0, 0.0]}``.
        coupling: the off-diagonal ``V``: a scalar (float or 0-d tensor), or an
            ``[n_states, n_states]`` symmetric tensor (its diagonal is ignored).

    Returns:
        :class:`ChargeStateResult`.
    """
    if len(states) < 2:
        raise ValueError("adiabatic_charge_states needs at least two diabatic states")
    solved = [state_fn(**dict(spec)) for spec in states]
    energies = torch.stack([s.energy + (0.0 if s.dispersion is None else s.dispersion) for s in solved])
    n = len(states)
    dtype, device = energies.dtype, energies.device
    V = torch.as_tensor(coupling, dtype=dtype, device=device)
    if V.dim() == 0:
        off = V * (torch.ones(n, n, dtype=dtype, device=device) - torch.eye(n, dtype=dtype, device=device))
    else:
        if V.shape != (n, n):
            raise ValueError(f"coupling must be a scalar or [{n}, {n}], got {tuple(V.shape)}")
        off = 0.5 * (V + V.transpose(0, 1)) * (1.0 - torch.eye(n, dtype=dtype, device=device))
    if n == 2:
        # Closed form: smooth wherever V != 0, including at the crossing.
        mean = 0.5 * (energies[0] + energies[1])
        half_gap = 0.5 * (energies[0] - energies[1])
        coupling_01 = off[0, 1]
        root = torch.sqrt(half_gap * half_gap + coupling_01 * coupling_01)
        energy = mean - root
        # Ground state (c_0, c_1) of [[E0, V], [V, E1]]: c_0^2 = (1 + h/root)/2 with
        # h = (E1 - E0)/2 = -half_gap.
        safe = torch.where(root > 0, root, torch.ones_like(root))
        w0 = 0.5 * (1.0 - half_gap / safe)
        weights = torch.stack([w0, 1.0 - w0])
    else:
        H = torch.diag(energies) + off
        values, vectors = torch.linalg.eigh(H)
        energy = values[0]
        weights = vectors[:, 0] ** 2
    charges = sum(w * s.solve_result.q for w, s in zip(weights, solved))
    return ChargeStateResult(
        energy=energy,
        diabatic_energies=energies,
        weights=weights,
        charges=charges,
        states=solved,
    )
