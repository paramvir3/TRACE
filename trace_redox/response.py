"""The whole model's electronic response with several configurations (trace_redox SPEC 30).

Eliminating the bound sector exactly at fixed configuration n in a uniform field F (state-independent K),

    E(n; F) = E(n; 0) - F . M(n) - 1/2 F^T alpha_b F,     alpha_b = D^T K^-1 D (the same for every n),
    M(n)    = sum_u q_u(n) r_u + D^T y*(n),               y*(n) = y0 - K^-1 A^T q(n)  [- K^-1 dF(n) with SPEC 26],

so the response of the canonical ensemble over the diabatic Hamiltonian H is (with SPEC 28's state-dependent stiffness
K(n), alpha_b(n) = alpha_b + dalpha(n) enters the diagonal as 1/2 F^T dalpha(n) F; polarizability.py)

    alpha = alpha_b + alpha_ens,   alpha_ens = -d2 F_ens / dF2,   F_ens(F) = -kT ln Tr exp(-[H - sum_k F_k diag M_k] / kT),

which contains the thermal term Var_p(M) / kT and the coherent charge-transfer response (two levels:
(mu^2 / 4) tanh(t / kT) / t).  Response units e^2 A^2 / eV, as trace_electrostatics.observables.

Periodic cells: sum_u q_u r_u is defined only modulo lattice vectors.  For a configuration, dQ = q(n) - q(ref) sums to
zero, so sum_u dQ_u p_u is origin-independent; p_u is the minimal-image position of unit u around the first atom unit
whose charge changed (a band carrier at the phi^2-weighted minimal-image centroid of its sites).  For one moved
electron this is the minimal-image hop vector r_donor - r_acceptor.

ESTABLISHED: linear response of a canonical ensemble; minimal-image polarisation of localised transfers.
NEW-TO-TRACE: the response priors evaluated on the whole configuration ensemble of the redox sector.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .diabatic import canonical_free_energy


def _minimal_image(vec: Tensor, cell: Tensor | None, pbc) -> Tensor:
    if cell is None or pbc is None or not any(bool(p) for p in pbc):
        return vec
    cell = cell.to(vec.dtype)
    frac = vec @ torch.linalg.inv(cell)
    mask = torch.tensor([bool(p) for p in pbc], device=vec.device)
    frac = torch.where(mask, frac - torch.round(frac), frac)
    return frac @ cell


def free_dipoles(configs: Tensor, max_charge: Tensor, pos: Tensor, cell: Tensor | None, pbc,
                 omega: Tensor | None = None) -> Tensor:
    """``[M, 3]`` free-charge dipoles of the configurations relative to the first, e A (module docstring).

    ``configs [M, U]`` unit charges; units beyond the atoms (a band unit) map onto the atoms through ``omega [U, n]``,
    or, for SPEC 38's Fermi band unit, through its :class:`trace_redox.band.BandCharges` (the band's charge then sits
    on its sites with each filling's own shape).  ``max_charge`` is accepted for the call signature of the move
    routines (unused)."""
    del max_charge
    from .band import BandCharges
    if isinstance(omega, BandCharges):
        q, omega = omega.atoms(configs).to(pos.dtype), None                 # [M, n]: every charge is on an atom
    else:
        q = configs.to(pos.dtype)
    n = int(pos.shape[0])
    U = int(q.shape[1])
    dq = q - q[0:1]
    out = torch.zeros(int(q.shape[0]), 3, dtype=pos.dtype, device=pos.device)
    for k in range(1, int(q.shape[0])):
        changed = dq[k].nonzero().flatten()
        if int(changed.numel()) == 0:
            continue
        atoms = changed[changed < n]
        ref = pos[int(atoms[0])] if int(atoms.numel()) else pos[0]
        rel = _minimal_image(pos - ref, cell, pbc)                          # [n, 3] minimal-image positions around ref
        p = torch.zeros(U, 3, dtype=pos.dtype, device=pos.device)
        p[:min(U, n)] = rel[:min(U, n)]
        if U > n:
            if omega is None:
                raise ValueError("units beyond the atoms need omega")
            p[n:] = omega[n:].to(pos.dtype) @ rel                           # phi^2-weighted centroids
        out[k] = dq[k] @ p
    return out


def ensemble_polarizability(H: Tensor, M: Tensor, kT: float, h: float = 1e-3,
                             quadratic: Tensor | None = None) -> Tensor:
    """``alpha_ens [3, 3]`` = -d2/dF_i dF_j of F_ens(F) = -kT ln Tr exp(-[H - diag(F . M + 1/2 F^T dalpha F)] / kT)
    at F = 0, by central differences at h and h / 2, Richardson-extrapolated (O(h^4)); differentiable to first order
    in the inputs through :func:`canonical_free_energy`.  ``H [m, m]`` eV, ``M [m, 3]`` e A, ``kT`` eV, ``h`` V/A;
    ``quadratic [m, 3, 3]`` (SPEC 28 on): each configuration's bound polarisability minus the common alpha_b, so the
    whole response is alpha_b + alpha_ens with the state-dependent stiffness exact."""
    def free(field: Tensor) -> Tensor:
        shift = M @ field
        if quadratic is not None:
            shift = shift + 0.5 * torch.einsum("i,kij,j->k", field, quadratic.to(field.dtype), field)
        return canonical_free_energy(H - torch.diag(shift), kT)

    def tensor(step: float) -> Tensor:
        f0 = free(torch.zeros(3, dtype=H.dtype, device=H.device))
        rows = []
        for i in range(3):
            row = []
            for j in range(3):
                ei = torch.zeros(3, dtype=H.dtype, device=H.device)
                ej = torch.zeros(3, dtype=H.dtype, device=H.device)
                ei[i] = step
                ej[j] = step
                if i == j:
                    row.append(-(free(ei) - 2.0 * f0 + free(-ei)) / (step * step))
                else:
                    row.append(-(free(ei + ej) - free(ei - ej) - free(-ei + ej) + free(-ei - ej)) / (4.0 * step * step))
            rows.append(torch.stack(row))
        return torch.stack(rows)

    a1, a2 = tensor(h), tensor(0.5 * h)
    alpha = (4.0 * a2 - a1) / 3.0
    return 0.5 * (alpha + alpha.transpose(0, 1))


def record_polarizability(record: dict) -> Tensor:
    """``alpha_ens [3, 3]`` of an ensemble response record (``RedoxResult.ensemble``), to be added to its
    ``alpha_bound``.

    One Hamiltonian (SPEC 30): :func:`ensemble_polarizability` of ``H``, ``M`` and SPEC 28's ``alpha_shift``.
    The cluster product (SPEC 36.10; ``record['terms']``): F = E0 + sum_c (F_c - E0) + sum_{c<c'} (F_cc' - F_c - F_c'
    + E0) is linear in the free energies of its terms, so alpha_ens = sum_X w_X alpha_ens(H_X, M[rows_X],
    alpha_shift[rows_X]) with the weights of the same combination (they sum to one, so alpha_bound enters once).
    At a Born-von Karman mesh (SPEC 37) the record is the supercell's and ``record['scale']`` = 1 / copies makes
    the result the polarisability per cell, as ``alpha_bound`` then is."""
    shift = record.get("alpha_shift")
    terms = record.get("terms")
    scale = float(record.get("scale", 1.0))               # SPEC 37: 1 / copies, the response per cell at a mesh
    if terms is None:
        return scale * ensemble_polarizability(record["H"], record["M"], record["kT"], quadratic=shift)
    total = None
    for term in terms:
        rows = term["rows"]
        part = float(term["weight"]) * ensemble_polarizability(
            term["H"], record["M"][rows], record["kT"], quadratic=None if shift is None else shift[rows])
        total = part if total is None else total + part
    return scale * total
