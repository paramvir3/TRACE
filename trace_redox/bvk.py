"""SPEC 37: Born-von Karman sampling of the configuration fluctuations.

At Gamma a cluster's excitation interacts in phase with its own periodic images, so the energy per cell depends on
the cell chosen for one crystal (benchmarks_4g README R15, R16).  The model's energy at a mesh M is defined as

    E_M(C) := E_Gamma(S_M[C]) / (M_1 M_2 M_3),

S_M[C] the M-fold supercell with the SPEC 36 cluster ensemble: the copies of a cluster fluctuate independently, an
excitation meets its own images at M a.  This module holds the two pure pieces, the mesh of a structure and the
tiling of its graph; ``RedoxElectrostatics.forward`` evaluates the supercell and returns the home cell's record.

The tiling keeps the caller's ``edge_vec`` and ``edge_len`` (repeated), and builds the positions and the cell from
the caller's: the supercell's energy is a differentiable function of the cell's positions, edge vectors and lattice,
so forces and stress per cell follow from the same backward pass as at Gamma.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from trace_electrostatics.energy import _fold_counts, resolve_pbc

UNIT_MESH = (1, 1, 1)


def mesh_for(cell: Tensor | None, pbc: Any, mesh: tuple, length: float) -> tuple:
    """The Born-von Karman mesh of one structure.

    ``length > 0``: ``M_d = ceil(length / w_d)`` along the periodic directions, ``w_d`` the spacing of the lattice
    planes (the convention of ``trace_electrostatics.energy._fold_counts``); cells of one crystal with equal
    ``M_d w_d`` then have the same energy per cell.  Otherwise ``mesh`` itself, which must be 1 along a direction
    that is not periodic.  A structure with no periodic direction has the unit mesh."""
    mesh = tuple(int(m) for m in mesh)
    flags, lattice = resolve_pbc(pbc, cell)
    if float(length) > 0.0:
        return UNIT_MESH if lattice is None else tuple(_fold_counts(lattice, flags, float(length)))
    for d in range(3):
        if mesh[d] > 1 and not flags[d]:
            raise ValueError(f"bvk_mesh = {mesh} asks for {mesh[d]} cells along direction {d}, which is not periodic "
                             f"(pbc = {flags})")
    return mesh


def tile(mesh: tuple, h: Tensor, pos: Tensor, edge_index: Tensor, edge_vec: Tensor, edge_len: Tensor,
         cell: Tensor, edge_shift: Tensor | None, species: Any = None) -> dict:
    """The M-fold supercell's features, positions, lattice and graph from the cell's.

    Copies are ordered as ``ase.Atoms.repeat`` orders them (first lattice index slowest), atom ``i`` of copy ``c`` is
    ``c n + i`` and the home cell is copy 0.  Convention (flashace): ``edge_vec = pos[src] - pos[dst] + shift @ cell``
    with ``edge_index = [src, dst]``; the edge of copy R then joins ``dst`` in R to ``src`` in ``(R + shift) mod M``
    across ``floor((R + shift) / M)`` supercells.  The convention is checked on the input."""
    if edge_shift is None:
        raise ValueError("Born-von Karman sampling needs the graph's image shifts (edge_shift)")
    dtype, device = pos.dtype, pos.device
    n, n_edges = int(pos.shape[0]), int(edge_index.shape[1])
    M = torch.tensor(mesh, dtype=torch.long, device=device)
    copies = int(M.prod())
    shift = edge_shift.round().to(torch.long)
    src, dst = edge_index[0], edge_index[1]
    if n_edges:
        expected = pos[src] - pos[dst] + shift.to(dtype) @ cell
        scale = float(cell.detach().abs().max())
        if float((expected - edge_vec).detach().abs().max()) > 1e-6 * scale:
            raise ValueError("the graph does not follow edge_vec = pos[src] - pos[dst] + edge_shift @ cell, so its "
                             "supercell cannot be built")
    R = torch.cartesian_prod(*[torch.arange(int(m), device=device) for m in mesh]).reshape(copies, 3)
    big_pos = (pos.unsqueeze(0) + (R.to(dtype) @ cell).unsqueeze(1)).reshape(copies * n, 3)
    target = R.unsqueeze(1) + shift.unsqueeze(0)                          # [copies, E, 3]
    wrapped = torch.remainder(target, M)
    crossing = torch.div(target - wrapped, M, rounding_mode="floor")
    partner = (wrapped[..., 0] * M[1] + wrapped[..., 1]) * M[2] + wrapped[..., 2]
    own = torch.arange(copies, device=device).unsqueeze(1)
    big_index = torch.stack([(partner * n + src.unsqueeze(0)).reshape(-1),
                             (own * n + dst.unsqueeze(0)).expand(copies, n_edges).reshape(-1)])
    return {
        "copies": copies,
        "h": h.repeat(copies, *([1] * (h.dim() - 1))),
        "pos": big_pos,
        "cell": M.to(dtype).unsqueeze(1) * cell,
        "edge_index": big_index,
        "edge_shift": crossing.reshape(copies * n_edges, 3).to(edge_shift.dtype),
        "edge_vec": edge_vec.repeat(copies, 1),
        "edge_len": edge_len.repeat(copies),
        "species": None if species is None else torch.as_tensor(species, device=device).repeat(copies),
    }
