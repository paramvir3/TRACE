"""solver='auto': the measured per-structure route rule, and that it IS that route.

``auto`` must be bit-identical to the explicit route it resolves to -- it only
chooses, it never computes differently -- for an open molecule below and above
the dense threshold, a 3-D periodic cell, a slab and a dispersion run.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import torch

from conftest import require


def _water(n, box=None, vacuum_axis=None, seed=0):
    rng = np.random.default_rng(seed)
    half = math.radians(104.52) / 2.0
    mol = np.array([[0, 0, 0], [0.9572 * math.sin(half), 0, 0.9572 * math.cos(half)],
                    [-0.9572 * math.sin(half), 0, 0.9572 * math.cos(half)]])
    side = math.ceil(n ** (1 / 3))
    spacing = 3.1 if box is None else box / side
    pos = []
    for k in range(n):
        i, j, l = k // (side * side), (k // side) % side, k % side
        pos.append(mol + np.array([i, j, l]) * spacing + 0.1 * rng.standard_normal(3))
    return torch.tensor(np.concatenate(pos)), torch.tensor(np.tile([8, 1, 1], n))


def _state(config, pos, species, *, cell=None, pbc=None, dispersion=False):
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    electrostatic_state = require("energy", "electrostatic_state")
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heads = ElectrostaticHeads(r_max=4.0, charge_transfer_decay=3.0, dispersion=dispersion)
    g = torch.Generator().manual_seed(1)
    feats = torch.randn(3, heads.irreps_in.dim, generator=g, dtype=torch.float64)
    feats[:, 64:] = 0.0
    h = feats.repeat(pos.shape[0] // 3, 1)
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    atoms = Atoms(numbers=species.numpy(), positions=pos.numpy(),
                  cell=None if cell is None else cell.numpy(),
                  pbc=False if pbc is None else pbc)
    i, j, shift = neighbor_list("ijS", atoms, 4.0)
    p = pos.clone().requires_grad_(True)
    edge_index = torch.tensor(np.stack([j, i]))
    edge_shift = torch.tensor(shift, dtype=torch.float64)
    vec = p[edge_index[0]] - p[edge_index[1]]
    kwargs = {}
    if pbc is not None:
        vec = vec + edge_shift @ cell
        kwargs = dict(pbc=pbc, cell=cell, edge_shift=edge_shift)
    state = electrostatic_state(h, p, edge_index, vec, vec.norm(dim=-1), heads, config,
                                species=species, **kwargs)
    energy = state.energy + (0.0 if state.dispersion is None else state.dispersion)
    force = -torch.autograd.grad(energy, p)[0]
    return energy.detach(), force


def test_the_route_rule():
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    resolve_solver = require("config", "resolve_solver")
    limit = require("config", "AUTO_DENSE_OPEN_MAX_ATOMS")
    auto = ElectrostaticConfig(solver="auto")
    assert resolve_solver(auto, (False,) * 3, limit) == "dense"
    assert resolve_solver(auto, (False,) * 3, limit + 1) == "matrix_free"
    assert resolve_solver(auto, (True,) * 3, 10) == "matrix_free"
    assert resolve_solver(auto, (True, True, False), 500) == "dense"
    assert resolve_solver(ElectrostaticConfig(solver="auto", dispersion=True), (True,) * 3, 50) == "dense"
    assert resolve_solver(ElectrostaticConfig(solver="dense"), (True,) * 3, 50) == "dense"
    ElectrostaticConfig(solver="auto", reciprocal="pme")          # allowed: periodic -> matrix-free
    with pytest.raises(ValueError, match="solver must be"):
        ElectrostaticConfig(solver="fast")


@pytest.mark.parametrize("case", ["open_small", "open_large", "periodic", "slab", "dispersion"])
def test_auto_is_bit_identical_to_the_route_it_picks(case):
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    if case == "open_small":
        pos, z = _water(8)
        kw, route = {}, "dense"
    elif case == "open_large":
        pos, z = _water(45)                                   # 135 atoms > 128
        kw, route = {}, "matrix_free"
    elif case == "periodic":
        pos, z = _water(8, box=6.2)
        kw, route = dict(cell=torch.eye(3, dtype=torch.float64) * 6.2, pbc=(True, True, True)), "matrix_free"
    elif case == "slab":
        pos, z = _water(8, box=6.2)
        cell = torch.diag(torch.tensor([6.2, 6.2, 30.0], dtype=torch.float64))
        kw, route = dict(cell=cell, pbc=(True, True, False)), "dense"
    else:
        pos, z = _water(8, box=6.2)
        kw, route = dict(cell=torch.eye(3, dtype=torch.float64) * 6.2, pbc=(True, True, True),
                         dispersion=True), "dense"
    extra = {"dispersion": True} if case == "dispersion" else {}
    e_auto, f_auto = _state(ElectrostaticConfig(solver="auto", **extra), pos, z, **kw)
    e_route, f_route = _state(ElectrostaticConfig(solver=route, **extra), pos, z, **kw)
    assert torch.equal(e_auto, e_route) and torch.equal(f_auto, f_route)
