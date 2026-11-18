"""Per-axis reciprocal grid from EwaldParams.auto_tune (speed-up R2).

``k_cut = 2 alpha sqrt(p)`` defines the accuracy of the reciprocal sum.  A vector
``k = sum_d n_d b_d`` has ``n_d = k . a_d / (2 pi)``, so ``|k| <= k_cut`` implies
``|n_d| <= k_cut |a_d| / (2 pi)``: the box ``|n_d| <= ceil(k_cut |a_d| / 2 pi)`` holds the whole
``k_cut`` sphere for ANY cell.  ``auto_tune`` used ``max_d`` of those for all three directions, which
for the 9.05 x 9.05 x 26.46 A Au2/MgO cell is 15 625 vectors where (4, 4, 12) needs 2 025.

What changes is only the set of vectors with ``|k| > k_cut`` the larger box happened to include;
the blocks move by at most ``sum over the dropped vectors of |w F|`` (each term is bounded by it,
``|cos|, |sin| <= 1``), plus round-off.  ``k_max`` itself is unchanged, and a cell whose three
counts agree keeps ``k_max_per_axis = None`` -- the scalar path, bit for bit.
"""
import itertools
import math

import pytest
import torch
from conftest import require

EPS = torch.finfo(torch.float64).eps
K_E = 14.399645

CELLS = {
    "au2_mgo": [[9.047, 0.0, 0.0], [0.0, 9.047, 0.0], [0.0, 0.0, 26.459]],
    "triclinic": [[7.0, 0.3, 0.0], [0.2, 6.5, 0.4], [0.0, 0.5, 9.5]],
    "sheared": [[6.0, 0.0, 0.0], [4.5, 5.0, 0.0], [0.3, 0.2, 11.0]],
    "cubic": [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
}


def _k_cut(params):
    return 2.0 * params.alpha * math.sqrt(-math.log(params.accuracy))


@pytest.mark.parametrize("name", list(CELLS))
@pytest.mark.parametrize("accuracy", [1e-6, 1e-10])
def test_the_per_axis_box_holds_the_whole_k_cut_sphere(name, accuracy):
    EwaldParams = require("ewald", "EwaldParams")
    cell = torch.tensor(CELLS[name], dtype=torch.float64)
    params = EwaldParams.auto_tune(cell, 10.0, accuracy=accuracy)
    lengths = cell.norm(dim=-1)
    k_cut = _k_cut(params)
    expected = tuple(max(1, math.ceil(k_cut * float(L) / (2 * math.pi))) for L in lengths)
    assert params.k_max == max(expected)                          # unchanged
    if len(set(expected)) == 1:
        assert params.k_max_per_axis is None and params.k_grid == params.k_max
    else:
        assert params.k_max_per_axis == expected and params.k_grid == expected
    grid = params.k_grid if isinstance(params.k_grid, tuple) else (params.k_grid,) * 3
    # every lattice vector inside the sphere, searched in a box two steps wider than the old one
    reach = params.k_max + 2
    n = torch.tensor(list(itertools.product(range(-reach, reach + 1), repeat=3)), dtype=torch.float64)
    k = n @ (2 * math.pi * torch.linalg.inv(cell).T)
    inside = n[k.norm(dim=-1) <= k_cut]
    assert bool((inside.abs() <= torch.tensor(grid, dtype=torch.float64)).all())


def _system():
    g = torch.Generator().manual_seed(4)
    cell = torch.tensor(CELLS["au2_mgo"], dtype=torch.float64) * 0.7          # 6.3 x 6.3 x 18.5 A
    frac = torch.rand(8, 3, generator=g, dtype=torch.float64) * torch.tensor([1.0, 1.0, 0.35], dtype=torch.float64)
    pos = frac @ cell
    n = len(pos)
    sigma = 0.8 + 0.4 * torch.rand(n, generator=g, dtype=torch.float64)
    sigma_mu = 0.7 + 0.4 * torch.rand(n, generator=g, dtype=torch.float64)
    ell = 0.6 + 0.6 * torch.rand(n, generator=g, dtype=torch.float64)
    return pos, cell, sigma, sigma_mu, ell


def test_blocks_move_by_no_more_than_the_dropped_tail():
    lattice = require("lattice")
    ewald = require("ewald")
    pos, cell, sigma, sigma_mu, ell = _system()
    per_axis = ewald.EwaldParams.auto_tune(cell, 9.0, accuracy=1e-6)
    assert per_axis.k_max_per_axis is not None
    isotropic = ewald.EwaldParams(alpha=per_axis.alpha, k_max=per_axis.k_max, r_cut=per_axis.r_cut,
                                  accuracy=per_axis.accuracy, image_shells=per_axis.image_shells)
    new = lattice.gaussian_lattice_blocks(pos, cell, sigma, sigma_mu, ell, per_axis)
    old = lattice.gaussian_lattice_blocks(pos, cell, sigma, sigma_mu, ell, isotropic)

    # the dropped vectors: in the isotropic grid, outside the per-axis box
    k, k_squared, nonzero = ewald._wave_vectors(per_axis.k_max, cell)
    n = torch.round(k @ cell.T / (2 * math.pi))
    box = torch.tensor(per_axis.k_max_per_axis, dtype=torch.float64)
    dropped = nonzero & (n.abs() > box).any(dim=-1)
    w = (4 * math.pi * K_E / float(torch.det(cell).abs())) * ewald._gaussian_weights(k_squared, nonzero, per_axis.alpha)
    kn = k.norm(dim=-1)
    gamma = int(nonzero.sum()) * EPS / (1 - int(nonzero.sum()) * EPS)
    ell_max = max(1.0, float(ell.max())) ** 2
    for key, power in (("j", 0), ("t01", 1), ("t2", 2)):
        tail = 2 * float((w * kn ** power)[dropped].sum()) * ell_max
        roundoff = 2 * gamma * 2 * float((w * kn ** power)[nonzero].sum()) * ell_max
        difference = float((new[key] - old[key]).abs().max())
        assert difference <= tail + roundoff, (key, difference, tail, roundoff)
        assert tail < 1e-6 * float(old[key].abs().max())             # the change is a tail effect
