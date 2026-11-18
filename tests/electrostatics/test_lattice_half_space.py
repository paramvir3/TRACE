"""Reciprocal lattice sums over half of k-space (speed-up R1; exact).

Every reciprocal block of ``gaussian_lattice_blocks`` is

    sum_{k != 0} w(k) F(k) g(k . (r_i - r_j)),

with ``w(-k) = w(k)`` and either an EVEN factor F (1, k k, k.W.k, ...) with ``g = cos`` or an ODD
factor (k, k (k.W.k)) with ``g = sin``.  In both cases the summand is invariant under k -> -k, so the
sum over all k != 0 is exactly twice the sum over one member of each pair {k, -k}.  The half-space is
built directly (first non-zero integer coordinate positive), so its size is the static
((2a+1)(2b+1)(2c+1) - 1) / 2.

Tolerances are the rigorous summation bound |fl(sum) - sum| <= gamma_n sum |terms|, gamma_n =
n u / (1 - n u), u = eps / 2, applied to both evaluations; |cos|, |sin| <= 1 bound every term by
|w F| (twice, for the two products of each trigonometric identity).
"""
import itertools
import math

import pytest
import torch
from conftest import require

EPS = torch.finfo(torch.float64).eps


def _gamma(n):
    u = EPS / 2
    return n * u / (1 - n * u)


@pytest.mark.parametrize("k_grid", [1, 3, (2, 1, 4), (1, 3, 1)])
def test_half_space_holds_one_of_each_pair_and_no_origin(k_grid):
    half = require("ewald", "_half_wave_vectors")
    cell = torch.tensor([[5.0, 0.2, 0.0], [0.1, 6.0, 0.3], [0.0, 0.4, 7.0]], dtype=torch.float64)
    k, k_squared = half(k_grid, cell)
    a, b, c = (k_grid,) * 3 if isinstance(k_grid, int) else k_grid
    integers = torch.round(k @ cell.T / (2 * math.pi)).to(torch.int64)      # n_j = k . a_j / 2 pi
    assert torch.allclose(integers.to(torch.float64) @ (2 * math.pi * torch.linalg.inv(cell).T), k,
                          rtol=0.0, atol=64 * EPS * float(k.abs().max()))
    pairs = {tuple(n) for n in integers.tolist()}
    full = {n for n in itertools.product(range(-a, a + 1), range(-b, b + 1), range(-c, c + 1)) if any(n)}
    assert len(pairs) == len(integers) == ((2 * a + 1) * (2 * b + 1) * (2 * c + 1) - 1) // 2
    assert pairs.isdisjoint({tuple(-x for x in n) for n in pairs})
    assert pairs | {tuple(-x for x in n) for n in pairs} == full
    assert torch.equal(k_squared, (k * k).sum(-1))


def _system(case, theta):
    g = torch.Generator().manual_seed(11)
    if case == "slab":
        cell = torch.tensor([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 16.0]], dtype=torch.float64)
        frac = torch.rand(7, 3, generator=g, dtype=torch.float64) * torch.tensor([1.0, 1.0, 0.4], dtype=torch.float64)
    else:
        cell = torch.tensor([[7.0, 0.3, 0.0], [0.2, 6.5, 0.4], [0.0, 0.5, 7.5]], dtype=torch.float64)
        frac = torch.rand(6, 3, generator=g, dtype=torch.float64)
    pos = (frac @ cell).requires_grad_(True)
    n = len(pos)
    sigma = 0.8 + 0.4 * torch.rand(n, generator=g, dtype=torch.float64)
    sigma_mu = 0.7 + 0.4 * torch.rand(n, generator=g, dtype=torch.float64)
    ell = 0.6 + 0.6 * torch.rand(n, generator=g, dtype=torch.float64)
    sigma_theta = (0.9 + 0.3 * torch.rand(n, generator=g, dtype=torch.float64)) if theta else None
    return pos, cell, sigma, sigma_mu, ell, sigma_theta


def _full_terms(k_grid, cell):
    """The reference: the whole grid, origin masked, every vector counted once."""
    k, k_squared, nonzero = require("ewald", "_wave_vectors")(k_grid, cell)
    return k, k_squared, nonzero, 1.0


@pytest.mark.parametrize("case", ["slab", "triclinic"])
@pytest.mark.parametrize("theta", [False, True])
def test_blocks_and_their_position_gradients_equal_the_full_grid(case, theta, monkeypatch):
    lattice = require("lattice")
    EwaldParams = require("ewald", "EwaldParams")
    pos, cell, sigma, sigma_mu, ell, sigma_theta = _system(case, theta)
    ewald = EwaldParams.auto_tune(cell, 9.0, accuracy=1e-8)
    g = torch.Generator().manual_seed(3)

    def evaluate():
        blocks = lattice.gaussian_lattice_blocks(pos, cell, sigma, sigma_mu, ell, ewald, sigma_theta=sigma_theta)
        probes = {key: torch.randn(block.shape, generator=g, dtype=torch.float64) for key, block in sorted(blocks.items())}
        scalar = sum((probes[key] * blocks[key]).sum() for key in probes)
        (grad,) = torch.autograd.grad(scalar, pos)
        return {key: block.detach() for key, block in blocks.items()}, grad, probes

    g.manual_seed(3)
    half_blocks, half_grad, probes = evaluate()
    monkeypatch.setattr(lattice, "_reciprocal_terms", _full_terms)
    g.manual_seed(3)
    full_blocks, full_grad, _ = evaluate()

    # sum_k |w_k| |F_k| (|k|) per block, on the full grid, as the bound's scale
    k, k_squared, nonzero = require("ewald", "_wave_vectors")(ewald.k_grid, cell)
    w = (4 * math.pi * 14.399645 / float(torch.det(cell).abs())) * require("ewald", "_gaussian_weights")(
        k_squared, nonzero, ewald.alpha).abs()
    kn = k.norm(dim=-1)
    n_k = int(nonzero.sum())
    ell_max = float(ell.max())
    # |F| <= |k|^p for the block's power p of k; ell factors and the 1/2, 1/4 of the quadrupole blocks <= max(1, ell^4)
    power = {"j": 0, "t01": 1, "t2": 2, "t02": 2, "t12": 3, "t22": 4}
    for key in full_blocks:
        scale = float((w * kn ** power[key]).sum()) * max(1.0, ell_max) ** 4
        bound = 2 * _gamma(n_k) * 2 * scale + 1e-300
        assert float((half_blocks[key] - full_blocks[key]).abs().max()) <= bound, key
    grad_scale = sum(float((w * kn ** (power[key] + 1)).sum()) * max(1.0, ell_max) ** 4 * float(probes[key].abs().sum())
                     for key in full_blocks)
    assert float((half_grad - full_grad).abs().max()) <= 2 * _gamma(n_k) * 2 * grad_scale


def test_half_space_halves_the_reciprocal_work():
    lattice = require("lattice")
    EwaldParams = require("ewald", "EwaldParams")
    pos, cell, sigma, sigma_mu, ell, _ = _system("slab", False)
    ewald = EwaldParams.auto_tune(cell, 9.0, accuracy=1e-8)
    k, _k2, _nz, multiplicity = lattice._reciprocal_terms(ewald.k_grid, cell)
    grid = (ewald.k_grid,) * 3 if isinstance(ewald.k_grid, int) else ewald.k_grid
    full = math.prod(2 * n + 1 for n in grid)
    assert len(k) == (full - 1) // 2 and multiplicity == 2.0
