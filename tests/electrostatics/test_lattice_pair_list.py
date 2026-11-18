"""Real-space lattice sums over the pair-images inside the round-off radius (speed-up R3).

The image box of ``gaussian_lattice_blocks`` holds ~75 images of every pair for the Au2/MgO cell
(907 500 pair-images), but almost all of them lie where every kernel is far below one unit in the
last place.  Every real-space factor is a difference of regularized upper incomplete gammas,

    lambda_{2n+1}(r) = Q(n + 1/2, alpha^2 r^2) - Q(n + 1/2, r^2 / s^2),   Q in [0, 1], decreasing,

(the charge kernel is the n = -1 case, (erfc(alpha r) - erfc(r / s)) / r), so
|lambda| <= Q(a, kappa^2 r^2), kappa = min(alpha, 1 / s_max), and

    Q(a, x) <= x^(a-1) e^(-x) / (Gamma(a) (1 - max(a - 1, 0) / x))     (x > max(a - 1, 0)).

The round-off radius R0 is the smallest r at which n_images * C * that bound <= 2^-70, C bounding every
kernel coefficient and basis contraction (r >= 1 A makes every 1/r^p <= 1).  Dropping every
pair-image beyond R0 changes a block entry by at most 2^-70 K_E ~ 1.2e-20 eV: the box sum, to
round-off.  The tests check the bound itself against mpmath, and the blocks and their position and
cell gradients against the same code with R0 = infinity (the whole box).
"""
import math

import pytest
import torch
from conftest import require

mpmath = pytest.importorskip("mpmath")
EPS = torch.finfo(torch.float64).eps
K_E = 14.399645


@pytest.mark.parametrize("a", [0.5, 1.5, 2.5, 3.5, 4.5])
@pytest.mark.parametrize("x", [3.0, 10.0, 30.0, 60.0, 90.0])
def test_the_tail_bound_is_an_upper_bound(a, x):
    bound = require("lattice", "_upper_gamma_tail_bound")
    mpmath.mp.dps = 40
    exact = float(mpmath.gammainc(a, x, mpmath.inf, regularized=True))
    assert exact <= bound(a, x)                                     # an upper bound everywhere
    if x >= 10.0:                                                   # and a tight one where R0 is found (x ~ 50)
        assert bound(a, x) <= 1.07 * exact


@pytest.mark.parametrize("theta", [False, True])
def test_the_radius_meets_its_criterion_and_is_the_smallest_that_does(theta):
    radius = require("lattice", "_round_off_radius")
    bound = require("lattice", "_upper_gamma_tail_bound")
    constant = require("lattice", "_KERNEL_BOUND")
    alpha, widest, n_images = 0.3717, 2.4, 75
    a = 4.5 if theta else 2.5
    r0 = radius(alpha, widest, theta, n_images)
    kappa = min(alpha, 1.0 / widest)
    assert r0 >= 1.0
    assert n_images * constant(theta) * bound(a, (kappa * r0) ** 2) <= 2.0 ** -70
    assert n_images * constant(theta) * bound(a, (kappa * 0.99 * r0) ** 2) > 2.0 ** -70


def _slab(theta):
    g = torch.Generator().manual_seed(9)
    cell = torch.tensor([[6.2, 0.0, 0.0], [0.0, 6.2, 0.0], [0.0, 0.0, 18.0]], dtype=torch.float64,
                        requires_grad=True)
    frac = torch.rand(8, 3, generator=g, dtype=torch.float64) * torch.tensor([1.0, 1.0, 0.4], dtype=torch.float64)
    pos = (frac @ cell.detach()).requires_grad_(True)
    n = len(pos)
    sigma = 0.8 + 0.4 * torch.rand(n, generator=g, dtype=torch.float64)
    sigma_mu = 0.7 + 0.4 * torch.rand(n, generator=g, dtype=torch.float64)
    ell = 0.6 + 0.6 * torch.rand(n, generator=g, dtype=torch.float64)
    sigma_theta = (0.9 + 0.3 * torch.rand(n, generator=g, dtype=torch.float64)) if theta else None
    return pos, cell, sigma, sigma_mu, ell, sigma_theta


def _recording(lattice, store):
    """Wrap ``_near_pair_geometry``: keep (i, j, r, shift) and the gradient reaching each r_vec."""
    original = lattice._near_pair_geometry

    def wrapped(pos, cell, ewald, radius):
        i, j, r_vec, r = original(pos, cell, ewald, radius)
        shift = torch.round((r_vec.detach() - (pos.detach()[i] - pos.detach()[j])) @ torch.linalg.inv(cell.detach()))
        record = {"i": i, "j": j, "r": r.detach(), "shift": shift, "radius": radius}
        if r_vec.requires_grad:
            r_vec.register_hook(lambda g: record.__setitem__("g", g.detach().clone()))
        store.append(record)
        return i, j, r_vec, r
    return wrapped


def _per_atom(record, values, n):
    """sum over the pair-images touching each atom of ``values`` [m]: the terms of d/dr_i."""
    return torch.zeros(n, dtype=values.dtype).index_add(0, record["i"], values).index_add(0, record["j"], values)


@pytest.mark.parametrize("theta", [False, True])
def test_blocks_and_gradients_equal_the_whole_box(theta, monkeypatch):
    """Rigorous bounds (u = eps / 2, gamma_M = M u / (1 - M u)):

    * block entry: |near - whole| <= 2^-70 K_E ell^4 (the dropped tail, by the radius) + the
      round-off of summing at most c terms each way, 2 gamma_c sum_k |v_k|, with
      |v_k| <= C K_E max(1/r_k, 1/r_k^5) ell^4 for every kernel (C = _KERNEL_BOUND);
    * gradients: the per-pair terms g_k = dS/dr_vec_k are captured, so
      |d grad_pos_a| <= sum_dropped |g_k| + gamma_M (sum_near |g_k| + sum_whole |g_k|) + the
      final addition of the (identical) reciprocal part, and the same with |shift_k| for the cell.
    """
    lattice = require("lattice")
    EwaldParams = require("ewald", "EwaldParams")
    pos, cell, sigma, sigma_mu, ell, sigma_theta = _slab(theta)
    ewald = EwaldParams.auto_tune(cell.detach(), 10.0, accuracy=1e-6)
    n = len(pos)
    g = torch.Generator()

    def evaluate():
        store = []
        monkeypatch.setattr(lattice, "_near_pair_geometry", _recording(original_geometry, store))
        blocks = lattice.gaussian_lattice_blocks(pos, cell, sigma, sigma_mu, ell, ewald, sigma_theta=sigma_theta)
        g.manual_seed(2)
        probes = {key: torch.randn(block.shape, generator=g, dtype=torch.float64) for key, block in sorted(blocks.items())}
        scalar = sum((probes[key] * blocks[key]).sum() for key in probes)
        grad_pos, grad_cell = torch.autograd.grad(scalar, (pos, cell))
        return {k: b.detach() for k, b in blocks.items()}, grad_pos, grad_cell, store[-1]

    class _Original:                                      # the unwrapped function, for both runs
        _near_pair_geometry = staticmethod(lattice._near_pair_geometry)
    original_geometry = _Original

    near, near_pos, near_cell, rec_near = evaluate()
    radius = rec_near["radius"]
    monkeypatch.setattr(lattice, "_round_off_radius", lambda *args: math.inf)
    whole, whole_pos, whole_cell, rec_whole = evaluate()
    m_near, m_whole = len(rec_near["r"]), len(rec_whole["r"])
    dropped = rec_whole["r"] > radius
    assert 0 < m_near < m_whole and int((~dropped).sum()) == m_near     # the same pairs, minus the far ones

    u = EPS / 2
    def gamma(M):
        return M * u / (1 - M * u)
    ell4 = max(1.0, float(ell.max())) ** 4
    # per-block kernel constants: |q q| <= K_E/r, |q mu| <= K_E/r^2, |mu mu| <= (1 + 3) K_E/r^3; the
    # quadrupole blocks take the conservative _KERNEL_BOUND (one constant for all of them)
    constants = {"j": 1.0, "t01": 1.0, "t2": 4.0}
    r = rec_whole["r"]
    inverse = torch.maximum(1.0 / r, 1.0 / r ** 5)
    n_images = math.prod(2 * s + 1 for s in ewald.image_shells)
    for key in whole:
        v_bound = constants.get(key, lattice._KERNEL_BOUND(True)) * K_E * inverse * ell4
        per_entry = torch.zeros(n * n, dtype=torch.float64).index_add(
            0, rec_whole["i"] * n + rec_whole["j"], v_bound).view(n, n)
        diff = (near[key] - whole[key]).abs()
        while diff.dim() > 2:
            diff = diff.amax(dim=-1)
        bound = 2.0 ** -70 * K_E * ell4 + 2 * gamma(n_images) * per_entry
        assert bool((diff <= bound).all()), (key, float((diff / bound).max()))

    g_whole, g_near = rec_whole["g"].norm(dim=-1), rec_near["g"].norm(dim=-1)
    M = 2 * m_whole
    pos_bound = (_per_atom({"i": rec_whole["i"][dropped], "j": rec_whole["j"][dropped]}, g_whole[dropped], n)
                 + gamma(M) * (_per_atom(rec_near, g_near, n) + _per_atom(rec_whole, g_whole, n)))
    pos_bound = pos_bound.unsqueeze(-1) + 4 * EPS * (near_pos.abs() + whole_pos.abs())
    assert bool(((near_pos - whole_pos).abs() <= pos_bound).all())
    s_whole, s_near = rec_whole["shift"].abs().sum(-1), rec_near["shift"].abs().sum(-1)
    cell_bound = (float((s_whole * g_whole)[dropped].sum())
                  + gamma(M) * (float((s_near * g_near).sum()) + float((s_whole * g_whole).sum())))
    cell_bound = cell_bound + 4 * EPS * (near_cell.abs() + whole_cell.abs())
    assert bool(((near_cell - whole_cell).abs() <= cell_bound).all())


def test_the_au2_mgo_box_keeps_a_minority_of_its_pair_images():
    """The speed claim, measured on the actual training cell: R0 ~ 20 A keeps under a third."""
    lattice = require("lattice")
    EwaldParams = require("ewald", "EwaldParams")
    ase_io = pytest.importorskip("ase.io")
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "benchmarks_4g", "data", "AuMgO_train.xyz")
    if not os.path.exists(path):
        pytest.skip("benchmark data not present")
    atoms = ase_io.read(path, "0")
    cell = torch.tensor(atoms.cell.array, dtype=torch.float64)
    pos = torch.tensor(atoms.positions, dtype=torch.float64)
    n = len(pos)
    ewald = EwaldParams.auto_tune(cell, 10.0, accuracy=1e-6)
    ones = torch.ones(n, dtype=torch.float64)
    widest = math.sqrt(2.0)
    n_images = math.prod(2 * s + 1 for s in ewald.image_shells)
    radius = lattice._round_off_radius(ewald.alpha, widest, False, n_images)
    i, j, r_vec, r = lattice._near_pair_geometry(pos, cell, ewald, radius)
    box = n * n * n_images - n
    assert len(r) < box / 3, (len(r), box, radius)
    assert float(r.max()) <= radius
