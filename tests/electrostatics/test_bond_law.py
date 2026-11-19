"""The two-level bond-hardness anchor (docs/ELECTROSTATICS_AUDIT_2026-10-09.md, step 1(b)).

One occupied and one empty orbital across a bond, level offset Delta and hopping h, bonding orbital
doubly occupied: N_B = 1 - Delta / G, G = sqrt(Delta^2 + 4 h^2), so dN_B / dDelta = -4 h^2 / G^3
and the split-charge hardness is kappa = G^3 / (4 h^2) exactly (2026-10-09 band note, Theorem 6).
Anchor: Delta = chi_i - chi_j (Mulliken, NIST IE / EA), h = 1.42 hbar^2 / (m r^2) (Froyen & Harrison,
PRB 20, 2420 (1979), the sp-sigma coefficient).  Learned: a bounded factor exp(L tanh(s / L)),
L = ln(range).  Beyond contact the existing SPEC 23 law stiffens it further.
"""
import math

import pytest
import torch
from conftest import require

HBAR2_OVER_M = 7.619964      # eV A^2


def _chi(Z):
    IE, EA = require("constants", "ATOMIC_IONIZATION_EV", "ATOMIC_ELECTRON_AFFINITY_EV")
    return 0.5 * (IE[Z] + EA[Z])


def _anchor_reference(r, Zi, Zj):
    h = 1.42 * HBAR2_OVER_M / r ** 2
    G = math.sqrt((_chi(Zi) - _chi(Zj)) ** 2 + 4 * h * h)
    return G ** 3 / (4 * h * h)


def test_two_level_law_is_the_exact_two_orbital_response():
    """kappa = -dDelta/dN_B of the exact two-orbital model, by finite difference."""
    h, Delta = 1.3, 6.1
    NB = lambda D: 1.0 - D / math.hypot(D, 2 * h)
    dN = (NB(Delta + 1e-5) - NB(Delta - 1e-5)) / 2e-5
    G = math.hypot(Delta, 2 * h)
    assert abs(-1.0 / dN - G ** 3 / (4 * h * h)) <= 1e-8 * G ** 3 / (4 * h * h)


@pytest.mark.parametrize("r,Zi,Zj", [(2.106, 12, 8), (2.75, 47, 47), (1.09, 6, 1), (2.82, 11, 17), (2.2, 79, 8)])
def test_anchor_values_and_symmetry(r, Zi, Zj):
    anchor = require("bond_law", "two_level_bond_hardness")
    rr = torch.tensor([r, r], dtype=torch.float64)
    k = anchor(rr, torch.tensor([Zi, Zj]), torch.tensor([Zj, Zi]))
    ref = _anchor_reference(r, Zi, Zj)
    assert abs(float(k[0]) - ref) <= 1e-12 * ref and float(k[0]) == float(k[1])


def test_homonuclear_anchor_is_twice_the_hopping():
    anchor = require("bond_law", "two_level_bond_hardness")
    r = torch.tensor([2.5, 2.9, 3.4], dtype=torch.float64)
    z = torch.tensor([47, 47, 47])
    h = 1.42 * HBAR2_OVER_M / r ** 2
    assert torch.allclose(anchor(r, z, z), 2 * h, rtol=1e-14, atol=0.0)


def test_anchor_gradient_is_exact():
    anchor = require("bond_law", "two_level_bond_hardness")
    r = torch.tensor([2.106, 1.8, 3.1], dtype=torch.float64, requires_grad=True)
    zi, zj = torch.tensor([12, 6, 79]), torch.tensor([8, 1, 8])
    (g,) = torch.autograd.grad(anchor(r, zi, zj).sum(), r)
    for b in range(3):
        rp, rm = r.detach().clone(), r.detach().clone()
        rp[b] += 1e-6
        rm[b] -= 1e-6
        fd = float((anchor(rp, zi, zj).sum() - anchor(rm, zi, zj).sum()) / 2e-6)
        assert abs(float(g[b]) - fd) <= 1e-7 * (1 + abs(fd))


def _heads(**kwargs):
    import warnings
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return require("heads", "ElectrostaticHeads")(r_max=5.0, **kwargs)


def _kappa(heads, z, pos):
    n = len(z)
    ei = torch.tensor([(i, j) for i in range(n) for j in range(n) if i != j and float((pos[i] - pos[j]).norm()) < 4.5]).T
    lo, hi = ei[0][ei[0] < ei[1]], ei[1][ei[0] < ei[1]]
    bond_index = torch.stack((lo, hi))
    length = (pos[hi] - pos[lo]).norm(dim=1)
    torch.manual_seed(1)
    h = torch.randn(n, heads.irreps_in.dim, dtype=torch.float64)
    readout = heads(h, bond_index, length, species=z)
    return readout.params.kappa, bond_index, length


def test_heads_default_is_unchanged_and_two_level_is_bounded():
    anchor = require("bond_law", "two_level_bond_hardness")
    z = torch.tensor([12, 8, 12, 8, 79, 47])
    g = torch.Generator().manual_seed(3)
    pos = 3.0 * torch.rand(6, 3, generator=g, dtype=torch.float64)
    k_default, _, _ = _kappa(_heads(), z, pos)
    k_free, _, _ = _kappa(_heads(bond_hardness_law="free"), z, pos)
    assert torch.equal(k_default, k_free)
    heads = _heads(bond_hardness_law="two_level", bond_hardness_range=10.0)
    k, bond_index, length = _kappa(heads, z, pos)
    a = anchor(length, z[bond_index[0]], z[bond_index[1]])
    floor = heads.kappa_floor
    assert bool(((k - floor) >= a / 10.0 * (1 - 1e-12)).all()) and bool(((k - floor) <= a * 10.0 * (1 + 1e-12)).all())
    with pytest.raises(ValueError):
        _heads(bond_hardness_law="metallic")
    with pytest.raises(ValueError):
        _heads(bond_hardness_law="two_level", bond_hardness_range=1.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_eps_inf_certificate_for_any_learned_value(seed):
    """Screening + the two-level law bound eps_inf for EVERY head output (Theorem 1 with kappa >= anchor/range).

    Rocksalt MgO: second-shell bonds carry exactly zero split charge (screening), every remaining bond
    has kappa >= kappa_anchor / range, so eps_inf <= 1 + 4 pi K_E / Omega lambda_max(sum_b p_b p_b^T
    range / kappa_anchor) + (eps_dip - 1), whatever the network learned.  Untrained heads at three
    seeds stand in for "anything training could produce"; with a free head the same cell gave 1517.
    """
    import sys
    sys.path.insert(0, "tests/electrostatics")
    from test_periodic import _config, _state
    import warnings
    anchor = require("bond_law", "two_level_bond_hardness")
    dielectric_tensor = require("observables", "dielectric_tensor")
    K_E = 14.399645
    a = 4.212
    frac = torch.tensor([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                         [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]], dtype=torch.float64)
    cell = a * torch.eye(3, dtype=torch.float64)
    pos = frac @ cell
    z = torch.tensor([12, 12, 12, 12, 8, 8, 8, 8])
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heads = require("heads", "ElectrostaticHeads")(r_max=5.0, bond_hardness_law="two_level")
    h = torch.randn(8, heads.irreps_in.dim, dtype=torch.float64)
    config = _config(ewald_real_cutoff=12.0, solver="dense", bond_screening=(2.0, 2.8))
    electrostatic_state = require("energy", "electrostatic_state")
    from test_periodic import _neighbours
    ei, shift, vec = _neighbours(pos, cell, 3.2)
    state = electrostatic_state(h, pos, ei, vec, vec.norm(dim=-1), heads, config, pbc=(True, True, True),
                                cell=cell, edge_shift=shift, species=z)
    asm = state.assembly
    eps = dielectric_tensor(state.solve_result, asm, route="minimiser").detach()
    p = asm.bond_dipole.detach()
    lo, hi = asm.bond_index[0], asm.bond_index[1]
    length = torch.linalg.norm(asm.bond_dipole.detach(), dim=1)
    nb, n = asm.n_bonds, asm.n_atoms
    H = asm.dense_hessian().detach()
    mu = torch.arange(nb + n, H.shape[0])
    D = asm.dipole_derivative().detach()[mu]
    dip = D.T @ torch.linalg.solve(H[mu][:, mu], D)
    # the first-shell bonds are unscreened (|p| = bond length) so the anchor is evaluated there
    live = length > 1e-12
    k_min = anchor(length[live], z[lo[live]], z[hi[live]]) / heads.bond_hardness_range
    bound_bond = (p[live] / k_min.unsqueeze(1)).T @ p[live]
    bound = 1.0 + 4 * math.pi * K_E / float(torch.linalg.det(cell)) * float(torch.linalg.eigvalsh(bound_bond + dip).max())
    assert float(torch.linalg.eigvalsh(0.5 * (eps + eps.T)).max()) <= bound * (1 + 1e-12)
    assert bound < 100.0
