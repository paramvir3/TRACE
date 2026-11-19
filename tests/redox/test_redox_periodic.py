"""TRACE-Redox under periodic boundaries (trace_redox.bound.periodic_free_charge_blocks).

The plane-wave Gram blocks are checked against trace_electrostatics' own Ewald
lattice blocks (a different construction: real + reciprocal space + self and
background terms), against the open-boundary limit of a large cell, and the
module's forces and stress against finite differences.
"""

import math

import pytest
import torch
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)
from redox_helpers import lattice_graph, make_heads, periodic_edges

from trace_electrostatics.config import ElectrostaticConfig
from trace_electrostatics.kernels import gram_self_energy
from trace_redox import RedoxConfig, RedoxElectrostatics
from trace_redox.bound import (
    BoundSectorError,
    assemble_neutral,
    charge_coulomb_block,
    periodic_charge_gram,
    periodic_free_charge_blocks,
)

TRICLINIC = torch.tensor([[7.0, 0.3, 0.0], [0.2, 6.5, 0.1], [0.0, 0.4, 7.5]])
SLAB = torch.tensor([[7.0, 0.3, 0.0], [0.2, 6.5, 0.0], [0.0, 0.0, 20.0]])
CONFIG = ElectrostaticConfig(ewald_accuracy=1e-12)
Z6 = torch.tensor([8, 1, 1, 11, 17, 6])


def _system(pbc, cell, quadrupoles=False, seed=4, n=6):
    heads = make_heads(quadrupoles=quadrupoles)
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(n, 3, generator=g) * torch.tensor([6.0, 6.0, 4.0])
    h = torch.randn(n, heads.irreps_in.dim, generator=g)
    ei, es = lattice_graph(pos, cell, pbc)
    return heads, pos, h, ei, es


@pytest.mark.parametrize("pbc,cell", [((True, True, True), TRICLINIC), ((True, True, False), SLAB)])
@pytest.mark.parametrize("quadrupoles", [False, True])
def test_equal_widths_reproduce_the_ewald_lattice_blocks(pbc, cell, quadrupoles):
    heads, pos, h, ei, es = _system(pbc, cell, quadrupoles)
    vec, length = periodic_edges(pos, cell, ei, es)
    a = assemble_neutral(h, pos, ei, vec, length, heads, CONFIG, pbc=pbc, cell=cell, edge_shift=es,
                         quadrupoles=quadrupoles)
    J, A = periodic_free_charge_blocks(a, a.params.sigma)
    n = a.n_atoms
    J_ref = a.j_pair + torch.diag(gram_self_energy(a.params.sigma))
    blocks = [J_ref @ a._dense_incidence(), a.t01_pair.reshape(n, 3 * n)]
    if quadrupoles:
        blocks.append(a.t02_pair.reshape(n, 5 * n))
    A_ref = torch.cat(blocks, dim=1)
    assert torch.allclose(J, J_ref, rtol=0.0, atol=1e-12 * float(J_ref.abs().max()))
    assert torch.allclose(A, A_ref, rtol=0.0, atol=1e-12 * float(A_ref.abs().max()))


def test_mixed_width_periodic_gram_is_positive_definite():
    """Wide (free) and narrow (bound) Gaussians on one lattice: the joint 2n x 2n Gram matrix > 0,
    its blocks are mutual transposes, and the free-free block is the one the model uses."""
    g = torch.Generator().manual_seed(3)
    cell = TRICLINIC
    pos = torch.rand(5, 3, generator=g) * 5.0
    wide = 1.5 + torch.rand(5, generator=g)
    narrow = 0.6 + 0.3 * torch.rand(5, generator=g)
    J_ww = periodic_charge_gram(pos, cell, wide, wide)
    J_wn = periodic_charge_gram(pos, cell, wide, narrow)
    J_nw = periodic_charge_gram(pos, cell, narrow, wide)
    J_nn = periodic_charge_gram(pos, cell, narrow, narrow)
    assert torch.allclose(J_wn, J_nw.T, atol=1e-13)
    joint = torch.cat((torch.cat((J_nn, J_nw), 1), torch.cat((J_wn, J_ww), 1)), 0)
    assert float(torch.linalg.eigvalsh(joint).min()) > 0.0
    heads, _, h, _, _ = _system((True, True, True), cell, n=5)
    ei, es = lattice_graph(pos, cell, (True, True, True))
    vec, length = periodic_edges(pos, cell, ei, es)
    a = assemble_neutral(h, pos, ei, vec, length, heads, CONFIG, pbc=(True, True, True), cell=cell, edge_shift=es)
    J_model, _ = periodic_free_charge_blocks(a, wide)
    assert torch.allclose(J_model, J_ww, atol=1e-13)


@pytest.mark.parametrize("dipolar", [False, True])
def test_large_cell_limit_is_the_open_boundary_energy(dipolar):
    """Tinfoil lattice sum -> isolated cluster as L grows (de Leeuw, Perram & Smith 1980):
    E_open = E_tinfoil + 2 pi K_E d^2 / (3 V) + O(L^-5) for a neutral cluster with dipole d
    in a cubic cell (the cubic dipole lattice sum vanishes)."""
    g = torch.Generator().manual_seed(11)
    pos = torch.rand(6, 3, generator=g) * 3.0
    wide = 1.5 + torch.rand(6, generator=g)
    f = torch.randn(6, generator=g)
    basis = torch.cat((torch.ones(6, 1), pos), dim=1)                     # 1, x, y, z
    f = f - basis @ torch.linalg.lstsq(basis, f.unsqueeze(1)).solution.squeeze(1)   # neutral, no dipole
    if dipolar:
        f = f + torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])              # neutral, d != 0
    d = f @ pos
    open_energy = float(0.5 * f @ (charge_coulomb_block(pos, wide, wide) @ f))
    errors = []
    for L in (16.0, 24.0, 32.0):
        J = periodic_charge_gram(pos, torch.eye(3) * L, wide, wide)
        estimate = float(0.5 * f @ (J @ f)) + 2.0 * math.pi * 14.399645 * float(d @ d) / (3.0 * L ** 3)
        errors.append(abs(estimate - open_energy))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] <= 2e-4 * max(1.0, abs(open_energy))
    if dipolar:
        # the wrong sign of the surface term would leave 2 x 2 pi K_E d^2 / (3 V)
        wrong = abs(float(0.5 * f @ (periodic_charge_gram(pos, torch.eye(3) * 32.0, wide, wide) @ f))
                    - 2.0 * math.pi * 14.399645 * float(d @ d) / (3.0 * 32.0 ** 3) - open_energy)
        assert wrong > 10 * errors[2]


def _module(**redox):
    heads = make_heads()
    torch.manual_seed(1)
    return RedoxElectrostatics(heads, CONFIG, RedoxConfig(**redox)), heads


def _energy(module, h, pos, cell, pbc, ei, es, z, Q):
    vec, length = periodic_edges(pos, cell, ei, es)
    return module(h, pos, ei, vec, length, pbc=pbc, cell=cell, edge_shift=es, data={"z": z, "total_charge": Q})


@pytest.mark.parametrize("pbc,cell,Q", [((True, True, True), TRICLINIC, 0.0), ((True, True, True), TRICLINIC, 1.0),
                                        ((True, True, False), SLAB, 0.0)])
def test_periodic_forces_are_minus_the_gradient(pbc, cell, Q):
    module, heads = _module()
    _, pos0, h, ei, es = _system(pbc, cell)
    pos = pos0.clone().requires_grad_(True)
    energy, res = _energy(module, h, pos, cell, pbc, ei, es, Z6, Q)
    (grad,) = torch.autograd.grad(energy, pos)
    assert abs(float(res.q.sum()) - Q) <= 1e-10
    step = 1e-5
    for i, a in [(0, 0), (3, 2), (5, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, a] += step
        minus[i, a] -= step
        fd = (float(_energy(module, h, plus, cell, pbc, ei, es, Z6, Q)[0])
              - float(_energy(module, h, minus, cell, pbc, ei, es, Z6, Q)[0])) / (2 * step)
        assert abs(fd - float(grad[i, a])) <= 1e-7 * (1.0 + abs(fd))


def test_periodic_stress_is_the_strain_derivative():
    """dE/d(strain) by autograd through cell, k-vectors and volume vs central differences."""
    pbc, cell0 = (True, True, True), TRICLINIC
    module, heads = _module()
    _, pos0, h, ei, es = _system(pbc, cell0)

    def energy_at(eps6):
        eps = torch.zeros(3, 3)
        eps[0, 0], eps[1, 1], eps[2, 2] = eps6[0], eps6[1], eps6[2]
        eps[0, 1] = eps[1, 0] = eps6[3]
        eps[0, 2] = eps[2, 0] = eps6[4]
        eps[1, 2] = eps[2, 1] = eps6[5]
        deform = torch.eye(3) + eps
        return _energy(module, h, pos0 @ deform, cell0 @ deform, pbc, ei, es, Z6, 0.0)[0]

    eps6 = torch.zeros(6, requires_grad=True)
    (grad,) = torch.autograd.grad(energy_at(eps6), eps6)
    step = 1e-6
    for k in range(6):
        e = torch.zeros(6)
        e[k] = step
        fd = (float(energy_at(e)) - float(energy_at(-e))) / (2 * step)
        assert abs(fd - float(grad[k])) <= 1e-6 * (1.0 + abs(fd))


def test_energy_is_invariant_under_a_lattice_translation_of_one_atom():
    pbc, cell = (True, True, True), TRICLINIC
    module, heads = _module()
    _, pos, h, ei, es = _system(pbc, cell)
    e1, r1 = _energy(module, h, pos, cell, pbc, ei, es, Z6, 1.0)
    moved = pos.clone()
    moved[2] = moved[2] + cell[0] - cell[2]
    ei2, es2 = lattice_graph(moved, cell, pbc)
    e2, r2 = _energy(module, h, moved, cell, pbc, ei2, es2, Z6, 1.0)
    assert abs(float(e1 - e2)) <= 1e-9 * (1.0 + abs(float(e1)))
    assert torch.allclose(r1.q, r2.q, atol=1e-9)


def test_symmetry_equivalent_ions_of_a_rock_salt_crystal_carry_equal_charges():
    """Conventional NaCl cell (8 atoms, a = 5.64 A), identical features per species:
    the four Na (and the four Cl) are equivalent, so their charges must agree."""
    a = 5.64
    frac = torch.tensor([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                         [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, .5, .5]])
    cell = torch.eye(3) * a
    pos = frac * a
    z = torch.tensor([11] * 4 + [17] * 4)
    module, heads = _module()
    g = torch.Generator().manual_seed(2)
    hz = {11: torch.randn(heads.irreps_in.dim, generator=g), 17: torch.randn(heads.irreps_in.dim, generator=g)}
    # scalar-only features keep the input itself symmetric (no 1o / 2e direction)
    for v in hz.values():
        v[heads.scalar_dim:] = 0.0
    h = torch.stack([hz[int(zz)] for zz in z])
    ei, es = lattice_graph(pos, cell, (True, True, True), r_max=3.5)
    energy, res = _energy(module, h, pos, cell, (True, True, True), ei, es, z, 0.0)
    assert float((res.q[:4] - res.q[0]).abs().max()) <= 1e-8
    assert float((res.q[4:] - res.q[4]).abs().max()) <= 1e-8
    assert abs(float(res.q.sum())) <= 1e-10


def test_periodic_refusals():
    module, heads = _module()
    _, pos, h, ei, es = _system((True, True, False), SLAB)
    with pytest.raises(BoundSectorError, match="slab"):
        _energy(module, h, pos, SLAB, (True, True, False), ei, es, Z6, 1.0)
    _, pos, h, ei, es = _system((True, True, True), TRICLINIC)
    vec, length = periodic_edges(pos, TRICLINIC, ei, es)
    with pytest.raises(BoundSectorError, match="field"):
        module(h, pos, ei, vec, length, pbc=(True, True, True), cell=TRICLINIC, edge_shift=es,
               data={"z": Z6, "total_charge": 0.0, "external_field": torch.tensor([0.0, 0.0, 0.01])})
