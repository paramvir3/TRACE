"""SPEC 23: the charge-transfer contact law of the bond hardness.

    kappa *= exp(ln(K) sigmoid(x) + beta w softplus(x)),  x = (r - R_i - R_j - delta)/w

Checked: off by default and then bit-identical; the three regimes (1 in contact,
a jump of K across the onset, exp(beta r) beyond); exchange symmetry; exact
forces; and the property it exists for -- the polarizability of a water chain
becomes size-extensive (per-molecule alpha_iso grows +20 % from 1 to 6 molecules
with the law, x3.5 without, same heads).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import torch

from conftest import require


def _heads(decay=None, **kwargs):
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ElectrostaticHeads(r_max=5.0, charge_transfer_decay=decay, **kwargs)


def _features(heads, n, seed=1):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(3, heads.irreps_in.dim, generator=g, dtype=torch.float64)
    base[:, 64:] = 0.0
    return base.repeat(n, 1)


def _water_chain(n, spacing=2.9):
    mol = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
    pos = np.concatenate([mol + [k * spacing, 0.0, 0.0] for k in range(n)])
    return torch.tensor(pos), torch.tensor(np.tile([8, 1, 1], n))


def _state(heads, pos, species, cutoff=5.0, **kwargs):
    electrostatic_state = require("energy", "electrostatic_state")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    d = torch.cdist(pos.detach(), pos.detach())
    src, dst = torch.nonzero((d < cutoff) & (d > 0), as_tuple=True)
    edge_index = torch.stack([src, dst])
    vec = pos[edge_index[0]] - pos[edge_index[1]]
    n = pos.shape[0] // 3
    return electrostatic_state(_features(heads, n), pos, edge_index, vec, vec.norm(dim=-1), heads,
                               ElectrostaticConfig(), species=species, **kwargs)


def _law(heads, r, zi, zj):
    return float(heads.charge_transfer_law(torch.tensor([[0], [1]]), torch.tensor([float(r)]),
                                           torch.tensor([zi, zj])))


def test_the_law_is_off_by_default_and_the_model_is_unchanged():
    legacy = _heads()
    assert legacy.charge_transfer_law(torch.tensor([[0], [1]]), torch.tensor([2.0]), None) is None
    pos, species = _water_chain(2)
    with_species = _state(legacy, pos, species)
    without = _state(legacy, pos, None)
    assert torch.equal(with_species.energy, without.energy)
    assert torch.equal(with_species.assembly.params.kappa, without.assembly.params.kappa)


def test_the_three_regimes_of_the_law():
    beta, K, delta, w = 3.0, 1.0e3, 0.5, 0.1
    heads = _heads(beta)
    R = require("constants", "COVALENT_RADII")
    contact = R[8] + R[8] + delta
    # In contact: ~1 (O-H 0.96 A, 1.04; Na-Cl 2.82 A, 1.21).
    assert 1.0 <= _law(heads, 0.96, 8, 1) < 1.05
    assert 1.0 <= _law(heads, 2.82, 11, 17) < 1.25
    # Beyond the onset: K exp(beta (r - contact)) to within the softplus offset.
    for excess in (1.0, 2.0):
        expected = K * math.exp(beta * excess)
        assert abs(_law(heads, contact + excess, 8, 8) / expected - 1.0) < 1e-3
    # Tail slope d ln(law)/dr = beta.
    h = 1e-4
    slope = (math.log(_law(heads, contact + 2.0 + h, 8, 8))
             - math.log(_law(heads, contact + 2.0 - h, 8, 8))) / (2 * h)
    assert abs(slope - beta) < 1e-6
    # Exchange symmetric, bit for bit.
    assert _law(heads, 1.9, 8, 1) == _law(heads, 1.9, 1, 8)
    with pytest.raises(ValueError, match="species"):
        heads.charge_transfer_law(torch.tensor([[0], [1]]), torch.tensor([2.0]), None)
    with pytest.raises(ValueError, match="atomic numbers"):
        heads.charge_transfer_law(torch.tensor([[0], [1]]), torch.tensor([2.0]), torch.tensor([0, 8]))


def test_water_chain_polarizability_becomes_size_extensive():
    polarizability = require("observables", "polarizability")

    def per_molecule(heads):
        values = []
        for n in (1, 6):
            pos, species = _water_chain(n)
            st = _state(heads, pos, species)
            alpha = polarizability(st.solve_result, st.assembly, as_volume=True)
            values.append(float(torch.trace(alpha)) / 3.0 / n)
        return values[1] / values[0]

    growth_without = per_molecule(_heads())
    growth_with = per_molecule(_heads(3.0))
    assert growth_without > 3.0          # measured x3.5: the charge-transfer catastrophe
    assert growth_with < 1.3             # measured x1.20: induced-dipole coupling only


def test_forces_through_the_law_match_finite_differences():
    heads = _heads(3.0)
    pos0, species = _water_chain(3)
    pos = pos0.clone().requires_grad_(True)
    energy = _state(heads, pos, species).energy
    force = -torch.autograd.grad(energy, pos)[0]
    h = 1e-5
    for atom, axis in ((1, 0), (3, 0), (5, 1)):
        up, down = pos0.clone(), pos0.clone()
        up[atom, axis] += h
        down[atom, axis] -= h
        fd = -(float(_state(heads, up, species).energy) - float(_state(heads, down, species).energy)) / (2 * h)
        assert abs(float(force[atom, axis]) - fd) < 1e-8, (atom, axis, float(force[atom, axis]), fd)
    # The law is in the force: the hardness of the hydrogen-bond pairs it
    # stiffens depends on their length.
    kappa = _state(heads, pos0, species).assembly.params.kappa
    assert float(kappa.max() / kappa.min()) > 1e3


def test_the_hook_reads_the_atomic_numbers_from_the_batch():
    ElectrostaticModule = require("energy", "ElectrostaticModule")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    heads = _heads(3.0)
    pos, species = _water_chain(2)
    d = torch.cdist(pos, pos)
    src, dst = torch.nonzero((d < 5.0) & (d > 0), as_tuple=True)
    edge_index = torch.stack([src, dst])
    vec = pos[edge_index[0]] - pos[edge_index[1]]
    module = ElectrostaticModule(heads, ElectrostaticConfig()).eval()
    energy, _ = module(_features(heads, 2), pos, edge_index, vec, vec.norm(dim=-1), data={"z": species})
    assert abs(float(energy - _state(heads, pos, species).energy)) < 1e-12
    with pytest.raises(ValueError, match="atomic numbers"):
        module(_features(heads, 2), pos, edge_index, vec, vec.norm(dim=-1), data={})
