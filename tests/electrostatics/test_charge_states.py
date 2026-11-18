"""SPEC 21: integer charge localisation through coupled diabatic charge states.

Two identical 3-atom molecules; state A puts +1 on the first, state B on the
second.  Checked: the closed form equals the eigenvalue; symmetry gives equal
weights and E = E_A - |V|; far from the crossing the ground state is the
lower diabatic state (integer charges to O(V^2/dE)); V -> 0 is min(E_A, E_B);
forces through the crossing match finite differences; the weighted charges
conserve the total.
"""

from __future__ import annotations

import functools
import warnings

import pytest
import torch

from conftest import require


@pytest.fixture(scope="module")
def pair():
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heads = ElectrostaticHeads(r_max=3.0)
    g = torch.Generator().manual_seed(4)
    molecule_features = torch.randn(3, heads.irreps_in.dim, generator=g, dtype=torch.float64)
    molecule_features[:, 64:] = 0.0      # scalars only: invariant under the inversion below
    features = torch.cat([molecule_features, molecule_features])
    molecule = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [-0.3, 0.95, 0.0]], dtype=torch.float64)
    return heads, features, molecule


def _positions(molecule, shift):
    """Molecule B is the point inversion of A through (3, 0, 0), so exchanging
    A and B is a symmetry of the pair; ``shift`` compresses A and breaks it."""
    a = molecule * (1.0 + shift)
    b = torch.tensor([6.0, 0.0, 0.0], dtype=torch.float64) - molecule
    return torch.cat([a, b])


def _solve(pair, pos, coupling):
    electrostatic_state = require("energy", "electrostatic_state")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    adiabatic_charge_states = require("charge_states", "adiabatic_charge_states")
    heads, features, _ = pair
    distance = torch.cdist(pos.detach(), pos.detach())
    src, dst = torch.nonzero((distance < 3.0) & (distance > 0), as_tuple=True)
    edge_index = torch.stack([src, dst])
    vec = pos[edge_index[0]] - pos[edge_index[1]]
    state_fn = functools.partial(
        electrostatic_state, features, pos, edge_index, vec, vec.norm(dim=-1), heads, ElectrostaticConfig()
    )
    ids = torch.tensor([0, 0, 0, 1, 1, 1])
    states = [
        {"fragment_id": ids, "fragment_charges": [1.0, 0.0]},
        {"fragment_id": ids, "fragment_charges": [0.0, 1.0]},
    ]
    return adiabatic_charge_states(state_fn, states, coupling)


def test_symmetric_pair_splits_by_twice_the_coupling(pair):
    _, _, molecule = pair
    result = _solve(pair, _positions(molecule, 0.0), 0.05)
    e_a, e_b = result.diabatic_energies
    assert abs(float(e_a - e_b)) < 1e-9
    assert abs(float(result.energy - (e_a - 0.05))) < 1e-9
    assert torch.allclose(result.weights, torch.tensor([0.5, 0.5], dtype=torch.float64), atol=1e-6)
    assert abs(float(result.charges.sum()) - 1.0) < 1e-10


def test_closed_form_equals_the_eigenvalue_and_localises_far_from_the_crossing(pair):
    _, _, molecule = pair
    V = 0.002
    result = _solve(pair, _positions(molecule, 0.08), V)
    H = torch.diag(result.diabatic_energies.detach()) + V * (1 - torch.eye(2, dtype=torch.float64))
    assert abs(float(result.energy) - float(torch.linalg.eigvalsh(H)[0])) < 1e-12
    gap = float((result.diabatic_energies[0] - result.diabatic_energies[1]).abs())
    assert gap > 10 * V                                 # far from the crossing
    assert float(result.weights.max()) > 1.0 - 2 * (V / gap) ** 2
    zero = _solve(pair, _positions(molecule, 0.08), 0.0)
    assert abs(float(zero.energy) - float(zero.diabatic_energies.min())) < 1e-12


def test_forces_through_the_crossing_match_finite_differences(pair):
    _, _, molecule = pair
    pos = _positions(molecule, 0.002).clone().requires_grad_(True)
    result = _solve(pair, pos, 0.05)
    force = -torch.autograd.grad(result.energy, pos)[0]
    h = 1e-5
    for atom, axis in ((0, 0), (4, 1)):
        up = _positions(molecule, 0.002).clone()
        down = up.clone()
        up[atom, axis] += h
        down[atom, axis] -= h
        fd = -(float(_solve(pair, up, 0.05).energy) - float(_solve(pair, down, 0.05).energy)) / (2 * h)
        assert abs(float(force[atom, axis]) - fd) < 1e-7


def test_the_model_hook_reads_charge_states_from_the_batch(pair):
    ElectrostaticModule = require("energy", "ElectrostaticModule")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    heads, features, molecule = pair
    pos = _positions(molecule, 0.0)
    distance = torch.cdist(pos, pos)
    src, dst = torch.nonzero((distance < 3.0) & (distance > 0), as_tuple=True)
    edge_index = torch.stack([src, dst])
    vec = pos[edge_index[0]] - pos[edge_index[1]]
    ids = torch.tensor([0, 0, 0, 1, 1, 1])
    data = {
        "charge_states": [
            {"fragment_id": ids, "fragment_charges": [1.0, 0.0]},
            {"fragment_id": ids, "fragment_charges": [0.0, 1.0]},
        ],
        "charge_state_coupling": 0.05,
    }
    module = ElectrostaticModule(heads, ElectrostaticConfig()).eval()
    energy, _ = module(features, pos, edge_index, vec, vec.norm(dim=-1), data=data)
    direct = _solve(pair, pos, 0.05)
    assert abs(float(energy - direct.energy)) < 1e-12
