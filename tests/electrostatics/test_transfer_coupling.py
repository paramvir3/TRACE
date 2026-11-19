"""SPEC 22: through-bond electronic coupling of diabatic charge states.

Exact statements, not fits:
* a uniform bridge of n sites gives V(n) = h sinh(theta) / sinh((n+1) theta),
  cosh(theta) = Delta / (2h) -- McConnell's exponential decay, in closed form;
* the bridge operator is positive definite for ANY parameters, with smallest
  eigenvalue at least the smallest gap excess (Gershgorin);
* V(D, A) = V(A, D), and V does not depend on the order of the atoms;
* the matrix-free route (CG, implicit derivatives) equals the dense route, in
  value and in gradient;
* the learned hopping is symmetric in the bond's endpoints and vanishes with
  zero slope at the cutoff;
* only one-electron moves couple; forces through the learned coupling and the
  EVB ground state match finite differences; the model hook and the YAML /
  checkpoint path carry the head.
"""

from __future__ import annotations

import functools
import math
import warnings

import pytest
import torch

from conftest import require


def _module():
    import trace_electrostatics.transfer_coupling as tc

    return tc


def _chain_graph(n_bridge, h, g):
    tc = _module()
    n = n_bridge + 2
    lo = torch.arange(n - 1)
    return tc.TransferGraph(
        bond_index=torch.stack((lo, lo + 1)),
        hopping=torch.full((n - 1,), h, dtype=torch.float64),
        gap_excess=torch.full((n,), g, dtype=torch.float64),
        amplitude_logit=torch.zeros(n, dtype=torch.float64),
    )


@pytest.mark.parametrize("h, g", [(0.3, 0.05), (1.0, 1.0), (0.05, 2.0)])
def test_a_uniform_bridge_decays_exactly_as_mcconnell(h, g):
    tc = _module()
    delta = 2.0 * h + g                      # every bridge site has two bonds
    theta = math.acosh(delta / (2.0 * h))
    for n_bridge in range(0, 13):
        graph = _chain_graph(n_bridge, h, g)
        n = n_bridge + 2
        donor = torch.zeros(n, dtype=torch.bool)
        acceptor = torch.zeros(n, dtype=torch.bool)
        donor[0] = True
        acceptor[-1] = True
        V = float(tc.bridge_coupling(graph, donor, acceptor))
        exact = h * math.sinh(theta) / math.sinh((n_bridge + 1) * theta)
        assert abs(V - exact) <= 1e-13 * exact, (n_bridge, V, exact)


def _random_graph(n_atoms, seed, scale=1.0):
    tc = _module()
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(n_atoms, 3, generator=g, dtype=torch.float64) * 6.0
    d = torch.cdist(pos, pos)
    lo, hi = torch.nonzero(torch.triu(d < 2.5, diagonal=1), as_tuple=True)
    hopping = scale * torch.rand(lo.numel(), generator=g, dtype=torch.float64)
    gap = 0.1 + torch.rand(n_atoms, generator=g, dtype=torch.float64)
    logit = torch.randn(n_atoms, generator=g, dtype=torch.float64)
    return tc.TransferGraph(torch.stack((lo, hi)), hopping, gap, logit)


def test_the_bridge_operator_is_positive_definite_for_any_parameters():
    tc = _module()
    for seed, scale in ((0, 1.0), (1, 1e3), (2, 1e-3)):
        graph = _random_graph(40, seed, scale)
        n = graph.n_atoms
        M = torch.diag(graph.gaps())
        lo, hi = graph.bond_index
        M = M.index_put((lo, hi), -graph.hopping, accumulate=True)
        M = M.index_put((hi, lo), -graph.hopping, accumulate=True)
        smallest = float(torch.linalg.eigvalsh(M)[0])
        assert smallest >= float(graph.gap_excess.min()) * (1.0 - 1e-12)


def test_the_coupling_is_symmetric_and_independent_of_atom_order():
    tc = _module()
    graph = _random_graph(30, 3)
    donor = torch.zeros(30, dtype=torch.bool)
    acceptor = torch.zeros(30, dtype=torch.bool)
    donor[:3] = True
    acceptor[-4:] = True
    V = tc.bridge_coupling(graph, donor, acceptor)
    assert float(V) > 0.0
    assert abs(float(V - tc.bridge_coupling(graph, acceptor, donor))) < 1e-15 * float(V)
    perm = torch.randperm(30, generator=torch.Generator().manual_seed(9))
    inverse = torch.argsort(perm)                 # new index of old atom k is inverse[k]
    lo, hi = inverse[graph.bond_index[0]], inverse[graph.bond_index[1]]
    permuted = tc.TransferGraph(
        torch.stack((torch.minimum(lo, hi), torch.maximum(lo, hi))),
        graph.hopping,
        graph.gap_excess[perm],
        graph.amplitude_logit[perm],
    )
    V_perm = tc.bridge_coupling(permuted, donor[perm], acceptor[perm])
    assert abs(float(V - V_perm)) < 1e-13 * float(V)


def test_the_matrix_free_route_equals_the_dense_route_with_gradients():
    tc = _module()
    base = _random_graph(36, 4)
    leaves = [base.hopping.clone().requires_grad_(True), base.gap_excess.clone().requires_grad_(True),
              base.amplitude_logit.clone().requires_grad_(True)]
    donor = torch.zeros(36, dtype=torch.bool)
    acceptor = torch.zeros(36, dtype=torch.bool)
    donor[:2] = True
    acceptor[-3:] = True
    values, grads = [], []
    for dense_max in (10_000, 0):
        graph = tc.TransferGraph(base.bond_index, *leaves)
        V = tc.bridge_coupling(graph, donor, acceptor, dense_max=dense_max)
        values.append(float(V))
        grads.append(torch.autograd.grad(V, leaves))
    assert abs(values[0] - values[1]) < 1e-11 * abs(values[0])
    for a, b in zip(*grads):
        assert float((a - b).abs().max()) < 1e-10 * float(a.abs().max())


def test_second_derivatives_through_the_matrix_free_route_match_the_dense_route():
    tc = _module()
    base = _random_graph(24, 5)
    donor = torch.zeros(24, dtype=torch.bool)
    acceptor = torch.zeros(24, dtype=torch.bool)
    donor[0] = True
    acceptor[-1] = True
    hessians = []
    for dense_max in (10_000, 0):
        h = base.hopping.clone().requires_grad_(True)
        graph = tc.TransferGraph(base.bond_index, h, base.gap_excess, base.amplitude_logit)
        V = tc.bridge_coupling(graph, donor, acceptor, dense_max=dense_max)
        (g,) = torch.autograd.grad(V, h, create_graph=True)
        (hv,) = torch.autograd.grad((g * torch.linspace(0, 1, g.numel(), dtype=torch.float64)).sum(), h)
        hessians.append(hv)
    assert float((hessians[0] - hessians[1]).abs().max()) < 1e-9 * float(hessians[0].abs().max())


@pytest.fixture(scope="module")
def heads_and_features():
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heads = ElectrostaticHeads(r_max=3.0)
    g = torch.Generator().manual_seed(2)
    features = torch.randn(9, heads.irreps_in.dim, generator=g, dtype=torch.float64)
    features[:, 64:] = 0.0
    return heads, features


def _trained_like_head(heads, seed=1):
    """A head whose networks are randomised (the zero-initialised last layers
    too); the two physical scalars h_0 and beta stay at their initial values."""
    tc = _module()
    torch.manual_seed(seed)
    head = tc.TransferCouplingHead.for_heads(heads)
    with torch.no_grad():
        for module in (head.embed, head.bond_net, head.site_net):
            for p in module.parameters():
                p.add_(0.3 * torch.randn_like(p))
    return head


def test_hoppings_stay_within_two_decades_of_the_distance_law(heads_and_features):
    tc = _module()
    heads, features = heads_and_features
    head = _trained_like_head(heads)
    with torch.no_grad():
        for p in head.bond_net.parameters():
            p.mul_(50.0)                              # a wildly over-trained network
    pos = _three_molecules()
    edge_index, _, edge_len = _edges(pos, 3.0)
    graph = head(heads.node_invariants(features), edge_index, edge_len)
    lo, hi = graph.bond_index
    r = (pos[lo] - pos[hi]).norm(dim=-1)
    envelope = require("energy", "bond_envelope")(r, head.r_max, onset=head.envelope_onset)
    law = envelope * torch.exp(head.log_hopping_scale - head.decay * (r - head.reference_length))
    ratio = graph.hopping[envelope > 0] / law[envelope > 0]
    # Inclusive bound: a saturated tanh is exactly +-1 in float64.
    assert float(ratio.max()) <= 100.0 * (1 + 1e-12) and float(ratio.min()) >= 0.01 * (1 - 1e-12)
    assert float(ratio.max() / ratio.min()) > 10.0      # the bound is reached, not idle


def _edges(pos, cutoff):
    d = torch.cdist(pos.detach(), pos.detach())
    src, dst = torch.nonzero((d < cutoff) & (d > 0), as_tuple=True)
    edge_index = torch.stack([src, dst])
    vec = pos[edge_index[0]] - pos[edge_index[1]]
    return edge_index, vec, vec.norm(dim=-1)


def test_the_learned_hopping_is_symmetric_and_vanishes_smoothly_at_the_cutoff(heads_and_features):
    heads, features = heads_and_features
    head = _trained_like_head(heads)
    inv = heads.node_invariants(features[:2])

    def hop(r):
        pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64)
        edge_index, _, edge_len = _edges(pos, 10.0)
        return head(inv, edge_index, edge_len).hopping

    forward = hop(1.7)
    swapped = head(inv.flip(0), *_edges(torch.tensor([[0.0, 0.0, 0.0], [1.7, 0.0, 0.0]],
                                                     dtype=torch.float64), 10.0)[::2]).hopping
    assert forward.numel() == 1 and float(forward) > 0.0
    assert abs(float(forward - swapped)) < 1e-15 * float(forward)
    # C2 contact with the cutoff: h ~ (r_c - r)^3, so halving the distance to
    # the cutoff divides h by 8 and dh/dr by 4 (the smooth prefactor changes by
    # O(d) over the step).
    r_max = head.r_max

    def value_and_slope(d):
        r = torch.tensor(r_max - d, dtype=torch.float64, requires_grad=True)
        pos = torch.stack([torch.zeros(3, dtype=torch.float64),
                           torch.stack([r, torch.zeros((), dtype=torch.float64),
                                        torch.zeros((), dtype=torch.float64)])])
        edge_index, _, edge_len = _edges(pos, 10.0)
        value = head(inv, edge_index, edge_len).hopping.sum()
        (slope,) = torch.autograd.grad(value, r)
        return float(value), float(slope)

    v1, s1 = value_and_slope(2e-3)
    v2, s2 = value_and_slope(1e-3)
    assert v2 > 0.0 and abs(v1 / v2 - 8.0) < 0.05
    assert abs(s1 / s2 - 4.0) < 0.05
    assert hop(r_max + 1e-6).numel() == 0


def test_only_one_electron_moves_couple_and_the_sign_is_nonpositive():
    tc = _module()
    graph = _random_graph(12, 6)
    ids = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    states = [
        {"fragment_id": ids, "fragment_charges": [1.0, 0.0, 0.0]},
        {"fragment_id": ids, "fragment_charges": [0.0, 1.0, 0.0]},
        {"fragment_id": ids, "fragment_charges": [0.0, 0.0, 1.0]},
        {"fragment_id": ids, "fragment_charges": [-1.0, 2.0, 0.0]},    # two electrons from state 0
    ]
    C = tc.coupling_matrix(graph, states)
    assert torch.equal(C, C.T) and bool((C <= 0).all())
    assert float(C[0, 1]) < 0 and float(C[1, 2]) < 0 and float(C[0, 2]) < 0
    assert float(C[0, 3]) == 0.0                      # differs by 2 on two fragments
    assert float(C[1, 3]) < 0.0                       # (0,1,0) -> (-1,2,0): one electron 0 -> 1
    V01 = tc.bridge_coupling(graph, ids == 0, ids == 1)
    assert abs(float(C[0, 1] + V01)) < 1e-15
    with pytest.raises(ValueError, match="identical"):
        tc.coupling_matrix(graph, [states[0], states[0]])
    with pytest.raises(ValueError, match="same fragment"):
        tc.coupling_matrix(graph, [states[0], {"fragment_id": torch.zeros(12, dtype=torch.long),
                                                "fragment_charges": [1.0]}])


def _three_molecules(shift=0.0):
    a = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [-0.3, 0.95, 0.0]], dtype=torch.float64)
    return torch.cat([a, a + torch.tensor([2.4 + shift, 0.0, 0.0]), a + torch.tensor([4.8, 0.2, 0.0])])


def _evb_energy(heads, features, head, pos):
    ElectrostaticModule = require("energy", "ElectrostaticModule")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    edge_index, vec, edge_len = _edges(pos, 3.0)
    ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    data = {
        "charge_states": [
            {"fragment_id": ids, "fragment_charges": [1.0, 0.0, 0.0]},
            {"fragment_id": ids, "fragment_charges": [0.0, 0.0, 1.0]},
        ],
        "charge_state_coupling": "through_bond",
    }
    module = ElectrostaticModule(heads, ElectrostaticConfig(), head).eval()
    energy, _ = module(features, pos, edge_index, vec, edge_len, data=data)
    return energy


def test_forces_through_the_learned_coupling_match_finite_differences(heads_and_features):
    heads, features = heads_and_features
    head = _trained_like_head(heads)
    pos = _three_molecules().requires_grad_(True)
    energy = _evb_energy(heads, features, head, pos)
    force = -torch.autograd.grad(energy, pos)[0]
    # Teeth: the coupling's own geometry dependence must be large enough that a
    # 1e-7 force comparison would see it go missing.
    tc = _module()
    edge_index, _, edge_len = _edges(pos, 3.0)
    graph = head(heads.node_invariants(features), edge_index, edge_len)
    ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    V = tc.bridge_coupling(graph, ids == 0, ids == 2)
    (dV,) = torch.autograd.grad(V, pos)
    assert float(V) > 1e-3 and float(dV.abs().max()) > 1e-3
    h = 1e-5
    for atom, axis in ((0, 0), (4, 1), (7, 2)):
        up, down = _three_molecules(), _three_molecules()
        up[atom, axis] += h
        down[atom, axis] -= h
        fd = -(float(_evb_energy(heads, features, head, up))
               - float(_evb_energy(heads, features, head, down))) / (2 * h)
        assert abs(float(force[atom, axis]) - fd) < 1e-7, (atom, axis, float(force[atom, axis]), fd)


def test_the_hook_uses_the_learned_coupling_and_it_lowers_the_energy(heads_and_features):
    """The bridged donor and acceptor (molecules 0 and 2, 4.8 A apart) couple only
    through molecule 1; the adiabatic energy equals the closed form with that V."""
    tc = _module()
    heads, features = heads_and_features
    head = _trained_like_head(heads)
    pos = _three_molecules()
    energy = _evb_energy(heads, features, head, pos)
    edge_index, _, edge_len = _edges(pos, 3.0)
    graph = head(heads.node_invariants(features), edge_index, edge_len)
    ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    V = float(tc.bridge_coupling(graph, ids == 0, ids == 2))
    direct = graph.bond_index
    assert not bool(((ids[direct[0]] == 0) & (ids[direct[1]] == 2)).any())   # no direct D-A bond
    assert V > 0.0
    zero = require("energy", "ElectrostaticModule")(heads, require("config", "ElectrostaticConfig")()).eval()
    data = {"charge_states": [
        {"fragment_id": ids, "fragment_charges": [1.0, 0.0, 0.0]},
        {"fragment_id": ids, "fragment_charges": [0.0, 0.0, 1.0]}], "charge_state_coupling": 0.0}
    e0, _ = zero(features, pos, edge_index, pos[edge_index[0]] - pos[edge_index[1]], edge_len, data=data)
    # E_a, E_b from the uncoupled solve: E = min(E_a, E_b); coupled: mean - sqrt(gap^2/4 + V^2)
    adiabatic = require("charge_states", "adiabatic_charge_states")
    electrostatic_state = require("energy", "electrostatic_state")
    state_fn = functools.partial(electrostatic_state, features, pos, edge_index,
                                 pos[edge_index[0]] - pos[edge_index[1]], edge_len, heads,
                                 require("config", "ElectrostaticConfig")())
    diabatic = adiabatic(state_fn, data["charge_states"], 0.0).diabatic_energies
    mean, half = float(diabatic.mean()), 0.5 * float(diabatic[0] - diabatic[1])
    assert abs(float(energy) - (mean - math.sqrt(half * half + V * V))) < 1e-10
    assert float(energy) < float(e0)
    with pytest.raises(ValueError, match="through_bond"):
        zero(features, pos, edge_index, pos[edge_index[0]] - pos[edge_index[1]], edge_len,
             data={**data, "charge_state_coupling": "through_bond"})
