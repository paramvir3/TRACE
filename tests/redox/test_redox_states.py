"""Charge states beyond +1 (RedoxConfig.max_charge; trace_redox/SPEC.md §5a).

States -1, 0, +1, ..., +K with NIST anchors E_+k = IE_1 + ... + IE_k: an
isolated ion has exactly that energy; a state an atom cannot reach (k > Z) is
absent (+inf) and never populated; the levels are strictly increasing (Ehat
convex on the integers) for every parameter value; K = 1 is the earlier model.
"""

import math

import pytest
import torch
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)
from redox_helpers import cluster, edges, make_heads

from trace_electrostatics.config import ElectrostaticConfig
from trace_redox import RedoxConfig, RedoxElectrostatics
from trace_redox.constants import ATOMIC_SUCCESSIVE_IONIZATION_EV as IE
from trace_redox.free_solver import solve_free_charge


def _module(seed=1, **redox):
    heads = make_heads()
    torch.manual_seed(seed)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), RedoxConfig(**redox)), heads


def _single(module, heads, z, Q, seed=3):
    h = torch.randn(1, heads.irreps_in.dim, generator=torch.Generator().manual_seed(seed))
    pos = torch.zeros(1, 3)
    ei = torch.zeros(2, 0, dtype=torch.long)
    return module(h, pos, ei, torch.zeros(0, 3), torch.zeros(0), data={"z": torch.tensor([z]), "total_charge": Q})


@pytest.mark.parametrize("z,K,Q", [(12, 2, 2.0), (13, 3, 3.0), (26, 3, 3.0), (2, 4, 2.0), (1, 3, 1.0)])
def test_a_saturated_isolated_cation_has_the_cumulative_nist_energy(z, K, Q):
    module, heads = _module(max_charge=K)
    energy, res = _single(module, heads, z, Q)
    expected = sum(IE[z][j] for j in range(int(Q)))
    assert abs(float(energy) - expected) <= 1e-11 * expected
    assert float(res.populations[0, int(Q) + 1]) == 1.0


@pytest.mark.parametrize("z,K,Q", [(26, 3, 2.0), (12, 2, 1.0), (29, 3, 2.0)])
def test_an_interior_cation_is_within_the_entropy_bound_of_its_nist_energy(z, K, Q):
    """c = Q is forced on one unit; the ensemble mixes neighbouring states only by entropy:
    E_n - kT ln(K + 2) <= F <= E_n (the Lemma 1 bound), and the gaps make it exponentially tight."""
    kT = 0.1
    module, heads = _module(max_charge=K, kT=kT)
    energy, res = _single(module, heads, z, Q)
    expected = sum(IE[z][j] for j in range(int(Q)))
    assert expected - kT * math.log(K + 2) - 1e-9 <= float(energy) <= expected + 1e-9
    assert float(res.populations[0, int(Q) + 1]) > 0.999


def test_capacity_is_the_electron_count():
    module, heads = _module(max_charge=3)
    with pytest.raises(ValueError, match="capacity"):
        _single(module, heads, 1, 2.0)                       # H holds at most +1
    _, res = _single(module, heads, 2, 2.0)                  # He2+ exists
    assert float(res.unit_charge[0]) == 2.0
    with pytest.raises(ValueError, match="capacity"):
        _single(module, heads, 2, 3.0)
    # absent states are +inf and never populated
    _, res = _single(module, heads, 1, 0.0)
    assert torch.isinf(res.state_energies[0, 3:]).all()
    assert float(res.populations[0, 3:].abs().max()) == 0.0


def test_levels_are_strictly_increasing_for_random_shifts():
    """Ehat_n = E_n - 1/2 J_uu n^2 must be strictly convex on its states (positive gaps),
    whatever the learned shifts.  J_uu = S0(sigma_f) without orbitals; sic='none' so that
    the reported E_n carry no removal."""
    from trace_electrostatics.kernels import gram_self_energy

    module, heads = _module(max_charge=4, sic="none")
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        last = module.charge_states.shift_mlp[-1]
        last.weight.copy_(torch.randn(last.weight.shape, generator=g) * 5.0)
        last.bias.copy_(torch.randn(last.bias.shape, generator=g) * 5.0)
    pos, h, ei = cluster(6, heads.irreps_in.dim, 6)
    vec, length = edges(pos, ei)
    _, res = module(h, pos, ei, vec, length, data={"z": torch.tensor([26, 8, 1, 1, 8, 12]), "total_charge": 2.0})
    J = gram_self_energy(res.sigma_free)
    e_hat = res.state_energies - 0.5 * J.unsqueeze(1) * (res.states ** 2).unsqueeze(0)
    checked = 0
    for u in range(6):
        e = e_hat[u][torch.isfinite(e_hat[u])]
        second = e[2:] - 2 * e[1:-1] + e[:-2]
        assert bool((second > 0).all()), (u, e)
        checked += int(second.numel())
    assert checked == 4 + 4 + 1 + 1 + 4 + 4   # (finite states - 2) per unit: Fe, O, H, H, O, Mg; H stops at +1


def test_k2_equals_k1_where_the_second_cation_state_is_irrelevant():
    """Na, Cl, O, H: IE_2 is tens of eV above IE_1, so the +2 populations are ~exp(-30 eV / kT)."""
    pos, h, ei = cluster(6, make_heads().irreps_in.dim, 7)
    vec, length = edges(pos, ei)
    z = torch.tensor([11, 17, 8, 1, 1, 8])
    for Q in (0.0, 1.0, -1.0):
        energies = []
        for K in (1, 2):
            module, _ = _module(max_charge=K)
            energies.append(float(module(h, pos, ei, vec, length, data={"z": z, "total_charge": Q})[0]))
        assert abs(energies[0] - energies[1]) <= 1e-9


@pytest.mark.parametrize("K,Q", [(2, 2.0), (3, 2.0)])
def test_forces_and_a_force_loss_gradient_with_dications(K, Q):
    module, heads = _module(max_charge=K)
    g = torch.Generator().manual_seed(9)
    with torch.no_grad():
        last = module.charge_states.shift_mlp[-1]
        last.weight.copy_(torch.randn(last.weight.shape, generator=g) * 0.3)
    pos0, h, ei = cluster(5, heads.irreps_in.dim, 9, scale=1.4)
    z = torch.tensor([12, 8, 1, 1, 8])
    pos = pos0.clone().requires_grad_(True)
    vec, length = edges(pos, ei)
    energy, res = module(h, pos, ei, vec, length, data={"z": z, "total_charge": Q})
    (grad,) = torch.autograd.grad(energy, pos)
    assert abs(float(res.q.sum()) - Q) <= 1e-10
    step = 1e-5
    for i, a in [(0, 0), (2, 1), (4, 2)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, a] += step
        minus[i, a] -= step
        e_p = float(module(h, plus, ei, *edges(plus, ei), data={"z": z, "total_charge": Q})[0])
        e_m = float(module(h, minus, ei, *edges(minus, ei), data={"z": z, "total_charge": Q})[0])
        fd = (e_p - e_m) / (2 * step)
        assert abs(fd - float(grad[i, a])) <= 1e-7 * (1.0 + abs(fd))
    weight = module.charge_states.shift_mlp[-1].weight

    def loss():
        p = pos0.clone().requires_grad_(True)
        e, _ = module(h, p, ei, *edges(p, ei), data={"z": z, "total_charge": Q})
        (gr,) = torch.autograd.grad(e, p, create_graph=True)
        return (gr ** 2).sum() + 0.1 * e

    (gw,) = torch.autograd.grad(loss(), weight)
    for k in (0, 11):
        with torch.no_grad():
            weight[0, k] += 1e-6
        up = float(loss())
        with torch.no_grad():
            weight[0, k] -= 2e-6
        down = float(loss())
        with torch.no_grad():
            weight[0, k] += 1e-6
        fd = (up - down) / 2e-6
        assert abs(fd - float(gw[0, k])) <= 1e-6 * (1.0 + abs(fd))


def test_the_solver_refuses_a_charge_beyond_the_finite_states():
    e_hat = torch.tensor([[1.0, 0.0, 2.0, float("inf")], [1.5, 0.0, 3.0, 9.0]])
    states = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    W = torch.eye(2)
    with pytest.raises(ValueError):
        solve_free_charge(torch.zeros(2), W, e_hat, states, 3.0, 0.1)    # capacity 1 + 2 = 3: not interior
    z, mu, *_ = solve_free_charge(torch.zeros(2), W, e_hat, states, 2.5, 0.1)
    assert math.isfinite(float(mu))
