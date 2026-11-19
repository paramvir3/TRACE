"""Exclusion on bound fluctuations: Parr-Pearson finite differences of the exclusion energy as the bound sector's
electronegativity and hardness (``RedoxConfig(exclusion_bound=True)``, trace_redox SPEC 19.2; theory note
2026-10-16 §4).

    chi^P_u = 1/2 [Pi_u(f_u + 1) - Pi_u(f_u - 1)],     eta^P_u = Pi_u(f_u + 1) + Pi_u(f_u - 1)     (Pi_u(f_u) = 0)
    chi_u -> chi_u + chi^P_u,   eta_u -> eta_u + eta^P_u        (bound sector; ESTABLISHED closure: Parr & Pearson 1983)

With carriers='covalent_units' an electron-hole pair inside a covalent unit is a bound fluctuation, so the exclusion
that steered step 9's Au+Au- resonance must reach the bound charges; this is that coupling.
"""
import math

import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)


def _pi(z, pos):
    from trace_redox.constants import formal_frame
    from trace_redox.exclusion import Exclusion
    z = torch.tensor(z)
    ei = neighbour_list(pos, r_max=5.0)
    _, length = edges(pos, ei)
    states = torch.arange(-2, 4, dtype=torch.float64)
    return Exclusion(r_cut=5.0)(z, states, ei, length, formal_frame(z)), states, formal_frame(z)


def test_the_anchors_are_the_finite_differences():
    from trace_redox.exclusion import exclusion_bound_anchors
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.2], [0.0, 0.0, 4.7], [2.1, 0.0, 0.1]], dtype=torch.float64)
    Pi, states, frames = _pi([8, 79, 79, 12], pos)
    chi, eta = exclusion_bound_anchors(Pi, states, frames)
    k = {int(s): i for i, s in enumerate(states.tolist())}
    for u, f in enumerate(frames.tolist()):
        if f < 0:
            assert float(chi[u]) == 0.0 and float(eta[u]) == 0.0                    # anions carry no Pi
            continue
        plus, minus = float(Pi[u, k[f + 1]]), float(Pi[u, k[f - 1]])
        assert abs(float(chi[u]) - 0.5 * (plus - minus)) <= 1e-14 * (1 + abs(plus) + abs(minus))
        assert abs(float(eta[u]) - (plus + minus)) <= 1e-14 * (1 + abs(plus) + abs(minus))
    # the near Au is the donor (DFT: near +0.11, far -0.13 Hirshfeld)
    assert float(chi[1]) < float(chi[2]) - 0.5
    assert float(eta[1]) > 0.0


def test_the_anchors_vanish_without_closed_shell_neighbours():
    from trace_redox.exclusion import exclusion_bound_anchors
    Pi, states, frames = _pi([79], torch.zeros(1, 3, dtype=torch.float64))
    chi, eta = exclusion_bound_anchors(Pi, states, frames)
    assert float(chi.abs().max()) == 0.0 and float(eta.abs().max()) == 0.0
    Pi, states, frames = _pi([79, 79, 6], torch.tensor([[0.0, 0, 0], [2.5, 0, 0], [0, 2.0, 0]], dtype=torch.float64))
    chi, eta = exclusion_bound_anchors(Pi, states, frames)
    assert float(chi.abs().max()) == 0.0 and float(eta.abs().max()) == 0.0


def test_a_negative_hardness_is_refused():
    from trace_redox.exclusion import exclusion_bound_anchors
    states = torch.arange(-1, 2, dtype=torch.float64)
    Pi = torch.tensor([[-0.5, 0.0, -0.2]], dtype=torch.float64)                    # Pi(-1) + Pi(+1) < 0
    with pytest.raises(ValueError, match="convex"):
        exclusion_bound_anchors(Pi, states, torch.tensor([0]))


def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True, exclusion=True,
                         **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


def _system():
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0]],
                       dtype=torch.float64)
    return torch.tensor([8, 79, 79, 12, 13]), pos


def test_bound_state_reproduces_energy_bound():
    module, heads = _module(carriers="covalent_units", exclusion_bound=True)
    z, pos = _system()
    h = torch.randn(5, heads.irreps_in.dim, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    _, result = module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
    state = module.bound_state(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
    e_state = float(state.energy)
    assert abs(e_state - float(result.energy_bound)) <= 1e-9 * (1.0 + abs(e_state))
    off, _ = _module(carriers="covalent_units", exclusion_bound=False)
    off.load_state_dict(module.state_dict())
    _, result_off = off(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
    assert abs(float(result.energy_bound) - float(result_off.energy_bound)) > 1e-3     # the anchors act


def test_an_isolated_atom_is_unchanged():
    """The response priors (Au, Ag atoms) see no closed shell: identical energy and bound sector."""
    on, heads = _module(exclusion_bound=True)
    off, _ = _module(exclusion_bound=False)
    off.load_state_dict(on.state_dict())
    pos = torch.zeros(1, 3, dtype=torch.float64)
    h = torch.randn(1, heads.irreps_in.dim, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    ei = torch.zeros(2, 0, dtype=torch.long)
    vec, length = torch.zeros(0, 3, dtype=torch.float64), torch.zeros(0, dtype=torch.float64)
    kw = dict(data={"z": torch.tensor([79]), "total_charge": 0.0})
    assert float(on(h, pos, ei, vec, length, **kw)[0]) == float(off(h, pos, ei, vec, length, **kw)[0])


def test_exclusion_bound_forces_are_minus_the_gradient():
    module, heads = _module(carriers="covalent_units", exclusion_bound=True)
    z, pos0 = _system()
    h = torch.randn(5, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos0)

    def energy(p):
        vec, length = edges(p, ei)
        return module(h, p, ei, vec, length, data={"z": z, "total_charge": 0.0})[0]

    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    for i, axis in [(1, 2), (2, 0), (4, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-7 * (1.0 + abs(fd))


def test_it_needs_exclusion():
    from trace_redox import RedoxConfig
    with pytest.raises(ValueError, match="exclusion_bound"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", exclusion_bound=True)
