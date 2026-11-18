"""The bound sector's own state (``RedoxElectrostatics.bound_state``), the source of the redox model's
electronic response.

With ``anchor='formal'`` the free charges are integers.  An applied field E shifts each configuration
energy by the free dipole and the bound response,

    E_n(E) = E_n(0) - d_n . E - 1/2 E . alpha_b . E      (W and b do not depend on E),

so the canonical free energy has

    -d2F/dE2 = alpha_b + Var_p(d) / kT,

with alpha_b the SAME for every configuration.  The thermal term is weighted by the populations above
the ground configuration, ~exp(-gap / kT).  The model's electronic polarizability (and eps_inf in a
crystal) is therefore the bound sector's, up to that term, and ``bound_state`` is that sector as a
trace_electrostatics state: energy ``energy_bound``, solved on the dense (exact) route.
"""
import math

import pytest
import torch
from redox_helpers import edges, lattice_graph, make_heads, neighbour_list, periodic_edges
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

Z_FRAGMENT = torch.tensor([8, 12, 12, 8, 11, 17])


def _formal(kT=0.1):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, kT=kT)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


def _fragment(heads, seed=3):
    """A distorted rocksalt fragment O Mg Mg O Na Cl, nearest distances 2.1-2.6 A."""
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor([[0.0, 0.0, 0.0], [2.1, 0.0, 0.0], [0.0, 2.1, 0.0], [2.1, 2.1, 0.0],
                         [0.0, 0.0, 2.5], [2.4, 0.1, 2.6]], dtype=torch.float64)
    pos = base + 0.1 * torch.randn(6, 3, generator=g, dtype=torch.float64)
    h = torch.randn(6, heads.irreps_in.dim, generator=g, dtype=torch.float64)
    return pos, h, neighbour_list(pos)


def _rocksalt(heads, a=4.212, seed=5):
    """Conventional rocksalt MgO cell (8 atoms), slightly displaced."""
    frac = torch.tensor([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
                         [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, .5, .5]], dtype=torch.float64)
    cell = a * torch.eye(3, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    pos = frac @ cell + 0.05 * torch.randn(8, 3, generator=g, dtype=torch.float64)
    z = torch.tensor([12] * 4 + [8] * 4)
    h = torch.randn(1, heads.irreps_in.dim, generator=g, dtype=torch.float64).repeat(8, 1)
    h[4:] = torch.randn(1, heads.irreps_in.dim, generator=g, dtype=torch.float64)
    ei, es = lattice_graph(pos, cell, (True, True, True))
    return pos, h, ei, es, cell, z


def test_the_bound_state_energy_is_energy_bound_open_and_periodic():
    module, heads = _formal()
    pos, h, ei = _fragment(heads)
    vec, length = edges(pos, ei)
    data = {"z": Z_FRAGMENT, "total_charge": 0.0}
    _, result = module(h, pos, ei, vec, length, data=data)
    state = module.bound_state(h, pos, ei, vec, length, data=data)
    scale = 1.0 + abs(float(result.energy_bound))
    assert abs(float(state.energy) - float(result.energy_bound)) <= 1e-10 * scale

    pos, h, ei, es, cell, z = _rocksalt(heads)
    vec, length = periodic_edges(pos, cell, ei, es)
    keywords = dict(pbc=(True, True, True), cell=cell, edge_shift=es, data={"z": z, "total_charge": 0.0})
    _, result = module(h, pos, ei, vec, length, **keywords)
    state = module.bound_state(h, pos, ei, vec, length, **keywords)
    scale = 1.0 + abs(float(result.energy_bound))
    assert abs(float(state.energy) - float(result.energy_bound)) <= 1e-10 * scale


def test_the_formal_models_polarizability_is_the_bound_sectors():
    """Central differences of the FULL formal free energy in an applied field (open boundaries) against
    alpha_b of bound_state.  kT = 0.02 eV: the thermal term Var_p(d) / kT carries exp(-gap / kT)."""
    from trace_electrostatics.observables import polarizability
    module, heads = _formal(kT=0.02)
    pos, h, ei = _fragment(heads)
    vec, length = edges(pos, ei)

    def free_energy(field):
        data = {"z": Z_FRAGMENT, "total_charge": 0.0, "external_field": field}
        return float(module(h, pos, ei, vec, length, data=data)[0])

    step = 0.02                                                                   # V / A
    alpha_fd = torch.zeros(3, 3, dtype=torch.float64)
    F0 = free_energy(torch.zeros(3))
    for a in range(3):
        for b in range(3):
            ea, eb = torch.eye(3)[a] * step, torch.eye(3)[b] * step
            if a == b:
                second = (free_energy(ea) - 2.0 * F0 + free_energy(-ea)) / step ** 2
            else:
                second = (free_energy(ea + eb) - free_energy(ea - eb) - free_energy(-ea + eb)
                          + free_energy(-ea - eb)) / (4.0 * step ** 2)
            alpha_fd[a, b] = -second
    state = module.bound_state(h, pos, ei, vec, length, data={"z": Z_FRAGMENT, "total_charge": 0.0})
    for route in ("fixed_k", "minimiser"):
        alpha = polarizability(state.solve_result, state.assembly, route=route).detach()
        # F is exactly quadratic in E for one dominant configuration: the central difference is exact up to
        # round-off, eps * |F| / step^2 ~ 1e-16 * 1e2 / 4e-4 ~ 3e-11
        assert float((alpha_fd - alpha).abs().max()) <= 1e-7 * float(alpha.abs().max()), (route, alpha_fd, alpha)
    assert float(torch.linalg.eigvalsh(alpha).min()) > 0.0


def test_the_bound_dielectric_tensor_is_differentiable_in_the_parameters():
    from trace_electrostatics.observables import dielectric_tensor
    module, heads = _formal()
    pos, h, ei, es, cell, z = _rocksalt(heads)
    vec, length = periodic_edges(pos, cell, ei, es)
    state = module.bound_state(h, pos, ei, vec, length, pbc=(True, True, True), cell=cell, edge_shift=es,
                               data={"z": z, "total_charge": 0.0})
    eps = dielectric_tensor(state.solve_result, state.assembly, create_graph=True)
    assert all(math.isfinite(float(x)) for x in eps.flatten())
    assert float(torch.linalg.eigvalsh(0.5 * (eps + eps.T)).min()) > 1.0
    params = [p for p in heads.parameters() if p.requires_grad]
    grads = torch.autograd.grad(eps.trace(), params, allow_unused=True)
    assert any(g is not None and float(g.abs().max()) > 0.0 for g in grads)


def test_the_bound_state_refuses_what_the_module_refuses():
    module, heads = _formal()
    pos, h, ei = _fragment(heads)
    vec, length = edges(pos, ei)
    with pytest.raises(NotImplementedError, match="fragment_id"):
        module.bound_state(h, pos, ei, vec, length,
                           data={"z": Z_FRAGMENT, "fragment_id": torch.zeros(6, dtype=torch.long)})


@pytest.mark.parametrize("redox,routed", [
    (dict(fragment_ensemble="diabatic", anchor="formal", max_charge=3), True),
    (dict(), False),
    (dict(fragment_ensemble="diabatic"), False),
], ids=["formal", "mean-field", "diabatic-atomic"])
def test_flashace_routes_observables_to_the_bound_sector_only_for_formal_anchors(redox, routed):
    """The bound sector is the model's response only with integer free charges; any other redox model
    keeps RedoxElectrostatics.state's refusal, so no observable silently belongs to the wrong response."""
    from flashace.model import _bound_sector_is_the_response
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    module = RedoxElectrostatics(make_heads(), ElectrostaticConfig(), RedoxConfig(**redox))
    assert _bound_sector_is_the_response(module) is routed
    if not routed:
        with pytest.raises(NotImplementedError, match="bound_state"):
            module.state()
