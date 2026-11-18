"""Electronegativity enters once: the formal mode without the bound sector's own drive
(``RedoxConfig(bound_electronegativity='none')``, trace_redox SPEC 17.8).

With anchor='formal' the exact ionisation ladders E_Z(n) carry each element's electronegativity and
hardness.  The bound sector's electronegativity heads (chi, chi_Theta) add a second, independent drive:
the charges y0 they induce with no free charge, their energy E_b0 and their potential b at every free
site.  Measured on Au2/MgO:Al (step 5), b decides on which Au the transferred electron sits, by 0.39 eV,
against DFT; nothing constrains it.

'none' zeroes chi and chi_Theta in the bound assembly.  The bound sector is then linear response only.
Exact consequences, tested here:
- with no applied field, y0 = 0, so E_b0 = 0 and b = 0 identically;
- K, A, J and W do not contain chi, so the response is bit for bit unchanged (W, polarizability);
- for every integer configuration n, E_learned(n) - E_none(n) = E_b0 + b . n;
- bound_state carries the same functional (its energy is energy_bound = 0).
"""
import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

Z = torch.tensor([8, 12, 12, 8, 11, 17])


def _module(bound_electronegativity="learned", **redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3,
                         bound_electronegativity=bound_electronegativity, **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


def _fragment(heads, seed=3):
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor([[0.0, 0.0, 0.0], [2.1, 0.0, 0.0], [0.0, 2.1, 0.0], [2.1, 2.1, 0.0],
                         [0.0, 0.0, 2.5], [2.4, 0.1, 2.6]], dtype=torch.float64)
    pos = base + 0.1 * torch.randn(6, 3, generator=g, dtype=torch.float64)
    h = torch.randn(6, heads.irreps_in.dim, generator=g, dtype=torch.float64)
    return pos, h, neighbour_list(pos)


def _capture(module, h, pos, ei, data):
    """The search's inputs (e_hat, states, b, W) and the configurations, through the module's own call."""
    import trace_redox.module as module_code
    captured = {}
    original = module_code.search_configurations

    def capture(e_hat, states, b, W, K, Q, **kw):
        out = original(e_hat, states, b, W, K, Q, **kw)
        captured.update(e_hat=e_hat, states=states, b=b, W=W, configs=out[0])
        return out

    module_code.search_configurations = capture
    try:
        vec, length = edges(pos, ei)
        energy, result = module(h, pos, ei, vec, length, data=data)
    finally:
        module_code.search_configurations = original
    return energy, result, captured


def test_without_the_bound_drive_E_b0_and_b_vanish_and_the_response_is_unchanged():
    learned, heads = _module("learned")
    none, _ = _module("none")
    none.load_state_dict(learned.state_dict())
    pos, h, ei = _fragment(heads)
    data = {"z": Z, "total_charge": 0.0}
    _, r_l, c_l = _capture(learned, h, pos, ei, data)
    _, r_n, c_n = _capture(none, h, pos, ei, data)
    assert float(r_l.energy_bound.abs()) > 1e-3 and float(c_l["b"].abs().max()) > 1e-3    # the drive was there
    assert float(r_n.energy_bound) == 0.0 and float(c_n["b"].abs().max()) == 0.0
    assert torch.equal(c_n["W"], c_l["W"]) and torch.equal(c_n["e_hat"], c_l["e_hat"])


def test_every_configuration_differs_by_exactly_E_b0_plus_b_dot_n():
    from trace_redox.diabatic import configuration_energies
    learned, heads = _module("learned")
    none, _ = _module("none")
    none.load_state_dict(learned.state_dict())
    pos, h, ei = _fragment(heads)
    data = {"z": Z, "total_charge": 0.0}
    _, r_l, c_l = _capture(learned, h, pos, ei, data)
    _, r_n, c_n = _capture(none, h, pos, ei, data)
    configs = c_l["configs"]
    E_l = configuration_energies(configs, c_l["e_hat"], c_l["states"], c_l["b"], c_l["W"]) + r_l.energy_bound
    E_n = configuration_energies(configs, c_n["e_hat"], c_n["states"], c_n["b"], c_n["W"]) + r_n.energy_bound
    expected = r_l.energy_bound + configs.to(E_l.dtype) @ c_l["b"]
    scale = 1.0 + float(E_l.abs().max())
    assert float(((E_l - E_n) - expected).abs().max()) <= 1e-12 * scale


def test_the_bound_state_carries_the_same_functional_and_the_same_polarizability():
    from trace_electrostatics.observables import polarizability
    learned, heads = _module("learned")
    none, _ = _module("none")
    none.load_state_dict(learned.state_dict())
    pos, h, ei = _fragment(heads)
    vec, length = edges(pos, ei)
    data = {"z": Z, "total_charge": 0.0}
    _, r_n = none(h, pos, ei, vec, length, data=data)
    s_n = none.bound_state(h, pos, ei, vec, length, data=data)
    s_l = learned.bound_state(h, pos, ei, vec, length, data=data)
    assert abs(float(s_n.energy) - float(r_n.energy_bound)) <= 1e-12
    for route in ("fixed_k", "minimiser"):
        a_n = polarizability(s_n.solve_result, s_n.assembly, route=route)
        a_l = polarizability(s_l.solve_result, s_l.assembly, route=route)
        assert float((a_n - a_l).abs().max()) <= 1e-12 * float(a_l.abs().max())


def test_an_applied_field_still_polarises_the_bound_sector():
    none, heads = _module("none")
    pos, h, ei = _fragment(heads)
    vec, length = edges(pos, ei)
    field = torch.tensor([0.0, 0.0, 0.05], dtype=torch.float64)
    _, r = none(h, pos, ei, vec, length, data={"z": Z, "total_charge": 0.0, "external_field": field})
    assert float(r.energy_bound) < 0.0                       # -1/2 E.alpha.E, strictly negative


@pytest.mark.parametrize("redox", [{}, {"fragment_ensemble": "diabatic"}, {"fragments": "none"}],
                         ids=["mean-field", "diabatic-atomic", "no-fragments"])
def test_it_needs_formal_anchors(redox):
    from trace_redox import RedoxConfig
    with pytest.raises(ValueError, match="formal"):
        RedoxConfig(bound_electronegativity="none", **redox)


def test_unknown_values_are_refused():
    from trace_redox import RedoxConfig
    with pytest.raises(ValueError, match="bound_electronegativity"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", bound_electronegativity="partial")
