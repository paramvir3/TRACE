"""Free-charge widths by formal frame (``RedoxConfig(free_width_rule='frame')``, trace_redox SPEC 17.10).

Ionic-frame elements (constants.formal_frame != 0: Mg, Al, O, Na, Cl, ...) keep the compact ionic (Shannon)
width with its learned factor: their Madelung energy needs it, and in a closed-shell cage their charge stays
confined (step 7: diffuse carriers inside an O2- cage gain a spurious penetration energy and kill the transfer).
Neutral-frame elements (metals and covalent elements: Au, Ag, C, H, ...) carry their free charge on the
ladder width sigma_car = sqrt(2) K_E / (sqrt(pi) U_min), fixed (no learned factor), so S0 = U_min and every
one-electron move between two such units has Delta^2 E = d^T W d >= 0: no ionic collapse.
Measured motive (step-6 weights): the Au carrier alone made diffuse gives the full transfer, -1.0 e on Au2
in 6/6 doped structures (single width: -0.69 .. -1.0).
"""
import itertools

import pytest
import torch
from redox_helpers import edges, lattice_graph, make_heads, neighbour_list, periodic_edges
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)


def _module(rule="frame", seed=1, **redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(seed)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, free_width_rule=rule, **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


def _run(module, heads, z, pos):
    import trace_redox.module as module_code
    g = torch.Generator().manual_seed(7)
    h = torch.randn(len(z), heads.irreps_in.dim, generator=g, dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    captured = {}
    original = module_code.search_configurations

    def capture(e_hat, states, b, W, K, Q, **kw):
        out = original(e_hat, states, b, W, K, Q, **kw)
        captured.update(e_hat=e_hat.detach(), states=states, b=b.detach(), W=W.detach(), K=K, L=kw.get("min_charge"))
        return out

    module_code.search_configurations = capture
    try:
        energy, result = module(h, pos, ei, vec, length, data={"z": torch.tensor(z), "total_charge": 0.0})
    finally:
        module_code.search_configurations = original
    return energy, result, captured


def test_widths_follow_the_formal_frame():
    from trace_redox.module import carrier_widths
    frame, heads = _module("frame")
    ionic, _ = _module("ionic")
    ionic.load_state_dict(frame.state_dict())
    with torch.no_grad():
        frame.width.log_factor.normal_(0.0, 0.4)            # learned factors away from 1
        ionic.width.log_factor.copy_(frame.width.log_factor)
    z = torch.tensor([12, 8, 13, 79, 47, 6, 1])
    s_frame, s_ionic = frame.free_widths(z), ionic.free_widths(z)
    neutral = torch.tensor([False, False, False, True, True, True, True])
    assert torch.equal(s_frame[~neutral], s_ionic[~neutral])                        # ionic frames untouched
    expected = carrier_widths(z, s_ionic, 3)
    assert torch.allclose(s_frame[neutral], expected[neutral], rtol=0, atol=1e-14)  # fixed ladder widths
    assert abs(float(s_frame[3]) - 1.661) < 2e-3                                     # Au


@pytest.mark.parametrize("distance", [1.9, 2.09, 2.5])
def test_no_collapse_between_neutral_frame_units(distance):
    from trace_redox.diabatic import configuration_energies
    module, heads = _module("frame")
    pos = torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=torch.float64)
    _, _, c = _run(module, heads, [79, 79], pos)
    d2 = []
    for k in range(-1, 3):
        n, d = torch.tensor([k, -k]), torch.tensor([1, -1])
        plus, minus = n + d, n - d
        if all(bool((x >= c["L"]).all() and (x <= c["K"]).all()) for x in (n, plus, minus)):
            E = configuration_energies(torch.stack([plus, minus, n]), c["e_hat"], c["states"], c["b"], c["W"])
            d2.append(float(E[0] + E[1] - 2 * E[2]))
    assert d2 and min(d2) >= -1e-10, d2


def test_an_ionic_crystal_is_unchanged():
    """No neutral-frame atom in formal MgO: the 'frame' and 'ionic' rules give the same energy bit for bit."""
    frame, heads = _module("frame")
    ionic, _ = _module("ionic")
    ionic.load_state_dict(frame.state_dict())
    a = 4.212
    frac = torch.tensor([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
                         [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, .5, .5]], dtype=torch.float64)
    cell = a * torch.eye(3, dtype=torch.float64)
    pos = frac @ cell + 0.03 * torch.randn(8, 3, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    z = torch.tensor([12] * 4 + [8] * 4)
    h = torch.randn(8, heads.irreps_in.dim, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    ei, es = lattice_graph(pos, cell, (True, True, True))
    vec, length = periodic_edges(pos, cell, ei, es)
    kw = dict(pbc=(True, True, True), cell=cell, edge_shift=es, data={"z": z, "total_charge": 0.0})
    assert torch.equal(frame(h, pos, ei, vec, length, **kw)[0], ionic(h, pos, ei, vec, length, **kw)[0])


def test_frame_width_forces_are_minus_the_gradient():
    module, heads = _module("frame")
    g = torch.Generator().manual_seed(5)
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [2.2, 0.1, 0.0], [4.6, 0.0, 0.3], [2.1, 2.2, 0.0]], dtype=torch.float64)
    z = torch.tensor([8, 79, 79, 12])
    h = torch.randn(4, heads.irreps_in.dim, generator=g, dtype=torch.float64)
    ei = neighbour_list(pos0)

    def energy(p):
        vec, length = edges(p, ei)
        return module(h, p, ei, vec, length, data={"z": z, "total_charge": 0.0})[0]

    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    for i, axis in [(1, 0), (2, 2), (3, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-7 * (1.0 + abs(fd))


def test_it_needs_formal_anchors_and_a_known_rule():
    from trace_redox import RedoxConfig
    with pytest.raises(ValueError, match="formal"):
        RedoxConfig(free_width_rule="frame")
    with pytest.raises(ValueError, match="free_width_rule"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", free_width_rule="atomic")
