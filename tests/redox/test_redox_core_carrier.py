"""Core + carrier free charges (``RedoxConfig(free_charge_model='core_carrier')``, trace_redox SPEC 17.9).

Each unit's integer charge n is split into a frame charge f (the formal ion core: Mg +2, Al +3, O -2,
...; 0 for the other elements) on a compact Gaussian (the ionic width) and a carrier charge c = n - f
on a diffuse Gaussian of width

    sigma_car = sqrt(2) K_E / (sqrt(pi) U_min),   U_min = min_n [E_Z(n+1) + E_Z(n-1) - 2 E_Z(n)]

over the unit's interior states (the exact ladder's smallest curvature).  Then S0(sigma_car) = U_min,
and every one-electron move d = e_v - e_u has a non-negative second difference

    E(n + d) + E(n - d) - 2 E(n) = [U_u(n_u) - S0_car,u] + [U_v(n_v) - S0_car,v] + d^T W_cc d  >=  0

(W_cc is a Gram Schur complement): no ionic collapse at any distance, under any screening.  With one
width for all of a unit's charge, S0(sigma_ion) exceeds U_min for every element (Au: 12.03 > 6.92 eV),
and the compressed Au2 of the test set collapses to Au+Au- (step 6, structure 87).

Exact identities tested here:
- sigma_car = sigma_core reduces to the single-width model bit for bit (any frames);
- the all-frame configuration has the single-width energy exactly (bulk formal MgO is unchanged);
- an isolated ion has exactly E_Z(n) (+ its learned shift);
- forces are -grad E.
"""
import itertools
import math

import pytest
import torch
from redox_helpers import edges, lattice_graph, make_heads, neighbour_list, periodic_edges
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)


def _module(model="core_carrier", seed=1, **redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(seed)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, free_charge_model=model, **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


def _pair(module, heads, z, pos):
    import trace_redox.module as module_code
    g = torch.Generator().manual_seed(7)
    h = torch.randn(len(z), heads.irreps_in.dim, generator=g, dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    captured = {}
    original = module_code.search_configurations

    def capture(e_hat, states, b, W, K, Q, **kw):
        out = original(e_hat, states, b, W, K, Q, **kw)
        captured.update(e_hat=e_hat.detach(), states=states, b=b.detach(), W=W.detach(), K=K, L=kw.get("min_charge"),
                        configs=out[0])
        return out

    module_code.search_configurations = capture
    try:
        energy, result = module(h, pos, ei, vec, length, data={"z": torch.tensor(z), "total_charge": 0.0})
    finally:
        module_code.search_configurations = original
    return energy, result, captured


def _second_differences(c):
    """E(n + d) + E(n - d) - 2 E(n) for d = e_1 - e_2 around EVERY charge-conserving configuration n of the
    dimer with n and n +- d in range (not only the searched ones: a strong collapse leaves one configuration
    in the window)."""
    from trace_redox.diabatic import configuration_energies
    e_hat, states, b, W = c["e_hat"], c["states"], c["b"], c["W"]
    K, L = c["K"], c["L"]
    lo, hi = int(L.max()), int(K.min())
    out = []
    for k in range(lo, hi + 1):
        n = torch.tensor([k, -k])
        d = torch.tensor([1, -1])
        plus, minus = n + d, n - d
        if all(bool((x >= L).all() and (x <= K).all()) for x in (n, plus, minus)):
            E = configuration_energies(torch.stack([plus, minus, n]), e_hat, states, b, W)
            out.append(float(E[0] + E[1] - 2 * E[2]))
    return out


@pytest.mark.parametrize("distance", [1.9, 2.09, 2.3])
def test_no_ionic_collapse_for_a_compressed_gold_dimer(distance):
    """Au2 at and below the compressed test geometries: every one-electron second difference is >= 0 with
    core + carrier charges; the single-width model has negative ones (the collapse it is built against)."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=torch.float64)
    cc, heads = _module("core_carrier")
    _, _, c_cc = _pair(cc, heads, [79, 79], pos)
    d2 = _second_differences(c_cc)
    assert d2 and min(d2) >= -1e-10, min(d2)
    single, heads = _module("single")
    _, _, c_s = _pair(single, heads, [79, 79], pos)
    assert min(_second_differences(c_s)) < -1.0          # S0(0.955 A) - U = 12.03 - 6.92 eV, minus the pair gain


def test_equal_widths_reduce_to_the_single_width_model_exactly(monkeypatch):
    import trace_redox.module as module_code
    cc, heads = _module("core_carrier")
    single, _ = _module("single")
    single.load_state_dict(cc.state_dict())
    monkeypatch.setattr(module_code, "carrier_widths", lambda species, sigma_core, max_charge: sigma_core)
    g = torch.Generator().manual_seed(3)
    base = torch.tensor([[0.0, 0.0, 0.0], [2.1, 0.0, 0.0], [0.0, 2.1, 0.0], [2.1, 2.1, 0.0],
                         [0.0, 0.0, 2.5], [2.4, 0.1, 2.6]], dtype=torch.float64)
    pos = base + 0.1 * torch.randn(6, 3, generator=g, dtype=torch.float64)
    z = [8, 12, 13, 8, 79, 79]
    e_cc, r_cc, _ = _pair(cc, heads, z, pos)
    e_s, r_s, _ = _pair(single, heads, z, pos)
    assert abs(float(e_cc - e_s)) <= 1e-10 * (1 + abs(float(e_s)))
    assert torch.allclose(r_cc.free_charge, r_s.free_charge, rtol=0, atol=1e-10)
    assert torch.allclose(r_cc.q_bound, r_s.q_bound, rtol=0, atol=1e-10)


def test_the_all_frame_configuration_of_bulk_mgo_has_the_single_width_energy():
    """Formal rocksalt MgO sits entirely at its frames (Mg +2, O -2): carriers are zero, so the energy is
    the single-width model's exactly (the compact ionic Madelung energy is untouched)."""
    cc, heads = _module("core_carrier")
    single, _ = _module("single")
    single.load_state_dict(cc.state_dict())
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
    e_cc, r_cc = cc(h, pos, ei, vec, length, **kw)
    e_s, r_s = single(h, pos, ei, vec, length, **kw)
    assert torch.equal(torch.round(r_cc.free_charge), torch.tensor([2.0] * 4 + [-2.0] * 4))
    # one configuration in both (a gapped ionic crystal), so the energies are the frame energies
    assert int(r_cc.iterations) == 1 and int(r_s.iterations) == 1
    assert abs(float(e_cc - e_s)) <= 1e-9 * (1 + abs(float(e_s)))


@pytest.mark.parametrize("z,Q", [(13, 3.0), (13, 2.0), (79, -1.0), (79, 1.0), (12, 1.0)])
def test_an_isolated_ion_keeps_exactly_its_tabulated_energy(z, Q):
    from trace_redox.diabatic import formal_ladder
    cc, heads = _module("core_carrier")
    pos = torch.zeros(1, 3, dtype=torch.float64)
    h = torch.randn(1, heads.irreps_in.dim, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    ei = torch.zeros(2, 0, dtype=torch.long)
    vec, length = edges(pos, ei)
    energy, result = cc(h, pos, ei, vec, length, data={"z": torch.tensor([z]), "total_charge": Q})
    states, E, _, _ = formal_ladder(torch.tensor([z]), max_charge=3)
    k = int(Q - states[0])
    shift = float(result.state_energies[0, k] - E[0, k])         # the learned bounded shift of that state
    assert abs(float(energy - result.energy_bound) - (float(E[0, k]) + shift)) <= 1e-9 * (1 + abs(float(E[0, k])))


def test_core_carrier_forces_are_minus_the_gradient():
    cc, heads = _module("core_carrier")
    g = torch.Generator().manual_seed(5)
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [2.2, 0.1, 0.0], [4.5, 0.0, 0.3], [2.1, 2.2, 0.0]], dtype=torch.float64)
    z = torch.tensor([8, 79, 79, 12])
    h = torch.randn(4, heads.irreps_in.dim, generator=g, dtype=torch.float64)
    ei = neighbour_list(pos0)

    def energy(p):
        vec, length = edges(p, ei)
        return cc(h, p, ei, vec, length, data={"z": z, "total_charge": 0.0})[0]

    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    for i, axis in [(1, 0), (2, 2), (3, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-7 * (1.0 + abs(fd))


def test_carrier_widths_come_from_the_exact_ladder():
    from trace_electrostatics.constants import K_E
    from trace_redox.module import carrier_widths
    z = torch.tensor([79, 12, 8, 13])
    sigma = carrier_widths(z, torch.ones(4, dtype=torch.float64), 3)
    U_min = {79: 6.917, 12: 7.389, 8: 10.211, 13: 5.553}          # interior curvatures of formal_ladder (eV)
    for k, Z in enumerate(z.tolist()):
        expected = math.sqrt(2.0) * K_E / (math.sqrt(math.pi) * U_min[Z])
        assert abs(float(sigma[k]) - expected) <= 2e-3 * expected


def test_frames_are_the_main_group_formal_ions_and_zero_elsewhere():
    from trace_redox.constants import formal_frame
    z = torch.tensor([11, 12, 13, 8, 17, 7, 79, 47, 6, 1, 26])
    assert formal_frame(z).tolist() == [1, 2, 3, -2, -1, -3, 0, 0, 0, 0, 0]


def test_it_needs_formal_anchors_and_a_known_model():
    from trace_redox import RedoxConfig
    with pytest.raises(ValueError, match="formal"):
        RedoxConfig(free_charge_model="core_carrier")
    with pytest.raises(ValueError, match="free_charge_model"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", free_charge_model="shell")


def test_the_eps_prior_trains_with_core_carrier_charges(tmp_path):
    """bound_state's energy is energy_bound (the frame constant lives in the free energy), so the response
    priors run on a formal crystal whose frames are nonzero."""
    import warnings
    import train
    from test_redox_dielectric_train import FORMAL, _prior_config
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history = train.main(["--config", str(_prior_config(tmp_path / "run",
                                                             redox=dict(FORMAL, free_charge_model="core_carrier")))])
    val = history["val_dielectric_prior_rmse"]
    assert all(math.isfinite(v) for v in val) and val[-1] < val[0]
