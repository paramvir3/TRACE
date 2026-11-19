"""State-dependent exclusion in the configuration energies (``RedoxConfig(exclusion=True)``, trace_redox SPEC 18).

    Pi_u(n) = K sum_{v in closed-shell frames} N_v [ N_u(n) S(rho_u^(n), rho_v; R_uv) - N_u(f_u) S(rho_u^(f_u), rho_v; R_uv) ] s(R_uv)
    e_hat_u(n) -> e_hat_u(n) + Pi_u(n)

- rho_u^(n): the frontier density of state n from the exact ladder (kappa = sqrt(2 I), I = E_Z(n+1) - E_Z(n));
  N_u(n): its electrons beyond the closed core; only cation-frame and neutral-frame units carry the term (an extra
  electron or a hole on Al, Mg, Au, ...).
- rho_v: the closed shell of a frame neighbour: an anion's valence shell (in-crystal removal energy I_eff, one bounded
  learned scalar per gas-unbound anion element, initial 9 eV) or a cation's core (the next ionisation energy).
- K: one universal positive scalar, initial 10 Ha (the density-overlap model's value).
Exact properties tested here: zero at every frame (so formal MgO is unchanged), zero for an isolated atom, an extra
electron pays and a hole relieves, closed shells only, forces -grad E, smooth at the cutoff.
"""
import math

import pytest
import torch
from redox_helpers import edges, lattice_graph, make_heads, neighbour_list, periodic_edges
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

HARTREE = 27.211386245988


def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


def _shifts(module, z, pos):
    """Pi [U, S] for the structure, with every directed neighbour pair inside the model's cutoff."""
    ei = neighbour_list(pos, r_max=5.0)
    _, length = edges(pos, ei)
    from trace_redox.constants import formal_frame
    states = torch.arange(-2, 4, dtype=torch.float64)
    return module.exclusion(torch.tensor(z), states, ei, length, formal_frame(torch.tensor(z))), states


def _octahedron(center_z, distance=2.10):
    pos = [[0.0, 0.0, 0.0]] + [[s * distance * (a == k) for a in range(3)] for k in range(3) for s in (1.0, -1.0)]
    return [center_z] + [8] * 6, torch.tensor(pos, dtype=torch.float64)


def test_zero_at_every_frame_and_for_an_isolated_atom():
    module, _ = _module(exclusion=True)
    z, pos = _octahedron(13)
    Pi, states = _shifts(module, z, pos)
    from trace_redox.constants import formal_frame
    frames = formal_frame(torch.tensor(z))
    for u, f in enumerate(frames.tolist()):
        assert float(Pi[u, int(f - states[0])]) == 0.0
    Pi_alone, _ = _shifts(module, [13], torch.zeros(1, 3, dtype=torch.float64))
    bound = torch.isfinite(Pi_alone)                       # states with a bound frontier level
    assert bool(bound[0, 1:].all()) and float(Pi_alone[bound].abs().max()) == 0.0


def test_an_extra_electron_in_an_oxide_cage_pays_with_the_predicted_magnitude():
    """Al2+ in an O2- octahedron at 2.10 A, default K = 10 Ha and I_eff(O2-) = 9 eV, against an independent adaptive
    integral of the SAME (regularised) densities: 6 K N_O N_Al2+ S = 8.0888 eV.  (The theory note's pre-check used
    the unregularised r^(2 beta) e^(-2 kappa r) and gave 8.945 eV: a 9.6 % definitional difference, stated there.)
    Mg+ likewise pays; Al3+ (frame) pays nothing."""
    from test_redox_exclusion_overlap import BOHR, _reference
    module, _ = _module(exclusion=True)
    z, pos = _octahedron(13)
    Pi, states = _shifts(module, z, pos)
    k = {int(s): i for i, s in enumerate(states.tolist())}
    k_al = math.sqrt(2 * 28.447642 / HARTREE)
    expected = 6 * 10 * HARTREE * 8 * _reference(k_al, 3 / k_al - 1, math.sqrt(2 * 9.0 / HARTREE), 0.0, 2.10 / BOHR)
    assert abs(float(Pi[0, k[2]]) - expected) <= 2e-6 * expected            # quadrature: 6.4e-7 relative
    assert expected > 1.0
    assert float(Pi[0, k[1]]) > float(Pi[0, k[2]]) > 0.0           # two 3s electrons cost more than one
    z, pos = _octahedron(12)
    Pi, _ = _shifts(module, z, pos)
    assert float(Pi[0, k[1]]) > 0.0


def test_au_minus_atop_oxygen_pays_more_than_far_from_it_and_a_hole_relieves():
    module, _ = _module(exclusion=True)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.2], [0.0, 0.0, 4.7]], dtype=torch.float64)   # O, near Au, far Au
    Pi, states = _shifts(module, [8, 79, 79], pos)
    k = {int(s): i for i, s in enumerate(states.tolist())}
    near, far = float(Pi[1, k[-1]]), float(Pi[2, k[-1]])
    assert near > far > 0.0
    assert near - far >= 0.4                                         # the site penalty the pre-check predicted
    assert float(Pi[1, k[1]]) < 0.0                                  # Au+ has no 6s electron to exclude


def test_only_closed_shell_frames_exclude():
    """Au- next to neutral-frame Au0 (open shell, covalent partner) pays nothing: that interaction belongs to the
    couplings and the ACE."""
    module, _ = _module(exclusion=True)
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=torch.float64)
    Pi, states = _shifts(module, [79, 79], pos)
    finite = torch.isfinite(Pi)
    assert float(Pi[finite].abs().max()) == 0.0
    assert torch.equal(~finite, (states == -2).unsqueeze(0).expand_as(Pi))      # only Au2- (no bound level) removed


def test_exclusion_leaves_formal_mgo_unchanged():
    on, heads = _module(exclusion=True)
    off, _ = _module(exclusion=False)
    off.load_state_dict(on.state_dict(), strict=False)
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
    e_on, r_on = on(h, pos, ei, vec, length, **kw)
    e_off, _ = off(h, pos, ei, vec, length, **kw)
    assert int(r_on.iterations) == 1
    assert abs(float(e_on - e_off)) <= 1e-9 * (1.0 + abs(float(e_off)))


def test_exclusion_forces_are_minus_the_gradient():
    module, heads = _module(exclusion=True)
    g = torch.Generator().manual_seed(5)
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0]],
                        dtype=torch.float64)
    z = torch.tensor([8, 79, 79, 12, 13])
    h = torch.randn(5, heads.irreps_in.dim, generator=g, dtype=torch.float64)
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


def test_the_term_switches_off_smoothly_at_the_cutoff():
    module, _ = _module(exclusion=True)
    r_cut = float(module.exclusion.r_cut)
    vals = []
    for d in (r_cut - 1e-3, r_cut, r_cut + 0.2):
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, d]], dtype=torch.float64)
        ei = torch.tensor([[0, 1], [1, 0]])
        _, length = edges(pos, ei)
        from trace_redox.constants import formal_frame
        z = torch.tensor([8, 79])
        Pi = module.exclusion(z, torch.arange(-2, 4, dtype=torch.float64), ei, length, formal_frame(z))
        finite = torch.isfinite(Pi[1])                                      # bound states (Au2- has no bound level)
        vals.append(float(Pi[1][finite].abs().max()))
    assert vals[1] == 0.0 and vals[2] == 0.0 and vals[0] <= 1e-6


def test_it_needs_formal_anchors():
    from trace_redox import RedoxConfig
    with pytest.raises(ValueError, match="formal"):
        RedoxConfig(exclusion=True)


def test_an_unbound_extra_electron_is_not_a_localised_state():
    """Mg- (Mg's electron affinity is negative: no bound frontier level in vacuum): an extra electron with no bound
    level is maximally diffuse, not a localised configuration, so its state is removed (Pi = +inf), in a cage and in
    vacuum alike (a neighbour-dependent rule would switch admissibility at the cutoff).
    (The first version gave it no density and therefore ZERO exclusion: one Mg- swallowed all three donor electrons
    of Au2/MgO:Al on step-6 weights, measured 2026-10-15.)  A bound extra electron (Mg+) keeps a finite cost, and a
    unit without close closed shells keeps its states."""
    module, _ = _module(exclusion=True)
    z, pos = _octahedron(12)
    Pi, states = _shifts(module, z, pos)
    k = {int(s): i for i, s in enumerate(states.tolist())}
    assert math.isinf(float(Pi[0, k[-1]])) and float(Pi[0, k[-1]]) > 0
    assert math.isfinite(float(Pi[0, k[1]])) and math.isfinite(float(Pi[0, k[0]]))
    Pi_alone, _ = _shifts(module, [12], torch.zeros(1, 3, dtype=torch.float64))
    assert math.isinf(float(Pi_alone[0, k[-1]]))                             # inadmissible in vacuum as well
    assert float(Pi_alone[0, k[0]]) == 0.0 and float(Pi_alone[0, k[1]]) == 0.0 and float(Pi_alone[0, k[2]]) == 0.0


def test_the_unbound_rule_keeps_energies_and_forces_finite():
    module, heads = _module(exclusion=True)
    z, pos0 = _octahedron(12)
    z = torch.tensor(z)
    h = torch.randn(7, heads.irreps_in.dim, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    ei = neighbour_list(pos0)
    pos = pos0.clone().requires_grad_(True)
    vec, length = edges(pos, ei)
    energy, result = module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
    (grad,) = torch.autograd.grad(energy, pos)
    assert math.isfinite(float(energy)) and bool(torch.isfinite(grad).all())
    assert float(result.free_charge[0]) >= 0.0                                  # no Mg- sink
