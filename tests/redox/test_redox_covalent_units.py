"""Free carriers vs bound fluctuations: covalent units (``RedoxConfig(carriers='covalent_units')``, trace_redox
SPEC 19; theory note 2026-10-16 §3).

Covalent units are the connected components of the chemical-bond graph restricted to bonds between two
open-shell (neutral-frame) atoms; a closed-shell frame atom is its own unit.  A configuration is admissible iff the
deviations d_u = n_u - f_u are sign-definite inside every covalent unit (all >= 0 or all <= 0): a unit holds free
carriers of one sign, which may sit on any of its atoms; electron-hole pairs inside a unit are bound-sector
fluctuations.  Between units every configuration is allowed (real charge transfer: dissociation, donor -> acceptor).

Measured motive: the formal stack's configuration search refuses C10H3+ (1451 configurations within 2.5 eV; the
lowest a charge-ordered C+/C- chain), because configuration energies carry no covalent-bond energy loss.
"""
import itertools

import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)


def _random_problem(U, seed, K=1):
    g = torch.Generator().manual_seed(seed)
    states = torch.arange(-1, K + 1, dtype=torch.float64)
    chi = 3.0 * torch.rand(U, generator=g, dtype=torch.float64)
    gap = 2.0 + 4.0 * torch.rand(U, generator=g, dtype=torch.float64)
    e_hat = chi.unsqueeze(1) * states + 0.5 * gap.unsqueeze(1) * states * states
    b = 0.5 * torch.randn(U, generator=g, dtype=torch.float64)
    A = torch.randn(U, U, generator=g, dtype=torch.float64)
    W = 0.6 * (A @ A.T) / U + torch.diag(torch.full((U,), 0.5, dtype=torch.float64)) - 0.3
    return states, e_hat, b, 0.5 * (W + W.T)


def _admissible(n, units, frames):
    d = [int(x) - int(f) for x, f in zip(n, frames)]
    for unit in set(units):
        members = [d[i] for i in range(len(d)) if units[i] == unit]
        if min(members) < 0 < max(members):
            return False
    return True


def _connected(configs):
    index = {tuple(n): i for i, n in enumerate(configs)}
    seen, stack = {0}, [0]
    while stack:
        n = configs[stack.pop()]
        for a, b in itertools.permutations(range(len(n)), 2):
            m = list(n)
            m[a] += 1
            m[b] -= 1
            j = index.get(tuple(m))
            if j is not None and j not in seen:
                seen.add(j)
                stack.append(j)
    return len(seen) == len(configs)


def test_the_admissibility_rule():
    from trace_redox.diabatic import admissible
    units = torch.tensor([0, 0, 0, 1, 2, 2])
    frames = torch.tensor([0, 0, 0, 2, 0, 0])
    good = torch.tensor([[1, 1, 0, 2, 0, 0], [0, -1, -1, 2, 0, 0], [0, 0, 0, 1, 1, 0], [-1, 0, 0, 3, -1, -1]])
    bad = torch.tensor([[1, -1, 0, 2, 0, 0], [0, 0, 0, 2, 1, -1]])
    assert bool(admissible(good, units, frames).all())
    assert not bool(admissible(bad, units, frames).any())


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("Q", [-1, 0, 1])
@pytest.mark.parametrize("window", [2.5, 6.0])
def test_the_search_finds_every_admissible_configuration(seed, Q, window):
    """Brute force over every charge vector of a 6-atom problem with a 3-atom and a 2-atom covalent unit and one
    singleton: the search returns exactly the admissible configurations within the window of the lowest
    admissible one (where that set is connected under one-electron moves, the search's stated condition)."""
    from trace_redox.diabatic import configuration_energies, search_configurations
    U, K = 6, 1
    units, frames = [0, 0, 0, 1, 2, 2], [0, 0, 0, 0, 0, 0]
    states, e_hat, b, W = _random_problem(U, 300 + seed)
    every = [n for n in itertools.product(range(-1, K + 1), repeat=U) if sum(n) == Q and _admissible(n, units, frames)]
    E_all = configuration_energies(torch.tensor(every), e_hat, states, b, W)
    lowest = float(E_all.min())
    inside = [every[k] for k in range(len(every)) if float(E_all[k]) <= lowest + window]
    if not _connected(inside):
        pytest.skip("window set not connected under single moves (outside the search's stated domain)")
    found, E_found = search_configurations(e_hat, states, b, W, torch.full((U,), K), Q, window=window, cap=4096,
                                           units=torch.tensor(units), frames=torch.tensor(frames))
    assert sorted(map(tuple, found.tolist())) == sorted(inside)
    assert abs(float(E_found.min()) - lowest) <= 1e-12 * (1 + abs(lowest))


def test_covalent_units_from_the_bond_graph():
    from trace_redox.constants import formal_frame
    from trace_redox.fragments import chemical_bonds, covalent_units
    cases = {
        "chain C-C-C with H": ([6, 6, 6, 1], [[0, 0, 0], [1.3, 0, 0], [2.6, 0, 0], [3.66, 0, 0]], [0, 0, 0, 0]),
        "C2 and a far H": ([6, 6, 1], [[0, 0, 0], [1.3, 0, 0], [3.4, 0, 0]], [0, 0, 1]),
        "Au2 atop O": ([8, 79, 79], [[0, 0, 0], [0, 0, 2.2], [0, 0, 4.7]], [0, 1, 1]),
        "Na-Cl contact": ([11, 17], [[0, 0, 0], [2.4, 0, 0]], [0, 1]),
    }
    for name, (z, pos, expected) in cases.items():
        z, pos = torch.tensor(z), torch.tensor(pos, dtype=torch.float64)
        ei = neighbour_list(pos, r_max=4.0)
        _, length = edges(pos, ei)
        bi, bj, bs = chemical_bonds(z, ei, length, coverage=4.0)
        units = covalent_units(formal_frame(z), bi, bj, int(z.numel()))
        # same partition as expected (labels up to renaming)
        pairs = {(a, b) for a in range(len(expected)) for b in range(len(expected))}
        assert all((units[a] == units[b]) == (expected[a] == expected[b]) for a, b in pairs), (name, units.tolist())


def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True, **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


def _captured_configs(module, heads, z, pos, Q):
    import trace_redox.module as module_code
    h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    captured = {}
    original = module_code.search_configurations

    def capture(*args, **kw):
        out = original(*args, **kw)
        captured.update(configs=out[0], E=out[1])
        return out

    module_code.search_configurations = capture
    try:
        energy, result = module(h, pos, ei, vec, length, data={"z": torch.tensor(z), "total_charge": float(Q)})
    finally:
        module_code.search_configurations = original
    return energy, result, captured


@pytest.mark.parametrize("charge, atoms", [(0.0, 12), (1.0, 13)])
def test_the_carbon_chain_has_only_its_carriers(charge, atoms):
    """C10H2: the frame alone (one configuration).  C10H3+: one hole, on any atom of the single covalent unit
    (at most 13 configurations); the atom-resolved formal stack refuses this structure (> 512 within 2.5 eV)."""
    import os
    from ase.io import read
    path = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "benchmarks_4g", "data_pfn",
                        "Carbon_chain_test.xyz")
    if not os.path.exists(path):
        pytest.skip("experiments/benchmarks_4g/data_pfn/Carbon_chain_test.xyz not present (derived benchmark data, "
                    "not in git); make it with experiments/benchmarks_4g/prepare_all.sh (raw 4G-HDNNP data under "
                    "$TRACE_4G_RAW)")
    data = [a for a in read(path, ":60")
            if float(a.info["charge"]) == charge][0]
    assert len(data) == atoms
    module, heads = _module(carriers="covalent_units")
    z, pos = data.numbers.tolist(), torch.tensor(data.positions, dtype=torch.float64)
    energy, result, c = _captured_configs(module, heads, z, pos, charge)
    configs = c["configs"]
    if charge == 0.0:
        assert configs.shape[0] == 1 and int(configs.abs().sum()) == 0
    else:
        assert 1 <= configs.shape[0] <= 13
        assert bool((configs.sum(dim=1) == 1).all()) and bool((configs >= 0).all())
    assert abs(float(result.free_charge.sum()) - charge) <= 1e-9


def test_charge_transfer_between_units_is_kept():
    """Na and Cl at 15 A: closed-shell frames Na+, Cl-, two units.  The neutral atoms (Na0 Cl0: IE(Na) - EA(Cl) =
    1.53 eV > K_E / 15 A = 0.96 eV) are admissible and lowest; the ionic frame is in the window too."""
    module, heads = _module(carriers="covalent_units")
    z, pos = [11, 17], torch.tensor([[0.0, 0.0, 0.0], [15.0, 0.0, 0.0]], dtype=torch.float64)
    _, _, c = _captured_configs(module, heads, z, pos, 0)
    configs, E = c["configs"], c["E"]
    rows = {tuple(n): float(e) for n, e in zip(configs.tolist(), E.tolist())}
    assert (0, 0) in rows and (1, -1) in rows
    assert rows[(0, 0)] < rows[(1, -1)]


def test_covalent_unit_forces_are_minus_the_gradient():
    module, heads = _module(carriers="covalent_units", exclusion=True)
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


def test_it_needs_formal_diabatic_and_a_known_value():
    from trace_redox import RedoxConfig
    with pytest.raises(ValueError, match="carriers"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", carriers="bonds")
    with pytest.raises(ValueError, match="formal"):
        RedoxConfig(carriers="covalent_units")


def test_small_units_keep_every_configuration():
    """Units of at most ``full_unit_atoms`` atoms (default 3) are solved exactly in configuration space (every
    intra-unit configuration: for a 2-site bond, Au0Au0, Au+Au-, Au-Au+, the full CI of the minimal basis); larger
    units are partitioned into carriers and bound fluctuations."""
    from trace_redox.fragments import covalent_units
    frames = torch.zeros(7, dtype=torch.long)
    i, j = torch.tensor([0, 2, 3, 4]), torch.tensor([1, 3, 4, 5])          # a dimer {0,1}, a 4-chain {2..5}, atom 6
    units = covalent_units(frames, i, j, 7, full_unit_atoms=3)
    assert len({int(units[0]), int(units[1])}) == 2                       # the dimer: unrestricted (split)
    assert len({int(units[k]) for k in (2, 3, 4, 5)}) == 1                # the 4-chain: one restricted unit
    units = covalent_units(frames, i, j, 7, full_unit_atoms=4)
    assert len({int(units[k]) for k in (2, 3, 4, 5)}) == 4                # now solved exactly too


def test_au2_on_an_oxide_is_unchanged_by_the_partition():
    """Au2 is a 2-atom unit and every other atom a closed-shell frame: carriers='covalent_units' gives the same
    configurations, so the same energy, as carriers='atoms' (step 9 is unchanged)."""
    on, heads = _module(carriers="covalent_units", exclusion=True)
    off, _ = _module(carriers="atoms", exclusion=True)
    off.load_state_dict(on.state_dict())
    z = torch.tensor([8, 79, 79, 12, 13])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0]],
                       dtype=torch.float64)
    h = torch.randn(5, heads.irreps_in.dim, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    kw = dict(data={"z": z, "total_charge": 0.0})
    assert float(on(h, pos, ei, vec, length, **kw)[0]) == float(off(h, pos, ei, vec, length, **kw)[0])
