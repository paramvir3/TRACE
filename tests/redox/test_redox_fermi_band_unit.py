"""SPEC 38, stage 2: the Fermi band as the band unit (``RedoxConfig(carrier_band=True, band_form='fermi')``; C4,
experiments/benchmarks_4g/README.md, the addendum of 2026-11-10 22:35).

The band unit's state s stands for the filling N = N0 - s of the band (charge -N); its energy with the atoms at n is

    E(n, N) = E_at(n) + F_b(N) + sum_v rho_N(v) c_v(n),
    c_v(n)  = - sum_u W_vu (n_u - f_u) + sum_{u not a site} delta_uv(n_u),

F_b and rho_N from ``trace_redox.fermi.band_free_energy`` (stage 1), eps, V and tau as in SPEC 20.  T1-T4 hold the
unit's arrays against the atom-level configuration energies; T5, T6 are the module's.
"""
import dataclasses
import itertools
import math

import pytest
import torch
from redox_helpers import edges, lattice_graph, make_heads, neighbour_list, periodic_edges
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

KT = 0.1
SITES = [0, 1, 2, 3]
FRAMES = torch.tensor([2, 2, 2, 2, -2, 0])
FIRST, LAST = -2, 3


def _system(seed=0, second=True, lowered=0.0):
    """Six units: four band sites (frame +2) on a 2.1 A square, an anion (frame -2) and an adatom (frame 0), with a
    random ladder, potential, positive-definite W and pair tables of every kind (site-site, site-other in both orders,
    other-other), each zero at its pair's frame states as the model's are (measured, README addendum item 3).
    ``second`` False: site 2 has no second-electron state.  ``lowered``: eV by which site 0's one-electron level is
    lowered (a band with a gap)."""
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float64)
    U, S = 6, LAST - FIRST + 1
    states = torch.arange(FIRST, LAST + 1, dtype=torch.float64)
    L = torch.tensor([0, 0, 0, 0, -2, -1])
    K = torch.tensor([3, 3, 3, 3, 0, 1])
    if not second:
        L[2] = 1
    e_hat = rnd(U, S)
    e_hat[0, 1 - FIRST] -= lowered                                   # site 0 at f - 1 = +1
    b = 0.3 * rnd(U)
    A = rnd(U, U)
    W = A @ A.T / U + torch.eye(U, dtype=torch.float64)
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.1, 0.0, 0.0], [2.1, 2.1, 0.0], [0.0, 2.1, 0.0],
                        [1.05, 1.05, 1.6], [1.05, 1.05, 4.0]], dtype=torch.float64)
    ei = neighbour_list(pos, r_max=4.5)
    _, el = edges(pos, ei)
    pi = torch.tensor([0, 2, 0, 5, 4, 4])
    pj = torch.tensor([1, 3, 4, 1, 5, 2])
    Xi = 0.2 * rnd(6, S, S)
    p = torch.arange(6)
    Xi[p, FRAMES[pi] - FIRST, FRAMES[pj] - FIRST] = 0.0
    return dict(e_hat=e_hat, states=states, b=b, W=W, L=L, K=K, frames=FRAMES, ei=ei, el=el, pairs=(pi, pj, Xi))


def _atoms_energy(system, n):
    from trace_redox.diabatic import configuration_energies
    return configuration_energies(n.reshape(1, -1), system["e_hat"], system["states"], system["b"], system["W"],
                                  pairs=system["pairs"])[0]


def _band_arrays(system, unit):
    """eps, V of the band from atom-level configuration energies (SPEC 20's definitions as finite differences of the
    exact E_at), and tau."""
    f = system["frames"]
    e = torch.eye(6, dtype=torch.long)
    E = lambda n: _atoms_energy(system, n)
    E0 = E(f)
    eps = torch.stack([E(f - e[v]) - E0 for v in SITES])
    V = torch.zeros(4, 4, dtype=torch.float64)
    for a, v in enumerate(SITES):
        for c, w in enumerate(SITES):
            if v != w:
                V[a, c] = E(f - e[v] - e[w]) - E(f - e[v]) - E(f - e[w]) + E0
            elif int(system["L"][v]) <= int(f[v]) - 2:
                V[a, a] = E(f - 2 * e[v]) - 2.0 * E(f - e[v]) + E0
            else:
                V[a, a] = float("inf")
    tau = unit.hopping(torch.tensor(SITES), system["ei"], system["el"], 6)
    return eps, V, tau


def _coupling(system, n):
    """c_v(n) [4] written out: the screened interaction with the atoms' charges off their frames, and the pair tables'
    change next to one band electron at v for the units that are not band sites."""
    f, W = system["frames"], system["W"]
    pi, pj, Xi = system["pairs"]
    c = -(W[SITES] @ (n - f).to(W.dtype))
    k = lambda m: int(m) - FIRST
    for p, (i, j) in enumerate(zip(pi.tolist(), pj.tolist())):
        for v, u, table in ((i, j, Xi[p].T), (j, i, Xi[p])):            # table[state of u, state of v]
            if v in SITES and u not in SITES:
                c[SITES.index(v)] += (table[k(n[u]), k(f[v] - 1)] - table[k(n[u]), k(f[v])]
                                      - table[k(f[u]), k(f[v] - 1)] + table[k(f[u]), k(f[v])])
    return c


def _unit_energy(system, aug, n, s):
    """The energy of atoms at n and the band unit at state s, from the augmented arrays."""
    from trace_redox.diabatic import configuration_energies
    pairs = tuple(torch.cat([x, y]) for x, y in zip(system["pairs"], aug["pairs"]))
    cfg = torch.cat([n, torch.tensor([s])]).reshape(1, -1)
    return configuration_energies(cfg, aug["e_hat"], system["states"], aug["b"], aug["W"], pairs=pairs)[0]


def _augment(system, unit, reference, T=None):
    return unit.augment_fermi(system["e_hat"], system["states"], system["b"], system["W"], system["L"], system["K"],
                              system["frames"], T, system["ei"], system["el"], pairs=system["pairs"], kT=KT,
                              reference=reference)


# ------------------------------------------------------------------------------------------------ T1: the energy
@pytest.mark.parametrize("seed", [0, 1])
def test_the_unit_energy_is_the_band_functional_in_the_atoms_field(seed):
    """T1.  For every filling the unit holds and atoms off their frames, the augmented arrays give
    E_at(n) + F_b(N) + sum_v rho_N(v) c_v(n).  With the sites at their frames c_v(n) is the exact change of one band
    electron's energy at v, [E_at(n - e_v) - E_at(n)] - eps_v; with a site above its frame it is the written form."""
    from trace_redox.band import BandUnit
    from trace_redox.fermi import band_free_energy
    system, unit = _system(seed), BandUnit(r_cut=3.5)
    eps, V, tau = _band_arrays(system, unit)
    N0 = 2
    aug = _augment(system, unit, N0)
    assert int(aug["reference"]) == N0 and aug["omega"] is None
    assert int(aug["K"][-1]) == min(LAST, N0) and int(aug["L"][-1]) == max(FIRST, N0 - 8)
    f, e = system["frames"], torch.eye(6, dtype=torch.long)
    fillings = [N0 - s for s in range(int(aug["L"][-1]), int(aug["K"][-1]) + 1)]
    assert fillings == [4, 3, 2, 1, 0]
    F, rho = band_free_energy(torch.diag(eps) - tau, V, fillings, KT)
    worst = 0.0
    for extra in ((0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 1)):          # sites at, and above, their frames
        for anion, adatom in itertools.product((-2, -1, 0), (-1, 0, 1)):
            n = f.clone()
            n[:4] += torch.tensor(extra)
            n[4], n[5] = anion, adatom
            c = _coupling(system, n)
            if not any(extra):
                exact = torch.stack([_atoms_energy(system, n - e[v]) - _atoms_energy(system, n) for v in SITES]) - eps
                assert float((c - exact).abs().max()) <= 1e-11 * (1.0 + float(exact.abs().max()))
            for k, N in enumerate(fillings):
                reference = _atoms_energy(system, n) + F[k] + rho[k] @ c
                value = _unit_energy(system, aug, n, N0 - N)
                worst = max(worst, abs(float(value - reference)) / (1.0 + abs(float(reference))))
    assert worst <= 1e-11


def test_the_band_row_of_b_and_W_is_empty_and_missing_states_cost_nothing_in_the_tables():
    """T1, the assembly: the band's coupling to the atoms depends on the filling through rho_N, so it sits in pair
    tables; its entries of b and W are zero.  States the band does not have are +inf in its ladder and zero (finite)
    in the tables, which the move routines index without a bound."""
    from trace_redox.band import BandUnit
    system, unit = _system(0), BandUnit(r_cut=3.5)
    aug = _augment(system, unit, 1)
    assert float(aug["b"][-1]) == 0.0 and float(aug["W"][-1].abs().max()) == 0.0 and float(aug["W"][:, -1].abs().max()) == 0.0
    assert torch.equal(aug["W"][:6, :6], system["W"]) and torch.equal(aug["b"][:6], system["b"])
    row = aug["e_hat"][-1]
    valid = torch.tensor([FIRST <= s <= LAST and 0 <= 1 - s <= 8 for s in range(FIRST, LAST + 1)])
    assert bool(torch.isfinite(row[valid]).all()) and bool(torch.isinf(row[~valid]).all())
    pi, pj, Xi = aug["pairs"]
    assert bool((pj == 6).all()) and sorted(pi.tolist()) == list(range(6))
    assert bool(torch.isfinite(Xi).all()) and float(Xi[:, :, ~valid].abs().max()) == 0.0
    assert float(Xi[:, :, 1 - FIRST].abs().max()) == 0.0                # the empty band (N = 0, state +1) couples to nothing
    # the sites keep their frames as their lowest state (the band carries their electrons)
    assert aug["L"][:4].tolist() == [2, 2, 2, 2] and aug["L"][4:6].tolist() == [-2, -1]


# ------------------------------------------------------------------------------------- T2: the reference filling
def test_the_energy_of_a_filling_does_not_depend_on_the_reference():
    """T2.  N0 only places the window of fillings on the state axis: a filling held under two references has one
    energy, and one charge shape."""
    from trace_redox.band import BandUnit
    system, unit = _system(2), BandUnit(r_cut=3.5)
    low, high = _augment(system, unit, 1), _augment(system, unit, 3)
    assert (int(low["L"][-1]), int(low["K"][-1])) == (-2, 1) and (int(high["L"][-1]), int(high["K"][-1])) == (-2, 3)
    n = system["frames"].clone()
    n[1], n[4], n[5] = 3, -1, 1
    cfg = lambda s: torch.cat([n, torch.tensor([s])]).reshape(1, -1)
    for N in (1, 2, 3):
        a, c = _unit_energy(system, low, n, 1 - N), _unit_energy(system, high, n, 3 - N)
        assert abs(float(a - c)) <= 1e-11 * (1.0 + abs(float(a)))
        qa, qc = low["charges"].atoms(cfg(1 - N)), high["charges"].atoms(cfg(3 - N))
        assert float((qa - qc).abs().max()) <= 1e-12


def test_a_site_without_a_second_electron_state_limits_the_band_to_one_electron():
    """T2.  Infinite U on one site: the first-order energy of two electrons is infinite wherever both spins reach
    that site, so the unit holds N = 0, 1 (as SPEC 20's did); a reference outside what the band holds is refused."""
    from trace_redox.band import BandUnit
    system, unit = _system(0, second=False), BandUnit(r_cut=3.5)
    aug = _augment(system, unit, 1)
    assert (int(aug["L"][-1]), int(aug["K"][-1])) == (0, 1)
    assert bool(torch.isfinite(aug["e_hat"][-1, [0 - FIRST, 1 - FIRST]]).all())
    with pytest.raises(ValueError, match="reference"):
        _augment(system, unit, 2)
    with pytest.raises(ValueError, match="reference"):
        _augment(_system(0), unit, -1)
    with pytest.raises(ValueError, match="reference"):
        _augment(_system(0), unit, 9)


# ------------------------------------------------------------------------------------------ T3: the SPEC 20 limit
def test_a_gapped_band_at_reference_zero_is_the_spec20_unit():
    """T3.  With a gap far above kT the Fermi band's N = 0, 1, 2 are SPEC 20's m = 0, 1, 2 (lambda_1 and
    2 lambda_1 + J_11, the charge in phi_1^2): the two forms give one energy for every configuration."""
    from trace_redox.band import BandUnit
    from trace_redox.diabatic import configuration_energies
    system, unit = _system(3, lowered=18.0), BandUnit(r_cut=3.5)
    args = (system["e_hat"], system["states"], system["b"], system["W"], system["L"], system["K"], system["frames"],
            None, system["ei"], system["el"])
    old = unit.augment(*args, pairs=system["pairs"])
    new = unit.augment_fermi(*args, pairs=system["pairs"], kT=KT, reference=0)
    ev = torch.linalg.eigvalsh(torch.diag(new["eps"]) - new["tau"])
    assert float(ev[1] - ev[0]) > 10.0
    assert (int(new["L"][-1]), int(new["K"][-1])) == (int(old["L"][-1]), int(old["K"][-1])) == (-2, 0)
    old_pairs = tuple(torch.cat([x, y]) for x, y in zip(system["pairs"], old["pairs"]))
    worst = 0.0
    for extra in ((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 1, 1)):
        for anion, adatom, s in itertools.product((-2, -1, 0), (-1, 0, 1), (0, -1, -2)):
            n = system["frames"].clone()
            n[:4] += torch.tensor(extra)
            n[4], n[5] = anion, adatom
            cfg = torch.cat([n, torch.tensor([s])]).reshape(1, -1)
            legacy = configuration_energies(cfg, old["e_hat"], system["states"], old["b"], old["W"], pairs=old_pairs)[0]
            worst = max(worst, abs(float(_unit_energy(system, new, n, s) - legacy)))
    assert worst <= 1e-9


# ------------------------------------------------------------------------------------------------ T4: the charges
def test_the_band_charge_sits_on_the_sites_with_the_fillings_shape():
    """T4.  Atom charges of a configuration: n_v - rho_N(v) on the sites, n_u elsewhere; they sum to the total."""
    from trace_redox.band import BandUnit, atom_charges
    from trace_redox.fermi import band_free_energy
    system, unit = _system(4), BandUnit(r_cut=3.5)
    eps, V, tau = _band_arrays(system, unit)
    N0 = 3
    aug = _augment(system, unit, N0)
    charges = aug["charges"]
    rows, expected = [], []
    for s, anion in itertools.product(range(-2, 4), (-2, 0)):
        n = system["frames"].clone()
        n[4] = anion
        rows.append(torch.cat([n, torch.tensor([s])]))
        rho = band_free_energy(torch.diag(eps) - tau, V, [N0 - s], KT)[1][0]
        q = n.to(torch.float64)
        q[:4] -= rho
        expected.append(q)
    configs, expected = torch.stack(rows), torch.stack(expected)
    q = charges.atoms(configs)
    assert q.shape == (len(rows), 6) and float((q - expected).abs().max()) <= 1e-12
    total = configs[:, :6].sum(1).to(torch.float64) - (N0 - configs[:, 6]).to(torch.float64)
    assert float((q.sum(1) - total).abs().max()) <= 1e-12
    assert float((charges.onehot(configs) @ charges.shape - charges.band(configs)).abs().max()) == 0.0
    # one reading of "atom charges" for every kind of unit map
    assert torch.equal(atom_charges(configs, charges, torch.float64), q)
    assert torch.equal(atom_charges(configs[:, :6], None, torch.float64), configs[:, :6].to(torch.float64))
    omega = torch.rand(7, 6, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    assert torch.equal(atom_charges(configs, omega, torch.float64), configs.to(torch.float64) @ omega)


# ------------------------------------------------------------------------------------------------ T5: the module
def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True, **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


def _doped_block(au=False):
    """A 3x3x2 rock-salt MgO block (a = 4.212 A) with three Mg -> Al: the frames sum to +3, so a neutral block holds
    three electrons that are in no closed-shell frame state.  Optionally an Au atop an O."""
    a = 4.212
    pos, z = [], []
    for i in range(3):
        for j in range(3):
            for k in range(2):
                pos.append([i * a / 2, j * a / 2, k * a / 2])
                z.append(12 if (i + j + k) % 2 == 0 else 8)
    for i in (0, 8, 16):
        assert z[i] == 12
        z[i] = 13
    if au:
        o = next(idx for idx, zz in enumerate(z) if zz == 8 and pos[idx][2] > 1.0)
        pos.append([pos[o][0], pos[o][1], pos[o][2] + 2.2])
        z.append(79)
    return torch.tensor(z), torch.tensor(pos, dtype=torch.float64)


def _run(module, heads, z, pos, Q, seed=7):
    h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    return module(h, pos, ei, vec, length, data={"z": z, "total_charge": float(Q)})


def test_the_band_form_is_validated():
    from trace_redox import RedoxConfig
    base = dict(fragment_ensemble="diabatic", anchor="formal", max_charge=3)
    assert RedoxConfig(carrier_band=True, **base).band_form == "phi1"            # the legacy unit stays the default
    assert RedoxConfig(carrier_band=True, band_form="fermi", **base).band_form == "fermi"
    with pytest.raises(ValueError, match="band_form"):
        RedoxConfig(carrier_band=True, band_form="bloch", **base)
    with pytest.raises(ValueError, match="band_form"):
        RedoxConfig(band_form="fermi", **base)                                    # needs the band unit


@pytest.mark.parametrize("space", ["global", "clusters"])
def test_the_module_holds_three_band_electrons(space):
    """T5.  Three Al in a neutral block: the atoms cannot go below their frames, so the band holds at least three
    electrons.  SPEC 20's unit has two states for them and refuses the structure; the Fermi unit evaluates it, with
    the band's charge on the cation sites and the total conserved."""
    from trace_redox.constants import formal_frame
    from trace_redox.diabatic import ConfigurationSearchError
    z, pos = _doped_block()
    frames = formal_frame(z)
    assert int(frames.sum()) == 3
    legacy, heads = _module(carrier_band=True, diabatic_space=space)
    with pytest.raises(ConfigurationSearchError, match="capacity"):
        _run(legacy, heads, z, pos, 0)
    module, heads = _module(carrier_band=True, band_form="fermi", diabatic_space=space)
    energy, result = _run(module, heads, z, pos, 0)
    assert math.isfinite(float(energy))
    assert result.omega is None
    assert abs(float(result.free_charge.sum())) <= 1e-9 and abs(float(result.unit_charge.sum())) <= 1e-9
    held = -float(result.unit_charge[-1])
    assert held >= 3.0 - 1e-9
    excess = result.free_charge - frames.to(result.free_charge)
    cation = frames > 0
    assert float(excess[~cation].min()) >= -1e-9                                  # anions never below their frames
    assert abs(float(excess[cation].sum()) + held - float((result.unit_charge[:-1] - frames).sum())) <= 1e-9
    last = module.band_diagnostics(*_inputs(heads, z, pos, 0))
    populations = last["populations"]                                             # of the band unit's states
    states = torch.arange(-2, 4, dtype=populations.dtype)
    assert abs(float(populations.sum()) - 1.0) <= 1e-9
    assert abs(last["reference"] - float(populations @ states) - held) <= 1e-9   # -<N> is the unit's charge


def _inputs(heads, z, pos, Q, seed=7):
    h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    return h, pos, ei, vec, length, {"z": z, "total_charge": float(Q)}


#: T5's occupied bands: what the module is given, whether an Au sits above the block, and the premise each must meet
FILLED = {
    "adatom": (dict(), True),                               # the Au takes one of the three electrons, the band two
    "three": (dict(), False),                               # three electrons, more than SPEC 20's unit holds
    "mixed": (dict(diabatic_window=10.0, kT=1.0), False),   # an ensemble that mixes the fillings 3 and 4
}


def _filled(case, space="global"):
    """Module, heads, species, positions and features of one occupied-band case, with its premise checked: the test
    would say nothing about the Fermi unit if the band were empty or the ensemble held one filling."""
    redox, au = FILLED[case]
    module, heads = _module(carrier_band=True, band_form="fermi", diabatic_space=space, **redox)
    z, pos = _doped_block(au=au)
    h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    data = {"z": z, "total_charge": 0.0}
    _, result = module(h, pos, ei, vec, length, data=data)
    held = -float(result.unit_charge[-1])
    populations = module._band_last["populations"]
    if case == "adatom":
        assert float(result.free_charge[-1]) < -0.5 and held > 1.5
    elif case == "three":
        assert held > 2.5
    else:
        assert float(populations.sort().values[-2]) > 1e-3 and held > 2.5
    return module, heads, z, pos, h, ei


@pytest.mark.parametrize("space", ["global", "clusters"])
@pytest.mark.parametrize("case", sorted(FILLED))
def test_forces_are_minus_the_gradient_with_a_filled_band(case, space):
    """T5.  Central differences of the energy against autograd with the band occupied: two electrons beside a charged
    adatom, three (beyond SPEC 20), and an ensemble that mixes two fillings (the pair tables and the filling-dependent
    charge shape then carry part of the force)."""
    module, heads, z, pos0, h, ei = _filled(case, space)

    def energy(p):
        vec, length = edges(p, ei)
        return module(h, p, ei, vec, length, data={"z": z, "total_charge": 0.0})[0]

    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    for i, axis in [(0, 0), (1, 2), (len(z) - 1, 2), (8, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-7 * (1.0 + abs(fd))


@pytest.mark.parametrize("case", ["three", "mixed"])
def test_the_force_loss_has_a_gradient_with_a_filled_band(case):
    """T5.  Training differentiates the forces: d/ds of sum |dE/dR|^2 for the band's hopping parameter s, by autograd
    through the Fermi operator's polynomial and by central differences of the autograd forces."""
    module, heads, z, pos0, h, ei = _filled(case)

    def loss(create_graph):
        pos = pos0.clone().requires_grad_(True)
        vec, length = edges(pos, ei)
        e = module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})[0]
        (g,) = torch.autograd.grad(e, pos, create_graph=create_graph)
        return (g * g).sum()

    s = module.band.log_factor
    (analytic,) = torch.autograd.grad(loss(True), s)
    assert math.isfinite(float(analytic))
    step = 1e-4
    with torch.no_grad():
        s.add_(step)
    up = float(loss(False))
    with torch.no_grad():
        s.sub_(2 * step)
    down = float(loss(False))
    with torch.no_grad():
        s.add_(step)
    fd = (up - down) / (2 * step)
    assert abs(fd - float(analytic)) <= 1e-6 * (1.0 + abs(fd))


# ------------------------------------------------------------------------------------------------ T6: SPEC 37
def test_bvk_sampling_takes_an_occupied_fermi_band_and_still_refuses_the_legacy_unit():
    """T6.  SPEC 37 refuses a structure whose SPEC 20 band unit is occupied: that unit holds two electrons per
    STRUCTURE, so a supercell is not a replication of the cell.  The Fermi unit holds what the supercell needs: the
    mesh evaluates and is the explicit supercell's energy per cell."""
    from test_redox_bvk import BULK, FORMAL, _evaluate, _supercell, _with
    from test_redox_bvk import _module as bvk_module
    from test_redox_penetration import _rocksalt
    z, pos, cell = _rocksalt()
    pos = pos + 0.05 * torch.randn(8, 3, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    module, heads = bvk_module(**FORMAL, carrier_band=True, band_form="fermi")
    h = torch.randn(8, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    with torch.no_grad():
        record = _evaluate(module, pos, h, z, cell, BULK, 0.0)[1]
        assert -float(record.unit_charge[-1]) > 1.0                               # the band is occupied at Gamma
        sampled, home = _evaluate(_with(module, bvk_mesh=(2, 1, 1)), pos, h, z, cell, BULK, 0.0)
        assert home.bvk_mesh == (2, 1, 1) and home.free_charge.shape == (8,)
        _with(module, bvk_mesh=(1, 1, 1))
        reference = float(_evaluate(module, *_supercell(pos, h, z, cell, (2, 1, 1)), BULK, 0.0)[0]) / 2
        assert abs(float(sampled) - reference) <= 1e-9 * (1.0 + abs(reference))
        with pytest.raises(NotImplementedError, match="band unit"):
            _evaluate(_with(module, band_form="phi1", bvk_mesh=(2, 1, 1)), pos, h, z, cell, BULK, 0.0)
