"""SPEC 40 (2026-11-13): the two-site sector of a pair of s1 frame atoms from first principles.

For two half-filled s frames the ensemble is a two-site Hubbard model.  With the pair's ground-state interaction energy
E_gs and its vertical sigma sigma* excitations E_T, E_S (PBE, TDA), the covalent configuration is E_cov = E_gs + E_T,
the ionic gap U = E_S - E_T and the hop t = sqrt(E_T E_S) / 2 (ESTABLISHED: the two-electron two-orbital mapping).
The module takes two of the three from first principles (trace_redox/two_site.py):

    the covalent relaxation   O_ab(f_a, f_b; R) = -rho_ab(R),  rho = Phi_ab (SPEC 39) - E_cov, on the frame configuration only
    the hop                   tau_ab(R) = -t_ab(R) w(R), with spin pairing (SPEC 29): sqrt(2) t for the neutral pair, t for one carrier
"""
import math

import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

EPS = torch.finfo(torch.float64).eps
PAIRS = [(1, 1), (47, 47), (79, 79)]


def _tensor(x):
    return torch.as_tensor(x, dtype=torch.float64)


def test_the_first_principles_points_obey_the_two_site_identities():
    """For every stored point: E_cov = E_gs + E_T, U = E_S - E_T, t = sqrt(E_T E_S) / 2, and the two-site ground state
    E_cov + (U - sqrt(U^2 + 16 t^2)) / 2 is E_gs again (the identity that makes E_cov the hop's partner).  The triplet
    root carries at least half of the sigma sigma* weight; t is positive and falls with R; U is positive."""
    from trace_redox.two_site import two_site
    ts = two_site()
    assert ts.pairs == PAIRS
    left_out = {p: [(f["R"], f["why"]) for f in ts.flagged[p]] for p in PAIRS}
    assert left_out == {(1, 1): [(0.6, "singlet weight below 0.5")], (47, 47): [],
                        (79, 79): [(2.6, "singlet weight below 0.5"), (2.8, "singlet weight below 0.5"), (3.0, "root jump")]}
    for p in PAIRS:
        d = ts.points[p]
        ok = torch.isfinite(d["t"])
        assert int(ok.sum()) >= 4 and bool((d["weight_T"] >= 0.5).all())
        assert bool((d["weight_S"][ok] >= 0.5).all()), p          # the pre-registered rule: a root below 0.5 is not used
        assert torch.allclose(d["E_cov"], d["E_gs"] + d["E_T"], rtol=0, atol=2e-6)       # the file stores six decimals
        U, t, ET, ES = d["U"][ok], d["t"][ok], d["E_T"][ok], d["E_S"][ok]
        assert torch.allclose(U, ES - ET, rtol=0, atol=2e-6) and torch.allclose(t, torch.sqrt(ET * ES) / 2, rtol=0, atol=2e-6)
        ground = d["E_cov"][ok] + 0.5 * (U - torch.sqrt(U * U + 16 * t * t))
        assert float((ground - d["E_gs"][ok]).abs().max()) <= 1e-5, p
        assert bool((t > 0).all()) and bool((t[1:] < t[:-1]).all()) and bool((U > 0).all()), p


@pytest.mark.parametrize("pair", PAIRS)
def test_the_hop_goes_through_its_points_keeps_its_sign_and_falls_with_distance(pair):
    """t(R) equals the first-principles value at every point, is positive and strictly decreasing from 0.6 of the
    first point to 1.5 A beyond the last (an exponential continuation), and is C^1 at the ends."""
    from trace_redox.two_site import two_site
    ts = two_site()
    d = ts.points[pair]
    ok = torch.isfinite(d["t"])
    R, t = d["R"][ok], d["t"][ok]
    got = ts.hop(pair[0], pair[1], R)
    assert float((got - t).abs().max()) <= 64 * EPS * float(t.max()), (got - t).tolist()
    x = torch.linspace(0.6 * float(R[0]), float(R[-1]) + 1.5, 400, dtype=torch.float64).requires_grad_(True)
    y = ts.hop(pair[0], pair[1], x)
    (g,) = torch.autograd.grad(y.sum(), x)
    assert bool((y > 0).all()) and bool((g < 0).all()), pair
    for end in (float(R[0]), float(R[-1])):
        h = 1e-5
        e = _tensor([end - h, end, end + h])
        v = torch.log(ts.hop(pair[0], pair[1], e))
        left, right = float((v[1] - v[0]) / h), float((v[2] - v[1]) / h)
        assert abs(left - right) <= 1e-3 * (1.0 + abs(left)), (pair, end, left, right)


@pytest.mark.parametrize("pair", PAIRS)
def test_the_covalent_level_is_the_first_principles_one(pair):
    """Phi (SPEC 39) - rho equals E_cov at every first-principles distance: by construction of rho, to the frame table's
    accuracy; rho is positive, falls with distance and is zero at and beyond its cutoff."""
    from trace_redox.frame_pair import frame_pair
    from trace_redox.two_site import RHO_CUT, RHO_ON, two_site
    ts, fp = two_site(), frame_pair()
    d = ts.points[pair]
    R = d["R"]
    es, pauli = fp.pair_parts(pair[0], pair[1], R)
    level = es + pauli - ts.relaxation(pair[0], pair[1], R)
    assert float((level - d["E_cov"]).abs().max()) <= 1e-9 * (1.0 + float((es + pauli).abs().max())), (level - d["E_cov"]).tolist()
    x = torch.linspace(float(R[0]) - 0.3, RHO_ON, 200, dtype=torch.float64)
    rho = ts.relaxation(pair[0], pair[1], x)
    assert bool((rho > 0).all()) and bool((rho[1:] < rho[:-1]).all()), pair
    assert float(ts.relaxation(pair[0], pair[1], _tensor([RHO_CUT, RHO_CUT + 1.0])).abs().max()) == 0.0


def test_a_pair_whose_frozen_term_lies_below_its_covalent_level_is_refused(tmp_path):
    """rho = Phi - E_cov must be positive (the frozen high-spin product is a trial state of the no-hop level).  A table
    whose covalent level lies above SPEC 39's frozen term is refused, not applied: this was H0-H0 while that pair
    carried SPEC 39's electron-gas default (2.62 eV at 0.74 A against the high-spin frozen product's 10.03 eV), before
    it had its own first-principles points."""
    import json

    from trace_redox.two_site import DATA_TWO_SITE, TwoSite
    with open(DATA_TWO_SITE) as fh:
        data = json.load(fh)
    data["pairs"]["79-79"]["E_cov"] = [v + 20.0 for v in data["pairs"]["79-79"]["E_cov"]]
    path = tmp_path / "two_site_raised.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="frozen term"):
        TwoSite(str(path)).relaxation(79, 79, _tensor([2.47]))
    assert float(TwoSite(DATA_TWO_SITE).relaxation(1, 1, _tensor([0.74]))[0]) > 4.0        # H0-H0 with its own points


def test_only_neutral_frames_and_no_self_images():
    """The tabulated pairs are pairs of neutral s1 frames: another frame for such an atom is refused; an s1 atom that
    meets its own periodic image inside the relaxation's range is refused; atoms of other elements, and cells whose
    other atoms meet their images, are not looked at."""
    from trace_redox.two_site import RHO_CUT, two_site
    ts = two_site()
    states = _tensor([-1.0, 0.0, 1.0, 2.0])
    pos = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5], [2.0, 0.0, 0.0]])
    z = torch.tensor([79, 79, 8])
    with pytest.raises(ValueError, match="model's settings"):        # entries present: the call must name the model
        ts.relaxation_pairs(z, states, torch.tensor([0, 0, -2]), pos, None, None)
    out = ts.relaxation_pairs(z, states, torch.tensor([0, 0, -2]), pos, None, None, **_hook())
    assert out is not None and out[0].tolist() == [0] and out[1].tolist() == [1]
    Xi = out[2][0]
    k = {int(n): int((states == n).nonzero()[0]) for n in (-1, 0, 1, 2)}
    rho_minus, rho_plus, lam = (float(v[0]) for v in ts.entry_values(79, 79, _tensor([2.5])))
    assert float(Xi[k[0], k[0]]) == -float(ts.relaxation(79, 79, _tensor([2.5]))[0])
    assert float(Xi[k[-1], k[0]]) == float(Xi[k[0], k[-1]]) == -rho_minus and rho_minus > 0.5
    assert float(Xi[k[1], k[0]]) == float(Xi[k[0], k[1]]) == -rho_plus and rho_plus > 0.5
    assert float(Xi[k[1], k[-1]]) == float(Xi[k[-1], k[1]]) == lam
    assert int((Xi != 0).sum()) == 7 and float(Xi[k[2]].abs().sum()) == 0.0 and float(Xi[k[1], k[1]]) == 0.0
    with pytest.raises(ValueError, match="NEUTRAL"):
        ts.relaxation_pairs(z, states, torch.tensor([1, 0, -2]), pos, None, None)
    small = torch.eye(3, dtype=torch.float64) * (RHO_CUT - 1.0)
    with pytest.raises(ValueError, match="own periodic image"):
        ts.relaxation_pairs(z, states, torch.tensor([0, 0, -2]), pos, small, [True, True, True])
    # ONE gold atom in the same small cell meets its own image at 5 A (rho 0.008 eV there): refused too.  Until the
    # review of 2026-11-14 this case was skipped silently, so the same gold chain was refused, relaxed or left alone
    # according to the number of atoms in the cell.
    with pytest.raises(ValueError, match="own periodic image"):
        ts.relaxation_pairs(torch.tensor([79, 8, 8]), states, torch.tensor([0, -2, -2]), pos, small, [True, True, True])
    large = torch.eye(3, dtype=torch.float64) * (RHO_CUT + 0.1)
    assert ts.relaxation_pairs(torch.tensor([79, 8, 8]), states, torch.tensor([0, -2, -2]), pos, large, [True, True, True]) is None
    # a cell with no s1 atom of the applied pairs is not looked at, whatever its size
    assert ts.relaxation_pairs(torch.tensor([12, 8, 8]), states, torch.tensor([2, -2, -2]), pos, small, [True, True, True]) is None


# ---------------------------------------------------------------------------------------------------- the module

#: the option set the gold pair's entries were built against (tools/electrostatics/build_two_site_entries.py: the single
#: model's, step 33), in the 'global' space so that one Hamiltonian is read
REFERENCE = dict(anchor="formal", fragment_ensemble="diabatic", band_form="fermi", bond_partition=True, bond_pauli=True,
                 bond_pauli_level="almo", bond_pauli_wall=True, carrier_band=True, carrier_induction=True,
                 carrier_overlap="gordon_kim", carrier_penetration=True, coupling_bound="harrison", couplings=True,
                 diabatic_selection="amplitude", diabatic_space="global", exclusion=False, frame_induction_damping=True,
                 frame_polarizability_anchor=True, frame_polarizability_scope="ions", max_charge=3,
                 pauli_table="first_principles", shift_gate=True, state_polarizability=True,
                 state_polarizability_elements=(8, 79), state_polarizability_stiffness="onsite")


def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    cfg = RedoxConfig(**dict(REFERENCE, **redox))
    return RedoxElectrostatics(heads, ElectrostaticConfig(), cfg), heads


def _dimer(R, charge=0.0, Z=79, fixed=False, **redox):
    """A gas-phase homonuclear dimer (Au2 by default) through the module; returns (E, the ensemble's configurations
    [M, 2], its Hamiltonian, the module).  ``fixed``: the package's fixed terms alone (no learned level shift)."""
    module, heads = _module(**redox)
    if fixed:
        module.eval()                                    # the switch is refused in training mode
        module.fixed_terms_only = True
    z = torch.tensor([Z, Z])
    pos = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, R]])
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    h = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    E = module(h, pos, ei, vec, length, data={"z": z, "total_charge": charge})[0]
    last = module._ensemble_last
    return float(E), last["configs"][:, :2].to(torch.long), last["H"].detach(), module


def _row(configs, a, b):
    hit = ((configs[:, 0] == a) & (configs[:, 1] == b)).nonzero().flatten()
    assert int(hit.numel()) == 1, (configs.tolist(), a, b)
    return int(hit[0])


ON = dict(frame_pair=True, spin_pairing=True, two_site=True)
#: what the module hands the pair slot's hook; a direct call must hand it too where the pair has built entries
HOOK = None


def _hook():
    global HOOK
    if HOOK is None:
        from trace_electrostatics.config import ElectrostaticConfig
        from trace_redox import RedoxConfig
        HOOK = dict(config=RedoxConfig(**dict(REFERENCE, **ON)), gaussian_a=float(ElectrostaticConfig().gaussian_a))
    return HOOK
OFF = dict(frame_pair=True, spin_pairing=True)


@pytest.mark.parametrize("R", [2.0, 2.2, 2.47, 2.8, 3.3])
def test_the_module_shifts_each_class_of_configuration_by_its_entry(R):
    """Gas-phase Au2, Au2- and Au2+ against the same module without the terms (the same heads, so every learned part
    cancels): the Hamiltonian's Au0Au0 entry is lower by rho(R); the two ionic entries are higher by lambda_ionic(R);
    the two one-carrier entries of the anion are lower by rho_minus(R) and those of the cation by rho_plus(R).  3.3 A
    lies beyond the last knot (the continuations)."""
    from trace_redox.two_site import two_site
    ts = two_site()
    rho = float(ts.relaxation(79, 79, _tensor([R]))[0])
    rho_minus, rho_plus, lam = (float(v[0]) for v in ts.entry_values(79, 79, _tensor([R])))
    for charge, expect in ((0.0, {(0, 0): -rho, (1, -1): lam, (-1, 1): lam}),
                           (-1.0, {(-1, 0): -rho_minus, (0, -1): -rho_minus}),
                           (1.0, {(1, 0): -rho_plus, (0, 1): -rho_plus})):
        _, c0, H0, _ = _dimer(R, charge=charge, **OFF)
        _, c1, H1, _ = _dimer(R, charge=charge, **ON)
        scale = 1.0 + float(H0.abs().max())
        rows0 = {tuple(c): k for k, c in enumerate(c0.tolist())}
        rows1 = {tuple(c): k for k, c in enumerate(c1.tolist())}
        seen = 0
        for (a, b), shift in expect.items():
            if (a, b) in rows0 and (a, b) in rows1:
                got = float(H1[rows1[(a, b)], rows1[(a, b)]] - H0[rows0[(a, b)], rows0[(a, b)]])
                assert abs(got - shift) <= 256 * EPS * scale, (charge, a, b, got, shift)
                seen += 1
        assert seen >= (1 if charge == 0.0 else 2), (charge, sorted(rows0), sorted(rows1))
    if R <= 3.0:
        assert rho > 0.5 and rho_minus > 0.5 and rho_plus > 1.0


def test_the_isolated_pair_is_first_principles_in_three_charge_states():
    """SPEC 40.7's three conditions, at every tabulated distance, through a module of the reference option set with no
    learned shift: the neutral pair's energy is the table's E_gs, the anion's is E_gs - EA* and the cation's E_gs + IP*
    (EA*, IP* the first-principles values on the exact atomic values), to 1e-6 eV.  The entries were solved by the
    builder on another module (other heads, other features): the test also says they do not depend on either."""
    from trace_redox.two_site import two_site
    ts = two_site()
    car, pts = ts.carriers[(79, 79)], ts.points[(79, 79)]
    e_gs = {round(float(r), 6): float(e) for r, e in zip(pts["R"], pts["E_gs"])}
    assert car["R"].tolist() == [2.0, 2.1, 2.2, 2.3, 2.47, 2.6, 2.8, 3.0] == ts.entries[(79, 79)]["R"].tolist()
    for k, R in enumerate(car["R"].tolist()):
        E0 = _dimer(R, fixed=True, **ON)[0]
        Em = _dimer(R, charge=-1.0, fixed=True, **ON)[0]
        Ep = _dimer(R, charge=1.0, fixed=True, **ON)[0]
        assert abs(E0 - e_gs[round(R, 6)]) <= 1e-6, (R, E0, e_gs[round(R, 6)])
        assert abs((E0 - Em) - float(car["EA"][k])) <= 1e-6, (R, E0 - Em, float(car["EA"][k]))
        assert abs((Ep - E0) - float(car["IP"][k])) <= 1e-6, (R, Ep - E0, float(car["IP"][k]))
    # without the entries the same module misses them by eV: the test can fail
    saved = ts.entries.pop((79, 79))
    try:
        E0, Em = _dimer(2.47, fixed=True, **ON)[0], _dimer(2.47, charge=-1.0, fixed=True, **ON)[0]
        assert abs((E0 - Em) - 1.893) > 1.0
    finally:
        ts.entries[(79, 79)] = saved


def test_the_fixed_terms_of_the_pair_do_not_depend_on_the_weights():
    """What the entries rest on: with no learned shift, the three energies of the isolated gold pair are the same
    whatever the weights (the frame polarizability of Au0 is tabulated, the pair's electrostatics and overlap are
    tables).  Every parameter of the heads and of the redox module is perturbed at random; with the learned shift on
    the same perturbation moves the energies, so the comparison can fail."""
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads(0)
    torch.manual_seed(1)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), RedoxConfig(**dict(REFERENCE, **ON)))
    module.eval()
    z = torch.tensor([79, 79])
    pos = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.47]])
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    h = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(9), dtype=torch.float64)

    def energies(fixed):
        module.fixed_terms_only = fixed
        return [float(module(h, pos, ei, vec, length, data={"z": z, "total_charge": q})[0]) for q in (0.0, -1.0, 1.0)]

    before = {fixed: energies(fixed) for fixed in (True, False)}
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for prm in list(module.parameters()) + list(heads.parameters()):
            prm.add_(0.5 * torch.randn(prm.shape, generator=g, dtype=prm.dtype))
    after = {fixed: energies(fixed) for fixed in (True, False)}
    assert max(abs(a - b) for a, b in zip(before[True], after[True])) <= 1e-10, (before[True], after[True])
    assert max(abs(a - b) for a, b in zip(before[False], after[False])) > 1e-3, (before[False], after[False])


def test_the_fixed_terms_switch_is_declared_off_and_refused_in_training():
    """``fixed_terms_only`` is an attribute of every module, False, not a parameter or a buffer (so no checkpoint
    carries it), and a forward pass in training mode with it on is refused."""
    module, heads = _module(**ON)
    assert module.fixed_terms_only is False and "fixed_terms_only" not in module.state_dict()
    module.fixed_terms_only = True
    z = torch.tensor([79, 79])
    pos = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.47]])
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    h = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    assert module.training
    with pytest.raises(RuntimeError, match="training mode"):
        module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})


def _fixed_energies(redox, electrostatic=None, without_entries=True, distances=(2.0, 2.47, 3.0)):
    """The isolated gold pair's three energies at each distance through the package's fixed terms, for an option set; with
    ``without_entries`` the pair's built entries are taken out for the call (so that no option check runs and the bare
    fixed terms are seen)."""
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    from trace_redox.two_site import two_site
    ts = two_site()
    saved = ts.entries.pop((79, 79)) if without_entries else None
    try:
        heads = make_heads(0)
        torch.manual_seed(1)
        module = RedoxElectrostatics(heads, ElectrostaticConfig() if electrostatic is None else electrostatic, RedoxConfig(**redox))
        module.eval()
        module.fixed_terms_only = True
        z = torch.tensor([79, 79])
        h = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        out = []
        for R in distances:
            pos = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, R]])
            ei = neighbour_list(pos, r_max=5.3)
            vec, length = edges(pos, ei)
            out += [float(module(h, pos, ei, vec, length, data={"z": z, "total_charge": q})[0]) for q in (0.0, -1.0, 1.0)]
        return out
    finally:
        if saved is not None:
            ts.entries[(79, 79)] = saved


@pytest.mark.parametrize("change,size", [(dict(bond_partition=False), 0.5), (dict(pauli_table="gordon_kim"), 0.5),
                                         (dict(carrier_induction=False), 0.3), (dict(frame_polarizability_anchor=False), 0.1)])
def test_a_model_with_other_fixed_terms_is_refused(change, size):
    """The entries are the package's fixed terms against first principles.  Each of these options changes the isolated
    pair's fixed-term energies by more than ``size`` eV (measured here with the entries taken out), and a model that
    has it differently is refused with the option named."""
    reference = dict(REFERENCE, **ON)
    moved = max(abs(a - b) for a, b in zip(_fixed_energies(dict(reference, **change)), _fixed_energies(reference)))
    assert moved > size, (change, moved)
    with pytest.raises(ValueError, match=next(iter(change))):
        _dimer(2.47, **dict(ON, **change))


def test_other_settings_the_entries_depend_on_are_refused():
    """The pair's element must be in the state-polarizability list; the Gaussian exponent factor of the electrostatic
    settings (it sets the dipoles' widths) changes the pair's fixed terms by 0.2 eV and is refused when it differs;
    and the entries record rho at their knots: a frame table that has changed since is refused."""
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    from trace_redox.two_site import two_site
    with pytest.raises(ValueError, match="state_polarizability_elements"):
        _dimer(2.47, state_polarizability_elements=(8,), **ON)
    reference = dict(REFERENCE, **ON)
    other = ElectrostaticConfig(gaussian_a=0.25)
    moved = max(abs(a - b) for a, b in zip(_fixed_energies(reference, other), _fixed_energies(reference)))
    assert moved > 0.1, moved
    heads = make_heads(0)
    close = ElectrostaticConfig(gaussian_a=float(ElectrostaticConfig().gaussian_a) + 1e-10)     # 100 x the tolerance
    module = RedoxElectrostatics(heads, close, RedoxConfig(**reference))
    z = torch.tensor([79, 79])
    pos = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.47]])
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    h = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    with pytest.raises(ValueError, match="gaussian_a"):
        module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
    ts = two_site()
    kept = ts.entries[(79, 79)]["rho"]
    ts.entries[(79, 79)]["rho"] = kept + 1e-7                    # ten times the check's tolerance
    try:
        with pytest.raises(ValueError, match="frame table"):
            _dimer(2.47, **ON)
    finally:
        ts.entries[(79, 79)]["rho"] = kept


def test_every_other_option_leaves_the_pairs_fixed_terms_alone():
    """Completeness of the refusal: every field of RedoxConfig and of ElectrostaticConfig is changed in turn (booleans
    flipped, numbers halved and doubled, the named alternatives of the string options).  Each change is either
    refused (an illegal combination, or the option check), or leaves the isolated pair's three fixed-term energies
    where they were to 1e-6 eV, or is the temperature with its own thermal weight (below 1e-4 eV).  The four options
    of the test above and ``gaussian_a`` are the ones that move them; a new option that does must be added to the
    check, and this test says so."""
    import dataclasses

    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig
    reference = dict(REFERENCE, **ON)
    base = _fixed_energies(reference)
    known = {"bond_partition", "pauli_table", "carrier_induction", "frame_polarizability_anchor", "two_site"}
    alternatives = {"band_form": ["phi1"], "diabatic_space": ["clusters"], "coupling_bound": [None], "pauli_table": ["gordon_kim"],
                    "state_polarizability_stiffness": ["block"], "frame_polarizability_scope": ["all"], "bond_pauli_level": ["frz"]}

    def trials(value):
        if isinstance(value, bool):
            return [not value]
        if isinstance(value, int):
            return [v for v in (value // 2, value * 2) if v != value]
        if isinstance(value, float):
            return [0.5 * value, 2.0 * value]
        return []

    moved, tried = [], 0
    for f in dataclasses.fields(RedoxConfig):
        current = reference.get(f.name, getattr(RedoxConfig(), f.name))
        for v in trials(current) + [x for x in alternatives.get(f.name, []) if x != current]:
            try:
                d = max(abs(a - b) for a, b in zip(_fixed_energies(dict(reference, **{f.name: v})), base))
            except (ValueError, NotImplementedError, RuntimeError):
                continue
            tried += 1
            if d > (1e-4 if f.name == "kT" else 1e-6) and f.name not in known:
                moved.append((f.name, v, d))
    for f in dataclasses.fields(ElectrostaticConfig):
        for v in trials(getattr(ElectrostaticConfig(), f.name)):
            try:
                d = max(abs(a - b) for a, b in zip(_fixed_energies(reference, dataclasses.replace(ElectrostaticConfig(), **{f.name: v})), base))
            except (ValueError, NotImplementedError, RuntimeError, TypeError):
                continue
            tried += 1
            if d > 1e-6 and f.name != "gaussian_a":
                moved.append(("electrostatic." + f.name, v, d))
    assert tried > 60 and not moved, (tried, moved)


def test_an_applied_pair_needs_its_entries_and_they_must_belong_to_the_table(tmp_path):
    """A pair marked applied without built entries is refused when the file is read (with the neutral entry alone its
    electron affinity and ionisation energy are 1 eV off); so is a file whose entries were solved with other table
    values (here: one E_gs changed by 1 meV after the build)."""
    import json

    from trace_redox.two_site import DATA_TWO_SITE, TwoSite
    with open(DATA_TWO_SITE) as fh:
        good = json.load(fh)
    assert "solved_with" in good["pairs"]["79-79"]["entries"]
    data = json.loads(json.dumps(good))
    del data["pairs"]["79-79"]["entries"]
    path = tmp_path / "unbuilt.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="no built entries"):
        TwoSite(str(path))
    assert TwoSite(str(path), allow_unbuilt=True).applied == [(79, 79)]            # the builder's own way in
    data = json.loads(json.dumps(good))
    data["pairs"]["79-79"]["E_gs"][4] += 1e-3
    data["pairs"]["79-79"]["E_cov"][4] += 1e-3
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="another table"):
        TwoSite(str(path))


def test_the_entries_end_one_angstrom_beyond_their_last_point():
    """Every entry is its line times the quintic switch between 3.0 and 4.0 A (written out here), a half of the line at
    3.5 A and nothing at and beyond 4.0 A, where the coupling's window has closed the hop the entries complete.
    (Before the review's finding rho_plus continued as an exponential of 2 A decay length and was 0.40 eV at 5 A.)"""
    from trace_redox.two_site import ENTRY_NAMES, ENTRY_TAIL, two_site
    ts = two_site()
    e = ts.entries[(79, 79)]
    assert ENTRY_TAIL == 1.0 and float(e["R"][-1]) == 3.0
    for k, name in enumerate(ENTRY_NAMES):
        y, m = float(e[name][-1]), float(e["slopes"][name][-1])
        for R in (3.25, 3.5, 3.75):
            x = R - 3.0
            want = (y + m * x) * (1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x))
            got = float(ts.entry_values(79, 79, _tensor([R]))[k][0])
            assert abs(got - want) <= 64 * EPS * (1.0 + abs(y)), (name, R, got, want)
        assert float(ts.entry_values(79, 79, _tensor([4.0, 4.5, 5.5]))[k].abs().max()) == 0.0
        assert float(ts.entry_values(79, 79, _tensor([3.0]))[k][0]) == y
        # below the first point (2.0 A): the line of the end slope (rho_plus reaches zero near 1.37 A on it)
        y0, m0 = float(e[name][0]), float(e["slopes"][name][0])
        for R in (1.95, 1.9, 1.8):
            got = float(ts.entry_values(79, 79, _tensor([R]))[k][0])
            assert abs(got - (y0 + m0 * (R - 2.0))) <= 64 * EPS * (1.0 + abs(y0)), (name, R, got)


@pytest.mark.parametrize("R", [2.0, 2.2, 2.47, 2.8])
def test_the_modules_hop_is_the_first_principles_one(R):
    """Gas-phase Au2 with spin pairing: the element between Au0Au0 and Au+Au- is sqrt(2) t(R) in size (a move that
    unpairs two frontier electrons), and in Au2- the element between Au-Au0 and Au0Au- is t(R) (one carrier): one t for
    both sectors, as first principles has it.  The learned head no longer sets either."""
    from trace_redox.two_site import two_site
    t = float(two_site().hop(79, 79, _tensor([R]))[0])
    _, c, H, _ = _dimer(R, **ON)
    cov, ion = _row(c, 0, 0), _row(c, 1, -1)
    assert abs(abs(float(H[cov, ion])) - math.sqrt(2.0) * t) <= 256 * EPS * (1.0 + t), (float(H[cov, ion]), t)
    assert abs(abs(float(H[cov, _row(c, -1, 1)])) - math.sqrt(2.0) * t) <= 256 * EPS * (1.0 + t)
    _, c, H, _ = _dimer(R, charge=-1.0, **ON)
    left, right = _row(c, -1, 0), _row(c, 0, -1)
    assert abs(abs(float(H[left, right])) - t) <= 256 * EPS * (1.0 + t), (float(H[left, right]), t)
    _, c_off, H_off, _ = _dimer(R, **OFF)                               # the learned head's value is something else
    rows = {tuple(v): k for k, v in enumerate(c_off.tolist())}
    if (0, 0) in rows and (1, -1) in rows:                              # (at 2.0 A the fixed terms without SPEC 40 put the
        assert abs(abs(float(H_off[rows[(0, 0)], rows[(1, -1)]])) - math.sqrt(2.0) * t) > 0.1    # ionic pair 2 eV below it)


def test_without_the_flag_the_two_site_code_is_never_reached_and_the_combinations_are_checked(monkeypatch):
    """Off by default; off means the table is not touched.  The term needs SPEC 39's frame term (rho is defined against
    it), spin pairing (one t for both sectors) and couplings."""
    from trace_redox import RedoxConfig
    from trace_redox import two_site as module_file
    assert RedoxConfig().two_site is False

    def forbidden(*args, **kwargs):
        raise AssertionError("two_site was used with the flag off")

    monkeypatch.setattr(module_file.TwoSite, "replace_hops", forbidden)
    monkeypatch.setattr(module_file.TwoSite, "relaxation_pairs", forbidden)
    E, _, _, _ = _dimer(2.47, **OFF)
    assert math.isfinite(E)
    with pytest.raises(AssertionError, match="flag off"):
        _dimer(2.47, **ON)
    base = dict(REFERENCE)
    assert RedoxConfig(frame_pair=True, spin_pairing=True, two_site=True, **base).two_site is True
    with pytest.raises(ValueError, match="two_site"):
        RedoxConfig(spin_pairing=True, two_site=True, **base)                    # no frame term
    with pytest.raises(ValueError, match="two_site"):
        RedoxConfig(frame_pair=True, two_site=True, **base)                      # no spin pairing
    window = dict(base, diabatic_selection="window")
    with pytest.raises(ValueError, match="amplitude"):
        RedoxConfig(frame_pair=True, spin_pairing=True, two_site=True, **window)  # the selection must see the hop
    with pytest.raises(ValueError, match="two_site"):
        RedoxConfig(frame_pair=True, spin_pairing=True, two_site="false", **base)  # a string is not a switch


def test_forces_are_the_gradient_with_the_two_site_terms_on():
    """Au2 next to an O and a Mg (so the pair is not alone): autograd forces of the module against central differences
    of its energy along the Au-Au axis, with the term on."""
    module, heads = _module(**ON)
    z = torch.tensor([79, 79, 8, 12])
    pos0 = _tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.4], [2.3, 0.2, -0.3], [2.2, 2.1, 0.4]])
    h = torch.randn(4, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)

    def energy(pos):
        ei = neighbour_list(pos, r_max=5.3)
        vec, length = edges(pos, ei)
        return module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})[0]

    pos = pos0.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(energy(pos), pos)
    step, tol = EPS ** (1.0 / 3.0), 1e4 * EPS ** (2.0 / 3.0)
    for i, ax in ((1, 2), (0, 2), (1, 0)):
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, ax] += step
        minus[i, ax] -= step
        fd = (float(energy(plus)) - float(energy(minus))) / (2 * step)
        assert abs(fd - float(g[i, ax])) <= tol * (1.0 + abs(fd)), (i, ax, float(g[i, ax]), fd)


def test_periodic_images_add_in_both_terms():
    """Two Au atoms 2.4 A apart along a 6.5 A lattice vector meet a second time through the cell, at 4.1 A: the
    relaxation lists both images of the pair (the sum of the two rho's on the frame-frame entry), and the hop is the
    sum over the pair's edges within the coupling's cutoff, as for the learned coupling."""
    from redox_helpers import lattice_graph, periodic_edges
    from trace_redox.heads import ANCHOR_SWITCH_WIDTH, window_switch
    from trace_redox.two_site import two_site
    ts = two_site()
    z = torch.tensor([79, 79])
    cell = torch.diag(_tensor([9.0, 9.0, 6.5]))
    pbc = (True, True, True)
    pos = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.4]])
    states = _tensor([-1.0, 0.0, 1.0])
    i, j, Xi = ts.relaxation_pairs(z, states, torch.tensor([0, 0]), pos, cell, pbc, **_hook())
    assert i.tolist() == [0, 0] and j.tolist() == [1, 1]
    want = float(ts.relaxation(79, 79, _tensor([2.4]))[0] + ts.relaxation(79, 79, _tensor([4.1]))[0])
    assert abs(float(Xi[:, 1, 1].sum()) + want) <= 64 * EPS * want and int((Xi[:, 1, 1] != 0).sum()) == 2
    r_cut = 5.3
    ei, es = lattice_graph(pos, cell, pbc, r_max=r_cut, shells=1)
    _, length = periodic_edges(pos, cell, ei, es)
    assert sorted(round(float(v), 6) for v in length[ei[0] != ei[1]].tolist()) == [2.4, 2.4, 4.1, 4.1]
    T = ts.replace_hops(torch.full((2, 2), 0.37, dtype=torch.float64), z, ei, length, r_cut)
    R = _tensor([2.4, 4.1])
    hop = -float((ts.hop(79, 79, R) * window_switch(R, r_cut, ANCHOR_SWITCH_WIDTH)).sum())
    assert abs(float(T[0, 1]) - hop) <= 64 * EPS * abs(hop) and float(T[1, 0]) == float(T[0, 1])
    assert float(T[0, 0]) == 0.37 and float(T[1, 1]) == 0.37                       # only the pair's entries are replaced


def test_other_pairs_keep_the_learned_coupling():
    """Au-O, O-Mg and the mixed Ag-Au pair (no points) are left exactly as the head gave them."""
    from trace_redox.two_site import two_site
    z = torch.tensor([79, 47, 8, 12, 79])
    pos = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.6], [2.1, 0.0, 0.0], [2.1, 2.1, 0.0], [0.0, 2.7, 0.3]])
    ei = neighbour_list(pos, r_max=5.3)
    _, length = edges(pos, ei)
    T0 = torch.randn(5, 5, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    T0 = 0.5 * (T0 + T0.T)
    T = two_site().replace_hops(T0, z, ei, length, 5.3)
    changed = (T != T0).nonzero().tolist()
    assert sorted(changed) == [[0, 4], [4, 0]], changed                             # the one Au-Au pair
    assert float(T[0, 4]) < 0.0


def test_a_force_loss_differentiates_with_the_two_site_terms_on():
    """Training differentiates the forces: with the terms on, under anomaly detection, the gradient of a force loss
    with respect to the positions and to every parameter that receives one is finite."""
    module, heads = _module(**ON)
    z = torch.tensor([79, 79, 8, 12])
    pos = _tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.4], [2.3, 0.2, -0.3], [2.2, 2.1, 0.4]]).requires_grad_(True)
    h = torch.randn(4, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos.detach(), r_max=5.3)
    with torch.autograd.set_detect_anomaly(True):
        vec, length = edges(pos, ei)
        E = module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})[0]
        (g,) = torch.autograd.grad(E, pos, create_graph=True)
        (g ** 2).sum().backward()
    assert bool(torch.isfinite(pos.grad).all()) and float(pos.grad.abs().max()) > 0.0
    grads = [p.grad for p in list(module.parameters()) + list(heads.parameters()) if p.grad is not None]
    assert grads and all(bool(torch.isfinite(v).all()) for v in grads)


@pytest.mark.parametrize("space", ["global", "clusters"])
def test_the_module_in_a_periodic_cell(space):
    """A periodic cell of six atoms (7 A; Au2 at 2.5 A beside two MgO units, every contact above 2 A), both ensemble
    spaces: with the terms on the energy is finite, differs from the one without them, the neutral pair is in the
    ensemble, and the force on a gold atom is the central difference of the energy.  (The shared random cluster of
    the helpers puts its first two atoms 0.73 A apart, where the ensemble holds no neutral gold pair: not used.)"""
    from redox_helpers import lattice_graph, periodic_edges
    z = torch.tensor([79, 79, 8, 12, 12, 8])
    pbc = (True, True, True)
    cell = 7.0 * torch.eye(3, dtype=torch.float64)
    pos0 = _tensor([[0.0, 0.0, 0.0], [0.2, 0.1, 2.5], [2.2, 0.1, -0.2], [2.3, 2.2, 0.2], [4.4, 0.2, 3.3], [4.5, 2.4, 3.2]])

    def run(flags, shift=0.0, grad=False):
        module, heads = _module(diabatic_space=space, **flags)
        h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        pos = pos0.clone()
        pos[0, 2] += shift
        pos.requires_grad_(grad)
        ei, es = lattice_graph(pos.detach(), cell, pbc, shells=1)
        vec, length = periodic_edges(pos, cell, ei, es)
        E = module(h, pos, ei, vec, length, pbc=pbc, cell=cell, edge_shift=es, data={"z": z, "total_charge": 0.0})[0]
        if grad:
            (g,) = torch.autograd.grad(E, pos)
            return float(E), float(g[0, 2]), module
        return float(E)

    off = run(OFF)
    on, g, module = run(ON, grad=True)
    assert math.isfinite(on) and abs(on - off) > 1e-3, (on, off)
    last = module._ensemble_last
    if last.get("space") != "clusters":
        cfg = last["configs"][:, :2]
        assert bool(((cfg[:, 0] == 0) & (cfg[:, 1] == 0)).any()), cfg.tolist()
    step, tol = EPS ** (1.0 / 3.0), 1e4 * EPS ** (2.0 / 3.0)
    fd = (run(ON, shift=step) - run(ON, shift=-step)) / (2 * step)
    assert abs(fd - g) <= tol * (1.0 + abs(fd)), (g, fd)


# ------------------------------------------------------------------ added after the review of 2026-11-14 (README)

def test_the_table_carries_the_numbers_of_the_record():
    """The first-principles numbers of SPEC 40.3, written out: a change of the file, of the collector's rules or of
    SPEC 39's table shows here (the identities above are consequences of the collector's own definitions)."""
    from trace_redox.two_site import two_site
    ts = two_site()
    au, ag, hh = ts.points[(79, 79)], ts.points[(47, 47)], ts.points[(1, 1)]
    k = int((au["R"] == 2.47).nonzero()[0])
    # the log's own digits (au2_no_hop_curve.log): E_gs -2.2119, triplet 2.049 (weight 0.77), singlet 3.044 (weight 0.51)
    assert (float(au["E_gs"][k]), float(au["E_T"][k]), float(au["E_S"][k]), float(au["t"][k]), float(au["U"][k])) \
        == (-2.2119, 2.049, 3.044, 1.248715, 0.995)
    assert (float(au["weight_T"][k]), float(au["weight_S"][k])) == (0.77, 0.51)
    assert au["R"].tolist() == [2.0, 2.1, 2.2, 2.3, 2.47, 2.6, 2.8, 3.0] and int(torch.isfinite(au["t"]).sum()) == 5
    assert float(ag["t"][int((ag["R"] == 2.53).nonzero()[0])]) == 1.236255 and int(torch.isfinite(ag["t"]).sum()) == 7
    assert float(hh["t"][int((hh["R"] == 0.74).nonzero()[0])]) == 5.738403 and int(torch.isfinite(hh["t"]).sum()) == 4
    for pair, R, rho in (((79, 79), 2.0, 3.5017), ((79, 79), 2.47, 1.6834), ((79, 79), 2.8, 0.9727),
                         ((47, 47), 2.53, 0.943), ((1, 1), 0.74, 4.351)):
        got = float(ts.relaxation(pair[0], pair[1], _tensor([R]))[0])
        assert abs(got - rho) <= 6e-4, (pair, R, got, rho)                 # the record's last digit
    # SPEC 40.7: the carriers' first-principles values (PBE, on the exact atomic values) and gold's built entries
    car = ts.carriers[(79, 79)]
    k = int((car["R"] == 2.47).nonzero()[0])
    assert (float(car["EA"][k]), float(car["IP"][k])) == (1.893, 9.192)
    assert ts.carriers[(47, 47)]["R"].tolist() == [2.2, 2.3, 2.4, 2.53, 2.65, 2.8, 3.0] and (47, 47) not in ts.entries
    e = ts.entries[(79, 79)]
    for name, values in (("rho_minus", (2.5257, 2.0293, 1.6851, 1.4307, 1.1174, 0.9417, 0.7346, 0.5855)),
                         ("rho_plus", (1.2848, 1.4627, 1.5911, 1.6513, 1.6268, 1.5354, 1.3355, 1.1250)),
                         ("lambda_ionic", (0.9445, 0.7007, 0.4588, 0.2573, 0.0084, -0.1298, -0.3005, -0.4435))):
        assert float((e[name] - _tensor(values)).abs().max()) <= 6e-5, (name, e[name].tolist())


def test_the_hydrogen_pair_is_tabulated_but_not_applied():
    """H-H has points (H2, 0.6-1.4 A) but the module does not use them: in a molecule the two hydrogens are each bonded
    to a third atom and lie beyond the last point, and the bare pair's hop (2.5 eV by continuation at water's 1.51 A)
    is not that pair's.  The hooks leave every H-H entry to the learned head and add no relaxation, in water and in H2."""
    from trace_redox.two_site import two_site
    ts = two_site()
    assert ts.applied == [(79, 79)] and ts.elements == [79]
    assert ts.has(1, 1) and not ts.is_applied(1, 1) and ts.is_applied(79, 79) and not ts.is_applied(47, 79)
    assert ts.has(47, 47) and not ts.is_applied(47, 47)            # silver: no entries can be built yet (see the data file)
    half = math.radians(104.52 / 2)
    water = _tensor([[0.0, 0.0, 0.0], [0.9572 * math.sin(half), 0.0, 0.9572 * math.cos(half)],
                     [-0.9572 * math.sin(half), 0.0, 0.9572 * math.cos(half)]])
    z = torch.tensor([8, 1, 1])
    ei = neighbour_list(water, r_max=5.3)
    _, length = edges(water, ei)
    T0 = torch.randn(3, 3, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    T0 = 0.5 * (T0 + T0.T)
    assert torch.equal(ts.replace_hops(T0, z, ei, length, 4.0), T0)
    assert ts.relaxation_pairs(z, _tensor([-1.0, 0.0, 1.0]), torch.tensor([-2, 0, 0]), water, None, None) is None
    E_on, _, _, _ = _dimer(0.74, Z=1, **ON)
    E_off, _, _, _ = _dimer(0.74, Z=1, **OFF)
    assert E_on == E_off


def test_each_pair_gets_its_own_functions_or_none():
    """Which curve the hooks hand to which pair: gold's to Au-Au; nothing to Ag-Ag (tabulated, withheld: its curves
    differ from gold's by 3-25 % at 2.6 A, so handing it gold's or its own would show) and nothing to the mixed pair."""
    from trace_redox.two_site import two_site
    ts = two_site()
    R = 2.6
    pos = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, R]])
    ei = neighbour_list(pos, r_max=5.3)
    _, length = edges(pos, ei)
    states = _tensor([-1.0, 0.0, 1.0])
    z = torch.tensor([79, 79])
    T = ts.replace_hops(torch.zeros(2, 2, dtype=torch.float64), z, ei, length, 4.0)
    t = float(ts.hop(79, 79, _tensor([R]))[0])
    assert abs(float(T[0, 1]) + t) <= 64 * EPS * t
    i, j, Xi = ts.relaxation_pairs(z, states, torch.tensor([0, 0]), pos, None, None, **_hook())
    assert float(Xi[0, 1, 1]) == -float(ts.relaxation(79, 79, _tensor([R]))[0])
    assert abs(float(ts.hop(47, 47, _tensor([R]))[0]) - t) > 0.02 * t
    assert abs(float(ts.relaxation(47, 47, _tensor([R]))[0]) + float(Xi[0, 1, 1])) > 0.2
    T0 = torch.full((2, 2), 0.37, dtype=torch.float64)
    for other in (torch.tensor([47, 47]), torch.tensor([47, 79])):
        assert torch.equal(ts.replace_hops(T0, other, ei, length, 4.0), T0)
        assert ts.relaxation_pairs(other, states, torch.tensor([0, 0]), pos, None, None) is None


@pytest.mark.parametrize("Z", [79])
def test_the_hop_carries_the_couplings_window(Z):
    """tau = -t(R) w(R) with the coupling's window written out here, w = 1 - x^3 (10 - 15 x + 6 x^2), x = R - (r_cut - 1),
    for r_cut = 4 A: a half at 3.5 A, and exactly zero at and beyond the cutoff (the neighbour list reaches further)."""
    from trace_redox.two_site import two_site
    ts = two_site()
    z = torch.tensor([Z, Z])

    def entry(R):
        pos = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, R]])
        ei = neighbour_list(pos, r_max=5.3)
        _, length = edges(pos, ei)
        assert int(ei.shape[1]) == 2
        return float(ts.replace_hops(torch.full((2, 2), 0.37, dtype=torch.float64), z, ei, length, 4.0)[0, 1])

    for R in (3.25, 3.5, 3.75, 3.99):
        x = R - 3.0
        w = 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x)
        t = float(ts.hop(Z, Z, _tensor([R]))[0])
        assert abs(entry(R) + t * w) <= 64 * EPS * t, (R, entry(R), t * w)
    assert abs(entry(3.5) + 0.5 * float(ts.hop(Z, Z, _tensor([3.5]))[0])) <= 64 * EPS
    assert entry(4.0) == 0.0 and entry(4.5) == 0.0
    assert abs(entry(2.9) + float(ts.hop(Z, Z, _tensor([2.9]))[0])) <= 64 * EPS * 2.0          # below the window: all of t


def test_the_relaxations_switch():
    """rho is switched off between 5 and 6 A by the same quintic, written out here: a half at 5.5 A, nothing at 6 A."""
    from trace_redox.two_site import RHO_CUT, RHO_ON, _log_curve, two_site
    assert (RHO_ON, RHO_CUT) == (5.0, 6.0)
    ts = two_site()
    for pair in ((47, 47), (79, 79)):
        raw = lambda R: float(_log_curve(_tensor([R]), *ts._relaxation_knots(pair))[0])
        for R in (5.25, 5.5, 5.75):
            x = R - 5.0
            s = 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x)
            got = float(ts.relaxation(pair[0], pair[1], _tensor([R]))[0])
            assert abs(got - s * raw(R)) <= 64 * EPS * raw(R), (pair, R, got, s * raw(R))
        assert abs(float(ts.relaxation(pair[0], pair[1], _tensor([5.5]))[0]) - 0.5 * raw(5.5)) <= 64 * EPS * raw(5.5)
        assert float(ts.relaxation(pair[0], pair[1], _tensor([4.999]))[0]) == raw(4.999)
        assert float(ts.relaxation(pair[0], pair[1], _tensor([6.0 - 1e-6]))[0]) <= 1e-13 * raw(6.0)


def test_the_modules_hop_inside_and_beyond_the_window():
    """Au2- through the module: the one-carrier element is t(R) w(R) inside the window (3.25 and 3.5 A) and there is no
    coupling at 4.2 A."""
    from trace_redox.two_site import two_site
    ts = two_site()
    for R in (3.25, 3.5):
        x = R - 3.0
        want = float(ts.hop(79, 79, _tensor([R]))[0]) * (1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x))
        _, c, H, _ = _dimer(R, charge=-1.0, **ON)
        got = abs(float(H[_row(c, -1, 0), _row(c, 0, -1)]))
        assert abs(got - want) <= 256 * EPS * (1.0 + want), (R, got, want)
    _, c, H, _ = _dimer(4.2, charge=-1.0, **ON)
    rows = [k for k in range(int(c.shape[0])) if sorted(c[k].tolist()) == [-1, 0]]
    assert rows and all(float(H[a, b]) == 0.0 for a in rows for b in rows if a != b)


@pytest.mark.parametrize("charge", [0.0, -1.0, 1.0])
def test_the_energy_of_the_dimer_has_no_steps(charge):
    """Gas-phase Au2, Au2- and Au2+ from 1.80 to 3.60 A in steps of 0.01 A: no step of the energy exceeds 1.5 times the
    larger force at its ends times the spacing (the knots of every entry, the last one at 3.0 A, and the hop's window
    are crossed).  (On the 'window' selection the first-principles hop is not seen by the selection and the neutral
    scan has steps of 1.6-2.0 eV where the ionic configuration enters; that selection is refused.)"""
    module, heads = _module(**ON)
    z = torch.tensor([79, 79])
    h = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    E, F = [], []
    grid = [1.80 + 0.01 * k for k in range(181)]
    for R in grid:
        pos = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, R]]).requires_grad_(True)
        ei = neighbour_list(pos.detach(), r_max=5.3)
        vec, length = edges(pos, ei)
        e = module(h, pos, ei, vec, length, data={"z": z, "total_charge": charge})[0]
        (g,) = torch.autograd.grad(e, pos)
        E.append(float(e))
        F.append(abs(float(g[1, 2])))
    worst = max((abs(E[k + 1] - E[k]) / (1.5 * max(F[k], F[k + 1]) * 0.01 + 1e-9), grid[k]) for k in range(180))
    assert worst[0] <= 1.0, worst


def test_the_tabulated_hop_is_given_to_s_moves_only():
    """The s-s hop is the coupling of moves in which an s electron leaves one gold atom (charge -1 or 0 before) and
    enters the other's s shell (charge 0 or +1 before).  Every other move of the pair (Au+ Au0 <-> Au2+ Au-: a 5d
    electron leaves) keeps the learned head's coupling.  Until 2026-11-14 the table's hop sat on the atom pair and all
    16 moves carried it (the review's finding; a strict xfail recorded it; SPEC 40.6 ix)."""
    from trace_redox.diabatic import configuration_importance, diabatic_hamiltonian, switched_hamiltonian
    from trace_redox.two_site import two_site
    ts = two_site()
    pos = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.47]])
    ei = neighbour_list(pos, r_max=5.3)
    _, length = edges(pos, ei)
    z = torch.tensor([79, 79])
    learned = torch.full((2, 2), 0.37, dtype=torch.float64) - 0.37 * torch.eye(2, dtype=torch.float64)
    T, steps = ts.resolve_hops(learned, z, ei, length, 4.0)
    t = float(ts.hop(79, 79, _tensor([2.47]))[0])
    assert abs(float(T[0, 1]) + t) <= 64 * EPS * t and torch.equal(steps.other, learned) and bool(steps.mask[0, 1])
    configs = torch.tensor([[1, 0], [2, -1], [0, 1], [-1, 2], [1, 1], [2, 0]])[:4]
    K = torch.tensor([3, 3])
    E = torch.zeros(4, dtype=torch.float64)
    for H in (diabatic_hamiltonian(E, configs, K, T, electrons=z, steps=steps),
              switched_hamiltonian(E, configs, K, T, 1e-6, 2e-6, 0.1, electrons=z, steps=steps)):
        assert abs(abs(float(H[0, 2])) - t) <= 64 * EPS * t                   # the s move: Au+Au0 <-> Au0Au+
        assert abs(abs(float(H[0, 1])) - 0.37) <= 64 * EPS                    # the d move: the learned coupling
        assert abs(abs(float(H[2, 3])) - 0.37) <= 64 * EPS and torch.equal(H, H.T)
    # the neutral and the anion sectors are s moves throughout
    configs = torch.tensor([[0, 0], [1, -1], [-1, 1]])
    H = diabatic_hamiltonian(torch.zeros(3, dtype=torch.float64), configs, K, T, electrons=z, steps=steps)
    assert abs(abs(float(H[0, 1])) - math.sqrt(2.0) * t) <= 64 * EPS * 2 * t and abs(abs(float(H[0, 2])) - math.sqrt(2.0) * t) <= 64 * EPS * 2 * t
    # without the step record every move carries the pair's entry (the earlier behaviour, kept for a learned coupling)
    H = diabatic_hamiltonian(E, torch.tensor([[1, 0], [2, -1], [0, 1], [-1, 2]]), K, T, electrons=z)
    assert abs(abs(float(H[0, 1])) - t) <= 64 * EPS * t
    r, e, _ = configuration_importance(E, torch.tensor([[1, 0], [2, -1], [0, 1], [-1, 2]]), K, T, 0.1, electrons=z, steps=steps)
    assert bool(torch.isfinite(r).all()) and bool(torch.isfinite(e).all())


@pytest.mark.parametrize("R", [2.0, 2.47])
def test_the_modules_d_moves_keep_the_learned_coupling(R):
    """The module itself, learned parts ON, with a learned Au-Au coupling made large (-0.7 eV added) so that the
    selection holds Au2+ Au- and Au- Au2+ (reached from Au+ Au0 by a d move): in the global space the element of that
    move is the learned coupling, the element of the s move Au+ Au0 <-> Au0 Au+ is the table's t(R); and the cluster
    space, whose Hamiltonian is built by other code, gives the same energy.  (The second review's mutants: either
    route's Hamiltonian without the step record passes every other test and moves this energy by 0.27-0.64 eV.)"""
    from trace_redox.two_site import two_site
    t = float(two_site().hop(79, 79, _tensor([R]))[0])
    out = {}
    for space in ("global", "clusters"):
        module, heads = _module(**dict(ON, diabatic_space=space))
        seen = {}
        forward = module.coupling.forward

        def stronger(*a, _forward=forward, **k):
            T = _forward(*a, **k)
            T = T - 0.7 * (1.0 - torch.eye(int(T.shape[0]), dtype=T.dtype, device=T.device))
            seen["T"] = T.detach().clone()
            return T

        module.coupling.forward = stronger
        z = torch.tensor([79, 79])
        pos = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, R]])
        ei = neighbour_list(pos, r_max=5.3)
        vec, length = edges(pos, ei)
        h = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        E = float(module(h, pos, ei, vec, length, data={"z": z, "total_charge": 1.0})[0])
        last = module._ensemble_last
        out[space] = E
        if space == "global":
            rows = {tuple(v): k for k, v in enumerate(last["configs"][:, :2].tolist())}
            assert {(1, 0), (0, 1), (2, -1), (-1, 2)} <= set(rows), sorted(rows)
            H = last["H"].detach()
            learned = abs(float(seen["T"][0, 1]))
            assert learned > 0.5
            d_move = abs(float(H[rows[(1, 0)], rows[(2, -1)]]))
            if R == 2.0:                     # Au2+ Au- lies 1.9 eV up: the amplitude switch of SPEC 21 is whole (measured)
                assert abs(d_move - learned) <= 256 * EPS * (1.0 + learned), (d_move, learned, t)
            else:                            # 3.4 eV up: the switch is 0.76 (measured); never more than the coupling
                assert 0.1 < d_move <= learned * (1.0 + 256 * EPS), (d_move, learned, t)
            assert abs(abs(float(H[rows[(1, 0)], rows[(0, 1)]])) - t) <= 256 * EPS * (1.0 + t)
            assert abs(d_move - t) > 0.5
    assert abs(out["global"] - out["clusters"]) <= 1e-9 * (1.0 + abs(out["global"])), out


def test_the_step_record_leaves_a_band_unit_alone():
    """Units after the atoms (the band unit of SPEC 20 / 38) are not in the step record: every coupling to and from
    them is the matrix's, whatever the charges, while a d move of the masked pair takes the learned coupling.  (The
    second review's mutant that masks the band unit passed every other test and moved a 'phi1' cell by 0.16 eV.)"""
    from trace_redox.diabatic import StepCouplings, one_electron_moves, resolve_matrix, resolve_moves
    n = 2
    T = _tensor([[0.0, -1.25, 0.31], [-1.25, 0.0, 0.47], [0.31, 0.47, 0.0]])          # two gold atoms and a band unit
    learned = _tensor([[0.0, 0.05], [0.05, 0.0]])
    steps = StepCouplings(mask=torch.tensor([[False, True], [True, False]]), other=learned)
    for m in ([1, 0, -1], [2, -1, 0], [1, 0, -2], [-1, 2, 0], [0, 1, -1], [2, 0, -2]):
        M = resolve_matrix(T, steps, torch.tensor(m))
        assert torch.equal(M[n, :], T[n, :]) and torch.equal(M[:, n], T[:, n]), m
    configs = torch.tensor([[1, 0, -1], [2, -1, -1], [0, 1, -1], [1, 1, -2], [2, 0, -2], [0, 0, 0]])
    rows, cols, donor, acceptor, sign = one_electron_moves(configs, torch.tensor([3, 3, 2]))
    t = resolve_moves(T[donor, acceptor], T, steps, configs, rows, donor, acceptor)
    band = (donor == n) | (acceptor == n)
    assert int(band.sum()) > 0 and torch.equal(t[band], T[donor, acceptor][band])
    qd, qa = configs[rows, donor], configs[rows, acceptor]
    d_move = ~band & ~((qd >= -1) & (qd <= 0) & (qa >= 0) & (qa <= 1))
    assert int(d_move.sum()) > 0 and torch.equal(t[d_move], learned[donor[d_move], acceptor[d_move]])


def test_the_importance_of_a_move_uses_its_resolved_coupling():
    """SPEC 21's pair importance e_ij = max(r_i g(t, x_j), r_j g(t, x_i)), with t the coupling the move actually has:
    for a d move of the pair (Au+ Au0 -> Au2+ Au-) the learned one (0.05 eV here), not the table's s-s hop (1.25 eV).
    Checked against the formula on the function's own r, for every move of a four-configuration cation sector."""
    from trace_redox.diabatic import configuration_importance, coupling_amplitude, spin_pairing_factors
    from trace_redox.two_site import two_site
    ts = two_site()
    pos = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.47]])
    ei = neighbour_list(pos, r_max=5.3)
    _, length = edges(pos, ei)
    z = torch.tensor([79, 79])
    learned = 0.05 * (1.0 - torch.eye(2, dtype=torch.float64))
    T, steps = ts.resolve_hops(learned, z, ei, length, 4.0)
    configs = torch.tensor([[1, 0], [0, 1], [2, -1], [-1, 2]])
    E = _tensor([0.0, 0.0, 1.5, 1.5])
    kT = 0.1
    r, e, (rows, cols, donor, acceptor, sign) = configuration_importance(E, configs, torch.tensor([3, 3]), T, kT, electrons=z, steps=steps)
    x = E + kT * torch.logsumexp(-E / kT, 0)
    f = spin_pairing_factors(configs, rows, donor, acceptor, z)
    t = float(ts.hop(79, 79, _tensor([2.47]))[0])
    d_moves = 0
    for k in range(int(rows.numel())):
        qd, qa = int(configs[rows[k], donor[k]]), int(configs[rows[k], acceptor[k]])
        s_move = -1 <= qd <= 0 and 0 <= qa <= 1
        coupling = (-t if s_move else 0.05) * float(f[k])
        g = lambda xx: float(coupling_amplitude(torch.tensor([coupling], dtype=torch.float64), torch.tensor([float(xx)], dtype=torch.float64))[0])
        want = max(float(r[rows[k]]) * g(x[cols[k]]), float(r[cols[k]]) * g(x[rows[k]]))
        assert abs(float(e[k]) - want) <= 64 * EPS * (1.0 + want), (k, qd, qa, float(e[k]), want)
        d_moves += int(not s_move)
    assert d_moves >= 2


def test_two_site_is_refused_with_the_downfolded_transfer():
    """The fold of SPEC 32 builds its own couplings, which are not resolved by charge step: the combination is refused.
    (With the reference's Fermi band the fold is refused for another reason, SPEC 38; the 'phi1' band and the global
    space are a set the fold accepts without two_site, so the refusal tested here is two_site's own.)"""
    from trace_redox import RedoxConfig
    legal = dict(REFERENCE, downfold_transfer=True, diabatic_space="global", band_form="phi1", frame_pair=True, spin_pairing=True)
    assert RedoxConfig(**legal).downfold_transfer is True
    with pytest.raises(ValueError, match="two_site=True is not built with downfold_transfer"):
        RedoxConfig(**dict(legal, two_site=True))


@pytest.mark.parametrize("field", ["E_gs", "E_T", "t", "EA", "IP"])
def test_entries_solved_with_another_value_of_any_input_are_refused(field, tmp_path):
    """Each input of the build, changed in the file after the build, is caught when the file is read: the ground-state
    energy, the triplet excitation, the hop (through the singlet root it is computed from), and the carriers' EA*, IP*."""
    import json

    from trace_redox.two_site import DATA_TWO_SITE, TwoSite
    with open(DATA_TWO_SITE) as fh:
        data = json.load(fh)
    p = data["pairs"]["79-79"]
    k = p["R"].index(2.47)
    step = 1e-8                                                       # ten times the check's tolerance
    if field in ("E_gs", "E_T"):
        p[field][k] += step
        p["E_cov"][k] = p["E_gs"][k] + p["E_T"][k]
    elif field == "t":
        p["t"][k] += step
    else:
        p["carriers"][field][p["carriers"]["R"].index(2.47)] += step
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="another table"):
        TwoSite(str(path))


@pytest.mark.parametrize("change", ["E_gs missing at a knot", "carriers moved", "data file changed"])
def test_the_loader_sees_a_missing_value_moved_carriers_and_changed_data(change, tmp_path, monkeypatch):
    """Three ways a table could change under its entries that the first check missed (second review): a missing
    (null) E_gs at one of the entries' distances, the carriers' EA*, IP* moved to other distances with the same
    values, and a data file the pair's fixed terms read (here ion_polarizabilities.json, whose alpha(Au+) moved the
    three conditions by 0.016-0.023 eV when scaled by 1.5) whose content is not the one the entries were built on."""
    import json

    import trace_redox.two_site as two_site_file
    from trace_redox.two_site import DATA_TWO_SITE, TwoSite
    with open(DATA_TWO_SITE) as fh:
        data = json.load(fh)
    p = data["pairs"]["79-79"]
    if change == "E_gs missing at a knot":
        p["E_gs"][p["R"].index(3.0)] = None
    elif change == "carriers moved":
        p["carriers"]["R"] = [r + 0.01 for r in p["carriers"]["R"]]
    else:
        source = two_site_file.FIXED_TERM_DATA["ion_polarizabilities.json"]
        copy = tmp_path / "ion_polarizabilities.json"
        text = open(source).read()
        copy.write_text(text.replace('"method"', '"method_"', 1) if '"method"' in text else text + " ")
        monkeypatch.setitem(two_site_file.FIXED_TERM_DATA, "ion_polarizabilities.json", str(copy))
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="another table|data file"):
        TwoSite(str(path))


@pytest.mark.parametrize("missing", ["solved_with", "gaussian_a"])
def test_entries_without_their_record_are_refused(missing, tmp_path):
    """Entries are only usable with the record of what they were solved with: a block that lacks it (an older file, or
    a hand edit) would otherwise escape every consistency check.  Measured before the fix: with 'solved_with' removed
    and E_gs changed by 0.5 eV the file loaded and ran."""
    import json

    from trace_redox.two_site import DATA_TWO_SITE, TwoSite
    with open(DATA_TWO_SITE) as fh:
        data = json.load(fh)
    entries = data["pairs"]["79-79"]["entries"]
    if missing == "solved_with":
        del entries["solved_with"]
    else:
        del entries["options"]["gaussian_a"]
    path = tmp_path / "unrecorded.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="record"):
        TwoSite(str(path))


@pytest.mark.parametrize("space", ["global", "clusters"])
@pytest.mark.parametrize("R", [2.0, 2.47, 3.0])
def test_the_cation_of_the_isolated_pair_is_the_s_resonance(R, space):
    """Au2+ through the module with no learned part: its ground state is the two-level resonance of Au+Au0 and Au0Au+
    exactly, E = level - t(R), whichever other configurations the selection holds (Au2+Au- lies 1-5 eV up at zero
    shift and is selected; it is reached by a d move and has no tabulated coupling).  Before the hop was resolved by
    charge step that configuration lowered the cation by 0.88 / 0.33 / 0.06 eV at 2.0 / 2.47 / 3.0 A and the cation's
    entry had been solved through it."""
    from trace_redox.two_site import two_site
    E, c, H, module = _dimer(R, charge=1.0, fixed=True, **dict(ON, diabatic_space=space))
    # the selection judges every move by its resolved coupling: a configuration reached only by a d move (Au2+ Au-) has
    # no coupling here and is not selected (with the s-s hop on it, it was, in both spaces: the review's mutants)
    selected = sorted(tuple(v) for v in module._ensemble_last["configs"][:, :2].tolist())
    assert selected == [(0, 1), (1, 0)], (space, R, selected)
    if space == "clusters":
        return
    rows = {tuple(v): k for k, v in enumerate(c.tolist())}
    t = float(two_site().hop(79, 79, _tensor([R]))[0])
    level = float(H[rows[(1, 0)], rows[(1, 0)]])
    assert abs(abs(float(H[rows[(1, 0)], rows[(0, 1)]])) - t) <= 256 * EPS * (1.0 + t)
    for other in ((2, -1), (-1, 2)):
        if other in rows:
            assert float(H[rows[(1, 0)], rows[other]]) == 0.0 and float(H[rows[(0, 1)], rows[other]]) == 0.0
    ground = float(torch.linalg.eigvalsh(H)[0])                         # (the module's energy adds the frame terms)
    assert abs(ground - (level - t)) <= 256 * EPS * (1.0 + abs(level)), (ground, level - t)


@pytest.mark.parametrize("change", [dict(kT=0.05), dict(kT=0.2), dict(selection_amplitude_off=0.025),
                                    dict(selection_amplitude_off=0.1), dict(selection_amplitude_on=0.1),
                                    dict(selection_amplitude_on=0.4)])
def test_the_pairs_fixed_terms_do_not_depend_on_the_selection(change):
    """The three energies of the isolated pair (no learned part) at the last two knots, where the review found them
    moving by up to 0.16 eV with the selection's thresholds and by 1-3 meV with kT (the switch on the Au+Au0 <->
    Au2+Au- coupling): the same to 1e-6 eV for other thresholds and another temperature."""
    from trace_redox.two_site import two_site
    for R in (2.8, 3.0):
        t = float(two_site().hop(79, 79, _tensor([R]))[0])
        # a change of temperature has its own, real effect: the thermal weight of the doublet's upper member, 2 t above
        # the lower (kT exp(-2 t / kT): 3e-5 eV at kT = 0.2 eV and 3.0 A); the thresholds have none
        tol = 1e-6 + (2.0 * change["kT"] * math.exp(-2.0 * t / change["kT"]) if "kT" in change else 0.0)
        for charge in (0.0, -1.0, 1.0):
            a = _dimer(R, charge=charge, fixed=True, **ON)[0]
            b = _dimer(R, charge=charge, fixed=True, **dict(ON, **change))[0]
            assert abs(a - b) <= tol, (R, charge, change, a, b, tol)


def test_the_coupling_cutoff_must_leave_the_points_alone():
    """The window starts 1 A below the coupling's cutoff.  A cutoff that puts it inside a pair's tabulated distances
    would cut the first-principles hop at its own data, and the pair's entries were solved with the whole hop there:
    refused (gold's entries reach 3.0 A, so 3.9 A is refused and the default 4.0 A accepted)."""
    with pytest.raises(ValueError, match="window"):
        _dimer(2.6, coupling_cutoff=3.9, **ON)
    E, _, _, _ = _dimer(2.6, coupling_cutoff=4.0, **ON)
    assert math.isfinite(E)


def test_a_malformed_table_is_refused(tmp_path):
    """The curves assume increasing distances, positive hops and at least two hop points per pair, and the mapping is
    the symmetric one: a file that breaks any of these is refused when it is read."""
    import json

    from trace_redox.two_site import DATA_TWO_SITE, TwoSite
    with open(DATA_TWO_SITE) as fh:
        good = json.load(fh)

    def broken(change):
        data = json.loads(json.dumps(good))
        change(data["pairs"])
        path = tmp_path / "t.json"
        path.write_text(json.dumps(data))
        return str(path)

    def reverse(pairs):
        for key in ("R", "E_gs", "E_T", "E_S", "E_cov", "U", "t", "weight_T", "weight_S"):
            pairs["47-47"][key] = pairs["47-47"][key][::-1]

    def duplicate(pairs):
        pairs["47-47"]["R"][1] = pairs["47-47"]["R"][0]

    def negative(pairs):
        pairs["47-47"]["t"][2] = -0.3

    def single(pairs):
        pairs["47-47"]["t"] = [pairs["47-47"]["t"][0]] + [None] * (len(pairs["47-47"]["t"]) - 1)

    def mixed(pairs):
        pairs["47-79"] = pairs.pop("47-47")

    def entries_out_of_order(pairs):
        pairs["79-79"]["entries"]["R"][1] = pairs["79-79"]["entries"]["R"][0]

    for change in (reverse, duplicate, negative, single, mixed, entries_out_of_order):
        with pytest.raises(ValueError, match="two-site table"):
            TwoSite(broken(change))
    TwoSite(broken(lambda pairs: None))


def test_the_relaxation_follows_the_frame_table_in_use(monkeypatch):
    """rho = Phi - E_cov is computed from SPEC 39's table once and kept; if the process's table object is replaced, it
    is computed again (a kept rho against another Phi would no longer give E_cov)."""
    import trace_redox.frame_pair as frame_file
    from trace_redox.two_site import DATA_TWO_SITE, TwoSite
    ts = TwoSite(DATA_TWO_SITE)
    R = _tensor([2.47])
    before = float(ts.relaxation(79, 79, R)[0])
    real = frame_file.frame_pair()

    class Shifted:
        def pair_parts(self, Za, Zb, R, scaled=True):
            es, pauli = real.pair_parts(Za, Zb, R, scaled=scaled)
            return es, pauli + 1.0

    shifted = Shifted()
    monkeypatch.setattr(frame_file, "frame_pair", lambda *a, **k: shifted)
    after = float(ts.relaxation(79, 79, R)[0])
    assert abs(after - before - 1.0) <= 1e-12 * (1.0 + before), (before, after)


def test_the_module_in_a_periodic_cell_entry_by_entry():
    """The periodic cell of the test above, global space, matched configuration by configuration against the same
    module without the terms: every configuration is shifted by the entry of its gold pair's class, summed over the
    pair and its image along the cell (2.50998 and 4.50555 A): -rho for Au0Au0, -rho_minus for one extra electron on
    either gold, -rho_plus for one hole, +lambda_ionic for Au+Au-, nothing for any other class (Au-Au-, ...); and the
    cluster space gives the same energies to its closure's accuracy.  (The couplings of a crowded cell are switched by
    the amplitude selection, so their size is tested on the pair alone, below.)"""
    from redox_helpers import lattice_graph, periodic_edges
    from trace_redox.two_site import two_site
    ts = two_site()
    z = torch.tensor([79, 79, 8, 12, 12, 8])
    pbc = (True, True, True)
    cell = 7.0 * torch.eye(3, dtype=torch.float64)
    pos = _tensor([[0.0, 0.0, 0.0], [0.2, 0.1, 2.5], [2.2, 0.1, -0.2], [2.3, 2.2, 0.2], [4.4, 0.2, 3.3], [4.5, 2.4, 3.2]])
    R1, R2 = math.sqrt(0.2 ** 2 + 0.1 ** 2 + 2.5 ** 2), math.sqrt(0.2 ** 2 + 0.1 ** 2 + 4.5 ** 2)
    assert R1 < 3.0 and R2 > 4.0
    both = _tensor([R1, R2])
    rho = float(ts.relaxation(79, 79, both).sum())
    rho_minus, rho_plus, lam = (float(v.sum()) for v in ts.entry_values(79, 79, both))

    def entry(qa, qb):
        key = tuple(sorted((int(qa), int(qb))))
        return {(0, 0): -rho, (-1, 0): -rho_minus, (0, 1): -rho_plus, (-1, 1): lam}.get(key, 0.0)

    def run(space, flags):
        module, heads = _module(diabatic_space=space, **flags)
        h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        ei, es = lattice_graph(pos, cell, pbc, shells=1)
        vec, length = periodic_edges(pos, cell, ei, es)
        E = module(h, pos, ei, vec, length, pbc=pbc, cell=cell, edge_shift=es, data={"z": z, "total_charge": 0.0})[0]
        return float(E), module._ensemble_last

    E_off, off = run("global", OFF)
    E_on, on = run("global", ON)
    rows_off = {tuple(c): k for k, c in enumerate(off["configs"].tolist())}
    rows_on = {tuple(c): k for k, c in enumerate(on["configs"].tolist())}
    common = sorted(set(rows_off) & set(rows_on))
    H_off, H_on = off["H"].detach(), on["H"].detach()
    tol = 1e3 * EPS * (1.0 + float(H_off.abs().max()))
    moved = still = 0
    for c in common:
        shift = float(H_on[rows_on[c], rows_on[c]] - H_off[rows_off[c], rows_off[c]])
        want = entry(c[0], c[1])
        assert abs(shift - want) <= tol, (c, shift, want)
        moved += int(abs(want) > 1e-3)
        still += int(want == 0.0)
    assert moved >= 1 and still >= 1, (moved, still, common)
    # the cluster space on this crowded cell is the cluster product's closure, not the global space bit for bit (5.6e-7
    # eV here, with or without the terms); a hop or an entry missing in that space would move the energy by 0.03 eV or more
    for flags, E_global in ((OFF, E_off), (ON, E_on)):
        E_clusters, _ = run("clusters", flags)
        assert abs(E_clusters - E_global) <= 1e-5, (E_clusters, E_global)


def test_the_hop_in_a_periodic_cell_of_the_pair_alone():
    """The gold pair alone in a periodic cell (9 x 9 x 7 A; itself at 2.50998 A and its image along the cell at
    4.50555 A, beyond the coupling's cutoff), both ensemble spaces.  The couplings of the selected ensemble are at
    full strength here (in a crowded cell the amplitude selection switches the unimportant ones, so their size is not
    the table's): the covalent-ionic element is sqrt(2) t(R1), the Au0Au0 entry is lower than without the terms by
    rho(R1) + rho(R2), the ionic ones higher by lambda_ionic(R1) + lambda_ionic(R2), and the energy is the same in
    the cluster space."""
    from redox_helpers import lattice_graph, periodic_edges
    from trace_redox.two_site import two_site
    ts = two_site()
    z = torch.tensor([79, 79])
    pbc = (True, True, True)
    cell = torch.diag(_tensor([9.0, 9.0, 7.0]))
    pos = _tensor([[0.0, 0.0, 0.0], [0.2, 0.1, 2.5]])
    R1, R2 = math.sqrt(0.2 ** 2 + 0.1 ** 2 + 2.5 ** 2), math.sqrt(0.2 ** 2 + 0.1 ** 2 + 4.5 ** 2)
    both = _tensor([R1, R2])

    def run(space, flags):
        module, heads = _module(diabatic_space=space, **flags)
        h = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        ei, es = lattice_graph(pos, cell, pbc, shells=1)
        vec, length = periodic_edges(pos, cell, ei, es)
        E = module(h, pos, ei, vec, length, pbc=pbc, cell=cell, edge_shift=es, data={"z": z, "total_charge": 0.0})[0]
        return float(E), module._ensemble_last

    E_off, off = run("global", OFF)
    E_on, on = run("global", ON)
    rows_off = {tuple(c[:2]): k for k, c in enumerate(off["configs"].tolist())}
    rows_on = {tuple(c[:2]): k for k, c in enumerate(on["configs"].tolist())}
    H_off, H_on = off["H"].detach(), on["H"].detach()
    assert {(0, 0), (1, -1), (-1, 1)} <= set(rows_on) and (0, 0) in rows_off
    t = float(ts.hop(79, 79, _tensor([R1]))[0])
    for ion in ((1, -1), (-1, 1)):
        assert abs(abs(float(H_on[rows_on[(0, 0)], rows_on[ion]])) - math.sqrt(2.0) * t) <= 256 * EPS * (1.0 + t)
    tol = 1e3 * EPS * (1.0 + float(H_off.abs().max()))
    shift = float(H_on[rows_on[(0, 0)], rows_on[(0, 0)]] - H_off[rows_off[(0, 0)], rows_off[(0, 0)]])
    assert abs(shift + float(ts.relaxation(79, 79, both).sum())) <= tol, shift
    lam = float(ts.entry_values(79, 79, both)[2].sum())
    for ion in ((1, -1), (-1, 1)):
        if ion in rows_off:
            got = float(H_on[rows_on[ion], rows_on[ion]] - H_off[rows_off[ion], rows_off[ion]])
            assert abs(got - lam) <= tol, (ion, got, lam)
    E_clusters, _ = run("clusters", ON)
    assert abs(E_clusters - E_on) <= 1e-9 * (1.0 + abs(E_on)), (E_clusters, E_on)
