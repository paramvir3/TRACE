"""SPEC 35 (2026-11-07; written red before the implementation): the frame-polarizability anchor.  Step 20's bound sector
carries alpha_O = 0.205 and alpha_Mg = 1.916 A^3 — inverted against the first-principles table (O2- 16.44 bohr^3 = 2.437
A^3 in the Watson sphere, Mg2+ 0.499 bohr^3 = 0.074 A^3; data/ion_polarizabilities.json): the eps_inf prior fixed the
sum and nothing fixed the split.  The anchor sets the dipole stiffness of every frame ion with a reliably tabulated
alpha_Z(f) to A_mu = K_E ell^2 / (g alpha_Z) I, with one learned crystal factor g = exp(log_alpha_scale) (init 1), and
leaves untabulated species to the heads.  Applied at the heads' output, before the assembly builds its operator, in
forward and in bound_state alike."""
import math

import pytest
import torch
from redox_helpers import lattice_graph, make_heads, periodic_edges
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

EPS = torch.finfo(torch.float64).eps
BOHR3 = 0.529177210903 ** 3
PBC = (True, True, True)


def mgo_cell(a=4.265):
    from ase.build import bulk
    at = bulk("MgO", "rocksalt", a=a, cubic=True)
    return torch.tensor(at.numbers), torch.tensor(at.positions, dtype=torch.float64), torch.tensor(at.cell.array, dtype=torch.float64)


def test_the_anchored_stiffness_is_the_table_value_over_the_crystal_factor():
    from trace_electrostatics.constants import K_E
    from trace_redox.polarizability_anchor import anchored_params, frame_alpha
    z = torch.tensor([8, 12, 79, 13, 26])                                    # O2-, Mg2+, Au0, Al3+, Fe (untabulated frame)
    a = frame_alpha(z)                                                       # A^3, nan where not tabulated
    assert abs(float(a[0]) - 16.442105873434414 * BOHR3) <= 1e-12 and abs(float(a[1]) - 0.49884410465421714 * BOHR3) <= 1e-12
    assert abs(float(a[2]) - 36.49814652046999 * BOHR3) <= 1e-12 and abs(float(a[3]) - 0.26330617101323067 * BOHR3) <= 1e-12
    assert math.isnan(float(a[4]))
    heads = make_heads(); torch.manual_seed(2)
    h = torch.randn(5, heads.irreps_in.dim, dtype=torch.float64)
    bond_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]); bond_len = torch.full((4,), 2.2, dtype=torch.float64)
    out = heads(h, bond_index, bond_len)
    log_scale = torch.tensor(math.log(0.8), dtype=torch.float64)
    new = anchored_params(out.params, z, log_scale, scope="all")           # SPEC 35 as first built (legacy, kept)
    eye = torch.eye(3, dtype=torch.float64)
    for i in range(4):
        want = K_E * float(out.params.ell[i]) ** 2 / (0.8 * float(a[i])) * eye
        assert torch.allclose(new.A_mu[i], want, rtol=1e-12, atol=0.0)
    assert torch.equal(new.A_mu[4], out.params.A_mu[4])                      # untabulated: the heads' own
    for name in ("chi", "eta", "sigma", "ell", "kappa"):
        assert torch.equal(getattr(new, name), getattr(out.params, name))


def test_the_module_anchors_both_routes_and_the_scale_is_learnable():
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_electrostatics.constants import K_E
    from trace_electrostatics.observables import dielectric_tensor
    from trace_redox import RedoxConfig, RedoxElectrostatics
    from trace_redox.polarizability_anchor import frame_alpha
    assert RedoxConfig().frame_polarizability_anchor is False
    with pytest.raises(ValueError):
        RedoxConfig(frame_polarizability_anchor=True, anchor="atomic")
    z, pos, cell = mgo_cell()
    ei, es = lattice_graph(pos, cell, PBC, r_max=4.0, shells=1)
    vec, L = periodic_edges(pos, cell, ei, es)
    data = {"z": z, "total_charge": 0.0, "pbc": PBC}
    heads = make_heads(); torch.manual_seed(1)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, frame_polarizability_anchor=True))
    assert isinstance(module.log_alpha_scale, torch.nn.Parameter) and float(module.log_alpha_scale) == 0.0
    h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    st = module.bound_state(h, pos, ei, vec, L, data=data, cell=cell, edge_shift=es, pbc=PBC)
    p = st.assembly.params; alpha = frame_alpha(z); eye = torch.eye(3, dtype=torch.float64)
    for i in range(len(z)):
        assert torch.allclose(p.A_mu[i], K_E * float(p.ell[i]) ** 2 / float(alpha[i]) * eye, rtol=1e-12, atol=0.0)
    eps0 = dielectric_tensor(st.solve_result, st.assembly).detach()
    # forward uses the same anchored parameters: its bound energy equals the bound state's
    out = module(h, pos, ei, vec, L, data=data, cell=cell, edge_shift=es, pbc=PBC)
    assert abs(float(out[1].energy_bound) - float(st.energy)) <= 1e-9 * (1.0 + abs(float(st.energy)))
    # the crystal factor acts: a larger g raises eps_inf, and the energy is differentiable in it
    with torch.no_grad():
        module.log_alpha_scale.fill_(math.log(1.5))
    st2 = module.bound_state(h, pos, ei, vec, L, data=data, cell=cell, edge_shift=es, pbc=PBC)
    eps2 = dielectric_tensor(st2.solve_result, st2.assembly).detach()
    assert float(torch.diagonal(eps2).mean()) > float(torch.diagonal(eps0).mean())
    pos_g = pos.clone().requires_grad_(True); v2, L2 = periodic_edges(pos_g, cell, ei, es)
    E = module(h, pos_g, ei, v2, L2, data=data, cell=cell, edge_shift=es, pbc=PBC)[0]
    g_pos, g_scale = torch.autograd.grad(E, (pos_g, module.log_alpha_scale), allow_unused=True)
    assert g_scale is not None and bool(torch.isfinite(g_scale)) and bool(torch.isfinite(g_pos).all())


# --------------------------------------------------------------------- SPEC 35.1 (2026-11-12): the factor's scope

def test_the_crystal_factor_acts_on_the_charged_frame_ions_only():
    """SPEC 35.1.  g is the in-crystal reduction of an ION's free polarizability; a neutral frame atom (Au0) is its
    own free atom and keeps the table value.  Measured before the change (benchmarks_phonons README, 2026-11-12): with
    one g on every tabulated species the Au-atom polarizability prior held g at 0.997 against the eps_inf prior, the
    two gradients cancelling to 0.04 %."""
    from trace_electrostatics.constants import K_E
    from trace_redox.polarizability_anchor import anchored_params, frame_alpha
    z = torch.tensor([8, 12, 79, 13, 26])                                    # O2-, Mg2+, Au0, Al3+, Fe (untabulated frame)
    a = frame_alpha(z)
    heads = make_heads(); torch.manual_seed(2)
    h = torch.randn(5, heads.irreps_in.dim, dtype=torch.float64)
    bond_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]); bond_len = torch.full((4,), 2.2, dtype=torch.float64)
    out = heads(h, bond_index, bond_len)
    log_scale = torch.tensor(math.log(0.8), dtype=torch.float64)
    new = anchored_params(out.params, z, log_scale, scope="ions")
    eye = torch.eye(3, dtype=torch.float64)
    for i, g in ((0, 0.8), (1, 0.8), (2, 1.0), (3, 0.8)):                    # Au0: no crystal factor
        want = K_E * float(out.params.ell[i]) ** 2 / (g * float(a[i])) * eye
        assert torch.allclose(new.A_mu[i], want, rtol=1e-12, atol=0.0), i
    assert torch.equal(new.A_mu[4], out.params.A_mu[4])
    assert torch.equal(anchored_params(out.params, z, log_scale).A_mu, new.A_mu)          # 'ions' is the default
    with pytest.raises(ValueError, match="scope"):
        anchored_params(out.params, z, log_scale, scope="anions")


def _atom_alpha(module, heads, z):
    """The bound polarizability (A^3, mean of the diagonal) of one isolated atom of the module."""
    from trace_electrostatics.observables import polarizability
    pos = torch.zeros(1, 3, dtype=torch.float64)
    ei = torch.zeros(2, 0, dtype=torch.long); vec = torch.zeros(0, 3, dtype=torch.float64); L = torch.zeros(0, dtype=torch.float64)
    h = torch.randn(1, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    st = module.bound_state(h, pos, ei, vec, L, data={"z": torch.tensor([z]), "total_charge": 0.0})
    return float(torch.diagonal(polarizability(st.solve_result, st.assembly, route="minimiser", as_volume=True)).mean())


@pytest.mark.parametrize("scope,moves", [("ions", False), ("all", True)])
def test_the_free_au_atom_does_not_hold_the_crystal_factor(scope, moves):
    """The polarizability prior's structure is one Au atom.  Under 'ions' its polarizability is the table's free-atom
    value whatever g is, so that prior exerts no force on g; under 'all' (legacy) it is g times that value.  (The
    derivative is taken by a central difference in log g, and the response on the 'minimiser' route: one atom has no
    bond variable, and the default fixed-k route differentiates with respect to them.  The test first assumed a graph
    and the default route; both assumptions were wrong.)"""
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    from trace_redox.polarizability_anchor import frame_alpha
    heads = make_heads(); torch.manual_seed(1)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), RedoxConfig(
        fragment_ensemble="diabatic", anchor="formal", max_charge=3, frame_polarizability_anchor=True,
        frame_polarizability_scope=scope))
    table = float(frame_alpha(torch.tensor([79]))[0])

    def at(log_g):
        with torch.no_grad():
            module.log_alpha_scale.fill_(log_g)
        return _atom_alpha(module, heads, 79)

    base, step = math.log(0.9), 1e-4
    value, up, down = at(base), at(base + step), at(base - step)
    assert abs(value - (0.9 if moves else 1.0) * table) <= 1e-9 * table, (value, table)
    slope = (up - down) / (2.0 * step)                                    # d alpha / d log g
    if moves:
        assert abs(slope - value) <= 1e-6 * value, (slope, value)          # alpha = g alpha_Z; the difference is O(step^2)
    else:
        assert up == value == down                                         # no dependence at all


def test_the_crystals_eps_inf_still_follows_the_factor_under_ions():
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_electrostatics.observables import dielectric_tensor
    from trace_redox import RedoxConfig, RedoxElectrostatics
    assert RedoxConfig().frame_polarizability_scope == "ions"
    with pytest.raises(ValueError, match="frame_polarizability_scope"):
        RedoxConfig(frame_polarizability_scope="anions")
    z, pos, cell = mgo_cell()
    ei, es = lattice_graph(pos, cell, PBC, r_max=4.0, shells=1)
    vec, L = periodic_edges(pos, cell, ei, es)
    data = {"z": z, "total_charge": 0.0, "pbc": PBC}
    heads = make_heads(); torch.manual_seed(1)
    h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    eps = {}
    for scope in ("ions", "all"):                         # every atom of MgO is a charged frame ion: the two agree
        module = RedoxElectrostatics(heads, ElectrostaticConfig(), RedoxConfig(
            fragment_ensemble="diabatic", anchor="formal", max_charge=3, frame_polarizability_anchor=True,
            frame_polarizability_scope=scope))
        for g in (1.0, 0.9):
            with torch.no_grad():
                module.log_alpha_scale.fill_(math.log(g))
            st = module.bound_state(h, pos, ei, vec, L, data=data, cell=cell, edge_shift=es, pbc=PBC)
            eps[(scope, g)] = float(torch.diagonal(dielectric_tensor(st.solve_result, st.assembly)).mean())
    assert eps[("ions", 0.9)] < eps[("ions", 1.0)]
    for g in (1.0, 0.9):
        assert abs(eps[("ions", g)] - eps[("all", g)]) <= 1e-12 * eps[("all", g)]


def test_a_checkpoint_from_before_the_scope_keeps_the_statement_it_trained():
    """28b and the PBEsol phonon models trained g on every tabulated species.  Their saved configs have no scope
    field and load as 'all', with a warning; a config that states the scope is taken as written."""
    import dataclasses
    from trace_redox import RedoxConfig
    from trace_redox.module import statement_changes
    new = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, frame_polarizability_anchor=True)
    saved = {f.name: getattr(new, f.name) for f in dataclasses.fields(new)}
    assert RedoxConfig.from_checkpoint(saved).frame_polarizability_scope == "ions"
    old = dict(saved); del old["frame_polarizability_scope"]
    with pytest.warns(RuntimeWarning, match="frame_polarizability_scope"):
        assert RedoxConfig.from_checkpoint(old).frame_polarizability_scope == "all"
    assert statement_changes(old, new)["frame_polarizability_scope"] == ("all", "ions")
    off = dict(old, frame_polarizability_anchor=False)                      # the anchor off: nothing to warn about
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert RedoxConfig.from_checkpoint(off).frame_polarizability_scope == "all"
