"""SPEC 36.10: the response record of the cluster product ensemble (SPEC 36's limit L5, lifted 2026-11-12).

The cluster product's free energy is linear in the free energies of its terms,

    F = E0 + sum_c (F_c - E0) + sum_{c<c'} (F_cc' - F_c - F_c' + E0),

and in a uniform field every configuration's energy is E(n; F) = E(n; 0) - F . M(n) - 1/2 F^T [alpha_b + dalpha(n)] F
(SPEC 30, 28).  So the ensemble's polarisability is the same combination of the terms' polarisabilities,

    alpha = alpha_b + sum_X w_X alpha_ens(H_X, M_X, dalpha_X),   w_0 = 1 - n_c + n_p,  w_c = 2 - n_c,  w_cc' = 1,

exactly (sum_X w_X = 1).  Before this the record of several clusters was refused, so a response prior stopped the
training whenever its structure reached two clusters (benchmarks_4g README R26: step 31's s42 on the eps prior cell).

Tests: the record against the whole model's finite-field response of the total energy (open boundaries, where the
field is unambiguous), with a pair increment that matters and with SPEC 28; one cluster keeps SPEC 30's record.
"""
import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)


def _module(scale=0.3, **redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=1, couplings=True,
                         diabatic_space="clusters", **redox)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), config)
    module.record_ensemble = True                           # opt-in (train.py's response priors set it)
    with torch.no_grad():
        module.coupling.pair[-1].weight.normal_(0.0, scale, generator=torch.Generator().manual_seed(4))
    return module, heads


def _two_dimers(z, bond, shift):
    """Two dimers of element ``z``; the second is the first moved by ``shift`` (beyond the 4 A coupling cutoff)."""
    one = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.2, bond]], dtype=torch.float64)
    pos = torch.cat([one, one + torch.tensor(shift, dtype=torch.float64)])
    return torch.tensor([z] * 4), pos


def _runner(module, heads, z, pos):
    h = torch.randn(int(pos.shape[0]), heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)

    def run(field):
        return module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0, "external_field": field})
    return run


def _finite_field(run, energy0, axis):
    def fd(h):
        e = torch.zeros(3, dtype=torch.float64)
        e[axis] = h
        return -(float(run(e)[0]) - 2.0 * float(energy0) + float(run(-e)[0])) / h ** 2
    return (4.0 * fd(5e-4) - fd(1e-3)) / 3.0


@pytest.mark.parametrize("redox,scale,z,bond,shift", [
    ({}, 0.3, 11, 3.0, (6.5, 0.4, 0.3)),
    ({"diabatic_selection": "amplitude"}, 3.0, 11, 3.0, (6.5, 0.4, 0.3)),
    ({}, 0.3, 79, 2.5, (6.0, 0.5, 0.2)),
    ({"diabatic_selection": "amplitude"}, 3.0, 79, 2.5, (6.0, 0.5, 0.2)),
    ({"state_polarizability": True}, 0.3, 79, 2.5, (6.0, 0.5, 0.2)),
], ids=["window", "amplitude-saturated", "couplings", "couplings, amplitude-saturated", "state polarizabilities"])
def test_the_record_of_several_clusters_is_the_whole_models_finite_field_response(redox, scale, z, bond, shift):
    from trace_redox.response import record_polarizability
    module, heads = _module(scale=scale, **redox)
    species, pos = _two_dimers(z, bond, shift)
    run = _runner(module, heads, species, pos)
    energy0, result = run(torch.zeros(3, dtype=torch.float64))
    last = module._ensemble_last
    assert len(last["clusters"]) >= 2 and min(int(S.shape[0]) for S in last["clusters"]) >= 2      # the case of L5
    ens = result.ensemble
    assert ens is not None and ens["exact"], ens.get("reason")
    assert ens["H"] is None and len(ens["terms"]) == 1 + len(last["clusters"]) + len(last["pair_sizes"])
    assert abs(sum(float(t["weight"]) for t in ens["terms"]) - 1.0) < 1e-12
    coupled = max(float((t["H"] - torch.diag(torch.diagonal(t["H"]))).detach().abs().max()) for t in ens["terms"])
    # one-electron couplings in H (the coherent response) in every case but Na2 in the window, whose two configurations
    # differ by two electrons (measured; the amplitude selection brings in the configurations between them)
    assert (coupled > 1e-3) == (not (z == 11 and not redox))
    alpha = ens["alpha_bound"] + record_polarizability(ens)
    singles = ens["alpha_bound"] + record_polarizability(dict(ens, terms=[t for t in ens["terms"] if t["kind"] != "pair"]))
    worst = 0.0
    for axis in range(3):
        reference = _finite_field(run, energy0, axis)
        assert abs(float(alpha[axis, axis]) - reference) <= 1e-5 * (1.0 + abs(reference)), (axis, float(alpha[axis, axis]), reference)
        worst = max(worst, abs(float(singles[axis, axis]) - reference) / (1.0 + abs(reference)))
    assert worst > 1e-5                                    # counterfactual: the pair terms are needed here


def test_one_cluster_keeps_the_single_hamiltonian_record():
    from trace_redox.response import ensemble_polarizability, record_polarizability
    module, heads = _module()
    species, pos = torch.tensor([11, 11]), torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.2, 3.0]], dtype=torch.float64)
    run = _runner(module, heads, species, pos)
    energy0, result = run(torch.zeros(3, dtype=torch.float64))
    ens = result.ensemble
    assert len(module._ensemble_last["clusters"]) == 1 and ens["exact"] and ens["H"] is not None and "terms" not in ens
    alpha = record_polarizability(ens)
    assert torch.equal(alpha, ensemble_polarizability(ens["H"], ens["M"], ens["kT"], quadratic=ens.get("alpha_shift")))
    total = ens["alpha_bound"] + alpha
    for axis in range(3):
        reference = _finite_field(run, energy0, axis)
        assert abs(float(total[axis, axis]) - reference) <= 1e-5 * (1.0 + abs(reference))


def test_the_record_carries_the_graph():
    """The priors train through it: the ensemble part of the record's polarisability is differentiable in the model's
    parameters.  (Not in the couplings here: the two selected configurations of a dimer differ by two electrons, so no
    coupling enters its Hamiltonian; first written against the couplings, which was this test's error.)"""
    from trace_redox.response import record_polarizability
    module, heads = _module()
    species, pos = _two_dimers(11, 3.0, (6.5, 0.4, 0.3))
    _, result = _runner(module, heads, species, pos)(torch.zeros(3, dtype=torch.float64))
    assert result.ensemble.get("terms") is not None
    alpha_ens = record_polarizability(result.ensemble)
    parameters = [p for p in module.parameters() if p.requires_grad]
    grads = torch.autograd.grad(alpha_ens[2, 2], parameters, allow_unused=True)
    assert sum(g is not None and float(g.abs().max()) > 0.0 for g in grads) > 0


# ------------------------------------------------------------------------------------------- at a mesh (SPEC 37)

def _periodic_case():
    from redox_helpers import cluster, lattice_graph, periodic_edges  # noqa: F401
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", couplings=True, diabatic_selection="amplitude",
                         diabatic_space="clusters", anchor="formal", max_charge=3)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), config)
    module.record_ensemble = True
    with torch.no_grad():
        module.coupling.pair[-1].weight.mul_(3.0)
    pos, h, _ = cluster(6, heads.irreps_in.dim, 5)
    return module, pos, h, torch.tensor([79, 79, 12, 8, 12, 8]), 7.0 * torch.eye(3, dtype=torch.float64)


def _periodic_run(module, pos, h, z, cell, pbc=(True, True, False)):
    from redox_helpers import lattice_graph, periodic_edges
    ei, es = lattice_graph(pos, cell, pbc, shells=1)
    vec, length = periodic_edges(pos, cell, ei, es)
    return module(h, pos, ei, vec, length, pbc=pbc, cell=cell, edge_shift=es, data={"z": z, "total_charge": 0.0})


def test_at_a_mesh_the_record_is_per_cell():
    """SPEC 37 evaluates the supercell; its ensemble response is extensive.  The record a cell returns at a mesh gives
    the polarisability per cell: the supercell's own record at the unit mesh, divided by the number of copies."""
    import dataclasses
    from trace_redox.response import record_polarizability
    module, pos, h, z, cell = _periodic_case()
    mesh, copies = (2, 1, 1), 2
    module.redox = dataclasses.replace(module.redox, bvk_mesh=mesh)
    _, at_mesh = _periodic_run(module, pos, h, z, cell)
    assert at_mesh.bvk_mesh == mesh and at_mesh.ensemble is not None and at_mesh.ensemble["exact"], at_mesh.ensemble
    module.redox = dataclasses.replace(module.redox, bvk_mesh=(1, 1, 1))
    big_cell = torch.tensor(mesh, dtype=pos.dtype).unsqueeze(1) * cell
    big_pos = torch.cat([pos, pos + cell[0]])
    _, whole = _periodic_run(module, big_pos, h.repeat(copies, 1), z.repeat(copies), big_cell)
    assert whole.ensemble is not None and whole.ensemble["exact"]
    per_cell = at_mesh.ensemble["alpha_bound"] + record_polarizability(at_mesh.ensemble)
    supercell = whole.ensemble["alpha_bound"] + record_polarizability(whole.ensemble)
    # ensemble_polarizability takes second differences of F at steps h = 1e-3 and h / 2 (V / A): round-off of the
    # Hamiltonians' entries, eps |H|, is amplified by 1 / (h / 2)^2.  The two routes build the same supercell from a
    # tiled and from an explicit neighbour list, so their H agree to round-off only.  (Measured: 1.9e-7 at |H| = 70 eV;
    # a first tolerance of 1e-9 relative was not derived from the operation and failed.)
    eps = torch.finfo(torch.float64).eps
    h_scale = max(float(t["H"].detach().abs().max()) for t in whole.ensemble.get("terms", [{"H": whole.ensemble["H"]}])
                  if t["H"] is not None)
    tolerance = 16.0 * eps * h_scale / (0.5e-3) ** 2
    assert float((per_cell - supercell / copies).abs().max()) <= tolerance, (per_cell, supercell / copies, tolerance)
    ensemble_part = float(record_polarizability(whole.ensemble).abs().max())
    assert ensemble_part > 1e3 * tolerance, (ensemble_part, tolerance)                    # the ensemble part is exercised
