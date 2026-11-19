"""SPEC 37: Born-von Karman sampling of the configuration fluctuations (benchmarks_4g README, pre-registered
2026-11-10).  E_M(C) := E_Gamma(S_M[C]) / |M|, S_M[C] the M-fold supercell with the SPEC 36 cluster ensemble: a
cluster's excitation meets its own images at M a instead of a (README R15, R16).  ``bvk_mesh`` gives M,
``bvk_length`` chooses it per structure so that cells of one crystal with the same M a have one energy per cell.

The supercells of these tests are built outside the module with their own neighbour list; the module tiles the cell's
graph.  Both sides then evaluate one structure with the atoms or the edges in another order, so they agree to the
round-off of the reordered sums through the solves: measured <= 2e-13 eV on energies of 1 to 70 eV (2026-11-10, Mac);
TOL below leaves room for a solver that stops at another iterate on another machine."""
import dataclasses

import pytest
import torch

from redox_helpers import cluster, lattice_graph, make_heads, periodic_edges, redox_float64  # noqa: F401

EPS = torch.finfo(torch.float64).eps
TOL = 1e6 * EPS                                    # relative, on energies of 1..100 eV: 2e-10
Z6 = torch.tensor([79, 79, 12, 8, 12, 8])
BULK, SLAB = (True, True, True), (True, True, False)
FORMAL = dict(anchor="formal", max_charge=3)
#: SPEC 36 as first built (legacy): the linked set without SPEC 36.7's product closure.  The charged bulk case keeps it
#: (changed 2026-11-11; README, C2b scored, Q7): its 2 x 1 x 1 supercell is one cluster of 35 linked rows whose closure
#: holds 5184, refused at the default cap of 512 (measured: 47 s at a cap of 8192, the energy per cell 6.086458 eV
#: against 6.312849 eV).  The identities of this file do not depend on the closure; the other three cases run with it,
#: and the refusal is asserted below (test_the_closure_of_the_charged_cell_is_refused_not_truncated).
LINKED = dict(cluster_closure=False)
#: name -> (config, pbc, seed, box / A, total charge).  At (1, 1, 1) the 2 x 1 x 1 supercells of the first two differ
#: from twice the cell by -286 and -202 meV per cell, the formal one by -0.08 meV (measured before the code).
CASES = {
    "bulk": ({}, BULK, 5, 7.0, 0.0),
    "slab": ({}, SLAB, 3, 8.0, 0.0),
    "slab, formal anchors": (FORMAL, SLAB, 5, 7.0, 0.0),
    "bulk, charged cell": (LINKED, BULK, 3, 8.0, 1.0),
}


def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", couplings=True, diabatic_selection="amplitude",
                         diabatic_space="clusters", **redox)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), config)
    with torch.no_grad():                         # couplings up to ~1.2 eV, the Au2 size (init ~0.01 eV)
        module.coupling.pair[-1].weight.mul_(3.0)
    return module, heads


def _case(name):
    redox, pbc, seed, box, charge = CASES[name]
    module, heads = _module(**redox)
    pos, h, _ = cluster(6, heads.irreps_in.dim, seed)
    return module, pos, h, Z6, box * torch.eye(3, dtype=torch.float64), pbc, charge


def _supercell(pos, h, z, cell, reps):
    """Copies in the order of ase's ``Atoms.repeat``: the first index slowest."""
    shifts = [torch.tensor([a, b, c], dtype=pos.dtype) for a in range(reps[0]) for b in range(reps[1])
              for c in range(reps[2])]
    big = torch.tensor(reps, dtype=pos.dtype).unsqueeze(1) * cell
    return torch.cat([pos + s @ cell for s in shifts]), h.repeat(len(shifts), 1), z.repeat(len(shifts)), big


def _with(module, **changes):
    module.redox = dataclasses.replace(module.redox, **changes)
    return module


def _evaluate(module, pos, h, z, cell, pbc, charge):
    ei, es = lattice_graph(pos, cell, pbc, shells=1)
    vec, length = periodic_edges(pos, cell, ei, es)
    return module(h, pos, ei, vec, length, pbc=pbc, cell=cell, edge_shift=es, data={"z": z, "total_charge": charge})


def _copies(mesh):
    return mesh[0] * mesh[1] * mesh[2]


# ------------------------------------------------------------------------------------------------ the tiling

@pytest.mark.parametrize("mesh", [(2, 1, 1), (1, 3, 1), (2, 2, 1), (2, 3, 2)])
def test_the_tiled_graph_is_the_supercells_neighbour_list(mesh):
    """A sheared cell shorter than the cutoff (3.2 A against 3.5 A): atoms bond to images, their own included."""
    from trace_redox.bvk import tile
    g = torch.Generator().manual_seed(2)
    pos = 3.2 * torch.rand(3, 3, generator=g, dtype=torch.float64)
    cell = 3.2 * torch.eye(3, dtype=torch.float64) + 0.2 * torch.randn(3, 3, generator=g, dtype=torch.float64)
    h = torch.randn(3, 4, generator=g, dtype=torch.float64)
    z = torch.tensor([8, 12, 79])
    ei, es = lattice_graph(pos, cell, BULK, shells=2)
    vec, length = periodic_edges(pos, cell, ei, es)
    assert bool((ei[0] == ei[1]).any())                                # self-image edges are present
    t = tile(mesh, h, pos, ei, vec, length, cell, es, z)
    P, H, Z, C = _supercell(pos, h, z, cell, mesh)
    assert t["copies"] == _copies(mesh)
    assert torch.equal(t["species"], Z) and torch.equal(t["h"], H)
    assert float((t["pos"] - P).abs().max()) <= 16 * EPS * float(P.abs().max())
    assert float((t["cell"] - C).abs().max()) <= 16 * EPS * float(C.abs().max())
    ei2, es2 = lattice_graph(P, C, BULK, shells=2)
    mine = {(int(a), int(b)) + tuple(int(x) for x in s) for a, b, s in zip(t["edge_index"][0], t["edge_index"][1],
                                                                             t["edge_shift"])}
    theirs = {(int(a), int(b)) + tuple(int(x) for x in s) for a, b, s in zip(ei2[0], ei2[1], es2)}
    assert len(mine) == int(t["edge_index"].shape[1]) == _copies(mesh) * int(ei.shape[1])
    assert mine == theirs
    vec2, len2 = periodic_edges(t["pos"], t["cell"], t["edge_index"], t["edge_shift"].to(pos.dtype))
    assert float((t["edge_vec"] - vec2).abs().max()) <= 64 * EPS * float(C.abs().max())
    assert float((t["edge_len"] - len2).abs().max()) <= 64 * EPS * float(C.abs().max())


def test_the_mesh_of_a_length():
    from trace_redox.bvk import mesh_for
    cell = torch.diag(torch.tensor([9.0474, 9.0474, 26.4589], dtype=torch.float64))
    assert mesh_for(cell, BULK, (1, 1, 1), 18.0) == (2, 2, 1)
    assert mesh_for(2.0 * cell, BULK, (1, 1, 1), 18.0) == (1, 1, 1)
    assert mesh_for(cell, SLAB, (1, 1, 1), 60.0) == (7, 7, 1)          # nothing along a non-periodic direction
    assert mesh_for(cell, BULK, (2, 3, 1), 0.0) == (2, 3, 1)
    assert mesh_for(None, None, (1, 1, 1), 18.0) == (1, 1, 1)          # a molecule
    assert mesh_for(cell, BULK, (1, 1, 1), 2 * 9.0474) == (2, 2, 1)    # a width that divides L exactly is not doubled
    sheared = cell.clone()
    sheared[1, 0] = 9.0474                                             # the perpendicular width, not the edge length
    assert mesh_for(sheared, BULK, (1, 1, 1), 18.0) == (3, 2, 1)       # widths 9.0474 / sqrt(2) and 9.0474; edges would give (2, 2, 1)
    with pytest.raises(ValueError):
        mesh_for(cell, SLAB, (1, 1, 2), 0.0)                           # a mesh along a non-periodic direction
    with pytest.raises(ValueError):
        mesh_for(None, None, (2, 1, 1), 0.0)


# -------------------------------------------------------------------------------------------- B1, the definition

@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("mesh", [(2, 1, 1), (2, 2, 1)])
def test_the_energy_is_the_supercells_energy_per_cell(name, mesh):
    module, pos, h, z, cell, pbc, charge = _case(name)
    copies = _copies(mesh)
    with torch.no_grad():
        one = float(_evaluate(module, pos, h, z, cell, pbc, charge)[0])
        reference = float(_evaluate(module, *_supercell(pos, h, z, cell, mesh), pbc, copies * charge)[0]) / copies
        sampled = float(_evaluate(_with(module, bvk_mesh=mesh), pos, h, z, cell, pbc, charge)[0])
    assert abs(sampled - reference) <= TOL * (1.0 + abs(reference))
    assert abs(reference - one) > 1e3 * TOL * (1.0 + abs(reference))   # the mesh changes the energy: not vacuous


def test_the_closure_of_the_charged_cell_is_refused_not_truncated():
    """SPEC 36.7's limit, stated in its pre-registration: the closure is exponential in a cluster's disjoint parts and
    is refused beyond the cap, never truncated (a truncated closure is not size consistent)."""
    from trace_redox.diabatic import ConfigurationSearchError
    module, pos, h, z, cell, pbc, charge = _case("bulk, charged cell")
    _with(module, cluster_closure=True)
    mesh = (2, 1, 1)
    with torch.no_grad():
        _evaluate(module, pos, h, z, cell, pbc, charge)                      # the cell itself is inside the cap
        assert [int(S.shape[0]) for S in module._ensemble_last["clusters"]] != []
        with pytest.raises(ConfigurationSearchError, match="product closure"):
            _evaluate(module, *_supercell(pos, h, z, cell, mesh), pbc, _copies(mesh) * charge)
        with pytest.raises(ConfigurationSearchError, match="product closure"):
            _evaluate(_with(module, bvk_mesh=mesh), pos, h, z, cell, pbc, charge)


# ------------------------------------------------------------------------------- B2, commensurate replication

@pytest.mark.parametrize("name", ["bulk", "slab, formal anchors"])
def test_a_replicated_cell_at_a_mesh_is_the_cell_at_the_product_mesh(name):
    module, pos, h, z, cell, pbc, charge = _case(name)
    with torch.no_grad():
        left = float(_evaluate(_with(module, bvk_mesh=(2, 1, 1)), *_supercell(pos, h, z, cell, (1, 2, 1)), pbc,
                               2 * charge)[0])
        right = 2.0 * float(_evaluate(_with(module, bvk_mesh=(2, 2, 1)), pos, h, z, cell, pbc, charge)[0])
    assert abs(left - right) <= TOL * (1.0 + abs(right))


# ---------------------------------------------------------------------------- B3, extensivity at a fixed length

@pytest.mark.parametrize("name", ["bulk", "slab", "slab, formal anchors"])
def test_at_a_fixed_length_the_energy_is_extensive(name):
    """L between one and two cell widths, and above the supercell's width along x only: the cell takes its own
    2 x 2 (x 2) supercell, and the 2 x 1 x 1 supercell the same Born-von Karman cell from its side."""
    module, pos, h, z, cell, pbc, charge = _case(name)
    from trace_redox.bvk import mesh_for
    L = 1.5 * float(cell[0, 0])
    big = _supercell(pos, h, z, cell, (2, 1, 1))
    assert mesh_for(cell, pbc, (1, 1, 1), L) == (2, 2, 2 if pbc[2] else 1)
    assert mesh_for(big[3], pbc, (1, 1, 1), L) == (1, 2, 2 if pbc[2] else 1)
    _with(module, bvk_length=L)
    with torch.no_grad():
        one = float(_evaluate(module, pos, h, z, cell, pbc, charge)[0])
        two = float(_evaluate(module, *big, pbc, 2 * charge)[0])
        gamma = float(_evaluate(_with(module, bvk_length=0.0), *big, pbc, 2 * charge)[0])
        gamma_one = float(_evaluate(module, pos, h, z, cell, pbc, charge)[0])
    assert abs(two - 2.0 * one) <= TOL * (1.0 + abs(two))
    assert abs(gamma - 2.0 * gamma_one) > 1e3 * TOL * (1.0 + abs(gamma))   # at Gamma the same pair is not extensive


# --------------------------------------------------------------------------------------------------- B4, forces

@pytest.mark.parametrize("name", ["bulk", "slab, formal anchors"])
def test_forces_are_minus_the_gradient_through_the_tiling(name):
    module, pos0, h, z, cell, pbc, charge = _case(name)
    mesh = (2, 1, 1)
    _with(module, bvk_mesh=mesh)
    ei, es = lattice_graph(pos0, cell, pbc, shells=1)                  # one graph: no atom crosses the cutoff below

    def energy(p):
        vec, length = periodic_edges(p, cell, ei, es)
        return module(h, p, ei, vec, length, pbc=pbc, cell=cell, edge_shift=es,
                      data={"z": z, "total_charge": charge})[0]

    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    for i, axis in [(0, 0), (3, 2), (5, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-7 * (1.0 + abs(fd))
    # the same forces as the first copy of the supercell evaluated at Gamma (all copies are equivalent)
    _with(module, bvk_mesh=(1, 1, 1))
    P, H, Z, C = _supercell(pos0, h, z, cell, mesh)
    P = P.clone().requires_grad_(True)
    ei2, es2 = lattice_graph(P.detach(), C, pbc, shells=1)
    vec2, len2 = periodic_edges(P, C, ei2, es2)
    e2 = module(H, P, ei2, vec2, len2, pbc=pbc, cell=C, edge_shift=es2, data={"z": Z, "total_charge": 2 * charge})[0]
    (grad2,) = torch.autograd.grad(e2, P)
    n = int(pos0.shape[0])
    assert float((grad2[:n] - grad).abs().max()) <= 1e-9 * (1.0 + float(grad.abs().max()))
    assert float((grad2[n:] - grad2[:n]).abs().max()) <= 1e-9 * (1.0 + float(grad.abs().max()))


# ------------------------------------------------------------------------------------------------- B5, identity

def test_the_unit_mesh_is_the_model_as_shipped():
    module, pos, h, z, cell, pbc, charge = _case("bulk")
    from trace_redox import RedoxConfig
    assert RedoxConfig().bvk_mesh == (1, 1, 1) and RedoxConfig().bvk_length == 0.0
    with torch.no_grad():
        shipped, record = _evaluate(module, pos, h, z, cell, pbc, charge)
        unit = _evaluate(_with(module, bvk_mesh=(1, 1, 1), bvk_length=0.0), pos, h, z, cell, pbc, charge)[0]
        wide = _evaluate(_with(module, bvk_length=0.5 * float(cell[0, 0])), pos, h, z, cell, pbc, charge)[0]
    assert torch.equal(shipped, unit) and torch.equal(shipped, wide)   # L below every width: the cell itself
    assert record.bvk_mesh == (1, 1, 1)


def test_a_molecule_has_no_mesh():
    from redox_helpers import edges
    module, heads = _module()
    pos, h, ei = cluster(6, heads.irreps_in.dim, 3)
    vec, length = edges(pos, ei)
    data = {"z": Z6, "total_charge": 1.0}
    with torch.no_grad():
        shipped = module(h, pos, ei, vec, length, data=data)[0]
        with_length = _with(module, bvk_length=18.0)(h, pos, ei, vec, length, data=data)[0]
    assert torch.equal(shipped, with_length)
    with pytest.raises(ValueError):
        _with(module, bvk_length=0.0, bvk_mesh=(2, 1, 1))(h, pos, ei, vec, length, data=data)


# ----------------------------------------------------------------------------------------------- B6, validation

def test_the_configuration_is_validated():
    from trace_redox import RedoxConfig
    base = dict(fragment_ensemble="diabatic", diabatic_space="clusters")
    assert RedoxConfig(**base, bvk_mesh=[2, 2, 1]).bvk_mesh == (2, 2, 1)          # a YAML list
    assert RedoxConfig.from_checkpoint({"fragment_ensemble": "diabatic"}).bvk_mesh == (1, 1, 1)
    assert RedoxConfig.from_checkpoint({"fragment_ensemble": "diabatic"}).bvk_length == 0.0
    for bad in ((0, 1, 1), (2, 2), (1.5, 1, 1), (-2, 1, 1)):
        with pytest.raises(ValueError):
            RedoxConfig(**base, bvk_mesh=bad)
    with pytest.raises(ValueError):
        RedoxConfig(**base, bvk_length=-1.0)
    with pytest.raises(ValueError):
        RedoxConfig(**base, bvk_mesh=(2, 1, 1), bvk_length=18.0)                  # one or the other
    with pytest.raises(ValueError):
        RedoxConfig(fragment_ensemble="diabatic", bvk_mesh=(2, 1, 1))             # needs the cluster ensemble
    with pytest.raises(ValueError):
        RedoxConfig(bvk_length=18.0)


def test_a_mesh_along_a_non_periodic_direction_is_refused():
    module, pos, h, z, cell, pbc, charge = _case("slab")
    with pytest.raises(ValueError):
        _evaluate(_with(module, bvk_mesh=(1, 1, 2)), pos, h, z, cell, pbc, charge)


# ----------------------------------------------------------------------------------------------- B7, the record

@pytest.mark.parametrize("name", ["bulk", "bulk, charged cell", "slab, formal anchors"])
def test_the_record_is_the_home_cells(name):
    module, pos, h, z, cell, pbc, charge = _case(name)
    mesh = (2, 1, 1)
    n = int(pos.shape[0])
    with torch.no_grad():
        e_big, big = _evaluate(module, *_supercell(pos, h, z, cell, mesh), pbc, 2 * charge)
        energy, record = _evaluate(_with(module, bvk_mesh=mesh), pos, h, z, cell, pbc, charge)
    assert record.bvk_mesh == mesh
    assert torch.equal(record.energy, energy)
    for field in ("q", "q_bound", "free_charge", "sigma_free", "rho"):
        assert tuple(getattr(record, field).shape) == (n,), field
    assert tuple(record.mu.shape) == (n, 3)
    assert abs(float(record.q.sum()) - charge) <= 1e-9
    assert float((record.q - big.q[:n]).abs().max()) <= 1e-9
    assert float((big.q[n:] - big.q[:n]).abs().max()) <= 1e-9          # the copies are equivalent
    assert float((record.mu - big.mu[:n]).abs().max()) <= 1e-9
    for field in ("energy", "energy_bound", "energy_free"):            # per cell
        assert abs(float(getattr(record, field)) - 0.5 * float(getattr(big, field))) <= TOL * (1.0 + abs(float(e_big)))


# ------------------------------------------------------------------------------------- the band unit (README L3)

def test_an_occupied_band_unit_is_refused():
    """The band unit takes at most two electrons per STRUCTURE (SPEC 20; README R16, L3), so a supercell is not a
    replication of a cell whose band is occupied.  Rock-salt MgO with these random heads holds 1.7 e in it at Gamma."""
    from test_redox_penetration import _rocksalt
    z, pos, cell = _rocksalt()
    pos = pos + 0.05 * torch.randn(8, 3, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    module, heads = _module(**FORMAL, carrier_band=True)
    h = torch.randn(8, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    with torch.no_grad():
        record = _evaluate(module, pos, h, z, cell, BULK, 0.0)[1]
        assert abs(float(record.unit_charge[-1])) > 1.0
        with pytest.raises(NotImplementedError, match="band unit"):
            _evaluate(_with(module, bvk_mesh=(2, 1, 1)), pos, h, z, cell, BULK, 0.0)
