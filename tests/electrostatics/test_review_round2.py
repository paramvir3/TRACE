"""Guards for defects found in the second peer-review round.

Each was verified by the lead reviewer before being fixed; the numbers in the
docstrings are the measured before/after values.
"""

from __future__ import annotations

import dataclasses
import warnings

import pytest
import torch

from conftest import require


def _heads(**kwargs):
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ElectrostaticHeads(**kwargs)


# ---------------------------------------------------------------------------
# 1. Slab reciprocal grid sized by the vacuum axis
# ---------------------------------------------------------------------------
def test_a_slab_is_tuned_per_axis_not_by_its_vacuum():
    """The in-plane grid must not grow with the padding.

    ``auto_tune`` sizes one isotropic k_max off the LONGEST lattice vector,
    which on a padded slab is the vacuum. Measured on an 8 x 8 x Lz cell:
    27,818,127 wave vectors isotropic against 329,967 per-axis at Lz = 80 A --
    84x the work, and 6.7 s -> 38.7 s for a 4-atom slab as Lz went 40 -> 70 A.
    """
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    config = dataclasses.replace(
        ElectrostaticConfig(), ewald_real_cutoff=3.9, ewald_accuracy=1.0e-10
    )
    grids = []
    for lz in (20.0, 40.0, 80.0):
        cell = torch.diag(torch.tensor([8.0, 8.0, lz], dtype=torch.float64))
        grid = config.ewald_params(cell, (True, True, False)).k_grid
        assert isinstance(grid, tuple), (
            f"a slab must get a per-axis grid; got isotropic k_max={grid}"
        )
        grids.append(grid)

    in_plane = {g[:2] for g in grids}
    assert len(in_plane) == 1, (
        f"the in-plane grid changed with the padding: {grids}; only the "
        "out-of-plane half-width may grow with Lz"
    )
    assert grids[-1][2] > grids[0][2], "the out-of-plane grid must grow with Lz"

    bulk = config.ewald_params(
        torch.eye(3, dtype=torch.float64) * 20.0, (True, True, True)
    ).k_grid
    assert isinstance(bulk, int), "a 3-D cell must keep the isotropic grid"


def test_slab_energy_is_padding_independent_through_the_module():
    """The per-axis grid must not cost accuracy: spread <= 1e-9 eV, Lz 40..80."""
    ElectrostaticModule = require("energy", "ElectrostaticModule")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    heads = _heads()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [4.0, 4.0, 0.0], [0.0, 4.0, 2.5], [4.0, 0.0, 2.5]],
        dtype=torch.float64,
    )
    torch.manual_seed(3)
    features = torch.randn(4, heads.irreps_in.dim, dtype=torch.float64) * 2.0
    config = dataclasses.replace(
        ElectrostaticConfig(), ewald_real_cutoff=3.9, ewald_accuracy=1.0e-10
    )
    module = ElectrostaticModule(heads, config)
    empty = torch.zeros(0, 3, dtype=torch.float64)
    data = {
        "fragment_id": torch.arange(4),
        "fragment_charges": [1.0, 1.0, -1.0, -1.0],
    }
    energies = []
    for lz in (40.0, 60.0, 80.0):
        cell = torch.diag(torch.tensor([8.0, 8.0, lz], dtype=torch.float64))
        energy, _ = module(
            features, positions, torch.zeros(2, 0, dtype=torch.long), empty,
            empty.norm(dim=-1), cell=cell, pbc=(True, True, False),
            edge_shift=empty, data=data,
        )
        energies.append(float(energy))
    spread = max(energies) - min(energies)
    assert spread < 1.0e-9, f"slab energy moved {spread:.3e} eV with padding"


# ---------------------------------------------------------------------------
# 2. Fixed-k polarizability went NaN once the CG residual underflowed
# ---------------------------------------------------------------------------
def test_fixed_k_polarizability_stays_finite_at_large_k():
    """alpha = -d2U/dE2 through the unrolled CG, at k far past convergence.

    Before the converged-iteration freeze: exact through k = 150, then NaN at
    k = 250 and 400 on this 6-atom chain (residual 5e-111), with the energy
    still finite. SPEC 2.0's own diagnostic -- rerun at large k -- tripped it.
    """
    electrostatic_state = require("energy", "electrostatic_state")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    polarizability = require("observables", "polarizability")

    heads = _heads()
    n_atoms = 6
    positions = torch.tensor(
        [[1.5 * i, 0.0, 0.0] for i in range(n_atoms)], dtype=torch.float64
    )
    torch.manual_seed(1)
    features = torch.randn(n_atoms, heads.irreps_in.dim, dtype=torch.float64) * 2.0
    pairs = [(i, j) for i in range(n_atoms) for j in range(n_atoms) if i != j]
    edge_index = torch.tensor(pairs, dtype=torch.long).T
    edge_vec = positions[edge_index[0]] - positions[edge_index[1]]

    values = {}
    for k in (50, 400):
        state = electrostatic_state(
            features, positions, edge_index, edge_vec, edge_vec.norm(dim=-1),
            heads, dataclasses.replace(ElectrostaticConfig(solve_mode="fixed_k"), n_cg_iter=k),
        )
        alpha = polarizability(state.solve_result, state.assembly, route="fixed_k")
        assert bool(torch.isfinite(alpha).all()), f"fixed-k alpha is NaN at k={k}"
        values[k] = alpha
        converged = polarizability(state.solve_result, state.assembly, route="minimiser")
    torch.testing.assert_close(values[400], converged, rtol=1e-10, atol=1e-12)


def test_the_cg_freeze_never_triggers_at_the_default_k():
    """At k = 6 the freeze must be inert, so the default solve is unchanged.

    The freeze only substitutes values once ``residual^2 <= eps^2 r0^2``; while
    active, ``torch.where`` selects the original expressions exactly. So the
    default path is bitwise unchanged IF AND ONLY IF the relative residual at
    k = 6 stays above ``eps`` -- which is what is asserted, by a wide margin.
    """
    electrostatic_state = require("energy", "electrostatic_state")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    fixed_k_residual = require("observables", "fixed_k_residual")

    heads = _heads()
    n_atoms = 6
    positions = torch.tensor(
        [[1.5 * i, 0.0, 0.0] for i in range(n_atoms)], dtype=torch.float64
    )
    torch.manual_seed(1)
    features = torch.randn(n_atoms, heads.irreps_in.dim, dtype=torch.float64) * 2.0
    pairs = [(i, j) for i in range(n_atoms) for j in range(n_atoms) if i != j]
    edge_index = torch.tensor(pairs, dtype=torch.long).T
    edge_vec = positions[edge_index[0]] - positions[edge_index[1]]
    state = electrostatic_state(
        features, positions, edge_index, edge_vec, edge_vec.norm(dim=-1),
        heads, ElectrostaticConfig(solve_mode="fixed_k"),
    )
    ratio = float(fixed_k_residual(state.solve_result))
    eps = torch.finfo(torch.float64).eps
    assert ratio > 1.0e6 * eps, (
        f"relative residual at the default k is {ratio:.3e}, within 1e6 of the "
        f"freeze floor eps = {eps:.3e}; the freeze could be altering default runs"
    )


# ---------------------------------------------------------------------------
# 3. The size guard bounded the wrong tensor, and only on one route
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("periodic", [False, True])
def test_the_dense_operator_is_bounded_on_both_routes(periodic):
    """K ([n_bonds + 3N]^2) dominates: 1.45 GB against 0.10 GB of pair blocks
    at 1000 atoms. The old guard checked only the pair blocks, only on the open
    route, and its message advised adding a cell -- the MORE expensive route.
    """
    assemble = require("energy", "assemble")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    heads = _heads()

    side = 4
    grid = torch.stack(
        torch.meshgrid(*[torch.arange(side, dtype=torch.float64)] * 3, indexing="ij"),
        -1,
    ).reshape(-1, 3) * 2.5
    n_atoms = grid.shape[0]
    delta = grid.unsqueeze(1) - grid.unsqueeze(0)
    distance = delta.norm(dim=-1)
    mask = (distance < 5.0) & (distance > 1.0e-9)
    features = torch.randn(n_atoms, heads.irreps_in.dim, dtype=torch.float64)
    config = dataclasses.replace(ElectrostaticConfig(), max_operator_bytes=1.0e5)

    kwargs = {}
    if periodic:
        kwargs = dict(
            cell=torch.eye(3, dtype=torch.float64) * side * 2.5,
            pbc=(True, True, True),
            edge_shift=torch.zeros(int(mask.sum()), 3, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="max_operator_bytes"):
        assemble(
            features, grid, mask.nonzero().T, delta[mask], distance[mask],
            heads, config, **kwargs,
        )


# ---------------------------------------------------------------------------
# 4. The solved periodic energy depended on the choice of simulation cell
# ---------------------------------------------------------------------------
def _rocksalt(repeat, lattice=5.0):
    import itertools
    fractional = torch.tensor(
        [[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
         [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, .5, .5]],
        dtype=torch.float64,
    )
    cells = torch.tensor(
        list(itertools.product(range(repeat), repeat=3)), dtype=torch.float64
    )
    positions = torch.cat([(fractional + c) * lattice for c in cells])
    return positions, torch.eye(3, dtype=torch.float64) * lattice * repeat


def _periodic_edges(positions, cell, cutoff):
    """All image edges within ``cutoff``, in the PACKAGE convention
    ``edge_vec = pos[i] - pos[j] + edge_shift @ cell``."""
    import itertools
    import math
    reach = [int(math.ceil(cutoff / float(x))) for x in torch.diagonal(cell)]
    senders, receivers, shifts = [], [], []
    for image in itertools.product(*[range(-r, r + 1) for r in reach]):
        shift = torch.tensor(image, dtype=torch.float64)
        vec = positions.unsqueeze(1) - positions.unsqueeze(0) + shift @ cell
        length = vec.norm(dim=-1)
        for i, j in ((length < cutoff) & (length > 1.0e-9)).nonzero().tolist():
            senders.append(i)
            receivers.append(j)
            shifts.append(list(image))
    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    edge_shift = torch.tensor(shifts, dtype=torch.float64)
    edge_vec = positions[edge_index[0]] - positions[edge_index[1]] + edge_shift @ cell
    # The check that caught a sign error in the reviewer's own first version of
    # this test: every edge vector must be as short as the list says it is.
    assert float(edge_vec.norm(dim=-1).max()) < cutoff
    return edge_index, edge_shift, edge_vec


def test_the_solved_periodic_energy_is_extensive():
    """E(2x2x2 supercell) == 8 E(primitive), charges identical.

    A pair bonded through several periodic images is several physical
    charge-transfer channels. The energy path built bonds WITHOUT edge_shift,
    so they were keyed by atom pair and merged: an 8-atom rocksalt cell has 24
    nearest-neighbour bonds but 12 distinct pairs, got 12 bond variables, and
    measured E(super) - 8 E(prim) = -5.5e-4 eV (2.6%). topology.py's own
    docstring described the failure and the fix; the seam never passed the
    shift. Shift-resolved: 1.2e-16 eV.
    """
    ElectrostaticModule = require("energy", "ElectrostaticModule")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    heads = _heads()
    torch.manual_seed(1)
    primitive_features = torch.randn(8, heads.irreps_in.dim, dtype=torch.float64) * 2.0
    module = ElectrostaticModule(
        heads,
        dataclasses.replace(
            ElectrostaticConfig(), n_cg_iter=300, ewald_accuracy=1.0e-12
        ),
    )

    results = {}
    for repeat in (1, 2):
        positions, cell = _rocksalt(repeat)
        n_atoms = positions.shape[0]
        edge_index, edge_shift, edge_vec = _periodic_edges(positions, cell, 3.0)
        energy, result = module(
            primitive_features.repeat(repeat ** 3, 1), positions, edge_index,
            edge_vec, edge_vec.norm(dim=-1), cell=cell, pbc=(True, True, True),
            edge_shift=edge_shift,
            data={
                "fragment_id": torch.zeros(n_atoms, dtype=torch.long),
                "fragment_charges": [0.0],
            },
        )
        results[repeat] = (float(energy), result.q.detach(), result.t.shape[0])

    (e1, q1, bonds1), (e2, q2, bonds2) = results[1], results[2]
    assert 8 * bonds1 == bonds2, (
        f"primitive has {bonds1} bond variables (x8 = {8 * bonds1}), supercell "
        f"{bonds2}: bonds to different periodic images are being merged"
    )
    assert abs(e2 - 8 * e1) < 1.0e-10 * abs(e2), (
        f"E(supercell) - 8 E(primitive) = {e2 - 8 * e1:.3e} eV; the energy of a "
        "crystal must not depend on the cell chosen to describe it"
    )
    torch.testing.assert_close(q2, q1.repeat(8), rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. A pinned ewald_alpha got a reciprocal grid sized for a different alpha
# ---------------------------------------------------------------------------
def test_energy_is_independent_of_a_pinned_ewald_alpha():
    """Ewald's defining invariance, with the grid left to the tuner.

    ``ewald_params`` took k_max from ``auto_tune``'s OWN balanced alpha and
    paired it with the caller's pinned one, so k_max stayed 27 for every alpha
    on a 30 A cell and the energy at fixed moments moved by 0.22 eV at
    alpha = 1.6 A^-1. It must not move at all.
    """
    assemble = require("energy", "assemble")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    heads = _heads()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.6, 0.0, 0.0], [0.0, 1.7, 0.0],
         [0.0, 0.0, 1.8], [1.1, 1.1, 1.1]], dtype=torch.float64,
    )
    torch.manual_seed(3)
    features = torch.randn(5, heads.irreps_in.dim, dtype=torch.float64) * 2.0
    torch.manual_seed(9)
    charges = torch.randn(5, dtype=torch.float64)
    charges = charges - charges.mean()
    dipoles = torch.randn(5, 3, dtype=torch.float64) * 0.1
    cell = torch.eye(3, dtype=torch.float64) * 30.0
    empty = torch.zeros(0, 3, dtype=torch.float64)

    energies, grids = [], []
    for alpha in (0.5, 0.8, 1.0, 1.6):
        config = dataclasses.replace(
            ElectrostaticConfig(), ewald_real_cutoff=10.0,
            ewald_accuracy=1.0e-12, ewald_alpha=alpha,
        )
        assembly = assemble(
            features, positions, torch.zeros(2, 0, dtype=torch.long), empty,
            empty.norm(dim=-1), heads, config, cell=cell,
            pbc=(True, True, True), edge_shift=empty,
        )
        energies.append(
            float(assembly.energy(torch.zeros(assembly.n_bonds), charges, dipoles))
        )
        grids.append(int(config.ewald_params(cell, (True, True, True)).k_max))

    assert grids == sorted(grids) and grids[-1] > grids[0], (
        f"k_max did not follow the pinned alpha: {grids}"
    )
    spread = max(energies) - min(energies)
    assert spread < 1.0e-10, (
        f"energy at fixed moments moved {spread:.3e} eV across alpha; an Ewald "
        "sum must not depend on its splitting parameter"
    )


# ---------------------------------------------------------------------------
# 6. Energy and force stepped when a bond left the neighbour list
# ---------------------------------------------------------------------------
def _dimer(r, n_cg_iter, seed=0, list_cutoff=6.0, bond_cutoff=5.0, envelope_onset=0.5,
           solve_mode="converged"):
    ElectrostaticModule = require("energy", "ElectrostaticModule")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heads = require("heads", "ElectrostaticHeads")(
            r_max=bond_cutoff, envelope_onset=envelope_onset
        )
    module = ElectrostaticModule(
        heads,
        dataclasses.replace(ElectrostaticConfig(solve_mode=solve_mode), n_cg_iter=n_cg_iter),
    )
    torch.manual_seed(100 + seed)
    features = torch.randn(2, heads.irreps_in.dim, dtype=torch.float64) * 2.0
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    delta = positions.unsqueeze(1) - positions.unsqueeze(0)
    distance = delta.norm(dim=-1)
    mask = (distance < list_cutoff) & (distance > 1.0e-9)
    energy, _ = module(
        features, positions, mask.nonzero().T, delta[mask], distance[mask],
        data={"fragment_id": torch.tensor([0, 0]), "fragment_charges": [0.0]},
    )
    grad = (
        torch.autograd.grad(energy, positions, allow_unused=True)[0]
        if energy.requires_grad else None
    )
    force = 0.0 if grad is None else float(-grad[1, 0])
    return float(energy), force


@pytest.mark.parametrize("n_cg_iter", [6, 400])
def test_energy_and_force_are_continuous_as_a_bond_dissociates(n_cg_iter):
    """Before: E -1.68e-3 eV -> exactly 0 and F 1.0e-4 eV/A -> 0 across the
    cutoff, identical at k = 6 and 400. With t = f(r) tau the step is ~1e-28,
    and continuity is exact at FINITE k, not only at convergence."""
    for seed in (0, 1, 2):
        inside = _dimer(5.0 - 1.0e-6, n_cg_iter, seed)
        outside = _dimer(5.0 + 1.0e-6, n_cg_iter, seed)
        assert abs(inside[0] - outside[0]) < 1.0e-20, (
            f"energy step {inside[0] - outside[0]:.3e} eV at the bond cutoff "
            f"(seed {seed}, k = {n_cg_iter})"
        )
        assert abs(inside[1] - outside[1]) < 1.0e-15, (
            f"force step {inside[1] - outside[1]:.3e} eV/A at the bond cutoff"
        )


def test_the_dissociating_bond_energy_vanishes_as_the_sixth_power():
    """SPEC 17.4: the C2 quintic shell gives f ~ 10 (1 - x)^3 at the cutoff, so
    E ~ f^2 ~ (r_c - r)^6 -- a factor 1e6 per decade (measured 8.9e5).  The
    legacy full-range cosine gave f ~ (r_c - r)^2 and E ~ (r_c - r)^4."""
    ratio = _dimer(4.9, 6)[0] / _dimer(4.99, 6)[0]
    assert 5.0e5 < ratio < 2.0e6, (
        f"E(4.9)/E(4.99) = {ratio:.3e}; the shell envelope predicts ~1e6"
    )
    legacy = _dimer(4.9, 6, envelope_onset=None, solve_mode="fixed_k")[0] / _dimer(
        4.99, 6, envelope_onset=None, solve_mode="fixed_k"
    )[0]
    assert 5.0e3 < legacy < 2.0e4, (
        f"legacy E(4.9)/E(4.99) = {legacy:.3e}; the cosine envelope predicts ~1e4"
    )


def test_attach_refuses_a_model_cutoff_shorter_than_the_bond_cutoff():
    attach_electrostatics = require("energy", "attach_electrostatics")

    class Model(torch.nn.Module):
        r_max = 4.5

    heads = _heads(r_max=5.0)
    with pytest.raises(ValueError, match="shorter"):
        attach_electrostatics(Model(), heads)
    Model.r_max = 5.0
    attach_electrostatics(Model(), heads)
