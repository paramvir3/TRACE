"""SPEC 24: several independent open structures assembled as ONE block-diagonal system.

The claim is exactness, not approximation: for every structure, the energy,
the forces, the charges and the PARAMETER gradient of a training loss that
contains the forces (a second derivative through the implicit solve) equal
those of the structure solved on its own, to the solve tolerance.  The
structures are placed on top of one another in space, so any interaction left
between two of them would show at once.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from conftest import require

R_HEADS = 3.5


def _heads(kernel="gaussian", quadrupoles=False):
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    extra = dict(quadrupole_blocks=("A_Theta", "chi_theta"), allow_truncated_a_theta=True) if quadrupoles else {}
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ElectrostaticHeads(
            r_max=R_HEADS, hardness_margin="legacy" if kernel == "thole" else 1.0, **extra
        )


def _structure(n, seed, dim, split=False):
    """n atoms near the origin; with ``split`` two clusters 8 A apart (two fragments)."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g, dtype=torch.float64) * 1.1
    if split:
        pos[n // 2:, 0] += 8.0
    h = torch.randn(n, dim, generator=g, dtype=torch.float64)
    distance = torch.cdist(pos, pos)
    src, dst = torch.nonzero((distance < R_HEADS) & (distance > 0), as_tuple=True)
    return pos, h, torch.stack([src, dst])


#: (atoms, charge per fragment, two fragments?) -- neutral, cation, anion, and a
#: two-fragment structure whose charge sits on one fragment.
STRUCTURES = [(5, (0.0,), False), (7, (1.0,), False), (4, (-1.0,), False), (8, (0.0, -1.0), True)]


def _fragments(n, charges, split):
    if not split:
        return torch.zeros(n, dtype=torch.long)
    return torch.cat([torch.zeros(n // 2, dtype=torch.long), torch.ones(n - n // 2, dtype=torch.long)])


def _loss_and_grads(heads, energies, forces, params):
    """A training-style loss on energies AND forces, and its parameter gradient."""
    loss = sum((e - 0.3 * k) ** 2 for k, e in enumerate(energies)) + sum((f ** 2).sum() for f in forces)
    return torch.autograd.grad(loss, params, allow_unused=True)


def _solve_one_by_one(heads, config, cases, quadrupoles):
    electrostatic_state = require("energy", "electrostatic_state")
    energies, forces, charges = [], [], []
    for (pos0, h, edge_index), (n, frag_charges, split) in zip(cases, STRUCTURES):
        pos = pos0.clone().requires_grad_(True)
        vec = pos[edge_index[0]] - pos[edge_index[1]]
        state = electrostatic_state(
            h, pos, edge_index, vec, vec.norm(dim=-1), heads, config,
            fragment_id=_fragments(n, frag_charges, split),
            fragment_charges=torch.tensor(frag_charges, dtype=torch.float64),
            quadrupoles=quadrupoles,
        )
        energies.append(state.energy)
        forces.append(-torch.autograd.grad(state.energy, pos, create_graph=True)[0])
        charges.append(state.solve_result.q)
    return energies, forces, charges


def _solve_batched(heads, config, cases, quadrupoles):
    electrostatic_state = require("energy", "electrostatic_state")
    electrostatic_energies = require("energy", "electrostatic_energies")
    offset, frag_offset = 0, 0
    pos_all, h_all, edges, sid, fid, fq, sizes = [], [], [], [], [], [], []
    for s, ((pos0, h, edge_index), (n, frag_charges, split)) in enumerate(zip(cases, STRUCTURES)):
        pos_all.append(pos0)
        h_all.append(h)
        edges.append(edge_index + offset)
        sid.append(torch.full((n,), s, dtype=torch.long))
        fid.append(_fragments(n, frag_charges, split) + frag_offset)
        fq.extend(frag_charges)
        sizes.append(n)
        offset += n
        frag_offset += len(frag_charges)
    pos = torch.cat(pos_all).clone().requires_grad_(True)
    edge_index = torch.cat(edges, dim=1)
    vec = pos[edge_index[0]] - pos[edge_index[1]]
    state = electrostatic_state(
        torch.cat(h_all), pos, edge_index, vec, vec.norm(dim=-1), heads, config,
        fragment_id=torch.cat(fid), fragment_charges=torch.tensor(fq, dtype=torch.float64),
        quadrupoles=quadrupoles, structure_id=torch.cat(sid),
    )
    energies = electrostatic_energies(state)
    grad = -torch.autograd.grad(energies.sum(), pos, create_graph=True)[0]
    return (list(energies), list(torch.split(grad, sizes)),
            list(torch.split(state.solve_result.q, sizes)), state)


@pytest.mark.parametrize("kernel,quadrupoles", [("gaussian", False), ("gaussian", True), ("thole", False)])
def test_block_diagonal_batch_equals_each_structure_alone(kernel, quadrupoles):
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    heads = _heads(kernel, quadrupoles)
    config = ElectrostaticConfig(multipole_kernel=kernel)
    cases = [_structure(n, 10 + k, heads.irreps_in.dim, split) for k, (n, _, split) in enumerate(STRUCTURES)]
    params = [p for p in heads.parameters() if p.requires_grad]

    e1, f1, q1 = _solve_one_by_one(heads, config, cases, quadrupoles)
    g1 = _loss_and_grads(heads, e1, f1, params)
    e2, f2, q2, state = _solve_batched(heads, config, cases, quadrupoles)
    g2 = _loss_and_grads(heads, e2, f2, params)

    for a, b in zip(e1, e2):
        assert abs(float(a - b)) < 1e-10, (float(a), float(b))
    assert abs(float(sum(e2)) - float(state.energy)) < 1e-10      # the split is a partition
    for a, b in zip(f1, f2):
        assert float((a - b).abs().max()) < 1e-8
    for a, b in zip(q1, q2):
        assert float((a - b).abs().max()) < 1e-9
    scale = max(float(g.abs().max()) for g in g1 if g is not None)
    for a, b in zip(g1, g2):
        if a is None:
            assert b is None or float(b.abs().max()) == 0.0
            continue
        assert float((a - b).abs().max()) <= 1e-7 * scale


def test_energy_by_structure_partitions_the_energy_at_any_moments():
    """Not only at the minimum: sum_s U_s(m) = U(m) for arbitrary moments."""
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    heads = _heads()
    cases = [_structure(n, 20 + k, heads.irreps_in.dim, split) for k, (n, _, split) in enumerate(STRUCTURES)]
    *_, state = _solve_batched(heads, ElectrostaticConfig(), cases, False)
    assembly = state.assembly
    g = torch.Generator().manual_seed(3)
    t = torch.randn(assembly.n_bonds, generator=g, dtype=torch.float64)
    mu_hat = torch.randn(assembly.n_atoms, 3, generator=g, dtype=torch.float64)
    rho = state.solve_result.rho.detach()
    parts = assembly.energy_by_structure(t, rho, mu_hat)
    whole = assembly.energy(t, rho, mu_hat)
    assert parts.shape == (len(STRUCTURES),)
    assert abs(float(parts.sum() - whole)) <= 1e-12 * max(1.0, abs(float(whole)))


def _refusal_inputs(heads):
    pos, h, edge_index = _structure(6, 31, heads.irreps_in.dim)
    vec = pos[edge_index[0]] - pos[edge_index[1]]
    return h, pos, edge_index, vec, vec.norm(dim=-1)


@pytest.mark.parametrize(
    "change, match",
    [
        (dict(pbc=(True, True, True), cell=torch.eye(3, dtype=torch.float64) * 12.0), "open-boundary"),
        (dict(structure_id=torch.tensor([0, 0, 0, 1, 1, 1])), "edge joins two structures|fragment spans"),
        (dict(fragment_id=None), "fragment_id and fragment_charges"),
        (dict(external_field=torch.tensor([0.0, 0.0, 0.1], dtype=torch.float64)), "applied field"),
        (dict(structure_id=torch.tensor([0, 0, 0, 2, 2, 2])), "every label"),
        (dict(solver="matrix_free"), "dense-only"),
    ],
)
def test_block_diagonal_assembly_refuses_what_it_does_not_carry(change, match):
    electrostatic_state = require("energy", "electrostatic_state")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    heads = _heads()
    h, pos, edge_index, vec, length = _refusal_inputs(heads)
    kwargs = dict(fragment_id=torch.zeros(6, dtype=torch.long),
                  fragment_charges=torch.zeros(1, dtype=torch.float64),
                  structure_id=torch.zeros(6, dtype=torch.long))
    solver = change.pop("solver", "auto")
    kwargs.update(change)
    if kwargs.get("fragment_id") is None:
        kwargs["fragment_charges"] = None
    with pytest.raises((ValueError, NotImplementedError), match=match):
        electrostatic_state(h, pos, edge_index, vec, length, heads, ElectrostaticConfig(solver=solver), **kwargs)
