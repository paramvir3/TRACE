"""Bond topology and fragment constraints -- SPEC section 1.3.

The load-bearing property is structural, not numerical:

    q_i = sum_{j in N(i)} t_ij + rho_i ,   t_ij = -t_ji   =>   1^T B == 0 exactly

so the bond variables can never create or destroy charge, whatever the solver
does with them.  "Exactly" here means bitwise, not to a tolerance.
"""

import pytest
import torch

from conftest import require


TOPOLOGIES = [
    ("single bond", torch.tensor([[0], [1]]), 2),
    ("water", torch.tensor([[0, 0], [1, 2]]), 3),
    ("chain of 6", torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]), 6),
    ("ring of 5", torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]), 5),
    ("both directions listed", torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), 3),
    ("duplicated edges", torch.tensor([[0, 0, 0, 1], [1, 1, 1, 2]]), 3),
    ("two disconnected fragments", torch.tensor([[0, 3], [1, 4]]), 6),
    ("isolated atoms only", torch.zeros(2, 0, dtype=torch.long), 4),
]


@pytest.mark.parametrize("name,edge_index,num_atoms", TOPOLOGIES)
def test_incidence_columns_sum_to_zero_exactly(name, edge_index, num_atoms):
    """SPEC 1.3: "Antisymmetry is structural, so ``1^T B == 0`` exactly"."""
    build_incidence = require("topology", "build_incidence")
    incidence, _ = build_incidence(edge_index, num_atoms)
    column_sums = incidence.sum(dim=0)
    assert torch.equal(column_sums, torch.zeros_like(column_sums)), (
        f"{name}: 1^T B = {column_sums.tolist()}"
    )


@pytest.mark.parametrize("name,edge_index,num_atoms", TOPOLOGIES)
def test_every_bond_column_is_one_plus_one_and_one_minus_one(name, edge_index, num_atoms):
    """The only way ``1^T B == 0`` can hold structurally is +1/-1 per column."""
    build_incidence = require("topology", "build_incidence")
    incidence, bond_index = build_incidence(edge_index, num_atoms)
    n_bonds = incidence.shape[1]
    assert bond_index.shape == (2, n_bonds)
    for b in range(n_bonds):
        column = incidence[:, b]
        assert int((column == 1.0).sum()) == 1
        assert int((column == -1.0).sum()) == 1
        assert int((column == 0.0).sum()) == num_atoms - 2
    # An undirected pair appears exactly once, so t_ij = -t_ji is not a
    # constraint to enforce -- there is only one variable.
    pairs = {tuple(sorted(p)) for p in bond_index.t().tolist()}
    assert len(pairs) == n_bonds


def test_bond_charges_are_exactly_neutral_for_arbitrary_t():
    """q = B t must carry zero net charge bit for bit, for any t.

    SPEC 1.3 puts the net charge entirely in ``rho``; if ``B t`` leaked even a
    rounding error of charge, the per-fragment constraint would be fighting the
    bond variables.
    """
    build_incidence = require("topology", "build_incidence")
    torch.manual_seed(5)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 0, 2], [1, 2, 3, 4, 5, 3, 5]], dtype=torch.long
    )
    incidence, _ = build_incidence(edge_index, 6)
    for _ in range(20):
        t = torch.randn(incidence.shape[1]) * 3.7
        q = incidence @ t
        # Exact: summing q is the same additions in a different order, so this
        # is at worst a few ulp; assert against the tightest meaningful bound.
        assert abs(float(q.sum())) < 1.0e-14


def test_incidence_is_deterministic_and_orientation_independent():
    """Bond ordering must not depend on how the neighbour list was listed."""
    build_incidence = require("topology", "build_incidence")
    forward = torch.tensor([[0, 1, 2], [1, 2, 3]])
    shuffled = torch.tensor([[2, 1, 3, 0], [1, 0, 2, 1]])
    a, index_a = build_incidence(forward, 4)
    b, index_b = build_incidence(shuffled, 4)
    assert torch.equal(index_a, index_b)
    assert torch.equal(a, b)
    again, _ = build_incidence(forward, 4)
    assert torch.equal(a, again)


def test_self_loops_carry_no_split_charge():
    """An atom cannot transfer charge to itself; the bond must be dropped."""
    build_incidence = require("topology", "build_incidence")
    incidence, bond_index = build_incidence(
        torch.tensor([[0, 1, 1], [1, 1, 2]]), 3
    )
    assert incidence.shape[1] == 2
    assert (bond_index[0] != bond_index[1]).all()
    column_sums = incidence.sum(dim=0)
    assert torch.equal(column_sums, torch.zeros_like(column_sums))


def test_incidence_rejects_out_of_range_atom_indices():
    build_incidence = require("topology", "build_incidence")
    with pytest.raises(ValueError):
        build_incidence(torch.tensor([[0], [7]]), 3)


# ---------------------------------------------------------------------------
# Fragments -- the per-fragment constraint of SPEC 1.3
# ---------------------------------------------------------------------------
def test_fragments_are_the_connected_components_of_the_bond_graph():
    assign_fragments, count_fragments = require(
        "topology", "assign_fragments", "count_fragments"
    )
    # Atoms 0-1-2 bonded, 3-4 bonded, 5 isolated  ->  three fragments.
    bond_index = torch.tensor([[0, 1, 3], [1, 2, 4]])
    fragment_id = assign_fragments(6, bond_index)
    assert count_fragments(fragment_id) == 3
    labels = fragment_id.tolist()
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert len({labels[0], labels[3], labels[5]}) == 3


def test_fragment_constraint_matrix_selects_fragment_sums():
    """C q must be the vector of per-fragment total charges."""
    assign_fragments, constraint_matrix = require(
        "topology", "assign_fragments", "fragment_constraint_matrix"
    )
    bond_index = torch.tensor([[0, 1, 3], [1, 2, 4]])
    fragment_id = assign_fragments(6, bond_index)
    C = constraint_matrix(fragment_id)
    assert C.shape == (3, 6)
    # Every atom belongs to exactly one fragment.
    assert torch.equal(C.sum(dim=0), torch.ones(6))
    q = torch.tensor([0.3, -0.1, 0.8, -0.25, 1.25, -2.0])
    totals = C @ q
    expected = torch.tensor(
        [
            float(q[0] + q[1] + q[2]),
            float(q[3] + q[4]),
            float(q[5]),
        ]
    )
    torch.testing.assert_close(totals, expected, rtol=0, atol=1e-15)


def test_a_single_global_constraint_is_representable_and_is_not_the_default():
    """SPEC 1.3 contrasts the two; both must be constructible so the suite can
    measure the size-consistency gap between them (see test_solver.py)."""
    assign_fragments, constraint_matrix = require(
        "topology", "assign_fragments", "fragment_constraint_matrix"
    )
    n = 8
    per_fragment = constraint_matrix(assign_fragments(n, torch.tensor([[0], [1]])))
    global_only = constraint_matrix(torch.zeros(n, dtype=torch.long))
    assert global_only.shape == (1, n)
    assert per_fragment.shape[0] > 1  # the default really does split the system


def test_edge_to_bond_map_symmetrises_and_antisymmetrises_exactly():
    """SPEC 1.4: kappa must come from the symmetric part, the seed from the
    antisymmetric part, of a deliberately asymmetric attention logit."""
    build_incidence, map_edges, sym, anti = require(
        "topology",
        "build_incidence",
        "map_edges_to_bonds",
        "symmetrise_edge_scalar",
        "antisymmetrise_edge_scalar",
    )
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    _, bond_index = build_incidence(edge_index, 3)
    slot, sign = map_edges(edge_index, bond_index, 3)

    # Asymmetric "attention logits": Q_i . K_j != Q_j . K_i (SPEC 1.4).
    values = torch.tensor([2.0, -6.0, 1.5, 0.5])
    symmetric = sym(values, slot, bond_index.shape[1])
    antisymmetric = anti(values, slot, sign, bond_index.shape[1])

    # Bond (0,1) sees edges 0 and 1 with values 2.0 and -6.0:
    #   symmetric part     = 0.5 (2.0 + (-6.0)) = -2.0
    #   antisymmetric part = 0.5 (2.0 - (-6.0)) = +4.0  (sign fixed by orientation)
    # Bond (1,2) sees 1.5 and 0.5: symmetric 1.0, antisymmetric 0.5.
    torch.testing.assert_close(
        symmetric, torch.tensor([-2.0, 1.0]), rtol=0, atol=1e-15
    )
    torch.testing.assert_close(
        antisymmetric.abs(), torch.tensor([4.0, 0.5]), rtol=0, atol=1e-15
    )
