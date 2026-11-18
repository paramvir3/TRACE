"""R25 (benchmarks_4g README, 2026-11-12): ``max_operator_bytes`` bounds what the chosen elimination allocates.

``trace_electrostatics.energy.assemble`` refuses a dense n_m x n_m operator K, n_m = n_bonds + 3 n (+ 5 n with
quadrupoles): its own solver and ``bound_elimination='dense'`` build that K.  The condensed elimination
(``trace_redox.bound._CondensedK``, the default) never forms K; its largest arrays are [n, n_m] (the incidence, the
coupling A, K^-1 A^T).  Measured before the fix: the 990-atom Au2/MgO supercell was refused for a 10.25 GB K that the
route does not build, and ran in 127 s once the limit was raised by hand.

The limit keeps its meaning (bytes of the largest operator array the route holds) and its default; only the array it
is compared with follows the route.
"""
import dataclasses

import pytest
import torch

from redox_helpers import cluster, lattice_graph, make_heads, periodic_edges, redox_float64  # noqa: F401

EPS = torch.finfo(torch.float64).eps
Z6 = torch.tensor([79, 79, 12, 8, 12, 8])
BULK = (True, True, True)
ITEM = 8                                            # bytes of a float64


def _module(elimination):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", couplings=True, diabatic_selection="amplitude",
                         diabatic_space="clusters", bound_elimination=elimination)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


def _graph(heads):
    pos, h, _ = cluster(6, heads.irreps_in.dim, 5)
    cell = 7.0 * torch.eye(3, dtype=torch.float64)
    ei, es = lattice_graph(pos, cell, BULK, shells=1)
    vec, length = periodic_edges(pos, cell, ei, es)
    return pos, h, cell, ei, es, vec, length


def _evaluate(module, heads, limit=None):
    if limit is not None:
        module.config = dataclasses.replace(module.config, max_operator_bytes=float(limit))
    pos, h, cell, ei, es, vec, length = _graph(heads)
    return module(h, pos, ei, vec, length, pbc=BULK, cell=cell, edge_shift=es, data={"z": Z6, "total_charge": 0.0})


def _sizes():
    """(n, n_m) of the 6-atom cell, read off an evaluation at the default limit."""
    module, heads = _module("condensed")
    _, record = _evaluate(module, heads)
    assert record.theta_hat is None
    n = int(record.q.shape[0])
    return n, int(record.t.shape[0]) + 3 * n


def test_the_two_bounds_differ_on_the_test_cell():
    n, n_m = _sizes()
    assert n == 6 and n_m > n                        # so a limit between n n_m and n_m^2 bytes exists


def test_the_condensed_elimination_is_not_bound_by_the_dense_operator():
    n, n_m = _sizes()
    between = 0.5 * (n * n_m + n_m * n_m) * ITEM     # above the route's [n, n_m] arrays, below the dense K
    module, heads = _module("condensed")
    reference, _ = _evaluate(module, heads)
    energy, _ = _evaluate(module, heads, limit=between)
    assert abs(float(energy - reference)) <= 64 * EPS * abs(float(reference))


def test_the_dense_elimination_keeps_the_dense_bound():
    n, n_m = _sizes()
    module, heads = _module("dense")
    with pytest.raises(ValueError, match=f"dense {n_m}x{n_m} operator K"):
        _evaluate(module, heads, limit=0.5 * (n * n_m + n_m * n_m) * ITEM)


def test_the_condensed_elimination_has_its_own_bound():
    from trace_redox.bound import BoundSectorError
    n, n_m = _sizes()
    module, heads = _module("condensed")
    _evaluate(module, heads, limit=n * n_m * ITEM)                       # at the bound: runs
    with pytest.raises(BoundSectorError, match=rf"condensed elimination .*\[{n}, {n_m}\]"):
        _evaluate(module, heads, limit=n * n_m * ITEM - 1)


def test_the_wrapper_alone_keeps_the_legacy_bound_unless_told_the_route():
    """Callers of ``assemble_neutral`` that do not name the elimination (probes, tools) see no change."""
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox.bound import assemble_neutral
    n, n_m = _sizes()
    heads = make_heads()
    pos, h, cell, ei, es, vec, length = _graph(heads)
    config = dataclasses.replace(ElectrostaticConfig(), max_operator_bytes=0.5 * (n * n_m + n_m * n_m) * ITEM)
    keys = dict(pbc=BULK, cell=cell, edge_shift=es, species=Z6)
    with pytest.raises(ValueError, match="dense .* operator K"):
        assemble_neutral(h, pos, ei, vec, length, heads, config, **keys)
    assembly = assemble_neutral(h, pos, ei, vec, length, heads, config, elimination="condensed", **keys)
    assert assembly.n_atoms == n and assembly.n_bonds + 3 * n == n_m
