"""C5 (2026-11-10): the diabatic ensemble does not enumerate the bond-percolation partitions.

SPEC 12.6 enumerates the 2^k patterns of the k uncertain connections between molecules for the two-level and
exact fragment ensembles, and refuses k > ``fragment_max_uncertain``.  The diabatic ensemble (SPEC 17) carries
integer charges on every atom and never reads the partitions, so the limit must not refuse it (measured on the
Au2 / MgO doped 2x2 cell, 12 connections: experiments/benchmarks_4g/README.md, R17).  The bond graph's coverage
check stays for every ensemble.
"""
import pytest
import torch
from redox_helpers import cluster, edges, make_heads, neighbour_list, redox_float64  # noqa: F401  (autouse fixture)

Z6 = torch.tensor([8, 1, 1, 11, 17, 6])
SEED = 6                                         # this cluster has two independent uncertain connections


def _module(ensemble, **redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), RedoxConfig(fragment_ensemble=ensemble, **redox)), heads


def _evaluate(module, heads):
    pos, h, ei = cluster(6, heads.irreps_in.dim, SEED)
    pos = pos.clone().requires_grad_(True)
    vec, length = edges(pos, ei)
    energy = module(h, pos, ei, vec, length, data={"z": Z6, "total_charge": 0.0})[0]
    (grad,) = torch.autograd.grad(energy, pos)
    return energy.detach(), grad


def test_the_geometry_has_uncertain_connections():
    """Not vacuous: the enumeration itself refuses this bond graph at the limit 0 and gives 4 partitions at 10."""
    from trace_redox.fragments import FragmentEnsembleError, chemical_bonds, percolation_partitions
    heads = make_heads()
    pos, _, ei = cluster(6, heads.irreps_in.dim, SEED)
    _, length = edges(pos, ei)
    bi, bj, bs = chemical_bonds(Z6, ei, length, coverage=4.0)
    with pytest.raises(FragmentEnsembleError, match="2 independent connections"):
        percolation_partitions(6, bi, bj, bs, 0)
    assert len(percolation_partitions(6, bi, bj, bs, 10)) == 4


def test_the_diabatic_ensemble_does_not_enumerate_partitions():
    """C5-a: evaluates at the limit 0, bitwise equal to the default limit (the partitions are never read)."""
    reference = _evaluate(*_module("diabatic"))
    value = _evaluate(*_module("diabatic", fragment_max_uncertain=0))
    assert float(value[0]) == float(reference[0])
    assert bool((value[1] == reference[1]).all())


@pytest.mark.parametrize("ensemble", ["exact", "two-level"])
def test_the_partition_ensembles_still_refuse(ensemble):
    """C5-b: the ensembles that sum over the partitions keep the limit."""
    from trace_redox.fragments import FragmentEnsembleError
    with pytest.raises(FragmentEnsembleError, match="switching shell"):
        _evaluate(*_module(ensemble, fragment_max_uncertain=0))


def test_the_diabatic_ensemble_keeps_the_coverage_check():
    """C5-c: Cs-Cs bonds reach 5.48 A, beyond the 4 A neighbour list: refused whatever the ensemble."""
    from trace_redox.fragments import FragmentEnsembleError
    module, heads = _module("diabatic")
    pos = torch.tensor([[0.0, 0, 0], [3.0, 0, 0]])
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    h = torch.zeros(2, heads.irreps_in.dim)
    with pytest.raises(FragmentEnsembleError, match="attention_cutoff"):
        module(h, pos, ei, vec, length, data={"z": torch.tensor([55, 55]), "total_charge": 0.0})
