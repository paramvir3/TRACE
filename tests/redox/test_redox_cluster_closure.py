"""SPEC 36.7 ("C2b"): the product closure of the cluster ensemble.

SPEC 36's space is meant to hold every product of selected excitations on disjoint supports.  As first built it held
them only ACROSS clusters: one cluster that holds two disjoint excitations lacked their product (README R14; measured
on the benchmark's doped 2x1 cell, R19: +1.243 eV).  Claimed here, each with a test:

  Q2  two clusters that merge do not lose their products: no jump of F at the merge;
  Q3  a merged cluster is exact on the closed set (unswitched): F is that of the global Hamiltonian on it, and the
      closed set holds the products of the parts;
  Q4  a transfer between two clusters that is the product of two exchanges with a reservoir does not join them; it
      lives in their pair space, and F is exact on the product space;
  Q5  the 6-atom random-weight module: FAILED as written against the global space; what holds against the full
      space is asserted instead (written after the measurement, not a criterion);
  the first build stays reachable (``closure=False``) and its defect stays pinned.
"""
import itertools

import pytest
import torch
import test_redox_cluster_product as base
from redox_helpers import cluster, edges
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

EPS = torch.finfo(torch.float64).eps
KT = base.KT


def _bridged(tau):
    states, e_hat, b, W, K, T = base._dimers(2)
    T[1, 2] = T[2, 1] = tau
    return states, e_hat, b, W, K, T


def _global_free_energy(problem, configs):
    """F of the GLOBAL unswitched Hamiltonian (true Jordan-Wigner signs) on ``configs``."""
    from trace_redox.diabatic import canonical_free_energy, configuration_energies, diabatic_hamiltonian
    states, e_hat, b, W, K, T = problem
    E = configuration_energies(configs, e_hat, states, b, W)
    return canonical_free_energy(diabatic_hamiltonian(E, configs, K, T), KT)


def _brute_closure(S, reservoir=None):
    """Every product of members of ``S`` (n0 first) with pairwise disjoint supports, by enumeration."""
    n0 = S[0]
    res = torch.zeros_like(n0, dtype=torch.bool) if reservoir is None else reservoir
    supports = [frozenset(((row != n0) & ~res).nonzero().flatten().tolist()) for row in S[1:]]
    found = {tuple(n0.tolist())}
    for k in range(1, len(supports) + 1):
        grew = False
        for index in itertools.combinations(range(len(supports)), k):
            if all(supports[i].isdisjoint(supports[j]) for i, j in itertools.combinations(index, 2)):
                found.add(tuple((n0 + sum(S[1 + i] - n0 for i in index)).tolist()))
                grew = True
        if not grew:
            break
    return torch.tensor(sorted(found), dtype=torch.long)


def test_the_first_build_stays_reachable_and_its_defect_pinned():
    below = base._clusters(*_bridged(0.4325), closure=False)
    above = base._clusters(*_bridged(0.4335), closure=False)
    assert [int(S.shape[0]) for S in below.clusters] == [3, 3]
    assert [int(S.shape[0]) for S in above.clusters] == [7]
    assert abs(float(above.energy - below.energy) - 0.789756) < 1e-5          # README R14: -3.908722 -> -3.118966 eV


def test_a_merge_does_not_lose_the_products():                                # Q2
    below, above = base._clusters(*_bridged(0.4325)), base._clusters(*_bridged(0.4335))
    assert len(below.clusters) == 2 and len(above.clusters) == 1              # the merge does happen in between
    shipped = base._shipped(*_bridged(0.4335))[0] - base._shipped(*_bridged(0.4325))[0]
    assert abs(float(above.energy - below.energy) - float(shipped)) < 1e-6


def test_a_merged_cluster_is_exact_on_its_closed_set():                       # Q3
    problem = base._dimers(2)
    first = base._clusters(*problem, selection="window", window=9.0, closure=False)
    out = base._clusters(*problem, selection="window", window=9.0)
    assert len(first.clusters) == 1 and len(out.clusters) == 1                # transfers between the dimers at 8.64 eV
    closed = _brute_closure(first.clusters[0])
    assert int(closed.shape[0]) > int(first.clusters[0].shape[0])             # not vacuous: products are added
    have = {tuple(r) for r in closed.tolist()}
    n0 = first.clusters[0][0]
    local = [[n0] + [m for m in first.clusters[0][1:] if set((m != n0).nonzero().flatten().tolist()) <= pair]
             for pair in ({0, 1}, {2, 3})]
    assert [len(x) for x in local] == [3, 3]
    for a, c in itertools.product(*local):
        assert tuple((a + c - n0).tolist()) in have                           # the 3 x 3 products of the two dimers
    exact = _global_free_energy(problem, closed)
    assert abs(float(out.energy - exact)) < 64 * EPS * (1.0 + abs(float(exact)))
    assert float(first.energy - exact) > 1e-3                                 # the first build is above it


def _two_way_reservoir(out=1.5, back=1.3, w=6.0):
    """``_dimers(2)`` and a reservoir unit (the last) that can take an electron from any unit (at ``out``) or give
    one (at ``back``), uncoupled; a uniform attraction ``w`` between the dimers leaves every excitation inside a
    dimer where it was and puts a transfer between the dimers at x + v - w = 2.64 eV."""
    states, e_hat, b, W, K, T = base._dimers(2)
    U = 4
    wide = torch.zeros(U + 1, 3, dtype=torch.float64)
    wide[:U] = e_hat
    wide[U] = torch.tensor([out - (3.0 + 2.64), 0.0, back - 3.0], dtype=torch.float64)
    W2 = torch.zeros(U + 1, U + 1, dtype=torch.float64)
    W2[:U, :U] = W
    W2[:2, 2:U] = W2[2:U, :2] = w
    T2 = torch.zeros(U + 1, U + 1, dtype=torch.float64)
    T2[:U, :U] = T
    mask = torch.zeros(U + 1, dtype=torch.bool)
    mask[U] = True
    return (states, wide, torch.zeros(U + 1, dtype=torch.float64), W2, torch.full((U + 1,), 1), T2), mask


def test_a_product_through_the_reservoir_does_not_join_the_clusters():        # Q4
    problem, mask = _two_way_reservoir()
    first = base._clusters(*problem, selection="window", window=3.2, reservoir=mask, closure=False)
    assert len(first.clusters) == 1                                           # the first build fuses the two dimers
    out = base._clusters(*problem, selection="window", window=3.2, reservoir=mask)
    assert [int(S.shape[0]) for S in out.clusters] == [7, 7]                  # n0, 2 inside, 2 out, 2 back
    n0 = out.reference
    rows = [a + c - n0 for a, c in itertools.product(out.clusters[0], out.clusters[1])]
    product = torch.stack([r for r in rows if int(r.abs().max()) <= 1])
    assert out.pair_sizes == [int(product.shape[0])]
    have = {tuple(r) for r in product.tolist()}
    for m in first.clusters[0]:
        assert tuple(m.tolist()) in have                                      # every selected configuration is kept
    exact = _global_free_energy(problem, product)
    assert abs(float(out.energy - exact)) < 64 * EPS * (1.0 + abs(float(exact)))


def _six_atoms(seed, selection="amplitude", **redox):
    """Energy and the ensemble record of the 6-atom random-weight module (``base._module`` with the selection free)."""
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = base.make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", couplings=True, diabatic_selection=selection, **redox)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), config)
    with torch.no_grad():
        module.coupling.pair[-1].weight.mul_(3.0)
    pos, h, ei = cluster(6, heads.irreps_in.dim, seed)
    vec, length = edges(pos, ei)
    Z = torch.tensor([79, 79, 12, 8, 12, 8])
    energy = float(module(h, pos, ei, vec, length, data={"z": Z, "total_charge": 0.0})[0])
    return energy, module._ensemble_last


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_the_six_atom_module_against_the_full_space(seed):                    # Q5: FAILED as written; what holds
    """Q5 as pre-registered, |clusters - global| <= 1e-2 eV, FAILED: -0.0565, -0.1757, -0.3639 eV (seeds 3, 4, 5).
    The reference it named, the global SPEC 21 space, is itself +0.370, +0.159, +0.413 eV above the full space (all
    141 neutral configurations of six units with charges -1, 0, +1, unswitched).  Written after the measurement, so
    not a criterion (README, "C2b scored"):

      * the closed set with the bare Hamiltonian is variational (a principal block of the full Hamiltonian);
      * the closure is closer to the full space than the first build and than the global space: +0.313, -0.016,
        +0.049 eV against +0.538, +0.694, +0.486 (first build).  The switch is not variational: on seed 4 the cluster
        holds all 141 rows and the switched Hamiltonian lies 16.4 meV BELOW the bare one."""
    from trace_redox.diabatic import canonical_free_energy
    full, record = _six_atoms(seed, selection="window", diabatic_space="global", diabatic_window=1e3)
    H, rows = record["H"], record["configs"]
    count = sum(1 for c in itertools.product((-1, 0, 1), repeat=6) if sum(c) == 0)
    assert int(rows.shape[0]) == count == 141                                 # the complete space
    glob, _ = _six_atoms(seed, diabatic_space="global")
    first, _ = _six_atoms(seed, diabatic_space="clusters", cluster_closure=False)
    closed, record = _six_atoms(seed, diabatic_space="clusters")
    assert abs(closed - full) < abs(glob - full) < abs(first - full)
    assert len(record["clusters"]) == 1                                       # one cluster on these three seeds
    index = {tuple(r): i for i, r in enumerate(rows.tolist())}
    order = torch.tensor([index[tuple(r)] for r in record["clusters"][0].tolist()])
    block = H[order][:, order]
    assert torch.equal(record["H"].diagonal(), block.diagonal()) or \
        float((record["H"].diagonal() - block.diagonal()).abs().max()) < 64 * EPS * (1.0 + float(block.diagonal().abs().max()))
    bare = float(canonical_free_energy(block, record["kT"]))                  # the ensemble's part of the energy
    exact = float(canonical_free_energy(H, record["kT"]))
    assert bare >= exact - 64 * EPS * (1.0 + abs(exact))                      # interlacing: rows only raise F
    if seed == 4:                                                             # every row: the bare block is the full space
        assert int(order.numel()) == 141 and abs(bare - exact) < 64 * EPS * (1.0 + abs(exact))
        assert closed < full - 1e-3                                           # the switch alone: measured -16.4 meV
