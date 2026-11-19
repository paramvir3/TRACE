"""SPEC 36: the cluster product ensemble (``RedoxConfig(diabatic_space='clusters')``).

SPEC 21 measures a configuration's importance from the thermal reference of the whole structure, so the energy of M
identical, non-interacting clusters is not M times one cluster's (experiments/benchmarks_4g/README.md, R13).  SPEC 36
selects linked configurations only, gives each cluster its own switched Hamiltonian and couples clusters by exact pair
increments of the free energy.  Claimed here, each with a test:

  P1  one cluster reproduces the shipped ensemble;
  P2  clusters that do not interact are additive to round-off, for any number of copies (the shipped ensemble is
      not: its failure is pinned, so the defect stays on record);
  P3  two clusters are exact: F is that of the global Hamiltonian on their product space;
  P4  with three clusters the error is of second order in the inter-cluster interaction: three-body induction, the
      mean charges of two clusters acting through the susceptibility of the third (<n_c> W chi_c' W <n_c''>).  It
      needs clusters with a mean moment (polar dimers here); without one the leading error is of third order;
  P5  the first and second derivatives are those of the function evaluated; the charge (increments of rho's diagonal)
      is dF/db exactly when no switch has a slope (unswitched, or every switch saturated), as in the shipped ensemble.

Corrected before any package code was written (hand derivation, 2026-11-10): the reservoir tests' counts (5
configurations per cluster, 2 with the reservoir charged; 21 products when the reservoir holds one electron), P4's
polar dimers, P5's condition, and P2's tolerance (it scales with the number of terms summed).
"""
import itertools

import pytest
import torch
from redox_helpers import cluster, edges, make_heads
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

EPS = torch.finfo(torch.float64).eps
KT = 0.1
AMPLITUDES = (0.05, 0.2)


def _dimers(M, x=2.64, gap=0.01, t=2.12, v=6.0, cross=0.0, seed=0, interleave=False):
    """M two-unit clusters, states -1..1, Q = 0.  In each: (0, 0) at 0, (+1, -1) at x, (-1, +1) at x + gap, coupled
    by t (the wetting dimer's numbers, README R13).  An on-site cost v/2 and an attraction v inside the pair keep those
    energies and put a transfer BETWEEN clusters at x + v, outside any window used here.  ``cross``: the scale of a
    seeded random interaction between units of different clusters.  ``interleave``: the units of the clusters
    alternate (a0, a1, ..., d0, d1, ...), so that Jordan-Wigner strings cross other clusters."""
    U = 2 * M
    g = torch.Generator().manual_seed(seed)
    states = torch.arange(-1, 2, dtype=torch.float64)
    first = [c if interleave else 2 * c for c in range(M)]
    second = [M + c if interleave else 2 * c + 1 for c in range(M)]
    e_hat = torch.zeros(U, 3, dtype=torch.float64)                  # columns: states -1, 0, +1
    W = torch.zeros(U, U, dtype=torch.float64)
    T = torch.zeros(U, U, dtype=torch.float64)
    owner = torch.zeros(U, dtype=torch.long)
    for c, (a, d) in enumerate(zip(first, second)):
        e_hat[a] = torch.tensor([0.5 * v, 0.0, 0.5 * v + x], dtype=torch.float64)
        e_hat[d] = torch.tensor([0.5 * v, 0.0, 0.5 * v + x + gap], dtype=torch.float64)
        W[a, d] = W[d, a] = v
        T[a, d] = T[d, a] = t
        owner[a] = owner[d] = c
    if cross:
        R = cross * torch.randn(U, U, generator=g, dtype=torch.float64)
        R = 0.5 * (R + R.T)
        W = W + torch.where(owner.unsqueeze(0) != owner.unsqueeze(1), R, torch.zeros_like(R))
    return states, e_hat, torch.zeros(U, dtype=torch.float64), W, torch.full((U,), 1), T


def _shipped(states, e_hat, b, W, K, T, Q=0, *, selection="amplitude", window=2.5, cap=100000):
    from trace_redox.diabatic import (canonical_free_energy, configuration_energies, density_matrix,
                                      diabatic_hamiltonian, select_configurations, switched_hamiltonian)
    configs = select_configurations(e_hat, states, b, W, K, Q, window=window, cap=cap,
                                    T=T if selection == "amplitude" else None, amplitudes=AMPLITUDES, kT=KT)
    E = configuration_energies(configs, e_hat, states, b, W)
    H = (switched_hamiltonian(E, configs, K, T, *AMPLITUDES, KT) if selection == "amplitude"
         else diabatic_hamiltonian(E, configs, K, T))
    return canonical_free_energy(H, KT), torch.diagonal(density_matrix(H, KT)) @ configs.to(E.dtype)


def _clusters(states, e_hat, b, W, K, T, Q=0, *, selection="amplitude", window=2.5, cap=100000, **kw):
    from trace_redox.clusters import cluster_ensemble
    return cluster_ensemble(e_hat, states, b, W, K, Q, window=window, cap=cap, T=T, amplitudes=AMPLITUDES, kT=KT,
                            selection=selection, **kw)


def _random_problem(U, seed, t_scale=0.6):
    g = torch.Generator().manual_seed(seed)
    states = torch.arange(-1, 2, dtype=torch.float64)
    chi = 3.0 * torch.rand(U, generator=g, dtype=torch.float64)
    gap = 2.0 + 4.0 * torch.rand(U, generator=g, dtype=torch.float64)
    e_hat = chi.unsqueeze(1) * states + 0.5 * gap.unsqueeze(1) * states * states
    b = 0.5 * torch.randn(U, generator=g, dtype=torch.float64)
    A = torch.randn(U, U, generator=g, dtype=torch.float64)
    W = 0.6 * (A @ A.T) / U + torch.diag(torch.full((U,), 0.5, dtype=torch.float64)) - 0.3
    T = t_scale * torch.randn(U, U, generator=g, dtype=torch.float64)
    T = 0.5 * (T + T.T)
    T.fill_diagonal_(0.0)
    return states, e_hat, b, 0.5 * (W + W.T), torch.full((U,), 1), T


# ---------------------------------------------------------------------------------------------------------- P1

@pytest.mark.parametrize("seed", range(6))
def test_one_cluster_is_the_shipped_ensemble(seed):
    """Three units: every move shares a unit with any support, so every configuration is linked and there is one
    cluster.  The selection's importance is measured from n0 instead of the set's thermal reference, so the set can
    hold configurations the shipped one does not; their couplings are switched off and they lie outside the window,
    so they weigh at most exp(-window / kT) = 1.4e-11."""
    problem = _random_problem(3, seed)
    F0, q0 = _shipped(*problem)
    out = _clusters(*problem)
    assert len(out.clusters) <= 1
    assert abs(float(out.energy - F0)) <= 1e-9
    assert float((out.charge - q0).abs().max()) <= 1e-8


def test_a_structure_with_one_configuration_is_its_energy():
    states, e_hat, b, W, K, T = _dimers(1, x=40.0, t=0.0)
    out = _clusters(states, e_hat, b, W, K, T)
    assert out.clusters == [] and float(out.energy) == 0.0 and float(out.charge.abs().max()) == 0.0


# ---------------------------------------------------------------------------------------------------------- P2

@pytest.mark.parametrize("interleave", [False, True])
def test_non_interacting_clusters_are_additive(interleave):
    one = _clusters(*_dimers(1))
    assert float(one.energy) < -1.0                                   # the coupling binds: nothing trivial is summed
    for M in (2, 3, 4, 6):
        out = _clusters(*_dimers(M, interleave=interleave))
        assert len(out.clusters) == M
        terms = M + M * (M - 1) // 2                                   # each a free energy of size |F1|, +- E0
        assert abs(float(out.energy) - M * float(one.energy)) <= 256 * EPS * terms * (1.0 + abs(float(one.energy)))
        charges = out.charge.reshape(2, M).T if interleave else out.charge.reshape(M, 2)
        assert float((charges - one.charge).abs().max()) <= 1e-12


def test_the_shipped_ensemble_is_not_additive_beyond_two_copies():
    """The defect, pinned (README R13.1): with the importance measured from the whole structure's reference the
    triply excited configurations are almost switched off, and three copies are not three times one."""
    F1, _ = _shipped(*_dimers(1))
    F2, _ = _shipped(*_dimers(2))
    F3, _ = _shipped(*_dimers(3))
    assert abs(float(F2) - 2 * float(F1)) <= 1e-9
    assert float(F3) - 3 * float(F1) > 1e-2                           # eV: weaker binding than three separate clusters


# ---------------------------------------------------------------------------------------------------------- P3

def _product_space_free_energy(problem, lists):
    """F of the GLOBAL unswitched Hamiltonian (true Jordan-Wigner signs) on the product space of ``lists``."""
    from trace_redox.diabatic import canonical_free_energy, configuration_energies, diabatic_hamiltonian
    states, e_hat, b, W, K, T = problem
    n0 = lists[0][0]
    rows = [n0 + sum(S[i] - n0 for S, i in zip(lists, index))
            for index in itertools.product(*[range(int(S.shape[0])) for S in lists])]
    configs = torch.stack(rows)
    E = configuration_energies(configs, e_hat, states, b, W)
    return canonical_free_energy(diabatic_hamiltonian(E, configs, K, T), KT)


@pytest.mark.parametrize("gap,window", [(0.01, 3.2), (1.2, 4.2)])     # non-polar and polar dimers
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_two_interacting_clusters_are_exact(seed, interleave, gap, window):
    """With ``interleave`` the Jordan-Wigner strings of one cluster's moves cross the other cluster's units: the
    Kronecker sum differs from the global Hamiltonian by a diagonal +-1 gauge, which no free energy sees."""
    problem = _dimers(2, gap=gap, cross=0.4, seed=seed, interleave=interleave)
    out = _clusters(*problem, selection="window", window=window)
    assert len(out.clusters) == 2 and all(int(S.shape[0]) == 3 for S in out.clusters)
    exact = _product_space_free_energy(problem, out.clusters)
    separate = 2 * float(_clusters(*_dimers(1, gap=gap), selection="window", window=window).energy)
    tolerance = 1e3 * EPS * (1.0 + abs(float(exact)))
    # the interaction matters in this test: it moves F by 7e-5 to 2e-2 eV over these cases (measured 2026-11-10; the
    # first guard written, 1e-3 eV, asked more than the random couplings of three of them give), 1e7 tolerances at least
    assert abs(float(exact) - separate) > 1e7 * tolerance
    assert abs(float(out.energy - exact)) <= tolerance


# ---------------------------------------------------------------------------------------------------------- P4

def test_three_clusters_miss_only_a_second_order_chain():
    """Polar dimers (gap 1.2 eV: a mean charge of opposite sign on the two units), so the three-body induction term
    is there and the error falls by 4 when the interaction is halved."""
    errors, pair_parts = [], []
    polar = dict(gap=1.2)
    one = float(_clusters(*_dimers(1, **polar), selection="window", window=4.2).energy)
    assert float(_clusters(*_dimers(1, **polar), selection="window", window=4.2).charge.abs().max()) > 0.02
    for scale in (0.2, 0.1, 0.05):
        problem = _dimers(3, cross=scale, seed=5, interleave=True, **polar)
        out = _clusters(*problem, selection="window", window=4.2)
        assert len(out.clusters) == 3 and all(int(S.shape[0]) == 3 for S in out.clusters)
        exact = float(_product_space_free_energy(problem, out.clusters))
        errors.append(abs(float(out.energy) - exact))
        pair_parts.append(abs(exact - 3 * one))
    assert errors[0] > 1e-9                                           # a real error, not round-off: the order is measurable
    for coarse, fine in zip(errors, errors[1:]):
        assert 3.0 <= coarse / fine <= 5.5                            # second order: 4 for a halved interaction
    assert errors[0] <= 0.05 * pair_parts[0]                          # and small against what the pairs carry


# ---------------------------------------------------------------------------------------------------------- P5

@pytest.mark.parametrize("selection,window", [("window", 4.2), ("amplitude", 2.5)])
def test_the_charge_is_the_field_derivative_of_the_free_energy(selection, window):
    """Unswitched: exact.  Amplitude rule: exact here because every switch is saturated (importance 0.63 > a_on =
    0.2 and 0.48 > a_on inside each dimer, so d sigma = 0); a switch on its slope adds d sigma / db, as in the shipped
    ensemble."""
    states, e_hat, b, W, K, T = _dimers(3, gap=1.2, cross=0.3, seed=1)   # polar: charges of 0.06 e to compare
    b = b.clone().requires_grad_(True)
    out = _clusters(states, e_hat, b, W, K, T, selection=selection, window=window)
    assert len(out.clusters) == 3 and all(int(S.shape[0]) == 3 for S in out.clusters)
    (grad,) = torch.autograd.grad(out.energy, b)
    assert float(out.charge.abs().max()) > 1e-3
    assert float((grad - out.charge).abs().max()) <= 1e-10


@pytest.mark.parametrize("selection", ["window", "amplitude"])
def test_the_derivatives_are_those_of_the_function(selection):
    states, e_hat, b0, W0, K, T0 = _dimers(2, cross=0.3, seed=2, interleave=True)
    window = 3.2 if selection == "window" else 2.5

    def energy(b, w, t):
        return _clusters(states, e_hat, b, W0 * w, K, T0 * t, selection=selection, window=window).energy

    b = (0.1 * torch.randn(4, generator=torch.Generator().manual_seed(3), dtype=torch.float64)).requires_grad_(True)
    w = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    t = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(energy, (b, w, t), eps=1e-6, atol=1e-7, rtol=1e-6)
    assert torch.autograd.gradgradcheck(energy, (b, w, t), eps=1e-6, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------------------------- the selection

def test_the_selection_is_linked_and_scales_with_the_number_of_clusters():
    """Three configurations per cluster whatever the number of clusters: no product configuration is selected."""
    for M in (2, 5, 12):
        out = _clusters(*_dimers(M))
        assert [int(S.shape[0]) for S in out.clusters] == [3] * M
        for S in out.clusters:
            assert bool((S[0] == out.reference).all())
            touched = (S != out.reference).any(0).nonzero().flatten().tolist()
            assert len(touched) == 2


def _with_reservoir(M, y=4.0, t_r=1.0, floor=-2):
    """``_dimers(M)`` and one more unit that takes an electron from the first unit of every cluster (at y, coupling
    t_r) and holds none at the reference: a band-like unit shared by all clusters."""
    states, e_hat, b, W, K, T = _dimers(M)
    U = 2 * M
    states = torch.arange(-2, 2, dtype=torch.float64)
    wide = torch.full((U + 1, 4), 1e3, dtype=torch.float64)           # columns: states -2, -1, 0, +1
    wide[:U, 1:] = e_hat
    wide[U] = torch.tensor([2 * (y - 2.64 - 3.0) + 0.5, y - 2.64 - 3.0, 0.0, 1e3], dtype=torch.float64)
    W2 = torch.zeros(U + 1, U + 1, dtype=torch.float64)
    W2[:U, :U] = W
    T2 = torch.zeros(U + 1, U + 1, dtype=torch.float64)
    T2[:U, :U] = T
    T2[U, 0:U:2] = T2[0:U:2, U] = t_r
    K2 = torch.cat([K, torch.tensor([0])])
    L2 = torch.cat([torch.full((U,), -1), torch.tensor([floor])])
    mask = torch.zeros(U + 1, dtype=torch.bool)
    mask[U] = True
    return (states, wide, torch.zeros(U + 1, dtype=torch.float64), W2, K2, T2), L2, mask


def test_a_reservoir_unit_does_not_link_the_clusters():
    problem, L, mask = _with_reservoir(3)
    fused = _clusters(*problem, min_charge=L)
    assert len(fused.clusters) == 1                                   # without the rule the shared unit fuses them
    first = _clusters(*problem, min_charge=L, reservoir=mask, closure=False)      # SPEC 36 as first built (legacy)
    out = _clusters(*problem, min_charge=L, reservoir=mask)
    assert len(first.clusters) == 3 and len(out.clusters) == 3
    for S in first.clusters:
        # n0; the two transfers inside the dimer; a -> reservoir from n0 (4.00 eV, amplitude 0.2425) and from
        # (a: -1, d: +1) (4.01 eV, 0.6247 x 0.2420 = 0.151 > a_off).  Two electrons there: 0.0104, not selected.
        assert int(S.shape[0]) == 5 and int((S[:, -1] != 0).sum()) == 2
    for S, linked in zip(out.clusters, first.clusters):
        # SPEC 36.7 (changed 2026-11-11; the count was 5): the closure adds the one product of the linked set,
        # (a: +1, R: -1) (+) (d: +1, R: -1), the two-electron row the amplitudes had left out.  README, C2b scored.
        assert int(S.shape[0]) == 6 and torch.equal(S[:5], linked)
        added = S[5] - out.reference
        assert int(added[-1]) == -2 and sorted(added[:-1].tolist()) == [0, 0, 0, 0, 1, 1]
    assert out.energy < first.energy and float(first.energy - out.energy) < 1e-3  # measured 1.405e-4 eV for the three


def test_a_product_that_overfills_the_reservoir_is_left_out():
    problem, L, mask = _with_reservoir(2, floor=-1)
    out = _clusters(*problem, min_charge=L, reservoir=mask)
    assert len(out.clusters) == 2 and out.pair_sizes == [21]          # 5 x 5 products less the 2 x 2 with two electrons there
    assert torch.isfinite(out.energy)


def test_the_selection_refuses_beyond_its_cap():
    from trace_redox.diabatic import ConfigurationSearchError
    with pytest.raises(ConfigurationSearchError):
        _clusters(*_dimers(6), cap=8)


# ------------------------------------------------------------------------------------------------- the module

Z6 = torch.tensor([79, 79, 12, 8, 12, 8])


def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", couplings=True, diabatic_selection="amplitude", **redox)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), config)
    with torch.no_grad():                         # couplings up to ~1.2 eV, the Au2 size (init ~0.01 eV)
        module.coupling.pair[-1].weight.mul_(3.0)
    return module, heads


def test_the_configuration_is_validated():
    from trace_redox import RedoxConfig
    assert RedoxConfig().diabatic_space == "global"                   # the shipped ensemble stays the default
    assert RedoxConfig.from_checkpoint({"fragment_ensemble": "diabatic"}).diabatic_space == "global"
    with pytest.raises(ValueError):
        RedoxConfig(fragment_ensemble="diabatic", diabatic_space="product")
    with pytest.raises(ValueError):
        RedoxConfig(diabatic_space="clusters")                        # needs the diabatic ensemble


def test_module_forces_are_minus_the_gradient_with_the_cluster_ensemble():
    module, heads = _module(diabatic_space="clusters")
    pos0, h, ei = cluster(6, heads.irreps_in.dim, 3)

    def energy(p):
        vec, length = edges(p, ei)
        return module(h, p, ei, vec, length, data={"z": Z6, "total_charge": 1.0})[0]

    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos, create_graph=True)
    assert module._ensemble_last["space"] == "clusters"
    for i, axis in [(0, 0), (3, 2), (5, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-7 * (1.0 + abs(fd))
    # a force loss can be differentiated (training on forces): the second derivatives exist and are finite
    params = [p for p in module.parameters() if p.requires_grad]
    grads = torch.autograd.grad((grad ** 2).sum(), params, allow_unused=True)
    assert all(g is None or bool(torch.isfinite(g).all()) for g in grads)
    assert any(g is not None and float(g.abs().max()) > 0 for g in grads)
