"""Pauli exclusion between two carrier units (``RedoxConfig(exclusion_pairs=True)``, trace_redox SPEC 18.5).

Two frontier densities repel (density-overlap model) when at least one of them is a closed shell; two open shells
pair into a bond instead:

    Xi_uv(n_u, n_v) = K s(R) [P(n_u, n_v) S(rho_u^(n_u), rho_v^(n_v)) - P(f_u, f_v) S(rho_u^(f_u), rho_v^(f_v))]
    P = N_u N_v  if closed(n_u) or closed(n_v)  else 0          (closed: N in {2, 8}; N = 0 has no density)

for pairs of open-shell (neutral-frame) units -- SPEC 18 is the case of a closed-shell frame neighbour.  For Au2:
Au0Au0 (sigma^2) 0; Au-Au0 (sigma^2 sigma*^1) 2KS; Au-Au- (sigma^2 sigma*^2, bond order 0) 4KS; Au+Au- 0 (empty 6s).
Configuration energies gain a two-body term, E(n) = sum e_hat + b^T n + 1/2 n^T W n + sum_pairs Xi_p(n_i, n_j); the
search's move energies stay exact.
"""
import itertools
import math

import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

HARTREE = 27.211386245988
BOHR = 0.529177210903


def _random_problem(U, seed, K=1):
    g = torch.Generator().manual_seed(seed)
    states = torch.arange(-1, K + 1, dtype=torch.float64)
    chi = 3.0 * torch.rand(U, generator=g, dtype=torch.float64)
    gap = 2.0 + 4.0 * torch.rand(U, generator=g, dtype=torch.float64)
    e_hat = chi.unsqueeze(1) * states + 0.5 * gap.unsqueeze(1) * states * states
    b = 0.5 * torch.randn(U, generator=g, dtype=torch.float64)
    A = torch.randn(U, U, generator=g, dtype=torch.float64)
    W = 0.6 * (A @ A.T) / U + torch.diag(torch.full((U,), 0.5, dtype=torch.float64)) - 0.3
    S = states.numel()
    pi, pj = torch.tensor([0, 1, 3]), torch.tensor([1, 2, 4])
    Xi = 1.5 * torch.rand(3, S, S, generator=g, dtype=torch.float64)
    return states, e_hat, b, 0.5 * (W + W.T), (pi, pj, Xi)


def _energy_loop(n, states, e_hat, b, W, pairs):
    first = int(states[0])
    total = sum(float(e_hat[u, nu - first]) + float(b[u]) * nu for u, nu in enumerate(n))
    total += 0.5 * sum(float(W[u, v]) * nu * nv for u, nu in enumerate(n) for v, nv in enumerate(n))
    pi, pj, Xi = pairs
    total += sum(float(Xi[p, n[int(pi[p])] - first, n[int(pj[p])] - first]) for p in range(pi.numel()))
    return total


def test_configuration_energies_with_pairs_are_the_closed_form():
    from trace_redox.diabatic import configuration_energies
    states, e_hat, b, W, pairs = _random_problem(5, 1)
    configs = [n for n in itertools.product(range(-1, 2), repeat=5) if sum(n) == 0]
    E = configuration_energies(torch.tensor(configs), e_hat, states, b, W, pairs=pairs)
    for k, n in enumerate(configs):
        assert abs(float(E[k]) - _energy_loop(n, states, e_hat, b, W, pairs)) <= 1e-12 * (1 + abs(float(E[k])))


@pytest.mark.parametrize("seed", range(4))
def test_move_energies_with_pairs_are_exact(seed):
    from trace_redox.diabatic import _move_energies, configuration_energies
    states, e_hat, b, W, pairs = _random_problem(5, 10 + seed)
    K = torch.full((5,), 1)
    for n in [torch.tensor(c) for c in itertools.product(range(-1, 2), repeat=5) if sum(c) == 0][:20]:
        delta = _move_energies(n, e_hat, K, b, W, states=states, k=1, pairs=pairs)
        base = float(configuration_energies(n.unsqueeze(0), e_hat, states, b, W, pairs=pairs)[0])
        for a, c in itertools.permutations(range(5), 2):
            m = n.clone()
            m[a] += 1
            m[c] -= 1
            if bool((m > 1).any() or (m < -1).any()):
                assert math.isinf(float(delta[a, c]))
                continue
            exact = float(configuration_energies(m.unsqueeze(0), e_hat, states, b, W, pairs=pairs)[0]) - base
            assert abs(float(delta[a, c]) - exact) <= 1e-12 * (1 + abs(exact))


def _connected(configs):
    index = {tuple(n): i for i, n in enumerate(configs)}
    seen, stack = {0}, [0]
    while stack:
        n = configs[stack.pop()]
        for a, b in itertools.permutations(range(len(n)), 2):
            m = list(n)
            m[a] += 1
            m[b] -= 1
            j = index.get(tuple(m))
            if j is not None and j not in seen:
                seen.add(j)
                stack.append(j)
    return len(seen) == len(configs)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("Q", [-1, 0, 1])
def test_the_search_with_pairs_finds_every_configuration(seed, Q):
    from trace_redox.diabatic import configuration_energies, search_configurations
    U, window = 6, 3.0
    states, e_hat, b, W, pairs = _random_problem(U, 200 + seed)
    every = [n for n in itertools.product(range(-1, 2), repeat=U) if sum(n) == Q]
    E_all = configuration_energies(torch.tensor(every), e_hat, states, b, W, pairs=pairs)
    lowest = float(E_all.min())
    inside = [every[k] for k in range(len(every)) if float(E_all[k]) <= lowest + window]
    if not _connected(inside):
        pytest.skip("window set not connected under single moves")
    found, _ = search_configurations(e_hat, states, b, W, torch.full((U,), 1), Q, window=window, cap=4096,
                                     pairs=pairs)
    assert sorted(map(tuple, found.tolist())) == sorted(inside)


def test_au2_pair_values():
    """Au2 at 2.47 A, default K = 10 Ha: Au-Au- = 4 K S(Au-, Au-), Au-Au0 = 2 K S(Au-, Au0) (independent quadrature);
    Au0Au0, Au+Au-, Au+Au+ = 0.  Au-O pairs are absent (O is a closed-shell frame: SPEC 18's one-body term)."""
    from trace_redox.constants import formal_frame
    from trace_redox.exclusion import Exclusion, density_overlap
    z = torch.tensor([79, 79, 8])
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.47, 0.0, 0.0], [0.0, 0.0, -2.2]], dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.0)
    _, length = edges(pos, ei)
    states = torch.arange(-2, 4, dtype=torch.float64)
    pi, pj, Xi = Exclusion(r_cut=5.0).pair_terms(z, states, ei, length, formal_frame(z))
    assert pi.tolist() == [0] and pj.tolist() == [1]
    k = {int(s): i for i, s in enumerate(states.tolist())}

    def kb(I, n):
        kk = math.sqrt(2 * I / HARTREE)
        return torch.tensor(kk), torch.tensor((n + 1) / kk - 1)

    am, a0 = kb(2.30861, -1), kb(9.225554, 0)
    R = torch.tensor(2.47 / BOHR)
    K = 10 * HARTREE
    assert abs(float(Xi[0, k[-1], k[-1]]) - 4 * K * float(density_overlap(*am, *am, R))) <= 1e-7 * float(Xi[0, k[-1], k[-1]])
    assert abs(float(Xi[0, k[-1], k[0]]) - 2 * K * float(density_overlap(*am, *a0, R))) <= 1e-7 * float(Xi[0, k[-1], k[0]])
    assert float(Xi[0, k[-1], k[0]]) == float(Xi[0, k[0], k[-1]])
    for a, c in [(0, 0), (1, -1), (-1, 1), (1, 1), (1, 0)]:
        assert float(Xi[0, k[a], k[c]]) == 0.0


def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True, exclusion=True,
                         **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


def test_pair_exclusion_forces_are_minus_the_gradient():
    module, heads = _module(exclusion_pairs=True)
    g = torch.Generator().manual_seed(5)
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0]],
                        dtype=torch.float64)
    z = torch.tensor([8, 79, 79, 12, 13])
    h = torch.randn(5, heads.irreps_in.dim, generator=g, dtype=torch.float64)
    ei = neighbour_list(pos0)

    def energy(p):
        vec, length = edges(p, ei)
        return module(h, p, ei, vec, length, data={"z": z, "total_charge": -1.0})[0]

    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    for i, axis in [(1, 2), (2, 2), (2, 0), (4, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-7 * (1.0 + abs(fd))


def test_it_needs_exclusion():
    from trace_redox import RedoxConfig
    with pytest.raises(ValueError, match="exclusion_pairs"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", exclusion_pairs=True)
