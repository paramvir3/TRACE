"""The canonical free energy of a diabatic Hamiltonian (audit step C2.1; docs/ELECTROSTATICS_AUDIT_2026-10-09.md).

    F(H) = -kT ln Tr exp(-H / kT),   H = diag(E_N) + T   (real symmetric, over configurations N)

ESTABLISHED: the two-state Marcus-Hush Hamiltonian, the Robin-Day classes (class II, a double well,
iff lambda > 2|t| for the symmetric pair), the canonical density matrix rho = exp(-H/kT) / Z as
dF/dH, and the Daleckii-Krein derivative of a matrix function.  NEW-TO-TRACE: its use over the exact
polarisable configuration energies of trace_redox, and the cancellation-free divided difference
Gamma_ij = -(p_lo / kT) psi(Delta / kT), psi(y) = -expm1(-y) / y, which keeps every derivative finite
and exact at a degenerate spectrum (autograd through eigh gives 0/0 there).

Tests: t = 0 is the existing exact ensemble; the two-state closed form; Robin-Day emerges; first and
second derivatives against finite differences, including exact degeneracy.
"""
import math

import pytest
import torch
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

EPS = torch.finfo(torch.float64).eps


def _free_energy():
    from trace_redox.diabatic import canonical_free_energy
    return canonical_free_energy


def _symmetric(n, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=torch.float64) * scale
    return 0.5 * (a + a.T)


@pytest.mark.parametrize("kT", [0.01, 0.1, 1.0])
def test_without_couplings_it_is_the_exact_configuration_ensemble(kT):
    F = _free_energy()
    E = torch.tensor([0.3, -0.2, 1.7, -0.2, 0.05], dtype=torch.float64, requires_grad=True)
    value = F(torch.diag(E), kT)
    reference = -kT * torch.logsumexp(-E / kT, dim=0)
    assert abs(float(value - reference)) <= 4 * EPS * (abs(float(reference)) + kT)
    (grad,) = torch.autograd.grad(value, E)
    p = torch.softmax(-E.detach() / kT, dim=0)
    assert torch.allclose(grad, p, rtol=0.0, atol=8 * EPS)            # dF/dE_N = P(N)


@pytest.mark.parametrize("delta,t,kT", [(0.0, 0.3, 0.05), (0.4, 0.1, 0.1), (-1.2, 0.5, 0.025), (0.2, 0.0, 0.1)])
def test_two_states_have_the_closed_form(delta, t, kT):
    """lambda_pm = E_bar +- sqrt(delta^2 / 4 + t^2);  F = -kT ln(e^{-lambda_-/kT} + e^{-lambda_+/kT})."""
    F = _free_energy()
    e_bar = 0.37
    H = torch.tensor([[e_bar + 0.5 * delta, t], [t, e_bar - 0.5 * delta]], dtype=torch.float64)
    root = math.sqrt(delta * delta / 4 + t * t)
    lo, hi = e_bar - root, e_bar + root
    reference = lo - kT * math.log1p(math.exp(-(hi - lo) / kT))
    assert abs(float(F(H, kT)) - reference) <= 8 * EPS * (1 + abs(reference))


@pytest.mark.parametrize("ratio,double_well", [(1.9, False), (2.1, True), (4.0, True), (1.0, False)])
def test_robin_day_class_ii_iff_lambda_exceeds_twice_the_coupling(ratio, double_well):
    """Symmetric Marcus-Hush pair E_1 = lam x^2, E_2 = lam (x - 1)^2, coupling t: the lower adiabatic
    surface has curvature 2 lam - lam^2 / |t| at x = 1/2, so it is a barrier (class II) iff lam > 2|t|.
    The ground state is F at kT -> 0 (kT = 1e-4 eV, gap >= 2|t| = 0.2 eV: excited weight e^{-2000})."""
    F = _free_energy()
    t = 0.1
    lam = ratio * t

    def ground(x):
        H = torch.tensor([[lam * x * x, t], [t, lam * (x - 1) ** 2]], dtype=torch.float64)
        return float(F(H, 1e-4))

    h = 1e-3
    curvature = (ground(0.5 + h) - 2 * ground(0.5) + ground(0.5 - h)) / (h * h)
    expected = 2 * lam - lam * lam / t
    assert abs(curvature - expected) <= 1e-5 * (1 + abs(expected))
    assert (curvature < 0) == double_well


@pytest.mark.parametrize("case", ["generic", "degenerate", "degenerate_coupled"])
def test_first_and_second_derivatives_are_exact(case):
    """dF/dH = rho; d2F[dH, dH'] by finite differences of the gradient.  'degenerate': two equal
    configuration energies and no coupling; 'degenerate_coupled': a symmetric three-site ring."""
    F = _free_energy()
    kT = 0.05
    if case == "generic":
        H0 = _symmetric(5, 1, 0.2) + torch.diag(torch.tensor([0.0, 0.1, -0.05, 0.3, 0.2], dtype=torch.float64))
    elif case == "degenerate":
        H0 = torch.diag(torch.tensor([0.2, 0.2, 0.5, -0.1], dtype=torch.float64))
    else:
        H0 = torch.full((3, 3), 0.07, dtype=torch.float64)
        H0.fill_diagonal_(0.1)                                          # spectrum {0.24, -0.03, -0.03}
    n = H0.shape[0]
    dH = _symmetric(n, 7, 1.0)
    dK = _symmetric(n, 8, 1.0)

    H = H0.clone().requires_grad_(True)
    value = F(H, kT)
    (grad,) = torch.autograd.grad(value, H, create_graph=True)
    assert bool(torch.isfinite(grad).all())
    # rho, independently
    evals, evecs = torch.linalg.eigh(H0)
    p = torch.softmax(-evals / kT, dim=0)
    rho = evecs @ torch.diag(p) @ evecs.T
    assert torch.allclose(grad.detach(), rho, rtol=0.0, atol=64 * EPS)

    # first directional derivative against central differences (O(h^2) error, h = 1e-5)
    h = 1e-5
    fd1 = (float(F(H0 + h * dH, kT)) - float(F(H0 - h * dH, kT))) / (2 * h)
    assert abs(float((grad.detach() * dH).sum()) - fd1) <= 1e-8 * (1 + abs(fd1))

    # second: d/dH <grad, dK> along dH, against differences of the analytic gradient
    (hvp,) = torch.autograd.grad((grad * dK).sum(), H)
    assert bool(torch.isfinite(hvp).all())

    def directional(shift):
        Hs = (H0 + shift).requires_grad_(True)
        (g,) = torch.autograd.grad(F(Hs, kT), Hs)
        return float((g * dK).sum())

    fd2 = (directional(h * dH) - directional(-h * dH)) / (2 * h)
    assert abs(float((hvp * dH).sum()) - fd2) <= 1e-6 * (1 + abs(fd2))


def test_free_energy_is_invariant_under_a_relabelling_of_configurations():
    F = _free_energy()
    H = _symmetric(4, 3, 0.3)
    perm = torch.tensor([2, 0, 3, 1])
    assert abs(float(F(H, 0.1)) - float(F(H[perm][:, perm], 0.1))) <= 16 * EPS * (1 + abs(float(F(H, 0.1))))


# ---- 17.2 configurations as determinants: one-electron moves and their fermionic signs --------------

import itertools  # noqa: E402

import numpy as np  # noqa: E402


def _fock_reference(n_configs, K):
    """<n_j| sum_{p in a, q in b} c_q^dag c_p |n_i> for every configuration pair and unit pair, by explicit
    Jordan-Wigner matrices on the full Fock space (independent of the module).  Returns {(i, j, a, b): value}."""
    offsets = np.concatenate(([0], np.cumsum([k + 1 for k in K])))
    n_orb = int(offsets[-1])
    a_op = np.array([[0.0, 1.0], [0.0, 0.0]])                        # annihilates |1> -> |0> in the (|0>, |1>) basis
    Z = np.diag([1.0, -1.0])
    I = np.eye(2)

    def annihilate(p):
        m = np.array([[1.0]])
        for o in range(n_orb):
            m = np.kron(m, Z if o < p else a_op if o == p else I)
        return m

    c = [annihilate(p) for p in range(n_orb)]

    def ket(n):
        bits = np.zeros(n_orb, dtype=int)
        for u, k in enumerate(K):
            filled = k - n[u]                                         # occupied local orbitals of unit u
            bits[offsets[u]:offsets[u] + filled] = 1
        index = int("".join(str(b) for b in bits), 2)                 # orbital 0 is the most significant factor
        vec = np.zeros(2 ** n_orb)
        vec[index] = 1.0
        return vec

    kets = [ket(n) for n in n_configs]
    out = {}
    for i, j in itertools.product(range(len(n_configs)), repeat=2):
        for a, b in itertools.permutations(range(len(K)), 2):
            value = 0.0
            for p in range(offsets[a], offsets[a + 1]):
                for q in range(offsets[b], offsets[b + 1]):
                    value += kets[j] @ (c[q].T @ c[p]) @ kets[i]
            if value != 0.0:
                out[(i, j, a, b)] = value
    return out


@pytest.mark.parametrize("K", [(1, 1, 1), (0, 2, 1), (1, 0, 1, 1)])
def test_one_electron_moves_and_signs_equal_the_fock_space_construction(K):
    from trace_redox.diabatic import one_electron_moves
    states = [range(-1, k + 1) for k in K]
    configs = [n for n in itertools.product(*states)]                # every charge vector, any total
    configs = [n for n in configs if sum(n) == sum(configs[len(configs) // 2])]   # one total charge
    reference = _fock_reference(configs, K)
    rows, cols, donor, acceptor, sign = one_electron_moves(torch.tensor(configs), torch.tensor(K))
    got = {(int(i), int(j), int(a), int(b)): float(s) for i, j, a, b, s in zip(rows, cols, donor, acceptor, sign)}
    assert got == reference
    assert any(v < 0 for v in reference.values())                   # the signs are exercised


# ---- 17.3 configuration energies and the configuration search ----------------------------------------

def _random_problem(U, seed, K=1):
    """A small free-charge problem: ladders e_hat [U, K + 2] over states -1..K, b, and a symmetric W."""
    g = torch.Generator().manual_seed(seed)
    states = torch.arange(-1, K + 1, dtype=torch.float64)
    # convex-ish ladders around 0 with random levels, like IE / EA anchors minus a self term
    chi = 3.0 * torch.rand(U, generator=g, dtype=torch.float64)
    gap = 2.0 + 4.0 * torch.rand(U, generator=g, dtype=torch.float64)
    e_hat = chi.unsqueeze(1) * states + 0.5 * gap.unsqueeze(1) * states * states
    b = 0.5 * torch.randn(U, generator=g, dtype=torch.float64)
    A = torch.randn(U, U, generator=g, dtype=torch.float64)
    W = 0.6 * (A @ A.T) / U + torch.diag(torch.full((U,), 0.5, dtype=torch.float64)) - 0.3   # not positive definite in general
    W = 0.5 * (W + W.T)
    return states, e_hat, b, W


def _all_configurations(U, K, Q):
    return [n for n in itertools.product(range(-1, K + 1), repeat=U) if sum(n) == Q]


def _energy_loop(n, states, e_hat, b, W):
    total = 0.0
    for u, nu in enumerate(n):
        total += float(e_hat[u, int(nu) + 1]) + float(b[u]) * nu
        for v, nv in enumerate(n):
            total += 0.5 * float(W[u, v]) * nu * nv
    return total


def test_configuration_energies_are_the_closed_form():
    from trace_redox.diabatic import configuration_energies
    states, e_hat, b, W = _random_problem(5, 1)
    configs = _all_configurations(5, 1, 0)
    E = configuration_energies(torch.tensor(configs), e_hat, states, b, W)
    for k, n in enumerate(configs):
        assert abs(float(E[k]) - _energy_loop(n, states, e_hat, b, W)) <= 1e-12 * (1 + abs(float(E[k])))


def _connected(configs):
    """Is the set connected under one-electron moves?"""
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


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("Q", [-1, 0, 1])
@pytest.mark.parametrize("window", [2.5, 6.0])
def test_the_search_finds_every_configuration_in_the_window(seed, Q, window):
    """Against brute force over every charge vector of a 6-unit problem (729 vectors), where the window
    set is connected under one-electron moves (asserted -- the stated condition of the search)."""
    from trace_redox.diabatic import configuration_energies, search_configurations
    U, K = 6, 1
    states, e_hat, b, W = _random_problem(U, 100 + seed)
    every = _all_configurations(U, K, Q)
    E_all = configuration_energies(torch.tensor(every), e_hat, states, b, W)
    lowest = float(E_all.min())
    inside = [every[k] for k in range(len(every)) if float(E_all[k]) <= lowest + window]
    if not _connected(inside):
        pytest.skip("window set not connected under single moves (outside the search's stated domain)")
    found, E_found = search_configurations(e_hat, states, b, W, torch.full((U,), K), Q, window=window, cap=4096)
    assert sorted(map(tuple, found.tolist())) == sorted(inside)
    assert abs(float(E_found.min()) - lowest) <= 1e-12 * (1 + abs(lowest))


def test_the_search_refuses_beyond_its_cap():
    from trace_redox.diabatic import ConfigurationSearchError, search_configurations
    states, e_hat, b, W = _random_problem(6, 3)
    with pytest.raises(ConfigurationSearchError, match="cap"):
        search_configurations(e_hat, states, b, W, torch.full((6,), 1), 0, window=1e3, cap=10)


def test_the_wide_windows_exercise_the_flood_fill():
    """The 6 eV windows above hold 14-28 of the 126 configurations each (measured), so the search is
    tested well beyond the minimum."""
    from trace_redox.diabatic import configuration_energies
    sizes = []
    for seed in range(8):
        states, e_hat, b, W = _random_problem(6, 100 + seed)
        every = _all_configurations(6, 1, 0)
        E = configuration_energies(torch.tensor(every), e_hat, states, b, W)
        sizes.append(int((E <= E.min() + 6.0).sum()))
    assert min(sizes) >= 10, sizes


def test_the_descent_terminates_whatever_the_move_estimates(monkeypatch):
    """Inconsistent move energies (every move claimed to gain 1 eV) once made the descent cycle forever
    (measured 2026-10-09); the exact-energy acceptance ends it."""
    import trace_redox.diabatic as diabatic
    states, e_hat, b, W = _random_problem(6, 4)

    def lying(n, e_hat, K, b, W, **keywords):
        delta = torch.full((n.shape[0], n.shape[0]), -1.0, dtype=W.dtype)
        return delta.fill_diagonal_(float("inf"))

    monkeypatch.setattr(diabatic, "_move_energies", lying)
    configs, E = diabatic.search_configurations(e_hat, states, b, W, torch.full((6,), 1), 0, window=2.0, cap=4096)
    assert configs.shape[0] >= 1


@pytest.mark.parametrize("seed", range(4))
def test_move_energies_are_exact_energy_differences(seed):
    """The search's move estimates must equal E(n + e_a - e_b) - E(n) for every valid move: a wrong
    estimate can wrongly prune a configuration reachable along one path only (the brute-force tests
    above are blind to that, by redundancy of paths -- found with a planted sign error)."""
    from trace_redox.diabatic import _move_energies, _valid_moves, configuration_energies
    states, e_hat, b, W = _random_problem(7, 300 + seed, K=2)
    K = torch.full((7,), 2)
    g = torch.Generator().manual_seed(seed)
    for _ in range(5):
        n = torch.randint(-1, 3, (7,), generator=g)
        delta = _move_energies(n, e_hat, K, b, W)
        valid = _valid_moves(n, K)
        base = float(configuration_energies(n.unsqueeze(0), e_hat, states, b, W)[0])
        for a, c in torch.nonzero(valid).tolist():
            m = n.clone()
            m[a] += 1
            m[c] -= 1
            exact = float(configuration_energies(m.unsqueeze(0), e_hat, states, b, W)[0]) - base
            assert abs(float(delta[a, c]) - exact) <= 1e-12 * (1 + abs(base)), (a, c)
        assert bool(torch.isinf(delta[~valid]).all())                 # invalid moves, diagonal included


# ---- 17.4 the module: fragment_ensemble='diabatic' --------------------------------------------------

from redox_helpers import cluster, edges, make_heads  # noqa: E402

Z6 = torch.tensor([8, 1, 1, 11, 17, 6])


def _diabatic_module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), RedoxConfig(fragment_ensemble="diabatic", **redox)), heads


@pytest.mark.parametrize("Q", [0.0, 1.0, -1.0])
def test_without_couplings_the_module_is_the_exact_ensemble_over_atomic_integer_charges(Q):
    """F = -kT ln sum_n exp(-E(n)/kT) over EVERY integer atomic charge vector (brute force, 3^6), with E(n)
    from the bound-sector elimination and the ladder: no mean field anywhere.  Window 3 eV at kT = 0.1:
    the omitted weight is <= 729 e^-30 ~ 7e-11 relative."""
    from trace_redox.bound import assemble_neutral, eliminate_bound_sector
    from trace_electrostatics.config import ElectrostaticConfig
    module, heads = _diabatic_module(diabatic_window=3.0)
    pos, h, ei = cluster(6, heads.irreps_in.dim, 3)
    vec, length = edges(pos, ei)
    energy, result = module(h, pos, ei, vec, length, data={"z": Z6, "total_charge": Q})
    a = assemble_neutral(h, pos, ei, vec, length, heads, ElectrostaticConfig())
    elim = eliminate_bound_sector(a, module.width(Z6))
    states = result.states
    e_hat = result.state_energies - 0.5 * torch.diagonal(elim.J).unsqueeze(1) * (states * states).unsqueeze(0)
    kT = module.redox.kT
    logs = []
    for n in itertools.product(range(-1, 2), repeat=6):
        if sum(n) != round(Q):
            continue
        nt = torch.tensor(n, dtype=torch.float64)
        E = sum(float(e_hat[u, n[u] + 1]) for u in range(6)) + float(elim.u0 @ nt) + 0.5 * float(nt @ elim.W @ nt)
        logs.append(-E / kT)
    reference = float(elim.e_b0) - kT * float(torch.logsumexp(torch.tensor(logs), 0))
    assert abs(float(energy) - reference) <= 1e-9 * (1 + abs(reference))
    assert abs(float(result.free_charge.sum()) - Q) <= 1e-12


def test_diabatic_module_forces_are_minus_the_gradient():
    module, heads = _diabatic_module()
    pos0, h, ei = cluster(6, heads.irreps_in.dim, 3)

    def energy(p):
        vec, length = edges(p, ei)
        return module(h, p, ei, vec, length, data={"z": Z6, "total_charge": 1.0})[0]

    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    for i, axis in [(0, 0), (3, 2), (5, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-7 * (1.0 + abs(fd))


# ---- per-unit charge ranges [L_u, K_u] (formal oxidation states: O down to -2, Mg 0..+2, ...) ----------

def _fock_reference_ranges(n_configs, L, K):
    """As _fock_reference, with unit u holding K_u - L_u local orbitals (charge L_u: all filled)."""
    sizes = [k - l for l, k in zip(L, K)]
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    n_orb = int(offsets[-1])
    a_op = np.array([[0.0, 1.0], [0.0, 0.0]])
    Z = np.diag([1.0, -1.0])
    I = np.eye(2)

    def annihilate(p):
        m = np.array([[1.0]])
        for o in range(n_orb):
            m = np.kron(m, Z if o < p else a_op if o == p else I)
        return m

    c = [annihilate(p) for p in range(n_orb)]

    def ket(n):
        bits = np.zeros(n_orb, dtype=int)
        for u in range(len(K)):
            bits[offsets[u]:offsets[u] + K[u] - n[u]] = 1
        vec = np.zeros(2 ** n_orb)
        vec[int("".join(str(x) for x in bits), 2)] = 1.0
        return vec

    kets = [ket(n) for n in n_configs]
    out = {}
    for i, j in itertools.product(range(len(n_configs)), repeat=2):
        for a, b in itertools.permutations(range(len(K)), 2):
            value = sum(kets[j] @ (c[q].T @ c[p]) @ kets[i]
                        for p in range(offsets[a], offsets[a + 1]) for q in range(offsets[b], offsets[b + 1]))
            if value != 0.0:
                out[(i, j, a, b)] = value
    return out


@pytest.mark.parametrize("L,K", [((-2, 0, -1), (0, 2, 1)), ((-1, -2, 0, -1), (1, 0, 1, 0))])
def test_moves_with_charge_ranges_equal_the_fock_space_construction(L, K):
    from trace_redox.diabatic import one_electron_moves
    configs = [n for n in itertools.product(*[range(l, k + 1) for l, k in zip(L, K)])]
    target = sum(configs[len(configs) // 2])
    configs = [n for n in configs if sum(n) == target]
    reference = _fock_reference_ranges(configs, L, K)
    rows, cols, donor, acceptor, sign = one_electron_moves(torch.tensor(configs), torch.tensor(K))
    got = {(int(i), int(j), int(a), int(b)): float(s) for i, j, a, b, s in zip(rows, cols, donor, acceptor, sign)}
    assert got == reference


def _ranged_problem(L, K, seed):
    """Ladders over the global state range min(L)..max(K), +inf outside each unit's own range."""
    g = torch.Generator().manual_seed(seed)
    U = len(L)
    states = torch.arange(min(L), max(K) + 1, dtype=torch.float64)
    chi = 3.0 * torch.rand(U, generator=g, dtype=torch.float64)
    gap = 2.0 + 4.0 * torch.rand(U, generator=g, dtype=torch.float64)
    e_hat = chi.unsqueeze(1) * states + 0.5 * gap.unsqueeze(1) * states * states
    for u in range(U):
        e_hat[u, (states < L[u]) | (states > K[u])] = float("inf")
    b = 0.5 * torch.randn(U, generator=g, dtype=torch.float64)
    A = torch.randn(U, U, generator=g, dtype=torch.float64)
    W = 0.5 * ((0.6 * (A @ A.T) / U - 0.3) + (0.6 * (A @ A.T) / U - 0.3).T)
    return states, e_hat, b, W


RANGES = [((-2, 0, -1, -2, 0, -1), (0, 2, 1, 0, 2, 1)), ((-1, -1, 0, 0, -2, -1), (1, 1, 2, 2, 0, 0))]


@pytest.mark.parametrize("L,K", RANGES)
@pytest.mark.parametrize("seed", range(4))
def test_the_search_with_charge_ranges_finds_every_configuration_in_the_window(L, K, seed):
    from trace_redox.diabatic import configuration_energies, search_configurations
    states, e_hat, b, W = _ranged_problem(L, K, 500 + seed)
    every = [n for n in itertools.product(*[range(l, k + 1) for l, k in zip(L, K)])]
    Q = sum(every[len(every) // 2])
    every = [n for n in every if sum(n) == Q]
    E_all = configuration_energies(torch.tensor(every), e_hat, states, b, W)
    lowest = float(E_all.min())
    inside = [every[k] for k in range(len(every)) if float(E_all[k]) <= lowest + 6.0]
    if not _connected(inside):
        pytest.skip("window set not connected under single moves")
    found, E_found = search_configurations(e_hat, states, b, W, torch.tensor(K), Q, window=6.0, cap=4096,
                                           min_charge=torch.tensor(L))
    assert sorted(map(tuple, found.tolist())) == sorted(inside)
    assert abs(float(E_found.min()) - lowest) <= 1e-12 * (1 + abs(lowest))


@pytest.mark.parametrize("L,K", RANGES)
def test_move_energies_with_charge_ranges_are_exact(L, K):
    from trace_redox.diabatic import _move_energies, _valid_moves, configuration_energies
    states, e_hat, b, W = _ranged_problem(L, K, 9)
    Lt, Kt = torch.tensor(L), torch.tensor(K)
    g = torch.Generator().manual_seed(1)
    for _ in range(6):
        n = torch.stack([torch.randint(l, k + 1, (1,), generator=g)[0] for l, k in zip(L, K)])
        delta = _move_energies(n, e_hat, Kt, b, W, states=states, min_charge=Lt)
        valid = _valid_moves(n, Kt, min_charge=Lt)
        base = float(configuration_energies(n.unsqueeze(0), e_hat, states, b, W)[0])
        for a, c in torch.nonzero(valid).tolist():
            m = n.clone()
            m[a] += 1
            m[c] -= 1
            exact = float(configuration_energies(m.unsqueeze(0), e_hat, states, b, W)[0]) - base
            assert abs(float(delta[a, c]) - exact) <= 1e-12 * (1 + abs(base))
        assert bool(torch.isinf(delta[~valid]).all())


# ---- 17.5 formal oxidation states in the diabatic mode -------------------------------------------------

def test_the_formal_ladder_holds_the_exact_isolated_ion_energies():
    from trace_redox.constants import ATOMIC_ELECTRON_AFFINITY_EV as EA, ATOMIC_SUCCESSIVE_IONIZATION_EV as IES
    from trace_redox.diabatic import formal_ladder
    species = torch.tensor([13, 8, 12, 17, 1])
    states, E, L, K = formal_ladder(species, max_charge=3)
    col = lambda n: int(n - states[0])                                   # noqa: E731
    assert states.tolist() == [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    # K_Z = min(max_charge, tabulated IEs): Mg^3+ (IE_3 = 80 eV) is allowed and simply never populated
    assert L.tolist() == [-1, -2, -1, -1, -1] and K.tolist() == [3, 3, 3, 3, 1]
    assert abs(float(E[0, col(3)]) - float(IES[13][0] + IES[13][1] + IES[13][2])) <= 1e-12
    assert abs(float(E[1, col(-2)]) - (-float(EA[8]) + 8.75)) <= 1e-12
    assert abs(float(E[2, col(-1)]) - 0.42) <= 1e-12                      # Mg: EA = -0.42 (unbound anion)
    assert float(E[3, col(-1)]) == -float(EA[17])
    assert all(float(E[u, col(0)]) == 0.0 for u in range(5))
    assert torch.isinf(E[4, col(2)]) and torch.isinf(E[0, col(-2)])       # H has one electron; no Al^2-


def _formal_module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


@pytest.mark.parametrize("z,Q", [(13, 3.0), (8, -2.0), (17, -1.0), (12, 2.0)])
def test_an_isolated_ion_has_exactly_its_tabulated_energy(z, Q):
    from trace_redox.diabatic import formal_ladder
    module, heads = _formal_module()
    pos = torch.zeros(1, 3)
    h = torch.randn(1, heads.irreps_in.dim, generator=torch.Generator().manual_seed(2))
    ei = torch.zeros(2, 0, dtype=torch.long)
    vec, length = edges(pos, ei)
    energy, result = module(h, pos, ei, vec, length, data={"z": torch.tensor([z]), "total_charge": Q})
    states, E, _, _ = formal_ladder(torch.tensor([z]), max_charge=3)
    expected = float(E[0, int(Q - states[0])])
    assert abs(float(energy - result.energy_bound) - expected) <= 1e-9 * (1 + abs(expected))


Z_FORMAL = torch.tensor([8, 12, 12, 8, 11, 17])


def _physical_fragment(heads, seed=3):
    """A distorted rocksalt fragment O Mg Mg O Na Cl, nearest distances 2.1-2.6 A (the cluster helper put
    two atoms 0.29 A apart: with compact ionic charges that landscape is pathological, an artifact)."""
    from redox_helpers import neighbour_list
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor([[0.0, 0.0, 0.0], [2.1, 0.0, 0.0], [0.0, 2.1, 0.0], [2.1, 2.1, 0.0],
                         [0.0, 0.0, 2.5], [2.4, 0.1, 2.6]], dtype=torch.float64)
    pos = base + 0.1 * torch.randn(6, 3, generator=g, dtype=torch.float64)
    h = torch.randn(6, heads.irreps_in.dim, generator=g, dtype=torch.float64)
    return pos, h, neighbour_list(pos)


def test_formal_module_without_couplings_equals_the_enumeration():
    """Every configuration of an O/Mg/Na/Cl cluster in its own ranges (O -2..3, Mg, Na, Cl -1..3),
    E(n) = sum (E_Z + Delta - S_0 n^2 / 2) + u0.n + n.W.n / 2, against the module (window 3 eV, kT 0.1)."""
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_electrostatics.kernels import gram_self_energy
    from trace_redox.bound import assemble_neutral, eliminate_bound_sector
    from trace_redox.diabatic import formal_ladder
    module, heads = _formal_module(diabatic_window=3.0)
    pos, h, ei = _physical_fragment(heads)
    vec, length = edges(pos, ei)
    energy, result = module(h, pos, ei, vec, length, data={"z": Z_FORMAL, "total_charge": 0.0})
    a = assemble_neutral(h, pos, ei, vec, length, heads, ElectrostaticConfig())
    sigma = module.width(Z_FORMAL)
    elim = eliminate_bound_sector(a, sigma)
    states, E_Z, L, K = formal_ladder(Z_FORMAL, max_charge=3)
    S0 = gram_self_energy(sigma)
    logs = []
    for n in itertools.product(*[range(int(l), int(k) + 1) for l, k in zip(L, K)]):
        if sum(n) != 0:
            continue
        nt = torch.tensor(n, dtype=torch.float64)
        onsite = sum(float(E_Z[u, int(n[u] - states[0])]) - 0.5 * float(S0[u]) * n[u] ** 2 for u in range(6))
        logs.append(-(onsite + float(elim.u0 @ nt) + 0.5 * float(nt @ elim.W @ nt)) / module.redox.kT)
    reference = float(elim.e_b0) - module.redox.kT * float(torch.logsumexp(torch.tensor(logs), 0))
    assert abs(float(energy) - reference) <= 1e-9 * (1 + abs(reference))


def test_formal_mode_forces_are_minus_the_gradient():
    module, heads = _formal_module()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        last = module.charge_states.shift_mlp[-1]
        last.weight.copy_(torch.randn(last.weight.shape, generator=g) * 0.3)
        last.bias.copy_(torch.randn(last.bias.shape, generator=g) * 0.3)
    pos0, h, ei = _physical_fragment(heads)

    def energy(p):
        vec, length = edges(p, ei)
        return module(h, p, ei, vec, length, data={"z": Z_FORMAL, "total_charge": 0.0})[0]

    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    for i, axis in [(0, 0), (3, 2), (5, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-7 * (1.0 + abs(fd))


def test_formal_anchors_need_the_diabatic_ensemble():
    from trace_redox import RedoxConfig
    with pytest.raises(ValueError, match="diabatic"):
        RedoxConfig(anchor="formal", fragment_ensemble="exact", max_charge=3)


# ---- 17.6 one-electron couplings tau_uv between configurations -------------------------------------

def _coupling(seed=0, **kwargs):
    from trace_redox.heads import TransferCoupling
    torch.manual_seed(seed)
    return TransferCoupling(12, **kwargs)


def test_couplings_are_symmetric_invariant_and_vanish_smoothly_at_the_cutoff():
    head = _coupling(init_scale=0.3)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4, 12, generator=g, dtype=torch.float64)
    pos = 2.2 * torch.randn(4, 3, generator=g, dtype=torch.float64)
    ei = torch.tensor([(i, j) for i in range(4) for j in range(4) if i != j]).T

    def T(p):
        return head(x, ei, (p[ei[1]] - p[ei[0]]).norm(dim=-1))

    T0 = T(pos)
    assert torch.allclose(T0, T0.T, rtol=0.0, atol=0.0) and bool((torch.diagonal(T0) == 0).all())
    R = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))[0]
    assert torch.allclose(T(pos @ R.T + 1.7), T0, rtol=0.0, atol=1e-13)
    # the radial factor and its first two derivatives vanish at r_cut
    r = torch.tensor([head.r_cut - 1e-3, head.r_cut, head.r_cut + 0.5], dtype=torch.float64)
    edge = head.edge_values(x[:2], torch.tensor([[0, 0, 0], [1, 1, 1]]), r)
    assert float(edge[1]) == 0.0 and float(edge[2]) == 0.0
    assert abs(float(edge[0])) <= 1e-6                  # (1 - f) ~ 10 x^3 deficit: O((1e-3 / r_cut)^3)


def _dimer(module, heads, z, Q, separation=3.0):
    pos = torch.tensor([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]], dtype=torch.float64)
    h = torch.randn(1, heads.irreps_in.dim, generator=torch.Generator().manual_seed(4), dtype=torch.float64).repeat(2, 1)
    ei = torch.tensor([[0, 1], [1, 0]])
    vec, length = edges(pos, ei)
    return module(h, pos, ei, vec, length, data={"z": torch.tensor([z, z]), "total_charge": Q}), (h, ei, length)


def test_a_symmetric_dimer_cation_gains_exactly_minus_kT_ln_cosh():
    """Na2+: (Na+, Na0) and (Na0, Na+) are degenerate by symmetry (identical features); coupled by tau
    they split to E0 -+ tau, so F - F(tau = 0) = -kT ln cosh(tau / kT) exactly, and each Na holds +1/2."""
    module, heads = _formal_module(couplings=True)
    with torch.no_grad():
        for p in module.coupling.parameters():
            p.normal_(0.0, 0.2)
    (energy, result), (h, ei, length) = _dimer(module, heads, 11, 1.0)
    invariants = heads.node_invariants(h)
    tau = float(module.coupling(invariants, ei, length)[0, 1])
    off, _ = _formal_module(couplings=False)
    (energy0, _), _ = _dimer(off, heads, 11, 1.0)
    kT = module.redox.kT
    assert abs(tau) > 1e-3
    expected = -kT * math.log(math.cosh(tau / kT))
    assert abs(float(energy - energy0) - expected) <= 1e-9 * (1 + abs(expected))
    assert torch.allclose(result.free_charge, torch.tensor([0.5, 0.5], dtype=torch.float64), rtol=0.0, atol=1e-12)


def test_zero_couplings_reproduce_the_uncoupled_module_exactly():
    module, heads = _formal_module(couplings=True)
    with torch.no_grad():
        for p in module.coupling.parameters():
            p.zero_()
    off, _ = _formal_module(couplings=False)
    pos, h, ei = _physical_fragment(heads)
    vec, length = edges(pos, ei)
    e1 = module(h, pos, ei, vec, length, data={"z": Z_FORMAL, "total_charge": 0.0})[0]
    e0 = off(h, pos, ei, vec, length, data={"z": Z_FORMAL, "total_charge": 0.0})[0]
    assert abs(float(e1 - e0)) <= 1e-12 * (1 + abs(float(e0)))


def test_coupled_forces_are_minus_the_gradient():
    module, heads = _formal_module(couplings=True)
    with torch.no_grad():
        for p in module.coupling.parameters():
            p.normal_(0.0, 0.2)
    pos0, h, ei = _physical_fragment(heads)

    def energy(p):
        vec, length = edges(p, ei)
        return module(h, p, ei, vec, length, data={"z": Z_FORMAL, "total_charge": 1.0})[0]

    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    for i, axis in [(0, 0), (3, 2), (5, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-7 * (1.0 + abs(fd))


def test_the_hamiltonian_carries_the_fermionic_signs_of_two_carriers():
    """H = diag(E) + sign tau over every one-electron move, with the signs of the explicit Fock-space
    construction (independent of one_electron_moves): two carriers in the configurations, so loops where an
    electron hops past the other carrier exist.  A dropped sign changes the spectrum (a planted bug the
    one-carrier module tests could not see)."""
    from trace_redox.diabatic import canonical_free_energy, diabatic_hamiltonian
    L, K = [-1, -1, -1, -1], [1, 1, 1, 1]
    configs = [n for n in itertools.product(range(-1, 2), repeat=4) if sum(n) == 2]
    g = torch.Generator().manual_seed(3)
    E = torch.randn(len(configs), generator=g, dtype=torch.float64) * 0.2
    A = torch.randn(4, 4, generator=g, dtype=torch.float64) * 0.3
    T = (A + A.T).fill_diagonal_(0.0)
    H = diabatic_hamiltonian(E, torch.tensor(configs), torch.tensor(K), T)
    reference = torch.diag(E).clone()
    signs = _fock_reference_ranges(configs, L, K)
    for (i, j, donor, acceptor), s in signs.items():
        reference[i, j] += s * T[donor, acceptor]
    assert torch.equal(H, reference)
    unsigned = torch.diag(E).clone()
    for (i, j, donor, acceptor), s in signs.items():
        unsigned[i, j] += T[donor, acceptor]
    assert any(s < 0 for s in signs.values())
    assert abs(float(canonical_free_energy(unsigned, 0.05)) - float(canonical_free_energy(reference, 0.05))) > 1e-6


def test_an_ion_hop_is_found_by_the_search():
    """Na4 2+ on a 3.0 A square: the lowest configurations are the four Na- + 3 Na+ (6.29-6.65 eV,
    measured), connected only by moving TWO electrons at once (every single move passes 5 eV up).  The
    search with k-fold moves finds all four; the module's F equals the enumeration over every vector."""
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_electrostatics.kernels import gram_self_energy
    from trace_redox.bound import assemble_neutral, eliminate_bound_sector
    from trace_redox.diabatic import formal_ladder
    from redox_helpers import neighbour_list
    module, heads = _formal_module_k1()
    z = torch.tensor([11, 11, 11, 11])
    pos = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 3.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float64)
    h = torch.randn(4, heads.irreps_in.dim, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    energy, result = module(h, pos, ei, vec, length, data={"z": z, "total_charge": 2.0})
    a = assemble_neutral(h, pos, ei, vec, length, heads, ElectrostaticConfig())
    sigma = module.width(z)
    elim = eliminate_bound_sector(a, sigma, sigma_self=sigma)
    states, E_Z, L, K = formal_ladder(z, max_charge=1)
    S0 = gram_self_energy(sigma)
    logs = []
    for n in itertools.product(range(-1, 2), repeat=4):
        if sum(n) != 2:
            continue
        nt = torch.tensor(n, dtype=torch.float64)
        onsite = sum(float(E_Z[u, int(n[u] - states[0])]) - 0.5 * float(S0[u]) * n[u] ** 2 for u in range(4))
        logs.append(-(onsite + float(elim.u0 @ nt) + 0.5 * float(nt @ elim.W @ nt)) / module.redox.kT)
    reference = float(elim.e_b0) - module.redox.kT * float(torch.logsumexp(torch.tensor(logs), 0))
    assert int(result.iterations) == 4
    assert abs(float(energy) - reference) <= 1e-9 * (1 + abs(reference))


def dataclasses_replace(obj, **changes):
    import dataclasses
    return dataclasses.replace(obj, **changes)


def _formal_module_k1(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=1, **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("L,K", RANGES)
def test_k_fold_move_energies_are_exact(L, K, k):
    from trace_redox.diabatic import _move_energies, _valid_moves, configuration_energies
    states, e_hat, b, W = _ranged_problem(L, K, 21)
    Lt, Kt = torch.tensor(L), torch.tensor(K)
    g = torch.Generator().manual_seed(k)
    for _ in range(6):
        n = torch.stack([torch.randint(l, kk + 1, (1,), generator=g)[0] for l, kk in zip(L, K)])
        delta = _move_energies(n, e_hat, Kt, b, W, states=states, min_charge=Lt, k=k)
        valid = _valid_moves(n, Kt, min_charge=Lt, k=k)
        base = float(configuration_energies(n.unsqueeze(0), e_hat, states, b, W)[0])
        for a, c in torch.nonzero(valid).tolist():
            m = n.clone()
            m[a] += k
            m[c] -= k
            exact = float(configuration_energies(m.unsqueeze(0), e_hat, states, b, W)[0]) - base
            assert abs(float(delta[a, c]) - exact) <= 1e-12 * (1 + abs(base))
        assert bool(torch.isinf(delta[~valid]).all())
