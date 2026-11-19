"""Amplitude selection of the configuration space (``RedoxConfig(diabatic_selection='amplitude')``; trace_redox
SPEC 21).

The diabatic sector diagonalises H = diag(E) + T over a finite set P of integer configurations.  With the energy
window alone (SPEC 17.3) P(R) changes discontinuously: a configuration crossing the window edge with a coupling t
to the low block moves the lowest level by ~ -t^2 / Delta, so F(R) jumps and the forces never see it.  Measured
2026-10-18 on step 12 (Au2/MgO, tau(Au, Au) = 1.18 eV): -834 meV on one test structure (seed 43), -219 meV
(seed 42).

The amplitude selection (NEW-TO-TRACE; its criterion is the perturbative amplitude of selected CI, ESTABLISHED:
Huron, Malrieu & Rancurel, J. Chem. Phys. 58, 5745 (1973)).  With E_ref = -kT ln sum exp(-E / kT) and
x_c = E_c - E_ref:

    r_c   = max( sqrt(p_c), max over coupled i of r_i g(T_ic, x_c) ),  p_c = exp(-x_c / kT),
    g     = |t| / sqrt(x^2 + t^2)       (the first-order amplitude |t| / x, saturated at 1)
    e_ij  = max(r_i g(T, x_j), r_j g(T, x_i)),     sigma = smoothstep((ln e - ln a_off) / ln(a_on / a_off))
    H~    = diag(E) + sum_moves sign * sigma(e_ij) * T_ab,    P = window set + {r > a_off}.

r_c is the leading Rayleigh-Schroedinger amplitude of c along its best path from the thermally weighted reference.
Static mixing is explicit; virtual excitations, and chains of them, are left to the bound sector.  Since g <= 1,
r_c >= e_ic for every neighbour, so P is closed (sigma = 0 on every pair leaving it): F changes with the window only
through the Boltzmann weight of decoupled configurations, and is continuous in the geometry.

Tests: the window selection jumps where the amplitude selection is continuous; closure and window independence; the
importance is the path product, so a ring of virtual hole hops does not grow the set; the complete limit is
brute-force full diagonalisation; the switch is C^2; the closed form in the switch region; the closure refuses
beyond its cap; no couplings means no change; module forces are minus the gradient.
"""
import itertools
import math

import pytest
import torch
from redox_helpers import cluster, edges, make_heads
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

EPS = torch.finfo(torch.float64).eps
KT = 0.1


def _dimer(x, t=1.2, gap=3.0):
    """Two units, states -1..1, Q = 0: (0, 0) at 0, (+1, -1) at x, (-1, +1) at x + gap; b = W = 0."""
    states = torch.arange(-1, 2, dtype=torch.float64)
    e_hat = torch.tensor([[0.0, 0.0, x], [0.0, 0.0, x + gap]], dtype=torch.float64)   # columns: states -1, 0, +1
    b = torch.zeros(2, dtype=torch.float64)
    W = torch.zeros(2, 2, dtype=torch.float64)
    K = torch.tensor([1, 1])
    T = torch.tensor([[0.0, t], [t, 0.0]], dtype=torch.float64)
    return states, e_hat, b, W, K, T


def _free_energy(states, e_hat, b, W, K, T, Q, *, selection, window=2.5, amplitudes=(0.05, 0.2), cap=10000,
                 min_charge=None):
    from trace_redox.diabatic import (canonical_free_energy, configuration_energies, diabatic_hamiltonian,
                                      select_configurations, switched_hamiltonian)
    configs = select_configurations(e_hat, states, b, W, K, Q, window=window, cap=cap, min_charge=min_charge,
                                    T=T if selection == "amplitude" else None, amplitudes=amplitudes, kT=KT)
    E = configuration_energies(configs, e_hat, states, b, W)
    if selection == "amplitude":
        H = switched_hamiltonian(E, configs, K, T, *amplitudes, KT)
    else:
        H = diabatic_hamiltonian(E, configs, K, T)
    return canonical_free_energy(H, KT), configs, H


def test_the_window_selection_jumps_where_the_amplitude_selection_is_continuous():
    """(+1, -1) crosses the 2.5 eV window edge at x = 2.5.  Window selection: F jumps from the 2x2 eigenvalue
    x/2 - sqrt(x^2/4 + t^2) = -0.483 eV to 0.  Amplitude selection (importances g = 1.2 / sqrt(2.5^2 + 1.2^2) = 0.43
    and 1.2 / sqrt(5.5^2 + 1.2^2) = 0.21, both above a_on): every configuration is in P on both sides and F is
    continuous, equal to the complete 3x3 value."""
    h = 1e-7
    lo = [_free_energy(*_dimer(2.5 - h), 0, selection=s)[0] for s in ("window", "amplitude")]
    hi = [_free_energy(*_dimer(2.5 + h), 0, selection=s)[0] for s in ("window", "amplitude")]
    assert float(hi[0] - lo[0]) > 0.3                                         # the defect, measured
    assert abs(float(hi[1] - lo[1])) <= 2 * h + 64 * EPS                     # |dF/dx| <= 1
    for x, F in ((2.5 - h, lo[1]), (2.5 + h, hi[1])):
        H = torch.tensor([[0.0, -1.2, 1.2], [-1.2, x, 0.0], [1.2, 0.0, x + 3.0]], dtype=torch.float64)
        ev = torch.linalg.eigvalsh(H)
        reference = -KT * torch.logsumexp(-ev / KT, 0)
        assert abs(float(F - reference)) <= 64 * EPS * (1 + abs(float(reference)))


def _random_problem(U, seed, t_scale=0.6, K=1):
    g = torch.Generator().manual_seed(seed)
    states = torch.arange(-1, K + 1, dtype=torch.float64)
    chi = 3.0 * torch.rand(U, generator=g, dtype=torch.float64)
    gap = 2.0 + 4.0 * torch.rand(U, generator=g, dtype=torch.float64)
    e_hat = chi.unsqueeze(1) * states + 0.5 * gap.unsqueeze(1) * states * states
    b = 0.5 * torch.randn(U, generator=g, dtype=torch.float64)
    A = torch.randn(U, U, generator=g, dtype=torch.float64)
    W = 0.6 * (A @ A.T) / U + torch.diag(torch.full((U,), 0.5, dtype=torch.float64)) - 0.3
    W = 0.5 * (W + W.T)
    T = t_scale * torch.randn(U, U, generator=g, dtype=torch.float64)
    T = 0.5 * (T + T.T)
    T.fill_diagonal_(0.0)
    return states, e_hat, b, W, torch.full((U,), K), T


@pytest.mark.parametrize("seed", range(6))
def test_the_amplitude_set_is_closed_so_a_wider_window_adds_only_decoupled_blocks(seed):
    """P(2.5) is closed: inside the 6 eV set's H~, every element between P(2.5) and the rest is exactly zero, so
    F(6) = -kT ln(Z_P + Z_rest), Z_P the 2.5 eV value (to the shift of E_ref by the added configurations' Boltzmann
    weight, < 1e-10 eV here)."""
    from trace_redox.diabatic import canonical_free_energy
    problem = _random_problem(6, seed)
    F_small, P_small, _ = _free_energy(*problem, 0, selection="amplitude", window=2.5)
    F_big, P_big, H_big = _free_energy(*problem, 0, selection="amplitude", window=6.0)
    big = {tuple(c): i for i, c in enumerate(P_big.tolist())}
    inside = torch.tensor([big[tuple(c)] for c in P_small.tolist()])
    outside = torch.tensor(sorted(set(range(len(big))) - set(inside.tolist())), dtype=torch.long)
    assert len(inside) == P_small.shape[0]                                    # P(2.5) is a subset of P(6)
    if outside.numel():
        assert float(H_big[inside][:, outside].abs().max()) == 0.0
        Z_rest = canonical_free_energy(H_big[outside][:, outside], KT)
        reference = -KT * torch.logaddexp(-F_small / KT, -Z_rest / KT)
        assert abs(float(F_big - reference)) <= 1e-10 * (1 + abs(float(reference)))
    else:
        assert abs(float(F_big - F_small)) <= 1e-10 * (1 + abs(float(F_small)))


def _all_configurations(U, K, Q):
    return [n for n in itertools.product(range(-1, K + 1), repeat=U) if sum(n) == Q]


@pytest.mark.parametrize("seed", range(4))
def test_the_complete_limit_is_full_diagonalisation(seed):
    """Thresholds -> 0: every non-zero coupling is fully on and the closure reaches every configuration connected
    by one-electron moves, here all 3^5 vectors of total charge 0 (brute force)."""
    from trace_redox.diabatic import canonical_free_energy, configuration_energies, diabatic_hamiltonian
    states, e_hat, b, W, K, T = _random_problem(5, seed)
    F, configs, _ = _free_energy(states, e_hat, b, W, K, T, 0, selection="amplitude", amplitudes=(1e-12, 2e-12))
    full = torch.tensor(_all_configurations(5, 1, 0))
    assert configs.shape[0] == full.shape[0]
    H = diabatic_hamiltonian(configuration_energies(full, e_hat, states, b, W), full, K, T)
    reference = canonical_free_energy(H, KT)
    assert abs(float(F - reference)) <= 1e-12 * (1 + abs(float(reference)))


def test_the_switch_is_c2_and_the_amplitude_saturates():
    from trace_redox.diabatic import amplitude_switch, coupling_amplitude
    a_off, a_on = 0.05, 0.2
    assert float(amplitude_switch(torch.tensor(a_off * 0.999), a_off, a_on)) == 0.0
    assert float(amplitude_switch(torch.tensor(a_on * 1.001), a_off, a_on)) == 1.0
    assert abs(float(amplitude_switch(torch.tensor(math.sqrt(a_off * a_on)), a_off, a_on)) - 0.5) <= 8 * EPS
    grid = torch.exp(torch.linspace(math.log(a_off), math.log(a_on), 201, dtype=torch.float64))
    values = amplitude_switch(grid, a_off, a_on)
    assert bool((values[1:] >= values[:-1]).all())
    for a in (a_off, a_on):
        x = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        (d1,) = torch.autograd.grad(amplitude_switch(x, a_off, a_on), x, create_graph=True)
        (d2,) = torch.autograd.grad(d1, x)
        assert abs(float(d1)) <= 1e-12 and abs(float(d2)) <= 1e-9
    # g = |t| / sqrt(x^2 + t^2): ~ |t| / x for a large gap, 1 at zero gap, 0 at zero coupling, finite gradients
    t, x = torch.tensor(0.27, dtype=torch.float64), torch.tensor(14.2, dtype=torch.float64)
    assert abs(float(coupling_amplitude(t, x)) - 0.27 / math.sqrt(14.2 ** 2 + 0.27 ** 2)) <= 4 * EPS
    zero = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    gap = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    assert float(coupling_amplitude(t, gap)) == 1.0
    assert float(coupling_amplitude(zero, x)) == 0.0
    g = torch.autograd.grad(coupling_amplitude(zero, gap) + coupling_amplitude(t, gap) + amplitude_switch(zero, a_off, a_on),
                            (zero, gap))
    assert all(bool(torch.isfinite(v)) for v in g)


def test_the_switched_hamiltonian_has_the_closed_form_in_the_switch_region():
    """Two configurations at gap 2 with coupling 0.2: importance e = r_0 0.2 / sqrt(2^2 + 0.2^2) = 0.0995 (r_0 = 1 -
    1e-9: E_ref is the soft minimum), in the switch region; the lowest level is 1 - sqrt(1 + (sigma t)^2); dF/dt through sigma
    against finite differences."""
    from trace_redox.diabatic import canonical_free_energy, smoothstep, switched_hamiltonian
    configs = torch.tensor([[0, 0], [1, -1]])
    K = torch.tensor([1, 1])

    def F(t):
        T = torch.zeros(2, 2, dtype=torch.float64)
        T = T.index_put((torch.tensor([0, 1]), torch.tensor([1, 0])), torch.stack([t, t]))
        E = torch.tensor([0.0, 2.0], dtype=torch.float64)
        return canonical_free_energy(switched_hamiltonian(E, configs, K, T, 0.05, 0.2, KT), KT)

    t = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    value = F(t)
    x0 = KT * math.log1p(math.exp(-2.0 / KT))                  # E_0 - E_ref (E_ref is the soft minimum)
    x1 = 2.0 + x0
    e = math.exp(-0.5 * x0 / KT) * 0.2 / math.sqrt(x1 * x1 + 0.04)    # r_0 g(t, x_1)
    s = float(smoothstep(torch.tensor((math.log(e) - math.log(0.05)) / math.log(4.0))))
    assert 0.0 < s < 1.0
    lo, hi = 1.0 - math.sqrt(1.0 + (s * 0.2) ** 2), 1.0 + math.sqrt(1.0 + (s * 0.2) ** 2)
    reference = lo - KT * math.log1p(math.exp(-(hi - lo) / KT))
    assert abs(float(value) - reference) <= 1e-12
    (grad,) = torch.autograd.grad(value, t)
    h = 1e-6
    fd = (float(F(torch.tensor(0.2 + h, dtype=torch.float64))) - float(F(torch.tensor(0.2 - h, dtype=torch.float64)))) / (2 * h)
    assert abs(float(grad) - fd) <= 1e-8


def _donor_ring(n, t_entry=2.0, t_hop=0.3):
    """Acceptors A1, A2 (cheap to take an electron) and n donors D_k (a hole costs 10 + 0.1 k eV); only D_1 couples
    to the acceptors (t_entry), and holes hop between every pair of donors (t_hop): a ligand-to-metal transfer
    followed by a ligand-hole band, as O -> Au on MgO (measured 2026-10-18: |T| 2.0 eV at 10.5 eV, O-O 0.14-0.37)."""
    U = n + 2
    states = torch.arange(-1, 2, dtype=torch.float64)
    e_hat = torch.zeros(U, 3, dtype=torch.float64)
    e_hat[:2, 2] = 100.0                                       # acceptors: no hole
    e_hat[2:, 0] = 100.0                                       # donors: no extra electron
    e_hat[2:, 2] = 10.0 + 0.1 * torch.arange(n, dtype=torch.float64)
    T = torch.zeros(U, U, dtype=torch.float64)
    T[2, 0] = T[0, 2] = T[2, 1] = T[1, 2] = t_entry
    T[2:, 2:] = t_hop
    T.fill_diagonal_(0.0)
    return states, e_hat, torch.zeros(U, dtype=torch.float64), torch.zeros(U, U, dtype=torch.float64), \
        torch.ones(U, dtype=torch.long), T


@pytest.mark.parametrize("n", [6, 12])
def test_the_importance_is_the_path_product_so_virtual_hole_hops_stay_out(n):
    """r(D1+ A-) = g(2, 10) = 0.196: in.  A hole hop from there reaches r = 0.196 g(0.3, 10.1 + ...) ~ 0.006 < a_off:
    out, as is every double transfer.  P = {ground, D1+ A1-, D1+ A2-} whatever n (pairwise amplitudes between the
    near-degenerate hole configurations, 0.3 / 0.1, would have taken in all n holes and their pairs)."""
    from trace_redox.diabatic import configuration_energies, configuration_importance
    states, e_hat, b, W, K, T = _donor_ring(n)
    F, P, H = _free_energy(states, e_hat, b, W, K, T, 0, selection="amplitude")
    assert P.shape[0] == 3
    E = configuration_energies(P, e_hat, states, b, W)
    r, _, _ = configuration_importance(E, P, K, T, KT)
    x = E - (-KT * torch.logsumexp(-E / KT, 0))
    one = (P[:, 0] == -1).nonzero().flatten()
    assert abs(float(r[one[0]]) - 2.0 / math.sqrt(float(x[one[0]]) ** 2 + 4.0)) <= 1e-14
    ev = torch.linalg.eigvalsh(H)
    assert abs(float(F) - float(-KT * torch.logsumexp(-ev / KT, 0))) <= 1e-12


def test_the_closure_refuses_beyond_its_cap():
    from trace_redox.diabatic import ConfigurationSearchError
    states, e_hat, b, W, K, T = _random_problem(7, 3, t_scale=3.0)
    with pytest.raises(ConfigurationSearchError, match="cap"):
        _free_energy(states, e_hat, b, W, K, T, 0, selection="amplitude", amplitudes=(1e-6, 2e-6), cap=20)


def test_without_couplings_it_is_the_window_selection():
    from trace_redox.diabatic import select_configurations, search_configurations
    states, e_hat, b, W, K, _ = _random_problem(6, 2)
    a = select_configurations(e_hat, states, b, W, K, 0, window=2.5, cap=1000, T=None)
    c, _ = search_configurations(e_hat, states, b, W, K, 0, window=2.5, cap=1000)
    assert sorted(map(tuple, a.tolist())) == sorted(map(tuple, c.tolist()))


# ---- the module ---------------------------------------------------------------------------------------------

Z6 = torch.tensor([8, 1, 1, 11, 17, 6])


def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", couplings=True, **redox)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), config)
    with torch.no_grad():                         # couplings up to ~1.2 eV, the Au2 size (init ~0.01 eV)
        module.coupling.pair[-1].weight.mul_(3.0)
    return module, heads


def test_module_forces_are_minus_the_gradient_with_the_amplitude_selection():
    module, heads = _module(diabatic_selection="amplitude")
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


def test_the_module_energy_does_not_depend_on_the_window_beyond_decoupled_blocks():
    """Same weights, windows 2.5 and 6 eV: the amplitude selection's energies agree to the Boltzmann weight of the
    decoupled blocks (bounded here by e^-(2.5 - 2 max|T|) / kT per configuration)."""
    import dataclasses
    module, heads = _module(diabatic_selection="amplitude")
    pos, h, ei = cluster(6, heads.irreps_in.dim, 3)
    vec, length = edges(pos, ei)
    values = []
    for w in (2.5, 6.0):
        module.redox = dataclasses.replace(module.redox, diabatic_window=w)
        values.append(float(module(h, pos, ei, vec, length, data={"z": Z6, "total_charge": 1.0})[0]))
    assert abs(values[1] - values[0]) <= 1e-6


def test_the_configuration_is_validated():
    from trace_redox import RedoxConfig
    with pytest.raises(ValueError, match="diabatic_selection"):
        RedoxConfig(fragment_ensemble="diabatic", diabatic_selection="energy")
    with pytest.raises(ValueError, match="selection_amplitude"):
        RedoxConfig(fragment_ensemble="diabatic", diabatic_selection="amplitude", selection_amplitude_off=0.2,
                    selection_amplitude_on=0.05)
    with pytest.raises(ValueError, match="selection_amplitude"):
        RedoxConfig(fragment_ensemble="diabatic", diabatic_selection="amplitude", selection_amplitude_off=0.0)
    with pytest.raises(ValueError, match="diabatic_selection"):
        RedoxConfig(fragment_ensemble="exact", diabatic_selection="amplitude")
