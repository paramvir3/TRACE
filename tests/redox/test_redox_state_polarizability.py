"""State-dependent polarisabilities in the configuration energies (``RedoxConfig(state_polarizability=True)``; SPEC 28).

The bound sector gives atom j one dipole stiffness P_jj, its isolated polarisability ell^2 P_jj^-1, the same in every
charge state.  A charge state's own polarisability differs by a large factor: measured with the ion-density library's
method (X2C-PBE0/ANO-RCC, finite field), alpha(Au-) / alpha(Au0) = 3.39 and alpha(Au+) / alpha(Au0) = 0.37 (bohr^3:
123.7 / 36.50 / 13.45; Au0 exp 36 +- 3).  With alpha_j(n) = rho_j(n) alpha_j(f), K(n) = K + sum_j P_j^T Delta_j P_j,
Delta_j = P_jj (1 / rho_j - 1), and by Woodbury, exactly for every configuration,

    dE(n) = 1/2 y_S(n)^T Delta_S (I + C_SS Delta_S)^-1 y_S(n),   y(n) = y0 - K^-1 A^T q(n),   C = K^-1 on the dipoles,

S the atoms away from their frame.  One atom with no neighbours: dE = -1/2 (alpha(n) - alpha(f)) E_loc^2.  Measured on
frozen step-12 weights: 0.1-0.9 eV on the Au- configurations of the T5 references.  Many-body (no exact pair split), so
it is added on the selected set (the importance search uses the energies without it; the selection is detached and
the correction continuous, so F stays continuous).

Tests: the correction equals the brute-force re-solve with the modified K on random problems; the isolated-dipole
limit; no change with all ratios 1; forces -grad E in the module; validated.
"""
import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)


class _Op:
    def __init__(self, K):
        self.K = K

    def solve(self, R):
        return torch.linalg.solve(self.K, R)

    def matrix(self):
        return self.K


class _Elim:
    def __init__(self, K, A, y0, nb):
        self.operator, self.A, self.y0, self.n_bonds = _Op(K), A, y0, nb


def _problem(seed, n=4, nb=3, coupled=True):
    """Random K with couplings weak against the dipole blocks (X scaled by 0.3), so that every K(n) of the tests'
    ratios stays positive definite, as a physical bound sector must (checked in each test; with unscaled X some K(n)
    were indefinite, measured 2026-10-19: the brute-force comparison was then at a saddle)."""
    g = torch.Generator().manual_seed(seed)
    m = nb + 3 * n
    X = 0.3 * torch.randn(m, m, generator=g, dtype=torch.float64)
    K = (X @ X.T if coupled else torch.diag(torch.rand(m, generator=g, dtype=torch.float64))) + m * torch.eye(m)
    A = torch.randn(n, m, generator=g, dtype=torch.float64)
    y0 = torch.randn(m, generator=g, dtype=torch.float64)
    return _Elim(K, A, y0, nb)


@pytest.mark.parametrize("seed", range(4))
def test_the_correction_is_the_exact_resolve(seed):
    from trace_redox.polarizability import state_polarizability_corrections
    elim = _problem(seed)
    n = 4
    frames = torch.tensor([0, 0, 2, -2])
    configs = torch.tensor([[0, 0, 2, -2], [-1, 1, 2, -2], [0, -1, 1, -1], [1, 0, 2, -1]])
    states = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float64)
    ratio = lambda j, s: {(1, -1): 3.4, (1, 1): 0.37, (2, 1): 5.0, (3, -1): 0.44}.get((j, s), 1.0)
    table = torch.ones(n, states.numel(), dtype=torch.float64)
    for j in range(n):
        for k, s in enumerate(states.tolist()):
            table[j, k] = ratio(j, int(s))
    dE = state_polarizability_corrections(configs, elim, frames, table, states)
    K, A, y0, nb = elim.operator.K, elim.A, elim.y0, elim.n_bonds
    g = -K @ y0
    for c in range(configs.shape[0]):
        q = configs[c].to(torch.float64)
        s = g + A.T @ q
        Kn = K.clone()
        for j in range(n):
            r = ratio(j, int(configs[c, j]))
            if int(configs[c, j]) != int(frames[j]):
                sl = slice(nb + 3 * j, nb + 3 * j + 3)
                Kn[sl, sl] = K[sl, sl] / r
        assert float(torch.linalg.eigvalsh(Kn).min()) > 0.0                # a physical (convex) bound sector
        exact = -0.5 * s @ torch.linalg.solve(Kn, s) + 0.5 * s @ torch.linalg.solve(K, s)
        assert abs(float(dE[c]) - float(exact)) <= 1e-12 * (1.0 + abs(float(exact)))
    assert float(dE[0]) == 0.0                                     # the frame configuration: no change


def test_the_isolated_dipole_limit():
    """K block-diagonal (no coupling between sites): dE = -1/2 (rho - 1) alpha E^2 for the one changed atom, with
    alpha = ell^2 / P_jj and the local field E = ell y / alpha (here ell = 1)."""
    from trace_redox.polarizability import state_polarizability_corrections
    elim = _problem(7, n=2, nb=1, coupled=False)
    states = torch.tensor([-1, 0, 1], dtype=torch.float64)
    table = torch.tensor([[3.0, 1.0, 0.5], [1.0, 1.0, 1.0]], dtype=torch.float64)
    frames = torch.tensor([0, 0])
    configs = torch.tensor([[-1, 0], [1, 0]])
    dE = state_polarizability_corrections(configs, elim, frames, table, states)
    K, A, y0, nb = elim.operator.K, elim.A, elim.y0, elim.n_bonds
    for c, rho in ((0, 3.0), (1, 0.5)):
        y = y0 - torch.linalg.solve(K, A.T @ configs[c].to(torch.float64))
        P = torch.diagonal(K)[nb:nb + 3]
        alpha = 1.0 / P                                          # per component
        E = y[nb:nb + 3] / alpha
        expect = float((-0.5 * (rho - 1.0) * alpha * E * E).sum())
        assert abs(float(dE[c]) - expect) <= 1e-12 * (1.0 + abs(expect))


def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True, **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


Z6 = torch.tensor([8, 79, 79, 12, 13, 8])
POS = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0],
                    [2.0, 2.1, 0.2]], dtype=torch.float64)


def _energy(module, heads, pos):
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    return module(h, pos, ei, vec, length, data={"z": Z6, "total_charge": 0.0})[0]


def test_unit_ratios_change_nothing(monkeypatch):
    import trace_redox.polarizability as P
    plain, heads = _module()
    on, _ = _module(state_polarizability=True)
    assert abs(float(_energy(on, heads, POS)) - float(_energy(plain, heads, POS))) > 1e-6
    monkeypatch.setattr(P, "ratio_table", lambda species, states: torch.ones(int(species.numel()), int(states.numel()),
                                                                          dtype=torch.float64))
    assert float(_energy(on, heads, POS)) == float(_energy(plain, heads, POS))


def test_module_forces_are_minus_the_gradient_with_state_polarizabilities():
    module, heads = _module(state_polarizability=True, carrier_band=True)
    pos = POS.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(_energy(module, heads, pos), pos)
    for i, axis in [(0, 2), (1, 2), (3, 0), (5, 1)]:
        plus, minus = POS.clone(), POS.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(_energy(module, heads, plus)) - float(_energy(module, heads, minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-6 * (1.0 + abs(fd))


def test_it_is_validated():
    from trace_redox import RedoxConfig
    assert RedoxConfig().state_polarizability is False
    with pytest.raises(ValueError, match="state_polarizability"):
        RedoxConfig(state_polarizability=True)                    # needs the diabatic formal ensemble


@pytest.mark.parametrize("seed", range(3))
def test_the_correction_is_exact_together_with_carrier_induction(seed):
    """With SPEC 26 on, the source of configuration n is s'(n) = g + A^T q + sum_u Q_u v_u(s_u); the SPEC 28
    correction must use the moments of s': dE = -1/2 s'^T K(n)^-1 s' + 1/2 s'^T K^-1 s' exactly."""
    from trace_redox.polarizability import state_polarizability_corrections
    elim = _problem(10 + seed)
    n, S = 4, 5
    states = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float64)
    frames = torch.tensor([0, 0, 2, -2])
    g = torch.Generator().manual_seed(20 + seed)
    m = int(elim.y0.shape[0])
    V = torch.randn(m, n * S, generator=g, dtype=torch.float64)
    Q = states.unsqueeze(0) - frames.to(torch.float64).unsqueeze(1)
    V = V * (Q.reshape(1, -1) != 0)
    N = torch.linalg.solve(elim.operator.K, V)
    ratio = lambda j, s: {(1, -1): 3.4, (1, 1): 0.37, (2, 1): 5.0, (3, -1): 0.44}.get((j, s), 1.0)
    table = torch.tensor([[ratio(j, int(s)) for s in states.tolist()] for j in range(n)], dtype=torch.float64)
    configs = torch.tensor([[-1, 1, 2, -2], [0, -1, 1, -1], [1, 0, 2, -1]])
    dE = state_polarizability_corrections(configs, elim, frames, table, states, induction=(N, Q))
    K, A, y0, nb = elim.operator.K, elim.A, elim.y0, elim.n_bonds
    g0 = -K @ y0
    for c in range(configs.shape[0]):
        q = configs[c].to(torch.float64)
        k = (configs[c] - int(states[0])).tolist()
        s = g0 + A.T @ q + sum(Q[u, k[u]] * V[:, u * S + k[u]] for u in range(n))
        Kn = K.clone()
        for j in range(n):
            if int(configs[c, j]) != int(frames[j]):
                sl = slice(nb + 3 * j, nb + 3 * j + 3)
                Kn[sl, sl] = K[sl, sl] / ratio(j, int(configs[c, j]))
        assert float(torch.linalg.eigvalsh(Kn).min()) > 0.0                # a physical (convex) bound sector
        exact = -0.5 * s @ torch.linalg.solve(Kn, s) + 0.5 * s @ torch.linalg.solve(K, s)
        assert abs(float(dE[c]) - float(exact)) <= 1e-12 * (1.0 + abs(float(exact)))


def test_module_forces_with_the_step_17_stack():
    """Penetration + Gordon-Kim Pauli + carrier induction + state polarisabilities + band: forces -grad E."""
    import os
    from trace_redox.penetration import DATA
    if not os.path.exists(DATA):
        pytest.skip("library not built")
    module, heads = _module(state_polarizability=True, carrier_band=True, carrier_penetration=True,
                            carrier_overlap="gordon_kim", exclusion=False, carrier_induction=True)
    pos = POS.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(_energy(module, heads, pos), pos)
    for i, axis in [(0, 2), (1, 2), (2, 1), (4, 0)]:
        plus, minus = POS.clone(), POS.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(_energy(module, heads, plus)) - float(_energy(module, heads, minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-6 * (1.0 + abs(fd))


def _soft_problem(seed, n=4, nb=3, floor=1.0):
    """K = X X^T + floor I: dipole blocks only a little stiffer than their couplings, so a large ratio (a soft
    charge state such as Mg+ at 70.7 x Mg2+) can make K(n) indefinite."""
    g = torch.Generator().manual_seed(100 + seed)
    m = nb + 3 * n
    X = torch.randn(m, m, generator=g, dtype=torch.float64)
    K = X @ X.T + floor * torch.eye(m, dtype=torch.float64)
    return _Elim(K, torch.randn(n, m, generator=g, dtype=torch.float64), torch.randn(m, generator=g, dtype=torch.float64), nb)


def test_the_inertia_criterion_is_the_spectrum_of_the_configurations_stiffness():
    """SPEC 28: K(n) = K + P_S^T Delta_S P_S must stay positive definite, or the bound sector of configuration n has no
    minimum (a polarisation catastrophe) and Woodbury's value is a saddle's.  By Haynsworth inertia additivity on
    [[K, P_S^T], [P_S, -Delta_S^-1]] with K positive definite: K(n) > 0 iff Delta_S^-1 + C_SS has exactly as many
    negative eigenvalues as Delta_S.  Checked against eigvalsh(K(n)) on random problems with softening and
    stiffening ratios; both outcomes occur."""
    from trace_redox.polarizability import stiffness_defects
    n, nb = 4, 3
    frames = torch.tensor([0, 0, 2, -2])
    states = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float64)
    configs = torch.tensor([[0, 0, 2, -2], [-1, 1, 2, -2], [0, -1, 1, -1], [1, 0, 2, -1], [-1, -1, 1, -1]])
    outcomes = set()
    for seed in range(6):
        for soft in (1.5, 4.0, 100.0):
            elim = _soft_problem(seed)
            ratio = {(0, -1): soft, (1, 1): 0.37, (1, -1): soft, (2, 1): soft, (3, -1): 0.44}
            table = torch.tensor([[ratio.get((j, int(s)), 1.0) for s in states.tolist()] for j in range(n)],
                                 dtype=torch.float64)
            defects = stiffness_defects(configs, elim, frames, table, states)
            K = elim.operator.K
            for c in range(configs.shape[0]):
                Kn = K.clone()
                for j in range(n):
                    if int(configs[c, j]) != int(frames[j]):
                        sl = slice(nb + 3 * j, nb + 3 * j + 3)
                        Kn[sl, sl] = K[sl, sl] / ratio.get((j, int(configs[c, j])), 1.0)
                exact = bool(torch.linalg.eigvalsh(Kn).min() <= 0.0)
                assert bool(defects[c]) == exact, (seed, soft, c)
                outcomes.add(exact)
    assert outcomes == {True, False}


def test_a_polarisation_catastrophe_is_refused():
    """A configuration whose K(n) is indefinite has no bound-sector minimum: the energy is refused, not returned."""
    from trace_redox.polarizability import state_polarizability_corrections
    elim = _soft_problem(0)
    frames = torch.tensor([0, 0, 2, -2])
    states = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float64)
    table = torch.ones(4, 5, dtype=torch.float64)
    table[0, 1] = table[1, 1] = 1000.0                                 # the -1 state of atoms 0 and 1, 1000 x softer
    configs = torch.tensor([[0, 0, 2, -2], [-1, -1, 2, 0]])
    with pytest.raises(ValueError, match="positive definite"):
        state_polarizability_corrections(configs, elim, frames, table, states)


def test_the_ratios_can_be_restricted_to_listed_elements(monkeypatch):
    """``state_polarizability_elements``: only the listed elements take their states' own polarisabilities; every
    other element keeps its frame's (rho = 1).  Motive (measured 2026-10-19, probe_stiffness_constructed.py): with
    the frame's damping width, free-ion ratios of reduced cations in an oxide (Mg+ 70.7, Mg0 150.5, Al2+ 53.9)
    make K(n) indefinite in 16 of 17 Mg+ O- / Mg0 configurations of trained bulk MgO and 21 of 22 Al2+ O-
    configurations of the doped slab.  The restricted energy equals the full one with the other rows set to 1."""
    import trace_redox.polarizability as P
    restricted, heads = _module(state_polarizability=True, state_polarizability_elements=(79,))
    full, _ = _module(state_polarizability=True)
    e_restricted = float(_energy(restricted, heads, POS))
    assert abs(e_restricted - float(_energy(full, heads, POS))) > 1e-6          # the restriction acts here
    original = P.ratio_table

    def only_au(species, states):
        table = original(species, states)
        return torch.where((species == 79).unsqueeze(1), table, torch.ones_like(table))

    monkeypatch.setattr(P, "ratio_table", only_au)
    assert float(_energy(full, heads, POS)) == e_restricted


def test_the_element_restriction_is_validated():
    from trace_redox import RedoxConfig
    base = dict(fragment_ensemble="diabatic", anchor="formal")
    assert RedoxConfig().state_polarizability_elements is None
    with pytest.raises(ValueError, match="state_polarizability_elements"):
        RedoxConfig(**base, state_polarizability_elements=(79,))                 # needs state_polarizability
    with pytest.raises(ValueError, match="state_polarizability_elements"):
        RedoxConfig(**base, state_polarizability=True, state_polarizability_elements=())
    with pytest.raises(ValueError, match="state_polarizability_elements"):
        RedoxConfig(**base, state_polarizability=True, state_polarizability_elements=(0,))
