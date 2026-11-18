"""Exact isolated limits (trace_redox SPEC 24).

The formal ladder E_Z(n) holds the exact isolated-ion energies (NIST IE, EA).  The learned shifts that correct them for
the environment did not vanish without one: measured 2026-10-18 on isolated atoms, Au- +1.18 / Au+ +1.25 eV (AuMgO step
12 seed 42), Ag+ -1.36 eV (Ag3 step 9), Na- +0.19, Cl- -0.11 eV (NaCl step 9).  Every dissociation into ions then misses
its exact limit, e.g. the ionic-covalent crossing of Na + Cl at IE(Na) - EA(Cl) = K / R_c (R_c = 9.4 A).

SPEC 24 (NEW-TO-TRACE): the shift of unit u is gated by its environment,

    shift_u(n) = g_u Delta_u(n),   g_u = 1 - prod_{v != u} (1 - s(r_uv)),

with s the C^2 quintic switch of the charge-state head's cutoff: g = 0 for an isolated atom, g = 1 as soon as one
neighbour is inside the switch's onset, C^2 in between.  Then an isolated ion's energy is E_Z(n) exactly, whatever the
weights.  And the band unit (SPEC 20) can hold its two electrons whatever elements are present: the states run down to
-2 whenever there is a band (an isolated Mg was refused: "total charge 0 is outside the capacity [1, 3]").

Tests: an isolated atom's charge-state energies are the exact ladder with trained-size shifts; the Na + Cl ionic
configuration crosses the covalent one at the exact R_c; a single bonded neighbour leaves the shift whole; forces are
minus the gradient through the gate; off by default; an isolated Mg is neutral with the band and has its exact energy.
"""
import math

import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

K_E = 14.399645


def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, **redox)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), config)
    with torch.no_grad():                    # shifts of the trained size (~1 eV): the head's last layer is zero at init
        module.charge_states.shift_mlp[-1].weight.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(7))
        module.charge_states.shift_mlp[-1].bias.fill_(0.8)
    return module, heads


def _capture(module):
    import trace_redox.module as M
    store = {}
    original = M.RedoxElectrostatics._diabatic_solution

    def wrapped(self, e_hat, states, *args, **kwargs):
        store["e_hat"], store["states"] = e_hat.detach().clone(), states.detach().clone()
        return original(self, e_hat, states, *args, **kwargs)

    return store, wrapped


def _run(module, heads, z, pos, monkeypatch, store_wrapped):
    import trace_redox.module as M
    store, wrapped = store_wrapped
    monkeypatch.setattr(M.RedoxElectrostatics, "_diabatic_solution", wrapped)
    h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    ei = neighbour_list(pos, r_max=6.0) if len(z) > 1 else torch.zeros(2, 0, dtype=torch.long)
    vec, length = edges(pos, ei)
    out = module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
    return out, store


@pytest.mark.parametrize("Z", [79, 47, 11, 17, 8])
def test_an_isolated_atom_has_the_exact_ladder(Z, monkeypatch):
    """e_hat(n) + S0 n^2 / 2 = E_Z(n) for every state of an isolated atom (the shift is zero), and without the gate
    the same weights shift it."""
    import trace_redox.module as M
    from trace_redox.diabatic import formal_ladder
    z = torch.tensor([Z])
    pos = torch.zeros(1, 3, dtype=torch.float64)
    for gate, expect_exact in ((True, True), (False, False)):
        module, heads = _module(shift_gate=gate)
        _, store = _run(module, heads, z, pos, monkeypatch, _capture(module))
        states, E_Z, _, _ = formal_ladder(z, 3)
        S0 = float(M.gram_self_energy(module.free_widths(z).to(torch.float64))[0])
        worst = 0.0
        for k, s in enumerate(store["states"].tolist()):
            if torch.isfinite(E_Z[0, k]):
                worst = max(worst, abs(float(store["e_hat"][0, k]) + 0.5 * S0 * s * s - float(E_Z[0, k])))
        if expect_exact:
            assert worst <= 1e-12 * (1.0 + float(E_Z[0].abs()[torch.isfinite(E_Z[0])].max()))
        else:
            assert worst > 0.1


def test_one_bonded_neighbour_leaves_the_shift_whole():
    from trace_redox.module import shift_gate
    r = torch.tensor([[0.0, 2.4], [2.4, 0.0]], dtype=torch.float64)
    ei = torch.tensor([[0, 1], [1, 0]])
    g = shift_gate(2, ei, r[ei[0], ei[1]], r_cut=4.0)
    assert torch.equal(g, torch.ones(2, dtype=torch.float64))
    g_far = shift_gate(2, ei, torch.tensor([4.5, 4.5], dtype=torch.float64), r_cut=4.0)
    assert torch.equal(g_far, torch.zeros(2, dtype=torch.float64))


def test_the_nacl_ionic_curve_crosses_at_the_exact_radius(monkeypatch):
    """Na + Cl beyond every cutoff: E(Na+ Cl-) - E(Na0 Cl0) = IE(Na) - EA(Cl) - W_12(R), W_12 -> K/R (Gaussians far
    apart, the bound sector's induction ~ alpha K / R^4, 1e-3 eV at 9 A); the zero is R_c = K / (IE - EA) to the size of
    that induction (|dR| <= 0.05 A)."""
    from trace_redox.constants import ATOMIC_ELECTRON_AFFINITY_EV as EA, ATOMIC_IONIZATION_EV as IE
    from trace_redox.diabatic import configuration_energies
    module, heads = _module(shift_gate=True)
    z = torch.tensor([11, 17])
    R_c = K_E / (float(IE[11]) - float(EA[17]))
    import trace_redox.module as M
    store = {}
    original = M.RedoxElectrostatics._diabatic_solution

    def wrapped(self, e_hat, states, b, W, *args, **kwargs):
        store.update(e_hat=e_hat.detach(), states=states, b=b.detach(), W=W.detach())
        return original(self, e_hat, states, b, W, *args, **kwargs)

    monkeypatch.setattr(M.RedoxElectrostatics, "_diabatic_solution", wrapped)
    gaps = []
    for R in (R_c - 0.3, R_c + 0.3):
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, R]], dtype=torch.float64)
        h = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
        ei = torch.tensor([[0, 1], [1, 0]])
        vec, length = edges(pos, ei)
        module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
        configs = torch.tensor([[1, -1], [0, 0]])
        E = configuration_energies(configs, store["e_hat"], store["states"], store["b"], store["W"])
        gaps.append(float(E[0] - E[1]))
    # linear interpolation of the gap to its zero
    R_zero = (R_c - 0.3) + 0.6 * gaps[0] / (gaps[0] - gaps[1])
    assert gaps[0] < 0.0 < gaps[1] or gaps[0] > 0.0 > gaps[1]
    assert abs(R_zero - R_c) <= 0.05, (R_zero, R_c, gaps)


def test_forces_are_minus_the_gradient_through_the_gate():
    """Two atoms in the gate's switching shell (3.2-4 A): the gate's derivative enters the forces."""
    module, heads = _module(shift_gate=True, couplings=True)
    z = torch.tensor([79, 79, 8])
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.55], [2.2, 0.0, 0.0]], dtype=torch.float64)
    h = torch.randn(3, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos0, r_max=6.0)

    def energy(p):
        vec, length = edges(p, ei)
        return module(h, p, ei, vec, length, data={"z": z, "total_charge": 0.0})[0]

    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    for i, axis in [(1, 2), (0, 2), (2, 0)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-6 * (1.0 + abs(fd))


def test_it_is_off_by_default():
    from trace_redox import RedoxConfig
    assert RedoxConfig().shift_gate is False


def test_an_isolated_magnesium_is_neutral_with_the_band(monkeypatch):
    """Mg alone with carrier_band: Mg2+ + two band electrons = Mg0, where the band was refused before for want of the
    state -2.  The module's energy reference is the neutral atom (E_Z(0) = 0), so the exact free energy of an isolated
    neutral Mg is 0: the band's E_b(2) = 2 lambda_1 + J_11 reproduces E_Z(0) - E_Z(+2) exactly (SPEC 20), and the next
    configuration (Mg+ + one band electron is the same atom; Mg3+ + three is not allowed) lies IE-scale above, its
    Boltzmann weight e^-76."""
    module, heads = _module(shift_gate=True, carrier_band=True)
    z = torch.tensor([12])
    pos = torch.zeros(1, 3, dtype=torch.float64)
    h = torch.randn(1, heads.irreps_in.dim, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    ei = torch.zeros(2, 0, dtype=torch.long)
    vec, length = edges(pos, ei)
    energy, result = module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
    assert abs(float(result.free_charge.sum())) <= 1e-12
    assert abs(float(result.energy_free)) <= 1e-9


@pytest.mark.parametrize("state_polarizability", [False, True], ids=["alpha(frame)", "alpha(n) SPEC 28"])
def test_an_ion_pair_beyond_every_cutoff_is_the_exact_classical_asymptote(state_polarizability, monkeypatch):
    """Au + Au beyond the shift gate's and the heads' cutoffs (R = 10, 12 A): the ionic gap is exactly

        U(R) = E(Au+ Au-) - E(Au0 Au0) = IE - EA - K/R - 1/2 e0^T [[a_A^-1, -T], [-T, a_B^-1]]^-1 e0,

    T = K (3 z z^T - I) / R^3 the point dipole tensor, e0 = (0, 0, K/R^2, 0, 0, K/R^2) the ions' fields, a_A, a_B
    the full 3x3 polarisabilities of the isolated atoms (D_j^T K_jj^-1 D_j) -- with SPEC 28, of their states:
    rho(Au+) a_A and rho(Au-) a_B.  The Gaussian damping of the charges and moments is the only neglect, below 1e-14
    eV beyond 8 A (measured: 2.3e-14 at 8 A, 1.5e-8 at 6 A; a 2x2 collinear formula left an alpha_xz^2 / R^7 term).
    The induction itself is ~1e-3 eV here, so it is checked to ten digits."""
    import trace_redox.module as M
    from trace_redox.bound import _variational_indices
    from trace_redox.constants import ATOMIC_ELECTRON_AFFINITY_EV as EA, ATOMIC_IONIZATION_EV as IE, formal_frame
    from trace_redox.diabatic import configuration_energies
    from trace_redox.polarizability import ratio_table, state_polarizability_corrections
    module, heads = _module(shift_gate=True, state_polarizability=state_polarizability)
    store = {}
    solution, eliminate = M.RedoxElectrostatics._diabatic_solution, M.eliminate_bound_sector

    def wrapped_solution(self, e_hat, states, b, W, *args, **kwargs):
        store.update(e_hat=e_hat.detach(), states=states, b=b.detach(), W=W.detach())
        return solution(self, e_hat, states, b, W, *args, **kwargs)

    def wrapped_eliminate(*args, **kwargs):
        store["elim"] = eliminate(*args, **kwargs)
        return store["elim"]

    monkeypatch.setattr(M.RedoxElectrostatics, "_diabatic_solution", wrapped_solution)
    monkeypatch.setattr(M, "eliminate_bound_sector", wrapped_eliminate)
    z = torch.tensor([79, 79])
    configs = torch.tensor([[1, -1], [0, 0]])
    for R in (10.0, 12.0):
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, R]], dtype=torch.float64)
        h = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
        ei = torch.zeros(2, 0, dtype=torch.long)
        vec, length = edges(pos, ei)
        with torch.no_grad():
            module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
        elim, states = store["elim"], store["states"]
        E = configuration_energies(configs, store["e_hat"], states, store["b"], store["W"])
        rho = [1.0, 1.0]
        if state_polarizability:
            table = ratio_table(z, states)
            E = E + state_polarizability_corrections(configs, elim, formal_frame(z), table, states)
            first = int(round(float(states[0])))
            rho = [float(table[0, 1 - first]), float(table[1, -1 - first])]
            assert abs(rho[0] - 0.3686) < 1e-3 and abs(rho[1] - 3.389) < 1e-3          # Au+, Au- (SPEC 28 data)
        U = float(E[0] - E[1])
        K, nb = elim.operator.matrix(), int(elim.n_bonds)
        D = elim.assembly.dipole_derivative()[_variational_indices(elim.assembly)]
        alpha = []
        for j, r in enumerate(rho):
            rows = slice(nb + 3 * j, nb + 3 * j + 3)
            alpha.append(r * (D[rows].transpose(0, 1) @ torch.linalg.solve(K[rows, rows], D[rows])))
        zz = torch.zeros(3, 3, dtype=torch.float64)
        zz[2, 2] = 1.0
        T = K_E * (3.0 * zz - torch.eye(3, dtype=torch.float64)) / R ** 3
        A = torch.cat([torch.cat([torch.linalg.inv(alpha[0]), -T], 1), torch.cat([-T, torch.linalg.inv(alpha[1])], 1)], 0)
        e0 = torch.tensor([0.0, 0.0, K_E / R ** 2, 0.0, 0.0, K_E / R ** 2], dtype=torch.float64)
        induction = -0.5 * float(e0 @ torch.linalg.solve(A, e0))
        exact = float(IE[79]) - float(EA[79]) - K_E / R + induction
        assert abs(induction) > 1e-3
        assert abs(U - exact) <= 256 * torch.finfo(torch.float64).eps * abs(exact), (R, U, exact, U - exact)
