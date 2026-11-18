"""The whole model's electronic response with several configurations (trace_redox/response.py; SPEC 30).

The eps_inf and polarizability priors trained the bound sector's response alone and refused any structure with more
than one configuration.  Measured 2026-10-19: step 13F (an Au2 alpha prior that raised the couplings) crashed when
bulk MgO, the eps_inf prior's structure, reached 5 configurations.  The exact response of the ensemble is the field
Hessian of its free energy.  Eliminating the bound sector exactly at fixed configuration n in a uniform field F,

    E(n; F) = E(n; 0) - F . M(n) - 1/2 F^T alpha_b F,   M(n) = sum_i q_i(n) r_i + D^T y*(n),

alpha_b = D^T K^-1 D the same for every n (state-independent K), so

    alpha = alpha_b + alpha_ens,   alpha_ens = -d2 F_ens / dF2,   F_ens(F) = -kT ln Tr exp(-[H - F . diag M] / kT),

which holds the thermal term Var_p(M) / kT and the coherent charge transfer.  Periodic cells: sum q r is defined
modulo lattice vectors, so M is accumulated along the one-electron moves with minimal-image hop vectors (a band
carrier at the phi^2-weighted minimal-image centroid of its sites).  NEW-TO-TRACE: the response priors on the whole
configuration ensemble.

Tests: the two-level closed form; the decomposition equals the whole model's finite-field alpha of the total energy
(open boundaries, where the field is unambiguous); minimal-image hops; the eps_inf prior accepts a periodic structure
with several configurations and trains.
"""
import math
import warnings

import numpy as np
import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)


@pytest.mark.parametrize("t,kT", [(0.5, 0.1), (0.05, 0.1), (1.2, 0.025)])
def test_the_two_level_closed_form(t, kT):
    from trace_redox.response import ensemble_polarizability
    mu = 2.4
    H = torch.tensor([[0.0, -t], [-t, 0.0]], dtype=torch.float64)
    M = torch.tensor([[mu / 2, 0.0, 0.0], [-mu / 2, 0.0, 0.0]], dtype=torch.float64)
    alpha = ensemble_polarizability(H, M, kT)
    exact = (mu * mu / 4.0) * math.tanh(t / kT) / t
    assert abs(float(alpha[0, 0]) - exact) <= 1e-7 * exact, (float(alpha[0, 0]), exact)
    assert float(alpha[1, 1].abs()) <= 1e-12 and float(alpha[0, 1].abs()) <= 1e-12


def test_minimal_image_hops_across_a_periodic_boundary():
    from trace_redox.response import free_dipoles
    cell = torch.eye(3, dtype=torch.float64) * 10.0
    pos = torch.tensor([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]], dtype=torch.float64)   # 1 A apart through the boundary
    configs = torch.tensor([[0, 0], [1, -1]])
    M = free_dipoles(configs, torch.tensor([1, 1]), pos, cell, (True, True, True))
    # one electron from atom 0 to atom 1: dM = r_0 - r_1 by minimal image = +1 A along x
    assert torch.allclose(M[1] - M[0], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-12)
    M_open = free_dipoles(configs, torch.tensor([1, 1]), pos, None, None)
    assert torch.allclose(M_open[1] - M_open[0], torch.tensor([-9.0, 0.0, 0.0], dtype=torch.float64), atol=1e-12)


def _module(scale=0.3, **redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=1, couplings=True, **redox)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), config)
    module.record_ensemble = True                           # opt-in (train.py's response priors set it)
    with torch.no_grad():
        module.coupling.pair[-1].weight.normal_(0.0, scale, generator=torch.Generator().manual_seed(4))
    return module, heads


def test_the_record_is_opt_in():
    """Ordinary forwards build no ensemble record (no cost to energies and forces); the priors switch it on."""
    module, heads = _module()
    module.record_ensemble = False
    z = torch.tensor([11, 11])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.2, 3.0]], dtype=torch.float64)
    h_feat = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    _, result = module(h_feat, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
    assert result.iterations > 1 and result.ensemble is None


@pytest.mark.parametrize("redox,scale", [({}, 0.3), ({"diabatic_selection": "amplitude"}, 3.0)],
                         ids=["window", "amplitude-saturated"])
def test_the_decomposition_is_the_whole_models_finite_field_response(redox, scale):
    """Neutral Na2 at 3.0 A (open boundaries), several configurations.  alpha_b + alpha_ens from the module's
    ensemble record equals -d2E/dF2 of the module's total energy in an applied field (Richardson central differences),
    all three diagonal components.  Amplitude selection: couplings large enough that every pair is switched fully on
    (sigma = 1), so no switch derivative enters (see the next test for the switch band)."""
    from trace_redox.response import ensemble_polarizability
    from trace_electrostatics.constants import K_E
    module, heads = _module(scale=scale, **redox)
    z = torch.tensor([11, 11])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.2, 3.0]], dtype=torch.float64)
    h_feat = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)

    def run(field):
        return module(h_feat, pos, ei, vec, length, data={"z": z, "total_charge": 0.0, "external_field": field})

    energy0, result = run(torch.zeros(3, dtype=torch.float64))
    assert result.iterations > 1 and result.ensemble is not None
    alpha = result.ensemble["alpha_bound"] + ensemble_polarizability(result.ensemble["H"], result.ensemble["M"],
                                                                      result.ensemble["kT"])

    def fd(axis, h):
        e = torch.zeros(3, dtype=torch.float64)
        e[axis] = h
        return -(float(run(e)[0]) - 2.0 * float(energy0) + float(run(-e)[0])) / h ** 2

    for axis in range(3):
        richardson = (4.0 * fd(axis, 5e-4) - fd(axis, 1e-3)) / 3.0
        assert abs(float(alpha[axis, axis]) - richardson) <= 1e-5 * (1.0 + abs(richardson)), (axis, float(alpha[axis, axis]), richardson)
    assert float(alpha[2, 2]) > float(alpha[0, 0])           # the transfer along the bond adds to alpha_zz
    del K_E


def test_inside_the_switch_band_the_switch_leaks_into_the_whole_models_response():
    """SPEC 21's couplings are sigma(e) tau with e depending on the configuration energies, which shift with the field:
    with a pair inside the switch band (a_off < e < a_on) -d2E/dF2 of the model contains the switch's field derivatives,
    which are not physics.  Measured 2026-10-19 on this test model: whole-model alpha_zz -3.99 against the fixed-H
    ensemble's +2.36 (e^2 A^2 / eV).  On trained step-13 models no pair sat in the band and the two agree to 0.01 A^3
    (probe_switch_leak.py).  The response record (and so the priors) carries the fixed-H physics."""
    from trace_redox.response import ensemble_polarizability
    module, heads = _module(scale=0.3, diabatic_selection="amplitude")
    z = torch.tensor([11, 11])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.2, 3.0]], dtype=torch.float64)
    h_feat = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)

    def run(fz):
        return module(h_feat, pos, ei, vec, length,
                      data={"z": z, "total_charge": 0.0, "external_field": torch.tensor([0.0, 0.0, fz], dtype=torch.float64)})

    e0, result = run(0.0)
    fixed = float((result.ensemble["alpha_bound"] + ensemble_polarizability(result.ensemble["H"], result.ensemble["M"],
                                                                            result.ensemble["kT"]))[2, 2])
    h = 5e-4
    whole = -(float(run(h)[0]) - 2.0 * float(e0) + float(run(-h)[0])) / h ** 2
    assert fixed > 0.0                                       # the physical response is positive
    assert abs(whole - fixed) > 1.0                          # the switch band leaks a large spurious term


@pytest.mark.parametrize("seed", range(3))
def test_the_state_polarizability_response_pieces_are_exact(seed):
    """SPEC 28 + 30: with K(n) = K + P_S^T Delta P_S, the configuration's bound dipole D^T y(n) and polarisability
    D^T K(n)^-1 D, by Woodbury, against brute-force inversion of K(n) on random problems."""
    from trace_redox.polarizability import state_polarizability_response
    g = torch.Generator().manual_seed(30 + seed)
    n, nb = 4, 3
    m = nb + 3 * n
    X = 0.3 * torch.randn(m, m, generator=g, dtype=torch.float64)      # couplings weak enough that K(n) > 0
    K = X @ X.T + m * torch.eye(m, dtype=torch.float64)

    class Op:
        def solve(self, R):
            return torch.linalg.solve(K, R)

        def matrix(self):
            return K

    class Elim:
        operator, A, n_bonds = Op(), torch.randn(n, m, generator=g, dtype=torch.float64), nb
        y0 = torch.randn(m, generator=g, dtype=torch.float64)

    D = torch.randn(m, 3, generator=g, dtype=torch.float64)
    states = torch.tensor([-1, 0, 1], dtype=torch.float64)
    frames = torch.tensor([0, 0, 1, -1])
    ratio = {(0, -1): 3.4, (0, 1): 0.37, (1, -1): 2.0, (2, 0): 0.5, (3, 0): 1.8}
    table = torch.tensor([[ratio.get((j, int(s)), 1.0) for s in states.tolist()] for j in range(n)], dtype=torch.float64)
    configs = torch.tensor([[0, 0, 1, -1], [-1, 1, 1, -1], [1, -1, 0, 0]])
    dM, dalpha = state_polarizability_response(configs, Elim, frames, table, states, D)
    alpha_ref = D.T @ torch.linalg.solve(K, D)
    for c in range(configs.shape[0]):
        q = configs[c].to(torch.float64)
        s = -K @ Elim.y0 + Elim.A.T @ q
        Kn = K.clone()
        for j in range(n):
            if int(configs[c, j]) != int(frames[j]):
                sl = slice(nb + 3 * j, nb + 3 * j + 3)
                Kn[sl, sl] = K[sl, sl] / ratio.get((j, int(configs[c, j])), 1.0)
        assert float(torch.linalg.eigvalsh(Kn).min()) > 0.0                # a physical (convex) bound sector
        y_ref = -torch.linalg.solve(K, s)
        y_n = -torch.linalg.solve(Kn, s)
        assert torch.allclose(dM[c], D.T @ (y_n - y_ref), rtol=0, atol=1e-11)
        assert torch.allclose(dalpha[c], D.T @ torch.linalg.solve(Kn, D) - alpha_ref, rtol=0, atol=1e-11)
    assert float(dM[0].abs().max()) == 0.0 and float(dalpha[0].abs().max()) == 0.0     # the frame configuration


def test_the_decomposition_is_exact_with_state_polarizabilities():
    """Au2 at 2.5 A with SPEC 28 on (Au- 3.39, Au+ 0.37 of the neutral alpha): three configurations, the whole model's
    finite-field alpha of the total energy against alpha_b + alpha_ens with the per-configuration quadratic terms."""
    from trace_redox.response import ensemble_polarizability
    module, heads = _module(state_polarizability=True)
    z = torch.tensor([79, 79])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.2, 2.5]], dtype=torch.float64)
    h_feat = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)

    def run(field):
        return module(h_feat, pos, ei, vec, length, data={"z": z, "total_charge": 0.0, "external_field": field})

    energy0, result = run(torch.zeros(3, dtype=torch.float64))
    ens = result.ensemble
    assert result.iterations > 1 and ens["exact"], ens.get("reason")
    alpha = ens["alpha_bound"] + ensemble_polarizability(ens["H"], ens["M"], ens["kT"], quadratic=ens.get("alpha_shift"))
    without = ens["alpha_bound"] + ensemble_polarizability(ens["H"], ens["M"], ens["kT"])     # counterfactual
    assert float(ens["alpha_shift"].abs().max()) > 1e-2                  # the ions' own stiffness is exercised

    def fd(axis, h):
        e = torch.zeros(3, dtype=torch.float64)
        e[axis] = h
        return -(float(run(e)[0]) - 2.0 * float(energy0) + float(run(-e)[0])) / h ** 2

    misses = []
    for axis in range(3):
        richardson = (4.0 * fd(axis, 5e-4) - fd(axis, 1e-3)) / 3.0
        assert abs(float(alpha[axis, axis]) - richardson) <= 1e-5 * (1.0 + abs(richardson)), (axis, float(alpha[axis, axis]), richardson)
        misses.append(abs(float(without[axis, axis]) - richardson) / (1.0 + abs(richardson)))
    assert max(misses) > 1e-3, misses                                    # dropping the per-state terms is visible
