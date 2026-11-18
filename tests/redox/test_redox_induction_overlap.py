"""Induction by the carriers' true fields (``RedoxConfig(carrier_induction=True)``; trace_redox SPEC 26).

The bound sector's moments feel each free charge through the Gaussian kernel lambda_3 = P(3/2, R^2 / (s_f^2 + w^2)).
By Gauss, the true field of a spherical carrier is K Q lambda(R, w) / R^2 with lambda the fraction of the carrier
density, smeared by the moment's Gaussian, inside R (trace_redox.radial.field_factor).  Measured 2026-10-18 at an O
site (w = 0.84 A) at 2.2 A: lambda = 0.475 (Au-), 0.668 (Au+) against the model's 0.827, so the model overstates the
induction those carriers drive by 3x and 1.5x.

With the coupling of unit u in state s written A_u n_u + Q_u v_u(s) (v the correction column, only near u), the bound
energy of a configuration is exactly

    E_b(n) = E_b^G(n) + dF(n)^T (y0 - M n) - 1/2 dF(n)^T K^-1 dF(n),   dF(n) = sum_u Q_u v_u(s_u),  M = K^-1 A^T,

which splits exactly into one-body terms (into the ladder) and two-body terms (the SPEC 18.5 pair slot):

    E1_u(s) = Q c0_us - Q^2 v_us . M_u - 1/2 Q^2 v_us . N_us,          c0 = V^T (y0 - M f),  N = K^-1 V
    E2_uw(s, t) = -Q_us Q_wt (v_us . M_w + v_wt . M_u + v_us . N_wt).

Tests: the split reproduces the dense formula for every configuration of random problems; the field-factor tables
reproduce the direct integral; a Gaussian carrier gives no correction; forces are minus the gradient; validated.
"""
import itertools

import numpy as np
import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

BOHR = 0.529177210903


@pytest.mark.parametrize("seed", range(4))
def test_the_split_is_the_dense_bound_energy_for_every_configuration(seed):
    from trace_redox.induction import overlap_induction_terms
    from trace_redox.diabatic import configuration_energies
    g = torch.Generator().manual_seed(seed)
    n, m = 4, 9
    states = torch.arange(-1, 2, dtype=torch.float64)                # states -1, 0, +1
    S = int(states.numel())
    frames = torch.tensor([0, 0, 1, -1])
    X = torch.randn(m, m, generator=g, dtype=torch.float64)
    K = X @ X.T + m * torch.eye(m, dtype=torch.float64)
    A = torch.randn(n, m, generator=g, dtype=torch.float64)
    y0 = torch.randn(m, generator=g, dtype=torch.float64)
    V = torch.randn(m, n * S, generator=g, dtype=torch.float64)      # columns u * S + k
    Q = states.unsqueeze(0) - frames.unsqueeze(1).to(torch.float64)  # [n, S]
    V = V * (Q.reshape(1, -1) != 0)                                  # no correction at the frame state
    solve = lambda B: torch.linalg.solve(K, B)
    E1, (pi, pj, Xi) = overlap_induction_terms(V, solve(A.T), solve(V), y0, frames.to(torch.float64), Q)
    zero_e = torch.zeros(n, S, dtype=torch.float64)
    for config in itertools.product(range(S), repeat=n):
        idx = torch.tensor(config)
        nvec = states[idx]
        dF = sum(Q[u, idx[u]] * V[:, u * S + idx[u]] for u in range(n))
        dense = float(dF @ (y0 - solve(A.T) @ nvec) - 0.5 * dF @ solve(dF))
        split = float(configuration_energies(nvec.unsqueeze(0), E1, states, torch.zeros(n, dtype=torch.float64),
                                             torch.zeros(n, n, dtype=torch.float64), pairs=(pi, pj, Xi))[0])
        assert abs(split - dense) <= 1e-12 * (1.0 + abs(dense))
    assert float(E1[torch.arange(n), (frames - int(states[0]))].abs().max()) == 0.0   # zero at frames
    del zero_e


def _library():
    import os
    from trace_redox.penetration import DATA
    if not os.path.exists(DATA):
        pytest.skip("library not built")
    d = np.load(DATA)
    if "field_keys" not in d.files:
        pytest.skip("library without field-factor tables")
    return d


def test_the_field_factor_tables_reproduce_the_direct_integral():
    from trace_redox.induction import FieldFactorTable
    from trace_redox.radial import field_factor
    d = _library()
    table = FieldFactorTable()
    r = d["grid"]
    D = {tuple(k): v for k, v in zip(d["density_keys"].tolist(), d["densities"])}
    frames = {int(z): int(f) for z, f in d["frames"].tolist()}
    for Z, q in ((79, -1), (79, 1), (12, 1), (8, -1)):
        delta = D[(Z, q)] - D[(Z, frames[Z])]
        R = torch.tensor([1.3, 2.2, 3.1, 5.0], dtype=torch.float64)
        w = torch.tensor([0.5, 0.84, 1.2, 2.0], dtype=torch.float64)
        got = table.evaluate(torch.full((4,), table.index[(Z, q)]), R, w)
        for k in range(4):
            assert abs(float(got[k]) - field_factor(r, delta, float(R[k]) / BOHR, float(w[k]) / BOHR)) <= 2e-5


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


def test_a_gaussian_carrier_gives_no_correction(monkeypatch):
    """With the tables replaced by the model's own Gaussian factor every correction column is zero, so the energy is
    the uncorrected one to round-off."""
    import trace_redox.induction as I
    _library()
    plain, heads = _module(carrier_penetration=True, carrier_overlap="gordon_kim", exclusion=False)
    with_terms, _ = _module(carrier_penetration=True, carrier_overlap="gordon_kim", exclusion=False, carrier_induction=True)
    monkeypatch.setattr(I, "_carrier_factor", lambda table, types, R, w, gauss: gauss)
    assert abs(float(_energy(with_terms, heads, POS)) - float(_energy(plain, heads, POS))) <= 1e-10


def test_module_forces_are_minus_the_gradient_with_carrier_induction():
    _library()
    module, heads = _module(carrier_penetration=True, carrier_overlap="gordon_kim", exclusion=False,
                            carrier_induction=True, carrier_band=True)
    plain, _ = _module(carrier_penetration=True, carrier_overlap="gordon_kim", exclusion=False, carrier_band=True)
    assert abs(float(_energy(module, heads, POS)) - float(_energy(plain, heads, POS))) > 1e-6
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
    assert RedoxConfig().carrier_induction is False
    with pytest.raises(ValueError, match="carrier_induction"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", carrier_induction=True)


def test_the_force_loss_gradient_in_the_weights_is_exact():
    """Training fits forces, so it needs d/dtheta of |F|^2, a second derivative of E through the tables, gammainc and
    the K solves.  The autograd directional derivative along a random unit direction in weight space equals the
    Richardson-extrapolated central difference.  Measured 2026-10-18 (direction of norm 247): the plain difference
    misses by 0.21, 0.024, 2.1e-3, 2.4e-4, 2.1e-5 at h = 3e-4 ... 3e-6, i.e. exactly O(h^2) truncation, the same with
    the induction off; so the unit direction, h = 1e-3 and h / 2 combined to O(h^4), relative 1e-7."""
    _library()
    module, heads = _module(carrier_penetration=True, carrier_overlap="gordon_kim", exclusion=False,
                            carrier_induction=True, carrier_band=True)
    params = [p for p in module.parameters() if p.requires_grad]
    g = torch.Generator().manual_seed(11)
    direction = [torch.randn(p.shape, generator=g, dtype=p.dtype) for p in params]
    norm = sum(float((d * d).sum()) for d in direction) ** 0.5
    direction = [d / norm for d in direction]

    def loss():
        pos = POS.clone().requires_grad_(True)
        (F,) = torch.autograd.grad(_energy(module, heads, pos), pos, create_graph=True)
        return (F * F).sum()

    grads = torch.autograd.grad(loss(), params, allow_unused=True)
    analytic = sum(float((gr * d).sum()) for gr, d in zip(grads, direction) if gr is not None)
    assert all(torch.isfinite(gr).all() for gr in grads if gr is not None)
    def central(h):
        values = []
        for sign in (1.0, -1.0):
            with torch.no_grad():
                for p, d in zip(params, direction):
                    p.add_(sign * h * d)
            values.append(float(loss()))
            with torch.no_grad():
                for p, d in zip(params, direction):
                    p.sub_(sign * h * d)
        return (values[0] - values[1]) / (2 * h)

    fd = (4.0 * central(5e-4) - central(1e-3)) / 3.0
    assert abs(analytic) > 1e-6
    assert abs(fd - analytic) <= 1e-7 * (1.0 + abs(fd)), (fd, analytic)


# -- scalable route (2026-10-19): V on the dipole rows only, (K^-1)_dd in place of N = K^-1 V -----------------------

def _capture_elim(monkeypatch):
    import trace_redox.module as M
    store, original = {}, M.eliminate_bound_sector

    def wrapped(*args, **kwargs):
        store["elim"] = original(*args, **kwargs)
        return store["elim"]

    monkeypatch.setattr(M, "eliminate_bound_sector", wrapped)
    return store


@pytest.mark.parametrize("method", ["condensed", "dense"])
def test_the_dipole_block_of_the_inverse_is_exact(method, monkeypatch):
    """(K^-1)_dd from the condensed operator is S^-1's dipole block (block elimination with zero bond source: the
    Schur complement S = P - T^T Lam T is already factorised), from the dense one a solve; both against inv(K)."""
    from trace_redox.induction import dipole_block_inverse
    store = _capture_elim(monkeypatch)
    module, heads = _module(bound_elimination=method)
    _energy(module, heads, POS)
    elim = store["elim"]
    nb, n = int(elim.n_bonds), int(elim.A.shape[0])
    K = elim.operator.matrix()
    d = slice(nb, nb + 3 * n)
    exact = torch.linalg.inv(K)[d, d]
    got = dipole_block_inverse(elim.operator, nb, 3 * n)
    assert float((got - exact).abs().max()) <= 1e-11 * float(exact.abs().max())


@pytest.mark.parametrize("seed", range(3))
def test_the_dipole_row_terms_equal_the_dense_terms(seed):
    """V has no bond rows (the carriers' correction acts on the moments), so c0 = V^T (y0 - M f), C1 = V^T M and
    C2 = V^T K^-1 V need only the dipole rows of y0 and M and the dipole block of K^-1."""
    from trace_redox.induction import overlap_induction_terms, overlap_induction_terms_dipole
    g = torch.Generator().manual_seed(40 + seed)
    n, nb, S = 4, 5, 3
    m = nb + 3 * n
    X = torch.randn(m, m, generator=g, dtype=torch.float64)
    K = X @ X.T + m * torch.eye(m, dtype=torch.float64)
    A = torch.randn(n, m, generator=g, dtype=torch.float64)
    y0 = torch.randn(m, generator=g, dtype=torch.float64)
    states = torch.arange(-1, 2, dtype=torch.float64)
    frames = torch.tensor([0, 0, 1, -1])
    Q = states.unsqueeze(0) - frames.unsqueeze(1).to(torch.float64)
    Vp = torch.randn(3 * n, n * S, generator=g, dtype=torch.float64) * (Q.reshape(1, -1) != 0)
    V = torch.zeros(m, n * S, dtype=torch.float64)
    V[nb:] = Vp
    M = torch.linalg.solve(K, A.T)
    E1, (pi, pj, Xi) = overlap_induction_terms(V, M, torch.linalg.solve(K, V), y0, frames.to(torch.float64), Q)
    Cdd = torch.linalg.inv(K)[nb:, nb:]
    E1n, (pin, pjn, Xin) = overlap_induction_terms_dipole(Vp, M, Cdd, y0, frames.to(torch.float64), Q, nb)
    assert torch.equal(pi, pin) and torch.equal(pj, pjn)
    scale = 1.0 + float(E1.abs().max()) + float(Xi.abs().max())
    assert float((E1n - E1).abs().max()) <= 1e-12 * scale
    assert float((Xin - Xi).abs().max()) <= 1e-12 * scale


@pytest.mark.parametrize("extra", [{}, {"state_polarizability": True, "carrier_band": True}],
                         ids=["SPEC 26", "SPEC 26 + 28 + band"])
def test_the_module_equals_the_dense_route(extra, monkeypatch):
    """Energy, forces and the SPEC 30 record (with SPEC 28's per-configuration moments) through the scalable route
    equal the dense route (V on every row, N = K^-1 V) to round-off."""
    import trace_redox.induction as I
    kw = dict(carrier_penetration=True, carrier_overlap="gordon_kim", exclusion=False, carrier_induction=True, **extra)
    out = {}
    for dense in (True, False):
        monkeypatch.setattr(I, "USE_DENSE_REFERENCE", dense)
        module, heads = _module(**kw)
        module.record_ensemble = True
        pos = POS.clone().requires_grad_(True)
        h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        ei = neighbour_list(POS)
        vec, length = edges(pos, ei)
        energy, result = module(h, pos, ei, vec, length, data={"z": Z6, "total_charge": 0.0})
        (grad,) = torch.autograd.grad(energy, pos)
        out[dense] = (float(energy), grad.detach(), result.ensemble)
    (e_d, g_d, ens_d), (e_s, g_s, ens_s) = out[True], out[False]
    assert abs(e_s - e_d) <= 1e-12 * (1.0 + abs(e_d))
    assert float((g_s - g_d).abs().max()) <= 1e-10 * (1.0 + float(g_d.abs().max()))
    assert (ens_d is None) == (ens_s is None)
    if extra:                                     # the record is exercised: several configurations, SPEC 28 on
        assert ens_d is not None and ens_d["exact"] and "alpha_shift" in ens_d and int(ens_d["H"].shape[0]) > 1
    if ens_d is not None and ens_d.get("exact"):
        assert torch.allclose(ens_s["M"], ens_d["M"], rtol=0, atol=1e-10)
        if "alpha_shift" in ens_d:
            assert torch.allclose(ens_s["alpha_shift"], ens_d["alpha_shift"], rtol=0, atol=1e-10)
