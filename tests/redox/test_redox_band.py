"""A band unit for delocalised carriers (``RedoxConfig(carrier_band=True)``, trace_redox SPEC 20; theory note
2026-10-16 §10).

The extra-electron states of cation-frame atoms are carried by one band unit: the lowest orbital phi_1 of

    h = diag(eps) - tau,   eps_v = E(f - e_v) - E(f),   tau_vv' = V_sss(R) x factor x switch,   V_sss = 1.40 hbar^2/(m R^2)

holding m = 0, 1, 2 electrons: E_b(1) = lambda_1 (exact one-carrier tight binding), E_b(2) = 2 lambda_1 + J_11.
phi_1 comes from a detached eigh refined by two differentiable Newton steps on the bordered eigen-equation, so its
derivatives are exact to third order (Newton converges quadratically) even where higher states are degenerate.
"""
import math

import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)


def _ring(n, t=1.0, seed=0):
    """A symmetric matrix with exactly degenerate excited states (a ring) plus a small random diagonal that keeps
    the ground state non-degenerate."""
    g = torch.Generator().manual_seed(seed)
    h = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        h[i, (i + 1) % n] = h[(i + 1) % n, i] = -t
    return h


def test_the_eigenpair_matches_eigh_and_hellmann_feynman():
    from trace_redox.band import lowest_eigenpair
    h0 = _ring(8)
    A = torch.randn(8, 8, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    A = 0.5 * (A + A.T)
    x = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    h = h0 + x * A
    lam, phi = lowest_eigenpair(h)
    ev, V = torch.linalg.eigh(h0)
    assert abs(float(lam) - float(ev[0])) <= 1e-12
    assert abs(abs(float(phi @ V[:, 0])) - 1.0) <= 1e-12
    assert bool((phi > 0).all())                                                # Perron-Frobenius sign
    (g,) = torch.autograd.grad(lam, x, create_graph=True)
    assert abs(float(g) - float(V[:, 0] @ A @ V[:, 0])) <= 1e-10                  # Hellmann-Feynman
    (g2,) = torch.autograd.grad(g, x)
    second = -2.0 * sum(float(V[:, j] @ A @ V[:, 0]) ** 2 / float(ev[j] - ev[0]) for j in range(1, 8))
    assert math.isfinite(float(g2)) and abs(float(g2) - second) <= 1e-8 * (1 + abs(second))


def test_the_eigenvector_derivative_is_exact():
    from trace_redox.band import lowest_eigenpair
    h0 = _ring(6) + torch.diag(torch.tensor([0.0, 0.1, -0.2, 0.05, 0.0, 0.3], dtype=torch.float64))
    A = torch.randn(6, 6, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    A = 0.5 * (A + A.T)

    def phi_of(x):
        return lowest_eigenpair(h0 + x * A)[1]

    x = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    p = phi_of(x)
    J = torch.stack([torch.autograd.grad(p[i], x, retain_graph=True)[0] for i in range(6)])
    eps = 1e-6
    fd = (phi_of(torch.tensor(eps, dtype=torch.float64)) - phi_of(torch.tensor(-eps, dtype=torch.float64))) / (2 * eps)
    assert float((J - fd).abs().max()) <= 1e-8


def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True, **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


def _rocksalt_cluster(dopant=None, au=False):
    """A 2x2x2 rocksalt MgO block (a = 4.212 A), optionally one Mg -> Al and an Au atop an O."""
    a = 4.212
    pos, z = [], []
    for i in range(3):
        for j in range(3):
            for k in range(2):
                pos.append([i * a / 2, j * a / 2, k * a / 2])
                z.append(12 if (i + j + k) % 2 == 0 else 8)
    if dopant is not None:
        z[0] = dopant
    if au:
        o = next(idx for idx, zz in enumerate(z) if zz == 8 and pos[idx][2] > 1.0)
        pos.append([pos[o][0], pos[o][1], pos[o][2] + 2.2])
        z.append(79)
    return torch.tensor(z), torch.tensor(pos, dtype=torch.float64)


def _run(module, heads, z, pos, Q, capture=False):
    import trace_redox.module as module_code
    h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    return module(h, pos, ei, vec, length, data={"z": z, "total_charge": float(Q)})


def test_no_carriers_no_change():
    """Formal MgO block with its frame charges summing to Q: the band unit is empty; same energy."""
    on, heads = _module(carrier_band=True)
    off, _ = _module(carrier_band=False)
    off.load_state_dict(on.state_dict(), strict=False)
    z, pos = _rocksalt_cluster()
    from trace_redox.constants import formal_frame
    Q = int(formal_frame(z).sum())
    e_on, r_on = _run(on, heads, z, pos, Q)
    e_off, _ = _run(off, heads, z, pos, Q)
    assert abs(float(e_on - e_off)) <= 1e-9 * (1 + abs(float(e_off)))
    assert abs(float(r_on.free_charge.sum()) - Q) <= 1e-9


def test_one_band_electron_is_tight_binding_and_localises_without_hopping():
    """The band unit's one-electron energy (relative to the frame) is the lowest eigenvalue of diag(eps) - tau
    built from the atom-level configuration energies; with the hopping factor at its floor it is min_v eps_v."""
    from trace_redox.band import BandUnit
    module, heads = _module(carrier_band=True)
    z, pos = _rocksalt_cluster(dopant=13)
    from trace_redox.constants import formal_frame
    frames = formal_frame(z)
    captured = module.band_diagnostics(*_inputs(module, heads, z, pos, int(frames.sum()) - 1))
    eps, tau, lam1 = captured["eps"], captured["tau"], captured["lambda1"]
    ev = torch.linalg.eigvalsh(torch.diag(eps) - tau)
    assert abs(float(lam1) - float(ev[0])) <= 1e-10
    with torch.no_grad():
        module.band.log_factor.fill_(-50.0)                       # factor -> 1/20: weak hopping
    captured = module.band_diagnostics(*_inputs(module, heads, z, pos, int(frames.sum()) - 1))
    eps, tau, lam1 = captured["eps"], captured["tau"], captured["lambda1"]
    assert float(tau.abs().max()) <= 1.40 * 7.619964 / 2.9 ** 2 / 20 * 1.0001
    assert abs(float(lam1) - float(torch.linalg.eigvalsh(torch.diag(eps) - tau)[0])) <= 1e-10
    assert float(lam1) <= float(eps.min()) + 1e-12


def _inputs(module, heads, z, pos, Q):
    h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    return h, pos, ei, vec, length, {"z": z, "total_charge": float(Q)}


def test_band_charge_is_conserved_and_lands_on_cation_sites():
    module, heads = _module(carrier_band=True)
    z, pos = _rocksalt_cluster(dopant=13, au=True)
    from trace_redox.constants import formal_frame
    Q = int(formal_frame(z).sum()) - 1                            # one excess electron
    energy, result = _run(module, heads, z, pos, Q)
    assert math.isfinite(float(energy))
    assert abs(float(result.free_charge.sum()) - Q) <= 1e-9
    excess = result.free_charge - formal_frame(z).to(result.free_charge)
    cation = formal_frame(z) > 0
    assert float(excess[~cation & (z != 79)].abs().max()) <= 1e-9       # anions stay at their frames


def test_band_forces_are_minus_the_gradient():
    module, heads = _module(carrier_band=True)
    z, pos0 = _rocksalt_cluster(dopant=13, au=True)
    from trace_redox.constants import formal_frame
    Q = int(formal_frame(z).sum()) - 1
    h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos0)

    def energy(p):
        vec, length = edges(p, ei)
        return module(h, p, ei, vec, length, data={"z": z, "total_charge": float(Q)})[0]

    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    for i, axis in [(0, 0), (1, 2), (len(z) - 1, 2), (5, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-7 * (1.0 + abs(fd))


def test_it_needs_formal_diabatic():
    from trace_redox import RedoxConfig
    with pytest.raises(ValueError, match="carrier_band"):
        RedoxConfig(carrier_band=True)


def test_a_structure_without_cation_sites_has_no_band_unit():
    """An isolated Au atom (the polarisability-prior structure) or a neutral-frame cluster has no cation-frame site:
    the band unit is absent and the energy equals carrier_band=False exactly (measured failure 2026-10-17: step 12
    stopped on the Au-atom prior at its first training call)."""
    on, heads = _module(carrier_band=True)
    off, _ = _module(carrier_band=False)
    off.load_state_dict(on.state_dict(), strict=False)
    for z, pos in ((torch.tensor([79]), torch.zeros(1, 3, dtype=torch.float64)),
                   (torch.tensor([79, 79]), torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=torch.float64))):
        h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
        ei = neighbour_list(pos)
        vec, length = edges(pos, ei)
        kw = dict(data={"z": z, "total_charge": 0.0})
        assert float(on(h, pos, ei, vec, length, **kw)[0]) == float(off(h, pos, ei, vec, length, **kw)[0])


def test_band_initial_factor_sets_the_hopping_factor_and_scales_tau():
    """RedoxConfig(band_initial_factor=f) starts the band unit at hopping factor f (2026-10-22).  Motive, measured:
    PBE's conduction-band mass of bulk MgO at the 4G lattice constant is 0.322-0.330 m_e (tools/electrostatics/
    mgo_pbe.py), and the nearest-neighbour s band has m* = hbar^2 / (2 f tau_H a^2), so f = 0.55, where Harrison's
    f = 1 gives 0.179 m_e.  The default stays exactly 1, and tau scales exactly linearly with f."""
    from trace_redox.constants import formal_frame
    default, heads = _module(carrier_band=True)
    assert float(default.band.factor) == 1.0
    module, _ = _module(carrier_band=True, band_initial_factor=0.55)
    eps = torch.finfo(torch.float64).eps
    # f = exp(ln 20 tanh(atanh(ln f / ln 20))): a few roundings
    assert abs(float(module.band.factor) - 0.55) <= 16 * eps * 0.55
    z, pos = _rocksalt_cluster(dopant=13)
    Q = int(formal_frame(z).sum()) - 1
    t1 = default.band_diagnostics(*_inputs(default, heads, z, pos, Q))["tau"]
    t2 = module.band_diagnostics(*_inputs(module, heads, z, pos, Q))["tau"]
    ratio = float(module.band.factor) / float(default.band.factor)
    assert float(t1.abs().max()) > 0.1
    assert float((t2 - ratio * t1).abs().max()) <= 8 * eps * float(t1.abs().max())


@pytest.mark.parametrize("f", [0.0, -1.0, 1.0 / 20.0, 20.0, 25.0, float("nan"), float("inf")])
def test_band_initial_factor_is_inside_the_bounded_range(f):
    """The factor is exp(ln 20 tanh s): open range (1/20, 20); anything else is refused at configuration."""
    with pytest.raises(ValueError, match="band_initial_factor"):
        _module(carrier_band=True, band_initial_factor=f)
