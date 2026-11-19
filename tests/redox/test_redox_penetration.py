"""Charge penetration of charge-state densities (``RedoxConfig(carrier_penetration=True)``; trace_redox SPEC 23).

The formal model holds every free charge on a compact Gaussian.  Measured 2026-10-18 with X2C-PBE0 ion densities: at
2.2 A from O2- (Watson sphere), the exact electrostatics of overlapping ions differs from the model by +1.38 eV for an
Au+ carrier and -2.30 eV for an Au- carrier, so the model favours Au+ next to O2- over Au- by 3.7 eV too much (3.4-4.0
eV for Watson radii 1.0-1.4 A): the direction of the over-polarised upright Au2 on MgO (free charge +-0.47 against
Hirshfeld +-0.12).

SPEC 23 replaces, for every pair of atoms within 8 A, the model's Gaussian Coulomb of the carriers by the exact pair
electrostatics of first-principles charge-state densities, relative to the frames:

    Xi_ab(n_a, n_b; R) = s(R) { D_ab(n_a, n_b; R) + (n_a n_b - f_a f_b) [K / R - J^G_ab(R)] }
    D_ab = [E_es(n_a, n_b) - E_es(f_a, f_b)] - (n_a n_b - f_a f_b) K / R,   R D tabulated (Chebyshev, 96 terms)

Tests: the radial numerics against closed forms; the library's densities (electron counts, exact tails); the tables
against direct integration; zero at frames, zero beyond the cutoff, symmetric; periodic images counted once.
"""
import math

import numpy as np
import pytest
import torch
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

HARTREE = 27.211386245988
BOHR = 0.529177210903


def _library():
    from trace_redox.penetration import DATA, library
    import os
    if not os.path.exists(DATA):
        pytest.skip("trace_redox/data/ion_densities.npz not built")
    return library()


def test_the_radial_numerics_match_the_closed_forms():
    """1s Slater densities zeta^3/pi exp(-2 zeta r): (rho|rho)(R) = 1/R - e^{-2 zeta R}(1/R + 11 zeta/8 + 3 zeta^2 R/4 +
    zeta^3 R^2/6) and phi(R) = 1/R - e^{-2 zeta R}(zeta + 1/R) (Roothaan, J. Chem. Phys. 19, 1445 (1951)), from the
    Au 1s core (zeta 79) to an anion's valence (0.41), on the library grid.  The trapezoid rule on the log grid is
    second order; 12000 points give <= 3e-6 hartree."""
    from trace_redox.radial import RadialDensity
    r = np.exp(np.linspace(math.log(1e-7), math.log(45.0), 12000))
    for z in (79.0, 1.3, 0.41):
        a = RadialDensity(r, z ** 3 / np.pi * np.exp(-2 * z * r))
        for R in (0.5, 2.0, 4.2, 8.0):
            J = 1 / R - math.exp(-2 * z * R) * (1 / R + 11 * z / 8 + 3 * z * z * R / 4 + z ** 3 * R * R / 6)
            phi = 1 / R - math.exp(-2 * z * R) * (z + 1 / R)
            assert abs(a.coulomb(a, R) - J) <= 3e-6
            assert abs(a.phi_at(R) - phi) <= 3e-6


def test_the_library_densities_hold_their_electrons():
    """Every tabulated density integrates to its electron count Z - q (renormalised on the grid; the bare proton is
    zero), and every element of the library has its frame density."""
    import os
    from trace_redox.penetration import DATA
    from trace_redox.radial import electron_count
    if not os.path.exists(DATA):
        pytest.skip("library not built")
    data = np.load(DATA)
    r = data["grid"]
    keys = data["density_keys"].tolist()
    for (Z, q), rho in zip(keys, data["densities"]):
        assert abs(electron_count(r, rho) - (Z - q)) <= 1e-9 * max(Z - q, 1)
    for Z, f in data["frames"].tolist():
        assert [Z, f] in keys


def _library_densities():
    import os
    from trace_redox.penetration import DATA
    if not os.path.exists(DATA):
        pytest.skip("library not built")
    data = np.load(DATA)
    r = data["grid"]
    dens = {tuple(k): rho for k, rho in zip(data["density_keys"].tolist(), data["densities"])}
    frames = {int(z): int(f) for z, f in data["frames"].tolist()}
    return r, dens, frames


@pytest.mark.parametrize("pair", [(79, -1, 8, -2), (79, 1, 8, -2), (79, 1, 79, -1), (8, -1, 12, 2), (11, 0, 17, -1),
                                  (12, 1, 8, -2)])
def test_the_tables_reproduce_the_direct_penetration(pair):
    """The Chebyshev series against the cancellation-free difference route (trace_redox.radial.penetration_difference),
    at random distances.  Tolerance from the operation: a fixed 96-term series over a dynamic range of 1e5 (R D from
    ~100 eV A at 1 A for Au3+-Au3+ to 1e-3 at 7.5 A) and a shell average that interpolates G linearly on the log grid;
    measured: median worst residual 1e-4 eV A per type, worst 2.1e-3 at 1.17 A (2e-5 relative).  Asserted
    1e-4 + 1e-5 |R D|."""
    from trace_redox.radial import penetration_difference
    lib = _library()
    r, dens, frames = _library_densities()
    Za, na, Zb, nb = pair
    fa, fb = frames[Za], frames[Zb]
    key = (Za, na, Zb, nb) if Za <= Zb else (Zb, nb, Za, na)
    g = np.random.default_rng(0)
    R = 0.6 + 7.3 * g.random(12)
    got = lib.r_times_D(torch.full((12,), lib.index[key]), torch.as_tensor(R)).numpy()
    for k, RA in enumerate(R):
        D = penetration_difference(r, Za, dens[(Za, fa)], dens[(Za, na)], fa, Zb, dens[(Zb, fb)], dens[(Zb, nb)], fb,
                                   RA / BOHR) * HARTREE
        assert abs(got[k] - RA * D) <= 1e-4 + 1e-5 * abs(RA * D)


@pytest.mark.parametrize("pair", [(11, 0, 17, -1), (11, 1, 17, 0), (1, -1, 6, 0)])
def test_the_difference_route_is_the_total_energy_route_for_light_ions(pair):
    """The total-energy route E_es = Z_a Z_b / R - Z_a phi_b - Z_b phi_a + (rho_a | rho_b) agrees with the difference
    route.  Its conditioning sets the tolerance: it cancels Z_a Z_b / R against the clouds, so an electron count off by
    1e-6 relative (the trapezoid normalisation) moves it by ~ Z_a dN_b / R; measured on Na0-Cl- at 2.5 A, both routes
    converge under 2x and 4x grid refinement and stay 4.7e-5 eV apart (1e-4 asserted)."""
    from trace_redox.radial import RadialDensity, pair_electrostatic_energy, penetration_difference
    from trace_redox.penetration import K_E
    r, dens, frames = _library_densities()
    Za, na, Zb, nb = pair
    fa, fb = frames[Za], frames[Zb]
    rd = lambda Z, q: RadialDensity(r, dens[(Z, q)]) if Z - q > 0 else None
    for RA in (1.2, 2.5, 4.0):
        E = lambda qa, qb: pair_electrostatic_energy(Za, rd(Za, qa), Zb, rd(Zb, qb), RA / BOHR) * HARTREE
        total = (E(na, nb) - E(fa, fb)) - (na * nb - fa * fb) * K_E / RA
        diff = penetration_difference(r, Za, dens[(Za, fa)], dens[(Za, na)], fa, Zb, dens[(Zb, fb)], dens[(Zb, nb)], fb,
                                      RA / BOHR) * HARTREE
        assert abs(total - diff) <= 1e-4, (RA, total, diff)


def test_the_pair_term_is_zero_at_frames_beyond_the_cutoff_and_symmetric():
    from trace_redox.constants import formal_frame
    from trace_redox.diabatic import formal_ladder
    from trace_redox.penetration import R_CUT, penetration_pairs
    _library()
    z = torch.tensor([79, 8, 12, 79])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.2], [2.1, 0.0, 2.3], [0.0, 0.0, R_CUT + 0.5]])
    states, _, _, _ = formal_ladder(z, 3)
    frames = formal_frame(z)
    sigma = torch.tensor([1.11, 0.81, 0.56, 1.11])
    i, j, Xi = penetration_pairs(z, states, frames, sigma, pos)
    first = int(states[0])
    for p in range(int(i.numel())):
        a, b = int(i[p]), int(j[p])
        assert float(Xi[p, int(frames[a]) - first, int(frames[b]) - first]) == 0.0
        assert float(torch.linalg.norm(pos[a] - pos[b])) < R_CUT
    # the far Au (beyond the cutoff from atom 0) has no pair with atom 0
    assert not bool(((i == 0) & (j == 3)).any())
    # symmetry: swapping the order of the atoms transposes each pair's table
    perm = torch.tensor([1, 0, 2, 3])
    i2, j2, Xi2 = penetration_pairs(z[perm], states, frames[perm], sigma[perm], pos[perm])
    a = ((i == 0) & (j == 1)).nonzero().flatten()[0]
    b = ((i2 == 0) & (j2 == 1)).nonzero().flatten()[0]
    assert torch.allclose(Xi[a], Xi2[b].transpose(0, 1), atol=1e-12, rtol=0.0)


def test_the_measured_au_o_preference_is_reproduced_through_the_pair_term():
    """Au carrier at 2.2 A from O2-, widths 1.1114 / 0.8148 A (step 12 seed 42): Xi(Au-) - Xi(Au+) is the excess of the
    model's Au+ preference, measured 3.7 eV with the unaugmented basis (3.4-4.0 over Watson radii)."""
    from trace_redox.constants import formal_frame
    from trace_redox.diabatic import formal_ladder
    from trace_redox.penetration import penetration_pairs
    _library()
    z = torch.tensor([79, 8])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.2]])
    states, _, _, _ = formal_ladder(z, 3)
    i, j, Xi = penetration_pairs(z, states, formal_frame(z), torch.tensor([1.1114, 0.8148]), pos)
    k = {int(s): n for n, s in enumerate(states.tolist())}
    plus, minus = float(Xi[0, k[1], k[-2]]), float(Xi[0, k[-1], k[-2]])
    assert 3.0 <= plus - minus <= 4.5, (plus, minus)


def test_periodic_images_are_counted_once():
    from trace_redox.penetration import pairs_within
    g = torch.Generator().manual_seed(1)
    cell = torch.tensor([[8.6, 0.0, 0.0], [0.3, 8.9, 0.0], [0.0, 0.0, 9.4]])
    pos = torch.rand(7, 3, generator=g) @ cell
    i, j, R = pairs_within(pos, cell, (True, True, True), 8.0)
    found = sorted(zip(i.tolist(), j.tolist(), [round(float(x), 9) for x in R]))
    brute = []
    for a in range(7):
        for b in range(a + 1, 7):
            for n1 in range(-2, 3):
                for n2 in range(-2, 3):
                    for n3 in range(-2, 3):
                        T = torch.tensor([n1, n2, n3], dtype=torch.float64) @ cell
                        d = float(torch.linalg.norm(pos[b] + T - pos[a]))
                        if d < 8.0:
                            brute.append((a, b, round(d, 9)))
    assert found == sorted(brute)


# ---- the module and the band ---------------------------------------------------------------------------------

def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    from redox_helpers import make_heads
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True, **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


POS = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0],
                    [2.0, 2.1, 0.2]], dtype=torch.float64)
Z = torch.tensor([8, 79, 79, 12, 13, 8])


def _energy(module, heads, pos):
    from redox_helpers import edges, neighbour_list
    h = torch.randn(len(Z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    return module(h, pos, ei, vec, length, data={"z": Z, "total_charge": 0.0})


@pytest.mark.parametrize("band", [False, True])
def test_module_forces_are_minus_the_gradient_with_penetration(band):
    _library()
    module, heads = _module(carrier_penetration=True, carrier_band=band)
    pos = POS.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(_energy(module, heads, pos)[0], pos)
    reference, _ = _module(carrier_band=band)
    assert abs(float(_energy(module, heads, POS)[0]) - float(_energy(reference, heads, POS)[0])) > 1e-4
    for i, axis in [(0, 2), (1, 2), (3, 0), (5, 1)]:
        plus, minus = POS.clone(), POS.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(_energy(module, heads, plus)[0]) - float(_energy(module, heads, minus)[0])) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-6 * (1.0 + abs(fd))


def test_the_band_level_is_the_exact_one_carrier_energy_with_the_pair_terms():
    """eps_v of the band (SPEC 20) with SPEC 23 pairs = E(f - e_v) - E(f) of configuration_energies with those pairs."""
    from trace_redox.band import BandUnit
    from trace_redox.constants import formal_frame
    from trace_redox.diabatic import configuration_energies, formal_ladder
    from trace_redox.penetration import penetration_pairs
    _library()
    z = torch.tensor([8, 12, 8, 12, 13, 79])
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.1, 0.0, 0.0], [4.2, 0.0, 0.0], [0.0, 2.1, 0.0], [2.1, 2.1, 0.0],
                        [1.0, 1.0, 2.4]], dtype=torch.float64)
    states, E_Z, L, K = formal_ladder(z, 3)
    frames = formal_frame(z)
    g = torch.Generator().manual_seed(2)
    S0 = torch.full((6,), 9.0, dtype=torch.float64)
    e_hat = E_Z - 0.5 * S0.unsqueeze(1) * states * states
    A = torch.randn(6, 6, generator=g, dtype=torch.float64)
    W = torch.cdist(pos, pos).clamp_min(1e-9).reciprocal() * 14.4
    W = torch.where(torch.eye(6, dtype=torch.bool), S0.diag(), W) - 0.05 * (A @ A.T)
    W = 0.5 * (W + W.T)
    b = 0.3 * torch.randn(6, generator=g, dtype=torch.float64)
    pairs = penetration_pairs(z, states, frames, torch.tensor([0.81, 0.56, 0.81, 0.56, 0.44, 1.11]), pos)
    from redox_helpers import edges, neighbour_list
    ei = neighbour_list(pos)
    _, length = edges(pos, ei)
    aug = BandUnit().augment(e_hat, states, b, W, L, K, frames, None, ei, length, pairs=pairs)
    sites = aug["sites"]
    E0 = configuration_energies(frames.unsqueeze(0), e_hat, states, b, W, pairs=pairs)[0]
    for k, v in enumerate(sites.tolist()):
        n = frames.clone()
        n[v] -= 1
        exact = configuration_energies(n.unsqueeze(0), e_hat, states, b, W, pairs=pairs)[0] - E0
        assert abs(float(aug["eps"][k] - exact)) <= 1e-10 * (1 + abs(float(exact)))


def test_penetration_is_validated_and_off_by_default():
    from trace_redox import RedoxConfig
    assert RedoxConfig().carrier_penetration is False
    with pytest.raises(ValueError, match="carrier_penetration"):
        RedoxConfig(carrier_penetration=True)
    with pytest.raises(ValueError, match="carrier_penetration"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", carrier_penetration=True,
                    free_charge_model="core_carrier")


def _rocksalt(a=4.212, reps=1):
    """Conventional rock-salt MgO (8 atoms, a < R_CUT: every atom meets its own images) or a reps^3 supercell."""
    base = torch.tensor([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], dtype=torch.float64)
    frac = torch.cat([base, base + torch.tensor([0.5, 0.0, 0.0])]) % 1.0
    z = torch.tensor([12] * 4 + [8] * 4)
    shifts = torch.tensor([[i, j, k] for i in range(reps) for j in range(reps) for k in range(reps)], dtype=torch.float64)
    pos = torch.cat([(frac + s) * a for s in shifts])
    return z.repeat(reps ** 3), pos, torch.eye(3, dtype=torch.float64) * a * reps


def test_a_small_periodic_cell_is_its_supercell():
    """A cell shorter than the cutoff: an atom and its images are always in the same state, so the self-image pairs
    are the one-body term sum over half the lattice vectors of Xi_ii(|T|; s, s) (penetration_self_terms).  For a
    periodic configuration the 8-atom cell then reproduces its 2x2x2 supercell (no self-image within 8 A) per cell,
    for both overlap models.  (Before: refused, 'an atom meets its own periodic image'.)"""
    from trace_redox.constants import formal_frame
    from trace_redox.diabatic import configuration_energies, formal_ladder
    from trace_redox.penetration import penetration_pairs, penetration_self_terms
    _library()
    for model in ("electrostatic", "gordon_kim"):
        totals = []
        for reps in (1, 2):
            z, pos, cell = _rocksalt(reps=reps)
            states, _, _, _ = formal_ladder(z, 3)
            frames = formal_frame(z)
            sigma = torch.where(z == 12, 0.56, 0.81).to(torch.float64)
            pbc = (True, True, True)
            pairs = penetration_pairs(z, states, frames, sigma, pos, cell=cell, pbc=pbc, model=model)
            one = penetration_self_terms(z, states, frames, sigma, pos, cell=cell, pbc=pbc, model=model)
            config = frames.to(torch.float64).clone()
            config[0::8] = 1.0                                  # one Mg+ and one O- per conventional cell
            config[5::8] = -1.0
            n = int(z.numel())
            E = configuration_energies(config.unsqueeze(0), one, states, torch.zeros(n, dtype=torch.float64),
                                       torch.zeros(n, n, dtype=torch.float64), pairs=pairs)[0]
            totals.append(float(E) / reps ** 3)
            if reps == 2:
                assert float(one.abs().max()) == 0.0            # 8.42 A > R_CUT: no self-image term
        assert abs(totals[0]) > 1e-3                            # the term is not trivially zero
        assert abs(totals[0] - totals[1]) <= 1e-10 * (1.0 + abs(totals[1])), (model, totals)


def test_same_element_tables_are_exactly_exchange_symmetric():
    """Xi between two atoms of one element is symmetric under their exchange.  The builder fits (Z, n, Z, m) and
    (Z, m, Z, n) separately (measured asymmetry: electrostatic <= 7.8e-6 eV A, Pauli <= 2.4e-2 eV A, e.g. K(Mg+, Mg2+)
    30.66071 against 30.66047 Ha-scaled); the loaded library holds their mean, so the two are bitwise equal."""
    from trace_redox.penetration import PenetrationLibrary
    _library()
    lib = PenetrationLibrary()
    checked = 0
    for (za, na, zb, nb), t in lib.index.items():
        if za == zb and na != nb:
            u = lib.index[(za, nb, zb, na)]
            assert torch.equal(lib.coefficients[t], lib.coefficients[u])
            if lib.gk_coefficients is not None:
                assert torch.equal(lib.gk_coefficients[t], lib.gk_coefficients[u])
            checked += 1
    assert checked > 50


def test_the_module_runs_on_a_small_periodic_cell_with_exact_forces():
    """Rock-salt MgO, 8 atoms, a = 4.212 A < R_CUT, with penetration, Gordon-Kim Pauli and carrier induction (whose
    self-image columns cancel exactly in +-T pairs): refused before ('meets its own periodic image'); now finite, and
    the forces are minus the gradient."""
    from redox_helpers import lattice_graph, make_heads, periodic_edges
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    _library()
    z, pos0, cell = _rocksalt()
    pos0 = pos0 + torch.tensor([[0.05, -0.03, 0.02]] + [[0.0, 0.0, 0.0]] * 7, dtype=torch.float64)   # break symmetry
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True,
                         carrier_penetration=True, carrier_overlap="gordon_kim", carrier_induction=True)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), config)
    pbc = (True, True, True)
    ei, es = lattice_graph(pos0, cell, pbc, r_max=4.0, shells=1)
    h = torch.randn(8, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)

    def energy(p):
        vec, length = periodic_edges(p, cell, ei, es)
        return module(h, p, ei, vec, length, pbc=pbc, cell=cell, edge_shift=es,
                      data={"z": z, "total_charge": 0.0})[0]

    pos = pos0.clone().requires_grad_(True)
    E = energy(pos)
    assert torch.isfinite(E)
    (grad,) = torch.autograd.grad(E, pos)
    for i, axis in [(0, 0), (0, 2), (5, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-6 * (1.0 + abs(fd))


def test_pairs_within_is_double_backward_safe_on_a_periodic_cell():
    """The pair geometry must not put the zero-length self-pairs of the [n, n] image table through the norm: torch's
    norm backward at a zero vector is masked at first order but 0/0 at second order, which detect_anomaly flags in
    every force-loss step (it blinded the nanguard's retry, 2026-10-29).  The norm is taken on the selected pairs
    only: the double backward of a force-like loss through pairs_within is finite under anomaly detection."""
    import torch
    from trace_redox.penetration import pairs_within
    torch.manual_seed(0)
    cell = torch.eye(3, dtype=torch.float64) * 6.0
    pos = (torch.rand(5, 3, dtype=torch.float64) * 6.0).requires_grad_(True)
    with torch.autograd.set_detect_anomaly(True):
        i, j, R, vec = pairs_within(pos, cell, (True, True, True), 4.5, return_vectors=True, allow_self=True)
        y = (R ** 2).sum() + (vec ** 2).sum()
        g, = torch.autograd.grad(y, pos, create_graph=True)
        loss = (g ** 2).sum()
        loss.backward()
    assert pos.grad is not None and bool(torch.isfinite(pos.grad).all())
    assert int(i.numel()) > 0 and bool((R > 0).all())
