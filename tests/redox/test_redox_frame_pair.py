"""SPEC 39 (2026-11-13): the frame pair's own interaction, for every element pair of the density library.

SPEC 23 (penetration) and SPEC 25 (Pauli) are differences against the frame pair and vanish at it, so the frame pair's
own interaction was no physics term: the learned backbone carried it where there are data, and the learned couplings
stood in for it below (benchmarks_4g README, 2026-11-13: the Au-Au wall).  The term is

    Phi_ab(R) = s(R) { D_ff(R) + f_a f_b [K / R - J^G_ab(R)] + c_ab(R) P_ab(R) },

D_ff = E_es(f_a, f_b) - f_a f_b K / R the exact electrostatics of the frozen frame densities beyond point charges,
J^G the model's own Gaussian kernel, P the non-additive Thomas-Fermi kinetic + Dirac exchange energy of the frame
densities and c_ab(R) = 1 or the first-principles ratio at the computed distances (trace_redox/frame_pair.py,
tools/electrostatics/build_frame_pair.py).  With SPEC 23 the electrostatics of ANY state pair is then
E_es(n_a, n_b) - n_a n_b J^G: SPEC 23 made absolute.

What is independent of the builder here: the electrostatics against PySCF's values and against the direct four-term
form; P against a quadrature written out in this file; the first-principles points are the file's own record of the
cached calculations (their agreement with the term is exact by construction at the knots and says the knots are read
right, no more).
"""
import dataclasses
import math
import os

import numpy as np
import pytest
import torch
from redox_helpers import cluster, edges, lattice_graph, make_heads, neighbour_list, periodic_edges
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

EPS = torch.finfo(torch.float64).eps
K_E = 14.399645
BOHR, HARTREE = 0.529177210903, 27.211386245988
#: the frame pairs whose frozen-density PBE interaction is on record (the Pauli build's cache; Au0-Au0 extended 2026-11-13)
MEASURED = {(1, 1), (8, 8), (8, 12), (8, 13), (8, 79), (11, 17), (12, 79), (17, 17), (47, 47), (79, 79)}


def _tensor(x):
    return torch.as_tensor(x, dtype=torch.float64)


def _densities():
    import trace_redox
    d = np.load(os.path.join(os.path.dirname(os.path.abspath(trace_redox.__file__)), "data", "ion_densities.npz"), allow_pickle=False)
    return d["grid"], {(int(k[0]), int(k[1])): d["densities"][i] for i, k in enumerate(d["density_keys"])}


def _direct_penetration(ka, kb, R):
    """E_es - q_a q_b K / R (eV) of two library states by the direct four-term form of trace_redox.radial
    (Z_a Z_b / R - Z_a phi_b - Z_b phi_a + (rho_a | rho_b)): a different expression from the builder's, accurate for
    light pairs (its four terms cancel at order Z_a Z_b / R: 1e-3 eV for these, 5e-2 for two Au)."""
    from trace_redox import radial
    r, dens = _densities()
    a, b = radial.RadialDensity(r, dens[ka]), radial.RadialDensity(r, dens[kb])
    return radial.pair_electrostatic_energy(ka[0], a, kb[0], b, R / BOHR) * HARTREE - ka[1] * kb[1] * K_E / R


def _electron_gas_overlap(ka, kb, R, n_mu=320, n_nu=140):
    """T_TF^nad + E_x^nad (eV) of two library densities, written out here: prolate-spheroidal quadrature of
    C_F [(a + b)^(5/3) - a^(5/3) - b^(5/3)] - C_X [(a + b)^(4/3) - a^(4/3) - b^(4/3)], C_F = (3/10)(3 pi^2)^(2/3),
    C_X = (3/4)(3/pi)^(1/3), the densities interpolated linearly on the radial grid (coarser than the builder's)."""
    r, dens = _densities()
    Rb = R / BOHR
    t = np.exp(np.linspace(np.log(1e-7), np.log(60.0 / Rb), n_mu))
    w_t = np.gradient(t)
    x, wx = np.polynomial.legendre.leggauss(n_nu)
    MU, NU = np.meshgrid(1.0 + t, x, indexing="ij")
    ra, rb = np.maximum(Rb * (MU + NU) / 2.0, 1e-12), np.maximum(Rb * (MU - NU) / 2.0, 1e-12)
    dV = 2.0 * np.pi * (Rb / 2.0) ** 3 * (MU * MU - NU * NU) * w_t[:, None] * wx[None, :]
    A = np.interp(ra, r, dens[ka], right=0.0)
    B = np.interp(rb, r, dens[kb], right=0.0)
    c_f, c_x = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0), 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
    f = c_f * ((A + B) ** (5.0 / 3.0) - A ** (5.0 / 3.0) - B ** (5.0 / 3.0)) - c_x * ((A + B) ** (4.0 / 3.0) - A ** (4.0 / 3.0) - B ** (4.0 / 3.0))
    return float(np.sum(f * dV)) * HARTREE


# ------------------------------------------------------------------------------------------------- the table

def test_the_table_holds_every_pair_of_the_library_in_its_formal_frames():
    """45 pairs of nine elements, each in the formal frame of trace_redox.constants; the ten pairs with first-principles
    points carry their ratio, every other pair the electron-gas value c = 1 at every distance."""
    from trace_redox.constants import formal_frame
    from trace_redox.frame_pair import frame_pair
    fp = frame_pair()
    assert fp.elements == [1, 6, 8, 11, 12, 13, 17, 47, 79]
    assert len(fp.pairs) == 45 and len(set(fp.pairs)) == 45
    for (a, b), (fa, fb) in zip(fp.pairs, fp.frames):
        assert a <= b
        assert [fa, fb] == formal_frame(torch.tensor([a, b])).tolist()
    assert {p for p in fp.pairs if fp.measured[p]} == MEASURED == set(fp.ratio)
    R = torch.linspace(0.6, 7.9, 60, dtype=torch.float64)
    for p in fp.pairs:
        _, bare = fp.pair_parts(p[0], p[1], R, scaled=False)
        _, scaled = fp.pair_parts(p[0], p[1], R)
        if fp.measured[p]:
            # two one-electron atoms are the electron gas's worst case: the spin-compensated gas of the summed density
            # against the high-spin product of two 1s orbitals (ratio 3.5-5.6 over 0.6-2.2 A); every other measured
            # pair lies within 0.5-1.6
            lo, hi = (3.0, 6.0) if p == (1, 1) else (0.5, 1.6)
            assert lo < fp.c[p] < hi, (p, fp.c[p])
        else:
            assert fp.c[p] == 1.0 and bool((scaled == bare).all()), p


def test_the_series_reproduce_their_direct_integrals():
    """The Chebyshev series against the builder's direct integrals at the stored check points (this guards the
    runtime's evaluation of the series, not the integrals): 2e-5 eV wherever a structure can reach (R >= 0.6 of the
    covalent radii's sum; stored worst 7e-6, the quadrature's own smoothness) and 1e-6 eV beyond 4 A."""
    from trace_redox.constants import COVALENT_RADII_A
    from trace_redox.frame_pair import frame_pair
    fp = frame_pair()
    R = fp.check_R
    for p in fp.pairs:
        es, bare = fp.pair_parts(p[0], p[1], R, scaled=False)
        _, scaled = fp.pair_parts(p[0], p[1], R)
        reach = R >= 0.6 * (COVALENT_RADII_A[p[0]] + COVALENT_RADII_A[p[1]])
        for value, direct in ((es, fp.check_es[p]), (bare, fp.check_pauli_bare[p]), (scaled, fp.check_pauli[p])):
            err = (value - direct).abs()
            assert float(err[reach].max()) <= 2e-5 * (1.0 + float(direct[reach].abs().max()) * 1e-2), (p, float(err[reach].max()))
            assert float(err[R >= 4.0].max()) <= 1e-6, (p, float(err[R >= 4.0].max()))


@pytest.mark.parametrize("pair,R,reference", [
    ((79, 79), 2.0, -16.4904), ((79, 79), 2.1, -12.3099), ((79, 79), 2.2, -9.2062),   # Au0-Au0, computed after the formula was fixed
    ((79, 79), 2.331, -6.3047), ((79, 79), 2.4472, -4.5129), ((79, 79), 2.5635, -3.2337), ((79, 79), 2.6798, -2.3203),
    ((12, 79), 1.8, -0.1161), ((12, 79), 2.2, -0.0064),              # Mg2+-Au0
    ((8, 12), 2.098, -27.5069 + 4.0 * K_E / 2.098),                  # O2--Mg2+: PySCF's E_es plus the point charges' 4 K / R
])
def test_the_frame_electrostatics_is_the_first_principles_one(pair, R, reference):
    """D_ff against the electrostatic interaction of the two frozen monomers computed by PySCF on the real densities
    ('vac_elst' of the cached records).  The references are printed to 1e-4 eV and the library's densities are the
    spherical averages of the same calculations: 5e-4 eV + 1e-3 relative (a term that vanished would fail every row)."""
    from trace_redox.frame_pair import frame_pair
    es, _ = frame_pair().pair_parts(pair[0], pair[1], _tensor([R]))
    assert abs(float(es[0]) - reference) <= 5e-4 + 1e-3 * abs(reference), (float(es[0]), reference)


@pytest.mark.parametrize("ka,kb,R", [((1, 0), (1, 0), 0.74), ((6, 0), (1, 0), 1.1), ((8, -2), (12, 2), 2.098),
                                     ((11, 1), (17, -1), 2.4), ((6, 0), (6, 0), 1.3)])
def test_the_frame_electrostatics_equals_the_direct_form_for_light_pairs(ka, kb, R):
    """The builder's cancellation-free D_ff against the direct four-term form, which is a different expression and is
    accurate for light pairs: 1e-3 eV (the direct form's own cancellation; the reviewers measured 2e-4)."""
    from trace_redox.frame_pair import frame_pair
    es, _ = frame_pair().pair_parts(ka[0], kb[0], _tensor([R]))
    direct = _direct_penetration(ka, kb, R)
    assert abs(float(es[0]) - direct) <= 1e-3, (float(es[0]), direct)


@pytest.mark.parametrize("ka,kb,R", [((1, 0), (1, 0), 0.74), ((11, 1), (17, -1), 2.4), ((47, 0), (47, 0), 2.6),
                                     ((79, 0), (79, 0), 2.331), ((8, -2), (12, 2), 2.1)])
def test_the_overlap_function_is_the_electron_gas_one(ka, kb, R):
    """P against a quadrature of T_TF^nad + E_x^nad written out in this file (other grid sizes, linear interpolation
    of the densities): 5e-3 relative.  It pins the absolute scale of the 39 pairs that carry c = 1."""
    from trace_redox.frame_pair import frame_pair
    _, bare = frame_pair().pair_parts(ka[0], kb[0], _tensor([R]), scaled=False)
    own = _electron_gas_overlap(ka, kb, R)
    assert abs(float(bare[0]) - own) <= 5e-3 * abs(own), (float(bare[0]), own)


@pytest.mark.parametrize("ka,kb,R", [((79, 1), (79, -1), 2.5), ((12, 1), (8, -1), 2.1), ((11, 0), (17, 0), 2.4),
                                     ((79, -1), (8, -2), 2.3), ((6, 1), (1, 0), 1.1)])
def test_spec23_is_made_absolute(ka, kb, R):
    """For a state pair (n_a, n_b): SPEC 23's tabulated difference D(n_a, n_b) plus the frame entry D_ff equals the
    E_es(n_a, n_b) - n_a n_b K / R of the two STATE densities by the direct four-term form: 5e-3 eV + the direct form's
    cancellation for the pairs with Au (5e-2 eV)."""
    from trace_redox.frame_pair import frame_pair
    from trace_redox.penetration import library
    lib = library()
    (Za, na), (Zb, nb) = (ka, kb) if ka[0] <= kb[0] else (kb, ka)
    t = torch.tensor([lib.index[(Za, na, Zb, nb)]])
    relative = float(lib.r_times_D(t, _tensor([R]))[0]) / R                       # [E_es(n) - E_es(f)] - (nn - ff) K / R
    frame, _ = frame_pair().pair_parts(Za, Zb, _tensor([R]))
    exact = _direct_penetration((Za, na), (Zb, nb), R)
    tol = 5e-2 if 79 in (Za, Zb) else 5e-3
    assert abs(relative + float(frame[0]) - exact) <= tol, (relative, float(frame[0]), exact)


def test_the_overlap_term_goes_through_every_first_principles_point():
    """c(R) P at each stored first-principles distance equals the stored frozen-density PBE value (all ten pairs, 74
    points): the ratio's knots are those points, so this holds to the P series' accuracy, 1e-5 relative +
    2e-6 eV.  Beyond its first and last point the ratio keeps its end slope (C^1 there) and saturates within
    ``ratio_tail``: it never leaves [c_end - |m_end| w, c_end + |m_end| w], so no trend is extrapolated further."""
    from trace_redox.frame_pair import frame_pair
    fp = frame_pair()
    w = fp.ratio_tail
    points = 0
    for p in sorted(MEASURED):
        R, E = fp.targets[p]
        _, pauli = fp.pair_parts(p[0], p[1], R)
        assert float((pauli - E).abs().max()) <= 2e-6 + 1e-5 * float(E.abs().max()), (p, (pauli - E).tolist())
        points += int(R.numel())
        i = fp.index[p]
        x, y, m = fp._knots[i]
        c = lambda values: fp._ratio(torch.full((len(values),), i, dtype=torch.long), _tensor(values))
        for end, sign in ((0, -1.0), (-1, +1.0)):
            x_e, y_e, m_e = float(x[end]), float(y[end]), float(m[end])
            far = c([x_e + sign * 6.0 * w])                                       # tanh(6) = 1 - 1.2e-5
            assert abs(float(far[0]) - (y_e + sign * m_e * w)) <= 2e-5 * abs(m_e) * w + 16 * EPS, (p, end)
            outside = c([x_e + sign * t for t in (0.01, 0.1, 0.5, 1.5)])
            assert bool(((outside - y_e).abs() <= abs(m_e) * w + 16 * EPS).all()), (p, end)
            if int(x.numel()) > 1:                                                # the slope is continuous across the end
                h = 1e-4
                inner = (float(c([x_e - sign * h])[0]) - y_e) / (-sign * h)
                outer = (float(c([x_e + sign * h])[0]) - y_e) / (sign * h)
                curvature = 4.0 * (abs(m_e) / w + float((y.max() - y.min()) / (x[-1] - x[0]) ** 2) * 6.0)
                assert abs(inner - outer) <= curvature * h + 1e-9, (p, end, inner, outer)
    assert points == 74                                              # ten pairs (2026-11-14)
    es, pauli = fp.pair_parts(79, 79, _tensor([2.0, 2.1, 2.2, 2.331]))
    whole = (es + pauli).tolist()
    for value, reference in zip(whole, (24.9064 - 16.4904, 18.0591 - 12.3099, 13.1676 - 9.2062, 8.7714 - 6.3047)):   # PySCF, both halves
        assert abs(value - reference) <= 2e-3, (value, reference)


def test_two_neutral_atoms_repel_and_ever_more_strongly_on_compression():
    """The physics the model lacked: the Au0-Au0 frame interaction is positive from 1.9 to 3.0 A, falls monotonically
    with distance, and its force at 2.089 A (the shortest dimer of the 4G test set, where DFT has +15.2 eV/A in all)
    is repulsive and larger than 15 eV/A; Ag0-Ag0 is positive and monotone too, and the Na+-Cl- term is monotone."""
    from trace_redox.frame_pair import frame_pair
    fp = frame_pair()
    R = torch.linspace(1.9, 3.0, 45, dtype=torch.float64)
    for (a, b), sig in (((79, 79), None), ((47, 47), None), ((11, 17), 1.0)):
        s2 = None if sig is None else _tensor([sig * sig]).expand(int(R.numel()))
        v = fp.pair_energy(a, b, R, s2, s2)
        if (a, b) != (11, 17):
            assert bool((v > 0).all()), (a, b, float(v.min()))
        assert bool((v[1:] < v[:-1]).all()), (a, b)
    x = _tensor([2.089]).requires_grad_(True)
    (g,) = torch.autograd.grad(fp.pair_energy(79, 79, x).sum(), x)
    assert -float(g) > 15.0, -float(g)
    x = torch.linspace(1.9, 3.0, 221, dtype=torch.float64).requires_grad_(True)       # and the force itself falls smoothly:
    (g,) = torch.autograd.grad(fp.pair_energy(79, 79, x).sum(), x)                    # no bump at the knots of c(R)
    assert bool((-g[1:] < -g[:-1]).all()), "the Au-Au frame force is not monotone in R"


def test_charged_frames_take_the_gaussian_correction_and_neutral_ones_do_not():
    """Phi = s (D_ff + c P + f_a f_b [K / R - K erf(R / sqrt(sigma_a^2 + sigma_b^2)) / R]): checked against the formula
    for Mg2+-O2-, and a pair with a neutral frame does not depend on the widths."""
    from trace_redox.frame_pair import frame_pair
    fp = frame_pair()
    R = _tensor([1.9, 2.1, 2.6, 3.4])
    sa, sb = _tensor([0.7 ** 2]).expand(4), _tensor([1.3 ** 2]).expand(4)
    es, pauli = fp.pair_parts(8, 12, R)
    excess = K_E / R - K_E * torch.erf(R / torch.sqrt(sa + sb)) / R
    expected = es + pauli + (-2.0) * (2.0) * excess                       # below the switch's start (6.5 A)
    got = fp.pair_energy(8, 12, R, sb, sa)
    assert torch.allclose(got, expected, rtol=0, atol=64 * EPS * float(expected.abs().max())), (got, expected)
    assert float(excess.min()) > 0.0 and bool(((got - es - pauli) < 0).all())      # the point ions attract more
    a = fp.pair_energy(12, 79, R, sa, sb)
    b = fp.pair_energy(12, 79, R, 4.0 * sa, 9.0 * sb)
    assert bool((a == b).all())
    with pytest.raises(ValueError, match="widths"):
        fp.pair_energy(8, 12, R)


# ------------------------------------------------------------------------------------------------ the pair sum

def test_isolated_limits_and_additivity():
    """One atom: zero.  Two atoms at and beyond the cutoff: zero.  An open cluster: the sum over its pairs."""
    from trace_redox.frame_pair import frame_pair
    fp = frame_pair()
    sigma = _tensor([1.0, 1.2, 1.5, 0.8])
    assert float(fp.energy(torch.tensor([79]), sigma[:1], torch.zeros(1, 3, dtype=torch.float64), None, None)) == 0.0
    far = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, fp.r_cut]])
    assert float(fp.energy(torch.tensor([79, 79]), sigma[:2], far, None, None)) == 0.0
    far[1, 2] = fp.r_cut + 0.3
    assert float(fp.energy(torch.tensor([79, 8]), sigma[:2], far, None, None)) == 0.0
    z = torch.tensor([79, 79, 8, 12])
    pos = _tensor([[0.0, 0.0, 0.0], [0.0, 0.3, 2.5], [2.2, 0.0, 0.1], [2.1, 2.2, 0.0]])
    total = float(fp.energy(z, sigma, pos, None, None))
    by_pairs, size = 0.0, 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            Rij = torch.linalg.norm(pos[i] - pos[j]).reshape(1)
            v = float(fp.pair_energy(int(z[i]), int(z[j]), Rij, sigma[i:i + 1] ** 2, sigma[j:j + 1] ** 2)[0])
            by_pairs, size = by_pairs + v, size + abs(v)
    assert abs(total - by_pairs) <= 64 * EPS * size, (total, by_pairs)             # six terms, summed in another order


def test_a_cell_and_its_supercell_have_the_same_energy_per_cell():
    """Extensivity under periodic boundaries: an 8-atom rock-salt MgO cell (4.2 A, every atom within the cutoff of its
    own images) and its 2 x 2 x 2 supercell, per cell.  This needs the self-image terms.  Tolerance: the two sums hold
    the same ~3e3 terms per cell (each up to ~1 eV, of both signs) in different orders."""
    from trace_redox.frame_pair import frame_pair
    fp = frame_pair()
    a = 4.2
    frac = _tensor([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]])
    z = torch.tensor([12, 12, 12, 12, 8, 8, 8, 8])
    wid = torch.where(z == 12, _tensor(0.75), _tensor(1.25))
    pos = frac * a + _tensor([[0.02, -0.01, 0.03]]) * torch.arange(8, dtype=torch.float64).unsqueeze(1) / 8.0
    cell = torch.eye(3, dtype=torch.float64) * a
    pbc = (True, True, True)
    one = float(fp.energy(z, wid, pos, cell, pbc))
    shifts = _tensor([[i, j, k] for i in range(2) for j in range(2) for k in range(2)]) * a
    big = float(fp.energy(z.repeat(8), wid.repeat(8), (pos.unsqueeze(0) + shifts.unsqueeze(1)).reshape(-1, 3), cell * 2.0, pbc))
    assert abs(big / 8.0 - one) <= 3e3 * 16 * EPS * (1.0 + abs(one)), (big / 8.0, one)
    assert abs(one) > 1.0                                             # 24 Mg-O contacts: not a vacuous comparison


def test_forces_and_the_cell_derivative_are_the_gradient():
    """A small sheared periodic cell with like and unlike pairs and self images.  Autograd forces against central
    differences of the energy, and the derivative under a homogeneous strain (which is where the self-image terms
    enter: they depend on the cell alone) against the same.  Step eps^(1/3); tolerance 1e3 eps^(2/3), the central
    difference's own error with the third derivative's size allowed for.  A force loss differentiates under anomaly
    detection."""
    from trace_redox.frame_pair import frame_pair
    fp = frame_pair()
    z = torch.tensor([79, 79, 8, 12, 12])
    sigma = _tensor([1.66, 1.66, 1.2, 0.7, 0.7])
    cell0 = _tensor([[6.3, 0.0, 0.0], [0.4, 6.1, 0.0], [0.0, 0.3, 6.6]])
    pos0 = _tensor([[0.0, 0.0, 0.0], [0.3, 0.2, 2.3], [2.1, 0.1, -0.2], [2.2, 2.0, 0.1], [4.3, 0.2, 3.1]])
    pbc = (True, True, True)
    h, tol = EPS ** (1.0 / 3.0), 1e3 * EPS ** (2.0 / 3.0)
    pos = pos0.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(fp.energy(z, sigma, pos, cell0, pbc), pos)
    for i, ax in ((0, 2), (1, 0), (2, 1), (4, 2)):
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, ax] += h
        minus[i, ax] -= h
        fd = (float(fp.energy(z, sigma, plus, cell0, pbc)) - float(fp.energy(z, sigma, minus, cell0, pbc))) / (2 * h)
        assert abs(fd - float(g[i, ax])) <= tol * (1.0 + abs(fd)), (i, ax, float(g[i, ax]), fd)
    strain = torch.zeros((), dtype=torch.float64, requires_grad=True)
    energy = lambda e: fp.energy(z, sigma, pos0 * (1.0 + e), cell0 * (1.0 + e), pbc)
    (dE,) = torch.autograd.grad(energy(strain), strain)
    fd = (float(energy(_tensor(h))) - float(energy(_tensor(-h)))) / (2 * h)
    assert abs(fd) > 1.0 and abs(fd - float(dE)) <= tol * (1.0 + abs(fd)), (float(dE), fd)
    only_images = lambda e: fp.energy(z[:1], sigma[:1], pos0[:1] * (1.0 + e), cell0 * (1.0 + e), pbc)   # one Au and its images
    (dS,) = torch.autograd.grad(only_images(strain), strain)
    fds = (float(only_images(_tensor(h))) - float(only_images(_tensor(-h)))) / (2 * h)
    assert abs(fds) > 1e-3 and abs(fds - float(dS)) <= tol * (1.0 + abs(fds)), (float(dS), fds)
    pos = pos0.clone().requires_grad_(True)
    with torch.autograd.set_detect_anomaly(True):
        (g,) = torch.autograd.grad(fp.energy(z, sigma, pos, cell0, pbc), pos, create_graph=True)
        (g ** 2).sum().backward()
    assert bool(torch.isfinite(pos.grad).all())


def test_the_hydrogen_pair_carries_its_first_principles_points():
    """H0-H0, the electron gas's worst case (the spin-compensated gas of the summed density against the high-spin
    product of two 1s orbitals): the overlap term the model evaluates, against the first-principles values written
    out here (frame_pair_wall.py, 2026-11-13; hh_frame_points.log), and the electrostatic term against the same
    calculations.  The two points left out of the table by the builder's rule (the electron-gas term is below 5e-3 eV
    there) are the measured limit of the ratio form: 0.045 eV low at 2.6 A, and of the wrong sign at 3.0 A."""
    from trace_redox.frame_pair import frame_pair
    fp = frame_pair()
    R = _tensor([0.6, 0.74, 0.9, 1.1, 1.4, 1.8, 2.2])
    pauli_fp = _tensor([13.9218, 10.0089, 6.8059, 4.1559, 1.9429, 0.6812, 0.2297])
    es_fp = _tensor([1.3483, 0.0227, -0.4410, -0.4740, -0.2925, -0.1142, -0.0392])
    es, pauli = fp.pair_parts(1, 1, R)
    assert float((pauli - pauli_fp).abs().max()) <= 1e-4 and float((es - es_fp).abs().max()) <= 1.5e-4
    assert fp.targets[(1, 1)][0].tolist() == R.tolist()
    _, bare = fp.pair_parts(1, 1, R, scaled=False)
    ratio = pauli / bare
    assert 3.5 < float(ratio.min()) and float(ratio.max()) < 5.7       # the default, c = 1, would be 3.5-5.6 times low
    _, tail = fp.pair_parts(1, 1, _tensor([2.6, 3.0]))
    assert abs(float(tail[0]) - 0.0289) <= 2e-4 and abs(float(tail[1]) + 0.0253) <= 2e-4      # first principles: +0.0735, +0.0213


def test_an_unknown_element_is_refused():
    """An element outside the density library raises: the energy would be silently incomplete."""
    from trace_redox.frame_pair import frame_pair
    fp = frame_pair()
    pos = _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.3], [2.2, 0.0, 0.0]])
    with pytest.raises(ValueError, match="no frame-pair table"):
        fp.energy(torch.tensor([79, 26, 8]), _tensor([1.0, 1.0, 1.0]), pos, None, None)
    with pytest.raises(ValueError, match="no frame-pair table"):
        fp.pair_parts(79, 26, _tensor([2.0]))


def test_pairs_are_found_for_unwrapped_positions():
    """The pair list scans its lattice shifts around the minimum-image difference (fixed 2026-11-13; before, around the
    raw difference, a pair was lost once two atoms were more than about one cell apart in unwrapped coordinates: found by
    the review of SPEC 39, inherited from SPEC 23's helper).  Two Au 3 A apart through the boundary of a 12 A cell: the
    energy does not depend on which image of the second atom is given; and in a sheared cell the list of pairs is the
    same, as a set of (i, j, R), after any atom is moved by lattice vectors."""
    from trace_redox.frame_pair import frame_pair
    from trace_redox.penetration import pairs_within
    fp = frame_pair()
    z, sigma = torch.tensor([79, 79]), _tensor([1.0, 1.0])
    cell = torch.eye(3, dtype=torch.float64) * 12.0
    pbc = (True, True, True)
    wrapped = float(fp.energy(z, sigma, _tensor([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]]), cell, pbc))
    unwrapped = float(fp.energy(z, sigma, _tensor([[1.0, 0.0, 0.0], [22.0, 0.0, 0.0]]), cell, pbc))
    assert wrapped > 0.1
    assert abs(unwrapped - wrapped) <= 64 * EPS * wrapped, (unwrapped, wrapped)
    g = torch.Generator().manual_seed(11)
    cell = _tensor([[6.3, 0.0, 0.0], [1.1, 5.9, 0.0], [-0.7, 0.9, 6.6]])
    pos = torch.rand(7, 3, generator=g, dtype=torch.float64) @ cell

    def listed(p, c=cell, periodic=pbc):
        i, j, R = pairs_within(p, c, periodic, 8.0, allow_self=True)[:3]
        return sorted((int(a), int(b), round(float(r), 9)) for a, b, r in zip(i, j, R))

    reference = listed(pos)
    assert len(reference) > 100
    moved = pos.clone()
    moved[2] += 2.0 * cell[0] - 1.0 * cell[1] + 3.0 * cell[2]
    moved[5] -= 4.0 * cell[1]
    assert listed(moved) == reference
    slab = (True, True, False)                                         # only the periodic directions are recentred
    ref_slab = listed(pos, periodic=slab)
    moved = pos.clone()
    moved[3] += 3.0 * cell[0] - 2.0 * cell[1]
    assert listed(moved, periodic=slab) == ref_slab
    # the slab against an independent enumeration (not this function): only the periodic directions are recentred.
    # Two atoms 8.5 A apart along the open direction of a 10 A cell are NOT a pair within 5 A (wrapped along z they
    # would be 1.5 A apart); the same list after in-plane lattice moves, and after a move along z it is another list.
    box = _tensor([[5.0, 0.0, 0.0], [1.0, 4.6, 0.0], [0.0, 0.0, 10.0]])
    atoms = _tensor([[0.3, 0.2, 0.5], [0.4, 0.3, 9.0], [2.6, 1.9, 1.4], [4.1, 3.3, 8.2]])

    def brute(p, r_cut=5.0):
        out = []
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                for n1 in range(-4, 5):
                    for n2 in range(-4, 5):
                        d = float(torch.linalg.norm(p[j] + n1 * box[0] + n2 * box[1] - p[i]))
                        if d < r_cut:
                            out.append((i, j, round(d, 9)))
        return sorted(out)

    def slab_list(p):
        i, j, R = pairs_within(p, box, slab, 5.0, allow_self=True)[:3]
        return sorted((int(a), int(b), round(float(r), 9)) for a, b, r in zip(i, j, R))

    want = brute(atoms)
    assert len(want) > 6 and not any(i == 0 and j == 1 and r < 2.0 for i, j, r in want)
    assert slab_list(atoms) == want
    shifted = atoms.clone()
    shifted[1] += 3.0 * box[0] - 2.0 * box[1]
    shifted[2] -= 1.0 * box[0]
    assert slab_list(shifted) == want
    lifted = atoms.clone()
    lifted[1] -= box[2]                                                # along the open direction: a different structure
    assert slab_list(lifted) == brute(lifted) != want
    lifted = pos.clone()
    lifted[3, 2] += 30.0                                               # along the open direction a displacement is real
    assert listed(lifted, periodic=slab) != ref_slab


# ------------------------------------------------------------------------- SPEC 33's in-situ wall and the pair sum

def _otop_site(height, d=2.1325):
    """The site SPEC 33's wall was measured on: an O, its four in-plane Mg and the Mg below, with an Au above the O."""
    z = torch.tensor([8, 12, 12, 12, 12, 12, 79])
    pos = _tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0], [-d, 0.0, 0.0], [0.0, d, 0.0], [0.0, -d, 0.0], [0.0, 0.0, -d], [0.0, 0.0, height]])
    return z, pos


@pytest.mark.parametrize("height", [1.9, 2.26, 2.9, 3.5])
def test_on_the_measured_site_the_frame_terms_add_up_to_the_in_situ_measurement(height):
    """SPEC 33's wall is the lone atom's interaction with its WHOLE site, not an Au-O pair function.  With frame_pair on
    the module adds (pair sum) + (wall) - (site reference); on the site the wall was measured on, the atom's part of
    the pair sum equals the site reference, so the frame terms add up to the wall's own table value.  (Without the
    reference the four Mg would be counted twice: 0.34 eV at 2.26 A.)"""
    from trace_redox.bond_pauli import bond_pauli_wall
    from trace_redox.frame_pair import frame_pair
    fp = frame_pair()
    z, pos = _otop_site(height)
    sigma = torch.where(z == 8, _tensor(1.25), _tensor(0.75))
    with_atom = fp.energy(z, sigma, pos, None, None)
    substrate = fp.energy(z[:-1], sigma[:-1], pos[:-1], None, None)
    reference = fp.site_reference(z, pos, None, None, "almo")
    atom_part = float(with_atom - substrate)
    assert abs(atom_part - float(reference)) <= 256 * EPS * (1.0 + abs(float(with_atom))), (atom_part, float(reference))
    for level in ("almo", "frz"):
        wall = float(bond_pauli_wall(level).energy(z, pos, None, None))
        table = float(bond_pauli_wall(level)._e1((79, 8), _tensor([height]))[0])
        total = atom_part + wall - float(fp.site_reference(z, pos, None, None, level))
        assert abs(total - table) <= 256 * EPS * (1.0 + abs(float(with_atom))), (level, total, table)
    four_mg = 4.0 * float(fp.pair_energy(79, 12, _tensor([math.hypot(height, 2.1325)]))[0])
    assert four_mg > (0.05 if height < 3.0 else 0.005)                # what would be counted twice without the reference


def test_beyond_the_walls_cutoff_the_pair_sum_is_whole():
    """The wall ends at 4.3 A (switched from 3.8 A).  Beyond it nothing is taken out: an Au-O pair at 5 A carries its
    whole frame interaction; in the switching range the reference fades with the wall's own switch."""
    from trace_redox.bond_pauli import R_CUT, R_ON, _switch
    from trace_redox.frame_pair import frame_pair
    fp = frame_pair()
    z, sigma = torch.tensor([79, 8]), _tensor([1.0, 1.0])
    at = lambda R: _tensor([[0.0, 0.0, 0.0], [0.0, 0.0, R]])
    assert float(fp.site_reference(z, at(5.0), None, None)) == 0.0
    pair = float(fp.energy(z, sigma, at(5.0), None, None))
    assert pair == float(fp.pair_energy(79, 8, _tensor([5.0]))[0]) and abs(pair) > 1e-3
    R = 0.5 * (R_ON + R_CUT)
    ref = float(fp.site_reference(z, at(R), None, None))
    d = 2.1325
    full = float((fp.pair_energy(79, 8, _tensor([R])) + 4.0 * fp.pair_energy(79, 12, _tensor([math.hypot(R, d)]))
                  + fp.pair_energy(79, 12, _tensor([R + d])))[0])
    assert abs(ref - float(_switch(_tensor([R]), R_ON, R_CUT)[0]) * full) <= 64 * EPS * (1.0 + abs(full)), (ref, full)


# -------------------------------------------------------------------------------------------------- the module

def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    cfg = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True, carrier_penetration=True,
                      carrier_overlap="gordon_kim", exclusion=False, **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), cfg), heads


Z6 = torch.tensor([8, 79, 79, 12, 13, 8])
POS6 = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0], [2.0, 2.1, 0.2]],
                    dtype=torch.float64)


def _open(**redox):
    module, heads = _module(**redox)
    ei = neighbour_list(POS6, r_max=5.3)
    vec, length = edges(POS6, ei)
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    E, result = module(h, POS6, ei, vec, length, data={"z": Z6, "total_charge": 0.0})[:2]
    return float(E), result, module


def test_the_module_adds_exactly_the_frame_interaction():
    """RedoxConfig(frame_pair=True) adds FramePair.energy with the module's own free-charge widths and nothing else: the
    populations are those of the model without it (the term is the same in every configuration).  With SPEC 33's wall
    on it adds the pair sum minus the wall's site reference.  The same in the cluster space with the Fermi band (the
    single model's route)."""
    from trace_redox import RedoxConfig
    from trace_redox.frame_pair import frame_pair
    assert RedoxConfig().frame_pair is False
    fp = frame_pair()
    for extra in ({}, {"diabatic_space": "clusters", "carrier_band": True, "band_form": "fermi"}):
        off, r_off, _ = _open(**extra)
        on, r_on, module = _open(frame_pair=True, **extra)
        term = float(fp.energy(Z6, r_on.sigma_free.detach(), POS6, None, None))
        assert abs(term) > 0.1
        # two module energies of |E| ~ 1e2..1e3 eV built from ~1e3 floating-point terms each
        assert abs(on - off - term) <= 1e3 * 16 * EPS * (1.0 + abs(off) + abs(term)), (extra, on, off, term)
        assert torch.equal(r_on.populations, r_off.populations)
        assert abs(module._frame_pair_last - term) <= 16 * EPS * (1.0 + abs(term))
    base, _, _ = _open(bond_pauli=True, bond_pauli_wall=True)
    both, r_w, module = _open(frame_pair=True, bond_pauli=True, bond_pauli_wall=True)
    pair_sum = float(fp.energy(Z6, r_w.sigma_free.detach(), POS6, None, None))
    reference = float(fp.site_reference(Z6, POS6, None, None, "almo"))
    assert abs(reference) > 0.05                                      # the Au at 2.25 A above the O: a real subtraction
    assert abs(both - base - (pair_sum - reference)) <= 1e3 * 16 * EPS * (1.0 + abs(base) + abs(pair_sum)), (both, base, pair_sum, reference)
    assert abs(module._frame_pair_last - (pair_sum - reference)) <= 16 * EPS * (1.0 + abs(pair_sum))


def test_without_the_flag_the_term_is_never_evaluated(monkeypatch):
    """The default is off, and off means the frame-pair code is not reached at all (so a model trained before the key
    existed is untouched): FramePair.energy is replaced by a function that raises, and the module still evaluates."""
    from trace_redox import frame_pair as module_file

    def forbidden(*args, **kwargs):
        raise AssertionError("FramePair.energy was called with frame_pair off")

    monkeypatch.setattr(module_file.FramePair, "energy", forbidden)
    monkeypatch.setattr(module_file.FramePair, "site_reference", forbidden)
    E, _, _ = _open(bond_pauli=True, bond_pauli_wall=True)
    assert math.isfinite(E)
    with pytest.raises(AssertionError, match="frame_pair off"):
        _open(frame_pair=True)


@pytest.mark.parametrize("mesh,wall", [((1, 1, 1), False), ((2, 1, 1), False), ((1, 1, 1), True)])
def test_the_module_adds_the_term_per_cell_under_periodic_boundaries_and_a_mesh(mesh, wall):
    """A periodic cell of six atoms (7 A), in the cluster space: the module with the term minus the module without it is
    FramePair.energy of the CELL, images included, at the unit mesh and at a Born-von Karman mesh of two cells (the
    supercell is evaluated whole and divided by its copies, and the pair sum of a supercell is its copies times the
    cell's); the diagnostic is per cell too.  With SPEC 33's wall on, the difference is the pair sum minus the wall's
    site reference over the same periodic images."""
    from trace_redox.frame_pair import frame_pair
    z = torch.tensor([79, 79, 12, 8, 12, 8])
    pbc = (True, True, True)
    cell = 7.0 * torch.eye(3, dtype=torch.float64)
    extra = dict(bond_pauli=True, bond_pauli_wall=True) if wall else {}
    out = {}
    for flag in (False, True):
        module, heads = _module(diabatic_space="clusters", frame_pair=flag, **extra)
        pos, h, _ = cluster(6, heads.irreps_in.dim, 5)
        module.redox = dataclasses.replace(module.redox, bvk_mesh=mesh)
        ei, es = lattice_graph(pos, cell, pbc, shells=1)
        vec, length = periodic_edges(pos, cell, ei, es)
        E, result = module(h, pos, ei, vec, length, pbc=pbc, cell=cell, edge_shift=es, data={"z": z, "total_charge": 0.0})[:2]
        out[flag] = (float(E), result, module, pos)
    (off, _, _, _), (on, r_on, module, pos) = out[False], out[True]
    fp = frame_pair()
    term = float(fp.energy(z, r_on.sigma_free.detach(), pos, cell, pbc))
    assert abs(term) > 0.1
    if wall:
        reference = float(fp.site_reference(z, pos, cell, pbc, "almo"))
        assert abs(reference) > 1e-3, reference                       # an Au within the wall's range of an O, images included
        term -= reference
    assert abs(on - off - term) <= 1e4 * 16 * EPS * (1.0 + abs(off) + abs(term)), (mesh, wall, on, off, term)
    assert abs(module._frame_pair_last - term) <= 1e3 * 16 * EPS * (1.0 + abs(term)), (mesh, module._frame_pair_last, term)


def test_the_configuration_refuses_what_would_count_twice():
    """frame_pair completes SPEC 23 and SPEC 25, so it needs both; it holds the Mg-Au pair of the older one-pair wall,
    so the two together are refused; and it is not yet defined for pairs inside a covalent unit.  A wall table without
    a record of the site it was measured on is refused too."""
    from trace_redox import RedoxConfig
    from trace_redox import frame_pair as module_file
    base = dict(fragment_ensemble="diabatic", anchor="formal", max_charge=3, exclusion=False)
    assert RedoxConfig(carrier_penetration=True, carrier_overlap="gordon_kim", frame_pair=True, **base).frame_pair is True
    with pytest.raises(ValueError, match="frame_pair"):
        RedoxConfig(frame_pair=True, **base)
    with pytest.raises(ValueError, match="frame_pair"):
        RedoxConfig(carrier_penetration=True, frame_pair=True, **base)
    with pytest.raises(ValueError, match="frame_wall"):
        RedoxConfig(carrier_penetration=True, carrier_overlap="gordon_kim", frame_pair=True, frame_wall=True, **base)
    with pytest.raises(ValueError, match="covalent_units"):         # not defined inside a covalent unit (SPEC 39.6)
        RedoxConfig(carrier_penetration=True, carrier_overlap="gordon_kim", frame_pair=True, carriers="covalent_units", **base)
    fp = module_file.frame_pair()
    z, pos = _otop_site(2.26)
    saved = dict(module_file.IN_SITU_SITES)
    module_file.IN_SITU_SITES.clear()
    try:
        with pytest.raises(ValueError, match="counted twice"):
            fp.site_reference(z, pos, None, None, "almo")
    finally:
        module_file.IN_SITU_SITES.update(saved)
