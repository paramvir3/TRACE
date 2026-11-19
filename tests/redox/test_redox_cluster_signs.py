"""The signs of the configuration couplings against exact s-orbital Hubbard clusters (2026-11-13; README, "Measured:
the sign structure of the configuration couplings").

Exact reference: H = -t sum_<ij>,sigma (c+_i,sigma c_j,sigma + h.c.) + U sum_i n_i,up n_i,down, diagonalised in Fock
space; the level compared is the lowest one of the lowest total spin (the configuration model holds one state per
occupation pattern, the pattern with every moving pair in its singlet, SPEC 29).

Model: trace_redox.diabatic.diabatic_hamiltonian on units with charges +1, 0, -1 (0, 1, 2 s electrons), E = U per
doubly occupied site, T = -t on the bonds, SPEC 29's factors.  Its signs are the Jordan-Wigner ones of spin-less
local orbitals (SPEC 17.2).

What is established here: the model is exact for ONE carrier of either kind on every cluster (the particle-hole sign
of a hole on an odd ring comes out of the Jordan-Wigner string), exact on two sites at every filling, and independent
of the order of the units.  What is recorded as not holding: two or more carriers on three or more sites.
- Two electrons (strict xfail, and the measured error pinned): two spin-less fermions exchange with a minus sign, the
  two electrons of a singlet with a plus, so the model's level lies ABOVE the exact singlet by 0.4-0.5 t on the open
  chain of three, 1.1 t on the triangle and the square, 2.1 t on the tetrahedron.
- Two holes: the same error on the bipartite clusters (chain +0.51 t, square +1.16 t); on the triangle the model
  lies 0.27-0.63 t BELOW the exact singlet (it sits on the triplet, the true ground state there); exact on the
  tetrahedron.
- Half filling (the neutral sector of three or more s1 atoms, on which SPEC 40's hop acts): the model lies BELOW the
  exact ground state of any spin, by 0.11-0.23 t (chain of three), 0.61-0.73 t (triangle), 0.48-0.50 t (square) and
  1.08-1.19 t (tetrahedron): it over-binds, and no test of an upper bound can be made of it.
The pinned numbers were added after the review of 2026-11-14 so that a change of any of these errors is seen.
"""
import itertools

import numpy as np
import pytest
import torch
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

EPS = float(np.finfo(np.float64).eps)
T_HOP = 1.0
SHAPES = {"dimer": (2, [(0, 1)]),
          "chain3": (3, [(0, 1), (1, 2)]),
          "triangle": (3, [(0, 1), (1, 2), (0, 2)]),
          "square": (4, [(0, 1), (1, 2), (2, 3), (0, 3)]),
          "tetrahedron": (4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])}


def _spectrum(bonds, n, N, U, twice_sz):
    orb = 2 * n                                                     # spin-orbital 2 i + spin
    states = [s for s in range(1 << orb) if bin(s).count("1") == N
              and sum(((s >> (2 * i)) & 1) - ((s >> (2 * i + 1)) & 1) for i in range(n)) == twice_sz]
    if not states:
        return np.zeros(0)
    index = {s: k for k, s in enumerate(states)}
    H = np.zeros((len(states), len(states)))

    def hop(s, a, b):                                               # c+_a c_b |s> with its fermion sign
        if not (s >> b) & 1 or (s >> a) & 1:
            return None, 0
        sign = (-1) ** bin(s & ((1 << b) - 1)).count("1")
        s2 = s ^ (1 << b)
        sign *= (-1) ** bin(s2 & ((1 << a) - 1)).count("1")
        return s2 | (1 << a), sign

    for s in states:
        k = index[s]
        H[k, k] = U * sum(((s >> (2 * i)) & 1) & ((s >> (2 * i + 1)) & 1) for i in range(n))
        for (i, j) in bonds:
            for sp in (0, 1):
                for a, b in ((2 * i + sp, 2 * j + sp), (2 * j + sp, 2 * i + sp)):
                    s2, sign = hop(s, a, b)
                    if s2 is not None:
                        H[index[s2], k] += -T_HOP * sign
    return np.linalg.eigvalsh(H)


def exact_lowest_spin(bonds, n, N, U):
    """The lowest level of the lowest total spin: the S_z = S_min levels that are not in the S_z = S_min + 1 sector."""
    low = _spectrum(bonds, n, N, U, N % 2)
    high = list(_spectrum(bonds, n, N, U, N % 2 + 2))
    for e in low:
        hit = next((k for k, x in enumerate(high) if abs(x - e) < 1e-9), None)
        if hit is None:
            return float(e)
        high.pop(hit)
    raise AssertionError("no level of the lowest spin")


def model_level(bonds, n, N, U, order=None):
    from trace_redox.diabatic import diabatic_hamiltonian
    order = list(range(n)) if order is None else list(order)
    place = {site: k for k, site in enumerate(order)}
    cfgs = [c for c in itertools.product((1, 0, -1), repeat=n) if sum(1 - q for q in c) == N]
    E = torch.tensor([U * sum(1 for q in c if q == -1) for c in cfgs], dtype=torch.float64)
    T = torch.zeros(n, n, dtype=torch.float64)
    for (i, j) in bonds:
        T[place[i], place[j]] = T[place[j], place[i]] = -T_HOP
    H = diabatic_hamiltonian(E, torch.tensor(cfgs, dtype=torch.long), torch.ones(n, dtype=torch.long), T,
                             electrons=torch.full((n,), 79, dtype=torch.long))
    assert torch.equal(H, H.T)
    return float(torch.linalg.eigvalsh(H)[0])


def _tol(n, U):
    return 256 * EPS * (2 * n * T_HOP + 2 * n * U)                  # eigenvalues of matrices of norm <= 2 n (t + U)


def test_the_exact_solver_reproduces_the_two_site_closed_form():
    """Guard of the reference itself: two sites, two electrons, E = (U - sqrt(U^2 + 16 t^2)) / 2; one electron, -t."""
    for U in (0.0, 1.0, 4.0):
        assert abs(exact_lowest_spin([(0, 1)], 2, 2, U) - 0.5 * (U - np.sqrt(U * U + 16.0))) <= _tol(2, U)
        assert abs(exact_lowest_spin([(0, 1)], 2, 1, U) + 1.0) <= _tol(2, U)


@pytest.mark.parametrize("shape", ["chain3", "triangle", "square", "tetrahedron"])
@pytest.mark.parametrize("U", [1.0, 4.0])
def test_one_carrier_of_either_kind_is_exact(shape, U):
    """One electron, and one hole (2 n - 1 electrons): the model's lowest level is the exact one.  On the triangle the
    electron gains 2 t and the hole only t (levels -2 t and -t + 2 U): the particle-hole sign of a hop on an odd ring,
    which the Jordan-Wigner string of the filled local orbitals supplies."""
    n, bonds = SHAPES[shape]
    for N in (1, 2 * n - 1):
        assert abs(model_level(bonds, n, N, U) - exact_lowest_spin(bonds, n, N, U)) <= _tol(n, U), (shape, N)
    if shape == "triangle":
        assert abs(model_level(bonds, n, 1, U) + 2.0) <= _tol(n, U) and abs(model_level(bonds, n, 5, U) - (-1.0 + 2.0 * U)) <= _tol(n, U)


@pytest.mark.parametrize("U", [1.0, 4.0])
def test_two_sites_are_exact_at_every_filling(U):
    for N in (1, 2, 3):
        assert abs(model_level([(0, 1)], 2, N, U) - exact_lowest_spin([(0, 1)], 2, N, U)) <= _tol(2, U), N


@pytest.mark.parametrize("shape", ["chain3", "triangle", "tetrahedron"])
def test_the_level_does_not_depend_on_the_order_of_the_units(shape):
    """The Jordan-Wigner string depends on the order of the units; the spectrum must not."""
    n, bonds = SHAPES[shape]
    for N in range(1, 2 * n):
        ref = model_level(bonds, n, N, 1.0)
        for order in itertools.permutations(range(n)):
            assert abs(model_level(bonds, n, N, 1.0, order=order) - ref) <= _tol(n, 1.0), (shape, N, order)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="the couplings carry the signs of spin-less fermions: two carriers on three or more "
                                       "sites exchange with a minus sign, the two electrons of a singlet with a plus "
                                       "(measured 2026-11-13: +0.51 t on the chain of three, +1.06 t on the triangle at U = t)")
@pytest.mark.parametrize("shape", ["chain3", "triangle"])
def test_two_electrons_on_three_sites_reach_the_singlet(shape):
    """The ground state of two electrons is a node-less singlet on every cluster; the model should reach its level."""
    n, bonds = SHAPES[shape]
    assert abs(model_level(bonds, n, 2, 1.0) - exact_lowest_spin(bonds, n, 2, 1.0)) <= _tol(n, 1.0)


#: model minus the exact lowest level of the lowest spin, in t, at U = t and U = 4 t (measured 2026-11-14)
MEASURED_ERRORS = {
    ("chain3", 2): (0.507903, 0.396125), ("chain3", 3): (-0.108738, -0.228034), ("chain3", 4): (0.507903, 0.396125),
    ("triangle", 2): (1.055811, 1.123106), ("triangle", 3): (-0.607656, -0.725083), ("triangle", 4): (-0.267949, -0.627719),
    ("square", 2): (1.159063, 1.078378), ("square", 4): (-0.504422, -0.477142), ("square", 6): (1.159063, 1.078378),
    ("tetrahedron", 2): (2.070440, 2.168397), ("tetrahedron", 4): (-1.083581, -1.188754), ("tetrahedron", 6): (0.0, 0.0),
}


@pytest.mark.parametrize("case", sorted(MEASURED_ERRORS))
def test_the_measured_errors_beyond_one_carrier(case):
    """Two electrons, half filling and two holes on each cluster: the model's error against the exact lowest-spin
    level is the recorded one to 1e-6 t (a record of the present sign structure, not a statement that it is right)."""
    shape, N = case
    n, bonds = SHAPES[shape]
    for U, want in zip((1.0, 4.0), MEASURED_ERRORS[case]):
        got = model_level(bonds, n, N, U) - exact_lowest_spin(bonds, n, N, U)
        assert abs(got - want) <= 1e-6, (shape, N, U, got, want)


@pytest.mark.parametrize("shape", ["chain3", "triangle", "square", "tetrahedron"])
def test_at_half_filling_the_model_lies_below_every_exact_level(shape):
    """The neutral sector of n s1 atoms: the model's lowest level is below the exact ground state of ANY spin (the
    lowest eigenvalue of the full Fock-space sector), so the configuration model is not variational there."""
    n, bonds = SHAPES[shape]
    for U in (1.0, 4.0):
        ground = min(float(_spectrum(bonds, n, n, U, sz)[0]) for sz in range(n % 2, n + 1, 2))
        assert model_level(bonds, n, n, U) < ground - 0.1, (shape, U, model_level(bonds, n, n, U), ground)
