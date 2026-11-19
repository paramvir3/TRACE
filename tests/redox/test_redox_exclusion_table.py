"""Tabulated density overlaps for the exclusion term (trace_redox SPEC 18.3).

S(R) of a carrier state against a closed-shell neighbour depends on two numbers only: the distance R and, for an
anion neighbour, its learned in-crystal removal energy I_eff (cation cores and carrier states are fixed by the
exact ladder).  log S is expanded once per pair type in Chebyshev polynomials of R (and of kappa_v = sqrt(2 I_eff))
from its values at Chebyshev-Gauss nodes: analytic, so every R-derivative is smooth (the relaxed exclusion, SPEC
18.4, uses S' and S'').  (A first version used bicubic Hermite interpolation: C^1 only, so S'' jumped at the nodes
and the relaxed energy's forces failed a finite-difference test; replaced 2026-10-15.)  Pre-registered accuracy against the quadrature (density_overlap): relative error <= 1e-7 in S and
<= 1e-5 in dS/dR, dS/dI over 0.8-5 A and the whole I_eff range; the Exclusion energies with the table agree with
the direct quadrature to <= 1e-7 relative.
Amendment (2026-10-15, measured): dS/dI changes sign near R = 1.81 A (d ln S / d ln I = -0.003 there, range -2.9 to
+0.56), so its relative error is unbounded at that zero whatever the grid; the I criterion is stated on the
dimensionless log-sensitivity instead: |error of d ln S / d ln I| <= 1e-6 (measured max 2.0e-7).
"""
import math

import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

HARTREE = 27.211386245988
BOHR = 0.529177210903

# carrier states (removal energy eV, charge n): Au- (EA), Au0, Al2+, Mg+, Al0
CARRIERS = [(2.30861, -1), (9.2255, 0), (28.447642, 2), (15.035271, 1), (5.985769, 0)]
CORES = [(80.1437, 2), (119.9924, 3)]          # Mg2+ and Al3+ cores: the next ionisation energy


def _kb(I, n):
    k = math.sqrt(2.0 * I / HARTREE)
    return k, (n + 1) / k - 1.0


def _random_points(n, lo, hi, seed):
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(n, generator=g, dtype=torch.float64)


@pytest.mark.parametrize("carrier", CARRIERS)
def test_anion_table_matches_the_quadrature(carrier):
    from trace_redox.exclusion import OverlapTable, density_overlap
    ku, bu = _kb(*carrier)
    table = OverlapTable(r_max=6.0)
    tid = table.anion_type(ku, bu)
    R = _random_points(400, 0.8, 5.0, 1).requires_grad_(True)
    I = _random_points(400, 4.0, 16.0, 2).requires_grad_(True)
    got = table.overlap(torch.full((400,), tid), R, I)
    kv = torch.sqrt(2.0 * I / HARTREE)
    ref = density_overlap(torch.tensor(ku), torch.tensor(bu), kv, torch.zeros_like(kv), R / BOHR)
    assert float(((got - ref) / ref).abs().max()) <= 1e-7
    g_R, g_I = torch.autograd.grad(got.sum(), (R, I))
    r_R, r_I = torch.autograd.grad(ref.sum(), (R, I))
    assert float(((g_R - r_R) / r_R).abs().max()) <= 1e-5
    assert float((I * (g_I - r_I) / ref).abs().max()) <= 1e-6          # error of d ln S / d ln I


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("core", CORES)
def test_core_table_matches_the_quadrature(carrier, core):
    from trace_redox.exclusion import OverlapTable, density_overlap
    ku, bu = _kb(*carrier)
    kv, bv = _kb(*core)
    table = OverlapTable(r_max=6.0)
    tid = table.core_type(ku, bu, kv, bv)
    R = _random_points(400, 0.8, 5.0, 3).requires_grad_(True)
    got = table.overlap(torch.full((400,), tid), R, torch.full((400,), 9.0))
    ref = density_overlap(torch.tensor(ku), torch.tensor(bu), torch.tensor(kv), torch.tensor(bv), R / BOHR)
    assert float(((got - ref) / ref).abs().max()) <= 1e-7
    (g,) = torch.autograd.grad(got.sum(), R)
    (r,) = torch.autograd.grad(ref.sum(), R)
    assert float(((g - r) / r).abs().max()) <= 1e-5


@pytest.mark.parametrize("carrier", CARRIERS)
def test_second_derivative_matches_the_quadrature(carrier):
    """S'' (used by the relaxed exclusion) against autograd of the quadrature, over 1.5-5 A (S'' > 0 there)."""
    from trace_redox.exclusion import OverlapTable, density_overlap
    ku, bu = _kb(*carrier)
    table = OverlapTable(r_max=6.0)
    tid = table.anion_type(ku, bu)
    R = _random_points(200, 1.5, 5.0, 4).requires_grad_(True)
    I = _random_points(200, 4.0, 16.0, 5)
    S, S1, S2 = table.overlap_derivatives(torch.full((200,), tid), R, I)
    kv = torch.sqrt(2.0 * I / HARTREE)
    ref = density_overlap(torch.tensor(ku), torch.tensor(bu), kv, torch.zeros_like(kv), R / BOHR)
    (r1,) = torch.autograd.grad(ref.sum(), R, create_graph=True)
    (r2,) = torch.autograd.grad(r1.sum(), R)
    assert float(((S1 - r1) / r1).abs().max()) <= 1e-6
    assert float(((S2 - r2) / r2).abs().max()) <= 1e-6
    (g2,) = torch.autograd.grad(S1.sum(), R)                       # the series is differentiable: d/dR S' = S''
    assert float(((g2 - S2) / S2).abs().max()) <= 1e-9


def test_exclusion_with_tables_equals_the_direct_quadrature():
    from trace_redox.constants import formal_frame
    from trace_redox.exclusion import Exclusion
    torch.manual_seed(0)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.2], [0.0, 0.1, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0],
                        [-2.0, 0.2, 0.0], [0.1, -2.1, 0.3]], dtype=torch.float64)
    z = torch.tensor([8, 79, 79, 12, 13, 8, 12])
    ei = neighbour_list(pos, r_max=5.0)
    _, length = edges(pos, ei)
    states = torch.arange(-2, 4, dtype=torch.float64)
    direct = Exclusion(r_cut=5.0, tabulate=False)
    fast = Exclusion(r_cut=5.0, tabulate=True)
    with torch.no_grad():
        direct.raw_i_eff[8] = 0.37                     # an I_eff away from its initial value
        fast.raw_i_eff[8] = 0.37
    a = direct(z, states, ei, length, formal_frame(z))
    b = fast(z, states, ei, length, formal_frame(z))
    finite = torch.isfinite(a)
    assert torch.equal(finite, torch.isfinite(b))
    assert float((a[finite] - b[finite]).abs().max()) <= 1e-7 * float(a[finite].abs().max())


def test_exclusion_tables_follow_the_learned_i_eff_and_k():
    """Gradients reach I_eff (through the table) and K, and agree with the direct quadrature's."""
    from trace_redox.constants import formal_frame
    from trace_redox.exclusion import Exclusion
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.25], [2.1, 0.0, 0.1]], dtype=torch.float64)
    z = torch.tensor([8, 79, 13])
    ei = neighbour_list(pos, r_max=5.0)
    _, length = edges(pos, ei)
    states = torch.arange(-1, 4, dtype=torch.float64)
    grads = []
    for tab in (False, True):
        m = Exclusion(r_cut=5.0, tabulate=tab)
        Pi = m(z, states, ei, length, formal_frame(z))
        E = torch.where(torch.isfinite(Pi), Pi, torch.zeros_like(Pi)).sum()
        g_i, g_k = torch.autograd.grad(E, (m.raw_i_eff, m.log_k))
        grads.append((float(g_i[8]), float(g_k)))
    (a_i, a_k), (b_i, b_k) = grads
    assert abs(a_i) > 0 and abs(a_i - b_i) <= 1e-5 * abs(a_i)
    assert abs(a_k - b_k) <= 1e-7 * abs(a_k)


def test_tabulated_exclusion_forces_are_minus_the_gradient():
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(),
                                 RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, exclusion=True))
    assert module.exclusion.tabulate
    g = torch.Generator().manual_seed(5)
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0]],
                        dtype=torch.float64)
    z = torch.tensor([8, 79, 79, 12, 13])
    h = torch.randn(5, heads.irreps_in.dim, generator=g, dtype=torch.float64)
    ei = neighbour_list(pos0)

    def energy(p):
        vec, length = edges(p, ei)
        return module(h, p, ei, vec, length, data={"z": z, "total_charge": 0.0})[0]

    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    for i, axis in [(1, 2), (2, 0), (4, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-7 * (1.0 + abs(fd))
