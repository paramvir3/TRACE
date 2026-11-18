"""Relaxed exclusion: each charge state's frontier density polarises away from the closed shells
(``RedoxConfig(exclusion_relaxation=True)``, trace_redox SPEC 18.4).

Rigid-shell model of state n's frontier density (N_u(n) electrons displaced by d; shell model, Dick & Overhauser,
Phys. Rev. 112, 90 (1958)), restoring energy 1/2 k |d|^2 with k = N_u(n)^2 K_E / alpha_u (alpha_u: the unit's
polarisability, the bound sector's own, A^3), second order in d with each pair's overlap Hessian projected onto its
positive (longitudinal) part:

    Phi_u(n) = sum_v c_v S_v - 1/2 g^T (k I + sum_v c_v max(S_v'', 0) Rhat_v Rhat_v^T)^-1 g,
    g = sum_v c_v S_v' Rhat_v,   c_v = K N_u(n) N_v s(R_uv),   Pi_u(n) = Phi_u(n) - Phi_u(f_u).

Exact properties tested: the frozen limit (alpha -> 0 gives SPEC 18), 0 <= Phi <= frozen, no relief in a
centrosymmetric cage, the single-neighbour closed form against an independent route, the shell moves away from the
closed shell (the dipole points at it), forces -grad E, smooth at the cutoff.
"""
import math

import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

HARTREE = 27.211386245988
BOHR = 0.529177210903
K_E = 14.399645


def _parts(z, pos, alpha, r_max=5.0):
    from trace_redox.constants import formal_frame
    from trace_redox.exclusion import Exclusion
    ex = Exclusion(r_cut=5.0, relax=True)
    z = torch.tensor(z)
    ei = neighbour_list(pos, r_max=r_max)
    vec, length = edges(pos, ei)
    states = torch.arange(-2, 4, dtype=torch.float64)
    a = torch.full((len(z),), float(alpha), dtype=torch.float64) if not torch.is_tensor(alpha) else alpha
    frozen, relief, shift = ex.relaxation(z, states, ei, length, vec, a, formal_frame(z))
    Pi = ex(z, states, ei, length, formal_frame(z), edge_vec=vec, alpha=a)
    return ex, states, frozen, relief, shift, Pi


def _au_on_o(distance=2.2):
    return [8, 79], torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, distance]], dtype=torch.float64)


def test_the_frozen_limit_is_spec_18():
    from trace_redox.constants import formal_frame
    from trace_redox.exclusion import Exclusion
    z = [8, 79, 79, 12, 13]
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0]],
                       dtype=torch.float64)
    _, states, _, relief, _, Pi = _parts(z, pos, 1e-9)
    ei = neighbour_list(pos, r_max=5.0)
    _, length = edges(pos, ei)
    frozen_pi = Exclusion(r_cut=5.0)(torch.tensor(z), states, ei, length, formal_frame(torch.tensor(z)))
    finite = torch.isfinite(frozen_pi)
    assert torch.equal(finite, torch.isfinite(Pi))
    assert float((Pi[finite] - frozen_pi[finite]).abs().max()) <= 1e-9 * float(frozen_pi[finite].abs().max())


def test_relief_is_bounded_by_the_frozen_exclusion():
    z = [8, 79, 79, 12, 13, 8]
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.2], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0],
                        [2.0, 2.0, 0.3]], dtype=torch.float64)
    for alpha in (0.5, 5.435, 50.0):
        _, _, frozen, relief, _, _ = _parts(z, pos, alpha)
        ok = torch.isfinite(frozen)
        assert bool((relief[ok] >= 0).all())
        assert bool((relief[ok] <= frozen[ok] + 1e-12).all())


def test_no_relief_in_a_centrosymmetric_cage():
    d = 2.10
    pos = torch.tensor([[0.0, 0.0, 0.0]] + [[s * d * (a == k) for a in range(3)] for k in range(3) for s in (1.0, -1.0)],
                       dtype=torch.float64)
    _, states, frozen, relief, shift, _ = _parts([13] + [8] * 6, pos, 5.0)
    j = int((states == 2).nonzero())
    assert float(frozen[0, j]) > 1.0                                   # Al2+ pays several eV
    assert float(relief[0].abs().max()) <= 1e-12 * float(frozen[0, j])
    assert float(shift[0].abs().max()) <= 1e-12


def test_single_neighbour_closed_form():
    """Au0 atop O2- at 2.2 A, default K = 10 Ha, I_eff = 9 eV, alpha = 5.435 A^3: relief = c^2 S'^2 / (2 (k + c S''))
    with S', S'' by autograd of the quadrature (an independent route), c = K * 1 * 8, k = K_E / alpha."""
    from trace_redox.exclusion import density_overlap
    z, pos = _au_on_o(2.2)
    _, states, frozen, relief, shift, _ = _parts(z, pos, 5.435)
    j = int((states == 0).nonzero())
    ku = math.sqrt(2 * 9.225554 / HARTREE)             # NIST IE1(Au), the package value
    R = torch.tensor(2.2, dtype=torch.float64, requires_grad=True)
    S = density_overlap(torch.tensor(ku), torch.tensor(1 / ku - 1), torch.tensor(math.sqrt(2 * 9.0 / HARTREE)),
                        torch.tensor(0.0), R / BOHR)
    (S1,) = torch.autograd.grad(S, R, create_graph=True)
    (S2,) = torch.autograd.grad(S1, R)
    c, k = 10 * HARTREE * 8, K_E / 5.435
    expected = c * c * float(S1) ** 2 / (2 * (k + c * float(S2)))
    assert abs(float(relief[1, j]) - expected) <= 1e-5 * expected
    assert abs(float(frozen[1, j]) - c * float(S)) <= 1e-7 * c * float(S)
    assert 0.45 < expected < 0.55                                      # 0.50 eV: about 37 % of the frozen 1.35 eV
    # the shell moves away from the O (the dipole -N e d points at it)
    assert float(shift[1, j, 2]) > 0.0 and float(shift[1, j, :2].abs().max()) <= 1e-12


def test_a_hole_relieves_less_once_the_neutral_has_relaxed():
    """Au+ atop O: no 6s electron, nothing to relax; Au0 relaxes, so Pi(Au+) = 0 - Phi(Au0) is smaller in magnitude
    than the frozen difference."""
    z, pos = _au_on_o(2.2)
    _, states, frozen, relief, _, Pi = _parts(z, pos, 5.435)
    j0, jp = int((states == 0).nonzero()), int((states == 1).nonzero())
    assert float(relief[1, jp]) == 0.0 and float(frozen[1, jp]) == 0.0
    assert float(Pi[1, j0]) == 0.0
    assert -float(frozen[1, j0]) < float(Pi[1, jp]) < 0.0
    assert abs(float(Pi[1, jp]) + float(frozen[1, j0]) - float(relief[1, j0])) <= 1e-12


def test_relief_grows_with_the_polarisability():
    z, pos = _au_on_o(2.3)
    reliefs = [float(_parts(z, pos, a)[3][1, 2]) for a in (1.0, 5.435, 20.0)]
    assert reliefs[0] < reliefs[1] < reliefs[2]


def test_relaxed_exclusion_forces_are_minus_the_gradient():
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), RedoxConfig(
        fragment_ensemble="diabatic", anchor="formal", max_charge=3, exclusion=True, exclusion_relaxation=True))
    assert module.exclusion.relax
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
    for i, axis in [(1, 2), (1, 0), (2, 0), (4, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-7 * (1.0 + abs(fd))


def test_relaxed_term_switches_off_smoothly_at_the_cutoff():
    vals = []
    for d in (5.0 - 1e-3, 5.0, 5.2):
        z, pos = _au_on_o(d)
        _, _, frozen, relief, _, Pi = _parts(z, pos, 5.435, r_max=6.0)
        finite = torch.isfinite(Pi[1])
        vals.append(float(Pi[1][finite].abs().max()))
    assert vals[1] == 0.0 and vals[2] == 0.0 and vals[0] <= 1e-6


def test_it_needs_exclusion():
    from trace_redox import RedoxConfig
    with pytest.raises(ValueError, match="exclusion_relaxation"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", exclusion_relaxation=True)
