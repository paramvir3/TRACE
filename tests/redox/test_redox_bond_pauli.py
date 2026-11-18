"""The bond-oriented Pauli repulsion of a bonded s-class atom against a closed-shell anion (trace_redox SPEC 33,
2026-11-05).  Measured on Au2 over the O-top O of MgO(100) by ALMO energy decomposition (tools/electrostatics/
frz_angular.py, almo_angular.py; theory note 2026-10-18 section 4c, 2026-11-04):

    E_bp(a, j) = s_b(R_ab) s_c(R_aj) [ D1(R_aj) cos t + D2(R_aj) P2(cos t) ],   cos t = r_aj . (-e_ab),
    D1(R) = C1 f(R)^0.888,  D2(R) = C2 sqrt(f(R)),  f(R) = E1(R) / E1(2.26 A),

E1 the lone Au0's frozen repulsion on the O-top site (1.4254 / 0.8454 / 0.3891 / 0.1045 eV at 2.26 / 2.4 / 2.6 / 2.9 A),
the constants fitted jointly over R and theta = 0-70 deg with the module's own two-atom form (SPEC 33).  The tests
assert the measured law, not a machine's rounding: tolerances are twice the fits' largest residuals (frozen 3.7 meV over
four R; relaxed 6.2 meV over three R).  90 deg is not tested: the far Au then sits over a Mg, whose
non-additivity the law has no term for, and the measured D(90) holds it.
"""
import math

import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

EPS = torch.finfo(torch.float64).eps
# the measured D(theta) (eV) at the O-top site, R = 2.26 A: frozen (frz_angular.py) and relaxed (almo_angular.py)
MEASURED_FRZ_2p26 = {0: -0.7639, 20: -0.6390, 40: -0.3287, 60: 0.0197, 70: 0.1647, 90: 0.3434}
MEASURED_ALMO_2p26 = {0: -0.4944, 20: -0.4061, 40: -0.1799, 60: 0.0887, 70: 0.2061, 90: 0.3738}
# the frozen law at two R outside the fit (the pre-registered radial test, 2026-11-04 21:19): D(0), D(70)
MEASURED_FRZ_RADIAL = {2.4: (-0.5729, 0.1236), 2.9: (-0.1984, 0.0449)}
# the relaxed law at 2.6 and 2.9 A (almo_angular.py, 2026-11-05): D_ALMO(0), D_ALMO(70); the relaxed constants are fitted to them
MEASURED_ALMO_RADIAL = {2.6: (-0.2966, 0.1156), 2.9: (-0.1756, 0.0643)}


def dimer_over_oxygen(theta_deg, R=2.26, bond=2.545, extra_ions=()):
    """Au2 with its contact atom R above an O at the origin, the bond rotated by theta from the normal (toward +y)."""
    th = math.radians(theta_deg)
    contact = torch.tensor([0.0, 0.0, R])
    far = contact + bond * torch.tensor([0.0, math.sin(th), math.cos(th)])
    pos = torch.stack([torch.zeros(3), contact, far] + [torch.tensor(p, dtype=torch.float64) for _, p in extra_ions])
    z = torch.tensor([8, 79, 79] + [int(s) for s, _ in extra_ions])
    return z, pos.to(torch.float64)


def rotation_difference(law, theta_deg, R=2.26):
    """E_bp(theta) - E_bp(0) of the dimer over one O (the isotropic part cancels)."""
    z0, p0 = dimer_over_oxygen(0.0, R)
    z1, p1 = dimer_over_oxygen(theta_deg, R)
    return float(law.energy(z1, p1, None, None) - law.energy(z0, p0, None, None))


def test_the_relaxed_law_reproduces_the_measured_rotation_at_the_fit_residual():
    """Over one O at 2.26 A, E(theta) - E(0) equals the measured ALMO D(theta) - D(0) within twice the joint fit's
    largest residual (9.5 meV) at 0-70 deg: the law's anisotropic part is the measured one."""
    from trace_redox.bond_pauli import bond_pauli
    law = bond_pauli(level="almo")
    for th, d in MEASURED_ALMO_2p26.items():
        if th == 90:
            continue
        got = rotation_difference(law, th)
        want = d - MEASURED_ALMO_2p26[0]
        assert abs(got - want) <= 0.02, (th, got, want)


def test_the_relaxed_law_reproduces_its_other_distances():
    """level='almo' at 2.6 and 2.9 A: D(70) - D(0) within twice the joint fit's largest residual (9.5 meV over three R
    and five angles) of the measured +0.4122 and +0.2399 eV (SPEC 33)."""
    from trace_redox.bond_pauli import bond_pauli
    law = bond_pauli(level="almo")
    for R, (d0, d70) in MEASURED_ALMO_RADIAL.items():
        got = rotation_difference(law, 70, R=R)
        assert abs(got - (d70 - d0)) <= 0.02, (R, got, d70 - d0)


def test_the_frozen_law_reproduces_the_frozen_rotation_and_its_radial_test():
    """level='frz': the frozen rotation at 2.26 A and D(70) - D(0) at 2.4 and 2.9 A, each within twice the joint fit's
    largest residual (4.7 meV over four R and five angles)."""
    from trace_redox.bond_pauli import bond_pauli
    law = bond_pauli(level="frz")
    for th, d in MEASURED_FRZ_2p26.items():
        if th < 90:
            assert abs(rotation_difference(law, th) - (d - MEASURED_FRZ_2p26[0])) <= 0.01, th
    for R, (d0, d70) in MEASURED_FRZ_RADIAL.items():
        got = rotation_difference(law, 70, R=R)
        assert abs(got - (d70 - d0)) <= 0.01, (R, got, d70 - d0)           # the joint fit's residual, twice


def test_the_law_is_zero_without_a_bond_an_anion_or_within_reach():
    """A lone Au over an O (no partner within the bond switch), a bonded dimer with no anion within the cutoff, and a
    dimer over a Mg only: zero exactly.  A partner beyond 3.4 A does not count as a bond."""
    from trace_redox.bond_pauli import bond_pauli
    law = bond_pauli()
    z, pos = dimer_over_oxygen(40.0)
    assert float(law.energy(z[:2], pos[:2], None, None)) == 0.0
    far_partner = pos.clone()
    far_partner[2] = pos[1] + torch.tensor([0.0, 0.0, 3.6])
    assert float(law.energy(z, far_partner, None, None)) == 0.0
    high = pos.clone()
    high[1:, 2] += 3.0                                          # the contact Au at 5.26 A from the O: beyond the cutoff
    assert float(law.energy(z, high, None, None)) == 0.0
    z_mg = z.clone()
    z_mg[0] = 12
    assert float(law.energy(z_mg, pos, None, None)) == 0.0
    assert float(law.energy(z, pos, None, None)) != 0.0


def test_the_forces_are_the_gradient_and_the_term_is_double_backward_safe():
    """Autograd forces against central differences (1e-5 A; the error of the difference is ~h^2 E''' + eps E / h, below
    1e-7 eV/A here) on a tilted dimer over an O and a Mg, in a periodic cell whose images enter the sum."""
    from trace_redox.bond_pauli import bond_pauli
    law = bond_pauli()
    z, pos0 = dimer_over_oxygen(55.0, R=2.3, extra_ions=((12, (2.1325, 0.0, 0.0)),))       # off the table's knots
    cell = torch.eye(3, dtype=torch.float64) * 6.0
    pbc = (True, True, False)
    pos = pos0.clone().requires_grad_(True)
    E = law.energy(z, pos, cell, pbc)
    g, = torch.autograd.grad(E, pos)
    for i in range(3):
        for ax in range(3):
            plus, minus = pos0.clone(), pos0.clone()
            plus[i, ax] += 1e-5
            minus[i, ax] -= 1e-5
            fd = (float(law.energy(z, plus, cell, pbc)) - float(law.energy(z, minus, cell, pbc))) / 2e-5
            assert abs(fd - float(g[i, ax])) <= 1e-7 * (1.0 + abs(fd)), (i, ax, float(g[i, ax]), fd)
    pos = pos0.clone().requires_grad_(True)
    with torch.autograd.set_detect_anomaly(True):
        E = law.energy(z, pos, cell, pbc)
        g, = torch.autograd.grad(E, pos, create_graph=True)
        (g ** 2).sum().backward()
    assert bool(torch.isfinite(pos.grad).all())
    # the table's interpolant is C^1 (monotone cubic Hermite): the force is continuous across a knot (2.26 A), while
    # the second derivative may jump there (a central difference straddling the knot sees that jump, so the finite
    # differences above are taken off the knots)
    forces = []
    for R in (2.26 - 1e-6, 2.26 + 1e-6):
        z1, p1 = dimer_over_oxygen(55.0, R=R)
        p1 = p1.requires_grad_(True)
        forces.append(torch.autograd.grad(law.energy(z1, p1, None, None), p1)[0])
    assert float((forces[0] - forces[1]).abs().max()) <= 1e-5, float((forces[0] - forces[1]).abs().max())


def test_the_periodic_sum_counts_the_images_once_and_is_translation_invariant():
    """A dimer over the O of a 2 x 2 surface cell: translating every atom by a lattice vector (and across the cell
    boundary) leaves the energy unchanged to round-off, and the energy with images exceeds the open-boundary one
    (the image O's are within the cutoff)."""
    from trace_redox.bond_pauli import bond_pauli
    law = bond_pauli()
    z, pos = dimer_over_oxygen(60.0)                 # the far Au is 4.09 A from an image O: inside the switch
    cell = torch.diag(torch.tensor([4.265, 4.265, 20.0], dtype=torch.float64))
    pbc = (True, True, False)
    E = float(law.energy(z, pos, cell, pbc))
    E_open = float(law.energy(z, pos, None, None))
    assert E != E_open
    shifted = pos + torch.tensor([4.265 * 0.7, -4.265 * 1.3, 0.0])
    E2 = float(law.energy(z, shifted, cell, pbc))
    assert abs(E - E2) <= 64 * EPS * (1.0 + abs(E)), (E, E2)


def test_an_untabulated_pair_is_refused_and_the_config_validates():
    """Ag2 over an O: no table for (47, 8), so bond_pauli refuses rather than return an incomplete energy; an unknown
    level is refused; the default configuration is off."""
    from trace_redox import RedoxConfig
    from trace_redox.bond_pauli import bond_pauli
    assert RedoxConfig().bond_pauli is False
    with pytest.raises(ValueError):
        bond_pauli(level="hartree-fock")
    with pytest.raises(ValueError):
        RedoxConfig(bond_pauli=True, bond_pauli_level="other")
    z, pos = dimer_over_oxygen(40.0)
    z_ag = z.clone()
    z_ag[1:] = 47
    with pytest.raises(ValueError):
        bond_pauli().energy(z_ag, pos, None, None)


def test_the_module_adds_the_term_when_asked():
    """RedoxConfig(bond_pauli=True) adds exactly the term's energy to the module's (a frame term, configuration
    independent); the default leaves the energy bit for bit; forces through the module equal the finite difference."""
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    from trace_redox.bond_pauli import bond_pauli
    z = torch.tensor([8, 79, 79, 12, 8])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.3, 1.4, 4.3], [2.1325, 0.0, 0.0], [2.1325, 2.1325, 0.0]],
                       dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    out = {}
    for flag in (False, True):
        heads = make_heads()
        torch.manual_seed(1)
        cfg = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True, bond_pauli=flag)
        module = RedoxElectrostatics(heads, ElectrostaticConfig(), cfg)
        h = torch.randn(5, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        out[flag] = float(module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})[0])
    term = float(bond_pauli().energy(z, pos, None, None))
    assert term != 0.0
    assert abs(out[True] - out[False] - term) <= 1e-10 * (1 + abs(out[False])), (out, term)
