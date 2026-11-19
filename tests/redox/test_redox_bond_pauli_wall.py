"""The isotropic frame wall of an s-class atom against a closed-shell anion (trace_redox SPEC 33, the wall part;
2026-11-05): E_wall(a, j) = s_c(R) E1(R), E1 the lone Au0's frozen interaction (penetration electrostatics + Pauli) on
the O-top site of MgO(100), the quantity SPEC 33's bond-oriented law is measured relative to.  Why: the frozen step-20
model's Au-O wall below the data's 2.0 A is the backbone's extrapolation (a cliff: -300 eV at 1.6 A), and the law's
theta = 0 shielding (attractive, measured relative to the lone atom) tips it over; with the lone atom's wall explicit,
E1 + D is repulsive at every angle and rises below the data by physics.
"""
import math

import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

EPS = torch.finfo(torch.float64).eps


def lone_atom_over_oxygen(R):
    return torch.tensor([8, 79]), torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, R]], dtype=torch.float64)


def dimer_over_oxygen(theta_deg, R=2.26, bond=2.545):
    th = math.radians(theta_deg)
    contact = torch.tensor([0.0, 0.0, R])
    far = contact + bond * torch.tensor([0.0, math.sin(th), math.cos(th)])
    return torch.tensor([8, 79, 79]), torch.stack([torch.zeros(3), contact, far]).to(torch.float64)


def test_the_wall_is_the_tabulated_lone_atom_interaction_at_its_level_and_vanishes_at_the_cutoff():
    """At each level a lone Au over an O reproduces that level's table (frozen E1 / relaxed I_ALMO) at every measured R
    inside the switch (the interpolant passes through its knots exactly); beyond R_CUT it is zero; a Mg gives zero; an
    untabulated pair and an unknown level are refused."""
    from trace_redox import bond_pauli as B
    for level in ("frz", "almo"):
        wall = B.bond_pauli_wall(level)
        R_tab, E_tab = B.WALL_TABLES[level][(79, 8)]
        for R, E in zip(R_tab, E_tab):
            if E is not None and R <= B.R_ON:
                z, pos = lone_atom_over_oxygen(R)
                got = float(wall.energy(z, pos, None, None))
                assert abs(got - E) <= 64 * EPS * (1.0 + abs(E)), (level, R, got, E)
        z, pos = lone_atom_over_oxygen(B.R_CUT + 0.1)
        assert float(wall.energy(z, pos, None, None)) == 0.0
    z, pos = lone_atom_over_oxygen(2.3)
    z_mg = z.clone(); z_mg[0] = 12
    assert float(B.bond_pauli_wall().energy(z_mg, pos, None, None)) == 0.0
    z_ag = z.clone(); z_ag[1] = 47
    with pytest.raises(ValueError):
        B.bond_pauli_wall().energy(z_ag, pos, None, None)
    with pytest.raises(ValueError):
        B.bond_pauli_wall("hartree-fock")


def test_the_wall_acts_on_bonded_and_lone_atoms_alike():
    """The wall is isotropic and needs no partner: the dimer's wall equals the sum of its two atoms' lone walls."""
    from trace_redox import bond_pauli as B
    wall = B.bond_pauli_wall("frz")
    z, pos = dimer_over_oxygen(50.0)
    both = float(wall.energy(z, pos, None, None))
    one = float(wall.energy(z[:2], pos[:2], None, None)) + float(wall.energy(z[[0, 2]], pos[[0, 2]], None, None))
    assert abs(both - one) <= 64 * EPS * (1.0 + abs(both)), (both, one)


# the measured frozen interaction of the upright dimer (theta = 0) with the cluster, eV (frz_angular.py: E1 dimer)
MEASURED_E1_DIMER_UPRIGHT = {2.26: 0.6533, 2.4: 0.2655, 2.6: 0.0064, 2.9: -0.0977}
# the measured relaxed (ALMO, transfer-free) interaction of the upright dimer, eV (almo_angular.py: dimer I_ALMO)
MEASURED_IALMO_DIMER_UPRIGHT = {2.26: -0.0295, 2.6: -0.2548, 2.9: -0.2067}


def test_wall_plus_law_is_the_dimer_interaction_at_each_level():
    """Wall + law at theta = 0 is the dimer's interaction at the law's level up to the omitted isotropic D0 and the fit's
    residual: frozen, the measured E1 of the upright dimer within 0.06 eV at the four R (D0 <= 0.04); relaxed, the
    measured I_ALMO within 0.10 eV at the three R (D0 <= 0.07).  The frozen sum decreases monotonically from 1.8 to 2.9 A
    at theta = 0, 40, 70 deg (no well, no cliff; the frozen table reaches 1.8 A); the law alone at theta = 0 is attractive
    (the measured shielding) at both levels."""
    from trace_redox import bond_pauli as B
    Rs = [1.8, 2.0, 2.1, 2.26, 2.4, 2.6, 2.9]
    wall, law = B.bond_pauli_wall("frz"), B.bond_pauli("frz")
    for th in (0.0, 40.0, 70.0):
        tot = []
        for R in Rs:
            z, pos = dimer_over_oxygen(th, R=R)
            tot.append(float(wall.energy(z, pos, None, None) + law.energy(z, pos, None, None)))
        assert all(a > b for a, b in zip(tot, tot[1:])), (th, tot)
    for R, e1 in MEASURED_E1_DIMER_UPRIGHT.items():
        z, pos = dimer_over_oxygen(0.0, R=R)
        got = float(wall.energy(z, pos, None, None) + law.energy(z, pos, None, None))
        assert abs(got - e1) <= 0.06, ("frz", R, got, e1)
    wall, law = B.bond_pauli_wall("almo"), B.bond_pauli("almo")
    for R, i_almo in MEASURED_IALMO_DIMER_UPRIGHT.items():
        z, pos = dimer_over_oxygen(0.0, R=R)
        got = float(wall.energy(z, pos, None, None) + law.energy(z, pos, None, None))
        assert abs(got - i_almo) <= 0.10, ("almo", R, got, i_almo)
    for level in ("frz", "almo"):
        z, pos = dimer_over_oxygen(0.0)
        assert float(B.bond_pauli(level).energy(z, pos, None, None)) < 0.0


def test_the_wall_forces_are_its_gradient_with_periodic_images():
    """Autograd forces against central differences (1e-5 A) for a dimer over an O in a periodic surface cell whose
    image O's are inside the switch; a finite double backward under anomaly detection."""
    from trace_redox import bond_pauli as B
    wall = B.bond_pauli_wall("frz")                              # the complete table (the relaxed one is being measured)
    z, pos0 = dimer_over_oxygen(60.0, R=2.3)                     # off the table's knots (the interpolant is C^1)
    cell = torch.diag(torch.tensor([4.265, 4.265, 20.0], dtype=torch.float64))
    pbc = (True, True, False)
    pos = pos0.clone().requires_grad_(True)
    E = wall.energy(z, pos, cell, pbc)
    g, = torch.autograd.grad(E, pos)
    for i in (1, 2):
        for ax in range(3):
            plus, minus = pos0.clone(), pos0.clone()
            plus[i, ax] += 1e-5
            minus[i, ax] -= 1e-5
            fd = (float(wall.energy(z, plus, cell, pbc)) - float(wall.energy(z, minus, cell, pbc))) / 2e-5
            assert abs(fd - float(g[i, ax])) <= 1e-7 * (1.0 + abs(fd)), (i, ax, float(g[i, ax]), fd)
    pos = pos0.clone().requires_grad_(True)
    with torch.autograd.set_detect_anomaly(True):
        E = wall.energy(z, pos, cell, pbc)
        g, = torch.autograd.grad(E, pos, create_graph=True)
        (g ** 2).sum().backward()
    assert bool(torch.isfinite(pos.grad).all())
    # force continuity across the knot at 2.26 A: over the 2e-6 A gap a C^1 force changes by at most |E''| 2e-6 with
    # E'' ~ E1 s^2 ~ 1.43 eV x (3.7 /A)^2 = 20 eV/A^2, i.e. 4e-5 eV/A; a slope jump of the earlier piecewise form was
    # 4e-2 eV/A, a thousand times more
    forces = []
    for R in (2.26 - 1e-6, 2.26 + 1e-6):
        z1, p1 = lone_atom_over_oxygen(R)
        p1 = p1.requires_grad_(True)
        forces.append(torch.autograd.grad(wall.energy(z1, p1, None, None), p1)[0])
    assert float((forces[0] - forces[1]).abs().max()) <= 1e-4, float((forces[0] - forces[1]).abs().max())


def test_the_module_adds_the_wall_when_asked():
    """RedoxConfig(bond_pauli_wall=True) adds exactly the wall's energy; it needs bond_pauli=True (the wall is the law's
    reference); the default is off."""
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    from trace_redox.bond_pauli import bond_pauli_wall
    assert RedoxConfig().bond_pauli_wall is False
    with pytest.raises(ValueError):
        RedoxConfig(bond_pauli_wall=True)
    z = torch.tensor([8, 79, 79, 12, 8])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.3, 1.4, 4.3], [2.1325, 0.0, 0.0], [2.1325, 2.1325, 0.0]],
                       dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    out = {}
    for flag in (False, True):
        heads = make_heads()
        torch.manual_seed(1)
        cfg = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True, bond_pauli=True,
                          bond_pauli_wall=flag)
        module = RedoxElectrostatics(heads, ElectrostaticConfig(), cfg)
        h = torch.randn(5, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        out[flag] = float(module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})[0])
    wall = float(bond_pauli_wall("almo").energy(z, pos, None, None))
    assert wall != 0.0
    assert abs(out[True] - out[False] - wall) <= 1e-10 * (1 + abs(out[False])), (out, wall)
