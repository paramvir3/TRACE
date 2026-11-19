"""SPEC 33 per charge state (2026-11-06; written red before the implementation, which follows the scoring of steps 25
and 26): the bond-oriented law's coefficients depend on the s atom's charge state.  Measured (theory note section 4c,
2026-11-06): Au2- over the O-top O at 2.26 A has an orientation dependence of +0.200 eV (relaxed) against the neutral
dimer's +0.700 -- the sigma* electron restores the axial density the sigma bond removed -- with five-angle coefficients
C1-, C2- = -0.1181, -0.0911 eV (relaxed; frozen +0.1011, -0.2905) against the neutral's -0.0694, -0.4973 (frozen -0.0720,
-0.6649).  Entered as a one-body term of the configuration energies (an [n, S] table added to the ladder, as the
penetration self-terms are), so each configuration carries its own law; the wall stays a frame term.

Amended 2026-11-08, before the implementation (README step 27 amendment): (i) the 2.6 A anion scan was read and the
relaxed level's anion law takes its own fitted p2 (C1-, C2-, p2- = -0.1094, -0.0957, 0.4561; p1 the neutral's), so the
state -1 column is C1- A1(p1-) + C2- A2(p2-) with the geometric sums at that state's exponents (test 1); (ii) the ladder
carries T(n) - T(0), the frame path keeps the neutral law (equal up to a configuration-independent constant); test 4's
first form asserted a frame configuration (0, ..., 0) that this geometry cannot have (total charge 0 against a formal
frame summing to -2; probe 2026-11-08: 19 configurations, none of them the frame), so its invariant is restated on the
configurations it does have.
"""
import math

import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

EPS = torch.finfo(torch.float64).eps
# the red stage of step 27 (pre-registered 2026-11-06): the API is not built until steps 25 / 26 are scored.  strict
# xfail keeps the suite green now and fails loudly once the implementation lands, so the marker is then removed.
# step 27 built 2026-11-08: the strict xfail marker of the red stage is removed (it would now fail loudly)
# the measured anion rotation at 2.26 A (almo_angular_anion.py): D-(theta), relaxed and frozen
MEASURED_ANION_ALMO = {0: -0.8624, 20: -0.8378, 40: -0.7756, 60: -0.7023, 70: -0.6619}
MEASURED_ANION_FRZ = {0: -0.9927, 20: -0.9428, 40: -0.8288, 60: -0.7198, 70: -0.6682}


def dimer_over_oxygen(theta_deg, R=2.26, bond=2.545, extra_ions=()):
    th = math.radians(theta_deg)
    contact = torch.tensor([0.0, 0.0, R])
    far = contact + bond * torch.tensor([0.0, math.sin(th), math.cos(th)])
    pos = torch.stack([torch.zeros(3), contact, far] + [torch.tensor(p, dtype=torch.float64) for _, p in extra_ions])
    z = torch.tensor([8, 79, 79] + [int(s) for s, _ in extra_ions])
    return z, pos.to(torch.float64)


STATES = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0])


def test_the_neutral_column_is_the_frame_law_and_the_anion_column_is_its_own():
    """state_terms [n, S]: the state-0 column summed over atoms equals the frame law's energy to round-off; a non-s atom's
    row is zero; the state -1 column of a bonded Au differs from its state-0 entry by (C1- - C1) A1 + (C2- - C2) A2 with
    the module's own geometric sums; the cation column (+1, unmeasured) carries the neutral law, as stated."""
    from trace_redox import bond_pauli as B
    for level in ("almo", "frz"):
        law = B.bond_pauli(level)
        z, pos = dimer_over_oxygen(55.0, extra_ions=((12, (2.1325, 0.0, 0.0)), (8, (2.1325, 2.1325, 0.0))))
        T = law.state_terms(z, STATES, pos, None, None)
        assert T.shape == (5, 5)
        k0 = int((STATES == 0).nonzero()[0]); km = int((STATES == -1).nonzero()[0]); kp = int((STATES == 1).nonzero()[0])
        frame = float(law.energy(z, pos, None, None))
        assert abs(float(T[:, k0].sum()) - frame) <= 64 * EPS * (1.0 + abs(frame)), (float(T[:, k0].sum()), frame)
        assert float(T[0].abs().sum()) == 0.0 and float(T[3].abs().sum()) == 0.0 and float(T[4].abs().sum()) == 0.0
        A1, A2 = law.geometric_sums(z, pos, None, None)                       # [n] each, the neutral's exponents
        A1m, A2m = law.geometric_sums(z, pos, None, None, state=-1)           # the anion's exponents
        C1, C2 = law.state_coefficients(0); C1m, C2m = law.state_coefficients(-1)
        for a in (1, 2):
            want0 = C1 * float(A1[a]) + C2 * float(A2[a])
            assert abs(float(T[a, k0]) - want0) <= 64 * EPS * (1.0 + abs(want0)), (a, float(T[a, k0]), want0)
            want = C1m * float(A1m[a]) + C2m * float(A2m[a]) - want0
            assert abs(float(T[a, km] - T[a, k0]) - want) <= 64 * EPS * (1.0 + abs(want)), (a, float(T[a, km] - T[a, k0]), want)
            assert float(T[a, kp]) == float(T[a, k0])


def test_the_anion_law_reproduces_the_measured_anion_rotation():
    """Over one O at 2.26 A, the state -1 column's rotation difference E-(theta) - E-(0) (the contact and far Au both in
    state -1 is the dimer anion's bookkeeping here: the law is a one-body term per atom) equals the measured D-(theta) -
    D-(0) within twice the five-angle fit's residual (2.4 meV relaxed, 6.5 frozen) plus the far atom's own term, at
    theta = 20, 40, 60, 70 deg; the anion's anisotropy is a fraction of the neutral's (0.20 against 0.70 eV at 70 deg)."""
    from trace_redox import bond_pauli as B
    for level, meas, tol in (("almo", MEASURED_ANION_ALMO, 0.02), ("frz", MEASURED_ANION_FRZ, 0.03)):
        law = B.bond_pauli(level)
        km = int((STATES == -1).nonzero()[0]); k0 = int((STATES == 0).nonzero()[0])
        z0, p0 = dimer_over_oxygen(0.0)
        T0 = law.state_terms(z0, STATES, p0, None, None)
        for th, d in meas.items():
            if th == 0:
                continue
            z1, p1 = dimer_over_oxygen(float(th))
            T1 = law.state_terms(z1, STATES, p1, None, None)
            got = float(T1[:, km].sum() - T0[:, km].sum())
            assert abs(got - (d - meas[0])) <= tol, (level, th, got, d - meas[0])
        anion70 = float(law.state_terms(z0.clone(), STATES, dimer_over_oxygen(70.0)[1], None, None)[:, km].sum() - T0[:, km].sum())
        neutral70 = float(law.state_terms(z0.clone(), STATES, dimer_over_oxygen(70.0)[1], None, None)[:, k0].sum() - T0[:, k0].sum())
        assert 0.0 < anion70 < 0.5 * neutral70, (level, anion70, neutral70)


def test_the_state_terms_forces_are_the_gradient():
    """Autograd forces of the state -1 column's sum against central differences (off the table's knots), with periodic
    images; a finite double backward."""
    from trace_redox import bond_pauli as B
    law = B.bond_pauli("almo")
    km = int((STATES == -1).nonzero()[0])
    z, pos0 = dimer_over_oxygen(55.0, R=2.3, extra_ions=((12, (2.1325, 0.0, 0.0)),))
    cell = torch.eye(3, dtype=torch.float64) * 6.0
    pbc = (True, True, False)
    f = lambda p: law.state_terms(z, STATES, p, cell, pbc)[:, km].sum()
    pos = pos0.clone().requires_grad_(True)
    g, = torch.autograd.grad(f(pos), pos)
    for i in range(3):
        for ax in range(3):
            plus, minus = pos0.clone(), pos0.clone()
            plus[i, ax] += 1e-5
            minus[i, ax] -= 1e-5
            fd = (float(f(plus)) - float(f(minus))) / 2e-5
            assert abs(fd - float(g[i, ax])) <= 1e-7 * (1.0 + abs(fd)), (i, ax, float(g[i, ax]), fd)
    pos = pos0.clone().requires_grad_(True)
    with torch.autograd.set_detect_anomaly(True):
        g, = torch.autograd.grad(f(pos), pos, create_graph=True)
        (g ** 2).sum().backward()
    assert bool(torch.isfinite(pos.grad).all())


def test_the_module_applies_the_law_per_configuration_when_asked():
    """RedoxConfig(bond_pauli_states=True) needs bond_pauli=True and is off by default.  On a structure with two bonded Au
    over O's, every configuration shared by the two ensembles keeps its diagonal energy when no Au is at -1 (the cation
    and higher states carry the neutral law, as the frame does), and one with an Au at -1 is shifted by exactly that
    atom's (state -1 minus state 0) law entry; the off-diagonal couplings are unchanged."""
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    from trace_redox.bond_pauli import bond_pauli
    assert RedoxConfig().bond_pauli_states is False
    with pytest.raises(ValueError):
        RedoxConfig(bond_pauli_states=True)
    z = torch.tensor([8, 79, 79, 12, 8])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.3, 1.4, 4.3], [2.1325, 0.0, 0.0], [2.1325, 2.1325, 0.0]],
                       dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    H, cfgs, states = {}, {}, {}
    for flag in (False, True):
        heads = make_heads()
        torch.manual_seed(1)
        cfg = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True, bond_pauli=True,
                          bond_pauli_states=flag)
        module = RedoxElectrostatics(heads, ElectrostaticConfig(), cfg)
        h = torch.randn(5, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
        last = module._ensemble_last
        H[flag], cfgs[flag] = last["H"].detach(), last["configs"].to(torch.long)
    rows = {flag: {tuple(c.tolist()): i for i, c in enumerate(cfgs[flag])} for flag in (False, True)}
    law = bond_pauli()
    S = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
    T = law.state_terms(z, S, pos, None, None)
    shared = [c for c in rows[True] if c in rows[False]]
    with_anion = [c for c in shared if c[1] == -1 or c[2] == -1]
    without = [c for c in shared if c not in with_anion]
    assert with_anion and without, (len(with_anion), len(without))
    for c in shared:
        want = sum(float(T[a, 0] - T[a, 1]) for a in (1, 2) if c[a] == -1)
        got = float(H[True][rows[True][c], rows[True][c]] - H[False][rows[False][c], rows[False][c]])
        assert abs(got - want) <= 1e-9 * (1.0 + abs(want)), (c, got, want)
    for c in shared:
        for d in shared:
            if c != d:
                hc = float(H[True][rows[True][c], rows[True][d]] - H[False][rows[False][c], rows[False][d]])
                assert abs(hc) <= 1e-10, (c, d, hc)
