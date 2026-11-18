"""The frame pair's own Pauli wall (2026-10-31): the non-additive Thomas-Fermi kinetic energy of the frozen frame
densities times one constant fitted to frozen-density PBE (SPEC 25's overlap term holds only carrier-minus-frame
differences, so the frame pair's own repulsion was carried by the learned backbone alone, which the 4G data constrain
only above Au-Mg 2.43 A)."""
import math

import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

EPS = torch.finfo(torch.float64).eps


def test_the_frame_wall_reproduces_its_first_principles_targets():
    """At the nine frozen-density PBE points (1.8-3.49 A) the table is within 5 % at R <= 2.45 A and 0.04 eV beyond;
    and the Chebyshev table reproduces its own direct values K tf(R) to 1e-6 relative over 1.6-4.0 A."""
    from trace_redox.wall import frame_wall
    w = frame_wall()
    assert (12, 79) in w.pairs
    R, E = w.targets[(12, 79)]
    for r, e in zip(R.tolist(), E.tolist()):
        v = float(w.pair_energy(12, 79, torch.tensor([r], dtype=torch.float64))[0])
        if r <= 2.45:
            assert abs(v - e) <= 0.05 * abs(e), (r, v, e)
        else:
            assert abs(v - e) <= 0.04, (r, v, e)
    Rc, Ec = w.checks[(12, 79)]
    v = w.pair_energy(12, 79, Rc)
    assert torch.allclose(v, Ec, rtol=1e-6, atol=1e-9), float((v - Ec).abs().max())


def test_the_frame_wall_forces_are_its_gradient_and_double_backward_safe():
    """A Mg and an Au in a periodic cell (images included): autograd forces equal the finite difference of the
    energy, and a force-like loss has a finite double backward under anomaly detection."""
    from trace_redox.wall import frame_wall
    w = frame_wall()
    z = torch.tensor([12, 79, 8])
    cell = torch.eye(3, dtype=torch.float64) * 7.0
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.3, 2.1], [3.5, 3.5, 3.5]], dtype=torch.float64)
    pos = pos0.clone().requires_grad_(True)
    E = w.energy(z, pos, cell, (True, True, True))
    g, = torch.autograd.grad(E, pos)
    for i, ax in ((0, 2), (1, 0), (1, 2)):
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, ax] += 1e-5
        minus[i, ax] -= 1e-5
        fd = (float(w.energy(z, plus, cell, (True, True, True))) - float(w.energy(z, minus, cell, (True, True, True)))) / 2e-5
        assert abs(fd - float(g[i, ax])) <= 1e-6 * (1.0 + abs(fd)), (i, ax, float(g[i, ax]), fd)
    pos = pos0.clone().requires_grad_(True)
    with torch.autograd.set_detect_anomaly(True):
        E = w.energy(z, pos, cell, (True, True, True))
        g, = torch.autograd.grad(E, pos, create_graph=True)
        (g ** 2).sum().backward()
    assert bool(torch.isfinite(pos.grad).all())


def test_the_frame_wall_selects_its_pairs_and_vanishes_at_the_cutoff():
    """Only the tabulated element pairs contribute (Mg-O and Au-O give zero), the energy is additive over pairs, and
    it is zero at and beyond the table's cutoff."""
    from trace_redox.wall import frame_wall
    w = frame_wall()
    z = torch.tensor([12, 8, 79, 12])
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.1, 0.0, 0.0], [0.0, 0.0, 2.3], [30.0, 0.0, 0.0]], dtype=torch.float64)
    E = float(w.energy(z, pos, None, None))
    single = float(w.pair_energy(12, 79, torch.tensor([2.3], dtype=torch.float64))[0])
    assert abs(E - single) <= 64 * EPS * (1 + abs(single)), (E, single)
    zo = torch.tensor([12, 8])
    assert float(w.energy(zo, pos[:2], None, None)) == 0.0
    far = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, w.r_cut + 0.1]], dtype=torch.float64)
    assert float(w.energy(torch.tensor([12, 79]), far, None, None)) == 0.0


def test_the_module_adds_the_frame_wall_when_asked():
    """RedoxConfig(frame_wall=True) adds exactly the wall's energy to the module's; the default is off and leaves the
    energy bit for bit."""
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    from trace_redox.wall import frame_wall
    assert RedoxConfig().frame_wall is False
    z = torch.tensor([8, 79, 79, 12, 13, 8])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0],
                        [2.0, 2.1, 0.2]], dtype=torch.float64)
    ei = neighbour_list(pos, r_max=5.3)
    vec, length = edges(pos, ei)
    out = {}
    for flag in (False, True):
        heads = make_heads()
        torch.manual_seed(1)
        cfg = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True, frame_wall=flag)
        module = RedoxElectrostatics(heads, ElectrostaticConfig(), cfg)
        h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        out[flag] = float(module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})[0])
    wall = float(frame_wall().energy(z, pos, None, None))
    assert wall > 0.0
    assert abs(out[True] - out[False] - wall) <= 1e-10 * (1 + abs(out[False])), (out, wall)
