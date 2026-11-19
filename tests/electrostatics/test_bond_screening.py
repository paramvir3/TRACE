"""Pauli bond screening of the split-charge channel (docs/ELECTROSTATICS_AUDIT_2026-10-09.md, step 1).

A third atom k between i and j blocks the direct i-j orbital overlap; charge then moves through k's
own bonds.  The screening of the direct channel is the many-body factor of MEAM (Baskes) and of
environment-dependent tight binding (Tang, Wang, Chan & Ho, PRB 53, 979 (1996)):

    X_ik = (r_ik / r_ij)^2,  X_kj = (r_kj / r_ij)^2,
    C_ikj = [2 (X_ik + X_kj) - (X_ik - X_kj)^2 - 1] / [1 - (X_ik - X_kj)^2]   (k inside the ellipse),
    S_ij = prod_k s(C_ikj),  s = quintic C^2 switch, 0 for C <= C_min, 1 for C >= C_max,
    and s = 1 when (X_ik - X_kj)^2 >= 1 (k beyond an endpoint).

Values derived by hand for the tests (C_min = 2.0, C_max = 2.8):
* rocksalt second shell (cation-cation, anion-anion at a/sqrt(2)): the two bridging anions sit on
  the sphere of diameter r_ij, X_ik = X_kj = 1/2, C = 1 < C_min, so S = 0 exactly;
* rocksalt and fcc nearest neighbours: S = 1 (no atom between them inside the ellipse);
* fcc second neighbours (a): bridged by face atoms with X = 1/2, C = 1, S = 0;
* a linear chain's 1-3 pair: the middle atom has X = 1/4, 1/4, C = 0, S = 0.
"""
import itertools
import math

import pytest
import torch
from conftest import require

C_MIN, C_MAX = 2.0, 2.8
EPS = torch.finfo(torch.float64).eps


def _bonds(pos, cell, cutoff, span=2):
    """Unique bonds (lo <= hi, one per image) within ``cutoff``: (bond_index [2, nb], vector lo -> hi)."""
    lo, hi, vec = [], [], []
    n = len(pos)
    shifts = [torch.zeros(3, dtype=torch.float64)] if cell is None else [
        torch.tensor(s, dtype=torch.float64) @ cell for s in itertools.product(range(-span, span + 1), repeat=3)]
    for i in range(n):
        for j in range(i, n):
            for t in shifts:
                d = pos[j] + t - pos[i]
                if float(d.norm()) < 1e-9 or float(d.norm()) >= cutoff:
                    continue
                if i == j:
                    # one bond per +/- self-image pair: keep the image whose first nonzero
                    # Cartesian component is positive
                    first = next(float(x) for x in t if abs(float(x)) > 1e-12)
                    if first < 0:
                        continue
                lo.append(i)
                hi.append(j)
                vec.append(d)
    return torch.tensor([lo, hi]), torch.stack(vec)


def _switch(x):
    x = min(max(x, 0.0), 1.0)
    return x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x)


def _brute_force(pos, cell, bond_index, bond_vector, reach=8.0, span=3):
    """MEAM screening over EVERY periodic site, no neighbour list: the independent reference."""
    n = len(pos)
    shifts = [torch.zeros(3, dtype=torch.float64)] if cell is None else [
        torch.tensor(s, dtype=torch.float64) @ cell for s in itertools.product(range(-span, span + 1), repeat=3)]
    out = []
    for b in range(bond_index.shape[1]):
        i = int(bond_index[0, b])
        ri = pos[i]
        rj = ri + bond_vector[b]
        rij2 = float(bond_vector[b] @ bond_vector[b])
        S = 1.0
        for k in range(n):
            for t in shifts:
                rk = pos[k] + t
                dik, dkj = rk - ri, rj - rk
                if float(dik.norm()) < 1e-9 or float(dkj.norm()) < 1e-9 or float(dik.norm()) > reach:
                    continue
                Xik, Xkj = float(dik @ dik) / rij2, float(dkj @ dkj) / rij2
                D = (Xik - Xkj) ** 2
                if D >= 1.0:
                    continue
                C = (2.0 * (Xik + Xkj) - D - 1.0) / (1.0 - D)
                S *= _switch((C - C_MIN) / (C_MAX - C_MIN))
        out.append(S)
    return torch.tensor(out)


def _screening():
    return require("screening", "bond_screening")


def _rocksalt(a=4.212):
    frac = torch.tensor([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                         [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]], dtype=torch.float64)
    cell = a * torch.eye(3, dtype=torch.float64)
    return frac @ cell, cell


def test_rocksalt_first_shell_open_second_shell_closed():
    bond_screening = _screening()
    pos, cell = _rocksalt()
    bond_index, vec = _bonds(pos, cell, cutoff=3.2)              # first (2.106) and second (2.978) shells
    S = bond_screening(bond_index, vec, n_atoms=len(pos), c_min=C_MIN, c_max=C_MAX)
    length = vec.norm(dim=1)
    nn, nnn = length < 2.2, (length > 2.9) & (length < 3.0)
    assert int(nn.sum()) == 8 * 6 // 2 and int(nnn.sum()) == 8 * 12 // 2
    assert torch.equal(S[nn], torch.ones(int(nn.sum()), dtype=torch.float64))
    assert torch.equal(S[nnn], torch.zeros(int(nnn.sum()), dtype=torch.float64))


def test_fcc_metal_keeps_nearest_neighbour_bonds_and_screens_second_neighbours():
    bond_screening = _screening()
    a = 4.086
    frac = torch.tensor([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], dtype=torch.float64)
    cell = a * torch.eye(3, dtype=torch.float64)
    pos = frac @ cell
    bond_index, vec = _bonds(pos, cell, cutoff=4.2)
    S = bond_screening(bond_index, vec, n_atoms=4, c_min=C_MIN, c_max=C_MAX)
    length = vec.norm(dim=1)
    nn, nnn = length < 3.0, length > 4.0
    assert torch.equal(S[nn], torch.ones(int(nn.sum()), dtype=torch.float64))
    assert torch.equal(S[nnn], torch.zeros(int(nnn.sum()), dtype=torch.float64))


def test_linear_chain_screens_the_one_three_pair():
    bond_screening = _screening()
    pos = torch.tensor([[0.0, 0, 0], [1.3, 0, 0], [2.6, 0, 0]], dtype=torch.float64)
    bond_index, vec = _bonds(pos, None, cutoff=3.0)
    S = bond_screening(bond_index, vec, n_atoms=3, c_min=C_MIN, c_max=C_MAX)
    pairs = {(int(i), int(j)): float(s) for i, j, s in zip(bond_index[0], bond_index[1], S)}
    assert pairs == {(0, 1): 1.0, (1, 2): 1.0, (0, 2): 0.0}


@pytest.mark.parametrize("periodic", [False, True])
def test_matches_brute_force_over_all_sites(periodic):
    """Neighbour-list implementation == the definition summed over every site and image."""
    bond_screening = _screening()
    g = torch.Generator().manual_seed(7)
    n = 9
    cell = torch.tensor([[6.0, 0.3, 0.0], [0.2, 5.7, 0.4], [0.0, 0.5, 6.4]], dtype=torch.float64) if periodic else None
    box = torch.diag(torch.tensor([6.0, 5.7, 6.4], dtype=torch.float64))
    pos = torch.rand(n, 3, generator=g, dtype=torch.float64) @ (cell if periodic else box)
    bond_index, vec = _bonds(pos, cell, cutoff=4.0)
    S = bond_screening(bond_index, vec, n_atoms=n, c_min=C_MIN, c_max=C_MAX)
    ref = _brute_force(pos, cell, bond_index, vec)
    assert float((S - ref).abs().max()) <= 64 * EPS
    assert float(S.min()) >= 0.0 and float(S.max()) <= 1.0
    assert 0 < int(((S > 0) & (S < 1)).sum())                      # the switch region is exercised


def test_invariances_and_orientation():
    bond_screening = _screening()
    g = torch.Generator().manual_seed(11)
    pos = 4.0 * torch.rand(10, 3, generator=g, dtype=torch.float64)
    bond_index, vec = _bonds(pos, None, cutoff=3.5)
    S = bond_screening(bond_index, vec, n_atoms=10, c_min=C_MIN, c_max=C_MAX)
    # rigid rotation (bond vectors rotate) leaves S unchanged
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    S_rot = bond_screening(bond_index, vec @ q.T, n_atoms=10, c_min=C_MIN, c_max=C_MAX)
    assert float((S - S_rot).abs().max()) <= 64 * EPS
    # reversing a bond (swap endpoints, negate its vector) leaves its S unchanged
    flipped = torch.stack((bond_index[1], bond_index[0]))
    S_flip = bond_screening(flipped, -vec, n_atoms=10, c_min=C_MIN, c_max=C_MAX)
    assert float((S - S_flip).abs().max()) <= 64 * EPS


def test_gradient_is_exact_and_continuous():
    """Autograd = central differences, including across the switch (C^2)."""
    bond_screening = _screening()
    # a third atom moved through the switch region of the 0-1 bond
    base = torch.tensor([[0.0, 0, 0], [2.8, 0, 0]], dtype=torch.float64)

    def total(y):
        third = torch.stack((torch.tensor(1.4, dtype=torch.float64), y, torch.tensor(0.0, dtype=torch.float64)))
        return _three(torch.cat((base, third.unsqueeze(0))))

    def _three(pos):
        bond_index, vec = _bonds(pos, None, cutoff=10.0)
        S = bond_screening(bond_index, vec, n_atoms=3, c_min=C_MIN, c_max=C_MAX)
        return S[(bond_index[0] == 0) & (bond_index[1] == 1)].sum()

    # on the bisector C = y^2 / (x (1 - x)) = 4 (y / r_ij)^2: y = 1.75 .. 2.55 A spans C = 1.56 .. 3.32,
    # i.e. both plateaus and the whole switch (C_min = 2.0 at y = 1.98 A, C_max = 2.8 at y = 2.34 A)
    ys = torch.linspace(1.75, 2.55, 81, dtype=torch.float64)
    h = 1e-6
    values, slopes = [], []
    for y in ys:
        y = y.clone().requires_grad_(True)
        v = total(y)
        (g,) = torch.autograd.grad(v, y)
        fd = (float(total(y.detach() + h)) - float(total(y.detach() - h))) / (2 * h)
        assert abs(float(g) - fd) <= 1e-7 * (1 + abs(fd))
        values.append(float(v))
        slopes.append(float(g))
    assert min(values) == 0.0 and max(values) == 1.0               # both ends of the switch reached
    jumps = [abs(slopes[k + 1] - slopes[k]) for k in range(len(slopes) - 1)]
    assert max(jumps) < 0.5 * max(abs(s) for s in slopes)          # no kink: the slope varies smoothly


# ---------------------------------------------------------------------------
# Integration: ElectrostaticConfig(bond_screening=(C_min, C_max)) in the assembly
# ---------------------------------------------------------------------------
def test_config_validates_and_defaults_off():
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    assert ElectrostaticConfig().bond_screening is None
    assert ElectrostaticConfig(bond_screening=[2.0, 2.8]).bond_screening == (2.0, 2.8)
    for bad in ((2.8, 2.0), (-0.1, 1.0), (1.0,), "on"):
        with pytest.raises((ValueError, TypeError)):
            ElectrostaticConfig(bond_screening=bad)


def _rocksalt_state(screening, displace=0.0):
    import sys
    sys.path.insert(0, "tests/electrostatics")
    from test_periodic import _config, _heads, _state
    heads = _heads(r_max=5.0)
    torch.manual_seed(2)
    h = torch.randn(8, heads.irreps_in.dim)
    pos, cell = _rocksalt()
    pos = pos.clone()
    pos[0, 2] += displace
    config = _config(ewald_real_cutoff=12.0, solver="dense", bond_screening=screening)
    return _state(heads, h, pos, cell, 3.2, config), cell


def test_screening_removes_the_second_shell_channel_exactly():
    """Theorem 1 holds with screening, and the rocksalt second shell drops out of eps_inf exactly."""
    dielectric_tensor = require("observables", "dielectric_tensor")
    K_E = 14.399645
    state_off, cell = _rocksalt_state(None)
    state_on, _ = _rocksalt_state((C_MIN, C_MAX))
    p_off, p_on = state_off.assembly.bond_dipole.detach(), state_on.assembly.bond_dipole.detach()
    nnn = (p_off.norm(dim=1) > 2.9) & (p_off.norm(dim=1) < 3.0)
    assert int(nnn.sum()) > 0 and torch.equal(p_on[nnn], torch.zeros_like(p_on[nnn]))
    assert torch.equal(p_on[~nnn], p_off[~nnn])                     # first shell untouched
    eps = {k: dielectric_tensor(s.solve_result, s.assembly, route="minimiser").detach()
           for k, s in (("off", state_off), ("on", state_on))}
    asm = state_on.assembly
    kappa = asm.params.kappa.detach()
    nb, n = asm.n_bonds, asm.n_atoms
    H = asm.dense_hessian().detach()
    mu = torch.arange(nb + n, H.shape[0])
    D = asm.dipole_derivative().detach()[mu]
    dip = D.T @ torch.linalg.solve(H[mu][:, mu], D)
    predicted = torch.eye(3) + 4 * math.pi * K_E / float(torch.linalg.det(cell)) * ((p_on / kappa.unsqueeze(1)).T @ p_on + dip)
    free = torch.cat((torch.arange(nb), mu))
    tol = 64.0 * float(torch.linalg.cond(H[free][:, free])) * EPS
    assert float((eps["on"] - predicted).abs().max()) / float(eps["on"].abs().max()) <= tol
    assert float(torch.diagonal(eps["on"]).max()) < float(torch.diagonal(eps["off"]).min())


def test_forces_include_the_screening_exactly():
    """-dE/dR by autograd equals central differences with screening on (third atoms move S)."""
    import sys
    sys.path.insert(0, "tests/electrostatics")
    from test_periodic import _config, _heads
    electrostatic_state = require("energy", "electrostatic_state")
    heads = _heads(r_max=4.0)
    torch.manual_seed(4)
    n = 6
    h = torch.randn(n, heads.irreps_in.dim)
    g = torch.Generator().manual_seed(9)
    base = torch.tensor([[0.0, 0, 0], [1.5, 0.1, 0], [3.0, 0.0, 0.2], [1.4, 1.6, 0.1],
                         [0.2, 1.5, 1.2], [2.9, 1.5, 1.3]], dtype=torch.float64)
    base = base + 0.05 * torch.randn(n, 3, generator=g, dtype=torch.float64)
    config = _config(bond_screening=(C_MIN, C_MAX))

    def energy(pos):
        src, dst = zip(*[(i, j) for i in range(n) for j in range(n) if i != j and float((pos[i] - pos[j]).norm()) < 3.9])
        ei = torch.tensor([src, dst])
        vec = pos[ei[1]] - pos[ei[0]]
        return electrostatic_state(h, pos, ei, vec, vec.norm(dim=-1), heads, config).energy

    pos = base.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    step = 1e-5
    for atom, axis in ((2, 0), (3, 1), (4, 2)):
        plus, minus = base.clone(), base.clone()
        plus[atom, axis] += step
        minus[atom, axis] -= step
        fd = (float(energy(plus)) - float(energy(minus))) / (2 * step)
        assert abs(float(grad[atom, axis]) - fd) <= 1e-6 * (1.0 + abs(fd))
