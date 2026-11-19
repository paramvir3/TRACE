"""Molecule-level exactness: the two-level certified fragment ensemble (trace_redox.fragments; SPEC §12).

B1 is checked against the molecule-exact target computed here independently
(integer molecular charges, mean field inside each molecule, the full
Coulomb operator, enumerated configuration by configuration, each by its own
Newton solve of a two-multiplier dual).  B2, the certificate, the percolation
weights, forces, the second derivative through the nested solve, continuity
through a breaking bond, periodic cells and the refusals are tested directly.
"""

import math

import pytest
import torch
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)
from redox_helpers import edges, lattice_graph, make_heads, neighbour_list, periodic_edges

from trace_electrostatics.config import ElectrostaticConfig
from trace_redox import RedoxConfig, RedoxElectrostatics, ensemble
from trace_redox.bound import assemble_neutral, eliminate_bound_sector
from trace_redox.fragments import (
    BOND_TOLERANCE_OFF,
    BOND_TOLERANCE_ON,
    FragmentEnsembleError,
    block_certificate,
    chemical_bonds,
    percolation_partitions,
)

EPS = torch.finfo(torch.float64).eps
HEADS = make_heads()
_A = math.radians(104.52) / 2
WATER = torch.tensor([[0.0, 0, 0], [0.9572 * math.sin(_A), 0.9572 * math.cos(_A), 0.0],
                      [-0.9572 * math.sin(_A), 0.9572 * math.cos(_A), 0.0]])
_B = math.radians(92.1) / 2
H2S = torch.tensor([[0.0, 0, 0], [1.336 * math.sin(_B), 1.336 * math.cos(_B), 0.0],
                    [-1.336 * math.sin(_B), 1.336 * math.cos(_B), 0.0]])
H_W = torch.randn(3, HEADS.irreps_in.dim, generator=torch.Generator().manual_seed(2))
H_S = torch.randn(3, HEADS.irreps_in.dim, generator=torch.Generator().manual_seed(5))


def _module(**redox):
    torch.manual_seed(1)
    return RedoxElectrostatics(HEADS, ElectrostaticConfig(), RedoxConfig(**redox))


def _energy(module, pos, z, h, Q, periodic=None):
    if periodic is None:
        ei = neighbour_list(pos.detach())
        vec, length = edges(pos, ei)
        return module(h, pos, ei, vec, length, data={"z": z, "total_charge": Q})
    cell, pbc = periodic
    ei, es = lattice_graph(pos.detach(), cell, pbc)
    vec, length = periodic_edges(pos, cell, ei, es)
    return module(h, pos, ei, vec, length, pbc=pbc, cell=cell, edge_shift=es, data={"z": z, "total_charge": Q})


def _pair(second, R):
    if second == "h2s":
        return torch.cat([WATER, H2S + torch.tensor([R, 0.0, 0.0])]), torch.tensor([8, 1, 1, 16, 1, 1]), torch.cat([H_W, H_S])
    return torch.cat([WATER, WATER + torch.tensor([R, 0.0, 0.0])]), torch.tensor([8, 1, 1, 8, 1, 1]), torch.cat([H_W, H_W])


def _newton_max(f, x, iters=200):
    """Damped Newton for a smooth concave f (autograd Hessian; tiny problems only)."""
    for _ in range(iters):
        x = x.detach().requires_grad_(True)
        val = f(x)
        (g,) = torch.autograd.grad(val, x, create_graph=True)
        if float(g.detach().abs().max()) < 1e-11:
            return float(val)
        H = torch.stack([torch.autograd.grad(g[i], x, retain_graph=True)[0] for i in range(len(x))]).detach()
        evals, evecs = torch.linalg.eigh(-0.5 * (H + H.T))
        step = evecs @ ((evecs.T @ g.detach()) / evals.clamp_min(1e-12 * float(evals.abs().max())))
        if float(step.abs().max()) > 1.0:
            step = step / float(step.abs().max())
        slope, t = float(g.detach() @ step), 1.0
        while t > 1e-14 and float(f((x + t * step).detach())) < float(val) + 1e-4 * t * slope:
            t *= 0.5
        x = (x + t * step).detach()
    raise RuntimeError("reference Newton did not converge")


def _molecule_exact(module, pos, z, h, Q, kT):
    """-kT ln sum_{N_A + N_B = Q} exp(-E(N_A, N_B)/kT): integer molecular charges, full W."""
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    _, res = module(h, pos, ei, vec, length, data={"z": z, "total_charge": Q})
    asm = assemble_neutral(h, pos, ei, vec, length, HEADS, module.config, species=z)
    el = eliminate_bound_sector(asm, res.sigma_free)
    W, b = el.W.detach(), el.u0.detach()
    J = torch.diagonal(el.J).detach()
    states = res.states
    e_hat = (res.state_energies - 0.5 * J.unsqueeze(1) * (states ** 2).unsqueeze(0)).detach()
    frag = torch.tensor([0, 0, 0, 1, 1, 1])
    energies = []
    for NA in (-1, 0, 1, 2):
        NB = Q - NA

        def dual(x):
            zz, mu = x[:6], x[6:]
            nu = mu[frag] - b - W @ zz
            return -0.5 * zz @ (W @ zz) + mu[0] * NA + mu[1] * NB + ensemble.grand_potential(e_hat, states, nu, kT).sum()

        energies.append(_newton_max(dual, torch.zeros(8)))
    energies = torch.tensor(energies)
    return float(el.e_b0) + float(-kT * torch.logsumexp(-energies / kT, 0))


# ---------------------------------------------------------------------------
# bonds and the percolation ensemble
# ---------------------------------------------------------------------------

def test_bond_switch_marks_covalent_bonds_and_not_hydrogen_bonds():
    # water dimer at its equilibrium O...O 2.91 A: the H-bond O...H ~1.95 A is not a bond
    pos = torch.cat([WATER, WATER + torch.tensor([0.0, 0.0, 2.91])])
    z = torch.tensor([8, 1, 1, 8, 1, 1])
    ei = neighbour_list(pos, r_max=4.0)
    _, length = edges(pos, ei)
    i, j, s = chemical_bonds(z, ei, length, coverage=4.0)
    pairs = sorted(zip(i.tolist(), j.tolist()))
    assert pairs == [(0, 1), (0, 2), (3, 4), (3, 5)]
    assert torch.equal(s, torch.ones(4))
    parts = percolation_partitions(6, i, j, s, 10)
    assert len(parts) == 1 and [f.tolist() for f in parts[0][0]] == [[0, 1, 2], [3, 4, 5]]


def test_percolation_weights_are_exact_probabilities():
    """Two uncertain bonds in a triangle (cycle) and one bridge: closed-form partition probabilities."""
    i = torch.tensor([0, 1, 0, 2])
    j = torch.tensor([1, 2, 2, 3])
    s = torch.tensor([1.0, 0.3, 0.6, 0.2], requires_grad=True)       # 0-1 certain; 1-2, 0-2 in a cycle; 2-3 bridge
    parts = percolation_partitions(4, i, j, s, 10)
    weights = {tuple(tuple(f.tolist()) for f in frags): w for frags, w in parts}
    total = sum(weights.values())
    assert abs(float(total) - 1.0) <= 4 * EPS
    p_triangle = 1 - (1 - 0.3) * (1 - 0.6)                            # 2 joins {0,1} if either bond holds
    assert abs(float(weights[((0, 1, 2, 3),)]) - p_triangle * 0.2) <= 4 * EPS
    assert abs(float(weights[((0, 1, 2), (3,))]) - p_triangle * 0.8) <= 4 * EPS
    assert abs(float(weights[((0, 1), (2, 3))]) - (1 - p_triangle) * 0.2) <= 4 * EPS
    assert abs(float(weights[((0, 1), (2,), (3,))]) - (1 - p_triangle) * 0.8) <= 4 * EPS
    (g,) = torch.autograd.grad(weights[((0, 1, 2, 3),)], s)
    assert abs(float(g[1]) - (1 - 0.6) * 0.2) <= 4 * EPS                # d/ds_12 of (1-(1-s12)(1-s02)) s23


def test_bonds_inside_a_molecule_are_irrelevant_and_parallel_bonds_merge():
    """Exact reductions: an uncertain bond inside a certain component changes nothing;
    two uncertain bonds between the same two components connect them with 1 - (1-a)(1-b)."""
    i = torch.tensor([0, 0, 1, 2, 1, 3])
    j = torch.tensor([1, 2, 2, 3, 3, 4])
    # certain: 0-1, 0-2 (component {0,1,2}); uncertain inside: 1-2; parallel to {3}: 2-3, 1-3; certain 3-4
    s = torch.tensor([1.0, 1.0, 0.4, 0.3, 0.5, 1.0])
    parts = percolation_partitions(5, i, j, s, 1)                        # k = 1 after the reductions
    weights = {tuple(tuple(f.tolist()) for f in frags): float(w) for frags, w in parts}
    joined = 1 - (1 - 0.3) * (1 - 0.5)
    assert abs(weights[((0, 1, 2, 3, 4),)] - joined) <= 4 * EPS
    assert abs(weights[((0, 1, 2), (3, 4))] - (1 - joined)) <= 4 * EPS
    assert len(weights) == 2


def test_too_many_uncertain_bonds_and_short_coverage_are_refused():
    with pytest.raises(FragmentEnsembleError, match="switching shell"):
        percolation_partitions(4, torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]), torch.full((3,), 0.5), 2)
    pos = torch.tensor([[0.0, 0, 0], [3.0, 0, 0]])
    z = torch.tensor([55, 55])                                          # Cs: R = 2.44 A; Cs-Cs needs 5.48 A
    ei = neighbour_list(pos)
    _, length = edges(pos, ei)
    with pytest.raises(FragmentEnsembleError, match="attention_cutoff"):
        chemical_bonds(z, ei, length, coverage=4.0)
    chemical_bonds(torch.tensor([55, 8]), ei, length, coverage=4.0)       # one Cs with O: 3.70 A, covered


def test_block_certificate_on_the_models_own_operator():
    pos, z, h = _pair("h2s", 5.0)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    module = _module()
    _, res = module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
    asm = assemble_neutral(h, pos, ei, vec, length, HEADS, module.config, species=z)
    W = eliminate_bound_sector(asm, res.sigma_free).W
    M = torch.linalg.inv(W)
    for frags in ([torch.arange(3), torch.arange(3, 6)], [torch.tensor([0]), torch.tensor([1, 2]), torch.arange(3, 6)],
                  [torch.tensor([k]) for k in range(6)]):
        D, theta, S = block_certificate(W, M, frags)
        assert float(torch.linalg.eigvalsh(S - (1 - 0.999) * W).min()) >= -1e-12 * float(W.abs().max())
        for F, DF in zip(frags, D):
            assert float(torch.linalg.eigvalsh(W[F][:, F] - DF).min()) >= -1e-12 * float(W.abs().max())
        assert bool((theta > 0).all() and (theta < 1).all())


# ---------------------------------------------------------------------------
# B1, B2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("second", ["water", "h2s"])
def test_separated_molecules_are_exact(second):
    """B1 at 30 A: the two-level energy equals the molecule-exact target; the mean field does not."""
    kT = 0.01
    pos, z, h = _pair(second, 30.0)
    exact = _molecule_exact(_module(sic="none", kT=kT, fragments="none"), pos, z, h, 1.0, kT)
    energy, res = _energy(_module(sic="none", kT=kT, fragments="bonds"), pos, z, h, 1.0)
    mf, _ = _energy(_module(sic="none", kT=kT, fragments="none"), pos, z, h, 1.0)
    # canonical ensemble (§12.13): the residual is theta < 1 only (measured 3.5 meV for the
    # water pair; the mean-constrained ensemble had kT ln 2 = 6.9 meV more)
    assert abs(float(energy) - exact) <= 0.006
    assert float(mf) < exact - 0.1                                        # the mean field's measured failure
    windows = res.molecules[0][2]
    n_mean = [float(p @ torch.tensor(w, dtype=torch.float64)) for w, p in windows]
    if second == "h2s":
        assert abs(n_mean[1] - 1.0) <= 1e-6 and abs(n_mean[0]) <= 1e-6   # the hole is an integer on H2S


@pytest.mark.parametrize("Q", [0.0, 1.0, -1.0])
def test_a_single_molecule_is_the_present_model_to_the_certificate_bound(Q):
    """B2: one molecule = the present model, except that the nested atomic removal is
    r(cW) = c r(W) (every term of the certificate scales as 1/c), so
    0 <= E_none - E_bonds <= (1 - c)/2 sum_u r_u Var_u <= (1 - c)/2 sum_u r_u max(n^2)."""
    pos = WATER.clone()
    z = torch.tensor([8, 1, 1])
    e1, r1 = _energy(_module(fragments="bonds"), pos, z, H_W, Q)
    e2, r2 = _energy(_module(fragments="none"), pos, z, H_W, Q)
    assert r1.molecules is not None and len(r1.molecules) == 1
    bound = 0.5 * (1 - 0.999) * float(r2.sic_removal.sum()) * 1.0       # states -1..+1: Var <= 1
    assert -1e-9 <= float(e2 - e1) <= bound + 1e-9
    assert abs(float(r1.q.sum()) - Q) <= 1e-9


# ---------------------------------------------------------------------------
# derivatives and continuity
# ---------------------------------------------------------------------------

def _fd(module, pos0, z, h, Q, i, a, step=1e-5, periodic=None):
    plus, minus = pos0.clone(), pos0.clone()
    plus[i, a] += step
    minus[i, a] -= step
    return (float(_energy(module, plus, z, h, Q, periodic)[0]) - float(_energy(module, minus, z, h, Q, periodic)[0])) / (2 * step)


def _stretched_water(d):
    p = WATER.clone()
    p[1] = p[1] / p[1].norm() * d
    return p


@pytest.mark.parametrize("case", ["two_molecules", "breaking_bond"])
def test_forces_are_minus_the_gradient(case):
    module = _module(fragments="bonds")
    if case == "two_molecules":
        pos0, z, h = _pair("h2s", 5.0)
        Q = 1.0
    else:
        pos0, z, h = _stretched_water(1.40), torch.tensor([8, 1, 1]), H_W   # two partitions, weights ~0.5
        Q = 1.0
    pos = pos0.clone().requires_grad_(True)
    energy, res = _energy(module, pos, z, h, Q)
    if case == "breaking_bond":
        assert len(res.molecules) == 2
    (g,) = torch.autograd.grad(energy, pos)
    for i, a in [(0, 0), (1, 1), (2, 2)]:
        fd = _fd(module, pos0, z, h, Q, i, a)
        assert abs(fd - float(g[i, a])) <= 1e-7 * (1.0 + abs(fd))


def test_energy_and_force_are_continuous_where_a_bond_enters_and_leaves_its_switch():
    """O-H through d = R_O + R_H + 0.25 A (switch onset) and + 0.60 A (bond gone):
    the second difference of E and the force on either side, at 1e-6 A."""
    module = _module(fragments="bonds")
    z = torch.tensor([8, 1, 1])
    total = 0.66 + 0.31
    for d0 in (total + BOND_TOLERANCE_ON, total + BOND_TOLERANCE_OFF):
        energies, forces = [], []
        for dd in (-1e-6, 0.0, 1e-6):
            pos = _stretched_water(d0 + dd).requires_grad_(True)
            e, _ = _energy(module, pos, z, H_W, 1.0)
            (g,) = torch.autograd.grad(e, pos)
            energies.append(float(e))
            forces.append(float(g[1] @ (pos[1] / pos[1].norm()).detach()))
        assert abs(energies[2] - 2 * energies[1] + energies[0]) <= 1e-9
        assert abs(forces[2] - forces[0]) <= 1e-5


@pytest.mark.parametrize("case", ["two_molecules", "shared_hole", "breaking_bond"])
def test_parameter_gradient_of_a_force_loss(case):
    """d/dtheta of sum F^2 + E: second derivatives through the outer dual, every inner fixed-N
    solve, the block certificate and the percolation weights.  'shared_hole' (two waters,
    <N> ~ 1/2 each) needs the canonical cross-molecule covariance in the outer Hessian."""
    module = _module(fragments="bonds", kT=0.05) if case == "shared_hole" else _module(fragments="bonds")
    g = torch.Generator().manual_seed(9)
    with torch.no_grad():
        last = module.charge_states.shift_mlp[-1]
        last.weight.copy_(torch.randn(last.weight.shape, generator=g) * 0.3)
    if case == "two_molecules":
        pos0, z, h = _pair("h2s", 5.0)
    elif case == "shared_hole":
        pos0, z, h = _pair("water", 6.0)
    else:
        pos0, z, h = _stretched_water(1.40), torch.tensor([8, 1, 1]), H_W
    weight = module.charge_states.shift_mlp[-1].weight

    def loss():
        pos = pos0.clone().requires_grad_(True)
        e, _ = _energy(module, pos, z, h, 1.0)
        (gr,) = torch.autograd.grad(e, pos, create_graph=True)
        return (gr ** 2).sum() + 0.1 * e

    (gw,) = torch.autograd.grad(loss(), weight)
    for k in (0, 7):
        with torch.no_grad():
            weight[0, k] += 1e-6
        up = float(loss())
        with torch.no_grad():
            weight[0, k] -= 2e-6
        down = float(loss())
        with torch.no_grad():
            weight[0, k] += 1e-6
        fd = (up - down) / 2e-6
        assert abs(fd - float(gw[0, k])) <= 1e-6 * (1.0 + abs(fd))


def test_periodic_box_with_two_molecules():
    cell, pbc = torch.eye(3) * 12.0, (True, True, True)
    pos0, z, h = _pair("h2s", 5.0)
    pos0 = pos0 + torch.tensor([2.0, 3.0, 4.0])
    module = _module(fragments="bonds")
    pos = pos0.clone().requires_grad_(True)
    energy, res = _energy(module, pos, z, h, 1.0, (cell, pbc))
    assert [m for m in res.molecules[0][1]] == [[0, 1, 2], [3, 4, 5]]
    (g,) = torch.autograd.grad(energy, pos)
    for i, a in [(0, 1), (3, 0)]:
        fd = _fd(module, pos0, z, h, 1.0, i, a, periodic=(cell, pbc))
        assert abs(fd - float(g[i, a])) <= 1e-7 * (1.0 + abs(fd))


def test_refusals():
    with pytest.raises(ValueError):
        RedoxConfig(fragments="bonds", orbitals=True)
    with pytest.raises(ValueError):
        RedoxConfig(fragments="bonds", sic="none", self_interaction=0.01)
    pos, z, h = _pair("water", 6.0)
    with pytest.raises(ValueError, match="integer"):
        _energy(_module(fragments="bonds"), pos, z, h, 0.5)


def test_adaptive_windows_equal_a_wide_fixed_window():
    """§12.7: a window that starts at +-1 and widens wherever an edge state is populated
    (kT = 0.5 eV populates several charge states) equals a window that is wide from the start."""
    pos, z, h = _pair("h2s", 8.0)
    narrow, r1 = _energy(_module(kT=0.5, fragments="bonds", fragment_window=1), pos, z, h, 0.0)
    wide, r2 = _energy(_module(kT=0.5, fragments="bonds", fragment_window=6), pos, z, h, 0.0)
    assert abs(float(narrow - wide)) <= 1e-7
    assert max(len(w) for w, _ in r1.molecules[0][2]) > 3                 # it did widen beyond -1..+1


# ---------------------------------------------------------------------------
# the canonical total charge (§12.13)
# ---------------------------------------------------------------------------

def _random_molecules(seed, windows):
    """Discrete-convex e_F(n) (as the fixed-N energies are) over contiguous windows."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for lo, hi in windows:
        n = torch.arange(lo, hi + 1, dtype=torch.float64)
        out.append(0.4 * torch.randn((), generator=g) * n + (0.05 + torch.rand((), generator=g)) * n ** 2)
    return out


@pytest.mark.parametrize("Q", [0, 1, -2])
def test_canonical_sums_equal_brute_force_enumeration_for_every_tilt(Q):
    """A = -kT ln sum_{sum N = Q} exp(-sum e/kT), the marginals and the pair covariances
    against explicit enumeration of all configurations, for three molecules; and the
    tilt mu (numerics only) changes nothing."""
    from trace_redox.fragments import _canonical, _cross_covariance, _tilt
    kT = 0.07
    windows = [(-2, 2), (-1, 3), (-3, 1)]
    es = _random_molecules(11, windows)
    wins = [list(range(lo, hi + 1)) for lo, hi in windows]
    confs = [(a, b, c) for a in wins[0] for b in wins[1] for c in wins[2] if a + b + c == Q]
    energies = torch.stack([es[0][a - windows[0][0]] + es[1][b - windows[1][0]] + es[2][c - windows[2][0]]
                            for a, b, c in confs])
    A_ref = float(-kT * torch.logsumexp(-energies / kT, 0))
    p_conf = torch.softmax(-energies / kT, 0)
    marg_ref = [torch.zeros(len(w), dtype=torch.float64) for w in wins]
    joint01 = torch.zeros(len(wins[0]), len(wins[1]), dtype=torch.float64)
    for p, (a, b, c) in zip(p_conf, confs):
        marg_ref[0][a - windows[0][0]] += p
        marg_ref[1][b - windows[1][0]] += p
        marg_ref[2][c - windows[2][0]] += p
        joint01[a - windows[0][0], b - windows[1][0]] += p
    C01_ref = joint01 - marg_ref[0].unsqueeze(1) * marg_ref[1].unsqueeze(0)
    mu0 = _tilt(es, wins, Q, kT)
    for mu in (mu0, mu0 + 0.3, mu0 - 0.5):
        can = _canonical(es, wins, Q, kT, mu)
        tol = 1e3 * EPS * (1.0 + abs(A_ref))
        assert abs(float(can.value) - A_ref) <= tol
        for p, ref in zip(can.marginals, marg_ref):
            assert float((p - ref).abs().max()) <= 1e3 * EPS
        assert float((_cross_covariance(can)[(0, 1)] - C01_ref).abs().max()) <= 1e3 * EPS


@pytest.mark.parametrize("ensemble_kind", ["exact", "two-level"])
def test_two_molecules_are_mirror_images_and_the_charge_is_conserved(ensemble_kind):
    """Canonical: with two molecules, N_B = Q - N_A in every configuration, so
    p_B(n) = p_A(Q - n) exactly -- a soft molecule cannot gain or lose charge with no
    counter-charge (the mean-constrained ensemble let a 108-atom MgO slab do that;
    §12.13).  Two waters at 8 A, kT = 0.3 eV: the hole is genuinely shared."""
    pos, z, h = _pair("water", 8.0)
    _, res = _energy(_module(kT=0.3, fragments="bonds", fragment_ensemble=ensemble_kind), pos, z, h, 1.0)
    (wa, pa), (wb, pb) = res.molecules[0][2]
    pa_of = dict(zip(wa, pa.tolist()))
    pb_of = dict(zip(wb, pb.tolist()))
    assert max(pa_of.values()) < 0.999                                      # genuinely mixed
    for n, p in pb_of.items():
        assert abs(p - pa_of.get(1 - n, 0.0)) <= 1e3 * EPS
    assert abs(float(res.q.sum()) - 1.0) <= 1e3 * EPS


def test_an_underflowed_connection_is_certain_and_weightless_partitions_are_not_enumerated():
    """Atom 0 has 25 bonds near their switch end (absent probability 1e-15 each) into one
    molecule {1..25}: the merged absent probability 1e-375 is 0.0 in float64, so the
    connection is certain.  Enumerating it as uncertain gave a partition of weight exactly
    0 whose unconverged potentials drove the window loop forever (Au2/MgO, §12.12)."""
    chain = [(a, a + 1) for a in range(1, 25)]
    to_zero = [(0, a) for a in range(1, 26)]
    pairs = chain + to_zero + [(0, 26)]
    i = torch.tensor([a for a, _ in pairs])
    j = torch.tensor([b for _, b in pairs])
    s = torch.tensor([1.0] * len(chain) + [1.0 - 1e-15] * len(to_zero) + [0.5], dtype=torch.float64)
    parts = percolation_partitions(27, i, j, s, 1)               # k = 1: the certain connection is not counted
    assert len(parts) == 2                                       # 26 joined or not; nothing of weight 0
    assert all(float(w) > 0.0 for _, w in parts)
    assert abs(sum(float(w) for _, w in parts) - 1.0) <= 1e3 * EPS
    sizes = sorted(sorted(int(f.numel()) for f in frags) for frags, _ in parts)
    assert sizes == [[1, 26], [27]]


def test_rare_partitions_have_exact_weights_and_negligible_ones_are_dropped():
    """Atom 0 joins A = {1, 2} and B = {3, 4} through two bonds each (absent probability 1e-10
    per bond, a = 1e-20 per connection).  Weights: joined (1 - a)^2, one connection absent
    a (1 - a) = 1e-20 -- exact, not 1 - (1 - a) = 0 in float64 -- and both absent
    a^2 = 1e-40 < PARTITION_WEIGHT_FLOOR: not enumerated."""
    from trace_redox.fragments import PARTITION_WEIGHT_FLOOR
    pairs = [(1, 2), (3, 4), (0, 1), (0, 2), (0, 3), (0, 4)]
    i = torch.tensor([a for a, _ in pairs])
    j = torch.tensor([b for _, b in pairs])
    s = torch.tensor([1.0, 1.0] + [1.0 - 1e-10] * 4, dtype=torch.float64)
    a = (1.0 - (1.0 - 1e-10)) ** 2                                             # the absent probability as built
    parts = percolation_partitions(5, i, j, s, 10)
    assert a * a < PARTITION_WEIGHT_FLOOR < a
    assert len(parts) == 3
    rare = sorted(float(w) for frags, w in parts if len(frags) == 2)
    assert len(rare) == 2
    for w in rare:
        assert abs(w - a * (1.0 - a)) <= 1e3 * EPS * a


def test_window_widening_is_weighted_by_the_partition_probability():
    """An edge population of 1 in a partition of weight 1e-30 moves the energy by ~1e-30:
    no widening.  The same edge population at weight 1 widens the window."""
    from trace_redox.fragments import FragmentSolution, _Molecule, edge_populations
    mol = _Molecule(atoms=torch.tensor([0, 1]), D=torch.eye(2), e_hat=torch.zeros(2, 3), window=[-1, 0, 1],
                    low=-2, high=2, top=torch.tensor([2, 2]))
    pops = [torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)]

    def solution(w):
        return FragmentSolution(energy=torch.zeros(()), charge=torch.zeros(2), mu_e=torch.zeros(()),
                                potentials=torch.zeros(2), partitions=[(torch.tensor(w), [mol], pops)],
                                atomic_populations=torch.zeros(2, 3), iterations=0, residual=0.0)

    assert edge_populations(solution(1e-30)) == {}
    assert edge_populations(solution(1.0)) == {(0, 1): (-2, 1)}


# ---------------------------------------------------------------------------
# the exact canonical configuration ensemble (SPEC §15), the default
# ---------------------------------------------------------------------------

def _enumerated_marginal(module, pos, z, h, Q, kT):
    """The exact target's p(N_A) by explicit enumeration (same construction as _molecule_exact)."""
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    _, res = module(h, pos, ei, vec, length, data={"z": z, "total_charge": Q})
    asm = assemble_neutral(h, pos, ei, vec, length, HEADS, module.config, species=z)
    el = eliminate_bound_sector(asm, res.sigma_free)
    W, b = el.W.detach(), el.u0.detach()
    J = torch.diagonal(el.J).detach()
    states = res.states
    e_hat = (res.state_energies - 0.5 * J.unsqueeze(1) * (states ** 2).unsqueeze(0)).detach()
    frag = torch.tensor([0, 0, 0, 1, 1, 1])
    energies = []
    for NA in (-1, 0, 1, 2):
        NB = Q - NA

        def dual(x):
            zz, mu = x[:6], x[6:]
            nu = mu[frag] - b - W @ zz
            return -0.5 * zz @ (W @ zz) + mu[0] * NA + mu[1] * NB + ensemble.grand_potential(e_hat, states, nu, kT).sum()

        energies.append(_newton_max(dual, torch.zeros(8)))
    return dict(zip((-1, 0, 1, 2), torch.softmax(-torch.tensor(energies) / kT, 0).tolist()))


@pytest.mark.parametrize("second,R", [("water", 4.0), ("water", 6.0), ("water", 30.0), ("h2s", 6.0)])
def test_the_exact_ensemble_is_the_exact_canonical_target(second, R):
    """Energy AND hole localisation equal the independent enumeration (configuration-resolved,
    full Coulomb), where the averaged two-level ensemble is off by up to 0.11 eV and puts the
    hole half on each water (§14.7, §15)."""
    kT = 0.01
    pos, z, h = _pair(second, R)
    target = _molecule_exact(_module(sic="none", kT=kT, fragments="none"), pos, z, h, 1.0, kT)
    energy, res = _energy(_module(sic="none", kT=kT, fragments="bonds"), pos, z, h, 1.0)
    assert abs(float(energy) - target) <= 1e-9
    reference = _enumerated_marginal(_module(sic="none", kT=kT, fragments="none"), pos, z, h, 1.0, kT)
    window, p = res.molecules[0][2][0]
    for n, value in zip(window, p.tolist()):
        assert abs(value - reference.get(n, 0.0)) <= 1e-8


def test_the_exact_ensemble_at_the_ionic_neutral_crossing():
    """Na...Cl at 9.43 A (Q = 0, default certified removal): the exact enumeration over
    (n, -n), to round-off.  For integer configurations the removal cancels exactly
    (-1/2 r n^2 in W~, +1/2 r n^2 in the state energies)."""
    kT = 0.01
    heads = HEADS
    h = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(4))
    z = torch.tensor([11, 17])
    pos = torch.tensor([[0.0, 0.0, 0.0], [9.43, 0.0, 0.0]])
    ei = torch.tensor([[0, 1], [1, 0]])
    vec, length = edges(pos, ei)
    ref_module = _module(kT=kT, sic="none", fragments="none")
    _, res = ref_module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
    el = eliminate_bound_sector(assemble_neutral(h, pos, ei, vec, length, heads, ref_module.config, species=z),
                                res.sigma_free)
    W, b = el.W.detach(), el.u0.detach()
    states = res.states
    e_hat = (res.state_energies - 0.5 * torch.diagonal(el.J).detach().unsqueeze(1) * states.unsqueeze(0) ** 2).detach()
    configs = [torch.tensor([float(n), float(-n)]) for n in (-1, 0, 1)]
    E = torch.stack([e_hat[0, int(c[0]) + 1] + e_hat[1, int(c[1]) + 1] + b @ c + 0.5 * c @ (W @ c) for c in configs])
    target = float(el.e_b0) - kT * float(torch.logsumexp(-E / kT, 0))
    energy, res_b = _module(kT=kT, fragments="bonds")(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
    assert abs(float(energy) - target) <= 1e3 * EPS * (1.0 + abs(target))
    q_na = float(torch.softmax(-E / kT, 0) @ torch.tensor([-1.0, 0.0, 1.0]))
    assert abs(float(res_b.free_charge[0]) - q_na) <= 1e-9


@pytest.mark.parametrize("Q", [0.0, 1.0])
def test_one_molecule_is_exactly_the_atom_level_model(Q):
    energy_b, _ = _energy(_module(fragments="bonds"), WATER.clone(), torch.tensor([8, 1, 1]), H_W, Q)
    energy_n, _ = _energy(_module(fragments="none"), WATER.clone(), torch.tensor([8, 1, 1]), H_W, Q)
    assert abs(float(energy_b - energy_n)) <= 1e3 * EPS * (1.0 + abs(float(energy_n)))


@pytest.mark.parametrize("kT", [0.05, 0.002])
def test_the_multi_constraint_solver(kT):
    """E(N) and its constraints against an autograd-Newton reference of the same dual; the
    0.002 eV case puts molecules deep in their dead zone (flat mu directions)."""
    from trace_redox.free_solver import _membership, differentiable_constrained_solution
    g = torch.Generator().manual_seed(3)
    U = 7
    X = torch.randn(U, U, generator=g)
    W = X @ X.T / U + 0.5 * torch.eye(U)
    b = torch.randn(U, generator=g)
    states = torch.tensor([-1.0, 0.0, 1.0])
    e_hat = torch.stack([2 + torch.rand(U, generator=g), torch.zeros(U), 3 + torch.rand(U, generator=g)], 1)
    group = torch.tensor([u % 3 for u in range(U)])
    N = torch.tensor([1.0, 0.0, -1.0])
    sol = differentiable_constrained_solution(b, W, e_hat, states, group, N, kT)
    G = _membership(group, 3, torch.float64, "cpu")
    assert float((G @ sol.charge - N).abs().max()) <= 1e-10

    def dual(x):
        zz, mu = x[:U], x[U:]
        nu = G.T @ mu - b - W @ zz
        return -0.5 * zz @ (W @ zz) + mu @ N + ensemble.grand_potential(e_hat, states, nu, kT).sum()

    assert abs(float(sol.energy) - _newton_max(dual, torch.zeros(U + 3))) <= 1e3 * EPS * (1.0 + abs(float(sol.energy)))
    assert abs(sol.duality_gap) <= 1e-9


def test_a_molecule_at_its_capacity_is_fixed_exactly():
    """configuration_energy with one molecule at its upper bound (every atom in its top state)
    equals the direct evaluation for that molecule plus the constrained solve of the rest."""
    from trace_redox.fragments import _Molecule, configuration_energy
    from trace_redox.free_solver import differentiable_solution
    g = torch.Generator().manual_seed(11)
    U = 5
    X = torch.randn(U, U, generator=g)
    W = X @ X.T / U + 0.5 * torch.eye(U)
    b = torch.randn(U, generator=g)
    states = torch.tensor([-1.0, 0.0, 1.0])
    e_hat = torch.stack([1 + torch.rand(U, generator=g), torch.zeros(U), 1.5 + torch.rand(U, generator=g)], 1)
    top = torch.full((U,), 2)
    A = _Molecule(atoms=torch.tensor([0, 1]), D=torch.eye(2), e_hat=e_hat[:2], window=[0, 1, 2], low=-2, high=2, top=top[:2])
    B = _Molecule(atoms=torch.tensor([2, 3, 4]), D=torch.eye(3), e_hat=e_hat[2:], window=[-3, -2, -1, 0], low=-3, high=3, top=top[2:])
    E, c, _, _ = configuration_energy((2, -1), [A, B], b, W, e_hat, states, 0.05, 1e-12)
    cA = torch.tensor([1.0, 1.0])
    rest = torch.tensor([2, 3, 4])
    sol = differentiable_solution(b[rest] + W[rest][:, :2] @ cA, W[rest][:, rest], e_hat[rest], states, -1.0, 0.05, tol=1e-12)
    direct = e_hat[0, 2] + e_hat[1, 2] + b[:2] @ cA + 0.5 * cA @ (W[:2, :2] @ cA) + sol.energy
    assert abs(float(E - direct)) <= 1e3 * EPS * (1.0 + abs(float(direct)))
    assert torch.equal(c[:2], cA)


def test_the_configuration_search_is_complete():
    """select_configurations returns exactly {N : sum N = Q, log p(N) >= log CONFIG_EPS_LO}."""
    import itertools
    from trace_redox.fragments import CONFIG_EPS_LO, select_configurations
    g = torch.Generator().manual_seed(2)
    windows = [[-2, -1, 0, 1, 2], [-1, 0, 1, 2], [-2, -1, 0, 1], [0, 1]]
    a = [(-40.0 * torch.rand(len(w), generator=g) * torch.arange(len(w), dtype=torch.float64)).tolist() for w in windows]
    Q = 1
    full = [N for N in itertools.product(*windows) if sum(N) == Q]
    logs = {N: sum(a[F][windows[F].index(n)] for F, n in enumerate(N)) for N in full}
    log_Z = float(torch.logsumexp(torch.tensor(list(logs.values())), 0))
    expected = {N for N, v in logs.items() if v - log_Z >= math.log(CONFIG_EPS_LO)}
    found = set(select_configurations(a, windows, Q, log_Z, cap=10_000))
    assert found == expected and len(expected) < len(full)


def test_the_inclusion_weight_is_c2():
    from trace_redox.fragments import CONFIG_EPS_HI, CONFIG_EPS_LO, inclusion_weight
    lo, hi = math.log(CONFIG_EPS_LO), math.log(CONFIG_EPS_HI)
    x = torch.tensor([lo, hi], requires_grad=True)
    w = inclusion_weight(x)
    assert float(w[0]) == 0.0 and float(w[1]) == 1.0
    for point in (lo + 1e-9, hi - 1e-9):
        xp = torch.tensor(point, requires_grad=True)
        (d1,) = torch.autograd.grad(inclusion_weight(xp), xp, create_graph=True)
        (d2,) = torch.autograd.grad(d1, xp)
        assert abs(float(d1)) <= 1e-12 and abs(float(d2)) <= 1e-6


# ---------------------------------------------------------------------------
# the exact ensemble: values against independent enumeration, invariance, the band
# ---------------------------------------------------------------------------

def _model_operators(pos, z, h, Q, kT, sic, max_charge=1):
    """W, b, e_hat (removal-added), W~ (removal-subtracted), E_b0 of the module's own heads,
    rebuilt independently of the fragment code."""
    from trace_redox.sic import certified_removal
    module = _module(sic="none", kT=kT, fragments="none", max_charge=max_charge)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    _, res = module(h, pos, ei, vec, length, data={"z": z, "total_charge": Q})
    el = eliminate_bound_sector(assemble_neutral(h, pos, ei, vec, length, HEADS, module.config, species=z), res.sigma_free)
    W, b, J = el.W.detach(), el.u0.detach(), el.J.detach()
    states = res.states
    e_hat = (res.state_energies - 0.5 * torch.diagonal(J).unsqueeze(1) * states.unsqueeze(0) ** 2).detach()
    if sic == "certified":
        r = certified_removal(W, J, theta=0.999, smoothing=0.01).detach()
        W = W - torch.diag(r)
        W = 0.5 * (W + W.T)
        e_hat = e_hat + 0.5 * r.unsqueeze(1) * states.unsqueeze(0) ** 2
    return W, b, e_hat, states, float(el.e_b0)


def _enumerated_partition(W, b, e_hat, states, molecules, Q, kT):
    """-kT ln sum_N exp(-E(N)/kT) over every N with sum N = Q inside the capacities.

    A molecule at a capacity bound has every atom in its extreme state (fixed; a one-atom
    molecule's charged states are all bounds); the rest maximise the multi-constraint dual with
    an autograd-Hessian Newton -- no trace_redox solver."""
    import itertools
    finite = torch.isfinite(e_hat)
    top = finite.sum(1) - 1
    lo = [int(-len(a)) for a in molecules]
    hi = [int(sum(float(states[top[u]]) for u in a)) for a in molecules]
    energies = []
    for N in itertools.product(*[range(l, h + 1) for l, h in zip(lo, hi)]):
        if sum(N) != Q:
            continue
        fixed, index, free_mols, targets = [], {}, [], []
        for atoms, n, l, h in zip(molecules, N, lo, hi):
            if n in (l, h):
                for u in atoms:
                    fixed.append(u)
                    index[u] = 0 if n == l else int(top[u])
            else:
                free_mols.append(atoms)
                targets.append(float(n))
        c_f = torch.tensor([float(states[index[u]]) for u in fixed])
        f = torch.tensor(fixed, dtype=torch.long)
        energy = 0.0
        if fixed:
            energy = float(sum(e_hat[u, index[u]] for u in fixed) + b[f] @ c_f + 0.5 * c_f @ (W[f][:, f] @ c_f))
        if free_mols:
            free = torch.tensor([u for atoms in free_mols for u in atoms], dtype=torch.long)
            bb = b[free] + (W[free][:, f] @ c_f if fixed else 0.0)
            WW, ee = W[free][:, free], e_hat[free]
            G = torch.zeros(len(free_mols), int(free.numel()))
            k = 0
            for F, atoms in enumerate(free_mols):
                G[F, k:k + len(atoms)] = 1.0
                k += len(atoms)
            Nt = torch.tensor(targets)

            def dual(x, WW=WW, bb=bb, ee=ee, G=G, Nt=Nt):
                zz, mu = x[:WW.shape[0]], x[WW.shape[0]:]
                nu = G.T @ mu - bb - WW @ zz
                return -0.5 * zz @ (WW @ zz) + mu @ Nt + ensemble.grand_potential(ee, states, nu, kT).sum()

            energy += _newton_max(dual, torch.zeros(int(free.numel()) + len(free_mols)))
        energies.append(energy)
    return float(-kT * torch.logsumexp(-torch.tensor(energies) / kT, 0))


def test_two_partitions_have_the_value_of_independent_enumeration():
    """A breaking O-H bond (pi ~ 1/2 each; default certified removal, Q = +1): the module's
    E = E_b0 + sum_P pi_P F_P with each F_P the canonical sum over molecular configurations."""
    kT = 0.05
    pos, z = _stretched_water(1.40), torch.tensor([8, 1, 1])
    energy, res = _energy(_module(kT=kT, fragments="bonds"), pos, z, H_W, 1.0)
    assert len(res.molecules) == 2 and all(0.01 < w < 0.99 for w, _, _ in res.molecules)
    W, b, e_hat, states, e_b0 = _model_operators(pos, z, H_W, 1.0, kT, "certified")
    total = e_b0
    for weight, molecules, _ in res.molecules:
        total += weight * _enumerated_partition(W, b, e_hat, states, molecules, 1, kT)
    assert abs(float(energy) - total) <= 1e-9


def test_dications_against_enumeration():
    """max_charge = 2 (H has no +2 state: an absent, +inf, state inside the constrained solver);
    two waters at 6 A with Q = +2: (2, 0), (1, 1), (0, 2) and the rest, exactly."""
    kT = 0.05
    pos, z, h = _pair("water", 6.0)
    energy, _ = _energy(_module(sic="none", kT=kT, fragments="bonds", max_charge=2), pos, z, h, 2.0)
    W, b, e_hat, states, e_b0 = _model_operators(pos, z, h, 2.0, kT, "none", max_charge=2)
    target = e_b0 + _enumerated_partition(W, b, e_hat, states, [[0, 1, 2], [3, 4, 5]], 2, kT)
    assert abs(float(energy) - target) <= 1e-9


def test_the_exact_ensemble_is_invariant_under_atom_permutation():
    """Relabelling atoms (inside and across molecules, molecule order swapped) leaves E unchanged
    and permutes the forces: the search order and molecule indexing carry no physics."""
    pos0, z, h = _pair("h2s", 5.0)
    perm = torch.tensor([4, 3, 5, 1, 0, 2])
    results = []
    for p in (torch.arange(6), perm):
        pos = pos0[p].clone().requires_grad_(True)
        energy, _ = _energy(_module(kT=0.1, fragments="bonds"), pos, z[p], h[p], 1.0)
        (g,) = torch.autograd.grad(energy, pos)
        results.append((float(energy), g, p))
    (e0, g0, _), (e1, g1, p) = results
    assert abs(e1 - e0) <= 1e4 * EPS * (1.0 + abs(e0))
    assert float((g1 - g0[p]).abs().max()) <= 1e4 * EPS * (1.0 + float(g0.abs().max()))


def _h2s_pair_at(R):
    pos, z, h = _pair("h2s", R)
    return pos, z, h


def test_forces_and_force_loss_gradient_inside_the_inclusion_band():
    """kT = 0.1 eV (the training default), H2O...H2S+ at 6 A: the water-hole configuration has
    screening probability ~3e-11, inside the band where w_N rises.  Forces equal -dE/dx and the
    force-loss parameter gradient equals finite differences."""
    module = _module(kT=0.1, fragments="bonds")
    _, res2 = _energy(_module(kT=0.1, fragments="bonds", fragment_ensemble="two-level"), *_h2s_pair_at(6.0), 1.0)
    w2, p2 = res2.molecules[0][2][0]
    assert 1e-12 < float(p2[w2.index(1)]) < 1e-10                       # the band is exercised
    pos0, z, h = _h2s_pair_at(6.0)
    pos = pos0.clone().requires_grad_(True)
    energy, _ = _energy(module, pos, z, h, 1.0)
    (g,) = torch.autograd.grad(energy, pos)
    for i, a in [(0, 0), (3, 0), (4, 1)]:
        fd = _fd(module, pos0, z, h, 1.0, i, a)
        assert abs(fd - float(g[i, a])) <= 1e-7 * (1.0 + abs(fd))
    weight = module.charge_states.shift_mlp[-1].weight

    def loss():
        p = pos0.clone().requires_grad_(True)
        e, _ = _energy(module, p, z, h, 1.0)
        (gr,) = torch.autograd.grad(e, p, create_graph=True)
        return (gr ** 2).sum() + 0.1 * e

    (gw,) = torch.autograd.grad(loss(), weight)
    for k in (0, 7):
        with torch.no_grad():
            weight[0, k] += 1e-6
        up = float(loss())
        with torch.no_grad():
            weight[0, k] -= 2e-6
        down = float(loss())
        with torch.no_grad():
            weight[0, k] += 1e-6
        fd = (up - down) / 2e-6
        assert abs(fd - float(gw[0, k])) <= 1e-6 * (1.0 + abs(fd))


def test_energy_and_force_are_continuous_where_a_configuration_leaves_the_band():
    """The separation R* at which the water-hole configuration's screening probability is exactly
    CONFIG_EPS_LO (found by bisection, kT = 0.1): E and the force along R either side, 1e-6 A apart."""
    from trace_redox.fragments import CONFIG_EPS_LO
    two_level = _module(kT=0.1, fragments="bonds", fragment_ensemble="two-level")

    def p_water(R):
        _, res = _energy(two_level, *_h2s_pair_at(R), 1.0)
        w, p = res.molecules[0][2][0]
        return float(p[w.index(1)])

    lo, hi = 8.0, 20.0
    assert p_water(lo) > CONFIG_EPS_LO > p_water(hi)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if p_water(mid) > CONFIG_EPS_LO else (lo, mid)
    R_star = 0.5 * (lo + hi)
    module = _module(kT=0.1, fragments="bonds")
    energies, forces = [], []
    for dR in (-1e-6, 0.0, 1e-6):
        pos0, z, h = _h2s_pair_at(R_star + dR)
        pos = pos0.clone().requires_grad_(True)
        e, _ = _energy(module, pos, z, h, 1.0)
        (g,) = torch.autograd.grad(e, pos)
        energies.append(float(e))
        forces.append(float(g[3:, 0].sum()))                         # d E / d R
    assert abs(energies[2] - 2 * energies[1] + energies[0]) <= 1e-12
    assert abs(forces[2] - forces[0]) <= 1e-6
