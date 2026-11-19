"""Periodic systems: the Gram lattice operator, the bond-current polarization,
and periodic dispersion (SPEC 13.1, 14, 15).

Each claim the periodic extension rests on is asserted here, with the number
that was measured when it was built:

SPEC 14 -- the lattice Gram operator (Gaussian q, mu, Theta)
* every block is independent of the Ewald splitting parameter;
* its charge block is the package's existing, independently written Ewald plus
  exactly the Gram background term;
* the lattice Hessian is positive definite at the parameter floors;
* a neutral cluster in a growing cubic box tends to the open-boundary energy
  minus the tinfoil term, with an L^-5 residual;
* the energy per cell is the same for a cell and its supercell.

SPEC 13.1 -- periodic dispersion
* folding onto a supercell converges; the non-analytic Gamma term makes it
  converge fast; the result is rotation invariant.

SPEC 15 -- the bond-current polarization and the electric enthalpy
* a monatomic chain (q = 0 by symmetry) polarizes through its self-image bond,
  exactly as the analytic value says;
* an open chain's polarizability per atom tends to the periodic one;
* P is unchanged when an atom is wrapped into another image (sum q r is not);
* Born charges from the enthalpy equal those of the minimiser (Maxwell), and
  obey the acoustic sum rule;
* eps_inf is intensive -- identical for a cell and its supercell -- which it
  was not when self-image bonds were dropped;
* the enthalpy is exactly quadratic in the field.

SPEC 15.2 -- the fixed-k degenerate point, fallback and error.
"""

from __future__ import annotations

import itertools
import math
import warnings

import pytest
import torch

from conftest import require

K_E = 14.399645


def _spd(n, d, floor, generator, scale=0.5):
    factor = scale * torch.randn(n, d, d, generator=generator)
    return factor @ factor.transpose(-1, -2) + floor * torch.eye(d)


def _neighbours(pos, cell, cutoff, span=2):
    src, dst, shift = [], [], []
    for i in range(len(pos)):
        for j in range(len(pos)):
            for n in itertools.product(range(-span, span + 1), repeat=3):
                if i == j and n == (0, 0, 0):
                    continue
                vector = pos[i] - pos[j] + torch.tensor(n, dtype=torch.float64) @ cell
                if vector.norm() < cutoff:
                    src.append(i)
                    dst.append(j)
                    shift.append(n)
    edge_index = torch.tensor([src, dst])
    edge_shift = torch.tensor(shift, dtype=torch.float64)
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]] + edge_shift @ cell
    return edge_index, edge_shift, edge_vec


def _heads(seed=0, **kwargs):
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return require("heads", "ElectrostaticHeads")(**kwargs)


def _state(heads, h, pos, cell, cutoff, config, field=None, quadrupoles=False):
    electrostatic_state = require("energy", "electrostatic_state")
    edge_index, edge_shift, edge_vec = _neighbours(pos, cell, cutoff)
    return electrostatic_state(
        h, pos, edge_index, edge_vec, edge_vec.norm(dim=-1), heads, config,
        pbc=(True, True, True), cell=cell, edge_shift=edge_shift,
        external_field=field, quadrupoles=quadrupoles,
    )


def _supercell(pos, cell, h, M):
    shifts = torch.tensor(list(itertools.product(range(M), repeat=3)), dtype=torch.float64)
    return (pos[None] + (shifts @ cell)[:, None]).reshape(-1, 3), M * cell, h.repeat(M ** 3, 1)


def _config(**kwargs):
    return require("config", "ElectrostaticConfig")(**kwargs)


# ===========================================================================
# SPEC 14: the Gram lattice operator
# ===========================================================================
@pytest.fixture(scope="module")
def lattice_case():
    width = require("kernels", "gaussian_multipole_width")
    g = torch.Generator().manual_seed(2)
    n = 5
    cell = torch.tensor([[7.0, 0.3, 0.0], [0.2, 6.5, 0.4], [0.0, 0.5, 7.5]])
    pos = torch.rand(n, 3, generator=g) @ cell
    sigma = 0.8 + 0.4 * torch.rand(n, generator=g)
    ell = 0.6 + 0.6 * torch.rand(n, generator=g)
    A_mu, A_theta = _spd(n, 3, 3.0, g), _spd(n, 5, 2.0, g)
    return dict(
        pos=pos, cell=cell, sigma=sigma, ell=ell, A_mu=A_mu, A_theta=A_theta,
        sigma_mu=width(A_mu, ell, 1, 0.5), sigma_theta=width(A_theta, ell, 2, 0.5),
    )


def _lattice(case, alpha, r_cut=14.0, accuracy=1e-14):
    gaussian_lattice_blocks = require("lattice", "gaussian_lattice_blocks")
    EwaldParams = require("ewald", "EwaldParams")
    ewald = EwaldParams.auto_tune(case["cell"], r_cut, accuracy=accuracy, alpha=alpha)
    return gaussian_lattice_blocks(
        case["pos"], case["cell"], case["sigma"], case["sigma_mu"], case["ell"], ewald,
        sigma_theta=case["sigma_theta"],
    )


def test_lattice_blocks_are_independent_of_the_ewald_split(lattice_case):
    reference = _lattice(lattice_case, 0.30)
    for alpha in (0.24, 0.36):
        other = _lattice(lattice_case, alpha)
        for key, block in reference.items():
            assert float((other[key] - block).abs().max()) < 1e-11 * float(block.abs().max()), key


def test_lattice_charge_block_is_the_existing_ewald_plus_the_gram_term(lattice_case):
    """Two independent constructions; they must differ by (pi K_E / V) s_ij^2 exactly."""
    periodic_pair_blocks = require("energy", "_periodic_pair_blocks")
    ElectrostaticParams = require("solver", "ElectrostaticParams")
    case = lattice_case
    n = case["pos"].shape[0]
    params = ElectrostaticParams(
        chi=torch.zeros(n), eta=torch.full((n,), 30.0), sigma=case["sigma"], ell=case["ell"],
        kappa=torch.ones(0), A_mu=case["A_mu"],
    )
    config = _config(ewald_real_cutoff=14.0, ewald_accuracy=1e-14, ewald_alpha=0.30)
    j_existing, _, _ = periodic_pair_blocks(
        case["pos"], case["cell"], params, torch.ones(n), config, pbc=(True, True, True)
    )
    volume = torch.det(case["cell"]).abs()
    gram = (math.pi * K_E / volume) * (case["sigma"][:, None] ** 2 + case["sigma"][None] ** 2)
    assert float((_lattice(case, 0.30)["j"] - j_existing - gram).abs().max()) < 1e-12


def test_lattice_hessian_is_positive_definite_at_the_floors(lattice_case):
    kernel_self = require("kernels", "kernel_self")
    case = lattice_case
    b = _lattice(case, 0.30)
    n = case["pos"].shape[0]
    J = b["j"] + torch.diag(kernel_self(case["sigma"]))
    T01 = b["t01"].reshape(n, 3 * n)
    T2 = b["t2"].permute(0, 2, 1, 3).reshape(3 * n, 3 * n) + torch.block_diag(*case["A_mu"])
    T02 = b["t02"].reshape(n, 5 * n)
    T12 = b["t12"].permute(0, 2, 1, 3).reshape(3 * n, 5 * n)
    T22 = b["t22"].permute(0, 2, 1, 3).reshape(5 * n, 5 * n) + torch.block_diag(*case["A_theta"])
    K = torch.cat([torch.cat([J, T01, T02], 1), torch.cat([T01.T, T2, T12], 1),
                   torch.cat([T02.T, T12.T, T22], 1)], 0)
    assert float((K - K.T).abs().max()) < 1e-12
    assert float(torch.linalg.eigvalsh(0.5 * (K + K.T)).min()) > 0.0


def test_neutral_cluster_tends_to_open_boundaries_minus_the_tinfoil_term():
    """E_per(L) - [E_open - 2 pi K_E |M|^2 / 3V] shrinks like L^-5 (x25 then x31 measured)."""
    gaussian_lattice_blocks = require("lattice", "gaussian_lattice_blocks")
    EwaldParams = require("ewald", "EwaldParams")
    width = require("kernels", "gaussian_multipole_width")
    ElectrostaticParams = require("solver", "ElectrostaticParams")
    direct, direct_quadrupole = require("energy", "_direct_pair_blocks", "_direct_quadrupole_blocks")
    g = torch.Generator().manual_seed(4)
    n = 4
    cluster = torch.randn(n, 3, generator=g) * 1.2
    sigma, ell = 0.8 + 0.3 * torch.rand(n, generator=g), 0.6 + 0.5 * torch.rand(n, generator=g)
    A_mu, A_th = _spd(n, 3, 3.0, g), _spd(n, 5, 2.0, g)
    params = ElectrostaticParams(chi=torch.zeros(n), eta=torch.full((n,), 40.0), sigma=sigma,
                                 ell=ell, kappa=torch.ones(0), A_mu=A_mu, A_Theta=A_th)
    s_mu, s_th = width(A_mu, ell, 1, 0.5), width(A_th, ell, 2, 0.5)
    q = torch.randn(n, generator=g)
    q = q - q.mean()
    mu, th = torch.randn(n, 3, generator=g), torch.randn(n, 5, generator=g)

    def energy(b):
        return (0.5 * q @ b["j"] @ q + torch.einsum("i,ija,ja->", q, b["t01"], mu)
                + 0.5 * torch.einsum("ia,ijab,jb->", mu, b["t2"], mu)
                + torch.einsum("i,ijm,jm->", q, b["t02"], th)
                + torch.einsum("ia,ijam,jm->", mu, b["t12"], th)
                + 0.5 * torch.einsum("im,ijmn,jn->", th, b["t22"], th))

    j, t01, t2 = direct(cluster, params, torch.ones(n), 0.39, kernel="gaussian", gaussian_a=0.5)
    t02, t12, t22 = direct_quadrupole(cluster, params, 0.5)
    e_open = energy(dict(j=j, t01=t01, t2=t2, t02=t02, t12=t12, t22=t22))
    M = q @ cluster + (ell[:, None] * mu).sum(0)
    residuals = []
    for L in (12.0, 24.0):
        cell = L * torch.eye(3)
        blocks = gaussian_lattice_blocks(
            cluster + L / 2, cell, sigma, s_mu, ell,
            EwaldParams.auto_tune(cell, 12.0, accuracy=1e-13), sigma_theta=s_th,
        )
        residuals.append(float(energy(blocks) - (e_open - 2 * math.pi * K_E / (3 * L ** 3) * (M @ M))))
    assert abs(residuals[0]) / abs(residuals[1]) > 20.0
    assert abs(residuals[1]) < 2e-3


def test_too_wide_gaussian_for_the_image_box_raises():
    gaussian_lattice_blocks, LatticeWidthError = require("lattice", "gaussian_lattice_blocks",
                                                         "LatticeWidthError")
    EwaldParams = require("ewald", "EwaldParams")
    cell = 6.0 * torch.eye(3)
    ewald = EwaldParams.auto_tune(cell, 6.0, accuracy=1e-8)
    with pytest.raises(LatticeWidthError, match="ewald_real_cutoff"):
        gaussian_lattice_blocks(torch.zeros(1, 3), cell, torch.tensor([1.0]),
                                torch.tensor([4.0]), torch.ones(1), ewald)


def test_periodic_gaussian_energy_is_extensive_with_quadrupoles():
    heads = _heads(quadrupole_blocks=("A_Theta", "chi_theta"), allow_truncated_a_theta=True, r_max=3.2)
    torch.manual_seed(0)
    h = torch.randn(2, heads.irreps_in.dim)
    a = 5.0
    cell = torch.tensor([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    pos = torch.tensor([[0.0, 0, 0], [a / 2, 0, 0]])
    config = _config(multipole_kernel="gaussian", ewald_real_cutoff=12.0)
    primitive = _state(heads, h, pos, cell, 3.1, config, quadrupoles=True)
    sp, sc, sh = _supercell(pos, cell, h, 2)
    supercell = _state(heads, sh, sp, sc, 3.1, config, quadrupoles=True)
    assert abs(float(supercell.energy) / 8 - float(primitive.energy)) < 1e-12
    assert abs(float(primitive.energy - primitive.solve_result.energy)) < 1e-14


@pytest.mark.parametrize("kernel", ["thole", "gaussian"])
def test_fixed_k_energy_is_the_same_in_a_cell_and_its_supercell(kernel):
    """SPEC 14.1: the seed steps precondition with A_mu / A_Theta, not the diagonal
    blocks of K, which under PBC carry a cell-dependent self-image sum (that made
    the k = 6 energy differ by 3.8e-7 eV per cell after one step). A displaced,
    non-centrosymmetric cell, so the dipole seed gradient is not zero."""
    heads = _heads(r_max=3.2)
    torch.manual_seed(5)
    h = torch.randn(2, heads.irreps_in.dim)
    a = 5.0
    cell = torch.tensor([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    pos = torch.tensor([[0.10, -0.05, 0.02], [a / 2 - 0.12, 0.07, 0.03]])
    config = _config(multipole_kernel=kernel, ewald_real_cutoff=12.0)
    primitive = _state(heads, h, pos, cell, 3.1, config)
    assert float(primitive.solve_result.mu.abs().max()) > 1e-6     # the dipole step acts
    sp, sc, sh = _supercell(pos, cell, h, 2)
    supercell = _state(heads, sh, sp, sc, 3.1, config)
    assert abs(float(supercell.energy) / 8 - float(primitive.energy)) < 1e-14


# ===========================================================================
# SPEC 13.1: periodic dispersion
# ===========================================================================
def _dispersion_per_cell(M, gamma):
    gaussian_lattice_blocks = require("lattice", "gaussian_lattice_blocks")
    EwaldParams = require("ewald", "EwaldParams")
    width = require("kernels", "gaussian_multipole_width")
    dispersion_energy = require("dispersion", "dispersion_energy")
    a = 5.0
    cell = torch.tensor([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    pos = torch.tensor([[0.0, 0, 0], [a / 2, 0, 0]])
    A_mu = torch.stack([torch.eye(3) * K_E / 1.5, torch.eye(3) * K_E / 3.0])
    ell, electrons, sigma = torch.ones(2), torch.full((2,), 2.0), torch.tensor([1.0, 1.3])
    s_mu = width(A_mu, ell, 1, 0.5)
    shifts = torch.tensor(list(itertools.product(range(M), repeat=3)), dtype=torch.float64)
    P = (pos[None] + (shifts @ cell)[:, None]).reshape(-1, 3)
    C, copies, n = M * cell, M ** 3, 2 * M ** 3
    b = gaussian_lattice_blocks(P, C, sigma.repeat(copies), s_mu.repeat(copies), ell.repeat(copies),
                                EwaldParams.auto_tune(C, 10.0, accuracy=1e-8), include_charge=False)
    T = b["t2"].permute(0, 2, 1, 3).reshape(3 * n, 3 * n)
    energy = dispersion_energy(T, A_mu.repeat(copies, 1, 1), ell.repeat(copies),
                               electrons.repeat(copies),
                               macroscopic_volume=torch.det(C).abs() if gamma else None)
    return float(energy) / copies


def test_gamma_term_makes_periodic_dispersion_converge():
    """Measured: corrected -0.4944, -0.49353 at 3^3, 4^3; uncorrected -0.631, -0.551."""
    corrected = [_dispersion_per_cell(M, True) for M in (3, 4)]
    tinfoil = [_dispersion_per_cell(M, False) for M in (3, 4)]
    assert abs(corrected[0] / corrected[1] - 1.0) < 5e-3
    assert abs(tinfoil[1] / corrected[1] - 1.0) > 0.05
    assert abs(tinfoil[0] - tinfoil[1]) > 20 * abs(corrected[0] - corrected[1])


def test_gamma_term_is_rotation_invariant():
    """The sphere rule is exact to degree 23; a rotated crystal's energy must not move."""
    gaussian_lattice_blocks = require("lattice", "gaussian_lattice_blocks")
    EwaldParams = require("ewald", "EwaldParams")
    width = require("kernels", "gaussian_multipole_width")
    dispersion_energy = require("dispersion", "dispersion_energy")
    o3 = pytest.importorskip("e3nn.o3")
    a = 4.4
    cell = torch.tensor([[a, 0.2, 0.0], [0.1, a * 0.9, 0.3], [0.0, 0.4, a * 1.1]])
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.9, 2.1, 1.7]])
    A_mu = torch.stack([torch.diag(torch.tensor([7.0, 9.0, 11.0])), torch.eye(3) * 6.0])
    ell, electrons, sigma = torch.ones(2), torch.tensor([2.0, 1.5]), torch.tensor([1.0, 1.2])
    energies = []
    for rotation in (torch.eye(3), o3.rand_matrix()):
        A_rot = rotation @ A_mu @ rotation.T
        c, p = cell @ rotation.T, pos @ rotation.T
        b = gaussian_lattice_blocks(p, c, sigma, width(A_rot, ell, 1, 0.5), ell,
                                    EwaldParams.auto_tune(c, 10.0, accuracy=1e-10), include_charge=False)
        T = b["t2"].permute(0, 2, 1, 3).reshape(6, 6)
        energies.append(float(dispersion_energy(T, A_rot, ell, electrons,
                                                macroscopic_volume=torch.det(c).abs())))
    assert abs(energies[1] / energies[0] - 1.0) < 1e-10


# ===========================================================================
# SPEC 15: bond-current polarization and the electric enthalpy
# ===========================================================================
def test_self_image_bond_has_a_zero_column_and_one_orientation():
    build_bond_topology = require("topology", "build_bond_topology")
    topology = build_bond_topology(torch.tensor([[0, 0], [0, 0]]), 1,
                                   edge_shift=torch.tensor([[1, 0, 0], [-1, 0, 0]]),
                                   self_image_bonds=True)
    assert topology.incidence.shape == (1, 1)
    assert float(topology.incidence.abs().sum()) == 0.0
    assert topology.bond_shift.tolist() == [[1, 0, 0]]
    assert topology.edge_sign.tolist() == [1.0, -1.0]


@pytest.fixture(scope="module")
def chain():
    heads = _heads(r_max=3.0)
    torch.manual_seed(0)
    h = torch.randn(1, heads.irreps_in.dim)
    return heads, h, 2.4


def test_monatomic_chain_polarizes_through_its_self_image_bond(chain):
    """q = 0 by symmetry; alpha_ct = sum_b (f_b d_b)^2 / kappa_b exactly (0.0516027596)."""
    contributions = require("observables", "polarizability_contributions")
    FixedKDegenerateWarning = require("observables", "FixedKDegenerateWarning")
    heads, h, a = chain
    cell = torch.diag(torch.tensor([a, 30.0, 30.0]))
    state = _state(heads, h, torch.zeros(1, 3), cell, 2.6, _config(ewald_real_cutoff=10.0))
    assembly = state.assembly
    assert assembly.n_bonds == 1 and float(assembly.incidence.abs().sum()) == 0.0
    exact = float((assembly.bond_dipole[:, 0] ** 2 / assembly.params.kappa).sum())
    charge, _ = contributions(state.solve_result, assembly, route="minimiser")
    assert abs(float(charge[0, 0]) - exact) < 1e-12 * exact
    # Default (converged) route: no degenerate point exists, no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", FixedKDegenerateWarning)
        charge_default, _ = contributions(state.solve_result, assembly)
    assert abs(float(charge_default[0, 0]) - exact) < 1e-9 * exact
    # Legacy fixed-k route: the zero residual is detected and replaced.
    legacy = _state(heads, h, torch.zeros(1, 3), cell, 2.6,
                    _config(ewald_real_cutoff=10.0, solve_mode="fixed_k"))
    with pytest.warns(FixedKDegenerateWarning):
        charge_fixed_k, _ = contributions(legacy.solve_result, legacy.assembly)
    assert abs(float(charge_fixed_k[0, 0]) - exact) < 1e-12 * exact


def test_open_chain_polarizability_tends_to_the_periodic_one(chain):
    """alpha/N tends to the cell value with a 1/N deficit.

    Measured with the default shell envelope (SPEC 17.4, onset 0.75): 0.327
    (N=10), 0.754 (N=40), 0.936 (N=160); deficit ratio 2.73 then 3.86 -> 4.
    The nearest-neighbour bond (2.4 A of a 3.0 A cutoff) now carries nearly
    its full transfer, so the charge-transfer screening length is long and
    N = 10 is not yet asymptotic; the 1/N law -- the physics under test -- is
    asserted where it holds.  (Onset 0.5: 0.738 / 0.929 / 0.982; the legacy
    cosine, which suppressed the transfer ~10x: 0.904 / 0.975 / 0.993.)"""
    polarizability = require("observables", "polarizability")
    electrostatic_state = require("energy", "electrostatic_state")
    heads, h, a = chain
    cell = torch.diag(torch.tensor([a, 30.0, 30.0]))
    periodic = _state(heads, h, torch.zeros(1, 3), cell, 2.6, _config(ewald_real_cutoff=10.0))
    alpha_cell = float(polarizability(periodic.solve_result, periodic.assembly, route="minimiser")[0, 0])
    ratios = []
    for N in (10, 40, 160):
        pos = torch.zeros(N, 3)
        pos[:, 0] = torch.arange(N) * a
        pairs = [(i, j) for i in range(N) for j in (i - 1, i + 1) if 0 <= j < N]
        edge_index = torch.tensor([[i for i, _ in pairs], [j for _, j in pairs]])
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        state = electrostatic_state(h.repeat(N, 1), pos, edge_index, edge_vec,
                                    edge_vec.norm(dim=-1), heads, _config())
        alpha = float(polarizability(state.solve_result, state.assembly, route="minimiser")[0, 0])
        ratios.append(alpha / N / alpha_cell)
    assert ratios[0] < ratios[1] < ratios[2] < 1.0
    assert ratios[2] > 0.9
    assert 3.0 < (1 - ratios[1]) / (1 - ratios[2]) < 4.5


@pytest.fixture(scope="module")
def rocksalt():
    heads = _heads(r_max=3.0)
    torch.manual_seed(1)
    h = torch.randn(2, heads.irreps_in.dim)
    L = 4.2
    cell = torch.tensor([[0, L / 2, L / 2], [L / 2, 0, L / 2], [L / 2, L / 2, 0]])
    pos = torch.tensor([[0.05, 0.02, -0.03], [L / 2, 0.1, 0.04]])
    return heads, h, cell, pos


def test_bond_polarization_is_invariant_under_wrapping(rocksalt):
    bond_polarization, polarization = require("observables", "bond_polarization", "polarization")
    heads, h, cell, pos = rocksalt
    config = _config(ewald_real_cutoff=10.0)
    first = _state(heads, h, pos, cell, 3.1, config)
    wrapped_pos = pos.clone()
    wrapped_pos[1] = wrapped_pos[1] + cell[0] - cell[2]
    second = _state(heads, h, wrapped_pos, cell, 3.1, config)
    assert abs(float(second.energy - first.energy)) < 1e-15
    p1 = bond_polarization(first.solve_result, first.assembly)
    p2 = bond_polarization(second.solve_result, second.assembly)
    assert float((p2 - p1).abs().max()) < 1e-15
    naive = [polarization(s.solve_result.q, s.assembly.pos, s.solve_result.mu) for s in (first, second)]
    assert float((naive[1] - naive[0]).norm()) > 1e-3


def test_born_charges_from_the_enthalpy_equal_the_minimisers(rocksalt):
    born_from_energy, born_minimiser, asr = require(
        "observables", "born_charges_from_energy", "born_charges_minimiser",
        "acoustic_sum_rule_residual",
    )
    heads, h, cell, pos = rocksalt
    config = _config(ewald_real_cutoff=10.0, n_cg_iter=200)
    z_enthalpy = born_from_energy(_state(heads, h, pos.clone().requires_grad_(True), cell, 3.1, config).assembly)
    z_minimiser = born_minimiser(_state(heads, h, pos.clone().requires_grad_(True), cell, 3.1, config).assembly)
    assert torch.isfinite(z_minimiser).all()
    assert float((z_enthalpy - z_minimiser).abs().max()) < 1e-12
    assert float(asr(z_minimiser, 0.0).abs().max()) < 1e-14


def test_enthalpy_is_exactly_quadratic_in_the_field(rocksalt):
    bond_polarization, polarizability = require("observables", "bond_polarization", "polarizability")
    heads, h, cell, pos = rocksalt
    config = _config(ewald_real_cutoff=10.0, n_cg_iter=200)
    zero = _state(heads, h, pos, cell, 3.1, config)
    field = torch.tensor([0.02, -0.01, 0.03])
    fielded = _state(heads, h, pos, cell, 3.1, config, field=field)
    alpha = polarizability(zero.solve_result, zero.assembly, route="minimiser")
    P0 = bond_polarization(zero.solve_result, zero.assembly)
    predicted = zero.energy - field @ P0 - 0.5 * field @ alpha @ field
    assert abs(float(fielded.energy - predicted)) < 1e-15


def test_dielectric_tensor_is_intensive_and_needs_self_image_bonds():
    """CsCl: primitive (6 of 14 bonds self-image) == 2x2x2 supercell to 1e-10;
    dropping the self-image bonds, as before SPEC 15, cost 12.8 % of chi."""
    dielectric_tensor = require("observables", "dielectric_tensor")
    energy_module = pytest.importorskip("trace_electrostatics.energy")
    heads = _heads(r_max=5.0)
    torch.manual_seed(2)
    h = torch.randn(2, heads.irreps_in.dim)
    a = 4.0
    cell, pos = a * torch.eye(3), torch.tensor([[0.0, 0, 0], [a / 2, a / 2, a / 2]])
    config = _config(ewald_real_cutoff=12.0)

    def eps(p, c, hh):
        state = _state(heads, hh, p, c, 4.5, config)
        return torch.diagonal(dielectric_tensor(state.solve_result, state.assembly, route="minimiser"))

    primitive = eps(pos, cell, h)
    supercell = eps(*_supercell(pos, cell, h, 2))
    assert float((primitive - supercell).abs().max()) < 1e-10
    original = energy_module.build_bond_topology
    energy_module.build_bond_topology = lambda *x, **k: original(*x, **{**k, "self_image_bonds": False})
    try:
        dropped = eps(pos, cell, h)
    finally:
        energy_module.build_bond_topology = original
    assert float(((dropped - 1) / (supercell - 1) - 1).abs().max()) > 0.05


@pytest.mark.parametrize("displace", [0.0, 0.15])
def test_bond_compliance_identity_for_centrosymmetric_crystals(displace):
    """docs/ELECTROSTATICS_THEORY_2026-10-08.md, Theorem 1.

    If every atom is an inversion centre of the crystal and of its bond star, a uniform
    field drives bond currents t_b = p_b.E / kappa_b that leave every atomic charge at
    zero, so the Coulomb kernel drops out of the response exactly:

        eps_inf = I + 4 pi K_E / Omega (sum_b p_b p_b^T / kappa_b + alpha_dip),

    alpha_dip the response of the dipoles alone (t = 0).  CsCl: both atoms are inversion
    centres.  The control displaces Cl by 0.15 A, breaking the inversion: the identity
    must then fail by far more than the round-off tolerance.
    """
    dielectric_tensor = require("observables", "dielectric_tensor")
    heads = _heads(r_max=5.0)
    torch.manual_seed(2)
    h = torch.randn(2, heads.irreps_in.dim)
    a = 4.0
    cell = a * torch.eye(3)
    pos = torch.tensor([[0.0, 0.0, 0.0], [a / 2, a / 2, a / 2 + displace]])
    state = _state(heads, h, pos, cell, 4.5, _config(ewald_real_cutoff=12.0, solver="dense"))
    asm = state.assembly
    eps = dielectric_tensor(state.solve_result, asm, route="minimiser").detach()
    p, kappa = asm.bond_dipole.detach(), asm.params.kappa.detach()
    bonds = (p / kappa.unsqueeze(1)).T @ p
    nb, n = asm.n_bonds, asm.n_atoms
    H = asm.dense_hessian().detach()
    mu = torch.arange(nb + n, H.shape[0])
    D = asm.dipole_derivative().detach()[mu]
    dipoles = D.T @ torch.linalg.solve(H[mu][:, mu], D)
    predicted = torch.eye(3) + 4.0 * math.pi * K_E / float(torch.linalg.det(cell)) * (bonds + dipoles)
    gap = float((eps - predicted).abs().max()) / float(eps.abs().max())
    free = torch.cat((torch.arange(nb), mu))
    tol = 64.0 * float(torch.linalg.cond(H[free][:, free])) * torch.finfo(torch.float64).eps
    if displace == 0.0:
        assert gap <= tol
    else:
        assert gap > 1e3 * tol


def test_static_dielectric_and_the_nonanalytic_term():
    static, nonanalytic = require("observables", "static_dielectric_tensor", "nonanalytic_force_constants")
    g = torch.Generator().manual_seed(3)
    n = 3
    born = torch.randn(n, 3, 3, generator=g)
    born = born - born.mean(0)                              # acoustic sum rule
    phi = _spd(3 * n, 3 * n, 0.0, g, scale=1.0)[0]          # a positive semidefinite matrix
    translations = torch.eye(3).repeat(n, 1) / math.sqrt(n)
    projector = torch.eye(3 * n) - translations @ translations.T
    phi = projector @ (phi + torch.eye(3 * n)) @ projector
    eps_inf = torch.diag(torch.tensor([3.0, 4.0, 5.0]))
    eps_0 = static(eps_inf, born, phi, torch.tensor(50.0))
    assert torch.allclose(eps_0, eps_0.T, atol=1e-12)
    assert float(torch.linalg.eigvalsh(eps_0 - eps_inf).min()) >= -1e-12
    q = torch.tensor([0.0, 0.0, 2.0])
    na = nonanalytic(born, eps_inf, torch.tensor(50.0), q)
    zq = born[:, 2, :].reshape(-1)
    assert torch.allclose(na, (4 * math.pi * K_E / 50.0) * torch.outer(zq, zq) / 5.0)


# ===========================================================================
# SPEC 15.2: the fixed-k degenerate point
# ===========================================================================
def test_fixed_k_degenerate_point_falls_back_or_raises():
    """LEGACY: a homonuclear diatomic, where the fixed-k alpha used to be EXACTLY 0."""
    electrostatic_state = require("energy", "electrostatic_state")
    polarizability, FixedKDegenerateWarning, FixedKDegenerateError, born_from_energy = require(
        "observables", "polarizability", "FixedKDegenerateWarning", "FixedKDegenerateError",
        "born_charges_from_energy",
    )
    heads = _heads()
    torch.manual_seed(0)
    h = torch.randn(1, heads.irreps_in.dim).repeat(2, 1)
    pos = torch.tensor([[0.0, 0, 0], [1.1, 0, 0]], requires_grad=True)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    state = electrostatic_state(h, pos, edge_index, edge_vec, edge_vec.norm(dim=-1), heads,
                                _config(solve_mode="fixed_k"))
    exact = polarizability(state.solve_result, state.assembly, route="minimiser")
    with pytest.warns(FixedKDegenerateWarning):
        fixed_k = polarizability(state.solve_result, state.assembly)
    assert torch.allclose(fixed_k, exact) and float(exact[0, 0]) > 0.1
    with pytest.raises(FixedKDegenerateError):
        born_from_energy(state.assembly)


# ===========================================================================
# SPEC 16: the matrix-free lattice route
# ===========================================================================
def _random_multipole_case(seed=2, n=6):
    width = require("kernels", "gaussian_multipole_width")
    g = torch.Generator().manual_seed(seed)
    cell = torch.tensor([[7.0, 0.3, 0.0], [0.2, 6.5, 0.4], [0.0, 0.5, 7.5]])
    pos = torch.rand(n, 3, generator=g) @ cell
    sigma = 0.8 + 0.4 * torch.rand(n, generator=g)
    ell = 0.6 + 0.6 * torch.rand(n, generator=g)
    A_mu, A_th = _spd(n, 3, 3.0, g), _spd(n, 5, 2.0, g)
    return dict(pos=pos, cell=cell, sigma=sigma, ell=ell, A_mu=A_mu,
                s_mu=width(A_mu, ell, 1, 0.5), s_th=width(A_th, ell, 2, 0.5), g=g)


def test_matrix_free_product_equals_the_dense_lattice_operator():
    """Measured 9e-14 (phi), 3e-13 (field), 8e-13 (quadrupole) relative."""
    gaussian_lattice_blocks = require("lattice", "gaussian_lattice_blocks")
    lattice_pairs, LatticeOperator = require("lattice_operator", "lattice_pairs", "LatticeOperator")
    EwaldParams = require("ewald", "EwaldParams")
    c = _random_multipole_case()
    n = c["pos"].shape[0]
    ewald = EwaldParams.auto_tune(c["cell"], 14.0, accuracy=1e-13)
    dense = gaussian_lattice_blocks(c["pos"], c["cell"], c["sigma"], c["s_mu"], c["ell"], ewald,
                                    sigma_theta=c["s_th"])
    op = LatticeOperator(c["pos"], c["cell"], ewald, lattice_pairs(c["pos"], c["cell"], 14.0),
                         sigma=c["sigma"], kernel="gaussian", sigma_mu=c["s_mu"], sigma_theta=c["s_th"])
    q, mh, th = torch.randn(n, generator=c["g"]), torch.randn(n, 3, generator=c["g"]), torch.randn(n, 5, generator=c["g"])
    phi, field, grad = op.apply_reduced(q, mh, c["ell"], th)
    phi_d = dense["j"] @ q + torch.einsum("ija,ja->i", dense["t01"], mh) + torch.einsum("ijm,jm->i", dense["t02"], th)
    field_d = (torch.einsum("jia,j->ia", dense["t01"], q) + torch.einsum("ijab,jb->ia", dense["t2"], mh)
               + torch.einsum("ijam,jm->ia", dense["t12"], th))
    grad_d = (torch.einsum("jim,j->im", dense["t02"], q) + torch.einsum("jiam,ja->im", dense["t12"], mh)
              + torch.einsum("ijmn,jn->im", dense["t22"], th))
    for got, want in ((phi, phi_d), (field, field_d), (grad, grad_d)):
        assert float((got - want).abs().max()) < 1e-11 * float(want.abs().max())


def test_linked_cell_pairs_equal_the_image_box_search():
    lattice_pairs = require("lattice_operator", "lattice_pairs")
    g = torch.Generator().manual_seed(0)
    cell = torch.tensor([[14.0, 0.5, 0.3], [0.2, 15.0, 0.4], [0.1, 0.6, 16.0]])
    pos = (torch.rand(300, 3, generator=g) * 1.3 - 0.15) @ cell      # atoms outside the cell too
    fast = lattice_pairs(pos, cell, 4.5)             # 3 bins per axis: the linked-cell route
    slow = _image_box_pairs(pos, cell, 4.5)          # independent brute force

    def key(p):
        return set(zip(p[0].tolist(), p[1].tolist(), map(tuple, p[2].round().long().tolist())))

    assert key((fast.first, fast.second, fast.shift)) == key(slow)


def _image_box_pairs(pos, cell, r):
    shells = require("ewald", "_image_shell_counts")(cell, r)
    offsets = torch.cartesian_prod(*[torch.arange(-s, s + 1, dtype=torch.float64) for s in shells])
    n = pos.shape[0]
    delta = pos.unsqueeze(1) - pos.unsqueeze(0)
    minimum = torch.round(delta @ torch.linalg.inv(cell))
    shift = offsets.view(1, 1, -1, 3) - minimum.unsqueeze(2)
    vector = delta.unsqueeze(2) + shift @ cell
    inside = (vector * vector).sum(-1) < r * r
    same = (torch.arange(n).view(-1, 1, 1) == torch.arange(n).view(1, -1, 1)) & (shift.abs().sum(-1) == 0)
    index = (inside & ~same).nonzero()
    return index[:, 0], index[:, 1], shift[index[:, 0], index[:, 1], index[:, 2]]


def test_sparse_incidence_equals_the_dense_one():
    build_bond_topology = require("topology", "build_bond_topology")
    edge_index = torch.tensor([[0, 1, 0, 0, 1, 2], [1, 0, 0, 0, 2, 1]])
    edge_shift = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    dense = build_bond_topology(edge_index, 3, edge_shift=edge_shift, self_image_bonds=True)
    sparse = build_bond_topology(edge_index, 3, edge_shift=edge_shift, self_image_bonds=True,
                                 sparse_incidence=True)
    assert sparse.incidence.is_sparse
    assert torch.equal(sparse.incidence.to_dense(), dense.incidence)


@pytest.mark.parametrize("kernel, quadrupoles", [("thole", False), ("gaussian", False), ("gaussian", True)])
def test_matrix_free_route_equals_the_dense_route(kernel, quadrupoles):
    """Energy, forces, charges, Born charges and eps_inf, both routes, with a field."""
    born_from_energy, dielectric_tensor = require("observables", "born_charges_from_energy", "dielectric_tensor")
    extra = {"quadrupole_blocks": ("A_Theta", "chi_theta"), "allow_truncated_a_theta": True} if quadrupoles else {}
    heads = _heads(r_max=3.2, **extra)
    torch.manual_seed(5)
    h = torch.randn(2, heads.irreps_in.dim)
    a = 5.0
    cell = torch.tensor([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    base = torch.tensor([[0.10, -0.05, 0.02], [a / 2 - 0.12, 0.07, 0.03]])
    sp, sc, sh = _supercell(base, cell, h, 2)
    field = torch.tensor([0.02, -0.01, 0.015])
    results = {}
    for solver in ("dense", "matrix_free"):
        config = _config(multipole_kernel=kernel, ewald_real_cutoff=12.0, ewald_accuracy=1e-12, solver=solver)
        p = sp.clone().requires_grad_(True)
        state = _state(heads, sh, p, sc, 3.1, config, field=field, quadrupoles=quadrupoles)
        (force,) = torch.autograd.grad(state.energy, p)
        born = born_from_energy(
            _state(heads, sh, sp.clone().requires_grad_(True), sc, 3.1, config, quadrupoles=quadrupoles).assembly
        )
        plain = _state(heads, sh, sp, sc, 3.1, config, quadrupoles=quadrupoles)
        eps = dielectric_tensor(plain.solve_result, plain.assembly)
        results[solver] = (state.energy.detach(), force, state.solve_result.q.detach(), born, eps)
    dense, free = results["dense"], results["matrix_free"]
    assert abs(float(dense[0] - free[0])) < 1e-11
    for a_, b_ in zip(dense[1:], free[1:]):
        assert float((a_ - b_).abs().max()) < 1e-9


def test_matrix_free_refuses_what_needs_the_dense_operator():
    born_minimiser = require("observables", "born_charges_minimiser")
    heads = _heads(r_max=3.2)
    torch.manual_seed(5)
    h = torch.randn(2, heads.irreps_in.dim)
    a = 5.0
    cell = torch.tensor([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    pos = torch.tensor([[0.1, 0.0, 0.0], [a / 2, 0.0, 0.0]], requires_grad=True)
    state = _state(heads, h, pos, cell, 3.1, _config(solver="matrix_free"))
    with pytest.raises(NotImplementedError, match="matrix_free"):
        state.assembly.dense_hessian()
    with pytest.raises(NotImplementedError):
        born_minimiser(state.assembly)
    # Open boundaries are no longer refused: SPEC 18 routes them through
    # OpenOperator (tests/electrostatics/test_open_matrix_free.py). The same
    # dense-operator refusals apply there.
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    open_state = require("energy", "electrostatic_state")(
        h, pos, edge_index, edge_vec, edge_vec.norm(dim=-1), heads, _config(solver="matrix_free")
    )
    assert open_state.assembly.lattice_operator is not None
    with pytest.raises(NotImplementedError, match="matrix_free"):
        open_state.assembly.dense_hessian()
