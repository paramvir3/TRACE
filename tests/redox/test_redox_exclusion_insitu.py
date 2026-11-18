"""In-situ removal energies for anion densities (``RedoxConfig(exclusion_anions='in_situ')``, trace_redox SPEC 18.6;
theory note 2026-10-16 §11).

An anion-frame unit's frontier densities (its frame and its hole states) decay with the model's own in-situ removal
energy, the configuration-energy difference for one electron leaving it with everything else at frames (no Pi):

    I_env,u(n) = e_u(n+1) - e_u(n) + (b + W f)_u + (2d + 1) W_uu / 2,     d = n - f_u
    I~ = smooth clamp of I_env to [1, 40] eV,   kappa = sqrt(2 I~),   beta = 0

replacing the learned I_eff and admitting anion carriers (holes: O-, Cl0).  Cation- and neutral-frame units keep the
gas ladder.  Measured on step 9's weights: surface O2- 9.8 eV, bulk 11.2-11.5 eV (learned I_eff 9.29); O- hole 14.2.
"""
import math

import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

HARTREE = 27.211386245988
BOHR = 0.529177210903
K_E = 14.399645


def test_the_in_situ_energy_is_the_configuration_energy_difference():
    from trace_redox.diabatic import configuration_energies
    from trace_redox.exclusion import in_situ_removal
    g = torch.Generator().manual_seed(3)
    U = 5
    states = torch.arange(-2, 3, dtype=torch.float64)
    e_hat = torch.randn(U, 5, generator=g, dtype=torch.float64) * 3.0
    b = torch.randn(U, generator=g, dtype=torch.float64)
    A = torch.randn(U, U, generator=g, dtype=torch.float64)
    W = A @ A.T / U + torch.eye(U, dtype=torch.float64)
    frames = torch.tensor([-2, 1, 0, -1, 2])
    I = in_situ_removal(e_hat, states, b, W, frames)
    for u in range(U):
        for n in range(-2, 2):
            lo = frames.clone()
            lo[u] = n
            hi = lo.clone()
            hi[u] = n + 1
            exact = configuration_energies(torch.stack([hi, lo]), e_hat, states, b, W)
            assert abs(float(I[u, n + 2]) - float(exact[0] - exact[1])) <= 1e-12 * (1 + abs(float(exact[0])))


def test_an_isolated_anion_has_its_gas_removal_energy():
    """Cl- alone, no learned shift, b = 0, W = S0: I_env(-1) = EA(Cl) exactly."""
    from trace_redox.constants import ATOMIC_ELECTRON_AFFINITY_EV as EA
    from trace_redox.diabatic import formal_ladder
    from trace_redox.exclusion import in_situ_removal
    z = torch.tensor([17])
    states, E_Z, _, _ = formal_ladder(z, 3)
    S0 = torch.tensor([7.3], dtype=torch.float64)
    e_hat = E_Z - 0.5 * S0.unsqueeze(1) * states * states
    I = in_situ_removal(e_hat, states, torch.zeros(1, dtype=torch.float64), S0.reshape(1, 1), torch.tensor([-1]))
    k = int((states == -1).nonzero())
    assert abs(float(I[0, k]) - float(EA[17])) <= 1e-12


def test_the_madelung_field_binds_bulk_anions_more_than_surface_ones():
    """A 3x3x3 rocksalt NaCl cube with bare Coulomb W (no bound sector): the Cl- removal energy grows with the
    Madelung potential, so the centre Cl is the most bound and the corners least (corner < edge < face < centre)."""
    from trace_redox.diabatic import formal_ladder
    from trace_redox.exclusion import in_situ_removal
    a = 5.64
    pos, z = [], []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pos.append([i * a / 2, j * a / 2, k * a / 2])
                z.append(17 if (i + j + k) % 2 == 0 else 11)
    pos = torch.tensor(pos, dtype=torch.float64)
    z = torch.tensor(z)
    states, E_Z, _, _ = formal_ladder(z, 3)
    r = torch.cdist(pos, pos)
    S0 = torch.full((len(z),), 6.0, dtype=torch.float64)
    W = torch.where(r > 0, K_E / r.clamp_min(1e-9), torch.diag(S0))
    e_hat = E_Z - 0.5 * S0.unsqueeze(1) * states * states
    frames = torch.where(z == 17, torch.tensor(-1), torch.tensor(1))
    I = in_situ_removal(e_hat, states, torch.zeros(len(z), dtype=torch.float64), W, frames)
    col = int((states == -1).nonzero())
    centre = pos.mean(0)
    cl = (z == 17).nonzero().flatten()
    zero_dims = (pos[cl] - centre).abs().lt(1e-9).sum(1)     # 3 = centre, 2 = face, 1 = edge, 0 = corner
    by_kind = {k: float(I[cl[zero_dims == k], col].mean()) for k in (0, 1, 2, 3) if bool((zero_dims == k).any())}
    kinds = sorted(by_kind)
    assert all(by_kind[a_] < by_kind[b_] for a_, b_ in zip(kinds, kinds[1:])), by_kind


def test_the_three_dimensional_anion_table_matches_the_quadrature():
    from trace_redox.exclusion import OverlapTable, density_overlap
    table = OverlapTable(r_max=6.0)
    g = torch.Generator().manual_seed(4)
    R = 0.8 + 4.2 * torch.rand(300, generator=g, dtype=torch.float64)
    Iu = 1.0 + 39.0 * torch.rand(300, generator=g, dtype=torch.float64)
    Iv = 1.0 + 39.0 * torch.rand(300, generator=g, dtype=torch.float64)
    got = table.overlap_anion_anion(R, Iu, Iv)
    ku, kv = torch.sqrt(2 * Iu / HARTREE), torch.sqrt(2 * Iv / HARTREE)
    ref = density_overlap(ku, torch.zeros_like(ku), kv, torch.zeros_like(kv), R / BOHR)
    assert float(((got - ref) / ref).abs().max()) <= 1e-7


def _octahedron(center_z, ligand_z, distance):
    pos = [[0.0, 0.0, 0.0]] + [[s * distance * (a == k) for a in range(3)] for k in range(3) for s in (1.0, -1.0)]
    return torch.tensor([center_z] + [ligand_z] * 6), torch.tensor(pos, dtype=torch.float64)


@pytest.mark.parametrize("center, ligand, d", [(8, 12, 2.10), (17, 11, 2.82)])
def test_a_hole_on_an_anion_relieves_its_exclusion(center, ligand, d):
    """O- in an Mg2+ octahedron and Cl0 in an Na+ octahedron, with in-situ removal energies from a bare Coulomb
    environment: the hole state (fewer, more tightly bound electrons) pays less exclusion than the frame."""
    from trace_redox.constants import formal_frame
    from trace_redox.diabatic import formal_ladder
    from trace_redox.exclusion import Exclusion, in_situ_removal, smooth_removal
    z, pos = _octahedron(center, ligand, d)
    frames = formal_frame(z)
    states, E_Z, _, _ = formal_ladder(z, 3)
    r = torch.cdist(pos, pos)
    S0 = torch.full((len(z),), 6.0, dtype=torch.float64)
    W = torch.where(r > 0, K_E / r.clamp_min(1e-9), torch.diag(S0))
    e_hat = E_Z - 0.5 * S0.unsqueeze(1) * states * states
    I = smooth_removal(in_situ_removal(e_hat, states, torch.zeros(len(z), dtype=torch.float64), W, frames))
    ei = neighbour_list(pos, r_max=5.0)
    _, length = edges(pos, ei)
    Pi = Exclusion(r_cut=5.0)(z, states, ei, length, frames, anion_removal=I, anion_holes=True)
    f0 = int(frames[0])
    k = {int(s): i for i, s in enumerate(states.tolist())}
    assert float(Pi[0, k[f0]]) == 0.0
    assert float(Pi[0, k[f0 + 1]]) < 0.0


def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True, exclusion=True,
                         **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


def test_in_situ_forces_are_minus_the_gradient():
    module, heads = _module(exclusion_anions="in_situ")
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0],
                         [2.0, 2.1, 0.2]], dtype=torch.float64)
    z = torch.tensor([8, 79, 79, 12, 13, 8])
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos0)

    def energy(p):
        vec, length = edges(p, ei)
        return module(h, p, ei, vec, length, data={"z": z, "total_charge": 0.0})[0]

    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    for i, axis in [(0, 2), (1, 2), (3, 0), (5, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-7 * (1.0 + abs(fd))


def test_it_needs_exclusion_and_refuses_unsupported_combinations():
    from trace_redox import RedoxConfig
    with pytest.raises(ValueError, match="exclusion_anions"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", exclusion_anions="in_situ")
    with pytest.raises(ValueError, match="exclusion_anions"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", exclusion=True, exclusion_anions="bulk")
    with pytest.raises(ValueError, match="in_situ"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", exclusion=True, exclusion_anions="in_situ",
                    exclusion_bound=True)


def test_in_situ_frames_alone_give_anions_no_carrier_states():
    """Default in-situ mode: anion frame densities in situ, anion units carry no exclusion of their own (the hole
    extension is off: on step 9's weights it drove spurious O -> Au transfer, measured 2026-10-17)."""
    from trace_redox.constants import formal_frame
    from trace_redox.diabatic import formal_ladder
    from trace_redox.exclusion import Exclusion, in_situ_removal, smooth_removal
    z, pos = _octahedron(8, 12, 2.10)
    frames = formal_frame(z)
    states, E_Z, _, _ = formal_ladder(z, 3)
    r = torch.cdist(pos, pos)
    S0 = torch.full((len(z),), 6.0, dtype=torch.float64)
    W = torch.where(r > 0, K_E / r.clamp_min(1e-9), torch.diag(S0))
    e_hat = E_Z - 0.5 * S0.unsqueeze(1) * states * states
    I = smooth_removal(in_situ_removal(e_hat, states, torch.zeros(len(z), dtype=torch.float64), W, frames))
    ei = neighbour_list(pos, r_max=5.0)
    _, length = edges(pos, ei)
    Pi = Exclusion(r_cut=5.0)(z, states, ei, length, frames, anion_removal=I)
    finite = torch.isfinite(Pi[0])
    assert float(Pi[0][finite].abs().max()) == 0.0


def test_holes_need_in_situ():
    from trace_redox import RedoxConfig
    with pytest.raises(ValueError, match="exclusion_anion_holes"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", exclusion=True, exclusion_anion_holes=True)
