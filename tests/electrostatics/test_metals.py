"""SPEC 20: metallic domains and constant-potential electrodes.

Exact identities, not fits:
* a metallic domain is equipotential -- u_i = dU/dq_i, computed independently
  by autograd of the assembly's own energy, is uniform over it;
* its total charge is conserved;
* the external circuit imposes u_e - u_ref = V_e - V_ref, conserves the total
  charge, and the electrode charges are linear in the applied difference;
* Legendre: dE/dV_e = -(charge the circuit delivered to e);
* forces are exact; the matrix-free open route equals the dense one;
* the image-charge law is approached as a metallic plate grows (continuum
  limit, via forces so the rim-charge offset of a neutral plate drops out).
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from conftest import require

K_E = 14.399645


@pytest.fixture(scope="module")
def setup():
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heads = ElectrostaticHeads(r_max=3.0)
    g = torch.Generator().manual_seed(1)
    features = torch.randn(3, heads.irreps_in.dim, generator=g, dtype=torch.float64)
    features[:, 64:] = 0.0
    return heads, features


def _plate(n, z, a=2.5):
    return torch.tensor([[i * a, j * a, z] for i in range(n) for j in range(n)], dtype=torch.float64)


def _run(setup, pos, species, metallic, fragment_id, fragment_charges, potential=None, solver="dense"):
    electrostatic_state = require("energy", "electrostatic_state")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    heads, features = setup
    distance = torch.cdist(pos, pos)
    src, dst = torch.nonzero((distance < 3.0) & (distance > 0), as_tuple=True)
    edge_index = torch.stack([src, dst])
    p = pos.clone().requires_grad_(True)
    vec = p[edge_index[0]] - p[edge_index[1]]
    state = electrostatic_state(
        features[species], p, edge_index, vec, vec.norm(dim=-1), heads,
        ElectrostaticConfig(solver=solver), fragment_id=fragment_id,
        fragment_charges=fragment_charges, metallic=metallic, electrode_potential=potential,
    )
    return state, p


def _potentials(state):
    """u_i = dU/dq_i at the solution, from the assembly's own energy."""
    assembly, result = state.assembly, state.solve_result
    rho = result.rho.detach().clone().requires_grad_(True)
    mu_hat = (result.mu / assembly.params.ell.unsqueeze(-1)).detach()
    energy = assembly.energy(result.t.detach(), rho, mu_hat)
    return torch.autograd.grad(energy, rho)[0]


def test_a_metallic_domain_is_equipotential_and_conserves_its_charge(setup):
    pos = torch.cat([_plate(4, 0.0), torch.tensor([[3.75, 3.75, 4.0]])])
    species = torch.tensor([0] * 16 + [1])
    fragment = torch.tensor([0] * 16 + [1])
    metallic = torch.tensor([True] * 16 + [False])
    state, p = _run(setup, pos, species, metallic, fragment, [0.0, 1.0])
    u = _potentials(state)
    q = state.solve_result.q.detach()
    assert float(u[:16].max() - u[:16].min()) < 1e-7          # measured 1.3e-9 V
    assert abs(float(q[:16].sum())) < 1e-12
    dielectric, _ = _run(setup, pos, species, torch.zeros(17, dtype=torch.bool), fragment, [0.0, 1.0])
    u0 = _potentials(dielectric)
    assert float(u0[:16].max() - u0[:16].min()) > 1e-3        # a dielectric is not (6.8e-2 V)
    # forces exact
    force = -torch.autograd.grad(state.energy, p)[0][16, 2]
    h = 1e-5
    up, down = pos.clone(), pos.clone()
    up[16, 2] += h
    down[16, 2] -= h
    fd = -(float(_run(setup, up, species, metallic, fragment, [0.0, 1.0])[0].energy)
           - float(_run(setup, down, species, metallic, fragment, [0.0, 1.0])[0].energy)) / (2 * h)
    assert abs(float(force) - fd) < 1e-8


def test_electrodes_hold_the_applied_potential_difference(setup):
    pos = torch.cat([_plate(4, 0.0), _plate(4, 6.0)])
    species = torch.zeros(32, dtype=torch.long)
    metallic = torch.ones(32, dtype=torch.bool)
    fragment = torch.zeros(32, dtype=torch.long)
    charges = []
    for dV in (0.5, 1.0):
        V = torch.tensor([0.0] * 16 + [dV] * 16)
        state, _ = _run(setup, pos, species, metallic, fragment, [0.0], V)
        u = _potentials(state)
        q = state.solve_result.q.detach()
        assert abs(float(u[16:].mean() - u[:16].mean()) - dV) < 1e-7   # 1e-8 bond hardness x tau
        assert float(u[:16].max() - u[:16].min()) < 1e-7
        assert abs(float(q.sum())) < 1e-12
        charges.append(float(q[16:].sum()))
    assert charges[0] > 0.0                                  # the higher-potential plate is positive
    assert abs(charges[1] - 2.0 * charges[0]) < 1e-8 * abs(charges[1])

    def at(dV):
        V = torch.tensor([0.0] * 16 + [dV] * 16)
        return _run(setup, pos, species, metallic, fragment, [0.0], V)[0]

    h = 1e-5
    slope = (float(at(1.0 + h).energy) - float(at(1.0 - h).energy)) / (2 * h)
    state = at(1.0)
    delivered = float(state.solve_result.t[-1] * state.assembly.envelope[-1])
    assert abs(slope + delivered) < 1e-7                      # Legendre: dE/dV = -tau_circuit
    free, _ = _run(setup, pos, species, metallic, fragment, [0.0],
                   torch.tensor([0.0] * 16 + [1.0] * 16), solver="matrix_free")
    assert abs(float(free.energy - state.energy)) < 1e-10
    assert float((free.solve_result.q - state.solve_result.q).abs().max()) < 1e-9


def test_electrode_errors_are_named(setup):
    pos = torch.cat([_plate(3, 0.0), _plate(3, 6.0)])
    species = torch.zeros(18, dtype=torch.long)
    metallic = torch.ones(18, dtype=torch.bool)
    with pytest.raises(ValueError, match="counter-electrode"):
        _run(setup, pos, species, metallic, torch.zeros(18, dtype=torch.long), [0.0],
             torch.tensor([1.0] * 9 + [float("nan")] * 9))
    with pytest.raises(ValueError, match="different fragments"):
        _run(setup, pos, species, metallic, torch.tensor([0] * 9 + [1] * 9), [0.0, 0.0],
             torch.tensor([0.0] * 9 + [1.0] * 9))
    with pytest.raises(ValueError, match="metallic"):
        _run(setup, pos, species, torch.zeros(18, dtype=torch.bool), torch.zeros(18, dtype=torch.long),
             [0.0], torch.tensor([0.0] * 9 + [1.0] * 9))


@pytest.mark.slow
def test_the_image_charge_law_is_approached_as_the_plate_grows(setup):
    """(-F_z)^(-1/2) = 2 (z - z0)/sqrt(K_E) for a charge above a conductor.

    Measured slope over z = 4-6 A: 0.648 (27.5 A plate), 0.552 (57.5 A plate)
    against 2/sqrt(K_E) = 0.527 -- the finite-plate excess shrinks with size.
    """
    ideal = 2.0 / math.sqrt(K_E)
    slopes = []
    for n in (12, 24):
        values = []
        for z in (4.0, 6.0):
            N = n * n + 1
            c = (n - 1) * 2.5 / 2
            pos = torch.cat([_plate(n, 0.0), torch.tensor([[c, c, z]])])
            state, p = _run(setup, pos, torch.tensor([0] * (N - 1) + [1]),
                            torch.tensor([True] * (N - 1) + [False]),
                            torch.tensor([0] * (N - 1) + [1]), [0.0, 1.0], solver="matrix_free")
            force = -torch.autograd.grad(state.energy, p)[0][-1, 2]
            values.append(float(-force) ** -0.5)
        slopes.append((values[1] - values[0]) / 2.0)
    assert slopes[1] < slopes[0]
    assert abs(slopes[1] - ideal) < 0.1 * ideal
