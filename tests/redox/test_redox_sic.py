"""Certified self-interaction removal (trace_redox.sic; SPEC §8a).

The certificate is a theorem; it is checked as a property on random positive
definite matrices and on the model's own screened operators.  Exactness is
checked against the EXACT classical ensemble of a two-unit system, enumerated
configuration by configuration -- an object that shares nothing with the
mean-field solve.
"""

import math

import pytest
import torch
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)
from redox_helpers import edges, make_heads, neighbour_list

from trace_electrostatics.config import ElectrostaticConfig
from trace_redox import RedoxConfig, RedoxElectrostatics
from trace_redox.bound import assemble_neutral, eliminate_bound_sector
from trace_redox.sic import certified_removal

EPS = torch.finfo(torch.float64).eps


def _random_pd(n, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    return A @ A.T + 1e-2 * torch.eye(n)


@pytest.mark.parametrize("seed", range(40))
def test_certificate_holds_for_random_positive_definite_matrices(seed):
    n = 2 + seed % 30
    W = _random_pd(n, seed)
    for smoothing in (0.0, 0.01, 1.0):
        r = certified_removal(W, W, theta=0.999, smoothing=smoothing)
        assert bool((r > 0).all())
        # strictly positive definite with theta < 1: lambda_min >= (1 - theta) * lambda_min-ish > 0
        assert float(torch.linalg.eigvalsh(W - torch.diag(r)).min()) > -64 * EPS * float(W.abs().max())
        _, info = torch.linalg.cholesky_ex(W - torch.diag(r))
        assert int(info) == 0
        # never more than the conditional (Schur) self-energy, never more than W_uu
        M = torch.linalg.inv(W)
        assert bool((r <= 1.0 / torch.diagonal(M) * (1 + 1e-12)).all())
        assert bool((r <= torch.diagonal(W) * (1 + 1e-12)).all())


def test_smoothing_only_loosens():
    W = _random_pd(12, 3)
    r0 = certified_removal(W, W, smoothing=0.0)
    for smoothing in (1e-3, 0.1, 10.0):
        assert bool((certified_removal(W, W, smoothing=smoothing) <= r0 + 1e-15).all())


@pytest.mark.parametrize("a,d,w", [(1.0, 1.0, 0.3), (2.3, 4.67, 0.144), (2.3, 4.67, 1.44), (5.0, 1.0, -0.9)])
def test_two_units_tight_with_the_closed_form_transfer_curvature(a, d, w):
    W = torch.tensor([[a, w], [w, d]])
    r = certified_removal(W, W, theta=1.0 - 1e-12, smoothing=0.0)
    Q = W - torch.diag(r)
    assert abs(float(torch.det(Q))) <= 1e-9 * a * d                       # tight: rank one
    curvature = float(Q[0, 0] + Q[1, 1] - 2 * Q[0, 1] * math.copysign(1.0, w))
    expected = abs(w) * (a - d) ** 2 / ((a + abs(w)) * (d + abs(w)))
    assert abs(curvature - expected) <= 1e-9 * (a + d)


def test_an_isolated_unit_loses_its_whole_on_site_coulomb():
    W = torch.diag(torch.tensor([2.0, 3.5, 7.0]))
    r = certified_removal(W, W, theta=0.999, smoothing=0.01)
    assert torch.allclose(r, 0.999 * torch.diagonal(W), rtol=1e-14, atol=0.0)


def test_removal_is_smooth_through_a_sign_change_of_an_inverse_entry():
    """r is differentiable where an off-diagonal (W^-1)_uv crosses zero."""
    def W_of(t):
        return torch.stack([
            torch.stack([torch.tensor(3.0), 1.0 + t, torch.tensor(1.0)]),
            torch.stack([1.0 + t, torch.tensor(3.0), torch.tensor(1.0)]),
            torch.stack([torch.tensor(1.0), torch.tensor(1.0), torch.tensor(3.0)]),
        ])
    # (W^-1)_12 = (W_13 W_23 - W_12 W_33)/det vanishes at W_12 = 1/3, t = -2/3
    t = torch.tensor(-2.0 / 3.0, requires_grad=True)
    W = W_of(t)
    assert abs(float(torch.linalg.inv(W.detach())[0, 1])) <= 1e-12
    r = certified_removal(W, W, smoothing=0.01)
    (g,) = torch.autograd.grad(r.sum(), t, create_graph=True)
    (h,) = torch.autograd.grad(g, t)
    assert math.isfinite(float(g)) and math.isfinite(float(h))


def _module(**redox):
    """The atom-level certificate of §8a: the mean field over atoms (fragments='none')."""
    heads = make_heads()
    torch.manual_seed(1)
    redox.setdefault("fragments", "none")
    return RedoxElectrostatics(heads, ElectrostaticConfig(), RedoxConfig(**redox)), heads


def _exact_two_unit_free_energy(module, heads, h, pos, z, Q, kT):
    """-kT ln sum_n exp(-E(n)/kT) over the integer configurations with sum n = Q.

    E(n) = E_b0 + sum_u Ehat_u(n_u) + b.n + 1/2 n.W.n, the v1 state energies
    and the screened W (no SIC): the exact classical lattice gas of the model.
    """
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    _, res = module(h, pos, ei, vec, length, data={"z": z, "total_charge": Q})
    assembly = assemble_neutral(h, pos, ei, vec, length, heads, module.config, species=z)
    elim = eliminate_bound_sector(assembly, res.sigma_free)
    chi, gap = res.electronegativity, res.gap
    e_hat = {-1: -chi + 0.5 * gap, 0: torch.zeros_like(chi), 1: chi + 0.5 * gap}
    energies = []
    for na in (-1, 0, 1):
        for nb in (-1, 0, 1):
            if na + nb != Q:
                continue
            n = torch.tensor([float(na), float(nb)])
            energies.append(elim.e_b0 + e_hat[na][0] + e_hat[nb][1] + elim.u0 @ n + 0.5 * n @ (elim.W @ n))
    energies = torch.stack(energies)
    return float(-kT * torch.logsumexp(-energies / kT, dim=0)), res, elim


@pytest.mark.parametrize("R", [12.0, 25.0, 60.0])
def test_separated_na_cl_is_exact_with_the_certified_removal(R):
    """Mean field + certified SIC reproduces the exact ensemble; v1 does not (SPEC §8)."""
    kT = 0.005
    pos = torch.tensor([[0.0, 0.0, 0.0], [R, 0.0, 0.0]])
    z = torch.tensor([11, 17])
    heads = make_heads()
    h = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(7))
    cert, _ = _module(kT=kT, sic="certified")
    cert.heads = heads
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    energy, res = cert(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
    exact, _, elim = _exact_two_unit_free_energy(cert, heads, h, pos, z, 0.0, kT)
    W = elim.W - torch.diag(res.sic_removal)
    curvature = float(W[0, 0] + W[1, 1] - 2 * W[0, 1])
    # mean-field deficit <= curvature / 8 (max of 1/2 curv delta(1 - delta)), entropies <= kT ln 3 each
    bound = curvature / 8.0 + 3.0 * kT * math.log(3.0)
    assert abs(float(energy) - exact) <= bound
    assert abs(float(res.unit_charge[0])) <= 0.02                       # no spurious transfer
    v1, _ = _module(kT=kT, sic="none")
    v1.heads = heads
    energy_v1, res_v1 = v1(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
    assert float(res_v1.unit_charge[0]) > 0.25                          # the measured v1 error
    assert float(energy_v1) < exact - 0.1                               # spuriously low


def test_a_hole_shared_by_two_distant_equal_atoms_costs_the_exact_ionisation():
    """Na...Na at 40 A with Q = +1: exact F = E(Na+ Na0) - kT ln 2; equal units have zero residual curvature."""
    kT = 0.005
    pos = torch.tensor([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]])
    z = torch.tensor([11, 11])
    heads = make_heads()
    h = torch.randn(1, heads.irreps_in.dim, generator=torch.Generator().manual_seed(8)).repeat(2, 1)
    cert, _ = _module(kT=kT, sic="certified")
    cert.heads = heads
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    energy, res = cert(h, pos, ei, vec, length, data={"z": z, "total_charge": 1.0})
    exact, _, _ = _exact_two_unit_free_energy(cert, heads, h, pos, z, 1.0, kT)
    assert abs(float(energy) - exact) <= 3.0 * kT * math.log(3.0)
    v1, _ = _module(kT=kT, sic="none")
    v1.heads = heads
    energy_v1, _ = v1(h, pos, ei, vec, length, data={"z": z, "total_charge": 1.0})
    assert float(energy_v1) < exact - 0.3                               # v1: the H2+-like SIE


def test_certified_mode_refuses_a_uniform_lambda():
    with pytest.raises(ValueError):
        RedoxConfig(sic="certified", self_interaction=0.01)
    with pytest.raises(ValueError):
        RedoxConfig(sic="certified", sic_theta=1.0)
