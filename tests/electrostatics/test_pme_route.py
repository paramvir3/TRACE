"""SPEC 19: the matrix-free periodic route with the PME reciprocal sum.

``reciprocal="pme"`` must reproduce the exact (direct k-sum) route to the
requested Ewald accuracy -- energy, forces, charges, eps_inf -- for both
kernels, with quadrupoles and with an applied field.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from conftest import require


@pytest.mark.parametrize("kernel,quadrupoles", [("gaussian", True), ("thole", False)])
def test_pme_route_equals_the_direct_sum_to_the_requested_accuracy(kernel, quadrupoles):
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    electrostatic_state = require("energy", "electrostatic_state")
    dielectric_tensor = require("observables", "dielectric_tensor")
    lattice_pairs = require("lattice_operator", "lattice_pairs")
    a = 5.64
    basis = torch.tensor([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], dtype=torch.float64)
    conv = torch.cat([basis, basis + torch.tensor([.5, 0, 0])]) * a
    g = torch.Generator().manual_seed(2)
    pos0 = conv + 0.08 * torch.randn(8, 3, generator=g, dtype=torch.float64)
    cell = torch.eye(3) * a + 0.1 * torch.randn(3, 3, generator=g, dtype=torch.float64)
    extra = dict(quadrupole_blocks=("A_Theta", "chi_theta"), allow_truncated_a_theta=True) if quadrupoles else {}
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heads = ElectrostaticHeads(
            r_max=4.0, hardness_margin="legacy" if kernel == "thole" else 1.0, **extra
        )
    h = torch.randn(2, heads.irreps_in.dim, generator=g, dtype=torch.float64)[
        torch.tensor([0] * 4 + [1] * 4)
    ]
    pairs = lattice_pairs(pos0, cell, heads.r_max, (True, True, True))
    edge_index = torch.stack([torch.cat([pairs.first, pairs.second]), torch.cat([pairs.second, pairs.first])])
    edge_shift = torch.cat([-pairs.shift, pairs.shift]).to(torch.float64)
    field = torch.tensor([0.03, -0.01, 0.02], dtype=torch.float64)
    results = {}
    for reciprocal in ("ewald", "pme"):
        for external in (None, field):
            pos = pos0.clone().requires_grad_(True)
            vec = pos[edge_index[0]] - pos[edge_index[1]] + edge_shift @ cell
            config = ElectrostaticConfig(
                multipole_kernel=kernel, solver="matrix_free", reciprocal=reciprocal,
                ewald_accuracy=1e-9, ewald_real_cutoff=9.0, ewald_balance=False,
            )
            state = electrostatic_state(
                h, pos, edge_index, vec, vec.norm(dim=-1), heads, config, pbc=(True, True, True),
                cell=cell, edge_shift=edge_shift, external_field=external, quadrupoles=quadrupoles,
            )
            forces = -torch.autograd.grad(state.energy, pos, retain_graph=True)[0]
            eps = None if external is not None else dielectric_tensor(state.solve_result, state.assembly)
            results[(reciprocal, external is None)] = (state.energy.detach(), forces, state.solve_result.q.detach(), eps)
    for zero in (True, False):
        e1, f1, q1, eps1 = results[("ewald", zero)]
        e2, f2, q2, eps2 = results[("pme", zero)]
        assert abs(float(e1 - e2)) < 1e-8 * abs(float(e1))
        assert float((f1 - f2).abs().max()) < 1e-9
        assert float((q1 - q2).abs().max()) < 1e-9   # requested accuracy 1e-9
        if eps1 is not None:
            assert float((eps1 - eps2).abs().max()) < 1e-7


def test_pme_requires_the_matrix_free_solver():
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    with pytest.raises(ValueError, match="matrix_free"):
        ElectrostaticConfig(reciprocal="pme")
