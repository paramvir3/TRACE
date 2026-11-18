"""SPEC 18: open boundaries through the matrix-free route (``OpenOperator``).

The same functional as the dense open route, with no ``n x n`` object: energy,
forces, charges and the polarizability agree to the solve tolerance, for both
kernels, with quadrupoles and with an applied field.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from conftest import require


def _case(kernel, quadrupoles, n=14, seed=4):
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g, dtype=torch.float64) * 1.7
    extra = dict(quadrupole_blocks=("A_Theta", "chi_theta"), allow_truncated_a_theta=True) if quadrupoles else {}
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heads = ElectrostaticHeads(
            r_max=3.5, hardness_margin="legacy" if kernel == "thole" else 1.0, **extra
        )
    torch.manual_seed(1)
    h = torch.randn(n, heads.irreps_in.dim, dtype=torch.float64)
    distance = torch.cdist(pos, pos)
    src, dst = torch.nonzero((distance < 3.5) & (distance > 0), as_tuple=True)
    return heads, h, pos, torch.stack([src, dst])


@pytest.mark.parametrize("kernel,quadrupoles", [("gaussian", False), ("gaussian", True), ("thole", False)])
@pytest.mark.parametrize("field", [None, (0.05, -0.02, 0.03)])
def test_open_matrix_free_equals_the_dense_route(kernel, quadrupoles, field):
    electrostatic_state = require("energy", "electrostatic_state")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    polarizability = require("observables", "polarizability")
    heads, h, pos0, edge_index = _case(kernel, quadrupoles)
    field_t = None if field is None else torch.tensor(field, dtype=torch.float64)
    results = {}
    for solver in ("dense", "matrix_free"):
        pos = pos0.clone().requires_grad_(True)
        vec = pos[edge_index[0]] - pos[edge_index[1]]
        state = electrostatic_state(
            h, pos, edge_index, vec, vec.norm(dim=-1), heads,
            ElectrostaticConfig(multipole_kernel=kernel, solver=solver),
            total_charge=1.0, external_field=field_t, quadrupoles=quadrupoles,
        )
        forces = -torch.autograd.grad(state.energy, pos, retain_graph=True)[0]
        alpha = polarizability(state.solve_result, state.assembly) if field is None else None
        results[solver] = (state.energy.detach(), forces, state.solve_result.q.detach(), alpha)
        if solver == "matrix_free":
            assert state.assembly.lattice_operator is not None
    (e1, f1, q1, a1), (e2, f2, q2, a2) = results["dense"], results["matrix_free"]
    assert abs(float(e1 - e2)) < 1e-12
    assert float((f1 - f2).abs().max()) < 1e-9
    assert float((q1 - q2).abs().max()) < 1e-9
    if a1 is not None:
        assert float((a1 - a2).abs().max()) < 1e-8 * float(a1.abs().max())
