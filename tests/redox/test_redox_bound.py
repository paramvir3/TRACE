"""Exact elimination of the bound sector (trace_redox.bound).

Checked against trace_electrostatics' own, separately tested, constructions:
with the free width equal to the bound width the free blocks ARE the
assembly's pair blocks, and a free charge is then the assembly's ``rho``
minus its chi and (eta - S0) terms -- an identity through ``assembly.energy``
that shares no code with the Schur elimination.
"""

import pytest
import torch

from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)
from redox_helpers import cluster, edges, make_heads
from trace_electrostatics.config import ElectrostaticConfig
from trace_electrostatics.energy import electrostatic_state
from trace_electrostatics.kernels import K_E, gram_self_energy
from trace_redox.bound import (
    BoundSectorError,
    assemble_neutral,
    charge_coulomb_block,
    eliminate_bound_sector,
    free_charge_blocks,
)

EPS = torch.finfo(torch.float64).eps


def _assembly(quadrupoles=False, field=None, n=6, seed=5):
    heads = make_heads(quadrupoles=quadrupoles)
    pos, h, ei = cluster(n, heads.irreps_in.dim, seed)
    vec, length = edges(pos, ei)
    assembly = assemble_neutral(
        h, pos, ei, vec, length, heads, ElectrostaticConfig(),
        quadrupoles=quadrupoles, external_field=field,
    )
    return assembly, heads, (h, pos, ei)


def _bound_energy(assembly, y, rho=None):
    n, nb = assembly.n_atoms, assembly.n_bonds
    rho = torch.zeros(n) if rho is None else rho
    theta = y[nb + 3 * n:].reshape(n, 5) if assembly.quadrupoles else None
    return assembly.energy(y[:nb], rho, y[nb:nb + 3 * n].reshape(n, 3), theta)


@pytest.mark.parametrize("quadrupoles", [False, True])
def test_equal_widths_reproduce_the_assembly_blocks(quadrupoles):
    assembly, _, _ = _assembly(quadrupoles)
    sigma = assembly.params.sigma
    J, A = free_charge_blocks(assembly, sigma)
    n = assembly.n_atoms
    J_ref = assembly.j_pair + torch.diag(gram_self_energy(sigma))
    blocks = [J_ref @ assembly._dense_incidence(), assembly.t01_pair.reshape(n, 3 * n)]
    if quadrupoles:
        blocks.append(assembly.t02_pair.reshape(n, 5 * n))
    A_ref = torch.cat(blocks, dim=1)
    assert torch.allclose(J, J_ref, rtol=0.0, atol=32 * EPS * float(J_ref.abs().max()))
    assert torch.allclose(A, A_ref, rtol=0.0, atol=32 * EPS * float(A_ref.abs().max()))


def test_mixed_width_coulomb_block_is_a_gram_matrix():
    g = torch.Generator().manual_seed(2)
    pos = torch.randn(7, 3, generator=g) * 1.3
    a = 0.6 + torch.rand(7, generator=g)
    b = 1.5 + 2.0 * torch.rand(7, generator=g)
    Mab = charge_coulomb_block(pos, a, b)
    Mba = charge_coulomb_block(pos, b, a)
    assert torch.equal(Mab, Mba.T)
    # the joint Gram matrix of 14 Gaussians (two per centre) is positive definite
    joint = torch.cat((torch.cat((charge_coulomb_block(pos, a, a), Mab), 1),
                       torch.cat((Mba, charge_coulomb_block(pos, b, b)), 1)), 0)
    assert float(torch.linalg.eigvalsh(joint).min()) > 0.0
    # concentric clouds: 2 K_E / (sqrt(pi) sqrt(a^2 + b^2)), the r -> 0 limit of K_E erf(r/s)/r
    expected = 2.0 * K_E / (torch.sqrt(torch.tensor(torch.pi)) * torch.sqrt(a * a + b * b))
    assert torch.allclose(torch.diagonal(Mab), expected, rtol=8 * EPS, atol=0.0)
    # far field: K_E / r, to erfc(r / s) ~ 1e-22 at r = 60 A
    far = torch.tensor([[0.0, 0.0, 0.0], [60.0, 0.0, 0.0]])
    value = charge_coulomb_block(far, a[:2], b[:2])[0, 1]
    assert abs(float(value) - K_E / 60.0) <= 8 * EPS * K_E / 60.0


@pytest.mark.parametrize("quadrupoles", [False, True])
def test_bound_energy_is_exactly_quadratic(quadrupoles):
    assembly, _, _ = _assembly(quadrupoles)
    elim = eliminate_bound_sector(assembly, assembly.params.sigma)
    m = int(elim.y0.shape[0])
    K = elim.operator.matrix()
    g = -(K @ elim.y0)
    e0 = _bound_energy(assembly, torch.zeros(m))
    gen = torch.Generator().manual_seed(3)
    for _ in range(4):
        y = torch.randn(m, generator=gen) * 0.3
        exact = _bound_energy(assembly, y)
        model = e0 + g @ y + 0.5 * y @ (K @ y)
        assert abs(float(exact - model)) <= 1e-10 * (1.0 + abs(float(exact)))


@pytest.mark.parametrize("quadrupoles,field", [(False, None), (True, None), (False, (0.05, -0.02, 0.03))])
def test_free_charge_with_equal_width_is_rho_without_chi_and_eta(quadrupoles, field):
    """U(t, rho=f, ...) = U_b(y) + f^T A y + 1/2 f^T J f + f^T h + chi^T f + sum (eta - S0)(q_b f + f^2/2)."""
    field = None if field is None else torch.tensor(field)
    assembly, _, _ = _assembly(quadrupoles, field)
    sigma = assembly.params.sigma
    J, A = free_charge_blocks(assembly, sigma)
    n, m = assembly.n_atoms, A.shape[1]
    gen = torch.Generator().manual_seed(4)
    y = torch.randn(m, generator=gen) * 0.2
    f = torch.randn(n, generator=gen) * 0.3
    h = torch.zeros(n) if field is None else -(assembly.pos @ field)
    q_b = assembly.charges(y[:assembly.n_bonds], torch.zeros(n))
    extra = assembly.params.chi @ f + ((assembly.params.eta - gram_self_energy(sigma)) * (q_b * f + 0.5 * f * f)).sum()
    lhs = _bound_energy(assembly, y, rho=f)
    rhs = _bound_energy(assembly, y) + f @ (A @ y) + 0.5 * f @ (J @ f) + f @ h + extra
    assert abs(float(lhs - rhs)) <= 1e-11 * (1.0 + abs(float(lhs)))


@pytest.mark.parametrize("width,field", [("equal", None), ("wide", None), ("wide", (0.04, 0.03, -0.05))])
def test_elimination_is_the_minimum_over_the_bound_moments(width, field):
    field = None if field is None else torch.tensor(field)
    assembly, _, _ = _assembly(quadrupoles=True, field=field)
    wide = 1.0 + 2.5 * torch.rand(assembly.n_atoms, generator=torch.Generator().manual_seed(8))
    sigma = assembly.params.sigma if width == "equal" else wide
    elim = eliminate_bound_sector(assembly, sigma)
    n = assembly.n_atoms
    h = torch.zeros(n) if field is None else -(assembly.pos @ field)     # -r . E on a free charge
    gen = torch.Generator().manual_seed(6)
    for _ in range(3):
        f = torch.randn(n, generator=gen) * 0.4
        y = elim.bound_moments(f).detach().requires_grad_(True)
        total = (_bound_energy(assembly, y) + f @ (elim.A.detach() @ y)
                 + 0.5 * f @ (elim.J.detach() @ f) + f @ h)
        (grad,) = torch.autograd.grad(total, y)
        assert float(grad.abs().max()) <= 1e-10
        reduced = elim.e_b0 + elim.u0 @ f + 0.5 * f @ (elim.W @ f)
        assert abs(float(total - reduced)) <= 1e-11 * (1.0 + abs(float(total)))


def test_screened_operator_is_positive_definite_and_below_the_bare_one():
    assembly, _, _ = _assembly(quadrupoles=True, n=8, seed=9)
    sigma = 1.0 + 2.5 * torch.rand(assembly.n_atoms, generator=torch.Generator().manual_seed(1))
    elim = eliminate_bound_sector(assembly, sigma)
    assert torch.equal(elim.W, elim.W.T)
    assert float(torch.linalg.eigvalsh(elim.W).min()) > 0.0
    gap = torch.linalg.eigvalsh(elim.J - elim.W)                       # = A K^-1 A^T >= 0
    assert float(gap.min()) >= -64 * EPS * float(elim.J.abs().max())
    assert elim.asymmetry <= 1e-13


def test_bound_minimum_is_spec1_for_a_neutral_structure():
    assembly, heads, (h, pos, ei) = _assembly(quadrupoles=False, n=7, seed=12)
    elim = eliminate_bound_sector(assembly, assembly.params.sigma)
    vec, length = edges(pos, ei)
    state = electrostatic_state(h, pos, ei, vec, length, heads, ElectrostaticConfig())
    assert abs(float(elim.e_b0 - state.energy)) <= 1e-9 * (1.0 + abs(float(state.energy)))


def test_refusals():
    heads = make_heads()
    pos, h, ei = cluster(5, heads.irreps_in.dim, 3)
    vec, length = edges(pos, ei)
    with pytest.raises(BoundSectorError):
        assemble_neutral(h, pos, ei, vec, length, heads, ElectrostaticConfig(),
                         fragment_id=torch.zeros(5, dtype=torch.long))
    cell = torch.eye(3) * 12.0
    periodic = assemble_neutral(h, pos, ei, vec, length, heads, ElectrostaticConfig(),
                                pbc=(True, True, True), cell=cell, edge_shift=torch.zeros(ei.shape[1], 3),
                                external_field=torch.tensor([0.0, 0.0, 0.01]))
    with pytest.raises(BoundSectorError, match="field"):
        eliminate_bound_sector(periodic, periodic.params.sigma)


# ---------------------------------------------------------------------------
# the condensed elimination (SPEC §13) against the dense reference
# ---------------------------------------------------------------------------

def _both_routes(quadrupoles, field, periodic, n=8, seed=5):
    """(dense, condensed): E(f), forces and d(sum F^2)/dtheta for a fixed free charge f."""
    from redox_helpers import lattice_graph, periodic_edges
    heads = make_heads(quadrupoles=quadrupoles)
    pos0, h, ei = cluster(n, heads.irreps_in.dim, seed)
    sigma = 1.0 + 2.0 * torch.rand(n, generator=torch.Generator().manual_seed(3))
    f = 0.3 * torch.randn(n, generator=torch.Generator().manual_seed(4))
    out = []
    for method in ("dense", "condensed"):
        pos = pos0.clone().requires_grad_(True)
        if periodic is None:
            vec, length = edges(pos, ei)
            keywords, e_i = {}, ei
        else:
            cell, pbc = periodic
            e_i, shift = lattice_graph(pos0, cell, pbc)
            vec, length = periodic_edges(pos, cell, e_i, shift)
            keywords = dict(pbc=pbc, cell=cell, edge_shift=shift)
        assembly = assemble_neutral(h, pos, e_i, vec, length, heads, ElectrostaticConfig(),
                                    quadrupoles=quadrupoles, external_field=field, **keywords)
        elim = eliminate_bound_sector(assembly, sigma, method=method)
        energy = elim.e_b0 + elim.u0 @ f + 0.5 * f @ (elim.W @ f)
        (force,) = torch.autograd.grad(energy, pos, create_graph=True)
        params = [p for p in heads.parameters() if p.requires_grad]
        grads = torch.autograd.grad((force ** 2).sum(), params, allow_unused=True)
        K = elim.operator.matrix().detach()
        out.append(dict(energy=energy.detach(), W=elim.W.detach(), u0=elim.u0.detach(), y0=elim.y0.detach(),
                        force=force.detach(), grad=torch.cat([g.reshape(-1) for g in grads if g is not None]),
                        K=K))
    return out


@pytest.mark.parametrize("quadrupoles,field,periodic", [
    (False, None, None),
    (True, None, None),
    (False, (0.05, -0.02, 0.03), None),
    (True, None, "3d"),
    (False, None, "slab"),
])
def test_condensed_elimination_equals_the_dense_one(quadrupoles, field, periodic):
    """Same E_b0, u0, W, y0, forces and force-loss parameter gradient as the dense Cholesky route,
    to 64 cond(K) eps (the two are different exact factorisations; measured <= 3 cond(K) eps);
    and the condensed operator's K is the dense K in the variational basis."""
    field = None if field is None else torch.tensor(field)
    cells = {"3d": (torch.eye(3) * 9.0, (True, True, True)),
             "slab": (torch.diag(torch.tensor([9.0, 9.0, 30.0])), (True, True, False))}
    dense, condensed = _both_routes(quadrupoles, field, cells.get(periodic))
    evals = torch.linalg.eigvalsh(dense["K"])
    tol = 64.0 * float(evals.max() / evals.min()) * EPS
    scale = float(dense["K"].abs().max())
    assert float((condensed["K"] - dense["K"]).abs().max()) <= 64 * EPS * scale
    for key in ("energy", "W", "u0", "y0", "force", "grad"):
        ref = dense[key]
        assert float((condensed[key] - ref).abs().max()) <= tol * float(ref.abs().max()), key


def test_the_module_is_the_same_on_both_routes():
    """RedoxElectrostatics end to end (fragments, periodic cell): energy and forces."""
    from redox_helpers import lattice_graph, periodic_edges
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    pos0, h, _ = cluster(6, heads.irreps_in.dim, 9, scale=1.4)
    z = torch.tensor([8, 1, 1, 8, 1, 1])
    cell, pbc = torch.eye(3) * 10.0, (True, True, True)
    results = []
    for method in ("dense", "condensed"):
        torch.manual_seed(1)
        module = RedoxElectrostatics(heads, ElectrostaticConfig(), RedoxConfig(bound_elimination=method))
        pos = pos0.clone().requires_grad_(True)
        ei, shift = lattice_graph(pos0, cell, pbc)
        vec, length = periodic_edges(pos, cell, ei, shift)
        energy, _ = module(h, pos, ei, vec, length, pbc=pbc, cell=cell, edge_shift=shift,
                           data={"z": z, "total_charge": 1.0})
        (grad,) = torch.autograd.grad(energy, pos)
        results.append((float(energy), grad))
    (e_d, g_d), (e_c, g_c) = results
    assert abs(e_c - e_d) <= 1e4 * EPS * (1.0 + abs(e_d))
    assert float((g_c - g_d).abs().max()) <= 1e4 * EPS * (1.0 + float(g_d.abs().max()))
