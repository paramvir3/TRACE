"""TRACE-Redox end to end (trace_redox.module; SPEC 25.5 T2-T4, 25.7).

Forces against central differences, invariances, the isolated-atom limits
(the NIST values exactly), the dead zone (SPEC 1 exactly), the two-site
transfer law on a real Na...Cl pair, the parameter gradient of a force loss
(a second derivative through both solves), and the flashace seam.
"""

import math

import pytest
import torch

from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)
from redox_helpers import cluster, edges, make_heads, neighbour_list
from trace_electrostatics.config import ElectrostaticConfig
from trace_electrostatics.energy import electrostatic_state
from trace_redox import RedoxConfig, RedoxElectrostatics, attach_redox, ensemble
from trace_redox.bound import assemble_neutral, eliminate_bound_sector
from trace_redox.heads import smooth_switch
from trace_redox.bound import BoundSectorError
from trace_redox.constants import ATOMIC_ELECTRON_AFFINITY_EV, ATOMIC_IONIZATION_EV

Z_MIX = torch.tensor([8, 1, 1, 11, 17, 6])


def _module(seed=1, quadrupoles=False, **redox):
    heads = make_heads(quadrupoles=quadrupoles)
    torch.manual_seed(seed)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), RedoxConfig(**redox)), heads


def _randomise(module, seed, scale=0.3):
    """Give the zero-initialised output layers non-zero weights (a trained-like model)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        last = module.charge_states.shift_mlp[-1]
        last.weight.copy_(torch.randn(last.weight.shape, generator=g) * scale)
        last.bias.copy_(torch.randn(last.bias.shape, generator=g) * scale)
        if module.width is not None:
            module.width.logit.add_(torch.randn(module.width.logit.shape, generator=g) * 0.3)
        if module.orbitals is not None:
            spread = module.orbitals.spread[-1]
            spread.weight.copy_(torch.randn(spread.weight.shape, generator=g) * scale)


def _energy(module, h, pos, ei, z, Q, **data):
    vec, length = edges(pos, ei)
    return module(h, pos, ei, vec, length, data={"z": z, "total_charge": Q, **data})


CASES = [
    dict(Q=0.0),
    dict(Q=1.0),
    dict(Q=-1.0),
    dict(Q=0.0, orbitals=True, fragments="none"),
    dict(Q=1.0, orbitals=True, self_interaction=0.01, sic="none", fragments="none"),   # lam_max = 0.021 here
    dict(Q=-1.0, quadrupoles=True),
    dict(Q=2.0, anchor="gap", width="bound", gap=8.0),
    dict(Q=1.0, sic="certified"),
    dict(Q=-1.0, sic="certified", orbitals=True, quadrupoles=True, fragments="none"),
]


@pytest.mark.parametrize("case", CASES, ids=[str(c) for c in CASES])
def test_forces_are_minus_the_gradient(case):
    case = dict(case)
    Q, quadrupoles = case.pop("Q"), case.pop("quadrupoles", False)
    module, heads = _module(quadrupoles=quadrupoles, **case)
    _randomise(module, 3)
    pos0, h, ei = cluster(6, heads.irreps_in.dim, 3)
    pos = pos0.clone().requires_grad_(True)
    energy, result = _energy(module, h, pos, ei, Z_MIX, Q)
    (grad,) = torch.autograd.grad(energy, pos)
    assert abs(float(result.q.sum()) - Q) <= 1e-10
    step = 1e-5
    for i, a in [(0, 0), (2, 1), (3, 2), (4, 0)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, a] += step
        minus[i, a] -= step
        fd = (float(_energy(module, h, plus, ei, Z_MIX, Q)[0]) - float(_energy(module, h, minus, ei, Z_MIX, Q)[0])) / (2 * step)
        # central difference: O(step^2 E''') truncation and ~1e-15 / step of round-off
        assert abs(fd - float(grad[i, a])) <= 1e-7 * (1.0 + abs(fd))


def test_rotation_translation_and_permutation_invariance():
    module, heads = _module(orbitals=True, fragments="none")
    _randomise(module, 4)
    pos, h, ei = cluster(6, heads.irreps_in.dim, 4)
    pos = pos.requires_grad_(True)
    e1, r1 = _energy(module, h, pos, ei, Z_MIX, 1.0)
    (g1,) = torch.autograd.grad(e1, pos)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(5)))
    if float(torch.det(q)) < 0:
        q[:, 0] = -q[:, 0]                                   # a proper rotation
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    inverse = torch.argsort(perm)
    pos2 = (pos.detach() @ q.T + torch.tensor([0.7, -1.1, 2.3]))[perm].requires_grad_(True)
    # the 1o / 2e feature blocks rotate with the frame (A_mu is built from 2e);
    # rotating positions alone changes the input (measured: dE = 3e-4 eV)
    h2 = (h @ heads.irreps_in.D_from_matrix(q).T)[perm]
    ei2 = neighbour_list(pos2.detach())
    e2, r2 = _energy(module, h2, pos2, ei2, Z_MIX[perm], 1.0)
    (g2,) = torch.autograd.grad(e2, pos2)
    assert abs(float(e1 - e2)) <= 1e-11 * (1.0 + abs(float(e1)))
    assert torch.allclose(g2[inverse], g1 @ q.T, atol=1e-10)
    assert torch.allclose(r2.q[inverse], r1.q, atol=1e-10)


@pytest.mark.parametrize("z,Q,expected", [
    (11, 1.0, ATOMIC_IONIZATION_EV[11]),
    (17, -1.0, -ATOMIC_ELECTRON_AFFINITY_EV[17]),
    (79, 1.0, ATOMIC_IONIZATION_EV[79]),
    (8, -1.0, -ATOMIC_ELECTRON_AFFINITY_EV[8]),
])
def test_an_isolated_ion_has_the_nist_energy(z, Q, expected):
    module, heads = _module()
    pos = torch.zeros(1, 3)
    h = torch.randn(1, heads.irreps_in.dim, generator=torch.Generator().manual_seed(1))
    ei = torch.zeros(2, 0, dtype=torch.long)
    energy, result = _energy(module, h, pos, ei, torch.tensor([z]), Q)
    assert abs(float(energy) - expected) <= 1e-12 * (1.0 + abs(expected))
    assert float(result.populations[0, 2 if Q > 0 else 0]) == 1.0


def test_an_isolated_neutral_atom_is_neutral_and_stays_on_its_ground_state():
    """c = 0 forced; the ensemble at c = 0 has entropy only: -kT ln 3 <= E <= 0."""
    module, heads = _module()
    pos = torch.zeros(1, 3)
    h = torch.randn(1, heads.irreps_in.dim, generator=torch.Generator().manual_seed(2))
    ei = torch.zeros(2, 0, dtype=torch.long)
    for z in (1, 11, 17, 79):
        energy, result = _energy(module, h, pos, ei, torch.tensor([z]), 0.0)
        assert -0.1 * math.log(3.0) <= float(energy) <= 0.0
        assert float(result.populations[0, 1]) > 0.99


def test_dead_zone_is_spec1_exactly():
    """T2: a large gap leaves no free charge; energy and forces are SPEC 1's."""
    module, heads = _module(anchor="gap", width="bound", gap=50.0)
    pos0, h, ei = cluster(6, heads.irreps_in.dim, 6)
    pos = pos0.clone().requires_grad_(True)
    energy, result = _energy(module, h, pos, ei, Z_MIX, 0.0)
    (grad,) = torch.autograd.grad(energy, pos)
    pos1 = pos0.clone().requires_grad_(True)
    vec, length = edges(pos1, ei)
    state = electrostatic_state(h, pos1, ei, vec, length, heads, ElectrostaticConfig())
    (grad_ref,) = torch.autograd.grad(state.energy, pos1)
    assert float(result.unit_charge.abs().max()) <= 1e-12
    assert abs(float(energy - state.energy)) <= 1e-9 * (1.0 + abs(float(state.energy)))
    assert torch.allclose(grad, grad_ref, atol=1e-9)


@pytest.mark.parametrize("R", [2.5, 6.0, 15.0, 40.0])
def test_na_cl_transfer_follows_the_two_site_law(R):
    """SPEC 25.4 (25.3) on the model's own W, b and state energies, at kT -> 0.

    delta_0 = clip(-(Ehat_Na,+ + Ehat_Cl,- + b_Na - b_Cl) / Jt, 0, 1), Jt = W_aa + W_bb - 2 W_ab,
    with |delta_T - delta_0| <= sqrt(4 kT ln 3 / Jt) (strong convexity).
    """
    kT = 0.002
    module, heads = _module(kT=kT, sic="none", fragments="none")   # the v1 law; certified: test_redox_sic.py
    pos = torch.tensor([[0.0, 0.0, 0.0], [R, 0.0, 0.0]])
    h = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(7))
    ei = neighbour_list(pos)
    energy, result = _energy(module, h, pos, ei, torch.tensor([11, 17]), 0.0)
    # Rebuild W and b exactly as the module does (no orbitals, lam = 0).
    vec, length = edges(pos, ei)
    assembly = assemble_neutral(h, pos, ei, vec, length, heads, module.config, species=torch.tensor([11, 17]))
    elim = eliminate_bound_sector(assembly, result.sigma_free)
    W, b = elim.W, elim.u0
    jt = float(W[0, 0] + W[1, 1] - 2 * W[0, 1])
    e_hat_na_plus = float(result.electronegativity[0] + 0.5 * result.gap[0])
    e_hat_cl_minus = float(-result.electronegativity[1] + 0.5 * result.gap[1])
    delta0 = min(max(-(e_hat_na_plus + e_hat_cl_minus + float(b[0] - b[1])) / jt, 0.0), 1.0)
    bound = math.sqrt(4.0 * kT * math.log(3.0) / jt)
    assert abs(float(result.unit_charge[0]) - delta0) <= bound
    assert abs(float(result.unit_charge.sum())) <= 1e-10


def test_saturated_total_charge_is_exact_and_differentiable():
    module, heads = _module(fragments="none")       # K-Na needs 4.29 A of neighbour coverage for 'bonds'
    pos0, h, ei = cluster(3, heads.irreps_in.dim, 8)
    z = torch.tensor([11, 19, 3])
    pos = pos0.clone().requires_grad_(True)
    energy, result = _energy(module, h, pos, ei, z, 3.0)
    (grad,) = torch.autograd.grad(energy, pos)
    assert torch.equal(result.unit_charge, torch.ones(3))
    assert bool(torch.isfinite(grad).all())
    with pytest.raises(ValueError):
        _energy(module, h, pos0, ei, z, 3.5)


@pytest.mark.parametrize("sic", ["none", "certified"])
def test_parameter_gradient_of_a_force_loss(sic):
    """d/dtheta of sum F^2: a second derivative through the dual Newton AND the bound Cholesky
    (and, certified, through W^-1 and the smooth absolute value of the SIC certificate)."""
    module, heads = _module(orbitals=True, sic=sic, fragments="none")
    _randomise(module, 9)
    pos0, h, ei = cluster(5, heads.irreps_in.dim, 9)
    z = torch.tensor([8, 1, 11, 17, 6])
    weight = module.charge_states.shift_mlp[-1].weight

    def loss():
        pos = pos0.clone().requires_grad_(True)
        energy, _ = _energy(module, h, pos, ei, z, 1.0)
        (grad,) = torch.autograd.grad(energy, pos, create_graph=True)
        return (grad ** 2).sum() + 0.1 * energy

    (g,) = torch.autograd.grad(loss(), weight)
    step = 1e-6
    for k in (0, 7, 20):
        with torch.no_grad():
            weight[0, k] += step
        up = float(loss())
        with torch.no_grad():
            weight[0, k] -= 2 * step
        down = float(loss())
        with torch.no_grad():
            weight[0, k] += step
        fd = (up - down) / (2 * step)
        assert abs(fd - float(g[0, k])) <= 1e-6 * (1.0 + abs(fd))


def test_refusals():
    module, heads = _module()
    pos, h, ei = cluster(4, heads.irreps_in.dim, 10)
    z = torch.tensor([8, 1, 1, 8])
    with pytest.raises(NotImplementedError):
        _energy(module, h, pos, ei, z, 0.0, fragment_id=torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError):
        vec, length = edges(pos, ei)
        module(h, pos, ei, vec, length, data={"total_charge": 0.0})        # no atomic numbers
    with pytest.raises(BoundSectorError, match="slab"):              # periodic itself: test_redox_periodic.py
        vec, length = edges(pos, ei)
        module(h, pos, ei, vec, length, pbc=(True, True, False), cell=torch.diag(torch.tensor([15.0, 15.0, 40.0])),
               edge_shift=torch.zeros(ei.shape[1], 3), data={"z": z, "total_charge": 1.0})
    big, _ = _module(self_interaction=0.95, sic="none", fragments="none")
    with pytest.raises(BoundSectorError, match="lam_max"):
        _energy(big, h, pos, ei, z, 0.0)
    assert module.take_solver_seed_loss() is None
    with pytest.raises(NotImplementedError):
        module.state()


def test_a_flashace_model_runs_with_trace_redox_attached():
    from flashace.model import TransformersACEV3
    from trace_electrostatics.heads import ElectrostaticHeads

    torch.manual_seed(23)
    model = TransformersACEV3(r_max=4.5, l_max=2, num_radial=4, hidden_dim=8, num_layers=1,
                              correlation_order=4, correlation_channels=4, attention_num_heads=2).eval()
    heads = ElectrostaticHeads(irreps_in=str(model.attention_irreps), r_max=3.5, hardness_margin=1.0)
    module = attach_redox(model, heads, ElectrostaticConfig(), RedoxConfig())
    assert model.electrostatics is module
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.4, 0.1, 0.0], [0.3, 1.9, 0.4], [1.9, 1.7, 1.3]])
    n = pos.shape[0]
    src, dst = zip(*[(s, r) for r in range(n) for s in range(n) if s != r])
    data = {"z": torch.tensor([11, 17, 8, 1]), "pos": pos,
            "edge_index": torch.tensor([src, dst]), "total_charge": 1.0}
    out = model(data)
    energy, forces = out[0], out[1]
    assert torch.isfinite(energy) and bool(torch.isfinite(forces).all())
    step = 1e-5
    for i, a in [(0, 0), (2, 2)]:
        plus, minus = dict(data), dict(data)
        plus["pos"], minus["pos"] = pos.clone(), pos.clone()
        plus["pos"][i, a] += step
        minus["pos"][i, a] -= step
        fd = -(float(model(plus)[0]) - float(model(minus)[0])) / (2 * step)
        assert abs(fd - float(forces[i, a])) <= 1e-6 * (1.0 + abs(fd))


@pytest.mark.parametrize("orbitals,lam,sic", [(False, 0.0, "none"), (True, 0.0, "none"), (True, 0.01, "none"),
                                              (False, 0.0, "certified"), (True, 0.0, "certified")])
def test_energy_decomposes_into_its_physical_terms(orbitals, lam, sic):
    """F = E_b0 + u0.f + 1/2 f.W.f - 1/2 sum (lam J_uu + r_u) c_u^2 + sum_u Phi_u(c_u),  f = Omega^T c.

    Phi_u from the state energies with the self-Coulomb (1 - lam) J_uu removed and
    the certified r_u added back to the populations.

    W, u0 rebuilt from the bound elimination on the ATOM free charges; Phi_u
    from the ensemble at the returned unit potentials.  A misplaced Omega (b, W
    or J_uu in the unit basis) breaks this identity while leaving forces
    consistent with the (wrong) energy.
    """
    module, heads = _module(orbitals=orbitals, self_interaction=lam, sic=sic, fragments="none")
    _randomise(module, 12)
    pos, h, ei = cluster(6, heads.irreps_in.dim, 12, scale=1.6)
    energy, r = _energy(module, h, pos, ei, Z_MIX, 1.0)
    vec, length = edges(pos, ei)
    assembly = assemble_neutral(h, pos, ei, vec, length, heads, module.config, species=Z_MIX)
    elim = eliminate_bound_sector(assembly, r.sigma_free)
    f, c = r.free_charge, r.unit_charge
    omega = torch.eye(6) if r.omega is None else r.omega
    J_self = torch.einsum("ui,ij,uj->u", omega, elim.J, omega)
    removal = torch.zeros(6) if r.sic_removal is None else r.sic_removal
    e_hat = (r.state_energies - 0.5 * (1.0 - lam) * J_self.unsqueeze(1) * (r.states ** 2).unsqueeze(0)
             + 0.5 * removal.unsqueeze(1) * (r.states ** 2).unsqueeze(0))
    c_check, phi = ensemble.conjugate_at(e_hat, r.states, r.nu, module.redox.kT)
    assert torch.allclose(c_check, c, atol=1e-12)
    assert torch.allclose(omega.T @ c, f, atol=1e-14)
    expected = (elim.e_b0 + elim.u0 @ f + 0.5 * f @ (elim.W @ f)
                - 0.5 * ((lam * J_self + removal) * c * c).sum() + phi.sum())
    assert abs(float(energy - expected)) <= 1e-10 * (1.0 + abs(float(energy)))


def test_smooth_switch_is_c2_at_both_ends():
    r = torch.tensor([0.0, 4.0 - 1e-12, 4.0, 4.0 + 1e-3], requires_grad=True)
    f = smooth_switch(r, 4.0)
    (d1,) = torch.autograd.grad(f.sum(), r, create_graph=True)
    (d2,) = torch.autograd.grad(d1.sum(), r)
    assert float(f[0]) == 1.0 and float(f[2]) == 0.0 and float(f[3]) == 0.0
    assert float(d1[0].abs()) == 0.0 and float(d2[0].abs()) == 0.0
    # exact values ~1e-25 just inside; the polynomial's cancellation leaves ~eps
    eps = torch.finfo(torch.float64).eps
    assert float(d1[1:].abs().max()) <= 64 * eps and float(d2[1:].abs().max()) <= 1e-10


def test_energy_is_continuous_when_a_neighbour_crosses_the_attention_cutoff():
    """An atom moved through r_cut = 4.0 A of the charge-state / orbital attention.

    Moving the atom changes E by -F dx, so continuity is tested by the second
    difference (O(dx^2 E'')) and by the force on either side (O(dx E'')): a
    jump or a kink at the cutoff would show in both.
    """
    module, heads = _module(orbitals=True, fragments="none")
    _randomise(module, 13, scale=1.0)
    base = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.4, 0.9, 0.0]])
    h = torch.randn(4, heads.irreps_in.dim, generator=torch.Generator().manual_seed(13))
    z = torch.tensor([8, 1, 1, 11])
    dx = 1e-7
    energies, forces = [], []
    for d in (-dx, 0.0, dx):
        pos = torch.cat((base, torch.tensor([[4.0 + d, 0.0, 0.0]]))).requires_grad_(True)
        ei = neighbour_list(pos.detach(), r_max=4.5)    # the model's list; the bond heads end at 3.5 A
        energy, _ = _energy(module, h, pos, ei, z, 1.0)
        (grad,) = torch.autograd.grad(energy, pos)
        energies.append(float(energy))
        forces.append(float(grad[3, 0]))
    assert abs(energies[2] - 2 * energies[1] + energies[0]) <= 1e-10
    assert abs(forces[2] - forces[0]) <= 1e-5


@pytest.mark.parametrize("spread_bias", [0.0, 8.0])
def test_orbitals_are_row_stochastic_and_invertible(spread_bias):
    """Rows sum to 1; the weight leaving the unit is <= s_max < 1/2 even with the spread saturated."""
    module, heads = _module(orbitals=True, fragments="none")
    _randomise(module, 14, scale=2.0)
    with torch.no_grad():
        module.orbitals.spread[-1].bias.fill_(spread_bias)   # sigmoid -> 1: the cap must hold
    pos, h, ei = cluster(7, heads.irreps_in.dim, 14)
    _, r = _energy(module, h, pos, ei, torch.tensor([8, 1, 1, 11, 17, 6, 1]), 0.0)
    omega, s_max = r.omega, module.redox.orbital_s_max
    assert torch.allclose(omega.sum(1), torch.ones(7), atol=1e-14)
    assert bool((omega >= 0).all())
    leaving = omega.sum(1) - torch.diagonal(omega)
    assert float(leaving.max()) <= s_max + 1e-15
    # ||diag(1-s)^-1 (Omega - diag(1-s))||_inf <= s/(1-s) < 1  =>  Omega invertible
    assert float(torch.linalg.svdvals(omega).min()) > 0.0


def test_a_failing_diagnostic_still_raises_the_refusal(monkeypatch):
    """When W - lam D is not positive definite the module reports lam_max; if that eigenvalue
    problem itself fails (an ill-conditioned W -- measured with indefinite compact-width blocks,
    2026-10-09), the refusal must still be the BoundSectorError, not a raw LinAlgError."""
    import trace_redox.module as module_code

    def failing(*args, **kwargs):
        raise torch.linalg.LinAlgError("eigh did not converge")

    monkeypatch.setattr(module_code, "lambda_max", failing)
    big, heads = _module(self_interaction=0.95, sic="none", fragments="none")
    pos, h, ei = cluster(4, heads.irreps_in.dim, 5)
    with pytest.raises(BoundSectorError, match="positive definite"):
        _energy(big, h, pos, ei, torch.tensor([8, 1, 1, 8]), 0.0)
