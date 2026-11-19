"""Harrison-anchored couplings for open-shell frames (``RedoxConfig(coupling_anchor='harrison')``; trace_redox SPEC 31).

Measured 2026-10-27 (experiments/benchmarks_4g README; theory note 2026-10-18 section 4c):
- PBE's relaxation at Au2's O-top contact is -1.98 eV (28 ions); the trained model's is -0.22 / -0.14 eV
  (probe_t5_first_order.py, step 17), and 83 % of PBE's charge transfer there is O -> Au (almo_split.py --direction).
- The model's O -> Au configuration (Au- + an O hole) carries a learned coupling of 0.1-1.0 eV against Harrison's
  universal V_sp sigma(2.26 A) = 2.75 eV (probe_otop_channels.py): F is even in tau, so a coupling to a configuration
  eV above the frame gets a vanishing gradient and stays where it started (initial scale 0.01).
- Couplings no data constrain are a seed lottery: step 20 seed 42 learned O-O 28 eV (ten times Harrison's largest
  universal coefficient), seed 43 0.37 eV; IP -268 eV at charge +1.  Anchoring only the Au pairs (a first rule:
  open-shell frames) let the opened O -> Au channel's hole delocalise through those O-O couplings: wet_u +541 ->
  +4990 meV in seed 42 on frozen weights (+735 -> +1090 in seed 43; probe_anchor_counterfactual.py, 2026-10-27).

Construction (NEW-TO-TRACE; Harrison's universal two-centre coefficients ESTABLISHED: Harrison, Electronic Structure
and the Properties of Solids (1980); Froyen and Harrison, Phys. Rev. B 20, 2420 (1979)):
    every pair is anchored except a cation and an anion of the formal frame (f_u f_v < 0).  Those transfers cross
    the ionic gap: their second-order energy is the frame's covalency, smooth and state-independent, and stays in
    the backbone and the moments (SPEC 22).  Every other transfer moves a carrier: a hole between anions, an
    electron between cations, an electron to or from a neutral frame (Au0).  Then
        tau = eta(l_u, l_v) hbar^2 / (m (r^2 + r0^2)) exp(c tanh(x / c)) f(r),   c = ln 2,
    x the head's learned radial sum, f the C^2 quintic switch on the window [r_cut - 1 A, r_cut] (1 below it: a
    full-range switch, as the learned head uses, is 0.38 at the 2.26 A Au-O contact and 0.26 at the Au-Au bond, and
    took 60-75 % off Harrison's values in a first version; measured 2026-10-27); eta_ss = -1.40, eta_sp = +1.84 (sigma: the one combination
    of the p hole a single partner couples to), eta_pp = (3.24 + 2 (-0.81)) / 3 (the trace average of the p-p block:
    a hole in a lattice cannot be sigma to every neighbour).  The frontier shell l: s for groups 1, 2, 11, 12 and H,
    p for groups 13-18; d- and f-block partners of an anchored pair are refused.
    The cation-anion pairs are learned and bounded by SPEC 27.
"""
import math

import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

HBAR2_OVER_ME = 7.619964


def _harrison(eta, r):
    return eta * HBAR2_OVER_ME / (r * r + 0.25)


def _head(scale=None):
    from trace_redox.heads import TransferCoupling
    torch.manual_seed(0)
    c = TransferCoupling(16, r_cut=4.0, anchor="harrison")
    with torch.no_grad():
        if scale is None:
            c.pair[-1].weight.zero_()
        else:
            c.pair[-1].weight.normal_(0.0, scale, generator=torch.Generator().manual_seed(1))
    return c


def _system(n_extra=0):
    """Au2 upright over an O, with Mg and O neighbours (A); edges within 4 A."""
    z = torch.tensor([79, 79, 8, 12, 12, 8] + [8] * n_extra)
    pos = torch.tensor([[0.0, 0.0, 2.26], [0.0, 0.0, 4.80], [0.0, 0.0, 0.0], [2.13, 0.0, 0.0], [0.0, 2.13, 0.0],
                        [2.13, 2.13, 0.0]], dtype=torch.float64)
    ei = neighbour_list(pos, r_max=4.0)
    _, length = edges(pos, ei)
    x = torch.randn(len(z), 16, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    return z, pos, ei, length, x


def test_the_frontier_shell_and_the_anchored_pairs():
    from trace_redox.heads import anchored_pairs, frontier_shell
    assert [frontier_shell(z) for z in (1, 3, 8, 12, 13, 47, 79, 30)] == [0, 0, 1, 0, 1, 0, 0, 0]
    for z in (21, 26, 28, 57, 64):
        assert frontier_shell(z) is None                          # d and f blocks: not anchored
    z = torch.tensor([79, 8, 12, 13, 11, 17])
    pairs = [(0, 1), (0, 0), (0, 2), (1, 1), (2, 2), (2, 3), (1, 2), (1, 3), (4, 5), (5, 5), (4, 4)]
    src, dst = torch.tensor([p[0] for p in pairs]), torch.tensor([p[1] for p in pairs])
    #           Au-O  Au-Au Au-Mg O-O   Mg-Mg Mg-Al O-Mg   O-Al   Na-Cl  Cl-Cl Na-Na
    expected = [True, True, True, True, True, True, False, False, False, True, True]
    assert anchored_pairs(z, src, dst).tolist() == expected


def _window(r, r_cut=4.0, width=1.0):
    t = ((r - (r_cut - width)) / width).clamp(0.0, 1.0)
    return 1.0 - t ** 3 * (10.0 - 15.0 * t + 6.0 * t * t)


def test_an_anchored_pair_starts_at_harrison():
    """Zero learned weights: anchored pairs are Harrison's value (the window switch is 1 below 3 A); O-Mg is zero."""
    z, pos, ei, r, x = _system()
    tau = _head().edge_values(x, ei, r, species=z)
    src, dst = ei
    f = _window(r)
    assert bool((f[r <= 3.0] == 1.0).all())
    eta_pp = (3.24 - 2 * 0.81) / 3
    for a, b, eta in ((79, 8, 1.84), (79, 79, -1.40), (79, 12, -1.40), (8, 8, eta_pp), (12, 12, -1.40)):
        m = (z[src] == a) & (z[dst] == b)
        assert bool(m.any())
        assert torch.allclose(tau[m], _harrison(eta, r[m]) * f[m], rtol=1e-12, atol=0)
    m = (z[src] == 8) & (z[dst] == 12)
    assert bool(m.any()) and torch.equal(tau[m], torch.zeros_like(tau[m]))


def test_the_learned_factor_is_bounded_and_the_rest_by_spec_27():
    from trace_redox.heads import harrison_bound
    z, pos, ei, r, x = _system()
    head = _head(scale=50.0)
    tau = head.edge_values(x, ei, r, species=z)
    src, dst = ei
    from trace_redox.constants import formal_frame
    f = formal_frame(z)
    anchored = f[src] * f[dst] >= 0
    n_p = (z[src] == 8).to(torch.long) + (z[dst] == 8).to(torch.long)
    eta = torch.where(n_p == 0, -1.40, torch.where(n_p == 1, 1.84, (3.24 - 2 * 0.81) / 3)).to(torch.float64)
    base = _harrison(eta, r) * _window(r)
    ratio = tau[anchored] / base[anchored]
    assert bool((ratio > 0.5 - 1e-12).all()) and bool((ratio < 2.0 + 1e-12).all())
    assert float(ratio.max()) > 1.5 and float(ratio.min()) < 0.6        # the factor is used, both ways
    B = 3.24 * HBAR2_OVER_ME / (r ** 2 + 0.25)
    assert bool((tau[~anchored].abs() <= B[~anchored] * (1 + 1e-15)).all())     # tanh rounds to 1 at saturation
    raw = head.raw_values(x, ei, r)
    assert torch.equal(tau[~anchored], harrison_bound(raw, r)[~anchored])


def test_the_coupling_matrix_is_symmetric_and_needs_the_species():
    z, pos, ei, r, x = _system()
    head = _head(scale=0.3)
    T = head(x, ei, r, species=z)
    assert torch.equal(T, T.transpose(0, 1))
    with pytest.raises(ValueError, match="species"):
        head(x, ei, r)


def test_a_d_block_partner_of_an_anchored_pair_is_refused():
    from trace_redox.heads import TransferCoupling
    head = TransferCoupling(16, r_cut=4.0, anchor="harrison")
    z = torch.tensor([79, 26])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.4]], dtype=torch.float64)
    ei = neighbour_list(pos, r_max=4.0)
    _, r = edges(pos, ei)
    x = torch.zeros(2, 16, dtype=torch.float64)
    with pytest.raises(ValueError, match="frontier"):
        head.edge_values(x, ei, r, species=z)


def test_off_by_default_unchanged_and_validated():
    from trace_redox import RedoxConfig
    from trace_redox.heads import TransferCoupling
    assert RedoxConfig().coupling_anchor is None
    with pytest.raises(ValueError, match="coupling_anchor"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", coupling_anchor="harrison")      # no couplings
    with pytest.raises(ValueError, match="coupling_anchor"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", couplings=True, coupling_anchor="slater")
    z, pos, ei, r, x = _system()
    torch.manual_seed(0)
    plain = TransferCoupling(16, r_cut=4.0)
    torch.manual_seed(0)
    same = TransferCoupling(16, r_cut=4.0, anchor=None)
    assert torch.equal(plain(x, ei, r), same(x, ei, r, species=z))     # anchor off: the species change nothing


def test_module_forces_are_minus_the_gradient_with_anchored_couplings():
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True,
                         coupling_anchor="harrison")
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), config)
    with torch.no_grad():
        module.coupling.pair[-1].weight.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(4))
    z = torch.tensor([8, 79, 79, 12, 13, 8])
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0],
                         [2.0, 2.1, 0.2]], dtype=torch.float64)
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos0)

    def energy(p):
        vec, length = edges(p, ei)
        return module(h, p, ei, vec, length, data={"z": z, "total_charge": 0.0})[0]

    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    for i, axis in [(1, 2), (2, 2), (3, 0), (0, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-6 * (1.0 + abs(fd))


def test_the_module_routes_the_species_to_the_anchor():
    """The module's coupling matrix on the Au-O edge equals the anchored head's value (zero learned weights)."""
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    import trace_redox.module as M
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True,
                         coupling_anchor="harrison")
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), config)
    with torch.no_grad():
        module.coupling.pair[-1].weight.zero_()
    z = torch.tensor([8, 79, 12])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.26], [2.13, 0.0, 0.0]], dtype=torch.float64)
    h = torch.randn(3, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    seen = {}
    inner = M.RedoxElectrostatics._diabatic_solution

    def capture(self, e_hat, states, b, W, L, K, Q, T=None, **kw):
        seen["T"] = T
        return inner(self, e_hat, states, b, W, L, K, Q, T=T, **kw)

    M.RedoxElectrostatics._diabatic_solution = capture
    try:
        module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})
    finally:
        M.RedoxElectrostatics._diabatic_solution = inner
    r = torch.tensor(2.26, dtype=torch.float64)
    expected = _harrison(1.84, r) * _window(r, float(config.coupling_cutoff))
    assert abs(float(seen["T"][0, 1]) - float(expected)) <= 1e-12 * float(expected)
    assert math.isfinite(float(seen["T"][0, 2])) and float(seen["T"][0, 2]) == 0.0      # O-Mg: closed frames
