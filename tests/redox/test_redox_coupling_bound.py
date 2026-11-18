"""A physical bound on the learned couplings (``RedoxConfig(coupling_bound='harrison')``; trace_redox SPEC 27).

Measured 2026-10-18 on step 13 seed 43 (epoch 19, frozen weights): relaxing the undoped wetting T5 reference, the flat
Au2 first goes ionic (free charges +-0.03 -> +-1.00, free energy -1.27 eV over 20 BFGS steps; forces exact against
finite differences), then one Au is driven into a surface Mg (1.30 A) by the learned coupling T(Mg, Au), which grows
from 0.35 to 9200 eV: the coupling head is an MLP of the invariants times a radial basis, unbounded, so the coherent
energy -|T| has no lower bound off the training distribution, and the SPEC 21 closure exceeds its cap.

A hopping integral is bounded.  Harrison's universal tight-binding matrix elements (ESTABLISHED: Harrison, Electronic
Structure and the Properties of Solids, 1980) are V = eta hbar^2 / (m d^2) with |eta| <= 3.24 (V_pp sigma), so

    B(r) = 3.24 hbar^2 / (m (r^2 + r0^2)),  r0 = 0.5 A,   tau = f(tau_raw),
    f(x) = x                                              for |x| <= a = 0.75 B,
         = sign(x) [a + (B - a) tanh((|x| - a) / (B - a))]  beyond             (NEW-TO-TRACE: the bound on learned tau)

C^2 at the knee (tanh''(0) = 0), never above B.  r0: Harrison's 1/d^2 diverges at coincidence while a hopping
integral between orthogonalised orbitals stays finite (bounded by the orbital's energy spread, tens of eV); measured,
with the bare 1/r^2 the seed-43 relaxation still ran the Au onto the Mg nucleus (r = 0.003 A, -2.2e6 eV = -B(r)).
r0 = 0.5 A gives B(0) = 99 eV and moves every data ratio by at most (1 + r0^2 / r^2): a coupling at 0.65 B crosses the
knee only below r = 1.27 A, shorter than any edge in the data.  A plain B tanh(x / B) would change the trained couplings by up to
12 %: measured, the step-12 couplings reach 0.64 B on the AuMgO data (1.78 eV at 2.98 A), so the identity below
0.75 B keeps every trained value exactly and changes only extrapolation.
"""
import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

HBAR2_OVER_ME = 7.619964


def _coupling(bound, scale):
    from trace_redox.heads import TransferCoupling
    torch.manual_seed(0)
    c = TransferCoupling(16, r_cut=4.0, bound=bound)
    with torch.no_grad():
        c.pair[-1].weight.normal_(0.0, scale, generator=torch.Generator().manual_seed(1))
    return c


def _edges(n=12, seed=2):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(n, 3, generator=g, dtype=torch.float64) * 5.0
    ei = neighbour_list(pos, r_max=4.0)
    _, length = edges(pos, ei)
    x = torch.randn(n, 16, generator=g, dtype=torch.float64) * 3.0
    return x, ei, length


def test_the_bounded_coupling_never_exceeds_harrison():
    x, ei, r = _edges()
    B = 3.24 * HBAR2_OVER_ME / (r ** 2 + 0.25)
    free = _coupling(None, 1e3).edge_values(x, ei, r)
    bounded = _coupling("harrison", 1e3).edge_values(x, ei, r)
    assert bool((free.abs() > B).any())                         # the defect: unbounded
    assert bool((bounded.abs() <= B * (1 + 1e-15)).all())


def test_it_is_the_identity_below_the_knee_and_c2_at_it():
    from trace_redox.heads import harrison_bound
    r = torch.full((2001,), 2.5, dtype=torch.float64)
    B = 3.24 * HBAR2_OVER_ME / (2.5 ** 2 + 0.25)
    x = torch.linspace(-2.0 * B, 2.0 * B, 2001, dtype=torch.float64)
    f = harrison_bound(x, r)
    inside = x.abs() <= 0.75 * B
    assert torch.equal(f[inside], x[inside])                     # bitwise: trained values untouched
    assert bool((f.abs() < B).all())
    a = 0.75 * B
    for sign in (1.0, -1.0):                                     # value, slope, curvature continuous at the knee
        h = 1e-4
        pts = torch.tensor([sign * a + k * h for k in (-2, -1, 0, 1, 2)], dtype=torch.float64)
        y = harrison_bound(pts, torch.full((5,), 2.5, dtype=torch.float64))
        d1_left, d1_right = (y[2] - y[1]) / h, (y[3] - y[2]) / h
        d2 = (y[3] - 2 * y[2] + y[1]) / h ** 2
        assert abs(float(d1_left - d1_right)) <= 1e-3 and abs(float(d2)) <= 1e-2


def test_it_is_validated():
    from trace_redox import RedoxConfig
    assert RedoxConfig().coupling_bound is None
    with pytest.raises(ValueError, match="coupling_bound"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", coupling_bound="harrison")   # no couplings
    with pytest.raises(ValueError, match="coupling_bound"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", couplings=True, coupling_bound="ohno")


def test_module_forces_are_minus_the_gradient_with_saturated_couplings():
    """Couplings driven into the tanh's nonlinear regime (raw ~ B): forces are still -grad E."""
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True,
                         coupling_bound="harrison")
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), config)
    with torch.no_grad():
        module.coupling.pair[-1].weight.normal_(0.0, 2.0, generator=torch.Generator().manual_seed(4))
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


def test_the_bound_is_finite_at_coincidence():
    from trace_redox.heads import harrison_bound
    r = torch.tensor([1e-6, 0.003, 0.5, 2.5], dtype=torch.float64)
    f = harrison_bound(torch.full((4,), 1e9, dtype=torch.float64), r)
    B0 = 3.24 * HBAR2_OVER_ME / 0.25
    assert bool(torch.isfinite(f).all()) and float(f.max()) < B0 and float(f[0]) > 0.99 * B0
