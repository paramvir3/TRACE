"""Damped Coulomb kernel and Thole damping -- SPEC sections 1.1, 1.6 and 6."""

import math

import numpy as np
import pytest
import torch

import elec_reference as ref
from conftest import require


K_E = 14.399645  # SPEC 0, quoted not imported


def test_constants_match_the_spec_unit_system():
    """SPEC 0: K_E = 14.399645 eV A / e^2 is the single source of truth."""
    package_k_e = require("constants", "K_E")
    assert float(package_k_e) == K_E  # SPEC 0, exact literal


def test_kernel_self_is_exactly_two_k_e_over_sigma_root_pi():
    """SPEC 1.1 / 6: ``J(0) == 2 K_E / (sigma sqrt(pi))``, required exact."""
    kernel_self = require("kernels", "kernel_self")
    sigma = torch.tensor([0.5, 1.0, 1.5, 2.7, 4.0])
    expected = 2.0 * K_E / (sigma * math.sqrt(math.pi))
    # SPEC section 6 lists this one as "exact", so it is compared bit for bit.
    assert torch.equal(kernel_self(sigma), expected)


def test_hardness_floor_at_sigma_1p5_is_10p832_ev():
    """SPEC 1.2: the floor is ~10.832 eV at sigma = 1.5 A."""
    kernel_self = require("kernels", "kernel_self")
    floor = float(kernel_self(torch.tensor(1.5)))
    # SPEC 1.2 quotes "~10.832 eV"; the exact value of 2 K_E / (1.5 sqrt(pi))
    # is 10.8321729543807..., so three decimals is the right comparison.
    assert round(floor, 3) == 10.832
    # SPEC 1.2 table row 1: Mulliken eta_C = 9.997 eV sits BELOW this floor.
    assert 9.997 < floor


def test_combine_sigma_is_the_quadrature_sum():
    """SPEC 1.1: ``sigma_ij = sqrt(sigma_i^2 + sigma_j^2)``."""
    combine_sigma = require("kernels", "combine_sigma")
    sigma_i = torch.tensor([0.7, 1.5, 2.0])
    sigma_j = torch.tensor([1.1, 1.5, 0.3])
    expected = torch.sqrt(sigma_i**2 + sigma_j**2)
    torch.testing.assert_close(combine_sigma(sigma_i, sigma_j), expected, rtol=0, atol=0)


def test_kernel_approaches_j_zero_at_short_range():
    """SPEC 1.1: J(r) -> J(0) = 2 K_E / (sigma sqrt(pi)) as r -> 0."""
    coulomb_kernel, kernel_self = require("kernels", "coulomb_kernel", "kernel_self")
    sigma = torch.tensor(1.5)
    limit = kernel_self(sigma)

    assert float(coulomb_kernel(torch.tensor(0.0), sigma)) == float(limit)

    # Series expansion (derived here from erf(x) = (2/sqrt(pi))(x - x^3/3 + x^5/10)):
    #   J(r) = (2 K_E / (sigma sqrt(pi))) (1 - r^2/(3 sigma^2) + r^4/(10 sigma^4) - ...)
    # The first neglected term is r^6/(42 sigma^6), which is 2e-15 relative at
    # r = 1e-2, sigma = 1.5 -- below the 1e-13 bound used here.
    for r in (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2):
        value = float(coulomb_kernel(torch.tensor(r), sigma))
        x2 = r**2 / 1.5**2
        predicted = float(limit) * (1.0 - x2 / 3.0 + x2**2 / 10.0)
        assert abs(value - predicted) < 1.0e-13 * float(limit)


def test_kernel_gradient_is_finite_at_and_near_the_origin():
    """A ``torch.where`` that masks a NaN still leaks a NaN gradient.

    SPEC 1.1 fixes the value at r = 0; the *derivative* is what a naive
    ``torch.where(r > 0, K_E erf(r/s)/r, J0)`` gets wrong, because the untaken
    branch still evaluates 0/0 and 0 * inf = NaN reaches ``r.grad``.  The
    analytic derivative is dJ/dr = 0 at r = 0 (the expansion above is even in r).
    """
    coulomb_kernel = require("kernels", "coulomb_kernel")
    sigma = torch.full((6,), 1.5)
    r = torch.tensor([0.0, 1.0e-14, 1.0e-10, 1.0e-6, 1.0e-3, 1.0], requires_grad=True)
    value = coulomb_kernel(r, sigma)
    assert torch.isfinite(value).all()
    value.sum().backward()
    assert r.grad is not None
    assert torch.isfinite(r.grad).all(), f"non-finite kernel gradient: {r.grad}"

    # Differentiating the series above term by term gives
    #   dJ/dr = -4 K_E r / (3 sigma^3 sqrt(pi)) + O(r^3),
    # which is exactly 0 at r = 0 and -3.2095e-06 eV/A^2 at r = 1e-6, sigma = 1.5.
    assert float(r.grad[0]) == 0.0
    slope = -4.0 * K_E / (3.0 * 1.5**3 * math.sqrt(math.pi))
    for index, radius in ((1, 1.0e-14), (2, 1.0e-10), (3, 1.0e-6)):
        assert float(r.grad[index]) == pytest.approx(slope * radius, rel=1e-6, abs=1e-20)

    # And the value at r = 0 must not itself be NaN-producing under a second
    # derivative path either: the same masking bug shows up there too.
    r2 = torch.zeros(3, requires_grad=True)
    out = coulomb_kernel(r2, torch.full((3,), 1.5))
    (grad,) = torch.autograd.grad(out.sum(), r2, create_graph=True)
    assert torch.isfinite(grad).all()


def test_kernel_tends_to_bare_coulomb_at_long_range():
    """SPEC 1.1: erf(r/sigma) -> 1, so J(r) -> K_E / r."""
    coulomb_kernel = require("kernels", "coulomb_kernel")
    sigma = torch.tensor(1.5)
    for r in (20.0, 50.0, 200.0):
        value = float(coulomb_kernel(torch.tensor(r), sigma))
        # erf(20/1.5) = erf(13.3) differs from 1 by < 1e-78 in exact arithmetic,
        # so in float64 the two are indistinguishable.
        torch.testing.assert_close(
            torch.tensor(value), torch.tensor(K_E / r), rtol=1e-15, atol=0.0
        )


def test_the_long_range_deviation_at_six_angstrom_is_the_erfc_tail():
    """SPEC 6, verbatim: "at r=6, sigma=1.5 the deviation is **3.700051e-08** =
    K_E*erfc(4)/6.  That IS the erfc tail; 1e-10 is first reached at
    **r = 6.975 A**, and 8.3e-14 by r=8.  An implementation hitting 1e-10 at
    r=6 would not be the erf kernel".

    This is a two-sided regression and both sides matter:

      * an implementation that deviates by MORE than the erfc tail is wrong;
      * an implementation that deviates by LESS is *also* wrong -- it is not
        the erf kernel at all, and SPEC 1.1 forbids every other form.

    So the deviation is pinned to the analytic tail, not merely bounded.  At
    r = 6 and sigma = 1.5, r/sigma = 4 exactly, and

        K_E / r - K_E erf(4) / r = K_E erfc(4) / 6 = 3.700051e-08 eV.
    """
    coulomb_kernel = require("kernels", "coulomb_kernel")
    from scipy.special import erfc

    sigma = 1.5
    for r, quoted in ((6.0, 3.700051e-08), (8.0, 8.3e-14)):
        value = float(coulomb_kernel(torch.tensor(r), torch.tensor(sigma)))
        deviation = K_E / r - value
        analytic = K_E * float(erfc(r / sigma)) / r
        # Pinned to the analytic tail to 1e-6 relative: this is an identity, so
        # the only slack is float64 evaluation of erf/erfc near its tail.
        assert deviation == pytest.approx(analytic, rel=1.0e-6), (
            f"r={r}: deviation {deviation:.6e}, erfc tail {analytic:.6e}"
        )
        # ... and matches the number SPEC section 6 quotes, to its own digits.
        assert deviation == pytest.approx(quoted, rel=2.0e-6)

    # The trap SPEC names explicitly: 1e-10 is NOT reached at r = 6.
    at_six = K_E / 6.0 - float(coulomb_kernel(torch.tensor(6.0), torch.tensor(sigma)))
    assert at_six > 1.0e-10, (
        f"deviation at r=6 is {at_six:.3e}, below the 1e-10 that SPEC says "
        "cannot be reached until r = 6.975 A -- this is not the erf kernel"
    )


def test_the_one_e_minus_ten_crossing_is_at_six_point_nine_seven_five():
    """SPEC 6: "1e-10 is first reached at **r = 6.975 A**".

    Bracketed rather than asserted at a point, because "first reached" is a
    statement about a crossing: the deviation must still exceed 1e-10 just
    below 6.975 A and be under it just above.  Solving
    K_E erfc(r/1.5) / r = 1e-10 gives r = 6.97455 A, which SPEC rounds to 6.975.
    """
    coulomb_kernel = require("kernels", "coulomb_kernel")
    sigma = torch.tensor(1.5)

    def deviation(r):
        return K_E / r - float(coulomb_kernel(torch.tensor(r), sigma))

    assert deviation(6.90) > 1.0e-10
    assert deviation(7.05) < 1.0e-10
    # The crossing itself, to the three decimals SPEC quotes.
    low, high = 6.5, 7.5
    for _ in range(60):
        middle = 0.5 * (low + high)
        if deviation(middle) > 1.0e-10:
            low = middle
        else:
            high = middle
    assert round(0.5 * (low + high), 3) == 6.975, (
        f"1e-10 crossing at r = {0.5 * (low + high):.6f} A, SPEC says 6.975"
    )


def test_kernel_is_the_gaussian_erf_form_not_ohno_klopman():
    """SPEC 1.1: "Gaussian/erf ONLY", the Ohno/Klopman 1/sqrt(r^2+s^2) is forbidden.

    Checked numerically rather than by reading the source: at r = 1 A with
    sigma = 1.5 A the two forms differ by more than 10%, so agreement with the
    erf reference and disagreement with Ohno pins the functional form.
    """
    coulomb_kernel = require("kernels", "coulomb_kernel")
    r = np.array([0.5, 1.0, 2.0, 3.0])
    sigma = 1.5
    value = coulomb_kernel(torch.tensor(r), torch.tensor(sigma)).numpy()
    erf_form = ref.coulomb_kernel_ref(r, sigma)
    ohno = K_E / np.sqrt(r**2 + sigma**2)
    np.testing.assert_allclose(value, erf_form, rtol=1e-14, atol=0.0)
    assert np.abs(value - ohno).max() > 0.1  # the two forms are far apart


def test_kernel_is_monotonically_decreasing_and_bounded_by_j_zero():
    """Consequence of SPEC 1.1: erf(x)/x decreases on (0, inf)."""
    coulomb_kernel, kernel_self = require("kernels", "coulomb_kernel", "kernel_self")
    sigma = torch.tensor(1.5)
    r = torch.linspace(0.0, 12.0, 400)
    value = coulomb_kernel(r, sigma.expand_as(r))
    assert torch.all(value[1:] < value[:-1])
    assert float(value.max()) <= float(kernel_self(sigma))
    assert torch.all(value > 0.0)


# ---------------------------------------------------------------------------
# Thole damping -- SPEC 1.6 and SPEC 6 ("Thole a >= 1 raises")
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a", [1.0, 1.0000001, 1.5, 2.0, 10.0])
def test_thole_a_at_or_above_one_is_rejected(a):
    """SPEC 1.6 / 6: ``a < 1`` strictly; at a = 1.0 "the relay catastrophes"."""
    thole_damping = require("kernels", "thole_damping")
    r = torch.tensor([1.2, 2.5])
    alpha = torch.tensor([1.0, 4.0])
    with pytest.raises((ValueError, AssertionError)):
        thole_damping(r, alpha, alpha, a=a)


@pytest.mark.parametrize("a", [0.39, 0.5, 0.9, 0.999])
def test_thole_a_below_one_is_accepted(a):
    """SPEC 1.6: the admissible range is the open interval (0, 1)."""
    thole_damping = require("kernels", "thole_damping")
    r = torch.tensor([1.2, 2.5])
    alpha = torch.tensor([1.0, 4.0])
    lam3, lam5 = thole_damping(r, alpha, alpha, a=a)
    assert torch.isfinite(lam3).all() and torch.isfinite(lam5).all()


def test_thole_default_is_the_spec_recommended_value():
    """SPEC 1.6: "Recommended a = 0.39 (2.5x margin)"."""
    default_a, a_max = require("kernels", "DEFAULT_THOLE_A", "THOLE_A_MAX")
    assert float(default_a) == 0.39
    assert float(a_max) <= 1.0
    # 2.5x margin: 1 / 0.39 = 2.564
    assert 1.0 / float(default_a) > 2.5


def test_thole_damping_is_symmetric_under_exchange_of_the_two_atoms():
    """Reciprocity (SPEC 4 / 6) fails at the source if the damping is not symmetric."""
    thole_damping = require("kernels", "thole_damping")
    r = torch.tensor([1.2, 2.0, 3.5])
    alpha_i = torch.tensor([0.3, 1.6, 11.0])
    alpha_j = torch.tensor([11.0, 0.3, 1.6])
    forward = thole_damping(r, alpha_i, alpha_j)
    reverse = thole_damping(r, alpha_j, alpha_i)
    for a, b in zip(forward, reverse):
        torch.testing.assert_close(a, b, rtol=0, atol=0)  # exchange symmetry, exact


def test_the_kernel_at_a_real_close_contact_is_well_below_j_zero(close_contact_pair):
    """1.2 A is the separation SPEC section 6 names as the one where a Thole
    bug becomes visible, and SPEC 1.2 names 1.533 A as the distance below which
    a pair at eta = 8 eV goes non-convex.  Both statements are about the same
    quantity: how close ``J(r)`` has come to its ``r -> 0`` ceiling.

    At r = 1.2 A with sigma_i = sigma_j = 1.5 A (so sigma_ij = 2.1213 A),
    J = K_E erf(0.56569) / 1.2 = 6.9153 eV, against J(0) = 7.6595 eV: already
    90% of the ceiling.  That is why the hardness floor binds at chemical bond
    lengths rather than only at pathological ones.
    """
    coulomb_kernel, combine_sigma, kernel_self = require(
        "kernels", "coulomb_kernel", "combine_sigma", "kernel_self"
    )
    separation = float((close_contact_pair[0] - close_contact_pair[1]).norm())
    assert separation == pytest.approx(1.2, abs=1e-12)

    sigma_ij = combine_sigma(torch.tensor(1.5), torch.tensor(1.5))
    assert float(sigma_ij) == pytest.approx(1.5 * math.sqrt(2.0), rel=1e-15)

    value = float(coulomb_kernel(torch.tensor(separation), sigma_ij))
    ceiling = float(kernel_self(sigma_ij))
    assert value == pytest.approx(6.9153, abs=1.0e-3)
    assert ceiling == pytest.approx(7.6595, abs=1.0e-3)
    assert value / ceiling > 0.9, (
        f"J(1.2 A) / J(0) = {value / ceiling:.4f}; the kernel is not saturating "
        "at chemical contact, so the hardness floor would not bind"
    )


def test_thole_damping_has_finite_gradients_at_contact():
    """The relay is evaluated inside an autograd graph; r -> 0 must not NaN."""
    thole_damping = require("kernels", "thole_damping")
    r = torch.tensor([1.0e-12, 1.0e-6, 0.5, 1.2], requires_grad=True)
    alpha = torch.tensor([0.3, 1.6, 11.0, 2.0])
    lam3, lam5 = thole_damping(r, alpha, alpha)
    (lam3.sum() + lam5.sum()).backward()
    assert torch.isfinite(r.grad).all(), f"non-finite Thole gradient: {r.grad}"
