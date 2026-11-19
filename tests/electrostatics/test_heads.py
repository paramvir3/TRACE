"""Readout heads -- SPEC section 3.

Two failure modes from the SPEC are silent, which is why they get tests:

  * a parity-mismatched ``o3.Linear`` constructs with zero instructions, returns
    identically zero, and emits no warning -- so a mis-declared head trains to
    nothing;
  * ``torch.set_default_dtype(torch.float64)`` after the first e3nn module is
    constructed is a no-op, and the model is then quietly wrong at 1e-8.
"""

import math

import pytest
import torch

from conftest import require, require_any


K_E = 14.399645  # SPEC 0


def _floor(sigma):
    """J(0) = 2 K_E / (sigma sqrt(pi))   (SPEC 1.2)."""
    return 2.0 * K_E / (sigma * math.sqrt(math.pi))


def test_default_dtype_is_float64_before_any_e3nn_module_exists():
    """SPEC 3: ``torch.set_default_dtype(torch.float64)`` BEFORE constructing any
    e3nn module.  conftest.py does this on its third executable line."""
    assert torch.get_default_dtype() is torch.float64


# ---------------------------------------------------------------------------
# SPEC 3: the parity-mismatch trap
# ---------------------------------------------------------------------------
def test_e3nn_parity_mismatch_really_is_silent():
    """Document the underlying e3nn behaviour the SPEC guard exists for.

    This asserts the *bug*, not the fix: ``o3.Linear("1o", "1e")`` builds
    happily, has zero instructions, and returns zeros.  If e3nn ever starts
    raising on its own, this test fails and the guard below can be reconsidered.
    """
    o3 = pytest.importorskip("e3nn.o3")
    linear = o3.Linear("1o", "1e")
    assert len(linear.instructions) == 0
    output = linear(torch.randn(4, 3))
    assert torch.equal(output, torch.zeros_like(output))


def test_checked_linear_raises_on_a_parity_mismatch():
    """SPEC 3: ``assert len(lin.instructions) > 0`` on EVERY equivariant head."""
    checked = require_any("heads", "checked_o3_linear", "_checked_o3_linear")
    with pytest.raises((ValueError, AssertionError)):
        checked("1o", "1e", "parity_mismatch")
    with pytest.raises((ValueError, AssertionError)):
        checked("64x0e", "1x2e", "no_l2_input")


def test_a_one_instruction_head_can_still_return_an_identically_zero_block():
    """SPEC 3: "NECESSARY BUT NOT SUFFICIENT: also require every OUTPUT irrep to
    be reachable.  A head declaring "0e + 2e" from a 0e-only input has ONE
    instruction and still returns an identically zero 2e block."

    This documents the e3nn behaviour, so the assertion in the next test is
    known to be guarding something real rather than restating
    ``len(instructions) > 0``.
    """
    o3 = pytest.importorskip("e3nn.o3")
    linear = o3.Linear("64x0e", "1x0e + 1x2e")
    # One instruction -- so the SPEC's first, weaker guard passes.
    assert len(linear.instructions) == 1
    output = linear(torch.randn(4, 64))
    assert output.shape[1] == 6  # 1 scalar + 5 components of 2e
    # ... and the 2e block is identically zero, for every input, forever.
    assert float(output[:, 1:].abs().max()) == 0.0


def test_the_reachability_check_is_stronger_than_a_nonzero_instruction_count():
    """SPEC 3: "This is what rejects l_max = 1 features, where the anisotropic
    part of A_mu would silently be dead."

    Three cases, of increasing subtlety:

      * ``1o -> 1e``  : parity mismatch, zero instructions (the weak guard
        catches this one);
      * ``64x0e -> 0e + 2e``: ONE instruction, dead 2e block -- the weak guard
        does NOT catch this, and it is the shape of every real A_mu head;
      * ``64x0e + 32x1o -> 0e + 2e``: l_max = 1 features.  An ``o3.Linear``
        cannot couple ``1o (x) 1o`` into 2e, so the anisotropic part of A_mu is
        dead here too -- which is exactly the l_max = 1 rejection SPEC names.
    """
    checked = require_any("heads", "checked_o3_linear", "_checked_o3_linear")
    o3 = pytest.importorskip("e3nn.o3")

    weak_guard_would_catch = o3.Linear("1o", "1e")
    assert len(weak_guard_would_catch.instructions) == 0
    with pytest.raises((ValueError, AssertionError)):
        checked("1o", "1e", "parity_mismatch")

    # The cases the weak guard misses.
    for irreps_in, irreps_out, label in (
        ("64x0e", "1x0e + 1x2e", "scalar_only_input"),
        ("64x0e + 32x1o", "1x0e + 1x2e", "l_max_1_features"),
    ):
        survives_weak_guard = o3.Linear(irreps_in, irreps_out)
        assert len(survives_weak_guard.instructions) > 0, (
            f"{label}: this case is only interesting if the weak guard passes"
        )
        with pytest.raises((ValueError, AssertionError)) as excinfo:
            checked(irreps_in, irreps_out, label)
        # The message must name the unreachable block, not just say "bad head".
        assert "2e" in str(excinfo.value)


def test_checked_linear_accepts_a_well_formed_head():
    checked = require_any("heads", "checked_o3_linear", "_checked_o3_linear")
    node_irreps = require("heads", "TRACE_NODE_IRREPS")
    linear = checked(node_irreps, "1x0e", "scalar_head")
    assert len(linear.instructions) > 0


def test_head_construction_rejects_input_irreps_that_cannot_feed_a_mu():
    """A_mu_i is 0e + 2e, the symmetric part of 1o (x) 1o (SPEC 3).

    Input irreps with no 1o give a 2e head with zero instructions -- the exact
    silent failure SPEC 3 names -- so construction must fail loudly.
    """
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    with pytest.raises((ValueError, AssertionError)):
        ElectrostaticHeads(irreps_in="64x0e")


# ---------------------------------------------------------------------------
# SPEC 1.2 / 3: the eta head
# ---------------------------------------------------------------------------
def _heads_and_input(seed, n_atoms=5):
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    torch.manual_seed(seed)
    heads = ElectrostaticHeads()
    features = torch.randn(n_atoms, heads.irreps_in.dim) * 3.0
    bond_index = torch.tensor(
        [list(range(n_atoms - 1)), list(range(1, n_atoms))], dtype=torch.long
    )
    bond_length = torch.full((n_atoms - 1,), 1.5)
    return heads, features, bond_index, bond_length


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 123])
def test_eta_head_never_emits_a_value_below_j_zero(seed):
    """SPEC 1.2 / 17.3: the eta head emits ``softplus(raw) + S_0(sigma) + margin``.

    ``S_0 = sqrt(2) K_E / (sigma sqrt(pi))`` is the tight Gram bound; the head
    guarantees ``eta - S_0 >= hardness_margin`` (1 eV/e^2 by default), which is
    the smallest eigenvalue of the charge block for every geometry.  Checked
    behaviourally over random initialisations and deliberately large features,
    so a raw linear head cannot pass by luck.
    """
    heads, features, bond_index, bond_length = _heads_and_input(seed)
    readout = heads(features, bond_index, bond_length)
    params = readout.params
    floor = math.sqrt(2.0) * K_E / (params.sigma * math.sqrt(math.pi))
    margin = params.eta - floor
    assert torch.all(margin >= heads.hardness_margin), (
        f"eta - S_0 = {float(margin.min()):.3e} eV, below the head's margin "
        f"{heads.hardness_margin}"
    )


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_head_outputs_pass_the_runtime_assertions(seed):
    """The heads and the diagnostics must agree: SPEC 2.1 items 1 and 2."""
    assert_hardness_floor, assert_positive_kappa = require(
        "diagnostics", "assert_hardness_floor", "assert_positive_kappa"
    )
    heads, features, bond_index, bond_length = _heads_and_input(seed)
    params = heads(features, bond_index, bond_length).params
    assert_hardness_floor(params.eta, params.sigma)
    assert_positive_kappa(params.kappa)
    assert torch.all(params.sigma > 0.0)
    assert torch.all(params.ell > 0.0)


def test_kappa_is_bitwise_identical_when_the_bond_is_listed_backwards():
    """SPEC 1.4: kappa must be symmetric; ``0.5 (s_ij + s_ji)`` is exactly
    commutative in IEEE-754, so this is an exact comparison, not a tolerance."""
    heads, features, bond_index, bond_length = _heads_and_input(5)
    forward = heads(features, bond_index, bond_length).params.kappa
    reversed_index = bond_index.flip(0)
    backward = heads(features, reversed_index, bond_length).params.kappa
    assert torch.equal(forward, backward)


def test_solver_seed_changes_sign_when_the_bond_is_listed_backwards():
    """SPEC 2: the seed is the antisymmetrised logit, so it transforms like t_ij."""
    heads, features, bond_index, bond_length = _heads_and_input(5)
    forward = heads(features, bond_index, bond_length).t_seed
    backward = heads(features, bond_index.flip(0), bond_length).t_seed
    torch.testing.assert_close(forward, -backward, rtol=0, atol=1e-15)


def test_seed_is_geometry_determined_and_reproducible():
    """SPEC 2: "Never seed from the previous MD step".

    A seed that depended on history would make two identical calls differ.
    """
    heads, features, bond_index, bond_length = _heads_and_input(9)
    first = heads(features, bond_index, bond_length).t_seed
    second = heads(features, bond_index, bond_length).t_seed
    assert torch.equal(first, second)


def test_symmetrise_and_antisymmetrise_split_the_logit_exactly():
    """s_ij = sym + anti and s_ji = sym - anti, to the last bit."""
    symmetrise, antisymmetrise = require(
        "heads", "symmetrise_bond_logits", "antisymmetrise_bond_logits"
    )
    forward = torch.tensor([1.25, -3.5, 0.0, 7.75])
    reverse = torch.tensor([-0.75, 2.5, 4.0, 7.75])
    symmetric = symmetrise(forward, reverse)
    antisymmetric = antisymmetrise(forward, reverse)
    torch.testing.assert_close(symmetric + antisymmetric, forward, rtol=0, atol=0)
    torch.testing.assert_close(symmetric - antisymmetric, reverse, rtol=0, atol=0)
    assert torch.equal(symmetrise(forward, reverse), symmetrise(reverse, forward))


def test_bond_hardness_is_softplus_plus_a_strictly_positive_floor():
    """SPEC 1.4: ``kappa_ij = softplus(0.5 (s_ij + s_ji)) + kappa_0``, kappa_0 > 0."""
    bond_hardness = require("heads", "bond_hardness")
    logits = torch.tensor([-50.0, -5.0, 0.0, 5.0, 50.0])
    kappa = bond_hardness(logits, logits, kappa_floor=0.1)
    expected = torch.nn.functional.softplus(logits) + 0.1
    torch.testing.assert_close(kappa, expected, rtol=0, atol=1e-15)
    assert torch.all(kappa >= 0.1)
    with pytest.raises(ValueError):
        bond_hardness(logits, logits, kappa_floor=0.0)


def test_dipole_stiffness_is_symmetric_and_rotation_equivariant():
    """SPEC 3: A_mu is 0e + 2e, i.e. the symmetric part of 1o (x) 1o.

    Equivariance is checked against e3nn's own Wigner D matrices on the input
    irreps, so it tests the head, not the test.
    """
    o3 = pytest.importorskip("e3nn.o3")
    heads, features, _, _ = _heads_and_input(4)
    stiffness = heads.dipole_stiffness(features)
    assert stiffness.shape == (features.shape[0], 3, 3)
    torch.testing.assert_close(
        stiffness, stiffness.transpose(-1, -2), rtol=0, atol=1e-13
    )

    rotation = o3.rand_matrix()
    wigner = heads.irreps_in.D_from_matrix(rotation)
    rotated = heads.dipole_stiffness(features @ wigner.T)
    expected = rotation @ stiffness @ rotation.T
    torch.testing.assert_close(rotated, expected, rtol=1e-10, atol=1e-10)


def test_scalar_readouts_are_rotation_invariant():
    """chi, eta, sigma, ell are 0e (SPEC 3): a rotation must not move them."""
    o3 = pytest.importorskip("e3nn.o3")
    heads, features, bond_index, bond_length = _heads_and_input(6)
    base = heads(features, bond_index, bond_length).params
    wigner = heads.irreps_in.D_from_matrix(o3.rand_matrix())
    rotated = heads(features @ wigner.T, bond_index, bond_length).params
    for name in ("chi", "eta", "sigma", "ell", "kappa"):
        torch.testing.assert_close(
            getattr(rotated, name), getattr(base, name), rtol=1e-11, atol=1e-11
        )
