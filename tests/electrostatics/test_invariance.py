"""Full O(3) invariance of the electrostatic energy, and Newton's third law.

SPEC 3.1 is the reason this module is separate from ``test_heads.py``.  The
energy is a function of *two* arguments that transform under a rotation: the
positions, and the TRACE node features ``h``.  ``h`` carries 1o and 2e blocks,
so it must be transformed with the Wigner D matrix of its own irreps:

    pos -> pos R^T ,    h -> h D(R)^T ,   D = irreps_in.D_from_matrix(R)

Rotating the positions **alone** is a test bug, not a test.  It changes the
answer, and a suite that does it measures the size of its own mistake rather
than the equivariance of the model.  Both halves are asserted below: the correct
transformation must leave the energy invariant to round-off, and the incorrect
one must visibly *not*.

SPEC 3.1 also records that in e3nn 0.4.4 ``o3.Irrep("1o").D_from_matrix(R)``
returns ``R`` itself, so no cartesian relabelling ("(y,z,x) ordering") is
correct here -- applying it gives 0.78 relative error.  That claim is checked
directly, because the whole module rests on it.
"""

import math

import pytest
import torch

from conftest import complete_edge_index, require


# A proper rotation with no special structure: Euler angles chosen so that no
# entry of R is 0, 1 or 1/sqrt(2), and det R = +1 to machine precision.
def _rotation(alpha=0.7, beta=1.3, gamma=-0.45):
    def rz(t):
        c, s = math.cos(t), math.sin(t)
        return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def ry(t):
        c, s = math.cos(t), math.sin(t)
        return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    return rz(alpha) @ ry(beta) @ rz(gamma)


def _energy_fn(n_atoms, seed, total_charge=0.0):
    """Return ``(run, positions_free)``: an energy closure over (pos, h)."""
    electrostatic_energy = require("energy", "electrostatic_energy")
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")

    torch.manual_seed(seed)
    heads = ElectrostaticHeads()
    features = torch.randn(n_atoms, heads.irreps_in.dim) * 2.0
    config = ElectrostaticConfig()
    edge_index = complete_edge_index(n_atoms)

    def run(pos, h):
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        return electrostatic_energy(
            h,
            pos,
            edge_index,
            edge_vec,
            edge_vec.norm(dim=-1),
            heads,
            config,
            total_charge=total_charge,
        )

    return run, heads, features


POSITIONS = torch.tensor(
    [
        [0.00, 0.00, 0.00],
        [1.19, 0.00, 0.00],
        [0.37, 1.00, 0.53],
        [1.55, 0.88, -0.61],
    ]
)


# ---------------------------------------------------------------------------
# SPEC 3.1: the basis convention this module depends on
# ---------------------------------------------------------------------------
def test_degree_one_wigner_matrix_is_the_rotation_itself():
    """SPEC 3.1: "``o3.Irrep("1o").D_from_matrix(R)`` returns ``R`` itself".

    The widely repeated "(y,z,x) ordering" folklore is WRONG here; applying it
    gives 0.78 relative equivariance error.

    The IDENTITY is exact on both supported e3nn series; what differs is the
    precision with which e3nn builds the matrix, and the tolerance has to cover
    the whole supported range rather than the series that happened to be
    installed when the number was first recorded:

        e3nn 0.4.4   max |D - R| = 6.9e-18   reads a float64 ``_Jd`` cache
        e3nn 0.6.0   max |D - R| = 3.0e-15   rebuilds via ``so3_generators``

    0.6.0's ``wigner_D`` calls ``so3_generators(l)`` with no dtype, so the
    generators are built at ``torch.get_default_dtype()`` and exponentiated;
    under this suite's float64 default that costs about three decimal digits
    against the cached matrices (and at a float32 default it would be 7.3e-07 --
    see the Conventions section of ``heads.py``).  It is a precision regression
    in e3nn, not a change of convention, and ``D_from_matrix`` is used only by
    tests -- never in this package's forward path -- so it costs accuracy in the
    harness and nothing in the physics.

    1e-13 is float64 slack above the worse of the two, and still eleven orders
    below the 0.1 at which the folklore relabelling below would show up.
    """
    o3 = pytest.importorskip("e3nn.o3")
    rotation = _rotation()
    wigner = o3.Irrep("1o").D_from_matrix(rotation)
    assert float((wigner - rotation).abs().max()) < 1.0e-13

    # And the folklore relabelling is a different matrix, by a lot -- so it
    # cannot be dismissed as an equivalent convention.
    relabel = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    folklore = relabel.T @ rotation @ relabel
    assert float((folklore - rotation).abs().max()) > 0.1


# ---------------------------------------------------------------------------
# SPEC: the energy is an O(3) scalar
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "angles",
    [(0.7, 1.3, -0.45), (2.1, 0.6, 1.9), (-1.4, 2.7, 0.33)],
)
def test_energy_is_invariant_under_a_proper_rotation(angles):
    """U is 0e: rotating positions AND features must leave it unchanged.

    Measured on this test bed: the two energies agree bitwise (|dU| = 0.0).
    The bound below is 1e-12 eV -- generous next to the observed exact
    agreement, and still three orders tighter than the ~1e-06 that a wrong
    basis convention produces.
    """
    run, heads, features = _energy_fn(4, seed=8, total_charge=1.0)
    rotation = _rotation(*angles)
    wigner = heads.irreps_in.D_from_matrix(rotation)

    base, _ = run(POSITIONS, features)
    rotated, _ = run(POSITIONS @ rotation.T, features @ wigner.T)
    assert abs(float(rotated - base)) < 1.0e-12, (
        f"|dU| = {abs(float(rotated - base)):.3e} eV under a proper rotation"
    )


def test_rotating_positions_without_the_features_is_a_test_bug():
    """The control that makes the test above mean something.

    SPEC 3.1's failure mode is a suite that rotates ``pos`` and leaves ``h``
    alone.  ``h`` carries 1o and 2e blocks, so that is a *different physical
    system*, and the energy genuinely changes.  A suite that made this mistake
    would be measuring its own error (SPEC's audit saw ~1e-06; on the random
    features used here it is 1.9e-01) and calling it equivariance.

    This test asserts the bug, so that if the energy ever did become invariant
    to a positions-only rotation -- which would mean the 1o and 2e blocks of
    ``h`` had stopped reaching the readout at all -- the suite notices.
    """
    run, heads, features = _energy_fn(4, seed=8, total_charge=1.0)
    rotation = _rotation()
    base, _ = run(POSITIONS, features)
    positions_only, _ = run(POSITIONS @ rotation.T, features)

    correct, _ = run(POSITIONS @ rotation.T, features @ heads.irreps_in.D_from_matrix(rotation).T)
    bug = abs(float(positions_only - base))
    right = abs(float(correct - base))
    assert bug > 1.0e-6, (
        f"a positions-only rotation changed U by only {bug:.3e} eV; the 1o/2e "
        "blocks of h are not reaching the readout"
    )
    assert bug > 1.0e6 * max(right, 1.0e-16)


def test_energy_is_invariant_under_inversion():
    """Parity: r -> -r with h -> h D(-I)^T.  D(-I) is +1 on even irreps and
    -1 on odd ones, so this exercises the parity bookkeeping that a
    parity-mismatched ``o3.Linear`` (SPEC 3) would silently break."""
    run, heads, features = _energy_fn(4, seed=8, total_charge=1.0)
    inversion = -torch.eye(3)
    parity = heads.irreps_in.D_from_matrix(inversion)
    base, _ = run(POSITIONS, features)
    flipped, _ = run(-POSITIONS, features @ parity.T)
    assert abs(float(flipped - base)) < 1.0e-12


def test_energy_is_invariant_under_improper_rotations():
    """O(3), not just SO(3): every improper element is -R for a proper R."""
    run, heads, features = _energy_fn(4, seed=8, total_charge=1.0)
    base, _ = run(POSITIONS, features)
    for angles in ((0.7, 1.3, -0.45), (2.1, 0.6, 1.9)):
        improper = -_rotation(*angles)
        assert float(torch.det(improper)) < 0.0  # genuinely improper
        wigner = heads.irreps_in.D_from_matrix(improper)
        moved, _ = run(POSITIONS @ improper.T, features @ wigner.T)
        assert abs(float(moved - base)) < 1.0e-12


def test_energy_is_invariant_under_translation():
    """Translations do not act on ``h`` at all, so the features stay put.

    This is the one invariance that the *edge* construction is responsible for:
    it holds because the energy is built from ``edge_vec``, never from absolute
    positions.
    """
    run, _, features = _energy_fn(4, seed=8, total_charge=1.0)
    base, _ = run(POSITIONS, features)
    for shift in ([2.3, -1.1, 0.4], [-100.0, 250.0, 17.5]):
        moved, _ = run(POSITIONS + torch.tensor(shift), features)
        assert abs(float(moved - base)) < 1.0e-12, f"translation by {shift}"


def test_charges_and_bond_variables_are_rotation_invariant_scalars():
    """q and t are 0e.  A rotation must not move them either -- if it did, the
    scalar heads would be picking up an orientation from somewhere."""
    run, heads, features = _energy_fn(4, seed=2, total_charge=1.0)
    rotation = _rotation()
    wigner = heads.irreps_in.D_from_matrix(rotation)
    _, base = run(POSITIONS, features)
    _, rotated = run(POSITIONS @ rotation.T, features @ wigner.T)
    torch.testing.assert_close(rotated.q, base.q, rtol=0, atol=1e-12)
    torch.testing.assert_close(rotated.t, base.t, rtol=0, atol=1e-12)


def test_dipoles_rotate_as_vectors():
    """mu is 1o: under R it must become R mu, atom by atom."""
    run, heads, features = _energy_fn(4, seed=2, total_charge=1.0)
    rotation = _rotation()
    wigner = heads.irreps_in.D_from_matrix(rotation)
    _, base = run(POSITIONS, features)
    _, rotated = run(POSITIONS @ rotation.T, features @ wigner.T)
    torch.testing.assert_close(
        rotated.mu, base.mu @ rotation.T, rtol=1e-10, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Newton's third law
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("total_charge", [0.0, 1.0, -2.0])
def test_forces_sum_to_zero(total_charge):
    """sum_i F_i = 0.

    This is translation invariance differentiated: U(pos + u) = U(pos) for every
    u, so sum_i dU/dr_i = 0 identically.  It is an exact identity, and the bound
    is the float64 accumulation budget for a sum of 4 x 3 terms of order
    1 eV/A -- measured 1.1e-16 here, asserted at 1e-12 to leave room for a
    larger or stiffer system.
    """
    run, _, features = _energy_fn(4, seed=8, total_charge=total_charge)
    x = POSITIONS.clone().requires_grad_(True)
    energy, _ = run(x, features)
    (gradient,) = torch.autograd.grad(energy, x)
    forces = -gradient
    residual = float(forces.sum(dim=0).abs().max())
    assert residual < 1.0e-12, f"sum_i F_i = {forces.sum(dim=0).tolist()}"


def test_forces_rotate_as_vectors():
    """F is 1o: rotating the system must rotate the forces, not scramble them.

    Weaker than the energy test on its own, but it catches a class of bug the
    energy cannot see -- a gradient assembled in the wrong frame can still
    integrate to an invariant scalar.
    """
    run, heads, features = _energy_fn(4, seed=8, total_charge=1.0)
    rotation = _rotation()
    wigner = heads.irreps_in.D_from_matrix(rotation)

    def forces(pos, h):
        x = pos.clone().requires_grad_(True)
        energy, _ = run(x, h)
        return -torch.autograd.grad(energy, x)[0].detach()

    base = forces(POSITIONS, features)
    rotated = forces(POSITIONS @ rotation.T, features @ wigner.T)
    torch.testing.assert_close(rotated, base @ rotation.T, rtol=1e-9, atol=1e-11)
