"""The e3nn conventions this package's physics silently depends on.

Every one of these is an assumption the Cartesian operators in ``heads.py`` are
built on.  None of them is checked by e3nn itself, and none of them would fail
loudly if a future e3nn changed it -- the network would keep training and quietly
return wrong tensors.  So they are asserted here rather than documented.

Two distinct hazards, and they are not the same hazard:

1. **Basis ordering.**  e3nn's ``1o`` components are conventionally described as
   ordered ``(y, z, x)``.  Measured, they are not: degree-1 spherical harmonics
   act as the identity on a cartesian vector and ``D_from_matrix(R)`` returns
   ``R`` itself, so index ``a`` of a ``1o`` quantity IS cartesian axis ``a`` and
   the Clebsch-Gordan bases need no relabelling.  A rescaling of a basis is
   absorbed by the learned weights in front of it; a PERMUTATION is not.  This
   is the assumption that would be most expensive to get wrong.

2. **Build dtype.**  ``o3.wigner_3j(...)`` builds at ``torch.get_default_dtype()``,
   which is ``float32`` unless the process globally changed it.  Writing
   ``o3.wigner_3j(2, 1, 1).to(torch.float64)`` therefore does NOT give a
   double-precision tensor: it widens the storage of a value that has already
   been rounded to float32, and the ~6e-09 error is permanent.  The bases must
   be built with ``dtype=torch.float64`` instead.  Measured on the quadrupole
   basis, the difference shows up in orthonormality: ``|W W^T - I|`` is 3.4e-08
   built-then-cast against 1.8e-15 built wide.  The trace and symmetry of that
   basis are exactly zero either way, because those are structural zeros that
   survive rounding -- which is precisely why this defect is invisible to the
   obvious checks and needs its own test.
"""

from __future__ import annotations

import math
import pathlib
import re

import pytest
import torch
from e3nn import o3

from conftest import require


# ---------------------------------------------------------------------------
# 1.  Basis ordering: a permutation here is not absorbable by learned weights
# ---------------------------------------------------------------------------
def test_the_l1_irrep_basis_is_cartesian_xyz_with_no_permutation():
    """``D_from_matrix(R) == R`` for ``1o``, over many random rotations.

    If e3nn ever reorders the ``1o`` basis this fails with the offending
    deviation, instead of every cartesian tensor in ``heads.py`` silently
    picking up an axis permutation.
    """
    torch.manual_seed(0)
    worst = 0.0
    for _ in range(500):
        rotation = o3.rand_matrix(dtype=torch.float64)
        wigner = o3.Irreps("1o").D_from_matrix(rotation)
        worst = max(worst, float((wigner - rotation).abs().max()))

    assert worst < 1.0e-6, (
        f"max |D(1o) - R| = {worst:.3e} over 500 rotations. Either the 1o basis "
        "is no longer cartesian xyz -- in which case every Clebsch-Gordan basis "
        "in heads.py needs relabelling and the learned weights CANNOT absorb it "
        "-- or D_from_matrix has lost precision (e3nn 0.6.0 builds its SO(3) "
        "generators at the default dtype, giving 7.3e-07 at float32 against "
        "1.7e-15 on 0.4.4)."
    )


def test_degree_one_spherical_harmonics_are_parallel_to_the_vector():
    """``Y_1(v)`` points along ``v``: the other half of the ordering claim.

    ``D_from_matrix`` could agree with ``R`` while the spherical harmonics fed
    into the network used a different order, so this is checked separately
    rather than inferred.
    """
    torch.manual_seed(1)
    for _ in range(64):
        vector = torch.randn(3, dtype=torch.float64)
        harmonic = o3.spherical_harmonics(
            "1o", vector, normalize=True, normalization="component"
        )
        deviation = float(
            (harmonic / harmonic.norm() - vector / vector.norm()).abs().max()
        )
        assert deviation < 1.0e-12, (
            f"Y_1(v) deviates from v by {deviation:.3e}; TRACE feeds plain "
            "cartesian edge_vec to o3.SphericalHarmonics and relies on index a "
            "being cartesian axis a"
        )


# ---------------------------------------------------------------------------
# 2.  Build dtype: the bases must be built wide, not cast wide
# ---------------------------------------------------------------------------
def test_wigner_3j_is_never_built_at_the_default_dtype():
    """Static guard: every ``o3.wigner_3j`` call in the package passes ``dtype``.

    A behavioural test alone cannot catch a reintroduction of this bug in a code
    path that happens not to be exercised, and the failure mode is a silent loss
    of seven digits rather than an exception -- so the call pattern itself is
    what is asserted.
    """
    package = pathlib.Path(__file__).resolve().parents[2] / "trace_electrostatics"
    call = re.compile(r"o3\.wigner_3j\(([^)]*)\)")

    offenders = []
    for source in sorted(package.glob("*.py")):
        for number, line in enumerate(source.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "``" in line:
                continue
            for match in call.finditer(line):
                if "dtype=" not in match.group(1):
                    offenders.append(f"{source.name}:{number}: {stripped}")

    assert not offenders, (
        "o3.wigner_3j built at torch.get_default_dtype() (float32) and cast "
        "afterwards -- the value is already rounded, so `.to(torch.float64)` "
        "only widens the storage. Pass dtype=torch.float64 to the call:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("degree", [2, 4])
def test_the_cartesian_bases_are_orthonormal_to_double_precision(degree):
    """``W W^T = I`` for the rank-2 bases, at 1e-14 rather than 1e-08.

    This is the property the float32 build actually damaged.  The trace and the
    symmetry of these bases are exactly zero in either build, so they are no
    evidence at all -- orthonormality is the one that moves.
    """
    source = (2, 1, 1) if degree == 2 else (4, 2, 2)
    scale = math.sqrt(5.0) if degree == 2 else 3.0
    basis = o3.wigner_3j(*source, dtype=torch.float64) * scale

    dimension = 2 * source[0] + 1
    flattened = basis.reshape(dimension, -1)
    gram = flattened @ flattened.T
    deviation = float((gram - torch.eye(dimension, dtype=torch.float64)).abs().max())

    assert deviation < 1.0e-14, (
        f"|W W^T - I| = {deviation:.3e} for the l={source[0]} basis; built at "
        "the float32 default this is 3.4e-08, built wide it is 1.8e-15"
    )


def test_the_package_quadrupole_basis_is_traceless_symmetric_and_orthonormal():
    """The basis ``heads.py`` actually builds, checked on all three properties."""
    quadrupole_cartesian_basis = require("heads", "quadrupole_cartesian_basis")
    basis = quadrupole_cartesian_basis().to(torch.float64)
    trace = float(torch.einsum("maa->m", basis).abs().max())
    asymmetry = float((basis - basis.transpose(-1, -2)).abs().max())
    dimension = basis.shape[0]
    gram = torch.einsum("mab,nab->mn", basis, basis)
    orthonormality = float(
        (gram - torch.eye(dimension, dtype=torch.float64)).abs().max()
    )

    assert trace < 1.0e-14, f"quadrupole basis has trace {trace:.3e}"
    assert asymmetry < 1.0e-14, f"quadrupole basis asymmetry {asymmetry:.3e}"
    assert orthonormality < 1.0e-14, (
        f"quadrupole basis |W W^T - I| = {orthonormality:.3e}"
    )


def test_the_package_octupole_basis_is_fully_symmetric_and_traceless():
    """The J=3 Zyss channel: symmetric in its last three axes and detraced.

    Without this the octupolar part of ``beta`` is not a pure ``3o``, and the
    zincblende hyperpolarizability it exists to capture leaks back into the
    dipolar channel it was separated from.
    """
    build = require("heads", "_octupole_cartesian_basis")
    basis = build(torch.float64, torch.device("cpu"))
    assert basis.shape == (7, 3, 3, 3), f"octupole basis shape {tuple(basis.shape)}"

    symmetry = max(
        float((basis - basis.permute(0, *perm)).abs().max())
        for perm in ((1, 3, 2), (2, 1, 3), (3, 2, 1))
    )
    detraced = float(torch.einsum("maab->mb", basis).abs().max())

    assert symmetry < 1.0e-14, (
        f"octupole basis is not fully symmetric: {symmetry:.3e}"
    )
    assert detraced < 1.0e-14, (
        f"octupole basis retains a trace of {detraced:.3e}; the J=1 part was "
        "supposed to be projected out"
    )
