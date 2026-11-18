"""The periodic energy must converge to the open-boundary energy as the box grows.

This is a CROSS-MODULE test and that is the point: ``ewald.py`` owns the lattice
sum, ``energy.py`` owns the assembly, ``config.py`` owns the parameters, and each
of the three can be internally consistent while the three together are wrong.
Nothing inside any one module can see the failure.

The physics is not negotiable.  For a neutral cluster held fixed while the cell
grows, the periodic images recede and their interaction with the cluster must
fall to zero, so ``E_periodic(L) -> E_open`` monotonically.  If it does not, the
lattice sum is not converged, whatever any internal residual reports.

The bug this test was written for
---------------------------------
``ElectrostaticConfig.ewald_k_max`` was a static ``DEFAULT_EWALD_K_MAX = 8`` that
no cell could move, and ``ewald_params()`` never called ``EwaldParams.auto_tune``
-- which existed and was correct.  The number of wave vectors a fixed accuracy
needs grows linearly with the box, so the reciprocal sum silently under-converged
on anything larger than a small cell, and the error grew with L instead of
shrinking.  Measured at ``r_cut = 6 A``, ``accuracy = 1e-12``:

    L (A)             12        20        28        40        56
    k_max needed      18        30        42        59        83
    pinned at 8   -2.4e-05  -1.1e-03  +4.9e-03  -3.8e-03  -3.7e-02   eV
    auto-tuned    -1.8e-05  -3.7e-06  -1.4e-06  -4.6e-07  -1.7e-07   eV

At L = 56 A the pinned sum was wrong by 3.7e-02 eV -- about eight times the
cluster's entire electrostatic energy of 4.6e-03 eV -- with no warning.  A
production supercell is exactly where this bites, and exactly where nobody has
an open-boundary reference to notice.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import pytest
import torch

from conftest import require

R_CUT = 6.0
#: Five atoms, all pairs well inside ``R_CUT`` (largest separation 2.476 A), so
#: the open-boundary calculation loses nothing to the cutoff and the only thing
#: that changes across the scan is the periodic images.
CLUSTER = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.6, 0.0, 0.0],
        [0.0, 1.7, 0.0],
        [0.0, 0.0, 1.8],
        [1.1, 1.1, 1.1],
    ],
    dtype=torch.float64,
)


def _open_edges(positions):
    delta = positions.unsqueeze(1) - positions.unsqueeze(0)
    distance = delta.norm(dim=-1)
    mask = (distance < R_CUT) & (distance > 1.0e-9)
    return mask.nonzero().T, delta[mask], distance[mask]


def _periodic_edges(positions, cell):
    n_atoms = positions.shape[0]
    lengths = torch.diagonal(cell)
    reach = [int(math.ceil(R_CUT / float(length))) for length in lengths]
    senders, receivers, shifts = [], [], []
    for image in itertools.product(*[range(-r, r + 1) for r in reach]):
        offset = torch.tensor(image, dtype=torch.float64) @ cell
        # Package convention: edge_vec = pos[i] - pos[j] + edge_shift @ cell.
        # The distance must be measured with the SAME sign as the edge vector
        # the caller then builds, or every image edge carries the wrong length.
        distance = (
            positions.unsqueeze(1) - positions.unsqueeze(0) + offset
        ).norm(dim=-1)
        for i in range(n_atoms):
            for j in range(n_atoms):
                if image == (0, 0, 0) and i == j:
                    continue
                if float(distance[i, j]) < R_CUT:
                    senders.append(i)
                    receivers.append(j)
                    shifts.append(list(image))
    return (
        torch.tensor([senders, receivers], dtype=torch.long),
        torch.tensor(shifts, dtype=torch.float64),
    )


def _module(**overrides):
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    ElectrostaticModule = require("energy", "ElectrostaticModule")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")

    torch.manual_seed(0)
    heads = ElectrostaticHeads()
    config = dataclasses.replace(
        ElectrostaticConfig(),
        ewald_accuracy=1.0e-12,
        ewald_real_cutoff=R_CUT,
        **overrides,
    )
    torch.manual_seed(3)
    features = torch.randn(
        CLUSTER.shape[0], heads.irreps_in.dim, dtype=torch.float64
    ) * 2.0
    return ElectrostaticModule(heads, config), features


def _scan(module, features, lengths):
    out = []
    for length in lengths:
        cell = torch.eye(3, dtype=torch.float64) * length
        edge_index, shift = _periodic_edges(CLUSTER, cell)
        edge_vec = CLUSTER[edge_index[0]] - CLUSTER[edge_index[1]] + shift @ cell
        energy, _ = module(
            features,
            CLUSTER,
            edge_index,
            edge_vec,
            edge_vec.norm(dim=-1),
            cell=cell,
            pbc=(True, True, True),
            edge_shift=shift,
        )
        out.append(float(energy))
    return out


@pytest.mark.slow
def test_periodic_energy_converges_to_the_open_boundary_limit():
    """The headline claim, and the one the k_max bug violated."""
    module, features = _module()
    edge_index, edge_vec, edge_len = _open_edges(CLUSTER)
    reference = float(module(features, CLUSTER, edge_index, edge_vec, edge_len)[0])

    lengths = (12.0, 20.0, 28.0, 40.0, 56.0)
    errors = [abs(e - reference) for e in _scan(module, features, lengths)]

    for smaller, larger, a, b in zip(lengths, lengths[1:], errors, errors[1:]):
        assert b < a, (
            f"periodic energy moved AWAY from the open limit between L = "
            f"{smaller} and {larger} A ({a:.3e} -> {b:.3e} eV). The images "
            "recede as the box grows, so this must fall monotonically; it "
            "rising means the lattice sum is under-converged."
        )
    assert errors[-1] < 1.0e-6, (
        f"at L = {lengths[-1]} A the periodic energy is still {errors[-1]:.3e} eV "
        "from the open-boundary limit; recorded 1.7e-07"
    )


def test_k_max_is_derived_from_the_cell_not_pinned():
    """``ewald_k_max=None`` must grow the reciprocal grid with the box.

    A unit test on the parameter itself, so the cause is named even if the
    slow convergence test above is deselected.
    """
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    config = dataclasses.replace(
        ElectrostaticConfig(), ewald_accuracy=1.0e-12, ewald_real_cutoff=R_CUT
    )
    assert config.ewald_k_max is None, (
        "the default must be None ('derive from the cell'); a static default is "
        "the bug this test exists for"
    )

    chosen = [
        int(config.ewald_params(torch.eye(3, dtype=torch.float64) * length).k_max)
        for length in (12.0, 20.0, 28.0, 40.0, 56.0)
    ]
    assert chosen == sorted(chosen) and chosen[-1] > chosen[0], (
        f"k_max did not grow with the cell: {chosen}"
    )
    assert chosen[-1] >= 4 * chosen[0], (
        f"k_max grew only {chosen[0]} -> {chosen[-1]} over a 4.7x box; it must "
        "scale roughly linearly with L (recorded 18 -> 83)"
    )


def test_an_explicit_k_max_is_still_honoured():
    """Pinning must still work -- the regression suite relies on it."""
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    config = dataclasses.replace(ElectrostaticConfig(), ewald_k_max=20)
    for length in (12.0, 56.0):
        params = config.ewald_params(torch.eye(3, dtype=torch.float64) * length)
        assert int(params.k_max) == 20, (
            f"an explicitly pinned ewald_k_max=20 became {int(params.k_max)} at "
            f"L = {length} A; auto-tuning must not override an explicit value"
        )
