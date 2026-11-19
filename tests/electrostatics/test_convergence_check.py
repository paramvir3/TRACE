"""``check_convergence``: making the fixed-``k`` truncation visible.

LEGACY MODE.  Since SPEC 17 the default energy is the converged minimum and
has no truncation gap; everything below is about ``solve_mode="fixed_k"``
(with the kernel and envelope it was measured with), which is kept as an
opt-in and whose truncation hazard is exactly what these tests document.

The energy this package reports is ``U(m_k)`` after exactly ``n_cg_iter``
conjugate-gradient steps, and SPEC 2 is explicit that this is a *definition*
rather than an approximation -- the forces are the exact gradient of it, which
is what makes them conservative.  Nothing here disputes that.

What these tests pin down is the part the definition leaves implicit: ``U(m_k)``
is not ``U(m_inf)``, and the two do not differ by a constant: the eigenvalue
spread of K widens with N, so the gap grows with system size. (Not a hop limit:
K is dense through the Coulomb kernel -- see SPEC 2.0.)  Measured on
jittered simple-cubic clusters at 2.5 A spacing with untrained heads and
``n_cg_iter = 6``, against a 400-step reference:

    N atoms        8        27        64       125
    |dU|/atom  7.0e-05   8.1e-04   1.6e-03   4.0e-03  eV

That is the transferability hazard: heads fitted at one system size learn to
absorb one truncation error, and the compensation does not carry to another.

It is NOT ill-conditioning, and the distinction matters because the two have
different fixes.  Over the same range the smallest eigenvalue of the constrained
Hessian on the variational ``(t, mu_hat)`` sector stays healthy (8.1e-01 down to
6.5e-01, never negative), and diagonal preconditioning changes the condition
number by a factor of 0.94 to 1.07 -- i.e. not at all.  More iterations is the
fix; a preconditioner is not.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import pytest
import torch

from conftest import require


def _cluster(n_atoms, spacing=2.5, seed=5):
    """``n_atoms`` on a jittered simple-cubic lattice -- physical separations."""
    side = math.ceil(n_atoms ** (1.0 / 3.0))
    points = [
        [i, j, k]
        for i in range(side)
        for j in range(side)
        for k in range(side)
    ][:n_atoms]
    torch.manual_seed(seed)
    grid = torch.tensor(points, dtype=torch.float64) * spacing
    return grid + (torch.rand(n_atoms, 3, dtype=torch.float64) - 0.5) * 0.25 * spacing


def _solve(n_atoms, n_iter, check=False, spacing=2.5):
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    ElectrostaticModule = require("energy", "ElectrostaticModule")
    ElectrostaticConfig = require("config", "ElectrostaticConfig")

    torch.manual_seed(0)
    heads = ElectrostaticHeads(envelope_onset=None)
    positions = _cluster(n_atoms, spacing)
    torch.manual_seed(5)
    features = torch.randn(n_atoms, heads.irreps_in.dim, dtype=torch.float64) * 2.0

    delta = positions.unsqueeze(1) - positions.unsqueeze(0)
    distance = delta.norm(dim=-1)
    mask = (distance < 8.0) & (distance > 1.0e-9)
    edge_index = mask.nonzero().T

    config = dataclasses.replace(
        ElectrostaticConfig(solve_mode="fixed_k", multipole_kernel="thole"),
        n_cg_iter=n_iter,
        check_convergence=check,
    )
    module = ElectrostaticModule(heads, config)
    return module(features, positions, edge_index, delta[mask], distance[mask])


def test_the_truncation_gap_grows_with_system_size():
    """The claim the warning exists for, asserted rather than asserted-about.

    If this ever stops holding -- if six steps became enough at 125 atoms -- the
    warning and its documentation would be describing a hazard that is no longer
    there, and this test is what would say so.
    """
    per_atom = {}
    for n_atoms in (8, 64):
        truncated, _ = _solve(n_atoms, 6)
        converged, _ = _solve(n_atoms, 400)
        per_atom[n_atoms] = abs(float(truncated) - float(converged)) / n_atoms

    assert per_atom[8] < 1.0e-3, (
        f"at 8 atoms the k=6 truncation is {per_atom[8]:.2e} eV/atom; recorded "
        "7.0e-05, i.e. small systems are nearly converged at the default"
    )
    assert per_atom[64] > 5.0 * per_atom[8], (
        f"truncation per atom went {per_atom[8]:.2e} -> {per_atom[64]:.2e} eV "
        "from 8 to 64 atoms; it is supposed to GROW with system size, which is "
        "the whole reason check_convergence exists"
    )


def test_more_iterations_close_the_gap():
    """The fix is iterations, so the gap must fall monotonically in ``k``."""
    reference = float(_solve(64, 400)[0])
    gaps = [abs(float(_solve(64, k)[0]) - reference) for k in (6, 12, 50)]

    assert gaps[0] > gaps[1] > gaps[2], (
        f"gaps at k=6,12,50 are {gaps[0]:.2e}, {gaps[1]:.2e}, {gaps[2]:.2e}; "
        "they must fall monotonically"
    )
    assert gaps[2] < 0.01 * gaps[0], (
        f"k=50 leaves {gaps[2]:.2e} eV against {gaps[0]:.2e} at k=6; fifty "
        "steps are supposed to recover essentially all of it"
    )


def test_check_convergence_warns_on_a_truncated_solve():
    """Opt-in, and a warning rather than an exception."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _solve(125, 6, check=True)
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]

    assert runtime, "a 125-atom solve at k=6 is supposed to warn"
    message = str(runtime[0].message)
    assert "far from equilibrium" in message
    assert "n_cg_iter=6" in message, f"the warning must name k; got: {message}"


def test_check_convergence_is_silent_on_a_converged_solve():
    """No warning when the solve really did equilibrate -- and none when off."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _solve(125, 400, check=True)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)], (
        "a converged solve must not warn"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _solve(125, 6, check=False)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)], (
        "check_convergence=False must stay silent even when truncated -- it is "
        "opt-in because the check costs a host synchronisation"
    )


def test_the_default_config_leaves_the_check_off():
    """Off by default: a truncated solve is the SPEC 2 definition, not an error."""
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    config = ElectrostaticConfig()
    assert config.check_convergence is False
    assert config.convergence_tol == pytest.approx(1.0e-2)
