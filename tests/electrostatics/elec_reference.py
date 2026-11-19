"""Independent reference implementations for the electrostatics regression suite.

Nothing in this file imports ``trace_electrostatics``.  Everything here is
written from the formulas in SPEC.md (or from textbook electrostatics) so that
a test can compare the package against a second, independently coded answer
rather than against itself.

That independence is the point of, in particular:

  * :func:`global_qeq` -- the reference for the SPEC 1.4 ``kappa -> 0`` test,
    which must not call the package's own solver on both sides.
  * :func:`ewald_reference` -- a direct real-plus-reciprocal Ewald sum used to
    establish that the NaCl test geometry really does carry the Madelung
    constant before the package is asked to reproduce it.
  * :func:`fixed_k_cg` -- k steps of conjugate gradient, used to show that
    differentiating *through* the iterations and Hellmann-Feynman give
    measurably different gradients at small k (SPEC 2).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch

#: SPEC section 0.  Quoted, not imported, so a change to the package constant
#: is a test failure rather than a silent agreement.
K_E = 14.399645

SQRT_PI = math.sqrt(math.pi)


# ---------------------------------------------------------------------------
# Kernels (SPEC 1.1)
# ---------------------------------------------------------------------------
def combine_sigma_ref(sigma_i, sigma_j):
    """sigma_ij = sqrt(sigma_i^2 + sigma_j^2)   (SPEC 1.1)."""
    return np.sqrt(np.asarray(sigma_i) ** 2 + np.asarray(sigma_j) ** 2)


def kernel_self_ref(sigma):
    """J(0) = 2 K_E / (sigma sqrt(pi))   (SPEC 1.1)."""
    return 2.0 * K_E / (np.asarray(sigma, dtype=float) * SQRT_PI)


def coulomb_kernel_ref(r, sigma_ij):
    """J(r) = K_E erf(r / sigma_ij) / r, with the r -> 0 limit filled in."""
    from scipy.special import erf

    r = np.asarray(r, dtype=float)
    sigma_ij = np.asarray(sigma_ij, dtype=float)
    out = np.empty(np.broadcast(r, sigma_ij).shape, dtype=float)
    small = r < 1.0e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        value = K_E * erf(r / sigma_ij) / r
    limit = kernel_self_ref(sigma_ij)
    out = np.where(small, limit, value)
    return out


# ---------------------------------------------------------------------------
# Global charge equilibration (the reference for SPEC 1.4 / 1.3)
# ---------------------------------------------------------------------------
def coulomb_matrix(positions, sigma, include_self=False):
    """Off-diagonal damped Coulomb matrix J_ij, zero diagonal by default.

    The diagonal of the charge Hessian is the hardness eta_i (SPEC 1.2), which
    already stands in for the self interaction, so the reference keeps J_ii = 0
    and adds eta on the diagonal separately.
    """
    positions = np.asarray(positions, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    delta = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(delta, axis=-1)
    sigma_ij = combine_sigma_ref(sigma[:, None], sigma[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = coulomb_kernel_ref(r, sigma_ij)
    if not include_self:
        np.fill_diagonal(matrix, 0.0)
    return matrix


def global_qeq(eta, chi, coulomb, groups, targets):
    """Constrained charge equilibration, solved as a dense KKT system.

    Minimises ``0.5 q^T (J + diag(eta)) q + chi^T q`` subject to
    ``sum_{i in F} q_i = Q_F`` for every group F.  This is plain QEq: there are
    no bond variables and no bond hardness, which is exactly the ``kappa -> 0``
    limit that SPEC 1.4 requires the SQE solver to reproduce to 1e-15.

    Returns the charge vector.  Written with numpy and ``solve`` so that it
    shares no code path at all with the package's conjugate-gradient solver.
    """
    eta = np.asarray(eta, dtype=float)
    chi = np.asarray(chi, dtype=float)
    n = eta.shape[0]
    hessian = np.array(coulomb, dtype=float, copy=True)
    hessian[np.diag_indices(n)] += eta

    n_groups = len(groups)
    constraint = np.zeros((n_groups, n))
    for row, members in enumerate(groups):
        constraint[row, list(members)] = 1.0

    kkt = np.zeros((n + n_groups, n + n_groups))
    kkt[:n, :n] = hessian
    kkt[:n, n:] = constraint.T
    kkt[n:, :n] = constraint
    rhs = np.concatenate([-chi, np.asarray(targets, dtype=float)])
    return np.linalg.solve(kkt, rhs)[:n]


def qeq_energy(q, eta, chi, coulomb):
    """U = 0.5 q^T (J + diag(eta)) q + chi^T q."""
    q = np.asarray(q, dtype=float)
    hessian = np.array(coulomb, dtype=float, copy=True)
    hessian[np.diag_indices(len(q))] += np.asarray(eta, dtype=float)
    return 0.5 * q @ hessian @ q + np.asarray(chi, dtype=float) @ q


# ---------------------------------------------------------------------------
# Ewald (SPEC 6: the NaCl Madelung constant)
# ---------------------------------------------------------------------------
def ewald_reference(positions, charges, cell, alpha, n_real=6, k_max=10):
    """Textbook point-charge Ewald energy in eV.

        U = U_real + U_recip + U_self + U_background

        U_real  = (K_E/2) sum_{i,j,n}' q_i q_j erfc(alpha r) / r
        U_recip = (2 pi K_E / V) sum_{k != 0} exp(-k^2 / 4 alpha^2) |S(k)|^2 / k^2
        U_self  = -K_E alpha / sqrt(pi) sum_i q_i^2
        U_bg    = -K_E pi / (2 V alpha^2) (sum_i q_i)^2

    Plain numpy, explicit image loops, no neighbour list: slow, but it shares
    nothing with the package implementation.
    """
    from scipy.special import erfc

    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    cell = np.asarray(cell, dtype=float)
    n = charges.shape[0]
    volume = abs(np.linalg.det(cell))
    qq = np.outer(charges, charges)

    real = 0.0
    for image in itertools.product(range(-n_real, n_real + 1), repeat=3):
        offset = np.asarray(image, dtype=float) @ cell
        delta = positions[:, None, :] - positions[None, :, :] + offset
        r = np.linalg.norm(delta, axis=-1)
        if image == (0, 0, 0):
            mask = ~np.eye(n, dtype=bool)
            real += 0.5 * np.sum(qq[mask] * erfc(alpha * r[mask]) / r[mask])
        else:
            real += 0.5 * np.sum(qq * erfc(alpha * r) / r)

    reciprocal_cell = 2.0 * np.pi * np.linalg.inv(cell).T
    recip = 0.0
    for index in itertools.product(range(-k_max, k_max + 1), repeat=3):
        if index == (0, 0, 0):
            continue
        k_vector = np.asarray(index, dtype=float) @ reciprocal_cell
        k_squared = k_vector @ k_vector
        structure = np.sum(charges * np.exp(1j * (positions @ k_vector)))
        recip += np.exp(-k_squared / (4.0 * alpha**2)) / k_squared * abs(structure) ** 2
    recip *= 2.0 * np.pi / volume

    self_energy = -alpha / SQRT_PI * np.sum(charges**2)
    background = -np.pi / (2.0 * volume * alpha**2) * np.sum(charges) ** 2
    return K_E * (real + recip + self_energy + background)


def madelung_from_energy(energy, nearest_neighbour, n_pairs):
    """Invert U_cell = -M K_E q^2 n_pairs / r0 for the Madelung constant M."""
    return -float(energy) * nearest_neighbour / (n_pairs * K_E)


# ---------------------------------------------------------------------------
# Fixed-k conjugate gradient (SPEC 2)
# ---------------------------------------------------------------------------
def fixed_k_cg(matrix, rhs, iterations, x0=None):
    """Exactly ``iterations`` steps of conjugate gradient on ``matrix @ x = rhs``.

    Differentiable: every step is built from torch ops on the inputs, so
    ``autograd`` propagates through the iteration, which is what SPEC 2 demands
    ("DIFFERENTIATED THROUGH", not Hellmann-Feynman, not argmin).
    """
    x = torch.zeros_like(rhs) if x0 is None else x0
    residual = rhs - matrix @ x
    direction = residual
    residual_squared = residual @ residual
    for _ in range(iterations):
        matrix_direction = matrix @ direction
        step = residual_squared / (direction @ matrix_direction)
        x = x + step * direction
        residual = residual - step * matrix_direction
        new_residual_squared = residual @ residual
        direction = residual + (new_residual_squared / residual_squared) * direction
        residual_squared = new_residual_squared
    return x


def quadratic_energy(matrix, chi, moments):
    """U = 0.5 m^T K m + chi^T m   (SPEC 1.5)."""
    return 0.5 * moments @ matrix @ moments + chi @ moments


# ---------------------------------------------------------------------------
# Finite differences
# ---------------------------------------------------------------------------
def fd_gradient(function, x, step=1.0e-4):
    """Fourth-order central difference of a scalar function of a tensor.

        f'(x) ~ [f(x-2h) - 8 f(x-h) + 8 f(x+h) - f(x+2h)] / (12 h)

    The 5-point stencil has O(h^4) truncation error, so at h = 1e-4 the
    truncation term is ~1e-16 * |f''''| while round-off is ~eps |f| / h ~
    1e-12 |f|.  That is what makes the SPEC section 6 tolerance of 1e-10 on
    "analytic gradient vs finite difference" reachable at all; a 2-point
    stencil at any h cannot get there.
    """
    x = x.detach().clone()
    gradient = torch.zeros_like(x)
    flat = x.reshape(-1)
    flat_gradient = gradient.reshape(-1)
    for index in range(flat.numel()):
        original = flat[index].item()
        values = []
        for offset in (-2.0, -1.0, 1.0, 2.0):
            flat[index] = original + offset * step
            values.append(float(function(x)))
        flat[index] = original
        flat_gradient[index] = (
            values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]
        ) / (12.0 * step)
    return gradient


# ---------------------------------------------------------------------------
# Thole relay (SPEC 1.6 / 6 -- the reciprocity test bed)
# ---------------------------------------------------------------------------
def thole_relay_matrix(positions, alpha_volume, a=2.089, asymmetric=False):
    """Dipole relay matrix ``A = diag(1/alpha_i) - T`` with Thole exponential damping.

    ``asymmetric=True`` injects a *real* Thole bug: the damping length uses only
    the source polarizability instead of the symmetric combination
    ``(alpha_i alpha_j)^(1/6)``, so ``T_ij != T_ji`` whenever the two atoms have
    different polarizabilities.  It exists so the suite can show what a broken
    relay looks like on a sparse geometry (almost nothing) and on a close-contact
    geometry (a lot) -- SPEC section 6's warning about the test bed.
    """
    positions = np.asarray(positions, dtype=float)
    alpha_volume = np.asarray(alpha_volume, dtype=float)
    n = alpha_volume.shape[0]
    matrix = np.zeros((3 * n, 3 * n))
    for i in range(n):
        matrix[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = np.eye(3) / alpha_volume[i]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            delta = positions[i] - positions[j]
            r = np.linalg.norm(delta)
            unit = delta / r
            if asymmetric:
                s = a * r / alpha_volume[j] ** (1.0 / 3.0)
            else:
                s = a * r / (alpha_volume[i] * alpha_volume[j]) ** (1.0 / 6.0)
            decay = math.exp(-s)
            damp_e = 1.0 - (1.0 + s + s * s / 2.0) * decay
            damp_t = 1.0 - (1.0 + s + s * s / 2.0 + s**3 / 6.0) * decay
            tensor = (3.0 * damp_t * np.outer(unit, unit) - damp_e * np.eye(3)) / r**3
            matrix[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = -tensor
    return matrix


def relay_polarizability(relay):
    """Molecular polarizability tensor: the 3x3 sum of all blocks of A^-1."""
    n = relay.shape[0] // 3
    inverse = np.linalg.inv(relay)
    total = np.zeros((3, 3))
    for i in range(n):
        for j in range(n):
            total += inverse[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
    return total


def relative_asymmetry(matrix):
    """max|M - M^T| / max|M| -- the scale-free asymmetry used by SPEC section 6."""
    matrix = np.asarray(matrix, dtype=float)
    return float(np.abs(matrix - matrix.T).max() / np.abs(matrix).max())
