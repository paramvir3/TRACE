"""Many-body dispersion as the zero-point energy of the master equation (SPEC 13).

One functional, two limits
--------------------------
The induced moments of the master equation live in the quadratic form
``U(y) = 0.5 y^T K_yy y + ...`` with ``y = mu_hat`` (and ``Theta_hat`` when the
quadrupole sector is on).  Its CLASSICAL minimum is induction -- that is what
the solver computes.  Give every induced moment an inertia and the same form is
a set of coupled harmonic oscillators; their QUANTUM zero-point energy, minus
that of the uncoupled atoms, is the many-body dispersion energy:

    E_disp = 1/2 sum_p omega_p  -  1/2 sum_p omega_p^(0)
           = (1 / 2 pi) int_0^inf d omega  [ ln det(K_yy + omega^2 M)
                                            - ln det(D_yy + omega^2 M) ]

with ``D_yy`` the on-site (block-diagonal) part of ``K_yy``.  The second line is
the adiabatic-connection / RPA form of the same quantity and is exact for
harmonic oscillators (``int_0^inf ln((a^2 + w^2)/(b^2 + w^2)) dw = pi (a - b)``).

ESTABLISHED, cited, not claimed: coupled fluctuating dipoles as a model of
dispersion (the quantum Drude oscillator: Jones, Crain, Sokhan, Whitfield,
Martyna, Phys. Rev. B 87, 144103 (2013)); many-body dispersion from atomic
polarizabilities (Tkatchenko, DiStasio, Car, Scheffler, Phys. Rev. Lett. 108,
236402 (2012)); its ACFDT/RPA form (Tkatchenko, Ambrosetti, DiStasio,
J. Chem. Phys. 138, 074106 (2013)); the Slater-Kirkwood / Thomas-Reiche-Kuhn
link between polarizability, electron number and frequency.

TRACE-native: the operator.  ``K_yy`` here is not a separate dispersion model
fitted beside the electrostatics -- it is the SAME Hessian block the solver
minimises, with the same learned, environment-dependent polarizabilities
``alpha_i = ell_i^2 A_mu,i^-1`` and the same damped coupling.  Under the Gaussian
kernel of SPEC 12 that block is a Coulomb Gram matrix plus positive on-site
residuals, so ``K_yy + omega^2 M`` is positive definite for EVERY geometry and
every frequency: the imaginary-frequency / polarization-catastrophe failure of
MBD cannot occur.

Inertia from a sum rule, not a fit
----------------------------------
A Drude oscillator of response polarizability ``alpha_a`` along axis ``a``
carrying ``N`` electrons of oscillator strength has ``alpha_a omega_a^2 = N
hbar^2/m_e`` (TRK).  Its inertia ``m = 1/(alpha_a omega_a^2) = 1/(N hbar^2/m_e)``
is therefore ISOTROPIC and independent of alpha: anisotropy lives in the spring
``A_mu``, the mass is set by the electron count.  In reduced moments
``mu_hat = mu/(e ell)`` it is ``m_i = ell_i^2 / (N_i hbar^2/m_e)``.  The only new
per-atom quantity is ``N_i``, the effective number of polarizable electrons,
emitted by a positive head.  Pairwise, this reproduces the London /
Slater-Kirkwood coefficient ``C6 = (3/2) alpha_i alpha_j w_i w_j / (w_i + w_j)``
exactly in the large-separation limit -- measured 1.000001 at R = 12 A.

Why log-determinants and not eigenvalues
----------------------------------------
Both give the same energy (measured agreement 1e-10 on a cube, an octahedron and
a 32-atom fcc block).  They do not give the same SECOND derivative: at an exactly
degenerate spectrum -- any crystal, any symmetric molecule -- the eigenvector
derivative of ``eigh`` divides by eigenvalue gaps and the force-loss gradient
came out NaN on all three structures, while the Cholesky route stayed finite.
Force training needs that second derivative, so this module never calls
``eigh``.

Cost: ``n_quadrature`` Cholesky factorisations of the ``3n`` (or ``8n``) block,
i.e. O(n^3) like the dense solve itself.  16 points reach ~1e-11 relative.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .constants import HBAR2_OVER_ME, K_E

__all__ = [
    "DispersionError",
    "dipole_inertia",
    "isolated_frequencies",
    "dispersion_energy",
    "sphere_quadrature",
]

_SPATIAL_DIM = 3
_QUADRUPOLE_DIM = 5


class DispersionError(ValueError):
    """``K_yy + omega^2 M`` failed to factorise: the moment block is not positive definite."""


def dipole_inertia(electrons: Tensor, ell: Tensor) -> Tensor:
    """``m_i = ell_i^2 / (N_i hbar^2/m_e)``: inertia of the reduced dipole, eV^-1.

    From the Thomas-Reiche-Kuhn sum rule; see the module docstring.
    """
    return ell * ell / (electrons * HBAR2_OVER_ME)


def isolated_frequencies(A_mu: Tensor, ell: Tensor, electrons: Tensor) -> Tensor:
    """``[n, 3]`` excitation energies ``hbar omega_a`` (eV) of each isolated atom.

    ``omega_a^2 = lambda_a(A_mu) / m`` along each principal axis, i.e.
    ``N hbar^2/m_e / alpha_a``.  A diagnostic: a free hydrogen atom
    (alpha = 0.6668 A^3, N = 1) comes out at 12.83 eV, and fitting its exact
    C6 = 6.499 a.u. needs N = 0.82 -- the effective, not the nominal, electron
    count.
    """
    stiffness = torch.linalg.eigvalsh(A_mu)
    return torch.sqrt(stiffness / dipole_inertia(electrons, ell).unsqueeze(-1))


def _gauss_legendre(n_points: int, dtype: torch.dtype, device: torch.device):
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    nodes = torch.as_tensor(0.5 * (nodes + 1.0), dtype=dtype, device=device)
    weights = torch.as_tensor(0.5 * weights, dtype=dtype, device=device)
    return nodes, weights


def sphere_quadrature(
    n_polar: int = 12, n_azimuth: int = 24, *, dtype=torch.float64, device=None
) -> tuple[Tensor, Tensor]:
    """Unit directions and weights for averaging a smooth function over the sphere.

    Gauss-Legendre in ``cos(theta)`` times the trapezoid rule in ``phi``: exact
    for every spherical harmonic of degree ``< min(2 n_polar, n_azimuth)``.  The
    functions averaged here (SPEC 13.1) are analytic on the sphere, so the
    error falls exponentially with the degree; the default (288 directions,
    degree 23) moves a rotated crystal's dispersion by < 1e-10 relative.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_polar)
    cos_theta = torch.as_tensor(nodes, dtype=dtype, device=device)
    polar_weight = torch.as_tensor(weights, dtype=dtype, device=device) / 2.0
    phi = torch.arange(n_azimuth, dtype=dtype, device=device) * (2.0 * math.pi / n_azimuth)
    sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)
    directions = torch.stack(
        [
            sin_theta[:, None] * torch.cos(phi)[None, :],
            sin_theta[:, None] * torch.sin(phi)[None, :],
            cos_theta[:, None].expand(-1, n_azimuth),
        ],
        dim=-1,
    ).reshape(-1, 3)
    direction_weight = (polar_weight[:, None] / n_azimuth).expand(-1, n_azimuth).reshape(-1)
    return directions, direction_weight


def _scale(inverse_factor: Tensor, block: Tensor, inverse_factor_right: Tensor) -> Tensor:
    """``blockdiag(L_row^-1) T blockdiag(L_col^-1)^T`` for a ``[n, a, n, b]`` block."""
    return torch.einsum("iab,ibjc,jdc->iajd", inverse_factor, block, inverse_factor_right)


def dispersion_energy(
    dipole_pair: Tensor,
    A_mu: Tensor,
    ell: Tensor,
    electrons: Tensor,
    *,
    A_Theta: Optional[Tensor] = None,
    dipole_quadrupole_pair: Optional[Tensor] = None,
    quadrupole_pair: Optional[Tensor] = None,
    n_quadrature: int = 16,
    frequency_scale: float = 10.0,
    macroscopic_volume: Optional[Tensor] = None,
) -> Tensor:
    """Zero-point energy of the induced-moment block of the master equation, eV.

    Args:
        dipole_pair: ``[3n, 3n]`` INTER-site part of the reduced dipole block,
            ``K_mu_mu - blockdiag(A_mu)`` (under a lattice sum its diagonal
            carries the self-image term, which belongs here).
        A_mu: ``[n, 3, 3]`` on-site dipole stiffness, eV.
        ell: ``[n]`` dipole length, A.
        electrons: ``[n]`` effective polarizable electron count ``N_i > 0``.
        A_Theta, dipole_quadrupole_pair, quadrupole_pair: all three or none.
            ``[n, 5, 5]`` on-site quadrupole stiffness, ``[3n, 5n]`` and
            ``[5n, 5n]`` inter-site blocks.  With them the quadrupoles fluctuate
            too, at the single-pole QDO frequency ``2 omega_bar_i`` (a harmonic
            oscillator's quadrupole transition is two quanta), which adds the
            dipole-quadrupole (C8-type) dispersion.  An approximation of stated
            form: a true Drude quadrupole is quadratic in the displacement, not
            an independent harmonic coordinate.
        n_quadrature: Gauss-Legendre points on ``x in (0, 1)``,
            ``omega = s x / (1 - x)``.
        frequency_scale: ``s``, eV.
        macroscopic_volume: the cell volume ``Omega`` of a 3-D periodic block,
            to add the NON-ANALYTIC Gamma term (SPEC 13.1).  A lattice-summed
            ``dipole_pair`` under tinfoil boundaries drops ``k = 0``, which is
            the right q -> 0 limit only for TRANSVERSE fluctuations; a
            longitudinal one also carries the depolarization
            ``(4 pi K_E / Omega) q_hat q_hat``.  By the matrix determinant lemma
            the Gamma sector's missing term is
            ``(1/2pi) int dw < ln(1 + (4 pi K_E/Omega) q_hat . chi(iw) . q_hat) >``
            with ``chi(iw) = U^T (K + w^2 M)^-1 U`` the cell's macroscopic
            polarizability at imaginary frequency (``U_i = ell_i I``) -- the
            longitudinal dielectric function, averaged over directions.  It is
            the coupled-dipole form of the q -> 0 head treatment of plane-wave
            RPA.  ``None`` (open boundaries, slabs): no term.

    Returns:
        Scalar dispersion energy (negative), differentiable in every input.

    Raises:
        DispersionError: if a factorisation fails, i.e. the moment block is not
            positive definite -- impossible under the Gaussian kernel with a
            valid parameter set; a Thole block can in principle get there.
    """
    quadrupolar = A_Theta is not None
    if quadrupolar != (dipole_quadrupole_pair is not None) or quadrupolar != (
        quadrupole_pair is not None
    ):
        raise ValueError(
            "A_Theta, dipole_quadrupole_pair and quadrupole_pair go together: "
            "a fluctuating quadrupole needs its stiffness and both couplings."
        )
    n_atoms = int(A_mu.shape[0])
    dtype, device = A_mu.dtype, A_mu.device
    if n_atoms == 0:
        return A_mu.new_zeros(())

    inertia = dipole_inertia(electrons, ell)                       # [n]
    eye3 = torch.eye(_SPATIAL_DIM, dtype=dtype, device=device)
    t_mm = dipole_pair.reshape(n_atoms, _SPATIAL_DIM, n_atoms, _SPATIAL_DIM)
    if quadrupolar:
        # omega_bar^2: the mean of the three per-axis omega_a^2 = lambda_a / m.
        omega_bar_sq = torch.diagonal(A_mu, dim1=-2, dim2=-1).sum(-1) / (3.0 * inertia)
        chol_theta = torch.linalg.cholesky(A_Theta)
        eye5 = torch.eye(_QUADRUPOLE_DIM, dtype=dtype, device=device)
        inv_chol_theta = torch.linalg.solve_triangular(
            chol_theta, eye5.expand(n_atoms, -1, -1), upper=False
        )
        t_mq = dipole_quadrupole_pair.reshape(
            n_atoms, _SPATIAL_DIM, n_atoms, _QUADRUPOLE_DIM
        )
        t_qq = quadrupole_pair.reshape(
            n_atoms, _QUADRUPOLE_DIM, n_atoms, _QUADRUPOLE_DIM
        )

    nodes, weights = _gauss_legendre(int(n_quadrature), dtype, device)
    total = A_mu.new_zeros(())
    if macroscopic_volume is not None:
        directions, direction_weight = sphere_quadrature(dtype=dtype, device=device)
        depolarization = 4.0 * math.pi * K_E / macroscopic_volume
        # U = d(Omega P)/d mu_hat = ell_i I, stacked; zero on the Theta rows.
        cell_dipole = (ell.view(-1, 1, 1) * eye3).reshape(n_atoms * _SPATIAL_DIM, 3)
    for node, weight in zip(nodes.unbind(0), weights.unbind(0)):
        omega_sq = (frequency_scale * node / (1.0 - node)) ** 2
        jacobian = frequency_scale / (1.0 - node) ** 2
        # D(omega) = A_mu + omega^2 m I per atom; its Cholesky factor whitens T.
        on_site = A_mu + (omega_sq * inertia).reshape(-1, 1, 1) * eye3
        inv_chol = torch.linalg.solve_triangular(
            torch.linalg.cholesky(on_site), eye3.expand(n_atoms, -1, -1), upper=False
        )
        s_mm = _scale(inv_chol, t_mm, inv_chol).reshape(
            n_atoms * _SPATIAL_DIM, n_atoms * _SPATIAL_DIM
        )
        if not quadrupolar:
            whitened = s_mm
        else:
            # D_Theta(omega) = A_Theta (1 + omega^2 / (2 omega_bar)^2): same
            # factor, rescaled, so its Cholesky is taken once, outside the loop.
            rescale = torch.rsqrt(1.0 + omega_sq / (4.0 * omega_bar_sq))
            inv_q = inv_chol_theta * rescale.reshape(-1, 1, 1)
            s_mq = _scale(inv_chol, t_mq, inv_q).reshape(
                n_atoms * _SPATIAL_DIM, n_atoms * _QUADRUPOLE_DIM
            )
            s_qq = _scale(inv_q, t_qq, inv_q).reshape(
                n_atoms * _QUADRUPOLE_DIM, n_atoms * _QUADRUPOLE_DIM
            )
            whitened = torch.cat(
                [torch.cat([s_mm, s_mq], dim=1), torch.cat([s_mq.T, s_qq], dim=1)], dim=0
            )
        # ln det(D + T) - ln det(D) = ln det(I + L^-1 T L^-T).  Symmetrised:
        # the Cholesky reads one triangle, so an asymmetric round-off would
        # otherwise make the energy depend on which triangle that is.
        identity = torch.eye(int(whitened.shape[0]), dtype=dtype, device=device)
        whitened = identity + 0.5 * (whitened + whitened.T)
        try:
            factor = torch.linalg.cholesky(whitened)
        except RuntimeError as error:
            raise DispersionError(
                "the induced-moment block K_yy + omega^2 M is not positive definite "
                f"at hbar*omega = {math.sqrt(float(omega_sq)):.4g} eV, so the "
                "zero-point energy has an imaginary mode -- the polarization "
                "catastrophe. Under multipole_kernel='gaussian' this cannot happen "
                "for a valid parameter set (SPEC 12); under Thole it means the "
                "damping did not hold at this geometry."
            ) from error
        log_det = 2.0 * torch.log(torch.diagonal(factor)).sum()
        if macroscopic_volume is not None:
            # chi = U^T (D + T)^-1 U = (L^-1 U)^T S^-1 (L^-1 U), with the
            # whitening L already in hand and S = F F^T just factorised.
            whitened_dipole = (
                inv_chol @ cell_dipole.reshape(n_atoms, _SPATIAL_DIM, 3)
            ).reshape(n_atoms * _SPATIAL_DIM, 3)
            if quadrupolar:
                whitened_dipole = torch.cat(
                    [
                        whitened_dipole,
                        whitened_dipole.new_zeros(n_atoms * _QUADRUPOLE_DIM, 3),
                    ]
                )
            projected = torch.linalg.solve_triangular(factor, whitened_dipole, upper=False)
            chi = projected.T @ projected
            longitudinal = torch.einsum("da,ab,db->d", directions, chi, directions)
            log_det = log_det + (
                direction_weight * torch.log1p(depolarization * longitudinal)
            ).sum()
        total = total + weight * jacobian * log_det
    return total / (2.0 * math.pi)
