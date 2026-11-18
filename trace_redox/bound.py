"""Exact elimination of the bound (SPEC 1) sector in the presence of free charge.

The bound sector is trace_electrostatics' assembly, used unchanged and read
only through its operator: the moments ``y = (t, mu_hat[, Theta_hat])`` with
``rho = 0`` (every bound fragment neutral -- the free sector of SPEC 25
carries the net charge).  Its energy is exactly quadratic,

    U_b(y) = E0 + g^T y + 1/2 y^T K y          (assembly.energy, rho = 0)

A free charge ``f_i`` on atom ``i`` is a normalised Gaussian of its OWN width
``sigma_f,i`` (SPEC 25.2; :class:`trace_redox.heads.FreeChargeWidth` derives it
from the atomic hardness).  It couples to the bound moments and to itself only
through Coulomb's law -- never through the bound sector's ``chi`` or ``eta``,
which would double count the free sector's IE / EA:

    f^T A y + 1/2 f^T J f + f^T h
    A = [J_fb B,  T01_f,  T02_f]                 free charge -- bound (q, mu_hat, Theta_hat)
    J_fb[i, j] = J(r_ij; s(sigma_f,i, sigma_j)),  J_fb[i, i] = 2 K_E / (sqrt(pi) s(sigma_f,i, sigma_i))
    J[i, j]    = J(r_ij; s(sigma_f,i, sigma_f,j)), J[i, i]   = S0(sigma_f,i) = sqrt(2) K_E / (sqrt(pi) sigma_f,i)
    h          = -r . E                           (applied field, open boundaries)

with ``J(r; s) = K_E erf(r/s)/r``, ``s(a, b) = sqrt(a^2 + b^2)``, and ``T01_f`` /
``T02_f`` the Gaussian charge--dipole / charge--quadrupole tensors of
trace_electrostatics.kernels with the FREE width on the charge index.  Every
entry is the Coulomb integral of two normalised Gaussian multipole densities,
so the joint operator in ``(y, f)`` is K's Gram construction (SPEC 12) extended
by more Gaussians: positive definite, because Gaussians of different widths on
the same centre are linearly independent and the on-site floors of SPEC 12 /
17.3 hold.  Minimising over ``y`` in closed form:

    y*(f)  = y0 - K^{-1} A^T f,        y0 = -K^{-1} g
    E_b(f) = E_b0 + u0^T f + 1/2 f^T W f
    E_b0   = E0 + 1/2 g^T y0           (SPEC 1's own minimum for the neutral system)
    u0     = h + A y0                  (potential of that solution at the free charges)
    W      = J - A K^{-1} A^T          (screened Coulomb: a Schur complement, positive definite)

With ``sigma_f = sigma`` the blocks are exactly the assembly's own
(``J = j_pair + diag(S0)``, ``T01_f = t01_pair``, ``T02_f = t02_pair``); the tests
use that identity to check this construction against trace_electrostatics.

K^{-1} is applied through K's SQE structure (diagonal bond hardness plus a
rank-n coupling through the incidence; :class:`_CondensedK`, SPEC §13): exact,
differentiable to every order by plain autograd, cost O(n^3 + n_p^3 + n n_b k).
The dense route (``method='dense'``, cost ``O((n_bonds + 3n [+5n])^3)``) is
kept as the reference the tests compare against.  Periodic boundaries (3-D and 2-D slab) build the
free blocks in the plane-wave Gram form, :func:`periodic_free_charge_blocks`,
equal to the assembly's own Ewald blocks for equal widths (tested to 1e-14).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import Tensor

import math

from trace_electrostatics.constants import K_E
from trace_electrostatics.energy import ElectrostaticAssembly, _gaussian_cutoff_floor, _open_pair_geometry, _slab_geometry
from trace_electrostatics.kernels import (
    charge_dipole_tensor,
    charge_quadrupole_tensor,
    combine_sigma,
    coulomb_kernel,
    gaussian_damping,
    kernel_self,
    quadrupole_cartesian_basis,
)
from trace_electrostatics.solver import gaussian_widths


class BoundSectorError(ValueError):
    """The bound sector cannot be eliminated (indefinite or unsupported operator)."""


class _DenseK:
    """K held densely with its Cholesky factor (the reference route)."""

    def __init__(self, K: Tensor, factor: Tensor):
        self.K = K
        self.factor = factor

    def solve(self, R: Tensor) -> Tensor:
        return torch.cholesky_solve(R, self.factor)

    def matrix(self) -> Tensor:
        return self.K


class _CondensedK:
    """K applied and inverted exactly through its SQE structure, never formed (NEW-TO-TRACE).

    In the variational basis ``y = (t, p)``, t the bond charge transfers and
    p = (mu_hat[, Theta_hat]) the atom-centred moments (trace_redox/SPEC.md §13):

        K = blockdiag(D, P) + V^T C V,   D = diag(kappa),   V = blockdiag(B, I),   C = [[H, T], [T^T, 0]]

    with B the [n, n_b] bond incidence, H = J + diag(eta) the atomic hardness
    (Gaussian Coulomb Gram matrix plus on-site hardness: positive definite),
    T the charge--multipole couplings and P the multipole block.  The bond
    block is a diagonal plus a rank-n term, so with H = R R^T and G = R^T B

        K_tt^-1 = D^-1 - D^-1 G^T F^-1 G D^-1,       F = I + G D^-1 G^T   (eigenvalues >= 1)
        S       = P - T^T Lam T,                      Lam = B K_tt^-1 B^T  (Schur complement of K)
        K^-1 [r_t; r_p]:  u = K_tt^-1 r_t,  x_p = S^-1 (r_p - T^T B u),  x_t = u - K_tt^-1 B^T T x_p

    Exact (block elimination and Woodbury); only [n, n] and [n_p, n_p]
    factorisations, so O(n^3 + n_p^3 + n n_b k) instead of O((n_b + n_p)^3),
    and every step is differentiable by plain autograd at that cost.
    """

    def __init__(self, kappa: Tensor, B: Tensor, H: Tensor, T: Tensor, P: Tensor):
        self.kappa, self.B, self.H, self.T, self.P = kappa, B, H, T, P
        self.nb = int(kappa.shape[0])
        self.d_inv = 1.0 / kappa
        R, info = torch.linalg.cholesky_ex(H)
        if int(info) != 0:
            raise BoundSectorError(f"the atomic hardness H = J + diag(eta) is not positive definite (column {int(info)})")
        self.G = R.transpose(0, 1) @ B                                           # [n, n_b]
        eye = torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
        F = eye + (self.G * self.d_inv.unsqueeze(0)) @ self.G.transpose(0, 1)
        self.F_factor = torch.linalg.cholesky(0.5 * (F + F.transpose(0, 1)))
        L = (B * self.d_inv.unsqueeze(0)) @ B.transpose(0, 1)                     # B D^-1 B^T
        LR = L @ R
        Lam = L - LR @ torch.cholesky_solve(LR.transpose(0, 1), self.F_factor)
        S = P - T.transpose(0, 1) @ Lam @ T
        S_factor, info = torch.linalg.cholesky_ex(0.5 * (S + S.transpose(0, 1)))
        if int(info) != 0:
            raise BoundSectorError(f"the bound operator K is not positive definite (multipole Schur complement, column {int(info)})")
        self.S_factor = S_factor

    def _solve_tt(self, X: Tensor) -> Tensor:
        Y = self.d_inv.unsqueeze(1) * X
        Z = torch.cholesky_solve(self.G @ Y, self.F_factor)
        return Y - self.d_inv.unsqueeze(1) * (self.G.transpose(0, 1) @ Z)

    def solve(self, R: Tensor) -> Tensor:
        """``K^-1 R`` for R of shape [n_b + n_p, k]."""
        r_t, r_p = R[:self.nb], R[self.nb:]
        u = self._solve_tt(r_t)
        x_p = torch.cholesky_solve(r_p - self.T.transpose(0, 1) @ (self.B @ u), self.S_factor)
        x_t = u - self._solve_tt(self.B.transpose(0, 1) @ (self.T @ x_p))
        return torch.cat((x_t, x_p), dim=0)

    def matrix(self) -> Tensor:
        """K densely (tests and diagnostics only: O((n_b + n_p)^2) memory)."""
        BT = self.B.transpose(0, 1)
        top = torch.cat((torch.diag(self.kappa) + BT @ self.H @ self.B, BT @ self.T), dim=1)
        bottom = torch.cat((self.T.transpose(0, 1) @ self.B, self.P), dim=1)
        return torch.cat((top, bottom), dim=0)


@dataclasses.dataclass
class BoundElimination:
    """The bound sector reduced onto the free-charge sites (atoms)."""

    assembly: ElectrostaticAssembly
    e_b0: Tensor      #: [] minimum bound energy with no free charge (eV)
    u0: Tensor        #: [n] potential of that solution at the free charges (+ field), eV / e
    W: Tensor         #: [n, n] screened free-free Coulomb operator, eV / e^2
    J: Tensor         #: [n, n] bare free-free Gaussian Coulomb operator, eV / e^2
    y0: Tensor        #: [m] bound minimiser with no free charge
    operator: Any     #: K (``solve``, ``matrix``): _CondensedK, or _DenseK on the reference route
    A: Tensor         #: [n, m] free-bound coupling
    n_bonds: int
    asymmetry: float  #: relative asymmetry of the assembly's operator blocks (round-off)
    method: str = "condensed"
    #: [] free-sector constant: 0, or the frame terms of the core + carrier reduction (SPEC 17.9).  It belongs
    #: to the free energy, not to e_b0 (the bound sector alone).
    constant: Any = 0.0

    def bound_moments(self, f: Tensor) -> Tensor:
        """``y*(f) = y0 - K^{-1} A^T f``."""
        rhs = (self.A.transpose(0, 1) @ f).unsqueeze(1)
        return self.y0 - self.operator.solve(rhs).squeeze(1)

    def split_moments(self, y: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """``(t, mu_hat [n, 3], Theta_hat [n, 5] or None)`` from ``y``."""
        n, nb = self.assembly.n_atoms, self.n_bonds
        theta = y[nb + 3 * n:].reshape(n, 5) if self.assembly.quadrupoles else None
        return y[:nb], y[nb:nb + 3 * n].reshape(n, 3), theta


def _variational_indices(assembly: ElectrostaticAssembly) -> Tensor:
    """Indices of ``(t, mu_hat[, Theta_hat])`` in the packed ``(t, rho, mu_hat[, Theta_hat])``."""
    n, nb = assembly.n_atoms, assembly.n_bonds
    keep = torch.ones(assembly.moment_dim, dtype=torch.bool)
    keep[nb:nb + n] = False
    return torch.nonzero(keep, as_tuple=False).squeeze(1).to(assembly.pos.device)


def _refuse_unsupported(assembly: ElectrostaticAssembly) -> None:
    if assembly.lattice_operator is not None:
        raise BoundSectorError("trace_redox v1 needs the dense route (solver='dense')")
    if assembly.periodic and (assembly.j_pair is None or assembly.t01_pair is None):
        raise BoundSectorError("a periodic trace_redox assembly needs the dense lattice blocks")
    if assembly.config.multipole_kernel != "gaussian":
        raise BoundSectorError("trace_redox needs multipole_kernel='gaussian' (SPEC 12 / 17.5)")
    if assembly.metal_domain is not None:
        raise BoundSectorError("metallic domains (SPEC 20) are not supported by trace_redox v1")
    if assembly.fragment_spins is not None:
        raise BoundSectorError("the spin sector (SPEC 9) is not supported by trace_redox v1")
    if assembly.structure_id is not None:
        raise BoundSectorError("block-diagonal batches (SPEC 24) are not supported by trace_redox v1")
    if assembly.params.bond_bias is not None:
        raise BoundSectorError("electrode bond bias (SPEC 20) is not supported by trace_redox v1")
    if bool((assembly.fragment_charges != 0).any()):
        raise BoundSectorError(
            "trace_redox keeps every bound fragment neutral; the free sector carries the "
            f"net charge. Got fragment charges {assembly.fragment_charges.tolist()}"
        )
    if assembly.external_field is not None and assembly.periodic:
        raise BoundSectorError(
            "an applied field under periodic boundaries: the free charges' -sum c_u r_u . E "
            "is branch dependent (no bond-current form exists for a free charge); refused"
        )


def charge_coulomb_block(pos: Tensor, sigma_row: Tensor, sigma_col: Tensor) -> Tensor:
    """Coulomb integrals of normalised Gaussian charges, row widths x column widths (open boundaries).

    ``M[i, j] = K_E erf(r_ij / s_ij) / r_ij`` with ``s_ij = sqrt(sigma_row_i^2 +
    sigma_col_j^2)``, including ``i = j`` (two concentric clouds):
    ``M[i, i] = 2 K_E / (sqrt(pi) s_ii)``, which for equal widths is the Gram
    self-energy ``S0(sigma)``.  ``M(a, b) = M(b, a)^T``.  Distinct atoms at one
    point take the same ``r -> 0`` limit (trace_electrostatics' coincidence guard).
    """
    _, dist, keep = _open_pair_geometry(pos)
    s = combine_sigma(sigma_row.unsqueeze(1), sigma_col.unsqueeze(0))
    eye = torch.eye(int(pos.shape[0]), dtype=dist.dtype, device=dist.device)
    off_diagonal = 1.0 - eye
    pair = torch.where(keep > 0, coulomb_kernel(dist, s), kernel_self(s)) * off_diagonal
    return pair + torch.diag(kernel_self(combine_sigma(sigma_row, sigma_col)))


def free_charge_blocks(assembly: ElectrostaticAssembly, sigma_free: Tensor) -> tuple[Tensor, Tensor]:
    """``(J [n, n], A [n, m])`` for free Gaussians of widths ``sigma_free`` (open boundaries).

    The coincidence guard is trace_electrostatics' own (``_open_pair_geometry``):
    two distinct atoms at one point take the ``r -> 0`` limit of the charge
    kernel and zero multipole tensors, exactly as ``_direct_pair_blocks`` does.
    """
    params = assembly.params
    n = assembly.n_atoms
    vec, dist, keep = _open_pair_geometry(assembly.pos)
    J = charge_coulomb_block(assembly.pos, sigma_free, sigma_free)
    J_fb = charge_coulomb_block(assembly.pos, sigma_free, params.sigma)

    sigma_mu, sigma_theta = gaussian_widths(params, assembly.config.gaussian_a)
    ell = params.ell
    s01 = torch.sqrt(sigma_free.unsqueeze(1) ** 2 + sigma_mu.unsqueeze(0) ** 2)
    lam3 = gaussian_damping(dist, s01)[0]
    t01 = charge_dipole_tensor(vec, dist, lam3) * ell.view(1, n, 1) * keep.unsqueeze(-1)
    blocks = [J_fb @ assembly._dense_incidence(), t01.reshape(n, 3 * n)]
    if assembly.quadrupoles:
        basis = quadrupole_cartesian_basis().to(dtype=dist.dtype, device=dist.device)
        s02 = torch.sqrt(sigma_free.unsqueeze(1) ** 2 + sigma_theta.unsqueeze(0) ** 2)
        lam3_q, lam5_q, _, _ = gaussian_damping(dist, s02)
        t02 = torch.einsum("ijab,mab->ijm", charge_quadrupole_tensor(vec, dist, lam3_q, lam5_q), basis)
        t02 = t02 * (keep * (ell * ell).unsqueeze(0)).unsqueeze(-1)
        blocks.append(t02.reshape(n, 5 * n))
    return J, torch.cat(blocks, dim=1)


#: k_max = PERIODIC_K_FACTOR / s_min: the omitted tail of the slowest Gaussian
#: pair is ~erfc(k_max s_min / 2) = erfc(6) = 2e-17 relative (times the k^2 of a
#: quadrupole factor, still < 1e-14).
PERIODIC_K_FACTOR = 12.0
#: Refuse a reciprocal sum larger than this many half-space vectors.
PERIODIC_MAX_K = 4_000_000
#: Reciprocal vectors per matrix-product chunk (bounds memory to ~n x chunk).
PERIODIC_K_CHUNK = 8192


def reciprocal_half_space(cell: Tensor, k_max: float) -> Tensor:
    """Reciprocal vectors ``k = 2 pi cell^-T m``, ``0 < |k| <= k_max``, one of each +-k pair.

    The index set is chosen from the detached cell (a static shape per call); the
    vectors themselves carry the cell's graph, so the stress follows by autograd.
    """
    recip = 2.0 * math.pi * torch.linalg.inv(cell).transpose(0, 1)       # rows b_i, a_i . b_j = 2 pi delta_ij
    lengths = cell.detach().norm(dim=1)
    m_max = [int(math.ceil(k_max * float(L) / (2.0 * math.pi))) for L in lengths]
    count = (2 * m_max[0] + 1) * (2 * m_max[1] + 1) * (2 * m_max[2] + 1)
    if count > 8 * PERIODIC_MAX_K:
        raise BoundSectorError(f"the periodic free-charge sum needs ~{count} reciprocal vectors; cell too large")
    ranges = [torch.arange(-m, m + 1, device=cell.device) for m in m_max]
    m = torch.cartesian_prod(*ranges).to(cell.dtype)                       # [N, 3]
    # half space: first non-zero index positive (excludes 0 and one of each +-m)
    first = torch.where(m[:, 0] != 0, m[:, 0], torch.where(m[:, 1] != 0, m[:, 1], m[:, 2]))
    m = m[first > 0]
    k = m @ recip.detach()
    m = m[(k * k).sum(-1) <= k_max * k_max]
    if int(m.shape[0]) > PERIODIC_MAX_K:
        raise BoundSectorError(f"the periodic free-charge sum needs {int(m.shape[0])} reciprocal vectors; cell too large")
    return m @ recip


def periodic_charge_gram(pos: Tensor, cell: Tensor, sigma_row: Tensor, sigma_col: Tensor) -> Tensor:
    """Plane-wave Gram matrix of normalised Gaussian charges on a lattice (tinfoil, k = 0 removed).

    ``M_ij = (4 pi K_E / V) sum_{k != 0} exp(-k^2 (sigma_row_i^2 + sigma_col_j^2) / 4) cos(k.r_ij) / k^2``:
    the charge block of :func:`periodic_free_charge_blocks` for arbitrary row and
    column widths (no slab correction).  Exposed for tests and analysis.
    """
    s_min = math.sqrt(float(sigma_row.detach().min()) ** 2 + float(sigma_col.detach().min()) ** 2)
    k_vectors = reciprocal_half_space(cell, PERIODIC_K_FACTOR / s_min)
    prefactor = 2.0 * (4.0 * math.pi * K_E) / torch.det(cell).abs()
    out = pos.new_zeros(int(pos.shape[0]), int(pos.shape[0]))
    for start in range(0, int(k_vectors.shape[0]), PERIODIC_K_CHUNK):
        k = k_vectors[start:start + PERIODIC_K_CHUNK]
        k2 = (k * k).sum(-1)
        phase = pos @ k.transpose(0, 1)
        g_row = torch.exp(-0.25 * k2.unsqueeze(0) * (sigma_row * sigma_row).unsqueeze(1))
        g_col = torch.exp(-0.25 * k2.unsqueeze(0) * (sigma_col * sigma_col).unsqueeze(1))
        weight = prefactor / k2
        out = out + (g_row * torch.cos(phase) * weight) @ (g_col * torch.cos(phase)).transpose(0, 1)
        out = out + (g_row * torch.sin(phase) * weight) @ (g_col * torch.sin(phase)).transpose(0, 1)
    return out


def periodic_free_charge_blocks(
    assembly: ElectrostaticAssembly, sigma_free: Tensor
) -> tuple[Tensor, Tensor]:
    """``(J [n, n], A [n, m])`` under periodic boundaries, in the plane-wave Gram form.

    For normalised Gaussian densities (Fourier factor ``g_i(k) = exp(-k^2 sigma_i^2 / 4)``)
    under tinfoil boundaries with the ``k = 0`` component removed (the neutralising
    background -- the convention of trace_electrostatics.lattice, SPEC 14):

        J_ij     = (4 pi K_E / V) sum_{k != 0} g^f_i g^f_j cos(k.r_ij) / k^2
        J^fb_ij  = same with g^f_i g^b_j               (free charge i, bound charge j)
        T01_ija  = (4 pi K_E / V) sum_k g^f_i g^mu_j k_a sin(k.r_ij) / k^2 * ell_j
        T02_ijm  = -(1/2)(4 pi K_E / V) sum_k g^f_i g^Th_j (k.W_m.k) cos(k.r_ij) / k^2 * ell_j^2

    ``r_ij = r_i - r_j``.  The Gaussian factor separates per site, so no Ewald
    splitting parameter enters and any mixture of widths is exact up to the
    k-cutoff.  The diagonal is the full periodic self-energy (same-image Gram self
    term included), i.e. the assembly's ``j_pair + diag(S0)`` for equal widths --
    tested.  A 2-D slab adds the Yeh-Berkowitz term ``(4 pi K_E / V) z_i z_j`` to the
    charge blocks and ``(4 pi K_E / V) z_i ell_j n`` to ``T01`` (width independent;
    trace_electrostatics.energy._slab_correction_blocks, same normal and volume).
    """
    from trace_electrostatics.kernels import quadrupole_cartesian_basis

    params = assembly.params
    pos, cell = assembly.pos, assembly.cell
    n = assembly.n_atoms
    dtype = pos.dtype
    sigma_mu, sigma_theta = gaussian_widths(params, assembly.config.gaussian_a)
    widths = [sigma_free, params.sigma, sigma_mu] + ([sigma_theta] if assembly.quadrupoles else [])
    s_min = min(math.sqrt(float(sigma_free.detach().min()) ** 2 + float(w.detach().min()) ** 2) for w in widths)
    k_vectors = reciprocal_half_space(cell, PERIODIC_K_FACTOR / s_min)
    volume = torch.det(cell).abs()
    prefactor = 2.0 * (4.0 * math.pi * K_E) / volume                         # 2: the +-k pair
    basis = quadrupole_cartesian_basis().to(dtype=dtype, device=pos.device) if assembly.quadrupoles else None

    J = pos.new_zeros(n, n)
    J_fb = pos.new_zeros(n, n)
    t01 = pos.new_zeros(n, n, 3)
    t02 = pos.new_zeros(n, n, 5) if assembly.quadrupoles else None
    for start in range(0, int(k_vectors.shape[0]), PERIODIC_K_CHUNK):
        k = k_vectors[start:start + PERIODIC_K_CHUNK]                        # [c, 3]
        k2 = (k * k).sum(-1)
        weight = prefactor / k2                                               # [c]
        phase = pos @ k.transpose(0, 1)                                       # [n, c]
        cos, sin = torch.cos(phase), torch.sin(phase)

        def factors(sigma: Tensor) -> tuple[Tensor, Tensor]:
            g = torch.exp(-0.25 * k2.unsqueeze(0) * (sigma * sigma).unsqueeze(1))
            return g * cos, g * sin

        cf, sf = factors(sigma_free)
        cb, sb = factors(params.sigma)
        cm, sm = factors(sigma_mu)
        J = J + (cf * weight) @ cf.transpose(0, 1) + (sf * weight) @ sf.transpose(0, 1)
        J_fb = J_fb + (cf * weight) @ cb.transpose(0, 1) + (sf * weight) @ sb.transpose(0, 1)
        for a in range(3):
            wk = weight * k[:, a]
            t01[:, :, a] = t01[:, :, a] + (sf * wk) @ cm.transpose(0, 1) - (cf * wk) @ sm.transpose(0, 1)
        if t02 is not None:
            ct, st = factors(sigma_theta)
            kk = torch.einsum("ca,mab,cb->cm", k, basis, k)                   # k.W_m.k
            for mm in range(5):
                wk = -0.5 * weight * kk[:, mm]
                t02[:, :, mm] = t02[:, :, mm] + (cf * wk) @ ct.transpose(0, 1) + (sf * wk) @ st.transpose(0, 1)
    ell = params.ell
    t01 = t01 * ell.view(1, n, 1)
    if t02 is not None:
        t02 = t02 * (ell * ell).view(1, n, 1)
    if assembly.slab:
        normal, _height, slab_volume = _slab_geometry(pos, cell, assembly.slab_axis)
        coefficient = 2.0 * (2.0 * math.pi * K_E / slab_volume)
        z = pos @ normal
        zz = coefficient * z.unsqueeze(1) * z.unsqueeze(0)
        J = J + zz
        J_fb = J_fb + zz
        t01 = t01 + coefficient * z.view(n, 1, 1) * (ell.unsqueeze(-1) * normal).view(1, n, 3)
    blocks = [J_fb @ assembly._dense_incidence(), t01.reshape(n, 3 * n)]
    if t02 is not None:
        blocks.append(t02.reshape(n, 5 * n))
    return 0.5 * (J + J.transpose(0, 1)), torch.cat(blocks, dim=1)


def periodic_free_charge_blocks_ewald(
    assembly: ElectrostaticAssembly, sigma_free: Tensor
) -> tuple[Tensor, Tensor]:
    """``(J [n, n], A [n, m])`` under periodic boundaries by the Ewald split.

    The same blocks as :func:`periodic_free_charge_blocks` (that function's docstring has the
    formulas and the tinfoil / background convention), but with the width dependence in real space,
    so the reciprocal sum depends on alpha alone: compact (ionic) widths, which would need 143 687
    plane waves on the Au2/MgO cell against 1 654 for the hardness widths, cost nothing extra.
    ``trace_electrostatics.lattice.free_charge_lattice_blocks`` gives the pair parts; the same-image
    on-site terms are added in closed form (free-free ``gram_self_energy``, free-bound
    ``kernel_self``).  Equal to the plane-wave route to 1e-10 (tests/redox/test_redox_free_width.py).
    """
    from trace_electrostatics.kernels import gram_self_energy
    from trace_electrostatics.lattice import free_charge_lattice_blocks

    params, config = assembly.params, assembly.config
    pos, cell = assembly.pos, assembly.cell
    n = assembly.n_atoms
    sigma_mu, sigma_theta = gaussian_widths(params, config.gaussian_a)
    widths = [sigma_free, params.sigma, sigma_mu] + ([sigma_theta] if assembly.quadrupoles else [])
    floor = _gaussian_cutoff_floor(widths, config)
    if float(config.ewald_real_cutoff) < floor:
        config = dataclasses.replace(config, ewald_real_cutoff=floor)
    from . import nanprobe
    # the NaN probe (off unless TRACE_NANPROBE=1): a node of its own for sigma_f's lattice use, exact (x 1.0)
    s_lattice = nanprobe.watch("sigma_free->lattice", sigma_free * 1.0) if nanprobe.ENABLED else sigma_free
    blocks = free_charge_lattice_blocks(
        pos, cell, s_lattice, params.sigma, sigma_mu, params.ell, config.ewald_params(cell, assembly.pbc),
        sigma_theta=sigma_theta if assembly.quadrupoles else None,
    )
    for key in ("j", "j_fb", "t01"):
        nanprobe.watch(f"lattice.{key}", blocks[key])
    J = blocks["j"] + torch.diag(gram_self_energy(sigma_free))
    J_fb = blocks["j_fb"] + torch.diag(kernel_self(torch.sqrt(sigma_free * sigma_free + params.sigma * params.sigma)))
    t01, t02 = blocks["t01"], blocks.get("t02")
    if assembly.slab:
        normal, _height, slab_volume = _slab_geometry(pos, cell, assembly.slab_axis)
        coefficient = 2.0 * (2.0 * math.pi * K_E / slab_volume)
        z = pos @ normal
        zz = coefficient * z.unsqueeze(1) * z.unsqueeze(0)
        J = J + zz
        J_fb = J_fb + zz
        t01 = t01 + coefficient * z.view(n, 1, 1) * (params.ell.unsqueeze(-1) * normal).view(1, n, 3)
    columns = [J_fb @ assembly._dense_incidence(), t01.reshape(n, 3 * n)]
    if t02 is not None:
        columns.append(t02.reshape(n, 5 * n))
    return nanprobe.watch("free.J", 0.5 * (J + J.transpose(0, 1))), nanprobe.watch("free.A", torch.cat(columns, dim=1))


def free_cross_block(assembly: ElectrostaticAssembly, sigma_row: Tensor, sigma_col: Tensor) -> Tensor:
    """``M [n, n]``: Coulomb integrals of free Gaussians of widths ``sigma_row`` (rows) and ``sigma_col``
    (columns) on the atoms, every image and the same-atom (concentric) term included.

    Open boundaries: :func:`charge_coulomb_block`.  Periodic: the Ewald pair part of
    ``free_charge_lattice_blocks`` (its free-bound block has exactly this width rule, s^2 = s_row^2 +
    s_col^2) plus the same-image term in closed form, and the slab dipole term of
    :func:`periodic_free_charge_blocks_ewald` (it depends on the charges' heights only).
    """
    if not assembly.periodic:
        return charge_coulomb_block(assembly.pos, sigma_row, sigma_col)
    from trace_electrostatics.lattice import free_charge_lattice_blocks

    params, config = assembly.params, assembly.config
    pos, cell = assembly.pos, assembly.cell
    sigma_mu, _ = gaussian_widths(params, config.gaussian_a)
    floor = _gaussian_cutoff_floor([sigma_row, sigma_col, params.sigma, sigma_mu], config)
    if float(config.ewald_real_cutoff) < floor:
        config = dataclasses.replace(config, ewald_real_cutoff=floor)
    blocks = free_charge_lattice_blocks(pos, cell, sigma_row, sigma_col, sigma_mu, params.ell,
                                        config.ewald_params(cell, assembly.pbc))
    M = blocks["j_fb"] + torch.diag(kernel_self(torch.sqrt(sigma_row * sigma_row + sigma_col * sigma_col)))
    if assembly.slab:
        normal, _height, slab_volume = _slab_geometry(pos, cell, assembly.slab_axis)
        z = pos @ normal
        M = M + 2.0 * (2.0 * math.pi * K_E / slab_volume) * z.unsqueeze(1) * z.unsqueeze(0)
    return M


def eliminate_core_carrier(
    assembly: ElectrostaticAssembly, sigma_core: Tensor, sigma_carrier: Tensor, frames: Tensor, *,
    method: str = "condensed",
) -> BoundElimination:
    """The bound sector reduced onto core + carrier free charges, in the unit-charge basis (SPEC 17.9).

    Unit u carries its frame charge f_u on a compact Gaussian (``sigma_core``) and its carrier c_u = n_u - f_u
    on a diffuse one (``sigma_carrier``).  With x = [f; c] the free-sector interaction is
    u0_2 . x + 1/2 x^T W_2 x, W_2 = J_2 - A_2 K^-1 A_2^T.  For fixed frames this is, in n = f + c,

        b_n . n + 1/2 n^T W_cc n + C,      b_n = u0_c + W_fc^T f - W_cc f,
        C = u0_f . f + 1/2 f^T W_ff f - (u0_c + W_fc^T f) . f + 1/2 f^T W_cc f,

    returned as a :class:`BoundElimination` with u0 = b_n, W = W_cc, constant = C (e_b0 unchanged), J = the bare carrier
    block, A = A_c and y0 - K^-1 (A_f - A_c)^T f, so that ``bound_moments(n) = y0 - K^-1 A_2^T x``.
    The all-frame configuration n = f has energy u0_f . f + 1/2 f^T W_ff f: the single-width model's.
    """
    dtype = assembly.pos.dtype
    sigma_core, sigma_carrier = sigma_core.to(dtype), sigma_carrier.to(dtype)
    f = frames.to(dtype)
    core = eliminate_bound_sector(assembly, sigma_core, method=method,
                                  sigma_self=sigma_core if assembly.periodic else None)
    J_c, A_c = (periodic_free_charge_blocks_ewald if assembly.periodic else free_charge_blocks)(assembly, sigma_carrier)
    J_fc = free_cross_block(assembly, sigma_core, sigma_carrier)
    solve = core.operator.solve
    K_inv_Ac_T = solve(A_c.transpose(0, 1))
    W_cc = J_c - A_c @ K_inv_Ac_T
    W_cc = 0.5 * (W_cc + W_cc.transpose(0, 1))
    W_fc = J_fc - core.A @ K_inv_Ac_T
    u0_c = A_c @ core.y0
    if assembly.external_field is not None:
        u0_c = u0_c - assembly.pos @ assembly.external_field
    u0_f, W_ff = core.u0, core.W
    linear_c = u0_c + W_fc.transpose(0, 1) @ f
    b_n = linear_c - W_cc @ f
    constant = u0_f @ f + 0.5 * (f @ (W_ff @ f)) - linear_c @ f + 0.5 * (f @ (W_cc @ f))
    y0 = core.y0 - solve(((core.A - A_c).transpose(0, 1) @ f).unsqueeze(1)).squeeze(1)
    return BoundElimination(
        assembly=assembly, e_b0=core.e_b0, u0=b_n, W=W_cc, J=J_c, y0=y0, operator=core.operator,
        A=A_c, n_bonds=core.n_bonds, asymmetry=core.asymmetry, method=method, constant=constant,
    )


def on_site_self_width(
    assembly: ElectrostaticAssembly, J: Tensor, A: Tensor, sigma_free: Tensor, sigma_self: Tensor
) -> tuple[Tensor, Tensor]:
    """Give the same-atom, same-image terms the width ``sigma_self``; leave every other term alone.

    The on-site terms are energetic bookkeeping, not geometry: the fraction ``h`` of the atomic
    hardness that is classical Hartree energy (:class:`~trace_redox.heads.FreeChargeWidth`) and the
    free-bound coupling on one atom.  Interactions between different sites and images are the
    geometry, which the ionic width sets (audit step 2, B3).  The same-image term of a Gaussian pair
    is its ``r = 0`` kernel in the open and the periodic form alike, so the swap is exact:

        J_ii  += gram_self_energy(sigma_self) - gram_self_energy(sigma_free)
        J^fb_ii += kernel_self(sqrt(ss^2 + sb^2)) - kernel_self(sqrt(sf^2 + sb^2)),

    the second entering ``A = [J^fb B, T01, T02]`` through the incidence ``B``.  The charge-dipole
    and charge-quadrupole on-site terms vanish (odd kernel; traceless basis on an isotropic one).
    """
    from trace_electrostatics.kernels import gram_self_energy

    sb = assembly.params.sigma
    d_J = gram_self_energy(sigma_self) - gram_self_energy(sigma_free)
    d_fb = kernel_self(torch.sqrt(sigma_self * sigma_self + sb * sb)) - kernel_self(
        torch.sqrt(sigma_free * sigma_free + sb * sb)
    )
    nb = assembly.n_bonds
    bonds = A[:, :nb] + d_fb.unsqueeze(1) * assembly._dense_incidence()
    return J + torch.diag(d_J), torch.cat((bonds, A[:, nb:]), dim=1)


def _condensed_operator(assembly: ElectrostaticAssembly, symmetry_tol: float) -> tuple[_CondensedK, float]:
    """The blocks of K exactly as ``assembly.dense_hessian()`` builds them, minus the rho rows
    (the variational basis), symmetrised after the same round-off check."""
    n = assembly.n_atoms
    B = assembly._dense_incidence()
    H = assembly.j_pair + torch.diag(assembly.params.eta)
    T = assembly.t01_pair.reshape(n, 3 * n)
    P = assembly.t2_pair.permute(0, 2, 1, 3).reshape(3 * n, 3 * n) + torch.block_diag(*assembly.params.A_mu.unbind(0))
    if assembly.quadrupoles:
        n_theta = 5 * n
        T = torch.cat((T, assembly.t02_pair.reshape(n, n_theta)), dim=1)
        k_mu_theta = assembly.t12_pair.permute(0, 2, 1, 3).reshape(3 * n, n_theta)
        k_theta_theta = (assembly.t22_pair.permute(0, 2, 1, 3).reshape(n_theta, n_theta)
                         + torch.block_diag(*assembly.params.A_Theta.unbind(0)))
        P = torch.cat((torch.cat((P, k_mu_theta), dim=1),
                       torch.cat((k_mu_theta.transpose(0, 1), k_theta_theta), dim=1)), dim=0)
    asymmetry = 0.0
    for block in (H, P):
        scale = float(block.detach().abs().max())
        asymmetry = max(asymmetry, float((block - block.transpose(0, 1)).detach().abs().max()) / max(scale, 1e-300))
    if asymmetry > symmetry_tol:
        raise BoundSectorError(f"the bound operator is not symmetric: relative block asymmetry {asymmetry:.2e}")
    H = 0.5 * (H + H.transpose(0, 1))
    P = 0.5 * (P + P.transpose(0, 1))
    return _CondensedK(assembly.params.kappa, B, H, T, P), asymmetry


def eliminate_bound_sector(
    assembly: ElectrostaticAssembly, sigma_free: Tensor, *, symmetry_tol: float = 1e-10, method: str = "condensed",
    sigma_self: Tensor | None = None,
) -> BoundElimination:
    """Reduce the bound sector of ``assembly`` onto free Gaussians of widths ``sigma_free``.

    ``method='condensed'`` (default) inverts K through its SQE structure
    (:class:`_CondensedK`, exact); ``'dense'`` assembles and factorises K
    (the reference route, O((n_b + n_p)^3)).
    """
    _refuse_unsupported(assembly)
    n, nb = assembly.n_atoms, assembly.n_bonds
    dtype, device = assembly.pos.dtype, assembly.pos.device
    sigma_free = sigma_free.to(dtype)
    if sigma_free.shape != (n,) or bool((sigma_free.detach() <= 0).any()):
        raise BoundSectorError("sigma_free must be [n_atoms] and strictly positive")
    if method not in ("condensed", "dense"):
        raise ValueError(f"method must be 'condensed' or 'dense', got {method!r}")

    if method == "dense":
        # K in the variational basis, from the stored pair blocks.  dense_hessian()
        # never symmetrises (an asymmetric K is a bug it exists to expose); the
        # asymmetry is measured here and must be round-off.
        full = assembly.dense_hessian()
        index = _variational_indices(assembly)
        K = full.index_select(0, index).index_select(1, index)
        scale = float(K.detach().abs().max())
        asymmetry = float((K - K.transpose(0, 1)).detach().abs().max()) / max(scale, 1e-300)
        if asymmetry > symmetry_tol:
            raise BoundSectorError(f"the bound operator is not symmetric: max|K - K^T| / max|K| = {asymmetry:.2e}")
        K = 0.5 * (K + K.transpose(0, 1))
        factor, info = torch.linalg.cholesky_ex(K)
        if int(info) != 0:
            raise BoundSectorError(f"the bound operator K is not positive definite (Cholesky failed at column {int(info)})")
        operator = _DenseK(K, factor)
    else:
        operator, asymmetry = _condensed_operator(assembly, symmetry_tol)

    # Linear term and constant of U_b(y) at rho = 0, by autograd on the
    # assembly's own energy (one functional), keeping the graph.
    m = nb + int(operator.P.shape[0]) if method == "condensed" else int(operator.K.shape[0])
    zero_rho = torch.zeros(n, dtype=dtype, device=device)

    def energy_at(vector: Tensor) -> Tensor:
        theta = vector[nb + 3 * n:].reshape(n, 5) if assembly.quadrupoles else None
        return assembly.energy(vector[:nb], zero_rho, vector[nb:nb + 3 * n].reshape(n, 3), theta)

    with torch.enable_grad():
        y = torch.zeros(m, dtype=dtype, device=device, requires_grad=True)
        (g,) = torch.autograd.grad(energy_at(y), y, create_graph=True)
    e0 = energy_at(torch.zeros(m, dtype=dtype, device=device))
    y0 = -operator.solve(g.unsqueeze(1)).squeeze(1)
    e_b0 = e0 + 0.5 * (g @ y0)

    if sigma_self is None:
        J, A = (periodic_free_charge_blocks if assembly.periodic else free_charge_blocks)(assembly, sigma_free)
    else:
        # ``sigma_self`` given (width='ionic'): pair terms at sigma_free, same-image on-site terms at
        # sigma_self; under periodic boundaries by the Ewald split, whose cost does not grow as the
        # widths shrink.
        sigma_self = sigma_self.to(dtype)
        if sigma_self.shape != (n,) or bool((sigma_self.detach() <= 0).any()):
            raise BoundSectorError("sigma_self must be [n_atoms] and strictly positive")
        J, A = (periodic_free_charge_blocks_ewald if assembly.periodic else free_charge_blocks)(assembly, sigma_free)
        J, A = on_site_self_width(assembly, J, A, sigma_free, sigma_self)
    if int(A.shape[1]) != m:
        raise BoundSectorError(f"coupling has {int(A.shape[1])} columns, K has {m}")
    u0 = A @ y0
    if assembly.external_field is not None:
        u0 = u0 - assembly.pos @ assembly.external_field
    W = J - A @ operator.solve(A.transpose(0, 1))
    W = 0.5 * (W + W.transpose(0, 1))
    return BoundElimination(
        assembly=assembly, e_b0=e_b0, u0=u0, W=W, J=J, y0=y0, operator=operator, A=A,
        n_bonds=nb, asymmetry=asymmetry, method=method,
    )


def assemble_neutral(h: Tensor, pos: Tensor, edge_index: Tensor, edge_vec: Tensor, edge_len: Tensor,
                     heads: Any, config: Any, *, elimination: str | None = None, **keywords: Any) -> ElectrostaticAssembly:
    """trace_electrostatics.assemble with every bound fragment neutral, on the dense route.

    ``elimination`` names the route the caller takes next (``eliminate_bound_sector``'s ``method``), so that
    ``config.max_operator_bytes`` bounds what that route holds (benchmarks_4g README R25):

    * ``None`` (legacy, every caller that does not name it) and ``'dense'``: trace_electrostatics' own guard, the
      dense n_m x n_m operator K, n_m = n_bonds + 3 n (+ 5 n with quadrupoles);
    * ``'condensed'``: that K is never formed (:class:`_CondensedK`).  The route's largest operator arrays are
      [n, n_m] (the incidence, the coupling A, K^-1 A^T), and the limit is compared with one of them.  The bound on the
      quadrupole pair tensors, which the assembly itself allocates, is kept as trace_electrostatics states it.
    """
    from trace_electrostatics.energy import assemble

    if elimination not in (None, "condensed", "dense"):
        raise ValueError(f"elimination must be None, 'condensed' or 'dense', got {elimination!r}")
    if config.solver != "dense":
        config = dataclasses.replace(config, solver="dense")
    for name in ("fragment_id", "fragment_charges", "fragment_spins", "metallic",
                 "electrode_potential", "structure_id"):
        if keywords.get(name) is not None:
            raise BoundSectorError(f"{name} is not supported by trace_redox v1")
        keywords.pop(name, None)
    keywords["total_charge"] = 0.0
    if elimination != "condensed":
        return assemble(h, pos, edge_index, edge_vec, edge_len, heads, config, **keywords)

    limit = float(config.max_operator_bytes)
    n, item = int(pos.shape[0]), torch.finfo(pos.dtype).bits // 8
    if keywords.get("quadrupoles"):                                   # the assembly's own allocation: [n, n, 81], ~4 copies
        quadrupole_bytes = 4 * 81 * n * n * item
        if quadrupole_bytes > limit:
            raise BoundSectorError(
                f"the quadrupole pair tensors for {n} atoms need ~{quadrupole_bytes / 1e9:.2f} GB, above "
                f"max_operator_bytes={limit / 1e9:.2f} GB. Raise the limit deliberately.")
    assembly = assemble(h, pos, edge_index, edge_vec, edge_len, heads,
                        dataclasses.replace(config, max_operator_bytes=math.inf), **keywords)
    n_m = assembly.n_bonds + (8 if assembly.quadrupoles else 3) * n
    held = n * n_m * item
    if held > limit:
        raise BoundSectorError(
            f"the condensed elimination holds operator arrays of [{n}, {n_m}] ({held / 1e9:.2f} GB each; "
            f"n_bonds={assembly.n_bonds} for {n} atoms), above max_operator_bytes={limit / 1e9:.2f} GB. Raise the limit "
            f"deliberately. (The dense {n_m}x{n_m} operator K is never built on this route.)")
    return assembly
