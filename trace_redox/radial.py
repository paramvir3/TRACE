"""Spherical charge distributions on a radial grid (numpy, float64): the numerics of trace_redox SPEC 23, torch-free
so that tools/electrostatics/build_ion_densities.py can use them.  See trace_redox/penetration.py for the physics."""
from __future__ import annotations

import numpy as np


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))

def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def electron_count(r: np.ndarray, rho: np.ndarray) -> float:
    return _trapezoid(4.0 * np.pi * r * r * rho, r)


def potential(r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """phi(d) on the grid: the electron potential of a spherical density (positive, hartree per unit charge)."""
    inner = _cumtrapz(4.0 * np.pi * r * r * rho, r)
    outer = _cumtrapz(4.0 * np.pi * r * rho, r)
    return inner / r + (outer[-1] - outer)


class RadialDensity:
    """A spherical electron density on a radial grid with its potential and the shell integral W, precomputed."""

    def __init__(self, r: np.ndarray, rho: np.ndarray) -> None:
        self.r, self.rho = r, rho
        self.N = electron_count(r, rho)
        self.phi = potential(r, rho)
        self.W = _cumtrapz(self.phi * r, r)

    def phi_at(self, d: float) -> float:
        if d >= self.r[-1]:
            return self.N / d
        return float(np.interp(d, self.r, self.phi))

    def _W_at(self, x: np.ndarray) -> np.ndarray:
        out = np.interp(x, self.r, self.W)
        over = x > self.r[-1]
        out[over] = self.W[-1] + self.N * (x[over] - self.r[-1])            # phi = N / d beyond the grid
        return out

    def coulomb(self, other: "RadialDensity", R: float) -> float:
        """(other | self)(R): the Coulomb energy of the two densities (hartree), shells of ``other`` averaged."""
        r = other.r
        shell = (self._W_at(R + r) - self._W_at(np.abs(R - r))) / (2.0 * r * R)
        return _trapezoid(4.0 * np.pi * r * r * other.rho * shell, r)


def pair_electrostatic_energy(Za: int, a: RadialDensity | None, Zb: int, b: RadialDensity | None, R: float) -> float:
    """E_es(R) of two spherical ions (hartree; R in bohr); a density None is a bare nucleus (no electrons)."""
    e = Za * Zb / R
    if b is not None:
        e -= Za * b.phi_at(R)
    if a is not None:
        e -= Zb * a.phi_at(R)
    if a is not None and b is not None:
        e += b.coulomb(a, R)
    return e


def chebyshev_nodes(n: int, lo: float, hi: float) -> np.ndarray:
    k = np.arange(n)
    return 0.5 * (hi + lo) + 0.5 * (hi - lo) * np.cos(np.pi * (k + 0.5) / n)


def chebyshev_fit(values: np.ndarray, n: int) -> np.ndarray:
    """Coefficients c_m of sum_m c_m T_m(x) interpolating ``values`` at the n Chebyshev nodes (first kind)."""
    k = np.arange(n)
    theta = np.pi * (k + 0.5) / n
    c = np.array([2.0 / n * np.sum(values * np.cos(m * theta)) for m in range(n)])
    c[0] *= 0.5
    return c




# ---- the penetration difference without large cancellations (SPEC 23) -----------------------------------------
#
# With delta = rho(n) - rho(f) (an electron-density difference, N_delta = -(n - f) electrons) and the frame ion's
# potential V^f(d) = Z/d - phi^f(d) = q_f / d + psi^f(d), where
#
#     psi(d) = N/d - phi(d) = int_d^inf 4 pi r^2 rho(r) (1/d - 1/r) dr  >= 0      (from the tail alone: no cancellation)
#
# the penetration D = [E_es(n_a, n_b) - E_es(f_a, f_b)] - (n_a n_b - f_a f_b) / R splits exactly into short-range
# pieces (atomic units):
#
#     D = - q_b I_a(R) - S_a[psi_b^f](R) - q_a I_b(R) - S_b[psi_a^f](R) + N_db I_a(R) - S_a[psi_db](R)
#     I_a(R)     = int_{r > R} 4 pi r^2 delta_a(r) (1/r - 1/R) dr
#     S_a[g](R)  = int 4 pi r^2 delta_a(r) <g>(r; R) dr,   <g>(r; R) = [G(R + r) - G(|R - r|)] / (2 r R),  G' = g d
#
# (q the frame charges f; the last pair is (delta_a | delta_b) - N_da N_db / R).  The absolute errors then scale with
# the penetration itself, not with Z_a Z_b / R.


def _tail(r: np.ndarray, rho: np.ndarray):
    """(A(d), B(d)) = (int_d^inf 4 pi r^2 rho, int_d^inf 4 pi r rho) on the grid."""
    a = _cumtrapz(4.0 * np.pi * r * r * rho, r)
    b = _cumtrapz(4.0 * np.pi * r * rho, r)
    return a[-1] - a, b[-1] - b


def psi(r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """psi(d) = N/d - phi(d) >= 0 for a density rho (any sign pattern; short-ranged)."""
    A, B = _tail(r, rho)
    return A / r - B


class ShellAverage:
    """<g>(r; R) for a radial function g on the grid, through G(x) = int_0^x g(d) d dd (constant beyond the grid:
    g is short-ranged here)."""

    def __init__(self, r: np.ndarray, g: np.ndarray) -> None:
        self.r = r
        self.G = _cumtrapz(g * r, r)

    def __call__(self, R: float) -> np.ndarray:
        r = self.r
        hi = np.interp(R + r, r, self.G, right=self.G[-1])
        lo = np.interp(np.abs(R - r), r, self.G, right=self.G[-1])
        return (hi - lo) / (2.0 * r * R)


def penetration_difference(r: np.ndarray, Za: int, fa_rho: np.ndarray, na_rho: np.ndarray, qa: int,
                           Zb: int, fb_rho: np.ndarray, nb_rho: np.ndarray, qb: int, R: float) -> float:
    """D (hartree) of the module docstring for ions a, b with frame densities ``f*_rho`` (frame charges qa, qb) and
    state densities ``n*_rho``, at distance R (bohr)."""
    da, db = na_rho - fa_rho, nb_rho - fb_rho
    w = 4.0 * np.pi * r * r
    out = 0.0
    N_db = _trapezoid(w * db, r)
    shells = ((da, fb_rho, qb), (db, fa_rho, qa))
    for delta, frame_other, q_other in shells:
        if not np.any(delta):
            continue
        far = r > R
        I = _trapezoid(np.where(far, w * delta * (1.0 / r - 1.0 / R), 0.0), r)
        S = _trapezoid(w * delta * ShellAverage(r, psi(r, frame_other))(R), r) if np.any(frame_other) else 0.0
        out += -q_other * I - S
    if np.any(da) and np.any(db):
        far = r > R
        I_a = _trapezoid(np.where(far, w * da * (1.0 / r - 1.0 / R), 0.0), r)
        out += N_db * I_a - _trapezoid(w * da * ShellAverage(r, psi(r, db))(R), r)
    return out


# ---- the Gordon-Kim electron-gas part of the overlap (SPEC 25) ------------------------------------------------
#
# Electron-gas energy of the summed densities minus the parts (Gordon & Kim, J. Chem. Phys. 56, 3122 (1972)),
# state-dependent part relative to the frames:
#
#     G = sum_f { int f[rho_a^n + rho_b^n] - f[rho_a^n] - f[rho_b^n] } - { same at the frames },
#     f = C_F rho^(5/3) + (1/9) |grad rho|^2 / (8 rho)            (Thomas-Fermi + the second-order gradient term)
#         - C_x rho^(4/3) - 0.44 rho / (7.8 + r_s)                  (LDA exchange, Wigner correlation)
#
# C_F = (3/10)(3 pi^2)^(2/3), C_x = (3/4)(3/pi)^(1/3), r_s = (3 / (4 pi rho))^(1/3); atomic units.  The gradient of
# the sum combines the two radial gradients at the angle between r_a and r_b: |grad|^2 = a'^2 + b'^2 + 2 a' b' cos,
# cos = (r_a^2 + r_b^2 - R^2) / (2 r_a r_b).  Prolate spheroidal grid: r_a = R (mu + nu)/2, r_b = R (mu - nu)/2,
# dV = 2 pi (R/2)^3 (mu^2 - nu^2) dmu dnu, mu - 1 log-spaced to 60 / R, nu on Gauss-Legendre nodes.  The integrand
# is formed pointwise as the difference, so no large core energy is ever subtracted.

C_F = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
C_X = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
VW_FRACTION = 1.0 / 9.0


def _gas_density(rho: np.ndarray, grad2: np.ndarray) -> np.ndarray:
    rho = np.maximum(rho, 0.0)
    safe = np.maximum(rho, 1e-300)
    rs = (3.0 / (4.0 * np.pi * safe)) ** (1.0 / 3.0)
    value = C_F * rho ** (5.0 / 3.0) - C_X * rho ** (4.0 / 3.0) - 0.44 * rho / (7.8 + rs)
    value = value + VW_FRACTION * grad2 / (8.0 * safe)
    return np.where(rho > 1e-300, value, 0.0)


class RadialProfile:
    """ln rho and d ln rho / d ln r on the grid, interpolated in ln r (smooth for exponential tails)."""

    def __init__(self, r: np.ndarray, rho: np.ndarray) -> None:
        self.lr = np.log(r)
        self.zero = not np.any(rho > 0.0)
        self.lrho = np.log(np.maximum(rho, 1e-300))
        self.dl = np.gradient(self.lrho, self.lr)

    def __call__(self, d: np.ndarray):
        if self.zero:
            z = np.zeros_like(d)
            return z, z
        x = np.log(d)
        rho = np.exp(np.interp(x, self.lr, self.lrho))
        return rho, rho * np.interp(x, self.lr, self.dl) / d


def gordon_kim_difference(r: np.ndarray, fa_rho: np.ndarray, na_rho: np.ndarray, fb_rho: np.ndarray,
                          nb_rho: np.ndarray, R: float, n_mu: int = 450, n_nu: int = 200) -> float:
    """G (hartree) of the block comment for ions a, b at distance R (bohr), frames f*, states n*."""
    t = np.exp(np.linspace(np.log(1e-7), np.log(60.0 / R), n_mu))
    w_t = np.gradient(t)
    x, wx = np.polynomial.legendre.leggauss(n_nu)
    MU, NU = np.meshgrid(1.0 + t, x, indexing="ij")
    ra = np.maximum(R * (MU + NU) / 2.0, 1e-12)
    rb = np.maximum(R * (MU - NU) / 2.0, 1e-12)
    cos = np.clip((ra * ra + rb * rb - R * R) / (2.0 * ra * rb), -1.0, 1.0)
    dV = 2.0 * np.pi * (R / 2.0) ** 3 * (MU * MU - NU * NU) * w_t[:, None] * wx[None, :]
    total = np.zeros_like(MU)
    for a_rho, b_rho, sign in ((na_rho, nb_rho, 1.0), (fa_rho, fb_rho, -1.0)):
        A, dA = RadialProfile(r, a_rho)(ra)
        Bv, dB = RadialProfile(r, b_rho)(rb)
        total += sign * (_gas_density(A + Bv, dA * dA + dB * dB + 2.0 * dA * dB * cos)
                         - _gas_density(A, dA * dA) - _gas_density(Bv, dB * dB))
    return float(np.sum(total * dV))


def gordon_kim_table(r: np.ndarray, densities: dict, frames: dict, keys: list, R_nodes: np.ndarray,
                     n_mu: int = 450, n_nu: int = 200) -> np.ndarray:
    """``G [len(keys), len(R_nodes)]`` (hartree) for pair types (Za, na, Zb, nb), frames {Z: f}, densities {(Z, n): rho},
    at R_nodes (bohr): the same integrand as :func:`gordon_kim_difference`, with each node's grid and every density's
    values on it computed once."""
    profiles = {k: RadialProfile(r, v) for k, v in densities.items()}
    out = np.zeros((len(keys), len(R_nodes)))
    x, wx = np.polynomial.legendre.leggauss(n_nu)
    for j, R in enumerate(R_nodes):
        t = np.exp(np.linspace(np.log(1e-7), np.log(60.0 / R), n_mu))
        w_t = np.gradient(t)
        MU, NU = np.meshgrid(1.0 + t, x, indexing="ij")
        ra = np.maximum(R * (MU + NU) / 2.0, 1e-12)
        rb = np.maximum(R * (MU - NU) / 2.0, 1e-12)
        cos = np.clip((ra * ra + rb * rb - R * R) / (2.0 * ra * rb), -1.0, 1.0)
        dV = 2.0 * np.pi * (R / 2.0) ** 3 * (MU * MU - NU * NU) * w_t[:, None] * wx[None, :]
        on_a, on_b = {}, {}
        self_a, self_b = {}, {}

        def at_a(k):
            if k not in on_a:
                on_a[k] = profiles[k](ra)
                self_a[k] = _gas_density(on_a[k][0], on_a[k][1] ** 2)
            return on_a[k], self_a[k]

        def at_b(k):
            if k not in on_b:
                on_b[k] = profiles[k](rb)
                self_b[k] = _gas_density(on_b[k][0], on_b[k][1] ** 2)
            return on_b[k], self_b[k]

        for i, (Za, na, Zb, nb) in enumerate(keys):
            total = np.zeros_like(MU)
            for ka, kb, sign in (((Za, na), (Zb, nb), 1.0), ((Za, frames[Za]), (Zb, frames[Zb]), -1.0)):
                (A, dA), sa = at_a(ka)
                (Bv, dB), sb = at_b(kb)
                total += sign * (_gas_density(A + Bv, dA * dA + dB * dB + 2.0 * dA * dB * cos) - sa - sb)
            out[i, j] = float(np.sum(total * dV))
    return out


def density_overlap_difference(r: np.ndarray, fa_rho: np.ndarray, na_rho: np.ndarray, fb_rho: np.ndarray,
                               nb_rho: np.ndarray, R: float) -> float:
    """dS = S(rho_a^n, rho_b^n) - S(rho_a^f, rho_b^f), S(x, y)(R) = int x(|r|) y(|r - R|) d^3r (bohr^-3): bilinear, so
    dS = (delta_a | rho_b^f) + (rho_a^f | delta_b) + (delta_a | delta_b), each by the shell average of the partner."""
    w = 4.0 * np.pi * r * r
    da, db = na_rho - fa_rho, nb_rho - fb_rho
    out = 0.0
    for left, right in ((da, fb_rho), (fa_rho, db), (da, db)):
        if np.any(left) and np.any(right):
            out += _trapezoid(w * left * ShellAverage(r, right)(R), r)
    return out


def density_overlap_parts(r: np.ndarray, fa_rho: np.ndarray, na_rho: np.ndarray, fb_rho: np.ndarray,
                          nb_rho: np.ndarray, R: float):
    """The three bilinear pieces of :func:`density_overlap_difference`: ((delta_a | rho_b^f), (rho_a^f | delta_b),
    (delta_a | delta_b)), bohr^-3."""
    w = 4.0 * np.pi * r * r
    da, db = na_rho - fa_rho, nb_rho - fb_rho
    parts = []
    for left, right in ((da, fb_rho), (fa_rho, db), (da, db)):
        parts.append(_trapezoid(w * left * ShellAverage(r, right)(R), r) if (np.any(left) and np.any(right)) else 0.0)
    return tuple(parts)


# ---- the carrier's field on a smeared moment (SPEC 26) ---------------------------------------------------------
#
# Gauss: the field at distance R of a spherical charge distribution is K Q_inside(R) / R^2.  For a moment smeared as
# a Gaussian exp(-x^2 / w^2) the relevant quantity is the fraction of the carrier, convolved with that Gaussian, inside
# R.  A point at distance r from the centre, smeared, lies inside R with probability
#
#     F(r; R, w) = 1/2 [erf((R - r)/w) + erf((R + r)/w)] + (w / (2 sqrt(pi) r)) [e^-((R+r)/w)^2 - e^-((R-r)/w)^2]
#
# (-> P(3/2, R^2/w^2) as r -> 0), so the field factor is lambda(R, w) = int 4 pi r^2 delta(r) F dr / int 4 pi r^2 delta.
# A Gaussian carrier of width sigma gives P(3/2, R^2 / (sigma^2 + w^2)), the model's lambda_3: the correction vanishes.


def _inside_probability(r: np.ndarray, R: float, w: float) -> np.ndarray:
    from scipy.special import erf
    out = 0.5 * (erf((R - r) / w) + erf((R + r) / w))
    return out + w / (2.0 * np.sqrt(np.pi) * np.maximum(r, 1e-300)) * (np.exp(-((R + r) / w) ** 2) - np.exp(-((R - r) / w) ** 2))


def field_factor(r: np.ndarray, delta: np.ndarray, R: float, w: float) -> float:
    """lambda(R, w) of the block comment for a carrier density difference ``delta`` (any sign; normalised by its own
    integral), R and w in the units of r."""
    weight = 4.0 * np.pi * r * r * delta
    total = _trapezoid(weight, r)
    if total == 0.0:
        return 0.0
    return _trapezoid(weight * _inside_probability(r, R, w), r) / total


#: Field-factor tables (SPEC 26): R in [R_MIN, R_CUT] A, the moment width w in [W_MIN, W_MAX] A.
FIELD_W_RANGE = (0.2, 2.5)
FIELD_N_R, FIELD_N_W = 64, 16


def chebyshev_fit_2d(values: np.ndarray) -> np.ndarray:
    """Coefficients c[m, k] of sum c T_m(x) T_k(y) interpolating ``values[i, j]`` at the Chebyshev nodes in x (rows)
    and y (columns)."""
    nr, nw = values.shape
    theta_r = np.pi * (np.arange(nr) + 0.5) / nr
    theta_w = np.pi * (np.arange(nw) + 0.5) / nw
    Tr = np.cos(np.outer(np.arange(nr), theta_r))                       # [m, i]
    Tw = np.cos(np.outer(np.arange(nw), theta_w))                       # [k, j]
    c = (2.0 / nr) * (2.0 / nw) * Tr @ values @ Tw.T
    c[0, :] *= 0.5
    c[:, 0] *= 0.5
    return c
