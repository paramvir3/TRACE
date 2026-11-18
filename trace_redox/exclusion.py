"""Exclusion (Pauli) of charge-state densities by closed shells (trace_redox SPEC 18; theory note 2026-10-15).

Frontier density of a charge state, exact in its asymptotic tail and finite at the nucleus (atomic units):

    rho(r) = (r^2 + kappa^-2)^beta exp(-2 kappa r) / Z(kappa, beta),   kappa = sqrt(2 I),   beta = (q + 1)/kappa - 1

(the r^beta exp(-kappa r) form of the outermost Dyson orbital: Katriel & Davidson, PNAS 77, 4403 (1980); I the
removal energy, q + 1 the charge left behind).  Exclusion energy between closed shells: E = K S, S = int rho_a rho_b
(density-overlap model: Kim et al.; Wheatley & Price, Mol. Phys. 69, 507 (1990); Soederhjelm, Karlstroem & Ryde,
J. Chem. Phys. 124, 244101 (2006)).

The overlap is evaluated in prolate spheroidal coordinates,

    r_a = R (xi + eta) / 2,   r_b = R (xi - eta) / 2,   d^3r = (R^3 / 8)(xi^2 - eta^2) dxi deta dphi,

with xi - 1 = t / (R (kappa_a + kappa_b)) on Gauss-Laguerre nodes and eta on composite Gauss-Legendre panels; every
exponent is combined into exp(-R (kappa_a + kappa_b) xi - R (kappa_a - kappa_b) eta) <= 1, so nothing overflows.
Fixed nodes: differentiable in R, kappa and beta.  Accuracy against an adaptive integral: tests/redox/
test_redox_exclusion_overlap.py.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

#: Gauss-Laguerre nodes for xi (decay e^-t) and for the radial normalisation.  24 x (8 x 6) = 1152 nodes: worst
#: relative error 6.4e-7 against an adaptive integral over Au-, Au0, Al2+, O2- and cation-core densities at
#: 1.8-6 A (measured 2026-10-15; 48 x (16 x 8): 2.6e-11, 5.3x the cost).
_N_LAGUERRE = 24
_N_LAGUERRE_NORM = 96
#: Composite Gauss-Legendre in eta: panels x points.  Each panel spans 2/_N_PANELS of eta; the eta exponent
#: R |kappa_a - kappa_b| eta varies by at most R |dk| * 2 / _N_PANELS per panel.
_N_PANELS = 8
_N_LEGENDRE = 6


def _laguerre(n: int):
    x, w = np.polynomial.laguerre.laggauss(n)
    return x, w


def _eta_nodes():
    x, w = np.polynomial.legendre.leggauss(_N_LEGENDRE)
    edges = np.linspace(-1.0, 1.0, _N_PANELS + 1)
    xs, ws = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        xs.append(0.5 * (hi - lo) * x + 0.5 * (hi + lo))
        ws.append(0.5 * (hi - lo) * w)
    return np.concatenate(xs), np.concatenate(ws)


_T, _WT = _laguerre(_N_LAGUERRE)
_TN, _WTN = _laguerre(_N_LAGUERRE_NORM)
_ETA, _WETA = _eta_nodes()


def radial_normalisation(kappa: Tensor, beta: Tensor) -> Tensor:
    """``Z = 4 pi int_0^inf r^2 (r^2 + kappa^-2)^beta exp(-2 kappa r) dr``

    = (pi / (2 kappa^3)) kappa^(-2 beta) int_0^inf t^2 (1 + t^2/4)^beta e^-t dt   (t = 2 kappa r)."""
    t = torch.as_tensor(_TN, dtype=kappa.dtype, device=kappa.device)
    w = torch.as_tensor(_WTN, dtype=kappa.dtype, device=kappa.device)
    integral = (w * t * t * (1.0 + 0.25 * t * t) ** beta.unsqueeze(-1)).sum(-1)
    return (math.pi / 2.0) * kappa ** (-3.0 - 2.0 * beta) * integral


def density_overlap(kappa_a: Tensor, beta_a: Tensor, kappa_b: Tensor, beta_b: Tensor, R: Tensor) -> Tensor:
    """``S(R) = int rho_a(|r|) rho_b(|r - R|) d^3r`` of two unit-normalised frontier densities (bohr^-3 * bohr^3)."""
    kappa_a, beta_a, kappa_b, beta_b, R = torch.broadcast_tensors(kappa_a, beta_a, kappa_b, beta_b, R)
    dtype, device = R.dtype, R.device
    t = torch.as_tensor(_T, dtype=dtype, device=device)                      # [L]
    wt = torch.as_tensor(_WT, dtype=dtype, device=device)
    eta = torch.as_tensor(_ETA, dtype=dtype, device=device)                  # [M]
    weta = torch.as_tensor(_WETA, dtype=dtype, device=device)
    c = R * (kappa_a + kappa_b)                                              # [...]
    xi = 1.0 + t / c.unsqueeze(-1)                                           # [..., L]
    X = xi.unsqueeze(-2)                                                     # [..., 1, L]
    E = eta.view(*([1] * R.dim()), -1, 1)                                    # [..., M, 1]
    Rx = R.unsqueeze(-1).unsqueeze(-1)
    ra = 0.5 * Rx * (X + E)
    rb = 0.5 * Rx * (X - E)
    ka, kb = kappa_a.unsqueeze(-1).unsqueeze(-1), kappa_b.unsqueeze(-1).unsqueeze(-1)
    ba, bb = beta_a.unsqueeze(-1).unsqueeze(-1), beta_b.unsqueeze(-1).unsqueeze(-1)
    # the combined exponent -2 ka ra - 2 kb rb = -c xi - R (ka - kb) eta; the e^-t of the Laguerre weight is
    # e^{-c (xi - 1)}, so what remains is exp(-c - R (ka - kb) eta) <= 1
    exponent = -c.unsqueeze(-1).unsqueeze(-1) - Rx * (ka - kb) * E
    integrand = (X * X - E * E) * (ra * ra + ka ** -2) ** ba * (rb * rb + kb ** -2) ** bb * torch.exp(exponent)
    inner = (integrand * wt).sum(-1)                                         # over t
    total = (inner * weta).sum(-1)                                           # over eta
    prefactor = 2.0 * math.pi * R ** 3 / 8.0 / c
    return prefactor * total / (radial_normalisation(kappa_a, beta_a) * radial_normalisation(kappa_b, beta_b))


# ---- the exclusion term of the configuration energies (SPEC 18) -------------------------------------------------

HARTREE_EV = 27.211386245988          #: CODATA 2018
BOHR_ANGSTROM = 0.529177210903        #: CODATA 2018


# ---- tabulated overlaps (SPEC 18.3) -----------------------------------------------------------------------------

#: Chebyshev expansion of log S on R in [_R_MIN, r_max] (A; _N_R terms) and, for an anion neighbour, kappa_v =
#: sqrt(2 I_eff) over Exclusion.I_EFF_RANGE (_N_KAPPA terms).  log S is analytic there (the quadrature uses fixed,
#: R-scaled nodes), so the series converges spectrally: 96 x 20 terms reproduce the quadrature to ~1e-12 relative
#: (measured 2026-10-15, Au-, Au0, Mg+, Al2+ against O2- over 0.8-5 A and I_eff 4-16 eV), and every R-derivative is
#: smooth.  Coefficients come from values at Chebyshev-Gauss nodes only.
_R_MIN = 0.2
_N_R = 96
_N_KAPPA = 40
_I_RANGE = (1.0, 40.0)
_TABLE_CACHE: dict = {}


def _chebyshev_nodes(n: int) -> np.ndarray:
    return np.cos(np.pi * (np.arange(n) + 0.5) / n)


def _chebyshev_basis(x: Tensor, n: int) -> Tensor:
    """``[..., n]``: T_0(x) .. T_{n-1}(x) by the three-term recurrence (differentiable to every order)."""
    T = [torch.ones_like(x), x]
    for _ in range(2, n):
        T.append(2.0 * x * T[-1] - T[-2])
    return torch.stack(T[:n], dim=-1)


def _log_overlap_coefficients(ku: float, bu: float, kv, bv: float, r_lo: float, r_hi: float, k_lo: float,
                              k_hi: float) -> np.ndarray:
    """``[3, N_R, N_K]``: Chebyshev coefficients of log S and of its first and second R-derivatives (per A, per A^2).
    ``kv`` a float: fixed neighbour (N_K = 1); None: anion neighbour, kappa_v on [k_lo, k_hi], beta_v = 0."""
    xr = _chebyshev_nodes(_N_R)
    R = 0.5 * (r_hi + r_lo) + 0.5 * (r_hi - r_lo) * xr
    if kv is None:
        xk = _chebyshev_nodes(_N_KAPPA)
        K = 0.5 * (k_hi + k_lo) + 0.5 * (k_hi - k_lo) * xk
        RR, KK = np.meshgrid(R, K, indexing="ij")
        S = density_overlap(torch.tensor(ku), torch.tensor(bu), torch.tensor(KK), torch.zeros(KK.shape, dtype=torch.float64),
                            torch.tensor(RR) / BOHR_ANGSTROM).numpy()
        TK = np.polynomial.chebyshev.chebvander(xk, _N_KAPPA - 1)
    else:
        S = density_overlap(torch.tensor(ku), torch.tensor(bu), torch.tensor(float(kv)), torch.tensor(float(bv)),
                            torch.tensor(R) / BOHR_ANGSTROM).numpy()[:, None]
        TK = np.ones((1, 1))
    TR = np.polynomial.chebyshev.chebvander(xr, _N_R - 1)
    C = np.linalg.solve(TR, np.linalg.solve(TK, np.log(S).T).T)                  # log S = TR C TK^T
    scale = 2.0 / (r_hi - r_lo)
    C1 = np.polynomial.chebyshev.chebder(C, axis=0) * scale
    C2 = np.polynomial.chebyshev.chebder(C, m=2, axis=0) * scale * scale
    pad = lambda M: np.concatenate([M, np.zeros((_N_R - M.shape[0], M.shape[1]))], axis=0)
    return np.stack([C, pad(C1), pad(C2)])


class OverlapTable:
    """``S(R)`` of a carrier state against a closed-shell neighbour, as a Chebyshev series per pair type (SPEC 18.3).

    The overlap of two ladder densities depends on the distance and, for an anion neighbour, on its learned I_eff
    only.  ``log S`` is expanded in Chebyshev polynomials of R (and of kappa_v = sqrt(2 I_eff)) from its values at
    Chebyshev-Gauss nodes; the R-derivatives are the derivative series.  Outside [0.2 A, r_max] the quadrature is used
    directly for values (and refused for derivatives).  Coefficients are built once per process and pair type."""

    def __init__(self, r_max: float, dtype=torch.float64) -> None:
        self.r_lo, self.r_hi = _R_MIN, float(r_max)
        self.k_lo, self.k_hi = (math.sqrt(2.0 * i / HARTREE_EV) for i in _I_RANGE)
        self.dtype = dtype
        self._index: dict = {}
        self._params: list = []                       # (ku, bu, kv or nan, bv, anion)
        self._coefficients: list = []                 # [3, N_R, N_K or 1] per type

    def _add(self, key, params, build):
        if key in self._index:
            return self._index[key]
        cache_key = (key, self.r_lo, self.r_hi, _N_R, _N_KAPPA)
        if cache_key not in _TABLE_CACHE:
            _TABLE_CACHE[cache_key] = torch.tensor(build(), dtype=self.dtype)
        self._index[key] = len(self._params)
        self._params.append(params)
        self._coefficients.append(_TABLE_CACHE[cache_key])
        return self._index[key]

    def anion_type(self, ku: float, bu: float) -> int:
        key = ("anion", float(ku), float(bu))
        return self._add(key, (float(ku), float(bu), float("nan"), 0.0, True),
                         lambda: _log_overlap_coefficients(float(ku), float(bu), None, 0.0, self.r_lo, self.r_hi,
                                                           self.k_lo, self.k_hi))

    def core_type(self, ku: float, bu: float, kv: float, bv: float) -> int:
        key = ("core", float(ku), float(bu), float(kv), float(bv))
        return self._add(key, (float(ku), float(bu), float(kv), float(bv), False),
                         lambda: _log_overlap_coefficients(float(ku), float(bu), float(kv), float(bv), self.r_lo,
                                                           self.r_hi, self.k_lo, self.k_hi))

    def overlap(self, tid: Tensor, R: Tensor, I: Tensor) -> Tensor:
        """``S`` (bohr^-3) for type ids ``tid``, distances ``R`` (A) and anion removal energies ``I`` (eV; ignored for
        a cation core).  Same shapes; differentiable in R and I."""
        return self._evaluate(tid, R, I, derivatives=False)[0]

    def _anion_anion_coefficients(self) -> Tensor:
        """``[N_R, N_K, N_K]``: Chebyshev coefficients of log S for two beta = 0 densities, kappa_u and kappa_v both on
        the anion kappa interval (in-situ anion densities, SPEC 18.6).  Built once per process, in kappa_u slices."""
        key = ("anion_anion", self.r_lo, self.r_hi, _N_R, _N_KAPPA)
        if key not in _TABLE_CACHE:
            xr, xk = _chebyshev_nodes(_N_R), _chebyshev_nodes(_N_KAPPA)
            R = 0.5 * (self.r_hi + self.r_lo) + 0.5 * (self.r_hi - self.r_lo) * xr
            K = 0.5 * (self.k_hi + self.k_lo) + 0.5 * (self.k_hi - self.k_lo) * xk
            logS = np.empty((_N_R, _N_KAPPA, _N_KAPPA))
            RR, KV = np.meshgrid(R, K, indexing="ij")
            zeros = torch.zeros(RR.shape, dtype=torch.float64)
            for j, ku in enumerate(K):
                S = density_overlap(torch.full(RR.shape, float(ku), dtype=torch.float64), zeros, torch.tensor(KV), zeros,
                                    torch.tensor(RR) / BOHR_ANGSTROM).numpy()
                logS[:, j, :] = np.log(S)
            TR = np.polynomial.chebyshev.chebvander(xr, _N_R - 1)
            TK = np.polynomial.chebyshev.chebvander(xk, _N_KAPPA - 1)
            C = np.linalg.solve(TR, logS.reshape(_N_R, -1)).reshape(_N_R, _N_KAPPA, _N_KAPPA)
            C = np.linalg.solve(TK, C.transpose(1, 0, 2).reshape(_N_KAPPA, -1)).reshape(_N_KAPPA, _N_R, _N_KAPPA)
            C = C.transpose(1, 0, 2)
            C = np.linalg.solve(TK, C.transpose(2, 0, 1).reshape(_N_KAPPA, -1)).reshape(_N_KAPPA, _N_R, _N_KAPPA)
            _TABLE_CACHE[key] = torch.tensor(C.transpose(1, 2, 0), dtype=self.dtype)
        return _TABLE_CACHE[key]

    def overlap_anion_anion(self, R: Tensor, I_u: Tensor, I_v: Tensor) -> Tensor:
        """``S`` between two in-situ anion densities (beta = 0, kappa = sqrt(2 I)), R in A, I in eV; differentiable in
        all three.  Outside the R interval the quadrature is used directly."""
        dtype, device = R.dtype, R.device
        shape = R.shape
        R, I_u, I_v = R.reshape(-1), I_u.reshape(-1), I_v.reshape(-1)
        ku, kv = torch.sqrt(2.0 * I_u / HARTREE_EV), torch.sqrt(2.0 * I_v / HARTREE_EV)
        inside = (R >= self.r_lo) & (R <= self.r_hi)
        Rc = torch.where(inside, R, torch.full_like(R, self.r_lo))
        x = (2.0 * Rc - (self.r_lo + self.r_hi)) / (self.r_hi - self.r_lo)
        span = self.k_hi - self.k_lo
        yu = ((2.0 * ku - (self.k_lo + self.k_hi)) / span).clamp(-1.0, 1.0)
        yv = ((2.0 * kv - (self.k_lo + self.k_hi)) / span).clamp(-1.0, 1.0)
        C = self._anion_anion_coefficients().to(device=device, dtype=dtype)       # [N_R, N_K, N_K]
        A = (_chebyshev_basis(x, _N_R) @ C.reshape(_N_R, -1)).reshape(-1, _N_KAPPA, _N_KAPPA)
        p = (A * _chebyshev_basis(yu, _N_KAPPA).unsqueeze(2) * _chebyshev_basis(yv, _N_KAPPA).unsqueeze(1)).sum((1, 2))
        S = torch.exp(p)
        if bool((~inside).any()):
            w = (~inside).nonzero(as_tuple=True)
            zero = torch.zeros_like(ku[w])
            S = S.index_put(w, density_overlap(ku[w], zero, kv[w], zero, R[w] / BOHR_ANGSTROM))
        return S.reshape(shape)

    def overlap_derivatives(self, tid: Tensor, R: Tensor, I: Tensor):
        """``(S, dS/dR, d2S/dR2)`` with R in A, from the derivative series (differentiable in R and I).  Distances
        outside the expansion interval are refused."""
        return self._evaluate(tid, R, I, derivatives=True)

    def _evaluate(self, tid: Tensor, R: Tensor, I: Tensor, derivatives: bool):
        dtype, device = R.dtype, R.device
        shape = R.shape
        tid, R, I = tid.reshape(-1), R.reshape(-1), I.reshape(-1)
        kappa = torch.sqrt(2.0 * I / HARTREE_EV)
        inside = (R >= self.r_lo) & (R <= self.r_hi)
        if derivatives and not bool(inside.all()):
            raise ValueError(f"a pair distance lies outside the overlap expansion [{self.r_lo}, {self.r_hi}] A")
        Rc = torch.where(inside, R, torch.full_like(R, self.r_lo))
        x = (2.0 * Rc - (self.r_lo + self.r_hi)) / (self.r_hi - self.r_lo)
        y = ((2.0 * kappa - (self.k_lo + self.k_hi)) / (self.k_hi - self.k_lo)).clamp(-1.0, 1.0)
        TR = _chebyshev_basis(x, _N_R)                                               # [N, N_R]
        TK = _chebyshev_basis(y, _N_KAPPA)                                           # [N, N_K]
        orders = 3 if derivatives else 1
        p = [torch.zeros_like(R) for _ in range(orders)]
        for t in torch.unique(tid).tolist():
            rows = (tid == t).nonzero(as_tuple=True)[0]
            C = self._coefficients[t].to(device=device, dtype=dtype)                # [3, N_R, n_k]
            n_k = int(C.shape[-1])
            tr, tk = TR[rows], TK[rows, :n_k]
            for o in range(orders):
                p[o] = p[o].index_put((rows,), ((tr @ C[o]) * tk).sum(-1))
        S = torch.exp(p[0])
        if derivatives:
            return S.reshape(shape), (S * p[1]).reshape(shape), (S * (p[2] + p[1] * p[1])).reshape(shape)
        if bool((~inside).any()):
            params = torch.tensor([q[:4] for q in self._params], dtype=dtype, device=device)
            anion = torch.tensor([q[4] for q in self._params], device=device)
            where = (~inside).nonzero(as_tuple=True)
            pt = params[tid[where]]
            kv = torch.where(anion[tid[where]], kappa[where], pt[:, 2])
            S = S.index_put(where, density_overlap(pt[:, 0], pt[:, 1], kv, pt[:, 3], R[where] / BOHR_ANGSTROM))
        return (S.reshape(shape),)


#: Closed-core electron counts (noble-gas or pseudo-noble-gas core below the valence shell) and the electrons in that
#: core's outermost shell, by atomic number range: (Z_max, core, outer shell of the core).
_CORES = ((2, 0, 0), (10, 2, 2), (18, 10, 8), (28, 18, 8), (36, 28, 10), (46, 36, 8), (54, 46, 10),
          (71, 54, 8), (78, 68, 14), (86, 78, 10))


def _core(z: int):
    for z_max, core, outer in _CORES:
        if z <= z_max:
            return core, outer
    raise ValueError(f"no closed core defined for Z = {z}")


def removal_energy(z: int, n: int) -> float:
    """``I = E_Z(n + 1) - E_Z(n)`` (eV): removing one electron from charge state n, gas phase; nan where the state or
    its successor is not tabulated, or the electron is unbound (I <= 0)."""
    from .constants import ATOMIC_ELECTRON_AFFINITY_EV as EA, ATOMIC_SUCCESSIVE_IONIZATION_EV as IES
    if n >= 0:
        ies = IES[z]
        value = float(ies[n]) if n < len(ies) else float("nan")
    elif n == -1:
        value = float(EA[z])
    else:
        value = float("nan")                                   # doubly charged anions: unbound in vacuum
    return value if math.isfinite(value) and value > 0.0 else float("nan")


def in_situ_removal(e_hat: Tensor, states: Tensor, b: Tensor, W: Tensor, frames: Tensor) -> Tensor:
    """``I [U, S]`` (eV): the in-situ removal energy of state n of unit u, all other units at their frames (SPEC 18.6),

        I_u(n) = E(f + (d+1) e_u) - E(f + d e_u) = e_u(n+1) - e_u(n) + (b + W f)_u + (2d + 1) W_uu / 2,   d = n - f_u

    from the configuration energies without the exclusion (``e_hat`` before Pi).  The last state column (no n + 1)
    and states outside the ladder are nan."""
    S = int(e_hat.shape[1])
    first = int(round(float(states[0])))
    f = frames.to(e_hat.dtype)
    field = b + W @ f
    Wuu = torch.diagonal(W)
    n = states.to(e_hat.dtype).unsqueeze(0)                                       # [1, S]
    d = n - f.unsqueeze(1)
    step = e_hat[:, 1:] - e_hat[:, :-1]                                           # [U, S - 1]: n -> n + 1
    I = step + field.unsqueeze(1) + 0.5 * (2.0 * d[:, :-1] + 1.0) * Wuu.unsqueeze(1)
    nan = torch.full((int(e_hat.shape[0]), 1), float("nan"), dtype=e_hat.dtype, device=e_hat.device)
    I = torch.cat([I, nan], dim=1)
    return torch.where(torch.isfinite(I), I, torch.full_like(I, float("nan")))


def smooth_removal(I: Tensor, lo: float = 1.0, hi: float = 40.0, width: float = 0.25) -> Tensor:
    """A smooth clamp of removal energies to [lo, hi] eV (the anion table's interval), nan kept."""
    finite = torch.isfinite(I)
    J = torch.where(finite, I, torch.full_like(I, 0.5 * (lo + hi)))
    out = lo + width * torch.nn.functional.softplus((J - lo) / width) - width * torch.nn.functional.softplus((J - hi) / width)
    return torch.where(finite, out, torch.full_like(I, float("nan")))


def exclusion_bound_anchors(Pi: Tensor, states: Tensor, frames: Tensor):
    """``(chi_P [U], eta_P [U])``: the exclusion's Parr-Pearson finite differences at each unit's frame (SPEC 19.2),

        chi_P = 1/2 [Pi(f + 1) - Pi(f - 1)],    eta_P = Pi(f + 1) + Pi(f - 1)        (Pi(f) = 0),

    the exclusion's shift of the in-situ IE and EA, for the bound sector's electronegativity and hardness.  Units
    whose frame +-1 states are not among ``states`` contribute zero on that side.  A non-finite neighbour state
    (an unbound extra electron) or a negative eta_P (the bound sector would lose convexity) is refused."""
    s_list = [int(x) for x in states.tolist()]
    U = int(Pi.shape[0])
    chi = torch.zeros(U, dtype=Pi.dtype, device=Pi.device)
    eta = torch.zeros(U, dtype=Pi.dtype, device=Pi.device)
    zero = torch.zeros((), dtype=Pi.dtype, device=Pi.device)
    for u, f in enumerate(int(x) for x in frames.tolist()):
        side = [Pi[u, s_list.index(g)] if g in s_list else zero for g in (f + 1, f - 1)]
        if not all(bool(torch.isfinite(x)) for x in side):
            raise ValueError(f"unit {u}: a neighbouring state of its frame has no bound level; the exclusion anchors "
                             "are not defined there")
        chi[u] = 0.5 * (side[0] - side[1])
        eta[u] = side[0] + side[1]
    if bool((eta.detach() < -1e-12).any()):
        worst = int(torch.argmin(eta.detach()))
        raise ValueError(f"unit {worst}: eta_P = {float(eta[worst]):.4f} eV < 0; the exclusion is not convex in the "
                         "electron count there and the bound sector would lose positive definiteness")
    return chi, eta


class Exclusion(torch.nn.Module):
    """``Pi [U, S]``: exclusion of each unit's charge-state density by its closed-shell frame neighbours (SPEC 18).

        Pi_u(n) = K sum_{v: f_v != 0} N_v s(R_uv) [ N_u(n) S_u(n),v(R_uv) - N_u(f_u) S_u(f_u),v(R_uv) ]

    Only units with a cation or neutral frame (f_u >= 0) carry it.  Unit u in state n: N_u(n) = Z - n - core
    electrons, frontier density with kappa = sqrt(2 I_u(n)), beta = (n + 1)/kappa - 1, I from the exact gas ladder
    (states with no bound frontier electron contribute nothing).  Neighbour v: an anion frame's valence shell
    (N_v = Z - f - core, beta = 0, I_eff(Z) in (4, 16) eV, one learned scalar per element, initial 9 eV; the anion is
    unbound in vacuum) or a cation frame's outer core shell (the next ionisation energy).  K = 10 Ha x
    exp(ln 4 tanh(k)), k learned (initial 0).  s: C^2 switch from 0.8 r_cut to r_cut.  Zero at every frame.
    An extra electron with no bound level in vacuum (Mg-: negative electron affinity) is not a localised configuration:
    its state gets +inf (removed) everywhere, never zero exclusion.
    """

    I_EFF_RANGE = (4.0, 16.0)
    I_EFF_INIT = 9.0

    def __init__(self, r_cut: float = 5.0, onset: float = 0.8, tabulate: bool = True, relax: bool = False) -> None:
        super().__init__()
        from .constants import MAX_Z
        self.r_cut = float(r_cut)
        self.onset = float(onset)
        self.tabulate = bool(tabulate)
        self.relax = bool(relax)
        self.table = OverlapTable(r_max=self.r_cut + 1.0) if self.tabulate else None
        self.log_k = torch.nn.Parameter(torch.zeros(()))
        lo, hi = self.I_EFF_RANGE
        x = (self.I_EFF_INIT - lo) / (hi - lo)
        self.raw_i_eff = torch.nn.Parameter(torch.full((MAX_Z + 1,), math.log(x / (1.0 - x))))

    @property
    def K(self) -> Tensor:
        """eV per unit density overlap."""
        return 10.0 * HARTREE_EV * torch.exp(math.log(4.0) * torch.tanh(self.log_k))

    def i_eff(self, species: Tensor) -> Tensor:
        lo, hi = self.I_EFF_RANGE
        return lo + (hi - lo) * torch.sigmoid(self.raw_i_eff[species])

    def _switch(self, r: Tensor) -> Tensor:
        x = ((r - self.onset * self.r_cut) / ((1.0 - self.onset) * self.r_cut)).clamp(0.0, 1.0)
        return 1.0 - x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)

    def _pairs(self, species: Tensor, states: Tensor, edge_index: Tensor, edge_len: Tensor, frames: Tensor,
               anion_removal: Tensor | None = None):
        """Carrier-side and neighbour-side densities, and the kept (carrier, closed-shell neighbour) edges.  With
        ``anion_removal`` an anion frame's density uses its in-situ removal energy instead of the learned I_eff."""
        dtype, device = edge_len.dtype, edge_len.device
        n_units, n_states = int(species.shape[0]), int(states.shape[0])
        z_list, f_list = species.tolist(), frames.tolist()
        s_list = [int(s) for s in states.tolist()]
        # per unit and state: frontier electrons, kappa, beta (carrier side; nan kappa = no bound frontier electron)
        N_u = torch.zeros(n_units, n_states, dtype=dtype, device=device)
        k_u = torch.full((n_units, n_states), float("nan"), dtype=dtype, device=device)
        for u, (z, f) in enumerate(zip(z_list, f_list)):
            if f < 0:
                continue
            core, _ = _core(z)
            for j, n in enumerate(s_list):
                N_u[u, j] = max(0, z - n - core)
                I = removal_energy(z, n)
                if N_u[u, j] > 0 and math.isfinite(I):
                    k_u[u, j] = math.sqrt(2.0 * I / HARTREE_EV)
        # neighbour side (closed-shell frames): electrons, kappa (anions: learned I_eff), beta
        N_v = torch.zeros(n_units, dtype=dtype, device=device)
        k_v = torch.full((n_units,), float("nan"), dtype=dtype, device=device)
        b_v = torch.zeros(n_units, dtype=dtype, device=device)
        anion = torch.tensor([f < 0 for f in f_list], device=device)
        i_eff = self.i_eff(species).to(dtype)
        if anion_removal is not None:
            frame_col = torch.tensor([s_list.index(int(f)) if int(f) in s_list else 0 for f in f_list], device=device)
            frame_I = anion_removal.to(dtype).gather(1, frame_col.unsqueeze(1)).squeeze(1)
            i_eff = torch.where(anion & torch.isfinite(frame_I), frame_I, i_eff)
        for v, (z, f) in enumerate(zip(z_list, f_list)):
            if f == 0:
                continue
            core, outer = _core(z)
            if f < 0:
                N_v[v] = z - f - core
            else:
                N_v[v] = outer
                I = removal_energy(z, f)
                if math.isfinite(I):
                    k_v[v] = math.sqrt(2.0 * I / HARTREE_EV)
                    b_v[v] = (f + 1) / float(k_v[v]) - 1.0
        k_v = torch.where(anion, torch.sqrt(2.0 * i_eff / HARTREE_EV), k_v)
        src, dst = edge_index[0], edge_index[1]
        keep = (torch.tensor([f_list[i] >= 0 for i in src.tolist()], device=device, dtype=torch.bool)
                & torch.tensor([f_list[i] != 0 for i in dst.tolist()], device=device, dtype=torch.bool)
                & (edge_len < self.r_cut) & torch.isfinite(k_v[dst]) & (N_v[dst] > 0))
        return dict(s_list=s_list, N_u=N_u, k_u=k_u, N_v=N_v, k_v=k_v, b_v=b_v, i_eff=i_eff, keep=keep)

    def _unbound(self, species, states, frames, P, X: Tensor) -> Tensor:
        return torch.where(self.unbound_states(species, states, frames, P["N_u"], P["k_u"]),
                           torch.full_like(X, float("inf")), X)

    def forward(self, species: Tensor, states: Tensor, edge_index: Tensor, edge_len: Tensor, frames: Tensor,
                edge_vec: Tensor | None = None, alpha: Tensor | None = None,
                anion_removal: Tensor | None = None, anion_holes: bool = False) -> Tensor:
        """``Pi [U, S]``.  With ``relax=True`` the edge vectors (``pos[src] - pos[dst]``, A) and each unit's
        polarisability volume ``alpha`` (A^3) are required (SPEC 18.4).  With ``anion_removal [U, S]`` (eV, the
        smoothed in-situ removal energies, SPEC 18.6) anion frame densities are in situ; ``anion_holes`` also gives
        anion units hole states (not recommended: it releases frame-frame anion overlap, SPEC 18.6)."""
        if anion_removal is not None and self.relax:
            raise ValueError("in-situ anion densities are not implemented with the relaxed exclusion")
        if self.relax:
            if edge_vec is None or alpha is None:
                raise ValueError("the relaxed exclusion needs edge_vec and alpha")
            frozen, relief, _ = self.relaxation(species, states, edge_index, edge_len, edge_vec, alpha, frames)
            phi = frozen - relief
            s_list = [int(s) for s in states.tolist()]
            frame_col = torch.tensor([s_list.index(int(f)) if int(f) in s_list else 0 for f in frames.tolist()],
                                     device=phi.device)
            Pi = phi - phi.gather(1, frame_col.unsqueeze(1))
            P = self._pairs(species, states, edge_index, edge_len, frames)
            return self._unbound(species, states, frames, P, Pi)
        dtype, device = edge_len.dtype, edge_len.device
        n_units, n_states = int(species.shape[0]), int(states.shape[0])
        Pi = torch.zeros(n_units, n_states, dtype=dtype, device=device)
        P = self._pairs(species, states, edge_index, edge_len, frames, anion_removal=anion_removal)
        s_list, N_u, k_u, N_v, k_v, b_v, i_eff, keep = (P[k] for k in ("s_list", "N_u", "k_u", "N_v", "k_v", "b_v",
                                                                       "i_eff", "keep"))
        f_list = frames.tolist()
        if anion_removal is not None and anion_holes:
            Pi = Pi + self._anion_carriers(species, states, edge_index, edge_len, frames, anion_removal.to(dtype), P)
        if not bool(keep.any()):
            return self._unbound(species, states, frames, P, Pi)
        src, dst, r = edge_index[0][keep], edge_index[1][keep], edge_len[keep]
        R = (r / BOHR_ANGSTROM).unsqueeze(1).expand(-1, n_states)               # [E, S] bohr
        ku = k_u[src]                                                            # [E, S]
        valid = torch.isfinite(ku) & (N_u[src] > 0)
        ku_safe = torch.where(valid, ku, torch.ones_like(ku))
        bu = torch.where(valid, (states.to(dtype).unsqueeze(0) + 1.0) / ku_safe - 1.0, torch.zeros_like(ku))
        if self.tabulate:
            S = self._tabulated(species, frames, s_list, src, dst, r, valid, k_u, k_v, b_v, i_eff)
        else:
            kv = k_v[dst].unsqueeze(1).expand(-1, n_states)
            bv = b_v[dst].unsqueeze(1).expand(-1, n_states)
            S = density_overlap(ku_safe, bu, kv, bv, R)                          # [E, S]
        term = torch.where(valid, N_u[src] * S, torch.zeros_like(S))
        # the frame state's own term, subtracted so that Pi_u(f_u) = 0
        frame_col = torch.tensor([s_list.index(int(f_list[i])) if int(f_list[i]) in s_list else 0 for i in src.tolist()],
                                 device=device)
        term = term - term.gather(1, frame_col.unsqueeze(1))
        weight = (self.K.to(dtype) * N_v[dst] * self._switch(r)).unsqueeze(1)
        Pi = Pi.index_add(0, src, weight * term)
        return self._unbound(species, states, frames, P, Pi)

    def _anion_carriers(self, species, states, edge_index, edge_len, frames, anion_removal, P) -> Tensor:
        """``[U, S]``: SPEC 18 for anion-frame units (holes and any state n != f), with in-situ densities on both
        sides (SPEC 18.6): Pi_u(n) = K sum_v N_v s(R) [N_u(n) S_u(n),v - N_u(f) S_u(f),v] over closed-shell frames v."""
        dtype, device = edge_len.dtype, edge_len.device
        n_units, n_states = int(species.shape[0]), int(states.shape[0])
        z_list, f_list = species.tolist(), frames.tolist()
        s_list = [int(x) for x in states.tolist()]
        out = torch.zeros(n_units, n_states, dtype=dtype, device=device)
        src, dst = edge_index[0], edge_index[1]
        keep = (torch.tensor([f_list[i] < 0 for i in src.tolist()], device=device, dtype=torch.bool)
                & torch.tensor([f_list[i] != 0 for i in dst.tolist()], device=device, dtype=torch.bool)
                & (src != dst) & (edge_len < self.r_cut) & torch.isfinite(P["k_v"][dst]) & (P["N_v"][dst] > 0))
        if not bool(keep.any()):
            return out
        src, dst, r = src[keep], dst[keep], edge_len[keep]
        E = int(src.numel())
        N_u = torch.zeros(n_units, n_states, dtype=dtype, device=device)
        for u, (z, f) in enumerate(zip(z_list, f_list)):
            if f < 0:
                core, _ = _core(z)
                for j, n in enumerate(s_list):
                    N_u[u, j] = max(0, z - n - core)
        I_u = anion_removal[src]                                                    # [E, S]
        valid = torch.isfinite(I_u) & (N_u[src] > 0)
        I_safe = torch.where(valid, I_u, torch.full_like(I_u, 9.0))
        R = r.unsqueeze(1).expand(-1, n_states)
        neighbour_anion = torch.tensor([f_list[v] < 0 for v in dst.tolist()], device=device, dtype=torch.bool)
        S = torch.zeros(E, n_states, dtype=dtype, device=device)
        if bool(neighbour_anion.any()):
            rows = neighbour_anion.nonzero(as_tuple=True)[0]
            I_v = P["i_eff"][dst[rows]].unsqueeze(1).expand(-1, n_states)
            S = S.index_put((rows,), self.table.overlap_anion_anion(R[rows], I_safe[rows], I_v))
        if bool((~neighbour_anion).any()):
            rows = (~neighbour_anion).nonzero(as_tuple=True)[0]
            tid = torch.tensor([self.table.anion_type(float(P["k_v"][int(v)]), float(P["b_v"][int(v)]))
                                for v in dst[rows].tolist()], device=device).unsqueeze(1).expand(-1, n_states)
            S = S.index_put((rows,), self.table.overlap(tid, R[rows], I_safe[rows]))
        term = torch.where(valid, N_u[src] * S, torch.zeros_like(S))
        frame_col = torch.tensor([s_list.index(int(f_list[i])) for i in src.tolist()], device=device)
        term = term - term.gather(1, frame_col.unsqueeze(1))
        weight = (self.K.to(dtype) * P["N_v"][dst] * self._switch(r)).unsqueeze(1)
        return out.index_add(0, src, weight * term)

    def relaxation(self, species: Tensor, states: Tensor, edge_index: Tensor, edge_len: Tensor, edge_vec: Tensor,
                   alpha: Tensor, frames: Tensor):
        """``(frozen [U, S], relief [U, S], shift [U, S, 3])`` of the rigid-shell relaxation (SPEC 18.4).

            frozen_u(n) = sum_v c_v S_v,   c_v = K N_u(n) N_v s(R_uv)
            relief_u(n) = 1/2 g^T (k I + sum_v c_v max(S_v'', 0) Rhat Rhat^T)^-1 g,   g = sum_v c_v S_v' Rhat
            shift_u(n)  = (k I + ...)^-1 g   (A; the shell's displacement, away from the closed shells)

        Rhat: unit vector from u to v; S', S'' in A^-1, A^-2; k = N_u(n)^2 K_E / alpha_u (eV / A^2).  States with no
        frontier electron (N = 0) or no bound level give zero."""
        if not self.tabulate:
            raise ValueError("the relaxed exclusion uses the overlap tables (tabulate=True)")
        from trace_electrostatics.constants import K_E
        dtype, device = edge_len.dtype, edge_len.device
        n_units, n_states = int(species.shape[0]), int(states.shape[0])
        P = self._pairs(species, states, edge_index, edge_len, frames)
        N_u, keep = P["N_u"], P["keep"]
        frozen = torch.zeros(n_units, n_states, dtype=dtype, device=device)
        g = torch.zeros(n_units, n_states, 3, dtype=dtype, device=device)
        H = torch.zeros(n_units, n_states, 3, 3, dtype=dtype, device=device)
        valid_units = torch.isfinite(P["k_u"]) & (N_u > 0)
        if bool(keep.any()):
            src, dst, r = edge_index[0][keep], edge_index[1][keep], edge_len[keep]
            rhat = -edge_vec[keep].to(dtype) / r.unsqueeze(1)                       # u -> v
            valid = valid_units[src]
            S, S1, S2 = self._tabulated(species, frames, P["s_list"], src, dst, r, valid, P["k_u"], P["k_v"],
                                        P["b_v"], P["i_eff"], derivatives=True)
            c = torch.where(valid, self.K.to(dtype) * N_u[src] * (P["N_v"][dst] * self._switch(r)).unsqueeze(1),
                            torch.zeros_like(S))                                    # [E, S]
            zero = torch.zeros_like(S)
            frozen = frozen.index_add(0, src, torch.where(valid, c * S, zero))
            g = g.index_add(0, src, torch.where(valid, c * S1, zero).unsqueeze(-1) * rhat.unsqueeze(1))
            curvature = torch.where(valid, c * S2.clamp_min(0.0), zero)
            outer = rhat.unsqueeze(-1) * rhat.unsqueeze(-2)                         # [E, 3, 3]
            H = H.index_add(0, src, curvature.unsqueeze(-1).unsqueeze(-1) * outer.unsqueeze(1))
        a = alpha.to(dtype).unsqueeze(1)
        k = torch.where(valid_units, N_u * N_u * K_E / a, torch.ones_like(N_u))
        H = H + k.unsqueeze(-1).unsqueeze(-1) * torch.eye(3, dtype=dtype, device=device)
        shift = torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)
        relief = 0.5 * (g * shift).sum(-1)
        return frozen, relief, shift

    def _tabulated(self, species, frames, s_list, src, dst, r, valid, k_u, k_v, b_v, i_eff, derivatives=False):
        """``S [E, S]`` from the overlap tables (SPEC 18.3): one table per (carrier element and state, neighbour
        element and frame); an anion neighbour's table is read at its learned I_eff.  With ``derivatives``:
        ``(S, dS/dR, d2S/dR2)``, R in A."""
        n_states = len(s_list)
        z_list, f_list = species.tolist(), frames.tolist()
        tid = torch.zeros(int(src.numel()), n_states, dtype=torch.long, device=r.device)
        valid_cpu = valid.cpu()
        combos: dict = {}
        for e, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
            combos.setdefault((z_list[u], z_list[v], int(f_list[v])), []).append(e)
        for (z_u, z_v, f_v), rows in combos.items():
            u = int(src[rows[0]])
            for j, n in enumerate(s_list):
                if not bool(valid_cpu[rows[0], j]):
                    continue
                ku = float(k_u[u, j])
                bu = (n + 1.0) / ku - 1.0
                if f_v < 0:
                    t = self.table.anion_type(ku, bu)
                else:
                    t = self.table.core_type(ku, bu, float(k_v[dst[rows[0]]]), float(b_v[dst[rows[0]]]))
                tid[rows, j] = t
        R = r.unsqueeze(1).expand(-1, n_states)
        I = i_eff[dst].unsqueeze(1).expand(-1, n_states)
        if derivatives:
            return self.table.overlap_derivatives(tid, R, I)
        return self.table.overlap(tid, R, I)

    #: Frontier electron counts of a closed shell (s^2, s^2 p^6).  N = 0 has no frontier density.
    CLOSED_COUNTS = (2, 8)

    def pair_terms(self, species: Tensor, states: Tensor, edge_index: Tensor, edge_len: Tensor, frames: Tensor):
        """``(i [P], j [P], Xi [P, S, S])``: two-body exclusion between open-shell-frame units (SPEC 18.5),

            Xi(n_u, n_v) = K s(R) [P(n_u, n_v) S(rho_u^(n_u), rho_v^(n_v)) - P(f_u, f_v) S(f_u, f_v)],
            P = N_u N_v if either state is a closed shell (N in CLOSED_COUNTS), else 0 (two open shells bond).

        Pairs: each neighbour pair i < j within the cutoff whose frames are both 0.  States without a bound frontier
        density (or N = 0) contribute nothing; their admissibility is SPEC 18's (unbound states are +inf there)."""
        dtype, device = edge_len.dtype, edge_len.device
        n_states = int(states.shape[0])
        z_list, f_list = species.tolist(), frames.tolist()
        s_list = [int(x) for x in states.tolist()]
        src, dst = edge_index[0], edge_index[1]
        keep = (src < dst) & (edge_len < self.r_cut)
        keep = keep & torch.tensor([f_list[a] == 0 and f_list[b] == 0 for a, b in zip(src.tolist(), dst.tolist())],
                                   dtype=torch.bool, device=device)
        pi, pj, r = src[keep], dst[keep], edge_len[keep]
        P = int(pi.numel())
        Xi = torch.zeros(P, n_states, n_states, dtype=dtype, device=device)
        if P == 0:
            return pi, pj, Xi

        def density(z: int, n: int):
            core, _ = _core(z)
            N = max(0, z - n - core)
            I = removal_energy(z, n)
            if N == 0 or not math.isfinite(I):
                return N, None
            kk = math.sqrt(2.0 * I / HARTREE_EV)
            return N, (kk, (n + 1.0) / kk - 1.0)

        weight = self.K.to(dtype) * self._switch(r)                                 # [P]
        frame_index = [s_list.index(0) if 0 in s_list else None]
        for a, na in enumerate(s_list):
            for c, nc in enumerate(s_list):
                rows, tids = [], []
                for p, (u, v) in enumerate(zip(pi.tolist(), pj.tolist())):
                    Nu, du = density(z_list[u], na)
                    Nv, dv = density(z_list[v], nc)
                    if du is None or dv is None:
                        continue
                    if not (Nu in self.CLOSED_COUNTS or Nv in self.CLOSED_COUNTS):
                        continue
                    rows.append(p)
                    tids.append((Nu * Nv, self.table.core_type(du[0], du[1], dv[0], dv[1])))
                if not rows:
                    continue
                rows_t = torch.tensor(rows, device=device)
                tid = torch.tensor([t for _, t in tids], device=device)
                count = torch.tensor([float(m) for m, _ in tids], dtype=dtype, device=device)
                S = self.table.overlap(tid, r[rows_t], torch.full_like(r[rows_t], 9.0))
                Xi[rows_t, a, c] = weight[rows_t] * count * S
        # the frame pair's own term (both frames 0), subtracted so that Xi(f, f) = 0
        if frame_index[0] is not None:
            f0 = frame_index[0]
            Xi = Xi - Xi[:, f0, f0].clone().view(P, 1, 1)
        return pi, pj, Xi

    @staticmethod
    def unbound_states(species, states, frames, N_u, k_u) -> Tensor:
        """``[U, S]`` True where a carrier unit's state holds an extra electron (beyond its frame's) with no bound level
        in vacuum.  Such an electron is maximally diffuse, not a localised configuration, in vacuum or in a solid:
        the state is removed (+inf) everywhere, with no dependence on neighbours (which would switch admissibility on
        and off at a cutoff)."""
        s_list = [int(s) for s in states.tolist()]
        frame_idx = torch.tensor([s_list.index(int(f)) if int(f) in s_list else 0 for f in frames.tolist()],
                                 device=N_u.device)
        N_frame = N_u.gather(1, frame_idx.unsqueeze(1))
        carrier = torch.tensor([int(f) >= 0 for f in frames.tolist()], device=N_u.device, dtype=torch.bool).unsqueeze(1)
        return carrier & (N_u > N_frame) & ~torch.isfinite(k_u)
