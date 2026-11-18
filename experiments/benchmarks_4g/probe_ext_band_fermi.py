"""Extensivity, C4-0 (2026-11-10): the candidate many-electron band energy on the saved band arrays (no model is loaded,
nothing in the package is used or changed).

    python probe_ext_band_fermi.py ARRAYS.npz [kT]

Candidate (frozen orbitals of h = diag(eps) - tau, Fermi-Dirac occupations per spin channel at the model's kT,
N_up = ceil(N/2), N_dn = floor(N/2); first order in the band's own interaction V of SPEC 20):

    F_b(N) = sum_s [ sum_k f_k^s lam_k - kT S_s ] + 1/2 sum_uv [ rho_u rho_v - sum_s (P^s_uv)^2 ] V_uv,
    P^s = sum_k f_k^s phi_k phi_k^T,   rho = diag(P^up + P^dn).

It prints (1) the limits against SPEC 20 (F_b(1) - lambda_1, F_b(2) - 2 lambda_1 - J_11) with the measure
exp(-gap / 2kT) they should scale with; (2) F_b(N) per cell copy against the filling per copy; (3) the error of a
Chebyshev expansion of the Fermi operator against the eigendecomposition, by order."""
import sys

import numpy as np

data = np.load(sys.argv[1])
kT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1


def fermi(x):
    return 0.5 * (1.0 - np.tanh(0.5 * x))


def occupations(lam, n):
    """Fermi-Dirac occupations of one spin channel holding n electrons, and mu (bisection; tr f = n)."""
    if n == 0:
        return np.zeros_like(lam), -np.inf
    lo, hi = lam.min() - 60 * kT, lam.max() + 60 * kT
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        if fermi((lam - mu) / kT).sum() < n:
            lo = mu
        else:
            hi = mu
    return fermi((lam - mu) / kT), mu


def channel(lam, vec, n):
    f, mu = occupations(lam, n)
    g = np.clip(f, 1e-300, 1.0); h = np.clip(1.0 - f, 1e-300, 1.0)
    entropy = -(f * np.log(g) + (1.0 - f) * np.log(h)).sum()
    return (vec * f) @ vec.T, float((f * lam).sum() - kT * entropy), mu


def band_free_energy(lam, vec, V, N):
    Pu, Au, _ = channel(lam, vec, (N + 1) // 2)
    Pd, Ad, _ = channel(lam, vec, N // 2)
    rho = np.diag(Pu) + np.diag(Pd)
    return Au + Ad + 0.5 * float(((np.outer(rho, rho) - Pu * Pu - Pd * Pd) * V).sum())


def chebyshev_projector(h, mu, order):
    """P = f((h - mu)/kT) by a Chebyshev expansion on the Gershgorin interval (Chebyshev-Gauss coefficients)."""
    radius = np.abs(h - np.diag(np.diag(h))).sum(1)
    lo, hi = (np.diag(h) - radius).min(), (np.diag(h) + radius).max()
    c, r = 0.5 * (hi + lo), 0.5 * (hi - lo)
    M = 4 * order
    theta = np.pi * (np.arange(M) + 0.5) / M
    values = fermi((c + r * np.cos(theta) - mu) / kT)
    coef = 2.0 / M * (values[None, :] * np.cos(np.outer(np.arange(order), theta))).sum(1)
    x = (h - c * np.eye(len(h))) / r
    T0, T1 = np.eye(len(h)), x
    P = 0.5 * coef[0] * T0 + coef[1] * T1
    for k in range(2, order):
        T0, T1 = T1, 2.0 * x @ T1 - T0
        P = P + coef[k] * T1
    return P, r


print(f"kT {kT} eV")
tags = sorted({k.split("/")[0] for k in data.files})
for tag in tags:
    eps, tau, V = data[f"{tag}/eps"], data[f"{tag}/tau"], data[f"{tag}/V"]
    copies = int(tag[-3]) * int(tag[-1]); n_x = int(data[f"{tag}/n_x"])
    h = np.diag(eps) - tau
    lam, vec = np.linalg.eigh(h)
    if not np.isfinite(V).all():
        print(f"== {tag}: V has non-finite entries; skipped"); continue
    l1, j11 = float(data[f"{tag}/lambda1"]), float(data[f"{tag}/J11"])
    F = [band_free_energy(lam, vec, V, N) for N in range(0, 4 * copies + 1)]
    gap = lam[1] - lam[0]
    print(f"== {tag}: sites {len(eps)}; excess electrons {n_x}; gap above lambda_1 {gap:.4f} eV; exp(-gap/2kT) {np.exp(-gap / (2 * kT)):.2e}")
    print(f"   F_b(1) - lambda_1 {F[1] - l1:+.3e} eV;  F_b(2) - (2 lambda_1 + J_11) {F[2] - 2 * l1 - j11:+.3e} eV")
    print("   N: F_b(N)/copies (eV): " + "  ".join(f"{N}: {F[N] / copies:+.4f}" for N in range(1, len(F))))
    Nmin = int(np.argmin(F))
    print(f"   lowest F_b at N = {Nmin} ({Nmin / copies:.2f} per copy)")
    if tag.endswith("2x2") or tag.endswith("1x1"):
        f, mu = occupations(lam, copies)                                   # one electron per copy and spin
        P = (vec * f) @ vec.T
        errs = []
        for order in (200, 400, 800, 1200):
            Pc, r = chebyshev_projector(h, mu, order)
            errs.append(f"{order}: {np.abs(Pc - P).max():.1e}")
        print(f"   Chebyshev Fermi operator, N_s = {copies}, half-width {r:.2f} eV: max|P - P_eigh| by order: " + "  ".join(errs))
