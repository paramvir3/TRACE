"""Build trace_redox/data/frame_pair.npz: the frame pair's own interaction for EVERY element pair of the density library
(trace_redox SPEC 39, 2026-11-13).  numpy only; no first-principles calculation is run here.

    python tools/electrostatics/build_frame_pair.py [--cache $TRACE_FP_CACHE/pauli_fp] [--out PATH]

SPEC 23 (penetration) and SPEC 25 (Pauli) are differences against the frame pair and vanish at it.  This table holds
what they leave out, for every pair (Za <= Zb) of the library's elements in their frame states (fa, fb):

    D_ff(R) = E_es(fa, fb)(R) - fa fb K / R       the exact electrostatics of the two frozen frame densities beyond
                                                  point charges, in the cancellation-free form below
    P(R)    = T_TF^nad + E_x^nad                  the non-additive Thomas-Fermi kinetic and Dirac exchange energy of
                                                  the frozen frame densities (gk_parts 'tf' + 'x'): the electron-gas
                                                  Pauli repulsion (Gordon and Kim 1972), without its gradient and
                                                  correlation parts, which are wrong in the tails (SPEC 25)
    c(R)    = 1, or the first-principles ratio    where the frame pair's own frozen-density PBE interaction has been
                                                  computed ('sph_nonelst_pbe' of compare_overlap.run_pair, cached per
                                                  distance in --cache): E_fp(R_k) / P(R_k) at the computed distances,
                                                  interpolated between them by a monotone cubic; beyond the first and
                                                  the last it keeps its end slope and saturates within RATIO_TAIL =
                                                  0.3 A (a zero end slope, the first choice, put a bump of several
                                                  eV/A into the Au0-Au0 force between the first two points)

D_ff and P each as a Chebyshev series of R times the function on [R_MIN, R_CUT] (the penetration tables' range), and
c(R) as its knots and slopes (a series cannot carry a piecewise cubic: baked into the P series it cost 2e-2 eV for
Au0-Au0).  No constant is fitted to a benchmark; the first-principles points are stored in the file.  The runtime
(trace_redox/frame_pair.py) adds fa fb [K / R - J^G(R)] with the model's own Gaussian kernel, so that with SPEC 23 the
pair's electrostatics is E_es(na, nb) - na nb J^G for every state pair.

Why a function c(R) and not one constant (measured 2026-11-13).  The ratio drifts with distance: Au0-Au0 0.843 at
2.68 A to 0.880 at 2.10 A, Au0-O2- 0.842 at 2.0 A to 0.690 at 4.0 A.  One least-squares constant is pinned by the
shortest points and, the term being a difference of two large parts, leaves the whole interaction 11 % low at 2.1 A
(Au0-Au0) and 13-65 % high at 2.26-2.6 A (Au0-O2-).  Separate constants for the kinetic and the exchange part are not
determined (the two are nearly collinear).  The spin state: the first-principles points of two open shells are the
high-spin product; P itself is the spin-compensated electron gas of the spin-summed densities, so a pair of open
shells WITHOUT first-principles points (c = 1) carries the spin-averaged value, which is below the high-spin one
(H0-H0 at 0.74 A: 2.60 against 4.67 eV in the same functional).

The electrostatics without cancellation.  With psi_x(d) = N_x / d - phi_x(d) >= 0 (short-ranged: the total potential of
ion x is q_x / d + psi_x(d)),

    D_ff(R) = q_a psi_b(R) + Z_b psi_a(R) - (rho_b | psi_a)(R)          (and the same with a, b exchanged; the mean is stored)

Every term is short-ranged.  The direct form Z_a Z_b / R - Z_a phi_b - Z_b phi_a + (rho_a | rho_b)
(radial.pair_electrostatic_energy) cancels four terms of order Z_a Z_b / R and is wrong by up to 0.05 eV for two heavy
neutral atoms (Au0-Au0 at 2.331 A: -6.3577 against PySCF's -6.3047; this form gives -6.3047).  Quadrature: psi and its
integral G(x) = int_0^x psi(d) d dd are interpolated by cubic Hermite with their exact derivatives (psi' = -N_out / d^2,
G' = psi x), and the shell integral is split at x = R, where the shell average has its kink; with linear interpolation
and no split D_ff is rough at 4e-4 eV rms (3e-3 max) for Au0-Au0, with both at 1e-5 (6e-5 max, at 1.26 A).
"""
import argparse
import ast
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gk_parts as G  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
radial = G.radial
BOHR, HARTREE = 0.529177210903, 27.211386245988
R_MIN, R_CUT, N_CHEB = 0.5, 8.0, 96
SYMBOLS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo "
           "Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb "
           "Bi Po At Rn").split()
#: a first-principles point enters c(R) only above this (eV): below it the target is numerical noise
E_FLOOR = 5e-3
#: first-principles points closer than this (A) are one point (the mean): the two label orientations of the cache overlap
MERGE = 0.01
MAX_TARGETS = 32


def covalent_radii():
    """trace_redox/constants.py's covalent radii, read without importing the package (torch is absent in the PySCF env)."""
    src = open(os.path.join(REPO, "trace_redox", "constants.py")).read()
    for node in ast.parse(src).body:
        targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(getattr(t, "id", None) == "COVALENT_RADII_A" for t in targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("COVALENT_RADII_A not found in trace_redox/constants.py")


def hermite(xq, x, y, dy):
    """C^1 cubic Hermite interpolation of (x, y, dy) at xq (clamped to the ends of x)."""
    xq = np.clip(xq, x[0], x[-1])
    i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
    h = x[i + 1] - x[i]
    t = (xq - x[i]) / h
    return ((2 * t ** 3 - 3 * t ** 2 + 1) * y[i] + (t ** 3 - 2 * t ** 2 + t) * h * dy[i]
            + (-2 * t ** 3 + 3 * t ** 2) * y[i + 1] + (t ** 3 - t ** 2) * h * dy[i + 1])


class Ion:
    """A library state (Z, q): its density and the short-ranged part psi of its total potential, with exact derivatives."""

    def __init__(self, r, Z, q, rho):
        self.r, self.Z, self.q, self.rho = r, Z, q, rho
        A, B = radial._tail(r, rho)                       # A(d) = N_out(d), B(d) = int_d^inf 4 pi x rho dx
        self.psi = A / r - B
        self.dpsi = -A / (r * r)
        self.G = radial._cumtrapz(self.psi * r, r)
        self.dG = self.psi * r
        self.drho = np.gradient(rho, r)

    def psi_at(self, x):
        return float(hermite(np.array([x]), self.r, self.psi, self.dpsi)[0]) if x < self.r[-1] else 0.0


def overlap(a, b, R):
    """(rho_b | psi_a)(R) = int 4 pi x^2 rho_b(x) <psi_a>(x; R) dx (hartree; R in bohr), the grid split at x = R."""
    r = a.r
    x, rho = r, b.rho
    if r[0] < R < r[-1]:
        k = int(np.searchsorted(r, R))
        x = np.insert(r, k, R)
        rho = np.insert(b.rho, k, hermite(np.array([R]), r, b.rho, b.drho)[0])
    shell = (hermite(R + x, r, a.G, a.dG) - hermite(np.abs(R - x), r, a.G, a.dG)) / (2.0 * x * R)
    f = 4.0 * np.pi * x * x * rho * shell
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(x)))


def penetration(a, b, R_A):
    """D_ff = E_es - q_a q_b K / R of two ions (eV; R in A): the mean of the two orientations of the cancellation-free form."""
    R = R_A / BOHR
    h1 = a.q * b.psi_at(R) + b.Z * a.psi_at(R) - overlap(a, b, R)
    h2 = b.q * a.psi_at(R) + a.Z * b.psi_at(R) - overlap(b, a, R)
    return 0.5 * (h1 + h2) * HARTREE


#: Beyond the first and the last first-principles point the ratio keeps its end slope and saturates within this
#: distance (A): c = c_end + m_end RATIO_TAIL tanh((R - R_end) / RATIO_TAIL).  C^1 at the end, bounded beyond it.
RATIO_TAIL = 0.3


def _end_slope(h0, h1, d0, d1):
    """The shape-preserving three-point end slope of a monotone cubic (Fritsch and Butland's rule, as in PCHIP)."""
    m = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if m * d0 <= 0.0:
        return 0.0
    if d0 * d1 <= 0.0 and abs(m) > 3.0 * abs(d0):
        return 3.0 * d0
    return m


def pchip_slopes(x, y):
    """Fritsch-Carlson slopes of a monotone cubic through (x, y), with the shape-preserving end slopes."""
    m = np.zeros_like(y)
    if len(x) < 2:
        return m
    h, d = np.diff(x), np.diff(y) / np.diff(x)
    if len(x) == 2:
        m[:] = d[0]
        return m
    for k in range(1, len(y) - 1):
        if d[k - 1] * d[k] > 0.0:
            w1, w2 = 2.0 * h[k] + h[k - 1], h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    m[0] = _end_slope(h[0], h[1], d[0], d[1])
    m[-1] = _end_slope(h[-1], h[-2], d[-1], d[-2])
    return m


def ratio_curve(R_k, c_k):
    """(c(R), slopes): through the first-principles ratios, monotone between them; beyond the first and the last point
    it keeps the end slope and saturates within RATIO_TAIL (no trend is extrapolated further than m_end RATIO_TAIL)."""
    if len(R_k) == 0:
        return (lambda R: np.ones_like(np.asarray(R, dtype=float))), np.zeros(0)
    if len(R_k) == 1:
        return (lambda R: np.full_like(np.asarray(R, dtype=float), c_k[0])), np.zeros(1)
    m = pchip_slopes(R_k, c_k)

    def c(R):
        R = np.asarray(R, dtype=float)
        inside = hermite(R, R_k, c_k, m)
        below = c_k[0] + m[0] * RATIO_TAIL * np.tanh((R - R_k[0]) / RATIO_TAIL)
        above = c_k[-1] + m[-1] * RATIO_TAIL * np.tanh((R - R_k[-1]) / RATIO_TAIL)
        return np.where(R < R_k[0], below, np.where(R > R_k[-1], above, inside))
    return c, m


def cached_targets(cache, Za, fa, Zb, fb):
    """[(R, sph_nonelst_pbe)] of the frame pair from the cache, both orientations of the label, points within MERGE
    of one another averaged, the ones below E_FLOOR left out."""
    rows = []
    for (z1, f1, z2, f2) in {(Za, fa, Zb, fb), (Zb, fb, Za, fa)}:
        for path in glob.glob(os.path.join(cache, f"{SYMBOLS[z1]}{f1:+d}_{SYMBOLS[z2]}{f2:+d}_*.json")):
            with open(path) as fh:
                rec = json.load(fh)
            if "sph_nonelst_pbe" in rec.get("fp", {}) and float(rec["fp"]["sph_nonelst_pbe"]) >= E_FLOOR:
                rows.append((float(rec["pair"][2]), float(rec["fp"]["sph_nonelst_pbe"])))
    rows.sort()
    merged = []
    for R, E in rows:
        if merged and R - merged[-1][0][-1] < MERGE:
            merged[-1][0].append(R)
            merged[-1][1].append(E)
        else:
            merged.append(([R], [E]))
    return [(float(np.mean(a)), float(np.mean(b))) for a, b in merged][:MAX_TARGETS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(os.environ.get("TRACE_FP_CACHE", "fp_cache"), "pauli_fp"),
                    help="first-principles cache of the frame pairs (default $TRACE_FP_CACHE/pauli_fp, TRACE_FP_CACHE defaulting to ./fp_cache)")
    ap.add_argument("--out", default=os.path.join(REPO, "trace_redox", "data", "frame_pair.npz"))
    ap.add_argument("--ratios-only", action="store_true", help="recompute c(R)'s slopes and check values in an existing file")
    args = ap.parse_args()
    if args.ratios_only:
        d = dict(np.load(args.out, allow_pickle=False))
        for i, n in enumerate(d["targets_n"]):
            n = int(n)
            R_t, c_k = d["targets_R"][i][:n], d["ratio_c"][i][:n]
            c_of, m_k = ratio_curve(R_t, c_k)
            d["ratio_slope"][i][:n] = m_k
            d["check_pauli"][i] = c_of(d["check_R"]) * d["check_pauli_bare"][i]
        d["ratio_tail"] = np.array(RATIO_TAIL)
        np.savez(args.out, **d)
        print(f"rewrote the ratio slopes and check values of {args.out} ({int(np.sum(d['targets_n'] > 0))} pairs with points)")
        return
    covalent = covalent_radii()
    d = np.load(os.path.join(REPO, "trace_redox", "data", "ion_densities.npz"), allow_pickle=False)
    r = d["grid"]
    dens = {(int(k[0]), int(k[1])): d["densities"][i] for i, k in enumerate(d["density_keys"])}
    frames = {int(z): int(f) for z, f in d["frames"]}
    elements = [int(z) for z in d["elements"]]
    ion = {z: Ion(r, z, frames[z], dens[(z, frames[z])]) for z in elements}
    nodes = radial.chebyshev_nodes(N_CHEB, R_MIN, R_CUT)
    check_R = np.round(np.concatenate([np.arange(0.6, 4.0, 0.05) + 0.013, np.arange(4.0, 8.0, 0.25) + 0.013]), 3)
    out = {k: [] for k in ("pairs", "frames", "es", "pauli", "c", "measured", "tR", "tE", "tN", "spread", "chk_es", "chk_pa",
                           "chk_bare", "errors", "ck", "cm")}
    worst, t0 = 0.0, time.time()
    for ia, Za in enumerate(elements):
        for Zb in elements[ia:]:
            fa, fb = frames[Za], frames[Zb]
            P = lambda R: (lambda p: (p["tf"] + p["x"]) * HARTREE)(G.gk_parts(r, dens[(Za, fa)], dens[(Zb, fb)], R / BOHR))
            D = lambda R: penetration(ion[Za], ion[Zb], R)
            rows = cached_targets(args.cache, Za, fa, Zb, fb)
            R_t, E_t = np.array([a for a, _ in rows]), np.array([b for _, b in rows])
            if len(rows):
                P_t = np.array([P(v) for v in R_t])
                # 2026-11-13: a point enters only where the electron-gas term itself is above the floor.  The ratio
                # of a first-principles value to a reference that passes through zero carries no information (H0-H0:
                # the electron-gas term is +0.004 eV at 2.6 A and -0.004 eV at 3.0 A, the first-principles one
                # +0.074 and +0.021, ratios 18 and -6); every other pair's points lie above 7e-3 eV and are unchanged.
                held = P_t >= E_FLOOR
                if not held.all():
                    print(f"  {SYMBOLS[Za]}{fa:+d}-{SYMBOLS[Zb]}{fb:+d}: {int((~held).sum())} point(s) left out, electron-gas term below "
                          f"{E_FLOOR} eV there: " + ", ".join(f"{r:.2f} A (first principles {e:+.4f}, electron gas {g:+.4f})"
                                                              for r, e, g in zip(R_t[~held], E_t[~held], P_t[~held])), flush=True)
                R_t, E_t, P_t = R_t[held], E_t[held], P_t[held]
                rows = [rows[i] for i in np.nonzero(held)[0]]
            if len(rows):
                c_k = E_t / P_t
                c_mean, sp = float(E_t @ P_t / (P_t @ P_t)), float(c_k.max() / c_k.min())
            else:
                c_k, c_mean, sp = np.zeros(0), 1.0, float("nan")
            c_of, m_k = ratio_curve(R_t, c_k)
            p_nodes = np.array([P(x) for x in nodes])
            ces = radial.chebyshev_fit(np.array([x * D(x) for x in nodes]), N_CHEB)
            cpa = radial.chebyshev_fit(nodes * p_nodes, N_CHEB)                    # R P(R); c(R) is applied at runtime
            x = (2.0 * check_R - (R_CUT + R_MIN)) / (R_CUT - R_MIN)
            d_direct, p_bare = np.array([D(v) for v in check_R]), np.array([P(v) for v in check_R])
            p_direct = c_of(check_R) * p_bare
            e_es = np.abs(np.polynomial.chebyshev.chebval(x, ces) / check_R - d_direct)
            e_pa = np.abs(np.polynomial.chebyshev.chebval(x, cpa) / check_R - p_bare)
            reach = check_R >= 0.6 * (covalent[Za] + covalent[Zb])          # the range a structure can reach
            err = float(max(e_es[reach].max(), e_pa[reach].max()))
            worst = max(worst, err)
            pad = lambda a: np.concatenate([a, np.full(MAX_TARGETS - len(a), np.nan)])
            for key, value in (("pairs", [Za, Zb]), ("frames", [fa, fb]), ("es", ces), ("pauli", cpa), ("c", c_mean),
                               ("measured", bool(len(rows))), ("tR", pad(R_t)), ("tE", pad(E_t)), ("tN", len(rows)), ("spread", sp),
                               ("ck", pad(c_k)), ("cm", pad(m_k)),
                               ("chk_es", d_direct), ("chk_pa", p_direct), ("chk_bare", p_bare),
                               ("errors", [float(e_es[reach].max()), float(e_pa[reach].max()), float(e_es.max()), float(e_pa.max())])):
                out[key].append(value)
            note = (f"c(R) through {len(rows)} first-principles points at {R_t.min():.2f}-{R_t.max():.2f} A: {c_k.min():.3f}-{c_k.max():.3f} "
                    f"(mean {c_mean:.4f})" if len(rows) else "c = 1 (electron gas; no first-principles point)")
            print(f"  {SYMBOLS[Za]}{fa:+d}-{SYMBOLS[Zb]}{fb:+d}: {note}; table error in reach: D_ff {e_es[reach].max():.1e}, P {e_pa[reach].max():.1e} eV"
                  f"  [{time.time() - t0:.0f} s]", flush=True)
    np.savez(args.out, pairs=np.array(out["pairs"]), frames=np.array(out["frames"]), es_coefficients=np.array(out["es"]),
             pauli_coefficients=np.array(out["pauli"]), c=np.array(out["c"]), c_measured=np.array(out["measured"]),
             targets_R=np.array(out["tR"]), targets_E=np.array(out["tE"]), targets_n=np.array(out["tN"]),
             ratio_c=np.array(out["ck"]), ratio_slope=np.array(out["cm"]),
             c_spread=np.array(out["spread"]), table_errors=np.array(out["errors"]), check_R=check_R,
             check_es=np.array(out["chk_es"]), check_pauli=np.array(out["chk_pa"]), check_pauli_bare=np.array(out["chk_bare"]),
             r_min=R_MIN, r_cut=R_CUT, chebyshev_worst_error=worst, ratio_tail=RATIO_TAIL,
             method=np.array("SPEC 39: D_ff = E_es(frame densities) - fa fb K/R (cancellation-free, Hermite quadrature split at x = R); "
                             "P = non-additive Thomas-Fermi kinetic + Dirac exchange of the frame densities (ion_densities.npz); "
                             "c(R) = 1 or the monotone-cubic interpolant of the frame pair's frozen-density PBE interaction "
                             "(sph_nonelst_pbe) over P at the computed distances (targets_R, ratio_c, ratio_slope), constant beyond them; "
                             "pauli_coefficients hold R P(R); build_frame_pair.py 2026-11-13"))
    print(f"wrote {args.out}: {len(out['pairs'])} pairs, {int(np.sum(out['measured']))} with first-principles points; worst table error at "
          f"the check points beyond 0.6 (r_cov,a + r_cov,b) {worst:.2e} eV; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
