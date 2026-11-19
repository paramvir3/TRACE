"""Step 20's Pauli overlay: SPEC 25's K fitted to first principles instead of Gordon-Kim (2026-10-25).

    OMP_NUM_THREADS=2 nice python tools/electrostatics/build_pauli_fp.py check     # (v1) only
    OMP_NUM_THREADS=2 nice python tools/electrostatics/build_pauli_fp.py WORK_DIR  # build
    (python: an interpreter with PySCF installed)

The runtime Pauli term (SPEC 25) is E_P = K_a (delta_a | rho_b^f) + K_b (rho_a^f | delta_b) + K_cc (delta_a | delta_b),
tabulated as Chebyshev series of R E_P (tools/electrostatics/build_ion_densities.py), with the one-carrier K the
least-squares ratio of the Gordon-Kim term G over a contact window.  This builds the same series with one change: the
target.  For the AuMgO element pairs (O, Mg, Al, Au) with contacts in the 4G data, each one-carrier K is fitted to
the frame-relative frozen-density PBE interaction in the Watson-sphere convention (frozen_interaction.
decomposed_interaction, 'sph_nonelst_pbe'; validated against the lattice-embedded values within 0.05-0.10 eV for all six
AuMgO one-carrier types, theory note 2026-10-18 section 4c), over the data's own contact distances (5th-95th
percentile of the element pair's distances below 3.8 A in the 4G AuMgO set, 5 points; the covalent-radius window of
the Gordon-Kim fit lies inside the O Watson sphere for O-O and is not where these ionic contacts are).  Two-carrier K_cc
and every other element pair keep their Gordon-Kim values.  Output: trace_redox/data/pauli_fp.npz (the library's pair
keys, in order; the full pair_pauli_coefficients array with the refitted rows replaced; K, windows, residuals).

Pre-registered checks: (v1) fed the library's own Gordon-Kim K, the reconstruction reproduces the library's Pauli
series to 1e-9 relative (the pipeline is the builder's); (v2) the refitted K dS against its first-principles targets
at the window points (the fit's relative residual, reported per type); (v3) at the six embedded contacts the new
runtime value is within 0.1 eV of the embedded first-principles value.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from gk_parts import radial  # noqa: E402  (trace_redox/radial.py by path: numpy only)

BOHR = 0.529177210903
HARTREE = 27.211386245988
LIB = os.path.join(REPO, "trace_redox", "data", "ion_densities.npz")
OUT = os.path.join(REPO, "trace_redox", "data", "pauli_fp.npz")
R_MIN, R_CUT, N_CHEB = 0.5, 8.0, 96                     # trace_redox.penetration's table range and order
SYM = {8: "O", 12: "Mg", 13: "Al", 79: "Au"}
FRAME = {8: -2, 12: 2, 13: 3, 79: 0}
CARRIERS = {8: [-1], 12: [1], 13: [2], 79: [-1, 1]}
#: measured 2026-10-25 from the 4G AuMgO training data (every 3rd frame; $TRACE_FP_CACHE/pauli_windows.json)
WINDOWS = {(8, 8): (2.970, 3.063), (8, 12): (2.098, 3.731), (8, 13): (2.099, 3.733), (8, 79): (2.262, 3.777),
           (12, 12): (2.970, 3.063), (12, 13): (2.969, 3.063), (12, 79): (2.658, 3.493), (79, 79): (2.331, 2.796)}
N_POINTS = 5
#: Deviation (decision of 2026-10-26): the Au-Au points took 2.5-4.8 h each, so its K is fitted from the first
#: 3 of its 5 window points (2.331-2.563 A, which cover the T5 references' Au2 bond lengths 2.545-2.578 A); the last two
#: (2.680, 2.796) can be added later.  (v3) uses no Au-Au contact.
POINTS_USED = {(79, 79): 3}
#: (v3): the embedded first-principles frame-relative Pauli values (theory note 2026-10-18 section 4c), eV
EMBEDDED = {(8, -2, 79, -1, 2.26): 1.2785, (8, -2, 79, 1, 2.26): -1.3574, (12, 2, 79, -1, 2.90): 0.2757,
            (12, 2, 79, 1, 2.90): -0.1211, (8, -2, 12, 1, 2.106): 1.303, (8, -1, 12, 2, 2.1325): -0.601}


def load_library():
    d = np.load(LIB, allow_pickle=False)
    dens = {(int(k[0]), int(k[1])): d["densities"][i] for i, k in enumerate(d["density_keys"])}
    frames = {int(z): int(f) for z, f in d["frames"]}
    keys = [tuple(int(v) for v in k) for k in d["pair_keys"]]
    return d, d["grid"], dens, frames, keys


def parts(grid, dens, frames, Za, na, Zb, nb, R_A):
    zero = np.zeros_like(grid)
    g = lambda k: dens.get(k, zero)
    return np.array(radial.density_overlap_parts(grid, g((Za, frames[Za])), g((Za, na)), g((Zb, frames[Zb])),
                                                 g((Zb, nb)), R_A / BOHR))


def series(grid, dens, frames, key, Kvec):
    Za, na, Zb, nb = key
    nodes = radial.chebyshev_nodes(N_CHEB, R_MIN, R_CUT)
    return radial.chebyshev_fit(np.array([x * (parts(grid, dens, frames, Za, na, Zb, nb, x) @ Kvec) for x in nodes]),
                                N_CHEB)


def check_v1(n_types=12, seed=0):
    """(v1): the library's own Gordon-Kim K through this reconstruction reproduce its stored Pauli series."""
    d, grid, dens, frames, keys = load_library()
    K = d["pair_pauli_K"]
    stored = d["pair_pauli_coefficients"]
    rng = np.random.default_rng(seed)
    nonzero = [t for t in range(len(keys)) if np.abs(K[t]).max() > 0]
    worst = 0.0
    for t in rng.choice(nonzero, size=min(n_types, len(nonzero)), replace=False):
        c = series(grid, dens, frames, keys[t], K[t])
        worst = max(worst, float(np.abs(c - stored[t]).max() / max(np.abs(stored[t]).max(), 1e-300)))
    ok = worst <= 1e-9
    print(f"(v1) reconstruction of {min(n_types, len(nonzero))} Gordon-Kim series: worst relative {worst:.1e} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def runtime_value(grid, dens, frames, key, Kvec, R):
    Za, na, Zb, nb = key
    c = series(grid, dens, frames, key, Kvec)
    x = (2.0 * R - (R_CUT + R_MIN)) / (R_CUT - R_MIN)
    return float(np.polynomial.chebyshev.chebval(x, c)) / R


def build(work):
    import compare_overlap as C
    d, grid, dens, frames, keys = load_library()
    lib = C.library()
    os.makedirs(work, exist_ok=True)
    index = {k: t for t, k in enumerate(keys)}
    targets, K_fp, resid = {}, {}, {}
    order = sorted(WINDOWS, key=lambda p: (79 in p, p))                  # the light pairs first, the Au pairs last
    for (Za, Zb) in order:
        lo, hi = WINDOWS[(Za, Zb)]
        Rs = np.linspace(lo, hi, N_POINTS)[: POINTS_USED.get((Za, Zb), N_POINTS)]
        types = [(Za, n, Zb, FRAME[Zb]) for n in CARRIERS[Za]] + [(Za, FRAME[Za], Zb, n) for n in CARRIERS[Zb]]
        if Za == Zb:
            types = [(Za, n, Zb, FRAME[Zb]) for n in CARRIERS[Za]]          # one orientation; the other by symmetry
        for R in Rs:
            fr = C.run_pair((SYM[Za], FRAME[Za]), (SYM[Zb], FRAME[Zb]), float(round(R, 4)), work, lib)
            for key in types:
                o = C.run_pair((SYM[key[0]], key[1]), (SYM[key[2]], key[3]), float(round(R, 4)), work, lib)
                targets.setdefault(key, []).append((float(R), o["fp"]["sph_nonelst_pbe"] - fr["fp"]["sph_nonelst_pbe"]))
            print(f"  {SYM[Za]}-{SYM[Zb]} R {R:.3f}: " + ", ".join(f"{k}: {targets[k][-1][1]:+.4f}" for k in types),
                  flush=True)
        for key in types:
            Za_, na, Zb_, nb = key
            col = 0 if na != FRAME[Za_] else 1
            Rk = np.array([r for r, _ in targets[key]])
            P = np.array([v for _, v in targets[key]])
            s = np.array([parts(grid, dens, frames, Za_, na, Zb_, nb, r)[col] for r in Rk])
            K = float(P @ s / (s @ s))
            K_fp[key] = K
            resid[key] = float(np.linalg.norm(P - K * s) / max(np.linalg.norm(P), 1e-12))
            if Za_ == Zb_:                                                    # the mirrored orientation, same K
                K_fp[(Za_, nb, Zb_, na)] = K
                resid[(Za_, nb, Zb_, na)] = resid[key]
            print(f"  K {key}: {K / HARTREE:.3f} Ha (relative residual {resid[key]:.3f})", flush=True)
    # every library type of a refitted element pair: K_a, K_b from the first-principles one-carrier K, K_cc kept
    K_gk = d["pair_pauli_K"]
    coeff = np.array(d["pair_pauli_coefficients"], copy=True)
    K_out = np.array(K_gk, copy=True)
    refit = np.zeros(len(keys), dtype=bool)
    for t, (Za, na, Zb, nb) in enumerate(keys):
        if (Za, Zb) not in WINDOWS:
            continue
        fa, fb = frames[Za], frames[Zb]
        Ka = K_fp.get((Za, na, Zb, fb), K_gk[t][0]) if na != fa else 0.0
        Kb = K_fp.get((Za, fa, Zb, nb), K_gk[t][1]) if nb != fb else 0.0
        Kvec = np.array([Ka, Kb, K_gk[t][2]])
        if np.allclose(Kvec, K_gk[t], rtol=0, atol=0):
            continue
        coeff[t] = series(grid, dens, frames, (Za, na, Zb, nb), Kvec)
        K_out[t] = Kvec
        refit[t] = True
    # (v3) at the embedded contacts
    ok3 = True
    for (Za, na, Zb, nb, R), ref in EMBEDDED.items():
        t = index[(Za, na, Zb, nb)]
        v = runtime_value(grid, dens, frames, (Za, na, Zb, nb), K_out[t], R)
        g = runtime_value(grid, dens, frames, (Za, na, Zb, nb), K_gk[t], R)
        ok = abs(v - ref) <= 0.1
        ok3 &= ok
        print(f"(v3) {(Za, na, Zb, nb)} at {R} A: first-principles table {v:+.4f}, Gordon-Kim table {g:+.4f}, embedded "
              f"{ref:+.4f} eV -> {'PASS' if ok else 'FAIL'}", flush=True)
    np.savez(OUT, pair_keys=d["pair_keys"], pair_pauli_coefficients=coeff, pair_pauli_K=K_out, refit=refit,
             windows=json.dumps({f"{a}-{b}": w for (a, b), w in WINDOWS.items()}),
             K_fp=json.dumps({str(k): v for k, v in K_fp.items()}), residuals=json.dumps({str(k): v for k, v in resid.items()}),
             method="SPEC 25 K fitted to frozen-density PBE (Watson-sphere convention) over the 4G AuMgO contact "
                    "distances (5 points; Au-Au 3 of 5, 2.331-2.563 A); build_pauli_fp.py 2026-10-25/26")
    print(f"wrote {OUT}: {int(refit.sum())} of {len(keys)} types refitted; (v3) {'PASS' if ok3 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        sys.exit(0 if check_v1() else 1)
    check_v1()
    build(sys.argv[1])
