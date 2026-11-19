"""Fit SPEC 33's constants to the measured angular scans with the module's own two-atom form (2026-11-05).

    python tools/electrostatics/fit_bond_pauli.py [--level frz|almo] [--p1 FIXED] [--anion]

For each level, the measured D(R, theta) (frz_angular.py: D = E1(dimer) - E1(contact) - E1(far); almo_angular.py: the
same at the ALMO level) at theta = 0, 20, 40, 60, 70 deg and the measured R are fitted jointly by least squares to

    sum over the two Au of  s_c(R_aj) [ C1 f(R_aj)^p1 cos t + C2 f(R_aj)^p2 P2(cos t) ]  +  D0(R),

f(R) = E1(R)/E1(2.26 A) taken from the module's own table (trace_redox.bond_pauli), one free D0 per R (the isotropic
part the module leaves to the backbone).  90 deg is excluded: there the far Au sits 2.26 A over a quantum Mg and its (Au,
Mg) non-additivity, for which there is no law, is in the measured D.  At the relaxed level p1 is held fixed (default
1.877, the frozen fit's): three R do not determine the small cos term's decay (freed, it runs to 0).  Prints the
constants, the per-R D0, the largest and rms residuals, and D(70) - D(0) model against measured per R."""
import json
import math
import os
import sys

import numpy as np
import torch
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
torch.set_default_dtype(torch.float64)
from trace_redox import bond_pauli as B  # noqa: E402

#: first-principles cache: $TRACE_FP_CACHE (default ./fp_cache)
FP_CACHE = os.environ.get("TRACE_FP_CACHE", "fp_cache")
W = os.path.join(FP_CACHE, "almo_tilt", "")
THETAS = [0, 20, 40, 60, 70]
FILES = {"frz": {2.26: "frz_angular.json", 2.4: "frz_angular_R2.4.json", 2.6: "frz_angular_R2.6.json", 2.9: "frz_angular_R2.9.json"},
         "almo": {2.26: "almo_angular.json", 2.6: "almo_angular_R2.6.json", 2.9: "almo_angular_R2.9.json"}}


ANION_DIR = os.path.join(FP_CACHE, "almo_anion", "")
#: the Au2- scans (almo_angular_anion.py; each file holds [frozen, relaxed] per angle), 2026-11-06 / 2026-11-08
ANION_FILES = {2.26: "almo_angular_anion.json", 2.6: "almo_angular_anion_R2.6.json"}


def load(path, level, thetas=THETAS, pair=False):
    d = json.load(open(path))
    if pair:                                          # the anion files: [frozen, relaxed]
        return np.array([d[f"D {th}"][1 if level == "almo" else 0] for th in thetas])
    return np.array([d[f"D {th}"][1] if level == "almo" else d[f"D {th}"] for th in thetas])


def geometry(law, R0, th):
    t = math.radians(th)
    contact = torch.tensor([0.0, 0.0, R0]); far = contact + 2.545 * torch.tensor([0.0, math.sin(t), math.cos(t)])
    rows = []
    for a, partner in ((contact, far), (far, contact)):
        d = -a; R = torch.linalg.norm(d); e = (partner - a) / torch.linalg.norm(partner - a)
        rows.append((float((d / R) @ (-e)), float(law._f((79, 8), R.reshape(1))[0]),
                     float(B._switch(R.reshape(1), B.R_ON, B.R_CUT)[0])))
    return rows


def fit(level, p1_fixed, anion=False):
    law = B.bond_pauli(level)
    if anion:
        th_of = {R: [th for th in THETAS if f"D {th}" in json.load(open(ANION_DIR + f))] for R, f in ANION_FILES.items()}
        data = {R: load(ANION_DIR + f, level, th_of[R], pair=True) for R, f in ANION_FILES.items()}
    else:
        th_of = {R: THETAS for R in FILES[level]}
        data = {R: load(W + f, level) for R, f in FILES[level].items()}
    Rs = sorted(data)
    geo = {(R, th): geometry(law, R, th) for R in Rs for th in th_of[R]}

    def model(x, R, th):
        C1, C2, p2 = x[0], x[1], x[2]
        p1 = p1_fixed if p1_fixed is not None else x[3]
        return sum(sw * (C1 * f ** p1 * c + C2 * f ** p2 * 0.5 * (3 * c * c - 1)) for c, f, sw in geo[(R, th)])

    k = 3 if p1_fixed is not None else 4
    resid = lambda x: np.array([model(x, R, th) + x[k + i] - data[R][j] for i, R in enumerate(Rs) for j, th in enumerate(th_of[R])])
    x0 = [-0.1, -0.5, 0.4] + ([] if p1_fixed is not None else [1.0]) + [0.0] * len(Rs)
    lo = [-2, -2, 0.0] + ([] if p1_fixed is not None else [0.0]) + [-1] * len(Rs)
    hi = [2, 2, 3.0] + ([] if p1_fixed is not None else [4.0]) + [1] * len(Rs)
    r = least_squares(resid, x0, bounds=(lo, hi))
    x = r.x
    p1 = p1_fixed if p1_fixed is not None else x[3]
    print(f"{level}{' ANION' if anion else ''}: C1 {x[0]:+.4f}  C2 {x[1]:+.4f}  p1 {p1:.4f}{' (fixed)' if p1_fixed is not None else ''}  p2 {x[2]:.4f};  "
          f"D0 per R {dict(zip(Rs, np.round(x[k:], 4)))};  largest residual {np.abs(r.fun).max() * 1e3:.1f} meV, rms "
          f"{np.sqrt((r.fun ** 2).mean()) * 1e3:.1f} meV")
    for R in Rs:
        j70 = th_of[R].index(70)
        print(f"   {R} A: D(70) - D(0) model {model(x, R, 70) - model(x, R, 0):+.4f}  measured {data[R][j70] - data[R][0]:+.4f}")
    interior = 0.0 + 1e-6 < x[2] < 3.0 - 1e-6
    return x, float(np.abs(r.fun).max()), interior


if __name__ == "__main__":
    args = sys.argv[1:]
    levels = [args[args.index("--level") + 1]] if "--level" in args else ["frz", "almo"]
    p1 = float(args[args.index("--p1") + 1]) if "--p1" in args else None
    if "--anion" in args:          # 2026-11-08: p1 held at the neutral's value of that level (two R cannot fix it)
        for level in levels:
            held = p1 if p1 is not None else B.LEVELS[level][2]
            x, worst, interior = fit(level, held, anion=True)
            ok = interior and worst <= 0.010
            print(f"   adoption rule (README step 27 amendment): p2 interior {interior}, largest residual "
                  f"{worst * 1e3:.1f} meV <= 10 -> {'ADOPT the fitted (C1-, C2-, p2-)' if ok else 'KEEP the neutral exponents'}")
        sys.exit(0)
    for level in levels:
        fit(level, None if level == "frz" and p1 is None else (p1 if p1 is not None else 1.877))
