"""Step 34's new criteria (README, "Pre-registered: step 34"; 2026-11-15).

    python score_step34_s7_s8.py RUN        (benchmarks_4g; reads RUN/scores/S7_auau_scan.txt and S8_force_outliers.npz)

S7: from probe_auau_two_site.py's "as trained" block, the total bond force on test 167's rigid scan: repulsive and
monotone from 2.00 to 2.40 A, and within +-30 % of DFT's +15.25 eV/A at 2.089 A.
S8: from probe_force_outliers.py's rows, the undoped test force RMSE without the five dimers below 2.18 A
(115, 167, 207, 222, 245) at most 0.060 eV/A; and S3's two pooled numbers for reference.
S5: the crystal factor g = exp(log_alpha_scale) from RUN/model.pt, and the eps prior cell's eps_inf at the training's
last evaluation (RUN/log.txt), the phonon scorer's value beside it.  Each criterion is scored on its own.
"""
import os
import re
import sys

import numpy as np

run = sys.argv[1]


def s7():
    text = open(os.path.join(run, "scores", "S7_auau_scan.txt")).read()
    block = text.split("  as trained\n", 1)[1].split("\n\n", 1)[0]
    rows = [tuple(float(x) for x in line.split()[:3]) for line in block.splitlines() if re.match(r"^\s+\d", line)]
    d = np.array([r[0] for r in rows]); f = np.array([r[2] for r in rows])     # d, f_q, f total
    grid = [2.000, 2.050, 2.089, 2.150, 2.200, 2.250, 2.300, 2.400]                # the probe's grid, named in the entry
    assert all(np.isclose(d, g).any() for g in grid), (d, grid)
    fi = np.array([float(f[np.argmin(np.abs(d - g))]) for g in grid])
    repulsive, monotone = bool((fi > 0).all()), bool((np.diff(fi) < 0).all())
    f2089 = float(fi[2])
    ok = repulsive and monotone and abs(f2089 - 15.248) <= 0.3 * 15.248
    print("S7: total bond force on test 167 at " + ", ".join(f"{x:.3f}: {y:+.2f}" for x, y in zip(grid, fi))
          + f" eV/A; repulsive {repulsive}, monotone {monotone}; at 2.089 A {f2089:+.2f} against +15.25 "
          f"({100 * (f2089 / 15.248 - 1):+.0f} %): {'PASS' if ok else 'FAIL'}")


def s8():
    z = np.load(os.path.join(run, "scores", "S8_force_outliers.npz"), allow_pickle=True)["rows"]
    idx, doped, mse = z[:, 0].astype(int), z[:, 1].astype(bool), z[:, 2]
    five = np.isin(idx, [115, 167, 207, 222, 245])
    und = ~doped
    rest = float(np.sqrt(mse[und & ~five].mean()))
    print(f"S3 (reference): undoped {np.sqrt(mse[und].mean()):.4f}, doped {np.sqrt(mse[doped].mean()):.4f} eV/A; the five dimers "
          f"carry {100 * mse[und & five].sum() / mse[und].sum():.1f} % of the undoped squared error")
    print(f"S8: undoped test force RMSE without the five dimers {rest:.4f} eV/A (criterion <= 0.060): {'PASS' if rest <= 0.060 else 'FAIL'}")


def s5():
    import torch
    c = torch.load(os.path.join(run, "model.pt"), map_location="cpu", weights_only=False)
    g = float(torch.exp(c["model_state_dict"]["electrostatics.log_alpha_scale"]))
    eps = None
    for line in open(os.path.join(run, "log.txt")):
        m = re.search(r"dielectric prior RMSE train/val: .*mean diagonal \(eval\) \[([-\d.]+)\]", line)
        if m:
            eps = float(m.group(1))
    ph = open(os.path.join(run, "scores", "P_phonons_pbe.txt")).read()
    m = re.search(r"P2 eps_inf within [^:]*: ([-\d.]+)", ph)
    eps_ph = float(m.group(1)) if m else float("nan")
    ok = 0.85 <= g <= 0.95 and eps is not None and abs(eps - 3.235) <= 0.05
    print(f"S5: crystal factor g {g:.4f} (0.85-0.95); the prior cell's eps_inf at the training's last evaluation {eps} "
          f"(within 0.05 of 3.235; the phonon scorer's {eps_ph:.3f}): {'PASS' if ok else 'FAIL'}")


for name, fn in (("S7", s7), ("S8", s8), ("S5", s5)):
    try:
        fn()
    except Exception as err:                                   # each criterion stands alone: a missing input is NOT SCORED
        print(f"{name}: NOT SCORED ({type(err).__name__}: {str(err)[:160]})")
