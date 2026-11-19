"""Read step 30 (the energy weight on the full protocol) against X1-X6 (README, 2026-11-08 21:54 IST), from the score
files only.  Usage (benchmarks_4g): python read_step30.py"""
import json
import os
import re

B = "AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bopw"
DFT = {"wet_d": -2.7, "wet_u": 928.9, "T5": -931.6}
F26 = {42: (0.0617, 0.0853), 43: (0.0768, 0.1056)}       # step 26 test F RMSE, undoped / doped (eV/A)


def read(run):
    d = os.path.join("runs", run, "scores")
    out = {}
    t5 = os.path.join(d, "T5_wetting.txt")
    t = open(t5).read() if os.path.exists(t5) else ""
    out["T5_crashed"] = "Traceback" in t
    m = re.search(r"doped ([-+0-9.]+) meV, undoped ([-+0-9.]+) meV", t)
    if m:
        out["wet_d"], out["wet_u"] = float(m.group(1)), float(m.group(2))
        out["T5"] = out["wet_d"] - out["wet_u"]
    for cls in ("doped", "undoped"):
        m = re.search(rf"^\s+{cls}\s+n=\d+\s+E RMSE ([0-9.]+) meV/atom\s+F RMSE ([0-9.]+)", t, re.M)
        if m:
            out[f"E_{cls}"], out[f"F_{cls}"] = float(m.group(1)), float(m.group(2))
    p = os.path.join(d, "H1_class_residuals_train.txt")
    if os.path.exists(p):
        for m in re.finditer(r"^\s+(doped|undoped)\s.*flat - upright\s+([-+0-9.]+)", open(p).read(), re.M):
            out[f"off_{m.group(1)}"] = float(m.group(2))
    j = os.path.join("runs", run, "eval_aumgo_transfer.json")
    if os.path.exists(j):
        try:
            out["T2"] = json.load(open(j))["shift_doped_minus_undoped"]["q_au2"]
        except (KeyError, ValueError):
            pass
    return out


ok_all = True
for s in (42, 43):
    r = read(f"{B}_e1k_bondlaw_r5_s{s}")
    print(f"s{s}: " + "  ".join(f"{k} {v:+.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in sorted(r.items())))
    try:
        x1 = abs(r["off_undoped"]) <= 30.0
        x2 = not r["T5_crashed"] and abs(r["wet_u"] - DFT["wet_u"]) <= 100.0
        x3 = not r["T5_crashed"] and abs(r["wet_d"] - DFT["wet_d"]) <= 150.0
        x4 = not r["T5_crashed"] and abs(r["T5"] - DFT["T5"]) <= 0.15 * abs(DFT["T5"])
        x5 = r["F_undoped"] <= 1.10 * F26[s][0] and r["F_doped"] <= 1.10 * F26[s][1]
        dd = r["off_doped"] - r["off_undoped"]
        print(f"   X1 {'PASS' if x1 else 'FAIL'}  X2 {'PASS' if x2 else 'FAIL'}  X3 {'PASS' if x3 else 'FAIL'}  "
              f"X4 {'PASS' if x4 else 'FAIL'}  X5 {'PASS' if x5 else 'FAIL'}  | X6 doped - undoped {dd:+.1f} "
              f"({'in' if -120 <= dd <= -30 else 'outside'} the predicted -120..-30), T5 {100 * r['T5'] / DFT['T5']:.1f} %")
        ok_all &= x1 and x2 and x3 and x4 and x5
    except KeyError as exc:
        print(f"   not scored ({exc}) -> FAIL for this seed")
        ok_all = False
print(f"PASS (X1-X5 on both seeds): {ok_all}")
