"""Read step 29 (the energy-weight test) against its pre-registered criteria W1-W4 (README, 2026-11-08), from the score
files only.  Usage (benchmarks_4g): python read_step29.py"""
import os
import re

B = "AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bopw"
RUNS = {"test (ew 1000)": f"{B}_ew1000_bondlaw_r5_s43", "control (ew 10)": f"{B}_ew10_bondlaw_r5_s43",
        "step 26 s43": f"{B}_bondlaw_r5_s43"}


def read(run):
    d = os.path.join("runs", run, "scores")
    out = {}
    t = open(os.path.join(d, "T5_wetting.txt")).read() if os.path.exists(os.path.join(d, "T5_wetting.txt")) else ""
    m = re.search(r"doped ([-+0-9.]+) meV, undoped ([-+0-9.]+) meV", t)
    if m:
        out["wet_d"], out["wet_u"] = float(m.group(1)), float(m.group(2))
    for cls in ("doped", "undoped"):
        m = re.search(rf"^\s+{cls}\s+n=\d+\s+E RMSE ([0-9.]+) meV/atom\s+F RMSE ([0-9.]+)", t, re.M)
        if m:
            out[f"E_{cls}"], out[f"F_{cls}"] = float(m.group(1)), float(m.group(2))
    for split, f in (("train", "H1_class_residuals_train.txt"), ("test", "H1_class_residuals.txt")):
        p = os.path.join(d, f)
        if os.path.exists(p):
            for m in re.finditer(r"^\s+(doped|undoped)\s.*flat - upright\s+([-+0-9.]+)", open(p).read(), re.M):
                out[f"off_{split}_{m.group(1)}"] = float(m.group(2))
    log = os.path.join("runs", run, "log.txt")
    if os.path.exists(log):
        rows = [l for l in open(log) if re.match(r"^\[t=.*\]\s+\d+ \|", l)]
        if rows:
            c = [x.strip() for x in rows[-1].split("|")]
            out["E_train_meV_atom"], out["F_MSE"] = float(c[2]), float(c[5])
    return out


r = {k: read(v) for k, v in RUNS.items()}
for k, v in r.items():
    print(f"{k:16s} " + "  ".join(f"{a} {b:+.4g}" for a, b in sorted(v.items())))
t, c = r["test (ew 1000)"], r["control (ew 10)"]
try:
    w1 = all(abs(t[f"off_train_{x}"]) <= 0.5 * abs(c[f"off_train_{x}"]) for x in ("doped", "undoped"))
    w2 = abs(t["wet_d"] + 2.7) <= abs(c["wet_d"] + 2.7) - 50.0
    w3 = all(t[f"F_{x}"] <= 1.10 * c[f"F_{x}"] for x in ("doped", "undoped"))
    dd = t["off_train_doped"] - t["off_train_undoped"]
    w4 = -120.0 <= dd <= -30.0
    print(f"W1 class offsets <= 50 % of control: {'PASS' if w1 else 'FAIL'}; W2 wet_d >= 50 meV closer to DFT: "
          f"{'PASS' if w2 else 'FAIL'}; W3 forces within 10 %: {'PASS' if w3 else 'FAIL'}; W4 doped - undoped {dd:+.1f} "
          f"in [-120, -30]: {'as predicted' if w4 else 'not as predicted'}")
    print(f"PASS (W1 and W2 and W3): {w1 and w2 and w3}")
    for k in ("test (ew 1000)", "control (ew 10)"):
        v = r[k]
        if "F_MSE" in v:
            ew = 1000.0 if "1000" in k else 10.0
            e = ew * (v["E_train_meV_atom"] * 1e-3) ** 2
            print(f"  {k}: final energy term {e:.2e} vs force MSE {v['F_MSE']:.2e} -> energy share {100 * e / (e + v['F_MSE']):.1f} %")
except KeyError as exc:
    print(f"not scored yet ({exc})")
