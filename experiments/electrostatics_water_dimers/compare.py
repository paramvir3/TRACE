"""Side-by-side comparison of evaluated runs, plus per-epoch wall times from logs.

Usage: python compare.py CKPT_TAG RUN_A RUN_B [RUN_C RUN_D ...]   (runs in pairs: base, elec)
CKPT_TAG is model or model_last.
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1]
runs = sys.argv[2:]


def epoch_times(run):
    times, prev = [], None
    first = None
    for line in open(os.path.join(HERE, "runs", run, "log.txt")):
        m = re.match(r"\[t=\s*([0-9.]+)s\]\s+(\d+) \|", line)
        if m:
            t = float(m.group(1))
            if prev is not None:
                times.append(t - prev)
            else:
                first = t
            prev = t
        elif "Epoch |" in line and prev is None:
            prev = float(re.match(r"\[t=\s*([0-9.]+)s\]", line).group(1))
    return np.array(times)


def history(run):
    import csv
    path = os.path.join(HERE, "runs", run, "plots", "training_history.csv")
    with open(path) as handle:
        return list(csv.DictReader(handle))


data = {}
for run in runs:
    with open(os.path.join(HERE, "runs", run, f"test_errors_{tag}.json")) as handle:
        data[run] = json.load(handle)["summary"]

for i in range(0, len(runs), 2):
    a, b = runs[i], runs[i + 1]
    sa, sb = data[a], data[b]
    print(f"\n### {a} (epoch {sa['epoch']}) vs {b} (epoch {sb['epoch']}), r_max {sa['r_max']}, checkpoint {tag}")
    print(f"{'group':44s} {'n':>4s} | {'E_RMSE a':>9s} {'E_RMSE b':>9s} {'b/a':>6s} | {'bias a':>7s} {'bias b':>7s} | {'F_RMSE a':>9s} {'F_RMSE b':>9s} {'b/a':>6s}")
    for key, ga in sa["groups"].items():
        gb = sb["groups"][key]
        if ga["n"] == 0:
            print(f"{key:44s} {0:4d} |")
            continue
        print(f"{key:44s} {ga['n']:4d} | {ga['E_rmse_meV']:9.2f} {gb['E_rmse_meV']:9.2f} {gb['E_rmse_meV'] / ga['E_rmse_meV']:6.2f} | "
              f"{ga['E_bias_meV']:+7.1f} {gb['E_bias_meV']:+7.1f} | {ga['F_rmse_meV_A']:9.2f} {gb['F_rmse_meV_A']:9.2f} "
              f"{gb['F_rmse_meV_A'] / ga['F_rmse_meV_A']:6.2f}")
    for run in (a, b):
        t = epoch_times(run)
        h = history(run)
        best = min(h, key=lambda r: float(r["val_loss"]))
        last = h[-1]
        print(f"  {run}: {len(t)} epoch intervals, wall/epoch median {np.median(t):.1f} s (mean {t.mean():.1f}, "
              f"total {t.sum() / 60:.1f} min, concurrent); best val loss epoch {best['epoch']} "
              f"(val E_RMSE {float(best['val_energy_rmse']):.2f} meV/atom, val F_RMSE {1e3 * float(best['val_force_rmse']):.1f} meV/A); "
              f"last epoch val E {float(last['val_energy_rmse']):.2f}, F {1e3 * float(last['val_force_rmse']):.1f}; "
              f"train F last {1e3 * float(last['train_force_rmse']):.1f}")
        for ep in (10, 20, 40, 60, 80):
            if ep <= len(h):
                r = h[ep - 1]
                print(f"      epoch {ep:3d}: val_loss {float(r['val_loss']):.5f}  val E {float(r['val_energy_rmse']):6.2f} meV/atom  "
                      f"val F {1e3 * float(r['val_force_rmse']):6.1f} meV/A  train F {1e3 * float(r['train_force_rmse']):6.1f}")
