"""P5 like-for-like figure (README, "Pre-registered: P5 like-for-like on Togo's 512-atom supercell", 2026-11-16): TRACE
WITHOUT NAC against PBE WITHOUT NAC (Togo, fc2 on 4x4x4), one column per seed; top row the model's force constants on
the 2x2x2 conventional supercell (64 atoms), bottom row on 4x4x4 (512 atoms, Togo's own).  Model against DFT only.
    python plot_nonac_sc2_sc4.py OUT.png TITLE RUN_s42 RUN_s43     (RUN = runs/<name>; reads RUN and RUN_sc4)"""
import os
import sys

import matplotlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks_4g", "phonons_mgo"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plot_mgo_phonons import load  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "reference", "MgO_PBE_togo_band_nonac.yaml")


def rms(a, b):
    return float(np.sqrt(((np.sort(a, 1) - np.sort(b, 1)) ** 2).mean()))


out, title, r42, r43 = sys.argv[1:5]
xr, fr, _, _ = load(REF)
fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), sharey=True)
for col, (seed, run) in enumerate((("42", r42), ("43", r43))):
    for row, (tag, d) in enumerate((("2×2×2 supercell (64 atoms)", run), ("4×4×4 supercell (512 atoms, as PBE)", run + "_sc4"))):
        ax = axes[row, col]
        x, f, ticks, names = load(os.path.join(d, "band_nonac.yaml"))
        for j in range(f.shape[1]):
            ax.plot(x, f[:, j], color="tab:red", lw=1.6, label="TRACE step 34b, no NAC" if j == 0 else None)
        xs = xr * x[-1] / xr[-1]
        for j in range(fr.shape[1]):
            ax.plot(xs, fr[:, j], color="k", lw=1.3, label="DFT PBE, no NAC (Togo, fc2 on 4×4×4)" if j == 0 else None)
        for t in ticks:
            ax.axvline(t, color="k", lw=0.5)
        ax.set_xticks(ticks); ax.set_xticklabels(names); ax.set_xlim(x[0], x[-1]); ax.set_ylim(0, 800)
        sfx = os.path.basename(run.rstrip("/")).split("_")[-1]   # 'epNN' for the early-look folders; final folders end in the seed
        ax.set_title(f"seed {seed}{f' ({sfx})' if sfx.startswith('ep') else ''}, {tag}: path RMS {rms(f, fr):.1f} cm$^{{-1}}$", fontsize=9)
    axes[0, col].legend(fontsize=8, loc="upper right")
for row in range(2):
    axes[row, 0].set_ylabel("frequency (cm$^{-1}$)")
fig.suptitle(title, fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=150)
print("wrote", out)
