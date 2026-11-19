"""MgO phonons of the released TRACE-Electrostatics model against PBE (Togo, NIMS MDR, CC BY 4.0), without and with
the non-analytic (NAC) correction, force constants on PBE's own 4x4x4 supercell (512 atoms); model against DFT only.
    python experiments/benchmarks_phonons/plot_release_phonons.py RUN_s42_sc4 RUN_s43_sc4 OUT.png
RUN_*_sc4 = directories written by score_phonons.py --supercell 4 (or --p5-supercell 4), holding band_{nonac,nac}.yaml."""
import os
import sys

import matplotlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks_4g", "phonons_mgo"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plot_mgo_phonons import load  # noqa: E402

REF = {"nonac": os.path.join(HERE, "reference", "MgO_PBE_togo_band_nonac.yaml"),
       "nac": os.path.join(HERE, "reference", "MgO_PBE_togo_band_nac.yaml")}


def rms(a, b):
    return float(np.sqrt(((np.sort(a, 1) - np.sort(b, 1)) ** 2).mean()))


r42, r43, out = sys.argv[1:4]
fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharey=True)
for row, (seed, run) in enumerate((("42", r42), ("43", r43))):
    for col, (kind, title) in enumerate((("nonac", "without NAC"), ("nac", "with NAC"))):
        ax = axes[row, col]
        x, f, ticks, names = load(os.path.join(run, f"band_{kind}.yaml"))
        xr, fr, _, _ = load(REF[kind])
        xs = xr * x[-1] / xr[-1]
        for j in range(fr.shape[1]):
            ax.plot(xs, fr[:, j], color="k", lw=1.4, label="DFT (PBE)" if j == 0 else None)
        for j in range(f.shape[1]):
            ax.plot(x, f[:, j], color="tab:red", lw=1.4, ls="--", label="TRACE-Electrostatics" if j == 0 else None)
        for t in ticks:
            ax.axvline(t, color="0.6", lw=0.5)
        ax.set_xticks(ticks); ax.set_xticklabels(names); ax.set_xlim(x[0], x[-1]); ax.set_ylim(0, 800)
        ax.set_title(f"{title}, seed {seed}: RMS {rms(f, fr):.1f} cm$^{{-1}}$", fontsize=10)
    axes[row, 0].set_ylabel("frequency (cm$^{-1}$)")
axes[0, 0].legend(fontsize=9, loc="upper right")
fig.suptitle("MgO phonons: TRACE-Electrostatics vs DFT (PBE), force constants on the 512-atom supercell", fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=150)
print("wrote", out)
