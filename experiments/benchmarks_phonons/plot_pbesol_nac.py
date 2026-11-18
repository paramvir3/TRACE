"""One model against PBEsol DFT (phonondb mp-1265) with the non-analytic correction, each side with its own Z* and
eps_inf; TRACE and DFT only, fonts sized for a paper column.
Usage: python plot_pbesol_nac.py BAND_DIR TITLE OUT.png   (BAND_DIR holds band_nac.yaml from score_phonons.py)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks_4g", "phonons_mgo"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_mgo_phonons import load

d, title, out = sys.argv[1], sys.argv[2], sys.argv[3]
P = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "xtick.labelsize": 16, "ytick.labelsize": 15,
                     "legend.fontsize": 14, "axes.titlesize": 14, "axes.linewidth": 1.1})
x, f, ticks, names = load(os.path.join(d, "band_nac.yaml"))
xr, fr, _, _ = load(os.path.join(P, "reference", "MgO_PBEsol_phonondb_band_nac.yaml"))
xr = xr * x[-1] / xr[-1]
fig, ax = plt.subplots(figsize=(7.6, 5.6))
for j in range(fr.shape[1]):
    ax.plot(xr, fr[:, j], color="black", lw=2.0, label="DFT (PBEsol)" if j == 0 else None, zorder=2)
for j in range(f.shape[1]):
    ax.plot(x, f[:, j], color="#c1272d", lw=2.0, ls=(0, (5, 2)), label="TRACE with electrostatics" if j == 0 else None, zorder=3)
for t in ticks[1:-1]:
    ax.axvline(t, color="0.55", lw=0.8, zorder=1)
rms = float(np.sqrt(((np.sort(f, axis=1) - np.sort(fr, axis=1)) ** 2).mean()))
ax.set_xticks(ticks)
ax.set_xticklabels(names)
ax.set_xlim(x[0], x[-1])
ax.set_ylim(0, 760)
ax.set_ylabel("Frequency (cm$^{-1}$)")
ax.set_title(f"{title}\npath RMS {rms:.1f} cm$^{{-1}}$ (with NAC)")
ax.legend(loc="upper center", bbox_to_anchor=(0.41, 0.99), frameon=False)   # the empty region above K
fig.tight_layout()
fig.savefig(out, dpi=200)
print("wrote", out, f"(RMS {rms:.1f})")
