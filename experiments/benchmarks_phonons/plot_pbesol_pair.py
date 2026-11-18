"""One model against PBEsol DFT (phonondb mp-1265), without and with NAC side by side; TRACE and DFT only.
Usage: python plot_pbesol_pair.py BAND_DIR LABEL OUT.png   (BAND_DIR holds band_nonac.yaml and band_nac.yaml)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks_4g", "phonons_mgo"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from plot_mgo_phonons import load
d, label, out = sys.argv[1], sys.argv[2], sys.argv[3]; P = os.path.dirname(os.path.abspath(__file__))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, tag, title in ((axes[0], "nonac", "without NAC"), (axes[1], "nac", "with NAC (each side its own Z*, ε∞)")):
    x, f, ticks, names = load(os.path.join(d, f"band_{tag}.yaml")); xr, fr, _, _ = load(os.path.join(P, "reference", f"MgO_PBEsol_phonondb_band_{tag}.yaml")); xr = xr * x[-1] / xr[-1]
    for j in range(f.shape[1]): ax.plot(x, f[:, j], color="tab:red", lw=1.6, label=f"TRACE {label}" if j == 0 else None)
    for j in range(fr.shape[1]): ax.plot(xr, fr[:, j], color="k", lw=1.3, label="DFT PBEsol (phonondb mp-1265)" if j == 0 else None)
    for t in ticks: ax.axvline(t, color="k", lw=0.5)
    rms = np.sqrt(((np.sort(f, axis=1) - np.sort(fr, axis=1)) ** 2).mean())
    ax.set_xticks(ticks); ax.set_xticklabels(names); ax.set_xlim(x[0], x[-1]); ax.set_ylim(0, 900); ax.set_title(f"{title}: path RMS {rms:.1f} cm$^{{-1}}$", fontsize=9.5)
axes[0].set_ylabel("frequency (cm$^{-1}$)"); axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Bulk MgO phonons at a = 4.217 Å vs PBEsol DFT", fontsize=10); fig.tight_layout(); fig.savefig(out, dpi=150); print("wrote", out)
