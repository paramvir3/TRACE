"""TRACE (with NAC) against PBE DFT (Togo's phono3py PBE set, a = 4.2557 A, own PBE NAC) along Gamma-X-K-Gamma-L, one panel
per seed; optional second TRACE model (e.g. step 20) dashed.  Usage:
python plot_vs_pbe.py OUT.png TITLE  LABEL1 BAND_s42 BAND_s43  [LABEL2 BAND2_s42 BAND2_s43]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks_4g", "phonons_mgo"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from plot_mgo_phonons import load
HERE = os.path.dirname(os.path.abspath(__file__))
REF_NAME = os.environ.get("PHONON_REF", "pbe")
REF = os.path.join(HERE, "reference", "MgO_PBE_togo_band_nac.yaml" if REF_NAME == "pbe" else "MgO_PBEsol_phonondb_band_nac.yaml")
REF_LABEL = "DFT PBE (Togo phono3py: PBE Z* 1.99, ε∞ 3.24)" if REF_NAME == "pbe" else "DFT PBEsol (phonondb mp-1265: Z* 1.98, ε∞ 3.24)"
out, title = sys.argv[1], sys.argv[2]; models = [(sys.argv[i], sys.argv[i + 1], sys.argv[i + 2]) for i in range(3, len(sys.argv), 3)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
xr, fr, _, _ = load(REF)
for k, (ax, seed) in enumerate(zip(axes, ("42", "43"))):
    for m, (label, b42, b43) in enumerate(models):
        path = b42 if seed == "42" else b43
        if not path or not os.path.exists(path):
            continue
        x, f, ticks, names = load(path)
        style = dict(color="tab:red", lw=1.6) if m == 0 else dict(color="tab:blue", lw=1.1, ls="--", alpha=0.8)
        for j in range(f.shape[1]): ax.plot(x, f[:, j], label=label if j == 0 else None, **style)
    xs = xr * x[-1] / xr[-1]
    for j in range(fr.shape[1]): ax.plot(xs, fr[:, j], color="k", lw=1.3, label=REF_LABEL if j == 0 else None)
    for t in ticks: ax.axvline(t, color="k", lw=0.5)
    ax.set_xticks(ticks); ax.set_xticklabels(names); ax.set_xlim(x[0], x[-1]); ax.set_ylim(0, 900); ax.set_title(f"seed {seed}", fontsize=10)
axes[0].set_ylabel("frequency (cm$^{-1}$)"); axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(out, dpi=150); print("wrote", out)
