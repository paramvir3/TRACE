"""The model's bands against PBEsol DFT, without and with NAC side by side (model and DFT only).
Usage (repo root): python examples/mgo_phonons_local/plot_bands.py BAND_DIR LABEL OUT.png
BAND_DIR holds band_nonac.yaml and, when Z* and eps_inf were supplied, band_nac.yaml; without it the model has no Born
charges and its NAC band is its no-NAC band."""
import os
import sys

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THZ = 33.35640951981521
HERE = os.path.dirname(os.path.abspath(__file__))


def load(p):
    d = yaml.safe_load(open(p))
    x = np.array([q["distance"] for q in d["phonon"]])
    f = np.array([[b["frequency"] for b in q["band"]] for q in d["phonon"]]) * THZ
    seg = [int(n) for n in d["segment_nqpoint"]] if "segment_nqpoint" in d else None
    ticks = [x[0]] + ([x[int(np.sum(seg[:k + 1])) - 1] for k in range(len(seg))] if seg else [x[-1]])
    return x, f, ticks


d, label, out = sys.argv[1], sys.argv[2], sys.argv[3]
names = ["Γ", "X", "K", "Γ", "L"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
nac = os.path.exists(os.path.join(d, "band_nac.yaml"))
for ax, tag, title in ((axes[0], "nonac", "without NAC"),
                       (axes[1], "nac", "with NAC" if nac else "with NAC (DFT); the model has no Born charges")):
    x, f, ticks = load(os.path.join(d, f"band_{tag}.yaml" if nac or tag == "nonac" else "band_nonac.yaml"))
    xr, fr, _ = load(os.path.join(HERE, "reference", f"MgO_PBEsol_phonondb_band_{tag}.yaml"))
    xr = xr * x[-1] / xr[-1]
    for j in range(f.shape[1]):
        ax.plot(x, f[:, j], color="tab:red", lw=1.6, label=label if j == 0 else None)
    for j in range(fr.shape[1]):
        ax.plot(xr, fr[:, j], color="k", lw=1.3, label="DFT PBEsol (phonondb mp-1265)" if j == 0 else None)
    for t in ticks:
        ax.axvline(t, color="k", lw=0.5)
    rms = np.sqrt(((np.sort(f, 1) - np.sort(fr, 1)) ** 2).mean())
    ax.set_xticks(ticks[:len(names)])
    ax.set_xticklabels(names[:len(ticks)])
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(0, 900)
    ax.set_title(f"{title}: path RMS {rms:.1f} cm$^{{-1}}$", fontsize=9.5)
axes[0].set_ylabel("frequency (cm$^{-1}$)"); axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Bulk MgO phonons at a = 4.217 Å against PBEsol DFT", fontsize=10)
fig.tight_layout(); fig.savefig(out, dpi=150); print("wrote", out)
