"""Publication figure: the model's MgO bands against PBEsol DFT, without and with NAC (model and DFT only).
Usage (repo root): python examples/mgo_phonons_local/plot_bands.py BAND_DIR LABEL OUT.png [--color HEX] [--nac-note TEXT]
BAND_DIR holds band_nonac.yaml and, when Z* and eps_inf were supplied, band_nac.yaml; without it the model has no Born
charges and its NAC band is its no-NAC band.  Writes OUT.png (300 dpi) and the vector OUT.pdf."""
import argparse
import os

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import AutoMinorLocator  # noqa: E402

THZ = 33.35640951981521
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = [r"$\Gamma$", "X", "K", r"$\Gamma$", "L"]
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 11.5, "xtick.labelsize": 12, "ytick.labelsize": 11,
    "legend.fontsize": 11, "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 1.0, "ytick.major.width": 1.0, "ytick.minor.width": 0.8, "ytick.right": True,
    "mathtext.default": "regular", "pdf.fonttype": 42, "savefig.dpi": 300,
})


def load(p):
    d = yaml.safe_load(open(p))
    x = np.array([q["distance"] for q in d["phonon"]])
    f = np.array([[b["frequency"] for b in q["band"]] for q in d["phonon"]]) * THZ
    seg = [int(n) for n in d["segment_nqpoint"]] if "segment_nqpoint" in d else None
    ticks = [x[0]] + ([x[int(np.sum(seg[:k + 1])) - 1] for k in range(len(seg))] if seg else [x[-1]])
    return x, f, ticks


ap = argparse.ArgumentParser()
ap.add_argument("band_dir"); ap.add_argument("label"); ap.add_argument("out")
ap.add_argument("--color", default="#C8102E")
ap.add_argument("--nac-note", default=None, help="appended to panel (b)'s title, e.g. \"own $Z^*$, $\\epsilon_\\infty$\"")
args = ap.parse_args()
nac = os.path.exists(os.path.join(args.band_dir, "band_nac.yaml"))
note = args.nac_note if args.nac_note else ("" if nac else "model without Born charges")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
for ax, tag, title in ((axes[0], "nonac", "(a) without NAC"), (axes[1], "nac", "(b) with NAC" + (f": {note}" if note else ""))):
    x, f, ticks = load(os.path.join(args.band_dir, f"band_{tag}.yaml" if nac or tag == "nonac" else "band_nonac.yaml"))
    xr, fr, _ = load(os.path.join(HERE, "reference", f"MgO_PBEsol_phonondb_band_{tag}.yaml"))
    xr = xr * x[-1] / xr[-1]
    for t in ticks[1:-1]:
        ax.axvline(t, color="0.6", lw=0.7, zorder=0)
    for j in range(fr.shape[1]):
        ax.plot(xr, fr[:, j], color="k", lw=1.8, solid_capstyle="round", zorder=2)
    for j in range(f.shape[1]):
        ax.plot(x, f[:, j], color=args.color, lw=1.6, dashes=(3.2, 1.8), zorder=3)
    rms = np.sqrt(((np.sort(f, 1) - np.sort(fr, 1)) ** 2).mean())
    ax.text(0.975, 0.965, f"RMS {rms:.1f} cm$^{{-1}}$", transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", lw=0.6))
    ax.set_xticks(ticks[:len(NAMES)]); ax.set_xticklabels(NAMES[:len(ticks)])
    ax.tick_params(axis="x", length=0, pad=5)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(0, 800)
    ax.set_title(title, loc="left", pad=5)
axes[0].set_ylabel("Frequency (cm$^{-1}$)")
handles = [Line2D([], [], color="k", lw=1.8, label="DFT (PBEsol, phonondb mp-1265)"),
           Line2D([], [], color=args.color, lw=1.6, dashes=(3.2, 1.8), label=args.label)]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.6)
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.6)
root = os.path.splitext(args.out)[0]
fig.savefig(root + ".png"); fig.savefig(root + ".pdf")
print("wrote", root + ".png", root + ".pdf")
