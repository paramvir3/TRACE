"""Plot the model's MgO phonon dispersions (band_*.yaml from tools/electrostatics/mgo_phonons.py), optionally over a
reference phonopy band.yaml (DFT).  Usage: python plot_mgo_phonons.py [--ref band.yaml --ref-label 'phonondb PBEsol']"""
import argparse, os
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
THZ = 33.35641


def load(path):
    d = yaml.safe_load(open(path))
    x = np.array([p["distance"] for p in d["phonon"]])
    f = np.array([[b["frequency"] for b in p["band"]] for p in d["phonon"]]) * THZ
    ticks, names, i = [], [], 0
    for seg, (a, b) in zip(d["segment_nqpoint"], d["labels"]):
        ticks.append(x[i]); names.append(a); i += seg
    ticks.append(x[-1]); names.append(d["labels"][-1][1])
    # merge coincident boundaries (K|G etc.) and name them
    out_t, out_n = [], []
    for t, n in zip(ticks, names):
        if out_t and abs(t - out_t[-1]) < 1e-8:
            if out_n[-1] != n: out_n[-1] = out_n[-1] + "|" + n
        else:
            out_t.append(t); out_n.append(n)
    out_n = [n.replace("G", "Γ") for n in out_n]
    return x, f, out_t, out_n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", default=None); ap.add_argument("--ref-label", default="DFT reference")
    ap.add_argument("--root", default=os.path.join(os.environ.get("TRACE_FP_CACHE", "fp_cache"), "mgo_phonons")); ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mgo_phonons_step20.png"))
    args = ap.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    panels = (("step20_s42", "TRACE-Redox step 20, seed 42"), ("step20_s43", "TRACE-Redox step 20, seed 43"), ("meanfield_s42", "mean-field SQE (AuMgO_elec_r5_s42)"))
    for ax, (tag, title) in zip(axes, panels):
        for name, color, ls, lab in (("band_backbone", "0.6", "--", "backbone alone (no electrostatic term)"),
                                     ("band_nonac", "tab:blue", "-", "with electrostatics, no NAC"),
                                     ("band_nac", "tab:red", "-", "with electrostatics + NAC (model Z*, eps_inf)")):
            x, f, ticks, names = load(os.path.join(args.root, tag, name + ".yaml"))
            for j in range(f.shape[1]):
                ax.plot(x, f[:, j], color=color, ls=ls, lw=1.3, label=lab if j == 0 else None)
        if args.ref:
            xr, fr, _, _ = load(args.ref)
            xr = xr * x[-1] / xr[-1]                      # the same path; distances rescaled to the model's reciprocal lattice
            for j in range(fr.shape[1]):
                ax.plot(xr, fr[:, j], color="k", ls="-", lw=1.0, alpha=0.8, label=args.ref_label if j == 0 else None)
        for t in ticks: ax.axvline(t, color="k", lw=0.5)
        ax.set_xticks(ticks); ax.set_xticklabels(names); ax.set_xlim(x[0], x[-1]); ax.set_title(title, fontsize=10)
        ax.axhline(401, color="k", lw=0.8, ls=":", label="experiment at Γ: TO 401, LO 718"); ax.axhline(718, color="k", lw=0.8, ls=":")
        ax.axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("frequency (cm$^{-1}$); negative = imaginary"); axes[0].legend(fontsize=7.5, loc="upper right"); axes[0].set_ylim(-250, 900)
    fig.suptitle("Bulk MgO phonons from TRACE checkpoints (a = 4.265 Å, 2x2x2 supercell, phonopy, Γ-X-K-Γ-L)", fontsize=10)
    fig.tight_layout(); fig.savefig(args.out, dpi=150); print("wrote", args.out)


if __name__ == "__main__":
    main()
