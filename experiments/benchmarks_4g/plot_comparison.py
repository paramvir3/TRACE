"""Figures: TRACE vs 4G-HDNNP (Ko et al. 2021) and PFN NequIP on the 4G benchmarks.

Usage: python plot_comparison.py   (after compare_published.py) -> figures/*.png

Colour = model family (validated all-pairs, 3 slots): TRACE blue, Behler
HDNNPs orange, PFN NequIP aqua; DFT in ink.  Within a family the marker shape
and fill carry the variant, and every row is labelled, so nothing is read
from colour alone.  Log axes where the values span decades (dots, not bars:
bar length on a log axis is meaningless).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TRACE, HDNNP, NEQUIP = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": INK2, "xtick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlelocation": "left",
})

# (label, colour, marker, filled)
METHODS = {
    "TRACE + electrostatics": (TRACE, "o", True),
    "TRACE local": (TRACE, "o", False),
    "NequIP + QEq": (NEQUIP, "^", True),
    "NequIP + E_ele": (NEQUIP, "s", False),
    "NequIP": (NEQUIP, "o", False),
    "4G-HDNNP": (HDNNP, "D", True),
    "3G-HDNNP": (HDNNP, "s", False),
    "3G-HDNNP (scaled)": (HDNNP, "s", False),
    "2G-HDNNP": (HDNNP, "v", False),
    "DFT": (INK, "o", True),
}
TITLES = {"Carbon_chain": "C10H2 / C10H3+ chain", "NaCl": "Na9Cl8+ / Na8Cl8+",
          "Ag_cluster": "Ag3+ / Ag3−", "AuMgO": "Au2 on MgO(001), periodic"}


def dot(ax, x, y, method, size=8, xerr=None, zorder=3):
    colour, marker, filled = METHODS[method]
    if xerr is not None:
        ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor=colour, elinewidth=1.2, capsize=0, zorder=zorder - 1)
    ax.plot(x, y, marker=marker, ms=size, ls="none", zorder=zorder,
            mfc=colour if filled else SURFACE, mec=colour if not filled else SURFACE,
            mew=1.6 if not filled else 1.2)


def handle(method, label=None):
    colour, marker, filled = METHODS[method]
    return Line2D([], [], marker=marker, ls="none", ms=8, mfc=colour if filled else SURFACE,
                  mec=colour if not filled else SURFACE, mew=1.6 if not filled else 1.2, label=label or method)


def fig_force_rmse(c):
    rows = ["TRACE + electrostatics", "TRACE local", "NequIP + QEq", "NequIP + E_ele", "NequIP",
            "4G-HDNNP", "3G-HDNNP", "2G-HDNNP"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), sharex=True)
    for ax, dataset in zip(axes.ravel(), ("Carbon_chain", "NaCl", "Ag_cluster", "AuMgO")):
        pfn, trace = c["pfn_fig2_digitised"][dataset], c["trace"].get(dataset, {})
        y = {m: len(rows) - 1 - i for i, m in enumerate(rows)}
        for m in rows:
            if m.startswith("TRACE"):
                value = trace.get(m, {}).get("pooled_force_rmse_eV_A")
                err = None
            else:
                key = {"4G-HDNNP": "4G", "3G-HDNNP": "3G", "2G-HDNNP": "2G", "NequIP + QEq": "NequIP+QEq",
                       "NequIP + E_ele": "NequIP+E_ele", "NequIP": "NequIP"}[m]
                raw = pfn.get(key)
                if isinstance(raw, list):
                    value, err = raw[0], [[raw[0] - raw[1]], [raw[2] - raw[0]]]
                else:
                    value, err = raw, None
            if value is None:
                ax.text(0.0062, y[m], "running" if m.startswith("TRACE") else "not reported",
                        va="center", ha="left", color=MUTED, fontsize=8.5)
                continue
            dot(ax, value, y[m], m, xerr=err)
            if m in ("TRACE + electrostatics", "4G-HDNNP", "NequIP + QEq", "NequIP"):
                ax.text(max(value * 1.25, value + (err[1][0] if err else 0) + value * 0.18), y[m],
                        f"{value * 1e3:.0f}", va="center", ha="left", color=INK2, fontsize=8.5)
        ax.set_xscale("log")
        ax.set_xlim(0.006, 3.0)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows[::-1])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.grid(axis="y", visible=False)
        ax.set_title(TITLES[dataset])
        ax.tick_params(axis="y", length=0)
        if ax in axes[:, 1]:
            ax.set_yticklabels([])
    for ax in axes[1]:
        ax.set_xlabel("Force RMSE, all charge states pooled (eV/Å, log scale)")
    fig.suptitle("Force error on the 4G-HDNNP benchmarks (labels in meV/Å)", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold", color=INK)
    fig.text(0.01, 0.005,
             "TRACE: held-out test set (1,000; Au2-MgO 400), trained on 1,000 structures for 40 epochs (Au2-MgO: 300, 25 epochs), one seed.  "
             "NequIP: PFN validation set, trained on 8,000 (Au2-MgO 4,000) for 2,000–4,000 epochs; values\n"
             "digitised from PFN's Fig. 2 at the last epoch, bars = scatter band of the final epochs (±4 % per pixel).  "
             "2G/3G/4G-HDNNP: the reference lines in PFN's Fig. 2 (trained on 90 % ≈ 9,000; Au2-MgO 4,500).",
             fontsize=7.6, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(os.path.join(OUT, "fig1_force_rmse.png"), dpi=200)
    plt.close(fig)


def fig_minima(c):
    paper = c["paper_4g"]
    systems = [("C10H2+0", "C10H2"), ("C10H3+1", "C10H3+"), ("Ag3+1", "Ag3+"), ("Ag3-1", "Ag3−")]
    methods = ["2G-HDNNP", "3G-HDNNP (scaled)", "4G-HDNNP", "TRACE local", "TRACE + electrostatics"]
    fig, (ax_e, ax_f) = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    ticks, labels = [], []
    for g, (key, name) in enumerate(systems):
        base = (len(systems) - 1 - g) * (len(methods) + 1.5)
        ticks.append(base + (len(methods) - 1) / 2)
        labels.append(name)
        dataset = "Carbon_chain" if key.startswith("C10") else "Ag_cluster"
        for k, m in enumerate(methods):
            y = base + (len(methods) - 1 - k)
            e = f = None
            if m.startswith("TRACE"):
                entry = c["trace"][dataset].get(m, {}).get("at_minima", {}).get(key)
                if entry:
                    e, f = entry["energy_error_meV_atom"], entry["mean_force_error_norm_eV_A"]
            elif dataset == "Carbon_chain":
                short = {"2G-HDNNP": "2G", "3G-HDNNP (scaled)": "3G (scaled)", "4G-HDNNP": "4G"}[m]
                e, _, f = paper["table1"][key][short]
            else:
                short = {"2G-HDNNP": "2G", "4G-HDNNP": "4G"}.get(m)
                e = paper["ag3_minimum_energy_error_meV_atom"][key].get(short) if short else None
            if e is not None:
                dot(ax_e, e, y, m)
                if m in ("4G-HDNNP", "TRACE + electrostatics"):
                    ax_e.text(e * 1.25, y, f"{e:.3g}", va="center", ha="left", color=INK2, fontsize=8)
            if f is not None:
                dot(ax_f, f, y, m)
                if m in ("4G-HDNNP", "TRACE + electrostatics"):
                    ax_f.text(f + 0.008, y, f"{f:.3f}", va="center", ha="left", color=INK2, fontsize=8)
            elif dataset == "Ag_cluster" and m == "3G-HDNNP (scaled)":
                ax_f.text(0.004, y, "paper: energies only", va="center", ha="left", color=MUTED, fontsize=7.5)
    for ax in (ax_e, ax_f):
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
    ax_e.set_xscale("log")
    ax_e.set_xlim(0.03, 8000)
    ax_e.set_xlabel("|energy error| at the DFT minimum (meV/atom, log scale)")
    ax_e.set_title("Energy")
    ax_f.set_xlim(0, 0.30)
    ax_f.set_xlabel("mean |F_model − F_DFT| per atom at the DFT minimum (eV/Å)")
    ax_f.set_title("Forces")
    fig.legend(handles=[handle(m) for m in methods], loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.945),
               fontsize=8.5, handletextpad=0.3, columnspacing=1.4)
    fig.suptitle("Errors at the DFT-optimised structures (the 4G paper's Table I and Ag3 test)", x=0.01,
                 ha="left", fontsize=12, fontweight="semibold", color=INK)
    fig.text(0.01, 0.005,
             "Paper values from Ko et al., Nat. Commun. 12, 398 (2021): Table I (carbon chain) and the Ag3 text "
             "(energies only; 3G not given for Ag3).  TRACE at the lowest-force structure of each charge state\n"
             "in the full data set (relaxations converged to 0.001–0.0015 eV/Å).  The C10H3+ and Ag3+ minima happen to "
             "be in TRACE's 1,000-structure training subset; the C10H2 and Ag3− minima are not.",
             fontsize=7.6, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    fig.savefig(os.path.join(OUT, "fig2_dft_minima.png"), dpi=200)
    plt.close(fig)


def fig_nacl(c):
    with open(os.path.join(HERE, "published", "ko2021_nacl_fig7_fig8.json")) as h:
        paper = json.load(h)["energy"]
    trace = c["trace"]["NaCl"]
    curves = {
        "DFT": {s: (paper[f"DFT {t}"]["d"], paper[f"DFT {t}"]["E"]) for s, t in (("9", "Na9"), ("8", "Na8"))},
        "4G-HDNNP": {s: (paper[f"4G {t}"]["d"], paper[f"4G {t}"]["E"]) for s, t in (("9", "Na9"), ("8", "Na8"))},
        "TRACE local": {s: (trace["TRACE local"]["scan"][k]["d_A"], trace["TRACE local"]["scan"][k]["E_eV"])
                        for s, k in (("9", "Cl8Na9+1"), ("8", "Cl8Na8+1"))},
        "TRACE + electrostatics": {s: (trace["TRACE + electrostatics"]["scan"][k]["d_A"],
                                       trace["TRACE + electrostatics"]["scan"][k]["E_eV"])
                                   for s, k in (("9", "Cl8Na9+1"), ("8", "Cl8Na8+1"))},
    }
    minima = {m: {s: paper[f"{t} {n}"]["min_d"] for s, n in (("9", "Na9"), ("8", "Na8"))}
              for m, t in (("DFT", "DFT"), ("4G-HDNNP", "4G"), ("3G-HDNNP", "3G"), ("2G-HDNNP", "2G"))}
    for m in ("TRACE local", "TRACE + electrostatics"):
        minima[m] = {"9": trace[m]["scan"]["Cl8Na9+1"]["minimum_A"], "8": trace[m]["scan"]["Cl8Na8+1"]["minimum_A"]}

    fig = plt.figure(figsize=(10.5, 6.4))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.95], hspace=0.55, wspace=0.12)
    for k, m in enumerate(curves):
        ax = fig.add_subplot(grid[0, k])
        colour = METHODS[m][0]
        for s, ls in (("9", "-"), ("8", "--")):
            d, e = np.array(curves[m][s][0]), np.array(curves[m][s][1])
            e = (e - e.min()) * 1e3
            ax.plot(d, e, ls=ls, lw=2, color=colour, solid_capstyle="round")
        for s, ls in (("9", "-"), ("8", "--")):
            ax.axvline(minima["DFT"][s], color=MUTED, lw=0.8, ls=":" if s == "8" else "-", zorder=0)
        ax.set_xlim(3.15, 3.70)
        ax.set_ylim(-5, 140)
        ax.set_title(m if m != "TRACE + electrostatics" else "TRACE + elec.", fontsize=9.5)
        ax.set_xticks([3.2, 3.4, 3.6])
        if k:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("E − E_min (meV)")
        ax.set_xlabel("Na–Na distance (Å)", fontsize=8.5)
    fig.legend(handles=[Line2D([], [], color=INK2, lw=2, ls="-", label="Na9Cl8+"),
                        Line2D([], [], color=INK2, lw=2, ls="--", label="Na8Cl8+"),
                        Line2D([], [], color=MUTED, lw=0.8, label="DFT minima (grey verticals)")],
               loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.965), fontsize=8.5)

    ax = fig.add_subplot(grid[1, :])
    order = ["DFT", "4G-HDNNP", "3G-HDNNP", "2G-HDNNP", "TRACE local", "TRACE + electrostatics"]
    for i, m in enumerate(order):
        y = len(order) - 1 - i
        a, b = minima[m]["9"], minima[m]["8"]
        colour = METHODS[m][0]
        ax.plot([a, b], [y, y], color=colour, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(a, y, "o", ms=8, mfc=colour, mec=SURFACE, mew=1.2, zorder=3)
        ax.plot(b, y, "o", ms=8, mfc=SURFACE, mec=colour, mew=1.6, zorder=3)
        ax.text(max(a, b) + 0.006, y, f"shift {round(b - a, 3) + 0.0:+.3f} Å", va="center", ha="left", color=INK2, fontsize=8.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order[::-1])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(3.34, 3.53)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("position of the energy minimum (Å);  filled = Na9Cl8+, open = Na8Cl8+")
    ax.set_title("Where the minimum sits, and how far the distant Na moves it")
    fig.suptitle("NaCl: a sodium 10 Å away shifts the equilibrium by 0.107 Å (the 4G paper's Figs. 6–7)", x=0.01,
                 ha="left", fontsize=12, fontweight="semibold", color=INK)
    fig.text(0.01, 0.005,
             "DFT, 2G, 3G, 4G: extracted exactly from the vector paths of Ko et al.'s Fig. 7 (each curve shifted to its "
             "own minimum here).  TRACE: scanned on the mean Na9Cl8+ geometry of the data,\nNa8Cl8+ = the same minus "
             "the distant Na, as in the paper; the paper scans its DFT-optimised geometry, so the absolute positions "
             "differ slightly and the shift is the comparable number.",
             fontsize=7.6, color=MUTED, va="bottom")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.86, bottom=0.14)
    fig.savefig(os.path.join(OUT, "fig3_nacl.png"), dpi=200)
    plt.close(fig)


def fig_aumgo(c):
    trace = c["trace"].get("AuMgO", {})
    paper = c["paper_4g"]
    order = ["DFT", "4G-HDNNP", "2G-HDNNP", "TRACE local", "TRACE + electrostatics"]
    fig, (ax_e, ax_b) = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for i, m in enumerate(order):
        y = len(order) - 1 - i
        colour = METHODS[m][0]
        if m.startswith("TRACE"):
            entry = trace.get(m, {}).get("aumgo")
            if entry is None:
                for ax in (ax_e, ax_b):
                    ax.text(0.02, y, "running", transform=ax.get_yaxis_transform(), va="center", color=MUTED,
                            fontsize=8.5)
                continue
            de, bond = entry["table2_meV"], entry["au_o_bond_A"]
        else:
            short = {"DFT": "DFT", "4G-HDNNP": "4G", "2G-HDNNP": "2G"}[m]
            de = {d: paper["table2_meV"][d][short] for d in ("doped", "undoped")}
            bond = {d: paper["au_o_bond_A"][d][short] for d in ("doped", "undoped")}
        for ax, v, fmt in ((ax_e, de, "{:+.0f}"), (ax_b, bond, "{:.3f}")):
            ax.plot([v["doped"], v["undoped"]], [y, y], color=colour, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(v["doped"], y, "o", ms=8, mfc=colour, mec=SURFACE, mew=1.2, zorder=3)
            ax.plot(v["undoped"], y, "o", ms=8, mfc=SURFACE, mec=colour, mew=1.6, zorder=3)
            if m in ("DFT", "4G-HDNNP", "TRACE + electrostatics"):
                for d, va, dy in (("doped", "bottom", 0.18), ("undoped", "top", -0.18)):
                    ax.text(v[d], y + dy, fmt.format(v[d]), ha="center", va=va, color=INK2, fontsize=7.8)
    ax_e.axvline(0, color=AXIS, lw=0.8, zorder=0)
    ax_e.set_xlabel("E(wetting) − E(non-wetting), Au relaxed, substrate fixed (meV)")
    ax_e.set_title("Which Au2 geometry wins (Table II)")
    ax_b.set_xlabel("relaxed non-wetting Au–O bond (Å)")
    ax_b.set_title("Au–O bond (Fig. 11)")
    for ax in (ax_e, ax_b):
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order[::-1])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.7, len(order) - 0.3)
    fig.legend(handles=[Line2D([], [], marker="o", ls="none", ms=8, mfc=INK2, mec=SURFACE, label="Al-doped MgO"),
                        Line2D([], [], marker="o", ls="none", ms=8, mfc=SURFACE, mec=INK2, mew=1.6,
                               label="undoped MgO")],
               loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.99), fontsize=8.5)
    fig.suptitle("Au2 on MgO(001): dopants > 10 Å away flip the adsorption geometry", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold", color=INK)
    fig.text(0.01, 0.005,
             "DFT, 2G, 4G: Ko et al. Table II and Fig. 11 text.  TRACE: substrate = mean geometry of each doping's "
             "data, both Au2 geometries relaxed on it (BFGS, fmax 0.005 eV/Å).  A local model cannot see the dopants,\n"
             "so its doped and undoped values must coincide (as 2G's do).",
             fontsize=7.6, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    fig.savefig(os.path.join(OUT, "fig4_aumgo.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, "comparison.json")) as h:
        c = json.load(h)
    fig_force_rmse(c)
    fig_minima(c)
    if "scan" in c["trace"].get("NaCl", {}).get("TRACE local", {}):
        fig_nacl(c)
    fig_aumgo(c)
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
