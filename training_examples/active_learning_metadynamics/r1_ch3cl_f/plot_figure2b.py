#!/usr/bin/env python3
"""Plot the final-panel Figure 2b analogue from a relaxed TRACE PES scan."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from ase.optimize import LBFGS


EXAMPLE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLE_ROOT.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from transformers_ace import TransformersACECalculator  # noqa: E402


EV_TO_KCAL_MOL = 23.06054783061903


def physical_coordinates(atoms) -> tuple[float, float, float]:
    symbols = list(atoms.symbols)
    carbon = symbols.index("C")
    fluorine = symbols.index("F")
    chlorine = symbols.index("Cl")
    r_f = float(np.linalg.norm(atoms.positions[carbon] - atoms.positions[fluorine]))
    r_cl = float(np.linalg.norm(atoms.positions[carbon] - atoms.positions[chlorine]))
    return r_f, r_cl, r_cl - r_f


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan", type=Path, default=EXAMPLE_ROOT / "results/pes_scan.csv"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=EXAMPLE_ROOT / "models/r1_trace_v2_combined.pt",
    )
    parser.add_argument(
        "--reference", type=Path, default=EXAMPLE_ROOT / "structures/ch3cl_f.xyz"
    )
    parser.add_argument(
        "--irc-reference", type=Path, default=EXAMPLE_ROOT / "data/irc_test.xyz"
    )
    parser.add_argument(
        "--training-data", type=Path, default=EXAMPLE_ROOT / "data/wtmetad_all.xyz"
    )
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=EXAMPLE_ROOT / "results/figure2b_trace.png"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference-fmax", type=float, default=0.01)
    parser.add_argument("--reference-steps", type=int, default=500)
    args = parser.parse_args()

    if args.reference_fmax <= 0.0 or args.reference_steps < 1:
        raise ValueError("Invalid reactant-reference optimization settings")

    with args.scan.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if int(row["converged"]) == 1]
    if len(selected) < 3:
        raise ValueError("At least three converged scan points are required")
    r_f = np.asarray([float(row["r_C_F_final_A"]) for row in selected])
    r_cl = np.asarray([float(row["r_C_Cl_final_A"]) for row in selected])
    energies = np.asarray([float(row["energy_eV"]) for row in selected])

    reference = read(args.reference, index=0)
    reference.pbc = False
    reference.set_cell(np.zeros((3, 3)))
    reference.calc = TransformersACECalculator(
        model_path=str(args.model.resolve()), device=args.device
    )
    reference_optimizer = LBFGS(reference, logfile=None)
    reference_converged = bool(
        reference_optimizer.run(
            fmax=args.reference_fmax, steps=args.reference_steps
        )
    )
    if not reference_converged:
        raise RuntimeError(
            "The TRACE reactant reference did not converge; increase "
            "--reference-steps or inspect the model before plotting"
        )
    reference_energy = float(reference.get_potential_energy())
    relative_kcal = (energies - reference_energy) * EV_TO_KCAL_MOL

    training = read(args.training_data, index=":")
    training_r_f, training_r_cl, training_energies = [], [], []
    for atoms in training:
        value_f, value_cl, _ = physical_coordinates(atoms)
        training_r_f.append(value_f)
        training_r_cl.append(value_cl)
        training_energies.append(float(atoms.get_potential_energy()))
    training_energies = np.asarray(training_energies)
    irc_frames = read(args.irc_reference, index=":")
    if not irc_frames:
        raise ValueError("No IRC structures were read for the reactant reference")
    irc_coordinates = np.asarray(
        [physical_coordinates(atoms)[2] for atoms in irc_frames]
    )
    irc_reactant = irc_frames[int(np.argmin(irc_coordinates))]
    training_reference = float(irc_reactant.get_potential_energy())
    training_relative = (training_energies - training_reference) * EV_TO_KCAL_MOL

    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7.2, 6.0), layout="constrained")
    levels = np.linspace(-30.0, 30.0, 13)
    filled = axis.tricontourf(
        r_f, r_cl, relative_kcal, levels=levels, cmap="coolwarm", extend="both"
    )
    axis.tricontour(
        r_f, r_cl, relative_kcal, levels=levels, colors="#555555", linewidths=0.45,
        alpha=0.65,
    )
    axis.scatter(
        training_r_f, training_r_cl, s=20, color="#111111", edgecolors="none",
        label=f"Published WTMetaD-IB data ({len(training)})", zorder=4,
    )

    if args.trajectory is not None:
        trajectory = read(args.trajectory, index=":")
        trajectory_points = np.asarray(
            [physical_coordinates(atoms)[:2] for atoms in trajectory], dtype=float
        )
        axis.scatter(
            trajectory_points[:, 0], trajectory_points[:, 1], s=7, color="#009E73",
            alpha=0.25, edgecolors="none", label="New TRACE WTMetaD trajectory", zorder=3,
        )

    axis.set_xlim(1.5, 4.0)
    axis.set_ylim(1.5, 4.0)
    axis.set_xlabel(r"$r_{\mathrm{C-F}}$ ($\AA$)")
    axis.set_ylabel(r"$r_{\mathrm{C-Cl}}$ ($\AA$)")
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=8,
        markerscale=0.9,
    )
    colorbar = figure.colorbar(filled, ax=axis, pad=0.02)
    colorbar.set_label(r"$\Delta E$ (kcal mol$^{-1}$)")

    inset = axis.inset_axes([0.55, 0.67, 0.40, 0.27])
    inset.hist(training_relative, bins=16, density=True, color="#777777", alpha=0.85)
    inset.set_xlabel(r"$\Delta E$ (kcal mol$^{-1}$)", fontsize=8)
    inset.set_ylabel("Density", fontsize=8)
    inset.tick_params(labelsize=8)
    inset.set_title("WTMetaD-IB training energies", fontsize=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)
    print(
        "TRACE reactant reference: "
        f"E={reference_energy:.9f} eV after {reference_optimizer.nsteps} steps"
    )
    print(f"Figure: {args.output}")


if __name__ == "__main__":
    main()
