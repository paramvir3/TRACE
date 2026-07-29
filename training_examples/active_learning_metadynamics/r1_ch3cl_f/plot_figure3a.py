#!/usr/bin/env python3
"""Reproduce the Figure 3a IRC validation for one combined TRACE model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read


EXAMPLE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLE_ROOT.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from transformers_ace import TransformersACECalculator  # noqa: E402


KCAL_MOL_TO_EV = 1.0 / 23.06054783061903


def reaction_coordinate(atoms) -> float:
    symbols = list(atoms.symbols)
    carbon = symbols.index("C")
    fluorine = symbols.index("F")
    chlorine = symbols.index("Cl")
    r_f = np.linalg.norm(atoms.positions[carbon] - atoms.positions[fluorine])
    r_cl = np.linalg.norm(atoms.positions[carbon] - atoms.positions[chlorine])
    return float(r_cl - r_f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=EXAMPLE_ROOT / "models/r1_trace_v2_combined.pt",
    )
    parser.add_argument(
        "--irc", type=Path, default=EXAMPLE_ROOT / "data/irc_test.xyz"
    )
    parser.add_argument(
        "--output", type=Path, default=EXAMPLE_ROOT / "results/figure3a_trace.png"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    frames = read(args.irc, index=":")
    if not frames or any(atoms.pbc.any() for atoms in frames):
        raise ValueError("The IRC input must contain finite nonperiodic structures")
    calculator = TransformersACECalculator(
        model_path=str(args.model.resolve()), device=args.device
    )
    coordinates, reference_energies, predicted_energies = [], [], []
    for atoms in frames:
        coordinates.append(reaction_coordinate(atoms))
        reference_energies.append(float(atoms.get_potential_energy()))
        predicted = atoms.copy()
        predicted.calc = calculator
        predicted_energies.append(float(predicted.get_potential_energy()))

    coordinates = np.asarray(coordinates)
    reference_energies = np.asarray(reference_energies)
    predicted_energies = np.asarray(predicted_energies)
    reactant_index = int(np.argmin(coordinates))
    reference_relative = reference_energies - reference_energies[reactant_index]
    predicted_relative = predicted_energies - predicted_energies[reactant_index]
    atom_count = len(frames[0])
    error_mev_per_atom = (
        1000.0 * (predicted_relative - reference_relative) / atom_count
    )
    transition_index = int(np.argmax(reference_relative))
    chemical_accuracy_mev_per_atom = 1000.0 * KCAL_MOL_TO_EV / atom_count
    metrics = {
        "frames": len(frames),
        "atoms_per_frame": atom_count,
        "reactant_coordinate_A": float(coordinates[reactant_index]),
        "transition_coordinate_A": float(coordinates[transition_index]),
        "relative_energy_mae_meV_per_atom": float(np.mean(np.abs(error_mev_per_atom))),
        "relative_energy_rmse_meV_per_atom": float(
            np.sqrt(np.mean(error_mev_per_atom**2))
        ),
        "maximum_relative_energy_error_meV_per_atom": float(
            np.max(np.abs(error_mev_per_atom))
        ),
        "transition_energy_error_meV_per_atom": float(error_mev_per_atom[transition_index]),
        "one_kcal_per_mol_threshold_meV_per_atom": chemical_accuracy_mev_per_atom,
        "points_outside_one_kcal_per_mol": int(
            np.count_nonzero(np.abs(error_mev_per_atom) > chemical_accuracy_mev_per_atom)
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame", "cv_r_C_Cl_minus_r_C_F_A", "reference_relative_eV",
                "trace_relative_eV", "error_meV_per_atom",
            ]
        )
        for index in range(len(frames)):
            writer.writerow(
                [
                    index, coordinates[index], reference_relative[index],
                    predicted_relative[index], error_mev_per_atom[index],
                ]
            )
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")

    import matplotlib.pyplot as plt
    order = np.argsort(coordinates)
    figure, axis = plt.subplots(figsize=(7.0, 5.2), layout="constrained")
    chemical_accuracy_eV = KCAL_MOL_TO_EV
    axis.fill_between(
        coordinates[order],
        reference_relative[order] - chemical_accuracy_eV,
        reference_relative[order] + chemical_accuracy_eV,
        color="#BBBBBB",
        alpha=0.35,
        linewidth=0,
        label="Reference +/- 1 kcal/mol",
    )
    axis.plot(
        coordinates[order], reference_relative[order], "o-", color="#222222",
        markersize=4, linewidth=1.8,
        label="CPCM(water)-PBE0-D3BJ/def2-SVP",
    )
    axis.plot(
        coordinates[order], predicted_relative[order], "o-", color="#D55E00",
        markersize=3.5, linewidth=1.5, label="TRACE combined",
    )
    axis.axvline(coordinates[transition_index], color="#666666", linestyle="--", linewidth=0.9)
    axis.set_xlabel(r"$r_{\mathrm{C-Cl}}-r_{\mathrm{C-F}}$ ($\AA$)")
    axis.set_ylabel(r"$\Delta E$ (eV), relative to reactant")
    axis.legend(frameon=False, fontsize=9)

    inset = axis.inset_axes([0.075, 0.205, 0.39, 0.285])
    inset.plot(
        coordinates[order], np.abs(error_mev_per_atom[order]), "o-",
        color="#0072B2", markersize=3, linewidth=1.0,
    )
    inset.axhline(
        chemical_accuracy_mev_per_atom,
        color="#222222", linestyle="--", linewidth=0.9,
    )
    inset.set_xlabel(r"$r_{\mathrm{C-Cl}}-r_{\mathrm{C-F}}$ ($\AA$)", fontsize=8)
    inset.set_ylabel("Error (meV/atom)", fontsize=8)
    inset.tick_params(labelsize=8)
    inset.set_title(
        f"MAE = {metrics['relative_energy_mae_meV_per_atom']:.2f} meV/atom",
        fontsize=8,
    )

    figure.savefig(args.output, dpi=300)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)
    print(json.dumps(metrics, indent=2))
    print(f"Figure: {args.output}")


if __name__ == "__main__":
    main()
