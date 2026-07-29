#!/usr/bin/env python3
"""Plot and audit one R1 TRACE/PLUMED well-tempered metadynamics run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io.trajectory import Trajectory


EV_TO_KCAL_MOL = 23.06054783061903


def read_plumed_table(path: Path) -> tuple[list[str], np.ndarray]:
    fields = None
    with path.open() as handle:
        for line in handle:
            if line.startswith("#! FIELDS"):
                fields = line.split()[2:]
                break
    if fields is None:
        raise ValueError(f"No '#! FIELDS' header found in {path}")
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != len(fields):
        raise ValueError(
            f"Header in {path} has {len(fields)} fields but data has "
            f"{data.shape[1]} columns"
        )
    return fields, data


def named_columns(fields: list[str], data: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(data[:, index]) for index, name in enumerate(fields)}


def find_plumed(requested: str | None) -> str:
    if requested:
        executable = Path(requested).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"PLUMED executable not found: {executable}")
        return str(executable)
    executable = shutil.which("plumed")
    if executable is None:
        raise FileNotFoundError(
            "PLUMED was not found on PATH. Source PLUMED's sourceme.sh or "
            "pass --plumed /absolute/path/to/plumed."
        )
    return executable


def reconstruct_fes(
    executable: str,
    hills: Path,
    grid_min: float,
    grid_max: float,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="trace-r1-sum-hills-") as directory:
        output = Path(directory) / "fes.dat"
        environment = os.environ.copy()
        environment.setdefault("OMPI_MCA_btl", "self")
        command = [
            executable,
            "sum_hills",
            "--hills",
            str(hills),
            "--outfile",
            str(output),
            "--min",
            str(grid_min),
            "--max",
            str(grid_max),
            "--bin",
            str(bins - 1),
            "--mintozero",
        ]
        result = subprocess.run(
            command,
            cwd=directory,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"PLUMED sum_hills failed with exit code {result.returncode}:\n"
                f"{result.stdout}"
            )
        data = np.loadtxt(output, comments="#", ndmin=2)
    if data.shape[1] < 2:
        raise ValueError("PLUMED sum_hills did not produce a one-dimensional FES")
    coordinate = np.asarray(data[:, 0])
    free_energy = np.asarray(data[:, 1])
    free_energy -= np.nanmin(free_energy)
    return coordinate, free_energy


def trajectory_coordinates(
    path: Path,
) -> tuple[np.ndarray, list[str]]:
    trajectory = Trajectory(str(path))
    if len(trajectory) == 0:
        raise ValueError(f"Trajectory is empty: {path}")
    symbols = trajectory[0].get_chemical_symbols()
    carbon = [index for index, symbol in enumerate(symbols) if symbol == "C"]
    fluorine = [index for index, symbol in enumerate(symbols) if symbol == "F"]
    chlorine = [index for index, symbol in enumerate(symbols) if symbol == "Cl"]
    hydrogens = [index for index, symbol in enumerate(symbols) if symbol == "H"]
    if len(carbon) != 1 or len(fluorine) != 1 or len(chlorine) != 1:
        raise ValueError("R1 analysis requires exactly one C, F, and Cl atom")
    if not hydrogens:
        raise ValueError("R1 analysis requires at least one H atom")

    rows = []
    for atoms in trajectory:
        positions = atoms.positions
        c = positions[carbon[0]]
        r_cf = np.linalg.norm(c - positions[fluorine[0]])
        r_ccl = np.linalg.norm(c - positions[chlorine[0]])
        r_ch = [np.linalg.norm(c - positions[index]) for index in hydrogens]
        rows.append([r_cf, r_ccl, *r_ch])
    labels = ["r_C-F_A", "r_C-Cl_A"] + [
        f"r_C-H{index + 1}_A" for index in range(len(hydrogens))
    ]
    return np.asarray(rows, dtype=float), labels


def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, header: list[str], columns: list[np.ndarray]) -> None:
    length = min(len(column) for column in columns)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(zip(*(column[:length] for column in columns)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plumed", default=None)
    parser.add_argument("--grid-min", type=float, default=-2.0)
    parser.add_argument("--grid-max", type=float, default=2.0)
    parser.add_argument("--bins", type=int, default=401)
    parser.add_argument(
        "--max-c-h",
        type=float,
        default=1.5,
        help="C-H distance above which the molecular trajectory is invalid (A)",
    )
    args = parser.parse_args()

    run = args.run.expanduser().resolve()
    output = (args.output or (run / "analysis")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.grid_min >= args.grid_max or args.bins < 50:
        raise ValueError("Use an increasing FES interval and at least 50 bins")
    if args.max_c_h <= 0.0:
        raise ValueError("The C-H validity threshold must be positive")

    required = ["COLVAR", "HILLS", "trajectory.traj", "md.log", "run_metadata.json"]
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run is missing required files: {', '.join(missing)}")

    metadata = json.loads((run / "run_metadata.json").read_text())
    fields, table = read_plumed_table(run / "COLVAR")
    colvar = named_columns(fields, table)
    required_fields = {
        "time",
        "r_f",
        "r_cl",
        "avg_r",
        "diff_r",
        "avg_wall.bias",
        "metad.bias",
    }
    absent = sorted(required_fields.difference(colvar))
    if absent:
        raise ValueError(f"COLVAR is missing fields: {', '.join(absent)}")

    sample_interval_ps = float(metadata["sample_interval_fs"]) / 1000.0
    time_ps = np.arange(len(table), dtype=float) * sample_interval_ps
    raw_time = colvar["time"]
    raw_time_matches_ps = bool(
        len(raw_time) == len(time_ps)
        and np.allclose(raw_time, time_ps, rtol=1.0e-5, atol=1.0e-8)
    )
    raw_time_scale_to_ps = None
    nonzero = np.abs(raw_time) > 0.0
    if nonzero.any():
        raw_time_scale_to_ps = float(np.median(time_ps[nonzero] / raw_time[nonzero]))

    trajectory, trajectory_labels = trajectory_coordinates(run / "trajectory.traj")
    trajectory_time_ps = np.arange(len(trajectory), dtype=float) * sample_interval_ps
    ch_distances = trajectory[:, 2:]
    max_ch = np.max(ch_distances, axis=1)
    invalid_indices = np.flatnonzero(max_ch > args.max_c_h)
    first_invalid_index = int(invalid_indices[0]) if len(invalid_indices) else None
    first_invalid_ps = (
        float(trajectory_time_ps[first_invalid_index])
        if first_invalid_index is not None
        else None
    )

    md = np.loadtxt(run / "md.log", skiprows=1, ndmin=2)
    if md.shape[1] < 5:
        raise ValueError("md.log does not contain the expected five columns")
    md_time_ps = md[:, 0]
    temperature = md[:, 4]

    executable = find_plumed(args.plumed)
    fes_coordinate, fes_ev = reconstruct_fes(
        executable,
        run / "HILLS",
        args.grid_min,
        args.grid_max,
        args.bins,
    )
    fes_kcal = fes_ev * EV_TO_KCAL_MOL
    stable = first_invalid_index is None
    validity_message = (
        "No C-H distance exceeded the configured validity threshold."
        if stable
        else (
            f"Diagnostic only: C-H rupture begins at {first_invalid_ps:.2f} ps "
            f"(threshold {args.max_c_h:.2f} A)."
        )
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
        }
    )
    colors = {"cf": "#0072B2", "ccl": "#D55E00", "cv": "#222222"}

    figure, axes = plt.subplots(3, 1, figsize=(8.0, 8.3), sharex=True, constrained_layout=True)
    axes[0].plot(time_ps, colvar["r_f"], color=colors["cf"], lw=1.0, label=r"$r_{\mathrm{C-F}}$")
    axes[0].plot(time_ps, colvar["r_cl"], color=colors["ccl"], lw=1.0, label=r"$r_{\mathrm{C-Cl}}$")
    axes[0].set_ylabel("Distance (A)")
    axes[0].legend(ncol=2, loc="upper left")
    axes[0].set_title("R1 WTMetaD collective variables")
    axes[1].plot(time_ps, colvar["diff_r"], color=colors["cv"], lw=0.9)
    axes[1].axhline(0.0, color="#888888", lw=0.8, ls="--")
    axes[1].set_ylabel(r"$s=r_{\mathrm{C-Cl}}-r_{\mathrm{C-F}}$ (A)")
    axes[2].plot(time_ps, colvar["metad.bias"], color="#009E73", lw=1.0, label="WTMetaD bias")
    axes[2].plot(time_ps, colvar["avg_wall.bias"], color="#CC79A7", lw=1.0, label="wall bias")
    axes[2].set_ylabel("Bias energy (eV)")
    axes[2].set_xlabel("Time (ps)")
    axes[2].legend(ncol=2)
    if first_invalid_ps is not None:
        for axis in axes:
            axis.axvspan(first_invalid_ps, time_ps[-1], color="#E69F00", alpha=0.12)
            axis.axvline(first_invalid_ps, color="#B35C00", lw=0.8, ls=":")
        axes[1].text(
            0.99,
            0.95,
            validity_message,
            transform=axes[1].transAxes,
            ha="right",
            va="top",
            color="#8A3B00",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
    save_figure(figure, output / "colvar")

    common = min(len(trajectory), len(time_ps))
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 8.0), constrained_layout=True)
    valid_count = first_invalid_index if first_invalid_index is not None else common
    scatter = axes[0].scatter(
        trajectory[:valid_count, 0],
        trajectory[:valid_count, 1],
        c=trajectory_time_ps[:valid_count],
        s=12,
        cmap="viridis",
        linewidths=0.0,
        label="connectivity retained",
    )
    if valid_count < common:
        axes[0].scatter(
            trajectory[valid_count:common, 0],
            trajectory[valid_count:common, 1],
            s=9,
            color="#D55E00",
            marker="x",
            alpha=0.45,
            label="after C-H rupture",
        )
    colorbar = figure.colorbar(scatter, ax=axes[0], pad=0.02)
    colorbar.set_label("Time (ps)")
    axes[0].plot([0.4, 4.6], [0.4, 4.6], color="#888888", ls="--", lw=0.8)
    axes[0].set_xlabel(r"$r_{\mathrm{C-F}}$ (A)")
    axes[0].set_ylabel(r"$r_{\mathrm{C-Cl}}$ (A)")
    axes[0].set_title("Trajectory in the substitution-coordinate plane")
    axes[0].legend(loc="upper left")

    for index in range(ch_distances.shape[1]):
        axes[1].plot(
            trajectory_time_ps,
            ch_distances[:, index],
            lw=0.8,
            label=rf"$r_{{\mathrm{{C-H}}_{index + 1}}}$",
        )
    axes[1].axhline(args.max_c_h, color="#222222", lw=0.8, ls="--", label="validity threshold")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_ylabel("C-H distance (A)")
    temperature_axis = axes[1].twinx()
    temperature_axis.plot(md_time_ps, temperature, color="#999999", lw=0.65, alpha=0.55)
    temperature_axis.set_ylabel("Instantaneous temperature (K)", color="#666666")
    axes[1].legend(ncol=2, loc="upper left")
    save_figure(figure, output / "trajectory")

    figure, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    axis.plot(fes_coordinate, fes_kcal, color="#0072B2", lw=2.0)
    axis.set_xlabel(r"$s=r_{\mathrm{C-Cl}}-r_{\mathrm{C-F}}$ (A)")
    axis.set_ylabel(r"$\Delta G$ (kcal mol$^{-1}$)")
    axis.set_title("Final well-tempered bias estimate")
    if not stable:
        axis.text(
            0.03,
            0.95,
            validity_message + "\nThis curve is not a reportable free energy.",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="#8A3B00",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#D55E00", "alpha": 0.92},
        )
    save_figure(figure, output / "free_energy")

    write_csv(
        output / "colvar.csv",
        ["time_ps"] + fields[1:],
        [time_ps] + [colvar[name] for name in fields[1:]],
    )
    write_csv(
        output / "trajectory_diagnostics.csv",
        ["time_ps"] + trajectory_labels + ["max_C-H_A"],
        [trajectory_time_ps] + [trajectory[:, index] for index in range(trajectory.shape[1])] + [max_ch],
    )
    write_csv(
        output / "free_energy.csv",
        ["cv_r_C-Cl_minus_r_C-F_A", "free_energy_eV", "free_energy_kcal_per_mol"],
        [fes_coordinate, fes_ev, fes_kcal],
    )

    valid_colvar_count = min(valid_count, len(colvar["diff_r"]))
    valid_cv = colvar["diff_r"][:valid_colvar_count]
    crossings_before_failure = int(
        np.count_nonzero(np.diff(np.signbit(valid_cv))) if len(valid_cv) > 1 else 0
    )
    summary = {
        "run": str(run),
        "output": str(output),
        "saved_duration_ps": float(time_ps[-1]),
        "colvar_rows": int(len(table)),
        "trajectory_frames": int(len(trajectory)),
        "raw_colvar_time_matches_ps": raw_time_matches_ps,
        "raw_colvar_time_scale_to_ps": raw_time_scale_to_ps,
        "maximum_C-H_distance_A": float(np.max(max_ch)),
        "C-H_validity_threshold_A": args.max_c_h,
        "first_C-H_threshold_crossing_ps": first_invalid_ps,
        "maximum_instantaneous_temperature_K": float(np.max(temperature)),
        "median_instantaneous_temperature_K": float(np.median(temperature)),
        "CV_zero_crossings_before_C-H_failure": crossings_before_failure,
        "free_energy_estimator": "PLUMED sum_hills final well-tempered bias",
        "free_energy_valid_for_reporting": stable,
        "validity_message": validity_message,
    }
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Plots and tables: {output}")
    if not stable:
        print("WARNING: the reconstructed FES is diagnostic only; rerun after stabilizing the potential and sampling protocol.")


if __name__ == "__main__":
    main()
