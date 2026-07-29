#!/usr/bin/env python3
"""Compute a constrained relaxed R1 PES scan for a Figure 2b analogue."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from ase.constraints import FixInternals
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS


EXAMPLE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLE_ROOT.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from transformers_ace import TransformersACECalculator  # noqa: E402


FIELDS = [
    "r_C_F_target_A",
    "r_C_Cl_target_A",
    "r_C_F_final_A",
    "r_C_Cl_final_A",
    "energy_eV",
    "maximum_projected_force_eV_per_A",
    "optimizer_steps",
    "converged",
]


def atom_indices(atoms) -> tuple[int, int, int]:
    symbols = list(atoms.symbols)
    indices = []
    for symbol in ("C", "F", "Cl"):
        matches = [index for index, value in enumerate(symbols) if value == symbol]
        if len(matches) != 1:
            raise ValueError(f"Expected one {symbol} atom, found {len(matches)}")
        indices.append(matches[0])
    return tuple(indices)


def distances(atoms, indices: tuple[int, int, int]) -> tuple[float, float]:
    carbon, fluorine, chlorine = indices
    r_f = float(np.linalg.norm(atoms.positions[carbon] - atoms.positions[fluorine]))
    r_cl = float(np.linalg.norm(atoms.positions[carbon] - atoms.positions[chlorine]))
    return r_f, r_cl


def set_distance(atoms, center: int, ligand: int, target: float) -> None:
    vector = atoms.positions[ligand] - atoms.positions[center]
    norm = np.linalg.norm(vector)
    if norm < 1.0e-10:
        raise ValueError("Cannot set a bond distance from coincident atoms")
    atoms.positions[ligand] = atoms.positions[center] + target * vector / norm


def nearest_seed(seeds, target_f: float, target_cl: float):
    best = None
    best_distance = float("inf")
    for atoms in seeds:
        r_f, r_cl = distances(atoms, atom_indices(atoms))
        value = (r_f - target_f) ** 2 + (r_cl - target_cl) ** 2
        if value < best_distance:
            best = atoms
            best_distance = value
    return best.copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=EXAMPLE_ROOT / "models/r1_trace_v2_combined.pt",
    )
    parser.add_argument(
        "--seeds", type=Path, default=EXAMPLE_ROOT / "data/combined_all.xyz"
    )
    parser.add_argument(
        "--output", type=Path, default=EXAMPLE_ROOT / "results/pes_scan.csv"
    )
    parser.add_argument("--r-min", type=float, default=1.5)
    parser.add_argument("--r-max", type=float, default=4.0)
    parser.add_argument("--n-points", type=int, default=26)
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-points", type=int, default=None)
    args = parser.parse_args()

    if args.r_min <= 0.0 or args.r_max <= args.r_min:
        raise ValueError("Require 0 < r_min < r_max")
    if args.n_points < 2 or args.fmax <= 0.0 or args.steps < 1:
        raise ValueError("Invalid scan resolution or optimizer settings")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output exists; pass --resume to continue: {output}")

    seeds = read(args.seeds, index=":")
    if not seeds:
        raise ValueError("No seed structures were read")
    for seed in seeds:
        seed.pbc = False
        seed.set_cell(np.zeros((3, 3)))
        atom_indices(seed)

    completed = set()
    if output.exists():
        with output.open(newline="") as handle:
            for row in csv.DictReader(handle):
                completed.add(
                    (round(float(row["r_C_F_target_A"]), 10),
                     round(float(row["r_C_Cl_target_A"]), 10))
                )

    metadata = {
        "model": str(args.model.resolve()),
        "seed_structures": str(args.seeds.resolve()),
        "r_min_A": args.r_min,
        "r_max_A": args.r_max,
        "n_points_per_axis": args.n_points,
        "fmax_eV_per_A": args.fmax,
        "maximum_steps": args.steps,
        "constraints": "fixed C-F and C-Cl distances; all remaining DOFs relaxed",
    }
    metadata_path = output.with_suffix(".json")
    if args.resume:
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Cannot validate the resumed scan without {metadata_path}"
            )
        existing_metadata = json.loads(metadata_path.read_text())
        if existing_metadata != metadata:
            raise ValueError(
                "The resumed model or scan settings differ from the original run"
            )
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    calculator = TransformersACECalculator(
        model_path=str(args.model.resolve()), device=args.device
    )
    trajectory_path = output.with_suffix(".traj")
    trajectory = Trajectory(trajectory_path, "a" if args.resume else "w")
    values = np.linspace(args.r_min, args.r_max, args.n_points)
    points = []
    for row_index, r_cl in enumerate(values):
        r_f_values = values if row_index % 2 == 0 else values[::-1]
        points.extend((float(r_f), float(r_cl)) for r_f in r_f_values)
    pending = [
        point for point in points
        if (round(point[0], 10), round(point[1], 10)) not in completed
    ]
    if args.max_points is not None:
        pending = pending[: args.max_points]

    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for point_index, (target_f, target_cl) in enumerate(pending, start=1):
            atoms = nearest_seed(seeds, target_f, target_cl)
            carbon, fluorine, chlorine = atom_indices(atoms)
            set_distance(atoms, carbon, fluorine, target_f)
            set_distance(atoms, carbon, chlorine, target_cl)
            atoms.set_constraint(
                FixInternals(
                    bonds=[
                        [target_f, [carbon, fluorine]],
                        [target_cl, [carbon, chlorine]],
                    ]
                )
            )
            atoms.calc = calculator
            optimizer = LBFGS(atoms, logfile=None)
            converged = bool(optimizer.run(fmax=args.fmax, steps=args.steps))
            final_f, final_cl = distances(atoms, (carbon, fluorine, chlorine))
            projected_forces = atoms.get_forces()
            record = {
                "r_C_F_target_A": target_f,
                "r_C_Cl_target_A": target_cl,
                "r_C_F_final_A": final_f,
                "r_C_Cl_final_A": final_cl,
                "energy_eV": float(atoms.get_potential_energy()),
                "maximum_projected_force_eV_per_A": float(
                    np.linalg.norm(projected_forces, axis=1).max()
                ),
                "optimizer_steps": optimizer.nsteps,
                "converged": int(converged),
            }
            writer.writerow(record)
            handle.flush()
            if converged:
                trajectory.write(atoms)
            print(
                f"{point_index}/{len(pending)} r_CF={target_f:.3f} "
                f"r_CCl={target_cl:.3f} E={record['energy_eV']:.6f} "
                f"converged={converged}"
            )
    trajectory.close()
    print(f"Scan table: {output}")


if __name__ == "__main__":
    main()
