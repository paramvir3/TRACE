#!/usr/bin/env python3
"""Run a continuous R1 WTMetaD coverage test with TRACE, ASE, and PLUMED."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import units
from ase.constraints import FixCom
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)


EXAMPLE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLE_ROOT.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from transformers_ace import TransformersACECalculator  # noqa: E402


def unique_index(symbols: list[str], symbol: str) -> int:
    indices = [index for index, value in enumerate(symbols) if value == symbol]
    if len(indices) != 1:
        raise ValueError(f"Expected one {symbol} atom, found {len(indices)}")
    return indices[0]


def plumed_lines(
    atoms,
    output: Path,
    temperature: float,
    time_step_fs: float,
    sample_fs: float,
) -> list[str]:
    symbols = list(atoms.symbols)
    carbon = unique_index(symbols, "C") + 1
    fluorine = unique_index(symbols, "F") + 1
    chlorine = unique_index(symbols, "Cl") + 1
    pace = int(round(10.0 / time_step_fs))
    stride = int(round(sample_fs / time_step_fs))
    if not np.isclose(pace * time_step_fs, 10.0):
        raise ValueError("The timestep must represent the published 10 fs hill pace exactly")
    if not np.isclose(stride * time_step_fs, sample_fs):
        raise ValueError("The timestep must represent the requested output interval exactly")

    height = 5.0 * units.kB * temperature
    return [
        "UNITS LENGTH=A TIME=ps ENERGY=eV",
        f"r_f: DISTANCE ATOMS={carbon},{fluorine}",
        f"r_cl: DISTANCE ATOMS={carbon},{chlorine}",
        "avg_r: COMBINE ARG=r_f,r_cl COEFFICIENTS=0.5,0.5 PERIODIC=NO",
        "avg_wall: UPPER_WALLS ARG=avg_r AT=2.5 KAPPA=1000.0 EXP=2",
        "diff_r: COMBINE ARG=r_cl,r_f COEFFICIENTS=1.0,-1.0 PERIODIC=NO",
        (
            f"metad: METAD ARG=diff_r PACE={pace} HEIGHT={height:.17g} "
            f"SIGMA=0.05 TEMP={temperature:.8g} BIASFACTOR=100 "
            f"FILE={output / 'HILLS'}"
        ),
        (
            "PRINT ARG=r_f,r_cl,avg_r,diff_r,avg_wall.bias,metad.bias "
            f"STRIDE={stride} FILE={output / 'COLVAR'}"
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=EXAMPLE_ROOT / "models/r1_trace_v2_combined.pt",
    )
    parser.add_argument(
        "--structure", type=Path, default=EXAMPLE_ROOT / "structures/ch3cl_f.xyz"
    )
    parser.add_argument(
        "--output", type=Path, default=EXAMPLE_ROOT / "results/wtmetad"
    )
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--time-step-fs", type=float, default=0.5)
    parser.add_argument("--duration-ps", type=float, default=20.0)
    parser.add_argument("--sample-fs", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=202602)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for name, value in (
        ("temperature", args.temperature),
        ("time step", args.time_step_fs),
        ("duration", args.duration_ps),
        ("sample interval", args.sample_fs),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")

    structure = args.structure.resolve()
    model = args.model.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output}. Pass --overwrite to "
            "replace this run, or choose a new directory with --output."
        )
    output.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for filename in (
            "COLVAR",
            "HILLS",
            "md.log",
            "plumed.dat",
            "plumed.log",
            "run_metadata.json",
            "trajectory.traj",
        ):
            path = output / filename
            if path.is_file():
                path.unlink()

    atoms = read(structure, index=0)
    atoms.pbc = False
    atoms.set_cell(np.zeros((3, 3)))
    input_lines = plumed_lines(
        atoms, output, args.temperature, args.time_step_fs, args.sample_fs
    )
    (output / "plumed.dat").write_text("\n".join(input_lines) + "\n")

    steps = int(round(1000.0 * args.duration_ps / args.time_step_fs))
    sample_stride = int(round(args.sample_fs / args.time_step_fs))
    metadata = {
        "model": str(model),
        "structure": str(structure),
        "temperature_K": args.temperature,
        "time_step_fs": args.time_step_fs,
        "duration_ps": args.duration_ps,
        "steps": steps,
        "sample_interval_fs": args.sample_fs,
        "seed": args.seed,
        "langevin_friction_inverse_ASE_time": 0.02,
        "center_of_mass_constraint": "ase.constraints.FixCom",
        "collective_variable": "r_C_Cl - r_C_F",
        "upper_wall": "0.5*(r_C_F+r_C_Cl) <= 2.5 A; kappa=1000 eV/A^2",
        "hill_sigma_A": 0.05,
        "hill_height": "5*k_B*T",
        "hill_pace_fs": 10.0,
        "bias_factor": 100,
        "interpretation": (
            "continuous final-model coverage test; not the paper's iterative "
            "WTMetaD-IB active-learning history"
        ),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.write_only:
        print(f"Wrote {output / 'plumed.dat'}")
        return

    try:
        import plumed  # noqa: F401
        from ase.calculators.plumed import Plumed
    except ImportError as exc:
        raise ImportError(
            "The PLUMED Python module is unavailable. Source the PLUMED "
            "sourceme.sh file in this shell and verify that `python -c "
            "\"import plumed\"` succeeds before running dynamics."
        ) from exc

    class ScalarEnergyPlumed(Plumed):
        """Normalize the PLUMED bias energy for ASE with NumPy 2."""

        def compute_energy_and_forces(self, positions, step):
            energy, forces = super().compute_energy_and_forces(positions, step)
            scalar_energy = float(np.asarray(energy).reshape(-1)[0])
            return scalar_energy, forces

    base_calculator = TransformersACECalculator(model_path=str(model), device=args.device)
    atoms.calc = ScalarEnergyPlumed(
        calc=base_calculator,
        input=input_lines,
        timestep=args.time_step_fs * units.fs,
        atoms=atoms,
        kT=units.kB * args.temperature,
        log=str(output / "plumed.log"),
    )

    rng = np.random.default_rng(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)
    atoms.set_constraint(FixCom())
    dynamics = Langevin(
        atoms,
        args.time_step_fs * units.fs,
        temperature_K=args.temperature,
        friction=0.02,
        rng=rng,
        fixcm=False,
    )
    trajectory = Trajectory(output / "trajectory.traj", "w", atoms)
    try:
        dynamics.attach(trajectory.write, interval=sample_stride)
        dynamics.attach(
            MDLogger(
                dynamics,
                atoms,
                output / "md.log",
                header=True,
                stress=False,
                peratom=False,
                mode="w",
            ),
            interval=sample_stride,
        )
        dynamics.run(steps)
    finally:
        trajectory.close()
        if hasattr(atoms.calc, "plumed"):
            atoms.calc.plumed.finalize()


if __name__ == "__main__":
    main()
