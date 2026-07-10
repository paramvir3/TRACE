"""Compile a fixed-capacity TRACE v3/v4 energy/force/virial program for AOTI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
from ase.data import atomic_numbers
from ase.io import read

from transformers_ace.aot import compile_aot_force_program, make_aot_compatible, pad_lammps_inputs
from transformers_ace import TransformersACECalculator
from transformers_ace.deploy import LAMMPSEnergyModel, LAMMPSAOTForceModel, _example_tensors, _synthetic_atoms


def _metadata(
    type_map: Sequence[str],
    r_max: float,
    max_atoms: int,
    max_edges: int,
) -> str:
    return "\n".join(
        (
            "format=transformers_ace_aoti_v1",
            "units=metal",
            "architecture=trace_v3_or_v4",
            f"r_max={float(r_max):.12g}",
            f"max_atoms={int(max_atoms)}",
            f"max_edges={int(max_edges)}",
            "outputs=energy_eV forces_eV_per_angstrom virial_eV",
            "force_convention=negative_position_gradient",
            "virial_convention=minus_diagonal_strain_gradient_minus_half_shear_gradient",
            "type_symbols=" + " ".join(type_map),
            "type_atomic_numbers=" + " ".join(str(atomic_numbers[symbol]) for symbol in type_map),
            "",
        )
    )


def compile_lammps_aot_model(
    checkpoint: Path,
    output: Path,
    type_map: Sequence[str],
    max_atoms: int,
    max_edges: int,
    example_structure: Path | None = None,
    device: str = "cuda",
) -> Path:
    """Create an architecture-specific AOTI shared library on the target GPU host."""
    if not device.startswith("cuda"):
        raise ValueError("AOT LAMMPS deployment is intended for a CUDA target device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; compile this artifact on the H100/B100 target host")

    calculator = TransformersACECalculator(model_path=str(checkpoint), device=device)
    if int(getattr(calculator.model, "architecture_version", 0)) not in (3, 4):
        raise ValueError("AOT LAMMPS deployment currently supports TRACE architecture_version=3 or 4")
    if calculator.model.l_max > 3:
        raise ValueError("AOT LAMMPS deployment currently supports l_max <= 3")

    source_model = calculator.model.eval()
    export_model = make_aot_compatible(source_model).to(device).eval()
    atomic_energy_tensor = calculator.atomic_energy_tensor
    if atomic_energy_tensor is not None:
        atomic_energy_tensor = atomic_energy_tensor.detach().to(device)
    energy_model = LAMMPSEnergyModel(
        export_model,
        atomic_energy_tensor=atomic_energy_tensor,
        energy_shift_per_atom=calculator.energy_shift_per_atom,
    ).to(device).eval()
    program = LAMMPSAOTForceModel(energy_model).to(device).eval()

    atoms = (
        read(example_structure.expanduser())
        if example_structure is not None
        else _synthetic_atoms(type_map, calculator.r_max)
    )
    inputs = tuple(tensor.to(device) for tensor in _example_tensors(atoms, calculator.r_max))
    padded_inputs = pad_lammps_inputs(
        inputs,
        max_atoms=max_atoms,
        max_edges=max_edges,
        r_max=calculator.r_max,
    )
    output = compile_aot_force_program(program, padded_inputs, output)
    output.with_suffix(output.suffix + ".metadata.txt").write_text(
        _metadata(type_map, calculator.r_max, max_atoms, max_edges)
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output AOTI shared library, e.g. trace_v3.so")
    parser.add_argument("--type-map", nargs="+", required=True)
    parser.add_argument("--max-atoms", type=int, required=True, help="Maximum local-plus-ghost atoms per MPI rank")
    parser.add_argument("--max-edges", type=int, required=True, help="Maximum directed local edges per MPI rank")
    parser.add_argument("--example-structure", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = compile_lammps_aot_model(
        checkpoint=args.checkpoint.expanduser().resolve(),
        output=args.output,
        type_map=args.type_map,
        max_atoms=args.max_atoms,
        max_edges=args.max_edges,
        example_structure=args.example_structure,
        device=args.device,
    )
    print(f"Wrote TRACE AOTI force program: {output}")
    print(f"Wrote metadata: {output.with_suffix(output.suffix + '.metadata.txt')}")


if __name__ == "__main__":
    main()
