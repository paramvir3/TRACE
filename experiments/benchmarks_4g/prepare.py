"""4G-HDNNP benchmark datasets (Ko, Finkler, Goedecker, Behler, Nat. Commun. 12, 398 (2021))
-> extxyz in eV / A, with fixed-seed train / valid / test subsets.

Source: Materials Cloud record 2020.137 (CC-BY-4.0), ``datasets.tar.gz``,
RuNNer ``input.data`` in atomic units (Bohr, Ha, Ha/Bohr, e), PBE/FHI-aims
'light'.  Each structure: ``lattice`` lines (periodic, Bohr), ``atom x y z
element q_Hirshfeld e_atom fx fy fz``, ``energy`` (Ha), ``charge`` (total, e).

Written per structure:
* positions (A), cell + pbc=True when lattice lines are present;
* SinglePointCalculator energy (eV) and forces (eV/A);
* ``info['charge']`` -- the TOTAL charge, which trace_electrostatics reads;
* ``arrays['hirshfeld_charges']`` -- reference atomic charges, kept for
  comparison only (they are a partitioning convention, never trained on);
* ``info['system']`` -- formula + charge state, for per-subsystem errors.
Checked: sum of Hirshfeld charges == total charge (to 1e-6 e).

Usage: python prepare.py DATASETS_DIR NAME N_TRAIN N_VALID N_TEST [SEED] [OUT_DIR]
       NAME in {Ag_cluster, Carbon_chain, NaCl, AuMgO}; N_TEST = -1 takes every
       structure not in train or valid.
writes OUT_DIR/NAME_{train,valid,test}.xyz (default: data/ next to this script).
"""
import os
import sys

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

BOHR = 0.529177210903            # A   (CODATA 2018)
HARTREE = 27.211386245988        # eV  (CODATA 2018)
HERE = os.path.dirname(os.path.abspath(__file__))


def read_runner(path):
    frames, lattice, atoms, energy, charge = [], [], [], None, None
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            if key == "begin":
                lattice, atoms, energy, charge = [], [], None, None
            elif key == "lattice":
                lattice.append([float(x) * BOHR for x in parts[1:4]])
            elif key == "atom":
                atoms.append((parts[4], [float(x) * BOHR for x in parts[1:4]], float(parts[5]),
                              [float(x) * HARTREE / BOHR for x in parts[7:10]]))
            elif key == "energy":
                energy = float(parts[1]) * HARTREE
            elif key == "charge":
                charge = float(parts[1])
            elif key == "end":
                symbols = [a[0] for a in atoms]
                structure = Atoms(symbols, positions=[a[1] for a in atoms])
                if lattice:
                    if len(lattice) != 3:
                        raise ValueError(f"structure {len(frames)}: {len(lattice)} lattice lines")
                    structure.cell = lattice
                    structure.pbc = True
                hirshfeld = np.array([a[2] for a in atoms])
                if abs(hirshfeld.sum() - charge) > 1e-6:
                    raise ValueError(
                        f"structure {len(frames)}: Hirshfeld charges sum to {hirshfeld.sum():.6f}, "
                        f"total charge {charge}")
                structure.arrays["hirshfeld_charges"] = hirshfeld
                structure.info["charge"] = 0.0 if charge == 0 else charge     # no -0.0
                structure.info["system"] = f"{structure.get_chemical_formula()}{charge + 0.0:+.0f}"
                structure.calc = SinglePointCalculator(
                    structure, energy=energy, forces=np.array([a[3] for a in atoms]))
                frames.append(structure)
    return frames


def main():
    root, name = sys.argv[1], sys.argv[2]
    n_train, n_valid, n_test = (int(x) for x in sys.argv[3:6])
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    out_dir = sys.argv[7] if len(sys.argv) > 7 else os.path.join(HERE, "data")
    frames = read_runner(os.path.join(root, name, "input.data"))
    systems = sorted({f.info["system"] for f in frames})
    # Stratified by subsystem: every split has the same mix of charge states.
    rng = np.random.default_rng(seed)
    parts = {"train": [], "valid": [], "test": []}
    for system in systems:
        members = [i for i, f in enumerate(frames) if f.info["system"] == system]
        rng.shuffle(members)
        share = len(members) / len(frames)
        counts = [round(n_train * share), round(n_valid * share)]
        counts.append(len(members) - sum(counts) if n_test < 0 else round(n_test * share))
        start = 0
        for split, count in zip(("train", "valid", "test"), counts):
            parts[split] += members[start:start + count]
            start += count
    os.makedirs(out_dir, exist_ok=True)
    for split, indices in parts.items():
        write(os.path.join(out_dir, f"{name}_{split}.xyz"), [frames[i] for i in sorted(indices)],
              format="extxyz")
    print(f"{name}: {len(frames)} structures, systems {systems}; "
          + ", ".join(f"{k} {len(v)}" for k, v in parts.items()))


if __name__ == "__main__":
    main()
