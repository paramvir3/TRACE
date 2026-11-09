"""Togo's PBEsol lattice-thermal-conductivity training set for MgO -> extended xyz for TRACE.

Source: A. Togo, NIMS Materials Data Repository, https://mdr.nims.go.jp/datasets/59c5d436-339d-4b32-ba59-a3da5ea68bc7
(CC BY 4.0): `phonopy_training_dataset.yaml` (VASP, PBEsol, 200 rocksalt 2x2x2 supercells of 64 atoms at a = 4.21726 A,
every atom displaced 0.03 A in a random direction; forces and supercell energies).

Usage (repo root): python examples/mgo_phonons_local/extract_ltc_pbesol.py PHONOPY_TRAINING_YAML examples/mgo_phonons_local/data/MgO_PBEsol_ltc
Writes <prefix>_train.xyz (180 cells) and <prefix>_valid.xyz (20), split with numpy seed 0, energies and forces both kept.
"""
import sys

import numpy as np
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

d = yaml.load(open(sys.argv[1]), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
sc = d["supercell"]
cell = np.array(sc["lattice"])
sym = [p["symbol"] for p in sc["points"]]
pos0 = np.array([p["coordinates"] for p in sc["points"]]) @ cell
ds = d["dataset"]
disp, forc, E = np.array(ds["displacements"]), np.array(ds["forces"]), np.array(ds["supercell_energies"])
assert disp.shape == forc.shape == (len(E), len(sym), 3), (disp.shape, forc.shape, len(E), len(sym))
frames = []
for k in range(len(E)):
    a = Atoms(symbols=sym, positions=pos0 + disp[k], cell=cell, pbc=True)
    a.calc = SinglePointCalculator(a, energy=float(E[k]), forces=forc[k])
    a.info.update(source="togo_mdr_ltc_PBEsol_MgO", ltc_index=k)
    frames.append(a)
idx = np.random.default_rng(0).permutation(len(frames))
val = set(idx[:20].tolist())
write(sys.argv[2] + "_train.xyz", [f for i, f in enumerate(frames) if i not in val])
write(sys.argv[2] + "_valid.xyz", [f for i, f in enumerate(frames) if i in val])
print(f"{len(frames)} cells ({len(sym)} atoms, a = {cell[0, 0] / 2:.5f} A): train {len(frames) - 20}, valid 20; "
      f"RMS force {np.sqrt((forc ** 2).mean()):.4f} eV/A")
