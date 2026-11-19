# Togo's PBEsol LTC training set for MgO (NIMS MDR 59c5d436, CC BY 4.0) -> extxyz: the 2x2x2 supercell (from the yaml's
# `supercell` block, atom order of the displacements) + displacements, VASP forces, supercell energy kept in info only
# (has_energy=F: VASP's zero differs from the FHI-aims references), split 180 / 20 with numpy seed 0.
# Usage: python -I extract_ltc_pbesol.py PHONOPY_TRAINING_YAML OUT_PREFIX
import sys, yaml, numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
d = yaml.load(open(sys.argv[1]), Loader=yaml.CSafeLoader if hasattr(yaml, "CSafeLoader") else yaml.SafeLoader)
sc = d["supercell"]; cell = np.array(sc["lattice"]); sym = [p["symbol"] for p in sc["points"]]; frac = np.array([p["coordinates"] for p in sc["points"]])
pos0 = frac @ cell; ds = d["dataset"]
disp, forc, E = np.array(ds["displacements"]), np.array(ds["forces"]), np.array(ds["supercell_energies"])
assert disp.shape == forc.shape == (len(E), len(sym), 3), (disp.shape, forc.shape, len(E), len(sym))
frames = []
for k in range(len(E)):
    a = Atoms(symbols=sym, positions=pos0 + disp[k], cell=cell, pbc=True)
    a.calc = SinglePointCalculator(a, forces=forc[k])
    a.info.update(source="togo_mdr_ltc2026_PBEsol_MgO", vasp_energy=float(E[k]), has_energy=False, charge=0.0, ltc_index=k)
    frames.append(a)
idx = np.random.default_rng(0).permutation(len(frames)); val = set(idx[:20].tolist())
train = [f for i, f in enumerate(frames) if i not in val]; valid = [f for i, f in enumerate(frames) if i in val]
write(sys.argv[2] + "_train.xyz", train); write(sys.argv[2] + "_valid.xyz", valid)
print(f"{len(frames)} cells ({len(sym)} atoms, a = {cell[0, 0] / 2:.5f} A): train {len(train)}, valid {len(valid)}; RMS force {np.sqrt((forc ** 2).mean()):.4f} eV/A; net force per cell max {np.abs(forc.sum(axis=1)).max():.2e} eV/A")
