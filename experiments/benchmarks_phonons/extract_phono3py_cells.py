"""Displaced supercells with PBE forces from a phono3py_params.yaml (Togo's PBE set, NIMS MDR, CC BY 4.0) -> extxyz.
Every entry of `phonon_displacements` (fc2 supercell) and `displacement_pairs` (fc3 supercell) is one structure: the
supercell with the displacement(s) applied and the VASP forces.  No energies exist in the dataset: `energy` is not written
and `has_energy=F` is set in info, so a trainer must mask the energy term for these structures.
Usage: python -I extract_phono3py_cells.py PHONO3PY_PARAMS_YAML OUT_EXTXYZ   (downloaded data: -I, paths as arguments)"""
import sys
import numpy as np, yaml
from ase import Atoms
from ase.io import write
src, out = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(src))
def cell_atoms(block):
    return Atoms(symbols=[p["symbol"] for p in block["points"]], cell=np.array(block["lattice"]), scaled_positions=np.array([p["coordinates"] for p in block["points"]]), pbc=True)
structures = []
fc2 = cell_atoms(d["phonon_supercell"])
for x in d["phonon_displacements"]:
    a = fc2.copy(); a.positions[int(x["atom"]) - 1] += np.array(x["displacement"])
    a.arrays["forces"] = np.array(x["forces"]); a.info.update(source="togo_phono3py_PBE_MgO_fc2", has_energy="F", charge=0.0); structures.append(a)
sc = cell_atoms(d["supercell"])
for first in d["displacement_pairs"]:
    i, di = int(first["atom"]) - 1, np.array(first["displacement"])
    if "forces" in first:                                              # the single displacement on the fc3 supercell
        a = sc.copy(); a.positions[i] += di
        a.arrays["forces"] = np.array(first["forces"]); a.info.update(source="togo_phono3py_PBE_MgO_fc3_single", has_energy="F", charge=0.0); structures.append(a)
    for second in first.get("paired_with", first.get("second_atoms", [])):
        a = sc.copy(); a.positions[i] += di; a.positions[int(second["atom"]) - 1] += np.array(second["displacement"])
        a.arrays["forces"] = np.array(second["forces"]); a.info.update(source="togo_phono3py_PBE_MgO_fc3", has_energy="F", charge=0.0); structures.append(a)
write(out, structures)
n2 = sum(1 for a in structures if a.info["source"].endswith("fc2")); print(f"wrote {len(structures)} structures ({n2} on the {len(fc2)}-atom fc2 supercell, {len(structures) - n2} on the {len(sc)}-atom fc3 supercell) to {out}")
f = np.concatenate([a.arrays["forces"].ravel() for a in structures]); print(f"forces: max |F| {np.abs(f).max():.3f} eV/A, rms {np.sqrt((f**2).mean()):.4f}")
