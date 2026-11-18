# Writes data/MgO_bulk_eps_prior.xyz: the dielectric prior structure for the Au2/MgO runs (2026-10-09).
# Usage (this directory): python make_mgo_eps_prior.py
"""Bulk rocksalt MgO with its measured eps_inf, for ``electrostatics.dielectric_prior``.

- Geometry: the 8-atom conventional cell at the lattice constant of the Au2/MgO DFT data: the slab
  cell is |a1| = 9.0474 A = 3 a / sqrt(2) (a 3 x 3 (001) surface cell), so a = 4.2650 A.
  The 8-atom cell gives the model's eps_inf to 7e-11 of the 64-atom 2x2x2 cell (step-1 checkpoint:
  7.5084156275 vs 7.5084156270), so no image artifact.
- Label: eps_inf = 2.95 (isotropic), Sun, Allen, Stahnke, Jacobsen & Homes, arXiv:0704.2267, Table I
  (pure MgO, 295 K, reflectance fit); their Table II compiles 2.96.  A GGA functional is reported a few
  per cent higher; the label is the measured value.
- No energies or forces: the prior structure enters no energy, force or validation loss.
"""
import math

import numpy as np
from ase.build import bulk
from ase.io import read, write

SLAB_CELL = 9.0474                              # |a1| of the Au2/MgO DFT cells (3 x 3 surface cell)
A = SLAB_CELL * math.sqrt(2.0) / 3.0              # 4.2650 A
EPS_INF = 2.95

if __name__ == "__main__":
    first = read("data/AuMgO_test.xyz", 0)
    assert abs(float(np.linalg.norm(first.cell[0])) - SLAB_CELL) < 1e-4, "the slab cell differs from 9.0474 A"
    atoms = bulk("MgO", "rocksalt", a=A, cubic=True)
    atoms.info["dielectric_tensor"] = (EPS_INF * np.eye(3)).tolist()
    atoms.info["charge"] = 0
    write("data/MgO_bulk_eps_prior.xyz", [atoms], format="extxyz")
    back = read("data/MgO_bulk_eps_prior.xyz", 0)
    # ase's extxyz reader moves dielectric_tensor into calc.results (train.py reads it from there first)
    label = back.calc.results["dielectric_tensor"] if back.calc is not None else back.info["dielectric_tensor"]
    assert np.allclose(np.asarray(label).reshape(3, 3), EPS_INF * np.eye(3), rtol=0, atol=0)
    print(f"wrote data/MgO_bulk_eps_prior.xyz: {back.get_chemical_formula()}, a = {A:.4f} A, "
          f"eps_inf label {EPS_INF} x I, pbc {back.pbc.tolist()}")
