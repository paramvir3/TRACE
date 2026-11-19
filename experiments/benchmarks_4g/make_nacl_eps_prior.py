# Writes data/NaCl_bulk_eps_prior.xyz: the dielectric prior structure for the NaCl step-9 run (2026-10-16).
# Usage (this directory): python make_nacl_eps_prior.py
"""Bulk rocksalt NaCl with its measured eps_inf, for ``electrostatics.dielectric_prior`` (the NaCl counterpart of
make_mgo_eps_prior.py).

- Label: eps_inf = 2.3295 (isotropic): the electronic dielectric constant, the lambda -> infinity limit of the
  ultraviolet (electronic) terms of the Sellmeier fit of H. H. Li, J. Phys. Chem. Ref. Data 5, 329 (1976), NaCl at
  297 K: 1 + 0.00055 + 0.19800 + 0.48398 + 0.38696 + 0.25998 = 2.32947.  Check: adding the three infrared (phonon)
  terms gives eps(0) = 5.89, the measured static constant (~5.9).
- Geometry: the 8-atom conventional cell at the room-temperature lattice constant a = 5.640 A (the clusters of the
  data have no bulk lattice to take it from).
- No energies or forces: the prior structure enters no energy, force or validation loss.
"""
import numpy as np
from ase.build import bulk
from ase.io import read, write

A = 5.640
EPS_INF = 1.0 + 0.00055 + 0.19800 + 0.48398 + 0.38696 + 0.25998

if __name__ == "__main__":
    atoms = bulk("NaCl", "rocksalt", a=A, cubic=True)
    atoms.info["dielectric_tensor"] = (EPS_INF * np.eye(3)).tolist()
    atoms.info["charge"] = 0
    write("data_pfn/NaCl_bulk_eps_prior.xyz", [atoms], format="extxyz")
    back = read("data_pfn/NaCl_bulk_eps_prior.xyz", 0)
    print(back, back.info.get("dielectric_tensor"))
