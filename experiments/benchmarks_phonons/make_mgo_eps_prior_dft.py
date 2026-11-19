# Writes data/MgO_bulk_eps_prior_pbesol.xyz and data/MgO_bulk_eps_prior_pbe.xyz (decision of 2026-11-12: use the DFT
# value 3.24).  Usage (repo root): python experiments/benchmarks_phonons/make_mgo_eps_prior_dft.py
"""Bulk rocksalt MgO with the DFT eps_inf of the phonon references, for ``electrostatics.dielectric_prior``.

A model trained on DFT forces is compared with DFT phonons, whose LO branch is set by DFT's own Z*^2 / eps_inf.  The
earlier prior (benchmarks_4g/data/MgO_bulk_eps_prior.xyz, kept) carries the measured 2.95 at the slab data's lattice
constant.  These two carry each reference's own DFPT value at its own lattice constant (benchmarks_phonons README,
"Reference"; score_phonons.REFS):

- PBEsol: phonondb mp-1265, a = 4.21725941 A, eps_inf 3.2417;
- PBE: Togo's phono3py set, a = 4.255656884758435 A, eps_inf 3.235.

The 8-atom conventional cell, as the earlier prior (its eps_inf equals the 64-atom cell's to 7e-11 on a model in its
frame configuration).  No energies or forces: the structure enters no energy, force or validation loss.
"""
import os
import sys

import numpy as np
from ase.build import bulk
from ase.io import read, write

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_phonons import REFS  # noqa: E402

if __name__ == "__main__":
    for name in ("pbesol", "pbe"):
        a, eps = float(REFS[name]["a"]), float(REFS[name]["eps"])
        atoms = bulk("MgO", "rocksalt", a=a, cubic=True)
        atoms.info["dielectric_tensor"] = (eps * np.eye(3)).tolist()
        atoms.info["charge"] = 0
        path = os.path.join(HERE, "data", f"MgO_bulk_eps_prior_{name}.xyz")
        assert not os.path.exists(path), f"{path} exists; it is not overwritten"
        write(path, [atoms], format="extxyz")
        back = read(path, 0)
        label = back.calc.results["dielectric_tensor"] if back.calc is not None else back.info["dielectric_tensor"]
        assert np.allclose(np.asarray(label).reshape(3, 3), eps * np.eye(3), rtol=0, atol=0)
        print(f"wrote {os.path.relpath(path)}: {back.get_chemical_formula()}, a = {a:.8f} A, eps_inf label {eps} x I, pbc {back.pbc.tolist()}")
