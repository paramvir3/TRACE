# Writes data/Au_atom_alpha_prior.xyz: the polarizability prior structure for the Au2/MgO runs (2026-10-10).
# Usage (this directory): python make_au_alpha_prior.py
"""An isolated Au atom with its measured-quality static dipole polarizability, for
``electrostatics.polarizability_prior``.

- Label: alpha(Au) = 36.68 +- 0.78 a.u. (isotropic), relativistic coupled cluster with perturbative triples:
  Y. Cheng, "Relativistic and electron-correlation effects in static dipole polarizabilities for group 11
  elements", arXiv:2410.01493 (abstract and Table 3).  1 a.u. = a0^3 = 0.529177210903^3 A^3, so
  alpha = 5.4354 A^3.  (An atomic-polarizability paper.)
- One atom, open boundaries, neutral: one configuration in the formal window, so the model's alpha is the
  bound sector's exactly (trace_redox SPEC 17.7; tests/redox/test_redox_polarizability_prior.py).
- Measured before the run: the step-5 model gives 0.251 A^3 for this atom (22x too small), and Au2
  alpha_perp = 0.53 A^3.
"""
import numpy as np
from ase import Atoms
from ase.io import read, write

A0 = 0.529177210903                  # A (CODATA 2018)
ALPHA_AU = 36.68 * A0 ** 3           # A^3

if __name__ == "__main__":
    atoms = Atoms("Au", positions=[[0.0, 0.0, 0.0]])
    atoms.info["polarizability"] = (ALPHA_AU * np.eye(3)).tolist()
    atoms.info["charge"] = 0
    write("data/Au_atom_alpha_prior.xyz", [atoms], format="extxyz")
    back = read("data/Au_atom_alpha_prior.xyz", 0)
    label = back.calc.results["polarizability"] if back.calc is not None and "polarizability" in back.calc.results \
        else back.info["polarizability"]
    assert np.allclose(np.asarray(label).reshape(3, 3), ALPHA_AU * np.eye(3), rtol=0, atol=1e-12)
    print(f"wrote data/Au_atom_alpha_prior.xyz: Au, alpha = {ALPHA_AU:.4f} A^3 (36.68 a.u.), pbc {back.pbc.tolist()}")
