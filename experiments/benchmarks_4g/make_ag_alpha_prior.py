# Writes data_pfn/Ag_atom_alpha_prior.xyz: the polarizability prior structure for the Ag3 step-9 run (2026-10-16).
# Usage (this directory): python make_ag_alpha_prior.py
"""An isolated Ag atom with its reference static dipole polarizability, for ``electrostatics.polarizability_prior``
(the Ag counterpart of make_au_alpha_prior.py).

- Label: alpha(Ag) = 50.97 +- 1.93 a.u. (isotropic), relativistic coupled cluster with perturbative triples:
  Y. Cheng, arXiv:2410.01493 (abstract), the same source as the Au value.  1 a.u. = a0^3, so alpha = 7.5531 A^3.
  (An atomic-polarizability paper.)
- One atom, open boundaries, neutral: one configuration in the formal window (Ag's frame is neutral), so the model's
  alpha is the bound sector's exactly (trace_redox SPEC 17.7).
"""
import numpy as np
from ase import Atoms
from ase.io import read, write

A0 = 0.529177210903                  # A (CODATA 2018)
ALPHA_AG = 50.97 * A0 ** 3           # A^3

if __name__ == "__main__":
    atoms = Atoms("Ag", positions=[[0.0, 0.0, 0.0]])
    atoms.info["polarizability"] = (ALPHA_AG * np.eye(3)).tolist()
    atoms.info["charge"] = 0
    write("data_pfn/Ag_atom_alpha_prior.xyz", [atoms], format="extxyz")
    back = read("data_pfn/Ag_atom_alpha_prior.xyz", 0)
    label = back.calc.results["polarizability"] if back.calc is not None and "polarizability" in back.calc.results \
        else back.info["polarizability"]
    assert np.allclose(np.asarray(label).reshape(3, 3), ALPHA_AG * np.eye(3), rtol=0, atol=1e-12)
    print("alpha(Ag) =", ALPHA_AG, "A^3")
