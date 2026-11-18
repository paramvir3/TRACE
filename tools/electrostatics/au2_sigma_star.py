# One of the measurement scripts of the theory note's section 4c (2026-10-29); run from the repo root.
"""Au2 (bond 2.545 A), PBE/def2-TZVP with the def2 ECP: the sigma* LUMO energy (the acceptor level of the fold's
denominators) and the HOMO, plus the single Au 6s for comparison with the table's -6.01 eV."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyscf import gto, dft
import numpy as np
import wetting_eda as W
HA = 27.211386245988
for name, atoms, spin in [("Au", [["Au", (0, 0, 0)]], 1), ("Au2", [["Au", (0, 0, 0)], ["Au", (0, 0, 2.545)]], 0)]:
    mol = gto.M(atom=atoms, basis=W.BASIS, ecp={"Au": W.BASIS}, spin=spin, unit="Angstrom", verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol); mf.xc = "pbe"; mf.conv_tol = 1e-8; e = mf.kernel()
    if spin:
        occ = mf.mo_occ[0]; eps = mf.mo_energy[0] * HA
        homo = eps[occ > 0].max(); lumo = eps[occ == 0].min()
        print(f"{name}: E {e*HA:.4f} eV; alpha HOMO (6s) {homo:.3f} eV, alpha LUMO {lumo:.3f} eV")
    else:
        occ = mf.mo_occ; eps = mf.mo_energy * HA
        homo = eps[occ > 0].max(); lumo = eps[occ == 0].min(); lumo2 = np.sort(eps[occ == 0])[1]
        print(f"{name}: E {e*HA:.4f} eV; HOMO (sigma) {homo:.3f} eV, LUMO (sigma*) {lumo:.3f} eV, next {lumo2:.3f} eV; converged {mf.converged}")
