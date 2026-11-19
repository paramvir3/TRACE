# One of the measurement scripts of the theory note's section 4c (2026-10-29); run from the repo root.
"""Overlaps of a probe Au 6s with the ten-ion cluster's O 2p band orbitals at the standing-acceptor sites (vertical
O-top at 3.26 / 3.50 A, over Mg0 at 2.89 A, over the (+,+) hollow at 2.89 A), weighed by the measured levels:
sum_i S_ai^2 / (eps_a - eps_i) with eps_a the Au2 sigma* KS level (-4.297 eV, PBE/def2-TZVP, 2.545 A).  Overlaps
only (the Fock elements of the union-basis probe were invalid: 2026-10-29 11:25).  Pre-registered prediction for the
hollow_pp site run under the site-symmetry picture."""
import sys, os, numpy as np
sys.path.insert(0, "tools/electrostatics")
import transfer_overlap as TO, wetting_eda as W
from pyscf import gto, dft
out = os.path.join(os.environ.get("TRACE_FP_CACHE", "fp_cache"), "transfer_overlap")   # first-principles cache: $TRACE_FP_CACHE (default ./fp_cache)
mol_b, mf, i_o = TO.cluster_B(chk=os.path.join(out, "cluster_B.chk"))
G = W.geometries()
qsites = W.quantum_sites(G["flat"] + G["upright"], r_au=3.6)
surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in qsites]
C_p, eps, w_p, w_z = TO.o2p_band(mol_b, mf, i_o)
mol_au = gto.M(atom=[["Au", (0, 0, 0)]], basis={"Au": W.BASIS}, ecp={"Au": W.BASIS}, spin=1, verbose=0)
mfa = dft.UKS(mol_au); mfa.xc = "pbe"; mfa.conv_tol = 1e-9; mfa.kernel()
c6s = mfa.mo_coeff[0][:, mfa.mo_occ[0] > 0][:, -1]
O = TO.O_SITE; D = W.D; ea = -4.297
print("band eps (eV):", np.round(eps, 3).tolist())
sites = {"vertical 3.26": (O + [0, 0, 3.26], 1), "vertical 3.50": (O + [0, 0, 3.50], 1), "vertical 3.59": (O + [0, 0, 3.59], 1),
         "over Mg0 2.89": (np.array([0.0, 0.0, 2.89]), 2), "hollow_pp 2.89": (np.array([0.5 * D, 0.5 * D, 2.89]), 2)}
nb = mol_b.nao_nr(); res = {}
for name, (R, n) in sites.items():
    atoms = [[sym, tuple(p)] for sym, p in surf] + [["Au", tuple(R)]]
    union = gto.M(atom=atoms, basis={"Au": W.BASIS, "Mg": W.BASIS, "O": W.BASIS}, ecp={"Au": W.BASIS}, charge=mol_b.charge + 1, spin=0, unit="Angstrom", verbose=0)
    S = union.intor_symmetric("int1e_ovlp"); nu = union.nao_nr()
    a = np.zeros(nu); a[nb:] = c6s
    Ci = np.zeros((nu, C_p.shape[1])); Ci[:nb] = C_p
    Sai = a @ S @ Ci
    s2 = float(np.sum(Sai ** 2)); s2d = float(np.sum(Sai ** 2 / (ea - eps)))
    res[name] = (s2, s2d, n)
    print(f"{name:15s}: S_ai {np.round(Sai, 3).tolist()}  sum S^2 {s2:.5f}  sum S^2/Delta {s2d:.5f} /eV  contacts {n}", flush=True)
pc = {k: v[1] / v[2] for k, v in res.items()}
print(f"per contact, sum S^2/Delta: over Mg0 2.89 (r 3.59) / vertical 3.59 = {pc['over Mg0 2.89'] / pc['vertical 3.59']:.3f}")
print(f"per contact, sum S^2/Delta: hollow_pp 2.89 (r 3.26) / vertical 3.26 = {pc['hollow_pp 2.89'] / pc['vertical 3.26']:.3f}")
print(f"hollow_pp per contact / over-Mg0 per contact (raw, r 3.26 vs 3.59) = {pc['hollow_pp 2.89'] / pc['over Mg0 2.89']:.3f}; "
      f"vertical 3.26 / vertical 3.59 (the distance factor in this measure) = {pc['vertical 3.26'] / pc['vertical 3.59']:.3f}")
