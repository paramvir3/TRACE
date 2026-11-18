"""Falsifying check on the minimal two-site reading of Au2 (2026-11-01).  The two-site Hubbard model with U, t from the
adiabatic sigma-sigma* excitations predicts the ground state's ionic weight p = r^2 / (1 + r^2), r = E_g / 2t, and a
bond-axis transition dipole to the ionic singlet |<0|z|S>| = sqrt(p) R (ionic configurations carry e R; ESTABLISHED
two-state algebra), hence a charge-transfer polarisability alpha_CT = 2 p R^2 / E_S (x K_E for A^3).  TDA-PBE (def2-TZVP
+ ECP) gives the picked singlet root's actual transition dipole, and the sum over the computed Sigma_u+ roots of
2 |mu_z|^2 / E.  Usage: python au2_transition_dipole.py R [R ...]"""
import sys
import numpy as np
from pyscf import gto, dft, tdscf
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from au2_hubbard import sigma_vectors, pick, HARTREE  # noqa: E402
BOHR = 0.529177210903
K_E = 14.399645
for R in map(float, sys.argv[1:]):
    mol = gto.M(atom=[["Au", (0, 0, 0)], ["Au", (0, 0, R)]], basis="def2-tzvp", ecp="def2-tzvp", unit="Angstrom", verbose=0)
    ks = dft.RKS(mol); ks.xc = "pbe"; ks.conv_tol = 1e-10; ks.kernel()
    o, v = sigma_vectors(mol, ks, R)
    out = {}
    for singlet in (True, False):
        td = tdscf.TDA(ks); td.nstates = 24; td.singlet = singlet; td.kernel()
        xs = [x[0] for x in td.xy]
        E, w, i, _ = pick(td.e, xs, o, v)
        out[singlet] = (E, w, i, td)
    ES, wS, iS, td = out[True]; ET = out[False][0]
    U, t = ES - ET, np.sqrt(ES * ET) / 2
    Eg = (U - np.sqrt(U * U + 16 * t * t)) / 2; r = Eg / (2 * t); p = r * r / (1 + r * r)
    mu = td.transition_dipole()                      # [nstates, 3] in e bohr (spin-adapted singlet)
    muz = abs(float(mu[iS, 2])) * BOHR                # e A
    pred = np.sqrt(p) * R
    a_pred = 2 * p * R * R / ES * K_E
    a_sos = sum(2 * (float(m[2]) * BOHR) ** 2 / (float(e) * HARTREE) * K_E for m, e in zip(mu, td.e))
    a_root = 2 * muz ** 2 / ES * K_E
    print(f"R {R:.2f}: U {U:.3f}, t {t:.3f} eV -> p_ion {p:.3f}; singlet root {iS} {ES:.3f} eV (6s sigma weight {wS:.2f}): "
          f"|mu_z| {muz:.3f} e A against the two-site sqrt(p) R {pred:.3f} (ratio {muz / pred:.2f}); "
          f"alpha_CT two-site {a_pred:.1f} A^3, this root {a_root:.1f}, all {len(td.e)} computed singlets (z) {a_sos:.1f}", flush=True)
