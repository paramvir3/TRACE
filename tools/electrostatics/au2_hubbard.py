"""The gold dimer's ionic-state anchor from first principles (2026-11-01): Au2's own two-site Hubbard parameters.

TRACE's explicit ensemble for a bonded pair of s-class atoms is an exact two-site Hubbard model (SPEC 29): the covalent
frame Au0Au0, the ionic Au+Au- and Au-Au+ at the diabatic gap U = Delta_ion, and the hop t (with the singlet's sqrt 2).
Its exact spectrum: ground singlet (U - W)/2, sigma sigma* triplet 0, ionic singlet (1Sigma_u+) U, upper singlet
(U + W)/2, W = sqrt(U^2 + 16 t^2).  So the two lowest sigma -> sigma* excitations of the gas-phase dimer, the triplet
Delta_ST = (W - U)/2 and the singlet E_S = (U + W)/2, fix the model's own U and t:  U = E_S - Delta_ST,
t = sqrt(Delta_ST E_S) / 2  (ESTABLISHED: the two-electron two-orbital Hubbard / valence-bond mapping, as for H2).
EOM-EE-CCSD (the reference) and TDA-PBE (the 4G data's functional), def2-TZVP with the def2 ECP; in each spin manifold
the root with the largest HOMO -> LUMO (6s sigma_g -> 6s sigma_u*) single-excitation weight is taken, the weight printed
(5d excitations are nearby).  Usage: python au2_hubbard.py R [R ...]"""
import sys
import numpy as np
from pyscf import gto, scf, dft, cc, tdscf, lo
HARTREE = 27.211386245988


def sigma_check(mol, mf):
    """HOMO / LUMO 6s weight (Mulliken, both atoms) and parity: sigma_g has equal-sign s coefficients."""
    labels = mol.ao_labels(fmt=False)
    S = mol.intor_symmetric("int1e_ovlp")
    out = []
    for idx in (mol.nelectron // 2 - 1, mol.nelectron // 2):
        c = mf.mo_coeff[:, idx]
        pop = c * (S @ c)
        s6 = [k for k, l in enumerate(labels) if l[2] == "6s"]
        w = float(pop[s6].sum())
        sa = sum(c[k] for k in s6 if labels[k][0] == 0); sb = sum(c[k] for k in s6 if labels[k][0] == 1)
        out.append((idx, w, "g" if sa * sb > 0 else "u"))
    return out


def sigma_vectors(mol, mf, R):
    """o (occupied MO components of sigma_g = chi_A + chi_B) and v (virtual MO components of sigma_u* = chi_A - chi_B),
    chi the Au atom's own 6s orbital (PBE, UKS doublet, alpha SOMO) placed on each centre.  Basis-independent
    identification (2026-11-01: the PBE KS HOMO of Au2 is a 5d orbital, and the RHF LUMO is 73 % 6p, so a
    HOMO -> LUMO weight picks the wrong excitation)."""
    atom = gto.M(atom=[["Au", (0, 0, 0)]], basis="def2-tzvp", ecp="def2-tzvp", spin=1, verbose=0)
    ma = dft.UKS(atom); ma.xc = "pbe"; ma.conv_tol = 1e-10; ma.kernel()
    c6s = ma.mo_coeff[0][:, ma.mo_occ[0] > 0][:, -1]
    n1 = atom.nao_nr()
    chiA = np.zeros(mol.nao_nr()); chiB = np.zeros(mol.nao_nr()); chiA[:n1] = c6s; chiB[n1:] = c6s
    S = mol.intor_symmetric("int1e_ovlp")
    nocc = mol.nelectron // 2
    Co, Cv = mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]
    return Co.T @ S @ (chiA + chiB), Cv.T @ S @ (chiA - chiB)


def pick(energies, r1s, o, v):
    def w(r):
        return float((o @ r @ v) ** 2 / ((o @ o) * (v @ v) * max(float((r ** 2).sum()), 1e-30)))
    ws = [w(r) for r in r1s]
    best = int(np.argmax(ws))
    return float(energies[best]) * HARTREE, ws[best], best, sorted(((round(float(e) * HARTREE, 3), round(x, 2)) for e, x in zip(energies, ws)))


def hubbard(dst, es):
    return es - dst, np.sqrt(dst * es) / 2.0


if __name__ == "__main__":     # importable (au2_hubbard_nto.py reuses sigma_vectors / pick)
    for R in (float(x) for x in sys.argv[1:]):
        mol = gto.M(atom=[["Au", (0, 0, 0)], ["Au", (0, 0, R)]], basis="def2-tzvp", ecp="def2-tzvp", unit="Angstrom", verbose=0)
        nocc = mol.nelectron // 2
        mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.kernel()
        sc = sigma_check(mol, mf)
        mycc = cc.CCSD(mf); mycc.conv_tol = 1e-8; mycc.kernel()
        es, cs = mycc.eomee_ccsd_singlet(nroots=10)
        et, ct = mycc.eomee_ccsd_triplet(nroots=10)
        from pyscf.cc import eom_rccsd
        r1s = [eom_rccsd.EOMEESinglet(mycc).vector_to_amplitudes(c)[0] for c in cs]
        r1t = [eom_rccsd.EOMEETriplet(mycc).vector_to_amplitudes(c)[0] for c in ct]
        o, v = sigma_vectors(mol, mf, R)
        ES, wS, iS, allS = pick(es, r1s, o, v); ET, wT, iT, allT = pick(et, r1t, o, v)
        U, t = hubbard(ET, ES)
        print(f"R {R:.2f} A", flush=True)
        print(f"  EOM-CCSD: sigma sigma* triplet {ET:.3f} eV (weight {wT:.2f}, root {iT}), singlet {ES:.3f} eV (weight {wS:.2f}, root {iS})"
              f"  ->  U = {U:+.3f} eV, t = {t:.3f} eV", flush=True)
        print(f"    roots (eV, sigma sigma* weight): singlets {allS}; triplets {allT}", flush=True)
        ks = dft.RKS(mol); ks.xc = "pbe"; ks.conv_tol = 1e-10; ks.kernel()
        sck = sigma_check(mol, ks)
        td_s = tdscf.TDA(ks); td_s.nstates = 12; td_s.singlet = True; td_s.kernel()
        td_t = tdscf.TDA(ks); td_t.nstates = 12; td_t.singlet = False; td_t.kernel()
        xs = [x[0] for x in td_s.xy]; xt = [x[0] for x in td_t.xy]
        o2, v2 = sigma_vectors(mol, ks, R)
        ES2, wS2, iS2, allS2 = pick(td_s.e, xs, o2, v2); ET2, wT2, iT2, allT2 = pick(td_t.e, xt, o2, v2)
        U2, t2 = hubbard(ET2, ES2)
        print(f"  TDA-PBE : sigma sigma* triplet {ET2:.3f} eV (weight {wT2:.2f}), singlet {ES2:.3f} eV (weight {wS2:.2f})  ->  "
              f"U = {U2:+.3f} eV, t = {t2:.3f} eV", flush=True)
        print(f"    roots (eV, sigma sigma* weight): singlets {allS2}; triplets {allT2}", flush=True)
