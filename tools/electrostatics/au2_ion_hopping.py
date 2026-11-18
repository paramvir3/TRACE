"""SPEC 29's premise, measured (2026-11-01): one hop t for every electron count of the Au2 two-site model.
In the two-site Hubbard model (ESTABLISHED) the 1- and 3-electron dimers (Au2+, Au2-) have the sigma -> sigma*
excitation E = 2t exactly (the resonance doublets; U cancels), and the neutral gives t = sqrt(Delta_ST E_S) / 2
(au2_hubbard.py).  TDA-PBE (UKS, spin-conserving; the 4G data's functional), def2-TZVP + def2 ECP, vertical at R: the
root with the largest sigma_g -> sigma_u* weight in the spin channel that holds the move (cation: the alpha sigma_g
electron; anion: the beta sigma_g electron into the half-filled sigma_u*), its NTO share; t_N = E / 2.
Usage: python au2_ion_hopping.py R [R ...]"""
import sys
import numpy as np
from pyscf import gto, dft, tdscf
HARTREE = 27.211386245988


def atomic_6s():
    atom = gto.M(atom=[["Au", (0, 0, 0)]], basis="def2-tzvp", ecp="def2-tzvp", spin=1, verbose=0)
    ma = dft.UKS(atom); ma.xc = "pbe"; ma.conv_tol = 1e-10; ma.kernel()
    return ma.mo_coeff[0][:, ma.mo_occ[0] > 0][:, -1], atom.nao_nr()


C6S, N1 = atomic_6s()


def run(R, charge):
    mol = gto.M(atom=[["Au", (0, 0, 0)], ["Au", (0, 0, R)]], basis="def2-tzvp", ecp="def2-tzvp", unit="Angstrom",
                charge=charge, spin=1, verbose=0)
    ks = dft.UKS(mol); ks.xc = "pbe"; ks.conv_tol = 1e-10; ks.max_cycle = 200; ks.kernel()
    S = mol.intor_symmetric("int1e_ovlp")
    chiA = np.zeros(mol.nao_nr()); chiB = np.zeros(mol.nao_nr()); chiA[:N1] = C6S; chiB[N1:] = C6S
    g, u = chiA + chiB, chiA - chiB
    proj = []
    for s in (0, 1):
        occ = ks.mo_occ[s] > 0
        Co, Cv = ks.mo_coeff[s][:, occ], ks.mo_coeff[s][:, ~occ]
        proj.append((Co.T @ S @ g, Cv.T @ S @ u))
    td = tdscf.TDA(ks); td.nstates = 16; td.kernel()
    best = None
    for k, (e, xy) in enumerate(zip(td.e, td.xy)):
        xa, xb = xy[0]
        norm = float((xa ** 2).sum() + (xb ** 2).sum())
        w = 0.0
        for (o, v), x in zip(proj, (xa, xb)):
            w += float((o @ x @ v) ** 2 / ((o @ o) * (v @ v)))
        w /= max(norm, 1e-30)
        if best is None or w > best[1]:
            sv = np.concatenate([np.linalg.svd(xa, compute_uv=False), np.linalg.svd(xb, compute_uv=False)])
            best = (float(e) * HARTREE, w, k, float(sv.max() ** 2 / (sv ** 2).sum()))
    return best, float(ks.e_tot)


for R in map(float, sys.argv[1:]):
    for q, name in ((+1, "Au2+ (N = 1)"), (-1, "Au2- (N = 3)")):
        (E, w, k, share), etot = run(R, q)
        print(f"R {R:.2f} {name}: sigma -> sigma* root {k} {E:.3f} eV (sigma weight {w:.2f}, leading NTO share {share:.3f})"
              f"  ->  t = E / 2 = {E / 2:.3f} eV", flush=True)
