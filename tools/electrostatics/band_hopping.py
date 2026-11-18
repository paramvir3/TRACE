"""The band unit's hopping from first principles: the Pauli-projected cation-cation coupling (2026-10-22).

    OMP_NUM_THREADS=4 nice python tools/electrostatics/band_hopping.py OUT_DIR
    (python: an interpreter with PySCF installed)

Why: the band unit (trace_redox SPEC 20) hops with Harrison's V_ssSigma = 1.40 hbar^2/(m d^2) times a learned factor
f (initial 1).  PBE's conduction-band mass of MgO at the 4G lattice constant, 0.322 / 0.330 m_e (mgo_pbe.py, basis
dzvp / tzv2p), fixes the nearest-neighbour s band's hopping at tau = hbar^2 / (2 m* a^2) = 0.651 / 0.635 eV, i.e.
f = 0.55, while Harrison gives 1.173 eV at d = 3.016 A.

Derivation (the Phillips-Kleinman projection, ESTABLISHED: Phillips & Kleinman, Phys. Rev. 116, 287 (1959); its use
for the band unit NEW-TO-TRACE).  The band electron lives in the frame's virtual space: its orbital on cation u is
Q phi_u, Q = 1 - P_occ.  With the occupied anion orbitals chi_k eigenfunctions of the frame operator F (eps_k), to
first order in S_uk = <phi_u|chi_k>,

    t_uv = [F_uv - eps S_uv]  +  sum_k (eps - eps_k) S_uk S_vk ,

the bare coupling plus a Pauli term that, for anion levels far below the carrier (eps - eps_k ~ 20-25 eV) and s
carriers, weakens the bonding coupling.  In rock salt the two second-neighbour cations share two anions at 90 deg,
so the anion-p part (~ S_sp^2 cos theta) vanishes and the anion 2s carries the narrowing.

Computed here exactly within the frozen-frame model (no perturbation theory): the carriers are the library's Mg+ 3s
orbitals (singly occupied orbital of the UKS PBE0 monomer, frozen_interaction._monomer) on two second-neighbour Mg
sites A = (0,0,0), B = (1,1,0) d; the frame is Mg2+ x 2 + the O2- around them, frozen library orbitals; F is the
PBE Kohn-Sham operator of the frozen frame density (the antisymmetrised product's), in the point-ion rock-salt
lattice of cluster_interaction (bulk, n = 4; d = 2.1325 A).  The two-orbital problem {Q phi_A, Q phi_B} with F gives
E+ and E-, and tau = (E+ - E-) / 2; without Q, the bare tau0.

Pre-registered:
  (h1) the projected tau with all ten O neighbours of A and B is within 15 % of the PBE-mass value (0.635-0.651 eV);
  (h2) locality: with only the two shared O quantum (the other eight as point ions) tau is within 10 % of (h1)'s;
  (h3) reported: the bare tau0 against Harrison's 1.173 eV, and both without the lattice.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cluster_interaction as CI  # noqa: E402
import frozen_interaction as FI  # noqa: E402

HARTREE = 27.211386245988
D = 2.1325
HBAR2_OVER_ME = 7.619964
TAU_HARRISON = 1.40 * HBAR2_OVER_ME / (np.sqrt(2) * D) ** 2
TAU_PBE = {m: HBAR2_OVER_ME / (2 * m * (2 * D) ** 2) for m in (0.322, 0.330)}


def _somo(symbol, q):
    """The singly occupied orbital of the library's monomer (alpha, highest occupied), own basis."""
    occ, spin, _ = FI._monomer(symbol, q, np.zeros(3), None, None)
    assert spin == 1 and occ[0].shape[1] == occ[1].shape[1] + 1
    return occ[0][:, -1]


def two_site(o_sites, lattice=True, grid_level=3):
    from pyscf import dft, qmmm
    A, B = (0, 0, 0), (1, 1, 0)
    ions = [("Mg", 2, A), ("Mg", 2, B)] + [("O", -2, s) for s in o_sites]
    mol = FI._mol([(sym, D * np.array(s, float)) for sym, _, s in ions], sum(q for _, q, _ in ions), 0)
    nao = mol.nao_nr()
    off = mol.aoslice_by_atom()[:, 2]
    S = mol.intor_symmetric("int1e_ovlp")
    cols = []
    for m, (sym, q, s) in enumerate(ions):
        occ, spin, _ = FI._monomer(sym, q, np.zeros(3), None, None)
        assert spin == 0
        c = np.zeros((nao, occ[0].shape[1]))
        c[off[m]:off[m] + occ[0].shape[0]] = occ[0]
        cols.append(c)
    C = np.hstack(cols)
    P = C @ np.linalg.solve(C.T @ S @ C, C.T)                       # projector onto the frame's occupied space
    mf = dft.RKS(mol).density_fit()
    mf.xc = "pbe"
    mf.grids.level = grid_level
    if lattice:
        lat = CI.rocksalt_lattice(D, 4, False, exclude=[A, B] + list(o_sites))
        mf = qmmm.mm_charge(mf, lat[0], lat[1], radii=np.full(len(lat[1]), 0.3), unit="Angstrom")
    F = mf.get_hcore() + mf.get_veff(mol, 2.0 * P)
    somo = _somo("Mg", 1)
    phi = np.zeros((nao, 2))
    phi[off[0]:off[0] + somo.size, 0] = somo
    phi[off[1]:off[1] + somo.size, 1] = somo
    out = {"nao": int(nao), "n_O": len(o_sites), "lattice": bool(lattice)}
    for label, vec in (("bare", phi), ("projected", phi - P @ S @ phi)):
        Sm, Hm = vec.T @ S @ vec, vec.T @ F @ vec
        w = np.linalg.eigvals(np.linalg.solve(Sm, Hm)).real
        w.sort()
        out[f"tau_{label}"] = 0.5 * (w[1] - w[0]) * HARTREE
        out[f"site_{label}"] = float(Hm[0, 0] / Sm[0, 0]) * HARTREE
        out[f"norm_{label}"] = float(Sm[0, 0])
        out[f"S_AB_{label}"] = float(Sm[0, 1] / Sm[0, 0])
    return out


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    A, B = np.array((0, 0, 0)), np.array((1, 1, 0))
    nn = [tuple(int(x) for x in v) for v in np.eye(3, dtype=int).tolist() + (-np.eye(3, dtype=int)).tolist()]
    o_all = sorted({tuple(int(x) for x in A + np.array(u)) for u in nn} | {tuple(int(x) for x in B + np.array(u))
                                                                            for u in nn})
    shared = [(1, 0, 0), (0, 1, 0)]
    assert len(o_all) == 10 and all(s in o_all for s in shared)
    res = {"tau_harrison": TAU_HARRISON, "tau_pbe_mass": TAU_PBE}
    path = os.path.join(out_dir, "band_hopping.json")
    for key, sites, lat in (("shared_lattice", shared, True), ("all_lattice", o_all, True),
                            ("shared_vacuum", shared, False), ("all_vacuum", o_all, False)):
        r = two_site(sites, lattice=lat)
        res[key] = r
        json.dump(res, open(path, "w"), indent=1)
        print(f"{key}: tau projected {r['tau_projected']:.4f} eV, bare {r['tau_bare']:.4f} eV (Harrison "
              f"{TAU_HARRISON:.4f}); norm of Q phi {r['norm_projected']:.4f}; site level {r['site_projected']:+.3f} eV",
              flush=True)
    t_all, t_sh = res["all_lattice"]["tau_projected"], res["shared_lattice"]["tau_projected"]
    lo, hi = min(TAU_PBE.values()), max(TAU_PBE.values())
    ok1 = 0.85 * lo <= t_all <= 1.15 * hi
    ok2 = abs(t_sh / t_all - 1) < 0.10
    print(f"(h1) projected tau, ten O: {t_all:.4f} eV against PBE-mass {lo:.3f}-{hi:.3f}: {'PASS' if ok1 else 'FAIL'}")
    print(f"(h2) locality, two shared O: {t_sh:.4f} eV ({(t_sh / t_all - 1) * 100:+.1f} %): {'PASS' if ok2 else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1])
