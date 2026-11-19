"""The 4G wetting energy of Au2 on MgO(100), decomposed from first principles (2026-10-22).

    OMP_NUM_THREADS=6 nice python tools/electrostatics/wetting_eda.py OUT_DIR
    (python: an interpreter with PySCF installed)

Why: step 17's T5 passes by cancellation: the flat (wetting) Au2 is too stable against the upright one by 240-450
meV in both doped and undoped cells (benchmarks README, step 17), and on the training set too; the exact term split
(probe_t5_terms.py) puts it in the redox module (the backbone's share is equal for doped and undoped), but cannot say
which term is wrong, since DFT is not decomposed.  This decomposes the DFT side.

Model (PBE, the data's functional; def2-TZVP, def2 ECP on Au): Au2 (A) over a quantum MgO(100) cluster (B) embedded
in the point-ion rock-salt slab of cluster_interaction (n = 5, six layers, Evjen in plane; the 4G lattice d =
2.1325 A).  One quantum cluster serves both geometries: the flat dimer over a nearest Mg pair M0 = (0,0,0),
M1 = (1,1,0) d, each Au 2.89 A up and 0.22 A inward (the 4G reference); the upright dimer on the O at (1,0,0) d,
Au at 2.26 and 4.805 A (the reference).  B holds the surface sites near either dimer and every cation adjacent to a
quantum anion, so no bare point cation touches a quantum O2- (the boundary artifact of point-ion embedding,
ESTABLISHED).  For each geometry:

    Delta E_SCF = E_AB - E_A - E_B           (counterpoise: monomers in the dimer basis; B in the point ions)
    E1          = E_AB[P_HL] - E_A - E_B      (first order: the antisymmetrised product of the SCF monomers)
    elst        = classical Coulomb of the two frozen monomers (with the point ions acting on A)
    nonelst     = E1 - elst                   (Pauli + the functional's xc of the overlap)
    relax       = Delta E_SCF - E1            (polarisation + charge transfer)

all on one grid (the dimer's), so no grid difference enters.  The wetting energy of the cluster model is
wet = [Delta E_SCF + E_Au2](flat) - [Delta E_SCF + E_Au2](upright), E_Au2 the isolated dimer at each geometry's
bond length (the surface is frozen and the same for both).

Pre-registered validation, before any piece is read: |wet - 929 meV| <= 150 meV (DFT, undoped, 4G references).  If it
fails, the cluster is not a model of the periodic DFT and the decomposition is not interpreted.  Convergence (added
before the first run): the quantum cluster within 3.6 A of the Au (10 ions) against 3.8 A (28 ions); the pieces are
read only if wet changes by < 100 meV between them.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cluster_interaction as CI  # noqa: E402

HARTREE = 27.211386245988
D = 2.1325
BASIS = "def2-tzvp"


def geometries():
    """Au positions (A) on the ideal surface (layer k = 0 at z = 0, vacuum above), from the 4G references."""
    u = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    M0, M1, O = np.zeros(3), D * np.array([1.0, 1.0, 0.0]), D * np.array([1.0, 0.0, 0.0])
    flat = [M0 + 0.218 * u + np.array([0, 0, 2.887]), M1 - 0.219 * u + np.array([0, 0, 2.893])]
    upright = [O + np.array([0, 0, 2.26]), O + np.array([0, 0, 4.805])]
    return {"flat": flat, "upright": upright}


def quantum_sites(au_all, r_au=3.8):
    """Surface-layer sites within r_au (3D) of any Au of either geometry, and every cation adjacent to a quantum
    anion (in plane and below), so no bare point cation touches a quantum O2- (integer lattice indices; cation =
    even index sum).  (A first version with every site within 3.3 A in plane plus the layer below gave 53 ions,
    charge +46 and 1761 AO: too costly for six SCFs; the overlap physics reaches the ions near the Au only.)"""
    sites = set()
    for p in au_all:
        for i in range(-4, 5):
            for j in range(-4, 5):
                if np.linalg.norm(np.array([i * D, j * D, 0.0]) - p) <= r_au:
                    sites.add((i, j, 0))
    for s in list(sites):
        if sum(s) % 2 != 0:                                       # an anion: add its six neighbours' cations
            for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1)):
                t = (s[0] + v[0], s[1] + v[1], s[2] + v[2])
                if t[2] <= 0:
                    sites.add(t)
    return sorted(sites)


def _mol(atoms, charge, ghosts=()):
    from pyscf import gto
    spec = [[sym, tuple(p)] for sym, p in atoms] + [["ghost-" + sym, tuple(p)] for sym, p in ghosts]
    basis = {sym: BASIS for sym, _ in list(atoms)} | {"ghost-" + sym: BASIS for sym, _ in ghosts}
    ecp = {"Au": BASIS} if any(sym == "Au" for sym, _ in atoms) else {}
    return gto.M(atom=spec, basis=basis, ecp=ecp, charge=charge, spin=0, unit="Angstrom", verbose=0)


def _rks(mol, grids, lattice=None):
    from pyscf import dft, qmmm
    mf = dft.RKS(mol).density_fit()
    mf.xc = "pbe"
    mf.conv_tol = 1e-9
    mf.max_cycle = 300
    mf.level_shift = 0.3
    mf.grids = grids
    if lattice is not None:
        mf = qmmm.mm_charge(mf, lattice[0], lattice[1], radii=np.full(len(lattice[1]), 0.3), unit="Angstrom")
    return mf


def _solve(mf):
    # one progress line per SCF cycle (added 2026-10-24: silent SCFs gave no basis for a finish-time estimate)
    import time
    t0 = time.time()

    def progress(envs):
        print(f"    scf cycle {envs.get('cycle')}: E {envs.get('e_tot'):.10f}, |dE| "
              f"{abs(envs.get('e_tot') - envs.get('last_hf_e', envs.get('e_tot'))):.2e}, {time.time() - t0:.0f} s",
              flush=True)

    mf.callback = progress
    e = mf.kernel()
    if not mf.converged:
        mf.level_shift = 0.0
        mf = mf.newton()
        e = mf.kernel(dm0=mf.make_rdm1())
    assert mf.converged, "SCF did not converge"
    return mf, float(e)


def run_geometry(name, au, sites, lattice):
    from pyscf import dft
    surf = [("Mg" if sum(s) % 2 == 0 else "O", D * np.array(s, float)) for s in sites]
    qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf)
    A_atoms = [("Au", p) for p in au]
    dimer = _mol(A_atoms + surf, qB)
    grids = dft.gen_grid.Grids(dimer)
    grids.level = 3
    grids.build()
    molA = _mol(A_atoms, 0, ghosts=surf)
    # the same AO order in all three: A's atoms first, then B's
    molB = _mol_order_B(surf, qB, A_atoms)
    mfA, eA = _solve(_rks(molA, grids))
    mfB, eB = _solve(_rks(molB, grids, lattice))
    mfAB, eAB = _solve(_rks(dimer, grids, lattice))
    S = dimer.intor_symmetric("int1e_ovlp")
    nA = molA.nao_nr() - _mol(surf, qB).nao_nr()
    CA = mfA.mo_coeff[:, mfA.mo_occ > 0]
    CB = mfB.mo_coeff[:, mfB.mo_occ > 0]
    C = np.hstack([CA, CB])
    P_hl = 2.0 * C @ np.linalg.solve(C.T @ S @ C, C.T)
    DA, DB = mfA.make_rdm1(), mfB.make_rdm1()
    E_hl = float(mfAB.energy_tot(dm=P_hl))
    hAB, hA, hB = mfAB.get_hcore(), mfA.get_hcore(), mfB.get_hcore()
    vjB = mfAB.get_j(dimer, DB)
    elst = (float(np.einsum("ij,ji->", DA, hAB - hA)) + float(np.einsum("ij,ji->", DB, hAB - hB))
            + float(np.einsum("ij,ji->", vjB, DA)) + mfAB.energy_nuc() - mfA.energy_nuc() - mfB.energy_nuc())
    out = {"n_quantum_surface": len(surf), "charge_B": qB, "nao": int(dimer.nao_nr()), "nA_ao": int(nA),
           "E_AB": eAB * HARTREE, "E_A": eA * HARTREE, "E_B": eB * HARTREE,
           "dE_scf": (eAB - eA - eB) * HARTREE, "E1": (E_hl - eA - eB) * HARTREE, "elst": elst * HARTREE,
           "number_check": float(np.einsum("ij,ji->", P_hl - DA - DB, S))}
    out["nonelst"] = out["E1"] - out["elst"]
    out["relax"] = out["dE_scf"] - out["E1"]
    # Mulliken charge of Au2 in the dimer (orientation only: a basis-dependent partition)
    pop = mfAB.mulliken_pop(verbose=0)[1]
    out["q_Au2_mulliken"] = float(sum(pop[:len(A_atoms)]))
    print(f"{name}: dE_scf {out['dE_scf']:+.4f} eV = E1 {out['E1']:+.4f} (elst {out['elst']:+.4f}, nonelst "
          f"{out['nonelst']:+.4f}) + relax {out['relax']:+.4f}; Au2 Mulliken {out['q_Au2_mulliken']:+.3f} e; "
          f"{len(surf)} quantum ions, {dimer.nao_nr()} AO; number check {out['number_check']:+.1e}", flush=True)
    return out


def _mol_order_B(surf, qB, A_atoms, tips=()):
    """B with the ghost A functions first, so the AO order matches the dimer (A then B, then any TIPs)."""
    from pyscf import gto
    spec = [["ghost-Au", tuple(p)] for _, p in A_atoms] + [[sym, tuple(p)] for sym, p in surf]
    basis = {"ghost-Au": BASIS} | {sym: BASIS for sym, _ in surf}
    if len(tips):
        spec += [["Mg1", tuple(t)] for t in tips]
        basis["Mg1"] = TIP_BASIS
        return gto.M(atom=spec, basis=basis, ecp={"Mg1": "stuttgart"}, charge=qB + 2 * len(tips), spin=0,
                     unit="Angstrom", verbose=0)
    return gto.M(atom=spec, basis=basis, charge=qB, spin=0, unit="Angstrom", verbose=0)


# ---- total-ion pseudopotentials on the boundary (added 2026-10-26) --------------------------------------------------
#
# With bare point cations the doped cell's extra electron spills into the quantum Mg (doped_spin.py: spin on the
# cluster's Mg +1.090, on Au2 -0.102; the bare-lattice UKS returns to that state from the capped solution's density,
# tip_checks.py c1).  Capping the point Mg2+ within 3.5 A of a quantum ion by Mg2+ total-ion pseudopotentials (the
# Stuttgart 10-electron-core ECP with no valence electrons; ESTABLISHED: Winter, Pitzer & Klein 1987; Sushko, Shluger &
# Catlow 2000; Nasluzov et al. 2001) puts it on Au2 (+0.830) and changes the undoped numbers by 2-4 meV (c2).  PySCF
# refuses an atom with no basis, so each TIP carries one tight s function no electron occupies (population 2e-12 e).

TIP_R = 3.5
TIP_BASIS = [[0, [1.0e4, 1.0]]]


def tip_capped(surf, lattice, r=TIP_R):
    """(TIP positions, the lattice without them): the point cations within r (A) of a quantum ion."""
    coords, q = lattice
    dist = np.linalg.norm(coords[:, None, :] - np.array([p for _, p in surf])[None], axis=2).min(1)
    j = np.nonzero((dist < r) & (q > 0))[0]
    assert np.allclose(q[j], 2.0), "only Mg2+ sites are capped"
    keep = np.ones(len(q), dtype=bool)
    keep[j] = False
    return [tuple(coords[i]) for i in j], (coords[keep], q[keep])


# ---- the doped cell (added 2026-10-23, before its run) ------------------------------------------------------------
#
# The 4G doped cells hold Al3+ on a Mg site four layers below the surface (k = -4), one per 3x3 cell, and the dopant's
# electron on Au2 (B1).  Here A = Au2- (doublet, UKS), and the point-ion lattice carries the Al3+ (+1 against Mg2+) at
# the 4G site relative to each geometry: flat (-2, 0, -4) d, upright (2, 0, -4) d (from the doped references).  One Al
# only: each 4G cell is neutral (one Al3+ per Au2-), so the dilute equivalent is the single pair, whose periodic images
# are neutral dipolar pairs; a first version that kept the six Al images inside the point-ion slab without their Au2-
# added +5 e of unbalanced charge (caught before any run).  Each geometry's interaction energy is taken in its own lattice,
# so the cluster's energy in the dopant's field cancels within each.  Pre-registered validation: |wet_d - (-2.7 meV)|
# <= 150 meV (DFT, doped references).
#
# Erratum (2026-10-27): the 4G doped cells hold THREE Al per cell (Mg51 Al3 O54 Au2, all in the k = -4 layer), not one,
# so their ground state has Au2- and two conduction-band electrons (B1).  This one-Al model is the sub-problem "Au2- over
# the insulating surface with one ionised donor"; it lacks the band electrons and their screening (theory note
# 2026-10-18 section 4c, erratum of 2026-10-27).

AL_SITE = {"flat": (-2, 0, -4), "upright": (2, 0, -4)}


def doped_lattice(sites, geom, n=5):
    coords, q = CI.rocksalt_lattice(D, n, True, exclude=sites)
    al = np.array(AL_SITE[geom], float)
    q = q.copy()
    j = np.nonzero(np.linalg.norm(coords - D * al, axis=1) < 1e-6)[0]
    assert len(j) == 1 and abs(q[j[0]] - 2.0) < 1e-12, "the Al site must be a full-weight cation of the point lattice"
    q[j[0]] = 3.0                                                       # Mg2+ -> Al3+
    return (coords, q), 1


def _uks(mol, grids, lattice=None):
    from pyscf import dft, qmmm
    mf = dft.UKS(mol).density_fit()
    mf.xc = "pbe"
    mf.conv_tol = 1e-9
    mf.max_cycle = 300
    mf.level_shift = 0.3
    mf.grids = grids
    if lattice is not None:
        mf = qmmm.mm_charge(mf, lattice[0], lattice[1], radii=np.full(len(lattice[1]), 0.3), unit="Angstrom")
    return mf


def run_geometry_doped(name, au, sites, lattice, tips=()):
    """As run_geometry with A = Au2- (doublet): UKS for A and the dimer, the antisymmetrised product per spin.

    tips (added 2026-10-26): TIP positions (tip_capped) held by B and the dimer after B's atoms; A's AOs are the
    dimer's first ones, and A's matrices are zero-padded over the TIP functions (exact: A has none).  Also reported
    from then on: the unpaired spin on Au2 in the dimer, and the pseudopotential share of elst (the Au ECP acting on
    B's density and the TIP ECPs on A's: elst here takes each ECP as part of its centre's external potential)."""
    from pyscf import dft, gto
    surf = [("Mg" if sum(s) % 2 == 0 else "O", D * np.array(s, float)) for s in sites]
    qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf)
    A_atoms = [("Au", p) for p in au]

    def mol(atoms, charge, spin, ghosts=(), with_tips=False):
        spec = [[sym, tuple(p)] for sym, p in atoms] + [["ghost-" + sym, tuple(p)] for sym, p in ghosts]
        basis = {sym: BASIS for sym, _ in list(atoms)} | {"ghost-" + sym: BASIS for sym, _ in ghosts}
        ecp = {"Au": BASIS} if any(sym == "Au" for sym, _ in atoms) else {}
        if with_tips and len(tips):
            spec += [["Mg1", tuple(t)] for t in tips]
            basis["Mg1"] = TIP_BASIS
            ecp["Mg1"] = "stuttgart"
            charge += 2 * len(tips)                     # each TIP nucleus is Z_eff = 2 and brings no electron
        return gto.M(atom=spec, basis=basis, ecp=ecp, charge=charge, spin=spin, unit="Angstrom", verbose=0)

    dimer = mol(A_atoms + surf, qB - 1, 1, with_tips=True)
    grids = dft.gen_grid.Grids(dimer)
    grids.level = 3
    grids.build()
    molA = mol(A_atoms, -1, 1, ghosts=surf)
    molB = _mol_order_B(surf, qB, A_atoms, tips)
    n, nA = dimer.nao_nr(), molA.nao_nr()
    assert molB.nao_nr() == n and nA == dimer.aoslice_by_atom()[len(A_atoms) + len(surf) - 1][3]
    rows = lambda m: np.pad(m, ((0, n - nA), (0, 0)))
    both = lambda m: np.pad(m, ((0, n - nA), (0, n - nA)))
    mfA, eA = _solve(_uks(molA, grids))
    mfB, eB = _solve(_rks(molB, grids, lattice))
    mfAB, eAB = _solve(_uks(dimer, grids, lattice))
    S = dimer.intor_symmetric("int1e_ovlp")
    CB = mfB.mo_coeff[:, mfB.mo_occ > 0]
    P = []
    for sp in (0, 1):
        CA = rows(mfA.mo_coeff[sp][:, mfA.mo_occ[sp] > 0])
        C = np.hstack([CA, CB])
        P.append(C @ np.linalg.solve(C.T @ S @ C, C.T))
    P_hl = np.array(P)
    DA = both(mfA.make_rdm1().sum(0))
    DB = mfB.make_rdm1()
    E_hl = float(mfAB.energy_tot(dm=P_hl))
    hAB, hA, hB = mfAB.get_hcore(), both(mfA.get_hcore()), mfB.get_hcore()
    vjB = mfAB.get_j(dimer, DB)
    vjB = vjB[0] if vjB.ndim == 3 else vjB
    elst = (float(np.einsum("ij,ji->", DA, hAB - hA)) + float(np.einsum("ij,ji->", DB, hAB - hB))
            + float(np.einsum("ij,ji->", vjB, DA)) + mfAB.energy_nuc() - mfA.energy_nuc() - mfB.energy_nuc())

    def ecp_of(atom_idx):
        x = dimer.copy()
        x._ecpbas = dimer._ecpbas[np.isin(dimer._ecpbas[:, 0], atom_idx)]
        return x.intor_symmetric("ECPscalar") if len(x._ecpbas) else np.zeros((n, n))

    au_idx = list(range(len(A_atoms)))
    tip_idx = list(range(len(A_atoms) + len(surf), dimer.natm))
    elst_ecp = float(np.einsum("ij,ji->", DB, ecp_of(au_idx))) + float(np.einsum("ij,ji->", DA, ecp_of(tip_idx)))
    sl = dimer.aoslice_by_atom()
    dm_ab = mfAB.make_rdm1()
    spin_m = dm_ab[0] - dm_ab[1]
    spin_au = sum(float(np.einsum("ij,ji->", spin_m[:, sl[a][2]:sl[a][3]], S[sl[a][2]:sl[a][3], :])) for a in au_idx)
    out = {"n_quantum_surface": len(surf), "n_tips": len(tips), "nao": int(n), "dE_scf": (eAB - eA - eB) * HARTREE,
           "E1": (E_hl - eA - eB) * HARTREE, "elst": elst * HARTREE, "elst_ecp": elst_ecp * HARTREE,
           "spin_Au2": spin_au, "number_check": float(np.einsum("ij,ji->", P_hl.sum(0) - DA - DB, S))}
    out["nonelst"] = out["E1"] - out["elst"]
    out["relax"] = out["dE_scf"] - out["E1"]
    pop = mfAB.mulliken_pop(verbose=0)[1]
    out["q_Au2_mulliken"] = float(sum(pop[:len(A_atoms)]))
    print(f"doped {name}: dE_scf {out['dE_scf']:+.4f} eV = E1 {out['E1']:+.4f} (elst {out['elst']:+.4f}, nonelst "
          f"{out['nonelst']:+.4f}) + relax {out['relax']:+.4f}; Au2 Mulliken {out['q_Au2_mulliken']:+.3f} e, spin "
          f"{spin_au:+.3f}; ECP share of elst {out['elst_ecp']:+.4f}; {len(tips)} TIPs; number check "
          f"{out['number_check']:+.1e}", flush=True)
    return out


def au2m_vacuum(r):
    from pyscf import dft, gto
    m = gto.M(atom=[["Au", (0, 0, 0)], ["Au", (0, 0, r)]], basis={"Au": BASIS}, ecp={"Au": BASIS}, charge=-1, spin=1,
              unit="Angstrom", verbose=0)
    mf = dft.UKS(m).density_fit()
    mf.xc = "pbe"
    mf.conv_tol = 1e-10
    return _solve(mf)[1] * HARTREE


def main_doped(out_dir, r_au=3.8, tip=False):
    """tip (added 2026-10-26): cap the boundary with TIPs (tip_capped).  The decomposition is read only if the spin on
    Au2 exceeds 0.8 in both geometries (the 4G cells keep the electron on Au2, B1)."""
    os.makedirs(out_dir, exist_ok=True)
    G = geometries()
    sites = quantum_sites(G["flat"] + G["upright"], r_au=r_au)
    surf = [("Mg" if sum(s) % 2 == 0 else "O", D * np.array(s, float)) for s in sites]
    res = {"sites": sites}
    path = os.path.join(out_dir, f"wetting_eda_doped{'_tip' if tip else ''}_r{r_au:.1f}.json")
    for name in ("upright", "flat"):
        lat, hits = doped_lattice(sites, name)
        tips = ()
        if tip:
            tips, lat = tip_capped(surf, lat)
        print(f"doped {name}: one Al3+ at {AL_SITE[name]} d in the point-ion lattice; {len(tips)} TIPs", flush=True)
        res[name] = run_geometry_doped(name, G[name], sites, lat, tips)
        json.dump(res, open(path, "w"), indent=1)
    res["E_Au2m_flat"] = au2m_vacuum(float(np.linalg.norm(G["flat"][0] - G["flat"][1])))
    res["E_Au2m_upright"] = au2m_vacuum(float(np.linalg.norm(G["upright"][0] - G["upright"][1])))
    json.dump(res, open(path, "w"), indent=1)
    f, u = res["flat"], res["upright"]
    wet = (f["dE_scf"] + res["E_Au2m_flat"]) - (u["dE_scf"] + res["E_Au2m_upright"])
    ok = abs(wet * 1000 + 2.7) <= 150.0
    print(f"wet_d (cluster, PBE) {wet * 1000:+.1f} meV against DFT -2.7: {'PASS' if ok else 'FAIL'} (150 meV)", flush=True)
    for k in ("E1", "elst", "nonelst", "relax"):
        print(f"  wet_d share {k:8s} {(f[k] - u[k]) * 1000:+8.1f} meV", flush=True)
    print(f"  wet_d share Au2- deformation {(res['E_Au2m_flat'] - res['E_Au2m_upright']) * 1000:+8.1f} meV", flush=True)
    if "spin_Au2" in f:
        print(f"  wet_d share of elst from the pseudopotentials {(f['elst_ecp'] - u['elst_ecp']) * 1000:+8.1f} meV",
              flush=True)
        gate = all(res[k]["spin_Au2"] > 0.8 for k in ("flat", "upright"))
        print(f"  spin on Au2: flat {f['spin_Au2']:+.3f}, upright {u['spin_Au2']:+.3f} -> "
              f"{'the electron is on Au2 in both' if gate else 'NOT on Au2 in both: the decomposition is not read'}",
              flush=True)


def au2_vacuum(r):
    from pyscf import dft
    mol = _mol([("Au", np.zeros(3)), ("Au", np.array([0.0, 0.0, r]))], 0)
    mf = dft.RKS(mol).density_fit()
    mf.xc = "pbe"
    mf.conv_tol = 1e-10
    return _solve(mf)[1] * HARTREE


def main(out_dir, r_au=3.8):
    os.makedirs(out_dir, exist_ok=True)
    G = geometries()
    sites = quantum_sites(G["flat"] + G["upright"], r_au=r_au)
    lattice = CI.rocksalt_lattice(D, 5, True, exclude=sites)
    n_mg = sum(1 for s in sites if sum(s) % 2 == 0)
    print(f"quantum surface cluster: {len(sites)} ions ({n_mg} Mg, {len(sites) - n_mg} O); point ions {len(lattice[1])}",
          flush=True)
    res = {"sites": sites}
    path = os.path.join(out_dir, f"wetting_eda_r{r_au:.1f}.json")
    for name in ("upright", "flat"):
        res[name] = run_geometry(name, G[name], sites, lattice)
        json.dump(res, open(path, "w"), indent=1)
    res["E_Au2_flat"] = au2_vacuum(float(np.linalg.norm(G["flat"][0] - G["flat"][1])))
    res["E_Au2_upright"] = au2_vacuum(float(np.linalg.norm(G["upright"][0] - G["upright"][1])))
    json.dump(res, open(path, "w"), indent=1)
    f, u = res["flat"], res["upright"]
    wet = (f["dE_scf"] + res["E_Au2_flat"]) - (u["dE_scf"] + res["E_Au2_upright"])
    ok = abs(wet * 1000 - 929.0) <= 150.0
    print(f"wet (cluster, PBE) {wet * 1000:+.1f} meV against DFT +929: {'PASS' if ok else 'FAIL'} (150 meV)", flush=True)
    for k in ("E1", "elst", "nonelst", "relax"):
        print(f"  wet share {k:8s} {(f[k] - u[k]) * 1000:+8.1f} meV", flush=True)
    print(f"  wet share Au2 deformation {(res['E_Au2_flat'] - res['E_Au2_upright']) * 1000:+8.1f} meV", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--doped", "--doped-tip"):
        main_doped(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 3.8, tip=sys.argv[1] == "--doped-tip")
    else:
        main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 3.8)
