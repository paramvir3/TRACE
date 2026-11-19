"""MgO's conduction band in PBE: effective mass and electron affinity, to test the band unit's prior (2026-10-20).

    OMP_NUM_THREADS=4 nice python tools/electrostatics/mgo_pbe.py bulk OUT_DIR
    OMP_NUM_THREADS=4 nice python tools/electrostatics/mgo_pbe.py slab OUT_DIR
    (python: an interpreter with PySCF installed)

Why: on the step 15 dump the band electron (SPEC 20) sits at lambda_1 = -6.60 eV below a vacuum electron, of which
-10.80 eV is hopping with Harrison's V_ssSigma at factor 1 (probe_dump_terms.py).  Two PBE numbers decide whether
that is physical:
  m*   the conduction-band effective mass of bulk MgO.  A nearest-neighbour s band on the fcc cation sublattice has
       E(k) = eps - 12 tau + tau a^2 k^2 near Gamma, so tau = hbar^2 / (2 m* a^2) (exact for that model);
  EA   the electron affinity of the 4G slab (MgO(100), six layers): V_vac - E_CBM, which lambda_1 approximates
       (a delocalised extra electron: E(N+1) - E(N) -> the KS CBM).
Method: periodic PBE (PySCF), GTH-PBE pseudopotentials, GTH basis, Gamma-centred k meshes; the geometry of the 4G
AuMgO data (in-plane cell 9.0474 / 3 = 3.01580 A, so a = 4.26495 A; c = 26.4589 A; ideal layers).
The vacuum level is the planar average of the local electrostatic potential (Hartree + local pseudopotential, built
exactly as pyscf.pbc.df.fft.get_pp and get_j build them; checked against mf.get_j) in the middle of the vacuum,
on the eigenvalues' own reference.

Pre-registered checks: (b1) the SCF and the band energies converged in the k mesh (6^3 -> 8^3: gap and m* within
0.02 eV and 2 %); (b2) basis: gth-dzvp-molopt-sr against gth-tzv2p, gap within 0.1 eV, m* within 5 %; (s1) the
Hartree potential reproduces mf.get_j to 1e-7 hartree; (s2) the vacuum plateau is flat to 10 meV over 4 A.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HARTREE = 27.211386245988
BOHR = 0.529177210903
HBAR2_2ME = 3.80998212                       # eV A^2
A_4G = 9.0474 / 3.0 * np.sqrt(2.0)            # 4.26495 A
C_4G = 26.4589


#: plane-wave cutoff of the FFT grid (hartree).  PySCF's default (precision 1e-8) gave a 137^3-143^3 mesh for the
#: two-atom cell, ~100x the grid a GTH pseudopotential needs; the cutoff is set explicitly and its convergence is a
#: pre-registered check (c1: 120 -> 160 Ha, gap within 0.01 eV and m* within 1 %).  Env MGO_KE overrides.
KE_CUTOFF = float(os.environ.get("MGO_KE", "120"))


def bulk_cell(a, basis, ke_cutoff=None):
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.a = [[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]]
    cell.atom = [["Mg", (0.0, 0.0, 0.0)], ["O", (a / 2, 0.0, 0.0)]]
    cell.basis, cell.pseudo, cell.unit, cell.verbose = basis, "gth-pbe", "Angstrom", 0
    cell.ke_cutoff = KE_CUTOFF if ke_cutoff is None else ke_cutoff
    cell.build()
    return cell


def scf(cell, kmesh, chk=None):
    """KRKS PBE; with ``chk`` the run checkpoints there and restarts from it (an SCF of the slab costs ~1.5 h; on
    2026-10-20 one was lost to a crash after convergence)."""
    from pyscf.pbc import dft
    kpts = cell.make_kpts(kmesh)
    mf = dft.KRKS(cell, kpts)
    mf.xc = "pbe"
    mf.conv_tol = 1e-10
    if chk is not None:
        if os.path.exists(chk):
            mf.init_guess = "chk"
        mf.chkfile = chk
    e = mf.kernel()
    assert mf.converged
    return mf, e


def band_edges(mf, cell, a):
    """Gap at Gamma and m* (isotropy check over [100], [110], [111]) from bands at small k."""
    n_occ = cell.nelectron // 2
    deltas = np.array([0.0, 0.015, 0.03, 0.045, 0.06])
    dirs = {"100": np.array([1.0, 0, 0]), "110": np.array([1.0, 1, 0]) / np.sqrt(2), "111": np.ones(3) / np.sqrt(3)}
    kband, labels = [], []
    for name, u in dirs.items():
        for dlt in deltas:
            kband.append(dlt * (2 * np.pi / (a / BOHR)) * u)              # bohr^-1
            labels.append((name, dlt))
    e_k, _ = mf.get_bands(np.array(kband))
    e_k = np.array(e_k) * HARTREE
    out = {"gap_Gamma": float(e_k[0][n_occ] - e_k[0][n_occ - 1]), "CBM_Gamma": float(e_k[0][n_occ]),
           "VBM_Gamma": float(e_k[0][n_occ - 1])}
    for name in dirs:
        idx = [i for i, (n, _) in enumerate(labels) if n == name]
        k_A = np.array([np.linalg.norm(kband[i]) / BOHR for i in idx])  # A^-1
        E = np.array([e_k[i][n_occ] for i in idx]) - e_k[idx[0]][n_occ]
        A = np.vstack([k_A ** 2, k_A ** 4]).T
        coef, *_ = np.linalg.lstsq(A[1:], E[1:], rcond=None)
        out[f"m_star_{name}"] = float(HBAR2_2ME / coef[0])
    return out


def run_bulk(out_dir):
    res = {}
    for basis in ("gth-dzvp-molopt-sr", "gth-tzv2p"):
        for kmesh in ((6, 6, 6), (8, 8, 8)):
            cell = bulk_cell(A_4G, basis)
            mf, e = scf(cell, kmesh)
            r = band_edges(mf, cell, A_4G)
            r["E_tot_eV"] = e * HARTREE
            res[f"{basis}_{kmesh[0]}"] = r
            print(f"bulk a={A_4G:.5f} {basis} k{kmesh[0]}: gap(Gamma) {r['gap_Gamma']:.4f} eV, m* "
                  f"{r['m_star_100']:.4f} / {r['m_star_110']:.4f} / {r['m_star_111']:.4f}", flush=True)
            json.dump(res, open(os.path.join(out_dir, "mgo_bulk.json"), "w"), indent=1)
    # lattice constant of this method (orientation: the 4G data's a is FHI-aims PBE)
    es = []
    a_s = A_4G * np.array([0.98, 0.99, 1.0, 1.01, 1.02])
    for a in a_s:
        mf, e = scf(bulk_cell(a, "gth-tzv2p"), (6, 6, 6))
        es.append(e * HARTREE)
    p = np.polyfit(a_s, es, 3)
    roots = np.roots(np.polyder(p))
    a0 = float([r.real for r in roots if abs(r.imag) < 1e-9 and a_s[0] < r.real < a_s[-1]][0])
    res["a0_tzv2p"] = a0
    print(f"PBE a0 (gth-tzv2p, k6): {a0:.4f} A against the 4G cell's {A_4G:.4f} A", flush=True)
    for key in ("gth-dzvp-molopt-sr", "gth-tzv2p"):
        r6, r8 = res[f"{key}_6"], res[f"{key}_8"]
        ok = abs(r6["gap_Gamma"] - r8["gap_Gamma"]) < 0.02 and abs(r6["m_star_100"] / r8["m_star_100"] - 1) < 0.02
        print(f"(b1) k convergence {key}: {'PASS' if ok else 'FAIL'}", flush=True)
    d, t = res["gth-dzvp-molopt-sr_8"], res["gth-tzv2p_8"]
    ok = abs(d["gap_Gamma"] - t["gap_Gamma"]) < 0.1 and abs(d["m_star_100"] / t["m_star_100"] - 1) < 0.05
    print(f"(b2) basis: {'PASS' if ok else 'FAIL'} (gap {d['gap_Gamma']:.3f} vs {t['gap_Gamma']:.3f}, m* "
          f"{d['m_star_100']:.3f} vs {t['m_star_100']:.3f})", flush=True)
    json.dump(res, open(os.path.join(out_dir, "mgo_bulk.json"), "w"), indent=1)


def slab_cell(a, c, n_layers, basis):
    from pyscf.pbc import gto
    s = a / np.sqrt(2.0)
    atoms = []
    z0 = 0.5 * c - 0.5 * (n_layers - 1) * a / 2
    for k in range(n_layers):
        z = z0 + k * a / 2
        mg, o = ((0.0, 0.0), (s / 2, s / 2)) if k % 2 == 0 else ((s / 2, s / 2), (0.0, 0.0))
        atoms += [["Mg", (mg[0], mg[1], z)], ["O", (o[0], o[1], z)]]
    cell = gto.Cell()
    cell.a = [[s, 0, 0], [0, s, 0], [0, 0, c]]
    cell.atom = atoms
    cell.basis, cell.pseudo, cell.unit, cell.verbose = basis, "gth-pbe", "Angstrom", 0
    cell.ke_cutoff = KE_CUTOFF
    cell.build()
    return cell


def local_potential(mf, cell):
    """Planar average along z (eV) of the electron's local electrostatic potential energy (Hartree + local PP), on
    the Hamiltonian's own reference, and the check of the Hartree part against mf.get_j at the first k point."""
    from pyscf.pbc import tools
    from pyscf.pbc.gto import pseudo
    df = mf.with_df
    mesh = df.mesh
    Gv = cell.get_Gv(mesh)
    SI = cell.get_SI(mesh=mesh)
    vpplocG = -np.einsum("ij,ij->j", SI, pseudo.get_vlocG(cell, Gv))
    # get_pp uses ifft(vpplocG) in matrix elements without the grid weight, so that array is w V_loc (w = vol / N;
    # get_vlocG carries no 1/vol).  Before 2026-10-23 it was added to the true Hartree potential unscaled: the slab's
    # vacuum potential varied by 134.7 eV (check s2 caught it).  The true local potential is ifft(vpplocG) / w.
    w = cell.vol / int(np.prod(mesh))
    vloc = tools.ifft(vpplocG, mesh).real / w
    dm = mf.make_rdm1()
    grids = df.grids
    rhoR = mf._numint.get_rho(cell, dm, grids, mf.kpts)
    ngrids = int(np.prod(mesh))
    coulG = tools.get_coulG(cell, mesh=mesh)
    rhoG = tools.fft(rhoR, mesh)
    vH = tools.ifft(coulG * rhoG, mesh).real
    # check: J at the first k point from this vH against mf.get_j (the matrix elements weight = vol / ngrids)
    # KNumInt.eval_ao returns one array per k point (the 2026-10-20 crash: it was used as an array)
    ao = mf._numint.eval_ao(cell, grids.coords, kpts=mf.kpts[:1])[0]
    J_mine = (ao.conj().T * (vH * w)) @ ao
    J_ref = mf.get_j(cell, dm, kpts=mf.kpts)[0]
    err = float(np.abs(J_mine - J_ref).max())
    # (s3) consistency of the local pseudopotential's scale: its matrix elements against get_pp minus the analytic
    # non-local part.  get_pp evaluates the non-local projectors on the FFT grid, so the two differ by the grid's
    # convergence (measured 0.36 % at 80 Ha on bulk); criterion 1 % of the matrix scale (the unscaled array is ~99 %
    # off).  The physical test is (s2): the vacuum potential of the neutral symmetric slab is flat.
    from pyscf.pbc.gto.pseudo import pp_int
    V_mine = (ao.conj().T * (vloc * w)) @ ao
    V_ref = df.get_pp(kpts=mf.kpts[:1])[0] - pp_int.get_pp_nl(cell, mf.kpts[:1])[0]
    rel_loc = float(np.abs(V_mine - V_ref).max() / np.abs(V_ref).max())
    v = (vH + vloc).reshape(mesh)
    z = np.arange(mesh[2]) * cell.lattice_vectors()[2, 2] / mesh[2] * BOHR
    return z, v.mean(axis=(0, 1)) * HARTREE, err, rel_loc


def run_slab(out_dir, n_layers=6, basis="gth-dzvp-molopt-sr", kmesh=(6, 6, 1)):
    cell = slab_cell(A_4G, C_4G, n_layers, basis)
    mf, e = scf(cell, kmesh, chk=os.path.join(out_dir, f"mgo_slab_{n_layers}_{basis}.chk"))
    n_occ = cell.nelectron // 2
    k0 = int(np.argmin(np.linalg.norm(mf.kpts, axis=1)))
    eps = np.array(mf.mo_energy[k0]) * HARTREE
    z, vz, err, rel = local_potential(mf, cell)
    vac = np.abs(z - 0.5 * C_4G) > 0.5 * C_4G - 2.0                 # within 2 A of the vacuum middle (z = 0 = c)
    v_vac = float(vz[vac].mean())
    flat = float(np.ptp(vz[vac]))
    out = {"n_layers": n_layers, "basis": basis, "kmesh": kmesh, "E_tot_eV": e * HARTREE,
           "CBM_Gamma": float(eps[n_occ]), "VBM_Gamma": float(eps[n_occ - 1]), "V_vac": v_vac, "plateau_ptp": flat,
           "J_check_max_abs": err, "local_pp_rel": rel, "EA": v_vac - float(eps[n_occ]), "IP_Gamma": v_vac - float(eps[n_occ - 1]),
           "z": z.tolist(), "v_planar": vz.tolist()}
    print(f"slab {n_layers} layers {basis} k{kmesh[0]}: CBM(Gamma) {eps[n_occ]:+.4f}, VBM {eps[n_occ - 1]:+.4f}, "
          f"V_vac {v_vac:+.4f} eV -> EA {out['EA']:+.4f} eV, IP(Gamma) {out['IP_Gamma']:+.4f} eV; "
          f"(s1) J check {err:.1e} Ha {'PASS' if err < 1e-7 else 'FAIL'}; (s3) local PP {rel:.2%} "
          f"{'PASS' if rel < 0.01 else 'FAIL'}; (s2) plateau ptp {flat * 1000:.1f} meV "
          f"{'PASS' if flat < 0.01 else 'FAIL'}", flush=True)
    json.dump(out, open(os.path.join(out_dir, f"mgo_slab_{n_layers}_{basis}.json"), "w"), indent=1)
    return out


def run_bz(out_dir, basis="gth-dzvp-molopt-sr", k=4):
    """Band energies at the fcc high-symmetry points (2026-10-22): does one nearest-neighbour s hopping describe the
    conduction band beyond its curvature at Gamma?  The fcc nn s band E = e - 4 tau sum_pairs cos cos predicts
    E(L) - E(G) = 12 tau, E(X) - E(G) = E(W) - E(G) = 16 tau, E(K) - E(G) = 15.66 tau.  Also the O 2p valence band's
    width (the hole band the model lacks).  Reported, not scored."""
    cell = bulk_cell(A_4G, basis)
    mf, e = scf(cell, (k, k, k))
    b = cell.reciprocal_vectors()                                   # rows, bohr^-1
    frac = {"G": (0, 0, 0), "X": (0.5, 0, 0.5), "L": (0.5, 0.5, 0.5), "W": (0.5, 0.25, 0.75),
            "K": (0.375, 0.375, 0.75)}
    names = list(frac)
    kpts = np.array([np.dot(frac[n], b) for n in names])
    e_k, _ = mf.get_bands(kpts)
    e_k = np.array(e_k) * HARTREE
    n_occ = cell.nelectron // 2
    out = {n: {"CB1": float(e_k[i][n_occ]), "CB2": float(e_k[i][n_occ + 1]), "VB_top": float(e_k[i][n_occ - 1]),
               "VB_bottom_2p": float(e_k[i][n_occ - 3])} for i, n in enumerate(names)}
    G = out["G"]["CB1"]
    for n in names:
        print(f"bz {n}: CB1 - CB1(G) {out[n]['CB1'] - G:+.3f} eV, CB2 - CB1(G) {out[n]['CB2'] - G:+.3f}, VB top "
              f"{out[n]['VB_top'] - out['G']['VB_top']:+.3f} (rel. VBM(G)), VB 2p bottom {out[n]['VB_bottom_2p']:+.3f}",
              flush=True)
    vb = [out[n]["VB_top"] for n in names] + [out[n]["VB_bottom_2p"] for n in names]
    print(f"bz: O 2p valence band width (over these points) {max(vb) - min(vb):.3f} eV", flush=True)
    json.dump(out, open(os.path.join(out_dir, "mgo_bz.json"), "w"), indent=1)


def run_band_average(out_dir, basis="gth-dzvp-molopt-sr", k=6):
    """Band averages over the SCF k mesh (2026-10-23): the localised-versus-delocalised energy of a carrier in a single
    isolated band is the band's mean (its Wannier on-site energy) minus its edge.  For the O 2p valence band (three
    bands) that is E_top - <E>_2p, the hole's delocalisation; for the lowest conduction band <E>_CB1 - E_bottom.
    Reported, not scored (the lowest conduction band is entangled with higher ones near the zone boundary)."""
    cell = bulk_cell(A_4G, basis)
    mf, e = scf(cell, (k, k, k))
    n_occ = cell.nelectron // 2
    E = np.array(mf.mo_energy) * HARTREE                              # [nk, nmo]
    vb = E[:, n_occ - 3:n_occ]
    cb = E[:, n_occ]
    out = {"VB_top": float(vb.max()), "VB_mean_2p": float(vb.mean()), "VB_bottom_2p": float(vb.min()),
           "CB_bottom": float(cb.min()), "CB1_mean": float(cb.mean()), "nk": int(E.shape[0])}
    out["hole_delocalisation"] = out["VB_top"] - out["VB_mean_2p"]
    out["electron_delocalisation"] = out["CB1_mean"] - out["CB_bottom"]
    print(f"band averages ({k}^3 = {E.shape[0]} k): O 2p width {out['VB_top'] - out['VB_bottom_2p']:.3f} eV, top - mean "
          f"{out['hole_delocalisation']:.3f} eV; CB1 mean - bottom {out['electron_delocalisation']:.3f} eV "
          f"(nearest-neighbour s band: 12 tau)", flush=True)
    json.dump(out, open(os.path.join(out_dir, "mgo_band_average.json"), "w"), indent=1)


def run_cutoff_check(out_dir, cutoffs=(80.0, 120.0, 160.0)):
    """(c1) the FFT cutoff: gap and m* at 80, 120, 160 Ha (gth-dzvp-molopt-sr, 4^3).  Measured: 120 -> 160 moved the
    gap 0.016 eV (FAIL at 0.01) and m* 0.6 % (pass); the increments fall ~8x per step, so 160 Ha is adopted and
    checked against 200 Ha (c1', pre-registered: within 0.01 eV and 1 %)."""
    path = os.path.join(out_dir, "mgo_cutoff.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for ke in cutoffs:
        if str(ke) in res:
            continue
        cell = bulk_cell(A_4G, "gth-dzvp-molopt-sr", ke_cutoff=ke)
        mf, e = scf(cell, (4, 4, 4))
        r = band_edges(mf, cell, A_4G)
        r["mesh"] = [int(x) for x in cell.mesh]
        res[str(ke)] = r
        print(f"cutoff {ke:.0f} Ha (mesh {cell.mesh}): gap(Gamma) {r['gap_Gamma']:.4f} eV, m* {r['m_star_100']:.4f} / "
              f"{r['m_star_110']:.4f} / {r['m_star_111']:.4f}", flush=True)
        json.dump(res, open(path, "w"), indent=1)
    for lo, hi in (("120.0", "160.0"), ("160.0", "200.0")):
        if lo in res and hi in res:
            a, b = res[lo], res[hi]
            ok = abs(a["gap_Gamma"] - b["gap_Gamma"]) < 0.01 and abs(a["m_star_100"] / b["m_star_100"] - 1) < 0.01
            print(f"(c1) cutoff {lo} -> {hi} Ha: {'PASS' if ok else 'FAIL'} (gap {a['gap_Gamma'] - b['gap_Gamma']:+.4f} eV,"
                  f" m* {a['m_star_100'] / b['m_star_100'] - 1:+.4%})", flush=True)


def run_bulk_reduced(out_dir):
    """Amended 2026-10-20 16:40, before any result: under the machine's load one 6^3 SCF took > 30 min, so (b1) is
    checked 4^3 -> 6^3 (same tolerances) and (b2) at 6^3; the lattice-constant scan (orientation only) is dropped.
    Reuses results already in mgo_bulk.json."""
    path = os.path.join(out_dir, "mgo_bulk.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for basis, k in (("gth-dzvp-molopt-sr", 6), ("gth-dzvp-molopt-sr", 4), ("gth-tzv2p", 6)):
        key = f"{basis}_{k}"
        if key in res:
            continue
        cell = bulk_cell(A_4G, basis)
        mf, e = scf(cell, (k, k, k))
        r = band_edges(mf, cell, A_4G)
        r["E_tot_eV"] = e * HARTREE
        res[key] = r
        print(f"bulk a={A_4G:.5f} {basis} k{k}: gap(Gamma) {r['gap_Gamma']:.4f} eV, m* "
              f"{r['m_star_100']:.4f} / {r['m_star_110']:.4f} / {r['m_star_111']:.4f}", flush=True)
        json.dump(res, open(path, "w"), indent=1)
    r4, r6 = res["gth-dzvp-molopt-sr_4"], res["gth-dzvp-molopt-sr_6"]
    ok = abs(r4["gap_Gamma"] - r6["gap_Gamma"]) < 0.02 and abs(r4["m_star_100"] / r6["m_star_100"] - 1) < 0.02
    print(f"(b1) k convergence 4 -> 6: {'PASS' if ok else 'FAIL'}", flush=True)
    d, t = r6, res["gth-tzv2p_6"]
    ok = abs(d["gap_Gamma"] - t["gap_Gamma"]) < 0.1 and abs(d["m_star_100"] / t["m_star_100"] - 1) < 0.05
    print(f"(b2) basis: {'PASS' if ok else 'FAIL'} (gap {d['gap_Gamma']:.3f} vs {t['gap_Gamma']:.3f}, m* "
          f"{d['m_star_100']:.3f} vs {t['m_star_100']:.3f})", flush=True)


if __name__ == "__main__":
    mode, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    if mode == "bulk":
        run_bulk(out_dir)
    elif mode == "bulk_reduced":
        run_bulk_reduced(out_dir)
    elif mode == "selftest":
        # (s1) on the cheap bulk cell first: the Hartree potential rebuilt as get_j builds it reproduces mf.get_j
        cell = bulk_cell(A_4G, "gth-szv", ke_cutoff=80.0)
        mf, e = scf(cell, (2, 2, 2))
        z, vz, err, rel = local_potential(mf, cell)
        print(f"selftest bulk: (s1) J check {err:.2e} Ha ({'PASS' if err < 1e-7 else 'FAIL'}); (s3) local PP "
              f"{rel:.2%} ({'PASS' if rel < 0.01 else 'FAIL'})", flush=True)
    elif mode == "bz":
        run_bz(out_dir)
    elif mode == "bandavg":
        run_band_average(out_dir)
    elif mode == "cutoff":
        run_cutoff_check(out_dir)
    elif mode == "cutoff200":
        run_cutoff_check(out_dir, cutoffs=(200.0,))
    elif mode == "slab":
        run_slab(out_dir)
