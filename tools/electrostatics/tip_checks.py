"""Two artifact checks on the total-ion-pseudopotential (TIP) result of doped_spin.py  (2026-10-26; diagnostics).

    OMP_NUM_THREADS=2 nice python tools/electrostatics/tip_checks.py doped_guess OUT_DIR
    OMP_NUM_THREADS=2 nice python tools/electrostatics/tip_checks.py undoped OUT_DIR
    (python: an interpreter with PySCF installed)

Measured (doped_spin.py, 10-ion cluster, upright Au2-, one Al3+): with bare point cations the unpaired spin sits on
the cluster's Mg (+1.090; Au2 -0.102, E = -2033.7038420659 Ha); with the 26 point Mg2+ within 3.5 A capped by TIPs it
sits on Au2 (+0.830; Mg +0.129).

(c1) 'doped_guess': is the bare-lattice spill a property of that Hamiltonian, or an SCF local minimum?  The TIP
solution's density on the quantum AOs (the first 398 AOs in both molecules) starts the bare-lattice UKS.  Pre-registered
reading: if it converges with spin on Au2 > 0.8 and an energy below -2033.7038420659 Ha, the spill was a local minimum
of the SCF (and H1, leakage in the Hamiltonian, is falsified as its cause); if it returns to the spilled state, or
stays on Au2 at a higher energy, the bare-lattice Hamiltonian itself prefers the spill.

(c2) 'undoped': do the TIPs change the undoped wetting numbers (wetting_eda.py r_au = 3.6, bare point cations:
dE_scf upright -1.2058, flat -0.0973 eV; wet +1118.8 meV)?  Same cluster, the same 26 cations capped, E_A, E_B
(each with the other's ghost functions, as wetting_eda) and E_AB per geometry.  Pre-registered: the undoped cluster
results stand if |wet(TIP) - wet(bare)| <= 0.10 eV and |dE_scf(TIP) - dE_scf(bare)| <= 0.10 eV in both geometries;
otherwise the undoped decomposition and the 28-ion ALMO split need TIP boundaries.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cluster_interaction as CI  # noqa: E402
import doped_spin as DS  # noqa: E402
import wetting_eda as W  # noqa: E402

E_SPILL = -2033.7038420659
BARE = {"upright": -1.205825388328977, "flat": -0.09729894234382003}
E_AU2 = {"flat": -7389.903531682977, "upright": -7389.91381537512}
WET_BARE = (BARE["flat"] + E_AU2["flat"]) - (BARE["upright"] + E_AU2["upright"])


def _start_from(mf, dm0):
    mf.get_init_guess = lambda *args, **kwargs: dm0
    return mf


def _spin_report(mol, mf, n_au, surf, tag):
    dm = mf.make_rdm1()
    S = mol.intor_symmetric("int1e_ovlp")
    sl = mol.aoslice_by_atom()
    pop = lambda m, a: float(np.einsum("ij,ji->", m[:, sl[a][2]:sl[a][3]], S[sl[a][2]:sl[a][3], :]))
    spin = [pop(dm[0] - dm[1], a) for a in range(mol.natm)]
    nq = n_au + len(surf)
    s_au = sum(spin[:n_au])
    s_mg = sum(s for (n, _), s in zip(surf, spin[n_au:nq]) if n == "Mg")
    print(f"[{tag}] E {mf.e_tot:.10f} Ha; spin on Au2 {s_au:+.3f}, on the cluster's Mg {s_mg:+.3f}", flush=True)
    return s_au, s_mg


def doped_guess(out):
    from pyscf import dft
    mol_t, lat_t, A, surf, tips = DS.build("tip", 3.6)
    g = dft.gen_grid.Grids(mol_t)
    g.level = 3
    g.build()
    mf_t, _ = W._solve(W._uks(mol_t, g, lat_t))
    _spin_report(mol_t, mf_t, len(A), surf, "TIP, default guess")
    dm_t = mf_t.make_rdm1()
    np.save(os.path.join(out, "doped_tip_dm_r3.6.npy"), dm_t)
    mol_p, lat_p, _, _, _ = DS.build("plain", 3.6)
    n = mol_p.nao
    assert mol_t.aoslice_by_atom()[len(A) + len(surf) - 1][3] == n, "the quantum AOs come first, in the same order"
    g = dft.gen_grid.Grids(mol_p)
    g.level = 3
    g.build()
    mf_p = _start_from(W._uks(mol_p, g, lat_p), dm_t[:, :n, :n])
    mf_p, e = W._solve(mf_p)
    s_au, _ = _spin_report(mol_p, mf_p, len(A), surf, "bare lattice, from the TIP density")
    dE = (e - E_SPILL) * W.HARTREE
    if s_au > 0.8 and e < E_SPILL:
        verdict = "the spill was an SCF local minimum (H1 falsified as its cause)"
    else:
        verdict = "the bare-lattice Hamiltonian prefers the spill (consistent with H1)"
    print(f"(c1) E(from TIP density) - E(spilled) = {dE * 1000:+.1f} meV; Au2 spin {s_au:+.3f} -> {verdict}", flush=True)


def _tip_capped(surf, lattice):
    coords, q = lattice
    dist = np.linalg.norm(coords[:, None, :] - np.array([p for _, p in surf])[None], axis=2).min(1)
    tips = np.nonzero((dist < DS.R_TIP) & (q > 0))[0]
    assert np.allclose(q[tips], 2.0)
    keep = np.ones(len(q), dtype=bool)
    keep[tips] = False
    return [tuple(coords[j]) for j in tips], (coords[keep], q[keep])


def undoped(out):
    from pyscf import dft, gto
    G = W.geometries()
    sites = W.quantum_sites(G["flat"] + G["upright"], r_au=3.6)
    surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
    qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf)
    tips, lat = _tip_capped(surf, CI.rocksalt_lattice(W.D, 5, True, exclude=sites))
    print(f"{len(surf)} quantum ions (charge {qB:+d}), {len(tips)} TIPs, {len(lat[1])} point ions", flush=True)

    def mol(au_real, au_ghost, with_surf):
        spec = [["Au", tuple(p)] for p in au_real] + [["ghost-Au", tuple(p)] for p in au_ghost]
        spec += [[s if with_surf else "ghost-" + s, tuple(p)] for s, p in surf]
        basis = {"Au": W.BASIS, "ghost-Au": W.BASIS, "Mg": W.BASIS, "O": W.BASIS, "ghost-Mg": W.BASIS,
                 "ghost-O": W.BASIS}
        ecp = {"Au": W.BASIS} if au_real else {}
        charge = 0
        if with_surf:
            spec += [["Mg1", t] for t in tips]
            basis["Mg1"] = DS.TIGHT
            ecp["Mg1"] = "stuttgart"
            charge = qB + 2 * len(tips)
        return gto.M(atom=spec, basis=basis, ecp=ecp, charge=charge, spin=0, unit="Angstrom", verbose=0)

    res = {"n_tips": len(tips)}
    path = os.path.join(out, "tip_undoped_r3.6.json")
    for name in ("upright", "flat"):
        au = G[name]
        dimer = mol(au, [], True)
        g = dft.gen_grid.Grids(dimer)
        g.level = 3
        g.build()
        _, eA = W._solve(W._rks(mol(au, [], False), g))
        _, eB = W._solve(W._rks(mol([], au, True), g, lat))
        mfAB, eAB = W._solve(W._rks(dimer, g, lat))
        q_au = float(sum(mfAB.mulliken_pop(verbose=0)[1][:2]))
        res[name] = {"E_A": eA * W.HARTREE, "E_B": eB * W.HARTREE, "E_AB": eAB * W.HARTREE,
                     "dE_scf": (eAB - eA - eB) * W.HARTREE, "q_Au2_mulliken": q_au}
        print(f"{name}: dE_scf {res[name]['dE_scf']:+.4f} eV with TIPs (bare {BARE[name]:+.4f}); Au2 Mulliken "
              f"{q_au:+.3f} e", flush=True)
        json.dump(res, open(path, "w"), indent=1)
    wet = (res["flat"]["dE_scf"] + E_AU2["flat"]) - (res["upright"]["dE_scf"] + E_AU2["upright"])
    res["wet"] = wet
    json.dump(res, open(path, "w"), indent=1)
    ok = abs(wet - WET_BARE) <= 0.10 and all(abs(res[k]["dE_scf"] - BARE[k]) <= 0.10 for k in BARE)
    print(f"(c2) wet {wet * 1000:+.1f} meV with TIPs, {WET_BARE * 1000:+.1f} bare (DFT +929); dE_scf shifts upright "
          f"{(res['upright']['dE_scf'] - BARE['upright']) * 1000:+.1f}, flat {(res['flat']['dE_scf'] - BARE['flat']) * 1000:+.1f}"
          f" meV -> {'the undoped cluster results stand' if ok else 'the undoped results need TIP boundaries'}",
          flush=True)


if __name__ == "__main__":
    os.makedirs(sys.argv[2], exist_ok=True)
    {"doped_guess": doped_guess, "undoped": undoped}[sys.argv[1]](sys.argv[2])
