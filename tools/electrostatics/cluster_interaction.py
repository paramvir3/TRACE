"""First-order (frozen-density) interaction of N charge-state ions, and its non-additivity (2026-10-20).

    OMP_NUM_THREADS=4 nice python tools/electrostatics/cluster_interaction.py OUT_DIR
    (python: an interpreter with PySCF installed)

trace_redox's overlap term is a sum over pairs.  Electrostatics of frozen densities is exactly pairwise; the exchange
(the orthogonalisation of the monomers' occupied spaces to one another) is not.  For a carrier with many neighbours,
the band electron on Mg with six O2- in rock salt, the question is whether the pair sum is the cluster's value.

For monomers m = 1..N (library orbitals, frozen; frozen_interaction._monomer), the Hamiltonian of the N ions alone
('vac' convention of frozen_interaction.decomposed_interaction):

    E_HL  = E[P],  P^s = projector onto all monomers' occupied spin orbitals together (antisymmetrised product)
    E_HP  = sum_m E_m + elst                       (Hartree product: exchange inside each monomer only)
    exch  = E_HL - E_HP                            (HF expression: the first-order exchange)
    nonelst_xc = Tr[dP h] + (dP | sum_m rho_m) + 1/2 (dP | dP) + exx_xc + dfa_xc,   dP = P - sum_m P_m

the same exact decomposition as the pair case, which it reproduces for N = 2 (checked).  The non-additivity of a
quantity X over the carrier's environment, relative to the frames, is

    A = [X(carrier state, env) - X(frame state, env)] - sum_neighbours [X(carrier, nb) - X(frame, nb)]

(the environment-environment pairs cancel in the frame difference; what is left is three-body and higher).
Pre-registered: the pair model is adequate for the band electron if |A| < 10 % of the pair sum, for exch (HF) and
nonelst (PBE).

Embedding (added 2026-10-20, before the first cluster run).  A bare Mg+(O2-)6 cluster carries six unscreened -2 ions,
whose fields act on every exchange density; in the crystal the rest of the lattice cancels most of them (the pair
measurements show the field on the exchange density is worth ~0.5 eV per contact: 'vac' against 'sph').  So the
decisive runs put the quantum ions in the rest of the rock-salt lattice as point ions (Gaussian width 0.3 A; Evjen
weights on the faces of a cube of half-width n d, or of a slab for the (100) surface; the quantum sites excluded), the
same lattice in every Hamiltonian of a comparison.  Then

    nonelst = Tr[dP (h + V_lattice)] + ... ,    dP_field = Tr[dP V_lattice]  (reported),
    elst_lattice = the quantum ions' frozen densities and nuclei in V_lattice (reported, not part of the comparison).

Pre-registered, besides the 10 % criterion (now for the embedded cluster against the embedded pairs):
  (m) the lattice's potential at a bulk cation site (all other ions as points) is within 0.5 % of the Madelung value
      -alpha_M 2 K / d = -23.90 V (alpha_M = 1.747565), and changes by < 0.5 % from n = 4 to n = 6 (bulk and surface);
  (c) a pair convention ('vac', the ions alone; 'sph', the O state's Watson sphere in the pair Hamiltonian) is
      adequate if its frame-relative nonelst (PBE) is within 0.1 eV of the embedded pair's.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import frozen_interaction as FI  # noqa: E402

HARTREE = 27.211386245988


def cluster_first_order(monomers, xcs=("hf", "pbe"), grid_level=4, lattice=None, return_dP=False, with_elst=True):
    """monomers: [(symbol, q, centre (A))]; lattice: None or (coords [M, 3] A, charges [M]) point ions.
    eV: {'elst', 'exch', 'nonelst_<xc>', 'dP_field', 'elst_lattice', ...}."""
    from pyscf import dft, scf
    occs, spins = [], []
    for sym, q, c in monomers:
        occ, spin, _ = FI._monomer(sym, q, np.asarray(c, float), None, None)
        occs.append(occ)
        spins.append(spin)
    mol = FI._mol([(sym, np.asarray(c, float)) for sym, _, c in monomers], sum(q for _, q, _ in monomers), sum(spins))
    nao = mol.nao_nr()
    offsets = mol.aoslice_by_atom()[:, 2]

    def embed(c, offset):
        out = np.zeros((nao, c.shape[1]))
        out[offset:offset + c.shape[0]] = c
        return out

    C = [[embed(occs[m][s], offsets[m]) for m in range(len(monomers))] for s in (0, 1)]
    S = mol.intor_symmetric("int1e_ovlp")

    def proj(c):
        return c @ np.linalg.solve(c.T @ S @ c, c.T) if c.shape[1] else np.zeros((nao, nao))

    P_all = np.array([proj(np.hstack(C[s])) for s in (0, 1)])
    P_m = [np.array([proj(C[s][m]) for s in (0, 1)]) for m in range(len(monomers))]
    P_sum = sum(P_m)
    dP = (P_all - P_sum).sum(0)
    Zs = [FI.SYMBOL_TO_Z[sym] for sym, _, _ in monomers]
    # elst below uses the non-relativistic nuclear potential: refused with X2C unless it is switched off (nonelst and
    # dP_field use the X2C Hamiltonian itself and are exact for it)
    assert with_elst is False or max(Zs) <= 36, "X2C cluster: pass with_elst=False"
    mf = FI._decorate(scf.UHF(mol), Zs, [None])
    h = mf.get_hcore()
    dP_field, elst_lattice = 0.0, 0.0
    if lattice is not None:
        from pyscf import qmmm
        coords, charges = np.asarray(lattice[0], float), np.asarray(lattice[1], float)
        mf_l = qmmm.mm_charge(FI._decorate(scf.UHF(mol), Zs, [None]), coords, charges,
                              radii=np.full(len(charges), 0.3), unit="Angstrom")
        V_l = mf_l.get_hcore() - h
        h = h + V_l
        dP_field = float(np.einsum("ij,ji->", dP, V_l))
        elst_lattice = float(np.einsum("ij,ji->", P_sum.sum(0), V_l)) + mf_l.energy_nuc() - mol.energy_nuc()
    # one pass over the cluster's two-electron integrals: J and K of [P^a, P^b, P_sum, dP] (only the needed parts used)
    vj, vk = mf.get_jk(mol, np.array([P_all[0], P_all[1], P_sum.sum(0), dP]), hermi=1)
    k_all = sum(float(np.einsum("ij,ji->", vk[s], P_all[s])) for s in (0, 1))
    # each monomer's own Hartree and exchange: its orbitals live on its own functions, so the integrals are the
    # monomer's own (computed once per species in its own basis)
    own = [_own(sym, q) for sym, q, _ in monomers]
    k_m = sum(o["K"] for o in own)
    j_self = sum(o["J"] for o in own)
    # elst, exactly pairwise: nuclei, electron-nucleus, and sum_{m<n} (rho_m|rho_n) = 1/2 [(rho_sum|rho_sum) - sum_m (rho_m|rho_m)]
    elst = float("nan")
    if with_elst:
        elst = 0.5 * (float(np.einsum("ij,ji->", vj[2], P_sum.sum(0))) - j_self)
        coords = mol.atom_coords()
        for m in range(len(monomers)):
            for n in range(len(monomers)):
                if m == n:
                    continue
                with mol.with_rinv_origin(coords[n]):
                    elst -= Zs[n] * float(np.einsum("ij,ji->", P_m[m].sum(0), mol.intor_symmetric("int1e_rinv")))
                if n > m:
                    elst += Zs[m] * Zs[n] / np.linalg.norm(coords[m] - coords[n])
    out = {"nao": int(nao), "number": float(np.einsum("ij,ji->", dP, S)), "elst": elst * HARTREE,
           "dP_field": dP_field * HARTREE, "elst_lattice": elst_lattice * HARTREE}
    onebody = float(np.einsum("ij,ji->", dP, h))
    coul = float(np.einsum("ij,ji->", vj[2], dP)) + 0.5 * float(np.einsum("ij,ji->", vj[3], dP))
    out["onebody"], out["coul"] = onebody * HARTREE, coul * HARTREE
    out["kin"] = float(np.einsum("ij,ji->", dP, mol.intor_symmetric("int1e_kin"))) * HARTREE
    grids = dft.gen_grid.Grids(mol)
    grids.level = grid_level
    grids.build()
    ni = dft.numint.NumInt()
    for xc in xcs:
        a = 1.0 if xc == "hf" else ni.hybrid_coeff(xc)
        if xc != "hf":
            assert ni.rsh_coeff(xc)[0] == 0
        exx = -0.5 * a * (k_all - k_m)
        if xc == "hf":
            dfa = 0.0
        else:
            dfa = float(ni.nr_uks(mol, grids, xc, P_all)[1]) - sum(float(ni.nr_uks(mol, grids, xc, P)[1]) for P in P_m)
        out[f"exx_{xc}"], out[f"dfa_{xc}"] = exx * HARTREE, dfa * HARTREE
        out[f"nonelst_{xc}"] = (onebody + coul + exx + dfa) * HARTREE
    out["exch"] = out["nonelst_hf"] if "hf" in xcs else None
    if return_dP:
        return out, mol, dP
    return out


_OWN = {}


def _own(sym, q):
    """The monomer's own (rho|rho) and sum_s Tr[P^s K(P^s)] (hartree), in its own basis."""
    if (sym, q) not in _OWN:
        from pyscf import scf
        occ, spin, _ = FI._monomer(sym, q, np.zeros(3), None, None)
        mol = FI._mol([(sym, np.zeros(3))], q, spin)
        P = np.array([occ[s] @ occ[s].T for s in (0, 1)])
        vj, vk = scf.UHF(mol).get_jk(mol, P, hermi=1)
        _OWN[(sym, q)] = {"J": float(np.einsum("ij,ji->", vj[0] + vj[1], P[0] + P[1])),
                          "K": sum(float(np.einsum("ij,ji->", vk[s], P[s])) for s in (0, 1))}
    return _OWN[(sym, q)]


def rocksalt_site(carrier, frame, ligand, d, n_ligands=6):
    """The carrier at the origin with its octahedral (6) or square-pyramidal surface (5) ligands at distance d."""
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1)][:n_ligands]
    env = [(ligand[0], ligand[1], tuple(d * np.array(u, float))) for u in dirs]
    return [(carrier[0], carrier[1], (0.0, 0.0, 0.0))] + env, [(frame[0], frame[1], (0.0, 0.0, 0.0))] + env


K_E = 14.399645
ALPHA_ROCKSALT = 1.747565


def rocksalt_lattice(d, n, surface=False, exclude=()):
    """Point ions of rock-salt MgO with a cation (+2) at the origin: sites (i, j, k) d with |i|, |j| <= n and
    |k| <= n (bulk) or -n <= k <= 0 (the (100) surface at k = 0, vacuum above), Evjen weights 1/2 per outer face (the
    surface layer itself is not a cut face).  ``exclude``: integer sites left out (the quantum ions).
    Returns (coords [M, 3] A, charges [M])."""
    pts, q = [], []
    ex = {tuple(e) for e in exclude}
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            for k in (range(-n, 1) if surface else range(-n, n + 1)):
                if (i, j, k) in ex:
                    continue
                w = (0.5 if abs(i) == n else 1.0) * (0.5 if abs(j) == n else 1.0)
                w *= (0.5 if k == -n else 1.0) if surface else (0.5 if abs(k) == n else 1.0)
                pts.append((i * d, j * d, k * d))
                q.append(w * (2.0 if (i + j + k) % 2 == 0 else -2.0))
    return np.array(pts), np.array(q)


def site_potential(d, n, surface, site=(0, 0, 0)):
    """V (volts) at ``site`` from every other point ion of the lattice."""
    pts, q = rocksalt_lattice(d, n, surface, exclude=[site])
    r = np.linalg.norm(pts - np.asarray(site, float) * d, axis=1)
    return float(np.sum(q * K_E / r))


def _diff(X1, X0, k):
    return X1[k] - X0[k]


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    d = 2.106
    path = os.path.join(out_dir, "cluster_mg_o.json")
    results = {}

    def save():
        json.dump(results, open(path, "w"), indent=1)

    # (1) N = 2 must reproduce decomposed_interaction (vac): the code path check
    pair = cluster_first_order([("Mg", 2, (0.0, 0.0, 0.0)), ("O", -2, (0.0, 0.0, d))])
    ref = FI.decomposed_interaction(("Mg", 2), ("O", -2), d, xcs=("hf", "pbe"), grid_level=4)
    for k_c, k_r in (("elst", "vac_elst"), ("exch", "vac_nonelst_hf"), ("nonelst_pbe", "vac_nonelst_pbe")):
        diff = pair[k_c] - ref[k_r]
        print(f"N=2 check {k_c}: cluster {pair[k_c]:+.6f}, pair code {ref[k_r]:+.6f}, diff {diff:+.2e} eV "
              f"({'PASS' if abs(diff) < 1e-6 else 'FAIL'}, criterion 1e-6)", flush=True)
    results["check_N2"] = {"cluster": pair, "pair": ref}
    # (m) the lattice's Madelung potential
    exact = -ALPHA_ROCKSALT * 2.0 * K_E / d
    v4, v6 = site_potential(d, 4, False), site_potential(d, 6, False)
    s4, s6 = site_potential(d, 4, True), site_potential(d, 6, True)
    ok_m = abs(v4 / exact - 1) < 5e-3 and abs(v4 / v6 - 1) < 5e-3 and abs(s4 / s6 - 1) < 5e-3
    print(f"Madelung check: bulk cation site n=4 {v4:+.4f} V, n=6 {v6:+.4f} V, exact {exact:+.4f} V; (100) surface "
          f"cation n=4 {s4:+.4f} V, n=6 {s6:+.4f} V: {'PASS' if ok_m else 'FAIL'}", flush=True)
    results["madelung"] = {"bulk_n4": v4, "bulk_n6": v6, "exact": exact, "surface_n4": s4, "surface_n6": s6,
                           "pass": ok_m}
    save()
    if not ok_m:
        return
    n_lat = 4
    # (2) bulk (6 O) and (100) surface (4 in-plane O + 1 below), embedded; the pairs vac and embedded
    for label, surface, ligand_sites in (("bulk", False, [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
                                                          (0, 0, -1)]),
                                         ("surface", True, [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, -1)])):
        res = {}
        env = [("O", -2, tuple(d * np.array(u, float))) for u in ligand_sites]
        lat_cluster = rocksalt_lattice(d, n_lat, surface, exclude=[(0, 0, 0)] + ligand_sites)
        for q in (1, 2):
            res[f"cluster_emb_{q}"] = cluster_first_order([("Mg", q, (0.0, 0.0, 0.0))] + env, lattice=lat_cluster)
            print(f"{label} cluster Mg{q:+d} done", flush=True)
        # the distinct ligand environments (bulk: all equivalent; surface: in-plane and the one below)
        kinds = {"bulk": {(1, 0, 0): 6}, "surface": {(1, 0, 0): 4, (0, 0, -1): 1}}[label]
        for site, mult in kinds.items():
            lat_pair = rocksalt_lattice(d, n_lat, surface, exclude=[(0, 0, 0), site])
            for q in (1, 2):
                mono = [("Mg", q, (0.0, 0.0, 0.0)), ("O", -2, tuple(d * np.array(site, float)))]
                res[f"pair_emb_{site}_{q}"] = cluster_first_order(mono, lattice=lat_pair)
                res[f"pair_vac_{site}_{q}"] = cluster_first_order(mono)
        results[label] = {k: v for k, v in res.items()}
        save()
        for k in ("exch", "nonelst_pbe", "dP_field"):
            cl = _diff(res["cluster_emb_1"], res["cluster_emb_2"], k)
            pe = sum(m * _diff(res[f"pair_emb_{s}_1"], res[f"pair_emb_{s}_2"], k) for s, m in kinds.items())
            pv = sum(m * _diff(res[f"pair_vac_{s}_1"], res[f"pair_vac_{s}_2"], k) for s, m in kinds.items())
            print(f"{label} Mg+ (frame Mg2+), {k}: embedded cluster {cl:+.4f} eV; sum of embedded pairs {pe:+.4f} "
                  f"(non-additivity {cl - pe:+.4f}, {(cl - pe) / pe * 100 if pe else float('nan'):+.1f} %); sum of "
                  f"vacuum pairs {pv:+.4f} (vacuum pair model error {cl - pv:+.4f})", flush=True)
        for s, m in kinds.items():
            pe = _diff(res[f"pair_emb_{s}_1"], res[f"pair_emb_{s}_2"], "nonelst_pbe")
            pv = _diff(res[f"pair_vac_{s}_1"], res[f"pair_vac_{s}_2"], "nonelst_pbe")
            print(f"{label} ligand {s}: frame-relative nonelst (PBE) embedded pair {pe:+.4f}, vac pair {pv:+.4f} eV "
                  f"(vac {'adequate' if abs(pe - pv) < 0.1 else 'NOT adequate'} at 0.1 eV)", flush=True)
    # (3) diagnostic, last: the bare clusters (no lattice), to show the size of the unscreened field
    for n_lig in (6, 5):
        c_state, c_frame = rocksalt_site(("Mg", 1), ("Mg", 2), ("O", -2), d, n_lig)
        X1, X0 = cluster_first_order(c_state), cluster_first_order(c_frame)
        results[f"bare_cluster_{n_lig}"] = {"state": X1, "frame": X0}
        save()
        print(f"bare Mg+(O2-){n_lig} (no lattice), nonelst_pbe frame-relative {_diff(X1, X0, 'nonelst_pbe'):+.4f} eV",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
