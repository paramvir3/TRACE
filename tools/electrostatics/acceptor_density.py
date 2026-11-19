# One of the measurement scripts of the theory note's section 4c (2026-10-29); run from the repo root.
"""The acceptor orbital as actually used: the surface -> Au2 transfer density dD = D(B->A minimum) - D(ALMO minimum)
in the ten-ion cluster (almo_split's directional machinery), analysed per Au: Mulliken shells (s, p_z, p_xy, d, f) of
the gained density and its dipole about the Au along the normal, split into the intra-Au part (the s-p_z cross term:
the acceptor's own lobe), the Au-Au' part and the Au-surface part (the bond-overlap density, always toward the donor).
Geometries: 'flat' and 'upright' (the 4G references), 'mgtop' (the vertical dimer over Mg0, h = 2.89).
Usage: python acceptor_density.py NAME OUT"""
import sys, os, json, numpy as np
sys.path.insert(0, "tools/electrostatics")
import almo_split as AS, wetting_eda as W, cluster_interaction as CI
from pyscf import dft
BOHR = 0.52917721092
name, out = sys.argv[1], sys.argv[2]
G = W.geometries()
sites = W.quantum_sites(G["flat"] + G["upright"], r_au=3.6)
lattice = CI.rocksalt_lattice(W.D, 5, True, exclude=sites)
if name.startswith("upright_h"):                       # the upright dimer at an arbitrary O-top height (2026-10-30)
    h = float(name[len("upright_h"):])
    O_site = W.D * np.array([1.0, 0.0, 0.0])
    au = [O_site + np.array([0.0, 0.0, h]), O_site + np.array([0.0, 0.0, h + 2.545])]
else:
    au = {"flat": G["flat"], "upright": G["upright"],
          "mgtop": [np.array([0.0, 0.0, 2.89]), np.array([0.0, 0.0, 2.89 + 2.545])]}[name]
surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf)
A_atoms = [("Au", np.asarray(p, float)) for p in au]
dimer = W._mol(A_atoms + surf, qB)
grids = dft.gen_grid.Grids(dimer); grids.level = 3; grids.build()
molA_own, molB_own = W._mol(A_atoms, 0), W._mol(surf, qB)
nA = molA_own.nao_nr()
mfA, eA = W._solve(W._rks(molA_own, AS._grid_like(molA_own, grids)))
mfB, eB = W._solve(W._rks(molB_own, AS._grid_like(molB_own, grids), lattice))
mfAB, eAB = W._solve(W._rks(dimer, grids, lattice))
se = AS.SpanEnergy(mfAB); S = se.S
dens, energies = {}, {}
for key, allow in (("ALMO", {"A": ("A",), "B": ("B",)}), ("B->A", {"A": ("A",), "B": ("B", "A")})):
    de = AS.DirectionalEnergy(se, mfA, mfB, nA, allow)
    e0, e, res = AS.minimise_rot(de, label=f" {name} {key}")
    C = de.orbitals(res.x)
    M = np.linalg.inv(C.T @ S @ C)
    dens[key], energies[key] = 2.0 * C @ M @ C.T, e
    if key == "ALMO":
        C_almo, de_almo = C, de
    print(f"{name} {key}: E {e * W.HARTREE:.4f} eV, converged {bool(res.success)}, |g|_max {np.abs(res.jac).max():.1e}", flush=True)
# H3 diagnostic: the acceptor and donor levels in the ALMO state.  F of the ALMO density; A's own virtual space
# (the monomer virtuals embedded in the dimer basis) projected orthogonal to every occupied ALMO orbital (P S, the
# S-metric projector), canonically orthogonalised; its lowest eigenvalue is the polarised sigma*; B's occupied block's
# highest eigenvalue is the donor top.  Reference: the free dimer's LUMO and the free surface's HOMO.
from scipy.linalg import eigh
def _levels(C, de_almo):
    M = np.linalg.inv(C.T @ S @ C); P = C @ M @ C.T
    F = se.h + se.mf.get_veff(se.mol, 2.0 * P)
    VA = de_almo.virt["A"]; Vp = VA - P @ S @ VA
    def canon(X):
        Sx = X.T @ S @ X; w, U = np.linalg.eigh(Sx); keep = w > 1e-8
        T = U[:, keep] / np.sqrt(w[keep]); return np.linalg.eigvalsh(T.T @ (X.T @ F @ X) @ T)
    ev = canon(Vp); nA_occ = de_almo.occ["A"].shape[1]; eb = canon(C[:, nA_occ:])
    return float(ev[0]) * W.HARTREE, float(ev[1]) * W.HARTREE, float(eb[-1]) * W.HARTREE
e_acc, e_acc2, e_don = _levels(C_almo, de_almo)
e_lumo_free = float(mfA.mo_energy[mfA.mo_occ == 0].min()) * W.HARTREE
e_homo_free = float(mfB.mo_energy[mfB.mo_occ > 0].max()) * W.HARTREE
print(f"{name} ALMO levels: acceptor (A virtual, lowest) {e_acc:+.3f} eV (next {e_acc2:+.3f}), donor top (B occupied) {e_don:+.3f} eV, "
      f"Delta {e_acc - e_don:.3f} eV; free dimer LUMO {e_lumo_free:+.3f}, free surface HOMO {e_homo_free:+.3f}, Delta_free {e_lumo_free - e_homo_free:.3f} eV", flush=True)
dD = dens["B->A"] - dens["ALMO"]
ct = (energies["B->A"] - energies["ALMO"]) * W.HARTREE
pop = np.einsum("ij,ji->i", dD, S)
labels = dimer.ao_labels(fmt=False)
idx_atom = {ia: [k for k, lab in enumerate(labels) if lab[0] == ia] for ia in range(len(A_atoms))}
idxB = list(range(nA, dimer.nao_nr()))
assert all(k >= nA for k in idxB) and max(max(v) for v in idx_atom.values()) < nA
print(f"{name}: CT(B->A) raw {ct:+.4f} eV; electron check tr(dD S) {np.trace(dD @ S):+.1e}; gained on A {pop[:nA].sum():+.4f} e, on B {pop[nA:].sum():+.4f} e", flush=True)
rec = {"name": name, "CT_raw_B->A_eV": ct, "levels_eV": {"acceptor": e_acc, "acceptor_2": e_acc2, "donor_top": e_don, "free_LUMO": e_lumo_free, "free_HOMO": e_homo_free}, "gained_A": float(pop[:nA].sum()), "gained_B": float(pop[nA:].sum()), "Au": []}
# the dimer frame: u along the bond (from this Au to its partner), q in-plane perpendicular, z the normal; the 6p
# populations are rotated into it from the symmetrised Mulliken matrix N = (dD S + S dD) / 2 restricted to each p shell
# (p_axis = u^T N u, p_perp = q^T N q, p_z = z^T N z; their sum is the shell's Mulliken population).
N = 0.5 * (dD @ S + S @ dD)
zhat = np.array([0.0, 0.0, 1.0])
for ia, (sym, R) in enumerate(A_atoms):
    me = idx_atom[ia]; other = idx_atom[1 - ia]
    u = A_atoms[1 - ia][1] - R; u = u / np.linalg.norm(u)
    q = np.cross(zhat, u); q = q / np.linalg.norm(q) if np.linalg.norm(q) > 1e-8 else np.array([1.0, 0.0, 0.0])
    frame = {"p_axis": 0.0, "p_perp": 0.0, "p_z_frame": 0.0}
    shells_p = {}
    for k in me:
        if labels[k][2][-1] == "p":
            shells_p.setdefault(labels[k][2], {})[labels[k][3]] = k
    for nl, comp in shells_p.items():
        kk = [comp["x"], comp["y"], comp["z"]]
        Np = N[np.ix_(kk, kk)]
        frame["p_axis"] += float(u @ Np @ u); frame["p_perp"] += float(q @ Np @ q); frame["p_z_frame"] += float(zhat @ Np @ zhat)
    shells = {"s": 0.0, "pz": 0.0, "pxy": 0.0, "d": 0.0, "f": 0.0}
    for k in me:
        l, m = labels[k][2][-1], labels[k][3]
        if l == "p":
            shells["pz" if m == "z" else "pxy"] += float(pop[k])
        else:
            shells[l] += float(pop[k])
    with dimer.with_common_origin(R / BOHR):
        z_int = dimer.intor("int1e_r")[2]
    n_me = float(pop[me].sum())
    parts = {}
    for tag, cols in (("intra", me), ("partner", other), ("surface", idxB)):
        parts[tag] = float(np.einsum("ij,ji->", dD[np.ix_(me, cols)], z_int[np.ix_(cols, me)])) * BOHR
    eff = parts["intra"] / n_me if abs(n_me) > 1e-6 else float("nan")
    print(f"  Au{ia} z {R[2]:.3f}: gained {n_me:+.4f} e; shells " + ", ".join(f"{k} {v:+.4f}" for k, v in shells.items())
          + f"; dimer frame p_axis {frame['p_axis']:+.4f}, p_perp {frame['p_perp']:+.4f}, p_z {frame['p_z_frame']:+.4f}"
          + f"; mu_z (e A) intra {parts['intra']:+.5f}, with partner {parts['partner']:+.5f}, with surface {parts['surface']:+.5f}"
          f"; intra per gained electron {eff:+.3f} A", flush=True)
    rec["Au"].append({"z": float(R[2]), "gained": n_me, "shells": shells, "frame": frame, "mu_z_eA": parts, "intra_per_electron_A": eff})
json.dump(rec, open(os.path.join(out, f"acceptor_{name}.json"), "w"), indent=1)
