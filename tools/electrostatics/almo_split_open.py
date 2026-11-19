"""The direction of the O-top charge transfer for the doped dimer Au2- (open shell; 2026-11-02).

    OMP_NUM_THREADS=4 nice python tools/electrostatics/almo_split_open.py --check
    OMP_NUM_THREADS=4 nice python tools/electrostatics/almo_split_open.py OUT_DIR [R_AU]
    (python: an interpreter with PySCF installed)

Why: the model's folded O 2p -> Au 6s donation (SPEC 32) is quenched by the doped cell's extra electron. On step 23
seed 42 the upright dimer's fold dressing is -0.67 / -1.16 eV (undoped configurations) against 0.0 / -0.54 eV (doped:
0.0 when the contact Au holds the extra electron, whose 6s is then full). The quench matches the in-distribution T5 error
that the fold introduced (training-set class offsets, doped minus undoped: step 20 -1 / -12 meV, step 23 -318 / -598).
PBE's one-Al cluster (10 ions, 26 TIPs) relaxes the upright Au2- by -1.772 eV against -1.870 eV for the neutral dimer.
Which channel keeps it: O -> Au donation into the half-filled sigma*, or Au2- -> surface back-donation?

Method: almo_split.py's one-directional delocalisation (ESTABLISHED in substance: Khaliullin, Bell and Head-Gordon,
J. Chem. Phys. 128, 184112 (2008)), generalised to unrestricted Kohn-Sham: one density matrix per spin,
P_s = C_s (C_s^T S C_s)^-1 C_s^T, E = E_UKS[P_a, P_b], dE/dC_s = 2 (1 - S P_s) F_s C_s (C_s^T S C_s)^-1 (one electron
per spin orbital, so 2 where the closed-shell form has 4). Each fragment's occupied orbitals of spin s rotate into the
spin-s virtual orbitals of the fragments it is allowed; a closed-shell fragment carries identical alpha and beta sets.
A = Au2- (UKS doublet), B = the quantum MgO cluster with its point-ion lattice, the Al3+ of wetting_eda's doped model and
the TIPs (Mg2+ total-ion pseudopotentials within 3.5 A); the counterpoise attribution as almo_split.run_direction.

Pre-registered checks (written before any cluster number is read):
(u1) the closed-shell limit: the water dimer run unrestricted (spin 0) reproduces almo_split's restricted E_ALMO,
     CT(A->B) and CT(B->A) to 1e-6 eV;
(u2) the open-shell gradient: OH (doublet) + H2O against central differences, relative 1e-5, all directions open;
(u3) the open-shell E[all] reaches E_SCF of the same UKS state to 1e-5 eV;
(u4) on the cluster: |E[all] - E_SCF| < 1 meV, and the unpaired spin on Au2 in the SCF > 0.8 (wetting_eda's gate).
Scope, stated: the one-Al cluster is the sub-problem "Au2- over the insulating surface with one ionised donor"; it has
neither the 4G cell's three donors nor its two band electrons, and 10 ions are not converged (undoped 10 -> 28 ions
moved the wetting energy by -142 meV). It answers which channel the anion's O-top relaxation uses, not the 4G numbers.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import almo_split as AS  # noqa: E402
import cluster_interaction as CI  # noqa: E402
import wetting_eda as W  # noqa: E402

HARTREE = W.HARTREE


class SpanEnergyU:
    """E(C_a, C_b) for full-row occupied coefficients of each spin and its gradient per spin."""

    def __init__(self, mf):
        self.mf, self.mol = mf, mf.mol
        self.S = self.mol.intor_symmetric("int1e_ovlp")
        self.h = mf.get_hcore()
        self.nao = self.mol.nao_nr()

    def __call__(self, C):
        P, M = [], []
        for Cs in C:
            Ms = np.linalg.inv(Cs.T @ self.S @ Cs)
            P.append(Cs @ Ms @ Cs.T)
            M.append(Ms)
        dm = np.array(P)
        veff = self.mf.get_veff(self.mol, dm)
        E = float(self.mf.energy_tot(dm=dm, h1e=self.h, vhf=veff))
        G = [2.0 * (np.eye(self.nao) - self.S @ P[s]) @ (self.h + veff[s]) @ C[s] @ M[s] for s in (0, 1)]
        return E, G


def _spin_sets(mf):
    """(mo_coeff, mo_occ, mo_energy) per spin for an RKS or UKS object (a closed shell gives identical sets)."""
    if np.asarray(mf.mo_coeff).ndim == 3:
        return [(mf.mo_coeff[s], mf.mo_occ[s], mf.mo_energy[s]) for s in (0, 1)]
    return [(mf.mo_coeff, mf.mo_occ, mf.mo_energy)] * 2


class DirectionalEnergyU:
    """almo_split.DirectionalEnergy per spin: fragment X's spin-s occupied orbitals rotate into the spin-s virtuals of
    the fragments ``allow[X]`` lists (preconditioner sqrt(eps_a - eps_i), floored at 0.05 Ha)."""

    def __init__(self, se, mfA, mfB, nA, allow):
        self.se, self.allow = se, allow
        nao = se.nao
        self.occ, self.virt, self.eo, self.ev = {}, {}, {}, {}
        for name, mf, rows in (("A", mfA, slice(0, nA)), ("B", mfB, slice(nA, nao))):
            for s, (C, occ, e) in enumerate(_spin_sets(mf)):
                o = occ > 0
                C0 = np.zeros((nao, int(o.sum())))
                V = np.zeros((nao, int((~o).sum())))
                C0[rows], V[rows] = C[:, o], C[:, ~o]
                self.occ[name, s], self.virt[name, s] = C0, V
                self.eo[name, s], self.ev[name, s] = e[o], e[~o]
        self.blocks = [(X, Y, s) for s in (0, 1) for X in ("A", "B") for Y in allow[X]]
        self.d = {(X, Y, s): np.sqrt(np.maximum(self.ev[Y, s][:, None] - self.eo[X, s][None, :], 0.05))
                  for X, Y, s in self.blocks}

    def x0(self):
        return np.zeros(sum(self.d[b].size for b in self.blocks))

    def orbitals(self, x):
        C = {(X, s): self.occ[X, s].copy() for X in ("A", "B") for s in (0, 1)}
        k = 0
        for X, Y, s in self.blocks:
            d = self.d[X, Y, s]
            C[X, s] = C[X, s] + self.virt[Y, s] @ (x[k:k + d.size].reshape(d.shape) / d)
            k += d.size
        return [np.hstack([C["A", s], C["B", s]]) for s in (0, 1)]

    def __call__(self, x):
        C = self.orbitals(x)
        E, G = self.se(C)
        g = []
        for X, Y, s in self.blocks:
            nA_occ = self.occ["A", s].shape[1]
            cols = slice(0, nA_occ) if X == "A" else slice(nA_occ, C[s].shape[1])
            g.append(((self.virt[Y, s].T @ G[s][:, cols]) / self.d[X, Y, s]).ravel())
        return E, np.concatenate(g)


def directional_split_open(mfAB, mfA, mfB, nA, label=""):
    se = SpanEnergyU(mfAB)
    out = {}
    for key, allow in (("ALMO", {"A": ("A",), "B": ("B",)}), ("A->B", {"A": ("A", "B"), "B": ("B",)}),
                       ("B->A", {"A": ("A",), "B": ("B", "A")}), ("all", {"A": ("A", "B"), "B": ("B", "A")})):
        _, e, res = AS.minimise_rot(DirectionalEnergyU(se, mfA, mfB, nA, allow), label=f"{label} {key}")
        out["E_" + key], out["converged_" + key] = e, bool(res.success)
    out["CT_A->B"] = (out["E_A->B"] - out["E_ALMO"]) * HARTREE
    out["CT_B->A"] = (out["E_B->A"] - out["E_ALMO"]) * HARTREE
    out["CT_SCF"] = (float(mfAB.e_tot) - out["E_ALMO"]) * HARTREE
    out["rest"] = out["CT_SCF"] - out["CT_A->B"] - out["CT_B->A"]
    out["d3_all_minus_SCF"] = (out["E_all"] - float(mfAB.e_tot)) * HARTREE
    return out


def check():
    """(u1)-(u3), PBE/def2-SVP."""
    from pyscf import dft, gto
    donor = [["O", (0.0, 0.0, 0.0)], ["H", (0.957, 0.0, 0.0)], ["H", (-0.240, 0.927, 0.0)]]
    acceptor = [["O", (2.91, 0.0, 0.0)], ["H", (3.25, 0.76, 0.48)], ["H", (3.25, -0.76, 0.48)]]

    def ks(atoms, spin, restricted):
        m = gto.M(atom=atoms, basis="def2-svp", unit="Angstrom", spin=spin, verbose=0)
        mf = dft.RKS(m) if restricted else dft.UKS(m)
        mf.xc, mf.conv_tol = "pbe", 1e-12
        mf.kernel()
        return mf

    rA, rB, rAB = ks(donor, 0, True), ks(acceptor, 0, True), ks(donor + acceptor, 0, True)
    uA, uB, uAB = ks(donor, 0, False), ks(acceptor, 0, False), ks(donor + acceptor, 0, False)
    nA = rA.mol.nao_nr()
    r = AS.directional_split(rAB, rA, rB, nA, label=" water RKS")
    u = directional_split_open(uAB, uA, uB, nA, label=" water UKS")
    d_almo = abs(r["E_ALMO"] - u["E_ALMO"]) * HARTREE
    d_ab, d_ba = abs(r["CT_A->B"] - u["CT_A->B"]), abs(r["CT_B->A"] - u["CT_B->A"])
    worst1 = max(d_almo, d_ab, d_ba)
    print(f"(u1) closed-shell limit: |dE_ALMO| {d_almo:.1e}, |dCT A->B| {d_ab:.1e}, |dCT B->A| {d_ba:.1e} eV -> "
          f"{'PASS' if worst1 < 1e-6 else 'FAIL'}", flush=True)
    # an open-shell pair: the OH radical (doublet) hydrogen-bonded to water
    oh = [["O", (0.0, 0.0, 0.0)], ["H", (0.98, 0.0, 0.0)]]
    water = [["O", (2.85, 0.0, 0.0)], ["H", (3.20, 0.77, 0.45)], ["H", (3.20, -0.77, 0.45)]]
    oA, oB, oAB = ks(oh, 1, False), ks(water, 0, True), ks(oh + water, 1, False)
    nA = oA.mol.nao_nr()
    full = DirectionalEnergyU(SpanEnergyU(oAB), oA, oB, nA, {"A": ("A", "B"), "B": ("B", "A")})
    rng = np.random.default_rng(5)
    y = 1e-2 * rng.standard_normal(full.x0().size)
    _, g = full(y)
    worst2 = 0.0
    for _ in range(3):
        v = rng.standard_normal(y.size)
        v /= np.linalg.norm(v)
        fd = (full(y + 1e-4 * v)[0] - full(y - 1e-4 * v)[0]) / 2e-4
        worst2 = max(worst2, abs(fd - g @ v) / max(abs(g @ v), 1e-12))
    print(f"(u2) open-shell gradient: relative {worst2:.1e} -> {'PASS' if worst2 < 1e-5 else 'FAIL'}", flush=True)
    o = directional_split_open(oAB, oA, oB, nA, label=" OH-water")
    print(f"(u3) open-shell E[all] - E_SCF {o['d3_all_minus_SCF'] * 1000:+.4f} meV -> "
          f"{'PASS' if abs(o['d3_all_minus_SCF']) < 1e-5 else 'FAIL'}; CT OH->water {o['CT_A->B'] * 1000:+.2f}, "
          f"water->OH {o['CT_B->A'] * 1000:+.2f}, total {o['CT_SCF'] * 1000:+.2f} meV", flush=True)


def run_direction_doped(name, au, sites, lattice, tips):
    """A = Au2- (own basis), B = the cluster with its TIPs (own basis, the dimer's AO order: A first, then the
    surface, then the TIPs); the ghost monomers for each direction's BSSE."""
    from pyscf import dft, gto
    surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
    qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf)
    A_atoms = [("Au", p) for p in au]

    def mol(atoms, charge, spin, ghosts=(), with_tips=False):
        spec = [[sym, tuple(p)] for sym, p in atoms] + [["ghost-" + sym, tuple(p)] for sym, p in ghosts]
        basis = {sym: W.BASIS for sym, _ in list(atoms)} | {"ghost-" + sym: W.BASIS for sym, _ in ghosts}
        ecp = {"Au": W.BASIS} if any(sym == "Au" for sym, _ in atoms) else {}
        if with_tips and len(tips):
            spec += [["Mg1", tuple(t)] for t in tips]
            basis["Mg1"] = W.TIP_BASIS
            ecp["Mg1"] = "stuttgart"
            charge += 2 * len(tips)
        return gto.M(atom=spec, basis=basis, ecp=ecp, charge=charge, spin=spin, unit="Angstrom", verbose=0)

    dimer = mol(A_atoms + surf, qB - 1, 1, with_tips=True)
    grids = dft.gen_grid.Grids(dimer)
    grids.level = 3
    grids.build()
    molA_own = mol(A_atoms, -1, 1)
    molB_own = mol(surf, qB, 0, with_tips=True)
    nA = molA_own.nao_nr()
    assert nA + molB_own.nao_nr() == dimer.nao_nr(), "the fragments' own bases must tile the dimer's"
    mfA, eA = W._solve(W._uks(molA_own, AS._grid_like(molA_own, grids)))
    mfB, eB = W._solve(W._rks(molB_own, AS._grid_like(molB_own, grids), lattice))
    _, eAg = W._solve(W._uks(mol(A_atoms, -1, 1, ghosts=surf), grids))
    _, eBg = W._solve(W._rks(W._mol_order_B(surf, qB, A_atoms, tips), grids, lattice))
    mfAB, eAB = W._solve(W._uks(dimer, grids, lattice))
    S = dimer.intor_symmetric("int1e_ovlp")
    sl = dimer.aoslice_by_atom()
    dm = mfAB.make_rdm1()
    spin_m = dm[0] - dm[1]
    spin_au = sum(float(np.einsum("ij,ji->", spin_m[:, sl[a][2]:sl[a][3]], S[sl[a][2]:sl[a][3], :]))
                  for a in range(len(A_atoms)))
    print(f"doped {name}: SCF spin on Au2 {spin_au:+.3f} (gate > 0.8: {'PASS' if spin_au > 0.8 else 'FAIL'}); "
          f"dE_scf {(eAB - eA - eB) * HARTREE:+.4f} eV", flush=True)
    out = directional_split_open(mfAB, mfA, mfB, nA, label=f" doped {name}")
    out["spin_Au2"] = spin_au
    out["dE_scf_own"] = (eAB - eA - eB) * HARTREE
    out["BSSE_A"], out["BSSE_B"] = (eA - eAg) * HARTREE, (eB - eBg) * HARTREE
    out["CT_cp_Au->surface"] = out["CT_A->B"] + out["BSSE_A"]
    out["CT_cp_surface->Au"] = out["CT_B->A"] + out["BSSE_B"]
    out["CT_cp"] = out["CT_SCF"] + out["BSSE_A"] + out["BSSE_B"]
    print(f"doped {name}: CT_cp {out['CT_cp']:+.4f} eV = Au->surface {out['CT_cp_Au->surface']:+.4f} (raw "
          f"{out['CT_A->B']:+.4f}, BSSE_A {out['BSSE_A']:+.4f}) + surface->Au {out['CT_cp_surface->Au']:+.4f} (raw "
          f"{out['CT_B->A']:+.4f}, BSSE_B {out['BSSE_B']:+.4f}) + rest {out['rest']:+.4f}; (u4) E[all] - E_SCF "
          f"{out['d3_all_minus_SCF'] * 1000:+.3f} meV {'PASS' if abs(out['d3_all_minus_SCF']) < 1e-3 else 'FAIL'}; "
          f"converged {[k[10:] for k in out if k.startswith('converged_') and out[k]]}", flush=True)
    for k in [k for k in out if k.startswith("E_")]:
        out[k] = out[k] * HARTREE
    return out


def main(out_dir, r_au=3.6):
    os.makedirs(out_dir, exist_ok=True)
    G = W.geometries()
    sites = W.quantum_sites(G["flat"] + G["upright"], r_au=r_au)
    surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
    res = {}
    path = os.path.join(out_dir, f"almo_direction_doped_tip_r{r_au:.1f}.json")
    for name in ("upright", "flat"):
        lattice, _ = W.doped_lattice(sites, name)
        tips, lattice = W.tip_capped(surf, lattice)
        res[name] = run_direction_doped(name, G[name], sites, lattice, tips)
        json.dump(res, open(path, "w"), indent=1)
    f, u = res["flat"], res["upright"]
    for k in ("CT_cp_Au->surface", "CT_cp_surface->Au", "rest", "CT_cp"):
        print(f"  wet_d share {k:18s} {(f[k] - u[k]) * 1000:+8.1f} meV", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        check()
    else:
        main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 3.6)
