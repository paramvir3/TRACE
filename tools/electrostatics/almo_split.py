"""The O-top relaxation split into polarisation and charge transfer (ALMO-style; 2026-10-24).

    OMP_NUM_THREADS=4 nice python tools/electrostatics/almo_split.py OUT_DIR [R_AU]
    (python: an interpreter with PySCF installed)

Why: the converged first-principles wetting decomposition (wetting_eda.py, 28/34 ions) puts the whole upright
preference of Au2 on MgO(100) in the electronic relaxation at the O-top contact (+1.24 eV of the +0.98 eV wetting).
TRACE carries relaxation by two terms: polarisation by the bound sector's induction (-1/2 s^T K^-1 s) and charge
transfer by the configuration ensemble.  This splits the PBE relaxation the same way.

Method (ESTABLISHED: absolutely localised molecular orbitals, Khaliullin, Cobar, Lochan, Bell and Head-Gordon,
J. Phys. Chem. A 111, 8753 (2007); SCF-MI, Stoll, Wagenblast and Preuss, Theor. Chim. Acta 57, 169 (1980)):
  FRZ  = E_AB[P_frz] - E_A - E_B           monomers SCF in their own basis (no ghosts), antisymmetrised product
  POL  = E_ALMO - E_AB[P_frz]              fragments' orbitals relaxed in the full PBE Hamiltonian, each expanded in its
                                           own atomic functions only (block-diagonal coefficients)
  CT   = E_SCF - E_ALMO                    the delocalisation between the fragments (contains the BSSE)
  BSSE = (E_A - E_A^ghost) + (E_B - E_B^ghost) >= 0,   CT_cp = CT + BSSE
  so FRZ + POL + CT_cp = Delta E_SCF(counterpoise), the interaction of wetting_eda.py.  (Erratum 2026-10-24: the first
  version had CT_cp = CT - BSSE; check (a3) failed by exactly 2 BSSE and caught it.  The ghost functions lower the
  monomers, so the uncorrected interaction is too negative by BSSE and the corrected transfer is CT + BSSE.)
Also POL_A (only Au2 relaxes) and POL_B (only the surface relaxes).
The minimisation: P = C (C^T S C)^-1 C^T, D = 2P, dE/dC = 4 (1 - S P) F C (C^T S C)^-1 (derived; F the Kohn-Sham
matrix at D), restricted to each fragment's own block; L-BFGS from the frozen monomers.

Pre-registered checks (before any piece is read): (a1) the gradient against central finite differences, relative 1e-5
along random block directions; (a2) E_FRZ >= E_ALMO >= E_SCF to 1e-6 Ha; (a3) FRZ + POL + CT_cp = Delta E_SCF(cp) to
1e-6 eV (a bookkeeping identity, true by construction: it guards the arithmetic only; (a1) and (a2) are the
tests); (a4) POL_A + POL_B against POL reported.  All energies on the dimer's grid points.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cluster_interaction as CI  # noqa: E402
import wetting_eda as W  # noqa: E402

HARTREE = W.HARTREE


class BlockEnergy:
    """E(C) and its block gradient for block-diagonal occupied coefficients C = diag(C_A, C_B) in the dimer basis."""

    def __init__(self, mf, nA, CA, CB, relax=("A", "B")):
        self.mf, self.nA = mf, nA
        self.mol = mf.mol
        self.S = self.mol.intor_symmetric("int1e_ovlp")
        self.h = mf.get_hcore()
        self.CA, self.CB = CA.copy(), CB.copy()
        self.relax = relax
        self.nao = self.mol.nao_nr()
        self.calls = 0

    def unpack(self, x):
        CA, CB = self.CA, self.CB
        k = 0
        if "A" in self.relax:
            CA = x[k:k + CA.size].reshape(CA.shape)
            k += CA.size
        if "B" in self.relax:
            CB = x[k:k + CB.size].reshape(CB.shape)
        return CA, CB

    def pack(self, CA, CB):
        parts = []
        if "A" in self.relax:
            parts.append(CA.ravel())
        if "B" in self.relax:
            parts.append(CB.ravel())
        return np.concatenate(parts)

    def full(self, CA, CB):
        C = np.zeros((self.nao, CA.shape[1] + CB.shape[1]))
        C[: self.nA, : CA.shape[1]] = CA
        C[self.nA:, CA.shape[1]:] = CB
        return C

    def __call__(self, x):
        CA, CB = self.unpack(x)
        C = self.full(CA, CB)
        M = np.linalg.inv(C.T @ self.S @ C)
        P = C @ M @ C.T
        D = 2.0 * P
        veff = self.mf.get_veff(self.mol, D)
        E = float(self.mf.energy_tot(dm=D, h1e=self.h, vhf=veff))
        F = self.h + veff
        G = 4.0 * (np.eye(self.nao) - self.S @ P) @ F @ C @ M
        gA = G[: self.nA, : CA.shape[1]]
        gB = G[self.nA:, CA.shape[1]:]
        self.calls += 1
        return E, self.pack(gA, gB)


class RotationEnergy:
    """The same energy in preconditioned occupied-virtual rotations inside each fragment's own basis (2026-10-24; the
    raw-coefficient L-BFGS crawled: 400 iterations gave POL -0.636 eV, |g| 3.5e-2 at 100): C_X = C0_X + V_X K_X,
    V_X the fragment's virtual orbitals (its own basis, so the constraint is unchanged), K = K' / sqrt(eps_a - eps_i)
    (the monomer's orbital-energy gaps, floored at 0.05 Ha).  No occupied-occupied (gauge) direction remains; the
    gradient is dE/dK'_X = (V_X^T dE/dC_X) / sqrt(eps_a - eps_i), from BlockEnergy's validated dE/dC."""

    def __init__(self, be, mfA, mfB, relax=("A", "B")):
        self.be, self.relax = be, relax
        self.frag = {}
        for name, mf in (("A", mfA), ("B", mfB)):
            occ = mf.mo_occ > 0
            C0, V = mf.mo_coeff[:, occ], mf.mo_coeff[:, ~occ]
            eo, ev = mf.mo_energy[occ], mf.mo_energy[~occ]
            d = np.sqrt(np.maximum(ev[:, None] - eo[None, :], 0.05))
            self.frag[name] = (C0, V, d)

    def blocks(self, x):
        out, k = {}, 0
        for name in ("A", "B"):
            C0, V, d = self.frag[name]
            if name in self.relax:
                Kp = x[k:k + d.size].reshape(d.shape)
                k += d.size
                out[name] = C0 + V @ (Kp / d)
            else:
                out[name] = C0
        return out

    def x0(self):
        return np.concatenate([np.zeros(self.frag[n][2].size) for n in ("A", "B") if n in self.relax])

    def __call__(self, x):
        c = self.blocks(x)
        be = self.be
        save = be.relax
        be.relax = ("A", "B")
        E, g = be(be.pack(c["A"], c["B"]))
        be.relax = save
        nA = c["A"].size
        gA, gB = g[:nA].reshape(c["A"].shape), g[nA:].reshape(c["B"].shape)
        parts = []
        for name, gC in (("A", gA), ("B", gB)):
            if name in self.relax:
                C0, V, d = self.frag[name]
                parts.append(((V.T @ gC) / d).ravel())
        return E, np.concatenate(parts)


def minimise_rot(re, tol=1e-6, maxiter=2000, label=""):
    from scipy.optimize import minimize
    x0 = re.x0()
    e0, _ = re(x0)
    hist = {"it": 0}

    def cb(xk):
        hist["it"] += 1
        if hist["it"] % 10 == 0:
            e, g = re(xk)
            print(f"    almo-rot{label} it {hist['it']}: E - E_frz {(e - e0) * HARTREE:+.6f} eV, |g|_max {np.abs(g).max():.2e}",
                  flush=True)

    res = minimize(re, x0, jac=True, method="L-BFGS-B", callback=cb,
                   options={"maxiter": maxiter, "gtol": tol, "ftol": 1e-15, "maxcor": 30})
    return e0, float(res.fun), res


def minimise(be, tol=1e-6, maxiter=4000, label=""):
    """L-BFGS over the block coefficients.  The energy is logged every 25 iterations (2026-10-24: 400 iterations did
    not converge the 10-ion upright case; the log measures the convergence instead of assuming it)."""
    from scipy.optimize import minimize
    x0 = be.pack(be.CA, be.CB)
    e0, _ = be(x0)
    hist = {"it": 0}

    def cb(xk):
        hist["it"] += 1
        if hist["it"] % 25 == 0:
            e, g = be(xk)
            print(f"    almo{label} it {hist['it']}: E {e:.10f} Ha, E - E_frz {(e - e0) * HARTREE:+.6f} eV, "
                  f"|g|_max {np.abs(g).max():.2e}", flush=True)

    res = minimize(be, x0, jac=True, method="L-BFGS-B", callback=cb,
                   options={"maxiter": maxiter, "gtol": tol, "ftol": 1e-14, "maxcor": 30})
    return e0, float(res.fun), res


def gradient_check(be, n_dir=2, h=1e-4, seed=7):
    rng = np.random.default_rng(seed)
    x0 = be.pack(be.CA, be.CB)
    e0, g = be(x0)
    worst = 0.0
    for _ in range(n_dir):
        v = rng.standard_normal(x0.size)
        v /= np.linalg.norm(v)
        ep, _ = be(x0 + h * v)
        em, _ = be(x0 - h * v)
        fd = (ep - em) / (2 * h)
        an = float(g @ v)
        worst = max(worst, abs(fd - an) / max(abs(an), 1e-12))
    return worst


# ---- the direction of the charge transfer (added 2026-10-27) ---------------------------------------------------------
#
# Why: the model's relaxation at the O-top contact is a tenth of PBE's (probe_t5_first_order.py, step 17), and PBE's
# ALMO split puts 62 % of the upright preference in charge transfer.  TRACE carries transfer in two channels with
# different physics: O -> Au (Au- + an O hole) and Au -> surface (Au+ + a band electron).  Which one PBE uses decides
# which channel must carry it.  Method (ESTABLISHED in substance: the one-directional delocalisation of absolutely
# localised orbitals, Khaliullin, Bell and Head-Gordon, J. Chem. Phys. 128, 184112 (2008)): variational, the occupied
# orbitals of fragment X are allowed to mix with the virtual orbitals of the fragments listed for it, and no others:
#   E[A->B] = min over {A occ -> A, B virt; B occ -> B virt},  CT(A->B) = E[A->B] - E_ALMO,  and likewise B->A;
#   E[all] (both directions) must reach E_SCF (check d3), and CT - CT(A->B) - CT(B->A) is the non-additive rest.

class SpanEnergy:
    """E(C) for full-row occupied coefficients C [nao, nocc] (closed shell, D = 2 C (C^T S C)^-1 C^T) and its gradient
    dE/dC = 4 (1 - S P) F C (C^T S C)^-1 (BlockEnergy's, all rows kept)."""

    def __init__(self, mf):
        self.mf, self.mol = mf, mf.mol
        self.S = self.mol.intor_symmetric("int1e_ovlp")
        self.h = mf.get_hcore()
        self.nao = self.mol.nao_nr()

    def __call__(self, C):
        M = np.linalg.inv(C.T @ self.S @ C)
        P = C @ M @ C.T
        veff = self.mf.get_veff(self.mol, 2.0 * P)
        E = float(self.mf.energy_tot(dm=2.0 * P, h1e=self.h, vhf=veff))
        G = 4.0 * (np.eye(self.nao) - self.S @ P) @ (self.h + veff) @ C @ M
        return E, G


class DirectionalEnergy:
    """The energy in preconditioned rotations of each fragment's occupied orbitals into the virtual orbitals of the
    fragments ``allow[X]`` lists (each fragment's own virtuals, embedded in the dimer basis; preconditioner
    sqrt(eps_a - eps_i) from the monomers' orbital energies, floored at 0.05 Ha).  allow = {A: (A,), B: (B,)} is
    RotationEnergy's ALMO energy."""

    def __init__(self, se, mfA, mfB, nA, allow):
        self.se, self.allow = se, allow
        nao = se.nao
        self.occ, self.virt, self.eo, self.ev = {}, {}, {}, {}
        for name, mf, rows in (("A", mfA, slice(0, nA)), ("B", mfB, slice(nA, nao))):
            occ = mf.mo_occ > 0
            C0 = np.zeros((nao, int(occ.sum())))
            V = np.zeros((nao, int((~occ).sum())))
            C0[rows], V[rows] = mf.mo_coeff[:, occ], mf.mo_coeff[:, ~occ]
            self.occ[name], self.virt[name] = C0, V
            self.eo[name], self.ev[name] = mf.mo_energy[occ], mf.mo_energy[~occ]
        self.pairs = [(X, Y) for X in ("A", "B") for Y in allow[X]]
        self.d = {(X, Y): np.sqrt(np.maximum(self.ev[Y][:, None] - self.eo[X][None, :], 0.05)) for X, Y in self.pairs}

    def x0(self):
        return np.zeros(sum(self.d[p].size for p in self.pairs))

    def orbitals(self, x):
        C = {X: self.occ[X].copy() for X in ("A", "B")}
        k = 0
        for X, Y in self.pairs:
            d = self.d[(X, Y)]
            C[X] = C[X] + self.virt[Y] @ (x[k:k + d.size].reshape(d.shape) / d)
            k += d.size
        return np.hstack([C["A"], C["B"]])

    def __call__(self, x):
        C = self.orbitals(x)
        E, G = self.se(C)
        nA_occ = self.occ["A"].shape[1]
        cols = {"A": slice(0, nA_occ), "B": slice(nA_occ, C.shape[1])}
        g = [((self.virt[Y].T @ G[:, cols[X]]) / self.d[(X, Y)]).ravel() for X, Y in self.pairs]
        return E, np.concatenate(g)


def directional_split(mfAB, mfA, mfB, nA, label=""):
    """{E_ALMO, E_A->B, E_B->A, E_all} (Ha) and CT(A->B), CT(B->A), CT, rest (eV); minimise_rot's L-BFGS."""
    se = SpanEnergy(mfAB)
    out = {}
    for key, allow in (("ALMO", {"A": ("A",), "B": ("B",)}), ("A->B", {"A": ("A", "B"), "B": ("B",)}),
                       ("B->A", {"A": ("A",), "B": ("B", "A")}), ("all", {"A": ("A", "B"), "B": ("B", "A")})):
        _, e, res = minimise_rot(DirectionalEnergy(se, mfA, mfB, nA, allow), label=f"{label} {key}")
        out["E_" + key], out["converged_" + key] = e, bool(res.success)
    out["CT_A->B"] = (out["E_A->B"] - out["E_ALMO"]) * HARTREE
    out["CT_B->A"] = (out["E_B->A"] - out["E_ALMO"]) * HARTREE
    out["CT_all"] = (out["E_all"] - out["E_ALMO"]) * HARTREE
    out["CT_SCF"] = (float(mfAB.e_tot) - out["E_ALMO"]) * HARTREE
    out["rest"] = out["CT_SCF"] - out["CT_A->B"] - out["CT_B->A"]
    out["d3_all_minus_SCF"] = (out["E_all"] - float(mfAB.e_tot)) * HARTREE
    return out


def check_direction():
    """(d1) DirectionalEnergy with ALMO's allow equals RotationEnergy at random x (1e-10 Ha); (d2) its gradient against
    central differences (relative 1e-5, h 1e-4) with both directions open; (d3) both directions reach E_SCF (1e-5 eV);
    (d4) the water dimer's transfer runs from the H-bond acceptor to the donor (ESTABLISHED: Khaliullin, Bell and
    Head-Gordon 2008, about 90 % one way).  PBE/def2-SVP, the water dimer near its minimum (O-O 2.91 A)."""
    from pyscf import dft, gto
    donor = [["O", (0.0, 0.0, 0.0)], ["H", (0.957, 0.0, 0.0)], ["H", (-0.240, 0.927, 0.0)]]
    acceptor = [["O", (2.91, 0.0, 0.0)], ["H", (3.25, 0.76, 0.48)], ["H", (3.25, -0.76, 0.48)]]

    def rks(atoms):
        m = gto.M(atom=atoms, basis="def2-svp", unit="Angstrom", verbose=0)
        mf = dft.RKS(m)
        mf.xc, mf.conv_tol = "pbe", 1e-11
        mf.kernel()
        return mf

    mfA, mfB, mfAB = rks(donor), rks(acceptor), rks(donor + acceptor)
    nA = mfA.mol.nao_nr()
    be = BlockEnergy(mfAB, nA, mfA.mo_coeff[:, mfA.mo_occ > 0], mfB.mo_coeff[:, mfB.mo_occ > 0])
    re = RotationEnergy(be, mfA, mfB)
    de = DirectionalEnergy(SpanEnergy(mfAB), mfA, mfB, nA, {"A": ("A",), "B": ("B",)})
    rng = np.random.default_rng(3)
    x = 1e-2 * rng.standard_normal(re.x0().size)
    d1 = abs(re(x)[0] - de(x)[0])
    full = DirectionalEnergy(SpanEnergy(mfAB), mfA, mfB, nA, {"A": ("A", "B"), "B": ("B", "A")})
    y = 1e-2 * rng.standard_normal(full.x0().size)
    e0, g = full(y)
    worst = 0.0
    for _ in range(3):
        v = rng.standard_normal(y.size)
        v /= np.linalg.norm(v)
        fd = (full(y + 1e-4 * v)[0] - full(y - 1e-4 * v)[0]) / 2e-4
        worst = max(worst, abs(fd - g @ v) / max(abs(g @ v), 1e-12))
    out = directional_split(mfAB, mfA, mfB, nA, label=" water")
    print(f"(d1) ALMO energies agree to {d1:.1e} Ha {'PASS' if d1 < 1e-10 else 'FAIL'}; (d2) gradient {worst:.1e} "
          f"{'PASS' if worst < 1e-5 else 'FAIL'}; (d3) E[all] - E_SCF {out['d3_all_minus_SCF'] * 1000:+.4f} meV "
          f"{'PASS' if abs(out['d3_all_minus_SCF']) < 1e-5 else 'FAIL'}", flush=True)
    a2b, b2a = out["CT_A->B"], out["CT_B->A"]
    print(f"(d4) water dimer: CT donor->acceptor {a2b * 1000:+.2f}, acceptor->donor {b2a * 1000:+.2f}, total "
          f"{out['CT_SCF'] * 1000:+.2f}, rest {out['rest'] * 1000:+.2f} meV -> "
          f"{'PASS' if abs(b2a) > 3 * abs(a2b) else 'FAIL'} (acceptor -> donor dominant)", flush=True)


def run_direction(name, au, sites, lattice):
    """The directional split for one geometry of the wetting cluster (A = Au2, B = the surface), with the monomers
    in their own basis on the dimer grid (as run_geometry) and the ghost monomers for each direction's BSSE: A's
    occupied orbitals in B's functions lower A alone by BSSE_A = E_A - E_A(ghost B), so CT_cp(A->B) = CT(A->B) +
    BSSE_A and CT_cp(B->A) = CT(B->A) + BSSE_B (the attribution of the counterpoise correction, stated)."""
    from pyscf import dft
    surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
    qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf)
    A_atoms = [("Au", p) for p in au]
    dimer = W._mol(A_atoms + surf, qB)
    grids = dft.gen_grid.Grids(dimer)
    grids.level = 3
    grids.build()
    molA_own, molB_own = W._mol(A_atoms, 0), W._mol(surf, qB)
    nA = molA_own.nao_nr()
    mfA, eA = W._solve(W._rks(molA_own, _grid_like(molA_own, grids)))
    mfB, eB = W._solve(W._rks(molB_own, _grid_like(molB_own, grids), lattice))
    _, eAg = W._solve(W._rks(W._mol(A_atoms, 0, ghosts=surf), grids))
    _, eBg = W._solve(W._rks(W._mol_order_B(surf, qB, A_atoms), grids, lattice))
    mfAB, eAB = W._solve(W._rks(dimer, grids, lattice))
    out = directional_split(mfAB, mfA, mfB, nA, label=f" {name}")
    out["BSSE_A"], out["BSSE_B"] = (eA - eAg) * HARTREE, (eB - eBg) * HARTREE
    out["CT_cp_Au->surface"] = out["CT_A->B"] + out["BSSE_A"]
    out["CT_cp_surface->Au"] = out["CT_B->A"] + out["BSSE_B"]
    out["CT_cp"] = out["CT_SCF"] + out["BSSE_A"] + out["BSSE_B"]
    print(f"{name}: CT_cp {out['CT_cp']:+.4f} eV = Au->surface {out['CT_cp_Au->surface']:+.4f} (raw {out['CT_A->B']:+.4f}, "
          f"BSSE_A {out['BSSE_A']:+.4f}) + surface->Au {out['CT_cp_surface->Au']:+.4f} (raw {out['CT_B->A']:+.4f}, BSSE_B "
          f"{out['BSSE_B']:+.4f}) + rest {out['rest']:+.4f}; (d3) E[all] - E_SCF {out['d3_all_minus_SCF'] * 1000:+.3f} meV "
          f"{'PASS' if abs(out['d3_all_minus_SCF']) < 1e-3 else 'FAIL'}; converged "
          f"{[k[10:] for k in out if k.startswith('converged_') and out[k]]}", flush=True)
    for k in [k for k in out if k.startswith("E_")]:
        out[k] = out[k] * HARTREE
    return out


def main_direction(out_dir, r_au=3.6):
    os.makedirs(out_dir, exist_ok=True)
    G = W.geometries()
    sites = W.quantum_sites(G["flat"] + G["upright"], r_au=r_au)
    lattice = CI.rocksalt_lattice(W.D, 5, True, exclude=sites)
    res = {}
    path = os.path.join(out_dir, f"almo_direction_r{r_au:.1f}.json")
    for name in ("upright", "flat"):
        res[name] = run_direction(name, G[name], sites, lattice)
        json.dump(res, open(path, "w"), indent=1)
    f, u = res["flat"], res["upright"]
    for k in ("CT_cp_Au->surface", "CT_cp_surface->Au", "rest", "CT_cp"):
        print(f"  wet share {k:18s} {(f[k] - u[k]) * 1000:+8.1f} meV", flush=True)


def run_geometry(name, au, sites, lattice):
    from pyscf import dft
    surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
    qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf)
    A_atoms = [("Au", p) for p in au]
    dimer = W._mol(A_atoms + surf, qB)
    grids = dft.gen_grid.Grids(dimer)
    grids.level = 3
    grids.build()
    molA_own = W._mol(A_atoms, 0)
    molB_own = W._mol(surf, qB)
    nA = molA_own.nao_nr()
    # monomers in their own basis, on the dimer's grid points (each molecule's own screening mask), so no grid
    # difference enters FRZ or the BSSE
    mfA, eA = W._solve(W._rks(molA_own, _grid_like(molA_own, grids)))
    mfB, eB = W._solve(W._rks(molB_own, _grid_like(molB_own, grids), lattice))
    # the same monomers in the dimer basis (ghosts), on the dimer grid, for the BSSE and the counterpoise interaction
    mfAg, eAg = W._solve(W._rks(W._mol(A_atoms, 0, ghosts=surf), grids))
    mfBg, eBg = W._solve(W._rks(W._mol_order_B(surf, qB, A_atoms), grids, lattice))
    mfAB, eAB = W._solve(W._rks(dimer, grids, lattice))
    CA = mfA.mo_coeff[:, mfA.mo_occ > 0]
    CB = mfB.mo_coeff[:, mfB.mo_occ > 0]
    out = {"nao": int(dimer.nao_nr()), "E_SCF": eAB * HARTREE}
    be = BlockEnergy(mfAB, nA, CA, CB, relax=("A", "B"))
    out["a1_grad_rel_err"] = gradient_check(be, h=3e-4)            # the step scan's optimum (truncation vs round-off)
    # the preconditioned rotation solver (2026-10-24: 19 iterations to |g| < 1e-6 where raw coefficients took > 400)
    e_frz, e_almo, res = minimise_rot(RotationEnergy(be, mfA, mfB, relax=("A", "B")), label=f" {name} AB")
    out["E_FRZ"], out["E_ALMO"], out["almo_iterations"], out["almo_converged"] = e_frz * HARTREE, e_almo * HARTREE, \
        int(res.nit), bool(res.success)
    for part in ("A", "B"):
        _, e_p, r_p = minimise_rot(RotationEnergy(be, mfA, mfB, relax=(part,)), label=f" {name} {part}")
        out[f"E_ALMO_{part}"] = e_p * HARTREE
        out[f"almo_{part}_converged"] = bool(r_p.success)
    out["FRZ"] = (e_frz - eA - eB) * HARTREE
    out["POL"] = (e_almo - e_frz) * HARTREE
    out["POL_A"] = out["E_ALMO_A"] - out["E_FRZ"]
    out["POL_B"] = out["E_ALMO_B"] - out["E_FRZ"]
    out["CT"] = (eAB - e_almo) * HARTREE
    out["BSSE"] = ((eA - eAg) + (eB - eBg)) * HARTREE
    out["CT_cp"] = out["CT"] + out["BSSE"]
    out["dE_cp"] = (eAB - eAg - eBg) * HARTREE
    out["a3_identity"] = out["FRZ"] + out["POL"] + out["CT_cp"] - out["dE_cp"]
    out["a2_order"] = bool(e_frz >= e_almo - 1e-6 and e_almo >= eAB - 1e-6)
    print(f"{name}: dE_cp {out['dE_cp']:+.4f} eV = FRZ {out['FRZ']:+.4f} + POL {out['POL']:+.4f} (A {out['POL_A']:+.4f}, "
          f"B {out['POL_B']:+.4f}) + CT_cp {out['CT_cp']:+.4f} (CT {out['CT']:+.4f}, BSSE {out['BSSE']:+.4f}); "
          f"(a1) grad {out['a1_grad_rel_err']:.1e} {'PASS' if out['a1_grad_rel_err'] < 1e-4 else 'FAIL'} (h 3e-4; below it the energy round-off dominates); (a2) "
          f"{'PASS' if out['a2_order'] else 'FAIL'}; (a3) {out['a3_identity']:+.1e} eV "
          f"{'PASS' if abs(out['a3_identity']) < 1e-6 else 'FAIL'}; ALMO {out['almo_iterations']} it "
          f"({'converged' if out['almo_converged'] else 'NOT converged'})", flush=True)
    return out


def _grid_like(mol, grids):
    """The points and weights of ``grids`` with the screening mask of ``mol`` (whose basis is a subset)."""
    from pyscf import dft
    g = dft.gen_grid.Grids(mol)
    g.level = grids.level
    g.coords, g.weights = grids.coords, grids.weights
    g.non0tab = g.make_mask(mol, g.coords)
    return g


def main(out_dir, r_au=3.6):
    os.makedirs(out_dir, exist_ok=True)
    G = W.geometries()
    sites = W.quantum_sites(G["flat"] + G["upright"], r_au=r_au)
    lattice = CI.rocksalt_lattice(W.D, 5, True, exclude=sites)
    res = {}
    path = os.path.join(out_dir, f"almo_split_r{r_au:.1f}.json")
    for name in ("upright", "flat"):
        res[name] = run_geometry(name, G[name], sites, lattice)
        json.dump(res, open(path, "w"), indent=1)
    f, u = res["flat"], res["upright"]
    for k in ("FRZ", "POL", "POL_A", "POL_B", "CT_cp", "BSSE", "dE_cp"):
        print(f"  wet share {k:6s} {(f[k] - u[k]) * 1000:+8.1f} meV", flush=True)
    print(f"  (a4) POL_A + POL_B - POL: upright {(u['POL_A'] + u['POL_B'] - u['POL']) * 1000:+.1f}, flat "
          f"{(f['POL_A'] + f['POL_B'] - f['POL']) * 1000:+.1f} meV", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--check-direction":
        check_direction()
    elif sys.argv[1] == "--direction":
        main_direction(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 3.6)
    else:
        main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 3.6)
