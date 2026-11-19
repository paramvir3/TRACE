"""Build trace_redox/data/ion_densities.npz: first-principles charge-state densities and the SPEC 23 pair tables.

Three stages (two Python environments: PySCF has no torch here):

  python tools/electrostatics/build_ion_densities.py spec WORK            # main env: elements, states, frames, I_q
  python tools/electrostatics/build_ion_densities.py densities WORK [SYMBOL ...]   # a python with PySCF installed
  python tools/electrostatics/build_ion_densities.py tables WORK          # main env: tails, pair tables, the npz

Densities: spin-unrestricted PBE0 (ESTABLISHED: Adamo & Barone 1999), X2C spin-free for Z > 36 (ESTABLISHED),
ANO-RCC (Roos et al.) plus two even-tempered diffuse s and p shells (ratio 3.2) for the anion tails; the lowest of the
two lowest spin states; spherically averaged over 302 Fibonacci directions on a 12000-point log grid
(1e-7 .. 45 bohr).  O2- (unbound in vacuum), and with it every O state, sits in a Watson sphere (Phys. Rev. 111, 1108
(1958)) of +2 at R_W = 1.205 A, the radius at which the sphere's potential 2 K / R_W equals the Madelung potential of
rock-salt MgO at the O site (alpha_M = 1.7476, a = 4.212 A).  The pair tables use the cancellation-free difference
route (trace_redox.radial.penetration_difference), grid-converged to 1e-7 eV (2x, 4x refinement).  Mg- (EA < 0) is unbound and left out.  Bound states get the exact
Katriel-Davidson tail is NOT spliced on (see stage_tables): the raw densities are renormalised to their electron counts.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ELEMENTS = {1: "H", 6: "C", 8: "O", 11: "Na", 12: "Mg", 13: "Al", 17: "Cl", 47: "Ag", 79: "Au"}
GRID = np.exp(np.linspace(math.log(1e-7), math.log(45.0), 12000))
#: Every O state in the same Watson sphere: one lattice site's charge states share its Madelung confinement, so their
#: density differences are the carrier alone (a free O- against a confined O2- would carry the confinement too).
WATSON = {(8, q): (1.205, 2.0) for q in range(-2, 4)}
UNBOUND = {(12, -1)}
#: The functional.  Compared on Au- (2026-10-18; same basis, X2C): carrier <r> PBE0 2.236, wB97X 2.250, HF 2.500 A;
#: the penetration Xi(Au-, O2-) at 2.2 A -2.39 / -2.46 / -2.87 eV; dSCF EA 2.00 / 1.98 / 0.62 (exact 2.31);
#: -eps_HOMO(Au-) wB97X 1.90, HF 1.15.  PBE0 and wB97X agree to 3 % in the bulk that sets the penetration at contact;
#: self-interaction mainly spoils the far tail, which the exact C^1 tail replaces.  PBE0 is kept (stated uncertainty
#: ~3 % from the functional, +-9 % from the Watson radius).
METHOD = os.environ.get("ION_DENSITY_METHOD", "pbe0")
HARTREE = 27.211386245988


def stage_spec(work: str) -> None:
    sys.path.insert(0, REPO)
    import torch
    from trace_redox.constants import ATOMIC_SUCCESSIVE_IONIZATION_EV, formal_frame
    from trace_redox.diabatic import formal_ladder
    z = torch.tensor(sorted(ELEMENTS))
    states, E, L, K = formal_ladder(z, 3)
    frames = formal_frame(z)
    spec = []
    first = int(states[0])
    for k, zz in enumerate(z.tolist()):
        for q in range(int(L[k]), int(K[k]) + 1):
            if (zz, q) in UNBOUND:
                continue
            nxt = q + 1
            if nxt <= int(K[k]):
                I = float(E[k, nxt - first] - E[k, q - first])
            else:                                       # beyond the ladder: the next NIST successive IE
                series = ATOMIC_SUCCESSIVE_IONIZATION_EV[zz]
                I = float(series[q]) if q < len(series) and math.isfinite(series[q]) else None
            spec.append(dict(Z=zz, symbol=ELEMENTS[zz], q=q, frame=int(frames[k]), I=I,
                             watson=WATSON.get((zz, q))))
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "spec.json"), "w") as fh:
        json.dump(spec, fh, indent=1)
    print(f"{len(spec)} states written to {work}/spec.json")


def _basis(symbol: str):
    from pyscf import gto
    base = gto.basis.load("ano-rcc", symbol)
    out = list(base)
    for l in (0, 1):
        exps = [prim[0] for shell in base if shell[0] == l for prim in shell[1:]]
        a = min(exps)
        for k in (1, 2):
            out.append([l, [a / 3.2 ** k, 1.0]])
    return out


def stage_densities(work: str, symbols) -> None:
    from pyscf import dft, gto, qmmm
    with open(os.path.join(work, "spec.json")) as fh:
        spec = json.load(fh)
    n_dir = 302
    i = np.arange(n_dir) + 0.5
    phi = np.arccos(1 - 2 * i / n_dir)
    th = np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    for entry in spec:
        if symbols and entry["symbol"] not in symbols:
            continue
        Z, q, sym = entry["Z"], entry["q"], entry["symbol"]
        out_path = os.path.join(work, f"{Z}_{q:+d}.npz")
        if os.path.exists(out_path):
            continue
        n_el = Z - q
        if n_el == 0:
            np.savez(out_path, r=GRID, rho=np.zeros_like(GRID), energy=0.0, spin=0, converged=True)
            print(f"{sym}{q:+d}: bare nucleus")
            continue
        best = None
        for spin in (n_el % 2, n_el % 2 + 2):
            if spin > n_el:
                continue
            mol = gto.M(atom=f"{sym} 0 0 0", basis={sym: _basis(sym)}, charge=q, spin=spin, verbose=0)
            mf = dft.UKS(mol)
            if Z > 36:
                mf = mf.x2c()
            mf.xc = METHOD
            mf.conv_tol = 1e-10
            mf.max_cycle = 300
            mf.level_shift = 0.2
            if entry["watson"] is not None:
                RW, QW = entry["watson"]
                mf = qmmm.mm_charge(mf, dirs * RW, np.full(n_dir, QW / n_dir), radii=np.full(n_dir, 0.3))
            try:
                e = mf.kernel()
                if not mf.converged:
                    mf.level_shift = 0.0
                    mf = mf.newton()
                    e = mf.kernel()
            except Exception as err:                                   # noqa: BLE001
                print(f"  {sym}{q:+d} spin {spin}: {type(err).__name__} {str(err)[:80]}")
                continue
            if mf.converged and (best is None or e < best[0]):
                best = (e, spin, mf, mol)
        if best is None:
            print(f"{sym}{q:+d}: NO converged state")
            continue
        e, spin, mf, mol = best
        dm = mf.make_rdm1()
        dm = dm[0] + dm[1]
        rho = np.zeros_like(GRID)
        for lo in range(0, len(GRID), 500):
            rr = GRID[lo:lo + 500]
            pts = (rr[:, None, None] * dirs[None, :, :]).reshape(-1, 3)
            ao = mol.eval_gto("GTOval", pts)
            rho[lo:lo + 500] = np.einsum("pi,ij,pj->p", ao, dm, ao).reshape(len(rr), n_dir).mean(1)
        occ, en = mf.mo_occ, mf.mo_energy
        homo = max(float(np.max(en[k][occ[k] > 0])) for k in (0, 1) if np.any(occ[k] > 0)) * HARTREE
        np.savez(out_path, r=GRID, rho=rho, energy=e, spin=spin, converged=True, homo=homo, method=METHOD)
        N = float(np.sum(0.5 * (4 * np.pi * GRID[1:] ** 2 * rho[1:] + 4 * np.pi * GRID[:-1] ** 2 * rho[:-1]) * np.diff(GRID)))
        print(f"{sym}{q:+d}: E = {e:.6f} Ha, 2S = {spin}, N(grid) {N:.5f} (expect {n_el}), HOMO {homo:+.3f} eV", flush=True)


def _splice(r, rho, N, I, q):
    """The exact Katriel-Davidson tail (r^2 + kappa^-2)^beta exp(-2 kappa r), matched C^1: beyond the outermost maximum of
    4 pi r^2 rho, at the first radius where the computed decay rate -d ln rho / dr has fallen to the exact one
    2 kappa - 2 beta r / (r^2 + kappa^-2) (value matched there; the rates are equal to the grid's resolution).  An
    over-diffuse density (self-interaction) is corrected from there on, a basis-truncated one where it first steepens
    past the exact rate; with no crossing before 25 bohr nothing is replaced.  Returns (rho, splice radius or None)."""
    if I is None or I <= 0.0 or N <= 0:
        return rho, None
    kappa = math.sqrt(2.0 * I / HARTREE)
    beta = (q + 1) / kappa - 1.0
    shell = 4 * np.pi * r * r * rho
    interior = (r > 0.3) & (r < 25.0)
    idx = np.where(interior)[0]
    peaks = [k for k in idx[1:-1] if shell[k] >= shell[k - 1] and shell[k] >= shell[k + 1]]
    start = peaks[-1] if peaks else idx[0]
    rate = -np.gradient(np.log(np.maximum(rho, 1e-300)), r)
    exact = 2.0 * kappa - 2.0 * beta * r / (r * r + kappa ** -2)
    cand = [k for k in range(start + 1, idx[-1]) if (rate[k - 1] - exact[k - 1]) * (rate[k] - exact[k]) <= 0.0]
    if not cand:
        return rho, None
    m = cand[0]
    tail = (r * r + kappa ** -2) ** beta * np.exp(-2.0 * kappa * r)
    out = rho.copy()
    out[m:] = rho[m] * tail[m:] / tail[m]
    return out, r[m]


def stage_tables(work: str) -> None:
    sys.path.insert(0, REPO)
    from trace_redox.penetration import K_E, N_CHEBYSHEV, R_CUT, R_MIN, BOHR_A, HARTREE_EV
    from trace_redox.radial import RadialDensity, chebyshev_fit, chebyshev_nodes, gordon_kim_table, penetration_difference
    with open(os.path.join(work, "spec.json")) as fh:
        spec = json.load(fh)
    dens, frames, keys_d, report, splices = {}, {}, [], [], {}
    for entry in spec:
        Z, q = entry["Z"], entry["q"]
        path = os.path.join(work, f"{Z}_{q:+d}.npz")
        frames[Z] = entry["frame"]
        if not os.path.exists(path):
            print(f"  missing density {Z} {q:+d}: left out")
            continue
        d = np.load(path)
        N = Z - q
        rho = d["rho"]
        if N > 0:
            rho = rho * (N / max(1e-300, float(np.sum(0.5 * (4 * np.pi * GRID[1:] ** 2 * rho[1:] + 4 * np.pi * GRID[:-1] ** 2 * rho[:-1]) * np.diff(GRID)))))
            r_m = None              # no tail splice (2026-10-18): the C^1 rule was not robust (no crossing for most cations,
            # spurious crossings inside the valence shell of Cl-, Na0, Al0), and the comparison of functionals shows the
            # far tail does not set the penetration at contact (beyond ~5 A it contributes ~1 meV)
            splices[(Z, q)] = -1.0 if r_m is None else float(r_m)
            rho = rho * (N / float(np.sum(0.5 * (4 * np.pi * GRID[1:] ** 2 * rho[1:] + 4 * np.pi * GRID[:-1] ** 2 * rho[:-1]) * np.diff(GRID))))
            dens[(Z, q)] = RadialDensity(GRID, rho)
            report.append(f"{entry['symbol']}{q:+d}: N {dens[(Z, q)].N:.6f}, tail from {r_m if r_m is None else round(r_m * BOHR_A, 3)} A")
        else:
            dens[(Z, q)] = None
            report.append(f"{entry['symbol']}{q:+d}: bare nucleus")
        keys_d.append((Z, q))
    print("\n".join(report))
    nodes_A = chebyshev_nodes(N_CHEBYSHEV, R_MIN, R_CUT)
    check_A = np.linspace(R_MIN, R_CUT, 157)[1:-1]
    # an element is in the library only with its frame density (else the runtime refuses it, never zeroes it)
    elements = sorted({k[0] for k in keys_d if (k[0], frames[k[0]]) in dens})
    dropped = sorted({k[0] for k in keys_d} - set(elements))
    if dropped:
        print(f"  elements without a frame density, left out of the library: {dropped}")
    pair_keys, coeffs, worst = [], [], 0.0

    zero = np.zeros_like(GRID)
    rho_of = lambda Z, q: zero if dens[(Z, q)] is None else dens[(Z, q)].rho

    for ia, Za in enumerate(elements):
        for Zb in elements[ia:]:
            fa, fb = frames[Za], frames[Zb]
            if (Za, fa) not in dens or (Zb, fb) not in dens:
                continue
            for qa in [k[1] for k in keys_d if k[0] == Za]:
                for qb in [k[1] for k in keys_d if k[0] == Zb]:
                    def RD(R_A):
                        D = penetration_difference(GRID, Za, rho_of(Za, fa), rho_of(Za, qa), fa, Zb, rho_of(Zb, fb),
                                                   rho_of(Zb, qb), fb, R_A / BOHR_A) * HARTREE_EV
                        return R_A * D
                    values = np.array([RD(x) for x in nodes_A])
                    c = chebyshev_fit(values, N_CHEBYSHEV)
                    x = (2 * check_A[::13] - (R_CUT + R_MIN)) / (R_CUT - R_MIN)
                    fit = np.polynomial.chebyshev.chebval(x, c)
                    exact = np.array([RD(v) for v in check_A[::13]])
                    worst = max(worst, float(np.abs(fit - exact).max()))
                    pair_keys.append((Za, qa, Zb, qb))
                    coeffs.append(c)
            print(f"  tables {Za}-{Zb} done; worst Chebyshev error so far {worst:.2e} eV A", flush=True)
    # SPEC 25: the Pauli half of the overlap.  G (the Gordon-Kim electron-gas term) is reliable at contact and wrong in
    # the tails (spurious LDA / gradient attraction beyond ~3.5 A, measured); the density overlap dS has the right decay
    # and G / dS is nearly constant at contact (measured K = 4.5-5.3 Ha for Au, Mg vs O over 2.0-3.0 A).  So
    # E_Pauli = K_ab dS_ab with K_ab the least-squares ratio over the contact window [0.75, 1.25] (r_cov,a + r_cov,b).
    from trace_redox.constants import COVALENT_RADII_A
    from trace_redox.radial import density_overlap_difference
    raw = {k: (np.zeros_like(GRID) if dens[k] is None else dens[k].rho) for k in keys_d}
    from trace_redox.radial import density_overlap_parts
    K_pair, K_resid, pauli_coeffs = [], [], []
    windows = {}
    for (Za, na, Zb, nb) in pair_keys:
        windows.setdefault((Za, Zb), np.linspace(0.75, 1.25, 12) * (COVALENT_RADII_A[Za] + COVALENT_RADII_A[Zb]))
    by_window = {}
    for t, (Za, na, Zb, nb) in enumerate(pair_keys):
        by_window.setdefault((Za, Zb), []).append(t)
    G_win = {}
    for (Za, Zb), ts in by_window.items():
        G = gordon_kim_table(GRID, raw, frames, [pair_keys[t] for t in ts], windows[(Za, Zb)] / BOHR_A) * HARTREE_EV
        for k, t in enumerate(ts):
            G_win[t] = G[k]
    index_of = {tuple(k): t for t, k in enumerate(pair_keys)}

    def parts(Za, na, Zb, nb, R_A):
        return np.array(density_overlap_parts(GRID, raw[(Za, frames[Za])], raw[(Za, na)], raw[(Zb, frames[Zb])],
                                              raw[(Zb, nb)], R_A / BOHR_A))

    def fit(G, s):
        denom = float(np.dot(s, s))
        return float(np.dot(G, s) / denom) if denom > 0 else 0.0

    # one-carrier types first: K for a carrier of (Z, n) in the frame cloud of Z'
    K_one = {}
    for t, (Za, na, Zb, nb) in enumerate(pair_keys):
        fa, fb = frames[Za], frames[Zb]
        if (na != fa) != (nb != fb):
            P = np.array([parts(Za, na, Zb, nb, x) for x in windows[(Za, Zb)]])
            column = 0 if na != fa else 1
            K_one[t] = fit(G_win[t], P[:, column])
    K_cc_all = []
    for t, (Za, na, Zb, nb) in enumerate(pair_keys):
        fa, fb = frames[Za], frames[Zb]
        Ka = K_one.get(index_of.get((Za, na, Zb, fb), -1), 0.0) if na != fa else 0.0
        Kb = K_one.get(index_of.get((Za, fa, Zb, nb), -1), 0.0) if nb != fb else 0.0
        Kc = 0.0
        if na != fa and nb != fb:
            # the non-additive two-carrier remainder of G against (delta_a | delta_b)
            rest = G_win[t] - G_win[index_of[(Za, na, Zb, fb)]] - G_win[index_of[(Za, fa, Zb, nb)]]
            P = np.array([parts(Za, na, Zb, nb, x) for x in windows[(Za, Zb)]])
            Kc = fit(rest, P[:, 2])
            K_cc_all.append(Kc)
        Kvec = np.array([Ka, Kb, Kc])
        win = np.array([parts(Za, na, Zb, nb, x) @ Kvec for x in windows[(Za, Zb)]]) if (na != fa or nb != fb) else np.zeros(12)
        resid = float(np.linalg.norm(G_win[t] - win) / max(np.linalg.norm(G_win[t]), 1e-12)) if (na != fa or nb != fb) else 0.0
        K_pair.append([Ka, Kb, Kc])
        K_resid.append(resid)
        pauli_coeffs.append(chebyshev_fit(np.array([x * (parts(Za, na, Zb, nb, x) @ Kvec) for x in nodes_A]), N_CHEBYSHEV))
    pauli_coeffs = np.array(pauli_coeffs)
    big = [t for t in range(len(pair_keys)) if np.abs(G_win[t]).max() > 0.05]
    ones = [K_one[t] / HARTREE_EV for t in K_one if np.abs(G_win[t]).max() > 0.05]
    print(f"  Pauli: one-carrier K over {len(ones)} types with |G| > 50 meV: median {np.median(ones):.2f} Ha, range "
          f"{min(ones):.2f} .. {max(ones):.2f}; K_cc median {np.median(K_cc_all) / HARTREE_EV:.2f} Ha; relative residual "
          f"against G at contact over {len(big)} types: median {np.median([K_resid[t] for t in big]):.3f}, "
          f"90th percentile {np.percentile([K_resid[t] for t in big], 90):.3f}, worst {max(K_resid[t] for t in big):.3f}")
    # SPEC 26: the Gauss field factor lambda(R, w) of every carrier state's density difference, on a 2-D Chebyshev
    # grid (R in [R_MIN, R_CUT], the moment width w in FIELD_W_RANGE)
    from trace_redox.radial import FIELD_N_R, FIELD_N_W, FIELD_W_RANGE, chebyshev_fit_2d, field_factor
    field_keys, field_coeffs = [], []
    nodes_R = chebyshev_nodes(FIELD_N_R, R_MIN, R_CUT)
    nodes_W = chebyshev_nodes(FIELD_N_W, *FIELD_W_RANGE)
    for (Z, q) in keys_d:
        if q == frames[Z] or (Z, frames[Z]) not in dens:
            continue
        delta = raw[(Z, q)] - raw[(Z, frames[Z])]
        if not np.any(delta):
            continue
        values = np.array([[field_factor(GRID, delta, x / BOHR_A, y / BOHR_A) for y in nodes_W] for x in nodes_R])
        field_keys.append((Z, q))
        field_coeffs.append(chebyshev_fit_2d(values))
    print(f"  field-factor tables: {len(field_keys)} carrier states")
    out = os.path.join(REPO, "trace_redox", "data", "ion_densities.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, pair_keys=np.array(pair_keys, dtype=np.int64), pair_coefficients=np.array(coeffs),
             elements=np.array(elements, dtype=np.int64), r_min=R_MIN, r_cut=R_CUT,
             density_keys=np.array(keys_d, dtype=np.int64), grid=GRID,
             densities=np.array([dens[k].rho if dens[k] is not None else np.zeros_like(GRID) for k in keys_d]),
             frames=np.array([[z, frames[z]] for z in elements], dtype=np.int64),
             chebyshev_worst_error=worst,
             field_keys=np.array(field_keys, dtype=np.int64), field_coefficients=np.array(field_coeffs),
             field_w_range=np.array(FIELD_W_RANGE),
             pair_pauli_coefficients=pauli_coeffs, pair_pauli_K=np.array(K_pair), pair_pauli_K_residual=np.array(K_resid),
             splice_radii=np.array([splices.get(k, -1.0) for k in keys_d]),
             method=np.array("UKS " + METHOD + "; X2C spin-free for Z > 36; ANO-RCC + 2 diffuse s,p (ratio 3.2); 302-direction "
                             "spherical average; no tail splice; every O state in "
                             "a Watson sphere +2 at 1.205 A"))
    print(f"wrote {out}: {len(pair_keys)} pair types, worst Chebyshev error {worst:.2e} eV A")


if __name__ == "__main__":
    stage, work = sys.argv[1], sys.argv[2]
    if stage == "spec":
        stage_spec(work)
    elif stage == "densities":
        stage_densities(work, sys.argv[3:])
    elif stage == "tables":
        stage_tables(work)
    else:
        raise SystemExit(f"unknown stage {stage}")
