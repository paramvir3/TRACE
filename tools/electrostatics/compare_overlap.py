"""The overlap term of the AuMgO pairs: first principles against Gordon-Kim and the runtime tables (2026-10-20).

    OMP_NUM_THREADS=4 nice python tools/electrostatics/compare_overlap.py OUT_DIR
    (python: an interpreter with PySCF installed)

For each pair (state of a, state of b, R) at the AuMgO contacts (Au-O on an O top 2.26 A, Au-Mg flat 2.90 A, Mg-O
2.106 A):
  first principles (frozen_interaction.decomposed_interaction; library monomers, PBE0, X2C, ANO-RCC + diffuse):
      elst, the exact pieces of nonelst for HF / PBE / PBE0, in the bare pair Hamiltonian ('vac') and with the O
      states' Watson spheres in it ('sph');
  the library (trace_redox/data/ion_densities.npz): E_es of the spherical densities (SPEC 23), the four Gordon-Kim
      parts (gk_parts), and Gordon-Kim relative to the frames.  The runtime series R D and R K dS
      (penetration.PenetrationLibrary, needs torch) are added by `runtime_series` from the main environment:
      python tools/electrostatics/compare_overlap.py --runtime OUT_DIR
Open-shell O- is averaged over three orthogonal bond axes (one monomer orientation).  One JSON per pair in OUT_DIR,
skipped if present (restartable).  Pre-registered checks, printed PASS/FAIL:
  (i)   |Tr[dP S]| < 1e-8 electrons;
  (ii)  each monomer's spherically averaged density matches the library's: int 4 pi r^2 |drho| dr < 1e-3 e
        (r < min(splice radius, 10 bohr));
  (iii) closed-shell pairs: |elst_vac - E_es(library)| < 5 meV;
  (iv)  sum of the Gordon-Kim parts = gordon_kim_difference (gk_parts.check).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
import frozen_interaction as FI  # noqa: E402
from gk_parts import check as gk_check, radial  # noqa: E402

RadialDensity, gordon_kim_difference = radial.RadialDensity, radial.gordon_kim_difference
pair_electrostatic_energy = radial.pair_electrostatic_energy

HARTREE = 27.211386245988
BOHR = 0.529177210903
K_E = 14.399645
LIB = os.path.join(REPO, "trace_redox", "data", "ion_densities.npz")
# 2026-11-13: every element of the density library (was O, Mg, Al, Au: the 4G AuMgO pairs), for the frame pairs of SPEC 39
FRAME = {"H": 0, "C": 0, "O": -2, "Na": 1, "Mg": 2, "Al": 3, "Cl": -1, "Ag": 0, "Au": 0}
Z = {"H": 1, "C": 6, "O": 8, "Na": 11, "Mg": 12, "Al": 13, "Cl": 17, "Ag": 47, "Au": 79}
PAIRS = [(("Au", q), ("O", -2), 2.26) for q in (-1, 0, 1)] + [(("Au", q), ("Mg", 2), 2.90) for q in (-1, 0, 1)] \
    + [(("Mg", q), ("O", -2), 2.106) for q in (1, 2)] + [(("O", -1), ("Mg", 2), 2.106)]
OPEN_SHELL_ANISOTROPIC = {("O", -1)}
# `--scan`: the band electron's and the O hole's contacts at 0.9, 1.0, 1.1 of the rock-salt distances (Mg-O 2.106,
# O-O and Mg-Mg 2.978 A), each with its frame pair, to test whether first principles and K dS share their R dependence
SCAN = [(sa, sb, round(f * R0, 3)) for f in (0.9, 1.0, 1.1)
        for sa, sb, R0 in ((("Mg", 1), ("O", -2), 2.106), (("Mg", 2), ("O", -2), 2.106), (("O", -1), ("Mg", 2), 2.106),
                           (("O", -1), ("O", -2), 2.978), (("O", -2), ("O", -2), 2.978),
                           (("Mg", 1), ("Mg", 2), 2.978), (("Mg", 2), ("Mg", 2), 2.978))]


def library():
    d = np.load(LIB, allow_pickle=False)
    r = d["grid"]
    dens = {(int(k[0]), int(k[1])): d["densities"][i] for i, k in enumerate(d["density_keys"])}
    splice = {(int(k[0]), int(k[1])): float(d["splice_radii"][i]) for i, k in enumerate(d["density_keys"])}
    return r, dens, splice


def density_check(symbol, q, r, rho_lib, splice):
    """int 4 pi r^2 |rho_tool - rho_lib| dr (electrons) over r < min(splice, 10 bohr), 302-direction average."""
    occ, spin, _ = FI._monomer(symbol, q, np.zeros(3), None, None)
    mol = FI._mol([(symbol, np.zeros(3))], q, spin)
    rmax = min(splice if np.isfinite(splice) and splice > 0 else 10.0, 10.0)
    sel = np.nonzero((r > 1e-4) & (r < rmax))[0][::6]
    dirs, _ = FI._sphere(np.zeros(3), 1.0, 1.0)                          # the library's 302 unit directions
    rho = np.zeros(len(sel))
    for idx in np.array_split(np.arange(len(sel)), 8):
        xyz = (r[sel[idx]][:, None, None] * dirs[None, :, :]).reshape(-1, 3)     # bohr, as eval_gto takes
        ao = mol.eval_gto("GTOval", xyz)
        val = sum(np.sum((ao @ occ[s]) ** 2, axis=1) for s in (0, 1))
        rho[idx] = val.reshape(len(idx), len(dirs)).mean(axis=1)
    rr = r[sel]
    trap = getattr(np, "trapezoid", None) or np.trapz
    diff = trap(4 * np.pi * rr ** 2 * np.abs(rho - rho_lib[sel]), rr)
    n_tool = trap(4 * np.pi * rr ** 2 * rho, rr)
    return float(diff), float(n_tool), rmax


def run_pair(spec_a, spec_b, R, out_dir, lib):
    label = f"{spec_a[0]}{spec_a[1]:+d}_{spec_b[0]}{spec_b[1]:+d}_{R:.3f}"
    path = os.path.join(out_dir, label + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    r, dens, splice = lib
    axes = [(1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)] if {spec_a, spec_b} & OPEN_SHELL_ANISOTROPIC else [(0, 0, 1.0)]
    rows = [FI.decomposed_interaction(spec_a, spec_b, R, axis=ax) for ax in axes]
    fp = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
    fp_spread = {k: float(np.ptp([x[k] for x in rows])) for k in rows[0]}
    ka, kb = (Z[spec_a[0]], spec_a[1]), (Z[spec_b[0]], spec_b[1])
    Rb = R / BOHR
    e_es = pair_electrostatic_energy(ka[0], RadialDensity(r, dens[ka]), kb[0], RadialDensity(r, dens[kb]), Rb) * HARTREE
    gk = {k: v * HARTREE for k, v in gk_check(r, dens[ka], dens[kb], Rb).items()}
    fa, fb = (Z[spec_a[0]], FRAME[spec_a[0]]), (Z[spec_b[0]], FRAME[spec_b[0]])
    gk_rel = gordon_kim_difference(r, dens[fa], dens[ka], dens[fb], dens[kb], Rb) * HARTREE
    checks = {}
    for (sym, q), key in ((spec_a, ka), (spec_b, kb)):
        d, n, rmax = density_check(sym, q, r, dens[key], splice[key])
        checks[f"density_{sym}{q:+d}"] = {"int_abs_drho": d, "N_inside": n, "r_max_bohr": rmax, "pass": d < 1e-3}
    checks["number"] = {"value": fp["number"], "pass": abs(fp["number"]) < 1e-8}
    closed = not ({spec_a, spec_b} & OPEN_SHELL_ANISOTROPIC)
    checks["elst_vs_library"] = {"diff_eV": fp["vac_elst"] - e_es, "pass": (abs(fp["vac_elst"] - e_es) < 5e-3) if closed
                                 else None}
    out = {"pair": [list(spec_a), list(spec_b), R], "axes": len(axes), "fp": fp, "fp_spread": fp_spread,
           "lib_E_es": e_es, "gk_abs": gk, "gk_rel_frames": gk_rel, "checks": checks}
    json.dump(out, open(path, "w"), indent=1)
    return out


def runtime_series(spec_a, spec_b, R):
    """R D and R K dS of the runtime tables (eV A), frame-relative, at R (A)."""
    import torch
    from trace_redox.penetration import PenetrationLibrary
    lib = PenetrationLibrary()
    (sa, qa), (sb, qb) = spec_a, spec_b
    za, zb = Z[sa], Z[sb]
    key = (za, qa, zb, qb) if za <= zb else (zb, qb, za, qa)
    if key not in lib.index:
        return None, None
    t = torch.tensor([lib.index[key]])
    Rt = torch.tensor([R], dtype=torch.float64)
    rd = float(lib.r_times_D(t, Rt))
    saved = lib.coefficients
    lib.coefficients = lib.gk_coefficients
    rg = float(lib.r_times_D(t, Rt))
    lib.coefficients = saved
    return rd, rg


def add_runtime(out_dir):
    """Main (torch) environment: the runtime series at each stored pair, written into its JSON."""
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(out_dir, name)
        o = json.load(open(path))
        if "pair" not in o:
            continue
        (sa, qa), (sb, qb), R = o["pair"]
        rd, rg = runtime_series((sa, qa), (sb, qb), R)
        o["runtime_RD"], o["runtime_RKdS"] = rd, rg
        json.dump(o, open(path, "w"), indent=1)
        print(name, "R D", rd, "R K dS", rg)


def report(out_dir):
    """Frame-relative comparison (eV) of every stored pair whose frame pair is stored too:
    D = the penetration (exact electrostatics of the two densities minus the point charges), relative to the frames;
    P = the non-electrostatic overlap term, relative to the frames.  fp: first principles (vac); lib: SPEC 23 from the
    library densities; tab: the runtime series; GK: Gordon-Kim (SPEC 25's target); tab P = K dS (SPEC 25's runtime)."""
    store = {}
    for name in os.listdir(out_dir):
        if name.endswith(".json"):
            o = json.load(open(os.path.join(out_dir, name)))
            if "pair" not in o:                                   # e.g. cluster_interaction's output
                continue
            (sa, qa), (sb, qb), R = o["pair"]
            store[(sa, qa, sb, qb, round(R, 3))] = o
            store[(sb, qb, sa, qa, round(R, 3))] = o           # exchange symmetric (every quantity is)
    rows = []
    for (sa, qa, sb, qb, R), o in sorted(store.items()):
        fa, fb = FRAME[sa], FRAME[sb]
        if (qa, qb) == (fa, fb) or (sa, qa, sb, qb) != tuple(o["pair"][0] + o["pair"][1]):
            continue
        f = store.get((sa, fa, sb, fb, R))
        if f is None:
            continue
        point = (qa * qb - fa * fb) * K_E / R
        D_fp = o["fp"]["vac_elst"] - f["fp"]["vac_elst"] - point
        D_lib = o["lib_E_es"] - f["lib_E_es"] - point
        P = {xc: o["fp"][f"vac_nonelst_{xc}"] - f["fp"][f"vac_nonelst_{xc}"] for xc in ("hf", "pbe", "pbe0")}
        P_sph = o["fp"]["sph_nonelst_pbe"] - f["fp"]["sph_nonelst_pbe"]
        D_tab = o.get("runtime_RD") / R if o.get("runtime_RD") is not None else float("nan")
        P_tab = o.get("runtime_RKdS") / R if o.get("runtime_RKdS") is not None else float("nan")
        rows.append((f"{sa}{qa:+d}/{sb}{qb:+d}", R, D_fp, D_lib, D_tab, P["hf"], P["pbe"], P["pbe0"], P_sph,
                     o["gk_rel_frames"], P_tab, D_fp + P["pbe"], D_tab + P_tab))
    head = ("pair", "R", "D fp", "D lib", "D tab", "P hf", "P pbe", "P pbe0", "P pbe sph", "P GK", "P tab KdS",
            "net fp(pbe)", "net tab")
    print(("{:>12s} {:>6s} " + " ".join(["{:>10s}"] * 11)).format(*head))
    for r in rows:
        print(("{:>12s} {:6.3f} " + " ".join(["{:+10.4f}"] * 11)).format(*r))
    print("\nabsolute pieces (eV): first principles vac PBE vs Gordon-Kim parts of the superposed densities")
    print(("{:>12s} {:>6s} " + " ".join(["{:>10s}"] * 9)).format("pair", "R", "onebody", "coul", "dfa_pbe", "dfa_sum",
                                                                  "nonelst", "GK tf", "GK vw", "GK x", "GK c"))
    for (sa, qa, sb, qb, R), o in sorted(store.items()):
        if (sa, qa, sb, qb) != tuple(o["pair"][0] + o["pair"][1]):
            continue
        fp, gk = o["fp"], o["gk_abs"]
        print(("{:>12s} {:6.3f} " + " ".join(["{:+10.4f}"] * 9)).format(
            f"{sa}{qa:+d}/{sb}{qb:+d}", R, fp["vac_onebody"], fp["coul"], fp["dfa_pbe"], fp["dfa_sum_pbe"],
            fp["vac_nonelst_pbe"], gk["tf"], gk["vw"], gk["x"], gk["c"]))
    print("\nchecks")
    for (sa, qa, sb, qb, R), o in sorted(store.items()):
        if (sa, qa, sb, qb) == tuple(o["pair"][0] + o["pair"][1]):
            print(f"  {sa}{qa:+d}/{sb}{qb:+d}: " + "; ".join(f"{k} {v}" for k, v in o["checks"].items()))


def main():
    if sys.argv[1] == "--runtime":
        add_runtime(sys.argv[2])
        return
    if sys.argv[1] == "--report":
        report(sys.argv[2])
        return
    pairs = PAIRS
    if sys.argv[1] == "--scan":
        pairs = SCAN
        sys.argv.pop(1)
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    lib = library()
    for spec_a, spec_b, R in pairs:
        o = run_pair(spec_a, spec_b, R, out_dir, lib)
        c = o["checks"]
        print(f"{spec_a[0]}{spec_a[1]:+d}/{spec_b[0]}{spec_b[1]:+d} R {R}: elst_vac {o['fp']['vac_elst']:+.4f} "
              f"(from h {o['fp']['vac_elst_h']:+.4f}, mixed-relativity h {o['fp']['vac_elst_h_mixed']:+.4f}; "
              f"library E_es {o['lib_E_es']:+.4f}) nonelst_vac pbe {o['fp']['vac_nonelst_pbe']:+.4f} "
              f"hf {o['fp']['vac_nonelst_hf']:+.4f} | GK abs {sum(o['gk_abs'].values()):+.4f} | checks "
              + " ".join(f"{k}:{'PASS' if v['pass'] else ('n/a' if v['pass'] is None else 'FAIL')}" for k, v in c.items()),
              flush=True)


if __name__ == "__main__":
    main()
