"""The flat dimer's tilt path (2026-11-03): where the relaxed flat dimer of steps 23-24 goes, and the energy along it.

Usage (repo root):
  python experiments/benchmarks_4g/probe_tilt_path.py --path MODEL.pt OUT.json     relax the undoped flat reference as the
        T5 protocol does (eval_aumgo.relax; substrate fixed), report the sites under each Au, and write the path:
        lambda = -0.2 ... 1.2, linear in the two Au positions from the reference (0) to the relaxed state (1), in the
        4G cell and mapped to wetting_eda's cluster frame (ideal MgO(100), surface at z = 0, vacuum above)
  python experiments/benchmarks_4g/probe_tilt_path.py --scan PATH.json MODEL.pt [...]   each model's E(lambda) - E(0)
        along a stored path, as trained

The cluster frame: the flat reference's two Au sit over a nearest Mg pair of the surface's diagonal (wetting_eda:
M0 = (0, 0, 0), M1 = (1, 1, 0) d). The 4G surface Mg under each flat Au is found, M0' -> M0 by translation, M0'->M1'
-> (1, 1, 0)/sqrt 2 by rotation in the plane, and the in-plane handedness is fixed so that the surface O nearest the
tilted Au's footprint maps onto an O of the ideal lattice; height = (the slab's lowest layer) - z (the 4G slab lies
above the Au). Self-check: the mapped flat reference must reproduce wetting_eda.geometries()['flat'] (built from the
same 4G reference) to 0.05 A, and the mapped surface atoms near the Au must sit on ideal sites to 0.15 A (the 4G mean
substrate is relaxed; that residual is stated, not removed).
"""
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "tools", "electrostatics")))
warnings.simplefilter("ignore")
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
from eval_aumgo import AU, references, relax  # noqa: E402

D = 2.1325
LAMBDAS = [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]


def surface_frame(at):
    """(top z, surface-layer indices) of the slab (atoms < 108)."""
    z = at.positions[:108, 2]
    top = z.min()
    surf = [i for i in range(108) if at.positions[i, 2] < top + 0.6]
    return top, surf


def nearest_site(at, p, surf):
    """The surface atom nearest p in the plane (minimum image), its species and in-plane distance."""
    cell = at.cell.array
    best = None
    for i in surf:
        d = p[:2] - at.positions[i, :2]
        # minimum image in the plane (orthogonal cell)
        d = d - np.round(d / np.diag(cell)[:2]) * np.diag(cell)[:2]
        r = float(np.linalg.norm(d))
        if best is None or r < best[2]:
            best = (i, at.get_chemical_symbols()[i], r, d)
    return best


def to_cluster(at, positions, ref_au, surf, top):
    """Map 4G-cell Au positions to the cluster frame (see the module docstring)."""
    cell = np.diag(at.cell.array)[:2]
    m0 = nearest_site(at, ref_au[0], surf)
    m1 = nearest_site(at, ref_au[1], surf)
    assert m0[1] == "Mg" and m1[1] == "Mg", f"the flat Au are not over Mg: {m0[1]}, {m1[1]}"
    origin = at.positions[m0[0], :2]
    e = at.positions[m1[0], :2] - origin
    e = e - np.round(e / cell) * cell
    ang = np.arctan2(e[1], e[0]) - np.pi / 4

    def rot(v, flip):
        c, s = np.cos(-ang), np.sin(-ang)
        w = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
        return w[::-1] if flip else w            # the mirror through the (1,1) diagonal

    def mapped(p, flip):
        d = p[:2] - origin
        d = d - np.round(d / cell) * cell
        w = rot(d, flip)
        return np.array([w[0], w[1], top - p[2]])

    # handedness: either is a symmetry of the ideal surface for the flat dimer; fix it so the low Au of the last
    # position maps as it lies (both choices are reported identical for the flat reference)
    return [mapped(p, False) for p in positions], m0, m1


def path(model, out):
    from flashace.calculator import TransformersACECalculator
    import wetting_eda as W
    calc = TransformersACECalculator(model_path=model, device="cpu")
    torch.set_num_threads(3)
    ref = references()[("undoped", "wetting")]
    a = ref.copy(); a.info = {"charge": 0.0}; a.calc = calc; calc.reset()
    e0 = a.get_potential_energy()
    rel, e1 = relax(calc, ref)
    top, surf = surface_frame(ref)
    r0, r1 = ref.positions[AU].copy(), rel.positions[AU].copy()
    print(f"=== {os.path.basename(os.path.dirname(model))}: relaxation {(e1 - e0) * 1e3:+.1f} meV; heights "
          f"{np.round(top - r0[:, 2], 3).tolist()} -> {np.round(top - r1[:, 2], 3).tolist()} A; Au-Au "
          f"{np.linalg.norm(r0[0] - r0[1]):.3f} -> {np.linalg.norm(r1[0] - r1[1]):.3f} A", flush=True)
    for k, p in enumerate(r1):
        i, sym, d, _ = nearest_site(rel, p, surf)
        dist = [float(np.linalg.norm(rel.positions[j] - p)) for j in range(108)]
        j = int(np.argmin(dist))
        print(f"    relaxed Au {AU[k]}: height {top - p[2]:.3f} A, over {sym}{i} (in-plane {d:.3f} A); nearest atom "
              f"{rel.get_chemical_symbols()[j]}{j} at {dist[j]:.3f} A", flush=True)
    lam_pos = {lam: (1 - lam) * r0 + lam * r1 for lam in LAMBDAS}
    cl = {lam: to_cluster(ref, list(p), r0, surf, top)[0] for lam, p in lam_pos.items()}
    flat_cluster = np.array(cl[0.0])
    target = np.array(W.geometries()["flat"])
    err = min(np.abs(flat_cluster - target).max(), np.abs(flat_cluster[::-1] - target).max())
    print(f"    self-check: mapped flat reference against wetting_eda's flat geometry, max deviation {err:.3f} A "
          f"({'PASS' if err < 0.05 else 'FAIL'})", flush=True)
    # the surface atoms within 4 A (in plane) of any Au along the path, mapped: their distance to the ideal sites
    worst = 0.0
    _, m0, m1 = to_cluster(ref, [], r0, surf, top)
    for i in surf:
        q = to_cluster(ref, [ref.positions[i]], r0, surf, top)[0][0]
        near = min(np.linalg.norm(q[:2] - np.array(c)[:2]) for c in np.vstack(list(cl.values())))
        if near < 4.0:
            ideal = np.round(q[:2] / D) * D
            worst = max(worst, float(np.linalg.norm(q[:2] - ideal)), abs(q[2]))
    print(f"    mapped surface atoms near the path: largest deviation from ideal sites {worst:.3f} A "
          f"({'PASS' if worst < 0.15 else 'FAIL'}; the 4G mean substrate is relaxed)", flush=True)
    json.dump({"model": model, "relaxation_meV": (e1 - e0) * 1e3, "lambdas": LAMBDAS,
               "cell_au": {str(l): p.tolist() for l, p in lam_pos.items()},
               "cluster_au": {str(l): np.array(p).tolist() for l, p in cl.items()}}, open(out, "w"), indent=1)
    print(f"    path written to {out}", flush=True)


def scan(path_json, models):
    from flashace.calculator import TransformersACECalculator
    P = json.load(open(path_json))
    ref = references()[("undoped", "wetting")]
    for model in models:
        calc = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(3)
        E = {}
        for lam in P["lambdas"]:
            a = ref.copy()
            a.positions[AU] = np.array(P["cell_au"][str(lam)])
            a.info = {"charge": 0.0}; a.calc = calc; calc.reset()
            E[lam] = a.get_potential_energy()
        print(f"  {os.path.basename(os.path.dirname(model))[-40:]:40s} " +
              "  ".join(f"{lam:+.1f}: {(E[lam] - E[0.0]):+.3f}" for lam in P["lambdas"]) + "  eV", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--path":
        path(sys.argv[2], sys.argv[3])
    else:
        scan(sys.argv[2], sys.argv[3:])
