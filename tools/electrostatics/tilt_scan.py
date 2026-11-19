"""PBE along the flat dimer's tilt path (2026-11-03): is the 4G flat (wetting) geometry a minimum against the tilt that
the step-24 models relax into?

    OMP_NUM_THREADS=4 nice python tools/electrostatics/tilt_scan.py PATH.json OUT.json [R_AU] [lambda ...]
    (python: an interpreter with PySCF installed)

PATH.json from experiments/benchmarks_4g/probe_tilt_path.py --path (the Au positions in wetting_eda's cluster frame
for lambda = -0.2 ... 1.2: 0 the 4G flat reference, 1 the model's relaxed state, linear in between). For each lambda:
E_SCF of Au2 over the quantum MgO(100) cluster in the point-ion slab (wetting_eda's model: PBE, def2-TZVP with the def2
ECP on Au, closed shell, undoped, bare point-ion boundary: TIPs changed the undoped numbers by 2-4 meV, tip_checks.py
c2). One quantum cluster for every lambda: the surface sites within R_AU of any Au of any lambda (and the cations adjacent
to a quantum anion), so E(lambda) - E(0) holds the Au2-surface interaction and the dimer's own deformation and nothing
of the cluster's composition. Each geometry gets its own integration grid (level 3).
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


def main(path_json, out_json, r_au=3.6, lambdas=None):
    from pyscf import dft
    P = json.load(open(path_json))
    # numeric lambdas, or a label stored in the path file (e.g. "upright": the upright dimer on the tilted Au's O)
    lams = [x if x in P["cluster_au"] and not x.replace(".", "").lstrip("+-").isdigit() else float(x)
            for x in (lambdas or P["lambdas"])]
    au_all = [np.array(p) for lam in P["lambdas"] for p in P["cluster_au"][str(lam)]]
    sites = W.quantum_sites(au_all, r_au=r_au)
    surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
    qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf)
    lattice = CI.rocksalt_lattice(W.D, 5, True, exclude=sites)
    print(f"{len(surf)} quantum ions (charge {qB:+d}) within {r_au} A of the path; {len(lattice[1])} point ions", flush=True)
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}
    res.update({"path": path_json, "r_au": r_au, "n_quantum": len(surf)})
    for lam in lams:
        key = f"{lam:+.1f}" if isinstance(lam, float) else lam
        if key in res.get("E", {}):
            continue
        au = [("Au", np.array(p)) for p in P["cluster_au"][str(lam) if isinstance(lam, float) else lam]]
        mol = W._mol(au + surf, qB)
        grids = dft.gen_grid.Grids(mol)
        grids.level = 3
        grids.build()
        mf, e = W._solve(W._rks(mol, grids, lattice))
        res.setdefault("E", {})[key] = e * HARTREE
        res.setdefault("converged", {})[key] = bool(mf.converged)
        heights = [float(p[2]) for _, p in au]
        print(f"  {key}: E {e:.8f} Ha; Au heights {np.round(heights, 3).tolist()} A; converged {mf.converged}",
              flush=True)
        json.dump(res, open(out_json, "w"), indent=1)
    E = res["E"]
    if "+0.0" in E:
        print("E - E(0) (eV): " + "  ".join(f"{k}: {E[k] - E['+0.0']:+.3f}" for k in E),
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 3.6,
         sys.argv[4:] if len(sys.argv) > 4 else None)
