"""The O-top bond's orientation dependence (2026-11-04): ALMO direction split of the tilted dimer (the step-24 s42
relaxed state, one Au 2.197 A straight above an O, the dimer 70 deg from vertical) against the upright dimer on the
same O (2.26 / 4.805 A), undoped, closed shell; almo_split.run_direction unchanged (validated: d1-d4, 2026-10-27).

    OMP_NUM_THREADS=4 nice python tools/electrostatics/almo_tilt.py PATH.json OUT_DIR [R_AU]
    (python: an interpreter with PySCF installed)

The donor side is identical in the two (the same O, its dangling p_z, the Au straight above it, 2.20 against 2.26 A);
only the acceptor's orientation turns. One quantum cluster serves both: the surface sites within R_AU of any Au of
either geometry (wetting_eda.quantum_sites)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import almo_split as AS  # noqa: E402
import cluster_interaction as CI  # noqa: E402
import wetting_eda as W  # noqa: E402


def main(path_json, out_dir, r_au=3.6):
    os.makedirs(out_dir, exist_ok=True)
    P = json.load(open(path_json))
    G = {"tilted": [np.array(p) for p in P["cluster_au"]["1.0"]], "upright": [np.array(p) for p in P["cluster_au"]["upright"]]}
    sites = W.quantum_sites(G["tilted"] + G["upright"], r_au=r_au)
    lattice = CI.rocksalt_lattice(W.D, 5, True, exclude=sites)
    print(f"{len(sites)} quantum ions within {r_au} A of either geometry", flush=True)
    out = {"r_au": r_au, "n_quantum": len(sites)}
    path = os.path.join(out_dir, f"almo_tilt_r{r_au:.1f}.json")
    for name in ("upright", "tilted"):
        out[name] = AS.run_direction(name, G[name], sites, lattice)
        json.dump(out, open(path, "w"), indent=1)
    u, t = out["upright"], out["tilted"]
    for k in ("CT_cp_surface->Au", "CT_cp_Au->surface", "rest", "CT_cp"):
        print(f"  tilted - upright {k:18s} {(t[k] - u[k]) * 1000:+8.1f} meV; ratio tilted / upright "
              f"{t[k] / u[k]:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 3.6)
