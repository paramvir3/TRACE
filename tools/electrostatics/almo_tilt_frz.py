"""FRZ / POL / CT of the tilted and the upright dimer on the same O (2026-11-04; theory note §4c): almo_split.run_geometry
unchanged (checks a1-a3) on almo_tilt.py's cluster.  Usage: almo_tilt_frz.py PATH.json OUT_DIR [R_AU]"""
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
    P = json.load(open(path_json))
    G = {"tilted": [np.array(p) for p in P["cluster_au"]["1.0"]], "upright": [np.array(p) for p in P["cluster_au"]["upright"]]}
    sites = W.quantum_sites(G["tilted"] + G["upright"], r_au=r_au)
    lattice = CI.rocksalt_lattice(W.D, 5, True, exclude=sites)
    out = {"r_au": r_au, "n_quantum": len(sites)}
    path = os.path.join(out_dir, f"almo_tilt_frz_r{r_au:.1f}.json")
    for name in ("upright", "tilted"):
        out[name] = AS.run_geometry(name, G[name], sites, lattice)
        json.dump(out, open(path, "w"), indent=1)
    u, t = out["upright"], out["tilted"]
    for k in ("FRZ", "POL", "POL_A", "POL_B", "CT_cp", "dE_cp"):
        print(f"  tilted - upright {k:6s} {(t[k] - u[k]) * 1000:+8.1f} meV", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 3.6)
