"""The angular law of the bonded Au's frozen repulsion against the O-top (2026-11-04; theory note §4c, pre-registered 18:29).

    OMP_NUM_THREADS=4 nice python tools/electrostatics/frz_angular.py PATH.json OUT.json
    (python: an interpreter with PySCF installed)

The dimer (Au-Au 2.545 A) rotated rigidly about its contact Au, held 2.26 A straight above the path's O-top O, in the
vertical plane toward the tilted state's far Au, theta = 0, 20, 40, 60, 70, 90 deg from vertical; one quantum cluster for
every theta (almo_tilt.py's 6 ions, so the cluster never changes; at the largest theta the far Au approaches point ions,
stated). E1 of the dimer, of the contact atom alone (once) and of the far atom alone at each theta
(frz_nonadditivity.first_order); D(theta) = E1(dimer) - E1(contact) - E1(far) and the least-squares fit
D = D0 + D1 cos(theta) + D2 P2(cos(theta)) with its largest residual."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cluster_interaction as CI  # noqa: E402
import frz_nonadditivity as FN  # noqa: E402
import wetting_eda as W  # noqa: E402

THETAS = [0, 20, 40, 60, 70, 90]
R_AUAU, H = 2.545, 2.26


def main(path_json, out_json, h=H):
    P = json.load(open(path_json))
    tilted = [np.array(p) for p in P["cluster_au"]["1.0"]]
    upright = [np.array(p) for p in P["cluster_au"]["upright"]]
    sites = W.quantum_sites(tilted + upright, r_au=3.6)
    surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
    qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf)
    lattice = CI.rocksalt_lattice(W.D, 5, True, exclude=sites)
    o_site = np.array([upright[0][0], upright[0][1], 0.0])
    contact = o_site + [0, 0, h]
    low = int(np.argmin([p[2] for p in tilted]))
    e = tilted[1 - low][:2] - tilted[low][:2]
    e = np.array([*(e / np.linalg.norm(e)), 0.0])
    print(f"{len(surf)} quantum ions; contact Au at {np.round(contact, 3).tolist()}; rotation toward {np.round(e, 3).tolist()}",
          flush=True)
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}
    if "contact" not in res:
        res["contact"] = FN.first_order([contact], surf, qB, lattice)
        json.dump(res, open(out_json, "w"), indent=1)
    for th in THETAS:
        far = contact + R_AUAU * (np.sin(np.radians(th)) * e + np.cos(np.radians(th)) * np.array([0, 0, 1.0]))
        for label, atoms in (("dimer", [contact, far]), ("far", [far])):
            key = f"{label} {th}"
            if key not in res:
                res[key] = FN.first_order(atoms, surf, qB, lattice)
                json.dump(res, open(out_json, "w"), indent=1)
        D = res[f"dimer {th}"] - res["contact"] - res[f"far {th}"]
        res[f"D {th}"] = D
        print(f"  theta {th:3d}: far Au height {far[2]:.3f} A; E1 dimer {res[f'dimer {th}']:+.4f}, far atom "
              f"{res[f'far {th}']:+.4f}, contact {res['contact']:+.4f} -> D {D:+.4f} eV", flush=True)
    json.dump(res, open(out_json, "w"), indent=1)
    c = np.cos(np.radians(THETAS))
    A = np.stack([np.ones_like(c), c, 0.5 * (3 * c * c - 1)], 1)
    y = np.array([res[f"D {th}"] for th in THETAS])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = np.abs(A @ coef - y).max()
    mono = all(y[k + 1] >= y[k] - 1e-3 for k in range(len(y) - 1))
    print(f"  fit D = {coef[0]:+.4f} + {coef[1]:+.4f} cos + {coef[2]:+.4f} P2; largest residual {resid:.4f} eV; monotonic in "
          f"theta {mono} -> {'(g1)' if mono and resid <= 0.05 else '(g2)'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else H)
