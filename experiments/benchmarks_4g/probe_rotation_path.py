"""The rigid rotation of the dimer about its O-top contact (2026-11-04; theory note §4c): writes a path file in the
format of probe_tilt_path.py (keys = theta in degrees) for tilt_scan.py (PBE) and probe_tilt_path.py --scan (the models).
The contact Au 2.26 A straight below the path's O-top O (the 4G slab lies above), the partner 2.545 A away, rotated from
the surface normal by theta toward the tilted state's far Au.  Usage: probe_rotation_path.py PATH_IN.json PATH_OUT.json"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import probe_tilt_path as T  # noqa: E402
from eval_aumgo import AU, references  # noqa: E402

THETAS = [0.0, 20.0, 40.0, 60.0, 70.0, 90.0]
P = json.load(open(sys.argv[1]))
ref = references()[("undoped", "wetting")]
top, surf = T.surface_frame(ref)
tilted = np.array(P["cell_au"]["1.0"])
low = int(np.argmax(tilted[:, 2]))
O = [i for i in range(108) if ref.get_chemical_symbols()[i] == "O"]
o = O[int(np.argmin([np.linalg.norm(ref.positions[i] - tilted[low]) for i in O]))]
contact = ref.positions[o] - [0, 0, 2.26]
e = tilted[1 - low] - tilted[low]
e[2] = 0.0
e /= np.linalg.norm(e)
cell = {}
for th in THETAS:
    far = contact + 2.545 * (np.sin(np.radians(th)) * e - np.cos(np.radians(th)) * np.array([0, 0, 1.0]))
    cell[th] = np.array([contact, far]) if low == 0 else np.array([far, contact])
r0 = ref.positions[AU].copy()
clus = {th: T.to_cluster(ref, list(p), r0, surf, top)[0] for th, p in cell.items()}
json.dump({"model": "rigid rotation about O%d" % o, "lambdas": THETAS,
           "cell_au": {str(t): p.tolist() for t, p in cell.items()},
           "cluster_au": {str(t): np.array(p).tolist() for t, p in clus.items()}}, open(sys.argv[2], "w"), indent=1)
for th in THETAS:
    print(f"theta {th:4.0f}: cluster-frame Au {np.round(np.array(clus[th]), 3).tolist()}")
