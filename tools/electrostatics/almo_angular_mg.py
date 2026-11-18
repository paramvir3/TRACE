"""Does the Au-Au bond's orientation modulate the Au-Mg2+ frozen / relaxed repulsion at the flat dimer's contact?
(2026-11-05; theory note section 4c, pre-registered.)

    OMP_NUM_THREADS=4 nice python tools/electrostatics/almo_angular_mg.py OUT.json [H] [theta ...]
    (python: an interpreter with PySCF installed)

The dimer (Au-Au 2.545 A) with its contact Au H = 2.89 A straight above the surface Mg at the origin (wetting_eda's
M0), rotated by theta from the normal toward the (1,1,0) diagonal (the flat dimer's partner direction); the quantum
cluster: the surface sites within 3.6 A of any Au of the scan (quantum_sites) in the point-ion slab.  Per theta:
I_FRZ and I_ALMO of the dimer (almo_split.run_geometry: FRZ, FRZ + POL) and of each Au alone (almo_angular.atom_almo),
D_X = I_X(dimer) - I_X(contact) - I_X(far)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import almo_angular as AA, almo_split as AS, cluster_interaction as CI, wetting_eda as W  # noqa: E402
out_json = sys.argv[1]; h = float(sys.argv[2]) if len(sys.argv) > 2 else 2.89
thetas = [float(x) for x in sys.argv[3:]] or [0.0, 45.0, 90.0]
e = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
contact = np.array([0.0, 0.0, h])
geoms = {th: [contact, contact + 2.545 * (np.sin(np.radians(th)) * e + np.cos(np.radians(th)) * np.array([0, 0, 1.0]))] for th in thetas}
sites = W.quantum_sites([p for g in geoms.values() for p in g], r_au=3.6)
surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf); lattice = CI.rocksalt_lattice(W.D, 5, True, exclude=sites)
print(f"{len(surf)} quantum ions ({sum(1 for s, _ in surf if s == 'Mg')} Mg) around the Mg-top contact at {h} A", flush=True)
res = json.load(open(out_json)) if os.path.exists(out_json) else {}
if "contact" not in res:
    res["contact"] = AA.atom_almo(contact, surf, qB, lattice); json.dump(res, open(out_json, "w"), indent=1)
print(f"lone Au0 over Mg at {h} A: I_FRZ {res['contact'][0]:+.4f}, I_ALMO {res['contact'][1]:+.4f} eV", flush=True)
for th in thetas:
    key = f"{th:.0f}"
    if f"far {key}" not in res:
        res[f"far {key}"] = AA.atom_almo(geoms[th][1], surf, qB, lattice); json.dump(res, open(out_json, "w"), indent=1)
    if f"dimer {key}" not in res:
        o = AS.run_geometry(f"Mg theta {key}", geoms[th], sites, lattice)
        res[f"dimer {key}"] = (o["FRZ"], o["FRZ"] + o["POL"]); json.dump(res, open(out_json, "w"), indent=1)
    D = [res[f"dimer {key}"][k] - res["contact"][k] - res[f"far {key}"][k] for k in (0, 1)]
    res[f"D {key}"] = D; json.dump(res, open(out_json, "w"), indent=1)
    print(f"  theta {th:3.0f}: dimer I_FRZ {res[f'dimer {key}'][0]:+.4f} I_ALMO {res[f'dimer {key}'][1]:+.4f}; far {res[f'far {key}'][0]:+.4f} / "
          f"{res[f'far {key}'][1]:+.4f} -> D_FRZ {D[0]:+.4f}, D_ALMO {D[1]:+.4f} eV", flush=True)
k0, k1 = f"D {thetas[0]:.0f}", f"D {thetas[-1]:.0f}"
print(f"D_ALMO({thetas[-1]:.0f}) - D_ALMO({thetas[0]:.0f}) = {res[k1][1] - res[k0][1]:+.4f} eV; frozen {res[k1][0] - res[k0][0]:+.4f}", flush=True)
