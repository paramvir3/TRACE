"""Review check: the reference data's own Hirshfeld charges on the two Au atoms, by class (all 5000 frames)."""
import sys, os, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np
import eval_aumgo as E
frames = E.read_runner(os.path.join(E.DATA, "AuMgO", "input.data"))
keys = [k for k in frames[0].arrays if k not in ("numbers", "positions")]
print("arrays:", keys, "| info:", list(frames[0].info)[:6], "| n frames", len(frames), "| cell", np.round(frames[0].cell.lengths(), 3))
def q_of(f):
    for k in ("hirshfeld_charges", "hirshfeld", "charges", "initial_charges", "ref_charges"):
        if k in f.arrays: return f.arrays[k]
    return f.get_initial_charges()
for doping in ("doped", "undoped"):
    for geometry in ("wetting", "non-wetting"):
        rows = []
        for f in frames:
            if (("Al" in f.get_chemical_symbols()) == (doping == "doped")) and E.geometry_class(f) == geometry:
                q = q_of(f); au = list(E.AU); z = f.positions[au, 2]
                slab = f.positions[:108, 2]
                d = [min(abs(zz - slab.min()), abs(zz - slab.max())) for zz in z]      # distance to the nearer slab face
                near, far = (au[0], au[1]) if d[0] < d[1] else (au[1], au[0])
                rows.append((q[near], q[far], q[:108].sum(), abs(d[0] - d[1])))
        r = np.array(rows)
        print(f"{doping:8s}{geometry:12s} n {len(r):4d}  q(near Au) {r[:,0].mean():+.4f} +- {r[:,0].std():.4f}  q(far Au) {r[:,1].mean():+.4f} +- {r[:,1].std():.4f}"
              f"  q(Au2) {(r[:,0]+r[:,1]).mean():+.4f}  slab {r[:,2].mean():+.4f}  <|dz|> {r[:,3].mean():.3f} A")
