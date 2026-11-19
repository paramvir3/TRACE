"""Frequencies at Gamma (both ends of the path), X, K, L and the RMS / max deviation over the whole Gamma-X-K-Gamma-L path
between a model band.yaml and a reference band.yaml (same path, same number of points).  Usage:
python band_table.py MODEL_BAND_YAML REF_BAND_YAML [label]"""
import sys
import numpy as np, yaml
THZ = 33.35641


def load(path):
    d = yaml.safe_load(open(path))
    f = np.array([[b["frequency"] for b in p["band"]] for p in d["phonon"]]) * THZ
    x = np.array([p["distance"] for p in d["phonon"]])
    seg = d["segment_nqpoint"]; bounds = np.cumsum([0] + seg)
    return x, np.sort(f, axis=1), bounds, d["labels"]


xm, fm, bm, lm = load(sys.argv[1]); xr, fr, br, lr = load(sys.argv[2]); label = sys.argv[3] if len(sys.argv) > 3 else sys.argv[1]
assert fm.shape == fr.shape, (fm.shape, fr.shape)
points = {"Γ": bm[0], "X": bm[1] - 1, "K": bm[2] - 1, "Γ'": bm[3] - 1, "L": bm[4] - 1}
print(f"{label}")
print(f"  {'point':4s} {'model (cm^-1)':48s} {'reference':48s} max|Δ|")
for name, i in points.items():
    print(f"  {name:4s} {str(np.round(fm[i], 1).tolist()):48s} {str(np.round(fr[i], 1).tolist()):48s} {np.abs(fm[i] - fr[i]).max():6.1f}")
d = fm - fr
print(f"  whole path, 6 branches x {fm.shape[0]} q: RMS {np.sqrt((d ** 2).mean()):.1f} cm^-1, max |Δ| {np.abs(d).max():.1f}; acoustic (3 lowest) RMS {np.sqrt((d[:, :3] ** 2).mean()):.1f}, optical (3 highest) RMS {np.sqrt((d[:, 3:] ** 2).mean()):.1f}")
