"""P5 like-for-like check (README, "Pre-registered: P5 like-for-like on Togo's 512-atom supercell", 2026-11-16): compares
one checkpoint's bands from 2x2x2 (64-atom) and 4x4x4 (512-atom) conventional force-constant supercells with each other
and with the PBE reference (Togo, fc2 on 4x4x4), without and with NAC.  Reads band yaml files only.

    python experiments/benchmarks_phonons/compare_sc2_sc4.py RUNS/<run>  RUNS/<run>_sc4

C1 as recorded for step 34b: frequencies at Gamma, X and L (6 path points) agree within 1 cm^-1, without and with NAC.
The C1 pre-registered for later steps (score_phonons.py --p5-supercell) uses every path point commensurate with 2x2x2
(7: also the middle of Gamma-X) on the bands without NAC; the 'exact on both' line below is that set.
Also: the supercell's effect on the model's own bands, and the RMS against PBE, total and per sorted branch."""
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_phonons import commensurate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
THZ = 33.35641
REF = {"nonac": os.path.join(HERE, "reference", "MgO_PBE_togo_band_nonac.yaml"),
       "nac": os.path.join(HERE, "reference", "MgO_PBE_togo_band_nac.yaml")}


def load(path):
    d = yaml.safe_load(open(path))
    q = np.array([p["q-position"] for p in d["phonon"]])
    f = np.sort(np.array([[b["frequency"] for b in p["band"]] for p in d["phonon"]]) * THZ, axis=1)
    return q, f


def rms(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def main(d2, d4):
    for kind in ("nonac", "nac"):
        q2, f2 = load(os.path.join(d2, f"band_{kind}.yaml"))
        q4, f4 = load(os.path.join(d4, f"band_{kind}.yaml"))
        qr, fr = load(REF[kind])
        assert q2.shape == q4.shape == qr.shape and np.allclose(q2, q4) and np.allclose(q2, qr), "q paths differ"
        print(f"=== {kind}: {len(q2)} path points")
        # commensurate points: Gamma, X (0.5, 0, 0.5), L (0.5, 0.5, 0.5) in fcc primitive reduced coordinates
        for name, qq in (("Gamma", (0, 0, 0)), ("X", (0.5, 0, 0.5)), ("L", (0.5, 0.5, 0.5))):
            idx = np.nonzero(np.abs(q2 - np.array(qq)).max(1) < 1e-9)[0]
            for i in idx:
                dmax = np.abs(f2[i] - f4[i]).max()
                print(f"  {name:5s} point {i:3d}: 2x2x2 {np.round(f2[i], 1).tolist()}  4x4x4 {np.round(f4[i], 1).tolist()}  "
                      f"PBE {np.round(fr[i], 1).tolist()}  max |2-4| {dmax:.3f}  (C1 {'pass' if dmax <= 1.0 else 'FAIL'})")
        print(f"  RMS model 2x2x2 vs 4x4x4: {rms(f2, f4):.2f}; vs PBE: 2x2x2 {rms(f2, fr):.2f}, 4x4x4 {rms(f4, fr):.2f}")
        print(f"  per sorted branch vs PBE, 2x2x2: {np.round(np.sqrt(((f2 - fr) ** 2).mean(0)), 1).tolist()}")
        print(f"  per sorted branch vs PBE, 4x4x4: {np.round(np.sqrt(((f4 - fr) ** 2).mean(0)), 1).tolist()}")
        print(f"  min frequency 2x2x2 {f2.min():.2f}, 4x4x4 {f4.min():.2f}")
        c2, c4 = commensurate(os.path.join(d2, f"band_{kind}.yaml"), 2), commensurate(os.path.join(d2, f"band_{kind}.yaml"), 4)
        assert np.all(c4[c2]), "a point commensurate with 2x2x2 must be commensurate with 4x4x4"
        for name, m in (("exact on both (commensurate with 2x2x2)", c2), ("exact on 4x4x4 only", c4 & ~c2),
                        ("interpolated on both", ~c4)):
            if m.any():
                print(f"  {name:40s} {int(m.sum()):3d} points: max |2-4| {np.abs(f2 - f4)[m].max():8.3f}; RMS vs PBE "
                      f"2x2x2 {rms(f2[m], fr[m]):5.1f}, 4x4x4 {rms(f4[m], fr[m]):5.1f}")
        k = np.argsort(-np.abs(f4 - fr).max(1))[:3]
        print(f"  largest |4x4x4 - PBE| at points {k.tolist()}: {[round(float(np.abs(f4[i] - fr[i]).max()), 1) for i in k]}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
