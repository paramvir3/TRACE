"""Path RMS (cm^-1, frequencies sorted per q point) and Gamma optical frequencies of a band directory against the PBEsol
or PBE reference (2026-11-09; for band directories written by phonons_model.py).
Usage: python score_band_dir.py BAND_DIR [--ref pbesol|pbe]   (BAND_DIR holds band_nonac.yaml [and band_nac.yaml])"""
import os
import sys

import numpy as np
import yaml

THZ = 33.35640951981521
HERE = os.path.dirname(os.path.abspath(__file__))
d = sys.argv[1]
ref = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else "pbesol"
R = {"pbesol": "MgO_PBEsol_phonondb_band_{}.yaml", "pbe": "MgO_PBE_togo_band_{}.yaml"}[ref]


def band(p):
    return np.array([[b["frequency"] for b in q["band"]] for q in yaml.safe_load(open(p))["phonon"]]) * THZ


for tag in ("nonac", "nac"):
    p = os.path.join(d, f"band_{tag}.yaml")
    if not os.path.exists(p):
        continue
    m, r = np.sort(band(p), 1), np.sort(band(os.path.join(HERE, "reference", R.format(tag))), 1)
    print(f"{os.path.basename(os.path.normpath(d))} {tag:5s}: path RMS {np.sqrt(((m - r) ** 2).mean()):.1f} cm^-1; "
          f"Gamma optical {np.round(m[0, 3:], 1).tolist()} (ref {np.round(r[0, 3:], 1).tolist()})")
