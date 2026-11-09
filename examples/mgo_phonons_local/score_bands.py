"""Path RMS (cm^-1, frequencies sorted per q point) and Gamma optical frequencies of a band directory against the PBEsol
DFT reference (phonondb mp-1265, a = 4.21726 A).
Usage (repo root): python examples/mgo_phonons_local/score_bands.py BAND_DIR   (band_nonac.yaml [and band_nac.yaml])
Without band_nac.yaml (a model with no Born charges and none supplied) the model's NAC band is its no-NAC band, and that is
what is scored against DFT's NAC band."""
import os
import sys

import numpy as np
import yaml

THZ = 33.35640951981521
HERE = os.path.dirname(os.path.abspath(__file__))


def band(p):
    return np.array([[b["frequency"] for b in q["band"]] for q in yaml.safe_load(open(p))["phonon"]]) * THZ


d = sys.argv[1]
for tag in ("nonac", "nac"):
    p = os.path.join(d, f"band_{tag}.yaml")
    if not os.path.exists(p):
        p = os.path.join(d, "band_nonac.yaml")
    if os.path.exists(p):
        m = np.sort(band(p), 1)
        r = np.sort(band(os.path.join(HERE, "reference", f"MgO_PBEsol_phonondb_band_{tag}.yaml")), 1)
        print(f"{tag:5s}: path RMS {np.sqrt(((m - r) ** 2).mean()):.1f} cm^-1; Gamma optical {np.round(m[0, 3:], 1).tolist()} "
              f"(DFT {np.round(r[0, 3:], 1).tolist()})"
              + ("" if p.endswith(f"band_{tag}.yaml") else "  [no Born charges: the model's no-NAC band]"))
