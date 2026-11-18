"""The upright dimer's vertical approach to its O-top site, as trained (2026-11-05; theory note §4c, SPEC 33's wall).

Usage (repo root): python experiments/benchmarks_4g/probe_upright_approach.py MODEL.pt [...]

The undoped non-wetting (upright) T5 reference with the dimer translated rigidly along the surface normal so that its
contact Au sits at h = 1.8, 2.0, 2.1, 2.26 (the reference), 2.5, 2.9 A above the O-top O; E(h) - E(2.26) in eV, the
ensemble size, and the PBE values of the same approach on the 6-ion cluster (tools/electrostatics/otop_upright_scan.py,
dE_SCF(cp) relative to 2.26 A) beside them.  A model whose curve has no wall below the data (a fall at 2.0 or 1.8 A) will
collapse the upright dimer in the T5 protocol's relaxation.
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
warnings.simplefilter("ignore")
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
from flashace.calculator import TransformersACECalculator  # noqa: E402
from eval_aumgo import AU, references  # noqa: E402

HEIGHTS = [1.8, 2.0, 2.1, 2.26, 2.5, 2.9]
#: PBE, the 6-ion cluster: dE_SCF(cp)(h) - dE_SCF(cp)(2.26) in eV (otop_upright_scan.json); None until measured
PBE = {}


def load_pbe():
    import json
    path = os.path.join(os.environ.get("TRACE_FP_CACHE", "fp_cache"), "almo_tilt", "otop_upright_scan.json")   # first-principles cache
    if os.path.exists(path):
        d = json.load(open(path))
        if "2.26" in d:
            return {float(k): v - d["2.26"] for k, v in d.items()}
    return {}


def main():
    pbe = load_pbe() or PBE
    ref = references()[("undoped", "non-wetting")]
    contact = AU[int(np.argmax(ref.positions[AU, 2]))]                  # the Au nearest the slab (the slab lies above)
    O = [i for i in range(108) if ref.get_chemical_symbols()[i] == "O"]
    o = O[int(np.argmin([np.linalg.norm(ref.positions[i] - ref.positions[contact]) for i in O]))]
    h0 = float(ref.positions[o, 2] - ref.positions[contact, 2])
    print(f"upright reference: contact Au {contact} at {h0:.3f} A below O{o}; PBE (6 ions) E(h) - E(2.26): "
          + ", ".join(f"{h:.2f}: {v:+.3f}" for h, v in sorted(pbe.items())) + " eV", flush=True)
    for model in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(3)
        mod = calc.model.electrostatics
        E, n = {}, {}
        for h in HEIGHTS:
            a = ref.copy()
            a.positions[AU, 2] += (h0 - h)                                # move the dimer toward the slab by h0 - h
            a.info = {"charge": 0.0}; a.calc = calc; calc.reset()
            try:
                E[h] = a.get_potential_energy(); n[h] = int(mod._ensemble_last["configs"].shape[0])
            except Exception as exc:
                E[h] = float("nan"); n[h] = f"{type(exc).__name__}"
        print(f"=== {os.path.basename(os.path.dirname(model))[-40:]}: E(h) - E(2.26) (eV; configurations): "
              + "  ".join(f"{h:.2f}: {E[h] - E[2.26]:+.3f} ({n[h]})" for h in HEIGHTS), flush=True)
        if pbe:
            dev = {h: E[h] - E[2.26] - pbe[h] for h in HEIGHTS if h in pbe and E[h] == E[h]}
            print("    model - PBE: " + "  ".join(f"{h:.2f}: {v:+.3f}" for h, v in dev.items()) + " eV", flush=True)


if __name__ == "__main__":
    main()
