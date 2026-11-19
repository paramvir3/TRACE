"""The model's tilted - upright difference on the same O, by term (2026-11-04; theory note §4c): probe_t5_terms.terms
on the path's lambda = 1 state and the upright dimer on its contact O (probe_tilt_fold.py's geometries), plus the
backbone = E_total - E_q.  Models without the fold only.  Usage: probe_tilt_upright_terms.py PATH.json MODEL.pt [...]"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe_t5_terms as PT  # noqa: E402
from eval_aumgo import AU, references  # noqa: E402


def main():
    PT._install()
    P = json.load(open(sys.argv[1]))
    ref = references()[("undoped", "wetting")]
    tilted = np.array(P["cell_au"]["1.0"])
    low = int(np.argmax(tilted[:, 2]))
    O = [i for i in range(108) if ref.get_chemical_symbols()[i] == "O"]
    o = O[int(np.argmin([np.linalg.norm(ref.positions[i] - tilted[low]) for i in O]))]
    geoms = {"tilted": tilted, "upright": np.array([ref.positions[o] - [0, 0, 2.26], ref.positions[o] - [0, 0, 4.805]])}
    for model in sys.argv[2:]:
        calc = PT.TransformersACECalculator(model_path=model, device="cpu")
        PT.torch.set_num_threads(3)
        R = {}
        for name, pos in geoms.items():
            a = ref.copy(); a.positions[AU] = pos; a.info = {"charge": 0.0}
            res = PT.terms(calc, a)
            a.calc = calc; calc.reset()
            res["backbone"] = a.get_potential_energy() - sum(res.values())
            R[name] = res
        print(f"=== {os.path.basename(os.path.dirname(model))}  tilted - upright on O{o} (eV)")
        for k in R["tilted"]:
            d = R["tilted"][k] - R["upright"][k]
            if abs(d) > 0.005:
                print(f"  {k:10s} {d:+.3f}")
        print(f"  {'total':10s} {sum(R['tilted'].values()) - sum(R['upright'].values()):+.3f}")


if __name__ == "__main__":
    main()
