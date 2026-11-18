"""The model's energy along the tilt path, split by term (2026-11-03; theory note §4c, the tilt-path entry).

Usage (repo root): python experiments/benchmarks_4g/probe_tilt_terms.py PATH.json MODEL.pt [...]
For lambda = 0, 0.2, 0.4, 0.6, 0.8, 1.0 of the stored path (probe_tilt_path.py --path): the free-charge sector's exact
term split (probe_t5_terms.terms: e_b0, ladder, shifts, self, screened, overlap, ind26, alpha28, band, coherent, entropy;
checked to sum to H's diagonal and to the module's energy) and the backbone = E_total - E_q (the learned local energy and
the per-atom baseline, constant along the path), each relative to lambda = 0.  Models without the fold only (the term
split does not name SPEC 32's dressing).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe_t5_terms as PT  # noqa: E402  (its imports set float64 and the repo path)
from eval_aumgo import AU, references  # noqa: E402

LAMS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def main():
    PT._install()
    P = json.load(open(sys.argv[1]))
    ref = references()[("undoped", "wetting")]
    for model in sys.argv[2:]:
        calc = PT.TransformersACECalculator(model_path=model, device="cpu")
        PT.torch.set_num_threads(3)
        rows = {}
        for lam in LAMS:
            a = ref.copy()
            a.positions[AU] = np.array(P["cell_au"][str(lam)])
            a.info = {"charge": 0.0}
            res = PT.terms(calc, a)
            a.calc = calc
            calc.reset()
            E = a.get_potential_energy()
            res["backbone"] = E - sum(res.values())
            rows[lam] = res
        keys = [k for k in rows[0.0] if max(abs(rows[l][k] - rows[0.0][k]) for l in LAMS) > 0.01]
        print(f"=== {os.path.basename(os.path.dirname(model))}  (eV, relative to lambda = 0; terms moving > 10 meV)")
        print("  term       " + "".join(f"{l:>9.1f}" for l in LAMS))
        for k in keys:
            print(f"  {k:10s} " + "".join(f"{rows[l][k] - rows[0.0][k]:+9.3f}" for l in LAMS))
        tot = {l: sum(rows[l].values()) for l in LAMS}
        print(f"  {'total':10s} " + "".join(f"{tot[l] - tot[0.0]:+9.3f}" for l in LAMS))


if __name__ == "__main__":
    main()
