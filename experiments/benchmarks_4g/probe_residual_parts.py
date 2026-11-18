# Diagnostic probe (2026-10-18; reported, not a criterion).  Usage (this directory): python probe_residual_parts.py RUN_DIR [...]
"""Per-structure energy residuals on the AuMgO test set, split into the model's parts.

For every test structure: r = E_model - E_DFT (meV), the parts short / bound / free, the number of integer
configurations in the diabatic window and their diagonal spread, the Au free charges, the Au2 tilt and the
lower Au's height.  Written to RUN_DIR/scores/residual_parts.npz; per class the residual scatter (sd, offset
removed) and the Pearson r of the residual with each part.  A part that carries structure-to-structure scatter
without forces (a selection that changes between structures) shows as r(residual, part) near +-1 with a part
sd of the residual's size.
"""
import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)
from ase.io import read  # noqa: E402
import trace_redox.module as redox_module  # noqa: E402
from eval_aumgo import AU, geometry_class  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

_search = redox_module.search_configurations
LOG = []


def logged_search(*args, **kwargs):
    configs, E = _search(*args, **kwargs)
    LOG.append((int(configs.shape[0]), float(E.max() - E.min())))
    return configs, E


redox_module.search_configurations = logged_search


def main():
    test = read(os.path.join(HERE, "data", "AuMgO_test.xyz"), ":")
    for run in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu")
        torch.set_num_threads(3)
        cap = []
        calc.model.electrostatics.register_forward_hook(lambda m, i, o: cap.append((o[0].detach(), o[1])))
        rows, cls = [], []
        for a0 in test:
            a = a0.copy()
            a.info = dict(a0.info)
            a.calc = calc
            calc.reset()
            cap.clear()
            LOG.clear()
            total = a.get_potential_energy()
            e_elec, r = cap[-1]
            q = r.free_charge.detach().numpy()[AU]
            p = a0.positions
            v = p[AU[1]] - p[AU[0]]
            tilt = np.degrees(np.arccos(abs(v[2]) / np.linalg.norm(v)))
            o = np.where(a0.numbers == 8)[0]
            low = AU[int(np.argmax(p[AU, 2]))]
            rows.append([1e3 * (total - a0.get_potential_energy()), 1e3 * (total - float(e_elec)),
                         1e3 * float(r.energy_bound), 1e3 * float(r.energy_free), LOG[-1][0], LOG[-1][1],
                         q[0], q[1], tilt, p[o, 2].min() - p[low, 2]])
            cls.append(("doped" if (a0.numbers == 13).any() else "undoped") + " " + geometry_class(a0))
        x = np.array(rows)
        cls = np.array(cls)
        os.makedirs(os.path.join(run, "scores"), exist_ok=True)
        np.savez(os.path.join(run, "scores", "residual_parts.npz"), x=x, cls=cls,
                 columns=np.array(["residual", "short", "bound", "free", "M", "spread", "q_au0", "q_au1", "tilt",
                                   "height"]))
        print(f"=== {os.path.basename(run.rstrip('/'))}  (meV per structure)")
        for c in sorted(set(cls)):
            s = x[cls == c]
            res = s[:, 0] - s[:, 0].mean()
            line = f"  {c:20s} n={len(s):3d}  residual mean {s[:, 0].mean():+8.1f} sd {res.std():7.1f}"
            for k, name in ((1, "short"), (2, "bound"), (3, "free")):
                part = s[:, k] - s[:, k].mean()
                line += f"  | {name} sd {part.std():7.1f} r {np.corrcoef(res, part)[0, 1]:+.2f}"
            line += f"  | M {int(s[:, 4].min())}-{int(s[:, 4].max())}"
            print(line)


if __name__ == "__main__":
    main()
