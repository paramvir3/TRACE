# Diagnostic probe (2026-10-18; reported, not a criterion).  Usage (this directory): python probe_window_deficit.py RUN_DIR [...]
"""Does the configuration-window truncation carry a model's per-structure energy scatter?

For every AuMgO test structure, same weights: E at the trained window (2.5 eV) and at 8 eV (cap 50000).  Between
2.5 and 8 eV lie the strongly coupled Au2 configurations (tau ~ 1.2 eV); the O -> band transfers of MgO lie at
~14 eV (amplitude ~0.02) and stay out.  The deficit d = E(2.5) - E(8) is energy the 2.5 eV selection drops; it
jumps whenever a configuration crosses the window edge and the forces never see it.

Per class: residual r = E(2.5) - E_DFT (sd, offset removed), the deficit's sd and the fraction of structures with
|d| > 10 meV, Pearson r(residual, d), and the sd of r - d (the residual at 8 eV).  If the truncation carries the
scatter, r(residual, d) is near +1 and the scatter of r - d is much smaller than that of r.
"""
import dataclasses
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
from eval_aumgo import geometry_class  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

_search = redox_module.search_configurations
LOG = []


def logged_search(*args, **kwargs):
    configs, E = _search(*args, **kwargs)
    LOG.append(int(configs.shape[0]))
    return configs, E


redox_module.search_configurations = logged_search


def main():
    test = read(os.path.join(HERE, "data", "AuMgO_test.xyz"), ":")
    for run in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu")
        torch.set_num_threads(3)
        mod = calc.model.electrostatics
        base = mod.redox
        rows, cls = [], []
        for a0 in test:
            out = []
            for w in (2.5, 8.0):
                mod.redox = dataclasses.replace(base, diabatic_window=w, fragment_max_configurations=50000)
                a = a0.copy()
                a.info = dict(a0.info)
                a.calc = calc
                calc.reset()
                LOG.clear()
                out += [a.get_potential_energy(), LOG[-1]]
            mod.redox = base
            rows.append([1e3 * (out[0] - a0.get_potential_energy()), 1e3 * (out[0] - out[2]), out[1], out[3]])
            cls.append(("doped" if (a0.numbers == 13).any() else "undoped") + " " + geometry_class(a0))
        x, cls = np.array(rows), np.array(cls)
        np.savez(os.path.join(run, "scores", "window_deficit.npz"), x=x, cls=cls,
                 columns=np.array(["residual_2.5", "deficit_2.5_minus_8", "M_2.5", "M_8"]))
        print(f"=== {os.path.basename(run.rstrip('/'))}  (meV per structure; deficit d = E(2.5) - E(8 eV))")
        for c in sorted(set(cls)) + ["all"]:
            s = x if c == "all" else x[cls == c]
            r, d = s[:, 0] - s[:, 0].mean(), s[:, 1] - s[:, 1].mean()
            corr = np.corrcoef(r, d)[0, 1] if d.std() > 0 else float("nan")
            print(f"  {c:20s} n={len(s):3d}  residual sd {r.std():7.1f}  deficit mean {s[:, 1].mean():+8.1f} sd "
                  f"{d.std():7.1f}  |d|>10 meV {np.mean(np.abs(s[:, 1]) > 10):.2f}  r(res, d) {corr:+.2f}  "
                  f"sd(res - d) {(r - d).std():7.1f}   M {int(s[:, 2].min())}-{int(s[:, 2].max())} / "
                  f"{int(s[:, 3].min())}-{int(s[:, 3].max())}")
        for d in ("doped", "undoped"):
            fl, up = x[cls == f"{d} wetting"], x[cls == f"{d} non-wetting"]
            off = fl[:, 0].mean() - up[:, 0].mean()
            off8 = (fl[:, 0] - fl[:, 1]).mean() - (up[:, 0] - up[:, 1]).mean()
            print(f"  {d}: class offset flat - upright at 2.5 eV {off:+.1f}, at 8 eV {off8:+.1f} meV")


if __name__ == "__main__":
    main()
