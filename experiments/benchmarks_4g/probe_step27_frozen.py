"""Step 27's frozen estimate (README step 27, pre-registered: "the frozen estimate is computed when the term is built";
2026-11-08).  Step 26's trained weights, RedoxConfig.bond_pauli_states off and on (nothing retrained):
(1) the training set's class offsets, mean model - DFT residual flat minus upright per doping (D2's quantity);
(2) wet_u, wet_d and T5 at the reference geometries of eval_aumgo.references() (single points, not relaxed).
Usage (repo root): python experiments/benchmarks_4g/probe_step27_frozen.py RUN_DIR [RUN_DIR ...]"""
import dataclasses
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
from ase.io import read  # noqa: E402

from flashace.calculator import TransformersACECalculator  # noqa: E402
from eval_aumgo import references  # noqa: E402


def geom(a):
    p = a.positions[a.numbers == 79]; v = p[1] - p[0]
    return "flat" if np.degrees(np.arccos(abs(v[2]) / np.linalg.norm(v))) > 45 else "upright"


def main():
    train = read(os.path.join(HERE, "data", "AuMgO_train.xyz"), ":")
    refs = references()
    for run in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu")
        torch.set_num_threads(4)
        mod = calc.model.electrostatics
        base = mod.redox
        out = {}
        for flag in (False, True):
            mod.redox = dataclasses.replace(base, bond_pauli_states=flag)
            res = {}
            for a0 in train:
                a = a0.copy(); a.info = dict(a0.info); a.calc = calc; calc.reset()
                key = ("doped" if (a0.numbers == 13).any() else "undoped", geom(a0))
                res.setdefault(key, []).append(1e3 * (a.get_potential_energy() - a0.get_potential_energy()))
            e = {}
            for key, at in refs.items():
                a = at.copy(); a.info = dict(at.info); a.calc = calc; calc.reset()
                e[key] = 1e3 * a.get_potential_energy()
            out[flag] = {"offset": {d: float(np.mean(res[(d, "flat")]) - np.mean(res[(d, "upright")])) for d in ("doped", "undoped")},
                         "mean": {k: float(np.mean(v)) for k, v in res.items()},
                         "wet_u": e[("undoped", "wetting")] - e[("undoped", "non-wetting")],
                         "wet_d": e[("doped", "wetting")] - e[("doped", "non-wetting")]}
        mod.redox = base
        print(f"=== {os.path.basename(os.path.normpath(run))} (step 26 weights; train set {len(train)}; meV)")
        for flag in (False, True):
            o = out[flag]
            print(f"  bond_pauli_states {str(flag):5s}: class offset flat - upright: doped {o['offset']['doped']:+8.1f}, "
                  f"undoped {o['offset']['undoped']:+8.1f}, doped - undoped {o['offset']['doped'] - o['offset']['undoped']:+8.1f}; "
                  f"references: wet_u {o['wet_u']:+8.1f}, wet_d {o['wet_d']:+8.1f}, T5 {o['wet_d'] - o['wet_u']:+8.1f}")
        d = {k: out[True]["mean"][k] - out[False]["mean"][k] for k in out[True]["mean"]}
        print("  frozen change of the class mean residual (on - off): " + ", ".join(f"{k[0]}/{k[1]} {v:+.1f}" for k, v in sorted(d.items())))


if __name__ == "__main__":
    main()
