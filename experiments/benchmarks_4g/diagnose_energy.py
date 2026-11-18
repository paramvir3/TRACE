"""Where does a model's energy error come from?  Bias vs scatter, and how it grows away from the minimum.

Usage: python diagnose_energy.py DATASET DATA_DIR RUN_DIR [RUN_DIR ...]

For each run, on DATA_DIR/DATASET_test.xyz and per subsystem (charge state):
* the mean signed energy error (a constant offset the model has not learned),
  its standard deviation (scatter) and the RMSE, in meV/atom;
* the force RMSE in meV/A;
* the signed energy error in bins of the reference energy above the
  subsystem's lowest test energy (0-0.25, 0.25-0.5, 0.5-1, 1-2, > 2 eV per
  structure): a smooth error that grows with height is under-fitting of the
  high-energy region, not noise.
Writes RUN_DIR/diagnose_energy.json.
"""
import json
import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval_4g  # noqa: E402  (float64, thread count)
from ase.io import read  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

BINS = (0.0, 0.25, 0.5, 1.0, 2.0, np.inf)        # eV per structure above the subsystem minimum


def main():
    dataset, data_dir, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    test = read(os.path.join(data_dir, f"{dataset}_test.xyz"), ":")
    for run in runs:
        calc = TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu")
        rows = {}
        for atoms in test:
            e, f = eval_4g.predict(calc, atoms)
            rows.setdefault(atoms.info["system"], []).append(
                (atoms.get_potential_energy(), (e - atoms.get_potential_energy()) / len(atoms) * 1e3,
                 (f - atoms.get_forces()).ravel() * 1e3))
        result = {"run": os.path.basename(run), "subsystems": {}}
        print(f"\n=== {dataset} / {os.path.basename(run)}")
        for system, items in sorted(rows.items()):
            ref = np.array([r[0] for r in items])
            de = np.array([r[1] for r in items])
            df = np.concatenate([r[2] for r in items])
            height = ref - ref.min()
            bins = []
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                m = (height >= lo) & (height < hi)
                if m.any():
                    bins.append({"range_eV": [lo, None if np.isinf(hi) else hi], "n": int(m.sum()),
                                 "mean_meV_atom": float(de[m].mean()), "std_meV_atom": float(de[m].std())})
            entry = {"n": len(items), "E_bias_meV_atom": float(de.mean()), "E_std_meV_atom": float(de.std()),
                     "E_rmse_meV_atom": float(np.sqrt(np.mean(de ** 2))),
                     "F_rmse_meV_A": float(np.sqrt(np.mean(df ** 2))), "by_height": bins}
            result["subsystems"][system] = entry
            print(f"  {system:10s} n={entry['n']:4d}  E bias {entry['E_bias_meV_atom']:+7.3f}  std "
                  f"{entry['E_std_meV_atom']:6.3f}  RMSE {entry['E_rmse_meV_atom']:6.3f} meV/atom  "
                  f"F RMSE {entry['F_rmse_meV_A']:6.1f} meV/A")
            print("      by height above min: " + "  ".join(
                f"[{b['range_eV'][0]}-{b['range_eV'][1] or 'inf'}) n={b['n']} {b['mean_meV_atom']:+.1f}"
                f"±{b['std_meV_atom']:.1f}" for b in bins))
        with open(os.path.join(run, "diagnose_energy.json"), "w") as handle:
            json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()
