"""Is a set's force RMSE spread over its structures or carried by a few?  Per-structure force error of a checkpoint
through the ASE calculator (the path eval_aumgo.py uses), for the undoped and doped slabs of a data file.

    python probe_force_outliers.py RUN_DIR FILE.xyz [n=all] [threads=4] [out=FILE.npz]

Prints, per substrate: the pooled RMSE (eval_aumgo.py's number), the median, the 90th percentile and the largest
per-structure RMSE, the share of the pooled squared error carried by the worst 5 % of structures, the pooled RMSE
without them, and for the worst structures the atom with the largest force error.  Nothing is written to a run directory.
"""
import os
import sys
import warnings

warnings.simplefilter("ignore")
import numpy as np
import torch

opt = dict(a.split("=", 1) for a in sys.argv[3:])
torch.set_num_threads(int(opt.get("threads", 4)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from ase.io import read                                           # noqa: E402
from flashace.calculator import TransformersACECalculator         # noqa: E402

run, path = sys.argv[1].rstrip("/"), sys.argv[2]
calc = TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu")
frames = [a for a in read(path, ":") if "Au" in a.get_chemical_symbols()]
if opt.get("n", "all") != "all":
    frames = frames[: int(opt["n"])]
print(f"{os.path.basename(run)} on {os.path.basename(path)}: {len(frames)} slabs", flush=True)
rows = []
for k, a in enumerate(frames):
    b = a.copy()
    b.calc = calc
    d = b.get_forces() - a.get_forces()
    per_atom = np.sqrt((d ** 2).sum(axis=1))
    j = int(per_atom.argmax())
    rows.append((k, int("Al" in a.get_chemical_symbols()), float((d ** 2).mean()), float(per_atom[j]), j, a.get_atomic_numbers()[j],
                 float(a.positions[j, 2]), float(np.abs(a.get_forces()).max())))
    if (k + 1) % 50 == 0:
        print(f"   {k + 1}/{len(frames)}", flush=True)
r = np.array(rows)
if "out" in opt:
    np.savez(opt["out"], rows=r, columns="index doped force_mse largest_atom_error atom Z z_coordinate largest_reference_force_component")
for doped, name in ((0, "undoped"), (1, "doped")):
    s = r[r[:, 1] == doped]
    if not len(s):
        continue
    rms = np.sqrt(s[:, 2])
    order = np.argsort(-s[:, 2])
    worst = order[: max(1, len(s) // 20)]
    rest = np.setdiff1d(np.arange(len(s)), worst)
    print(f"== {name} ({len(s)}): pooled F RMSE {np.sqrt(s[:, 2].mean()):.4f} eV/A; per structure: median {np.median(rms):.4f}, 90th percentile "
          f"{np.percentile(rms, 90):.4f}, largest {rms.max():.4f}; the worst {len(worst)} carry {100 * s[worst, 2].sum() / s[:, 2].sum():.0f} % of the squared error; "
          f"pooled RMSE without them {np.sqrt(s[rest, 2].mean()):.4f}")
    for j in order[:6]:
        print(f"   structure {int(s[j, 0])}: F RMSE {rms[j]:.4f}; largest atomic error {s[j, 3]:.3f} eV/A on atom {int(s[j, 4])} (Z {int(s[j, 5])}, z = {s[j, 6]:.2f} A); "
              f"largest reference force component {s[j, 7]:.2f} eV/A")
