"""The bulk-augmented Au2/MgO training set: the 4G slabs plus Togo's displaced PBE MgO supercells (forces only).
Train: 300 slabs + the fc3 cells not held out; valid: the 76 4G slabs + 10 held-out fc3 cells; the two fc2 cells (the
reference's own force constants) are excluded from both.  Deterministic split (seed 0).
Usage (repo root): python experiments/benchmarks_phonons/make_bulk_augmented_data.py

Three outputs, one per invocation (the bulk cells and their split are the same in all three):
  (no flag)            data/AuMgO_bulkaug_{train,valid}.xyz   the 4G slabs + the bulk cells
  --force-weight W     data/AuMgO_bulkaugw_{train,valid}.xyz  the same, with info['force_weight'] = W on the bulk cells
                       (W = 122.4, the measured ratio of the mean squared force components of the slabs and the bulk
                       cells, reproduces the files of the reweighted bulk run and of steps 33 / 34b)
  --bulk-only          data/MgO_bulkonly_{train,valid}.xyz    the bulk cells alone (136 / 10)
Inputs: data/MgO_PBE_togo_displaced.extxyz (extract_phono3py_cells.py) and ../benchmarks_4g/data/AuMgO_{train,valid}.xyz
(benchmarks_4g/prepare_all.sh).  --out-dir writes elsewhere (default: data/ next to this script)."""
import argparse
import os
import numpy as np
from ase.io import read, write
HERE = os.path.dirname(os.path.abspath(__file__)); D4G = os.path.join(HERE, "..", "benchmarks_4g", "data")
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
mode = ap.add_mutually_exclusive_group()
mode.add_argument("--force-weight", type=float, default=None, metavar="W",
                  help="set info['force_weight'] = W on the bulk cells and write AuMgO_bulkaugw_* (122.4 reproduces it)")
mode.add_argument("--bulk-only", action="store_true", help="write the bulk cells alone as MgO_bulkonly_*")
ap.add_argument("--cells", default=os.path.join(HERE, "data", "MgO_PBE_togo_displaced.extxyz"), help="the displaced PBE cells")
ap.add_argument("--data-4g", default=D4G, help="directory holding AuMgO_{train,valid}.xyz of the 4G benchmark")
ap.add_argument("--out-dir", default=os.path.join(HERE, "data"), help="where the two files are written")
args = ap.parse_args()
cells = read(args.cells, ":")
fc3 = [a for a in cells if a.info["source"] != "togo_phono3py_PBE_MgO_fc2"]
rng = np.random.default_rng(0); idx = rng.permutation(len(fc3)); held = set(idx[:10].tolist())
train_cells = [a for i, a in enumerate(fc3) if i not in held]; valid_cells = [a for i, a in enumerate(fc3) if i in held]
for a in train_cells + valid_cells:
    a.info["has_energy"] = "F"
if args.force_weight is not None:
    for a in train_cells + valid_cells:
        a.info["force_weight"] = args.force_weight
if args.bulk_only:
    name, train, valid = "MgO_bulkonly", train_cells, valid_cells
else:
    name = "AuMgO_bulkaug" if args.force_weight is None else "AuMgO_bulkaugw"
    train = read(os.path.join(args.data_4g, "AuMgO_train.xyz"), ":") + train_cells
    valid = read(os.path.join(args.data_4g, "AuMgO_valid.xyz"), ":") + valid_cells
os.makedirs(args.out_dir, exist_ok=True)
write(os.path.join(args.out_dir, f"{name}_train.xyz"), train); write(os.path.join(args.out_dir, f"{name}_valid.xyz"), valid)
print(f"train {len(train)} ({len(train_cells)} bulk cells), valid {len(valid)} ({len(valid_cells)} bulk cells); fc2 cells excluded: {len(cells) - len(fc3)}"
      + ("" if args.force_weight is None else f"; force_weight {args.force_weight!r} on the bulk cells") + f" -> {name}_{{train,valid}}.xyz")
