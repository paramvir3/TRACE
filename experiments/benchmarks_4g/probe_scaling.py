"""Wall time of one energy + forces of the redox model against system size (2026-10-19; reported).

Usage (repo root): python experiments/benchmarks_4g/probe_scaling.py MODEL.pt N1,N2,...   (rocksalt MgO n x n x n
conventional cells, 8 n^3 atoms; env SCALING_STACK=17 applies step 17's redox block)

Two repetitions per size after one warm-up; the minimum is reported with the number of selected configurations.
The fitted exponent between consecutive sizes, log(t2 / t1) / log(N2 / N1), says which term dominates."""
import dataclasses
import math
import os
import sys
import time
import warnings

import torch
import yaml

warnings.simplefilter("ignore")
sys.path.insert(0, ".")
torch.set_default_dtype(torch.float64)
from ase.build import bulk  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

model, sizes = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
calc = TransformersACECalculator(model_path=model, device="cpu")
torch.set_num_threads(int(os.environ.get("SCALING_THREADS", "2")))
mod = calc.model.electrostatics
if os.environ.get("SCALING_STACK") == "17":
    run = "experiments/benchmarks_4g/runs/AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_bondlaw_r5_s42"
    block = yaml.safe_load(open(f"{run}/config.yaml"))["electrostatics"]["redox"]
    names = {f.name for f in dataclasses.fields(mod.redox)}
    mod.redox = dataclasses.replace(mod.redox, **{k: (tuple(v) if isinstance(v, list) else v)
                                                  for k, v in block.items() if k in names})
prev = None
for n in sizes:
    atoms = bulk("MgO", "rocksalt", a=4.21, cubic=True).repeat((n, n, n))
    atoms.rattle(0.02, seed=1)
    atoms.info = {"charge": 0.0}
    atoms.calc = calc
    times = []
    for rep in range(3):
        calc.reset()
        t0 = time.perf_counter()
        atoms.get_forces()
        times.append(time.perf_counter() - t0)
    t = min(times[1:])
    m = int(mod._ensemble_last["configs"].shape[0]) if getattr(mod, "_ensemble_last", None) else -1
    expo = "" if prev is None else f"  exponent {math.log(t / prev[1]) / math.log(len(atoms) / prev[0]):.2f}"
    print(f"N = {len(atoms):5d}: {t:8.2f} s per energy + forces, {m} configurations{expo}", flush=True)
    prev = (len(atoms), t)
