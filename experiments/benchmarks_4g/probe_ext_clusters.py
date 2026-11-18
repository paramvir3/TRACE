"""Extensivity, SPEC 36 (2026-11-10): the cluster product ensemble (``diabatic_space='clusters'``) against the shipped
global ensemble, frozen weights.  Scores the pre-registered criteria (b)-(d) of the README (SPEC 36).

    python probe_ext_clusters.py MODEL.pt FORM [nx ny ...] [doped]      FORM = checkpoint | onsite | block

Per geometry: the 1x1 cell in both spaces (E, forces, clusters), then per replication E / M - E(1x1) and max|dF|
against the 1x1 cell of the SAME space.  Writes nothing into the run directory."""
import dataclasses
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np
import torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references

DOPED = sys.argv[-1] == "doped"
if DOPED:
    sys.argv.pop()
model_path, form = sys.argv[1], sys.argv[2]
reps = [(int(sys.argv[i]), int(sys.argv[i + 1]), 1) for i in range(3, len(sys.argv) - 1, 2)] or [(2, 1, 1), (2, 2, 1)]
calc = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
if form != "checkpoint":
    mod.redox = dataclasses.replace(mod.redox, state_polarizability_stiffness=form)
trained = mod.redox
print(f"model {model_path.split('/')[-2]}\nevaluated with state_polarizability_stiffness = {mod.redox.state_polarizability_stiffness!r}, "
      f"kT = {mod.redox.kT}, window = {mod.redox.diabatic_window}, amplitudes = ({mod.redox.selection_amplitude_off}, "
      f"{mod.redox.selection_amplitude_on})\n")
refs = references()


def evaluate(key, rep, space):
    mod.redox = dataclasses.replace(trained, diabatic_space=space)
    a = refs[key].copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
    a.calc = calc; calc.reset(); mod._ensemble_last = None
    t0 = time.time()
    e = a.get_potential_energy(); f = a.get_forces()
    last = mod._ensemble_last
    note = f"configurations {int(last['configs'].shape[0]):4d}"
    if space == "clusters":
        n0 = last["reference"]
        sup = [sorted(set((S != n0).nonzero()[:, 1].tolist())) for S in last["clusters"]]
        note += f"  clusters {len(sup)} (sizes {[int(S.shape[0]) for S in last['clusters']]}, pair spaces {last['pair_sizes']})"
        if rep == (1, 1, 1):
            note += f"  units {sup}"
    return dict(e=e, f=f, seconds=time.time() - t0, note=note)


for doping in (("doped",) if DOPED else ("undoped",)):
    for geometry in ("wetting", "non-wetting"):
        key = (doping, geometry)
        one = {}
        for space in ("global", "clusters"):
            try:
                one[space] = evaluate(key, (1, 1, 1), space)
            except Exception as err:                                         # noqa: BLE001  (reported, not hidden)
                print(f"== {key} 1x1 {space:8s}: REFUSED {type(err).__name__}: {str(err)[:200]}", flush=True)
                continue
            print(f"== {key} 1x1 {space:8s}: E {one[space]['e']:+.9f} eV  {one[space]['note']}  ({one[space]['seconds']:.0f} s)", flush=True)
        if len(one) == 2:
            print(f"   1x1 clusters - global: dE {one['clusters']['e'] - one['global']['e']:+.3e} eV   max|dF| "
                  f"{np.abs(one['clusters']['f'] - one['global']['f']).max():.3e} eV/A", flush=True)
        for rep in reps:
            copies = rep[0] * rep[1]
            for space in ("global", "clusters"):
                if space not in one:
                    continue
                try:
                    r = evaluate(key, rep, space)
                except Exception as err:                                     # noqa: BLE001
                    print(f"   {rep[0]}x{rep[1]} {space:8s}: REFUSED {type(err).__name__}: {str(err)[:200]}", flush=True)
                    continue
                df = np.abs(r["f"] - np.tile(one[space]["f"], (copies, 1))).max()
                print(f"   {rep[0]}x{rep[1]} {space:8s}: E/cell - E(1x1) {1e3 * (r['e'] / copies - one[space]['e']):+11.4f} meV  "
                      f"max|dF| {df:.2e} eV/A  {r['note']}  ({r['seconds']:.0f} s)", flush=True)
