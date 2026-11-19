"""M1 (README, pre-registered 2026-11-11): the mesh convergence of Table II's pair on frozen weights.

E per cell of the reference geometries (not relaxed) on the 1x1 cell at Born-von Karman meshes (SPEC 37), and
E(wetting) - E(non-wetting) with its steps between meshes.  Settings: 'onsite' stiffness (SPEC 28.6), clusters
space with the closure (SPEC 36, 36.7), band_form 'fermi' (SPEC 38).  Nothing in the package or a run directory
changes; a refusal is printed and the next case runs.

    PYTHONPATH=. python experiments/benchmarks_4g/probe_ext_mesh_convergence.py MODEL.pt [undoped] [doped] [mesh=3,3,1 ...]

`c=40.0` (M1c) lengthens the cell's third vector to that many angstrom without moving an atom.
`bound=harrison` (B1) switches SPEC 27's bound on for the frozen coupling head; `cap=2048` sets
`fragment_max_configurations`.
`opbytes=12e9` (M1d) raises `max_operator_bytes`, the guard on the dense K that `bound_elimination='condensed'` never
builds; the peak resident memory is printed at the end.
"""
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

model_path = sys.argv[1]
args = sys.argv[2:]
dopings = [d for d in ("undoped", "doped") if d in args] or ["undoped"]
meshes = [tuple(int(x) for x in a.split("=")[1].split(",")) for a in args if a.startswith("mesh=")] \
    or [(1, 1, 1), (2, 2, 1), (3, 3, 1)]
c_length = next((float(a.split("=")[1]) for a in args if a.startswith("c=")), None)
calc = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
base = dataclasses.replace(mod.redox, state_polarizability_stiffness="onsite", diabatic_space="clusters", band_form="fermi")
cap = next((int(a.split("=")[1]) for a in args if a.startswith("cap=")), None)
if cap is not None:                                              # B1: the closure's cap (SPEC 36.8 L2)
    base = dataclasses.replace(base, fragment_max_configurations=cap)
if "bound=harrison" in args:                                     # B1: SPEC 27's bound on the frozen coupling head
    mod.coupling.bound = "harrison"
print(f"coupling bound {mod.coupling.bound!r}; cap {base.fragment_max_configurations}")
op_bytes = next((float(a.split("=")[1]) for a in args if a.startswith("opbytes=")), None)
if op_bytes is not None:                                         # M1d: the guard bounds a K the condensed route never builds
    mod.config = dataclasses.replace(mod.config, max_operator_bytes=op_bytes)
    print(f"max_operator_bytes raised to {op_bytes:.3g} (bound_elimination {mod.redox.bound_elimination!r})")
SEEN = {"mesh": None}


def record(module, inputs, output):                               # the mesh the module used, read off its own record
    SEEN["mesh"] = tuple(output[1].bvk_mesh)


mod.register_forward_hook(record)
refs = references()
print(f"model {model_path.split('/')[-2]}; 'onsite' stiffness; clusters space (closure {base.cluster_closure}); "
      f"band_form 'fermi'; frozen weights; reference geometries, not relaxed; cap {base.fragment_max_configurations}")
print(f"cell third vector: {'as in the reference' if c_length is None else f'{c_length} A (atoms not moved)'}")
per = {}
wet = {}
for doping in dopings:
    for mesh in meshes:
        mod.redox = dataclasses.replace(base, bvk_mesh=mesh)
        out = {}
        for geometry in ("wetting", "non-wetting"):
            a = refs[(doping, geometry)].copy(); a.set_constraint(); a.info = {"charge": 0.0}
            if c_length is not None:                              # M1c: the vacuum period only, atoms fixed
                cell = a.cell.array.copy(); assert abs(cell[2, 0]) + abs(cell[2, 1]) < 1e-12
                cell[2, 2] = c_length; a.set_cell(cell, scale_atoms=False)
                assert a.positions[:, 2].min() > 0.0 and a.positions[:, 2].max() < c_length
            a.calc = calc; calc.reset(); mod._ensemble_last = None; SEEN["mesh"] = None; t0 = time.time()
            try:
                e = a.get_potential_energy(); f = a.get_forces()
            except Exception as err:                              # a refusal (the cap) is a result, not a crash
                print(f"   {doping:7s} bvk_mesh {mesh} {geometry:11s}: REFUSED {type(err).__name__}: {str(err)[:200]}", flush=True)
                continue
            assert SEEN["mesh"] == mesh, SEEN["mesh"]
            last = mod._ensemble_last
            sizes = sorted(int(H.shape[0]) for H in last["hamiltonians"]) if last and "hamiltonians" in last else None
            out[geometry] = e
            per[(doping, geometry, mesh)] = e
            print(f"   {doping:7s} bvk_mesh {mesh} {geometry:11s}: E/cell {e:.9f} eV  max|F(Au)| {np.abs(f[108:]).max():.4e} eV/A  "
                  f"cluster sizes {sizes}  ({time.time() - t0:.0f} s)", flush=True)
        if len(out) == 2:
            wet[(doping, mesh)] = 1e3 * (out["wetting"] - out["non-wetting"])
            print(f"== {doping} bvk_mesh {mesh}: E(wetting) - E(non-wetting) = {wet[(doping, mesh)]:+.4f} meV", flush=True)
    done = [m for m in meshes if (doping, m) in wet]
    for m0, m1 in zip(done[:-1], done[1:]):
        print(f"== {doping}: step {m0} -> {m1}: {wet[(doping, m1)] - wet[(doping, m0)]:+.4f} meV", flush=True)
    steps = [wet[(doping, m1)] - wet[(doping, m0)] for m0, m1 in zip(done[:-1], done[1:])]
    for i, d in enumerate(steps[1:], start=1):                    # each later step against the first one
        print(f"== {doping}: step {done[i]} -> {done[i + 1]} over step {done[0]} -> {done[1]} = {d / steps[0]:+.4f}", flush=True)
for doping in dopings:                                           # each geometry's own steps and their ratios
    for geometry in ("wetting", "non-wetting"):
        e = [per[(doping, geometry, m)] for m in meshes if (doping, geometry, m) in per]
        st = [1e3 * (y - x) for x, y in zip(e[:-1], e[1:])]
        if len(st) >= 2:
            print(f"== {doping} {geometry:11s}: steps (meV) " + " ".join(f"{x:+.4f}" for x in st)
                  + "; ratios to the step before " + " ".join(f"{y / x:+.4f}" for x, y in zip(st[:-1], st[1:])))
import resource  # noqa: E402
print(f"peak resident memory of the process {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB")
print(time.strftime("done %Y-%m-%d %H:%M %Z"))
