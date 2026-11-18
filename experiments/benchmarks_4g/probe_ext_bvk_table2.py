"""Extensivity, L4 on the benchmark number (2026-11-10): Table II's undoped pair with the cell's fluctuations evaluated
in a Born-von Karman supercell.  Frozen weights, clusters space.

    python probe_ext_bvk_table2.py MODEL.pt FORM [nx ny ...]

The calculator below replicates the 1x1 cell, evaluates the model on the supercell and returns the energy per cell and
the forces on the first cell's atoms (all copies move together, so by symmetry that is the gradient of the energy per
cell).  Table II's protocol is then run unchanged: Au positions relaxed (fmax 0.005 eV/A), substrate fixed.  The 1x1
row repeats eval_aumgo's own number and is the check of this script.  Writes nothing into the run directory."""
import dataclasses
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references, relax, au_o_bond

model_path, form = sys.argv[1], sys.argv[2]
reps = [(int(sys.argv[i]), int(sys.argv[i + 1]), 1) for i in range(3, len(sys.argv) - 1, 2)] or [(1, 1, 1), (2, 2, 1)]
inner = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = inner.model.electrostatics
mod.redox = dataclasses.replace(mod.redox, state_polarizability_stiffness=form, diabatic_space="clusters")


class Supercell(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, rep):
        super().__init__()
        self.rep, self.calls, self.spread = rep, 0, 0.0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n, copies = len(atoms), int(np.prod(self.rep))
        big = atoms.copy(); big.set_constraint(); big = big.repeat(self.rep); big.info = dict(atoms.info)
        big.calc = inner; inner.reset()
        e = big.get_potential_energy(); f = big.get_forces().reshape(copies, n, 3)
        self.spread = max(self.spread, float(np.abs(f - f[0]).max()))      # the copies' forces must agree
        self.calls += 1
        self.results = {"energy": e / copies, "forces": f[0]}


refs = references()
print(f"model {model_path.split('/')[-2]}; state_polarizability_stiffness = {form!r}; clusters space; Table II protocol")
for rep in reps:
    out = {}
    for geometry in ("wetting", "non-wetting"):
        calc = Supercell(rep); t0 = time.time()
        a, e = relax(calc, refs[("undoped", geometry)])
        d = a.positions[109] - a.positions[108]
        out[geometry] = e
        print(f"   {rep[0]}x{rep[1]} {geometry:11s}: E/cell {e:.6f} eV  Au-Au {np.linalg.norm(d):.4f} A  Au-O {au_o_bond(a):.4f} A  "
              f"max|F(Au)| {np.abs(a.get_forces()[108:]).max():.1e} eV/A  evaluations {calc.calls}  "
              f"copies' forces agree to {calc.spread:.1e} eV/A  ({time.time() - t0:.0f} s)", flush=True)
    print(f"== {rep[0]}x{rep[1]}: E(wetting) - E(non-wetting), undoped, relaxed: {1e3 * (out['wetting'] - out['non-wetting']):+.2f} meV "
          f"(DFT +928.9; 4G +975.4)", flush=True)
