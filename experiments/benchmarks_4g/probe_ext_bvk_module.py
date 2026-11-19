"""SPEC 37 scored on the benchmark (2026-11-10): Born-von Karman sampling as the MODULE's option, frozen weights.

    python probe_ext_bvk_module.py MODEL.pt [P2]

P1  Table II's undoped pair on the 1x1 cell with ``bvk_mesh = (2, 2, 1)``: no supercell is built outside the module
    (probe_ext_bvk_table2.py built it in a calculator).  Protocol unchanged: Au relaxed (fmax 0.005 eV/A), substrate
    fixed.  Pre-registered: +935.11 +- 0.01 meV (s43), +890.85 +- 0.01 meV (s42), the numbers of README R16.
P2  (with the argument P2) ``bvk_length = 18.0`` A at the two undoped reference geometries: the 1x1 cell (mesh (2,2,1))
    against its own 2x2 replication (mesh (1,1,1)).  Pre-registered: |E(2x2)/4 - E(1x1)| <= 1e-3 meV and
    max|dF| <= 1e-6 eV/A.  The same pair at the unit mesh is printed beside it (the defect: +2.79 and -38.03 meV).

``state_polarizability_stiffness = 'onsite'`` and ``diabatic_space = 'clusters'`` as in R16.  Writes nothing into the
run directory."""
import dataclasses
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np
import torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references, relax, au_o_bond

model_path = sys.argv[1]
calc = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
base = dataclasses.replace(mod.redox, state_polarizability_stiffness="onsite", diabatic_space="clusters")
SEEN = {"mesh": None, "calls": 0}


def record(module, inputs, output):                               # the mesh the module used, read off its own record
    SEEN["mesh"] = tuple(output[1].bvk_mesh); SEEN["calls"] += 1


mod.register_forward_hook(record)
refs = references()
print(f"model {model_path.split('/')[-2]}; 'onsite' stiffness; clusters space; frozen weights")

print("P1  Table II, undoped, relaxed, on the 1x1 cell")
for mesh in ((1, 1, 1), (2, 2, 1)):
    mod.redox = dataclasses.replace(base, bvk_mesh=mesh)
    out = {}
    for geometry in ("wetting", "non-wetting"):
        calc.reset(); SEEN["calls"] = 0; t0 = time.time()
        a, e = relax(calc, refs[("undoped", geometry)])
        assert len(a) == 110 and SEEN["mesh"] == mesh, (len(a), SEEN["mesh"])
        out[geometry] = e
        print(f"   bvk_mesh {mesh} {geometry:11s}: E/cell {e:.6f} eV  Au-Au {np.linalg.norm(a.positions[109] - a.positions[108]):.4f} A  "
              f"Au-O {au_o_bond(a):.4f} A  max|F(Au)| {np.abs(a.get_forces()[108:]).max():.1e} eV/A  "
              f"module calls {SEEN['calls']}  ({time.time() - t0:.0f} s)", flush=True)
    print(f"== bvk_mesh {mesh}: E(wetting) - E(non-wetting) = {1e3 * (out['wetting'] - out['non-wetting']):+.4f} meV "
          f"(DFT +928.9; 4G +975.4)", flush=True)

if "P2" in sys.argv[2:]:
    print("P2  extensivity at a fixed sampling length, reference geometries (unrelaxed)")

    def single(atoms, rep):
        a = atoms.copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
        a.calc = calc; calc.reset()
        return a.get_potential_energy(), a.get_forces(), SEEN["mesh"]

    for label, redox in (("bvk_length = 18.0", dataclasses.replace(base, bvk_length=18.0)), ("unit mesh (shipped)", base)):
        mod.redox = redox
        for geometry in ("wetting", "non-wetting"):
            e1, f1, m1 = single(refs[("undoped", geometry)], (1, 1, 1))
            e4, f4, m4 = single(refs[("undoped", geometry)], (2, 2, 1))
            df = np.abs(f4 - np.tile(f1, (4, 1))).max()
            print(f"   {label:20s} {geometry:11s}: mesh 1x1 {m1}, 2x2 {m4};  E(2x2)/4 - E(1x1) = {1e3 * (e4 / 4 - e1):+.6f} meV;  "
                  f"max|dF| = {df:.2e} eV/A", flush=True)
