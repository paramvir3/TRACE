"""The mean cell dipole of the reference geometries and the tin-foil term it carries (M1c's vacuum dependence).

3-D Ewald with tin-foil boundaries gives a cell of dipole M_z the term -2 pi K_E M_z^2 / V relative to the isolated
slab (the slab correction of trace_electrostatics/ewald.py with the opposite sign), so lengthening the vacuum from c0
to c1 changes the mean state's energy by 2 pi K_E M_z(c0) M_z(c1) (1/V0 - 1/V1) to first order.  This probe reads
M_z = sum q_i z_i + sum mu_iz off the module's own record (the home cell's, SPEC 37); settings as in probe_ext_mesh_convergence.py.

    PYTHONPATH=.:experiments/benchmarks_4g python experiments/benchmarks_4g/probe_ext_vacuum_dipole.py MODEL.pt [mesh=4,1,1]
"""
import dataclasses
import math
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np
import torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references
from trace_electrostatics.constants import K_E

calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
MESH = next((tuple(int(x) for x in v.split("=")[1].split(",")) for v in sys.argv[2:] if v.startswith("mesh=")), (1, 1, 1))
mod.redox = dataclasses.replace(mod.redox, state_polarizability_stiffness="onsite", diabatic_space="clusters", band_form="fermi",
                                bvk_mesh=MESH)
SEEN = {}


def record(module, inputs, output):
    SEEN["q"] = output[1].q.detach().cpu().numpy().copy(); SEEN["mu"] = output[1].mu.detach().cpu().numpy().copy()


mod.register_forward_hook(record)
refs = references()
print(f"model {sys.argv[1].split('/')[-2]}; bvk_mesh {MESH} (the home cell's charges and dipoles); 'onsite', clusters space, 'fermi'; reference geometries, not relaxed")
for geometry in ("wetting", "non-wetting"):
    got = {}
    for c in (None, 40.0):
        a = refs[("undoped", geometry)].copy(); a.set_constraint(); a.info = {"charge": 0.0}
        if c is not None:
            cell = a.cell.array.copy(); cell[2, 2] = c; a.set_cell(cell, scale_atoms=False)
        a.calc = calc; calc.reset(); mod._ensemble_last = None
        e = a.get_potential_energy(); q, mu = SEEN["q"], SEEN["mu"]
        assert q.shape[0] == len(a), q.shape
        mz_q, mz_mu = float(q @ a.positions[:, 2]), float(mu[:, 2].sum())
        got[c] = (e, mz_q + mz_mu, a.get_volume())
        print(f"   {geometry:11s} c = {a.cell.array[2, 2]:8.4f} A: E/cell {e:.9f} eV; sum q {q.sum():+.2e} e; "
              f"M_z = {mz_q + mz_mu:+.6f} e A (charges {mz_q:+.6f}, dipoles {mz_mu:+.6f}); "
              f"M_x = {float(q @ a.positions[:, 0]) + float(mu[:, 0].sum()):+.6f}, M_y = {float(q @ a.positions[:, 1]) + float(mu[:, 1].sum()):+.6f}")
    (e0, m0, v0), (e1, m1, v1) = got[None], got[40.0]
    print(f"== {geometry:11s}: E changes by {1e3 * (e1 - e0):+.4f} meV; tin-foil term of the mean dipole, first order "
          f"{1e3 * 2 * math.pi * K_E * m0 * m1 * (1 / v0 - 1 / v1):+.4f} meV")
