"""Rigid dissociation of one test dimer: model energy vs O-O distance.

Usage: python dissociation.py SRC_INDEX RUN_DIR [RUN_DIR ...]

Molecule B (the one not containing atom 0) is translated rigidly along the O-O
unit vector. Reported: E(R) - E(30 A) per model; for electrostatic models also
the local part (electrostatics detached) and U_elec, and the point-dipole
energy K_E [p_A.p_B - 3 (p_A.n)(p_B.n)] / R^3 of the model's own fragment
dipoles at that R (a self-consistency check of the tail, not a reference).
There is no quantum-chemistry reference along this path.
"""
import math
import os
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from ase import Atoms  # noqa: E402
from ase.io import read  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_models import build_data  # noqa: E402  (module-level eval code is guarded below)

K_E = 14.399645
src = int(sys.argv[1])
runs = sys.argv[2:]
frames = read(os.path.join(HERE, "data", "test.xyz"), ":") + read(os.path.join(HERE, "data", "train.xyz"), ":")
base = [a for a in frames if int(a.info["src_index"]) == src][0]
s = base.get_chemical_symbols()
oxy = [k for k, x in enumerate(s) if x == "O"]
owner = [int(np.argmin([np.linalg.norm(base.positions[k] - base.positions[o]) for o in oxy])) for k in range(len(base))]
B = [k for k in range(len(base)) if owner[k] == 1]
A = [k for k in range(len(base)) if owner[k] == 0]
r0 = float(np.linalg.norm(base.positions[oxy[1]] - base.positions[oxy[0]]))
n = (base.positions[oxy[1]] - base.positions[oxy[0]]) / r0
grid = [r0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 30.0]
grid = sorted(set(g for g in grid if g >= r0 - 1e-9))


def geometry(R):
    pos = base.positions.copy()
    pos[B] += (R - r0) * n
    return Atoms(numbers=base.numbers, positions=pos, pbc=False)


print(f"dimer src {src}: R_OO(ref) = {r0:.3f} A, E_ref = {1e3 * base.get_potential_energy():.1f} meV")
for run in runs:
    calc = TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu")
    model = calc.model
    rows = []
    for R in grid:
        a = geometry(R)
        a.calc = calc
        e_tot = a.get_potential_energy()
        row = {"R": R, "E": e_tot}
        if calc.electrostatics is not None:
            from trace_electrostatics import polarization
            data = build_data(calc, a)
            _, _, _, aux = model(data, training=False, return_electrostatics=True)
            out = aux["electrostatics"][0]
            q, mu, pos = out.solve_result.q.detach(), out.solve_result.mu.detach(), out.positions.detach()
            u_el = float(out.energy.detach())
            pA = polarization(q[A], pos[A], mu[A]).numpy()
            pB = polarization(q[B], pos[B], mu[B]).numpy()
            row.update(U_elec=u_el, qA=float(q[A].sum()), pA=pA, pB=pB,
                       E_dd=K_E * (pA @ pB - 3 * (pA @ n) * (pB @ n)) / R ** 3)
        rows.append(row)
    e_inf = rows[-1]["E"]
    print(f"\n== {os.path.basename(run)} (r_max {calc.r_max})")
    if calc.electrostatics is None:
        print(f"{'R_OO':>6s} {'E-E(30)':>10s}  [meV]")
        for r in rows:
            print(f"{r['R']:6.2f} {1e3 * (r['E'] - e_inf):10.3f}")
    else:
        u_inf = rows[-1]["U_elec"]
        dd_inf = rows[-1]["E_dd"]
        print(f"{'R_OO':>6s} {'E-E(30)':>10s} {'local':>10s} {'U_el-U(30)':>11s} {'pdip-pdip(30)':>10s} {'q_A (e)':>9s} {'|pA|,|pB| (e A)':>16s}  [meV]")
        for r in rows:
            loc = (r["E"] - r["U_elec"]) - (e_inf - u_inf)
            print(f"{r['R']:6.2f} {1e3 * (r['E'] - e_inf):10.3f} {1e3 * loc:10.3f} {1e3 * (r['U_elec'] - u_inf):11.3f} "
                  f"{1e3 * (r['E_dd'] - dd_inf):10.3f} {r['qA']:+9.4f} {np.linalg.norm(r['pA']):7.4f},{np.linalg.norm(r['pB']):7.4f}")
