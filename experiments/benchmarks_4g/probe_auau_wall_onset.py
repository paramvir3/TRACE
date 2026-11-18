"""The onset of the Au-Au wall's collapse on test structure 167 (follow-up record to probe_auau_wall_terms.py, README
2026-11-13; reported quantities, not pre-registered criteria; frozen weights, nothing trained or written to a run).

    PYTHONPATH=../.. python probe_auau_wall_onset.py RUN_DIR [RUN_DIR ...] [threads=2]

For each model, the dimer of test 167 moved rigidly along its bond (substrate frozen): the analytic bond force f
(positive = apart), the energy relative to d = 2.57 A, the total and free charges on the two Au, and the populations
of the charge states -1 / 0 / +1 (electrons removed) on each Au unit (RedoxResult.populations, states from
RedoxResult.states), with the number of configurations in the ensemble when the module records one space.
"""
import os
import sys
import warnings

warnings.simplefilter("ignore")
RUNS = [a for a in sys.argv[1:] if "=" not in a]
sys.argv = [sys.argv[0], "", ""] + [a for a in sys.argv[1:] if "=" in a]      # the probe module reads its options here
import numpy as np  # noqa: E402
import probe_auau_wall_terms as P  # noqa: E402
from ase.io import read  # noqa: E402

GRID = [2.00, 2.04, 2.06, 2.07, 2.08, 2.089, 2.10, 2.11, 2.12, 2.13, 2.14, 2.16, 2.20, 2.30, 2.57]
test = read(os.path.join(P.HERE, "data", "AuMgO_test.xyz"), ":")
a0 = test[P.MAIN]
au, u, d0 = P.dimer(a0)
for run in RUNS:
    m = P.Model(run)
    print(f"=== {m.name}")
    rows = []
    for d in GRID:
        a = P.at_distance(a0, d)
        b = a.copy(); b.info = dict(a.info); b.calc = m.calc; m.calc.reset()
        cap = []
        hook = m.mod.register_forward_hook(lambda mod, i, o: cap.append(o[1]))
        E = float(b.get_potential_energy()); F = b.get_forces()
        hook.remove()
        r = cap[-1]
        states = [int(round(float(s))) for s in r.states]
        pop = r.populations.detach().cpu().numpy()
        cols = [states.index(s) for s in (-1, 0, 1)]
        last = getattr(m.mod, "_ensemble_last", None)
        ncfg = int(last["configs"].shape[0]) if isinstance(last, dict) and last.get("configs") is not None else -1
        rows.append((d, float(0.5 * (F[au[0]] - F[au[1]]) @ u), E, [float(r.q[int(i)]) for i in au],
                     [float(r.free_charge[int(i)]) for i in au], [[float(pop[int(i), c]) for c in cols] for i in au], ncfg))
    E_ref = rows[-1][2]
    print(f"  {'d':>6s} {'f':>8s} {'E - E(2.57)':>12s} | {'q a / b':>15s} | {'free a / b':>15s} | "
          f"{'Au a: p(-1) p(0) p(+1)':>24s} | {'Au b: p(-1) p(0) p(+1)':>24s} | configs")
    for d, f, E, q, fr, pp, n in rows:
        print(f"  {d:6.3f} {f:+8.3f} {E - E_ref:+12.4f} | {q[0]:+7.3f}/{q[1]:+7.3f} | {fr[0]:+7.3f}/{fr[1]:+7.3f} | "
              f"{pp[0][0]:8.4f}{pp[0][1]:8.4f}{pp[0][2]:8.4f} | {pp[1][0]:8.4f}{pp[1][1]:8.4f}{pp[1][2]:8.4f} | {n}", flush=True)
    del m
