# Diagnostic probe (2026-10-18; reported, not a criterion).  Usage (this directory): python probe_window_convergence.py RUN_DIR [...]
"""Is the free energy converged in the configuration window?

The diabatic sector (SPEC 17.3-17.5) diagonalises H over the integer configurations within `diabatic_window` of
the lowest one and takes F = -kT ln Tr exp(-H / kT).  Without couplings a configuration at the edge Delta changes F
by kT ln(1 + exp(-Delta / kT)) ~ 1e-12 eV at Delta = 2.5, kT = 0.1.  With a coupling t to the low block it shifts
the lowest level by ~ -t^2 / Delta (second order), so F jumps whenever a configuration crosses the window edge:
energy that the forces (computed inside one selection) never see.

Same weights, windows 2.5 (trained), 3.5, 5.0 eV, on K test snapshots per (doping, geometry) class and on the four
T5 reference geometries (unrelaxed).  Reported: E(w) - E(2.5) per structure (meV) and the number of configurations.
"""
import dataclasses
import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)
from ase.io import read  # noqa: E402
import trace_redox.module as redox_module  # noqa: E402
from eval_aumgo import geometry_class, references  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

K = 6
WINDOWS = (2.5, 3.5, 5.0)
_search = redox_module.search_configurations
LOG = []


def logged_search(*args, **kwargs):
    configs, E = _search(*args, **kwargs)
    LOG.append((int(configs.shape[0]), float(E.max() - E.min())))
    return configs, E


redox_module.search_configurations = logged_search


def energy(calc, atoms):
    a = atoms.copy()
    a.info = {"charge": 0.0}
    a.set_constraint()
    a.calc = calc
    calc.reset()
    LOG.clear()
    e = a.get_potential_energy()
    return e, LOG[-1]


def main():
    test = read(os.path.join(HERE, "data", "AuMgO_test.xyz"), ":")
    refs = references()
    structures = []
    for (doping, geometry), ref in refs.items():
        sel = [a for a in test if ("Al" in a.get_chemical_symbols()) == (doping == "doped")
               and geometry_class(a) == geometry][:K]
        structures += [(f"{doping} {geometry} snap{i}", a) for i, a in enumerate(sel)]
        structures.append((f"{doping} {geometry} REF", ref))
    for run in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu")
        torch.set_num_threads(3)
        mod = calc.model.electrostatics
        base = mod.redox
        print(f"=== {os.path.basename(run.rstrip('/'))}   windows {WINDOWS} eV (kT {base.kT}); "
              "E(w) - E(2.5) in meV, [configurations, spread of included diagonal energies eV]")
        table = {}
        for w in WINDOWS:
            mod.redox = dataclasses.replace(base, diabatic_window=w, fragment_max_configurations=20000)
            table[w] = [energy(calc, a) for _, a in structures]
        mod.redox = base
        for i, (name, _) in enumerate(structures):
            e0 = table[WINDOWS[0]][i][0]
            cells = [f"{1e3 * (table[w][i][0] - e0):+9.2f} [{table[w][i][1][0]:4d}, {table[w][i][1][1]:.2f}]"
                     for w in WINDOWS]
            print(f"  {name:28s} " + "  ".join(cells))
        for w in WINDOWS[1:]:
            d = np.array([1e3 * (table[w][i][0] - table[WINDOWS[0]][i][0]) for i in range(len(structures))])
            print(f"  window {w}: |E(w) - E(2.5)| mean {np.abs(d).mean():.2f} max {np.abs(d).max():.2f} meV")


if __name__ == "__main__":
    main()
