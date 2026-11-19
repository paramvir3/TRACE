# Diagnostic probe (2026-10-18; reported, not a criterion).  Usage (this directory): python probe_amplitude_selection.py RUN_DIR [...]
"""The amplitude selection (trace_redox SPEC 21) on a trained checkpoint's frozen weights.

For K test structures per (doping, geometry) class and the four T5 references (unrelaxed): E with the trained
window selection (2.5 eV), E with the amplitude selection at windows 2.5 and 8 eV (their difference is the
window dependence the construction removes; it must be the Boltzmann weight of decoupled blocks, < 1e-6 eV
here), the configuration counts and the time per structure.  Per class: residual sd under each selection.  The
weights were trained with the window selection, so the residuals are reported only; a retrained model decides.
"""
import dataclasses
import os
import sys
import time
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
import trace_redox.diabatic as diabatic  # noqa: E402
from eval_aumgo import geometry_class, references  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

K = 25
_closure = diabatic.coupled_closure
COUNT = []


def counted_closure(*args, **kwargs):
    c = _closure(*args, **kwargs)
    COUNT.append(int(c.shape[0]))
    return c


diabatic.coupled_closure = counted_closure


def energy(calc, mod, base, atoms, **change):
    mod.redox = dataclasses.replace(base, **change)
    a = atoms.copy()
    a.info = dict(atoms.info)
    a.set_constraint()
    a.calc = calc
    calc.reset()
    COUNT.clear()
    t0 = time.perf_counter()
    e = a.get_potential_energy()
    return e, (COUNT[-1] if COUNT else -1), time.perf_counter() - t0


def main():
    test = read(os.path.join(HERE, "data", "AuMgO_test.xyz"), ":")
    refs = references()
    for run in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu")
        torch.set_num_threads(3)
        mod = calc.model.electrostatics
        base = mod.redox
        print(f"=== {os.path.basename(run.rstrip('/'))}  (meV per structure)")
        worst = 0.0
        for (doping, geometry), ref in refs.items():
            sel = [a for a in test if ("Al" in a.get_chemical_symbols()) == (doping == "doped")
                   and geometry_class(a) == geometry][:K]
            rows = []
            for a0 in sel + [ref]:
                ew, _, tw = energy(calc, mod, base, a0, diabatic_selection="window")
                ea, Ma, ta = energy(calc, mod, base, a0, diabatic_selection="amplitude")
                e8, M8, _ = energy(calc, mod, base, a0, diabatic_selection="amplitude", diabatic_window=8.0,
                                   fragment_max_configurations=50000)
                dft = a0.get_potential_energy() if a0 is not ref else float("nan")
                rows.append([1e3 * (ew - dft), 1e3 * (ea - dft), 1e3 * (ea - ew), 1e3 * (e8 - ea), Ma, M8, tw, ta])
            mod.redox = base
            x = np.array(rows)
            s = x[:-1]
            worst = max(worst, float(np.abs(x[:, 3]).max()))
            print(f"  {doping:8s} {geometry:11s} n={len(s)}  residual sd window {np.std(s[:, 0]):7.1f}  amplitude "
                  f"{np.std(s[:, 1]):7.1f}  (means {s[:, 0].mean():+7.1f} / {s[:, 1].mean():+7.1f})   E(amp) - E(win) "
                  f"mean {s[:, 2].mean():+8.1f} range {s[:, 2].min():+8.1f} .. {s[:, 2].max():+8.1f}   M {int(s[:, 4].min())}-"
                  f"{int(s[:, 4].max())}   time {s[:, 6].mean():.2f} / {s[:, 7].mean():.2f} s   REF E(amp) - E(win) "
                  f"{x[-1, 2]:+8.1f}")
        print(f"  window dependence of the amplitude selection |E(8) - E(2.5)| max {worst:.2e} meV")


if __name__ == "__main__":
    main()
