"""Where do step 21 seed 42's gradient spikes come from?  The SPEC 21 switch band, tested (2026-10-27; a probe).

Usage (repo root): python experiments/benchmarks_4g/probe_switch_forces.py MODEL.pt IDX [IDX ...]
       (IDX: indices into data/AuMgO_train.xyz, the training loader's indices in nanguard's log)

Measured (step 21 s42 log): from epoch 11 the parameter gradient spikes to |g| 39-400 (typical 0.1-10) on six undoped
structures, upright ones with a stretched O-top contact (Au-O 2.59-2.70 A) and flat ones with an Au near a cation
(2.63-2.86 A); validation force plateaus at 0.18-0.23 eV/A against step 20's 0.09.  Hypothesis: with the couplings
anchored at 1-3 eV (SPEC 31), more one-electron moves sit inside the amplitude switch band a_off < e < a_on, where the
coupling sigma(e) tau depends on the configuration energies; its derivative is SPEC 21's stated non-physical part.

For each structure: the moves in the band (count, their |tau|, e, sigma) and the forces with the switch's derivative
removed (sigma detached: the same energy, sigma held at its value) against the full forces.  Pre-registered
reading: the hypothesis holds if the switch-derivative force (max per atom) on the spike structures exceeds 0.5 eV/A
and is at least ten times that on control structures (indices given after '--'); otherwise it is falsified.
"""
import dataclasses
import os
import sys
import warnings

import numpy as np
import torch

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)

import trace_redox.diabatic as D  # noqa: E402
import trace_redox.module as M  # noqa: E402
from ase.io import read  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

SEEN = {}
_switch = D.amplitude_switch
_solution = M.RedoxElectrostatics._diabatic_solution


def _switch_maybe_frozen(e, a_off, a_on):
    s = _switch(e, a_off, a_on)
    SEEN.setdefault("e", []).append(e.detach())
    return s.detach() if SEEN.get("freeze") else s


def _capture(self, e_hat, states, b, W, L, K, Q, T=None, **kw):
    SEEN["T"], SEEN["K"], SEEN["electrons"] = (None if T is None else T.detach()), K, kw.get("electrons")
    return _solution(self, e_hat, states, b, W, L, K, Q, T=T, **kw)


D.amplitude_switch = _switch_maybe_frozen
M.RedoxElectrostatics._diabatic_solution = _capture


def forces(calc, atoms, freeze):
    a = atoms.copy()
    a.info = dict(atoms.info)
    a.info.setdefault("charge", 0.0)
    a.calc = calc
    calc.reset()
    SEEN.clear()
    SEEN["freeze"] = freeze
    F = a.get_forces().copy()
    mod = calc.model.electrostatics
    return F, mod._ensemble_last, dict(SEEN)


def main():
    args = sys.argv[1:]
    model = args[0]
    spikes = [int(x) for x in args[1:args.index("--")]] if "--" in args else [int(x) for x in args[1:]]
    controls = [int(x) for x in args[args.index("--") + 1:]] if "--" in args else []
    frames = read(os.path.join(HERE, "data", "AuMgO_train.xyz"), ":")
    calc = TransformersACECalculator(model_path=model, device="cpu")
    torch.set_num_threads(2)
    mod = calc.model.electrostatics
    mod.redox = dataclasses.replace(mod.redox, fragment_max_configurations=20000)
    a_off, a_on = float(mod.redox.selection_amplitude_off), float(mod.redox.selection_amplitude_on)
    out = {}
    for label, idx in (("spike", spikes), ("control", controls)):
        for i in idx:
            at = frames[i]
            F_full, last, seen = forces(calc, at, freeze=False)
            F_frozen, _, _ = forces(calc, at, freeze=True)
            dF = np.linalg.norm(F_full - F_frozen, axis=1)
            e = torch.cat([x.flatten() for x in seen.get("e", [])]) if seen.get("e") else torch.zeros(0)
            band = (e > a_off) & (e < a_on)
            err = np.linalg.norm(F_full - at.get_forces(), axis=1)
            out[(label, i)] = float(dF.max())
            print(f"[{label} {i}] configurations {int(last['configs'].shape[0])}; importances evaluated {int(e.numel())}, "
                  f"in the band ({a_off}, {a_on}) {int(band.sum())}; switch-derivative force max {dF.max():.4f} eV/A "
                  f"(atom {int(dF.argmax())}); force error vs DFT max {err.max():.3f}, rms {np.sqrt((err ** 2).mean()):.4f}",
                  flush=True)
    s = [v for (lab, _), v in out.items() if lab == "spike"]
    c = [v for (lab, _), v in out.items() if lab == "control"]
    if s and c:
        holds = min(s) > 0.5 and min(s) >= 10 * max(c)
        print(f"pre-registered reading: switch-derivative force spike min {min(s):.3f}, control max {max(c):.3f} eV/A -> "
              f"hypothesis {'holds' if holds else 'falsified (as stated)'}", flush=True)


if __name__ == "__main__":
    main()
