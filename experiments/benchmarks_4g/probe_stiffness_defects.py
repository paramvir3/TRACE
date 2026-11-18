"""Polarisation catastrophes under SPEC 28 on trained AuMgO bound sectors (frozen weights; 2026-10-19; reported).

Usage: python probe_stiffness_defects.py MODEL.pt [...]   (env PROBE_TEST=N: also N seeded random test structures;
PROBE_FILE=path: also the structures of an extxyz, e.g. the eps_inf prior's bulk MgO; PROBE_NO_T5=1: skip T5;
PROBE_KT=x: the redox kT in eV, to reproduce a training epoch of the redox_kT schedule; PROBE_ELEMENTS=79,8: the
SPEC 28 element restriction)

Switches RedoxConfig.state_polarizability on in a trained checkpoint and records, for every selected configuration
with an atom away from its frame at a ratio rho != 1 (Mg+ 70.7, Mg0 150.5, Al2+ 53.9, Al+ 94.4, Au- 3.39, O- 0.44 ...),
whether its stiffness K(n) = K + P_S^T Delta_S P_S stays positive definite (Haynsworth inertia criterion,
polarizability._positive_definite_after; exact).  The check is recorded instead of raised, so every configuration is
seen.  Reported per structure: configurations, those with a softened atom, defects, and the elements/states of the
defective configurations' softened atoms.
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

import trace_redox.polarizability as P  # noqa: E402
from ase.io import read  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from eval_aumgo import references  # noqa: E402

LOG = []
_orig = P._positive_definite_after


def _record(Delta, C_SS):
    ok = _orig(Delta, C_SS)
    with torch.no_grad():
        ev = torch.linalg.eigvalsh(0.5 * (torch.linalg.inv(Delta) + torch.linalg.inv(Delta).T) + 0.5 * (C_SS + C_SS.T))
    LOG.append((ok, int(Delta.shape[0]) // 3, float(ev.min()), float(ev.max())))
    return True                                                # record, do not raise


def main():
    P._positive_definite_after = _record
    structures = [] if os.environ.get("PROBE_NO_T5") else [(f"T5 {k[0]}/{k[1]}", a) for k, a in references().items()]
    if os.environ.get("PROBE_FILE"):                                   # extra structures, e.g. the eps_inf prior's
        for i, a in enumerate(read(os.environ["PROBE_FILE"], ":")):
            a.info.setdefault("charge", 0.0)
            structures.append((f"{os.path.basename(os.environ['PROBE_FILE'])}[{i}]", a))
    n_test = int(os.environ.get("PROBE_TEST", "0"))
    if n_test:
        test = read(os.path.join(HERE, "data", "AuMgO_test.xyz"), ":")
        for i in sorted(np.random.default_rng(0).choice(len(test), n_test, replace=False)):
            structures.append((f"test {i}", test[i]))
    for model in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(2)
        mod = calc.model.electrostatics
        mod.redox = dataclasses.replace(mod.redox, state_polarizability=True)
        if os.environ.get("PROBE_ELEMENTS"):               # SPEC 28 restricted to these atomic numbers
            mod.redox = dataclasses.replace(mod.redox, state_polarizability_elements=tuple(
                int(z) for z in os.environ["PROBE_ELEMENTS"].split(",")))
        if os.environ.get("PROBE_KT"):                     # a training-time electronic temperature (redox_kT schedule)
            mod.redox = dataclasses.replace(mod.redox, kT=float(os.environ["PROBE_KT"]))
        total = {"configs": 0, "softened": 0, "defects": 0}
        print(f"=== {os.path.basename(os.path.dirname(model))} (state_polarizability switched on, frozen weights, "
              f"kT = {float(mod.redox.kT):.4f} eV)")
        for label, atoms in structures:
            a = atoms.copy()
            a.info = dict(atoms.info)
            a.calc = calc
            calc.reset()
            LOG.clear()
            try:
                a.get_potential_energy()
                err = ""
            except Exception as e:                                     # noqa: BLE001
                err = f"  ERROR {type(e).__name__}: {str(e)[:120]}"
            n_cfg = int(mod._ensemble_last["configs"].shape[0]) if getattr(mod, "_ensemble_last", None) else -1
            bad = [x for x in LOG if not x[0]]
            total["configs"] += max(n_cfg, 0)
            total["softened"] += len(LOG)
            total["defects"] += len(bad)
            worst = min((x[2] for x in LOG), default=float("nan"))
            print(f"  {label:26s} configurations {n_cfg:3d}, with a softened atom {len(LOG):3d}, defects {len(bad):3d}; "
                  f"min eig(Delta^-1 + C) {worst:+.3e}{err}")
        print(f"  total: {total}")


if __name__ == "__main__":
    main()
