"""The embedded first-principles Pauli values as a frozen-weight counterfactual on the wetting energies (2026-10-23).

Usage (repo root): python experiments/benchmarks_4g/probe_pauli_refit.py MODEL.pt [...]

The runtime Pauli term (SPEC 25, R K dS) was compared on 2026-10-20/22 with the frozen-density PBE value of each
AuMgO one-carrier type embedded in the point-ion MgO lattice (theory note 2026-10-18 §4c table).  Here each type's
series is scaled by embedded / runtime at its contact distance, a probe-level stand-in for K refitted to the embedded
values (a constant factor near contact), and the whole model's energies of the four T5 references give wet_u, wet_d
and T5 with and without it.  Pre-registered: the refit is a lever worth a training run only if it moves wet_u by
>= 100 meV toward DFT's +928.9 (frozen weights: direction and size only).
"""
import dataclasses
import os
import sys
import warnings

import torch

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)

from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_redox.penetration import library  # noqa: E402
from eval_aumgo import references  # noqa: E402

# (Za, qa, Zb, qb) as tabulated (Za <= Zb): embedded / runtime at contact (measured, theory note §4c)
SCALE = {(8, -2, 79, -1): 1.2785 / 1.1491, (8, -2, 79, 1): 1.3574 / 1.0488, (12, 2, 79, -1): 0.2757 / 0.1903,
         (12, 2, 79, 1): 0.1211 / 0.0746, (8, -2, 12, 1): 1.303 / 1.227, (8, -1, 12, 2): 0.601 / 0.425}


def wetting(calc, refs):
    E = {}
    for key, at in refs.items():
        a = at.copy()
        a.info = dict(at.info)
        a.calc = calc
        calc.reset()
        E[key] = float(a.get_potential_energy())
    wu = E[("undoped", "wetting")] - E[("undoped", "non-wetting")]
    wd = E[("doped", "wetting")] - E[("doped", "non-wetting")]
    return wu, wd, wd - wu


def main():
    refs = references()
    lib = library()
    saved = lib.gk_coefficients.clone()
    for model in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(2)
        mod = calc.model.electrostatics
        mod.redox = dataclasses.replace(mod.redox, fragment_max_configurations=20000)
        lib.gk_coefficients.copy_(saved)
        base = wetting(calc, refs)
        for key, f in SCALE.items():
            if key not in lib.index:
                raise KeyError(f"type {key} is not tabulated")
            lib.gk_coefficients[lib.index[key]] *= f
        refit = wetting(calc, refs)
        lib.gk_coefficients.copy_(saved)
        name = os.path.basename(os.path.dirname(model)) or os.path.basename(model)
        for label, (wu, wd, t5) in (("as trained", base), ("Pauli at embedded values", refit)):
            print(f"=== {name} {label}: wet_u {wu * 1000:+.1f}, wet_d {wd * 1000:+.1f}, T5 {t5 * 1000:+.1f} meV "
                  f"(DFT +928.9 / -2.7 / -931.6)", flush=True)
        print(f"    change: wet_u {(refit[0] - base[0]) * 1000:+.1f}, wet_d {(refit[1] - base[1]) * 1000:+.1f}, "
              f"T5 {(refit[2] - base[2]) * 1000:+.1f} meV", flush=True)


if __name__ == "__main__":
    main()
