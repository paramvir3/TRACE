# Diagnostic probe (2026-10-18; reported, not a criterion).  Usage (this directory): python probe_t5_path.py RUN_DIR [...]
"""Is a model's T5 deficit on the data or between the data and the reference geometry?

For each (doping, geometry) class, K test snapshots x_i are moved straight to the class reference x_ref of
eval_aumgo.references() (mean substrate per doping, mean Au positions per class; unrelaxed):

    x(lam) = (1 - lam) x_i + lam x_ref,   lam = 0, 1/4, 1/2, 3/4, 1.

Reported per class, as means over the K snapshots, relative to the reference (lam = 1):

    D(lam) = E(lam) - E_ref, split into short (ACE + baseline), bound and free parts,

with the residual E_model - E_DFT at lam = 0 and the Au free charges.  A harmonic surface gives
D(lam) = (1 - lam)^2 D(0): the ratio D(1/2) / D(0) is 1/4.  Two classes with the same number of moving atoms have
the same thermal potential energy to harmonic order, so D_wet(0) - D_non-wet(0) is ~0 for the DFT surface; a model
whose D_wet(0) - D_non-wet(0) differs from another's by X while both fit the snapshots moves its T5 by X.
"""
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
from eval_aumgo import AU, geometry_class, references  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

K = 6
LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)


def parts(calc, cap, atoms):
    a = atoms.copy()
    a.info = dict(atoms.info)
    a.set_constraint()
    a.calc = calc
    calc.reset()
    cap.clear()
    total = a.get_potential_energy()
    e_elec, r = cap[-1]
    p = np.array([total, total - float(e_elec), float(r.energy_bound), float(r.energy_free)])
    assert abs(p[1] + p[2] + p[3] - total) <= 1e-8 * (1 + abs(total))
    return p, r.free_charge.detach().numpy()[AU]


def main():
    test = read(os.path.join(HERE, "data", "AuMgO_test.xyz"), ":")
    refs = references()
    for run in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu")
        torch.set_num_threads(3)
        cap = []
        calc.model.electrostatics.register_forward_hook(lambda m, i, o: cap.append((o[0].detach(), o[1])))
        print(f"=== {os.path.basename(run.rstrip('/'))}   (meV; columns total / short / bound / free)")
        D0 = {}
        for key, ref in refs.items():
            doping, geometry = key
            sel = [a for a in test if ("Al" in a.get_chemical_symbols()) == (doping == "doped")
                   and geometry_class(a) == geometry][:K]
            p_ref, q_ref = parts(calc, cap, ref)
            D = np.zeros((len(LAMS), 4))
            resid, q0 = [], []
            for snap in sel:
                for k, lam in enumerate(LAMS[:-1]):
                    x = snap.copy()
                    x.info = {"charge": 0.0}
                    x.positions = (1 - lam) * snap.positions + lam * ref.positions
                    p, q = parts(calc, cap, x)
                    D[k] += (p - p_ref) / len(sel)
                    if k == 0:
                        resid.append(p[0] - snap.get_potential_energy())
                        q0.append(q)
            D0[key] = D[0]
            print(f"  {doping:8s} {geometry:11s} n={len(sel)}  residual(lam=0) {1e3 * np.mean(resid):+8.1f} "
                  f"+- {1e3 * np.std(resid) / np.sqrt(len(resid)):.1f}   Au free charge snapshots "
                  f"{np.round(np.mean(q0, 0), 3).tolist()} (sd {np.round(np.std(q0, 0), 3).tolist()}), ref "
                  f"{np.round(q_ref, 3).tolist()}")
            for k, lam in enumerate(LAMS[:-1]):
                print(f"      D({lam:.2f}) = " + " / ".join(f"{1e3 * v:+9.1f}" for v in D[k])
                      + (f"   D/D(0) total {D[k, 0] / D[0, 0]:.3f}" if k else ""))
        for doping in ("doped", "undoped"):
            d = D0[(doping, "wetting")] - D0[(doping, "non-wetting")]
            print(f"  {doping}: D_wet(0) - D_non-wet(0) = " + " / ".join(f"{1e3 * v:+9.1f}" for v in d))


if __name__ == "__main__":
    main()
