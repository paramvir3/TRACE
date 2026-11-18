# Diagnostic probe (2026-10-10; reported, not a criterion): WHERE does a model's T5 wetting energy come from?
# Usage (this directory): python probe_wetting_decomposition.py RUN_DIR [RUN_DIR ...]
"""E(wetting) - E(non-wetting) of T5 (eval_aumgo.py's protocol: Au relaxed on the fixed mean substrate),
split into the model's own terms at the relaxed geometries:

    E = E_short + E_b0 + E_free
    E_short = E - E_elec                (ACE backbone + per-element baseline: the local, short-range part)
    E_b0    = RedoxResult.energy_bound  (bound sector alone: induced polarisation / SQE, no free charge)
    E_free  = RedoxResult.energy_free   (integer free charges + their screening + transfer + couplings)

For a coupling model, E_free(tau = 0) at the SAME geometry gives the coherent (resonance) energy
E_free - E_free(tau = 0).  The sum of the parts is checked against the total (1e-8 eV).
DFT: -2.7 / +928.9 meV (doped / undoped); 2G (local only) 375.0 / 375.0; 4G -41.0 / +975.4.
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
torch.set_num_threads(4)
from eval_aumgo import references, relax  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402


def parts(calc, atoms):
    mod = calc.model.electrostatics
    out = []
    hook = mod.register_forward_hook(lambda m, i, o: out.append((o[0].detach(), o[1])))
    a = atoms.copy()
    a.info = dict(atoms.info)
    a.calc = calc
    total = a.get_potential_energy()
    hook.remove()
    e_elec, r = out[-1]
    p = {"total": total, "short": total - float(e_elec), "bound": float(r.energy_bound), "free": float(r.energy_free)}
    assert abs(p["short"] + p["bound"] + p["free"] - total) <= 1e-8 * (1 + abs(total)), p
    p["au_free"] = r.free_charge.detach()[a.numbers == 79].tolist()
    if getattr(mod, "coupling", None) is not None:
        saved = {k: v.clone() for k, v in mod.coupling.state_dict().items()}
        with torch.no_grad():
            for prm in mod.coupling.parameters():
                prm.zero_()
        out.clear()
        hook = mod.register_forward_hook(lambda m, i, o: out.append((o[0].detach(), o[1])))
        b = atoms.copy()
        b.info = dict(atoms.info)
        b.calc = calc
        calc.reset()
        b.get_potential_energy()
        hook.remove()
        mod.coupling.load_state_dict(saved)
        calc.reset()
        p["free_tau0"] = float(out[-1][1].energy_free)
    return p


def main():
    refs = references()
    for run in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu")
        torch.set_num_threads(4)
        rows = {}
        for key, atoms in refs.items():
            relaxed, _ = relax(calc, atoms)
            relaxed.set_constraint()
            calc.reset()
            rows[key] = parts(calc, relaxed)
        print(f"=== {os.path.basename(run.rstrip('/'))}")
        for doping in ("doped", "undoped"):
            w, n = rows[(doping, "wetting")], rows[(doping, "non-wetting")]
            d = {k: 1e3 * (w[k] - n[k]) for k in ("total", "short", "bound", "free")}
            line = (f"  {doping:8s} E(wet) - E(non-wet) = {d['total']:+8.1f} meV  =  short {d['short']:+8.1f}"
                    f"  + bound {d['bound']:+8.1f}  + free {d['free']:+8.1f}")
            if "free_tau0" in w:
                res_w, res_n = w["free"] - w["free_tau0"], n["free"] - n["free_tau0"]
                line += (f"   [coherent tau energy: wet {1e3 * res_w:+.1f}, non-wet {1e3 * res_n:+.1f}, "
                         f"difference {1e3 * (res_w - res_n):+.1f} meV]")
            print(line)
            print(f"           Au free charges: wetting {np.round(w['au_free'], 3).tolist()}, "
                  f"non-wetting {np.round(n['au_free'], 3).tolist()}")


if __name__ == "__main__":
    main()
