"""T5 on frozen weights under each queued step's exact redox settings (2026-10-19; direction and size only).

Usage: python probe_t5_stack.py MODEL.pt [...]

For each run directory in STEPS, the `electrostatics.redox` block of its config.yaml is applied to the frozen
checkpoint (dataclasses.replace with every key RedoxConfig has), and T5 = (E_dw - E_dn) - (E_uw - E_un) is computed
at the four reference geometries (eval_aumgo.references; unrelaxed) with E_q's part, the upright undoped dimer's
p(Au+ Au-) and the doped flat p(Au-).  Untrained weights: a counterfactual of direction, not a prediction of the
trained step.  DFT: T5 -931.6, doped -2.7, undoped +928.9 meV.
"""
import dataclasses
import os
import sys
import warnings

import torch
import yaml

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)

from flashace.calculator import TransformersACECalculator  # noqa: E402
from eval_aumgo import AU, references  # noqa: E402

STEPS = [("step 13 (as trained)", None),
         ("step 13S (+ spin)", "AuMgO_formal_kT_eps_cpl_alpha_exband_amp_spin_bondlaw_r5_s42"),
         ("step 14 (+ 22)", "AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_bondlaw_r5_s42"),
         ("step 15 (+ 23/24/25)", "AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_bondlaw_r5_s42"),
         ("step 16 (+ 26)", "AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_bondlaw_r5_s42"),
         ("step 17 (+ 28 Au,O)", "AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_bondlaw_r5_s42")]


def redox_of(run):
    cfg = yaml.safe_load(open(os.path.join(HERE, "runs", run, "config.yaml")))
    return cfg["electrostatics"]["redox"]


def main():
    refs = references()
    for model in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(2)
        mod = calc.model.electrostatics
        base = mod.redox
        fields = {f.name for f in dataclasses.fields(base)}
        cap = []
        mod.register_forward_hook(lambda m, i, o: cap.append(o[1]))
        print(f"=== {os.path.basename(os.path.dirname(model))}")
        for label, run in STEPS:
            if run is not None:
                block = {k: (tuple(v) if isinstance(v, list) else v) for k, v in redox_of(run).items() if k in fields}
                changed = {k: v for k, v in block.items() if getattr(base, k) != v}
                mod.redox = dataclasses.replace(base, **changed)
            else:
                changed, mod.redox = {}, base
            E, Eq, pion, pm = {}, {}, None, None
            try:
                for key, atoms in refs.items():
                    a = atoms.copy()
                    a.info = dict(atoms.info)
                    a.calc = calc
                    calc.reset()
                    cap.clear()
                    E[key] = float(a.get_potential_energy())
                    r = cap[-1]
                    Eq[key] = float(r.energy)
                    st = [int(round(float(s))) for s in r.states.tolist()]
                    if key == ("undoped", "non-wetting"):
                        pion = float(r.populations.detach()[AU, st.index(1)].max())
                    if key == ("doped", "wetting"):
                        pm = float(r.populations.detach()[AU, st.index(-1)].sum())
            except Exception as err:                                   # noqa: BLE001
                print(f"  {label:22s} ERROR {type(err).__name__}: {str(err)[:150]}")
                continue
            t5 = lambda d: 1e3 * ((d[("doped", "wetting")] - d[("doped", "non-wetting")])
                                  - (d[("undoped", "wetting")] - d[("undoped", "non-wetting")]))
            wd = 1e3 * (E[("doped", "wetting")] - E[("doped", "non-wetting")])
            wu = 1e3 * (E[("undoped", "wetting")] - E[("undoped", "non-wetting")])
            print(f"  {label:22s} T5 {t5(E):+8.1f} (E_q {t5(Eq):+8.1f})  wet doped {wd:+7.1f} undoped {wu:+7.1f}  "
                  f"p_ion {pion:.3f}  p(Au-) {pm:.3f}   changed: {sorted(changed)}", flush=True)
        mod.redox = base


if __name__ == "__main__":
    main()
