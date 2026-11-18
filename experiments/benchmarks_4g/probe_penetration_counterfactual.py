# Diagnostic probe (2026-10-18; reported, not a criterion).  Usage (this directory): python probe_penetration_counterfactual.py RUN_DIR [...]
"""trace_redox SPEC 23 (carrier penetration) switched on a checkpoint's frozen weights.

At the four T5 references (unrelaxed): the Au free and total charges and E(wet) - E(non-wet) per doping, with the
checkpoint's own configuration and with carrier_penetration=True forced (and bond_partition, if the checkpoint has
it, kept).  The weights were trained without the term, so this shows its size and direction, not a result.
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
torch.set_num_threads(2)
from eval_aumgo import AU, references  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402


def main():
    refs = references()
    for run in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu")
        torch.set_num_threads(2)
        mod = calc.model.electrostatics
        base = mod.redox
        cap = []
        mod.register_forward_hook(lambda m, i, o: cap.append(o[1]))
        print(f"=== {os.path.basename(run.rstrip('/'))}")
        for label, cfg in (("as trained", base), ("penetration", dataclasses.replace(base, carrier_penetration=True))):
            mod.redox = cfg
            E, line = {}, []
            for key, a0 in refs.items():
                a = a0.copy()
                a.info = dict(a0.info)
                a.set_constraint()
                a.calc = calc
                calc.reset()
                cap.clear()
                E[key] = a.get_potential_energy()
                r = cap[-1]
                line.append(f"{key[0][:2]}-{key[1][:3]} free {np.round(r.free_charge.detach().numpy()[AU], 3).tolist()} "
                            f"total {np.round(r.q.detach().numpy()[AU], 3).tolist()}")
            print(f"  {label:12s}: " + "; ".join(line))
            print(f"  {'':12s}  E(wet) - E(non-wet): doped {1e3 * (E[('doped', 'wetting')] - E[('doped', 'non-wetting')]):+.0f}, "
                  f"undoped {1e3 * (E[('undoped', 'wetting')] - E[('undoped', 'non-wetting')]):+.0f} meV")
        mod.redox = base


if __name__ == "__main__":
    main()
