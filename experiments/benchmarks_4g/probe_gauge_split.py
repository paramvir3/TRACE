"""Where does the seed dependence of T5 live?  Proposition 6 test (frozen weights; 2026-10-19; reported).

Usage: python probe_gauge_split.py MODEL.pt [...]

E = E_s + E_q: E_q the electrostatics/redox module's energy (RedoxResult.energy), E_s the rest (short-range network
and atomic references, which cancel in T5).  At the four T5 reference geometries (eval_aumgo.references) this prints
T5 = (E_dw - E_dn) - (E_uw - E_un) split into its E_s and E_q parts, and the two Au atoms' learned shifts
Delta(+1), Delta(-1) = state_energies - E_Z (eV) with the populations p(+1), p(-1).  Proposition 6 predicts that two
seeds on different points of the gauge orbit differ by large, opposite E_s and E_q parts; a difference carried by
one part alone is not the gauge.
"""
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
from trace_redox.diabatic import formal_ladder  # noqa: E402
from eval_aumgo import AU, references  # noqa: E402


def main():
    refs = references()
    for model in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(2)
        cap = []
        calc.model.electrostatics.register_forward_hook(lambda m, i, o: cap.append(o[1]))
        E, Eq, rows = {}, {}, {}
        for key, atoms in refs.items():
            a = atoms.copy()
            a.info = dict(atoms.info)
            a.calc = calc
            calc.reset()
            cap.clear()
            E[key] = float(a.get_potential_energy())
            r = cap[-1]
            Eq[key] = float(r.energy)
            states = [int(round(float(s))) for s in r.states.tolist()]
            _, E_Z, _, _ = formal_ladder(torch.as_tensor(a.numbers), int(calc.model.electrostatics.redox.max_charge))
            pad = len(states) - int(E_Z.shape[1])                    # band states (-2) prepended, as the module does
            E_Z = torch.cat([torch.full((E_Z.shape[0], pad), float("inf"), dtype=E_Z.dtype), E_Z], dim=1)
            shift = (r.state_energies.detach() - E_Z.to(r.state_energies.dtype)).numpy()
            pops = r.populations.detach().numpy()
            ip, im = states.index(1), states.index(-1)
            rows[key] = [(shift[j, ip], shift[j, im], pops[j, ip], pops[j, im]) for j in AU]
        t5 = lambda d: (d[("doped", "wetting")] - d[("doped", "non-wetting")]) - (d[("undoped", "wetting")] - d[("undoped", "non-wetting")])
        Es = {k: E[k] - Eq[k] for k in E}
        print(f"=== {os.path.basename(os.path.dirname(model))}")
        print(f"  T5 = {1e3 * t5(E):+.1f} meV = E_s {1e3 * t5(Es):+.1f} + E_q {1e3 * t5(Eq):+.1f}   (DFT -931.6)")
        for k in refs:
            print(f"  {k[0]:8s} {k[1]:12s} E_s {E[k] - Eq[k]:+12.4f}  E_q {Eq[k]:+10.4f} eV;  Au shifts D(+1), D(-1) / "
                  "p(+1), p(-1): " + "; ".join(f"{a:+.3f} {b:+.3f} / {c:.3f} {d:.3f}" for a, b, c, d in rows[k]))


if __name__ == "__main__":
    main()
