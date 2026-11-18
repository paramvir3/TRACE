"""Can the Au charge-state shifts alone reach DFT's T5?  A representability scan (frozen weights; 2026-10-19).

Usage: python probe_t5_shift_scan.py MODEL.pt [...]     (env SCAN="-1.5,-1,-0.5,0,0.5,1,1.5": offsets in eV)

T5 = (E_dw - E_dn) - (E_uw - E_un) is carried by E_q alone (probe_gauge_split.py: the short-range part contributes
< 5 meV, since it cannot see the dopant), and E_s's common wetting split can absorb any change of the two wetting
energies' common part.  So with the network frozen, the representable T5 under a change of the Au shifts is
T5(delta_+, delta_-) with constant offsets delta_+ on Au(+1) and delta_- on Au(-1) (added to the exact ladder E_Z,
equivalent to offsetting the learned shift; the tanh bound is bypassed, stated).  Configurations are re-selected at
every point (amplitude selection).  Printed per point: T5, the doped and undoped wetting energies of E_q, the upright
undoped dimer's p(Au+ Au-), and the doped flat p(Au-).  DFT: T5 -931.6, doped -2.7, undoped +928.9 meV.
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

import trace_redox.module as TM  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from eval_aumgo import AU, references  # noqa: E402

OFFSET = {"+": 0.0, "-": 0.0}
_ladder = TM.formal_ladder


def _shifted_ladder(species, max_charge):
    states, E, L, K = _ladder(species, max_charge)
    E = E.clone()
    au = (species.to(torch.long) == 79)
    for key, s in (("+", 1), ("-", -1)):
        col = [int(x) for x in states.tolist()].index(s)
        E[au, col] = E[au, col] + OFFSET[key]
    return states, E, L, K


def main():
    TM.formal_ladder = _shifted_ladder
    grid = [float(x) for x in os.environ.get("SCAN", "-1.5,-1,-0.5,0,0.5,1,1.5").split(",")]
    refs = references()
    for model in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(2)
        cap = []
        calc.model.electrostatics.register_forward_hook(lambda m, i, o: cap.append(o[1]))
        print(f"=== {os.path.basename(os.path.dirname(model))}   (delta_+, delta_-) eV -> T5, wet_doped, wet_undoped "
              f"(meV, E_q); p(Au+Au-) upright undoped; p(Au-) flat doped")
        for dp in grid:
            for dm in grid:
                OFFSET["+"], OFFSET["-"] = dp, dm
                Eq, pion, pminus = {}, None, None
                for key, atoms in refs.items():
                    a = atoms.copy()
                    a.info = dict(atoms.info)
                    a.calc = calc
                    calc.reset()
                    cap.clear()
                    a.get_potential_energy()
                    r = cap[-1]
                    Eq[key] = float(r.energy)
                    st = [int(round(float(s))) for s in r.states.tolist()]
                    pops = r.populations.detach()
                    if key == ("undoped", "non-wetting"):
                        pion = float(pops[AU, st.index(1)].max())
                    if key == ("doped", "wetting"):
                        pminus = float(pops[AU, st.index(-1)].sum())
                wd = 1e3 * (Eq[("doped", "wetting")] - Eq[("doped", "non-wetting")])
                wu = 1e3 * (Eq[("undoped", "wetting")] - Eq[("undoped", "non-wetting")])
                print(f"  ({dp:+.2f}, {dm:+.2f})  T5 {wd - wu:+8.1f}   wet_d {wd:+8.1f}   wet_u {wu:+8.1f}   "
                      f"p_ion {pion:.3f}   p(Au-) {pminus:.3f}", flush=True)


if __name__ == "__main__":
    main()
