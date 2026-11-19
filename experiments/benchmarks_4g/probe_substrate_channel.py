"""Does the substrate's image response survive when it moves from bond charge flow to ionic dipoles?
(frozen weights; 2026-10-19; a counterfactual of step 14's retraining, substrate polarisability only)

Usage (repo root): python experiments/benchmarks_4g/probe_substrate_channel.py MODEL.pt

Step 13's models carry ~80 % of MgO's electronic susceptibility in the SQE bond charge flow: SPEC 22 (bond partition)
removes that channel and eps_inf falls 3.16 -> 1.40 (probe_eps_bound.py --partition).  Physically MgO's eps_inf is
the ions' dipoles (Clausius-Mossotti with alpha(O2-) ~ 1.7 A^3 gives 2.95; ESTABLISHED).  Step 14 retrains under
the eps_inf prior, which must move the response into the dipoles.  This probe does that move alone: every O and Mg
dipole stiffness A_mu -> A_mu / lambda (alpha -> lambda alpha; the Gaussian widths follow from the stiffness,
SPEC 12), lambda bisected so that eps_inf(MgO, SPEC 22 on) equals the checkpoint's own eps_inf without SPEC 22;
then T5 split exactly into the terms of the configuration energy (probe_t5_terms.py) at step 14's settings.
"""
import dataclasses
import os
import sys
import warnings

import torch

warnings.simplefilter("ignore")
sys.path.insert(0, ".")
sys.path.insert(0, "experiments/benchmarks_4g")
torch.set_default_dtype(torch.float64)
from ase.build import bulk  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
import trace_electrostatics.heads as HD  # noqa: E402
from trace_electrostatics.observables import dielectric_tensor  # noqa: E402
import probe_t5_terms as PT  # noqa: E402
from eval_aumgo import references  # noqa: E402

SCALE = {"lambda": 1.0, "species": None}
_forward = HD.ElectrostaticHeads.forward


def scaled_forward(self, node_features, bond_index, bond_length=None, species=None):
    out = _forward(self, node_features, bond_index, bond_length, species=species)
    z = species if species is not None else SCALE["species"]
    if SCALE["lambda"] == 1.0 or z is None:
        return out
    sub = ((z == 8) | (z == 12)).to(out.params.A_mu.dtype)
    factor = 1.0 + (SCALE["lambda"] - 1.0) * sub                         # alpha -> lambda alpha on O and Mg
    return out._replace(params=dataclasses.replace(out.params, A_mu=out.params.A_mu / factor.view(-1, 1, 1)))


def eps_inf(calc):
    mod = calc.model.electrostatics
    atoms = bulk("MgO", "rocksalt", a=4.212, cubic=True).repeat((2, 2, 2))
    atoms.info = {"charge": 0.0}
    SCALE["species"] = torch.as_tensor(atoms.numbers)
    cap = []
    h = mod.register_forward_pre_hook(lambda m, a, k: cap.append((a, k)) or None, with_kwargs=True)
    a = atoms.copy()
    a.info = dict(atoms.info)
    a.calc = calc
    calc.reset()
    try:
        a.get_potential_energy()
    except Exception:                                                     # noqa: BLE001
        if not cap:
            raise
    h.remove()
    args, kwargs = cap[0]
    hh, pos, ei, vec, L = [x.detach() for x in args[:5]]
    keys = {k: kwargs[k] for k in ("pbc", "cell", "edge_shift") if k in kwargs}
    st = mod.bound_state(hh, pos.clone().requires_grad_(True), ei, vec, L, data=kwargs.get("data", {}), **keys)
    return float(dielectric_tensor(st.solve_result, st.assembly).detach().diagonal().mean())


def main():
    HD.ElectrostaticHeads.forward = scaled_forward
    PT._install()
    ck = sys.argv[1]
    calc = TransformersACECalculator(model_path=ck, device="cpu")
    torch.set_num_threads(2)
    mod = calc.model.electrostatics
    base = mod.redox
    target = eps_inf(calc)
    mod.redox = dataclasses.replace(base, bond_partition=True)
    lo, hi = 1.0, 30.0
    for _ in range(30):
        SCALE["lambda"] = 0.5 * (lo + hi)
        if eps_inf(calc) < target:
            lo = SCALE["lambda"]
        else:
            hi = SCALE["lambda"]
    SCALE["lambda"] = 0.5 * (lo + hi)
    got = eps_inf(calc)
    print(f"=== {ck.split('/')[-2]}: eps_inf {target:.3f} (as trained, charge flow); SPEC 22 frozen 1.x; dipoles x"
          f"{SCALE['lambda']:.3f} restore {got:.3f}")
    refs = references()
    for label, lam in (("SPEC 22, frozen alpha", 1.0), (f"SPEC 22, alpha(O, Mg) x {SCALE['lambda']:.2f}", SCALE["lambda"])):
        SCALE["lambda"] = lam
        R = {}
        for key, at in refs.items():
            SCALE["species"] = torch.as_tensor(at.numbers)
            R[key] = PT.terms(calc, at)
        keys = [k for k in ("screened", "shifts", "ladder", "self", "exclusion18", "band", "coherent") if k in R[key]]
        t5 = {k: 1e3 * ((R[("doped", "wetting")][k] - R[("doped", "non-wetting")][k])
                        - (R[("undoped", "wetting")][k] - R[("undoped", "non-wetting")][k])) for k in keys}
        total = 1e3 * sum(((R[("doped", "wetting")][k] - R[("doped", "non-wetting")][k])
                           - (R[("undoped", "wetting")][k] - R[("undoped", "non-wetting")][k])) for k in R[key])
        print(f"  {label:32s} T5_q {total:+8.1f}  | " + "  ".join(f"{k} {v:+.1f}" for k, v in t5.items()))
    mod.redox = base


if __name__ == "__main__":
    main()
