"""The learned O-hole shift and what it does to the wetting energies (2026-10-23; a probe, frozen weights).

Usage (repo root): python experiments/benchmarks_4g/probe_hole_shift.py MODEL.pt [...]

Measured (probe_hole_terms.py, step 16 s43): the clean-slab IP is +7.21 eV against PBE's +4.80, and +2.151 eV of it
is the learned shift of the O- state over the O2- frame; without it the explicit terms give +5.06 eV.  Hypothesis:
the shift also suppresses O -> Au donation (Au- + O- configurations) at the O-top site, under-stabilising the upright
dimer, which is the measured common wetting error (flat too stable by 0.24-0.45 eV in doped and undoped cells).

1. Delta_O = shift(O, -1) - shift(O, -2) per O atom, from the module's state energies (state_energies - E_Z, exact).
2. Counterfactual (frozen weights): Delta_O subtracted from every O's state -1 in the configuration energies right
   before the solve (a wrapper of RedoxElectrostatics._diabatic_solution; pass 1 reads Delta_O, pass 2 applies it).
   The whole model's energies of the four T5 references give wet_u, wet_d and T5 with and without the shift.
Pre-registered reading: if zeroing the shift moves wet_u by < 100 meV, the hole shift is not the wetting lever
(hypothesis falsified); if it raises wet_u toward DFT's +929 by more, it supports it (direction and size only: the
other weights were trained with the shift).
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

import trace_redox.module as M  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_redox.diabatic import formal_ladder  # noqa: E402
from eval_aumgo import references  # noqa: E402

CF = {"on": False, "delta": None}
_sol = M.RedoxElectrostatics._diabatic_solution


def _wrapped(self, e_hat, states, b, W, L, K, Q, T=None, **kwargs):
    if CF["on"] and CF["delta"] is not None:
        col = int((states == -1).nonzero().flatten()[0])
        d = CF["delta"]
        e_hat = e_hat.clone()
        e_hat[: d.numel(), col] = e_hat[: d.numel(), col] - d.to(e_hat)
    return _sol(self, e_hat, states, b, W, L, K, Q, T=T, **kwargs)


M.RedoxElectrostatics._diabatic_solution = _wrapped


def energy(calc, atoms, cf):
    a = atoms.copy()
    a.info = dict(atoms.info)
    a.calc = calc
    calc.reset()
    cap = []
    hook = calc.model.electrostatics.register_forward_hook(lambda m, i, o: cap.append(o[1]))
    CF["on"] = cf
    E = float(a.get_potential_energy())
    CF["on"] = False
    hook.remove()
    return E, cap[-1]


def hole_shift(calc, atoms):
    """Delta_O [n] (zero on non-O atoms) from a normal evaluation."""
    _, r = energy(calc, atoms, False)
    z = torch.as_tensor(atoms.numbers)
    mod = calc.model.electrostatics
    states, E_Z, _, _ = formal_ladder(z, int(mod.redox.max_charge))
    st = r.state_energies.detach()[: len(z)]
    pad = st.shape[1] - E_Z.shape[1]
    E_Z = torch.cat([torch.full((len(z), pad), float("inf")), E_Z], 1)
    sts = r.states.detach() if hasattr(r, "states") else None
    states_all = sts if sts is not None else torch.cat([torch.arange(-2, -2 + pad), states])
    shift = st - E_Z
    cm1 = int((states_all == -1).nonzero().flatten()[0])
    cm2 = int((states_all == -2).nonzero().flatten()[0])
    d = torch.where(z == 8, shift[:, cm1] - shift[:, cm2], torch.zeros(len(z)))
    return torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    refs = references()
    for model in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(2)
        mod = calc.model.electrostatics
        mod.redox = dataclasses.replace(mod.redox, fragment_max_configurations=20000)
        E = {}
        for key, at in refs.items():
            d = hole_shift(calc, at)
            CF["delta"] = d
            E[(key, False)] = energy(calc, at, False)[0]
            E[(key, True)] = energy(calc, at, True)[0]
            o = d[torch.as_tensor(at.numbers) == 8]
            print(f"  {key}: Delta_O median {float(o.median()):+.3f} eV (min {float(o.min()):+.3f}, max {float(o.max()):+.3f})",
                  flush=True)
        name = os.path.basename(os.path.dirname(model))
        for cf in (False, True):
            wu = E[(("undoped", "wetting"), cf)] - E[(("undoped", "non-wetting"), cf)]
            wd = E[(("doped", "wetting"), cf)] - E[(("doped", "non-wetting"), cf)]
            print(f"=== {name} {'O-hole shift zeroed' if cf else 'as trained'}: wet_u {wu * 1000:+.1f}, wet_d "
                  f"{wd * 1000:+.1f}, T5 {(wd - wu) * 1000:+.1f} meV  (DFT +928.9 / -2.7 / -931.6; references, unrelaxed)",
                  flush=True)


if __name__ == "__main__":
    main()
