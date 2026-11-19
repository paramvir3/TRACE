"""Extensivity, C4 stage 2 (2026-11-10): how much do the couplings between the atoms and the band unit carry?

    python probe_ext_band_coupling.py MODEL.pt

Frozen weights, the four 1x1 benchmark cells, 'onsite' stiffness and the clusters space.  ``BandUnit.augment`` is
wrapped in this process only: once unchanged, once with the band's row and column of T set to zero.  Prints the size
of T_band = sum_v T_uv phi_1(v) (SPEC 20), the sign of the atom-atom couplings, and the change of the energy and of
the forces when T_band is removed.  Writes nothing into the run directory."""
import dataclasses
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np
import torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references

model_path = sys.argv[1]
calc = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
mod.redox = dataclasses.replace(mod.redox, state_polarizability_stiffness="onsite", diabatic_space="clusters")
print(f"model {model_path.split('/')[-2]}; 'onsite' stiffness; clusters space")
original = mod.band.augment
STATE = {"zero": False}
CAP = {}


def augment(e_hat, states, b, W, L, K, frames, T, edge_index, edge_len, pairs=None):
    out = original(e_hat, states, b, W, L, K, frames, T, edge_index, edge_len, pairs=pairs)
    if out is not None:
        U = int(e_hat.shape[0])
        CAP.update(T=None if T is None else T.detach().clone(), Tb=out["T"][:U, U].detach().clone(), sites=out["sites"])
        if STATE["zero"]:
            T2 = out["T"].clone()
            T2[:U, U] = 0.0
            T2[U, :U] = 0.0
            out = dict(out, T=T2)
    return out


mod.band.augment = augment
refs = references()
for key in (("doped", "wetting"), ("doped", "non-wetting"), ("undoped", "wetting"), ("undoped", "non-wetting")):
    res = {}
    for zero in (False, True):
        STATE["zero"] = zero
        a = refs[key].copy(); a.set_constraint(); a.info = {"charge": 0.0}
        a.calc = calc; calc.reset()
        res[zero] = (a.get_potential_energy(), a.get_forces())
    z = a.get_atomic_numbers()
    T, Tb = CAP["T"], CAP["Tb"]
    nz = Tb.abs() > 0
    top = torch.argsort(Tb.abs(), descending=True)[:4].tolist()
    print(f"== {key}: atom-atom T: min {float(T.min()):+.4f}, max {float(T.max()):+.4f} eV, nonzero {int((T != 0).sum())}; "
          f"T_band nonzero on {int(nz.sum())} atoms, max |T_band| {float(Tb.abs().max()):.4e} eV; largest: "
          + ", ".join(f"Z{int(z[i])}:{float(Tb[i]):+.3e}" for i in top))
    dE = res[True][0] - res[False][0]
    dF = np.abs(res[True][1] - res[False][1]).max()
    print(f"   T_band removed: dE {1e3 * dE:+.6f} meV; max|dF| {dF:.3e} eV/A", flush=True)
