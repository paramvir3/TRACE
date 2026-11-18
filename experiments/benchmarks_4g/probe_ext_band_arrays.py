"""Extensivity, C5-e and C4-0 (2026-11-10): the band unit's arrays on the benchmark cells, saved for the design of the
extensive band form.  Frozen weights; nothing in the package is changed (``BandUnit.augment`` is wrapped in this
process only, and nothing is bypassed).

    python probe_ext_band_arrays.py MODEL.pt OUT.npz [nx ny ...]

Per doping, geometry and replication it prints the energy (or the refusal, by type) and saves eps [B], tau [B, B],
V [B, B] (SPEC 20's band interaction, rebuilt as in probe_ext_band_spectrum.py), the sites' atomic numbers, lambda_1,
J_11 and the number of excess electrons.  The arrays are captured before the configuration search, so a cell the
search refuses is saved too.  Writes nothing into the run directory."""
import dataclasses
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np
import torch
from trace_redox.band import _band_pair_corrections
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references

model_path, out_path = sys.argv[1], sys.argv[2]
reps = [(int(sys.argv[i]), int(sys.argv[i + 1]), 1) for i in range(3, len(sys.argv) - 1, 2)] or [(1, 1, 1), (2, 1, 1), (2, 2, 1)]
calc = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
mod.redox = dataclasses.replace(mod.redox, state_polarizability_stiffness="onsite", diabatic_space="clusters")
print(f"model {model_path.split('/')[-2]}; 'onsite' stiffness; clusters space; kT {mod.redox.kT}")
CAP = {}
original_augment = mod.band.augment


def augment(e_hat, states, b, W, L, K, frames, T, edge_index, edge_len, pairs=None):
    out = original_augment(e_hat, states, b, W, L, K, frames, T, edge_index, edge_len, pairs=pairs)
    CAP.update(e_hat=e_hat.detach(), states=states, W=W.detach(), L=L, frames=frames, pairs=pairs, out=out)
    return out


mod.band.augment = augment
refs = references()


def band_interaction():
    e_hat, states, W, L, f = CAP["e_hat"], CAP["states"], CAP["W"], CAP["L"], CAP["frames"].to(torch.long)
    sites = CAP["out"]["sites"]; U, S = int(e_hat.shape[0]), int(e_hat.shape[1]); first = int(round(float(states[0])))
    rows = torch.arange(U); col = lambda n: (n - first).clamp(0, S - 1)
    e_f, e_m1 = e_hat[rows, col(f)], e_hat[rows, col(f - 1)]
    e_m2 = torch.where(f - 2 >= L, e_hat[rows, col(f - 2)], torch.full_like(e_f, float("inf")))
    V = W[sites][:, sites].clone()
    onsite = (e_m2 - 2.0 * e_m1 + e_f + torch.diagonal(W))[sites]
    V = V - torch.diag(torch.diagonal(V)) + torch.diag(onsite)
    if CAP["pairs"] is not None and int(CAP["pairs"][0].numel()):
        V = V + _band_pair_corrections(CAP["pairs"], f, first, sites, U, S)[1].detach()
    return V


saved = {}
for doping in ("doped", "undoped"):
    for geometry in ("wetting", "non-wetting"):
        for rep in reps:
            a = refs[(doping, geometry)].copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
            a.calc = calc; calc.reset(); CAP.clear(); t0 = time.time()
            tag = f"{doping}_{geometry}_{rep[0]}x{rep[1]}"
            try:
                e = a.get_potential_energy()
                print(f"== {tag}: E {e:.9f} eV  ({time.time() - t0:.0f} s)", flush=True)
            except Exception as err:                                         # noqa: BLE001  (reported, not hidden)
                e = float("nan")
                print(f"== {tag}: REFUSED: {type(err).__name__}: {str(err)[:150]}  ({time.time() - t0:.0f} s)", flush=True)
            if "out" not in CAP or CAP["out"] is None:
                print("   no band arrays captured"); continue
            out = CAP["out"]; sites = out["sites"]
            z = torch.as_tensor(a.get_atomic_numbers())
            n_x = int(CAP["frames"].to(torch.long).sum())                    # Q = 0: electrons in no frame state
            for k, v in dict(eps=out["eps"], tau=out["tau"], V=band_interaction(), z=z[sites], lambda1=out["lambda1"],
                             J11=out["J11"], n_x=torch.tensor(n_x), energy=torch.tensor(e)).items():
                saved[f"{tag}/{k}"] = v.detach().cpu().numpy()
np.savez(out_path, **saved)
print(f"saved {len(saved)} arrays to {out_path}")
