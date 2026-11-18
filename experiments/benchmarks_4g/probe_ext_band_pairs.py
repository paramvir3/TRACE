"""Extensivity, C4 stage 2 (2026-11-10): what the pair tables handed to the band unit hold at the frame states.
SPEC 20's d_eps uses Xi(f_v - 1, f_w) as the whole change of a pair when v takes one band electron, which is exact
only if Xi(f_v, f_w) = 0.  Frozen weights; ``BandUnit.augment`` is wrapped in this process only; writes nothing.

    python experiments/benchmarks_4g/probe_ext_band_pairs.py MODEL.pt"""
import dataclasses
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references

calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
mod.redox = dataclasses.replace(mod.redox, state_polarizability_stiffness="onsite", diabatic_space="clusters")
CAP = {}
original = mod.band.augment


def augment(e_hat, states, b, W, L, K, frames, T, edge_index, edge_len, pairs=None):
    out = original(e_hat, states, b, W, L, K, frames, T, edge_index, edge_len, pairs=pairs)
    CAP.update(states=states, frames=frames.to(torch.long), pairs=pairs, out=out, L=L, K=K)
    return out


mod.band.augment = augment
refs = references()
for key in (("doped", "wetting"), ("doped", "non-wetting"), ("undoped", "wetting"), ("undoped", "non-wetting")):
    a = refs[key].copy(); a.set_constraint(); a.info = {"charge": 0.0}; a.calc = calc; calc.reset(); CAP.clear()
    a.get_potential_energy()
    pairs, f, out = CAP["pairs"], CAP["frames"], CAP["out"]
    first = int(round(float(CAP["states"][0])))
    sites = out["sites"]; U = int(f.numel())
    is_site = torch.zeros(U, dtype=torch.bool); is_site[sites] = True
    if pairs is None or not int(pairs[0].numel()):
        print(f"== {key}: no pair tables"); continue
    pi, pj, Xi = pairs[0], pairs[1], pairs[2].detach()
    p = torch.arange(int(pi.numel()))
    at_frame = Xi[p, f[pi] - first, f[pj] - first]
    kinds = (is_site[pi] & is_site[pj], is_site[pi] ^ is_site[pj], ~is_site[pi] & ~is_site[pj])
    print(f"== {key}: {int(pi.numel())} pair tables {tuple(Xi.shape)}; site-site {int(kinds[0].sum())}, site-other "
          f"{int(kinds[1].sum())}, other-other {int(kinds[2].sum())}; max|Xi(f_i, f_j)| {float(at_frame.abs().max()):.3e} eV; "
          f"max|Xi| {float(Xi[torch.isfinite(Xi)].abs().max()):.3e} eV; non-finite entries {int((~torch.isfinite(Xi)).sum())}")
    Lr, Kr = out["L"], out["K"]
    print(f"   band sites {int(sites.numel())}; site atoms' bounds after augment: L - f in "
          f"[{int((Lr[sites] - f[sites]).min())}, {int((Lr[sites] - f[sites]).max())}], K - f in "
          f"[{int((Kr[sites] - f[sites]).min())}, {int((Kr[sites] - f[sites]).max())}]; band unit L {int(Lr[-1])}, K {int(Kr[-1])}", flush=True)
