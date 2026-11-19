"""Extensivity, C4 (2026-11-10): the terms of the band sites' second-electron cost U_v (the diagonal of SPEC 20's
band interaction V) on the benchmark cells.  Frozen weights; nothing in the package is changed (``BandUnit.augment`` is
wrapped in this process only).

    python probe_ext_band_onsite.py MODEL.pt [nx ny ...]

SPEC 20:  U_v = [e_hat(f-2) - 2 e_hat(f-1) + e_hat(f)]_v + W_vv (+ the pair tables' second difference),  f the frame's
formal state.  Per doping and geometry it prints the range of each term over the Mg and the Al band sites.  Writes
nothing into the run directory."""
import dataclasses
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import torch
from trace_redox.band import _band_pair_corrections
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references

model_path = sys.argv[1]
reps = [(int(sys.argv[i]), int(sys.argv[i + 1]), 1) for i in range(2, len(sys.argv) - 1, 2)] or [(1, 1, 1)]
calc = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
mod.redox = dataclasses.replace(mod.redox, state_polarizability_stiffness="onsite", diabatic_space="clusters")
print(f"model {model_path.split('/')[-2]}; 'onsite' stiffness; clusters space")
CAP = {}
original_augment = mod.band.augment


def augment(e_hat, states, b, W, L, K, frames, T, edge_index, edge_len, pairs=None):
    out = original_augment(e_hat, states, b, W, L, K, frames, T, edge_index, edge_len, pairs=pairs)
    CAP.update(e_hat=e_hat.detach(), states=states, W=W.detach(), L=L, frames=frames, pairs=pairs, out=out)
    return out


mod.band.augment = augment
refs = references()
span = lambda x: f"{float(x.min()):+.3f} .. {float(x.max()):+.3f}" if int(x.numel()) else "-"
for doping in ("doped", "undoped"):
    for geometry in ("wetting", "non-wetting"):
        for rep in reps:
            a = refs[(doping, geometry)].copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
            a.calc = calc; calc.reset(); CAP.clear()
            try:
                a.get_potential_energy()
            except Exception as err:                                         # noqa: BLE001  (reported, not hidden)
                print(f"== {doping} {geometry} {rep[0]}x{rep[1]}: REFUSED: {type(err).__name__}: {str(err)[:100]}")
            if "out" not in CAP:
                continue
            e_hat, states, W, L, f = CAP["e_hat"], CAP["states"], CAP["W"], CAP["L"], CAP["frames"].to(torch.long)
            sites = CAP["out"]["sites"]; U, S = int(e_hat.shape[0]), int(e_hat.shape[1]); first = int(round(float(states[0])))
            rows = torch.arange(U); col = lambda n: (n - first).clamp(0, S - 1)
            ladder = (e_hat[rows, col(f - 2)] - 2.0 * e_hat[rows, col(f - 1)] + e_hat[rows, col(f)])[sites]
            absent = (f - 2 < L)[sites]
            w = torch.diagonal(W)[sites]
            pair = torch.zeros_like(w)
            if CAP["pairs"] is not None and int(CAP["pairs"][0].numel()):
                pair = torch.diagonal(_band_pair_corrections(CAP["pairs"], f, first, sites, U, S)[1].detach())
            z = torch.as_tensor(a.get_atomic_numbers())[sites]
            print(f"== {doping} {geometry} {rep[0]}x{rep[1]}: {int(sites.numel())} band sites; sites without an f-2 state: {int(absent.sum())}")
            for name, m in (("Mg", z == 12), ("Al", z == 13)):
                if not bool(m.any()):
                    continue
                print(f"   {name} ({int(m.sum())}): ladder second difference {span(ladder[m])}; W_vv {span(w[m])}; "
                      f"pair tables {span(pair[m])}; U = sum {span((ladder + w + pair)[m])} eV", flush=True)
