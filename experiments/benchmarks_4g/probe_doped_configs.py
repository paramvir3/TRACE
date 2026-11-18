"""The doped references' explicit configurations with the band unit (2026-11-02; theory note §4c): Au and Al
charges, the band unit's state and its bounds (L, K).  Shows that the band holds at most two electrons, so the
three-donor cell forces Au2-.  Usage (repo root): python experiments/benchmarks_4g/probe_doped_configs.py MODEL.pt"""
import sys, warnings
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
warnings.simplefilter("ignore")
import numpy as np, torch
torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from flashace.calculator import TransformersACECalculator
from eval_aumgo import AU, references
refs = references()
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu"); torch.set_num_threads(3)
mod = calc.model.electrostatics
for key in (("doped", "non-wetting"), ("doped", "wetting")):
    at = refs[key].copy(); at.info = {"charge": 0.0}; at.calc = calc; calc.reset(); at.get_potential_energy()
    last = mod._ensemble_last; cfg = last["configs"]; K = last["K"]
    al = [i for i in range(len(at)) if at.numbers[i] == 13]
    n_at = len(at)
    print(f"=== {key}: units {cfg.shape[1]} (atoms {n_at}); K (max charge) of extra units {K[n_at:].tolist()}; Al {al}")
    for i in range(cfg.shape[0]):
        c = cfg[i]
        print(f"   config {i}: Au {[int(c[a]) for a in AU]}, Al {[int(c[a]) for a in al]}, extra units {c[n_at:].tolist()}, sum {int(c.sum())}")
    ba = mod._band_last
    if ba is not None:
        print("   band unit: L", ba["L"][-1].item() if hasattr(ba["L"], "shape") else ba["L"], "K", ba["K"][-1].item())
