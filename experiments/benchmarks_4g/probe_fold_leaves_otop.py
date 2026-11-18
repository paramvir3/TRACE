"""Every fold leaf that uses the O-top donor, per configuration, on the doped and undoped upright references
(2026-11-02; theory note §4c): the contact Au is the one 2.26 A from the O-top O (the "bottom" label printed is the
smaller-z Au, the far one).  Usage (repo root): python experiments/benchmarks_4g/probe_fold_leaves_otop.py MODEL.pt"""
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
for key in (("doped", "non-wetting"), ("undoped", "non-wetting")):
    at = refs[key].copy(); at.info = {"charge": 0.0}; at.calc = calc; calc.reset(); mod._downfold_last = None
    at.get_potential_energy()
    rec = mod._downfold_last or {}
    cfg = mod._ensemble_last["configs"]
    au = list(AU); z = at.positions[:, 2]; bottom = au[int(np.argmin(z[au]))]
    otop = int(np.argmin([np.linalg.norm(at.positions[i] - at.positions[bottom]) if at.numbers[i] == 8 else 99 for i in range(len(at))]))
    print(f"=== {key}: O-top atom {otop} at {np.linalg.norm(at.positions[otop] - at.positions[bottom]):.3f} A from the bottom Au {bottom}; keys {sorted(rec.keys())}")
    print("   folded_states:", rec.get("folded_states") if rec.get("folded_states") is None or len(rec.get("folded_states")) < 12 else f"{len(rec['folded_states'])} rows")
    print("   promoted:", rec.get("promoted"))
    full = rec.get("all_leaves_full") or []
    for ci in range(cfg.shape[0]):
        L = [x for x in full if x[0] == ci]
        near = [x for x in L if x[1] == otop]
        print(f"   config {ci} Au {[int(cfg[ci][a]) for a in au]}: {len(L)} leaves; with the O-top donor: {len(near)}")
        for x in sorted(near, key=lambda x: x[4])[:6]:
            print(f"      O {x[1]} -> Au {x[2]} ({'bottom' if x[2] == bottom else 'top'}), kind {x[3]}, Delta {x[4]:+.3f}, v {x[5]:+.3f}")
