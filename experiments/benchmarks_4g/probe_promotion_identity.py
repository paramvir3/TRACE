# Written on 2026-10-30 (CHANGELOG Round 30, theory note 4c); run from the repo root.
"""The re-partition identity in the module on a failure dump: the ground energy after promotion against the ground of
the unpromoted explicit-plus-folded block (the first exact-block call with promote=False, strict=False)."""
import sys, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
from flashace.calculator import TransformersACECalculator
from trace_redox import RedoxConfig
import trace_redox.downfold as F
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
mod = calc.model.electrostatics
d = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
mod.load_state_dict(d["state"], strict=False); mod.redox = RedoxConfig(**d["redox"]); mod.train(False)
original = F.dressing_exact; state = {"n": 0, "full": None}
def spy(*a, **kw):
    state["n"] += 1
    if state["n"] == 1:
        kw2 = dict(kw); kw2.update(promote=False, strict=False)
        _, rec = original(*a, **kw2)
        state["full"] = (rec["ground"], rec["explicit_weight"])
    return original(*a, **kw)
F.dressing_exact = spy
try:
    with torch.no_grad():
        mod(*d["args"], **d["kwargs"])
finally:
    F.dressing_exact = original
H = mod._ensemble_last["H"].detach()
g = float(torch.linalg.eigvalsh(H)[0])
print(f"unpromoted full block: ground {state['full'][0]}, explicit weight {state['full'][1]}; promoted ({len(mod._ensemble_last['promoted'])} states): ground {g:.6f} eV; difference {g - state['full'][0][0]:+.2e} eV", flush=True)
