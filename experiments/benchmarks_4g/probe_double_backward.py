# Written on 2026-10-30 (the diagnosis of step 22's first training step: CHANGELOG Round 28, theory note 4c); run from the repo root.
"""The training's double backward on structure 76 (energy + force loss to the parameters) under anomaly detection,
with the fold off, per_state and exact; step-20 weights.  Names the first backward node that makes a non-finite
value, with its forward call."""
import sys, dataclasses, traceback, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
from flashace.calculator import TransformersACECalculator
from ase.io import read
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
model = calc.model; mod = model.electrostatics
at = read("experiments/benchmarks_4g/data/AuMgO_train.xyz", sys.argv[2] if len(sys.argv) > 2 else "76")
captured = {}
orig = model.forward
def capture(data, *a, **k):
    captured["data"] = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in data.items()}
    return orig(data, *a, **k)
model.forward = capture
a = at.copy(); a.calc = calc; a.get_potential_energy()
model.forward = orig
data = captured["data"]
for label, on, blk in (("fold off", False, "per_state"), ("per_state", True, "per_state"), ("exact", True, "exact")):
    mod.redox = dataclasses.replace(mod.redox, downfold_transfer=on, downfold_hop_ordering="pauli", downfold_block=blk,
                                    downfold_scale=0.85, downfold_free_axes=False, downfold_anisotropy=0.0, downfold_crystal_field=0.0)
    model.zero_grad(set_to_none=True)
    d = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in data.items()}
    try:
        with torch.autograd.set_detect_anomaly(True):
            E, F, S_, _ = model(d, training=True)
            loss = E.sum() + (F ** 2).sum()
            loss.backward()
        bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        print(f"[{label}] E {float(E.sum()):.4f}, |F|max {float(F.abs().max()):.3f}, loss finite {bool(torch.isfinite(loss))}, non-finite parameter grads: {len(bad)} {bad[:4]}", flush=True)
    except Exception as exc:
        msg = str(exc)
        print(f"[{label}] ANOMALY: {type(exc).__name__}: {msg[:160]}", flush=True)
        tb = traceback.format_exc().splitlines()
        hits = [l for l in tb if "trace_redox" in l or "trace_electrostatics" in l or "flashace" in l]
        print("   frames:", " | ".join(h.strip()[:90] for h in hits[-6:]), flush=True)
