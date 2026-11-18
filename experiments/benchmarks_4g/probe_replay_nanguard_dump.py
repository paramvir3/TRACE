# Written on 2026-10-30 (the diagnosis of step 22's first training step: CHANGELOG Round 28, theory note 4c); run from the repo root.
"""Replay the seed-42 failure: the run's own weights (nanguard.pt 'state') in the model, structure 76, the exact block,
the training-like double backward under anomaly detection, after the pairs_within fix."""
import sys, dataclasses, traceback, warnings, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
from flashace.calculator import TransformersACECalculator
from ase.io import read
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
model = calc.model; mod = model.electrostatics
dump = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(dump["state"], strict=False)
print(f"state loaded: missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
at = read("experiments/benchmarks_4g/data/AuMgO_train.xyz", "76")
captured = {}; orig = model.forward
def capture(data, *a, **k):
    captured["data"] = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in data.items()}; return orig(data, *a, **k)
model.forward = capture; a = at.copy(); a.calc = calc; a.get_potential_energy(); model.forward = orig
data = captured["data"]
for label, on, blk in (("exact", True, "exact"), ("per_state", True, "per_state"), ("fold off", False, "per_state")):
    mod.redox = dataclasses.replace(mod.redox, downfold_transfer=on, downfold_hop_ordering="pauli", downfold_block=blk,
                                    downfold_scale=0.85, downfold_free_axes=False, downfold_anisotropy=0.0, downfold_crystal_field=0.0)
    for mode in (False, True):
        model.zero_grad(set_to_none=True)
        d = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in data.items()}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with torch.autograd.set_detect_anomaly(mode):
                    E, F, S_, _ = model(d, training=True)
                    loss = E.sum() + (F ** 2).sum(); loss.backward()
            bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            rec = getattr(mod, "_downfold_last", {}) or {}
            print(f"[{label}, anomaly {mode}] E {float(E.sum()):.4f} |F|max {float(F.abs().max()):.3f} loss finite {bool(torch.isfinite(loss))}; non-finite grads {len(bad)}; "
                  f"fold gap {rec.get('mu_min')} weights {rec.get('explicit_weight')} ground {rec.get('ground')} spectrum {rec.get('spectrum')}", flush=True)
        except Exception as exc:
            print(f"[{label}, anomaly {mode}] RAISED {type(exc).__name__}: {str(exc)[:150]}", flush=True)
            for w in caught:
                t = str(w.message)
                if "Traceback of forward call" in t:
                    lines = [l.strip() for l in t.splitlines() if "trace_redox" in l or "trace_electrostatics" in l]
                    print("   forward frames:", " | ".join(l[:110] for l in lines[-5:]), flush=True)
