# Written on 2026-10-30 (CHANGELOG Round 30, theory note 4c); run from the repo root.
"""Force consistency of the redox module through the promotion path on a failure dump: positions enter the module's
edge vectors consistently (edge_vec = edge_vec0 + dpos[src] - dpos[dst]); dE/dpos by autograd against central finite
differences, the node features held fixed (the module's own derivatives)."""
import sys, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
from flashace.calculator import TransformersACECalculator
from trace_redox import RedoxConfig
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
mod = calc.model.electrostatics
d = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
mod.load_state_dict(d["state"], strict=False); mod.redox = RedoxConfig(**d["redox"]); mod.train(False)
args = list(d["args"]); kwargs = dict(d["kwargs"])
h, pos0, ei, vec0 = args[0], args[1], args[2], args[3]
def energy(pos):
    dp = pos - pos0
    vec = vec0 + dp[ei[0]] - dp[ei[1]]
    a = list(args); a[1] = pos; a[3] = vec; a[4] = vec.norm(dim=1)
    out = mod(*a, **kwargs)
    return (out[0] if isinstance(out, (tuple, list)) else out).sum()
pos = pos0.clone().requires_grad_(True)
E = energy(pos)
print(f"promoted {len(mod._ensemble_last.get('promoted', []))}; explicit weights {[round(w, 3) for w in mod._downfold_last.get('explicit_weight', [])]}", flush=True)
g, = torch.autograd.grad(E, pos)
z = kwargs["data"]["z"]
AU = [k for k, zz in enumerate(z.tolist()) if zz == 79]; O = [k for k, zz in enumerate(z.tolist()) if zz == 8]
picks = [(AU[0], 2), (AU[1], 0), (O[0], 2)] + [(k, 2) for k in sorted(range(len(z)), key=lambda k: -float(g[k].abs().max()))[:2]]
for k, ax in picks:
    vals = []
    for sgn in (1, -1):
        p = pos0.clone(); p[k, ax] += sgn * 1e-5
        with torch.no_grad():
            vals.append(float(energy(p)))
    fd = (vals[0] - vals[1]) / 2e-5
    print(f"   dE/dx[{k},{ax}] (Z {int(z[k])}): autograd {float(g[k, ax]):+.8e}, FD {fd:+.8e}, relative {abs(float(g[k, ax]) - fd) / max(abs(fd), 1e-12):.1e}", flush=True)
