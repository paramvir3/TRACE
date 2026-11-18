# Written on 2026-10-30 (the derivative checks of the fold on a real structure: CHANGELOG Round 29); run from the repo root.
"""The training's mixed second derivative on a real structure: d F[k, a] / d theta for one redox-head weight, by
autograd (training=True, create_graph) and by central finite differences of the force (first derivatives only),
with the fold on (exact block, Pauli, c = 0.85).  Usage: python mixed_fd.py MODEL.pt INDEX"""
import sys, dataclasses, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
from flashace.calculator import TransformersACECalculator
from ase.io import read
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
model = calc.model; mod = model.electrostatics
mod.redox = dataclasses.replace(mod.redox, downfold_transfer=True, downfold_hop_ordering="pauli", downfold_block="exact",
                                downfold_scale=0.85, downfold_free_axes=False, downfold_anisotropy=0.0, downfold_crystal_field=0.0)
at = read("experiments/benchmarks_4g/data/AuMgO_train.xyz", sys.argv[2])
captured = {}; orig = model.forward
def capture(data, *a, **k):
    captured["data"] = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in data.items()}; return orig(data, *a, **k)
model.forward = capture; a = at.copy(); a.calc = calc; a.get_potential_energy(); model.forward = orig
data = captured["data"]
def forces(training):
    d = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in data.items()}
    E, F, S_, _ = model(d, training=training)
    return E, F
AU = [k for k, z in enumerate(data["z"].tolist()) if z == 79]
k, comp = AU[0], 2
model.zero_grad(set_to_none=True)
E, F = forces(True)
target = F[k, comp]
target.backward()
cands = [(n, p) for n, p in mod.named_parameters() if p.grad is not None and p.grad.abs().max() > 0]
name, par = max(cands, key=lambda np_: float(np_[1].grad.abs().max()))
idx = int(par.grad.abs().argmax()); g_auto = float(par.grad.reshape(-1)[idx])
rec = mod._downfold_last
print(f"structure {sys.argv[2]}: configs {int(mod._ensemble_last['configs'].shape[0])}, promoted {len(rec.get('promoted', []) or [])}, explicit weights {[round(w, 3) for w in rec.get('explicit_weight', [])]}", flush=True)
print(f"parameter {name}[{idx}]: autograd dF[{k},{comp}]/dtheta = {g_auto:+.8e}", flush=True)
for h in (1e-4, 1e-5):
    vals = []
    for sgn in (1, -1):
        with torch.no_grad():
            par.view(-1)[idx] += sgn * h
        _, Fh = forces(False)
        vals.append(float(Fh[k, comp]))
        with torch.no_grad():
            par.view(-1)[idx] -= sgn * h
    fd = (vals[0] - vals[1]) / (2 * h)
    print(f"  h {h:g}: finite difference {fd:+.8e}; relative difference {abs(g_auto - fd) / max(abs(fd), 1e-30):.2e}", flush=True)
