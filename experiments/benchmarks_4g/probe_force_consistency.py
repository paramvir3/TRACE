# Written on 2026-10-30 (the derivative checks of the fold on a real structure: CHANGELOG Round 29); run from the repo root.
"""First-derivative consistency with the fold on (exact block, Pauli, c = 0.85) on a real structure: dE/dtheta for one
coupling weight and dE/dx for the lower Au's z, autograd against central finite differences.  Usage: MODEL INDEX [PARAM IDX]"""
import sys, dataclasses, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
from flashace.calculator import TransformersACECalculator
from ase.io import read
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
model = calc.model; mod = model.electrostatics
import os
FOLD = os.environ.get("FOLD", "1") != "0"; BLOCK = os.environ.get("BLOCK", "exact")
mod.redox = dataclasses.replace(mod.redox, downfold_transfer=FOLD, downfold_block=BLOCK, downfold_hop_ordering="pauli",
                                downfold_scale=0.85, downfold_free_axes=False, downfold_anisotropy=0.0, downfold_crystal_field=0.0)
at = read("experiments/benchmarks_4g/data/AuMgO_train.xyz", sys.argv[2])
captured = {}; orig = model.forward
def capture(data, *a, **k):
    captured["data"] = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in data.items()}; return orig(data, *a, **k)
model.forward = capture; a = at.copy(); a.calc = calc; a.get_potential_energy(); model.forward = orig
data = captured["data"]
def energy(pos=None, training=False):
    d = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in data.items()}
    if pos is not None:
        d["pos"] = pos
    E, F, S_, _ = model(d, training=training)
    return E.sum(), F
pname = sys.argv[3] if len(sys.argv) > 3 else "coupling.pair.2.weight"; pidx = int(sys.argv[4]) if len(sys.argv) > 4 else 313
par = dict(mod.named_parameters())[pname]
model.zero_grad(set_to_none=True)
E, F = energy(training=True); E.backward()
g_auto = float(par.grad.reshape(-1)[pidx])
h = 1e-5; vals = []
for sgn in (1, -1):
    with torch.no_grad():
        par.view(-1)[pidx] += sgn * h
    vals.append(float(energy()[0]))
    with torch.no_grad():
        par.view(-1)[pidx] -= sgn * h
fd = (vals[0] - vals[1]) / (2 * h)
print(f"dE/d{pname}[{pidx}]: autograd {g_auto:+.8e}, finite difference {fd:+.8e}, relative {abs(g_auto - fd) / max(abs(fd), 1e-30):.2e}", flush=True)
_, F0 = energy()
O_ = [k for k, z in enumerate(data["z"].tolist()) if z == 8]; AU = [k for k, z in enumerate(data["z"].tolist()) if z == 79]
for k, ax in ((AU[0], 2), (AU[1], 0), (O_[0], 2)):
    for h in (1e-4, 1e-5):
        vals = []
        for sgn in (1, -1):
            p = data["pos"].clone(); p[k, ax] += sgn * h
            vals.append(float(energy(p)[0]))
        fdx = -(vals[0] - vals[1]) / (2 * h)
        print(f"FOLD={FOLD} BLOCK={BLOCK} force on atom {k} axis {ax}, h {h:g}: model {float(F0[k, ax]):+.8e}, -dE/dx FD {fdx:+.8e}, relative {abs(float(F0[k, ax]) - fdx) / max(abs(fdx), 1e-30):.2e}", flush=True)
