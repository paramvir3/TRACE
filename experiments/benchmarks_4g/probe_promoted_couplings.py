# Written on 2026-10-30 (CHANGELOG Round 30, theory note 4c); run from the repo root.
"""Seed 42's failure dump replayed with batch promotion: the explicit Hamiltonian after promotion, its largest
couplings between configurations (the atoms that differ, their distance), the ground energy, and the same with the
O-O couplings between promoted configurations removed (the fold's own approximation: no O-O hop among folded states)."""
import sys, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
from flashace.calculator import TransformersACECalculator
from trace_redox import RedoxConfig
from trace_redox.constants import formal_frame
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
mod = calc.model.electrostatics
d = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
mod.load_state_dict(d["state"], strict=False); mod.redox = RedoxConfig(**d["redox"]); mod.train(False)
with torch.no_grad():
    mod(*d["args"], **d["kwargs"])
last = mod._ensemble_last; H = last["H"].detach(); cfg = last["configs"]; prom = last.get("promoted", [])
z = d["kwargs"]["data"]["z"] if "data" in d["kwargs"] else None
fr = formal_frame(z) if z is not None else None
C = H.shape[0]; n_prom = len(prom); first_prom = C - n_prom
E = torch.diagonal(H); w = torch.linalg.eigvalsh(H)
print(f"{C} explicit configurations ({n_prom} promoted, from index {first_prom}); ground {float(w[0]):.3f}, lowest diagonal {float(E.min()):.3f} eV", flush=True)
off = H - torch.diag(E)
vals, idx = torch.topk(off.abs().flatten(), 12)
seen = set()
for v_, ij in zip(vals.tolist(), idx.tolist()):
    i, j = divmod(ij, C)
    if (j, i) in seen or i == j: continue
    seen.add((i, j))
    diff = [k for k in range(cfg.shape[1]) if cfg[i, k] != cfg[j, k]]
    tag = ("P-P" if i >= first_prom and j >= first_prom else "P-E" if (i >= first_prom) != (j >= first_prom) else "E-E")
    print(f"   |H_{i},{j}| = {v_:.3f} eV [{tag}]: atoms {diff} (Z {[int(z[k]) for k in diff] if z is not None else '?'})", flush=True)
# remove the O-O couplings between promoted configurations
H2 = H.clone()
for i in range(first_prom, C):
    for j in range(first_prom, C):
        if i == j: continue
        diff = [k for k in range(cfg.shape[1]) if cfg[i, k] != cfg[j, k]]
        if len(diff) == 2 and z is not None and all(int(z[k]) == 8 for k in diff):
            H2[i, j] = 0.0
print(f"ground without the O-O couplings among promoted configurations: {float(torch.linalg.eigvalsh(H2)[0]):.3f} eV", flush=True)
