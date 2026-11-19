# Written on 2026-10-30 (CHANGELOG Round 30, theory note 4c); run from the repo root.
"""What produces seed 42's -292 eV at Au-Mg 2.0 A (fold off): the ensemble's configurations, their energies relative to
the frame, the populations, and the terms (probe_t5_first_order's evaluate)."""
import sys, dataclasses, numpy as np, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
import probe_t5_first_order as Q
from flashace.calculator import TransformersACECalculator
from eval_aumgo import AU, references
from trace_redox.constants import formal_frame
Q._install()
ref = references()[("undoped", "wetting")]
sym = ref.get_chemical_symbols(); MG = [k for k, s in enumerate(sym) if s == "Mg"]
mg = MG[int(np.argmin(ref.get_distances(AU[0], MG, mic=True)))]
n = np.array([0.0, 0.0, np.sign(ref.positions[AU[0], 2] - ref.positions[mg, 2])])
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu"); torch.set_num_threads(2)
mod = calc.model.electrostatics; mod.redox = dataclasses.replace(mod.redox, downfold_transfer=False, fragment_max_configurations=20000)
for h in (2.30, 2.00):
    at = ref.copy(); at.info = dict(ref.info)
    at.positions[AU[0]] = ref.positions[mg] + h * n; at.positions[AU[1]] = ref.positions[mg] + (h + 2.545) * n
    s = Q.evaluate(calc, at, 0.0)
    z = s["z"]; fr = formal_frame(z); cfg = s["configs"]; H = s["H"].detach()
    E = torch.diagonal(H); w, V = torch.linalg.eigh(H); p0 = V[:, 0] ** 2
    print(f"h {h:.2f}: E_total {s['E_tot']:.3f} eV; {cfg.shape[0]} configurations; lowest diagonal {float(E.min()):.2f}, ground {float(w[0]):.2f}", flush=True)
    order = torch.argsort(E)[:5]
    for c in order.tolist():
        lab = " + ".join(f"{sym[k]}{k}{int(cfg[c, k] - fr[k]):+d}" for k in range(len(fr)) if cfg[c, k] != fr[k]) or "frame"
        print(f"   config {c}: E - E_min {float(E[c] - E.min()):+.3f} eV, ground weight {float(p0[c]):.3f}: {lab}", flush=True)
    st = s["store"]
    big = sorted(((k, float(v)) for k, v in st.items() if isinstance(v, (int, float)) or (torch.is_tensor(v) and v.numel() == 1)), key=lambda kv: -abs(kv[1]))[:8]
    print("   largest stored scalar terms:", [(k, round(v, 2)) for k, v in big], flush=True)
    off = H - torch.diag(E)
    vals, idx = torch.topk(off.abs().flatten(), 8)
    seen = set()
    for v_, ij in zip(vals.tolist(), idx.tolist()):
        i, j = divmod(ij, H.shape[0])
        if (j, i) in seen: continue
        seen.add((i, j))
        diff = [k for k in range(len(fr)) if cfg[i, k] != cfg[j, k]]
        pair = ", ".join(f"{sym[k]}{k} {int(cfg[i, k] - fr[k]):+d}->{int(cfg[j, k] - fr[k]):+d}" for k in diff)
        dist = at.get_distance(diff[0], diff[1], mic=True) if len(diff) == 2 else float("nan")
        print(f"   |H_{i},{j}| = {v_:.2f} eV: {pair}; distance {dist:.2f} A", flush=True)
