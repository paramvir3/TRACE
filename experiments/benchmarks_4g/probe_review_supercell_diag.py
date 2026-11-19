"""Review check: is the frame configuration's own energy extensive, and the in-phase excitation's?  Undoped references,
1x1 against 2x1 (frozen weights).  Splits E(2x1) - 2 E(1x1) into the part outside the ensemble (E - lam_min), the frame
diagonal and the mixing energy (lam_min - frame diagonal)."""
import sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references
refs = references()
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu"); torch.set_num_threads(4)
mod = calc.model.electrostatics
for key in (("undoped", "non-wetting"), ("undoped", "wetting")):
    out = {}
    for rep in ((1, 1, 1), (2, 1, 1)):
        a = refs[key].copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
        a.calc = calc; calc.reset(); mod._ensemble_last = None
        e = float(a.get_potential_energy()); last = mod._ensemble_last
        cfg = last["configs"].to(torch.long); H = last["H"].detach(); H = 0.5 * (H + H.T)
        lam = float(torch.linalg.eigvalsh(H)[0]); d = torch.diagonal(H); i0 = int(torch.argmin(d))
        n = len(a) // 110
        au = [108 + 110 * k for k in range(n)] + [109 + 110 * k for k in range(n)]
        rows = {tuple(int(cfg[i, j]) for j in sorted(au)): i for i in range(cfg.shape[0])
                if all(int(cfg[i, j]) == int(cfg[i0, j]) for j in range(cfg.shape[1]) if j not in au)}
        out[rep] = (e, lam, float(d[i0]), {k: float(d[i]) for k, i in rows.items()})
    (e1, l1, f1, r1), (e2, l2, f2, r2) = out[(1, 1, 1)], out[(2, 1, 1)]
    print(f"== {key}")
    print(f"   total            E(2x1) - 2 E(1x1) = {1e3 * (e2 - 2 * e1):+10.3f} meV")
    print(f"   outside ensemble (E - lam_min)     = {1e3 * ((e2 - l2) - 2 * (e1 - l1)):+10.3f} meV")
    print(f"   frame diagonal                     = {1e3 * (f2 - 2 * f1):+10.3f} meV")
    print(f"   mixing (lam_min - frame diagonal)  = {1e3 * ((l2 - f2) - 2 * (l1 - f1)):+10.3f} meV   [1x1: {1e3 * (l1 - f1):+.3f} per cell; 2x1: {1e3 * (l2 - f2) / 2:+.3f} per cell]")
    for k1, v1 in sorted(r1.items()):
        if v1 == f1: continue
        k2 = tuple(x for x in k1 for _ in range(2))            # sorted(au) interleaves the copies: 108, 109, 218, 219
        k2 = (k1[0], k1[1], k1[0], k1[1])
        if k2 in r2:
            print(f"   in-phase excitation Au {k1}: 1x1 {v1 - f1:+.4f} eV x 2 = {2 * (v1 - f1):+.4f}; 2x1 {r2[k2] - f2:+.4f} eV; difference {1e3 * ((r2[k2] - f2) - 2 * (v1 - f1)):+.1f} meV")
        else:
            print(f"   in-phase excitation Au {k1}: 1x1 {v1 - f1:+.4f} eV; not in the 2x1 selection")
