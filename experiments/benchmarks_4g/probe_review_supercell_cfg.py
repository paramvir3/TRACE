"""Review check: which configurations does the ensemble hold in the cell and in its 2x1 supercell (frozen weights)?"""
import sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np, torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references
refs = references()
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu"); torch.set_num_threads(4)
mod = calc.model.electrostatics
for key in (("doped", "non-wetting"), ("undoped", "non-wetting")):
    for rep in ((1, 1, 1), (2, 1, 1)):
        a = refs[key].copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
        a.calc = calc; calc.reset(); mod._ensemble_last = None
        e = float(a.get_potential_energy()); last = mod._ensemble_last
        print(f"\n== {key} {rep}: E {e:.6f} eV; keys {sorted(last.keys())}")
        for k, v in last.items():
            if torch.is_tensor(v) and v.numel() <= 12: print(f"   {k}: {v.detach().tolist()}")
            elif torch.is_tensor(v): print(f"   {k}: shape {tuple(v.shape)}")
            elif not torch.is_tensor(v) and not isinstance(v, (list, tuple, dict)): print(f"   {k}: {v}")
        cfg = last["configs"].to(torch.long); H = last["H"].detach(); H = 0.5 * (H + H.T)
        lam, V = torch.linalg.eigh(H)
        vary = [j for j in range(cfg.shape[1]) if int(cfg[:, j].min()) != int(cfg[:, j].max())]
        print(f"   units {cfg.shape[1]}, configurations {cfg.shape[0]}, varying units {vary}")
        vals, counts = torch.unique(cfg[0], return_counts=True)
        print(f"   first configuration's charges (value: count): {dict(zip(vals.tolist(), counts.tolist()))}; sum {int(cfg[0].sum())}")
        order = torch.argsort(torch.diagonal(H))
        for i in order[:8].tolist():
            print(f"   cfg {i}: varying charges {[int(cfg[i, j]) for j in vary]}  E_diag - min {float(H[i, i] - torch.diagonal(H).min()):8.4f} eV  ground weight {float(V[i, 0] ** 2):.4f}")
