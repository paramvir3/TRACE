"""Review check (2026-11-10), frozen weights: where does the model's ground state put the doped electron?
D31's M4 reads a DIAGONAL energy difference (near - far Au).  With a two-configuration H and a 4-5 eV eigen-gap, the
ground state's weights are 1/2 (1 +- Delta / gap).  Print the configurations (Au entries), the diagonal, the ground
state's weights, and the Au charges from the density matrix."""
import sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references, AU
refs = references()
for model in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=model, device="cpu"); torch.set_num_threads(4)
    mod = calc.model.electrostatics
    print("===", model.split("/")[-2][-22:], flush=True)
    for key in (("doped", "wetting"), ("doped", "non-wetting"), ("undoped", "non-wetting")):
        at = refs[key]; a = at.copy(); a.info = {"charge": 0.0}; a.calc = calc; calc.reset(); mod._ensemble_last = None
        a.get_potential_energy()
        last = mod._ensemble_last; H = last["H"].detach(); H = 0.5 * (H + H.T); cfg = last["configs"]
        lam, V = torch.linalg.eigh(H); w = (V[:, 0] ** 2).numpy()
        z = a.positions[list(AU), 2]; zs = a.positions[:108, 2]
        print(f"  {key[0]} {key[1]}: configs shape {tuple(cfg.shape)}; Au z {np.round(z, 3).tolist()} (slab z range {zs.min():.2f}-{zs.max():.2f})")
        nonzero_cols = [int(c) for c in range(cfg.shape[1]) if len(set(cfg[:, c].tolist())) > 1]
        for i in range(cfg.shape[0]):
            print(f"     cfg {i}: varying units {nonzero_cols} = {[float(cfg[i, c]) for c in nonzero_cols]}  diag {float(H[i, i]) - float(torch.diagonal(H).min()):+.4f} eV  ground-state weight {w[i]:.4f}")
        offd = H - torch.diag(torch.diagonal(H))
        print(f"     largest |off-diagonal| {float(offd.abs().max()):.4f} eV; eigen-gap {float(lam[1] - lam[0]):.4f} eV")
