"""Au charges against DFT Hirshfeld over the AuMgO test set (2026-10-18; reported).

Usage: python probe_au_charges.py MODEL.pt [...]
Per class (doped/undoped x wetting/non-wetting): mean DFT and model total charges of the two Au, and the polarisation
p = (q_near - q_far) / 2 (near = the Au nearer the slab), with its RMSE against DFT over the class and the
correlation.  Hirshfeld and the model's total charges are different partitions; p, a difference, is the less
convention-bound comparison (DFT p is the target of Q1 / I1 / S3)."""
import os, sys, warnings
import numpy as np
import torch
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", ".."))); sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)
from ase.io import read  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from eval_aumgo import AU, geometry_class  # noqa: E402

test = read(os.path.join(HERE, "data", "AuMgO_test.xyz"), ":")
for model in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=model, device="cpu"); torch.set_num_threads(2)
    cap = []
    calc.model.electrostatics.register_forward_hook(lambda m, i, o: cap.append(o[1]))
    rows = {}
    for a0 in test:
        a = a0.copy(); a.info = dict(a0.info); a.calc = calc; calc.reset(); cap.clear(); a.get_potential_energy()
        r = cap[-1]
        zs = a.positions[:108, 2]
        d = [abs(a.positions[k, 2] - zs.mean()) for k in AU]           # near = the Au closer to the slab's mean plane
        order = [0, 1] if d[0] <= d[1] else [1, 0]
        qd = a0.arrays["hirshfeld_charges"][AU][order]
        qm = r.q[AU].detach().numpy()[order]
        key = ("doped" if (a.numbers == 13).any() else "undoped", geometry_class(a))
        rows.setdefault(key, []).append((qd, qm))
    print(f"=== {os.path.basename(os.path.dirname(model))}")
    for key in sorted(rows):
        qd = np.array([x[0] for x in rows[key]]); qm = np.array([x[1] for x in rows[key]])
        pd, pm = (qd[:, 0] - qd[:, 1]) / 2, (qm[:, 0] - qm[:, 1]) / 2
        r = float(np.corrcoef(pd, pm)[0, 1]) if np.std(pd) > 0 and np.std(pm) > 0 else float("nan")
        print(f"  {key[0]:8s} {key[1]:12s} n={len(qd):3d}  DFT near/far {qd[:, 0].mean():+.3f}/{qd[:, 1].mean():+.3f}  "
              f"model {qm[:, 0].mean():+.3f}/{qm[:, 1].mean():+.3f}  p DFT {pd.mean():+.3f} model {pm.mean():+.3f}  "
              f"p RMSE {np.sqrt(np.mean((pm - pd) ** 2)):.3f}  r {r:+.2f}  sum q DFT {qd.sum(1).mean():+.3f} model {qm.sum(1).mean():+.3f}")
