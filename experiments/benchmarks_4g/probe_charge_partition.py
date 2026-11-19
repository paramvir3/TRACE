"""Is the Au 'over-polarisation' a charge-partition artifact?  Model total charges against DFT Hirshfeld for every
element over N test structures (2026-10-18; reported).

Usage: python probe_charge_partition.py MODEL.pt [N]
Per element: mean Hirshfeld and model charge, and the least-squares slope of model on Hirshfeld through the origin
and with an intercept.  If the substrate ions carry the same model/Hirshfeld ratio as the Au polarisation, the Au
comparison measures the partition, not the physics."""
import os, sys, warnings
import numpy as np
import torch
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", ".."))); sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)
from ase.io import read  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

model = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
test = read(os.path.join(HERE, "data", "AuMgO_test.xyz"), ":")
sel = test[::max(1, len(test) // N)][:N]
calc = TransformersACECalculator(model_path=model, device="cpu"); torch.set_num_threads(2)
cap = []
calc.model.electrostatics.register_forward_hook(lambda m, i, o: cap.append(o[1]))
H, Mq, Z = [], [], []
for a0 in sel:
    a = a0.copy(); a.info = dict(a0.info); a.calc = calc; calc.reset(); cap.clear(); a.get_potential_energy()
    H.append(a0.arrays["hirshfeld_charges"]); Mq.append(cap[-1].q.detach().numpy()); Z.append(a0.numbers)
H, Mq, Z = np.concatenate(H), np.concatenate(Mq), np.concatenate(Z)
print(f"=== {os.path.basename(os.path.dirname(model))}: {len(sel)} structures")
for z, name in ((8, "O"), (12, "Mg"), (13, "Al"), (79, "Au")):
    m = Z == z
    if not m.any():
        continue
    h, q = H[m], Mq[m]
    slope0 = float(h @ q / (h @ h))
    A = np.stack([h, np.ones_like(h)], 1); (s1, c1), *_ = np.linalg.lstsq(A, q, rcond=None)
    print(f"  {name:2s} n={m.sum():5d}  Hirshfeld {h.mean():+.3f} (sd {h.std():.3f})  model {q.mean():+.3f} (sd {q.std():.3f})  "
          f"ratio of means {q.mean() / h.mean():+.2f}  slope through 0 {slope0:+.2f}  slope {s1:+.2f} (intercept {c1:+.3f})  "
          f"ratio of sd {q.std() / h.std():.2f}")
