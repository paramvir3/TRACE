"""How close the trained couplings come to the Harrison bound on the AuMgO data (SPEC 27; 2026-10-18, reported).

Usage: python probe_coupling_bound.py MODEL.pt [N]
For N test structures per doping (and the four T5 references): max |tau_raw| / B(r), B = 3.24 hbar^2 / (m (r^2 + r0^2)), r0 = 0.5 A, and
the largest relative change of tau that the bound tanh would make, max (|tau_raw| - |B tanh(tau_raw / B)|) / |tau_raw|."""
import os, sys, warnings
import torch
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", ".."))); sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)
from ase.io import read  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from eval_aumgo import references  # noqa: E402

model = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
calc = TransformersACECalculator(model_path=model, device="cpu"); torch.set_num_threads(2)
coup = calc.model.electrostatics.coupling
orig = coup.edge_values
rec = []
def ev(x, ei, r):
    raw = orig(x, ei, r)
    B = 3.24 * 7.619964 / (r ** 2 + 0.25)
    k = int((raw / B).abs().argmax())
    rec.append((float((raw / B).abs().max()), float(((raw.abs() - (B * torch.tanh(raw / B)).abs()) / raw.abs().clamp_min(1e-300)).max()),
                float(raw.abs().max()), float(r[k])))
    return raw
coup.edge_values = ev
test = read(os.path.join(HERE, "data", "AuMgO_test.xyz"), ":")
sel = [a for a in test if (a.numbers == 13).any()][:N] + [a for a in test if not (a.numbers == 13).any()][:N] + list(references().values())
for a in sel:
    a = a.copy(); a.info = dict(a.info); a.calc = calc; calc.reset(); a.get_potential_energy()
worst = max(rec, key=lambda t: t[0])
print(f"{os.path.basename(os.path.dirname(model))}: {len(sel)} structures; max |tau_raw|/B = {worst[0]:.4f} "
      f"(|tau| {worst[2]:.3f} eV at r = {worst[3]:.3f} A); largest relative change by the bound {max(t[1] for t in rec):.2e}")
