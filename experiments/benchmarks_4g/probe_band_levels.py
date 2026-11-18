"""The band unit's one-electron spectrum on the doped T5 references (2026-11-02; theory note §4c): the sites' eps,
the module's lambda_1 against eigh, J_11, and the six lowest levels with their weight on the Al sites and their
participation number.  Usage (repo root): python experiments/benchmarks_4g/probe_band_levels.py MODEL.pt"""
import sys, warnings
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
warnings.simplefilter("ignore")
import numpy as np, torch
torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from flashace.calculator import TransformersACECalculator
from eval_aumgo import AU, references
refs = references()
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu"); torch.set_num_threads(3)
mod = calc.model.electrostatics
for key in (("doped", "non-wetting"), ("doped", "wetting")):
    at = refs[key].copy(); at.info = {"charge": 0.0}; at.calc = calc; calc.reset(); at.get_potential_energy()
    ba = mod._band_last
    sites = ba["sites"].tolist(); eps = ba["eps"].detach().numpy(); tau = ba["tau"].detach().numpy()
    h = np.diag(eps) - tau
    lam, vec = np.linalg.eigh(h)
    al = [k for k, s in enumerate(sites) if at.numbers[s] == 13]
    print(f"=== {key}: {len(sites)} band sites ({len(al)} Al); eps(Al) {np.round(eps[al], 3).tolist()}, eps(Mg) min/max {eps[[k for k in range(len(sites)) if k not in al]].min():.3f}/{eps.max():.3f} eV")
    print(f"   lambda1 module {float(ba['lambda1']):+.4f}, eigh {lam[0]:+.4f}; J11 {float(ba['J11']):+.4f} eV")
    for k in range(6):
        w = vec[:, k] ** 2
        print(f"   lambda{k+1} {lam[k]:+.4f} eV (gap to lambda1 {lam[k]-lam[0]:+.4f}); weight on Al {w[al].sum():.3f} [{', '.join(f'{x:.2f}' for x in w[al])}]; participation 1/sum w^2 = {1/np.sum(w**2):.1f} sites")
