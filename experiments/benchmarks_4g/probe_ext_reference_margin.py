"""Is the reference n0 of the doped non-wetting 2x2 cell near a tie?  Per cluster: the diagonal energies of the
configurations above the reference, H_c[i,i] - H_c[0,0] (eV), sorted.  Usage: PYTHONPATH=. python experiments/benchmarks_4g/probe_ext_reference_margin.py MODEL.pt   (writes nothing)"""
import dataclasses, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np, torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
mod.redox = dataclasses.replace(mod.redox, state_polarizability_stiffness="onsite", diabatic_space="clusters", band_form="fermi")
refs = references()
for geometry in ("non-wetting", "wetting"):
    a = refs[("doped", geometry)].copy() * (2, 2, 1); a.info = {"charge": 0.0}; a.calc = calc; calc.reset()
    e = a.get_potential_energy()
    last = mod._ensemble_last
    print(f"{geometry} 2x2: E/copy {e / 4:.9f} eV; clusters {[int(S.shape[0]) for S in last['clusters']]}")
    for c, H in enumerate(last["hamiltonians"]):
        d = (torch.diagonal(H) - H[0, 0]).detach().numpy()
        off = (H - torch.diag(torch.diagonal(H))).abs().max().item()
        print(f"   cluster {c}: diag - E0 = {np.array2string(np.sort(d), precision=6)} eV; max|off-diagonal| {off:.4e} eV")
