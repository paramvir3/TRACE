"""Review check: separate the two candidate causes of the undoped supercell's energy change -- (i) the selection
truncating the product space, (ii) one H per cell making every periodic image take the cell's configuration.  Frozen
weights; the selection is switched to the plain energy window so that the cell and the 2x1 supercell can both hold
their whole (product) space: a window W in the cell against 2 W in the supercell."""
import sys, warnings, dataclasses, time
warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np, torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references
refs = references()
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu"); torch.set_num_threads(4)
mod = calc.model.electrostatics; base_cfg = mod.redox
for key in (("undoped", "non-wetting"), ("undoped", "wetting")):
    ref = {}
    for rep, arms in (((1, 1, 1), (("amplitude", 2.5), ("window", 2.5), ("window", 4.5), ("window", 8.0))),
                      ((2, 1, 1), (("amplitude", 2.5), ("window", 2.5), ("window", 5.0), ("window", 9.0), ("window", 16.0)))):
        for sel, W in arms:
            mod.redox = dataclasses.replace(base_cfg, diabatic_selection=sel, diabatic_window=W)
            a = refs[key].copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
            cells = rep[0] * rep[1]; a.calc = calc; calc.reset(); mod._ensemble_last = None
            t = time.perf_counter()
            try:
                e = float(a.get_potential_energy()) / cells
            except Exception as ex:
                print(f"{key} {rep} {sel} W={W}: REFUSED {type(ex).__name__}: {str(ex)[:150]}", flush=True); continue
            H = mod._ensemble_last["H"].detach(); H = 0.5 * (H + H.T); lam = torch.linalg.eigvalsh(H)
            if rep == (1, 1, 1): ref[(sel, W)] = e
            e0 = ref[("amplitude", 2.5)]
            print(f"{key[0]} {key[1]:12s} {rep} {sel:9s} W {W:4.1f} eV: n_cfg {H.shape[0]:4d}  E/cell - E(1x1, as trained) {(e - e0) * 1e3:+9.3f} meV"
                  f"  time {time.perf_counter() - t:6.1f} s", flush=True)
    mod.redox = base_cfg
