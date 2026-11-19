"""Review check (2026-11-10): is the full model's energy per cell the same in a supercell, and how does one energy + force
evaluation scale with the number of atoms?  Frozen weights; the four T5 reference structures' upright members, tiled in
the surface plane.  Pre-stated expectation: a local backbone and converged lattice sums are extensive to round-off; the
configuration ensemble treats the periodic images of a cell as being in the SAME configuration (one H per cell), so a
state that mixes configurations with different dipoles carries the images' interaction with <mu^2>, where a mean-field
density (DFT's Hartree term) carries <mu>^2.  If that matters, E(supercell)/cells differs from E(cell)."""
import sys, time, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np, torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references
refs = references()
for label, path in (("local", sys.argv[1]), ("full", sys.argv[2])):
    calc = TransformersACECalculator(model_path=path, device="cpu"); torch.set_num_threads(4)
    mod = getattr(calc.model, "electrostatics", None)
    for key in (("doped", "non-wetting"), ("doped", "wetting"), ("undoped", "non-wetting"), ("undoped", "wetting")):
        base = None
        for rep in ((1, 1, 1), (2, 1, 1), (2, 2, 1)):
            a = refs[key].copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
            n1 = len(refs[key]); cells = rep[0] * rep[1] * rep[2]
            a.calc = calc; calc.reset()
            if mod is not None: mod._ensemble_last = None
            t = time.perf_counter()
            try:
                e = float(a.get_potential_energy()); f = a.get_forces(); dt = time.perf_counter() - t
            except Exception as ex:
                print(f"{label} {key} {rep}: REFUSED {type(ex).__name__}: {str(ex)[:160]}", flush=True); continue
            ncfg = "-"
            if mod is not None and getattr(mod, "_ensemble_last", None) is not None:
                ncfg = int(mod._ensemble_last["H"].shape[0])
            if base is None: base = (e, f.copy())
            dF = max(float(np.abs(f[k * n1:(k + 1) * n1] - base[1]).max()) for k in range(cells))
            print(f"{label} {key[0]:8s}{key[1]:12s} {rep}: atoms {len(a):4d}  n_cfg {ncfg!s:>4}  E/cell - E(1x1) "
                  f"{(e / cells - base[0]) * 1e3:+10.4f} meV  max|dF| {dF:.2e} eV/A  time {dt:7.2f} s", flush=True)
