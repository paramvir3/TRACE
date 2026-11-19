# Written on 2026-10-29 (the measurement scripts of the theory note's section 4c); run from the repo root.
"""Trace the (c2) relaxation of the flat reference step by step under the fold (exact block, Pauli, c = 0.85, the
decay from DF_DECAY, default 0.45): Au positions, nearest Au-O / Au-Mg distances, energy, the fold record's gap and
explicit weights; the last state before any failure.  Usage: python trace_collapse.py MODEL.pt"""
import sys, os, dataclasses, traceback, numpy as np, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from flashace.calculator import TransformersACECalculator
from eval_aumgo import AU, references
decay = float(os.environ.get("DF_DECAY", "0.45"))
refs = references()
for model in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=model, device="cpu"); torch.set_num_threads(2)
    mod = calc.model.electrostatics
    mod.redox = dataclasses.replace(mod.redox, fragment_max_configurations=20000, downfold_transfer=True,
                                    downfold_hop_ordering="pauli", downfold_block="exact", downfold_scale=0.85,
                                    downfold_anisotropy=0.0, downfold_crystal_field=0.0, downfold_free_axes=False,
                                    downfold_decay=decay)
    a = refs[("undoped", "wetting")].copy(); a.info = dict(a.info)
    a.set_constraint(FixAtoms(indices=list(range(108)))); a.calc = calc
    sym = a.get_chemical_symbols()
    O_idx = [i for i, s_ in enumerate(sym) if s_ == "O"]; Mg_idx = [i for i, s_ in enumerate(sym) if s_ == "Mg"]
    opt = BFGS(a, logfile=None)
    step = {"n": 0}
    def log():
        rec = getattr(mod, "_downfold_last", {}) or {}
        z = a.positions[AU, 2]
        dO = [float(a.get_distances(g, O_idx, mic=True).min()) for g in AU]
        dMg = [float(a.get_distances(g, Mg_idx, mic=True).min()) for g in AU]
        e = a.get_potential_energy(); f = np.abs(a.get_forces()[AU]).max()
        print(f"  step {step['n']:3d}: E {e:+.4f} eV  |F|max(Au) {f:.3f}  Au z {z[0]:.3f} {z[1]:.3f}  dAuO {dO[0]:.2f} {dO[1]:.2f}  dAuMg {dMg[0]:.2f} {dMg[1]:.2f}"
              f"  gap {rec.get('mu_min', float('nan')):+.3f}  w_expl {[round(w, 3) for w in rec.get('explicit_weight', [])]}  ground {[round(g, 2) for g in rec.get('ground', [])]}", flush=True)
        step["n"] += 1
    print(f"=== {model.split('/')[-2][-3:]} decay {decay}: z(slab top) ~ {a.positions[:108, 2].max():.2f}", flush=True)
    try:
        opt.attach(log, interval=1)
        log()
        opt.run(fmax=0.005, steps=400)
        print("   converged / finished", flush=True)
    except Exception as exc:
        print(f"   FAILED at step {step['n']}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        z = a.positions[AU, 2]
        print(f"   last geometry: Au z {z[0]:.3f} {z[1]:.3f}; nearest O {[round(float(a.get_distances(g, O_idx, mic=True).min()), 2) for g in AU]}; nearest Mg {[round(float(a.get_distances(g, Mg_idx, mic=True).min()), 2) for g in AU]}", flush=True)
