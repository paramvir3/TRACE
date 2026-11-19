# Written on 2026-10-30 (the diagnosis of step 22's first training step: CHANGELOG Round 28, theory note 4c); run from the repo root.
"""Where the exact block's forward time goes: one training structure, the step-20 checkpoint, per_state against
exact (Pauli, c = 0.85), with the fold's pieces timed and a cProfile of the exact forward."""
import sys, time, dataclasses, cProfile, pstats, io, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
from flashace.calculator import TransformersACECalculator
from ase.io import read
import trace_redox.module as M, trace_redox.diabatic as D, trace_redox.downfold as F, trace_redox.polarizability as P
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
mod = calc.model.electrostatics
at = read("experiments/benchmarks_4g/data/AuMgO_train.xyz", "205")
counts = {}
def wrap(obj, name):
    f = getattr(obj, name)
    def g(*a, **k):
        t0 = time.time(); out = f(*a, **k); c = counts.setdefault(name, [0, 0.0]); c[0] += 1; c[1] += time.time() - t0; return out
    setattr(obj, name, g)
for obj, name in ((F, "dressing"), (F, "dressing_exact"), (F, "_leaves"), (F, "block_ground"), (D, "search_configurations"),
                  (D, "coupled_closure"), (P, "state_polarizability_corrections"), (D, "configuration_energies")):
    wrap(obj, name)
for label, blk in (("per_state", "per_state"), ("exact", "exact")):
    mod.redox = dataclasses.replace(mod.redox, downfold_transfer=True, downfold_hop_ordering="pauli", downfold_block=blk,
                                    downfold_scale=0.85, downfold_free_axes=False, downfold_anisotropy=0.0, downfold_crystal_field=0.0)
    counts.clear()
    a = at.copy(); a.calc = calc; calc.reset()
    t0 = time.time(); E = a.get_potential_energy(); dt = time.time() - t0
    rec = getattr(mod, "_downfold_last", {}) or {}
    print(f"[{label}] forward {dt:.2f} s; configs {int(mod._ensemble_last['configs'].shape[0])}; leaves {rec.get('leaves')}; clusters {rec.get('clusters')}; "
          + "; ".join(f"{k} {v[0]}x {v[1]:.2f}s" for k, v in counts.items()), flush=True)
    a = at.copy(); a.calc = calc; calc.reset()
    t0 = time.time(); fo = a.get_forces(); print(f"[{label}] energy+forces {time.time() - t0:.2f} s", flush=True)
a = at.copy(); a.calc = calc; calc.reset()
pr = cProfile.Profile(); pr.enable(); a.get_forces(); pr.disable()
s_ = io.StringIO(); pstats.Stats(pr, stream=s_).sort_stats("cumulative").print_stats(28); print(s_.getvalue()[:6000])
