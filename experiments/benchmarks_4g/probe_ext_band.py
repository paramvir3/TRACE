"""Extensivity, L3 (2026-11-10): where the doped cells' electrons sit.  Frozen weights, clusters space.

    python probe_ext_band.py MODEL.pt FORM [nx ny ...]

Per doped geometry and replication: the reference configuration n0 by element (how many atoms in which charge state),
the band unit's state, and the cluster configurations.  Writes nothing into the run directory."""
import collections
import dataclasses
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references

model_path, form = sys.argv[1], sys.argv[2]
reps = [(int(sys.argv[i]), int(sys.argv[i + 1]), 1) for i in range(3, len(sys.argv) - 1, 2)] or [(1, 1, 1), (2, 1, 1)]
calc = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
mod.redox = dataclasses.replace(mod.redox, state_polarizability_stiffness=form, diabatic_space="clusters")
refs = references()
for key in (("doped", "wetting"), ("doped", "non-wetting"), ("undoped", "wetting")):
    for rep in reps:
        a = refs[key].copy(); a.set_constraint(); a = a.repeat(rep); a.info = dict(refs[key].info)
        a.calc = calc; calc.reset(); mod._ensemble_last = None
        e = a.get_potential_energy()
        last = mod._ensemble_last
        n0 = last["reference"].tolist(); nat = len(a)
        count = collections.Counter((int(z), int(n)) for z, n in zip(a.numbers, n0[:nat]))
        print(f"== {key} {rep[0]}x{rep[1]}: charge {a.info.get('charge')}  E/cell {e / (rep[0] * rep[1]):.6f} eV  atoms {nat}  units {len(n0)}")
        print("   n0 by (Z, state): " + ", ".join(f"Z{z}:{n:+d} x{c}" for (z, n), c in sorted(count.items())))
        print(f"   units beyond the atoms (band): {n0[nat:]};  sum of n0 {sum(n0)}")
        for c, S in enumerate(last["clusters"]):
            for row in S[1:].tolist():
                ch = [(u, row[u] - n0[u]) for u in range(len(n0)) if row[u] != n0[u]]
                print(f"   cluster {c}: " + ", ".join(f"{'band' if u >= nat else 'Z%d#%d' % (a.numbers[u], u)}:{d:+d}" for u, d in ch))
