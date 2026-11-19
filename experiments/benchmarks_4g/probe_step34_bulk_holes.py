"""Step 34 stage A stopped in epoch 2 (2026-11-15): on bulk MgO (64-atom fc3 cells, train 404 / 347) the cluster's
product closure held 529 > 512 configurations, every placement of 0, 1 or 2 O holes on the 32 oxygens with the band
taking the electrons (replay_step34_closure.py).  This probe asks whether the epoch-1 models already put holes there.

    PYTHONPATH=../.. python probe_step34_bulk_holes.py OUT.txt      (the Mac; CPU)

Models (eval mode, frozen): stage A's epoch-1 best of each seed (the checkpoint that passed epoch 1), and, for
contrast, step 33's final model of seed 43 with frame_pair, spin_pairing and two_site switched on (as in
gpu_check_step34.py).  Structures: train 404 and 347 (where the seeds stopped) and the eps prior cell.
For each: the band's reference filling N0 (SPEC 38's descent), every product closure (linked, closed), the
population of configurations by number of O holes, and the energy.
"""
import collections
import dataclasses
import os
import sys
import warnings

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import torch  # noqa: E402
from ase.io import read  # noqa: E402

import trace_redox.clusters as clusters_mod  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_redox.constants import formal_frame  # noqa: E402
from trace_redox.diabatic import density_matrix  # noqa: E402

RECORD = []
_orig = clusters_mod._closure


def _recording(n0, S, reservoir, L, K, units, frames, cap):
    try:
        out = _orig(n0, S, reservoir, L, K, units, frames, cap)
    except Exception:
        RECORD.append((int(S.shape[0]), None))
        raise
    RECORD.append((int(S.shape[0]), int(out.shape[0])))
    return out


clusters_mod._closure = _recording
MODELS = {
    "stage A s42 epoch 1": ("runs/AuMgO_twosite_bondlaw_r5_s42/model.pt", False),
    "stage A s43 epoch 1": ("runs/AuMgO_twosite_bondlaw_r5_s43/model.pt", False),
    "step 33 s43 final + 3 switches": ("runs/AuMgO_single_all_e1kw_bondlaw_r5_s43/model.pt", True),
}


def main():
    out = open(sys.argv[1], "w")

    def say(msg):
        print(msg, flush=True)
        out.write(msg + "\n")
        out.flush()

    train = read(os.path.join(HERE, "..", "benchmarks_phonons", "data", "AuMgO_bulkaugw_train.xyz"), ":")
    structs = {"train 404": train[404], "train 347": train[347],
               "eps prior cell": read(os.path.join(HERE, "..", "benchmarks_phonons", "data", "MgO_bulk_eps_prior_pbe.xyz"))}
    for label, (path, switch) in MODELS.items():
        calc = TransformersACECalculator(model_path=os.path.join(HERE, path), device="cpu")
        mod = calc.model.electrostatics
        if switch:
            mod.redox = dataclasses.replace(mod.redox, frame_pair=True, spin_pairing=True, two_site=True)
        mod.redox = dataclasses.replace(mod.redox, fragment_max_configurations=20000)
        say(f"== {label}: {path}")
        for name, atoms in structs.items():
            a = atoms.copy()
            a.set_constraint()
            a.info = {"charge": 0.0}
            a.calc = calc
            calc.reset()
            RECORD.clear()
            try:
                E = float(a.get_potential_energy())
            except Exception as err:                                             # noqa: BLE001
                say(f"  {name}: {type(err).__name__}: {str(err)[:120]}")
                continue
            last, band = mod._ensemble_last, mod._band_last
            N0 = None if band is None else band.get("reference")
            z = torch.as_tensor(a.numbers)
            n = len(a)
            f = formal_frame(z)
            holes = collections.Counter()
            if last.get("space") == "clusters" and last.get("hamiltonians"):
                for S, H in zip(last["clusters"], last["hamiltonians"]):
                    p = torch.diagonal(density_matrix(H, float(last["kT"])))
                    for c in range(int(S.shape[0])):
                        k = int(((S[c, :n] - f) * (z == 8)).clamp(min=0).sum())
                        holes[k] += float(p[c])
            sizes = sorted(RECORD, key=lambda x: -(x[1] or 0))[:4]
            say(f"  {name}: E {E:+.6f} eV; band reference filling N0 {N0}; closures {len(RECORD)}, largest "
                f"(linked, closed) {sizes}; population by O holes (largest cluster set) "
                f"{ {k: round(v, 4) for k, v in sorted(holes.items())} }")


if __name__ == "__main__":
    main()
