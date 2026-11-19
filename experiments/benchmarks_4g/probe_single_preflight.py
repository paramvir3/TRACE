"""Pre-flight for the single model (README, step 33): step 31's weights with EVERY term on, frozen, on real training
structures.  The slab-plus-bulk trainings of 2026-11-07 died three times in their first epoch (the configuration cap on
bulk cells, a refused key); this checks, before a launch, that the settings evaluate at all, what they cost, and how far
the forces are from the labels at the start.

Builds runs/AuMgO_single_preflight_s{seed}/model.pt (step 31's weights, the redox block of the single model, and
log_alpha_scale = 0), loaded as train.py's warm start will be, strictly.

    PYTHONPATH=.:experiments/benchmarks_4g python experiments/benchmarks_4g/probe_single_preflight.py 43 [42]
"""
import copy
import os
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np
import torch
from ase.io import read
from flashace.calculator import TransformersACECalculator

R = "experiments/benchmarks_4g/runs/"
P = "AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bopw_e1kw_bondlaw_r5_s"
EVERYTHING = {"frame_induction_damping": True, "frame_polarizability_anchor": True, "frame_polarizability_scope": "ions",
              "coupling_bound": "harrison", "state_polarizability_stiffness": "onsite", "diabatic_space": "clusters",
              "band_form": "fermi"}
frames = read("experiments/benchmarks_phonons/data/AuMgO_bulkaugw_train.xyz", index=":")
slabs = [a for a in frames if len(a) == 110]; cells = [a for a in frames if len(a) == 64]
rng = np.random.default_rng(20261112)
pick = [("slab", slabs[i]) for i in rng.choice(len(slabs), 6, replace=False)] + [("bulk", cells[i]) for i in rng.choice(len(cells), 6, replace=False)]
prior = read("experiments/benchmarks_phonons/data/MgO_bulk_eps_prior_pbe.xyz")
torch.set_num_threads(4)
for seed in sys.argv[1:]:
    src = torch.load(R + P + seed + "/model.pt", map_location="cpu", weights_only=False)
    new = {"epoch": 0, "model_state_dict": dict(src["model_state_dict"]), "config": copy.deepcopy(src["config"])}
    new["config"]["electrostatics"]["redox"].update(EVERYTHING)
    new["model_state_dict"]["electrostatics.log_alpha_scale"] = torch.zeros((), dtype=torch.float64)
    d = R + "AuMgO_single_preflight_s" + seed
    os.makedirs(d, exist_ok=True); torch.save(new, d + "/model.pt")
    calc = TransformersACECalculator(model_path=d + "/model.pt", device="cpu")
    mod = calc.model.electrostatics
    r = mod.redox
    print(f"== seed {seed}: " + ", ".join(f"{k}={getattr(r, k)!r}" for k in EVERYTHING) + f"; bond_pauli={r.bond_pauli}; coupling head bound {mod.coupling.bound!r}")
    stats = {"slab": [], "bulk": []}
    for kind, atoms in pick + [("prior", prior)]:
        a = atoms.copy(); a.set_constraint(); a.info = {"charge": float(atoms.info.get("charge", 0.0))}
        a.calc = calc; calc.reset(); mod._ensemble_last = None; t0 = time.time()
        try:
            e = a.get_potential_energy(); f = a.get_forces()
        except Exception as err:
            print(f"   {kind:5s}: REFUSED {type(err).__name__}: {str(err)[:170]}"); continue
        last = mod._ensemble_last or {}
        sizes = sorted(int(H.shape[0]) for H in last.get("hamiltonians", [])) if last.get("space") == "clusters" else None
        note = ""
        if kind != "prior":
            ref = atoms.get_forces(); err = float(np.sqrt(((f - ref) ** 2).mean())); stats[kind].append((err, float(np.sqrt((ref ** 2).mean()))))
            note = f"; force RMSE {err:.4f} against an RMS force of {stats[kind][-1][1]:.4f} eV/A"
        print(f"   {kind:5s}: {len(a):3d} atoms; clusters {sizes}{note}; {time.time() - t0:.1f} s", flush=True)
    for kind in ("slab", "bulk"):
        if stats[kind]:
            e_, r_ = np.array(stats[kind]).T
            print(f"   {kind}: force RMSE {np.sqrt((e_ ** 2).mean()):.4f} eV/A over {len(e_)} structures ({100 * np.sqrt((e_ ** 2).mean()) / np.sqrt((r_ ** 2).mean()):.0f} % of the RMS force)")
