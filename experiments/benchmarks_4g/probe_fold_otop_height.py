# Written on 2026-10-30 (the model's O-top charge-transfer level versus height; theory note 4c); run from the repo root.
"""The model's own charge-transfer level at the O-top contact versus height: the undoped upright reference with the
dimer moved rigidly along the O -> lower-Au axis to h = 2.0 / 2.26 / 2.6 / 3.0 A; the frame's O -> lower-Au p_z leaf
(Delta, v) from the fold record, against the ALMO series Delta_sigma*(h) = 3.446 / 2.922 / 2.427 / 2.064 eV.  The model's
leaf Delta is the diabatic level before the s-s hop; its sigma* level is Delta - |T| f.  Usage: python model_delta_h.py MODEL.pt"""
import sys, dataclasses, numpy as np, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
import probe_t5_first_order as Q
from flashace.calculator import TransformersACECalculator
Q._install()
ref = Q.references()[("undoped", "non-wetting")]
ALMO = {2.00: 3.446, 2.26: 2.922, 2.60: 2.427, 3.00: 2.064}
for model in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=model, device="cpu"); torch.set_num_threads(2)
    mod = calc.model.electrostatics
    mod.redox = dataclasses.replace(mod.redox, fragment_max_configurations=20000, downfold_transfer=True,
                                    downfold_hop_ordering="pauli", downfold_block="exact", downfold_scale=0.85,
                                    downfold_anisotropy=0.0, downfold_crystal_field=0.0, downfold_free_axes=False)
    sym = ref.get_chemical_symbols(); AU = [k for k, s_ in enumerate(sym) if s_ == "Au"]
    O_ = [k for k, s_ in enumerate(sym) if s_ == "O"]
    lower = min(AU, key=lambda g: float(ref.get_distances(g, O_, mic=True).min()))
    o_top = O_[int(np.argmin(ref.get_distances(lower, O_, mic=True)))]
    u = ref.get_distance(o_top, lower, mic=True, vector=True); h0 = float(np.linalg.norm(u)); u = u / h0
    rows = []
    for h in (2.00, 2.26, 2.60, 3.00):
        at = ref.copy(); at.info = dict(ref.info)
        for g in AU:
            at.positions[g] += (h - h0) * u
        s = Q.evaluate(calc, at, 0.0)
        rec = mod._downfold_last; full = rec.get("all_leaves_full") or []
        frame = [l for l in full if l[0] == 0 and l[1] == o_top and l[2] == lower and l[3] < 3]
        pz = max(frame, key=lambda l: abs(l[5])) if frame else None
        T_fold = float(abs(mod._ensemble_last["H"].new_tensor(0.0)))   # placeholder (the s-s hop enters below)
        rows.append((h, pz[4] if pz else float("nan"), pz[5] if pz else float("nan"), len(rec.get("promoted", []) or []),
                     [round(w, 3) for w in rec.get("explicit_weight", [])]))
        print(f"  {model.split('/')[-2][-3:]} h {h:.2f}: model leaf Delta {rows[-1][1]:+.3f} eV, v {rows[-1][2]:+.3f} eV; "
              f"ALMO Delta_sigma* {ALMO[h]:.3f}; promoted {rows[-1][3]}; explicit weights {rows[-1][4]}", flush=True)
    d = np.array([r[1] for r in rows]); a = np.array([ALMO[r[0]] for r in rows])
    print(f"  {model.split('/')[-2][-3:]}: model Delta(h) - Delta(3.0) = {np.round(d - d[-1], 3).tolist()}; ALMO {np.round(a - a[-1], 3).tolist()} "
          f"(the Pauli shift the model must carry: 1.38 / 0.86 / 0.36 / 0 eV)", flush=True)
