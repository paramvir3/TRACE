# Written on 2026-10-31 (the dimer-over-Mg curve, fold and frame wall variants; theory note 4c, step 24's W1); run from the repo root.
"""The model's binding curve of the Au2 dimer standing over a surface Mg in the 4G undoped cell: the flat reference's
dimer re-placed standing over the Mg nearest to Au108, lower Au at height h along the surface normal; energy relative
to h = 3.30 A with the fold off and on (exact block, Pauli; c / kappa from FOLDS), for comparison with the PBE curve
(tools/electrostatics/mgtop_scan.py).  Usage: python model_mgtop_scan.py MODEL.pt"""
import sys, dataclasses, numpy as np, torch
sys.path.insert(0, "experiments/benchmarks_4g"); sys.path.insert(0, ".")
from flashace.calculator import TransformersACECalculator
from eval_aumgo import AU, references
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
ref = references()[("undoped", "wetting")]
sym = ref.get_chemical_symbols(); MG = [k for k, s in enumerate(sym) if s == "Mg"]; O_ = [k for k, s in enumerate(sym) if s == "O"]
mg = MG[int(np.argmin(ref.get_distances(AU[0], MG, mic=True)))]
nz = np.sign(ref.positions[AU[0], 2] - ref.positions[mg, 2]); n = np.array([0.0, 0.0, nz])
H = (1.80, 2.00, 2.30, 2.60, 2.90, 3.30)
FOLDS = (("as trained", "keep", None), ("fold off", None, False), ("fold off + wall", None, True),
         ("c 0.85, kappa 0 + wall", (0.85, 0.0), True), ("c 0.67, kappa 0.31 + wall", (0.67, 0.31), True))
PBE = {2.00: 2.017, 2.30: 0.540, 2.60: 0.107, 2.90: 0.004, 3.30: 0.0}   # tools/electrostatics/mgtop_scan.py (2026-10-30)
for model in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=model, device="cpu")
    mod = calc.model.electrostatics
    print(f"=== {model.split('/')[-2][-3:]}: dimer standing over Mg{mg}; E(h) - E(3.30) in eV", flush=True)
    base = mod.redox
    for label, cfg, wall in FOLDS:
        if cfg == "keep":
            mod.redox = base
        elif cfg is None:
            mod.redox = dataclasses.replace(mod.redox, downfold_transfer=False, frame_wall=wall)
        else:
            mod.redox = dataclasses.replace(mod.redox, downfold_transfer=True, downfold_hop_ordering="pauli", frame_wall=wall,
                                            downfold_block="exact", downfold_scale=cfg[0], downfold_decay=cfg[1],
                                            downfold_anisotropy=0.0, downfold_crystal_field=0.0, downfold_free_axes=False)
        E = {}
        for h in H:
            at = ref.copy()
            at.positions[AU[0]] = ref.positions[mg] + h * n
            at.positions[AU[1]] = ref.positions[mg] + (h + 2.545) * n
            at.calc = calc; calc.reset()
            try:
                E[h] = at.get_potential_energy()
            except Exception as exc:
                E[h] = float("nan"); print(f"   {label} h {h}: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
        dO = [float(ref.get_distances(AU[0], O_, mic=True).min())]
        print(f"   {label:20s}: " + "  ".join(f"{h:.2f}: {E[h] - E[3.30]:+.3f}" for h in H), flush=True)
        if cfg == "keep":
            ok = all(abs((E[h] - E[3.30]) - PBE[h]) <= 0.30 * PBE[h] for h in (2.00, 2.30))
            print(f"   (W1) as trained within 30 % of PBE at 2.0 and 2.3 A: {'PASS' if ok else 'FAIL'} "
                  f"(PBE +{PBE[2.00]:.3f} / +{PBE[2.30]:.3f})", flush=True)
