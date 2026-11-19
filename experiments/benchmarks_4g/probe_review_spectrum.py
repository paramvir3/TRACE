"""Review check (2026-11-10), frozen weights, no model change.
Striking number: S5's undoped non-wetting "Gibbs F - F1" is -69.05 meV; kT ln 2 at kT = 0.1 eV is 69.31 meV.
Written before the run:
  - if that term is the entropy of a two-fold degenerate ground level, then F - lambda_min ~ -69.3 meV, with two
    eigenvalues of H within << kT of each other;
  - if it is a coherent (energy) stabilisation, then F - lambda_min ~ 0 and the gap above the ground level is >> kT.
Also printed: wet_u, wet_d, T5 at the four reference structures at the trained kT and at smaller kT (frozen weights)."""
import dataclasses, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references
refs = references()
for model in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=model, device="cpu"); torch.set_num_threads(4)
    mod = calc.model.electrostatics; base = mod.redox
    print("===", model.split("/")[-2][-22:], "trained kT", base.kT, flush=True)
    for kT in (base.kT, 0.05, 0.025):
        mod.redox = dataclasses.replace(base, kT=kT)
        E = {}
        for key, at in refs.items():
            a = at.copy(); a.info = {"charge": 0.0}; a.calc = calc; calc.reset(); mod._ensemble_last = None
            try:
                E[key] = float(a.get_potential_energy())
            except Exception as err:
                print(f"  kT {kT}: {key} refused ({type(err).__name__}: {str(err)[:90]})"); continue
            last = mod._ensemble_last
            H = last["H"].detach(); H = 0.5 * (H + H.T)
            lam = torch.linalg.eigvalsh(H); gaps = (lam - lam[0])
            F = float(-kT * torch.logsumexp(-lam / kT, 0)); ent = F - float(lam[0])
            diag = torch.sort(torch.diagonal(H)).values
            print(f"  kT {kT:5.3f} {key[0]:8s}{key[1]:12s} n_cfg {H.shape[0]:5d}  F - lam_min {1e3 * ent:+8.3f} meV  "
                  f"lowest gaps (eV) {np.round(gaps[1:5].numpy(), 4).tolist()}  n within kT {int((gaps < kT).sum())}, 3kT {int((gaps < 3 * kT).sum())}  "
                  f"diag gaps {np.round((diag[1:4] - diag[0]).numpy(), 4).tolist()}", flush=True)
        if len(E) == 4:
            wu = E[("undoped", "wetting")] - E[("undoped", "non-wetting")]; wd = E[("doped", "wetting")] - E[("doped", "non-wetting")]
            print(f"  kT {kT:5.3f}: wet_u {1e3 * wu:+8.1f}  wet_d {1e3 * wd:+8.1f}  T5 {1e3 * (wd - wu):+8.1f} meV   (DFT +928.9, -2.7, -931.6)", flush=True)
    mod.redox = base
