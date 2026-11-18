"""Frozen-weight counterfactual of SPEC 32 (the downfolded donor -> acceptor bond) on a trained model (2026-10-28).

Usage (repo root): python experiments/benchmarks_4g/probe_downfold_counterfactual.py MODEL.pt [...]
       (DF_SCALE, DF_ANISO, DF_CF, DF_FREE in the environment set the pair's measured constants and the dangling-axis
       rule; default the plain term)

The model as trained, then with ``downfold_transfer=True`` (no other change: the O -> Au explicit couplings go to
zero and each configuration is dressed by its exact local star root with Harrison's fixed V_sp sigma).  Reported:
- the four T5 references (unrelaxed): wet_u, wet_d, T5, configuration counts, the smallest transfer energy Delta;
- the separated limit (undoped wetting, Au2 at mid-vacuum): dE = E(AB) - E(A in the cell) - E(B);
- the undoped flat dimer relaxed as eval_aumgo.py relaxes it (step 21's collapse: -3.0 / -3.6 eV onto O).
Pre-registered checks (2026-10-28, before the run): (c1) |dE| at separation < 1 meV; (c2) the relaxed flat dimer
stays within 0.3 A of the reference heights (2.89 A) and its relaxation energy is above -0.5 eV; (c3) the smallest
mu (the folded manifold's lowest level above each explicit state) on the references is > 0; a system without a
folded pair reports 'no folded pair' rather than a verdict; the min mu at the relaxed flat geometry is reported.  Direction and size only otherwise: the
other weights were trained without the dressing.
"""
import dataclasses
import os
import sys
import warnings

import numpy as np
import torch

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)

from ase import Atoms  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from eval_aumgo import AU, references, relax  # noqa: E402

DFT = (928.9, -2.7, -931.6)


def energy(calc, atoms, charge=0.0):
    a = atoms.copy()
    a.info = {"charge": float(charge)}
    a.calc = calc
    calc.reset()
    mod = calc.model.electrostatics
    mod._downfold_last = None
    E = float(a.get_potential_energy())
    rec = getattr(mod, "_downfold_last", None) or {}
    if rec and rec.get("leaves"):
        print(f"      [{a.get_chemical_formula()} Q={charge:+.0f}] fold: {rec['leaves']} leaves in {rec['clusters']} clusters, "
              f"min Delta {rec['delta_min']:+.3f}, min mu {rec['mu_min']:+.3f}, most negative lambda "
              f"{rec['lambda_min']:+.3f} eV", flush=True)
    return E, int(mod._ensemble_last["configs"].shape[0]), rec.get("mu_min", float("nan"))


def heights(atoms):
    top = atoms.positions[:AU[0], 2].min()
    return [float(top - atoms.positions[k, 2]) for k in AU]


def main():
    refs = references()
    for model in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(2)
        mod = calc.model.electrostatics
        base = dataclasses.replace(mod.redox, fragment_max_configurations=20000)
        name = os.path.basename(os.path.dirname(model))
        for label, on in (("as trained", False), ("downfold on", True)):
            consts = {k: float(os.environ[e]) for k, e in (("downfold_scale", "DF_SCALE"), ("downfold_anisotropy", "DF_ANISO"),
                                                             ("downfold_crystal_field", "DF_CF")) if e in os.environ}
            if "DF_FREE" in os.environ:
                consts["downfold_free_axes"] = os.environ["DF_FREE"] not in ("0", "", "false")
            if "DF_HOP" in os.environ:                                     # 'bonding' (legacy) or 'pauli'
                consts["downfold_hop_ordering"] = os.environ["DF_HOP"]
            if "DF_BLOCK" in os.environ:                                   # 'per_state' or 'exact'
                consts["downfold_block"] = os.environ["DF_BLOCK"]
            if "DF_DECAY" in os.environ:                                   # the coupling's decay beyond Harrison, 1/A
                consts["downfold_decay"] = float(os.environ["DF_DECAY"])
            mod.redox = dataclasses.replace(base, downfold_transfer=on, **consts)
            if on and consts:
                print(f"    measured constants {consts}", flush=True)
            E = {k: energy(calc, at) for k, at in refs.items()}
            wu = E[("undoped", "wetting")][0] - E[("undoped", "non-wetting")][0]
            wd = E[("doped", "wetting")][0] - E[("doped", "non-wetting")][0]
            n = {k[0][0] + k[1][0]: v[1] for k, v in E.items()}
            mus = [v[2] for v in E.values() if v[2] == v[2]]
            dmin = min(mus) if mus else float("nan")
            print(f"=== {name} {label}: wet_u {wu * 1e3:+.1f}, wet_d {wd * 1e3:+.1f}, T5 {(wd - wu) * 1e3:+.1f} meV "
                  f"(DFT {DFT[0]:+.1f} / {DFT[1]:+.1f} / {DFT[2]:+.1f}); configurations {n}; min mu "
                  f"{dmin:+.3f} eV" if mus else f"=== {name} {label}: wet_u {wu * 1e3:+.1f}, wet_d {wd * 1e3:+.1f}, "
                  f"T5 {(wd - wu) * 1e3:+.1f} meV; configurations {n}; no folded pair", flush=True)
            ab = refs[("undoped", "wetting")].copy()
            ab.positions[AU, 2] += 29.15 - float(ab.positions[AU, 2].mean())
            au = Atoms("Au2", positions=ab.positions[AU], cell=ab.cell, pbc=ab.pbc)
            dE = energy(calc, ab)[0] - energy(calc, au)[0] - energy(calc, ab[:AU[0]])[0]
            print(f"    separated limit dE {dE * 1e3:+.3f} meV" + (f" -> (c1) {'PASS' if abs(dE) < 1e-3 else 'FAIL'}"
                                                                    if on else ""), flush=True)
            ref = refs[("undoped", "wetting")]
            e0 = energy(calc, ref)[0]
            try:
                out, _ = relax(calc, ref)
                e1, _, mu1 = energy(calc, out)
                h0, h1 = heights(ref), heights(out)
                ok = all(abs(a - b) <= 0.3 for a, b in zip(h0, h1)) and (e1 - e0) > -0.5
                print(f"    flat undoped relaxed: heights {np.round(h0, 3).tolist()} -> {np.round(h1, 3).tolist()} A, "
                      f"Au-Au {out.get_distance(AU[0], AU[1], mic=True):.3f} A, relaxation {(e1 - e0) * 1e3:+.1f} meV, "
                      f"min mu there {mu1:+.3f} eV" + (f" -> (c2) {'PASS' if ok else 'FAIL'}" if on else ""), flush=True)
            except Exception as exc:                         # a refused fold during the relaxation is a result
                print(f"    flat undoped relaxed: refused ({type(exc).__name__}: {str(exc)[:160]})"
                      + (" -> (c2) FAIL" if on else ""), flush=True)
            if on:
                verdict = "no folded pair" if not mus else ("PASS" if dmin > 0 else "FAIL")
                print(f"    (c3) min mu on the references {dmin:+.3f} eV -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
