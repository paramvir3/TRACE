"""The model's own electron addition / removal energies on the 4G AuMgO geometry (2026-10-20; a probe, no model change).

Usage (repo root): python experiments/benchmarks_4g/probe_spectrum.py MODEL.pt [...]

TRACE's H is a many-electron Hamiltonian over integer charge configurations, so its one-electron spectrum is defined
exactly by the Lehmann representation (ESTABLISHED); its edges are total-energy differences the module already
computes, by evaluating the same structure at total charge 0, -1 and +1 (atoms.info['charge']):

    EA = E(N) - E(N+1),   IP = E(N-1) - E(N),   E_g = IP - EA = E(N+1) + E(N-1) - 2 E(N)

with E the ground-state energy of H (lowest eigenvalue; T -> 0) of the redox module, and also the whole model's
energy (backbone included).  For each charge the lowest configuration is classified (where the extra electron or the
hole sits), and at N the band unit's full tight-binding spectrum eig(diag(eps_v) - tau) is printed (its bottom is
lambda_1; SPEC 20).  Structures: the undoped wetting reference (eval_aumgo.references) and the same with Au2 removed
(the clean six-layer 3x3 MgO(100) slab, the geometry of tools/electrostatics/mgo_pbe.py's slab).

Reported, not scored: the PBE comparison numbers come from mgo_pbe.py (slab EA, IP, gap) when they are in.
"""
import collections
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

import trace_redox.band as BD  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_redox.constants import formal_frame  # noqa: E402
from eval_aumgo import AU, references  # noqa: E402

SYMBOL = {8: "O", 12: "Mg", 13: "Al", 79: "Au"}
AUG = {}


def _install():
    f = BD.BandUnit.augment

    def g(self, *a, **k):
        out = f(self, *a, **k)
        AUG["last"] = out
        return out

    BD.BandUnit.augment = g


def evaluate(calc, atoms, charge):
    a = atoms.copy()
    a.info = dict(atoms.info)
    a.info["charge"] = float(charge)
    a.calc = calc
    calc.reset()
    AUG.clear()
    cap = []
    hook = calc.model.electrostatics.register_forward_hook(lambda m, i, o: cap.append(o[1]))
    E_total = float(a.get_potential_energy())
    hook.remove()
    mod = calc.model.electrostatics
    last = mod._ensemble_last
    H, configs = last["H"].detach(), last["configs"]
    w, v = torch.linalg.eigh(H)
    E0 = float(w[0])                                       # ground state of H (T -> 0)
    z = torch.as_tensor(a.numbers)
    n = len(a)
    fr = formal_frame(z)
    weight = collections.defaultdict(float)
    for c in range(int(configs.shape[0])):
        changed = (configs[c, :n] != fr).nonzero().flatten().tolist()
        key = tuple(sorted((SYMBOL.get(int(z[j]), str(int(z[j]))), int(configs[c, j]) - int(fr[j])) for j in changed))
        band = int(configs[c, n:].abs().sum()) if configs.shape[1] > n else 0
        weight[(key, band)] += float(v[c, 0] ** 2)
    top = sorted(weight.items(), key=lambda kv: -kv[1])[:3]
    band = None
    if AUG.get("last") is not None:
        aug = AUG["last"]
        ev = torch.linalg.eigvalsh(torch.diag(aug["eps"].detach()) - aug["tau"].detach())
        band = {"lambda1": float(aug["lambda1"]), "top": float(ev[-1]), "n_sites": int(ev.numel()),
                "eps_min": float(aug["eps"].min()), "eps_max": float(aug["eps"].max())}
    r = cap[-1]
    return {"E_total": E_total, "E0_H": E0, "E_redox": float(r.energy), "n_configs": int(configs.shape[0]),
            "ground": [((" + ".join(f"{s}{q:+d}" for s, q in k) or "frame") + f" | band {b}", round(p, 3))
                       for (k, b), p in top], "band": band}


def main():
    _install()
    refs = references()
    base = refs[("undoped", "wetting")]
    clean = base.copy()
    del clean[[i for i in AU]]
    clean.info = dict(base.info)
    clean.set_constraint()
    for model in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(2)
        mod = calc.model.electrostatics
        mod.redox = dataclasses.replace(mod.redox, fragment_max_configurations=20000)
        print(f"=== {os.path.basename(os.path.dirname(model))}  (band factor "
              f"{float(mod.band.factor) if getattr(mod, 'band', None) is not None else float('nan'):.3f})")
        for label, atoms in (("Au2 + slab (undoped wetting)", base), ("clean slab (Au2 removed)", clean)):
            R = {q: evaluate(calc, atoms, q) for q in (0, -1, 1)}
            for key in ("E0_H", "E_redox", "E_total"):
                EA = R[0][key] - R[-1][key]
                IP = R[1][key] - R[0][key]
                print(f"  {label:30s} [{key:7s}] EA {EA:+8.3f} eV  IP {IP:+8.3f} eV  gap {IP - EA:+8.3f} eV")
            for q in (0, -1, 1):
                print(f"    charge {q:+d}: {R[q]['n_configs']} configurations; ground state {R[q]['ground']}")
            b = R[0]["band"]
            if b is not None:
                print(f"    band unit at N: lambda_1 {b['lambda1']:+.3f} eV, top {b['top']:+.3f} eV, width "
                      f"{b['top'] - b['lambda1']:.3f} eV over {b['n_sites']} sites; site energies "
                      f"{b['eps_min']:+.3f} .. {b['eps_max']:+.3f} eV")


if __name__ == "__main__":
    main()
