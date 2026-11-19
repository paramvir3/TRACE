"""Would the configurations training can select be polarisation catastrophes under SPEC 28? (frozen weights;
2026-10-19; reported)

Usage: python probe_stiffness_constructed.py MODEL.pt [...]

Step 13F's eps_inf prior reached 5 configurations in bulk MgO during training (weights not saved), and the doped slab
can return the dopant's electron to Al.  This captures the trained bound sector (the elimination) of the eps_inf prior's
bulk MgO and of the doped T5 references, and tests hand-built configurations with the exact inertia criterion
(polarizability.stiffness_defects): every nearest Mg+ O- pair, Mg0 with two O-, and in the doped slab Al2+ (the
electron on the dopant) and Mg+ next to Al.  Ratios: data/ion_polarizabilities.json (Mg+ 70.7, Mg0 150.5, Al2+ 53.9,
O- 0.44 of their frames).  Also reports the SPEC 12 guarantee bound rho < 3^(1/4) / a.
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

import trace_redox.polarizability as P  # noqa: E402
from ase.io import read  # noqa: E402
from ase.neighborlist import neighbor_list  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from eval_aumgo import references  # noqa: E402

CAP = {}
_orig = P.state_polarizability_corrections


def _capture(configs, elim, frames, table, states, omega=None, induction=None):
    CAP.update(elim=elim, frames=frames, table=table, states=states, U=int(configs.shape[1]))
    return _orig(configs, elim, frames, table, states, omega=omega, induction=induction)


def constructed(atoms, frames, U):
    """Charge-neutral configurations (the frames of a doped slab sum to +1: the dopant's electron must sit somewhere)."""
    Z = atoms.numbers
    base = np.zeros(U, dtype=np.int64)
    base[:len(Z)] = frames.numpy()[:len(Z)]
    au = [k for k in range(len(Z)) if Z[k] == 79]
    excess = int(base[:len(Z)].sum())                                 # +1 for the Al-doped slab
    neutral = base.copy()
    if excess == 1 and au:
        neutral[au[0]] -= 1                                           # the dopant's electron on Au: Au-
    i, j = neighbor_list("ij", atoms, 2.4)
    mg = [k for k in range(len(Z)) if Z[k] == 12]
    al = [k for k in range(len(Z)) if Z[k] == 13]
    out = [("frame + Au-" if excess else "frame", neutral)]
    for a in (al if al else mg[:4]):
        name = "Al" if Z[a] == 13 else "Mg"
        o_nb = sorted({int(jj) for ii, jj in zip(i, j) if ii == a and Z[jj] == 8})
        for o in o_nb[:6]:
            c = neutral.copy(); c[a] -= 1; c[o] += 1                   # one electron O -> M: M^(f-1) O-
            out.append((f"{name}{c[a]:+d} O{c[o]:+d} (atoms {a}, {o})", c))
        if len(o_nb) >= 2:
            c = neutral.copy(); c[a] -= 2; c[o_nb[0]] += 1; c[o_nb[1]] += 1
            out.append((f"{name}{c[a]:+d} with two O- (atom {a})", c))
    if al and excess == 1:
        for a in al:
            c = base.copy(); c[a] -= 1                                 # the electron back on the dopant: Al2+
            out.append((f"Al{c[a]:+d}, Au neutral (atom {a})", c))
            for m in sorted(mg, key=lambda k: atoms.get_distance(a, k, mic=True))[:3]:
                c = base.copy(); c[m] -= 1                             # the electron on an Mg next to Al: Mg+
                out.append((f"Mg{c[m]:+d} near Al (atom {m}, {atoms.get_distance(a, m, mic=True):.2f} A)", c))
    return out


def main():
    P.state_polarizability_corrections = _capture
    structures = [("bulk MgO (eps prior)", a) for a in read(os.path.join(HERE, "data", "MgO_bulk_eps_prior.xyz"), ":")]
    structures += [(f"T5 doped/{k[1]}", a) for k, a in references().items() if k[0] == "doped"]
    for model in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(2)
        mod = calc.model.electrostatics
        mod.redox = dataclasses.replace(mod.redox, state_polarizability=True)
        print(f"=== {os.path.basename(os.path.dirname(model))}")
        for label, atoms in structures:
            a = atoms.copy()
            a.info = dict(atoms.info)
            a.info.setdefault("charge", 0.0)
            a.calc = calc
            calc.reset()
            CAP.clear()
            a.get_potential_energy()
            if not CAP:
                print(f"  {label}: SPEC 28 not reached")
                continue
            rows = constructed(a, CAP["frames"], CAP["U"])
            cfgs = torch.tensor(np.stack([c for _, c in rows]))
            bad = P.stiffness_defects(cfgs, CAP["elim"], CAP["frames"], CAP["table"], CAP["states"])
            print(f"  {label}: {len(rows)} constructed configurations, defects {int(bad.sum())}")
            for (name, _), b in zip(rows, bad.tolist()):
                if b:
                    print(f"     DEFECT {name}")


if __name__ == "__main__":
    main()
