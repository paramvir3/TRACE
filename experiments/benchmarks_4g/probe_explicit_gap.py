"""The isolated dimer's ionic gap from the explicit physics alone (frozen weights; 2026-10-19; reported).

Usage (repo root): python experiments/benchmarks_4g/probe_explicit_gap.py SYMBOL D1,D2,... CKPT [...]

U(R) = E(A+ B-) - E(A0 B0), the diagonal of the diabatic Hamiltonian, for the neutral homonuclear dimer, with the
learned charge-state shifts as trained and with every shift set to zero (chi_bound = 1e-9: delta = bound tanh(raw /
bound), |delta| <= 1e-9 eV exactly), each without and with the overlap stack (SPEC 22 bond partition, 23
penetration, 25 Gordon-Kim Pauli, 26 induction at overlap, 28 state polarisabilities; frozen weights).  The
zero-shift gap is what the physics gives: the exact ladder IP - EA (Au 9.226 - 2.309 = 6.917 eV), the screened
Gaussian Coulomb of the ion pair, and the overlap terms.  Also printed: the coupling |tau|.
"""
import dataclasses
import sys
import warnings

import torch

warnings.simplefilter("ignore")
sys.path.insert(0, ".")
torch.set_default_dtype(torch.float64)
from ase import Atoms  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

STACK = dict(bond_partition=True, carrier_penetration=True, carrier_overlap="gordon_kim", exclusion=False,
             carrier_induction=True, state_polarizability=True)


def gap(calc, sym, d):
    a = Atoms(sym + "2", positions=[[0, 0, 0], [0, 0, d]])
    a.info = {"charge": 0.0}
    a.calc = calc
    calc.reset()
    a.get_potential_energy()
    last = calc.model.electrostatics._ensemble_last
    H, cfg = last["H"].detach(), last["configs"].tolist()
    if [0, 0] not in cfg or [1, -1] not in cfg:
        return float("nan"), float("nan")
    i0, i1 = cfg.index([0, 0]), cfg.index([1, -1])
    return float(H[i1, i1] - H[i0, i0]), float(H[i0, i1].abs())


def main():
    sym, ds = sys.argv[1], [float(x) for x in sys.argv[2].split(",")]
    for ck in sys.argv[3:]:
        calc = TransformersACECalculator(model_path=ck, device="cpu")
        torch.set_num_threads(2)
        mod = calc.model.electrostatics
        base = mod.redox
        print(f"=== {ck.split('/')[-2]}")
        for label, extra in (("as trained", {}), ("overlap stack", STACK)):
            for shifts in ("learned", "zero"):
                kw = dict(extra)
                if shifts == "zero":
                    kw["chi_bound"] = 1e-9
                mod.redox = dataclasses.replace(base, **kw)
                row = [gap(calc, sym, d) for d in ds]
                print(f"  {label:13s} shifts {shifts:7s}: " + " | ".join(f"{d:.2f} U {u:+.3f} tau {t:.3f}" for d, (u, t) in zip(ds, row)))
        mod.redox = base


if __name__ == "__main__":
    main()
