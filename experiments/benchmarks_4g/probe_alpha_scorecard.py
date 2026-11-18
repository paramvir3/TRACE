# Diagnostic probe (2026-10-18; reported).  Usage (repo root): python experiments/benchmarks_4g/probe_alpha_scorecard.py PRIOR.xyz CKPT [...]
"""The model's static polarisability against a set of first-principles tensors (the field-prior format: diagonal
labels), by finite fields of the total energy through the whole model, for the model as trained and with the
first-principles physics switched on (frozen weights).  RMSE over all diagonal components, and per structure."""
import dataclasses, sys, warnings
import numpy as np, torch
warnings.simplefilter("ignore"); sys.path.insert(0, "."); torch.set_default_dtype(torch.float64)
from ase.io import read
from flashace.calculator import TransformersACECalculator
K_E = 14.399645
def energy(calc, a0, field):
    a = a0.copy(); a.info = {"charge": float(a0.info.get("charge", 0)), "external_field": list(field)}; a.calc = calc
    calc.reset(); return a.get_potential_energy()
def alpha_diag(calc, a0, h=0.01):
    e0 = energy(calc, a0, [0, 0, 0]); out = []
    for k in range(3):
        f = np.zeros(3); f[k] = h
        out.append(-(energy(calc, a0, f) - 2 * e0 + energy(calc, a0, -f)) / h ** 2 * K_E)
    return np.array(out)
prior = read(sys.argv[1], ":")
VARIANTS = [("as trained", {}),
            ("+ SPEC 22-28", dict(bond_partition=True, carrier_penetration=True, carrier_overlap="gordon_kim", exclusion=False,
                                  carrier_induction=True, state_polarizability=True)),
            ("+ SPEC 22-29", dict(bond_partition=True, carrier_penetration=True, carrier_overlap="gordon_kim", exclusion=False,
                                  carrier_induction=True, state_polarizability=True, spin_pairing=True))]
for ck in sys.argv[2:]:
    calc = TransformersACECalculator(model_path=ck, device="cpu"); torch.set_num_threads(2)
    mod = calc.model.electrostatics; base, exc = mod.redox, mod.exclusion
    for label, kw in VARIANTS:
        mod.redox = dataclasses.replace(base, **kw); mod.exclusion = None if kw.get("exclusion") is False else exc
        errs, rows = [], []
        for a in prior:
            ref = np.diag(np.array(a.info["polarizability"]))
            got = alpha_diag(calc, a)
            errs.append(got - ref)
            rows.append(f"q{int(a.info['charge']):+d}@{np.linalg.norm(a.positions[1] - a.positions[0]):.2f}: {got[2]:6.2f}/{got[0]:5.2f} ({ref[2]:5.2f}/{ref[0]:5.2f})")
        e = np.concatenate(errs)
        print(f"{ck.split('/')[-2][-26:]} {label:13s}: RMSE {np.sqrt(np.mean(e ** 2)):7.3f} A^3, max |err| {np.abs(e).max():7.2f} | " + "  ".join(rows), flush=True)
    mod.redox, mod.exclusion = base, exc
