# Diagnostic probe (2026-10-18; reported).  Usage (repo root): python experiments/benchmarks_4g/probe_dimer_alpha.py SYMBOL D1,D2,... CKPT [...]
"""The model's polarisability of an isolated atom and homonuclear dimer: central differences of the TOTAL energy in
a uniform applied field (open boundaries, h = 0.01 V/A), alpha = -d2E/dE2 x K_E -> A^3, so the free ensemble's
coherent charge-transfer response is included (probe_ag_polarizability.py, generalised).  The reference is
tools/electrostatics/dimer_polarizability.py.  DIMER_CHARGE=-1: the anion dimer (ratios still per neutral atom)."""
import sys, warnings
import numpy as np, torch
warnings.simplefilter("ignore")
sys.path.insert(0, "."); torch.set_default_dtype(torch.float64)
from ase import Atoms
from flashace.calculator import TransformersACECalculator
K_E = 14.399645; AU3 = 0.529177210903 ** 3
import os
CHARGE = float(os.environ.get("DIMER_CHARGE", "0"))
def energy(calc, atoms, field, charge=None):
    a = atoms.copy(); a.info = {"charge": CHARGE if charge is None else charge, "external_field": list(field)}; a.calc = calc
    calc.reset(); return a.get_potential_energy()
def alpha(calc, atoms, axis, h=0.01, charge=None):
    e = np.zeros(3); e[axis] = h
    return -(energy(calc, atoms, e, charge) - 2 * energy(calc, atoms, 0 * e, charge) + energy(calc, atoms, -e, charge)) / h ** 2 * K_E
sym, ds = sys.argv[1], [float(x) for x in sys.argv[2].split(",")]
for ck in sys.argv[3:]:
    calc = TransformersACECalculator(model_path=ck, device="cpu"); torch.set_num_threads(2)
    a1 = alpha(calc, Atoms(sym, positions=[[0, 0, 0]]), 2, charge=0.0)          # the neutral atom, for the ratios
    print(f"{ck.split('/')[-2]:60s} {sym} atom {a1:7.3f} A^3")
    for d in ds:
        dimer = Atoms(sym + "2", positions=[[0, 0, 0], [0, 0, d]])
        ap, an = alpha(calc, dimer, 2), alpha(calc, dimer, 0)
        print(f"    {sym}2 d={d:.2f}: alpha_par {ap:8.3f}  alpha_perp {an:7.3f}  (per 2 alpha_atom: {ap / (2 * a1):.3f} / {an / (2 * a1):.3f})")
