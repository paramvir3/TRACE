# Diagnostic probe (2026-10-18; criterion U6).  Usage (repo root): python experiments/benchmarks_4g/probe_ag_polarizability.py CKPT [...]
"""probe_au_polarizability.py for silver: alpha of the isolated Ag atom and Ag2 dimer: central differences of the TOTAL energy in a uniform applied field
(open boundaries), alpha = -d2E/dE2 (e^2 A^2 / eV) x K_E -> A^3.  Reference: Ag 50.97 a.u. = 7.553 A^3
(arXiv:2410.01493, the source of make_ag_alpha_prior.py)."""
import sys, warnings; sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np, torch; torch.set_default_dtype(torch.float64); torch.set_num_threads(4)
from ase import Atoms
from flashace.calculator import TransformersACECalculator
K_E = 14.399645; AU3 = 0.529177210903 ** 3
def energy(calc, atoms, field):
    a = atoms.copy(); a.info = {"charge": 0.0, "external_field": list(field)}; a.calc = calc
    calc.reset(); return a.get_potential_energy()
def alpha(calc, atoms, axis, h=0.01):
    e = np.zeros(3); e[axis] = h
    return -(energy(calc, atoms, e) - 2 * energy(calc, atoms, 0 * e) + energy(calc, atoms, -e)) / h ** 2 * K_E
for ck in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=ck, device="cpu"); torch.set_num_threads(4)
    a1 = alpha(calc, Atoms("Ag", positions=[[0, 0, 0]]), 2)
    print(f"=== {ck.split('/')[-2]}")
    print(f"  Ag atom   alpha = {a1:7.3f} A^3 = {a1 / AU3:6.1f} a.u.   (reference 50.97 a.u. = 7.553 A^3)")
    for d in (2.53,):
        dimer = Atoms("Ag2", positions=[[0, 0, 0], [0, 0, d]])
        ap, an = alpha(calc, dimer, 2), alpha(calc, dimer, 0)
        print(f"  Ag2 d={d} A  alpha_par = {ap:7.3f} A^3 ({ap / AU3:6.1f} a.u.)  alpha_perp = {an:7.3f} A^3 ({an / AU3:6.1f} a.u.)")
