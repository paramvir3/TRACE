# Diagnostic probe (2026-10-18; reported).  Usage (repo root): python experiments/benchmarks_4g/probe_dimer_ip_ea.py SYMBOL D1,D2,... CKPT [...]
"""The model's vertical electron affinity and ionisation energy of an isolated atom and homonuclear dimer:
E(q = -1) - E(0) and E(+1) - E(0) of the whole model (configuration ensemble, couplings, shifts), open boundaries.
References (PBE0/def2-TZVPPD, dimer minus atom added to the exact atomic values) in the benchmarks README."""
import sys, warnings
import torch
warnings.simplefilter("ignore")
sys.path.insert(0, "."); torch.set_default_dtype(torch.float64)
from ase import Atoms
from flashace.calculator import TransformersACECalculator
def energy(calc, atoms, q):
    a = atoms.copy(); a.info = {"charge": float(q)}; a.calc = calc; calc.reset(); return a.get_potential_energy()
sym, ds = sys.argv[1], [float(x) for x in sys.argv[2].split(",")]
for ck in sys.argv[3:]:
    calc = TransformersACECalculator(model_path=ck, device="cpu"); torch.set_num_threads(2)
    at = Atoms(sym, positions=[[0, 0, 0]])
    e0 = energy(calc, at, 0)
    print(f"{ck.split('/')[-2]:60s} {sym} atom EA {e0 - energy(calc, at, -1):.3f}  IP {energy(calc, at, 1) - e0:.3f} eV")
    for d in ds:
        dim = Atoms(sym + "2", positions=[[0, 0, 0], [0, 0, d]])
        e0 = energy(calc, dim, 0)
        print(f"    {sym}2 d={d:.2f}: EA {e0 - energy(calc, dim, -1):.3f}  IP {energy(calc, dim, 1) - e0:.3f} eV")
