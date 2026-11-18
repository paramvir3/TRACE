"""Step 21's failure, diagnosed on frozen weights (2026-10-28; a probe, no model change).

Usage (repo root):
    python experiments/benchmarks_4g/probe_step21_diagnostics.py separated MODEL.pt [...]
    python experiments/benchmarks_4g/probe_step21_diagnostics.py relaxed MODEL.pt [...]

D1 'separated': the undoped wetting reference with Au2 moved to mid-vacuum (8 A from either slab face, as
probe_t5_first_order's separated check) has dE = E(AB) - E(A in the cell) - E(B) = +1343 meV (s42) and +1.54 meV
(s43) under the training selection; steps 17 and 20 give -0.14 / -0.12 meV.  Hypothesis: SPEC 21's amplitude closure
truncates the product space once couplings are 1-4 eV (strong mixing), so the selected ensemble of AB is not the
product of A's and B's.  dE is measured at a_off = 0.05 (training), 0.02 and 0.005 (a_on = 4 a_off), cap 20000.
Pre-registered reading: the hypothesis holds if |dE| at a_off = 0.005 is below 10 % of its value at 0.05; it is
falsified if |dE| does not fall.

D3 'relaxed': the undoped wetting (flat) reference relaxed as eval_aumgo.py does (Au free, slab fixed, BFGS fmax 0.005);
reported: the Au positions against the reference (heights above the surface, nearest O and cation distances), the
Au2 bond, the free charges, the ensemble's classes and the energy drop along the relaxation.  Descriptive.
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
from trace_redox.constants import formal_frame  # noqa: E402
from trace_redox.diabatic import density_matrix  # noqa: E402
from eval_aumgo import AU, references, relax  # noqa: E402


def energy(calc, atoms, charge):
    a = atoms.copy()
    a.info = {"charge": float(charge)}
    a.calc = calc
    calc.reset()
    E = float(a.get_potential_energy())
    return E, int(calc.model.electrostatics._ensemble_last["configs"].shape[0])


def separated(model):
    calc = TransformersACECalculator(model_path=model, device="cpu")
    torch.set_num_threads(2)
    mod = calc.model.electrostatics
    base = mod.redox
    ab = references()[("undoped", "wetting")].copy()
    ab.positions[AU, 2] += 29.15 - float(ab.positions[AU, 2].mean())
    au = Atoms("Au2", positions=ab.positions[AU], cell=ab.cell, pbc=ab.pbc)
    slab = ab[:AU[0]]
    rows = []
    for a_off in (0.05, 0.02, 0.005):
        mod.redox = dataclasses.replace(base, selection_amplitude_off=a_off, selection_amplitude_on=4.0 * a_off,
                                        fragment_max_configurations=20000)
        try:
            (eAB, nAB), (eA, nA), (eB, nB) = energy(calc, ab, 0), energy(calc, au, 0), energy(calc, slab, 0)
            dE = eAB - eA - eB
            rows.append((a_off, dE))
            print(f"  a_off {a_off}: dE {dE * 1e3:+.3f} meV; configurations AB {nAB}, A {nA}, B {nB} "
                  f"(E_AB {eAB:.6f}, E_A {eA:.6f}, E_B {eB:.6f})", flush=True)
        except Exception as exc:                                   # a refused closure is a result too
            print(f"  a_off {a_off}: refused ({type(exc).__name__}: {str(exc)[:120]})", flush=True)
    mod.redox = base
    if len(rows) == 3:
        holds = abs(rows[2][1]) < 0.1 * abs(rows[0][1])
        print(f"  D1 reading: |dE| {abs(rows[0][1]) * 1e3:.3f} -> {abs(rows[2][1]) * 1e3:.3f} meV -> truncation hypothesis "
              f"{'holds' if holds else 'falsified'}", flush=True)


def describe(calc, atoms, label):
    a = atoms.copy()
    a.info = {"charge": 0.0}
    a.calc = calc
    calc.reset()
    E = float(a.get_potential_energy())
    mod = calc.model.electrostatics
    last = mod._ensemble_last
    z = torch.as_tensor(a.numbers)
    n = len(a)
    fr = formal_frame(z)
    p = torch.diagonal(density_matrix(last["H"], float(last["kT"])))
    q = (p @ last["configs"][:, :n].to(torch.float64)).tolist()
    sym = a.get_chemical_symbols()
    top = a.positions[:AU[0], 2].min()                              # the slab face toward the Au (slab above)
    o = [i for i, s in enumerate(sym) if s == "O"]
    cat = [i for i, s in enumerate(sym) if s in ("Mg", "Al")]
    info = []
    for k in AU:
        d_o = a.get_distances(k, o, mic=True)
        d_c = a.get_distances(k, cat, mic=True)
        info.append(f"Au{k}: height {top - a.positions[k, 2]:.3f} A, nearest O {d_o.min():.3f} (n<2.6: "
                    f"{int((d_o < 2.6).sum())}), nearest cation {d_c.min():.3f}, charge {q[k] - int(fr[k]):+.3f}")
    classes = {}
    for c in range(int(last["configs"].shape[0])):
        ch = (last["configs"][c, :n] != fr).nonzero().flatten().tolist()
        key = tuple(sorted((sym[j], int(last["configs"][c, j]) - int(fr[j])) for j in ch))
        band = int(last["configs"][c, n:].abs().sum()) if last["configs"].shape[1] > n else 0
        classes[(key, band)] = classes.get((key, band), 0.0) + float(p[c])
    top_classes = sorted(classes.items(), key=lambda kv: -kv[1])[:5]
    print(f"  {label}: E {E:.4f} eV; Au-Au {a.get_distance(AU[0], AU[1], mic=True):.3f} A; " + "; ".join(info), flush=True)
    print(f"    {len(last['configs'])} configurations; populations " +
          ", ".join(f"{' + '.join(f'{s}{d:+d}' for s, d in k) or 'frame'} | band {b}: {v:.3f}" for (k, b), v in top_classes),
          flush=True)
    return E


def relaxed(model):
    calc = TransformersACECalculator(model_path=model, device="cpu")
    torch.set_num_threads(2)
    mod = calc.model.electrostatics
    mod.redox = dataclasses.replace(mod.redox, fragment_max_configurations=20000)
    ref = references()[("undoped", "wetting")]
    e0 = describe(calc, ref, "reference (flat, unrelaxed)")
    out, e1 = relax(calc, ref)
    e1 = describe(calc, out, "relaxed")
    print(f"  relaxation energy {(e1 - e0) * 1e3:+.1f} meV", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    for m in sys.argv[2:]:
        print(f"=== {os.path.basename(os.path.dirname(m))} [{mode}]", flush=True)
        {"separated": separated, "relaxed": relaxed}[mode](m)
