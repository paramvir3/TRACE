"""Au2 on MgO(001): the periodic 4G-HDNNP benchmark (Ko et al. 2021, Table II / Figs. 9-11).

Usage: python eval_aumgo.py [--states] RUN_DIR [RUN_DIR ...]

--states declares the SPEC 21 charge states of annotate_charge_states.py
(substrate / Au2; (0, 0), (+1, -1), (-1, +1); through-bond coupling) on every
structure, test and relaxation alike: required for a model trained with them,
which is a different energy function without them.

1. Test RMSE per substrate (doped / undoped): energy meV/atom, forces eV/A.
2. Table II: E(wetting) - E(non-wetting) with the Au positions optimised and
   the substrate fixed, for the doped and the undoped slab.  The substrate of
   each doping is the MEAN geometry of its data (the 0.02 A Gaussian
   displacements average out), and BOTH Au2 geometries are placed on that
   same substrate, so no substrate noise enters the difference.  Starting Au
   positions: the mean Au positions of each (doping, geometry) class.
   Paper: DFT -2.7 / 928.9 meV (doped / undoped); 2G 375.0 / 375.0;
   4G -41.0 / 975.4.  A local model cannot see the Al dopants (> 10 A away),
   so its two differences must be equal.
3. The relaxed non-wetting Au-O bond (lower Au to its nearest O).  Paper:
   DFT 2.332 / 2.190 A, 4G 2.342 / 2.177, 2G 2.256 for both.
Writes RUN_DIR/eval_aumgo.json.
"""
import json
import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)
from ase.constraints import FixAtoms  # noqa: E402
from ase.io import read  # noqa: E402
from ase.optimize import BFGS  # noqa: E402

from flashace.calculator import TransformersACECalculator  # noqa: E402
from prepare import read_runner  # noqa: E402

#: the raw 4G-HDNNP data (Materials Cloud record 2020.137, its datasets/ folder with <SET>/input.data; not redistributed):
#: $TRACE_4G_RAW, default experiments/benchmarks_4g/raw/datasets
DATA = os.environ.get("TRACE_4G_RAW", os.path.join(HERE, "raw", "datasets"))
AU = [108, 109]
STATES = np.array([[0, 0], [1, -1], [-1, 1]], dtype=float)


def with_states(atoms):
    """The SPEC 21 charge states of annotate_charge_states.py, declared on a copy."""
    a = atoms.copy()
    a.info = dict(atoms.info)
    labels = np.zeros(len(a), dtype=int)
    labels[AU] = 1
    a.arrays["charge_state_fragment"] = labels
    a.info["charge_state_charges"] = STATES.ravel()
    a.info["charge_state_coupling"] = "through_bond"
    return a


def geometry_class(atoms):
    v = atoms.positions[AU[1]] - atoms.positions[AU[0]]
    angle = np.degrees(np.arccos(abs(v[2]) / np.linalg.norm(v)))
    return "wetting" if angle > 45 else "non-wetting"


def references():
    """{(doping, geometry): Atoms} on one mean substrate per doping."""
    frames = read_runner(os.path.join(DATA, "AuMgO", "input.data"))
    out = {}
    for doping in ("doped", "undoped"):
        members = [f for f in frames if ("Al" in f.get_chemical_symbols()) == (doping == "doped")]
        substrate = np.mean([f.positions[:108] for f in members], axis=0)
        for geometry in ("wetting", "non-wetting"):
            cls = [f for f in members if geometry_class(f) == geometry]
            a = cls[0].copy()
            a.positions[:108] = substrate
            a.positions[AU] = np.mean([f.positions[AU] for f in cls], axis=0)
            a.info = {"charge": 0.0}
            a.set_constraint(FixAtoms(indices=list(range(108))))
            out[(doping, geometry)] = a
    return out


def relax(calc, atoms, fmax=0.005, steps=400, states=False):
    a = with_states(atoms) if states else atoms.copy()
    a.info = dict(a.info)
    a.set_constraint(FixAtoms(indices=list(range(108))))
    a.calc = calc
    BFGS(a, logfile=None).run(fmax=fmax, steps=steps)
    return a, a.get_potential_energy()


def au_o_bond(atoms):
    low = AU[int(np.argmax(atoms.positions[AU, 2]))]       # the Au nearest the slab (slab is above, larger z)
    oxygens = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "O"]
    d = atoms.get_distances(low, oxygens, mic=True)
    return float(d.min())


def main():
    args = sys.argv[1:]
    states = "--states" in args
    runs = [a for a in args if a != "--states"]
    test = read(os.path.join(HERE, "data", "AuMgO_test.xyz"), ":")
    refs = references()
    for run in runs:
        calc = TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu")
        result = {"run": os.path.basename(run), "subsystems": {}, "charge_states": states}
        print(f"\n=== AuMgO / {os.path.basename(run)}{'  (with SPEC 21 charge states)' if states else ''}")
        for doping in ("doped", "undoped"):
            sel = [a for a in test if ("Al" in a.get_chemical_symbols()) == (doping == "doped")]
            de, df = [], []
            for atoms in sel:
                a = with_states(atoms) if states else atoms.copy()
                a.info = dict(a.info)
                a.calc = calc
                de.append((a.get_potential_energy() - atoms.get_potential_energy()) / len(a) * 1e3)
                df.append((a.get_forces() - atoms.get_forces()).ravel())
            entry = {"n": len(sel), "E_rmse_meV_atom": float(np.sqrt(np.mean(np.square(de)))),
                     "F_rmse_eV_A": float(np.sqrt(np.mean(np.square(np.concatenate(df)))))}
            result["subsystems"][doping] = entry
            print(f"  {doping:8s} n={entry['n']}  E RMSE {entry['E_rmse_meV_atom']:.3f} meV/atom"
                  f"  F RMSE {entry['F_rmse_eV_A']:.4f} eV/A")
        energies, bonds = {}, {}
        for key, atoms in refs.items():
            relaxed, energy = relax(calc, atoms, states=states)
            energies[key] = energy
            if key[1] == "non-wetting":
                bonds[key[0]] = au_o_bond(relaxed)
        result["table2_meV"] = {d: 1e3 * (energies[(d, "wetting")] - energies[(d, "non-wetting")])
                                for d in ("doped", "undoped")}
        result["au_o_bond_A"] = bonds
        print(f"  E(wetting) - E(non-wetting): doped {result['table2_meV']['doped']:+.1f} meV, "
              f"undoped {result['table2_meV']['undoped']:+.1f} meV   (DFT -2.7 / +928.9; 2G 375.0 / 375.0; "
              "4G -41.0 / +975.4)")
        print(f"  relaxed non-wetting Au-O: doped {bonds['doped']:.3f} A, undoped {bonds['undoped']:.3f} A"
              "   (DFT 2.332 / 2.190; 4G 2.342 / 2.177; 2G 2.256 / 2.256)")
        with open(os.path.join(run, "eval_aumgo.json"), "w") as handle:
            json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()
