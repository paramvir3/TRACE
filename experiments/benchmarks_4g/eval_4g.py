"""Evaluate trained models on the 4G-HDNNP benchmarks, reproducing the paper's tests.

Ko, Finkler, Goedecker, Behler, Nat. Commun. 12, 398 (2021); arXiv:2009.06484.

Usage: python eval_4g.py DATASET RUN_DIR [RUN_DIR ...]

For every run (``model.pt``, through the ASE calculator, so the total charge
is read from ``info['charge']`` exactly as in deployment):

1. Test RMSE per subsystem (charge state / composition): energy in meV/atom
   (the paper's unit) and forces in eV/A.
2. Carbon_chain: mean |F_model - F_DFT| per atom index, per molecule (the
   paper's Fig. 4). Atom order is the dataset's.
3. Ag_cluster: relax both charge states from the reference minimum (the
   lowest-max-force structure of each charge state in the full dataset,
   whose relaxation trajectories were converged below 0.0015 eV/A), and
   report the RMSD of the model minimum from it (Kabsch-aligned), as in the
   paper's Fig. 5 (2G: 0.420 / 1.291 A; 4G: 0.002 / 0.006 A).
4. NaCl: the paper's construction. Na9Cl8+ is the MEAN geometry of its data
   (Gaussian displacements average out); Na8Cl8+ is the same geometry with
   the distant Na (index 11) deleted. Atom 0 is scanned along the line
   through atom 3, and the model's minimum of the 0-3 Na-Na distance is
   reported. The paper's DFT minima of the two clusters differ by more than
   0.1 A; a local model has one minimum for both (an energy offset only).

Writes RUN_DIR/eval_4g.json.
"""
import json
import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
from ase.io import read  # noqa: E402
from ase.optimize import BFGS  # noqa: E402

from flashace.calculator import TransformersACECalculator  # noqa: E402

#: the raw 4G-HDNNP data (Materials Cloud record 2020.137, its datasets/ folder with <SET>/input.data; not redistributed):
#: $TRACE_4G_RAW, default experiments/benchmarks_4g/raw/datasets
DATA = os.environ.get("TRACE_4G_RAW", os.path.join(HERE, "raw", "datasets"))


def predict(calc, atoms):
    a = atoms.copy()
    a.info = dict(atoms.info)
    a.calc = calc
    return a.get_potential_energy(), a.get_forces()


def kabsch_rmsd(p, q):
    p = p - p.mean(0)
    q = q - q.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum(1).mean()))


def reference_minima(dataset):
    """Lowest max-force structure per subsystem in the FULL dataset."""
    sys.path.insert(0, HERE)
    from prepare import read_runner

    frames = read_runner(os.path.join(DATA, dataset, "input.data"))
    best = {}
    for f in frames:
        fmax = float(np.abs(f.get_forces()).max())
        key = f.info["system"]
        if key not in best or fmax < best[key][0]:
            best[key] = (fmax, f)
    return {k: v[1] for k, v in best.items()}, {k: v[0] for k, v in best.items()}


def relax(calc, atoms, fmax=0.005, steps=500):
    a = atoms.copy()
    a.info = dict(atoms.info)
    a.calc = calc
    BFGS(a, logfile=None).run(fmax=fmax, steps=steps)
    return a


def nacl_scan(calc, atoms, distances):
    """Energy of the cluster with atom 0 moved along the line through atom 3."""
    base = atoms.copy()
    direction = base.positions[0] - base.positions[3]
    direction /= np.linalg.norm(direction)
    energies = []
    for d in distances:
        a = base.copy()
        a.info = dict(atoms.info)
        a.positions[0] = base.positions[3] + d * direction
        a.calc = calc
        energies.append(a.get_potential_energy())
    energies = np.array(energies)
    k = int(np.argmin(energies))
    if 0 < k < len(distances) - 1:                 # parabola through the three lowest points
        x, y = distances[k - 1:k + 2], energies[k - 1:k + 2]
        c = np.polyfit(x, y, 2)
        return float(-c[1] / (2 * c[0])), energies
    return float(distances[k]), energies


def nacl_references():
    """Na9Cl8+ at the mean geometry of its data; Na8Cl8+ = the same minus atom 11."""
    sys.path.insert(0, HERE)
    from prepare import read_runner

    frames = [f for f in read_runner(os.path.join(DATA, "NaCl", "input.data"))
              if f.info["system"] == "Cl8Na9+1"]
    na9 = frames[0].copy()
    na9.positions = np.mean([f.positions for f in frames], axis=0)
    na9.info = {"charge": 1.0, "system": "Cl8Na9+1"}
    na8 = na9.copy()
    del na8[11]
    na8.info = {"charge": 1.0, "system": "Cl8Na8+1"}
    return {"Cl8Na9+1": na9, "Cl8Na8+1": na8}


def main():
    dataset, runs = sys.argv[1], sys.argv[2:]
    # EVAL_DATA_DIR=data_pfn scores the protocol runs on their own held-out test set
    test = read(os.path.join(HERE, os.environ.get("EVAL_DATA_DIR", "data"), f"{dataset}_test.xyz"), ":")
    minima, minima_fmax = reference_minima(dataset) if dataset == "Ag_cluster" else ({}, {})
    for run in runs:
        calc = TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu")
        rows = []
        for atoms in test:
            e, f = predict(calc, atoms)
            rows.append((atoms.info["system"], len(atoms),
                         (e - atoms.get_potential_energy()) / len(atoms),
                         f - atoms.get_forces()))
        result = {"run": os.path.basename(run), "subsystems": {}}
        print(f"\n=== {dataset} / {os.path.basename(run)}")
        for system in sorted({r[0] for r in rows}):
            sel = [r for r in rows if r[0] == system]
            de = np.array([r[2] for r in sel]) * 1e3
            df = np.concatenate([r[3].ravel() for r in sel])
            entry = {"n": len(sel), "E_rmse_meV_atom": float(np.sqrt(np.mean(de ** 2))),
                     "F_rmse_eV_A": float(np.sqrt(np.mean(df ** 2)))}
            result["subsystems"][system] = entry
            print(f"  {system:10s} n={entry['n']:4d}  E RMSE {entry['E_rmse_meV_atom']:9.3f} meV/atom"
                  f"  F RMSE {entry['F_rmse_eV_A']:.4f} eV/A")
        if dataset == "Carbon_chain":
            result["force_error_per_atom"] = {}
            for system in sorted({r[0] for r in rows}):
                sel = [r for r in rows if r[0] == system]
                per_atom = np.mean([np.linalg.norm(r[3], axis=1) for r in sel], axis=0)
                result["force_error_per_atom"][system] = per_atom.tolist()
                print(f"  |dF| per atom {system}: " + " ".join(f"{x:.3f}" for x in per_atom))
        if dataset == "Ag_cluster":
            result["relaxation_rmsd_A"] = {}
            for system, ref in minima.items():
                relaxed = relax(calc, ref)
                rmsd = kabsch_rmsd(relaxed.positions, ref.positions)
                result["relaxation_rmsd_A"][system] = rmsd
                print(f"  relaxed {system}: RMSD from reference minimum {rmsd:.3f} A "
                      f"(reference max force {minima_fmax[system]:.4f} eV/A)")
        if dataset == "NaCl":
            distances = np.linspace(3.15, 3.70, 23)
            result["scan_minimum_A"] = {}
            for system, ref in nacl_references().items():
                d_min, _ = nacl_scan(calc, ref, distances)
                result["scan_minimum_A"][system] = d_min
                print(f"  {system}: scan minimum of the atom 0 - atom 3 distance {d_min:.3f} A")
            values = list(result["scan_minimum_A"].values())
            print(f"  difference of the two minima: {abs(values[0] - values[1]):.3f} A "
                  "(paper, DFT: > 0.1 A; a local model: 0 by construction)")
        with open(os.path.join(run, "eval_4g.json"), "w") as handle:
            json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()
