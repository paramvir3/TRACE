"""Evaluate trained checkpoints on the held-out test set.

Usage: python eval_models.py RUN_DIR [RUN_DIR ...] [--ckpt model.pt|model_last.pt]

Energies and forces come from flashace.calculator.TransformersACECalculator
(the deployment path, energy shift added back), in float64 for every model.
Per-structure errors are written to RUN_DIR/test_errors_<ckpt>.json.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from ase import Atoms  # noqa: E402
from ase.io import read  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E_A_TO_DEBYE = 4.80320
args = [a for a in sys.argv[1:] if not a.startswith("--")]
ckpt_name = "model.pt"
if "--ckpt" in sys.argv:
    ckpt_name = sys.argv[sys.argv.index("--ckpt") + 1]
    args = [a for a in args if a != ckpt_name]
test = read(os.path.join(HERE, "data", "test.xyz"), ":")
DIMER_BINS = [0.0, 2.5, 3.0, 3.5, 4.5, 6.0, 1e9]


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(math.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def stats(rows):
    if not rows:
        return {"n": 0}
    de = np.array([r["dE"] for r in rows]) * 1e3  # meV per structure
    df = np.concatenate([np.asarray(r["dF"]).ravel() for r in rows]) * 1e3  # meV/A
    return {
        "n": len(rows),
        "E_rmse_meV": rms(de),
        "E_mae_meV": float(np.mean(np.abs(de))),
        "E_bias_meV": float(np.mean(de)),
        "E_rmse_meV_per_H2O": rms(de / np.array([r["n_water"] for r in rows])),
        "F_rmse_meV_A": rms(df),
    }


def visibility(atoms, r_max):
    """Per molecule pair: 'inside' (R_OO < r_max), 'partial', or 'invisible'."""
    roo = json.loads(atoms.info["pair_roo"])
    dmin = json.loads(atoms.info["pair_dmin"])
    classes = []
    for r, d in zip(roo, dmin):
        classes.append("inside" if r < r_max else ("invisible" if d > r_max else "partial"))
    return classes


def build_data(calc, atoms):
    """The data dict the calculator builds, for a direct model call."""
    from ase.neighborlist import neighbor_list
    from flashace.charge_state import charge_state_from_atoms, charge_state_tensors

    i, j, shifts = neighbor_list("ijS", atoms, calc.r_max)
    data = {
        "z": torch.tensor(atoms.numbers, dtype=torch.long),
        "pos": torch.tensor(atoms.positions, dtype=calc.dtype),
        "cell": torch.tensor(atoms.cell.array, dtype=calc.dtype),
        "edge_index": torch.stack([torch.tensor(j), torch.tensor(i)]).long(),
        "edge_shift": torch.tensor(shifts, dtype=calc.dtype),
        "volume": torch.tensor(1.0, dtype=calc.dtype),
    }
    if calc.electrostatics is not None:
        data.update(charge_state_tensors(charge_state_from_atoms(atoms), calc.dtype, "cpu"))
    return data


def dipoles(calc, atoms):
    """Total dipole P = sum_i (q_i r_i + mu_i) (e*A), atomic charges, per-molecule dipoles."""
    from trace_electrostatics import polarization

    data = build_data(calc, atoms)
    _, _, _, aux = calc.model(data, training=False, return_electrostatics=True)
    out = aux["electrostatics"][0]
    q = out.solve_result.q.detach()
    mu = out.solve_result.mu.detach()
    pos = out.positions.detach()
    total = polarization(q, pos, mu).detach().numpy()
    s = atoms.get_chemical_symbols()
    oxy = [k for k, x in enumerate(s) if x == "O"]
    mols = {o: [o] for o in oxy}
    for k, x in enumerate(s):
        if x == "H":
            d = [np.linalg.norm(atoms.positions[k] - atoms.positions[o]) for o in oxy]
            mols[oxy[int(np.argmin(d))]].append(k)
    per_mol = []
    for o, members in mols.items():
        idx = torch.tensor(members)
        per_mol.append({
            "charge": float(q[idx].sum()),
            "dipole_about_O": np.linalg.norm(
                ((q[idx, None] * (pos[idx] - pos[o])) + mu[idx]).sum(0).numpy()
            ),
        })
    return total, q.numpy(), per_mol



def main():
    summary = {}
    for run in args:
        run = os.path.abspath(run)
        path = os.path.join(run, ckpt_name)
        calc = TransformersACECalculator(model_path=path, device="cpu")
        r_max = calc.r_max
        ck = torch.load(path, map_location="cpu", weights_only=False)
        epoch = ck.get("epoch")
        rows = []
        t0 = time.perf_counter()
        for atoms in test:
            a = atoms.copy()
            a.calc = calc
            e = a.get_potential_energy()
            f = a.get_forces()
            rows.append({
                "src_index": int(atoms.info["src_index"]),
                "n_water": int(atoms.info["n_water"]),
                "roo_min": float(atoms.info["roo_min"]),
                "dmin_inter": float(atoms.info["dmin_inter"]),
                "vis": visibility(atoms, r_max),
                "E_ref": float(atoms.get_potential_energy()),
                "E_pred": float(e),
                "dE": float(e - atoms.get_potential_energy()),
                "dF": (f - atoms.get_forces()).tolist(),
            })
        t_eval = time.perf_counter() - t0

        groups = {"all": rows}
        for n in (1, 2, 3, 4):
            groups[f"n_water={n}"] = [r for r in rows if r["n_water"] == n]
        dimers = [r for r in rows if r["n_water"] == 2]
        for lo, hi in zip(DIMER_BINS[:-1], DIMER_BINS[1:]):
            groups[f"dimer R_OO [{lo:g},{hi:g})"] = [r for r in dimers if lo <= r["roo_min"] < hi]
        groups["dimer R_OO < r_max"] = [r for r in dimers if r["roo_min"] < r_max]
        groups["dimer R_OO >= r_max"] = [r for r in dimers if r["roo_min"] >= r_max]
        groups["dimer invisible (all inter pairs > r_max)"] = [r for r in dimers if r["vis"] == ["invisible"]]
        clusters = [r for r in rows if r["n_water"] >= 2]
        groups["n>=2, every pair has R_OO < r_max"] = [r for r in clusters if all(v == "inside" for v in r["vis"])]
        groups["n>=2, >=1 pair with R_OO >= r_max"] = [r for r in clusters if any(v != "inside" for v in r["vis"])]
        groups["n>=2, >=1 invisible pair"] = [r for r in clusters if "invisible" in r["vis"]]
        result = {"run": os.path.basename(run), "ckpt": ckpt_name, "epoch": epoch, "r_max": r_max,
                  "eval_seconds_test_set": t_eval, "n_test": len(rows),
                  "groups": {k: stats(v) for k, v in groups.items()}}

        # Locality check: a local model's energy of a dimer with every intermolecular
        # pair beyond r_max must equal the sum of its two monomers' energies.
        checks = []
        for atoms in test:
            if atoms.info["n_water"] != 2 or visibility(atoms, r_max) != ["invisible"]:
                continue
            s = atoms.get_chemical_symbols()
            o = [k for k, x in enumerate(s) if x == "O"]
            part = [[], []]
            for k in range(len(atoms)):
                d = [np.linalg.norm(atoms.positions[k] - atoms.positions[oo]) for oo in o]
                part[int(np.argmin(d))].append(k)
            energies = []
            for sel in [list(range(len(atoms)))] + part:
                m = Atoms(numbers=atoms.numbers[sel], positions=atoms.positions[sel], pbc=False)
                m.calc = calc
                energies.append(m.get_potential_energy())
            checks.append({"src_index": int(atoms.info["src_index"]), "roo": float(atoms.info["roo_min"]),
                           "E_dimer_minus_monomers_meV": 1e3 * (energies[0] - energies[1] - energies[2]),
                           "E_ref_meV": 1e3 * float(atoms.get_potential_energy())})
        result["locality_check"] = checks

        if calc.electrostatics is not None:
            mono = []
            for atoms in test:
                if atoms.info["n_water"] != 1:
                    continue
                total, q, _ = dipoles(calc, atoms)
                p = atoms.positions
                oh = sorted(float(np.linalg.norm(p[k] - p[0])) for k in (1, 2))
                v1, v2 = p[1] - p[0], p[2] - p[0]
                ang = math.degrees(math.acos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)))
                mono.append({"src_index": int(atoms.info["src_index"]), "E_ref_meV": 1e3 * atoms.get_potential_energy(),
                             "r_OH": oh, "HOH_deg": ang, "q": q.tolist(),
                             "dipole_eA": float(np.linalg.norm(total)),
                             "dipole_D": float(np.linalg.norm(total)) * E_A_TO_DEBYE})
            # Experimental gas-phase equilibrium geometry (r_OH 0.9572 A, HOH 104.52 deg).
            half = math.radians(104.52) / 2.0
            eq = Atoms("OH2", positions=[[0, 0, 0],
                                         [0.9572 * math.sin(half), 0, 0.9572 * math.cos(half)],
                                         [-0.9572 * math.sin(half), 0, 0.9572 * math.cos(half)]], pbc=False)
            total, q, _ = dipoles(calc, eq)
            mono.append({"src_index": -1, "E_ref_meV": float("nan"), "r_OH": [0.9572, 0.9572], "HOH_deg": 104.52,
                         "q": q.tolist(), "dipole_eA": float(np.linalg.norm(total)),
                         "dipole_D": float(np.linalg.norm(total)) * E_A_TO_DEBYE})
            result["monomer_dipoles"] = mono
            frag = []
            for atoms in test:
                if atoms.info["n_water"] == 2:
                    total, q, per_mol = dipoles(calc, atoms)
                    frag.append({"src_index": int(atoms.info["src_index"]), "roo": float(atoms.info["roo_min"]),
                                 "vis": visibility(atoms, r_max)[0],
                                 "mol_charges": [m["charge"] for m in per_mol],
                                 "mol_dipoles_eA": [float(m["dipole_about_O"]) for m in per_mol]})
            result["dimer_molecule_moments"] = frag

        with open(os.path.join(run, f"test_errors_{ckpt_name.replace('.pt', '')}.json"), "w") as handle:
            json.dump({"summary": result, "rows": rows}, handle, indent=1)
        summary[os.path.basename(run)] = result

        print(f"\n=== {os.path.basename(run)} [{ckpt_name}, epoch {epoch}, r_max {r_max}] "
              f"test eval {t_eval:.1f} s for {len(rows)} structures")
        print(f"{'group':44s} {'n':>4s} {'E_RMSE':>9s} {'E_MAE':>8s} {'E_bias':>8s} {'E/H2O':>8s} {'F_RMSE':>9s}")
        for k, v in result["groups"].items():
            if v["n"] == 0:
                print(f"{k:44s} {0:4d}")
                continue
            print(f"{k:44s} {v['n']:4d} {v['E_rmse_meV']:9.2f} {v['E_mae_meV']:8.2f} {v['E_bias_meV']:+8.2f} "
                  f"{v['E_rmse_meV_per_H2O']:8.2f} {v['F_rmse_meV_A']:9.2f}")
        for c in checks:
            print(f"  locality: dimer src {c['src_index']} R_OO {c['roo']:.2f} A: "
                  f"E(dimer)-E(A)-E(B) = {c['E_dimer_minus_monomers_meV']:+.4f} meV (ref total E {c['E_ref_meV']:+.1f} meV)")
        if "monomer_dipoles" in result:
            for m in result["monomer_dipoles"]:
                print(f"  monomer src {m['src_index']:4d}: r_OH {m['r_OH'][0]:.3f}/{m['r_OH'][1]:.3f} A, "
                      f"HOH {m['HOH_deg']:6.1f} deg, E_ref {m['E_ref_meV']:6.1f} meV: |P| = {m['dipole_eA']:.4f} e*A "
                      f"= {m['dipole_D']:.3f} D; q(O,H,H) = {np.round(m['q'], 3).tolist()}")


if __name__ == "__main__":
    main()
