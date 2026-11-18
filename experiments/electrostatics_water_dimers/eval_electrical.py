"""Electrical properties of trained models against the PBE0/aug-cc-pVDZ reference.

Usage: python eval_electrical.py RUN_DIR [RUN_DIR ...] [--ckpt model.pt]

For every test structure (data_qm/test.xyz): the model's dipole
P = sum_i (q_i r_i + mu_i) (e*A) and its static polarizability volume (A^3, the
second field derivative of the model's own energy, observables.polarizability),
against the reference dipole and finite-field polarizability.  Reported per
cluster size: dipole RMSE (e*A and D), polarizability-tensor RMSE (A^3),
isotropic polarizability per molecule (model vs reference) and its relative
error, and the charge-transfer / induced-dipole split of the model's alpha.
Writes RUN_DIR/electrical_<ckpt>.json.
"""
import json
import math
import os
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from ase.io import read  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_electrostatics import (  # noqa: E402
    K_E, polarizability, polarizability_contributions, polarization,
)

HERE = os.path.dirname(os.path.abspath(__file__))
E_A_TO_DEBYE = 4.80320
args = [a for a in sys.argv[1:] if not a.startswith("--")]
ckpt_name = "model.pt"
if "--ckpt" in sys.argv:
    ckpt_name = sys.argv[sys.argv.index("--ckpt") + 1]
    args = [a for a in args if a != ckpt_name]
test = read(os.path.join(HERE, "data_qm", "test.xyz"), ":")


def model_data(calc, atoms):
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
    data.update(charge_state_tensors(charge_state_from_atoms(atoms), calc.dtype, "cpu"))
    return data


def electrical(calc, atoms):
    _, _, _, aux = calc.model(model_data(calc, atoms), training=False, return_electrostatics=True)
    out = aux["electrostatics"][0]
    solve = out.solve_result
    P = polarization(solve.q, out.positions, solve.mu).detach().numpy()
    state = out.state()
    alpha = polarizability(state.solve_result, state.assembly, as_volume=True).detach().numpy()
    a_ct, a_ind = polarizability_contributions(state.solve_result, state.assembly)
    return P, alpha, float(np.trace(a_ct.detach().numpy())) * K_E / 3.0, \
        float(np.trace(a_ind.detach().numpy())) * K_E / 3.0


def main():
    for run in args:
        run = os.path.abspath(run)
        calc = TransformersACECalculator(model_path=os.path.join(run, ckpt_name), device="cpu")
        if calc.electrostatics is None:
            print(f"{os.path.basename(run)}: no electrostatics, nothing to evaluate")
            continue
        rows = []
        for atoms in test:
            P, alpha, iso_ct, iso_ind = electrical(calc, atoms)
            # ASE's extxyz reader moves 'dipole' (a calculator property) into calc.results.
            dipole = atoms.calc.results.get("dipole") if atoms.calc is not None else None
            ref_P = np.asarray(atoms.info["dipole"] if dipole is None else dipole, dtype=float)
            ref_a = np.asarray(atoms.info["polarizability"], dtype=float).reshape(3, 3)
            rows.append({
                "n_water": int(atoms.info["n_water"]),
                "dP": (P - ref_P).tolist(),
                "P_model": P.tolist(), "P_ref": ref_P.tolist(),
                "dalpha": (alpha - ref_a).ravel().tolist(),
                "iso_model": float(np.trace(alpha)) / 3.0,
                "iso_ref": float(np.trace(ref_a)) / 3.0,
                "iso_ct": iso_ct, "iso_ind": iso_ind,
            })
        summary = {}
        for label, sel in [("all", rows)] + [(f"n_water={n}", [r for r in rows if r["n_water"] == n])
                                             for n in (1, 2, 3, 4)]:
            if not sel:
                continue
            dP = np.array([r["dP"] for r in sel])
            da = np.array([r["dalpha"] for r in sel])
            n_w = np.array([r["n_water"] for r in sel], dtype=float)
            iso_m = np.array([r["iso_model"] for r in sel]) / n_w
            iso_r = np.array([r["iso_ref"] for r in sel]) / n_w
            summary[label] = {
                "n": len(sel),
                "dipole_rmse_eA": float(math.sqrt(np.mean(dP ** 2))),
                "dipole_rmse_D": float(math.sqrt(np.mean(dP ** 2))) * E_A_TO_DEBYE,
                "alpha_rmse_A3": float(math.sqrt(np.mean(da ** 2))),
                "alpha_iso_per_H2O_model_A3": float(iso_m.mean()),
                "alpha_iso_per_H2O_ref_A3": float(iso_r.mean()),
                "alpha_iso_rel_error_mean": float(np.mean(iso_m / iso_r - 1.0)),
                "alpha_ct_fraction_mean": float(np.mean(
                    [r["iso_ct"] / (r["iso_ct"] + r["iso_ind"]) for r in sel])),
            }
        out = {"run": os.path.basename(run), "ckpt": ckpt_name, "summary": summary, "rows": rows}
        with open(os.path.join(run, f"electrical_{ckpt_name.replace('.pt', '')}.json"), "w") as handle:
            json.dump(out, handle, indent=1)
        print(f"\n=== {os.path.basename(run)} [{ckpt_name}]")
        print(f"{'group':12s} {'n':>4s} {'mu RMSE (D)':>12s} {'alpha RMSE':>11s} {'a_iso/H2O model':>16s} "
              f"{'ref':>7s} {'rel err':>8s} {'CT frac':>8s}")
        for k, v in summary.items():
            print(f"{k:12s} {v['n']:4d} {v['dipole_rmse_D']:12.3f} {v['alpha_rmse_A3']:11.3f} "
                  f"{v['alpha_iso_per_H2O_model_A3']:16.3f} {v['alpha_iso_per_H2O_ref_A3']:7.3f} "
                  f"{v['alpha_iso_rel_error_mean']:+8.3f} {v['alpha_ct_fraction_mean']:8.3f}")


if __name__ == "__main__":
    main()
