"""Long-range electron transfer in Au2/MgO(001):Al -- does the model move charge onto Au2?

Usage: python eval_aumgo_transfer.py RUN_DIR [RUN_DIR ...]   (from this directory; Mac CPU)

DFT (Hirshfeld, data/AuMgO_test.xyz): the Au2 charge is +0.076 e without the three Al
dopants and -0.189 e with them (shift -0.265 e), the nearest Al 10.5-11.2 A away (minimum
image).  Only the energies, forces and total charge were trained on; no charge labels.

For each run and test structure: the energy and force errors, and the model's charge on
Au2 -- total q, and for a TRACE-Redox run its split into the bound (polarization / SQE)
part and the free (redox) part.  Hirshfeld and model charges are different partitions,
so the comparison is the doped-minus-undoped SHIFT and its per-structure correlation, not
the absolute values.

Controls (artifacts ruled out before any claim):
* standard errors of every mean (n = 200 per subsystem);
* ALCHEMICAL control: the doped test geometries with the three Al replaced by Mg
  (same positions).  A shift caused by the dopants' electron must vanish there;
  a shift that survives is geometric (flat vs upright Au2), not electron transfer.

Writes RUN_DIR/eval_aumgo_transfer.json.
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
torch.set_num_threads(int(os.environ.get("EVAL_THREADS", "4")))
from ase.io import read  # noqa: E402

from flashace.calculator import TransformersACECalculator  # noqa: E402

AU, AL = 79, 13
DOPED, UNDOPED = "Al3Au2Mg51O54+0", "Au2Mg54O54+0"
ALCHEMICAL = "doped geometry, Al->Mg"


def main():
    test = read(os.path.join(HERE, os.environ.get("EVAL_DATA_DIR", "data"), "AuMgO_test.xyz"), ":")
    limit = int(os.environ.get("EVAL_LIMIT", "0"))
    if limit:
        test = [a for s in (UNDOPED, DOPED) for a in [x for x in test if x.info["system"] == s][:limit]]
    alchemical = []
    for atoms in test:
        if atoms.info["system"] == DOPED:
            swapped = atoms.copy()
            swapped.info = dict(atoms.info, system=ALCHEMICAL)
            numbers = swapped.numbers.copy()
            numbers[numbers == AL] = 12
            swapped.numbers = numbers
            alchemical.append(swapped)
    for run in sys.argv[1:]:
        calc = TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu")
        captured = []
        hook = calc.model.electrostatics.register_forward_hook(lambda m, i, o: captured.append(o[1]))
        rows = []
        for atoms in test + alchemical:
            a = atoms.copy()
            a.info = dict(atoms.info)
            a.calc = calc
            captured.clear()
            e, f = a.get_potential_energy(), a.get_forces()
            assert len(captured) == 1, "one electrostatics evaluation per structure expected"
            res = captured[0]
            au = torch.as_tensor(atoms.numbers == AU)
            al = torch.as_tensor(atoms.numbers == AL)
            reference = atoms.info["system"] != ALCHEMICAL
            row = {
                "system": atoms.info["system"],
                "dE_meV_atom": 1e3 * (e - atoms.get_potential_energy()) / len(atoms) if reference else float("nan"),
                "dF": (f - atoms.get_forces()).ravel().tolist() if reference else [],
                "q_au2": float(res.q.detach().cpu()[au].sum()),
                "hirshfeld_au2": float(atoms.arrays["hirshfeld_charges"][atoms.numbers == AU].sum()) if reference else float("nan"),
            }
            for name in ("q_bound", "free_charge"):
                value = getattr(res, name, None)
                if value is not None:
                    row[name + "_au2"] = float(value.detach().cpu()[au].sum())
                    if al.any():
                        row[name + "_al"] = float(value.detach().cpu()[al].sum())
            if al.any():
                row["q_al"] = float(res.q.detach().cpu()[al].sum())
            rows.append(row)
        hook.remove()
        summary = {"run": os.path.basename(os.path.normpath(run)), "n": len(rows), "by_system": {}}
        print(f"\n=== {summary['run']}  ({len(rows)} test structures)")
        keys = [k for k in ("q_au2", "q_bound_au2", "free_charge_au2", "hirshfeld_au2") if k in rows[0]]
        for system in (UNDOPED, DOPED, ALCHEMICAL):
            sel = [r for r in rows if r["system"] == system]
            entry = {"n": len(sel)}
            if system != ALCHEMICAL:
                de = np.array([r["dE_meV_atom"] for r in sel])
                df = np.concatenate([np.array(r["dF"]) for r in sel])
                entry.update(E_rmse_meV_atom=float(np.sqrt(np.mean(de ** 2))), F_rmse_eV_A=float(np.sqrt(np.mean(df ** 2))))
            for k in keys + [k for k in ("q_al", "free_charge_al", "q_bound_al") if k in sel[0]]:
                v = np.array([r[k] for r in sel])
                if np.isnan(v).all():
                    continue
                entry[k] = {"mean": float(v.mean()), "std": float(v.std()), "sem": float(v.std(ddof=1) / np.sqrt(len(v)))}
            summary["by_system"][system] = entry
            accuracy = (f"E RMSE {entry['E_rmse_meV_atom']:7.3f} meV/atom  F RMSE {entry['F_rmse_eV_A']:.4f} eV/A"
                        if system != ALCHEMICAL else "(no reference: Al -> Mg at the doped geometries)")
            print(f"  {system:16s} n={entry['n']:3d}  {accuracy}")
            print("      Au2: " + "  ".join(f"{k} {entry[k]['mean']:+.4f}±{entry[k]['sem']:.4f}" for k in keys if k in entry)
                  + "   (mean ± standard error)")
            if "q_al" in entry:
                print("      Al (sum of 3): " + "  ".join(f"{k} {entry[k]['mean']:+.3f}" for k in
                                                     ("q_al", "free_charge_al", "q_bound_al") if k in entry))
        by = summary["by_system"]
        shift = {k: by[DOPED][k]["mean"] - by[UNDOPED][k]["mean"] for k in keys}
        summary["shift_doped_minus_undoped"] = shift
        summary["shift_sem"] = {k: float(np.hypot(by[DOPED][k]["sem"], by[UNDOPED][k]["sem"])) for k in keys}
        model_keys = [k for k in keys if k != "hirshfeld_au2"]
        summary["dopant_effect_at_fixed_geometry"] = {k: by[DOPED][k]["mean"] - by[ALCHEMICAL][k]["mean"] for k in model_keys}
        summary["geometric_effect"] = {k: by[ALCHEMICAL][k]["mean"] - by[UNDOPED][k]["mean"] for k in model_keys}
        print("  dopant effect at fixed geometry (doped - Al->Mg): " + "  ".join(
            f"{k} {v:+.4f}" for k, v in summary["dopant_effect_at_fixed_geometry"].items()))
        print("  geometric effect (Al->Mg - undoped):               " + "  ".join(
            f"{k} {v:+.4f}" for k, v in summary["geometric_effect"].items()))
        rows = [r for r in rows if r["system"] != ALCHEMICAL]
        q = np.array([r["q_au2"] for r in rows])
        h = np.array([r["hirshfeld_au2"] for r in rows])
        summary["pearson_q_au2_vs_hirshfeld"] = float(np.corrcoef(q, h)[0, 1])
        print("  shift doped - undoped (e): " + "  ".join(f"{k} {v:+.4f}±{summary['shift_sem'][k]:.4f}" for k, v in shift.items()))
        print(f"  per-structure Pearson r(model q_Au2, Hirshfeld Au2) = {summary['pearson_q_au2_vs_hirshfeld']:+.3f}")
        with open(os.path.join(run, "eval_aumgo_transfer.json"), "w") as handle:
            json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    main()
