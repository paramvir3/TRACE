"""Round-5 protocol runs vs the published 4G-HDNNP and PFN numbers, like for like.

Usage: python compare_protocol.py DATASET RUN_DIR [RUN_DIR ...]

* pooled force RMSE over every force component of the held-out
  data_pfn/{DATASET}_test.xyz (all charge states pooled) -- the quantity PFN
  plot in Fig. 2 (their validation set; ours is a test set never used for
  training or early stopping);
* for Ag_cluster and Carbon_chain, the 4G paper's errors at the lowest-max-force
  structure of each charge state (Table I for the chain), flagged when that
  structure is in our training set.
Published numbers: compare_published.PAPER_4G / PFN_FIG2 (sources documented there).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_published as cp  # noqa: E402  (float64, thread count, published numbers)
import eval_4g  # noqa: E402
from ase.io import read  # noqa: E402


def main():
    dataset, runs = sys.argv[1], sys.argv[2:]
    test = read(os.path.join(HERE, "data_pfn", f"{dataset}_test.xyz"), ":")
    train = read(os.path.join(HERE, "data_pfn", f"{dataset}_train.xyz"), ":")
    published = cp.PFN_FIG2[dataset]
    print(f"{dataset}: PFN Fig. 2 validation force RMSE (eV/A, digitised): " +
          ", ".join(f"{k} {v if isinstance(v, float) else v[0]}" for k, v in published.items()))
    out = {}
    for run in runs:
        calc = cp.calculator(run)
        pooled, per = [], {}
        for atoms in test:
            _, f = eval_4g.predict(calc, atoms)
            d = (f - atoms.get_forces()).ravel()
            pooled.append(d)
            per.setdefault(atoms.info["system"], []).append(d)
        rmse = float(np.sqrt(np.mean(np.concatenate(pooled) ** 2)))
        row = {"pooled_force_rmse_eV_A": rmse, "n_test": len(test),
               "by_system": {k: float(np.sqrt(np.mean(np.concatenate(v) ** 2))) for k, v in per.items()}}
        print(f"  {os.path.basename(run):28s} pooled force RMSE {rmse:.4f} eV/A  " +
              "  ".join(f"{k} {v:.4f}" for k, v in row["by_system"].items()))
        if dataset in ("Ag_cluster", "Carbon_chain"):
            minima, fmax = eval_4g.reference_minima(dataset)
            for system, ref in minima.items():
                e, f = eval_4g.predict(calc, ref)
                err = abs(e - ref.get_potential_energy()) / len(ref) * 1e3
                df = f - ref.get_forces()
                inside = cp.in_training_set(ref, train)
                row.setdefault("minima", {})[system] = {
                    "energy_error_meV_atom": float(err), "in_training_set": inside,
                    "mean_force_error_norm_eV_A": float(np.linalg.norm(df, axis=1).mean()),
                    "mean_force_error_component_eV_A": float(np.abs(df).mean())}
                if dataset == "Ag_cluster":
                    ref_4g = f"4G {cp.PAPER_4G['ag3_minimum_energy_error_meV_atom'][system]['4G']}"
                else:                                   # Table I: (meV/atom, 1e-3 e, eV/A)
                    e4, _, f4 = cp.PAPER_4G["table1"][system]["4G"]
                    ref_4g = f"4G {e4} meV/atom, F {f4} eV/A"
                print(f"      at the {system} minimum (max |F_DFT| {fmax[system]:.4f}): |dE| {err:.3f} meV/atom, "
                      f"<|dF|> {row['minima'][system]['mean_force_error_norm_eV_A']:.4f} (norm) / "
                      f"{row['minima'][system]['mean_force_error_component_eV_A']:.4f} (component) eV/A "
                      f"({ref_4g}); in our training set: {inside}")
        out[os.path.basename(run)] = row
        with open(os.path.join(run, "compare_protocol.json"), "w") as handle:        # per run (2026-11-08)
            json.dump(row, handle, indent=1)
    shared = os.path.join(HERE, "runs", "protocol", f"compare_{dataset}.json")
    merged = json.load(open(shared)) if os.path.exists(shared) else {}   # merge, never drop other runs (2026-11-08)
    merged.update(out)
    with open(shared, "w") as handle:
        json.dump(merged, handle, indent=1)


if __name__ == "__main__":
    main()
