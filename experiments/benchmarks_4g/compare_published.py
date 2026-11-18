"""TRACE vs the published 4G-HDNNP and PFN (NequIP) numbers on the 4G benchmarks.

Usage: python compare_published.py            -> writes comparison.json

Published numbers, and where each one comes from (nothing here is estimated
by us except where marked DIGITISED):

* 4G-HDNNP paper (Ko, Finkler, Goedecker, Behler, Nat. Commun. 12, 398 (2021);
  arXiv:2009.06484), read from the arXiv PDF:
  - Table I: energy error (meV/atom), mean charge error (1e-3 e) and mean
    force error (eV/A) of C10H2 / C10H3+ AT THE DFT-OPTIMISED STRUCTURES.
  - Ag3 text: energy errors at the optimised structures, 4G 1.166 (Ag3-) /
    0.320 (Ag3+) meV/atom; 2G 0.605 / 2.017 eV/atom.
  - Table II: E(wetting) - E(non-wetting) of Au2 on MgO(001), meV.
  - Fig. 11 text: relaxed non-wetting Au-O bond, A.
  - Methods: 90 % of each data set for training, 10 % test.
  - Figs. 7 / 8 (NaCl scan energies and forces): extracted EXACTLY from the
    PDF's vector paths (pdftocairo -svg; the 50 plotted points per curve,
    mapped through a linear fit to the tick marks, residual < 3e-5 in data
    units) into published/ko2021_nacl_fig7_fig8.json.  The energy minima and
    the force zeros agree to 1e-3 A: DFT 3.366 (Na9Cl8+) / 3.473 (Na8Cl8+) A,
    4G 3.364 / 3.473, 3G 3.404 / 3.421, 2G 3.415 / 3.415.
  The paper's test-set RMSEs are in its SI, which we have not read; the values
  below labelled "test RMSE" for 2G/3G/4G are the dotted lines of the PFN
  figure (next item), which PFN drew as the 4G-paper reference.
* PFN (pfnet-research/charge_transfer_nnp; blog "Neural network potential with
  charge transfer", 2022): the ONLY published accuracy numbers are Fig. 2,
  validation force RMSE (eV/A) vs epoch.  DIGITISED from the 490x244 PNG:
  log axis, 56 px per decade, gridlines located to sub-pixel; dotted lines by
  intensity-weighted row centre; NequIP curves as the intensity-weighted centre
  of the colour band in the last 4 pixel columns (the final epochs).  One pixel
  is 4 % in value; the band half-widths are listed.  Training sizes from the
  repo configs: n_train 8000 / n_val 1000 (Au2-MgO 4000 / 500), r_max 4 A,
  baseline trained on forces only, charge/QEq on forces (100) + Hirshfeld
  charges (1).

TRACE numbers are computed here from the trained runs, on our held-out test
set (1000 structures, never used for training or early stopping), with the
ASE calculator as in deployment.
"""
import json
import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval_4g  # noqa: E402  (sets float64 and the thread count)
from ase.io import read  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

PAPER_4G = {
    "table1": {   # energy meV/atom, charges 1e-3 e, forces eV/A, at the DFT minima
        "C10H2+0": {"2G": (0.684, None, 0.095), "3G (unscaled)": (1.255, 19.72, 0.430),
                    "3G (scaled)": (2.193, 10.76, 0.138), "4G": (0.463, 4.820, 0.032)},
        "C10H3+1": {"2G": (0.922, None, 0.127), "3G (unscaled)": (0.046, 17.82, 0.658),
                    "3G (scaled)": (1.425, 17.72, 0.259), "4G": (0.176, 5.048, 0.042)},
    },
    "ag3_minimum_energy_error_meV_atom": {"Ag3-1": {"4G": 1.166, "2G": 605.0},
                                          "Ag3+1": {"4G": 0.320, "2G": 2017.0}},
    "ag3_relaxation_rmsd_A": {"Ag3+1": {"2G": 0.420, "4G": 0.002}, "Ag3-1": {"2G": 1.291, "4G": 0.006}},
    "table2_meV": {"doped": {"DFT": -2.7, "2G": 375.0, "4G": -41.0},
                   "undoped": {"DFT": 928.9, "2G": 375.0, "4G": 975.4}},
    "au_o_bond_A": {"doped": {"DFT": 2.332, "2G": 2.256, "4G": 2.342},
                    "undoped": {"DFT": 2.190, "2G": 2.256, "4G": 2.177}},
    "n_train_fraction": 0.9,
}

# DIGITISED from PFN Fig. 2: validation force RMSE, eV/A.  (centre, low, high)
PFN_FIG2 = {
    "Carbon_chain": {"2G": 0.1278, "3G": 0.2276, "4G": 0.0757,
                     "NequIP": (0.0495, 0.0458, 0.0518), "NequIP+E_ele": (0.3156, 0.2913, 0.3433),
                     "NequIP+QEq": (0.0601, 0.0540, 0.0663)},
    "NaCl": {"2G": 0.0574, "3G": 0.0764, "4G": 0.0329,
             "NequIP": (0.0104, 0.0104, 0.0104), "NequIP+E_ele": (0.0160, 0.0145, 0.0178),
             "NequIP+QEq": (0.0117, 0.0109, 0.0128)},
    "Ag_cluster": {"2G": 1.7783, "3G": 1.9012, "4G": 0.0310,
                   "NequIP": (0.0971, 0.0884, 0.1086), "NequIP+E_ele": (0.0252, 0.0237, 0.0268),
                   "NequIP+QEq": (0.0204, 0.0185, 0.0228)},
    "AuMgO": {"2G": 0.1509, "4G": 0.0667,
              "NequIP": (0.0090, 0.0078, 0.0109), "NequIP+E_ele": (0.1861, 0.1707, 0.2012),
              "NequIP+QEq": (0.0843, 0.0750, 0.0921)},
}
PFN_N_TRAIN = {"Carbon_chain": 8000, "NaCl": 8000, "Ag_cluster": 8000, "AuMgO": 4000}
DATASET_SIZE = {"Carbon_chain": 10019, "NaCl": 5000, "Ag_cluster": 11013, "AuMgO": 5000}
RUNS = {"TRACE local": "base_s42", "TRACE + electrostatics": "elec_s42"}
SCAN_DISTANCES = np.linspace(3.15, 3.70, 45)        # the paper's Fig. 7 range, 0.0125 A steps


def calculator(run):
    return TransformersACECalculator(model_path=os.path.join(run, "model.pt"), device="cpu")


def pooled_force_rmse(calc, dataset):
    """Force RMSE over every force component of the whole test set (all charge states pooled),
    the quantity PFN plot.  Also the per-subsystem values."""
    test = read(os.path.join(HERE, "data", f"{dataset}_test.xyz"), ":")
    per, pooled = {}, []
    for atoms in test:
        _, f = eval_4g.predict(calc, atoms)
        d = (f - atoms.get_forces()).ravel()
        pooled.append(d)
        per.setdefault(atoms.info["system"], []).append(d)
    pooled = np.concatenate(pooled)
    return (float(np.sqrt(np.mean(pooled ** 2))),
            {k: float(np.sqrt(np.mean(np.concatenate(v) ** 2))) for k, v in per.items()})


def in_training_set(atoms, train):
    return any(len(t) == len(atoms) and np.allclose(t.positions, atoms.positions, atol=1e-6)
               for t in train)


def minimum_errors(calc, dataset):
    """Errors at the lowest-max-force structure of each charge state (the paper's DFT minima:
    the data include the relaxation trajectories converged to 0.001-0.0015 eV/A)."""
    minima, fmax = eval_4g.reference_minima(dataset)
    train = read(os.path.join(HERE, "data", f"{dataset}_train.xyz"), ":")
    out = {}
    for system, ref in minima.items():
        e, f = eval_4g.predict(calc, ref)
        df = f - ref.get_forces()
        out[system] = {
            "reference_max_force_eV_A": fmax[system],
            "in_our_training_set": in_training_set(ref, train),
            "energy_error_meV_atom": float(abs(e - ref.get_potential_energy()) / len(ref) * 1e3),
            "mean_force_error_norm_eV_A": float(np.linalg.norm(df, axis=1).mean()),
            "mean_force_error_component_eV_A": float(np.abs(df).mean()),
        }
    return out


def main():
    result = {"paper_4g": PAPER_4G, "pfn_fig2_digitised": PFN_FIG2, "pfn_n_train": PFN_N_TRAIN,
              "trace": {}}
    for dataset in ("Carbon_chain", "NaCl", "Ag_cluster", "AuMgO"):
        result["trace"][dataset] = {}
        for label, suffix in RUNS.items():
            run = os.path.join(HERE, "runs", f"{dataset}_{suffix}")
            if not os.path.exists(os.path.join(run, "model.pt")):
                continue
            if dataset == "AuMgO" and not os.path.exists(os.path.join(run, "eval_aumgo.json")):
                continue                                  # still training
            calc = calculator(run)
            pooled, per = pooled_force_rmse(calc, dataset)
            entry = {"n_train": len(read(os.path.join(HERE, "data", f"{dataset}_train.xyz"), ":")),
                     "pooled_force_rmse_eV_A": pooled, "force_rmse_by_system_eV_A": per}
            if dataset in ("Carbon_chain", "Ag_cluster"):
                entry["at_minima"] = minimum_errors(calc, dataset)
            if dataset == "AuMgO":
                with open(os.path.join(run, "eval_aumgo.json")) as handle:
                    entry["aumgo"] = json.load(handle)
            if dataset == "NaCl":
                entry["scan"] = {}
                for system, ref in eval_4g.nacl_references().items():
                    d_min, energies = eval_4g.nacl_scan(calc, ref, SCAN_DISTANCES)
                    entry["scan"][system] = {"minimum_A": d_min, "d_A": SCAN_DISTANCES.tolist(),
                                             "E_eV": energies.tolist()}
            result["trace"][dataset][label] = entry
            print(f"{dataset:13s} {label:24s} pooled F RMSE {pooled:.4f} eV/A"
                  + "".join(f"  | {k} at min: dE {v['energy_error_meV_atom']:.3f} meV/atom, "
                            f"<|dF|> {v['mean_force_error_norm_eV_A']:.4f} eV/A"
                            f"{' (in train)' if v['in_our_training_set'] else ''}"
                            for k, v in entry.get("at_minima", {}).items()), flush=True)
    with open(os.path.join(HERE, "comparison.json"), "w") as handle:
        json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()
