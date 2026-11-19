# Writes runs/protocol/{system}_step9_s42/config.yaml (2026-10-16): step 9's physics on the other 4G/PFN benchmarks.
# Usage (any directory): python make_step9_cluster_configs.py [batch_size grad_accum]
# Reads step 9's released config, configs/AuMgO_formal_kT_eps_cpl_alpha_ex_bondlaw_r5_s42.yaml (its electrostatics
# block and four training keys; no path of it is used), and each system's round-5 like-for-like config
# runs/protocol/{system}_elec_s42/config.yaml, written by make_protocol.py (make_protocol.py elec {system} 42 cpu).
# This chain reproduces the recorded step-9 configs of NaCl, Ag3 and the chain byte for byte apart from the checkout's
# own absolute path (checked 2026-11-16);
# their released, path-relative copies are in configs/protocol/.
"""Each config = the round-5 like-for-like config of that system (runs/protocol/{system}_elec_s42: PFN's split sizes,
300 epochs, effective batch 32, seed 42, the same data) with step 9's electrostatics (Au2/MgO step 9:
formal integer charge configurations, diabatic ensemble with couplings, exclusion by closed shells (trace_redox SPEC
18), the SPEC 23 bond law, the training kT schedule 0.5 -> 0.1 over 20 epochs, model-solved reference energies,
seeded batches).  Step 9's measured response priors map to the same kind of datum for each system:
  Au2/MgO: MgO eps_inf + Au atom alpha   ->   NaCl: NaCl eps_inf (Li 1976)   Ag3: Ag atom alpha (Cheng 2024)
  Carbon chain: no counterpart (no ionic crystal, no coinage-metal atom): no prior.
"""
import os
import sys
import yaml

#: this directory (experiments/benchmarks_4g), resolved at run time; the written configs carry absolute paths under it
ROOT = os.path.dirname(os.path.abspath(__file__))
#: step 9 (Au2/MgO): the released config; the run directory's own copy is the fallback for an existing local run
STEP9 = os.path.join(ROOT, "configs", "AuMgO_formal_kT_eps_cpl_alpha_ex_bondlaw_r5_s42.yaml")
STEP9_RUN = os.path.join(ROOT, "runs", "AuMgO_formal_kT_eps_cpl_alpha_ex_bondlaw_r5_s42", "config.yaml")
PRIORS = {
    "NaCl": {"dielectric_prior": {"file": f"{ROOT}/data_pfn/NaCl_bulk_eps_prior.xyz", "weight": 0.05}},
    "Ag_cluster": {"polarizability_prior": {"file": f"{ROOT}/data_pfn/Ag_atom_alpha_prior.xyz", "weight": 0.3}},
    "Carbon_chain": {},
}

if __name__ == "__main__":
    batch, accum = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (32, 1)
    step9_file = STEP9 if os.path.exists(STEP9) else STEP9_RUN
    if not os.path.exists(step9_file):
        sys.exit(f"step 9's config not found: {STEP9}")
    step9 = yaml.safe_load(open(step9_file))
    for system, priors in PRIORS.items():
        base = os.path.join(ROOT, "runs", "protocol", f"{system}_elec_s42", "config.yaml")
        if not os.path.exists(base):
            sys.exit(f"{base} not found: write it first with make_protocol.py elec {system} 42 cpu")
        cfg = yaml.safe_load(open(base))
        elec = {k: v for k, v in step9["electrostatics"].items() if k not in ("dielectric_prior", "polarizability_prior")}
        elec.update(priors)
        cfg["electrostatics"] = elec
        for key in ("atomic_energies_from_model", "batch_seeded_rng", "redox_kT_epochs", "redox_kT_start"):
            cfg[key] = step9[key]
        cfg["batch_size"], cfg["grad_accum_steps"] = batch, accum
        run = f"{ROOT}/runs/protocol/{system}_step9_s42"
        cfg["checkpoint_dir"], cfg["model_save_path"], cfg["plot_dir"] = f"{run}/checkpoints", f"{run}/model.pt", f"{run}/plots"
        os.makedirs(run, exist_ok=True)
        yaml.safe_dump(cfg, open(f"{run}/config.yaml", "w"), sort_keys=True)
        print(system, "->", run, "epochs", cfg["epochs"], "batch", batch, "x", accum, "priors", list(priors))
