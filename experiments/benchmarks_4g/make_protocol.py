"""Configs of the like-for-like protocol against 4G-HDNNP and PFN (round 5).

Usage: python make_protocol.py ARM DATASET SEED DEVICE [ROOT] [OUT_BASE]

ARM in {local, elec, elecq, redox}; DATASET in {Ag_cluster, Carbon_chain, NaCl, AuMgO}.
Writes runs/protocol/{DATASET}_{ARM}_s{SEED}/config.yaml with paths under ROOT
(default: this directory; pass the GPU host's checkout directory for a run there), into
OUT_BASE (default: this directory) -- a separate local base for configs that
are copied to another machine.

Fixed for every arm, so the arms differ only in the physics:
* data: data_pfn/ -- PFN's sizes (8000 / 1000 for Ag3 and the chain, 4000 /
  500 for NaCl), the rest of the data set held out as the test set;
* the model of the first round (hidden 64, l_max 2, one interaction layer,
  each data set's paper cutoff);
* 300 epochs, cosine schedule to lr_min, batch 32, energy weight 10, forces 1,
  batched graphs, no Sobolev term, the best-validation checkpoint kept.
Arms:
* local: no electrostatics;
* elec:  + trace_electrostatics with the new-run defaults (energies, forces and
  total charges only -- no charge labels), except the SPEC 23 contact-law width
  w = 0.3 A (CONTACT_WIDTH): the 0.1 A default is sharper than the hopping decay
  allows (w >= ln K / (8 beta) = 0.29 A) and put a 5.3 eV/A spurious force on a
  stretched Na-Cl bond; measured on NaCl and on water (README, round 5);
* elecq: + the Hirshfeld partial-charge loss (weight 1), as 4G and PFN train on;
* redox: elec + TRACE-Redox (trace_redox, the SPEC 25 free-charge sector with
  the certified self-interaction removal; ``electrostatics.redox: {}``).
"""
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
EPOCHS = 300
CONTACT_WIDTH = 0.3


def main():
    arm, dataset, seed, device = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    root = sys.argv[5] if len(sys.argv) > 5 else HERE
    out_base = sys.argv[6] if len(sys.argv) > 6 else HERE
    template = yaml.safe_load(open(os.path.join(HERE, "runs", f"{dataset}_{'elec' if arm != 'local' else 'base'}_s42",
                                                "config.yaml")))
    tag = f"{dataset}_{arm}_s{seed}"
    run = os.path.join(root, "runs", "protocol", tag)
    config = dict(template)
    config.update(
        train_file=os.path.join(root, "data_pfn", f"{dataset}_train.xyz"),
        valid_file=os.path.join(root, "data_pfn", f"{dataset}_valid.xyz"),
        seed=seed, epochs=EPOCHS, lr_scheduler_t_max=EPOCHS, batch_size=32,
        energy_weight=10.0, forces_weight=1.0, batched_graphs=True, sobolev_weight=0.0,
        device=device, torch_num_threads=1 if device == "cuda" else 2, num_workers=0,
        plot_training_curves=True, save_last_checkpoint=True,
        model_save_path=os.path.join(run, "model.pt"), plot_dir=os.path.join(run, "plots"),
        checkpoint_dir=os.path.join(run, "checkpoints"),
    )
    if arm == "local":
        config.pop("electrostatics", None)
    else:
        block = {"enabled": True, "config": {}, "heads": {"charge_transfer_width": CONTACT_WIDTH}}
        if arm == "elecq":
            block.update(partial_charge_weight=1.0, partial_charge_key="hirshfeld_charges")
        if arm == "redox":
            block["redox"] = {}
        config["electrostatics"] = block
    out = os.path.join(out_base, "runs", "protocol", tag)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.yaml"), "w") as handle:
        yaml.safe_dump(config, handle)
    print(os.path.join(out, "config.yaml"))


if __name__ == "__main__":
    main()
