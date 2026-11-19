"""Write the training configs.

Trunk, optimiser, loss and schedule follow training_examples/water_ccsd/config.yaml
(architecture v2, r_max 6, hidden 64, Muon, lr 1e-3, batch 8, Sobolev 1e-3,
attention temperature curriculum) with these changes, applied to BOTH arms:
  * data: the open-boundary gas-phase split in ./data (explicit valid_file);
  * stress weights 0 (no stress labels; train.py refuses a nonzero weight);
  * temperature_force_ref/exponent removed (train.py refuses them now);
  * epochs / cosine T_max set to the CPU budget; early stopping off so both
    arms run the same number of epochs; best-validation checkpoint tracked
    from epoch 1.
Usage: python make_configs.py NAME EPOCHS R_MAX THREADS [elec] [extra key=value ...]

Extras: ``key=value`` sets a top-level key; ``elec.config.X=`` / ``elec.heads.X=``
an ElectrostaticConfig / ElectrostaticHeads option; ``elec.X=`` any other key of
the electrostatics block (e.g. ``elec.dipole_weight=100``); ``data_dir=data_qm``
trains on the split with PBE0 dipoles and polarizabilities (qm_reference.py).
"""
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
name, epochs, r_max, threads = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
rest = sys.argv[5:]
elec = "elec" in rest
extras = [x for x in rest if "=" in x]
data_dir = "data"
for item in list(extras):
    if item.startswith("data_dir="):
        data_dir = item.split("=", 1)[1]
        extras.remove(item)
run_dir = os.path.join(HERE, "runs", name)
os.makedirs(run_dir, exist_ok=True)

config = {
    "train_file": os.path.join(HERE, data_dir, "train.xyz"),
    "valid_file": os.path.join(HERE, data_dir, "valid.xyz"),
    "seed": 42,
    "model_save_path": os.path.join(run_dir, "model.pt"),
    "save_last_checkpoint": True,
    "plot_training_curves": True,
    "plot_dir": os.path.join(run_dir, "plots"),
    # model (config.yaml)
    "r_max": r_max,
    "l_max": 2,
    "num_radial": 12,
    "hidden_dim": 64,
    "num_layers": 1,
    "architecture_version": 2,
    "correlation_order": 4,
    "correlation_channels": 16,
    "attention_num_heads": 2,
    "attention_key_dim": None,
    "attention_ffn_hidden": None,
    "attention_dropout": 0.03,
    "attention_layer_scale_init": 0.01,
    "attention_distance_penalty": True,
    "radial_basis_type": "bessel",
    "radial_trainable": False,
    "gaussian_width": 0.5,
    "radial_mlp_hidden": 32,
    "radial_mlp_layers": 2,
    # training (config.yaml)
    "epochs": epochs,
    "batch_size": 8,
    "learning_rate": 0.001,
    "optimizer": "muon",
    "weight_decay": 1.0e-5,
    "amsgrad": True,
    "muon_momentum": 0.95,
    "muon_ns_steps": 5,
    "muon_ns_dtype": "auto",
    "muon_nesterov": True,
    "muon_parameter_mode": "hidden",
    "muon_include": [],
    "muon_exclude": [],
    "muon_aux_betas": [0.9, 0.95],
    "muon_aux_eps": 1.0e-10,
    "lr_scheduler_t_max": epochs,
    "lr_warmup_epochs": 0,
    "lr_warmup_start_factor": 0.1,
    "lr_scheduler_interval": "epoch",
    "lr_scheduler_use_restarts": False,
    "lr_restart_mult": 1.0,
    "lr_min": 5.0e-5,
    "clip_grad_norm": 1.0,
    "use_amp": False,
    "amp_dtype": "bfloat16",
    "grad_accum_steps": 1,
    "precompute_neighbors": True,
    "random_rotation": False,
    "device": "cpu",
    "torch_num_threads": threads,
    "torch_num_interop_threads": 1,
    "num_workers": 0,
    "checkpoint_dir": os.path.join(run_dir, "checkpoints"),
    "checkpoint_interval": 0,
    "early_stopping_patience": 0,
    "early_stopping_min_epoch": 1,
    "early_stopping_min_delta": 0.0,
    "resume_from": None,
    "energy_shift_per_atom": None,
    "atomic_energies": None,
    "solve_atomic_energies": False,
    # loss (config.yaml, minus stress)
    "energy_weight": 1.0,
    "forces_weight": 1.0,
    "forces_weight_final": 1.0,
    "forces_weight_decay_epochs": 0,
    "stress_weight": 0.0,
    "stress_weight_final": 0.0,
    "stress_weight_ramp_epochs": 0,
    "aux_force_weight": 0.0,
    "aux_stress_weight": 0.0,
    "force_consistency_weight": 0.0,
    "sobolev_weight": 0.001,
    "sobolev_sigma": 0.02,
    "displacement_prob": 0.0,
    "displacement_sigma": 0.0,
    "temperature_scale_start": 1.3,
    "temperature_scale_end": 1.0,
    "temperature_scale_epochs": 15 if epochs > 15 else max(0, epochs - 1),
}
if elec:
    # docs/electrostatics.md section 6; package defaults: converged solve,
    # Gaussian kernel, dense solver; heads r_max defaults to the model r_max.
    config["electrostatics"] = {"enabled": True, "config": {}, "heads": {}}
for item in extras:
    key, value = item.split("=", 1)
    value = yaml.safe_load(value)
    if key.startswith("elec.config."):
        config["electrostatics"]["config"][key[len("elec.config."):]] = value
    elif key.startswith("elec.heads."):
        config["electrostatics"]["heads"][key[len("elec.heads."):]] = value
    elif key.startswith("elec."):
        config["electrostatics"][key[len("elec."):]] = value
    else:
        config[key] = value
path = os.path.join(run_dir, "config.yaml")
with open(path, "w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
print(path)
