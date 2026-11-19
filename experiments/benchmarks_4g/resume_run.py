"""Resume a run_train_nanguard.py training from its last checkpoint after an interruption (2026-10-18; reboots).

Usage: python resume_run.py RUN_DIR TAG
- archives log.txt -> log_<TAG>_epoch<E>.txt and model.pt -> model_before_<TAG>.pt (E = the checkpoint's epoch);
- writes config_resume_<TAG>.yaml = config.yaml + resume_from (that copy) + resume_load_optimizer/scheduler: true;
- prints the command to launch it: NANGUARD_SKIP_EPOCHS=E nohup python run_train_nanguard.py ... (the loader then
  replays E epochs, so the batch order is the uninterrupted run's; the attention-dropout masks are not reproduced).
Refuses if the checkpoint has no optimizer state or if model_before_<TAG>.pt already exists."""
import os, shutil, sys
import torch, yaml

run, tag = os.path.abspath(sys.argv[1]), sys.argv[2]
ckpt = os.path.join(run, "model.pt")
c = torch.load(ckpt, map_location="cpu", weights_only=False)
if "optimizer_state_dict" not in c:
    raise SystemExit(f"{ckpt} has no optimizer state")
epoch = int(c["epoch"])
saved = os.path.join(run, f"model_before_{tag}.pt")
if os.path.exists(saved):
    raise SystemExit(f"{saved} exists")
shutil.copy2(ckpt, saved)
if os.path.exists(os.path.join(run, "log.txt")):
    os.replace(os.path.join(run, "log.txt"), os.path.join(run, f"log_{tag}_epoch{epoch}.txt"))
cfg = yaml.safe_load(open(os.path.join(run, "config.yaml")))
cfg.pop("atomic_energies_from_model", None)       # refused with resume_from: the checkpoint keeps its references
cfg.update(resume_from=saved, resume_load_optimizer=True, resume_load_scheduler=True)
out = os.path.join(run, f"config_resume_{tag}.yaml")
with open(out, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=True)
print(f"NANGUARD_SKIP_EPOCHS={epoch} nohup python run_train_nanguard.py {os.path.relpath(out)} > {os.path.relpath(os.path.join(run, 'log.txt'))} 2>&1 &")
