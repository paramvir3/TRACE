"""Weights-only warm start from a training checkpoint: the file a warm-started run names in `resume_from`.

Usage (from experiments/benchmarks_4g):
    python make_warm_start.py PARENT.pt OUT.pt [--reset-log-alpha-scale] [--key-order epoch-first|config-first]
                              [--archive-name NAME] [--force]

Writes {"epoch": 0, "model_state_dict": <PARENT's>, "config": <PARENT's stored config>}: the parent's best weights and
the config stored with them, with no optimiser, scheduler or scaler state.  train.py then builds a fresh optimiser and
learning-rate schedule, starts at epoch 0 (it reads `epoch` as the epoch to resume from) and takes the per-species
reference energies from the stored config.  The released runs used it as follows (configs/<run>.yaml names the output
runs/<run>/warm_start.pt and gives the exact command):

- step 31 (`..._pfp_bopw_e1kw_bondlaw_r5_sXX`): PARENT = step 26's model.pt (`..._pfp_bopw_bondlaw_r5_sXX`);
- step 33 stage A (`AuMgO_single_all_bondlaw_r5_sXX`): PARENT = step 31's model.pt, with --reset-log-alpha-scale.
  Step 33 switches on SPEC 35's frame-polarizability anchor, whose crystal factor g = exp(log_alpha_scale) step 31's
  model does not have.  The flag adds that parameter at its initial value 0.0 (g = 1); without it the step-33 model
  refuses the state dict (missing key).  As in the published file, the state dict is then written as a plain dict
  (without torch's per-module version metadata); no module of this model reads that metadata when loading;
- step 33 stage B (`AuMgO_single_all_e1kw_bondlaw_r5_sXX`) and step 34b stage B
  (`AuMgO_twosite_bf055_e1kw_bondlaw_r5_sXX`): PARENT = the same seed's stage-A model.pt.  This is the recipe the
  training queue ran inline on the GPU host: {"epoch": 0, "model_state_dict": c["model_state_dict"], "config":
  c["config"]} of the loaded stage-A checkpoint c.

Byte-for-byte reproduction.  torch.save (torch 2.2.0) is deterministic, but its bytes depend on the order of the
top-level keys and on the archive name inside the zip file, which torch takes from the file name.  --key-order and
--archive-name set both, so the warm starts of the published runs can be rebuilt bit for bit under any output name.
Checked on 2026-11-18 (Python 3.8.16, torch 2.2.0) against the four published warm starts on disk: step 31's (key order
config-first, archive names warm_from_step26_s42 / _s43) and step 33 stage A's (--reset-log-alpha-scale, archive names
warm_from_step31_s42 / _s43) come out with the published sha256 (the PUBLISHED table below; the script reports the
match).  The stage-B warm starts were written on the GPU host (torch 2.11.0+cu128) and are not on the machine of this
check; for both seeds of steps 33 and 34b the file this script writes holds the same content as the queue's inline
recipe applied to the released stage-A model (every tensor equal in value, dtype and shape, same state-dict order,
same stored config, epoch 0).  Another torch version may write other bytes for the same content: compare contents
then, not checksums.

OUT is never overwritten unless --force is given.
"""
import argparse
import hashlib
import os
import tempfile

import torch

LOG_ALPHA_SCALE = "electrostatics.log_alpha_scale"

#: sha256 of a released parent checkpoint -> (the warm start the published run trained from: its sha256, the options
#: that rebuild it byte for byte with torch 2.2.0)
PUBLISHED = {
    # step 26 model.pt -> step 31's warm_from_step26_sXX.pt
    "daf31bfe85250278812a04c889d80c42cbd893d38563ae8cb8738b42514725e4": (
        "11bbcc91140072748fe2cd22b2504661a305c7c0a2c1123e1ad884454f5898c2",
        "--key-order config-first --archive-name warm_from_step26_s42"),
    "005f01704613c5087136ae7fdabf1c461b726b2e48eea0fe81eca1074884f609": (
        "76813360c4a1a7606581fd75277767b4c9c67ede3030c0a067ad5650e98ce1a7",
        "--key-order config-first --archive-name warm_from_step26_s43"),
    # step 31 model.pt -> step 33 stage A's warm_from_step31_sXX.pt
    "653c648f86019443fbdf385d440716829750b33341da5565a93165603a580d1f": (
        "253e63dff93baa992a0dec6cd05529f3d07dd99b216fdd92aa2de2cb017b255c",
        "--reset-log-alpha-scale --archive-name warm_from_step31_s42"),
    "3699c1f94b922247da24699a7a92b6e4b917d307d31ef48dc2d829f9b73e799e": (
        "0f2976ad8183c166750c1dc0786f91e937b139225d80ab1aad766adc6e348d0b",
        "--reset-log-alpha-scale --archive-name warm_from_step31_s43"),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_warm_start(parent, reset_log_alpha_scale=False, key_order="epoch-first"):
    """The warm-start dict built from the checkpoint at ``parent``, the parent's epoch, and a note on the change."""
    checkpoint = torch.load(parent, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "config"):
        if key not in checkpoint:
            raise KeyError(f"{parent} has no {key!r}: not a training checkpoint")
    state = checkpoint["model_state_dict"]
    note = None
    if reset_log_alpha_scale:
        previous = state.get(LOG_ALPHA_SCALE)
        state = dict(state)
        state[LOG_ALPHA_SCALE] = torch.zeros((), dtype=torch.float64)
        note = (f"{LOG_ALPHA_SCALE} = 0.0 " +
                ("added (absent from the parent)" if previous is None else f"set (parent: {float(previous):.6g})"))
    if key_order == "config-first":
        warm = {"config": checkpoint["config"], "epoch": 0, "model_state_dict": state}
    else:
        warm = {"epoch": 0, "model_state_dict": state, "config": checkpoint["config"]}
    return warm, checkpoint.get("epoch"), note


def save(obj, out, archive_name=None):
    """torch.save to ``out``; with ``archive_name`` the zip archive inside carries that name (torch names it after the
    file), so the bytes equal those of a file saved under ``archive_name``.pt."""
    folder = os.path.dirname(os.path.abspath(out))
    os.makedirs(folder, exist_ok=True)
    if archive_name is None:
        torch.save(obj, out)
        return
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        staged = os.path.join(tmp, archive_name + ".pt")
        torch.save(obj, staged)
        os.replace(staged, out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("parent", help="training checkpoint (runs/<parent run>/model.pt)")
    parser.add_argument("out", help="warm-start file to write (runs/<run>/warm_start.pt)")
    parser.add_argument("--reset-log-alpha-scale", action="store_true",
                        help=f"add or set {LOG_ALPHA_SCALE} = 0.0 (step 33's stage A from step 31)")
    parser.add_argument("--key-order", choices=("epoch-first", "config-first"), default="epoch-first",
                        help="order of the top-level keys (bytes only; config-first rebuilds step 31's published file)")
    parser.add_argument("--archive-name", default=None,
                        help="name of the zip archive inside the file (bytes only; default: OUT's file name)")
    parser.add_argument("--force", action="store_true", help="overwrite OUT if it exists")
    args = parser.parse_args(argv)
    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"{args.out} exists; not overwritten (pass --force to replace it)")
    warm, epoch, note = make_warm_start(args.parent, args.reset_log_alpha_scale, args.key_order)
    save(warm, args.out, args.archive_name)
    digest = sha256(args.out)
    print(f"warm start {args.out}: weights of {args.parent} (its epoch {epoch}), {len(warm['model_state_dict'])} "
          f"tensors, stored config kept, epoch 0, no optimiser state" + (f"; {note}" if note else "")
          + f"; sha256 {digest}")
    known = PUBLISHED.get(sha256(args.parent))
    if known is not None:
        want, options = known
        if digest == want:
            print("identical, byte for byte, to the warm start the published run trained from")
        else:
            print(f"not byte-identical to the published warm start (sha256 {want}); it is rebuilt with {options} "
                  f"under torch 2.2.0 (this is torch {torch.__version__}); compare contents otherwise")


if __name__ == "__main__":
    main()
