"""Pre-registered speed measurement of data-parallel training (README, "Training speed on the Mac").

Usage: python bench_data_parallel.py OUT_DIR      (on an idle Mac; runs every configuration in turn)

The step-1 config (runs/AuMgO_redoxq_bondlaw_r5_s42/config.yaml) on the first 32 training and the
first 8 validation structures (110 atoms each), 1 epoch = 4 optimiser steps of 8 batches, with
``batch_seeded_rng`` on, under run_train_nanguard.py as the production runs are.  The epoch time is
the stamp of the epoch row minus the stamp of the table header (run_train.py stamps every line), so
start-up (data, neighbour lists, model) is excluded.

V1' (criterion, amended before any timing run; README): d(4x2, 1x2) <= 10 d(1x2, 1x2b), where
    d(X, Y) = max over tensors of max|theta_X - theta_Y| / max|theta_Y| and 1x2b repeats 1x2.  (Bitwise
    V1 cannot hold: Accelerate's LAPACK makes the sequential run itself non-reproducible.)
V2 (criterion): the fastest configuration is adopted if it is >= 1.5x faster than 1 x 4.
Reported: whether 1x2 and 1x2b are bitwise equal; every configuration's d against 1 x T at its own T.
"""
import json
import os
import re
import socket
import subprocess
import sys

import torch
import yaml
from ase.io import read, write

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "runs", "AuMgO_redoxq_bondlaw_r5_s42", "config.yaml")
CONFIGS = [(1, 4, ""), (1, 2, ""), (1, 2, "b"), (2, 4, ""), (3, 2, ""), (4, 2, ""), (5, 2, ""), (8, 1, "")]
STAMP = re.compile(r"^\[t=\s*([0-9.]+)s\]")


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _epoch_time(log):
    header = row = None
    for line in log.splitlines():
        stamp = STAMP.match(line)
        if not stamp:
            continue
        if header is None and re.search(r"\]\s+Epoch \|", line):
            header = float(stamp.group(1))
        elif header is not None and re.search(r"\]\s+1 \|", line):
            row = float(stamp.group(1))
            break
    return None if header is None or row is None else row - header


def _states(a, b):
    return (torch.load(a, map_location="cpu")["model_state_dict"], torch.load(b, map_location="cpu")["model_state_dict"])


def _same_checkpoint(a, b):
    sa, sb = _states(a, b)
    return sa.keys() == sb.keys() and all(sa[k].dtype == sb[k].dtype and torch.equal(sa[k], sb[k]) for k in sa)


def _distance(a, b):
    """max over floating tensors of max|theta_a - theta_b| / max|theta_b|."""
    sa, sb = _states(a, b)
    return max(float((sa[k] - sb[k]).abs().max()) / max(float(sb[k].abs().max()), 1e-300)
               for k in sb if sb[k].is_floating_point() and sb[k].numel())


def main(out):
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    base = yaml.safe_load(open(BASE))
    train_xyz, valid_xyz = os.path.join(out, "train32.xyz"), os.path.join(out, "valid8.xyz")
    write(train_xyz, read(base["train_file"], ":32"))
    write(valid_xyz, read(base["valid_file"], ":8"))
    results = {}
    for world, threads, tag in CONFIGS:
        name = f"{world}x{threads}{tag}"
        run = os.path.join(out, name)
        os.makedirs(run, exist_ok=True)
        config = dict(base, train_file=train_xyz, valid_file=valid_xyz, epochs=1, temperature_scale_epochs=0,
                      lr_scheduler_t_max=1, torch_num_threads=threads, batch_seeded_rng=True,
                      model_save_path=os.path.join(run, "model.pt"), checkpoint_dir=os.path.join(run, "checkpoints"),
                      plot_dir=os.path.join(run, "plots"), plot_training_curves=False, save_last_checkpoint=False)
        path = os.path.join(run, "config.yaml")
        with open(path, "w") as handle:
            yaml.safe_dump(config, handle)
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), GLOO_SOCKET_IFNAME="lo0")
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        launcher = [] if world == 1 else ["-m", "torch.distributed.run", "--nnodes=1", f"--nproc_per_node={world}",
                                          "--master_addr=127.0.0.1", f"--master_port={_port()}"]
        done = subprocess.run([sys.executable, *launcher, os.path.join(HERE, "run_train_nanguard.py"), path],
                              capture_output=True, text=True, env=env, cwd=HERE)
        with open(os.path.join(run, "log.txt"), "w") as handle:
            handle.write(done.stdout + "\n--- stderr ---\n" + done.stderr)
        seconds = _epoch_time(done.stdout)
        results[name] = {"processes": world, "threads": threads, "exit": done.returncode, "epoch_s": seconds}
        print(f"{name}: exit {done.returncode}, epoch {seconds} s", flush=True)

    reference = results.get("1x4", {}).get("epoch_s")
    for name, r in results.items():
        r["speedup_vs_1x4"] = None if not (r["epoch_s"] and reference) else reference / r["epoch_s"]
        same = os.path.join(out, f"1x{r['threads']}", "model.pt")
        mine = os.path.join(out, name, "model.pt")
        if name != f"1x{r['threads']}" and os.path.exists(same) and os.path.exists(mine):
            r["bitwise_equal_to_1x_same_threads"] = _same_checkpoint(mine, same)
            r["d_to_1x_same_threads"] = _distance(mine, same)
    d_par = results.get("4x2", {}).get("d_to_1x_same_threads")
    d_seq = results.get("1x2b", {}).get("d_to_1x_same_threads")
    v1 = None if d_par is None or d_seq is None else d_par <= 10.0 * d_seq
    timed = {n: r for n, r in results.items() if r["epoch_s"]}
    fastest = min(timed, key=lambda n: timed[n]["epoch_s"]) if timed else None
    summary = {"results": results, "V1prime_d(4x2,1x2)<=10*d(1x2,1x2b)": v1, "d_par": d_par, "d_seq": d_seq,
               "sequential_repeat_bitwise": results.get("1x2b", {}).get("bitwise_equal_to_1x_same_threads"),
               "fastest": fastest,
               "V2_fastest_at_least_1.5x": bool(fastest and (timed[fastest]["speedup_vs_1x4"] or 0.0) >= 1.5)}
    with open(os.path.join(out, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
