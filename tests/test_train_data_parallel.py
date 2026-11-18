"""Data-parallel training across processes (torchrun, gloo, CPU) reproduces the sequential run BITWISE.

The speed lever on CPU is across structures, not within one: a single redox structure is
overhead-bound, so more threads barely help (2.25 s at 4 threads, 2.64 s at 6, measured
2026-10-09), while independent processes scale.  An optimiser step accumulates the gradients of
``grad_accum_steps`` batches that are independent until they are summed, so the batches of each
window are dealt out across the ranks.

Exactness, not round-off equivalence:

* every batch draws its random numbers (attention dropout, random rotation) from a generator seeded
  by (seed, epoch, batch index) -- ``batch_seeded_rng``, forced on in parallel -- so a batch sees the
  same masks whichever process evaluates it;
* each rank keeps its batches' gradients apart, and every rank sums ALL the window's batch gradients
  in batch order, the order in which a sequential run's ``.grad`` accumulates them.  IEEE addition is
  exactly rounded and commutative, so the sum is the sequential one bit for bit, on every rank;
* metric and loss updates are recorded per batch and replayed in batch order.

So with the same threads per process (here 1; BLAS results can depend on the thread count) the
parallel run equals the sequential run with ``batch_seeded_rng: true`` exactly: the same batches
evaluated, the same gradients at every step, the same parameters after it, the same history.
"""
import math
import os
import signal
import socket
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch
import yaml
from ase.io import write

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "redox"))
from test_train_redox import (  # noqa: E402
    HEADS, MODEL_KWARGS, hydroxide, rocksalt, sodium_cation, sodium_chloride, water,
)

import train  # noqa: E402

WORKER = HERE / "data_parallel_worker.py"


def _write_config(directory, **overrides):
    """Seven training structures (windows of 4 and 3 batches) and three validation structures."""
    directory.mkdir(parents=True, exist_ok=True)
    train_frames = [water(0), hydroxide(1), sodium_chloride(2), sodium_cation(3), rocksalt(4),
                    water(7, shift=0.02), sodium_chloride(8, separation=2.7)]
    valid_frames = [water(5, shift=0.03), sodium_chloride(6, separation=3.0), rocksalt(9, displacement=0.15)]
    write(directory / "train.extxyz", train_frames)
    write(directory / "valid.extxyz", valid_frames)
    config = {
        "device": "cpu", "seed": 3,
        "train_file": str(directory / "train.extxyz"), "valid_file": str(directory / "valid.extxyz"),
        "architecture_version": 2, **MODEL_KWARGS,
        "attention_dropout": 0.1, "random_rotation": True,       # both draw random numbers
        "batch_size": 1, "grad_accum_steps": 4, "epochs": 2,
        "optimizer": "muon", "learning_rate": 1e-3, "clip_grad_norm": 1.0,
        "energy_weight": 1.0, "forces_weight": 10.0, "num_workers": 0,
        "batched_graphs": True, "batch_seeded_rng": True,
        "torch_num_threads": 1, "torch_num_interop_threads": 1,
        "model_save_path": str(directory / "model.pt"),
        "save_last_checkpoint": False, "plot_training_curves": False,
        "electrostatics": {
            "enabled": True, "heads": dict(HEADS),
            "dipole_weight": 1.0, "partial_charge_weight": 0.1,
            "redox": {"kT": 0.1},
        },
    }
    config.update(overrides)
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(config, prefix, world, perturb_rank=None):
    """One process, or ``world`` under torchrun with a static rendezvous on the loopback interface.

    (``--standalone`` hangs on macOS: its rendezvous resolves an IPv6 reverse name that does not
    exist, measured 2026-10-09.)  The launch gets its own process group, killed whole on a timeout.
    """
    environment = {k: v for k, v in os.environ.items()
                   if k not in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")}
    environment["OMP_NUM_THREADS"] = "1"
    environment["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    environment.pop("TRACE_TEST_PERTURB_RANK", None)
    if perturb_rank is not None:
        environment["TRACE_TEST_PERTURB_RANK"] = str(perturb_rank)
    launcher = [] if world == 1 else ["-m", "torch.distributed.run", "--nnodes=1", f"--nproc_per_node={world}",
                                      "--master_addr=127.0.0.1", f"--master_port={_free_port()}"]
    process = subprocess.Popen([sys.executable, *launcher, str(WORKER), str(config), str(prefix)],
                               env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail("the training run hung:\n" + stdout[-3000:] + stderr[-3000:])
    assert process.returncode == 0, stdout[-4000:] + stderr[-4000:]
    return [torch.load(f"{prefix}_rank{rank}.pt") for rank in range(world)]


def _equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and torch.equal(a, b)


def _same_history(a, b):
    """Exactly equal, entry by entry; an observable with no labels is nan in both."""
    def same(x, y):
        return x == y or (math.isnan(x) and math.isnan(y))
    return a.keys() == b.keys() and all(
        len(a[k]) == len(b[k]) and all(same(x, y) for x, y in zip(a[k], b[k])) for k in a)


@pytest.fixture(scope="module")
def sequential(tmp_path_factory):
    directory = tmp_path_factory.mktemp("sequential")
    (record,) = _run(_write_config(directory), directory / "out", 1)
    return directory, record


@pytest.mark.parametrize("world", [2, 3])
def test_parallel_run_equals_the_sequential_run_bitwise(sequential, tmp_path, world):
    seq_dir, seq = sequential
    ranks = _run(_write_config(tmp_path), tmp_path / "out", world)

    # the work was split: each batch evaluated by exactly one rank, every batch evaluated
    evaluated = sorted(batch for record in ranks for batch in record["batches"])
    assert evaluated == sorted(seq["batches"])
    assert all(len(record["batches"]) < len(seq["batches"]) for record in ranks)

    assert len(seq["grads"]) == 4                          # 2 epochs x windows of 4 and 3 batches
    for record in ranks:
        assert len(record["grads"]) == len(seq["grads"])
        for step, (g_par, g_seq) in enumerate(zip(record["grads"], seq["grads"])):
            assert all(_equal(a, b) for a, b in zip(g_par, g_seq)), f"gradients differ at step {step}"
        for step, (p_par, p_seq) in enumerate(zip(record["params"], seq["params"])):
            assert all(_equal(a, b) for a, b in zip(p_par, p_seq)), f"parameters differ after step {step}"
        assert _same_history(record["history"], seq["history"])

    # one log (rank 0's, the launcher's stdout); every other rank's output in a file of its own
    for rank in range(1, world):
        assert "Training finished" in (tmp_path / f"stdout_rank{rank}.txt").read_text()

    # one checkpoint, written by rank 0, equal to the sequential one
    saved = torch.load(tmp_path / "model.pt", map_location="cpu")["model_state_dict"]
    reference = torch.load(seq_dir / "model.pt", map_location="cpu")["model_state_dict"]
    assert saved.keys() == reference.keys()
    assert all(_equal(saved[k], reference[k]) for k in reference)


def test_batch_seeding_changes_nothing_but_the_random_numbers(tmp_path):
    """Without dropout or rotation no random number is drawn, so the flag is invisible."""
    histories, states = [], []
    for flag in (False, True):
        directory = tmp_path / str(flag)
        path = _write_config(directory, attention_dropout=0.0, random_rotation=False, batch_seeded_rng=flag)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            histories.append(train.main(["--config", str(path)]))
        states.append(torch.load(directory / "model.pt", map_location="cpu")["model_state_dict"])
    assert _same_history(histories[0], histories[1])
    assert all(_equal(states[0][k], states[1][k]) for k in states[0])


def test_seeded_runs_differ_from_each_other_only_through_the_seed(tmp_path):
    """Per-batch seeds are a function of (seed, epoch, batch): distinct batches, distinct seeds."""
    seeds = {train.batch_seed(3, epoch, batch) for epoch in range(50) for batch in range(2000)}
    assert len(seeds) == 50 * 2000
    assert train.batch_seed(3, 1, 7) == train.batch_seed(3, 1, 7) != train.batch_seed(4, 1, 7)


@pytest.mark.parametrize("override,match", [
    ({"batched_graphs": False}, "batched_graphs"),
    ({"device": "cuda"}, "cpu"),
])
def test_unsupported_parallel_settings_are_refused(override, match):
    config = {"device": "cpu", "batched_graphs": True, **override}
    with pytest.raises(ValueError, match=match):
        train.data_parallel_world(config, {"WORLD_SIZE": "2", "RANK": "0"})
    assert train.data_parallel_world(config, {}) == (0, 1)       # a single process is never refused


def test_a_resumed_parallel_run_equals_the_resumed_sequential_run(tmp_path):
    """Resuming from a checkpoint (model, optimiser, scheduler) under torchrun changes nothing."""
    first = tmp_path / "epoch1"
    _run(_write_config(first, epochs=1), first / "out", 1)
    records = {}
    for world in (1, 2):
        directory = tmp_path / f"resume_w{world}"
        config = _write_config(directory, epochs=2, resume_from=str(first / "model.pt"),
                               resume_load_optimizer=True, resume_load_scheduler=True)
        records[world] = _run(config, directory / "out", world)
    (seq,), ranks = records[1], records[2]
    assert len(seq["grads"]) == 2                          # epoch 2 only: windows of 4 and 3 batches
    for record in ranks:
        for g_par, g_seq in zip(record["grads"], seq["grads"]):
            assert all(_equal(a, b) for a, b in zip(g_par, g_seq))
        for p_par, p_seq in zip(record["params"], seq["params"]):
            assert all(_equal(a, b) for a, b in zip(p_par, p_seq))
        assert _same_history(record["history"], seq["history"]) and record["history"]["epoch"] == [2]


def test_rank_zero_state_is_broadcast_after_every_step(sequential, tmp_path):
    """A rank whose optimiser step lands one ulp off is overwritten by rank 0's parameters.

    All ranks step from the same summed gradients, so they agree unless a kernel is not
    deterministic -- and Apple Accelerate's LAPACK is not (``cholesky_solve``, ~1 ulp,
    intermittent).  Rank 0's parameters and buffers are broadcast after every step, so the run is
    rank 0's arithmetic by construction.  Rank 1 is nudged here (parameters and optimiser state).
    """
    seq_dir, seq = sequential
    rank0, rank1 = _run(_write_config(tmp_path), tmp_path / "out", 2, perturb_rank=1)
    assert not all(_equal(a, b) for a, b in zip(rank1["params"][0], seq["params"][0]))   # the nudge happened
    for step, (p0, p_seq) in enumerate(zip(rank0["params"], seq["params"])):
        assert all(_equal(a, b) for a, b in zip(p0, p_seq)), f"rank 0 parameters differ after step {step}"
    for record in (rank0, rank1):
        for step, (g_par, g_seq) in enumerate(zip(record["grads"], seq["grads"])):
            assert all(_equal(a, b) for a, b in zip(g_par, g_seq)), f"gradients differ at step {step}"
        assert _same_history(record["history"], seq["history"])
    saved = torch.load(tmp_path / "model.pt", map_location="cpu")["model_state_dict"]
    reference = torch.load(seq_dir / "model.pt", map_location="cpu")["model_state_dict"]
    assert all(_equal(saved[k], reference[k]) for k in reference)
