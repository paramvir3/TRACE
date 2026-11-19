"""Cost of one batched TRAINING step of a trained model, and where it goes.

Usage: python profile_batch_step.py MODEL.pt DATA.xyz DEVICE [BATCH] [REPEATS] [MAX_ATOMS]

MAX_ATOMS sets flashace.model.BATCHED_ELECTROSTATICS_MAX_ATOMS, the size of
one block-diagonal electrostatic solve (SPEC 24).

Builds the batch exactly as train.py does (AtomisticDataset with the
electrostatics labels, collate_batched), loads the model exactly as the ASE
calculator does, and times REPEATS steps of: forward_batched(training=True)
-> energy + force loss -> backward.  Reports per step and per structure:

* wall time, synchronised on CUDA;
* on CUDA, the number of host-device synchronisations in one step
  (torch.cuda.set_sync_debug_mode('warn') turns each into a warning: every
  .item(), bool(tensor), nonzero() ... stalls the GPU pipeline until the
  queue drains, and the electrostatics runs once per structure);
* the top operators by self CPU time (torch.profiler), which on a launch-bound
  GPU is where the time is.
"""
import os
import sys
import time
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

import train  # noqa: E402
from ase.io import read  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402


def main():
    model_path, data_path, device = sys.argv[1], sys.argv[2], sys.argv[3]
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    repeats = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    if len(sys.argv) > 6:
        import flashace.model

        flashace.model.BATCHED_ELECTROSTATICS_MAX_ATOMS = int(sys.argv[6])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc = TransformersACECalculator(model_path=model_path, device=device)
    model = calc.model.train()
    if getattr(model, "electrostatics", None) is not None:
        # New runs train with the solver-seed loss off (the train.py default under
        # the converged solve); older checkpoints may carry it on.
        import dataclasses

        model.electrostatics.config = dataclasses.replace(
            model.electrostatics.config, solver_seed_loss_weight=0.0)
    frames = read(data_path, f":{batch}")
    has_elec = getattr(model, "electrostatics", None) is not None
    dataset = train.AtomisticDataset(frames, float(model.r_max), electrostatics=has_elec,
                                     dtype=torch.float64)
    items = [dataset[i] for i in range(len(dataset))]
    bat = train.AtomisticDataset.collate_batched(items)
    bat = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in bat.items()}
    n_at = bat["n_atoms"].to(device).to(torch.float64)

    def step():
        model.zero_grad(set_to_none=True)
        E, F, _, _ = model.forward_batched(bat, training=True, compute_stress=False)
        loss = (((E - E.detach() - 1.0) / n_at) ** 2).sum() + ((F - bat["t_F"]) ** 2).mean()
        loss.backward()

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    for _ in range(2):                       # warm-up: allocator, kernels, caches
        step()
    sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        step()
    sync()
    per_step = (time.perf_counter() - t0) / repeats
    print(f"{os.path.basename(data_path)}  device {device}  batch {len(items)}  electrostatics {has_elec}")
    print(f"  step {per_step * 1e3:.1f} ms  ->  {per_step / len(items) * 1e3:.2f} ms per structure")

    if device.startswith("cuda"):
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            step()
            sync()
        torch.cuda.set_sync_debug_mode(0)
        syncs = [w for w in caught if "synchroniz" in str(w.message).lower()]
        sites = {}
        for w in syncs:
            key = f"{os.path.relpath(w.filename, ROOT)}:{w.lineno}"
            sites[key] = sites.get(key, 0) + 1
        print(f"  host-device synchronisations in one step: {len(syncs)} "
              f"({len(syncs) / len(items):.1f} per structure)")
        for key, count in sorted(sites.items(), key=lambda kv: -kv[1])[:15]:
            print(f"    {count:5d}  {key}")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as prof:
        step()
        sync()
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=18))


if __name__ == "__main__":
    main()
