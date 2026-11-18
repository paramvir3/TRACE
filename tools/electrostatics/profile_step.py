"""Where does an electrostatic training step spend its time?  (CPU profile)

Runs one training step (forward with forces, create_graph, loss on E and F,
backward) of the throughput benchmark's model on a system of the benchmark,
under torch.profiler, and prints the top operators and the Python-level
breakdown of the electrostatics (assemble / solve / backward).

Usage: python tools/electrostatics/profile_step.py {box192|mol240} [THREADS]
"""
import os
import sys
import time
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter("ignore")

import torch  # noqa: E402

torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
import throughput_bench as tb  # noqa: E402


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "box192"
    atoms = tb.water_cluster(64, box=12.42) if which == "box192" else tb.water_cluster(80)
    model = tb.build(True, torch.float64)
    data = tb.item(atoms, torch.float64)

    import trace_electrostatics.energy as energy_mod
    import trace_electrostatics.implicit as implicit_mod

    timings = {}

    def wrap(module, name):
        original = getattr(module, name)

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            out = original(*args, **kwargs)
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - t0
            return out

        setattr(module, name, timed)

    wrap(energy_mod, "assemble")
    wrap(implicit_mod, "_pcg")
    wrap(implicit_mod, "_explicit_partial")

    tb.step(model, data)                      # warm-up
    timings.clear()
    t0 = time.perf_counter()
    E, F, _, _ = model(data, training=True, compute_stress=False)
    t_forward = time.perf_counter() - t0
    forward_parts = dict(timings)
    timings.clear()
    t0 = time.perf_counter()
    ((E ** 2).sum() + (F ** 2).sum()).backward()
    t_backward = time.perf_counter() - t0
    print(f"{which}: {len(atoms)} atoms")
    print(f"  forward (energy + forces, create_graph) {t_forward:.2f}s: "
          + ", ".join(f"{k} {v:.2f}s" for k, v in forward_parts.items()))
    print(f"  backward (loss -> parameters)          {t_backward:.2f}s: "
          + ", ".join(f"{k} {v:.2f}s" for k, v in timings.items()))

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        tb.step(model, data)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=18))


if __name__ == "__main__":
    main()
