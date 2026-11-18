"""S1 (README, pre-registered 2026-11-11 16:45 IST): where the time of one energy-and-forces evaluation goes.

Frozen weights, the undoped wetting reference cell, CPU.  Each timed evaluation is at a new geometry (all atoms
displaced by a seeded 0.01 A Gaussian) after one untimed warm-up.  The electrostatics module is also timed alone, on
the inputs a forward pre-hook captured (detached leaves with requires_grad), forward and backward separately.
Nothing in the package or a run directory changes.

    PYTHONPATH=.:experiments/benchmarks_4g python experiments/benchmarks_4g/probe_speed_profile.py MODEL.pt \
        setting=stored|extensive [mesh=2,2,1] [threads=4] [out=PREFIX]
"""
import cProfile
import dataclasses
import io
import pstats
import resource
import statistics
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np
import torch

opt = dict(a.split("=", 1) for a in sys.argv[2:])
torch.set_num_threads(int(opt.get("threads", 4)))             # before any parallel region: a later call did not
_set_threads, torch.set_num_threads = torch.set_num_threads, lambda n: None   # take effect (measured), and
from flashace.calculator import TransformersACECalculator      # eval_aumgo sets 4 threads on import
from eval_aumgo import references
torch.set_num_threads = _set_threads
setting, threads = opt.get("setting", "stored"), int(opt.get("threads", 4))
mesh = tuple(int(x) for x in opt.get("mesh", "1,1,1").split(","))
prefix = opt.get("out")
torch.manual_seed(0)
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu")
assert torch.get_num_threads() == threads, torch.get_num_threads()
mod = calc.model.electrostatics
if setting == "extensive":
    mod.redox = dataclasses.replace(mod.redox, state_polarizability_stiffness="onsite", diabatic_space="clusters",
                                    band_form="fermi", bvk_mesh=mesh)
else:
    assert setting == "stored" and mesh == (1, 1, 1), (setting, mesh)
r = mod.redox
print(f"model {sys.argv[1].split('/')[-2]}; setting {setting}: stiffness {r.state_polarizability_stiffness!r}, "
      f"space {r.diabatic_space!r}, band {r.band_form!r}, bvk_mesh {r.bvk_mesh}; threads {torch.get_num_threads()}; "
      f"dtype {torch.get_default_dtype()}")
CAP = {"on": True, "n": 0}


def pre(module, args, kwargs):
    if CAP["on"]:
        CAP["n"] += 1; CAP["args"], CAP["kwargs"] = args, kwargs


mod.register_forward_pre_hook(pre, with_kwargs=True)
base = references()[("undoped", "wetting")].copy(); base.set_constraint()
rng = np.random.default_rng(20261111)


def fresh():
    a = base.copy(); a.positions = a.positions + rng.normal(0.0, 0.01, a.positions.shape)
    a.info = {"charge": 0.0}; a.calc = calc; calc.reset(); mod._ensemble_last = None
    return a


def full():
    a = fresh(); CAP["n"] = 0; t0 = time.perf_counter()
    a.get_potential_energy(); a.get_forces()
    return time.perf_counter() - t0, CAP["n"]


def leaves():
    det = lambda x: x.detach().clone().requires_grad_(x.is_floating_point()) if torch.is_tensor(x) else x
    args = [det(x) for x in CAP["args"]]
    kw = {k: (dict(v) if isinstance(v, dict) else (v.detach().clone() if torch.is_tensor(v) else v))
          for k, v in CAP["kwargs"].items()}
    return args, kw, [x for x in args if torch.is_tensor(x) and x.requires_grad]


def module_alone(args, kw, grads):
    CAP["on"] = False; mod._ensemble_last = None; t0 = time.perf_counter()
    out = mod(*args, **kw); t1 = time.perf_counter()
    torch.autograd.grad(out[0], grads, allow_unused=True); t2 = time.perf_counter()
    CAP["on"] = True
    return t1 - t0, t2 - t1


full()                                                            # warm-up, untimed
rows = []
for _ in range(5):
    t_full, calls = full()
    args, kw, grads = leaves()
    t_fw, t_bw = module_alone(args, kw, grads)
    rows.append((t_full, t_fw, t_bw, calls))
    print(f"   whole {t_full:7.3f} s; electrostatics forward {t_fw:7.3f} s, backward {t_bw:7.3f} s; "
          f"module calls per evaluation {calls}", flush=True)
med = [statistics.median(x[i] for x in rows) for i in range(3)]
rng_ = [(min(x[i] for x in rows), max(x[i] for x in rows)) for i in range(3)]
print(f"== median: whole {med[0]:.3f} s [{rng_[0][0]:.3f}, {rng_[0][1]:.3f}]; electrostatics forward {med[1]:.3f} s "
      f"[{rng_[1][0]:.3f}, {rng_[1][1]:.3f}], backward {med[2]:.3f} s [{rng_[2][0]:.3f}, {rng_[2][1]:.3f}]; "
      f"local model (whole - electrostatics, a difference) {med[0] - med[1] - med[2]:.3f} s; "
      f"electrostatics share {(med[1] + med[2]) / med[0]:.3f}")

args, kw, grads = leaves()                                        # function-level profile of the module's forward
CAP["on"] = False; mod._ensemble_last = None
prof = cProfile.Profile(); prof.enable(); out = mod(*args, **kw); prof.disable()
s = io.StringIO(); pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(r"trace_redox|trace_electrostatics", 40)
text = s.getvalue()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as tp:
    torch.autograd.grad(out[0], grads, allow_unused=True)
table = tp.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
CAP["on"] = True
if prefix:
    open(prefix + "_cprofile_forward.txt", "w").write(text)
    open(prefix + "_torch_backward.txt", "w").write(table)
print("== cProfile of the forward, by cumulative time (top lines):")
print("\n".join(l[:200] for l in text.splitlines()[:60] if l.strip()))
print("== torch profiler of the backward, by self CPU time:")
print("\n".join(l[:200] for l in table.splitlines()[:32]))

cfg = mod.config                                                  # the size of the dense operator, read off its guard
mod.config = dataclasses.replace(cfg, max_operator_bytes=1.0); mod._ensemble_last = None
try:
    mod(*leaves()[:2][0], **leaves()[1]); print("== operator guard not reached")
except ValueError as err:
    print("== operator guard: " + str(err)[:190])
mod.config = cfg
print(f"== peak resident memory of the process {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB")
print("done " + time.strftime("%Y-%m-%d %H:%M IST"))
