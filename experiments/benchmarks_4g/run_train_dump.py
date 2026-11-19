"""run_train.py with a failure dump and timing for the TRACE-Redox electrostatics.

Usage: python run_train_dump.py CONFIG

Wraps RedoxElectrostatics.forward (the package is not modified):
* every 50 calls, prints the call count and the mean wall time per call;
* on an exception, saves the module's state_dict, its construction settings
  and its exact inputs to DIR(CONFIG)/redox_failure.pt, so the failing
  evaluation can be rebuilt and replayed on another machine, and re-raises.
"""
import dataclasses
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
torch.set_default_dtype(torch.float64)

from trace_redox.module import RedoxElectrostatics  # noqa: E402

# under torchrun every rank dumps to a file of its own
_RANK = "" if int(os.environ.get("WORLD_SIZE", "1")) <= 1 else f"_rank{int(os.environ['RANK'])}"
DUMP = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), f"redox_failure{_RANK}.pt")
_original = RedoxElectrostatics.forward
_count = [0, 0.0]


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(v) for v in value)
    return value


def forward(self, *args, **kwargs):
    start = time.perf_counter()
    try:
        out = _original(self, *args, **kwargs)
    except Exception as error:
        # a state_dict, not the pickled module: e3nn's pickles do not cross e3nn versions
        torch.save({"state": {k: v.detach().cpu() for k, v in self.state_dict().items()},
                    "irreps": str(self.heads.irreps_in), "r_max": float(getattr(self.heads, "r_max", float("nan"))),
                    "redox": dataclasses.asdict(self.redox), "config": os.path.abspath(sys.argv[1]),
                    "args": _cpu(args), "kwargs": _cpu(kwargs), "training": self.training,
                    "error": repr(error), "call": _count[0] + 1}, DUMP)
        print(f"[redox dump] call {_count[0] + 1}: {error!r} -> {DUMP}", flush=True)
        raise
    _count[0] += 1
    dt = time.perf_counter() - start
    _count[1] += dt
    if dt > 5.0:                                        # name the slow forwards (2026-10-30: seed 43 at 31 s per call)
        try:
            data = kwargs.get("data") or (args[5] if len(args) > 5 else {})
            z = data.get("z") if isinstance(data, dict) else None
            n_al = int((z == 13).sum()) if z is not None else -1
            rec = getattr(self, "_downfold_last", None) or {}
            last = getattr(self, "_ensemble_last", None) or {}
            C = int(last["configs"].shape[0]) if "configs" in last else -1
            print(f"[redox call {_count[0]}] {dt:.1f} s; Al {n_al}; configs {C}; leaves {rec.get('leaves')}; "
                  f"clusters {rec.get('clusters')}; ground {rec.get('ground')}", flush=True)
        except Exception as exc:                        # the log line must never break the training
            print(f"[redox call {_count[0]}] {dt:.1f} s (no record: {exc!r})", flush=True)
    if _count[0] % 50 == 0:
        print(f"[redox timing] {_count[0]} forward calls, {_count[1] / _count[0]:.2f} s per call (forward only)",
              flush=True)
    return out


RedoxElectrostatics.forward = forward
sys.argv = [os.path.join(HERE, "run_train.py")] + sys.argv[1:]
with open(sys.argv[0]) as handle:
    code = compile(handle.read(), sys.argv[0], "exec")
exec(code, {"__name__": "__main__", "__file__": sys.argv[0]})
