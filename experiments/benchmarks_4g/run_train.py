"""Run train.main() with an optional float64 default dtype and timestamped output.

Usage: python run_train.py CONFIG [--float32]

The float64 default is set BEFORE train.py builds anything, so the baseline
(no electrostatics block) is built and trained in the same precision the
electrostatics run is forced into. train.py itself is not modified: its
configure_default_dtype() returns torch.get_default_dtype() when no
electrostatics block is enabled, so the dataset and model follow it.
"""
import builtins
import os
import sys
import time

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)

config = sys.argv[1]
if "--float32" not in sys.argv[2:]:
    torch.set_default_dtype(torch.float64)

import train  # noqa: E402

T0 = time.perf_counter()
_print = builtins.print


def stamped(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(f"[t={time.perf_counter() - T0:10.2f}s]", *args, **kwargs)


train.print = stamped
stamped(f"default dtype {torch.get_default_dtype()}, torch {torch.__version__}")
train.main(["--config", config])
stamped("DONE")
