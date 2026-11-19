"""Every Python block of docs/electrostatics.md runs, top to bottom, in one session.

The guide says its examples run in order in one Python session; this holds it to
that.  A stale example (an argument renamed, a key moved, a refusal added) fails
here instead of in a user's terminal.  Blocks are executed with the repository
on ``sys.path``, from a temporary directory holding the ``model.pt`` the
guide's inference examples load -- written by the real ``train.main`` on the
synthetic dataset of ``tests/test_train_electrostatics.py`` -- and their output
is discarded.
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import re
import warnings

import pytest

GUIDE = pathlib.Path(__file__).resolve().parents[1] / "docs" / "electrostatics.md"


def _python_blocks(text: str):
    return re.findall(r"^```python\n(.*?)^```", text, flags=re.S | re.M)


@pytest.mark.slow
def test_every_guide_example_runs_in_order(tmp_path, monkeypatch):
    import torch

    import train
    from test_train_electrostatics import _write_config

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train.main(["--config", str(_write_config(tmp_path, batched=False))])
    assert (tmp_path / "model.pt").exists()
    monkeypatch.chdir(tmp_path)
    torch.set_default_dtype(torch.float32)      # the guide sets float64 itself, first
    blocks = _python_blocks(GUIDE.read_text())
    assert len(blocks) >= 10, "the guide's examples were not found"
    namespace: dict = {"__name__": "__guide__"}
    for number, block in enumerate(blocks, start=1):
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            try:
                exec(compile(block, f"{GUIDE.name}[block {number}]", "exec"), namespace)
            except Exception as error:  # noqa: BLE001 - report which block broke
                pytest.fail(f"guide block {number} raised {type(error).__name__}: {error}\n{block}")
