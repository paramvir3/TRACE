import re
import tempfile
from pathlib import Path

import e3nn
import numpy as np
from packaging.version import Version
import torch

from flashace.checkpoint import load_checkpoint


def _declared_e3nn_range():
    """The e3nn floor and ceiling ``setup.py`` actually declares.

    Read from ``setup.py`` rather than restated here, because a hand-copied
    bound is exactly how this test came to disagree with the package it is
    supposed to police: it asserted ``>=0.6.0`` while the reference environment
    ran 0.4.4, so the only thing it measured was that the two had drifted.
    """
    text = (Path(__file__).resolve().parent.parent / "setup.py").read_text()
    match = re.search(r'"e3nn>=([0-9.]+),<([0-9.]+)"', text)
    assert match, "setup.py no longer declares an e3nn range in the expected form"
    return Version(match.group(1)), Version(match.group(2))


def test_e3nn_stable_api_series_is_supported():
    """The installed e3nn is inside the range ``setup.py`` claims to support.

    Both 0.4.x and 0.6.x are supported, and that range is measured rather than
    assumed -- the two published CsPbI3 checkpoints, trained under 0.4.4,
    evaluate to bit-identical energies on 0.6.0.  See the comment in
    ``setup.py`` and ``tests/electrostatics/test_e3nn_conventions.py``, which
    asserts the conventions the package depends on against whichever series is
    installed.
    """
    floor, ceiling = _declared_e3nn_range()
    version = Version(e3nn.__version__)
    assert floor <= version < ceiling, (
        f"installed e3nn {version} is outside the {floor} <= v < {ceiling} that "
        "setup.py declares. Either the environment or the declaration is wrong; "
        "do not widen this test without checking that checkpoints still evaluate "
        "identically on the version being added."
    )


def test_numpy_scalar_checkpoint_loads_with_weights_only_mode():
    checkpoint = {
        "model_state_dict": {"weight": torch.arange(3, dtype=torch.float32)},
        "config": {"energy_shift_per_atom": np.float64(-1.25)},
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        torch.save(checkpoint, path)
        loaded = load_checkpoint(path, map_location="cpu")

    torch.testing.assert_close(
        loaded["model_state_dict"]["weight"],
        checkpoint["model_state_dict"]["weight"],
    )
    assert float(loaded["config"]["energy_shift_per_atom"]) == -1.25
