"""A training-time electronic temperature for TRACE-Redox (audit C3: a continuation heuristic, no physics claim).

A configuration that lies Delta above the populated one carries weight exp(-Delta / kT); at the
deployed kT = 0.1 eV and the Au2/MgO:Al margins of 0.6-2.4 eV (probe P5) that is e^-6 to e^-24, so its
energy receives no gradient (B6).  Training starts hotter and anneals geometrically,

    kT(e) = kT_end (kT_start / kT_end)^max(0, 1 - e / E_s),

reaching EXACTLY kT_end = redox.kT at e = E_s < epochs.  Validation, checkpoints and deployment use
kT_end always, as the attention temperature does (T_att = 1).
"""
import math
import warnings

import pytest
import torch
import yaml
from ase.io import write
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)
from test_train_redox import HEADS, MODEL_KWARGS, hydroxide, sodium_cation, sodium_chloride, water

import train


def test_the_schedule_is_geometric_and_ends_exactly():
    kT = [train.redox_kT(e, 0.8, 0.1, 3) for e in range(6)]
    assert kT[0] == 0.8 and kT[3] == 0.1 and kT[4] == 0.1 and kT[5] == 0.1
    assert abs(kT[1] - 0.1 * 8.0 ** (2 / 3)) <= 1e-15 and abs(kT[2] - 0.1 * 8.0 ** (1 / 3)) <= 1e-15
    assert all(a > b for a, b in zip(kT[:3], kT[1:4]))


def _config(tmp_path, **extra):
    write(tmp_path / "train.extxyz", [water(0), hydroxide(1), sodium_chloride(2), sodium_cation(3)])
    write(tmp_path / "valid.extxyz", [water(5, shift=0.03)])
    config = {
        "device": "cpu", "seed": 0,
        "train_file": str(tmp_path / "train.extxyz"), "valid_file": str(tmp_path / "valid.extxyz"),
        "architecture_version": 2, **MODEL_KWARGS,
        "batch_size": 4, "epochs": 3, "learning_rate": 1e-3,
        "energy_weight": 1.0, "forces_weight": 10.0, "num_workers": 0, "batched_graphs": True,
        "model_save_path": str(tmp_path / "model.pt"),
        "save_last_checkpoint": False, "plot_training_curves": False,
        "electrostatics": {"enabled": True, "heads": dict(HEADS), "redox": {"kT": 0.1}},
        **extra,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_training_runs_hot_and_validation_and_checkpoint_stay_at_the_deployed_kT(tmp_path, monkeypatch):
    from trace_redox import RedoxElectrostatics
    seen = []
    original = RedoxElectrostatics.forward

    def recording(self, *args, **kwargs):
        seen.append((bool(self.training), float(self.redox.kT)))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RedoxElectrostatics, "forward", recording)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train.main(["--config", str(_config(tmp_path, redox_kT_start=0.4, redox_kT_epochs=2))])
    training = [kT for is_training, kT in seen if is_training]
    validation = [kT for is_training, kT in seen if not is_training]
    # one batch of 4 structures per epoch: per-structure forwards inside the batch carry the epoch's kT
    epochs = sorted(set(training), reverse=True)
    assert len(epochs) == 3 and all(abs(a - b) <= 1e-15 for a, b in zip(epochs, [0.4, math.sqrt(0.4 * 0.1), 0.1]))
    assert set(validation) == {0.1}
    checkpoint = torch.load(tmp_path / "model.pt", map_location="cpu")
    assert abs(checkpoint["config"]["electrostatics"]["redox"]["kT"] - 0.1) <= 0.0


@pytest.mark.parametrize("extra,match", [
    ({"redox_kT_start": 0.4, "redox_kT_epochs": 3}, "before training ends"),
    ({"redox_kT_start": 0.05, "redox_kT_epochs": 2}, "above"),
    ({"redox_kT_start": 0.4}, "together"),
])
def test_invalid_schedules_are_refused(tmp_path, extra, match):
    with pytest.raises(ValueError, match=match):
        train.main(["--config", str(_config(tmp_path, **extra))])
