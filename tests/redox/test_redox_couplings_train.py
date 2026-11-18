"""One-electron couplings tau_uv (SPEC §17.6) through the training pipeline.

The module tests fix the physics (the symmetric dimer gains exactly -kT ln cosh(tau / kT); zero couplings
are the uncoupled module bit for bit; forces are -grad E).  What they cannot see is the pipeline:

- F depends on tau only at second order for an isolated move (F - F0 = -kT ln cosh(tau / kT) ~ -tau^2 / 2kT),
  so tau = 0 is stationary and the gradient at the initial scale is O(tau).  It must still be nonzero:
  with lr = 0 the head stays at its seeded initialisation, with lr > 0 it moves;
- the checkpoint carries the head, the calculator rebuilds it from the saved RedoxConfig and reloads
  its weights, and the reloaded head is active (zeroing it changes the energy of a coupled cation).
"""
import math
import warnings

import torch
import yaml
from ase import Atoms
from ase.io import write
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)
from test_train_redox import HEADS, MODEL_KWARGS, hydroxide, rocksalt, sodium_cation, sodium_chloride, water

import train

REDOX = {"fragment_ensemble": "diabatic", "anchor": "formal", "max_charge": 1, "couplings": True}


def _config(tmp_path, learning_rate):
    tmp_path.mkdir(parents=True, exist_ok=True)
    write(tmp_path / "train.extxyz", [water(0), hydroxide(1), sodium_chloride(2), sodium_cation(3), rocksalt(4)])
    write(tmp_path / "valid.extxyz", [water(5, shift=0.03), sodium_chloride(6, separation=3.0)])
    config = {
        "device": "cpu", "seed": 0,
        "train_file": str(tmp_path / "train.extxyz"), "valid_file": str(tmp_path / "valid.extxyz"),
        "architecture_version": 2, **MODEL_KWARGS,
        "batch_size": 5, "epochs": 2, "learning_rate": learning_rate,
        "energy_weight": 1.0, "forces_weight": 10.0, "num_workers": 0, "batched_graphs": True,
        "model_save_path": str(tmp_path / "model.pt"),
        "save_last_checkpoint": False, "plot_training_curves": False,
        "electrostatics": {"enabled": True, "heads": dict(HEADS), "partial_charge_weight": 0.0,
                           "redox": dict(REDOX)},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _train(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return train.main(["--config", str(path)])


def _coupling_state(path):
    state = torch.load(path, map_location="cpu")["model_state_dict"]
    return {k: v for k, v in state.items() if k.startswith("electrostatics.coupling.")}


def test_the_coupling_head_trains_saves_and_reloads(tmp_path):
    from flashace.calculator import TransformersACECalculator

    history = _train(_config(tmp_path / "trained", 1e-3))
    _train(_config(tmp_path / "frozen", 0.0))
    assert all(math.isfinite(v) for v in history["train_loss"])
    trained, frozen = _coupling_state(tmp_path / "trained" / "model.pt"), _coupling_state(tmp_path / "frozen" / "model.pt")
    assert trained and set(trained) == set(frozen)
    # same seed, same initialisation; only a nonzero gradient moves the head
    moved = max(float((trained[k] - frozen[k]).abs().max()) for k in trained)
    assert moved > 1e-6, moved
    saved = torch.load(tmp_path / "trained" / "model.pt", map_location="cpu")
    assert saved["config"]["electrostatics"]["redox"]["couplings"] is True

    calc = TransformersACECalculator(model_path=str(tmp_path / "trained" / "model.pt"), device="cpu")
    head = calc.model.electrostatics.coupling
    assert head is not None
    for name, value in head.state_dict().items():
        assert torch.equal(value, trained["electrostatics.coupling." + name]), name

    # the reloaded head is active: Na2+ at 3 A has two degenerate configurations coupled by tau
    def energy():
        atoms = Atoms("Na2", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        atoms.info["charge"] = 1
        atoms.calc = calc
        return atoms.get_potential_energy()

    coupled = energy()
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
    calc.reset()
    assert abs(coupled - energy()) > 1e-10
