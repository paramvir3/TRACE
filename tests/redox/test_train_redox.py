"""TRACE-Redox in the training pipeline (train.py ``electrostatics.redox``, flashace).

Three real optimiser steps through ``train.main`` on neutral, charged, ion-pair
and periodic structures; the checkpoint carries the redox settings, reloads in
the ASE calculator, and the calculator reproduces the model forward exactly.
The block's refusals are tested by key.
"""

import math
import warnings

import numpy as np
import pytest
import torch
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

import train

R_MAX = 4.0
MODEL_KWARGS = dict(
    r_max=R_MAX, l_max=2, num_radial=4, hidden_dim=8, num_layers=1,
    correlation_order=2, correlation_channels=4, attention_num_heads=2,
)
HEADS = {"hidden_dim": 16, "key_dim": 8, "bond_radial_basis": 4, "r_max": 3.5}


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _labelled(atoms, seed, energy):
    rng = np.random.default_rng(seed)
    atoms.calc = SinglePointCalculator(
        atoms, energy=float(energy + rng.normal(0.0, 0.05)),
        forces=rng.normal(0.0, 0.1, size=(len(atoms), 3)),
    )
    return atoms


def water(seed=0, shift=0.0):
    atoms = Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.757 + shift, 0.586, 0.0], [-0.757, 0.586 - shift, 0.05]])
    atoms = _labelled(atoms, seed, -0.5)
    atoms.calc.results["dipole"] = np.array([0.02, 0.39, -0.01])
    atoms.arrays["partial_charges"] = np.array([-0.7, 0.35, 0.35])
    return atoms


def hydroxide(seed=1):
    atoms = _labelled(Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.97, 0.05, 0.02]]), seed, -2.0)
    atoms.info["charge"] = -1
    atoms.arrays["partial_charges"] = np.array([-1.3, 0.3])
    return atoms


def sodium_chloride(seed=2, separation=2.4):
    atoms = _labelled(Atoms("NaCl", positions=[[0.0, 0.0, 0.0], [separation, 0.1, 0.0]]), seed, -1.5)
    atoms.arrays["partial_charges"] = np.array([0.8, -0.8])
    return atoms


def sodium_cation(seed=3):
    atoms = _labelled(Atoms("Na", positions=[[0.0, 0.0, 0.0]]), seed, 5.1)
    atoms.info["charge"] = 1
    atoms.arrays["partial_charges"] = np.array([1.0])
    return atoms


def rocksalt(seed=4, displacement=0.1):
    atoms = _labelled(
        Atoms("NaCl", positions=[[0.0, 0.0, 0.0], [2.82 + displacement, 0.1, 0.05]],
              cell=[[0.0, 2.82, 2.82], [2.82, 0.0, 2.82], [2.82, 2.82, 0.0]], pbc=True),
        seed, -3.0,
    )
    atoms.arrays["partial_charges"] = np.array([0.8, -0.8])
    return atoms


def _write_config(tmp_path, batched, redox=None):
    write(tmp_path / "train.extxyz", [water(0), hydroxide(1), sodium_chloride(2), sodium_cation(3), rocksalt(4)])
    write(tmp_path / "valid.extxyz", [water(5, shift=0.03), sodium_chloride(6, separation=3.0)])
    config = {
        "device": "cpu", "seed": 0,
        "train_file": str(tmp_path / "train.extxyz"), "valid_file": str(tmp_path / "valid.extxyz"),
        "architecture_version": 2, **MODEL_KWARGS,
        "batch_size": 5, "epochs": 3, "learning_rate": 1e-3,
        "energy_weight": 1.0, "forces_weight": 10.0, "num_workers": 0,
        "batched_graphs": batched,
        "model_save_path": str(tmp_path / "model.pt"),
        "save_last_checkpoint": False, "plot_training_curves": False,
        "electrostatics": {
            "enabled": True, "heads": dict(HEADS),
            "dipole_weight": 1.0, "partial_charge_weight": 0.1,
            "redox": {"kT": 0.1} if redox is None else redox,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("batched", [False, True])
def test_three_training_steps_with_redox_have_finite_losses(tmp_path, batched):
    history = train.main(["--config", str(_write_config(tmp_path, batched))])
    assert torch.get_default_dtype() is torch.float64
    assert len(history["epoch"]) == 3
    # observables without labels (polarizability, Born, dielectric: refused with redox)
    # are reported as nan by the pipeline's convention; every active quantity is finite
    absent = ("polarizability", "born_charges", "dielectric_tensor")
    for key, values in history.items():
        if key != "epoch" and not any(name in key for name in absent):
            assert all(math.isfinite(v) for v in values), (key, values)
    assert history["train_partial_charges_rmse"][-1] > 0.0
    assert history["train_dipole_rmse"][-1] > 0.0
    checkpoint = torch.load(tmp_path / "model.pt", map_location="cpu")
    # the RESOLVED RedoxConfig, every field: a model reloads with the physics it was trained
    # under, whatever trace_redox's defaults are by then
    import dataclasses
    from trace_redox import RedoxConfig
    assert checkpoint["config"]["electrostatics"]["redox"] == dataclasses.asdict(RedoxConfig(kT=0.1))
    state = checkpoint["model_state_dict"]
    assert any(k.startswith("electrostatics.charge_states.") for k in state)
    assert any(k.startswith("electrostatics.width.") for k in state)


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("trained_redox")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.set_default_dtype(torch.float64)
        train.main(["--config", str(_write_config(directory, batched=False))])
    return directory / "model.pt"


def _calculator(path, **kwargs):
    from flashace.calculator import TransformersACECalculator

    return TransformersACECalculator(model_path=str(path), device="cpu", **kwargs)


def test_the_checkpoint_reloads_trace_redox_in_the_calculator(trained_checkpoint):
    from trace_redox import RedoxElectrostatics

    torch.set_default_dtype(torch.float32)          # as in a fresh inference process
    calc = _calculator(trained_checkpoint)
    assert torch.get_default_dtype() is torch.float64
    assert isinstance(calc.model.electrostatics, RedoxElectrostatics)
    assert calc.model.electrostatics.redox.kT == 0.1
    saved = torch.load(trained_checkpoint, map_location="cpu")["model_state_dict"]
    for name, value in calc.model.state_dict().items():
        assert torch.equal(value, saved[name]), name


@pytest.mark.parametrize("structure", [water, hydroxide, sodium_chloride, sodium_cation, rocksalt],
                         ids=["neutral", "anion", "ion-pair", "cation", "periodic"])
def test_calculator_matches_the_model_forward(trained_checkpoint, structure):
    calc = _calculator(trained_checkpoint)
    atoms, reference = structure(), structure()
    atoms.calc = calc
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    item = train.AtomisticDataset([reference], R_MAX, electrostatics=True)[0]
    periodic = bool(reference.pbc.any())
    E, F, S, _ = calc.model(item, training=False, compute_stress=periodic)
    baseline = calc.energy_shift_per_atom * len(reference)
    assert abs(energy - (float(E) + baseline)) < 1e-10
    assert np.abs(forces - F.detach().numpy()).max() < 1e-10


def test_charge_changes_the_energy_by_the_free_sector(trained_checkpoint):
    """The calculator reads atoms.info['charge']: Na with charge +1 differs from neutral Na by ~IE."""
    calc = _calculator(trained_checkpoint)
    neutral = Atoms("Na", positions=[[0.0, 0.0, 0.0]])
    cation = neutral.copy()
    cation.info["charge"] = 1
    neutral.calc = calc
    e0 = neutral.get_potential_energy()
    cation.calc = calc
    e1 = cation.get_potential_energy()
    # the free sector of an isolated Na: E(+1) = IE_Na exactly at the anchors, plus
    # the trained bounded shifts (|chi shift| <= 2 eV, gap factor e^{+-1}) -- so between 1 and 12 eV
    assert 1.0 < e1 - e0 < 12.0


def test_fragment_policy_initial_is_refused_for_a_redox_checkpoint(trained_checkpoint):
    with pytest.raises(ValueError, match="fragment_policy"):
        _calculator(trained_checkpoint, fragment_policy="initial")


@pytest.mark.parametrize("block,match", [
    ({"redox": {"kTT": 0.1}}, "unknown key"),
    ({"redox": {"kT": -1.0}}, "kT"),
    ({"redox": {}, "polarizability_weight": 0.1}, "polarizability_weight"),
    ({"redox": {}, "born_charge_weight": 0.1}, "born_charge_weight"),
    ({"redox": {}, "transfer_coupling": {}}, "transfer_coupling"),
    ({"redox": {"attention_cutoff": 6.0}}, "attention_cutoff"),
    ({"redox": [1, 2]}, "mapping"),
])
def test_invalid_redox_blocks_are_refused(block, match):
    full = {"enabled": True, "heads": dict(HEADS)}
    full.update(block)
    with pytest.raises(ValueError, match=match):
        train.parse_electrostatics_config({"r_max": R_MAX, "electrostatics": full})


def test_yaml_float_strings_reach_the_redox_config():
    settings = train.parse_electrostatics_config(
        {"r_max": R_MAX, "electrostatics": {"enabled": True, "heads": dict(HEADS), "redox": {"kT": "5e-2"}}}
    )
    assert settings.redox == {"kT": 0.05}
