"""Per-element reference energies solved on the residual of the initial model (``atomic_energies_from_model``).

The formal electrostatics of TRACE-Redox (SPEC 17.5) carries the ionisation and Madelung energy of the
ionic state: about -8 eV/atom for Au2/MgO (probe P5: -880 eV per 110-atom cell).  Reference energies
solved from the DFT energies alone leave that offset to the network (smoke run: energy RMSE
7842 meV/atom against an offset-free 139), and a bias moving at most ~lr per step cannot learn it.

Solving the per-element references on r_s = E_DFT,s - E_model,s (the initial model, eval mode) puts
the offset in the reference: the least-squares fit makes the mean residual per composition zero, so
the energy RMSE at initialisation equals its offset-free part up to what composition cannot absorb.
"""
import warnings

import pytest
import torch
import yaml
from ase.io import write
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)
from test_train_redox import HEADS, MODEL_KWARGS, water

import train


def _config(tmp_path, **extra):
    """Five geometries of ONE composition (water): a per-element reference can then absorb the whole
    mean error, so the fitted run's energy bias is zero by construction (to round-off)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    frames = [water(s, shift=0.02 * s) for s in range(5)]
    write(tmp_path / "train.extxyz", frames)
    write(tmp_path / "valid.extxyz", [water(9, shift=0.03)])
    config = {
        "device": "cpu", "seed": 0,
        "train_file": str(tmp_path / "train.extxyz"), "valid_file": str(tmp_path / "valid.extxyz"),
        "architecture_version": 2, **MODEL_KWARGS,
        "batch_size": 5, "epochs": 1, "learning_rate": 0.0, "solve_atomic_energies": True,
        "energy_weight": 1.0, "forces_weight": 10.0, "num_workers": 0, "batched_graphs": True,
        "model_save_path": str(tmp_path / "model.pt"),
        "save_last_checkpoint": False, "plot_training_curves": False,
        "electrostatics": {"enabled": True, "heads": dict(HEADS),
                           "redox": {"fragment_ensemble": "diabatic", "anchor": "formal", "max_charge": 1}},
        **extra,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _history(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return train.main(["--config", str(path)])


def test_references_on_the_initial_residual_absorb_the_electrostatic_offset(tmp_path):
    """lr = 0, no dropout: the epoch's train metrics are the initial model's.  Without the fit the mean
    per-atom error (the bias) is the electrostatic offset; with it, zero."""
    plain = _history(_config(tmp_path / "plain"))
    fitted = _history(_config(tmp_path / "fitted", atomic_energies_from_model=True))
    bias_plain, bias_fit = plain["train_energy_bias"][0], fitted["train_energy_bias"][0]   # meV/atom
    assert abs(bias_plain) > 100.0
    assert abs(bias_fit) <= 1e-6 * abs(bias_plain)
    # the offset-free part is the same model's, untouched by any reference
    assert abs(fitted["train_energy_rmse_offset_free"][0] - plain["train_energy_rmse_offset_free"][0]) <= \
        1e-9 * plain["train_energy_rmse_offset_free"][0]
    checkpoint = torch.load(tmp_path / "fitted" / "model.pt", map_location="cpu")
    assert checkpoint["config"]["atomic_energies"]


def test_the_fit_is_refused_on_a_resumed_run(tmp_path):
    path = _config(tmp_path, atomic_energies_from_model=True, resume_from=str(tmp_path / "missing.pt"))
    with pytest.raises(ValueError, match="resume"):
        _history(path)
