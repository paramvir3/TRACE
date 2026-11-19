"""eps_inf as a training target for TRACE-Redox with formal anchors (the bound sector's response).

``anchor='formal'``: integer free charges, so the model's electronic response is the bound sector's up to
the thermal term Var_p(d) / kT ~ exp(-gap / kT) (test_redox_bound_state.py).  The dielectric tensor is
then the bound state's, and a loss on it is a loss on the model.  The mean-field anchors keep the
refusal: their continuous free charges respond linearly as well.
"""
import math
import warnings

import numpy as np
import pytest
import torch
import yaml
from ase.io import write
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)
from test_train_redox import HEADS, MODEL_KWARGS, rocksalt

import train

FORMAL = {"fragment_ensemble": "diabatic", "anchor": "formal", "max_charge": 1}


def _labelled_rocksalt(seed, displacement, eps):
    atoms = rocksalt(seed, displacement=displacement)
    atoms.info["dielectric_tensor"] = (eps * np.eye(3)).tolist()
    return atoms


def _config(tmp_path, redox, **extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    write(tmp_path / "train.extxyz", [_labelled_rocksalt(4, 0.10, 2.4), _labelled_rocksalt(5, 0.05, 2.4)])
    write(tmp_path / "valid.extxyz", [_labelled_rocksalt(6, 0.08, 2.4)])
    config = {
        "device": "cpu", "seed": 0,
        "train_file": str(tmp_path / "train.extxyz"), "valid_file": str(tmp_path / "valid.extxyz"),
        "architecture_version": 2, **MODEL_KWARGS,
        "batch_size": 2, "epochs": 4, "learning_rate": 5e-3,
        "energy_weight": 0.0, "forces_weight": 0.0, "num_workers": 0, "batched_graphs": True,
        "model_save_path": str(tmp_path / "model.pt"),
        "save_last_checkpoint": False, "plot_training_curves": False,
        "electrostatics": {"enabled": True, "heads": dict(HEADS), "partial_charge_weight": 0.0,
                           "dielectric_weight": 1.0, "redox": dict(redox)},
        **extra,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _train(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return train.main(["--config", str(path)])


@pytest.mark.parametrize("redox", [{}, {"fragment_ensemble": "diabatic"}], ids=["mean-field", "diabatic-atomic"])
def test_the_dielectric_loss_stays_refused_without_formal_anchors(tmp_path, redox):
    with pytest.raises(ValueError, match="formal"):
        _train(_config(tmp_path, redox))


def test_formal_models_train_eps_inf_on_the_bound_sector(tmp_path):
    history = _train(_config(tmp_path, FORMAL))
    rmse = history["train_dielectric_tensor_rmse"]
    assert len(rmse) == 4 and all(math.isfinite(v) for v in rmse)
    # a dielectric-only loss must lower its own error: every epoch below the first, the last lowest
    assert all(v < rmse[0] for v in rmse[1:]), rmse
    assert rmse[-1] == min(rmse), rmse
    assert math.isfinite(history["val_dielectric_tensor_rmse"][-1])


# ---- electrostatics.dielectric_prior: eps_inf on structures that carry no energies (bulk MgO for Au2/MgO) ----

def _prior_file(path, *, periodic=True, label=True, eps=2.4):
    atoms = rocksalt(7, displacement=0.0)
    atoms.calc = None                                         # no energy, no forces: only the eps_inf label
    atoms.arrays.pop("partial_charges", None)
    if not periodic:
        atoms.pbc = False
    if label:
        atoms.info["dielectric_tensor"] = (eps * np.eye(3)).tolist()
    write(path, [atoms])
    return str(path)


def _prior_config(tmp_path, redox=FORMAL, **prior):
    """Training structures WITHOUT dielectric labels, energy and force weights 0: only the prior trains."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    write(tmp_path / "train.extxyz", [rocksalt(4), rocksalt(5, displacement=0.05)])
    write(tmp_path / "valid.extxyz", [rocksalt(6, displacement=0.08)])
    block = {"file": _prior_file(tmp_path / "prior.extxyz"), "weight": 1.0}
    block.update(prior)
    config = {
        "device": "cpu", "seed": 0,
        "train_file": str(tmp_path / "train.extxyz"), "valid_file": str(tmp_path / "valid.extxyz"),
        "architecture_version": 2, **MODEL_KWARGS,
        "batch_size": 2, "epochs": 4, "learning_rate": 5e-3,
        "energy_weight": 0.0, "forces_weight": 0.0, "num_workers": 0, "batched_graphs": True,
        "model_save_path": str(tmp_path / "model.pt"),
        "save_last_checkpoint": False, "plot_training_curves": False,
        "electrostatics": {"enabled": True, "heads": dict(HEADS), "partial_charge_weight": 0.0,
                           "redox": dict(redox), "dielectric_prior": block},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_a_dielectric_prior_trains_eps_inf_on_structures_without_energies(tmp_path):
    history = _train(_prior_config(tmp_path))
    train_rmse, val_rmse = history["train_dielectric_prior_rmse"], history["val_dielectric_prior_rmse"]
    assert len(train_rmse) == 4 and all(math.isfinite(v) for v in train_rmse + val_rmse)
    assert train_rmse[-1] < train_rmse[0] and val_rmse[-1] < val_rmse[0], (train_rmse, val_rmse)
    assert val_rmse[-1] == min(val_rmse), val_rmse


@pytest.mark.parametrize("change,match", [
    ({"redox": {}}, "formal"),
    ({"periodic": False}, "3-D periodic"),
    ({"label": False}, "dielectric_tensor"),
    ({"extra": {"weigth": 1.0}}, "unknown"),
])
def test_invalid_dielectric_priors_are_refused(tmp_path, change, match):
    redox = change.get("redox", FORMAL)
    prior = dict(change.get("extra", {}))
    if "periodic" in change or "label" in change:
        prior["file"] = _prior_file(tmp_path / "bad.extxyz", periodic=change.get("periodic", True),
                                    label=change.get("label", True))
    with pytest.raises(ValueError, match=match):
        _train(_prior_config(tmp_path, redox=redox, **prior))


@pytest.mark.parametrize("extra", [{}, {"state_polarizability": True},
                                   {"state_polarizability": True, "state_polarizability_elements": [11]}],
                         ids=["K", "K(n)", "K(n) listed elements"])
def test_a_periodic_prior_structure_with_several_configurations_trains_on_the_whole_response(tmp_path, extra):
    """The step-13F failure (2026-10-19): the eps_inf prior refused bulk MgO once the couplings had grown enough that
    its window held 5 configurations.  Since SPEC 30 the prior adds the configuration ensemble's response,
    eps = eps_b + 4 pi K_E alpha_ens / Omega, with minimal-image configuration dipoles.  A neutral Na2 at 3.0 A in a
    12 A box holds three configurations (Na+ Na- lies inside the window): accepted, and the prior trains.  K(n): with
    SPEC 28 (Na+ at its own polarisability; step 17's path), refused before and exact since (alpha_shift)."""
    from ase import Atoms
    atoms = Atoms("Na2", positions=[[1.0, 1.0, 1.0], [4.0, 1.2, 1.3]], cell=12.0 * np.eye(3), pbc=True)
    atoms.info["dielectric_tensor"] = (1.05 * np.eye(3)).tolist()
    path = tmp_path / "prior_na2.extxyz"
    tmp_path.mkdir(parents=True, exist_ok=True)
    write(path, [atoms])
    config_path = _prior_config(tmp_path / "run", redox=dict(FORMAL, couplings=True, **extra))
    cfg = yaml.safe_load(open(config_path))
    cfg["electrostatics"]["dielectric_prior"]["file"] = str(path)
    config_path.write_text(yaml.safe_dump(cfg))
    history = _train(config_path)
    val = history["val_dielectric_prior_rmse"]
    assert all(math.isfinite(v) for v in val + history["train_dielectric_prior_rmse"]), val
    assert val[-1] < val[0], val
