"""Atomic polarizabilities as a training target for TRACE-Redox with formal anchors (``polarizability_prior``).

The model's electronic response is the bound sector's (``bound_state``) when the configuration window holds ONE
configuration: then -d2F/dE2 = alpha_b exactly.  With several configurations it is not:
- the thermal term Var_p(d) / kT is added (degenerate configurations make it large);
- with couplings, a coherent charge-transfer response sum_k |<0|d|k>|^2 / (E_k - E_0) is added.
Since SPEC 30 (2026-10-19) the prior adds both, alpha_ens of trace_redox/response.py, so such structures train on
the whole response instead of being refused.  An isolated neutral atom is the clean case: alpha = alpha_b, the measured atomic polarizability.
"""
import math
import warnings

import numpy as np
import pytest
import torch
import yaml
from ase import Atoms
from ase.io import write
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)
from test_train_redox import HEADS, MODEL_KWARGS, rocksalt

import train

FORMAL = {"fragment_ensemble": "diabatic", "anchor": "formal", "max_charge": 1}


def _write(path, atoms_list):
    write(path, atoms_list)
    return str(path)


def _atom(symbol="Na", alpha=24.0, periodic=False, label=True):
    atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]])
    if periodic:
        atoms.cell = 12.0 * np.eye(3)
        atoms.pbc = True
    if label:
        atoms.info["polarizability"] = (alpha * np.eye(3)).tolist()
    return atoms


def _cation_dimer(alpha=40.0):
    """Na2+ at 3.0 A: (Na+, Na0) and (Na0, Na+) are two configurations of equal energy by symmetry."""
    atoms = Atoms("Na2", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    atoms.info["charge"] = 1
    atoms.info["polarizability"] = (alpha * np.eye(3)).tolist()
    return atoms


def _config(tmp_path, prior_file, redox=FORMAL, **prior):
    tmp_path.mkdir(parents=True, exist_ok=True)
    write(tmp_path / "train.extxyz", [rocksalt(4), rocksalt(5, displacement=0.05)])
    write(tmp_path / "valid.extxyz", [rocksalt(6, displacement=0.08)])
    block = {"file": prior_file, "weight": 1.0}
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
                           "redox": dict(redox), "polarizability_prior": block},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _train(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return train.main(["--config", str(path)])


def test_a_polarizability_prior_trains_alpha_on_an_isolated_atom(tmp_path):
    prior = _write(tmp_path / "atom.extxyz", [_atom()])
    history = _train(_config(tmp_path / "run", prior))
    train_rmse, val_rmse = history["train_polarizability_prior_rmse"], history["val_polarizability_prior_rmse"]
    assert len(val_rmse) == 4 and all(math.isfinite(v) for v in train_rmse + val_rmse)
    assert val_rmse[-1] < val_rmse[0] and val_rmse[-1] == min(val_rmse), val_rmse


def test_the_prior_polarizability_is_the_models_own_finite_field_response(tmp_path):
    """The trained checkpoint's alpha of the prior atom by central differences of the TOTAL energy in an
    applied field (the calculator, open boundaries) equals the bound-state alpha the prior trained on."""
    from flashace.calculator import TransformersACECalculator
    from trace_electrostatics.constants import K_E
    from trace_electrostatics.observables import polarizability
    prior = _write(tmp_path / "atom.extxyz", [_atom()])
    _train(_config(tmp_path / "run", prior))
    calc = TransformersACECalculator(model_path=str(tmp_path / "run" / "model.pt"), device="cpu")

    def energy(field):
        atoms = _atom(label=False)
        atoms.info["external_field"] = list(field)
        atoms.calc = calc
        calc.reset()
        return atoms.get_potential_energy()

    h = 0.02
    fd = -(energy([0, 0, h]) - 2 * energy([0, 0, 0]) + energy([0, 0, -h])) / h ** 2 * K_E      # A^3
    items = train._response_prior_items({"file": prior, "weight": 1.0}, "polarizability", MODEL_KWARGS["r_max"],
                                        torch.float64, train.parse_electrostatics_config(
                                            yaml.safe_load(open(tmp_path / "run" / "config.yaml"))), False)
    bat = train.AtomisticDataset.collate_batched(items)
    with torch.enable_grad():
        _, _, _, aux = calc.model.forward_batched(bat, training=False, return_electrostatics=True)
        state = aux["electrostatics"][0].state()
        alpha = float(polarizability(state.solve_result, state.assembly, as_volume=True)[2, 2])
    # E is exactly quadratic in the field for one configuration: the difference is round-off,
    # eps |E| / h^2 * K_E ~ 1e-16 * 10 / 4e-4 * 14 ~ 4e-11 A^3
    assert abs(fd - alpha) <= 1e-7 * (1.0 + abs(alpha)), (fd, alpha)


@pytest.mark.parametrize("couplings", [False, True], ids=["uncoupled", "coupled"])
def test_a_structure_with_several_configurations_trains_on_the_whole_response(tmp_path, couplings):
    """Na2+ has two degenerate configurations.  Before SPEC 30 the prior refused it (the thermal and, coupled, the
    coherent response are not in bound_state).  Now it trains alpha_b + alpha_ens, the whole response of the ensemble,
    and that equals the trained checkpoint's finite-field alpha of the TOTAL energy (window selection: no switch)."""
    from flashace.calculator import TransformersACECalculator
    from trace_electrostatics.constants import K_E
    prior = _write(tmp_path / "dimer.extxyz", [_cation_dimer()])
    redox = dict(FORMAL, couplings=couplings)
    history = _train(_config(tmp_path / "run", prior, redox=redox))
    val = history["val_polarizability_prior_rmse"]
    assert all(math.isfinite(v) for v in val) and val[-1] < val[0], val
    calc = TransformersACECalculator(model_path=str(tmp_path / "run" / "model.pt"), device="cpu")

    def energy(field):
        atoms = _cation_dimer()
        atoms.info["external_field"] = list(field)
        atoms.calc = calc
        calc.reset()
        return atoms.get_potential_energy()

    items = train._response_prior_items({"file": prior, "weight": 1.0}, "polarizability", MODEL_KWARGS["r_max"],
                                        torch.float64, train.parse_electrostatics_config(
                                            yaml.safe_load(open(tmp_path / "run" / "config.yaml"))), False)
    settings = train.parse_electrostatics_config(yaml.safe_load(open(tmp_path / "run" / "config.yaml")))
    _, errors = train._response_prior_loss(calc.model, items, "polarizability", "cpu", settings, 1.0, training=False)
    alpha = errors[0] + torch.tensor(40.0 * np.eye(3), dtype=torch.float64)
    for axis in range(3):
        e = np.zeros(3)
        rich = []
        for h in (1e-3, 5e-4):
            e[:] = 0.0
            e[axis] = h
            rich.append(-(energy(e) - 2 * energy(0 * e) + energy(-e)) / h ** 2 * K_E)
        fd = (4.0 * rich[1] - rich[0]) / 3.0
        assert abs(float(alpha[axis, axis]) - fd) <= 1e-5 * (1.0 + abs(fd)), (axis, float(alpha[axis, axis]), fd)


@pytest.mark.parametrize("structure,match", [
    (lambda: _atom(periodic=True), "needs open boundaries"),
    (lambda: _atom(label=False), "carries no polarizability label"),
], ids=["periodic", "unlabelled"])
def test_invalid_polarizability_priors_are_refused(tmp_path, structure, match):
    prior = _write(tmp_path / "bad.extxyz", [structure()])
    with pytest.raises(ValueError, match=match):
        _train(_config(tmp_path / "run", prior))


def test_the_dielectric_prior_accepts_a_crystal_with_one_configuration_under_couplings(tmp_path):
    """The guard is shared with the eps_inf prior; a formal rocksalt crystal has one configuration in the
    window, so it is accepted with couplings on."""
    from test_redox_dielectric_train import FORMAL as F2, _prior_config
    path = _prior_config(tmp_path / "run", redox=dict(F2, couplings=True))
    history = _train(path)                                     # rocksalt NaCl: one configuration, accepted
    assert math.isfinite(history["val_dielectric_prior_rmse"][-1])


def test_two_clusters_train_on_the_whole_response_in_the_clusters_space(tmp_path):
    """SPEC 36.10 (2026-11-12): two neutral Na2 6.5 A apart are two clusters of two configurations each.  SPEC 36 as
    first built refused the response record of several clusters (limit L5), so this prior stopped the training; the
    record now holds one term per free energy of the cluster product and the prior trains on the whole response."""
    from flashace.calculator import TransformersACECalculator
    atoms = Atoms("Na4", positions=[[0.0, 0.0, 0.0], [0.3, 0.2, 3.0], [6.5, 0.4, 0.3], [6.8, 0.6, 3.3]])
    atoms.info["charge"] = 0
    atoms.info["polarizability"] = (60.0 * np.eye(3)).tolist()
    prior = _write(tmp_path / "dimers.extxyz", [atoms])
    redox = dict(FORMAL, couplings=True, diabatic_space="clusters")
    history = _train(_config(tmp_path / "run", prior, redox=redox))
    val = history["val_polarizability_prior_rmse"]
    assert all(math.isfinite(v) for v in val) and val[-1] < val[0], val
    calc = TransformersACECalculator(model_path=str(tmp_path / "run" / "model.pt"), device="cpu")
    module = calc.model.electrostatics
    atoms.calc = calc
    atoms.get_potential_energy()
    clusters = module._ensemble_last["clusters"]                       # the case of L5: several clusters, each with moves
    assert module._ensemble_last["H"] is None and len(clusters) >= 2 and min(int(S.shape[0]) for S in clusters) >= 2
