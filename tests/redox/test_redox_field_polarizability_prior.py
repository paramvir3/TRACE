"""The whole model's static polarisability as a training target (``field_polarizability_prior``; 2026-10-18).

``polarizability_prior`` trains the bound sector's response and must refuse any structure with several configurations
(the thermal and coherent charge-transfer terms are not in bound_state).  But the charge-transfer response is the
observable that identifies the free sector: measured, isolated Au2 at 2.30-3.10 A, alpha_par / (2 alpha_atom) is 1.36
-> 2.30 from first principles (PBE0/def2-TZVPPD), while two seeds of one model give 1.23 -> 1.76 and 1.00 -> 1.16
(non-monotone), with alpha_perp right to 2-5 % in every model.  Charge partitions (Hirshfeld) cannot do this.

The field prior is the model's own field Hessian, alpha_ii = -[E(+h e_i) - 2 E(0) + E(-h e_i)] / h^2 x K_E
(h = 0.01 V/A, total energies through the whole model: bound sector, configuration ensemble, couplings), so it reaches
every parameter of the free sector and needs only first derivatives in the weights.  NEW-TO-TRACE: training the
configuration ensemble's field response against first-principles polarisabilities.
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

COUPLED = {"fragment_ensemble": "diabatic", "anchor": "formal", "max_charge": 1, "couplings": True}


def _dimer(alpha=(60.0, 30.0, 30.0), off=0.0, periodic=False):
    """Na2+ at 3.0 A along x: two degenerate configurations, coupled."""
    atoms = Atoms("Na2", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    atoms.info["charge"] = 1
    tensor = np.diag(alpha)
    tensor[0, 1] = tensor[1, 0] = off
    atoms.info["polarizability"] = tensor.tolist()
    if periodic:
        atoms.cell = 14.0 * np.eye(3)
        atoms.pbc = True
    return atoms


def _config(tmp_path, prior_file, epochs=4):
    tmp_path.mkdir(parents=True, exist_ok=True)
    write(tmp_path / "train.extxyz", [rocksalt(4), rocksalt(5, displacement=0.05)])
    write(tmp_path / "valid.extxyz", [rocksalt(6, displacement=0.08)])
    config = {
        "device": "cpu", "seed": 0,
        "train_file": str(tmp_path / "train.extxyz"), "valid_file": str(tmp_path / "valid.extxyz"),
        "architecture_version": 2, **MODEL_KWARGS,
        "batch_size": 2, "epochs": epochs, "learning_rate": 5e-3,
        "energy_weight": 0.0, "forces_weight": 0.0, "num_workers": 0, "batched_graphs": True,
        "model_save_path": str(tmp_path / "model.pt"),
        "save_last_checkpoint": False, "plot_training_curves": False,
        "electrostatics": {"enabled": True, "heads": dict(HEADS), "partial_charge_weight": 0.0,
                           "redox": dict(COUPLED),
                           "field_polarizability_prior": {"file": prior_file, "weight": 1.0}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _train(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return train.main(["--config", str(path)])


def _items(prior, config_path):
    settings = train.parse_electrostatics_config(yaml.safe_load(open(config_path)))
    return train._response_prior_items({"file": prior, "weight": 1.0}, "field_polarizability", MODEL_KWARGS["r_max"],
                                       torch.float64, settings, False), settings


def test_the_field_prior_trains_a_dimer_with_several_configurations(tmp_path):
    prior = str(tmp_path / "dimer.extxyz")
    write(prior, [_dimer()])
    history = _train(_config(tmp_path / "run", prior))
    val = history["val_field_polarizability_prior_rmse"]
    assert len(val) == 4 and all(math.isfinite(v) for v in val + history["train_field_polarizability_prior_rmse"])
    assert val[-1] < val[0], val


def test_its_alpha_is_the_calculators_finite_field_response(tmp_path):
    from flashace.calculator import TransformersACECalculator
    from trace_electrostatics.constants import K_E
    prior = str(tmp_path / "dimer.extxyz")
    write(prior, [_dimer()])
    _train(_config(tmp_path / "run", prior, epochs=1))
    calc = TransformersACECalculator(model_path=str(tmp_path / "run" / "model.pt"), device="cpu")

    def energy(field):
        atoms = _dimer()
        atoms.info["external_field"] = list(field)
        atoms.calc = calc
        calc.reset()
        return atoms.get_potential_energy()

    h = train.FIELD_PRIOR_STEP
    fd = [-(energy(h * e) - 2 * energy(0 * e) + energy(-h * e)) / h ** 2 * K_E for e in np.eye(3)]
    items, settings = _items(prior, tmp_path / "run" / "config.yaml")
    _, errors = train._response_prior_loss(calc.model, items, "field_polarizability", "cpu", settings, 1.0, training=False)
    alpha = errors[0] + torch.tensor([60.0, 30.0, 30.0], dtype=torch.float64)
    # the same central difference of the same energies: round-off, eps |E| / h^2 K_E ~ 1e-16 * 1e3 / 1e-4 * 14
    for k in range(3):
        assert abs(float(alpha[k]) - fd[k]) <= 1e-5 * (1.0 + abs(fd[k])), (k, float(alpha[k]), fd[k])
    assert float(alpha[0]) - float(alpha[1]) > 1e-3       # the coherent transfer along the bond adds to alpha_xx


def test_it_reaches_the_free_sector(tmp_path):
    """The loss gradient reaches the couplings and the charge-state shifts, which the bound-state prior cannot."""
    prior = str(tmp_path / "dimer.extxyz")
    write(prior, [_dimer()])
    _train(_config(tmp_path / "run", prior, epochs=1))
    from flashace.calculator import TransformersACECalculator
    model = TransformersACECalculator(model_path=str(tmp_path / "run" / "model.pt"), device="cpu").model
    items, settings = _items(prior, tmp_path / "run" / "config.yaml")
    loss, _ = train._response_prior_loss(model, items, "field_polarizability", "cpu", settings, 1.0, training=True)
    model.zero_grad(set_to_none=True)
    loss.backward()
    named = dict(model.named_parameters())
    for prefix in ("electrostatics.coupling.", "electrostatics.charge_states.shift_mlp."):
        grads = [p.grad for n, p in named.items() if n.startswith(prefix) and p.grad is not None]
        assert grads and max(float(g.abs().max()) for g in grads) > 0.0, prefix


@pytest.mark.parametrize("structure,match", [
    (lambda: _dimer(periodic=True), "needs open boundaries"),
    (lambda: _dimer(off=5.0), "diagonal"),
], ids=["periodic", "off-diagonal"])
def test_invalid_field_priors_are_refused(tmp_path, structure, match):
    prior = str(tmp_path / "bad.extxyz")
    write(prior, [structure()])
    with pytest.raises(ValueError, match=match):
        _train(_config(tmp_path / "run", prior))
