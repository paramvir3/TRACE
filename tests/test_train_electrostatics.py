"""Training-pipeline integration of the variational electrostatics.

Pins the four things a peer review found missing from train.py / flashace:

1. an ``electrostatics:`` YAML block that attaches the package (off when absent);
2. float64 enforced when -- and only when -- it is enabled, AMP refused;
3. charged / fragmented structures routed through ``forward_batched`` per
   structure, agreeing with the single-structure path;
4. dipole, polarizability and Born-charge loss terms that reach the heads.

NOTE on dtype: ``tests/conftest.py`` gives every test whose node id contains
"electrostatics" -- this file included -- a float64 default dtype.  Tests that
need the float32 default set it explicitly.
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

import train
from flashace.model import TransformersACE

R_MAX = 4.0
MODEL_KWARGS = dict(
    r_max=R_MAX, l_max=2, num_radial=4, hidden_dim=8, num_layers=1,
    correlation_order=2, correlation_channels=4, attention_num_heads=2,
)
HEADS = {"hidden_dim": 16, "key_dim": 8, "bond_radial_basis": 4}

#: The item keys of a run without electrostatics, which must not change.
PLAIN_ITEM_KEYS = {
    "z", "pos", "cell", "edge_index", "edge_shift", "volume",
    "t_E", "t_F", "t_S", "has_stress", "has_energy", "force_weight",
}
PLAIN_BATCH_KEYS = {
    "z", "pos", "cell", "volume", "batch", "n_structures", "edge_index",
    "edge_shift", "t_E", "t_F", "t_S", "has_stress", "has_energy", "force_weight", "n_atoms",
}


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# --------------------------------------------------------------------------
# Synthetic structures
# --------------------------------------------------------------------------
def _labelled(atoms, seed):
    rng = np.random.default_rng(seed)
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=float(rng.normal(-5.0, 0.5)),
        forces=rng.normal(0.0, 0.1, size=(len(atoms), 3)),
    )
    return atoms


def water(seed=0, shift=0.0):
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.757 + shift, 0.586, 0.0], [-0.757, 0.586 - shift, 0.05]],
    )
    atoms = _labelled(atoms, seed)
    atoms.calc.results["dipole"] = np.array([0.02, 0.39, -0.01])      # e*A
    atoms.info["polarizability"] = np.diag([1.53, 1.42, 1.47]).ravel()  # A^3
    return atoms


def hydroxide(seed=1):
    atoms = _labelled(Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.97, 0.05, 0.02]]), seed)
    atoms.info["charge"] = -1
    atoms.calc.results["dipole"] = np.array([-0.3, 0.01, 0.0])
    atoms.info["polarizability"] = np.diag([2.1, 1.8, 1.8]).ravel()
    return atoms


def water_hydroxide_pair(seed=2):
    """Two fragments, 5.5 A apart, charges 0 and -1: a per-fragment constraint."""
    atoms = _labelled(
        Atoms(
            "OH2OH",
            positions=[
                [0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.05],
                [5.5, 0.3, 0.1], [6.45, 0.4, 0.2],
            ],
        ),
        seed,
    )
    atoms.arrays["fragment_id"] = np.array([0, 0, 0, 1, 1])
    atoms.info["fragment_charges"] = np.array([0.0, -1.0])
    atoms.calc.results["dipole"] = np.array([-5.0, -0.2, -0.1])
    return atoms


def rocksalt(seed=3, displacement=0.1):
    atoms = _labelled(
        Atoms(
            "NaCl",
            positions=[[0.0, 0.0, 0.0], [2.82 + displacement, 0.1, 0.05]],
            cell=[[0.0, 2.82, 2.82], [2.82, 0.0, 2.82], [2.82, 2.82, 0.0]],
            pbc=True,
        ),
        seed,
    )
    born = np.stack([np.diag([1.1, 1.1, 1.2]), -np.diag([1.1, 1.1, 1.2])])
    atoms.arrays["born_charges"] = born.reshape(2, 9)
    atoms.info["dielectric_tensor"] = np.diag([2.3, 2.3, 2.4]).ravel()
    return atoms


def _settings(**overrides):
    block = {"enabled": True, "heads": dict(HEADS)}
    block.update(overrides)
    return train.parse_electrostatics_config({"r_max": R_MAX, "electrostatics": block})


def _model(settings, seed=0):
    train.configure_default_dtype(settings)
    torch.manual_seed(seed)
    model = TransformersACE(**MODEL_KWARGS)
    train.attach_electrostatics_from_settings(model, settings)
    return model


# --------------------------------------------------------------------------
# 1. Config parsing and validation
# --------------------------------------------------------------------------
def test_absent_or_disabled_block_turns_nothing_on():
    assert train.parse_electrostatics_config({"r_max": R_MAX}) is None
    disabled = {"r_max": R_MAX, "electrostatics": {"enabled": False, "dipole_weight": 1.0}}
    assert train.parse_electrostatics_config(disabled) is None


@pytest.mark.parametrize(
    "block, match",
    [
        ({"enabled": True, "dipol_weight": 1.0}, "unknown key"),
        ({"enabled": True, "config": {"n_cg_iters": 6}}, "electrostatics.config: unknown"),
        ({"enabled": True, "heads": {"hiden_dim": 8}}, "electrostatics.heads: unknown"),
        ({"enabled": True, "heads": {"irreps_in": "8x0e"}}, "must not be set"),
        ({"dipole_weight": 1.0}, "enabled must be given explicitly"),
        ({"enabled": "yes"}, "enabled must be given explicitly"),
        ({"enabled": True, "heads": {"r_max": 5.0}}, "exceeds the model cutoff"),
        ({"enabled": True, "polarizability_weight": -1.0}, "finite number >= 0"),
        ({"enabled": True, "polarizability_route": "exact"}, "polarizability_route"),
        ({"enabled": True, "born_charge_method": "moments"}, "born_charge_method"),
        ({"enabled": True, "config": {"thole_a": 1.5}}, "thole_a"),
        # Typos are caught even while the block is switched off.
        ({"enabled": False, "config": {"n_cg_iters": 6}}, "electrostatics.config: unknown"),
        # SPEC 22: the through-bond coupling head.
        ({"enabled": True, "transfer_coupling": {"hoping_init": 1.0}}, "transfer_coupling: unknown"),
        ({"enabled": True, "transfer_coupling": {"r_max": 5.0}}, "exceeds the model cutoff"),
        ({"enabled": True, "transfer_coupling": 3}, "must be a mapping"),
        ({"enabled": True, "spin_key": "S"}, "spin_key"),
    ],
)
def test_invalid_blocks_are_refused_loudly(block, match):
    with pytest.raises(ValueError, match=match):
        train.parse_electrostatics_config({"r_max": R_MAX, "electrostatics": block})


def test_pass_through_fields_and_yaml_float_strings():
    text = """
r_max: 4.0
electrostatics:
  enabled: true
  config:
    constraint_tol: 1e-10        # PyYAML reads this as the STRING '1e-10'
    n_cg_iter: 8
  heads:
    hidden_dim: 16
    a_mu_floor: 1               # an int where the heads want a float
  dipole_weight: 2
  polarizability_route: minimiser
"""
    settings = train.parse_electrostatics_config(yaml.safe_load(text))
    assert settings.config.constraint_tol == 1e-10
    assert settings.config.n_cg_iter == 8
    assert settings.heads["hidden_dim"] == 16
    assert settings.heads["a_mu_floor"] == 1.0 and isinstance(settings.heads["a_mu_floor"], float)
    # The heads' bond cutoff defaults to the model cutoff.
    assert settings.heads["r_max"] == R_MAX
    assert settings.dipole_weight == 2.0 and settings.observable_losses
    assert settings.polarizability_route == "minimiser"


def test_amp_is_refused_with_electrostatics():
    config = {"r_max": R_MAX, "use_amp": True, "electrostatics": {"enabled": True}}
    with pytest.raises(ValueError, match="use_amp"):
        train.parse_electrostatics_config(config)
    # ...but not when the block is off.
    config["electrostatics"]["enabled"] = False
    assert train.parse_electrostatics_config(config) is None


# --------------------------------------------------------------------------
# 2. dtype: float64 only when enabled, float32 path unchanged otherwise
# --------------------------------------------------------------------------
def test_float32_pipeline_is_unchanged_without_electrostatics():
    torch.set_default_dtype(torch.float32)
    assert train.configure_default_dtype(None) is torch.float32
    atoms = rocksalt()
    item = train.AtomisticDataset([atoms], R_MAX)[0]
    assert set(item) == PLAIN_ITEM_KEYS
    for key in ("pos", "cell", "volume", "t_E", "t_F", "t_S", "edge_shift"):
        assert item[key].dtype is torch.float32, key
    assert torch.equal(item["pos"], torch.tensor(atoms.positions, dtype=torch.float32))
    assert torch.equal(item["t_F"], torch.tensor(atoms.get_forces(), dtype=torch.float32))
    assert set(train.AtomisticDataset.collate_batched([item, item])) == PLAIN_BATCH_KEYS
    assert train.atomic_energy_tensor({1: -13.6}, "cpu").dtype is torch.float32


def test_float64_is_enforced_when_enabled():
    settings = _settings()
    torch.set_default_dtype(torch.float32)
    # A model built before the dtype was set cannot be given electrostatics.
    stale = TransformersACE(**MODEL_KWARGS)
    with pytest.raises(RuntimeError, match="float64"):
        train.attach_electrostatics_from_settings(stale, settings)

    assert train.configure_default_dtype(settings) is torch.float64
    dataset = train.AtomisticDataset([water(), rocksalt()], R_MAX, electrostatics=True)
    for item in (dataset[0], dataset[1]):
        for key in ("pos", "cell", "t_E", "t_F", "edge_shift", "t_dipole", "t_born"):
            assert item[key].dtype is torch.float64, key
    model = _model(settings)
    assert next(model.parameters()).dtype is torch.float64
    assert model.electrostatics.heads.irreps_in == model.attention_irreps


# --------------------------------------------------------------------------
# 3. Batched charged systems
# --------------------------------------------------------------------------
def _batch_structures():
    return [water(), hydroxide(), water_hydroxide_pair(), rocksalt()]


def test_batched_matches_single_structure_for_charged_and_periodic_mix():
    settings = _settings()
    model = _model(settings).eval()
    dataset = train.AtomisticDataset(_batch_structures(), R_MAX, electrostatics=True)
    items = [dataset[i] for i in range(len(dataset))]

    single = [
        model(it, training=False, compute_stress=True, return_electrostatics=True)
        for it in items
    ]
    bat = train.AtomisticDataset.collate_batched(items)
    E, F, S, aux = model.forward_batched(
        bat, training=False, compute_stress=True, return_electrostatics=True
    )
    E_plain, F_plain, _, _ = model.forward_batched(bat, training=False, compute_stress=False)

    ref_E = torch.stack([s[0] for s in single])
    ref_F = torch.cat([s[1] for s in single])
    assert float((E - ref_E).abs().max()) < 1e-10
    assert float((F - ref_F).abs().max()) < 1e-10
    assert float((E_plain - ref_E).abs().max()) < 1e-10
    assert float((F_plain - ref_F).abs().max()) < 1e-10
    assert float((S[3] - single[3][2]).abs().max()) < 1e-10  # the periodic cell

    # The electrostatic energy is genuinely part of E (not a zero term).
    assert all(abs(float(o.energy)) > 1e-6 for o in aux["electrostatics"])

    # Born charges from the batched outputs are the COMPLETE derivative: equal to
    # the single-structure ones, which differentiate the one tensor that h, the
    # bond geometry and the assembly all descend from.
    for s in (1, 3):
        batched = train._born_charges(aux["electrostatics"][s].state().assembly, "energy")
        alone = train._born_charges(single[s][3]["electrostatics"][0].state().assembly, "energy")
        assert float((batched - alone).abs().max()) < 1e-10
        # sum_i Z*_i = Q_total * I (a translation identity; -1 for hydroxide).
        total = aux["electrostatics"][s].total_charge
        assert float((batched.sum(0) - total * torch.eye(3)).abs().max()) < 1e-8


def _open_batch():
    """Open structures only: neutral, anion, an explicit two-fragment pair, neutral."""
    return [water(0), hydroxide(1), water_hydroxide_pair(2), water(3, shift=0.05)]


@pytest.mark.parametrize("cap", [None, 4])     # one joint solve / several chunks
def test_block_diagonal_batch_equals_the_per_structure_loop(cap, monkeypatch):
    """SPEC 24 through the model: one block-diagonal solve per batch gives, for
    every structure, the energy, forces, charges, dipoles, bond transfers and the
    PARAMETER gradient of an energy + force training loss of the per-structure
    loop, and an observable rebuilt through state() agrees too.  The route is
    verified to be taken (it is not silently skipped)."""
    import flashace.model

    monkeypatch.setattr(flashace.model, "BATCHED_ELECTROSTATICS_MAX_ATOMS", cap)
    settings = _settings()
    dataset = train.AtomisticDataset(_open_batch(), R_MAX, electrostatics=True)
    items = [dataset[i] for i in range(len(dataset))]
    bat = train.AtomisticDataset.collate_batched(items)

    results = {}
    for fast in (False, True):
        model = _model(settings, seed=0).train()
        model.batched_electrostatics_solve = fast
        calls = []
        original = model.electrostatics.energies_by_structure

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        model.electrostatics.energies_by_structure = counted
        model.zero_grad(set_to_none=True)
        E, F, _, aux = model.forward_batched(
            bat, training=True, compute_stress=False, return_electrostatics=True
        )
        target = torch.linspace(-1.0, 1.0, len(items), dtype=E.dtype)
        loss = ((E - target) ** 2).sum() + 10.0 * (F ** 2).sum()
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        outputs = aux["electrostatics"]
        alpha = train.polarizability_of(outputs[0]) if hasattr(train, "polarizability_of") else None
        results[fast] = (E.detach(), F.detach(), outputs, grads, len(calls), alpha)

    (E0, F0, out0, g0, n0, _), (E1, F1, out1, g1, n1, _) = results[False], results[True]
    assert n0 == 0 and n1 == (1 if cap is None else 4)   # loop vs joint solve(s)
    assert float((E0 - E1).abs().max()) < 1e-10
    assert float((F0 - F1).abs().max()) < 1e-9
    for a, b in zip(out0, out1):
        assert float((a.energy - b.energy).abs()) < 1e-10
        for name in ("q", "mu", "rho", "t"):
            x, y = getattr(a.solve_result, name), getattr(b.solve_result, name)
            assert x.shape == y.shape and float((x - y).abs().max()) < 1e-9, name
        assert a.total_charge == b.total_charge
    assert set(g0) == set(g1)
    scale = max(float(g.abs().max()) for g in g0.values())
    worst = max(float((g0[n] - g1[n]).abs().max()) for n in g0)
    assert worst <= 1e-7 * scale, worst
    # An observable through the lazy per-structure rebuild (verifies the energy).
    from trace_electrostatics import polarizability

    for s in (0, 1):
        a = polarizability(out0[s].state().solve_result, out0[s].state().assembly)
        b = polarizability(out1[s].state().solve_result, out1[s].state().assembly)
        assert float((a - b).abs().max()) < 1e-9


def test_structure_atom_lists_match_nonzero_for_any_layout():
    """The contiguous fast path and the general fallback list the same atoms."""
    from flashace.model import _structure_atom_lists

    for batch in (torch.tensor([0, 0, 1, 1, 1, 2, 3, 3]), torch.tensor([1, 0, 2, 0, 1, 3, 3, 2])):
        expected = [(batch == s).nonzero(as_tuple=False).view(-1) for s in range(4)]
        got = _structure_atom_lists(batch, 4)
        assert all(torch.equal(a, b) for a, b in zip(expected, got))


def test_block_diagonal_route_is_skipped_for_what_it_does_not_carry():
    """A periodic structure in the batch sends the whole batch through the loop."""
    settings = _settings()
    model = _model(settings).eval()
    dataset = train.AtomisticDataset(_open_batch() + [rocksalt(4)], R_MAX, electrostatics=True)
    bat = train.AtomisticDataset.collate_batched([dataset[i] for i in range(len(dataset))])
    calls = []
    original = model.electrostatics.energies_by_structure
    model.electrostatics.energies_by_structure = lambda *a, **k: calls.append(1) or original(*a, **k)
    model.forward_batched(bat, training=False, compute_stress=False)
    assert calls == []


def test_per_structure_charges_land_in_the_right_structure():
    settings = _settings()
    model = _model(settings).eval()
    # Charged structures in two different positions of the batch, plus a
    # two-fragment structure whose charge must sit on its second fragment.
    structures = [hydroxide(), water(), water_hydroxide_pair(), rocksalt(), hydroxide(seed=9)]
    dataset = train.AtomisticDataset(structures, R_MAX, electrostatics=True)
    items = [dataset[i] for i in range(len(dataset))]
    bat = train.AtomisticDataset.collate_batched(items)
    assert bat["total_charge"].tolist() == [-1.0, 0.0, -1.0, 0.0, -1.0]
    assert bat["n_fragments"].tolist() == [0, 0, 2, 0, 0]

    _, _, _, aux = model.forward_batched(bat, training=False, return_electrostatics=True)
    sums = [float(o.solve_result.q.sum()) for o in aux["electrostatics"]]
    np.testing.assert_allclose(sums, [-1.0, 0.0, -1.0, 0.0, -1.0], atol=1e-9)
    q_pair = aux["electrostatics"][2].solve_result.q
    np.testing.assert_allclose(
        [float(q_pair[:3].sum()), float(q_pair[3:].sum())], [0.0, -1.0], atol=1e-9
    )
    # Charge actually moved: the charged fragment is not a uniform smear.
    assert float(q_pair[3:].std()) > 1e-6

    # A single non-zero charge for a whole batch is ambiguous and refused.
    ambiguous = dict(bat)
    ambiguous["total_charge"] = -1.0
    with pytest.raises(ValueError, match="does not say which structure"):
        model.forward_batched(ambiguous, training=False)
    # Fragment labels without their per-structure counts are refused.
    orphan = dict(bat)
    del orphan["n_fragments"]
    with pytest.raises(ValueError, match="n_fragments"):
        model.forward_batched(orphan, training=False)


def test_spin_multiplicity_reaches_the_spin_constraint():
    """multiplicity 2S+1 -> spin target S -> sum_i s_i = 2S unpaired electrons."""
    settings = _settings(heads=dict(HEADS, spin=True))
    model = _model(settings).eval()
    radical = _labelled(Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.97, 0.05, 0.02]]), 4)
    radical.info["multiplicity"] = 2          # 9 electrons: a doublet
    anion = hydroxide()
    anion.info["multiplicity"] = 1            # 10 electrons: a singlet
    dataset = train.AtomisticDataset([radical, anion], R_MAX, electrostatics=True, spin=True)
    bat = train.AtomisticDataset.collate_batched([dataset[0], dataset[1]])
    _, _, _, aux = model.forward_batched(bat, training=False, return_electrostatics=True)
    spins = [float(o.solve_result.s.sum()) for o in aux["electrostatics"]]
    np.testing.assert_allclose(spins, [1.0, 0.0], atol=1e-9)

    radical.info["multiplicity"] = 1          # an odd-electron singlet is impossible
    with pytest.raises(ValueError, match="parity"):
        train.AtomisticDataset([radical], R_MAX, electrostatics=True, spin=True)
    radical.info.pop("multiplicity")
    radical.info["spin"] = 0.5                # ambiguous key: not read
    with pytest.raises(ValueError, match="multiplicity"):
        train.AtomisticDataset([radical], R_MAX, electrostatics=True, spin=True)


# --------------------------------------------------------------------------
# Data keys
# --------------------------------------------------------------------------
def test_label_parsing_and_refusals():
    labels = train.electrostatic_labels_from_atoms(water_hydroxide_pair())
    assert labels["total_charge"] == -1.0
    assert labels["fragment_charges"].tolist() == [0.0, -1.0]

    periodic = rocksalt()
    periodic.info["dipole"] = [0.1, 0.0, 0.0]
    assert train.electrostatic_labels_from_atoms(periodic)["dipole"] is None
    assert train.electrostatic_labels_from_atoms(periodic)["ignored"] == ("dipole",)

    bad = water_hydroxide_pair()
    bad.info["fragment_charges"] = np.array([0.0, 1.0])
    bad.info["charge"] = -1
    with pytest.raises(ValueError, match="sums to"):
        train.electrostatic_labels_from_atoms(bad)
    guess = water_hydroxide_pair()
    del guess.info["fragment_charges"]
    guess.info["charge"] = -1
    with pytest.raises(ValueError, match="guess"):
        train.electrostatic_labels_from_atoms(guess)
    scalar = water()
    scalar.info["polarizability"] = 1.5
    with pytest.raises(ValueError, match="isotropic"):
        train.electrostatic_labels_from_atoms(scalar)
    # eps_inf is a bulk property: kept on a 3-D periodic cell, ignored elsewhere.
    assert train.electrostatic_labels_from_atoms(rocksalt())["dielectric_tensor"].shape == (3, 3)
    molecule = water()
    molecule.info["dielectric_tensor"] = np.eye(3).ravel() * 2.0
    parsed = train.electrostatic_labels_from_atoms(molecule)
    assert parsed["dielectric_tensor"] is None and parsed["ignored"] == ("dielectric_tensor",)


def test_partial_charge_labels_are_read_from_the_named_key_and_checked():
    """Partial charges come from atoms.arrays[key]; a label that cannot satisfy
    sum_i q_i = Q (the constraint the solve enforces exactly) is refused."""
    ion = hydroxide()
    ion.arrays["hirshfeld_charges"] = np.array([-1.25, 0.25])
    assert train.electrostatic_labels_from_atoms(ion)["partial_charges"] is None   # default key absent
    read = train.electrostatic_labels_from_atoms(ion, partial_charge_key="hirshfeld_charges")
    assert read["partial_charges"].tolist() == [-1.25, 0.25]
    assert train.electrostatic_labels_from_atoms(ion, partial_charge_key=None)["partial_charges"] is None

    wrong_sum = hydroxide()
    wrong_sum.arrays["partial_charges"] = np.array([-0.5, 0.25])      # sums to -0.25, Q = -1
    with pytest.raises(ValueError, match="sum to"):
        train.electrostatic_labels_from_atoms(wrong_sum)
    wrong_shape = water()
    wrong_shape.arrays["partial_charges"] = np.zeros((3, 2))
    with pytest.raises(ValueError, match="shape"):
        train.electrostatic_labels_from_atoms(wrong_shape)
    not_finite = water()
    not_finite.arrays["partial_charges"] = np.array([np.nan, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        train.electrostatic_labels_from_atoms(not_finite)
    settings = _settings(partial_charge_weight=1.0, partial_charge_key="hirshfeld_charges")
    assert settings.partial_charge_key == "hirshfeld_charges" and settings.observable_losses
    with pytest.raises(ValueError, match="partial_charge_key"):
        _settings(partial_charge_key="")


def test_partial_charge_error_is_the_solved_charge_minus_the_label_atom_by_atom():
    """In a batched graph, the error of every atom is q_solve - q_ref of THAT atom.

    Labels are set to the model's own solved charges, so every error must be
    zero to rounding; then a charge-conserving shift of delta from one atom to
    another in the second structure must appear as exactly -delta / +delta on
    those two atoms and nowhere else.  A permutation, an offset between
    structures or a stale label would break one of the two checks."""
    settings = _settings(partial_charge_weight=1.0)
    model = _model(settings).eval()
    frames = [water(), hydroxide(), water(seed=7, shift=0.04)]
    for frame in frames:
        frame.arrays["partial_charges"] = np.zeros(len(frame)) + frame.info.get("charge", 0) / len(frame)
    dataset = train.AtomisticDataset(frames, R_MAX, electrostatics=True)
    items = [dataset[i] for i in range(len(dataset))]
    batch = train.AtomisticDataset.collate_batched(items)
    _, _, _, aux = model.forward_batched(batch, training=False, return_electrostatics=True)
    solved = [out.solve_result.q.detach().clone() for out in aux["electrostatics"]]
    for item, q in zip(items, solved):
        item["t_partial_charges"] = q.clone()
    errors = train._electrostatic_errors(aux["electrostatics"], items, settings)["partial_charges"]
    scale = max(float(q.abs().max()) for q in solved)
    for error in errors:
        assert float(error.abs().max()) <= 64 * torch.finfo(torch.float64).eps * max(scale, 1.0)

    delta = 0.125
    items[1]["t_partial_charges"][0] -= delta
    items[1]["t_partial_charges"][1] += delta
    errors = train._electrostatic_errors(aux["electrostatics"], items, settings)["partial_charges"]
    tol = 64 * torch.finfo(torch.float64).eps * max(scale, 1.0)
    assert float(errors[0].abs().max()) <= tol and float(errors[2].abs().max()) <= tol
    assert abs(float(errors[1][0]) - delta) <= tol and abs(float(errors[1][1]) + delta) <= tol


def test_the_dielectric_label_survives_an_extxyz_round_trip(tmp_path):
    """ASE's extxyz reader moves info['dielectric_tensor'] -- a calculator
    property name -- into atoms.calc.results; the label must still be found."""
    from ase.io import read

    write(tmp_path / "crystal.extxyz", [rocksalt()])
    back = read(tmp_path / "crystal.extxyz")
    assert "dielectric_tensor" not in back.info               # where ASE put it instead:
    assert "dielectric_tensor" in back.calc.results
    label = train.electrostatic_labels_from_atoms(back)["dielectric_tensor"]
    np.testing.assert_allclose(label, np.diag([2.3, 2.3, 2.4]))


def test_random_rotation_rotates_every_electrostatic_target(monkeypatch):
    atoms = water()
    atoms.arrays["born_charges"] = np.random.default_rng(0).normal(size=(3, 9))
    torch.manual_seed(3)
    rot = train.o3.rand_matrix()
    monkeypatch.setattr(train.o3, "rand_matrix", lambda: rot.clone())
    rotated = train.AtomisticDataset([atoms], R_MAX, random_rotation=True, electrostatics=True)[0]
    plain = train.AtomisticDataset([atoms], R_MAX, electrostatics=True)[0]
    # pos -> pos R^T, so p -> R p, alpha -> R alpha R^T, Z_i -> R Z_i R^T.
    assert torch.allclose(rotated["pos"], plain["pos"] @ rot.T, atol=1e-12)
    assert torch.allclose(rotated["t_dipole"], rot @ plain["t_dipole"], atol=1e-12)
    assert torch.allclose(rotated["t_polarizability"], rot @ plain["t_polarizability"] @ rot.T, atol=1e-12)
    assert torch.allclose(rotated["t_born"], rot @ plain["t_born"] @ rot.T, atol=1e-12)
    crystal = rocksalt()
    crystal.info["dielectric_tensor"] = np.array([[2.3, 0.1, 0.0], [0.1, 2.5, 0.2], [0.0, 0.2, 2.4]]).ravel()
    rotated = train.AtomisticDataset([crystal], R_MAX, random_rotation=True, electrostatics=True)[0]
    plain = train.AtomisticDataset([crystal], R_MAX, electrostatics=True)[0]
    assert torch.allclose(rotated["t_dielectric"], rot @ plain["t_dielectric"] @ rot.T, atol=1e-12)


# --------------------------------------------------------------------------
# 4. Loss terms
# --------------------------------------------------------------------------
def test_observables_are_in_the_documented_units():
    """Dipole = sum_i (q_i r_i + mu_i) in e*A; alpha converted to A^3 by K_E."""
    from trace_electrostatics import K_E, polarizability

    settings = _settings(dipole_weight=1.0, polarizability_weight=1.0)
    model = _model(settings).eval()
    item = train.AtomisticDataset([water()], R_MAX, electrostatics=True)[0]
    item["t_dipole"] = torch.zeros(3)
    item["t_polarizability"] = torch.zeros(3, 3)
    _, _, _, aux = model(item, training=False, return_electrostatics=True)
    errors = train._electrostatic_errors(aux["electrostatics"], [item], settings)
    output = aux["electrostatics"][0]
    solve = output.solve_result
    manual = (solve.q[:, None] * output.positions).sum(0) + solve.mu.sum(0)
    assert torch.allclose(errors["dipole"][0], manual, atol=1e-12)
    state = output.state()
    response = polarizability(state.solve_result, state.assembly)  # e^2 A^2 / eV
    assert torch.allclose(errors["polarizability"][0], K_E * response, rtol=1e-10)
    # A physical polarizability: positive definite volume of order 1 A^3.
    eigenvalues = torch.linalg.eigvalsh(0.5 * (errors["polarizability"][0] + errors["polarizability"][0].T))
    assert float(eigenvalues.min()) > 0.0


def test_the_dielectric_tensor_is_eps_inf_of_the_cell():
    """eps_inf = I + 4 pi K_E alpha_cell / Omega, dimensionless, > 1 on its axes."""
    from trace_electrostatics import K_E, polarizability

    settings = _settings(dielectric_weight=1.0)
    model = _model(settings).eval()
    crystal = rocksalt()
    item = train.AtomisticDataset([crystal], R_MAX, electrostatics=True)[0]
    item["t_dielectric"] = torch.zeros(3, 3)
    _, _, _, aux = model(item, training=False, return_electrostatics=True)
    errors = train._electrostatic_errors(aux["electrostatics"], [item], settings)
    state = aux["electrostatics"][0].state()
    alpha = polarizability(state.solve_result, state.assembly)
    volume = abs(float(np.linalg.det(crystal.cell.array)))
    expected = torch.eye(3) + 4.0 * math.pi * K_E * alpha / volume
    assert torch.allclose(errors["dielectric_tensor"][0], expected, rtol=1e-10, atol=1e-12)
    assert float(torch.linalg.eigvalsh(0.5 * (expected + expected.T)).min()) > 1.0


def _head_gradients(model):
    """|grad| summed per head sub-module, the seed head excluded (its own loss)."""
    norms = {}
    for name, parameter in model.electrostatics.heads.named_parameters():
        if "solver_seed" in name:
            continue
        block = name.split(".")[0]
        grad = 0.0 if parameter.grad is None else float(parameter.grad.abs().sum())
        norms[block] = norms.get(block, 0.0) + grad
    return norms


def with_partial_charges(atoms, values):
    """A structure carrying reference partial charges (e) under the default key."""
    atoms.arrays["partial_charges"] = np.asarray(values, dtype=float)
    return atoms


@pytest.mark.parametrize(
    "observable, structures",
    [
        ("dipole", lambda: [water(), hydroxide(), water_hydroxide_pair()]),
        ("polarizability", lambda: [water(), hydroxide()]),
        ("born_charges", lambda: [rocksalt()]),
        ("dielectric_tensor", lambda: [rocksalt(), rocksalt(seed=4, displacement=0.2)]),
        ("partial_charges", lambda: [with_partial_charges(water(), [-0.7, 0.35, 0.35]),
                                     with_partial_charges(hydroxide(), [-1.3, 0.3]),
                                     with_partial_charges(rocksalt(), [0.8, -0.8])]),
    ],
)
def test_each_observable_loss_trains_the_heads(observable, structures):
    weights = {"dipole_weight": 0.0, "polarizability_weight": 0.0, "born_charge_weight": 0.0,
               "dielectric_weight": 0.0, "partial_charge_weight": 0.0}
    key = {"dipole": "dipole_weight", "polarizability": "polarizability_weight",
           "born_charges": "born_charge_weight", "dielectric_tensor": "dielectric_weight",
           "partial_charges": "partial_charge_weight"}[observable]
    model = _model(_settings(**weights)).train()
    dataset = train.AtomisticDataset(structures(), R_MAX, electrostatics=True)
    items = [dataset[i] for i in range(len(dataset))]
    zero = lambda z: torch.zeros(())  # noqa: E731

    def step(settings):
        model.zero_grad(set_to_none=True)
        metrics = train.ElectrostaticMetricTracker()
        loss = train._batched_train_step(
            model, items, "cpu", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, zero,
            train.MetricTracker(), train._make_grad_scaler(False), len(items),
            electrostatics=settings, electrostatic_metrics=metrics,
        )
        return loss, metrics.rmse(), _head_gradients(model)

    # Energy, force and stress weights are zero: without the observable loss
    # nothing but the seed head (excluded) can receive a gradient...
    _, _, baseline = step(_settings(**weights))
    assert all(value == 0.0 for value in baseline.values()), baseline
    # ...and with it the physics heads do.
    loss, rmse, grads = step(_settings(**dict(weights, **{key: 1.0})))
    assert math.isfinite(float(loss)) and float(loss) > 0.0
    assert math.isfinite(rmse[observable])
    assert grads["scalar_net"] > 0.0, grads          # chi, eta, sigma, ell
    if observable in ("polarizability", "dielectric_tensor"):
        assert grads["a_mu_linear"] > 0.0, grads     # the dipole stiffness A_mu


# --------------------------------------------------------------------------
# End to end: train.main on a synthetic dataset
# --------------------------------------------------------------------------
def _write_config(tmp_path, batched, electrostatics=True):
    train_frames = [water(0), hydroxide(1), water_hydroxide_pair(2), rocksalt(3)]
    valid_frames = [water(5, shift=0.03), rocksalt(6, displacement=0.15)]
    reference_charges = {"H2O": [-0.7, 0.35, 0.35], "HO": [-1.3, 0.3],
                         "H3O2": [-0.7, 0.35, 0.35, -1.3, 0.3], "ClNa": [0.8, -0.8]}
    for frame in train_frames + valid_frames:
        if "born_charges" in frame.arrays:
            frame.arrays["born_charges"] = frame.arrays["born_charges"].reshape(len(frame), 9)
        frame.arrays["partial_charges"] = np.array(reference_charges[frame.get_chemical_formula()])
    write(tmp_path / "train.extxyz", train_frames)
    write(tmp_path / "valid.extxyz", valid_frames)
    config = {
        "device": "cpu",
        "seed": 0,
        "train_file": str(tmp_path / "train.extxyz"),
        "valid_file": str(tmp_path / "valid.extxyz"),
        "architecture_version": 2,
        **MODEL_KWARGS,
        "batch_size": 4,           # the whole training set: one step per epoch
        "epochs": 3,               # -> three optimiser steps
        "learning_rate": 1e-3,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "num_workers": 0,
        "batched_graphs": batched,
        "model_save_path": str(tmp_path / "model.pt"),
        "save_last_checkpoint": False,
        "plot_training_curves": False,
    }
    if electrostatics:
        config["electrostatics"] = {
            "enabled": True,
            "heads": dict(HEADS),
            "dipole_weight": 1.0,
            "polarizability_weight": 0.1,
            "born_charge_weight": 0.1,
            "dielectric_weight": 0.1,
            "partial_charge_weight": 0.1,
        }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("batched", [False, True])
def test_three_training_steps_have_finite_losses(tmp_path, batched):
    path = _write_config(tmp_path, batched)
    history = train.main(["--config", str(path)])
    assert torch.get_default_dtype() is torch.float64
    assert len(history["epoch"]) == 3
    for key, values in history.items():
        if key == "epoch":
            continue
        assert all(math.isfinite(v) for v in values), (key, values)
    for name in ("dipole", "polarizability", "born_charges", "dielectric_tensor", "partial_charges"):
        assert history[f"train_{name}_rmse"][-1] > 0.0
    assert history["val_born_charges_rmse"][-1] > 0.0
    assert history["val_dielectric_tensor_rmse"][-1] > 0.0
    checkpoint = torch.load(tmp_path / "model.pt", map_location="cpu")
    assert checkpoint["config"]["electrostatics"]["heads"]["r_max"] == R_MAX
    assert any(k.startswith("electrostatics.heads.") for k in checkpoint["model_state_dict"])


def test_training_without_electrostatics_stays_float32(tmp_path):
    torch.set_default_dtype(torch.float32)
    path = _write_config(tmp_path, batched=False, electrostatics=False)
    history = train.main(["--config", str(path)])
    assert torch.get_default_dtype() is torch.float32
    assert not any("dipole" in key for key in history)
    checkpoint = torch.load(tmp_path / "model.pt", map_location="cpu")
    assert "electrostatics" not in checkpoint["config"]
    assert all(v.dtype is torch.float32 for v in checkpoint["model_state_dict"].values() if v.is_floating_point())


# --------------------------------------------------------------------------
# Inference: a trained checkpoint in the ASE calculator; export refused
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory):
    """A checkpoint written by the real training loop (3 steps, electrostatics on)."""
    directory = tmp_path_factory.mktemp("trained")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train.main(["--config", str(_write_config(directory, batched=False))])
    return directory / "model.pt"


def _calculator(path):
    from flashace.calculator import TransformersACECalculator

    return TransformersACECalculator(model_path=str(path), device="cpu")


def test_trained_checkpoint_loads_in_the_calculator(trained_checkpoint):
    torch.set_default_dtype(torch.float32)   # as in a fresh inference process
    calc = _calculator(trained_checkpoint)
    # float64 was set before the model and heads were built, not cast afterwards.
    assert torch.get_default_dtype() is torch.float64
    assert calc.dtype is torch.float64
    assert calc.model.electrostatics is calc.electrostatics
    assert calc.model.electrostatics.heads.irreps_in == calc.model.attention_irreps
    assert calc.model.electrostatics.heads.r_max == R_MAX
    # strict load: the trained head parameters, not a fresh initialisation.
    saved = torch.load(trained_checkpoint, map_location="cpu")["model_state_dict"]
    for name, value in calc.model.state_dict().items():
        assert torch.equal(value, saved[name]), name


@pytest.mark.parametrize(
    "structure", [water, hydroxide, water_hydroxide_pair, rocksalt],
    ids=["neutral", "charged", "fragments", "periodic"],
)
def test_calculator_matches_the_model_forward(trained_checkpoint, structure):
    calc = _calculator(trained_checkpoint)
    atoms = structure()
    reference = structure()        # keeps its SinglePointCalculator for the dataset
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    # The training-pipeline forward on the same structure: same keys, same values.
    item = train.AtomisticDataset([reference], R_MAX, electrostatics=True)[0]
    periodic = bool(reference.pbc.any())
    E, F, S, _ = calc.model(item, training=False, compute_stress=periodic)
    baseline = calc.energy_shift_per_atom * len(reference)
    assert abs(energy - (float(E) + baseline)) < 1e-10
    assert np.abs(forces - F.detach().numpy()).max() < 1e-10
    if periodic:
        stress = atoms.get_stress()
        voigt = train.stress_to_voigt(S.detach()).numpy()
        assert np.abs(stress - voigt).max() < 1e-10


def test_calculator_hands_the_charge_state_to_the_solve(trained_checkpoint, monkeypatch):
    calc = _calculator(trained_checkpoint)
    module = calc.model.electrostatics
    solves = []
    original = module.forward

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        solves.append(result[1])
        return result

    monkeypatch.setattr(module, "forward", spy)

    atoms = hydroxide()
    atoms.calc = calc
    charged = atoms.get_potential_energy()
    assert float(solves[-1].q.sum()) == pytest.approx(-1.0, abs=1e-9)

    # Same geometry, charge changed in info only: ASE's own cache would not
    # notice, the calculator must, and the energy must change.
    atoms.info["charge"] = 0
    neutral = atoms.get_potential_energy()
    assert float(solves[-1].q.sum()) == pytest.approx(0.0, abs=1e-9)
    assert abs(charged - neutral) > 1e-6

    pair = water_hydroxide_pair()
    pair.calc = calc
    pair.get_potential_energy()
    q = solves[-1].q
    np.testing.assert_allclose([float(q[:3].sum()), float(q[3:].sum())], [0.0, -1.0], atol=1e-9)


def test_export_refuses_a_model_with_electrostatics(trained_checkpoint, tmp_path):
    from transformers_ace.aot_deploy import compile_lammps_aot_model
    from transformers_ace.deploy import LAMMPSEnergyModel, export_lammps_model

    types = ["H", "O", "Na", "Cl"]
    with pytest.raises(NotImplementedError, match="electrostatic"):
        export_lammps_model(trained_checkpoint, tmp_path / "model.pt", types)
    with pytest.raises(NotImplementedError, match="electrostatic"):
        compile_lammps_aot_model(
            trained_checkpoint, tmp_path / "model.pt2", types,
            max_atoms=8, max_edges=64, device="cpu",
        )
    with pytest.raises(NotImplementedError, match="silently dropped"):
        LAMMPSEnergyModel(_calculator(trained_checkpoint).model)
    assert not any(tmp_path.iterdir())


def test_fragment_policy_initial_keeps_a_dissociating_ion_continuous(trained_checkpoint):
    """A charged molecule pulled apart through the bond cutoff.

    ``fragment_policy="auto"`` re-derives connected fragments at every call and
    raises once the charged molecule splits (the net charge has no fragment).
    ``"initial"`` fixes the fragments from the first structure: the net charge
    stays on the union, the call does not raise, and the energy is continuous
    across the split (the dissociating bond's transfer vanishes with its
    envelope).
    """
    from flashace.calculator import TransformersACECalculator

    def molecule(separation):
        atoms = water(seed=5)
        atoms.info["charge"] = 1.0
        atoms.info.pop("total_charge", None)
        positions = atoms.get_positions()
        positions[1] = positions[0] + (positions[1] - positions[0]) / np.linalg.norm(
            positions[1] - positions[0]
        ) * separation
        atoms.set_positions(positions)
        atoms.calc = None
        for key in ("fragment_id",):
            atoms.arrays.pop(key, None)
        atoms.info.pop("fragment_charges", None)
        return atoms

    auto = TransformersACECalculator(model_path=str(trained_checkpoint), device="cpu")
    frozen = TransformersACECalculator(
        model_path=str(trained_checkpoint), device="cpu", fragment_policy="initial"
    )
    cutoff = float(frozen.electrostatics.heads.r_max)
    atoms = molecule(1.0)
    atoms.calc = frozen
    atoms.get_potential_energy()
    assert atoms.arrays["fragment_id"].tolist() == [0, 0, 0]
    energies, forces, positions = [], [], []
    for separation in (cutoff - 1e-6, cutoff + 1e-6, cutoff + 2.0):
        atoms.set_positions(molecule(separation).get_positions())
        energies.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())
        positions.append(atoms.get_positions())
    # Continuous across the cutoff: the step is the smooth change the forces
    # predict (-F . dR over the 2e-6 A move), with no jump on top of it.
    predicted = -float(np.sum(0.5 * (forces[0] + forces[1]) * (positions[1] - positions[0])))
    assert abs((energies[1] - energies[0]) - predicted) < 1e-10
    far = molecule(cutoff + 2.0)
    far.calc = auto
    with pytest.raises(ValueError):
        far.get_potential_energy()


def test_calculator_passes_field_metals_and_charge_states(trained_checkpoint):
    """atoms.info['external_field'], arrays['metallic'] / ['electrode_potential']
    and info['charge_states'] reach the solve, and changing them invalidates the
    cached result (trace_electrostatics SPEC 15, 20, 21)."""
    calc = _calculator(trained_checkpoint)
    atoms = water(seed=7)
    atoms.calc = calc
    e0 = atoms.get_potential_energy()
    atoms.info["external_field"] = [0.0, 0.2, 0.0]
    e_field = atoms.get_potential_energy()
    assert e_field != e0
    atoms.info.pop("external_field")
    atoms.arrays["metallic"] = np.array([True, True, True])
    e_metal = atoms.get_potential_energy()
    assert np.isfinite(e_metal) and e_metal != e0
    del atoms.arrays["metallic"]

    pair = water_hydroxide_pair(seed=8)
    pair.calc = calc
    n = len(pair)
    ids = [0] * 3 + [1] * (n - 3)
    pair.info.pop("charge", None)
    pair.info["total_charge"] = -1.0
    pair.info["charge_states"] = [
        {"fragment_id": ids, "fragment_charges": [0.0, -1.0]},
        {"fragment_id": ids, "fragment_charges": [-1.0, 0.0]},
    ]
    pair.info["charge_state_coupling"] = 0.05
    e_mixed = pair.get_potential_energy()
    pair.info["charge_state_coupling"] = 0.0
    e_uncoupled = pair.get_potential_energy()
    assert e_mixed < e_uncoupled           # coupling lowers the adiabatic ground state


def _pair_with_file_form_states(seed=8):
    """The water-hydroxide pair with two diabatic states in the extxyz form."""
    pair = water_hydroxide_pair(seed=seed)
    pair.info.pop("charge", None)
    pair.info.pop("fragment_charges", None)
    del pair.arrays["fragment_id"]
    pair.info["total_charge"] = -1.0
    pair.arrays["charge_state_fragment"] = np.array([0, 0, 0, 1, 1])
    pair.info["charge_state_charges"] = np.array([0.0, -1.0, -1.0, 0.0])
    pair.info["charge_state_coupling"] = 0.05
    return pair


def test_file_form_charge_states_survive_extxyz_and_equal_the_list_form(trained_checkpoint, tmp_path):
    """arrays['charge_state_fragment'] + info['charge_state_charges'] (SPEC 21 in
    a form an extxyz file can hold) round-trip through write/read, expand to
    exactly the in-memory list form, and give the calculator the same energy
    and forces bit for bit."""
    from flashace.charge_state import declared_charge_states

    pair = _pair_with_file_form_states()
    write(tmp_path / "pair.extxyz", pair)
    from ase.io import read
    reread = read(tmp_path / "pair.extxyz")
    states = declared_charge_states(reread)
    assert [s["fragment_charges"] for s in states] == [[0.0, -1.0], [-1.0, 0.0]]
    assert all(np.array_equal(s["fragment_id"], [0, 0, 0, 1, 1]) for s in states)
    # Row-major, one row per state: an asymmetric three-state declaration pins the order.
    three = _pair_with_file_form_states()
    three.info["charge_state_charges"] = np.array([0.0, -1.0, -1.0, 0.0, 1.0, -2.0])
    assert [s["fragment_charges"] for s in declared_charge_states(three)] == [
        [0.0, -1.0], [-1.0, 0.0], [1.0, -2.0]]

    listed = water_hydroxide_pair(seed=8)
    listed.info.pop("charge", None)
    listed.info.pop("fragment_charges", None)
    del listed.arrays["fragment_id"]
    listed.info["total_charge"] = -1.0
    listed.info["charge_states"] = [
        {"fragment_id": [0, 0, 0, 1, 1], "fragment_charges": [0.0, -1.0]},
        {"fragment_id": [0, 0, 0, 1, 1], "fragment_charges": [-1.0, 0.0]},
    ]
    listed.info["charge_state_coupling"] = 0.05
    energies, forces = [], []
    for atoms in (reread, listed):
        atoms.calc = _calculator(trained_checkpoint)
        energies.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())
    assert energies[0] == energies[1]
    np.testing.assert_array_equal(forces[0], forces[1])

    # The calculator cache must see a change of the declared charges.
    reread.info["charge_state_charges"] = np.array([0.0, -1.0, 1.0, -2.0])
    assert reread.get_potential_energy() != energies[0]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda a: a.info.__setitem__("charge_states", [{"fragment_id": [0] * 5, "fragment_charges": [-1.0]}]),
         "declared twice"),
        (lambda a: a.info.pop("charge_state_charges"), "together"),
        (lambda a: a.info.__setitem__("charge_state_charges", np.array([0.0, -1.0, -0.5, -0.5])), "integers"),
        (lambda a: a.info.__setitem__("charge_state_charges", np.array([0.0, 0.0, -1.0, 0.0])), "sum to the net"),
        (lambda a: a.info.__setitem__("charge_state_charges", np.array([0.0, -1.0])), "S >= 2"),
        (lambda a: a.arrays.__setitem__("charge_state_fragment", np.array([0, 0, 0, 2, 2])), "every label"),
        (lambda a: a.arrays.__setitem__("charge_state_fragment", np.zeros(5, dtype=int)), "two fragments"),
    ],
)
def test_malformed_charge_state_declarations_are_refused(mutate, match):
    from flashace.charge_state import declared_charge_states

    pair = _pair_with_file_form_states()
    mutate(pair)
    with pytest.raises(ValueError, match=match):
        declared_charge_states(pair)


def test_transfer_coupling_head_trains_saves_and_reloads(tmp_path):
    """SPEC 22 through the real pipeline: the YAML block builds the head, the
    checkpoint records it, the calculator rebuilds it with a STRICT load, and
    'through_bond' reaches the EVB solve."""
    from trace_electrostatics import TransferCouplingHead

    path = _write_config(tmp_path, batched=False)
    config = yaml.safe_load(path.read_text())
    config["electrostatics"]["transfer_coupling"] = {"gap_init": "2.0", "hidden_dim": 8}
    path.write_text(yaml.safe_dump(config))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train.main(["--config", str(path)])
    checkpoint = torch.load(tmp_path / "model.pt", map_location="cpu")
    saved = checkpoint["config"]["electrostatics"]["transfer_coupling"]
    assert saved == {"gap_init": 2.0, "hidden_dim": 8, "r_max": R_MAX}
    calc = _calculator(tmp_path / "model.pt")
    head = calc.model.electrostatics.transfer_coupling
    assert isinstance(head, TransferCouplingHead) and head.r_max == R_MAX
    state = checkpoint["model_state_dict"]
    assert any(name.startswith("electrostatics.transfer_coupling.") for name in state)
    for name, value in calc.model.state_dict().items():
        assert torch.equal(value, state[name]), name

    # Hydrogen-bonded contact (O...O 2.8 A), inside the 4 A tunnelling cutoff.
    # (The 5.5 A fixture pair has no path between its fragments: V = 0 exactly,
    # as it must.)
    pair = Atoms(
        "OH2OH",
        positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.05],
                   [2.8, 0.3, 0.1], [3.75, 0.4, 0.2]],
    )
    pair.calc = calc
    ids = [0] * 3 + [1] * (len(pair) - 3)
    pair.info["total_charge"] = -1.0
    pair.info["charge_states"] = [
        {"fragment_id": ids, "fragment_charges": [0.0, -1.0]},
        {"fragment_id": ids, "fragment_charges": [-1.0, 0.0]},
    ]
    pair.info["charge_state_coupling"] = 0.0
    e_uncoupled = pair.get_potential_energy()
    pair.info["charge_state_coupling"] = "through_bond"
    e_coupled = pair.get_potential_energy()
    forces = pair.get_forces()
    assert np.isfinite(e_coupled) and np.all(np.isfinite(forces))
    assert e_coupled < e_uncoupled


def test_observable_losses_rebuild_the_quadrupole_functional():
    """The state an observable is taken from is the functional the module
    evaluated, quadrupole sector included.  (Round 3 switched the module's
    quadrupole default on but left the rebuild's off: every observable loss on
    a quadrupole model failed its energy cross-check, -0.0033470 vs -0.0033566 eV.)"""
    heads = dict(HEADS, quadrupole_blocks=("A_Theta",), allow_truncated_a_theta=True)
    settings = _settings(dipole_weight=1.0, polarizability_weight=1.0, heads=heads)
    model = _model(settings).eval()
    item = train.AtomisticDataset([water()], R_MAX, electrostatics=True)[0]
    _, _, _, aux = model(item, training=False, return_electrostatics=True)
    output = aux["electrostatics"][0]
    state = output.state()
    assert state.assembly.params.quadrupole_resolved
    assert abs(float(state.energy - output.energy)) < 1e-12
    errors = train._electrostatic_errors(aux["electrostatics"], [item], settings)
    assert all(bool(torch.isfinite(errors[k][0]).all()) for k in ("dipole", "polarizability"))


def test_observables_are_refused_on_charge_state_structures():
    from trace_electrostatics import ElectrostaticModule  # noqa: F401  (module API)

    model = _model(_settings()).eval()
    pair = water_hydroxide_pair()
    item = train.AtomisticDataset([pair], R_MAX, electrostatics=True)[0]
    ids = item["fragment_id"]
    item["charge_states"] = [
        {"fragment_id": ids, "fragment_charges": [0.0, -1.0]},
        {"fragment_id": ids, "fragment_charges": [-1.0, 0.0]},
    ]
    item["charge_state_coupling"] = 0.05
    _, _, _, aux = model(item, training=False, return_electrostatics=True)
    with pytest.raises(ValueError, match="charge_states"):
        aux["electrostatics"][0].state()


def _environment_structures():
    """A field-driven water, a metallic Na4 cluster and an H-bonded donor-acceptor
    pair with two diabatic charge states, beside an ordinary charged structure."""
    field = water(seed=11)
    field.info["external_field"] = [0.0, 0.15, -0.05]                 # V/A
    metal = _labelled(Atoms("Na4", positions=[[0, 0, 0], [3.0, 0, 0], [0, 3.0, 0.1], [3.0, 3.0, 0]]), 12)
    metal.arrays["metallic"] = np.ones(4, dtype=bool)
    # In a field, so that metal (perfect screening) and dielectric differ: a
    # neutral cluster of identical atoms has almost nothing to equalise
    # without one (metal vs dielectric 1e-12 eV).
    metal.info["external_field"] = [0.1, 0.0, 0.0]
    pair = _labelled(Atoms("OH2OH", positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.05],
                                               [2.8, 0.3, 0.1], [3.75, 0.4, 0.2]]), 13)
    pair.info["charge"] = -1
    pair.arrays["fragment_id"] = np.array([0, 0, 0, 1, 1])
    pair.info["fragment_charges"] = np.array([0.0, -1.0])
    ids = [0, 0, 0, 1, 1]
    pair.info["charge_states"] = [
        {"fragment_id": ids, "fragment_charges": [0.0, -1.0]},
        {"fragment_id": ids, "fragment_charges": [-1.0, 0.0]},
    ]
    pair.info["charge_state_coupling"] = 0.05
    return [field, metal, pair, hydroxide()]


def test_fields_metals_and_charge_states_train_batched_exactly_as_single():
    settings = _settings()
    model = _model(settings).eval()
    dataset = train.AtomisticDataset(_environment_structures(), R_MAX, electrostatics=True)
    items = [dataset[i] for i in range(len(dataset))]
    assert torch.allclose(items[0]["external_field"], torch.tensor([0.0, 0.15, -0.05]))
    assert bool(items[1]["metallic"].all()) and len(items[2]["charge_states"]) == 2

    single = [model(it, training=False) for it in items]
    bat = train.AtomisticDataset.collate_batched(items)
    assert bat["has_external_field"].tolist() == [True, True, False, False]
    assert bat["charge_states"][0] is None and bat["charge_states"][2] is not None
    E, F, _, _ = model.forward_batched(bat, training=False)
    assert float((E - torch.stack([s[0] for s in single])).abs().max()) < 1e-10
    assert float((F - torch.cat([s[1] for s in single])).abs().max()) < 1e-10

    # Each environment is genuinely in the energy (not a dropped key).
    stripped = [dict(it) for it in items]
    del stripped[0]["external_field"]
    del stripped[1]["metallic"]
    del stripped[2]["charge_states"]
    for s in range(3):
        assert abs(float(model(stripped[s], training=False)[0] - single[s][0])) > 1e-6, s


def test_a_rotated_item_rotates_its_field(monkeypatch):
    atoms = _environment_structures()[0]
    torch.manual_seed(5)
    rot = train.o3.rand_matrix()
    monkeypatch.setattr(train.o3, "rand_matrix", lambda: rot.clone())
    rotated = train.AtomisticDataset([atoms], R_MAX, random_rotation=True, electrostatics=True)[0]
    plain = train.AtomisticDataset([atoms], R_MAX, electrostatics=True)[0]
    assert torch.allclose(rotated["external_field"], rot @ plain["external_field"], atol=1e-12)
    model = _model(_settings()).eval()
    # E(R x, R F) = E(x, F): rotating structure and field together changes nothing.
    assert abs(float(model(rotated, training=False)[0] - model(plain, training=False)[0])) < 1e-9


def test_a_training_step_on_environments_has_finite_loss_and_gradients():
    model = _model(_settings()).train()
    dataset = train.AtomisticDataset(_environment_structures(), R_MAX, electrostatics=True)
    items = [dataset[i] for i in range(len(dataset))]
    zero = lambda z: torch.zeros(())  # noqa: E731
    model.zero_grad(set_to_none=True)
    loss = train._batched_train_step(
        model, items, "cpu", 1.0, 1.0, 10.0, 0.0, 0.0, 0.0, zero,
        train.MetricTracker(), train._make_grad_scaler(False), len(items),
    )
    assert math.isfinite(float(loss))
    grads = [p.grad for p in model.electrostatics.heads.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0.0


def test_the_contact_law_trains_batched_exactly_as_single():
    """SPEC 23 through the pipeline: the dataset's atomic numbers reach the heads
    in both paths (data['z'] single, sliced per structure batched)."""
    settings = _settings(heads=dict(HEADS, charge_transfer_decay=3.0))
    model = _model(settings).eval()
    dataset = train.AtomisticDataset(_batch_structures(), R_MAX, electrostatics=True)
    items = [dataset[i] for i in range(len(dataset))]
    single = [model(it, training=False) for it in items]
    E, F, _, _ = model.forward_batched(train.AtomisticDataset.collate_batched(items), training=False)
    assert float((E - torch.stack([s[0] for s in single])).abs().max()) < 1e-10
    assert float((F - torch.cat([s[1] for s in single])).abs().max()) < 1e-10
    plain = _model(_settings(heads=dict(HEADS, charge_transfer_decay=None))).eval()
    assert abs(float(model(items[2], training=False)[0] - plain(items[2], training=False)[0])) > 1e-6



def test_new_runs_default_to_the_contact_law_and_can_turn_it_off():
    """SPEC 23 is the default for NEW runs, recorded explicitly so the checkpoint
    rebuilds the same functional; an explicit null switches it off."""
    assert _settings().heads["charge_transfer_decay"] == train.DEFAULT_CHARGE_TRANSFER_DECAY
    assert _settings(heads=dict(HEADS, charge_transfer_decay=None)).heads["charge_transfer_decay"] is None
    assert _settings(heads=dict(HEADS, charge_transfer_decay=2.0)).heads["charge_transfer_decay"] == 2.0


def _omol25_style(symbols, positions, charge, spin):
    atoms = _labelled(Atoms(symbols, positions=positions), 21)
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin          # OMol25 / fairchem: 2S+1, singlet = 1
    return atoms


def test_omol25_charge_and_spin_are_read_with_the_explicit_key():
    from flashace.charge_state import charge_state_changed, charge_state_from_atoms

    methyl = _omol25_style("CH3", [[0, 0, 0], [1.08, 0, 0], [-0.54, 0.94, 0], [-0.54, -0.94, 0]], 0, 2)
    hydroxide_anion = _omol25_style("OH", [[0, 0, 0], [0.97, 0, 0]], -1, 1)
    state = charge_state_from_atoms(methyl, spin=True, spin_key="spin")
    assert state["fragment_spins"].tolist() == [0.5] and state["multiplicity"] == 2.0
    state = charge_state_from_atoms(hydroxide_anion, spin=True, spin_key="spin")
    assert state["total_charge"] == -1.0 and state["fragment_spins"].tolist() == [0.0]
    # Not read by default: the key is ambiguous across datasets.
    with pytest.raises(ValueError, match="spin_key='spin'"):
        charge_state_from_atoms(methyl, spin=True)
    # A dataset storing 2S (not 2S+1) under 'spin' is caught by the parity of the
    # electron count: methyl has 9 electrons, so 2S = spin - 1 = 0 is impossible.
    wrong = _omol25_style("CH3", methyl.positions, 0, 1)
    with pytest.raises(ValueError, match="parity"):
        charge_state_from_atoms(wrong, spin=True, spin_key="spin")
    # The calculator's cache sees a spin-state change.
    triplet = methyl.copy()
    triplet.info["spin"] = 4
    assert charge_state_changed(methyl, triplet)


def test_open_shell_structures_are_counted_for_closed_shell_heads():
    settings = _settings(spin_key="spin")
    assert settings.spin_key == "spin"
    methyl = _omol25_style("CH3", [[0, 0, 0], [1.08, 0, 0], [-0.54, 0.94, 0], [-0.54, -0.94, 0]], 0, 2)
    dataset = train.AtomisticDataset([methyl, water()], R_MAX, electrostatics=True, spin_key="spin")
    assert dataset.electrostatic_label_counts()["open_shell"] == 1
    metadata = train.electrostatics_checkpoint_metadata(settings, _model(settings).electrostatics)
    assert metadata["spin_key"] == "spin" and metadata["heads"]["charge_transfer_decay"] == 3.0



def test_new_runs_choose_the_solver_route_per_structure():
    assert _settings().config.solver == "auto"
    assert _settings(config={"solver": "dense"}).config.solver == "dense"
    assert _settings(born_charge_method="minimiser").config.solver == "dense"
    assert _settings(polarizability_route="minimiser").config.solver == "dense"



def test_new_converged_runs_do_not_train_the_solver_seed():
    """Under the converged solve the seed changes only the iteration count."""
    assert _settings().config.solver_seed_loss_weight == 0.0
    assert _settings(config={"solve_mode": "fixed_k"}).config.solver_seed_loss_weight == 1.0
    assert _settings(config={"solver_seed_loss_weight": 0.5}).config.solver_seed_loss_weight == 0.5
