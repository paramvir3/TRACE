"""Structures without an energy label train on forces alone (2026-11-06, written red before the implementation): the
bulk-augmented phonon line adds Togo's displaced PBE MgO supercells, which carry VASP forces and no energies
(``has_energy=F`` in the extxyz header).  Contract: the dataset item carries ``has_energy`` (False when the structure has
no energy, or says so), ``t_E`` is then 0 and reading it does not raise; the collate carries the flag; the energy loss
of a batch is the sum over the structures that have an energy only; the metrics exclude them from the energy statistics
and keep them in the force statistics."""
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

import train as T


def _structures():
    a = Atoms("NaCl", positions=[[0, 0, 0], [0, 0, 2.4]], cell=np.eye(3) * 8.0, pbc=True)
    a.calc = SinglePointCalculator(a, energy=-3.0, forces=np.array([[0, 0, 0.1], [0, 0, -0.1]]))
    b = Atoms("NaCl", positions=[[0, 0, 0], [0, 0, 2.5]], cell=np.eye(3) * 8.0, pbc=True)
    b.calc = SinglePointCalculator(b, forces=np.array([[0, 0, 0.2], [0, 0, -0.2]]))     # forces only
    b.info["has_energy"] = "F"
    c = Atoms("NaCl", positions=[[0, 0, 0], [0, 0, 2.6]], cell=np.eye(3) * 8.0, pbc=True)
    c.calc = SinglePointCalculator(c, energy=-2.0, forces=np.array([[0, 0, 0.3], [0, 0, -0.3]]))
    c.info["has_energy"] = "F"                                                            # an energy present but declared unusable
    return [a, b, c]


def test_the_dataset_flags_structures_without_an_energy():
    ds = T.AtomisticDataset(_structures(), 4.0, random_rotation=False, dtype=torch.float64)
    items = [ds[i] for i in range(3)]
    assert [bool(it["has_energy"]) for it in items] == [True, False, False]
    assert float(items[0]["t_E"]) == -3.0 and float(items[1]["t_E"]) == 0.0 and float(items[2]["t_E"]) == 0.0
    bat = T.AtomisticDataset.collate_batched(items)
    assert bat["has_energy"].tolist() == [True, False, False]
    assert torch.equal(bat["t_F"], torch.cat([it["t_F"] for it in items]))


def test_the_energy_loss_sums_over_labelled_structures_only():
    E = torch.tensor([-2.5, 1.0, 5.0], dtype=torch.float64); target = torch.tensor([-3.0, 0.0, -2.0], dtype=torch.float64)
    n_at = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64); has_e = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    loss = T.energy_loss(E, target, n_at, has_e)
    assert float(loss) == pytest.approx(((-2.5 + 3.0) / 2.0) ** 2)                      # the unlabelled structures add nothing
    assert float(T.energy_loss(E, target, n_at, torch.ones_like(has_e))) == pytest.approx(sum(((E - target) / n_at) ** 2).item())


def test_the_metrics_exclude_unlabelled_energies_and_keep_their_forces():
    m = T.MetricTracker()
    p_F = torch.zeros(2, 3, dtype=torch.float64); t_F = torch.full((2, 3), 0.5, dtype=torch.float64)
    m.update(torch.tensor(-2.0), p_F, None, torch.tensor(-3.0), t_F, None, False, 2, has_energy=True)
    m.update(torch.tensor(7.0), p_F, None, torch.tensor(0.0), t_F, None, False, 2, has_energy=False)
    rmse_e_meV, force_rmse, _, _, force_mae, _, _ = m.get_metrics()
    assert rmse_e_meV == pytest.approx(500.0)                                           # |(-2 - -3)| / 2 atoms = 0.5 eV/atom, the labelled one alone
    assert force_mae == pytest.approx(0.5) and force_rmse == pytest.approx(0.5)         # both structures' forces count
    # the batched accumulator agrees with the per-structure one
    b = T.MetricTracker()
    b.update_batched(torch.tensor([-2.0, 7.0]), torch.zeros(4, 3, dtype=torch.float64), torch.zeros(2, 3, 3), torch.tensor([-3.0, 0.0]),
                     torch.full((4, 3), 0.5, dtype=torch.float64), torch.zeros(2, 3, 3), torch.zeros(2), torch.tensor([0, 0, 1, 1]),
                     torch.tensor([2, 2]), has_energy=torch.tensor([True, False]))
    assert b.get_metrics()[0] == pytest.approx(500.0) and b.get_metrics()[4] == pytest.approx(0.5)


def test_the_dataset_statistics_skip_unlabelled_structures():
    """The composition baseline and the mean energy per atom are fitted on the structures that carry an energy; a
    forces-only structure must not raise (it did, at the launch of 2026-11-07 00:00: ase PropertyNotImplementedError)."""
    atoms = _structures()
    assert T.compute_mean_energy_per_atom(atoms) == pytest.approx(-3.0 / 2)
    ref = T.compute_atomic_energies_from_dataset(atoms)
    assert set(ref) == {11, 17} and ref[11] + ref[17] == pytest.approx(-3.0)                 # one labelled NaCl, energy -3
    assert T.compute_atomic_energies_from_dataset(atoms[1:]) == {} or all(v == 0.0 for v in T.compute_atomic_energies_from_dataset(atoms[1:]).values())
