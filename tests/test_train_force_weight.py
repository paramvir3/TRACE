"""Per-structure force weights (2026-11-07; written red before the implementation).  The bulk-augmented phonon line
showed the 136 displaced PBE MgO cells (RMS force 0.038 eV/A) carry 0.37 % of the force loss at equal relative error
against the 300 slabs (RMS 0.42 eV/A): the per-structure force MSE weights a structure by its force scale squared.
Contract: ``atoms.info['force_weight']`` (default 1) multiplies that structure's force-loss term in the batched and
the per-structure paths; the dataset item and the collate carry it; the metrics stay unweighted."""
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
    b.calc = SinglePointCalculator(b, forces=np.array([[0, 0, 0.02], [0, 0, -0.02]]))
    b.info["has_energy"] = False; b.info["force_weight"] = 122.4
    return [a, b]


def test_the_dataset_and_collate_carry_the_force_weight():
    ds = T.AtomisticDataset(_structures(), 4.0, random_rotation=False, dtype=torch.float64)
    items = [ds[0], ds[1]]
    assert float(items[0]["force_weight"]) == 1.0 and float(items[1]["force_weight"]) == 122.4
    bat = T.AtomisticDataset.collate_batched(items)
    assert bat["force_weight"].tolist() == [1.0, 122.4]


def test_the_force_loss_weights_each_structure():
    per = torch.tensor([0.06, 0.0012], dtype=torch.float64)          # sum over atoms of |dF|^2
    n_at = torch.tensor([2.0, 2.0], dtype=torch.float64)
    w = torch.tensor([1.0, 122.4], dtype=torch.float64)
    assert float(T.force_loss(per, n_at, w)) == pytest.approx(0.06 / 6.0 + 122.4 * 0.0012 / 6.0)
    assert float(T.force_loss(per, n_at, torch.ones_like(w))) == pytest.approx(float((per / (3.0 * n_at)).sum()))
