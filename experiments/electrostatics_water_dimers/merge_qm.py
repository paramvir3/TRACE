"""Merge data_qm/train_a.xyz and data_qm/train_b.xyz into data_qm/train.xyz.

The halves are the even and odd frames of data/train.xyz (computed on two
machines); the merge restores the original order and checks that every frame
is present exactly once, with positions identical to the source.
"""
import os

import numpy as np
from ase.io import read, write

HERE = os.path.dirname(os.path.abspath(__file__))
source = read(os.path.join(HERE, "data", "train.xyz"), ":")
halves = [read(os.path.join(HERE, "data_qm", f"train_{part}.xyz"), ":") for part in ("a", "b")]
merged = [None] * len(source)
merged[0::2], merged[1::2] = halves[0], halves[1]
assert len(halves[0]) + len(halves[1]) == len(source), "frame count mismatch"
for original, labelled in zip(source, merged):
    assert original.info["src_index"] == labelled.info["src_index"], "order mismatch"
    assert np.array_equal(original.positions, labelled.positions), "positions changed"
    dipole = labelled.calc.results.get("dipole") if labelled.calc is not None else None
    assert dipole is not None or "dipole" in labelled.info, "missing dipole label"
    assert "polarizability" in labelled.info, "missing polarizability label"
write(os.path.join(HERE, "data_qm", "train.xyz"), merged, format="extxyz")
print(f"merged {len(merged)} frames")
