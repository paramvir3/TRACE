"""Convert the gas-phase MB-pol frames to open boundaries and split them.

Source: training_examples/water_ccsd/MB-pol-1250.xyz (superset of train.xyz).
Kept: the 658 gas-phase clusters (1-4 waters, stored in a 200 A box with
pbc=TTT). Dropped: the 573 periodic 256-water boxes and 19 512-water slabs,
which are genuinely periodic and far too large for a CPU budget with a
periodic electrostatic solve.

Each kept frame is written with pbc=False, no cell, and a SinglePointCalculator
holding energy and forces (the source stores forces as the non-standard
per-atom array 'force', which atoms.get_forces() does not see).

Split: stratified by cluster size, and for dimers additionally by O-O distance
bin; 25% test, 10% validation, 65% train; fixed seed. Strata of 2-4 frames get
at least one frame in train and one in test.
"""
import itertools
import json
import os

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                   "training_examples", "water_ccsd", "MB-pol-1250.xyz"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)
DIMER_BINS = [0.0, 2.5, 3.0, 3.5, 4.5, 6.0, 1e9]
SEED = 20260925
TEST_FRAC, VAL_FRAC = 0.25, 0.10


def molecules(atoms):
    s = atoms.get_chemical_symbols()
    p = atoms.positions
    oxy = [i for i, x in enumerate(s) if x == "O"]
    mol = {o: [o] for o in oxy}
    for i, x in enumerate(s):
        if x == "H":
            d = [np.linalg.norm(p[i] - p[o]) for o in oxy]
            mol[oxy[int(np.argmin(d))]].append(i)
    assert all(len(v) == 3 for v in mol.values())
    return oxy, mol


def pair_geometry(atoms):
    oxy, mol = molecules(atoms)
    p = atoms.positions
    roo, dmin = [], []
    for a, b in itertools.combinations(oxy, 2):
        roo.append(float(np.linalg.norm(p[a] - p[b])))
        dmin.append(float(min(np.linalg.norm(p[i] - p[j]) for i in mol[a] for j in mol[b])))
    return roo, dmin


frames = read(SRC, ":")
kept = []
for idx, a in enumerate(frames):
    if len(a) > 12:
        continue
    assert np.allclose(a.cell.lengths(), 200.0)
    energy = a.calc.results["energy"]
    forces = np.array(a.arrays["force"], dtype=float)
    b = Atoms(numbers=a.numbers, positions=a.positions, pbc=False)
    roo, dmin = pair_geometry(a)
    b.info["src_index"] = idx
    b.info["n_water"] = len(a) // 3
    b.info["roo_min"] = min(roo) if roo else -1.0
    b.info["roo_max"] = max(roo) if roo else -1.0
    b.info["dmin_inter"] = min(dmin) if dmin else -1.0
    b.info["pair_roo"] = json.dumps(roo)
    b.info["pair_dmin"] = json.dumps(dmin)
    b.calc = SinglePointCalculator(b, energy=energy, forces=forces)
    kept.append(b)
print(f"kept {len(kept)} gas-phase frames of {len(frames)}")


def stratum(b):
    n = b.info["n_water"]
    if n != 2:
        return f"n{n}"
    r = b.info["roo_min"]
    for lo, hi in zip(DIMER_BINS[:-1], DIMER_BINS[1:]):
        if lo <= r < hi:
            return f"n2_{lo:g}-{hi:g}"
    raise AssertionError


rng = np.random.default_rng(SEED)
strata = {}
for i, b in enumerate(kept):
    strata.setdefault(stratum(b), []).append(i)
split = {}
for key in sorted(strata):
    members = list(strata[key])
    rng.shuffle(members)
    n = len(members)
    if n <= 4:
        n_test, n_val = max(1, n // 2), 0
    else:
        n_test = int(round(TEST_FRAC * n))
        n_val = int(round(VAL_FRAC * n))
    for j, m in enumerate(members):
        split[m] = "test" if j < n_test else ("valid" if j < n_test + n_val else "train")
    print(f"  stratum {key:14s} n={n:4d}: train {sum(split[m]=='train' for m in members):3d} "
          f"valid {sum(split[m]=='valid' for m in members):3d} test {sum(split[m]=='test' for m in members):3d}")

for part in ("train", "valid", "test"):
    subset = [kept[i] for i in range(len(kept)) if split[i] == part]
    for b in subset:
        b.info["split"] = part
    write(os.path.join(OUT, f"{part}.xyz"), subset, format="extxyz")
    print(f"{part}: {len(subset)} frames")
