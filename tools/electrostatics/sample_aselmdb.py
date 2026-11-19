"""Draw a small, reproducible sample from a fairchem ``*.aselmdb`` file (OMat24 / OMol25).

The file is an LMDB whose values are zlib-compressed JSON rows of an ASE
database (``ase.io.jsonio``), keyed by the row id; entries that are not
integers are metadata.  Each row becomes ``ase.Atoms`` with its energy, forces
and stress (and ``info`` keys such as OMol25's ``charge`` / ``spin``) through
``ase.db.row.AtomsRow``.  Nothing from fairchem is needed -- only ``lmdb``.

Usage:
    python sample_aselmdb.py DATA.aselmdb OUT_PREFIX N_TRAIN N_TEST [SEED]

writes OUT_PREFIX_train.xyz and OUT_PREFIX_test.xyz (extxyz) and prints the
element and size statistics of the sample.  The draw is uniform without
replacement over all rows, fixed by SEED (default 0).
"""
import sys
import zlib
from collections import Counter

import lmdb
import numpy as np
from ase.db.row import AtomsRow
from ase.io import write
from ase.io.jsonio import decode


def read_rows(env, ids):
    out = []
    with env.begin() as txn:
        for i in ids:
            raw = txn.get(str(i).encode("ascii"))
            row = AtomsRow(decode(zlib.decompress(raw).decode("utf-8")))
            atoms = row.toatoms(add_additional_information=True)
            # Row 'data' and key-value pairs (e.g. OMol25 charge / spin) -> info.
            for key, value in (row.get("key_value_pairs") or {}).items():
                atoms.info.setdefault(key, value)
            atoms.info["source_row"] = int(i)
            out.append(atoms)
    return out


def main():
    path, prefix, n_train, n_test = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    env = lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        ids = [int(k) for k, _ in txn.cursor() if k.isdigit()]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(sorted(ids)), size=n_train + n_test, replace=False)
    frames = read_rows(env, chosen.tolist())
    for name, part in (("train", frames[:n_train]), ("test", frames[n_train:])):
        write(f"{prefix}_{name}.xyz", part, format="extxyz")
    sizes = np.array([len(a) for a in frames])
    elements = Counter(s for a in frames for s in set(a.get_chemical_symbols()))
    print(f"{len(ids)} rows; sampled {n_train} train + {n_test} test (seed {seed})")
    print(f"atoms per structure: min {sizes.min()} median {int(np.median(sizes))} max {sizes.max()}")
    print(f"{len(elements)} elements; most common: {elements.most_common(12)}")
    print(f"labels: {sorted(frames[0].calc.results) if frames[0].calc else None}; "
          f"info keys: {sorted(frames[0].info)}; pbc {frames[0].pbc.tolist()}")


if __name__ == "__main__":
    main()
