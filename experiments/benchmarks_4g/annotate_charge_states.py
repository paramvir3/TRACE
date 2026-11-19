"""Declare the SPEC 21 diabatic charge states of Au2 / MgO(001)(:Al) in the extxyz files.

Usage: python annotate_charge_states.py IN_DIR OUT_DIR

Two fragments: the substrate (fragment 0) and the Au2 cluster (fragment 1,
atoms 108 and 109).  Three integer charge states, in e (substrate, Au2):

    ( 0,  0)   neutral fragments,
    (+1, -1)   one electron from the substrate onto Au2 -- the donor -> acceptor
               transfer of the Al-doped slab (Hirshfeld q(Au2) -0.2 e doped),
    (-1, +1)   one electron from Au2 into the substrate (undoped, wetting:
               Hirshfeld q(Au2) +0.18 e).

Within each state the SQE polarization (SPEC 1) relaxes as usual; the states
are mixed by the learned through-bond coupling (SPEC 22, 'through_bond').  The
states describe the physics of the benchmark; they are written in the extxyz
file form of flashace.charge_state.declared_charge_states and are checked by it.
"""
import os
import sys

import numpy as np
from ase.io import read, write

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from flashace.charge_state import declared_charge_states  # noqa: E402

AU = (108, 109)
STATES = np.array([[0, 0], [1, -1], [-1, 1]], dtype=float)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for split in ("train", "valid", "test"):
        frames = read(os.path.join(src, f"AuMgO_{split}.xyz"), ":")
        for atoms in frames:
            assert [atoms.get_chemical_symbols()[i] for i in AU] == ["Au", "Au"]
            labels = np.zeros(len(atoms), dtype=int)
            labels[list(AU)] = 1
            atoms.arrays["charge_state_fragment"] = labels
            atoms.info["charge_state_charges"] = STATES.ravel()
            atoms.info["charge_state_coupling"] = "through_bond"
            states = declared_charge_states(atoms)
            assert [s["fragment_charges"] for s in states] == STATES.tolist()
        write(os.path.join(dst, f"AuMgO_{split}.xyz"), frames, format="extxyz")
        print(f"{split}: {len(frames)} structures, {len(STATES)} charge states each")


if __name__ == "__main__":
    main()
