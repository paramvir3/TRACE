"""Bulk MgO phonons of a TRACE model with the phonopy settings of the reference calculation: rocksalt conventional
cell at a, force-constant supercell 2^3 conventional (64 atoms), displacement 0.01 A, primitive 'auto', symmetrised force constants, band path
Gamma-X-K-Gamma-L (fcc primitive, 51 points per segment), q -> 0 limit at |q| = 1e-3 (reduced).

Usage (repo root): python examples/mgo_phonons_local/phonons_model.py RUN_DIR --a 4.21726 --out DIR
                               [--born-from FILE]
RUN_DIR holds the TRACE model.pt.  Writes DIR/band_nonac.yaml (forces only) and, with --born-from (a JSON {"Z_Mg": z,
"eps": e}, e.g. DFT's own Z* / eps_inf), DIR/band_nac.yaml.
"""
import argparse
import json
import os
import sys

import numpy as np
from ase import Atoms
from ase.build import bulk
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.atoms import PhonopyAtoms

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from flashace.calculator import TransformersACECalculator  # noqa: E402

THZ_TO_CM = 33.35640951981521
K_E = 14.399645
Q_LIMIT = 1.0e-3
PATH = [[[0, 0, 0], [0.5, 0, 0.5], [0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]]]
LABELS = ["G", "X", "K", "G", "L"]


def gamma(ph, qdir):
    q = [0.0, 0.0, 0.0] if qdir is None else (Q_LIMIT * np.asarray(qdir, float) / np.linalg.norm(qdir)).tolist()
    ph.run_qpoints([q])
    return np.sort(ph.get_qpoints_dict()["frequencies"][0]) * THZ_TO_CM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--a", type=float, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--born-from", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    calc = TransformersACECalculator(model_path=os.path.join(args.run, "model.pt"), device="cpu")
    conv = bulk("MgO", "rocksalt", a=args.a, cubic=True)
    ph = Phonopy(PhonopyAtoms(symbols=conv.get_chemical_symbols(), cell=conv.cell.array,
                              scaled_positions=conv.get_scaled_positions()),
                 supercell_matrix=np.diag([2, 2, 2]), primitive_matrix="auto")
    ph.generate_displacements(distance=0.01)
    forces = []
    for sc in ph.supercells_with_displacements:
        at = Atoms(numbers=sc.numbers, positions=sc.positions, cell=sc.cell, pbc=True)
        at.info = {"charge": 0.0}
        at.calc = calc
        forces.append(at.get_forces())
    ph.forces = np.array(forces)
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    qpoints, connections = get_band_qpoints_and_path_connections(PATH, npoints=51)
    f0 = gamma(ph, None)
    ph.run_band_structure(qpoints, path_connections=connections, labels=LABELS, with_eigenvectors=False)
    ph.write_yaml_band_structure(filename=os.path.join(args.out, "band_nonac.yaml"))
    print(f"=== {os.path.basename(os.path.normpath(args.run))} at a = {args.a} A ({len(forces)} displaced 64-atom supercells)")
    print(f"(2) Gamma without NAC (cm^-1): {np.round(f0, 1).tolist()}")
    if args.born_from:
        b = json.load(open(args.born_from))
        born = np.array([np.eye(3) * (b["Z_Mg"] if s == "Mg" else -b["Z_Mg"]) for s in ph.primitive.symbols])
        ph.nac_params = {"born": born, "dielectric": np.eye(3) * b["eps"], "factor": K_E}
        res = gamma(ph, [1, 0, 0])
        print(f"    with NAC (Z* {b['Z_Mg']:.3f}, eps_inf {b['eps']:.3f} from {args.born_from}), q -> 0 along [100]: {np.round(res, 1).tolist()}")
        print(f"    TO {res[3]:.1f}, LO {res[5]:.1f} cm^-1")
        ph.run_band_structure(qpoints, path_connections=connections, labels=LABELS, with_eigenvectors=False)
        ph.write_yaml_band_structure(filename=os.path.join(args.out, "band_nac.yaml"))


if __name__ == "__main__":
    main()
