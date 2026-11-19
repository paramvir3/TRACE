"""Phonon band structure of a phonondb (Togo, NIMS MDR) phonopy_params.yaml along Gamma-X-K-Gamma-L with its own NAC,
written as band.yaml next to the model's, plus the Gamma (q -> 0), X, K, L frequencies and Z*, eps_inf, a.
Usage: python -I compare_phonondb.py PHONOPY_PARAMS_YAML OUT_DIR   (the yaml is downloaded data: run with -I, paths as args)"""
import os, sys
import numpy as np
import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
THZ_TO_CM = 33.35641
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
ph = phonopy.load(src, is_nac=True, produce_fc=True, symmetrize_fc=True)
a = float(ph.unitcell.cell[0][0])
Z = ph.nac_params["born"]; eps = ph.nac_params["dielectric"]
sym = ph.primitive.symbols
print(f"phonondb mp-1265: a = {a:.5f} A, supercell {np.diag(ph.supercell_matrix).tolist()}, primitive {sym}, NAC factor {ph.nac_params.get('factor')}")
print(f"  Z* {[(s, round(float(Z[i][0][0]), 4)) for i, s in enumerate(sym)]}, eps_inf {float(eps[0][0]):.4f}")
path = [[[0, 0, 0], [0.5, 0, 0.5], [0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]]]
qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
ph.run_band_structure(qpoints, path_connections=connections, labels=["G", "X", "K", "G", "L"], with_eigenvectors=False)
ph.write_yaml_band_structure(filename=os.path.join(out, "band_dft_nac.yaml"))
def f_at(q):
    ph.run_qpoints([q]); return np.sort(ph.get_qpoints_dict()["frequencies"][0]) * THZ_TO_CM
print(f"  Gamma (q -> 0, |q| = 1e-3 along [100]): {np.round(f_at([1e-3, 0, 0]), 1).tolist()}")
print(f"  Gamma exactly (no NAC term applied): {np.round(f_at([0, 0, 0]), 1).tolist()}")
for name, q in (("X", [0.5, 0, 0.5]), ("K", [0.375, 0.375, 0.75]), ("L", [0.5, 0.5, 0.5])):
    print(f"  {name}: {np.round(f_at(q), 1).tolist()}")
ph.nac_params = None
ph.run_band_structure(qpoints, path_connections=connections, labels=["G", "X", "K", "G", "L"], with_eigenvectors=False)
ph.write_yaml_band_structure(filename=os.path.join(out, "band_dft_nonac.yaml"))
print("written", out)
