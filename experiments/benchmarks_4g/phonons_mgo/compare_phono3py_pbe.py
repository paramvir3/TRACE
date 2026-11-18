"""Harmonic phonons of Togo's PBE phono3py dataset (rocksalt MgO, NIMS MDR) from its fc2 displacement data (the
`phonon_displacements` on the phonon supercell) with its own PBE Born charges and eps_inf, along Gamma-X-K-Gamma-L.
Usage: python -I compare_phono3py_pbe.py PHONO3PY_PARAMS_YAML OUT_DIR      (downloaded data: -I, paths as arguments)"""
import os, sys
import numpy as np, yaml
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
THZ_TO_CM = 33.35641
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
d = yaml.safe_load(open(src))
uc = d["unit_cell"]
unit = PhonopyAtoms(symbols=[p["symbol"] for p in uc["points"]], cell=np.array(uc["lattice"]), scaled_positions=np.array([p["coordinates"] for p in uc["points"]]))
ph = Phonopy(unit, supercell_matrix=np.array(d["phonon_supercell_matrix"]), primitive_matrix=np.array(d["primitive_matrix"]))
n_sc = len(ph.supercell)
disp = d["phonon_displacements"]
ph.dataset = {"natom": n_sc, "first_atoms": [{"number": int(x["atom"]) - 1, "displacement": np.array(x["displacement"]), "forces": np.array(x["forces"])} for x in disp]}
ph.produce_force_constants(); ph.symmetrize_force_constants()
nac = d["nac"]
born_uc = np.array(nac["born_effective_charge"]); eps = np.array(nac["dielectric_constant"])
sym_uc = [p["symbol"] for p in uc["points"]]
if len(born_uc) == len(ph.primitive):                                 # per primitive atom, in the primitive's order
    by_symbol = {s: born_uc[i] for i, s in enumerate(ph.primitive.symbols)}
else:                                                                 # per unit-cell atom
    by_symbol = {s: born_uc[i] for i, s in enumerate(sym_uc)}
    for i, s in enumerate(sym_uc):
        assert np.allclose(born_uc[i], by_symbol[s], atol=1e-6), (i, s)
assert float(by_symbol["Mg"][0, 0]) > 0 > float(by_symbol["O"][0, 0]), by_symbol   # the assignment by order is checked by sign
born_prim = np.array([by_symbol[s] for s in ph.primitive.symbols])
factor = float(nac.get("unit_conversion_factor", 14.399652))
ph.nac_params = {"born": born_prim, "dielectric": eps, "factor": factor}
a = float(np.array(uc["lattice"])[0, 0])
print(f"phono3py PBE MgO: a = {a:.5f} A, fc2 supercell {np.diag(d['phonon_supercell_matrix']).tolist()} ({n_sc} atoms), {len(disp)} displacements of "
      f"{np.linalg.norm(disp[0]['displacement']):.3f} A, primitive {ph.primitive.symbols}, NAC factor {factor}")
print(f"  Z* {[(s, round(float(by_symbol[s][0, 0]), 4)) for s in ('Mg', 'O')]}, eps_inf {float(eps[0, 0]):.4f}")
path = [[[0, 0, 0], [0.5, 0, 0.5], [0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]]]
qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
def f_at(q):
    ph.run_qpoints([q]); return np.sort(ph.get_qpoints_dict()["frequencies"][0]) * THZ_TO_CM
print(f"  Gamma (q -> 0, |q| = 1e-3): {np.round(f_at([1e-3, 0, 0]), 1).tolist()}")
for name, q in (("X", [0.5, 0, 0.5]), ("K", [0.375, 0.375, 0.75]), ("L", [0.5, 0.5, 0.5])):
    print(f"  {name}: {np.round(f_at(q), 1).tolist()}")
ph.run_band_structure(qpoints, path_connections=connections, labels=["G", "X", "K", "G", "L"], with_eigenvectors=False)
ph.write_yaml_band_structure(filename=os.path.join(out, "band_dft_nac.yaml"))
ph.nac_params = None
print(f"  Gamma without NAC: {np.round(f_at([0, 0, 0]), 1).tolist()}")
ph.run_band_structure(qpoints, path_connections=connections, labels=["G", "X", "K", "G", "L"], with_eigenvectors=False)
ph.write_yaml_band_structure(filename=os.path.join(out, "band_dft_nonac.yaml"))
print("written", out)
