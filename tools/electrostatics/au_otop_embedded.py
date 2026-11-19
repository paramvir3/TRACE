"""Au(q) on the O-top site of MgO(100), embedded: is the runtime Pauli table right there? (2026-10-20)

    OMP_NUM_THREADS=4 nice python tools/electrostatics/au_otop_embedded.py OUT_DIR
    (python: an interpreter with PySCF installed)

The T5 benchmark (4G Table II) turns on adding the dopant's electron to Au2 on an O-top site against Au2 over Mg.
The runtime Pauli term for Au(q)/O2- (SPEC 25, K dS) was compared on 2026-10-20 with the frozen-density PBE value
of the two ions alone (vac) and with the O state's Watson sphere in the pair Hamiltonian (sph); the embedded
Mg+(O2-)n clusters showed the vacuum convention is not the crystal.  Here the pair Au(q) + O2- sits on the (100)
surface point-ion lattice of cluster_interaction (Evjen slab, n = 4; the O site excluded), the Au 2.26 A above the
O (the T5 reference's O-top distance), lattice d = 2.1325 A (the 4G cell, a = 4.265 A).  Frame Au0.

Measured: the frame-relative nonelst (PBE) of the embedded pair, and of the same pair in vacuum, for q = -1, +1.
Pre-registered: the runtime K dS (+1.149 eV for Au-, -1.049 eV for Au+, at 2.26 A) is adequate if within 0.1 eV of
the embedded value.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cluster_interaction as CI  # noqa: E402

RUNTIME = {-1: +1.1491, 1: -1.0488}


def main(out_dir, d=2.1325, R=2.26):
    os.makedirs(out_dir, exist_ok=True)
    site = (1, 0, 0)                                           # an anion site of the surface layer (k = 0)
    lat = CI.rocksalt_lattice(d, 4, True, exclude=[site])
    O = d * np.array(site, float)
    Au = O + np.array([0.0, 0.0, R])
    res = {}
    for q in (-1, 0, 1):
        mono = [("Au", q, tuple(Au)), ("O", -2, tuple(O))]
        res[f"emb_{q}"] = CI.cluster_first_order(mono, xcs=("hf", "pbe"), lattice=lat, with_elst=False)
        res[f"vac_{q}"] = CI.cluster_first_order(mono, xcs=("hf", "pbe"), with_elst=False)
        print(f"Au{q:+d}/O2- O-top: nonelst_pbe embedded {res[f'emb_{q}']['nonelst_pbe']:+.4f}, vac "
              f"{res[f'vac_{q}']['nonelst_pbe']:+.4f} eV; E3 {res[f'emb_{q}']['dP_field']:+.4f}", flush=True)
        json.dump(res, open(os.path.join(out_dir, "au_otop_embedded.json"), "w"), indent=1)
    for q in (-1, 1):
        e = res[f"emb_{q}"]["nonelst_pbe"] - res["emb_0"]["nonelst_pbe"]
        v = res[f"vac_{q}"]["nonelst_pbe"] - res["vac_0"]["nonelst_pbe"]
        ok = abs(RUNTIME[q] - e) < 0.1
        print(f"Au{q:+d}/O2- frame-relative nonelst (PBE): embedded {e:+.4f}, vac {v:+.4f}, runtime {RUNTIME[q]:+.4f} "
              f"eV -> runtime {'adequate' if ok else 'NOT adequate'} at 0.1 eV", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
