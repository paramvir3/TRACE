"""The remaining AuMgO one-carrier pair types, embedded (2026-10-20).  Same method as au_otop_embedded.py.

    OMP_NUM_THREADS=2 nice python tools/electrostatics/embedded_pairs.py OUT_DIR
    (python: an interpreter with PySCF installed)

Measured 2026-10-20: for every pair with an O2- partner (Au-, Au+ on O-top; Mg+ bulk, surface in-plane, surface
below) the frozen-density PBE Pauli term with the O state's Watson sphere in the pair Hamiltonian ('sph') is within
0.05-0.10 eV of the value embedded in the point-ion lattice.  Does that hold where the partner is a cation (no
sphere) or the carrier is the O hole?
  (1) Au(q) above a surface Mg2+ at 2.90 A (the T5 flat site), q = -1, 0, +1; frame Au0;
  (2) the O hole: O(-1) / Mg2+ in the bulk, Mg at +x, +y, +z of the O site (one O- orientation: the three bonds
      average its p hole); frame O2- / Mg2+.
Lattice d = 2.1325 A (the 4G cell).  Pre-registered: a convention ('vac' / 'sph', from compare_overlap) or the
runtime K dS is adequate for a type if within 0.1 eV of the embedded frame-relative value.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cluster_interaction as CI  # noqa: E402

D = 2.1325


def au_over_mg(out, R=2.90):
    site = (0, 0, 0)                                           # the cation site of the surface layer
    lat = CI.rocksalt_lattice(D, 4, True, exclude=[site])
    Mg = np.zeros(3)
    Au = Mg + np.array([0.0, 0.0, R])
    for q in (-1, 0, 1):
        mono = [("Au", q, tuple(Au)), ("Mg", 2, tuple(Mg))]
        out[f"auMg_emb_{q}"] = CI.cluster_first_order(mono, xcs=("hf", "pbe"), lattice=lat, with_elst=False)
        out[f"auMg_vac_{q}"] = CI.cluster_first_order(mono, xcs=("hf", "pbe"), with_elst=False)
        print(f"Au{q:+d}/Mg2+ flat: nonelst_pbe embedded {out[f'auMg_emb_{q}']['nonelst_pbe']:+.4f}, vac "
              f"{out[f'auMg_vac_{q}']['nonelst_pbe']:+.4f}; E3 {out[f'auMg_emb_{q}']['dP_field']:+.4f}", flush=True)
    for q in (-1, 1):
        e = out[f"auMg_emb_{q}"]["nonelst_pbe"] - out["auMg_emb_0"]["nonelst_pbe"]
        v = out[f"auMg_vac_{q}"]["nonelst_pbe"] - out["auMg_vac_0"]["nonelst_pbe"]
        print(f"Au{q:+d}/Mg2+ frame-relative nonelst (PBE): embedded {e:+.4f}, vac {v:+.4f} eV", flush=True)


def o_hole_bulk(out):
    O_site = (1, 0, 0)                                         # an anion site; its Mg neighbours at O + d e_x, e_y, e_z
    O = D * np.array(O_site, float)
    rows = {}
    for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        mg_site = tuple(int(a + b) for a, b in zip(O_site, axis))
        lat = CI.rocksalt_lattice(D, 4, False, exclude=[O_site, mg_site])
        Mg = D * np.array(mg_site, float)
        for q in (-1, -2):
            mono = [("O", q, tuple(O)), ("Mg", 2, tuple(Mg))]
            rows[(axis, q, "emb")] = CI.cluster_first_order(mono, xcs=("hf", "pbe"), lattice=lat)["nonelst_pbe"]
            rows[(axis, q, "vac")] = CI.cluster_first_order(mono, xcs=("hf", "pbe"))["nonelst_pbe"]
        print(f"O-/Mg2+ bulk, bond {axis}: frame-relative nonelst (PBE) embedded "
              f"{rows[(axis, -1, 'emb')] - rows[(axis, -2, 'emb')]:+.4f}, vac "
              f"{rows[(axis, -1, 'vac')] - rows[(axis, -2, 'vac')]:+.4f}", flush=True)
    for conv in ("emb", "vac"):
        vals = [rows[(ax, -1, conv)] - rows[(ax, -2, conv)] for ax in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        out[f"ohole_{conv}"] = float(np.mean(vals))
        out[f"ohole_{conv}_spread"] = float(np.ptp(vals))
    print(f"O-/Mg2+ bulk, three-bond average: embedded {out['ohole_emb']:+.4f} (spread {out['ohole_emb_spread']:.4f}), "
          f"vac {out['ohole_vac']:+.4f} eV", flush=True)


if __name__ == "__main__":
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    au_over_mg(out)
    json.dump(out, open(os.path.join(out_dir, "embedded_pairs.json"), "w"), indent=1)
    o_hole_bulk(out)
    json.dump(out, open(os.path.join(out_dir, "embedded_pairs.json"), "w"), indent=1)
