"""The bond-oriented Pauli law at the flat geometry, against PBE (2026-11-04; theory note §4c, pre-registered).

    OMP_NUM_THREADS=4 nice python tools/electrostatics/bop_flat_check.py OUT.json
    (python: an interpreter with PySCF installed)

D_flat = E1(dimer) - E1(atom 0) - E1(atom 1) for wetting_eda's flat reference over a cluster built around it
(quantum_sites, r_au 3.6; frz_nonadditivity.first_order), against the law's absolute value
sum over the cluster's quantum O of [D1(R) cos t + D2(R) P2(cos t)], cos t = r_aj . (-e_ab), for both Au."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cluster_interaction as CI  # noqa: E402
import frz_nonadditivity as FN  # noqa: E402
import wetting_eda as W  # noqa: E402

R_TAB = np.array([2.26, 2.4, 2.6, 2.9])
LN = np.log([1.4254, 0.8454, 0.3891, 0.1045])


def f_of(R):
    s_lo = (LN[1] - LN[0]) / (R_TAB[1] - R_TAB[0]); s_hi = (LN[-1] - LN[-2]) / (R_TAB[-1] - R_TAB[-2])
    if R < R_TAB[0]:
        v = LN[0] + s_lo * (R - R_TAB[0])
    elif R > R_TAB[-1]:
        v = LN[-1] + s_hi * (R - R_TAB[-1])
    else:
        v = np.interp(R, R_TAB, LN)
    return np.exp(v) / 1.4254


def main(out_json):
    au = [np.array(p) for p in W.geometries()["flat"]]
    sites = W.quantum_sites(au, r_au=3.6)
    surf = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
    qB = sum(2 if sym == "Mg" else -2 for sym, _ in surf)
    lattice = CI.rocksalt_lattice(W.D, 5, True, exclude=sites)
    print(f"{len(surf)} quantum ions ({sum(1 for s, _ in surf if s == 'O')} O) around the flat reference", flush=True)
    law = 0.0
    for a, b in ((0, 1), (1, 0)):
        e = (au[b] - au[a]) / np.linalg.norm(au[b] - au[a])
        for sym, p in surf:
            if sym != "O":
                continue
            d = p - au[a]; R = np.linalg.norm(d); c = float(d @ (-e) / R)
            f = f_of(R)
            term = -0.183 * f ** 0.888 * c - 0.611 * np.sqrt(f) * 0.5 * (3 * c * c - 1)
            law += term
            print(f"  Au{a} - O at {R:.3f} A, cos t {c:+.3f}: law {term:+.4f} eV", flush=True)
    res = {"law_abs": law}
    res["dimer"] = FN.first_order(au, surf, qB, lattice)
    res["atom 0"] = FN.first_order(au[:1], surf, qB, lattice)
    res["atom 1"] = FN.first_order(au[1:], surf, qB, lattice)
    D = res["dimer"] - res["atom 0"] - res["atom 1"]
    res["D_flat"] = D
    json.dump(res, open(out_json, "w"), indent=1)
    ok = abs(D - law) <= max(0.3 * abs(law), 0.1)
    print(f"flat: E1 dimer {res['dimer']:+.4f} = atoms {res['atom 0']:+.4f} + {res['atom 1']:+.4f} + D {D:+.4f} eV; the law's "
          f"absolute value {law:+.4f} eV -> {'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
