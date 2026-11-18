"""Is the crystal field's action on a pair's exchange density a two-point 'exchange split charge'? (2026-10-20)

    OMP_NUM_THREADS=4 nice python tools/electrostatics/exchange_charge_test.py OUT_DIR
    (python: an interpreter with PySCF installed)

Exact within first order (frozen ions, frozen environment): the environment adds to a pair's first-order energy

    E3 = int dP(r) V_env(r) d^3r,      dP = P_HL - P_u - P_v,   int dP = 0,

V_env the electron's potential energy from every ion but u and v (cluster_interaction: 'dP_field').  Because dP holds
no charge, a potential linear along the bond gives exactly

    E3 = a [V_env(R_u) - V_env(R_v)],   a = d . (R_u - R_v) / R^2,   d = int r dP(r) d^3r     (two-point form)

and the multipole expansion about the bond midpoint c is E3 = d . grad V(c) + 1/2 Q : grad grad V(c) + ...,
Q = int (r - c)(r - c) dP.  Measured here for Mg+/O2- and Mg2+/O2- (frame) in the bulk and (100)-surface rock-salt
lattice of cluster_interaction (point ions, Evjen, n = 4), both ligand kinds at the surface.

Pre-registered: the two-point form is adequate if it reproduces the frame-relative E3 within 20 % for every pair;
the dipole-at-midpoint form is reported beside it, and the quadrupole term says what the first omits.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cluster_interaction as CI  # noqa: E402

HARTREE = 27.211386245988
BOHR = 0.529177210903
K_E = 14.399645


def V_electron(points_A, lattice):
    """Electron potential energy (eV) of the point ions at points (A); the 0.3 A Gaussians are point-like beyond ~1 A,
    and every ion sits >= 2.1 A from the points used here (checked)."""
    coords, q = lattice
    out = []
    for p in np.atleast_2d(points_A):
        r = np.linalg.norm(coords - p, axis=1)
        assert r.min() > 1.0
        out.append(-float(np.sum(q * K_E / r)))
    return np.array(out)


def grad_hess(c, lattice, h=1e-3):
    g = np.zeros(3)
    H = np.zeros((3, 3))
    V0 = V_electron(c, lattice)[0]
    for i in range(3):
        e = np.zeros(3); e[i] = h
        g[i] = (V_electron(c + e, lattice)[0] - V_electron(c - e, lattice)[0]) / (2 * h)
        for j in range(3):
            f = np.zeros(3); f[j] = h
            H[i, j] = (V_electron(c + e + f, lattice)[0] - V_electron(c + e - f, lattice)[0]
                       - V_electron(c - e + f, lattice)[0] + V_electron(c - e - f, lattice)[0]) / (4 * h * h)
    return V0, g, H


def measure(q, site, surface, d=2.106, n_lat=4):
    lat = CI.rocksalt_lattice(d, n_lat, surface, exclude=[(0, 0, 0), site])
    Ru, Rv = np.zeros(3), d * np.array(site, float)
    mono = [("Mg", q, tuple(Ru)), ("O", -2, tuple(Rv))]
    out, mol, dP = CI.cluster_first_order(mono, xcs=("hf",), lattice=lat, return_dP=True)
    c = 0.5 * (Ru + Rv)
    r_int = mol.intor_symmetric("int1e_r")                                     # bohr, about the origin
    dip = np.array([float(np.einsum("ij,ji->", dP, r_int[x])) for x in range(3)]) * BOHR   # electrons A
    with mol.with_common_origin(c / BOHR):
        rr = mol.intor_symmetric("int1e_rr").reshape(3, 3, mol.nao_nr(), mol.nao_nr())
    Q = np.array([[float(np.einsum("ij,ji->", dP, rr[x, y])) for y in range(3)] for x in range(3)]) * BOHR ** 2
    Vu, Vv = V_electron(np.array([Ru, Rv]), lat)
    a = float(dip @ (Ru - Rv)) / float((Ru - Rv) @ (Ru - Rv))
    V0, g, H = grad_hess(c, lat)
    return {"E3_measured": out["dP_field"], "two_point": a * (Vu - Vv), "dipole_mid": float(dip @ g),
            "quadrupole_mid": 0.5 * float(np.sum(Q * H)), "a_exchange_charge": a, "dipole_eA": dip.tolist(),
            "V_u": Vu, "V_v": Vv}


def main(out_dir):
    res = {}
    for label, surface, sites in (("bulk", False, [(1, 0, 0)]), ("surface", True, [(1, 0, 0), (0, 0, -1)])):
        for site in sites:
            for q in (1, 2):
                key = f"{label}_{site}_{q}"
                res[key] = measure(q, site, surface)
                r = res[key]
                print(f"{key}: E3 measured {r['E3_measured']:+.4f} eV; two-point {r['two_point']:+.4f}; dipole at "
                      f"midpoint {r['dipole_mid']:+.4f} (+ quadrupole {r['quadrupole_mid']:+.4f}); exchange charge a "
                      f"{r['a_exchange_charge']:+.5f} e; V_env(Mg) {r['V_u']:+.3f}, V_env(O) {r['V_v']:+.3f} eV",
                      flush=True)
                json.dump(res, open(os.path.join(out_dir, "exchange_charge_test.json"), "w"), indent=1)
            m = res[f"{label}_{site}_1"]["E3_measured"] - res[f"{label}_{site}_2"]["E3_measured"]
            t = res[f"{label}_{site}_1"]["two_point"] - res[f"{label}_{site}_2"]["two_point"]
            dm = res[f"{label}_{site}_1"]["dipole_mid"] - res[f"{label}_{site}_2"]["dipole_mid"]
            ok = abs(t - m) <= 0.2 * abs(m)
            print(f"{label} {site} frame-relative E3: measured {m:+.4f}, two-point {t:+.4f} "
                  f"({'PASS' if ok else 'FAIL'} at 20 %), dipole at midpoint {dm:+.4f}", flush=True)


if __name__ == "__main__":
    os.makedirs(sys.argv[1], exist_ok=True)
    main(sys.argv[1])
