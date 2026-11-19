"""M1b / M1c against R15's lattice sum: do the mesh steps of each geometry follow the tin-foil sum of a point dipole?

Reads the outputs of probe_ext_mesh_convergence.py (nothing is evaluated with the model).  For each seed the energy
per cell is modelled as E = c_x Lambda_xx + c_z Lambda_zz (1e-3 / A^3; Lambda from probe_ext_dipole_lattice.Lam on
the mesh's supercell), the constants are fitted on the first x step and the first y step at c = 26.4589 A, and every
other step is a prediction.  The upright dimer uses c_z alone, fitted on the first x step.

    PYTHONPATH=. python experiments/benchmarks_4g/probe_ext_mesh_lattice_fit.py
"""
import re
import sys

import numpy as np

sys.path.insert(0, "experiments/benchmarks_4g")
from probe_ext_dipole_lattice import Lam  # noqa: E402

R = "experiments/benchmarks_4g/runs/extensivity_2026-11-10/"
A, C0 = 9.0474, 26.4589
_cache = {}


def lam(mesh, c):
    key = (mesh, c)
    if mesh == INF:                                               # exact: sum over k != 0 of 1/(k c)^3 and -2/(k c)^3
        return 1e3 * np.array([2 * ZETA3 / c**3, -4 * ZETA3 / c**3])
    if key not in _cache:
        L = Lam(np.diag([mesh[0] * A, mesh[1] * A, c]))
        _cache[key] = 1e3 * np.array([L[0, 0], L[2, 2]])          # (xx, zz)
    return _cache[key]


def energies(name):
    out = {}
    for line in open(R + name):
        m = re.match(r"\s+undoped bvk_mesh \((\d), (\d), 1\) (\S+)\s*: E/cell (-[\d.]+) eV", line)
        if m:
            out[(m.group(3), (int(m.group(1)), int(m.group(2))))] = float(m.group(4))
    return out


INF = "limit"                                                     # the chain of images along z that a lateral mesh leaves
ZETA3 = 1.2020569031595942
for seed in ("s43", "s42"):
    E = {}
    for name in (f"mesh_convergence_undoped_{seed}.txt", f"mesh_convergence_x_undoped_{seed}.txt",
                 f"mesh_convergence_y_undoped_{seed}.txt"):
        E.update(energies(name))
    E40 = energies(f"mesh_convergence_x_c40_undoped_{seed}.txt")

    def step(g, m0, m1, src=E):
        return 1e3 * (src[(g, m1)] - src[(g, m0)])

    def dl(m0, m1, c=C0):
        return lam(m1, c) - lam(m0, c)

    cz_up = step("non-wetting", (1, 1), (2, 1)) / dl((1, 1), (2, 1))[1]
    M = np.array([dl((1, 1), (2, 1)), dl((1, 1), (1, 2))])
    cx, cz = np.linalg.solve(M, [step("wetting", (1, 1), (2, 1)), step("wetting", (1, 1), (1, 2))])
    print(f"== {seed}: upright dimer c_z = {cz_up:+.4f}; flat dimer c_x = {cx:+.4f}, c_z = {cz:+.4f} (meV per 1e-3 /A^3)")
    rows = [((2, 1), (3, 1)), ((3, 1), (4, 1)), ((1, 2), (1, 3)), ((1, 3), (1, 4)), ((1, 1), (2, 2))]
    for g, const in (("non-wetting", np.array([0.0, cz_up])), ("wetting", np.array([cx, cz]))):
        for m0, m1 in rows:
            p, q = float(const @ dl(m0, m1)), step(g, m0, m1)
            print(f"   {g:11s} {m0} -> {m1}: predicted {p:+9.4f}  measured {q:+9.4f} meV  (predicted / measured {p / q:.3f})")
        for m0, m1 in (((1, 1), (2, 1)), ((2, 1), (3, 1)), ((3, 1), (4, 1))):
            p, q = float(const @ dl(m0, m1, 40.0)), step(g, m0, m1, E40)
            print(f"   {g:11s} {m0} -> {m1} at c = 40 A: predicted {p:+9.4f}  measured {q:+9.4f} meV  ({p / q:.3f})")
    up = np.array([0.0, cz_up]); fl = np.array([cx, cz])
    for m0, m1 in (((2, 2), (3, 3)), ((3, 3), (4, 4)), ((2, 2), INF), ((3, 3), INF)):
        d = float(fl @ dl(m0, m1)) - float(up @ dl(m0, m1))
        print(f"   ESTIMATE, not measured: E(wetting) - E(non-wetting), {m0} -> {m1}: {d:+.3f} meV "
              f"(flat {float(fl @ dl(m0, m1)):+.3f}, upright {float(up @ dl(m0, m1)):+.3f})")
    # the change with the vacuum period at each mesh: the lattice sum's part, and what is left (a term of the mean
    # state, e.g. the slab's static dipole under tin-foil boundaries, would be the same at every mesh)
    for g, const in (("non-wetting", up), ("wetting", fl)):
        for m in ((1, 1), (2, 1), (3, 1), (4, 1)):
            q = 1e3 * (E40[(g, m)] - E[(g, m)]); p_ = float(const @ (lam(m, 40.0) - lam(m, C0)))
            print(f"   {g:11s} {m}: c 26.4589 -> 40 A changes E/cell by {q:+9.4f} meV; lattice sum {p_:+9.4f}; left {q - p_:+9.4f}")
    print(f"   ESTIMATE, not measured: the chain through the vacuum that a lateral mesh leaves (c = {C0} A): "
          f"flat {float(fl @ lam(INF, C0)):+.3f}, upright {float(up @ lam(INF, C0)):+.3f} meV; "
          f"on E(wetting) - E(non-wetting) {float(fl @ lam(INF, C0)) - float(up @ lam(INF, C0)):+.3f} meV")
for c in (C0, 40.0):
    r = {m: lam(m, c) - lam(INF, c) for m in ((1, 1), (2, 2), (3, 3), (4, 4))}
    print(f"c = {c} A: of the 1x1 value, a square mesh keeps  xx: " + ", ".join(f"{r[m][0] / r[(1, 1)][0]:.4f}" for m in ((2, 2), (3, 3), (4, 4)))
          + "   zz: " + ", ".join(f"{r[m][1] / r[(1, 1)][1]:.4f}" for m in ((2, 2), (3, 3), (4, 4))) + "   (2x2, 3x3, 4x4)")
    print("   of Lambda_xx - limit, the term -2 pi / V_M is: " + ", ".join(
        f"{-2e3 * np.pi / (m[0] * m[1] * A * A * c) / r[m][0]:.4f}" for m in ((2, 2), (3, 3), (4, 4))) + "   (2x2, 3x3, 4x4)")
    big = lam((12, 12), c); V = 144 * A * A * c
    print(f"   check of the limit at 12x12: Lambda_xx + 2 pi / V = {big[0] + 2e3 * np.pi / V:+.6f} against 2 zeta(3) / c^3 = {lam(INF, c)[0]:+.6f}; "
          f"Lambda_zz = {big[1]:+.6f} against -4 zeta(3) / c^3 = {lam(INF, c)[1]:+.6f}  (1e-3 /A^3)")
