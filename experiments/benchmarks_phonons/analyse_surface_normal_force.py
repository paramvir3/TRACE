"""The surface ions' pull on the crystal factor g, split into an offset, a stiffness and a remainder
(README, pre-registered 2026-11-13; numpy and ase only, no model is loaded).

    python experiments/benchmarks_phonons/analyse_surface_normal_force.py PER_ATOM.npz TRAIN.xyz

PER_ATOM.npz is probe_alpha_scale_atoms.py's per-atom record (atoms=...): for every atom of every undoped slab the
force error res = F - F_DFT and the response dF = dF / dlog g.  Along the outward normal n of a surface layer
(the facing layer's points at the dimer, the far layer's away from it):

    r = res . n,   s = dF . n,   u = (z - <z>_structures) n_z   (the ion's normal displacement from its own mean)

and with least-squares fits r = a_r + b_r u + eps, s = a_s + b_s u + eta over a group (u centred),

    sum r s = N a_r a_s  +  b_r b_s sum (u - ubar)^2  +  sum eps eta        (exact: eps, eta are orthogonal to 1 and u)
              offset        stiffness                    remainder

The pull of a group on log g is k sum (res . dF) with k = 2 forces_weight force_weight / (3 N_atoms).
The DFT's own mean normal force and normal force constant come from the same fit on F_DFT . n.
"""
import sys

import numpy as np
from ase.io import read

rec = np.load(sys.argv[1], allow_pickle=True)
rows, w_f = rec["rows"], float(rec["forces_weight"])
frames = read(sys.argv[2], index=":")
col = {name: k for k, name in enumerate(str(rec["columns"]).split())}
st, at, Z, layer, side = (rows[:, col[c]].astype(int) for c in ("structure", "atom", "Z", "layer", "side"))
z = rows[:, col["z"]]
res = rows[:, [col["res_x"], col["res_y"], col["res_z"]]]
dF = rows[:, [col["dF_x"], col["dF_y"], col["dF_z"]]]
k = 2.0 * w_f * rows[:, col["force_weight"]] / (3.0 * rows[:, col["n_atoms"]])
structures = np.unique(st)
M = len(structures)
assert len(np.unique(side)) == 1, "the slabs are not all on one side of the dimer; the mean positions below would mix them"
f_dft = np.zeros(len(rows))
for s_ in structures:                                                # the DFT normal (z) force of every atom
    m = st == s_
    f_dft[m] = frames[int(s_)].get_forces()[at[m], 2]

# every atom's mean height over the structures (the data: one geometry plus small displacements), cell-wrapped
L = float(frames[int(structures[0])].cell[2, 2])
u = np.zeros(len(rows))
spread = []
for a in np.unique(at):
    m = at == a
    ref = z[m][0]
    zz = ref + (z[m] - ref + 0.5 * L) % L - 0.5 * L
    u[m] = zz - zz.mean()
    spread.append(zz.std())
print(f"{M} structures, {len(rows)} atom records; rms displacement of an atom along z from its own mean {np.sqrt((u ** 2).mean()):.4f} A "
      f"(largest per-atom std {max(spread):.4f}); k = {k.mean():.4e} (spread {k.std():.1e})")

GROUPS = [("facing layer, O", (layer == 1) & (Z == 8), -1.0), ("facing layer, Mg", (layer == 1) & (Z == 12), -1.0),
          ("bare far layer, O", (layer == 6) & (Z == 8), +1.0), ("bare far layer, Mg", (layer == 6) & (Z == 12), +1.0),
          ("layer 2", layer == 2, -1.0), ("layer 3", layer == 3, -1.0), ("layer 4", layer == 4, +1.0), ("layer 5", layer == 5, +1.0)]


def fit(y, x):
    """y = a + b (x - xbar) + e: a, b, their standard errors, the residual."""
    xc = x - x.mean()
    b = float(xc @ y / (xc @ xc))
    a = float(y.mean())
    e = y - a - b * xc
    dof = max(len(y) - 2, 1)
    s2 = float(e @ e / dof)
    return a, b, np.sqrt(s2 / len(y)), np.sqrt(s2 / float(xc @ xc)), e, xc


def split(mask, sign, name):
    n = sign * float(side[0])                                        # outward normal's z component
    r, s, x, kk = res[mask, 2] * n, dF[mask, 2] * n, u[mask] * n, k[mask]
    a_r, b_r, sa_r, sb_r, e, xc = fit(r, x)
    a_s, b_s, sa_s, sb_s, h, _ = fit(s, x)
    a_d, b_d, sa_d, sb_d, _, _ = fit(f_dft[mask] * n, x)
    kbar = float(kk.mean())
    offset, stiff, rem = len(r) * a_r * a_s, b_r * b_s * float(xc @ xc), float(e @ h)
    total = float(r @ s)
    assert abs(offset + stiff + rem - total) <= 1e-9 * (abs(total) + 1.0), (offset + stiff + rem, total)
    lateral = float((res[mask, :2] * dF[mask, :2]).sum())
    out = {"name": name, "n": int(mask.sum()), "normal": kbar * total / M, "offset": kbar * offset / M, "stiffness": kbar * stiff / M,
           "remainder": kbar * rem / M, "lateral": kbar * lateral / M}
    print(f"  {name:20s} pull/slab along n {out['normal']:+.3e} = offset {out['offset']:+.3e} ({offset / total:+5.0%}) + stiffness {out['stiffness']:+.3e} "
          f"({stiff / total:+5.0%}) + remainder {out['remainder']:+.3e} ({rem / total:+5.0%}); lateral {out['lateral']:+.2e}")
    print(f"  {'':20s} force error along n: mean {a_r:+.4f} +- {sa_r:.4f} eV/A, slope in u {b_r:+.3f} +- {sb_r:.3f} eV/A^2 | "
          f"response dF.n/dlog g: mean {a_s:+.4f} +- {sa_s:.4f}, slope {b_s:+.3f} +- {sb_s:.3f} | "
          f"DFT: mean force {a_d:+.4f} +- {sa_d:.4f}, slope {b_d:+.2f} +- {sb_d:.2f} eV/A^2")
    return out


print("outward normal n; u > 0 = the ion displaced outward; a positive force error pushes the ion outward more than DFT does")
parts = [split(m, sg, name) for name, m, sg in GROUPS]
surf = parts[:4]
tot = {key: sum(p[key] for p in surf) for key in ("normal", "offset", "stiffness", "remainder", "lateral")}
print(f"== the four surface groups together, per slab: normal pull {tot['normal']:+.4e} = offset {tot['offset']:+.4e} ({tot['offset'] / tot['normal']:+.1%}) "
      f"+ stiffness {tot['stiffness']:+.4e} ({tot['stiffness'] / tot['normal']:+.1%}) + remainder {tot['remainder']:+.4e} ({tot['remainder'] / tot['normal']:+.1%}); "
      f"lateral {tot['lateral']:+.2e}")
everything = float((k[:, None] * res * dF).sum()) / M
print(f"   all atoms, all components, per slab: {everything:+.4e} (the force term's d/d log g of the undoped slabs)")
