"""Is the tilted dimer's frozen penalty pairwise? (2026-11-04; theory note §4c, pre-registered 14:09)

    python tools/electrostatics/frz_pair_sum.py PATH.json AU_O.log AU_MG.log [R_AU]

The frozen-density PBE pair energies E(R) = sph_elst + sph_nonelst_pbe of Au0-O2- and Au0-Mg2+ (frame_pair_wall.py logs),
interpolated in R (cubic in log|E| is not safe across a sign change: linear in E between the computed R, zero beyond the
largest R), summed for each Au of the tilted (path lambda = 1) and the upright dimer on the same O over
(i) the quantum ions of almo_tilt.py's cluster (the like-for-like sum: only they carry a density in the ALMO FRZ), with the
    point ions' electrostatics omitted (a neutral spherical Au density sees a distant point charge only through its
    penetration, small at the point ions' distances; stated, not computed);
(ii) every surface-lattice ion within 4.2 A of an Au, as a real surface would give.
Compared with ALMO's FRZ(tilted) - FRZ(upright) = +1.402 eV.
"""
import json
import re
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cluster_interaction as CI  # noqa: E402
import wetting_eda as W  # noqa: E402

FRZ_ALMO = 1.402


def curve(log):
    R, E, nonel = [], [], []
    for line in open(log):
        m = re.search(r"R ([0-9.]+): sph_nonelst_pbe ([+-][0-9.]+) eV \(kin [^,]+, elst ([+-][0-9.]+)\)", line)
        if m:
            R.append(float(m.group(1))); nonel.append(float(m.group(2))); E.append(float(m.group(2)) + float(m.group(3)))
    o = np.argsort(R)
    return np.array(R)[o], np.array(E)[o], np.array(nonel)[o]


def energy(c, r):
    R, E, _ = c
    if r > R[-1]:
        return 0.0
    if r < R[0]:
        raise ValueError(f"R {r:.3f} below the computed range {R[0]:.3f}")
    return float(np.interp(r, R, E))


def main(path_json, log_o, log_mg, r_au=3.6):
    P = json.load(open(path_json))
    G = {"tilted": [np.array(p) for p in P["cluster_au"]["1.0"]], "upright": [np.array(p) for p in P["cluster_au"]["upright"]]}
    cO, cMg = curve(log_o), curve(log_mg)
    print(f"Au0-O2- pairs at R {cO[0].tolist()}; Au0-Mg2+ at {cMg[0].tolist()}")
    sites = W.quantum_sites(G["tilted"] + G["upright"], r_au=r_au)
    quantum = [("Mg" if sum(s) % 2 == 0 else "O", W.D * np.array(s, float)) for s in sites]
    lat, q = CI.rocksalt_lattice(W.D, 5, True, exclude=())
    surface = [("Mg" if qi > 0 else "O", np.array(p)) for p, qi in zip(lat, q) if abs(p[2]) < 2.2 * W.D]
    out = {}
    for label, ions in (("(i) quantum ions", quantum), ("(ii) all ions within 4.2 A", surface)):
        tot = {}
        for name, au in G.items():
            s = 0.0
            parts = []
            for a in au:
                for sym, p in ions:
                    r = float(np.linalg.norm(a - p))
                    if r > 4.2:
                        continue
                    e = energy(cO if sym == "O" else cMg, r)
                    s += e
                    parts.append((sym, round(r, 3), round(e, 3)))
            tot[name] = s
            print(f"  {label} {name}: sum {s:+.3f} eV over {len(parts)} pairs; " +
                  ", ".join(f"{a}@{r}:{e:+.3f}" for a, r, e in sorted(parts, key=lambda x: x[1])[:8]))
        d = tot["tilted"] - tot["upright"]
        out[label] = d
        print(f"  {label}: tilted - upright {d:+.3f} eV against ALMO FRZ {FRZ_ALMO:+.3f} -> "
              f"{'within 30 %' if abs(d - FRZ_ALMO) <= 0.3 * FRZ_ALMO else 'outside 30 %'}")
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 3.6)
