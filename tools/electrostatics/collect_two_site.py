"""Collect the first-principles two-site functions of pairs of s-class frame atoms into trace_redox/data/two_site.json
(2026-11-13).  numpy only; it reads the logs of pair_no_hop_curve.py.

    python tools/electrostatics/collect_two_site.py LOG [LOG ...] [--out PATH]

For a pair of s1 frames the ensemble's sector is a two-site Hubbard model (au2_hubbard.py): with the ground-state
interaction energy E_gs(R) and the vertical sigma sigma* excitations E_T(R), E_S(R) of the pair (PBE, def2-TZVP, TDA),

    E_cov(R) = E_gs + E_T          the covalent (no-hop) configuration, by identity (W - U) / 2 = E_T
    U(R)     = E_S - E_T           the ionic gap
    t(R)     = sqrt(E_T E_S) / 2   the hop (one sign, falling monotonically with R for Au2, Ag2, H2)

Rules applied here, stated so that they can be checked: a point enters only if its triplet root carries at least half
of the sigma sigma* weight (E_cov needs the triplet alone); U and t are kept only where the singlet root ALSO carries
at least half of that weight (the rule pre-registered on 2026-11-13, "a root below 0.5 is flagged, not used"; until
the review of 2026-11-14 it was applied to the triplet only, and three hop points rested on singlet roots of weight
0.47, 0.45 and 0.36: H2 at 0.60 A, Au2 at 2.60 and 2.80 A) and continues the pair's monotone fall of E_S with R (a
rise is a root jump: Au2 at 3.0 A, 4.693 eV after 2.682).  What the rules leave out is kept in the file under
"flagged", not used by the model.

"applied" says whether trace_redox's module uses the pair (RedoxConfig.two_site).  A pair is applied where its functions
have been checked on something other than the bare pair, or where the pair cannot be anything else.  H-H is NOT
applied: in every molecule the two hydrogens of a geminal or vicinal pair lie beyond the last point (1.4 A) and are
each bonded to a third atom, and the bare pair's hop is not that pair's (review of 2026-11-14).
"""
import argparse
import json
import os
import re

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
Z = {"H": 1, "Ag": 47, "Au": 79}
#: pairs whose functions are tabulated but not used by the module, with the reason (see the docstring)
WITHHELD = {"H": "every H-H pair inside a molecule lies beyond the last point (1.4 A) and each H is bonded to a third atom; "
                 "the bare pair's hop is not that pair's, and no first-principles point of such a pair exists",
            "Ag": "the entries of SPEC 40.7 cannot be built for silver yet: data/ion_polarizabilities.json holds Ag- only, so the "
                  "neutral frame's polarizability is the learned head's and the package's fixed terms for a silver pair depend on "
                  "the weights (0.057 eV between two seeded modules, 2026-11-14); with the neutral entry alone the pair is in the "
                  "state the review of 2026-11-14 rejected (one of four levels corrected)"}
LINE = re.compile(r"^(\w+)2 R ([0-9.]+) A: ground singlet ([-+0-9.]+) eV.*?TDA sigma sigma\*: triplet ([0-9.]+) \(weight ([0-9.]+)\), "
                  r"singlet ([0-9.]+) \(weight ([0-9.]+)\)")


CARRIER = re.compile(r"^(\w+)2 q ([+-]\d) (\w+) (\S+) R ([0-9.]+) A: .*?(EA|IP)\(X2\) [-+0-9.]+, atom [-+0-9.]+, on the exact atom ([-+0-9.]+); (\w+)$")


def carriers(paths, xc="pbe"):
    """{symbol: {R: {'EA': .., 'IP': ..}}} from logs of pair_one_carrier_curve.py: the vertical EA and IP of the pair on
    the exact atomic values, VALID points of the functional ``xc`` only."""
    out = {}
    for path in paths:
        for line in open(path):
            m = CARRIER.match(line.strip())
            if m and m.group(3) == xc and m.group(8) == "VALID":
                out.setdefault(m.group(1), {}).setdefault(float(m.group(5)), {})[m.group(6)] = float(m.group(7))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--carriers", nargs="*", default=[], help="logs of pair_one_carrier_curve.py (SPEC 40.7)")
    ap.add_argument("--drop-entries", action="store_true",
                    help="do not carry the built 'entries' of the existing output file over (a pair without entries is "
                         "then no longer applied: the package refuses an applied pair without them)")
    ap.add_argument("--out", default=os.path.join(REPO, "trace_redox", "data", "two_site.json"))
    args = ap.parse_args()
    rows = {}
    for path in args.logs:
        for line in open(path):
            m = LINE.match(line.strip())
            if m:
                sym = m.group(1)
                rows.setdefault(sym, {})[float(m.group(2))] = tuple(float(m.group(k)) for k in (3, 4, 5, 6, 7))
    pairs = {}
    for sym, table in sorted(rows.items(), key=lambda kv: Z[kv[0]]):
        R = np.array(sorted(table))
        e_gs, e_t, w_t, e_s, w_s = (np.array([table[r][k] for r in R]) for k in range(5))
        keep = w_t >= 0.5
        R, e_gs, e_t, w_t, e_s, w_s = (a[keep] for a in (R, e_gs, e_t, w_t, e_s, w_s))
        jump = np.zeros(len(R), dtype=bool)                                # the singlet root: E_S falls with R
        for k in range(1, len(R)):
            last = max(j for j in range(k) if not jump[j])
            if e_s[k] > e_s[last]:
                jump[k] = True
        ok = ~jump & (w_s >= 0.5)                                          # and carries half of the sigma sigma* weight
        U = np.where(ok, e_s - e_t, np.nan)
        t = np.where(ok, np.sqrt(e_t * e_s) / 2.0, np.nan)
        flagged = [{"R": round(float(R[k]), 6), "E_S": round(float(e_s[k]), 6), "weight_S": round(float(w_s[k]), 6),
                    "U": round(float(e_s[k] - e_t[k]), 6), "t": round(float(np.sqrt(e_t[k] * e_s[k]) / 2.0), 6),
                    "why": "root jump" if jump[k] else "singlet weight below 0.5"} for k in range(len(R)) if not ok[k]]
        clean = lambda a: [None if not np.isfinite(x) else round(float(x), 6) for x in a]
        pairs[f"{Z[sym]}-{Z[sym]}"] = {
            "symbols": [sym, sym], "R": clean(R), "E_gs": clean(e_gs), "E_T": clean(e_t), "weight_T": clean(w_t),
            "E_S": clean(np.where(ok, e_s, np.nan)), "weight_S": clean(w_s), "E_cov": clean(e_gs + e_t), "U": clean(U), "t": clean(t),
            "flagged": flagged, "applied": sym not in WITHHELD, "why_not_applied": WITHHELD.get(sym, "")}
        print(f"{sym}-{sym}: {len(R)} points at {R.min():.2f}-{R.max():.2f} A; U and t at {int(ok.sum())} of them; "
              f"t {np.nanmax(t):.3f} -> {np.nanmin(t):.3f} eV, U {np.nanmin(U):.3f} .. {np.nanmax(U):.3f} eV, "
              f"E_cov {float((e_gs + e_t)[0]):+.3f} .. {float((e_gs + e_t)[-1]):+.3f} eV; t monotone: {bool(np.all(np.diff(t[ok]) < 0))}")
    ions = carriers(args.carriers)
    for sym, table in ions.items():
        key = f"{Z[sym]}-{Z[sym]}"
        if key not in pairs:
            continue
        Rs = [r for r in pairs[key]["R"] if r in table and {"EA", "IP"} <= set(table[r])]
        if len(Rs) >= 2:
            pairs[key]["carriers"] = {"R": Rs, "EA": [table[r]["EA"] for r in Rs], "IP": [table[r]["IP"] for r in Rs],
                                      "method": "PBE; cation def2-TZVP, anion def2-TZVPD (def2 ECP for Ag, Au); both resonance states by "
                                                "symmetry-constrained UKS; vertical EA and IP of the pair on the exact atomic values; "
                                                "tools/electrostatics/pair_one_carrier_curve.py"}
            print(f"{sym}-{sym}: carriers at {len(Rs)} distances; EA* {min(pairs[key]['carriers']['EA']):.3f} .. {max(pairs[key]['carriers']['EA']):.3f}, "
                  f"IP* {min(pairs[key]['carriers']['IP']):.3f} .. {max(pairs[key]['carriers']['IP']):.3f} eV")
    # the built entries of SPEC 40.7 are kept (they record the table values they were solved with, and the package refuses
    # them if the table has changed since); without entries a pair is not applied
    old = {}
    if os.path.exists(args.out) and not args.drop_entries:
        with open(args.out) as fh:
            old = json.load(fh)["pairs"]
    for key in pairs:
        if "entries" in old.get(key, {}):
            pairs[key]["entries"] = old[key]["entries"]
        elif pairs[key]["applied"]:
            pairs[key]["applied"] = False
            pairs[key]["why_not_applied"] = ("no built entries (SPEC 40.7): run tools/electrostatics/build_two_site_entries.py; with the "
                                             "neutral entry alone the pair's electron affinity and ionisation energy are about 1 eV off")
            print(f"{key}: NOT applied: it has no built entries")
    out = {"method": "PBE / def2-TZVP (def2 ECP for Ag, Au), counterpoise-corrected ground-state interaction energy; TDA sigma sigma* "
                     "triplet and singlet roots picked by their weight on the atoms' valence s combinations; "
                     "tools/electrostatics/pair_no_hop_curve.py, collected by collect_two_site.py 2026-11-14 (singlet-weight rule applied)",
           "units": "eV, A", "pairs": pairs}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
