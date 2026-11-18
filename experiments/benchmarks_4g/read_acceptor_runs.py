# Written on 2026-10-29 (the measurement scripts of the theory note's section 4c); run from the repo root.
"""The pre-registered readings of the acceptor-orbital runs (theory note §4c, 2026-10-29 12:36 / 19:19 / 19:22):
H2 (shape): intra-Au dipole per gained electron > +0.1 A on both flat Au's; the upright's lower Au < -0.1 A (H1,
axial lobe) or |.| < 0.1 A (H2 alone).  H3 (level): Delta(flat) / Delta(upright) > 2 -> the level carries it; < 1.3
-> the shape does.  The orientation idea dies if either flat Au's intra dipole is <= +0.1 A per gained electron."""
import json, os
W = os.path.join(os.environ.get("TRACE_FP_CACHE", "fp_cache"), "acceptor_density")   # first-principles cache (TRACE_FP_CACHE)
R = {}
for g in ("flat", "upright", "mgtop"):
    f = f"{W}/acceptor_{g}.json"
    if not os.path.exists(f):
        print(f"{g}: not in"); continue
    R[g] = j = json.load(open(f))
    L = j["levels_eV"]
    print(f"{g}: CT raw {j['CT_raw_B->A_eV']:+.4f} eV; gained A {j['gained_A']:+.4f} e; levels acceptor {L['acceptor']:+.3f} (next {L['acceptor_2']:+.3f}), "
          f"donor top {L['donor_top']:+.3f}, Delta {L['acceptor'] - L['donor_top']:.3f} eV (free {L['free_LUMO'] - L['free_HOMO']:.3f})")
    for i, a in enumerate(j["Au"]):
        sh = a["shells"]; n = a["gained"]
        print(f"   Au{i} z {a['z']:.2f}: gained {n:+.4f} e (s {sh['s']:+.4f}, pz {sh['pz']:+.4f}, pxy {sh['pxy']:+.4f}, d {sh['d']:+.4f}); "
              f"mu_z intra {a['mu_z_eA']['intra']:+.5f}, partner {a['mu_z_eA']['partner']:+.5f}, surface {a['mu_z_eA']['surface']:+.5f} e.A; "
              f"intra per electron {a['intra_per_electron_A']:+.3f} A")
if "flat" in R and "upright" in R:
    fl = [a["intra_per_electron_A"] for a in R["flat"]["Au"]]
    up = R["upright"]["Au"][0]["intra_per_electron_A"]     # Au0 is the lower one (z 2.26)
    shape = all(x > 0.1 for x in fl)
    print(f"H2 (shape): flat intra per electron {fl[0]:+.3f} / {fl[1]:+.3f} A -> " + ("lobes UP on both: the shape form holds" if shape else "NOT both > +0.1 A: the shape form is falsified"))
    print(f"   upright lower Au intra per electron {up:+.3f} A -> " + ("H1 axial lobe (down)" if up < -0.1 else "no axial lobe (H2 alone)" if abs(up) <= 0.1 else "lobe UP on the standing lower Au (unexpected under both)"))
    D = {g: R[g]["levels_eV"]["acceptor"] - R[g]["levels_eV"]["donor_top"] for g in R}
    r = D["flat"] / D["upright"]
    print(f"H3 (level): Delta flat {D['flat']:.3f} / upright {D['upright']:.3f} = {r:.2f}" + (f" / mgtop {D['mgtop']:.3f}" if "mgtop" in D else "") + " -> "
          + ("the LEVEL carries it (> 2)" if r > 2 else "the SHAPE carries it (< 1.3)" if r < 1.3 else "both, apportioned by the ratio"))
