# Written on 2026-10-30 (readings P24a / P24b of the O-top ALMO series; theory note 4c); run from the repo root.
"""P24a / P24b readings of the O-top series (theory note 4c, 2026-10-30 11:52): v_eff(h) = sqrt(|CT_raw| Delta) against
Harrison / exponential / saturation below the anchor, and the sigma* shift delta(h) = Delta(h) - 1.50 fitted to an
overlap-like exponential and to 1/h^n."""
import json, os, math, numpy as np
W = os.environ.get("TRACE_FP_CACHE", "fp_cache")   # first-principles cache (TRACE_FP_CACHE)
pts = {2.26: ("acceptor_density/acceptor_upright.json")}
for h in (2.00, 2.60, 3.00):
    pts[h] = f"otop_levels/acceptor_upright_h{h:.2f}.json"
rows = []
for h, f in sorted(pts.items()):
    path = f"{W}/{f}"
    if not os.path.exists(path):
        print(f"h {h:.2f}: not in"); continue
    j = json.load(open(path)); L = j["levels_eV"]; ct = -j["CT_raw_B->A_eV"]; D = L["acceptor"] - L["donor_top"]
    rows.append((h, ct, D, math.sqrt(ct * D), L["acceptor"], L["donor_top"]))
    print(f"h {h:.2f}: CT_raw {ct:.4f} eV, Delta {D:.3f} eV (acceptor {L['acceptor']:+.3f}, donor top {L['donor_top']:+.3f}), v_eff {math.sqrt(ct * D):.3f} eV")
by = {r[0]: r for r in rows}
if 2.00 in by and 2.26 in by:
    ratio = by[2.00][3] / by[2.26][3]
    harr = (2.26 ** 2 + 0.25) / (2.0 ** 2 + 0.25); expo = math.exp(1.13 * 0.26)
    print(f"(P24a) v_eff(2.0)/v_eff(2.26) = {ratio:.3f}; Harrison {harr:.3f}, exponential {expo:.3f}, saturation <= 1.1 -> "
          + ("growth holds below the anchor" if ratio > 1.2 else "SATURATION: an overlap-saturating form" if ratio < 1.1 else "between"))
if len(rows) >= 3:
    hs = np.array([r[0] for r in rows]); d = np.array([r[2] - 1.50 for r in rows])
    pos = d > 0
    if pos.sum() >= 3:
        A = np.vstack([np.ones(pos.sum()), -hs[pos]]).T; c = np.linalg.lstsq(A, np.log(d[pos]), rcond=None)[0]
        res_exp = np.sqrt(np.mean((np.log(d[pos]) - A @ c) ** 2)); lam = 1.0 / c[1] if c[1] != 0 else float("inf")
        B = np.vstack([np.ones(pos.sum()), -np.log(hs[pos])]).T; c2 = np.linalg.lstsq(B, np.log(d[pos]), rcond=None)[0]
        res_pow = np.sqrt(np.mean((np.log(d[pos]) - B @ c2) ** 2))
        print(f"(P24b) shift delta(h): {[(round(float(h_), 2), round(float(x), 3)) for h_, x in zip(hs, d)]}; exponential lambda {lam:.2f} A (log-residual {res_exp:.3f}) vs power h^-{c2[1]:.2f} (residual {res_pow:.3f}) -> "
              + ("overlap-like" if res_exp < res_pow else "power-law"))
    else:
        print(f"(P24b) shift delta(h): {[(round(float(h_), 2), round(float(x), 3)) for h_, x in zip(hs, d)]} (fewer than three positive points)")
