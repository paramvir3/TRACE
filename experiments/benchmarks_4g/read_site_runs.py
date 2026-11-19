# Written on 2026-10-29 (the measurement scripts of the theory note's section 4c); run from the repo root.
"""Readings (A) and (B) of the site runs against the pre-registered thresholds (theory note §4c, 2026-10-29).
Per-contact values; the distance correction uses the measured O-top decay (3.00 -> 3.50 A: e^{-1.64/A}) and, in
brackets, the band sum's own factor (3.26 / 3.59: 1.774)."""
import json, os, math
W = os.environ.get("TRACE_FP_CACHE", "fp_cache")   # first-principles cache (TRACE_FP_CACHE)
def ct(path):
    return -json.load(open(path))["CT_cp_surface->Au"]
otop = {h: ct(f"{W}/otop_height_ct/upright_h{h:.2f}.json") for h in (2.60, 3.00, 3.50)}
k = math.log(otop[3.00] / otop[3.50]) / 0.5
print(f"O-top single contact: {otop}; decay {k:.2f} /A")
def at(r, r0=3.50):  # the O-top contact brought from 3.50 A to r
    return otop[3.50] * math.exp(-k * (r - r0))
mg = ct(f"{W}/site_direction/vertical_mgtop.json"); mg_pc = mg / 2
print(f"standing over Mg0: {mg:.4f} eV, 2 contacts at 3.59 A -> per contact {mg_pc:.4f}; / O-top at 3.59 ({at(3.59):.4f}) = {mg_pc / at(3.59):.3f}  (band symmetry predicts 0.81)")
fr = f"{W}/site_direction/vertical_flat_ref.json"
if os.path.exists(fr):
    fl = ct(fr); fl_pc = fl / 4
    ratio_raw = mg_pc / fl_pc; ratio_eq = (mg_pc / at(3.59)) / (fl_pc / at(3.505))
    print(f"(A) flat reference h 2.89: {fl:.4f} eV, 4 contacts at 3.50/3.51 A -> per contact {fl_pc:.4f}; / O-top at 3.505 = {fl_pc / at(3.505):.3f}")
    print(f"    standing-over-Mg0 / lying per contact: raw {ratio_raw:.2f}, at equal r {ratio_eq:.2f} -> "
          + ("ORIENTATION carries the suppression (> 1.3)" if ratio_eq > 1.3 else "the SITE does (<= 1.3)" if ratio_eq > 0.77 else "lying contacts STRONGER (< 0.77): neither"))
else:
    print("(A) flat reference not in yet")
hp = f"{W}/site_direction/vertical_hollow_pp.json"
if os.path.exists(hp):
    ho = ct(hp); ho_pc = ho / 2
    r_decay = (ho_pc / at(3.26)) / (mg_pc / at(3.59)); r_band = (ho_pc / mg_pc) / 1.774
    print(f"(B) hollow_pp h 2.89: {ho:.4f} eV, 2 contacts at 3.26 A -> per contact {ho_pc:.4f}; / O-top at 3.26 = {ho_pc / at(3.26):.3f}")
    print(f"    hollow_pp / Mg-top per contact at equal r: {r_decay:.2f} [band factor: {r_band:.2f}] (site symmetry predicts 1.00) -> "
          + ("the cation below SUPPRESSES beyond symmetry (> 1.3)" if r_decay > 1.3 else "no cation effect beyond symmetry (0.8-1.25)" if r_decay >= 0.8 else "the cation ENHANCES (< 0.8)"))
else:
    print("(B) hollow_pp not in yet")
