"""Read a scored step against its pre-registered criteria (README), from the score files only (2026-10-19).

Usage: python read_criteria.py STEP        STEP in {13S, 14}; prints every number a criterion needs, the verdict,
and "missing" for any score file not yet written.  Comparators are read from their own score files, never retyped.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
B = "AuMgO_formal_kT_eps_cpl_alpha_exband_amp_"
RUNS = {"13": B + "bondlaw_r5_s{}", "12": "AuMgO_formal_kT_eps_cpl_alpha_exband_bondlaw_r5_s{}",
        "13S": B + "spin_bondlaw_r5_s{}", "14": B + "part_bondlaw_r5_s{}", "15": B + "part_pen_bondlaw_r5_s{}",
        "16": B + "part_pen_ind_bondlaw_r5_s{}", "17": B + "part_pen_ind_spol_bondlaw_r5_s{}",
        "18": B + "part_pen_ind_spol_fpriorq_bondlaw_r5_s{}", "19": B + "part_pen_ind_spol_bprior_bondlaw_r5_s{}",
        "20": B + "part_pen_ind_spol_pfp_bondlaw_r5_s{}", "21": B + "part_pen_ind_spol_pfp_anc_bondlaw_r5_s{}", "22": B + "part_pen_ind_spol_pfp_dfct_bondlaw_r5_s{}",
        "23": B + "part_pen_ind_spol_pfp_dfct_dec_bondlaw_r5_s{}", "24": B + "part_pen_ind_spol_pfp_dfct_dec_wall_bondlaw_r5_s{}",
        "25": B + "part_pen_ind_spol_pfp_bop_bondlaw_r5_s{}", "26": B + "part_pen_ind_spol_pfp_bopw_bondlaw_r5_s{}",
        "27": B + "part_pen_ind_spol_pfp_bops_bondlaw_r5_s{}",
        "28": B + "part_pen_ind_spol_pfp_fid_bondlaw_r5_s{}",
        "28a": B + "part_pen_ind_spol_pfp_fpa_bondlaw_r5_s{}",
        "28b": B + "part_pen_ind_spol_pfp_fpa_fid_bondlaw_r5_s{}"}
COMPARATOR = {"22": "20", "23": "22", "24": "23", "25": "20", "26": "25", "27": "26", "28": "20", "28a": "20", "28b": "28a"}     # the pre-registered comparators (README, steps 22-27)
PBE_PATH_A = {0.4: 0.255, 1.0: -0.197}                 # PBE along tilt path A (theory note, 2026-11-04; 25/28 ions)
DFT_WET_U, DFT_WET_D = 928.9, -2.7
DFT_T5 = -931.6


def text(run, name):
    path = os.path.join(HERE, "runs", run, "scores", name)
    return open(path).read() if os.path.exists(path) else None


def number(pattern, s, group=1):
    m = re.search(pattern, s or "")
    return float(m.group(group)) if m else None


def metrics(run):
    t5 = text(run, "T5_wetting.txt")
    out = {"run": run}
    if t5:
        out["wet_d"] = number(r"doped ([+-]?[0-9.]+) meV, undoped", t5)          # signed either way (13S s43: +35.2)
        out["wet_u"] = number(r"undoped ([+-]?[0-9.]+) meV", t5)
        if out["wet_d"] is not None and out["wet_u"] is not None:
            out["T5"] = out["wet_d"] - out["wet_u"]
        for cls in ("doped", "undoped"):
            m = re.search(rf"^\s+{cls}\s+n=\d+\s+E RMSE ([0-9.]+) meV/atom\s+F RMSE ([0-9.]+)", t5, re.M)
            if m:
                out[f"E_{cls}"], out[f"F_{cls}"] = float(m.group(1)), float(m.group(2))
    h1 = text(run, "H1_class_residuals.txt")
    out["offset_undoped"] = number(r"undoped .*flat - upright\s+(-?\+?[0-9.]+)", h1)
    out["offset_doped"] = number(r"\n\s+doped .*flat - upright\s+(-?\+?[0-9.]+)", h1)
    out["eps"] = number(r"eps_inf diag \[([0-9.]+)", text(run, "T6_eps_bound.txt"))
    out["alpha_au"] = number(r"Au atom\s+alpha =\s+([0-9.]+)", text(run, "A_au_polarizability.txt"))
    t4 = text(run, "T4_free_charges.txt")
    out["free_au2_doped"] = number(r"^doped\s+free:.*Au2 (-?[0-9.]+)", t4) if t4 else None
    if t4:
        m = re.search(r"^doped\s+free:.*Au2 (-?[0-9.]+)", t4, re.M)
        out["free_au2_doped"] = float(m.group(1)) if m else None
    out["delta_ion"] = number(r"Delta_ion = E\(\+1,-1\) - E\(0,0\) = ([+-][0-9.]+)", text(run, "D_dimer_ct.txt"))
    rw = text(run, "R_window_deficit.txt")
    if rw:
        m = re.search(r"undoped: class offset flat - upright at 2.5 eV ([+-]?[0-9.]+), at 8 eV ([+-]?[0-9.]+)", rw)
        if m:
            out["A1_window"] = abs(float(m.group(1)) - float(m.group(2)))
    gs = text(run, "G_gauge_split.txt")
    if gs:
        m = re.search(r"undoped\s+non-wetting.*p\(\+1\), p\(-1\): (.*)", gs)
        if m:
            ps = re.findall(r"/ ([0-9.]+) ([0-9.]+)", m.group(1))
            out["p_ion_upright"] = max(float(a) for a, _ in ps) if ps else None
        out["T5_Es"] = number(r"= E_s ([+-][0-9.]+)", gs)
        m = re.search(r"doped\s+non-wetting.*p\(\+1\), p\(-1\): ([^;]+); (.*)", gs)
        if m:                       # first Au listed is AU[0]; in the references AU[0] is the near (O-top) atom
            p0 = re.findall(r"/ ([0-9.]+) ([0-9.]+)", m.group(1)); p1 = re.findall(r"/ ([0-9.]+) ([0-9.]+)", m.group(2))
            if p0 and p1:
                a, b = float(p0[0][1]), float(p1[0][1])
                out["near_share_doped_upright"] = a / (a + b) if a + b > 0 else None
    return out


def fold_metrics(run, step):
    """The reference-geometry energies and (c1)-(c3) of probe_downfold_counterfactual.py (scores/D6_fold.txt): for a run
    trained with the fold its own configuration is the probe's 'downfold on' pass; for step 20 (no fold) it is 'as
    trained'.  The S5 relaxation share (wet_u row: the model's POL + CT against the PBE cluster's), the training log's
    DownfoldError count, and step 24's W1 verdict."""
    out = {}
    d6 = text(run, "D6_fold.txt")
    if d6:
        tag = "as trained" if step in ("20", "25", "26", "27") else "downfold on"      # steps without the fold
        m = re.search(rf"=== \S+ {tag}: wet_u ([+-][0-9.]+), wet_d ([+-][0-9.]+), T5 ([+-][0-9.]+) meV", d6)
        if m:
            out["ref_wet_u"], out["ref_wet_d"], out["ref_T5"] = (float(x) for x in m.groups())
        if step != "20":
            part = d6.split(f"{tag}:", 1)[1] if f"{tag}:" in d6 else ""
            for c in ("c1", "c2", "c3"):
                mm = re.search(rf"\({c}\)[^\n]*?(PASS|FAIL)", part)
                out[c] = None if not mm else mm.group(1) == "PASS"
            mm = re.search(r"flat undoped relaxed: heights \[([0-9., ]+)\] -> \[([0-9., ]+)\] A.*?relaxation ([+-][0-9.]+) meV", part)
            if mm:
                h0 = [float(x) for x in mm.group(1).split(",")]; h1 = [float(x) for x in mm.group(2).split(",")]
                out["flat_dh"] = max(abs(a - b) for a, b in zip(h0, h1)); out["flat_relax"] = float(mm.group(3))
    s5 = text(run, "S5_first_order.txt")
    if s5:
        mm = re.search(r"^\s+wet_u\s+([+-][0-9.]+)\s+([+-][0-9.]+)\s+([+-][0-9.]+)\s+([+-][0-9.]+)\s+([+-][0-9.]+)\s+[+-][0-9.]+\s+[+-][0-9.]+\s+[+-][0-9.]+\s+\|\s+([+-][0-9.]+)\s+([+-][0-9.]+)", s5, re.M)
        if mm:
            out["relax_model"], out["relax_pbe"] = float(mm.group(5)), float(mm.group(7))
    log = os.path.join(HERE, "runs", run, "log.txt")
    if os.path.exists(log):
        out["downfold_errors"] = open(log, errors="replace").read().count("DownfoldError")
    w1 = text(run, "W1_mgtop_curve.txt")
    if w1:
        out["W1"] = [ln.endswith("PASS") or "PASS (" in ln for ln in w1.splitlines() if ln.strip().startswith("(W1)")]
    tp = text(run, "T_tilt_path.txt")                   # step 25: probe_tilt_path.py --scan on path A, as trained
    if tp:
        vals = dict(re.findall(r"([+-][0-9.]+): ([+-][0-9.]+)", tp.splitlines()[-1] if tp.strip() else ""))
        out["tilt"] = {float(k): float(v) for k, v in vals.items()}
    ua = text(run, "T_upright_approach.txt")           # step 26: the upright dimer's approach, model - PBE per height
    if ua:
        m = re.search(r"model - PBE: (.*) eV", ua)
        if m:
            out["approach_dev"] = {float(k): float(v) for k, v in re.findall(r"([0-9.]+): ([+-][0-9.]+)", m.group(1))}
    h1t = text(run, "H1_class_residuals_train.txt")
    if h1t:
        d = {m.group(1): float(m.group(2)) for m in
             re.finditer(r"(doped|undoped)\s+mean residual.*flat - upright\s+([+-]?[0-9.]+)", h1t)}
        if "doped" in d and "undoped" in d:
            out["class_doped"], out["class_undoped"] = d["doped"], d["undoped"]
    return out


def fmt(x, spec=".1f"):
    return "missing" if x is None else format(x, spec)


def verdict(ok):
    return "missing" if ok is None else ("PASS" if ok else "FAIL")


def within(t5, frac):
    return None if t5 is None else abs(t5 - DFT_T5) <= frac * abs(DFT_T5)


def main():
    step = sys.argv[1]
    m = {s: metrics(RUNS[step].format(s)) for s in (42, 43)}
    for s in (42, 43):
        r = m[s]
        print(f"step {step} seed {s}: T5 {fmt(r.get('T5'))} meV ({fmt(None if r.get('T5') is None else 100 * r['T5'] / DFT_T5)} %),"
              f" wet d/u {fmt(r.get('wet_d'))}/{fmt(r.get('wet_u'))}; E d/u {fmt(r.get('E_doped'), '.3f')}/{fmt(r.get('E_undoped'), '.3f')}"
              f" meV/atom; F d/u {fmt(r.get('F_doped'), '.4f')}/{fmt(r.get('F_undoped'), '.4f')}; offsets d/u"
              f" {fmt(r.get('offset_doped'))}/{fmt(r.get('offset_undoped'))}; eps {fmt(r.get('eps'), '.3f')};"
              f" alpha(Au) {fmt(r.get('alpha_au'), '.3f')}; free Au2 (doped) {fmt(r.get('free_au2_doped'), '.3f')};"
              f" Delta_ion {fmt(r.get('delta_ion'), '+.3f')}; p_ion(upright) {fmt(r.get('p_ion_upright'), '.3f')};"
              f" doped upright near share {fmt(r.get('near_share_doped_upright'), '.2f')} (DFT 0.38)")
    t5 = [m[s].get("T5") for s in (42, 43)]
    spread = None if None in t5 else abs(t5[0] - t5[1])
    both15 = None if None in t5 else (within(t5[0], 0.15) and within(t5[1], 0.15))

    def b_criteria(r):
        b1 = None if r.get("free_au2_doped") is None else abs(r["free_au2_doped"] + 1.0) < 0.05
        b3 = None if r.get("offset_undoped") is None else abs(r["offset_undoped"]) <= 100.0
        a1 = None if r.get("A1_window") is None else r["A1_window"] <= 1.0
        return b1, b3, a1

    def no_worse(r, c, tol=0.10):
        keys = ("E_doped", "E_undoped", "F_doped", "F_undoped")
        if any(r.get(k) is None or c.get(k) is None for k in keys):
            return None
        return all(r[k] <= (1.0 + tol) * c[k] for k in keys)

    if step == "13S":
        c = {s: metrics(RUNS["13"].format(s)) for s in (42, 43)}
        c_t5 = [c[s].get("T5") for s in (42, 43)]
        c_spread = None if None in c_t5 else abs(c_t5[0] - c_t5[1])
        print(f"N1 T5 within 15 % for both: {verdict(both15)}")
        print(f"N2 spread {fmt(spread)} below step 13's {fmt(c_spread)}: {verdict(None if None in (spread, c_spread) else spread < c_spread)}")
        for s in (42, 43):
            print(f"N3 seed {s} E/F no worse than step 13 by > 10 %: {verdict(no_worse(m[s], c[s]))}")
            b1, b3, a1 = b_criteria(m[s])
            print(f"N4 seed {s}: B1 {verdict(b1)}, B3 {verdict(b3)}, A1 {verdict(a1)}")
    elif step == "14":
        d = [m[s].get("delta_ion") for s in (42, 43)]
        print(f"P1 |Delta_ion(42) - Delta_ion(43)| = {fmt(None if None in d else abs(d[0] - d[1]), '.3f')} eV <= 0.5: "
              f"{verdict(None if None in d else abs(d[0] - d[1]) <= 0.5)}")
        print(f"P2 T5 within 15 % for both: {verdict(both15)}")
        c12 = metrics(RUNS["12"].format(42))
        for s in (42, 43):
            r = m[s]
            p4 = None if r.get("eps") is None or r.get("alpha_au") is None else (2.5 <= r["eps"] <= 3.5 and abs(r["alpha_au"] - 5.435) <= 0.5435)
            b1, b3, a1 = b_criteria(r)
            print(f"P4 seed {s}: {verdict(p4)};  P5 seed {s}: B1 {verdict(b1)}, B3 {verdict(b3)}, A1 {verdict(a1)};"
                  f"  P6 seed {s} (against step 12 seed 42's own scores): {verdict(no_worse(r, c12))}")
    elif step in COMPARATOR:
        cs = COMPARATOR[step]
        c = {s: metrics(RUNS[cs].format(s)) for s in (42, 43)}
        f = {s: fold_metrics(RUNS[step].format(s), step) for s in (42, 43)}
        fc = {s: fold_metrics(RUNS[cs].format(s), cs) for s in (42, 43)}
        closer = lambda a, b: None if None in (a, b) else abs(a - DFT_WET_U) < abs(b - DFT_WET_U)
        for s in (42, 43):
            r, cc, ff, fcc = m[s], c[s], f[s], fc[s]
            e1_ref = closer(ff.get("ref_wet_u"), fcc.get("ref_wet_u"))
            e1_rel = closer(r.get("wet_u"), cc.get("wet_u"))
            print(f"E1 seed {s}: wet_u at the reference geometries {fmt(ff.get('ref_wet_u'))} against step {cs}'s "
                  f"{fmt(fcc.get('ref_wet_u'))} (DFT +928.9): {verdict(e1_ref)} [pre-registered]; T5 protocol (relaxed) "
                  f"{fmt(r.get('wet_u'))} against {fmt(cc.get('wet_u'))}: {verdict(e1_rel)}")
            e2 = None if None in (r.get("wet_d"), cc.get("wet_d")) else abs(r["wet_d"] - DFT_WET_D) <= abs(cc["wet_d"] - DFT_WET_D) + 50.0
            print(f"E2 seed {s}: wet_d {fmt(r.get('wet_d'))} against step {cs}'s {fmt(cc.get('wet_d'))} (DFT -2.7; "
                  f"not further by > 50 meV): {verdict(e2)}; at the references {fmt(ff.get('ref_wet_d'))}")
            e3 = None if None in (ff.get("relax_model"), ff.get("relax_pbe")) else ff["relax_model"] >= ff["relax_pbe"] / 3.0
            print(f"E3 seed {s}: the model's wet_u relaxation (POL + CT) {fmt(ff.get('relax_model'))} against PBE's "
                  f"{fmt(ff.get('relax_pbe'))} (above one third): {verdict(e3)}")
            e4 = within(r.get("T5"), 0.10)
            print(f"E4 seed {s}: T5 {fmt(r.get('T5'))} ({fmt(None if r.get('T5') is None else 100 * r['T5'] / DFT_T5)} %; "
                  f"90-110 %): {verdict(e4)}; at the references {fmt(ff.get('ref_T5'))}")
            e5_parts = [ff.get("downfold_errors") == 0 if ff.get("downfold_errors") is not None else None,
                        ff.get("c1"), ff.get("c2"), ff.get("c3"),
                        None if ff.get("flat_dh") is None else ff["flat_dh"] <= 0.3]
            e5 = None if None in e5_parts else all(e5_parts)
            print(f"E5 seed {s}: DownfoldError in training {fmt(ff.get('downfold_errors'), '.0f')}; (c1) {verdict(ff.get('c1'))}, "
                  f"(c2) {verdict(ff.get('c2'))}, (c3) {verdict(ff.get('c3'))}; flat dimer moved {fmt(ff.get('flat_dh'), '.2f')} A "
                  f"({fmt(ff.get('flat_relax'))} meV): {verdict(e5)}")
            print(f"E6 seed {s}: E / F RMSE within 10 % of step {cs}'s: {verdict(no_worse(r, cc))}")
            if step == "24":
                w1 = ff.get("W1")
                print(f"W1 seed {s}: dimer-over-Mg curve within 30 % of PBE at 2.0 / 2.3 A: {verdict(None if not w1 else w1[0])}; "
                      f"W2 = E5's flat-dimer part: {verdict(None if ff.get('flat_dh') is None else ff['flat_dh'] <= 0.3)}")
            if step in ("25", "26", "27"):
                b1 = None if ff.get("flat_dh") is None else ff["flat_dh"] <= 0.3
                tilt = ff.get("tilt") or {}
                b2 = None if not all(k in tilt for k in PBE_PATH_A) else (abs(tilt[0.4] - PBE_PATH_A[0.4]) <= 0.1
                                                                           and abs(tilt[1.0] - PBE_PATH_A[1.0]) <= 0.2)
                cd, cu = ff.get("class_doped"), ff.get("class_undoped")
                b3 = None if None in (cd, cu) else abs(cd - cu) <= 50.0
                print(f"B1 seed {s}: flat dimer relaxed within 0.3 A: {verdict(b1)} (moved {fmt(ff.get('flat_dh'), '.2f')} A, "
                      f"{fmt(ff.get('flat_relax'))} meV)")
                print(f"B2 seed {s}: tilt path A barrier {fmt(tilt.get(0.4), '.3f')} (PBE +0.255, within 0.1) and end "
                      f"{fmt(tilt.get(1.0), '.3f')} (PBE -0.197, within 0.2) eV: {verdict(b2)}")
                print(f"B3 seed {s}: training-set class offsets doped {fmt(cd)} / undoped {fmt(cu)} meV, doped - undoped "
                      f"{fmt(None if None in (cd, cu) else cd - cu)} (within 50): {verdict(b3)}; common offset "
                      f"{fmt(None if None in (cd, cu) else 0.5 * (cd + cu))}")
                if step == "27":
                    dev = ff.get("approach_dev") or {}
                    c4 = None if not all(h in dev for h in (2.0, 2.5)) else all(abs(dev[h]) <= 0.2 for h in (2.0, 2.5))
                    d1 = None if None in (r.get("wet_d"), cc.get("wet_d")) else (abs(r["wet_d"] - DFT_WET_D) <= abs(cc["wet_d"] - DFT_WET_D) - 50.0
                                                                                  or abs(r["wet_d"] - DFT_WET_D) <= 50.0)
                    cdc, cuc = fcc.get("class_doped"), fcc.get("class_undoped")
                    d2 = None if None in (cd, cu, cdc) else (abs(cd - cu) <= 50.0 and abs(cd) <= abs(cdc) + 30.0)
                    print(f"C4 seed {s}: upright approach, model - PBE at 2.0 / 2.5 A {fmt(dev.get(2.0), '.3f')} / {fmt(dev.get(2.5), '.3f')} eV: {verdict(c4)}")
                    print(f"D1 seed {s}: wet_d {fmt(r.get('wet_d'))} against step {cs}'s {fmt(cc.get('wet_d'))} (DFT -2.7; closer by >= 50 or within 50): {verdict(d1)}")
                    print(f"D2 seed {s}: doped offset {fmt(cd)} (step {cs}'s {fmt(cdc)}), doped - undoped {fmt(None if None in (cd, cu) else cd - cu)}: {verdict(d2)}")
                    print(f"  PASS seed {s} (B1 and B2 and D1 and D2): {verdict(None if None in (b1, b2, d1, d2) else (b1 and b2 and d1 and d2))}")
                    continue
                if step == "26":
                    dev = ff.get("approach_dev") or {}
                    c4 = None if not all(h in dev for h in (2.0, 2.5)) else all(abs(dev[h]) <= 0.2 for h in (2.0, 2.5))
                    print(f"C4 seed {s}: upright approach, model - PBE at 2.0 / 2.5 A {fmt(dev.get(2.0), '.3f')} / "
                          f"{fmt(dev.get(2.5), '.3f')} eV (within 0.2): {verdict(c4)}; at 1.8 {fmt(dev.get(1.8), '.3f')}")
                    print(f"  PASS seed {s} (B1 and B2 and B3 and C4): "
                          f"{verdict(None if None in (b1, b2, b3, c4) else (b1 and b2 and b3 and c4))}")
                    continue
                print(f"  PASS seed {s} (B1 and B2 and B3): {verdict(None if None in (b1, b2, b3) else (b1 and b2 and b3))}")
                continue
            print(f"  PASS seed {s} (E1 and E5{' and W1' if step == '24' else ''}): "
                  f"{verdict(None if None in (e1_ref, e5) else (e1_ref and e5 and (step != '24' or bool(ff.get('W1') and ff['W1'][0]))))}")
    else:
        print(f"seed spread {fmt(spread)}; both within 15 %: {verdict(both15)}")


if __name__ == "__main__":
    main()
