"""The phonon benchmark's scorer (experiments/benchmarks_phonons/score_phonons.py) and its P5 protocol.

Pre-registered 2026-11-16 (experiments/benchmarks_phonons/README.md, "Pre-registered: P5 on PBE's own 512-atom
supercell for every step after 34b"; decision of 2026-11-16):
  * without --p5-supercell the scorer is the protocol of steps up to 34b, line for line (scripts that parse its
    output rely on that);
  * --p5-supercell N is accepted only when N^3 is the reference's own force-constant supercell (PBE, Togo: 4; PBEsol,
    phonondb mp-1265: 2, already the default's) and N is a larger multiple of --supercell;
  * then P5 is read on N^3 (written to <out>_scN), P0 must hold on both supercells, and P5 is FAIL unless the two
    supercells' bands without NAC agree within 1 cm^-1 at every path point commensurate with --supercell (C1);
    P1-P4 and P6 are read on --supercell as before; P5 on --supercell and P6 on N^3 are printed as records.
No phonon calculation runs here: mgo_phonons.py is replaced by a stand-in that writes the reference's own bands shifted
by known amounts, so every expected RMS is known in closed form."""
import copy
import importlib.util
import math
import os
import sys
import types

import numpy as np
import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PH = os.path.join(ROOT, "experiments", "benchmarks_phonons")


def _load_scorer():
    spec = importlib.util.spec_from_file_location("score_phonons_under_test", os.path.join(PH, "score_phonons.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sp = _load_scorer()
THZ = 33.35641
REF_NONAC = os.path.join(PH, "reference", "MgO_PBE_togo_band_nonac.yaml")
REF_NAC = os.path.join(PH, "reference", "MgO_PBE_togo_band_nac.yaml")
# the path points commensurate with the 2x2x2 conventional supercell, named: Gamma (three times), the middle of
# Gamma-X, X (twice) and L; with 4x4x4 also K (twice) and the middle of Gamma-L
SHARED = {(0.0, 0.0, 0.0), (0.25, 0.0, 0.25), (0.5, 0.0, 0.5), (0.5, 0.5, 0.5)}
ONLY4 = {(0.375, 0.375, 0.75), (0.25, 0.25, 0.25)}
LOG = ("Z*(Mg) diag [{Z} {Z} {Z}]\neps_inf diag [3.300 3.300 3.300]\n"
       "Gamma: TO {TO}, LO 670.0 cm^-1\nmin frame population {pop}\nmax |F - E_frame| {fe} eV\n")
GOOD = {"Z": "1.950", "TO": "380.0", "pop": "1.000000", "fe": "0.0002"}


def _log(**kw):
    return LOG.format(**{**GOOD, **kw})


def _qs(path):
    return [tuple(float(x) for x in p["q-position"]) for p in yaml.safe_load(open(path))["phonon"]]


def _mask(named):
    return np.array([q in named for q in _qs(REF_NONAC)])


def _write_band(src, dst, shift_cm, move_q=None):
    """The reference band with every frequency at path point i shifted by shift_cm[i] (cm^-1): a shift common to the
    six branches keeps their order, so the RMS against the reference is exactly sqrt(mean(shift_cm**2)).  move_q = i
    moves path point i's q (to build a file whose path differs)."""
    d = copy.deepcopy(yaml.safe_load(open(src)))
    for p, s in zip(d["phonon"], shift_cm):
        for b in p["band"]:
            b["frequency"] = b["frequency"] + float(s) / THZ
    if move_q is not None:
        d["phonon"][move_q]["q-position"] = [x + 0.001 for x in d["phonon"][move_q]["q-position"]]
    yaml.safe_dump(d, open(dst, "w"))


def _write_out(out, nonac_shift, nac_shift, move_q=None, **log):
    os.makedirs(out, exist_ok=True)
    _write_band(REF_NONAC, os.path.join(out, "band_nonac.yaml"), nonac_shift, move_q)
    _write_band(REF_NAC, os.path.join(out, "band_nac.yaml"), nac_shift)
    open(os.path.join(out, "mgo_phonons.log"), "w").write(_log(**log))


def _shifts():
    m = _mask(SHARED)
    n = len(m)
    small = np.where(m, 5.0, 30.0)          # 2x2x2: right where it is exact, its own interpolation elsewhere
    return m, n, small, np.full(n, 5.0)      # 4x4x4: the reference's interpolation, offset 5 everywhere


def _run_main(monkeypatch, capsys, tmp_path, argv_extra, by_supercell, out=None):
    """main() with mgo_phonons.py replaced: by_supercell[N] = (nonac_shift, nac_shift, log keywords)."""
    calls = []

    def fake_run(cmd, capture_output, text, cwd):
        n = int(cmd[cmd.index("--supercell") + 1])
        o = cmd[cmd.index("--out") + 1]
        calls.append((n, o))
        nonac, nac, log = by_supercell[n]
        os.makedirs(o, exist_ok=True)
        _write_band(REF_NONAC, os.path.join(o, "band_nonac.yaml"), nonac)
        _write_band(REF_NAC, os.path.join(o, "band_nac.yaml"), nac)
        return types.SimpleNamespace(stdout=_log(**log))

    monkeypatch.setattr(sp.subprocess, "run", fake_run)
    out = out or str(tmp_path / "o")
    monkeypatch.setattr(sys, "argv", ["score_phonons.py", str(tmp_path / "RUN" / "model.pt"), "--out", out] + argv_extra)
    sp.main()
    return calls, out, capsys.readouterr().out


def _crit(crit, tag):
    return [c for c in crit if c[0].startswith(tag)][0]


def test_commensurate_points_of_the_path():
    m2 = sp.commensurate(REF_NONAC, 2)
    m4 = sp.commensurate(REF_NONAC, 4)
    qs = _qs(REF_NONAC)
    assert {qs[i] for i in np.nonzero(m2)[0]} == SHARED and int(m2.sum()) == 7
    assert {qs[i] for i in np.nonzero(m4 & ~m2)[0]} == ONLY4 and int(m4.sum()) == 10
    assert np.all(m4[m2])


def test_legacy_protocol_is_unchanged(monkeypatch, capsys, tmp_path):
    """All nine lines, as the scorer of steps up to 34b printed them (HEAD of 2026-11-16)."""
    m, n, small, _ = _shifts()
    calls, out, text = _run_main(monkeypatch, capsys, tmp_path, [], {2: (small, np.full(n, 7.0), {})})
    assert calls == [(2, out)]
    rms_small = math.sqrt((small ** 2).mean())
    assert text.splitlines() == [
        "=== phonon benchmark, RUN, reference pbe, a = 4.25566 A, force-constant supercell 2^3",
        "  PASS  P0 ensemble in the frame on every displaced supercell (population >= 1 - 1e-4, |F - E_frame| <= 1 meV): "
        "population 1.000000, |F - E_frame| 0.0002 eV",
        "  PASS  P1 Z*(Mg) within 1.90-2.10 (pbe 1.9923): 1.950",
        "  PASS  P2 eps_inf within 2.9-3.5 (pbe 3.235): 3.300",
        "  PASS  P3 TO within 3 % of pbe 373.3: 380.0 (+1.8 %)",
        "  PASS  P4 LO within 660-700 (pbe 681.3): 670.0",
        f"  FAIL  P5 path RMS without NAC <= 15 cm^-1: {rms_small:.1f}",
        "  PASS  P6 path RMS with NAC <= 20 cm^-1: 7.0",
        f"  6 / 7 passed; log and band yaml in {out}"]


def test_legacy_labels_and_thresholds(monkeypatch, capsys, tmp_path):
    m, n, small, _ = _shifts()
    _, _, text = _run_main(monkeypatch, capsys, tmp_path, [], {2: (small, np.full(n, 7.0), {"TO": "386.0"})})
    assert "  FAIL  P3 TO within 3 % of pbe 373.3: 386.0 (+3.4 %)" in text.splitlines()
    _, _, text = _run_main(monkeypatch, capsys, tmp_path, ["--ref", "pbesol"], {2: (small, np.full(n, 7.0), {})})
    lines = text.splitlines()
    assert lines[0] == "=== phonon benchmark, RUN, reference pbesol, a = 4.21726 A, force-constant supercell 2^3"
    assert lines[2] == "  PASS  P1 Z*(Mg) within 1.90-2.10 (pbesol 1.9791): 1.950"
    assert lines[5] == "  FAIL  P4 LO within 673-713 (pbesol 693.9): 670.0"


def test_p5_read_on_the_large_supercell(monkeypatch, capsys, tmp_path):
    m, n, small, large = _shifts()
    calls, out, text = _run_main(monkeypatch, capsys, tmp_path, ["--p5-supercell", "4"],
                                 {2: (small, np.full(n, 7.0), {}), 4: (large, np.full(n, 9.0), {})})
    assert calls == [(2, out), (4, out + "_sc4")]
    rms_small = math.sqrt((small ** 2).mean())
    assert text.splitlines() == [
        "=== phonon benchmark, RUN, reference pbe, a = 4.25566 A, force-constant supercell 2^3; P5 on 4^3 (the reference's own: 4^3)",
        "  PASS  P0 ensemble in the frame on every displaced supercell of 2^3 and 4^3 (population >= 1 - 1e-4, "
        "|F - E_frame| <= 1 meV): population 1.000000, |F - E_frame| 0.0002 eV",
        "  PASS  P1 Z*(Mg) within 1.90-2.10 (pbe 1.9923): 1.950",
        "  PASS  P2 eps_inf within 2.9-3.5 (pbe 3.235): 3.300",
        "  PASS  P3 TO within 3 % of pbe 373.3: 380.0 (+1.8 %)",
        "  PASS  P4 LO within 660-700 (pbe 681.3): 670.0",
        "  PASS  P5 path RMS without NAC <= 15 cm^-1, force constants on 4^3 (C1 <= 1 cm^-1): 5.0 (C1 0.000)",
        "  PASS  P6 path RMS with NAC <= 20 cm^-1: 7.0",                                 # P6 still on 2x2x2
        f"  7 / 7 passed; log and band yaml in {out} and {out}_sc4",
        f"  recorded, not criteria: P5 on 2^3 {rms_small:.1f}; P6 on 4^3 9.0"]


def test_large_output_is_a_sibling_even_with_a_trailing_slash(monkeypatch, capsys, tmp_path):
    m, n, small, large = _shifts()
    out = str(tmp_path / "o") + os.sep
    calls, _, text = _run_main(monkeypatch, capsys, tmp_path, ["--p5-supercell", "4"],
                               {2: (small, np.full(n, 7.0), {}), 4: (large, np.full(n, 9.0), {})}, out=out)
    assert calls == [(2, out), (4, str(tmp_path / "o_sc4"))]
    assert f"and {tmp_path / 'o_sc4'}" in text


def test_c1_is_a_maximum_and_its_failure_fails_p5(tmp_path):
    m, n, small, large = _shifts()
    at_L = _mask({(0.5, 0.5, 0.5)})
    _write_out(str(tmp_path / "a"), small, np.full(n, 7.0))
    _write_out(str(tmp_path / "a_sc4"), np.where(at_L, 7.0, 5.0), np.full(n, 9.0))   # 2 cm^-1 off at L only
    crit, records = sp.criteria(str(tmp_path / "a"), sp.REFS["pbe"], str(tmp_path / "a_sc4"), 2, 4)
    p5 = _crit(crit, "P5")
    assert not p5[1] and abs(records["C1"] - 2.0) < 1e-6
    assert p5[2] == f"{math.sqrt((np.where(at_L, 7.0, 5.0) ** 2).mean()):.1f} (C1 2.000)"   # the RMS alone would pass


def test_p5_fails_on_its_rms_when_c1_holds(tmp_path):
    m, n, small, large = _shifts()
    _write_out(str(tmp_path / "a"), np.where(m, 20.0, 30.0), np.full(n, 7.0))
    _write_out(str(tmp_path / "a_sc4"), np.full(n, 20.0), np.full(n, 9.0))
    crit, records = sp.criteria(str(tmp_path / "a"), sp.REFS["pbe"], str(tmp_path / "a_sc4"), 2, 4)
    p5 = _crit(crit, "P5")
    assert not p5[1] and p5[2] == "20.0 (C1 0.000)" and records["C1"] < 1e-6


@pytest.mark.parametrize("small_log, large_log, value", [
    ({"pop": "0.990000"}, {}, "population 0.990000, |F - E_frame| 0.0002 eV"),
    ({}, {"pop": "0.990000"}, "population 0.990000, |F - E_frame| 0.0002 eV"),
    ({"fe": "0.0020"}, {}, "population 1.000000, |F - E_frame| 0.0020 eV"),
    ({}, {"fe": "0.0020"}, "population 1.000000, |F - E_frame| 0.0020 eV")])
def test_p0_must_hold_on_both_supercells(tmp_path, small_log, large_log, value):
    m, n, small, large = _shifts()
    _write_out(str(tmp_path / "a"), small, np.full(n, 7.0), **small_log)
    _write_out(str(tmp_path / "a_sc4"), large, np.full(n, 9.0), **large_log)
    p0 = _crit(sp.criteria(str(tmp_path / "a"), sp.REFS["pbe"], str(tmp_path / "a_sc4"), 2, 4)[0], "P0")
    assert not p0[1] and p0[2] == value
    if not small_log:                                                                   # the small one alone passes
        assert _crit(sp.criteria(str(tmp_path / "a"), sp.REFS["pbe"], None, 2, None)[0], "P0")[1]


def test_p1_to_p4_are_read_on_the_small_supercell(tmp_path):
    m, n, small, large = _shifts()
    _write_out(str(tmp_path / "a"), small, np.full(n, 7.0))
    _write_out(str(tmp_path / "a_sc4"), large, np.full(n, 9.0), Z="2.500", TO="300.0")
    crit, _ = sp.criteria(str(tmp_path / "a"), sp.REFS["pbe"], str(tmp_path / "a_sc4"), 2, 4)
    assert _crit(crit, "P1")[1:] == (True, "1.950") and _crit(crit, "P3")[1:] == (True, "380.0 (+1.8 %)")


def test_the_two_supercells_must_share_the_reference_path(tmp_path):
    m, n, small, large = _shifts()
    _write_out(str(tmp_path / "a"), small, np.full(n, 7.0))
    _write_out(str(tmp_path / "a_sc4"), large, np.full(n, 9.0), move_q=100)
    with pytest.raises(ValueError, match="q path"):
        sp.criteria(str(tmp_path / "a"), sp.REFS["pbe"], str(tmp_path / "a_sc4"), 2, 4)


@pytest.mark.parametrize("argv, message", [
    (["--p5-supercell", "3"], "--p5-supercell 3 is not the pbe reference's own force-constant supercell (4^3)"),
    (["--p5-supercell", "2"], "--p5-supercell 2 is not the pbe reference's own force-constant supercell (4^3)"),
    (["--p5-supercell", "6"], "--p5-supercell 6 is not the pbe reference's own force-constant supercell (4^3)"),
    (["--ref", "pbesol", "--p5-supercell", "4"], "--p5-supercell 4 is not the pbesol reference's own force-constant supercell (2^3)"),
    (["--ref", "pbesol", "--p5-supercell", "2"], "--p5-supercell 2 must be a multiple of --supercell 2 and larger"),
    (["--supercell", "0", "--p5-supercell", "4"], "--p5-supercell 4 must be a multiple of --supercell 0 and larger"),
    (["--supercell", "3", "--p5-supercell", "4"], "--p5-supercell 4 must be a multiple of --supercell 3 and larger"),
    (["--supercell", "4", "--p5-supercell", "4"], "--p5-supercell 4 must be a multiple of --supercell 4 and larger")])
def test_p5_supercell_refused_before_any_run(monkeypatch, capsys, tmp_path, argv, message):
    """Only the reference's own supercell is like-for-like, and it must contain --supercell's commensurate points."""
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, capsys, tmp_path, argv, {})
    assert message in capsys.readouterr().err
    assert not os.path.exists(tmp_path / "o" / "band_nonac.yaml")
