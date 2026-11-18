"""Phonon benchmark of a checkpoint on bulk MgO against PBE DFT (Togo's phono3py PBE set): runs
tools/electrostatics/mgo_phonons.py at the PBE lattice constant, then scores the pre-registered criteria of README.md.
Usage (repo root): python experiments/benchmarks_phonons/score_phonons.py runs/RUN/model.pt [--supercell 2] [--p5-supercell 4]

--p5-supercell N: the protocol pre-registered on 2026-11-16 for every step after 34b (README, "Pre-registered: P5 on
PBE's own 512-atom supercell for every step after 34b"). N must be the reference's own force-constant supercell
(REFS[...]["fc2_supercell"]: PBE 4, i.e. 512 atoms; PBEsol 2, which the default already uses, so the flag is refused
for it) and a larger multiple of --supercell. P5 is then read with the model's force constants on the N^3 conventional
supercell, written to <out>_scN. That matches the reference band without NAC in supercell; the displacement amplitude
still differs (the model 0.01 A, Togo's PBE set 0.03 A; effect on P5 unmeasured). P0 must hold on both supercells.
P5 is FAIL unless C1 holds: the two supercells' bands without NAC agree within 1 cm^-1 at every path point commensurate
with --supercell, where both are exact. P1-P4 and P6 are read on --supercell as before. P5 on --supercell and P6 on N^3
are printed as records, not criteria.
Without --p5-supercell the output is the protocol of steps up to 34b, line for line (the watchers parse it), and stays so."""
import argparse, os, re, subprocess, sys
import numpy as np, yaml
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
A_PBE = 4.255656884758435
REFS = {   # the DFT reference sets; the P4 window is the reference LO -21 / +19 cm^-1 in both (pre-registered 2026-11-06 / -07)
    # fc2_supercell: the reference's own force-constant supercell, in repeats of the 8-atom conventional cell (checked
    # 2026-11-16 in the raw downloads ($TRACE_PHONON_RAW): Togo's phono3py_params.yaml phonon_supercell_matrix diag(4, 4, 4), 512
    # atoms, displacements 0.03 A; phonondb mp-1265 phonopy_params.yaml supercell_matrix diag(2, 2, 2), 64 atoms, 0.01 A)
    "pbe": {"a": A_PBE, "TO": 373.3, "LO": 681.3, "Z": 1.9923, "eps": 3.235, "LO_window": (660.0, 700.0), "fc2_supercell": 4,
            "band_nac": os.path.join(HERE, "reference", "MgO_PBE_togo_band_nac.yaml"),
            "band_nonac": os.path.join(HERE, "reference", "MgO_PBE_togo_band_nonac.yaml")},
    "pbesol": {"a": 4.21725941, "TO": 391.0, "LO": 693.9, "Z": 1.9791, "eps": 3.2417, "LO_window": (673.0, 713.0), "fc2_supercell": 2,
               "band_nac": os.path.join(HERE, "reference", "MgO_PBEsol_phonondb_band_nac.yaml"),
               "band_nonac": os.path.join(HERE, "reference", "MgO_PBEsol_phonondb_band_nonac.yaml")},
}
REF = REFS["pbe"]
THZ = 33.35641
C1_TOL = 1.0   # cm^-1; measured 0.0011-0.0050 on five step 34b checkpoints (README, "Why P5 depends on the supercell")


def band(path):
    d = yaml.safe_load(open(path)); return np.sort(np.array([[b["frequency"] for b in p["band"]] for p in d["phonon"]]) * THZ, axis=1)


def commensurate(path, n):
    """Path points of a band yaml whose q fits the n x n x n conventional supercell: q . (n a e_k) is an integer for
    k = x, y, z. The cubic constant a follows from the fcc primitive cell's volume, a^3 / 4, whatever the order of its
    vectors. The 1e-4 tolerance is set by the reciprocal lattice, which phonopy prints to 8 digits; on the reference
    path the nearest non-commensurate point is 0.01 away for n = 2 and 0.02 for n = 4."""
    d = yaml.safe_load(open(path))
    qc = np.array([p["q-position"] for p in d["phonon"]]) @ np.array(d["reciprocal_lattice"])   # 1/Angstrom, no 2 pi
    x = n * (4 * abs(np.linalg.det(np.array(d["lattice"])))) ** (1 / 3) * qc
    return np.abs(x - np.round(x)).max(1) < 1e-4


def parse_log(log):
    g = lambda pat: float(re.search(pat, log).group(1))
    return {"Z": g(r"Z\*\(Mg\) diag \[([-\d.]+)"), "eps": g(r"eps_inf diag \[([-\d.]+)"), "TO": g(r"TO ([-\d.]+), LO"),
            "LO": g(r"LO ([-\d.]+) cm"), "pop": g(r"min frame population ([\d.]+)"), "fe": g(r"max \|F - E_frame\| ([-\d.]+) eV")}


def rms(path, ref_path):
    return float(np.sqrt(((band(path) - band(ref_path)) ** 2).mean()))


def criteria(out, ref, out_p5=None, n=2, n_p5=None, ref_name="pbe"):
    """P0-P6 from mgo_phonons.py's outputs in out (force constants on n^3) and, for the protocol of 2026-11-16, in out_p5
    (on n_p5^3). Returns ([(name, ok, value)], records); records is empty without out_p5."""
    L = parse_log(open(os.path.join(out, "mgo_phonons.log")).read())
    rms0 = rms(os.path.join(out, "band_nonac.yaml"), ref["band_nonac"])
    rms1 = rms(os.path.join(out, "band_nac.yaml"), ref["band_nac"])
    in_frame = lambda l: l["pop"] >= 1 - 1e-4 and l["fe"] <= 1e-3
    p0 = ("P0 ensemble in the frame on every displaced supercell (population >= 1 - 1e-4, |F - E_frame| <= 1 meV)",
          in_frame(L), f"population {L['pop']:.6f}, |F - E_frame| {L['fe']:.4f} eV")
    p5 = ("P5 path RMS without NAC <= 15 cm^-1", rms0 <= 15, f"{rms0:.1f}")
    records = {}
    if out_p5 is not None:
        qref = np.array([p["q-position"] for p in yaml.safe_load(open(ref["band_nonac"]))["phonon"]])
        for f in (os.path.join(o, f"band_{k}.yaml") for o in (out, out_p5) for k in ("nonac", "nac")):
            q = np.array([p["q-position"] for p in yaml.safe_load(open(f))["phonon"]])
            if q.shape != qref.shape or not np.allclose(q, qref, atol=1e-9):
                raise ValueError(f"{f}: its q path is not the reference's ({ref['band_nonac']})")
        L4 = parse_log(open(os.path.join(out_p5, "mgo_phonons.log")).read())
        shared = commensurate(os.path.join(out, "band_nonac.yaml"), n)
        c1 = float(np.abs(band(os.path.join(out, "band_nonac.yaml")) - band(os.path.join(out_p5, "band_nonac.yaml")))[shared].max())
        rms0_p5 = rms(os.path.join(out_p5, "band_nonac.yaml"), ref["band_nonac"])
        records = {"P5_small": rms0, "P6_large": rms(os.path.join(out_p5, "band_nac.yaml"), ref["band_nac"]), "C1": c1}
        p0 = (f"P0 ensemble in the frame on every displaced supercell of {n}^3 and {n_p5}^3 (population >= 1 - 1e-4, |F - E_frame| <= 1 meV)",
              in_frame(L) and in_frame(L4), f"population {min(L['pop'], L4['pop']):.6f}, |F - E_frame| {max(L['fe'], L4['fe']):.4f} eV")
        p5 = (f"P5 path RMS without NAC <= 15 cm^-1, force constants on {n_p5}^3 (C1 <= {C1_TOL:g} cm^-1)",
              rms0_p5 <= 15 and c1 <= C1_TOL, f"{rms0_p5:.1f} (C1 {c1:.3f})")
    crit = [p0,
            (f"P1 Z*(Mg) within 1.90-2.10 ({ref_name} {ref['Z']})", 1.90 <= L["Z"] <= 2.10, f"{L['Z']:.3f}"),
            (f"P2 eps_inf within 2.9-3.5 ({ref_name} {ref['eps']})", 2.9 <= L["eps"] <= 3.5, f"{L['eps']:.3f}"),
            (f"P3 TO within 3 % of {ref_name} {ref['TO']}", abs(L["TO"] / ref["TO"] - 1) <= 0.03, f"{L['TO']:.1f} ({(L['TO'] / ref['TO'] - 1) * 100:+.1f} %)"),
            (f"P4 LO within {ref['LO_window'][0]:.0f}-{ref['LO_window'][1]:.0f} ({ref_name} {ref['LO']})", ref["LO_window"][0] <= L["LO"] <= ref["LO_window"][1], f"{L['LO']:.1f}"),
            p5,
            ("P6 path RMS with NAC <= 20 cm^-1", rms1 <= 20, f"{rms1:.1f}")]
    return crit, records


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--supercell", type=int, default=2); ap.add_argument("--out", default=None)
    ap.add_argument("--ref", choices=sorted(REFS), default="pbe", help="the DFT reference: pbe (Togo phono3py PBE) or pbesol (phonondb mp-1265)")
    ap.add_argument("--p5-supercell", type=int, default=None, help="read P5 on this conventional supercell (4: Togo's own; the protocol of every step after 34b)")
    args = ap.parse_args()
    if args.p5_supercell is not None:
        own = REFS[args.ref]["fc2_supercell"]
        if args.p5_supercell != own:
            ap.error(f"--p5-supercell {args.p5_supercell} is not the {args.ref} reference's own force-constant supercell ({own}^3)")
        if args.supercell < 1 or args.p5_supercell <= args.supercell or args.p5_supercell % args.supercell:
            ap.error(f"--p5-supercell {args.p5_supercell} must be a multiple of --supercell {args.supercell} and larger, "
                     "so that its commensurate points contain the smaller one's (C1)")
    global REF
    REF = REFS[args.ref]; a_ref = REF["a"]
    run = os.path.basename(os.path.dirname(os.path.abspath(args.model)))
    out = args.out or os.path.join(HERE, "runs", run + ("" if args.ref == "pbe" else f"_{args.ref}")); os.makedirs(out, exist_ok=True)

    def phonons(n, o):
        cmd = [sys.executable, os.path.join(ROOT, "tools", "electrostatics", "mgo_phonons.py"), args.model, "--cell", "2", "--supercell", str(n), "--a", repr(a_ref), "--out", o]
        log = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout
        open(os.path.join(o, "mgo_phonons.log"), "w").write(log)

    phonons(args.supercell, out)
    out_p5 = None
    if args.p5_supercell is not None:
        out_p5 = os.path.normpath(out) + f"_sc{args.p5_supercell}"; os.makedirs(out_p5, exist_ok=True)
        phonons(args.p5_supercell, out_p5)
    crit, records = criteria(out, REF, out_p5, args.supercell, args.p5_supercell, args.ref)
    head = f"=== phonon benchmark, {run}, reference {args.ref}, a = {a_ref:.5f} A, force-constant supercell {args.supercell}^3"
    if out_p5 is not None:
        head += f"; P5 on {args.p5_supercell}^3 (the reference's own: {REF['fc2_supercell']}^3)"
    print(head)
    for name, ok, val in crit:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {val}")
    print(f"  {sum(ok for _, ok, _ in crit)} / {len(crit)} passed; log and band yaml in {out}" + (f" and {out_p5}" if out_p5 else ""))
    if records:
        print(f"  recorded, not criteria: P5 on {args.supercell}^3 {records['P5_small']:.1f}; P6 on {args.p5_supercell}^3 {records['P6_large']:.1f}")


if __name__ == "__main__":
    main()
