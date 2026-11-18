"""Check every derived data file of the released benchmarks against DATA_SHA256.txt: sha256 first, then content.

Usage (any directory):
    python experiments/benchmarks_4g/verify_data.py [--raw] [--only SUBSTRING]
    python experiments/benchmarks_4g/verify_data.py --fingerprint FILE...     (print content lines; maintainers)

DATA_SHA256.txt (next to this script) lists, relative to <repo>/experiments:
- `<sha256>  <path>` lines (the format of `shasum -a 256 -c`, so `cd experiments && shasum -a 256 -c
  benchmarks_4g/DATA_SHA256.txt` also works, with warnings for the comment lines);
- `#content <path> <json>` lines: a content fingerprint of the same file (see below);
- `#raw <sha256> <md5 or -> <path>` lines: the third-party downloads, with ${TRACE_4G_RAW} and ${TRACE_PHONON_RAW}
  expanded (defaults experiments/benchmarks_4g/raw/datasets and experiments/benchmarks_phonons/raw); checked with --raw;
- section headers `## required`, `## optional` (no released run reads the file; absent is not an error) and
  `## shipped` (tracked in git).

The sha256 are those of the files written with ase 3.23.0 / numpy 1.24.2 / PyYAML 6.0.  Another ase may format the same
numbers differently, so a sha256 mismatch is followed by a content check against the fingerprint: for every quantity
of the file (positions, cell, pbc, atomic numbers, energies, forces and every other key ase stores in info, arrays or
the single-point calculator, merged by name; for .yaml / .json every numeric leaf in document order) the fingerprint
holds the count, the sum, the sum of squares, the sum of magnitudes, the minimum, the maximum and an index-weighted
sum (the weighted sum catches reordering); string values are hashed.  Numbers agree when every statistic is within
six standard deviations of what independent element-wise differences of size --tol (default 1e-7 in the file's own
units: Angstrom, eV, eV/Angstrom, e) produce; ase writes 8 decimals, so a rewrite by another version passes.  The check
is statistical: it finds a reordering, a missing or extra value, a changed extreme, or a change of one value larger
than about 6 * tol * sqrt(count) (1.3e-4 for 48,000 values); it cannot prove
element-wise equality.  The report names the file and the quantities that differ.  Exit status: 0 when every required and shipped file passes (by sha256 or by content), 1
otherwise.
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
LIST = os.path.join(HERE, "DATA_SHA256.txt")
SIGMAS = 6.0
RAW_DEFAULTS = {"TRACE_4G_RAW": os.path.join(HERE, "raw", "datasets"),
                "TRACE_PHONON_RAW": os.path.join(EXPERIMENTS, "benchmarks_phonons", "raw")}


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _stats(values):
    x = np.asarray(values, dtype=np.float64).ravel()
    w = np.cos(0.37 * np.arange(x.size))
    out = {"n": int(x.size), "sum": float(x.sum()), "sumsq": float((x * x).sum()), "wsum": float((w * x).sum()),
           "abssum": float(np.abs(x).sum())}
    if x.size:
        out.update(min=float(x.min()), max=float(x.max()))
    return out


def _add(bucket, name, value):
    if isinstance(value, (str, bytes)):
        bucket.setdefault(name, []).append(value if isinstance(value, str) else value.decode())
        return
    arr = np.asarray(value)
    if arr.dtype.kind in "US":
        bucket.setdefault(name, []).extend(str(v) for v in arr.ravel())
    elif arr.dtype.kind == "b":
        bucket.setdefault(name, []).extend(float(v) for v in arr.ravel())
    elif arr.dtype.kind in "iuf":
        bucket.setdefault(name, []).extend(arr.astype(np.float64).ravel().tolist())
    else:
        bucket.setdefault(name, []).append(repr(value))


def _finish(bucket):
    out = {}
    for name in sorted(bucket):
        vals = bucket[name]
        if vals and isinstance(vals[0], str):
            out[name] = {"str_sha256": hashlib.sha256("\x1f".join(vals).encode()).hexdigest(), "n": len(vals)}
        else:
            out[name] = _stats(vals)
    return out


def fingerprint_structures(path):
    from ase.io import read
    frames = read(path, ":")
    bucket = {}
    for atoms in frames:
        _add(bucket, "numbers", atoms.numbers)
        _add(bucket, "positions", atoms.positions)
        _add(bucket, "cell", np.asarray(atoms.cell))
        _add(bucket, "pbc", atoms.pbc)
        merged = {}
        for key, value in atoms.info.items():
            merged[key] = value
        for key, value in atoms.arrays.items():
            if key not in ("numbers", "positions"):
                merged[key] = value
        if atoms.calc is not None:
            for key, value in atoms.calc.results.items():
                merged[key] = value
        for key, value in merged.items():
            _add(bucket, key, value)
    out = _finish(bucket)
    out["_frames"] = {"n": len(frames), "sum": float(len(frames)), "sumsq": 0.0, "wsum": 0.0}
    return out


def _walk(node, bucket, keys):
    if isinstance(node, dict):
        for key, value in node.items():
            keys.add(str(key))
            _walk(value, bucket, keys)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _walk(value, bucket, keys)
    elif isinstance(node, bool):
        bucket.setdefault("bool", []).append(float(node))
    elif isinstance(node, (int, float)):
        bucket.setdefault("numbers", []).append(float(node))
    elif node is not None:
        bucket.setdefault("strings", []).append(str(node))


def fingerprint_document(path):
    if path.endswith(".json"):
        with open(path) as handle:
            doc = json.load(handle)
    else:
        import yaml
        with open(path) as handle:
            doc = yaml.load(handle, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    bucket, keys = {}, set()
    _walk(doc, bucket, keys)
    out = _finish(bucket)
    out["_keys"] = {"str_sha256": hashlib.sha256("\x1f".join(sorted(keys)).encode()).hexdigest(), "n": len(keys)}
    return out


def fingerprint(path):
    if path.endswith((".json", ".yaml", ".yml")):
        return fingerprint_document(path)
    return fingerprint_structures(path)


def compare(reference, measured, tol):
    """Names of the quantities whose statistics disagree (with the reason)."""
    problems = []
    for name in sorted(set(reference) | set(measured)):
        if name not in measured:
            problems.append(f"{name}: missing")
            continue
        if name not in reference:
            problems.append(f"{name}: not in the reference")
            continue
        a, b = reference[name], measured[name]
        if "str_sha256" in a or "str_sha256" in b:
            if a != b:
                problems.append(f"{name}: string values differ")
            continue
        if a["n"] != b["n"]:
            problems.append(f"{name}: {b['n']} values, expected {a['n']}")
            continue
        # independent element-wise differences of size <= tol add up as sqrt(n); SIGMAS standard deviations allowed,
        # plus the floating-point error of the summation itself (relative 1e-12 of the sum of magnitudes)
        n = max(a["n"], 1)
        roundoff = 1e-12 * a.get("abssum", abs(a["sum"]))
        limits = {"sum": SIGMAS * tol * math.sqrt(n), "wsum": SIGMAS * tol * math.sqrt(n),
                  "sumsq": SIGMAS * 2.0 * tol * math.sqrt(max(a["sumsq"], 0.0)) + n * tol * tol,
                  "min": tol, "max": tol}
        for stat, limit in limits.items():
            if stat not in a:
                continue
            delta = abs(a[stat] - b[stat])
            if delta > limit + roundoff:
                problems.append(f"{name}: {stat} differs by {delta:.3g} (allowed {limit + roundoff:.3g})")
    return problems


def parse_list(path=LIST):
    entries, contents, raws, section = [], {}, [], "required"
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("## "):
                section = line[3:].split()[0]
            elif line.startswith("#content "):
                _, rel, payload = line.split(" ", 2)
                contents[rel] = json.loads(payload)
            elif line.startswith("#raw "):
                _, sha, md5, rel = line.split(None, 3)
                raws.append((sha, md5, rel))
            elif line and not line.startswith("#"):
                sha, rel = line.split(None, 1)
                entries.append((sha, rel.lstrip("*"), section))
    return entries, contents, raws


def expand(rel):
    env = {k: os.environ.get(k, v) for k, v in RAW_DEFAULTS.items()}
    for key, value in env.items():
        rel = rel.replace("${" + key + "}", value)
    return rel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", action="store_true", help="also check the third-party downloads")
    parser.add_argument("--only", default=None, help="check only paths containing this substring")
    parser.add_argument("--tol", type=float, default=1e-7, help="element-wise tolerance of the content check")
    parser.add_argument("--fingerprint", nargs="+", metavar="FILE", help="print #content lines for FILE (paths "
                        "relative to <repo>/experiments or absolute; the printed path is relative to --root)")
    parser.add_argument("--root", default=EXPERIMENTS, help="directory the listed paths are relative to")
    args = parser.parse_args(argv)

    if args.fingerprint:
        for item in args.fingerprint:
            path = item if os.path.isabs(item) else os.path.join(args.root, item)
            rel = os.path.relpath(path, args.root)
            print(f"#content {rel} {json.dumps(fingerprint(path), separators=(',', ':'))}")
        return 0

    entries, contents, raws = parse_list()
    failed, counts = [], {"sha256": 0, "content": 0, "absent optional": 0}
    for sha, rel, section in entries:
        if args.only and args.only not in rel:
            continue
        path = os.path.join(args.root, rel)
        if not os.path.exists(path):
            if section == "optional":
                counts["absent optional"] += 1
                print(f"--   {rel}: absent (optional: no released run reads it)")
            else:
                failed.append(rel)
                print(f"FAIL {rel}: missing (run experiments/benchmarks_4g/prepare_all.sh)")
            continue
        got = digest(path)
        if got == sha:
            counts["sha256"] += 1
            print(f"ok   {rel}: sha256")
            continue
        reference = contents.get(rel)
        if reference is None:
            failed.append(rel)
            print(f"FAIL {rel}: sha256 {got}, expected {sha}; no content fingerprint to fall back on")
            continue
        problems = compare(reference, fingerprint(path), args.tol)
        if problems:
            failed.append(rel)
            print(f"FAIL {rel}: sha256 differs and the content differs: " + "; ".join(problems))
        else:
            counts["content"] += 1
            print(f"ok   {rel}: sha256 differs (other writer), content equal within {args.tol:g} per element")

    if args.raw:
        for sha, md5, rel in raws:
            path = expand(rel)
            if args.only and args.only not in path:
                continue
            if not os.path.exists(path):
                print(f"--   {path}: absent (raw download)")
                continue
            ok = digest(path) == sha and (md5 == "-" or digest(path, "md5") == md5)
            print(f"{'ok  ' if ok else 'FAIL'} {path}: raw download sha256" + ("" if md5 == "-" else " and md5"))
            if not ok:
                failed.append(path)

    print(f"\n{counts['sha256']} files equal by sha256, {counts['content']} equal by content, "
          f"{counts['absent optional']} optional absent, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
