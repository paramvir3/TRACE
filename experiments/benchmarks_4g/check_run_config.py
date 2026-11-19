"""Guard against a run writing into another run's directory (2026-11-06; after the 2026-11-05 incident in which steps 25
and 26, derived from step 20's config with one path key replaced, saved their models as step 20's model.pt for six hours).

Usage (benchmarks_4g): python check_run_config.py runs/RUN [runs/RUN ...]      exit 1 on any violation

For each run: every string value of config.yaml that lies under runs/ must lie under this run's own directory, except
`resume_from` (a warm start reads another run's checkpoint) and the data / prior files (under data/); `model_save_path`
must be runs/RUN/model.pt (what the scoring scripts read); `checkpoint_dir` under runs/RUN.  Prints the path keys.
"""
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
ALLOWED_ELSEWHERE = {"resume_from"}


def check(run_dir):
    run_dir = os.path.abspath(run_dir)
    name = os.path.basename(run_dir)
    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
    bad = []

    def walk(obj, key):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, k)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, key)
        elif isinstance(obj, str) and obj.startswith(RUNS + os.sep):
            inside = obj.startswith(run_dir + os.sep) or obj == run_dir
            print(f"  {key:22s} -> runs/{os.path.relpath(obj, RUNS)}{'' if inside else '   <-- OUTSIDE this run'}")
            if not inside and key not in ALLOWED_ELSEWHERE:
                bad.append((key, obj))

    walk(cfg, "")
    msp = cfg.get("model_save_path")
    if msp != os.path.join(run_dir, "model.pt"):
        bad.append(("model_save_path", f"{msp} != runs/{name}/model.pt (what the scoring reads)"))
    print(f"{name}: {'OK' if not bad else 'VIOLATIONS ' + str(bad)}")
    return not bad


if __name__ == "__main__":
    ok = all(check(r) for r in sys.argv[1:])
    sys.exit(0 if ok else 1)
