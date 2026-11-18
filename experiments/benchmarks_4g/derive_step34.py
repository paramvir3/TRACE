"""Derive step 34's run configs from step 33's (README, "Pre-registered: step 34"; 2026-11-15).

Step 34 = step 33's two-stage recipe and data with SPEC 39 (frame_pair), SPEC 29 (spin_pairing) and SPEC 40 / 40.7
(two_site) switched on, and stage A started COLD (no resume_from): weights trained without these terms hold the
Au-Au repulsion in the backbone.  Every string value naming the source run is rewritten and then asserted gone (the
incident of 2026-11-05).  Also writes a one-epoch smoke config for the Mac.

    python derive_step34.py            (writes runs/<run>/config.yaml for the five runs; refuses to overwrite)
"""
import copy
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
SRC_A, SRC_B = "AuMgO_single_all_bondlaw_r5", "AuMgO_single_all_e1kw_bondlaw_r5"
DST_A, DST_B = "AuMgO_twosite_bondlaw_r5", "AuMgO_twosite_e1kw_bondlaw_r5"
SMOKE = "AuMgO_twosite_smoke_mac_s43"
NEW_TERMS = {"frame_pair": True, "spin_pairing": True, "two_site": True}


def rewrite(obj, old, new):
    if isinstance(obj, dict):
        return {k: rewrite(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite(v, old, new) for v in obj]
    if isinstance(obj, str):
        return obj.replace(old, new)
    return obj


def strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)
    elif isinstance(obj, str):
        yield obj


def flat(obj, prefix=""):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(flat(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: obj}


def derive(src_run, dst_run, stage):
    cfg = yaml.safe_load(open(os.path.join(RUNS, src_run, "config.yaml")))
    old = copy.deepcopy(cfg)
    cfg = rewrite(cfg, src_run, dst_run)
    leftover = [s for s in strings(cfg) if src_run in s]
    assert not leftover, (dst_run, leftover)
    for k, v in NEW_TERMS.items():
        assert k not in cfg["electrostatics"]["redox"] or cfg["electrostatics"]["redox"][k] is False, (k, dst_run)
        cfg["electrostatics"]["redox"][k] = v
    if stage == "a":
        cfg.pop("resume_from")                                   # the cold start
    else:
        seed = cfg["seed"]
        cfg["resume_from"] = os.path.join(RUNS, dst_run, f"warm_from_twosite_a_s{seed}.pt")
    return old, cfg


def write(dst_run, cfg):
    d = os.path.join(RUNS, dst_run)
    path = os.path.join(d, "config.yaml")
    if os.path.exists(path):
        sys.exit(f"refused: {path} exists (never overwrite a run's config)")
    os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=True)


plans = []
for seed in (42, 43):
    plans.append((f"{SRC_A}_s{seed}", f"{DST_A}_s{seed}", "a"))
    plans.append((f"{SRC_B}_s{seed}", f"{DST_B}_s{seed}", "b"))
for src, dst, stage in plans:
    old, cfg = derive(src, dst, stage)
    a, b = flat(old), flat(cfg)
    changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    print(f"{dst}: keys that differ from {src}:")
    for k in changed:
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if isinstance(va, str) and isinstance(vb, str) and va.replace(src, dst) == vb:
            continue                                             # the run name inside a path
        print(f"    {k}: {va!r} -> {vb!r}")
    write(dst, cfg)
# the Mac smoke: stage A of seed 43, one epoch
old, cfg = derive(f"{SRC_A}_s43", SMOKE, "a")
cfg = rewrite(cfg, f"{DST_A}_s43", SMOKE)
cfg.update(epochs=1, lr_scheduler_t_max=1)
assert not [s for s in strings(cfg) if SRC_A in s or DST_A in s]
write(SMOKE, cfg)
print(f"{SMOKE}: stage A of seed 43, epochs 1 (the Mac's CPU)")
