"""Derive step 34b's run configs from step 34's (README, "Step 34 stage A stopped" and "Amendment: step 34b"; 2026-11-15).

Step 34b = step 34 with ONE change, `electrostatics.redox.band_initial_factor: 0.55` (SPEC 20's anchor from PBE's
conduction-band mass of MgO; the default is Harrison's 1).  Step 34's stage A, a cold start at f = 1, stopped in epoch
2 on both seeds with bulk MgO holding two O holes (the cluster closure refused 529 > 512).  New run names, so the
stopped runs stay as they are; every string value naming the source run is rewritten and then asserted gone (the
incident of 2026-11-05).  Also writes a one-epoch smoke config for the Mac.

    python derive_step34b.py            (writes runs/<run>/config.yaml for the five runs; refuses to overwrite)
"""
import copy
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
SRC_A, SRC_B = "AuMgO_twosite_bondlaw_r5", "AuMgO_twosite_e1kw_bondlaw_r5"
DST_A, DST_B = "AuMgO_twosite_bf055_bondlaw_r5", "AuMgO_twosite_bf055_e1kw_bondlaw_r5"
SMOKE = "AuMgO_twosite_bf055_smoke_mac_s43"
BAND_INITIAL_FACTOR = 0.55


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
    redox = cfg["electrostatics"]["redox"]
    assert "band_initial_factor" not in redox, (dst_run, redox.get("band_initial_factor"))
    assert redox.get("carrier_band") is True, dst_run
    redox["band_initial_factor"] = BAND_INITIAL_FACTOR
    if stage == "a":
        assert "resume_from" not in cfg, (dst_run, cfg.get("resume_from"))       # the cold start, as step 34
    else:
        assert cfg["resume_from"] == os.path.join(RUNS, dst_run, f"warm_from_twosite_a_s{cfg['seed']}.pt"), cfg["resume_from"]
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
# the Mac smoke: stage A of seed 43, one epoch (as step 34's smoke)
old, cfg = derive(f"{SRC_A}_s43", SMOKE, "a")
cfg = rewrite(cfg, f"{DST_A}_s43", SMOKE)
cfg.update(epochs=1, lr_scheduler_t_max=1)
assert not [s for s in strings(cfg) if SRC_A in s or DST_A in s]
write(SMOKE, cfg)
print(f"{SMOKE}: stage A of seed 43, epochs 1 (the Mac's CPU)")
