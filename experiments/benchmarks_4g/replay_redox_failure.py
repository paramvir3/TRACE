"""Replay a TRACE-Redox failure dump (run_train_dump.py / run_train_nanguard.py) and classify what the search finds.

Usage (repo root): python experiments/benchmarks_4g/replay_redox_failure.py RUN_DIR/redox_failure.pt [CAP]
       (env REPLAY_REDOX="carrier_penetration=False,carrier_overlap=electrostatic": counterfactual RedoxConfig overrides)
       (env REPLAY_PAULI_SCALE="12,1,8,-2=0.25;8,-1,12,2=1.3": multiply those types' runtime Pauli series R K dS by the
        factor, a probe-level stand-in for K refitted to first principles; a type is (Za, qa, Zb, qb) as tabulated)
       (env REPLAY_BAND_FACTOR=0.55: set the band unit's hopping factor, as RedoxConfig.band_initial_factor would at
        initialisation)

1. Rebuilds the module exactly as train.py does (parse_electrostatics_config of the dump's config.yaml; heads,
   ElectrostaticConfig, RedoxConfig from the dump), loads the dumped weights, and re-runs the failing call:
   the error must reproduce, or the replay is not the failure.
2. Re-runs it with fragment_max_configurations raised to CAP (default 20000): a diagnostic, not a fix.  Prints how
   many configurations the ensemble holds, and classifies them by the set of (element, state) changes away from the
   frames, with the lowest energy of each class and its population.
"""
import collections
import dataclasses
import os
import sys
import warnings

import torch
import yaml

warnings.simplefilter("ignore")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
torch.set_default_dtype(torch.float64)

import train  # noqa: E402
from trace_electrostatics import ElectrostaticHeads  # noqa: E402
from trace_redox import RedoxConfig, RedoxElectrostatics  # noqa: E402
from trace_redox.constants import formal_frame  # noqa: E402
from trace_redox.diabatic import density_matrix  # noqa: E402

SYMBOL = {8: "O", 12: "Mg", 13: "Al", 79: "Au"}


def build(dump):
    cfg = yaml.safe_load(open(dump["config"]))
    es = train.parse_electrostatics_config(cfg)
    heads = ElectrostaticHeads(irreps_in=dump["irreps"], **es.heads)
    module = RedoxElectrostatics(heads, es.config, RedoxConfig(**dump["redox"]))
    missing, unexpected = module.load_state_dict(dump["state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    module.train(bool(dump["training"]))
    return module


def reproduce(path):
    """(module, args, kwargs, dump) with the failure reproduced (raises if the replay does not fail)."""
    dump = torch.load(path, map_location="cpu", weights_only=False)
    module = build(dump)
    args, kwargs = dump["args"], dump["kwargs"]
    print(f"dump: call {dump['call']}, training {dump['training']}, error {dump['error'][:90]}")
    try:
        with torch.no_grad():
            module(*args, **kwargs)
    except Exception as err:                                                   # noqa: BLE001
        print(f"reproduced: {type(err).__name__}: {str(err)[:120]}")
        return module, args, kwargs, dump
    raise RuntimeError("REPLAY DID NOT FAIL: the replay is not the failure")


def apply_overrides(module, cap):
    """Raise the cap; REPLAY_REDOX counterfactual config overrides; REPLAY_PAULI_SCALE (see the module docstring)."""
    overrides = {}
    for item in filter(None, os.environ.get("REPLAY_REDOX", "").split(",")):   # counterfactuals: key=value,...
        key, value = item.split("=")
        overrides[key] = {"True": True, "False": False}.get(value, value)
    if overrides:
        print(f"counterfactual overrides: {overrides}")
    module.redox = dataclasses.replace(module.redox, fragment_max_configurations=cap, **overrides)
    bf = os.environ.get("REPLAY_BAND_FACTOR")
    if bf:
        import math
        with torch.no_grad():
            module.band.log_factor.fill_(math.atanh(math.log(float(bf)) / math.log(20.0)))
        print(f"band factor set to {float(module.band.factor):.6f}")
    scales = os.environ.get("REPLAY_PAULI_SCALE", "")
    if scales:
        from trace_redox.penetration import library
        lib = library()
        for item in filter(None, scales.split(";")):
            key, factor = item.split("=")
            key = tuple(int(v) for v in key.split(","))
            if key not in lib.index:
                raise KeyError(f"type {key} is not tabulated")
            rows = {lib.index[key]}
            swapped = (key[2], key[3], key[0], key[1])
            if key[0] == key[2] and swapped in lib.index:      # one pair, two tabulated orientations: keep them equal
                rows.add(lib.index[swapped])
            for t in rows:
                lib.gk_coefficients[t] *= float(factor)
            print(f"Pauli series of type {key} scaled by {float(factor):.4f} ({len(rows)} row(s))")


def main():
    path = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    try:
        module, args, kwargs, dump = reproduce(path)
    except RuntimeError as err:
        print(err)
        return
    apply_overrides(module, cap)
    with torch.no_grad():
        _, result = module(*args, **kwargs)
    last = module._ensemble_last
    configs, H, kT = last["configs"], last["H"], float(last["kT"])
    z = kwargs["data"]["z"]
    n = int(z.numel())
    f = formal_frame(z)
    E = torch.diagonal(H) - torch.diagonal(H).min()
    p = torch.diagonal(density_matrix(H, kT))
    classes = collections.defaultdict(list)
    for c in range(int(configs.shape[0])):
        changed = (configs[c, :n] != f).nonzero().flatten().tolist()
        key = tuple(sorted((SYMBOL.get(int(z[j]), str(int(z[j]))), int(configs[c, j]) - int(f[j])) for j in changed))
        band = int(configs[c, n:].abs().sum()) if configs.shape[1] > n else 0
        classes[(key, band)].append((float(E[c]), float(p[c])))
    print(f"with cap {cap}: {int(configs.shape[0])} configurations; kT {kT}; selection "
          f"{module.redox.diabatic_selection}; window {module.redox.diabatic_window}")
    rows = sorted(classes.items(), key=lambda kv: min(e for e, _ in kv[1]))
    for (key, band), v in rows[:25]:
        es = sorted(v)
        label = " + ".join(f"{s}{q:+d}" for s, q in key) or "frame"
        print(f"  {label:40s} band {band}: {len(v):5d} configurations, lowest {es[0][0]:+.3f} eV, "
              f"median {es[len(es) // 2][0]:+.3f}, total population {sum(q for _, q in v):.3f}")


if __name__ == "__main__":
    main()
