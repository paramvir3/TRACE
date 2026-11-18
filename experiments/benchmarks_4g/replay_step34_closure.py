"""Step 34 stage A stopped in epoch 2 on both seeds (2026-11-15): ConfigurationSearchError, the product closure of one
cluster holds 529 > cap = 512 configurations (63 linked) on a bulk MgO 64-atom supercell (train indices 404 / 347).

Usage (repo root): python experiments/benchmarks_4g/replay_step34_closure.py DUMP LOCAL_CONFIG

1. Rebuilds the module from the dump (replay_redox_failure.build) with the dump's config path replaced by the local
   copy of the same run config (only the data-file paths differ), and reproduces the refusal at cap 512.
2. Records every product closure (_closure: linked rows m, closed rows) at a raised cap for
     as trained:        frame_pair, spin_pairing, two_site on
     counterfactuals:   two_site off; two_site + spin_pairing off; two_site + frame_pair off; all three off
   The weights are frozen (the trained ones); a counterfactual says which term, at these weights, makes the cluster
   that large.  It is a diagnostic, not a fix and not a prediction of a run trained without the term.
3. Classifies the rows of the largest linked set by the (element, charge change) of the units they move.
"""
import collections
import dataclasses
import os
import sys
import time
import warnings

import torch

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)

import replay_redox_failure as rrf  # noqa: E402
import trace_redox.clusters as clusters_mod  # noqa: E402

RECORD = []
_orig = clusters_mod._closure


def _recording(n0, S, reservoir, L, K, units, frames, cap):
    try:
        out = _orig(n0, S, reservoir, L, K, units, frames, cap)
    except Exception:
        RECORD.append((int(S.shape[0]), None, S.clone()))
        raise
    RECORD.append((int(S.shape[0]), int(out.shape[0]), S.clone()))
    return out


clusters_mod._closure = _recording


def classify(S, n0, z, units_n):
    sym = rrf.SYMBOL
    rows = collections.Counter()
    for r in range(1, int(S.shape[0])):
        changed = (S[r] != n0).nonzero().flatten().tolist()
        key = []
        for j in changed:
            dq = int(S[r, j]) - int(n0[j])
            if j < units_n:
                key.append(f"{sym.get(int(z[j]), str(int(z[j])))}{dq:+d}")
            else:
                key.append(f"band{dq:+d}")
        rows[" ".join(sorted(key))] += 1
    return rows


def main():
    path, local_config = sys.argv[1], sys.argv[2]
    dump = torch.load(path, map_location="cpu", weights_only=False)
    dump["config"] = os.path.abspath(local_config)
    module = rrf.build(dump)
    args, kwargs = dump["args"], dump["kwargs"]
    z = kwargs["data"]["z"]
    print(f"dump: call {dump['call']}, training {dump['training']}, atoms {int(z.numel())}, "
          f"species {dict(collections.Counter(int(v) for v in z.tolist()))}")
    print(f"error: {dump['error'][:160]}")
    RECORD.clear()
    try:
        with torch.no_grad():
            module(*args, **kwargs)
        print("REPLAY DID NOT FAIL at cap 512: the replay is not the failure")
        return
    except Exception as err:                                                     # noqa: BLE001
        print(f"reproduced at cap {module.redox.fragment_max_configurations}: {type(err).__name__}: {str(err)[:110]}")
    base = module.redox
    cases = [("as trained", {}),
             ("two_site off", dict(two_site=False)),
             ("two_site + spin_pairing off", dict(two_site=False, spin_pairing=False)),
             ("two_site + frame_pair off", dict(two_site=False, frame_pair=False)),
             ("all three off", dict(two_site=False, spin_pairing=False, frame_pair=False))]
    for name, over in cases:
        module.redox = dataclasses.replace(base, fragment_max_configurations=20000, **over)
        RECORD.clear()
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                E, _ = module(*args, **kwargs)
            status = f"E {float(E):+.6f} eV"
        except Exception as err:                                                 # noqa: BLE001
            status = f"{type(err).__name__}: {str(err)[:90]}"
        dt = time.perf_counter() - t0
        sizes = sorted(((m, c) for m, c, _ in RECORD), key=lambda x: -(x[1] or 0))
        big = [s for s in sizes if (s[1] or 0) > 64]
        print(f"\n[{name}] {dt:.1f} s; {status}")
        print(f"  closures: {len(sizes)}; largest (linked, closed): {sizes[:6]}; over 64 closed: {len(big)}")
        if name == "as trained" and RECORD:
            m, c, S = max(RECORD, key=lambda x: x[1] or 0)
            n0 = module._ensemble_last["reference"]
            print(f"  largest linked set ({m} rows incl. n0, closed {c}); moves by (element, change):")
            for k, v in classify(S, n0, z, int(z.numel())).most_common(20):
                print(f"    {v:4d}  {k}")


if __name__ == "__main__":
    main()
