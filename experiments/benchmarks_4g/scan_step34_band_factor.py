"""Step 34 stage A's epoch-2 failure dumps (2026-11-15): with every weight frozen, set only the band unit's hopping
factor f (SPEC 20) and record the band's reference filling N0 (SPEC 38's descent), the product closures and the
population of O-hole configurations.  A diagnostic of which quantity puts holes into bulk MgO, not a fix.

    python experiments/benchmarks_4g/scan_step34_band_factor.py DUMP_LOCAL.pt f1,f2,...
"""
import collections, dataclasses, math, os, sys, warnings
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", ".."))); sys.path.insert(0, HERE)
import torch
torch.set_default_dtype(torch.float64)
import replay_redox_failure as rrf
import trace_redox.clusters as cm
from trace_redox.constants import formal_frame
from trace_redox.diabatic import density_matrix
REC = []
_orig = cm._closure
def rec(n0, S, res, L, K, u, fr, cap):
    out = _orig(n0, S, res, L, K, u, fr, cap); REC.append((int(S.shape[0]), int(out.shape[0]))); return out
cm._closure = rec
dump = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = rrf.build(dump); a, k = dump["args"], dump["kwargs"]
m.redox = dataclasses.replace(m.redox, fragment_max_configurations=20000)
z = k["data"]["z"]; n = int(z.numel()); f0 = formal_frame(z)
lf0 = float(m.band.log_factor)
print(f"dump band factor {math.exp(math.log(20) * math.tanh(lf0)):.4f}")
for fac in [float(x) for x in sys.argv[2].split(",")]:
    with torch.no_grad():
        m.band.log_factor.fill_(math.atanh(math.log(fac) / math.log(20.0)))
        REC.clear()
        E, _ = m(*a, **k)
    last = m._ensemble_last; N0 = m._band_last.get("reference") if m._band_last else None
    holes = collections.Counter()
    for S, H in zip(last.get("clusters") or [], last.get("hamiltonians") or []):
        p = torch.diagonal(density_matrix(H, float(last["kT"])))
        for c in range(int(S.shape[0])):
            holes[int(((S[c, :n] - f0) * (z == 8)).clamp(min=0).sum())] += float(p[c])
    nclu = len(last.get("clusters") or [])
    mean_holes = sum(h * v for h, v in holes.items()) / max(nclu, 1) if nclu else 0.0
    print(f"f {fac:.3f}: E {float(E):+.4f} eV; N0 {N0}; clusters {nclu}; largest closures {sorted(REC, key=lambda x: -x[1])[:3]}; "
          f"O holes per cluster (mean) {mean_holes:.4f}")
