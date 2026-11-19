"""Extensivity diagnosis 1 (2026-11-10): which term of the diagonal energy breaks E(2x1) = 2 E(1x1) for the SAME periodic
charge state.  The module's own arrays are captured at the call of _diabatic_solution (ladder e_hat, b, W, the pair slot,
the SPEC 28 many-body correction) and each term is evaluated for (a) the frame configuration and (b) the in-phase
transfer Au(+1, -1) / Au(-1, +1) in every copy.  Frozen weights, no training."""
import sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references
refs = references()
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu"); torch.set_num_threads(4)
mod = calc.model.electrostatics
captured = {}
orig = mod._diabatic_solution
def spy(e_hat, states, b, W, L, K, Q, T=None, **kw):
    captured.update(e_hat=e_hat.detach(), states=states.detach(), b=b.detach(), W=W.detach(), T=None if T is None else T.detach(),
                    pairs=kw.get("pairs"), diag=kw.get("diag_correction"), L=L, K=K)
    return orig(e_hat, states, b, W, L, K, Q, T, **kw)
mod._diabatic_solution = spy

def terms(cfg):
    c = captured; st = c["states"]; n = cfg.to(c["W"].dtype).unsqueeze(0)
    idx = torch.round(n - st[0]).to(torch.long)
    ladder = float(c["e_hat"].gather(1, idx.T).sum())
    lin = float(n @ c["b"]); quad = float(0.5 * ((n @ c["W"]) * n).sum())
    pair = 0.0
    if c["pairs"] is not None and int(c["pairs"][0].numel()):
        pi, pj, Xi = c["pairs"]; p = torch.arange(int(pi.numel()))
        pair = float(Xi.detach()[p, idx[0, pi], idx[0, pj]].sum())
    many = float(c["diag"](cfg.unsqueeze(0)).detach()) if c["diag"] is not None else 0.0
    return dict(ladder=ladder, linear=lin, quadratic=quad, pairs=pair, spec28=many)

for key in (("undoped", "non-wetting"), ("undoped", "wetting")):
    res = {}
    for rep in ((1, 1, 1), (2, 1, 1)):
        a = refs[key].copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
        a.calc = calc; calc.reset(); mod._ensemble_last = None
        a.get_potential_energy(); last = mod._ensemble_last
        cfg = last["configs"].to(torch.long); d = torch.diagonal(last["H"].detach()); frame = cfg[int(torch.argmin(d))].clone()
        ncopy = len(a) // 110
        for name, (qa, qb) in (("frame", (0, 0)), ("Au(+1,-1)", (1, -1)), ("Au(-1,+1)", (-1, 1))):
            x = frame.clone()
            for k in range(ncopy):
                x[108 + 110 * k] = frame[108 + 110 * k] + qa; x[109 + 110 * k] = frame[109 + 110 * k] + qb
            res[(rep, name)] = terms(x)
        res[(rep, "nunits")] = int(cfg.shape[1]); res[(rep, "npairs")] = 0 if captured["pairs"] is None else int(captured["pairs"][0].numel())
    print(f"== {key}: units {res[((1,1,1),'nunits')]} / {res[((2,1,1),'nunits')]}, pair-slot entries {res[((1,1,1),'npairs')]} / {res[((2,1,1),'npairs')]}")
    for name in ("frame", "Au(+1,-1)", "Au(-1,+1)"):
        t1, t2 = res[((1, 1, 1), name)], res[((2, 1, 1), name)]
        f1, f2 = res[((1, 1, 1), "frame")], res[((2, 1, 1), "frame")]
        print(f"   {name:10s}  " + "  ".join(f"{k} {1e3 * ((t2[k] - (0 if name == 'frame' else f2[k])) - 2 * (t1[k] - (0 if name == 'frame' else f1[k]))):+10.3f}" for k in t1)
              + f"   total {1e3 * sum((t2[k] - (0 if name == 'frame' else f2[k])) - 2 * (t1[k] - (0 if name == 'frame' else f1[k])) for k in t1):+10.3f} meV")
