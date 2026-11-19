"""Extensivity diagnosis 2 (2026-11-10).  Hypothesis H1: SPEC 28's Delta_j = P_jj (1/rho_j - 1) is built from the moment
block P_jj of the periodic operator, P_jj = A_mu,j + T2_jj, where T2_jj is the dipole-dipole lattice sum of atom j with
its OWN periodic images (zero in an open system, cell-dependent in a periodic one).  The state's polarisability changes
the on-site stiffness A_mu,j only.  Falsifying check: (a) T2_jj != 0 and differs between 1x1 and 2x1; (b) with
Delta_j = A_mu,j (1/rho_j - 1) the in-phase configuration's SPEC 28 term is extensive to round-off.
Frozen weights; nothing in the package is changed (the block is patched in this process only)."""
import sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references
import trace_redox.polarizability as PZ
refs = references()
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu"); torch.set_num_threads(4)
mod = calc.model.electrostatics
CAP = {}
orig_corr, orig_block = PZ.state_polarizability_corrections, PZ._dipole_block
def capture(cfgs, elim, *a, **k):
    CAP["elim"] = elim; CAP["args"] = (a, k)
    return orig_corr(cfgs, elim, *a, **k)
PZ.state_polarizability_corrections = capture
def onsite_block(operator, nb, j):
    return CAP["elim"].assembly.params.A_mu[j]

for key in (("undoped", "non-wetting"), ("undoped", "wetting")):
    out = {}
    for rep in ((1, 1, 1), (2, 1, 1)):
        a = refs[key].copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
        a.calc = calc; calc.reset(); mod._ensemble_last = None
        a.get_potential_energy(); last = mod._ensemble_last
        cfg = last["configs"].to(torch.long); d = torch.diagonal(last["H"].detach()); frame = cfg[int(torch.argmin(d))].clone()
        elim = CAP["elim"]; args, kw = CAP["args"]; asm = elim.assembly; ncopy = len(a) // 110
        t2 = asm.t2_pair.detach(); A = asm.params.A_mu.detach()
        for j in (108, 109):
            self_img = t2[j, j]; on = A[j]
            out[(rep, "blk", j)] = (torch.linalg.eigvalsh(0.5 * (on + on.T)).tolist(), torch.linalg.eigvalsh(0.5 * (self_img + self_img.T)).tolist())
        for name, (qa, qb) in (("Au(+1,-1)", (1, -1)), ("Au(-1,+1)", (-1, 1))):
            x = frame.clone()
            for k in range(ncopy):
                x[108 + 110 * k] += qa; x[109 + 110 * k] += qb
            for mode in ("block", "onsite"):
                PZ._dipole_block = orig_block if mode == "block" else onsite_block
                with torch.no_grad():
                    out[(rep, name, mode)] = float(orig_corr(x.unsqueeze(0), elim, *args, **kw))
        PZ._dipole_block = orig_block
    print(f"== {key}")
    for j in (108, 109):
        for rep in ((1, 1, 1), (2, 1, 1)):
            on, si = out[(rep, "blk", j)]
            print(f"   Au {j} {rep[0]}x1: on-site A_mu eigenvalues {['%.4f' % v for v in on]}   own-image T2_jj eigenvalues {['%+.5f' % v for v in si]} eV/A^2")
    for name in ("Au(+1,-1)", "Au(-1,+1)"):
        for mode in ("block", "onsite"):
            e1, e2 = out[((1, 1, 1), name, mode)], out[((2, 1, 1), name, mode)]
            print(f"   {name}  Delta from {mode:6s}: dE28(1x1) {1e3 * e1:+10.3f}  dE28(2x1) {1e3 * e2:+10.3f}  2x1 - 2*(1x1) {1e3 * (e2 - 2 * e1):+12.6f} meV")
