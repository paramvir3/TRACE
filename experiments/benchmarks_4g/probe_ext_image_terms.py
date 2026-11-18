"""Extensivity diagnosis 4 (2026-11-10, after SPEC 36).  What is left of E(M cells) / M - E(1 cell) with
``diabatic_space='clusters'`` is the 1x1 cell's own finite-cell term (README R13.3, L4).  This probe says WHICH term
of a configuration's excitation energy carries it.  Frozen weights; nothing in the package or a run directory changes.

    python probe_ext_image_terms.py MODEL.pt FORM [nx ny ...]

Per geometry and cell it prints, for cluster 0 and each of its configurations m != n0 (dn = m - n0):
    x        E(m) - E(n0), the Hamiltonian's diagonal
    ladder   sum_u e_hat[u, m_u] - e_hat[u, n0_u]
    linear   (b + W n0) . dn            the excitation in the field of the unexcited crystal
    self     1/2 dn W dn                the excitation with itself AND with its in-phase periodic images
    pairs    the pair tables
    spec28   SPEC 28's many-body correction
    t        the coupling to n0
and the free energy split: F_c - E0 of one cluster, and the pair increments per cell."""
import dataclasses
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references
from trace_redox.diabatic import canonical_free_energy, configuration_energies, density_matrix

model_path, form = sys.argv[1], sys.argv[2]
reps = [(int(sys.argv[i]), int(sys.argv[i + 1]), 1) for i in range(3, len(sys.argv) - 1, 2)] or [(2, 1, 1), (2, 2, 1)]
calc = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
if form != "checkpoint":
    mod.redox = dataclasses.replace(mod.redox, state_polarizability_stiffness=form)
mod.redox = dataclasses.replace(mod.redox, diabatic_space="clusters")
refs = references()
CAP = {}
original = mod._cluster_solution


def capture(e_hat, states, b, W, L_units, K_units, Q, T=None, units=None, frames=None, pairs=None,
            diag_correction=None, electrons=None, reservoir=None):
    sol = original(e_hat, states, b, W, L_units, K_units, Q, T=T, units=units, frames=frames, pairs=pairs,
                   diag_correction=diag_correction, electrons=electrons, reservoir=reservoir)
    CAP.update(e_hat=e_hat.detach(), states=states.detach(), b=b.detach(), W=W.detach(), pairs=pairs,
               corr=diag_correction, F=sol.energy.detach())
    return sol


mod._cluster_solution = capture


def report(key, rep):
    a = refs[key].copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
    a.calc = calc; calc.reset(); mod._ensemble_last = None
    e = a.get_potential_energy()
    last = mod._ensemble_last
    n0, S, H = last["reference"], last["clusters"][0], last["hamiltonians"][0].detach()
    kT = float(last["kT"]); copies = rep[0] * rep[1]
    e_hat, states, b, W = CAP["e_hat"], CAP["states"], CAP["b"], CAP["W"]
    out = {"e": e}
    with torch.no_grad():
        dtype = W.dtype
        nf, Sf = n0.to(dtype), S.to(dtype)
        dn = Sf[1:] - nf
        zero = torch.zeros_like(b); Wz = torch.zeros_like(W)
        ladder = configuration_energies(S, e_hat, states, zero, Wz)
        ladder = ladder[1:] - ladder[0]
        linear = dn @ (b + W @ nf)
        self_ = 0.5 * ((dn @ W) * dn).sum(1)
        full = configuration_energies(S, e_hat, states, b, W, pairs=CAP["pairs"])
        bare = configuration_energies(S, e_hat, states, b, W)
        pairs = (full - bare)[1:] - (full - bare)[0]
        if CAP["corr"] is not None:
            c = CAP["corr"](S).detach()
            spec28 = c[1:] - c[0]
        else:
            spec28 = torch.zeros_like(self_)
        x = torch.diagonal(H)[1:] - H[0, 0]
        E0 = H[0, 0]
        Fc = canonical_free_energy(H, kT) - E0
        p = torch.diagonal(density_matrix(H, kT))
        pair_inc = (CAP["F"] - E0 - len(last["clusters"]) * Fc) / copies      # clusters are copies of each other
    print(f"   {rep[0]}x{rep[1]}: clusters {len(last['clusters'])}; F_c - E0 = {1e3 * float(Fc):+.4f} meV; pair increments "
          f"per cell {1e3 * float(pair_inc):+.4f} meV; weight of n0 {float(p[0]):.4f}")
    for i in range(int(dn.shape[0])):
        ch = dn[i].nonzero().flatten().tolist()
        resid = float(x[i] - ladder[i] - linear[i] - self_[i] - pairs[i] - spec28[i])
        print(f"      m{i + 1} [" + ", ".join(f"{u % 110 if u < 110 * copies else 'band'}:{int(dn[i, u]):+d}" for u in ch) + "]"
              f"  x {float(x[i]):+.6f}  ladder {float(ladder[i]):+.6f}  linear {float(linear[i]):+.6f}  self {float(self_[i]):+.6f}"
              f"  pairs {float(pairs[i]):+.6f}  spec28 {float(spec28[i]):+.6f}  (residual {resid:+.1e})  t {float(H[0, i + 1]):+.6f}"
              f"  p {float(p[i + 1]):.4f}")
    # a configuration's identity is its content (which atom of the cell, which change), not its index: the
    # selected sets of different cells differ (the band configuration exists in the 1x1 cell only)
    keys = [tuple(sorted(((u % 110 if u < 110 * copies else -1), int(dn[i, u])) for u in dn[i].nonzero().flatten().tolist()))
            for i in range(int(dn.shape[0]))]
    out.update(x=x, self=self_, linear=linear, ladder=ladder, spec28=spec28, pairs=pairs, Fc=float(Fc), pair=float(pair_inc), p=p,
               keys=keys)
    return out


for key in (("undoped", "wetting"), ("undoped", "non-wetting")):
    print(f"== {key}")
    one = report(key, (1, 1, 1))
    for rep in reps:
        r = report(key, rep)
        both = [(i, one["keys"].index(k)) for i, k in enumerate(r["keys"]) if k in one["keys"]]
        print(f"      against 1x1 (meV), the {len(both)} configurations present in both cells: "
              + "; ".join(f"m{i + 1} dx {1e3 * float(r['x'][i] - one['x'][j]):+.3f} = ladder {1e3 * float(r['ladder'][i] - one['ladder'][j]):+.3f}"
                          f" + linear {1e3 * float(r['linear'][i] - one['linear'][j]):+.3f} + self {1e3 * float(r['self'][i] - one['self'][j]):+.3f}"
                          f" + pairs {1e3 * float(r['pairs'][i] - one['pairs'][j]):+.3f} + spec28 {1e3 * float(r['spec28'][i] - one['spec28'][j]):+.3f}"
                          for i, j in both))
        print(f"      E/cell - E(1x1) {1e3 * (r['e'] / (rep[0] * rep[1]) - one['e']):+.4f} meV = d(F_c - E0) {1e3 * (r['Fc'] - one['Fc']):+.4f}"
              f" + pair increments {1e3 * r['pair']:+.4f} + dE0/cell (the rest)", flush=True)
