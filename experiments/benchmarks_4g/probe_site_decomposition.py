# Diagnostic probe (2026-10-10; reported).  Usage (repo root): python experiments/benchmarks_4g/probe_site_decomposition.py CKPT [...]
"""E(Au- on the near Au) - E(Au- on the far Au) in 12 doped UPRIGHT test structures, split exactly:
E(n) = sum_u e_hat_u(n_u) + b.n + 1/2 n.W.n; moving the extra electron from far (f) to near (m), rest r fixed:
  dE = [e_m(-1) - e_m(0)] - [e_f(-1) - e_f(0)]  - (b_m - b_f)  - [(W r)_m - (W r)_f]  + 1/2 (W_mm - W_ff).
Negative dE: the model prefers the near Au; DFT prefers the far one (Hirshfeld +0.040 / -0.239)."""
import sys, warnings; sys.path.insert(0, 'experiments/benchmarks_4g'); sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np, torch; torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from ase.io import read
import trace_redox.module as module_code
from flashace.calculator import TransformersACECalculator
from trace_redox.diabatic import configuration_energies
test = read("experiments/benchmarks_4g/data/AuMgO_test.xyz", ":")
def upright(a):
    p = a.positions[a.numbers == 79]; v = p[1] - p[0]
    return np.degrees(np.arccos(abs(v[2]) / np.linalg.norm(v))) <= 45
doped = [a for a in test if (a.numbers == 13).any() and upright(a)][:12]
for ck in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=ck, device="cpu"); torch.set_num_threads(3)
    cap = {}
    orig = module_code.search_configurations
    def capture(e_hat, states, b, W, K, Q, **kw):
        out = orig(e_hat, states, b, W, K, Q, **kw)
        cap.update(e_hat=e_hat.detach(), states=states.detach(), b=b.detach(), W=W.detach(), configs=out[0], E=out[1].detach())
        return out
    module_code.search_configurations = capture
    rows = []
    for a0 in doped:
        a = a0.copy(); a.info = dict(a0.info); a.calc = calc; calc.reset(); a.get_potential_energy()
        e_hat, states, b, W, configs, E = cap["e_hat"], cap["states"], cap["b"], cap["W"], cap["configs"], cap["E"]
        z = a.numbers; au = np.where(z == 79)[0]
        m = au[int(np.argmax(a.positions[au, 2]))]; f = au[au != m][0]
        n0 = configs[int(torch.argmin(E))].clone(); n0[au] = 0
        col = {int(s): k for k, s in enumerate(states.tolist())}
        rest = n0.to(W.dtype)
        n_m, n_f = n0.clone(), n0.clone(); n_m[m] = -1; n_f[f] = -1
        E_m, E_f = [float(configuration_energies(n.unsqueeze(0), e_hat, states, b, W)[0]) for n in (n_m, n_f)]
        ladder = float((e_hat[m, col[-1]] - e_hat[m, col[0]]) - (e_hat[f, col[-1]] - e_hat[f, col[0]]))
        bound = -float(b[m] - b[f]); coulomb = -float((W @ rest)[m] - (W @ rest)[f]); self_ = 0.5 * float(W[m, m] - W[f, f])
        assert abs((E_m - E_f) - (ladder + bound + coulomb + self_)) < 1e-8
        rows.append([E_m - E_f, ladder, bound, coulomb, self_])
    module_code.search_configurations = orig
    x = np.array(rows); mu = x.mean(0)
    print(f"=== {ck.split('/')[-2]}  ({len(rows)} doped upright test structures; eV; negative = near Au preferred; DFT: far)")
    print(f"  dE = {mu[0]:+.3f} = ladder/shift {mu[1]:+.3f} + bound b {mu[2]:+.3f} + Coulomb with the lattice {mu[3]:+.3f} + self {mu[4]:+.3f}"
          f"   (dE range {x[:, 0].min():+.3f} .. {x[:, 0].max():+.3f})")
