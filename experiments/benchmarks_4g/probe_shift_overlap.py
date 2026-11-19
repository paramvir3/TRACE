"""Are the learned charge-state shifts a learned overlap law?  (frozen weights; 2026-10-19; reported)

Usage: python probe_shift_overlap.py MODEL.pt N [Z ...]     (N seeded test structures; elements to report, default 79)

For a model trained WITHOUT the overlap terms (SPEC 23 penetration + 25 Gordon-Kim), the overlap energy of atom u in
state n next to its neighbours' frames,

    pi_u(n) = sum_v [ Xi_uv(n, f_v) - Xi_uv(f_u, f_v) ]      (the library's first-principles pair tables),

is missing from its configuration energies.  If its learned shifts carry it, then across environments
Delta_u(n) = a_{Z,n} + b_{Z,n} pi_u(n) with b ~ +1 and high R^2 (a absorbs the shift's constant part).  Printed per
(element, state): the slope b, R^2, the spread of Delta and of pi, and their correlation.
"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)

from ase.io import read  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_redox.constants import formal_frame  # noqa: E402
from trace_redox.diabatic import formal_ladder  # noqa: E402
from trace_redox.penetration import penetration_pairs, penetration_self_terms  # noqa: E402


def main():
    model, count = sys.argv[1], int(sys.argv[2])
    elements = [int(z) for z in sys.argv[3:]] or [79]
    test = read(os.path.join(HERE, "data", "AuMgO_test.xyz"), ":")
    pick = np.random.default_rng(0).choice(len(test), count, replace=False)
    calc = TransformersACECalculator(model_path=model, device="cpu")
    torch.set_num_threads(2)
    mod = calc.model.electrostatics
    cap = []
    mod.register_forward_hook(lambda m, i, o: cap.append(o[1]))
    rows = {}
    for idx in pick:
        a = test[int(idx)].copy()
        a.info = {"charge": 0.0}
        a.calc = calc
        calc.reset()
        cap.clear()
        a.get_potential_energy()
        r = cap[-1]
        z = torch.as_tensor(a.numbers)
        n = len(a)
        states = r.states.detach()
        st = [int(round(float(s))) for s in states.tolist()]
        _, E_Z, _, _ = formal_ladder(z, int(mod.redox.max_charge))
        E_Z = torch.cat([torch.full((n, len(st) - int(E_Z.shape[1])), float("inf")), E_Z], 1)
        shift = (r.state_energies.detach()[:n] - E_Z).numpy()
        frames = formal_frame(z)
        sigma = (mod.free_widths(z) if mod.width is not None else r.sigma_free).detach()
        cell = torch.as_tensor(np.array(a.cell), dtype=torch.float64)
        pbc = tuple(bool(p) for p in a.pbc)
        pi_, pj, Xi = penetration_pairs(z, states, frames, sigma, torch.as_tensor(a.positions), cell=cell, pbc=pbc,
                                        model="gordon_kim")
        own = penetration_self_terms(z, states, frames, sigma, torch.as_tensor(a.positions), cell=cell, pbc=pbc,
                                     model="gordon_kim").numpy()
        first = st[0]
        f = frames.numpy()
        pi_ = pi_.numpy(); pj = pj.numpy(); Xi = Xi.numpy()
        for Z in elements:
            for s_ in st:
                if s_ == 0 and Z == 79:
                    continue
                k = s_ - first
                for u in np.where(a.numbers == Z)[0]:
                    if s_ == f[u] or not np.isfinite(shift[u, k]):
                        continue
                    m_i, m_j = pi_ == u, pj == u
                    val = (Xi[m_i, k, f[pj[m_i]] - first] - Xi[m_i, f[u] - first, f[pj[m_i]] - first]).sum() \
                        + (Xi[m_j, f[pi_[m_j]] - first, k] - Xi[m_j, f[pi_[m_j]] - first, f[u] - first]).sum() \
                        + own[u, k] - own[u, f[u] - first]
                    rows.setdefault((Z, s_), []).append((shift[u, k], val))
    print(f"=== {os.path.basename(os.path.dirname(model))}  ({len(pick)} test structures; Delta = learned shift, pi = overlap "
          f"one-body, eV)")
    for (Z, s_), v in sorted(rows.items()):
        d, p = np.array(v).T
        if len(d) < 5 or p.std() < 1e-9:
            continue
        b, a0 = np.polyfit(p, d, 1)
        r2 = 1.0 - ((d - (a0 + b * p)) ** 2).sum() / ((d - d.mean()) ** 2).sum()
        print(f"  Z={Z:2d} state {s_:+d}: n={len(d):4d}  slope b {b:+.3f}  R^2 {r2:.3f}  corr {np.corrcoef(p, d)[0, 1]:+.3f}"
              f"  Delta mean {d.mean():+.3f} sd {d.std():.3f}  pi mean {p.mean():+.3f} sd {p.std():.3f}")


if __name__ == "__main__":
    main()
