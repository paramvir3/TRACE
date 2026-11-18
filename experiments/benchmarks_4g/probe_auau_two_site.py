"""SPEC 40's mechanism check (README, pre-registered 2026-11-13 evening; frozen weights, a probe: nothing trained,
nothing written into a run directory).

    PYTHONPATH=../.. python probe_auau_two_site.py RUN_DIR [threads=2] [h=0.0005]

On test 167's rigid Au-Au scan: the bond force of the charge module's energy E_q alone (central differences of E_q),
as trained and with SPEC 39 (frame_pair), SPEC 29 (spin_pairing) and SPEC 40 (two_site) switched on one after the
other; the model's total bond force (autograd) beside it; and, in the 'global' space (exactly the same model on this
undoped cell when no coupling sits above the bound's knee; the total force is printed to check it), the element
between the neutral and the Au+Au- configuration.  The backbone of a model trained without SPEC 39 / 40 holds the
Au-Au repulsion at the data's distances, so the TOTAL force with the terms on counts it twice: only E_q's is read.
"""
import dataclasses
import json
import os
import sys
import warnings

warnings.simplefilter("ignore")
RUN = sys.argv[1]
sys.argv = [sys.argv[0], "", ""] + [a for a in sys.argv[2:] if a.startswith(("threads=", "h="))]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import probe_auau_wall_terms as P  # noqa: E402
from ase.io import read  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

GRID = [2.00, 2.05, 2.089, 2.15, 2.20, 2.25, 2.30, 2.40, 2.57]
VARIANTS = [("as trained", {}),
            ("+ SPEC 39", dict(frame_pair=True)),
            ("+ SPEC 39, 29", dict(frame_pair=True, spin_pairing=True)),
            ("+ SPEC 39, 29, 40", dict(frame_pair=True, spin_pairing=True, two_site=True))]
h = P.H_FD
m = P.Model(RUN)
test = read(os.path.join(P.HERE, "data", "AuMgO_test.xyz"), ":")
a0 = test[P.MAIN]
au, u, d0 = P.dimer(a0)
f_dft = float(0.5 * (a0.get_forces()[au[0]] - a0.get_forces()[au[1]]) @ u)
two = json.load(open(os.path.join(P.HERE, "..", "..", "trace_redox", "data", "two_site.json")))["pairs"]["79-79"]
gas = PchipInterpolator(np.array(two["R"]), np.array(two["E_gs"])).derivative()
print(f"=== {m.name}: test {P.MAIN}, the structure's own Au-Au distance {d0:.3f} A, DFT bond force there {f_dft:+.3f} eV/A; FD step {h} A")
print(f"    gas-phase first principles, -dE_gs/dR (monotone cubic through the table's points): "
      + "  ".join(f"{d:.3f}: {-float(gas(d)):+.2f}" for d in GRID))
out = {}
for name, changes in VARIANTS:
    m.set(changes)
    rows = []
    for d in GRID:
        mid = m.evaluate(P.at_distance(a0, d), forces=True)
        lo = m.evaluate(P.at_distance(a0, d - h), forces=False)
        hi = m.evaluate(P.at_distance(a0, d + h), forces=False)
        f_q = -(hi["q"] - lo["q"]) / (2 * h)
        f_tot_fd = -(hi["E"] - lo["E"]) / (2 * h)
        rows.append({"d": d, "f_q": f_q, "f": mid["f"], "f_fd": f_tot_fd, "q": mid["q"], "E": mid["E"]})
    out[name] = rows
    print(f"\n  {name}")
    print(f"  {'d':>6s} {'f_q (E_q alone)':>16s} {'f total':>9s} {'f total, FD':>12s} {'f - f_q':>9s} {'E_q - E_q(2.57)':>16s}")
    for r in rows:
        print(f"  {r['d']:6.3f} {r['f_q']:+16.3f} {r['f']:+9.3f} {r['f_fd']:+12.3f} {r['f'] - r['f_q']:+9.3f} {r['q'] - rows[-1]['q']:+16.4f}", flush=True)

# ---- the covalent-ionic element over the scan, in the 'global' space ---------------------------------------------
print("\n  the element between the neutral and the Au_a+ Au_b- configuration ('global' space; f: total bond force there)")
for name, changes in (VARIANTS[0], VARIANTS[-1]):
    m.mod.redox = dataclasses.replace(m.base, diabatic_space="global", band_form="phi1", coupling_bound=None, **changes)
    line = []
    for d in GRID:
        r = m.evaluate(P.at_distance(a0, d), forces=True)
        last = m.mod._ensemble_last
        cfg, H = last["configs"], last["H"].detach()
        p = torch.diagonal(torch.matrix_exp(-(H - torch.diag(H).min() * torch.eye(len(H), dtype=H.dtype)) / float(last["kT"])))
        top = int(torch.argmax(p))
        cols = [c for c in range(cfg.shape[1]) if c not in (int(au[0]), int(au[1]))]
        same = (cfg[:, cols] == cfg[top, cols]).all(1)
        i0 = (same & (cfg[:, int(au[0])] == 0) & (cfg[:, int(au[1])] == 0)).nonzero().flatten()
        i1 = (same & (cfg[:, int(au[0])] == 1) & (cfg[:, int(au[1])] == -1)).nonzero().flatten()
        hop = float(H[int(i0[0]), int(i1[0])]) if i0.numel() and i1.numel() else float("nan")
        gap = float(H[int(i1[0]), int(i1[0])] - H[int(i0[0]), int(i0[0])]) if i0.numel() and i1.numel() else float("nan")
        line.append((d, hop, gap, r["f"]))
    print(f"  {name}")
    print("    " + "  ".join(f"{d:.3f}: hop {hop:+.3f} gap {gap:+.3f} f {f:+.2f}" for d, hop, gap, f in line), flush=True)
