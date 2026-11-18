# One of the measurement scripts of the theory note's section 4c (2026-10-29); run from the repo root.
"""Which quantum ions the lower Au touches in the tilt scan and the site runs (the single-contact question)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, wetting_eda as W
G = W.geometries(); D = W.D
for r_au in (3.6, 3.8):
    sites = W.quantum_sites(G["flat"] + G["upright"], r_au=r_au)
    ions = [("Mg" if sum(s) % 2 == 0 else "O", D * np.array(s, float)) for s in sites]
    print(f"r_au {r_au}: {len(ions)} quantum ions; O at", [tuple(s) for s in sites if sum(s) % 2 == 1])
sites = W.quantum_sites(G["flat"] + G["upright"], r_au=3.6)
ions = [("Mg" if sum(s) % 2 == 0 else "O", D * np.array(s, float)) for s in sites]
O = D * np.array([1.0, 0.0, 0.0])
def report(tag, au):
    d = sorted((np.linalg.norm(au - p), sym, tuple(np.round(p / D).astype(int))) for sym, p in ions)
    print(f"{tag}: lower Au at {np.round(au, 3)}; nearest quantum ions:", ", ".join(f"{s}{idx} {r:.2f}" for r, s, idx in d[:5]))
for th in (0, 30, 45):
    t = np.radians(th); u = np.array([-np.sin(t), 0.0, np.cos(t)])
    report(f"tilt {th:2d} (2.6 A)", O + 2.6 * u)
for site, base in (("mgtop", np.array([0, 0, 2.89])), ("hollow", np.array([0.5 * D, -0.5 * D, 2.89]))):
    report(f"vertical {site}", base)
for g in ("flat", "upright"):
    for i, au in enumerate(G[g]):
        report(f"{g} Au{i}", np.asarray(au, float))
