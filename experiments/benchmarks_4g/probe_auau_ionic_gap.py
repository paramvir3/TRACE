"""Which term of the configuration energy lowers Au+Au- under compression (README, pre-registered 2026-11-13; frozen
weights, a probe: nothing trained, nothing written into a run directory).

    PYTHONPATH=../.. python probe_auau_ionic_gap.py RUN_DIR [single=1] [threads=2]

On test 167's scan (both Au moved rigidly along their bond), the configurations Au0Au0, Au_a+Au_b- and Au_a-Au_b+
(the substrate's units as in the most populated configuration) and, per term of SPEC 0 (probe_t5_terms.terms, whose
terms sum to H's diagonal to 1e-9, asserted there), the gap G = E(Au_a+Au_b-) - E(Au0Au0); also the hop between them
and the populations.  single=1: the single model with 'global' space, band 'phi1' and no coupling bound, which are
exactly inactive on this undoped cell (probe_auau_wall_terms.py: 0.000 eV/A); the bond force is printed to check it.
"""
import dataclasses
import os
import sys
import warnings

warnings.simplefilter("ignore")
RUN = sys.argv[1]
OPTS = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
sys.argv = [sys.argv[0], "", ""] + [a for a in sys.argv[2:] if a.startswith(("threads=", "h="))]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import probe_auau_wall_terms as P  # noqa: E402
import probe_t5_terms as PT  # noqa: E402
from ase.io import read  # noqa: E402

GRID = [2.57, 2.30, 2.20, 2.16, 2.12, 2.10, 2.089, 2.07, 2.06, 2.04, 2.00]
ORDER = ["ladder", "shifts", "self", "screened", "overlap", "ind26", "alpha28", "band"]
PT._install()
m = P.Model(RUN)
if OPTS.get("single") == "1":
    m.base = dataclasses.replace(m.base, diabatic_space="global", band_form="phi1", coupling_bound=None)
m.set({})
test = read(os.path.join(P.HERE, "data", "AuMgO_test.xyz"), ":")
a0 = test[P.MAIN]
au, u, d0 = P.dimer(a0)
print(f"=== {m.name}{'  (global, phi1, no bound: inactive on this cell)' if OPTS.get('single') == '1' else ''}")
rows = []
for d in GRID:
    a = P.at_distance(a0, d)
    PT.terms(m.calc, a)
    pc = PT.STORE["per_config"]
    cfg, out, p, H = pc["configs"], pc["out"], pc["p"], pc["H"]
    f = float(0.5 * (m.calc.results["forces"][au[0]] - m.calc.results["forces"][au[1]]) @ u) if "forces" in m.calc.results else float("nan")
    top = int(torch.argmax(p))
    cols = [c for c in range(cfg.shape[1]) if c not in (int(au[0]), int(au[1]))]
    same = (cfg[:, cols] == cfg[top, cols]).all(1)
    pick = {}
    for name, (na, nb) in (("neutral", (0, 0)), ("a+b-", (1, -1)), ("a-b+", (-1, 1))):
        hit = (same & (cfg[:, int(au[0])] == na) & (cfg[:, int(au[1])] == nb)).nonzero().flatten()
        pick[name] = int(hit[0]) if hit.numel() else None
    row = {"d": d, "f": f, "n": int(cfg.shape[0]), "pick": pick, "p": {k: (float(p[v]) if v is not None else None) for k, v in pick.items()}}
    if pick["neutral"] is not None and pick["a+b-"] is not None:
        i0, i1 = pick["neutral"], pick["a+b-"]
        row["gap"] = {k: float(out[k][i1] - out[k][i0]) for k in out}
        row["G"] = float(H[i1, i1] - H[i0, i0])
        row["hop"] = float(H[i0, i1])
    rows.append(row)
keys = [k for k in ORDER if any(k in r.get("gap", {}) for r in rows)]
extra = sorted({k for r in rows for k in r.get("gap", {})} - set(keys))
keys += extra
print(f"  {'d':>6s} {'f':>8s} {'G':>8s} | " + " ".join(f"{k:>9s}" for k in keys) + f" | {'hop':>7s} | p(neutral) p(a+b-) p(a-b+) | configs")
for r in rows:
    if "gap" not in r:
        print(f"  {r['d']:6.3f} {r['f']:+8.3f}  a configuration is not in the ensemble: {r['pick']}")
        continue
    pp = r["p"]
    print(f"  {r['d']:6.3f} {r['f']:+8.3f} {r['G']:+8.3f} | " + " ".join(f"{r['gap'].get(k, 0.0):+9.3f}" for k in keys)
          + f" | {r['hop']:+7.3f} | {pp['neutral']:9.4f} {pp['a+b-']:7.4f} " + (f"{pp['a-b+']:7.4f}" if pp['a-b+'] is not None else "   none") + f" | {r['n']}")
get = lambda d: next((r for r in rows if abs(r["d"] - d) < 1e-9 and "gap" in r), None)
hi, lo = get(2.20), get(2.06)
if hi and lo:
    fall = lo["G"] - hi["G"]
    print(f"\n  change of G from 2.20 to 2.06 A: {fall:+.3f} eV; by term (share of the change):")
    for k in keys:
        dk = lo["gap"].get(k, 0.0) - hi["gap"].get(k, 0.0)
        print(f"    {k:9s} {dk:+8.3f}  ({dk / fall:+.0%})")
    ind = sum(lo["gap"].get(k, 0.0) - hi["gap"].get(k, 0.0) for k in ("screened", "ind26", "alpha28"))
    print(f"    screened + ind26 + alpha28: {ind:+.3f} ({ind / fall:+.0%});  hop {hi['hop']:+.3f} -> {lo['hop']:+.3f} eV")


# ---- the bond force by term (Hellmann-Feynman on the ensemble's free energy; added 2026-11-13 00:2x) ----------------
# F = -kT ln Tr exp(-H / kT) gives dF/dd = Tr(rho dH/dd) exactly, so the ensemble's bond force splits into
#   f_term = -sum_n p_n d E_term(n)/dd   (the configuration energies, term by term)   and
#   f_hop  = -sum_{n != m} rho_nm d H_nm/dd   (the couplings),
# with dH/dd by central differences (step H_FD) at a fixed list of configurations.  Checked per point against the
# central difference of the free energy itself (res: E_q - e_b0 - frame).
from trace_redox.diabatic import density_matrix  # noqa: E402

HF_GRID = [2.30, 2.20, 2.16, 2.12, 2.10, 2.089, 2.07, 2.06, 2.04]


def snapshot(d):
    res = PT.terms(m.calc, P.at_distance(a0, d))
    pc = PT.STORE["per_config"]
    kT = float(m.mod._ensemble_last["kT"])
    free = sum(v for k, v in res.items() if k not in ("e_b0", "frame", "elim_const"))
    return {"cfg": pc["configs"].clone(), "out": {k: v.clone() for k, v in pc["out"].items()}, "H": pc["H"].clone(),
            "rho": density_matrix(pc["H"], kT), "F": free, "frame": res.get("frame", 0.0) + res.get("elim_const", 0.0) + res.get("e_b0", 0.0),
            "f": float(0.5 * (m.calc.results["forces"][au[0]] - m.calc.results["forces"][au[1]]) @ u)}


def align(ref, other):
    """other's rows in ref's order (None if the two lists hold different configurations)."""
    if ref["cfg"].shape == other["cfg"].shape and bool((ref["cfg"] == other["cfg"]).all()):
        return list(range(int(ref["cfg"].shape[0])))
    if ref["cfg"].shape[0] != other["cfg"].shape[0]:
        return None
    index = {tuple(r.tolist()): i for i, r in enumerate(other["cfg"])}
    perm = [index.get(tuple(r.tolist())) for r in ref["cfg"]]
    return None if any(x is None for x in perm) else perm


h = P.H_FD
print(f"\n  bond force of the ensemble by term (eV/A; f_ens = sum of the columns; FD step {h} A)")
print(f"  {'d':>6s} {'f':>8s} {'f_ens':>8s} | " + " ".join(f"{k:>9s}" for k in keys) + f" | {'hops':>8s} | {'check':>9s} | frame + constants")
hf = []
for d in HF_GRID:
    mid, lo, hi = snapshot(d), snapshot(d - h), snapshot(d + h)
    pl, ph = align(mid, lo), align(mid, hi)
    if pl is None or ph is None:
        print(f"  {d:6.3f} {mid['f']:+8.3f}  the configuration list changes within +-{h} A ({lo['cfg'].shape[0]}/{mid['cfg'].shape[0]}/{hi['cfg'].shape[0]}); skipped")
        continue
    p = torch.diagonal(mid["rho"])
    ft = {k: float(-(p * (hi["out"][k][ph] - lo["out"][k][pl])).sum() / (2 * h)) for k in mid["out"]}
    dH = (hi["H"][ph][:, ph] - lo["H"][pl][:, pl]) / (2 * h)
    off = dH - torch.diag(torch.diagonal(dH))
    f_hop = float(-(mid["rho"] * off).sum())
    f_ens_fd = -(hi["F"] - lo["F"]) / (2 * h)
    f_frame = -(hi["frame"] - lo["frame"]) / (2 * h)
    total = sum(ft.values()) + f_hop
    hf.append({"d": d, "f": mid["f"], "terms": ft, "hops": f_hop, "f_ens": f_ens_fd, "frame": f_frame})
    print(f"  {d:6.3f} {mid['f']:+8.3f} {f_ens_fd:+8.3f} | " + " ".join(f"{ft.get(k, 0.0):+9.3f}" for k in keys)
          + f" | {f_hop:+8.3f} | {total - f_ens_fd:+9.1e} | {f_frame:+.3f}", flush=True)
