"""D31, M1 and M3 (README, pre-registered 2026-11-10 10:55 IST): the training set's class offsets split into the model's
terms, and the residual against the dimer's geometry (frozen weights; a probe, no model change).

Usage (benchmarks_4g): python probe_class_terms.py MODEL.pt [...]          (env CLASS_SPLIT=train|test, default train)
                       D31_STATES=1 python probe_class_terms.py MODEL.pt    (M3: step 27's per-state law switched on, frozen;
                                                                             totals only, plus wet_d at the T5 references)

For every structure at its DFT geometry: residual = model - DFT; the model's energy split as
    E = E_backbone + E_q,    E_backbone = E - E_q (ACE backbone + per-element baseline),
    E_q = sum of probe_t5_terms.terms() (asserted there per structure to 1e-9),
the tilt theta of Au-Au from the surface normal, the contact (lower) Au's height above the top layer and its site (species
of the nearest top-layer atom and the lateral distance to it), and the ensemble's mean free charge on each Au.
Reported per doping: the class offset <res>_flat - <res>_upright (must equal probe_class_residuals.py's), each term's
flat - upright difference and the double difference DD = doped - undoped; the residual's slope in theta with
class-specific intercepts (res = a_class + s theta), and in the contact height.  Records go to
runs/<run>/scores/D31_class_terms[_states].json.
"""
import dataclasses
import json
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

import probe_t5_terms as PT  # noqa: E402
from ase.io import read  # noqa: E402
from eval_aumgo import references  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_redox.diabatic import density_matrix  # noqa: E402

SPLIT = os.environ.get("CLASS_SPLIT", "train")
STATES = os.environ.get("D31_STATES") == "1"
ORDER = ["backbone", "e_b0", "ladder", "shifts", "self", "screened", "exclusion18", "overlap", "ind26", "alpha28", "band",
         "coherent", "entropy", "frame"]


def geometry(a):
    """Tilt, heights and site, with the slab's surface that faces the dimer found by minimum image along z (in these
    cells the dimer sits below the slab: Au at z ~ 8, the slab at 10.5-21.3 A); contact = the Au nearer that surface."""
    au = np.where(a.numbers == 79)[0]
    p = a.positions[au]
    v = p[1] - p[0]
    theta = float(np.degrees(np.arccos(abs(v[2]) / np.linalg.norm(v))))
    L = float(a.cell[2, 2])
    wrap = lambda d: (d + 0.5 * L) % L - 0.5 * L
    sub = np.where(a.numbers != 79)[0]
    zc = float(p[:, 2].mean())
    zs = a.positions[sub[int(np.argmin(np.abs(wrap(a.positions[sub, 2] - zc))))], 2]
    top = sub[np.abs(wrap(a.positions[sub, 2] - zs)) < 0.5]                   # the facing layer (Mg and O)
    zt = float(zs + wrap(a.positions[top, 2] - zs).mean())
    h = np.abs(wrap(p[:, 2] - zt))
    lo = au[int(np.argmin(h))]
    d = a.get_distances(lo, top, mic=True, vector=True)
    lat = np.linalg.norm(d[:, :2], axis=1)
    k = int(np.argmin(lat))
    return {"theta": theta, "h_contact": float(h.min()), "h_far": float(h.max()),
            "site": a.get_chemical_symbols()[top[k]], "site_lateral": float(lat[k]), "contact_index": int(lo),
            "contact_O": float(a.get_distances(lo, top[a.numbers[top] == 8], mic=True).min())}


def populations(mod, atoms):
    last = mod._ensemble_last
    p = torch.diagonal(density_matrix(last["H"].detach(), float(last["kT"])))
    au = np.where(atoms.numbers == 79)[0]
    return [float(p @ last["configs"][:, int(i)].to(torch.float64)) for i in au]


def offsets(recs, key):
    out = {}
    for d in ("doped", "undoped"):
        f = np.array([r[key] for r in recs if r["doping"] == d and r["class"] == "flat"])
        u = np.array([r[key] for r in recs if r["doping"] == d and r["class"] == "upright"])
        out[d] = (f.mean() - u.mean(), np.sqrt(f.var(ddof=1) / len(f) + u.var(ddof=1) / len(u)))
    return out


def slope(recs, d, x):
    rows = [r for r in recs if r["doping"] == d]
    X = np.array([[r["class"] == "flat", r["class"] == "upright", r[x]] for r in rows], float)
    y = np.array([r["residual"] for r in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    cov = (res @ res / (len(y) - 3)) * np.linalg.inv(X.T @ X)
    return float(beta[2]), float(np.sqrt(cov[2, 2]))


def main():
    data = read(os.path.join(HERE, "data", f"AuMgO_{SPLIT}.xyz"), ":")
    if not STATES:
        PT._install()
    for model in sys.argv[1:]:
        run = os.path.basename(os.path.dirname(os.path.abspath(model)))
        calc = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(3)
        mod = calc.model.electrostatics
        recs = []
        for a0 in data:
            g = geometry(a0)
            rec = {"doping": "doped" if (a0.numbers == 13).any() else "undoped",
                   "class": "flat" if g["theta"] > 45 else "upright", **g}
            if STATES:
                a = a0.copy(); a.info = dict(a0.info); a.calc = calc
                mod.redox = dataclasses.replace(mod.redox, bond_pauli_states=False); calc.reset()
                e_off = a.get_potential_energy()
                mod.redox = dataclasses.replace(mod.redox, bond_pauli_states=True); calc.reset()
                e_on = a.get_potential_energy()
                mod.redox = dataclasses.replace(mod.redox, bond_pauli_states=False); calc.reset()
                rec.update(residual=1e3 * (e_off - a0.get_potential_energy()), residual_states=1e3 * (e_on - a0.get_potential_energy()))
            else:
                t = PT.terms(calc, a0)
                total = float(calc.results["energy"])
                rec["residual"] = 1e3 * (total - a0.get_potential_energy())
                rec["backbone"] = 1e3 * (total - sum(t.values()))
                rec.update({k: 1e3 * v for k, v in t.items()})
                rec["q_au"] = populations(mod, a0)
            recs.append(rec)
        tag = "_states" if STATES else ""
        with open(os.path.join(HERE, "runs", run, "scores", f"D31_class_terms{tag}_{SPLIT}.json"), "w") as fh:
            json.dump(recs, fh)
        n = {(d, c): sum(r["doping"] == d and r["class"] == c for r in recs) for d in ("doped", "undoped") for c in ("flat", "upright")}
        print(f"=== {run}  ({SPLIT}, {len(recs)} structures {n}; meV per structure)")
        off = offsets(recs, "residual")
        print(f"  class offset (residual, flat - upright): doped {off['doped'][0]:+.1f} +- {off['doped'][1]:.1f}, "
              f"undoped {off['undoped'][0]:+.1f} +- {off['undoped'][1]:.1f}, doped - undoped {off['doped'][0] - off['undoped'][0]:+.1f}")
        if STATES:
            on = offsets(recs, "residual_states")
            print(f"  M3, step 27's per-state law on (frozen): doped {on['doped'][0]:+.1f} (change {on['doped'][0] - off['doped'][0]:+.1f}), "
                  f"undoped {on['undoped'][0]:+.1f} (change {on['undoped'][0] - off['undoped'][0]:+.1f})")
            R = {}
            for key, at in references().items():
                a = at.copy(); a.info = dict(at.info); a.calc = calc
                out = []
                for flag in (False, True):
                    mod.redox = dataclasses.replace(mod.redox, bond_pauli_states=flag); calc.reset()
                    out.append(a.get_potential_energy())
                R[key] = out
            mod.redox = dataclasses.replace(mod.redox, bond_pauli_states=False)
            wd = [1e3 * (R[("doped", "wetting")][i] - R[("doped", "non-wetting")][i]) for i in (0, 1)]
            wu = [1e3 * (R[("undoped", "wetting")][i] - R[("undoped", "non-wetting")][i]) for i in (0, 1)]
            print(f"  M3 at the T5 references (unrelaxed): wet_d {wd[0]:+.1f} -> {wd[1]:+.1f} (change {wd[1] - wd[0]:+.1f}), "
                  f"wet_u {wu[0]:+.1f} -> {wu[1]:+.1f} (change {wu[1] - wu[0]:+.1f})")
            continue
        keys = [k for k in ORDER if any(k in r for r in recs)]
        print(f"  {'term':11s} {'doped f-u':>11s} {'undoped f-u':>12s} {'DD':>9s}")
        tot = {"doped": 0.0, "undoped": 0.0}
        for k in keys:
            for r in recs:
                r.setdefault(k, 0.0)
            o = offsets(recs, k)
            tot["doped"] += o["doped"][0]; tot["undoped"] += o["undoped"][0]
            print(f"  {k:11s} {o['doped'][0]:+11.1f} {o['undoped'][0]:+12.1f} {o['doped'][0] - o['undoped'][0]:+9.1f}")
        print(f"  {'model total':11s} {tot['doped']:+11.1f} {tot['undoped']:+12.1f} {tot['doped'] - tot['undoped']:+9.1f}")
        for d in ("doped", "undoped"):
            s, se = slope(recs, d, "theta"); sh, seh = slope(recs, d, "h_contact")
            q = np.array([sum(r["q_au"]) for r in recs if r["doping"] == d])
            print(f"  {d:8s} residual slope in theta {s:+.2f} +- {se:.2f} meV/deg; in the contact height {sh:+.1f} +- {seh:.1f} meV/A; "
                  f"<q(Au2)> {q.mean():+.3f} (range {q.min():+.3f} .. {q.max():+.3f})")
        sd, sed = slope(recs, "doped", "theta"); su, seu = slope(recs, "undoped", "theta")
        print(f"  theta slope, doped - undoped: {sd - su:+.2f} +- {np.hypot(sed, seu):.2f} meV/deg")


if __name__ == "__main__":
    main()
