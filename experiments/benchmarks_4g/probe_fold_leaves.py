"""The fold's lowest leaves on the four T5 references (2026-11-02; reported in the theory note §4c, "The doped
wetting energy, located"): each configuration's weight, H_ii and Au charges, the fold dressing per configuration and
the eight lowest leaves (O donor -> Au acceptor, kind, Delta, coupling v).  Note: "bottom" here is the Au with the
smaller z, which in these cells is the FAR Au (the slab lies above it); probe_fold_leaves_otop.py labels by the O-top
distance.  Usage (repo root): python experiments/benchmarks_4g/probe_fold_leaves.py MODEL.pt"""
import sys, warnings
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
warnings.simplefilter("ignore")
import numpy as np, torch
torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from flashace.calculator import TransformersACECalculator
from eval_aumgo import AU, references
refs = references()
calc = TransformersACECalculator(model_path=sys.argv[1], device="cpu"); torch.set_num_threads(3)
mod = calc.model.electrostatics
for key in (("undoped", "non-wetting"), ("doped", "non-wetting"), ("undoped", "wetting"), ("doped", "wetting")):
    at = refs[key].copy(); at.info = {"charge": 0.0}; at.calc = calc; calc.reset(); mod._downfold_last = None
    E = at.get_potential_energy()
    last = mod._ensemble_last; H = last["H"].detach().numpy(); cfg = last["configs"]
    w, V = np.linalg.eigh(H); p = V[:, 0] ** 2
    au = list(AU); z = at.positions[:, 2]; bottom = au[int(np.argmin(z[au]))]
    print(f"=== {key}: {cfg.shape[0]} configurations; Au atoms {au} (bottom/contact {bottom}); E {E:.4f}")
    nz = lambda c: {i: int(c[i]) for i in range(c.shape[0]) if int(c[i]) != 0 and i < len(at)}
    for i in range(cfg.shape[0]):
        c = cfg[i]
        print(f"   config {i}: p {p[i]:.3f}, H_ii {H[i, i]:+.3f}, Au charges {[int(c[a]) for a in au]}, nonzero non-frame charges {dict(list(nz(c - cfg[0]).items())[:8])}")
    rec = mod._downfold_last or {}
    print(f"   fold: {rec.get('leaves')} leaves, delta_min {rec.get('delta_min')}, dressing {[round(x, 3) for x in rec.get('dressing', [])]}")
    for (ci, o, g, k, d, v) in sorted(rec.get("lowest_leaves", []), key=lambda x: x[4])[:8]:
        print(f"     leaf: config {ci}, O {o} -> Au {g} ({'bottom' if g == bottom else 'top'}), kind {k}, Delta {d:+.3f} eV, v {v:+.3f} eV, O-Au {np.linalg.norm(at.positions[o] - at.positions[g]):.2f} A")
