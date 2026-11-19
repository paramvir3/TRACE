"""Which part of the single model softens the Au-Au wall (README, pre-registered 2026-11-12 23:53 IST; frozen weights,
a probe: nothing is trained, no package code is changed, nothing is written into a training run directory).

    PYTHONPATH=../.. python probe_auau_wall_terms.py SINGLE_RUN_DIR STEP31_RUN_DIR [threads=2] [out=FILE.json] [smoke=1] [h=0.005]

The bond force of the Au2 dimer is f = 1/2 (F_a - F_b).u, u the unit vector from Au b to Au a (minimum image),
positive = pushing the atoms apart.  With both Au moved rigidly along u at a frozen substrate, f = -dE/dd exactly.

1. Split.  E = E_net + E_q, E_q the charge module's own energy (RedoxResult.energy), and
   E_q = E_b0 + E_Pauli + E_ens: RedoxResult.energy_bound (the bound sector's minimum without free charge), SPEC 33's
   frame terms (the module's _bond_pauli_last + _bond_pauli_wall_last when their flags are on), and the rest.  Each
   part's bond force is its central difference in d (h = 0.005 A).  Checked per point: the parts' differences sum to
   the analytic f (1e-3 eV/A + 1e-3 |f|).
2. Reverts.  The network part must not see the redox settings (checked: E_net of every variant equals the base's to
   1e-8 eV), so a revert changes f only through E_q.  One setting at a time back to step 31's value, and all six.
3. Record: the induced dipoles and the charges on the two Au.
"""
import dataclasses
import json
import os
import sys
import time
import warnings

ARGS = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
THREADS = int(ARGS.get("threads", 2))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(THREADS)
from ase.io import read  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

H_FD = float(ARGS.get("h", 0.005))
SCAN = [2.00, 2.05, None, 2.15, 2.20, 2.25, 2.30, 2.40, 2.57]          # None: the structure's own distance
MAIN, OTHERS = 167, [222, 245, 115, 17, 28]
VARIANTS = [
    ("base (all on)", {}),
    ("no damping (SPEC 34)", dict(frame_induction_damping=False)),
    ("no anchor (SPEC 35)", dict(frame_polarizability_anchor=False)),
    ("anchor, scope 'all'", dict(frame_polarizability_scope="all")),
    ("no coupling bound (27)", dict(coupling_bound=None)),
    ("'block' stiffness (28.6)", dict(state_polarizability_stiffness="block")),
    ("'global' space (36)", dict(diabatic_space="global")),
    ("band 'phi1' (38)", dict(band_form="phi1")),
    ("all six (step 31's eqs)", dict(frame_induction_damping=False, frame_polarizability_anchor=False, coupling_bound=None,
                                    state_polarizability_stiffness="block", diabatic_space="global", band_form="phi1")),
]


def dimer(atoms):
    au = np.where(atoms.numbers == 79)[0]
    assert len(au) == 2, au
    v = atoms.get_distance(int(au[1]), int(au[0]), mic=True, vector=True)      # from Au b = au[1] to Au a = au[0]
    d = float(np.linalg.norm(v))
    return au, v / d, d


def at_distance(atoms, d):
    au, u, d0 = dimer(atoms)
    b = atoms.copy()
    b.info = dict(atoms.info)
    b.positions[au[0]] += 0.5 * (d - d0) * u
    b.positions[au[1]] -= 0.5 * (d - d0) * u
    return b


class Model:
    def __init__(self, run_dir):
        self.name = os.path.basename(os.path.normpath(run_dir))
        self.calc = TransformersACECalculator(model_path=os.path.join(run_dir, "model.pt"), device="cpu")
        torch.set_num_threads(THREADS)
        self.mod = self.calc.model.electrostatics
        self.base = self.mod.redox
        self.calls = 0

    def set(self, changes):
        self.mod.redox = dataclasses.replace(self.base, **changes)

    def evaluate(self, atoms, forces=True):
        a = atoms.copy()
        a.info = dict(atoms.info)
        a.calc = self.calc
        self.calc.reset()
        cap = []
        for name in ("_bond_pauli_last", "_bond_pauli_wall_last"):
            if hasattr(self.mod, name):
                setattr(self.mod, name, 0.0)
        hook = self.mod.register_forward_hook(lambda m, i, o: cap.append(o[1]))
        try:
            E = float(a.get_potential_energy())
            F = a.get_forces() if forces else None
        finally:
            hook.remove()
        self.calls += 1
        r, rc = cap[-1], self.mod.redox
        pauli = sum(float(getattr(self.mod, attr, 0.0)) for flag, attr in
                    (("bond_pauli", "_bond_pauli_last"), ("bond_pauli_wall", "_bond_pauli_wall_last")) if getattr(rc, flag, False))
        e_q, e_b0 = float(r.energy), float(r.energy_bound)
        au, u, d = dimer(a)
        out = {"d": d, "E": E, "net": E - e_q, "q": e_q, "b0": e_b0, "pauli": pauli, "ens": e_q - e_b0 - pauli,
               "forward_calls": len(cap)}
        if forces:
            out["f"] = float(0.5 * (F[au[0]] - F[au[1]]) @ u)
            out["f_ref"] = float(0.5 * (atoms.get_forces()[au[0]] - atoms.get_forces()[au[1]]) @ u) if atoms.calc is not None else None
            mu = r.mu.detach().cpu().numpy()[:len(a)]
            out["mu_along"] = [float(mu[i] @ u) for i in au]
            out["mu_norm"] = [float(np.linalg.norm(mu[i])) for i in au]
            out["q_au"] = [float(r.q[int(i)]) for i in au]
            out["free_au"] = [float(r.free_charge[int(i)]) for i in au]
        return out

    def split(self, atoms, d):
        """Analytic f and the parts' central differences at distance d (None: the structure's own)."""
        _, _, d0 = dimer(atoms)
        d = d0 if d is None else d
        mid = self.evaluate(atoms if d == d0 else at_distance(atoms, d))
        lo = self.evaluate(at_distance(atoms, d - H_FD), forces=False)
        hi = self.evaluate(at_distance(atoms, d + H_FD), forces=False)
        for k in ("E", "net", "q", "b0", "pauli", "ens"):
            mid["f_" + k] = -(hi[k] - lo[k]) / (2 * H_FD)
        mid["fd_gap"] = mid["f_E"] - mid["f"]
        mid["fd_ok"] = bool(abs(mid["fd_gap"]) <= 1e-3 + 1e-3 * abs(mid["f"]))
        if d != d0:
            mid["f_ref"] = None
        return mid


def attempt(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as exc:                                   # a variant the module refuses is a result, not a crash
        return {"refused": f"{type(exc).__name__}: {str(exc)[:200]}"}


def fmt(x, w=8, p=3):
    return f"{x:+{w}.{p}f}" if isinstance(x, (int, float)) and x is not None else f"{'-':>{w}s}"


def main():
    single_dir, ref_dir = sys.argv[1], sys.argv[2]
    smoke = ARGS.get("smoke") == "1"
    test = read(os.path.join(HERE, "data", "AuMgO_test.xyz"), ":")
    undoped_rank = {k: sum(1 for a in test[:k] if not (a.numbers == 13).any()) for k in [MAIN] + OTHERS}
    scan = [None] if smoke else SCAN
    variants = VARIANTS[:3] if smoke else VARIANTS
    others = OTHERS[:1] if smoke else OTHERS
    rec = {"h": H_FD, "single": single_dir, "reference": ref_dir, "scan": {}, "variants": {}, "others": {}}
    t0 = time.time()
    print(time.strftime("start %Y-%m-%d %H:%M %Z") + f"; threads {THREADS}; FD step {H_FD} A", flush=True)
    for k in [MAIN] + OTHERS:
        a = test[k]
        _, _, d = dimer(a)
        print(f"  test {k:3d}: {'doped' if (a.numbers == 13).any() else 'undoped'} (undoped ordinal {undoped_rank[k]}), "
              f"{len(a)} atoms, Au-Au {d:.3f} A", flush=True)
    for label, run in (("single", single_dir), ("step31", ref_dir)):
        m = Model(run)
        rc = m.base
        print(f"\n=== {label}: {m.name}\n    damping {rc.frame_induction_damping}, anchor {rc.frame_polarizability_anchor} "
              f"(scope {rc.frame_polarizability_scope}), bound {rc.coupling_bound}, stiffness {rc.state_polarizability_stiffness}, "
              f"space {rc.diabatic_space}, band {rc.band_form}; bond_pauli {rc.bond_pauli}, wall {rc.bond_pauli_wall}", flush=True)
        if hasattr(m.mod, "log_alpha_scale"):
            print(f"    crystal factor g = {float(torch.exp(m.mod.log_alpha_scale)):.4f}")
        # 1. the scan with the split
        print(f"  scan on test {MAIN} (bond forces in eV/A; parts by central differences; f_ref: DFT at the structure's own d)")
        print(f"  {'d':>6s} {'f':>8s} {'f_ref':>8s} {'net':>8s} {'E_q':>8s} | {'E_b0':>8s} {'Pauli':>8s} {'ens':>8s} | {'FD gap':>9s}  "
              f"{'mu.u a/b (e A)':>17s} {'q a/b':>15s}  calls")
        rows = []
        for d in scan:
            m.set({})
            s = m.split(test[MAIN], d)
            rows.append(s)
            print(f"  {s['d']:6.3f} {fmt(s['f'])} {fmt(s['f_ref'])} {fmt(s['f_net'])} {fmt(s['f_q'])} | {fmt(s['f_b0'])} {fmt(s['f_pauli'])} "
                  f"{fmt(s['f_ens'])} | {s['fd_gap']:+9.1e}{'' if s['fd_ok'] else ' FAIL'}  {s['mu_along'][0]:+8.3f}/{s['mu_along'][1]:+8.3f} "
                  f"{s['q_au'][0]:+7.3f}/{s['q_au'][1]:+7.3f}  {s['forward_calls']}", flush=True)
        rec["scan"][label] = rows
        # the other structures at their own geometry
        rec["others"][label] = {}
        for k in others:
            m.set({})
            rec["others"][label][k] = attempt(m.evaluate, test[k])
        if label == "single":
            # 2. the reverts
            at = [None] if smoke else [None, 2.20, 2.57]
            for name, changes in variants:
                m.set(changes)
                out = {"own": attempt(m.split, test[MAIN], None)}
                for d in at[1:]:
                    out[f"{d:.2f}"] = attempt(m.evaluate, at_distance(test[MAIN], d))
                out["others"] = {k: attempt(m.evaluate, test[k]) for k in others}
                rec["variants"][name] = out
                o = out["own"]
                print(f"  variant {name:26s}: " + (o["refused"] if "refused" in o else
                      f"f {o['f']:+8.3f} (net {o['f_net']:+8.3f}, E_b0 {o['f_b0']:+8.3f}, Pauli {o['f_pauli']:+7.3f}, ens {o['f_ens']:+8.3f}; "
                      f"FD gap {o['fd_gap']:+.1e}{'' if o['fd_ok'] else ' FAIL'})") + f"  [{time.time() - t0:.0f} s]", flush=True)
            m.set({})
        print(f"  evaluations so far: {m.calls}; {time.time() - t0:.0f} s", flush=True)
        del m
    out = ARGS.get("out")
    if out:
        with open(out, "w") as fh:
            json.dump(rec, fh, indent=1)
        print(f"record: {out}")
    report(rec, test)
    print(time.strftime("done %Y-%m-%d %H:%M %Z") + f"; {time.time() - t0:.0f} s")


def report(rec, test):
    own = lambda rows: next(r for r in rows if r.get("f_ref") is not None)
    s, r = own(rec["scan"]["single"]), own(rec["scan"]["step31"])
    D = s["f"] - r["f"]
    Dn, Dq = s["f_net"] - r["f_net"], s["f_q"] - r["f_q"]
    print(f"\n=== reading at test {MAIN}, d = {s['d']:.3f} A: DFT f {s['f_ref']:+.3f}; single {s['f']:+.3f}; step 31 {r['f']:+.3f}")
    print(f"  D = f(single) - f(step 31) = {D:+.3f} eV/A = network {Dn:+.3f} ({Dn / D:+.0%}) + E_q {Dq:+.3f} ({Dq / D:+.0%})")
    for k in ("b0", "pauli", "ens"):
        print(f"      E_q part {k:6s}: single {s['f_' + k]:+8.3f}, step 31 {r['f_' + k]:+8.3f}, difference {s['f_' + k] - r['f_' + k]:+8.3f}")
    base = rec["variants"].get("base (all on)")
    if base is None:
        return
    b = base["own"]
    net_gap = max((abs(v["own"]["net"] - b["net"]) for v in rec["variants"].values() if "refused" not in v["own"]), default=0.0)
    print(f"  check: the network part under every variant equals the base's to {net_gap:.1e} eV "
          f"({'pass' if net_gap <= 1e-8 else 'FAIL'}); base repeats the scan's f to {abs(b['f'] - s['f']):.1e}")
    ks = sorted(base["others"])
    print(f"\n  change of the bond force under each revert, f(v) - f(base) (eV/A); the hold criterion needs >= {0.5 * abs(Dq):.2f} "
          f"toward step 31 at {s['d']:.3f}")
    head = f"  {'variant':26s} {'own d':>8s} {'2.20':>8s} {'2.57':>8s} | " + " ".join(f"{'t' + str(k):>8s}" for k in ks) + " | share of D_q  same sign on"
    print(head)
    for name, v in rec["variants"].items():
        if name.startswith("base"):
            continue
        if "refused" in v["own"]:
            print(f"  {name:26s} refused: {v['own']['refused']}")
            continue
        d_own = v["own"]["f"] - b["f"]
        cols = []
        for key in ("2.20", "2.57"):
            cols.append(v[key]["f"] - base[key]["f"] if key in v and "refused" not in v[key] else None)
        oth = [v["others"][k]["f"] - base["others"][k]["f"] if "refused" not in v["others"][k] else None for k in ks]
        toward = -np.sign(Dq)                                              # the direction that undoes D_q
        same = 1 + sum(1 for x in oth if x is not None and np.sign(x) == toward) if np.sign(d_own) == toward else 0
        print(f"  {name:26s} {fmt(d_own)} {fmt(cols[0])} {fmt(cols[1])} | " + " ".join(fmt(x) for x in oth)
              + f" | {-d_own / Dq:+10.0%}   {same} of {1 + len(ks)}")
    print("\n  the other structures: bond force, DFT / single / step 31 (eV/A)")
    for k in ks:
        a, c = base["others"][k], rec["others"]["step31"][k]
        print(f"    test {k:3d} (Au-Au {a['d']:.3f} A): {a['f_ref']:+8.3f} / {a['f']:+8.3f} / {c['f']:+8.3f}")


if __name__ == "__main__":
    main()
