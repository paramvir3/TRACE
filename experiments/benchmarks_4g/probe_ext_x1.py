"""X1 (benchmarks_4g README, pre-registered 2026-11-10 16:10 IST): SPEC 28 on the atom's own stiffness, measured on
frozen weights through the SHIPPED path (RedoxConfig.state_polarizability_stiffness; nothing is patched).

Usage: python probe_ext_x1.py MODEL.pt FORM PART [PART ...]
  FORM  checkpoint (the config as the loader restores it), onsite or block (the key replaced on the loaded module)
  PART  inphase   criterion (b): the module's own SPEC 28 term of the in-phase transfer, 2x1 against twice 1x1
        single    reported: one copy's transfer at 2x1 / 2x2 against the cell's in-phase transfer (the image term)
        cells     reported: E per cell and forces of the replicated cell against the cell
        table2    criterion (c) with FORM = checkpoint; reported with onsite: Table II by eval_aumgo's own protocol
Writes nothing into the run directory."""
import dataclasses
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np
import torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import au_o_bond, references, relax

model_path, form, parts = sys.argv[1], sys.argv[2], sys.argv[3:]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    calc = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
said = [str(w.message)[:110] for w in caught if "state_polarizability_stiffness" in str(w.message)]
print(f"model {model_path.split('/')[-2]}\nloaded with state_polarizability_stiffness = {mod.redox.state_polarizability_stiffness!r}"
      f"   loader warning: {said if said else 'none'}")
if form != "checkpoint":
    mod.redox = dataclasses.replace(mod.redox, state_polarizability_stiffness=form)
print(f"evaluated with state_polarizability_stiffness = {mod.redox.state_polarizability_stiffness!r}\n")
refs = references()
NAT = 110
captured = {}
orig = mod._diabatic_solution


def spy(e_hat, states, b, W, L, K, Q, T=None, **kw):
    captured.update(e_hat=e_hat.detach(), states=states.detach(), b=b.detach(), W=W.detach(), pairs=kw.get("pairs"),
                    diag=kw.get("diag_correction"))
    return orig(e_hat, states, b, W, L, K, Q, T, **kw)


mod._diabatic_solution = spy


def terms(cfg):
    c = captured; st = c["states"]; n = cfg.to(c["W"].dtype).unsqueeze(0)
    idx = torch.round(n - st[0]).to(torch.long)
    out = dict(ladder=float(c["e_hat"].gather(1, idx.T).sum()), linear=float(n @ c["b"]),
               quadratic=float(0.5 * ((n @ c["W"]) * n).sum()), pairs=0.0, spec28=0.0)
    if c["pairs"] is not None and int(c["pairs"][0].numel()):
        pi, pj, Xi = c["pairs"]; p = torch.arange(int(pi.numel()))
        out["pairs"] = float(Xi.detach()[p, idx[0, pi], idx[0, pj]].sum())
    if c["diag"] is not None:
        with torch.no_grad():
            out["spec28"] = float(c["diag"](cfg.unsqueeze(0)))
    return out


def evaluate(key, rep, forces=False):
    a = refs[key].copy(); a.set_constraint(); a = a.repeat(rep); a.info = {"charge": 0.0}
    a.calc = calc; calc.reset(); mod._ensemble_last = None
    e = a.get_potential_energy(); f = a.get_forces() if forces else None
    last = mod._ensemble_last
    cfg = last["configs"].to(torch.long); d = torch.diagonal(last["H"].detach())
    return e, f, cfg, cfg[int(torch.argmin(d))].clone(), len(a) // NAT


def excited(frame, copies, qa, qb):
    x = frame.clone()
    for k in copies:
        x[108 + NAT * k] += qa; x[109 + NAT * k] += qb
    return x


TRANSFERS = (("Au(+1,-1)", (1, -1)), ("Au(-1,+1)", (-1, 1)))
UNDOPED = (("undoped", "non-wetting"), ("undoped", "wetting"))

if "inphase" in parts or "single" in parts:
    worst = 0.0
    for key in UNDOPED:
        base = {}
        reps = [(1, 1, 1), (2, 1, 1)] + ([(2, 2, 1)] if "single" in parts else [])
        for rep in reps:
            _, _, cfg, frame, ncopy = evaluate(key, rep)
            if rep == (1, 1, 1):
                frame1 = frame.clone()
            else:                                      # the comparison needs the same reference state in every copy
                same = all(torch.equal(frame[NAT * k:NAT * (k + 1)], frame1[:NAT]) for k in range(ncopy))
                print(f"   {key} {rep}: lowest diagonal configuration is the cell's in every copy: {same}; "
                      f"selected {int(cfg.shape[0])}")
            f0 = terms(frame)
            for name, (qa, qb) in TRANSFERS:
                t = terms(excited(frame, range(ncopy), qa, qb))
                base[(rep, name, "all")] = {k: t[k] - f0[k] for k in t}
                t = terms(excited(frame, [0], qa, qb))
                base[(rep, name, "one")] = {k: t[k] - f0[k] for k in t}
        print(f"== {key}")
        for name, _ in TRANSFERS:
            one = base[((1, 1, 1), name, "all")]
            if "inphase" in parts:
                two = base[((2, 1, 1), name, "all")]
                d28 = 1e3 * (two["spec28"] - 2 * one["spec28"]); worst = max(worst, abs(d28))
                dtot = 1e3 * (sum(two.values()) - 2 * sum(one.values()))
                print(f"   in-phase {name}: SPEC 28 term 1x1 {1e3 * one['spec28']:+10.3f}  2x1 {1e3 * two['spec28']:+10.3f}  "
                      f"2x1 - 2*(1x1) {d28:+.6f} meV;  whole diagonal element 2x1 - 2*(1x1) {dtot:+.6f} meV")
            if "single" in parts:
                for rep in ((2, 1, 1), (2, 2, 1)):
                    s = base[(rep, name, "one")]
                    print(f"   one copy {name} at {rep[0]}x{rep[1]}: excitation energy {sum(s.values()):+.4f} eV, the cell's "
                          f"in-phase {sum(one.values()):+.4f} eV, difference {1e3 * (sum(s.values()) - sum(one.values())):+9.3f} meV  ("
                          + ", ".join(f"{k} {1e3 * (s[k] - one[k]):+.3f}" for k in s) + ")")
    if "inphase" in parts:
        print(f"criterion (b): max |dE28(2x1) - 2 dE28(1x1)| = {worst:.2e} meV (<= 1e-3: {worst <= 1e-3})")

if "cells" in parts:
    for key in UNDOPED + (("doped", "non-wetting"), ("doped", "wetting")):
        e1, f1, cfg1, _, _ = evaluate(key, (1, 1, 1), forces=True)
        for rep in ((2, 1, 1), (2, 2, 1)):
            try:
                e, f, cfg, _, ncopy = evaluate(key, rep, forces=True)
            except Exception as err:                                         # noqa: BLE001  (reported, not hidden)
                print(f"   {key} {rep}: REFUSED {type(err).__name__}: {str(err)[:150]}")
                continue
            df = np.abs(f - np.tile(f1, (ncopy, 1))).max()
            print(f"   {key[0]:8s}{key[1]:12s} {rep[0]}x{rep[1]}: configurations {int(cfg1.shape[0])} -> {int(cfg.shape[0])}  "
                  f"E/cell - E(1x1) {1e3 * (e / ncopy - e1):+11.4f} meV  max|dF| {df:.2e} eV/A")

if "table2" in parts:
    energies, bonds = {}, {}
    for key, atoms in refs.items():
        relaxed, energies[key] = relax(calc, atoms)
        if key[1] == "non-wetting":
            bonds[key[0]] = au_o_bond(relaxed)
    wet = {d: 1e3 * (energies[(d, "wetting")] - energies[(d, "non-wetting")]) for d in ("doped", "undoped")}
    print(f"Table II ({mod.redox.state_polarizability_stiffness}): wet_u {wet['undoped']:+.1f}  wet_d {wet['doped']:+.1f}  "
          f"T5 {wet['doped'] - wet['undoped']:+.1f} meV   (DFT +928.9 / -2.7 / -931.6; 4G +975.4 / -41.0 / -1016.4)")
    print(f"   relaxed non-wetting Au-O: doped {bonds['doped']:.3f}  undoped {bonds['undoped']:.3f} A  (DFT 2.332 / 2.190)")
    print("   full precision: " + repr(wet))
