"""Why the SPEC 21 closure exceeds its cap on a structure (2026-10-18; diagnostic, reported).

Usage: python probe_closure_size.py MODEL.pt DOPING GEOMETRY [CAP]
On the T5 reference (eval_aumgo.references) the closure is rerun with the cap lifted; the configurations are
classified by which units differ from the lowest one (element and band), by excitation order, and by importance."""
import collections, os, sys, warnings
import torch
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", ".."))); sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)
import trace_redox.diabatic as D  # noqa: E402
import trace_redox.module as TM  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from eval_aumgo import references  # noqa: E402

model, doping, geometry = sys.argv[1:4]
cap = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
store = {}
orig = D.coupled_closure
def closure(configs, e_hat, states, b, W, max_charge, T, a_off, kT, *, cap, **kw):
    out = orig(configs, e_hat, states, b, W, max_charge, T, a_off, kT, cap=CAP, **kw)
    store.update(out=out, e_hat=e_hat.detach(), states=states, b=b.detach(), W=W.detach(), T=T.detach(), kT=kT,
                 pairs=kw.get("pairs"))
    return out
CAP = cap
D.coupled_closure = closure
calc = TransformersACECalculator(model_path=model, device="cpu"); torch.set_num_threads(2)
a = references()[(doping, geometry)].copy(); a.info = {"charge": 0.0}; a.calc = calc
E = a.get_potential_energy()
cfg = store["out"]; Z = list(a.numbers) + [0]                                    # the band unit last (Z = 0)
En = D.configuration_energies(cfg, store["e_hat"], store["states"], store["b"], store["W"], pairs=store["pairs"])
low = int(torch.argmin(En)); ref = cfg[low]
print(f"{doping} {geometry}: E = {E:.4f} eV; closure {cfg.shape[0]} configurations (cap lifted to {CAP}); "
      f"units {cfg.shape[1]} (band unit last: {len(Z) == cfg.shape[1]})")
order = collections.Counter(); kinds = collections.Counter()
for k in range(cfg.shape[0]):
    d = (cfg[k] - ref).nonzero().flatten().tolist()
    order[len(d)] += 1
    kinds[tuple(sorted((("band" if u == len(Z) - 1 else {8: "O", 12: "Mg", 13: "Al", 79: "Au"}.get(int(Z[u]), str(Z[u]))), int(cfg[k][u] - ref[u])) for u in d))] += 1
print("units changed vs the lowest configuration:", dict(sorted(order.items())))
for kind, c in kinds.most_common(12):
    print(f"  {c:6d}  {kind}")
x = En - (-store["kT"] * torch.logsumexp(-En / store["kT"], 0))
print(f"excitation energies x: min {float(x.min()):.3f}, median {float(x.median()):.3f}, max {float(x.max()):.3f} eV")
Tm = store["T"]; print(f"|T| > 0.5 eV pairs: {int((Tm.abs() > 0.5).sum()) // 2}; max |T| {float(Tm.abs().max()):.3f} eV; "
      f"largest by unit pair: {[(Z[i] if i < len(Z) else 'band', Z[j] if j < len(Z) else 'band', round(float(Tm[i, j]), 3)) for i, j in (Tm.abs() == Tm.abs().max()).nonzero().tolist()[:2]]}")

if "--relax" in sys.argv:
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS
    a.set_constraint(FixAtoms(indices=list(range(108))))
    step = {"n": 0}
    def report():
        cfg = store["out"]
        En = D.configuration_energies(cfg, store["e_hat"], store["states"], store["b"], store["W"], pairs=store["pairs"])
        ref = cfg[int(torch.argmin(En))]
        kinds = collections.Counter()
        for k in range(cfg.shape[0]):
            d = (cfg[k] - ref).nonzero().flatten().tolist()
            kinds[tuple(sorted(("band" if u == len(Z) - 1 else {8: "O", 12: "Mg", 13: "Al", 79: "Au"}.get(int(Z[u]), str(Z[u]))) for u in d))] += 1
        au = a.positions[[108, 109]]
        print(f"  step {step['n']:3d}: closure {cfg.shape[0]:6d}; Au-Au {float(((au[0]-au[1])**2).sum()**0.5):.3f} A; "
              f"max|T| {float(store['T'].abs().max()):.3f}; top kinds {kinds.most_common(3)}", flush=True)
        step["n"] += 1
    opt = BFGS(a, logfile=None); opt.attach(report, interval=1); opt.run(fmax=0.005, steps=60)
