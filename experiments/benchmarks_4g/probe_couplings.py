# Diagnostic probe (2026-10-10; reported, not a criterion): is the model's Au2 covalent?
# Usage (repo root): python experiments/benchmarks_4g/probe_couplings.py RUN/model.pt [RUN2/model.pt ...]
"""Au state populations of a FORMAL checkpoint on the first 6 undoped and 6 doped test structures.

- p(Au+), p(Au0), p(Au-): RedoxResult.populations of the two Au atoms, averaged over atoms and
  structures (the canonical ensemble at the checkpoint's kT, as deployed);
- <|n_Au1 - n_Au2|>: the charge separation inside the dimer (0 for a covalent or a symmetric
  Au2-; 2 for a pure Au+Au- pair), from the same ensemble when the module returns no per-configuration
  data: |<n_1> - <n_2>| is a lower bound and is what is printed;
- tau(Au1, Au2): the learned coupling between the two Au atoms (couplings=True only).
"""
import sys
import warnings

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
from ase.io import read  # noqa: E402

from flashace.calculator import TransformersACECalculator  # noqa: E402


def probe(checkpoint, structures):
    calc = TransformersACECalculator(model_path=checkpoint, device="cpu")
    torch.set_num_threads(2)
    mod = calc.model.electrostatics
    assert mod.redox.anchor == "formal", "a formal checkpoint is needed"
    results, couplings = [], []
    hook = mod.register_forward_hook(lambda m, i, o: results.append(o[1]))
    chook = (mod.coupling.register_forward_hook(lambda m, i, o: couplings.append(o.detach()))
             if mod.coupling is not None else None)
    rows = []
    try:
        for atoms in structures:
            a = atoms.copy()
            a.info = dict(atoms.info)
            a.calc = calc
            results.clear()
            couplings.clear()
            a.get_potential_energy()
            r = results[-1]
            z = torch.tensor(a.numbers)
            au = torch.nonzero(z == 79).flatten()
            states = r.states
            pops = r.populations.detach()[au]                                  # [2, n_states]
            col = {int(s): k for k, s in enumerate(states.tolist())}
            charges = r.free_charge.detach()[au]
            tau = float(couplings[-1][au[0], au[1]]) if couplings else float("nan")
            rows.append(dict(p_plus=float(pops[:, col[1]].mean()), p_zero=float(pops[:, col[0]].mean()),
                             p_minus=float(pops[:, col[-1]].mean()),
                             separation=float((charges[0] - charges[1]).abs()), tau=tau))
    finally:
        hook.remove()
        if chook is not None:
            chook.remove()
    return rows


def main():
    test = read("experiments/benchmarks_4g/data/AuMgO_test.xyz", ":")
    groups = (("undoped", [a for a in test if a.info["system"].startswith("Au2Mg54")][:6]),
              ("doped", [a for a in test if a.info["system"].startswith("Al")][:6]))
    for checkpoint in sys.argv[1:]:
        epoch = torch.load(checkpoint, map_location="cpu").get("epoch")
        print(f"=== {checkpoint} (epoch {epoch})")
        for label, group in groups:
            rows = probe(checkpoint, group)
            mean = {k: sum(row[k] for row in rows) / len(rows) for k in rows[0]}
            print(f"  {label:8s}: p(Au+) {mean['p_plus']:.3f}  p(Au0) {mean['p_zero']:.3f}  p(Au-) {mean['p_minus']:.3f}  "
                  f"|<n_1> - <n_2>| {mean['separation']:.3f}  tau(Au1, Au2) {mean['tau']:+.4f} eV   "
                  f"(per structure p(Au+): {[round(row['p_plus'], 3) for row in rows]})")


if __name__ == "__main__":
    main()
