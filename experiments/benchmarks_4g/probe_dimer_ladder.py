"""Configuration energies of an isolated pair, any elements (Proposition 8 across benchmarks; frozen weights; 2026-10-19).

Usage (repo root): python experiments/benchmarks_4g/probe_dimer_ladder.py A B D1,D2,... CKPT [...]

The module's own inputs to the diabatic solve are captured (e_hat, states, b, W, the pair terms, SPEC 28's
correction, the couplings T), and chosen configurations are evaluated exactly with trace_redox's
configuration_energies, so nothing depends on which configurations the selection keeps.  Printed per distance:
E(A+ B-) - E(A0 B0), E(A- B+) - E(A0 B0) (eV) and |tau_AB|, for the shifts as trained and zeroed (chi_bound =
1e-9: |delta| <= 1e-9 eV), each as trained, with the overlap stack (SPEC 22, 23, 25, 26), and with the stack and
SPEC 28 (all reliable ratios).  A configuration SPEC 28 refuses (K(n) not positive definite) is printed as such.
Classical point limit for comparison: U_cl = IE_A - EA_B - K/R - (a_A + a_B) K / (2 R^4) (theory note 4c)."""
import dataclasses
import sys
import warnings

import torch

warnings.simplefilter("ignore")
sys.path.insert(0, ".")
torch.set_default_dtype(torch.float64)
from ase import Atoms  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
import trace_redox.module as M  # noqa: E402
from trace_redox.diabatic import configuration_energies  # noqa: E402

STACK = dict(bond_partition=True, carrier_penetration=True, carrier_overlap="gordon_kim", exclusion=False,
             carrier_induction=True)
STORE = {}
_solution = M.RedoxElectrostatics._diabatic_solution


def _capture(self, e_hat, states, b, W, L_units, K_units, Q, T=None, units=None, frames=None, pairs=None,
             diag_correction=None, electrons=None):
    STORE.update(e_hat=e_hat, states=states, b=b, W=W, T=T, pairs=pairs, diag=diag_correction)
    return _solution(self, e_hat, states, b, W, L_units, K_units, Q, T=T, units=units, frames=frames, pairs=pairs,
                     diag_correction=diag_correction, electrons=electrons)


def ladder(calc, a, b, d):
    atoms = Atoms(a + b, positions=[[0, 0, 0], [0, 0, d]])
    atoms.info = {"charge": 0.0}
    atoms.calc = calc
    calc.reset()
    STORE.clear()
    atoms.get_potential_energy()
    U = int(STORE["e_hat"].shape[0])
    configs = torch.zeros(3, U, dtype=torch.long)
    configs[1, :2] = torch.tensor([1, -1])
    configs[2, :2] = torch.tensor([-1, 1])
    with torch.no_grad():
        E = configuration_energies(configs, STORE["e_hat"], STORE["states"], STORE["b"], STORE["W"],
                                   pairs=STORE["pairs"])
        note = ""
        if STORE["diag"] is not None:
            try:
                E = E + STORE["diag"](configs)
            except ValueError as err:
                note = " (SPEC 28 refused: " + str(err).split("(atoms")[0].split("gives")[0].strip() + ")"
                return float("nan"), float("nan"), float("nan"), note
        tau = float(STORE["T"][0, 1].abs()) if STORE["T"] is not None else float("nan")
    return float(E[1] - E[0]), float(E[2] - E[0]), tau, note


def main():
    a, b, ds = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3].split(",")]
    M.RedoxElectrostatics._diabatic_solution = _capture
    for ck in sys.argv[4:]:
        calc = TransformersACECalculator(model_path=ck, device="cpu")
        torch.set_num_threads(2)
        mod = calc.model.electrostatics
        base = mod.redox
        print(f"=== {ck.split('/')[-2]}  {a}{b}: E(+-) - E(00), E(-+) - E(00) eV, |tau| eV")
        for label, extra in (("as trained", {}), ("overlap stack", STACK),
                             ("stack + SPEC 28", dict(STACK, state_polarizability=True))):
            for shifts in ("learned", "zero"):
                kw = dict(extra)
                if shifts == "zero":
                    kw["chi_bound"] = 1e-9
                try:
                    mod.redox = dataclasses.replace(base, **kw)
                    cells = []
                    for d in ds:
                        up, um, tau, note = ladder(calc, a, b, d)
                        cells.append(f"{d:.2f}: {up:+.3f} {um:+.3f} tau {tau:.3f}{note}")
                    print(f"  {label:15s} shifts {shifts:7s} | " + " | ".join(cells), flush=True)
                except Exception as err:                                   # noqa: BLE001
                    print(f"  {label:15s} shifts {shifts:7s} | ERROR {type(err).__name__}: {str(err)[:140]}")
        mod.redox = base


if __name__ == "__main__":
    main()
