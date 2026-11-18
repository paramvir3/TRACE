# Diagnostic probe (2026-10-18; reported).  Usage (repo root): python experiments/benchmarks_4g/probe_dimer_ct.py CKPT SYMBOL BOND_A [...]
"""Why does an isolated covalent metal dimer over-polarise along its bond?  The charge-transfer configurations of
the formal model, decomposed.

For M2 at bond d (open boundaries, Q = 0): the diabatic inputs (ladder e_hat, potential b, screened operator W,
coupling T) captured inside RedoxElectrostatics._diabatic_solution, and the configuration energies

    E(n) = sum_u e_u(n_u) + b.n + 1/2 n.W.n,      Delta_ion = E(+1, -1) - E(0, 0)

split into the ladder part [e_1(+1) + e_2(-1) - e_1(0) - e_2(0)] (= IE - EA + learned shifts - the S0 self
terms), the on-site W part 1/2 (W_11 + W_22) and the inter-site part -W_12, against the bare K_E / d.  The coupling
tau_12, the two-state charge-transfer polarisability of the 3-configuration model in a field F along the bond
(E(+1,-1) -> E - F d, E(-1,+1) -> E + F d; alpha = -d^2 F_min / dF^2 x K_E, A^3), and the model's total alpha_par
and alpha_perp from finite fields (as probe_au_polarizability.py).
"""
import sys
import warnings

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)
from ase import Atoms  # noqa: E402

import trace_redox.module as M  # noqa: E402
from trace_redox.diabatic import configuration_energies  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402

K_E = 14.399645
CAP = {}
_solve = M.RedoxElectrostatics._diabatic_solution


def capture(self, e_hat, states, b, W, L_units, K_units, Q, T=None, **kwargs):
    """Every keyword of _diabatic_solution passes through (SPEC 28 added diag_correction, SPEC 29 electrons; a fixed
    signature crashed this probe on every model from step 13 on, 2026-10-19)."""
    pairs = kwargs.get("pairs")
    CAP.update(e_hat=e_hat.detach().clone(), states=states.detach().clone(), b=b.detach().clone(),
               W=W.detach().clone(), T=None if T is None else T.detach().clone(), K=K_units.clone(),
               pairs=None if pairs is None else tuple(x.detach() for x in pairs),
               diag=kwargs.get("diag_correction"))
    return _solve(self, e_hat, states, b, W, L_units, K_units, Q, T=T, **kwargs)


M.RedoxElectrostatics._diabatic_solution = capture


def energy(calc, atoms, field):
    a = atoms.copy()
    a.info = {"charge": 0.0, "external_field": list(field)}
    a.calc = calc
    calc.reset()
    return a.get_potential_energy()


def alpha(calc, atoms, axis, h=0.01):
    e = np.zeros(3)
    e[axis] = h
    return -(energy(calc, atoms, e) - 2 * energy(calc, atoms, 0 * e) + energy(calc, atoms, -e)) / h ** 2 * K_E


def lowest(Ecov, Eion, Erev, t, F, d):
    H = np.array([[Ecov, -t, t], [-t, Eion - F * d, 0.0], [t, 0.0, Erev + F * d]])
    return np.linalg.eigvalsh(H)[0]


def main():
    ck = sys.argv[1]
    calc = TransformersACECalculator(model_path=ck, device="cpu")
    torch.set_num_threads(3)
    print(f"=== {ck.split('/')[-2]}")
    args = sys.argv[2:]
    for sym, d in zip(args[0::2], map(float, args[1::2])):
        dimer = Atoms(sym + "2", positions=[[0, 0, 0], [0, 0, d]])
        energy(calc, dimer, np.zeros(3))
        e, st, b, W, T = CAP["e_hat"], CAP["states"], CAP["b"], CAP["W"], CAP["T"]
        first = int(round(float(st[0])))
        col = lambda n: n - first
        E = {}
        for n in ((0, 0), (1, -1), (-1, 1)):
            v = torch.tensor(n, dtype=torch.float64)
            E[n] = float(e[0, col(n[0])] + e[1, col(n[1])] + b @ v + 0.5 * v @ W @ v)
        # the complete configuration energies (pair terms of SPEC 18.5 / 23 / 26 and SPEC 28's correction included)
        U = int(e.shape[0])
        cfgs = torch.zeros(3, U, dtype=torch.long)
        cfgs[1, :2], cfgs[2, :2] = torch.tensor([1, -1]), torch.tensor([-1, 1])
        with torch.no_grad():
            full = configuration_energies(cfgs, e, st, b, W, pairs=CAP["pairs"])
            if CAP["diag"] is not None:
                full = full + CAP["diag"](cfgs)
        full_ion, full_rev = float(full[1] - full[0]), float(full[2] - full[0])
        ladder = float(e[0, col(1)] + e[1, col(-1)] - e[0, col(0)] - e[1, col(0)])
        onsite = 0.5 * float(W[0, 0] + W[1, 1])
        inter = -float(W[0, 1])
        field = float(b[0] - b[1])
        t = 0.0 if T is None else float(T[0, 1])
        d_ion, d_rev = full_ion, full_rev
        partial = E[(1, -1)] - E[(0, 0)]
        h = 0.01
        f = [lowest(0.0, d_ion, d_rev, t, F, d) for F in (-h, 0.0, h)]
        a_ct = -(f[0] - 2 * f[1] + f[2]) / h ** 2 * K_E
        ap, an = alpha(calc, dimer, 2), alpha(calc, dimer, 0)
        print(f"  {sym}2 d = {d:.3f} A")
        print(f"    Delta_ion = E(+1,-1) - E(0,0) = {d_ion:+.3f} eV  (reverse {d_rev:+.3f})  =  ladder {ladder:+.3f}"
              f"  + on-site 1/2(W11+W22) {onsite:+.3f}  + inter-site -W12 {inter:+.3f}  + b {field:+.3f}"
              f"   [bare -K/d = {-K_E / d:+.3f}]")
        print(f"    (Delta_ion is the complete configuration energy; ladder + W + b alone gives {partial:+.3f}, the "
              f"difference {d_ion - partial:+.3f} eV is the pair terms and SPEC 28)")
        print(f"    tau_12 = {t:+.3f} eV;  3-configuration CT polarisability {a_ct:7.2f} A^3;  model alpha_par {ap:7.2f},"
              f" alpha_perp {an:7.2f} A^3 (difference {ap - an:7.2f})")


if __name__ == "__main__":
    main()
