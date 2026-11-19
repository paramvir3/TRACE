# Frozen test probe (2026-10-09): pre-registered test P3, formal oxidation states with closed-shell capacity.
# Usage (repo root): python experiments/benchmarks_4g/probe_formal_states.py RUN/model.pt [N_STRUCTURES]
"""Where does Al's donor electron go if every atom carries its integer (formal) oxidation-state ladder
with the closed-shell capacity limit?  The redoxSQE rule (Dapp & Mueser, arXiv:1308.3424 / 1308.3573):
charges are referenced to formal oxidation states, and a closed-shell ion cannot accept (e.g. Cl-).

Integer-configuration energy, units = atoms, no intrinsic bound-sector charges (b = 0):

    E(n) = sum_u [E_Z(n_u) - J_uu n_u^2 / 2] + 1/2 n^T W n,

E_Z(n) the PPLB integer ladder of the isolated atom (sum of successive IEs for n > 0; -EA_1 for
n = -1; O: -EA_1 + 8.75 eV for n = -2, the O- + e -> O2- cost, +844 kJ/mol, chemguide; Born-Haber
spread 780-844 kJ/mol).  Capacity: O >= -2; Mg, Al, Au >= -1.  The -J_uu n^2/2 makes an isolated
ion's energy exactly E_Z(n) (the module's self-Coulomb-removed convention).

Two screening brackets for W: 'trained' (the model's W = J - A K^-1 A^T, metal-like) and 'bare'
(W = J, no electronic screening).  Experiment (eps_inf ~ 3) lies between.

Doped configurations, all on the ionic base Mg +2, O -2, Al +3 (net +3; three electrons to place):
    stay   : the electrons stay on Al (Al +2 x 3)
    Au2+Mg : Au -1, Au -1, and +1 on the Mg nearest the first Al
    3Mg    : +1 on the Mg nearest each Al
    Au2+Al : Au -1, Au -1, and one Al +2
Also: the ionic base against all-neutral (is the formal picture the ground state at all?).

Pre-registered criterion (P3): E(Au2+Mg) - E(stay) < -0.5 eV in BOTH brackets for >= 5 of 6 structures.
"""
import sys
import warnings

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/benchmarks_4g")
warnings.simplefilter("ignore")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)
from ase.io import read  # noqa: E402

from flashace.calculator import TransformersACECalculator  # noqa: E402
from probe_one_curve import capture  # noqa: E402
from trace_redox.bound import assemble_neutral, eliminate_bound_sector  # noqa: E402
from trace_redox.module import ATOMIC_ELECTRON_AFFINITY_EV, ATOMIC_SUCCESSIVE_IONIZATION_EV  # noqa: E402

AU, AL, MG, O = 79, 13, 12, 8
O_SECOND_STEP_EV = 844.0 / 96.4853           # O- + e -> O2-, +844 kJ/mol


def ladder(Z, n):
    """PPLB integer energy of the isolated atom, relative to neutral; inf outside the capacity."""
    if n > 0:
        series = ATOMIC_SUCCESSIVE_IONIZATION_EV[Z]
        return float(sum(series[:n]))
    if n == 0:
        return 0.0
    ea1 = float(ATOMIC_ELECTRON_AFFINITY_EV[Z])
    if n == -1:
        return -ea1
    if n == -2 and Z == O:
        return -ea1 + O_SECOND_STEP_EV
    return float("inf")


def energy(n, z, J, W):
    e = sum(ladder(int(Z), int(k)) - 0.5 * float(J[u, u]) * k * k for u, (Z, k) in enumerate(zip(z.tolist(), n.tolist())))
    nv = torch.as_tensor(n, dtype=W.dtype)
    return e + 0.5 * float(nv @ (W @ nv))


def main():
    ck = sys.argv[1]
    n_struct = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    test = read("experiments/benchmarks_4g/data/AuMgO_test.xyz", ":")
    doped = [a for a in test if a.info["system"].startswith("Al")][:n_struct]
    calc = TransformersACECalculator(model_path=ck, device="cpu")
    passes = 0
    for s, atoms in enumerate(doped):
        z = torch.tensor(atoms.numbers)
        mod, (args, kw), res = capture(calc, atoms)
        h, pos, ei, vec, L = [x.detach() for x in args[:5]]
        keys = {k: kw[k] for k in ("pbc", "cell", "edge_shift") if k in kw}
        asm = assemble_neutral(h, pos, ei, vec, L, mod.heads, mod.config, species=z, **keys)
        sigma_f = (mod.width(z) if mod.width is not None else asm.params.sigma).detach()
        elim = eliminate_bound_sector(asm, sigma_f, method=mod.redox.bound_elimination)
        J, W_tr = elim.J.detach(), elim.W.detach()
        al = np.where(atoms.numbers == AL)[0]
        au = np.where(atoms.numbers == AU)[0]
        mg = np.where(atoms.numbers == MG)[0]
        near_mg = [int(mg[np.argmin(atoms.get_distances(int(a), mg, mic=True))]) for a in al]
        base = torch.zeros(len(atoms))
        base[atoms.numbers == MG], base[atoms.numbers == O], base[atoms.numbers == AL] = 2.0, -2.0, 3.0
        configs = {}
        c = base.clone(); c[al] = 2.0; configs["stay"] = c
        c = base.clone(); c[au] = -1.0; c[near_mg[0]] = 1.0; configs["Au2+Mg"] = c
        c = base.clone(); c[near_mg] = 1.0; configs["3Mg"] = c
        c = base.clone(); c[au] = -1.0; c[al[0]] = 2.0; configs["Au2+Al"] = c
        line, ok = [], True
        for name, W in (("trained", W_tr), ("bare", J)):
            E = {k: energy(v, z, J, W) for k, v in configs.items()}
            assert all(abs(float(v.sum())) < 1e-12 for v in configs.values())
            neutral_vs_ionic = E["stay"] - 0.0                       # all-neutral has E = 0
            d = {k: E[k] - E["stay"] for k in ("Au2+Mg", "3Mg", "Au2+Al")}
            ok &= d["Au2+Mg"] < -0.5
            line.append(f"{name}: ionic(stay) - neutral {neutral_vs_ionic:+9.2f} eV; relative to stay: "
                        + "  ".join(f"{k} {v:+7.3f}" for k, v in d.items()))
        passes += ok
        print(f"structure {s}:  " + "\n              ".join(line))
    print(f"P3 (Au2+Mg below stay by > 0.5 eV in both brackets): {passes} of {len(doped)} structures -> "
          f"{'PASS' if passes >= 5 else 'FAIL'}")


if __name__ == "__main__":
    main()
