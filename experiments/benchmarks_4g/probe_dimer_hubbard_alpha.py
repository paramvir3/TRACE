"""Frozen check of the two-site Hubbard prior through an observable (2026-11-01): the gas-phase Au2 polarisability.

The prior would set the model's own two-site sector to the measured minimal-model values, Delta_ion(R) -> U(R) and
|H_cov,ion|(R) -> sqrt 2 t(R) (spin_pairing off; probe_dimer_hubbard.py).  The model's alpha (finite field, its own
screened configuration dipoles M(n) = sum q r + D^T y*(n) and its bound sector; SPEC 30) is then compared with the
first-principles alpha_par, alpha_perp (PBE0 / def2-TZVPPD+2d, $TRACE_FP_CACHE/dimer_alpha_Au2_*).  On frozen weights:
(i) as trained; (ii) Delta -> U only; (iii) coupling -> sqrt 2 t only; (iv) both.  The injection adds a constant
delta to the energies of (+1,-1) and (-1,+1) (through SPEC 28's diagonal slot) and scales the dimer's coupling by a
constant, both fixed at zero field, so the field enters only as the model applies it.
Usage (repo root): python experiments/benchmarks_4g/probe_dimer_hubbard_alpha.py MODEL.pt [...]"""
import sys, math, dataclasses, warnings
sys.path.insert(0, ".")
warnings.simplefilter("ignore")
import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
from ase import Atoms  # noqa: E402
import trace_redox.module as M  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
K_E = 14.399645
UT = {2.30: (0.900, 1.409), 2.47: (0.995, 1.249), 2.60: (1.063, 1.146), 2.80: (1.164, 1.009)}     # TDA-PBE U, t
ALPHA = {2.30: (14.012, 8.132), 2.47: (15.620, 8.312), 2.60: (17.033, 8.451), 2.80: (19.487, 8.669)}  # PBE0 par, perp
INJ = {}
_solve = M.RedoxElectrostatics._diabatic_solution


def solve(self, e_hat, states, b, W, L, K, Q, T=None, **kw):
    if INJ:
        d, s = INJ["delta"], INJ["scale"]
        base = kw.get("diag_correction")

        def dc(cfgs):
            out = base(cfgs) if base is not None else torch.zeros(cfgs.shape[0], dtype=W.dtype)
            c = cfgs[:, :2]
            ion = ((c[:, 0] == 1) & (c[:, 1] == -1)) | ((c[:, 0] == -1) & (c[:, 1] == 1))
            return out + d * ion.to(W.dtype)
        kw["diag_correction"] = dc
        T = None if T is None else T * s
    return _solve(self, e_hat, states, b, W, L, K, Q, T=T, **kw)


M.RedoxElectrostatics._diabatic_solution = solve


def energy(calc, atoms, field):
    a = atoms.copy(); a.info = {"charge": 0.0, "external_field": list(field)}; a.calc = calc; calc.reset()
    return a.get_potential_energy()


def sector(mod):
    last = mod._ensemble_last; H = last["H"].detach().numpy(); cfg = last["configs"].tolist()
    rows = {tuple(x[:2]): i for i, x in enumerate(cfg)}
    c0, i = rows[(0, 0)], rows[(1, -1)]
    w, V = np.linalg.eigh(H); g = V[:, 0] ** 2
    return H[i, i] - H[c0, c0], abs(H[c0, i]), float(g[i] + g[rows[(-1, 1)]])


def alpha(calc, atoms, axis, h=0.01):
    e = np.zeros(3); e[axis] = h
    return -(energy(calc, atoms, e) - 2 * energy(calc, atoms, 0 * e) + energy(calc, atoms, -e)) / h ** 2 * K_E


for model in sys.argv[1:]:
    calc = TransformersACECalculator(model_path=model, device="cpu")
    mod = calc.model.electrostatics
    mod.redox = dataclasses.replace(mod.redox, downfold_transfer=False)
    import os
    if os.environ.get("SP"):            # SPEC 29 on (frozen weights); the target on the H element stays sqrt 2 t
        mod.redox = dataclasses.replace(mod.redox, spin_pairing=True)
    print(f"=== {model.split('/')[-2][-34:]} (spin_pairing {mod.redox.spin_pairing})", flush=True)
    for R, (U, t) in UT.items():
        dimer = Atoms("Au2", positions=[[0, 0, 0], [0, 0, R]])
        INJ.clear(); energy(calc, dimer, np.zeros(3)); D0, C0, p0 = sector(mod)
        target = math.sqrt(2.0) * t
        line = []
        for label, delta, scale in (("as trained", 0.0, 1.0), ("Delta -> U", U - D0, 1.0),
                                    ("coupling -> sqrt2 t", 0.0, target / C0), ("both", U - D0, target / C0)):
            INJ.clear()
            if label != "as trained":
                INJ.update(delta=delta, scale=scale)
            energy(calc, dimer, np.zeros(3)); D, C, p = sector(mod)
            ap, an = alpha(calc, dimer, 2), alpha(calc, dimer, 0)
            line.append(f"{label}: Delta {D:.3f}, |H| {C:.3f}, p_ion {p:.3f}, alpha par {ap:.2f} / perp {an:.2f} (aniso {ap - an:.2f})")
        INJ.clear()
        print(f"  R {R:.2f} (PBE0 par {ALPHA[R][0]:.2f} / perp {ALPHA[R][1]:.2f}, aniso {ALPHA[R][0] - ALPHA[R][1]:.2f} A^3; "
              f"target Delta {U:.3f}, |H| {target:.3f})", flush=True)
        for l in line:
            print("     " + l, flush=True)
