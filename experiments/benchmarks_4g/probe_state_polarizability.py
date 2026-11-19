"""Counterfactual (frozen weights): state-dependent polarisabilities in the configuration energies (2026-10-18).

Usage: python probe_state_polarizability.py MODEL.pt POLARIZABILITIES.json

The bound sector gives atom j one dipole stiffness P_jj (its isolated polarisability alpha_j = ell^2 P_jj^-1) in every
charge state.  With the state's own polarisability alpha_j(n) = rho_j(n) alpha_j(f), rho = alpha_free(Z, n) /
alpha_free(Z, f) from first principles (tools/electrostatics/ion_polarizabilities.py), K(n) = K + sum_j P_j^T
Delta_j P_j with Delta_j = P_jj (1 / rho_j - 1), and by Woodbury, exactly, per configuration:

    dE(n) = 1/2 y_S(n)^T Delta_S (I + C_SS Delta_S)^-1 y_S(n),   y(n) = y0 - K^-1 A^T q(n),   C = K^-1 on the dipoles,

S the atoms away from their frame (the band unit's carrier keeps its stiffness: stated).  Single isolated atom:
dE = -1/2 (alpha(n) - alpha(f)) E^2.  The probe adds dE to every selected configuration's energy at the four T5
reference geometries (unrelaxed; relaxation moved T5 by < 40 meV) and reports T5 with and without it, for two
variants: every state, and Mg held at its frame (the free-ion Mg+/Mg2+ ratio ignores the lattice confinement and
overlaps the band unit).  Free-ion ratios are an upper bound for confined ions (Fowler-Madden).  States whose
finite-field alpha changes by more than 5 % between F = 2e-3 and 1e-3 au (diffuse anions such as Na-, nearly unbound
in PBE0) are left at the frame value and listed.
"""
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

import trace_redox.module as TM  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from trace_redox.constants import formal_frame  # noqa: E402
from eval_aumgo import references  # noqa: E402

SYMBOL = {8: "O", 11: "Na", 12: "Mg", 13: "Al", 17: "Cl", 47: "Ag", 79: "Au"}
STORE = {"on": False, "skip": set(), "report": []}


def ratios(table, frames_of):
    out = {}
    reliable = lambda v: float(v["richardson_step_bohr3"]) <= 0.05 * abs(float(v["alpha_bohr3"]))
    for key, v in table.items():
        Z, q = int(v["Z"]), int(v["q"])
        if not reliable(v):
            print(f"  {key}: alpha not converged in the field (step change {v['richardson_step_bohr3']:.1f} of "
                  f"{v['alpha_bohr3']:.1f} bohr^3): left at the frame value")
            continue
        f = frames_of.get(Z)
        ref = table.get(f"{SYMBOL.get(Z, '?')}{f:+d}")
        if f is not None and ref is not None and reliable(ref):
            out[(Z, q)] = float(v["alpha_bohr3"]) / float(ref["alpha_bohr3"])
    return out


def delta_energies(configs):
    elim, Z = STORE["elim"], STORE["Z"]
    n, nb = elim.assembly.n_atoms, elim.n_bonds
    dtype = elim.y0.dtype
    K = elim.operator.matrix()
    D = torch.tensor([nb + 3 * j + c for j in range(n) for c in range(3)])
    eye = torch.zeros(K.shape[0], D.numel(), dtype=dtype)
    eye[D, torch.arange(D.numel())] = 1.0
    C = elim.operator.solve(eye)[D]                                           # [3n, 3n]
    M = elim.operator.solve(elim.A.transpose(0, 1))                           # [m, n]
    aug = getattr(STORE["module"], "_band_last", None)
    omega = None if aug is None else aug["omega"]
    frames = formal_frame(Z).tolist()
    out = torch.zeros(configs.shape[0], dtype=dtype)
    rows = []
    for k in range(configs.shape[0]):
        cfg = configs[k].to(dtype)
        q = cfg if omega is None else omega.transpose(0, 1) @ cfg
        y = elim.y0 - M @ q
        S = [j for j in range(n) if int(round(float(cfg[j]))) != frames[j]]
        rho = [STORE["rho"].get((int(Z[j]), int(round(float(cfg[j])))), 1.0) for j in S]
        S = [j for j, r in zip(S, rho) if int(Z[j]) not in STORE["skip"] and abs(r - 1.0) > 0]
        rho = [STORE["rho"].get((int(Z[j]), int(round(float(cfg[j])))), 1.0) for j in S]
        if not S:
            continue
        idx = torch.tensor([3 * j + c for j in S for c in range(3)])
        Delta = torch.block_diag(*[K[nb + 3 * j:nb + 3 * j + 3, nb + 3 * j:nb + 3 * j + 3] * (1.0 / r - 1.0)
                                   for j, r in zip(S, rho)])
        yS = y[D][idx]
        CSS = C[idx][:, idx]
        dE = 0.5 * yS @ (Delta @ torch.linalg.solve(torch.eye(idx.numel(), dtype=dtype) + CSS @ Delta, yS))
        out[k] = dE
        rows.append((k, [(int(Z[j]), int(round(float(cfg[j])))) for j in S], float(dE)))
    STORE["report"].append(rows)
    return out


def main():
    model, table_path = sys.argv[1], sys.argv[2]
    table = json.load(open(table_path))
    calc = TransformersACECalculator(model_path=model, device="cpu")
    torch.set_num_threads(2)
    mod = calc.model.electrostatics
    STORE["module"] = mod
    frames_of = {}
    for Z in SYMBOL:
        frames_of[Z] = int(formal_frame(torch.tensor([Z]))[0])
    STORE["rho"] = ratios(table, frames_of)
    print("rho = alpha(Z, n) / alpha(Z, frame):", {f"{SYMBOL[z]}{q:+d}": round(r, 3) for (z, q), r in sorted(STORE["rho"].items())})

    orig_elim, orig_ce = TM.eliminate_bound_sector, TM.configuration_energies

    def elim_wrap(*a, **k):
        e = orig_elim(*a, **k)
        STORE["elim"] = e
        return e

    def ce_wrap(configs, e_hat, states, b, W, pairs=None):
        E = orig_ce(configs, e_hat, states, b, W, pairs=pairs)
        if STORE["on"]:
            E = E + delta_energies(configs.detach())
        STORE["configs"], STORE["E"] = configs.detach(), E.detach()
        return E

    TM.eliminate_bound_sector, TM.configuration_energies = elim_wrap, ce_wrap
    refs = references()
    variants = [("model (state-independent alpha)", False, set()), ("all states", True, set()),
                ("Mg at its frame alpha", True, {12})]
    results = {}
    for label, on, skip in variants:
        STORE["on"], STORE["skip"] = on, skip
        energies = {}
        for key, atoms in refs.items():
            a = atoms.copy()
            a.info = dict(atoms.info)
            STORE["Z"] = torch.as_tensor(a.numbers)
            STORE["report"] = []
            a.calc = calc
            calc.reset()
            energies[key] = a.get_potential_energy()
            if on:
                E, cfgs = STORE["E"], STORE["configs"]
                low = int(torch.argmin(E))
                rows = STORE["report"][-1] if STORE["report"] else []
                big = sorted(rows, key=lambda r: abs(r[2]), reverse=True)[:3]
                au = [int(v) for v in cfgs[low][[108, 109]].tolist()]
                print(f"  {label:32s} {key[0]:8s} {key[1]:12s}: {len(E)} configurations; lowest has Au {au}; "
                      f"largest |dE|: {[(r[1], round(r[2], 3)) for r in big]}")
        t = {d: 1e3 * (energies[(d, "wetting")] - energies[(d, "non-wetting")]) for d in ("doped", "undoped")}
        results[label] = (t, energies)
        print(f"{label:34s}: wetting doped {t['doped']:+.1f}, undoped {t['undoped']:+.1f}, "
              f"T5 = {t['doped'] - t['undoped']:+.1f} meV (DFT -931.6; fixed reference geometries)")
    base = results[variants[0][0]][1]
    for label, _, _ in variants[1:]:
        dE = {k: 1e3 * (results[label][1][k] - base[k]) for k in base}
        print(f"  dE per structure ({label}), meV: " + ", ".join(f"{k[0]}/{k[1]} {v:+.1f}" for k, v in dE.items()))


if __name__ == "__main__":
    main()
