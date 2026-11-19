"""Frozen checks of the bond-oriented Pauli term E_bop (2026-11-04; theory note §4c, pre-registered 21:19).

    E_bop = sum over the bonded s-class atoms a (partner b) and the O (frame -2) atoms j within 4.3 A of
            s_bond(R_ab) s_cut(R_aj) [D1(R)(cos t - 1) + D2(R)(P2(cos t) - 1)],   cos t = r_aj . (-e_ab),
    D2(R) = -0.611 sqrt(f(R)),  D1(R) = -0.183 f(R)^0.888,  f(R) = E1(R) / E1(2.26),
E1 the lone Au0's frozen repulsion on the O-top site (tools/electrostatics/frz_angular.py: 1.4254 / 0.8454 / 0.3891 /
0.1045 eV at 2.26 / 2.4 / 2.6 / 2.9 A), interpolated linearly in ln E1 and extrapolated with the end slopes;
s_cut a C2 switch from 1 at 3.8 A to 0 at 4.3 A; s_bond from 1 at 2.9 A to 0 at 3.4 A (the Au-Au bond).  Forces by
autograd.  Usage (repo root): python experiments/benchmarks_4g/probe_bop_frozen.py PATH_A.json MODEL.pt [...]
(F1) the tilt path A; (F2) the undoped flat reference relaxed by the T5 protocol; (F3) wet_u, wet_d, T5 at the references
and relaxed; (F4) E_bop's class means on the training set (it is geometric: it moves each class offset by them)."""
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
warnings.simplefilter("ignore")
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
from ase.calculators.calculator import Calculator, all_changes  # noqa: E402
from ase.io import read  # noqa: E402
from flashace.calculator import TransformersACECalculator  # noqa: E402
from eval_aumgo import AU, references, relax  # noqa: E402

R_TAB = torch.tensor([2.26, 2.4, 2.6, 2.9])
LN_E1 = torch.log(torch.tensor([1.4254, 0.8454, 0.3891, 0.1045]))
E1_REF = 1.4254


def _switch(x, a, b):
    """C2 switch: 1 for x <= a, 0 for x >= b."""
    u = ((x - a) / (b - a)).clamp(0.0, 1.0)
    return 1.0 - u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)


def _ln_e1(R):
    s_lo = (LN_E1[1] - LN_E1[0]) / (R_TAB[1] - R_TAB[0])
    s_hi = (LN_E1[-1] - LN_E1[-2]) / (R_TAB[-1] - R_TAB[-2])
    out = torch.where(R < R_TAB[0], LN_E1[0] + s_lo * (R - R_TAB[0]), LN_E1[-1] + s_hi * (R - R_TAB[-1]))
    for k in range(len(R_TAB) - 1):
        inside = (R >= R_TAB[k]) & (R < R_TAB[k + 1])
        val = LN_E1[k] + (LN_E1[k + 1] - LN_E1[k]) * (R - R_TAB[k]) / (R_TAB[k + 1] - R_TAB[k])
        out = torch.where(inside, val, out)
    return out


MODE = os.environ.get("BOP_MODE", "referenced")      # referenced | absolute | data (theory note §4c, 21:24 / 21:36)
ISO = None                                           # (bin centres, <D>) for the data-referenced form


LEVEL = os.environ.get("BOP_LEVEL", "frz")          # frz | almo: the constants of trace_redox.bond_pauli (SPEC 33)
from trace_redox.bond_pauli import LEVELS            # noqa: E402  (the module's own constants, so the probe never drifts)
C1, C2, P1, P2 = LEVELS[LEVEL]


def _law(R, cos):
    f = torch.exp(_ln_e1(R)) / E1_REF
    return C1 * f ** P1 * cos + C2 * f ** P2 * 0.5 * (3.0 * cos * cos - 1.0)


def build_iso(train):
    """E_iso(R) = <D(R, theta)> over the training set's bonded Au-O pairs (0.1 A bins, switch-weighted)."""
    edges = np.arange(2.0, 4.31, 0.1)
    num, den = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    for atoms in train:
        for R, cos, w in _pairs(torch.tensor(atoms.positions), atoms.numbers, atoms.cell.array, atoms.pbc):
            k = np.searchsorted(edges, R) - 1
            if 0 <= k < len(num):
                num[k] += w * float(_law(torch.tensor(R), torch.tensor(cos)))
                den[k] += w
    centres = 0.5 * (edges[1:] + edges[:-1])
    have = den > 0
    vals = np.interp(centres, centres[have], num[have] / den[have])        # empty bins: linear between, flat outside
    return torch.tensor(centres), torch.tensor(vals), den


def _pairs(pos, numbers, cell, pbc):
    """(R, cos t, weight) of every bonded Au - O pair (numbers only; for the table)."""
    au = [i for i, z in enumerate(numbers) if z == 79]
    ox = [i for i, z in enumerate(numbers) if z == 8]
    L = torch.as_tensor(np.diag(cell)); mask = torch.as_tensor(np.asarray(pbc, dtype=bool))
    mic = lambda d: torch.where(mask, d - torch.round(d / L) * L, d)
    out = []
    for a, b in ((au[0], au[1]), (au[1], au[0])):
        dab = mic(pos[b] - pos[a]); rab = torch.linalg.norm(dab); e = dab / rab
        d = mic(pos[ox] - pos[a]); R = torch.linalg.norm(d, dim=1); c = (d / R.unsqueeze(1)) @ (-e)
        w = _switch(rab, 2.9, 3.4) * _switch(R, 3.8, 4.3)
        for r_, c_, w_ in zip(R.tolist(), c.tolist(), w.tolist()):
            if w_ > 0:
                out.append((r_, c_, w_))
    return out


def bop_energy(pos, numbers, cell, pbc):
    """E_bop (eV) for positions ``pos`` [n, 3] (a torch tensor; differentiable)."""
    au = [i for i, z in enumerate(numbers) if z == 79]
    ox = [i for i, z in enumerate(numbers) if z == 8]
    if len(au) != 2 or not ox:
        return pos.sum() * 0.0
    L = torch.as_tensor(np.diag(cell), dtype=pos.dtype)
    assert np.allclose(cell, np.diag(np.diag(cell))), "an orthorhombic cell is assumed"
    mask = torch.as_tensor(np.asarray(pbc, dtype=bool))

    def mic(d):
        return torch.where(mask, d - torch.round(d / L) * L, d)

    E = pos.sum() * 0.0
    for a, b in ((au[0], au[1]), (au[1], au[0])):
        dab = mic(pos[b] - pos[a])
        rab = torch.linalg.norm(dab)
        e_ab = dab / rab
        d = mic(pos[ox] - pos[a])                                   # [m, 3]
        R = torch.linalg.norm(d, dim=1)
        cos = (d / R.unsqueeze(1)) @ (-e_ab)
        if MODE == "referenced":
            term = _law(R, cos) - _law(R, torch.ones_like(cos))
        elif MODE == "absolute":
            term = _law(R, cos)
        else:                                            # data-referenced: the law minus its data-averaged isotropic part
            c_, v_ = ISO[0].to(R.dtype), ISO[1].to(R.dtype)
            iso = torch.stack([torch.from_numpy(np.interp(r.detach().numpy().reshape(-1), c_.numpy(), v_.numpy()))
                               for r in [R]])[0]
            # linear interpolation, differentiable in R through the bin slope
            k = torch.clamp(torch.searchsorted(c_, R.detach()) - 1, 0, len(c_) - 2)
            x0, x1, y0, y1 = c_[k], c_[k + 1], v_[k], v_[k + 1]
            iso = torch.where(R <= c_[0], v_[0], torch.where(R >= c_[-1], v_[-1], y0 + (y1 - y0) * (R - x0) / (x1 - x0)))
            term = _law(R, cos) - iso
        E = E + _switch(rab, 2.9, 3.4) * (_switch(R, 3.8, 4.3) * term).sum()
    return E


def bop(atoms):
    pos = torch.tensor(atoms.positions, requires_grad=True)
    E = bop_energy(pos, atoms.numbers, atoms.cell.array, atoms.pbc)
    (g,) = torch.autograd.grad(E, pos)
    return float(E), -g.numpy()


class WithBop(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, base):
        super().__init__()
        self.base = base

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = atoms.copy(); a.info = dict(atoms.info); a.calc = self.base; self.base.reset()
        e, f = a.get_potential_energy(), a.get_forces()
        eb, fb = bop(atoms)
        self.results = {"energy": e + eb, "forces": f + fb}


def energy(calc, atoms):
    a = atoms.copy(); a.info = {"charge": 0.0}; a.calc = calc; calc.reset()
    return a.get_potential_energy()


def heights(atoms):
    top = atoms.positions[:AU[0], 2].min()
    return [float(top - atoms.positions[k, 2]) for k in AU]


def main():
    global ISO
    P = json.load(open(sys.argv[1]))
    refs = references()
    if MODE == "data":
        ISO = build_iso(read(os.path.join(HERE, "data", "AuMgO_train.xyz"), ":"))
        print("E_iso(R) (data-averaged law, eV): " + ", ".join(f"{c:.2f}: {v:+.3f} (w {d:.1f})" for c, v, d in
                                                              zip(ISO[0].tolist(), ISO[1].tolist(), ISO[2])), flush=True)
    print(f"BOP_MODE = {MODE}, BOP_LEVEL = {LEVEL} (C1 {C1}, C2 {C2}, p1 {P1}, p2 {P2})", flush=True)
    # (F4) geometric: E_bop's class means on the training set
    train = read(os.path.join(HERE, "data", "AuMgO_train.xyz"), ":")
    cls = {}
    for a in train:
        p = a.positions[a.numbers == 79]; v = p[1] - p[0]
        g = "flat" if np.degrees(np.arccos(abs(v[2]) / np.linalg.norm(v))) > 45 else "upright"
        cls.setdefault(("doped" if (a.numbers == 13).any() else "undoped", g), []).append(bop(a)[0])
    for d in ("doped", "undoped"):
        f, u = np.mean(cls[(d, "flat")]), np.mean(cls[(d, "upright")])
        print(f"(F4) training set, {d}: E_bop class means flat {f * 1e3:+.1f}, upright {u * 1e3:+.1f} -> moves the class "
              f"offset (flat - upright) by {(f - u) * 1e3:+.1f} meV", flush=True)
    for key, at in refs.items():
        print(f"     E_bop at the reference {key}: {bop(at)[0] * 1e3:+.1f} meV", flush=True)
    ref = refs[("undoped", "wetting")]
    for model in sys.argv[2:]:
        base = TransformersACECalculator(model_path=model, device="cpu")
        torch.set_num_threads(3)
        calc = WithBop(base)
        name = os.path.basename(os.path.dirname(model))[-40:]
        print(f"=== {name}", flush=True)
        # (F1) the tilt path A
        E0, E1 = {}, {}
        for lam in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            a = ref.copy(); a.positions[AU] = np.array(P["cell_au"][str(lam)])
            E0[lam], E1[lam] = energy(base, a), energy(calc, a)
        print("  (F1) path A, as trained / with E_bop (eV; PBE +0.078 / +0.255 / +0.236 / - / -0.197 at 0.2 / 0.4 / 0.6 / "
              "1.0): " + "  ".join(f"{l:.1f}: {E0[l] - E0[0.0]:+.3f} / {E1[l] - E1[0.0]:+.3f}" for l in E0), flush=True)
        # (F3) at the references and relaxed; (F2) the undoped flat reference's relaxation
        Er = {k: (energy(base, at), energy(calc, at)) for k, at in refs.items()}
        wu = [Er[("undoped", "wetting")][i] - Er[("undoped", "non-wetting")][i] for i in (0, 1)]
        wd = [Er[("doped", "wetting")][i] - Er[("doped", "non-wetting")][i] for i in (0, 1)]
        print(f"  (F3) references: wet_u {wu[0] * 1e3:+.1f} -> {wu[1] * 1e3:+.1f}, wet_d {wd[0] * 1e3:+.1f} -> "
              f"{wd[1] * 1e3:+.1f}, T5 {(wd[0] - wu[0]) * 1e3:+.1f} -> {(wd[1] - wu[1]) * 1e3:+.1f} meV "
              f"(DFT +928.9 / -2.7 / -931.6)", flush=True)
        rel = {}
        for k, at in refs.items():
            out, e = relax(calc, at)
            rel[k] = (out, e)
        h0, h1 = heights(ref), heights(rel[("undoped", "wetting")][0])
        e_flat0 = Er[("undoped", "wetting")][1]
        ok = all(abs(x - y) <= 0.3 for x, y in zip(h0, h1))
        print(f"  (F2) flat undoped relaxed with E_bop: heights {np.round(h0, 3).tolist()} -> {np.round(h1, 3).tolist()} A, "
              f"relaxation {(rel[('undoped', 'wetting')][1] - e_flat0) * 1e3:+.1f} meV -> {'PASS' if ok else 'FAIL'}", flush=True)
        wur = rel[("undoped", "wetting")][1] - rel[("undoped", "non-wetting")][1]
        wdr = rel[("doped", "wetting")][1] - rel[("doped", "non-wetting")][1]
        print(f"  (F3) relaxed (T5 protocol) with E_bop: wet_u {wur * 1e3:+.1f}, wet_d {wdr * 1e3:+.1f}, T5 "
              f"{(wdr - wur) * 1e3:+.1f} meV", flush=True)


if __name__ == "__main__":
    main()
