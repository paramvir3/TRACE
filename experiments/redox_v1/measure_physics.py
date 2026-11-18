"""TRACE-Redox v1 physics measurements (untrained heads; Mac CPU; float64).

1. lam_max = lambda_min(D^-1/2 W D^-1/2) on realistic geometries (SPEC 25.2):
   how much residual self-interaction can be removed while F stays convex.
2. Na...Cl at separation R, Q = 0: free charge on Na versus R, against the
   two-site law of SPEC 25.4 (spurious fractional transfer at large R).
Run: python experiments/redox_v1/measure_physics.py
"""
import math
import sys
import warnings

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, ".")
from trace_electrostatics.config import ElectrostaticConfig  # noqa: E402
from trace_electrostatics.heads import ElectrostaticHeads  # noqa: E402
from trace_redox import RedoxConfig, RedoxElectrostatics, lambda_max  # noqa: E402
from trace_redox.bound import assemble_neutral, eliminate_bound_sector  # noqa: E402
from trace_redox.constants import ATOMIC_ELECTRON_AFFINITY_EV as EA, ATOMIC_IONIZATION_EV as IE  # noqa: E402

R_HEADS = 3.5


def heads():
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ElectrostaticHeads(r_max=R_HEADS, hardness_margin=1.0)


def graph(pos, r_max=4.5):
    d = torch.cdist(pos, pos)
    src, dst = torch.nonzero((d < r_max) & (d > 0), as_tuple=True)
    ei = torch.stack([src, dst])
    vec = pos[ei[0]] - pos[ei[1]]
    return ei, vec, vec.norm(dim=-1)


def features(n, dim, seed=1):
    return torch.randn(n, dim, generator=torch.Generator().manual_seed(seed))


def water(o, flip=1.0):
    r, a = 0.9572, math.radians(104.52) / 2
    return [o, o + torch.tensor([r * math.sin(a), flip * r * math.cos(a), 0.0]),
            o + torch.tensor([-r * math.sin(a), flip * r * math.cos(a), 0.0])]


GEOMETRIES = {
    "NaCl dimer, R = 2.36 A": ([11, 17], torch.tensor([[0.0, 0, 0], [2.36, 0, 0]])),
    "water dimer, R_OO = 2.91 A": ([8, 1, 1, 8, 1, 1], torch.stack(water(torch.zeros(3)) + water(torch.tensor([0.0, 0.0, 2.91]), -1.0))),
    "(NaCl)4 rock-salt cube, a = 2.82 A": (
        [11 if (i + j + k) % 2 == 0 else 17 for i in (0, 1) for j in (0, 1) for k in (0, 1)],
        torch.tensor([[i * 2.82, j * 2.82, k * 2.82] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=torch.float64)),
    "benzene (C-C 1.39, C-H 1.09 A)": (
        [6] * 6 + [1] * 6,
        torch.tensor([[1.39 * math.cos(t), 1.39 * math.sin(t), 0.0] for t in [k * math.pi / 3 for k in range(6)]]
                     + [[2.48 * math.cos(t), 2.48 * math.sin(t), 0.0] for t in [k * math.pi / 3 for k in range(6)]])),
}


def main():
    hd = heads()
    cfg = ElectrostaticConfig()
    print("1. lam_max (SPEC 25.2), hardness widths with h = 0.5, untrained bound heads")
    for h_frac in (0.5, 0.8):
        torch.manual_seed(1)
        module = RedoxElectrostatics(hd, cfg, RedoxConfig(hartree_fraction=h_frac, sic="none", fragments="none"))
        for name, (z, pos) in GEOMETRIES.items():
            z = torch.tensor(z)
            ei, vec, length = graph(pos)
            h = features(len(z), hd.irreps_in.dim)
            assembly = assemble_neutral(h, pos, ei, vec, length, hd, cfg, species=z)
            sigma_f = module.width(z)
            elim = eliminate_bound_sector(assembly, sigma_f)
            lam = float(lambda_max(elim.W, torch.diagonal(elim.J)))
            print(f"   h = {h_frac}: {name:38s} sigma_f = {[round(float(s), 2) for s in sigma_f.unique()]} A  lam_max = {lam:.4f}")

    print("\n2. Na...Cl, Q = 0, lam = 0, sic='none' (v1): free charge on Na versus R")
    u_na, u_cl = IE[11] - EA[11], IE[17] - EA[17]
    print(f"   integer transfer cost at R -> inf: IE_Na - EA_Cl = {IE[11] - EA[17]:.4f} eV; "
          f"point-charge crossing K_E / (IE_Na - EA_Cl) = {14.399645 / (IE[11] - EA[17]):.3f} A")
    for h_frac in (0.5, 0.2):
        delta_inf = min(max((h_frac * (u_na + u_cl) / 2 - (IE[11] - EA[17])) / (h_frac * (u_na + u_cl)), 0.0), 1.0)
        print(f"   h = {h_frac}: two-site law at R -> inf, T -> 0: delta = clip(1/2 - dE/(h (U_Na + U_Cl))) = {delta_inf:.4f} e")
        for kT in (0.1, 0.01):
            torch.manual_seed(1)
            module = RedoxElectrostatics(hd, cfg, RedoxConfig(hartree_fraction=h_frac, kT=kT, sic="none", fragments="none"))
            row = []
            for R in (2.36, 4.0, 6.0, 9.43, 15.0, 30.0, 100.0):
                pos = torch.tensor([[0.0, 0, 0], [R, 0, 0]])
                ei, vec, length = graph(pos)
                energy, res = module(features(2, hd.irreps_in.dim), pos, ei, vec, length,
                                     data={"z": torch.tensor([11, 17]), "total_charge": 0.0})
                row.append(f"R={R:g}: {float(res.unit_charge[0]):.3f}")
            print(f"      kT = {kT}: " + "  ".join(row))


def certified():
    """3. Certified self-interaction removal (SPEC §8a): Na...Cl versus R, and r_u / W_uu in condensed geometries."""
    hd = heads()
    cfg = ElectrostaticConfig()
    print("\n3. sic='certified': free charge on Na versus R (Q = 0), kT = 0.01 eV")
    for h_frac in (0.5, 0.8):
        torch.manual_seed(1)
        module = RedoxElectrostatics(hd, cfg, RedoxConfig(hartree_fraction=h_frac, kT=0.01, sic="certified", fragments="none"))
        row = []
        for R in (2.36, 4.0, 6.0, 9.43, 15.0, 30.0, 100.0):
            pos = torch.tensor([[0.0, 0, 0], [R, 0, 0]])
            ei, vec, length = graph(pos)
            _, res = module(features(2, hd.irreps_in.dim), pos, ei, vec, length,
                            data={"z": torch.tensor([11, 17]), "total_charge": 0.0})
            row.append(f"R={R:g}: {float(res.unit_charge[0]):.3f}")
        print(f"   h = {h_frac}: " + "  ".join(row))
    print("   removed fraction r_u / W_uu (min / median / max) in condensed geometries, h = 0.5:")
    torch.manual_seed(1)
    module = RedoxElectrostatics(hd, cfg, RedoxConfig(sic="certified", fragments="none"))
    from trace_redox.sic import certified_removal
    for name, (z, pos) in GEOMETRIES.items():
        z = torch.tensor(z)
        ei, vec, length = graph(pos)
        assembly = assemble_neutral(features(len(z), hd.irreps_in.dim), pos, ei, vec, length, hd, cfg, species=z)
        elim = eliminate_bound_sector(assembly, module.width(z))
        frac = certified_removal(elim.W, elim.J) / torch.diagonal(elim.W)
        print(f"      {name:38s} {float(frac.min()):.4f} / {float(frac.median()):.4f} / {float(frac.max()):.4f}")


if __name__ == "__main__":
    main()
    certified()
