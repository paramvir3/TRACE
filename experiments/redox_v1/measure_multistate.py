"""Where does a +2 charge sit with states up to +1 (K = 1) versus +2 (K = 2)?

M(H2O)4 with M = Mg, Fe, Zn, Q = +2, M-O 2.08 A tetrahedral, untrained heads (the
anchors and the electrostatics only).  Run: python experiments/redox_v1/measure_multistate.py
"""
import math
import sys
import warnings

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, ".")
from trace_electrostatics.config import ElectrostaticConfig  # noqa: E402
from trace_electrostatics.heads import ElectrostaticHeads  # noqa: E402
from trace_redox import RedoxConfig, RedoxElectrostatics  # noqa: E402


def aqua(metal_z, r_mo=2.08):
    """M at the origin, four waters with O on a tetrahedron, H pointing away."""
    dirs = torch.tensor([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=torch.float64) / math.sqrt(3)
    pos, z = [torch.zeros(3)], [metal_z]
    for d in dirs:
        o = d * r_mo
        perp = torch.linalg.svd(d.unsqueeze(0)).Vh[1]          # a unit vector normal to d
        h1 = o + 0.9572 * (math.cos(math.radians(52.26)) * d + math.sin(math.radians(52.26)) * perp)
        h2 = o + 0.9572 * (math.cos(math.radians(52.26)) * d - math.sin(math.radians(52.26)) * perp)
        pos += [o, h1, h2]
        z += [8, 1, 1]
    return torch.tensor(z), torch.stack(pos)


def main():
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heads = ElectrostaticHeads(r_max=3.5, hardness_margin=1.0)
    for metal, zm in (("Mg", 12), ("Fe", 26), ("Zn", 30)):
        z, pos = aqua(zm)
        d = torch.cdist(pos, pos)
        src, dst = torch.nonzero((d < 4.5) & (d > 0), as_tuple=True)
        ei = torch.stack([src, dst])
        vec = pos[ei[0]] - pos[ei[1]]
        h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(1))
        row = []
        for K in (1, 2):
            torch.manual_seed(1)
            module = RedoxElectrostatics(heads, ElectrostaticConfig(), RedoxConfig(max_charge=K, kT=0.05, fragments="none"))
            energy, res = module(h, pos, ei, vec, vec.norm(dim=-1), data={"z": z, "total_charge": 2.0})
            c = res.unit_charge
            row.append(f"K={K}: E={float(energy):8.3f} eV, free charge on {metal} {float(c[0]):+.3f}, "
                       f"on the O {float(c[1::3].sum()):+.3f}, on the H {float(c[2::3].sum() + c[3::3].sum()):+.3f}")
        print(f"{metal}(H2O)4 2+: " + " | ".join(row))


if __name__ == "__main__":
    main()
