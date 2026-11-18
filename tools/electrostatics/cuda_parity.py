"""CPU vs CUDA parity of trace_electrostatics in float64, every route.

For each case: energy, forces (autograd), and where defined a response tensor,
computed on the CPU and on the GPU from IDENTICAL inputs (heads built once on
the CPU with a fixed seed, then copied).  Reports max |difference| relative to
the CPU magnitude, and wall time per device (GPU timings are synchronised; the
card may be shared, so they are indicative only).

Usage: python cuda_parity.py [REPO]
"""
import sys
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
REPO = sys.argv[1] if len(sys.argv) > 1 else "."
sys.path.insert(0, REPO)

from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402
from ase.neighborlist import neighbor_list  # noqa: E402

from trace_electrostatics import (  # noqa: E402
    ElectrostaticConfig,
    ElectrostaticHeads,
    born_charges,
    bond_polarization,
    dielectric_tensor,
    electrostatic_state,
    polarizability,
)

assert torch.cuda.is_available(), "no CUDA device"
torch.manual_seed(0)
HEADS_CPU = ElectrostaticHeads(r_max=4.0)


def graph(atoms, cutoff, device):
    i, j, shift = neighbor_list("ijS", atoms, cutoff)
    pos = torch.tensor(atoms.positions, device=device, requires_grad=True)
    cell = torch.tensor(atoms.cell.array, device=device)
    edge_index = torch.tensor(np.stack([j, i]), device=device)
    edge_shift = torch.tensor(shift, dtype=torch.float64, device=device)
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]] + edge_shift @ cell
    return pos, cell, edge_index, edge_shift, edge_vec, edge_vec.norm(dim=-1)


def features(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, HEADS_CPU.irreps_in.dim, generator=g)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_case(name, atoms, config, device, *, periodic, response=None, **extra):
    heads = HEADS_CPU.to(device)
    pos, cell, edge_index, edge_shift, edge_vec, edge_len = graph(atoms, heads.r_max, device)
    h = features(len(atoms)).to(device)
    kwargs = dict(extra)
    for key, value in list(kwargs.items()):
        if isinstance(value, torch.Tensor):
            kwargs[key] = value.to(device)
    if periodic:
        kwargs.update(pbc=tuple(bool(p) for p in atoms.pbc), cell=cell, edge_shift=edge_shift)
    sync(device)
    t0 = time.perf_counter()
    state = electrostatic_state(h, pos, edge_index, edge_vec, edge_len, heads, config, **kwargs)
    energy = state.energy
    forces = -torch.autograd.grad(energy, pos, retain_graph=response is not None)[0]
    out = {"E": energy.detach().cpu().numpy(), "F": forces.detach().cpu().numpy(),
           "q": state.solve_result.q.detach().cpu().numpy()}
    if response == "alpha":
        out["alpha"] = polarizability(state.solve_result, state.assembly).detach().cpu().numpy()
    elif response == "eps":
        out["eps"] = dielectric_tensor(state.solve_result, state.assembly).detach().cpu().numpy()
        out["Z"] = born_charges(bond_polarization(state.solve_result, state.assembly), pos).detach().cpu().numpy()
    sync(device)
    out["seconds"] = time.perf_counter() - t0
    out["iterations"] = int(state.solve_result.diagnostics.n_iter)
    return out


def cases():
    d = 1.03 / np.sqrt(3.0)
    nh4 = Atoms("NH4", positions=[[0, 0, 0], [d, d, d], [d, -d, -d], [-d, d, -d], [-d, -d, d]])
    yield "NH4+ dense open", nh4, ElectrostaticConfig(), dict(periodic=False, response="alpha", total_charge=1.0)

    rng = np.random.default_rng(3)
    waters = []
    for k in range(8):                       # 8 waters on a 3 A grid, jittered
        o = np.array([3.0 * (k % 2), 3.0 * ((k // 2) % 2), 3.0 * (k // 4)]) + 0.1 * rng.standard_normal(3)
        waters += [o, o + [0.757, 0.586, 0.0], o + [-0.757, 0.586, 0.0]]
    w8 = Atoms("OH2" * 8, positions=np.array(waters))
    yield "(H2O)8 dense open", w8, ElectrostaticConfig(), dict(periodic=False, response="alpha")
    yield "(H2O)8 matrix-free open", w8, ElectrostaticConfig(solver="matrix_free"), dict(periodic=False)

    nacl = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    nacl.rattle(0.05, seed=1)
    yield "NaCl 8 dense PBC", nacl, ElectrostaticConfig(), dict(periodic=True, response="eps")

    big = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(3)
    big.rattle(0.05, seed=1)
    yield "NaCl 216 matrix-free Ewald", big, ElectrostaticConfig(solver="matrix_free"), dict(periodic=True)
    yield "NaCl 216 matrix-free PME", big, ElectrostaticConfig(solver="matrix_free", reciprocal="pme"), dict(periodic=True)

    slab = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 1))
    slab.center(vacuum=10.0, axis=2)
    slab.pbc = (True, True, False)
    slab.rattle(0.05, seed=2)
    yield "NaCl slab 32 dense", slab, ElectrostaticConfig(), dict(periodic=True)

    plate = [[i * 2.5, j * 2.5, 0.0] for i in range(4) for j in range(4)] + [[3.75, 3.75, 4.0]]
    metal = Atoms("Cu16Na", positions=plate)
    n = len(metal)
    yield "Cu16 plate + Na+ (metal)", metal, ElectrostaticConfig(), dict(
        periodic=False,
        metallic=torch.tensor([True] * 16 + [False]),
        fragment_id=torch.tensor([0] * 16 + [1]),
        fragment_charges=[0.0, 1.0],
    )


def main():
    cpu, gpu = torch.device("cpu"), torch.device("cuda")
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    worst = 0.0
    for name, atoms, config, kw in cases():
        a = run_case(name, atoms, config, cpu, **kw)
        run_case(name, atoms, config, gpu, **kw)          # warm-up (kernels, caches)
        b = run_case(name, atoms, config, gpu, **kw)
        line = [f"{name:28s} n={len(atoms):4d} it {a['iterations']:3d}/{b['iterations']:3d}"]
        for key in ("E", "F", "q", "alpha", "eps", "Z"):
            if key not in a:
                continue
            scale = max(float(np.abs(a[key]).max()), 1e-300)
            rel = float(np.abs(a[key] - b[key]).max()) / scale
            worst = max(worst, rel)
            line.append(f"{key} {rel:.1e}")
        line.append(f"t cpu {a['seconds']:.3f}s gpu {b['seconds']:.3f}s")
        print("  ".join(line), flush=True)
    print(f"WORST relative CPU-GPU difference: {worst:.2e}")


if __name__ == "__main__":
    main()
