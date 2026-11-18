"""SPEC 36.8 L1 measured (README R22): the free energy of the shipped expansion (one- and two-cluster increments)
against the free energy on the full product space of all clusters (the same exact configuration energies, the same
switched couplings O_c).  F_pairs - F_full is what the expansion drops: the connected terms of three or more
clusters.  Frozen weights, doped cells, 'onsite', clusters space with the closure, band_form 'fermi'.  The copy of
the scratch script that produced R22's table; it patches the cluster routine in this process only and writes nothing.

    PYTHONPATH=. python experiments/benchmarks_4g/probe_ext_full_product.py MODEL.pt geometry nx ny atom dz1 [dz2 ...]

(atom and dz: one atom displaced along z by each dz, in A; dz = 0 is the reference geometry.)"""
import dataclasses, hashlib, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "experiments/benchmarks_4g")
import numpy as np, torch
from flashace.calculator import TransformersACECalculator
from eval_aumgo import references
import trace_redox.clusters as cl

model_path, geometry, nx, ny, atom = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
shifts = [float(x) for x in sys.argv[6:]]
original = cl.cluster_free_energy
LAST = {}

def wrapped(n0, clusters, energy, hamiltonian, kT, max_charge, min_charge, states, *, units=None, frames=None, **kw):
    out = original(n0, clusters, energy, hamiltonian, kT, max_charge, min_charge, states, units=units, frames=frames, **kw)
    with torch.no_grad():
        n0l = n0.to(torch.long); U = int(n0l.numel())
        K = max_charge.to(torch.long); L = cl._lower(min_charge, K)
        C = [S.to(torch.long) for S in out.clusters]
        P = n0l.clone().reshape(*([1] * len(C)), U)
        for k, S in enumerate(C):
            shape = [1] * len(C) + [U]; shape[k] = int(S.shape[0])
            P = P + (S - n0l).reshape(shape)
        P = P.reshape(-1, U)
        valid = ((P >= L) & (P <= K)).all(-1)
        if units is not None:
            valid = valid & cl.admissible(P, units, frames)
        keep = valid.nonzero().flatten()
        E = energy(P[keep]).detach()
        dims = [int(S.shape[0]) for S in C]
        H = torch.zeros(int(np.prod(dims)), int(np.prod(dims)), dtype=E.dtype)
        for k, Hc in enumerate(out.hamiltonians):
            O = (Hc - torch.diag(torch.diagonal(Hc))).detach()
            term = torch.ones(1, 1, dtype=E.dtype)
            for j, d in enumerate(dims):
                term = torch.kron(term, O if j == k else torch.eye(d, dtype=E.dtype))
            H = H + term
        H = H[keep][:, keep] + torch.diag(E)
        F_full = float(cl.canonical_free_energy(H, kT))
        LAST.update(F_pairs=float(out.energy), F_full=F_full, n_full=int(keep.numel()), dims=dims,
                    ref=hashlib.md5(n0l.cpu().numpy().tobytes()).hexdigest()[:8])
    return out

cl.cluster_free_energy = wrapped
calc = TransformersACECalculator(model_path=model_path, device="cpu")
torch.set_num_threads(4)
mod = calc.model.electrostatics
mod.redox = dataclasses.replace(mod.redox, state_polarizability_stiffness="onsite", diabatic_space="clusters", band_form="fermi")
calc.atomic_energy_tensor = torch.zeros_like(calc.atomic_energy_tensor)
a0 = references()[("doped", geometry)].copy(); a0.set_constraint(); a0 = a0.repeat((nx, ny, 1)); a0.info = {"charge": 0.0}
print(f"{geometry} {nx}x{ny} fermi clusters; atom {atom} along z")
for dz in shifts:
    a = a0.copy(); a.info = {"charge": 0.0}; a.positions[atom, 2] += dz
    a.calc = calc; calc.reset(); LAST.clear()
    e = a.get_potential_energy()
    print(f"  dz {dz:+.6f}: E {e:.10f}  ref {LAST['ref']}  clusters {LAST['dims']}  full space {LAST['n_full']}  "
          f"F_pairs {LAST['F_pairs']:.10f}  F_full {LAST['F_full']:.10f}  F_pairs - F_full {LAST['F_pairs'] - LAST['F_full']:+.4e}  "
          f"E - F_pairs {e - LAST['F_pairs']:.10f}", flush=True)
