"""Is the iteration growth caused by the far-field Coulomb missing from the preconditioner?

For rattled NaCl supercells, matrix-free PBC (direct k-sum), converged solve:
  (a) the shipped preconditioner (Woodbury with diag(eta) [+ near-field J if n <= 2048]);
  (b) an ORACLE: the same Woodbury with C = eta + J_full, the exact periodic charge
      kernel (real + reciprocal + self), built column by column from the operator.
If (b) is flat in n while (a) grows, the far field is the cause.

Also estimates the extreme eigenvalues of the preconditioned operator from the CG
(Lanczos) coefficients of run (a).
Usage: python precond_oracle.py REPO DEVICE
"""
import sys
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, sys.argv[1])
device = torch.device(sys.argv[2] if len(sys.argv) > 2 else "cpu")

from ase.build import bulk  # noqa: E402
from ase.neighborlist import neighbor_list  # noqa: E402

import trace_electrostatics.implicit as implicit  # noqa: E402
import trace_electrostatics.lattice_operator as lo  # noqa: E402
from trace_electrostatics import ElectrostaticConfig, ElectrostaticHeads, electrostatic_state  # noqa: E402

torch.manual_seed(0)
heads = ElectrostaticHeads(r_max=4.0).to(device)

# --- record CG coefficients of the outer solve -------------------------------
record = {}
_orig_pcg = implicit._pcg


def recording_pcg(product, rhs, mask, precondition, x0, rtol, max_iter, name):
    if name != "solve_lattice":
        return _orig_pcg(product, rhs, mask, precondition, x0, rtol, max_iter, name)
    alphas, betas = [], []
    keep = (lambda v: v) if mask is None else (lambda v: v * mask)
    pre = (lambda v: keep(v)) if precondition is None else (lambda v: keep(precondition(v)))
    r = rhs - keep(product(x0)) if x0 is not None else rhs.clone()
    z = pre(r)
    p = z.clone()
    rz = r @ z
    target = rtol * float(torch.linalg.vector_norm(rhs))
    for _ in range(max_iter):
        Kp = keep(product(p))
        a = rz / (p @ Kp)
        r = r - a * Kp
        alphas.append(float(a))
        if float(torch.linalg.vector_norm(r)) <= target:
            break
        z = pre(r)
        rz_new = r @ z
        b = rz_new / rz
        betas.append(float(b))
        p = z + b * p
        rz = rz_new
    record["alphas"], record["betas"] = alphas, betas
    return _orig_pcg(product, rhs, mask, precondition, x0, rtol, max_iter, name)


implicit._pcg = recording_pcg

# --- oracle: exact periodic charge kernel in the Woodbury factor ---------------
oracle = {"on": False}
_orig_woodbury = lo._woodbury_bond_inverse
_state = {}


def oracle_woodbury(maps, bond_index, envelope, kappa, onsite, n_atoms, near_field=None):
    if oracle["on"] and "J" in _state:
        return _orig_woodbury(maps, bond_index, envelope, kappa, onsite, n_atoms, _state["J"])
    return _orig_woodbury(maps, bond_index, envelope, kappa, onsite, n_atoms, near_field)


lo._woodbury_bond_inverse = oracle_woodbury
_orig_lattice_init = lo.LatticeOperator.__init__


def full_kernel(op, n):
    """J_ij = d phi_i / d q_j from the operator itself (unit-charge columns)."""
    ell = torch.ones(n, dtype=torch.float64, device=device)
    zeros_mu = torch.zeros(n, 3, dtype=torch.float64, device=device)
    cols = []
    for j in range(n):
        q = torch.zeros(n, dtype=torch.float64, device=device)
        q[j] = 1.0
        phi, _, _ = op.apply_reduced(q, zeros_mu, ell, None)
        cols.append(phi)
    return torch.stack(cols, dim=1)


def tridiagonal_extremes(alphas, betas):
    k = len(alphas)
    T = np.zeros((k, k))
    for i in range(k):
        T[i, i] = 1.0 / alphas[i] + (betas[i - 1] / alphas[i - 1] if i > 0 else 0.0)
        if i + 1 < k:
            T[i, i + 1] = T[i + 1, i] = np.sqrt(betas[i]) / alphas[i]
    w = np.linalg.eigvalsh(T)
    return w[0], w[-1]


def run(n_rep, use_oracle):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(n_rep)
    atoms.rattle(0.05, seed=1)
    i, j, shift = neighbor_list("ijS", atoms, heads.r_max)
    pos = torch.tensor(atoms.positions, device=device)
    cell = torch.tensor(atoms.cell.array, device=device)
    edge_index = torch.tensor(np.stack([j, i]), device=device)
    edge_shift = torch.tensor(shift, dtype=torch.float64, device=device)
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]] + edge_shift @ cell
    g = torch.Generator().manual_seed(0)
    h = torch.randn(len(atoms), heads.irreps_in.dim, generator=g).to(device)
    config = ElectrostaticConfig(solver="matrix_free")
    oracle["on"] = False
    if use_oracle:
        # First pass to get the operator, then the kernel, then the oracle pass.
        captured = {}
        orig = lo.solve_lattice

        def capture(operator, *a, **k):
            captured["op"] = operator
            return orig(operator, *a, **k)

        lo.solve_lattice = capture
        import trace_electrostatics.energy as en
        en_solve = getattr(en, "solve_lattice", None)
        if en_solve is not None:
            en.solve_lattice = capture
        with torch.no_grad():
            electrostatic_state(h, pos, edge_index, edge_vec, edge_vec.norm(dim=-1), heads, config,
                                pbc=True, cell=cell, edge_shift=edge_shift)
        lo.solve_lattice = orig
        if en_solve is not None:
            en.solve_lattice = en_solve
        J = full_kernel(captured["op"], len(atoms))
        _state["J"] = 0.5 * (J + J.T)
        oracle["on"] = True
    t0 = time.perf_counter()
    with torch.no_grad():
        state = electrostatic_state(h, pos, edge_index, edge_vec, edge_vec.norm(dim=-1), heads, config,
                                    pbc=True, cell=cell, edge_shift=edge_shift)
    dt = time.perf_counter() - t0
    oracle["on"] = False
    _state.pop("J", None)
    lo_, hi_ = tridiagonal_extremes(record["alphas"], record["betas"])
    return int(state.solve_result.diagnostics.n_iter), lo_, hi_, dt


print(f"device {device}")
for n_rep in (2, 3, 4, 5):
    n = 8 * n_rep ** 3
    a = run(n_rep, False)
    b = run(n_rep, True)
    print(f"n={n:5d}  shipped: {a[0]:4d} it, lambda [{a[1]:.3g}, {a[2]:.3g}] cond {a[2]/a[1]:8.1f}   "
          f"oracle(full J): {b[0]:4d} it, lambda [{b[1]:.3g}, {b[2]:.3g}] cond {b[2]/b[1]:8.1f}", flush=True)
