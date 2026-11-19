"""State-dependent polarisabilities in the configuration energies (trace_redox SPEC 28).

The bound sector gives atom j one dipole stiffness A_mu,j (its isolated polarisability ell^2 A_mu,j^-1) in every
charge state.  With the state's own polarisability alpha_j(n) = rho_j(n) alpha_j(f), rho = alpha(Z, n) / alpha(Z, f)
from first principles (data/ion_polarizabilities.json: X2C-PBE0/ANO-RCC finite field, the ion-density library's method;
tools/electrostatics/ion_polarizabilities.py), K(n) = K + sum_j P_j^T Delta_j P_j with
Delta_j = A_mu,j (1 / rho_j - 1), and by Woodbury, exactly for every configuration,

    dE(n) = 1/2 y_S(n)^T Delta_S (I + C_SS Delta_S)^-1 y_S(n),   y(n) = y0 - K^-1 A^T q(n),   C = K^-1 on the dipoles,

S the atoms away from their frame with rho != 1.  One isolated atom: dE = -1/2 (alpha(n) - alpha(f)) E_loc^2.
ESTABLISHED: the polarisability of an ion depends on its charge state; Woodbury.  NEW-TO-TRACE: the exact
state-resolved induction of every integer configuration on the polarisable bound sector.

The stiffness that a state scales (2026-11-10, SPEC 28.6; benchmarks_4g README R12 / X1).  As first written (legacy,
kept: ``stiffness=None`` here, ``RedoxConfig(state_polarizability_stiffness='block')``) Delta_j was built from the
operator's dipole block, Delta_j = P_jj (1 / rho_j - 1).  In an open system P_jj = A_mu,j and the two statements are
one.  In a periodic cell P_jj = A_mu,j + T2_jj, T2_jj the dipole-dipole lattice sum of atom j with its own periodic
images (and a slab's dipole correction): an interaction between different atoms of the crystal, which a charge state
does not change and which depends on the cell.  Scaling it made the in-phase configuration of a replicated cell differ
from the cell's own (measured on Au2/MgO(001), 2x1 against twice 1x1: +76.6, +19.0, -83.9, -84.4 meV; <= 1e-6 meV with
the atom's own stiffness).  The statement is therefore Delta_j = A_mu,j (1 / rho_j - 1) (``stiffness = A_mu``).

Ratios are free-ion values (O in the Watson sphere): an upper bound for an anion confined by its neighbours (Fowler-
Madden).  States not reliably converged in the field (diffuse anions: Na-, Al-, Ag-) keep rho = 1, as does every
element without a reliable frame value.
"""
from __future__ import annotations

import json
import os

import torch
from torch import Tensor

from .band import atom_charges

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ion_polarizabilities.json")
_ALPHA: dict | None = None


def _alphas() -> dict:
    global _ALPHA
    if _ALPHA is None:
        with open(DATA) as fh:
            d = json.load(fh)
        _ALPHA = {(int(v["Z"]), int(v["q"])): float(v["alpha_bohr3"]) for v in d["states"].values() if v["reliable"]}
    return _ALPHA


def ratio_table(species: Tensor, states: Tensor) -> Tensor:
    """``[n, S]``: rho = alpha(Z, s) / alpha(Z, frame), 1 where either is not reliably measured."""
    from .constants import formal_frame
    al = _alphas()
    frames = formal_frame(species).tolist()
    n, S = int(species.numel()), int(states.numel())
    out = torch.ones(n, S, dtype=torch.float64)
    st = [int(round(float(s))) for s in states.tolist()]
    for j, (z, f) in enumerate(zip(species.tolist(), frames)):
        ref = al.get((int(z), int(f)))
        if ref is None:
            continue
        for k, s in enumerate(st):
            a = al.get((int(z), s))
            if a is not None:
                out[j, k] = a / ref
    return out


def _dipole_block(operator, nb: int, j: int) -> Tensor:
    P = getattr(operator, "P", None)
    if P is not None:                                     # the condensed operator: K's moment block is P
        return P[3 * j:3 * j + 3, 3 * j:3 * j + 3]
    K = operator.matrix()
    return K[nb + 3 * j:nb + 3 * j + 3, nb + 3 * j:nb + 3 * j + 3]


def _stiffness_block(elim, nb: int, j: int, stiffness: Tensor | None) -> Tensor:
    """S_j of Delta_j = S_j (1 / rho_j - 1): the atom's own dipole stiffness ``stiffness[j]`` (A_mu,j, SPEC 28.6), or,
    with ``stiffness=None`` (legacy), the operator's dipole block P_jj, which in a periodic cell also holds the atom's
    interaction with its own images."""
    if stiffness is None:
        return _dipole_block(elim.operator, nb, j)
    return stiffness[j].to(elim.y0.dtype).to(elim.y0.device)


def _positive_definite_after(Delta: Tensor, C_SS: Tensor) -> bool:
    """Whether K(n) = K + P_S^T Delta P_S is positive definite (K is).  Haynsworth inertia additivity on
    [[K, P_S^T], [P_S, -Delta^-1]]: In(K(n)) = In(K) + In(-(Delta^-1 + C_SS)) - In(-Delta^-1), so K(n) > 0 iff
    Delta^-1 + C_SS is nonsingular with exactly as many negative eigenvalues as Delta (block diagonal, P_jj > 0:
    3 per softened atom).  Cost O(|S|^3), against O(m^3) for the spectrum of K(n)."""
    with torch.no_grad():
        sym = lambda X: 0.5 * (X + X.transpose(0, 1))                                          # noqa: E731
        negative = int((torch.linalg.eigvalsh(sym(Delta)) < 0.0).sum())
        ev = torch.linalg.eigvalsh(sym(torch.linalg.inv(Delta)) + sym(C_SS))
        tol = 64.0 * torch.finfo(ev.dtype).eps * float(ev.abs().max())
        return bool((ev.abs() > tol).all()) and int((ev < 0.0).sum()) == negative


def _selection(configs: Tensor, elim, frames: Tensor, table: Tensor, states: Tensor):
    n = int(elim.A.shape[0])
    cfg = configs.to(torch.long)
    atom_states = cfg[:, :n]
    first = int(round(float(states[0])))
    rho = table.to(elim.y0.dtype).to(elim.y0.device)[torch.arange(n, device=cfg.device).unsqueeze(0), atom_states - first]
    changed = (atom_states != frames.to(cfg.device).unsqueeze(0)) & (rho != 1.0)
    return rho, changed


def stiffness_defects(configs: Tensor, elim, frames: Tensor, table: Tensor, states: Tensor,
                      stiffness: Tensor | None = None) -> Tensor:
    """``[M]`` bool: True where configuration n's stiffness K(n) is not positive definite (a polarisation
    catastrophe: the bound sector has no minimum, so no configuration energy exists).  Arguments as
    :func:`state_polarizability_corrections` (``stiffness`` included)."""
    nb = int(elim.n_bonds)
    rho, changed = _selection(configs, elim, frames, table, states)
    out = torch.zeros(int(configs.shape[0]), dtype=torch.bool)
    if not bool(changed.any()):
        return out
    atoms = changed.any(0).nonzero().flatten()
    local = {int(j): k for k, j in enumerate(atoms.tolist())}
    cols = (nb + 3 * atoms.unsqueeze(1) + torch.arange(3, device=atoms.device)).reshape(-1)
    unit = torch.zeros(int(elim.y0.shape[0]), int(cols.numel()), dtype=elim.y0.dtype, device=elim.y0.device)
    unit[cols, torch.arange(int(cols.numel()), device=atoms.device)] = 1.0
    C = elim.operator.solve(unit)[cols]
    for c in range(int(configs.shape[0])):
        S = changed[c].nonzero().flatten().tolist()
        if not S:
            continue
        idx = torch.tensor([3 * local[j] + k for j in S for k in range(3)], device=atoms.device)
        Delta = torch.block_diag(*[_stiffness_block(elim, nb, j, stiffness) * (1.0 / rho[c, j] - 1.0) for j in S])
        out[c] = not _positive_definite_after(Delta, C[idx][:, idx])
    return out


def state_polarizability_corrections(configs: Tensor, elim, frames: Tensor, table: Tensor, states: Tensor,
                                     omega: Tensor | None = None, induction=None,
                                     stiffness: Tensor | None = None) -> Tensor:
    """``[M]`` dE (eV) of each configuration ``configs [M, U]`` (the first n units are the atoms; a band unit, if
    any, maps through ``omega [U, n]`` and keeps its stiffness), ``table [n, S]`` the ratios, ``frames [n]``.
    ``induction = (N, Q [n, S])`` with SPEC 26 on: N = K^-1 V of the carriers' correction columns (a dense
    ``[m, n S]`` tensor, or :class:`trace_redox.induction.InductionColumns`, which gives its rows), so the
    moments are those of the full source, y(n) = y0 - K^-1 A^T q - sum_u Q_u(s_u) N[:, u S + s_u].
    ``stiffness [n, 3, 3]``: each atom's own dipole stiffness A_mu (the assembly's ``params.A_mu``), the quantity a
    state scales (SPEC 28.6); ``None`` is the legacy statement on the operator's block P_jj (the same in an open
    system, not extensive in a periodic one)."""
    n, nb = int(elim.A.shape[0]), int(elim.n_bonds)
    dtype = elim.y0.dtype
    cfg = configs.to(torch.long)
    M = int(cfg.shape[0])
    out = torch.zeros(M, dtype=dtype, device=elim.y0.device)
    atom_states = cfg[:, :n]
    first = int(round(float(states[0])))
    table = table.to(dtype).to(elim.y0.device)
    rho = table[torch.arange(n, device=cfg.device).unsqueeze(0), atom_states - first]          # [M, n]
    changed = (atom_states != frames.to(cfg.device).unsqueeze(0)) & (rho != 1.0)
    if not bool(changed.any()):
        return out
    atoms = changed.any(0).nonzero().flatten()                                                 # [a]
    local = {int(j): k for k, j in enumerate(atoms.tolist())}
    cols = (nb + 3 * atoms.unsqueeze(1) + torch.arange(3, device=cfg.device)).reshape(-1)        # [3a]
    m = int(elim.y0.shape[0])
    unit = torch.zeros(m, int(cols.numel()), dtype=dtype, device=elim.y0.device)
    unit[cols, torch.arange(int(cols.numel()), device=cfg.device)] = 1.0
    C = elim.operator.solve(unit)[cols]                                                        # [3a, 3a]
    Mrows = elim.operator.solve(elim.A.transpose(0, 1))[cols]                                  # [3a, n]
    y0 = elim.y0[cols]
    blocks = {int(j): _stiffness_block(elim, nb, int(j), stiffness) for j in atoms.tolist()}
    q = atom_charges(cfg, omega, dtype)                                                        # [M, n]
    if induction is not None:
        N, Qc = induction
        S_st = int(states.numel())
        Nrows = N.rows(cols) if hasattr(N, "rows") else N[cols]                       # [3a, n S]
        k_idx = atom_states - first                                                             # [M, n]
        flat = torch.arange(n, device=cfg.device).unsqueeze(0) * S_st + k_idx                  # [M, n]
        Qsel = Qc.to(dtype)[torch.arange(n, device=cfg.device).unsqueeze(0), k_idx]            # [M, n]
    for c in range(M):
        S = changed[c].nonzero().flatten().tolist()
        if not S:
            continue
        idx = torch.tensor([3 * local[j] + k for j in S for k in range(3)], device=cfg.device)
        y_all = y0 - Mrows @ q[c]
        if induction is not None:
            y_all = y_all - Nrows[:, flat[c]] @ Qsel[c]
        y = y_all[idx]
        Delta = torch.block_diag(*[blocks[j] * (1.0 / rho[c, j] - 1.0) for j in S])
        if not _positive_definite_after(Delta, C[idx][:, idx]):
            raise ValueError(
                f"state_polarizability: configuration {cfg[c].tolist()} gives a stiffness K(n) that is not positive "
                f"definite (atoms {S} at ratios {[round(float(rho[c, j]), 3) for j in S]}): a polarisation "
                "catastrophe, so its bound sector has no minimum and no energy (SPEC 28)")
        eye = torch.eye(int(idx.numel()), dtype=dtype, device=elim.y0.device)
        out = out.index_put((torch.tensor([c], device=cfg.device),),
                            (0.5 * y @ (Delta @ torch.linalg.solve(eye + C[idx][:, idx] @ Delta, y))).reshape(1))
    return out


def state_polarizability_response(configs: Tensor, elim, frames: Tensor, table: Tensor, states: Tensor, D: Tensor,
                                  omega: Tensor | None = None, induction=None,
                                  stiffness: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """SPEC 30 with SPEC 28: each configuration's change of bound dipole and bound polarisability from its own
    stiffness K(n) = K + P_S^T Delta_S P_S, by Woodbury (G = K^-1 D, ``D [m, 3]`` the dipole derivative):

        y(n) = K(n)^-1 K y_ref(n) = y_ref(n) - K^-1 P_S^T Delta_S (I + C_SS Delta_S)^-1 y_S(n),
        dM(n)     = D^T (y(n) - y_ref(n))      = -G_S^T Delta_S (I + C_SS Delta_S)^-1 y_S(n),
        dalpha(n) = D^T (K(n)^-1 - K^-1) D     = -G_S^T Delta_S (I + C_SS Delta_S)^-1 G_S,

    so that E(n; F) = E(n; 0) - F . (M_ref(n) + dM(n)) - 1/2 F^T (alpha_b + dalpha(n)) F exactly.  Arguments and the
    selection S and ``stiffness`` as :func:`state_polarizability_corrections`.  Returns ``(dM [M, 3], dalpha [M, 3, 3])``,
    zero for a configuration with S empty."""
    n, nb = int(elim.A.shape[0]), int(elim.n_bonds)
    dtype, device = elim.y0.dtype, elim.y0.device
    cfg = configs.to(torch.long)
    M = int(cfg.shape[0])
    zero_m, zero_a = torch.zeros(3, dtype=dtype, device=device), torch.zeros(3, 3, dtype=dtype, device=device)
    atom_states = cfg[:, :n]
    first = int(round(float(states[0])))
    table = table.to(dtype).to(device)
    rho = table[torch.arange(n, device=cfg.device).unsqueeze(0), atom_states - first]          # [M, n]
    changed = (atom_states != frames.to(cfg.device).unsqueeze(0)) & (rho != 1.0)
    if not bool(changed.any()):
        return zero_m.expand(M, 3).clone(), zero_a.expand(M, 3, 3).clone()
    atoms = changed.any(0).nonzero().flatten()
    local = {int(j): k for k, j in enumerate(atoms.tolist())}
    cols = (nb + 3 * atoms.unsqueeze(1) + torch.arange(3, device=cfg.device)).reshape(-1)        # [3a]
    m = int(elim.y0.shape[0])
    unit = torch.zeros(m, int(cols.numel()), dtype=dtype, device=device)
    unit[cols, torch.arange(int(cols.numel()), device=cfg.device)] = 1.0
    C = elim.operator.solve(unit)[cols]                                                        # [3a, 3a]
    Mrows = elim.operator.solve(elim.A.transpose(0, 1))[cols]                                  # [3a, n]
    G = elim.operator.solve(D.to(dtype))[cols]                                                 # [3a, 3]
    y0 = elim.y0[cols]
    blocks = {int(j): _stiffness_block(elim, nb, int(j), stiffness) for j in atoms.tolist()}
    q = atom_charges(cfg, omega, dtype)                                                        # [M, n]
    if induction is not None:
        N, Qc = induction
        S_st = int(states.numel())
        Nrows = N.rows(cols) if hasattr(N, "rows") else N[cols]
        k_idx = atom_states - first
        flat = torch.arange(n, device=cfg.device).unsqueeze(0) * S_st + k_idx
        Qsel = Qc.to(dtype)[torch.arange(n, device=cfg.device).unsqueeze(0), k_idx]
    dM, dalpha = [], []
    for c in range(M):
        S = changed[c].nonzero().flatten().tolist()
        if not S:
            dM.append(zero_m)
            dalpha.append(zero_a)
            continue
        idx = torch.tensor([3 * local[j] + k for j in S for k in range(3)], device=cfg.device)
        y_all = y0 - Mrows @ q[c]
        if induction is not None:
            y_all = y_all - Nrows[:, flat[c]] @ Qsel[c]
        Delta = torch.block_diag(*[blocks[j] * (1.0 / rho[c, j] - 1.0) for j in S])
        eye = torch.eye(int(idx.numel()), dtype=dtype, device=device)
        G_S = G[idx]
        Z = Delta @ torch.linalg.solve(eye + C[idx][:, idx] @ Delta, torch.cat([y_all[idx].unsqueeze(1), G_S], 1))
        dM.append(-(G_S.transpose(0, 1) @ Z[:, 0]))
        a = -(G_S.transpose(0, 1) @ Z[:, 1:])
        dalpha.append(0.5 * (a + a.transpose(0, 1)))
    return torch.stack(dM), torch.stack(dalpha)
