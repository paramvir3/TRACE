"""The canonical free energy of a diabatic Hamiltonian over charge configurations (audit step C2).

The configurations N of trace_redox (integer free charges, SPEC 15) are the diabatic basis: each has
an exact polarisable energy E(N) from the eliminated bound sector.  Coupling configurations that
differ by one moved electron, H = diag(E_N) + T, gives the electronic free energy

    F(H) = -kT ln Tr exp(-H / kT) = lambda_min - kT ln sum_i exp(-(lambda_i - lambda_min) / kT).

With T = 0 it is the existing exact ensemble, F = -kT ln sum_N exp(-E_N / kT).  With two configurations
it is the two-state (Marcus-Hush) Hamiltonian, so the Robin-Day classes follow from the learned
energies and couplings rather than being imposed: class II (localised, a double well) iff
lambda > 2|t| for the symmetric pair (ESTABLISHED; tests/redox/test_redox_diabatic.py).

Derivatives, exact at any spectrum, degenerate included (NEW-TO-TRACE as a numerical construction):

* dF/dH = rho = exp(-H/kT) / Z, the canonical density matrix (Hellmann-Feynman forces
  -dF/dx = -Tr[rho dH/dx]).
* d rho, in the eigenbasis H = V diag(lambda) V^T, by Daleckii-Krein for f = exp(-lambda/kT) and the
  normalisation:

      (V^T d rho V)_ij = Gamma_ij (V^T dH V)_ij + (1/kT) p_i delta_ij sum_k p_k (V^T dH V)_kk,
      Gamma_ij = (p_i - p_j) / (lambda_i - lambda_j),   Gamma_ii = -p_i / kT.

  Gamma is evaluated without cancellation as -(p_lo / kT) psi(|lambda_i - lambda_j| / kT), with p_lo
  the larger weight of the pair and psi(y) = -expm1(-y) / y, psi(0) = 1.  Autograd through
  ``torch.linalg.eigh`` instead forms eigenvector derivatives with 1 / (lambda_i - lambda_j), which is
  0/0 at a degenerate spectrum (the NaN class fixed in free_solver, Round 11).

The map is self-adjoint, so the same formula is its vector-Jacobian product.  Third derivatives are not
implemented and are refused (force training needs the second).
"""
from __future__ import annotations

import dataclasses
import math

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


def _spectrum(H: Tensor, kT: float):
    """``(lambda, V, p, F)`` of the symmetrised ``H``: eigenpairs, Boltzmann weights, free energy."""
    evals, evecs = torch.linalg.eigh(0.5 * (H + H.transpose(-1, -2)))
    low = evals.min()
    weights = torch.exp(-(evals - low) / kT)
    total = weights.sum()
    return evals, evecs, weights / total, low - kT * torch.log(total)


def _divided_differences(evals: Tensor, p: Tensor, kT: float) -> Tensor:
    """``Gamma_ij = (p_i - p_j) / (lambda_i - lambda_j)`` (``-p_i / kT`` on equal eigenvalues), cancellation-free."""
    y = (evals.unsqueeze(1) - evals.unsqueeze(0)).abs() / kT
    positive = y > 0
    psi = torch.where(positive, -torch.expm1(-y) / torch.where(positive, y, torch.ones_like(y)), torch.ones_like(y))
    p_lo = torch.maximum(p.unsqueeze(1), p.unsqueeze(0))       # the lower eigenvalue's (larger) weight
    return -(p_lo / kT) * psi


class _DensityMatrix(torch.autograd.Function):
    """``rho(H) = exp(-H/kT) / Z`` with the exact (Daleckii-Krein) derivative."""

    @staticmethod
    def forward(ctx, H: Tensor, kT: float) -> Tensor:
        evals, evecs, p, _ = _spectrum(H, kT)
        ctx.kT = kT
        ctx.save_for_backward(evals, evecs, p)
        return evecs @ torch.diag(p) @ evecs.transpose(-1, -2)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad: Tensor):
        evals, evecs, p = ctx.saved_tensors
        kT = ctx.kT
        g = evecs.transpose(-1, -2) @ (0.5 * (grad + grad.transpose(-1, -2))) @ evecs
        inner = _divided_differences(evals, p, kT) * g
        inner = inner + torch.diag(p) * ((p * torch.diagonal(g)).sum() / kT)
        return evecs @ inner @ evecs.transpose(-1, -2), None


class _FreeEnergy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H: Tensor, kT: float) -> Tensor:
        ctx.kT = kT
        ctx.save_for_backward(H)
        return _spectrum(H, kT)[3]

    @staticmethod
    def backward(ctx, grad: Tensor):
        (H,) = ctx.saved_tensors
        return grad * _DensityMatrix.apply(H, ctx.kT), None


def canonical_free_energy(H: Tensor, kT: float) -> Tensor:
    """``F = -kT ln Tr exp(-H / kT)`` of a real symmetric ``[M, M]`` diabatic Hamiltonian (eV).

    Differentiable twice, exactly, at any spectrum: ``dF/dH = rho`` and its Daleckii-Krein derivative
    (module docstring).  ``H`` is symmetrised; ``kT > 0`` is the electronic temperature in eV.
    """
    if not kT > 0.0:
        raise ValueError(f"kT must be > 0, got {kT}")
    if H.dim() != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be a square matrix, got shape {tuple(H.shape)}")
    return _FreeEnergy.apply(H, float(kT))


def density_matrix(H: Tensor, kT: float) -> Tensor:
    """``rho = exp(-H / kT) / Tr exp(-H / kT)``: populations on its diagonal, coherences off it."""
    if not kT > 0.0:
        raise ValueError(f"kT must be > 0, got {kT}")
    return _DensityMatrix.apply(H, float(kT))


def one_electron_moves(configurations: Tensor, max_charge: Tensor):
    """Every ordered pair of configurations that differ by one electron moved between two units.

    ``configurations`` ``[M, U]``: integer free charges, unit ``u`` in ``-1 .. max_charge[u]``.
    Unit ``u`` holds ``K_u + 1`` local spin-less orbitals, one per charge step, ordered by
    (unit, local index); charge ``n_u`` fills the lowest ``o_u = K_u - n_u`` (SPEC 17.2).

    Returns ``(rows i, cols j, donor a, acceptor b, sign)`` for every pair with
    ``n_j = n_i + e_a - e_b``: the electron leaves ``a``'s top occupied orbital ``p`` and enters
    ``b``'s lowest empty orbital ``q``, and ``sign = <n_j| c_q^dag c_p |n_i>`` by Jordan-Wigner,

        sign = (-1)^(N_<(p) + N'_<(q)),   N_<(p) = sum_{u<a} o_u + o_a - 1,
                                         N'_<(q) = sum_{u<b} o_u - [a < b] + o_b,

    the occupied orbitals below ``p`` in ``n_i`` and below ``q`` once ``p`` is emptied.  The one-body
    Slater-Condon rule (ESTABLISHED); checked against explicit Fock-space operators
    (tests/redox/test_redox_diabatic.py).  Every other pair has no one-body matrix element.
    """
    n = configurations.to(torch.int64)
    K = max_charge.to(torch.int64).to(n.device)
    M, U = n.shape
    small = n.to(torch.int8)
    diff = small.unsqueeze(0) - small.unsqueeze(1)                    # [i, j, U] = n_j - n_i
    plus, minus = diff == 1, diff == -1
    single = (plus.sum(-1) == 1) & (minus.sum(-1) == 1) & ((diff == 0).sum(-1) == U - 2)
    rows, cols = single.nonzero(as_tuple=True)
    donor = plus[rows, cols].to(torch.int64).argmax(-1)
    acceptor = minus[rows, cols].to(torch.int64).argmax(-1)
    occupied = K.unsqueeze(0) - n                                     # [M, U]
    below = torch.cumsum(occupied, dim=1) - occupied                  # occupied orbitals in units before u
    pick = torch.arange(rows.shape[0], device=n.device)
    o = occupied[rows]
    n_p = below[rows, donor] + o[pick, donor] - 1
    n_q = below[rows, acceptor] - (donor < acceptor).to(torch.int64) + o[pick, acceptor]
    sign = 1.0 - 2.0 * ((n_p + n_q) % 2).to(torch.float64)
    return rows, cols, donor, acceptor, sign


#: Single-move descent steps before the search refuses (each step strictly lowers the exact energy).
_DESCENT_LIMIT = 100_000


class ConfigurationSearchError(RuntimeError):
    """The configuration window holds more configurations than the cap (refused, never truncated)."""


def configuration_energies(configurations: Tensor, e_hat: Tensor, states: Tensor, b: Tensor, W: Tensor,
                           pairs=None) -> Tensor:
    """``E(n) = sum_u e_hat[u, n_u] + b^T n + 1/2 n^T W n`` for integer configurations ``[M, U]`` -> ``[M]``.

    The closed form of a fixed molecule in ``fragments.configuration_energy``: the integer ladder of
    every unit, the bound sector's potential ``b`` and the screened operator ``W`` (bound sector
    equilibrated to this configuration exactly).  For integer charges the on-site self term that
    ``e_hat`` subtracts cancels against ``W``'s diagonal, so no positive definiteness is needed (SPEC 17.3).
    """
    index = torch.round(configurations.to(W.dtype) - states[0]).to(torch.long)          # [M, U]
    n = configurations.to(W.dtype)
    onsite = e_hat.unsqueeze(0).expand(n.shape[0], -1, -1).gather(2, index.unsqueeze(-1)).squeeze(-1)
    energy = onsite.sum(1) + n @ b + 0.5 * ((n @ W) * n).sum(1)
    if pairs is not None:
        # two-body exclusion between carrier units (SPEC 18.5): Xi_p[state of i_p, state of j_p]
        pi, pj, Xi = pairs
        if int(pi.numel()):
            p = torch.arange(int(pi.numel()), device=W.device)
            energy = energy + Xi[p.unsqueeze(0), index[:, pi], index[:, pj]].sum(1)
    return energy


def _lower(min_charge: Tensor | None, like: Tensor) -> Tensor:
    return torch.full_like(like, -1) if min_charge is None else min_charge.to(like.dtype).to(like.device)


def _valid_moves(n: Tensor, K: Tensor, min_charge: Tensor | None = None, k: int = 1) -> Tensor:
    """``[U, U]`` mask of the moves of ``k`` electrons a -> b that keep every unit in ``[L_u, K_u]``
    (checked exactly, never inferred from an energy estimate; ``L_u = -1`` unless given)."""
    return ((n + k <= K).unsqueeze(1) & (n - k >= _lower(min_charge, n)).unsqueeze(0)).fill_diagonal_(False)


def admissible(configs: Tensor, units: Tensor, frames: Tensor) -> Tensor:
    """``[M]`` True where every covalent unit's deviations ``d = n - f`` are sign-definite (all >= 0 or all <= 0):
    free carriers of one sign per unit; an electron-hole pair inside a unit is a bound-sector fluctuation
    (trace_redox SPEC 19; theory note 2026-10-16 §3)."""
    configs = configs.to(torch.long)
    d = configs - frames.to(torch.long).to(configs.device).unsqueeze(0)
    index = units.to(torch.long).to(configs.device).unsqueeze(0).expand_as(d)
    count = int(units.max()) + 1 if units.numel() else 0
    pos = torch.zeros(d.shape[0], count, dtype=torch.long, device=d.device).scatter_add(1, index, (d > 0).long())
    neg = torch.zeros(d.shape[0], count, dtype=torch.long, device=d.device).scatter_add(1, index, (d < 0).long())
    return ~((pos > 0) & (neg > 0)).any(dim=1)


def _admissible_moves(n: Tensor, units: Tensor, frames: Tensor, k: int) -> Tensor:
    """``[U, U]`` mask of the moves a -> b (n_a + k, n_b - k) that leave an admissible configuration admissible,
    exactly: only the units of a and b change, and their positive / negative deviation counts are updated."""
    units = units.to(torch.long).to(n.device)
    d = n.to(torch.long) - frames.to(torch.long).to(n.device)
    count = int(units.max()) + 1
    pos = torch.zeros(count, dtype=torch.long, device=n.device).scatter_add(0, units, (d > 0).long())
    neg = torch.zeros(count, dtype=torch.long, device=n.device).scatter_add(0, units, (d < 0).long())
    da, dc = d + k, d - k
    pa, na = (da > 0).long() - (d > 0).long(), (da < 0).long() - (d < 0).long()       # change if a gains +k
    pc, nc = (dc > 0).long() - (d > 0).long(), (dc < 0).long() - (d < 0).long()       # change if b gets -k
    ua, uc = units.unsqueeze(1), units.unsqueeze(0)                                     # [U, 1], [1, U]
    same = ua == uc
    pos_a = pos[ua] + pa.unsqueeze(1) + same * pc.unsqueeze(0)
    neg_a = neg[ua] + na.unsqueeze(1) + same * nc.unsqueeze(0)
    pos_c = pos[uc] + pc.unsqueeze(0) + same * pa.unsqueeze(1)
    neg_c = neg[uc] + nc.unsqueeze(0) + same * na.unsqueeze(1)
    return ~(((pos_a > 0) & (neg_a > 0)) | ((pos_c > 0) & (neg_c > 0)))


def _pair_move_energies(n: Tensor, pairs, first: int, last: int, k: int, U: int) -> Tensor:
    """``[U, U]``: the exact change of the two-body term for the move a -> b (n_a + k, n_b - k): every pair of a
    changes with n_a, every pair of b with n_b, and the pair (a, b) itself, if present, is corrected for both."""
    pi, pj, Xi = pairs
    dtype, device = Xi.dtype, Xi.device
    if not int(pi.numel()):
        return torch.zeros(U, U, dtype=dtype, device=device)
    idx = (n - first).to(torch.long)
    p = torch.arange(int(pi.numel()), device=device)
    i0, j0 = idx[pi], idx[pj]
    up_i, dn_i = (i0 + k).clamp(0, last), (i0 - k).clamp(0, last)
    up_j, dn_j = (j0 + k).clamp(0, last), (j0 - k).clamp(0, last)
    base = Xi[p, i0, j0]
    up = torch.zeros(U, dtype=dtype, device=device)
    down = torch.zeros(U, dtype=dtype, device=device)
    up = up.index_add(0, pi, Xi[p, up_i, j0] - base).index_add(0, pj, Xi[p, i0, up_j] - base)
    down = down.index_add(0, pi, Xi[p, dn_i, j0] - base).index_add(0, pj, Xi[p, i0, dn_j] - base)
    delta = up.unsqueeze(1) + down.unsqueeze(0)
    # the pair (a, b) = (i, j): a goes up, b down; and (a, b) = (j, i)
    corr_ij = Xi[p, up_i, dn_j] - Xi[p, up_i, j0] - Xi[p, i0, dn_j] + base
    corr_ji = Xi[p, dn_i, up_j] - Xi[p, i0, up_j] - Xi[p, dn_i, j0] + base
    delta = delta.index_put((pi, pj), corr_ij, accumulate=True).index_put((pj, pi), corr_ji, accumulate=True)
    return delta


def _move_energies(n: Tensor, e_hat: Tensor, K: Tensor, b: Tensor, W: Tensor,
                   states: Tensor | None = None, min_charge: Tensor | None = None, k: int = 1,
                   pairs=None) -> Tensor:
    """``[U, U]``: E(n + k e_a - k e_b) - E(n) for moving ``k`` electrons a -> b; +inf outside ``[L_u, K_u]``.

        Delta E = [e_a(n_a + k) - e_a(n_a)] + [e_b(n_b - k) - e_b(n_b)] + k (f_a - f_b)
                  + k^2 (W_aa + W_bb) / 2 - k^2 W_ab,          f = b + W n,

    exact for the quadratic E(n) (SPEC 17.3).  ``k > 1`` moves an ion's whole charge at once (an ion
    hop, e.g. Na- between sites), which single moves reach only through high intermediates.
    ``e_hat`` columns are the states ``states`` (default -1, 0, 1, ...): column ``n - states[0]``."""
    U = n.shape[0]
    first = -1 if states is None else int(round(float(states[0])))
    index = n - first
    last = e_hat.shape[1] - 1
    inf = torch.full((U,), float("inf"), dtype=W.dtype, device=W.device)
    rows = torch.arange(U, device=W.device)
    up = torch.where(n + k <= K, e_hat[rows, (index + k).clamp(0, last)] - e_hat[rows, index], inf)
    down = torch.where(n - k >= _lower(min_charge, n), e_hat[rows, (index - k).clamp(0, last)] - e_hat[rows, index], inf)
    field = b + W @ n.to(W.dtype)
    diag = torch.diagonal(W)
    delta = ((up + k * field).unsqueeze(1) + (down - k * field).unsqueeze(0)
             + (0.5 * k * k) * (diag.unsqueeze(1) + diag.unsqueeze(0)) - (k * k) * W)
    if pairs is not None:
        delta = delta + _pair_move_energies(n, pairs, first, last, k, U)
    return delta.fill_diagonal_(float("inf"))


def search_configurations(e_hat: Tensor, states: Tensor, b: Tensor, W: Tensor, max_charge: Tensor, Q: int, *,
                          window: float, cap: int, start: Tensor | None = None, min_charge: Tensor | None = None,
                          units: Tensor | None = None, frames: Tensor | None = None, pairs=None,
                          correction=None, bound: Tensor | None = None):
    """The integer configurations within ``window`` eV of the lowest one found, ``(configs [M, U], E [M])``.

    ``correction`` (SPEC 32): a detached ``[M, U] -> [M]`` diagonal term beyond the quadratic form, added to every
    energy the search decides on (+inf excludes a configuration); ``bound [U]`` bounds how much one move touching a
    unit can change it, so the move pre-screen stays a superset of the window.

    1. A configuration of total ``Q``: from ``start`` (default all neutral), charges added or removed
       one at a time where it costs least.
    2. Steepest single-move descent to a local minimum.
    3. Flood fill through one-electron moves, keeping every configuration within ``window`` of the
       lowest energy met so far (the window re-centres when a lower one appears).

    Exact under its stated condition: the configurations within the window are connected through
    one-electron moves that stay inside it (tests/redox/test_redox_diabatic.py checks this against
    brute force).  A general quadratic integer program has no cheap exact answer.  More than ``cap``
    configurations is refused, never truncated.  The search is detached: a selection, like SPEC 15.3.

    With ``units`` and ``frames`` (SPEC 19) only admissible configurations are visited (:func:`admissible`); the
    statement above then holds on the admissible set.
    """
    with torch.no_grad():
        e_hat, b, W = e_hat.detach(), b.detach(), W.detach()
        U = int(b.shape[0])
        K = max_charge.to(torch.long).to(b.device)
        L = _lower(min_charge, K)
        first = int(round(float(states[0])))
        if not (int(L.sum()) <= Q <= int(K.sum())):
            raise ConfigurationSearchError(f"total charge {Q} is outside the capacity [{int(L.sum())}, {int(K.sum())}]")
        if start is None:
            n = torch.minimum(torch.maximum(torch.zeros_like(K), L), K)
        else:
            n = start.to(torch.long).clone()
        rows = torch.arange(U, device=b.device)

        def energy(m: Tensor) -> float:
            value = float(configuration_energies(m.unsqueeze(0), e_hat, states, b, W, pairs=pairs)[0])
            return value if correction is None else value + float(correction(m.unsqueeze(0))[0])

        def margin(a: int, c: int) -> float:
            return 0.0 if bound is None else float(bound[a] + bound[c])

        restricted = units is not None
        if restricted:
            if frames is None:
                raise ValueError("units need frames")
            units, frames = units.to(torch.long).to(b.device), frames.to(torch.long).to(b.device)
            if not bool(admissible(n.unsqueeze(0), units, frames)[0]):
                raise ConfigurationSearchError("the start configuration is not admissible for the covalent units")

        def moves_ok(m: Tensor, k: int) -> Tensor:
            ok = _valid_moves(m, K, L, k)
            return ok & _admissible_moves(m, units, frames, k) if restricted else ok

        while int(n.sum()) != Q:
            step = 1 if int(n.sum()) < Q else -1
            allowed = (n < K) if step > 0 else (n > L)
            if restricted:
                trial = n.unsqueeze(0) + step * torch.eye(U, dtype=n.dtype, device=n.device)
                allowed = allowed & admissible(trial, units, frames)
            index = n - first
            change = e_hat[rows, (index + step).clamp(0, e_hat.shape[1] - 1)] - e_hat[rows, index]
            field = b + W @ n.to(W.dtype)
            cost = change + step * field + 0.5 * torch.diagonal(W)
            cost = torch.where(allowed, cost, torch.full_like(cost, float("inf")))
            if not bool(allowed.any()):
                raise ConfigurationSearchError(f"no unit can take total charge {Q}: capacity or admissibility")
            n[int(torch.argmin(cost))] += step
        # Steepest descent, a move accepted only if the EXACT energy drops: the energy strictly
        # decreases over a finite set, so this terminates whatever the move estimates do.
        current = energy(n)
        k_max = int((K - L).max())
        for _ in range(_DESCENT_LIMIT):
            best_k, best_delta = 1, None
            for k in range(1, k_max + 1):
                delta = _move_energies(n, e_hat, K, b, W, states=states, min_charge=L, k=k, pairs=pairs)
                delta = torch.where(moves_ok(n, k), delta, torch.full_like(delta, float("inf")))
                if best_delta is None or float(delta.min()) < float(best_delta.min()):
                    best_k, best_delta = k, delta
            if not float(best_delta.min()) < -1e-12:
                break
            a, c = divmod(int(torch.argmin(best_delta)), U)
            trial = n.clone()
            trial[a] += best_k
            trial[c] -= best_k
            value = energy(trial)
            if not value < current - 1e-12 * (1.0 + abs(current)):
                break
            n, current = trial, value
        else:
            raise ConfigurationSearchError(f"single-move descent did not settle in {_DESCENT_LIMIT} moves")
        found = {tuple(n.tolist()): current}
        lowest = found[tuple(n.tolist())]
        frontier = [n.clone()]
        while frontier:
            m = frontier.pop()
            base = found[tuple(m.tolist())]
            candidates = []
            for k in range(1, k_max + 1):
                delta = _move_energies(m, e_hat, K, b, W, states=states, min_charge=L, k=k, pairs=pairs)
                if bound is not None:
                    delta = delta - (bound.unsqueeze(1) + bound.unsqueeze(0)).to(delta.dtype)
                for a, c in torch.nonzero(((base + delta) <= lowest + window) & moves_ok(m, k)).tolist():
                    candidates.append((a, c, k))
            fresh = []
            for a, c, k in candidates:
                nxt = m.clone()
                nxt[a] += k
                nxt[c] -= k
                if tuple(nxt.tolist()) not in found:
                    fresh.append(nxt)
            if not fresh:
                continue
            batch = torch.stack(fresh)                                     # one evaluation per frontier configuration
            values = configuration_energies(batch, e_hat, states, b, W, pairs=pairs)
            if correction is not None:
                values = values + correction(batch)
            for nxt, value in zip(fresh, values.tolist()):
                key = tuple(nxt.tolist())
                if key in found or value > lowest + window:
                    continue
                found[key] = value
                lowest = min(lowest, value)
                if len(found) > cap:
                    raise ConfigurationSearchError(
                        f"more than cap = {cap} configurations within {window} eV: charge is band-like here "
                        "(many carriers fluctuating together); raise the cap or use the mean-field ensemble"
                    )
                frontier.append(nxt)
        keys = [k for k, v in found.items() if v <= lowest + window]
        configs = torch.tensor(keys, dtype=torch.long, device=b.device)
        E = configuration_energies(configs, e_hat, states, b, W, pairs=pairs)
        return configs, (E if correction is None else E + correction(configs))


def formal_ladder(species: Tensor, max_charge: int):
    """``(states, E_Z [U, S], L [U], K [U])``: exact isolated-ion energies relative to the neutral atom.

    ``E_Z(n) = sum_{k<=n} IE_k`` (NIST successive IEs) for ``n > 0``; ``-EA_1`` for ``n = -1``;
    ``-EA_1 + A_2`` for ``n = -2`` where the second-electron cost ``A_2`` is tabulated
    (``constants.SECOND_ELECTRON_COST``; O only); ``0`` for the neutral atom.  Unit ``u`` spans
    ``L_u = -2`` (if ``A_2`` is tabulated) or ``-1``, to ``K_u = min(max_charge, tabulated IEs)``;
    ``+inf`` outside.  ``states`` runs over ``min L .. max K`` (SPEC 17.5).  No state beyond the data.
    """
    from .constants import ATOMIC_ELECTRON_AFFINITY_EV, ATOMIC_SUCCESSIVE_IONIZATION_EV, SECOND_ELECTRON_COST
    import math as _math

    z = [int(x) for x in species.tolist()]
    lower = [-2 if zz in SECOND_ELECTRON_COST else -1 for zz in z]
    upper = []
    for zz in z:
        series = ATOMIC_SUCCESSIVE_IONIZATION_EV[zz]
        upper.append(min(int(max_charge), sum(1 for x in series if _math.isfinite(x))))
    states = torch.arange(min(lower), max(upper) + 1, dtype=torch.float64)
    E = torch.full((len(z), states.numel()), float("inf"), dtype=torch.float64)
    for u, zz in enumerate(z):
        for c, n in enumerate(int(x) for x in states.tolist()):
            if n < lower[u] or n > upper[u]:
                continue
            if n > 0:
                E[u, c] = float(sum(ATOMIC_SUCCESSIVE_IONIZATION_EV[zz][k] for k in range(n)))
            elif n == 0:
                E[u, c] = 0.0
            elif n == -1:
                E[u, c] = -float(ATOMIC_ELECTRON_AFFINITY_EV[zz])
            else:
                E[u, c] = -float(ATOMIC_ELECTRON_AFFINITY_EV[zz]) + float(SECOND_ELECTRON_COST[zz])
    device = species.device
    return states.to(device), E.to(device), torch.tensor(lower, device=device), torch.tensor(upper, device=device)


@dataclasses.dataclass(frozen=True)
class StepCouplings:
    """Couplings resolved by charge step for the atom pairs of ``mask`` (trace_redox SPEC 40.6 ix, cured 2026-11-14).

    The matrix ``T`` handed to the Hamiltonian holds, for a masked pair, the coupling of its s moves: an s electron
    leaves the donor (charge -1 or 0 before the move) and enters the acceptor's s shell (charge 0 or +1 before).
    Every other one-electron move of that pair takes ``other[donor, acceptor]`` (the learned coupling).  Pairs outside
    the mask are untouched.  Both tensors are ``[n, n]`` over the atoms; units beyond them (a band unit) are unmasked."""
    mask: Tensor
    other: Tensor


def _step_tables(steps: StepCouplings, T: Tensor):
    """(mask, other) padded to the shape of ``T`` (units appended after the atoms are not masked)."""
    U, n = int(T.shape[0]), int(steps.mask.shape[0])
    mask, other = steps.mask.to(T.device), steps.other.to(T.dtype).to(T.device)
    if U > n:
        mask = torch.zeros(U, U, dtype=torch.bool, device=T.device).index_put(
            tuple(torch.meshgrid(torch.arange(n, device=T.device), torch.arange(n, device=T.device), indexing="ij")), mask)
        other = torch.zeros(U, U, dtype=T.dtype, device=T.device).index_put(
            tuple(torch.meshgrid(torch.arange(n, device=T.device), torch.arange(n, device=T.device), indexing="ij")), other)
    return mask, other


def resolve_moves(t: Tensor, T: Tensor, steps: StepCouplings | None, configurations: Tensor, rows: Tensor, donor: Tensor,
                  acceptor: Tensor) -> Tensor:
    """``t [moves] = T[donor, acceptor]`` with the moves of a step-resolved pair that are not s moves replaced by the
    pair's other coupling.  The rule is symmetric under reversing a move (the reverse's donor is the acceptor after it)."""
    if steps is None:
        return t
    mask, other = _step_tables(steps, T)
    n = configurations.to(torch.long)
    q_donor, q_acceptor = n[rows, donor], n[rows, acceptor]
    s_move = (q_donor >= -1) & (q_donor <= 0) & (q_acceptor >= 0) & (q_acceptor <= 1)
    return torch.where(mask[donor, acceptor] & ~s_move, other[donor, acceptor], t)


def resolve_matrix(T: Tensor, steps: StepCouplings | None, m: Tensor) -> Tensor:
    """``[U, U]``: the coupling of every move donor a -> acceptor c out of the configuration ``m`` (charges)."""
    if steps is None:
        return T
    mask, other = _step_tables(steps, T)
    q = m.to(torch.long).to(T.device)
    q_donor, q_acceptor = q.unsqueeze(1), q.unsqueeze(0)
    s_move = (q_donor >= -1) & (q_donor <= 0) & (q_acceptor >= 0) & (q_acceptor <= 1)
    return torch.where(mask & ~s_move, other, T)


def spin_pairing_factors(configurations: Tensor, rows: Tensor, donor: Tensor, acceptor: Tensor,
                         electrons: Tensor) -> Tensor:
    """SPEC 29: ``f = sqrt 2`` for a move whose donor and acceptor hold electron counts of the same parity before
    it (it pairs two singly occupied frontier orbitals, or unpairs a doubly occupied one onto an empty one), else 1.

    ``electrons [n]``: each atom unit's electron count at charge 0 (Z); a unit beyond them (the band unit) counts
    -charge electrons.  Exact for the lowest-spin states of the two-site model (singlet covalent <-> ionic 2 t =
    sqrt 2 x sqrt 2 t over the two ionic configurations; doublet resonances t); for many singly occupied sites it
    takes each moving pair in its singlet, and high-spin frontier states (O 2p^4 triplet) are outside it."""
    n = configurations.to(torch.long)
    U = int(n.shape[1])
    z = electrons.to(torch.long).to(n.device)
    if U > int(z.numel()):
        z = torch.cat([z, torch.zeros(U - int(z.numel()), dtype=torch.long, device=n.device)])
    N = z.unsqueeze(0) - n
    same = (N[rows, donor] % 2) == (N[rows, acceptor] % 2)
    one = torch.ones(int(rows.numel()), dtype=torch.float64, device=n.device)
    return torch.where(same, one * math.sqrt(2.0), one)


def diabatic_hamiltonian(E: Tensor, configurations: Tensor, max_charge: Tensor, T: Tensor,
                         electrons: Tensor | None = None, steps: StepCouplings | None = None) -> Tensor:
    """``H = diag(E) + sum over one-electron moves of sign * f * T[donor, acceptor]`` (SPEC 17.2, 17.6; f the
    SPEC 29 spin-pairing factor when ``electrons`` is given, else 1).

    Only configurations that differ by one electron are coupled (the one-body Slater-Condon rule); the
    sign is the Jordan-Wigner one of :func:`one_electron_moves` (checked against explicit Fock-space
    operators with two carriers, tests/redox/test_redox_diabatic.py)."""
    H = torch.diag(E)
    rows, cols, donor, acceptor, sign = one_electron_moves(configurations, max_charge)
    t = resolve_moves(T[donor, acceptor], T, steps, configurations, rows, donor, acceptor)
    if electrons is not None:
        t = t * spin_pairing_factors(configurations, rows, donor, acceptor, electrons).to(t.dtype)
    return H.index_put((rows, cols), sign.to(E.dtype) * t, accumulate=True)


# ---- the amplitude selection (SPEC 21) ----------------------------------------------------------------------
#
# Importance of a configuration c relative to the low-energy reference (E_ref = -kT ln sum exp(-E / kT), the
# diagonal free energy of the set; every excitation x_c = E_c - E_ref >= 0):
#
#     r_c = max( sqrt(p_c),  max over coupled i of r_i g(T_ic, x_c) ),   p_c = exp(-x_c / kT),
#     g(t, x) = |t| / sqrt(x^2 + t^2)    (the first-order amplitude |t| / x, saturated smoothly at 1),
#
# the leading Rayleigh-Schroedinger amplitude of c along its best path from the thermally weighted reference
# (denominators from the reference energy, as in perturbation theory).  A coupled pair is switched by
# sigma(max(r_i g(T, x_j), r_j g(T, x_i))).  Since g <= 1, r_c >= that pair importance for every neighbour,
# so the set {r > a_off} is closed: sigma = 0 on every pair that leaves it.


def smoothstep(x: Tensor) -> Tensor:
    """The C^2 quintic smoothstep on [0, 1] (clamped): 0 and 1 with zero first and second derivatives at the ends."""
    x = x.clamp(0.0, 1.0)
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


def coupling_amplitude(t: Tensor, x: Tensor) -> Tensor:
    """``g = |t| / sqrt(x^2 + t^2)``: the first-order amplitude |t| / x saturated smoothly at 1 (x = 0, t != 0 gives
    1; t = x = 0 gives 0 with finite gradients)."""
    den2 = x * x + t * t
    ok = den2 > 0
    return torch.where(ok, t.abs() / torch.sqrt(torch.where(ok, den2, torch.ones_like(den2))), torch.zeros_like(den2))


def amplitude_switch(e: Tensor, a_off: float, a_on: float) -> Tensor:
    """``sigma(e) = smoothstep((ln e - ln a_off) / ln(a_on / a_off))``: 0 for e <= a_off, 1 for e >= a_on, C^2 and
    scale-free; e = 0 gives 0 with a finite gradient (SPEC 21)."""
    if not 0.0 < a_off < a_on:
        raise ValueError(f"need 0 < a_off < a_on, got {a_off}, {a_on}")
    pos = e > 0
    ln_e = torch.log(torch.where(pos, e, torch.ones_like(e)))
    return torch.where(pos, smoothstep((ln_e - math.log(a_off)) / math.log(a_on / a_off)), torch.zeros_like(e))


def _importance_tree(E: Tensor, rows: Tensor, cols: Tensor, t: Tensor, kT: float):
    """Detached max-product fixed point of the importance on a set: ``(r [M], parent [M], via [M])``, parent -1 where
    the seed sqrt(p_c) wins, else the configuration (and pair index ``via``) of the best path's last step."""
    with torch.no_grad():
        E, t = E.detach(), t.detach()
        x = E + kT * torch.logsumexp(-E / kT, 0)
        r = torch.exp(-0.5 * x / kT)
        parent = torch.full_like(rows[:1].new_empty(E.shape[0]), -1)
        via = parent.clone()
        g = coupling_amplitude(t, x[cols])
        for _ in range(int(E.shape[0]) + 1):
            cand = r[rows] * g
            best = r.clone().scatter_reduce(0, cols, cand, reduce="amax", include_self=True)
            improved = best > r
            if not bool(improved.any()):
                break
            # the pair that attains each improved maximum (first one, a deterministic choice at exact ties)
            hit = improved[cols] & (cand == best[cols])
            k = torch.nonzero(hit).flatten()
            first = torch.full_like(parent, -1).scatter_reduce(0, cols[k], k, reduce="amin", include_self=False)
            change = improved.nonzero().flatten()
            parent[change] = rows[first[change]]
            via[change] = first[change]
            r = best
        else:
            raise ConfigurationSearchError("the importance did not settle (a cycle with amplitude 1?)")
    return r, parent, via


def configuration_importance(E: Tensor, configurations: Tensor, max_charge: Tensor, T: Tensor, kT: float,
                             electrons: Tensor | None = None, steps: StepCouplings | None = None):
    """``(r [M], pair importance e [P], moves)`` on a configuration set (SPEC 21), differentiable: the argmax tree is
    found detached, then r is rebuilt along it by products (any order of derivatives; kinks only at exact ties of
    two paths).  With ``electrons``, every coupling carries its SPEC 29 factor (as in the Hamiltonian and the
    closure, so the closure lemma holds for the couplings actually used)."""
    rows, cols, donor, acceptor, sign = one_electron_moves(configurations, max_charge)
    t = resolve_moves(T[donor, acceptor], T, steps, configurations, rows, donor, acceptor)
    if electrons is not None:
        t = t * spin_pairing_factors(configurations, rows, donor, acceptor, electrons).to(t.dtype)
    x = E + kT * torch.logsumexp(-E / kT, 0)
    g_to_col = coupling_amplitude(t, x[cols])
    _, parent, via = _importance_tree(E, rows, cols, t, kT)
    r = torch.exp(-0.5 * x / kT)
    # rebuild along the tree, shallowest first (depth by pointer chasing)
    depth = torch.zeros_like(parent)
    with torch.no_grad():
        p = parent.clone()
        while bool((p >= 0).any()):
            depth = depth + (p >= 0).long()
            p = torch.where(p >= 0, parent[p.clamp_min(0)], p)
    for d in range(1, int(depth.max()) + 1 if depth.numel() else 1):
        at = (depth == d).nonzero().flatten()
        r = r.index_put((at,), r[parent[at]] * g_to_col[via[at]])
    e = torch.maximum(r[rows] * g_to_col, r[cols] * coupling_amplitude(t, x[rows]))
    return r, e, (rows, cols, donor, acceptor, sign)


def switched_hamiltonian(E: Tensor, configurations: Tensor, max_charge: Tensor, T: Tensor,
                         a_off: float, a_on: float, kT: float, electrons: Tensor | None = None,
                         steps: StepCouplings | None = None) -> Tensor:
    """``H~ = diag(E) + sum over one-electron moves of sign * sigma(e_ij) * f * T[donor, acceptor]`` (SPEC 21; f the
    SPEC 29 factor with ``electrons``), with e_ij the pair importance of :func:`configuration_importance`.  Symmetric
    because T is (e_ij = e_ji) and f is (the parities after a move are those before its reverse)."""
    _, e, (rows, cols, donor, acceptor, sign) = configuration_importance(E, configurations, max_charge, T, kT,
                                                                         electrons=electrons, steps=steps)
    sigma = amplitude_switch(e, a_off, a_on)
    t = resolve_moves(T[donor, acceptor], T, steps, configurations, rows, donor, acceptor)
    if electrons is not None:
        t = t * spin_pairing_factors(configurations, rows, donor, acceptor, electrons).to(t.dtype)
    return torch.diag(E).index_put((rows, cols), sign.to(E.dtype) * sigma * t, accumulate=True)


def coupled_closure(configs: Tensor, e_hat: Tensor, states: Tensor, b: Tensor, W: Tensor, max_charge: Tensor,
                    T: Tensor, a_off: float, kT: float, *, cap: int, min_charge: Tensor | None = None,
                    units: Tensor | None = None, frames: Tensor | None = None, pairs=None,
                    electrons: Tensor | None = None, correction=None, bound: Tensor | None = None,
                    steps: StepCouplings | None = None) -> Tensor:
    """``configs`` with every configuration of importance above ``a_off`` (SPEC 21): a best-first (max-product
    Dijkstra) search from the set's seeds sqrt(p_c) through one-electron moves, the reference energy that of
    ``configs``.  Exact energies (:func:`_move_energies`), the covalent-unit admissibility when ``units`` are given;
    more than ``cap`` configurations is refused, never truncated.  Detached."""
    import heapq
    with torch.no_grad():
        e_hat, b, W, T = e_hat.detach(), b.detach(), W.detach(), T.detach()
        K = max_charge.to(torch.long).to(b.device)
        L = _lower(min_charge, K)
        restricted = units is not None
        if restricted:
            if frames is None:
                raise ValueError("units need frames")
            units, frames = units.to(torch.long).to(b.device), frames.to(torch.long).to(b.device)
        configs = configs.to(torch.long)
        E0 = configuration_energies(configs, e_hat, states, b, W, pairs=pairs)
        if correction is not None:                        # SPEC 32: the importances see the dressed energies
            E0 = E0 + correction(configs)
        E_ref = float(-kT * torch.logsumexp(-E0 / kT, 0))
        energy = {tuple(c): float(v) for c, v in zip(configs.tolist(), E0.tolist())}
        best = {k: math.exp(-0.5 * (v - E_ref) / kT) for k, v in energy.items()}
        heap = [(-r, k) for k, r in best.items() if r > a_off]
        heapq.heapify(heap)
        done = set()
        while heap:
            neg, key = heapq.heappop(heap)
            if key in done or -neg < best[key]:
                continue
            done.add(key)
            r_m = -neg
            m = torch.tensor(key, dtype=torch.long, device=b.device)
            delta = _move_energies(m, e_hat, K, b, W, states=states, min_charge=L, k=1, pairs=pairs)
            T_here = resolve_matrix(T, steps, m)                 # SPEC 40.6 ix: the coupling of each move out of m
            ok = _valid_moves(m, K, L, 1) & torch.isfinite(delta) & (T_here != 0)
            if restricted:
                ok = ok & _admissible_moves(m, units, frames, 1)
            x = (energy[key] + delta) - E_ref
            T_m = T_here
            if electrons is not None:                     # SPEC 29: the factor of each move out of m
                z = electrons.to(torch.long).to(b.device)
                if int(m.numel()) > int(z.numel()):
                    z = torch.cat([z, torch.zeros(int(m.numel()) - int(z.numel()), dtype=torch.long, device=b.device)])
                parity = (z - m) % 2
                T_m = T_here * torch.where(parity.unsqueeze(1) == parity.unsqueeze(0),
                                           torch.full_like(T, math.sqrt(2.0)), torch.ones_like(T))
            if correction is not None and bound is not None:
                x = x - (bound.unsqueeze(1) + bound.unsqueeze(0)).to(x.dtype)       # a superset pre-screen
            cand = r_m * coupling_amplitude(T_m, torch.where(ok, x, torch.zeros_like(x)))
            found_pairs = torch.nonzero(ok & (cand > a_off)).tolist()
            if correction is not None and found_pairs:
                # the exact dressed energies of the candidates, in one batch
                batch = m.unsqueeze(0).repeat(len(found_pairs), 1)
                for i, (a, c) in enumerate(found_pairs):
                    batch[i, a] += 1
                    batch[i, c] -= 1
                E_batch = configuration_energies(batch, e_hat, states, b, W, pairs=pairs) + correction(batch)
                exact = {}
                for i, (a, c) in enumerate(found_pairs):
                    x_ac = float(E_batch[i]) - E_ref
                    amp = float(coupling_amplitude(T_m[a, c].reshape(1), torch.tensor([x_ac], dtype=T.dtype))[0])
                    exact[(a, c)] = (r_m * amp, float(E_batch[i]))
                found_pairs = [(a, c) for (a, c) in found_pairs if exact[(a, c)][0] > a_off]
            for a, c in found_pairs:
                nxt = m.clone()
                nxt[a] += 1
                nxt[c] -= 1
                k2 = tuple(nxt.tolist())
                value = float(cand[a, c]) if correction is None else exact[(a, c)][0]
                if value <= best.get(k2, 0.0):
                    continue
                if k2 not in energy:
                    energy[k2] = (energy[key] + float(delta[a, c])) if correction is None else exact[(a, c)][1]
                    if len(energy) > cap:
                        raise ConfigurationSearchError(
                            f"the amplitude closure holds more than cap = {cap} configurations: many configurations "
                            "mix strongly with the reference here; raise the cap"
                        )
                best[k2] = value
                heapq.heappush(heap, (-value, k2))
        return torch.tensor(list(energy), dtype=torch.long, device=b.device)


def select_configurations(e_hat: Tensor, states: Tensor, b: Tensor, W: Tensor, max_charge: Tensor, Q: int, *,
                          window: float, cap: int, T: Tensor | None = None, amplitudes=(0.05, 0.2), kT: float = 0.1,
                          start: Tensor | None = None, min_charge: Tensor | None = None,
                          units: Tensor | None = None, frames: Tensor | None = None, pairs=None,
                          search=None, electrons: Tensor | None = None, correction=None,
                          bound: Tensor | None = None, steps: StepCouplings | None = None) -> Tensor:
    """The configurations of the diabatic sector, ``[M, U]``.

    ``T`` None: the window set of :func:`search_configurations` (SPEC 17.3).  Otherwise the window set plus every
    configuration of importance above ``amplitudes[0]`` (:func:`coupled_closure`, SPEC 21).  Should the closure
    meet a configuration below the window set's lowest, the search restarts from it (the window re-centres).
    ``search`` replaces :func:`search_configurations` (same signature; the module passes its own name, so probes
    that wrap ``trace_redox.module.search_configurations`` still see every window search)."""
    search = search_configurations if search is None else search
    for _ in range(16):
        extra = {} if correction is None else {"correction": correction, "bound": bound}
        configs, E = search(e_hat, states, b, W, max_charge, Q, window=window, cap=cap, start=start,
                            min_charge=min_charge, units=units, frames=frames, pairs=pairs, **extra)
        if T is None:
            return configs
        closed = coupled_closure(configs, e_hat, states, b, W, max_charge, T, float(amplitudes[0]), float(kT),
                                 cap=cap, min_charge=min_charge, units=units, frames=frames, pairs=pairs,
                                 electrons=electrons, steps=steps, **extra)
        with torch.no_grad():
            E_closed = configuration_energies(closed, e_hat.detach(), states, b.detach(), W.detach(), pairs=pairs)
            if correction is not None:
                E_closed = E_closed + correction(closed)
        lowest = float(E.min())
        k = int(torch.argmin(E_closed))
        if float(E_closed[k]) >= lowest - 1e-12 * (1.0 + abs(lowest)):
            return closed
        start = closed[k]
    raise ConfigurationSearchError("the amplitude closure kept meeting configurations below the window's lowest")
