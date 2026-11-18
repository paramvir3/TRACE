"""Molecule-level exactness: the two-level certified fragment ensemble (SPEC §12).

The mean field over atoms spreads a charge over distant molecules and lowers
the energy by the molecular charging energy (two waters 30 A apart, Q = +1:
-1.34 eV).  Here every molecule F carries an INTEGER total charge N in a
window, the charge inside a molecule stays delocalised (the existing mean
field at fixed N), and molecules are combined in a convex ensemble:

    G = sum_F sum_N [ p_F(N) A_F(x_F^N / p_F(N)) + kT p_F(N) ln p_F(N) ] + b.c + 1/2 c.S.c
    A_F(y) = sum_{u in F} Phi_u(y_u) + 1/2 y.D_F.y,    S = W - blockdiag(D_F)

with D_F from the block certificate (§12.3):

    K_FG = M_FF^-1/2 M_FG M_GG^-1/2,  M = W^-1,  theta_F = c / (1 + sum_G ||K_FG||),  D_F = theta_F (M_FF)^-1

so S >= (1 - c) W > 0 (convex) and D_F -> W_FF for decoupled molecules (exact).
The total charge is fixed CANONICALLY: every configuration of integer molecular
charges has sum_F N_F = Q (§12.13; a closed system cannot trade charge with a
reservoir).  The computable dual (§12.4, §12.13), for partitions P with weights pi_P:

    E - E_b0 = max_{z, delta}  -1/2 z.Sbar.z + sum_P pi_P A^P(psi^P),   psi^P = -b - Sbar z + delta^P
    A^P(psi) = -kT ln sum_{N in windows, sum_F N_F = Q} exp(-sum_F e_F(N_F; psi_F) / kT)
    e_F(N; psi) = min_{1.y = N} [ A_F(y) - psi.y ]          (free_solver, b -> -psi, W -> D_F, Q -> N)

with sum_P pi_P delta^P = 0 and Sbar = sum_P pi_P S_P.  A^P is evaluated exactly
by convolving the molecules' tilted charge distributions (:func:`_canonical`).

Molecules are the connected components of the chemical-bond graph; a bond in
its switching shell (0 < s_b < 1) makes the partition uncertain, and the
energy averages over the bond-percolation ensemble of partitions (§12.6), a
polynomial in the s_b: no jump when a bond breaks.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import torch
from torch import Tensor

from . import ensemble
from .constants import COVALENT_RADII_A
from .free_solver import FreeChargeNotConvergedError, differentiable_constrained_solution, differentiable_solution
from .sic import certified_removal

#: Additive bond criterion d < R_i + R_j + tol (Open Babel uses tol = 0.45 A;
#: O'Boyle et al., J. Cheminform. 3, 33 (2011)); our C^2 switch brackets it.
BOND_TOLERANCE_ON = 0.25
BOND_TOLERANCE_OFF = 0.60
#: c of the block certificate: S >= (1 - c) W.
CERTIFICATE_C = 0.999
#: Smoothing of ||K_FG||_F (dimensionless; only lowers theta, never invalidates it).
CERTIFICATE_EPS = 1e-3
#: A partition of probability below this is not enumerated, and a connection absent with
#: probability below it is certain: the energy moves by < 1e-30 |A| (< 1e-28 eV here), 14
#: orders below the float64 resolution of any energy, and so do the forces (pi ~ t^3 at a
#: switch end while d ln pi / dx ~ 1/t).
PARTITION_WEIGHT_FLOOR = 1e-30


class FragmentEnsembleError(RuntimeError):
    """The fragment ensemble cannot be built or did not converge."""


# ---------------------------------------------------------------------------
# chemical bonds and the percolation ensemble of partitions
# ---------------------------------------------------------------------------

def bond_switch(r: Tensor, r_on: Tensor, r_off: Tensor) -> Tensor:
    """1 for r <= r_on, 0 for r >= r_off, the C^2 quintic in between."""
    t = ((r - r_on) / (r_off - r_on)).clamp(0.0, 1.0)
    return 1.0 - t ** 3 * (10.0 - 15.0 * t + 6.0 * t * t)


def chemical_bonds(species: Tensor, edge_index: Tensor, edge_len: Tensor, coverage: float,
                   tol_on: float = BOND_TOLERANCE_ON, tol_off: float = BOND_TOLERANCE_OFF):
    """``(i, j, s)``: one copy of every bonded pair image (src < dst), strength ``s`` in (0, 1].

    Refuses when a species pair present needs ``R_i + R_j + tol_off`` beyond the
    radius the neighbour list is guaranteed to cover (a bond with s > 0 could
    then be missing, and the partition would jump when it reappears).
    """
    radii = torch.tensor(COVALENT_RADII_A, dtype=edge_len.dtype, device=edge_len.device)[species]
    if int(radii.numel()) >= 2:
        # the longest bond the switch can see: the two largest radii present (an atom
        # pairs with itself only through an image, which carries no connectivity)
        top2 = torch.topk(radii.detach(), 2).values
        reach = float(top2.sum()) + tol_off
        if reach > coverage + 1e-12:
            raise FragmentEnsembleError(
                f"fragments='bonds' needs neighbour pairs up to R_i + R_j + {tol_off} = {reach:.3f} A "
                f"for the two largest atoms present, beyond the attention_cutoff {coverage} A that "
                "the neighbour list must cover; raise it"
            )
    src, dst = edge_index[0], edge_index[1]
    keep = src < dst                                    # one copy of each image pair; self images dropped
    i, j, r = src[keep], dst[keep], edge_len[keep]
    total = radii[i] + radii[j]
    s = bond_switch(r, total + tol_on, total + tol_off)
    bonded = s > 0
    return i[bonded], j[bonded], s[bonded]


def _components(n: int, edges_i, edges_j) -> list[int]:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(edges_i, edges_j):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return [find(a) for a in range(n)]


def covalent_units(frames: Tensor, i: Tensor, j: Tensor, n: int, full_unit_atoms: int = 0) -> Tensor:
    """``[n]`` unit label per atom (SPEC 19): connected components of the chemical bonds ``(i, j)`` joining two
    open-shell (neutral-frame, f = 0) atoms; every closed-shell frame atom is a unit of its own.

    A component of at most ``full_unit_atoms`` atoms is returned split into singletons: it is solved exactly in
    configuration space (every intra-unit configuration -- the full CI of the minimal basis, as for a 2-site bond),
    and only larger units restrict their configurations to sign-definite carriers."""
    f = frames.tolist()
    keep = [(int(a), int(b)) for a, b in zip(i.tolist(), j.tolist()) if f[int(a)] == 0 and f[int(b)] == 0]
    roots = _components(n, [a for a, _ in keep], [b for _, b in keep])
    size: dict = {}
    for r in roots:
        size[r] = size.get(r, 0) + 1
    relabel: dict = {}
    labels = [relabel.setdefault(("atom", a) if size[r] <= full_unit_atoms else ("unit", r), len(relabel))
              for a, r in enumerate(roots)]
    return torch.tensor(labels, dtype=torch.long, device=frames.device)


def percolation_partitions(n: int, i: Tensor, j: Tensor, s: Tensor, max_uncertain: int):
    """``[(fragments, weight)]``: the distinct partitions of the bond-percolation ensemble.

    A bond is present with probability ``s_b`` independently; bonds with s = 1
    are certain.  Exact reductions before enumerating:

    * an uncertain bond inside a component of the certain bonds never changes
      the partition: dropped;
    * the uncertain bonds between the same two components connect them iff at
      least one is present: one effective bond, s_AB = 1 - prod_b (1 - s_b).

    Then the 2^k patterns of the k effective bonds are enumerated.  ``weight``
    (a tensor, differentiable in the s_b) is the probability of the partition;
    the weights sum to one.  ``fragments`` is a tuple of LongTensors (sorted atom
    indices), ordered by lowest index.
    """
    certain = s >= 1.0
    sure_i, sure_j = i[certain].tolist(), j[certain].tolist()
    while True:
        roots = _components(n, sure_i, sure_j)
        groups: dict[tuple, Tensor] = {}
        for b_ in torch.nonzero(~certain).squeeze(1).tolist():
            ra, rb = roots[int(i[b_])], roots[int(j[b_])]
            if ra == rb:
                continue
            key = (min(ra, rb), max(ra, rb))
            absent = 1.0 - s[b_]
            groups[key] = groups[key] * absent if key in groups else absent
        # a connection absent with probability < PARTITION_WEIGHT_FLOOR (it underflows to 0.0
        # for ~21 parallel bonds near their switch end) is certain; it must not be enumerated
        # as a partition of weight ~0, whose potentials the solver need not converge
        sure = [key for key, absent in groups.items() if float(absent) < PARTITION_WEIGHT_FLOOR]
        if not sure:
            break
        sure_i += [a for a, _ in sure]
        sure_j += [b for _, b in sure]
    keys = list(groups)
    k = len(keys)
    if k > max_uncertain:
        raise FragmentEnsembleError(
            f"{k} independent connections between molecules are in their switching shell; the "
            f"percolation ensemble enumerates 2^k patterns and is limited to k <= {max_uncertain}"
        )
    absent_p = [groups[key] for key in keys]                            # 1 - s_AB, exactly (not 1 - (1 - a))
    weights: dict[tuple, Tensor] = {}
    for bits in itertools.product((0, 1), repeat=k):
        ei = [keys[t][0] for t in range(k) if bits[t]]
        ej = [keys[t][1] for t in range(k) if bits[t]]
        merged = _components(n, ei, ej)
        labels = tuple(merged[roots[a]] for a in range(n))
        w = torch.ones((), dtype=s.dtype, device=s.device)
        for t in range(k):
            w = w * ((1.0 - absent_p[t]) if bits[t] else absent_p[t])
        weights[labels] = weights[labels] + w if labels in weights else w
    out = []
    for labels, w in weights.items():
        if float(w) < PARTITION_WEIGHT_FLOOR:   # e.g. two rare connections absent at once
            continue
        frags = tuple(torch.tensor([a for a in range(n) if labels[a] == r], dtype=torch.long, device=s.device)
                      for r in sorted(set(labels)))
        out.append((frags, w))
    return out


# ---------------------------------------------------------------------------
# the block certificate
# ---------------------------------------------------------------------------

def block_certificate(W: Tensor, M: Tensor, fragments, c: float = CERTIFICATE_C,
                      eps: float = CERTIFICATE_EPS) -> tuple[list[Tensor], Tensor, Tensor]:
    """``(D_F list, theta [m], S)`` with ``W - blockdiag(D_F) >= (1 - c) W`` (SPEC §12.3).

    ``||K_FG||_F^2 = tr(M_FF^-1 M_FG M_GG^-1 M_GF)`` by Cholesky solves only: no
    matrix square root, hence no eigen-decomposition gradient (degenerate
    spectra are common by symmetry).  The smoothed norm sqrt(.. + eps^2) is an
    upper bound, so the certificate stays valid.
    """
    m = len(fragments)
    n = int(W.shape[0])
    inv_blocks = []
    for F in fragments:
        # (M_FF)^-1 = W_FF - W_FR W_RR^-1 W_RF (Schur complement over the rest R),
        # computed from W directly: inverting M = W^-1 back loses ~cond(W) eps, and
        # for one molecule spanning the system this is exactly W_FF.
        rest = torch.tensor([u for u in range(n) if u not in set(F.tolist())], dtype=torch.long, device=W.device)
        block = W[F][:, F]
        if int(rest.numel()) > 0:
            W_FR = W[F][:, rest]
            block = block - W_FR @ torch.linalg.solve(W[rest][:, rest], W_FR.transpose(0, 1))
        inv_blocks.append(0.5 * (block + block.transpose(0, 1)))
    coupling = [torch.zeros((), dtype=W.dtype, device=W.device) for _ in range(m)]
    for a in range(m):
        for b_ in range(a + 1, m):
            F, G = fragments[a], fragments[b_]
            P = inv_blocks[a] @ M[F][:, G]                  # M_FF^-1 M_FG
            R = inv_blocks[b_] @ M[G][:, F]                 # M_GG^-1 M_GF
            norm = torch.sqrt((P * R.transpose(0, 1)).sum().clamp_min(0.0) + eps * eps)
            coupling[a] = coupling[a] + norm
            coupling[b_] = coupling[b_] + norm
    theta = torch.stack([c / (1.0 + coupling[a]) for a in range(m)])
    D = [theta[a] * inv_blocks[a] for a in range(m)]
    S = W.clone()
    for a, F in enumerate(fragments):
        S = S.index_put((F.unsqueeze(1).expand(-1, F.numel()), F.unsqueeze(0).expand(F.numel(), -1)),
                        S[F][:, F] - D[a])
    return D, theta, 0.5 * (S + S.transpose(0, 1))


# ---------------------------------------------------------------------------
# one molecule at fixed total charge N
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _Molecule:
    atoms: Tensor          # [|F|]
    D: Tensor              # [|F|, |F|] effective intra-molecular quadratic (nested certificate applied)
    e_hat: Tensor          # [|F|, S] atomic state energies (nested removal added back)
    window: list[int]      # integer total charges N
    low: int               # capacity bounds of the molecule
    high: int
    top: Tensor            # index of each atom's highest state


def _conditional(mol: _Molecule, N: int, psi: Tensor, states: Tensor, kT: float, tol: float, need_chi: bool):
    """``(e, y, chi, pops)`` of ``min_{1.y = N} sum Phi_u(y_u) - psi.y + 1/2 y.D.y``.

    ``chi = dy/dpsi`` (the fixed-N susceptibility, or None), ``pops`` the atomic
    state populations.  At a capacity bound every atom is in its extreme state.
    """
    if N == mol.low or N == mol.high:
        index = torch.zeros_like(mol.top) if N == mol.low else mol.top
        y = states[index]
        e = mol.e_hat.gather(1, index.unsqueeze(1)).sum() - psi @ y + 0.5 * (y @ (mol.D @ y))
        chi = torch.zeros(y.numel(), y.numel(), dtype=y.dtype, device=y.device) if need_chi else None
        pops = torch.zeros_like(mol.e_hat)
        pops[torch.arange(y.numel(), device=y.device), index] = 1.0
        return e, y, chi, pops
    sol = differentiable_solution(-psi, mol.D, mol.e_hat, states, float(N), kT, tol=tol)
    chi = None
    if need_chi:
        with torch.no_grad():
            X = ensemble.susceptibility(mol.e_hat, states, sol.nu, kT)
            r = X.sqrt()
            core = torch.eye(r.numel(), dtype=r.dtype, device=r.device) + r.unsqueeze(1) * mol.D * r.unsqueeze(0)
            Minv = r.unsqueeze(1) * torch.linalg.solve(core, torch.diag(r))       # X^1/2 (I + X^1/2 D X^1/2)^-1 X^1/2
            Minv = 0.5 * (Minv + Minv.transpose(0, 1))
            v = Minv.sum(1)
            denom = v.sum()
            chi = Minv - torch.outer(v, v) / denom if float(denom) > 1e-300 else Minv
    return sol.energy, sol.charge, chi, sol.populations


# ---------------------------------------------------------------------------
# the canonical combination of the molecules of one partition (§12.13)
# ---------------------------------------------------------------------------

def _convolve(a: Tensor, b: Tensor) -> Tensor:
    """``(a * b)[k] = sum_j a[j] b[k - j]``: the distribution of a sum of independent integers."""
    i = torch.arange(a.numel(), device=a.device)
    j = torch.arange(b.numel(), device=a.device)
    index = (i.unsqueeze(1) + j.unsqueeze(0)).reshape(-1)
    out = torch.zeros(a.numel() + b.numel() - 1, dtype=a.dtype, device=a.device)
    return out.index_add(0, index, (a.unsqueeze(1) * b.unsqueeze(0)).reshape(-1))


def _at(dist: Tensor, k: Tensor) -> Tensor:
    """``dist[k]`` with zero outside ``0 <= k < len(dist)``."""
    inside = (k >= 0) & (k < dist.numel())
    return torch.where(inside, dist[k.clamp(0, dist.numel() - 1)], torch.zeros((), dtype=dist.dtype, device=dist.device))


def _tilt(energies: list, windows: list, Q: int, kT: float) -> float:
    """``mu`` with ``sum_F E_q[N_F] = Q`` under ``q_F(n) ~ exp(-(e_F(n) - mu n)/kT)`` (detached).

    The canonical quantities do not depend on ``mu`` (every configuration has
    ``sum N = Q``); ``mu`` only centres the product measure on Q, so that
    ``Pr_q(sum N = Q)`` is O(1) rather than exp(-gap/kT) (numerics only).
    """
    pairs = [(e.detach(), torch.tensor(w, dtype=e.dtype, device=e.device)) for e, w in zip(energies, windows)]

    def total(m):
        return sum(float(torch.softmax(-(e - m * Ns) / kT, 0) @ Ns) for e, Ns in pairs)

    span = max(float(e.abs().max()) for e, _ in pairs) + 60.0 * kT + 1.0
    lo, hi = -span, span
    for _ in range(64):
        if total(lo) <= Q:
            break
        lo *= 2.0
    for _ in range(64):
        if total(hi) >= Q:
            break
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < Q:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-3 * kT:
            break
    return 0.5 * (lo + hi)


@dataclasses.dataclass
class _Canonical:
    value: Tensor          # A^P = -kT ln Z_Q, eV
    marginals: list        # p_F(n) over each window
    q: list                # tilted product-measure distributions q_F
    prefix: list           # prefix[i] = distribution of N_0 + ... + N_{i-1} under q
    suffix: list           # suffix[i] = distribution of N_i + ... + N_{m-1} under q
    pr: Tensor             # Pr_q(sum N = Q)
    offset: int            # Q - sum of the window lows


def _canonical(energies: list, windows: list, Q: int, kT: float, mu: float) -> _Canonical:
    """The canonical ensemble of independent molecules at fixed ``sum_F N_F = Q`` (graph kept).

        Z_Q = sum_{sum N = Q} prod_F exp(-e_F(N_F)/kT) = exp(-mu Q/kT) prod_F zeta_F(mu) Pr_q(sum N = Q)
        A   = -kT ln Z_Q = mu Q + sum_F Omega_F(mu) - kT ln Pr_q(sum N = Q)
        p_F(n) = q_F(n) Pr_q(sum_{G != F} N_G = Q - n) / Pr_q(sum N = Q)

    with ``q_F(n) = exp(-(e_F(n) - mu n - Omega_F)/kT)`` and ``Omega_F = -kT ln zeta_F``.
    Exact for every ``mu``; the sums are convolutions of the q_F.
    """
    q, omegas = [], []
    for e, window in zip(energies, windows):
        Ns = torch.tensor(window, dtype=e.dtype, device=e.device)
        a = -(e - mu * Ns) / kT
        log_zeta = torch.logsumexp(a, 0)
        omegas.append(-kT * log_zeta)
        q.append(torch.exp(a - log_zeta))
    one = torch.ones(1, dtype=q[0].dtype, device=q[0].device)
    m = len(q)
    prefix = [one]
    for dist in q:
        prefix.append(_convolve(prefix[-1], dist))
    suffix = [one] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix[i] = _convolve(q[i], suffix[i + 1])
    offset = int(Q) - sum(w[0] for w in windows)
    if not 0 <= offset < prefix[m].numel():
        raise FragmentEnsembleError(f"no configuration of the molecular windows has total charge {Q}")
    pr = prefix[m][offset]
    if not float(pr) > 1e-280:
        raise FragmentEnsembleError(f"Pr(sum N = Q) underflowed ({float(pr):.1e}) at the tilt mu = {mu:.3f} eV")
    marginals = []
    for F in range(m):
        rest = _convolve(prefix[F], suffix[F + 1])            # sum over G != F
        k = offset - torch.arange(q[F].numel(), device=pr.device)
        marginals.append(q[F] * _at(rest, k) / pr)
    value = mu * Q + sum(omegas) - kT * torch.log(pr)
    return _Canonical(value=value, marginals=marginals, q=q, prefix=prefix, suffix=suffix, pr=pr, offset=offset)


def _cross_covariance(can: _Canonical) -> dict:
    """``C_FG(n, n') = Pr(N_F = n, N_G = n') - p_F(n) p_G(n')`` for F < G (no graph; the Hessian).

    ``Pr(N_F = n, N_G = n') = q_F(n) q_G(n') Pr_q(sum_{H != F, G} N_H = Q - n - n') / Pr_q(sum N = Q)``.
    """
    with torch.no_grad():
        q = [d.detach() for d in can.q]
        p = [d.detach() for d in can.marginals]
        pr = can.pr.detach()
        m = len(q)
        out = {}
        for F in range(m):
            middle = torch.ones(1, dtype=pr.dtype, device=pr.device)
            for G in range(F + 1, m):
                rest = _convolve(_convolve(can.prefix[F].detach(), middle), can.suffix[G + 1].detach())
                i = torch.arange(q[F].numel(), device=pr.device)
                j = torch.arange(q[G].numel(), device=pr.device)
                k = can.offset - (i.unsqueeze(1) + j.unsqueeze(0))
                joint = q[F].unsqueeze(1) * q[G].unsqueeze(0) * _at(rest, k) / pr
                out[(F, G)] = joint - p[F].unsqueeze(1) * p[G].unsqueeze(0)
                middle = _convolve(middle, q[G])
    return out


# ---------------------------------------------------------------------------
# the two-level ensemble
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FragmentSolution:
    energy: Tensor                 # [] max of the dual = min of the primal (without E_b0), eV
    charge: Tensor                 # [U] atomic free charges c
    mu_e: Tensor                   # [] NaN: the total charge is fixed canonically, there is no reservoir
    potentials: Tensor             # [U] pi-weighted molecular potentials psi
    partitions: list               # [(weight, molecules, populations per molecule over its window)]
    atomic_populations: Tensor     # [U, S] pi- and p-weighted atomic state populations (reporting, no graph)
    iterations: int
    residual: float
    screen: list = dataclasses.field(default_factory=list)   # per partition: (e_F(N) per molecule, A^P), graph kept


class _Problem:
    """The outer dual in the cancellation-free variables s = (z, delta^P for P != r).

    psi^P = -b - Sbar z + delta^P,  sum_P pi_P delta^P = 0  (delta^r eliminated, r = the
    heaviest partition).  Then sum_P pi_P psi^P + b = -Sbar z exactly, so

        L(z, delta) = -1/2 z.Sbar.z + sum_P pi_P A^P(psi^P)

    with A^P the canonical free energy of partition P (:func:`_canonical`; no Sbar^-1 and
    no cancellation, which had stalled a C10H3+ chain at a KKT residual of 1e-9).  At the
    optimum z = c (the atomic charges) and cbar^P = c for every partition; sum c = Q holds
    in every configuration, so no multiplier is needed.  A^P(psi + alpha 1) = A^P(psi) -
    alpha Q, so delta^P along 1 is a flat (gauge) direction, dropped by the pseudo-inverse.
    """

    def __init__(self, partitions, molecules, S_bar, b, states, Q, kT, tol):
        self.partitions = partitions            # [(weight, fragments)]
        self.molecules = molecules              # [[_Molecule per fragment] per partition]
        self.S_bar = S_bar
        self.b = b
        self.states = states
        self.Q = Q
        self.kT = kT
        self.tol = tol
        self.U = int(b.shape[0])
        self.n_p = len(partitions)
        weights = [float(w) for w, _ in partitions]           # floats only to pick the reference
        self.ref = max(range(self.n_p), key=lambda k: weights[k])
        self.others = [k for k in range(self.n_p) if k != self.ref]
        self.dim = self.U + self.U * len(self.others)

    def potentials(self, s):
        """``[psi^P]`` and z."""
        U = self.U
        z = s[:U]
        deltas = {P: s[U + k * U:U + (k + 1) * U] for k, P in enumerate(self.others)}
        w_ref = self.partitions[self.ref][0]
        d_ref = torch.zeros_like(z)
        for P in self.others:
            d_ref = d_ref - (self.partitions[P][0] / w_ref) * deltas[P]
        deltas[self.ref] = d_ref
        base = -self.b - self.S_bar @ z
        return [base + deltas[P] for P in range(self.n_p)], z

    def evaluate(self, s, need_hessian: bool, atomic_populations: bool = False):
        """L, grad, A (= -Hess, if asked), the charge residual, means, fragment populations
        (and, if asked, the pi-weighted atomic state populations as the last element)."""
        kT = self.kT
        psis, z = self.potentials(s)
        value = -0.5 * (z @ (self.S_bar @ z))
        xis, pops, means, screen = [], [], [], []
        atom_pops = torch.zeros(self.U, int(self.states.numel()), dtype=z.dtype, device=z.device)
        for (w, _), mols, psi in zip(self.partitions, self.molecules, psis):
            per_mol = []
            for mol in mols:
                parts = [_conditional(mol, N, psi[mol.atoms], self.states, kT, self.tol, need_hessian)
                         for N in mol.window]
                per_mol.append((torch.stack([e for e, _, _, _ in parts]), torch.stack([y for _, y, _, _ in parts]),
                                [chi for _, _, chi, _ in parts], [ap for _, _, _, ap in parts]))
            energies = [e for e, _, _, _ in per_mol]
            windows = [mol.window for mol in mols]
            can = _canonical(energies, windows, int(round(self.Q)), kT, _tilt(energies, windows, int(round(self.Q)), kT))
            value = value + w * can.value
            cbar = torch.zeros_like(z)
            xi = torch.zeros(self.U, self.U, dtype=z.dtype, device=z.device) if need_hessian else None
            for mol, (_, ys, chis, aps), p in zip(mols, per_mol, can.marginals):
                ybar = p @ ys
                cbar = cbar.index_put((mol.atoms,), ybar)
                if atomic_populations:
                    mix = torch.einsum("n,nus->us", p.detach(), torch.stack(aps).detach())
                    atom_pops = atom_pops.index_put((mol.atoms,), atom_pops[mol.atoms] + float(w) * mix)
                if need_hessian:
                    block = sum(pn * chi for pn, chi in zip(p, chis))
                    dev = ys - ybar.unsqueeze(0)
                    block = block + (dev.transpose(0, 1) * p.unsqueeze(0)) @ dev / kT
                    idx = mol.atoms
                    xi = xi.index_put((idx.unsqueeze(1).expand(-1, idx.numel()), idx.unsqueeze(0).expand(idx.numel(), -1)), block)
            if need_hessian and len(mols) > 1:
                # the canonical constraint correlates the molecules: Cov(c_F, c_G) = Y_F C_FG Y_G^T
                for (F, G), C in _cross_covariance(can).items():
                    block = per_mol[F][1].transpose(0, 1) @ C @ per_mol[G][1] / kT
                    a, b_ = mols[F].atoms, mols[G].atoms
                    xi = xi.index_put((a.unsqueeze(1).expand(-1, b_.numel()), b_.unsqueeze(0).expand(a.numel(), -1)), block)
                    xi = xi.index_put((b_.unsqueeze(1).expand(-1, a.numel()), a.unsqueeze(0).expand(b_.numel(), -1)),
                                      block.transpose(0, 1))
            xis.append(xi)
            pops.append(can.marginals)
            means.append(cbar)
            if atomic_populations:
                screen.append((energies, can.value))
        weights = [w for w, _ in self.partitions]
        mean = sum(w * cb for w, cb in zip(weights, means))
        r = self.ref
        potential_mismatch = self.S_bar @ (mean - z)
        grad = [potential_mismatch] + [weights[P] * (means[r] - means[P]) for P in self.others]
        grad = torch.cat(grad)
        # KKT residual: Sbar (cbar - z) in eV (z is undetermined along null(Sbar), where
        # nothing depends on it; the physical charges are cbar), the pi-weighted
        # disagreement between partitions in e, and |Q - sum cbar| (zero by construction:
        # a check of the canonical sums)
        residual = max([float(potential_mismatch.abs().max()), abs(float(self.Q - mean.sum()))]
                       + [float(weights[P]) * float((means[r] - means[P]).abs().max()) for P in self.others])
        A = None
        if need_hessian:
            S = self.S_bar
            xi_bar = sum(w * xi for w, xi in zip(weights, xis))
            A = torch.zeros(self.dim, self.dim, dtype=z.dtype, device=z.device)
            U = self.U
            A[:U, :U] = S + S @ xi_bar @ S
            for k, P in enumerate(self.others):
                rows = slice(U + k * U, U + (k + 1) * U)
                A[:U, rows] = -weights[P] * (S @ (xis[P] - xis[r]))
                for k2, P2 in enumerate(self.others):
                    cols = slice(U + k2 * U, U + (k2 + 1) * U)
                    blk = (weights[P] * weights[P2] / weights[r]) * xis[r]
                    if P == P2:
                        blk = blk + weights[P] * xis[P]
                    A[rows, cols] = blk
            A = torch.triu(A) + torch.triu(A, 1).transpose(0, 1)         # symmetric from the upper triangle
        if atomic_populations:
            self.screen = screen                 # [(conditional energies per molecule, A^P)] per partition
            return value, grad, A, residual, mean, pops, atom_pops
        return value, grad, A, residual, means, pops


def _pseudo_solve(A: Tensor, g: Tensor) -> Tensor:
    """``A^+ g`` with the near-null space (flat directions of the dead zone) dropped."""
    evals, evecs = torch.linalg.eigh(A)
    floor = 1e-11 * float(evals.abs().max())
    inv = torch.where(evals > floor, 1.0 / evals.clamp_min(floor), torch.zeros_like(evals))
    return evecs @ (inv * (evecs.transpose(0, 1) @ g))


def solve_fragment_ensemble(partitions, molecules, S_bar, b, states, Q, kT, *, tol=1e-9, max_iter=100,
                            inner_tol=1e-11) -> FragmentSolution:
    """Maximise the dual (detached), then re-attach the optimum to the graph (one Newton step).

    See the module docstring.  Every inner fixed-N problem is
    ``free_solver.differentiable_solution``; the outer step uses the exact
    Hessian ``-A`` in the variables of :class:`_Problem` (the canonical
    susceptibilities E[chi] + Cov/kT, cross-molecule covariance included, and
    Sbar), with the dead zone's and the gauge's flat directions dropped.
    """
    prob = _Problem(partitions, molecules, S_bar, b, states, Q, kT, inner_tol)          # carries the graph
    detached_mols = [[dataclasses.replace(m, D=m.D.detach(), e_hat=m.e_hat.detach()) for m in mols]
                     for mols in molecules]
    detached_parts = [(float(w), idx) for w, idx in partitions]
    work = _Problem(detached_parts, detached_mols, S_bar.detach(), b.detach(), states, Q, kT, inner_tol)
    with torch.no_grad():
        s = torch.zeros(work.dim, dtype=b.dtype, device=b.device)       # z = 0, delta = 0: psi^P = -b
        value, grad, A, residual, means, _ = work.evaluate(s, True)
        iteration = 0
        for iteration in range(1, max_iter + 1):
            # KKT residual in e: a partition of weight pi moves the energy by pi x its
            # mismatch, so an almost-absent partition (pi ~ 1e-18 at a switch end) need
            # not, and numerically cannot, converge.
            if residual < tol:
                break
            step = _pseudo_solve(A, grad)
            n = float(step.abs().max())
            if n > 2.0:
                step = step * (2.0 / n)
            slope = float(grad @ step)
            resolution = 1e4 * torch.finfo(b.dtype).eps * (1.0 + abs(float(value)))
            alpha = 1.0
            if slope > resolution:
                for _ in range(60):
                    trial = s + alpha * step
                    t_value = work.evaluate(trial, False)[0]
                    if float(t_value) >= float(value) + 1e-4 * alpha * slope:
                        break
                    alpha *= 0.5
                else:
                    raise FragmentEnsembleError(f"line search failed (slope {slope:.2e}, residual {residual:.2e})")
            s = s + alpha * step
            value, grad, A, residual, means, _ = work.evaluate(s, True)
        else:
            raise FragmentEnsembleError(f"fragment ensemble did not converge in {max_iter} iterations "
                                        f"(residual {residual:.2e})")
        A0 = A
    # one Newton step with the graph of the parameters, then the value at the stepped point
    _, grad_g, _, _, _, _ = prob.evaluate(s, False)
    step = _pseudo_solve(A0, grad_g)
    s_final = s + step
    value, _, _, _, c, pops, atom_pops = prob.evaluate(s_final, False, atomic_populations=True)
    psis, _ = prob.potentials(s_final)                                    # c = the mean charges, not z
    potentials = sum(w * psi for (w, _), psi in zip(prob.partitions, psis))
    mu_e = torch.full((), float("nan"), dtype=b.dtype, device=b.device)    # canonical: no reservoir (§12.13)
    return FragmentSolution(energy=value, charge=c, mu_e=mu_e, potentials=potentials,
                            partitions=[(w, mols, p) for (w, _), mols, p in zip(prob.partitions, prob.molecules, pops)],
                            atomic_populations=atom_pops, iterations=iteration, residual=residual,
                            screen=prob.screen)


# ---------------------------------------------------------------------------
# the exact canonical configuration ensemble (SPEC §15)
# ---------------------------------------------------------------------------
#
# Per partition, with W~ = W - diag(r) the atom-level certified operator (r from §8a,
# so every configuration problem is convex) and Phi_u carrying 1/2 r n^2:
#
#     E(N)  = min_{G y = N} sum_u Phi_u(y_u) + b^T y + 1/2 y^T W~ y        (one solve per N)
#     F_P   = -kT ln sum_{N in C} w_N exp(-E(N)/kT)
#
# No averaging between molecules and no block certificate: the inter-molecular
# Coulomb, the polarisation of every molecule (free and bound) and the canonical
# constraint are exact in every configuration (§14.2).  C is the set of
# configurations the canonical two-level ensemble gives probability >= CONFIG_EPS_LO;
# w_N rises smoothly (C^2 in log p) from 0 there to 1 at CONFIG_EPS_HI, so no
# configuration enters or leaves with a jump.

#: Configurations with two-level probability below this are not solved.
CONFIG_EPS_LO = 1e-12
#: ... and from this probability up they carry their full Boltzmann weight.
CONFIG_EPS_HI = 1e-10


def inclusion_weight(log_p: Tensor) -> Tensor:
    """C^2 quintic smoothstep in log p: 0 at CONFIG_EPS_LO, 1 at CONFIG_EPS_HI."""
    lo, hi = math.log(CONFIG_EPS_LO), math.log(CONFIG_EPS_HI)
    t = ((log_p - lo) / (hi - lo)).clamp(0.0, 1.0)
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def select_configurations(log_weights: list, windows: list, Q: int, log_Z: float, cap: int) -> list:
    """Every N in the windows with sum N = Q and log p(N) = sum_F a_F(N_F) - log Z >= log CONFIG_EPS_LO.

    ``log_weights[F][k] = -e_F(windows[F][k]) / kT`` (floats).  Depth-first, pruned
    exactly by the largest attainable remaining sum.  Refuses more than ``cap``.
    """
    m = len(windows)
    best_rest = [0.0] * (m + 1)
    for F in range(m - 1, -1, -1):
        best_rest[F] = best_rest[F + 1] + max(log_weights[F])
    floor = math.log(CONFIG_EPS_LO) + log_Z
    found = []

    def visit(F: int, partial: float, chosen: list, remaining: int) -> None:
        if F == m - 1:
            if remaining in windows[F]:
                total = partial + log_weights[F][windows[F].index(remaining)]
                if total >= floor:
                    found.append(tuple(chosen + [remaining]))
            return
        for k, n in enumerate(windows[F]):
            value = partial + log_weights[F][k]
            if value + best_rest[F + 1] < floor:
                continue
            visit(F + 1, value, chosen + [n], remaining - n)
            if len(found) > cap:
                raise FragmentEnsembleError(
                    f"more than {cap} molecular charge configurations carry probability >= "
                    f"{CONFIG_EPS_LO:g}; raise fragment_max_configurations or use fragment_ensemble='two-level'"
                )

    visit(0, 0.0, [], int(Q))
    return found


def configuration_energy(N: tuple, mols: list, b: Tensor, W: Tensor, e_hat: Tensor, states: Tensor,
                         kT: float, tol: float):
    """``(E(N), charge [U], populations [U, S], residual)`` for one molecular configuration.

    A molecule at a capacity bound has every atom in its extreme state: it is fixed
    exactly and enters the others' problem only through its charges.
    """
    U = int(b.shape[0])
    dtype, device = b.dtype, b.device
    fixed_mask = torch.zeros(U, dtype=torch.bool, device=device)
    index = torch.zeros(U, dtype=torch.long, device=device)
    free_groups, free_targets = [], []
    group = torch.full((U,), -1, dtype=torch.long, device=device)
    for mol, n in zip(mols, N):
        if n == mol.low or n == mol.high:
            fixed_mask[mol.atoms] = True
            index[mol.atoms] = 0 if n == mol.low else mol.top
        else:
            group[mol.atoms] = len(free_groups)
            free_groups.append(mol)
            free_targets.append(float(n))
    charge = torch.zeros(U, dtype=dtype, device=device)
    populations = torch.zeros(U, int(states.numel()), dtype=dtype, device=device)
    energy = torch.zeros((), dtype=dtype, device=device)
    fixed = torch.nonzero(fixed_mask).squeeze(1)
    if fixed.numel():
        c_f = states[index[fixed]]
        energy = (e_hat[fixed].gather(1, index[fixed].unsqueeze(1)).sum() + b[fixed] @ c_f
                  + 0.5 * (c_f @ (W[fixed][:, fixed] @ c_f)))
        charge = charge.index_put((fixed,), c_f)
        populations[fixed, index[fixed]] = 1.0
    residual = 0.0
    if free_groups:
        free = torch.nonzero(~fixed_mask).squeeze(1)
        b_free = b[free] + (W[free][:, fixed] @ charge[fixed] if fixed.numel() else 0.0)
        sol = differentiable_constrained_solution(
            b_free, W[free][:, free], e_hat[free], states, group[free],
            torch.tensor(free_targets, dtype=dtype, device=device), kT, tol=tol)
        energy = energy + sol.energy
        charge = charge.index_put((free,), sol.charge)
        populations = populations.index_put((free,), sol.populations.detach())
        residual = max(sol.residual_charge, sol.residual_total)
    return energy, charge, populations, residual


def exact_partition(mols: list, screen: tuple, b: Tensor, W: Tensor, e_hat: Tensor, states: Tensor, Q: int,
                    kT: float, tol: float, cap: int):
    """``(F_P, mean charges [U], populations [U, S], marginals per molecule, n_configs, residual)``.

    ``screen`` = (conditional energies e_F per molecule, canonical value A^P) of the
    two-level solve of this partition, which selects and smoothly weights C.
    """
    energies, A_value = screen
    log_weights = [[float(v) for v in (-e.detach() / kT)] for e in energies]
    windows = [mol.window for mol in mols]
    log_Z = -float(A_value.detach()) / kT
    configs = select_configurations(log_weights, windows, Q, log_Z, cap)
    terms, charges, pops, residual = [], [], [], 0.0
    for N in configs:
        log_p = sum(-energies[F][windows[F].index(n)] / kT for F, n in enumerate(N)) + A_value / kT
        w = inclusion_weight(log_p)
        E, c, pop, res = configuration_energy(N, mols, b, W, e_hat, states, kT, tol)
        terms.append(torch.log(w.clamp_min(1e-300)) - E / kT)
        charges.append(c)
        pops.append(pop)
        residual = max(residual, res)
    terms = torch.stack(terms)
    F_P = -kT * torch.logsumexp(terms, 0)
    P = torch.softmax(terms, 0)
    charge = sum(P[k] * charges[k] for k in range(len(configs)))
    populations = sum(float(P[k]) * pops[k] for k in range(len(configs)))
    marginals = []
    for F, mol in enumerate(mols):
        marg = torch.zeros(len(mol.window), dtype=b.dtype, device=b.device)
        for k, N in enumerate(configs):
            marg[mol.window.index(N[F])] += P[k].detach()
        marginals.append(marg)
    return F_P, charge, populations, marginals, len(configs), residual


# ---------------------------------------------------------------------------
# building the problem from the module's quantities
# ---------------------------------------------------------------------------

def build_molecules(fragments, D_list, e_hat, J, states, top, Q, window_half, nested, sic_theta, sic_smoothing,
                    windows=None):
    """Molecules of one partition: nested certificate (§12.5), windows (§12.7).

    ``windows`` maps a molecule's atom tuple to an enlarged (N_low, N_high) found
    by :func:`edge_populations` on an earlier solve.
    """
    mols = []
    single = len(fragments) == 1
    for F, D in zip(fragments, D_list):
        e_F = e_hat[F]
        if nested:
            r = certified_removal(D, J[F][:, F], theta=sic_theta, smoothing=sic_smoothing)
            D_eff = D - torch.diag(r)
            e_F = e_F + 0.5 * r.unsqueeze(1) * (states * states).unsqueeze(0)
        else:
            D_eff = D
        low = -int(F.numel())
        high = int(states[top[F]].sum())
        if single:
            window = [int(round(Q))]
        else:
            w = max(window_half, int(math.ceil(abs(Q))))
            n_lo, n_hi = max(low, -w), min(high, w)
            if windows is not None and tuple(F.tolist()) in windows:
                n_lo, n_hi = windows[tuple(F.tolist())]
            window = list(range(n_lo, n_hi + 1))
        mols.append(_Molecule(atoms=F, D=0.5 * (D_eff + D_eff.transpose(0, 1)), e_hat=e_F, window=window,
                              low=low, high=high, top=top[F]))
    return mols


def edge_populations(solution: FragmentSolution, threshold: float = 1e-8) -> dict:
    """Molecules whose pi-weighted population at a window edge (not a capacity bound) exceeds
    ``threshold``.

    Returns {atom tuple: (N_low, N_high)} with the offending side(s) widened by
    one.  Truncating a window is exact iff the omitted states carry no
    population (§12.7); the module widens and re-solves until this is empty.
    A partition enters the energy with its weight pi, so its edge population is
    weighted by pi, as its KKT residual is: an almost-absent partition (whose
    potentials the solver need not converge) cannot drive the windows.
    """
    out = {}
    for w, mols, pops in solution.partitions:
        weight = float(w)
        for mol, p in zip(mols, pops):
            if len(mol.window) <= 1:
                continue
            lo, hi = mol.window[0], mol.window[-1]
            new_lo = lo - 1 if (lo > mol.low and weight * float(p[0]) > threshold) else lo
            new_hi = hi + 1 if (hi < mol.high and weight * float(p[-1]) > threshold) else hi
            if (new_lo, new_hi) != (lo, hi):
                key = tuple(mol.atoms.tolist())
                prev = out.get(key, (new_lo, new_hi))
                out[key] = (min(prev[0], new_lo), max(prev[1], new_hi))
    return out


__all__ = [
    "BOND_TOLERANCE_OFF", "BOND_TOLERANCE_ON", "CERTIFICATE_C", "FragmentEnsembleError", "FragmentSolution",
    "PARTITION_WEIGHT_FLOOR",
    "block_certificate", "bond_switch", "build_molecules", "chemical_bonds", "edge_populations",
    "percolation_partitions", "solve_fragment_ensemble", "FreeChargeNotConvergedError",
    "CONFIG_EPS_HI", "CONFIG_EPS_LO", "configuration_energy", "exact_partition", "inclusion_weight",
    "select_configurations",
]
