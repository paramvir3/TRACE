"""The cluster product ensemble (trace_redox SPEC 36; ``RedoxConfig(diabatic_space='clusters')``).

The shipped diabatic ensemble (SPEC 17, 21) builds ONE Hamiltonian per structure and measures every configuration's
importance from the thermal reference of the whole structure.  M identical, non-interacting clusters then do not have
M times the energy of one: a configuration with k clusters excited lies k excitations above the reference, its
importance falls as the product of k path factors, and SPEC 21 switches its couplings off (measured on Au2/MgO(001),
2x2 against 1x1: +220.8 and +99.8 meV per cell; experiments/benchmarks_4g/README.md R13).

Here the free energy is built from linked pieces (NEW-TO-TRACE: the linked selection, clusters from the supports of
the selected configurations, the reservoir rule, increments of the free energy of the configuration Hamiltonian;
ESTABLISHED: the method of increments, an energy as one-body plus two-body increments of local pieces; H. Stoll,
Phys. Rev. B 46, 6700 (1992); review: B. Paulus, Phys. Rep. 428, 1 (2006); both checked 2026-11-11 from the
publishers' records and abstracts):

1. Reference n0: the lowest configuration found.
2. Linked selection.  From n0 every admissible move is tried.  From m != n0 a move is tried only if its donor or
   its acceptor lies in the support of m, supp(m) = {u not a reservoir : m_u != n0_u}.  A product of excitations on
   disjoint supports is therefore never selected; importances are measured from n0 (x = E - E(n0)), so they do not
   depend on what other clusters do.  Rules as in SPEC 17.3 (energy window) and SPEC 21 (amplitude closure).
3. Clusters: units joined whenever one selected configuration touches both (union-find on the supports).  Cluster c
   holds S_c = {n0} + its configurations, every other cluster at n0, with exact whole-system energies.
4. Each cluster has its own Hamiltonian H_c (SPEC 21's switch from the cluster's own thermal reference) and
   F_c = -kT ln Tr exp(-H_c / kT).
5. Each pair of clusters has the Hamiltonian on the product space S_c x S_c', diagonal = the exact energy of every
   product configuration, off-diagonal = O_c (x) 1 + 1 (x) O_c' (each cluster's switched couplings), and F_cc'.
6. F = E0 + sum_c (F_c - E0) + sum_{c<c'} (F_cc' - F_c - F_c' + E0),  E0 = E(n0); charges and populations by the
   same increments of rho's diagonal.
7. The product closure (SPEC 36.7, NEW-TO-TRACE; ``closure=True``, the default).  The space is meant to hold every
   product of selected excitations on disjoint supports; steps 2-5 alone hold them only ACROSS clusters.
   a. Products do not join clusters.  A selected configuration that is a product m1 (+) m2 of selected
      configurations with disjoint supports, with no coupling T between the two supports, does not join their units
      (a transfer between two clusters through a reservoir); if its parts end in two different clusters it is a row
      of their pair space and of neither list.  If no decomposition has each part inside one cluster it joins after
      all (a fixed point: the flags only grow).
   b. Closure.  A cluster's space is the product closure of its linked set S_c: every n0 + sum_{m in I} (m - n0),
      I a subset of S_c with pairwise disjoint supports (reservoir units carry the sum), inside the bounds and
      admissible.  Diagonal: the exact energy.  Off-diagonal: the true one-electron elements of the closed set, each
      multiplied by a switch read from the linked parts: for the element between p and p', the largest of SPEC 21's
      switches of the same move between linked rows (a, c) with p = a (+) s, s in the closed set and supp(s)
      disjoint from supp(a) and supp(c); 0 when there is none.  Elements between two linked rows keep SPEC 21's
      switch on S_c.  A spectator excitation therefore never weakens a coupling.
   ``closure=False`` is the first build (legacy): a cluster that holds two disjoint excitations then lacks their
   product, and two clusters that merge lose theirs (measured: +0.790 eV at the merge of two model dimers; +1.243 eV
   on the benchmark's doped 2x1 cell, where two transfers through the band unit merged the two Au dimers;
   experiments/benchmarks_4g/README.md R14, R19).

Exact statements (tests/redox/test_redox_cluster_product.py, test_redox_cluster_closure.py): one cluster is the
shipped ensemble on its closed set; clusters that do not interact are additive for any number of copies; two clusters
are exact on their product space (the Kronecker sum equals the global Hamiltonian up to a diagonal +-1 gauge when each
cluster conserves its electron number); an unswitched merged cluster is exact on its closed set; a merge changes
nothing at the merge.  Limits: three-cluster increments are dropped (second order in the inter-cluster interaction
through three-body induction); the closure inside one cluster is exponential in the number of its disjoint parts
(refused beyond the cap, never truncated); SPEC 21's switch is not variational (measured on a 6-atom random-weight
module: the switched Hamiltonian on the complete space lies 16.4 meV below the bare one), so neither is the cluster
energy; a reservoir unit (the band unit) shared by two clusters loses the exchange sign of two electrons placed in
it from different clusters.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .diabatic import (ConfigurationSearchError, _admissible_moves, _lower, _move_energies, _valid_moves, admissible,
                       canonical_free_energy, configuration_energies, coupling_amplitude, density_matrix,
                       diabatic_hamiltonian, resolve_matrix, search_configurations, switched_hamiltonian)

#: restarts of the selection when it meets a configuration below its reference (as in ``select_configurations``)
_RESTARTS = 16


def _split(n0: Tensor, configs: Tensor, reservoir: Tensor, couplings: Tensor | None = None,
           closure: bool = True) -> list:
    """The clusters of a linked selection: ``configs [m, U]`` (n0 first) -> a list of ``[m_c, U]``, n0 first in each.
    Units are joined when one configuration touches both; reservoir units join nothing.

    ``closure`` (SPEC 36.7, rule 2): a configuration that is the product m1 (+) m2 of two selected configurations
    with disjoint supports, with no coupling (``couplings [U, U]``, None: none anywhere) between the two supports,
    joins nothing.  If its parts end in two different clusters it is left out of both lists (it is a row of their
    pair space); if no decomposition has each part inside one cluster it joins after all (a fixed point)."""
    U = int(n0.numel())
    rows = configs[1:]
    masks = (rows != n0) & ~reservoir                                         # [m - 1, U]
    supports = [mask.nonzero().flatten().tolist() for mask in masks]
    for touched in supports:
        if not touched:
            raise NotImplementedError("SPEC 36: a selected configuration differs from the reference on reservoir units "
                                      "only (charge moved between reservoirs); it belongs to no cluster")
    splits = [[] for _ in supports]                       # the decompositions (j, k) of every configuration
    if closure:
        index = {tuple(row.tolist()): i for i, row in enumerate(rows)}
        linked = None if couplings is None else ((couplings != 0) | (couplings != 0).transpose(0, 1))
        for i in range(len(supports)):
            for j in range(len(supports)):
                if j == i or bool((masks[j] & ~masks[i]).any()) or len(supports[j]) == len(supports[i]):
                    continue
                k = index.get(tuple((rows[i] - rows[j] + n0).tolist()))
                if k is None or k < j or bool((masks[k] & masks[j]).any()):
                    continue
                if linked is not None and bool(linked[masks[j]][:, masks[k]].any()):
                    continue
                splits[i].append((j, k))
    joins = [not found for found in splits]
    while True:
        parent = list(range(U))

        def find(u: int) -> int:
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        for touched, join in zip(supports, joins):
            if not join:
                continue
            for u in touched[1:]:
                ra, rb = find(touched[0]), find(u)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
        roots = [{find(u) for u in touched} for touched in supports]
        again = False
        for i, found in enumerate(splits):
            if joins[i] or len(roots[i]) == 1:
                continue
            if not any(len(roots[j]) == 1 and len(roots[k]) == 1 for j, k in found):
                joins[i] = again = True
        if not again:
            break
    members: dict = {}
    for i, root in enumerate(roots):
        if len(root) == 1:                                # a product of two clusters' rows is left to their pair space
            members.setdefault(next(iter(root)), []).append(i + 1)
    out = []
    for root in sorted(members):                          # the root is the cluster's lowest unit: a fixed order
        index_ = torch.tensor(members[root], dtype=torch.long, device=configs.device)
        out.append(torch.cat([n0.unsqueeze(0), configs[index_]], dim=0))
    return out


def _closure(n0: Tensor, S: Tensor, reservoir: Tensor, L: Tensor, K: Tensor, units: Tensor | None,
             frames: Tensor | None, cap: int) -> Tensor:
    """SPEC 36.7, rule 1: ``S [m, U]`` (n0 first) and, after it, every other product n0 + sum_{i in I} (S[i] - n0) of
    members with pairwise disjoint supports (reservoir units carry the sum) that lies inside ``[L, K]`` and is
    admissible.  A product outside the bounds is not kept but is still extended (a third member can bring a reservoir
    back inside).  More than ``cap`` rows, or more than 64 ``cap`` products tried, is refused."""
    m = int(S.shape[0])
    masks = (S != n0) & ~reservoir
    found = {tuple(row.tolist()) for row in S}
    fresh = []
    stack = [(S[i], masks[i], i) for i in range(1, m)]
    tried = 0
    while stack:
        p, mask, top = stack.pop()
        for j in range(top + 1, m):
            if bool((mask & masks[j]).any()):
                continue
            tried += 1
            if tried > 64 * cap:
                raise ConfigurationSearchError(f"the product closure of one cluster tried more than 64 x cap = "
                                               f"{64 * cap} products; raise the cap")
            q = p + S[j] - n0
            stack.append((q, mask | masks[j], j))
            key = tuple(q.tolist())
            if key in found or not bool(((q >= L) & (q <= K)).all()):
                continue
            found.add(key)
            fresh.append(q)
    if not fresh:
        return S
    extra = torch.stack(fresh)
    if units is not None:
        extra = extra[admissible(extra, units, frames)]
    if m + int(extra.shape[0]) > cap:
        raise ConfigurationSearchError(f"the product closure of one cluster holds more than cap = {cap} "
                                       f"configurations ({m} linked, {m + int(extra.shape[0])} closed; the closure is "
                                       f"exponential in the cluster's disjoint parts, SPEC 36.7); raise the cap "
                                       f"(RedoxConfig.fragment_max_configurations)")
    return torch.cat([S, extra], dim=0)


def linked_configurations(e_hat: Tensor, states: Tensor, b: Tensor, W: Tensor, max_charge: Tensor, Q: int, *,
                          window: float, cap: int, T: Tensor | None = None, amplitudes=(0.05, 0.2), kT: float = 0.1,
                          start: Tensor | None = None, min_charge: Tensor | None = None,
                          units: Tensor | None = None, frames: Tensor | None = None, pairs=None,
                          electrons: Tensor | None = None, reservoir: Tensor | None = None, search=None,
                          couplings: Tensor | None = None, closure: bool = True, steps=None):
    """``(n0 [U], clusters)``: the reference and the linked selection split into clusters (module docstring, 1-3).

    ``T`` None: the energy window, every linked configuration with E - E(n0) <= ``window`` that is reached through
    linked moves inside the window.  Otherwise also the amplitude closure of SPEC 21 with x = E - E(n0): a best-first
    search from the window set's seeds exp(-x / 2 kT) through linked one-electron moves with T != 0, kept while
    r g(T, x) > ``amplitudes[0]``.  ``reservoir [U]`` (bool): units that never enter a support.  The reference is the
    lowest configuration the descent and the linked search find; meeting a lower one restarts from it.  More than
    ``cap`` configurations in all is refused, never truncated.  Detached: a selection.  ``couplings`` (every
    coupling, whatever the selection rule) and ``closure``: SPEC 36.7's rule 2 in :func:`_split`; ``closure`` False
    is the first build (legacy; README R14)."""
    search = search_configurations if search is None else search
    with torch.no_grad():
        e_hat, b, W = e_hat.detach(), b.detach(), W.detach()
        T = None if T is None else T.detach()
        device = b.device
        U = int(b.shape[0])
        K = max_charge.to(torch.long).to(device)
        L = _lower(min_charge, K)
        res = (torch.zeros(U, dtype=torch.bool, device=device) if reservoir is None
               else reservoir.to(torch.bool).to(device))
        restricted = units is not None
        if restricted:
            if frames is None:
                raise ValueError("units need frames")
            units, frames = units.to(torch.long).to(device), frames.to(torch.long).to(device)
        k_max = int((K - L).max())
        a_off = float(amplitudes[0])

        def moves_ok(m: Tensor, k: int) -> Tensor:
            ok = _valid_moves(m, K, L, k)
            return ok & _admissible_moves(m, units, frames, k) if restricted else ok

        def linked(m: Tensor, n0: Tensor) -> Tensor:
            touched = (m != n0) & ~res
            return touched.unsqueeze(1) | touched.unsqueeze(0)

        def too_many():
            return ConfigurationSearchError(
                f"the linked selection holds more than cap = {cap} configurations; raise the cap")

        for _ in range(_RESTARTS):
            low, E_low = search(e_hat, states, b, W, max_charge, Q, window=0.0, cap=cap, start=start,
                                min_charge=min_charge, units=units, frames=frames, pairs=pairs)
            i0 = int(torch.argmin(E_low))
            n0, E0 = low[i0].to(torch.long), float(E_low[i0])
            tol = 1e-12 * (1.0 + abs(E0))
            key0 = tuple(n0.tolist())
            found = {key0: E0}
            lower = None
            frontier = [n0]
            while frontier and lower is None:             # the linked energy window
                m = frontier.pop()
                key = tuple(m.tolist())
                base = found[key]
                link = None if key == key0 else linked(m, n0)
                fresh, seen = [], set()
                for k in range(1, k_max + 1):
                    delta = _move_energies(m, e_hat, K, b, W, states=states, min_charge=L, k=k, pairs=pairs)
                    ok = ((base + delta) <= E0 + window) & moves_ok(m, k)
                    if link is not None:
                        ok = ok & link
                    for a, c in torch.nonzero(ok).tolist():
                        nxt = m.clone()
                        nxt[a] += k
                        nxt[c] -= k
                        k2 = tuple(nxt.tolist())
                        if k2 not in found and k2 not in seen:
                            seen.add(k2)
                            fresh.append(nxt)
                if not fresh:
                    continue
                values = configuration_energies(torch.stack(fresh), e_hat, states, b, W, pairs=pairs)
                for nxt, value in zip(fresh, values.tolist()):
                    if value < E0 - tol:
                        lower = nxt
                        break
                    if value > E0 + window:
                        continue
                    found[tuple(nxt.tolist())] = value
                    if len(found) > cap:
                        raise too_many()
                    frontier.append(nxt)
            if lower is None and T is not None:           # the linked amplitude closure (SPEC 21 from n0)
                best = {k: math.exp(-0.5 * max(v - E0, 0.0) / kT) for k, v in found.items()}
                heap = [(-r, k) for k, r in best.items() if r > a_off]
                heapq.heapify(heap)
                done = set()
                while heap and lower is None:
                    neg, key = heapq.heappop(heap)
                    if key in done or -neg < best[key]:
                        continue
                    done.add(key)
                    r_m = -neg
                    m = torch.tensor(key, dtype=torch.long, device=device)
                    delta = _move_energies(m, e_hat, K, b, W, states=states, min_charge=L, k=1, pairs=pairs)
                    T_here = resolve_matrix(T, steps, m)     # SPEC 40.6 ix: the coupling of each move out of m
                    ok = moves_ok(m, 1) & torch.isfinite(delta) & (T_here != 0)
                    if key != key0:
                        ok = ok & linked(m, n0)
                    x = (found[key] + delta) - E0
                    T_m = T_here
                    if electrons is not None:             # SPEC 29: the factor of each move out of m
                        z = electrons.to(torch.long).to(device)
                        if U > int(z.numel()):
                            z = torch.cat([z, torch.zeros(U - int(z.numel()), dtype=torch.long, device=device)])
                        parity = (z - m) % 2
                        T_m = T_here * torch.where(parity.unsqueeze(1) == parity.unsqueeze(0),
                                                   torch.full_like(T, math.sqrt(2.0)), torch.ones_like(T))
                    cand = r_m * coupling_amplitude(T_m, torch.where(ok, x, torch.zeros_like(x)))
                    for a, c in torch.nonzero(ok & (cand > a_off)).tolist():
                        nxt = m.clone()
                        nxt[a] += 1
                        nxt[c] -= 1
                        k2 = tuple(nxt.tolist())
                        value = float(cand[a, c])
                        if value <= best.get(k2, 0.0):
                            continue
                        if k2 not in found:
                            E_new = found[key] + float(delta[a, c])
                            if E_new < E0 - tol:
                                lower = nxt
                                break
                            found[k2] = E_new
                            if len(found) > cap:
                                raise too_many()
                        best[k2] = value
                        heapq.heappush(heap, (-value, k2))
            if lower is None:
                break
            start = lower
        else:
            raise ConfigurationSearchError("the linked selection kept meeting configurations below its reference")
        configs = torch.tensor(list(found), dtype=torch.long, device=device)       # n0 first (insertion order)
        return n0, _split(n0, configs, res, couplings=couplings, closure=closure)


@dataclass
class ClusterEnsemble:
    """The cluster product ensemble of one structure (module docstring, 4-6)."""
    energy: Tensor                 # F
    charge: Tensor                 # [U], increments of rho's diagonal
    populations: Tensor            # [U, S], the same increments of the state populations (detached; a pair increment
    #                                can make one slightly negative)
    reference: Tensor              # n0 [U]
    clusters: list                 # [m_c, U] each, n0 first
    hamiltonians: list             # H_c [m_c, m_c]
    pair_sizes: list               # the dimension of every pair Hamiltonian, pairs in the order (0,1), (0,2), ...
    expectation: Tensor | None = None   # [D], the same increments of an observable's average (with the graph)
    #: SPEC 36.10 (the response record): every pair space as it entered F, pairs in the order of ``pair_sizes``:
    #: (i, j), its Hamiltonian [m, m] and its configurations [m, U] (with the graph; no extra work)
    pair_index: list = field(default_factory=list)
    pair_hamiltonians: list = field(default_factory=list)
    pair_configs: list = field(default_factory=list)


def _state_populations(configs: Tensor, p: Tensor, states: Tensor) -> Tensor:
    U = int(configs.shape[1])
    index = torch.round(configs.to(p.dtype) - states[0]).to(torch.long)
    out = torch.zeros(U, int(states.numel()), dtype=p.dtype, device=p.device)
    return out.index_put((torch.arange(U, device=p.device).repeat(int(configs.shape[0])), index.reshape(-1)),
                         p.detach().repeat_interleave(U), accumulate=True)


def _closed_hamiltonian(n0: Tensor, S: Tensor, m: int, E: Tensor, hamiltonian, bare, reservoir: Tensor) -> Tensor:
    """The Hamiltonian of a closed set ``S [M, U]`` whose first ``m`` rows are the linked set (SPEC 36.7, rule 1).

    Diagonal ``E``; off-diagonal: the true elements ``bare(E, S)`` of the closed set, each times a switch.  Between
    two linked rows: SPEC 21's switch on the linked set, sigma = hamiltonian / bare there.  Otherwise, for the move
    between p and p': the largest sigma of the same move between linked rows (a, c) such that p = a (+) s with s a
    row of the closed set whose support is disjoint from supp(a) | supp(c); 0 when there is none."""
    B = bare(E, S)
    H_l = hamiltonian(E[:m], S[:m])
    off = ~torch.eye(int(S.shape[0]), dtype=torch.bool, device=S.device)
    B_l = B[:m, :m]
    coupled = (B_l != 0) & off[:m, :m]
    sigma_l = torch.where(coupled, H_l / torch.where(coupled, B_l, torch.ones_like(B_l)), torch.zeros_like(B_l))
    masks = (S != n0) & ~reservoir
    index = {tuple(row.tolist()): i for i, row in enumerate(S)}
    moves: dict = {}                                      # the move's vector -> the linked pairs (a, c) that make it
    for a, c in torch.nonzero(coupled).tolist():
        moves.setdefault(tuple((S[c] - S[a]).tolist()), []).append((a, c))
    sigma = torch.zeros_like(B).index_put((torch.arange(m, device=S.device).repeat_interleave(m),
                                           torch.arange(m, device=S.device).repeat(m)), sigma_l.reshape(-1))
    rows, cols, values = [], [], []
    for i, j in torch.nonzero((B != 0) & off).tolist():
        if i > j or (i < m and j < m):
            continue
        best = None
        for a, c in moves.get(tuple((S[j] - S[i]).tolist()), []):
            s = index.get(tuple((S[i] - S[a] + n0).tolist()))
            if s is None or bool((masks[s] & (masks[a] | masks[c])).any()):
                continue
            best = sigma_l[a, c] if best is None else torch.maximum(best, sigma_l[a, c])
        if best is not None:
            rows += [i, j]
            cols += [j, i]
            values += [best, best]
    if rows:
        sigma = sigma.index_put((torch.tensor(rows, device=S.device), torch.tensor(cols, device=S.device)),
                                torch.stack(values))
    return torch.diag(E) + torch.where(off, B * sigma, torch.zeros_like(B))


def cluster_free_energy(n0: Tensor, clusters: list, energy, hamiltonian, kT: float, max_charge: Tensor,
                        min_charge: Tensor | None, states: Tensor, *, units: Tensor | None = None,
                        frames: Tensor | None = None, observable=None, reservoir: Tensor | None = None,
                        closure: bool = True, bare=None, cap: int = 100000) -> ClusterEnsemble:
    """F and the charges from one-cluster and pair increments (module docstring, 4-7).

    ``closure`` (SPEC 36.7, rule 1): every cluster's space is the product closure of its linked set
    (:func:`_closure`; ``reservoir`` as in the selection, ``cap`` its refusal).  ``bare(E, configs) -> [m, m]``: the
    Hamiltonian with every coupling at full strength; given when ``hamiltonian`` switches couplings, so that the
    closed set's elements carry the switch of their linked part (:func:`_closed_hamiltonian`).  Without it
    ``hamiltonian`` is applied to the closed set as it is.

    ``observable(configs [m, U]) -> [m, D]``, when given: its ensemble average by the same increments as the charges
    (``expectation [D]``, with the graph; SPEC 38 passes the band unit's state indicator).

    ``energy(configs [m, U]) -> [m]``: the exact whole-system energy of each configuration (with the graph);
    ``hamiltonian(E [m], configs [m, U]) -> [m, m]``: a cluster's Hamiltonian on its own configurations.  A product
    configuration outside ``[L, K]`` (two clusters filling one reservoir) or not admissible for the covalent units
    does not exist and is left out of the pair space.  Two calls of ``energy`` in all."""
    device = n0.device
    n0 = n0.to(torch.long)
    U = int(n0.numel())
    K = max_charge.to(torch.long).to(device)
    L = _lower(min_charge, K)
    res = (torch.zeros(U, dtype=torch.bool, device=device) if reservoir is None
           else reservoir.to(torch.bool).to(device))
    linked_sizes = [int(S.shape[0]) for S in clusters]
    if closure:
        clusters = [_closure(n0, S.to(torch.long), res, L, K, units, frames, int(cap)) for S in clusters]
    own = torch.cat([n0.unsqueeze(0)] + [S[1:] for S in clusters], dim=0)
    E_own = energy(own)
    E0 = E_own[0]
    dtype = E0.dtype
    def q_of(configs: Tensor) -> Tensor:                  # the charges, then the observable's columns
        q = configs.to(dtype)
        return q if observable is None else torch.cat([q, observable(configs).to(dtype)], dim=1)

    q0 = q_of(n0.unsqueeze(0))[0]
    pop0 = _state_populations(n0.unsqueeze(0), torch.ones(1, dtype=dtype, device=device), states)
    E_c, H_c, F_c, q_c, pop_c = [], [], [], [], []
    at = 1
    for S, m_linked in zip(clusters, linked_sizes):
        m = int(S.shape[0])
        E = torch.cat([E0.reshape(1), E_own[at:at + m - 1]])
        at += m - 1
        if m == m_linked or bare is None:
            H = hamiltonian(E, S)
        else:
            H = _closed_hamiltonian(n0, S, m_linked, E, hamiltonian, bare, res)
        p = torch.diagonal(density_matrix(H, kT))
        E_c.append(E)
        H_c.append(H)
        F_c.append(canonical_free_energy(H, kT))
        q_c.append(p @ q_of(S))
        pop_c.append(_state_populations(S, p, states))
    F = E0
    charge = q0
    populations = pop0
    for c in range(len(clusters)):
        F = F + (F_c[c] - E0)
        charge = charge + (q_c[c] - q0)
        populations = populations + (pop_c[c] - pop0)
    # the pair spaces; the product configurations no cluster holds get their exact energies in one batch
    slots, products = [], []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            P = clusters[i].unsqueeze(1) + clusters[j].unsqueeze(0) - n0             # [m_i, m_j, U]
            valid = ((P >= L) & (P <= K)).all(-1)
            if units is not None:
                shape = valid.shape
                valid = valid & admissible(P.reshape(-1, U), units, frames).reshape(shape)
            inner = valid.clone()
            inner[0, :] = False
            inner[:, 0] = False
            slots.append((i, j, P, valid, inner))
            products.append(P[inner])
    count = sum(int(x.shape[0]) for x in products)
    E_prod = energy(torch.cat(products, dim=0)) if count else torch.zeros(0, dtype=dtype, device=device)
    pair_sizes, pair_index, pair_hamiltonians, pair_configs = [], [], [], []
    at = 0
    for (i, j, P, valid, inner), prod in zip(slots, products):
        m1, m2 = int(P.shape[0]), int(P.shape[1])
        k = int(prod.shape[0])
        rows, cols = inner.nonzero(as_tuple=True)
        exact = torch.zeros(m1, m2, dtype=dtype, device=device).index_put((rows, cols), E_prod[at:at + k])
        at += k
        # row 0 and column 0 are the clusters' own configurations (E_c' - E0 is exactly 0 at t = 0)
        grid = torch.where(inner, exact, E_c[i].unsqueeze(1) + (E_c[j] - E0).unsqueeze(0))
        O1 = H_c[i] - torch.diag(torch.diagonal(H_c[i]))
        O2 = H_c[j] - torch.diag(torch.diagonal(H_c[j]))
        eye1 = torch.eye(m1, dtype=dtype, device=device)
        eye2 = torch.eye(m2, dtype=dtype, device=device)
        H = torch.diag(grid.reshape(-1)) + torch.kron(O1, eye2) + torch.kron(eye1, O2)
        keep = valid.reshape(-1).nonzero().flatten()
        H = H[keep][:, keep]
        configs = P.reshape(-1, U)[keep]
        p = torch.diagonal(density_matrix(H, kT))
        F = F + (canonical_free_energy(H, kT) - F_c[i] - F_c[j] + E0)
        charge = charge + (p @ q_of(configs) - q_c[i] - q_c[j] + q0)
        populations = populations + (_state_populations(configs, p, states) - pop_c[i] - pop_c[j] + pop0)
        pair_sizes.append(int(keep.numel()))
        pair_index.append((i, j))
        pair_hamiltonians.append(H)
        pair_configs.append(configs)
    expectation = None if observable is None else charge[U:]
    return ClusterEnsemble(energy=F, charge=charge[:U], populations=populations, reference=n0,
                           clusters=list(clusters), hamiltonians=H_c, pair_sizes=pair_sizes, expectation=expectation,
                           pair_index=pair_index, pair_hamiltonians=pair_hamiltonians, pair_configs=pair_configs)


def cluster_ensemble(e_hat: Tensor, states: Tensor, b: Tensor, W: Tensor, max_charge: Tensor, Q: int, *,
                     window: float, cap: int, T: Tensor | None = None, amplitudes=(0.05, 0.2), kT: float = 0.1,
                     selection: str = "amplitude", start: Tensor | None = None, min_charge: Tensor | None = None,
                     units: Tensor | None = None, frames: Tensor | None = None, pairs=None,
                     electrons: Tensor | None = None, reservoir: Tensor | None = None, diag_correction=None,
                     search=None, closure: bool = True, steps=None) -> ClusterEnsemble:
    """The cluster product ensemble of the quadratic configuration energy (``diabatic.configuration_energies``, plus
    ``diag_correction(configs) -> [m]`` when given): the linked selection, then :func:`cluster_free_energy`.
    ``selection`` 'amplitude' (SPEC 21's rule and switch; needs ``T``) or 'window' (every coupling at full strength)."""
    if selection not in ("window", "amplitude"):
        raise ValueError(f"selection must be 'window' or 'amplitude', got {selection!r}")
    amplitude = selection == "amplitude" and T is not None
    a_off, a_on = float(amplitudes[0]), float(amplitudes[1])
    detached = None if pairs is None else (pairs[0], pairs[1], pairs[2].detach())
    n0, clusters = linked_configurations(e_hat, states, b, W, max_charge, int(Q), window=float(window), cap=int(cap),
                                         T=T if amplitude else None, amplitudes=(a_off, a_on), kT=float(kT),
                                         start=start, min_charge=min_charge, units=units, frames=frames,
                                         pairs=detached, electrons=electrons, reservoir=reservoir, search=search,
                                         couplings=T, closure=closure, steps=steps)

    def energy(configs: Tensor) -> Tensor:
        E = configuration_energies(configs, e_hat, states, b, W, pairs=pairs)
        return E if diag_correction is None else E + diag_correction(configs)

    def hamiltonian(E: Tensor, configs: Tensor) -> Tensor:
        if T is None:
            return torch.diag(E)
        if amplitude:
            return switched_hamiltonian(E, configs, max_charge, T, a_off, a_on, float(kT), electrons=electrons, steps=steps)
        return diabatic_hamiltonian(E, configs, max_charge, T, electrons=electrons, steps=steps)

    def bare(E: Tensor, configs: Tensor) -> Tensor:
        return diabatic_hamiltonian(E, configs, max_charge, T, electrons=electrons, steps=steps)

    return cluster_free_energy(n0, clusters, energy, hamiltonian, float(kT), max_charge, min_charge, states,
                               units=units, frames=frames, reservoir=reservoir, closure=closure,
                               bare=bare if amplitude else None, cap=int(cap))
