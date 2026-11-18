"""The downfolded O 2p <-> metal s bond (trace_redox SPEC 32) — NEW-TO-TRACE.

A strong local charge-transfer bond (O 2p -> Au 6s at the O-top contact of Au2 on MgO: PBE's transfer -1.10 eV of a
-1.96 eV relaxation, 94 % of the transfer's wetting share O -> Au) cannot live in the selected configuration ensemble
at its physical coupling (measured, step 21: the selection stops converging, the SPEC 21 switch adds non-physical
force, short contacts overbind).  It is folded into the configuration energies instead (Loewdin partitioning,
ESTABLISHED: P.-O. Loewdin, J. Chem. Phys. 19, 1396 (1951)):

- the channel: every one-electron move between an O-class atom (a frame anion with a p frontier) and an s-class atom
  (a neutral frame with an odd electron count and an s frontier: Au, Ag, Cu, H), either direction, any ladder states;
  the module zeroes these pairs' explicit couplings and drops a selected configuration that is reachable from another
  by such a move;
- per explicit configuration n and per contact cluster (atoms connected by O-s edges within the cutoff and by s-s
  pairs within it), the folded states q are the single-move configurations, Delta_q(n) = E(q) - E(n) exact, with
      v_q = f_spin sqrt(g_orb) V(r) ,      V(r) = eta_sp hbar^2 / (m (r^2 + r0^2)) w(r)
  (Harrison's V_sp sigma, ESTABLISHED: Froyen and Harrison, Phys. Rev. B 20, 2420 (1979); r0 = 0.5 A the SPEC 27
  regulariser, which lowers V by 4.7 % at 2.26 A; w the C^2 window on [r_cut - 0.5 A, r_cut]; f_spin SPEC 29's
  factor; g_orb the isotropic occupation (donation) or vacancy (acceptance) fraction of the sigma p orbital);
  a closed-shell O donor (O2-) is resolved into its three p components, v_m = f V l_m with l the direction cosines
  O -> s (Slater-Koster, ESTABLISHED: Phys. Rev. 94, 1498 (1954)), so that sum_m v_m^2 = f^2 V^2, and the moved
  electron hops between the cluster's s atoms inside the folded manifold with hop_sign |T_ss'| f (the model's own
  s-s coupling, magnitude only; hop_sign -1, legacy: the bonding two-site state lowest; +1, the Pauli ordering of
  2026-10-29: the antibonding one lowest, since the pair's own s electrons fill the bonding orbital.  The sign is a
  gauge only for a donor coupled through one path; a donor coupled to both atoms of the pair fixes it);
- lambda(n, cluster) = the lowest eigenvalue of [[0, v^T], [v, M]], M = diag(Delta) + T_QQ, from the secular equation
  lambda = v^T (lambda - M)^-1 v below mu_min (the lowest eigenvalue of M; the left side minus the right is increasing
  there), by bracketed bisection on (min(0, mu_min) - ||v|| - 1, mu_min) and two Newton steps with the graph from the
  detached root (exact value, first and second derivatives); E(n) -> E(n) + sum_clusters lambda.

Properties: size-consistent (one root per cluster, additive over clusters), bounded (lambda >= min(0, mu_min) -
||v||), -sum v^2 / Delta at weak coupling, state-dependent (an Au- cannot accept, an Au+ accepts with the pairing
factor sqrt 2).  Valid while mu_min > 0: the folded manifold above the explicit state (the Loewdin P/Q separation);
a configuration that violates it is a folded state of something lower, so the selection excludes it and the final
solve refuses loudly (:class:`DownfoldError`).  Stated approximations: the hops among the folded states are the s-s
ones only (a hole does not hop between O donors inside the manifold: the learned O-O coupling is unconstrained by
neutral data); isotropic leaves carry no hop; a cluster that splits into disconnected parts in a given
configuration is still one root (mutually exclusive rather than additive: an under-estimate); the self-energy is
evaluated at each explicit state's own energy, exact when one explicit state dominates.
"""
from __future__ import annotations

import math
from typing import Callable, NamedTuple

import torch
from torch import Tensor

#: Harrison's universal V_sp sigma coefficient (Froyen and Harrison 1979)
ETA_SP_SIGMA = 1.84
#: the window switch width (A): the switch acts on [r_cut - SWITCH_WIDTH, r_cut].  Measured against the library's
#: PBE0 monomers (theory note 2026-10-18 section 4c), |<Au 6s|O 2p sigma>|(r) / S(2.26 A) = 0.62 / 0.53 / 0.44 /
#: 0.37 / 0.30 / 0.25 at 3.0 / 3.2 / 3.4 / 3.6 / 3.8 / 4.0 A against 1/(r^2 + r0^2)'s 0.57 / 0.50 / 0.44 / 0.39 /
#: 0.35 / 0.32: Harrison's form holds within 10 % to 3.6 A, so the switch sits beyond it (r_cut 4.5 A, window 0.5 A).
SWITCH_WIDTH = 0.5
#: bisection steps: the bracket (at most tens of eV) shrinks below double precision
BISECTION_STEPS = 60
#: padded (absent) leaves sit at this energy with v = 0: never the lowest, never coupled
PAD_ENERGY = 1.0e6


class DownfoldError(RuntimeError):
    """The folded manifold is not above the explicit state (mu_min <= 0): SPEC 32's P/Q separation fails."""

PROMOTION_TARGET = 0.6   #: the explicit weight a promotion pass restores (the dominant folded states, by weight)
PROMOTION_BATCH = 32     #: at most this many folded states promoted in one pass


class PromotionRequest(DownfoldError):
    """The exact block's ground state is not explicit-dominated: the dominant folded states must be promoted to
    explicit configurations.  ``rows`` (their charge configurations), ``os`` / ``kinds`` (donor O, hole component),
    ``energies`` [n], ``couplings`` [n, C] to every explicit configuration, ``mutual`` [n, n] (their hops to each
    other), ``weights`` (in the block's ground state), in descending weight: the set whose weight restores the
    explicit weight to ``PROMOTION_TARGET`` (2026-10-30: a close-contact snapshot needed nine, the p_z states of
    four O's on either Au, each of a tenth of the weight)."""

    def __init__(self, message, cluster, rows, os_, kinds, energies, couplings, mutual, weights):
        super().__init__(message)
        self.cluster = int(cluster)
        self.rows = [tuple(int(x) for x in r) for r in rows]
        self.os, self.kinds = [int(x) for x in os_], [int(x) for x in kinds]
        self.energies, self.couplings, self.mutual = energies, couplings, mutual
        self.weights = [float(w) for w in weights]

    @property
    def row(self):
        return self.rows[0]

    @property
    def o(self):
        return self.os[0]

    @property
    def kind(self):
        return self.kinds[0]

    @property
    def weight(self):
        return self.weights[0]



class DownfoldMoves(NamedTuple):
    o: Tensor               #: [P] O-class atom of each edge
    g: Tensor               #: [P] s-class atom of each edge
    t: Tensor               #: [P] V(r) on the edge, eV (differentiable)
    l: Tensor               #: [P, 3] unit vector o -> g (differentiable)
    component: Tensor       #: [P] contact cluster of the edge
    atom_component: Tensor  #: [n] contact cluster of every atom (-1: none)
    atom_bound: Tensor      #: [n] a bound on how much one move touching the atom can change the dressing, eV
    outward: Tensor         #: [n, 3] the sigma* lobe direction of a bonded s-class atom (zero: no lobe)
    cf_shift: Tensor        #: [n, 3] the crystal-field shift of each p-component hole on an O-class atom, eV
    free: Tensor            #: [n, 3] bool: the p axes of an O-class atom that couple (its dangling directions)

    def anisotropy(self, kappa: float) -> Tensor:
        """``[P]`` (1 + kappa cos theta) / sqrt(1 + kappa^2) per edge, cos theta = (g -> o) . outward[g]; 1 where the
        acceptor has no lobe.  The sp-hybridised sigma* of a bonded s-class acceptor (SPEC 32, measured)."""
        if int(self.o.numel()) == 0:
            return self.t.new_zeros(0)
        e = self.outward[self.g]
        has = (e * e).sum(1) > 0.25
        cos = (-self.l * e).sum(1)
        factor = (1.0 + kappa * cos) / math.sqrt(1.0 + kappa * kappa)
        return torch.where(has, factor, torch.ones_like(factor))


def frontier_shells(species: Tensor) -> Tensor:
    """``[n]`` the frontier shell per atom (0 s, 1 p, -1 none), one lookup per element."""
    from .heads import frontier_shell
    z = torch.as_tensor(species).to(torch.long)
    table = {int(v): frontier_shell(int(v)) for v in torch.unique(z).tolist()}
    return torch.tensor([-1 if table[int(v)] is None else table[int(v)] for v in z.tolist()], dtype=torch.long,
                        device=z.device)


def donor_mask(species: Tensor) -> Tensor:
    """``[n]`` bool: the O class (frame anions with a p frontier: O2-, halides)."""
    from .constants import formal_frame
    z = torch.as_tensor(species).to(torch.long)
    return (formal_frame(z).to(z.device) < 0) & (frontier_shells(z) == 1)


def acceptor_mask(species: Tensor) -> Tensor:
    """``[n]`` bool: the s class (neutral frames with an odd electron count and an s frontier: Au, Ag, Cu, H)."""
    from .constants import formal_frame
    from .heads import open_frame
    z = torch.as_tensor(species).to(torch.long)
    return (formal_frame(z).to(z.device) == 0) & open_frame(z) & (frontier_shells(z) == 0)


def _union_find(n: int, pairs: Tensor) -> Tensor:
    """``[n]`` component labels (0..) from undirected pairs ``[2, E]``."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs.t().tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = [find(i) for i in range(n)]
    label = {r: k for k, r in enumerate(sorted(set(roots)))}
    return torch.tensor([label[r] for r in roots], dtype=torch.long)


def _cation_axis_counts(species: Tensor, edge_index: Tensor, edge_vec: Tensor, edge_len: Tensor,
                        bond_scale: float) -> tuple[Tensor, Tensor]:
    """``(count [n, 3], has [n])``: for each O-class atom, its frame-cation neighbours (within ``bond_scale`` times
    its nearest cation distance) along each Cartesian axis (|l . e_m| > cos 30 deg), and whether it has any."""
    from .constants import formal_frame
    z = torch.as_tensor(species).to(torch.long).to(edge_index.device)
    n = int(z.numel())
    count = edge_len.new_zeros(n, 3)
    f = formal_frame(z).to(z.device)
    src, dst = edge_index[0], edge_index[1]
    keep = donor_mask(z)[src] & (f[dst] > 0) & (src != dst)
    if not bool(keep.any()):
        return count, torch.zeros(n, dtype=torch.bool, device=z.device)
    o, r = src[keep], edge_len[keep]
    l = -edge_vec[keep] / r.unsqueeze(1)                                     # o -> cation
    nearest = torch.full((n,), float("inf"), dtype=r.dtype, device=z.device).scatter_reduce(0, o, r, "amin")
    close = r <= bond_scale * nearest[o]
    along = (l.abs() > math.cos(math.radians(30.0))).to(r.dtype)              # [E, 3]
    count = count.index_add(0, o[close], along[close])
    return count, count.sum(1) > 0


def free_axes(species: Tensor, edge_index: Tensor, edge_vec: Tensor, edge_len: Tensor,
              bond_scale: float = 1.2) -> Tensor:
    """``[n, 3]`` bool: an O-class atom's dangling p axes, those with the fewest frame-cation neighbours (all three
    when it has none or the counts tie); every axis of any other atom.  Measured (2026-10-29): the O 2p components
    that bond to cations do not donate at the strength their overlap suggests; a (100) surface O couples through
    p_z alone."""
    count, has = _cation_axis_counts(species, edge_index, edge_vec, edge_len, bond_scale)
    least = count.min(1, keepdim=True).values
    free = count <= least
    return torch.where(has.unsqueeze(1), free, torch.ones_like(free))


def crystal_field_shifts(species: Tensor, edge_index: Tensor, edge_vec: Tensor, edge_len: Tensor, beta: float,
                         bond_scale: float = 1.2) -> Tensor:
    """``[n, 3]`` the extra cost (eV) of a hole in the p_x, p_y, p_z component of each O-class atom: ``beta`` per
    cation neighbour along the axis beyond the least-coordinated axis.  Neighbours: frame cations within
    ``bond_scale`` times the atom's nearest cation distance; along an axis when |l . e_m| > cos 30 deg.  A bulk O
    (two per axis) gets no shift; a (100) surface O (one below, two along each in-plane axis) gets ``beta`` on its
    in-plane components (measured 0.8-1.3 eV on MgO(100): the band's hole energies)."""
    n = int(torch.as_tensor(species).numel())
    if beta == 0.0:
        return edge_len.new_zeros(n, 3)
    count, has = _cation_axis_counts(species, edge_index, edge_vec, edge_len, bond_scale)
    base = torch.where(has, count.min(1).values, torch.zeros_like(count[:, 0]))
    return beta * (count - base.unsqueeze(1))


DECAY_RAMP = 0.2       #: A, the softplus width that turns the measured decay on above the anchor distance


def decay_factor(r: Tensor, decay: float, anchor: float, width: float = DECAY_RAMP) -> Tensor:
    """``exp(-decay * s(r - anchor))`` with ``s(x) = width * softplus(x / width)``: Harrison's coupling at and below
    the anchor (s -> 0), the measured decay ``exp(-decay (r - anchor))`` well above it, C^1 in between with a slope
    bounded by ``decay``; the anchor contact itself is reduced by exp(-decay width ln 2) (6 % at the measured
    constants).  The measurement (2026-10-29) covered 2.26-3.5 A only; no short-range extrapolation."""
    x = (r - float(anchor)) / float(width)
    return torch.exp(-float(decay) * float(width) * torch.nn.functional.softplus(x))


def downfold_moves(species: Tensor, edge_index: Tensor, edge_vec: Tensor, edge_len: Tensor,
                   r_cut: float, bond_cutoff: float = 3.2, beta: float = 0.0,
                   bond_scale: float = 1.2, free_axes_only: bool = False,
                   units: Tensor | None = None, decay: float = 0.0, anchor: float = 2.26) -> DownfoldMoves:
    """The O-s edges within ``r_cut`` (each image its own edge, with its own direction), their couplings, the contact
    clusters (O-s edges and s-s pairs within ``r_cut``), the partner states (``t = 0``: an s-class atom of the
    cluster the donor does not reach, held in the fold for the s-s hops) and the per-atom dressing bound.
    ``edge_vec`` is ``pos[src] - pos[dst]`` (+ shift), the model's convention.  ``decay`` (1/A, >= 0) multiplies
    Harrison's V by :func:`decay_factor`: the measured radial form of the O 2p - Au 6s coupling, e^(-1.1 r) against
    Harrison's 1/r^2 (the ALMO vertical series read against its own Delta; 2026-10-29), turned on above the anchor
    (the O-top distance) by a softplus ramp and Harrison's below it; zero is Harrison."""
    from .band import HBAR2_OVER_ME
    if not float(decay) >= 0.0:
        raise ValueError(f"decay must be >= 0 (1/A), got {decay!r}")
    from .heads import BOUND_R0, window_switch
    z = torch.as_tensor(species).to(torch.long).to(edge_index.device)
    n = int(z.numel())
    src, dst = edge_index[0], edge_index[1]
    is_o, is_g = donor_mask(z), acceptor_mask(z)
    close = (edge_len < r_cut) & (src != dst)
    if units is not None:                                                     # a pair inside one covalent unit is a
        u = units.to(torch.long).to(z.device)[:n]                             # bound-sector bond, never folded
        close = close & (u[src] != u[dst])
    og = is_o[src] & is_g[dst] & close                          # edge from the O to the s atom
    go = is_g[src] & is_o[dst] & close                          # the reverse orientation of the same pair
    keep = og | go
    o = torch.where(og, src, dst)[keep]
    g = torch.where(og, dst, src)[keep]
    r = edge_len[keep]
    sign = torch.where(og[keep], -torch.ones_like(r), torch.ones_like(r))   # o -> g = pos[g] - pos[o]
    l = sign.unsqueeze(1) * edge_vec[keep] / r.unsqueeze(1)
    t = ETA_SP_SIGMA * HBAR2_OVER_ME / (r * r + BOUND_R0 * BOUND_R0) * window_switch(r, r_cut, SWITCH_WIDTH)
    if float(decay) > 0.0:
        t = t * decay_factor(r, float(decay), float(anchor))
    # each (o, g, image) appears in both edge orientations: keep one of each
    key = torch.stack([o, g, (l * 1e6).round().to(torch.long).sum(1)])
    _, inverse = torch.unique(key, dim=1, return_inverse=True)
    first = torch.zeros(int(inverse.max()) + 1 if int(inverse.numel()) else 0, dtype=torch.long, device=z.device)
    first = first.index_put((inverse,), torch.arange(int(inverse.numel()), device=z.device), accumulate=False)
    pick = torch.zeros(int(o.numel()), dtype=torch.bool, device=z.device)
    pick[first] = True
    o, g, t, l = o[pick], g[pick], t[pick], l[pick]
    gg = is_g[src] & is_g[dst] & close
    links = torch.cat([torch.stack([o, g]), torch.stack([src[gg], dst[gg]])], dim=1)
    atom_component = torch.full((n,), -1, dtype=torch.long, device=z.device)
    touched = torch.unique(links.flatten()) if int(links.numel()) else torch.zeros(0, dtype=torch.long, device=z.device)
    if int(touched.numel()):
        local = _union_find(int(touched.numel()), torch.searchsorted(touched, links))
        atom_component[touched] = local.to(z.device)
    component = atom_component[o] if int(o.numel()) else torch.zeros(0, dtype=torch.long, device=z.device)
    # partner states (2026-10-29): every s-class atom of a contact cluster holds a folded state for each of the
    # cluster's donors, coupled through the s-s hops even where the donor does not reach it (zero coupling, t = 0,
    # l = 0): the two-site states of a bonded pair exist whichever atom the donor touches.  Without them the fold
    # of an O touching one atom of a pair had no two-site structure (the on-axis test: 36 meV on its toy).
    if int(o.numel()):
        o_all = torch.unique(o)
        g_all = (is_g & (atom_component >= 0)).nonzero(as_tuple=True)[0]
        oo = o_all.repeat_interleave(int(g_all.numel()))
        gg_all = g_all.repeat(int(o_all.numel()))
        same = atom_component[oo] == atom_component[gg_all]
        missing = same & ~torch.isin(oo * n + gg_all, o * n + g)
        if bool(missing.any()):
            oz, gz = oo[missing], gg_all[missing]
            o, g = torch.cat([o, oz]), torch.cat([g, gz])
            t = torch.cat([t, torch.zeros(int(oz.numel()), dtype=t.dtype, device=z.device)])
            l = torch.cat([l, torch.zeros(int(oz.numel()), 3, dtype=l.dtype, device=z.device)])
            component = torch.cat([component, atom_component[oz]])
    # |lambda| <= ||v|| per cluster, with v^2 <= 2 (1 + 1/3) t^2 per edge (pairing factor, both directions)
    n_comp = int(atom_component.max()) + 1 if int(touched.numel()) else 0
    comp_norm = torch.zeros(n_comp, dtype=t.dtype, device=z.device)
    if n_comp:
        comp_norm = comp_norm.index_add(0, component, (8.0 / 3.0) * t.detach() ** 2).sqrt()
    atom_bound = torch.zeros(n, dtype=t.dtype, device=z.device)
    if n_comp:
        atom_bound = torch.where(atom_component >= 0, comp_norm[atom_component.clamp(min=0)], atom_bound)
    # the sigma* lobe: an s-class atom bonded to s-class partners points its lobe away from them
    bonded = is_g[src] & is_g[dst] & (edge_len < bond_cutoff) & (src != dst)
    outward = torch.zeros(n, 3, dtype=edge_vec.dtype, device=z.device)
    if bool(bonded.any()):
        u = edge_vec[bonded] / edge_len[bonded].unsqueeze(1)                 # pos[src] - pos[dst]: away from the partner
        summed = outward.index_add(0, src[bonded], u)
        # the norm on the bonded atoms with a net lobe only: a norm over the zero rows (every other atom, or a
        # symmetric site) is masked at first order but 0/0 at second, the seed-42 failure of step 22 (2026-10-29)
        rows = torch.unique(src[bonded])
        n2 = (summed[rows] * summed[rows]).sum(1)
        sel = n2 > 0.25
        if bool(sel.any()):
            lobe = summed[rows][sel] / torch.sqrt(n2[sel]).unsqueeze(1)
            outward = outward.index_put((rows[sel],), lobe)
    cf_shift = crystal_field_shifts(z, edge_index, edge_vec, edge_len, beta, bond_scale)
    free = (free_axes(z, edge_index, edge_vec, edge_len, bond_scale) if free_axes_only
            else torch.ones(n, 3, dtype=torch.bool, device=z.device))
    return DownfoldMoves(o, g, t, l, component, atom_component, atom_bound, outward, cf_shift, free)


def block_ground(M: Tensor, v: Tensor, mask: Tensor, strict: bool = True) -> tuple[Tensor, Tensor]:
    """``(lambda [S], mu_min [S])``: the lowest eigenvalue of each block ``[[0, v^T], [v, M]]`` (``M [S, K, K]``
    symmetric, ``v [S, K]``, ``mask [S, K]`` the present leaves) and the lowest eigenvalue of ``M`` on the present,
    finite leaves joined to a coupled leaf by nonzero hops.  A leaf with a non-finite diagonal, or one no chain of
    hops joins to a coupled leaf, is absent.  Blocks with no coupled leaf give 0.  Blocks with
    ``mu_min <= 0`` are refused (``strict``) or returned as +inf (the selection's exclusion); a block with no coupled
    leaf has no manifold and reports ``mu`` NaN."""
    S, K = int(M.shape[0]), int(M.shape[1])
    dtype, device = M.dtype, M.device
    diag = torch.diagonal(M, dim1=1, dim2=2)
    present = mask & torch.isfinite(diag)
    coupled = present & (v != 0)
    # a present leaf matters only if a chain of nonzero hops joins it to a coupled leaf (a partner state whose s-s
    # coupling is zero is unreachable: absent, and it must not bound the root or stand in the solve's way)
    offdiag = (M != 0) & present.unsqueeze(1) & present.unsqueeze(2)
    offdiag = offdiag & ~torch.eye(K, dtype=torch.bool, device=device)
    reach = coupled.clone()
    for _ in range(K):
        grown = reach | (offdiag & reach.unsqueeze(1)).any(2)
        if bool(torch.equal(grown, reach)):
            break
        reach = grown
    present = present & reach
    has = coupled.any(1)
    big = torch.full((), PAD_ENERGY, dtype=dtype, device=device)
    pm = present.unsqueeze(1) & present.unsqueeze(2)
    Mp = torch.where(pm, M, torch.zeros_like(M))
    Mp = Mp + torch.diag_embed(torch.where(present, torch.zeros_like(diag), big.expand_as(diag)))
    vp = torch.where(coupled, v, torch.zeros_like(v))
    with torch.no_grad():
        mu = torch.linalg.eigvalsh(Mp.detach())[:, 0]
        mu = torch.where(has, mu, torch.full_like(mu, float("nan")))      # no manifold: no mu (the record skips it)
    # a block with no coupled leaf, or whose folded manifold is not above its state (refused below, or +inf), is
    # solved on the padded matrix alone: its secular function may have no root below mu (an uncoupled lowest
    # state), where the bisection would run onto an eigenvalue and the refinement's solve turn singular
    eye = torch.eye(K, dtype=dtype, device=device)
    ok = has & (mu > 0)
    Mp = torch.where(ok.unsqueeze(1).unsqueeze(2), Mp, big * eye)
    if bool((has & (mu <= 0)).any()):
        if strict:
            k = int((has & (mu <= 0)).nonzero()[0])
            pk = present[k]
            dk = diag[k][pk].detach().tolist()
            vk = v[k][pk].detach().tolist()
            hk = M[k][pk][:, pk].detach()
            hk = (hk - torch.diag(torch.diagonal(hk))).abs().max(dim=1).values.tolist()
            raise DownfoldError(f"SPEC 32: the folded manifold of block {k} is not above the explicit state "
                                f"(mu_min = {float(mu[k]):.4f} eV): the charge-transfer state must be explicit here; "
                                f"leaves (Delta, v, max |hop|): "
                                + ", ".join(f"({d:.3f}, {w:.3f}, {h:.3f})" for d, w, h in zip(dk, vk, hk)))

    def f_df(lam, Mx, vx):
        x = torch.linalg.solve(lam.unsqueeze(1).unsqueeze(2) * eye - Mx, vx.unsqueeze(2)).squeeze(2)
        return lam - (vx * x).sum(1), 1.0 + (x * x).sum(1)

    vp = torch.where(ok.unsqueeze(1), vp, torch.zeros_like(vp))
    with torch.no_grad():
        hi = torch.where(ok, mu, torch.zeros_like(mu))
        lo = torch.minimum(torch.zeros_like(hi), hi) - vp.norm(dim=1) - 1.0
        for _ in range(BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            fm, _ = f_df(torch.where(ok, mid, torch.zeros_like(mid)), Mp.detach(), vp.detach())
            below = fm < 0
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)
        lam0 = 0.5 * (lo + hi)
        lam0 = torch.minimum(lam0, torch.nextafter(torch.where(ok, mu, torch.zeros_like(mu)),
                                                   torch.full_like(mu, -float("inf"))))
    lam = torch.where(ok, lam0, torch.zeros_like(lam0))
    for _ in range(2):
        f, df = f_df(lam, Mp, vp)
        lam = torch.where(ok, lam - f / df, lam)
    lam = torch.where(has & (mu <= 0), torch.full_like(lam, float("inf")), lam)
    return lam, mu


def star_ground(delta: Tensor, v2: Tensor) -> Tensor:
    """``[S]``: the lowest root of each star ``[[0, v], [v, diag(delta)]]`` (``v2 = v^2``; a leaf with ``v2 = 0`` or
    a non-finite ``delta`` is absent); 0 where no leaf is coupled.  :func:`block_ground` with a diagonal ``M``."""
    present = v2 > 0
    v = torch.sqrt(torch.where(present, v2, torch.ones_like(v2)))         # no sqrt at zero (its backward is infinite)
    lam, _ = block_ground(torch.diag_embed(delta), torch.where(present, v, torch.zeros_like(v)), present, strict=True)
    return lam


def _orbital_factors(n_o: Tensor, f_o: Tensor) -> tuple[Tensor, Tensor]:
    """``(give, take)``: the isotropic fraction of the sigma p orbital that can donate (its occupation / 2) and accept
    (its vacancy / 2), from the O state's 2p count ``6 - (n_o - f_o)``."""
    n_p = (6 - (n_o - f_o)).clamp(0, 6).to(torch.float64)
    return n_p / 6.0, (6.0 - n_p) / 6.0


def _leaves(configs: Tensor, energy: Callable[[Tensor], Tensor], moves: DownfoldMoves, frames: Tensor,
            min_charge: Tensor | None, max_charge: Tensor, electrons: Tensor, base: Tensor | None, scale: float,
            anisotropy: float) -> dict | None:
    """The folded states ("leaves") of every configuration: per leaf its configuration ``cs``, move ``ps``, kind
    ``ks`` (0-2 the resolved p component, 3 / 4 the isotropic give / take), energy above its configuration ``delta``
    and coupling ``v``; the moved configurations ``moved[inv_move]``.  None when there is no leaf.  Shared by the
    per-state :func:`dressing` and the exact :func:`dressing_exact`."""
    dtype = moves.t.dtype if int(moves.t.numel()) else torch.float64
    device = configs.device
    n = configs.to(torch.long)
    f = frames.to(torch.long).to(device)
    z = electrons.to(torch.long).to(device)
    K = max_charge.to(torch.long).to(device)
    L = min_charge.to(torch.long).to(device) if min_charge is not None else torch.full_like(K, -(10 ** 6))
    o, g, t, l, comp = moves.o, moves.g, moves.t, moves.l, moves.component
    P = int(o.numel())
    n_o, n_g = n[:, o], n[:, g]                                              # [C, P]
    f_o, f_g = f[o], f[g]
    same_parity = ((z[o] - n_o) % 2) == ((z[g] - n_g) % 2)
    spin = torch.where(same_parity, math.sqrt(2.0), 1.0).to(dtype)
    give, take = _orbital_factors(n_o, f_o.unsqueeze(0))
    resolved = n_o == f_o.unsqueeze(0)                                      # the closed-shell O donor
    fwd = (n_o + 1 <= K[o]) & (n_g - 1 >= L[g])                              # O gives
    rev = (n_g + 1 <= K[g]) & (n_o - 1 >= L[o]) & (take > 0)                 # O takes
    # leaves: (c, p, kind): kind 0..2 the resolved p component m, 3 the isotropic forward, 4 the isotropic reverse
    kinds = []
    for m in range(3):
        kinds.append((fwd & resolved, m, 1, 0))
    kinds.append((fwd & ~resolved, 3, 1, 0))
    kinds.append((rev, 4, -1, 1))
    cs, ps, ks, dirs = [], [], [], []
    for mask, kind, direction, _ in kinds:
        c_i, p_i = mask.nonzero(as_tuple=True)
        cs.append(c_i)
        ps.append(p_i)
        ks.append(torch.full_like(c_i, kind))
        dirs.append(torch.full_like(c_i, direction))
    cs, ps, ks, dirs = (torch.cat(x) for x in (cs, ps, ks, dirs))
    if int(cs.numel()) == 0:
        return None
    # the folded configurations and their exact energies (one evaluation per distinct (c, p, direction))
    move_key = (cs * P + ps) * 2 + (dirs > 0).to(torch.long)
    uniq_move, inv_move = torch.unique(move_key, return_inverse=True)
    uc, rest = uniq_move // (2 * P), uniq_move % (2 * P)
    up, udir = rest // 2, torch.where(rest % 2 == 1, 1, -1)
    rows = torch.arange(int(uc.numel()), device=device)
    moved = n[uc].clone()
    moved[rows, o[up]] += udir
    moved[rows, g[up]] -= udir
    # one energy per distinct moved configuration: the same folded state is reached from many explicit
    # configurations (73 at a random initialisation shared a few hundred placements over 1600 leaves; 2026-10-30)
    moved_u, inv_u = torch.unique(moved, dim=0, return_inverse=True)
    E_moved = energy(moved_u.to(configs.dtype))[inv_u]
    E_base = energy(configs) if base is None else base
    delta_u = E_moved - E_base[uc]
    delta = delta_u[inv_move]                                               # [N_leaves]
    cf = moves.cf_shift[o[ps], ks.clamp(max=2)]                             # the resolved component's hole shift
    delta = delta + torch.where(ks < 3, cf, torch.zeros_like(cf))
    # the couplings: the pair's scale and the acceptor's sigma* anisotropy on every leaf
    amp = spin[cs, ps] * t[ps] * scale * moves.anisotropy(anisotropy)[ps]
    factor = torch.where(ks == 4, take[cs, ps].sqrt(), torch.where(ks == 3, give[cs, ps].sqrt(), torch.ones_like(amp)))
    direction_cos = torch.where((ks < 3).unsqueeze(1), l[ps] * moves.free[o[ps]].to(l.dtype), torch.ones_like(l[ps]))
    m_index = ks.clamp(max=2)
    v = amp * factor * torch.where(ks < 3, direction_cos[torch.arange(int(ks.numel()), device=device), m_index],
                                   torch.ones_like(amp))
    return dict(n=n, z=z, K=K, L=L, o=o, g=g, t=t, l=l, comp=comp, P=P, cs=cs, ps=ps, ks=ks, dirs=dirs, delta=delta,
                v=v, moved=moved, inv_move=inv_move, E_base=E_base, dtype=dtype, device=device)


def dressing(configs: Tensor, energy: Callable[[Tensor], Tensor], moves: DownfoldMoves, T_hop: Tensor,
             frames: Tensor, min_charge: Tensor | None, max_charge: Tensor, electrons: Tensor,
             strict: bool = True, base: Tensor | None = None, scale: float = 1.0,
             anisotropy: float = 0.0, hop_sign: int = -1) -> tuple[Tensor, dict]:
    """``([C] sum_clusters lambda(n, cluster), record)`` for configurations ``[C, U]`` (atom units first).

    ``energy`` returns the explicit diagonal (every term but this one) for a batch of configurations; ``base`` is
    its value on ``configs`` when the caller has it.  ``T_hop [U, U]`` is the solver's coupling matrix (the s-s
    entries are used, as magnitudes).  ``electrons`` are the atoms' Z (SPEC 29 parities).  ``strict`` False returns
    +inf for a configuration whose folded manifold is not above it (the selection's exclusion) instead of raising.
    ``hop_sign`` orders the two-site states of a bonded s pair inside the folded manifold: -1 (legacy) puts the
    bonding combination lowest, +1 the antibonding one (the Pauli ordering: the pair's own s electrons fill the
    bonding orbital, so the extra electron takes sigma*; 2026-10-29, the flat acceptor run).  The sign is physical
    whenever one donor couples to both atoms (the loop phase sign(v_g T v_g')), a gauge for a single path.
    """
    if hop_sign not in (-1, 1):
        raise ValueError(f"hop_sign must be -1 (bonding lowest, legacy) or +1 (Pauli ordering), got {hop_sign!r}")
    C, U = int(configs.shape[0]), int(configs.shape[1])
    dtype = moves.t.dtype if int(moves.t.numel()) else torch.float64
    device = configs.device
    record = {"mu_min": float("nan"), "dressed": 0, "clusters": 0}
    zero = torch.zeros(C, dtype=dtype, device=device)
    if int(moves.o.numel()) == 0 or C == 0:
        return zero, record
    lv = _leaves(configs, energy, moves, frames, min_charge, max_charge, electrons, base, scale, anisotropy)
    if lv is None:
        return zero, record
    n, z, o, g, comp = lv["n"], lv["z"], lv["o"], lv["g"], lv["comp"]
    cs, ps, ks, delta, v, moved, inv_move = lv["cs"], lv["ps"], lv["ks"], lv["delta"], lv["v"], lv["moved"], lv["inv_move"]
    # blocks: (configuration, cluster); slots within a block
    block_key = cs * (int(comp.max()) + 1) + comp[ps]
    uniq_block, inv_block = torch.unique(block_key, return_inverse=True)
    S = int(uniq_block.numel())
    order = torch.argsort(inv_block, stable=True)
    counts = torch.bincount(inv_block, minlength=S)
    start = torch.cumsum(counts, 0) - counts
    slot = torch.empty_like(inv_block)
    slot[order] = torch.arange(int(inv_block.numel()), device=device) - start[inv_block[order]]
    width = int(counts.max())
    M = torch.zeros(S, width, width, dtype=dtype, device=device)
    M = M + torch.diag_embed(torch.zeros(S, width, dtype=dtype, device=device).index_put((inv_block, slot), delta))
    V = torch.zeros(S, width, dtype=dtype, device=device).index_put((inv_block, slot), v)
    mask = torch.zeros(S, width, dtype=torch.bool, device=device).index_put((inv_block, slot), torch.ones_like(slot, dtype=torch.bool))
    # hops inside the folded manifold: resolved leaves of one O and one p component, electron on g <-> g'.
    # Vectorised over the groups (a Python loop over them cost a minute per forward at 74 explicit configurations,
    # 2026-10-30): the resolved leaves are sorted by their (c, o, m) key; a pair (i, i + d) of sorted neighbours with
    # the same key and different s atoms is a hop, enumerated for d = 1 .. (group size - 1).
    res = ks < 3
    if bool(res.any()):
        ri = res.nonzero(as_tuple=True)[0]
        key_r = (cs[ri] * U + o[ps[ri]]) * 3 + ks[ri]                          # (c, o, m) group: o < U atoms
        order_r = torch.argsort(key_r, stable=True)
        ri, key_r = ri[order_r], key_r[order_r]
        _, inv_g, size_g = torch.unique(key_r, return_inverse=True, return_counts=True)
        kmax = int(size_g.max())
        gi_all = g[ps[ri]]
        for d in range(1, kmax):
            a = torch.arange(int(ri.numel()) - d, device=device)
            b = a + d
            same = (key_r[a] == key_r[b]) & (gi_all[a] != gi_all[b])
            if not bool(same.any()):
                continue
            a, b = a[same], b[same]
            la, lb = ri[a], ri[b]
            ga, gb = gi_all[a], gi_all[b]
            hop = T_hop[ga, gb].abs()
            c_ab = cs[la]
            # the pairing factor of the hop g_b -> g_a in the folded state with the electron on g_b (SPEC 29)
            par_from = (z[gb] - (n[c_ab, gb] - 1)) % 2
            par_to = (z[ga] - n[c_ab, ga]) % 2
            fh = torch.where(par_from == par_to, math.sqrt(2.0), 1.0).to(dtype)
            val = float(hop_sign) * hop * fh
            blk = inv_block[la]
            M = M.index_put((blk, slot[la], slot[lb]), val, accumulate=True)
            M = M.index_put((blk, slot[lb], slot[la]), val, accumulate=True)
    try:
        lam, mu = block_ground(M, V, mask, strict=strict)
    except DownfoldError as err:                     # name the block's configuration and leaves (o, g, kind)
        k = int(str(err).split("block ")[1].split(" ")[0])
        members = (inv_block == k).nonzero(as_tuple=True)[0]
        c0 = int(cs[members[0]])
        desc = ", ".join(f"(o {int(o[ps[i]])}, g {int(g[ps[i]])}, kind {int(ks[i])}, Delta {float(delta[i]):.3f}, "
                         f"v {float(v[i]):.3f})" for i in members.tolist())
        raise DownfoldError(f"{err}; configuration {n[c0].tolist()}; leaves {desc}") from None
    out = zero.index_add(0, uniq_block // (int(comp.max()) + 1), lam)
    finite = torch.isfinite(mu)
    fin_d = torch.isfinite(delta)
    low = []                                                                # the six lowest leaves per configuration
    for c_i in torch.unique(cs).tolist():
        idx = (cs == c_i).nonzero(as_tuple=True)[0]
        low += idx[torch.argsort(delta.detach()[idx])[:6]].tolist()
    leaf = lambda i: (int(cs[i]), int(o[ps[i]]), int(g[ps[i]]), int(ks[i]), float(delta[i]), float(v[i]))
    record = {"lowest_leaves": [leaf(i) for i in low],
              "all_leaves": [leaf(i) for i in range(int(cs.numel()))] if int(cs.numel()) <= 256 else None,
              "all_leaves_full": ([leaf(i) + (moved[inv_move[i]].tolist(),) for i in range(int(cs.numel()))]
                                  if int(cs.numel()) <= 256 else None),
              "dressing": out.detach().tolist(),
              "mu_min": float(mu[finite].min()) if bool(finite.any()) else float("nan"),
              "delta_min": float(delta[fin_d].min()) if bool(fin_d.any()) else float("nan"),
              "dressed": int((lam != 0).sum()), "clusters": S, "leaves": int(delta.numel()),
              "lambda_min": float(lam[torch.isfinite(lam)].min()) if bool(torch.isfinite(lam).any()) else float("nan")}
    return out, record


def _ground_vector(H: Tensor) -> Tensor:
    """The ground eigenvector of a symmetric ``H`` as a function exact to first order in ``H``:
    g(H) = w0 - R (H w0 - (w0^T H w0) w0), R = sum_{j>0} w_j w_j^T / (lambda_j - lambda_0), with the eigenpairs taken
    at the evaluation point and held constant.  At the point g = w0, and dg = sum_j w_j (w_j^T dH w0) / (lambda_0 -
    lambda_j), the exact eigenvector derivative; so the Rayleigh quotient g^T H g / g^T g has the exact first AND
    second derivatives of lambda_0 — the eigenvector response the training's force loss needs (a detached w0 drops it;
    2026-10-30, step 22's force error rose epoch after epoch).  Only gaps from the ground eigenvalue enter (floored at
    1e-9 eV): degenerate excited pairs cannot produce 0/0, unlike an eigendecomposition in the graph."""
    with torch.no_grad():
        evals, evecs = torch.linalg.eigh(H.detach())
        w0 = evecs[:, 0]
        if int(H.shape[0]) > 1:
            gaps = (evals[1:] - evals[0]).clamp(min=1e-9)
            R = (evecs[:, 1:] / gaps) @ evecs[:, 1:].T
        else:
            R = torch.zeros_like(H)
    return w0 - R @ (H @ w0 - (w0 @ H @ w0) * w0)


def dressing_exact(configs: Tensor, energy: Callable[[Tensor], Tensor], moves: DownfoldMoves, T_hop: Tensor,
                   frames: Tensor, min_charge: Tensor | None, max_charge: Tensor, electrons: Tensor, H0: Tensor,
                   strict: bool = True, base: Tensor | None = None, scale: float = 1.0, anisotropy: float = 0.0,
                   hop_sign: int = -1, promoted: list | None = None, promote: bool = True) -> tuple[Tensor, dict]:
    """``(Sigma [C, C], record)``: the fold on the whole explicit block (2026-10-29).  ``H0 [C, C]`` is the explicit
    Hamiltonian of the configurations (their energies on the diagonal, the model's couplings off it).  The folded
    states are shared between configurations: one state per (hole configuration, O, component), reached from every
    configuration that can move an electron into it, coupled among themselves by the s-s hops (``hop_sign``, the
    pairing parities).  Per contact cluster k the fixed point E_k* is the lowest eigenvalue of
    [[H0, V_k], [V_k^T, H_Qk]] and Sigma_k = V_k (E_k* - H_Qk)^-1 V_k^T, so H0 + Sigma_k has E_k* as its lowest
    eigenvalue exactly (Loewdin at the block's own ground energy); Sigma = sum_k Sigma_k is additive over separated
    clusters and its diagonal is the per-state dressing when one configuration is explicit.  Refused (``strict``) or
    +inf when a cluster's ground state is not explicit-dominated (its folded manifold not above the explicit block).
    The record's ``mu_min`` is the smallest gap between a cluster's folded manifold and its block's ground energy
    (positive when valid; the per-state record's ``mu_min`` is the same quantity relative to the explicit state).
    ``promoted`` lists the explicit configurations that were folded states — (row, o, kind) — promoted on an earlier
    pass (2026-10-30): such a state is absent from the folded space and its hops to the remaining folded states are
    its couplings; a folded state equal to any other explicit configuration is absent too (its coupling is explicit
    already).  With ``promote`` a cluster whose ground state is not explicit-dominated raises
    :class:`PromotionRequest` naming its dominant folded state instead of refusing.
    """
    if hop_sign not in (-1, 1):
        raise ValueError(f"hop_sign must be -1 or +1, got {hop_sign!r}")
    C, U = int(configs.shape[0]), int(configs.shape[1])
    dtype = moves.t.dtype if int(moves.t.numel()) else torch.float64
    device = configs.device
    record = {"mu_min": float("nan"), "dressed": 0, "clusters": 0, "ground": [], "explicit_weight": []}
    zero = torch.zeros(C, C, dtype=dtype, device=device)
    if int(moves.o.numel()) == 0 or C == 0:
        return zero, record
    lv = _leaves(configs, energy, moves, frames, min_charge, max_charge, electrons, base, scale, anisotropy)
    if lv is None:
        return zero, record
    z, o, g, comp = lv["z"], lv["o"], lv["g"], lv["comp"]
    cs, ps, ks, delta, v, moved, inv_move, E_base = (lv["cs"], lv["ps"], lv["ks"], lv["delta"], lv["v"], lv["moved"],
                                                    lv["inv_move"], lv["E_base"])
    # a promoted configuration's own moves are not folded: promotion re-partitions the block it came from, it never
    # enlarges the folded space (its two-electron states, O- -> O0 with Au2(2-), are outside the fold's scope and
    # the model's trained charge states; a batch of four promotions reached them and fell 5.7 eV, 2026-10-30)
    prom_rows_early = {tuple(int(x) for x in r) for r, _, _ in (promoted or [])}
    if prom_rows_early:
        is_prom_cfg = torch.tensor([tuple(int(x) for x in configs[c].tolist()) in prom_rows_early for c in range(C)],
                                   device=device)
        keep_leaf = ~is_prom_cfg[cs]
        cs, ps, ks, delta, v, inv_move = cs[keep_leaf], ps[keep_leaf], ks[keep_leaf], delta[keep_leaf], v[keep_leaf], inv_move[keep_leaf]
    N = int(cs.numel())
    if N == 0:
        record.update(leaves=0, clusters=0, dressing=[0.0] * C, lowest_leaves=[], all_leaves=[], all_leaves_full=[],
                      folded_states=[], lambda_min=float("nan"), delta_min=float("nan"),
                      promoted=[list(r) + [o_, k_] for r, o_, k_ in sorted((tuple(int(x) for x in r), int(o_), int(k_)) for r, o_, k_ in (promoted or []))])
        return zero, record
    # the leaf record first, so that a refused (non-strict) call still returns a complete record
    record["leaves"] = N
    low = []
    for c_i in torch.unique(cs).tolist():
        ii = (cs == c_i).nonzero(as_tuple=True)[0]
        low += ii[torch.argsort(delta.detach()[ii])[:6]].tolist()
    leaf = lambda i: (int(cs[i]), int(o[ps[i]]), int(g[ps[i]]), int(ks[i]), float(delta[i]), float(v[i]))
    record["lowest_leaves"] = [leaf(i) for i in low]
    record["all_leaves"] = [leaf(i) for i in range(N)] if N <= 256 else None
    record["all_leaves_full"] = [leaf(i) + (moved[inv_move[i]].tolist(),) for i in range(N)] if N <= 256 else None
    fin_d = torch.isfinite(delta)
    record["delta_min"] = float(delta[fin_d].min()) if bool(fin_d.any()) else float("nan")
    record["dressing"] = [0.0] * C
    record["lambda_min"] = float("nan")
    # the folded states: one per (hole configuration, O, component)
    rows = torch.cat([moved[inv_move], o[ps].unsqueeze(1), ks.unsqueeze(1)], dim=1)
    qrows, inv_q = torch.unique(rows, dim=0, return_inverse=True)
    NQ = int(qrows.shape[0])
    # one representative leaf per folded state: every configuration reaching it gives the same energy, and an
    # index_put of all duplicates would hand the full gradient to each of them (forces 6-27 % off, 2026-10-30)
    first = torch.zeros(NQ, dtype=torch.long, device=device).index_put((inv_q,), torch.arange(N, device=device))
    E_q = (E_base[cs] + delta)[first]
    q_comp = torch.zeros(NQ, dtype=torch.long, device=device).index_put((inv_q,), comp[ps])
    V = torch.zeros(C, NQ, dtype=dtype, device=device).index_put((cs, inv_q), v, accumulate=True)
    # the hops: states of one (O, component) that differ by one electron between two s atoms of the cluster
    HQ = torch.diag(E_q)
    index = {tuple(r.tolist()): i for i, r in enumerate(qrows)}
    s_atoms = {}
    for gi, ci in zip(g.tolist(), comp.tolist()):
        s_atoms.setdefault(ci, set()).add(gi)
    hop_i, hop_j, hop_v = [], [], []
    for i in range(NQ):
        row = qrows[i]
        kind = int(row[U + 1])
        if kind >= 3:
            continue
        atoms = sorted(s_atoms.get(int(q_comp[i]), ()))
        for g_lose in atoms:
            for g_gain in atoms:
                if g_lose == g_gain or float(T_hop[g_lose, g_gain]) == 0.0:
                    continue
                target = row.clone()
                target[g_lose] += 1
                target[g_gain] -= 1
                j = index.get(tuple(target.tolist()))
                if j is None or j <= i:
                    continue
                same = ((int(z[g_lose]) - int(row[g_lose])) % 2) == ((int(z[g_gain]) - int(row[g_gain])) % 2)
                hop_i.append(i); hop_j.append(j)
                hop_v.append(float(hop_sign) * (math.sqrt(2.0) if same else 1.0) * T_hop[g_lose, g_gain].abs())
    if hop_i:
        hi, hj = torch.tensor(hop_i, device=device), torch.tensor(hop_j, device=device)
        hv = torch.stack(hop_v).to(dtype)
        HQ = HQ.index_put((hi, hj), hv, accumulate=True).index_put((hj, hi), hv, accumulate=True)
    # explicit configurations never sit in the folded space: a promoted state's hops become its couplings
    prom = {(tuple(int(x) for x in r), int(o_), int(k_)) for r, o_, k_ in (promoted or [])}
    prom_rows = {r for r, _, _ in prom}
    p_index = {tuple(int(x) for x in configs[c].tolist()): c for c in range(C)}
    keep = torch.ones(NQ, dtype=torch.bool, device=device)
    for i in range(NQ):
        r = tuple(qrows[i, :U].tolist()); key = (r, int(qrows[i, U]), int(qrows[i, U + 1]))
        if key in prom:
            keep[i] = False
            V = V.index_put((torch.tensor([p_index[r]], device=device),), HQ[i].unsqueeze(0), accumulate=True)
        elif r in p_index and r not in prom_rows:
            keep[i] = False
    if not bool(keep.all()):
        kept = keep.nonzero(as_tuple=True)[0]
        V, HQ, E_q, q_comp, qrows = V[:, kept], HQ[kept][:, kept], E_q[kept], q_comp[kept], qrows[kept]
        V = V * keep[kept].to(dtype)                                # the transferred hop columns to dropped states vanish
        NQ = int(kept.numel())
    record["promoted"] = [list(r) + [o_, k_] for r, o_, k_ in sorted(prom)]
    record["folded_states"] = [qrows[i].tolist() for i in range(NQ)] if NQ <= 256 else None   # (row..., o, kind)
    if NQ == 0:
        record["clusters"] = 0
        return zero, record
    adj = HQ != 0
    adj = adj & ~torch.eye(NQ, dtype=torch.bool, device=device)
    sigma = zero
    eye_c = torch.eye(C, dtype=dtype, device=device)
    for k in torch.unique(q_comp).tolist():
        in_k = q_comp == k
        coupled = in_k & (V != 0).any(0)
        if not bool(coupled.any()):
            continue
        reach = coupled.clone()
        for _ in range(NQ):
            grown = reach | ((adj & reach.unsqueeze(0)).any(1) & in_k)
            if bool(torch.equal(grown, reach)):
                break
            reach = grown
        idx = reach.nonzero(as_tuple=True)[0]
        Vk, Hk = V[:, idx], HQ[idx][:, idx]
        Nk = int(idx.numel())
        eye_k = torch.eye(Nk, dtype=dtype, device=device)
        # The ground energy of [[H0, V_k], [V_k^T, H_Qk]] as the fixed point E = F(E), F(E) = lambda_min(H0 + Sigma_k(E)),
        # Sigma_k(E) = V_k (E - H_Qk)^-1 V_k^T: bisection below the folded manifold, then two Newton steps
        # E <- E + (F(E) - E) / (1 + ||X w||^2) (F' = -||X w||^2 with X = (E - H_Qk)^-1 V_k^T, w the C x C ground
        # vector, which enters detached: Hellmann-Feynman makes F's first derivative exact).  No eigendecomposition
        # of the full block is in the graph: its backward divides by every eigenvalue difference, infinite on the
        # degenerate folded pairs — the seed-42 failure of step 22 (2026-10-29) in the training's double backward.

        def F_and_norm(E):
            X = torch.linalg.solve(E * eye_k - Hk, Vk.T)                            # [Nk, C]
            Heff = H0 + Vk @ X
            w = _ground_vector(Heff)
            return (w @ Heff @ w) / (w @ w), (X @ w).pow(2).sum() / (w @ w)

        with torch.no_grad():
            mu = float(torch.linalg.eigvalsh(Hk.detach())[0])
            lo = torch.tensor(float(torch.linalg.eigvalsh(H0.detach())[0]) - float(Vk.detach().norm()) - 1.0,
                              dtype=dtype, device=device)
            hi = torch.tensor(mu, dtype=dtype, device=device)
            for _ in range(8):                                                     # the bracket: g(lo) > 0
                if float(F_and_norm(lo)[0] - lo) > 0:
                    break
                lo = lo - 2.0 * (hi - lo)
            for _ in range(BISECTION_STEPS):
                mid = 0.5 * (lo + hi)
                if float(F_and_norm(mid)[0] - mid) > 0:
                    lo = mid
                else:
                    hi = mid
            E0 = 0.5 * (lo + hi)
        E_star = E0
        for _ in range(2):
            F_val, nx = F_and_norm(E_star)
            E_star = E_star + (F_val - E_star) / (1.0 + nx)
        X = torch.linalg.solve(E_star * eye_k - Hk, Vk.T)
        with torch.no_grad():
            full = torch.cat([torch.cat([H0, Vk], 1), torch.cat([Vk.T, Hk], 1)], 0).detach()
            evals, evecs = torch.linalg.eigh(full)
            p_weight = float((evecs[:C, 0] ** 2).sum())
        record.setdefault("spectrum", []).append(evals[:4].tolist())
        record["ground"].append(float(E_star)); record["explicit_weight"].append(p_weight)
        gap = mu - float(E_star)                                  # the folded manifold above the block's ground state
        record["mu_min"] = gap if record["mu_min"] != record["mu_min"] else min(record["mu_min"], gap)
        if not (float(E_star) < mu and p_weight > 0.5):
            message = (f"SPEC 32 (exact block): cluster {k}'s ground state is not explicit-dominated "
                       f"(E* {float(E_star):.4f} eV, folded manifold from {mu:.4f} eV, explicit weight "
                       f"{p_weight:.3f}): the charge-transfer state must be explicit here")
            if promote and strict:
                wq = evecs[C:, 0] ** 2                                      # the folded states' weights in the ground state
                order = torch.argsort(wq, descending=True)
                cum, chosen = p_weight, []
                for j in order.tolist():
                    chosen.append(j)
                    cum += float(wq[j])
                    if cum >= PROMOTION_TARGET or len(chosen) >= PROMOTION_BATCH:
                        break
                qi = idx[torch.tensor(chosen, device=device)]
                raise PromotionRequest(message + f" ({len(chosen)} promotions requested)", k,
                                       [qrows[q, :U].tolist() for q in qi.tolist()],
                                       [int(qrows[q, U]) for q in qi.tolist()], [int(qrows[q, U + 1]) for q in qi.tolist()],
                                       E_q[qi], V[:, qi].T, HQ[qi][:, qi], [float(wq[j]) for j in chosen])
            if strict:
                raise DownfoldError(message)
            return torch.full_like(zero, float("inf")), record
        sigma = sigma + Vk @ X
        record["dressed"] += 1
    record["clusters"] = int(torch.unique(q_comp).numel())
    record["dressing"] = torch.diagonal(sigma).detach().tolist()
    record["lambda_min"] = min(record["dressing"]) if record["dressing"] else float("nan")
    return sigma, record


def drop_folded_partners(configs: Tensor, energies: Tensor, moves: DownfoldMoves, min_charge: Tensor | None,
                         max_charge: Tensor) -> Tensor:
    """The configurations with every member reachable from a lower-energy member by one folded move removed
    (iterated to a fixed point): a folded state is never explicit beside its partner."""
    n = configs.to(torch.long)
    C, U = int(n.shape[0]), int(n.shape[1])
    if C == 0 or int(moves.o.numel()) == 0:
        return configs
    E = energies.detach()
    keep = torch.ones(C, dtype=torch.bool, device=n.device)
    index = {tuple(row): i for i, row in enumerate(n.tolist())}
    pairs = set(zip(moves.o.tolist(), moves.g.tolist()))
    changed = True
    while changed:
        changed = False
        for i in sorted(range(C), key=lambda k: float(E[k])):
            if not keep[i]:
                continue
            for o, g in pairs:
                for d in (1, -1):
                    row = n[i].clone()
                    row[o] += d
                    row[g] -= d
                    j = index.get(tuple(row.tolist()))
                    if j is not None and keep[j] and float(E[j]) > float(E[i]):
                        keep[j] = False
                        changed = True
    return configs[keep]
