"""Learned inputs of the free-charge sector (SPEC 25.8 B and C).

Every quantity here is a smooth, rotation-, translation- and permutation-
invariant function of the geometry that ENTERS the convex functional; none of
them replaces the minimisation ("attention parameterises, physics minimises").

* :class:`ChargeStateHead` -- the environment dependence of the charge-state
  energies ``E_u^(n)(G_u)``.  Each charged state is a query token attending over
  the unit's environment (itself and its neighbours inside ``r_cut``); the
  context it gathers sets BOUNDED shifts of the unit's electronegativity and
  gap around tabulated anchors (trace_redox.module).  The last layer is zero,
  so at initialisation the state energies ARE the NIST atomic values.
* :class:`FreeChargeWidth` -- the free charge's Gaussian width from the atomic
  hardness ``IE - EA``.
* :class:`OrbitalHead` -- attention-Wannier free-charge orbitals.  Unit ``u``'s
  free charge is spread over atoms as

      Omega_uj = (1 - s_u) delta_uj + s_u w_uj,   w_u. = softmax over {u} + N(u),
      0 < s_u < s_max < 1/2

  Rows sum to one (charge conserving) and ``Omega`` is invertible
  (``|| diag(1-s)^{-1} diag(s) w ||_inf <= s/(1-s) < 1``), so ``Omega W Omega^T``
  stays positive definite.

Attention over a neighbourhood with a hard cutoff would make the energy jump
when a neighbour crosses it.  Both heads therefore weight each neighbour's
exponential by the C2 switch ``f(x) = 1 - 10x^3 + 15x^4 - 6x^5``
(``x = r / r_cut``) INSIDE the softmax normalisation: a neighbour enters and
leaves with weight, slope and curvature all zero.  The unit itself always has
weight ``f(0) = 1``, so the normalisation never vanishes.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .constants import ATOMIC_ELECTRON_AFFINITY_EV, ATOMIC_IONIZATION_EV, MAX_Z, SHANNON_CRYSTAL_RADIUS

#: Charge states (electrons removed) of every unit in v1.
SUPPORTED_STATES = (-1, 0, 1)


def smooth_switch(r: Tensor, r_cut: float) -> Tensor:
    """``f(r / r_cut)`` with ``f(x) = 1 - 10 x^3 + 15 x^4 - 6 x^5`` on [0, 1], 0 beyond."""
    x = (r / r_cut).clamp(0.0, 1.0)
    return 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x)


def _radial_basis(r: Tensor, r_cut: float, n_radial: int) -> Tensor:
    centers = torch.linspace(0.0, r_cut, n_radial, dtype=r.dtype, device=r.device)
    width = r_cut / max(n_radial - 1, 1)
    return torch.exp(-(((r.unsqueeze(-1) - centers) / width) ** 2))


def neighbourhood_softmax(
    self_logit: Tensor, edge_logit: Tensor, unit: Tensor, edge_weight: Tensor, n_units: int
) -> tuple[Tensor, Tensor]:
    """Switch-weighted softmax over ``{u} + N(u)``.

    ``self_logit [n, ...]``, ``edge_logit [E, ...]``, ``unit [E]`` (the unit each
    edge belongs to), ``edge_weight [E]`` (the switch).  Returns the weights
    ``(w_self [n, ...], w_edge [E, ...])``; for every unit they sum to one.
    The per-unit maximum is removed (detached: the softmax does not depend on it).
    """
    trailing = self_logit.shape[1:]
    m = self_logit.detach().clone()
    index = unit.view(-1, *([1] * len(trailing))).expand_as(edge_logit)
    m = m.scatter_reduce(0, index, edge_logit.detach(), reduce="amax", include_self=True)
    e_self = torch.exp(self_logit - m)
    e_edge = edge_weight.view(-1, *([1] * len(trailing))) * torch.exp(edge_logit - m.index_select(0, unit))
    norm = e_self.index_add(0, unit, e_edge)
    return e_self / norm, e_edge / norm.index_select(0, unit)


class ChargeStateHead(nn.Module):
    """Charge-state cross-attention: raw shifts ``[n_units, n_tokens]``.

    One query token per charged state, in the order (+1, -1, +2, ..., +K) (the
    neutral state is the energy reference, ``E^(0) = 0``), attends over the unit
    and its neighbours; an MLP reads each token's context.  The last layer is
    zero, so the raw shifts are zero at initialisation.
    :class:`trace_redox.module.RedoxElectrostatics` turns them into the bounded
    electronegativity and gaps of SPEC 25.  ``n_tokens = 2`` (K = 1) draws the
    same random initialisation as before the generalisation.
    """

    def __init__(
        self,
        invariant_dim: int,
        *,
        hidden_dim: int = 64,
        key_dim: int = 32,
        r_cut: float = 4.0,
        n_radial: int = 8,
        n_tokens: int = 2,
    ) -> None:
        super().__init__()
        if r_cut <= 0.0:
            raise ValueError("r_cut must be positive")
        if n_tokens < 2:
            raise ValueError("n_tokens must be >= 2 (the states +1 and -1)")
        self.r_cut = float(r_cut)
        self.n_radial = int(n_radial)
        self.key_dim = int(key_dim)
        self.n_tokens = int(n_tokens)
        self.norm = nn.LayerNorm(invariant_dim)
        self.state_query = nn.Parameter(torch.randn(n_tokens, key_dim) / math.sqrt(key_dim))
        self.k_proj = nn.Linear(invariant_dim, key_dim, bias=False)
        self.v_proj = nn.Linear(invariant_dim, hidden_dim, bias=False)
        self.radial = nn.Linear(n_radial, n_tokens, bias=False)
        self.state_embed = nn.Parameter(torch.randn(n_tokens, hidden_dim) / math.sqrt(hidden_dim))
        self.shift_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.shift_mlp[-1].weight)
        nn.init.zeros_(self.shift_mlp[-1].bias)

    def forward(self, invariants: Tensor, edge_index: Tensor, edge_len: Tensor) -> Tensor:
        n = int(invariants.shape[0])
        x = self.norm(invariants)
        keys, values = self.k_proj(x), self.v_proj(x)                        # [n, d]
        neighbour, unit = edge_index[0], edge_index[1]
        inside = edge_len < self.r_cut
        neighbour, unit, r = neighbour[inside], unit[inside], edge_len[inside]
        scale = 1.0 / math.sqrt(self.key_dim)
        q = self.state_query.to(x.dtype)                                     # [T, d]
        r0 = torch.zeros(1, dtype=x.dtype, device=x.device)
        self_logit = (keys @ q.T) * scale + self.radial(_radial_basis(r0, self.r_cut, self.n_radial))
        edge_logit = (keys.index_select(0, neighbour) @ q.T) * scale + self.radial(
            _radial_basis(r, self.r_cut, self.n_radial)
        )
        w_self, w_edge = neighbourhood_softmax(
            self_logit, edge_logit, unit, smooth_switch(r, self.r_cut), n
        )                                                                    # [n, T], [E, T]
        context = w_self.unsqueeze(-1) * values.unsqueeze(1)                 # [n, T, h]
        context = context.index_add(
            0, unit, w_edge.unsqueeze(-1) * values.index_select(0, neighbour).unsqueeze(1)
        )
        token = torch.cat((context, self.state_embed.to(x.dtype).expand(n, -1, -1)), dim=-1)
        return self.shift_mlp(token).squeeze(-1)                             # [n, T]: (+1, -1, +2, ...)


class FreeChargeWidth(nn.Module):
    """Width of each atom's free-charge Gaussian from the atomic hardness (SPEC 25.2).

        sigma_f,Z = sigma_H,Z / h_Z,    sigma_H,Z = sqrt(2) K_E (1 - lam) / (sqrt(pi) U_Z),
        U_Z = IE_Z - EA_Z,              h_Z in (h_min, h_max), learned per element

    ``sigma_H`` is the width whose Gram self-energy, times ``(1 - lam)``, equals the
    atomic hardness: the QEq relation between hardness and orbital size
    (Rappe & Goddard, J. Phys. Chem. 95, 3358 (1991): J_AA = IE - EA), at which
    the free sector is exactly QEq -- linear state energies, no gap.  The
    fraction ``h`` of the hardness that is classical Hartree energy therefore
    leaves the gap ``(1 - h) U_Z > 0`` of the integer states: ``h -> 1`` is QEq
    (metallic), ``h -> 0`` the exact PPLB discontinuity with an infinitely
    diffuse free charge.
    """

    def __init__(self, self_interaction: float = 0.0, h_init: float = 0.5,
                 h_min: float = 0.05, h_max: float = 0.95) -> None:
        super().__init__()
        if not 0.0 < h_min < h_init < h_max < 1.0:
            raise ValueError("need 0 < h_min < h_init < h_max < 1")
        self.self_interaction = float(self_interaction)
        self.h_min, self.h_max = float(h_min), float(h_max)
        ratio = (h_init - h_min) / (h_max - h_min)
        self.logit = nn.Parameter(torch.full((MAX_Z + 1,), math.log(ratio / (1.0 - ratio))))
        ie = torch.tensor(ATOMIC_IONIZATION_EV, dtype=torch.float64)
        ea = torch.tensor(ATOMIC_ELECTRON_AFFINITY_EV, dtype=torch.float64)
        self.register_buffer("hardness", ie - ea, persistent=False)

    def fraction(self, species: Tensor) -> Tensor:
        """``h_Z`` per atom."""
        return self.h_min + (self.h_max - self.h_min) * torch.sigmoid(self.logit[species])

    def forward(self, species: Tensor) -> Tensor:
        """``sigma_f`` per atom, A."""
        from trace_electrostatics.constants import K_E

        u = self.hardness.to(self.logit.dtype)[species]
        sigma_h = math.sqrt(2.0) * K_E * (1.0 - self.self_interaction) / (math.sqrt(math.pi) * u)
        return sigma_h / self.fraction(species)


class IonicFreeChargeWidth(nn.Module):
    """Width of each atom's free-charge Gaussian from its ionic size (audit step 2, B3).

        sigma_Z = sqrt(2/5) R_Z exp(L tanh(s_Z / L)),   L = ln 2,   s_Z learned per element (init 0)

    The on-site ladder already holds an isolated ion's whole energy (its self-Coulomb is removed),
    so the width acts only between units, where it must be the ion's own charge distribution:
    compact for localised ionic states, not the neutral atom's diffuse frontier orbital that the
    hardness rule (:class:`FreeChargeWidth`) gives (1.9 A for O, 4.1 A for Al at h = 0.5).

    ``R_Z`` is Shannon's crystal radius (:data:`~trace_redox.constants.SHANNON_CRYSTAL_RADIUS`;
    ESTABLISHED).  ``sqrt(2/5)`` matches the second moment of a uniformly charged sphere of radius
    ``R`` (``3 R^2 / 5``) to the Gaussian whose potential is ``erf(r / sigma) / r``, the density
    ``(pi sigma^2)^(-3/2) exp(-r^2 / sigma^2)`` with ``<r^2> = 3 sigma^2 / 2`` (NEW-TO-TRACE).  The
    learned factor lies in ``[1/2, 2]``: room for the sign of the charge (an anion of a metal, e.g.
    Au-, is larger than the tabulated cation), never back to a band-carrier width.  Formal +/-2
    rocksalt MgO keeps 99.35 % of its point-charge Madelung energy (the hardness rule at h = 0.5: 42 %;
    tests/redox/test_redox_free_width.py).
    """

    SCALE = math.log(2.0)

    def __init__(self, fallback_qeq: bool = False) -> None:
        super().__init__()
        radius = torch.full((MAX_Z + 1,), float("nan"), dtype=torch.float64)
        for z, r in SHANNON_CRYSTAL_RADIUS.items():
            radius[z] = r
        if fallback_qeq:
            # elements with no tabulated ion (H, C, ...): the QEq width sqrt(2) K_E / (sqrt(pi) (IE - EA)),
            # the most compact width mean field allows (SPEC 16), written as an equivalent radius
            from trace_electrostatics.constants import K_E
            ie = torch.tensor(ATOMIC_IONIZATION_EV, dtype=torch.float64)
            ea = torch.tensor(ATOMIC_ELECTRON_AFFINITY_EV, dtype=torch.float64)
            qeq = math.sqrt(2.0) * K_E / (math.sqrt(math.pi) * (ie - ea)) / math.sqrt(0.4)
            radius = torch.where(torch.isnan(radius) & torch.isfinite(qeq), qeq, radius)
        self.register_buffer("radius", radius, persistent=False)
        self.log_factor = nn.Parameter(torch.zeros(MAX_Z + 1))

    def forward(self, species: Tensor) -> Tensor:
        """``sigma_f`` per atom, A."""
        radius = self.radius.to(self.log_factor.dtype)[species]
        missing = torch.isnan(radius)
        if bool(missing.any()):
            symbols = sorted({int(z) for z in species[missing]})
            raise ValueError(
                f"width='ionic' has no Shannon crystal radius for Z = {symbols} "
                f"({', '.join(_symbol(z) for z in symbols)}); add the element's ion to "
                "trace_redox.constants.SHANNON_CRYSTAL_RADIUS from the published table"
            )
        factor = torch.exp(self.SCALE * torch.tanh(self.log_factor[species] / self.SCALE))
        return math.sqrt(0.4) * radius * factor


def _symbol(z: int) -> str:
    from ase.data import chemical_symbols
    return chemical_symbols[z]


#: SPEC 27: Harrison's largest universal coefficient |eta| (V_pp sigma = 3.24 hbar^2 / (m d^2)) and the knee below
#: which the bound is the identity
HARRISON_ETA_MAX = 3.24
BOUND_KNEE = 0.75
#: the regularisation length of the bound (A): Harrison's 1/d^2 diverges at coincidence, a hopping integral between
#: orthogonalised orbitals does not; B(0) = 3.24 hbar^2 / (m r0^2) = 99 eV
BOUND_R0 = 0.5


def harrison_bound(raw: Tensor, r: Tensor) -> Tensor:
    """``x`` for ``|x| <= a = 0.75 B``, ``sign(x) [a + (B - a) tanh((|x| - a) / (B - a))]`` beyond, with
    ``B(r) = 3.24 hbar^2 / (m (r^2 + r0^2))``: C^2 (tanh''(0) = 0), strictly below B, finite at r = 0, the identity
    (bitwise) below the knee."""
    from .band import HBAR2_OVER_ME
    B = HARRISON_ETA_MAX * HBAR2_OVER_ME / (r * r + BOUND_R0 * BOUND_R0)
    a = BOUND_KNEE * B
    span = B - a
    over = raw.abs() - a
    clipped = torch.sign(raw) * (a + span * torch.tanh(over / span))
    return torch.where(over > 0, clipped, raw)


#: SPEC 31: Harrison's universal two-centre coefficients (ESTABLISHED: Froyen and Harrison, Phys. Rev. B 20, 2420
#: (1979); Harrison, Electronic Structure and the Properties of Solids, 1980), V = eta hbar^2 / (m d^2).  s-p takes the
#: sigma value (a single partner couples to one combination of the p hole, the sigma one); p-p the trace average
#: (V_pp sigma + 2 V_pp pi) / 3 (a hole in a lattice cannot be sigma to every neighbour).
ETA_SS_SIGMA = -1.40
ETA_SP_SIGMA = 1.84
ETA_PP_AVERAGE = (3.24 + 2.0 * (-0.81)) / 3.0
#: SPEC 31: the anchored coupling's learned factor exp(c tanh(x / c)) lies in (1 / 2, 2)
ANCHOR_LOG_RANGE = math.log(2.0)
#: SPEC 31: the anchored coupling's switch acts on [r_cut - w, r_cut] only (A).  The learned head's full-range
#: switch is 0.38 at the 2.26 A Au-O contact and 0.26 at the Au-Au bond: a first version that kept it took 60-75 % off
#: Harrison's values (measured 2026-10-27, probe_anchor_counterfactual.py).
ANCHOR_SWITCH_WIDTH = 1.0


def window_switch(r: Tensor, r_cut: float, width: float) -> Tensor:
    """1 for ``r <= r_cut - width``, the quintic ``1 - 10 t^3 + 15 t^4 - 6 t^5`` of ``t = (r - r_cut + width) / width``
    on the window, 0 beyond ``r_cut``: C^2."""
    t = ((r - (r_cut - width)) / width).clamp(0.0, 1.0)
    return 1.0 - t ** 3 * (10.0 - 15.0 * t + 6.0 * t * t)


def frontier_shell(z: int) -> int | None:
    """The frontier shell's l of element ``z`` (Z <= 86): 0 (s) for H, He and groups 1, 2, 11, 12; 1 (p) for groups
    13-18; None for the d and f blocks (groups 3-10, the lanthanides), which SPEC 31 does not anchor."""
    z = int(z)
    if z in (1, 2):
        return 0
    starts = ((3, 10), (11, 18), (19, 36), (37, 54), (55, 86))      # periods 2-6
    for first, last in starts:
        if first <= z <= last:
            k = z - first                                          # 0-based position in the period
            if k < 2:
                return 0                                           # groups 1, 2
            if last - first == 7:
                return 1                                           # periods 2, 3: groups 13-18
            if last - first == 17:                                 # periods 4, 5
                return None if k < 10 else (0 if k < 12 else 1)    # groups 3-10, 11-12, 13-18
            return None if k < 24 else (0 if k < 26 else 1)       # period 6: La-Lu, Hf-Pt; Au, Hg; Tl-Rn
    return None


def open_frame(species: Tensor) -> Tensor:
    """``[n]`` bool: the formal frame state holds an odd number of electrons (a singly occupied frontier orbital).
    (SPEC 31's first rule anchored only pairs with an open frame; superseded by :func:`anchored_pairs`, kept.)"""
    from .constants import formal_frame
    z = torch.as_tensor(species).to(torch.long)
    return ((z - formal_frame(z).to(z.device)) % 2) == 1


def anchored_pairs(species: Tensor, src: Tensor, dst: Tensor) -> Tensor:
    """SPEC 31: ``[E]`` bool, every pair except a cation and an anion of the formal frame (f_u f_v < 0).  Those
    transfers cross the ionic gap (the frame's covalency: the backbone's and the moments', SPEC 22); every other one
    moves a carrier (a hole between anions, an electron between cations, an electron to or from a neutral frame)."""
    from .constants import formal_frame
    z = torch.as_tensor(species).to(torch.long)
    f = formal_frame(z).to(src.device)
    return f[src] * f[dst] >= 0


class TransferCoupling(nn.Module):
    """One-electron coupling ``tau_uv`` between units: the hopping of one electron u <-> v (SPEC 17.6).

        tau_uv = sum_{edges u->v} [ sum_k c_k(x_u, x_v) g_k(r) ] f(r / r_cut),

    ``c = MLP(x_u + x_v, x_u * x_v)`` (symmetric in u, v, so tau_uv = tau_vu exactly), ``g_k`` Gaussians
    on [0, r_cut], ``f`` the C^2 quintic switch (value and first two derivatives zero at r_cut).  The
    two-centre form of a hopping integral (Slater-Koster; ESTABLISHED), learned from TRACE invariants
    (NEW-TO-TRACE).  Periodic images add (a Gamma-point Bloch sum); self-image edges are excluded (a
    move of an electron from u to u changes no charge).  The edge list must hold both directions.

    Initialised small but NON-ZERO: F is even in tau, so dF/dtau = 0 at tau = 0 and a zero coupling would
    never be learned.

    ``bound='harrison'`` (SPEC 27): tau = :func:`harrison_bound` (tau_raw), the identity up to 0.75 B and C^2
    saturation to B(r) = 3.24 hbar^2 / (m (r^2 + r0^2)) beyond (the largest of Harrison's universal coefficients, V_pp sigma;
    ESTABLISHED), so a hopping integral never exceeds the universal tight-binding scale however far the invariants
    extrapolate (measured: 9200 eV without it) while every coupling below the knee is untouched.

    ``anchor='harrison'`` (SPEC 31; needs the species): every pair but a frame cation-anion pair (:func:`anchored_pairs`)
    takes
    tau = eta(l_u, l_v) hbar^2 / (m (r^2 + r0^2)) exp(c tanh(x / c)) w(r), x the learned radial sum, c = ln 2, w the
    C^2 switch on [r_cut - 1 A, r_cut] (:func:`window_switch`), eta from the frontier shells (:func:`frontier_shell`);
    every other pair is the SPEC 27-bounded learned value.
    """

    def __init__(self, invariant_dim: int, *, hidden_dim: int = 32, n_radial: int = 8,
                 r_cut: float = 4.0, init_scale: float = 0.01, bound: str | None = None,
                 anchor: str | None = None) -> None:
        super().__init__()
        if bound not in (None, "harrison"):
            raise ValueError(f"bound must be None or 'harrison', got {bound!r}")
        if anchor not in (None, "harrison"):
            raise ValueError(f"anchor must be None or 'harrison', got {anchor!r}")
        self.bound = bound
        self.anchor = anchor
        self.r_cut = float(r_cut)
        self.n_radial = int(n_radial)
        self.pair = nn.Sequential(nn.Linear(2 * invariant_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, n_radial))
        with torch.no_grad():
            self.pair[-1].weight.normal_(0.0, float(init_scale))
            self.pair[-1].bias.zero_()

    def _radial(self, invariants: Tensor, edge_index: Tensor, edge_len: Tensor) -> Tensor:
        x_u, x_v = invariants[edge_index[0]], invariants[edge_index[1]]
        c = self.pair(torch.cat((x_u + x_v, x_u * x_v), dim=-1))
        return (c * _radial_basis(edge_len, self.r_cut, self.n_radial)).sum(-1)

    def raw_values(self, invariants: Tensor, edge_index: Tensor, edge_len: Tensor) -> Tensor:
        """The unbounded learned ``tau_raw`` on every directed edge, eV."""
        return self._radial(invariants, edge_index, edge_len) * smooth_switch(edge_len, self.r_cut)

    def edge_values(self, invariants: Tensor, edge_index: Tensor, edge_len: Tensor,
                    species: Tensor | None = None) -> Tensor:
        """``tau`` on every directed edge, eV."""
        radial = self._radial(invariants, edge_index, edge_len)
        switch = smooth_switch(edge_len, self.r_cut)
        raw = radial * switch
        if getattr(self, "anchor", None) is None:                 # checkpoints from before SPEC 31 lack the attribute
            if self.bound is None:
                return raw
            return harrison_bound(raw, edge_len)
        if species is None:
            raise ValueError("coupling anchor 'harrison' needs the species (SPEC 31)")
        from .band import HBAR2_OVER_ME
        z = torch.as_tensor(species).to(torch.long).to(edge_index.device)
        src, dst = edge_index[0], edge_index[1]
        anchored = anchored_pairs(z, src, dst)
        shells = {int(v): frontier_shell(int(v)) for v in torch.unique(z).tolist()}
        l_atom = torch.tensor([-1 if shells[int(v)] is None else shells[int(v)] for v in z.tolist()],
                              dtype=torch.long, device=z.device)
        l_src, l_dst = l_atom[src], l_atom[dst]
        bad = anchored & ((l_src < 0) | (l_dst < 0))
        if bool(bad.any()):
            k = int(bad.nonzero()[0])
            raise ValueError(f"coupling anchor 'harrison': no s or p frontier shell for the pair Z = "
                             f"{int(z[src[k]])}, {int(z[dst[k]])} (d and f blocks are not anchored; SPEC 31)")
        dtype = edge_len.dtype
        total = (l_src + l_dst).clamp(min=0)
        eta = torch.where(total == 0, torch.full_like(edge_len, ETA_SS_SIGMA),
                          torch.where(total == 1, torch.full_like(edge_len, ETA_SP_SIGMA),
                                      torch.full_like(edge_len, ETA_PP_AVERAGE))).to(dtype)
        base = eta * HBAR2_OVER_ME / (edge_len * edge_len + BOUND_R0 * BOUND_R0)
        c = ANCHOR_LOG_RANGE
        value = base * torch.exp(c * torch.tanh(radial / c)) * window_switch(edge_len, self.r_cut, ANCHOR_SWITCH_WIDTH)
        return torch.where(anchored, value, harrison_bound(raw, edge_len))

    def forward(self, invariants: Tensor, edge_index: Tensor, edge_len: Tensor,
                species: Tensor | None = None) -> Tensor:
        """``T [n, n]``: symmetric, zero diagonal, eV."""
        n = int(invariants.shape[0])
        values = self.edge_values(invariants, edge_index, edge_len, species=species)
        off = edge_index[0] != edge_index[1]
        T = values.new_zeros(n, n).index_put((edge_index[0][off], edge_index[1][off]), values[off], accumulate=True)
        return 0.5 * (T + T.transpose(0, 1))


class OrbitalHead(nn.Module):
    """Attention-Wannier free-charge orbitals ``Omega [n_units, n_atoms]`` (units = atoms)."""

    def __init__(
        self,
        invariant_dim: int,
        *,
        hidden_dim: int = 64,
        key_dim: int = 32,
        r_cut: float = 4.0,
        n_radial: int = 8,
        s_max: float = 0.45,
        s_init: float = 0.02,
    ) -> None:
        super().__init__()
        if not 0.0 < s_max < 0.5:
            raise ValueError(f"s_max must lie in (0, 1/2) for Omega to be invertible, got {s_max}")
        if not 0.0 < s_init < s_max:
            raise ValueError(f"s_init must lie in (0, s_max), got {s_init}")
        self.s_max = float(s_max)
        self.r_cut = float(r_cut)
        self.n_radial = int(n_radial)
        self.key_dim = int(key_dim)
        self.norm = nn.LayerNorm(invariant_dim)
        self.q_proj = nn.Linear(invariant_dim, key_dim, bias=False)
        self.k_proj = nn.Linear(invariant_dim, key_dim, bias=False)
        self.radial = nn.Linear(n_radial, 1, bias=False)
        self.spread = nn.Sequential(nn.Linear(invariant_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.spread[-1].weight)
        ratio = s_init / s_max
        nn.init.constant_(self.spread[-1].bias, math.log(ratio / (1.0 - ratio)))

    def forward(self, invariants: Tensor, edge_index: Tensor, edge_len: Tensor) -> Tensor:
        n = int(invariants.shape[0])
        x = self.norm(invariants)
        queries, keys = self.q_proj(x), self.k_proj(x)
        neighbour, unit = edge_index[0], edge_index[1]
        inside = edge_len < self.r_cut
        neighbour, unit, r = neighbour[inside], unit[inside], edge_len[inside]
        scale = 1.0 / math.sqrt(self.key_dim)
        r0 = torch.zeros(1, dtype=x.dtype, device=x.device)
        self_logit = ((queries * keys).sum(-1) * scale).unsqueeze(-1) + self.radial(
            _radial_basis(r0, self.r_cut, self.n_radial)
        )
        edge_logit = (
            (queries.index_select(0, unit) * keys.index_select(0, neighbour)).sum(-1) * scale
        ).unsqueeze(-1) + self.radial(_radial_basis(r, self.r_cut, self.n_radial))
        w_self, w_edge = neighbourhood_softmax(
            self_logit, edge_logit, unit, smooth_switch(r, self.r_cut), n
        )
        s = self.s_max * torch.sigmoid(self.spread(x).squeeze(-1))           # [n]
        eye = torch.eye(n, dtype=x.dtype, device=x.device)
        spread = torch.diag(w_self.squeeze(-1)).index_put(
            (unit, neighbour), w_edge.squeeze(-1), accumulate=True
        )
        return (1.0 - s).unsqueeze(1) * eye + s.unsqueeze(1) * spread
