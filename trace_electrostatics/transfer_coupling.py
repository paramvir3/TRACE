"""Through-bond electronic coupling between diabatic charge states (SPEC 22).

The problem
-----------
SPEC 21 makes integer charge localisation possible: each diabatic state fixes
integer fragment charges, and the adiabatic energy is the lowest eigenvalue of
``H = diag(E_a) + offdiag``.  The off-diagonal -- the electronic coupling that
lets an electron move between a donor and an acceptor -- was an INPUT.  But the
coupling is the physics of electron transfer: it decays exponentially with the
donor-acceptor separation, it depends on what lies between them (a covalent
bridge carries the electron farther than a hydrogen bond, which carries it
farther than empty space), and it is what a model of redox chemistry in a
protein or at an electrode must predict.

The construction
----------------
Every atom is a tunnelling site; every bond ``b = (i, j)`` of the model's own
neighbour graph carries a hopping amplitude ``h_b >= 0`` and every site a
tunnelling gap.  With ``D`` the donor fragment's atoms, ``A`` the acceptor's
and ``B`` everything else (the bridge), partitioning the tunnelling Hamiltonian
onto the donor and acceptor states (Loewdin 1951) gives

    V_DA = w_D^T [ H_DA + H_DB (Delta_B - H_BB)^-1 H_BA ] w_A          (1)

with ``H`` the symmetric hopping matrix (``H_ij = sum of h_b over the bonds
joining i and j``), ``Delta`` the diagonal gaps and ``w_D``, ``w_A`` the
donor and acceptor amplitudes on their fragments' atoms.  Expanding the inverse
as a Neumann series, ``(Delta - H)^-1 = sum_n (Delta^-1 H)^n Delta^-1``, turns
(1) into a sum over EVERY tunnelling path from D to A, each weighted by the
product of ``h / Delta`` along it -- McConnell's superexchange (J. Chem. Phys.
35, 508 (1961)) for one path, the Beratan-Onuchic pathway picture (Science 252,
1285 (1991)) for the dominant one, and here all of them at once.

Positivity by construction.  The gaps are

    Delta_i = sum_{b incident to i} h_b + softplus(gamma_i) + g_min,       (2)

so ``Delta_B - H_BB`` is strictly diagonally dominant with a positive diagonal,
hence symmetric positive definite, for EVERY value of the learned parameters
and every geometry.  The Neumann series converges, the Green's function is
entrywise non-negative (an inverse M-matrix), and (1) is a smooth function of
the positions: there is no resonance, no division by zero and no run-time
failure mode to guard.  The bound is tight where it should be: for a uniform
chain the condition ``Delta > 2h`` is exactly the band edge, and the coupling
across ``n`` bridge sites is, in closed form,

    V(n) = h sinh(theta) / sinh((n + 1) theta),   cosh(theta) = Delta / (2h)

-- exponential decay ``exp(-theta)`` per site, approached as ``Delta -> 2h`` only
logarithmically slowly; the regression suite checks this to round-off.

Which states couple, and with which sign.  States ``a`` and ``b`` (same
fragment partition) couple directly iff their fragment charges differ by
``+1`` on exactly one fragment and ``-1`` on exactly one other -- one electron
moved.  Multi-electron and multi-step transfer then follow from diagonalising
the EVB matrix itself.  The off-diagonal element is ``H_ab = -V_ab <= 0``: a
tunnelling matrix element, whose sign makes the ground state nodeless
(Perron-Frobenius) for any number of states; with two states only ``V^2``
enters.

The learned pieces (``TransferCouplingHead``) read the same rotation-invariant
node summaries as the electrostatic heads:

    h_b      = f(r_b) h_ref exp(-beta (r_b - r_ref) + L tanh(phi(e_i + e_j, e_i * e_j, rbf(r_b)) / L))
    gamma_i  , s_i    from a per-site network
    w_i      = sqrt(softmax_{i in fragment}(s_i))

``f`` is the C2 bond envelope of SPEC 17.4, so a bond leaving the neighbour list
removes its hopping smoothly; the bond input is symmetric in ``i, j`` so
``h_ij = h_ji`` exactly; ``h_ref`` and ``beta > 0`` are learned.  The chemical
modulation is bounded, ``|L tanh(phi / L)| <= L = ln 100``: the distance law
``h_ref exp(-beta (r - r_ref))`` carries the decay, and the environment moves a hopping by
at most two decades either way -- the span from saturated to conjugated
couplings.  (An unbounded ``exp(phi)`` let randomised weights reach hoppings
of 1e-39 to 1e15 eV and a saturated, geometry-blind 14 eV coupling.)  The final layers
start at zero: at initialisation every bond hops ``h_ref exp(-beta (r - r_ref)) f(r)``,
every gap excess is ``gamma_0`` and the amplitudes are uniform on each fragment.
The defaults put the prior where chemistry is known: 2.5 eV at 1.5 A (the
Hueckel scale of a covalent bond) falling with beta = 2.2 / A to ~0.03 eV at
3.5 A (a through-space contact), and a 1.5 eV tunnelling gap -- a saturated
bridge then decays by exp(-theta) = 0.47 per bond, the order of the empirical
0.6 per covalent bond of the pathway model.

Established, cited: Loewdin partitioning; McConnell superexchange; tunnelling
pathways; EVB (SPEC 21).  TRACE-native: the hoppings, gaps and redox-site
amplitudes learned from the equivariant attention features; the diagonally
dominant parameterisation (2) that makes the bridge Green's function exist,
positive and smooth for every parameter value; its evaluation on the model's
own neighbour graph with exact derivatives (dense, or matrix-free through the
implicit solve of SPEC 17), inside the electrostatic EVB.

Condon approximation: ``V`` depends on the geometry and the atoms' features,
not on the electrostatic state.  Field-dependent (non-Condon) couplings are not
modelled.  All couplings are non-negative, so destructive interference between
paths (e.g. meta-substituted aromatic bridges) is outside the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .implicit import QuadraticProblem, minimise

__all__ = [
    "DEFAULT_GAP_FLOOR",
    "DENSE_BRIDGE_MAX",
    "HOPPING_LOG_RANGE",
    "TransferCouplingHead",
    "TransferGraph",
    "bridge_coupling",
    "coupling_matrix",
]

#: Lower bound of the gap excess ``Delta_i - sum h``, eV.  Keeps the bridge
#: operator's smallest eigenvalue at least this far from zero.
DEFAULT_GAP_FLOOR = 0.1

#: Bound of the learned log-modulation of a hopping: a factor of 100 either way.
HOPPING_LOG_RANGE = math.log(100.0)

#: Bridges up to this many sites are solved densely (Cholesky); larger ones by
#: preconditioned CG with implicit derivatives (SPEC 17).
DENSE_BRIDGE_MAX = 2048


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


@dataclass
class TransferGraph:
    """The tunnelling graph of one configuration.

    Attributes:
        bond_index: ``[2, m]`` ``(i, j)`` with ``i < j``, one entry per bond
            (per periodic image); self-image bonds are excluded.
        hopping: ``[m]`` ``h_b >= 0``, eV.
        gap_excess: ``[n_atoms]`` ``Delta_i - sum_b h_b > 0``, eV.
        amplitude_logit: ``[n_atoms]`` ``s_i``; the donor/acceptor amplitude
            is ``sqrt(softmax(s))`` over the fragment's atoms.
    """

    bond_index: Tensor
    hopping: Tensor
    gap_excess: Tensor
    amplitude_logit: Tensor

    @property
    def n_atoms(self) -> int:
        return int(self.gap_excess.shape[0])

    def gaps(self) -> Tensor:
        """``Delta_i = sum_{b incident to i} h_b + gap_excess_i`` -- eq. (2)."""
        lo, hi = self.bond_index
        total = self.gap_excess.clone()
        total = total.index_add(0, lo, self.hopping)
        return total.index_add(0, hi, self.hopping)

    def hop(self, vector: Tensor) -> Tensor:
        """``H v`` with ``H_ij = sum of h_b over bonds joining i and j``."""
        lo, hi = self.bond_index
        out = torch.zeros_like(vector)
        out = out.index_add(0, lo, self.hopping * vector[hi])
        return out.index_add(0, hi, self.hopping * vector[lo])


class TransferCouplingHead(nn.Module):
    """Hoppings, gaps and redox-site amplitudes from TRACE's node invariants.

    Args:
        invariant_dim: width of the invariant node summary (for
            :class:`~trace_electrostatics.heads.ElectrostaticHeads`, use
            ``heads.node_invariants(h).shape[-1]``).
        r_max: tunnelling-bond cutoff, A; hoppings vanish smoothly (C2) at it.
        hidden_dim, n_radial: network widths.
        envelope_onset: fraction of ``r_max`` where the C2 switch starts.
        hopping_init: ``h_ref``, eV -- hopping at ``reference_length``.
        reference_length: ``r_ref``, A.
        decay_init: ``beta``, 1/A.
        gap_init: initial gap excess ``gamma_0``, eV (> ``gap_floor``).
        gap_floor: ``g_min``, eV.
    """

    def __init__(
        self,
        invariant_dim: int,
        r_max: float,
        *,
        hidden_dim: int = 32,
        n_radial: int = 8,
        envelope_onset: float = 0.75,
        hopping_init: float = 2.5,
        reference_length: float = 1.5,
        decay_init: float = 2.2,
        gap_init: float = 1.5,
        gap_floor: float = DEFAULT_GAP_FLOOR,
    ) -> None:
        super().__init__()
        if not r_max > 0.0:
            raise ValueError(f"r_max must be positive, got {r_max}")
        if not 0.0 <= envelope_onset < 1.0:
            raise ValueError(f"envelope_onset must lie in [0, 1), got {envelope_onset}")
        if not hopping_init > 0.0 or not decay_init > 0.0:
            raise ValueError("hopping_init and decay_init must be positive")
        if not gap_floor > 0.0 or not gap_init > gap_floor:
            raise ValueError(
                f"need 0 < gap_floor < gap_init, got gap_floor={gap_floor}, gap_init={gap_init}"
            )
        self.r_max = float(r_max)
        self.envelope_onset = float(envelope_onset)
        self.gap_floor = float(gap_floor)
        self.reference_length = float(reference_length)
        self.norm = nn.LayerNorm(invariant_dim)
        self.embed = nn.Sequential(
            nn.Linear(invariant_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.bond_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + n_radial, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.site_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2)
        )
        for last in (self.bond_net[-1], self.site_net[-1]):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        self.log_hopping_scale = nn.Parameter(torch.tensor(math.log(hopping_init)))
        self.decay_raw = nn.Parameter(torch.tensor(_inverse_softplus(decay_init)))
        self.gap_shift = float(_inverse_softplus(gap_init - gap_floor))
        self.register_buffer("rbf_centers", torch.linspace(0.0, float(r_max), n_radial))
        self.rbf_width = float(r_max) / max(n_radial - 1, 1)

    @classmethod
    def for_heads(cls, heads: nn.Module, **kwargs) -> "TransferCouplingHead":
        """A head reading ``heads.node_invariants(h)``; ``r_max`` defaults to the heads'."""
        kwargs.setdefault("r_max", float(heads.r_max))
        return cls(int(heads.invariant_norm.normalized_shape[0]), **kwargs)

    @property
    def decay(self) -> Tensor:
        """``beta > 0``, 1/A."""
        return F.softplus(self.decay_raw)

    def forward(self, node_invariants: Tensor, edge_index: Tensor, edge_len: Tensor) -> TransferGraph:
        """Build the tunnelling graph from the model's directed neighbour list.

        ``edge_index`` holds both directions of every (image) pair, as TRACE's
        neighbour list does; one copy of each is kept (``src < dst``), pairs
        beyond ``r_max`` and self-image pairs are dropped.
        """
        from .energy import bond_envelope

        src, dst = edge_index[0], edge_index[1]
        keep = (src < dst) & (edge_len < self.r_max)
        lo, hi, r = src[keep], dst[keep], edge_len[keep]
        e = self.embed(self.norm(node_invariants))
        rbf = torch.exp(-(((r[:, None] - self.rbf_centers) / self.rbf_width) ** 2))
        bond_in = torch.cat((e[lo] + e[hi], e[lo] * e[hi], rbf), dim=-1)
        phi = self.bond_net(bond_in).squeeze(-1)
        modulation = HOPPING_LOG_RANGE * torch.tanh(phi / HOPPING_LOG_RANGE)
        log_h = self.log_hopping_scale - self.decay * (r - self.reference_length) + modulation
        hopping = bond_envelope(r, self.r_max, onset=self.envelope_onset) * torch.exp(log_h)
        site = self.site_net(e)
        gap_excess = F.softplus(site[:, 0] + self.gap_shift) + self.gap_floor
        return TransferGraph(
            bond_index=torch.stack((lo, hi)),
            hopping=hopping,
            gap_excess=gap_excess,
            amplitude_logit=site[:, 1],
        )


def _amplitude(logit: Tensor, mask: Tensor) -> Tensor:
    """``sqrt(softmax(s))`` on the masked atoms, zero elsewhere (sum of squares 1).

    The softmax runs over the fragment's atoms only: there every probability
    is strictly positive, so ``sqrt`` is differentiable.  (A full-length
    softmax with ``-inf`` outside the fragment puts ``sqrt`` at 0, and its
    backward is ``inf * 0 = NaN``.)
    """
    members = torch.nonzero(mask).reshape(-1)
    out = torch.zeros_like(logit)
    return out.index_put((members,), torch.sqrt(torch.softmax(logit[members], dim=0)))


def bridge_coupling(
    graph: TransferGraph,
    donor: Tensor,
    acceptor: Tensor,
    *,
    dense_max: int = DENSE_BRIDGE_MAX,
    rtol: float = 1e-12,
) -> Tensor:
    """``V_DA`` of eq. (1), eV, differentiable in every input.

    Args:
        graph: the tunnelling graph.
        donor, acceptor: ``[n_atoms]`` bool masks of the two fragments (disjoint,
            non-empty).
        dense_max: bridges with more sites are solved by CG (implicit derivatives).
        rtol: relative residual of the CG route.
    """
    donor = donor.reshape(-1).to(torch.bool)
    acceptor = acceptor.reshape(-1).to(torch.bool)
    n = graph.n_atoms
    if donor.numel() != n or acceptor.numel() != n:
        raise ValueError("donor and acceptor masks must have one entry per atom")
    if not bool(donor.any()) or not bool(acceptor.any()):
        raise ValueError("the donor and the acceptor must each contain at least one atom")
    if bool((donor & acceptor).any()):
        raise ValueError("the donor and acceptor fragments overlap")
    w_d = _amplitude(graph.amplitude_logit, donor)
    w_a = _amplitude(graph.amplitude_logit, acceptor)
    hop_d = graph.hop(w_d)                    # H w_D: donor neighbours
    hop_a = graph.hop(w_a)
    direct = (w_d * hop_a).sum()              # w_D^T H_DA w_A
    bridge = ~(donor | acceptor)
    n_bridge = int(bridge.sum())
    if n_bridge == 0:
        return direct
    u = hop_d[bridge]                         # H_BD w_D
    v = hop_a[bridge]                         # H_BA w_A
    gaps = graph.gaps()
    index = torch.full((n,), -1, dtype=torch.long, device=gaps.device)
    index[bridge] = torch.arange(n_bridge, device=gaps.device)
    lo, hi = graph.bond_index
    inside = bridge[lo] & bridge[hi]
    b_lo, b_hi, b_h = index[lo[inside]], index[hi[inside]], graph.hopping[inside]
    if n_bridge <= dense_max:
        M = torch.diag(gaps[bridge])
        M = M.index_put((b_lo, b_hi), -b_h, accumulate=True)
        M = M.index_put((b_hi, b_lo), -b_h, accumulate=True)
        factor = torch.linalg.cholesky(M)     # SPD by eq. (2); cannot fail
        x = torch.cholesky_solve(v.unsqueeze(-1), factor).squeeze(-1)
        return direct + (u * x).sum()

    def factory(tensors):
        diag, hop, i, j = tensors[0], tensors[1], b_lo, b_hi

        def product(vector: Tensor) -> Tensor:
            out = diag * vector
            out = out.index_add(0, i, -hop * vector[j])
            return out.index_add(0, j, -hop * vector[i])

        return product

    diag = gaps[bridge]
    problem = QuadraticProblem(
        factory,
        (diag, b_h),
        precondition=lambda r: r / diag.detach(),
        rtol=rtol,
        name="bridge Green's function",
    )
    x, _ = minimise(problem, -v)
    return direct + (u * x).sum()


def _fragment_charges(state: Mapping[str, Any]) -> Tensor:
    return torch.as_tensor(state["fragment_charges"], dtype=torch.float64).reshape(-1)


def coupling_matrix(
    graph: TransferGraph,
    states: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 1e-9,
    **solve_options,
) -> Tensor:
    """The EVB off-diagonal ``[n_states, n_states]``: ``-V_ab`` for one-electron moves.

    All states must share one fragment partition (``fragment_id``).  States
    whose fragment charges differ by ``+1`` on one fragment and ``-1`` on
    another couple through :func:`bridge_coupling`; all other pairs are 0.
    """
    n_states = len(states)
    if n_states < 2:
        raise ValueError("a coupling matrix needs at least two states")
    fragment_id = torch.as_tensor(states[0]["fragment_id"]).reshape(-1)
    for state in states[1:]:
        other = torch.as_tensor(state["fragment_id"]).reshape(-1).to(fragment_id.device)
        if other.shape != fragment_id.shape or bool((other != fragment_id).any()):
            raise ValueError(
                "through-bond coupling needs every diabatic state on the same fragment "
                "partition (identical fragment_id); the states may differ only in "
                "fragment_charges"
            )
    if fragment_id.numel() != graph.n_atoms:
        raise ValueError(
            f"fragment_id has {fragment_id.numel()} entries for {graph.n_atoms} atoms"
        )
    fragment_id = fragment_id.to(graph.gap_excess.device)
    charges = [_fragment_charges(s) for s in states]
    entries = {}
    for a in range(n_states):
        for b in range(a + 1, n_states):
            if charges[a].shape != charges[b].shape:
                raise ValueError("diabatic states list different numbers of fragments")
            diff = charges[b] - charges[a]
            moved = torch.nonzero(diff.abs() > tolerance).reshape(-1).tolist()
            if not moved:
                raise ValueError(f"diabatic states {a} and {b} have identical fragment charges")
            if len(moved) != 2:
                continue
            x, y = moved
            if abs(float(diff[x]) + float(diff[y])) > tolerance or abs(abs(float(diff[x])) - 1.0) > tolerance:
                continue
            V = bridge_coupling(graph, fragment_id == x, fragment_id == y, **solve_options)
            entries[(a, b)] = V
    dtype, device = graph.gap_excess.dtype, graph.gap_excess.device
    matrix = torch.zeros(n_states, n_states, dtype=dtype, device=device)
    for (a, b), V in entries.items():
        e = torch.zeros(n_states, n_states, dtype=dtype, device=device)
        e[a, b] = 1.0
        e[b, a] = 1.0
        matrix = matrix - V * e
    return matrix
