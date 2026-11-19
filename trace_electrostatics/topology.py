"""Charge topology for the TRACE split-charge electrostatics model.

The split-charge equilibration (SQE) parameterisation writes the atomic charge
as a sum of *bond* variables plus a constraint-carrying reference charge
(SPEC 1.3),

    q_i = sum_{j in N(i)} t_ij + rho_i ,        t_ij = -t_ji .

The antisymmetry ``t_ij = -t_ji`` is the whole point of the parameterisation: it
makes charge transfer a *conserved flow* along bonds, so charge can never be
created by the network, only moved.  This module makes that antisymmetry
**structural** rather than a soft constraint, by storing one variable per
*undirected* bond and contracting it with a signed incidence matrix

    q = B t ,     B in {-1, 0, +1}^{n_atoms x n_bonds} ,     1^T B = 0  exactly.

The identity ``1^T B == 0`` is what guarantees that charge transfer conserves the
total charge to machine precision, so the only way net charge enters is through
``rho``, which is pinned by the per-fragment constraints.

Fragment definition is a **design input**, not something the model derives.  A
fragment is the chemical object whose net charge you are willing to assert
(a molecule, an ion, a solvated complex).  The model cannot invent it: the
audit failure that motivates per-fragment constraints (a +1 ion losing its
charge to distant spectators, 1.000 -> 0.016, SPEC 1.3) is precisely the
failure of asking a single global constraint to stand in for chemistry.
``assign_fragments`` therefore supports an explicit user-supplied labelling
first, and connected components of the bond graph only as a convenience default.

Periodic boundary conditions
----------------------------
A periodic neighbour list carries an integer cell offset per edge, and two
things follow that an open-boundary list never shows:

1. the same atom pair ``(i, j)`` appears through **several images**, at several
   different separations.  Those are ``m`` physically distinct charge-transfer
   channels at ``m`` distinct bond lengths, so they must be ``m`` bonds with
   ``m`` bond hardnesses.  Merging them into one bond keeps a single
   ``kappa_ij`` for a channel that has ``m`` parallel routes, which overstates
   the effective hardness by a factor of ``m`` and makes periodic charge
   transfer that much too stiff.  Pass ``edge_shift`` to
   :func:`build_bond_topology` / :func:`build_incidence` and each image becomes
   its own bond;
2. an atom is a neighbour of **its own image**, ``(i, i, n != 0)``.  See
   :func:`build_bond_topology` for what that channel is and why it is dropped.

Spin channels
-------------
SPEC 9 splits the charge sector into ``up`` and ``down``, giving a second bond
variable per bond and a second constraint per fragment,
``sum_{i in F} s_i = 2 S_F``.  Nothing about the *topology* changes: both
channels ride on the same incidence matrix ``B``, so the spin additions here are
the second constraint carrier (:func:`distribute_fragment_spins`), the block
diagonal two-constraint operator (:func:`spin_constraint_block`) and the basis
transforms (:func:`spin_channels_to_charge_spin` and its inverse).  Spin is
opt-in; with it off, every function below behaves exactly as it did.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from .diagnostics import ElectrostaticsAssertionError

__all__ = [
    "BondTopology",
    "build_bond_topology",
    "build_incidence",
    "assign_fragments",
    "count_fragments",
    "fragment_constraint_matrix",
    "distribute_fragment_charges",
    "distribute_fragment_spins",
    "spin_constraint_block",
    "spin_channels_to_charge_spin",
    "charge_spin_to_spin_channels",
    "map_edges_to_bonds",
    "symmetrise_edge_scalar",
    "antisymmetrise_edge_scalar",
]


# Fragment assignment strategies accepted by :func:`assign_fragments`.
FRAGMENT_METHOD_EXPLICIT = "explicit"
FRAGMENT_METHOD_CONNECTED = "connected"
FRAGMENT_METHOD_AUTO = "auto"


@dataclass(frozen=True)
class BondTopology:
    """Everything :func:`build_bond_topology` derives from a neighbour list.

    Attributes
    ----------
    incidence:
        ``[n_atoms, n_bonds]`` signed incidence ``B`` with ``q = B t``.
    bond_index:
        ``[2, n_bonds]`` ``(lo, hi)`` atom pair of each bond.  Under periodic
        boundary conditions the same ``(lo, hi)`` may appear several times --
        once per image -- and ``bond_shift`` is what distinguishes them.
    bond_shift:
        ``[n_bonds, 3]`` integer cell offset of each bond in the canonical
        ``lo -> hi`` orientation, or ``None`` when the list was not periodic.
    bond_slot:
        ``[n_bonds]`` index of one directed edge that produced each bond, so a
        caller can *gather* bond vectors and lengths from its own ``edge_vec``
        and ``edge_len`` rather than recomputing them from positions (which
        would drop the cell shift).  With ``edge_shift`` supplied each bond has
        at most two source edges, the two orientations of one image, so the
        choice is unambiguous; without it, the shortest edge wins, which is the
        minimum image.
    edge_slot:
        ``[n_edges]`` bond column of each directed edge, or ``n_bonds`` -- one
        past the end -- for an edge that carries no split charge.  See
        :func:`map_edges_to_bonds`.
    edge_sign:
        ``[n_edges]`` ``+1`` when the edge runs in the stored ``lo -> hi``
        orientation, ``-1`` when it runs the other way, ``0`` when it carries no
        split charge.
    n_self_image_edges:
        How many edges were dropped as atom-to-own-image channels.  Reported so
        a caller can tell "this system has no bonds" from "every channel in this
        system was a self-image one".
    """

    incidence: Tensor
    bond_index: Tensor
    bond_shift: Optional[Tensor]
    bond_slot: Tensor
    edge_slot: Tensor
    edge_sign: Tensor
    n_self_image_edges: int

    @property
    def n_atoms(self) -> int:
        return int(self.incidence.shape[0])

    @property
    def n_bonds(self) -> int:
        return int(self.incidence.shape[1])


def build_bond_topology(
    edge_index: Tensor,
    num_atoms: int,
    *,
    edge_shift: Optional[Tensor] = None,
    edge_len: Optional[Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    allow_no_bonds: bool = False,
    self_image_bonds: bool = False,
    sparse_incidence: bool = False,
) -> BondTopology:
    """Derive the bond variables of the SQE parameterisation from a neighbour list.

    Each *undirected* channel appears exactly once as a bond, so the split
    charge ``t_ij = -t_ji`` is represented by a single variable and the
    antisymmetry cannot be violated numerically.  For a bond stored as the
    ordered pair ``(lo, hi)`` with ``lo < hi`` the convention is

        B[lo, b] = +1 ,   B[hi, b] = -1 ,

    i.e. a positive ``t_b`` moves ``+t_b`` of charge onto the lower-indexed atom
    and ``-t_b`` onto the higher-indexed one.  Summing the column gives zero, so
    ``1^T B == 0`` structurally, and hence ``sum_i q_i = 0`` for any bond
    configuration whatsoever.

    What counts as one channel
    --------------------------
    Without ``edge_shift`` a channel is an atom pair: every listing of ``(i, j)``
    and ``(j, i)`` collapses onto one bond.  That is right for open boundaries,
    where a pair can only be a neighbour once.

    With ``edge_shift`` a channel is an atom pair **through one image**,
    ``(i, j, n)``, identified with its reverse ``(j, i, -n)``.  Under periodic
    boundary conditions the same pair is typically a neighbour through several
    images at several different separations -- with an ASE neighbour list at a
    chemical cutoff this is the rule, not the exception -- and those are
    physically ``m`` separate transfer channels of ``m`` different lengths.
    Collapsing them onto one bond leaves the channel with a single ``kappa_ij``
    where ``m`` parallel routes each carry their own, overstating the effective
    bond hardness ``m``-fold.

    Self-image channels
    -------------------
    An edge ``(i, i, n)`` with ``n != 0`` is atom ``i`` beside its own periodic
    image, and it **is** a real charge-transfer channel.  It is nevertheless
    dropped, for a structural reason rather than the naive one ("an atom cannot
    transfer charge to itself"):

    Under Born-von Karman periodicity the ``+n`` and ``-n`` images of atom ``i``
    are the same atom, so the incidence column of such a bond is ``+1`` and
    ``-1`` on the *same row*: identically zero.  Charge that leaves atom ``i``
    across the cell boundary in direction ``+n`` re-enters at ``-n``.  The
    variable is therefore a divergence-free circulating current: it changes no
    atomic charge, its rows and columns of ``B^T J B`` and ``B^T T01`` vanish so
    it is decoupled from every other variable, and its only contribution to the
    energy is ``+0.5 kappa t^2 >= 0``, minimised at ``t = 0``.  Keeping such
    variables is exactly equivalent to dropping them at the exact minimum, and
    strictly *worse* under the truncated fixed-k solve, where a non-zero seed
    would leave spurious positive energy behind.

    What that does cost is real but out of scope here: a circulating current
    carries macroscopic polarization in the modern (Berry-phase) theory, since
    it transports charge through the cell boundary.  Capturing it needs the
    electric-enthalpy formulation, which this package deliberately does not
    implement -- a uniform field under periodic boundaries is rejected upstream
    for exactly that reason, and the periodic linear response is not taken from
    a finite dipole-derivative matrix.  Within the periodic treatment this
    package *does* implement, nothing computed is lost.

    The case this makes loud
    ------------------------
    In a cell where **every** channel is a self-image one -- a monatomic
    primitive cell, e.g. Cu fcc at a 6 A cutoff, 78 edges, all self-image --
    dropping them leaves ``n_bonds = 0``.  There is then no ``t`` at all,
    ``q = rho`` is frozen at the fragment target, and both the SQE mechanism and
    the charge-transfer polarizability are gone.  For a monatomic primitive cell
    that is also the symmetry answer (every site is equivalent, so every charge
    must be equal), but it is not what a caller asking for charge equilibration
    in a metal expects, and silence about it is how the result gets believed.
    This raises instead; pass ``allow_no_bonds=True`` to accept frozen charges
    deliberately.

    Parameters
    ----------
    edge_index:
        ``[2, n_edges]`` integer tensor of directed edges ``(src, dst)``.  Both
        directions may be present and duplicates are tolerated.
    num_atoms:
        Number of atoms ``n``; sets the number of rows of ``B``.
    edge_shift:
        Optional ``[n_edges, 3]`` integer cell offsets, with the convention that
        edge ``e`` connects atom ``src[e]`` to the image of ``dst[e]``
        translated by ``edge_shift[e] @ cell``.  The reverse of ``(i, j, n)`` is
        then ``(j, i, -n)``.
    edge_len:
        Optional ``[n_edges]`` edge lengths, used only to pick the minimum-image
        representative edge for ``bond_slot`` when ``edge_shift`` is not given.
    dtype:
        Floating dtype of ``B``.  Defaults to ``torch.get_default_dtype()``;
        SPEC 3 requires float64 for the whole pipeline.
    allow_no_bonds:
        Accept a topology in which every channel was a self-image one.
    self_image_bonds:
        KEEP the self-image channels ``(i, i, n)`` as bond variables (SPEC 15),
        one per ``+-n`` pair, with an identically zero incidence column.  They
        change no atomic charge, but under the bond-current polarization
        ``Omega P = ... + sum_b f_b tau_b d_b`` they carry exactly the
        polarization the paragraph above says is lost -- the dielectric
        response of a crystal that the atomic charges cannot see (a monatomic
        crystal has q = 0 by symmetry and still polarizes).  With no field they
        stay at zero: decoupled from every other variable, with a zero seed and
        a zero linear term.  Requires ``edge_shift``.
    sparse_incidence:
        Return ``B`` as a sparse COO tensor (SPEC 16).  A dense ``[n_atoms,
        n_bonds]`` matrix is 640 MB at 4000 atoms and ~15 bonds per atom, and
        the matrix-free solve never needs it.
    """
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2, n_edges], got {tuple(edge_index.shape)}"
        )
    if num_atoms <= 0:
        raise ValueError(f"num_atoms must be positive, got {num_atoms}")

    dtype = torch.get_default_dtype() if dtype is None else dtype
    device = edge_index.device
    ei = edge_index.long()
    n_edges = int(ei.shape[1])

    if ei.numel() > 0:
        if int(ei.min()) < 0 or int(ei.max()) >= num_atoms:
            raise ValueError(
                "edge_index contains atom indices outside [0, num_atoms); "
                f"num_atoms={num_atoms}, range=({int(ei.min())}, {int(ei.max())})"
            )

    shift = None
    if edge_shift is not None:
        shift = edge_shift.long().reshape(-1, 3)
        if int(shift.shape[0]) != n_edges:
            raise ValueError(
                f"edge_shift must have one row per edge: got {int(shift.shape[0])} "
                f"rows for {n_edges} edges"
            )

    src, dst = ei[0], ei[1]
    keep = src != dst  # see "Self-image channels" above
    n_self_image = n_edges - int(keep.sum())

    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    forward = src <= dst
    if self_image_bonds:
        if shift is None:
            raise ValueError(
                "self_image_bonds=True needs edge_shift: a self-image channel is "
                "defined by its lattice offset."
            )
        self_image = (src == dst) & (shift.abs().sum(dim=1) > 0)
        keep = keep | self_image
        # (i, i, n) and (i, i, -n) are one channel. Canonical orientation: the
        # offset whose first non-zero component is positive.
        first_nonzero = torch.argmax((shift != 0).to(torch.int64), dim=1)
        leading = shift.gather(1, first_nonzero.unsqueeze(1)).squeeze(1)
        forward = torch.where(self_image, leading > 0, forward)
    rows = [lo, hi]
    if shift is not None:
        # Canonical orientation is lo -> hi, so flip the offset on reversed edges.
        canonical_shift = torch.where(forward.unsqueeze(1), shift, -shift)
        rows.extend(canonical_shift.unbind(dim=1))
    keys = torch.stack(rows, dim=1)[keep]

    unique_keys, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
    n_bonds = int(unique_keys.shape[0])

    if n_bonds == 0 and n_self_image > 0 and not allow_no_bonds:
        raise ElectrostaticsAssertionError(
            f"every one of the {n_self_image} charge-transfer channels in this "
            "system is an atom-to-own-periodic-image pair, so no bond variables "
            "survive and n_bonds = 0. With no t at all, q = rho is frozen at the "
            "fragment target: the SQE mechanism and the charge-transfer "
            "polarizability are both gone, and the solve would report success. "
            "Such a channel is a divergence-free circulating current -- its "
            "incidence column is +1 and -1 on the same row -- so it cannot be "
            "represented as a bond variable here, and the polarization it does "
            "carry needs the Berry-phase formulation this package does not "
            "implement. Use a cell with more than one symmetry-distinct site (a "
            "supercell, a surface, an alloy, a defect), or pass "
            "allow_no_bonds=True to accept frozen charges deliberately."
        )

    bond_index = unique_keys[:, :2].transpose(0, 1).contiguous()
    bond_shift = unique_keys[:, 2:].contiguous() if shift is not None else None

    slots = torch.arange(n_bonds, device=device)
    if sparse_incidence:
        # Self-image columns are simply absent: an exact zero column.
        proper = bond_index[0] != bond_index[1]
        rows = torch.cat([bond_index[0][proper], bond_index[1][proper]])
        columns = torch.cat([slots[proper], slots[proper]])
        values = torch.cat(
            [torch.ones(int(proper.sum()), dtype=dtype, device=device),
             -torch.ones(int(proper.sum()), dtype=dtype, device=device)]
        )
        incidence = torch.sparse_coo_tensor(
            torch.stack([rows, columns]), values, (num_atoms, n_bonds)
        ).coalesce()
    else:
        incidence = torch.zeros(num_atoms, n_bonds, dtype=dtype, device=device)
        incidence[bond_index[0], slots] = 1.0
        incidence[bond_index[1], slots] = -1.0
        # A kept self-image bond has lo == hi: charge leaving across +n
        # re-enters at -n, so its column is exactly zero (the writes collided).
        incidence[:, bond_index[0] == bond_index[1]] = 0.0

    # 1^T B == 0 is structural: bond_index[0] < bond_index[1] on every column by
    # construction, so the +1 and the -1 always land on different rows. The check
    # stays as a tripwire against a future edit that reintroduces a repeated atom
    # index (which is how the self-image channels above would come back), and is
    # affordable because this whole module is preprocessing, deliberately outside
    # the AOT-compiled hot path.
    column_sums = (
        torch.sparse.sum(incidence, dim=0).to_dense()
        if incidence.is_sparse and n_bonds > 0
        else incidence.to_dense().sum(dim=0) if incidence.is_sparse else incidence.sum(dim=0)
    )
    if not torch.equal(column_sums, torch.zeros_like(column_sums)):
        raise ElectrostaticsAssertionError(
            "1^T B != 0: the incidence matrix does not conserve charge exactly. "
            "This can only happen if a bond was registered with a repeated atom "
            "index, which would break the structural antisymmetry t_ij = -t_ji."
        )

    # Per-edge map. Dropped edges point one past the end, which the edge-scalar
    # reductions read as "discard" (see map_edges_to_bonds).
    edge_slot = torch.full((n_edges,), n_bonds, dtype=torch.long, device=device)
    edge_slot[keep] = inverse.reshape(-1)
    edge_sign = torch.zeros(n_edges, dtype=dtype, device=device)
    edge_sign[keep] = torch.where(forward[keep], 1.0, -1.0).to(dtype)

    bond_slot = _representative_edge(
        edge_slot, keep, n_bonds, edge_len=edge_len, exact=shift is not None
    )

    return BondTopology(
        incidence=incidence,
        bond_index=bond_index,
        bond_shift=bond_shift,
        bond_slot=bond_slot,
        edge_slot=edge_slot,
        edge_sign=edge_sign,
        n_self_image_edges=n_self_image,
    )


def _representative_edge(
    edge_slot: Tensor,
    keep: Tensor,
    n_bonds: int,
    *,
    edge_len: Optional[Tensor],
    exact: bool,
) -> Tensor:
    """One directed edge per bond, to gather bond geometry from.

    ``exact`` says the bonds are shift-resolved, so every edge mapping onto a
    bond describes the *same* image and any of them will do; the smallest edge
    index is picked for determinism.  Otherwise several images share the bond
    and the shortest edge is picked, which is the minimum image.

    Both reductions are ``scatter_reduce(amin)``, which is order independent and
    therefore deterministic on every backend.  Relying on "last write wins" with
    duplicate scatter indices is not: torch documents it as non-deterministic,
    and it measurably is on MPS.
    """
    device = edge_slot.device
    sentinel = torch.iinfo(torch.long).max
    if n_bonds == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    ids = torch.arange(edge_slot.numel(), device=device)
    slot_v, ids_v = edge_slot[keep], ids[keep]

    candidates = ids_v
    if not exact and edge_len is not None:
        lengths = edge_len.reshape(-1)[ids_v]
        infinity = torch.full(
            (n_bonds,), float("inf"), dtype=lengths.dtype, device=device
        )
        shortest = infinity.scatter_reduce(
            0, slot_v, lengths, reduce="amin", include_self=True
        )
        # Exact equality is correct: the minimum is one of the values itself.
        attains = lengths == shortest.index_select(0, slot_v)
        candidates = ids_v.masked_fill(~attains, sentinel)

    return torch.full((n_bonds,), sentinel, dtype=torch.long, device=device).scatter_reduce(
        0, slot_v, candidates, reduce="amin", include_self=True
    )


def build_incidence(
    edge_index: Tensor,
    num_atoms: int,
    *,
    edge_shift: Optional[Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    allow_no_bonds: bool = False,
) -> Tuple[Tensor, Tensor]:
    """``(B, bond_index)`` from a neighbour list; see :func:`build_bond_topology`.

    This is the two-value view of :func:`build_bond_topology`, kept for callers
    that only need the incidence matrix and the bond list.  Everything the
    periodic case needs -- the per-bond cell offsets, and the edge each bond
    should gather its geometry from -- is on :class:`BondTopology`, so under
    periodic boundary conditions call that instead.

    Returns
    -------
    (B, bond_index):
        ``B`` of shape ``[n_atoms, n_bonds]`` and ``bond_index`` of shape
        ``[2, n_bonds]`` holding the ``(lo, hi)`` atom pair of each bond, sorted
        lexicographically so the bond ordering is deterministic across calls.
    """
    topology = build_bond_topology(
        edge_index,
        num_atoms,
        edge_shift=edge_shift,
        dtype=dtype,
        allow_no_bonds=allow_no_bonds,
    )
    return topology.incidence, topology.bond_index


def assign_fragments(
    num_atoms: int,
    bond_index: Optional[Tensor] = None,
    *,
    fragment_id: Optional[Tensor] = None,
    method: str = FRAGMENT_METHOD_AUTO,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Assign every atom to a charge fragment.

    A *fragment* is the unit whose net charge is asserted as a constraint,

        sum_{i in F} q_i = Q_F        (SPEC 1.3)

    **Fragment definition is a design input.**  It encodes chemistry the model
    is not able to infer: whether a solvated proton belongs to the solute or the
    solvent, whether a physisorbed molecule shares electrons with a slab.  A
    wrong fragmentation is a wrong physical statement, not a numerical detail --
    replacing per-fragment constraints by a single global one made a +1 ion
    leak its charge to spectators 1000 A away (1.000 -> 0.016 at 200
    spectators; SPEC 1.3).  Prefer ``method="explicit"`` whenever the chemistry
    is known.

    Parameters
    ----------
    num_atoms:
        Number of atoms.
    bond_index:
        ``[2, n_bonds]`` undirected bond list from :func:`build_incidence`.
        Required for ``method="connected"``.
    fragment_id:
        ``[n_atoms]`` integer labels supplied by the user.  Labels may be
        arbitrary integers; they are relabelled to a contiguous ``0..n_frag-1``
        range **in increasing order of the original label value**.  So
        ``[5, 2, 5, 2, 9] -> [1, 0, 1, 0, 2]``: fragment 0 is the one labelled
        ``2``, not the one containing atom 0.  This matters because
        ``fragment_charges`` is indexed by the *returned* labels -- order the
        targets by sorted original label, or (simplest) hand in labels that are
        already ``0..n_frag-1``, which is then the identity.  For
        ``method="connected"`` the two readings coincide: a component is
        labelled by its lowest atom index, so sorting by label value is sorting
        by lowest atom index.
    method:
        ``"explicit"`` uses ``fragment_id``; ``"connected"`` uses the connected
        components of the bond graph; ``"auto"`` (default) uses ``fragment_id``
        if given and falls back to connected components.
    device:
        Device of the returned labels; defaults to the device of the provided
        inputs.

    Returns
    -------
    Tensor
        ``[n_atoms]`` int64 fragment labels in ``0..n_frag-1``.

    Notes
    -----
    Connected components are found by label propagation with pointer jumping.
    The loop is bounded by ``num_atoms`` and terminates on a fixed point; it is
    a *preprocessing* step, deliberately excluded from the AOT-compiled hot
    path (SPEC 2), where fragment labels are an input, not a computation.
    """
    if num_atoms <= 0:
        raise ValueError(f"num_atoms must be positive, got {num_atoms}")

    if method == FRAGMENT_METHOD_AUTO:
        method = (
            FRAGMENT_METHOD_EXPLICIT if fragment_id is not None else FRAGMENT_METHOD_CONNECTED
        )

    if method == FRAGMENT_METHOD_EXPLICIT:
        if fragment_id is None:
            raise ValueError("method='explicit' requires fragment_id")
        labels = fragment_id.long().reshape(-1)
        if labels.numel() != num_atoms:
            raise ValueError(
                f"fragment_id must have {num_atoms} entries, got {labels.numel()}"
            )
    elif method == FRAGMENT_METHOD_CONNECTED:
        if bond_index is None:
            raise ValueError("method='connected' requires bond_index")
        labels = _connected_components(bond_index, num_atoms)
    else:
        raise ValueError(
            f"unknown fragment method {method!r}; expected one of "
            f"{FRAGMENT_METHOD_EXPLICIT!r}, {FRAGMENT_METHOD_CONNECTED!r}, "
            f"{FRAGMENT_METHOD_AUTO!r}"
        )

    if device is not None:
        labels = labels.to(device)

    # Relabel to a contiguous range. torch.unique sorts by label VALUE, so
    # fragment 0 is the one carrying the smallest original label -- which is the
    # component containing the lowest atom index only because
    # _connected_components labels each component by exactly that.
    _, compact = torch.unique(labels, sorted=True, return_inverse=True)
    return compact.long().reshape(-1)


def _connected_components(bond_index: Tensor, num_atoms: int) -> Tensor:
    """Connected components of the undirected bond graph (label propagation)."""
    device = bond_index.device
    labels = torch.arange(num_atoms, device=device, dtype=torch.long)
    if bond_index.numel() == 0:
        return labels

    src = bond_index[0].long()
    dst = bond_index[1].long()
    for _ in range(num_atoms):
        propagated = labels.clone()
        propagated.scatter_reduce_(0, dst, labels[src], reduce="amin", include_self=True)
        propagated.scatter_reduce_(0, src, labels[dst], reduce="amin", include_self=True)
        # Pointer jumping: labels always point at a node index, so composing the
        # map with itself collapses chains in log(depth) rounds.
        propagated = propagated[propagated]
        if torch.equal(propagated, labels):
            break
        labels = propagated
    return labels


def count_fragments(fragment_id: Tensor) -> int:
    """Number of fragments in a contiguous labelling (``max + 1``)."""
    if fragment_id.numel() == 0:
        return 0
    return int(fragment_id.max()) + 1


def fragment_constraint_matrix(
    fragment_id: Tensor,
    num_fragments: Optional[int] = None,
    *,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Build the per-fragment charge constraint matrix ``C``.

    ``C[f, i] = 1`` when atom ``i`` belongs to fragment ``f``, so the physical
    constraint ``sum_{i in F} q_i = Q_F`` for every fragment reads

        C q = Q .

    One row per fragment -- **not** a single all-ones row.  The single-row
    (global) version is size-inconsistent: it lets an ion discharge into
    arbitrarily distant spectators because only the total is pinned
    (SPEC 1.3).

    Returns
    -------
    Tensor
        ``[n_fragments, n_atoms]`` matrix of ones and zeros.
    """
    labels = fragment_id.long().reshape(-1)
    n_atoms = int(labels.numel())
    if num_fragments is None:
        num_fragments = count_fragments(labels)
    if n_atoms > 0 and (int(labels.min()) < 0 or int(labels.max()) >= num_fragments):
        raise ValueError(
            "fragment_id entries must lie in [0, num_fragments); got range "
            f"({int(labels.min())}, {int(labels.max())}) for num_fragments={num_fragments}"
        )

    dtype = torch.get_default_dtype() if dtype is None else dtype
    matrix = torch.zeros(num_fragments, n_atoms, dtype=dtype, device=labels.device)
    matrix[labels, torch.arange(n_atoms, device=labels.device)] = 1.0

    populations = matrix.sum(dim=1)
    if num_fragments > 0 and bool((populations == 0).any()):
        empty = torch.nonzero(populations == 0).reshape(-1).tolist()
        raise ValueError(
            f"fragments {empty} contain no atoms; an empty fragment has no "
            "charge to constrain and makes the constraint system singular"
        )
    return matrix


def distribute_fragment_charges(
    constraint_matrix: Tensor,
    targets: Tensor,
    *,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Spread each fragment's net charge over its atoms as the reference ``rho``.

    ``rho`` is the constraint-carrying part of ``q = B t + rho`` (SPEC 1.3).  It
    exists because ``1^T B = 0``: bond variables can move charge but can never
    supply the net charge of a charged fragment, so something must.

    ``rho`` is *not* a variational degree of freedom and *not* a geometry
    dependent readout ``q0_i(h_i)``.  Both would be degenerate with ``t``: with
    a free per-atom ``rho`` the minimiser is ``t = 0`` exactly (any ``t`` can be
    absorbed into ``rho`` at the same ``q`` while saving the bond-hardness cost
    ``0.5 kappa t^2``), which collapses the model to global QEq and destroys the
    locality mechanism -- the same degeneracy SPEC 1.3 records for ``q0``.
    Within a fragment whose bond graph is connected the split is immaterial at
    the solution, since ``t`` redistributes charge freely; it matters only for
    fragments whose bonds do not connect all their atoms, where a uniform split
    is the least-committal choice.

    Parameters
    ----------
    constraint_matrix:
        ``[n_fragments, n_atoms]`` matrix from :func:`fragment_constraint_matrix`.
    targets:
        ``[n_fragments]`` net charge ``Q_F`` of each fragment, in units of ``e``.
    weights:
        Optional ``[n_atoms]`` non-negative weights (e.g. per-element
        electron counts).  Defaults to a uniform split inside each fragment.

    Returns
    -------
    Tensor
        ``[n_atoms]`` reference charges satisfying ``C rho = Q`` to round-off.
    """
    if constraint_matrix.dim() != 2:
        raise ValueError(
            f"constraint_matrix must be 2-D, got {tuple(constraint_matrix.shape)}"
        )
    n_fragments, n_atoms = constraint_matrix.shape
    targets = targets.reshape(-1)
    if targets.numel() != n_fragments:
        raise ValueError(
            f"targets must have {n_fragments} entries, got {targets.numel()}"
        )

    if weights is None:
        weights = torch.ones(n_atoms, dtype=constraint_matrix.dtype, device=constraint_matrix.device)
    weights = weights.reshape(-1)
    if weights.numel() != n_atoms:
        raise ValueError(f"weights must have {n_atoms} entries, got {weights.numel()}")
    if bool((weights < 0).any()):
        raise ValueError("weights must be non-negative")

    weighted = constraint_matrix * weights.unsqueeze(0)
    normalisation = weighted.sum(dim=1)
    if bool((normalisation == 0).any()):
        raise ValueError(
            "a fragment received zero total weight; its net charge cannot be placed"
        )
    share = targets / normalisation
    return (weighted * share.unsqueeze(1)).sum(dim=0)


# ---------------------------------------------------------------------------
# Spin-resolved charge transfer (SPEC 9)
# ---------------------------------------------------------------------------
#
# SPEC 9 splits the charge sector, and only the charge sector:
#
#     q_i^sigma = q_i^(0)sigma + sum_j t_ij^sigma + rho_i^sigma ,  sigma in {up, down}
#     q_i = q_i^up + q_i^down        (charge)
#     s_i = q_i^up - q_i^down        (spin population)
#
# The induced dipole responds to the TOTAL density and is not spin resolved.
#
# This package carries the bond variables in the (charge, spin) basis
#
#     t^c = t^up + t^down ,      t^s = t^up - t^down
#
# rather than as the two channels, because that basis is the one in which the
# physics is block diagonal and in which the closed-shell limit is exact:
#
# * q = B t^c + rho^c and s = B t^s + rho^s, with the SAME incidence matrix B,
#   so the two per-fragment constraints of SPEC 9 are two copies of one
#   structural operator C B (see :func:`spin_constraint_block`);
# * the Coulomb kernel couples TOTAL charges, so it touches t^c only.  In the
#   (up, down) basis every J_ij would appear in all four spin blocks, and the
#   single most likely physics error in this feature -- running J through the
#   spin channels separately -- becomes invisible.  Here it would be a term
#   that plainly is not there;
# * with t^s == 0 the charge sector is bit-for-bit the closed-shell one.
#
# The two bases are related by :func:`spin_channels_to_charge_spin` and
# :func:`charge_spin_to_spin_channels`, which are exact inverses.

def distribute_fragment_spins(
    constraint_matrix: Tensor,
    spin_targets: Tensor,
    *,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Spread each fragment's spin ``2 S_F`` over its atoms as the reference ``rho^s``.

    The spin analogue of :func:`distribute_fragment_charges`, and it carries the
    **factor of two** that SPEC 9 puts in the constraint:

        sum_{i in F} s_i = 2 S_F .

    ``spin_targets`` is ``S_F``, the total spin quantum number of the fragment
    (0 for a closed shell, 1/2 for a doublet, 1 for a triplet).  The quantity
    actually pinned is ``2 S_F`` -- the number of unpaired electrons, i.e.
    ``multiplicity - 1`` -- because ``s_i = q_i^up - q_i^down`` counts electrons,
    not spin.  Handing ``2 S_F`` in where ``S_F`` belongs doubles every moment
    in the system and is not detectable downstream, so the distinction is made
    once, here, and nowhere else.

    Like ``rho``, ``rho^s`` is a **fixed constraint carrier, not a variational
    degree of freedom** (SPEC 1.3).  The GENERAL PRINCIPLE recorded there --
    any per-atom charge channel that bypasses the bond-hardness penalty absorbs
    everything and drives ``t -> 0`` -- applies verbatim to the spin channel: a
    free per-atom ``rho^s`` would make ``t^s`` collapse the same way a free
    ``rho`` made ``t`` collapse to ``|t| = 1.65e-07``.

    No integrality check is made on ``2 S_F``.  Fractional spin is exactly the
    regime SPEC 9.2 is about, and a solver that refused it could not measure its
    own flat-plane error.

    Returns
    -------
    Tensor
        ``[n_atoms]`` reference spin populations satisfying ``C rho^s = 2 S``.
    """
    spin_targets = spin_targets.reshape(-1)
    return distribute_fragment_charges(
        constraint_matrix, 2.0 * spin_targets, weights=weights
    )


def spin_constraint_block(constraint_t: Tensor) -> Tensor:
    """Stack the bond-sector constraint ``C B`` for the ``(t^c, t^s)`` variables.

    SPEC 9 asserts two constraints per fragment,

        sum_{i in F} q_i = Q_F ,        sum_{i in F} s_i = 2 S_F ,

    and both are carried by ``rho`` and ``rho^s``, so what is left acting on the
    bond variables is the *homogeneous* pair ``C B t^c = 0``, ``C B t^s = 0``.
    Since ``q`` and ``s`` are built from the same incidence matrix, the two
    constraints are the same operator applied to two different vectors, and the
    combined constraint is block diagonal::

        A_spin = [ C B    0   ]        [2 n_fragments, 2 n_bonds]
                 [  0    C B  ]

    Block diagonal, not stacked into one wider row block: a single row
    ``[C B, C B]`` would only pin the *sum* ``t^c + t^s = 2 t^up`` and would
    leave the down channel free, which is the spin-sector version of the
    size-inconsistency failure of a single global charge constraint (SPEC 1.3).

    The projector onto the null space of this matrix is correspondingly block
    diagonal, ``P = diag(P_A, P_A)`` with ``P_A = I - A^+ A``, which is why
    :func:`~trace_electrostatics.solver.solve_charges` factorises ``C B`` once
    and applies it to both halves rather than forming this matrix in the hot
    path.  It *is* formed for the SPEC 2.1 curvature assertions, which need the
    constraint in the same space as the charge-sector Hessian block.

    Each ``C B`` block is rank deficient in exactly the same way the
    closed-shell one is -- its rows sum to ``1^T B = 0`` -- so the rank of the
    pair is ``2 rank(C B)``, never ``2 n_fragments``.
    """
    if constraint_t.dim() != 2:
        raise ValueError(
            f"constraint_t must be 2-D [n_fragments, n_bonds], got {tuple(constraint_t.shape)}"
        )
    n_rows, n_cols = constraint_t.shape
    zero = torch.zeros_like(constraint_t)
    return torch.cat(
        [
            torch.cat([constraint_t, zero], dim=1),
            torch.cat([zero, constraint_t], dim=1),
        ],
        dim=0,
    )


def spin_channels_to_charge_spin(up: Tensor, down: Tensor) -> Tuple[Tensor, Tensor]:
    """``(up, down) -> (charge, spin)``: ``c = up + down``, ``s = up - down``.

    The basis this package works in.  Applies to the bond variables
    (``t^c = t^up + t^down``), to the reference charges and to the atomic
    populations alike -- they are all the same linear map.

    It does **not** apply to the *parameters*: the linear term
    ``chi^up q^up + chi^down q^down`` equals ``chi q + chi^s s`` with
    ``chi = (chi^up + chi^down)/2`` and ``chi^s = (chi^up - chi^down)/2``, i.e.
    the potentials transform with the inverse-transpose (a factor of 1/2, not
    of 1).  Use
    :func:`~trace_electrostatics.solver.split_spin_potentials` for those, and do
    not reach for this function because the words look the same.
    """
    return up + down, up - down


def charge_spin_to_spin_channels(charge: Tensor, spin: Tensor) -> Tuple[Tensor, Tensor]:
    """``(charge, spin) -> (up, down)``: the exact inverse of
    :func:`spin_channels_to_charge_spin`, ``up = (c + s)/2``, ``down = (c - s)/2``.

    Use it to report per-spin populations or per-spin bond transfers from a
    solve, which works in the ``(c, s)`` basis throughout.
    """
    half = 0.5 * charge
    half_spin = 0.5 * spin
    return half + half_spin, half - half_spin


def map_edges_to_bonds(
    edge_index: Tensor,
    bond_index: Tensor,
    num_atoms: int,
    *,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    """Map directed edges onto undirected bond slots with an orientation sign.

    Returns ``(slot, sign)`` with ``slot[e]`` the bond column of edge ``e`` and
    ``sign[e] = +1`` when the edge runs ``lo -> hi`` (the stored orientation),
    ``-1`` when it runs ``hi -> lo``, and ``0`` for an edge that carries no
    split charge -- a self-image pair (see :func:`build_bond_topology`).

    An edge that carries no split charge gets ``slot = n_bonds``, one past the
    last bond, **not** an arbitrary in-range column.  That sentinel is what lets
    :func:`symmetrise_edge_scalar` and :func:`antisymmetrise_edge_scalar` throw
    its value away with an ordinary ``index_add`` into one extra bin, with no
    mask, no branch and no host synchronisation.  Indexing anything else with
    ``slot`` therefore has to filter on ``sign != 0`` first.

    ``dtype`` is the floating dtype of ``sign``; it defaults to
    ``torch.get_default_dtype()`` and should be set to the dtype of the edge
    scalars it will multiply.

    Note this function keys on the atom pair alone.  Where ``bond_index`` was
    built shift-aware, several of its columns share one ``(lo, hi)`` and cannot
    be told apart from the pair -- use :attr:`BondTopology.edge_slot` and
    :attr:`BondTopology.edge_sign`, which are built alongside the bonds and are
    exact.
    """
    if bond_index.dim() != 2 or bond_index.shape[0] != 2:
        raise ValueError(
            f"bond_index must have shape [2, n_bonds], got {tuple(bond_index.shape)}"
        )
    dtype = torch.get_default_dtype() if dtype is None else dtype
    ei = edge_index.long()
    src, dst = ei[0], ei[1]
    n_bonds = int(bond_index.shape[1])
    if n_bonds == 0:
        return torch.zeros_like(src), torch.zeros(src.shape, dtype=dtype, device=src.device)

    self_loop = src == dst
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    edge_key = lo * num_atoms + hi
    bond_key = bond_index[0].long() * num_atoms + bond_index[1].long()

    order = torch.argsort(bond_key)
    sorted_key = bond_key[order]
    position = torch.searchsorted(sorted_key, edge_key).clamp(max=n_bonds - 1)
    found = sorted_key[position] == edge_key
    if bool((~found & ~self_loop).any()):
        missing = torch.nonzero(~found & ~self_loop).reshape(-1)[:5].tolist()
        raise ValueError(
            f"edges {missing} are not present in bond_index; build the bond list "
            "from the same edge list you are mapping"
        )

    valid = found & ~self_loop
    discard = torch.full_like(src, n_bonds)
    slot = torch.where(valid, order[position], discard)
    orientation = torch.where(src < dst, 1.0, -1.0).to(dtype)
    zero = torch.zeros((), dtype=dtype, device=orientation.device)
    # torch.where, not a multiply by the mask: -1.0 * 0.0 is -0.0, which
    # compares equal to zero but prints as a sign the caller has no use for.
    return slot, torch.where(valid, orientation, zero)


def _reduce_edge_scalar(
    weighted: Tensor, slot: Tensor, n_bonds: int, counted: Tensor
) -> Tensor:
    """Mean of ``weighted`` over the edges mapping onto each bond.

    One extra accumulator bin absorbs everything ``slot`` sends to ``n_bonds``
    -- the edges that carry no split charge -- and is then discarded.  That is
    why :func:`map_edges_to_bonds` points invalid edges one past the end rather
    than at an arbitrary in-range column: no mask, no branch, no sync.
    """
    total = torch.zeros(n_bonds + 1, dtype=weighted.dtype, device=weighted.device)
    total = total.index_add(0, slot, weighted)
    count = torch.zeros(n_bonds + 1, dtype=weighted.dtype, device=weighted.device)
    count = count.index_add(0, slot, counted)
    return (total[:n_bonds] / count[:n_bonds].clamp_min(1.0))


def symmetrise_edge_scalar(
    values: Tensor,
    slot: Tensor,
    n_bonds: int,
    sign: Optional[Tensor] = None,
) -> Tensor:
    """Reduce a directed edge scalar to one symmetric value per bond.

    Computes ``0.5 * (s_ij + s_ji)``, the symmetrisation required for the bond
    hardness ``kappa_ij`` (SPEC 1.4): attention logits satisfy
    ``Q_i . K_j != Q_j . K_i``, and an asymmetric ``kappa`` makes the energy
    ill-defined because ``t_ij = -t_ji`` leaves the sign of the pair ambiguous.

    This is the **mean** over the edges that actually map onto the bond, not a
    sum scaled by a hard-coded ``0.5``.  The factor of two is only right when
    exactly two edges land on each bond, and three ordinary neighbour lists
    break that:

    * edges carrying no split charge (self-image pairs) used to land on an
      arbitrary in-range bond and add their value to it.  Measured on edges
      ``[(0,1),(1,0),(0,0),(1,1),(2,2)]`` with values ``[1,3,100,200,300]``:
      ``302.0`` where the answer is ``2.0``.  They are discarded here, via the
      sentinel slot :func:`map_edges_to_bonds` gives them;
    * several periodic images of one pair merged onto one bond contribute more
      than two values -- ``[2,2,2,2]`` over two images each way gave ``4.0``
      where the answer is ``2.0``.  This is the normal shape of a periodic
      neighbour list, not a corner case;
    * a list carrying only one direction per pair was silently halved.

    ``sign`` is accepted (and only used for its zeros) so a caller holding the
    output of :func:`map_edges_to_bonds` can pass both; it is redundant when
    ``slot`` came from that function, which already marks invalid edges.
    """
    values = values.reshape(-1)
    counted = torch.ones_like(values)
    if sign is not None:
        counted = counted * (sign.reshape(-1) != 0).to(values.dtype)
    return _reduce_edge_scalar(counted * values, slot, n_bonds, counted)


def antisymmetrise_edge_scalar(
    values: Tensor,
    slot: Tensor,
    sign: Tensor,
    n_bonds: int,
) -> Tensor:
    """Reduce a directed edge scalar to one antisymmetric value per bond.

    Computes ``0.5 * (s_ij - s_ji)`` in the stored ``lo -> hi`` orientation.
    This is the geometry-determined quantity SPEC 2 prescribes for the solver
    seed: it is a function of the current configuration alone, so the solve
    stays history-independent and the forces stay conservative.

    Like :func:`symmetrise_edge_scalar` this is the mean over the contributing
    edges rather than a fixed ``0.5``, which reproduces ``0.5 * (s_ij - s_ji)``
    when both directions are present and stays correct when they are not.
    Edges with ``sign == 0`` carry no split charge and drop out of both the sum
    and the count.
    """
    values = values.reshape(-1)
    sign = sign.reshape(-1).to(values.dtype)
    counted = (sign != 0).to(values.dtype)
    return _reduce_edge_scalar(sign * values, slot, n_bonds, counted)
