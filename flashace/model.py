from __future__ import annotations

import dataclasses
import functools
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from .physics import (
    ACE_Descriptor,
    ACEV2Descriptor,
    ACEV3TokenDescriptor,
    ACEV4CumulantTokenDescriptor,
    ACERadialBasis,
    SmoothACERadialBasis,
)


def _layer_scale_num_copies(blocks: tuple[tuple[int, int], ...]) -> int:
    """Number of irrep copies, i.e. the number of independent equivariant scales."""
    return int(sum(multiplicity for multiplicity, _ in blocks))


def _irrepwise_layer_scale(
    scale: torch.Tensor,
    blocks: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Broadcast one scalar per irrep copy to all of its magnetic components.

    Equivariance requires the residual multiplier to be constant over the
    ``2*l+1`` components ``m`` of a single irrep copy: only then does it commute
    with the Wigner rotation ``D^{(l)}``.  The parameter is therefore *stored*
    with one entry per copy, so a non-equivariant state is unrepresentable by
    construction rather than being projected away at forward time.

    ``scale`` has shape ``(sum_l mul_l,)`` and the return value has shape
    ``(sum_l mul_l * (2*l+1),)``, matching ``irreps.dim``.
    """
    pieces = []
    offset = 0
    for multiplicity, irrep_dim in blocks:
        copy_scale = scale[offset : offset + multiplicity]
        pieces.append(copy_scale.unsqueeze(-1).expand(multiplicity, irrep_dim).reshape(-1))
        offset += multiplicity
    return torch.cat(pieces)


def freeze_layer_scales(model: torch.nn.Module) -> int:
    """Bake the expanded residual layer scale as a constant, for inference only.

    ``_irrepwise_layer_scale`` broadcasts one scalar per irrep copy to all of its
    magnetic components. During inference the underlying parameter is fixed, so
    the broadcast is a compile-time constant that is otherwise recomputed on
    every step. Freezing it removes a ``(mul, 2l+1)`` expand-and-concatenate from
    the traced graph, which matters for ahead-of-time compilation, and saves the
    work at every MD step.

    Returns the number of blocks frozen. Call only on a model in ``eval()`` mode;
    training must keep the live parameter so gradients flow.
    """
    frozen = 0
    for module in model.modules():
        scale = getattr(module, "layer_scale_attn", None)
        blocks = getattr(module, "_layer_scale_blocks", None)
        if scale is None or blocks is None:
            continue
        module._frozen_layer_scale = _irrepwise_layer_scale(scale.detach(), blocks)
        frozen += 1
    return frozen


def _segment_softmax(logits: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax over incoming local edges for each receiver atom."""
    if logits.numel() == 0:
        return logits

    logits_f = logits.float()
    expanded_index = index[:, None].expand(-1, logits.shape[-1])
    max_per_node = torch.full(
        (num_nodes, logits.shape[-1]),
        float("-inf"),
        device=logits.device,
        dtype=logits_f.dtype,
    )
    max_per_node = max_per_node.scatter_reduce(
        0,
        expanded_index,
        logits_f,
        reduce="amax",
        include_self=True,
    )
    max_per_node = torch.where(
        torch.isfinite(max_per_node), max_per_node, torch.zeros_like(max_per_node)
    )
    centered = logits_f - max_per_node[index]
    exp_logits = torch.exp(centered)
    denom = torch.zeros_like(max_per_node).scatter_add(0, expanded_index, exp_logits)
    return (exp_logits / (denom[index] + 1e-9)).to(logits.dtype)


# ---------------------------------------------------------------------------
# Periodicity for the optional electrostatics -- SPEC 7
#
#     "Periodicity is an INPUT, never inferred from the cell."
#
# ASE hands a gas-phase frame ``cell = zeros(3, 3)``.  That is not ``None``, so
# a hook that forwards it verbatim tells the electrostatics "this system is
# periodic, here is its lattice", and the Ewald route then dies inside
# ``torch.linalg.inv`` ("The diagonal element 1 is zero", measured).  The data
# pipeline must therefore emit
#
#     data["pbc"] = torch.tensor(atoms.pbc, dtype=torch.bool)   # shape [3]
#
# (``[n_struct, 3]`` for the batched collate), which is the only place the
# information still exists.  ``train.py`` does (AtomisticDataset.__getitem__ and
# collate_batched, whenever electrostatics are enabled).  The two inference
# call sites -- ``flashace/calculator.py`` (TRACECalculator.calculate) and
# ``transformers_ace/deploy.py`` -- still build ``cell`` with
# ``torch.tensor(atoms.cell.array)`` and drop ``atoms.pbc``; for them
# ``_electrostatic_pbc`` below reconstructs the triple from the cell, and that
# reconstruction is the one inference in this file -- deliberate, named and
# documented, rather than the incidental "cell is not None" it replaces.
# ---------------------------------------------------------------------------

#: Batch-dictionary key holding the ASE periodic-boundary triple.
ELECTROSTATIC_PBC_KEY = "pbc"

#: SPEC 24: the batched path solves its open-boundary structures as ONE
#: block-diagonal system, in consecutive chunks of at most this many atoms.  The
#: joint operator is dense, so a chunk costs O(D^2) per CG iteration in its total
#: size D, while the per-structure loop costs a fixed launch / synchronisation
#: overhead per structure: the optimum depends on the device.  Measured per
#: structure, one training step (energy + forces + backward), batch 32, 2 CPU
#: threads / RTX 5060 Ti, float64, against the per-structure loop:
#:   CPU, cap 96:  Ag3 3.0 ms (loop 18.7), C10 chain 15.0 (26.7), NaCl 19.4 (29.2);
#:     the whole batch as one system is slower than the loop for NaCl (45 ms);
#:   GPU, cap 768: Ag3 6.5 ms (loop 77), C10 chain 9.5 (93), NaCl 14.1 (87);
#:     batch 128: Ag3 1.9 ms; the C10 chain as one 1664-atom system 36 ms.
#: ``None`` selects the device default below.
BATCHED_ELECTROSTATICS_MAX_ATOMS = None
BATCHED_ELECTROSTATICS_MAX_ATOMS_BY_DEVICE = {"cpu": 96, "cuda": 768}


def _block_diagonal_max_atoms(device):
    """Atoms per block-diagonal solve on ``device`` (SPEC 24)."""
    if BATCHED_ELECTROSTATICS_MAX_ATOMS is not None:
        return int(BATCHED_ELECTROSTATICS_MAX_ATOMS)
    return BATCHED_ELECTROSTATICS_MAX_ATOMS_BY_DEVICE.get(torch.device(device).type, 96)

#: Per-structure keys the block-diagonal route does not carry; a structure with
#: any of them sends its batch through the per-structure loop.
_BLOCK_DIAGONAL_UNSUPPORTED_KEYS = (
    "external_field", "metallic", "electrode_potential", "charge_states", "fragment_spins",
)


def _electrostatic_pbc(cell, data, structure=None):
    """Periodicity of one configuration as a ``(bool, bool, bool)`` tuple.

    Read from ``data['pbc']`` when the data pipeline supplies it -- that is the
    SPEC 7 contract, and it is the only source that can distinguish a molecule
    sitting in a 20 A box (``(F, F, F)``, a box that must be IGNORED) from the
    same box treated as a periodic cell.

    When the key is absent the triple is reconstructed from the lattice with
    ASE's own convention: direction ``i`` is periodic iff lattice vector
    ``cell[i]`` is non-zero.  A gas-phase ``zeros(3, 3)`` therefore gives
    ``(F, F, F)`` instead of a singular lattice handed to Ewald.  The
    reconstruction is a fallback for the call sites listed above, not the
    contract: it cannot see a non-periodic molecule that was given a box, and
    SPEC 7 is explicit that periodicity is an input.

    Args:
        cell: ``[3, 3]`` lattice (rows are lattice vectors), ``[n_struct, 3, 3]``
            in the batched path, or ``None``.
        data: the batch dictionary, or ``None``.
        structure: index into the leading structure dimension of a batched
            ``pbc``/``cell``; ``None`` for the single-structure path.
    """
    supplied = None if data is None else data.get(ELECTROSTATIC_PBC_KEY, None)
    if supplied is not None:
        pbc = supplied if torch.is_tensor(supplied) else torch.as_tensor(supplied)
        pbc = pbc.detach()
        if pbc.dim() == 2:
            if structure is None:
                raise ValueError(
                    f"data['{ELECTROSTATIC_PBC_KEY}'] has shape "
                    f"{tuple(pbc.shape)}, i.e. one triple per structure, but "
                    "this is the single-structure path. Pass a [3] triple, or "
                    "evaluate the batch through forward_batched()."
                )
            pbc = pbc[structure]
        pbc = pbc.reshape(-1)
        if pbc.numel() == 1:  # ASE accepts a scalar pbc; expand it as ASE does.
            pbc = pbc.expand(3)
        if pbc.numel() != 3:
            raise ValueError(
                f"data['{ELECTROSTATIC_PBC_KEY}'] must hold three booleans "
                f"(got {pbc.numel()} entries)."
            )
        return tuple(bool(flag) for flag in pbc)

    if cell is None:
        return (False, False, False)
    lattice = cell[structure] if structure is not None else cell
    # ASE's own rule: a zero lattice vector means that direction is not
    # periodic.  ``zeros(3, 3)`` -> (F, F, F), which is the whole point.
    return tuple(bool(flag) for flag in (lattice.detach() != 0).any(dim=1))


def _electrostatics_accepts_pbc(electrostatics) -> bool:
    """Whether the attached electrostatics takes a ``pbc`` keyword.

    ``trace_electrostatics.energy.ElectrostaticModule.forward`` is growing a
    ``pbc`` parameter (SPEC 7).  This check exists so that the hooks pass the
    triple the moment it is accepted and stay callable before then, and so that
    a duck-typed test double without the keyword still works.  When every
    supported version of the package takes ``pbc``, delete this and pass the
    keyword unconditionally -- the fallback below is strictly less expressive
    than the contract, and a capability probe that can never fail is noise.
    """
    forward = getattr(type(electrostatics), "forward", None)
    if forward is None:
        return False
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):  # C-implemented or otherwise unintrospectable
        return False
    return "pbc" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _electrostatic_energy_term(
    electrostatics,
    h,
    pos,
    edge_index,
    edge_vec,
    edge_len,
    cell,
    edge_shift,
    data,
    structure=None,
    pbc=None,
    return_output=False,
):
    """The electrostatic energy of one configuration, with SPEC 7 periodicity.

    The single place in this file that calls an attached electrostatics, so the
    boundary contract is stated once for all four hooks:

    * ``h`` is the FULL equivariant feature block, never ``h[:, :hidden_dim]``.
      The heads need the 1o and 2e parts (``A_mu`` is 0e + 2e, SPEC 3); handing
      them the scalar slice constructs an ``o3.Linear`` whose anisotropic
      instructions are unreachable, and SPEC 3 records that such a head returns
      identically zero and emits no warning.
    * ``pos``, ``cell`` and ``edge_vec`` are the caller's already-strained
      tensors, and the call sits upstream of ``torch.autograd.grad``, so forces
      and stress follow from the machinery that is already there.
    * the cell reaches the electrostatics only when some direction is actually
      periodic.  SPEC 7: under ``(F, F, F)`` "a supplied cell is IGNORED", so
      dropping it here is the contract, not a workaround -- and it is what
      keeps a gas-phase ``zeros(3, 3)`` out of ``EwaldParams.auto_tune``.

    Args:
        structure: index into a batched ``cell``/``pbc``; ``None`` outside the
            batched path, where ``cell`` is already ``[3, 3]``.
        pbc: the periodicity triple, when the caller has already resolved it.
            The batched path does, because it hands the electrostatics a
            per-structure charge specification sliced out of the batch
            (:func:`_structure_charge_specs`) rather than the batch dictionary
            itself, while ``data['pbc']`` is read from the batch, per structure.
        return_output: also return an :class:`ElectrostaticOutput` for this
            configuration -- the solve and the positions it was built from --
            for loss terms on dipoles, polarizabilities and Born charges.

    Returns:
        ``U_elec``, or ``(U_elec, ElectrostaticOutput)`` with ``return_output``.
    """
    if pbc is None:
        pbc = _electrostatic_pbc(cell, data, structure)
    periodic = any(pbc)
    lattice = None
    if periodic:
        lattice = cell[structure] if structure is not None else cell

    keywords = {"cell": lattice, "edge_shift": edge_shift, "data": data}
    if _electrostatics_accepts_pbc(electrostatics):
        keywords["pbc"] = pbc
    elif periodic and not all(pbc):
        # Mixed periodicity is different physics, not a flag (SPEC 7): a slab
        # needs the Yeh-Berkowitz correction or a genuine 2-D Ewald, and 3-D
        # Ewald on a slab is simply wrong.  Without a `pbc` keyword there is no
        # way to say so, and silently running 3-D Ewald -- or silently running
        # an open-boundary sum -- would be a wrong answer rather than an error.
        raise NotImplementedError(
            f"pbc={pbc} is a partially periodic system (SPEC 7: slab or wire), "
            "but the attached electrostatics takes no `pbc` keyword, so the "
            "boundary condition cannot be communicated to it. 3-D Ewald on a "
            "slab is wrong (the images along the non-periodic axis interact "
            "through their dipoles, a spurious 2 pi M_z^2 / V), and quietly "
            "falling back to an open-boundary sum would be just as wrong. Use "
            "a trace_electrostatics whose ElectrostaticModule.forward accepts "
            "pbc."
        )

    if not return_output:
        return electrostatics(h, pos, edge_index, edge_vec, edge_len, **keywords)[0]

    result = electrostatics(h, pos, edge_index, edge_vec, edge_len, **keywords)
    energy, solve_result = result[0], result[1]
    total_charge = 0.0 if data is None else data.get("total_charge", 0.0)
    # TRACE-Redox: the state rebuilt for observables is the bound sector's (bound_state), whose energy
    # is energy_bound, not the total (trace_redox SPEC 17.7)
    reference_energy = solve_result.energy_bound if _bound_sector_is_the_response(electrostatics) else energy
    output = ElectrostaticOutput(
        energy=energy,
        solve_result=solve_result,
        positions=pos,
        pbc=tuple(bool(flag) for flag in pbc),
        total_charge=float(total_charge),
        build_state=functools.partial(
            _rebuild_electrostatic_state,
            electrostatics,
            reference_energy,
            h,
            pos,
            edge_index,
            edge_vec,
            edge_len,
            lattice,
            edge_shift,
            tuple(bool(flag) for flag in pbc),
            data,
        ),
    )
    return energy, output


def _bound_sector_is_the_response(electrostatics) -> bool:
    """TRACE-Redox with integer free charges (anchor='formal'): observables come from bound_state."""
    redox = getattr(electrostatics, "redox", None)
    return hasattr(electrostatics, "bound_state") and getattr(redox, "anchor", None) == "formal"


class ElectrostaticOutput:
    """One configuration's electrostatics, as evaluated inside a forward pass.

    A forward called with ``return_electrostatics=True`` puts one of these per
    structure in ``aux['electrostatics']``.  It is what the dipole,
    polarizability and Born-charge loss terms of ``train.py`` are built from,
    and it carries nothing that the energy did not already compute, except on
    request (:meth:`state`).

    Attributes:
        energy: ``U_elec`` in eV, exactly the term that was added to ``E``
            (dispersion included when ``config.dispersion`` is on).
        solve_result: the ``SolveResult`` of that same evaluation -- charges
            ``q`` in e and dipoles ``mu`` in e*A on this structure's atoms, in
            the order of :attr:`positions`.
        positions: ``[n_atoms, 3]`` positions ``U_elec`` was built from.  Every
            quantity of this structure -- its node features, its bond geometry
            and the electrostatic assembly -- descends from this tensor, so a
            derivative taken with respect to it (a Born charge) is the complete
            derivative of the model, not only its explicit pair-kernel part.  In
            the batched path that is arranged by :func:`_structure_positions`.
        pbc: the periodicity triple the evaluation used (SPEC 7).
        total_charge: net charge in e.
    """

    __slots__ = (
        "energy", "solve_result", "positions", "pbc", "total_charge",
        "_build_state", "_state",
    )

    def __init__(self, energy, solve_result, positions, pbc, total_charge, build_state):
        self.energy = energy
        self.solve_result = solve_result
        self.positions = positions
        self.pbc = pbc
        self.total_charge = total_charge
        self._build_state = build_state
        self._state = None

    @property
    def periodic(self) -> bool:
        return any(self.pbc)

    def state(self):
        """The ``trace_electrostatics.ElectrostaticState`` of this evaluation.

        ``polarizability`` and the Born charges need the assembled operator,
        which ``ElectrostaticModule.forward`` does not return, so it is rebuilt
        on first request -- by :func:`trace_electrostatics.electrostatic_state`
        on exactly the tensors the energy was built from -- and memoised.  The
        rebuild is checked against :attr:`energy`, so an observable can never
        silently belong to a different functional from the one that was
        trained.  Costs one extra solve, paid only for structures whose
        observables are wanted.
        """
        if self._state is None:
            self._state = self._build_state()
        return self._state


def _rebuild_electrostatic_state(
    electrostatics,
    energy,
    h,
    pos,
    edge_index,
    edge_vec,
    edge_len,
    lattice,
    edge_shift,
    pbc,
    data,
):
    """``electrostatic_state`` on the arguments one hook handed the electrostatics.

    Reads the charge specification from ``data`` under the keys, and with the
    defaults, that ``ElectrostaticModule.forward`` reads, so the rebuilt state
    is the one the module evaluated.  That is then verified rather than assumed:
    the energy of the rebuilt state must reproduce the energy the module
    returned (for TRACE-Redox, the bound sector's ``energy_bound``).
    """
    from trace_electrostatics import electrostatic_state

    heads = getattr(electrostatics, "heads", None)
    config = getattr(electrostatics, "config", None)
    if heads is None or config is None:
        raise TypeError(
            "the attached electrostatics exposes no `heads`/`config`, so its "
            "state (the assembled operator that polarizabilities and Born "
            "charges are taken from) cannot be rebuilt. Attach it with "
            "trace_electrostatics.attach_electrostatics."
        )
    if _bound_sector_is_the_response(electrostatics):
        # TRACE-Redox, anchor='formal': the bound sector alone is the model's electronic response (the
        # free charges are integers); checked below against energy_bound.  Other redox models keep
        # their refusal (RedoxElectrostatics.state raises).
        state = electrostatics.bound_state(
            h, pos, edge_index, edge_vec, edge_len,
            pbc=pbc, cell=lattice, edge_shift=edge_shift, data=data,
        )
    elif hasattr(electrostatics, "state"):
        # The module's own input resolution (charge spec, spins, quadrupole
        # default, field, metals, electrodes): the same functional as its energy.
        state = electrostatics.state(
            h, pos, edge_index, edge_vec, edge_len,
            pbc=pbc, cell=lattice, edge_shift=edge_shift, data=data,
        )
    else:
        spec = {} if data is None else data
        state = electrostatic_state(
            h,
            pos,
            edge_index,
            edge_vec,
            edge_len,
            heads,
            config,
            cell=lattice,
            edge_shift=edge_shift,
            fragment_id=spec.get("fragment_id"),
            fragment_charges=spec.get("fragment_charges"),
            total_charge=spec.get("total_charge", 0.0),
            pbc=pbc,
            fragment_spins=spec.get("fragment_spins"),
            quadrupoles=bool(spec.get("quadrupoles", False)),
        )
    rebuilt = state.energy if state.dispersion is None else state.energy + state.dispersion
    if not torch.allclose(
        rebuilt.detach(), energy.detach().to(rebuilt.dtype), rtol=1e-8, atol=1e-8
    ):
        raise RuntimeError(
            "the rebuilt electrostatic state does not reproduce the energy the "
            f"attached module returned ({float(rebuilt):.12g} vs "
            f"{float(energy):.12g} eV). An observable taken from it would belong "
            "to a different functional from the one being trained; "
            "ElectrostaticModule.forward and electrostatic_state have diverged."
        )
    return state


def _structure_positions(pos, batch, n_struct):
    """Per-structure position tensors that everything downstream descends from.

    In a batched graph the node features of structure ``s`` are computed from
    the *whole* batched position tensor, so ``pos[atoms_of_s]`` -- a gather that
    nothing else depends on -- is the wrong variable for a derivative: taken
    with respect to it, a Born charge would see only the explicit pair-kernel
    dependence of ``U_elec`` and miss everything that enters through the node
    features and the bond geometry.  Here the batched tensor is instead
    REBUILT from per-structure pieces, so each piece is an ancestor of every
    quantity of its structure.  The values are unchanged (a permutation and its
    inverse).  Forces must then be taken w.r.t. the tensor that was SPLIT, not
    the rebuilt one: a structure's electrostatics also depends on its piece
    directly (the pair kernels), and only the split tensor is upstream of both
    paths -- differentiating the rebuilt one silently drops that term.

    Returns ``(pos, pieces)`` with ``pieces[s]`` the ``[n_s, 3]`` positions of
    structure ``s`` in the order ``(batch == s).nonzero()`` lists its atoms.
    """
    atom_index = _structure_atom_lists(batch, n_struct)
    pieces = [pos.index_select(0, index) for index in atom_index]
    order = torch.cat(atom_index)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return torch.cat(pieces).index_select(0, inverse), pieces


def _per_structure_counts(data, key, n_struct):
    """``data[key]`` as ``n_struct`` Python ints, and their running offsets."""
    counts = data.get(key, None)
    if counts is None:
        return None, None
    counts = [int(c) for c in torch.as_tensor(counts).reshape(-1).tolist()]
    if len(counts) != n_struct:
        raise ValueError(
            f"data['{key}'] has {len(counts)} entries for a batch of {n_struct} "
            "structures; it must hold one count per structure."
        )
    offsets = [0]
    for count in counts[:-1]:
        offsets.append(offsets[-1] + count)
    return counts, offsets


def _structure_atom_lists(batch, n_struct):
    """Atom indices of every structure of a batched graph, with two host reads.

    ``collate_batched`` concatenates structures, so ``batch`` is non-decreasing
    and structure ``s`` owns one contiguous range: one ordering check and one
    ``bincount`` replace ``n_struct`` calls of ``nonzero`` (each a host
    synchronisation, which stalls a GPU).  Any other layout falls back to
    ``nonzero`` per structure, so the result never depends on the layout.
    """
    if batch.numel() == 0 or bool((batch[1:] >= batch[:-1]).all()):
        counts = torch.bincount(batch, minlength=n_struct).tolist()
        lists, start = [], 0
        for count in counts[:n_struct]:
            lists.append(torch.arange(start, start + count, device=batch.device))
            start += count
        return lists
    return [(batch == structure).nonzero(as_tuple=False).view(-1) for structure in range(n_struct)]


def _structure_charge_specs(data, batch, n_struct, atom_lists=None):
    """Slice the per-structure charge specification out of a batch dictionary.

    The layout is the one ``train.AtomisticDataset.collate_batched`` emits:

    * ``total_charge``: ``[n_struct]`` net charges in e.  A scalar is accepted
      only if it is zero, because one number cannot say which structure of a
      batch carries a non-zero charge.
    * ``fragment_id``: ``[n_atoms_total]`` fragment labels LOCAL to each
      structure (``0 .. F_s - 1``; ``-1`` for a structure that supplies none),
      with ``n_fragments`` ``[n_struct]`` (0 = none supplied, fragments then
      derived from the bond graph as in the single-structure path) and
      ``fragment_charges`` ``[sum_s F_s]`` concatenated in structure order.
    * ``fragment_spins``: ``[sum_s F_s]`` concatenated, with
      ``n_fragment_spins`` ``[n_struct]``.

    Returns one dict per structure holding exactly the keys
    ``ElectrostaticModule.forward`` reads (``total_charge`` as a Python float,
    as the package requires), or ``None`` for a structure with nothing to say --
    which is what the single-structure path hands it for a neutral system.
    """
    specs = [dict() for _ in range(n_struct)]
    if data is None:
        return [None] * n_struct
    if atom_lists is None:
        atom_lists = _structure_atom_lists(batch, n_struct)

    total_charge = data.get("total_charge", None)
    if total_charge is not None:
        if torch.is_tensor(total_charge) and total_charge.dim() == 1:
            charges = [float(q) for q in total_charge.detach().cpu().tolist()]
        elif torch.is_tensor(total_charge) or isinstance(total_charge, (int, float)):
            if float(total_charge) != 0.0:
                raise ValueError(
                    "forward_batched was given ONE non-zero total_charge "
                    f"({float(total_charge)}) for a batch of {n_struct} "
                    "structures, which does not say which structure carries "
                    "it (SPEC 1.3 forbids guessing). Pass total_charge as a "
                    "[n_structures] tensor, as collate_batched does."
                )
            charges = [0.0] * n_struct
        else:
            charges = [float(q) for q in total_charge]
        if len(charges) != n_struct:
            raise ValueError(
                f"data['total_charge'] has {len(charges)} entries for a batch "
                f"of {n_struct} structures."
            )
        for spec, charge in zip(specs, charges):
            if charge != 0.0:
                spec["total_charge"] = charge

    fragment_id = data.get("fragment_id", None)
    fragment_charges = data.get("fragment_charges", None)
    fragment_spins = data.get("fragment_spins", None)
    counts, offsets = _per_structure_counts(data, "n_fragments", n_struct)
    spin_counts, spin_offsets = _per_structure_counts(data, "n_fragment_spins", n_struct)
    if (fragment_id is None) != (fragment_charges is None) or (
        fragment_id is not None and counts is None
    ):
        raise ValueError(
            "a batched fragment specification needs fragment_id "
            "([n_atoms_total], labels local to each structure), "
            "fragment_charges ([sum of fragments], in structure order) AND "
            "n_fragments ([n_structures]) together; without all three there "
            "is no way to tell which fragment and which charge belong to which "
            "structure (SPEC 1.3). Build the batch with "
            "AtomisticDataset.collate_batched."
        )
    if fragment_spins is not None and spin_counts is None:
        raise ValueError(
            "a batched fragment_spins needs n_fragment_spins ([n_structures]) "
            "to say which spin targets belong to which structure."
        )
    if fragment_id is not None and int(fragment_charges.shape[0]) != sum(counts):
        raise ValueError(
            f"fragment_charges has {int(fragment_charges.shape[0])} entries but "
            f"n_fragments sums to {sum(counts)}."
        )
    if fragment_spins is not None and int(fragment_spins.shape[0]) != sum(spin_counts):
        raise ValueError(
            f"fragment_spins has {int(fragment_spins.shape[0])} entries but "
            f"n_fragment_spins sums to {sum(spin_counts)}."
        )

    for structure, spec in enumerate(specs):
        if counts is not None and counts[structure] > 0:
            atom_index = atom_lists[structure]
            start = offsets[structure]
            spec["fragment_id"] = fragment_id[atom_index]
            spec["fragment_charges"] = fragment_charges[start : start + counts[structure]]
        if spin_counts is not None and spin_counts[structure] > 0:
            start = spin_offsets[structure]
            spec["fragment_spins"] = fragment_spins[start : start + spin_counts[structure]]
        if data.get("quadrupoles", None) is not None:
            spec["quadrupoles"] = data["quadrupoles"]

    # The environment (collate_batched layout): an applied field per structure,
    # metallic flags and electrode potentials per atom, diabatic charge states
    # per structure.  Each structure is handed only what it carries.
    field = data.get("external_field", None)
    has_field = data.get("has_external_field", None)
    metallic = data.get("metallic", None)
    potential = data.get("electrode_potential", None)
    states = data.get("charge_states", None)
    couplings = data.get("charge_state_coupling", None)
    species = data.get("z", None)
    for structure, spec in enumerate(specs):
        if species is not None:
            # Atomic numbers, for heads that read them (SPEC 23); the single-
            # structure path hands the electrostatics data['z'] the same way.
            spec["z"] = species[atom_lists[structure]]
        if field is not None and (has_field is None or bool(has_field[structure])):
            spec["external_field"] = field[structure]
        if metallic is not None or potential is not None:
            atom_index = atom_lists[structure]
            if metallic is not None and bool(metallic[atom_index].any()):
                spec["metallic"] = metallic[atom_index]
            if potential is not None and bool(torch.isfinite(potential[atom_index]).any()):
                spec["electrode_potential"] = potential[atom_index]
        if states is not None and states[structure] is not None:
            spec["charge_states"] = states[structure]
            spec["charge_state_coupling"] = 0.0 if couplings is None else couplings[structure]
    return [spec if spec else None for spec in specs]


class ScalarPreNorm(nn.Module):
    """Normalize invariant scalar channels without mixing equivariant components."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scalars = self.norm(x[..., : self.hidden_dim])
        rest = x[..., self.hidden_dim :]
        return torch.cat((scalars, rest), dim=-1)


class LocalEquivariantAttentionBlock(nn.Module):
    """Local attention on ACE irreps with invariant weights and equivariant values.

    This is intentionally not a message-passing stack: the only communication is
    a single local transformer-style attention operation over cutoff neighbors.
    The attention logits are scalar invariants, while values are transformed with
    e3nn linear maps, so the residual update preserves E(3) equivariance.
    """

    def __init__(
        self,
        irreps,
        hidden_dim: int,
        r_max: float,
        num_radial: int,
        num_heads: int = 4,
        key_dim: int | None = None,
        ffn_hidden: int | None = None,
        dropout: float = 0.0,
        layer_scale_init: float | None = 1e-2,
        radial_basis_type: str = "bessel",
        radial_trainable: bool = False,
        envelope_exponent: int = 5,
        gaussian_width: float = 0.5,
        use_distance_penalty: bool = True,
    ):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.hidden_dim = hidden_dim
        self.num_heads = max(1, int(num_heads))
        self.key_dim = key_dim or max(16, hidden_dim // self.num_heads)
        self.use_distance_penalty = bool(use_distance_penalty)

        self.norm1 = ScalarPreNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, self.num_heads * self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.num_heads * self.key_dim, bias=False)
        self.radial_basis = ACERadialBasis(
            r_max,
            num_radial,
            envelope_exponent=envelope_exponent,
            basis_type=radial_basis_type,
            trainable=radial_trainable,
            gaussian_width=gaussian_width,
        )
        self.radial_bias = nn.Sequential(
            nn.Linear(num_radial, max(16, self.num_heads * 4)),
            nn.SiLU(),
            nn.Linear(max(16, self.num_heads * 4), self.num_heads),
        )
        self.distance_log_scale = (
            nn.Parameter(torch.zeros(self.num_heads))
            if self.use_distance_penalty
            else None
        )

        self.value_proj = nn.ModuleList(
            [o3.Linear(self.irreps, self.irreps) for _ in range(self.num_heads)]
        )
        self.out_proj = o3.Linear(self.irreps, self.irreps)
        self.dropout = nn.Dropout(dropout)
        self._layer_scale_blocks = tuple(
            (int(multiplicity), int(irrep.dim)) for multiplicity, irrep in self.irreps
        )
        self.layer_scale_attn = (
            nn.Parameter(
                torch.full(
                    (_layer_scale_num_copies(self._layer_scale_blocks),),
                    float(layer_scale_init),
                )
            )
            if layer_scale_init is not None
            else None
        )

        ffn_hidden = ffn_hidden or hidden_dim * 4
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.scalar_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        self.layer_scale_ffn = (
            nn.Parameter(torch.full((hidden_dim,), float(layer_scale_init)))
            if layer_scale_init is not None
            else None
        )

    def _attention_layer_scale(self) -> torch.Tensor:
        """Expanded residual scale, using the frozen constant when available."""
        frozen = getattr(self, "_frozen_layer_scale", None)
        if frozen is not None:
            return frozen
        return _irrepwise_layer_scale(self.layer_scale_attn, self._layer_scale_blocks)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_len: torch.Tensor,
        temperature_scale: float = 1.0,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        if sender.numel() == 0:
            return x

        x_norm = self.norm1(x)
        scalars = x_norm[..., : self.hidden_dim]
        q = self.q_proj(scalars).view(-1, self.num_heads, self.key_dim)
        k = self.k_proj(scalars).view(-1, self.num_heads, self.key_dim)

        radial = self.radial_basis(edge_len)
        logits = (q[receiver] * k[sender]).sum(dim=-1) / math.sqrt(self.key_dim)
        logits = logits + self.radial_bias(radial)
        if self.distance_log_scale is not None:
            logits = logits - F.softplus(self.distance_log_scale)[None, :] * edge_len[:, None]
        logits = logits / max(float(temperature_scale), 1e-4)

        alpha = self.dropout(_segment_softmax(logits, receiver, x.shape[0]))
        out = torch.zeros_like(x)
        for head, value_layer in enumerate(self.value_proj):
            values = value_layer(x_norm)[sender]
            out.index_add_(0, receiver, alpha[:, head : head + 1].to(values.dtype) * values)
        out = out / self.num_heads
        out = self.out_proj(out)
        if self.layer_scale_attn is not None:
            out = out * self._attention_layer_scale()
        x = x + out

        scalars = x[..., : self.hidden_dim]
        rest = x[..., self.hidden_dim :]
        scalar_update = self.scalar_ffn(self.norm2(scalars))
        if self.layer_scale_ffn is not None:
            scalar_update = scalar_update * self.layer_scale_ffn
        return torch.cat((scalars + scalar_update, rest), dim=-1)

class LegacyTransformersACE(nn.Module):
    architecture_version = 1

    def __init__(
        self,
        r_max=5.0,
        l_max=2,
        num_radial=8,
        hidden_dim=128,
        num_layers=2,
        radial_basis_type: str = "bessel",
        radial_trainable: bool = False,
        envelope_exponent: int = 5,
        gaussian_width: float = 0.5,
        descriptor_passes: int = 1,
        descriptor_residual: bool = True,
        radial_mlp_hidden: int = 64,
        radial_mlp_layers: int = 2,
        correlation_order: int = 4,
        correlation_channels: int = 16,
        attention_num_heads: int | None = None,
        attention_key_dim: int | None = None,
        attention_ffn_hidden: int | None = None,
        attention_dropout: float = 0.0,
        attention_layer_scale_init: float | None = 1e-2,
        attention_distance_penalty: bool = True,
        transformer_num_heads: int = 4,
        transformer_ffn_hidden: int | None = None,
        transformer_dropout: float = 0.0,
        use_aux_force_head: bool = True,
        use_aux_stress_head: bool = True,
        interleave_descriptor: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.r_max = r_max
        self.l_max = l_max
        self.descriptor_passes = max(1, int(descriptor_passes))
        self.descriptor_residual = bool(descriptor_residual)
        self.attention_num_heads = max(1, int(attention_num_heads or transformer_num_heads))
        self.attention_key_dim = attention_key_dim
        self.attention_ffn_hidden = attention_ffn_hidden or transformer_ffn_hidden
        self.attention_dropout = float(attention_dropout if attention_dropout is not None else transformer_dropout)
        self.attention_layer_scale_init = attention_layer_scale_init
        self.attention_distance_penalty = bool(attention_distance_penalty)
        # Direct force/stress heads are intentionally disabled: MD forces and
        # stresses must come from derivatives of a single scalar energy.
        self.use_aux_force_head = False
        self.use_aux_stress_head = False
        self.interleave_descriptor = bool(interleave_descriptor)
        self.node_scalar_irreps = o3.Irreps(f"{hidden_dim}x0e")

        self.emb = nn.Embedding(118, hidden_dim)
        self.ace = ACE_Descriptor(
            r_max,
            l_max,
            num_radial,
            hidden_dim,
            radial_basis_type=radial_basis_type,
            radial_trainable=radial_trainable,
            envelope_exponent=envelope_exponent,
            gaussian_width=gaussian_width,
            radial_mlp_hidden=radial_mlp_hidden,
            radial_mlp_layers=radial_mlp_layers,
        )
        self.attention_irreps = self.ace.irreps_out

        self.layers = nn.ModuleList([
            LocalEquivariantAttentionBlock(
                self.attention_irreps,
                hidden_dim,
                r_max,
                num_radial,
                num_heads=self.attention_num_heads,
                key_dim=self.attention_key_dim,
                ffn_hidden=self.attention_ffn_hidden,
                dropout=self.attention_dropout,
                layer_scale_init=self.attention_layer_scale_init,
                radial_basis_type=radial_basis_type,
                radial_trainable=radial_trainable,
                envelope_exponent=envelope_exponent,
                gaussian_width=gaussian_width,
                use_distance_penalty=self.attention_distance_penalty,
            )
            for i in range(num_layers)
        ])
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 64), 
            nn.SiLU(), 
            nn.Linear(64, 1)
        )
        self.aux_force_head = None
        self.aux_stress_head = None

    def forward(
        self,
        data,
        training=False,
        temperature_scale: float = 1.0,
        detach_pos: bool = True,
        compute_stress: bool | None = None,
    ):
        z, pos, edge_index = data['z'], data['pos'], data['edge_index']
        cell_volume = data.get('volume', None)
        cell = data.get('cell', None)
        edge_shift = data.get('edge_shift', None)

        if compute_stress is None:
            compute_stress = training

        # We always need gradients w.r.t. atomic positions to compute forces.
        # Detach to ensure we work with a leaf tensor before enabling grads.
        if detach_pos:
            pos = pos.detach()
        pos.requires_grad_(True)

        if cell is not None:
            cell = cell.to(device=pos.device, dtype=pos.dtype)

        if compute_stress and cell_volume is not None:
            # Parameterize the small-strain tensor symmetrically so the stress
            # we backpropagate through corresponds to the symmetric Cauchy
            # stress and does not pick up spurious rotational components. This
            # matches ACE-style stress evaluation by differentiating with
            # respect to symmetric lattice strains.
            strain_params = torch.zeros(6, device=pos.device, dtype=pos.dtype, requires_grad=True)

            epsilon = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
            epsilon[0, 0] = strain_params[0]
            epsilon[1, 1] = strain_params[1]
            epsilon[2, 2] = strain_params[2]
            epsilon[0, 1] = epsilon[1, 0] = strain_params[3]
            epsilon[0, 2] = epsilon[2, 0] = strain_params[4]
            epsilon[1, 2] = epsilon[2, 1] = strain_params[5]

            deformation = torch.eye(3, device=pos.device, dtype=pos.dtype) + epsilon
            pos = pos @ deformation
            if cell is not None:
                cell = cell @ deformation
        else:
            strain_params = None
            epsilon = None

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        if edge_shift is not None and cell is not None:
            edge_shift = edge_shift.to(device=pos.device, dtype=pos.dtype)
            edge_vec = edge_vec + edge_shift @ cell
        edge_len = torch.norm(edge_vec, dim=1)

        # 1. Descriptor iterations (optionally residual) before local attention.
        h = self.emb(z)
        for i in range(self.descriptor_passes):
            scalars = h[..., : self.hidden_dim]
            desc = self.ace(scalars, edge_index, edge_vec, edge_len)
            if i == 0 or not self.descriptor_residual:
                h = desc
            else:
                h = h + desc

        for idx, layer in enumerate(self.layers):
            if self.interleave_descriptor:
                scalars = h[..., : self.hidden_dim]
                desc = self.ace(scalars, edge_index, edge_vec, edge_len)
                h = h + desc if self.descriptor_residual else desc
            h = layer(h, edge_index, edge_len, temperature_scale=temperature_scale)
            
        # 2. Readout
        # Note: We extract only the scalar (L=0) features for energy
        # The optimized physics.py puts scalars first, so this slice is correct.
        scalars = h[:, :self.hidden_dim] 
        E = torch.sum(self.readout(scalars))

        # Optional variational electrostatics (trace_electrostatics); see the
        # note in TransformersACE.forward.  h, not scalars: the heads need the
        # full equivariant features.  Periodicity goes through
        # _electrostatic_energy_term, which is where SPEC 7 is enforced for all
        # four hooks.
        electrostatics = getattr(self, "electrostatics", None)
        if electrostatics is not None:
            E = E + _electrostatic_energy_term(
                electrostatics,
                h,
                pos,
                edge_index,
                edge_vec,
                edge_len,
                cell,
                edge_shift,
                data,
            )

        aux = {}
        if self.use_aux_force_head and self.aux_force_head is not None:
            aux['force'] = self.aux_force_head(scalars)
        if self.use_aux_stress_head and self.aux_stress_head is not None:
            pooled = scalars.mean(dim=0, keepdim=True)
            stress_voigt = self.aux_stress_head(pooled).view(-1)
            aux['stress'] = stress_voigt

        # 3. Derivatives
        # Avoid building second-order graphs during evaluation to reduce memory.
        grads = torch.autograd.grad(
            E,
            pos,
            create_graph=training,  # only keep graph for higher-order grads when training
            retain_graph=training or epsilon is not None,
            allow_unused=True,
        )[0]
        F = -grads if grads is not None else torch.zeros_like(pos)
        
        S = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
        if compute_stress and epsilon is not None:
            # Retain the graph so the outer loss.backward() can still traverse
            # the computation graph built when taking the strain derivative.
            g_eps = torch.autograd.grad(
                E,
                strain_params,
                create_graph=training,
                retain_graph=training,
                allow_unused=True,
            )[0]
            if g_eps is not None:
                # Map the 6 unique components back to a symmetric stress tensor
                # and normalize by the deformed volume to avoid overestimating
                # stress under volumetric strain.
                stress = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
                stress[0, 0] = g_eps[0]
                stress[1, 1] = g_eps[1]
                stress[2, 2] = g_eps[2]
                # Each shear parameter changes two symmetric strain entries, so
                # its derivative is twice the corresponding tensor component.
                stress[0, 1] = stress[1, 0] = 0.5 * g_eps[3]
                stress[0, 2] = stress[2, 0] = 0.5 * g_eps[4]
                stress[1, 2] = stress[2, 1] = 0.5 * g_eps[5]

                if cell is not None:
                    volume = torch.det(cell).abs().clamp_min(1e-12)
                else:
                    volume = cell_volume * torch.det(deformation)
                # ASE uses sigma_ab = (1 / V) dE / d(epsilon_ab). Its cell
                # filters apply the minus sign when constructing cell forces.
                S = stress / volume

        return E, F, S, aux


def _segment_cutoff_softmax(
    logits: torch.Tensor,
    index: torch.Tensor,
    cutoff: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Cutoff-weighted attention with a unit self/null channel per center.

    The null channel prevents normalization from cancelling the cutoff when all
    neighbors approach ``r_max``. Consequently the entire local update and its
    first two radial derivatives vanish with the C2 envelope.
    """
    if logits.numel() == 0:
        return logits

    logits_f = logits.float()
    expanded_index = index[:, None].expand(-1, logits.shape[-1])
    max_per_node = torch.zeros(
        (num_nodes, logits.shape[-1]),
        device=logits.device,
        dtype=logits_f.dtype,
    )
    max_per_node = max_per_node.scatter_reduce(
        0,
        expanded_index,
        logits_f,
        reduce="amax",
        include_self=True,
    )
    numerator = cutoff.float()[:, None] * torch.exp(logits_f - max_per_node[index])
    null_weight = torch.exp(-max_per_node)
    denominator = null_weight.scatter_add(0, expanded_index, numerator)
    return (numerator / denominator[index].clamp_min(1e-12)).to(logits.dtype)


def _fixed_token_cutoff_softmax(
    logits: torch.Tensor,
    index: torch.Tensor,
    cutoff: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Cutoff attention for v3's nonempty edge-plus-moment token set."""

    logits_f = logits.float()
    expanded_index = index[:, None].expand(-1, logits.shape[-1])
    max_per_node = torch.zeros(
        (num_nodes, logits.shape[-1]),
        device=logits.device,
        dtype=logits_f.dtype,
    )
    max_per_node = max_per_node.scatter_reduce(
        0,
        expanded_index,
        logits_f,
        reduce="amax",
        include_self=True,
    )
    numerator = cutoff.float()[:, None] * torch.exp(logits_f - max_per_node[index])
    null_weight = torch.exp(-max_per_node)
    denominator = null_weight.scatter_add(0, expanded_index, numerator)
    return (numerator / denominator[index].clamp_min(1e-12)).to(logits.dtype)


class StrictLocalEquivariantAttentionBlock(nn.Module):
    """Neighbor-set attention without exchanging updated atomic hidden states."""

    def __init__(
        self,
        node_irreps,
        edge_irreps,
        hidden_dim: int,
        edge_scalar_dim: int,
        r_max: float,
        num_radial: int,
        num_heads: int = 2,
        key_dim: int | None = None,
        ffn_hidden: int | None = None,
        dropout: float = 0.0,
        layer_scale_init: float | None = 1e-2,
        radial_basis_type: str = "bessel",
        radial_trainable: bool = False,
        gaussian_width: float = 0.5,
        use_distance_penalty: bool = True,
        attention_mode: str = "full",
    ):
        super().__init__()
        # Ablation arms. Everything outside the logit construction is identical,
        # so an accuracy difference is attributable to the weighting itself and
        # not to a different aggregation, readout or training path.
        #
        #   full     l = (q.k)/sqrt(K) + b_h(r) - softplus(s_h) r   (as published)
        #   no_qk    l =                 b_h(r) - softplus(s_h) r   (distance only)
        #   uniform  l = 0            -> alpha is the cutoff-weighted mean
        #   none     the attention update is skipped entirely
        #
        # `no_qk` is the decisive control: it keeps a *learned radial filter*,
        # which is what plain ACE/MACE already provide, and removes only the
        # content-dependent query-key term. `uniform` drops learned weighting
        # too; `none` leaves the ACE descriptor plus the scalar FFN and readout.
        valid_modes = ("full", "no_qk", "uniform", "none")
        if attention_mode not in valid_modes:
            raise ValueError(
                f"attention_mode must be one of {valid_modes}, got {attention_mode!r}"
            )
        self.attention_mode = attention_mode
        self._uses_query_key = attention_mode == "full"
        self._uses_learned_logits = attention_mode in ("full", "no_qk")
        self._uses_attention_update = attention_mode != "none"
        self.node_irreps = o3.Irreps(node_irreps)
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.hidden_dim = int(hidden_dim)
        self.edge_scalar_dim = int(edge_scalar_dim)
        self.num_heads = max(1, int(num_heads))
        self.key_dim = key_dim or max(8, hidden_dim // self.num_heads)

        self.node_norm = ScalarPreNorm(hidden_dim)
        # Projections an arm does not use are not created, so each arm reports
        # an honest parameter count rather than one padded with dead weights.
        if self._uses_query_key:
            self.edge_scalar_norm = nn.LayerNorm(self.edge_scalar_dim)
            self.q_proj = nn.Linear(hidden_dim, self.num_heads * self.key_dim, bias=False)
            self.k_proj = nn.Linear(
                self.edge_scalar_dim, self.num_heads * self.key_dim, bias=False
            )
        else:
            self.edge_scalar_norm = None
            self.q_proj = None
            self.k_proj = None
        self.radial_basis = SmoothACERadialBasis(
            r_max,
            num_radial,
            basis_type=radial_basis_type,
            trainable=radial_trainable,
            gaussian_width=gaussian_width,
        )
        radial_hidden = max(16, self.num_heads * 4)
        self.radial_bias = (
            nn.Sequential(
                nn.Linear(num_radial, radial_hidden),
                nn.SiLU(),
                nn.Linear(radial_hidden, self.num_heads),
            )
            if self._uses_learned_logits
            else None
        )
        self.distance_log_scale = (
            nn.Parameter(torch.zeros(self.num_heads))
            if use_distance_penalty and self._uses_learned_logits
            else None
        )

        # The `none` arm performs no edge aggregation, so the equivariant value
        # and output projections would be dead weight; leaving them out keeps its
        # reported parameter count equal to what it actually uses.
        self.value_proj = (
            nn.ModuleList(
                [o3.Linear(self.edge_irreps, self.node_irreps) for _ in range(self.num_heads)]
            )
            if self._uses_attention_update
            else None
        )
        self.out_proj = (
            o3.Linear(self.node_irreps, self.node_irreps)
            if self._uses_attention_update
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self._layer_scale_blocks = tuple(
            (int(multiplicity), int(irrep.dim))
            for multiplicity, irrep in self.node_irreps
        )
        self.layer_scale_attn = (
            nn.Parameter(
                torch.full(
                    (_layer_scale_num_copies(self._layer_scale_blocks),),
                    float(layer_scale_init),
                )
            )
            if layer_scale_init is not None and self._uses_attention_update
            else None
        )

        self.non_scalar_irreps = o3.Irreps(self.node_irreps[1:])
        self.non_scalar_norm = o3.Norm(self.non_scalar_irreps, squared=True)
        invariant_dim = hidden_dim + self.non_scalar_norm.irreps_out.dim
        ffn_hidden = ffn_hidden or hidden_dim * 2
        self.scalar_norm = nn.LayerNorm(invariant_dim)
        self.scalar_ffn = nn.Sequential(
            nn.Linear(invariant_dim, ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        self.layer_scale_ffn = (
            nn.Parameter(torch.full((hidden_dim,), float(layer_scale_init)))
            if layer_scale_init is not None
            else None
        )

    def _apply_scalar_ffn(self, x: torch.Tensor) -> torch.Tensor:
        scalars = x[:, : self.hidden_dim]
        rest = x[:, self.hidden_dim :]
        non_scalar_invariants = self.non_scalar_norm(rest)
        invariants = torch.cat((scalars, non_scalar_invariants), dim=-1)
        scalar_update = self.scalar_ffn(self.scalar_norm(invariants))
        if self.layer_scale_ffn is not None:
            scalar_update = scalar_update * self.layer_scale_ffn
        return torch.cat((scalars + scalar_update, rest), dim=-1)

    def _attention_layer_scale(self) -> torch.Tensor:
        """Expanded residual scale, using the frozen constant when available."""
        frozen = getattr(self, "_frozen_layer_scale", None)
        if frozen is not None:
            return frozen
        return _irrepwise_layer_scale(self.layer_scale_attn, self._layer_scale_blocks)

    def forward(
        self,
        x: torch.Tensor,
        edge_features: torch.Tensor,
        receiver: torch.Tensor,
        edge_len: torch.Tensor,
        cutoff: torch.Tensor,
        temperature_scale: float = 1.0,
    ) -> torch.Tensor:
        if receiver.numel() == 0 or not self._uses_attention_update:
            # `none` still applies the scalar FFN, so the arms differ only by the
            # attention update and not by the presence of a nonlinearity.
            return self._apply_scalar_ffn(x)

        if self._uses_query_key:
            x_norm = self.node_norm(x)
            queries = self.q_proj(x_norm[:, : self.hidden_dim]).view(
                -1, self.num_heads, self.key_dim
            )
            edge_scalars = self.edge_scalar_norm(
                edge_features[:, : self.edge_scalar_dim]
            )
            keys = self.k_proj(edge_scalars).view(-1, self.num_heads, self.key_dim)
            logits = (queries[receiver] * keys).sum(dim=-1) / math.sqrt(self.key_dim)
        else:
            # Same shape and dtype as the full arm, so the softmax, the cutoff
            # weighting and the aggregation below are bit-for-bit the same code.
            logits = torch.zeros(
                (receiver.shape[0], self.num_heads),
                dtype=edge_features.dtype,
                device=edge_features.device,
            )
        if self.radial_bias is not None:
            logits = logits + self.radial_bias(self.radial_basis(edge_len))
        if self.distance_log_scale is not None:
            logits = logits - F.softplus(self.distance_log_scale)[None, :] * edge_len[:, None]
        logits = logits / max(float(temperature_scale), 1e-4)

        alpha = _segment_cutoff_softmax(logits, receiver, cutoff, x.shape[0])
        alpha = self.dropout(alpha)
        update = torch.zeros_like(x)
        for head, value_layer in enumerate(self.value_proj):
            values = value_layer(edge_features)
            update.index_add_(
                0,
                receiver,
                alpha[:, head : head + 1].to(values.dtype) * values,
            )
        update = self.out_proj(update / self.num_heads)
        if self.layer_scale_attn is not None:
            update = update * self._attention_layer_scale()
        x = x + update

        return self._apply_scalar_ffn(x)


class TransformersACE(nn.Module):
    """Corrected strictly local ACE-attention potential (architecture v2)."""

    architecture_version = 2

    def __init__(
        self,
        r_max=5.0,
        l_max=2,
        num_radial=8,
        hidden_dim=128,
        num_layers=2,
        radial_basis_type: str = "bessel",
        radial_trainable: bool = False,
        envelope_exponent: int = 5,
        gaussian_width: float = 0.5,
        descriptor_passes: int = 1,
        descriptor_residual: bool = True,
        radial_mlp_hidden: int = 32,
        radial_mlp_layers: int = 2,
        correlation_order: int = 4,
        correlation_channels: int = 16,
        attention_num_heads: int | None = None,
        attention_key_dim: int | None = None,
        attention_ffn_hidden: int | None = None,
        attention_dropout: float = 0.0,
        attention_layer_scale_init: float | None = 1e-2,
        attention_distance_penalty: bool = True,
        attention_mode: str = "full",
        transformer_num_heads: int = 4,
        transformer_ffn_hidden: int | None = None,
        transformer_dropout: float = 0.0,
        use_aux_force_head: bool = False,
        use_aux_stress_head: bool = False,
        interleave_descriptor: bool = False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.r_max = float(r_max)
        self.l_max = int(l_max)
        self.descriptor_passes = 1
        self.descriptor_residual = True
        self.interleave_descriptor = False
        self.use_aux_force_head = False
        self.use_aux_stress_head = False

        self.emb = nn.Embedding(118, hidden_dim)
        self.ace = ACEV2Descriptor(
            r_max=r_max,
            l_max=l_max,
            num_radial=num_radial,
            hidden_dim=hidden_dim,
            correlation_order=correlation_order,
            correlation_channels=correlation_channels,
            radial_basis_type=radial_basis_type,
            radial_trainable=radial_trainable,
            gaussian_width=gaussian_width,
            radial_mlp_hidden=radial_mlp_hidden,
            radial_mlp_layers=radial_mlp_layers,
        )
        self.attention_irreps = self.ace.irreps_out
        attention_heads = max(1, int(attention_num_heads or transformer_num_heads))
        attention_hidden = attention_ffn_hidden or transformer_ffn_hidden
        attention_dropout = float(
            attention_dropout if attention_dropout is not None else transformer_dropout
        )
        edge_scalar_dim = self.ace.irreps_correlation[0].mul
        self.attention_mode = str(attention_mode)
        self.layers = nn.ModuleList(
            [
                StrictLocalEquivariantAttentionBlock(
                    node_irreps=self.attention_irreps,
                    edge_irreps=self.ace.irreps_correlation,
                    hidden_dim=hidden_dim,
                    edge_scalar_dim=edge_scalar_dim,
                    r_max=r_max,
                    num_radial=num_radial,
                    num_heads=attention_heads,
                    key_dim=attention_key_dim,
                    ffn_hidden=attention_hidden,
                    dropout=attention_dropout,
                    layer_scale_init=attention_layer_scale_init,
                    radial_basis_type=radial_basis_type,
                    radial_trainable=radial_trainable,
                    gaussian_width=gaussian_width,
                    use_distance_penalty=attention_distance_penalty,
                    attention_mode=attention_mode,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.aux_force_head = None
        self.aux_stress_head = None

    def _batched_electrostatic_energy(
        self,
        electrostatics,
        h,
        pos,
        batch,
        n_struct,
        edge_index,
        edge_vec,
        edge_len,
        cell,
        edge_shift,
        data,
        positions=None,
        return_outputs=False,
    ):
        """Per-structure electrostatic energies of a batched graph, ``[n_struct]``.

        One call to the electrostatics per structure, on that structure's slice
        of the atom and edge arrays.  That is deliberate and is the whole
        content of this method; see the comment at the call site in
        :meth:`forward_batched` for why a single whole-batch call is wrong.

        Slicing gives the three properties the batched path needs:

        * **Per-structure fragmenting.**  Each call sees only its own atoms, so
          ``assign_fragments`` cannot merge structures, the per-fragment charge
          constraints of SPEC 1.3 are enforced within a structure, and no
          charge can equilibrate between structures that are not connected.
        * **A per-structure charge specification.**  ``total_charge``,
          ``fragment_id``/``fragment_charges`` and ``fragment_spins`` carry a
          structure dimension in the batch (see
          :func:`_structure_charge_specs` for the layout
          ``AtomisticDataset.collate_batched`` emits) and each call is handed
          its own structure's slice, in exactly the form the single-structure
          :meth:`forward` hands the electrostatics for that structure alone.
        * **A ``[3, 3]`` cell.**  ``cell`` is ``[n_struct, 3, 3]`` here;
          ``trace_electrostatics.energy.assemble`` and the Ewald route below it
          take a single cell per call.  ``cell[s]`` is that cell, and it is the
          *strained* one, so the stress contribution follows from the same
          ``strain_params`` the rest of this method differentiates.  The slice
          is taken inside :func:`_electrostatic_energy_term`, together with the
          SPEC 7 periodicity decision, so that one structure of a batch and the
          same structure alone are handed byte-identical arguments.

        The energy of structure ``s`` is therefore built from exactly the
        tensors the single-structure :meth:`forward` would have built it from,
        and agrees with it to machine precision.

        Args:
            positions: per-structure position tensors from
                :func:`_structure_positions`, used in place of
                ``pos[atoms_of_s]`` so that derivatives taken from the returned
                outputs are complete.  Values are identical either way.
            return_outputs: also return one :class:`ElectrostaticOutput` per
                structure.
        """
        atom_lists = _structure_atom_lists(batch, n_struct)
        charge_specs = _structure_charge_specs(data, batch, n_struct, atom_lists)

        sender_struct = batch[edge_index[0]]
        receiver_struct = batch[edge_index[1]]
        if bool((sender_struct != receiver_struct).any()):
            raise ValueError(
                "The batched graph has an edge connecting two structures. The "
                "batched path assumes a disjoint union (see collate_batched), "
                "and the electrostatics would silently equilibrate charge "
                "across the join."
            )

        # Periodicity read to the host once for the whole batch, not per structure.
        pbc_data = data
        if data is not None and torch.is_tensor(data.get(ELECTROSTATIC_PBC_KEY, None)):
            pbc_data = {ELECTROSTATIC_PBC_KEY: data[ELECTROSTATIC_PBC_KEY].detach().cpu()}
        pbc_cell = None if cell is None else cell.detach().cpu()
        pbcs = [_electrostatic_pbc(pbc_cell, pbc_data, structure) for structure in range(n_struct)]
        if self._block_diagonal_applicable(electrostatics, charge_specs, pbcs, atom_lists):
            return self._block_diagonal_electrostatic_energy(
                electrostatics, h, pos, batch, n_struct, edge_index, edge_vec, edge_len,
                charge_specs, atom_lists, positions, return_outputs,
            )

        energies, outputs = [], []
        for structure in range(n_struct):
            atom_index = atom_lists[structure]
            edge_mask = receiver_struct == structure
            # Global atom index -> index within this structure, so the sliced
            # edge list addresses the sliced atom arrays.
            local_index = torch.full_like(batch, -1)
            local_index[atom_index] = torch.arange(
                atom_index.numel(), device=batch.device
            )
            term = _electrostatic_energy_term(
                electrostatics,
                h[atom_index],
                pos[atom_index] if positions is None else positions[structure],
                local_index[edge_index[:, edge_mask]],
                edge_vec[edge_mask],
                edge_len[edge_mask],
                cell,
                None if edge_shift is None else edge_shift[edge_mask],
                # This structure's own charge specification, in the form the
                # single-structure path hands the electrostatics (None for a
                # neutral structure with nothing to say).  Periodicity is read
                # from the batch dictionary separately, per structure.
                charge_specs[structure],
                structure=structure,
                pbc=pbcs[structure],
                return_output=return_outputs,
            )
            if return_outputs:
                term, output = term
                outputs.append(output)
            energies.append(term)
        if return_outputs:
            return torch.stack(energies), outputs
        return torch.stack(energies)

    #: SPEC 24 switch: False forces the per-structure loop (for A/B tests).
    batched_electrostatics_solve = True

    def _block_diagonal_applicable(self, electrostatics, specs, pbcs, atom_lists):
        """Whether every structure of the batch can join one block-diagonal solve."""
        if not self.batched_electrostatics_solve or not hasattr(electrostatics, "energies_by_structure"):
            return False
        config = getattr(electrostatics, "config", None)
        heads = getattr(electrostatics, "heads", None)
        if config is None or heads is None or getattr(heads, "spin", False) or config.dispersion:
            return False
        if electrostatics.training and float(config.solver_seed_loss_weight) > 0.0:
            return False
        if any(any(pbc) for pbc in pbcs):
            return False
        if any(spec is not None and any(key in spec for key in _BLOCK_DIAGONAL_UNSUPPORTED_KEYS)
               for spec in specs):
            return False
        from trace_electrostatics.config import resolve_solver

        return all(
            resolve_solver(config, (False, False, False), int(atoms.numel())) == "dense"
            for atoms in atom_lists
        )

    def _block_diagonal_electrostatic_energy(
        self, electrostatics, h, pos, batch, n_struct, edge_index, edge_vec, edge_len,
        specs, atom_lists, positions, return_outputs,
    ):
        """SPEC 24: the structures of a batch in consecutive block-diagonal solves.

        Each structure's fragments and fragment charges follow exactly the
        single-structure rules (explicit labels and charges as given; otherwise
        the connected components, carrying the net charge if there is one
        component and zero if the structure is neutral -- the same function
        decides), offset into one global labelling.  Energies, forces, charges
        and every derivative equal those of the per-structure loop to the solve
        tolerance (tests/electrostatics/test_batched_structures.py,
        tests/test_train_electrostatics.py).
        """
        from trace_electrostatics.energy import _resolve_fragment_charges
        from trace_electrostatics.topology import assign_fragments

        cap = _block_diagonal_max_atoms(pos.device)
        chunks, current, size = [], [], 0
        for structure in range(n_struct):
            n = int(atom_lists[structure].numel())
            if current and size + n > cap:
                chunks.append(current)
                current, size = [], 0
            current.append(structure)
            size += n
        if current:
            chunks.append(current)

        energies = [None] * n_struct
        outputs = [None] * n_struct
        receiver = batch[edge_index[1]]
        for chunk in chunks:
            atoms = torch.cat([atom_lists[s] for s in chunk])
            local = torch.full_like(batch, -1)
            local[atoms] = torch.arange(atoms.numel(), device=batch.device)
            edge_mask = torch.isin(receiver, torch.as_tensor(chunk, device=batch.device))
            chunk_edges = local[edge_index[:, edge_mask]]
            structure_id = torch.cat([
                torch.full((int(atom_lists[s].numel()),), k, dtype=torch.long, device=batch.device)
                for k, s in enumerate(chunk)
            ])
            components = None
            fragment_ids, fragment_charges, offset, start = [], [], 0, 0
            for k, s in enumerate(chunk):
                n = int(atom_lists[s].numel())
                spec = specs[s] or {}
                if "fragment_id" in spec:
                    labels = torch.as_tensor(spec["fragment_id"], device=batch.device).long()
                    charges = torch.as_tensor(spec["fragment_charges"], dtype=pos.dtype, device=pos.device)
                else:
                    if components is None:
                        # One host transfer per chunk; the per-structure labels are then
                        # relabelled on the host, without a synchronisation each.
                        components = assign_fragments(int(atoms.numel()), chunk_edges).cpu()
                    own = components[start:start + n]
                    _, labels = torch.unique(own, sorted=True, return_inverse=True)
                    charges = _resolve_fragment_charges(
                        None, float(spec.get("total_charge", 0.0)), int(labels.max()) + 1, pos,
                    )
                    labels = labels.to(batch.device)
                fragment_ids.append(labels + offset)
                fragment_charges.append(charges.reshape(-1))
                offset += int(charges.numel())
                start += n
            chunk_pos = (
                torch.cat([positions[s] for s in chunk]) if positions is not None else pos[atoms]
            )
            quadrupoles = None
            for s in chunk:
                if specs[s] is not None and specs[s].get("quadrupoles") is not None:
                    quadrupoles = bool(specs[s]["quadrupoles"])
            chunk_energies, state = electrostatics.energies_by_structure(
                h[atoms], chunk_pos, chunk_edges, edge_vec[edge_mask], edge_len[edge_mask],
                structure_id=structure_id,
                fragment_id=torch.cat(fragment_ids),
                fragment_charges=torch.cat(fragment_charges),
                species=None if "z" not in (specs[chunk[0]] or {}) else torch.cat(
                    [specs[s]["z"] for s in chunk]),
                quadrupoles=quadrupoles,
            )
            for k, s in enumerate(chunk):
                energies[s] = chunk_energies[k]
            if return_outputs:
                self._block_diagonal_outputs(
                    electrostatics, state, chunk, chunk_energies, structure_id, atom_lists,
                    h, pos, positions, edge_index, edge_vec, edge_len, batch, specs, outputs,
                )
        if return_outputs:
            return torch.stack(energies), outputs
        return torch.stack(energies)

    def _block_diagonal_outputs(
        self, electrostatics, state, chunk, chunk_energies, structure_id, atom_lists,
        h, pos, positions, edge_index, edge_vec, edge_len, batch, specs, outputs,
    ):
        """One ElectrostaticOutput per structure of a block-diagonal solve.

        The solve result is sliced to the structure (charges, dipoles, reference
        charges, its own bonds' transfers, quadrupoles when present) with the
        structure's own energy.  ``state()`` rebuilds the structure ALONE, through
        the same function the per-structure loop uses, and verifies that the
        rebuilt energy reproduces this one -- so a polarizability or Born charge
        is never taken from an operator other than the one whose energy is trained.
        """
        result = state.solve_result
        bond_structure = structure_id.index_select(0, state.assembly.bond_index[0])
        receiver = batch[edge_index[1]]
        start = 0
        for k, s in enumerate(chunk):
            n = int(atom_lists[s].numel())
            rows = slice(start, start + n)
            start += n
            sliced = dataclasses.replace(
                result,
                t=result.t[bond_structure == k],
                rho=result.rho[rows],
                mu=result.mu[rows],
                q=result.q[rows],
                energy=chunk_energies[k],
                theta_hat=None if result.theta_hat is None else result.theta_hat[rows],
                theta=None if result.theta is None else result.theta[rows],
            )
            atom_index = atom_lists[s]
            edge_mask = receiver == s
            local_index = torch.full_like(batch, -1)
            local_index[atom_index] = torch.arange(atom_index.numel(), device=batch.device)
            structure_pos = pos[atom_index] if positions is None else positions[s]
            pbc = (False, False, False)
            spec = specs[s]
            outputs[s] = ElectrostaticOutput(
                energy=chunk_energies[k],
                solve_result=sliced,
                positions=structure_pos,
                pbc=pbc,
                total_charge=float(0.0 if spec is None else spec.get("total_charge", 0.0)),
                build_state=functools.partial(
                    _rebuild_electrostatic_state,
                    electrostatics,
                    chunk_energies[k],
                    h[atom_index],
                    structure_pos,
                    local_index[edge_index[:, edge_mask]],
                    edge_vec[edge_mask],
                    edge_len[edge_mask],
                    None,
                    None,
                    pbc,
                    spec,
                ),
            )

    def forward_batched(
        self,
        data,
        training=False,
        temperature_scale: float = 1.0,
        detach_pos: bool = True,
        compute_stress: bool | None = None,
        return_electrostatics: bool = False,
    ):
        """Evaluate several structures as one disconnected graph.

        ``data['batch']`` maps each atom to its structure, ``data['cell']`` and
        ``data['volume']`` carry a leading structure dimension, and the edge
        indices are already offset into the concatenated atom list.  Returns
        *per-structure* energies and stresses; forces stay per atom.

        This exists because the GPU is launch-bound for single small cells: one
        batched graph issues the same number of kernels as one structure but
        does B times the work inside them.  The single-structure ``forward`` is
        left byte-identical so the deployed TorchScript/AOT paths cannot regress.

        With ``return_electrostatics`` (attached electrostatics required),
        ``aux['electrostatics']`` holds one :class:`ElectrostaticOutput` per
        structure and the autograd graph is kept alive after the force
        derivative, so that dipole, polarizability and Born-charge losses can
        be built from it.
        """
        z, pos, edge_index = data['z'], data['pos'], data['edge_index']
        batch = data['batch']
        n_struct = int(data['n_structures'])
        cell = data.get('cell', None)
        edge_shift = data.get('edge_shift', None)
        cell_volume = data.get('volume', None)

        if compute_stress is None:
            compute_stress = training

        electrostatics = getattr(self, "electrostatics", None)
        if return_electrostatics and electrostatics is None:
            raise ValueError(
                "return_electrostatics=True, but no electrostatics is attached "
                "to this model (trace_electrostatics.attach_electrostatics)."
            )

        if detach_pos:
            pos = pos.detach()
        pos.requires_grad_(True)

        if cell is not None:
            cell = cell.to(device=pos.device, dtype=pos.dtype)

        if compute_stress and (cell is not None or cell_volume is not None):
            strain_params = torch.zeros(
                n_struct, 6, device=pos.device, dtype=pos.dtype, requires_grad=True
            )
            epsilon = torch.zeros(n_struct, 3, 3, device=pos.device, dtype=pos.dtype)
            epsilon[:, 0, 0] = strain_params[:, 0]
            epsilon[:, 1, 1] = strain_params[:, 1]
            epsilon[:, 2, 2] = strain_params[:, 2]
            epsilon[:, 0, 1] = epsilon[:, 1, 0] = strain_params[:, 3]
            epsilon[:, 0, 2] = epsilon[:, 2, 0] = strain_params[:, 4]
            epsilon[:, 1, 2] = epsilon[:, 2, 1] = strain_params[:, 5]
            deformation = torch.eye(3, device=pos.device, dtype=pos.dtype).unsqueeze(0) + epsilon
            pos = torch.bmm(pos.unsqueeze(1), deformation[batch]).squeeze(1)
            if cell is not None:
                cell = torch.bmm(cell, deformation)
        else:
            strain_params = None
            epsilon = None
            deformation = None

        # Per-structure position tensors that every quantity of their structure
        # descends from, so that a Born charge taken from the returned outputs
        # is the complete derivative (see _structure_positions).  Same values.
        # Forces are still taken w.r.t. the tensor BEFORE the split: each
        # structure's electrostatics also depends on its piece directly (the
        # pair kernels), and only this tensor is upstream of every path.
        force_pos = pos
        structure_positions = None
        if return_electrostatics:
            pos, structure_positions = _structure_positions(pos, batch, n_struct)

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        if edge_shift is not None and cell is not None:
            edge_shift = edge_shift.to(device=pos.device, dtype=pos.dtype)
            edge_cell = cell[batch[edge_index[1]]]
            edge_vec = edge_vec + torch.bmm(edge_shift.unsqueeze(1), edge_cell).squeeze(1)
        edge_len = torch.norm(edge_vec, dim=1)

        h, edge_features, cutoff = self.ace(
            self.emb(z), edge_index, edge_vec, edge_len, return_edge_features=True
        )
        receiver = edge_index[1]
        for layer in self.layers:
            h = layer(
                h, edge_features, receiver, edge_len, cutoff,
                temperature_scale=temperature_scale,
            )

        atom_energy = self.readout(h[:, : self.hidden_dim]).squeeze(-1)
        E = torch.zeros(n_struct, device=pos.device, dtype=atom_energy.dtype)
        E = E.index_add(0, batch, atom_energy)

        # Optional variational electrostatics (trace_electrostatics).  Absent
        # unless attach_electrostatics() has been called.  Added here, upstream
        # of the autograd calls below, so forces and stress come from the
        # existing machinery: pos and cell are already strained at this point.
        #
        # Unlike the single-structure forward this CANNOT be one call on the
        # whole batch.  U_elec is a property of one configuration, and the
        # batched graph is a disjoint union of configurations that share no
        # physics: a single call would build the open-boundary pair kernel
        # across structures and let charge equilibrate between systems that are
        # not connected -- the size-inconsistency failure of SPEC 1.3 -- and
        # would hand the Ewald route a [n_struct, 3, 3] cell where it documents
        # "a single periodic cell per call; batched systems are looped or
        # blocked by the caller" (trace_electrostatics/ewald.py, Conventions).
        # So the loop is the caller-side blocking that contract asks for: each
        # structure gets its own fragments, its own per-fragment charge
        # constraints, its own net charge and its own [3, 3] cell, and the
        # per-structure energy is identical to the one the single-structure
        # forward computes.
        aux = {}
        if electrostatics is not None:
            term = self._batched_electrostatic_energy(
                electrostatics, h, pos, batch, n_struct,
                edge_index, edge_vec, edge_len, cell, edge_shift, data,
                positions=structure_positions,
                return_outputs=return_electrostatics,
            )
            if return_electrostatics:
                term, aux["electrostatics"] = term
            E = E + term

        total = E.sum()

        # The observables in aux["electrostatics"] are derivatives of this same
        # graph (Born charges differentiate through it again), so it must
        # survive the force derivative when they are requested.
        keep_graph = training or return_electrostatics
        grads = torch.autograd.grad(
            total, force_pos,
            create_graph=training,
            retain_graph=keep_graph or epsilon is not None,
            allow_unused=True,
        )[0]
        F = -grads if grads is not None else torch.zeros_like(force_pos)

        S = torch.zeros(n_struct, 3, 3, device=pos.device, dtype=pos.dtype)
        if compute_stress and epsilon is not None:
            g_eps = torch.autograd.grad(
                total, strain_params,
                create_graph=training, retain_graph=keep_graph, allow_unused=True,
            )[0]
            if g_eps is not None:
                stress = torch.zeros(n_struct, 3, 3, device=pos.device, dtype=pos.dtype)
                stress[:, 0, 0] = g_eps[:, 0]
                stress[:, 1, 1] = g_eps[:, 1]
                stress[:, 2, 2] = g_eps[:, 2]
                stress[:, 0, 1] = stress[:, 1, 0] = 0.5 * g_eps[:, 3]
                stress[:, 0, 2] = stress[:, 2, 0] = 0.5 * g_eps[:, 4]
                stress[:, 1, 2] = stress[:, 2, 1] = 0.5 * g_eps[:, 5]
                if cell is not None:
                    volume = torch.det(cell).abs().clamp_min(1e-12)
                else:
                    volume = cell_volume * torch.det(deformation)
                S = stress / volume.view(-1, 1, 1)

        return E, F, S, aux

    def forward(
        self,
        data,
        training=False,
        temperature_scale: float = 1.0,
        detach_pos: bool = True,
        compute_stress: bool | None = None,
        return_electrostatics: bool = False,
    ):
        z, pos, edge_index = data['z'], data['pos'], data['edge_index']
        cell_volume = data.get('volume', None)
        cell = data.get('cell', None)
        edge_shift = data.get('edge_shift', None)

        if compute_stress is None:
            compute_stress = training
        if return_electrostatics and getattr(self, "electrostatics", None) is None:
            raise ValueError(
                "return_electrostatics=True, but no electrostatics is attached "
                "to this model (trace_electrostatics.attach_electrostatics)."
            )

        # Keep this derivative and strain path identical to architecture v1.
        if detach_pos:
            pos = pos.detach()
        pos.requires_grad_(True)

        if cell is not None:
            cell = cell.to(device=pos.device, dtype=pos.dtype)

        if compute_stress and cell_volume is not None:
            strain_params = torch.zeros(6, device=pos.device, dtype=pos.dtype, requires_grad=True)

            epsilon = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
            epsilon[0, 0] = strain_params[0]
            epsilon[1, 1] = strain_params[1]
            epsilon[2, 2] = strain_params[2]
            epsilon[0, 1] = epsilon[1, 0] = strain_params[3]
            epsilon[0, 2] = epsilon[2, 0] = strain_params[4]
            epsilon[1, 2] = epsilon[2, 1] = strain_params[5]

            deformation = torch.eye(3, device=pos.device, dtype=pos.dtype) + epsilon
            pos = pos @ deformation
            if cell is not None:
                cell = cell @ deformation
        else:
            strain_params = None
            epsilon = None

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        if edge_shift is not None and cell is not None:
            edge_shift = edge_shift.to(device=pos.device, dtype=pos.dtype)
            edge_vec = edge_vec + edge_shift @ cell
        edge_len = torch.norm(edge_vec, dim=1)

        h, edge_features, cutoff = self.ace(
            self.emb(z),
            edge_index,
            edge_vec,
            edge_len,
            return_edge_features=True,
        )
        receiver = edge_index[1]
        for layer in self.layers:
            h = layer(
                h,
                edge_features,
                receiver,
                edge_len,
                cutoff,
                temperature_scale=temperature_scale,
            )

        scalars = h[:, : self.hidden_dim]
        E = torch.sum(self.readout(scalars))

        # Optional variational electrostatics (trace_electrostatics).  Absent
        # unless attach_electrostatics() has been called.  Added here, upstream
        # of the autograd calls below, so forces and stress come from the
        # existing machinery: pos and cell are already strained at this point.
        # Note h, not scalars -- the heads need the full equivariant features
        # (A_mu is 0e+2e).  Periodicity goes through
        # _electrostatic_energy_term, which is where SPEC 7 is enforced for all
        # four hooks.  See docs/electrostatics.md.
        aux = {}
        electrostatics = getattr(self, "electrostatics", None)
        if electrostatics is not None:
            term = _electrostatic_energy_term(
                electrostatics,
                h,
                pos,
                edge_index,
                edge_vec,
                edge_len,
                cell,
                edge_shift,
                data,
                return_output=return_electrostatics,
            )
            if return_electrostatics:
                term, output = term
                aux["electrostatics"] = [output]
            E = E + term

        # The observables in aux["electrostatics"] are derivatives of this same
        # graph (Born charges differentiate through it again), so it must
        # survive the force derivative when they are requested.
        keep_graph = training or return_electrostatics
        grads = torch.autograd.grad(
            E,
            pos,
            create_graph=training,
            retain_graph=keep_graph or epsilon is not None,
            allow_unused=True,
        )[0]
        F = -grads if grads is not None else torch.zeros_like(pos)

        S = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
        if compute_stress and epsilon is not None:
            g_eps = torch.autograd.grad(
                E,
                strain_params,
                create_graph=training,
                retain_graph=keep_graph,
                allow_unused=True,
            )[0]
            if g_eps is not None:
                stress = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
                stress[0, 0] = g_eps[0]
                stress[1, 1] = g_eps[1]
                stress[2, 2] = g_eps[2]
                stress[0, 1] = stress[1, 0] = 0.5 * g_eps[3]
                stress[0, 2] = stress[2, 0] = 0.5 * g_eps[4]
                stress[1, 2] = stress[2, 1] = 0.5 * g_eps[5]

                if cell is not None:
                    volume = torch.det(cell).abs().clamp_min(1e-12)
                else:
                    volume = cell_volume * torch.det(deformation)
                S = stress / volume

        return E, F, S, aux


class TensorialFixedEnvironmentAttentionBlock(nn.Module):
    """Equivariant cross-attention from one center to frozen ACE tokens.

    A token may describe a directed neighbor density or a body-ordered ACE
    moment.  Keys and values are functions only of those fixed tokens.  The
    only evolving state is the receiving center feature, so stacking blocks
    cannot expand the physical receptive field.
    """

    def __init__(
        self,
        node_irreps,
        token_irreps,
        hidden_dim: int,
        token_scalar_dim: int,
        r_max: float,
        num_radial: int,
        num_token_kinds: int,
        num_heads: int = 2,
        key_dim: int | None = None,
        ffn_hidden: int | None = None,
        dropout: float = 0.0,
        layer_scale_init: float | None = 1e-2,
        radial_basis_type: str = "bessel",
        radial_trainable: bool = False,
        gaussian_width: float = 0.5,
        use_distance_penalty: bool = True,
    ):
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps)
        self.token_irreps = o3.Irreps(token_irreps)
        self.score_irreps = self.token_irreps
        self.hidden_dim = int(hidden_dim)
        self.token_scalar_dim = int(token_scalar_dim)
        self.num_heads = max(1, int(num_heads))
        self.key_dim = key_dim or max(8, hidden_dim // self.num_heads)

        self.node_norm = ScalarPreNorm(hidden_dim)
        self.token_scalar_norm = nn.LayerNorm(self.token_scalar_dim)
        self.token_kind_embedding = nn.Embedding(num_token_kinds, self.token_scalar_dim)
        self.q_scalar = nn.Linear(hidden_dim, self.num_heads * self.key_dim, bias=False)
        self.k_scalar = nn.Linear(
            self.token_scalar_dim,
            self.num_heads * self.key_dim,
            bias=False,
        )

        # Matching irreps are contracted to 0e scalars.  In the real e3nn
        # basis the normalized inner product is the corresponding CG scalar.
        self.q_tensor = o3.Linear(self.node_irreps, self.score_irreps)
        self.k_tensor = o3.Linear(self.token_irreps, self.score_irreps)
        self.score_channels = sum(mul for mul, _ in self.score_irreps)
        # Keep only primitive static metadata in the forward path.  e3nn's
        # Irrep objects intentionally do not implement ``len``; iterating over
        # them during forward prevents torch.export/AOTInductor capture.
        self._score_blocks = tuple(
            (int(multiplicity), int(irrep.dim))
            for multiplicity, irrep in self.score_irreps
        )
        self.multipole_score = nn.Linear(self.score_channels, self.num_heads, bias=False)

        self.radial_basis = SmoothACERadialBasis(
            r_max,
            num_radial,
            basis_type=radial_basis_type,
            trainable=radial_trainable,
            gaussian_width=gaussian_width,
        )
        radial_hidden = max(16, self.num_heads * 4)
        self.radial_bias = nn.Sequential(
            nn.Linear(num_radial, radial_hidden),
            nn.SiLU(),
            nn.Linear(radial_hidden, self.num_heads),
        )
        self.distance_log_scale = (
            nn.Parameter(torch.zeros(self.num_heads))
            if use_distance_penalty
            else None
        )

        self.value_tp = nn.ModuleList(
            [
                o3.FullyConnectedTensorProduct(
                    self.node_irreps,
                    self.token_irreps,
                    self.node_irreps,
                    internal_weights=True,
                    shared_weights=True,
                )
                for _ in range(self.num_heads)
            ]
        )
        self.out_proj = o3.Linear(self.node_irreps, self.node_irreps)
        self.dropout = nn.Dropout(dropout)
        self._layer_scale_blocks = tuple(
            (int(multiplicity), int(irrep.dim))
            for multiplicity, irrep in self.node_irreps
        )
        self.layer_scale_attn = (
            nn.Parameter(
                torch.full(
                    (_layer_scale_num_copies(self._layer_scale_blocks),),
                    float(layer_scale_init),
                )
            )
            if layer_scale_init is not None
            else None
        )

        self.non_scalar_irreps = o3.Irreps(self.node_irreps[1:])
        self.non_scalar_norm = o3.Norm(self.non_scalar_irreps, squared=True)
        invariant_dim = hidden_dim + self.non_scalar_norm.irreps_out.dim
        ffn_hidden = ffn_hidden or hidden_dim * 2
        self.scalar_norm = nn.LayerNorm(invariant_dim)
        self.scalar_ffn = nn.Sequential(
            nn.Linear(invariant_dim, ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        self.layer_scale_ffn = (
            nn.Parameter(torch.full((hidden_dim,), float(layer_scale_init)))
            if layer_scale_init is not None
            else None
        )

    def _multipole_invariants(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        receiver: torch.Tensor,
    ) -> torch.Tensor:
        terms = []
        offset = 0
        for multiplicity, irrep_dim in self._score_blocks:
            width = multiplicity * irrep_dim
            q_block = query[:, offset : offset + width].reshape(
                query.shape[0], multiplicity, irrep_dim
            )
            k_block = key[:, offset : offset + width].reshape(
                key.shape[0], multiplicity, irrep_dim
            )
            terms.append(
                (q_block[receiver] * k_block).sum(dim=-1) / math.sqrt(irrep_dim)
            )
            offset += width
        return torch.cat(terms, dim=-1)

    def _apply_scalar_ffn(self, x: torch.Tensor) -> torch.Tensor:
        scalars = x[:, : self.hidden_dim]
        rest = x[:, self.hidden_dim :]
        invariants = torch.cat((scalars, self.non_scalar_norm(rest)), dim=-1)
        update = self.scalar_ffn(self.scalar_norm(invariants))
        if self.layer_scale_ffn is not None:
            update = update * self.layer_scale_ffn
        return torch.cat((scalars + update, rest), dim=-1)

    def _attention_layer_scale(self) -> torch.Tensor:
        """Expanded residual scale, using the frozen constant when available."""
        frozen = getattr(self, "_frozen_layer_scale", None)
        if frozen is not None:
            return frozen
        return _irrepwise_layer_scale(self.layer_scale_attn, self._layer_scale_blocks)

    def forward(
        self,
        x: torch.Tensor,
        token_features: torch.Tensor,
        token_receiver: torch.Tensor,
        token_length: torch.Tensor,
        token_cutoff: torch.Tensor,
        token_kind: torch.Tensor,
        temperature_scale: float = 1.0,
    ) -> torch.Tensor:
        x_norm = self.node_norm(x)
        q_scalar = self.q_scalar(x_norm[:, : self.hidden_dim]).view(
            -1, self.num_heads, self.key_dim
        )
        token_scalars = self.token_scalar_norm(
            token_features[:, : self.token_scalar_dim]
        ) + self.token_kind_embedding(token_kind)
        k_scalar = self.k_scalar(token_scalars).view(
            -1, self.num_heads, self.key_dim
        )
        logits = (q_scalar[token_receiver] * k_scalar).sum(dim=-1) / math.sqrt(
            self.key_dim
        )

        q_tensor = self.q_tensor(x_norm)
        k_tensor = self.k_tensor(token_features)
        tensor_scores = self._multipole_invariants(q_tensor, k_tensor, token_receiver)
        logits = logits + self.multipole_score(tensor_scores)
        logits = logits + self.radial_bias(self.radial_basis(token_length))
        if self.distance_log_scale is not None:
            logits = logits - F.softplus(self.distance_log_scale)[None, :] * token_length[:, None]
        logits = logits / max(float(temperature_scale), 1e-4)

        alpha = self.dropout(
            _fixed_token_cutoff_softmax(logits, token_receiver, token_cutoff, x.shape[0])
        )
        update = torch.zeros_like(x)
        for head, value_layer in enumerate(self.value_tp):
            values = value_layer(x_norm[token_receiver], token_features)
            update.index_add_(
                0,
                token_receiver,
                alpha[:, head : head + 1].to(values.dtype) * values,
            )
        update = self.out_proj(update / self.num_heads)
        if self.layer_scale_attn is not None:
            update = update * self._attention_layer_scale()
        return self._apply_scalar_ffn(x + update)


class CumulantMultiQueryTensorialAttentionBlock(TensorialFixedEnvironmentAttentionBlock):
    """v4 local cross-attention with shared tensor values and invariant gates.

    Heads retain independent scalar and multipole queries/keys, while a single
    equivariant tensor product supplies the token value.  This removes the
    v3 ``num_heads``-fold tensor-product cost without introducing a sender
    hidden-state dependency.  Additive smooth shell logits are equivalent to a
    multiplicative shell factor before the per-center softmax.
    """

    def __init__(self, *args, num_shells: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_shells = max(1, int(num_shells))
        self.shared_value_tp = o3.FullyConnectedTensorProduct(
            self.node_irreps, self.token_irreps, self.node_irreps,
            internal_weights=True, shared_weights=True,
        )
        del self.value_tp
        self.value_head_gate = nn.Linear(self.hidden_dim, self.num_heads)
        centers = torch.linspace(0.0, self.radial_basis.cutoff.r_max, self.num_shells)
        self.register_buffer("shell_centers", centers)
        self.shell_log_width = nn.Parameter(torch.zeros(self.num_shells))
        self.shell_bias = nn.Linear(self.num_shells, self.num_heads, bias=False)

        self._non_scalar_blocks = tuple(
            (int(multiplicity), int(irrep.dim))
            for multiplicity, irrep in self.non_scalar_irreps
        )
        self._non_scalar_multiplicities = sum(multiplicity for multiplicity, _ in self._non_scalar_blocks)
        if self._non_scalar_multiplicities:
            invariant_dim = int(self.scalar_norm.normalized_shape[0])
            self.tensor_swiglu = nn.Sequential(
                nn.LayerNorm(invariant_dim),
                nn.Linear(invariant_dim, 2 * self._non_scalar_multiplicities),
            )
            # Identity initialization provides a conservative starting point.
            nn.init.zeros_(self.tensor_swiglu[-1].weight)
            nn.init.zeros_(self.tensor_swiglu[-1].bias)
        else:
            self.tensor_swiglu = None

    def _shell_features(self, token_length: torch.Tensor, token_cutoff: torch.Tensor) -> torch.Tensor:
        width = F.softplus(self.shell_log_width).clamp_min(1e-4)
        distances = token_length[:, None] - self.shell_centers[None, :]
        return torch.exp(-0.5 * (distances / width[None, :]).square()) * token_cutoff[:, None]

    def _apply_scalar_ffn(self, x: torch.Tensor) -> torch.Tensor:
        scalars, rest = x[:, : self.hidden_dim], x[:, self.hidden_dim :]
        invariants = torch.cat((scalars, self.non_scalar_norm(rest)), dim=-1)
        scalar_update = self.scalar_ffn(self.scalar_norm(invariants))
        if self.layer_scale_ffn is not None:
            scalar_update = scalar_update * self.layer_scale_ffn
        if self.tensor_swiglu is None:
            return torch.cat((scalars + scalar_update, rest), dim=-1)

        first, second = self.tensor_swiglu(invariants).chunk(2, dim=-1)
        gates = 0.1 * torch.tanh(F.silu(first) * second)
        blocks, feature_offset, gate_offset = [], 0, 0
        for multiplicity, irrep_dim in self._non_scalar_blocks:
            width = multiplicity * irrep_dim
            block = rest[:, feature_offset : feature_offset + width].reshape(-1, multiplicity, irrep_dim)
            gate = gates[:, gate_offset : gate_offset + multiplicity].unsqueeze(-1)
            blocks.append((block * (1.0 + gate)).reshape(-1, width))
            feature_offset += width
            gate_offset += multiplicity
        return torch.cat((scalars + scalar_update, *blocks), dim=-1)

    def forward(
        self, x, token_features, token_receiver, token_length, token_cutoff,
        token_kind, temperature_scale: float = 1.0,
    ):
        x_norm = self.node_norm(x)
        q_scalar = self.q_scalar(x_norm[:, : self.hidden_dim]).view(-1, self.num_heads, self.key_dim)
        token_scalars = self.token_scalar_norm(token_features[:, : self.token_scalar_dim])
        token_scalars = token_scalars + self.token_kind_embedding(token_kind)
        k_scalar = self.k_scalar(token_scalars).view(-1, self.num_heads, self.key_dim)
        logits = (q_scalar[token_receiver] * k_scalar).sum(dim=-1) / math.sqrt(self.key_dim)
        q_tensor, k_tensor = self.q_tensor(x_norm), self.k_tensor(token_features)
        logits = logits + self.multipole_score(self._multipole_invariants(q_tensor, k_tensor, token_receiver))
        logits = logits + self.radial_bias(self.radial_basis(token_length))
        logits = logits + self.shell_bias(self._shell_features(token_length, token_cutoff))
        if self.distance_log_scale is not None:
            logits = logits - F.softplus(self.distance_log_scale)[None, :] * token_length[:, None]
        logits = logits / max(float(temperature_scale), 1e-4)
        alpha = self.dropout(_fixed_token_cutoff_softmax(logits, token_receiver, token_cutoff, x.shape[0]))

        # The coefficient is scalar and center-local; only the fixed token
        # enters the shared equivariant value tensor product.
        values = self.shared_value_tp(x_norm[token_receiver], token_features)
        head_gate = torch.sigmoid(self.value_head_gate(x_norm[:, : self.hidden_dim]))
        coefficient = (alpha * head_gate[token_receiver]).mean(dim=-1, keepdim=True)
        update = torch.zeros_like(x)
        update.index_add_(0, token_receiver, coefficient.to(values.dtype) * values)
        update = self.out_proj(update)
        if self.layer_scale_attn is not None:
            update = update * self._attention_layer_scale()
        return self._apply_scalar_ffn(x + update)


class InvariantIrrepReadout(nn.Module):
    """Energy readout from scalar channels and explicit tensor norms."""

    def __init__(self, irreps, hidden_dim: int):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.hidden_dim = int(hidden_dim)
        self.non_scalar_irreps = o3.Irreps(self.irreps[1:])
        self.non_scalar_norm = o3.Norm(self.non_scalar_irreps, squared=True)
        input_dim = hidden_dim + self.non_scalar_norm.irreps_out.dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        invariants = torch.cat(
            (x[:, : self.hidden_dim], self.non_scalar_norm(x[:, self.hidden_dim :])),
            dim=-1,
        )
        return self.net(invariants)


class TransformersACEV3(nn.Module):
    """TRACE v3: fixed-environment tensorial cross-attention potential."""

    architecture_version = 3

    def __init__(
        self,
        r_max=5.0,
        l_max=2,
        num_radial=8,
        hidden_dim=128,
        num_layers=2,
        radial_basis_type: str = "bessel",
        radial_trainable: bool = False,
        envelope_exponent: int = 5,
        gaussian_width: float = 0.5,
        descriptor_passes: int = 1,
        descriptor_residual: bool = True,
        radial_mlp_hidden: int = 32,
        radial_mlp_layers: int = 2,
        correlation_order: int = 4,
        correlation_channels: int = 16,
        attention_num_heads: int | None = None,
        attention_key_dim: int | None = None,
        attention_ffn_hidden: int | None = None,
        attention_dropout: float = 0.0,
        attention_layer_scale_init: float | None = 1e-2,
        attention_distance_penalty: bool = True,
        transformer_num_heads: int = 4,
        transformer_ffn_hidden: int | None = None,
        transformer_dropout: float = 0.0,
        use_aux_force_head: bool = False,
        use_aux_stress_head: bool = False,
        interleave_descriptor: bool = False,
    ):
        super().__init__()
        if descriptor_passes != 1 or not descriptor_residual or interleave_descriptor:
            raise ValueError(
                "TRACE v3 uses one fixed ACE environment encoder; descriptor iteration "
                "and interleaving are not valid v3 options"
            )
        if envelope_exponent != 5:
            raise ValueError("TRACE v3 uses a fixed quintic C2 cutoff (envelope_exponent=5)")
        if use_aux_force_head or use_aux_stress_head:
            raise ValueError("TRACE v3 uses energy-derived forces and stress only")

        self.hidden_dim = int(hidden_dim)
        self.r_max = float(r_max)
        self.l_max = int(l_max)
        self.emb = nn.Embedding(118, hidden_dim)
        self.ace = ACEV3TokenDescriptor(
            r_max=r_max,
            l_max=l_max,
            num_radial=num_radial,
            hidden_dim=hidden_dim,
            correlation_order=correlation_order,
            correlation_channels=correlation_channels,
            radial_basis_type=radial_basis_type,
            radial_trainable=radial_trainable,
            gaussian_width=gaussian_width,
            radial_mlp_hidden=radial_mlp_hidden,
            radial_mlp_layers=radial_mlp_layers,
        )
        self.attention_irreps = self.ace.irreps_out
        attention_heads = max(1, int(attention_num_heads or transformer_num_heads))
        attention_hidden = attention_ffn_hidden or transformer_ffn_hidden
        if attention_dropout == 0.0 and transformer_dropout != 0.0:
            attention_dropout = transformer_dropout
        token_scalar_dim = self.ace.irreps_correlation[0].mul
        self.layers = nn.ModuleList(
            [
                TensorialFixedEnvironmentAttentionBlock(
                    node_irreps=self.attention_irreps,
                    token_irreps=self.ace.irreps_correlation,
                    hidden_dim=hidden_dim,
                    token_scalar_dim=token_scalar_dim,
                    r_max=r_max,
                    num_radial=num_radial,
                    num_token_kinds=self.ace.num_moment_tokens + 1,
                    num_heads=attention_heads,
                    key_dim=attention_key_dim,
                    ffn_hidden=attention_hidden,
                    dropout=float(attention_dropout),
                    layer_scale_init=attention_layer_scale_init,
                    radial_basis_type=radial_basis_type,
                    radial_trainable=radial_trainable,
                    gaussian_width=gaussian_width,
                    use_distance_penalty=attention_distance_penalty,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = InvariantIrrepReadout(self.attention_irreps, hidden_dim)

    def forward(
        self,
        data,
        training=False,
        temperature_scale: float = 1.0,
        detach_pos: bool = True,
        compute_stress: bool | None = None,
        return_electrostatics: bool = False,
    ):
        z, pos, edge_index = data["z"], data["pos"], data["edge_index"]
        cell_volume = data.get("volume", None)
        cell = data.get("cell", None)
        edge_shift = data.get("edge_shift", None)
        if compute_stress is None:
            compute_stress = training
        if return_electrostatics and getattr(self, "electrostatics", None) is None:
            raise ValueError(
                "return_electrostatics=True, but no electrostatics is attached "
                "to this model (trace_electrostatics.attach_electrostatics)."
            )

        if detach_pos:
            pos = pos.detach()
        pos.requires_grad_(True)
        if cell is not None:
            cell = cell.to(device=pos.device, dtype=pos.dtype)

        if compute_stress and cell_volume is not None:
            strain_params = torch.zeros(6, device=pos.device, dtype=pos.dtype, requires_grad=True)
            epsilon = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
            epsilon[0, 0] = strain_params[0]
            epsilon[1, 1] = strain_params[1]
            epsilon[2, 2] = strain_params[2]
            epsilon[0, 1] = epsilon[1, 0] = strain_params[3]
            epsilon[0, 2] = epsilon[2, 0] = strain_params[4]
            epsilon[1, 2] = epsilon[2, 1] = strain_params[5]
            deformation = torch.eye(3, device=pos.device, dtype=pos.dtype) + epsilon
            pos = pos @ deformation
            if cell is not None:
                cell = cell @ deformation
        else:
            strain_params = None
            epsilon = None

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        if edge_shift is not None and cell is not None:
            edge_vec = edge_vec + edge_shift.to(device=pos.device, dtype=pos.dtype) @ cell
        edge_len = torch.norm(edge_vec, dim=1)
        h, tokens, receiver, token_length, token_cutoff, token_kind = self.ace(
            self.emb(z), edge_index, edge_vec, edge_len
        )
        for layer in self.layers:
            h = layer(
                h,
                tokens,
                receiver,
                token_length,
                token_cutoff,
                token_kind,
                temperature_scale=temperature_scale,
            )
        E = torch.sum(self.readout(h))

        # Optional variational electrostatics (trace_electrostatics).  Absent
        # unless attach_electrostatics() has been called, so the plain TRACE
        # energy is unchanged by default.  It is added here, upstream of the
        # autograd calls below, so forces and stress follow from the existing
        # machinery: pos and cell are already the strained tensors at this
        # point.  Periodicity goes through _electrostatic_energy_term, which is
        # where SPEC 7 is enforced for all four hooks.  See
        # docs/electrostatics.md.
        aux = {}
        electrostatics = getattr(self, "electrostatics", None)
        if electrostatics is not None:
            term = _electrostatic_energy_term(
                electrostatics,
                h,
                pos,
                edge_index,
                edge_vec,
                edge_len,
                cell,
                edge_shift,
                data,
                return_output=return_electrostatics,
            )
            if return_electrostatics:
                term, output = term
                aux["electrostatics"] = [output]
            E = E + term

        # The observables in aux["electrostatics"] are derivatives of this same
        # graph (Born charges differentiate through it again), so it must
        # survive the force derivative when they are requested.
        keep_graph = training or return_electrostatics
        grads = torch.autograd.grad(
            E,
            pos,
            create_graph=training,
            retain_graph=keep_graph or epsilon is not None,
            allow_unused=True,
        )[0]
        F = -grads if grads is not None else torch.zeros_like(pos)

        S = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
        if compute_stress and epsilon is not None:
            g_eps = torch.autograd.grad(
                E,
                strain_params,
                create_graph=training,
                retain_graph=keep_graph,
                allow_unused=True,
            )[0]
            if g_eps is not None:
                stress = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
                stress[0, 0], stress[1, 1], stress[2, 2] = g_eps[:3]
                stress[0, 1] = stress[1, 0] = 0.5 * g_eps[3]
                stress[0, 2] = stress[2, 0] = 0.5 * g_eps[4]
                stress[1, 2] = stress[2, 1] = 0.5 * g_eps[5]
                volume = (
                    torch.det(cell).abs().clamp_min(1e-12)
                    if cell is not None
                    else cell_volume * torch.det(deformation)
                )
                S = stress / volume
        return E, F, S, aux


class TransformersACEV4(TransformersACEV3):
    """TRACE v4: connected local ACE tokens and shared-value cross-attention.

    TRACE v4 preserves v3's finite receptive field: each atom attends only to
    fixed descriptors of its original local environment.  It deliberately does
    not exchange hidden states between atoms and is therefore not a graph
    message-passing architecture.
    """

    architecture_version = 4

    def __init__(
        self,
        *args,
        attention_num_shells: int = 4,
        correlation_rank_initial: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ace = ACEV4CumulantTokenDescriptor(
            r_max=self.r_max,
            l_max=self.l_max,
            num_radial=kwargs.get("num_radial", 8),
            hidden_dim=self.hidden_dim,
            correlation_order=kwargs.get("correlation_order", 4),
            correlation_channels=kwargs.get("correlation_channels", 16),
            radial_basis_type=kwargs.get("radial_basis_type", "bessel"),
            radial_trainable=kwargs.get("radial_trainable", False),
            gaussian_width=kwargs.get("gaussian_width", 0.5),
            radial_mlp_hidden=kwargs.get("radial_mlp_hidden", 32),
            radial_mlp_layers=kwargs.get("radial_mlp_layers", 2),
        )
        self.attention_irreps = self.ace.irreps_out
        attention_heads = max(1, int(kwargs.get("attention_num_heads") or kwargs.get("transformer_num_heads", 4)))
        attention_hidden = kwargs.get("attention_ffn_hidden") or kwargs.get("transformer_ffn_hidden")
        attention_dropout = kwargs.get("attention_dropout", 0.0)
        if attention_dropout == 0.0 and kwargs.get("transformer_dropout", 0.0) != 0.0:
            attention_dropout = kwargs["transformer_dropout"]
        self.layers = nn.ModuleList(
            [
                CumulantMultiQueryTensorialAttentionBlock(
                    node_irreps=self.attention_irreps,
                    token_irreps=self.ace.irreps_correlation,
                    hidden_dim=self.hidden_dim,
                    token_scalar_dim=self.ace.irreps_correlation[0].mul,
                    r_max=self.r_max,
                    num_radial=kwargs.get("num_radial", 8),
                    num_token_kinds=self.ace.num_moment_tokens + 1,
                    num_heads=attention_heads,
                    key_dim=kwargs.get("attention_key_dim"),
                    ffn_hidden=attention_hidden,
                    dropout=float(attention_dropout),
                    layer_scale_init=kwargs.get("attention_layer_scale_init", 1e-2),
                    radial_basis_type=kwargs.get("radial_basis_type", "bessel"),
                    radial_trainable=kwargs.get("radial_trainable", False),
                    gaussian_width=kwargs.get("gaussian_width", 0.5),
                    use_distance_penalty=kwargs.get("attention_distance_penalty", True),
                    num_shells=attention_num_shells,
                )
                for _ in range(len(self.layers))
            ]
        )
        self.readout = InvariantIrrepReadout(self.attention_irreps, self.hidden_dim)
        if correlation_rank_initial is not None:
            self.ace.set_correlation_rank(correlation_rank_initial)

    def set_correlation_rank(self, rank: int) -> None:
        self.ace.set_correlation_rank(rank)


# The historical project name now points to v2 for newly created models. The
# calculator selects LegacyTransformersACE for checkpoints without a version.
FlashACE = TransformersACE
