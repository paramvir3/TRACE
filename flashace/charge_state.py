"""The charge state of a structure, read from ASE ``Atoms``, for the electrostatics.

One reader, shared by the training pipeline (``train.AtomisticDataset``) and the
ASE calculator, so that a structure means the same net charge, fragments and
spin state at inference as it did in training.  Units: charge in e.

Keys read, all optional:

``atoms.info['charge']`` or ``atoms.info['total_charge']``
    Net charge in e; 0 when absent.  Both may be given if they agree.
``atoms.arrays['fragment_id']``
    Integer fragment label per atom, ``0 .. F-1``, every label used.  The charge
    (and spin) constraints of trace_electrostatics are per fragment (SPEC 1.3).
``atoms.info['fragment_charges']``
    ``F`` net charges in e, one per fragment label; must sum to the net charge.
    Omitted with ``F > 1`` fragments it is resolved only when that is
    unambiguous (a neutral structure: all zero), otherwise refused.
``atoms.info['fragment_multiplicities']`` / ``atoms.info['multiplicity']``
    Spin multiplicity ``2S+1`` per fragment / of the whole structure.  Read only
    for spin-resolved heads, which need one for every structure.
    ``atoms.info['spin']`` is NOT read by default: datasets use it for S, for
    2S and for 2S+1, and a spin target off by a factor of two is silently wrong.
    It is read as 2S+1 only with ``spin_key='spin'`` -- the OMol25 / fairchem
    convention (singlet = 1), chosen explicitly (YAML ``electrostatics.spin_key``;
    stored in the checkpoint for the calculator).  The electron-count parity
    check then catches a dataset that stores S or 2S there instead.
"""

from __future__ import annotations

import math

import numpy as np
import torch


#: The ``atoms.info`` / ``atoms.arrays`` keys the charge state is read from.
CHARGE_STATE_INFO_KEYS = (
    'charge', 'total_charge', 'fragment_charges', 'multiplicity', 'fragment_multiplicities',
    'spin', 'external_field', 'charge_states', 'charge_state_coupling', 'charge_state_charges',
)
CHARGE_STATE_ARRAY_KEYS = ('fragment_id', 'metallic', 'electrode_potential', 'charge_state_fragment')

#: Largest |c - round(c)| accepted for a diabatic fragment charge (SPEC 21: integer).
INTEGER_CHARGE_TOLERANCE = 1e-9


def declared_charge_states(atoms):
    """The diabatic charge states of SPEC 21 declared on ``atoms``, or ``None``.

    Two equivalent forms; giving both is refused:

    * ``atoms.info['charge_states']`` -- a list of ``{'fragment_id': [n],
      'fragment_charges': [F]}``, one per state.  In memory only: an extxyz
      file cannot hold a list of dictionaries.
    * the FILE form, which survives extxyz: ``atoms.arrays
      ['charge_state_fragment']`` (integer label per atom, ``0 .. F-1``, every
      label used; one fragmentation shared by all states) and ``atoms.info
      ['charge_state_charges']`` (``S x F`` integer fragment charges, row-major,
      one row per state, ``S >= 2``).

    In the file form every state's fragment charges must be integers (the
    point of SPEC 21) and sum to the structure's net charge (all states
    describe one system).  Returned in the list form.
    """
    listed = atoms.info.get('charge_states')
    labels = atoms.arrays.get('charge_state_fragment')
    charges = atoms.info.get('charge_state_charges')
    if listed is not None and (labels is not None or charges is not None):
        raise ValueError(
            "charge states are declared twice: give either atoms.info['charge_states'] or "
            "atoms.arrays['charge_state_fragment'] with atoms.info['charge_state_charges']"
        )
    if listed is not None:
        return listed
    if labels is None and charges is None:
        return None
    if labels is None or charges is None:
        raise ValueError(
            "atoms.arrays['charge_state_fragment'] and atoms.info['charge_state_charges'] "
            "must be given together"
        )
    labels = np.asarray(labels).reshape(-1)
    if labels.shape != (len(atoms),) or not np.all(labels == np.round(labels)):
        raise ValueError(
            f"atoms.arrays['charge_state_fragment'] must hold one integer label per atom "
            f"({len(atoms)}), got shape {labels.shape}"
        )
    labels = labels.astype(int)
    n_fragments = int(labels.max()) + 1
    if labels.min() < 0 or sorted(set(labels.tolist())) != list(range(n_fragments)):
        raise ValueError(
            "atoms.arrays['charge_state_fragment'] must use every label 0 .. F-1, got "
            f"{sorted(set(labels.tolist()))}"
        )
    if n_fragments < 2:
        raise ValueError("charge states need at least two fragments to move charge between")
    charges = np.asarray(charges, dtype=float).reshape(-1)
    if charges.size % n_fragments or charges.size // n_fragments < 2:
        raise ValueError(
            f"atoms.info['charge_state_charges'] must hold S x F values with S >= 2 states and "
            f"F = {n_fragments} fragments, got {charges.size}"
        )
    charges = charges.reshape(-1, n_fragments)
    if not np.all(np.isfinite(charges)) or np.abs(charges - np.round(charges)).max() > INTEGER_CHARGE_TOLERANCE:
        raise ValueError(f"diabatic fragment charges must be integers, got {charges.tolist()}")
    total = 0.0
    for key in ('charge', 'total_charge'):
        if key in atoms.info:
            total = float(atoms.info[key])
            break
    sums = charges.sum(axis=1)
    if np.abs(sums - total).max() > INTEGER_CHARGE_TOLERANCE:
        raise ValueError(
            f"every charge state must sum to the net charge {total}; the states sum to {sums.tolist()}"
        )
    return [{'fragment_id': labels, 'fragment_charges': [float(c) for c in row]}
            for row in np.round(charges)]


def electrostatic_environment(atoms, dtype, device):
    """The optional electrostatic inputs beyond the charge state, as model data.

    ``atoms.info['external_field']`` ([3], V/A); ``atoms.arrays['metallic']``
    (bool per atom) and ``atoms.arrays['electrode_potential']`` (V per atom,
    NaN off-electrode) -- trace_electrostatics SPEC 20; the charge states of
    :func:`declared_charge_states` (``atoms.info['charge_states']``, or the
    extxyz form ``arrays['charge_state_fragment']`` + ``info
    ['charge_state_charges']``) with ``atoms.info['charge_state_coupling']`` (eV,
    or the string ``'through_bond'`` for the model's learned coupling) -- SPEC 21, 22.
    """
    data = {}
    field = atoms.info.get('external_field')
    if field is not None:
        data['external_field'] = torch.as_tensor(np.asarray(field, dtype=float), dtype=dtype, device=device).reshape(3)
    if 'metallic' in atoms.arrays:
        data['metallic'] = torch.as_tensor(np.asarray(atoms.arrays['metallic'], dtype=bool), device=device)
    if 'electrode_potential' in atoms.arrays:
        data['electrode_potential'] = torch.as_tensor(
            np.asarray(atoms.arrays['electrode_potential'], dtype=float), dtype=dtype, device=device
        )
    states = declared_charge_states(atoms)
    if states is not None:
        data['charge_states'] = [
            {
                'fragment_id': torch.as_tensor(np.asarray(s['fragment_id'], dtype=int), device=device),
                'fragment_charges': [float(c) for c in s['fragment_charges']],
            }
            for s in states
        ]
        coupling = atoms.info.get('charge_state_coupling', 0.0)
        # 'through_bond': the model's learned coupling head (SPEC 22); else eV.
        data['charge_state_coupling'] = (
            coupling if isinstance(coupling, str) and coupling == 'through_bond' else float(coupling)
        )
    return data


def charge_state_changed(old, new):
    """Whether two ``Atoms`` differ in any key the charge state is read from.

    ASE's calculator cache compares positions, numbers, cell, pbc and the
    initial charges/magmoms, but not ``info``: without this a structure whose
    ``info['charge']`` was changed would be handed the cached energy of the
    old charge state.
    """
    for key in CHARGE_STATE_INFO_KEYS:
        a, b = old.info.get(key), new.info.get(key)
        if (a is None) != (b is None) or (
            a is not None and not np.array_equal(np.asarray(a), np.asarray(b))
        ):
            return True
    for key in CHARGE_STATE_ARRAY_KEYS:
        a, b = old.arrays.get(key), new.arrays.get(key)
        if (a is None) != (b is None) or (a is not None and not np.array_equal(a, b)):
            return True
    return False


def _integral(values):
    values = np.asarray(values, dtype=float)
    return bool(np.all(np.isfinite(values)) and np.allclose(values, np.round(values), rtol=0.0, atol=1e-8))


#: atoms.info keys that may hold a structure's spin MULTIPLICITY 2S+1.
#: 'multiplicity' (default) is unambiguous; 'spin' is the OMol25 / fairchem
#: convention (``atoms.info['spin'] = 1`` for a singlet).  'spin' is read only
#: when chosen explicitly, because other datasets store S or 2S under it.
SPIN_KEYS = ('multiplicity', 'spin')


def charge_state_from_atoms(atoms, spin=False, spin_key='multiplicity'):
    """Periodicity, net charge, fragments and (optionally) spin of one structure.

    Returns a dict of plain Python/numpy values: ``pbc`` (3 bools),
    ``total_charge`` (float, e), ``fragment_id`` and ``fragment_charges``
    (arrays, or both ``None`` when no labels were given), and
    ``fragment_spins``.  The last holds spin QUANTUM NUMBERS
    ``S_F = (m_F - 1) / 2``, because that is what trace_electrostatics' spin
    constraint takes (``sum_i s_i = 2 S_F``, verified), and it is ``None``
    unless ``spin`` is true.  ``multiplicity`` is the structure's 2S+1 when
    ``atoms.info[spin_key]`` carries it (read whatever ``spin`` is, so a
    closed-shell model can report open-shell data), else ``None``.  Raises
    ``ValueError`` on anything inconsistent.

    ``spin_key`` names the ``atoms.info`` key holding 2S+1 (``SPIN_KEYS``):
    ``'multiplicity'``, or ``'spin'`` for OMol25-style data.  The parity check
    below (2S must have the parity of the electron count) is the guard against
    a dataset that stores S or 2S under the chosen key.
    """
    if spin_key not in SPIN_KEYS:
        raise ValueError(f"spin_key must be one of {SPIN_KEYS}, got {spin_key!r}")
    n_atoms = len(atoms)
    pbc = np.asarray(atoms.pbc, dtype=bool).reshape(-1)
    if pbc.size == 1:
        pbc = np.repeat(pbc, 3)

    charges = [float(atoms.info[key]) for key in ('charge', 'total_charge') if key in atoms.info]
    if len(charges) == 2 and not np.isclose(charges[0], charges[1], rtol=0.0, atol=1e-8):
        raise ValueError(
            f"atoms.info['charge'] = {charges[0]} and atoms.info['total_charge'] = "
            f"{charges[1]} disagree"
        )
    total_charge = charges[0] if charges else None
    if total_charge is not None and not math.isfinite(total_charge):
        raise ValueError(f"net charge must be finite, got {total_charge}")

    fragment_id = atoms.arrays.get('fragment_id')
    fragment_charges = atoms.info.get('fragment_charges')
    n_fragments = 1
    if fragment_id is not None:
        labels = np.asarray(fragment_id).reshape(-1)
        if labels.shape != (n_atoms,) or not _integral(labels):
            raise ValueError("atoms.arrays['fragment_id'] must hold one integer label per atom")
        labels = np.round(labels).astype(np.int64)
        n_fragments = int(labels.max()) + 1
        if labels.min() != 0 or np.unique(labels).size != n_fragments:
            raise ValueError(
                "atoms.arrays['fragment_id'] labels must be 0 .. F-1 with every "
                f"label used, got {sorted(np.unique(labels).tolist())}"
            )
        fragment_id = labels
    if fragment_charges is not None:
        fragment_charges = np.asarray(fragment_charges, dtype=float).reshape(-1)
        if fragment_charges.size != n_fragments:
            raise ValueError(
                f"atoms.info['fragment_charges'] has {fragment_charges.size} entries for "
                f"{n_fragments} fragment(s); multiple fragments need atoms.arrays['fragment_id']"
            )
        if total_charge is not None and not np.isclose(
            fragment_charges.sum(), total_charge, rtol=0.0, atol=1e-8
        ):
            raise ValueError(
                f"atoms.info['fragment_charges'] sums to {fragment_charges.sum()} but the "
                f"net charge is {total_charge}"
            )
        total_charge = float(fragment_charges.sum())
    total_charge = 0.0 if total_charge is None else total_charge
    if fragment_id is not None and fragment_charges is None:
        if n_fragments == 1:
            fragment_charges = np.array([total_charge])
        elif total_charge == 0.0:
            fragment_charges = np.zeros(n_fragments)
        else:
            raise ValueError(
                f"net charge {total_charge} on {n_fragments} fragments but no "
                "atoms.info['fragment_charges']: the charge constraint is per fragment "
                "(SPEC 1.3) and there is no size-consistent way to guess which fragment "
                "carries it"
            )
    if fragment_id is None:
        # One fragment spec without labels is just the net charge.
        fragment_charges = None

    fragment_spins = None
    if spin:
        multiplicities = atoms.info.get('fragment_multiplicities')
        if multiplicities is None and spin_key in atoms.info:
            if n_fragments > 1:
                raise ValueError(
                    f"atoms.info['{spin_key}'] describes the whole structure, but it has "
                    f"{n_fragments} fragments; give atoms.info['fragment_multiplicities']"
                )
            multiplicities = [atoms.info[spin_key]]
        if multiplicities is None:
            raise ValueError(
                "spin-resolved heads (electrostatics.heads.spin: true) need the spin "
                f"state of every structure: atoms.info['{spin_key}'] (2S+1) or "
                "atoms.info['fragment_multiplicities'] (one 2S_F+1 per fragment). "
                "For OMol25-style data, whose atoms.info['spin'] is 2S+1, set "
                "spin_key='spin'; it is not read by default because other datasets "
                "store S or 2S under 'spin'."
            )
        multiplicities = np.asarray(multiplicities, dtype=float).reshape(-1)
        if multiplicities.size != n_fragments:
            raise ValueError(
                f"{multiplicities.size} multiplicities for {n_fragments} fragment(s)"
            )
        if not _integral(multiplicities) or np.any(multiplicities < 1):
            raise ValueError(f"multiplicities must be integers >= 1, got {multiplicities.tolist()}")
        # 2S_F = m_F - 1 unpaired electrons must have the parity of the
        # fragment's electron count: an even-electron fragment cannot be a
        # doublet.  Checked where the charges make the count an integer.
        per_fragment_charge = (
            fragment_charges if fragment_charges is not None else np.array([total_charge])
        )
        labels = fragment_id if fragment_id is not None else np.zeros(n_atoms, dtype=np.int64)
        nuclear = np.bincount(labels, weights=atoms.numbers, minlength=n_fragments)
        electrons = nuclear - per_fragment_charge
        if _integral(electrons) and np.any(
            (np.round(electrons).astype(np.int64) - np.round(multiplicities - 1).astype(np.int64)) % 2
        ):
            raise ValueError(
                f"multiplicities {multiplicities.tolist()} are inconsistent with the "
                f"electron counts {np.round(electrons).astype(int).tolist()} of the "
                "fragments: 2S = multiplicity - 1 must have the parity of the electron count"
            )
        fragment_spins = (multiplicities - 1.0) / 2.0

    multiplicity = atoms.info.get(spin_key)
    if multiplicity is not None:
        multiplicity = float(np.asarray(multiplicity, dtype=float).reshape(-1)[0])
    return {
        'pbc': pbc,
        'multiplicity': multiplicity,
        'total_charge': float(total_charge),
        'fragment_id': fragment_id,
        'fragment_charges': fragment_charges,
        'fragment_spins': fragment_spins,
    }


def charge_state_tensors(state, dtype, device=None):
    """The model-input entries of one structure's charge state.

    Exactly the keys ``ElectrostaticModule.forward`` reads from the data
    dictionary: ``pbc`` (``[3]`` bool, SPEC 7), ``total_charge`` as a Python
    float (the package refuses a traced net charge), and ``fragment_id`` /
    ``fragment_charges`` / ``fragment_spins`` only when the structure has them.
    """
    entries = {
        'pbc': torch.tensor(state['pbc'], dtype=torch.bool, device=device),
        'total_charge': float(state['total_charge']),
    }
    if state['fragment_id'] is not None:
        entries['fragment_id'] = torch.tensor(state['fragment_id'], dtype=torch.long, device=device)
        entries['fragment_charges'] = torch.tensor(state['fragment_charges'], dtype=dtype, device=device)
    if state['fragment_spins'] is not None:
        entries['fragment_spins'] = torch.tensor(state['fragment_spins'], dtype=dtype, device=device)
    return entries
