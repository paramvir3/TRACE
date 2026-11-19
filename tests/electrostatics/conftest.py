"""Shared fixtures for the ``trace_electrostatics`` regression suite.

SPEC section 3 requires ``torch.set_default_dtype(torch.float64)`` *before* any
e3nn module is constructed: e3nn caches generated ``TensorProduct`` code under
the dtype in force at first construction, and a later ``.double()`` is a silent
no-op (SPEC 3 records 2.54e-08 against 1.94e-15).  pytest imports ``conftest.py``
before it imports any test module, so the assignment on the third executable
line below is the earliest hook the suite has.  Nothing above it may import
e3nn, directly or transitively.

This suite is deliberately written against SPEC.md, not against the
implementation.  Every expected value is either quoted from SPEC.md (with the
section number in a comment) or derived independently in ``elec_reference.py``.
"""

import torch

torch.set_default_dtype(torch.float64)  # SPEC 3 -- BEFORE the e3nn import below

import itertools  # noqa: E402
import math  # noqa: E402

import pytest  # noqa: E402


# --------------------------------------------------------------------------
# Symbol lookup.
#
# An earlier draft of this file resolved package symbols through helpers that
# called ``pytest.skip`` when a module or attribute was missing, because the
# package was being written concurrently with the suite.  The package is
# complete, so that behaviour is now a liability: a deleted or renamed entry
# point would turn a regression into a green SKIP.  These helpers therefore
# FAIL, never skip.  The only thing they still do is name the missing symbol.
# --------------------------------------------------------------------------
def require(module_name, *attributes):
    """Return the named attributes of ``trace_electrostatics.<module_name>``.

    Fails the calling test if the module or any named attribute is missing.
    """
    module = __import__(
        f"trace_electrostatics.{module_name}", fromlist=["_"]
    )
    resolved = []
    for attribute in attributes:
        if not hasattr(module, attribute):
            raise AssertionError(
                f"trace_electrostatics.{module_name}.{attribute} is missing "
                f"(module has: "
                f"{sorted(n for n in vars(module) if not n.startswith('_'))})"
            )
        resolved.append(getattr(module, attribute))
    if not resolved:
        return module
    return resolved[0] if len(resolved) == 1 else tuple(resolved)


def require_any(module_name, *candidates):
    """Return the first attribute of a module that exists, by name.

    Used where SPEC.md fixes the *behaviour* but not the spelling of the entry
    point.  Fails with the full candidate list if none is present, so a failing
    lookup names what the suite was looking for.
    """
    module = __import__(f"trace_electrostatics.{module_name}", fromlist=["_"])
    for name in candidates:
        if hasattr(module, name):
            return getattr(module, name)
    raise AssertionError(
        f"none of {candidates} found in trace_electrostatics.{module_name} "
        f"(has: {sorted(n for n in vars(module) if not n.startswith('_'))})"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running regression test")
    # e3nn's TorchScript scaffolding emits this on every module construction
    # (~260 times per run); it is third-party noise that buried real warnings.
    # Only this exact message is filtered.
    config.addinivalue_line(
        "filterwarnings",
        "ignore:The TorchScript type system doesn't support instance-level "
        "annotations on empty non-base types:UserWarning",
    )


# --------------------------------------------------------------------------
# Geometry helpers shared by several test modules.
# --------------------------------------------------------------------------
def full_neighbour_list(pos, cell, cutoff):
    """Every ordered periodic pair (i, j, n) with 0 < |r| <= cutoff.

    Returns ``(edge_index [2, n_edges], edge_shift [n_edges, 3])`` with the
    displacement convention of ``flashace.model`` and of
    ``trace_electrostatics.ewald.ewald_real``::

        r_vec = pos[edge_index[0]] - pos[edge_index[1]] + edge_shift @ cell

    Self-image pairs (i == j with a non-zero shift) are included; they are
    genuine members of the real-space Ewald sum.
    """
    n_atoms = pos.shape[0]
    widths = []
    for axis in range(3):
        other = [cell[k] for k in range(3) if k != axis]
        normal = torch.linalg.cross(other[0], other[1])
        normal = normal / normal.norm()
        widths.append(abs(float(cell[axis] @ normal)))
    repeats = [int(math.ceil(cutoff / w)) for w in widths]

    senders, receivers, shifts = [], [], []
    images = itertools.product(
        range(-repeats[0], repeats[0] + 1),
        range(-repeats[1], repeats[1] + 1),
        range(-repeats[2], repeats[2] + 1),
    )
    for image in images:
        offset = torch.tensor(image, dtype=pos.dtype) @ cell
        for i in range(n_atoms):
            for j in range(n_atoms):
                distance = float((pos[i] - pos[j] + offset).norm())
                if 1.0e-10 < distance <= cutoff:
                    senders.append(i)
                    receivers.append(j)
                    shifts.append(image)
    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    edge_shift = torch.tensor(shifts, dtype=pos.dtype)
    return edge_index, edge_shift


def complete_edge_index(n_atoms):
    """Every ordered pair (i, j), i != j -- an open-boundary "all neighbours" list."""
    pairs = [(i, j) for i in range(n_atoms) for j in range(n_atoms) if i != j]
    return torch.tensor(
        [[i for i, _ in pairs], [j for _, j in pairs]], dtype=torch.long
    )


def isotropic_features(n_atoms, seed, scale=2.0):
    """Random TRACE node features for ``ElectrostaticHeads`` default irreps.

    Returns ``(heads, h)``.  Constructed here rather than in each test so that
    every test shares one definition of "a plausible feature vector", and so
    that the e3nn modules are only ever built after conftest has set float64.
    """
    ElectrostaticHeads = require("heads", "ElectrostaticHeads")
    torch.manual_seed(seed)
    heads = ElectrostaticHeads()
    return heads, torch.randn(n_atoms, heads.irreps_in.dim) * scale


# --------------------------------------------------------------------------
# Fixtures.
# --------------------------------------------------------------------------
@pytest.fixture
def water_molecule():
    """A single water molecule -- experimental geometry, r(OH) = 0.9572 A,
    angle(HOH) = 104.52 deg.

    The one fixture with a real chemical geometry and a real C2v symmetry, so
    it is the test bed for anything whose answer symmetry constrains (the two
    hydrogens must come out with identical charges).
    """
    angle = math.radians(104.52) / 2.0
    r = 0.9572
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [r * math.sin(angle), 0.0, r * math.cos(angle)],
            [-r * math.sin(angle), 0.0, r * math.cos(angle)],
        ]
    )
    species = torch.tensor([8, 1, 1], dtype=torch.long)
    bond_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    return {"pos": positions, "z": species, "bond_index": bond_index}


@pytest.fixture
def small_molecule():
    """Five atoms in general position, no symmetry, chemical bond lengths.

    The default test bed for anything that needs a "small molecule": general
    enough that an accidental symmetry cannot make a broken tensor look right,
    small enough for dense reference linear algebra.
    """
    positions = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [1.43, 0.11, -0.22],
            [2.31, 1.30, 0.41],
            [-0.72, 1.09, 0.63],
            [1.90, -0.95, 1.44],
        ]
    )
    bond_index = torch.tensor([[0, 1, 0, 1], [1, 2, 3, 4]], dtype=torch.long)
    return {"pos": positions, "bond_index": bond_index}


@pytest.fixture
def charged_system():
    """A +1 system: five atoms, a bonded tree, and a single fragment at Q = +1.

    SPEC 1.3 puts the net charge in ``rho``, which is a FIXED constraint carrier,
    so the interesting behaviour of a charged system is only visible when there
    genuinely is a net charge to carry.  ``total_charge`` is a Python float
    because SPEC (energy.electrostatic_energy) treats the net charge as a static
    property of the system rather than a traced quantity.
    """
    positions = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [1.52, 0.00, 0.00],
            [2.10, 1.41, 0.00],
            [-0.63, 1.17, 0.35],
            [3.55, 1.62, 0.41],
        ]
    )
    bond_index = torch.tensor([[0, 1, 0, 2], [1, 2, 3, 4]], dtype=torch.long)
    return {
        "pos": positions,
        "bond_index": bond_index,
        "total_charge": 1.0,
        "chi": torch.tensor([-4.7, -3.1, -5.9, -2.2, -6.4]),
        "eta": torch.tensor([12.0, 13.5, 11.5, 14.0, 12.7]),
        "sigma": torch.full((5,), 1.5),
    }


@pytest.fixture
def close_contact_pair():
    """Two atoms at 1.2 A -- the separation at which SPEC section 6 says a
    Thole asymmetry bug becomes visible (14.9%) instead of invisible (0.005%)."""
    return torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])


@pytest.fixture
def contrasted_close_contact():
    """Three atoms in general position with strongly contrasted polarizability
    volumes, the closest pair at ~1.19 A.

    SPEC section 6: Maxwell reciprocity must be tested on a *close-contact,
    polarizability-contrasted* system.  A sparse system hides the bug.
    """
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.372, 0.996, 0.528]]
    )
    polarizability_volume = torch.tensor([0.30, 1.60, 11.0])  # A^3, ~H / ~C / ~I
    return {"pos": positions, "alpha_volume": polarizability_volume}


@pytest.fixture
def sparse_triple(contrasted_close_contact):
    """The same three atoms and polarizabilities, scaled out to ~8 A separation.

    This is the *invalid* test bed of SPEC section 6, kept as a fixture so the
    suite can demonstrate that it is invalid rather than assert it away.
    """
    scale = 8.0 / 1.2
    return {
        "pos": contrasted_close_contact["pos"] * scale,
        "alpha_volume": contrasted_close_contact["alpha_volume"],
    }


@pytest.fixture
def linear_chain():
    """Factory for an evenly spaced chain along z, default spacing 1 A.

    The spacing-1 chain is the geometry of the SPEC section 6 identity
    ``alpha = N (N^2 - 1) / 12`` at eta = 1, J = 0.
    """

    def build(n_atoms, spacing=1.0):
        z = torch.arange(n_atoms, dtype=torch.get_default_dtype()) * spacing
        positions = torch.zeros(n_atoms, 3)
        positions[:, 2] = z
        bonds = torch.tensor(
            [list(range(n_atoms - 1)), list(range(1, n_atoms))], dtype=torch.long
        )
        return {"pos": positions, "bond_index": bonds}

    return build


@pytest.fixture
def periodic_cell():
    """A simple cubic 6 A cell with four atoms inside, for periodic tests.

    Small enough that a full neighbour list out to 5 A is cheap, large enough
    that no atom sits on top of its own periodic image.
    """
    cell = torch.eye(3) * 6.0
    positions = torch.tensor(
        [
            [0.30, 0.40, 0.50],
            [1.70, 0.60, 0.90],
            [2.90, 2.30, 1.10],
            [4.20, 3.80, 4.40],
        ]
    )
    return {"cell": cell, "pos": positions}


@pytest.fixture
def triclinic_cell():
    """A non-orthogonal cell with no right angles, for periodic tests that must
    not pass only because the cell happened to be cubic."""
    cell = torch.tensor([[6.0, 0.0, 0.0], [0.8, 5.5, 0.0], [-0.4, 0.6, 6.7]])
    positions = torch.tensor(
        [
            [0.51, 0.83, 1.20],
            [2.90, 1.44, 3.10],
            [4.20, 3.95, 0.70],
            [1.15, 4.60, 5.05],
        ]
    )
    return {"cell": cell, "pos": positions}


@pytest.fixture
def nacl_crystal():
    """Rock-salt NaCl: conventional cubic cell, a = 5.64 A, 4 Na(+1) + 4 Cl(-1).

    Derived here rather than taken from the implementation.  The nearest
    neighbour distance is r0 = a / 2, and the Madelung constant follows from the
    total cell energy as

        M = -E_cell * r0 / (n_pairs * K_E),   n_pairs = 4

    because the Madelung energy of one ion pair is -M K_E q^2 / r0 and the cell
    holds four formula units.
    """
    a = 5.64
    cell = torch.eye(3) * a
    fractional = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.5],
        ]
    )
    charges = torch.tensor([1.0] * 4 + [-1.0] * 4)
    return {
        "cell": cell,
        "pos": fractional @ cell,
        "q": charges,
        "lattice_constant": a,
        "nearest_neighbour": a / 2.0,
        "n_pairs": 4,
    }


@pytest.fixture
def nacl_primitive():
    """Rock-salt NaCl in its PRIMITIVE rhombohedral cell -- a non-orthogonal cell.

    The FCC primitive vectors are a1 = (0, a/2, a/2), a2 = (a/2, 0, a/2),
    a3 = (a/2, a/2, 0); the basis is Na at the origin and Cl at (a/2, 0, 0).
    One formula unit per cell, volume a^3/4, nearest-neighbour distance a/2.

    Same crystal, same Madelung constant, but every cell angle is 60 degrees, so
    a reciprocal-space sum that quietly assumed an orthogonal cell (or that
    inverted the cell matrix on the wrong side) cannot pass this and the cubic
    test at the same time.
    """
    a = 5.64
    cell = torch.tensor(
        [[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]]
    )
    positions = torch.tensor([[0.0, 0.0, 0.0], [a / 2, 0.0, 0.0]])
    return {
        "cell": cell,
        "pos": positions,
        "q": torch.tensor([1.0, -1.0]),
        "lattice_constant": a,
        "nearest_neighbour": a / 2.0,
        "n_pairs": 1,
    }
