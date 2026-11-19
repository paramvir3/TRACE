"""Assembly and evaluation of the variational electrostatic energy.

This module is the place where the readout heads, the bond topology, the damped
kernels and the solver are combined into one differentiable scalar, the
constrained minimum (SPEC 17)

    U_elec = min_m 0.5 m^T K m + chi^T m ,      m = ( {t_ij}, {mu_hat_i}, [{Theta_hat_i}] )

with (SPEC 1.5)

    q      = B t + rho                          atomic charges, e
    K      = [ B^T H_qq B + diag(kappa)   B^T T01        ]
             [        .                   A_mu + T_Thole ]
    H_qq   = J(r_ij) off-diagonal + diag(eta) on-diagonal

and ``mu_hat_i = mu_i / (e ell_i)`` the reduced dipole that makes the mixed-rank
blocks of ``K`` dimensionally meaningful.

``rho`` is **not** part of ``m``.  SPEC 1.3/1.5 make it a *fixed constraint
carrier*: it holds the per-fragment net charge and is never varied.  Letting it
relax destroys the model -- with both ``t`` and ``rho`` free under a per-fragment
constraint the minimiser gives ``|t| = 1.65e-07`` against ``3.57e-01`` with
``rho`` pinned, because a per-atom charge channel that bypasses the bond-hardness
penalty absorbs all the charge.  The packing helpers below
(:meth:`ElectrostaticAssembly.flatten_moments`, :attr:`~ElectrostaticAssembly.moment_dim`)
nevertheless lay out ``(t, rho, mu_hat)`` as one ``n_bonds + 4 n_atoms`` vector,
because :meth:`~ElectrostaticAssembly.energy` is a function of all three and the
dense operator, the constraint operator and the dipole derivative are most
readable in that common layout.  That layout is a *storage convention*, not a
statement about which coordinates are free: everything that takes a derivative
with respect to the moments -- the solver, and
:func:`trace_electrostatics.observables.polarizability` -- must first drop the
``rho`` rows and columns.  Confusing the two is what made the reported
polarizability disagree with the finite-field one; see ``_field_response``.

Why this is a free function and not a method on the potential
-------------------------------------------------------------
``trace_electrostatics`` must never import ``flashace``: the dependency runs one
way, so that the electrostatics can be tested, audited and reused without the
potential, and so that a checkpoint trained without electrostatics keeps loading
unchanged.  Integration is therefore a three-line diff inside
``TransformersACEV3.forward`` that adds ``U_elec`` to ``E`` *before* the existing
``torch.autograd.grad`` call on ``pos`` and ``strain_params``.  See
``docs/electrostatics.md``.  Forces and stress then follow with no further
changes, which is the reason everything in this module is differentiable with
respect to both ``pos`` and ``cell``.

Differentiability contract
--------------------------
Nothing here detaches, rounds, or synchronises.  In particular:

* the default solve is CONVERGED (``solve_rtol``) and differentiated
  IMPLICITLY (SPEC 17, :mod:`trace_electrostatics.implicit`): the energy node's
  backward is the envelope theorem and the moments' backward an adjoint solve,
  both exact to every order -- conservative forces (closed-loop work at the
  quadrature floor) and exact responses.  The legacy ``solve_mode="fixed_k"``
  is k CG iterations differentiated through (SPEC 2);
* every shape is a function of ``n_atoms``, ``n_bonds`` and ``n_fragments``
  only, so the graph is static for ahead-of-time compilation.  The Python
  branches that decide shape are on structure known at trace time (periodic or
  not, how many fragments) and never on a traced value.

Host synchronisations -- what is actually true, measured
--------------------------------------------------------
An earlier version of this paragraph claimed "no ``.item()``, no ``nonzero`` and
no branch on a tensor value".  That was never true of a default call.  Re-counted
by instrumenting ``Tensor.__bool__``, ``Tensor.item`` and ``torch.equal`` over
one open-boundary ``electrostatic_state`` evaluation (4 atoms, default config):
**17** host-synchronising operations -- **1** ``.item()``, **12**
``bool(tensor)`` and **4** ``torch.equal``.  (An earlier count said 11
``bool(tensor)`` and did not count ``torch.equal`` at all; it predates the
``FragmentConstraints`` partition check and undercounted by five in total.)
Of those, this module owns exactly two, and both are guards rather than hot-path
arithmetic:

* ``_resolve_fragments`` calls ``.item()`` on ``assigned.max()`` -- and only on
  the branch where the caller did *not* supply ``fragment_id``.  Deriving the
  fragment count is inherently a reduction to the host, which is precisely why
  the docstring of :func:`assemble` tells deployment callers to pass
  ``fragment_id`` (and ``fragment_charges``) from the data pipeline.  Supply
  both and this call disappears;
* ``_derive_bond_slot`` calls ``bool()`` once, to raise when a bond has no edge
  to gather its geometry from.  A structural error that must fail loudly.

The remaining fifteen come from the sibling modules and are their contract, not
this one's: ``diagnostics.assert_hardness_floor`` (4 -- two per call, and it is
called twice, once from ``ElectrostaticParams.__post_init__`` and once from
:func:`assemble` under ``config.check_hardness_floor``),
``diagnostics.assert_positive_kappa`` (1, also from
``ElectrostaticParams.__post_init__``), ``diagnostics.assert_fragment_charges``
(1, unconditional inside ``solve_charges``),
``solver.FragmentConstraints.__post_init__`` (1, the partition check),
``solver._check_seed_binding`` (1 ``torch.equal``, the SPEC 2 warm-start
barrier), ``topology.fragment_constraint_matrix`` (1),
``topology.distribute_fragment_charges`` (2), ``topology.map_edges_to_bonds``
(1), ``topology.build_bond_topology`` (1 ``torch.equal``) and
``topology._connected_components`` (2 ``torch.equal``, the fixed-point test of
the label-propagation loop).  Only the second ``assert_hardness_floor`` is
switchable from :class:`~trace_electrostatics.config.ElectrostaticConfig`; the
rest are unconditional in the modules that own them, and they are the reason a
genuinely synchronisation-free AOT trace needs the topology precomputed *and*
those guards lifted out of the traced region.

The four ``diagnostics`` assertions are the ones that already carry the gate:
``_needs_host_report`` reads the flag on the host in eager mode and queues
``torch._assert_async`` while tracing, so under ``torch.compile`` they cost
**zero** synchronisations (measured: 3 ``bool(tensor)`` calls in
``ElectrostaticParams.__post_init__`` in eager, 0 with ``_is_tracing()`` true).
The ``topology`` and ``solver`` guards above have no such gate and are what
``fullgraph=True`` hits first -- ``solver._check_seed_binding``'s ``torch.equal``
(reported as "data dependent operator: aten.equal.default"), then
``topology.distribute_fragment_charges``'s ``bool((weights < 0).any())``.

Open boundaries versus periodic boundaries
------------------------------------------
With open boundaries the Coulomb sum is an explicit O(N^2) double sum over *all*
pairs, not over the neighbour list: the electrostatic energy is long-ranged, and
truncating it at the TRACE cutoff is exactly the error this module exists to
remove.  Under periodic boundaries the same physics routes through
:func:`ewald_multipole_energy`.  The bonded term ``0.5 kappa t^2`` is
short-ranged in both cases and always runs over the neighbour list.

Which of the two runs is decided by ``pbc``, **never by the cell** (SPEC 7)
------------------------------------------------------------------------------
ASE hands a gas-phase frame ``cell = zeros(3, 3)``, which is not ``None``.  The
old rule -- "periodic iff ``cell is not None``" -- therefore routed a singular
lattice into Ewald, where the reciprocal-lattice construction inverts the cell
and ``torch.linalg.inv`` raises ``The diagonal element 1 is zero``.  Verified
live on a four-atom frame before this was changed.  Periodicity is now an
*input*: a ``pbc`` triple in ASE's convention, resolved by
:func:`_normalise_pbc`, which also decides the three regimes of SPEC 7 --
``(F,F,F)`` open, ``(T,T,T)`` 3-D Ewald, ``(T,T,F)`` and permutations the 2-D
slab route, ``(T,F,F)`` refused.  The cell is *data* for whichever route ``pbc``
selects, and under ``(F,F,F)`` it is dropped outright so that "ignored" is
structural rather than a promise in a comment.

When ``pbc`` is absent the boundary condition is inferred, and the inference is
written so that the ASE frame lands on the right side of it: no cell, or an
all-zero cell, means **open**; any other cell means ``(T,T,T)``.  A cell that is
non-zero but singular is a caller error and raises, as does an all-zero cell
with ``pbc=(T,T,T)`` explicitly requested -- neither is papered over, because
both are a request for a lattice that has no reciprocal.

**Both periodic routes are live.**  Reporting the Ewald sum is only half the
job: the solver has to *minimise* it too, and ``solver.assemble_system`` builds
its own operator from raw unwrapped coordinates with no cell whenever it is not
handed one -- measured 0.513 eV apart from the reported energy on four atoms in
an 8 A cell.  :func:`_solver_geometry` therefore hands the solver all three
lattice-summed pair blocks, and :func:`_periodic_pair_blocks` builds them from
``ewald``'s own geometry so the two cannot drift.  Open boundaries are
unaffected and the two assemblies agree to 8.9e-16 eV.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, NamedTuple, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .config import ElectrostaticConfig
from .constants import to_polarizability_volume
from .diagnostics import assert_hardness_floor, assert_positive_definite_at_solution
from .constants import K_E
from .ewald import (
    TWO_PI,
    EwaldParams,
    ewald_jellium,
    ewald_multipole_energy,
    ewald_self,
    ewald_self_dipole,
)
# Private helpers, imported on purpose.  The periodic pair blocks have to be
# built from the *same* image box, minimum-image recentring, screened kernels
# and reciprocal grid as the lattice sum they reproduce; reimplementing any of
# them here is precisely the drift this import exists to prevent.  If one of
# these moves, the periodic operator must move with it -- which is the intended
# coupling, and the reason `_periodic_pair_blocks` says it belongs in `ewald`.
from .ewald import (
    _gaussian_weights as _ewald_gaussian_weights,
    _pair_geometry as _ewald_pair_geometry,
    _screened_kernels as _ewald_screened_kernels,
    _wave_vectors as _ewald_wave_vectors,
)
from .dispersion import dispersion_energy
from .lattice import gaussian_lattice_blocks
from .lattice_operator import LatticeOperator, balanced_real_cutoff, lattice_pairs, solve_lattice
from .open_operator import OpenOperator
from .metals import METAL_BOND_HARDNESS, metal_bonds
from .heads import ElectrostaticHeads
from .kernels import (
    DEFAULT_GAUSSIAN_A,
    DEGENERATE_SEPARATION,
    gaussian_multipole_width,
    charge_dipole_tensor,
    combine_sigma,
    coulomb_kernel,
    dipole_interaction_tensor,
    gram_self_energy,
    kernel_self,
    thole_damping,
)
from .solver import (
    MIN_POLARIZABILITY_VOLUME as _SOLVER_MIN_POLARIZABILITY_VOLUME,
    ElectrostaticParams,
    FragmentConstraints,
    Geometry,
    GeometrySeed,
    SolveResult,
    SolverOptions,
    assemble_system,
    gaussian_dipole_damping,
    gaussian_widths,
    quadrupole_pair_tensors,
    solve_charges,
)
from .topology import (
    assign_fragments,
    build_bond_topology,
    build_incidence,
    fragment_constraint_matrix,
    map_edges_to_bonds,
)

__all__ = [
    "ELECTROSTATICS_ATTR",
    "PBC_BULK",
    "PBC_OPEN",
    "ElectrostaticAssembly",
    "ElectrostaticModule",
    "ElectrostaticState",
    "assemble",
    "attach_electrostatics",
    "detach_electrostatics",
    "electrostatic_energy",
    "solver_seed_loss",
    "electrostatic_state",
    "resolve_pbc",
]


#: Attribute name the potential looks for.  Absent or ``None`` means off, which
#: is what every existing checkpoint and every existing test sees.
ELECTROSTATICS_ATTR = "electrostatics"

#: Floor on the isotropic polarizability volume fed to the Thole damping, in
#: A^3.  ``thole_damping`` divides by ``(alpha_i alpha_j)^(1/6)``, so an
#: untrained head that emits a near-singular dipole stiffness would produce an
#: infinity here rather than merely a bad number.  A NaN shield, not a physical
#: regularisation.
#:
#: **Re-exported from** :mod:`trace_electrostatics.solver`, deliberately, and not
#: given a value of its own.  This module and ``solver.assemble_system`` build
#: the *same* Thole damping from two independently assembled operators, so two
#: independent floors are two subtly different operators: with 1e-6 here and
#: 1e-12 there, any atom whose polarizability volume fell in between was damped
#: with one width in the reported energy and another in the functional the solver
#: minimised.  One definition, in the module that owns the solve.
MIN_POLARIZABILITY_VOLUME = _SOLVER_MIN_POLARIZABILITY_VOLUME

#: Open boundaries, ASE's spelling.  A supplied cell is ignored (SPEC 7).
PBC_OPEN = (False, False, False)

#: Fully periodic, 3-D Ewald under tinfoil boundary conditions (SPEC 7).
PBC_BULK = (True, True, True)

#: Relative tolerance on ``|det(cell)|`` below which a lattice counts as
#: singular.  Compared against the cube of the largest cell entry, so it is a
#: statement about the *shape* of the cell and not about the units it is in: the
#: all-zero ASE gas-phase frame, two parallel lattice vectors and a flattened
#: cell all fail it, while a legitimately thin slab cell does not.
_SINGULAR_CELL_RTOL = 1.0e-12

#: Relative tolerance for deciding that a cycle in the bond graph carries a
#: genuine lattice translation, and for deciding that the non-periodic axis of a
#: slab is orthogonal to the periodic plane.  Both are comparisons of a length
#: against the cell, so both are scaled by the largest cell entry.
_LATTICE_RTOL = 1.0e-10

#: Raised for ``(T,F,F)`` and permutations.  SPEC 7 permits refusing the wire,
#: and requires saying so rather than silently doing something else.
_WIRE_NOT_SUPPORTED = (
    "pbc={pbc} asks for a single periodic axis (a wire). SPEC 7 permits this "
    "package to refuse it, and it does, rather than silently running 3-D Ewald "
    "or an open sum on it: the 1-D lattice sum is conditionally convergent in a "
    "different way from both, and neither the Yeh-Berkowitz slab correction nor "
    "tinfoil boundaries apply. Use (True, True, True), a (True, True, False) "
    "slab, or open boundaries."
)

#: Raised when a uniform field is supplied on the charge sector under periodic
#: boundaries and the bond graph percolates.  SPEC 11: a uniform field on free
#: charges under PBC is unbounded below, and a percolating path of bonds is
#: exactly what makes the bond-transfer sector free rather than confined.
_FIELD_ON_PERCOLATING_CHARGES = (
    "external_field has a component along a direction in which the bond graph "
    "percolates across the periodic cell, so the charge sector is unbounded "
    "below: charge transfer along a bond path that closes on its own periodic "
    "image is a current, and a uniform field drives it without limit while the "
    "position operator sum_i q_i r_i is defined only modulo a polarization "
    "quantum. The DIPOLE sector is unaffected and is still coupled -- under "
    "tinfoil boundaries the cell dipole is well defined and -Omega E . P is the "
    "Nunes-Gonze / Souza-Iniguez-Vanderbilt electric enthalpy. Supply a field "
    "with no component along the percolating directions, or use open boundaries."
)

#: Raised when the solver cannot be handed a uniform field together with the

#: Added to the squared distance before the square root wherever the separation
#: can vanish, so that neither ``sqrt(0)`` nor its infinite derivative reaches
#: the backward pass.  The affected entries are masked out of the energy
#: afterwards; the value only has to be O(1) and constant, so it carries no
#: gradient of its own.
_DIAGONAL_SHIFT = 1.0

#: Components of a traceless symmetric 3x3 quadrupole (the 2e irrep).
_QUADRUPOLE_DIM = 5


class ElectrostaticState(NamedTuple):
    """Everything one electrostatic evaluation produces.

    ``energy`` is the scalar that is added to the TRACE energy; ``assembly``
    holds the operators, which the observables need; ``solve_result`` holds the
    moments.  Keeping the assembly reachable is what lets
    :func:`trace_electrostatics.observables.polarizability` build the *same*
    operator the solver used rather than a second, independently derived one.
    """

    energy: Tensor
    assembly: "ElectrostaticAssembly"
    solve_result: SolveResult
    #: SPEC 13 dispersion, eV, or ``None`` when ``config.dispersion`` is off.
    #: Kept apart from ``energy`` so the electrostatic functional stays the
    #: one the solver minimised; :func:`electrostatic_energy` returns the sum.
    dispersion: Tensor | None = None


@dataclass(frozen=True)
class ElectrostaticAssembly:
    """The assembled quadratic form for one configuration.

    This is the system object: it owns the topology, the head parameters and
    the pair operators, and it defines the energy as a function of the moments.
    The solver receives it and needs no second derivation of ``K`` -- because
    the energy is exactly quadratic, a matrix-vector product is available as

        K x = grad_x U(x) - grad_x U(0) ,      chi-like linear term = grad_x U(0)

    which is one backward pass, works identically for the open-boundary and the
    Ewald route, and cannot drift out of step with the energy that is reported.

    Attributes
    ----------
    pos:
        ``[n_atoms, 3]`` positions, already strained by the caller.  Used for
        the all-pairs sum and for the dipole derivative, so gradients with
        respect to positions *and* to the strain parameters flow through.
    pbc:
        The boundary condition, as a 3-tuple of Python bools in ASE's
        convention.  This -- not ``cell`` -- is what decides the route
        (SPEC 7).  ``(False, False, False)`` is the open-boundary double sum,
        ``(True, True, True)`` 3-D Ewald under tinfoil boundaries, and two
        ``True``\\ s the 2-D slab route.  A static, trace-time property: it
        selects which shapes are built, never a traced value.
    cell:
        ``[3, 3]`` lattice vectors (rows), already strained, or ``None``.  It
        is ``None`` for **every** open-boundary assembly, including one that
        was handed a cell alongside ``pbc=(False, False, False)``: SPEC 7 says
        a supplied cell is ignored there, and dropping it here is what makes
        "ignored" structural rather than a comment that a later ``if
        self.cell is not None`` could contradict.
    params:
        The six physical fields from the heads: chi, eta, sigma, ell, kappa, A_mu.
    incidence:
        ``B``, ``[n_atoms, n_bonds]``, with ``1^T B == 0`` exactly, so bond
        charge transfer is charge conserving by construction.
    bond_index:
        ``[2, n_bonds]`` endpoints, oriented consistently with ``B``.
    constraint:
        ``C``, ``[n_fragments, n_atoms]``, fragment membership indicator.  The
        constraint is ``C q = Q``; since fragments are connected components of
        the bond graph and ``1^T B == 0``, ``C B t == 0``, so it reduces to a
        constraint on ``rho`` alone.
    fragment_charges:
        ``[n_fragments]`` target charges.  Per fragment, never global: a single
        global constraint is size-inconsistent (SPEC 1.3, a +1 trimer bled from
        q = 1.000 down to 0.016 as distant spectators were added).
    j_pair, t01_pair, t2_pair:
        Open-boundary pair blocks, or ``None`` under periodic boundaries where
        the lattice sum owns them.  ``t01_pair`` and ``t2_pair`` already carry
        the ``ell`` factors that convert physical dipoles to reduced ones, and
        all three already carry ``K_E``.
    t_seed:
        Geometry-determined warm start for the CG solve.  Never a previous MD
        step: that would make ``U`` history dependent and forfeit conservative
        forces.
    external_field:
        Uniform applied field ``E`` in V/A, or ``None``.  Couples as
        ``-P . E`` with ``P = sum_i (q_i r_i + mu_i)``, so the moments polarise
        in response to it and ``alpha = -d2U/dE2`` follows by differentiation.

        Open boundaries couple both sectors through ``P = sum_i (q_i r_i +
        mu_i)``.  Under periodic boundaries ``sum_i q_i r_i`` is branch
        dependent, and the field couples through the BOND-CURRENT polarization
        of SPEC 15 instead, ``Omega P = sum_i rho_i r_i + sum_b f_b tau_b d_b +
        sum_i mu_i``, which is single valued and bounded by the bond hardness
        in every direction -- the Nunes-Gonze electric enthalpy of this model.
    bond_dipole:
        ``[n_bonds, 3]`` ``d(Omega P)/d tau_b = f_b d_b`` with ``d_b`` the
        image-resolved bond vector (SPEC 15), or ``None`` under open
        boundaries, where it would equal ``B^T r`` and the original
        single-expression coupling is kept bit for bit.
    fragment_spins:
        ``[n_fragments]`` total spin quantum numbers ``S_F`` (0 singlet, 1/2
        doublet, 1 triplet -- NOT ``2 S_F``; the constraint is
        ``sum_F s_i = 2 S_F``, SPEC 9), or ``None`` for the spin-unresolved model.
        **Opt-in, default off**; see :func:`assemble`.
    quadrupoles:
        Whether the variational quadrupole sector of SPEC 10/12 is on.  **Opt-in,
        default off**; see :func:`assemble`.  When on, the moment layout gains a
        ``Theta_hat`` tail, ``x = (t, rho, mu_hat, Theta_hat)``, and
        ``params.A_Theta`` is set.
    t02_pair, t12_pair, t22_pair:
        Open-boundary Gaussian quadrupole pair tensors ``[n, n, 5]``,
        ``[n, n, 3, 5]``, ``[n, n, 5, 5]`` in reduced moments (see
        :func:`~trace_electrostatics.solver.quadrupole_pair_tensors`), or
        ``None`` with the sector off.
    dispersion_electrons:
        ``[n_atoms]`` effective polarizable electron count ``N_i`` from the
        heads, or ``None``; read only by :func:`assembly_dispersion_energy`
        (SPEC 13).
    envelope:
        ``[n_bonds]`` dissociation envelope ``f_b`` (SPEC 1.4.1), the column
        scale of ``incidence``.
    lattice_operator:
        The :class:`~trace_electrostatics.lattice_operator.LatticeOperator` of
        a ``solver='matrix_free'`` assembly (SPEC 16), in place of the dense
        pair blocks, which are then ``None`` and ``incidence`` is sparse.
    """

    pos: Tensor
    cell: Tensor | None
    params: ElectrostaticParams
    incidence: Tensor
    bond_index: Tensor
    constraint: Tensor
    fragment_charges: Tensor
    fragment_id: Tensor
    alpha_volume: Tensor
    external_field: Tensor | None
    j_pair: Tensor | None
    t01_pair: Tensor | None
    t2_pair: Tensor | None
    t_seed: Tensor
    config: ElectrostaticConfig
    pbc: Tuple[bool, bool, bool] = PBC_OPEN
    bond_dipole: Tensor | None = None
    fragment_spins: Tensor | None = None
    quadrupoles: bool = False
    t02_pair: Tensor | None = None
    t12_pair: Tensor | None = None
    t22_pair: Tensor | None = None
    dispersion_electrons: Tensor | None = None
    envelope: Tensor | None = None
    lattice_operator: Any = None
    #: SPEC 20: metallic domain per atom (-1 for non-metallic), or None.
    metal_domain: Tensor | None = None
    #: SPEC 24: ``[n_atoms]`` structure label of every atom when several
    #: independent open-boundary structures are assembled as ONE block-diagonal
    #: system (``None``: one structure, every existing path).  Pair blocks across
    #: two structures are zero, every bond and fragment lies inside one structure.
    structure_id: Tensor | None = None
    #: Number of structures ``structure_id`` labels (``0 .. n_structures - 1``).
    n_structures: int = 1

    def __post_init__(self) -> None:
        """Validate, and put the topology in ``pos``'s dtype.

        Three invariants, each of which used to be enforced somewhere else or
        not at all:

        * **``pbc`` and ``cell`` must agree.**  A periodic ``pbc`` needs a
          non-singular cell, and an open one must not carry a cell at all --
          ``assemble`` drops it, and a direct construction that kept one would
          re-route every ``self.cell is not None`` test in this class.  This is
          the structural half of SPEC 7; the *decision* is
          :func:`_normalise_pbc`'s, and it runs before the heads;
        * **the external field may not couple to free periodic charges.**  The
          guard used to live in :func:`assemble` only, so every direct
          construction of this dataclass -- which the observables and the tests
          do -- walked straight past it and got a silently meaningless number.
          Here it applies to every instance however it was built.  It is no
          longer "a field with a cell raises": SPEC 11 makes the dipole sector
          admissible, so what is checked is that the *charge* sector has been
          projected onto its bounded directions;
        * **``incidence`` and ``constraint`` follow ``pos.dtype``.**
          :func:`trace_electrostatics.topology.build_incidence` and
          ``fragment_constraint_matrix`` produce their matrices at the global
          default dtype, which is float64 in this package (SPEC 3 requires it
          before any e3nn module is constructed).  ``flashace.calculator``
          nevertheless hands in float32 positions, and ``q = B t + rho`` then
          raised "expected scalar type Double but found Float" from inside
          :meth:`energy`.  ``solver.assemble_system`` already casts its own copy
          of ``B``; the assembly did not, so the two disagreed about dtype and
          only one of them noticed.  Casting here -- once, at the seam -- keeps
          both routes in the dtype the caller actually asked for.
        """
        pbc = _as_pbc_tuple(self.pbc)
        if pbc != self.pbc:
            object.__setattr__(self, "pbc", pbc)
        n_periodic = sum(pbc)
        if n_periodic == 0 and self.cell is not None:
            raise ValueError(
                "pbc is (False, False, False) but a cell was supplied. SPEC 7 "
                "says the cell is IGNORED under open boundaries; assemble() "
                "therefore drops it, and an assembly that kept one would make "
                "`self.cell is not None` disagree with `self.periodic`. Pass "
                "cell=None, or say which axes are periodic."
            )
        if n_periodic > 0:
            if self.cell is None:
                raise ValueError(
                    f"pbc={pbc} asks for periodic boundaries but cell is None. "
                    "Periodicity is an input (SPEC 7), but the lattice it names "
                    "still has to exist."
                )
            _require_non_singular_cell(self.cell, pbc)
        if n_periodic == 1:
            raise ValueError(_WIRE_NOT_SUPPORTED.format(pbc=pbc))
        if self.external_field is not None and n_periodic > 0 and self.bond_dipole is None:
            raise ValueError(
                "an external_field was supplied under periodic boundary "
                f"conditions (pbc={pbc}) with no bond_dipole. Under PBC the field "
                "couples through the bond-current polarization of SPEC 15, "
                "sum_b f_b tau_b d_b with image-resolved bond vectors; "
                "sum_i q_i r_i is branch dependent there. assemble() builds "
                "bond_dipole from the neighbour list; an assembly constructed by "
                "hand has to supply it."
            )
        dtype = self.pos.dtype
        for name in ("incidence", "constraint"):
            matrix = getattr(self, name)
            if matrix.dtype != dtype:
                object.__setattr__(self, name, matrix.to(dtype))
        for name in ("bond_dipole", "fragment_spins"):
            value = getattr(self, name)
            if value is not None and value.dtype != dtype:
                object.__setattr__(self, name, value.to(dtype))

    # -- shape and structure ----------------------------------------------

    @property
    def n_atoms(self) -> int:
        return int(self.pos.shape[0])

    @property
    def n_bonds(self) -> int:
        return int(self.incidence.shape[1])

    @property
    def n_fragments(self) -> int:
        return int(self.constraint.shape[0])

    @property
    def periodic(self) -> bool:
        """Whether *any* axis is periodic -- i.e. whether a lattice sum runs.

        Read from ``pbc``, never from ``cell`` (SPEC 7).  The two agree by
        construction because :meth:`__post_init__` refuses an assembly where
        they do not, which is what lets the rest of this class keep testing
        ``self.cell is None`` for "no lattice sum here".
        """
        return any(self.pbc)

    @property
    def slab(self) -> bool:
        """Whether this is the 2-D slab regime: exactly two periodic axes."""
        return sum(self.pbc) == 2

    @property
    def slab_axis(self) -> int:
        """Index of the non-periodic axis of a slab.

        Raises for any other boundary condition rather than returning a
        plausible default: "which axis is the vacuum" is meaningless in the
        bulk and in the gas phase, and a silent 2 would put the Yeh-Berkowitz
        correction on an axis nobody asked about.
        """
        if not self.slab:
            raise ValueError(
                f"slab_axis is only defined for a 2-D slab; pbc={self.pbc}."
            )
        return self.pbc.index(False)

    @property
    def moment_dim(self) -> int:
        """Length of the packed moment vector ``x = (t, rho, mu_hat)``.

        A *storage* layout, not the dimension of the variational space.  Only
        ``(t, mu_hat)`` are variational (SPEC 1.3/1.5); the ``n_atoms`` middle
        rows hold ``rho``, the fixed per-fragment constraint carrier.  Anything
        that inverts or factorises an operator in this layout must first drop
        those rows and columns, or it reports a response through a channel the
        model does not have -- see
        :func:`trace_electrostatics.observables._field_response`.
        """
        quadrupole = _QUADRUPOLE_DIM * self.n_atoms if self.quadrupoles else 0
        return self.n_bonds + 4 * self.n_atoms + quadrupole

    @property
    def total_charge(self) -> Tensor:
        """Net charge of the whole configuration, ``sum_F Q_F``."""
        return self.fragment_charges.sum()

    @property
    def hardness_floor(self) -> Tensor:
        """``S_0(sigma_i)``, the tight per-atom lower bound on ``eta_i`` (SPEC 17.3)."""
        return gram_self_energy(self.params.sigma)

    # -- the energy --------------------------------------------------------

    def charges(self, t: Tensor, rho: Tensor) -> Tensor:
        """Atomic charges ``q = B t + rho``, in e."""
        if self.lattice_operator is not None:
            flow = self.envelope * t
            q = torch.zeros_like(rho).index_add(0, self.bond_index[0], flow)
            return q.index_add(0, self.bond_index[1], -flow) + rho
        return self.incidence @ t + rho

    def dipoles(self, mu_hat: Tensor) -> Tensor:
        """Physical dipoles ``mu_i = e ell_i mu_hat_i``, in e*A."""
        return self.params.ell.unsqueeze(-1) * mu_hat

    def flatten_moments(
        self, t: Tensor, rho: Tensor, mu_hat: Tensor, theta_hat: Tensor | None = None
    ) -> Tensor:
        """Pack the moments into one ``[moment_dim]`` vector.

        Order ``(t, rho, mu_hat)``, with ``Theta_hat`` appended when the
        quadrupole sector is on.
        """
        if self.quadrupoles:
            return torch.cat((t, rho, mu_hat.reshape(-1), theta_hat.reshape(-1)))
        return torch.cat((t, rho, mu_hat.reshape(-1)))

    def unflatten_moments(self, x: Tensor) -> tuple[Tensor, ...]:
        """Inverse of :meth:`flatten_moments`: 3 tensors, or 4 with quadrupoles."""
        n_atoms, n_bonds = self.n_atoms, self.n_bonds
        mu_start = n_bonds + n_atoms
        if not self.quadrupoles:
            return (
                x[:n_bonds],
                x[n_bonds:mu_start],
                x[mu_start:].reshape(n_atoms, 3),
            )
        theta_start = mu_start + 3 * n_atoms
        return (
            x[:n_bonds],
            x[n_bonds:mu_start],
            x[mu_start:theta_start].reshape(n_atoms, 3),
            x[theta_start:].reshape(n_atoms, _QUADRUPOLE_DIM),
        )

    def energy(
        self, t: Tensor, rho: Tensor, mu_hat: Tensor, theta_hat: Tensor | None = None
    ) -> Tensor:
        """Evaluate ``U = 0.5 m^T K m + chi^T m`` at the given moments.

        The terms, in order: the linear electronegativity term; the on-site
        hardness; the long-range charge-charge sum (all pairs, or the lattice
        sum); the bond-hardness penalty that makes this SQE rather than global
        QEq; the charge-dipole cross term; the dipole-dipole relay; and the
        on-site dipole stiffness.

        Quadratic only.  There is no quartic, cubic or anharmonic-in-moments
        term: the quartic failed audit on eleven counts, among them a spurious
        hyperpolarizability 300x a real chemical beta in a nominally
        centrosymmetric system.
        """
        q = self.charges(t, rho)
        params = self.params

        energy = torch.dot(params.chi, q)
        energy = energy + 0.5 * torch.dot(params.eta * q, q)
        energy = energy + 0.5 * torch.dot(params.kappa * t, t)
        if params.bond_bias is not None:
            # SPEC 20: the external circuit's work on the electrode bonds.
            energy = energy + torch.dot(params.bond_bias, t)
        energy = energy + 0.5 * torch.einsum("ia,iab,ib->", mu_hat, params.A_mu, mu_hat)
        if self.quadrupoles:
            # SPEC 10/12: on-site stiffness and the permanent source chi_Theta.
            # A uniform field has no gradient, so _field_energy has no Theta term.
            if theta_hat is None:
                raise ValueError(
                    "this assembly carries the quadrupole sector, so energy() needs "
                    "theta_hat: evaluating it without the quadrupoles would report "
                    "a different functional from the one the solver minimised."
                )
            energy = energy + 0.5 * torch.einsum(
                "im,imn,in->", theta_hat, params.A_Theta, theta_hat
            )
            if params.chi_theta is not None:
                energy = energy + (params.chi_theta * theta_hat).sum()

        # One contraction for both routes: the periodic blocks have the open
        # blocks' shapes and conventions (SPEC 7, SPEC 14).
        energy = energy + self._direct_pair_energy(
            q, mu_hat, theta_hat if self.quadrupoles else None
        )

        if self.external_field is not None:
            energy = energy + self._field_energy(t, q, mu_hat, rho)
        return energy

    def energy_by_structure(
        self, t: Tensor, rho: Tensor, mu_hat: Tensor, theta_hat: Tensor | None = None
    ) -> Tensor:
        """:meth:`energy`, split exactly by structure: ``[n_structures]`` (SPEC 24).

        Every term of ``U`` is a sum over atoms, bonds or ordered atom pairs.  In a
        block-diagonal assembly the pair blocks across structures are zero, so
        attributing each atom term, each bond term and each ordered pair ``(i, j)``
        to the structure of ``i`` (equal to that of ``j`` whenever the block is
        non-zero) partitions ``U`` with nothing shared: ``sum_s U_s = U``
        identically, and ``U_s`` is the energy structure ``s`` would have alone.
        Evaluated at the solver's minimiser, whose derivatives are implicit
        (SPEC 17), every derivative of ``U_s`` is the derivative of structure
        ``s``'s own minimum energy.
        """
        if self.structure_id is None:
            return self.energy(t, rho, mu_hat, theta_hat).reshape(1)
        if self.external_field is not None or self.lattice_operator is not None:
            raise ValueError(
                "energy_by_structure is defined for block-diagonal open-boundary "
                "assemblies without an applied field (SPEC 24)"
            )
        if self.quadrupoles and theta_hat is None:
            raise ValueError(
                "this assembly carries the quadrupole sector, so energy_by_structure() "
                "needs theta_hat (the functional the solver minimised includes it)"
            )
        q = self.charges(t, rho)
        params = self.params
        atom = params.chi * q + 0.5 * params.eta * q * q
        atom = atom + 0.5 * torch.einsum("ia,iab,ib->i", mu_hat, params.A_mu, mu_hat)
        atom = atom + 0.5 * q * (self.j_pair @ q)
        atom = atom + q * torch.einsum("ija,ja->i", self.t01_pair, mu_hat)
        atom = atom + 0.5 * torch.einsum("ia,ijab,jb->i", mu_hat, self.t2_pair, mu_hat)
        if self.quadrupoles:
            atom = atom + 0.5 * torch.einsum("im,imn,in->i", theta_hat, params.A_Theta, theta_hat)
            if params.chi_theta is not None:
                atom = atom + (params.chi_theta * theta_hat).sum(dim=-1)
            atom = atom + q * torch.einsum("ijm,jm->i", self.t02_pair, theta_hat)
            atom = atom + torch.einsum("ia,ijam,jm->i", mu_hat, self.t12_pair, theta_hat)
            atom = atom + 0.5 * torch.einsum("im,ijmn,jn->i", theta_hat, self.t22_pair, theta_hat)
        bond = 0.5 * params.kappa * t * t
        if params.bond_bias is not None:
            bond = bond + params.bond_bias * t
        energies = torch.zeros(self.n_structures, dtype=atom.dtype, device=atom.device)
        energies = energies.index_add(0, self.structure_id, atom)
        return energies.index_add(0, self.structure_id.index_select(0, self.bond_index[0]), bond)

    def _field_energy(self, t: Tensor, q: Tensor, mu_hat: Tensor, rho: Tensor) -> Tensor:
        """``-E . Omega P``.

        Open boundaries: ``P = sum_i (q_i r_i + mu_i)``, the single expression
        this coupling has always been -- kept literally, so every open-boundary
        finite-field number is bit for bit unchanged (the polarizability
        regressions read second differences of this energy).

        Periodic boundaries (SPEC 15): the bond-current polarization,
        ``Omega P = sum_i rho_i r_i + sum_b f_b tau_b d_b + sum_i mu_i``.  The
        bond term is what ``sum_i q_i r_i`` becomes once each transfer is
        attributed to the image-resolved bond it crossed: identical under open
        boundaries (``B^T r`` IS the bond vector), single valued under PBC, and
        it sees the divergence-free currents -- including self-image bonds --
        that no atomic charge records and that carry a crystal's dielectric
        response.
        """
        field = self.external_field
        if self.bond_dipole is None:
            polarization = q @ self.pos + self.dipoles(mu_hat).sum(dim=0)
            return -torch.dot(polarization, field)
        polarization = (
            rho @ self.pos + t @ self.bond_dipole + self.dipoles(mu_hat).sum(dim=0)
        )
        return -torch.dot(polarization, field)

    def _direct_pair_energy(
        self, q: Tensor, mu_hat: Tensor, theta_hat: Tensor | None = None
    ) -> Tensor:
        """Contract the stored pair blocks with the moments.

        Open boundaries: the O(N^2) sum over all distinct pairs.  Periodic: the
        same contraction against the lattice-summed blocks, which is what
        :meth:`_periodic_pair_energy` defers to.

        The charge-charge and dipole-dipole sums run over ordered pairs and are
        halved; the charge-dipole sum runs over ordered pairs and is not, since
        ``(q_i, mu_j)`` and ``(q_j, mu_i)`` are physically distinct terms.
        """
        if self.lattice_operator is not None:
            phi, field, grad = self.lattice_operator.apply_reduced(
                q, mu_hat, self.params.ell, theta_hat
            )
            energy = 0.5 * (torch.dot(q, phi) + (mu_hat * field).sum())
            if theta_hat is not None:
                energy = energy + 0.5 * (theta_hat * grad).sum()
            return energy
        if self.j_pair is None or self.t01_pair is None or self.t2_pair is None:
            raise ValueError(
                "this assembly carries no pair blocks, so the pair energy "
                "cannot be evaluated. Both routes build them; an assembly "
                "constructed by hand has to supply them."
            )
        energy = 0.5 * torch.dot(q, self.j_pair @ q)
        energy = energy + torch.einsum("i,ija,ja->", q, self.t01_pair, mu_hat)
        energy = energy + 0.5 * torch.einsum("ia,ijab,jb->", mu_hat, self.t2_pair, mu_hat)
        if theta_hat is not None:
            # Ordered pairs for the two cross-rank sums (q_i Theta_j and q_j
            # Theta_i are distinct terms), halved ordered pairs for Theta-Theta,
            # exactly as for the dipole blocks above.
            energy = energy + torch.einsum("i,ijm,jm->", q, self.t02_pair, theta_hat)
            energy = energy + torch.einsum(
                "ia,ijam,jm->", mu_hat, self.t12_pair, theta_hat
            )
            energy = energy + 0.5 * torch.einsum(
                "im,ijmn,jn->", theta_hat, self.t22_pair, theta_hat
            )
        return energy

    def _periodic_pair_energy(self, q: Tensor, mu_hat: Tensor) -> Tensor:
        """Lattice sum of the same damped kernels, evaluated from the blocks.

        The three blocks of :meth:`_direct_pair_energy` -- charge-charge,
        charge-dipole, dipole-dipole -- summed over the periodic images instead
        of over all pairs, under conducting (tinfoil) boundary conditions.  The
        on-site hardness, the bond term and the on-site dipole stiffness are
        excluded here: they are added by :meth:`energy` and would otherwise be
        double counted.  What is *not* excluded, and must not be, is the
        interaction of an atom with its own periodic images, the neutralising
        background and its own screening cloud -- those sit on the diagonals of
        the blocks (see :func:`_periodic_pair_blocks`).

        The contraction is literally :meth:`_direct_pair_energy`'s, which is why
        it is deferred to it: one expression, two sets of pair blocks.  It used
        to call :func:`~trace_electrostatics.ewald.ewald_multipole_energy`
        directly instead, which meant the energy and the operator the solve
        minimised were two independent derivations that happened to be nearby --
        and under PBC they were not even nearby, since nothing built the
        operator from the cell at all.  Agreement with the lattice sum is now a
        property to be *measured* rather than assumed:
        :meth:`ewald_energy_residual` measures it, and it is at worst
        **9.8e-15 relative** over random moments.
        """
        return self._direct_pair_energy(q, mu_hat)

    def ewald_energy_residual(self, q: Tensor, mu_hat: Tensor) -> Tensor:
        """``pair energy from the blocks`` minus ``ewald_multipole_energy``.

        The check that :func:`_periodic_pair_blocks` really is the lattice sum
        it claims to be.  Zero to round-off: at worst 9.8e-15 relative over six
        draws of random moments on four atoms in an 8 A cell.  An analysis
        routine, not a hot path -- it evaluates the whole Ewald sum a second
        time.

        On a slab the reference is the 3-D sum **plus the Yeh-Berkowitz
        correction**, because that is the functional the blocks encode
        (:func:`_slab_correction_blocks`).  Comparing against the bare 3-D sum
        instead would report the correction itself as a residual and turn a
        working slab into an apparent block-assembly bug.
        """
        if self.cell is None:
            raise ValueError(
                "ewald_energy_residual is only defined for a periodic assembly."
            )
        reference = ewald_multipole_energy(
            charges=q,
            dipoles=self.dipoles(mu_hat),
            pos=self.pos,
            cell=self.cell,
            sigma=self.params.sigma,
            alpha_volume=self.alpha_volume,
            params=self.config.ewald_params(self.cell, self.pbc),
            thole_a=self.config.thole_a,
        )
        if self.slab:
            reference = reference + _slab_correction_energy(
                q, self.dipoles(mu_hat), self.pos, self.cell, self.slab_axis
            )
        return self._direct_pair_energy(q, mu_hat) - reference

    # -- the operator, for diagnostics and observables ---------------------

    def dense_hessian(self) -> Tensor:
        """Assemble ``K`` densely in the basis ``x = (t, rho, mu_hat)``.

        Built from the *stored pair blocks*, deliberately not from autograd on
        :meth:`energy`.  The Hessian of a scalar is symmetric by construction,
        so taking it from autograd would launder an asymmetric operator into a
        symmetric one and hide exactly the class of bug that Maxwell reciprocity
        is meant to catch: a damping factor keyed on the wrong atom of the pair
        shows up as ``K != K^T`` here, and therefore as an asymmetric
        polarizability, while the energy would only ever see the symmetric part
        and look innocent.  Do not symmetrise the result.

        Cost is O((n_bonds + 4 n_atoms)^2) memory and the caller's solve is
        cubic, so this is an analysis and diagnostics routine, never a hot path.
        """
        if self.lattice_operator is not None:
            raise NotImplementedError(
                "this is a solver='matrix_free' assembly (SPEC 16): it holds no "
                "dense operator by design. Use the fixed-k route (polarizability, "
                "dielectric_tensor, born_charges_from_energy), which only needs "
                "the solve."
            )
        if self.j_pair is None or self.t01_pair is None or self.t2_pair is None:
            raise ValueError(
                "dense_hessian() needs the stored open-boundary pair blocks and "
                "this assembly has none."
            )

        n_atoms, n_bonds = self.n_atoms, self.n_bonds
        incidence = self._dense_incidence()
        h_qq = self.j_pair + torch.diag(self.params.eta)

        # charge-transfer / on-site-charge blocks
        h_b = h_qq @ incidence                              # [n, n_bonds]
        k_tt = incidence.transpose(0, 1) @ h_b + torch.diag(self.params.kappa)
        k_t_rho = h_b.transpose(0, 1)                       # [n_bonds, n]

        # cross blocks with the reduced dipoles
        t01_flat = self.t01_pair.reshape(n_atoms, 3 * n_atoms)
        k_t_mu = incidence.transpose(0, 1) @ t01_flat       # [n_bonds, 3n]

        # dipole-dipole block: pair relay plus on-site stiffness
        k_mu_mu = self.t2_pair.permute(0, 2, 1, 3).reshape(3 * n_atoms, 3 * n_atoms)
        k_mu_mu = k_mu_mu + torch.block_diag(*self.params.A_mu.unbind(0))

        if not self.quadrupoles:
            top = torch.cat((k_tt, k_t_rho, k_t_mu), dim=1)
            middle = torch.cat((k_t_rho.transpose(0, 1), h_qq, t01_flat), dim=1)
            bottom = torch.cat(
                (k_t_mu.transpose(0, 1), t01_flat.transpose(0, 1), k_mu_mu), dim=1
            )
            return torch.cat((top, middle, bottom), dim=0)

        # SPEC 10/12: the Theta_hat tail, from the stored pair tensors.
        n_theta = _QUADRUPOLE_DIM * n_atoms
        t02_flat = self.t02_pair.reshape(n_atoms, n_theta)                 # [n, 5n]
        k_t_theta = incidence.transpose(0, 1) @ t02_flat                   # [n_bonds, 5n]
        k_mu_theta = self.t12_pair.permute(0, 2, 1, 3).reshape(3 * n_atoms, n_theta)
        k_theta_theta = self.t22_pair.permute(0, 2, 1, 3).reshape(n_theta, n_theta)
        k_theta_theta = k_theta_theta + torch.block_diag(*self.params.A_Theta.unbind(0))
        rows = (
            (k_tt, k_t_rho, k_t_mu, k_t_theta),
            (k_t_rho.transpose(0, 1), h_qq, t01_flat, t02_flat),
            (k_t_mu.transpose(0, 1), t01_flat.transpose(0, 1), k_mu_mu, k_mu_theta),
            (
                k_t_theta.transpose(0, 1),
                t02_flat.transpose(0, 1),
                k_mu_theta.transpose(0, 1),
                k_theta_theta,
            ),
        )
        return torch.cat([torch.cat(row, dim=1) for row in rows], dim=0)

    def curvature(self) -> Tensor:
        """The Hessian of :meth:`energy` with respect to the moments, by autograd.

        Use this, and not :meth:`dense_hessian`, for the positive-definiteness
        diagnostic.  The two differ in exactly one way and it matters in
        opposite directions:

        * an autograd Hessian is symmetric by construction, so it sees only the
          symmetric part of the operator.  Positive definiteness is a property
          of the symmetric part alone (``x^T K x`` cannot see the rest), so
          nothing is lost here -- and this route works under periodic boundary
          conditions, where the pair blocks live inside the lattice sum and are
          never materialised;
        * the polarizability, by contrast, is *not* a property of the symmetric
          part, so it is built from :meth:`dense_hessian`, which would otherwise
          launder an asymmetric operator into a symmetric one and hide the bug
          that Maxwell reciprocity exists to catch.

        Cost is one backward pass per degree of freedom.  Diagnostics only.
        """
        reference = self.pos
        origin = torch.zeros(
            self.moment_dim, dtype=reference.dtype, device=reference.device
        )

        def scalar(x: Tensor) -> Tensor:
            return self.energy(*self.unflatten_moments(x))

        return torch.autograd.functional.hessian(scalar, origin, vectorize=True)

    def constraint_operator(self) -> Tensor:
        """``A`` in ``A x = Q``, shape ``[n_fragments, moment_dim]``.

        The bond block is ``C B``, **not** zero.  The previous version hard-coded
        it to zero on the argument that fragments are connected components of the
        bond graph, so that ``C B == 0``.  That premise is false whenever the
        caller supplies ``fragment_id`` explicitly -- which :mod:`topology`
        actively recommends -- because any two molecules within the neighbour
        cutoff then have bonds crossing the fragment boundary.

        With the block zeroed, every constraint row vanished once the (non
        variational) ``rho`` columns were removed, and the polarizability came
        out fully unconstrained: measured 69% high on a 6-atom two-fragment
        chain and 309% high on a 10-atom five-fragment chain.

        Note the rows are linearly dependent: ``1^T B == 0`` means they sum to
        zero on the bond block.  Consumers must use a rank-revealing reduction
        (see :func:`trace_electrostatics.diagnostics.null_space_basis`), never a
        magnitude threshold.
        """
        if self.lattice_operator is not None:
            raise NotImplementedError(
                "this is a solver='matrix_free' assembly (SPEC 16): it holds no "
                "dense operator by design. Use the fixed-k route (polarizability, "
                "dielectric_tensor, born_charges_from_energy), which only needs "
                "the solve."
            )
        n_atoms, n_bonds = self.n_atoms, self.n_bonds
        block_t = self.constraint @ self.incidence
        n_moments = (3 + (_QUADRUPOLE_DIM if self.quadrupoles else 0)) * n_atoms
        zeros_moments = self.constraint.new_zeros((self.n_fragments, n_moments))
        return torch.cat((block_t, self.constraint, zeros_moments), dim=1)

    def dipole_derivative(self) -> Tensor:
        """``D = dP / dx``, shape ``[moment_dim, 3]``.

        The polarization is ``P = sum_i (q_i r_i + mu_i)``, so

            dP / dt_b      = sum_i B_ib r_i   (the bond vector, up to orientation)
            dP / drho_i    = r_i
            dP / dmu_hat_i = ell_i * identity

        ``D`` is what couples a uniform external field into the energy as
        ``-E . P``, and therefore what turns the inverse of ``K`` into the
        polarizability.  Both the charge-transfer rows and the dipole rows are
        present: the two contributions to alpha are different blocks of the same
        inverse, not two separate models.
        """
        if self.lattice_operator is not None:
            raise NotImplementedError(
                "this is a solver='matrix_free' assembly (SPEC 16): it holds no "
                "dense operator by design. Use the fixed-k route (polarizability, "
                "dielectric_tensor, born_charges_from_energy), which only needs "
                "the solve."
            )
        n_atoms = self.n_atoms
        if self.bond_dipole is not None:
            # SPEC 15: the bond-current rows, single valued under PBC. The rho
            # rows below are the fixed carrier's and are never relaxed.
            d_t = self.bond_dipole
        else:
            d_t = self._dense_incidence().transpose(0, 1) @ self.pos   # [n_bonds, 3]
        d_rho = self.pos                                              # [n, 3]
        identity = torch.eye(3, dtype=self.pos.dtype, device=self.pos.device)
        d_mu = (self.params.ell.view(n_atoms, 1, 1) * identity).reshape(3 * n_atoms, 3)
        if self.quadrupoles:
            # A quadrupole carries no dipole: dP/dTheta_hat = 0.  It still
            # changes alpha, through its coupling to q and mu in K^-1.
            d_theta = self.pos.new_zeros((_QUADRUPOLE_DIM * n_atoms, 3))
            return torch.cat((d_t, d_rho, d_mu, d_theta), dim=0)
        return torch.cat((d_t, d_rho, d_mu), dim=0)

    def _dense_incidence(self) -> Tensor:
        incidence = self.incidence
        return incidence.to_dense() if incidence.is_sparse else incidence


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble(
    h: Tensor,
    pos: Tensor,
    edge_index: Tensor,
    edge_vec: Tensor,
    edge_len: Tensor,
    heads: ElectrostaticHeads,
    config: ElectrostaticConfig,
    *,
    pbc: Any = None,
    cell: Tensor | None = None,
    edge_shift: Tensor | None = None,
    fragment_id: Tensor | None = None,
    fragment_charges: Tensor | Sequence[float] | None = None,
    total_charge: float = 0.0,
    external_field: Tensor | None = None,
    fragment_spins: Tensor | Sequence[float] | None = None,
    quadrupoles: bool = False,
    metallic: Tensor | None = None,
    electrode_potential: Tensor | None = None,
    species: Tensor | None = None,
    structure_id: Tensor | None = None,
) -> ElectrostaticAssembly:
    """Run the heads, build the topology, and form the quadratic form.

    ``structure_id`` (``[n_atoms]``, SPEC 24) assembles several INDEPENDENT
    open-boundary structures as one block-diagonal system: pair blocks across
    structures are zero, and the caller supplies every structure's fragments and
    fragment charges (``fragment_id`` / ``fragment_charges``, labels global over
    the batch).  The minimiser and every derivative then equal those of the
    structures solved one by one; :meth:`ElectrostaticAssembly.energy_by_structure`
    splits the energy.  Each structure must take the dense route on its own.

    ``metallic`` (``[n_atoms]`` bool) and ``electrode_potential``
    (``[n_atoms]``, V, NaN off-electrode) switch on SPEC 20: each connected
    metallic region becomes an equipotential conductor, and electrodes are held
    at fixed potential differences by an external circuit.

    Everything up to but not including the solve.  Separated out so that the
    observables can rebuild the operator the solver actually used.

    Args:
        h: ``[n_atoms, irreps.dim]`` TRACE node features.
        pos: ``[n_atoms, 3]`` positions, already strained by the caller.
        edge_index: ``[2, n_edges]`` neighbour list, ``(src, dst)``.
        edge_vec: ``[n_edges, 3]`` ``pos[src] - pos[dst]``, already including
            the periodic shift.  Taken from the caller rather than recomputed,
            so that the strain derivative is the caller's single definition.
        edge_len: ``[n_edges]`` ``|edge_vec|``.
        heads: the readout heads.
        config: validated :class:`ElectrostaticConfig`.
        pbc: the boundary condition, in ASE's convention -- a 3-tuple (or any
            3-element sequence, array or tensor) of bools, or a single bool
            meaning all three axes.  **This, not the cell, decides the route**
            (SPEC 7).  ``None`` infers it: no cell, or an all-zero cell, means
            open; any other cell means ``(True, True, True)``.  The all-zero
            case is the one that matters, because it is what ASE hands out for
            a gas-phase frame and what used to be routed into Ewald, where the
            reciprocal-lattice construction inverts a singular matrix and
            raises.  A zero (or otherwise singular) cell with an explicitly
            periodic ``pbc`` is a caller error and raises.
        cell: ``[3, 3]`` lattice vectors (rows).  Data for whichever route
            ``pbc`` selects; under open boundaries it is **ignored**, and the
            assembly is built with ``cell=None`` so that nothing downstream can
            act on it by accident.
        edge_shift: ``[n_edges, 3]`` integer cell offsets, if periodic.  Always
            shape-checked.  It is read on exactly one path -- deciding whether
            the bond graph percolates across the cell, which is what makes a
            uniform field on the charge sector admissible or not (SPEC 11) --
            and nowhere else, because everything else needs the *displacement*
            and ``edge_vec`` already carries the shift: the caller built it as
            ``pos[src] - pos[dst] + edge_shift @ cell``, and taking the
            displacement from the caller rather than reconstructing it here is
            what keeps the strain derivative a single definition.
        fragment_id: ``[n_atoms]`` fragment labels.  Supply these from the data
            pipeline for ahead-of-time deployment; deriving them here requires a
            connected-components pass and a host synchronisation.
        fragment_charges: ``[n_fragments]`` target charges, one per fragment.
        total_charge: net charge, used only when ``fragment_charges`` is omitted.
            A Python float, because the net charge is a static property of the
            system rather than a traced quantity.
        external_field: uniform applied field ``[3]`` in V/A, or ``None``.
            Under periodic boundaries it couples through the bond-current
            polarization of SPEC 15, in every direction.
        fragment_spins: ``[n_fragments]`` total spins ``S_F`` -- not ``2 S_F``;
            the constraint is ``sum_F s_i = 2 S_F`` (SPEC 9, the same convention
            as ``FragmentConstraints.spin_targets``).
            **Opt-in; ``None`` is the spin-unresolved model and is the default,
            so every existing path is untouched.**  Passing them turns on the
            spin-resolved charge sector, which needs ``solver`` support; see
            :func:`_spin_constraints`.
        quadrupoles: whether to enable the on-site quadrupole couplings of
            SPEC 10.  **Opt-in, default ``False``.**  Whether the feature irreps
            can carry them is the heads' decision, and a head whose anisotropic
            block would be silently dead must raise (SPEC 3's reachability
            rule); see :func:`_head_readout`.
    """
    n_atoms = int(pos.shape[0])
    pbc, cell = _normalise_pbc(pbc, cell)
    periodic = any(pbc)
    from .config import resolve_solver

    n_structures = 1
    if structure_id is not None:
        structure_id, n_structures, largest = _validate_structure_id(
            structure_id, n_atoms, pos.device, edge_index, periodic, config,
            fragment_id, fragment_charges, external_field, metallic, electrode_potential,
        )
    route = resolve_solver(config, pbc, n_atoms if structure_id is None else largest)
    if structure_id is not None and route != "dense":
        raise ValueError(
            f"structure_id: a structure of {largest} atoms takes the {route!r} route on "
            "its own (config.solver / AUTO_DENSE_OPEN_MAX_ATOMS); the block-diagonal "
            "assembly is dense-only, so solve that structure on its own (SPEC 24)"
        )
    if route != config.solver:
        # Everything below reads config.solver; hand it the resolved route.
        config = dataclasses.replace(config, solver=route)
    if not periodic and n_atoms > config.max_direct_atoms and config.solver != "matrix_free":
        raise ValueError(
            f"The dense open-boundary route materialises {n_atoms}x{n_atoms} pair "
            f"blocks, which exceeds max_direct_atoms={config.max_direct_atoms}. "
            "Use ElectrostaticConfig(solver='matrix_free') (SPEC 18: no n x n "
            "object), or raise the limit deliberately if the memory is there. Do "
            "NOT supply a cell to get around it: that changes the physics of an "
            "isolated system (periodic images)."
        )
    if not periodic and config.reciprocal == "pme":
        raise ValueError(
            "reciprocal='pme' is a periodic reciprocal sum; open boundaries have "
            "none. Drop it (the open matrix-free route sums all pairs directly, "
            "SPEC 18) or supply pbc for a genuinely periodic system."
        )

    if edge_shift is not None and edge_shift.shape != (edge_index.shape[1], 3):
        raise ValueError(
            f"edge_shift has shape {tuple(edge_shift.shape)}, expected "
            f"({int(edge_index.shape[1])}, 3) -- one integer cell offset per "
            "edge. A mismatch means the caller's edge arrays and shift array "
            "describe different neighbour lists."
        )

    # --- topology ---------------------------------------------------------
    matrix_free = config.solver == "matrix_free"
    incidence, bond_index, bond_slot, slot_sign = _build_incidence(
        edge_index, n_atoms, edge_len, edge_shift if periodic else None, sparse=matrix_free
    )
    bond_len = _bond_geometry(edge_len, bond_slot)

    # The solver builds K DENSELY as [n_m, n_m], n_m = n_bonds + 3 n_atoms (bond
    # variables doubled with spin). It is the dominant allocation on both routes
    # -- ~15x the pair blocks at 1000 atoms -- so it is what has to be bounded,
    # and it can only be bounded here, once n_bonds is known.
    n_bond_variables = int(incidence.shape[1]) * (
        2 if getattr(heads, "spin", False) else 1
    )
    n_moments = n_bond_variables + (3 + (_QUADRUPOLE_DIM if quadrupoles else 0)) * n_atoms
    operator_bytes = n_moments * n_moments * torch.finfo(pos.dtype).bits // 8
    if operator_bytes > config.max_operator_bytes and not matrix_free:
        raise ValueError(
            f"The solver would build a dense {n_moments}x{n_moments} operator K "
            f"({operator_bytes / 1e9:.2f} GB; n_bonds={int(incidence.shape[1])} "
            f"for {n_atoms} atoms), above max_operator_bytes="
            f"{config.max_operator_bytes / 1e9:.2f} GB. Peak memory is ~3x this "
            "once autograd holds the CG iterations. This bounds both the open "
            "and the periodic route. Raise the limit deliberately, or shorten "
            "the bond cutoff (n_bonds grows with the cube of it); systems of "
            "this size need a matrix-free solver."
        )

    kernel = config.multipole_kernel
    if quadrupoles:
        if kernel != "gaussian":
            raise ValueError(
                "quadrupoles=True requires ElectrostaticConfig(multipole_kernel="
                "'gaussian') (SPEC 12). The Thole-coupled (q, mu, Theta) Hessian "
                "is not positive definite by construction and was measured "
                "indefinite (-0.135 eV) at the heads' initial values with every "
                "block and every pairwise sub-block positive definite."
            )
        # T22 is materialised as [n, n, 81] with ~3 temporaries of that size.
        quadrupole_bytes = 4 * 81 * n_atoms * n_atoms * torch.finfo(pos.dtype).bits // 8
        if quadrupole_bytes > config.max_operator_bytes and not matrix_free:
            raise ValueError(
                f"the quadrupole pair tensors for {n_atoms} atoms need ~"
                f"{quadrupole_bytes / 1e9:.2f} GB, above max_operator_bytes="
                f"{config.max_operator_bytes / 1e9:.2f} GB. Raise the limit "
                "deliberately; a matrix-free route does not exist yet."
            )
    if config.dispersion_quadrupoles and not quadrupoles:
        raise ValueError(
            "dispersion_quadrupoles=True needs quadrupoles=True: there is no "
            "quadrupole block to fluctuate."
        )

    fragment_id, n_fragments = _resolve_fragments(
        edge_index, n_atoms, fragment_id, fragment_charges
    )

    # --- heads ------------------------------------------------------------
    readout = _head_readout(heads, h, bond_index, bond_len, quadrupoles, species)
    params, t_seed = readout.params, readout.t_seed
    if not quadrupoles and params.A_Theta is not None:
        # Heads built with quadrupole blocks route A_Theta into params; the
        # sector is still off unless THIS call asked for it (SPEC 10: opt-in).
        params = dataclasses.replace(params, A_Theta=None, chi_theta=None)
    dispersion_readout = getattr(readout, "dispersion", None)
    if config.dispersion and dispersion_readout is None:
        raise ValueError(
            "config.dispersion=True, but these heads emit no effective electron "
            "count N_i. Construct ElectrostaticHeads(dispersion=True) (SPEC 13)."
        )
    dispersion_electrons = (
        dispersion_readout.electrons
        if (config.dispersion and dispersion_readout is not None)
        else None
    )

    # --- one dtype for the whole assembly ---------------------------------
    # See _match_dtype. The positions and the head parameters can arrive in
    # different dtypes -- flashace.calculator builds pos as float32 while
    # SPEC 3 forces the heads to float64 -- and they first met inside a matmul,
    # which raised rather than deciding anything.
    work_dtype = torch.promote_types(pos.dtype, params.chi.dtype)
    params, t_seed = _match_dtype(params, t_seed, work_dtype)

    # --- dissociation envelope (SPEC 1.4.1) -------------------------------
    # The split charge on bond b is t_b = f(r_b) tau_b, with tau_b the solver's
    # variable and f a smooth envelope that vanishes with zero slope at the
    # bond cutoff. Enforced by scaling the incidence columns, q = B diag(f) tau,
    # so every consumer of B -- K, the linear term, the constraint C B, the
    # charges -- sees it with no other change, and by enveloping the seed.
    onset = getattr(heads, "envelope_onset", None)
    envelope = bond_envelope(bond_len.to(work_dtype), float(heads.r_max), onset)
    _warn_if_first_shell_in_envelope_shell(bond_index, bond_len, n_atoms, heads, onset)
    # Pauli screening of the split-charge channel (config.bond_screening): a third atom
    # between i and j blocks their direct overlap, so f_b -> f_b S_b.  Only relative bond
    # vectors enter, so S is the same for either edge_vec sign convention (a global flip is
    # an inversion of the whole configuration).
    screening = getattr(config, "bond_screening", None)
    if screening is not None:
        from .screening import bond_screening as _bond_screening

        oriented = slot_sign.to(work_dtype).unsqueeze(-1) * edge_vec.index_select(0, bond_slot).to(work_dtype)
        envelope = envelope * _bond_screening(
            bond_index, oriented, n_atoms, c_min=screening[0], c_max=screening[1]
        )
    if incidence.is_sparse:
        index = incidence.indices()
        incidence = torch.sparse_coo_tensor(
            index,
            incidence.values().to(work_dtype) * envelope.index_select(0, index[1]),
            incidence.shape,
        ).coalesce()
    else:
        incidence = incidence.to(work_dtype) * envelope.unsqueeze(0)
    t_seed = t_seed * envelope
    # SPEC 15: d(Omega P)/d tau_b = f_b d_b, with d_b the IMAGE-RESOLVED bond
    # vector gathered from the caller's edge_vec. Under open boundaries this is
    # B^T r exactly, and the existing chi-shift coupling is kept bit for bit.
    bond_dipole = None
    if periodic or matrix_free:
        # (Matrix-free open boundaries use the same bond-current coupling: it is
        # the chi -> chi - E.r shift written through B, SPEC 15.)
        bond_vector = slot_sign.to(work_dtype).unsqueeze(-1) * edge_vec.index_select(
            0, bond_slot
        ).to(work_dtype)
        bond_dipole = envelope.unsqueeze(-1) * bond_vector

    # --- SPEC 20: metallic domains and the external circuit ----------------
    metal_domain = None
    if metallic is not None and bool(torch.as_tensor(metallic).any()):
        if getattr(heads, "spin", False):
            raise NotImplementedError(
                "metallic domains with the spin sector are not implemented: the "
                "star and circuit bonds would need spin-channel counterparts."
            )
        metal = metal_bonds(
            torch.as_tensor(metallic, device=pos.device),
            bond_index,
            slot_sign.to(work_dtype).unsqueeze(-1)
            * edge_vec.index_select(0, bond_slot).to(work_dtype),
            fragment_id,
            pos.to(work_dtype),
            None
            if electrode_potential is None
            else torch.as_tensor(electrode_potential, device=pos.device).to(work_dtype),
        )
        n_extra = int(metal.bond_index.shape[1])
        metal_domain = metal.domain
        if n_extra:
            columns = torch.arange(n_extra, device=pos.device)
            n_physical = int(bond_index.shape[1])
            lo, hi = metal.bond_index[0].to(pos.device), metal.bond_index[1].to(pos.device)
            if incidence.is_sparse:
                extra_index = torch.stack(
                    [torch.cat([lo, hi]), torch.cat([columns, columns]) + n_physical]
                )
                extra_values = torch.cat(
                    [torch.ones(n_extra, dtype=work_dtype, device=pos.device),
                     -torch.ones(n_extra, dtype=work_dtype, device=pos.device)]
                )
                incidence = torch.sparse_coo_tensor(
                    torch.cat([incidence.indices(), extra_index], dim=1),
                    torch.cat([incidence.values(), extra_values]),
                    (n_atoms, n_physical + n_extra),
                ).coalesce()
            else:
                extra = torch.zeros(n_atoms, n_extra, dtype=incidence.dtype, device=pos.device)
                extra = extra.index_put((lo, columns), torch.ones(n_extra, dtype=incidence.dtype, device=pos.device))
                extra = extra.index_put((hi, columns), -torch.ones(n_extra, dtype=incidence.dtype, device=pos.device))
                incidence = torch.cat([incidence, extra], dim=1)
            bond_index = torch.cat([bond_index, metal.bond_index.to(bond_index.device)], dim=1)
            envelope = torch.cat([envelope, torch.ones(n_extra, dtype=envelope.dtype, device=envelope.device)])
            t_seed = torch.cat([t_seed, torch.zeros(n_extra, dtype=t_seed.dtype, device=t_seed.device)])
            if bond_dipole is not None:
                bond_dipole = torch.cat([bond_dipole, metal.vector.to(bond_dipole.dtype)], dim=0)
            params = dataclasses.replace(
                params,
                kappa=torch.cat([
                    params.kappa,
                    torch.full((n_extra,), METAL_BOND_HARDNESS, dtype=params.kappa.dtype, device=params.kappa.device),
                ]),
                bond_bias=torch.cat([
                    torch.zeros(n_physical, dtype=params.kappa.dtype, device=params.kappa.device),
                    metal.bias.to(params.kappa.dtype),
                ]),
            )
    elif electrode_potential is not None and bool(torch.isfinite(torch.as_tensor(electrode_potential)).any()):
        raise ValueError(
            "electrode_potential needs metallic=True on the electrode atoms: an "
            "electrode is a metallic domain (SPEC 20)."
        )
    pos = pos.to(work_dtype)
    cell = None if cell is None else cell.to(work_dtype)
    external_field = (
        None if external_field is None else external_field.to(work_dtype)
    )

    constraint = fragment_constraint_matrix(fragment_id, n_fragments, dtype=work_dtype)
    charges_per_fragment = _resolve_fragment_charges(
        fragment_charges, total_charge, n_fragments, pos
    )
    spins_per_fragment = _resolve_fragment_spins(fragment_spins, n_fragments, pos)

    if config.check_hardness_floor:
        # SPEC 2.1.1. O(n), and it is the assertion that stands between the
        # model and an energy that is unbounded below.
        assert_hardness_floor(params.eta, params.sigma)

    # Isotropic polarizability volume, A^3, for the Thole reduced distance.
    # alpha_response = ell^2 * tr(A_mu^-1) / 3 in e^2 A^2 / eV; it is NOT a
    # volume until multiplied by K_E (SPEC 0, the 14x trap).
    a_mu_inverse = torch.linalg.inv(params.A_mu)
    alpha_response = params.ell * params.ell * _trace(a_mu_inverse) / 3.0
    alpha_volume = to_polarizability_volume(alpha_response).clamp_min(
        MIN_POLARIZABILITY_VOLUME
    )

    # --- pair operators ---------------------------------------------------
    # Materialised on all three routes, and for the same reason: they are the
    # operator the solver has to minimise, and building them here is what makes
    # the functional it minimises and the energy reported below the same one.
    t02_pair = t12_pair = t22_pair = None
    lattice_operator = None
    if matrix_free and not periodic:
        # SPEC 18: open boundaries, matrix free. All pairs in chunks, O(n^2)
        # time and O(n) memory, no n x n object (open_operator.OpenOperator).
        if config.dispersion:
            raise NotImplementedError(
                "dispersion needs the dense induced-dipole block (a Cholesky per "
                "frequency); it is not available with solver='matrix_free'."
            )
        if kernel == "gaussian":
            sigma_mu, sigma_theta = gaussian_widths(params, config.gaussian_a)
            lattice_operator = OpenOperator(
                pos, sigma=params.sigma, kernel="gaussian", sigma_mu=sigma_mu,
                sigma_theta=sigma_theta if quadrupoles else None,
            )
        else:
            lattice_operator = OpenOperator(
                pos, sigma=params.sigma, kernel="thole",
                alpha_volume=alpha_volume, thole_a=config.thole_a,
            )
        j_pair = t01_pair = t2_pair = None
    elif matrix_free:
        # SPEC 16: no n x n object. The pair list reaches the Ewald real-space
        # cutoff; the operator reproduces the dense lattice blocks to round-off.
        if config.dispersion:
            raise NotImplementedError(
                "dispersion needs the dense induced-dipole block (a Cholesky per "
                "frequency); it is not available with solver='matrix_free'."
            )
        if kernel == "gaussian":
            sigma_mu, sigma_theta = gaussian_widths(params, config.gaussian_a)
            widths = [params.sigma, sigma_mu] + ([sigma_theta] if quadrupoles else [])
        else:
            widths = [params.sigma]
        # The real-space kernel of a Gaussian pair decays on its own width, so
        # the cutoff never goes below the width's own accuracy radius.
        r_cut = max(float(config.ewald_real_cutoff), _gaussian_cutoff_floor(widths, config))
        if config.ewald_balance and config.reciprocal == "ewald":
            r_cut = max(
                r_cut,
                balanced_real_cutoff(
                    n_atoms, float(torch.det(cell.detach()).abs()), float(config.ewald_accuracy)
                ),
            )
        ewald = EwaldParams.auto_tune(
            cell, r_cut, accuracy=float(config.ewald_accuracy), alpha=config.ewald_alpha
        )
        pairs = lattice_pairs(pos, cell, r_cut, pbc)
        reciprocal_kwargs = dict(reciprocal=config.reciprocal, pme_order=config.pme_order)
        if kernel == "gaussian":
            lattice_operator = LatticeOperator(
                pos, cell, ewald, pairs, sigma=params.sigma, kernel="gaussian",
                sigma_mu=sigma_mu, sigma_theta=sigma_theta if quadrupoles else None,
                **reciprocal_kwargs,
            )
        else:
            lattice_operator = LatticeOperator(
                pos, cell, ewald, pairs, sigma=params.sigma, kernel="thole",
                alpha_volume=alpha_volume, thole_a=config.thole_a, **reciprocal_kwargs,
            )
        if sum(pbc) == 2:
            raise NotImplementedError(
                "solver='matrix_free' does not yet carry the Yeh-Berkowitz slab "
                "correction; use the dense route for slabs."
            )
        j_pair = t01_pair = t2_pair = None
    elif periodic and kernel == "gaussian":
        # SPEC 14: the Gram lattice operator, every rank, one construction.
        sigma_mu, sigma_theta = gaussian_widths(params, config.gaussian_a)
        # A Gaussian pair's real-space kernel decays on its OWN width, which no
        # Ewald alpha can move into reciprocal space, so the real-space cutoff
        # is floored at that width's accuracy radius instead of raising: the
        # user asked for an accuracy, not for a cutoff. (SPEC 17.5)
        floor = _gaussian_cutoff_floor(
            [params.sigma, sigma_mu] + ([sigma_theta] if quadrupoles else []), config
        )
        lattice_config = (
            config
            if float(config.ewald_real_cutoff) >= floor
            else dataclasses.replace(config, ewald_real_cutoff=floor)
        )
        lattice = gaussian_lattice_blocks(
            pos,
            cell,
            params.sigma,
            sigma_mu,
            params.ell,
            lattice_config.ewald_params(cell, pbc),
            sigma_theta=sigma_theta if quadrupoles else None,
        )
        j_pair, t01_pair, t2_pair = lattice["j"], lattice["t01"], lattice["t2"]
        if quadrupoles:
            t02_pair, t12_pair, t22_pair = lattice["t02"], lattice["t12"], lattice["t22"]
        if sum(pbc) == 2:
            # Yeh-Berkowitz acts on the cell dipole only; a quadrupole carries
            # none, so the Theta blocks need no correction.
            j_pair, t01_pair, t2_pair = _slab_correction_blocks(
                j_pair, t01_pair, t2_pair, pos, cell, params, pbc.index(False)
            )
    elif periodic:
        j_pair, t01_pair, t2_pair = _periodic_pair_blocks(
            pos, cell, params, alpha_volume, config, pbc=pbc
        )
        if sum(pbc) == 2:
            # SPEC 7: 3-D Ewald on a slab is WRONG. The correction is applied to
            # the BLOCKS, not to the reported energy, so that the solve
            # minimises the corrected functional too.
            j_pair, t01_pair, t2_pair = _slab_correction_blocks(
                j_pair, t01_pair, t2_pair, pos, cell, params, pbc.index(False)
            )
    else:
        j_pair, t01_pair, t2_pair = _direct_pair_blocks(
            pos,
            params,
            alpha_volume,
            config.thole_a,
            kernel=kernel,
            gaussian_a=config.gaussian_a,
        )
    if quadrupoles and not periodic and not matrix_free:
        t02_pair, t12_pair, t22_pair = _direct_quadrupole_blocks(
            pos, params, config.gaussian_a
        )
    if structure_id is not None:
        # SPEC 24: independent structures do not interact. Zeroing every block
        # across two structures makes K block diagonal; the same masked blocks
        # are handed to the solver (_solver_geometry), so the functional it
        # minimises and the one reported are one operator.
        same = (structure_id.unsqueeze(1) == structure_id.unsqueeze(0)).to(j_pair.dtype)
        j_pair = j_pair * same
        t01_pair = t01_pair * same.unsqueeze(-1)
        t2_pair = t2_pair * same.unsqueeze(-1).unsqueeze(-1)
        if quadrupoles:
            t02_pair = t02_pair * same.unsqueeze(-1)
            t12_pair = t12_pair * same.unsqueeze(-1).unsqueeze(-1)
            t22_pair = t22_pair * same.unsqueeze(-1).unsqueeze(-1)

    return ElectrostaticAssembly(
        pos=pos,
        cell=cell,
        params=params,
        incidence=incidence,
        bond_index=bond_index,
        constraint=constraint,
        fragment_charges=charges_per_fragment,
        fragment_id=fragment_id,
        alpha_volume=alpha_volume,
        j_pair=j_pair,
        t01_pair=t01_pair,
        t2_pair=t2_pair,
        t_seed=t_seed,
        config=config,
        external_field=external_field,
        pbc=pbc,
        bond_dipole=bond_dipole,
        envelope=envelope,
        lattice_operator=lattice_operator,
        metal_domain=metal_domain,
        fragment_spins=spins_per_fragment,
        quadrupoles=bool(quadrupoles),
        t02_pair=t02_pair,
        t12_pair=t12_pair,
        t22_pair=t22_pair,
        dispersion_electrons=(
            None if dispersion_electrons is None else dispersion_electrons.to(work_dtype)
        ),
        structure_id=structure_id,
        n_structures=n_structures,
    )


def _validate_structure_id(
    structure_id, n_atoms, device, edge_index, periodic, config,
    fragment_id, fragment_charges, external_field, metallic, electrode_potential,
):
    """Checks of a SPEC 24 block-diagonal assembly; ``(labels, n_structures, largest)``.

    Each refusal names what would otherwise be silently wrong: a periodic cell
    (one lattice per structure, not one), an edge or a fragment joining two
    structures (charge would equilibrate between independent systems), a
    fragment specification left to inference (the per-structure charge rules
    cannot be applied to a union), and the inputs this route does not carry.
    """
    labels = torch.as_tensor(structure_id, device=device)
    if labels.dtype not in (torch.int32, torch.int64) or labels.shape != (n_atoms,):
        raise ValueError(
            f"structure_id must be an integer tensor of shape ({n_atoms},), got "
            f"{labels.dtype} {tuple(labels.shape)}"
        )
    labels = labels.long()
    if periodic:
        raise NotImplementedError(
            "structure_id assembles open-boundary structures only: periodic "
            "structures each carry their own lattice sum (SPEC 24)"
        )
    if config.dispersion:
        raise NotImplementedError("structure_id with dispersion is not implemented (SPEC 24)")
    if external_field is not None:
        raise NotImplementedError("structure_id with an applied field is not implemented (SPEC 24)")
    if (metallic is not None and bool(torch.as_tensor(metallic).any())) or (
        electrode_potential is not None
        and bool(torch.isfinite(torch.as_tensor(electrode_potential)).any())
    ):
        raise NotImplementedError("structure_id with metals or electrodes is not implemented (SPEC 24)")
    if fragment_id is None or fragment_charges is None:
        raise ValueError(
            "structure_id needs fragment_id and fragment_charges for the whole "
            "batch: the per-structure charge rules (SPEC 1.3) are applied by the "
            "caller, structure by structure, never to the union"
        )
    n_structures = int(labels.max().item()) + 1
    counts = torch.bincount(labels, minlength=n_structures)
    if int(labels.min().item()) < 0 or bool((counts == 0).any()):
        raise ValueError(f"structure_id must use every label 0 .. {n_structures - 1}")
    if edge_index.numel() and bool((labels[edge_index[0]] != labels[edge_index[1]]).any()):
        raise ValueError(
            "an edge joins two structures; independent structures must be a "
            "disjoint union, or charge would flow between them"
        )
    fragment_id = torch.as_tensor(fragment_id, device=device).long()
    pairs = torch.unique(torch.stack([fragment_id, labels]), dim=1)
    if int(pairs.shape[1]) != int(torch.unique(fragment_id).numel()):
        raise ValueError("a fragment spans two structures; every fragment must lie in one")
    return labels, n_structures, int(counts.max().item())


def _direct_pair_blocks(
    pos: Tensor,
    params: ElectrostaticParams,
    alpha_volume: Tensor,
    thole_a: float,
    *,
    kernel: str = "thole",
    gaussian_a: float = DEFAULT_GAUSSIAN_A,
) -> tuple[Tensor, Tensor, Tensor]:
    """All-pairs charge-charge, charge-dipole and dipole-dipole blocks.

    The self terms are removed with a constant 0/1 mask rather than by indexing,
    so the shapes stay static and the diagonal contributes no gradient.

    Coincident sites
    ----------------
    The square root is taken of a separation that has been substituted wherever
    it is degenerate -- **every** degenerate entry, not only the ``i == j``
    diagonal.  An earlier version shifted the diagonal alone,
    ``sqrt(squared + _DIAGONAL_SHIFT * eye)``, on the reasoning that only a site
    with itself can be at zero separation.  Two *distinct* atoms at coincident
    positions break that: their entry took ``sqrt(0)``, whose derivative is
    infinite, and although the energy stayed finite the forces did not.
    Measured on four atoms with atoms 1 and 2 superposed: ``U = 0.484398``
    (finite) with ``nan`` gradient components on both of them, and the NaN then
    spreads through the whole force array on the next optimiser step.
    :mod:`trace_electrostatics.kernels` guards this case with
    ``DEGENERATE_SEPARATION``, but that guard sits *downstream* -- it sanitises
    the ``r`` it is handed and cannot repair the derivative of the square root
    that produced it.  So the guard has to be here as well, on the same
    threshold.

    It also has to be in :func:`~trace_electrostatics.solver.assemble_system`,
    which is a **separate** copy of this construction and is the one the SOLVE
    uses: on the open-boundary route ``_solver_geometry`` passes no pair
    overrides, so ``assemble_system`` rebuilds these blocks from ``positions``
    itself.  That copy kept the diagonal-only shift long after this one was
    fixed, and because the reported energy comes from the solve, coincident
    atoms still produced ``nan`` forces through ``electrostatic_state``
    (measured E = -0.0037534496 with two of four atoms superposed).  Both now
    substitute on the same threshold with the same ``_DIAGONAL_SHIFT``; change
    one and change the other.

    The substituted entries are masked out of all three blocks, with one
    exception that is a limit rather than a mask: the charge-charge block takes
    its true ``r -> 0`` value ``kernel_self(sigma_ij)`` there, because two
    coincident Gaussian charge clouds have a perfectly well defined finite
    interaction and it is the value ``coulomb_kernel`` itself returns at zero.
    The dipole tensors have no such limit -- ``lambda_3 / r^3`` is finite but
    direction-dependent -- so they are zeroed, exactly as ``kernels.py`` does.
    Both are NaN shields at 1e-12 A, twelve orders of magnitude below any
    separation a neighbour list produces; neither is a physical regularisation.

    The reduced-dipole factors ``ell_j`` (charge-dipole) and ``ell_i ell_j``
    (dipole-dipole) are folded in here, so the blocks act directly on
    ``mu_hat``.  Note that the ``ell`` factor of ``t01_pair`` sits on the
    *dipole* index only: the block is deliberately not symmetric in ``i, j``
    because it couples two different kinds of moment.  ``t2_pair`` carries both
    factors and is symmetric under the simultaneous exchange of site and
    Cartesian indices -- which is what makes ``K`` symmetric, and hence what
    Maxwell reciprocity tests.
    """
    n_atoms = int(pos.shape[0])
    vec = pos.unsqueeze(1) - pos.unsqueeze(0)                    # [n, n, 3]
    squared = (vec * vec).sum(-1)
    eye = torch.eye(n_atoms, dtype=pos.dtype, device=pos.device)
    off_diagonal = 1.0 - eye

    # Substitute the *input* of the square root, never its output: masking the
    # output still sends a zero cotangent into an infinite derivative, and
    # 0 * inf = NaN.  Where the pair is genuinely separated this is bit-for-bit
    # the old expression, since `squared` passes through torch.where unchanged
    # and the old diagonal shift was multiplied by an eye that is zero there.
    separated = squared > DEGENERATE_SEPARATION * DEGENERATE_SEPARATION
    dist = torch.sqrt(
        torch.where(separated, squared, torch.full_like(squared, _DIAGONAL_SHIFT))
    )
    keep = separated.to(pos.dtype) * off_diagonal

    sigma_pair = combine_sigma(params.sigma.unsqueeze(1), params.sigma.unsqueeze(0))
    j_pair = torch.where(
        separated, coulomb_kernel(dist, sigma_pair), kernel_self(sigma_pair)
    ) * off_diagonal

    if kernel == "thole":
        lam3, lam5 = thole_damping(
            dist, alpha_volume.unsqueeze(1), alpha_volume.unsqueeze(0), thole_a
        )
        lam3_charge_dipole = lam3
    else:
        # SPEC 12, the same widths and damping the solver builds (one
        # construction site each, in solver.py).
        sigma_mu, _ = gaussian_widths(params, gaussian_a)
        lam3_charge_dipole, lam3, lam5 = gaussian_dipole_damping(
            dist, params.sigma, sigma_mu
        )

    ell = params.ell
    t01_pair = charge_dipole_tensor(vec, dist, lam3_charge_dipole)
    t01_pair = t01_pair * ell.view(1, n_atoms, 1) * keep.unsqueeze(-1)

    t2_pair = dipole_interaction_tensor(vec, dist, lam3, lam5)
    ell_pair = ell.view(n_atoms, 1, 1, 1) * ell.view(1, n_atoms, 1, 1)
    t2_pair = t2_pair * ell_pair * keep.view(n_atoms, n_atoms, 1, 1)

    return j_pair, t01_pair, t2_pair


def _direct_quadrupole_blocks(
    pos: Tensor, params: ElectrostaticParams, gaussian_a: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Open-boundary quadrupole pair tensors ``(t02, t12, t22)``, SPEC 10/12.

    The coincident-site substitution is :func:`_direct_pair_blocks`' own, on the
    same threshold, and the tensors come from
    :func:`~trace_electrostatics.solver.quadrupole_pair_tensors` -- the function
    ``assemble_system`` calls -- so the reported energy and the minimised
    functional are one construction.
    """
    vec, dist, keep = _open_pair_geometry(pos)
    sigma_mu, sigma_theta = gaussian_widths(params, gaussian_a)
    return quadrupole_pair_tensors(
        vec, dist, keep, params.sigma, sigma_mu, sigma_theta, params.ell
    )


def _open_pair_geometry(pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(r_vec, r, keep)`` with :func:`_direct_pair_blocks`' coincidence guard."""
    n_atoms = int(pos.shape[0])
    vec = pos.unsqueeze(1) - pos.unsqueeze(0)
    squared = (vec * vec).sum(-1)
    off_diagonal = 1.0 - torch.eye(n_atoms, dtype=pos.dtype, device=pos.device)
    separated = squared > DEGENERATE_SEPARATION * DEGENERATE_SEPARATION
    dist = torch.sqrt(
        torch.where(separated, squared, torch.full_like(squared, _DIAGONAL_SHIFT))
    )
    return vec, dist, separated.to(pos.dtype) * off_diagonal


def _widened_moment_blocks(
    pos: Tensor,
    params: ElectrostaticParams,
    gaussian_a: float,
    width: float,
    quadrupoles: bool,
) -> dict:
    """The long-range Gaussian moment blocks ``T_w`` of SPEC 13's range separation.

    Every per-site width is widened, ``sigma -> sqrt(sigma^2 + width^2/2)``, so
    each pair width becomes ``sqrt(s_ij^2 + width^2)``.  ``T - T_w`` then has
    the Fourier weight ``g_i g_j (1 - exp(-k^2 width^2/4)) 4 pi / k^2 >= 0``: a
    Gram form in a positive metric, so ``A + T - T_w`` is positive definite by
    the same argument as ``A + T``.
    """
    n_atoms = int(pos.shape[0])
    vec, dist, keep = _open_pair_geometry(pos)
    sigma_mu, sigma_theta = gaussian_widths(params, gaussian_a)
    extra = 0.5 * width * width
    sigma_mu = torch.sqrt(sigma_mu * sigma_mu + extra)
    _, lam3, lam5 = gaussian_dipole_damping(dist, params.sigma, sigma_mu)
    ell = params.ell
    t2 = dipole_interaction_tensor(vec, dist, lam3, lam5) * (
        keep * ell.unsqueeze(1) * ell.unsqueeze(0)
    )[..., None, None]
    blocks = {"dipole": t2.permute(0, 2, 1, 3).reshape(3 * n_atoms, 3 * n_atoms)}
    if quadrupoles:
        sigma_theta = torch.sqrt(sigma_theta * sigma_theta + extra)
        _, t12, t22 = quadrupole_pair_tensors(
            vec, dist, keep, params.sigma, sigma_mu, sigma_theta, ell
        )
        n_theta = _QUADRUPOLE_DIM * n_atoms
        blocks["dipole_quadrupole"] = t12.permute(0, 2, 1, 3).reshape(3 * n_atoms, n_theta)
        blocks["quadrupole"] = t22.permute(0, 2, 1, 3).reshape(n_theta, n_theta)
    return blocks


def _fold_counts(cell: Tensor, pbc: Tuple[bool, bool, bool], length: float) -> Tuple[int, int, int]:
    """Supercell multiplicities ``M_a = ceil(length / d_a)`` on the periodic axes.

    ``d_a = 2 pi / |b_a|`` is the spacing of lattice planes, the length a q-grid
    along reciprocal vector ``b_a`` has to reach.  A host read of the cell, at
    most once per evaluation, like the Ewald tuning it sits beside.
    """
    spacing = TWO_PI / (TWO_PI * torch.linalg.inv(cell.detach()).transpose(0, 1)).norm(dim=-1)
    return tuple(
        max(1, int(math.ceil(length / float(spacing[axis]) - 1e-9))) if pbc[axis] else 1
        for axis in range(3)
    )


def _supercell(tensor: Tensor, copies: int) -> Tensor:
    """Repeat a per-atom tensor over ``copies`` cells (cell index major)."""
    return tensor.repeat(copies, *([1] * (tensor.dim() - 1)))


def assembly_dispersion_energy(
    assembly: "ElectrostaticAssembly", config: ElectrostaticConfig
) -> Tensor:
    """SPEC 13 dispersion of an assembly: the zero-point energy of its moment block.

    Reads the same reduced dipole block (and, with
    ``config.dispersion_quadrupoles``, the quadrupole blocks) that the solver
    minimised, so induction and dispersion come from one operator.

    Periodic cells (SPEC 13.1).  A Gamma-only zero-point energy puts every cell
    of the crystal in phase, which on a small cell is wrong by factors.  The
    cell is folded onto a supercell (``config.dispersion_supercell_length``) --
    a Gamma-point supercell of ``M`` cells is an exact ``M``-point q-grid,
    block-diagonal in Bloch waves -- and the energy per cell is the supercell
    energy over ``M``.  On a 3-D lattice the Gamma sector also gets its
    non-analytic longitudinal term (``config.dispersion_gamma_correction``); see
    :func:`~trace_electrostatics.dispersion.dispersion_energy`.
    """
    electrons = assembly.dispersion_electrons
    if electrons is None:
        raise ValueError(
            "config.dispersion=True needs the effective electron count N_i, and "
            "these heads emit none. Construct ElectrostaticHeads(dispersion=True)."
        )
    if assembly.t2_pair is None:
        raise ValueError("this assembly carries no dipole pair block to fluctuate.")
    if config.dispersion_quadrupoles and not assembly.quadrupoles:
        raise ValueError(
            "dispersion_quadrupoles=True needs the quadrupole sector "
            "(quadrupoles=True): there is no quadrupole block to fluctuate."
        )
    params = assembly.params
    kernel = config.multipole_kernel
    n_atoms = assembly.n_atoms
    copies = 1
    fold = (1, 1, 1)
    if assembly.periodic:
        fold = _fold_counts(assembly.cell, assembly.pbc, float(config.dispersion_supercell_length))
        copies = fold[0] * fold[1] * fold[2]

    ell, A_mu, sigma = params.ell, params.A_mu, params.sigma
    A_theta = params.A_Theta if config.dispersion_quadrupoles else None
    if copies == 1:
        pos, cell = assembly.pos, assembly.cell
        t2, t12, t22 = assembly.t2_pair, assembly.t12_pair, assembly.t22_pair
    else:
        offsets = torch.cartesian_prod(
            *[torch.arange(m, dtype=assembly.pos.dtype, device=assembly.pos.device) for m in fold]
        )
        pos = (assembly.pos.unsqueeze(0) + (offsets @ assembly.cell).unsqueeze(1)).reshape(-1, 3)
        cell = torch.tensor(fold, dtype=assembly.cell.dtype, device=assembly.cell.device).view(3, 1) * assembly.cell
        ell, A_mu, sigma = _supercell(ell, copies), _supercell(A_mu, copies), _supercell(sigma, copies)
        electrons = _supercell(electrons, copies)
        A_theta = None if A_theta is None else _supercell(A_theta, copies)
        t2, t12, t22 = _lattice_moment_blocks(
            pos, cell, assembly, config, ell, A_mu, sigma, A_theta, widen=0.0
        )
    n_super = n_atoms * copies

    def flat(t2_, t12_, t22_):
        blocks = {"dipole": t2_.permute(0, 2, 1, 3).reshape(3 * n_super, 3 * n_super)}
        if A_theta is not None:
            n_theta = 5 * n_super
            blocks["A_Theta"] = A_theta
            blocks["dipole_quadrupole_pair"] = t12_.permute(0, 2, 1, 3).reshape(3 * n_super, n_theta)
            blocks["quadrupole_pair"] = t22_.permute(0, 2, 1, 3).reshape(n_theta, n_theta)
        return blocks

    def zero_point(blocks: dict, volume: Tensor | None) -> Tensor:
        moments = {key: value for key, value in blocks.items() if key != "dipole"}
        return dispersion_energy(
            blocks["dipole"],
            A_mu,
            ell,
            electrons.to(A_mu.dtype),
            n_quadrature=config.dispersion_quadrature,
            frequency_scale=config.dispersion_frequency_scale,
            macroscopic_volume=volume,
            **moments,
        )

    volume = None
    if assembly.periodic and sum(assembly.pbc) == 3 and config.dispersion_gamma_correction:
        volume = torch.det(cell).abs()
    full_blocks = flat(t2, t12, t22)
    energy = zero_point(full_blocks, volume)
    if config.dispersion_range != 0.0:
        # SPEC 13, range separation by SUBTRACTION: E = ZPE[A + T] - ZPE[A + T_SR],
        # T_SR = T - T_w. The long-range tail keeps the full intramolecular
        # screening of the first term. T_SR is analytic at q -> 0 (both have the
        # same 4 pi/k^2 head, which cancels), so it takes no Gamma term.
        if assembly.periodic:
            wide = flat(*_lattice_moment_blocks(
                pos, cell, assembly, config, ell, A_mu, sigma, A_theta,
                widen=float(config.dispersion_range),
            ))
        else:
            wide = _widened_moment_blocks(
                assembly.pos, params, config.gaussian_a,
                float(config.dispersion_range), A_theta is not None,
            )
            wide = {
                "dipole": wide["dipole"],
                **({} if A_theta is None else {
                    "A_Theta": A_theta,
                    "dipole_quadrupole_pair": wide["dipole_quadrupole"],
                    "quadrupole_pair": wide["quadrupole"],
                }),
            }
        short = {key: full_blocks[key] - wide[key] if key != "A_Theta" else full_blocks[key]
                 for key in full_blocks}
        energy = energy - zero_point(short, None)
    return energy / copies


def _lattice_moment_blocks(
    pos: Tensor,
    cell: Tensor,
    assembly: "ElectrostaticAssembly",
    config: ElectrostaticConfig,
    ell: Tensor,
    A_mu: Tensor,
    sigma: Tensor,
    A_theta: Tensor | None,
    *,
    widen: float,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Lattice dipole (and quadrupole) pair blocks of a (super)cell, for SPEC 13.1.

    Gaussian kernel: :func:`gaussian_lattice_blocks` with every width widened by
    ``widen/sqrt(2)`` (0 = the solver's own operator).  Thole kernel: the same
    Thole Ewald the solve uses, on the supercell; widening is Gaussian-only,
    which ``ElectrostaticConfig`` already enforces.
    """
    ewald = config.ewald_params(cell, assembly.pbc)
    if config.multipole_kernel == "gaussian":
        # The widths gaussian_widths() would derive, from the (supercell)
        # stiffnesses directly.
        extra = 0.5 * widen * widen
        sigma_mu = torch.sqrt(
            gaussian_multipole_width(A_mu, ell, 1, config.gaussian_a) ** 2 + extra
        )
        sigma_theta = (
            None
            if A_theta is None
            else torch.sqrt(
                gaussian_multipole_width(A_theta, ell, 2, config.gaussian_a) ** 2 + extra
            )
        )
        blocks = gaussian_lattice_blocks(
            pos, cell, sigma, sigma_mu, ell, ewald,
            sigma_theta=sigma_theta, include_charge=False,
        )
        return blocks["t2"], blocks.get("t12"), blocks.get("t22")
    copies = pos.shape[0] // assembly.n_atoms
    params = dataclasses.replace(
        assembly.params,
        chi=_supercell(assembly.params.chi, copies),
        eta=_supercell(assembly.params.eta, copies),
        sigma=sigma,
        ell=ell,
        kappa=assembly.params.kappa.new_ones(0),
        A_mu=A_mu,
        A_Theta=None,
        chi_theta=None,
    )
    alpha_response = ell * ell * _trace(torch.linalg.inv(A_mu)) / 3.0
    alpha_volume = to_polarizability_volume(alpha_response).clamp_min(MIN_POLARIZABILITY_VOLUME)
    _, _, t2 = _periodic_pair_blocks(pos, cell, params, alpha_volume, config, pbc=assembly.pbc)
    return t2, None, None

def _periodic_pair_blocks(
    pos: Tensor,
    cell: Tensor,
    params: ElectrostaticParams,
    alpha_volume: Tensor,
    config: ElectrostaticConfig,
    pbc: Any = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """The same three blocks as :func:`_direct_pair_blocks`, lattice summed.

    Returns ``(j_pair, t01_pair, t2_pair)`` with the shapes and the reduced-dipole
    conventions of the open-boundary blocks, so that
    :meth:`ElectrostaticAssembly._periodic_pair_energy` is the *same contraction*
    as :meth:`~ElectrostaticAssembly._direct_pair_energy` and the solver can be
    handed the operator through ``Geometry``'s three overrides.

    One difference from the open-boundary blocks, and it is the whole point of
    the seam: **the diagonals are not zero.**  Under a lattice sum an atom
    interacts with its own periodic images, with the neutralising background, and
    with its own screening cloud, and those are real contributions.  They are not
    the on-site parameters ``eta`` and ``A_mu``, which
    :meth:`~ElectrostaticAssembly.energy` and ``assemble_system`` still add
    separately; in particular they must not be folded into ``eta``, which is
    bound below by the SPEC 1.2 hardness floor while the lattice self term is
    negative (about -2.5 eV at an 8 A cubic cell).

    Why this is assembled here rather than contracted out of
    :func:`~trace_electrostatics.ewald.ewald_multipole_energy`
    ------------------------------------------------------------------------
    Because a matrix is needed, not a scalar, and the only generic way to get one
    from the scalar is a Hessian: measured at 70x the cost of one energy
    evaluation at 32 atoms and growing, which puts it out of the hot path.  The
    arithmetic below is therefore a second expression of the same sum, which is
    exactly the kind of duplication that drifts -- so two things keep it honest.
    It is built from ``ewald``'s *own* geometry, screened kernels, wave vectors
    and weights, so the image box, the minimum-image recentring, the splitting
    and the damping cannot diverge; and the periodic energy is then evaluated
    from these blocks rather than by calling the lattice sum a second time, so
    there is one operator and the reported energy is its value by construction.
    Equality with ``ewald_multipole_energy`` is a property that has to be
    measured, and is: over six draws of random ``(q, mu_hat)`` on four atoms in
    an 8 A cell the residual is at worst **9.8e-15 relative** (6.9e-14 eV
    absolute on energies of order 10 eV), which is the reassociation error of
    a sum of this many terms and nothing else.  Both blocks come out symmetric
    to 3.3e-16.  :meth:`ElectrostaticAssembly.ewald_energy_residual` is the
    measurement.

    This function would be better placed next to the lattice sum it mirrors, in
    :mod:`trace_electrostatics.ewald`, and should move there.

    Cost is ``O(n_atoms^2 * n_images)`` in time, the same as one evaluation of
    the lattice sum, and ``O(9 n_atoms^2)`` in memory for ``t2_pair`` -- which
    makes the periodic route as small-system-bound as the open-boundary one, for
    the same reason.
    """
    n_atoms = int(pos.shape[0])
    ewald = config.ewald_params(cell, pbc)
    alpha = float(ewald.alpha)
    ell = params.ell.reshape(-1)

    r_vec, r, included = _ewald_pair_geometry(pos, cell, ewald)
    keep = included.to(pos.dtype)
    b0, b1, b2 = _ewald_screened_kernels(r, alpha)

    # -- real space, charge-charge ----------------------------------------
    # The bare screened kernel plus the smearing correction, written exactly as
    # ewald_real_multipole writes them so the two cannot pick up different
    # conventions: coulomb_kernel(r) - K_E/r = -K_E erfc(r/sigma_ij)/r.
    sigma_pair = combine_sigma(
        params.sigma.view(1, n_atoms, 1), params.sigma.view(1, 1, n_atoms)
    )
    smearing = coulomb_kernel(r, sigma_pair) - K_E / r
    j_pair = (keep * (K_E * b0 + smearing)).sum(dim=0)

    # -- real space, the Thole-damped dipole kernels ----------------------
    lam3, lam5 = thole_damping(
        r,
        alpha_volume.view(1, n_atoms, 1),
        alpha_volume.view(1, 1, n_atoms),
        config.thole_a,
    )
    inverse_r3 = 1.0 / (r * r * r)
    delta_3 = (lam3 - 1.0) * inverse_r3
    delta_5 = (lam5 - 1.0) * inverse_r3 / (r * r)

    # U_cross = K_E sum q_i (mu_j . r) (B_1 + delta_3); the coefficient of
    # q_i mu_{j,b} is that, times r_b.
    cross_weight = keep * K_E * (b1 + delta_3)
    t01_pair = (cross_weight.unsqueeze(-1) * r_vec).sum(dim=0)

    # U_relay = (K_E/2) sum [ (mu_i . mu_j)(B_1 + delta_3)
    #                         - (mu_i . r)(mu_j . r)(B_2 + 3 delta_5) ]
    outer_weight = keep * K_E * (b2 + 3.0 * delta_5)
    identity = torch.eye(3, dtype=pos.dtype, device=pos.device)
    t2_pair = cross_weight.sum(dim=0).unsqueeze(-1).unsqueeze(-1) * identity
    t2_pair = t2_pair - _weighted_outer(outer_weight, r_vec)

    # -- reciprocal space --------------------------------------------------
    # |S(k)|^2 with S(k) = sum_j (q_j + i k . mu_j) exp(i k . r_j) expands to
    #   sum_jl [ (q_j q_l + (k.mu_j)(k.mu_l)) cos(phi_j - phi_l)
    #            + 2 q_j (k.mu_l) sin(phi_j - phi_l) ] ,
    # and U_recip = (2 pi K_E / V) sum_k w_k |S(k)|^2.  Matching that against
    # 0.5 q^T J q + q^T T01 mu + 0.5 mu^T T2 mu puts a factor 2 on every block,
    # hence the 4 pi rather than 2 pi below.
    # `k_grid`, not `k_max`: on a slab it is the per-axis triple. EwaldParams'
    # own docstring named this function as the consumer that read only k_max
    # and so summed a grid "too large rather than too small".
    k, k_squared, nonzero = _ewald_wave_vectors(ewald.k_grid, cell)
    weight = _ewald_gaussian_weights(k_squared, nonzero, alpha)
    phase = pos @ k.transpose(-1, -2)                            # [n_atoms, n_k]
    cosine, sine = torch.cos(phase), torch.sin(phase)
    prefactor = (2.0 * TWO_PI * K_E) / torch.det(cell).abs()

    weighted_cos = cosine * weight
    weighted_sin = sine * weight
    j_pair = j_pair + prefactor * (
        weighted_cos @ cosine.transpose(0, 1) + weighted_sin @ sine.transpose(0, 1)
    )

    # sin(phi_i - phi_j) = s_i c_j - c_i s_j, with one k_b factor from k . mu.
    t01_pair = t01_pair + prefactor * torch.stack(
        [
            (sine * (weight * k[:, b])) @ cosine.transpose(0, 1)
            - (cosine * (weight * k[:, b])) @ sine.transpose(0, 1)
            for b in range(3)
        ],
        dim=-1,
    )

    # cos(phi_i - phi_j) = c_i c_j + s_i s_j, with k_a k_b from the two k . mu.
    t2_pair = t2_pair + prefactor * torch.stack(
        [
            torch.stack(
                [
                    (cosine * (weight * k[:, a] * k[:, b])) @ cosine.transpose(0, 1)
                    + (sine * (weight * k[:, a] * k[:, b])) @ sine.transpose(0, 1)
                    for b in range(3)
                ],
                dim=-1,
            )
            for a in range(3)
        ],
        dim=-2,
    )

    # -- self interaction and neutralising background ----------------------
    # The coefficients are *read off* ewald's own functions by evaluating them
    # on a unit moment, rather than restated here: each is exactly quadratic and
    # homogeneous, so one call fixes it, and a change to any of the three
    # conventions upstream propagates instead of drifting.  0.5 x^T K x means
    # every coefficient enters doubled.
    unit = torch.ones(1, dtype=pos.dtype, device=pos.device)
    volume = torch.det(cell).abs()
    self_charge = ewald_self(unit, ewald)
    jellium = ewald_jellium(unit.reshape(()), volume, ewald)
    self_dipole = ewald_self_dipole(unit.expand(1, 3), ewald) / 3.0

    j_pair = j_pair + 2.0 * jellium
    j_pair = j_pair + torch.diag(
        (2.0 * self_charge).expand(n_atoms).to(dtype=pos.dtype)
    )
    t2_pair = t2_pair + (
        2.0 * self_dipole
    ) * torch.eye(n_atoms, dtype=pos.dtype, device=pos.device).view(
        n_atoms, n_atoms, 1, 1
    ) * identity

    # -- reduce the dipoles, exactly as _direct_pair_blocks does ------------
    t01_pair = t01_pair * ell.view(1, n_atoms, 1)
    t2_pair = t2_pair * (ell.view(n_atoms, 1, 1, 1) * ell.view(1, n_atoms, 1, 1))
    return j_pair, t01_pair, t2_pair


# ---------------------------------------------------------------------------
# SPEC 7 -- periodicity is an input
# ---------------------------------------------------------------------------


def _as_pbc_tuple(value: Any) -> Tuple[bool, bool, bool]:
    """Coerce ASE's many spellings of ``pbc`` into a 3-tuple of Python bools.

    Accepts a single bool (all three axes), or any 3-element sequence, numpy
    array or tensor.  The result is a tuple of *Python* bools, deliberately:
    ``pbc`` selects which shapes are built, so it has to be a trace-time
    constant.  A 0-d or 3-element tensor is converted here, once, at the
    boundary, rather than being carried into the module as a traced value that
    a later ``if`` would synchronise on in the hot path.
    """
    if isinstance(value, bool):
        return (value, value, value)
    if isinstance(value, Tensor):
        if value.dim() == 0:
            return _as_pbc_tuple(bool(value))
        value = value.reshape(-1).tolist()
    if isinstance(value, tuple) and len(value) == 3 and all(
        isinstance(item, bool) for item in value
    ):
        return value  # type: ignore[return-value]
    try:
        items = list(value)
    except TypeError:
        raise TypeError(
            f"pbc must be a bool or a 3-element sequence of bools, got "
            f"{type(value).__name__}."
        )
    if len(items) == 1:
        items = items * 3
    if len(items) != 3:
        raise ValueError(
            f"pbc must have 3 entries, one per lattice vector, got {len(items)}."
        )
    return (bool(items[0]), bool(items[1]), bool(items[2]))


def _require_non_singular_cell(cell: Tensor, pbc: Tuple[bool, bool, bool]) -> None:
    """Refuse a lattice that has no reciprocal, naming the ASE frame by name.

    ``|det(cell)|`` against the cube of the largest entry, so the test is on the
    *shape* of the cell and scale-free.  This is the guard the old
    ``cell is not None`` rule did not have: ASE's gas-phase ``zeros(3, 3)`` went
    straight into ``ewald._reciprocal_vectors``, where ``torch.linalg.inv``
    raised ``linalg.inv: The diagonal element 1 is zero, the inversion could not
    be completed because the input matrix is singular`` -- a message about a
    matrix the caller never built, from a routine they never called.

    One ``bool()`` host synchronisation, on a structural property, and only when
    an axis is actually periodic.
    """
    if cell.shape != (3, 3):
        raise ValueError(
            f"cell must have shape [3, 3] (lattice vectors as rows), got "
            f"{tuple(cell.shape)}."
        )
    scale = cell.detach().abs().max()
    volume = torch.det(cell.detach()).abs()
    if bool(volume <= _SINGULAR_CELL_RTOL * scale.pow(3)):
        zero = bool(scale == 0)
        detail = (
            "the cell is all zeros -- that is ASE's gas-phase frame, which means "
            "NO periodicity, not a lattice of zero volume. Pass "
            "pbc=(False, False, False) (or leave pbc unset, which infers it)"
            if zero
            else "the lattice vectors are linearly dependent, so the cell has no "
            "reciprocal and the Ewald reciprocal-space sum is undefined"
        )
        raise ValueError(
            f"pbc={pbc} asks for periodic boundaries but {detail}. "
            f"|det(cell)| = {float(volume):.3e} against a cell scale of "
            f"{float(scale):.3e} A."
        )


def resolve_pbc(
    pbc: Any, cell: Tensor | None
) -> Tuple[Tuple[bool, bool, bool], Tensor | None]:
    """Resolve ``(pbc, cell)`` into the boundary condition and the cell to use.

    SPEC 7: periodicity is an **input**, never inferred from the cell.  This is
    the one place that decides it, and it returns the pair the rest of the
    module works with -- the ``pbc`` triple, and a cell that is ``None`` exactly
    when no axis is periodic.

    Inference, when ``pbc`` is ``None``
    -----------------------------------
    Back-compatibility, for every caller written against the old
    ``cell is not None`` rule:

    * no cell -> ``(False, False, False)``;
    * an **all-zero cell** -> ``(False, False, False)``.  This is the case the
      whole change exists for.  ASE hands a gas-phase frame ``cell =
      zeros(3, 3)``, which is not ``None``, so the old rule called it periodic
      and routed a singular lattice into Ewald, where ``auto_tune`` dies inside
      ``linalg.inv``.  Verified live.  An all-zero cell now means NON-periodic;
    * any other cell -> ``(True, True, True)``, which is what every existing
      periodic caller and every existing test means by passing one.

    Inference never *invents* periodicity that the cell cannot support, and it
    never papers over a contradiction: a cell that is non-zero but singular is a
    caller error and raises, exactly as a zero cell with an explicit
    ``pbc=(True, True, True)`` does.  "I asked for a lattice and gave you
    something that is not one" is a bug in the call, not a case to be guessed
    at.

    Returns:
        ``(pbc, cell)``.  Under open boundaries the returned cell is ``None``
        however one was supplied -- SPEC 7 says it is ignored, and dropping it
        here is what makes every later ``cell is None`` test mean "open".
    """
    if pbc is None:
        if cell is None:
            return PBC_OPEN, None
        if cell.shape != (3, 3):
            raise ValueError(
                f"cell must have shape [3, 3] (lattice vectors as rows), got "
                f"{tuple(cell.shape)}."
            )
        if bool((cell.detach() == 0).all()):
            # ASE's gas-phase frame. Open boundaries, and the cell is dropped.
            return PBC_OPEN, None
        resolved = PBC_BULK
    else:
        resolved = _as_pbc_tuple(pbc)

    n_periodic = sum(resolved)
    if n_periodic == 0:
        return PBC_OPEN, None
    if cell is None:
        raise ValueError(
            f"pbc={resolved} asks for periodic boundaries but no cell was "
            "supplied. Periodicity is an input (SPEC 7), but the lattice it "
            "names still has to exist."
        )
    _require_non_singular_cell(cell, resolved)
    if n_periodic == 1:
        raise ValueError(_WIRE_NOT_SUPPORTED.format(pbc=resolved))
    return resolved, cell


# Used by :func:`assemble` and by ``ElectrostaticAssembly.__post_init__``.
_normalise_pbc = resolve_pbc


# ---------------------------------------------------------------------------
# SPEC 7 -- the 2-D slab, by Yeh-Berkowitz correction of the 3-D sum
# ---------------------------------------------------------------------------


def _slab_geometry(
    pos: Tensor, cell: Tensor, axis: int
) -> tuple[Tensor, Tensor, Tensor]:
    """``(normal, height, volume)`` for a slab whose non-periodic axis is ``axis``.

    ``normal`` is the unit surface normal, taken from the **lattice vector** of
    the non-periodic axis rather than from the Cartesian axis of the same index:
    a ``(True, True, False)`` cell whose third vector runs along Cartesian *y*
    is perfectly legal, and keying the correction on index 2 would then apply it
    to the wrong direction and look plausible.

    Two requirements, both checked, both from the derivation:

    * **the non-periodic lattice vector must be orthogonal to the periodic
      plane.**  Yeh-Berkowitz replaces the 3-D sum's spurious slab-image dipole
      interaction by ``2 pi M_n^2 / V``, which assumes the images are stacked
      along the normal.  A staggered stack is a different sum;
    * **there must be vacuum.**  The correction removes the *dipolar* coupling
      between images; it says nothing about their charge clouds overlapping.
      With no vacuum at all the images are in contact and no correction of this
      form is valid.  The convergence condition is that the vacuum gap be at
      least the slab thickness (Yeh & Berkowitz recommend ``L_n >= 3 h``, i.e.
      a gap of twice the thickness, at which the residual image interaction is
      already below the Ewald accuracy); only the *sign* of the gap is enforced
      here, because the rest is an accuracy choice the caller owns.
    """
    periodic_axes = [index for index in range(3) if index != axis]
    normal_vector = cell[axis]
    scale = cell.detach().abs().max()
    for other in periodic_axes:
        overlap = torch.dot(normal_vector.detach(), cell[other].detach()).abs()
        if bool(overlap > _LATTICE_RTOL * scale * scale):
            raise ValueError(
                f"the 2-D slab route needs lattice vector {axis} (the "
                "non-periodic one) orthogonal to the periodic plane, but it has "
                f"a component {float(overlap):.3e} A^2 along vector {other}. "
                "The Yeh-Berkowitz correction assumes the periodic images of the "
                "slab are stacked along the surface normal; a staggered stack is "
                "a different lattice sum. Re-orient the cell so the vacuum axis "
                "is orthogonal to the surface."
            )
    height = normal_vector.norm()
    normal = normal_vector / height
    coordinate = pos.detach() @ normal.detach()
    thickness = coordinate.max() - coordinate.min()
    if bool(thickness >= height.detach()):
        raise ValueError(
            "the 2-D slab route needs vacuum padding along the non-periodic "
            f"axis, but the atoms span {float(thickness):.3f} A of a "
            f"{float(height.detach()):.3f} A cell -- the periodic images are in "
            "contact. The Yeh-Berkowitz correction removes the dipolar coupling "
            "between slab images, not the overlap of their charge clouds. Pad "
            "the cell along the non-periodic axis; the convergence condition is "
            "a vacuum gap of at least the slab thickness, and Yeh & Berkowitz "
            "recommend a cell three times the thickness."
        )
    return normal, height, torch.det(cell).abs()


def _slab_correction_energy(
    q: Tensor, dipoles: Tensor, pos: Tensor, cell: Tensor, axis: int
) -> Tensor:
    """``+ 2 pi K_E M_n^2 / V`` -- the Yeh-Berkowitz slab correction, as a scalar.

    SPEC 7: 3-D Ewald on a slab is wrong, because the periodic images along the
    non-periodic axis interact through their dipoles and contribute a spurious
    ``2 pi M_n^2 / V``.  Under tinfoil boundary conditions that term is what the
    3-D sum has and the slab must not, so the correction *adds* its negative --
    which is the same thing as adding ``+2 pi M_n^2 / V`` to the tinfoil sum,
    since tinfoil is the sum with the surface term already removed.

    ``M_n = sum_i (q_i r_i + mu_i) . n`` with ``n`` the surface normal.  The
    route implemented here is therefore **3-D Ewald plus vacuum padding, minus
    the image-dipole term**, not a genuine 2-D Ewald: it keeps the O(N^2 +
    N log N) structure of the 3-D sum and the whole of ``ewald.py`` unchanged,
    at the cost of the padding requirement checked in :func:`_slab_geometry`.

    Used for the *reference* in :meth:`ElectrostaticAssembly.ewald_energy_residual`.
    The energy the model evaluates gets it through the blocks instead, so that
    the solve minimises the corrected functional -- see
    :func:`_slab_correction_blocks`.
    """
    normal, _height, volume = _slab_geometry(pos, cell, axis)
    moment = torch.dot(q, pos @ normal) + torch.dot(dipoles.sum(dim=0), normal)
    return (TWO_PI * K_E / volume) * moment * moment


def _slab_correction_blocks(
    j_pair: Tensor,
    t01_pair: Tensor,
    t2_pair: Tensor,
    pos: Tensor,
    cell: Tensor,
    params: ElectrostaticParams,
    axis: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """The same correction, expressed in the three pair blocks.

    ``2 pi K_E M_n^2 / V`` is exactly quadratic in the moments, so it is a
    rank-one addition to each block rather than a term bolted onto the reported
    energy.  With ``c = 2 pi K_E / V``, ``z_i = r_i . n`` and the reduced
    dipoles ``mu_i = ell_i mu_hat_i``:

        M_n^2 = (sum_i q_i z_i)^2
              + 2 (sum_i q_i z_i)(sum_j ell_j mu_hat_j . n)
              + (sum_i ell_i mu_hat_i . n)^2

    which, matched against ``0.5 q^T J q + q^T T01 mu_hat + 0.5 mu_hat^T T2
    mu_hat``, gives the three lines below.  Every coefficient enters doubled,
    for the same reason the Ewald self terms do: the halves in the quadratic
    form.

    Putting it here rather than in :meth:`ElectrostaticAssembly.energy` is the
    whole point.  The blocks are what :func:`_solver_geometry` hands the solver,
    so the corrected functional is the one that is *minimised* as well as the
    one that is reported.  A correction applied only to the reported energy
    would leave the moments equilibrated to the uncorrected slab -- the same
    class of defect as the 0.513 eV mismatch that the periodic overrides exist
    to prevent, and just as quiet.
    """
    n_atoms = int(pos.shape[0])
    normal, _height, volume = _slab_geometry(pos, cell, axis)
    coefficient = 2.0 * (TWO_PI * K_E / volume)
    z = pos @ normal                                        # [n]
    ell = params.ell.reshape(-1)
    ell_n = ell.unsqueeze(-1) * normal                      # [n, 3]

    j_pair = j_pair + coefficient * (z.unsqueeze(1) * z.unsqueeze(0))
    t01_pair = t01_pair + coefficient * (
        z.view(n_atoms, 1, 1) * ell_n.view(1, n_atoms, 3)
    )
    t2_pair = t2_pair + coefficient * (
        ell_n.view(n_atoms, 1, 3, 1) * ell_n.view(1, n_atoms, 1, 3)
    )
    return j_pair, t01_pair, t2_pair


# ---------------------------------------------------------------------------
# SPEC 11 -- a uniform field under periodic boundaries
#
# The decision this section implements, stated once:
#
#   * the DIPOLE sector is always coupled.  Under tinfoil boundaries the cell
#     dipole is well defined, Omega P = sum_i mu_i, and -Omega E . P is the
#     Nunes-Gonze / Souza-Iniguez-Vanderbilt electric enthalpy.  Bounded,
#     admissible, implemented;
#   * the CHARGE sector is coupled where charge transfer is CONFINED, and
#     refused where it percolates.  That is SPEC 11's option (a), the stronger
#     one: the safe option (b) -- refuse the charge sector outright under PBC --
#     would give up the entire molecular-crystal and localised-kappa case, which
#     is bounded, physical and testable.  Which of the two applies to a given
#     system is decided by the bond graph, below, and never by a flag.
# ---------------------------------------------------------------------------


def _bond_lattice_shifts(
    edge_index: Tensor, edge_shift: Tensor, bond_index: Tensor, bond_slot: Tensor
) -> Tensor:
    """Per-bond integer cell offset, oriented to match ``bond_index``.

    ``bond_slot`` already names the *shortest* edge that maps onto each bond --
    the minimum image -- so the shift gathered here is the minimum-image one.
    The orientation matters and the gather does not fix it: an edge stored as
    ``(j, i)`` carries the negated offset of the bond ``(i, j)``, so a bond
    whose representative edge happens to be the reverse one would otherwise
    close a cycle with the wrong sign and invent (or hide) a percolating path.

    Returns ``[n_bonds, 3]`` in the convention ``u[i] - u[j] = shift``, with
    ``u`` the image index of each atom.
    """
    src = edge_index[0].index_select(0, bond_slot)
    shift = edge_shift.index_select(0, bond_slot)
    forward = (src == bond_index[0]).to(shift.dtype)
    return shift * (2.0 * forward - 1.0).unsqueeze(-1)


def _weighted_outer(weight: Tensor, r_vec: Tensor) -> Tensor:
    """``sum_c w[c,i,j] r[c,i,j,a] r[c,i,j,b]`` -> ``[n, n, 3, 3]``.

    Written as nine reductions rather than one ``einsum`` on purpose: every
    pairwise contraction order an einsum could pick materialises a
    ``[n_images, n, n, 3, 3]`` intermediate, which is nine times the largest
    tensor the lattice sum itself needs.  This way the peak stays at
    ``[n_images, n, n]``.
    """
    return torch.stack(
        [
            torch.stack(
                [(weight * r_vec[..., a] * r_vec[..., b]).sum(dim=0) for b in range(3)],
                dim=-1,
            )
            for a in range(3)
        ],
        dim=-2,
    )


def _trace(matrices: Tensor) -> Tensor:
    """Batched trace of ``[..., 3, 3]`` without a host synchronisation."""
    return matrices.diagonal(dim1=-2, dim2=-1).sum(-1)


def _match_dtype(
    params: ElectrostaticParams, t_seed: Tensor, dtype: torch.dtype
) -> tuple[ElectrostaticParams, Tensor]:
    """Put the head outputs in the assembly's working dtype.

    The working dtype is ``torch.promote_types(pos.dtype, chi.dtype)``: whichever
    of the positions and the head parameters carries more precision wins, and
    everything else -- ``pos``, ``cell``, the external field, the incidence and
    constraint matrices (in
    :meth:`ElectrostaticAssembly.__post_init__`) and these parameters -- is
    brought to it.

    The mismatch this exists for is not hypothetical.  ``flashace.calculator``
    builds ``pos``, ``cell`` and ``edge_shift`` as float32 unconditionally,
    while SPEC 3 requires the global default dtype to be float64 before any e3nn
    module is constructed, so the head parameters are float64.  Nothing
    reconciled the two: ``solver.assemble_system`` casts *its* copy of the
    incidence matrix to ``positions.dtype`` but leaves the parameters alone, the
    assembly cast nothing at all, and the two dtypes first met inside a matmul,
    which raised "expected scalar type Double but found Float" from inside
    :meth:`~ElectrostaticAssembly.energy` rather than deciding anything.

    **Promotion, not truncation.**  Following ``assemble_system``'s
    ``dtype = positions.dtype`` rule instead would silently downcast a float64
    model to float32, and that is not merely lossy -- it breaks a mandatory
    assertion.  Measured on three atoms with float32 positions and float64
    heads: the per-fragment charge constraint closes to 1.863e-09, against the
    ``constraint_tol`` of 1e-10 that SPEC 2.1 item 5 and SPEC 6 require, so the
    solve raises.  Promoting costs one cast of a handful of ``[n_atoms]``
    tensors and keeps the arithmetic exact; the float32 inputs are simply less
    precise inputs, which is the caller's business.

    The cast is conditional, so the ordinary all-float64 path rebuilds nothing
    and pays neither the reconstruction nor the two host synchronisations that
    ``ElectrostaticParams.__post_init__`` performs.
    """
    if params.chi.dtype == dtype and t_seed.dtype == dtype:
        return params, t_seed
    cast = dataclasses.replace(
        params,
        chi=params.chi.to(dtype),
        eta=params.eta.to(dtype),
        sigma=params.sigma.to(dtype),
        ell=params.ell.to(dtype),
        kappa=params.kappa.to(dtype),
        A_mu=params.A_mu.to(dtype),
        A_Theta=None if params.A_Theta is None else params.A_Theta.to(dtype),
        chi_theta=None if params.chi_theta is None else params.chi_theta.to(dtype),
    )
    return cast, t_seed.to(dtype)


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def electrostatic_state(
    h: Tensor,
    pos: Tensor,
    edge_index: Tensor,
    edge_vec: Tensor,
    edge_len: Tensor,
    heads: ElectrostaticHeads,
    config: ElectrostaticConfig,
    *,
    cell: Tensor | None = None,
    edge_shift: Tensor | None = None,
    fragment_id: Tensor | None = None,
    fragment_charges: Tensor | Sequence[float] | None = None,
    total_charge: float = 0.0,
    external_field: Tensor | None = None,
    pbc: Any = None,
    fragment_spins: Tensor | Sequence[float] | None = None,
    quadrupoles: bool = False,
    metallic: Tensor | None = None,
    electrode_potential: Tensor | None = None,
    species: Tensor | None = None,
    structure_id: Tensor | None = None,
) -> ElectrostaticState:
    """Assemble, solve, and evaluate -- returning the assembly as well.

    Use this rather than :func:`electrostatic_energy` when the observables are
    wanted, since the polarizability needs the operator and not only the
    moments.

    ``pbc``, ``fragment_spins`` and ``quadrupoles`` are pass-throughs to
    :func:`assemble`; see there.  The last two are opt-in and default off, so a
    call that does not mention them builds exactly the model it always did.
    """
    assembly = assemble(
        h,
        pos,
        edge_index,
        edge_vec,
        edge_len,
        heads,
        config,
        pbc=pbc,
        cell=cell,
        edge_shift=edge_shift,
        fragment_id=fragment_id,
        fragment_charges=fragment_charges,
        total_charge=total_charge,
        external_field=external_field,
        fragment_spins=fragment_spins,
        quadrupoles=quadrupoles,
        metallic=metallic,
        electrode_potential=electrode_potential,
        species=species,
        structure_id=structure_id,
    )

    return state_from_assembly(assembly, config)


def state_from_assembly(assembly: "ElectrostaticAssembly", config: ElectrostaticConfig) -> ElectrostaticState:
    """Solve an assembled quadratic form and evaluate it: the second half of :func:`electrostatic_state`.

    ``electrostatic_state(...) == state_from_assembly(assemble(...), config)`` bit for bit (it is the same
    code).  Separate so that a caller holding an assembly with modified parameters (trace_redox's formal
    mode without the bound electronegativity, SPEC 17.8 of trace_redox) solves exactly that functional.
    """
    # SPEC 17: the converged constrained minimum with implicit derivatives
    # (legacy: k CG iterations differentiated through, solve_mode="fixed_k").
    solver_params, solver_field = _field_for_solver(assembly)
    if assembly.lattice_operator is not None:
        # SPEC 16: the same recurrence with K v from the lattice operator. The
        # seed is the same geometry-determined logit; there is no Geometry to
        # bind it to, because there is no dense operator to put in one.
        solve_result = _solve_matrix_free(assembly, config.n_cg_iter, solver_field)
    else:
        geometry = _solver_geometry(assembly)
        constraints = _fragment_constraints(assembly)
        # SPEC 2: the seed is geometry-determined -- the antisymmetrised
        # attention logit -- never a previous MD step.  It is BOUND to this
        # geometry, so the solver can verify that rather than trust the
        # provenance string: a string can lie, a fingerprint cannot.
        seed = GeometrySeed.from_bond_logits(
            assembly.t_seed,
            assembly.n_atoms,
            geometry=geometry,
        )
        solve_result = solve_charges(
            solver_params,
            geometry,
            constraints,
            config.n_cg_iter,
            seed=seed,
            options=_solver_options(config),
            external_field=solver_field,
        )
    moments = _moments(solve_result, assembly)
    t, rho = moments[0], moments[1]

    if config.check_convergence:
        _warn_if_unconverged(solve_result, assembly, config)

    if config.check_positive_definite and assembly.lattice_operator is None:
        _run_solution_diagnostics(assembly, t, rho, config)

    if config.solve_mode == "converged":
        # SPEC 17: the solver's own minimum-energy node. Its first derivative is
        # the envelope theorem (no adjoint solve for the forces) and every
        # higher derivative is the minimiser's. Re-evaluating U at the returned
        # moments would give the same number -- the two constructions agree to
        # round-off -- but would push a near-zero cotangent into an adjoint
        # solve on every force call.
        energy = solve_result.energy
    else:
        energy = assembly.energy(*moments)
    dispersion = (
        assembly_dispersion_energy(assembly, config) if config.dispersion else None
    )
    return ElectrostaticState(
        energy=energy,
        assembly=assembly,
        solve_result=solve_result,
        dispersion=dispersion,
    )


def electrostatic_energies(state: ElectrostaticState) -> Tensor:
    """Per-structure energies ``[n_structures]`` of a block-diagonal state (SPEC 24).

    :meth:`ElectrostaticAssembly.energy_by_structure` at the solver's moments.
    In the converged mode those carry implicit derivatives of every order
    (SPEC 17), so ``U_s`` and all its derivatives are those of structure ``s``
    solved alone.  They sum to ``state.energy`` to round-off.
    """
    if state.dispersion is not None:
        raise NotImplementedError("electrostatic_energies with dispersion (SPEC 24)")
    return state.assembly.energy_by_structure(*_moments(state.solve_result, state.assembly))


def solver_seed_loss(
    h: Tensor,
    pos: Tensor,
    edge_index: Tensor,
    edge_vec: Tensor,
    edge_len: Tensor,
    heads: ElectrostaticHeads,
    config: ElectrostaticConfig,
    *,
    n_iter: int = 1,
    pbc: Any = None,
    cell: Tensor | None = None,
    edge_shift: Tensor | None = None,
    fragment_id: Tensor | None = None,
    fragment_charges: Tensor | Sequence[float] | None = None,
    total_charge: float = 0.0,
    fragment_spins: Tensor | Sequence[float] | None = None,
    quadrupoles: bool = False,
    metallic: Tensor | None = None,
    electrode_potential: Tensor | None = None,
    species: Tensor | None = None,
) -> Tensor:
    """Variational loss for the solver seed: ``U(m_n) / n_atoms``, physics detached.

    The ONLY loss that may train :class:`~trace_electrostatics.heads.SolverSeedPotential`.
    Add it to the training objective alongside the data loss; it touches no
    other parameter.

    Why this trains the seed toward the equilibrated solution without knowing it
    --------------------------------------------------------------------------
    On the feasible set U is convex and CG never raises it, so
    ``U(m_n) >= U(m*)`` with equality only when the solve has arrived. Minimising
    ``U(m_n)`` over the seed's parameters therefore drives the seed toward ``m*``
    -- the minimiser is never computed, only the energy one short solve reaches.
    Measured in this package on held-out 27-216 atom clusters with TRACE-trunk
    features, trained on 16-64 atoms for 600 steps: relative k = 6 energy error
    1.3e-5 - 5.0e-5 against 1.7e-4 - 6.2e-4 from a zero seed, i.e. 12-17x, and
    nearly flat in N. During that training every non-seed head parameter moved
    by exactly 0.0.

    Everything the physics owns is detached -- node features, positions, and
    every head parameter that builds K and c -- so this loss cannot change the
    energy functional, only where the solve starts. ``n_iter = 1`` gives the
    strongest signal on the seed itself; the default energy uses k =
    ``config.n_cg_iter``.

    Honest limit: the achievable seed accuracy is set by how much the node
    features know about the local environment. With a random-initialised trunk
    it plateaus (training past ~600 steps stopped helping) and has NOT been
    shown to reach the eps <= 0.05 target of SPEC 2.0.1; that must be
    re-measured on a trained trunk.
    """
    with torch.no_grad():
        assembly = assemble(
            h.detach(),
            pos.detach(),
            edge_index,
            edge_vec.detach(),
            edge_len.detach(),
            heads,
            config,
            pbc=pbc,
            cell=None if cell is None else cell.detach(),
            edge_shift=edge_shift,
            fragment_id=fragment_id,
            fragment_charges=fragment_charges,
            total_charge=total_charge,
            fragment_spins=fragment_spins,
            quadrupoles=quadrupoles,
            metallic=metallic,
            electrode_potential=electrode_potential,
            species=species,
        )
    potential = heads.seed_potential(
        h.detach().to(assembly.params.chi.dtype), assembly.params.chi, trainable=True
    )
    # assembly.incidence is already B diag(f): tau0 = -kappa^-1 F B^T u_hat.
    tau0 = -(assembly.incidence.T @ potential) / assembly.params.kappa
    # The seed's objective is the energy ONE fixed-k step reaches from it
    # (SPEC 2.0.1), whatever the production solve mode: a converged solve would
    # erase the seed from its own loss.
    if assembly.lattice_operator is not None:
        result = _solve_matrix_free(
            assembly, int(n_iter), None, tau_seed=tau0, solve_mode="fixed_k"
        )
        return result.energy / assembly.n_atoms
    geometry = _solver_geometry(assembly)
    constraints = _fragment_constraints(assembly)
    solver_params, solver_field = _field_for_solver(assembly)
    seed = GeometrySeed.for_geometry(
        geometry, t=tau0, provenance="solver-seed training (geometry-determined)"
    )
    result = solve_charges(
        solver_params,
        geometry,
        constraints,
        int(n_iter),
        seed=seed,
        options=dataclasses.replace(_solver_options(config), solve_mode="fixed_k"),
        external_field=solver_field,
    )
    return result.energy / assembly.n_atoms


def electrostatic_energy(
    h: Tensor,
    pos: Tensor,
    edge_index: Tensor,
    edge_vec: Tensor,
    edge_len: Tensor,
    heads: ElectrostaticHeads,
    config: ElectrostaticConfig,
    cell: Tensor | None = None,
    edge_shift: Tensor | None = None,
    fragment_id: Tensor | None = None,
    fragment_charges: Tensor | Sequence[float] | None = None,
    total_charge: float = 0.0,
    external_field: Tensor | None = None,
    pbc: Any = None,
    fragment_spins: Tensor | Sequence[float] | None = None,
    quadrupoles: bool = False,
    metallic: Tensor | None = None,
    electrode_potential: Tensor | None = None,
    species: Tensor | None = None,
) -> tuple[Tensor, SolveResult]:
    """Variational electrostatic energy of one configuration, in eV.

    Runs the readout heads, builds the bond topology and the damped pair
    operators, solves for the charge-transfer and induced moments -- the
    converged constrained minimum with implicit derivatives by default (SPEC
    17), or k CG steps under ``solve_mode="fixed_k"`` -- and returns
    ``U = 0.5 m^T K m + chi^T m`` there.

    The returned scalar is differentiable with respect to ``pos`` and to
    ``cell``, so adding it to the TRACE energy before the existing
    ``torch.autograd.grad`` call yields the electrostatic contribution to the
    forces and to the stress with no further changes.

    Which boundary condition runs is decided by ``pbc``, not by ``cell``
    (SPEC 7).  Leaving ``pbc`` unset infers it, and the inference is written so
    that ASE's gas-phase ``cell = zeros(3, 3)`` means **open boundaries** rather
    than a singular lattice; see :func:`resolve_pbc`.

    Returns:
        ``(energy, solve_result)``.  The solve result carries the moments and
        the charges for logging, loss terms on partial charges, and observables.
        With ``config.dispersion`` on, ``energy`` includes the SPEC 13
        dispersion; :func:`electrostatic_state` keeps the two apart.
    """
    state = electrostatic_state(
        h,
        pos,
        edge_index,
        edge_vec,
        edge_len,
        heads,
        config,
        pbc=pbc,
        cell=cell,
        edge_shift=edge_shift,
        fragment_id=fragment_id,
        fragment_charges=fragment_charges,
        total_charge=total_charge,
        external_field=external_field,
        fragment_spins=fragment_spins,
        quadrupoles=quadrupoles,
        metallic=metallic,
        electrode_potential=electrode_potential,
        species=species,
    )
    if state.dispersion is not None:
        # SPEC 13: the zero-point energy of the same moment block, added here so
        # the potential's single autograd call yields its forces and stress too.
        return state.energy + state.dispersion, state.solve_result
    return state.energy, state.solve_result


class ElectrostaticModule(nn.Module):
    """Holder for the heads and the config, callable from the potential.

    Exists so that the integration into ``TransformersACEV3.forward`` is three
    lines and carries no configuration of its own: the potential asks whether an
    electrostatics module is attached and, if so, adds its energy.  A model that
    was never explicitly given one -- which includes every existing checkpoint
    and every existing test -- has the attribute absent or ``None`` and follows
    exactly the code path it followed before.

    The heads are a submodule, so their parameters join the optimiser and the
    state dict in the ordinary way.
    """

    def __init__(
        self,
        heads: ElectrostaticHeads,
        config: ElectrostaticConfig,
        transfer_coupling: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, ElectrostaticConfig):
            raise TypeError(
                "config must be an ElectrostaticConfig; it is validated on "
                "construction and a bare dict would skip that validation."
            )
        config.validate_dtype()
        self.heads = heads
        self.config = config
        # SPEC 22: the learned through-bond coupling of diabatic charge states,
        # used when the batch asks for charge_state_coupling="through_bond".
        # Absent (None) on every model built without it, so their state dicts
        # are unchanged.
        self.transfer_coupling = transfer_coupling
        # Seed loss accumulated over training-mode forwards (SPEC 2.0.1); see
        # take_solver_seed_loss. Not a buffer: it is a graph, never state.
        self._pending_seed_loss: Tensor | None = None

    def resolve_inputs(
        self,
        pos: Tensor,
        data: Mapping[str, Any] | None,
        *,
        pbc: Any = None,
        external_field: Tensor | None = None,
        fragment_spins: Tensor | Sequence[float] | None = None,
        quadrupoles: bool | None = None,
    ) -> dict:
        """The keyword arguments of :func:`electrostatic_state` this module uses.

        The ONE place the batch dictionary is read, so the energy :meth:`forward`
        returns and any state rebuilt for an observable (``flashace`` rebuilds
        it for the dipole / polarizability / Born-charge / dielectric losses
        through :meth:`state`) are the same functional by construction.  (They
        were not: the module learned to switch the quadrupole sector on by
        default while the rebuild kept it off, and every observable loss on a
        quadrupole model failed its energy cross-check.)  Explicit keywords win
        over the dictionary.
        """
        fragment_id, fragment_charges, total_charge = _charge_spec(data)
        if pbc is None and data is not None:
            pbc = data.get("pbc")
        if fragment_spins is None and data is not None:
            fragment_spins = data.get("fragment_spins")
        if quadrupoles is None and data is not None and data.get("quadrupoles") is not None:
            quadrupoles = bool(data.get("quadrupoles"))
        if quadrupoles is None:
            # Heads built with the quadrupole stiffness carry the sector: switch it
            # on. (It used to default to off, so quadrupole heads attached to the
            # potential trained and ran with the sector silently absent.)
            quadrupoles = "A_Theta" in tuple(getattr(self.heads, "quadrupole_blocks", ()) or ())
        if external_field is None and data is not None and data.get("external_field") is not None:
            external_field = torch.as_tensor(
                data.get("external_field"), dtype=pos.dtype, device=pos.device
            ).reshape(3)
        return {
            "pbc": pbc,
            "fragment_id": fragment_id,
            "fragment_charges": fragment_charges,
            "total_charge": total_charge,
            "fragment_spins": fragment_spins,
            "quadrupoles": quadrupoles,
            "external_field": external_field,
            "metallic": None if data is None else data.get("metallic"),
            "electrode_potential": None if data is None else data.get("electrode_potential"),
            "species": None if data is None else data.get("z"),
        }

    def state(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_vec: Tensor,
        edge_len: Tensor,
        *,
        pbc: Any = None,
        cell: Tensor | None = None,
        edge_shift: Tensor | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> "ElectrostaticState":
        """The full :class:`ElectrostaticState` of the energy :meth:`forward` returns.

        Refused for a batch with ``charge_states``: the adiabatic energy is an
        eigenvalue of several solves, and no single assembled state is its
        operator, so a polarizability or Born charge taken from one state would
        not be the derivative of the energy the module returns.
        """
        if data is not None and data.get("charge_states") is not None:
            raise ValueError(
                "observables (dipole, polarizability, Born charges, dielectric tensor) "
                "are not defined from a single state for a structure with "
                "charge_states (SPEC 21): the energy is the lowest eigenvalue of "
                "several diabatic solves. Train such structures on energies and "
                "forces, or give their observable labels to structures without "
                "charge states."
            )
        return electrostatic_state(
            h, pos, edge_index, edge_vec, edge_len, self.heads, self.config,
            cell=cell, edge_shift=edge_shift, **self.resolve_inputs(pos, data, pbc=pbc),
        )

    def energies_by_structure(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_vec: Tensor,
        edge_len: Tensor,
        *,
        structure_id: Tensor,
        fragment_id: Tensor,
        fragment_charges: Tensor,
        species: Tensor | None = None,
        quadrupoles: bool | None = None,
    ) -> tuple[Tensor, "ElectrostaticState"]:
        """Energies of several independent open structures in ONE solve (SPEC 24).

        ``structure_id`` labels every atom's structure; ``fragment_id`` and
        ``fragment_charges`` are global over the batch and already follow each
        structure's own charge rules.  Returns ``([n_structures] energies,
        state)``: the energies, forces and every derivative of structure ``s``
        are those :meth:`forward` gives for ``s`` alone, to the solve tolerance.
        What this route does not carry -- fields, metals, charge states, spin,
        dispersion, periodic cells, the solver-seed loss -- is refused, never
        approximated.
        """
        if self.training and float(self.config.solver_seed_loss_weight) > 0.0:
            raise ValueError(
                "energies_by_structure does not accumulate the solver-seed loss; "
                "with solver_seed_loss_weight > 0 evaluate structures through forward"
            )
        if getattr(self.heads, "spin", False):
            raise NotImplementedError("energies_by_structure with spin heads (SPEC 24)")
        if quadrupoles is None:
            quadrupoles = "A_Theta" in tuple(getattr(self.heads, "quadrupole_blocks", ()) or ())
        state = electrostatic_state(
            h, pos, edge_index, edge_vec, edge_len, self.heads, self.config,
            pbc=(False, False, False), fragment_id=fragment_id,
            fragment_charges=fragment_charges, quadrupoles=quadrupoles,
            species=species, structure_id=structure_id,
        )
        return electrostatic_energies(state), state

    def take_solver_seed_loss(self) -> Tensor | None:
        """Return and clear the seed loss accumulated since the last call.

        In training mode every forward also evaluates
        :func:`solver_seed_loss` on the same inputs (scaled by
        ``config.solver_seed_loss_weight``) and adds it here. A training loop
        adds the returned tensor to its loss; it carries gradient only to the
        seed head's parameters. ``None`` when nothing was accumulated -- eval
        mode, a zero weight, or no forward since the last call.
        """
        loss, self._pending_seed_loss = self._pending_seed_loss, None
        return loss

    def forward(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_vec: Tensor,
        edge_len: Tensor,
        *,
        pbc: Any = None,
        cell: Tensor | None = None,
        edge_shift: Tensor | None = None,
        data: Mapping[str, Any] | None = None,
        external_field: Tensor | None = None,
        fragment_spins: Tensor | Sequence[float] | None = None,
        quadrupoles: bool | None = None,
    ) -> tuple[Tensor, SolveResult]:
        """Evaluate the electrostatic energy for one configuration.

        ``data`` is the batch dictionary; the optional keys ``fragment_id``,
        ``fragment_charges``, ``total_charge``, ``pbc``, ``fragment_spins``,
        ``quadrupoles``, ``external_field`` (``[3]``, V/A), ``metallic``
        (``[n_atoms]`` bool) and ``electrode_potential`` (``[n_atoms]``, V, NaN
        off-electrode; SPEC 20), and ``charge_states`` (a list of
        ``{"fragment_id", "fragment_charges"}`` diabatic states) with
        ``charge_state_coupling`` (SPEC 21) are read from it when present, so a charged, periodic, spin-polarised, quadrupolar or
        field-driven system needs no change to the call site.  ``quadrupoles``
        defaults to whether the heads carry the quadrupole stiffness.
        An explicit keyword wins over the batch dictionary, because a caller who
        names it is being specific about *this* call.

        ``pbc`` is the one that matters for existing callers: ``flashace.model``
        passes ``cell=cell`` unconditionally and ASE's gas-phase frame is
        ``zeros(3, 3)``, so leaving ``pbc`` unset has to keep meaning "infer",
        and the inference has to call that frame open.  It does; see
        :func:`resolve_pbc`.
        """
        inputs = self.resolve_inputs(
            pos, data, pbc=pbc, external_field=external_field,
            fragment_spins=fragment_spins, quadrupoles=quadrupoles,
        )
        pbc = inputs["pbc"]
        fragment_id, fragment_charges = inputs["fragment_id"], inputs["fragment_charges"]
        total_charge, fragment_spins = inputs["total_charge"], inputs["fragment_spins"]
        quadrupoles, external_field = inputs["quadrupoles"], inputs["external_field"]
        metallic, electrode_potential = inputs["metallic"], inputs["electrode_potential"]
        species = inputs["species"]
        charge_states = None if data is None else data.get("charge_states")
        if charge_states is not None:
            # SPEC 21: coupled diabatic charge states (EVB). The adiabatic energy
            # is the lowest eigenvalue; the returned SolveResult is that of the
            # most populated state.
            from .charge_states import adiabatic_charge_states

            # A dataset built on the CPU hands its states' labels as CPU tensors.
            charge_states = [
                {**spec, "fragment_id": torch.as_tensor(spec["fragment_id"], device=pos.device)}
                for spec in charge_states
            ]
            state_fn = functools.partial(
                electrostatic_state, h, pos, edge_index, edge_vec, edge_len, self.heads,
                self.config, pbc=pbc, cell=cell, edge_shift=edge_shift,
                external_field=external_field, fragment_spins=fragment_spins,
                quadrupoles=quadrupoles, metallic=metallic,
                electrode_potential=electrode_potential, species=species,
            )
            coupling = data.get("charge_state_coupling", 0.0)
            if isinstance(coupling, str):
                if coupling != "through_bond":
                    raise ValueError(
                        f"charge_state_coupling must be a number, an [n, n] tensor or "
                        f"'through_bond', got {coupling!r}"
                    )
                if self.transfer_coupling is None:
                    raise ValueError(
                        "charge_state_coupling='through_bond' needs a learned coupling "
                        "head: attach_electrostatics(..., transfer_coupling="
                        "TransferCouplingHead.for_heads(heads)) (SPEC 22)"
                    )
                from .transfer_coupling import coupling_matrix

                graph = self.transfer_coupling(self.heads.node_invariants(h), edge_index, edge_len)
                coupling = coupling_matrix(graph, charge_states)
            result = adiabatic_charge_states(state_fn, charge_states, coupling)
            dominant = int(torch.argmax(result.weights.detach()))
            return result.energy, result.states[dominant].solve_result
        weight = float(self.config.solver_seed_loss_weight)
        if self.training and weight > 0.0 and external_field is None:
            seed_term = weight * solver_seed_loss(
                h,
                pos,
                edge_index,
                edge_vec,
                edge_len,
                self.heads,
                self.config,
                pbc=pbc,
                cell=cell,
                edge_shift=edge_shift,
                fragment_id=fragment_id,
                fragment_charges=fragment_charges,
                total_charge=total_charge,
                fragment_spins=fragment_spins,
                quadrupoles=quadrupoles,
                metallic=metallic,
                electrode_potential=electrode_potential,
                species=species,
            )
            self._pending_seed_loss = (
                seed_term
                if self._pending_seed_loss is None
                else self._pending_seed_loss + seed_term
            )
        return electrostatic_energy(
            h,
            pos,
            edge_index,
            edge_vec,
            edge_len,
            self.heads,
            self.config,
            pbc=pbc,
            cell=cell,
            edge_shift=edge_shift,
            fragment_id=fragment_id,
            fragment_charges=fragment_charges,
            total_charge=total_charge,
            external_field=external_field,
            fragment_spins=fragment_spins,
            quadrupoles=quadrupoles,
            metallic=metallic,
            electrode_potential=electrode_potential,
            species=species,
        )


def attach_electrostatics(
    model: nn.Module,
    heads: ElectrostaticHeads,
    config: ElectrostaticConfig | None = None,
    transfer_coupling: nn.Module | None = None,
) -> ElectrostaticModule:
    """Turn electrostatics on for ``model``.  The only supported way to do so.

    Duck-typed on purpose: this function never imports the potential, it only
    sets an attribute.  That keeps the dependency one-way and means the
    electrostatics can be attached to anything with the right forward signature,
    including a test double.

    Returns the module it attached, so the caller can put its parameters in a
    separate optimiser group.
    """
    config = ElectrostaticConfig() if config is None else config
    # The dissociation envelope (bond_envelope) switches a bond off smoothly at
    # heads.r_max. That is only continuous if the neighbour list still contains
    # the bond when it reaches r_max: a model whose cutoff is SHORTER drops the
    # bond while the envelope is still > 0, which is the energy/force step the
    # envelope exists to remove. Duck-typed: a model with no r_max is not checked.
    model_cutoff = getattr(model, "r_max", None)
    if model_cutoff is not None and float(model_cutoff) < float(heads.r_max):
        raise ValueError(
            f"the model's neighbour cutoff r_max={float(model_cutoff)} A is shorter "
            f"than the electrostatic heads' bond cutoff r_max={float(heads.r_max)} A. "
            "Bonds would leave the neighbour list while their dissociation "
            "envelope is still non-zero, making the energy and forces "
            "discontinuous there. Construct ElectrostaticHeads(r_max=...) at or "
            "below the model cutoff."
        )
    if transfer_coupling is not None and model_cutoff is not None and float(model_cutoff) < float(
        transfer_coupling.r_max
    ):
        raise ValueError(
            f"the model's neighbour cutoff r_max={float(model_cutoff)} A is shorter than "
            f"the transfer-coupling cutoff r_max={float(transfer_coupling.r_max)} A; "
            "hoppings would vanish abruptly when bonds leave the neighbour list"
        )
    module = ElectrostaticModule(heads, config, transfer_coupling)
    setattr(model, ELECTROSTATICS_ATTR, module)
    return module


def detach_electrostatics(model: nn.Module) -> None:
    """Turn electrostatics off again, restoring the plain TRACE energy."""
    if getattr(model, ELECTROSTATICS_ATTR, None) is not None:
        setattr(model, ELECTROSTATICS_ATTR, None)


# ---------------------------------------------------------------------------
# Seams onto the sibling modules
#
# Every call into topology.py, solver.py and diagnostics.py happens in this
# section and nowhere else. If one of those signatures moves, this is the only
# place that has to follow it.
# ---------------------------------------------------------------------------


def _gaussian_cutoff_floor(widths, config: ElectrostaticConfig) -> float:
    """Smallest real-space cutoff at which every Gaussian pair has decayed.

    The widest pair width is ``s = sqrt(2) max(sigma)``; its real-space tail
    is ``erfc(r/s)``.  Returns ``s x`` with ``erfc(x) <= accuracy / 10`` (and
    never below ``4.5 s``, erfc(4.5) = 2e-10), matching the check in
    :func:`trace_electrostatics.lattice.gaussian_lattice_blocks`.
    """
    widest = math.sqrt(2.0) * max(float(w.detach().max()) for w in widths)
    target = 0.1 * float(config.ewald_accuracy)
    x = 4.5
    while math.erfc(x) > target and x < 12.0:
        x += 0.05
    return widest * x


class EnvelopeShellWarning(UserWarning):
    """First-shell bonds lie inside the dissociation envelope's switching shell."""


def _warn_if_first_shell_in_envelope_shell(bond_index, bond_len, n_atoms, heads, onset) -> None:
    """Warn when an atom's NEAREST bonded neighbour is inside the switching shell.

    SPEC 17.4: the shell envelope carries no bias only for bonds in its flat
    region (r <= onset * r_max).  A first-shell bond inside the shell gets a
    fixed-sign d ln t/dr from the switch, which contaminates the Born charges
    (rocksalt, NN 2.80 A of a 3.5 A cutoff: Z* = -5.1 for q = +1.1).  Cheap:
    one scatter-min over the bonds.  A host read, once per assembly.
    """
    if onset is None or int(bond_index.shape[1]) == 0:
        return
    flat = float(onset) * float(heads.r_max)
    length = bond_len.detach()
    own = bond_index[0] == bond_index[1]                  # self-image bonds
    length = torch.where(own, torch.full_like(length, float("inf")), length)
    nearest = torch.full((n_atoms,), float("inf"), dtype=length.dtype, device=length.device)
    nearest = nearest.scatter_reduce(0, bond_index[0], length, reduce="amin")
    nearest = nearest.scatter_reduce(0, bond_index[1], length, reduce="amin")
    in_shell = torch.isfinite(nearest) & (nearest > flat)
    if bool(in_shell.any()):
        warnings.warn(
            f"{int(in_shell.sum())} atom(s) have their nearest bonded neighbour at "
            f"{float(nearest[in_shell].min()):.2f}-{float(nearest[in_shell].max()):.2f} A, "
            f"inside the dissociation envelope's switching shell (onset "
            f"{flat:.2f} A = {float(onset)} x r_max {float(heads.r_max)} A). The switch "
            "then biases the bond-length dependence of charge transfer and the Born "
            "charges (SPEC 17.4). Use ElectrostaticHeads(r_max >= ~1.4 x the "
            "first-shell bond length) or a larger envelope_onset.",
            EnvelopeShellWarning,
            stacklevel=3,
        )


def bond_envelope(
    bond_length: Tensor, cutoff: float, onset: float | None = None
) -> Tensor:
    """Smooth dissociation envelope ``f(r)``: 1 inside, 0 at and beyond ``r_c``.

    SPEC 17.4 -- the default, ``onset`` = a fraction ``0 <= o < 1`` of the
    cutoff: ``f = 1`` for ``r <= o r_c`` and the C2 quintic switch

        f = 1 - x^3 (10 - 15 x + 6 x^2),   x = (r - o r_c) / ((1 - o) r_c)

    on the shell, with ``f``, ``f'`` and ``f''`` continuous at both ends.

    Why the switch moved into a shell.  The legacy envelope,
    ``(1 + cos(pi r / r_c)) / 2`` over the WHOLE range (``onset=None``), made
    the split charge ``t = f tau`` depend on the bond length at every bonding
    distance, through a factor the data never chose: at r = 2 A of a 5 A cutoff
    f = 0.66, the effective bond hardness kappa/f^2 = 2.3 kappa, and
    d ln t / dr = 2 f'/f = -0.9 per A.  That term enters every Born charge,
    ``Z* - q = sum_b (dt_b/dr) d_b``, with a fixed sign -- shortening a bond
    always INCREASED its charge transfer -- which is the opposite of the
    covalency trend behind the anomalous Born charges of oxides.  Measured on
    rocksalt with the nearest neighbour at 2.8 A of a 3.5 A cutoff (f = 0.095):
    q_Na = +0.24, Z*_Na = -0.90, confirmed by finite differences of the model's
    own polarization.  The bond-length dependence of charge transfer belongs to
    the learned, radially resolved bond hardness kappa(r); the envelope's only
    job is continuity at the cutoff, and a shell does that job without touching
    the bonding region -- PROVIDED the first-shell bonds lie below the onset
    (``r_max`` >= ~1.4x the nearest-neighbour distance at the default 0.75;
    SPEC 17.4 has the measured failure when they do not).  Pairs in the shell
    -- 3.75 to 5 A at the default -- are still attenuated smoothly, so
    non-bonded "bonds" do not act as full-weight parallel charge channels.

    Why the bond variable is ``t = f(r) tau`` rather than ``t`` itself
    ------------------------------------------------------------------
    A split charge ``t_ij`` exists only while ``(i, j)`` is in the neighbour
    list. With ``t`` as the variable it therefore vanished ABRUPTLY when a bond
    crossed the cutoff: two atoms in one neutral fragment, cutoff 5 A, measured
    E -1.68e-3 eV -> exactly 0 and F 1.0e-4 eV/A -> 0 across r = 5 A +- 1e-6,
    identical at k = 6 and k = 400. Every neighbour-list crossing in MD was an
    energy and force discontinuity. The bond hardness offered no protection: it
    was nearly flat in r (0.81 / 0.76 / 0.77 eV/e^2 at 1.5 / 3.0 / 4.5 A), which
    also let 3-4.5 A "bonds" act as parallel charge channels and stretched the
    SQE screening length N* = 2 eta / kappa_eff to ~190 sites on a chain.

    With ``t = f tau`` and the hardness kept as ``1/2 kappa tau^2``, in physical
    variables ``1/2 kappa tau^2 = 1/2 (kappa / f^2) t^2``: the effective bond
    hardness diverges as the bond stretches, which is the dissociation limit of
    split charge equilibration (Mathieu, J. Chem. Phys. 127, 224103 (2007)). In
    tau-space the diagonal stays kappa, so CG conditioning is unaffected --
    putting kappa -> infinity directly into K would have wrecked it.

    Continuity is EXACT, at any fixed k, not only at convergence: at f = 0 every
    coupling of tau_b is proportional to f, so with the enveloped seed its
    residual, its search direction and its share of every CG inner product are
    zero, and the iterates on all other coordinates are identical to those of
    the system without the bond. The energy near r_c goes as f^2 ~ (r_c - r)^4.

    The cutoff is the heads' ``r_max``, the bond model's own range. The caller's
    neighbour list MUST contain every pair out to it: a list that stops short
    removes bonds while f > 0, which is the discontinuity again.
    """
    if onset is None:
        x = (bond_length / cutoff).clamp(max=1.0)
        return 0.5 * (torch.cos(math.pi * x) + 1.0)
    start = float(onset) * float(cutoff)
    x = ((bond_length - start) / (float(cutoff) - start)).clamp(min=0.0, max=1.0)
    return 1.0 - x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


def _build_incidence(
    edge_index: Tensor,
    n_atoms: int,
    edge_len: Tensor,
    edge_shift: Tensor | None = None,
    sparse: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``build_incidence`` -> ``(B, bond_index, bond_slot, slot_sign)``.

    ``slot_sign`` is ``+1`` when the source edge of a bond runs in the stored
    ``lo -> hi`` orientation and ``-1`` otherwise, so the image-resolved bond
    vector is ``slot_sign * edge_vec[bond_slot]`` (SPEC 15).

    ``bond_slot`` indexes the bonds back into the directed edge arrays, so that
    bond lengths are *gathered* from the caller's ``edge_len`` rather than
    recomputed.  That matters under periodic boundary conditions, where
    recomputing from ``pos`` alone would silently drop the cell shift and give a
    bond the wrong length across the boundary.

    ``build_incidence`` returns a 2-tuple, so ``bond_slot`` is always derived
    here.  A previous version also accepted a 3-tuple and an attribute-carrying
    object, "in case the signature moves".  Those branches were dead, and a dead
    compatibility branch is worse than none: had the real return type changed to
    one of the shapes they accept, this seam would have quietly adapted instead
    of failing at the one place that knows what the contract is.  If
    ``build_incidence`` changes, change this call.

    The incidence matrix comes back at the global default dtype.  It is brought
    to the assembly's working dtype in
    :meth:`ElectrostaticAssembly.__post_init__`, which is the one place that
    knows what that dtype is -- it depends on the head parameters, and the heads
    have not run yet when this is called.
    """
    if edge_shift is not None:
        # Periodic: one bond per atom pair PER IMAGE. Without the shift the
        # topology keys bonds by atom pair alone, so a pair bonded through m
        # images -- routine in a small cell -- got ONE charge-transfer variable
        # for m physical channels, i.e. an effective bond hardness m times too
        # stiff (topology.py's own module docstring describes exactly this and
        # says to pass edge_shift; this seam never did). The energy then
        # depended on the choice of simulation cell: an 8-atom rocksalt cell
        # has 24 nearest-neighbour bonds but only 12 distinct pairs, and
        # measured E(2x2x2 supercell) - 8 E(primitive) = -5.5e-4 eV (2.6%)
        # with charges differing by 5.4e-4 e. DFT training data is mostly small
        # cells, so a model fitted there would not transfer to supercells.
        # build_bond_topology also supplies the bond slot: with shifts, each
        # bond has at most two source edges (two orientations of one image),
        # so it is unambiguous and _derive_bond_slot is not needed.
        # SPEC 15: self-image channels are KEPT. They carry the polarization
        # current a periodic crystal's dielectric response lives in, and with
        # no field they sit at exactly zero (zero column, zero seed, zero c).
        topology = build_bond_topology(
            edge_index,
            n_atoms,
            edge_shift=edge_shift,
            edge_len=edge_len,
            self_image_bonds=True,
            allow_no_bonds=True,
            sparse_incidence=sparse,
        )
        slot_sign = topology.edge_sign.index_select(0, topology.bond_slot)
        return topology.incidence, topology.bond_index, topology.bond_slot, slot_sign
    incidence, bond_index = build_incidence(edge_index, n_atoms)
    bond_slot = _derive_bond_slot(edge_index, bond_index, n_atoms, edge_len)
    forward = edge_index.index_select(1, bond_slot)[0] == bond_index[0]
    slot_sign = torch.where(forward, 1.0, -1.0).to(edge_len.dtype)
    return incidence, bond_index, bond_slot, slot_sign


def _derive_bond_slot(
    edge_index: Tensor, bond_index: Tensor, n_atoms: int, edge_len: Tensor
) -> Tensor:
    """For each bond, the directed edge it should gather its geometry from.

    ``build_incidence`` returns ``(incidence, bond_index)`` only, so the link
    back into the caller's ``edge_len`` has to be rebuilt here.  Recomputing
    bond geometry from ``pos`` instead would drop the periodic image shift and
    mismeasure every bond crossing the cell boundary -- which is why
    :func:`_bond_geometry` has no fallback that does so.

    Where several periodic images of the same pair map onto one bond, the
    **shortest** edge wins: that is the minimum image, and it is what the cutoff
    and the bond hardness are physically about.  The reduction below picks it
    deterministically; see the comment on ``scatter_reduce``.
    """
    slot, sign = map_edges_to_bonds(edge_index, bond_index, n_atoms)

    # Both orientations are valid sources of geometry; only self loops (sign 0)
    # are not.  Filtering to sign > 0 instead left any bond present solely as
    # (hi, lo) with its initialised slot of 0, silently gathering edge 0's
    # length -- measured [1.0, 1.0] where the truth was [1.0, 7.0] -- and if no
    # edge was forward oriented at all, EVERY bond took edge 0.
    valid = sign != 0
    edge_ids = torch.arange(slot.shape[0], device=slot.device)
    slot_v, ids_v = slot[valid], edge_ids[valid]

    n_bonds = int(bond_index.shape[1])
    if n_bonds == 0:
        return torch.zeros(0, dtype=torch.long, device=slot.device)
    if slot_v.numel() == 0:
        raise ValueError(
            f"{n_bonds} bonds were built but no non-self-loop edge maps onto "
            "any of them, so bond geometry cannot be gathered from the edge "
            "arrays. Check that edge_index and bond_index describe the same "
            "neighbour list."
        )

    lengths = edge_len.index_select(0, ids_v)

    # Deterministic minimum-image selection.
    #
    # The previous implementation sorted longest-first and relied on scatter_
    # "last write wins" with duplicate indices.  torch documents that as
    # NON-DETERMINISTIC, and it is: on MPS the selected length varied over
    # 1.4399 ... 3.5072 across runs, never reaching the true minimum image of
    # 1.0, while CPU happened to be stable.  With an ASE neighbour list at the
    # model cutoff EVERY bond takes the duplicate-index path on a periodic cell
    # (Cu fcc 28/28, NaCl 28/28), and bond_len feeds kappa through the radial
    # basis, so the ENERGY was run-to-run non-deterministic on GPU.
    #
    # scatter_reduce with amin is an order-independent reduction, so both steps
    # below are deterministic on every backend: first the minimum length per
    # bond, then the smallest edge index among the edges attaining it.
    infinity = torch.full((n_bonds,), float("inf"), dtype=lengths.dtype, device=slot.device)
    shortest = infinity.scatter_reduce(0, slot_v, lengths, reduce="amin", include_self=True)

    # Exact equality is correct here: the minimum is one of the values itself.
    attains = lengths == shortest.index_select(0, slot_v)
    sentinel = torch.iinfo(torch.long).max
    candidates = ids_v.masked_fill(~attains, sentinel)
    bond_slot = torch.full(
        (n_bonds,), sentinel, dtype=torch.long, device=slot.device
    ).scatter_reduce(0, slot_v, candidates, reduce="amin", include_self=True)

    if bool((bond_slot == sentinel).any()):
        missing = int((bond_slot == sentinel).sum())
        raise ValueError(
            f"{missing} of {n_bonds} bonds have no edge to gather geometry "
            "from. Every bond must appear in edge_index in at least one "
            "orientation."
        )
    return bond_slot


#: Fields ``solver.Geometry`` has to expose before the periodic route can hand
#: the solver an operator that matches the energy this module reports.  See
#: :func:`_solver_geometry`.
_PERIODIC_OVERRIDE_FIELDS = ("coulomb_matrix", "charge_dipole", "dipole_block")


def _geometry_fields() -> frozenset[str]:
    """Names ``solver.Geometry`` accepts, read from the dataclass itself."""
    return frozenset(field.name for field in dataclasses.fields(Geometry))


def _solver_options(config: ElectrostaticConfig) -> SolverOptions:
    """Translate the package config into the solver's own options.

    ``fragment_charge_tol`` is routed from ``config.constraint_tol`` because
    ``solve_charges`` checks SPEC 2.1 item 5 unconditionally, through
    ``diagnostics.assert_fragment_charges``, and that is the *only* place it is
    checked (see :func:`_run_solution_diagnostics` for what used to check it a
    second time and why that is gone).  Without this the configured tolerance
    was ignored and the solver's own default silently decided the question.

    ``check_constraints=False`` maps to an infinite tolerance, which is as close
    to "off" as this seam reaches: the residual is still formed -- it is one
    mat-vec, and ``SolveDiagnostics.constraint_violation`` reports it -- but it
    can no longer raise.  Turning the check off *entirely*, host
    synchronisation included, is a ``solve_charges`` decision, not one this
    module can make from outside.
    """
    return SolverOptions(
        thole_a=config.thole_a,
        multipole_kernel=config.multipole_kernel,
        gaussian_a=config.gaussian_a,
        fragment_charge_tol=(
            config.constraint_tol if config.check_constraints else float("inf")
        ),
        # SPEC 2.1 items 3 and 4. Without these two lines they were INERT on
        # every production path: this function forwarded only `thole_a` and
        # `fragment_charge_tol`, so `SolverOptions.validate` stayed at its `None`
        # default, `None` defers to the process-global validation mode, and
        # `diagnostics.set_validation_mode` is never called anywhere in this
        # package -- only defined. Setting `check_positive_definite=True` or
        # `check_is_minimum=True` therefore did nothing at all, silently.
        #
        # Measured on SPEC's own canonical trap (charge block +0.58, dipole
        # block +16.45, coupled Schur complement -0.2078): called the way this
        # module called it, `assert_positive_definite_at_solution` returned
        # without raising; called with the gate set, it raises. The saddle that
        # every docstring in this file cites as THE failure mode was passing.
        #
        # `check_is_minimum` is forwarded SEPARATELY rather than bundled into
        # `validate`, because SolverOptions' own docstring warns that item 4
        # asks a different question from item 3 -- and says, exactly: "A caller
        # that exposes item 3 through a config flag and leaves this one bundled
        # has item 4 silently off."
        validate=bool(config.check_positive_definite or config.check_is_minimum),
        check_is_minimum=config.check_is_minimum,
        eigenvalue_tol=config.positive_definite_tol,
        solve_mode=config.solve_mode,
        solve_rtol=config.solve_rtol,
        solve_max_iter=config.solve_max_iter,
    )


def _solver_geometry(assembly: "ElectrostaticAssembly") -> Geometry:
    """The :class:`~trace_electrostatics.solver.Geometry` for this assembly.

    Open boundaries
    ---------------
    Nothing to override.  ``assemble_system`` rebuilds the same damped kernels
    from the same positions with the same ``thole_a`` and the same
    polarizability floor (one floor now -- see ``MIN_POLARIZABILITY_VOLUME``),
    so the functional it minimises and the energy reported by
    :meth:`ElectrostaticAssembly.energy` agree to round-off: measured
    **8.9e-16** eV apart.

    Periodic boundaries -- the three overrides
    ------------------------------------------
    Under a cell the solver must be *handed* the operator, and all three blocks
    of it.  Left to itself, ``assemble_system`` computes its kernel from
    ``pos.unsqueeze(1) - pos.unsqueeze(0)`` -- raw, unwrapped coordinates, no
    cell, no minimum image, no lattice sum -- while the assembly reports the
    Ewald sum.  Measured on four atoms in an 8 A cell: the functional the solver
    minimised came to **6.49826491 eV** against a reported **5.98496125 eV**,
    **0.513 eV** apart, because one pair 7.401 A away in raw coordinates is
    0.616 A by minimum image and the solve used 7.401.  The solve equilibrated
    the charges of a *different physical system* from the one whose energy was
    returned -- quietly, with finite, plausible, differentiable numbers.

    Overriding only ``coulomb_matrix`` does not fix it and is not allowed:
    ``Geometry`` requires the three together, because a functional in which the
    charges see the lattice sum and the dipoles see raw open-boundary
    coordinates is no more self-consistent than the original.

    Each block is passed **complete, diagonal included**.  Under a lattice sum
    an atom interacts with its own periodic images, the neutralising background
    and its own screening cloud; that on-site term is of the same order as
    ``eta`` (about -2.5 eV at an 8 A cubic cell against ``eta ~ 11 eV``) and is
    a property of the cell, not of the atom.  It cannot be folded into ``eta``
    -- it is negative, and ``ElectrostaticParams`` would rightly reject the sum
    against the SPEC 1.2 hardness floor, which is a statement about the
    open-boundary pair kernel and not about a lattice sum.  ``assemble_system``
    still adds ``diag(eta)`` and ``A_mu`` on top, exactly as
    :meth:`ElectrostaticAssembly.energy` does.

    The blocks come from :func:`_periodic_pair_blocks`, which is also what the
    reported energy is evaluated from, so the two are the same operator rather
    than two derivations that have to be kept in step.
    """
    if not assembly.periodic and assembly.structure_id is None:
        return Geometry(
            positions=assembly.pos,
            incidence=assembly.incidence,
            bond_index=assembly.bond_index,
        )
    n_atoms = assembly.n_atoms
    quadrupole_overrides = {}
    if assembly.quadrupoles:
        n_theta = 5 * n_atoms
        quadrupole_overrides = {
            "charge_quadrupole": assembly.t02_pair.reshape(n_atoms, n_theta),
            "dipole_quadrupole": assembly.t12_pair.permute(0, 2, 1, 3).reshape(
                3 * n_atoms, n_theta
            ),
            "quadrupole_pair": assembly.t22_pair.permute(0, 2, 1, 3).reshape(
                n_theta, n_theta
            ),
        }

    missing = [
        name for name in _PERIODIC_OVERRIDE_FIELDS if name not in _geometry_fields()
    ]
    if missing:
        raise NotImplementedError(
            "the periodic route cannot be solved consistently: "
            f"solver.Geometry does not accept {missing}, so solve_charges would "
            "build its pair blocks from raw, unwrapped positions with no cell "
            "while this module reports the Ewald lattice sum. Measured 0.513 eV "
            "apart on four atoms in an 8 A cell (solver functional 6.49826491, "
            "reported energy 5.98496125), because a pair 0.616 A apart by "
            "minimum image is 7.401 A apart in raw coordinates and the solve "
            "used 7.401. Refusing rather than minimising one functional and "
            "reporting another. Open boundaries are unaffected and agree to "
            "8.9e-16 eV."
        )

    j_pair, t01_pair, t2_pair = assembly.j_pair, assembly.t01_pair, assembly.t2_pair
    if j_pair is None or t01_pair is None or t2_pair is None:
        raise ValueError(
            "a periodic assembly must carry its lattice-summed pair blocks; "
            "this one does not, so the solver would silently fall back to "
            "open-boundary kernels on unwrapped coordinates."
        )
    return Geometry(
        positions=assembly.pos,
        incidence=assembly.incidence,
        bond_index=assembly.bond_index,
        coulomb_matrix=j_pair,
        # The two reshapes are the ones assemble_system and dense_hessian both
        # use: charge_dipole[i, 3j + b] and dipole_block[3i + a, 3j + b].
        charge_dipole=t01_pair.reshape(n_atoms, 3 * n_atoms),
        dipole_block=t2_pair.permute(0, 2, 1, 3).reshape(3 * n_atoms, 3 * n_atoms),
        # A 2-D slab carries the Yeh-Berkowitz correction U = 2 pi K_E M_n^2 / V,
        # a rank-1 GLOBAL term in the total dipole.  Its charge-dipole block goes
        # as z_i alone, independent of j, so it is NOT pair-antisymmetric
        # (measured |W_ij + W_ji| = 4.36 where a Coulomb tensor gives exactly
        # 0.0).  K stays symmetric regardless -- the off-diagonal blocks are
        # built as transposes -- so declaring this waives a bug detector for
        # something that never claimed to be a pair tensor, not a correctness
        # condition.  3-D and open boundaries keep the check ON.
        # Also False under the Gaussian kernel (SPEC 14): charge i (width
        # sigma_i) with dipole j (width sigma_mu_j) is a different Coulomb
        # overlap from charge j with dipole i, so ell_i ell_j W_ij != -W_ji even
        # in open boundaries -- measured 8.7 % on a 16-atom rocksalt supercell.
        # K is symmetric regardless; only the Thole-specific detector is waived.
        # SPEC 24 (block-diagonal open assembly): the masked open blocks, which
        # are pair tensors under Thole exactly as unmasked ones are.
        charge_dipole_is_pair=(
            sum(bool(p) for p in assembly.pbc) != 2
            and assembly.config.multipole_kernel == "thole"
        ),
        # SPEC 14: which smeared density built these blocks.
        pair_kernel=assembly.config.multipole_kernel,
        # SPEC 15: the field couples through the image-resolved bond currents.
        bond_dipole=assembly.bond_dipole,
        **quadrupole_overrides,
    )


def _bond_geometry(edge_len: Tensor, bond_slot: Tensor) -> Tensor:
    """Bond lengths, gathered from the caller's ``edge_len``.

    One line, and deliberately only one.  This used to take ``pos``,
    ``edge_vec``, ``bond_index`` and a ``periodic`` flag as well, and to fall
    back to ``|pos[i] - pos[j]|`` when ``bond_slot`` was ``None``, raising
    instead under PBC where that fallback drops the image shift.  Since
    :func:`_derive_bond_slot` always produces a slot (and raises if it cannot),
    ``bond_slot is None`` became unreachable: the PBC guard could never fire,
    and the open-boundary fallback -- which is correct, but only by accident of
    ``edge_vec`` and ``pos`` agreeing -- was never used either.  Two dead
    branches around a gather, one of which read like a safety net.  The gather
    is the whole function.

    A bond vector is not returned because nothing needs one: the heads take
    ``bond_len``, and the polarization's ``dP/dt`` is ``B^T pos``, which
    :meth:`ElectrostaticAssembly.dipole_derivative` computes for itself.
    """
    return edge_len.index_select(0, bond_slot)


def _resolve_fragments(
    edge_index: Tensor,
    n_atoms: int,
    fragment_id: Tensor | None,
    fragment_charges: Tensor | Sequence[float] | None,
) -> tuple[Tensor, int]:
    """Fragment labels and a *static* fragment count.

    When the caller supplies labels, the count comes from ``fragment_charges``
    if given, and otherwise from a single reduction.  Supplying both from the
    data pipeline is what keeps the ahead-of-time graph free of host
    synchronisations; deriving them here is a convenience for interactive use,
    and it is the one ``.item()`` this module performs (see the module
    docstring).

    ``assign_fragments`` returns a plain ``[n_atoms]`` label tensor.  A branch
    that also accepted a ``(labels, count)`` tuple used to sit here; it never
    ran, and had the real return type ever become a tuple it would have taken
    the count from an unvalidated second element instead of failing at the seam.
    Removed.
    """
    if fragment_id is None:
        # topology.assign_fragments(num_atoms, bond_index) -- atom count FIRST.
        assigned = assign_fragments(n_atoms, edge_index)
        return assigned, int(assigned.max().item()) + 1

    if fragment_charges is not None and not isinstance(fragment_charges, Tensor):
        return fragment_id, len(fragment_charges)
    if isinstance(fragment_charges, Tensor):
        return fragment_id, int(fragment_charges.shape[0])
    return fragment_id, int(fragment_id.max().item()) + 1


def _resolve_fragment_charges(
    fragment_charges: Tensor | Sequence[float] | None,
    total_charge: float,
    n_fragments: int,
    reference: Tensor,
) -> Tensor:
    """Per-fragment target charges, ``[n_fragments]``.

    A single global constraint is size-inconsistent (SPEC 1.3), so there is no
    code path here that spreads a net charge over several fragments by guesswork:
    with more than one fragment and a non-zero net charge, the caller has to say
    which fragment carries it.
    """
    if fragment_charges is not None:
        charges = (
            fragment_charges
            if isinstance(fragment_charges, Tensor)
            else torch.tensor(fragment_charges, dtype=reference.dtype, device=reference.device)
        )
        charges = charges.to(dtype=reference.dtype, device=reference.device)
        if charges.shape != (n_fragments,):
            raise ValueError(
                f"fragment_charges has shape {tuple(charges.shape)}, expected "
                f"({n_fragments},) -- one target charge per fragment."
            )
        return charges

    if not isinstance(total_charge, (int, float)) or isinstance(total_charge, bool):
        raise TypeError(
            "total_charge must be a Python number: the net charge is a static "
            "property of the system, and reading it from a tensor would force a "
            "host synchronisation in the hot path."
        )
    if n_fragments == 1:
        return torch.full((1,), float(total_charge), dtype=reference.dtype, device=reference.device)
    if float(total_charge) != 0.0:
        raise ValueError(
            f"total_charge={total_charge} was given for a system with {n_fragments} "
            "fragments, but the constraint is per fragment and there is no "
            "size-consistent way to guess how the charge is shared. Pass "
            "fragment_charges with one entry per fragment."
        )
    return torch.zeros(n_fragments, dtype=reference.dtype, device=reference.device)


# ---------------------------------------------------------------------------
# SPEC 9 and SPEC 10 -- spin channels and quadrupoles
#
# Both are OPT-IN and default OFF, and both live behind this module rather than
# inside it: the spin-resolved charge sector is `solver`'s, the quadrupole
# couplings are `heads`', and what belongs here is the plumbing that carries the
# caller's request across. Every helper below is a no-op on the default path --
# `fragment_spins=None`, `quadrupoles=False` -- so nothing existing changes
# shape, dtype, value or the number of host synchronisations.
#
# Where a sibling does not yet accept what the caller asked for, these raise and
# name the keyword they looked for. That is the `_solver_geometry` pattern: a
# structural probe of the seam, and a refusal that says what has to move, rather
# than a silent fallback that computes a different model from the one requested.
# ---------------------------------------------------------------------------

#: Names ``ElectrostaticHeads.forward`` may use for the SPEC 10 opt-in.

#: Names ``FragmentConstraints`` may use for the SPEC 9 spin targets ``2 S_F``.
_SPIN_TARGET_FIELDS = ("spin_targets", "fragment_spins", "spins")


def _head_readout(
    heads: ElectrostaticHeads,
    h: Tensor,
    bond_index: Tensor,
    bond_len: Tensor,
    quadrupoles: bool,
    species: Tensor | None = None,
) -> Any:
    """Run the readout heads, checking the quadrupole request against what they carry.

    The default path is the bare ``heads(h, bond_index, bond_len)`` call this
    module has always made -- same arguments, same graph.

    SPEC 10 is a *detection* rule: whether a quadrupole block can exist is
    decided at head construction from the feature irreps, and a head that
    cannot represent one raises a named error there.  What is checked here is
    that the heads carry what the FUNCTIONAL consumes (SPEC 12):

    * ``A_Theta`` -- required.  It is the on-site quadratic cost of Theta;
      without it the sector is unbounded below (SPEC 10.1).  At TRACE's
      l_max = 2 it is the truncated 0e + 2e form
      (``allow_truncated_a_theta=True``), positive definite by construction.
    * ``chi_theta`` -- optional, the permanent quadrupole source.  Without it
      every quadrupole is induced, which is small; with it, sigma-holes and lone
      pairs have somewhere to live.
    * ``q_theta`` / ``mu_theta`` -- REFUSED.  They are ON-SITE couplings between
      ranks, an off-diagonal block inside each atom's own (q, mu, Theta) block.
      The Gram guarantee of SPEC 12 covers the inter-site couplings, not an
      unbounded on-site one, and such a block is exactly what took the coupled
      Hessian to -1.05 with every diagonal block positive (SPEC 2.1.3).  They
      are not wired until they have a parameterisation that is bounded by
      construction.
    """
    # The atomic numbers reach only heads that use them (the charge-transfer
    # distance law, SPEC 23); every other head is called exactly as before.
    keywords = {}
    if getattr(heads, "bond_hardness_law", "free") != "free" and getattr(heads, "charge_transfer_decay", None) is None:
        if species is None:
            raise ValueError(
                "these heads use bond_hardness_law='two_level', which needs the atomic numbers "
                "(pass species / data['z'])"
            )
        keywords["species"] = species
    if getattr(heads, "charge_transfer_decay", None) is not None:
        if species is None:
            raise ValueError(
                "these heads use the charge-transfer distance law (SPEC 23), which "
                "needs the atomic numbers: pass species= (the model hook reads "
                "data['z'])"
            )
        keywords["species"] = species
    if not quadrupoles:
        return heads(h, bond_index, bond_len, **keywords)

    blocks = tuple(getattr(heads, "quadrupole_blocks", ()) or ())
    if "A_Theta" not in blocks:
        raise ValueError(
            f"quadrupoles=True, but these heads carry blocks {blocks or '()'} and "
            "no A_Theta. The on-site stiffness is what makes the quadrupole sector "
            "bounded below (SPEC 10.1): the inter-site T22 coupling alone is "
            "indefinite. Construct ElectrostaticHeads(quadrupole_blocks=("
            "'A_Theta', 'chi_theta'), allow_truncated_a_theta=True) at l_max = 2."
        )
    unwired = [name for name in blocks if name in ("q_theta", "mu_theta")]
    if unwired:
        raise NotImplementedError(
            f"quadrupole blocks {unwired} are not part of the functional. They are "
            "on-site couplings between ranks, and nothing bounds them: the Gram "
            "construction of SPEC 12 guarantees a positive definite K for the "
            "inter-site couplings only, and an unbounded on-site off-diagonal "
            "block is exactly what drove the coupled Hessian to -1.05 with every "
            "diagonal block positive (SPEC 2.1.3). Build the heads with "
            "('A_Theta', 'chi_theta') only."
        )
    return heads(h, bond_index, bond_len, **keywords)


def _resolve_fragment_spins(
    fragment_spins: Tensor | Sequence[float] | None,
    n_fragments: int,
    reference: Tensor,
) -> Tensor | None:
    """Per-fragment total spins ``S_F`` (not ``2 S_F``), ``[n_fragments]``, or ``None``.

    SPEC 9 puts a *second* per-fragment constraint alongside the charge one,
    ``sum_{i in F} s_i = 2 S_F`` with ``s_i = q_i^up - q_i^down``.  It is per
    fragment for exactly the reason the charge constraint is: a single global
    spin constraint is size-inconsistent, and distant spectators would drain the
    polarisation of the fragment that is meant to carry it.  There is therefore
    no ``total_spin`` convenience here that spreads a multiplicity over several
    fragments by guesswork -- the caller says which fragment is the radical.

    ``None`` is the default and means the spin-unresolved model of SPEC 1.3.
    """
    if fragment_spins is None:
        return None
    spins = (
        fragment_spins
        if isinstance(fragment_spins, Tensor)
        else torch.tensor(
            fragment_spins, dtype=reference.dtype, device=reference.device
        )
    )
    spins = spins.to(dtype=reference.dtype, device=reference.device)
    if spins.shape != (n_fragments,):
        raise ValueError(
            f"fragment_spins has shape {tuple(spins.shape)}, expected "
            f"({n_fragments},) -- one total spin S_F per fragment (0, 1/2, 1, ...; "
            "not 2 S_F). SPEC 9's spin "
            "constraint is per fragment for the same reason the charge one is: "
            "a single global constraint is size-inconsistent."
        )
    return spins


def _fragment_constraints(assembly: "ElectrostaticAssembly") -> FragmentConstraints:
    """``FragmentConstraints`` for this assembly -- with the spin row if asked for.

    Without ``fragment_spins`` this is the two-argument construction this module
    has always made, unchanged.

    With them it is SPEC 9's second constraint, and it has to reach the solver
    through the same object as the first: two constraints applied by two
    mechanisms is how a projected CG ends up feasible in one of them only.  The
    spin sector is the solver's to implement -- ``q_i^sigma`` doubles the bond
    variables, the Coulomb kernel still couples TOTAL charges, and the on-site
    exchange ``0.5 J^x_i s_i^2`` needs ``J^x > 0`` or the spin mode is unbounded
    (SPEC 9.1) -- so this refuses rather than half-applying it.
    """
    if assembly.fragment_spins is None:
        return FragmentConstraints(
            matrix=assembly.constraint,
            targets=assembly.fragment_charges,
        )
    field = None
    available = frozenset(item.name for item in dataclasses.fields(FragmentConstraints))
    for name in _SPIN_TARGET_FIELDS:
        if name in available:
            field = name
            break
    if field is None:
        raise NotImplementedError(
            "fragment_spins were supplied, but solver.FragmentConstraints "
            f"accepts none of {_SPIN_TARGET_FIELDS} -- it carries "
            f"{sorted(available)}. SPEC 9 needs TWO per-fragment constraints, "
            "sum_i q_i = Q_F and sum_i s_i = 2 S_F, and they have to be applied "
            "by the same projection: a second constraint enforced outside the "
            "projected CG is satisfied at the seed and nowhere after it. The "
            "spin-resolved charge sector is solver.py's; this module already "
            "carries the targets from assemble() to here."
        )
    return FragmentConstraints(
        matrix=assembly.constraint,
        targets=assembly.fragment_charges,
        **{field: assembly.fragment_spins},
    )


# ---------------------------------------------------------------------------
# SPEC 15 -- handing the field to the solver
# ---------------------------------------------------------------------------


def _solve_matrix_free(
    assembly: "ElectrostaticAssembly",
    n_iter: int,
    external_field: Tensor | None,
    *,
    charge_field: Tensor | None = None,
    tau_seed: Tensor | None = None,
    solve_mode: str | None = None,
) -> SolveResult:
    """The SPEC 16 solve of a ``solver='matrix_free'`` assembly.

    ``solve_mode`` overrides the config's (the seed loss is a one-step fixed-k
    objective whatever the production energy is).
    """
    options = _solver_options(assembly.config)
    if solve_mode is not None:
        options = dataclasses.replace(options, solve_mode=solve_mode)
    return solve_lattice(
        assembly.lattice_operator,
        assembly.params,
        assembly.bond_index,
        assembly.envelope,
        _fragment_constraints(assembly),
        int(n_iter),
        tau_seed=assembly.t_seed if tau_seed is None else tau_seed,
        options=options,
        bond_dipole=assembly.bond_dipole,
        external_field=external_field,
        charge_field=charge_field,
    )


def _field_for_solver(
    assembly: "ElectrostaticAssembly",
) -> tuple[ElectrostaticParams, Tensor | None]:
    """``(params, field)`` to hand ``solve_charges``.

    Under open boundaries ``assemble_system`` couples the field as
    ``chi_i -> chi_i - E . r_i`` (charge) and ``-ell_i E`` (dipole).  Under
    periodic boundaries the Geometry carries ``bond_dipole`` and the charge
    sector couples through the bond currents instead (SPEC 15), in every
    direction, bounded by the bond hardness.  Nothing is split by sector any
    more, so this hands both through unchanged.  (SPEC 11's percolation
    projector, which it used to apply, is superseded: it existed only because
    ``sum_i q_i r_i`` cannot be coupled under PBC, and the bond-current
    polarization can.)
    """
    return assembly.params, assembly.external_field

def _moments(
    solve_result: SolveResult, assembly: "ElectrostaticAssembly"
) -> tuple[Tensor, ...]:
    """``SolveResult`` -> ``(t, rho, mu_hat)``, with mu REDUCED.

    ``SolveResult.mu`` is the *physical* dipole in e*A.  The assembly works in
    reduced moments ``mu_hat = mu / (e * ell)`` so that cross-rank quadratic
    forms are dimensionally homogeneous (SPEC 1.5).  Reading the raw ``mu`` as
    if it were reduced would be wrong by a factor of ``ell`` -- silently, since
    both are [n_atoms, 3] tensors of plausible magnitude.

    The division is therefore unconditional.  A preceding lookup for a
    ``mu_hat`` / ``mu_reduced`` attribute used to guard it; ``SolveResult`` has
    neither, so the lookup always missed -- and if one were ever added under a
    different reduction convention, that dead branch would have silently
    preferred it over the one line of arithmetic this function exists for.
    """
    ell = assembly.params.ell.unsqueeze(-1)
    moments = (solve_result.t, solve_result.rho, solve_result.mu / ell)
    if not assembly.quadrupoles:
        return moments
    # Same convention as ElectrostaticAssembly.unflatten_moments: a fourth entry
    # exactly when the quadrupole sector is on, so `assembly.energy(*moments)`
    # is right for both layouts.  theta_hat is the solver's own REDUCED vector,
    # not recovered from the cartesian theta: one reduction convention, not two.
    return moments + (solve_result.theta_hat,)


def _warn_if_unconverged(
    solve_result: "SolveResult",
    assembly: "ElectrostaticAssembly",
    config: ElectrostaticConfig,
) -> None:
    """Warn when the fixed-``k`` solve left the moments far from equilibrium.

    A warning and not an exception, deliberately.  SPEC 2 defines the energy as
    ``U(m_k)`` after exactly ``config.n_cg_iter`` conjugate-gradient steps, and
    the forces are the exact gradient of *that* functional, so a large residual
    does not make the energy inconsistent or the dynamics non-conservative.  It
    makes it a different functional from the equilibrated one -- and the
    difference is not a constant offset, which is the part worth saying out
    loud: the eigenvalue spread of K widens with system size, so k fixed CG
    steps leave a larger gap to ``U(m_inf)`` on a larger system (SPEC 2.0).  See
    ``ElectrostaticConfig.check_convergence`` for the measured growth.

    Reads :func:`~trace_electrostatics.observables.fixed_k_residual`, which is
    already computed by the solve; this adds one host sync and no arithmetic,
    which is why it is opt-in rather than always on.
    """
    from trace_electrostatics.observables import fixed_k_residual

    ratio = float(fixed_k_residual(solve_result))
    if ratio <= config.convergence_tol:
        return

    warnings.warn(
        f"the electrostatic solve is far from equilibrium: relative residual "
        f"{ratio:.3e} after n_cg_iter={config.n_cg_iter} steps, against "
        f"convergence_tol={config.convergence_tol:.1e}, on {assembly.n_atoms} "
        f"atoms and {assembly.n_bonds} bonds. This is not an inconsistency -- "
        f"SPEC 2 defines the energy as U(m_k) and the forces are its exact "
        f"gradient -- but U(m_k) is not U(m_inf), and the gap GROWS with system "
        f"size because the eigenvalue spread of K widens with N, so k fixed CG "
        f"steps converge less far. Heads fitted at one system size do not "
        f"transfer their compensation to another. Raise n_cg_iter, or train the "
        f"solver seed (SPEC 2.0.1).",
        RuntimeWarning,
        stacklevel=3,
    )


def _run_solution_diagnostics(
    assembly: ElectrostaticAssembly,
    t: Tensor,
    rho: Tensor,
    config: ElectrostaticConfig,
) -> None:
    """SPEC 2.1 assertions that can only be made once the moments are known.

    The positive-definiteness check is the coupled one: the charge block and the
    dipole block were separately positive in audit (+0.58 and +16.45) while the
    full system was indefinite at -1.05, so checking the blocks apart is worse
    than not checking at all.

    SPEC 2.1 item 5 (``|sum_{i in F} q_i - Q_F| < tol``) is **not** here.
    ``solve_charges`` runs it unconditionally through
    ``diagnostics.assert_fragment_charges``, on the same ``q`` and against the
    same targets, before it returns.  This function used to check it a second
    time with ``torch._assert_async`` -- a private torch API -- on the stated
    grounds that the asynchronous form avoids a host synchronisation.  It does
    not avoid anything: the solver's ``bool(worst > tol)`` has already
    synchronised by the time control reaches here, so the "sync-free" check ran
    strictly after the sync, added a private-API dependency, and reported a bare
    message where the solver names the offending fragment and its charge.  The
    tolerance is routed to the real check instead -- see ``_solver_options`` --
    so ``config.constraint_tol`` still means what it says.
    """
    if config.check_positive_definite:
        # SPEC 2.1 item 3.  This previously passed the FULL (t, rho, mu_hat)
        # Hessian and the full constraint operator into a signature expecting
        # the three separate blocks, so it raised TypeError at call time --
        # before the `enabled` gate -- and the check has therefore never once
        # executed in production.  No test covered it.
        #
        # The blocks are taken over the VARIATIONAL sector only (t and mu_hat);
        # rho is a fixed constraint carrier, not a moment that can relax.
        hessian = assembly.curvature()
        n_bonds, n_atoms = assembly.n_bonds, assembly.n_atoms
        mu_start = n_bonds + n_atoms
        assert_positive_definite_at_solution(
            hessian[:n_bonds, :n_bonds],
            hessian[mu_start:, :n_bonds],
            hessian[mu_start:, mu_start:],
            assembly.constraint_operator()[:, :n_bonds],
            tol=config.positive_definite_tol,
            # Explicit, not inherited. `enabled=None` defers to the global
            # validation mode, which is False at rest and never switched on in
            # this package, so this call was a no-op even inside its own
            # `if config.check_positive_definite:` guard.
            enabled=config.check_positive_definite,
        )


def _charge_spec(
    data: Mapping[str, Any] | None,
) -> tuple[Tensor | None, Tensor | None, Any]:
    """Pull the optional charge specification out of the batch dictionary.

    ``total_charge`` is passed through **as it was found**, deliberately not
    through ``float(...)``.  ``float()`` accepts a 0-d tensor, so it silently
    did two things this package says it will not do: it forced a host
    synchronisation in the hot path, and it made the ``TypeError`` guard in
    :func:`_resolve_fragment_charges` -- whose whole job is to reject a traced
    net charge -- unreachable from :meth:`ElectrostaticModule.forward`.  A batch
    dictionary carrying ``total_charge`` as a tensor now fails with that
    guard's explanation instead of quietly working, one sync per step, on CPU
    and then stalling on GPU.
    """
    if data is None:
        return None, None, 0.0
    return (
        data.get("fragment_id"),
        data.get("fragment_charges"),
        data.get("total_charge", 0.0),
    )
