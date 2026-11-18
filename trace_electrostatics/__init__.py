"""Variational charge-equilibration electrostatics for TRACE.

A long-range, charge-transfer and induced-dipole energy term that is added to
the TRACE site energy before forces and stress are taken, so that both follow
from one differentiable scalar.  The model is split charge equilibration (SQE)
with induced dipoles (and optionally quadrupoles and many-body dispersion):
charges live on bonds and on sites, multipoles on sites, and the energy is the
constrained MINIMUM of one quadratic form, solved to a tolerance and
differentiated implicitly (SPEC 17), so forces, responses (polarizability,
Born charges, dielectric tensors) and training gradients are exact
derivatives of the minimiser.

UNITS -- eV, angstrom, e
------------------------
Energy in **eV**, length in **angstrom**, charge in **e**, and therefore dipole
moment in e*A, hardness in eV/e^2, and field in eV/(e*A).  The Coulomb constant
is ``K_E = 14.399645 eV*A/e^2`` and lives only in
:mod:`trace_electrostatics.constants`.

The unit trap worth stating twice: a charge-equilibration response
``R^2 / (eta_A + eta_B)`` is in e^2 A^2 / eV, **not** A^3.  It only becomes a
polarizability volume after multiplication by ``K_E``, a factor of 14.  Use
:func:`to_polarizability_volume` rather than open-coding it, and never compare a
raw response against a literature polarizability.

THE HARDNESS FLOOR (SPEC 1.2 / 17.3)
------------------------------------
The charge block ``diag(eta) + J`` is a Coulomb Gram matrix plus the on-site
residuals ``eta_i - S_0(sigma_i)``, with ``S_0 = sqrt(2) K_E / (sigma sqrt(pi))``
the self-energy of the atom's own Gaussian (7.66 eV at sigma = 1.5 A).  It is
positive definite for every geometry exactly when ``eta_i >= S_0``; the eta head
emits ``softplus(raw) + S_0 + margin`` (1 eV/e^2), and the check is asserted on
every parameter set.  Under the default Gaussian multipole kernel the full
(charge, dipole, quadrupole) operator is then positive definite by theorem
(SPEC 12, 14).  The Thole kernel has no such guarantee and needs
``ElectrostaticHeads(hardness_margin="legacy")``; the converged solve raises
``IndefiniteOperatorError`` rather than return a saddle.

Two further invariants of this package
--------------------------------------
* **One-way dependency.**  Nothing here imports ``flashace``.  The potential
  may import the electrostatics; the reverse would make this untestable on its
  own and would couple every checkpoint to it.
* **Opt-in, default off.**  A TRACE model has no electrostatics until
  :func:`attach_electrostatics` is called on it, so every existing checkpoint
  and every existing test follows exactly the path it followed before.  See
  ``docs/electrostatics.md`` for the three-line integration.

Double precision is required: call ``torch.set_default_dtype(torch.float64)``
before constructing any e3nn module, including :class:`ElectrostaticHeads`.
e3nn bakes the dtype into generated code at construction and a later
``.double()`` is a silent no-op.
"""

from __future__ import annotations

from .config import ElectrostaticConfig
from .constants import (
    CHARGE_UNIT,
    ENERGY_UNIT,
    K_E,
    LENGTH_UNIT,
    to_polarizability_volume,
    to_response_polarizability,
)
from .dispersion import dispersion_energy
from .energy import (
    ElectrostaticAssembly,
    ElectrostaticModule,
    ElectrostaticState,
    assemble,
    assembly_dispersion_energy,
    attach_electrostatics,
    detach_electrostatics,
    electrostatic_energies,
    electrostatic_energy,
    electrostatic_state,
)
from .heads import ElectrostaticHeads
from .implicit import IndefiniteOperatorError, SolverNotConvergedError
from .kernels import gram_self_energy, hardness_floor, kernel_self
from .open_operator import OpenOperator
from .charge_states import ChargeStateResult, adiabatic_charge_states
from .metals import METAL_BOND_HARDNESS
from .transfer_coupling import TransferCouplingHead, TransferGraph, bridge_coupling, coupling_matrix
from .observables import (
    acoustic_sum_rule_residual,
    bond_polarization,
    born_charges,
    born_charges_minimiser,
    dielectric_tensor,
    nonanalytic_force_constants,
    static_dielectric_tensor,
    hyperpolarizability,
    polarizability,
    polarizability_contributions,
    polarization,
)
from .solver import ElectrostaticParams, SolveResult

__all__ = [
    # units
    "ENERGY_UNIT",
    "LENGTH_UNIT",
    "CHARGE_UNIT",
    "K_E",
    "to_polarizability_volume",
    "to_response_polarizability",
    # configuration and integration
    "ElectrostaticConfig",
    "ElectrostaticHeads",
    "ElectrostaticModule",
    "attach_electrostatics",
    "detach_electrostatics",
    # energy
    "ElectrostaticAssembly",
    "ElectrostaticState",
    "assemble",
    "electrostatic_energy",
    "electrostatic_state",
    "electrostatic_energies",
    # SPEC 13: dispersion, the zero-point energy of the same moment block
    "dispersion_energy",
    "assembly_dispersion_energy",
    "ElectrostaticParams",
    "SolveResult",
    # observables
    "polarization",
    "polarizability",
    "polarizability_contributions",
    "born_charges",
    "acoustic_sum_rule_residual",
    # SPEC 15: periodic response
    "bond_polarization",
    "born_charges_minimiser",
    "dielectric_tensor",
    "static_dielectric_tensor",
    "nonanalytic_force_constants",
    "hyperpolarizability",
    # the hardness floor (SPEC 17.3) and the pair-kernel r -> 0 limit
    "gram_self_energy",
    "hardness_floor",
    "kernel_self",
    # SPEC 17 / 18: the converged solve's errors and the open matrix-free operator
    "IndefiniteOperatorError",
    "SolverNotConvergedError",
    "OpenOperator",
    # SPEC 20 / 21: metals and electrodes; coupled diabatic charge states
    "METAL_BOND_HARDNESS",
    # SPEC 22:
    "TransferCouplingHead",
    "TransferGraph",
    "bridge_coupling",
    "coupling_matrix",
    "ChargeStateResult",
    "adiabatic_charge_states",
]
