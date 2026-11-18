"""TRACE-Redox: free charge, the derivative discontinuity and the Fermi level (SPEC 25).

A separate package on purpose: ``trace_electrostatics`` (the SPEC 1 bound
sector) is used unchanged, through its public assembly, so every earlier
result stays reproducible and the two can be compared on the same data.

    from trace_redox import RedoxConfig, RedoxElectrostatics, attach_redox

Modules: :mod:`.ensemble` (Mermin ensemble over integer charge states),
:mod:`.free_solver` (exact concave-dual Newton solve with implicit
derivatives), :mod:`.bound` (exact Schur elimination of the bound sector),
:mod:`.heads` (charge-state cross-attention, attention-Wannier orbitals),
:mod:`.module` (the functional and the flashace seam).  Physics contract:
``trace_redox/SPEC.md``.  Float64 only: ``RedoxElectrostatics`` refuses
parameters of any other dtype (the rule of trace_electrostatics SPEC 3).
"""

from .ensemble import (
    conjugate_at, convex_envelope, grand_potential, mean_charge, occupations,
    potential_for_charge, susceptibility,
)
from .free_solver import (
    FreeChargeNotConvergedError, FreeChargeSolution, differentiable_solution, solve_free_charge,
)
from .bound import BoundElimination, BoundSectorError, charge_coulomb_block, eliminate_bound_sector, free_charge_blocks
from .heads import ChargeStateHead, FreeChargeWidth, OrbitalHead, smooth_switch
from .module import RedoxConfig, RedoxElectrostatics, RedoxResult, attach_redox, lambda_max

__all__ = [
    "BoundElimination", "BoundSectorError", "ChargeStateHead", "FreeChargeNotConvergedError",
    "FreeChargeWidth", "charge_coulomb_block", "free_charge_blocks",
    "FreeChargeSolution", "OrbitalHead", "RedoxConfig", "RedoxElectrostatics", "RedoxResult",
    "attach_redox", "conjugate_at", "convex_envelope", "differentiable_solution",
    "eliminate_bound_sector", "grand_potential", "lambda_max", "mean_charge", "occupations",
    "potential_for_charge", "smooth_switch", "solve_free_charge", "susceptibility",
]
