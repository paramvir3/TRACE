"""Unit system for :mod:`trace_electrostatics` -- the single source of truth.

Everything in this package is expressed in the TRACE/ASE unit system

    energy   eV
    length   angstrom (A)
    charge   elementary charge (e)

from which the derived units follow:

    dipole moment           e * A
    electrostatic potential eV / e
    electric field          eV / (e * A)
    hardness / stiffness    eV / e^2
    Coulomb kernel J(r)     eV / e^2

The Coulomb constant in this system is

    K_E = 1 / (4 pi eps_0) = 14.399645 eV * A / e^2

so that the bare pair energy of two point charges is ``K_E q_i q_j / r_ij`` in eV.

THE UNIT TRAP (SPEC section 0)
------------------------------
A charge-equilibration polarizability comes out of the linear response as

    alpha_response = R^2 / (eta_A + eta_B)

With ``R`` in A and hardness ``eta`` in eV / e^2 this quantity has units of
**e^2 A^2 / eV**, *not* A^3.  It is a polarizability in "energy units"; it is
NOT a polarizability volume, and it must never be reported or compared against
literature values (which are in A^3) without conversion:

    alpha_volume [A^3] = K_E * alpha_response [e^2 A^2 / eV]

The factor is 14.4, so the mistake is a 14x error, not a subtle one.  Use
:func:`to_polarizability_volume` and :func:`to_response_polarizability` rather
than open-coding the factor; every place in this package that consumes a
polarizability states which of the two it wants.  In particular the Thole
damping in :mod:`trace_electrostatics.kernels` consumes polarizability
*volumes* (A^3), because only then is the reduced distance dimensionless.
"""

from __future__ import annotations

import math
from typing import Final, TypeVar

import torch

__all__ = [
    "ENERGY_UNIT",
    "LENGTH_UNIT",
    "CHARGE_UNIT",
    "DIPOLE_UNIT",
    "K_E",
    "SQRT_PI",
    "HBAR2_OVER_ME",
    "to_polarizability_volume",
    "to_response_polarizability",
]

ENERGY_UNIT: Final[str] = "eV"
LENGTH_UNIT: Final[str] = "A"
CHARGE_UNIT: Final[str] = "e"
DIPOLE_UNIT: Final[str] = "e*A"

#: Coulomb constant 1 / (4 pi eps_0) in eV * A / e^2.  This is the only place
#: this number is allowed to appear.
K_E: Final[float] = 14.399645

#: sqrt(pi), used by the Gaussian/erf kernels and by the Ewald self energy.
SQRT_PI: Final[float] = math.sqrt(math.pi)

#: hbar^2 / m_e in eV * A^2 (CODATA 2018: E_h a_0^2, with E_h = 27.211386245988 eV
#: and a_0 = 0.529177210903 A; the same two numbers give E_h a_0 = 14.3996455,
#: i.e. K_E above to its stated digits).  It is the Thomas-Reiche-Kuhn constant of
#: the dispersion sector (SPEC 13): an oscillator of response polarizability
#: alpha [e^2 A^2 / eV] carrying N electrons of oscillator strength has
#: (hbar omega)^2 = N * HBAR2_OVER_ME / alpha, so the inertia of an induced dipole
#: is 1 / (N * HBAR2_OVER_ME) with no fitted constant in it.
HBAR2_OVER_ME: Final[float] = 7.619964232

_ScalarOrTensor = TypeVar("_ScalarOrTensor", float, torch.Tensor)


def to_polarizability_volume(alpha_response: _ScalarOrTensor) -> _ScalarOrTensor:
    """Convert a response polarizability (e^2 A^2 / eV) to a volume (A^3).

    The linear response of this package returns ``alpha = D^T K^-1 D`` with
    ``K`` in eV / e^2 and ``D`` in A, i.e. e^2 A^2 / eV.  Literature
    polarizabilities are volumes in A^3.  See the module docstring.
    """
    return K_E * alpha_response


def to_response_polarizability(alpha_volume: _ScalarOrTensor) -> _ScalarOrTensor:
    """Convert a polarizability volume (A^3) to a response value (e^2 A^2 / eV).

    Inverse of :func:`to_polarizability_volume`.  Use this when a literature or
    fitted polarizability volume has to enter an energy expression.
    """
    return alpha_volume / K_E

#: Covalent radii in A by atomic number, Z = 0..96 (index 0 is a placeholder),
#: from B. Cordero et al., Dalton Trans. 2008, 2832 -- the table Grimme's D3
#: uses for coordination numbers (values as distributed in ase.data).  Used by
#: the optional charge-transfer distance law of the heads (SPEC 23): charge
#: transfer is cheap within covalent/ionic contact, R_i + R_j, and suppressed
#: beyond it.
COVALENT_RADII = (
    0.20, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57,  # Z = 0..9
    0.58, 1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03,  # Z = 10..19
    1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32,  # Z = 20..29
    1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90,  # Z = 30..39
    1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42,  # Z = 40..49
    1.39, 1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03,  # Z = 50..59
    2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90,  # Z = 60..69
    1.87, 1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36,  # Z = 70..79
    1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15,  # Z = 80..89
    2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69,  # Z = 90..96
)

# Atomic first ionization energies and electron affinities, eV, index = Z (entry 0 unused).
# Generated once from pymatgen's element data (NIST Atomic Spectra Database values) on
# 2026-10-03; moved here from trace_redox.constants on 2026-10-09 so that both packages read one
# source.  Vacuum values; a NEGATIVE affinity means the gas-phase anion is unbound.  Used by the
# two-level bond-hardness anchor (bond_law.py) and by trace_redox's charge-state anchors.
#: First ionization energy, eV, Z = 1..96.
ATOMIC_IONIZATION_EV = (
    0.000000, 13.598435, 24.587389, 5.391715, 9.322699, 8.298019,
    11.260288, 14.534130, 13.618055, 17.422820, 21.564541, 5.139077,
    7.646236, 5.985769, 8.151680, 10.486686, 10.360010, 12.967633,
    15.759612, 4.340664, 6.113155, 6.561490, 6.828120, 6.746187,
    6.766510, 7.434038, 7.902468, 7.881010, 7.639878, 7.726380,
    9.394197, 5.999302, 7.899435, 9.788550, 9.752392, 11.813810,
    13.999605, 4.177128, 5.694867, 6.217260, 6.634126, 6.758850,
    7.092430, 7.119380, 7.360500, 7.458900, 8.336839, 7.576234,
    8.993820, 5.786356, 7.343918, 8.608389, 9.009808, 10.451260,
    12.129844, 3.893906, 5.211665, 5.576900, 5.538600, 5.470200,
    5.525000, 5.581870, 5.643710, 5.670385, 6.149800, 5.863800,
    5.939050, 6.021500, 6.107700, 6.184310, 6.254160, 5.425871,
    6.825070, 7.549571, 7.864030, 7.833520, 8.438230, 8.967020,
    8.958830, 9.225554, 10.437504, 6.108287, 7.416680, 7.285516,
    8.418070, 9.317510, 10.748500, 4.072741, 5.278424, 5.380226,
    6.306700, 5.890000, 6.194050, 6.265540, 6.025760, 5.973810,
    5.991410,
)

#: Electron affinity, eV, Z = 1..96 (negative: unbound anion).
ATOMIC_ELECTRON_AFFINITY_EV = (
    0.000000, 0.754598, -0.520000, 0.618049, -0.520000, 0.279723,
    1.262114, -0.070000, 1.461105, 3.401190, -1.220000, 0.547926,
    -0.420000, 0.432835, 1.389521, 0.746607, 2.077105, 3.612725,
    -1.020000, 0.501459, 0.024551, 0.188200, 0.075545, 0.527662,
    0.675841, -0.520000, 0.153236, 0.662265, 1.157161, 1.235784,
    -0.620000, 0.301201, 1.232676, 0.804820, 2.020605, 3.363588,
    -1.020000, 0.485916, 0.052066, 0.307120, 0.433289, 0.917407,
    0.747330, 0.552000, 1.046382, 1.142892, 0.562141, 1.304473,
    -0.720000, 0.383926, 1.112070, 1.047401, 1.970876, 3.059052,
    -0.820000, 0.471630, 0.144626, 0.557546, 0.572000, 0.109235,
    0.097493, 0.129000, 0.162000, 0.116130, 0.137000, 0.131318,
    0.352000, 0.338000, 0.312000, 1.029220, -0.020000, 0.238870,
    0.178070, 0.323120, 0.816268, 0.060397, 1.077801, 1.564361,
    2.125105, 2.308610, -0.520000, 0.320053, 0.356721, 0.942362,
    1.407000, 2.415787, -0.720000, 0.486000, 0.100000, 0.350000,
    1.170000, 0.550000, 0.530000, 0.480000, -0.500000, 0.100000,
    0.280000,
)
