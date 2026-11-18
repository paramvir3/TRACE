"""Atomic first ionization energies and electron affinities, eV (anchors of SPEC 25).

Generated once from pymatgen's element data (NIST Atomic Spectra Database values)
on 2026-10-03 (pymatgen ``Element(Z).ionization_energy`` / ``.electron_affinity``;
no generator script is kept); index = Z,
entry 0 unused.  A NEGATIVE electron affinity means the anion is unbound in the
gas phase (He, Be, N, Mg, Mn, Zn, noble gases, ...): its energy lies above the
neutral's, which the ensemble of SPEC 25 represents exactly (that charge state
is then never populated unless the environment stabilises it).

These are VACUUM atomic values.  They anchor the learned charge-state energies
(``E_+1 = IE_Z + shift``, ``E_-1 = -EA_Z + shift``, |shift| bounded); the
electrostatic part of any environment shift is computed by the model, not
learned.
"""

from math import nan

from trace_electrostatics.constants import ATOMIC_ELECTRON_AFFINITY_EV, ATOMIC_IONIZATION_EV  # noqa: F401  (one source)



MAX_Z = 96


#: Successive ionization energies IE_1..IE_4, eV, Z = 0..96 (entry 0 unused; NaN
#: where the atom has fewer electrons).  pymatgen ``Element(Z).ionization_energies``
#: (NIST ASD), 2026-10-03; checked: Mg IE_2 15.035, Al IE_3 28.448, Fe IE_3 30.651.
#: The energy of the bare cation of charge +k relative to the neutral atom is
#: IE_1 + ... + IE_k (trace_redox anchors for ``max_charge`` up to 4).
MAX_CATION_CHARGE = 4
ATOMIC_SUCCESSIVE_IONIZATION_EV = (
    (nan, nan, nan, nan),
    (13.598435, nan, nan, nan),
    (24.587389, 54.417765, nan, nan),
    (5.391715, 75.640097, 122.454359, nan),
    (9.322699, 18.211150, 153.896205, 217.718586),
    (8.298019, 25.154830, 37.930590, 259.371500),
    (11.260288, 24.383154, 47.887780, 64.493520),
    (14.534130, 29.601250, 47.445300, 77.473500),
    (13.618055, 35.121120, 54.935540, 77.413500),
    (17.422820, 34.970810, 62.707980, 87.175000),
    (21.564541, 40.962970, 63.423300, 97.190000),
    (5.139077, 47.286360, 71.620000, 98.936000),
    (7.646236, 15.035271, 80.143600, 109.265400),
    (5.985769, 18.828550, 28.447642, 119.992400),
    (8.151680, 16.345850, 33.493000, 45.141790),
    (10.486686, 19.769490, 30.202640, 51.443870),
    (10.360010, 23.337880, 34.860000, 47.222000),
    (12.967633, 23.813640, 39.800000, 53.240000),
    (15.759612, 27.629670, 40.735000, 59.580000),
    (4.340664, 31.625000, 45.803100, 60.917000),
    (6.113155, 11.871719, 50.913160, 67.273200),
    (6.561490, 12.799770, 24.756839, 73.489400),
    (6.828120, 13.575500, 27.491710, 43.267170),
    (6.746187, 14.634000, 29.311100, 46.709000),
    (6.766510, 16.486305, 30.959000, 49.160000),
    (7.434038, 15.639990, 33.668000, 51.210000),
    (7.902468, 16.199210, 30.651000, 54.910000),
    (7.881010, 17.084400, 33.500000, 51.270000),
    (7.639878, 18.168838, 35.187000, 54.920000),
    (7.726380, 20.292390, 36.841000, 57.380000),
    (9.394197, 17.964390, 39.723300, 59.573000),
    (5.999302, 20.515140, 30.725760, 63.241000),
    (7.899435, 15.934610, 34.057600, 45.715500),
    (9.788550, 18.589200, 28.349000, 50.150000),
    (9.752392, 21.196000, 31.697000, 42.947000),
    (11.813810, 21.591000, 34.871000, 47.782000),
    (13.999605, 24.359840, 35.838000, 50.850000),
    (4.177128, 27.289540, 39.247000, 52.200000),
    (5.694867, 11.030276, 42.883530, 56.280000),
    (6.217260, 12.223600, 20.524410, 60.607200),
    (6.634126, 13.130000, 23.170000, 34.418360),
    (6.758850, 14.320000, 25.040000, 37.611000),
    (7.092430, 16.160000, 27.130000, 40.330000),
    (7.119380, 15.260000, 29.550000, 41.000000),
    (7.360500, 16.760000, 28.470000, 45.000000),
    (7.458900, 18.080000, 31.060000, 42.000000),
    (8.336839, 19.430000, 32.930000, 46.000000),
    (7.576234, 21.484400, 34.800000, 49.000000),
    (8.993820, 16.908313, 37.468000, 51.000000),
    (5.786356, 18.870410, 28.044150, 55.450000),
    (7.343918, 14.633070, 30.506000, 40.740000),
    (8.608389, 16.626000, 25.323500, 43.804000),
    (9.009808, 18.600000, 27.840000, 37.415500),
    (10.451260, 19.131260, 29.570000, 40.357000),
    (12.129844, 20.975000, 31.050000, 42.200000),
    (3.893906, 23.157450, 33.195000, 43.000000),
    (5.211665, 10.003826, 35.843800, 47.000000),
    (5.576900, 11.184960, 19.177300, 49.950000),
    (5.538600, 10.956000, 20.197400, 36.906000),
    (5.470200, 10.631000, 21.623700, 38.981000),
    (5.525000, 10.783000, 22.090000, 40.600000),
    (5.581870, 10.938000, 22.440000, 41.170000),
    (5.643710, 11.078000, 23.550000, 41.640000),
    (5.670385, 11.240000, 24.840000, 42.940000),
    (6.149800, 12.076000, 20.540000, 44.440000),
    (5.863800, 11.513000, 21.820000, 39.330000),
    (5.939050, 11.647000, 22.890000, 41.230000),
    (6.021500, 11.781000, 22.790000, 42.520000),
    (6.107700, 11.916000, 22.700000, 42.420000),
    (6.184310, 12.065000, 23.660000, 42.410000),
    (6.254160, 12.179185, 25.053000, 43.610000),
    (5.425871, 14.130000, 20.959400, 45.249000),
    (6.825070, 14.610000, 22.550000, 33.370000),
    (7.549571, 16.200000, 23.100000, 35.000000),
    (7.864030, 16.370000, 26.000000, 38.200000),
    (7.833520, 16.600000, 27.000000, 39.100000),
    (8.438230, 17.000000, 25.000000, 41.000000),
    (8.967020, 17.000000, 28.000000, 40.000000),
    (8.958830, 18.560000, 29.000000, 43.000000),
    (9.225554, 20.203000, 30.000000, 45.000000),
    (10.437504, 18.756880, 34.490000, 48.550000),
    (6.108287, 20.428300, 29.852000, 51.140000),
    (7.416680, 15.032499, 31.937300, 42.332560),
    (7.285516, 16.703000, 25.570750, 45.370000),
    (8.418070, 19.300000, 27.300000, 36.000000),
    (9.317510, 17.880000, 26.580000, 39.650000),
    (10.748500, 21.400000, 29.400000, 36.900000),
    (4.072741, 22.400000, 33.500000, 39.100000),
    (5.278424, 10.147180, 31.000000, 41.000000),
    (5.380226, 11.750000, 17.431000, 44.800000),
    (6.306700, 12.100000, 18.320000, 28.648000),
    (5.890000, 11.900000, 18.600000, 30.900000),
    (6.194050, 11.600000, 19.800000, 36.700000),
    (6.265540, 11.500000, 19.700000, 33.800000),
    (6.025760, 11.500000, 21.100000, 35.000000),
    (5.973810, 11.700000, 21.700000, 36.800000),
    (5.991410, 12.400000, 20.100000, 37.700000),
)


#: Covalent radii, A, Z = 0..96 (entry 0 unused): Cordero, Gomez, Platero-Prats,
#: Reves, Echeverria, Cremades, Barragan & Alvarez, "Covalent radii revisited",
#: Dalton Trans. 2008, 2832-2838 (via ase.data.covalent_radii; C is sp3, 0.76).
#: Used only to decide which atoms are chemically bonded (trace_redox fragments).
COVALENT_RADII_A = (
    0.20, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66,
    1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.70, 1.60, 1.53,
    1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20,
    1.16, 2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45,
    1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03,
    2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87,
    1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48,
    1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80,
    1.69,
)


#: Shannon crystal radii, 6-coordinate, A: R. D. Shannon, Acta Cryst. A32, 751 (1976), the
#: "crystal radius" column (effective radius + 0.14 A for cations, - 0.14 A for anions), which
#: Shannon recommended as the physical size of an ion in a solid.  Read from the published table
#: (via the Wikipedia "Ionic radius" table of Shannon's values) on 2026-10-09.  One ion per element,
#: named in the comment; elements absent here are REFUSED by ``width='ionic'`` rather than given an
#: invented size (extend from the source, ion by ion).  The free charge's sign is not fixed by the
#: table: the learned factor of :class:`trace_redox.heads.IonicFreeChargeWidth` spans x0.5..x2.
SHANNON_CRYSTAL_RADIUS = {
    3: 0.90,     # Li+
    8: 1.26,     # O2-
    9: 1.19,     # F-
    11: 1.16,    # Na+
    12: 0.86,    # Mg2+
    13: 0.675,   # Al3+
    17: 1.67,    # Cl-
    19: 1.52,    # K+
    20: 1.14,    # Ca2+
    47: 1.29,    # Ag+
    79: 1.51,    # Au+
}


#: The cost of the SECOND electron of a dianion, A_2 = E(X^2-) - E(X^-) - 0 (eV, endothermic),
#: where tabulated.  O: O^- + e^- -> O^2-, +844 kJ/mol = 8.75 eV (chemguide; Born-Haber estimates
#: span 780-844 kJ/mol, 8.08-8.75 eV).  Gas-phase O^2- is unbound; this is a Born-Haber value, used
#: only by the formal diabatic ladders (SPEC 17.5).  No other element has a -2 state.
SECOND_ELECTRON_COST = {8: 8.75}


#: The frame (core) charge of the core + carrier free charge (SPEC 17.9): the formal ion of the main-group
#: elements in the ionic model -- groups 1, 2, 13 lose their valence shell, groups 15, 16, 17 fill it
#: (ESTABLISHED: the ionic-model oxidation states).  Hydrogen and every other element: 0 (all of their free
#: charge is carrier).  Atomic numbers per group (Z <= 86).
_FRAME_GROUPS = {
    +1: (3, 11, 19, 37, 55),
    +2: (4, 12, 20, 38, 56),
    +3: (5, 13, 31, 49, 81),
    -3: (7, 15, 33, 51, 83),
    -2: (8, 16, 34, 52, 84),
    -1: (9, 17, 35, 53, 85),
}


def formal_frame(species):
    """``[n]`` integer frame charges (the compact ion core) of the core + carrier model."""
    import torch

    frame = torch.zeros_like(torch.as_tensor(species), dtype=torch.long)
    for charge, members in _FRAME_GROUPS.items():
        for z in members:
            frame = torch.where(torch.as_tensor(species) == z, torch.full_like(frame, charge), frame)
    return frame
