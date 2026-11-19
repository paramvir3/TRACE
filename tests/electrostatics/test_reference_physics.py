"""EXTERNAL physics references for ``trace_electrostatics``.

Every assertion in this file compares the package against a result that exists
independently of it: an exact theorem of lattice electrostatics or of coupled
harmonic oscillators, or a literature constant.  None of them is a consistency
check of the package against itself.  Every literature constant is ALSO
recomputed inside this file by code that imports nothing from the package (an
independent point-charge Ewald sum, Benson's series, an exact diagonalisation),
so a mis-remembered digit fails loudly instead of passing silently.

Units and conventions (SPEC 0)
------------------------------
eV, angstrom, e.  ``K_E = 14.399645 eV A / e^2`` is ``1/(4 pi eps_0)``, so a
Gaussian-units formula becomes a package formula by attaching one ``K_E`` to
every Coulomb interaction.  A polarizability VOLUME ``alpha`` (A^3) is the
Gaussian-units polarizability; the package's response polarizability is
``alpha / K_E`` (e^2 A^2 / eV) and its dipole stiffness is
``A_mu = K_E ell^2 / alpha``.  ``eps = 1 + 4 pi K_E alpha_cell / Omega`` with
``alpha_cell`` in response units (``observables.dielectric_tensor``).

How the package is driven
-------------------------
The learned heads are untrained, so they are replaced by a test double,
:class:`_PhysicalHeads`, that emits CHOSEN physical parameters (hardness,
Gaussian widths, polarizability volumes).  Everything downstream is the
production pipeline: ``energy.electrostatic_state`` (topology, dissociation
envelope, the SPEC 14 Gram lattice operator or the Thole Ewald route, the
per-fragment constraints, the solve), and the public observables.  Charge
transfer is switched off by giving the pipeline an EMPTY neighbour list: no
bonds and no self-image bonds, so the only variables are the induced dipoles.
Rigid ions are single-atom fragments whose charge is the fixed carrier ``rho``.

The solve is the converged one (SPEC 17 where present; otherwise 400 CG steps),
and response functions use ``route="minimiser"`` (a dense, exact solve), so no
number here depends on a truncated iteration.

Contents
--------
1. Madelung constants (rocksalt, CsCl, zincblende, fluorite, wurtzite) through
   the periodic charge machinery, with WIDE Gaussian charges and the Gaussian
   overlap subtracted exactly.
2. Clausius-Mossotti / Lorentz local field for sc, bcc, fcc lattices of
   polarizable points, the exact damped generalisation, the vanishing of the
   damping correction, and the two-sublattice (CsCl) coupled Lorentz field.
3. Szigeti's effective charge, the Lyddane-Herzfeld/Szigeti transverse force
   constant, and the Lyddane-Sachs-Teller relation.
4. Dispersion of quantum Drude oscillators: the exact dimer (London C6) and the
   Axilrod-Teller-Muto three-body energy with C9 = (9/16) hbar omega alpha^3.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import pytest
import torch

from conftest import require

torch.set_default_dtype(torch.float64)

K_E = 14.399645            # eV A / e^2, SPEC 0 (quoted, and checked below)
HBAR2_OVER_ME = 7.619964232  # eV A^2 = E_h a_0^2, SPEC 13 (quoted, and checked below)
SQRT_PI = math.sqrt(math.pi)
#: Gaussian dipole self volume of SPEC 12, V_1 = 3 sqrt(pi) / sqrt(2): the
#: Coulomb self energy of a unit Gaussian dipole of width sigma is
#: K_E / (V_1 sigma^3).  Re-derived here: -d^2/dz^2 [erf(r/s)/r] at r = 0 is
#: 4 / (3 sqrt(pi) s^3) with s = sqrt(2) sigma.
V1_DIPOLE = 3.0 * SQRT_PI / math.sqrt(2.0)


def test_the_quoted_unit_constants_are_the_packages():
    """The references below are written with the SPEC 0 / SPEC 13 constants;
    if the package ever moved them this file would be comparing different
    units, so pin them."""
    constants = require("constants")
    assert constants.K_E == K_E
    assert constants.HBAR2_OVER_ME == HBAR2_OVER_ME
    # K_E = E_h a_0 (CODATA 2018), HBAR2_OVER_ME = E_h a_0^2.
    hartree, bohr = 27.211386245988, 0.529177210903
    assert abs(hartree * bohr / K_E - 1.0) < 1e-7
    assert abs(hartree * bohr ** 2 / HBAR2_OVER_ME - 1.0) < 1e-9


# ===========================================================================
# Driving the production pipeline with chosen physical parameters
# ===========================================================================
class _PhysicalHeads:
    """Test double for ``ElectrostaticHeads``: returns fixed physical parameters.

    ``assemble`` calls ``heads(h, bond_index, bond_len)`` and reads
    ``heads.r_max`` (the dissociation-envelope cutoff) and ``heads.spin``.
    Nothing else of the heads is used, so the whole pipeline downstream runs
    unchanged on the parameters supplied here.
    """

    spin = False

    def __init__(self, params, electrons=None, r_max=0.5):
        self.params = params
        self.electrons = electrons
        self.r_max = r_max

    def __call__(self, h, bond_index, bond_len=None):
        ElectrostaticReadout, DispersionReadout = require(
            "heads", "ElectrostaticReadout", "DispersionReadout"
        )
        n_bonds = int(bond_index.shape[1])
        params = dataclasses.replace(self.params, kappa=torch.ones(n_bonds))
        dispersion = None if self.electrons is None else DispersionReadout(electrons=self.electrons)
        return ElectrostaticReadout(
            params=params, t_seed=torch.zeros(n_bonds), dispersion=dispersion
        )


def _params(alpha_volume, sigma):
    """Isotropic polarizable Gaussian ions: ``A_mu = K_E / alpha`` (ell = 1).

    ``eta = 2 K_E / (sigma sqrt(pi)) + 1 eV`` is above both the original SPEC 1.2
    floor and the SPEC 17.3 floor ``S_0 = sqrt(2) K_E / (sigma sqrt(pi))``; it
    never enters a reference, because the on-site term ``0.5 eta q^2`` is
    subtracted exactly where charges are fixed and there are no charge
    variables anywhere else.
    """
    ElectrostaticParams = require("solver", "ElectrostaticParams")
    alpha_volume = torch.as_tensor(alpha_volume, dtype=torch.float64).reshape(-1)
    sigma = torch.as_tensor(sigma, dtype=torch.float64).reshape(-1)
    n = max(alpha_volume.numel(), sigma.numel())
    alpha_volume = alpha_volume.expand(n).clone()
    sigma = sigma.expand(n).clone()
    return ElectrostaticParams(
        chi=torch.zeros(n),
        eta=2.0 * K_E / (sigma * SQRT_PI) + 1.0,
        sigma=sigma,
        ell=torch.ones(n),
        kappa=torch.zeros(0),
        A_mu=torch.eye(3).repeat(n, 1, 1) * (K_E / alpha_volume).view(-1, 1, 1),
    )


def _config(**kwargs):
    """A converged configuration, robust to the SPEC 17 transition.

    With ``solve_mode`` present the default is the converged solve; its
    tolerance is tightened to 1e-12 so that finite-difference force constants
    (first-order sensitive to the residual) are limited by the step, not the
    solve.  ``n_cg_iter = 400`` makes the legacy fixed-k route converged too.
    ``ewald_accuracy = 1e-14`` puts both Ewald truncations below every
    tolerance used here.
    """
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    fields = {f.name for f in dataclasses.fields(ElectrostaticConfig)}
    kwargs.setdefault("ewald_accuracy", 1e-14)
    kwargs.setdefault("ewald_real_cutoff", 14.0)
    if "n_cg_iter" in fields:
        kwargs.setdefault("n_cg_iter", 400)
    if "solve_rtol" in fields:
        kwargs.setdefault("solve_rtol", 1e-12)
    return ElectrostaticConfig(**kwargs)


def _kernel_config(kernel, damping, **kwargs):
    key = "gaussian_a" if kernel == "gaussian" else "thole_a"
    return _config(multipole_kernel=kernel, **{key: damping}, **kwargs)


def _crystal(cell, pos, charges, alpha_volume, sigma, config, field=None):
    """Rigid, polarizable ions in a 3-D periodic cell, no charge transfer.

    One fragment per ion, carrying its fixed charge; an empty neighbour list,
    so ``n_bonds == 0`` (no bond and no self-image bond); the dipoles are the
    only variables.
    """
    electrostatic_state = require("energy", "electrostatic_state")
    n = int(pos.shape[0])
    state = electrostatic_state(
        torch.zeros(n, 1), pos,
        torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 3), torch.zeros(0),
        _PhysicalHeads(_params(alpha_volume, sigma)), config,
        pbc=(True, True, True), cell=cell, edge_shift=torch.zeros(0, 3),
        fragment_id=torch.arange(n),
        fragment_charges=torch.as_tensor(charges, dtype=torch.float64),
        external_field=field,
    )
    assert state.assembly.n_bonds == 0
    return state


def _fcc_cell(a):
    return 0.5 * a * torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])


def _bcc_cell(a):
    return 0.5 * a * torch.tensor([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]])


# ===========================================================================
# Independent reference code -- imports nothing from the package
# ===========================================================================
def _plane_spacings(cell):
    volume = float(torch.det(cell).abs())
    return [
        volume / float(torch.linalg.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]).norm())
        for i in range(3)
    ]


def _lattice_vectors(cell, radius):
    """Every lattice vector ``n @ cell`` with ``|R| <= radius`` (and more)."""
    reach = [int(math.ceil(radius / s)) + 1 for s in _plane_spacings(cell)]
    n = torch.tensor(
        list(itertools.product(*[range(-m, m + 1) for m in reach])), dtype=torch.float64
    )
    return n @ cell


def _point_charge_ewald(pos, charges, cell, eta):
    """Tinfoil Ewald energy per cell of POINT charges, in e^2/A (no K_E).

    The textbook split (e.g. Allen & Tildesley, ch. 5):
        E = 1/2 sum'_{ijR} q_i q_j erfc(eta r)/r
          + (2 pi / V) sum_{k != 0} exp(-k^2/4 eta^2) |S(k)|^2 / k^2
          - (eta/sqrt(pi)) sum q_i^2 - pi Q^2 / (2 V eta^2).
    Cut-offs eta r = 6.5 (erfc = 4e-20) and k = 13 eta (exp(-k^2/4eta^2) =
    5e-19): both truncations are far below 1e-15 relative.
    """
    q = torch.as_tensor(charges, dtype=torch.float64)
    volume = float(torch.det(cell).abs())
    r_max = 6.5 / eta
    diameter = float(cell.sum(0).norm() + cell.norm(dim=1).sum())
    shifts = _lattice_vectors(cell, r_max + diameter)
    e_real = 0.0
    for i in range(len(q)):
        for j in range(len(q)):
            r = (pos[i] - pos[j] + shifts).norm(dim=1)
            r = r[r > 1e-12]
            e_real += 0.5 * float(q[i] * q[j]) * float((torch.special.erfc(eta * r) / r).sum())
    reciprocal = 2.0 * math.pi * torch.linalg.inv(cell).T
    k_max = 13.0 * eta
    reach = [int(math.ceil(k_max * float(cell[i].norm()) / (2.0 * math.pi))) + 1 for i in range(3)]
    m = torch.tensor(
        [v for v in itertools.product(*[range(-r, r + 1) for r in reach]) if any(v)],
        dtype=torch.float64,
    )
    k = m @ reciprocal
    k2 = (k * k).sum(1)
    phase = k @ pos.T
    s_re = (torch.cos(phase) * q).sum(1)
    s_im = (torch.sin(phase) * q).sum(1)
    e_recip = (2.0 * math.pi / volume) * float(
        (torch.exp(-k2 / (4.0 * eta ** 2)) / k2 * (s_re ** 2 + s_im ** 2)).sum()
    )
    e_self = -eta / SQRT_PI * float((q * q).sum())
    e_background = -math.pi * float(q.sum()) ** 2 / (2.0 * volume * eta ** 2)
    return e_real + e_recip + e_self + e_background


def _gaussian_overlap_energy(pos, charges, sigma, cell):
    """``sum_{i<j, images} q_i q_j K_E erfc(r / s_ij) / r``, s_ij^2 = sigma_i^2 + sigma_j^2.

    Two Gaussian charges ``exp(-r^2/sigma^2)`` interact through
    ``erf(r/s_ij)/r`` (SPEC 1.1); the difference from point charges is
    ``-erfc(r/s_ij)/r``.  Adding this sum to a Gaussian-charge lattice energy
    gives the point-charge energy EXACTLY (the tail is summed until erfc <
    1e-30).  The per-site self energy is not a pair term and is not in the
    package's pair blocks.
    """
    q = torch.as_tensor(charges, dtype=torch.float64)
    sigma = torch.as_tensor(sigma, dtype=torch.float64)
    total = 0.0
    shifts = _lattice_vectors(cell, 12.0 * float(sigma.max()) + float(cell.norm(dim=1).sum()))
    for i in range(len(q)):
        for j in range(len(q)):
            r = (pos[i] - pos[j] + shifts).norm(dim=1)
            r = r[r > 1e-12]
            s = math.sqrt(float(sigma[i]) ** 2 + float(sigma[j]) ** 2)
            total += 0.5 * float(q[i] * q[j]) * float((torch.special.erfc(r / s) / r).sum())
    return K_E * total


def _benson_rocksalt(n_terms=49):
    """Benson's rapidly convergent series for the rocksalt Madelung constant,

        M = 12 pi sum_{m, n odd >= 1} sech^2( (pi/2) sqrt(m^2 + n^2) ),

    (see Borwein, Glasser, McPhedran, Wan & Zucker, *Lattice Sums Then and
    Now*, CUP 2013, ch. 1).  A formula with no Ewald splitting in it at all;
    terms decay as exp(-pi sqrt(m^2+n^2)), so m, n <= 49 is exact to round-off.
    """
    total = 0.0
    for m in range(1, n_terms + 1, 2):
        for n in range(1, n_terms + 1, 2):
            total += 1.0 / math.cosh(0.5 * math.pi * math.sqrt(m * m + n * n)) ** 2
    return 12.0 * math.pi * total


# --- damping shell sums for the local-field references ----------------------
def _gaussian_dipole_width(alpha_volume, gaussian_a):
    """SPEC 12 width rule, sigma^3 = K_E ell^2 ||A^-2||_F^(1/2) / (a V_1).

    For the isotropic stiffness ``A = (K_E ell^2 / alpha) I`` this is
    ``3^(1/4) alpha / (a V_1)``, independent of ``ell``.
    """
    return (3.0 ** 0.25 * alpha_volume / (gaussian_a * V1_DIPOLE)) ** (1.0 / 3.0)


def _distances(cell, offset, radius):
    r = (_lattice_vectors(cell, radius + float(offset.norm())) + offset).norm(dim=1)
    return r[(r > 1e-9) & (r < radius)]


def _gaussian_shell_sum(cell, offset, width):
    """``W = sum_R (lambda_3 - lambda_5)(r) / r^3`` for the Gaussian pair width ``s``.

    For the erf kernel ``lambda_{2n+1} = P(n + 1/2, r^2/s^2)`` (SPEC 12) and
    ``P(5/2, x) = P(3/2, x) - x^(3/2) e^-x / Gamma(5/2)``, so
    ``(lambda_3 - lambda_5)/r^3 = 4 exp(-r^2/s^2) / (3 sqrt(pi) s^3)``.
    """
    r = _distances(cell, offset, 12.0 * width)
    return float(4.0 / (3.0 * SQRT_PI * width ** 3) * torch.exp(-(r / width) ** 2).sum())


def _thole_shell_sum(cell, offset, thole_a, alpha_i, alpha_j):
    """``W`` for Thole's exponential damping, ``v = a r^3 / sqrt(alpha_i alpha_j)``:
    ``lambda_3 = 1 - e^-v``, ``lambda_5 = 1 - (1 + v) e^-v`` (Thole, Chem. Phys.
    59, 341 (1981); SPEC 1.6), so ``(lambda_3 - lambda_5)/r^3 = (a/sqrt(alpha_i
    alpha_j)) e^-v``."""
    pair = math.sqrt(alpha_i * alpha_j)
    r = _distances(cell, offset, (80.0 * pair / thole_a) ** (1.0 / 3.0))
    return float(thole_a / pair * torch.exp(-thole_a * r ** 3 / pair).sum())


def _lorentz_sublattice_reference(cell, basis, alpha, kernel, damping, charges=None, sigma=None):
    """Exact uniform response of a crystal whose every site has cubic symmetry.

    Gaussian units (K_E = 1) until the force constant at the end.

    THEOREM (Lorentz).  For a site of cubic (O_h or T_d) symmetry the tinfoil
    lattice sum of the bare dipole tensor ``T = (I - 3 r r / r^2) / r^3`` over
    any sublattice is ``-(4 pi / 3 Omega) I``: cubic symmetry makes it
    isotropic, and its trace is fixed by ``nabla^2 (1/r) = -4 pi delta`` with
    the k = 0 (background) term removed.  Damping replaces ``T`` by
    ``(lambda_3 I - 3 lambda_5 r r / r^2) / r^3``; the difference decays
    exponentially, so it sums absolutely, and over every cubic shell
    ``sum r r / r^2 = (N/3) I``: each site adds ``(lambda_3 - lambda_5)/r^3``.
    Hence, with ``c = 4 pi / (3 Omega)``, the per-cell dipole stiffness of the
    uniform sublattice moments is

        M_tu = delta_tu / alpha_t - c + W_tu ,

    ``alpha_cell = 1^T M^-1 1`` and ``eps = 1 + 4 pi alpha_cell / Omega``.
    With W = 0, Sherman-Morrison gives ``alpha_cell = S / (1 - c S)``,
    ``S = sum_t alpha_t``: Clausius-Mossotti with the summed polarizability.

    Rigid ions ``q_t`` (two sublattices, ``q_2 = -q_1 = -q``): displacing
    sublattice 1 by ``u`` is, to first order, a dipole ``q_t' (relative
    displacement)`` on every ion of the other sublattice -- ``+q u`` in both
    cases -- so the field at sublattice t is ``g_t q u`` with
    ``g_t = c - W^(01)_t`` (charge-dipole damping), ``mu = M^-1 g q u`` and

        Z* / q = 1 + 1^T M^-1 g           (Szigeti: (eps + 2)/3 when W = 0)
        Phi_11 = K_E [ -q^2 (c - W^(00)) - q^2 g^T M^-1 g ]

    the first term of Phi being the point-ion Coulomb curvature
    ``q_1 q_2 sum nabla nabla phi`` and the second the induction energy
    ``-1/2 u^2 q^2 g^T M^-1 g``.  With W = 0, ``Phi_11 = -c K_E q Z*``
    (Lyddane-Herzfeld / Szigeti).

    Widths follow the model's definitions: Gaussian rank-pair widths
    ``s = sqrt(sigma_a^2 + sigma_b^2)`` with the SPEC 12 dipole width; Thole
    ``v`` keyed on the two polarizability volumes (its charge-dipole block uses
    the same ``lambda_3``, SPEC 1.6 / energy._periodic_pair_blocks); charges are
    erf-smeared on both routes.
    """
    omega = float(torch.det(cell).abs())
    c = 4.0 * math.pi / (3.0 * omega)
    n = len(basis)

    def w(t, u, rank):
        offset = basis[u] - basis[t]
        if rank == "00":
            return _gaussian_shell_sum(cell, offset, math.hypot(sigma[t], sigma[u]))
        if kernel == "thole":
            return _thole_shell_sum(cell, offset, damping, alpha[t], alpha[u])
        sigma_mu = [_gaussian_dipole_width(a, damping) for a in alpha]
        if rank == "11":
            return _gaussian_shell_sum(cell, offset, math.hypot(sigma_mu[t], sigma_mu[u]))
        return _gaussian_shell_sum(cell, offset, math.hypot(sigma[u], sigma_mu[t]))  # dipole t, charge u

    stiffness = torch.tensor([[(1.0 / alpha[t] if t == u else 0.0) - c + w(t, u, "11")
                               for u in range(n)] for t in range(n)])
    ones = torch.ones(n)
    alpha_cell = float(ones @ torch.linalg.solve(stiffness, ones))
    out = {"eps": 1.0 + 4.0 * math.pi * alpha_cell / omega,
           "correction": float(stiffness.diagonal().sum() - sum(1.0 / a for a in alpha) + n * c)}
    if charges is not None:
        q = abs(charges[0])
        g = torch.tensor([c - w(0, 1, "01"), c - w(1, 0, "01")])
        response = torch.linalg.solve(stiffness, g)
        out["born_over_q"] = 1.0 + float(ones @ response)
        out["phi"] = K_E * (-q * q * (c - w(0, 1, "00")) - q * q * float(g @ response))
    return out


# ===========================================================================
# 1.  Madelung constants
# ===========================================================================
def _structure(name):
    """Geometry, charges and convention of each structure.

    Convention, one for every structure: ``E_cell = - M K_E z_+ |z_-| n_fu / r_0``
    -- the electrostatic energy per formula unit of POINT ions, referred to the
    nearest-neighbour cation-anion distance ``r_0``.  For the 1:1 structures
    this is the energy per ion pair; for fluorite (z_+ = 2, z_- = -1) it is the
    per-CaF2 energy divided by z_+|z_-| = 2, which is the convention of the
    tabulated 2.51939.  Primitive cells throughout (the fcc and hexagonal ones
    are non-orthogonal).
    """
    if name == "rocksalt":
        a = 5.0
        return dict(cell=_fcc_cell(a), pos=torch.tensor([[0.0, 0, 0], [a / 2, 0, 0]]),
                    q=[1.0, -1.0], sigma=[0.8, 1.1], r0=a / 2, n_fu=1, zz=1.0)
    if name == "cesium_chloride":
        a = 4.0
        return dict(cell=a * torch.eye(3), pos=torch.tensor([[0.0, 0, 0], [a / 2, a / 2, a / 2]]),
                    q=[1.0, -1.0], sigma=[0.8, 1.1], r0=a * math.sqrt(3.0) / 2, n_fu=1, zz=1.0)
    if name == "zincblende":
        a = 5.0
        return dict(cell=_fcc_cell(a), pos=torch.tensor([[0.0, 0, 0], [a / 4, a / 4, a / 4]]),
                    q=[1.0, -1.0], sigma=[0.8, 1.1], r0=a * math.sqrt(3.0) / 4, n_fu=1, zz=1.0)
    if name == "fluorite":
        a = 5.0
        return dict(cell=_fcc_cell(a),
                    pos=torch.tensor([[0.0, 0, 0], [a / 4, a / 4, a / 4], [3 * a / 4, 3 * a / 4, 3 * a / 4]]),
                    q=[2.0, -1.0, -1.0], sigma=[0.8, 1.1, 1.1], r0=a * math.sqrt(3.0) / 4, n_fu=1, zz=2.0)
    if name == "wurtzite":
        # Ideal wurtzite: c/a = sqrt(8/3), u = 3/8, so all four bonds equal u c.
        a = 3.5
        c = a * math.sqrt(8.0 / 3.0)
        u = 3.0 / 8.0
        cell = torch.tensor([[a, 0.0, 0.0], [-a / 2, a * math.sqrt(3.0) / 2, 0.0], [0.0, 0.0, c]])
        frac = torch.tensor([[1 / 3, 2 / 3, 0.0], [2 / 3, 1 / 3, 0.5],
                             [1 / 3, 2 / 3, u], [2 / 3, 1 / 3, 0.5 + u]])
        return dict(cell=cell, pos=frac @ cell, q=[1.0, 1.0, -1.0, -1.0],
                    sigma=[0.8, 0.8, 1.1, 1.1], r0=u * c, n_fu=2, zz=1.0)
    raise KeyError(name)


#: Literature values, quoted only to digits of which this file is certain; each
#: is ALSO recomputed below by the independent Ewald sum.
#:   rocksalt 1.747564594633, CsCl 1.762674773, zincblende 1.638055053: the
#:   high-precision lattice-sum values (Borwein et al., *Lattice Sums Then and
#:   Now*, 2013; Madelung, Phys. Z. 19, 524 (1918) for the method);
#:   fluorite 2.51939 and ideal wurtzite 1.64132: the standard tabulation in
#:   this convention (e.g. CRC Handbook of Chemistry and Physics, "Madelung
#:   constants"; Johnson & Templeton, J. Chem. Phys. 34, 2004 (1961)).
MADELUNG_LITERATURE = {
    "rocksalt": ("1.747564594633", 1.747564594633),
    "cesium_chloride": ("1.762674773", 1.762674773),
    "zincblende": ("1.638055053", 1.638055053),
    "fluorite": ("2.51939", 2.51939),
    "wurtzite": ("1.64132", 1.64132),
}
STRUCTURES = tuple(MADELUNG_LITERATURE)


@pytest.fixture(scope="module")
def independent_madelung():
    """``M`` of every structure from the independent point-charge Ewald sum."""
    values = {}
    for name in STRUCTURES:
        s = _structure(name)
        energy = _point_charge_ewald(s["pos"], s["q"], s["cell"], eta=0.9)
        values[name] = -energy * s["r0"] / (s["n_fu"] * s["zz"])
    return values


@pytest.mark.parametrize("name", STRUCTURES)
def test_independent_ewald_reproduces_the_literature_madelung_constant(name, independent_madelung):
    """Physics: the Madelung constant is a pure geometric property of the
    lattice (Madelung 1918; Ewald, Ann. Phys. 369, 253 (1921)).  Formula: the
    tinfoil Ewald sum written above, which shares no code with the package.

    Checks, in order: the sum is independent of its splitting parameter (the
    defining property of a correct Ewald split, 1e-13 relative: both
    truncations are < 1e-18 and the rest is summation round-off); the
    geometry really has r_0 as its shortest cation-anion distance; and the
    result equals the literature constant to half a unit in its last quoted
    digit.  Independent values: 1.747564594633182 (rocksalt), 1.762674773070988
    (CsCl), 1.638055053388789 (zincblende), 2.519392439924283 (fluorite),
    1.641321627371949 (ideal wurtzite).
    """
    s = _structure(name)
    other = -_point_charge_ewald(s["pos"], s["q"], s["cell"], eta=1.4) * s["r0"] / (s["n_fu"] * s["zz"])
    value = independent_madelung[name]
    assert abs(other - value) < 1e-13 * value, (name, value, other)

    q = torch.tensor(s["q"])
    shortest = min(
        float(_distances(s["cell"], s["pos"][j] - s["pos"][i], 20.0).min())
        for i in range(len(q)) for j in range(len(q)) if float(q[i] * q[j]) < 0
    )
    assert abs(shortest - s["r0"]) < 1e-12, (name, shortest, s["r0"])

    quoted, literature = MADELUNG_LITERATURE[name]
    half_ulp = 0.5 * 10.0 ** -(len(quoted.split(".")[1]))
    assert abs(value - literature) <= half_ulp + 1e-15, (name, value, quoted)


def test_benson_series_confirms_the_rocksalt_constant(independent_madelung):
    """Benson's sech^2 series has no Ewald split in it; it must agree with the
    independent Ewald value to round-off (measured 1.6e-15)."""
    assert abs(_benson_rocksalt() - independent_madelung["rocksalt"]) < 1e-14


@pytest.mark.parametrize("kernel", ["gaussian", "thole"])
@pytest.mark.parametrize("name", STRUCTURES)
def test_package_lattice_energy_gives_the_madelung_constant(name, kernel, independent_madelung):
    """The package's periodic charge machinery reproduces the Madelung constant.

    Physics: for point ions ``E_cell = -M K_E z_+|z_-| n_fu / r_0`` (convention
    of :func:`_structure`).  Route: rigid ions as single-atom fragments, no
    bonds, through ``electrostatic_state`` -- the SPEC 14 Gram lattice operator
    (``kernel="gaussian"``) or the Thole-route Ewald blocks
    (``energy._periodic_pair_blocks``, ``kernel="thole"``).

    The ions are deliberately WIDE, unequal Gaussians (sigma = 0.8 / 1.1 A
    against r_0 = 2.2-2.5 A), so the Gaussian lattice energy is 0.1-6 % away
    from the point-ion one; the exact overlap correction
    ``sum q_i q_j K_E erfc(r/s_ij)/r`` (independent code, SPEC 1.1) is then
    added.  This tests the Gaussian smearing and the lattice sum together.
    (The SPEC 14 ``s_ij^2`` Gram background term sums to exactly zero for any
    neutral cell, ``2 Q sum_i q_i sigma_i^2``, so no neutral-cell energy -- this
    one included -- can see it; verified by deleting it at run time: all pass.)
    The on-site ``0.5 eta q^2`` is subtracted exactly.

    ``assembly.energy`` is evaluated at the fixed charges with the dipoles
    clamped to zero (the pure charge lattice energy).  At the cubic and
    tetrahedral sites of the first four structures the field vanishes by
    symmetry, so the SOLVED dipoles must vanish and the solved energy must be
    the same number; wurtzite's C3v sites admit a polar axis, so there the
    solve can only LOWER the energy (induction is variational).

    Tolerance 1e-11 relative on M: the Ewald accuracy is 1e-14 and the rest is
    summation round-off (measured <= 4e-15, both kernels, all five).
    """
    s = _structure(name)
    alpha = 1e-9  # A^3: ions all but unpolarizable; the induced energy on
    #               wurtzite is then ~1e-14 of the lattice energy
    state = _crystal(s["cell"], s["pos"], s["q"], alpha, s["sigma"], _kernel_config(kernel, 0.5))
    assembly = state.assembly
    q = torch.tensor(s["q"])
    n = len(s["q"])
    charge_only = float(assembly.energy(torch.zeros(0), q, torch.zeros(n, 3)))
    onsite = 0.5 * float((assembly.params.eta * q * q).sum())
    gaussian_lattice = charge_only - onsite
    point_lattice = gaussian_lattice + _gaussian_overlap_energy(s["pos"], s["q"], s["sigma"], s["cell"])

    def madelung(energy):
        return -energy * s["r0"] / (K_E * s["n_fu"] * s["zz"])

    reference = independent_madelung[name]
    assert abs(madelung(point_lattice) / reference - 1.0) < 1e-11, (
        f"{name}/{kernel}: M = {madelung(point_lattice):.15f}, independent {reference:.15f}"
    )
    # the overlap correction was doing real work, so the smearing was tested
    assert abs(madelung(gaussian_lattice) / reference - 1.0) > 1e-3

    solved = float(state.energy)
    if name == "wurtzite":
        assert solved <= charge_only + 1e-12 * abs(charge_only)
        assert abs(solved - charge_only) < 1e-12 * abs(charge_only)
    else:
        assert float(state.solve_result.mu.abs().max()) < 1e-12
        assert abs(solved - charge_only) < 1e-12 * abs(charge_only)


# ===========================================================================
# 2.  Clausius-Mossotti and the Lorentz local field
# ===========================================================================
LATTICE_SPACING = 3.0
BRAVAIS = {
    "sc": LATTICE_SPACING * torch.eye(3),
    "bcc": _bcc_cell(LATTICE_SPACING),
    "fcc": _fcc_cell(LATTICE_SPACING),
}


def _alpha_for(x, cell):
    """The polarizability volume that puts the CM right-hand side at ``x``:
    ``4 pi alpha / (3 Omega) = x``."""
    return 3.0 * x * float(torch.det(cell).abs()) / (4.0 * math.pi)


def _monatomic_eps(cell, alpha, kernel, damping):
    dielectric_tensor = require("observables", "dielectric_tensor")
    state = _crystal(cell, torch.zeros(1, 3), [0.0], alpha, 0.25, _kernel_config(kernel, damping))
    return dielectric_tensor(state.solve_result, state.assembly, route="minimiser"), state


# (kernel, damping parameter, x): chosen so that the computed damping
# correction alpha W is < 1e-14, i.e. bare point dipoles to round-off.
NEGLIGIBLE_DAMPING = [("gaussian", 0.99, 0.02), ("thole", 0.9, 0.10)]


@pytest.mark.parametrize("kernel, damping, x", NEGLIGIBLE_DAMPING)
@pytest.mark.parametrize("lattice", list(BRAVAIS))
def test_clausius_mossotti_for_cubic_bravais_lattices(lattice, kernel, damping, x):
    """Clausius-Mossotti, ``(eps - 1)/(eps + 2) = 4 pi alpha / (3 Omega)``.

    Physics: a cubic Bravais lattice of identical isotropic polarizable POINT
    dipoles feels the Lorentz local field ``E_loc = E + 4 pi P / 3``
    (Lorentz 1880; Jackson, *Classical Electrodynamics*, 3rd ed., sec. 4.5;
    Kittel, *Introduction to Solid State Physics*, ch. 15), which gives CM
    exactly.  ``alpha`` is a polarizability VOLUME (A^3); the package's
    stiffness is ``A_mu = K_E / alpha`` (SPEC 0: the response polarizability is
    alpha / K_E, and ``eps = 1 + 4 pi K_E alpha_resp / Omega``).

    Route: ``observables.dielectric_tensor`` (SPEC 15) on a one-site PRIMITIVE
    cell (sc, bcc, fcc -- the latter two non-orthogonal), charge transfer off
    (no bonds of any kind), minimiser route.  Cross-checked against a
    finite-field solve through the production enthalpy: ``eps = 1 + 4 pi K_E
    (Omega P) / (Omega E)`` from ``bond_polarization``.

    Damping: the parameters make the exact damping correction ``alpha W``
    (reference above) < 1e-14, asserted -- 100x below the CM tolerance -- so
    the operator IS the bare point-dipole lattice to round-off.  Tolerance 1e-12 on the CM ratio (dense
    solves and a 1e-14 Ewald accuracy; measured <= 2.5e-15), 1e-10 on the
    finite-field route (the converged solve's 1e-12 tolerance).
    """
    bond_polarization = require("observables", "bond_polarization")
    cell = BRAVAIS[lattice]
    alpha = _alpha_for(x, cell)
    reference = _lorentz_sublattice_reference(cell, [torch.zeros(3)], [alpha], kernel, damping)
    assert abs(alpha * reference["correction"]) < 1e-14

    eps_tensor, _state = _monatomic_eps(cell, alpha, kernel, damping)
    eps = float(eps_tensor[2, 2])
    assert float((eps_tensor - eps * torch.eye(3)).abs().max()) < 1e-13  # cubic => isotropic
    assert abs(((eps - 1.0) / (eps + 2.0)) / x - 1.0) < 1e-12, (lattice, kernel, eps)

    field = torch.tensor([0.0, 0.0, 0.01])
    fielded = _crystal(cell, torch.zeros(1, 3), [0.0], alpha, 0.25,
                       _kernel_config(kernel, damping), field=field)
    cell_dipole = bond_polarization(fielded.solve_result, fielded.assembly)
    eps_field = 1.0 + 4.0 * math.pi * K_E * float(cell_dipole[2]) / (float(torch.det(cell).abs()) * 0.01)
    assert abs(eps_field / eps - 1.0) < 1e-10


DAMPED_CASES = [
    ("gaussian", 0.5, 0.3), ("gaussian", 0.5, 0.6), ("gaussian", 0.99, 0.3),
    ("thole", 0.39, 0.3), ("thole", 0.39, 0.6), ("thole", 0.9, 0.6),
]


@pytest.mark.parametrize("kernel, damping, x", DAMPED_CASES)
@pytest.mark.parametrize("lattice", list(BRAVAIS))
def test_damped_lattice_obeys_the_exact_local_field_formula(lattice, kernel, damping, x):
    """Where the damping is NOT negligible, the dielectric constant is still
    exactly predictable:

        (eps - 1)/(eps + 2) = x / (1 + alpha W),   W = sum_R (lambda_3 - lambda_5)/r^3,

    the Lorentz theorem with the model's damped dipole tensor (derivation in
    :func:`_lorentz_sublattice_reference`; damping functions are those of
    SPEC 12 and SPEC 1.6).  W > 0, so damping always LOWERS eps below CM.

    These cases span x = 0.3-0.6 (eps = 2.2-5.5) and CM deviations of 1.5e-3
    to 2.1e-1, so this pins the damped lattice operator -- Ewald sum, damping,
    width rule, K_E and the local field -- to one closed form at densities
    where no bare-dipole result applies.  Tolerance 1e-12 relative on
    ``eps - 1`` (measured <= 1.1e-15).
    """
    cell = BRAVAIS[lattice]
    alpha = _alpha_for(x, cell)
    reference = _lorentz_sublattice_reference(cell, [torch.zeros(3)], [alpha], kernel, damping)
    eps = float(_monatomic_eps(cell, alpha, kernel, damping)[0][2, 2])
    assert abs((eps - 1.0) / (reference["eps"] - 1.0) - 1.0) < 1e-12, (eps, reference["eps"])
    predicted_ratio = x / (1.0 + alpha * reference["correction"])
    assert abs((eps - 1.0) / (eps + 2.0) / predicted_ratio - 1.0) < 1e-12
    assert (eps - 1.0) / (eps + 2.0) < x  # damping lowers eps


@pytest.mark.parametrize("kernel, damping", [("gaussian", 0.99), ("thole", 0.9)])
def test_clausius_mossotti_deviation_vanishes_as_the_damping_range_shrinks(kernel, damping):
    """The deviation from CM is the damping and nothing else, and it vanishes.

    The damping range is set by the atomic size, ``alpha^(1/3)``: the Gaussian
    dipole width is ``~0.7 (alpha/a)^(1/3)`` (SPEC 12) and Thole's is
    ``(alpha/a)^(1/3)``.  At a fixed simple-cubic spacing, lowering ``x``
    (i.e. alpha) shrinks the range relative to the spacing.  The measured
    deviation ``delta = [(eps-1)/(eps+2)]/x - 1`` must be negative, fall
    monotonically (super-exponentially: ``exp(-d^2/s^2)`` or
    ``exp(-a d^3/alpha)``) and reach round-off.  Measured, Gaussian a = 0.99:
    x = 0.6, 0.3, 0.1, 0.05, 0.02 -> -1.1e-1, -1.4e-2, -2.6e-5, -2.2e-8,
    -2.4e-15; Thole a = 0.9: -1.0e-2, -1.9e-5, then round-off (1e-16).
    """
    cell = BRAVAIS["sc"]
    deviations = []
    for x in (0.6, 0.3, 0.1, 0.05, 0.02):
        eps = float(_monatomic_eps(cell, _alpha_for(x, cell), kernel, damping)[0][2, 2])
        deviations.append(((eps - 1.0) / (eps + 2.0)) / x - 1.0)
    floor = 1e-14
    assert all(d < floor for d in deviations), deviations
    magnitudes = [abs(d) for d in deviations]
    for bigger, smaller in zip(magnitudes, magnitudes[1:]):
        assert smaller < bigger or smaller < floor, magnitudes
    assert magnitudes[0] > 1e-3          # the sweep starts where damping matters
    assert magnitudes[-1] < 1e-13        # and ends at bare point dipoles


def test_clausius_mossotti_deviation_falls_as_thole_damping_tightens_at_fixed_density():
    """At fixed density (x = 0.3, eps ~ 2.3) the Thole range ``(alpha/a)^(1/3)``
    shrinks as ``a`` grows; the CM deviation must fall monotonically, as
    ``~6 a exp(-a d^3/alpha)``, and equal the exact prediction at every a."""
    cell = BRAVAIS["sc"]
    alpha = _alpha_for(0.3, cell)
    previous = None
    for thole_a in (0.1, 0.2, 0.39, 0.6, 0.9):
        eps = float(_monatomic_eps(cell, alpha, "thole", thole_a)[0][2, 2])
        deviation = ((eps - 1.0) / (eps + 2.0)) / 0.3 - 1.0
        reference = _lorentz_sublattice_reference(cell, [torch.zeros(3)], [alpha], "thole", thole_a)
        assert abs(deviation + alpha * reference["correction"] / (1 + alpha * reference["correction"])) \
            < 1e-12, (thole_a, deviation)
        if previous is not None:
            assert abs(deviation) < abs(previous)
        previous = deviation


CSCL_CELL = 4.0 * torch.eye(3)
CSCL_BASIS = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
ROCKSALT_CELL = _fcc_cell(5.0)
ROCKSALT_BASIS = torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])

# (kernel, damping, (alpha_1, alpha_2), damping negligible?)
TWO_SUBLATTICE_CASES = [
    ("thole", 0.9, (0.5, 2.0), True),
    ("gaussian", 0.9, (0.05, 0.3), True),
    ("thole", 0.39, (1.0, 4.0), False),
    ("gaussian", 0.9, (0.5, 2.0), False),
]


@pytest.mark.parametrize("kernel, damping, alphas, negligible", TWO_SUBLATTICE_CASES)
def test_two_sublattice_cscl_obeys_the_coupled_lorentz_field(kernel, damping, alphas, negligible):
    """CsCl structure, two different polarizabilities, both sites O_h.

    Physics: every site of CsCl is cubic, so the dipole sum of EITHER
    sublattice at EITHER site is ``-(4 pi/3 Omega) I`` and every site feels
    ``E_loc = E + 4 pi P / 3`` with P the TOTAL polarization.  Hence
    ``mu_t = alpha_t E_loc`` and ``(eps-1)/(eps+2) = 4 pi (alpha_1 + alpha_2) /
    (3 Omega)`` -- CM with the summed polarizability -- and the per-site
    enhancement ``E_loc/E = (eps + 2)/3`` is the SAME on both sublattices
    (Lorentz; Kittel ch. 15; the additivity is the basis of the
    Tessman-Kahn-Shockley ionic polarizabilities, Phys. Rev. 92, 890 (1953)).

    With the damping not negligible the exact coupled 2x2 result of
    :func:`_lorentz_sublattice_reference` applies instead.  Route:
    ``dielectric_tensor`` (minimiser) and, for the site dipoles, a
    finite-field converged solve (``solve_result.mu``).  Tolerances 1e-12 on
    eps - 1 and 1e-10 on the site moments (solve tolerance 1e-12).
    """
    dielectric_tensor = require("observables", "dielectric_tensor")
    config = _kernel_config(kernel, damping)
    state = _crystal(CSCL_CELL, CSCL_BASIS, [0.0, 0.0], list(alphas), 0.25, config)
    eps = float(dielectric_tensor(state.solve_result, state.assembly, route="minimiser")[2, 2])
    reference = _lorentz_sublattice_reference(CSCL_CELL, list(CSCL_BASIS), list(alphas), kernel, damping)
    assert abs((eps - 1.0) / (reference["eps"] - 1.0) - 1.0) < 1e-12

    x = 4.0 * math.pi * sum(alphas) / (3.0 * 64.0)
    cm_deviation = ((eps - 1.0) / (eps + 2.0)) / x - 1.0
    if not negligible:
        assert cm_deviation < -1e-7   # the damped cases really are away from CM
        return
    assert abs(cm_deviation) < 1e-11, cm_deviation

    field = 0.01
    fielded = _crystal(CSCL_CELL, CSCL_BASIS, [0.0, 0.0], list(alphas), 0.25, config,
                       field=torch.tensor([0.0, 0.0, field]))
    mu = fielded.solve_result.mu
    for site, alpha in enumerate(alphas):
        local_field = K_E * float(mu[site, 2]) / alpha          # mu = (alpha/K_E) E_loc
        assert abs(local_field / (field * (eps + 2.0) / 3.0) - 1.0) < 1e-10, (site, local_field)
    assert float(mu[:, :2].abs().max()) < 1e-14


# ===========================================================================
# 3.  Szigeti charge, Lyddane-Herzfeld force constant, Lyddane-Sachs-Teller
# ===========================================================================
IONIC_CASES = [
    # (structure, kernel, damping, (alpha_+, alpha_-), damping negligible?)
    ("cesium_chloride", "thole", 0.9, (0.5, 2.0), True),
    ("rocksalt", "thole", 0.9, (0.5, 2.0), False),
    ("cesium_chloride", "gaussian", 0.9, (0.05, 0.3), True),
    ("cesium_chloride", "gaussian", 0.9, (0.5, 2.0), False),
    ("rocksalt", "gaussian", 0.9, (0.5, 2.0), False),
    ("cesium_chloride", "thole", 0.9, (1e-9, 1e-9), True),   # rigid, unpolarizable ions
]


def _ionic(structure):
    return (CSCL_CELL, CSCL_BASIS) if structure == "cesium_chloride" else (ROCKSALT_CELL, ROCKSALT_BASIS)


def _ionic_response(structure, kernel, damping, alphas):
    dielectric_tensor, born_charges_minimiser = require(
        "observables", "dielectric_tensor", "born_charges_minimiser"
    )
    cell, basis = _ionic(structure)
    state = _crystal(cell, basis.clone().requires_grad_(True), [1.0, -1.0], list(alphas), 0.25,
                     _kernel_config(kernel, damping))
    eps = dielectric_tensor(state.solve_result, state.assembly, route="minimiser").detach()
    born = born_charges_minimiser(state.assembly).detach()
    return eps, born


@pytest.mark.parametrize("structure, kernel, damping, alphas, negligible", IONIC_CASES)
def test_born_charge_of_polarizable_rigid_ions_is_the_szigeti_charge(
    structure, kernel, damping, alphas, negligible
):
    """Szigeti: ``Z* = q (eps_inf + 2) / 3`` for polarizable point ions.

    Physics: for rigid ions of charge +-q with point polarizabilities and no
    charge transfer, on a lattice whose sites are all cubic (NaCl, CsCl), a
    sublattice displacement polarizes every ion through the Lorentz field of
    the displaced charges, and the transverse effective charge is enhanced by
    exactly the local-field factor (Szigeti, Trans. Faraday Soc. 45, 155
    (1949); Born & Huang, *Dynamical Theory of Crystal Lattices*, 1954, sec.
    9).  Derivation in :func:`_lorentz_sublattice_reference`.  The limit
    alpha -> 0 is the rigid-ion statement ``Z* = q``, ``eps_inf = 1``.

    Route: ``born_charges_minimiser`` (implicit derivative of the exact
    minimiser, SPEC 15) and ``dielectric_tensor``.  Also asserted: Z* is
    isotropic (cubic sites) and obeys the acoustic sum rule.  Where damping is
    not negligible, the exact damped result applies (measured agreement
    1e-15).  Tolerance 1e-11.
    """
    eps_tensor, born = _ionic_response(structure, kernel, damping, alphas)
    eps = float(eps_tensor[2, 2])
    z = float(born[0, 2, 2])
    assert float((born[0] - z * torch.eye(3)).abs().max()) < 1e-13
    assert float(born.sum(0).abs().max()) < 1e-13                  # acoustic sum rule
    cell, basis = _ionic(structure)
    reference = _lorentz_sublattice_reference(cell, list(basis), list(alphas), kernel, damping,
                                              charges=[1.0, -1.0], sigma=[0.25, 0.25])
    assert abs(z / reference["born_over_q"] - 1.0) < 1e-11, (z, reference["born_over_q"])
    assert abs(eps / reference["eps"] - 1.0) < 1e-12
    if negligible:
        assert abs(z / ((eps + 2.0) / 3.0) - 1.0) < 1e-11, (z, eps)
    else:
        assert abs(z / ((eps + 2.0) / 3.0) - 1.0) > 1e-9   # damping visible; still exact above
    if max(alphas) < 1e-6:
        # rigid ions: Z* -> q and eps -> 1, the remainder being exactly the
        # O(alpha) Lorentz term c S / (1 - c S), c S = 4 pi (alpha_+ + alpha_-)/(3 Omega)
        lorentz = 4.0 * math.pi * sum(alphas) / (3.0 * float(torch.det(cell).abs()))
        assert abs(z - 1.0 - lorentz / (1.0 - lorentz)) < 1e-14
        assert abs(eps - 1.0 - 3.0 * lorentz / (1.0 - lorentz)) < 1e-14


def _force_constants(structure, kernel, damping, alphas, step=1e-3):
    """``Phi = d^2 E / dr dr`` of the MODEL's electrostatic energy, ``[6, 6]``.

    Central differences of the pipeline's forces (autograd through the
    converged solve) at steps h and 2h, Richardson-combined: truncation
    O(h^4) ~ 1e-12, round-off ~ (force noise 1e-14)/h ~ 1e-11.
    """
    cell, basis = _ionic(structure)
    config = _kernel_config(kernel, damping)

    def gradient(displacement):
        pos = (basis + displacement).clone().requires_grad_(True)
        state = _crystal(cell, pos, [1.0, -1.0], list(alphas), 0.25, config)
        (grad,) = torch.autograd.grad(state.energy, pos)
        return grad.reshape(-1)

    phi = torch.zeros(6, 6)
    for column in range(6):
        unit = torch.zeros(6)
        unit[column] = 1.0
        unit = unit.reshape(2, 3)
        near = (gradient(step * unit) - gradient(-step * unit)) / (2 * step)
        far = (gradient(2 * step * unit) - gradient(-2 * step * unit)) / (4 * step)
        phi[:, column] = (4.0 * near - far) / 3.0
    return phi


@pytest.mark.parametrize("structure, kernel, damping, alphas, negligible",
                         [case for case in IONIC_CASES if case[1] == "thole"] + [IONIC_CASES[3]])
def test_transverse_force_constant_is_the_lyddane_herzfeld_szigeti_value(
    structure, kernel, damping, alphas, negligible
):
    """The model's own Gamma-point electrostatic force constant.

    Physics: for the polarizable point-ion lattice (all sites cubic) the
    analytic (tinfoil, transverse) electrostatic force constant of the
    sublattice displacement is ``Phi = -(4 pi K_E / 3 Omega) q Z*`` =
    ``-(4 pi K_E / 9 Omega) q^2 (eps_inf + 2)`` -- the point-ion Lorentz term
    enhanced by the local field.  It is the electrostatic part of Szigeti's
    ``mu omega_TO^2 = R_0 - (4 pi / 9 Omega)(eps_inf + 2) e_S^2``
    (Lyddane & Herzfeld, Phys. Rev. 54, 846 (1938); Szigeti, Proc. R. Soc.
    A 204, 51 (1950)).  With damping, the exact result of
    :func:`_lorentz_sublattice_reference`.

    Two routes: finite differences of the pipeline's forces
    (:func:`_force_constants`, tolerance 1e-9 relative from its error budget;
    measured <= 1e-11), and the AUTOGRAD second derivative at the exactly
    symmetric point -- the case SPEC 15.2 documents as losing the electronic
    relaxation under a fixed-k solve (tolerance 1e-10).  Also: the acoustic
    sum rule and cubic isotropy of the blocks.
    """
    cell, basis = _ionic(structure)
    phi = _force_constants(structure, kernel, damping, alphas)
    blocks = phi.reshape(2, 3, 2, 3)
    assert float(blocks.sum(dim=2).abs().max()) < 1e-9 * float(phi.abs().max())   # ASR
    assert float((blocks[0, :, 0, :] - blocks[0, 2, 0, 2] * torch.eye(3)).abs().max()) \
        < 1e-9 * float(phi.abs().max())
    reference = _lorentz_sublattice_reference(cell, list(basis), list(alphas), kernel, damping,
                                              charges=[1.0, -1.0], sigma=[0.25, 0.25])
    measured = float(blocks[0, 2, 0, 2])
    assert abs(measured / reference["phi"] - 1.0) < 1e-9, (measured, reference["phi"])
    if negligible:
        omega = float(torch.det(cell).abs())
        lyddane_herzfeld = -(4.0 * math.pi * K_E / (3.0 * omega)) * reference["born_over_q"]
        assert abs(measured / lyddane_herzfeld - 1.0) < 1e-9

    pos = basis.clone().requires_grad_(True)
    state = _crystal(cell, pos, [1.0, -1.0], list(alphas), 0.25, _kernel_config(kernel, damping))
    (grad,) = torch.autograd.grad(state.energy, pos, create_graph=True)
    (row,) = torch.autograd.grad(grad[0, 2], pos)
    assert abs(float(row[0, 2]) / reference["phi"] - 1.0) < 1e-10, (float(row[0, 2]), reference["phi"])


@pytest.mark.parametrize("structure, kernel, short_range",
                         [("cesium_chloride", "thole", 3.0), ("rocksalt", "gaussian", 6.0)])
def test_lyddane_sachs_teller_relation(structure, kernel, short_range):
    """Lyddane-Sachs-Teller: ``eps_0 / eps_inf = omega_LO^2 / omega_TO^2``.

    Physics: a diatomic cubic crystal has one triply degenerate IR-active
    optic mode at Gamma; the macroscopic field of a longitudinal wave stiffens
    it by the non-analytic term, and the static response adds the ionic
    polarization, so the ratio is exact (Lyddane, Sachs & Teller, Phys. Rev.
    59, 673 (1941); Born & Huang sec. 10).  Also exact, and checked with the
    TO frequency from the diagonalisation (not from the package's formula):
    ``eps_0 - eps_inf = 4 pi K_E Z*^2 / (Omega mu omega_TO^2)`` and
    ``omega_LO^2 - omega_TO^2 = 4 pi K_E Z*^2 / (Omega mu eps_inf)``.

    Inputs are all the model's own: ``eps_inf`` and ``Z*`` of polarizable
    rigid ions (the Szigeti crystal above), and ``Phi`` = the model's
    electrostatic force constants (finite differences) plus a nearest-
    neighbour short-range spring ``k [[I, -I], [-I, I]]`` that makes the
    lattice stable -- the most general Gamma-point analytic force constant a
    cubic diatomic admits (acoustic sum rule + cubic symmetry), so LST is a
    statement about the observables, not about the choice of k.  Masses of
    Cs/Na and Cl in amu (only ratios enter).  LO is independent of the
    direction of q in a cubic crystal: four directions are checked.  Routes:
    ``static_dielectric_tensor``, ``nonanalytic_force_constants``.
    Tolerance 1e-10 (eigenvalue/pseudo-inverse round-off; measured <= 1.2e-13,
    at eps_0/eps_inf = 2.36 (CsCl) and 2.60 (NaCl structure)).
    """
    static_dielectric_tensor, nonanalytic_force_constants = require(
        "observables", "static_dielectric_tensor", "nonanalytic_force_constants"
    )
    alphas, damping = (0.5, 2.0), 0.9
    eps_inf, born = _ionic_response(structure, kernel, damping, alphas)
    phi = _force_constants(structure, kernel, damping, alphas)
    phi = 0.5 * (phi + phi.T) + short_range * torch.kron(
        torch.tensor([[1.0, -1.0], [-1.0, 1.0]]), torch.eye(3)
    )
    cell, _basis = _ionic(structure)
    omega = torch.det(cell).abs()
    masses = torch.tensor([132.905 if structure == "cesium_chloride" else 22.990, 35.453])
    reduced = float(1.0 / (1.0 / masses).sum())
    inverse_sqrt_mass = torch.repeat_interleave(masses, 3) ** -0.5

    def squared_frequencies(force_constants):
        return torch.linalg.eigvalsh(inverse_sqrt_mass[:, None] * force_constants * inverse_sqrt_mass[None])

    transverse = squared_frequencies(phi)
    assert float(transverse[:3].abs().max()) < 1e-12 * float(transverse[-1])   # acoustic
    omega_to2 = float(transverse[3:].mean())
    assert float((transverse[3:] - omega_to2).abs().max()) < 1e-10 * omega_to2  # triply degenerate
    assert omega_to2 > 0.0

    eps_0 = static_dielectric_tensor(eps_inf, born, phi, omega)
    z = float(born[0, 2, 2])
    e_inf = float(eps_inf[2, 2])
    szigeti = 4.0 * math.pi * K_E * z * z / (float(omega) * reduced * omega_to2)
    assert abs(float(eps_0[2, 2] - eps_inf[2, 2]) / szigeti - 1.0) < 1e-10
    for direction in ([0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.3, -0.7, 0.2]):
        q_hat = torch.tensor(direction)
        longitudinal = squared_frequencies(phi + nonanalytic_force_constants(born, eps_inf, omega, q_hat))
        omega_lo2 = float(longitudinal[-1])
        assert float((longitudinal[3:5] - omega_to2).abs().max()) < 1e-10 * omega_to2  # 2 TO unchanged
        unit = q_hat / q_hat.norm()
        ratio = float(unit @ eps_0 @ unit) / float(unit @ eps_inf @ unit)
        assert abs(omega_lo2 / omega_to2 / ratio - 1.0) < 1e-10, (direction, omega_lo2 / omega_to2, ratio)
        splitting = 4.0 * math.pi * K_E * z * z / (float(omega) * reduced * e_inf)
        assert abs((omega_lo2 - omega_to2) / splitting - 1.0) < 1e-10


# ===========================================================================
# 4.  Dispersion of quantum Drude oscillators
# ===========================================================================
DRUDE_ALPHA = 1.0      # A^3
DRUDE_ELECTRONS = 1.0  # N in the TRK sum rule
#: hbar omega of the isolated oscillator: A_mu / m = K_E N (hbar^2/m_e) / alpha.
DRUDE_OMEGA = math.sqrt(K_E * DRUDE_ELECTRONS * HBAR2_OVER_ME / DRUDE_ALPHA)   # 10.475 eV
#: London: C6 = (3/4) hbar omega alpha^2 for identical Drude oscillators.
C6_LONDON = 0.75 * DRUDE_OMEGA * DRUDE_ALPHA ** 2
#: Axilrod-Teller-Muto for identical Drude oscillators: C9 = (9/16) hbar omega alpha^3.
C9_ATM = 9.0 / 16.0 * DRUDE_OMEGA * DRUDE_ALPHA ** 3


def _package_dispersion(pos, kernel="gaussian"):
    """SPEC 13 dispersion of isolated Drude atoms through the pipeline.

    Open boundaries, no bonds, neutral single-atom fragments, ``A_mu = K_E /
    alpha``, electron count N = 1 via the heads' dispersion readout.
    ``gaussian_a = 0.95`` / ``thole_a = 0.39``: at R >= 5 A the dipole damping
    changes T by < 3e-9 (Gaussian, e^-(R/s)^2 with s = 1.01 A) and < e^-49
    (Thole), far below every tolerance below.
    """
    electrostatic_state = require("energy", "electrostatic_state")
    n = int(pos.shape[0])
    heads = _PhysicalHeads(_params([DRUDE_ALPHA] * n, 0.5),
                           electrons=torch.full((n,), DRUDE_ELECTRONS))
    damping = 0.95 if kernel == "gaussian" else 0.39
    config = _kernel_config(kernel, damping, dispersion=True)
    state = electrostatic_state(
        torch.zeros(n, 1), pos, torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 3), torch.zeros(0),
        heads, config, fragment_id=torch.arange(n), fragment_charges=torch.zeros(n),
    )
    return float(state.dispersion)


def _bare_dipole_tensor(pos):
    """``[3n, 3n]`` bare T = (I - 3 r r / r^2)/r^3 (Gaussian units), zero on-site."""
    n = int(pos.shape[0])
    tensor = torch.zeros(3 * n, 3 * n)
    for i in range(n):
        for j in range(n):
            if i != j:
                r = pos[i] - pos[j]
                d = float(r.norm())
                tensor[3 * i:3 * i + 3, 3 * j:3 * j + 3] = (torch.eye(3) - 3.0 * torch.outer(r, r) / d ** 2) / d ** 3
    return tensor


def _exact_drude_energy(pos):
    """Zero-point energy of coupled identical Drude oscillators, EXACT.

    ``H = sum p^2/2m + 1/2 mu^T (alpha^-1 I + T) mu``: normal-mode frequencies
    ``omega_p = omega sqrt(1 + lambda_p)`` with ``lambda_p`` the eigenvalues of
    ``alpha T``; ``E = (hbar omega / 2) sum_p (sqrt(1 + lambda_p) - 1)``,
    evaluated as ``lambda/(sqrt(1+lambda)+1)`` so the small differences carry
    full relative precision.  Exact diagonalisation of the 3n x 3n problem
    (9 x 9 for a trimer) -- no quadrature, no expansion.
    """
    lam = torch.linalg.eigvalsh(DRUDE_ALPHA * _bare_dipole_tensor(pos))
    return 0.5 * DRUDE_OMEGA * float((lam / (torch.sqrt(1.0 + lam) + 1.0)).sum())


def _drude_dimer_closed_form(r):
    """The Drude dimer in closed form: with x = alpha/R^3 the six modes are
    omega sqrt(1 -+ 2x) (longitudinal) and omega sqrt(1 +- x) (x2, transverse),
    so ``E = (hbar omega/2)[f(2x) + f(-2x) + 2 f(x) + 2 f(-x)]``,
    ``f(y) = sqrt(1+y) - 1``; expanding, ``E = -(3/4) hbar omega alpha^2 / R^6
    (1 + (15/16) x^2 + ...)`` (London, Trans. Faraday Soc. 33, 8 (1937))."""
    x = DRUDE_ALPHA / r ** 3

    def f(y):
        return y / (math.sqrt(1.0 + y) + 1.0)

    return 0.5 * DRUDE_OMEGA * (f(2 * x) + f(-2 * x) + 2 * f(x) + 2 * f(-x))


#: Precision of one package dispersion energy, |dE| <= REL |E| + ABS, asserted
#: by the dimer test below.  REL covers the residual Gaussian damping at R = 5 A
#: (T changed by 2.5e-9, E by 5e-9) and the 16-point quadrature.  ABS is a
#: round-off floor of the log-det route, measured 2-3e-13 eV at every R from 8
#: to 24 A: ln det(I + S) is read from Cholesky diagonals 1 + O(S^2), whose
#: ~1e-16 absolute error is multiplied by the Jacobian of the frequency map
#: omega = s x/(1 - x) (~4e5 at the last Gauss node).  It does not shrink with
#: more nodes; relative to a weakly coupled pair it grows as R^6 (5e-9 at 8 A,
#: 9e-8 at 12 A, 5e-7 at 16 A for alpha = 1 A^3).
DISPERSION_REL = 2e-8
DISPERSION_ABS = 1e-12   # eV


@pytest.mark.parametrize("kernel", ["gaussian", "thole"])
def test_drude_dimer_dispersion_is_the_exact_london_result(kernel):
    """Two identical Drude oscillators: the package's E_disp equals the exact
    closed form to all orders in alpha/R^3, and its R -> inf limit is
    London's ``-C6/R^6`` with ``C6 = (3/4) hbar omega alpha^2``.

    The closed form's expansion ``1 + (15/16) x^2`` is itself verified against
    the leading London term (independent of the package), and an isolated
    atom has exactly zero dispersion (so E3 below needs no one-body term).
    Tolerance ``DISPERSION_REL |E| + DISPERSION_ABS`` (see there): the
    package's 16-point log-det quadrature (SPEC 13) and its round-off floor.
    """
    for r in (5.0, 6.0, 8.0, 12.0, 16.0):
        exact = _drude_dimer_closed_form(r)
        assert abs(exact / _exact_drude_energy(torch.tensor([[0.0, 0, 0], [0, 0, r]])) - 1.0) < 1e-12
        x = DRUDE_ALPHA / r ** 3
        london = -C6_LONDON / r ** 6
        # next term 1.8 x^4; round-off of the closed form ~ 1e-16 / x
        assert abs(exact / london - 1.0 - 15.0 / 16.0 * x * x) < 4.0 * x ** 4 + 1e-15 / x
        # an oblique axis, so no Cartesian alignment can hide an orientation error
        axis = torch.tensor([0.3, -0.2, 1.0])
        package = _package_dispersion(torch.stack([torch.zeros(3), r * axis / axis.norm()]), kernel)
        assert abs(package - exact) < DISPERSION_REL * abs(exact) + DISPERSION_ABS, (r, package, exact)
    assert _package_dispersion(torch.zeros(1, 3), kernel) == 0.0


def _triangle(shape, r):
    if shape == "equilateral":
        return torch.tensor([[0.0, 0, 0], [r, 0, 0], [r / 2, r * math.sqrt(3.0) / 2, 0]])
    if shape == "right_isosceles":
        return torch.tensor([[0.0, 0, 0], [r, 0, 0], [0, r, 0]])
    if shape == "linear":
        return torch.tensor([[0.0, 0, 0], [r, 0, 0], [2 * r, 0, 0]])
    if shape == "obtuse":
        return torch.tensor([[0.0, 0, 0], [r, 0, 0], [1.7 * r, 0.4 * r, 0.3 * r]])
    raise KeyError(shape)


def _atm_geometry(pos):
    """``(1 + 3 cos a cos b cos c) / (R12 R23 R31)^3`` with a, b, c the interior angles."""
    def cosine(apex, b, c):
        u, v = pos[b] - pos[apex], pos[c] - pos[apex]
        return float(u @ v / (u.norm() * v.norm()))

    angular = 1.0 + 3.0 * cosine(0, 1, 2) * cosine(1, 2, 0) * cosine(2, 0, 1)
    product = float((pos[0] - pos[1]).norm() * (pos[1] - pos[2]).norm() * (pos[2] - pos[0]).norm())
    return angular, angular / product ** 3


def _three_body(energy, pos):
    """``E3 = E(123) - sum_pairs E(ij) + sum_i E(i)``; E(i) = 0 (tested above)."""
    return energy(pos) - energy(pos[[0, 1]]) - energy(pos[[1, 2]]) - energy(pos[[0, 2]])


def test_axilrod_teller_muto_coefficient_derivation():
    """The derivation of C9 for Drude oscillators, checked term by term.

    E_disp = (hbar/2pi) int_0^inf du ln det(I + alpha(iu) T) (the coupled-
    oscillator zero-point energy; Tkatchenko et al., JCP 138, 074106 (2013)),
    alpha(iu) = alpha omega^2/(omega^2 + u^2) for a Drude oscillator.
    Expanding ln det(I + X) = sum_n (-1)^(n+1) Tr X^n / n, the n = 2 term is the
    pairwise C6 and the n = 3 term is the leading three-body energy:

        E3 = (hbar/2pi)(1/3) int alpha(iu)^3 du  Tr (T^3)
        Tr (T^3) = 6 Tr(T12 T23 T31)               (3! orderings, T_ii = 0)
        Tr(T12 T23 T31) = 3 (1 + 3 cos a cos b cos c) / (R12 R23 R31)^3
        int_0^inf (omega^2/(omega^2+u^2))^3 du = 3 pi omega / 16

    so E3 = (9/16) hbar omega alpha^3 (1 + 3 cos a cos b cos c)/(R12R23R31)^3,
    i.e. C9 = (9/16) hbar omega alpha^3 = (3/4) alpha C6 (Axilrod & Teller,
    J. Chem. Phys. 11, 299 (1943); Muto, Proc. Phys.-Math. Soc. Japan 17, 629
    (1943)).  Checked: the trace identity on random triangles (1e-12), the
    frequency integral by Gauss-Legendre quadrature in u = omega tan(theta)
    (1e-14), and C9 = (3/4) alpha C6.
    """
    generator = torch.Generator().manual_seed(11)
    for _ in range(6):
        pos = torch.randn(3, 3, generator=generator) * 3.0
        tensor = _bare_dipole_tensor(pos)
        t12, t23, t31 = tensor[0:3, 3:6], tensor[3:6, 6:9], tensor[6:9, 0:3]
        _angular, geometric = _atm_geometry(pos)
        assert abs(float(torch.trace(t12 @ t23 @ t31)) / (3.0 * geometric) - 1.0) < 1e-12
        assert abs(float(torch.trace(tensor @ tensor @ tensor)) /
                   (6.0 * float(torch.trace(t12 @ t23 @ t31))) - 1.0) < 1e-12
    nodes, weights = np.polynomial.legendre.leggauss(40)
    theta = 0.25 * math.pi * (nodes + 1.0)          # u = omega tan(theta), du = omega sec^2
    integral = float(np.sum(0.25 * math.pi * weights * np.cos(theta) ** 4))   # per unit omega
    assert abs(integral / (3.0 * math.pi / 16.0) - 1.0) < 1e-14
    assert abs((1.0 / (2.0 * math.pi)) * (1.0 / 3.0) * integral * 6.0 * 3.0 / (9.0 / 16.0) - 1.0) < 1e-14
    assert abs(C9_ATM / (0.75 * DRUDE_ALPHA * C6_LONDON) - 1.0) < 1e-15


ATM_SHAPES = ("equilateral", "right_isosceles", "linear", "obtuse")
ATM_RADII = (5.0, 6.0, 8.0)


@pytest.mark.parametrize("shape", ATM_SHAPES)
def test_three_body_dispersion_is_axilrod_teller_muto(shape):
    """E3 of three identical Drude oscillators: exact at finite R, ATM as R -> inf.

    (a) At R = 5, 6, 8 A (side scale) the package's
        E3 = E(123) - sum pairs equals the EXACT 9 x 9 diagonalisation
        (``_exact_drude_energy``) to the precision the subtraction allows:
        each package energy is good to ``REL |E| + ABS`` (dimer test), so
        ``|dE3| <= REL (|E123| + sum |E_ij|) + 4 ABS``; at these radii that is
        1e-5 to 5e-4 of E3.  Measured 1e-8 to 3e-5.
    (b) The exact E3 differs from ATM by the n >= 4 terms of the log-det
        expansion, a relative O(alpha/R^3): (E3/E_ATM - 1) R^3 is constant to
        a few % across R (it is the next-order coefficient).
    (c) Extrapolating E3 (R12R23R31)^3 / (1 + 3 cos cos cos) to R -> inf
        (quadratic in y = alpha/R^3 through the three radii) recovers
        C9 = (9/16) hbar omega alpha^3 for the exact reference to 5e-5 (the
        y^3 remainder, measured <= 1.1e-5), and for the package to that plus
        the propagated bound of (a) (measured <= 6.6e-5).  The same C9 for all
        four shapes -- equilateral (angular factor 11/8, repulsive), right
        isosceles (1), linear (-2, attractive), an obtuse triangle (-1.21) --
        is the angular factor 1 + 3 cos a cos b cos c.
    """
    ys, package_values, exact_values, bounds = [], [], [], []
    scaled = []
    for r in ATM_RADII:
        pos = _triangle(shape, r)
        angular, geometric = _atm_geometry(pos)
        e123 = _package_dispersion(pos)
        pairs = [_package_dispersion(pos[list(p)]) for p in ((0, 1), (1, 2), (0, 2))]
        e3_package = e123 - sum(pairs)
        e3_exact = _three_body(_exact_drude_energy, pos)
        bound = DISPERSION_REL * (abs(e123) + sum(abs(p) for p in pairs)) + 4.0 * DISPERSION_ABS
        assert abs(e3_package - e3_exact) < bound, (shape, r, e3_package, e3_exact, bound)
        assert math.copysign(1.0, e3_exact) == math.copysign(1.0, angular)   # sign of ATM
        # E3 (R12 R23 R31)^3 / (1 + 3 cos cos cos): tends to C9 as R -> inf
        ys.append(DRUDE_ALPHA / r ** 3)
        package_values.append(e3_package / geometric)
        exact_values.append(e3_exact / geometric)
        bounds.append(bound / abs(e3_exact))
        scaled.append((e3_exact / (C9_ATM * geometric) - 1.0) * r ** 3)
    # (b): the approach to ATM is O(alpha/R^3) -- R^3 (E3/E_ATM - 1) settles
    assert abs(scaled[-2] / scaled[-1] - 1.0) < 0.05, scaled

    # (c): quadratic extrapolation in y = alpha/R^3 to y = 0 (Lagrange weights)
    design = np.vstack([np.ones(3), ys, np.square(ys)]).T
    lagrange = np.linalg.inv(design)[0]
    exact_c9 = float(lagrange @ np.array(exact_values))
    package_c9 = float(lagrange @ np.array(package_values))
    assert abs(exact_c9 / C9_ATM - 1.0) < 5e-5, (shape, exact_c9 / C9_ATM - 1.0)
    propagated = float(np.abs(lagrange) @ np.array(bounds))
    assert abs(package_c9 / C9_ATM - 1.0) < 5e-5 + propagated, (
        shape, package_c9 / C9_ATM - 1.0, propagated
    )
