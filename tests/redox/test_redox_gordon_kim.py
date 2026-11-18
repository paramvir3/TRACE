"""The Pauli half of the overlap from first principles (``RedoxConfig(carrier_overlap='gordon_kim')``; SPEC 25).

SPEC 23 made the electrostatics of overlapping charge-state densities exact.  Alone it over-corrects (measured
2026-10-18): a band electron on Mg drops ~7 eV for six O2- neighbours, because the Pauli repulsion of the same overlap
is left to SPEC 18's fitted K S on other (Katriel-Davidson) densities.  The electron-gas functional of the summed
densities (Gordon & Kim, J. Chem. Phys. 56, 3122 (1972); ESTABLISHED) gives that half from the same densities:

    G = { int f[rho_a^n + rho_b^n] - f[rho_a^n] - f[rho_b^n] } - { same at the frames },
    f = C_F rho^(5/3) + (1/9) |grad rho|^2 / (8 rho) - C_x rho^(4/3) - 0.44 rho / (7.8 + r_s).

Measured (X2C-PBE0 densities; eV): Au- vs O2- at 2.2 A, penetration -2.39, G +1.28; Mg+ vs O2- at 2.1 A, -1.59 / +1.23;
Au+ Au- at 2.47 A, +0.05 / +0.57.  NEW-TO-TRACE: state-resolved Gordon-Kim carrier terms in the configuration energies,
replacing the fitted exclusion.

G is right at contact and wrong in the tails: beyond ~3.5 A its gradient, exchange and correlation parts give a spurious
attraction (Au- vs O2- at 5 A: -24 meV; Mg+ vs O2- at 4 A: -62 meV), the local functionals' known failure where the
density is low.  The density overlap dS = S(rho_a^n, rho_b^n) - S(rho_a^f, rho_b^f) decays correctly, and G / dS is
nearly constant at contact (K = 4.5-5.3 Ha for Au, Mg against O over 2.0-3.0 A).  The Pauli term is therefore bilinear,

    E_Pauli = K_a<-b (delta_a | rho_b^f) + K_b<-a (rho_a^f | delta_b) + K_cc (delta_a | delta_b),

the one-carrier K fitted to G over the contact window [0.75, 1.25](r_cov,a + r_cov,b), K_cc to G's non-additive
two-carrier remainder (the density-overlap model of exchange repulsion: ESTABLISHED, Wheatley & Price, Mol. Phys. 69,
507 (1990); K from the electron-gas functional at contact: NEW-TO-TRACE).

Tests: G is zero at frames, symmetric in the two ions and grid-converged; K dS reproduces G at contact and stays
repulsive in the tail where G is not; the tables reproduce K dS; the module refuses it together with the fitted exclusion
(double counting) and without the electrostatic half; forces are minus the gradient.
"""
import os

import numpy as np
import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

HARTREE = 27.211386245988
BOHR = 0.529177210903


def _densities():
    from trace_redox.penetration import DATA
    if not os.path.exists(DATA):
        pytest.skip("library not built")
    d = np.load(DATA)
    return d["grid"], {tuple(k): v for k, v in zip(d["density_keys"].tolist(), d["densities"])}, \
        {int(z): int(f) for z, f in d["frames"].tolist()}


def test_g_is_zero_at_frames_symmetric_and_converged():
    from trace_redox.radial import gordon_kim_difference
    r, D, frames = _densities()
    fa, fb = D[(79, 0)], D[(8, -2)]
    assert gordon_kim_difference(r, fa, fa, fb, fb, 2.2 / BOHR) == 0.0
    g = gordon_kim_difference(r, fa, D[(79, -1)], fb, fb, 2.2 / BOHR) * HARTREE
    swapped = gordon_kim_difference(r, fb, fb, fa, D[(79, -1)], 2.2 / BOHR) * HARTREE
    fine = gordon_kim_difference(r, fa, D[(79, -1)], fb, fb, 2.2 / BOHR, 900, 400) * HARTREE
    assert 1.0 < g < 1.6                                   # +1.28 eV measured
    assert abs(g - swapped) <= 1e-9 * abs(g)               # the nu grid is symmetric: exact up to rounding
    assert abs(g - fine) <= 1e-3                           # 450 x 200 against 900 x 400: measured 3e-4 eV


def _K(key):
    from trace_redox.penetration import DATA
    d = np.load(DATA)
    if "pair_pauli_K" not in d.files:
        pytest.skip("library without the Pauli tables")
    t = d["pair_keys"].tolist().index(list(key))
    return d["pair_pauli_K"][t], float(d["pair_pauli_K_residual"][t])


@pytest.mark.parametrize("key, residual", [((8, -2, 79, -1), 0.05), ((8, -2, 79, 1), 0.05), ((8, -2, 12, 1), 0.05),
                                           ((8, -2, 13, 2), 0.05), ((8, -1, 12, 2), 0.10)])
def test_k_ds_reproduces_g_at_contact_and_stays_repulsive_in_the_tail(key, residual):
    """The AuMgO one-carrier types: K dS within the measured residual of G over the contact window; beyond 4.5 A, where G
    is dominated by its tail artifact, K dS keeps the sign of the contact value and decays."""
    from trace_redox.radial import density_overlap_parts, gordon_kim_difference
    from trace_redox.constants import COVALENT_RADII_A
    r, D, frames = _densities()
    K, stored = _K(key)
    Za, na, Zb, nb = key
    fa, fb = frames[Za], frames[Zb]
    assert stored <= residual
    contact = 1.0 * (COVALENT_RADII_A[Za] + COVALENT_RADII_A[Zb])
    for RA in (contact, 4.5, 6.0):
        P = np.array(density_overlap_parts(r, D[(Za, fa)], D[(Za, na)], D[(Zb, fb)], D[(Zb, nb)], RA / BOHR))
        pauli = float(P @ K)
        if RA == contact:
            G = gordon_kim_difference(r, D[(Za, fa)], D[(Za, na)], D[(Zb, fb)], D[(Zb, nb)], RA / BOHR) * HARTREE
            assert abs(pauli - G) <= 2 * residual * abs(G)
            sign, first = np.sign(pauli), abs(pauli)
        else:
            assert np.sign(pauli) == sign and abs(pauli) < 0.05 * first


def test_the_tables_reproduce_k_ds():
    from trace_redox.penetration import PenetrationLibrary
    from trace_redox.radial import density_overlap_parts
    lib = PenetrationLibrary()
    if lib.gk_coefficients is None:
        pytest.skip("library without the Pauli tables")
    r, D, frames = _densities()
    def k_ds(key, RA):
        K, _ = _K(key)
        Za, na, Zb, nb = key
        P = np.array(density_overlap_parts(r, D[(Za, frames[Za])], D[(Za, na)], D[(Zb, frames[Zb])], D[(Zb, nb)], RA / BOHR))
        return RA * float(P @ K)

    for key in ((8, -2, 79, -1), (8, -2, 12, 1), (79, 1, 79, -1)):
        Za, na, Zb, nb = key
        for RA in (1.9, 2.2, 2.9, 4.0):
            # same element: the loaded table is the exchange-symmetric mean of the two separately fitted orientations
            ref = k_ds(key, RA) if Za != Zb else 0.5 * (k_ds(key, RA) + k_ds((Zb, nb, Za, na), RA))
            x = (2.0 * RA - (lib.r_cut + lib.r_min)) / (lib.r_cut - lib.r_min)
            fit = float(np.polynomial.chebyshev.chebval(x, lib.gk_coefficients[lib.index[key]].numpy()))
            # measured: <= 1e-6 eV A for one-carrier types, ~1e-3 for the two-carrier Au+ Au- (it changes sign near 2 A)
            assert abs(fit - ref) <= 2e-3 + 1e-5 * abs(ref)


def test_the_configuration_is_validated():
    from trace_redox import RedoxConfig
    base = dict(fragment_ensemble="diabatic", anchor="formal")
    assert RedoxConfig().carrier_overlap == "electrostatic"
    with pytest.raises(ValueError, match="twice"):
        RedoxConfig(**base, carrier_penetration=True, carrier_overlap="gordon_kim", exclusion=True)
    with pytest.raises(ValueError, match="carrier_penetration"):
        RedoxConfig(**base, carrier_overlap="gordon_kim")
    with pytest.raises(ValueError, match="carrier_overlap"):
        RedoxConfig(**base, carrier_penetration=True, carrier_overlap="tf")


def test_module_forces_are_minus_the_gradient_with_gordon_kim():
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    from trace_redox.penetration import PenetrationLibrary
    if PenetrationLibrary().gk_coefficients is None:
        pytest.skip("library without Gordon-Kim tables")
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True,
                         carrier_penetration=True, carrier_overlap="gordon_kim", carrier_band=True)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), config)
    z = torch.tensor([8, 79, 79, 12, 13, 8])
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0],
                         [2.0, 2.1, 0.2]], dtype=torch.float64)
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos0)

    def energy(p):
        vec, length = edges(p, ei)
        return module(h, p, ei, vec, length, data={"z": z, "total_charge": 0.0})[0]

    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    for i, axis in [(0, 2), (1, 2), (3, 0), (5, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-6 * (1.0 + abs(fd))
