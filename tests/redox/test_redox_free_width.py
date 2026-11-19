"""Physical free-charge sizes for localised ionic states (audit step 2, B3; docs/ELECTROSTATICS_AUDIT_2026-10-09.md).

The on-site ladder E_Z(n) already holds an isolated ion's whole energy (its self-Coulomb is
removed), so the free-charge width acts only BETWEEN units.  There it must describe the ion's own
charge distribution as the other ions see it, not the neutral atom's diffuse frontier orbital.
The hardness rule gave sigma = 1.9 A (O) to 4.1 A (Al), so formal +/-2 MgO kept only part of its
Madelung energy (B3: formal ionic MgO +856 eV above neutral, measured, P3).

Rule ``width='ionic'`` (NEW-TO-TRACE mapping, ESTABLISHED radii):

    sigma_Z = sqrt(2/5) R_Z * exp(L tanh(s_Z / L)),   L = ln 2,  s_Z learned (init 0)

``R_Z`` is Shannon's CRYSTAL radius (R. D. Shannon, Acta Cryst. A32, 751 (1976); 6-coordinate),
which Shannon recommended as the physical size of an ion in a solid.  ``sqrt(2/5)`` matches the second
moment of a uniformly charged sphere of radius R (<r^2> = 3R^2/5) to the code's Gaussian, whose
potential is erf(r/sigma)/r, i.e. density (pi sigma^2)^(-3/2) exp(-r^2/sigma^2), <r^2> = 3 sigma^2/2.

Pre-registered criterion: formal +/-2 rocksalt MgO (a = 4.212 A) reproduces the point-charge Madelung
energy -alpha q^2 K_E / d (alpha = 1.7475645946...) to within 1 %.
"""
import math

import numpy as np
import pytest
import torch
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

K_E = 14.399645
MADELUNG_ROCKSALT = 1.7475645946331822        # NaCl structure, nearest-neighbour distance
#: Shannon (1976) crystal radii, 6-coordinate, A (the values the module must hold)
CRYSTAL_RADII = {3: 0.90, 11: 1.16, 19: 1.52, 12: 0.86, 20: 1.14, 13: 0.675, 47: 1.29, 79: 1.51,
                 8: 1.26, 9: 1.19, 17: 1.67}


def test_the_gaussian_of_width_sigma_has_second_moment_three_halves_sigma_squared():
    """The code's kernel erf(r/sigma)/r is the potential of (pi sigma^2)^(-3/2) exp(-r^2/sigma^2):
    check by Poisson (rho = -(1/4 pi r) d^2(r phi)/dr^2) and the moment by quadrature."""
    sigma = 0.8
    r = np.linspace(1e-3, 8.0, 80001)
    h = r[1] - r[0]
    from scipy.special import erf
    r_phi = erf(r / sigma)
    rho = -np.gradient(np.gradient(r_phi, h), h) / (4 * math.pi * r)
    expected = (math.pi * sigma**2) ** -1.5 * np.exp(-(r / sigma) ** 2)
    interior = slice(100, -100)
    assert np.max(np.abs(rho[interior] - expected[interior])) <= 1e-5 * expected.max()
    trapezoid = getattr(np, "trapezoid", None) or np.trapz   # numpy >= 2.0 removed trapz
    second = trapezoid(4 * math.pi * r**4 * expected, r)
    assert abs(second - 1.5 * sigma**2) <= 1e-9


def test_ionic_widths_are_the_moment_matched_crystal_radii_and_bounded():
    from trace_redox.heads import IonicFreeChargeWidth
    torch.manual_seed(0)
    width = IonicFreeChargeWidth()
    species = torch.tensor(sorted(CRYSTAL_RADII))
    sigma = width(species)
    expected = torch.tensor([math.sqrt(0.4) * CRYSTAL_RADII[int(z)] for z in species], dtype=torch.float64)
    assert torch.allclose(sigma, expected, rtol=1e-15, atol=0.0)          # s = 0 at initialisation
    with torch.no_grad():
        width.log_factor.fill_(1e6)
    assert torch.allclose(width(species), 2.0 * expected, rtol=1e-15, atol=0.0)
    with torch.no_grad():
        width.log_factor.fill_(-1e6)
    assert torch.allclose(width(species), 0.5 * expected, rtol=1e-15, atol=0.0)


def test_an_element_without_a_tabulated_ion_is_refused():
    from trace_redox.heads import IonicFreeChargeWidth
    with pytest.raises(ValueError, match="C"):
        IonicFreeChargeWidth()(torch.tensor([6, 8]))


def _formal_mgo_energy(sigma_mg, sigma_o, r_cut):
    """1/2 q^T j q per formula unit for formal +2/-2 rocksalt MgO: the lattice energy of the Gaussian
    charges without each one's isolated self-energy -- the point-charge Madelung energy as sigma -> 0."""
    from trace_electrostatics.ewald import EwaldParams
    from trace_electrostatics.lattice import gaussian_lattice_blocks
    a = 4.212
    frac = torch.tensor([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                         [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]], dtype=torch.float64)
    cell = a * torch.eye(3, dtype=torch.float64)
    pos = frac @ cell
    sigma = torch.tensor([sigma_mg] * 4 + [sigma_o] * 4, dtype=torch.float64)
    q = torch.tensor([2.0] * 4 + [-2.0] * 4, dtype=torch.float64)
    ewald = EwaldParams.auto_tune(cell, r_cut, accuracy=1e-12)
    j = gaussian_lattice_blocks(pos, cell, sigma, sigma, torch.ones(8, dtype=torch.float64), ewald)["j"]
    return float(0.5 * q @ j @ q) / 4.0, -MADELUNG_ROCKSALT * 4.0 * K_E / (a / 2)


def test_point_like_charges_give_the_madelung_energy_exactly():
    """The test's own construction: tiny widths must reproduce the point-charge Madelung energy."""
    energy, madelung = _formal_mgo_energy(0.02, 0.02, 10.0)
    assert abs(energy / madelung - 1.0) <= 1e-9


def test_formal_mgo_keeps_its_madelung_energy_with_ionic_widths():
    from trace_redox.heads import FreeChargeWidth, IonicFreeChargeWidth
    torch.manual_seed(0)
    ionic = IonicFreeChargeWidth()(torch.tensor([12, 8]))
    energy, madelung = _formal_mgo_energy(float(ionic[0]), float(ionic[1]), 12.0)
    assert abs(energy / madelung - 1.0) <= 0.01, (energy, madelung)
    # the defect it replaces: the hardness rule at its initial Hartree fraction loses a large part
    hardness = FreeChargeWidth(h_init=0.5)(torch.tensor([12, 8]))
    energy_h, _ = _formal_mgo_energy(float(hardness[0]), float(hardness[1]), 30.0)
    assert abs(energy_h / madelung - 1.0) > 0.2, (energy_h, madelung)


# ---- the bound-sector elimination and the module with ionic widths --------------------------------

from redox_helpers import cluster, edges, lattice_graph, make_heads, periodic_edges  # noqa: E402

TRICLINIC = torch.tensor([[7.0, 0.3, 0.0], [0.2, 6.5, 0.1], [0.0, 0.4, 7.5]])
SLAB = torch.tensor([[7.0, 0.3, 0.0], [0.2, 6.5, 0.0], [0.0, 0.0, 20.0]])
Z_IONIC = torch.tensor([8, 12, 12, 11, 17, 13])                  # O Mg Mg Na Cl Al: all tabulated


def _config():
    from trace_electrostatics.config import ElectrostaticConfig
    return ElectrostaticConfig(ewald_accuracy=1e-12)


def _periodic_assembly(pbc, cell, quadrupoles=False, seed=4, n=6):
    from trace_redox.bound import assemble_neutral
    heads = make_heads(quadrupoles=quadrupoles)
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(n, 3, generator=g) * torch.tensor([6.0, 6.0, 4.0])
    h = torch.randn(n, heads.irreps_in.dim, generator=g)
    ei, es = lattice_graph(pos, cell, pbc)
    vec, length = periodic_edges(pos, cell, ei, es)
    return assemble_neutral(h, pos, ei, vec, length, heads, _config(), pbc=pbc, cell=cell, edge_shift=es,
                            quadrupoles=quadrupoles)


def _open_assembly(seed=3, n=6):
    from trace_redox.bound import assemble_neutral
    heads = make_heads()
    pos, h, ei = cluster(n, heads.irreps_in.dim, seed)
    vec, length = edges(pos, ei)
    return assemble_neutral(h, pos, ei, vec, length, heads, _config())


@pytest.mark.parametrize("pbc,cell", [((True, True, True), TRICLINIC), ((True, True, False), SLAB)])
@pytest.mark.parametrize("quadrupoles", [False, True])
def test_the_ewald_route_equals_the_plane_wave_route(pbc, cell, quadrupoles):
    """sigma_self = sigma_free: the Ewald-split route is the plane-wave Gram route, to the accuracy."""
    from trace_redox.bound import eliminate_bound_sector
    a = _periodic_assembly(pbc, cell, quadrupoles)
    g = torch.Generator().manual_seed(1)
    sigma_free = 1.1 + 0.3 * torch.rand(a.n_atoms, generator=g)
    plane = eliminate_bound_sector(a, sigma_free)
    ewald = eliminate_bound_sector(a, sigma_free, sigma_self=sigma_free)
    for name in ("J", "A", "W", "u0"):
        x, y = getattr(ewald, name), getattr(plane, name)
        assert float((x - y).abs().max()) <= 1e-10 * float(y.abs().max()), name


@pytest.mark.parametrize("periodic", [True, False])
def test_only_the_same_image_on_site_terms_take_the_self_width(periodic):
    """J_ii gains gram_self(sigma_self) - gram_self(sigma_free); the on-site free-bound term, which
    enters A through the incidence B, gains kernel_self(sqrt(ss^2 + sb^2)) - kernel_self(sqrt(sf^2 + sb^2));
    nothing else moves."""
    from trace_electrostatics.kernels import gram_self_energy, kernel_self
    from trace_redox.bound import eliminate_bound_sector
    a = _periodic_assembly((True, True, True), TRICLINIC) if periodic else _open_assembly()
    n, nb = a.n_atoms, a.n_bonds
    g = torch.Generator().manual_seed(2)
    sigma_free = 0.5 + 0.4 * torch.rand(n, generator=g)
    sigma_self = 1.6 + 0.8 * torch.rand(n, generator=g)
    pair = eliminate_bound_sector(a, sigma_free, sigma_self=sigma_free)
    split = eliminate_bound_sector(a, sigma_free, sigma_self=sigma_self)
    sb = a.params.sigma
    d_J = gram_self_energy(sigma_self) - gram_self_energy(sigma_free)
    d_fb = kernel_self(torch.sqrt(sigma_self**2 + sb**2)) - kernel_self(torch.sqrt(sigma_free**2 + sb**2))
    scale = float(pair.J.abs().max())
    assert float((split.J - pair.J - torch.diag(d_J)).abs().max()) <= 64 * 2.2e-16 * scale
    expected_A = pair.A.clone()
    expected_A[:, :nb] = expected_A[:, :nb] + d_fb.unsqueeze(1) * a._dense_incidence()
    assert float((split.A - expected_A).abs().max()) <= 64 * 2.2e-16 * float(pair.A.abs().max())
    if not periodic:                                   # the open route itself is unchanged
        from trace_redox.bound import free_charge_blocks
        J, A = free_charge_blocks(a, sigma_free)
        assert torch.equal(pair.J, J) and torch.equal(pair.A, A)


def test_compact_pairs_and_a_convex_ladder_are_incompatible_in_an_ionic_crystal():
    """The proposition behind the refusal below (measured 2026-10-09).  A convex ladder (exact
    isolated-ion energies) caps the on-site curvature at U_Z = IE - EA; compact pair widths make the
    Madelung field of formal rocksalt MgO alpha K_E / d = 11.95 eV/e^2.  The alternating charge mode
    then has negative curvature for every allowed on-site value, so the Gram operator is indefinite
    and the convex mean-field solver cannot hold localised ionic charges: B3 needs integer
    configurations (audit step C2), not fractional ones."""
    from trace_electrostatics.ewald import EwaldParams
    from trace_electrostatics.lattice import free_charge_lattice_blocks
    from trace_redox.constants import ATOMIC_ELECTRON_AFFINITY_EV as EA, ATOMIC_IONIZATION_EV as IE
    from trace_redox.heads import IonicFreeChargeWidth
    a = 4.212
    frac = torch.tensor([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                         [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]], dtype=torch.float64)
    cell = a * torch.eye(3, dtype=torch.float64)
    z = torch.tensor([12] * 4 + [8] * 4)
    U = torch.tensor([IE[int(x)] - EA[int(x)] for x in z], dtype=torch.float64)
    torch.manual_seed(0)
    ones = torch.ones(8, dtype=torch.float64)
    pair = free_charge_lattice_blocks(frac @ cell, cell, IonicFreeChargeWidth()(z).detach(), ones, ones, ones,
                                      EwaldParams.auto_tune(cell, 12.0, accuracy=1e-12))["j"]
    alternating = torch.tensor([1.0] * 4 + [-1.0] * 4, dtype=torch.float64)
    J = pair + torch.diag(U)                                   # the largest on-site curvature a convex ladder allows
    assert float(alternating @ J @ alternating) < 0.0


def test_the_mean_field_module_refuses_ionic_widths():
    """No mean-field mode for ionic widths: it would refuse in every ionic solid (proposition above)."""
    from trace_redox import RedoxConfig
    with pytest.raises(ValueError, match="C2"):
        RedoxConfig(width="ionic")
