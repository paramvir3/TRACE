"""Ewald-split lattice blocks of free Gaussian charges against the bound sector (trace_redox, audit step 2).

The plane-wave Gram form of trace_redox (``periodic_free_charge_blocks``) needs k up to 12 / s_min: for
compact ionic widths that is 143 687 half-space vectors on the Au2/MgO cell (1 654 with the
hardness widths; measured 2026-10-09).  The Ewald split puts the width dependence in real space, so
its reciprocal sum depends on alpha alone.

``free_charge_lattice_blocks`` returns the PAIR PARTS (every interaction except the same-image
coincident one, i = j at zero shift), in the conventions of ``gaussian_lattice_blocks``:

    j    [n, n]     free-free charge,                 s = sqrt(sf_i^2 + sf_j^2)
    j_fb [n, n]     free charge - bound charge,       s = sqrt(sf_i^2 + sb_j^2)
    t01  [n, n, 3]  free charge - bound dipole x ell_j
    t02  [n, n, 5]  free charge - bound quadrupole x ell_j^2

The reference here is the plane-wave Gram sum written from its formula, an independent route:

    J_ij  = (4 pi K_E / V) sum_{k != 0} g_i g_j cos(k.r_ij) / k^2,  g_i = exp(-k^2 s_i^2 / 4)
    T01   = (4 pi K_E / V) sum_k g^f_i g^mu_j k_a sin(k.r_ij) / k^2 * ell_j
    T02   = -(1/2)(4 pi K_E / V) sum_k g^f_i g^Th_j (k.W_m.k) cos(k.r_ij) / k^2 * ell_j^2

which INCLUDES the same-image term 2 K_E / (sqrt(pi) s_ii) on the charge diagonals (zero for the
odd and traceless blocks).  Truncated at |k| <= 14 / s_min: tail ~ erfc(7) ~ 4e-23 relative.
"""
import itertools
import math

import pytest
import torch
from conftest import require

K_E = 14.399645


def _plane_wave(pos, cell, sf, sb, smu, sth, ell, basis):
    recip = 2 * math.pi * torch.linalg.inv(cell).T
    s_min = math.sqrt(float(sf.min()) ** 2 + min(float(sb.min()), float(smu.min()), float(sth.min()), float(sf.min())) ** 2)
    k_max = 14.0 / s_min
    reach = [int(math.ceil(k_max * float(a.norm()) / (2 * math.pi))) for a in cell]
    m = torch.tensor(list(itertools.product(*[range(-r, r + 1) for r in reach])), dtype=torch.float64)
    k = m @ recip
    k2 = (k * k).sum(-1)
    keep = (k2 > 0) & (k2 <= k_max ** 2)
    k, k2 = k[keep], k2[keep]
    w = (4 * math.pi * K_E / float(torch.det(cell).abs())) / k2
    phase = pos @ k.T
    c, s = torch.cos(phase), torch.sin(phase)
    g = lambda sigma: torch.exp(-0.25 * k2.unsqueeze(0) * (sigma * sigma).unsqueeze(1))   # [n, nk]
    gf, gb, gm, gt = g(sf), g(sb), g(smu), g(sth)
    J = (gf * c * w) @ (gf * c).T + (gf * s * w) @ (gf * s).T
    J_fb = (gf * c * w) @ (gb * c).T + (gf * s * w) @ (gb * s).T
    t01 = torch.stack([(gf * s * w * k[:, a]) @ (gm * c).T - (gf * c * w * k[:, a]) @ (gm * s).T for a in range(3)], -1)
    kk = torch.einsum("ka,mab,kb->km", k, basis, k)
    t02 = torch.stack([-0.5 * ((gf * c * w * kk[:, q]) @ (gt * c).T + (gf * s * w * kk[:, q]) @ (gt * s).T)
                       for q in range(5)], -1)
    return J, J_fb, t01 * ell.view(1, -1, 1), t02 * (ell * ell).view(1, -1, 1)


@pytest.mark.parametrize("case", ["triclinic", "slab_like"])
def test_ewald_blocks_plus_the_same_image_terms_equal_the_plane_wave_sum(case):
    lattice = require("lattice")
    EwaldParams = require("ewald", "EwaldParams")
    basis = require("kernels", "quadrupole_cartesian_basis")().to(torch.float64)
    g = torch.Generator().manual_seed(5)
    if case == "triclinic":
        cell = torch.tensor([[5.6, 0.3, 0.0], [0.2, 5.2, 0.4], [0.0, 0.5, 6.0]], dtype=torch.float64)
        frac = torch.rand(5, 3, generator=g, dtype=torch.float64)
    else:
        cell = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 11.0]], dtype=torch.float64)
        frac = torch.rand(5, 3, generator=g, dtype=torch.float64) * torch.tensor([1.0, 1.0, 0.4], dtype=torch.float64)
    pos = frac @ cell
    n = len(pos)
    sf = 1.0 + 0.4 * torch.rand(n, generator=g, dtype=torch.float64)
    sb = 0.9 + 0.4 * torch.rand(n, generator=g, dtype=torch.float64)
    smu = 0.8 + 0.4 * torch.rand(n, generator=g, dtype=torch.float64)
    sth = 1.0 + 0.3 * torch.rand(n, generator=g, dtype=torch.float64)
    ell = 0.6 + 0.6 * torch.rand(n, generator=g, dtype=torch.float64)
    ewald = EwaldParams.auto_tune(cell, 12.0, accuracy=1e-13)
    blocks = lattice.free_charge_lattice_blocks(pos, cell, sf, sb, smu, ell, ewald, sigma_theta=sth)
    J, J_fb, t01, t02 = _plane_wave(pos, cell, sf, sb, smu, sth, ell, basis)
    same_image = lambda s: 2 * K_E / (math.sqrt(math.pi) * s)
    ewald_J = blocks["j"] + torch.diag(same_image(math.sqrt(2.0) * sf))
    ewald_fb = blocks["j_fb"] + torch.diag(same_image(torch.sqrt(sf * sf + sb * sb)))
    for name, a, b in (("j", ewald_J, J), ("j_fb", ewald_fb, J_fb), ("t01", blocks["t01"], t01), ("t02", blocks["t02"], t02)):
        scale = float(b.abs().max())
        # both routes are converged to <~1e-12 relative (Ewald accuracy 1e-13 per tail; plane-wave
        # tail erfc(7)); the sums run over ~1e4 terms, so round-off is ~1e-12 of the scale
        assert float((a - b).abs().max()) <= 1e-10 * scale, (name, float((a - b).abs().max()) / scale)


def test_blocks_are_differentiable_in_positions_and_cell():
    lattice = require("lattice")
    EwaldParams = require("ewald", "EwaldParams")
    g = torch.Generator().manual_seed(8)
    cell = torch.tensor([[5.6, 0.3, 0.0], [0.2, 5.2, 0.4], [0.0, 0.5, 6.0]], dtype=torch.float64, requires_grad=True)
    pos = (torch.rand(4, 3, generator=g, dtype=torch.float64) @ cell.detach()).requires_grad_(True)
    ones = torch.ones(4, dtype=torch.float64)
    ewald = EwaldParams.auto_tune(cell.detach(), 10.0, accuracy=1e-10)

    def energy(p, c):
        b = lattice.free_charge_lattice_blocks(p, c, 0.6 * ones, 0.9 * ones, 0.8 * ones, ones, ewald)
        q = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=torch.float64)
        return 0.5 * q @ b["j"] @ q + q @ b["j_fb"] @ q.flip(0) + (b["t01"].sum(-1) @ q).sum()

    (grad,) = torch.autograd.grad(energy(pos, cell), pos)
    for atom, axis in ((0, 0), (2, 1), (3, 2)):
        h = 1e-5
        p1, p2 = pos.detach().clone(), pos.detach().clone()
        p1[atom, axis] += h
        p2[atom, axis] -= h
        fd = float((energy(p1, cell) - energy(p2, cell)) / (2 * h))
        assert abs(float(grad[atom, axis]) - fd) <= 1e-7 * (1 + abs(fd))


def test_the_width_refusal_names_a_cutoff_that_works():
    """The LatticeWidthError advice must be sufficient: r_cut >= s erfc^-1(10 accuracy), not 4 s
    (4 s = 10.8 A failed at accuracy 1e-10 for s = 2.7 A, measured 2026-10-09)."""
    import re
    lattice = require("lattice")
    EwaldParams = require("ewald", "EwaldParams")
    cell = torch.eye(3, dtype=torch.float64) * 9.0
    pos = torch.tensor([[0.0, 0.0, 0.0], [4.5, 4.5, 4.5]], dtype=torch.float64)
    wide = torch.full((2,), 2.7 / math.sqrt(2.0), dtype=torch.float64)
    ones = torch.ones(2, dtype=torch.float64)
    with pytest.raises(lattice.LatticeWidthError) as caught:
        lattice.free_charge_lattice_blocks(pos, cell, wide, wide, wide, ones, EwaldParams.auto_tune(cell, 10.0, accuracy=1e-10))
    advised = float(re.search(r"at least ([0-9.]+) A", str(caught.value)).group(1))
    lattice.free_charge_lattice_blocks(pos, cell, wide, wide, wide, ones, EwaldParams.auto_tune(cell, advised, accuracy=1e-10))
