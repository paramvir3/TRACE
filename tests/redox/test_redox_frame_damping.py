"""SPEC 34 (2026-11-07; written red before the implementation): the overlap damping of the frame-charge -> bound-dipole
coupling on nearest-neighbour frame pairs.  Measured on the embedded Mg2+-O2- pair (ALMO, PBE/def2-tzvp; theory note
section 4e): the O2- dipole's response to its own cation is s(R) = 0.61 + 0.37 (R - 2.128 A) of the point-field value
over R = 1.93-2.33 A.  Built as SPEC 26's construction on the frame: the frame charge f_i drives the bound dipoles
through A + Delta A with Delta A[i, mu_j] = (s(R_ij) - 1) K lambda_3 (r_i - r_j)_c / R^3 ell_j on the tabulated frame
pairs within the first-shell window, so y0 -> y0 - K^-1 v, u0 -> u0 - A K^-1 v, constant -> constant + v.y0 - 1/2 v K^-1 v
with v = Delta A^T f.  eps_inf (the bound sector's own response) is untouched by construction; on step 20's bulk MgO the
frozen term moved Z*(Mg) from 2.43 to 1.93 at unchanged eps_inf (pre-registered, passed)."""
import math

import pytest
import torch
from redox_helpers import lattice_graph, make_heads, periodic_edges
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)

EPS = torch.finfo(torch.float64).eps
PBC = (True, True, True)


def mgo_cell(a=4.265, rep=1):
    from ase.build import bulk
    at = bulk("MgO", "rocksalt", a=a, cubic=True).repeat((rep, rep, rep))
    return torch.tensor(at.numbers), torch.tensor(at.positions, dtype=torch.float64), torch.tensor(at.cell.array, dtype=torch.float64)


def test_the_law_is_the_measured_one_and_switches_off_beyond_the_first_shell():
    from trace_redox.frame_damping import damping_factor, LAWS
    s0, s1, R0 = LAWS[(12, 8)][:3]
    assert (s0, s1, R0) == (0.61, 0.37, 2.128)
    R = torch.tensor([1.928, 2.128, 2.328, 2.6, 3.2, 4.0], dtype=torch.float64)
    s = damping_factor(R, 12, 8)
    assert abs(float(s[1]) - 0.61) <= 8 * EPS and abs(float(s[0]) - (0.61 - 0.37 * 0.2)) <= 8 * EPS
    assert float(s[-1]) == 1.0 and float(s[-2]) == 1.0                   # beyond the window: no correction
    assert bool((s[:-2] <= 1.0).all()) and bool((s > 0.0).all())
    assert damping_factor(R, 13, 8).allclose(torch.ones_like(R))        # an untabulated pair (Al3+-O2-) is left alone


def test_the_closed_form_shift_of_the_elimination():
    """y0 - K^-1 v, u0 - A K^-1 v and the constant from v = Delta A^T f, against an independent pair formula on the
    8-atom cell (dense method); the law s = 1 leaves everything bit-identical."""
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_electrostatics.solver import gaussian_widths
    from trace_redox.bound import assemble_neutral, eliminate_bound_sector
    from trace_redox.constants import formal_frame
    from trace_redox.frame_damping import damping_vector, frame_induction_damping, damping_factor
    from trace_redox.penetration import K_E
    z, pos, cell = mgo_cell()
    heads = make_heads(); torch.manual_seed(3)
    ei, es = lattice_graph(pos, cell, PBC, r_max=4.0, shells=1)
    vec, length = periodic_edges(pos, cell, ei, es)
    h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    cfg = ElectrostaticConfig()
    asm = assemble_neutral(h, pos, ei, vec, length, heads, cfg, pbc=PBC, cell=cell, edge_shift=es, species=z, total_charge=0.0)
    sigma_free = torch.full((len(z),), 0.8, dtype=torch.float64)
    elim = eliminate_bound_sector(asm, sigma_free, method="dense", sigma_self=sigma_free)
    f = formal_frame(z).to(torch.float64)
    w_mu, _ = gaussian_widths(asm.params, cfg.gaussian_a)
    v_sym = damping_vector(elim, z, f, sigma_free, w_mu, asm.params.ell, pos, cell, PBC)
    assert float(v_sym.abs().max()) <= 1e-12                                   # the perfect crystal: every site centrosymmetric
    # a distorted cell (one O off its site): the independent pair formula, every image within the window
    pos = pos.clone(); pos[int((f < 0).nonzero()[0]), 2] += 0.13
    vec, length = periodic_edges(pos, cell, ei, es)
    asm = assemble_neutral(h, pos, ei, vec, length, heads, cfg, pbc=PBC, cell=cell, edge_shift=es, species=z, total_charge=0.0)
    elim = eliminate_bound_sector(asm, sigma_free, method="dense", sigma_self=sigma_free)
    w_mu, _ = gaussian_widths(asm.params, cfg.gaussian_a)
    v = damping_vector(elim, z, f, sigma_free, w_mu, asm.params.ell, pos, cell, PBC)
    from torch.special import gammainc
    v_ref = torch.zeros_like(v); nb = int(elim.n_bonds)
    for i in range(len(z)):
        for j in range(len(z)):
            if f[i] * f[j] >= 0:
                continue
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    for c in (-1, 0, 1):
                        d = pos[i] - pos[j] - torch.tensor([a, b, c], dtype=torch.float64) @ cell; R = float(d.norm())   # r_i - (r_j + T)
                        if R >= 3.0:
                            continue
                        zc, za = (int(z[i]), int(z[j])) if f[i] > 0 else (int(z[j]), int(z[i]))
                        sfac = float(damping_factor(torch.tensor([R]), zc, za)[0])
                        lam = float(gammainc(torch.tensor(1.5), torch.tensor((R ** 2) / (float(sigma_free[i]) ** 2 + float(w_mu[j]) ** 2))))
                        v_ref[nb + 3 * j: nb + 3 * j + 3] += (sfac - 1.0) * K_E * lam * d / R ** 3 * float(asm.params.ell[j]) * f[i]
    assert float(v.abs().max()) > 1e-3
    assert torch.allclose(v, v_ref, atol=1e-10, rtol=1e-10), float((v - v_ref).abs().max())
    damped = frame_induction_damping(elim, z, f, sigma_free, w_mu, asm.params.ell, pos, cell, PBC)
    Kinv_v = elim.operator.solve(v.unsqueeze(1)).squeeze(1)
    assert torch.allclose(damped.y0, elim.y0 - Kinv_v, atol=1e-12)
    assert torch.allclose(damped.u0, elim.u0 - elim.A @ Kinv_v, atol=1e-12)
    assert abs(float(damped.constant) - (float(elim.constant) + float(v @ elim.y0) - 0.5 * float(v @ Kinv_v))) <= 1e-10
    assert torch.equal(damped.W, elim.W) and torch.equal(damped.A, elim.A)
    unit = frame_induction_damping(elim, z, f, sigma_free, w_mu, asm.params.ell, pos, cell, PBC, laws={})
    assert torch.equal(unit.y0, elim.y0) and torch.equal(unit.u0, elim.u0) and float(unit.constant) == float(elim.constant)


def test_the_module_flag_changes_energies_and_forces_but_not_eps_inf():
    """RedoxConfig(frame_induction_damping=True) is off by default and needs anchor='formal'; on the 8-atom MgO cell it
    changes the energy and the forces (finite differences agree, double backward finite) and leaves the bound
    state's eps_inf bit-identical."""
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_electrostatics.observables import dielectric_tensor
    from trace_redox import RedoxConfig, RedoxElectrostatics
    assert RedoxConfig().frame_induction_damping is False
    with pytest.raises(ValueError):
        RedoxConfig(frame_induction_damping=True, anchor="atomic")
    z, pos0, cell = mgo_cell()
    pos0[int((z == 8).nonzero()[0]), 2] += 0.13                            # off the centrosymmetric site: v != 0
    ei, es = lattice_graph(pos0, cell, PBC, r_max=4.0, shells=1)
    data = {"z": z, "total_charge": 0.0, "pbc": PBC}
    out = {}
    for flag in (False, True):
        heads = make_heads(); torch.manual_seed(1)
        cfg = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, frame_induction_damping=flag)
        module = RedoxElectrostatics(heads, ElectrostaticConfig(), cfg)
        h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)

        def energy(p):
            v, L = periodic_edges(p, cell, ei, es)
            return module(h, p, ei, v, L, data=data, cell=cell, edge_shift=es, pbc=PBC)[0]
        pos = pos0.clone().requires_grad_(True)
        E = energy(pos); g, = torch.autograd.grad(E, pos, create_graph=True)
        (g ** 2).sum().backward(); assert bool(torch.isfinite(pos.grad).all())
        v0, L0 = periodic_edges(pos0, cell, ei, es)
        st = module.bound_state(h, pos0, ei, v0, L0, data=data, cell=cell, edge_shift=es, pbc=PBC)
        out[flag] = (float(E), g.detach(), dielectric_tensor(st.solve_result, st.assembly).detach())
    assert abs(out[True][0] - out[False][0]) > 1e-6
    assert torch.equal(out[True][2], out[False][2])                         # eps_inf untouched by construction
    # finite differences of the damped energy on one O along z
    heads = make_heads(); torch.manual_seed(1)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, frame_induction_damping=True))
    h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    def energy(p):
        v, L = periodic_edges(p, cell, ei, es)
        return module(h, p, ei, v, L, data=data, cell=cell, edge_shift=es, pbc=PBC)[0]
    pos = pos0.clone().requires_grad_(True); g, = torch.autograd.grad(energy(pos), pos)
    i = int((z == 8).nonzero()[0])
    plus, minus = pos0.clone(), pos0.clone(); plus[i, 2] += 1e-5; minus[i, 2] -= 1e-5
    fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
    assert abs(fd - float(g[i, 2])) <= 1e-6 * (1.0 + abs(fd)), (fd, float(g[i, 2]))
