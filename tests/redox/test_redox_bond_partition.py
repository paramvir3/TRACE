"""One degree of freedom, one sector: bond charge flow (``RedoxConfig(bond_partition=True)``; trace_redox SPEC 22).

Charge flow along a bond i-j appears twice in the formal model: as the free sector's integer transfer
configurations (n_i, n_j) = (+-1, -+1) mixed by tau_ij (theory note 2026-10-16, Theorem A: the exact two-state
functional of a covalent bond), and as the bound sector's split-charge variable t_b with a learned hardness kappa_b.
The bound one screens the free ionic pair.  Measured 2026-10-18 on Au2 at 2.47 A (step 12 weights): the bond
channel alone screens Au+Au- by -3.41 eV (seed 42, kappa 13.3 eV) and -0.85 eV (seed 43, kappa 23.0 eV), so
Delta_ion = E(Au+Au-) - E(Au0Au0) is +0.17 against +2.47 eV; without the bond channel, +2.74 and +3.00 eV.

SPEC 22 (NEW-TO-TRACE): the split-charge variable of bond b = (i, j) is kept only when i and j lie in one covalent
unit of more than full_unit_atoms atoms (SPEC 19), whose internal charge-transfer pairs the free sector excludes;
everywhere else, charge flow between units is the free sector's, and f_b -> 0 (the dissociation envelope, so every
consumer of the incidence sees it: K, the coupling to the free charges, the seed, the periodic bond dipole).

Tests: with every atom its own unit the module is the bound sector without bonds exactly; a covalent unit keeps its
internal bonds and loses the rest; the ion pair of a dimer is screened by its moments alone; forces are minus the
gradient; bound_state sees the same partition; off by default; validated.
"""
import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)


def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True, **redox)
    return RedoxElectrostatics(heads, ElectrostaticConfig(), config), heads


POS = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0],
                    [2.0, 2.1, 0.2]], dtype=torch.float64)
Z = torch.tensor([8, 79, 79, 12, 13, 8])


def _energy(module, heads, pos, z, h=None):
    h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64) \
        if h is None else h
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    return module(h, pos, ei, vec, length, data={"z": z, "total_charge": 0.0})


def test_with_atom_units_it_is_the_bound_sector_without_bonds(monkeypatch):
    import trace_electrostatics.energy as E
    module, heads = _module(bond_partition=True)
    with_partition = float(_energy(module, heads, POS, Z)[0])
    reference_module, _ = _module()
    full = float(_energy(reference_module, heads, POS, Z)[0])
    monkeypatch.setattr(E, "bond_envelope", lambda bond_length, cutoff, onset=None: torch.zeros_like(bond_length))
    no_bonds = float(_energy(reference_module, heads, POS, Z)[0])
    assert abs(with_partition - no_bonds) <= 1e-12 * (1 + abs(no_bonds))
    assert abs(with_partition - full) > 1e-4                      # the bonds did carry energy here


def test_a_covalent_unit_keeps_its_internal_bonds_and_loses_the_rest():
    """A 5-carbon chain (one covalent unit > full_unit_atoms = 3) next to Na and Cl: the partitioned assembly keeps
    f_b on every C-C bond and zeroes every bond that touches Na or Cl."""
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox.bound import assemble_neutral
    module, heads = _module(bond_partition=True, carriers="covalent_units")
    pos = torch.tensor([[1.30 * k, 0.0, 0.0] for k in range(5)] + [[1.3, 2.6, 0.0], [3.9, 2.6, 0.0]],
                       dtype=torch.float64)
    z = torch.tensor([6, 6, 6, 6, 6, 11, 17])
    h = torch.randn(len(z), heads.irreps_in.dim, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    a = assemble_neutral(h, pos, ei, vec, length, heads, ElectrostaticConfig(), species=z)
    p = module._partition_bonds(a, z, ei, length)
    i, j = a.bond_index
    carbon = (z[i] == 6) & (z[j] == 6)
    assert bool((carbon).any()) and bool((~carbon).any())
    assert torch.equal(p.envelope[carbon], a.envelope[carbon])
    assert float(p.envelope[~carbon].abs().max()) == 0.0
    assert torch.equal(p.incidence[:, carbon], a.incidence[:, carbon])
    assert float(p.incidence[:, ~carbon].abs().max()) == 0.0


def test_the_ion_pair_of_a_dimer_is_screened_by_its_moments_alone():
    """Au2 at 2.47 A: W - J on n = (+1, -1) equals the moments-only Schur complement of the dense bound operator."""
    import trace_redox.bound as Bd
    from trace_electrostatics.config import ElectrostaticConfig
    module, heads = _module(bond_partition=True)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.47]], dtype=torch.float64)
    z = torch.tensor([79, 79])
    h = torch.randn(2, heads.irreps_in.dim, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    ei = neighbour_list(pos)
    vec, length = edges(pos, ei)
    a = Bd.assemble_neutral(h, pos, ei, vec, length, heads, ElectrostaticConfig(), species=z)
    sigma = module.free_widths(z).to(torch.float64)
    n = torch.tensor([1.0, -1.0], dtype=torch.float64)
    full = Bd.eliminate_bound_sector(a, sigma, method="dense")
    part = Bd.eliminate_bound_sector(module._partition_bonds(a, z, ei, length), sigma, method="dense")
    K, A, nb = full.operator.K, full.A, full.n_bonds
    v = A.transpose(0, 1) @ n
    moments_only = -float(v[nb:] @ torch.linalg.solve(K[nb:, nb:], v[nb:]))
    assert nb >= 1
    assert abs(float(n @ (part.W - part.J) @ n) - moments_only) <= 1e-10
    assert abs(float(n @ (full.W - full.J) @ n) - moments_only) > 1e-3      # the bond channel screens without it


def test_forces_are_minus_the_gradient_with_the_partition():
    module, heads = _module(bond_partition=True)
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(POS)

    def energy(p):
        vec, length = edges(p, ei)
        return module(h, p, ei, vec, length, data={"z": Z, "total_charge": 0.0})[0]

    pos = POS.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(pos), pos)
    for i, axis in [(0, 2), (1, 2), (3, 0), (5, 1)]:
        plus, minus = POS.clone(), POS.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(plus)) - float(energy(minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-7 * (1.0 + abs(fd))


def test_bound_state_sees_the_same_partition(monkeypatch):
    import trace_electrostatics.energy as E
    module, heads = _module(bond_partition=True)
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(POS)
    vec, length = edges(POS, ei)
    data = {"z": Z, "total_charge": 0.0}
    energy_bound = float(module(h, POS, ei, vec, length, data=data)[1].energy_bound)
    state = module.bound_state(h, POS, ei, vec, length, data=data)
    assert abs(float(state.energy) - energy_bound) <= 1e-10 * (1 + abs(energy_bound))
    reference, _ = _module()
    monkeypatch.setattr(E, "bond_envelope", lambda bond_length, cutoff, onset=None: torch.zeros_like(bond_length))
    no_bonds = reference.bound_state(h, POS, ei, vec, length, data=data)
    assert abs(float(state.energy) - float(no_bonds.energy)) <= 1e-12 * (1 + abs(float(no_bonds.energy)))


def test_it_is_off_by_default_and_validated():
    from trace_redox import RedoxConfig
    assert RedoxConfig().bond_partition is False
    with pytest.raises(ValueError, match="bond_partition"):
        RedoxConfig(bond_partition=True)                          # needs the formal frames
