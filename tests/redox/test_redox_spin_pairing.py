"""Spin-pairing factors on the configuration couplings (``RedoxConfig(spin_pairing=True)``; trace_redox SPEC 29).

The configurations are spinless: one local orbital per charge step, coupled by tau for every one-electron move.  For
the two-site Hubbard model (ESTABLISHED) the spinful matrix elements of the lowest-spin states are
- 2 electrons (singlet): covalent <-> symmetric ionic 2 t, so E0 = U/2 - sqrt(U^2/4 + 4 t^2);
- 1 or 3 electrons (doublet): the resonance t, so E0 = -t (or U - t).
The spinless model couples the neutral configuration to each ionic one with tau, i.e. sqrt(2) tau to their symmetric
combination, and the resonances with tau: no single tau is right for both (neutral needs tau = sqrt 2 t, the ions
tau = t).  Measured consequence (2026-10-18): the learned tau(Au, Au) cannot serve the undoped (ionic-covalent) and
doped (resonance) Au2 physics at once.

SPEC 29 (NEW-TO-TRACE): a move pairs or unpairs two singly occupied frontier orbitals exactly when donor and
acceptor hold electron counts of the same parity before it; such moves carry f = sqrt 2, every other f = 1.  Then the
spinless model reproduces the two-site lowest-spin energies exactly at every electron count with one tau = t.

Tests: the spinful Fock-space Hubbard ground energies for N = 1, 2, 3 and several U / t, against the lowest eigenvalue
of the spinless Hamiltonian with the factors (and the defect without them); module forces -grad E; validated.
"""
import itertools
import math

import pytest
import torch
from redox_helpers import edges, make_heads, neighbour_list
from redox_helpers import redox_float64  # noqa: F401  (autouse fixture)


def _hubbard_ground(U, t, N):
    """Exact two-site Hubbard ground energy in the N-electron sector (spin orbitals 0 = 1up, 1 = 1dn, 2 = 2up,
    3 = 2dn), by explicit fermionic Fock-space construction."""
    states = [s for s in range(16) if bin(s).count("1") == N]
    index = {s: k for k, s in enumerate(states)}
    H = torch.zeros(len(states), len(states), dtype=torch.float64)

    def hop(state, p, q):          # c_p^dag c_q |state>, with the Jordan-Wigner sign; None if zero
        if not (state >> q) & 1 or ((state >> p) & 1 and p != q):
            return None
        sign = (-1) ** bin(state & ((1 << q) - 1)).count("1")
        s1 = state & ~(1 << q)
        sign *= (-1) ** bin(s1 & ((1 << p) - 1)).count("1")
        return sign, s1 | (1 << p)

    for s in states:
        k = index[s]
        H[k, k] += U * (((s >> 0) & 1) * ((s >> 1) & 1) + ((s >> 2) & 1) * ((s >> 3) & 1))
        for p, q in ((0, 2), (2, 0), (1, 3), (3, 1)):
            r = hop(s, p, q)
            if r is not None:
                H[index[r[1]], k] += -t * r[0]
    return float(torch.linalg.eigvalsh(H)[0])


def _spinless_ground(U, t, N, electrons):
    """The lowest eigenvalue of TRACE's configuration Hamiltonian for the same problem: two units, charge 0 = one
    electron each, charges -1 .. +1, E = U x (number of doubly occupied units), T = -t."""
    from trace_redox.diabatic import diabatic_hamiltonian
    Q = 2 - N
    configs = [n for n in itertools.product((-1, 0, 1), repeat=2) if sum(n) == Q]
    E = torch.tensor([U * sum(1 for x in n if x == -1) for n in configs], dtype=torch.float64)
    T = torch.tensor([[0.0, -t], [-t, 0.0]], dtype=torch.float64)
    H = diabatic_hamiltonian(E, torch.tensor(configs), torch.tensor([1, 1]), T, electrons=electrons)
    return float(torch.linalg.eigvalsh(H)[0])


@pytest.mark.parametrize("U,t", [(1.0, 1.0), (4.0, 1.0), (0.3, 1.2), (7.0, 0.5)])
def test_the_spinless_model_with_pairing_factors_is_the_two_site_hubbard_model(U, t):
    electrons = torch.tensor([1, 1])
    for N in (1, 2, 3):
        exact = _hubbard_ground(U, t, N)
        assert abs(_spinless_ground(U, t, N, electrons) - exact) <= 1e-12 * (1.0 + abs(exact)), (U, t, N)
    # the defect without the factors: the neutral singlet is under-bound, the ions are exact
    assert _spinless_ground(U, t, 2, None) > _hubbard_ground(U, t, 2) + 1e-6
    for N in (1, 3):
        assert abs(_spinless_ground(U, t, N, None) - _hubbard_ground(U, t, N)) <= 1e-12 * (1.0 + abs(U))


def test_the_factor_is_sqrt2_exactly_when_the_parities_agree():
    from trace_redox.diabatic import one_electron_moves, spin_pairing_factors
    configs = torch.tensor([[0, 0], [1, -1], [-1, 1], [1, 0], [0, 1]])
    rows, cols, donor, acceptor, _ = one_electron_moves(configs, torch.tensor([1, 1]))
    f = spin_pairing_factors(configs, rows, donor, acceptor, torch.tensor([79, 79]))
    N = torch.tensor([79, 79]).unsqueeze(0) - configs
    same = (N[rows, donor] % 2) == (N[rows, acceptor] % 2)
    assert torch.equal(f, torch.where(same, torch.full_like(f, math.sqrt(2.0)), torch.ones_like(f)))
    assert bool((f[same] == math.sqrt(2.0)).all()) and bool(same.any()) and bool((~same).any())


def _module(**redox):
    from trace_electrostatics.config import ElectrostaticConfig
    from trace_redox import RedoxConfig, RedoxElectrostatics
    heads = make_heads()
    torch.manual_seed(1)
    config = RedoxConfig(fragment_ensemble="diabatic", anchor="formal", max_charge=3, couplings=True, **redox)
    module = RedoxElectrostatics(heads, ElectrostaticConfig(), config)
    with torch.no_grad():
        module.coupling.pair[-1].weight.normal_(0.0, 0.3, generator=torch.Generator().manual_seed(4))
    return module, heads


def test_module_forces_are_minus_the_gradient_with_spin_pairing():
    module, heads = _module(spin_pairing=True, diabatic_selection="amplitude", carrier_band=True)
    plain, _ = _module(diabatic_selection="amplitude", carrier_band=True)
    z = torch.tensor([8, 79, 79, 12, 13, 8])
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 2.25], [0.0, 0.2, 4.7], [2.1, 0.0, 0.1], [0.0, 2.1, 0.0],
                         [2.0, 2.1, 0.2]], dtype=torch.float64)
    h = torch.randn(6, heads.irreps_in.dim, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ei = neighbour_list(pos0)

    def energy(m, p):
        vec, length = edges(p, ei)
        return m(h, p, ei, vec, length, data={"z": z, "total_charge": 0.0})[0]

    assert abs(float(energy(module, pos0)) - float(energy(plain, pos0))) > 1e-6
    pos = pos0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(module, pos), pos)
    for i, axis in [(1, 2), (2, 2), (0, 0), (3, 1)]:
        plus, minus = pos0.clone(), pos0.clone()
        plus[i, axis] += 1e-5
        minus[i, axis] -= 1e-5
        fd = (float(energy(module, plus)) - float(energy(module, minus))) / 2e-5
        assert abs(fd - float(grad[i, axis])) <= 1e-6 * (1.0 + abs(fd))


def test_it_is_validated():
    from trace_redox import RedoxConfig
    assert RedoxConfig().spin_pairing is False
    with pytest.raises(ValueError, match="spin_pairing"):
        RedoxConfig(fragment_ensemble="diabatic", anchor="formal", spin_pairing=True)     # no couplings
