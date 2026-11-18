"""The fixed-k solve -- SPEC sections 1.3, 1.4, 2 and 6.

Everything in this module drives ``solver.solve_charges`` directly with
hand-built parameters, so the numbers under test are the solver's, not a
neural network's.  The reference answers come from ``elec_reference``, which
shares no code with the package.
"""

import math
from fractions import Fraction

import numpy as np
import pytest
import torch

import elec_reference as ref
from conftest import require


K_E = 14.399645  # SPEC 0


def _solver():
    return require(
        "solver",
        "ElectrostaticParams",
        "Geometry",
        "FragmentConstraints",
        "GeometrySeed",
        "SolverOptions",
        "solve_charges",
    )


def _isotropic(n_atoms, value):
    return torch.eye(3).repeat(n_atoms, 1, 1) * value


def run_solve(
    positions,
    bond_index,
    chi,
    eta,
    sigma,
    kappa,
    groups,
    targets,
    n_iter=6,
    coulomb=None,
    ell=1.0e-8,
    a_mu=1.0,
    **options_kwargs,
):
    """Drive ``solve_charges`` with explicit parameters.

    ``ell`` defaults to 1e-8 A, which makes the dipole sector numerically inert
    (mu = ell * mu_hat, and the charge-dipole block carries a factor ell) without
    making the dipole block ill-conditioned the way a huge ``A_mu`` would.  The
    charge-only limit is what the SPEC section 6 QEq regressions are stated in.
    """
    (
        ElectrostaticParams,
        Geometry,
        FragmentConstraints,
        GeometrySeed,
        SolverOptions,
        solve_charges,
    ) = _solver()
    build_incidence = require("topology", "build_incidence")

    n_atoms = positions.shape[0]
    incidence, bonds = build_incidence(bond_index, n_atoms)
    n_bonds = bonds.shape[1]

    params = ElectrostaticParams(
        chi=torch.as_tensor(chi, dtype=torch.get_default_dtype()).reshape(-1),
        eta=torch.as_tensor(eta, dtype=torch.get_default_dtype()).reshape(-1),
        sigma=torch.as_tensor(sigma, dtype=torch.get_default_dtype()).reshape(-1),
        ell=torch.full((n_atoms,), float(ell)),
        kappa=torch.full((n_bonds,), float(kappa)),
        A_mu=_isotropic(n_atoms, a_mu),
    )
    geometry = Geometry(
        positions=positions,
        incidence=incidence,
        bond_index=bonds,
        coulomb_matrix=coulomb,
    )
    constraint = torch.zeros(len(groups), n_atoms)
    for row, members in enumerate(groups):
        constraint[row, list(members)] = 1.0
    constraints = FragmentConstraints(
        matrix=constraint,
        targets=torch.as_tensor(targets, dtype=torch.get_default_dtype()).reshape(-1),
    )
    seed = GeometrySeed.zeros(n_bonds, n_atoms)
    return solve_charges(
        params,
        geometry,
        constraints,
        n_iter,
        seed=seed,
        options=SolverOptions(**options_kwargs),
    )


# ===========================================================================
# SPEC 6: chain polarizability alpha = N (N^2 - 1) / 12
# ===========================================================================
@pytest.mark.parametrize("n", list(range(1, 41)))
def test_chain_formula_is_the_variance_identity(n):
    """SPEC 6 note: "the chain formula is N(N^2-1)/12, verified as an exact identity".

    Derivation (independent of the package): at J = 0 the stationarity condition
    of ``0.5 eta sum q_i^2 - E sum q_i z_i`` under ``sum q_i = 0`` is
    ``q_i = E (z_i - z_mean) / eta``, so

        P = sum_i q_i z_i = (E / eta) sum_i (z_i - z_mean)^2,
        alpha = dP/dE = (1/eta) sum_i (z_i - z_mean)^2.

    For z_i = 0, 1, ..., N-1 the sum of squared deviations is N(N^2-1)/12 --
    the variance identity.  This test checks that identity in exact integer
    arithmetic, so it can never be satisfied by accident.
    """
    # Exact rational arithmetic: no floating point anywhere, so this is an
    # identity check rather than a numerical comparison.
    mean = Fraction(n - 1, 2)
    exact_sum = sum((Fraction(i) - mean) ** 2 for i in range(n))
    assert exact_sum == Fraction(n * (n * n - 1), 12)

    # ... and the float evaluation agrees, which is what the solver is compared to.
    z = np.arange(n, dtype=float)
    assert float(((z - z.mean()) ** 2).sum()) == pytest.approx(
        n * (n * n - 1) / 12.0, rel=0, abs=1e-9
    )


@pytest.mark.parametrize("n", [2, 3, 4, 5, 10, 25, 40])
def test_the_previously_circulated_chain_formula_is_wrong(n):
    """SPEC 6 note: "A previously quoted N^3/12 - N/8 + 1/24 is WRONG".

    The two differ by exactly (N-1)/24, so they agree only at N = 1.  Locking
    this in stops the wrong formula being reintroduced as a "fix".
    """
    correct = n * (n * n - 1) / 12.0
    circulated = n**3 / 12.0 - n / 8.0 + 1.0 / 24.0
    assert correct - circulated == pytest.approx((n - 1) / 24.0, rel=1e-12)
    assert correct != pytest.approx(circulated, rel=1e-9)


@pytest.mark.parametrize("n", list(range(1, 41)))
def test_solved_chain_polarizability_matches_n_n2_minus_1_over_12(n, linear_chain):
    """SPEC 6: chain polarizability ``N(N^2-1)/12`` at eta = 1, J = 0, spacing 1.

    Units note (SPEC 0): this is a *response* polarizability in e^2 A^2 / eV,
    not a volume in A^3.  Multiplying by K_E would be the 14x error the SPEC
    warns about, so no conversion is applied here.

    eta = 1 eV/e^2 is below the hardness floor at a chemical sigma, so sigma is
    set to 20 A (floor 2 K_E / (20 sqrt(pi)) = 0.812 eV) and J is forced to zero
    through ``Geometry.coulomb_matrix``.  kappa cannot be exactly zero
    (SPEC 1.4 requires kappa_0 > 0), so it is taken to 1e-13, which perturbs
    alpha by O(kappa/eta) ~ 1e-13 relative.
    """
    chain = linear_chain(n, spacing=1.0)
    positions = chain["pos"]
    z = positions[:, 2]

    def polarization(field):
        result = run_solve(
            positions,
            chain["bond_index"],
            chi=-field * z,
            eta=torch.ones(n),
            sigma=torch.full((n,), 20.0),
            kappa=1.0e-13,
            groups=[list(range(n))],
            targets=[0.0],
            n_iter=8 * n + 20,
            coulomb=torch.zeros(n, n),
        )
        return float((result.q * z).sum())

    alpha = polarization(1.0) - polarization(0.0)  # exactly linear response
    expected = n * (n * n - 1) / 12.0

    if n == 1:
        # No bonds at all, and a single atom constrained to q = 0 has nothing
        # to respond with.  N(N^2-1)/12 = 0 there, so the formula covers the
        # degenerate case rather than excluding it -- which is worth asserting,
        # because it is the one N at which the circulated formula happens to
        # agree (see the test above) and so the only N a wrong implementation
        # could pass by accident.
        assert alpha == 0.0
        assert expected == 0.0
        return

    # Relative, because alpha spans 0.5 at N = 2 to 5330 at N = 40.  The bound
    # is 1e-9; measured worst case over the whole range is 1.6e-11 at N = 40,
    # growing like the CG condition number, so 1e-9 is ~60x of headroom.
    assert alpha == pytest.approx(expected, rel=1.0e-9), (
        f"N={n}: alpha={alpha:.12f}, SPEC requires N(N^2-1)/12={expected:.12f}"
    )
    # And it is NOT the circulated formula, which differs by exactly (N-1)/24.
    wrong = n**3 / 12.0 - n / 8.0 + 1.0 / 24.0
    assert abs(alpha - wrong) == pytest.approx((n - 1) / 24.0, rel=1.0e-6)
    assert abs(alpha - wrong) > 1.0e-3


# ===========================================================================
# SPEC 1.4 / 6: kappa -> 0 reproduces a global QEq solve to 1e-15
# ===========================================================================
def test_kappa_to_zero_reproduces_an_independent_global_qeq_solve():
    """SPEC 1.4 / 6: "``kappa -> 0`` must reproduce global QEq to 1e-15
    (audit: 2.1e-15).  This is a required regression test -- it is what keeps
    metals reachable."

    The reference side is ``elec_reference.global_qeq``: a dense numpy KKT
    solve of ``min 0.5 q^T (J + diag(eta)) q + chi^T q`` subject to
    ``1^T q = Q``.  It shares no code with the package's projected CG.

    The bond graph is a tree, so B has full column rank and its columns span
    exactly the zero-sum charge subspace -- which is what makes the SQE
    parameterisation able to reach the QEq answer at all.
    """
    positions = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [1.52, 0.00, 0.00],
            [2.10, 1.41, 0.00],
            [-0.63, 1.17, 0.35],
            [3.55, 1.62, 0.41],
        ]
    )
    bond_index = torch.tensor([[0, 1, 0, 2], [1, 2, 3, 4]])  # a tree, 4 bonds
    chi = torch.tensor([-4.7, -3.1, -5.9, -2.2, -6.4])
    eta = torch.tensor([12.0, 13.5, 11.5, 14.0, 12.7])
    sigma = torch.full((5,), 1.5)
    total_charge = 1.0

    result = run_solve(
        positions,
        bond_index,
        chi=chi,
        eta=eta,
        sigma=sigma,
        kappa=1.0e-16,
        groups=[list(range(5))],
        targets=[total_charge],
        n_iter=60,  # >> n_bonds: CG is exact after n_bonds steps in exact arithmetic
    )

    reference = ref.global_qeq(
        eta.numpy(),
        chi.numpy(),
        ref.coulomb_matrix(positions.numpy(), sigma.numpy()),
        [list(range(5))],
        [total_charge],
    )
    error = float(np.abs(result.q.detach().numpy() - reference).max())
    # SPEC section 6 requires 1e-15; the audit measured 2.1e-15.  The bound here
    # is 1e-14 to leave room for the two different (both exact) linear solves.
    assert error < 1.0e-14, (
        f"kappa -> 0 does not reproduce global QEq: max|dq| = {error:.3e} "
        f"(SPEC requires 1e-15, audit measured 2.1e-15)"
    )
    assert abs(float(result.q.sum()) - total_charge) < 1.0e-12


def test_large_kappa_departs_from_global_qeq():
    """Guard against a vacuous ``kappa -> 0`` test.

    The test above is only meaningful if the SQE solver is *capable* of
    disagreeing with global QEq -- otherwise it would pass on an implementation
    that had quietly become plain QEq and thrown the bond variables away.

    Measured on this system, max|dq| against the independent global-QEq solve:

        kappa      1e-16    1e-8     0.1      1.0      5.0      10.0
        max|dq|    1.7e-16  4.6e-10  4.5e-03  3.9e-02  1.2e-01  1.7e-01

    The whole ladder is asserted, not just its top: monotone in kappa, at
    round-off when kappa -> 0, and unmistakably different at chemical kappa.
    SPEC 1.4 is a statement about a *limit*, and a limit needs both ends.
    """
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.52, 0.0, 0.0], [2.1, 1.41, 0.0], [-0.63, 1.17, 0.35]]
    )
    bond_index = torch.tensor([[0, 1, 0], [1, 2, 3]])
    chi = torch.tensor([-4.7, -3.1, -5.9, -2.2])
    eta = torch.tensor([12.0, 13.5, 11.5, 14.0])
    sigma = torch.full((4,), 1.5)
    reference = ref.global_qeq(
        eta.numpy(), chi.numpy(),
        ref.coulomb_matrix(positions.numpy(), sigma.numpy()),
        [list(range(4))], [0.0],
    )

    departures = {}
    for kappa in (1.0e-16, 1.0e-8, 0.1, 1.0, 5.0, 10.0):
        result = run_solve(
            positions, bond_index, chi, eta, sigma, kappa=kappa,
            groups=[list(range(4))], targets=[0.0], n_iter=40,
        )
        departures[kappa] = float(
            np.abs(result.q.detach().numpy() - reference).max()
        )

    # SPEC 1.4 / 6: the limit, to 1e-15 (audit 2.1e-15; measured 1.7e-16).
    assert departures[1.0e-16] < 1.0e-14
    # The control SPEC 1.4 implies: at chemical kappa the two genuinely differ.
    assert departures[5.0] > 1.0e-2, (
        f"kappa = 5 eV/e^2 departs from global QEq by only "
        f"{departures[5.0]:.3e}; the bond hardness is not localising anything"
    )
    assert departures[10.0] > departures[5.0]
    # Monotone: more bond hardness means less charge transfer, always.
    ladder = [departures[k] for k in (1.0e-16, 1.0e-8, 0.1, 1.0, 5.0, 10.0)]
    assert ladder == sorted(ladder), f"non-monotone in kappa: {ladder}"


# ===========================================================================
# SPEC 1.3 / 6: size consistency
# ===========================================================================
SPECTATOR_COUNTS = [0, 5, 20, 50, 200]


@pytest.mark.parametrize("n_spectators", SPECTATOR_COUNTS)
def test_per_fragment_constraint_keeps_the_ion_at_exactly_plus_one(n_spectators):
    """SPEC 1.3 / 6: "size consistency, per-fragment, 0-200 spectators | 1.000000 exactly".

    A 3-atom ion constrained to +1, with 0/5/20/50/200 unbonded spectator atoms
    at 1000 A.  Each spectator is its own fragment, constrained to 0.
    """
    positions, bond_index, groups, targets = _spectator_system(n_spectators)
    n_atoms = positions.shape[0]
    result = run_solve(
        positions,
        bond_index,
        chi=torch.tensor([-4.5, -4.5, -4.5] + [-3.0] * n_spectators),
        eta=torch.tensor([12.0, 12.0, 12.0] + [14.0] * n_spectators),
        sigma=torch.full((n_atoms,), 1.5),
        kappa=0.5,
        groups=groups,
        targets=targets,
        n_iter=12,
    )
    fragment_charge = float(result.q[:3].sum())
    # SPEC 2.1 item 5 sets the per-fragment constraint tolerance at 1e-10.
    assert abs(fragment_charge - 1.0) < 1.0e-10, (
        f"{n_spectators} spectators: fragment charge {fragment_charge:.12f}, "
        "SPEC requires 1.000000 at every spectator count"
    )


@pytest.mark.parametrize("n_spectators", SPECTATOR_COUNTS)
def test_a_single_global_constraint_leaks_charge_onto_the_spectators(n_spectators):
    """SPEC 1.3: the same system with ONE global constraint is size-inconsistent.

    The audit sequence quoted by SPEC 1.3 is q -> 1.000, 0.242, 0.102, 0.052,
    0.016 for 0/5/20/50/200 spectators, for chi/eta that SPEC does not record.

    For the SQE parameterisation the degradation is not merely qualitative --
    it has a closed form, and it is the one this test locks in.  No bond crosses
    the 1000 A gap, so ``B t`` cannot move charge between the ion and the
    spectators at all; the single global constraint is met by ``rho``, which
    spreads uniformly, so the ion retains exactly its share of the atom count:

        q_ion = 3 / (3 + n_spectators)
              = 1.000, 0.375, 0.130, 0.057, 0.015   for n = 0, 5, 20, 50, 200

    Those five numbers are asserted literally, to three decimals, alongside the
    per-fragment arm's exact 1.000000.  The contrast between the two columns is
    the entire argument for per-fragment constraints, so it is pinned rather
    than described.
    """
    #: q_ion = 3/(3+n), rounded to the three decimals SPEC 1.3 states its own
    #: audit sequence to.  Derived above, not measured.
    expected_by_count = {0: 1.000, 5: 0.375, 20: 0.130, 50: 0.057, 200: 0.015}
    positions, bond_index, _, _ = _spectator_system(n_spectators)
    n_atoms = positions.shape[0]
    result = run_solve(
        positions,
        bond_index,
        chi=torch.tensor([-4.5, -4.5, -4.5] + [-3.0] * n_spectators),
        eta=torch.tensor([12.0, 12.0, 12.0] + [14.0] * n_spectators),
        sigma=torch.full((n_atoms,), 1.5),
        kappa=0.5,
        groups=[list(range(n_atoms))],  # ONE global constraint
        targets=[1.0],
        n_iter=12,
    )
    fragment_charge = float(result.q[:3].sum())
    # The closed form, to nine figures.
    expected = 3.0 / (3.0 + n_spectators)
    assert fragment_charge == pytest.approx(expected, rel=1e-9)
    # ... and the literal three-decimal sequence SPEC's audit is quoted in.
    assert round(fragment_charge, 3) == expected_by_count[n_spectators], (
        f"{n_spectators} spectators: ion holds {fragment_charge:.6f}, the "
        f"global-constraint sequence requires {expected_by_count[n_spectators]}"
    )
    if n_spectators == 0:
        assert abs(fragment_charge - 1.0) < 1.0e-10
    else:
        assert fragment_charge < 0.95


@pytest.mark.parametrize("n_spectators", SPECTATOR_COUNTS)
def test_independent_global_qeq_reproduces_the_audit_degradation(n_spectators):
    """The same contrast in plain QEq, computed independently of the package.

    This is the arm that matches the audit's mechanism (atomic charges free, one
    global constraint).  Measured here: 1.000, 0.242, 0.074, 0.031, 0.008 --
    the SPEC 1.3 audit reports 1.000, 0.242, 0.102, 0.052, 0.016 for its own
    (unrecorded) parameters.  Same collapse, same order of magnitude.
    """
    positions, _, _, _ = _spectator_system(n_spectators)
    n_atoms = positions.shape[0]
    eta = np.array([12.0] * 3 + [14.0] * n_spectators)
    chi = np.array([-4.5] * n_atoms)
    coulomb = ref.coulomb_matrix(positions.numpy(), np.full(n_atoms, 1.5))

    per_fragment_groups = [[0, 1, 2]] + [[3 + i] for i in range(n_spectators)]
    per_fragment_targets = [1.0] + [0.0] * n_spectators
    q_fragment = ref.global_qeq(eta, chi, coulomb, per_fragment_groups, per_fragment_targets)
    assert abs(float(q_fragment[:3].sum()) - 1.0) < 1.0e-12  # SPEC 1.3: 1.000000

    q_global = ref.global_qeq(eta, chi, coulomb, [list(range(n_atoms))], [1.0])
    held = float(q_global[:3].sum())
    if n_spectators == 0:
        assert held == pytest.approx(1.0, abs=1e-12)
    else:
        # Brackets chosen to contain both the audit sequence and this reference:
        # audit 0.242/0.102/0.052/0.016, reference 0.242/0.074/0.031/0.008.
        ceiling = {5: 0.30, 20: 0.15, 50: 0.08, 200: 0.03}[n_spectators]
        assert held < ceiling, f"{n_spectators} spectators hold {held:.4f}"


def _spectator_system(n_spectators, separation=1000.0):
    """3-atom bonded ion plus ``n_spectators`` isolated atoms, all >= 1000 A away."""
    core = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8660254037844386, 0.0]]
    positions = list(core)
    side = int(math.ceil(max(n_spectators, 1) ** (1.0 / 3.0))) + 1
    grid = [
        [separation * (1 + a), separation * b, separation * c]
        for a in range(side)
        for b in range(side)
        for c in range(side)
    ]
    positions.extend(grid[:n_spectators])
    positions = torch.tensor(positions)
    bond_index = torch.tensor([[0, 1], [1, 2]])  # only the ion is bonded
    groups = [[0, 1, 2]] + [[3 + i] for i in range(n_spectators)]
    targets = [1.0] + [0.0] * n_spectators
    return positions, bond_index, groups, targets


# ===========================================================================
# SPEC 6: QEq dissociation pathology
# ===========================================================================
def test_qeq_dissociation_limit_is_chi_difference_over_hardness_sum():
    """SPEC 6: ``q_inf = (chi_B - chi_A) / (eta_A + eta_B)`` to 8 digits.

    Derivation: with q_A = -q_B = q and J(R) -> 0,
        U(q) = 0.5 (eta_A + eta_B) q^2 + (chi_A - chi_B) q
        dU/dq = 0  =>  q = (chi_B - chi_A) / (eta_A + eta_B).

    This is a PATHOLOGY, not a virtue: a neutral diatomic that dissociates into
    fractionally charged atoms is wrong physics.  The test documents the
    behaviour of plain QEq so a change to it is visible, it does not endorse it.
    """
    chi_a, chi_b = -4.2, -7.9
    eta_a, eta_b = 12.0, 13.5
    q_infinity = (chi_b - chi_a) / (eta_a + eta_b)

    separation = 1.0e9  # A; J(R) = K_E/R = 1.44e-8 eV, so 2J/(eta_A+eta_B) ~ 1.1e-9
    positions = torch.tensor([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]])
    result = run_solve(
        positions,
        torch.tensor([[0], [1]]),
        chi=torch.tensor([chi_a, chi_b]),
        eta=torch.tensor([eta_a, eta_b]),
        sigma=torch.tensor([1.5, 1.5]),
        kappa=1.0e-16,
        groups=[[0, 1]],
        targets=[0.0],
        n_iter=8,
    )
    q_a = float(result.q[0])
    assert q_a == pytest.approx(q_infinity, rel=1.0e-8), (
        f"q_A = {q_a:.12f}, SPEC limit = {q_infinity:.12f}"
    )
    assert float(result.q[1]) == pytest.approx(-q_infinity, rel=1.0e-8)


def test_qeq_dissociation_follows_the_finite_separation_formula():
    """At finite R the exact answer is ``(chi_B - chi_A) / (eta_A + eta_B - 2 J(R))``.

    Derived the same way as above but keeping the cross term
    ``J(R) q_A q_B = -J(R) q^2``.  Checking this at a chemical separation tests
    the solver where the kernel actually matters, instead of only in the limit.
    """
    chi_a, chi_b = -4.2, -7.9
    eta_a, eta_b = 12.0, 13.5
    sigma = 1.5
    for separation in (3.0, 6.0, 20.0):
        j = float(ref.coulomb_kernel_ref(separation, ref.combine_sigma_ref(sigma, sigma)))
        expected = (chi_b - chi_a) / (eta_a + eta_b - 2.0 * j)
        result = run_solve(
            torch.tensor([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]]),
            torch.tensor([[0], [1]]),
            chi=torch.tensor([chi_a, chi_b]),
            eta=torch.tensor([eta_a, eta_b]),
            sigma=torch.tensor([sigma, sigma]),
            kappa=1.0e-16,
            groups=[[0, 1]],
            targets=[0.0],
            n_iter=8,
        )
        assert float(result.q[0]) == pytest.approx(expected, rel=1.0e-10), (
            f"R={separation}: q_A={float(result.q[0]):.12f}, expected {expected:.12f}"
        )


# ===========================================================================
# SPEC 2 / 6: gradients
# ===========================================================================
def _gradient_case(n_iter, solve_mode=None):
    positions = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [1.43, 0.11, -0.22],
            [2.31, 1.30, 0.41],
            [-0.72, 1.09, 0.63],
        ]
    )
    bond_index = torch.tensor([[0, 1, 0], [1, 2, 3]])
    kwargs = dict(
        chi=torch.tensor([-4.7, -3.1, -5.9, -2.2]),
        eta=torch.tensor([12.0, 13.5, 11.5, 14.0]),
        sigma=torch.full((4,), 1.5),
        kappa=0.4,
        groups=[[0, 1, 2, 3]],
        targets=[0.0],
        n_iter=n_iter,
        ell=0.8,
        a_mu=9.0,
    )
    if solve_mode is not None:
        kwargs["solve_mode"] = solve_mode

    def energy_of(p):
        return run_solve(p, bond_index, **kwargs).energy

    return positions, energy_of


@pytest.mark.parametrize("solve_mode", ["converged", "fixed_k"])
@pytest.mark.parametrize("n_iter", [1, 2, 6, 30])
def test_energy_gradient_matches_finite_difference(n_iter, solve_mode):
    """SPEC 6: "analytic gradient vs finite difference | < 1e-10".

    The finite difference is taken of the *computed* energy function: the
    converged minimum with implicit derivatives (SPEC 17, where ``n_iter`` is
    irrelevant) and the legacy k steps of CG differentiated through (SPEC 2).
    """
    positions, energy_of = _gradient_case(n_iter, solve_mode)
    x = positions.clone().requires_grad_(True)
    (analytic,) = torch.autograd.grad(energy_of(x), x)
    numeric = ref.fd_gradient(lambda p: float(energy_of(p).detach()), positions, step=1e-4)
    error = float((analytic - numeric).abs().max())
    assert error < 1.0e-10, f"max |autograd - FD| = {error:.3e} eV/A at k={n_iter}"


def test_k_equals_one_cannot_discriminate_when_chi_is_geometry_independent():
    """Why the test below starts at k = 2, written down so it is not "fixed".

    SPEC 2's audit reports a Hellmann-Feynman error of 8.4e-2 at k = 1.  That is
    NOT reproducible on this module's test bed, and the reason is structural
    rather than a defect:

    CG's first step is an exact line minimisation along the initial residual
    direction ``d``.  In this module ``chi``, ``eta`` and ``kappa`` are
    hand-set constants and the seed is zero, so ``d`` does not depend on the
    positions at all -- only the step *length* does.  At an exact line minimum
    ``dU/d(step) = 0``, so the ``dm/dR`` term contributes nothing and the two
    gradients coincide to round-off (measured: 1.8e-12).

    Measured evidence for the mechanism: the cosine between the moment
    directions at two different geometries is 1.000000000000 at k = 1 and
    0.997981 at k = 2.  In the real pipeline ``chi`` is a readout of ``h``, so
    ``d`` does rotate with geometry and k = 1 discriminates -- which is the
    configuration the SPEC audit measured.
    """
    positions, energy_of = _gradient_case(1, "fixed_k")
    bond_index = torch.tensor([[0, 1, 0], [1, 2, 3]])
    common = dict(
        solve_mode="fixed_k",
        chi=torch.tensor([-4.7, -3.1, -5.9, -2.2]),
        eta=torch.tensor([12.0, 13.5, 11.5, 14.0]),
        sigma=torch.full((4,), 1.5),
        kappa=0.4,
        groups=[[0, 1, 2, 3]],
        targets=[0.0],
        ell=0.8,
        a_mu=9.0,
    )

    def direction(pos, n_iter):
        result = run_solve(pos, bond_index, n_iter=n_iter, **common)
        moments = torch.cat(
            [result.t.reshape(-1), (result.mu / 0.8).reshape(-1)]
        ).detach()
        return moments / moments.norm()

    moved = positions.clone()
    moved[2, 0] += 0.35
    cosine_k1 = float((direction(positions, 1) * direction(moved, 1)).sum())
    cosine_k2 = float((direction(positions, 2) * direction(moved, 2)).sum())
    assert cosine_k1 == pytest.approx(1.0, abs=1.0e-12), (
        "the k=1 direction now depends on geometry, so k=1 would discriminate "
        "after all and the parametrisation below should include it"
    )
    assert cosine_k2 < 1.0 - 1.0e-4  # k=2 genuinely rotates

    # And so the two gradients agree at k = 1 on this test bed.
    x = positions.clone().requires_grad_(True)
    through = torch.autograd.grad(energy_of(x), x)[0].detach()
    numeric = ref.fd_gradient(
        lambda p: float(energy_of(p).detach()), positions, step=1e-4
    )
    # SPEC 6's derivative tolerance still has to hold, degeneracy or not.
    assert float((through - numeric).abs().max()) < 1.0e-10


@pytest.mark.parametrize("n_iter", [2, 3])
def test_the_solve_is_differentiated_through_not_hellmann_feynman(n_iter):
    """SPEC 2: "NOT Hellmann-Feynman.  Audit: drift 2348 -> 0.14 K/ns/atom (12000x)".

    LEGACY MODE (``solve_mode="fixed_k"``): the converged default is exactly
    stationary, so there the two gradients coincide by the envelope theorem.

    Behavioural test.  At small k the iterate is not stationary, so
    ``dU/dR`` (which includes dm_k/dR) and the Hellmann-Feynman gradient
    ``(partial U / partial R)|_m`` are different numbers.  The package gradient
    must equal the finite difference of the computed energy -- i.e. the
    differentiate-through answer -- and must NOT equal the Hellmann-Feynman one.

    The Hellmann-Feynman reference is built from the package's own assembled
    system with the moments detached, so the only difference between the two
    sides is whether dm_k/dR is carried.
    """
    assemble_system, SolverOptions = require(
        "solver", "assemble_system", "SolverOptions"
    )
    positions, energy_of = _gradient_case(n_iter, "fixed_k")

    x = positions.clone().requires_grad_(True)
    through = torch.autograd.grad(energy_of(x), x)[0].detach()
    numeric = ref.fd_gradient(lambda p: float(energy_of(p).detach()), positions, step=1e-4)

    # --- Hellmann-Feynman gradient of the same configuration ----------------
    build_incidence = require("topology", "build_incidence")
    (
        ElectrostaticParams,
        Geometry,
        FragmentConstraints,
        _GeometrySeed,
        _SolverOptions,
        _solve,
    ) = _solver()
    bond_index = torch.tensor([[0, 1, 0], [1, 2, 3]])
    incidence, bonds = build_incidence(bond_index, 4)
    params = ElectrostaticParams(
        chi=torch.tensor([-4.7, -3.1, -5.9, -2.2]),
        eta=torch.tensor([12.0, 13.5, 11.5, 14.0]),
        sigma=torch.full((4,), 1.5),
        ell=torch.full((4,), 0.8),
        kappa=torch.full((bonds.shape[1],), 0.4),
        A_mu=_isotropic(4, 9.0),
    )
    constraints = FragmentConstraints(matrix=torch.ones(1, 4), targets=torch.zeros(1))

    solved = run_solve(
        positions,
        bond_index,
        chi=params.chi,
        eta=params.eta,
        sigma=params.sigma,
        kappa=0.4,
        groups=[[0, 1, 2, 3]],
        targets=[0.0],
        n_iter=n_iter,
        ell=0.8,
        a_mu=9.0,
        solve_mode="fixed_k",
    )
    mu_hat = (solved.mu / params.ell.unsqueeze(-1)).reshape(-1)
    moments = torch.cat([solved.t.reshape(-1), mu_hat]).detach()

    y = positions.clone().requires_grad_(True)
    system = assemble_system(
        params,
        Geometry(positions=y, incidence=incidence, bond_index=bonds),
        constraints,
        SolverOptions(),
    )
    hellmann_feynman = torch.autograd.grad(system.energy(moments), y)[0].detach()

    through_error = float((through - numeric).abs().max())
    hf_error = float((hellmann_feynman - numeric).abs().max())

    # SPEC 6 tolerance for "analytic gradient vs finite difference".
    assert through_error < 1.0e-10, f"through-CG gradient off by {through_error:.3e}"
    # The test is only meaningful if the two candidate answers actually differ.
    assert hf_error > 1.0e-6, (
        f"Hellmann-Feynman and differentiate-through agree at k={n_iter}, so "
        f"this test cannot discriminate (hf_error={hf_error:.3e})"
    )
    assert hf_error > 1.0e3 * through_error


def test_converged_solve_makes_the_two_gradients_agree():
    """Control for the test above: at large k the iterate is stationary and
    Hellmann-Feynman becomes correct.  That is why the discriminating test has
    to be run at small k."""
    positions, energy_of = _gradient_case(60)
    x = positions.clone().requires_grad_(True)
    through = torch.autograd.grad(energy_of(x), x)[0].detach()
    numeric = ref.fd_gradient(lambda p: float(energy_of(p).detach()), positions, step=1e-4)
    assert float((through - numeric).abs().max()) < 1.0e-10


# ===========================================================================
# SPEC 2: the seed and the fixed iteration count
# ===========================================================================
def test_energy_is_a_deterministic_function_of_geometry():
    """SPEC 2: "Never seed from the previous MD step -- that makes U
    history-dependent and forfeits conservative forces"."""
    positions, energy_of = _gradient_case(6)
    first = float(energy_of(positions).detach())
    for _ in range(3):
        assert float(energy_of(positions).detach()) == first


def test_solver_refuses_a_previous_solve_result_as_a_seed():
    """SPEC 2: the seed must be geometry-determined, never a warm start."""
    (
        _params,
        _geometry,
        _constraints,
        _seed,
        SolverOptions,
        solve_charges,
    ) = _solver()
    positions, _ = _gradient_case(6)
    solved = run_solve(
        positions,
        torch.tensor([[0, 1, 0], [1, 2, 3]]),
        chi=torch.tensor([-4.7, -3.1, -5.9, -2.2]),
        eta=torch.tensor([12.0, 13.5, 11.5, 14.0]),
        sigma=torch.full((4,), 1.5),
        kappa=0.4,
        groups=[[0, 1, 2, 3]],
        targets=[0.0],
        n_iter=6,
    )
    ElectrostaticParams, Geometry, FragmentConstraints = require(
        "solver", "ElectrostaticParams", "Geometry", "FragmentConstraints"
    )
    build_incidence = require("topology", "build_incidence")
    incidence, bonds = build_incidence(torch.tensor([[0, 1, 0], [1, 2, 3]]), 4)
    params = ElectrostaticParams(
        chi=torch.tensor([-4.7, -3.1, -5.9, -2.2]),
        eta=torch.tensor([12.0, 13.5, 11.5, 14.0]),
        sigma=torch.full((4,), 1.5),
        ell=torch.ones(4),
        kappa=torch.full((bonds.shape[1],), 0.4),
        A_mu=_isotropic(4, 9.0),
    )
    with pytest.raises(TypeError):
        solve_charges(
            params,
            Geometry(positions=positions, incidence=incidence, bond_index=bonds),
            FragmentConstraints(matrix=torch.ones(1, 4), targets=torch.zeros(1)),
            6,
            seed=solved,
            options=SolverOptions(),
        )


def test_fragment_constraints_hold_exactly_at_every_iteration_count():
    """SPEC 2.1 item 5: feasibility must not depend on convergence."""
    positions, _, groups, targets = _spectator_system(4)
    n_atoms = positions.shape[0]
    for n_iter in (1, 2, 3, 6, 20):
        result = run_solve(
            positions,
            torch.tensor([[0, 1], [1, 2]]),
            chi=torch.tensor([-4.5, -4.5, -4.5] + [-3.0] * 4),
            eta=torch.tensor([12.0, 12.0, 12.0] + [14.0] * 4),
            sigma=torch.full((n_atoms,), 1.5),
            kappa=0.5,
            groups=groups,
            targets=targets,
            n_iter=n_iter,
        )
        assert abs(float(result.q[:3].sum()) - 1.0) < 1.0e-12
        assert abs(float(result.q[3:].abs().max())) < 1.0e-12
