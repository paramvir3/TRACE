"""Drivers for the EXTERNAL-reference validation suite (``test_external_*.py``).

This module is not a reference implementation -- ``elec_reference.py`` is, and it
deliberately imports nothing from the package.  This one is the opposite: it is
the thinnest possible way to drive ``trace_electrostatics`` with *chosen*
physical parameters instead of the random node features the learned heads would
produce, so that a molecular polarizability computed here can be compared
against a literature number rather than against the package's own output.

Why the parameters have to be chosen
------------------------------------
The package predicts ``chi``, ``eta``, ``sigma``, ``ell``, ``kappa`` and
``A_mu`` from learned node features.  Without a trained checkpoint there is no
end-to-end number to compare against anything.  What CAN be validated is the
physics engine underneath: hand it published atomic polarizabilities, or the
hardness of a model conductor, and every closed-form and measured result that
electrostatics supplies becomes a test.  That is what these drivers are for.

The assembly is constructed directly, which is the route
``observables.polarizability`` documents ("a ``SolveResult`` does not carry
one ... which the observables and the tests do").  The three open-boundary pair
blocks come from the package's own ``energy._direct_pair_blocks``, so the
physics tests run on the production operator and not on a re-derivation of it;
``test_external_thole_polarizability.py`` separately checks those blocks against
a reconstruction from the public ``kernels`` API, so the private entry point is
pinned rather than trusted.

Conventions used throughout, all from SPEC 0:

* ``alpha_volume`` is in A^3 and ``A_mu = K_E * ell^2 / alpha_volume`` is its
  isotropic inverse, because ``U = 0.5 mu_hat^T A_mu mu_hat`` with
  ``mu = ell mu_hat`` gives ``alpha_response = ell^2 A_mu^-1`` in e^2 A^2 / eV
  and ``alpha_volume = K_E alpha_response``;
* every polarizability this module returns is asked for ``as_volume=True``, so
  it is directly comparable with a literature value in A^3.

Turning the Thole damping OFF, and why that is legitimate
---------------------------------------------------------
Several external references -- the Applequist two-atom closed forms, the
Clausius-Mossotti relation, the Lorentz local field -- are statements about
*bare point dipoles*.  The package always damps, so a direct comparison would
be testing a different operator.  The way out is not to bypass the package: it
is that ``energy._direct_pair_blocks`` and ``energy._periodic_pair_blocks`` take
the polarizability that sets the Thole width as a **separate argument** from
``params.A_mu``, and ``kernels.thole_damping`` documents its own limit --
``v = a r^3 / sqrt(alpha_i alpha_j) -> inf`` as that argument goes to zero, at
which point ``lambda_3 = lambda_5 = 1.0`` exactly (saturated at
``THOLE_V_SATURATION``, bit-identical).

So ``damping_alpha_volume=1e-12`` gives the package's own tensor code with the
damping factors at exactly 1.0 -- the undamped point-dipole propagator -- while
``A_mu`` keeps the physical polarizability.  Note this deliberately decouples
the assembly from the ``SolveResult`` returned alongside it, whose operator the
solver rebuilds from ``params.A_mu`` and therefore still damps.  That costs
nothing here because the closed forms are statements about the exact minimiser,
which ``observables.polarizability(route="minimiser")`` takes from the
assembly's stored blocks alone.  Do not use the fixed-k route on a
damping-decoupled assembly.
"""

from __future__ import annotations

import math

import torch

torch.set_default_dtype(torch.float64)  # SPEC 3; conftest has already done this

from conftest import require  # noqa: E402


K_E = 14.399645  # SPEC 0, quoted not imported


def _lookup():
    ElectrostaticConfig = require("config", "ElectrostaticConfig")
    ElectrostaticAssembly, direct_pair_blocks = require(
        "energy", "ElectrostaticAssembly", "_direct_pair_blocks"
    )
    (
        ElectrostaticParams,
        Geometry,
        FragmentConstraints,
        GeometrySeed,
        SolverOptions,
        solve_charges,
        isotropic_polarizability_volume,
    ) = require(
        "solver",
        "ElectrostaticParams",
        "Geometry",
        "FragmentConstraints",
        "GeometrySeed",
        "SolverOptions",
        "solve_charges",
        "isotropic_polarizability_volume",
    )
    build_incidence = require("topology", "build_incidence")
    return (
        ElectrostaticConfig,
        ElectrostaticAssembly,
        direct_pair_blocks,
        ElectrostaticParams,
        Geometry,
        FragmentConstraints,
        GeometrySeed,
        SolverOptions,
        solve_charges,
        isotropic_polarizability_volume,
        build_incidence,
    )


def isotropic_A_mu(alpha_volume, ell):
    """``A_mu`` whose isolated-atom polarizability volume is ``alpha_volume``.

    ``alpha_response = ell^2 A_mu^-1`` (solver.isotropic_polarizability_volume)
    and ``alpha_volume = K_E alpha_response`` (SPEC 0), so for an isotropic
    stiffness ``A_mu = K_E ell^2 / alpha_volume``.
    """
    alpha_volume = torch.as_tensor(alpha_volume, dtype=torch.float64).reshape(-1)
    n_atoms = alpha_volume.numel()
    scale = (K_E * float(ell) ** 2 / alpha_volume).view(n_atoms, 1, 1)
    return torch.eye(3, dtype=torch.float64).repeat(n_atoms, 1, 1) * scale


def open_system(
    positions,
    alpha_volume,
    *,
    bond_index=None,
    chi=None,
    eta=None,
    sigma=1.5,
    ell=1.0,
    kappa=1.0,
    thole_a=0.39,
    total_charge=0.0,
    fragment_id=None,
    n_iter=None,
    damping_alpha_volume=None,
):
    """An open-boundary assembly plus one solve of it, from explicit parameters.

    ``alpha_volume`` is the per-atom isolated polarizability volume in A^3 (a
    scalar broadcasts).  ``eta`` defaults to 1e8 times the SPEC 1.2 hardness
    floor, which makes the charge sector rigid; with ``bond_index=None`` there
    are no charge variables at all and the result is the pure Thole dipole
    relay.  The ``SolveResult`` is returned only because
    :func:`~trace_electrostatics.observables.polarizability` requires one; the
    minimiser route does not otherwise use it.

    ``damping_alpha_volume`` overrides the polarizability the *Thole width* is
    built from, leaving ``A_mu`` alone; ``1e-12`` switches the damping off
    exactly (module docstring).  The solver's own operator is unaffected, so
    only the minimiser route is meaningful when it is used.

    Returns ``(assembly, solve_result)``.
    """
    (
        ElectrostaticConfig,
        ElectrostaticAssembly,
        direct_pair_blocks,
        ElectrostaticParams,
        Geometry,
        FragmentConstraints,
        GeometrySeed,
        SolverOptions,
        solve_charges,
        isotropic_polarizability_volume,
        build_incidence,
    ) = _lookup()

    positions = torch.as_tensor(positions, dtype=torch.float64)
    n_atoms = int(positions.shape[0])

    alpha_volume = torch.as_tensor(alpha_volume, dtype=torch.float64).reshape(-1)
    if alpha_volume.numel() == 1:
        alpha_volume = alpha_volume.expand(n_atoms).clone()
    A_mu = isotropic_A_mu(alpha_volume, ell)

    sigma = torch.as_tensor(sigma, dtype=torch.float64).reshape(-1)
    if sigma.numel() == 1:
        sigma = sigma.expand(n_atoms).clone()
    floor = 2.0 * K_E / (sigma * math.sqrt(math.pi))  # SPEC 1.2
    if eta is None:
        eta = floor * 1.0e8
    eta = torch.as_tensor(eta, dtype=torch.float64).reshape(-1)
    if eta.numel() == 1:
        eta = eta.expand(n_atoms).clone()

    if bond_index is None:
        incidence = torch.zeros(n_atoms, 0, dtype=torch.float64)
        bonds = torch.zeros(2, 0, dtype=torch.long)
    else:
        incidence, bonds = build_incidence(
            torch.as_tensor(bond_index, dtype=torch.long), n_atoms
        )
    n_bonds = int(bonds.shape[1])

    params = ElectrostaticParams(
        chi=torch.zeros(n_atoms) if chi is None
        else torch.as_tensor(chi, dtype=torch.float64).reshape(-1),
        eta=eta,
        sigma=sigma,
        ell=torch.full((n_atoms,), float(ell)),
        kappa=torch.full((n_bonds,), float(kappa)),
        A_mu=A_mu,
    )
    # The assembly stores the polarizability volume the damping was built from,
    # and the package clamps it; take it from the package so the two agree.
    stored_alpha = isotropic_polarizability_volume(A_mu, params.ell)
    if damping_alpha_volume is not None:
        stored_alpha = torch.as_tensor(
            damping_alpha_volume, dtype=torch.float64
        ).reshape(-1).expand(n_atoms).clone()

    config = ElectrostaticConfig(thole_a=thole_a)
    j_pair, t01_pair, t2_pair = direct_pair_blocks(
        positions, params, stored_alpha, thole_a
    )

    if fragment_id is None:
        fragment_id = torch.zeros(n_atoms, dtype=torch.long)
    fragment_id = torch.as_tensor(fragment_id, dtype=torch.long)
    n_fragments = int(fragment_id.max()) + 1
    constraint = torch.zeros(n_fragments, n_atoms, dtype=torch.float64)
    constraint[fragment_id, torch.arange(n_atoms)] = 1.0
    fragment_charges = torch.full((n_fragments,), total_charge / n_fragments)

    assembly = ElectrostaticAssembly(
        pos=positions,
        cell=None,
        params=params,
        incidence=incidence,
        bond_index=bonds,
        constraint=constraint,
        fragment_charges=fragment_charges,
        fragment_id=fragment_id,
        alpha_volume=stored_alpha,
        external_field=None,
        j_pair=j_pair,
        t01_pair=t01_pair,
        t2_pair=t2_pair,
        t_seed=torch.zeros(n_bonds),
        config=config,
    )

    geometry = Geometry(
        positions=positions, incidence=incidence, bond_index=bonds
    )
    constraints = FragmentConstraints(matrix=constraint, targets=fragment_charges)
    seed = GeometrySeed.from_bond_logits(
        assembly.t_seed, n_atoms, geometry=geometry
    )
    solve_result = solve_charges(
        params,
        geometry,
        constraints,
        config.n_cg_iter if n_iter is None else int(n_iter),
        seed=seed,
        options=SolverOptions(thole_a=thole_a),
    )
    return assembly, solve_result


def cubic_dipole_assembly(
    lattice_constant,
    alpha_volume,
    *,
    n_repeat=1,
    ell=1.0,
    sigma=1.5,
    thole_a=0.39,
    cell_lengths=None,
    ewald_k_max=20,
    ewald_accuracy=1.0e-10,
    damping_alpha_volume=None,
):
    """A 3-D periodic, dipole-only assembly: ``n_repeat^3`` sites on a lattice.

    No bonds, so the charge sector has no variables and the only response is the
    Thole-damped induced dipole relay under the tinfoil Ewald sum.  That is the
    system Clausius-Mossotti is a statement about.

    ``cell_lengths`` overrides the cubic cell with an orthorhombic one (used for
    the trace sum rule, which must hold for any cell shape); ``lattice_constant``
    is then only the site spacing of a single-site cell.
    ``damping_alpha_volume`` is as in :func:`open_system`: ``1e-12`` gives the
    bare, undamped lattice sum of point dipoles, which is what
    Clausius-Mossotti and the Lorentz local field are statements about.

    Returns ``(assembly, periodic_pair_blocks)``.
    """
    (
        ElectrostaticConfig,
        ElectrostaticAssembly,
        _direct,
        ElectrostaticParams,
        _Geometry,
        _FragmentConstraints,
        _GeometrySeed,
        _SolverOptions,
        _solve,
        isotropic_polarizability_volume,
        _build_incidence,
    ) = _lookup()
    periodic_pair_blocks = require("energy", "_periodic_pair_blocks")

    index = torch.arange(n_repeat, dtype=torch.float64)
    positions = (
        torch.stack(torch.meshgrid(index, index, index, indexing="ij"), dim=-1)
        .reshape(-1, 3)
        * float(lattice_constant)
    )
    if cell_lengths is None:
        cell = torch.eye(3, dtype=torch.float64) * (n_repeat * float(lattice_constant))
    else:
        cell = torch.diag(torch.as_tensor(cell_lengths, dtype=torch.float64))
    n_atoms = int(positions.shape[0])

    sigma_t = torch.full((n_atoms,), float(sigma))
    params = ElectrostaticParams(
        chi=torch.zeros(n_atoms),
        # The floor is a strict lower bound in exact arithmetic; nudge it so
        # that float round-off cannot put eta an ulp below it.
        eta=2.0 * K_E / (sigma_t * math.sqrt(math.pi)) * (1.0 + 1.0e-9),
        sigma=sigma_t,
        ell=torch.full((n_atoms,), float(ell)),
        kappa=torch.zeros(0),
        A_mu=isotropic_A_mu(torch.full((n_atoms,), float(alpha_volume)), ell),
    )
    stored_alpha = isotropic_polarizability_volume(params.A_mu, params.ell)
    if damping_alpha_volume is not None:
        stored_alpha = torch.full((n_atoms,), float(damping_alpha_volume))
    config = ElectrostaticConfig(
        thole_a=thole_a,
        ewald_real_cutoff=4.0 * float(cell.norm(dim=-1).max()),
        ewald_k_max=ewald_k_max,
        ewald_accuracy=ewald_accuracy,
    )
    blocks = periodic_pair_blocks(positions, cell, params, stored_alpha, config)
    j_pair, t01_pair, t2_pair = blocks

    assembly = ElectrostaticAssembly(
        pos=positions,
        cell=cell,
        params=params,
        incidence=torch.zeros(n_atoms, 0, dtype=torch.float64),
        bond_index=torch.zeros(2, 0, dtype=torch.long),
        constraint=torch.ones(1, n_atoms, dtype=torch.float64),
        fragment_charges=torch.zeros(1),
        fragment_id=torch.zeros(n_atoms, dtype=torch.long),
        alpha_volume=stored_alpha,
        external_field=None,
        j_pair=j_pair,
        t01_pair=t01_pair,
        t2_pair=t2_pair,
        t_seed=torch.zeros(0),
        config=config,
        pbc=(True, True, True),
    )
    return assembly, blocks


def uniform_dipole_susceptibility(assembly, ell=1.0, axis=2):
    """``chi = P / E`` of a periodic, dipole-only assembly, in package units.

    Every site of the lattices used here is equivalent by translation, so the
    minimiser of ``U(mu) - E . sum_i mu_i`` is the uniform configuration and the
    whole problem is one scalar.  ``U`` is taken from
    :meth:`ElectrostaticAssembly.energy` -- the energy the model actually
    reports, Ewald sum, Thole damping, on-site stiffness and all -- rather than
    from a hand-assembled operator, and it is exactly quadratic in the moment,
    so two evaluations determine it and the third is a check.

    Returns ``(chi, curvature)``; ``eps = 1 + 4 pi K_E chi`` (the ``K_E`` is the
    Gaussian-to-eV/A unit conversion of SPEC 0).
    """
    n_atoms = assembly.n_atoms
    volume = float(torch.det(assembly.cell).abs())

    def energy(amplitude):
        mu_hat = torch.zeros(n_atoms, 3, dtype=torch.float64)
        mu_hat[:, axis] = amplitude
        return float(
            assembly.energy(
                torch.zeros(assembly.n_bonds),
                torch.zeros(n_atoms),
                mu_hat,
            )
        )

    zero = energy(0.0)
    assert abs(zero) < 1.0e-12, f"U(0) should vanish, got {zero}"
    curvature = 2.0 * energy(1.0)
    quartic_check = energy(2.0)
    assert abs(quartic_check - 2.0 * curvature) < 1.0e-9 * abs(curvature), (
        "the moment energy is not exactly quadratic: "
        f"U(1) = {0.5 * curvature}, U(2) = {quartic_check}"
    )

    # min over m of 0.5 c m^2 - E n ell m  ->  m = E n ell / c,  mu = ell m
    field = 1.0
    amplitude = field * n_atoms * float(ell) / curvature
    polarization = n_atoms * float(ell) * amplitude / volume
    return polarization / field, curvature


def uniform_dipole_moment(assembly, field, ell=1.0, axis=2):
    """Per-site induced dipole ``mu`` of a periodic lattice in a uniform field.

    The same stationarity :func:`uniform_dipole_susceptibility` solves, reported
    as the *moment* rather than as a susceptibility, because the Lorentz
    local-field statement is about the moment: ``mu = alpha_at E_loc / K_E``
    with ``E_loc = E + 4 pi K_E P / 3``.

    Minimising ``0.5 c m^2 - E N ell m`` over the uniform amplitude ``m`` gives
    ``m = E N ell / c`` and ``mu_per_site = ell m / N``... which for the
    single-site cells used here is simply ``ell^2 E / c``.  Written out for a
    general ``N`` so that a supercell gives the same per-site answer.

    Returns ``(mu_per_site, polarization)`` in e*A and e/A^2.
    """
    _chi, curvature = uniform_dipole_susceptibility(assembly, ell=ell, axis=axis)
    n_atoms = assembly.n_atoms
    volume = float(torch.det(assembly.cell).abs())
    amplitude = float(field) * n_atoms * float(ell) / curvature
    mu_per_site = float(ell) * amplitude
    return mu_per_site, n_atoms * mu_per_site / volume


def fibonacci_sphere(n_points, radius):
    """``n_points`` near-uniformly spaced points on a sphere of the given radius.

    The Fibonacci (golden-angle) spiral, which is the standard way to get a
    point set whose nearest-neighbour spacing is uniform to a few percent at
    every ``N`` -- unlike a latitude/longitude grid, which crowds at the poles
    and would make the "discretised conductor" of
    ``test_external_thole_polarizability.py`` converge to the wrong shape.

    The construction is a closed form in ``i`` and contains no lattice constant,
    so the whole point set scales exactly with ``radius``; that is what makes
    the ``alpha = a^3`` test scale invariant rather than merely repeated.
    """
    index = torch.arange(int(n_points), dtype=torch.float64) + 0.5
    polar = torch.acos(1.0 - 2.0 * index / float(n_points))
    azimuth = math.pi * (1.0 + math.sqrt(5.0)) * index
    return float(radius) * torch.stack(
        (
            torch.cos(azimuth) * torch.sin(polar),
            torch.sin(azimuth) * torch.sin(polar),
            torch.cos(polar),
        ),
        dim=-1,
    )
