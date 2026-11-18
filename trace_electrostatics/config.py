"""Configuration for the TRACE variational electrostatics module.

One frozen dataclass, :class:`ElectrostaticConfig`, carries every knob that is
not a learned parameter.  It is validated on construction, because every field
here has a failure mode that is silent rather than loud: a Thole parameter of
1.0 does not raise, it just returns a polarizability that is too large by an
amount nobody notices; a conjugate-gradient count that changes between training
and inference does not raise, it just makes the energy a different function.

Units are eV, angstrom, e (see :mod:`trace_electrostatics.constants`).

Three invariants that the type system cannot express, so they are checked here:

* **Every field is a Python scalar, never a tensor.**  The solver is compiled
  ahead of time with static shapes, so ``n_cg_iter`` and the reciprocal-space
  extent must be known at trace time.  A tensor here would either force a host
  synchronisation or silently bake in the value it happened to hold.  That
  applies to the ``bool`` switches too, and not as a formality: a
  ``check_hardness_floor=torch.tensor([1, 2])`` used to be accepted here and
  then raise "Boolean value of Tensor with more than one element is ambiguous"
  from inside ``energy.py``, and ``require_float64=0`` used to make
  :meth:`ElectrostaticConfig.validate_dtype` return silently - disabling the one
  check standing between the user and a model that is quietly wrong at 1e-8.
* **Every field is validated, on every route into the object.**  Dataclass
  ``__post_init__`` runs on construction but *not* on unpickling, and
  ``copy.deepcopy`` goes the same way: a config restored from a checkpoint used
  to arrive completely unvalidated (measured: 0 calls to ``__post_init__`` over
  a deepcopy plus a pickle round trip).  :meth:`__setstate__` closes that.
* **Electrostatics are opt-in.**  Constructing a config does not enable
  anything; nothing in TRACE reads a default config.  The only way to turn the
  model on is :func:`trace_electrostatics.energy.attach_electrostatics`, which
  demands an explicit heads module.  See ``docs/electrostatics.md``.

**What is deliberately NOT here.**  The bond-hardness floor ``kappa_0``.  It
used to be a field, and it did nothing: ``kappa`` is emitted by
:class:`trace_electrostatics.heads.ElectrostaticHeads`, which takes the floor as
a constructor argument and bakes it in, and no code outside this module ever
read ``config.kappa_0``.  ``ElectrostaticConfig(kappa_0=5.0)`` changed nothing
whatsoever while a comment here asserted it matched
``heads.DEFAULT_KAPPA_FLOOR``, with nothing enforcing that.  The floor has one
home now - ``ElectrostaticHeads(kappa_floor=...)``, defaulting to
``heads.DEFAULT_KAPPA_FLOOR`` - because it has to be set before the head that
uses it exists, which is strictly earlier than any config is consulted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .implicit import DEFAULT_SOLVE_MAX_ITER, DEFAULT_SOLVE_RTOL
from .kernels import DEFAULT_GAUSSIAN_A, DEFAULT_THOLE_A, GAUSSIAN_A_MAX, THOLE_A_MAX

__all__ = [
    "DEFAULT_CG_ITERATIONS",
    "ElectrostaticConfig",
]

#: Fixed conjugate-gradient iteration count (SPEC 2).  The solve is *defined*
#: as k steps of CG differentiated through, so k is part of the definition of
#: the energy, not a convergence tolerance.  Changing it changes the potential.
DEFAULT_CG_ITERATIONS = 6

#: Real-space cutoff for the Ewald sum, in A.
DEFAULT_EWALD_REAL_CUTOFF = 10.0

#: Reciprocal grid used ONLY when no cell is available to tune from (off the
#: periodic energy path).  On the periodic path k_max is derived from the cell
#: and the requested accuracy (SPEC 7.0), because a fixed k_max silently
#: under-converges any cell larger than a small box.
DEFAULT_EWALD_K_MAX = 8

#: Guard against materialising the dense O(N^2) pair blocks of the open-boundary
#: route for a system that should have been run periodically.  At 2048 atoms the
#: dipole-dipole block alone is 2048^2 * 9 * 8 B = 302 MB.
DEFAULT_MAX_DIRECT_ATOMS = 2048
#: Ceiling on the dense solver operator K, in bytes. See max_operator_bytes.
DEFAULT_MAX_OPERATOR_BYTES = 4.0e9

#: Gauss-Legendre points on the imaginary-frequency axis (SPEC 13).  Measured
#: against the exact eigenvalue zero-point energy on a 6-atom cluster with
#: omega_i in 8-18 eV: 8 points 8.9e-06, 12 points 1.0e-08, 16 points 6.2e-12
#: relative.
DEFAULT_DISPERSION_QUADRATURE = 16

#: Scale of the frequency map omega = s x / (1 - x), eV.  Any positive value
#: gives the same integral; this one sits inside the range of atomic
#: excitation energies so the default quadrature converges fast (SPEC 13).
DEFAULT_DISPERSION_FREQUENCY_SCALE = 10.0

_MULTIPOLE_KERNELS = ("thole", "gaussian")

#: SPEC 13.1.  Minimum plane spacing (A) of the supercell a periodic dispersion
#: energy is folded onto: a Gamma-point supercell of M1 x M2 x M3 cells is an
#: exact M1 x M2 x M3 q-grid.  Measured on a 2-atom rocksalt cell (plane spacing
#: 2.89 A) with the non-analytic Gamma term on: 2.9 A -48 %, 5.8 A +5.2 %,
#: 8.7 A +0.18 %, 11.5 A +2e-5 relative to the converged value; without the
#: Gamma term the same 11.5 A is still +12 % and 17.3 A +3.5 %.
DEFAULT_DISPERSION_SUPERCELL_LENGTH = 10.0


@dataclass(frozen=True)
class ElectrostaticConfig:
    """Non-learned settings for the charge-equilibration energy.

    Attributes
    ----------
    solve_mode:
        ``"converged"`` (default, SPEC 17): the energy is the constrained
        minimum, solved to ``solve_rtol`` by preconditioned CG and
        differentiated implicitly -- exact forces, responses and training
        gradients of every order, independent of the seed and of the gauge
        ``ell``.  ``"fixed_k"``: the legacy SPEC 2 energy ``U(m_k)``.
    solve_rtol, solve_max_iter:
        Converged mode: relative residual at which the solve stops (default
        1e-10; forces are consistent with the energy to O(rtol), measured
        1.2e-11 eV/A worst case), and the iteration ceiling, beyond which it
        RAISES rather than return a truncated iterate.
    n_cg_iter:
        ``solve_mode="fixed_k"`` only: the number of conjugate-gradient
        iterations, differentiated through, which is then part of the
        definition of the energy (SPEC 2).  Ignored by the converged solve.
    thole_a:
        Thole exponential screening parameter.  ``sup_r alpha |T| = a`` exactly,
        so the dipole relay is positive definite for every separation and every
        polarizability if and only if ``a < 1``.  0.39 is the AMOEBA value and
        leaves a 2.5x margin.
    check_hardness_floor:
        DEPRECATED, and never able to switch the check off.  The floor
        ``eta_i >= S_0(sigma_i)`` (SPEC 1.2 / 17.3) is an invariant of
        :class:`~trace_electrostatics.solver.ElectrostaticParams`, asserted
        whenever parameters are constructed; this flag only controls a
        redundant re-check in the energy path.  Setting it to ``False`` warns.
    check_constraints:
        Verify ``|sum_{i in F} q_i - Q_F| < constraint_tol`` per fragment after
        the solve (SPEC 2.1.5).  O(n) and it validates the whole topology path.
    check_positive_definite:
        Run the O(n^3) Schur-complement eigenvalue check at the solution
        (SPEC 2.1.3).  Off by default because it is cubic; turn it on in
        validation runs and in CI.  Checking the charge and dipole blocks
        separately is not a substitute: audit measured +0.58 and +16.45 for the
        two blocks with the full coupled system at -1.05.
    check_is_minimum:
        Verify the returned point is a **minimum**, not merely stationary
        (SPEC 2.1.4).  Newton-type solves converge silently onto saddles and then
        report a *negative* polarizability, because ``alpha = D^T K^-1 D``
        inherits the sign of the curvature it was evaluated at.  Also O(n^3) and
        also off by default.  Until this was added, SPEC 2.1 item 4 was the only
        runtime assertion with no config field at all: it could be reached only
        through the process-global
        :func:`trace_electrostatics.diagnostics.set_validation_mode`, so a
        per-run config could neither turn it on nor record that it had been on.
        Consumed via :meth:`solver_options`.
    check_convergence:
        Warn when the fixed-``k`` solve has not actually equilibrated the
        moments, i.e. when ``fixed_k_residual`` exceeds ``convergence_tol``.
        Off by default, and a **warning rather than an exception**, because a
        truncated solve is not an error in this package: SPEC 2 *defines* the
        energy as ``U(m_k)`` after exactly ``n_cg_iter`` steps, and the forces
        are the exact gradient of that definition.  What the warning exists for
        is the thing that definition does not make obvious -- the gap between
        ``U(m_k)`` and the equilibrated ``U(m_inf)`` grows with system size,
        because the eigenvalue spread of K -- which sets how fast CG converges
        -- widens with N. (An earlier note blamed "k steps propagate charge at
        most k hops"; that is false, K is dense through the Coulomb kernel.
        See SPEC 2.0.)

        Measured on jittered simple-cubic clusters at 2.5 A spacing, untrained
        heads, ``n_cg_iter = 6``, against a 400-step reference:

            N atoms        8        27        64       125
            |dU|/atom  7.0e-05   8.1e-04   1.6e-03   4.0e-03  eV

        A 57x growth over a 16x range in ``N``, and still rising at 125 -- so at
        a few hundred atoms the truncation alone is of order the 1 meV/atom that
        an ML potential is usually asked to hit.  The operator itself is healthy
        over this range (the smallest eigenvalue of the constrained Hessian on
        the variational sector is 8.1e-01 to 6.5e-01, never negative); this is
        truncation, not conditioning, and diagonal preconditioning does not
        touch it -- measured gain 0.94x to 1.07x, i.e. none.

        The consequence that matters is transferability: heads fitted on small
        systems learn to compensate a small truncation error, and that
        compensation does not carry to a protein or a large supercell.  Raise
        ``n_cg_iter`` with system size, or turn this on and watch the residual.
    convergence_tol:
        Relative residual above which ``check_convergence`` warns.  Compared
        against :func:`trace_electrostatics.observables.fixed_k_residual`, whose
        own docstring is explicit that the ratio is a guide and not a bound: a
        small ratio does imply a small energy gap, an intermediate one implies
        very little.  The default 1e-2 is chosen to sit below the ratio measured
        at ``k = 6`` on the eight-atom cluster in that docstring (1.4e-02), so
        the default configuration would flag that solve rather than pass it.
    constraint_tol:
        Tolerance for the per-fragment charge constraint.
    positive_definite_tol:
        Minimum acceptable eigenvalue of the constrained Schur complement.
    ewald_real_cutoff, ewald_k_max, ewald_alpha, ewald_accuracy:
        Periodic-sum settings.  ``ewald_alpha = None`` lets the Ewald module
        choose the splitting parameter from the cutoff and the accuracy.
    max_direct_atoms:
        Refuse to build the dense open-boundary pair blocks above this size,
        rather than dying in the allocator.
    solver_seed_loss_weight:
        Weight of the variational seed loss that ElectrostaticModule
        accumulates in training mode (SPEC 2.0.1; retrieve it with
        ``take_solver_seed_loss``). It trains ONLY the solver seed -- every
        physics parameter is detached inside it -- so it cannot trade against
        the data loss, and 1.0 is a safe default. 0 disables it, leaving the
        seed at its zero initialisation, which is still ~10x better than the
        old untrained seed. Costs one extra 1-step solve per structure in
        training mode, nothing in eval.
    max_operator_bytes:
        Refuse to build the dense solver operator ``K`` above this size, on BOTH
        the open and the periodic route.  ``K`` is ``[n_m, n_m]`` with
        ``n_m = n_bonds + 3 n_atoms`` (``2 n_bonds + 3 n_atoms`` with spin), and
        ``n_bonds`` is ~10-15 per atom in a bulk-like geometry at a 5 A cutoff,
        so K is the DOMINANT allocation -- measured 1.45 GB at 1000 atoms
        against 0.10 GB for the pair blocks ``max_direct_atoms`` bounds. That
        guard alone let through 2048-atom runs with a ~6 GB K, and the periodic
        route had no guard at all. Under the legacy fixed-k solve autograd held
        the unrolled CG iterations (~3x K, measured 8.2 GB at 1024 atoms); the
        converged solve keeps no iteration graph (216 atoms: 4.1 GB vs 7.0 GB).
        The 4 GB default is sized for a 16 GB machine; raise it deliberately,
        or use solver="matrix_free", which builds no K at all.
    multipole_kernel:
        ``"gaussian"`` (default since SPEC 17.5) or ``"thole"``.  ``"gaussian"``
        makes every block of K a Coulomb inner product of Gaussian multipoles,
        so K is positive definite for every geometry by construction, open and
        periodic (SPEC 12, 14); it is required by ``quadrupoles=True``.  Thole's
        ``a < 1`` bound is a pair bound only, and Thole needs the legacy hardness
        floor (``ElectrostaticHeads(hardness_margin="legacy")``, SPEC 17.3).
    gaussian_a:
        ``0 < a < 1``, the Gaussian counterpart of ``thole_a`` (SPEC 12).
    solver:
        ``"dense"`` (default), ``"matrix_free"`` or ``"auto"``: the same solve on the same
        functional with ``K v`` computed on the fly and no ``n x n`` object --
        periodic cells through a real-space pair list plus a reciprocal sum
        (SPEC 16), open boundaries through all pairs in bounded tiles (SPEC 18).
        The observables that need the dense operator (the explicit ``minimiser``
        route, ``dense_hessian``, dispersion) refuse on it; the default response
        route works and, under the converged solve, is exact.  ``"auto"``
        picks per structure (:func:`resolve_solver`): dense for slabs and with
        dispersion (matrix-free refuses both), matrix-free for 3-D periodic
        cells, and for open systems dense up to ``AUTO_DENSE_OPEN_MAX_ATOMS``
        atoms, matrix-free above.  Measured training-step cost on CPU: OMat24
        cells of 15/30/70/80 atoms 0.38/0.44/1.59/2.99 s dense vs
        0.29/0.23/0.60/1.17 s matrix-free; water clusters of 48/120/240/348
        atoms 0.24/0.89/5.62/15.3 s dense vs 0.34/0.86/1.99/2.78 s.  New
        ``train.py`` runs default to ``"auto"``.
    reciprocal, pme_order:
        ``solver='matrix_free'`` periodic cells: the reciprocal sum as the
        chunked direct k-sum (``"ewald"``, default) or smooth particle-mesh
        Ewald (``"pme"``, SPEC 19) with B-splines of order ``pme_order``, the
        mesh sized for ``ewald_accuracy``.  With PME the real-space cutoff is
        ``ewald_real_cutoff`` (floored at the Gaussian widths), not the
        direct-sum balance below.
    ewald_balance:
        ``solver='matrix_free'`` only: choose the real-space cutoff that
        minimises the cost of ``K v`` (``r ~ N^(1/6)``, ``O(N^1.5)`` per
        product; see :func:`~trace_electrostatics.lattice_operator.balanced_real_cutoff`)
        instead of ``ewald_real_cutoff``, which then acts as a lower bound.
    dispersion:
        Add the many-body dispersion energy of SPEC 13 -- the zero-point energy
        of the master equation's own induced-dipole Hessian -- to the returned
        energy.  Needs heads built with ``dispersion=True``.
    dispersion_quadrature, dispersion_frequency_scale:
        Gauss-Legendre points and map scale (eV) of the imaginary-frequency
        integral.  16 points reach ~1e-11 relative.
    dispersion_range:
        ``lambda`` in A, ``>= 0``.  0 (default) is the full zero-point energy of
        the solver's own moment block.  ``lambda > 0`` hands the short range to
        the local model by SUBTRACTION, ``E = ZPE[A + T] - ZPE[A + T_SR]`` with
        ``T_SR = T - T(widths widened by lambda/sqrt(2))``: both forms are
        positive definite Gram forms (SPEC 13), and the long-range interaction
        keeps its full intramolecular screening -- measured, the intermolecular
        C6 is unchanged to 1e-6 relative for lambda <= 2 A while a monomer's
        short-range dispersion drops from -0.55 to -0.25 eV.  Widening the
        coupling directly instead moved that C6 by 37 %.  Gaussian kernel only
        (open and periodic, SPEC 13.1).
    dispersion_supercell_length, dispersion_gamma_correction:
        Periodic dispersion (SPEC 13.1).  The cell is folded onto the smallest
        supercell whose plane spacings reach ``dispersion_supercell_length``
        (A) -- an exact q-grid -- and the Gamma point gets its non-analytic
        longitudinal term, ``<ln(1 + (4 pi K_E/Omega) q.chi(iw).q)>``.  Gamma-only
        on a primitive cell is wrong by factors, not percent: measured -3.94
        against a converged -0.494 eV/cell on 2-atom rocksalt.
    bond_screening:
        ``None`` (default: off, every earlier model bit for bit) or
        ``(C_min, C_max)``, with ``(2.0, 2.8)`` recommended.  This is the Pauli
        screening of the split-charge channel (:mod:`trace_electrostatics.screening`):
        a third atom between i and j blocks their direct overlap, and the bond's
        envelope is multiplied by the MEAM-type factor
        ``S_ij = prod_k s(C_ikj)``.  Rocksalt's second shell then carries exactly
        zero split charge, and charge flows through the shared anion's two bonds.
        Measured without it: the second-shell Mg-Mg bonds carried 1281 of a trained
        model's eps_inf = 1316 for MgO (experiment ~3).
    dispersion_quadrupoles:
        Let the quadrupoles fluctuate too (C8-type dispersion), at the single-pole
        QDO frequency ``2 omega``.  Requires the quadrupole sector.  Opt-in: the
        harmonic-quadrupole treatment is an approximation to a Drude oscillator,
        not an exact one.
    require_float64:
        Refuse to run in anything but double precision.  e3nn caches generated
        tensor-product code under the dtype in force when the module is first
        constructed, and a later ``.double()`` is a silent no-op (SPEC 3:
        2.54e-08 vs 1.94e-15).  The failure mode is a model that trains, runs,
        and is quietly wrong at the 1e-8 level.
    """

    n_cg_iter: int = DEFAULT_CG_ITERATIONS
    thole_a: float = DEFAULT_THOLE_A
    solve_mode: str = "converged"
    solve_rtol: float = DEFAULT_SOLVE_RTOL
    solve_max_iter: int = DEFAULT_SOLVE_MAX_ITER

    check_hardness_floor: bool = True
    check_constraints: bool = True
    check_positive_definite: bool = False
    check_is_minimum: bool = False
    check_convergence: bool = False

    convergence_tol: float = 1e-2
    constraint_tol: float = 1e-10
    positive_definite_tol: float = 1e-8

    ewald_real_cutoff: float = DEFAULT_EWALD_REAL_CUTOFF
    ewald_k_max: int | None = None
    ewald_alpha: float | None = None
    ewald_accuracy: float = 1e-6

    max_direct_atoms: int = DEFAULT_MAX_DIRECT_ATOMS
    max_operator_bytes: float = DEFAULT_MAX_OPERATOR_BYTES
    solver_seed_loss_weight: float = 1.0
    require_float64: bool = True

    multipole_kernel: str = "gaussian"
    gaussian_a: float = DEFAULT_GAUSSIAN_A
    solver: str = "dense"
    ewald_balance: bool = True
    reciprocal: str = "ewald"
    pme_order: int = 8
    dispersion: bool = False
    dispersion_quadrature: int = DEFAULT_DISPERSION_QUADRATURE
    dispersion_frequency_scale: float = DEFAULT_DISPERSION_FREQUENCY_SCALE
    dispersion_quadrupoles: bool = False
    dispersion_range: float = 0.0
    dispersion_supercell_length: float = DEFAULT_DISPERSION_SUPERCELL_LENGTH
    dispersion_gamma_correction: bool = True
    bond_screening: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        self._require_int("n_cg_iter", self.n_cg_iter, minimum=1)
        if self.solve_mode not in ("converged", "fixed_k"):
            raise ValueError(
                f"solve_mode must be 'converged' or 'fixed_k', got {self.solve_mode!r}."
            )
        self._require_positive("solve_rtol", self.solve_rtol)
        if not self.solve_rtol < 1.0:
            raise ValueError(f"solve_rtol must be < 1 (got {self.solve_rtol}).")
        self._require_int("solve_max_iter", self.solve_max_iter, minimum=1)
        if self.ewald_k_max is not None:
            self._require_int("ewald_k_max", self.ewald_k_max, minimum=1)
        self._require_int("max_direct_atoms", self.max_direct_atoms, minimum=1)

        if isinstance(self.thole_a, torch.Tensor) or not isinstance(self.thole_a, float):
            raise TypeError(
                f"thole_a must be a Python float, got {type(self.thole_a).__name__}."
            )
        if not 0.0 < self.thole_a < THOLE_A_MAX:
            raise ValueError(
                f"thole_a must satisfy 0 < a < {THOLE_A_MAX} (got {self.thole_a}). "
                "sup_r alpha|T| = a exactly, so at a >= 1 even a PAIR of sites loses "
                "positive definiteness at short range. a < 1 is necessary, not "
                "sufficient: three or more close sites can be indefinite below it "
                "(a 10-site head-to-tail chain at a = 0.95, SPEC 1.6). The "
                f"recommended value is {DEFAULT_THOLE_A}; the Gaussian kernel is "
                "positive definite by theorem."
            )

        self._require_positive("constraint_tol", self.constraint_tol)
        self._require_positive("positive_definite_tol", self.positive_definite_tol)
        self._require_positive("ewald_real_cutoff", self.ewald_real_cutoff)
        self._require_positive("ewald_accuracy", self.ewald_accuracy)

        if self.ewald_alpha is not None:
            self._require_positive("ewald_alpha", self.ewald_alpha)

        # The switches are fields too, and an unvalidated switch is worse than an
        # unvalidated tolerance: a falsy require_float64 silently turns off the
        # dtype guard, and a tensor check_hardness_floor raises "ambiguous truth
        # value" from inside energy.py instead of from here.
        for name in (
            "check_hardness_floor",
            "check_constraints",
            "check_positive_definite",
            "check_is_minimum",
            "require_float64",
            "dispersion",
            "dispersion_quadrupoles",
            "dispersion_gamma_correction",
            "ewald_balance",
        ):
            self._require_bool(name, getattr(self, name))

        if self.reciprocal not in ("ewald", "pme"):
            raise ValueError(f"reciprocal must be 'ewald' or 'pme', got {self.reciprocal!r}.")
        self._require_int("pme_order", self.pme_order, minimum=4)
        if self.reciprocal == "pme" and self.solver == "dense":
            raise ValueError(
                "reciprocal='pme' is the matrix-free route's reciprocal sum; set "
                "solver='matrix_free' (the dense route materialises the k-space blocks)."
            )
        if self.check_hardness_floor is False:
            import warnings

            warnings.warn(
                "ElectrostaticConfig(check_hardness_floor=False) is deprecated and "
                "does not disable the hardness-floor check: eta >= S_0(sigma) is an "
                "invariant asserted when ElectrostaticParams is built (SPEC 17.3). "
                "It only skips a redundant re-check.",
                DeprecationWarning,
                stacklevel=3,
            )
        if self.solver not in ("dense", "matrix_free", "auto"):
            raise ValueError(
                f"solver must be 'dense', 'matrix_free' or 'auto', got {self.solver!r}."
            )
        if self.multipole_kernel not in _MULTIPOLE_KERNELS:
            raise ValueError(
                f"multipole_kernel must be one of {_MULTIPOLE_KERNELS}, got "
                f"{self.multipole_kernel!r}."
            )
        self._require_positive("gaussian_a", self.gaussian_a)
        if not self.gaussian_a < GAUSSIAN_A_MAX:
            raise ValueError(
                f"gaussian_a must satisfy 0 < a < {GAUSSIAN_A_MAX} (got "
                f"{self.gaussian_a}): at a >= 1 the on-site residual of the Gram "
                "construction can vanish and K is no longer positive definite."
            )
        self._require_int("dispersion_quadrature", self.dispersion_quadrature, minimum=2)
        self._require_positive("dispersion_frequency_scale", self.dispersion_frequency_scale)
        self._require_positive("dispersion_supercell_length", self.dispersion_supercell_length)
        if isinstance(self.dispersion_range, (bool, torch.Tensor)) or not isinstance(
            self.dispersion_range, (int, float)
        ):
            raise TypeError(
                "dispersion_range must be a Python float, got "
                f"{type(self.dispersion_range).__name__}."
            )
        if self.dispersion_range < 0.0:
            raise ValueError(f"dispersion_range must be >= 0 (got {self.dispersion_range}).")
        if self.dispersion_range > 0.0 and self.multipole_kernel != "gaussian":
            raise ValueError(
                "dispersion_range > 0 subtracts a Gaussian short-range form whose "
                "positive definiteness is the Gram argument of SPEC 12/13; it "
                "needs multipole_kernel='gaussian'."
            )
        if self.bond_screening is not None:
            value = self.bond_screening
            if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)) or len(value) != 2 or any(
                isinstance(v, (bool, torch.Tensor)) or not isinstance(v, (int, float)) for v in value
            ):
                raise TypeError(
                    "bond_screening must be None or a pair of Python floats (C_min, C_max), "
                    f"got {value!r}."
                )
            c_min, c_max = float(value[0]), float(value[1])
            if not 0.0 <= c_min < c_max:
                raise ValueError(f"bond_screening needs 0 <= C_min < C_max, got ({c_min}, {c_max}).")
            object.__setattr__(self, "bond_screening", (c_min, c_max))

    def __setstate__(self, state: dict) -> None:
        """Re-validate after unpickling, which does not call ``__post_init__``.

        ``copy.deepcopy`` and ``pickle`` both rebuild a dataclass by restoring
        ``__dict__`` directly, so a config that arrives from a checkpoint has
        never been through :meth:`__post_init__` (measured: 0 calls over a
        deepcopy plus a pickle round trip).  That is precisely the route by which
        an invalid config is most likely to reach a production run, since it is
        the route that skips the human who typed the values.

        ``__dict__`` is updated in place rather than through ``setattr`` because
        the dataclass is frozen; the validation that follows is the whole point.
        """
        self.__dict__.update(state)
        self.__post_init__()

    # -- validation helpers ------------------------------------------------

    @staticmethod
    def _require_int(name: str, value: Any, minimum: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} must be a Python int (it is a static shape parameter for "
                f"ahead-of-time compilation), got {type(value).__name__}."
            )
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum} (got {value}).")

    @staticmethod
    def _require_positive(name: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            raise TypeError(
                f"{name} must be a Python float, not a tensor: configuration is "
                "resolved at trace time, and a tensor here would either force a "
                "host synchronisation or bake in whatever value it held."
            )
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
        if not value > 0.0:
            raise ValueError(
                f"{name} must be strictly positive (got {value}). Every positive "
                "quantity here is a physical scale or a tolerance that divides or "
                "bounds something: zero or a negative value does not weaken the "
                "setting, it makes the quantity meaningless. Pass a small positive "
                "number if the intent was 'effectively off'."
            )

    @staticmethod
    def _require_bool(name: str, value: Any) -> None:
        """Require a real Python ``bool``, not merely something truthy.

        ``require_float64=0`` and ``require_float64='no'`` both used to be
        accepted, and the first of them made :meth:`validate_dtype` return
        without checking anything - which is how a float32-built e3nn module gets
        into production at 1e-8 equivariance.  ``check_hardness_floor`` as a
        multi-element tensor was accepted here and blew up much later, inside
        ``energy.py``, with "Boolean value of Tensor with more than one element
        is ambiguous".
        """
        if isinstance(value, torch.Tensor):
            raise TypeError(
                f"{name} must be a Python bool, not a tensor: configuration is "
                "resolved at trace time, and a tensor here would either force a "
                "host synchronisation or bake in whatever value it held. A "
                "multi-element tensor does not even have a truth value, and the "
                "error it produces surfaces in energy.py rather than here."
            )
        if not isinstance(value, bool):
            raise TypeError(
                f"{name} must be a Python bool (True or False), got "
                f"{type(value).__name__} {value!r}. Truthiness is not accepted on "
                "purpose: these switches gate safety checks, and a value like 0 or "
                "'no' reads as a deliberate setting while silently disabling one."
            )

    # -- derived settings --------------------------------------------------

    def ewald_params(self, cell: Any = None, pbc: Any = None) -> Any:
        """Build the :class:`trace_electrostatics.ewald.EwaldParams` for this config.

        Imported lazily so that ``config`` stays importable without the periodic
        machinery, and so there is no import cycle.  This is the single place
        where the Ewald parameter object is constructed.

        ``k_max`` MUST scale with the cell, which is why ``None`` (the default)
        means "derive it from ``cell`` via :meth:`EwaldParams.auto_tune`", the
        same convention ``ewald_alpha`` already used.

        This used to be a static ``DEFAULT_EWALD_K_MAX = 8`` that no cell could
        move, and the reciprocal sum was therefore silently under-converged on
        anything bigger than a small box -- ``auto_tune`` existed, was correct,
        and was never called on the energy path.  The number of wave vectors a
        fixed accuracy needs grows linearly with the box: at ``r_cut = 6 A`` and
        ``accuracy = 1e-12``, ``auto_tune`` asks for ``k_max`` = 18, 30, 42, 59,
        83 at ``L`` = 12, 20, 28, 40, 56 A. Pinned at 8, a neutral cluster's
        periodic energy DIVERGED from its open-boundary limit as the box grew --
        the opposite of what must happen, since the images recede:

            L (A)          12        20        28        40        56
            k_max = 8   -2.4e-05  -1.1e-03  +4.9e-03  -3.8e-03  -3.7e-02   eV
            auto_tune   -1.8e-05  -3.7e-06  -1.4e-06  -4.6e-07  -1.7e-07   eV

        At L = 56 A the pinned sum was wrong by 3.7e-02 eV, about EIGHT TIMES
        the total electrostatic energy of the cluster (4.6e-03 eV), and it was
        silent. Auto-tuning converges monotonically, as an Ewald sum must.

        Passing an explicit ``ewald_k_max`` still pins it -- the regression
        suite does exactly that to hold a sum fixed while something else varies
        -- and with no ``cell`` the fallback is ``DEFAULT_EWALD_K_MAX``, which is
        only reachable off the periodic energy path.

        Args:
            cell: ``[3, 3]`` lattice, when the caller has one. Required for
                auto-tuning; without it the static fallback is used.
        """
        from .ewald import EwaldParams  # local import: single construction site

        k_max, alpha = self.ewald_k_max, self.ewald_alpha
        slab = pbc is not None and sum(bool(p) for p in pbc) == 2
        if cell is not None and (k_max is None or alpha is None):
            # A slab must be tuned PER AXIS. `auto_tune` sizes one isotropic
            # k_max from the LONGEST lattice vector, which on a padded slab is
            # the vacuum axis, so the in-plane grid became as fine as the
            # out-of-plane one: n_k grew as L_z^3 instead of L_z. Measured on an
            # 8 x 8 x L_z A cell (r_cut 3.9 A, accuracy 1e-10): 456,533 wave
            # vectors against 83,853 per-axis at L_z = 20, and 27,818,127
            # against 329,967 at L_z = 80 -- 84x the work for no accuracy,
            # because the in-plane directions were already converged at 16.
            # `auto_tune_slab` existed and was correct; it was never called.
            # The tuner is handed the pinned alpha, if any, so the grid it sizes
            # belongs to the alpha that will actually be used. Taking k_max from
            # the tuner's OWN balanced alpha and pairing it with a different
            # pinned one under-summed the reciprocal tail: measured on a 30 A
            # cell at accuracy 1e-12, k_max stayed 27 for every alpha and the
            # energy at fixed moments was off by 1.2e-6 / 2.8e-4 / 0.22 eV at
            # alpha = 0.8 / 1.0 / 1.6 A^-1, where it must be alpha-independent.
            if slab:
                tuned = EwaldParams.auto_tune_slab(
                    cell,
                    self.ewald_real_cutoff,
                    tuple(bool(p) for p in pbc),
                    accuracy=self.ewald_accuracy,
                    alpha=alpha,
                )
            else:
                tuned = EwaldParams.auto_tune(
                    cell,
                    self.ewald_real_cutoff,
                    accuracy=self.ewald_accuracy,
                    alpha=alpha,
                )
            if k_max is None:
                # Grid unpinned: the tuner's object is self-consistent -- alpha
                # (pinned or balanced), the reciprocal grid (per-axis on a slab)
                # and the real-space image shells all from ONE choice.
                return tuned
            # Grid pinned, alpha free: take the tuner's alpha with the pinned grid.
            if alpha is None:
                alpha = float(tuned.alpha)
        if k_max is None:
            k_max = DEFAULT_EWALD_K_MAX

        kwargs = {
            "r_cut": self.ewald_real_cutoff,
            "k_max": k_max,
            "alpha": alpha,
            "accuracy": self.ewald_accuracy,
        }
        try:
            return EwaldParams(**kwargs)
        except TypeError as exc:  # pragma: no cover - integration guard
            raise TypeError(
                "ElectrostaticConfig.ewald_params() could not construct EwaldParams "
                f"with {sorted(kwargs)}; adapt this one call site to the fields "
                f"EwaldParams actually declares. Original error: {exc}"
            ) from exc

    def validate_dtype(self) -> None:
        """Raise unless torch's default dtype is float64.

        **This check is a backstop, not the guard.**  It runs when the
        electrostatics are *attached*, which is strictly after
        :class:`~trace_electrostatics.heads.ElectrostaticHeads` was constructed
        and e3nn baked the dtype into its generated tensor-product code: a module
        built under float32 and then promoted with ``.double()`` passes here
        cleanly while carrying 6.16e-09 relative equivariance error against
        3.17e-16 for one built correctly.  The guard that can actually prevent
        that lives in ``ElectrostaticHeads.__init__``, before the first
        ``o3.Linear`` exists.  What this catches is the other order of events -
        heads built correctly, default dtype changed afterwards - and the case
        where the heads were built with ``require_float64=False``.
        """
        if not self.require_float64:
            return
        if torch.get_default_dtype() is not torch.float64:
            raise RuntimeError(
                "trace_electrostatics requires torch.set_default_dtype(torch.float64) "
                "to be in force BEFORE any e3nn module is constructed. e3nn caches "
                "generated TensorProduct code under the dtype at first construction, "
                "so calling .double() afterwards is a silent no-op that leaves "
                "equivariance accurate to 2.5e-08 instead of 1.9e-15. Set "
                "require_float64=False only for a deliberate single-precision "
                f"experiment. Current default dtype: {torch.get_default_dtype()}."
            )

    def solver_options(self) -> Any:
        """Build the :class:`trace_electrostatics.solver.SolverOptions` for this config.

        The single construction site, in the same spirit as :meth:`ewald_params`,
        and the only way ``check_is_minimum`` and ``check_positive_definite``
        reach the solve.  ``SolverOptions.validate`` is tri-state: ``None`` defers
        to the process-global validation mode, which is what the solver saw when
        it was handed a bare ``SolverOptions(thole_a=...)``.  Passing an explicit
        bool here is what makes the two O(n^3) checks a per-run configuration
        setting rather than a global nobody records.

        ``expect_converged`` stays ``False``: SPEC 2 defines the energy as the
        value after exactly ``n_cg_iter`` steps, so a non-zero projected gradient
        at the returned point is the intended behaviour and testing stationarity
        there would fail correct runs.
        """
        from .solver import SolverOptions  # local import: avoids an import cycle

        validate = bool(self.check_positive_definite or self.check_is_minimum)
        kwargs = {
            "thole_a": self.thole_a,
            "multipole_kernel": self.multipole_kernel,
            "gaussian_a": self.gaussian_a,
            "solve_mode": self.solve_mode,
            "solve_rtol": self.solve_rtol,
            "solve_max_iter": self.solve_max_iter,
            "validate": validate,
            "expect_converged": False,
            "eigenvalue_tol": self.positive_definite_tol,
            "fragment_charge_tol": self.constraint_tol,
        }
        try:
            return SolverOptions(**kwargs)
        except TypeError as exc:  # pragma: no cover - integration guard
            raise TypeError(
                "ElectrostaticConfig.solver_options() could not construct "
                f"SolverOptions with {sorted(kwargs)}; adapt this one call site to "
                f"the fields SolverOptions actually declares. Original error: {exc}"
            ) from exc


#: ``solver="auto"``: open systems up to this many atoms take the dense route
#: (cheaper there: 48 atoms 0.24 s dense vs 0.34 s matrix-free per training
#: step; the two tie near 120; 240 atoms 5.6 s vs 2.0 s).
AUTO_DENSE_OPEN_MAX_ATOMS = 128


def resolve_solver(config: "ElectrostaticConfig", pbc, n_atoms: int) -> str:
    """The route ``config.solver`` means for one structure: 'dense' or 'matrix_free'.

    Both compute the same functional -- to the solve tolerance in open
    boundaries, to ``ewald_accuracy`` under periodic ones -- so the choice is
    one of cost and of which features exist on each route:

    * slab (some, not all, axes periodic) -> dense: only it carries the
      Yeh-Berkowitz correction;
    * dispersion on -> dense: the zero-point energy needs the dense block;
    * 3-D periodic -> matrix-free;
    * open -> dense up to ``AUTO_DENSE_OPEN_MAX_ATOMS`` atoms, matrix-free above.
    """
    if config.solver != "auto":
        return config.solver
    flags = [bool(p) for p in pbc]
    if any(flags) and not all(flags):
        return "dense"
    if config.dispersion:
        return "dense"
    if all(flags):
        return "matrix_free"
    return "dense" if n_atoms <= AUTO_DENSE_OPEN_MAX_ATOMS else "matrix_free"
