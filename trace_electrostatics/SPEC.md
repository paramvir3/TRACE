# trace_electrostatics — interface contract

Variational charge-equilibration electrostatics for TRACE. Every number in this
document was established by numerical audit; the regression suite must keep them.

## 0. Units — single source of truth, `constants.py`

Energy **eV**, length **angstrom**, charge **e**.

    K_E = 14.399645            # eV * A / e^2

TRAP: `R**2 / (eta_A + eta_B)` is in `e^2 A^2 / eV`, **not** `A^3`.
Multiply by `K_E` to get a polarizability volume. Never report the bare
expression as a volume.

## 1. Physics

### 1.1 Damped Coulomb kernel (`kernels.py`)

    J(r_ij) = K_E * erf(r_ij / sigma_ij) / r_ij ,   sigma_ij = sqrt(sigma_i^2 + sigma_j^2)
    J(0)    = 2 K_E / (sigma sqrt(pi))

Gaussian/erf ONLY. The Ohno/Klopman form `1/sqrt(r^2+s^2)` produced an
indefinite charge block in audit and is forbidden.

### 1.2 HARDNESS FLOOR — corrected (SPEC 17.3)

    eta_i >= S_0(sigma_i) = sqrt(2) K_E / (sigma_i sqrt(pi))   # 7.659503 eV at sigma = 1.5 A

`J` with its true diagonal is the Coulomb Gram matrix of the Gaussian charge
clouds, so `diag(eta) + J_off = Gram + diag(eta - S_0)`: positive semi-definite
for EVERY geometry iff `eta_i >= S_0` (necessary: two coincident sites). The eta
head emits `softplus(raw) + S_0(sigma) + margin` (`kernels.hardness_floor`,
margin 1 eV/e^2), so the charge block's smallest eigenvalue is at least the
margin for any geometry. Not a raw linear head.

RETRACTED: this section used to state the floor as `2 K_E/(sigma sqrt(pi)) =
sqrt(2) S_0` (10.832 eV at 1.5 A) and to call everything below it "unbounded
below". That was a sufficient condition presented as a necessary one: eta = 8
eV at sigma = 1.5 A is positive definite over any cluster (measured min eig
+0.34 over 200 dense 30-atom clusters). The audit table that stood here
(carbon "unbounded" at 9.997 eV, pairs below 1.533 A at eta = 8, metals)
did not state the sigma it used and cannot be reproduced; it is withdrawn.
The multiplicative sqrt(2) also forced `eta sigma >= 16.25 eV A`, so hydrogen
at its Mulliken hardness (~12.8 eV) needed sigma >= 1.27 A.

### 1.3 Charge parameterisation (`topology.py`)

    q_i^sigma = sum_{j in N(i)} t_ij^sigma + rho_i^sigma ,   t_ij^sigma = -t_ji^sigma

`t` lives on bonds (incidence matrix `B`, shape `[n_atoms, n_bonds]`, `q = B t`).
Antisymmetry is structural, so `1^T B == 0` exactly.

`rho` carries net charge and is a **FIXED CONSTRAINT CARRIER — NOT variational.**
Only `m = (t, mu_hat)` is variational, subject to **per-fragment** constraints:

    sum_{i in F} q_i^sigma = Q_F^sigma     for each fragment F

A single GLOBAL constraint is size-inconsistent: 3-atom ion at +1 with distant
Ar spectators gave q -> 1.000, 0.242, 0.102, 0.052, 0.016 for 0/5/20/50/200
spectators at 1000 A. Per-fragment gives **1.000000 at every spectator count**.

CORRECTION (verified numerically): an earlier draft of this spec said `rho` was
variational. That is WRONG and destroys the model. With both `t` and `rho` free
under a per-fragment constraint the minimiser gives `|t| = 1.65e-07` versus
`3.57e-01` with `rho` fixed — `rho` absorbs everything, pocketing the whole
`0.5 kappa |t|^2` penalty at nearly identical charges.

### GENERAL PRINCIPLE — three instances of one bug

**Any per-atom charge channel that bypasses the bond-hardness penalty will absorb
all the charge and drive `t -> 0`.** Every such channel must be either fixed, or
penalised at least as hard as `t`. Instances found so far:

1. `q0_i(h_i)` as a geometry-dependent readout — 0.396 eV cheaper via `q0`.
2. `rho_i` as a free variational per-atom charge — `|t|` collapses to 1.65e-07.
3. (watch for) any future "correction" or "residual" charge head.

There is NO `q0_i(h_i)` geometry-dependent reference charge. It is degenerate
with `t` (identical charges, 0.396 eV cheaper via q0) and training drives
`t -> 0`, destroying the locality mechanism. A per-element constant is allowed;
a geometry-dependent readout is not.

### 1.4 Bond hardness

    kappa_ij = softplus( 0.5 * (s_ij + s_ji) ) + kappa_0 ,   kappa_0 > 0

MUST be symmetric: attention logits satisfy `Q_i . K_j != Q_j . K_i`, and an
asymmetric kappa makes the energy ill-defined because `t_ij = -t_ji`.
Small kappa destroys positive-definiteness of the SQE Hessian.

`kappa -> 0` must reproduce global QEq to **1e-15** (audit: 2.1e-15). This is a
required regression test — it is what keeps metals reachable.

### 1.4.1 DISSOCIATION ENVELOPE — the bond variable is t = f(r) tau

(The envelope FORM below -- the full-range cosine -- is superseded by the C2
shell switch of SPEC 17.4, onset 0.75 r_c; the t = f tau construction and its
continuity argument are unchanged.)

Split charges exist only while a bond is in the neighbour list. With `t` itself
as the variable they vanished ABRUPTLY at the cutoff: a 2-atom neutral fragment
measured E -1.68e-3 eV -> 0 and F 1.0e-4 eV/A -> 0 across r_c +- 1e-6 A,
identical at k = 6 and k = 400 — an energy/force discontinuity at every
neighbour-list crossing in MD. The learned kappa gave no protection: it was
nearly flat in r (0.81 / 0.76 / 0.77 eV/e^2 at 1.5 / 3.0 / 4.5 A), which also let
long "bonds" act as parallel charge channels and stretched the SQE screening
length to N* = 2 eta / kappa_eff ~ 190 sites on a chain.

Now `t_b = f(r_b) tau_b` with `f(r) = (1 + cos(pi r / r_c)) / 2`, 0 beyond,
`r_c = heads.r_max`; the hardness stays `1/2 kappa tau^2`; the seed is `f * logit`.
Implemented as `q = B diag(f) tau` (incidence columns scaled), so K, the linear
term, `C B` and the charges all see it with no other change.

- Physically `1/2 kappa tau^2 = 1/2 (kappa/f^2) t^2`: the effective hardness
  diverges on stretching — the SQE dissociation limit (Mathieu, JCP 127, 224103
  (2007)) — while the tau-space diagonal stays kappa, so CG conditioning is
  unaffected. Putting kappa -> infinity into K directly would not be.
- Continuity is EXACT at any fixed k: at f = 0 every coupling of tau_b is
  proportional to f, so its residual, direction and share of every CG inner
  product vanish, and the other iterates equal those without the bond.
  Measured: step 3e-28 eV at r_c +- 1e-6 A (was 1.68e-3); E ~ (r_c - r)^4, a
  factor 1e4 per decade.
- It removes the parallel long-bond channels: chain N* 190 -> 27 sites, local
  alpha exponent at N = 256 1.52 -> 1.10, in quantitative agreement with the
  series-resistor model using sum_n n f_n^2 / kappa_n.
- The kappa -> 0 QEq limit is unchanged (tau unconstrained, envelope drops out).
- CONTRACT: the neighbour list must contain every pair out to heads.r_max;
  `attach_electrostatics` refuses a model whose r_max is shorter.
- `SolveResult.t` is tau; the physical split charge is f * tau. Charges q are
  physical as before.

### 1.5 Energy

    U_elec = 0.5 * m^T K m + chi^T m

    m = ( {t_ij^sigma}, {mu_hat_i} )
    mu_hat_i = mu_i / (e * ell_i)       # reduced: M mixes e, e*A, e*A^2 and
                                        # cross-rank forms are otherwise
                                        # dimensionally meaningless

    K = [ B^T J B + diag(kappa)   B^T T01        ]
        [        .                A_mu + T_Thole ]

Quadratic ONLY. No quartic, no cubic, no anharmonic-in-moments term. The quartic
failed audit on eleven counts, including a spurious hyperpolarizability 300x a
real chemical beta in a nominally centrosymmetric system.

### 1.5.1 The full master equation (SPEC 10, 12, 13)

    E_LR = min_m U(m)  +  E_disp

    U(m) = 0.5 m^T K m + c^T m,     m = ( t, mu_hat, Theta_hat ),  q = B diag(f) t + rho
    Theta_hat_i = Theta_i / (e ell_i^2)   in the orthonormal 2e basis W (traceless
                                          second moment; Buckingham = 3/2 of it)

        [ B^T J B + diag(kappa)   B^T G01          B^T G02        ]
    K = [        .              A_mu + G11         G12            ]
        [        .                   .          A_Theta + G22     ]

    c = ( B^T (J rho + chi),  G01^T rho,  G02^T rho + chi_Theta )

    G_ab(i,j) = the (a,b)-rank derivative of K_E erf(r_ij / s_ab,ij) / r_ij,
    s_ab,ij = sqrt(sigma_a,i^2 + sigma_b,j^2)          (Gaussian kernel, SPEC 12)

    E_disp = (1/2pi) int_0^inf dw [ ln det(K_yy + w^2 M) - ln det(D_yy + w^2 M) ]
             y = mu_hat (and optionally Theta_hat),  M_i = ell_i^2 / (N_i hbar^2/m_e)   (SPEC 13)

ONE operator. Its classical minimum is electrostatics + induction; the
zero-point energy of the same moment block is dispersion. chi_Theta is the
permanent quadrupole source (sigma-holes, lone pairs); without it every
quadrupole is induced. The spin sector (SPEC 9) is untouched and orthogonal:
the moments see total charge only.

### 1.6 Thole damping

Exponential form, with `a < 1` **strictly**. `sup_r alpha |T| = a` exactly, so
a PAIR of sites has a positive definite relay for all r and all alpha iff
`a < 1`. CORRECTED: that is a pair statement only. Three or more close sites
can be indefinite below it -- measured with the package's own blocks: a
10-site head-to-tail chain at a = 0.95, spacing 0.63 alpha^(1/3), min
eigenvalue -0.48 A; at the recommended a = 0.39 a 40-site chain at spacing
0.089 alpha^(1/3), -0.079 A. Thole is therefore opt-in; the default kernel is
the Gaussian one, positive definite by theorem (SPEC 12, 14), and the
converged solve raises `IndefiniteOperatorError` on the first direction of
non-positive curvature (SPEC 17.5).

## 2. The solve (`solver.py`) — LEGACY, superseded by SPEC 17

The default energy is now the converged constrained minimum with implicit
derivatives (SPEC 17). Everything in this section describes
`solve_mode="fixed_k"`, which is kept, bit for bit, as an opt-in.

    m_k = k fixed CG iterations on  K m = -chi,  DIFFERENTIATED THROUGH

- NOT `argmin`. A global minimiser runs away on any stabilised nonlinearity.
- NOT Hellmann-Feynman. Audit: drift 2348 -> 0.14 K/ns/atom (12000x) when
  differentiating through the iterations instead.
- Seed MUST be geometry-determined (the antisymmetrised attention logit).
  **Never seed from the previous MD step** — that makes U history-dependent and
  forfeits conservative forces, which is the entire point of fixed-k.
- `k` is a fixed integer for AOT static shapes. Default 6.

### 2.0 THE FIXED-k TRUNCATION ERROR GROWS WITH SYSTEM SIZE

`U(m_k)` is not `U(m_inf)`, and **they do not differ by a constant** — the gap
grows with N.

CORRECTION. An earlier version of this section explained the growth as "`k` CG
steps propagate charge at most `k` hops along the bond graph". That is FALSE and
is withdrawn. K contains `B^T J B` with a dense Coulomb J, and CG's step length is
a global scalar, so a single step already moves charge everywhere: perturbing
one end of a 24-site chain changes the charge 7 bond-lengths away by 3e-5 after
ONE step. The growth is spectral — CG's convergence is set by the eigenvalue
spread of K, which widens with N. Where the k = 6 energy error lives, measured by
exact mode decomposition (the per-mode terms 1/2 lambda c^2 sum to the measured
dU): only 0.5-1.7% in the lowest 1% of modes and ~22% in the lowest quarter, so
it is NOT a long-wavelength remnant; and 62-80% in the bond sector, 20-38% in the
dipole sector. The converged solution is nearsighted — perturbing one atom of a
216-atom cluster moves charges by 4e-3 of the peak response at 7-9 A, 7e-4 at
9-11 A, 1.5e-4 at 11-13 A — which is what makes a learned seed viable (2.0.1). Measured on jittered
simple-cubic clusters at 2.5 A spacing, untrained heads, `k = 6`, against a
400-step reference:

    N atoms        8        27        64       125
    |dU|/atom  7.0e-05   8.1e-04   1.6e-03   4.0e-03  eV

A 57x growth over 16x in N, still rising at 125 — so by a few hundred atoms the
truncation alone is of order the 1 meV/atom an ML potential is asked to hit.

This is **truncation, not conditioning**, and the two have different fixes. Over
the same range the constrained Hessian on the variational `(t, mu_hat)` sector
stays healthy: `lam_min` 8.1e-01 -> 6.5e-01, never negative, `cond` 1.1e+02 ->
4.0e+03. Diagonal preconditioning is measured at 0.94x to 1.07x — i.e. it does
nothing, and a preconditioner is not the answer. More iterations is.

NOTE the analysis trap: `dense_hessian()` is in the STORAGE basis `(t, rho,
mu_hat)`, and `rho` is a fixed constraint carrier, not variational. Spectra taken
without dropping those `n_atoms` rows and columns are meaningless — doing so
manufactures a spurious negative eigenvalue at large N.

Measured SCALE-FREE, which is the honest way to state it: with untrained heads
the total electrostatic energy is only ~1e-03 eV/atom, so ANY eV/atom threshold
flatters the result. Relative truncation error `|U(k) - U*| / |U*|`:

    N          k=1       k=6      k=12      k=25      k=50     k=100
    8     2.61e+00  1.07e-01  2.59e-03  2.18e-07  0.00e+00  0.00e+00
    27    9.88e+01  7.55e-01  2.78e-01  2.93e-03  1.24e-06  4.46e-14
    64    3.75e+02  1.69e+00  1.00e+00  1.36e-01  7.76e-04  6.28e-08
    125   7.24e+02  2.95e+00  1.60e+00  5.36e-01  6.38e-03  1.04e-05

At the default `k = 6` and N = 125 the reported electrostatic energy differs from
the equilibrated one by **295% of the energy itself**. Reaching 0.1% needs k ~ 12
at N = 8 and k ~ 100 at N = 125.

### 2.0.1 THE SOLVER SEED — Ohm's law, a physics dipole step, trained behind a firewall

**First, the envelope (1.4.1) removed most of the truncation error on its own.**
Same geometries, package seed, k = 6, relative |U(m_6) - U*| / |U*|:

    features            N    envelope off   envelope on
    random (the 295%)   125     2.95e+00       1.22e-02   (240x)
    TRACE trunk         125     3.05e-01       5.56e-03    (55x)

The long 3-5 A "bonds" with flat kappa were soft, strongly coupled charge
channels that widened K's spectrum; switching them off smoothly narrowed it.

**The old seed was harmful.** The antisymmetric half of the attention logit
whose symmetric half is kappa: untrained, sharing kappa's weights, and measured
~10x WORSE than starting CG from zero (relative k = 6 error 3.4e-3 - 7.3e-3
against 1.9e-4 - 8.4e-4 on held-out 27-216 atom clusters, trunk features).

**The exact form of the answer.** At the solution, stationarity of the bond
block gives, for bonds inside a fragment,

    tau* = -kappa^-1 F B^T u* ,      u* = dU/dq (the potential on each atom)

verified to 1.9e-16. The optimal split charge is Ohm's law: conductance times
the potential difference across the bond. So the seed is

    tau_0 = -kappa^-1 F B^T u_hat(h) ,   u_hat = w chi + MLP(invariants(h))

— N learned numbers instead of n_bonds, with kappa, the envelope and B supplied
exactly by the physics. `heads.SolverSeedPotential` owns every parameter it
uses and is zero-initialised: an untrained seed is exactly zero.

**Dipoles come from physics, not learning.** A learned 1o dipole head recovered
nothing (its error equalled the whole dipole magnitude) although 20-38% of the
k = 6 error sits in the dipole sector. The solver instead takes one block-Jacobi
step on the dipoles from the seed charges, with an exact line search
(`solver._relax_dipole_seed`, `SolverOptions.relax_dipole_seed`, on by
default): it cannot raise U, touches no bond variable, and costs no parameters.
It uses only the charge-bond and dipole blocks (K[dipole, spin] is exactly 0).

**Training — a firewall in both directions.** `energy.solver_seed_loss` is
U(m_1)/n_atoms with every physics input detached: U(m_k) >= U(m*) with equality
only at the minimiser, so minimising it drives the seed toward m* without ever
computing m*. The data loss cannot reach the seed (the energy path evaluates it
with detached parameters, inputs attached, so forces stay exact); the seed loss
cannot reach the physics (measured: every non-seed head parameter moved by
exactly 0.0 over 600 steps). `ElectrostaticModule` accumulates the loss in
training mode (`take_solver_seed_loss`, weight `solver_seed_loss_weight`), and
train.py adds it to the backward pass only, never to the logged loss.

**Measured in the package** (held-out 27-216 atoms, TRACE-trunk features,
trained on 16-64 atoms for 600 steps), relative k = 6 error:

    N      zero seed    trained seed + dipole step    gain
    27     1.74e-04          1.29e-05                 13.5x
    64     3.73e-04          3.11e-05                 12.0x
    125    5.08e-04          2.96e-05                 17.2x
    216    6.24e-04          4.96e-05                 12.6x

The dipole step contributes 1.7-2.6x of that on top of the trained bond seed.
End to end, at N = 125 with trunk features: 30% before the envelope, 3.0e-5 now.

**Why a local seed can work at all.** The converged solution is nearsighted:
perturbing one atom of a 216-atom cluster moves charges by 4e-3 of the peak
response at 7-9 A, 7e-4 at 9-11 A, 1.5e-4 at 11-13 A.

**Honest limits.** With a random-initialised trunk the seed plateaus (training
past ~600 steps stopped helping) and has NOT been shown to reach the relative
error eps <= 0.05 that makes k = 1 size-independent — that figure came from
seeding at m* plus white noise, which carries information a local model does
not have. It must be re-measured on a trained trunk. The spin seed is still
zero and untrained.

### 2.0.2 Converged iterations are frozen, not unrolled into NaN

Past machine-precision convergence a CG step moves m by less than an ulp, but it
stays in the autograd graph, and the second derivative of its divisions carries
1/residual^4. The fixed-k polarizability (a second derivative of U) was exact
through k = 150 and NaN at k = 250 and 400 on a 6-atom chain (residual 5e-111),
with the energy finite. Iterations with `residual^2 <= eps^2 * residual_0^2` now
take step 0 and beta 0 through `torch.where` with safe inputs. While active every
expression is the original one, so the default k = 6 -- whose relative residual
is ~1e-2 -- is bitwise unchanged.

### 2.1 Runtime assertions — each from a measured failure

1. `eta_i >= S_0(sigma_i) = sqrt(2) K_E/(sigma_i sqrt(pi))` for all i, at
   construction (SPEC 1.2 / 17.3; the old `J(0)` floor is retracted).
2. `kappa_ij > 0` for all bonds, at construction.
3. **Schur complement**, projected onto the constraints, **AT THE SOLUTION**:

       S = H_qq - T01^T (A_mu + T_Thole)^-1 T01
       assert min_eig(P^T S P) > tol

   Checking the two blocks separately is THE trap: audit measured
   q block +0.58, dipole block +16.45, **full coupled -1.05**.
4. Verify the returned point is a MINIMUM, not merely stationary. Newton
   converges silently onto saddles and returns **negative** polarizability.
5. `|sum_{i in F} q_i - Q_F| < 1e-10` per fragment.

## 3. Readout heads (`heads.py`)

Emitted from TRACE node features `h` (`64x0e + 32x1o + 16x2e`, l_max = 2):

| quantity | irreps | note |
|---|---|---|
| chi_i, sigma_i, ell_i | 0e | |
| eta_i | 0e | `softplus(raw) + S_0(sigma) + margin` — mandatory (SPEC 17.3) |
| A_mu_i (dipole stiffness) | 0e + 2e | symmetric part of 1o (x) 1o |
| kappa_ij | edge scalar | symmetrised attention logit |

l_max = 2 does NOT support a general on-site quadrupole stiffness (needs 4e).
(v1 was monopoles and dipoles only. Quadrupoles are now wired with the
truncated 0e + 2e stiffness -- SPEC 10.1, 12, 14.)

    # on EVERY equivariant head -- use raise, NOT assert (python -O strips asserts
    # and would silently reinstate exactly this bug)
    if len(lin.instructions) == 0: raise ValueError(...)
    # NECESSARY BUT NOT SUFFICIENT: also require every OUTPUT irrep to be reachable.
    # A head declaring "0e + 2e" from a 0e-only input has ONE instruction and still
    # returns an identically zero 2e block. This is what rejects l_max = 1 features,
    # where the anisotropic part of A_mu would silently be dead.

A parity-mismatched `o3.Linear` constructs with zero instructions, returns
identically zero, and emits no warning. A mis-declared head trains to nothing.

### 3.1 e3nn basis convention — VERIFIED, do not "fix" it

In e3nn 0.4.4 `o3.Irrep("1o").D_from_matrix(R)` returns `R` itself (exact to
6.9e-18, checked against all six permutations). Degree-1 spherical harmonics act
as the identity. TRACE feeds plain cartesian `edge_vec` to `o3.SphericalHarmonics`,
so **no relabelling is correct** — `A_mu` is already in the cartesian frame and
contracts directly with physical dipoles.

The widely repeated "(y,z,x) ordering" folklore is WRONG here and applying it gives
0.78 relative equivariance error. Two independent checks confirmed this.

    torch.set_default_dtype(torch.float64)   # BEFORE constructing any e3nn module

e3nn caches generated TensorProduct code under the dtype at first construction.
Calling `.double()` afterwards is a silent no-op: 2.54e-08 vs 1.94e-15.

## 4. Observables (`observables.py`)

    P     = sum_i (q_i r_i + mu_i)
    alpha = D^T K^-1 D          # contains BOTH charge-transfer and induced parts
    Z*_i  = dP / dr_i

alpha is COMPUTED, never a readout. For a charged system `sum_i Z*_i = Q_tot`,
NOT zero — force-symmetrising to zero is wrong for any charged fragment.

Hyperpolarizability beta is NOT a derivative of this energy: U is quadratic
in the moments, so its exact third field derivative is zero.
`observables.hyperpolarizability` is a separate, constructed readout
(`beta_abc = d_a alpha_bc` from a learned 1o direction); see its docstring
for what that ansatz can and cannot represent.

## 5. Integration with flashace

`TransformersACEV3.forward` computes `h`, then `E = sum(readout(h))`, then takes
autograd on `pos` and `strain_params`. Add `U_elec` to `E` **before** that call;
forces and stress then follow automatically. Do not touch the autograd block.

Electrostatics must be **opt-in** and default OFF so every existing test passes
unchanged. `trace_electrostatics` must NOT import `flashace`.

## 6. Non-negotiable regression tests

| test | required value |
|---|---|
| `kappa -> 0` reproduces global QEq | 1e-15 |
| Ewald reproduces NaCl Madelung constant | 1.747564595 |
| size consistency, per-fragment, 0-200 spectators | 1.000000 exactly |
| `J(0) == 2 K_E / (sigma sqrt(pi))` | exact |
| analytic gradient vs finite difference (pos, charge, cell) | < 1e-10 absolute |
| gradient w.r.t. strain | **4.6e-10** — this is the FD roundoff floor for a -35 eV energy at h=1e-5 (eps*|f|/h ~ 3.5e-10), NOT an implementation defect. Relative error 3.8e-11 |
| `coulomb_kernel -> K_E/r` at large r | at r=6, sigma=1.5 the deviation is **3.700051e-08** = K_E*erfc(4)/6. That IS the erfc tail; 1e-10 is first reached at **r = 6.975 A**, and 8.3e-14 by r=8. An implementation hitting 1e-10 at r=6 would not be the erf kernel |
| `eta < J(0)` raises | must raise |
| asymmetric kappa raises | must raise |
| Thole `a >= 1` raises | must raise |
| Maxwell reciprocity `alpha == alpha^T` | 1e-10 |
| dissociation: QEq gives q_inf = (chi_B-chi_A)/(eta_A+eta_B) | 8 digits |
| chain polarizability `alpha = N(N^2-1)/12` at eta=1, J=0, spacing 1 | exact identity |

NOTE on the last two: the chain formula is `N(N^2-1)/12`, verified as an exact
identity. A previously quoted `N^3/12 - N/8 + 1/24` is WRONG for this model —
two independent audits confirmed the correction.

Maxwell reciprocity MUST be tested on a **close-contact, polarizability-
contrasted** system. On a sparse test system a known Thole bug shows only
0.005% asymmetry and passes silently; at 1.2 A contact the same bug gives 14.9%.

---

# PART II — extensions for general systems

Physics contracts for work beyond v1.  Same rule as Part I: every constraint here
is either derived or measured, and the derivation is stated so it is checkable.

## 7. Periodicity is an INPUT, never inferred from the cell

### 7.0 `k_max` MUST BE DERIVED FROM THE CELL — it is not a constant

The number of wave vectors a fixed Ewald accuracy needs grows linearly with the
box. `ElectrostaticConfig.ewald_k_max` therefore defaults to `None`, meaning
"derive from the cell via `EwaldParams.auto_tune`" — the same convention
`ewald_alpha` already used. An explicit value still pins it.

It used to be a static `DEFAULT_EWALD_K_MAX = 8`, and `ewald_params()` never
called `auto_tune`, which existed and was correct. The reciprocal sum was
therefore silently under-converged on any cell bigger than a small box. The test
is CROSS-MODULE and no single module can see it: for a neutral cluster held fixed
while the box grows, the images recede, so `E_periodic(L)` must fall to `E_open`.
Measured at `r_cut = 6 A`, `accuracy = 1e-12`:

    L (A)             12        20        28        40        56
    k_max needed      18        30        42        59        83
    pinned at 8   -2.4e-05  -1.1e-03  +4.9e-03  -3.8e-03  -3.7e-02   eV
    auto-tuned    -1.8e-05  -3.7e-06  -1.4e-06  -4.6e-07  -1.7e-07   eV

Pinned, the error GREW with L — the opposite of the physics — reaching 3.7e-02 eV
at L = 56 A, about EIGHT TIMES the cluster's total electrostatic energy of
4.6e-03 eV, silently. Auto-tuned it converges monotonically, as an Ewald sum must.
A production supercell is exactly where this bites and exactly where there is no
open-boundary reference to catch it. Guarded by
`tests/electrostatics/test_periodic_open_limit.py`.


ASE gives a gas-phase frame `cell = zeros(3,3)`.  Inferring "periodic" from
`cell is not None` then routes a singular lattice into Ewald, where `auto_tune`
dies in `linalg.inv`.  Periodicity must come from a `pbc` triple, matching ASE.

Three regimes, and they are different physics, not a flag:

| pbc | treatment |
|---|---|
| `(F,F,F)` | open boundary, direct pair sum.  A supplied cell is IGNORED |
| `(T,T,T)` | 3-D Ewald, tinfoil |
| `(T,T,F)` and permutations | **2-D slab.**  3-D Ewald on a slab is WRONG -- the periodic images along the non-periodic axis interact through their dipoles, giving a spurious `2 pi M_z^2 / V` term |

The slab case is not optional: SFG and every surface property this package
targets live there.  Either implement the Yeh-Berkowitz correction (3-D Ewald
plus vacuum padding, minus `2 pi M_z^2 / V`) with the padding requirement stated,
or a genuine 2-D Ewald.  State which, and its convergence condition.

`(T,F,F)` (wire) may raise; say so rather than silently doing something else.

### 7.0.1 A pinned alpha gets a grid sized for ITSELF

With `ewald_alpha` pinned and `ewald_k_max=None`, k_max was taken from
`auto_tune`'s own balanced alpha and paired with the pinned one: on a 30 A cell
k_max stayed 27 for every alpha and the fixed-moment energy moved 1.2e-6 /
2.8e-4 / 0.22 eV at alpha = 0.8 / 1.0 / 1.6 A^-1. Both tuners now take an
`alpha` override: k_max follows (26 -> 81) and the spread is 5.2e-13 eV. With a
pinned alpha the REAL-space tail erfc(alpha r_cut) is the caller's
responsibility.

### 7.1 A slab is tuned PER AXIS

`auto_tune` sizes one isotropic `k_max` from the LONGEST lattice vector, which on
a padded slab is the vacuum axis, so the in-plane grid became as fine as the
out-of-plane one and n_k grew as L_z^3 instead of L_z: 27,818,127 wave vectors
against 329,967 per-axis on an 8 x 8 x 80 A cell, and a 4-atom slab took 38.7 s
at L_z = 70 A. `ewald_params(cell, pbc)` now calls `auto_tune_slab` for
`sum(pbc) == 2` and the block builder reads `EwaldParams.k_grid`. Measured after:
~0.1 s at every padding, padding spread 1.6e-11 eV over L_z = 40..80 A.

### 7.2 Periodic bonds are resolved PER IMAGE — the energy must be extensive

A pair bonded through m periodic images is m physical charge-transfer channels
and must be m bond variables. The energy path built the topology WITHOUT
`edge_shift`, so bonds were keyed by atom pair and merged, one bond hardness for
m parallel routes. The energy then depended on the cell chosen to describe the
crystal: an 8-atom rocksalt cell has 24 nearest-neighbour bonds but 12 distinct
pairs, and measured E(2x2x2) - 8 E(primitive) = -5.5e-4 eV (2.6%), charges off
by 5.4e-4 e. DFT training data is mostly small cells, so this is a training-time
defect, not a corner case. `topology.build_bond_topology` already supported
`edge_shift`; `energy._build_incidence` now passes it. After: 1.2e-16 eV.

Neighbour-list convention, stated because getting it wrong fakes a residual:
`edge_vec = pos[i] - pos[j] + edge_shift @ cell`. A builder must measure
distances with the same sign it stores, and should assert
`max |edge_vec| < cutoff`.

## 8. Response must be the derivative of the energy the model DEFINES

(Still the rule. Since SPEC 17 the energy the model defines is the converged
minimum, so this route and the minimiser route coincide to the solve
tolerance; the fixed-k discussion below applies to `solve_mode="fixed_k"`.)

SPEC 2 defines `U` as the fixed-k CG iterate, not the minimiser.  `alpha =
D^T K^-1 D` is the response of the *exact* minimiser, so the two disagree at
finite k -- measured on an 8-atom cluster: `-d2U/dE2` = 2.871 at k=1, 7.224 at
the shipped k=6, 15.888 at k=200, against a reported 15.888.  **A factor of 2.2
at the default.**

The reported observable must be the derivative of the energy the model actually
evaluates.  Differentiate through the fixed-k solve.  `alpha` is then
`-d2U/dE2` by construction at every k, and Maxwell reciprocity
`alpha = alpha^T` remains a genuine test of the operator rather than an
artifact of using a symmetric inverse.

Keep the exact-minimiser route available and clearly named, since it is the
k -> infinity limit and a useful diagnostic.  Report the gap.

## 9. Spin channels

Split the charge sector only; the induced dipole responds to the TOTAL density:

    q_i^sigma = q_i^(0)sigma + sum_j t_ij^sigma + rho_i^sigma ,   sigma in {up, down}
    q_i = q_i^up + q_i^down          (charge)
    s_i = q_i^up - q_i^down          (spin population)

Per fragment F, TWO constraints:

    sum_{i in F} q_i = Q_F ,        sum_{i in F} s_i = 2 S_F

Energy additions: the Coulomb kernel couples TOTAL charges (`J_ij q_i q_j`, not
per spin), and an on-site exchange term `0.5 J^x_i s_i^2`.

### 9.0 Bond hardness: the (total, spin) basis, NOT per channel

An earlier draft of this spec said "bond hardness is per spin".  Read literally
that is WRONG -- it halves the effective hardness and breaks the closed-shell
limit.  With `t_up = t_dn = t/2`, a per-channel penalty gives
`2 * 0.5 kappa (t/2)^2 = 0.25 kappa t^2` against the single-channel
`0.5 kappa t^2`: a factor of exactly 2 (measured 0.027380000 vs 0.054760000).

Use the same decomposition SPEC 9.1 uses for the on-site term.  Define

    t_ij   = t_ij^up + t_ij^down      (total transfer along the bond)
    tau_ij = t_ij^up - t_ij^down      (spin transfer along the bond)

and penalise them separately:

    0.5 kappa_ij t_ij^2  +  0.5 kappa^s_ij tau_ij^2

`kappa_ij` is then exactly the closed-shell bond hardness, and `kappa^s_ij` is an
independent spin-transfer hardness.  Both must be `> 0` for the same reason
`J^x > 0` is required in 9.1 -- the two modes are the eigenvectors of the block,
so each eigenvalue must be positive.

**The closed-shell limit is a non-negotiable test, not a nicety.** Running a
closed-shell system through the spin path with `S = 0` must reproduce the
closed-shell energy to machine precision.  It is the single strongest
consistency check on the whole spin implementation, and a factor-of-2
normalisation error is exactly what it catches.

### 9.1 A sign constraint that is NOT negotiable

In the (q_up, q_down) basis, `0.5 eta q^2 + 0.5 J^x s^2` has Hessian

    [[eta + J^x, eta - J^x], [eta - J^x, eta + J^x]]

with eigenvalues `2 eta` (charge mode) and `2 J^x` (spin mode).  So positive
definiteness requires **`J^x_i > 0`**, exactly as it requires `eta_i > 0`.

`J^x > 0` PENALISES spin polarisation.  Hund's rule and Stoner ferromagnetism
need the opposite sign, and that sign is precisely the non-PD, unbounded case --
the same structure as the polaron instability audited earlier.  Therefore:

**Spin polarisation in this model comes from a spin-dependent linear term
`chi_i^sigma`, never from a Stoner instability.**  Say so in the docstring.  A
learned `J^x` must be `softplus(raw) + floor`, like `eta`.

### 9.2 The error this inherits, stated up front

A quadratic-in-`s` on-site energy violates the fractional-spin (flat-plane)
condition.  Measured: for a one-electron system where the exact E is FLAT in
spin polarisation, a quadratic term with J = 2 eV manufactures a **1.0 eV**
splitting at m = 1.  Real Fe(II) high-spin/low-spin gaps are **0.1-0.4 eV**, so
the artifact is 100-400% of the signal.

This is the same functional form AIMNet2-NSE and MACE-POLAR-1 use.  It is not a
reason to refuse the feature, but the spin-state splitting is a FITTED quantity,
not a derived one, and the docstring must say so.

## 10. Quadrupoles

Detect, do not hard-exclude.  The irreps needed for each coupling coefficient:

| block | irreps required | l_max |
|---|---|---|
| `A_mu`  (sym. part of 1o (x) 1o)   | 0e + 2e             | 2 |
| q-Theta                            | 2e                  | 2 |
| mu-Theta                           | 1o + 2o + 3o        | 3 |
| `A_Theta` (sym. part of 2e (x) 2e) | 0e + 2e + **4e**    | **4** |

Enable quadrupoles when the feature irreps carry what the block needs, and raise
a NAMED error when they do not -- never construct a head whose anisotropic block
is silently dead (SPEC 3's reachability rule).  At TRACE's l_max = 2, `A_Theta`
is representable only in its 0e+2e part, which is an isotropic-plus-partial
response, NOT a general quadrupole stiffness.  That restriction must be stated
at construction, not discovered later.

### 10.1 A VARIATIONAL Theta REQUIRES A_Theta — without it the energy is unbounded

`A_Theta` is not an accuracy refinement, it is what makes the Theta sector
solvable at all. It is the ON-SITE quadratic cost of Theta; the inter-site T22
coupling alone is a Coulomb-like operator with a symmetric spectrum about zero.
Measured on a 2.5 A pair, the Theta-Theta block from T22 with no on-site term
has eigenvalues -0.619 to +0.619 — indefinite — so minimising over Theta runs
to minus infinity along the negative modes. This is precisely the pathology the
hardness floor `eta >= J(0)` prevents in the charge sector, and it is why
`A_Theta` cannot be treated as optional.

Consequence at TRACE's feature set (`64x0e + 32x1o + 16x2e`, l_max = 2):

| block    | needs           | status at l_max = 2                      |
|----------|-----------------|------------------------------------------|
| q_theta  | 2e              | representable                            |
| mu_theta | 1o + 2o + 3o    | blocked by PARITY too — TRACE has 2e, not 2o |
| A_Theta  | 0e + 2e + **4e**| NOT representable — needs l_max >= 4      |

CORRECTION to the paragraph above, measured rather than assumed: the
unboundedness applies to having NO `A_Theta`, not to l_max = 2 as such. With
`allow_truncated_a_theta=True` the head builds `A_Theta` from its 0e + 2e part
alone, warns with `QuadrupoleTruncationWarning`, and the result IS positive
definite (measured min eigenvalue +0.814 over a 6-atom cluster, shape [n, 5, 5]).
So a variational Theta at l_max = 2 is BOUNDED and legitimate; what it loses is
the 4e part of the stiffness, i.e. the response is isotropic-plus-partial rather
than a general quadrupole polarizability. That is an approximation, not an
error.

WIRED (open boundaries). `ElectrostaticParams` carries `A_Theta` and
`chi_theta`; the solver appends `Theta_hat` after `mu_hat`; the energy, the
dense Hessian, the constraint operator (zero columns) and the dipole derivative
(zero rows: a quadrupole carries no dipole, but it still changes alpha through
K^-1) all contract it. Enable with
`ElectrostaticHeads(quadrupole_blocks=('A_Theta', 'chi_theta'),
allow_truncated_a_theta=True)` and `assemble(..., quadrupoles=True)` under
`multipole_kernel='gaussian'` (SPEC 12 says why that is mandatory).

REFUSED, deliberately: the ON-SITE couplings `q_theta` and `mu_theta`. They are
unbounded off-diagonal blocks inside each atom's own (q, mu, Theta) block; the
Gram guarantee of SPEC 12 covers inter-site couplings only, and an unbounded
on-site block is exactly the SPEC 2.1.3 failure (+0.58, +16.45, coupled -1.05).
They stay out until they have a parameterisation bounded by construction.

(The lattice sum for Gaussian dipoles and quadrupoles is SPEC 14.)

Measured (tests/electrostatics/test_quadrupole_dispersion.py): converged CG =
exact minimiser to 1e-12; isolated atom Theta_hat = -A_Theta^-1 chi_Theta to
1e-16; forces vs FD 3.5e-9 (step-limited); pipeline assembly energy vs solver
energy 6.9e-18; dense Hessian vs autograd 1.8e-15; mu and Theta rotate as a
vector and a rank-2 tensor to 1e-17; alpha symmetric to 8e-17 with Theta on.
Two permanent quadrupoles interact through the bare T22 with a relative
correction that shrinks x8.33, x8.08, x7.99 per doubling of R (15 -> 120 A):
R^-3, dipole induction; x32 with the dipoles frozen (Theta-Theta induction).
A convention error would have left a constant offset.

## 11. Uniform field under periodic boundaries — SUPERSEDED by SPEC 15

This section used to split the field by sector: the dipole half always
coupled, the charge half only along directions in which the bond graph does
not percolate, because ``-E . sum_i q_i r_i`` is unbounded along a percolating
direction.  That split never ran -- the solver refused a field together with
the lattice-summed operators, so every periodic field raised -- and it is no
longer needed: coupled through the bond currents (SPEC 15) the field is
bounded by the bond hardness in EVERY direction, percolating or not.  The
projector, the percolation-generator analysis and the solver probe have been
removed.

## 12. The Gaussian (Gram) multipole kernel — K positive definite by theorem

`multipole_kernel='gaussian'`. Every site carries one Gaussian shape per rank,
of width sigma_0 (charge: the existing sigma), sigma_1 (dipole), sigma_2
(quadrupole), and its moments are the 0th/1st/2nd derivative densities. Every
block of K is then the Coulomb inner product of two such densities, i.e. the
matching derivative of K_E erf(r/s)/r with s = sqrt(sigma_a,i^2 + sigma_b,j^2).
The damping factors are the regularised incomplete gamma functions

    lambda_{2n+1}(r) = P(n + 1/2, r^2 / s^2)          (P(1/2, x^2) = erf x)

and drop into the existing tensors unchanged: measured against autograd
derivatives of K_E erf(r/s)/r, T01 1.1e-14, T 1.7e-14, T02 1.7e-14, T12 2.3e-13,
T22 5.2e-13.

THEOREM. The full atomic (q, mu, Theta) interaction matrix INCLUDING its self
blocks is a Gram matrix, hence PSD, for every geometry. Self blocks:
S_0 = sqrt(2) K_E/(sqrt(pi) sigma), S_1 = K_E/(3 sqrt(pi)/sqrt(2) sigma^3),
S_2 = K_E/(5 sqrt(2 pi) sigma^5), cross-rank self blocks zero (odd, or a delta
contracted with a traceless Theta). So

    K = M^T (Gram + D) M + diag(kappa, 0, 0),   M = blockdiag(B diag(f), I, I)
    D = blockdiag(eta - S_0,  A_mu - ell^2 S_1,  A_Theta - ell^4 S_2)

is positive definite whenever kappa > 0 and D > 0. One floor per rank:
  * charge: the hardness floor already gives eta - S_0 >= (sqrt(2)-1) S_0;
  * dipole/quadrupole: the widths are DERIVED from the stiffness, Thole-style,

        sigma_l^(2l+1) = K_E ell^(2l) ||A^-2||_F^(1/2) / (a V_l),  0 < a < 1

    which gives ell^(2l) S_l <= a lambda_min(A), residual >= (1-a) lambda_min.
    The Schatten-4 norm replaces 1/lambda_min because eigenvalues are not
    differentiable at the degeneracy every isotropic atom sits on.

Measured: 300 clusters (60 near-coincident), min/max eigenvalue >= +1.0e-10;
30 solver Hessians at the parameter floors, all positive definite.

Why it is REQUIRED for quadrupoles: the Thole-coupled (q, mu, Theta) Hessian at
the heads' own initial values (alpha 1.5 A^3, C 6 A^5) was indefinite in 4 of
121 six-atom clusters, min -0.135 eV, while every diagonal block AND every
pairwise sub-block was positive definite. The three-way coupling alone breaks
it; no block check can see it; the Gram construction excludes it.
`gaussian_a` default 0.5. Periodic cells: SPEC 14. Default kernel since
SPEC 17.5.

Established ingredients, cited: Gaussian charges make J a Gram matrix (the
hardness-floor argument of SPEC 1.2); Gaussian induced dipoles (Elking, Darden,
Woods, J. Comput. Chem. 28, 1261, 2007 -- the Gaussian dipole damping itself is
theirs). TRACE-native: the per-rank floors derived from
the learned stiffnesses, and their use to make the bond-SQE + dipole +
quadrupole Hessian positive definite by construction.

## 13. Dispersion — the zero-point energy of the same moment block

    E_disp = 1/2 sum omega_p - 1/2 sum omega_p^0
           = (1/2pi) int_0^inf dw [ln det(K_yy + w^2 M) - ln det(D_yy + w^2 M)]

K_yy is the SOLVER's induced-moment block (same learned alpha_i, same damped
coupling); D_yy its on-site part. Inertia from the Thomas-Reiche-Kuhn sum rule:
alpha_a omega_a^2 = N hbar^2/m_e per axis, so the mass is isotropic,
m_i = ell_i^2 / (N_i hbar^2/m_e), and the only new quantity is N_i, the
effective polarizable electron count (`ElectrostaticHeads(dispersion=True)`,
softplus + floor, starts at 2.0). hbar^2/m_e = E_h a_0^2 = 7.619964 eV A^2,
from the same CODATA pair that gives K_E.

Measured: log-det quadrature vs eigenvalue ZPE 1e-11 (16 Gauss-Legendre points;
8 -> 8.9e-6, 12 -> 1.0e-8), 2e-12 with quadrupoles; pair limit = London /
Slater-Kirkwood C6 to 1e-6 at R = 12 A; free H at N = 1: 12.83 eV.

Log-det, not eigh: energies agree to 1e-10, but on a cube, an octahedron and a
32-atom fcc block the eigh route gave NaN second derivatives (eigenvector
derivatives divide by degenerate gaps) and the Cholesky route stayed finite.
Force training needs the second derivative. Under SPEC 12, K_yy + w^2 M is
positive definite for every geometry and frequency: MBD's imaginary-mode
catastrophe cannot occur.

Range separation by SUBTRACTION (`dispersion_range = lambda`, default 0):

    E_disp = ZPE[A + T] - ZPE[A + T - T_w],  T_w = T with every width widened
                                              by lambda/sqrt(2)

T - T_w has Fourier weight g_i g_j (1 - exp(-k^2 lambda^2/4)) 4pi/k^2 >= 0, so
both forms are PD. Measured on a 3-atom dimer: intermolecular C6 unchanged to
1e-6 for lambda <= 2 A while the monomer's short-range dispersion falls from
-0.55 to -0.25 eV (-0.12 at 3 A). Widening T directly instead moved the C6 by
37 %: it strips the intramolecular screening (the MBD-vs-rsSCS trap).

Opt-in extras: `dispersion_quadrupoles` lets Theta fluctuate at the single-pole
QDO frequency 2 omega_bar (C8-type terms); an approximation of stated form. On
random parameters it added 149 % to E_disp, so it stays off by default.

Established, cited: quantum Drude oscillators (Jones, Crain, Sokhan, Whitfield,
Martyna, PRB 87, 144103, 2013); MBD (Tkatchenko, DiStasio, Car, Scheffler, PRL
108, 236402, 2012); its ACFDT form (Tkatchenko, Ambrosetti, DiStasio, JCP 138,
074106, 2013); Slater-Kirkwood / TRK. TRACE-native: the operator is the master
equation's own PD Hessian; the subtractive Gram-consistent range separation.

NOT YET: charge-flow dispersion. Bond transfers t are clamped in E_disp. That
is standard MBD, but it is INCONSISTENT when a large share of the static
polarizability sits in charge flow (conjugated, low-gap, metallic systems):
then the atomic alpha_i underestimates the dispersing polarizability. The
natural extension -- let t fluctuate too, with the dissociation envelope
switching it off between fragments -- needs a bond inertia and is unvalidated.

### 13.1 Periodic dispersion: q-sampling by folding, and the non-analytic Gamma term

A Gamma-only zero-point energy puts every cell of a crystal in phase.  On a
small cell that is wrong by factors: measured on 2-atom rocksalt,
-3.94 eV/cell at Gamma against a converged -0.494.  A Gamma-point supercell of
M1 x M2 x M3 cells IS an exact M1 x M2 x M3 q-grid (block diagonal in Bloch
waves), so the cell is folded onto the smallest supercell whose plane
spacings reach ``dispersion_supercell_length`` (default 10 A) and the energy
per cell is the supercell's over M.

The Gamma point carries the error that remains.  Tinfoil boundaries drop
k = 0, which is the right q -> 0 limit only for transverse fluctuations; a
longitudinal one also feels the depolarization ``(4 pi K_E/Omega) q_hat q_hat``
(the LO-TO physics of phonons).  By the matrix determinant lemma the missing
Gamma term is

    (1/2pi) int_0^inf dw  < ln(1 + (4 pi K_E/Omega) q_hat . chi(iw) . q_hat) >_q_hat,
    chi(iw) = U^T (K + w^2 M)^-1 U ,   U_i = ell_i I

-- the longitudinal dielectric function at imaginary frequency, averaged over
directions, from the Cholesky factor already in hand.  It is the coupled-dipole
form of the q -> 0 head treatment of plane-wave RPA; the direction average is a
Gauss-Legendre x trapezoid sphere rule of degree 23 (288 directions; a rotated
crystal moves by < 1e-10 relative, tested).  Measured, eV/cell:

    supercell     2^3       3^3       4^3        5^3
    Gamma only   -0.980    -0.631    -0.551     -0.523
    + Gamma term -0.519    -0.4944   -0.49353   -0.49352

The range-separated short-range form (SPEC 13) takes no Gamma term: T and T_w
share the 4 pi/k^2 head, which cancels in T_SR.  Slabs: no Gamma term.

## 14. The periodic Gram operator (`lattice.py`)

The Gram construction of SPEC 12 summed over the lattice.  With tinfoil
boundaries and k = 0 removed,

    U = (K_E/2)(4 pi/Omega) sum_{k != 0} |rho(k)|^2 / k^2 ,
    rho(k) = sum_j e^{-ik.r_j} sum_a exp(-k^2 sigma_{a,j}^2/4) c_a(k).m_{a,j},
    c_0 = 1, c_1 = -ik, c_2 = -(1/2) kk,

a sum of non-negative terms: the PD theorem of SPEC 12 holds for every
crystal with EXACTLY the same floors.  Ewald split per rank pair of combined
width s: real space ``K_E [erf(r/s) - erf(alpha r)]/r`` with damping factors
``P(n+1/2, r^2/s^2) - P(n+1/2, alpha^2 r^2)`` fed to the existing tensors;
reciprocal space the point-multipole sum with block factors cos, k sin,
kk cos, -(1/2)(kWk) cos, (1/2)k(kWk) sin, (1/4)(kWk)(kWk) cos; the
alpha-Gaussian self terms -2/sqrt(pi) K_E alpha, -(4/3)/sqrt(pi) K_E alpha^3,
-(4/5)/sqrt(pi) K_E alpha^5 removed; the real-space kernel's k = 0 component
-(pi K_E/Omega)(1/alpha^2 - s_ij^2) removed on the charge block (the s^2 part
sums to zero for a neutral cell).  Gaussian multipole Ewald itself is the pGM
model's (Wei et al., J. Chem. Phys. 153, 114116 (2020)).

Measured (tests/electrostatics/test_periodic.py): every block alpha-independent
to 1.2e-13 relative; the charge block equals the package's own, independently
written Ewald plus exactly the Gram term (2.5e-15); lattice Hessian at the
floors min eigenvalue +1.61 eV; a neutral cluster in a cubic box tends to the
open-boundary energy minus 2 pi K_E |M|^2/3V with residual x25.4, x30.6 per
doubling of L (L^-5); energy per cell identical in a cell and its supercell
(1e-19 without, 1e-18 with quadrupoles).  ``LatticeWidthError`` when a
Gaussian is wider than the image box can converge.

Under the Gaussian kernel ``ell_i ell_j W_ij != -W_ji`` for the charge-dipole
block even in open boundaries -- charge i with dipole j is a different overlap
from charge j with dipole i -- so the Thole-only antisymmetry detector is
waived there (K is symmetric regardless).

### 14.1 The seed must not see the simulation cell

The block-Jacobi seed steps (SPEC 2.0.1) used the diagonal blocks of K as the
preconditioner.  Under PBC those hold each atom's interaction with its own
periodic images, which differs between a primitive cell and its supercell, so
the seed -- and the fixed-k energy -- depended on the cell: 3.8e-7 eV/cell
after one CG step on 2-atom rocksalt vs its 2x2x2 supercell.  Preconditioning
with A_mu / A_Theta (the atom's own) removes it: 1e-18 at every k, tested for
both kernels on a non-centrosymmetric cell.  Under open boundaries those
diagonal blocks ARE A_mu bit for bit, so nothing changed there.

## 15. The bond-current polarization and the electric enthalpy

    Omega P = sum_i rho_i r_i + sum_b f_b tau_b d_b + sum_i mu_i ,
    d_b = the IMAGE-RESOLVED bond vector (from the caller's edge_vec),
    F(E) = U(m) - E . Omega P                    (Nunes-Gonze enthalpy)

Under open boundaries ``f_b d_b = (B^T r)_b`` and this is exactly
``sum_i (q_i r_i + mu_i)``: the existing field coupling ``chi_i -> chi_i - E.r_i``
IS a bond-current coupling written through B.  Under PBC it is single valued
(an atom wrapped into another image moves the energy by 1e-19 and P by 5e-18,
while sum_i q_i r_i jumps by 3.1e-2 e A), bounded by the bond hardness in every
direction, and it SEES THE DIVERGENCE-FREE CURRENTS no atomic charge records.

Self-image bonds are therefore KEPT (``build_bond_topology(self_image_bonds=
True)``): one per +-n pair, zero incidence column, zero seed, zero linear term
at zero field.  They are the dielectric response of a crystal in SQE:

* monatomic chain, q = 0 by symmetry: alpha_ct = sum_b (f_b d_b)^2/kappa_b
  exactly (0.0516027596), where sum q r gives 0 and the old topology had no
  bonds at all;
* an open chain's alpha per atom tends to the periodic value: 0.854, 0.961,
  0.990 of it at N = 10, 40, 160 (error ~ 1/N);
* eps_inf of CsCl is identical in the primitive cell (6 of 14 bonds
  self-image) and its 2x2x2 supercell (none); dropping the self-image bonds,
  as the package did before, lost 12.8 % of the susceptibility -- a
  dielectric constant that depended on the choice of cell.

Observables (all boundaries): ``bond_polarization``; ``polarizability`` on both
routes (fixed-k enthalpy with a separate ``charge_field`` for the channel
split; minimiser with bond-current rows of D); ``dielectric_tensor``
(eps_inf = I + 4 pi K_E alpha/Omega); ``born_charges_from_energy``
(-d2F/dE dr) and ``born_charges_minimiser`` (implicit derivative of the
exact minimiser) -- equal to 6.9e-18, acoustic sum rule 1.1e-19;
``static_dielectric_tensor`` (eps_0 = eps_inf + 4 pi K_E Z^T Phi^+ Z/Omega);
``nonanalytic_force_constants`` (the LO-TO term).  The enthalpy is exactly
quadratic in E at convergence (2.2e-19 eV).  SQE with kappa > 0 is a
dielectric, never a metal (Nistor and Muser, Phys. Rev. B 79, 104303 (2009));
"itinerant" polarization = Berry phase is Sprik's.  TRACE-native: the SQE
bond transfer IS the time-integrated bond current, so the itinerant
polarization becomes a STATE function of the configuration, and the
self-image bonds that carry it.

Charged fragments (free ions under PBC): the rho term is the ionic part of the
itinerant polarization, only as continuous as the positions supplied.

### 15.2 The fixed-k map has no derivative at an exactly symmetric point (LEGACY)

Applies to `solve_mode="fixed_k"` only; the converged default has no such
point (SPEC 17). CORRECTION to the text below: the discontinuity is NOT
removable and not confined to an exactly zero residual. Near a symmetric
structure the fixed-k response depends on the DIRECTION of the residual
(27-atom cube, three perturbation directions of size 1e-10: alpha 2.655,
2.497 and -1720 at k = 6, +5.2e9 at k = 12, against 2.725), and at a
symmetric structure with a NON-zero residual it is wrong and does not
converge with k (Td 0.315 at k = 6 and 12 vs 0.363; Oh 0.399 vs 0.730; D6h
-2.25 / +3.82 vs 1.46; ideal rocksalt eps_inf 2.780 vs 2.823).

When the zero-field projected residual is EXACTLY zero -- chi drives no
charge transfer: a homonuclear diatomic, a monatomic crystal -- CG's step
sizes are 0/0 ratios, m_k(c) is homogeneous but not differentiable at c = 0,
and autograd returned 0 for every response derivative.  Pre-existing and
silent: fixed-k alpha of a homonuclear diatomic was exactly 0 at k = 6 and 50
(true 0.270, 0.217, 0.205); with a 1e-12 asymmetry it is the true value, so
this is a removable discontinuity.  ``polarizability`` / ``dielectric_tensor``
now detect it and use the minimiser route with ``FixedKDegenerateWarning``;
``born_charges_from_energy`` raises ``FixedKDegenerateError`` pointing to
``born_charges_minimiser``.  Forces are unaffected (the solve is exact there).
NOT fixed: second derivatives of the fixed-k energy at such points (e.g. a
Hessian for phonons at a high-symmetry structure) lose the electronic
relaxation the same way.  The principled cure is a solver that is LINEAR in
its right-hand side (a fixed Chebyshev polynomial rather than CG); that is a
change to SPEC 2 and has not been made.

## 16. The matrix-free lattice route (`lattice_operator.py`, `solver="matrix_free"`)

The fixed-k CG needs only ``K v``.  The dense periodic route materialises
``[n_images, n, n]`` real-space tensors, dense reciprocal blocks, a dense K
and a dense incidence matrix; it tops out near 500 atoms (432 atoms: 16.9 s,
7.0 GB for energy + forces).  The matrix-free route forms no ``n x n``
object:

* real space: a pair list ``(i, j, n)`` within the cutoff from a linked-cell
  search (O(n), identical pair set to the image-box search, tested), with a
  handful of per-pair scalars -- the SPEC 14 damping factors at each ORDERED
  rank-pair width (Gaussian q_i-mu_j and q_j-mu_i differ) -- and every
  multipole contraction in closed form, e.g. the traceless part of
  ``T22:Theta = (K_E/4)[105 l9/r^9 rr(rThr) - 30 l7/r^7 (r(Thr)+(Thr)r) + 6 l5/r^5 Th]``;
* reciprocal space: the multipole structure factor, half of each +-k pair
  inside the cutoff sphere (10x fewer wave vectors than the full integer
  cube, same sum to the requested accuracy), chunked and recomputed in
  backward (checkpointed) so memory is one chunk, not every CG product;
* bonds: ``B diag(f)`` as scatter/gather over ``(lo, hi, f)`` and a sparse
  incidence; the fragment projector acts only on bonds that cross fragments.

It is the same functional, not an approximation: ``K v`` equals the dense
lattice operator to 1e-13 relative, and the whole route equals the dense one
-- energy 4e-15 (4e-13 with quadrupoles), forces 6e-14, charges 1e-15, Born
charges 3e-12, eps_inf 3e-11 -- for both kernels, with and without
quadrupoles and fields (tested WITH THE SAME Ewald parameters on both routes;
with each route's own default parameters they agree to `ewald_accuracy`,
~2e-7 relative at the default 1e-6, not to round-off).  The seed steps precondition with A_mu /
A_Theta exactly as the dense route does (SPEC 14.1).

Ewald balance (``ewald_balance``, default on): the real-space cutoff
minimises the cost of an energy + force evaluation, ``r ~ N^(1/6)``
(``balanced_real_cutoff``), floored at 4.5 x the widest Gaussian pair.  The
cost ratio is MEASURED for the full pipeline (0.06); the forward product alone
measured 0.55, which put the cutoff at 17 A where energy + forces are slowest
(backward through the per-pair real-space terms dominates; the reciprocal
chunks are checkpointed, the real-space terms are not).

Measured, rocksalt supercells, CPU, 4 threads, energy + forces, accuracy 1e-8,
LEGACY fixed-k solve (k = 6) with the direct k-sum (see SPEC 19 for the
converged solve with PME):

    atoms        1024     2000     3456     6750
    time (s)      5.3     14.0     37.2    151.5
    peak RSS (GB) 1.4      2.8      4.2      4.9

energy per atom identical at every size.  Scaling ~ N^1.5 up to 2000 atoms,
~ N^1.8 over the whole range on this machine.  Particle-mesh Ewald: SPEC 19.
NOT DONE: real-space checkpointing (memory); slabs, dispersion and the
dense-operator observables (minimiser route, dense_hessian,
born_charges_minimiser) refuse on this route with a named error -- the
default-route observables (alpha, eps_inf, Born charges) work.  "No n x n
object" refers to K and the pair blocks: the converged solve's preconditioner
(SPEC 17.6) factorises one n_atoms x n_atoms atom-space matrix per solve up
to 8000 atoms (0.5 GB) and leaves the bond sector unpreconditioned above.


## 17. The converged energy and its implicit derivatives (`implicit.py`) — the default

    E(theta) = min_{x in V} U(x; theta),   U = 0.5 x^T K x + c^T x + u_const

solved to a relative residual `solve_rtol` (default 1e-10) by preconditioned
projected CG, and differentiated IMPLICITLY -- never through the iterations.
`solve_mode="converged"` (default) in `ElectrostaticConfig` / `SolverOptions`;
the SPEC 2 energy `U(m_k)` stays available, bit for bit, as
`solve_mode="fixed_k"`.

### 17.1 Why the fixed-k definition was replaced

The fixed-k iterate is `x_k = p_k(K) c`, with the polynomial fitted to the
eigenvalues the zero-field residual `c` excites. Its response to anything `c`
does not excite is arbitrary, and more k does not fix it. Measured (untrained
heads, isotropic species), fixed-k alpha at k = 6 / k = 12 against the exact
minimiser: CH4-like Td 0.315 / 0.315 vs 0.363; SF6-like Oh 0.399 / 0.399 vs
0.730; C6H6-like D6h -2.25 / +3.82 vs 1.46; ideal rocksalt eps_inf 2.780 /
2.780 vs 2.823 (a 0.02 A distortion restored agreement in every case); a
27-atom cube perturbed by 1e-10 in three directions: 2.655, 2.497, -1720
(+5.2e9 at k = 12) vs 2.725. CG is also not invariant under rescaling of the
variables, so the per-site length `ell` -- a pure gauge of the physics
(converged energy unchanged to 4e-16 under ell -> c ell, A -> c^2 A) -- moved
the k = 6 energy by 0.108 eV (Thole) / 0.013-0.028 eV (Gaussian) on a
12-atom cluster. The minimiser has neither defect; all eight cases above now
return the minimiser's value on the default route.

The reasons SPEC 2 gave against a converged solve do not apply to this
model: it is quadratic and (Gaussian kernel) positive definite by theorem, so
the minimiser is unique and smooth -- there is no stabilised nonlinearity for
a global minimiser to run away on; and the "Hellmann-Feynman drift" was the
drift of forces taken at an UNCONVERGED point. At a converged point the
envelope theorem makes the Hellmann-Feynman force exact to O(residual).

### 17.2 Implicit derivatives, to every order

* `_MinimumEnergy`: backward = envelope theorem, `dE = x*^T dc + 0.5 x*^T dK x*`.
  Forces need one extra operator application and NO linear solve.
* `_Minimiser`: `x*`, backward = adjoint solve `K s = g` on V,
  `grad_c = -s`, `grad_T = -d/dT <s, K_T x*>`; `s` is produced by the same
  Function, so it is differentiable again.
* The operator's explicit dependence is taken through a pure factory
  `factory(tensors) -> (v -> K v)`, differentiated on leaf copies re-joined to
  the originals (`T_eff = T_leaf + (T - T.detach())`), so one implementation
  serves the dense K and the matrix-free lattice operator (whose factory
  rebuilds the operator from positions, cell, widths and head outputs).

Checked against a closed form autograd differentiates on its own (a masked
`linalg.solve`): value, first, second and THIRD derivatives of E and first and
second derivatives of x*, to 1e-9 relative, with mask, preconditioner and a
multi-tensor factory (`tests/electrostatics/test_implicit.py`).

Conservativeness, measured without an integrator: the work of the forces
around a smooth closed loop in the configuration space of a 16-atom cluster
(400-point periodic trapezoid rule): -1.2e-9 eV at solve_rtol = 1e-10, the
same as the exactly conservative fixed-k energy (-1.3e-9, the quadrature
floor); +1.2e-7 at 1e-6; +2.2e-5 at 1e-3 (|E| = 4.7 eV). Forces against
finite differences: 1.2e-11 eV/A worst case at the default tolerance.

The seed and the dipole / quadrupole relaxation steps are now only the
starting point of the solve: they change the iteration count, never the
energy. The seed loss (SPEC 2.0.1) is still the one-step fixed-k energy, so
the seed keeps its training signal and keeps paying for itself in iterations.

### 17.3 The hardness floor is the Gram bound plus an explicit margin

`eta = softplus(raw) + S_0(sigma) + margin`, `S_0 = sqrt(2) K_E/(sigma sqrt(pi))`,
margin 1 eV/e^2 (`kernels.hardness_floor`). The theorem condition
`eta >= S_0` is what `assert_hardness_floor` checks. See SPEC 1.2 for the
retraction of the old sqrt(2)-larger floor.

The tight floor is safe under the GAUSSIAN kernel only, where the coupled K is
a Gram form plus positive on-site residuals. The Thole charge-dipole coupling
is not a Gram form and relied on the old floor's slack: a random 14-atom
cluster (min separation 0.83 A) measured min eig K = -0.999 at a 1 eV margin
with both diagonal blocks positive definite (+0.69, +3.44), and +0.69 under the
old floor. With `multipole_kernel="thole"` use
`ElectrostaticHeads(hardness_margin="legacy")` (eta floor sqrt(2) S_0, the old
one). That is NECESSARY, not sufficient: nothing bounds the Thole coupling at
close contact (a 14-atom cluster with a 0.14 A pair was still indefinite under
the legacy floor, and positive definite under the Gaussian kernel). The
converged solve raises `IndefiniteOperatorError` rather than return a saddle.

### 17.4 Explicit fragments exchange no charge; the envelope lives in a shell

Feasible set V: every bond joining two DIFFERENT fragments is pinned,
`tau_b = 0` (`solver.crossing_bond_selector`). The old constraint only kept
each fragment's net charge, `sum_{b crossing F} f_b tau_b = 0`, which let
current pass from one neighbouring fragment to another THROUGH a third, and
its projector carried the dissociation envelope, so it moved with the
positions and needed an SVD derivative (NaN at degenerate configurations).
Connected-component fragments, the default, have no crossing bonds.

Dissociation envelope (`energy.bond_envelope`, `ElectrostaticHeads(envelope_onset=0.75)`):
f = 1 for r <= 0.75 r_c and the C2 quintic switch `1 - x^3 (10 - 15x + 6x^2)`
on the shell. The old full-range cosine put a fixed d ln t/dr = 2 f'/f
(-0.9/A at 2 A of a 5 A cutoff) into every Born charge with a fixed sign
(rocksalt, NN 2.8 A of a 3.5 A cutoff: q_Na = +0.24, Z*_Na = -0.90, confirmed
by finite differences of the model's own P). The bond-length dependence of
charge transfer belongs to the learned radial bond hardness kappa(r).
`envelope_onset=None` restores the cosine.

The shell removes the bias only for bonds in the FLAT region. Rocksalt with a
+-3 eV electronegativity contrast (NN 2.80 A), Na Born charge:

    r_max   envelope        q_Na     Z*_Na
    5.0     cosine         +1.23    -0.54
    5.0     shell 0.5      +1.44    +1.05
    5.0     shell 0.75     +1.45    +1.40    (default)
    3.5     shell 0.5      +1.08    -5.14    (NN inside the steep shell)

Rule: the bond cutoff `r_max` must be at least ~1.4x the first-shell bond
length, so that the dominant transfer channels sit where f = 1. Pairs in the
shell (3.75-5 A at the default) are still attenuated smoothly.

### 17.5 Gaussian kernel by default; indefiniteness is detected, never returned

`multipole_kernel="gaussian"` is the default (positive definite by theorem,
SPEC 12 and 14). Thole stays available; its `a < 1` guarantee is a pair
guarantee (SPEC 1.6). The converged CG raises `IndefiniteOperatorError` on
the first direction with `p^T K p <= 0` -- the energy would be unbounded
below -- at no extra cost, and `SolverNotConvergedError` rather than return a
truncated iterate. The periodic Gaussian routes floor the real-space cutoff at
the widest pair's accuracy radius (`energy._gaussian_cutoff_floor`) instead of
raising `LatticeWidthError` at a user-chosen cutoff.

### 17.6 The preconditioner

The bond sector is a diagonal plus a weighted bond-graph Laplacian,
`kappa + B^T (eta + J) B`; its ill-conditioning is the Laplacian structure,
which no diagonal preconditioner sees (the bond diagonal was measured SLOWER
than none). The converged solve preconditions it with the Woodbury identity,

    (kappa + B^T C B)^-1 = kappa^-1 - kappa^-1 B^T (C^-1 + B kappa^-1 B^T)^-1 B kappa^-1,

one atom-space Cholesky per solve: `C = eta + J` (exact inverse of the bond
block) on the dense route, `C = diag(eta)` matrix free (dense atom-space
factor up to `WOODBURY_MAX_ATOMS` = 8000), `C = diag(J^x)` for the spin
channel (exact). Because `1^T B = 0`, `C + s 11^T` gives the same
preconditioner for any s, which makes C invertible when it is not definite
along the uniform charge direction (point-charge kernels, lattice sums without
k = 0). Moments: each atom's own 3x3 / 5x5 block. Iterations to
solve_rtol = 1e-10 on rocksalt supercells, (none / bond diagonal / Woodbury):
64 atoms 53 / 85 / 22 (dense); 216 atoms 67 / 109 / 25 (dense), 147 -> 43
(matrix free); 512 atoms 151 -> 56 (matrix free). Adding the operator's
near-field real-space Coulomb kernel to the matrix-free factor (n <= 2048,
`NEAR_FIELD_MAX_ATOMS`): 512 / 1728 / 4096 atoms 57 -> 48, 80 -> 69, 98 -> 85
iterations, time-neutral at 4096 (one extra n^3 inverse), so capped.

The remaining growth with size is the long-wavelength Coulomb, measured, not
assumed (round 4, rattled NaCl, matrix-free direct k-sum). Iterations and the
largest eigenvalue of the preconditioned operator, from the CG/Lanczos
coefficients:

    n                          64     216    512    1000
    shipped factor      it     32     40     49     54
                    lambda_max 2.81   4.90   7.67   10.4    (~ a + b L^2)
    oracle C = eta + J_full
                        it     24     27     29     29
                    lambda_max 2.27   2.36   2.42   2.43

`lambda_max` grows as L^2, the 1/k_min^2 stiffness of the longest charge-density
wave, which a local factor does not contain; with the exact periodic kernel the
condition number flattens at ~7. Consequence for the fix: the modes to correct
are all |k| < k_c, whose number is proportional to n, so no fixed-rank
correction (deflation of a few low-k modes) stops the growth. A Fourier-diagonal
(particle-mesh) correction of the atom-space factor is required, and it must
keep the preconditioner SPD: with `S = C^-1 + B kappa^-1 B^T` the Woodbury form is
SPD iff the approximation of `S^-1` stays below `(B kappa^-1 B^T)^+` on the
charge space, which a naive Fourier correction does not guarantee. Open.

## 18. Open boundaries without an n x n object (`open_operator.py`)

`solver="matrix_free"` now serves open boundaries too (molecules, clusters,
proteins). `OpenOperator` evaluates every pair i<j of the FULL kernel (Gaussian
`P(n+1/2, r^2/s^2)` at each ordered rank-pair width, or Thole) in dense tiles of
bounded size: O(n^2) time, O(n) memory, no `[n, n]` tensor, product chunks
checkpointed. Same interface as `LatticeOperator` (`apply`, `apply_reduced`,
`inputs`, `rebuilt`), so the same matrix-free solve and the same implicit
derivatives (SPEC 17) serve both boundary conditions. The field couples through
the bond currents, which under open boundaries IS `chi -> chi - E.r` written
through B (SPEC 15).

Measured: product vs the dense blocks 2e-16 - 1.1e-15 relative (both kernels,
with and without quadrupoles, including coincident pairs); the whole route vs
the dense one on a charged 14-atom cluster: energy <= 4e-16, forces <= 6e-12,
charges <= 2e-11, alpha <= 1e-10 relative, with and without a field
(`tests/electrostatics/test_open_matrix_free.py`). Cost per product (Gaussian,
dipoles, 10 threads): 0.004 / 0.06 / 0.46 s at 500 / 2000 / 8000 atoms with the
coefficient tiles precomputed (memory-bounded mode 6-8x slower, <= 0.5 GB at
8000 atoms), where the dense route no longer fits at 8000.

NOT DONE: a far-field approximation (FMM / treecode), so large proteins remain
O(n^2); dispersion and the dense-operator observables refuse on this route as
on the periodic one.

## 19. Particle-mesh Ewald for the matrix-free periodic route (`pme.py`)

`ElectrostaticConfig(solver="matrix_free", reciprocal="pme", pme_order=8)`:
the reciprocal half of the lattice operator -- the point-multipole Ewald sum
with the alpha Gaussian; the site widths stay in real space (SPEC 14) -- is
evaluated by smooth particle-mesh Ewald (Essmann, Perera, Berkowitz, Darden,
Lee, Pedersen, J. Chem. Phys. 103, 8577 (1995)) with the dipoles and
quadrupoles spread and gathered through B-spline derivatives (Sagui,
Pedersen, Darden, J. Chem. Phys. 120, 73 (2004)): O(n p^3 + M log M) per
product instead of O(n n_k). The approximate operator is exactly symmetric
and its autograd derivatives are exact derivatives of the PME energy, so CG
and the implicit derivatives of SPEC 17 apply unchanged. The mesh is sized
from `ewald_accuracy` by a per-rank error model calibrated on 1960 cases
(`pme.pme_mesh`; actual/model error within [0.85, 3.6] at rank 2, <= 6.75 at
ranks 0-1); the real-space cutoff is `ewald_real_cutoff` (floored at the
Gaussian widths), not the direct-sum balance.

Measured, whole matrix-free solve, PME vs the direct k-sum on a triclinic
8-atom cell (both kernels, quadrupoles, field):

    requested accuracy   energy (rel)   forces (abs)   charges    eps_inf
    1e-6                 <= 9e-7        <= 6e-9        <= 2e-9    <= 5e-6
    1e-9                 <= 1e-9        <= 2e-11       <= 8e-12   <= 3e-9

Cost, rocksalt supercells, converged solve (SPEC 17), accuracy 1e-6, CPU:

    atoms   direct k-sum   PME (mesh)            iterations
    512       2.4 s        1.3 s  (42^3)             57
    1728     23.3 s        3.8 s  (63^3)             80
    4096       --         11.7 s  (84^3)             98

energy + forces; a training step (energy, forces, force-loss backward) with
PME: 1.5 / 2.2 / 7.4 s at 216 / 512 / 1728 atoms. The previous default
(truncated fixed-k CG, direct sum) took 2.1 s for the 512-atom training step
and 37 s for energy + forces at 3456 atoms: the converged, exact solve now
costs about what the truncated one did.

Tested: `tests/electrostatics/test_pme.py` (the mesh sum against the exact one
per rank, forces and double backward against finite differences, lattice and
mesh translation invariance, charged cells) and
`tests/electrostatics/test_pme_route.py` (the whole route).

## 20. Metals and constant-potential electrodes (`metals.py`)

`electrostatic_state(..., metallic=[n_atoms] bool, electrode_potential=[n_atoms] V)`
(model hook: batch keys `metallic`, `electrode_potential`).

SQE with kappa > 0 is a dielectric; a metal is kappa -> 0, but on the physical
bond graph that limit is singular (every bond loop carries a free
divergence-free current). A connected metallic region (metallic atoms joined
by bonds within one fragment) gets a SPANNING STAR of zero-hardness bonds
(1e-8 eV/e^2) from its hub: a tree, so K stays positive definite (the bond
block on independent columns of the Gram-plus-residual charge kernel), and the
minimiser equalises u_i = dU/dq_i over the domain -- QEq inside a conductor,
independent of the tree chosen. Electrodes: domains with a potential are joined
hub to hub by external-circuit bonds carrying `-(V_e - V_ref) tau`, which
imposes `u_e - u_ref = V_e - V_ref` with total charge conserved (the
constant-potential electrode of Siepmann and Sprik, J. Chem. Phys. 102, 511
(1995); Reed, Lanning, Madden, J. Chem. Phys. 126, 084704 (2007)); only
differences are imposed, as PBC requires. All electrodes of a circuit share one
fragment; one electrode alone, electrodes in different fragments, and
potentials on non-metallic atoms raise named errors. Spin + metals: refused.

Measured (tests/electrostatics/test_metals.py): a 4x4 metallic plate next to
an ion is equipotential to 1.3e-9 V (the same plate as an SQE dielectric:
6.8e-2 V); two electrodes at dV = 0.5 / 1.0 V: u_top - u_bottom = dV to
4e-9 (the 1e-8 bond hardness times tau), charges exactly +-Q and exactly
linear in dV; Legendre dE/dV = -tau_circuit to 8 digits; forces vs finite
differences 1e-8; matrix-free = dense to 1e-12. Image-charge law, from the
force on a unit charge above a neutral square plate (the rim charge's offset
drops out of the force), slope of (-F_z)^(-1/2) over z = 4-6 A against the
ideal-conductor 2/sqrt(K_E) = 0.5271:

    plate edge     27.5 A   57.5 A   87.5 A
    metal          0.648    0.552    0.5315
    SQE dielectric 0.674    0.585    0.5687

The metal converges to the continuum image law (0.8 % at 87.5 A).

Limits: the in-plane polarizability of a PERIODIC metal is infinite
physically; the star (a tree) has no winding loops, so it stays finite (the
physical metal-metal bonds, kept with kappa > 0, carry the winding response).
The external-circuit bond vector (for the bond-current polarization) is the
direct hub-to-hub vector: the path through the external circuit is undefined.

## 21. Integer charge localisation: coupled diabatic charge states (`charge_states.py`)

Charge equilibration cannot localise an integer charge (the delocalisation
error of every QEq/SQE model): a dissociating ion or an electron transfer
leaves fractional charges. Which fragment carries the charge is a discrete,
diabatic choice; the physics of the transfer is the coupling between choices.
`adiabatic_charge_states(state_fn, states, coupling)` (model hook: batch keys
`charge_states`, `charge_state_coupling`) solves each diabatic state -- integer
`fragment_charges` on explicit fragments -- and returns the lowest eigenvalue
of `H = diag(E_a) + V` (empirical valence bond, Warshel and Weiss, J. Am.
Chem. Soc. 102, 6218 (1980)). The local short-range energy is the same in every
state and only shifts the spectrum, so mixing at the electrostatics level is
exact. Two states in closed form, `E = (E_1+E_2)/2 - sqrt(((E_1-E_2)/2)^2 +
V^2)`: smooth through the crossing for V != 0, gap 2|V| there, the lower
diabatic state (integer charges, to O(V^2/dE)) far from it. Weights `c_a^2`
and the population-weighted charges are returned. V is an input (constant, any
differentiable function of positions, or the learned through-bond coupling of
SPEC 22 via `charge_state_coupling="through_bond"`). Cost: one solve per state.

Measured (tests/electrostatics/test_charge_states.py): an inversion-symmetric
donor-acceptor pair has E_A = E_B to 1e-9, E = E_A - |V|, weights 1/2; the
closed form equals the 2x2 eigenvalue to 1e-12; V -> 0 gives min(E_A, E_B);
forces through the crossing vs finite differences 1e-7; the hook path equals
the direct call.

Complement for trajectories without declared states: the ASE calculator's
`fragment_policy="initial"` fixes the fragments of the first structure, so a
charged molecule that dissociates keeps its charge on the union of its parts
and the energy stays continuous (tested across the bond cutoff: the step equals
the force-predicted change to 1e-10); the parts then carry the fractional
reference shares, which is the limit this section's states remove.


## 22. Through-bond electronic coupling of charge states (`transfer_coupling.py`)

The coupling of SPEC 21 is the physics of electron transfer -- exponential in
the donor-acceptor distance and set by what lies between them -- so it is
predicted, not supplied. Every atom is a tunnelling site, every bond of the
model's neighbour graph (one entry per image pair, `src < dst`, self-images
excluded) a hopping `h_b >= 0`. Loewdin partitioning onto the donor (D) and
acceptor (A) fragments, with B the remaining atoms:

    V_DA = w_D^T [ H_DA + H_DB (Delta_B - H_BB)^-1 H_BA ] w_A                (22.1)
    Delta_i = sum_{b incident to i} h_b + softplus(gamma_i) + g_min           (22.2)

`H_ij` sums `h_b` over the bonds joining i and j; `w = sqrt(softmax(s))` over a
fragment's atoms (learned redox-site amplitudes, unit norm). (22.2) makes
`Delta_B - H_BB` strictly diagonally dominant with positive diagonal: SPD for
every parameter value and geometry, smallest eigenvalue >= min gap excess
(Gershgorin), Green's function entrywise >= 0 (inverse M-matrix). Its Neumann
series is the sum over all D -> A tunnelling paths of the products of `h/Delta`
along them: McConnell superexchange (J. Chem. Phys. 35, 508 (1961)) and the
tunnelling-pathway picture (Beratan, Onuchic, Science 252, 1285 (1991)),
evaluated over every path at once. Closed form for a uniform bridge of n sites
(each with two bonds, so `Delta = 2h + g`):

    V(n) = h sinh(theta) / sinh((n + 1) theta),   cosh(theta) = Delta / (2h)  (22.3)

States couple directly iff their fragment charges differ by +1 on one fragment
and -1 on one other (one electron moved); the EVB element is `H_ab = -V_ab <= 0`
(nodeless ground state by Perron-Frobenius; two states depend on V^2 only).

Learned pieces (`TransferCouplingHead`, on `heads.node_invariants(h)`):
`h_b = f(r_b) h_ref exp(-beta (r_b - r_ref) + L tanh(phi_b / L))` with f the C2
envelope of 17.4, `phi_b` symmetric in the endpoints (`e_i + e_j`, `e_i * e_j`,
radial basis), `L = ln 100` (the environment moves a hopping by at most two
decades around the learned distance law; an unbounded `exp(phi)` reached
1e-39..1e15 eV hoppings under randomised weights), `beta = softplus(.)`.
Defaults: h_ref 2.5 eV at r_ref 1.5 A, beta 2.2/A (~0.03 eV at 3.5 A), gap
excess 1.5 eV, g_min 0.1 eV; last layers zero-initialised. Bridges of up to
2048 sites are solved by Cholesky, larger ones by Jacobi-preconditioned CG with
the implicit derivatives of SPEC 17.

Limits (stated, not hidden): Condon approximation (V does not respond to the
electrostatic state or an applied field); all hoppings >= 0, so destructive
interference between paths is outside the model; no data has trained a
coupling yet -- the head is verified structurally, not against reference
couplings.

Measured (tests/electrostatics/test_transfer_coupling.py): (22.3) to 1e-13
relative for n = 0..12 at three (h, g); smallest bridge eigenvalue >= min gap
excess for hoppings scaled 1e-3..1e3; V(D, A) = V(A, D) to 1e-15 and invariant
under atom permutation to 1e-13; CG route = dense route to 1e-11 in value,
1e-10 in gradient, 1e-9 in a Hessian-vector product; hopping exactly symmetric
under endpoint exchange and cubic at the cutoff (ratios 8 and 4 on halving the
distance); forces through the learned coupling and the EVB vs finite
differences 1e-7 (a mutation that detaches the coupling's geometry fails it);
YAML block -> checkpoint -> strict reload in the ASE calculator, with
`info['charge_state_coupling'] = 'through_bond'` lowering the adiabatic energy
of a hydrogen-bonded pair (tests/test_train_electrostatics.py).


## 23. The charge-transfer contact law of the bond hardness (`heads.py`, opt-in)

**The failure it answers (measured, round 4).** Water-cluster models trained
on MB-pol energies and forces alone learned a bond hardness flat in distance.
The 6 A model has 0.25 eV/e^2 on O-H bonds and 0.21 between molecules
4.5-6 A apart; the 3 A model sits at the 0.1 floor everywhere. Their
polarizability per molecule against PBE0/aug-cc-pVDZ:

| molecules | 3 A model (A^3) | 6 A model (A^3) | PBE0 (A^3) |
|---|---|---|---|
| 1 | 2.48 | 2.52 | 1.35 |
| 4 | 6.13 | 6.27 | 1.44 |

Charge transfer carries 68-99 % of it. This is the charge-transfer
polarizability catastrophe of QEq-type models, here with SQE's bond hardness
left free by the data. Energies do not pin the bond hardness; the response
does.

**The physics.** A charge-transfer channel between two sites coupled by `h`
across a gap `Delta` has charge susceptibility `chi = 2 h^2 / Delta^3`
(second-order perturbation theory), so the bond hardness is

    kappa = Delta^3 / (2 h^2).                                               (23.1)

Two consequences follow:

1. Breaking orbital contact lowers `h` by more than an order of magnitude,
   from about 2.5 eV along a covalent bond to about 0.1 eV across a hydrogen
   bond. `kappa` must therefore JUMP by `K ~ (25)^2 ~ 10^3`.
2. Beyond contact, `h` decays as a tunnelling tail, and `kappa` grows as
   `exp(beta r)`, with `beta ~ 3/A` (the through-space decay of
   electron-transfer rates, `h^2 ~ exp(-beta r)`).

A gentle exponential alone is not enough. Even with `beta = 3/A` starting at
contact, a hydrogen-bonded charge transfer pays only the on-site hardnesses,
about 15-20 eV/e^2, over a 2 A lever arm, which is d^2 K_E / 20 ~ 3 A^3, twice
water's polarizability.

**The law.**

    kappa_b <- kappa_b * exp( ln(K) sigmoid(x) + beta w softplus(x) ),
    x = (r_b - R_i - R_j - delta) / w,                                       (23.2)

The pieces:

- `R` are covalent radii (Cordero et al., Dalton Trans. 2008, 2832;
  `constants.COVALENT_RADII`, Z = 1..96).
- The multiplier is >= 1, C-infinity (softplus as `logaddexp(x, 0)`, so there
  is no threshold kink) and exactly symmetric in i <-> j.
- It applies to the spin-channel hardness too.
- Defaults: `K = 1e3`, `delta = 0.5 A`, `w = 0.1 A`; `beta` is
  `charge_transfer_decay` (setting it enables the law; 3.0/A recommended).
  `K` and `beta` are FIXED hyperparameters, deliberately not learned: if they
  were learnable, training on energies alone would drive them flat again,
  which is exactly the failure above.

Factors at the defaults:

| contact | distance | factor |
|---|---|---|
| O-H | 0.96 A | 1.04 |
| C-C | 1.54 A | 1.06 |
| Si-O | 1.61 A | 1.01 |
| Na-Cl (ionic contact kept) | 2.82 A | 1.21 |
| C-C, stretched (charge transfer switching off as the bond breaks) | 2.0 A | 27 |
| O...H | 1.9 A | 3.3e3 |
| O...O | 2.8 A | 1.9e4 |

The atomic numbers reach the heads as `species` (`assemble`,
`electrostatic_state`, `electrostatic_energy`, the model hook through
`data['z']`, sliced per structure in batched training); heads without the law
never receive them.

**Measured** (`tests/electrostatics/test_contact_law.py`). The three regimes
hold: in contact the factor is 1.04 (O-H) and 1.21 (Na-Cl); beyond the
onset it is `K exp(beta (r - contact))` to 1e-3; the tail slope is `beta` to
1e-6. The exchange symmetry is bitwise. Forces against finite differences
agree to 1e-8. With the law off, energies and `kappa` are bit-identical.
Polarizability per molecule (alpha_iso) of water chains with the same random
heads, 1 -> 6 molecules:

| law | growth |
|---|---|
| none | x3.5 |
| gentle exponential (`beta = 3`, no jump) | x2.0 |
| (23.2) | x1.20 |

What remains is induced-dipole mutual polarization. Raising `K` from 1e3 to
1e4 changes it by less than 0.5 %, so no charge transfer is left in it.

Established, cited: the perturbative charge susceptibility; covalent radii as
the contact scale (as in Grimme's D3 coordination numbers); exponential decay
of electronic coupling. TRACE-native: (23.2) as a smooth, chemistry-aware
prior on SQE's bond hardness that turns charge transfer off out of contact
and on within it -- covalent and ionic contacts are untouched -- so a model
trained on energies alone keeps an extensive response.

**Limits.** Through-bond charge transfer over a long conjugated bridge is still
represented by the chain of in-contact bonds, as before. Charge transfer
between molecules in genuine contact -- charge-transfer complexes, ion pairs
closer than R_i + R_j + delta -- is not suppressed. `K`, `beta` and `delta`
are orders of magnitude from (23.1), not fitted constants. Water validation
of a trained model is in `experiments/electrostatics_water_dimers`.


## 24. Independent structures in one block-diagonal solve (`energy.py`, `flashace.model`)

`assemble(..., structure_id=[n_atoms])`, `electrostatic_state(..., structure_id=...)`,
`electrostatic_energies(state)` and `ElectrostaticModule.energies_by_structure(...)`.
`TransformersACE.forward_batched` uses the route automatically whenever every
structure of the batch qualifies; it is not an option to switch on.

**Why.** The batched training path used to call the electrostatics once per
structure. Each call issues about 5,000 small kernels and, before this round,
142 host synchronisations.

| one training step (energy + forces + backward), float64 | per structure |
|---|---|
| local model alone, RTX 5060 Ti | 3.5 ms |
| local + electrostatics, per-structure calls, RTX 5060 Ti (4,535 syncs per step of 32 Ag3) | 104 ms |
| local + electrostatics, per-structure calls, 2 CPU threads | 20 ms |

On the GPU that is about 5× slower than the CPU.

**The mathematics.** S independent structures with moment vectors x_s and
quadratic forms U_s = ½ x_s^T K_s x_s + c_s^T x_s give the batch functional
U = Σ_s U_s with K = blockdiag(K_s). Its minimiser is the concatenation of
the per-structure minimisers. Every derivative of U_s with respect to the
inputs of structure s is unchanged; none depends on another structure.

**The implementation, exact by construction:**
- Every pair block (J, T01, T2, and T02/T12/T22 with quadrupoles) is
  multiplied by 1[s(i) = s(j)].
- The SAME masked blocks go to the solver as the `Geometry` overrides the
  periodic route already uses. The open route's solver otherwise rebuilds its
  kernels from raw positions, so the masked blocks would change the reported
  energy but not the minimised one (the failure SPEC 7 measured at 0.513 eV
  for cells).
- Bonds and fragments never cross structures; this is checked.
- The caller applies each structure's own fragment and charge rules (SPEC 1.3),
  structure by structure, and supplies `fragment_id` / `fragment_charges`
  globally. The rules are never applied to the union.
- The per-structure energy partitions U term by term. Every term is a sum
  over atoms, bonds or ordered pairs (i, j), and with the masked blocks each
  belongs to the structure of i:

      U_s = Σ_{i∈s} [ χ_i q_i + ½ η_i q_i² + ½ μ̂_i·A_μ μ̂_i + ½ q_i (J q)_i
                     + q_i (T01 μ̂)_i + ½ μ̂_i·(T2 μ̂)_i (+ the Θ terms) ]
            + Σ_{b∈s} ½ κ_b t_b²                                          (24.1)

  Σ_s U_s = U holds identically, at any moments.
- U_s is evaluated at the solver's minimiser, whose derivatives are implicit
  (SPEC 17). So forces, and the parameter gradients of a force loss (second
  derivatives), are each structure's own.

**Refused, never approximated.** The model then uses the per-structure loop:
- periodic structures;
- applied fields, metals and electrodes, charge states (SPEC 21);
- spin heads, dispersion;
- a non-zero solver-seed loss in training;
- a structure whose own route would be matrix-free (more than
  `AUTO_DENSE_OPEN_MAX_ATOMS` atoms).

**Chunking.** The joint operator is dense, so one solve costs O(D²) per CG
iteration in its total size D. Consecutive structures are grouped into solves
of at most `flashace.model.BATCHED_ELECTROSTATICS_MAX_ATOMS_BY_DEVICE` atoms:
96 on CPU, 768 on CUDA.

**Measured** (one training step, batch 32 unless stated):

| system | CPU, cap 96 | CPU loop | GPU, cap 768 | GPU loop |
|---|---|---|---|---|
| Ag3 | 3.0 ms | 18.7 | 6.5 | 77 |
| C10 chain | 15.0 ms | 26.7 | 9.5 | 93 |
| NaCl | 19.4 ms | 29.2 | 14.1 | 87 |

- On CPU, one joint system for the whole NaCl batch costs 45 ms, more than the
  loop.
- On GPU, Ag3 at batch 128 costs 1.9 ms per structure. The C10 chain as one
  1,664-atom system costs 36 ms, hence the cap.
- Host synchronisations per step fall from 4,535 to 150–200. Contributing to
  this:
  - the CG of `implicit.py` reads curvature and residual in one transfer per
    iteration instead of three, with the same three checks and messages;
  - the model takes per-structure atom ranges from one `bincount`, not one
    `nonzero` per structure.

**Tests** (`tests/electrostatics/test_batched_structures.py`,
`tests/test_train_electrostatics.py`):
- **Package level.** Four structures placed on top of one another in space:
  neutral, cation, anion, and a two-fragment structure with its charge on one
  fragment. Against each solved alone, for Gaussian ± quadrupoles and Thole:
  energies 1e-10, forces 1e-8, charges 1e-9, and the parameter gradient of an
  energy + force loss to 1e-7 relative. The partition (24.1) holds to 1e-12 at
  random moments. Refusals are tested.
- **Model level.** Against the per-structure loop, in one chunk and in four:
  energies 1e-10, forces 1e-9, q/μ/ρ/t 1e-9, full model-parameter gradients
  1e-7 relative, and the polarizability through the lazy `state()` rebuild
  1e-9. The route is verified to be taken.
- **Mutation checks.** Each of these fails the tests: unmasked blocks; the
  solver rebuilding its own kernels; bond energy given to the wrong
  structure; the net charge ignored when composing fragment charges;
  energies returned in the wrong order.

**Not done.** Periodic structures still go one by one. They need a batched
Ewald; the Au2/MgO cells, for instance, are all identical.


## 25. PROPOSED: free charge, the derivative discontinuity and the Fermi level (design under review)

**Status (2026-10-03).** Implemented as **TRACE-Redox v1**, a separate package
(`trace_redox/`, contract in `trace_redox/SPEC.md`); nothing in
`trace_electrostatics` was changed for it. Implementing it found five errors in
this design, corrected in place and marked **[erratum]**: the sign of (25.1),
the convexity condition on the integer state energies, the λ_max design note,
R_c, and the softmax sign of 25.8 A. The measured physics of v1, including its
mean-field delocalization error, is in `trace_redox/SPEC.md` §7–8. Proofs are
given where a statement is mathematical. Every term is marked NEW-TO-TRACE or
ESTABLISHED (with a citation).

### 25.0 The diagnosis this answers (measured, SPEC 1 vs DFT)

SPEC 1 is a complete theory of **bound** charge: SQE bond fluxes, induced
dipoles and quadrupoles. For any region A, Q_A = 1_Aᵀρ + Σ_{b∈∂A} ± f_b τ_b,
and K ⪰ κ₀ I ≻ 0 is a gapped dielectric operator. A charge-neutral local
perturbation therefore puts net charge on a distant region only through
multipole tails, which decay with distance.

On Au2/MgO:Al, the charge inside R of the dopants, doped minus undoped:
- TRACE: returns to 0 by 5.5 Å, even when trained on Hirshfeld charges;
- DFT: +0.2…0.4 e out to 13 Å, balanced on Au2.

Table II comes out at −5 meV (DFT −932). Declared SPEC 21 states with the
whole slab as one unit recover −199 meV, but the hole spreads over the slab, so
the dopant barely matters. What is missing is **free** charge: a variational
electron-number degree of freedom with its own energetics. Polarization
screens free charge; it does not create it.

### 25.1 Prior art and where this differs (checked 2026-10-03)

- QEq-based ML potentials: 4G-HDNNP (Ko et al. 2021); Fe²⁺/Fe³⁺ (Kocer et al.
  2024). One χ per atom, a metallic global channel, no gap. Spurious transfer
  and over-polarization are reported (Vondrák, Reuter, Margraf, npj Comput.
  Mater. 11, 288 (2025)).
- MACE-POLAR-1 (arXiv:2602.19411): two non-self-consistent iterations and a
  single learned Fukui distribution of the total charge. No f₊ ≠ f₋, no gap.
- Baldwin, Batatia, Vondrák, Margraf, Csányi (arXiv:2603.14700): state the
  piecewise linearity and f₊ ≠ f₋, then restrict to "smooth surrogate models
  … a single Fukui function".
- Müser, EPJ B (2012): QE is metallic; SQE has a gap, i.e. is a dielectric.
- redoxSQE (Dapp & Müser, JCP 139, 064106 (2013)): Q_i = n_i e + Σ_j q_ij.
  **The bound-plus-integer decomposition is theirs.** But integer moves are
  random swaps across neighbouring bonds, accepted when the energy drops
  (history-dependent and non-variational), and one χ parabola acts on the total
  Q. No Fermi level, no long-range transfer, no gap.
- eReaxFF (Islam et al. 2016): explicit pseudoclassical electrons.
- bpopNN (Xie, Persson, Small 2020): an NN energy of atomic populations that
  needs constrained-DFT labels.
- Oxidation states as species (npj Comput. Mater. 2025, from DFT+U+V): labels,
  not variational.

What is NEW-TO-TRACE, as far as these works show:
1. a **convex, variational** free-charge sector built on the exact ensemble
   energy (the derivative discontinuity);
2. that sector equilibrated globally by one electrochemical potential and
   **screened exactly by the dielectric bound sector**, inside an ML potential
   with exact implicit derivatives;
3. the quantitative convexity–exactness theory of §25.4, with its link to the
   functional's delocalization error.

### 25.2 Objects

- **Redox units.** Default: atoms. Each unit u carries a free charge
  c_u = ⟨n_u⟩, where n_u is the integer number of electrons removed. The free
  charge is a Gaussian of learned width σ_u ≥ σ_min. σ_u is the extent of the
  donor or acceptor state; for a shallow donor it is the effective-mass radius,
  a_D = ε a₀ m/m*, which is large (ESTABLISHED, effective-mass theory).
- **Charge-state-resolved unit energies** E_u^{(n)}(G_u) for n ∈ 𝒩_u (default
  {−1, 0, +1}; with spin, pairs (n↑, n↓)). These are local, geometry-dependent
  and learned, defined **without** the long-range electrostatics that the model
  computes. E_u^{(0)} ≡ 0, absorbed into the local energy.
- **Anchors** (NEW-TO-TRACE):

      E^{(+1)} = IE_Z + δ_I(G),   E^{(−1)} = −EA_Z − δ_A(G),   |δ| ≤ L via tanh

  IE_Z and EA_Z are tabulated atomic values (NIST, ESTABLISHED). The
  electrostatic (Madelung) part of the environment shift is not learned: it
  enters through the computed potential u. δ is the small chemical part.
- **Total charge.** Q_i = q_i^b + c_i. The bound sector keeps its per-fragment
  constraints with every fragment neutral. The free sector carries the net
  charge: Σ_u c_u = Q_total, with no restriction to fragments.

### 25.3 The functional

    F(R) = min_{x, c}  E_bound[x; c]                      SPEC 1, unchanged; c enters like ρ
                     + ½ cᵀ J_ff c + cᵀ J_fb q^b          Coulomb of the free charge
                     + Σ_u Φ_u^T(c_u)                     the ensemble energy (25.1)
                     + ½ λ Σ_u J_uu {c_u}(1 − {c_u})      self-interaction removal, 0 ≤ λ < λ_max
           s.t. Σ_u c_u = Q_total                                                        (25.0)

    Φ_u^T(c) = sup_ν [ ν c − kT_e ln Σ_{n∈𝒩_u} e^{−(E_u^{(n)} − ½J_uu n² − ν n)/kT_e} ]   (25.1)

**[erratum]** The draft had `+ kT ln Σ e^{−(Ê + νn)/kT}`; that bracket is
convex in ν and its supremum is +∞. (25.1) as now written is the conjugate of
the grand potential Ω(ν) = −kT ln Σ e^{−(Ê − νn)/kT}, with N = −Ω' and
Φ' = ν (trace_redox/ensemble.py, tested).

Here {c} is the fractional part of c.

- (25.1) is the Legendre transform of the unit's grand potential: Mermin's
  finite-temperature ensemble (Phys. Rev. 137, A1441 (1965), ESTABLISHED).
- **Lemma 1 (convex envelope).** As T_e → 0, Φ_u^T converges to the lower
  convex envelope of the points (n, E_u^{(n)} − ½J_uu n²).
  - *Proof.* The ensemble energy minimises Σ_k p_k E_k over p_k ≥ 0 with
    Σ p_k = 1 and Σ p_k n_k = c. That linear program is the convex envelope,
    and by LP duality it is the T → 0 limit of the conjugate of log-sum-exp. ∎
- **Corollary.** For one electron, Φ(c) = χ̄c + γ|c| with χ̄ = (IE + EA)/2 and
  γ = (IE − EA)/2: the PPLB derivative discontinuity, an L1 kink (Perdew,
  Parr, Levy, Balduz, PRL 49, 1691 (1982), ESTABLISHED).
- **[erratum]** The Corollary describes the envelope of the TRUE integer
  energies. Lemma 1 acts on Ê = E − ½J n², whose kink is
  γ̂ = (IE − EA − (1−λ)J_uu)/2. When γ̂ < 0, state 0 lies above the
  (−1, +1) chord: a neutral unit mixes n = ±1 at zero mean, i.e. it
  disproportionates internally. This was measured in an implementation draft
  with the free charge on the bound σ = 1.5 Å (J = 7.66 eV > IE − EA for Na,
  Li, Al, Si, Cu, Au). trace_redox therefore parameterises γ̂ > 0 directly and
  gives the free charge its own hardness-derived width
  (trace_redox/SPEC.md §5).
- **Convexity needs no constraint on the learned E^{(n)}.** A charge state
  above the envelope is simply never populated (disproportionation). For
  T_e > 0, Φ_u^T is C^∞ and convex for any E^{(n)}.
- The −½J_uu n² in (25.1) removes the unit's self-Coulomb from the learned
  integer energies, because ½cᵀJ_ff c adds it back. At integer c the total
  on-site energy is E_u^{(n)} exactly.

### 25.4 The convexity–exactness theorem (NEW-TO-TRACE)

Let W = J_ff − MᵀK⁻¹M be the free-free block of the Coulomb operator after
exact elimination of the bound sector (its Schur complement).

- **W ≻ 0.** The full quadratic form ½τᵀκτ + ½q_bᵀη q_b + ½(q_b + c)ᵀJ(q_b + c)
  is positive definite: it vanishes only for τ = 0, then q_b = 0, then c = 0
  since J ≻ 0. A Schur complement of a PD matrix is PD.

Inside an integer cell the λ-term has curvature −λJ_uu, so

    F is convex in c   ⟺   W − λ D ⪰ 0,   D = diag(J_uu)
    λ_max = λ_min( D^{−1/2} W D^{−1/2} ) < 1      (since W_uu < J_uu)                (25.2)

Hence the **unique, history-free energy for 0 ≤ λ ≤ λ_max**.

Two units far apart (W_AB → 0). Moving δ ∈ [0, 1] from A to B:

    F(δ) = δ ΔE_int + ½ ( J̃_AB − λ(J_AA + J_BB) ) (δ² − δ),   J̃_AB = W_AA + W_BB − 2W_AB  (25.3)

ΔE_int is the integer transfer energy (IE_A − EA_B with screening). So:
- **spurious fractional transfer** (δ* > 0 although ΔE_int > 0) occurs iff
  ΔE_int < R_AB := ½(J̃_AB − λ(J_AA + J_BB));
- the energy is exactly linear (PPLB) along A → B iff λ = J̃_AB / (J_AA + J_BB).

**Correction** to the first draft: "no spurious charge
transfer" holds exactly only at λ = 1, outside the convex range. For
λ ≤ λ_max a residual self-interaction R_AB ≥ 0 remains on the less-screened
transfer directions. This is the force-field image of DFT's delocalization
error versus static correlation (Cohen, Mori-Sánchez, Yang, Science 321, 792
(2008), ESTABLISHED).

Design consequences:
1. **Spurious transfer shrinks as units grow.** J_uu ∝ 1/σ_u, so large
   learned widths shrink R_AB. Example:
   - molecule-sized units (σ ≈ 2 Å, J ≈ 4 eV) against a gap of ≥ 6 eV: no
     spurious transfer even at λ = 0;
   - bare atoms (σ ≈ 0.7 Å, J ≈ 11 eV) need λ ≳ 0.5.
2. **λ is a property of the reference functional.**
   - PBE delocalises (4G data: q(Au2) = −0.2 e from Hirshfeld): λ ≈ 0.
   - Range-separated hybrids such as ωB97M-V (OMol25) are nearer linear:
     λ → λ_max. **[erratum]** Not practical. Measured λ_max = 4.6e-5
     (benzene), 0.0074 (water dimer), 0.037 ((NaCl)₄) and 0.135 (NaCl dimer)
     at h = 0.5 with untrained bound heads (trace_redox/SPEC.md §7): a constant λ that matters is not
     convex in condensed molecules.
3. **First-order levels.** The levels at c = 0 are I_u = IE_u − ½(1−λ)J_uu and
   A_u = EA_u + ½(1−λ)J_uu. Residual self-interaction narrows the first-order
   gap by (1−λ)J_uu: the band-gap problem of the Kohn–Sham gap without the
   derivative discontinuity (ESTABLISHED).

### 25.5 Theorems to prove and test

| | statement |
|---|---|
| T1 | Existence and uniqueness of the minimiser for λ ≤ λ_max; F is C^∞ for T_e > 0. |
| T2 | **Dead zone.** At T_e = 0, c ≡ 0 iff max_u (A_u + u_u⁰) ≤ min_v (I_v + u_v⁰), u⁰ being the potential of the bound solution. Then F equals SPEC 1 **exactly**. A sparse (lasso-like) solution: free charge appears only where the electrochemical potential leaves a unit's first-order gap window. |
| T3 | Integer transfer between distant units iff ΔE_int < 0 for λ = J̃/ΣJ, with the residual of (25.3) otherwise. The Na···Cl point-charge crossing is at R_c = K_E/(IE_Na − EA_Cl) = 14.399645/1.526352 = 9.434 Å **[erratum: not 9.44]**, sharp as T_e → 0 at the exact λ. |
| T4 | **Forces.** F is a minimum over all internal variables, so −∂F/∂R is the explicit derivative (envelope theorem; Mermin forces are variational, ESTABLISHED). They include the population-weighted state forces −Σ_n p_un ∂E_u^{(n)}/∂R. |
| T5 | **Linear response.** About any solution, δc = −X(δν), with X_u = Var_u(n)/kT_e: an ACKS2-type response kernel (Verstraelen et al., JCP 138, 074108 (2013), ESTABLISHED), here as the small-perturbation limit of a non-linear functional. |
| T6 | **Size consistency.** Non-interacting subsystems that satisfy T2 jointly are additive; ones that exchange charge share the Fermi level (correct for metals and redox couples). |
| T7 | **Spin** (§25.6): exact charge and spin constraints both keep F convex. |

### 25.6 Solutions to the four open problems

**(a) Identifiability.** Can the levels be learned from energies and forces?
1. **Charged and open-shell data activate the sector.** Σc = Q_total forces
   free charge onto some units in every ion. Gradients reach E^{(±1)} directly.
   - OMol25 has charges 0 to ±10, multiplicities up to 11, and vertical IE/EA
     evaluation tasks at fixed geometry (arXiv:2505.08762).
   - AIMNet-NSE learned vertical IP, EA and Fukui indices from multi-charge
     data at 2–3 kcal/mol (Zubatyuk et al., Nat. Commun. 2021, ESTABLISHED
     feasibility).
2. **Pairs at fixed geometry** give exact finite-difference targets,
   E(Q+1) − E(Q) = vertical IE, at the cost of two forwards per pair.
3. **Anchors.** Bounded chemical shifts around NIST IE/EA, with the
   electrostatic shift computed, not learned.
4. **The dead zone kills gradients** (T2), like a dead ReLU: a level deep in
   its window never learns.
   - *Remedy:* deterministic annealing of T_e (ESTABLISHED, Rose, Proc. IEEE
     86, 2210 (1998)). Start at kT_e ≈ 0.5–1 eV, where every level carries a
     gradient, and anneal to production (≈ 0.05 eV).
   - Ions provide gradients throughout.
5. **Optional weak supervision** on OMol25's spin densities and population
   charges, weighted low: they are conventions.

**(b) Geometry dependence of redox states** (Au2⁻ binds unlike Au2⁰).
- The learned E_u^{(n)}(G) **are** state-specific local potential surfaces.
  (25.1) interpolates them variationally, and T4 gives population-weighted
  forces. This is the continuous, automatic analogue of EVB diabatic surfaces
  (Warshel & Weiss 1980, ESTABLISHED) and of 4G's charge-dependent short-range
  energy. Unlike 4G it stays variational, so no dq/dR is needed for forces,
  and convex in c.
- **The class preserving T1:** arbitrary in R, convex in c. Pair terms in c
  enter only through Coulomb (W ≻ 0). Charge-state-dependent many-body local
  terms are allowed if they are linear in the populations.

**(c) Spin** (OMol25 needs it).
- **Exact constraints:** piecewise linearity in charge plus constancy along
  fractional spin. These are the flat-plane conditions (Mori-Sánchez, Cohen,
  Yang, PRL 102, 066403 (2009); generalised to ensembles in J. Phys. Chem.
  Lett. 2024, "Ensemble Ground State … Flat-Plane Condition Generalized";
  ESTABLISHED).
- **Lemma 2.** The ensemble E(n↑, n↓) is the lower convex envelope of the
  integer-state energies E^{(N↑, N↓)}; the proof is Lemma 1 in two
  dimensions. The flat plane is a special case and is convex, given
  IE ≥ EA at each integer.
- So Φ_u(c↑, c↓) is (25.1) over spin-resolved states, carrying Hund and
  exchange splittings in the integer points. The global constraints
  Σc↑, Σc↓ come from Q and the multiplicity, read with OMol25's `spin` key
  (SPEC 9 conventions).
- **Outside the convex core, stated honestly:** inter-unit magnetic order
  (antiferromagnetic coupling J_ij s_i s_j) is indefinite, i.e. genuine
  symmetry breaking. It is handled by SPEC 21 states over spin configurations,
  or by the input multiplicity.

**(d) Solver.**
- For T_e > 0, (25.0) is smooth and strictly convex.
  - **Primal-dual Newton** on (x, c, μ_e). Each unit's Φ' is a monotone 1-D
    inverse of the softmax mean, done vectorised.
  - **Newton matrix:** [[K, M], [Mᵀ, J_ff + diag(1/X_u) − λD]] plus the
    constraint row.
  - **Units deep in their dead zone** have X_u ≈ e^{−Δ/kT}/kT, so 1/X_u is
    huge: they decouple. The active set emerges automatically.
  - **Each Newton step** is one PCG solve with the existing Woodbury
    preconditioner (SPEC 17.6), extended by the diagonal. Expect 5–10 steps.
  - **Warm starts** from the previous MD step are legitimate in the converged
    mode only: the minimum is unique, so the answer does not depend on the
    start. They remain forbidden in fixed-k mode (SPEC 2).
- **Derivatives.**
  - Forces by T4: no adjoint.
  - Training gradients by implicit differentiation of the stationarity
    conditions: one adjoint solve with the same matrix. This is the SPEC 17
    machinery, as in OptNet (Amos & Kolter, ICML 2017) and differentiable
    convex layers (Agrawal et al., NeurIPS 2019), ESTABLISHED.
- **T_e → 0** (exact kinks) needs semismooth Newton or active-set methods,
  with soft-thresholding as the proximal map of γ|c|. Not needed for
  production, which uses T_e > 0 as DFT codes use Fermi smearing.
- **Batching:** SPEC 24 applies unchanged.

### 25.7 Tests before any training claim (analytic first)

1. Lemma 1 and 2: Φ^T → convex envelope, including a state above the
   envelope never being populated.
2. T2: random insulating systems give c ≡ 0, and energies, forces and α equal
   SPEC 1 to machine precision at T_e → 0.
3. (25.2): a minimiser unique at λ_max(1 − 10⁻³); two minima reachable just
   above it.
4. (25.3): closed-form two-unit transfer, including the spurious window, as a
   function of λ and σ.
5. Na–Cl: R_c = 9.44 Å from atomic IE/EA.
6. T4: forces against finite differences across an occupation change at
   several T_e.
7. T5: linear response equals X_u numerically.
8. Benchmarks:
   - Au2/MgO: Table II, the Au–O bond split, and the dopant shells
     (with Al units localised by a learned σ_u);
   - regression on NaCl, Ag3, the C10 chain and water: no degradation allowed.

### 25.8 Attention in the free-charge sector (design)

**Principle: attention parameterises, physics minimises.** Every learned
quantity below is a smooth, invariant function of geometry that enters the
convex functional (25.0). The minimisation stays exact (unique minimum,
envelope-theorem forces, implicit derivatives). Explicitly **rejected**:
- learned non-variational update iterations in place of the solve (as in a
  fixed number of field updates): they lose uniqueness, exact forces and the
  physical limits;
- global attention as a substitute for Coulomb: it has the wrong long-range
  physics, and Ewald/PME stays exact;
- feeding solved charges back into local energies non-linearly: that breaks
  convexity and the variational structure. Only terms linear in the
  populations are allowed (§25.6 b).

**A. The ensemble is attention over charge states (an exact identity).**

    p_u(n) = softmax_n( −(Ê_u^{(n)} − ν_u n) / kT_e ),   c_u = Σ_n p_u(n) n,
    Φ_u^T  = conjugate of kT_e · logsumexp_n(·)

The Mermin ensemble of §25.3 *is* a softmax over the unit's charge (and spin)
states (**[erratum]**: the draft had +ν_u n in the softmax; the sign is that of
the corrected (25.1)). The keys are the states, the query is the local electrochemical
potential ν_u = μ_e − u_u, and the temperature is T_e. The free-charge solve is
the self-consistent fixed point of this attention under the global constraint
Σc = Q. Attention as a log-sum-exp energy minimiser is ESTABLISHED (Ramsauer
et al., "Hopfield Networks is All You Need", ICLR 2021; Hoover et al., Energy
Transformer, NeurIPS 2023). Using it as the **physical** ensemble of integer
charge states inside a convex electrostatic functional is NEW-TO-TRACE. In
practice fused softmax kernels serve the sector, and p_u(n) are observable
occupation probabilities.

**B. Charge-state cross-attention for the state surfaces E_u^{(n)}(G)**
(NEW-TO-TRACE).

    E_u^{(n)} = MLP_E( Σ_j softmax_j( q_nᵀ k_j / √d ) v_j ),   q_n = Emb(n, s)

Each charge or spin state is a query token over the unit's environment, whose
keys and values are invariant TRACE features. Geometric knowledge is shared
across states, so a charge state seen rarely in the data borrows from the
others. The construction extends to OMol25's multiplicities up to 11 at cost
O(N · |𝒩_u|).

**C. Attention-Wannier free-charge orbitals** (NEW-TO-TRACE).

    ρ_free(r) = Σ_u c_u Σ_j w_uj g_j(r),   w_uj = softmax_{j∈𝒩(u)}( a(h_u, h_j, r_uj) ),   Σ_j w_uj = 1

Each free charge lives in a learned, normalised spread over its unit's
neighbours, like a Wannier function. This is charge-conserving and linear in
c, so J_ff → Wᵀ J W stays positive semidefinite and convexity is untouched.
It answers two problems at once:
1. **Unit definition.** Au2 shares one orbital; a conjugated chain spreads it;
   a shallow donor learns its effective-mass radius.
2. **Self-interaction.** J_uu = w_uᵀ J w_u falls as the orbital spreads,
   which shrinks the residual R_AB of (25.3).

Related prior art predicts Wannier centres or densities (e.g. eMLP), but those
are not free-charge distributions inside a variational ensemble.

**D. Attention as a hopping Hamiltonian (level 2, later).**
H_uv = h_uv(r) · σ(symmetrised logit), reusing the symmetrised attention
logits that already give κ (SPEC 1) and the SPEC 22 hoppings. Occupations
f(H − μ_e) by Mermin are convex in the density matrix. This supplies the
kinetic delocalisation a classical sector lacks. Linear scaling comes from
Fermi-operator (Chebyshev) expansion (ESTABLISHED: Goedecker, RMP 71, 1085
(1999)). Learning Hamiltonians is ESTABLISHED (SchNOrb, PhiSNet, DeepH);
self-consistent Hartree coupling to a dielectric bound sector is not.

**E. Amortised solver seeds** (speed only). An attention head predicts the
initial (c, μ_e) for Newton. This is legitimate only in the converged mode,
where the unique minimum does not depend on the seed; never in fixed-k mode
(SPEC 2).

One set of equivariant attention logits thus plays four physical roles: bond
hardness (SPEC 1), through-bond hopping (SPEC 22), charge-state energies (B)
and free-charge orbitals (C). The ensemble itself (A) is attention by
construction.

### 25.9 Not done

Implemented in trace_redox v1: §25.2–25.5 with states {−1, 0, +1}, the dual
Newton solve with implicit derivatives (§25.6 d), and 25.8 A–C. Not done:
everything else, as listed in trace_redox/SPEC.md §10. That includes the
removal of the mean-field delocalization error measured there. Open points: a learned tight-binding extension
(hoppings from SPEC 22, giving bands, delocalisation and the Kohn–Sham-like
gap; SCC-DFTB-like, ESTABLISHED: Elstner et al., PRB 58, 7260 (1998)); the
choice of units beyond atoms; how T_e interacts with the electronic entropy in
the reference data.

## 26. A physical bond hardness: Pauli screening and the two-level anchor (`screening.py`, `bond_law.py`, opt-in)

Defect A2 of `docs/ELECTROSTATICS_AUDIT_2026-10-09.md`. A bond with t* ≈ 0
has ∂E*/∂κ_b = ½t*², so energies, forces and charges leave its κ_b wherever
it was initialised. A trained Au₂/MgO model kept its second-shell Mg–Mg bonds
at κ = 0.134 eV/e² and had ε∞(MgO) = 1316 (experiment ≈ 3). By Theorem 1
(below), 1281 of that came from those bonds.

**Theorem 1 (bond-compliance identity; tested in `test_periodic.py`).**
- Statement: if every atom is an inversion centre of the crystal and of its
  bond star, then

      ε∞ = ε∞^dip + (4π K_E / Ω) Σ_b p_b ⊗ p_b / κ_b,   p_b = f_b d_b.

- Reason: uniform-field bond currents leave every atomic charge at zero, so
  the Coulomb kernel drops out.
- Consequence: in a bulk crystal nothing but κ_b bounds ε∞.

**26.1 Pauli screening of the split-charge channel**
(`ElectrostaticConfig.bond_screening = (C_min, C_max)`, recommended
(2.0, 2.8); default `None`).
- The physics: a third atom k between i and j blocks their direct overlap,
  and charge moves through k's own bonds instead.
- The factor is the MEAM / environment-dependent tight-binding one (Baskes;
  Tang, Wang, Chan & Ho, PRB 53, 979 (1996)):

      C_ikj = y² / (x(1 − x))   (x along the bond, y across it, units of r_ij),
      S_ij = Π_k q((C_ikj − C_min)/(C_max − C_min)),   f_b → f_b S_b,

  with q the quintic C² switch.
- Exact values:
  - rocksalt and fcc second shells: S = 0;
  - fcc first shell: S = 1, so metals keep their bonds;
  - a linear chain's 1–3 pair: S = 0.
- Implementation:
  - neighbours are taken from the bonds themselves, as half-edges;
  - each screening atom counts once, through C² weights w(x), 1 − w(x) over
    x ∈ [0.45, 0.55];
  - only relative bond vectors enter, so either `edge_vec` sign convention
    gives the same S.
- Tests (`tests/electrostatics/test_bond_screening.py`, 10):
  - the exact crystal values;
  - agreement with a brute-force sum over every periodic site to 64·eps;
  - invariances;
  - C² smoothness;
  - forces = −dE/dR by finite difference;
  - the rocksalt second shell giving exactly zero dipole rows.

**26.2 The two-level anchor**
(`ElectrostaticHeads(bond_hardness_law="two_level", bond_hardness_range=R)`;
default `"free"`).
- The law: one occupied and one empty orbital across a bond give, exactly,
  κ = G³/(4h²) with G = √(Δ² + 4h²) (2026-10-09 band note, Theorem 6).
- The anchor:

      κ_anchor = (Δχ² + 4h²)^{3/2} / (4h²),   Δχ = χ_i − χ_j (Mulliken, NIST),   h = 1.42 ħ²/(m_e r²)
      κ = κ_anchor · exp(L tanh(s/L)) + κ_floor,   L = ln R

  The coefficient 1.42 is Froyen & Harrison's universal sp-sigma value. The
  SPEC 23 contact law multiplies on top, beyond contact.
- **Certificate.** κ ≥ κ_anchor/R for every learned value, so with
  screening Theorem 1 bounds ε∞ whatever training does.
- Measured on rocksalt MgO with untrained heads (3 seeds):

  | law | screening | ε∞ |
  |---|---|---|
  | free | off | 454–596 |
  | free | on | 100–117 |
  | two-level (R = 10) | on | 10.6–12.6 |

  The anchor's Mg–O hardness is 10.3 eV/e²; experiment is ε∞ ≈ 3.
- Tests (`tests/electrostatics/test_bond_law.py`, 12):
  - the exact two-orbital response;
  - anchor values and symmetry;
  - κ = 2h for identical atoms;
  - exact gradients;
  - the default head unchanged bit for bit;
  - the bounds;
  - the ε∞ certificate.

