# TRACE-Redox — physics contract

`trace_redox` implements the free-charge sector proposed in
`trace_electrostatics/SPEC.md` §25. It is a separate package: it calls the
bound (SPEC 1) sector through its public assembly and changes nothing in
`trace_electrostatics`, so earlier results stay reproducible and comparable.

Units: eV, Å, e; `K_E = 14.399645` eV·Å/e². Float64 only.

Every term below is marked **ESTABLISHED** (with a citation) or
**NEW-TO-TRACE**. "Measured" means run in this repository; the scripts are
named.

---

## 0. The model in three equations (formal anchors: every current model; 2026-10-19)

A unit u (an atom; a covalent unit, §19, or a band unit, §20, where configured) carries an integer charge n_u
(electrons removed). It has a frame f_u and a carrier Q_u = n_u − f_u. A configuration n has the energy

    E(n) = Σ_u ε_u(n_u)  +  ½ q(n)ᵀ J q(n)  −  ½ s(n)ᵀ K(n)⁻¹ s(n)  +  Σ_{u<v} O_uv(n_u, n_v)        (0.1)
              ions             Coulomb             induction                  overlap

and the configurations mix through one-electron hops:

    H_nn = E(n),     H_nm = ± σ_nm f_nm τ_uv     (m = n with one electron moved u → v)            (0.2)

    E_total = E_short(R) + F,     F = −kT ln Tr exp(−H / kT)                                      (0.3)

- **Ions.** ε_u(n) = E_Z(n) + g_u Δ_u(n) − ½ S₀,u n².
  - E_Z is the exact isolated ion (NIST IE, EA).
  - Δ_u is a learned shift, |Δ| ≤ χ, gated to zero without neighbours (g_u, §24).
  - The isolated Gaussian self energy S₀ is removed, so the next terms add only interactions. Under periodic
    boundaries an ion's interaction with its own lattice images is lattice energy and stays in J.
- **Coulomb.** q(n) are the charges on Gaussians of width σ_f, and J is their Coulomb matrix (S₀ on the diagonal).
- **Induction.** s(n) = g + Aᵀq(n) + Σ_u Q_u v_u(n_u) is the field the configuration exerts on the bound sector
  (SQE bond charge flows and induced moments). A holds the Gaussian charge–moment couplings, and v_u corrects them
  to the carrier's true density (§26). K(n) is the bound sector's stiffness, with each state's own polarisability
  (§28), and K(n) ≻ 0 is checked exactly. The bound sector is eliminated exactly: at K(n) = K and v = 0 this term
  and the Coulomb term are E_b0 + bᵀn + ½ nᵀWn with W = J − A K⁻¹ Aᵀ, as the code holds them. §26 adds its exact
  one- and two-body terms, and §28 its Woodbury correction.
- **Overlap.** O_uv is the two charge-state densities' exact Coulomb energy minus the Gaussian one (penetration,
  §23), plus their Gordon–Kim Pauli energy (§25). It is tabulated from first-principles ion densities. An atom and
  its own periodic images enter as one-body terms (§23.x).
- **Hops.** τ_uv are learned one-electron couplings (§17.6), or Harrison-anchored ones (§31); the O–s channel is
  folded into E(n) instead (§32). σ is the importance switch (§21), and f = √2 for a
  move that pairs or unpairs frontier electrons (§29).

**The equation as of 2026-11-07 (§32–35 added).** Three terms enter (0.1) and its induction field:

    E(n) = Σ_u ε_u(n_u) + ½ qᵀJq − ½ s(n)ᵀ K(n)⁻¹ s(n) + Σ_{u<v} O_uv + Σ_C λ_C(n) + E_bp(R)              (0.1′)
                                                                      fold (§32)   bond-oriented Pauli (§33)

    s(n) = g + Aᵀ q(n) + ΔAᵀ f + Σ_u Q_u v_u(n_u),   ΔA[i, μ_j] = (s(R_ij) − 1) A[i, μ_j]  (§34)

- **§34**: the frame charges f drive their nearest-neighbour counter-ions' induced dipoles with the overlap-damped
  coupling s(R) = 0.61 + 0.37 (R − 2.128 Å) (Mg²⁺–O²⁻, measured on the embedded ALMO pair); the carriers keep A (+ §26).
- **§35**: in K(n) the frame ions' dipole stiffness is A_μ,i = K_E ℓ_i² / (g α_Z(f_i)) with the first-principles α_Z
  and one learned crystal factor g; §28's ratios give each charge state's own α on top.
- **§33**: E_bp = Σ_a Σ_j λ_a s_c(R)[D₁(R) cos θ + D₂(R) P₂(cos θ)] (+ the wall Σ s_c I(R)), configuration independent.
- **§32**: λ_C the lowest root of the O–s block of cluster C; off in every current run (steps 22–24 failed).

**The equation as of 2026-11-11 (§28.6, §36–38 added: extensivity).** (0.1)–(0.3) with one Hamiltonian over the
whole structure, at one cell, is not extensive: replicating a periodic cell changed the energy per cell. Measured
on frozen step 31 s43 weights, Au₂/MgO(001): as shipped, wet_u moved by +4.8 meV (2×1) and +95.4 meV (2×2); with
§28.6 alone, undoped E(2×2)/4 − E(1×1) = +220.84 / +99.83 meV (wetting / non-wetting); the doped 2×1 cell was off
by +5.28 / +7.05 eV per cell and the doped 2×2 cell was refused. Four changes, each exact in a stated sense:

    K(n):   Δ_j = A_μ,j (1/ρ_j(n_j) − 1)                                                   §28.6 (default)
    F     = E⁰ + Σ_c (F_c − E⁰) + Σ_{c<c′} (F_cc′ − F_c − F_c′ + E⁰)                      (0.3′)  §36
    E_M(C) = E_Γ(S_M[C]) / |M|                                                            (0.4)   §37
    E(n, N) = E_at(n) + F_b(N) + Σ_v ρ_N(v) c_v(n)                                        (0.1″)  §38

- **§28.6**: a charge state changes its atom's own dipole stiffness A_μ,j, not the diagonal block of the periodic
  K, which contains the atom's images. The default since 2026-11-10; `'block'` is the legacy statement, restored
  for checkpoints saved before it.
- **§36** (`diabatic_space='clusters'`): F from the reference configuration n⁰, the clusters c of linked moves and
  their pairs. Separated clusters add exactly; the configuration count is linear in the cell.
- **§37** (`bvk_mesh` / `bvk_length`): a cell is evaluated as its Born–von Kármán supercell S_M, so a cluster's
  fluctuation is no longer in phase with its own images.
- **§38** (`band_form='fermi'`): the band unit holds N electrons in thermal Fermi–Dirac occupations of h, with the
  free energy F_b(N) and the charge shape ρ_N; §20's two-electron φ₁ band is its gapped N ≤ 2 limit.
- **Measured with all four on** (frozen s43, `bvk_length = 18.0` Å): E(2×2)/4 − E(1×1) = +1e-6 / < 1e-6 meV
  (undoped wetting / non-wetting) and +3e-6 / +7e-6 meV (doped); max|ΔF| ≤ 6.5e-8 eV/Å.
- **Not exact** (stated limits): (0.3′) drops the connected terms of three or more clusters and steps where n⁰
  changes on three or more clusters (§36.8 L1, L1′: +0.867 meV per cell on the doped non-wetting cell at a
  (2, 2, 1) mesh); the closure of (0.3′) is exponential inside a merged cluster and refuses beyond its cap
  (§36.8 L2); (0.4) is measured to four cells, and for the undoped cells at (3, 3, 1), the reference's sampling (refused when
  this was written; it runs since the route-aware operator guard of 2026-11-12, CHANGELOG Round 39; §37 limits); the band's coupling amplitudes to the other units are zero in §38.
- **Extensive is not the same as trained.** When this was written (2026-11-11) no model had been trained with any of
  the four. **[Updated 2026-11-16, from the run configurations:]** steps 33 and 34b trained with §28.6, §36 and §38,
  at Γ on one cell (no `bvk_mesh`, no `bvk_length`); no model has been trained with §37. Weights trained at Γ
  on one cell do not fix how a supercell's band fills: at the (2, 2, 1) mesh the two step 31 seeds give
  wet_d = −22.95 and +603.69 meV (§38.3, §38.4). No doped number at a mesh is claimed.

In words: exact ions, the classical physics of their true densities, and learned hops between integer
configurations.
- ESTABLISHED: the Gordon–Kim overlap model (Gordon and Kim, J. Chem. Phys. 56, 3122 (1972)) and polarisable-ion
  models (Madden and Wilson, Chem. Soc. Rev. 25, 339 (1996)).
- NEW-TO-TRACE: evaluating them per integer configuration with first-principles charge-state densities, inside a
  canonical ensemble mixed by learned hops.
- Why the overlap terms cannot be dropped: classical point or Gaussian ions put A⁺A⁻ below A⁰A⁰ at the bond
  length of most covalent bonds (Proposition 8, theory note 2026-10-18 §4c; measured in the AuMgO, NaCl and Ag₃
  models).

**Tested as one equation** (`test_redox_master_equation.py`, 4).
- The module never forms (0.1). It splits it into the ladder, b and W, the pair slot and the Woodbury correction.
- The test evaluates (0.1) directly, with a dense K(n)⁻¹ and a dense s(n), and checks that the module's diagonal
  equals it to 1e-10: three stacks in an open cluster and the full stack in rock-salt MgO (Ewald blocks,
  self-image terms).
- Leaving out v, K(n) or O breaks it by 0.51, 0.62 and 2.26 eV, and the self-image terms by ~12 meV in the
  periodic cell.

**Where every section sits**

| section | in (0.1)–(0.3) | status (2026-10-19; later entries dated; the training record checked against the run configurations on 2026-11-16) |
|---|---|---|
| §17.5 ladder, §17.6 couplings | ε (E_Z), τ | core |
| §24 shift gate | g_u | core: exact isolated ions |
| §23 penetration, §25 Gordon–Kim | O_uv | core (Proposition 8); §25 replaces §18 |
| §22 bond partition | the structure of K and A (no free–bound double counting) | under test, step 14; on in steps 33 and 34b |
| §26 carrier induction | v_u in s(n) | under test, step 16; on in steps 33 and 34b |
| §28 state polarisability | K(n) | under test, step 17; on in steps 33 and 34b (`state_polarizability_elements` [79, 8]) |
| §29 spin pairing | f | under test, step 13S (trained on both seeds; its N1–N4 verdict is not written in the benchmark log); trained again in step 34b |
| §21 amplitude selection | which n enter H; σ | numerics |
| §19 covalent units, §20 band unit | what a unit is | per system |
| §27 coupling bound | τ ≤ B(r) | off in every run through step 31. **Replaced 2026-11-12 (README R26):** "the identity on every benchmark structure" holds for step 31's s43 (largest \|τ\|/B 0.72) and not for s42 (O–O couplings up to +17.7 eV, 6.5 B, on 576 edges of every benchmark cell). **Trained since:** step 32 (seed 42 only; the O–O coupling sits at 0.973 B, README "Step 32 scored"), steps 33 and 34b |
| §31 anchored couplings | τ = η ħ²/(m r²) × factor (½, 2), all but frame cation-anion pairs | step 21: failed (S1–S4, S6) |
| §32 downfolded O–s bond | E(n) += Σ_clusters λ(n), λ the block root of [[0, vᵀ],[v, Δ + T_QQ]], V_spσ fixed | step 22 (under test) |
| §33 bond-oriented Pauli | E += Σ_a Σ_j λ_a s_c [D₁(R) cos θ + D₂(R) P₂(cos θ)], a frame term | step 25 (pre-registered 2026-11-05); on in steps 33 and 34b (`bond_pauli_level: almo`, with the wall) |
| §34 frame-pair overlap damping of induction | the frame's drive on its counter-ions' bound dipoles × s(R) = 0.61 + 0.37 (R − 2.128 Å): y0 − K⁻¹v, u0 − A K⁻¹v, constant + v·y0 − ½ v·K⁻¹v | frozen on step 20: Z*(Mg) 2.43 → 1.95 at ε∞ unchanged (2026-11-07); step 28 (to be pre-registered); trained in step 28b and in steps 33 and 34b |
| §35 frame-polarizability anchor | A_μ,i = K_E ℓ_i² / (g α_Z(f_i)) on tabulated frame ions, g learned (init 1) | step 28a (anchor) / 28b (anchor + §34), pre-registered 2026-11-07 10:20; 28b trained on the GPU host (28a cancelled in the queue, 2026-11-07); 28b doubled the undoped test force RMSE (0.155 / 0.163 against step 20's 0.073 / 0.069 eV/Å; benchmarks README); trained again in steps 33 and 34b |
| §28.6 on-site stiffness correction | K(n): Δ_j = A_μ,j (1/ρ_j − 1) | default since 2026-11-10 (X1); `'block'` legacy; nothing trained with it as of 2026-11-11; trained in steps 33 and 34b |
| §36 cluster product ensemble (§36.7 closure, §36.9 no percolation enumeration, §36.10 response record) | how F is taken in (0.3): (0.3′) | option `diabatic_space='clusters'` (P1–P5 pass; limits §36.8); default `'global'`; trained in steps 33 and 34b (one cell at Γ) |
| §37 Born–von Kármán sampling | the cell (0.3′) is evaluated on: (0.4) | option `bvk_mesh` / `bvk_length`; default the unit mesh; undoped sequence measured to four cells (§37 limits); (3,3,1) refused as first written, runs since the route-aware operator guard (2026-11-12, CHANGELOG Round 39); no model trained with it |
| §38 Fermi band | what the band unit is: (0.1″), F_b(N), ρ_N | option `band_form='fermi'`; default §20's `'phi1'`; C4: U1, U3 pass, U2 fails as written (§36.8 L1′ and §33's table knots); trained in steps 33 and 34b (one cell at Γ) |
| §39 frame pair's own interaction | E += Σ_pairs Φ_ab(R), (39.1): the frame entry of §23 and §25, every library pair | option `frame_pair` (2026-11-13); built, reviewed and tested; nothing trained with it until step 34b (2026-11-15/16, cold start, both seeds: FAILS as pre-registered on S2, S5 and S8, passes S1, S3, S4 and S7; README "Step 34b scored"); ten pairs with first-principles points; its other half is §40 |
| §40 two-site sector of two s¹ frames | O_ab(f_a, f_b) = −ρ_ab(R) (40.2) in the pair slot of (0.2); τ_ab = −t_ab(R) w(R) (40.3) in place of the learned coupling of that pair | option `two_site` (2026-11-13; reviewed 2026-11-14); built and tested; trained in step 34b with §39 and §29 (see §39's row); with §40.7 (O_ab(∓1, 0) = −ρ_∓, O_ab(+1, −1) = +λ: the isolated pair first-principles in three charge states) applied to Au–Au (Ag–Ag, H–H tabulated only); the wall mechanism and the pair's EA, IP, U pass frozen (§40.5, §40.7); trained by the decision of 2026-11-15 (step 34, stopped in epoch 2 by a cold-start band state; then step 34b with `band_initial_factor` 0.55) |
| §30 response record | the training priors' α, ε∞ | training only, not part of the model |
| §17.9 core + carrier, §18 fitted exclusion, §18.5 exclusion pairs | none | legacy, see below |
| §1–16 | the mean-field functional (continuous charges) | the mean-field anchors, not used by the formal models |

**Legacy paths are kept only to reproduce recorded runs.** Audit, 2026-10-19:
- §18 is used by 23 runs: the four running trainings (13S, step 14) and the NaCl, Ag₃ and carbon-chain benchmark
  models.
- Window selection is used by 22 runs, including those three benchmark models.
- §18.5 is used by 2 runs (step 11, negative), §17.9 by 1 (step 7, negative).
- None is deleted while a recorded result depends on it and `trace_redox/` is uncommitted.

**The minimal model** is (0.1)–(0.3) with v = 0, K(n) = K and f = 1. Each of §22, §26, §28 and §29 stays only if
its pre-registered step moves a benchmark number.

## 1. The functional (mean-field anchors)

    F(R) = E_b0 + min_{c : Σ_u c_u = Q} [ Σ_u Φ_u(c_u) + bᵀc + ½ cᵀ W_eff c ]          (1)

| symbol | meaning |
|---|---|
| `c_u` | free charge of unit u (electrons removed; e) |
| `Q` | total charge of the structure (`data['total_charge']`) |
| `E_b0` | minimum of the SPEC 1 bound sector with every fragment neutral (ρ = 0) |
| `b = Ω u0` | potential of that bound solution at the free charges, plus `−r·E` for an applied field |
| `W_eff = Ω W Ωᵀ − diag(r)` | screened free–free Coulomb operator, minus the certified self-interaction removal r (§8a, default) or λJ_uu (§7) |
| `Ω` | attention-Wannier orbitals (§4); `Ω = I` by default (units = atoms) |
| `Φ_u` | Mermin ensemble of unit u over its charge states (§3) |

**NEW-TO-TRACE:** a convex, variational free-charge sector with the
derivative discontinuity, equilibrated globally by one electrochemical
potential and screened exactly by the dielectric bound sector.

**ESTABLISHED ingredients:**
- the bound-plus-integer decomposition `Q_i = n_i + Σ_j q_ij` (redoxSQE:
  Dapp & Müser, JCP 139, 064106 (2013));
- the Mermin ensemble (Phys. Rev. 137, A1441 (1965));
- PPLB piecewise linearity (Perdew, Parr, Levy, Balduz, PRL 49, 1691 (1982)).

## 2. The bound sector, eliminated exactly (`bound.py`)

The SPEC 1 energy at ρ = 0 is exactly quadratic in `y = (t, μ̂[, Θ̂])`:

    U_b(y) = E0 + gᵀy + ½ yᵀKy

- `K` is the assembly's own `dense_hessian()` restricted to the variational
  rows. Its asymmetry is measured and must be round-off; measured ≤ 1e-13.
- Since 2026-10-08 the default never forms K: §13 inverts it exactly through
  its SQE structure (`bound_elimination='condensed'`). The dense form here is
  the reference route (`'dense'`).
- `g` and `E0` come by autograd on `assembly.energy`.

**The free charge has its own Gaussian.** Each free charge `f_i` is a normalised
Gaussian of its own width `σ_f,i` (§5). It couples to the bound sector by
Coulomb's law only, never through χ or η, which would double count IE and EA:

    f ᵀA y + ½ fᵀJ f + fᵀh,     A = [J_fb B, T01_f, T02_f]
    J_fb[i,j] = K_E erf(r_ij/s)/r_ij,  s = √(σ_f,i² + σ_j²);   J_fb[i,i] = 2K_E/(√π s_ii)
    J[i,j]    = same with s = √(σ_f,i² + σ_f,j²);               J[i,i]   = √2 K_E/(√π σ_f,i)
    T01_f, T02_f: trace_electrostatics' Gaussian charge–dipole / charge–quadrupole
                  tensors with the FREE width on the charge index;   h = −r·E

**Minimising over y in closed form:**

    y*(f) = y0 − K⁻¹Aᵀf,  y0 = −K⁻¹g
    E_b(f) = E_b0 + u0ᵀf + ½ fᵀWf
    E_b0 = E0 + ½gᵀy0,  u0 = h + Ay0,  W = J − AK⁻¹Aᵀ

**Why W ≻ 0.** The joint form in `(y, f)` is a Gram matrix of distinct
Gaussian multipole densities plus the PSD on-site floors of SPEC 12 / 17.3.
Gaussians of different widths on one centre are linearly independent. A Schur
complement of a positive definite matrix is positive definite. Both Cholesky
factorisations check this and refuse otherwise.

**Same-width check.** With `σ_f = σ` the free blocks are exactly the assembly's
own (`J = j_pair + diag S0`, `T01_f = t01_pair`, `T02_f = t02_pair`), and a free
charge is the assembly's ρ minus its χ and (η − S0) terms. Both identities
are tested (`test_redox_bound.py`).

## 2a. Periodic boundaries (`bound.periodic_free_charge_blocks`)

Every block that involves a free charge is a Gram matrix of normalised
Gaussian multipoles on the lattice. In the plane-wave convention of
trace_electrostatics.lattice (tinfoil boundaries, k = 0 removed, i.e. a
neutralising background):

    J_ij    = (4πK_E/V) Σ_{k≠0} g_i g_j cos(k·r_ij)/k²,        g_i = e^{−k²σ_i²/4}
    T01_ija = (4πK_E/V) Σ_k g^f_i g^μ_j k_a sin(k·r_ij)/k² · ℓ_j
    T02_ijm = −½(4πK_E/V) Σ_k g^f_i g^Θ_j (k·W_m·k) cos(k·r_ij)/k² · ℓ_j²

**Why this form.**
- The Gaussian factor separates per site, so there is no Ewald α and any
  mixture of widths is exact up to the k-cutoff,
  k_max = 12/s_min (tail ≈ erfc(6)).
- Every block involves at least one free Gaussian, so s_min ≥ σ_f,min
  (≈ 1.6–1.9 Å for typical elements at h = 0.5). That gives about
  12³/(6π² s_min³) ≈ 5–8 reciprocal vectors per Å³ of cell.

**Slab.** A 2-D slab adds the width-independent Yeh–Berkowitz terms
(4πK_E/V) z_i z_j to the charge blocks and (4πK_E/V) z_i ℓ_j n to T01, with
the assembly's own normal and volume.

**Checks:**
- With σ_f = σ, the blocks equal the assembly's own Ewald blocks to
  4e-15–1.7e-14 absolute: 3-D triclinic and slab, quadrupole block included.
- The large-cell limit reproduces the open-boundary energy, with the residual
  falling exactly as L⁻⁵ (ratios 7.5, 4.2, 7.6 against 1.5⁵ and (4/3)⁵).
- The de Leeuw–Perram–Smith surface term +2πK_E d²/(3V) (tinfoil → vacuum) is
  verified with its sign.
- Forces, the strain derivative (stress) and lattice-translation invariance
  match finite differences; symmetry-equivalent rock-salt ions carry equal
  charges.

**Refused under PBC:**
- an applied field, since −Σc r·E is branch dependent for a free charge;
- a slab with net free charge, which needs the charged-slab terms of
  Ballenegger, Arnold & Cerdà (JCP 131, 094107 (2009)), absent from the bound
  sector too.

A 3-D cell with net charge uses the neutralising background, as the bound
sector does.

## 3. The ensemble of one unit (`ensemble.py`)

States `n ∈ {−1, 0, +1}` (electrons removed). With self-Coulomb-removed state
energies `Ê_un`:

    Ω_u(ν) = −kT log Σ_n exp(−(Ê_un − νn)/kT)                   grand potential
    N_u(ν) = Σ_n p_un n,  p_un = softmax_n(−(Ê_un − νn)/kT)        mean charge
    X_u(ν) = dN/dν = Var_p(n)/kT ≥ 0                             susceptibility
    Φ_u(c) = sup_ν [νc + Ω_u(ν)] = νc + Ω_u(ν)|_{N(ν)=c}          Legendre conjugate

Properties:
- `Φ_u` is convex and C^∞ for kT > 0, with `Φ' = ν` and `Φ'' = 1/X`.
- **Lemma 1:** `env(c) − kT ln 3 ≤ Φ_T(c) ≤ env(c)`, where `env` is the lower
  convex envelope of the points `(n, Ê_un)` (the entropy term lies in
  `[−kT ln S, 0]`). So `Φ_T → env` as T → 0. A state above the envelope has
  population ∝ `e^{−D/kT}` (tested).

**Erratum to SPEC 25.3 (25.1).** The draft wrote
`sup_ν[νc + kT ln Σ e^{−(Ê+νn)/kT}]`. That supremum is +∞: the bracket is
convex in ν. The correct conjugate is the one above. With the sign convention
used here, `ν_u = μ_e − b_u − (W_λ c)_u`.

## 4. Units and attention-Wannier orbitals (`heads.OrbitalHead`, opt-in) — NEW-TO-TRACE

    Ω_uj = (1 − s_u) δ_uj + s_u w_uj,   w_u· = softmax over {u} ∪ N(u),   0 < s_u ≤ s_max < ½

- Rows sum to one, so free charge is conserved: `Σ_j f_j = Σ_u c_u`.
- The weight leaving a unit is at most `s_max`. Ω is invertible:
  `‖diag(1−s)⁻¹(Ω − diag(1−s))‖_∞ ≤ s/(1−s) < 1`. Hence `Ω W Ωᵀ ≻ 0`.
- Each neighbour's exponential is multiplied by the C² switch
  `f(x) = 1 − 10x³ + 15x⁴ − 6x⁵`, with `x = r/r_cut`, inside the softmax
  normalisation. A neighbour therefore enters and leaves with zero weight,
  slope and curvature. Tested by the second difference and by the forces
  across `r_cut`.

## 5. State energies, anchors and the free-charge width — NEW-TO-TRACE parameterisation

**The physical constraint.** `Ê_u` must be convex on the integers:
`Ê_+ + Ê_− − 2Ê_0 > 0`. Otherwise a neutral unit lowers its energy by mixing
`n = ±1` at zero mean, i.e. it disproportionates internally. This was measured
in a first draft that put the free charge on the bound σ ≈ 1.5 Å, where
`J = 7.66` eV exceeds `IE − EA` for Na (4.59 eV), Al, Si, Cu, Au and Li. The
draft was rejected for that reason. The parameterisation used now guarantees
convexity for every parameter value:

    Ê_+1 = χ_u + G_u/2,   Ê_0 = 0,   Ê_−1 = −χ_u + G_u/2
    χ_u = χ_Z + L_χ tanh(d_χ/L_χ),       G_u = G_Z exp(L_G tanh(d_G/L_G)) > 0
    E_un = Ê_un + ½(1 − λ) J_uu n²        (the physical integer-state energies)

**Shifts.** `d_χ = ½(r₊ − r₋)` and `d_G = ½(r₊ + r₋)` come from charge-state
cross-attention (`ChargeStateHead`). Each charged state is a query token over
the unit's neighbourhood. The last layer is zero, so all shifts vanish at
initialisation.

**Anchors** (`anchor='atomic'`):
- `χ_Z = (IE_Z + EA_Z)/2` (Mulliken, ESTABLISHED) and
  `G_Z = U_Z − (1 − λ)J_Z`, with `U_Z = IE_Z − EA_Z` and `J_Z = S0(σ_f)` the
  atom's own free self-energy.
- An isolated ion is then exact: `E_+1 = IE_Z` and `E_−1 = −EA_Z`. Tested to
  1e-12 for Na⁺, Cl⁻, Au⁺ and O⁻.
- IE and EA are NIST values via pymatgen (`constants.py`; all 96 entries
  re-checked against pymatgen, 0 mismatches).

`anchor='gap'` (`χ_Z = 0`, `G_Z = 2·gap`) is the SPEC 1 limit used by the
tests.

**Free-charge width** (`FreeChargeWidth`):

    σ_f,Z = σ_H,Z / h_Z,   σ_H,Z = √2 K_E (1 − λ) / (√π U_Z),   h_Z ∈ (0.05, 0.95)

- `h_Z` is learned per element; the initial value is `hartree_fraction`, default 0.5.
- `σ_H` is the width at which `(1 − λ)S0(σ) = U_Z`. That is QEq's
  hardness–size relation (Rappé & Goddard, J. Phys. Chem. 95, 3358 (1991):
  `J_AA = IE − EA`, ESTABLISHED).
- At `h = 1` the state energies are linear in n and the sector is exactly QEq:
  gapless and metallic.
- `h` is the fraction of the atomic hardness that is classical Hartree energy.
  The integer gap `G_Z = (1 − h)U_Z > 0` is the remainder.
- Example widths at h = 0.5 (Å): H 1.79, C 2.30, O 1.89, Cl 2.46, Au 3.32, Na 5.0.

## 5a. Charge states beyond +1 (`RedoxConfig.max_charge = K`, 1 ≤ K ≤ 4)

**States and anchors.** The states are n ∈ {−1, 0, 1, …, K}, anchored to NIST
successive ionisation energies (pymatgen; 0 mismatches over Z = 1–96, k ≤ 4):

    E_+k = IE_1 + … + IE_k,   E_−1 = −EA
    levels   ε_k = Ê_k − Ê_{k−1}:   ε_0 = χ − g_1/2,   ε_1 = χ + g_1/2,   ε_k = ε_{k−1} + g_k  (k ≥ 2)
    gaps     g_1 = (U − (1−λ)J) e^{x_1},   g_k = (IE_k − IE_{k−1} − (1−λ)J) e^{x_k},   |x_k| ≤ L_G

**Properties.**
- Every gap is positive, so Ê is strictly convex on the integers for every
  parameter value.
- An isolated ion has E_+k = Σ IE_j exactly at initialisation, tested for
  Mg²⁺, Al³⁺, Fe³⁺ and He²⁺ to 1e-11.
- An anchor gap g_k ≤ 0 is refused.
- K = 1 is the earlier model, and its random initialisation is
  byte-identical.

**Absent states.** A state an atom cannot reach (k > Z) is absent: its energy
is +∞, it is never populated, and its gradient is zero (a finite placeholder
avoids the NaN of `where`). The capacity is Σ_u min(K, Z_u). A charge at the
capacity is saturated, i.e. evaluated exactly; one beyond it is refused.

**Measured** (`experiments/redox_v1/measure_multistate.py`, untrained heads),
M(H₂O)₄²⁺ with the free charge on the metal:

| M | K = 1 | K = 2 |
|---|---|---|
| Mg | +1 (the other +1 forced onto the ligand H) | +2.000, 1.34 eV lower |
| Fe | +1 | +1.64 |
| Zn | +1 | +1 |

For Zn, IE₂ (17.96 eV) exceeds the cost of ionising a water ligand until the
bound sector's ion–dipole solvation is trained.

## 6. The solve (`free_solver.py`)

**The dual.** Fenchel duality, with `Φ* = −Ω` and
`½cᵀWc = max_z(zᵀWc − ½zᵀWz)`, gives

    D(z, μ) = −½ zᵀW_λ z + μQ + Σ_u Ω_u(μ − b_u − (W_λ z)_u)

- `D` is concave and smooth on all of R^{U+1}.
- Stationarity is `z = N`, `ΣN = Q`; at the optimum `z* = c*` and `μ* = μ_e`.
- Strong duality holds (Slater); the duality gap is reported. Measured
  |gap| ≤ 1.6e-13 eV on the 6-atom smoke runs; tests require
  ≤ 1e-10 (1 + |F|).

**Newton with Armijo backtracking.** The step comes from the Schur complement
on μ:

    A = W + W X W ≻ 0,  a = A⁻¹g_z,  v = A⁻¹Wx,  s = 1ᵀx − (Wx)ᵀv
    dμ = (g_μ + (Wx)ᵀa)/s,  dz = a + v dμ

**Two numerical regimes, both measured before the fix:**
- *Pure-Newton phase.* When the Newton decrement falls below the resolution of
  D (`1e4·eps·(1+|D|)`), the full step is taken. Before this fix, Armijo
  compared round-off at slope 9e-21 and stalled at `|N − z| = 7e-11`
  (Boyd & Vandenberghe §9.5).
- *Flat μ.* When `s ≤ 1e4·eps·U·max|n|/kT`, every unit is saturated and `X`
  has underflowed (gaps of 200–400 eV at kT = 1e-3 gave `X = 0` exactly and
  NaN). μ is then held fixed, and re-bracketed by bisection if `ΣN ≠ Q`. The
  coupling dropped this way is O(s).

**Implicit derivatives.** The iterations run detached. One Newton step with
the autograd graph, taken from the optimum, makes the returned `z`, `μ`, `p`
and `c` carry `−H⁻¹ ∂(∇D)/∂θ`. Because D is stationary there, first and second
derivatives of F are exact. Tested against the closed-form linear response
(T5)

    dc/db = −(M⁻¹ − M⁻¹11ᵀM⁻¹ / 1ᵀM⁻¹1),  M = W + diag(1/X)

and against central differences of a force loss in a network weight. The same
construction is the converged solve of SPEC 17 (ESTABLISHED: OptNet, Amos &
Kolter 2017).

**Saturated total charge.** For `|Q| = U` every unit sits in its extreme state
and `F` is evaluated directly; μ_e is reported as NaN.

## 7. Residual self-interaction λ (SPEC 25.4)

`½λJ{c}(1−{c}) = ½λJ(interp(n²)(c) − c²)`. The piecewise-linear part enters
through `E_un`; the concave part enters `W_λ`. This equals SPEC 25.3's term
whenever the envelope passes through every integer point, which the
parameterisation of §5 guarantees. Convexity requires `λ < λ_max =
λ_min(D^{−½} W D^{−½})`. This is checked, and the solve is refused otherwise.

**Measured λ_max** (`experiments/redox_v1/measure_physics.py`, untrained bound
heads):

| geometry | h = 0.5 | h = 0.8 |
|---|---|---|
| NaCl dimer, 2.36 Å | 0.135 | 0.144 |
| (NaCl)₄ rock-salt cube, a = 2.82 Å | 0.037 | 0.076 |
| water dimer, R_OO = 2.91 Å | 0.0074 | 0.025 |
| benzene | 4.6e-5 | 0.0010 |

Benzene's `J` has eigenvalues from 2.5e-4 to 48 eV/e²: diffuse Gaussians on a
1.4 Å ring are nearly linearly dependent. **Consequence:** a constant λ
large enough to matter is not convex in condensed molecules. SPEC 25.4's
design note "hybrids → λ → λ_max" is not practical, so λ = 0. Its
per-geometry, per-unit successor is the certified removal of §8a.

## 8. The delocalization error of v1 (measured; SPEC 25.4 made quantitative)

*This section describes `sic='none'` (v1). The default since 2026-10-03,
`sic='certified'` (§8a), removes this error for distant atomic units.*

**The law.** For two distant units at λ = 0 and T → 0 (W_AB → 0, W_uu → J_uu),
the transfer A → B is fractional iff

    IE_A − EA_B < ½ (h_A U_A + h_B U_B)                  (spurious-transfer window)

with `δ* = ½ − (IE_A − EA_B)/(h_A U_A + h_B U_B)`.

**Na···Cl, Q = 0.** Free charge on Na, from `measure_physics.py` (law at
R → ∞: 0.281 e for h = 0.5, 0 for h = 0.2):

| | R = 2.36 | 4 | 6 | 9.43 | 15 | 30 | 100 Å |
|---|---|---|---|---|---|---|---|
| h = 0.5, kT = 0.01 | 1.000 | 0.971 | 0.703 | 0.494 | 0.390 | 0.328 | 0.296 |
| h = 0.2, kT = 0.01 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |

- **No single h works.** A compact free charge gives the ionic bond at contact
  but keeps 0.3 e at infinite separation. A diffuse one removes the spurious
  transfer, and the ionic bond with it, because diffuse clouds barely attract.
- **Same physics as semilocal DFT.** This is the fractional-charge
  dissociation error of semilocal functionals (Ruzsinszky, Perdew, Csonka,
  Vydrov, Scuseria, JCP 125, 194112 (2006), ESTABLISHED). Its root is that the
  mean-field Coulomb `½cᵀWc` of the average charge misses the pair
  anti-correlation of the integer configurations. The exact ensemble Coulomb
  for an A⁺B⁻ / A⁰B⁰ mixture adds back `½J̃ δ(1−δ)` and is linear in δ.
- **Pairs inside the window at h = 0.5** (closed form, anchors only, no learned
  shifts): Na→Cl (δ* = 0.281), Li→F (0.288), Na→O (0.061). Outside it: C→O,
  H→H, Au→Au, Au→O, Mg→O, and Al→Au (by 0.56 eV).
- **Physical reading.** A convex mean-field free sector cannot be both exact
  for distant ion pairs and ionic at contact. The exact joint-configuration
  ensemble is also convex (a linear energy plus a convex entropy over the
  joint distribution), so exactness and convexity are not in conflict; the
  product (mean-field) approximation is what fails. This is stated as the
  open problem in §10.

## 8a. Certified self-interaction removal — the default (`sic.py`) — NEW-TO-TRACE

**Idea.** Move an amount r_u of the on-site Coulomb out of the quadratic form
and put it back exactly, in the populations:

    ½ cᵀWc  →  ½ cᵀ(W − diag r)c + ½ Σ_u r_u ⟨n_u²⟩

- The swap is exact at every integer configuration, since ⟨n²⟩ = c² there.
- At fractional occupation it removes ½ r_u Var_u of self-interaction.
- F stays convex iff W − diag(r) ⪰ 0. The largest certifiable r is therefore
  the largest SIC that convexity allows, per geometry.

**Theorem.** Let W ≻ 0, M = W⁻¹, s_u = Σ_v |M_uv|. Then W − diag(r) ⪰ 0 for
every 0 < r_u ≤ 1/s_u.

*Proof.*
1. With R = diag(r), W ⪰ R ⟺ λ_max(R^½MR^½) ≤ 1.
2. For the symmetric matrix B = R^½MR^½, λ_max(B) ≤ ρ(|B|).
3. By the Collatz–Wielandt bound with x_u = r_u^{−½},
   ρ(|B|) ≤ max_u (|B|x)_u/x_u = max_u r_u s_u ≤ 1. ∎

Checked numerically on 2000 random positive definite matrices: the worst
eigenvalue of W − diag(1/s) is −3e-16 relative, i.e. tight to round-off.

**Properties, all tested:**
- r_u ≤ 1/M_uu ≤ W_uu.
- An isolated unit gets r = W_uu: its full on-site Coulomb is removed, so its
  effective gap is G + r = U_Z, the exact atomic hardness (PPLB).
- For two units the bound is tight (det(W − R) = 0). The curvature left along
  the transfer direction is |w|(a−d)²/((a+|w|)(d+|w|)), which is zero for
  equal units.
- A dense, strongly screened W gives r → 0, which is the v1 mean field.
- Using `θ = 0.999` makes W − diag(r) strictly positive definite.
- |M_uv| is replaced by the smooth upper bound √(M_uv² + ε_uv²), with
  ε_uv = ε₀ √(M_uu M_vv) ρ_uv² + eps_machine. Here ρ_uv = J_uv/√(J_uu J_vv)
  and ε₀ = `sic_smoothing` = 0.01. Any ε ≥ 0 keeps the certificate valid; it
  only loosens it.

**Measured** (kT = 0.005 eV, untrained heads): the exact classical ensemble
`−kT ln Σ_n e^{−E(n)/kT}` over the integer configurations, against the mean
field with and without the certificate (`tests/redox/test_redox_sic.py`):

| system | exact F | certified F | v1 F |
|---|---|---|---|
| Na···Cl, 60 Å (eV) | 0.0000 | 0.0000, c = 0.000 | −0.302, c = 0.303 |
| Na···Cl, 9.43 Å (crossing) | −0.0001 | −0.042 | −0.49 |
| Na···Na, Q = +1, 60 Å | 5.1356 | 5.1316 | 4.618 |
| H···H, Q = +1 (H₂⁺-like), 60 Å | 13.5950 | 13.5900 | 12.046 |

- The remaining 4–5 meV is entropy; the bound is 3kT ln 3 = 16 meV.
- The only visible deviation is at the exact degeneracy point, and it stays
  below curvature/8.

**Na···Cl free charge with the certificate** (kT = 0.01, `measure_physics.py`,
h = 0.5 and 0.8 alike):
- 1.000 at R = 2.36, 4 and 6 Å;
- 0.45 at 9.43 Å, the classical ionic–neutral crossing
  K_E/(IE_Na − EA_Cl) = 9.434 Å;
- 0.000 at 15, 30 and 100 Å.

The h dilemma of §8 is gone: a compact free charge keeps the ionic bond, and
the spurious transfer at dissociation is removed.

**Removed fraction r_u/W_uu in condensed geometries** (min / median / max):

| geometry | min | median | max |
|---|---|---|---|
| benzene | 1e-4 | 1e-4 | 4e-4 |
| water dimer | 0.007 | 0.013 | 0.014 |
| (NaCl)₄ | 0.025 | 0.026 | 0.031 |
| NaCl dimer | 0.116 | 0.116 | 0.159 |

There the mean field remains, as convexity requires.

**Limitation (stated, not hidden).** The certificate is diagonal, i.e. per
unit. It makes the transfer between distant atomic units exact. For two
distant polyatomic fragments, each fragment's units are strongly coupled, so
r is small inside the fragment and the fragment-level self-interaction
(~½δ²/C_F) remains. Making that exact needs fragment-level (correlated)
ensembles; with product populations only diagonal removal is
exact-compatible.

## 9. Tests (`tests/redox/`, all analytic or independent)

| file | what is checked |
|---|---|
| `test_redox_ensemble.py` | float64 actually in force (guard), Legendre supremum, Φ' = ν, X = dN/dν, convexity for non-convex states, the Lemma 1 bound at 4 temperatures, disproportionation population |
| `test_redox_free_solver.py` | KKT with Φ' from bisection, the primal value, the global minimum on the constraint plane, envelope derivatives (dE/db = c, dE/dW = ½ccᵀ, dE/dÊ = p), the T5 Jacobian and Hessian, a golden-section brute force, the two-site T → 0 law with its rigorous bound, the underflow dead zone |
| `test_redox_bound.py` | equal-width blocks = the assembly's, concentric and far-field Coulomb, a joint Gram matrix ≻ 0, exact quadratic, free charge = ρ − χ − (η − S0) (with field and quadrupoles), the elimination is the minimum (with field), W ≻ 0 and J − W ⪰ 0, E_b0 = SPEC 1 |
| `test_redox_module.py` | forces vs central differences (9 configurations, certified SIC included), rotation (features rotated by their Wigner-D) + translation + permutation invariance, NIST isolated ions, the neutral atom, the dead zone = SPEC 1 (energy and forces), the Na···Cl two-site law on the model's own W at R = 2.5–40 Å, saturated Q, the parameter gradient of a force loss, the energy decomposition (with orbitals and λ), the C² switch, continuity across the attention cutoff, orbital bounds, refusals, a flashace `TransformersACEV3` with TRACE-Redox attached (forces vs central differences) |
| `test_redox_sic.py` | the certificate on 40 random PD matrices × 3 smoothings, smoothing only loosens, two-unit tightness and transfer curvature, isolated-unit full removal, smoothness through a sign change of (W⁻¹)_uv, exactness against the enumerated ensemble (Na···Cl at 12–60 Å, Na···Na⁺) and v1's measured failure there |
| `test_redox_periodic.py` | plane-wave = Ewald blocks (3-D triclinic and slab, ± quadrupoles), joint mixed-width Gram ≻ 0, large-cell open limit (dipole-free and dipolar), forces, stress, lattice translation, rock-salt symmetry, refusals |
| `test_redox_states.py` | states beyond +1: saturated and interior isolated cations at the NIST cumulative IE, capacity = electron count, absent states never populated, strictly convex levels under random shifts, K = 2 equal to K = 1 where IE_2 is irrelevant (1e-9), forces and a force-loss gradient with dications, the solver's capacity refusal |
| `test_train_redox.py` | the YAML block end to end (above), refusals, YAML float strings |

Helpers live in `tests/redox/redox_helpers.py`, not a `conftest.py`: a second
`conftest` module replaced `tests/electrostatics/conftest.py` in `sys.modules`
and broke `from conftest import require` in a combined run (measured, fixed).
Test files carry a `test_redox_` prefix for the same reason: rootdir-mode
pytest cannot collect two test modules with one basename (a `test_periodic.py`
collided with the electrostatics suite's).

**Mutation checks.** 13 planted bugs, each caught by at least one test:
- susceptibility without 1/kT;
- no implicit graph step;
- wrong concentric `J_fb`;
- field sign in u0;
- Ω transposed in b;
- gap anchor without the self-Coulomb;
- ±1 states swapped;
- a hard attention cutoff;
- uncapped orbital spread;
- unscreened W;
- the μ-gradient sign;
- λ omitted from `E_un`;
- `E_b0` missing its ½.

The uncapped orbital spread survived the first version of the orbital test,
which was then strengthened.

Fourteen more, each caught:
- **States:** the gap anchor without the self-Coulomb; absent states not
  masked; the cumulative energy adding the gap instead of the level; a capacity
  that ignores the electron count.
- **SIC:** no population add-back of r; the uncertified Schur r = 1/M_uu; full
  removal r = W_uu; r added instead of subtracted.
- **Periodic:** the charge–dipole sine sign; the slab correction dropped; the
  ±k pair not doubled; the Gaussian factor k²σ²/2; the quadrupole factor
  sign; too small a k cutoff.

## 10. Not done (all refused loudly)

**Refused features:**
- the matrix-free route (dense only; periodic cells are dense too, §2a);
- anions beyond −1 (second electron affinities are not tabulated, and gas-phase
  dianions of atoms are unbound); cations beyond +4;
- explicit bound fragment charges;
- spin and the flat-plane envelope (SPEC 25.6 c);
- metals (SPEC 20);
- SPEC 21 charge states;
- dispersion;
- block-diagonal batches (SPEC 24): the flashace per-structure loop is used
  instead;
- polarizability, Born-charge and dielectric observables (dipole and
  partial-charge losses work);
- a field under PBC, and a slab with net free charge (§2a).

**Polyatomic fragments: solved by §12** (the default since 2026-10-03). What
follows records the measurement that motivated it, made with the atom-level
mean field (`fragments='none'`).

Exact transfer between distant POLYATOMIC fragments. The per-unit certificate
(§8a) cannot remove the fragment-level self-interaction; that needs
correlated, fragment-level ensembles.

Measured with untrained heads (kT = 5 meV), two identical water molecules
30 Å apart, Q = +1:
- the hole splits 0.500 / 0.500;
- F is 1.344 eV below the exact E(H₂O⁺) + E(H₂O) − kT ln 2 with the
  certificate, and 1.358 eV below without it.

The same test on two Na atoms gives −0.004 eV (entropy only). The error is
the mean-field fragment self-energy, ¼ κ_F (κ_F ≈ 5.4 eV for water): the
semilocal-DFT failure for (H₂O)₂⁺ → H₂O⁺ + H₂O, and not yet fixed.

## 11. Training (`train.py`, `flashace`)

    electrostatics:
      enabled: true
      heads: {...}            # ElectrostaticHeads, as for trace_electrostatics
      dipole_weight: 1.0      # allowed: reads only the solved q, mu
      partial_charge_weight: 0.1
      redox: {kT: 0.1}        # RedoxConfig fields; {} for the defaults

**Behaviour.**
- With a `redox` mapping, `train.py` attaches `RedoxElectrostatics` instead of
  `ElectrostaticModule`.
- The checkpoint stores the resolved `RedoxConfig` (every field, not the YAML
  mapping) under `config.electrostatics.redox`, so a model reloads with the
  physics it was trained with whatever the defaults are later. Before
  2026-10-08 the raw mapping was stored.
- `flashace.checkpoint.attach_checkpoint_electrostatics` rebuilds it, so the
  ASE calculator loads it.
- Every path without `redox` is byte-identical to before.

**Refused with redox:**
- polarizability and Born-charge losses (`polarizability_weight`,
  `born_charge_weight`), which are defined from a single SPEC 1 state;
- the per-structure dielectric loss (`dielectric_weight`), except with
  `anchor='formal'`: there it reads the bound sector's ε∞
  (`RedoxElectrostatics.bound_state`, §17.7), and the configuration ensemble's
  response (§30) is not added to this loss. **[Corrected 2026-11-16:** this list
  said "polarizability, Born-charge and dielectric losses"; `train.py` allows the
  dielectric loss with formal anchors (§17.7).**]**
- the response priors `dielectric_prior` and `polarizability_prior` with any
  anchor but `'formal'`. With formal anchors they add the configuration
  ensemble's response when the window holds several configurations (§30,
  §36.10), and refuse a structure whose response record is not exact (the
  core + carrier elimination, a dressed §32 fold). `field_polarizability_prior`
  differentiates the total energy and is not restricted to formal anchors;
- spin heads;
- `transfer_coupling`;
- `attention_cutoff` > `r_max`;
- `fragment_policy='initial'` in the calculator.

**Tested.** Three optimiser steps (batched and per structure) on water, OH⁻,
NaCl, Na⁺ and rock salt. The checkpoint reloads with identical parameters,
and the calculator equals the model forward to 1e-10 in energy and forces.


## 12. Molecule-level exactness — the two-level certified ensemble

**Status (2026-10-03).** Implemented in `trace_redox/fragments.py` as
`RedoxConfig.fragments = 'bonds'`, **the default**; `'none'` is the atom-level
mean field of §1–8a. §12.12 lists the corrections made while implementing it.
- Every mathematical step is proved below or checked numerically, with the
  check named.
- `experiments/redox_v1/{fragment_anatomy,check_block_certificate,prototype_two_level}.py`
  reproduce §12.0, §12.3 and §12.9; the prototype reproduces every
  quoted number; it verifies the equations and is not an implementation.
  `measure_fragments_nacl.py` measures the implemented module against the
  exact Na···Cl ensemble (§12.9).
- Every term is marked NEW-TO-TRACE or ESTABLISHED.

### 12.0 Diagnosis (measured)

One water molecule, the model's own mean-field energy E_F(N) at total free
charge N ∈ [0, 1] (untrained heads, kT = 5 meV):

- **E_F(N) is quadratic to 1 %.** The curvature measured over N = 0…1 is
  5.946 / 5.923 / 5.920 eV.
- **The split error is that curvature.** Spreading one hole over two distant
  waters costs 2E(½) − E(0) − E(1) = −1.482 eV = −κ/4. The measured pair error
  (−1.344 eV at 30 Å) is this plus +¼·K_E/R = +0.12 eV of mean-field Coulomb
  between the half charges.
- **The electrostatic charging curvature is smaller.**
  κ_F = 1/(1ᵀW_F⁻¹1) = 5.85 eV is below the actual 5.92 eV. The hole lives on
  the two H atoms only, and a smaller active set charges harder.

**Consequence.** Removing only the electrostatic κ_F along each molecule's
total-charge direction leaves a chemistry-dependent residual. It is small for
water but of order W_uu for a hole confined to one atom of a large molecule.
Exactness needs each molecule's OWN energy at integer total charge, i.e. a
two-level ensemble.

### 12.1 The exact target

Molecules F (a partition 𝒫 of the units) carry integer total free charges
N_F. Inside a molecule the charge stays delocalised (mean field, as now).

**Conditional energy of molecule F at total N**, in the potential ψ (b
absorbed):

    E_F(N) = min_{y ∈ R^|F|, 1ᵀy = N} [ Σ_{u∈F} Φ_u(y_u) − ψᵀy + ½ yᵀ W_FF y ]

**Non-interacting molecules** (W_FG = 0 for F ≠ G):

    F_exact(Q) = −kT ln Σ_{N ∈ Z^m, Σ_F N_F = Q} exp(−Σ_F E_F(N_F)/kT)
               → min_{Σ N_F = Q} Σ_F E_F(N_F)        (T → 0)

E_F(N) is convex in N: a partial minimum of a jointly convex function under a
constraint linear in (y, N). So the T → 0 exact energy is piecewise linear in
each molecule's charge. That is PPLB at the molecular level (ESTABLISHED:
Perdew, Parr, Levy, Balduz 1982).

### 12.2 The functional for one partition 𝒫 (NEW-TO-TRACE)

**Variables.** For each molecule F:
- state probabilities p_F(N) ≥ 0 with Σ_N p_F(N) = 1, over an integer window
  N ∈ 𝒩_F (§12.7);
- charge "masses" x_F^N ∈ R^|F| with 1ᵀx_F^N = N p_F(N).

The mean atomic charges are c_F = Σ_N x_F^N, so Σ_F Σ_N N p_F(N) = 1ᵀc = Q.

    G_𝒫(p, x) = Σ_F Σ_N [ p_F(N) A_F(x_F^N / p_F(N)) + kT p_F(N) ln p_F(N) ]
                + bᵀc + ½ cᵀ S c
    A_F(y)    = Σ_{u∈F} Φ_u(y_u) + ½ yᵀ D_F y,      S = W − blockdiag(D_F)

    F_𝒫(c) = min_{(p, x) : Σ_N x_F^N = c_F} G_𝒫(p, x)

**Properties.**
- **Joint convexity.** p·A(x/p) is the perspective of the convex A_F, so it is
  jointly convex in (p, x) (ESTABLISHED: Boyd & Vandenberghe, Convex
  Optimization, §3.2.6). p ln p is convex, and ½cᵀSc is convex iff S ⪰ 0
  (§12.3). Partial minimisation preserves convexity, so F_𝒫 is convex in c.
- **Exact at integer molecular configurations.** If every p_F is a δ at N_F,
  then G = ΣΦ + bᵀc + ½cᵀ(blockdiag(D) + S)c = ΣΦ + bᵀc + ½cᵀWc: the original
  energy, exactly.
- **Error for a mixture.** Against the product of exact conditional ensembles,
  G is lower by ½ Σ_F [Σ_N p_F(N) c^{N}ᵀ S_FF c^{N} − c̄_Fᵀ S_FF c̄_F] ≥ 0.
  This is a variance in the S_FF metric, with S_FF = W_FF − D_F. The residual
  self-interaction is therefore exactly the part S_FF of each molecule's own
  Coulomb that convexity forces to stay in the mean field.
- **Entropy, no double counting.** The p ln p term is the molecular-state
  entropy. The atomic Mermin entropies inside Φ_u enter through the
  perspective as entropies conditional on N. That is the chain rule
  S(N, y) = S(N) + Σ_N p(N) S(y | N) (ESTABLISHED).
- **Canonical, as implemented (§12.13).** The first draft took the product
  over molecules, which fixes only the mean, Σ_F ⟨N_F⟩ = Q. The exact target of
  §12.1, and the physics of a closed system, is canonical: Σ_F N_F = Q in every
  configuration. The product form was replaced after it let a soft 108-atom
  MgO slab gain or lose an electron with no counter-charge anywhere (§12.13).
  The joint distribution over (N_F) is therefore the canonical one, and the
  functional above holds with the p_F read as its marginals.

### 12.3 The block certificate (NEW-TO-TRACE)

**Theorem.** Let W ≻ 0, M = W⁻¹, and 𝒫 a partition into blocks F. Set

    K_FG = M_FF^{−½} M_FG M_GG^{−½},   θ_F ≤ 1 / (1 + Σ_{G≠F} ‖K_FG‖₂),   D_F = θ_F (M_FF)⁻¹.

Then W − blockdiag(D_F) ⪰ 0, and 0 ⪯ D_F ⪯ W_FF.

*Proof.*
1. With D = blockdiag(D_F) ≻ 0, W ⪰ D ⟺ λ_max(D^½ M D^½) ≤ 1.
2. B = D^½ M D^½ has diagonal blocks θ_F I and off-diagonal blocks
   √(θ_F θ_G) K_FG.
3. For a unit vector v with a_F = ‖v_F‖:
   vᵀBv ≤ Σ_F θ_F a_F² + Σ_{F≠G} √(θ_F θ_G) ‖K_FG‖₂ a_F a_G = aᵀTa ≤ ρ(T).
   T is the symmetric non-negative matrix with T_FF = θ_F and
   T_FG = √(θ_Fθ_G)‖K_FG‖₂.
4. By Collatz–Wielandt with x_F = θ_F^{−½}:
   ρ(T) ≤ max_F (Tx)_F/x_F = max_F θ_F (1 + Σ_G ‖K_FG‖₂) ≤ 1.
5. Finally, (M_FF)⁻¹ = W_FF − W_{F,·} W_{·,·}⁻¹ W_{·,F} (the Schur complement
   over the other blocks) ⪯ W_FF, and θ_F ≤ 1. ∎

**Properties.**
- **Two blocks: tight.** The eigenvalues of B are θ(1 ± σ_i(K)).
- **Decoupled blocks.** M_FG = 0 gives θ_F = 1 and D_F = W_FF.
- **Single atoms.** It reduces to a per-atom certificate (a different, equally
  valid Collatz weighting from §8a's r_u = 1/Σ_v|M_uv|).
- **Smoothness.** For smooth geometry dependence use
  √(‖K_FG‖_F² + ε²) ≥ ‖K_FG‖₂ (Frobenius, smoothed). It only lowers θ, so the
  certificate stays valid.
- **Strictness.** Take θ_F = c/(1 + Σ_{G≠F}‖K_FG‖) with c = 0.999. Then
  λ_max(D^½MD^½) ≤ c, so D ⪯ cW and S = W − D ⪰ (1 − c)W ≻ 0: a capped
  min(0.999, 1/(1+Σ‖K‖)) gives only S ⪰ 0. (A first draft of this section had
  the capped form; corrected.)

**Checked.**
- 3000 random W and partitions (n ≤ 29, up to 8 blocks): the worst eigenvalue
  of W − blockdiag(D) is −1.7e-13 relative (round-off at tight cases).
- D_F ⪯ W_FF held in every case.
- For decoupled blocks, D_F = W_FF exactly.

### 12.4 The computable dual (reuses the existing solver)

Write ½cᵀSc = max_z (zᵀSc − ½zᵀSz) and use the conjugate of each molecular
term:

    f_F*(ψ) = sup_{p, y^N} [ Σ_N p_N (ψᵀy^N − A_F(y^N)) − kT p_N ln p_N ] = −Ω_F(ψ)
    Ω_F(ψ)  = −kT ln Σ_{N∈𝒩_F} exp(−e_F(N; ψ)/kT)
    e_F(N; ψ) = min_{1ᵀy = N} [ A_F(y) − ψᵀy ]

The middle line follows from x = p y, the inner supremum over y with
1ᵀy = N, then the softmax identity for the supremum over p. This gives the first draft's mean-constrained dual,

    L_mc(z, μ) = −½ zᵀSz + μQ + Σ_F Ω_F( (μ1 − b − Sz)_F ),

which the implementation replaces by the canonical dual of §12.13:

    E = E_b0 + max_z L(z),    L = −½ zᵀSz + A(−b − Sz)
    A(ψ) = −kT ln Σ_{N ∈ Π𝒩_F, Σ_F N_F = Q} exp(−Σ_F e_F(N_F; ψ_F)/kT)

**Concavity.** e_F(N; ψ) is a minimum of functions affine in ψ, hence concave
in ψ. Ω_F and A are soft-minima of concave functions (sums of the e_F); a
soft-min −kT ln Σ exp(−f_i/kT) is concave and non-decreasing in each f_i, so
both are concave. L is therefore concave: the same structure as the present
dual (§6), with S in place of W.

**Reuse of the existing solver.** e_F(N; ψ) is exactly the present free-charge
problem on the atoms of F, with b → −ψ, W → D_F and Q → N:
`free_solver.differentiable_solution` as it stands.

**Derivatives** (envelope theorem), with p the canonical joint distribution
and y(N) = ⊕_F y_F^{N_F} the conditional charges:
- ∂A/∂ψ = −c̄ = −E_p[y(N)];
- ∂²A/∂ψ² = −[E_p[χ(N)] + Cov_p(y(N))/kT], where χ(N) = blockdiag(χ_F^{N_F})
  holds the conditional (fixed-N) susceptibilities of the inner problems.
- The canonical constraint correlates the molecules, so Cov_p has
  off-diagonal blocks Y_F C_FG Y_Gᵀ, with
  C_FG(n, n') = Pr(N_F = n, N_G = n') − p_F(n) p_G(n'). Without them the
  outer Newton fails to converge on a shared-hole water pair (planted bug
  M39, §12.11).

**Initialisation.** The canonical dual has no μ. Its start is z = 0, δ = 0
(ψ = −b). The mean-constrained draft needed μ by bisection, because its
μ-curvature Σ_F Var(N_F)/kT ≈ e^{−gap/kT} vanishes in the dead zone.

### 12.5 Nested atomic certificate inside a molecule

Inside A_F, the atomic self-interaction can still be removed where convexity
allows: replace ½yᵀD_Fy by ½yᵀ(D_F − diag r^F)y and add ½ r_u^F n² to the
atomic state energies, with r_u^F = θ/Σ_v |(D_F⁻¹)_uv| (§8a applied to D_F).
This is convex by §8a. A one-atom molecule then has exact integer states.

### 12.6 Molecules without discrete jumps: the bond-percolation ensemble of partitions (NEW-TO-TRACE)

**Chemical bond strength** (as implemented; see §12.12 for the change from
the first draft):

    s_b = 𝒮(r_b; R_i + R_j + 0.25 Å, R_i + R_j + 0.60 Å),
          𝒮 = 1 below the first value, 0 above the second, the C² quintic in between

- R are covalent radii (ESTABLISHED: Cordero et al., Dalton Trans. 2008, 2832;
  tabulated in `constants.py`, 0 mismatches against ASE).
- The additive tolerance brackets the established bond criterion
  d < R_i + R_j + 0.45 Å (ESTABLISHED: Open Babel, O'Boyle et al.,
  J. Cheminform. 3, 33 (2011)).
- Hydrogen bonds stay unbonded: O···H ≈ 1.8–2.0 Å > 0.97 + 0.60 Å.
- This is independent of the bound sector's SQE cutoff.
- **Coverage.** The neighbour list must contain every pair the switch can see,
  so R₁ + R₂ + 0.60 ≤ attention_cutoff for the two largest atoms present.
  Otherwise the model refuses: K–Na needs 4.29 Å, K₂ 4.66 Å, Cs₂ 5.48 Å.

**Partition weights** (independent bond percolation; ESTABLISHED: Broadbent &
Hammersley 1957):

    π_𝒫 = Σ_{ω ∈ {0,1}^bonds : components(ω) = 𝒫} Π_b s_b^{ω_b} (1 − s_b)^{1−ω_b}
    F(c) = Σ_𝒫 π_𝒫 F_𝒫(c),        E = E_b0 + min_{Σc = Q} F(c)

**Properties.**
- **Convex.** A convex combination of convex functions.
- **Smooth.** π is a polynomial in the s_b, so E has the smoothness of 𝒮 (C²).
  A partition appears or disappears only where some s_b reaches 0 or 1, and
  there its weight vanishes with zero slope.
- **Exact limits.** Stable molecules (every s_b ∈ {0,1}) give one partition
  and the exact molecular limit. A breaking bond with strength s gives two
  partitions with weights s and 1 − s.
- **Exact reductions before enumerating.**
  - An uncertain bond inside a component of the certain (s = 1) bonds never
    changes the partition, so it is dropped.
  - The uncertain bonds between the same two components connect them iff at
    least one is present, so they become one effective bond with
    s_AB = 1 − Π_b(1 − s_b).
  - Without these, the Au₂/MgO slab had 55 switching-shell bonds; with them,
    a handful of effective ones.
- **Negligible probabilities** (PARTITION_WEIGHT_FLOOR = 1e-30).
  - A connection absent with probability below the floor is certain.
  - A partition of weight below the floor is not enumerated.
  - Either moves E by < 1e-30·|A|, under 1e-28 eV here, 14 orders below the
    float64 resolution of the energy. Forces are likewise unaffected:
    π ~ t³ at a switch end while d ln π/dx ~ 1/t.
  - Absent weights are taken as a = Π(1 − s_b) directly, not as 1 − (1 − a),
    which loses every a < 1.1e-16 and 7 digits at a = 1e-10 (§12.12).
- **Cost.** 2^k patterns over the k effective bonds, grouped by the partition
  they produce. A k above `fragment_max_uncertain` (10) is refused, never
  truncated silently.

### 12.7 The molecular charge window 𝒩_F

𝒩_F = {N_F⁻, …, N_F⁺} must contain every N a minimiser can use.
- For a single molecule it is exactly {Q}.
- Otherwise it starts at |N| ≤ max(`fragment_window`, ⌈|Q|⌉), clipped to the
  capacity [−|F|, Σ_u min(K, Z_u)].
- Truncating it is exact iff the omitted edge states carry zero population.
- **Adaptive.** After each solve, any molecule whose π-weighted population at
  a window edge exceeds 1e-8 (where that edge is not a capacity bound) is
  widened on that side, and the solve is repeated (at most 64 times, then
  refused).
  - The weight π matters. A partition moves the energy by π times its
    contribution, and its KKT residual is π-weighted, so the solver need not
    converge an almost-absent partition. Unweighted, such a partition's edge
    population drove the window to N = −65 (§12.12).
- **History.** The 108-atom MgO slab needed widening in the mean-constrained
  draft: its first window ±2 had 1.9e-5 at N = −2. Canonically, its charge is
  bounded by its partners' windows.
- Tested: starting at ±1 at kT = 0.5 eV gives the same energy as ±6, to 1e-7.

### 12.8 Theorems (proved above or immediate from it)

| | statement |
|---|---|
| B1 | **Molecular exactness.** For mutually non-interacting molecules, θ → 1 and T → 0: E = E_b0 + min_{N∈Z^m, ΣN_F=Q} Σ_F E_F(N_F). Proof: S → 0 decouples the molecules. Each molecular free energy tends to the convex envelope over N of the convex E_F(N), i.e. the piecewise-linear interpolation (Lemma 1 at the molecular level). A separable sum of piecewise-linear convex functions under ΣN̄ = Q (integer) is minimised at an integer vertex; ties have equal energy. |
| B2 | **Reduction, to a proved bound.** A single molecule has window {Q}, D = cW and S = (1−c)W, so it is the present model, except that the nested atomic removal is computed on cW. Every term of the §8a certificate scales as 1/c, so r(cW) = c·r(W) exactly, and 0 ≤ E_none − E_bonds ≤ ½(1−c) Σ_u r_u Var_u, zero in the dead zone. Measured on water: 0, 3.3e-5 and 6.1e-6 eV for Q = 0, +1, −1, against a bound of 1.5e-4 eV. Using the present path itself for a single molecule was tried and rejected: it put a 4.2e-5 eV jump where a bond enters its switch (§12.12). |
| B3 | **Convexity and uniqueness.** F is convex. With θ ≤ 0.999 and kT > 0 it is strictly convex in c, so the minimiser and the energy are unique and history independent. |
| B4 | **Smoothness.** C² in the positions for kT > 0 (§12.6, smooth certificates). |
| B5 | **Size consistency.** For non-interacting systems, E(A + B) = E(A) + E(B) with the charges distributed as in B1. This is exactly what the mean field violates (−1.34 eV, §10). |

### 12.9 Verified numbers (untrained heads, kT = 0.01 eV, Q = +1)

Errors are against the molecule-exact target of §12.1 (enumerated, canonical,
full Coulomb).
- "Mean-constrained" is the prototype of the first draft
  (`experiments/redox_v1/prototype_two_level.py`); the module reproduced it to
  every printed digit before §12.13.
- "Canonical" is the module as implemented.

| pair | R (Å) | θ | canonical (module) | mean-constrained | atomic mean field | hole localisation (canonical) |
|---|---|---|---|---|---|---|
| H₂O···H₂O | 4 | 0.502 | −0.049 eV | −0.056 eV | −0.221 eV | ⟨N_A⟩ = 0.539 |
| | 6 | 0.639 | −0.113 | −0.120 | −0.859 | 0.521 |
| | 10 | 0.785 | −0.031 | −0.038 | −1.116 | 0.522 |
| | 30 | 0.923 | −0.0035 | −0.010 | −1.359 | 0.511 |
| H₂O···H₂S | 4–30 | 0.50–0.91 | ≤ 1e-4 eV | ≤ 1e-4 eV | 0.000 → −0.138 eV (0.17 e on water at 30 Å) | ⟨N_H₂S⟩ = 1.000 at every R |

- **Symmetric pair.** ⟨N⟩ ≈ ½ is the correct ensemble average over two
  degenerate integer configurations.
  - Canonically the energy is right to 3.5 meV at 30 Å.
  - The mean-constrained draft was 6.5 meV further off, which is
    kT ln 2 = 6.9 meV: the entropy of the two unphysical configurations
    (0, 0) and (1, 1) that a product ensemble admits.
- **Certificate used.** θ_F = 0.999/(1 + ‖K_AB‖_F), the strict form; the
  table θ values are 1/(1 + ‖K‖).
- **Asymmetric pair.** The hole is an exact integer on the molecule with the
  lower level.
- **Intermediate distances (4–10 Å).** The residual comes from θ < 1, where
  the molecules' Coulomb coupling is still substantial, and it is bounded by
  the S_FF variance of §12.2.

**Ion pair through the module** (`experiments/redox_v1/measure_fragments_nacl.py`,
untrained heads, kT = 0.01 eV, Q = 0). Errors are against the exact ensemble
over integer atomic charges (full W, enumerated). The bond switches off at
1.66 + 1.02 + 0.60 = 3.28 Å.

| R (Å) | `fragments='bonds'` (default, canonical) | atom-level certified (`'none'`) | v1 | q_Na exact / bonds / none |
|---|---|---|---|---|
| 2.36 | −4e-16 eV | −2e-16 | −3e-14 | 1.000 / 1.000 / 1.000 |
| 6 | −2e-16 | −1e-10 | −0.127 | 1.000 / 1.000 / 1.000 |
| 8 | −1.1e-6 | −3.4e-3 | −0.357 | 1.000 / 1.000 / 0.910 |
| 9.43 | −0.021 | −0.049 | −0.498 | 0.110 / 0.412 / 0.453 |
| 10 | −0.0018 | −0.020 | −0.480 | 0.000 / 0.092 / 0.278 |
| 12 | −1.6e-13 | −7e-6 | −0.434 | 0.000 / 0.000 / 0.000 |
| 100 | 0 | 0 | −0.299 | 0.000 / 0.000 / 0.000 |

- The default is exact to round-off at contact and away from the crossing.
- It is off by at most 21 meV in the window around the ionic–neutral
  crossing, against 49 meV for the atom-level certificate.
- The mean-constrained draft was off by 28 meV at 9.43 Å, 1.9e-4 eV at 8 Å
  and 8e-8 eV at 12 Å.
- The error is negative wherever it exceeds round-off (the largest positive
  value, +4.8e-24 eV at 15 Å, is 8 orders below the float64 resolution of these
  energies), as the variance bound of §12.2 requires. In the crossing window
  the mean-field part of the inter-molecular coupling (θ < 1) still allows
  fractional sharing.

### 12.10 Status of the earlier hypotheses, and what remains

1. **Second-order implicit derivatives** through the nested solve: **verified.**
   The force-loss parameter gradient matches finite differences to 1e-6
   relative, for two molecules, for a shared hole (6.4e-10 measured) and
   through a breaking bond (two partitions).
   Planting a missing Cov/kT term in the outer Hessian, or a wrong sign in the
   fixed-N susceptibility, makes these tests fail.
2. **Bond thresholds:** additive +0.25 / +0.60 Å (§12.6, §12.12).
3. **Cost, measured.**
   - Real geometries (Ag₃±, C₁₀H₂/C₁₀H₃⁺, Na₉/₈Cl₈⁺, water clusters):
     0.02–0.33 s per structure.
   - Au₂/MgO slab (110 atoms, periodic): 2.48 s with forces, against 1.66 s
     for the mean field (mean-constrained draft).
   - One training-type evaluation (E, forces with create_graph, and
     d(ΣF² + 0.1E)/dθ) on Au₂/MgO training structures, canonical, Mac CPU with
     4 threads beside other jobs: 11.5–12.6 s with fragments, against
     10.9–12.5 s for `fragments='none'`. The dense bound-sector elimination
     dominates, not the fragments. Memory on the GPU is about 4 GB per structure
     (batch 2 used 8.1 GB).
   - Compressed random test clusters, with many switching-shell bonds: about
     5× the mean field.
4. **Not supported:** `orbitals=True` (an orbital could span two molecules),
   the uniform λ, and species pairs beyond the neighbour coverage. All are
   refused.
5. **Physics outside this classical ensemble.** Covalent sharing of a charge
   between molecules at short range (e.g. the (H₂O)₂⁺ hemibond) needs the
   inter-molecular hopping of SPEC 25.8 D / SPEC 22. Here such a charge is an
   integer or a thermal mixture, never coherently shared.
6. **Spin** is not included.

### 12.11 Tests (`tests/redox/test_redox_fragments.py`, 26)

**What is checked.**
- **Bonds.** Covalent bonds are found and hydrogen bonds are not; the
  percolation probabilities match their closed form (cycle and bridge); the
  exact reductions hold (internal bonds dropped, parallel bonds merged); the
  coverage and k limits refuse correctly.
- **Negligible probabilities.** A connection whose absent probability
  underflows (25 bonds of 1e-15) is certain. Rare partitions carry exact
  weights a(1 − a) = 1e-20. A partition of weight a² = 1e-40 is not
  enumerated.
- **Certificate.** The block certificate holds on the model's own W:
  S ⪰ (1−c)W and D_F ⪯ W_FF.
- **Canonical sums** (§12.13). A, the marginals and the pair covariances
  equal brute-force enumeration for three molecules and Q = 0, +1, −2, for
  three values of the tilt μ. Two molecules are exact mirror images,
  p_B(n) = p_A(Q − n), at kT = 0.3 eV.
- **B1** against the independent enumeration (H₂O···H₂O within 6 meV at 30 Å,
  measured 3.5 meV; H₂O···H₂S an exact integer hole). **B2** within its
  bound.
- **Derivatives.** Forces, and the force-loss parameter gradient, against
  finite differences (two molecules; a shared hole; a breaking bond).
- **Continuity** where a bond enters and leaves its switch. Measured: second
  differences −3.0e-14 and −2.5e-14 eV; force changes 5.7e-8 and 4.1e-8 eV/Å
  over 2e-6 Å.
- **Periodic box** with two molecules (forces). **Adaptive windows** equal to a
  wide window, and the widening is π-weighted. **Refusals.**

**Agreement with the prototype.** Before §12.13 the module reproduced the
independent (mean-constrained) prototype of §12.9 to all printed digits:
12.7541 / 13.1371 / 13.2671 / 13.2999 eV and 10.1464 … 10.3296 eV.

**Mutation checks.** Seventeen planted bugs, each caught:
- the certificate ignoring the coupling;
- an absent bond weighted by s;
- the outer Hessian without Cov/kT;
- no implicit graph step;
- a single-molecule window that is not {Q};
- the nested removal not added back;
- bonds counted in both directions;
- the fixed-N susceptibility with the wrong sign;
- parallel bonds not merged;
- bonds inside a molecule kept;
- (M38) product (mean-constrained) marginals instead of canonical ones;
- (M39) the outer Hessian without the cross-molecule covariance;
- (M40) a wrong convolution index;
- (M41) a rare connection left uncertain;
- (M42) negligible partitions kept;
- (M43) window widening not π-weighted;
- (M44) absent weights computed as 1 − (1 − a).

### 12.12 Corrections made while implementing (each found by a test or a sweep)

1. **Bond criterion.** Multiplicative 1.2 / 1.6 × (R_i + R_j) needs neighbour
   lists past 4 Å for heavier elements (Na–Cl: 4.29 Å). The additive
   +0.25 / +0.60 Å form brackets Open Babel's +0.45 Å.
2. **B2 shortcut removed.** Sending one molecule spanning the system down the
   present path gave a 4.2e-5 eV jump at a bond's switch onset. Every partition
   now goes through the same machinery, and B2 is the bound of §12.8.
3. **Outer variables.** The ψ-form dual divided by S̄ = 0.001W for a single
   molecule, and the cancellation in h = Σπψ + b − μ1 stalled a C₁₀H₃⁺ chain
   at a KKT residual of 1e-9. The implementation uses
   ψ^P = μ1 − b − S̄z + δ^P with Σ_P π_P δ^P = 0, so that h = −S̄z with no
   cancellation and no S̄⁻¹. Its Hessian is the ψ-form's, transformed.
4. **KKT residuals.**
   - Outer: |S̄(c̄ − z)| in eV, |Q − Σc̄| and the π-weighted disagreement
     between partitions in e. The reported charges are c̄, not z.
   - Inner (`free_solver`): |W(N − z)| in eV.
   - Reason: z is undetermined along the null space, where nothing depends on
     it. Wide free Gaussians in a dense crystal (σ_Na = 5 Å, 3–4 Å periods)
     make W_NaNa ~ 1e-20, and a charge-unit residual never converged there.
   - An almost-absent partition (π ~ 1e-18 at a switch end) is weighted by π
     in the residual.
5. **Precision of the certificate.** D_F = θ(M_FF)⁻¹ is computed as the Schur
   complement W_FF − W_FR W_RR⁻¹ W_RF, not by inverting M = W⁻¹ back. That is
   exact for a single molecule, and it removed a Cholesky failure on an
   ill-conditioned crystal W.
6. **Inner Newton.** When A = W + WXW is numerically singular, it falls back to
   an eigen pseudo-inverse.
7. **Molecular windows** are adaptive (§12.7).
8. **Canonical total charge** (§12.13; found training on Au₂/MgO). The
   mean-constrained ensemble let the 108-atom MgO slab, a separate molecule in
   partitions where the Au₂ bond is in its switch, take ±1 with probability 3%
   while Au₂ stayed neutral. The slab's model gap was 0.67 eV (untrained
   heads). That is charge created from a reservoir. Canonically,
   p_slab(n) = p_Au₂(−n) exactly.
9. **Partitions of negligible weight** (§12.6). On a dumped Au₂/MgO training
   structure, one Au atom's ~20 bonds near their switch end merged to an
   absent probability that underflowed to 0.0. The "uncertain" connection
   then produced a partition of weight exactly 0.
   - The solver correctly ignores such a partition (π-weighted residual), so
     its potentials were unconverged and its population sat on the window edge.
   - The unweighted window test then widened the 109-atom molecule's window to
     N = −65 and refused, after 64 full re-solves.
   - Fixed three ways: the floor PARTITION_WEIGHT_FLOOR, exact absent weights,
     and π-weighted widening.
   - The replayed structure now takes one solve (1.4 s) and gives
     E = −0.546731 eV, identical to the canonical result with the weight-0
     partition kept.

**Robustness sweep** (real geometries, untrained heads, the default
`attention_cutoff` raised to 4.5 Å to match the neighbour list): 60/60
structures converge (Ag₃±, C₁₀H₂/C₁₀H₃⁺, Na₉/₈Cl₈⁺, water clusters).
- Where every structure is one molecule (C₁₀ chains, NaCl clusters),
  E_bonds − E_none = 0.0000 eV.
- Ag₃ with a stretched bond (two partitions): −2.7 meV.

### 12.13 The canonical total charge (NEW-TO-TRACE; implemented)

**Physics.** A structure is a closed system: its total charge Q is fixed in
every microstate. The molecular ensemble must therefore be canonical over the
integer charges (N_F):

    p(N) ∝ Π_F exp(−e_F(N_F; ψ_F)/kT) · 1[Σ_F N_F = Q]

**Why the product fails.** The product of independent molecular ensembles
with a shared μ (the first draft) is the grand-canonical relaxation. It fixes
only Σ⟨N_F⟩ = Q, and is right only when many molecules exchange charge
(equivalence of ensembles). With a few molecules, a soft one fluctuates with
no counter-charge. The measured case is §12.12, item 8.

**Exact evaluation.** For any μ, write q_F(n) = exp(−(e_F(n) − μn − Ω_F)/kT),
the tilted molecular distribution, with Ω_F = −kT ln Σ_n exp(−(e_F(n) − μn)/kT).
Then

    Z_Q = Σ_{ΣN=Q} Π_F exp(−e_F(N_F)/kT) = exp(−μQ/kT) Π_F exp(−Ω_F/kT) · Pr_q(Σ_F N_F = Q)
    A   = −kT ln Z_Q = μQ + Σ_F Ω_F − kT ln Pr_q(Σ_F N_F = Q)
    p_F(n) = q_F(n) Pr_q(Σ_{G≠F} N_G = Q − n) / Pr_q(Σ_F N_F = Q)
    Pr(N_F = n, N_G = n') = q_F(n) q_G(n') Pr_q(Σ_{H≠F,G} N_H = Q − n − n') / Pr_q(Σ N = Q)

- **Independence of μ.** Every configuration has ΣN = Q, so μ cancels exactly.
- **Distributions of sums under q.** These are convolutions of the q_F,
  using prefix and suffix products and, for the pairs, middle products.
- **A link to the draft.** The canonical A equals the draft's mean-constrained
  dual at any μ plus the correction −kT ln Pr_q(ΣN = Q) ≥ 0. Hence
  A ≥ max_μ A_mc, and 0 ≤ E_can − E_mc ≤ kT Σ_F ln|𝒩_F|.
  - Upper bound: the product of the canonical optimum's marginals has the
    same energy, because G is separable in the molecular marginals.
  - Its entropy exceeds the canonical one by the total correlation, which is
    at most Σ_F H(p_F) ≤ Σ_F ln|𝒩_F|.
- **Numerics.** μ is chosen by bisection so that E_q[ΣN] = Q; it is detached,
  being exact for every μ. This keeps Pr_q(ΣN = Q) of order one rather than
  e^{−gap/kT}.
  - The e_F(N) are convex in N (a partial minimum of a convex function), so
    the q_F are log-concave, and so is their sum, whose mean is then Q.
  - Pr ≤ 1e-280 is refused, never silently used.

**Dual.** L(z, δ) = −½zᵀS̄z + Σ_P π_P A^P(−b − S̄z + δ^P), with Σ_P π_P δ^P = 0.
- Concave (§12.4).
- No μ variable, because Σc = Q in every configuration.
- A^P(ψ + α1) = A^P(ψ) − αQ makes δ^P ∥ 1 a gauge direction; the
  pseudo-inverse drops it, and the π-weighted ψ is gauge-free.
- `mu_e` is reported as NaN: there is no reservoir.

**Hessian.** −∂²A/∂ψ² = E[χ(N)] + Cov(y(N))/kT, including the cross-molecule
blocks Y_F C_FG Y_Gᵀ (§12.4).

**Measured effect** (§12.9):
- H₂O···H₂O⁺ is 6.5–7 meV closer to the exact target at every R
  (kT ln 2 = 6.9 meV).
- Na···Cl at the crossing: 28 → 21 meV.
- H₂O···H₂S is unchanged (exact).
- Au₂/MgO: the slab's charge mirrors Au₂'s, and no window widens.
- The redox suite runs in 105 s, against 163 s for the draft, since there is
  no μ bisection and fewer re-solves.


## 13. The condensed bound-sector elimination — the default (`bound._CondensedK`) — NEW-TO-TRACE

**Problem (measured, 2026-10-08).** §2 eliminates the bound sector through a
dense Cholesky factor of K, of order m = n_b + 3n (+5n). The force-loss
gradient then differentiates through that factorisation twice.
- Au₂/MgO (110 atoms, bond cutoff 5.29 Å): every atom has 66 neighbours, so
  n_b = 3,639 and m = 4,079; K is 133 MB.
- One training step (E, forces, d(ΣF² + 0.1E)/dθ) took 10.3–11.3 s on the Mac
  CPU. Of that, 66 % was dense `mm` and triangular solves of order m, and the
  double backward alone was 13 s in a separate profile.
- On the GPU host (an RTX 5060 Ti) the step took 26.6 s: FP64 at 1/64 rate, on a shared card.
- The cost grows as m³ ∝ (atoms × neighbours)³, so this route cannot scale to
  foundation-model data.

**Structure.** In the variational basis y = (t, p), with t the bond charge
transfers and p = (μ̂[, Θ̂]) the atom-centred moments, K is assembled from the
assembly's pair blocks (`ElectrostaticAssembly.dense_hessian`) as

    K = blockdiag(D, P) + Vᵀ C V,   D = diag(κ),   V = blockdiag(B, I),   C = [[H, T], [Tᵀ, 0]]

- B is the [n, n_b] bond incidence (q = Bt).
- H = J + diag(η) is the atomic hardness. J is the Gaussian Coulomb Gram
  matrix: positive semidefinite, and positive definite for distinct centres;
  under periodic boundaries it is the tinfoil Ewald sum, PSD with or without
  the k = 0 term. η > 0, so H ≻ 0.
- T = [T01, T02] are the charge–multipole couplings.
- P is the multipole block, with its on-site stiffness.
- κ = softplus(·) + κ_floor > 0 (ESTABLISHED in `heads.py`).

**Exact inverse.** With H = RRᵀ and G = RᵀB, the bond block is
K_tt = D + GᵀG, diagonal plus a rank-n term. Then

    K_tt⁻¹ = D⁻¹ − D⁻¹Gᵀ F⁻¹ G D⁻¹,        F = I + G D⁻¹ Gᵀ        (Woodbury; eig(F) ≥ 1)
    Λ      = B K_tt⁻¹ Bᵀ = L − L R F⁻¹ Rᵀ L,  L = B D⁻¹ Bᵀ
    S      = P − Tᵀ Λ T                       (Schur complement of K_tt in K; SPD iff K is)
    K⁻¹ [r_t; r_p]:  u = K_tt⁻¹ r_t,   x_p = S⁻¹ (r_p − Tᵀ B u),   x_t = u − K_tt⁻¹ Bᵀ T x_p

**Properties.**
- Only [n, n] and [n_p, n_p] matrices are factorised. The m-sized work is
  diagonal scaling and products with B, so the cost is
  O(n³ + n_p³ + n n_b k) for k right-hand sides, against O(m³).
- All operations are differentiable by plain autograd at that cost, so no
  implicit-function machinery is needed. First and second derivatives are
  those of the exact formula.
- F has eigenvalues ≥ 1, which is better conditioned than inverting H.
- H and P are symmetrised after the same round-off check as the dense route
  (relative block asymmetry ≤ 1e-10).
- With these blocks, Vᵀ C V reproduces `dense_hessian`'s off-diagonal blocks
  exactly, so the condensed K is the symmetrised dense K.

**Verified** (`tests/redox/test_redox_bound.py`; the dense route is kept as
`bound_elimination='dense'`):
- **Cases:** open, quadrupoles, applied field, 3-D periodic with quadrupoles,
  and 2-D slab.
- **Agreement:** the condensed K equals the dense K to 64 eps·max|K|.
  E_b0, u0, W, y0, the forces and the force-loss parameter gradient agree to
  ≤ 3 cond(K)·eps (measured); the test bound is 64 cond(K)·eps.
- **End to end:** the module (fragments, periodic) gives the same energy and
  forces to 1e4 eps.
- **Au₂/MgO** (two training structures): |ΔE| ≤ 1.7e-14 eV,
  max|ΔF| ≤ 3.6e-15 eV/Å, and the parameter gradient agrees to 2.3e-13
  relative.
- **Planted bugs**, each caught: (M45) the Woodbury term without D⁻¹; (M46) S
  without TᵀΛT; (M47) no back-substitution for x_t; (M48) H without η;
  (M49) Λ = L.

**Measured cost** (Au₂/MgO, Mac CPU, 6 threads, Spotlight indexing in the
background):

| route | energy | + forces | + force-loss gradient (one training step) |
|---|---|---|---|
| dense (§2) | 0.88–1.28 s | 3.65–4.32 s | 10.3–11.3 s |
| condensed | 0.60–0.83 s | 1.21–1.38 s | 1.73–1.95 s |

The elimination itself now takes 0.03 s. The rest of the step is the
`trace_electrostatics` periodic assembly (0.47 s forward) and the fragment
solve (0.30 s), with their backward passes.

## 14. Why average? The exact classical canonical ensemble, the two-level mean field, and a two-sided bound — NEW-TO-TRACE

This section asks one question: is the averaging in §12 forced by the
physics, by the mathematics, or only by cost? The answer is derived and then
measured (`experiments/redox_v1/bracket_two_level.py`).

### 14.1 The exact classical target, per partition

- **Configurations.** Molecules F = 1..m; molecular charge vectors
  N ∈ 𝒩 = {N ∈ Π_F 𝒩_F : Σ_F N_F = Q}.
- **Charges within a configuration.** Atomic charges y ∈ Y(N) =
  {y : 1_Fᵀ y_F = N_F for every F}, with energy

      𝓔(y) = Σ_u Φ_u(y_u) + bᵀy + ½ yᵀW̃y,     W̃ = W − diag(r) ⪰ 0

  Φ_u is the atomic Mermin functional with the certified removal added back.
  r is the removal the two-level model itself uses inside molecules (the
  nested r^F of §12.5, so that W̃ = S̃ + blockdiag(D_F − diag r^F)); r = 0 for
  `sic='none'`.
- **The ESTABLISHED Gibbs variational principle**, over a distribution p on 𝒩
  and configuration-resolved charges y(N) ∈ Y(N):

      F_ex = min_{p, y(·)} G_ex(p, y),   G_ex = Σ_N p(N) 𝓔(y(N)) + kT Σ_N p(N) ln p(N)
           = −kT ln Σ_{N∈𝒩} exp(−E(N)/kT),     E(N) = min_{y∈Y(N)} 𝓔(y)

### 14.2 The exact problem is convex without any certificate

- In the variables x(N) = p(N) y(N), each term p 𝓔(x/p) is the perspective
  of the convex 𝓔, hence jointly convex (ESTABLISHED: Boyd & Vandenberghe
  §3.2.6). The constraints 1_Fᵀ x_F(N) = N_F p(N) are linear.
- G_ex is therefore a convex program, strictly convex for kT > 0. It has a
  unique minimiser and is smooth in the geometry: each E(N) is a strictly
  convex minimum, and F_ex is an explicit soft-minimum of the E(N).
- **Convexity and uniqueness are not what forces averaging.** The exact
  ensemble already has both.

### 14.3 What forces averaging: the size of 𝒩

- The program has |𝒩|·U charge variables. Equivalently it needs |𝒩|
  constrained solves (E(N), one per configuration), then a soft-min.
- |𝒩| grows like w^{m−1} for m molecules with windows of w states:
  exponential in the number of molecules.
- **Averaging is the restriction to molecule-conditional charges**:

      y(N) = ⊕_F y_F(N_F)                                    (§12.2)

  This has Σ_F |𝒩_F|·|F| variables, polynomial in m.
- **m = 2: the restriction is empty.** Σ N = Q gives N_B = Q − N_A, so
  y(N) = (y_A(N_A), y_B(Q − N_A)) already spans every configuration-resolved
  charge field. For two molecules the exact problem has exactly the two-level
  problem's size.

### 14.4 Why the restricted problem needs the certificate

- Restricted to y(N) = ⊕_F y_F(N_F), the exact functional is

      G_r(p, y) = Σ_N p(N) [Σ_F A_F(y_F(N_F)) + bᵀy(N) + ½ y(N)ᵀ S̃ y(N)] + kT Σ p ln p
      A_F(y) = Σ_{u∈F} Φ_u(y_u) + ½ yᵀD_F y,     S̃ = W̃ − blockdiag(D_F)

- In the molecular masses x_F^n = p_F(n) y_F(n), the cross terms read
  Σ_N p(N) x_F^{N_F}ᵀ S̃_FG x_G^{N_G} / (p_F(N_F) p_G(N_G)). These are
  bilinear in x, so G_r is not jointly convex: multiple stationary points,
  history dependence, jumps.
- §12 restores convexity by replacing the quadratic's expectation with its
  value at the mean:

      E_p[½ yᵀS̃y]  →  ½ c̄ᵀS̃c̄,      c̄ = E_p[y]

  This is convex iff S̃ ⪰ 0, which is exactly what the block certificate θ_F
  guarantees (§12.3).
- So **averaging is the convexity device of the polynomial parametrisation**,
  and θ < 1 is its price.

### 14.5 The exact error of averaging (identity)

For every (p, y) in the restricted domain,

    G_r(p, y) − G_2L(p, y) = ½ E_p[yᵀS̃y] − ½ c̄ᵀS̃c̄ = ½ tr(S̃ Σ_p),     Σ_p = Cov_p(y) ⪰ 0

- **Block structure.** The block form is
  ½ Σ_F tr(S̃_FF Σ_FF) + Σ_{F<G} tr(S̃_FG Σ_GF).
  - The intra-molecular terms are ≥ 0: self-interaction kept by θ < 1.
  - The cross terms are < 0 for anticorrelated charges, e.g. a hole on A or
    on B. They remove the spurious repulsion between a hole and its own
    average on the other molecule.
  - S̃ ⪰ 0 makes the sum ≥ 0. That is the certificate's role.
- **Spectators drop out.** A molecule whose charge does not fluctuate has
  Σ_FF = 0 and Σ_FG = 0, so it contributes nothing. The error lives only in
  the fluctuating ("active") molecules.

### 14.6 Theorem: two-sided bound (canonical ensemble only)

With (p*, y*) the two-level optimum and Σ* = Cov_{p*}(y*):

    F_2L  ≤  F_ex  ≤  F_2L + ½ tr(S̃ Σ*)

**Proof.**
- **Upper bound.** (p*, ⊕_F y*_F) is a point of the exact domain, so
  F_ex ≤ G_ex(p*, y*) = G_r(p*, y*) = F_2L + ½ tr(S̃ Σ*) by 14.5.
- **Lower bound.** Take any exact-domain (p, y(·)). Set
  y'_F(n) = E_p[y_F(N) | N_F = n], which lies in the conditional constraint
  set because the constraint is linear.
  - S̃ ⪰ 0 gives E_p[½ yᵀS̃y] ≥ ½ c̄ᵀS̃c̄.
  - Jensen per molecule gives E_p[A_F(y_F) | N_F] ≥ A_F(y'_F(N_F)).
  - The linear term and the entropy are unchanged.
  - Hence G_ex(p, y) ≥ G_2L(p, y') ≥ F_2L. Minimising the left side gives
    F_ex ≥ F_2L. ∎
- **Exact decomposition.** The lower-bound chain at the exact optimum
  (p_ex, y_ex) splits F_ex − F_2L into three non-negative parts:
  - ½ tr(S̃ Cov(y_ex)), from averaging the quadratic;
  - Σ_F E[A_F(y_F) − A_F(E[y_F | N_F])], polarization that responds to the
    other molecules' actual charges;
  - G_2L(p_ex, y'_ex) − F_2L.
- **The bound needs the canonical ensemble.** With the mean-constrained
  product of the first draft, the two problems range over different
  configuration sets and the lower bound fails.

### 14.7 Measured (untrained heads, sic = 'none'; F in eV)

Checks on the same runs:
- The primal value rebuilt from (p*, y*) equals the dual to 3e-14 eV.
- The identity 14.5 holds to 3e-15 eV.

| system | R (Å) | θ | F_ex − F_2L | ½tr(S̃Σ*) | F_2L + ½tr(S̃Σ*) − F_ex |
|---|---|---|---|---|---|
| H₂O···H₂O⁺ (kT 0.01) | 4 | 0.502 | 0.0486 | 0.0582 | 0.0095 |
| | 6 | 0.639 | 0.1133 | 0.1319 | 0.0186 |
| | 10 | 0.785 | 0.0307 | 0.0335 | 0.0028 |
| | 30 | 0.923 | 0.0035 | 0.0035 | 0.0000 |
| H₂O···H₂S⁺ | 4–30 | 0.50–0.90 | ≤ 1e-9 | ≤ 1e-9 | ≤ 1e-9 |
| Na···Cl (Q = 0) | 8 | 0.654 | 1.1e-6 | 1.2e-5 | 1.1e-5 |
| | 9.00 | 0.676 | 0.0148 | 0.0226 | 0.0077 |
| | 9.43 | 0.685 | 0.0208 | 0.0238 | 0.0030 |
| | 10.0 | 0.696 | 0.0018 | 0.0078 | 0.0060 |
| | 12 | 0.732 | 1.6e-13 | 1.3e-12 | 1.1e-12 |

- The bound holds in every case.
- For the water pair the upper bound, the exact Coulomb energy of the
  two-level ensemble at no extra cost, is 5–6× closer to F_ex than F_2L is.
- At Na···Cl it is closer only at the crossing (9.0–9.43 Å). At 8 and 10 Å
  the lower bound is closer.
- Neither bound is uniformly best.

### 14.8 Physics: where averaging is right, and where it is not

- **Inside a bonded molecule averaging is physics, not just cost.** The
  electron is quantum-delocalised over the bonded atoms, which carry
  fractional charges. A classical integer ensemble on atoms would be wrong
  there. The atomic mean field (Mermin with the certified removal) is the
  appropriate model; the removal r stops it over-delocalising where convexity
  allows (§8a).
- **Between molecules with negligible electronic coupling**
  (|H_AB| ≪ kT and the reorganisation energy), charge is an integer on one
  molecule or the other. The classical canonical ensemble is the physics, and
  averaging is an approximation whose error 14.6 certifies.
- **At intermediate coupling** neither is right. The configuration set 𝒩 is
  exactly the diabatic basis of a charge-transfer Hamiltonian:

      H_{NN'} = E(N) δ_{NN'} + t_{NN'},     F = −kT ln tr exp(−H/kT)

  with t_{NN'} ≠ 0 only between configurations that differ by one electron
  moved between two molecules.
  - ESTABLISHED: the two-state Mulliken–Hush model (Mulliken 1952; Hush
    1967).
  - NEW-TO-TRACE: the diabatic energies E(N) are the exact canonical
    configuration energies of 14.1.
  - t → 0 recovers F_ex. The coupling t needs charge-transfer data
    (SPEC 22).

### 14.9 Consequences (proposals, not implemented)

1. **The bound as a certificate.** ½ tr(S̃Σ*) is available after every
   two-level solve at no extra cost: an a-posteriori bound on the averaging
   error, structure by structure.
2. **Exact active space.** By 14.5 the error lives only in fluctuating
   molecules. Enumerate 𝒩 exactly over the active molecules only, with every
   other molecule frozen at its integer charge but polarising in every
   configuration.
   - This is the exact convex program of 14.2 at the cost of |𝒩_active|
     constrained solves.
   - For m_active = 2 it costs the same as the two-level model.
   - It needs:
     - a free solver with one constraint per molecule;
     - per-configuration graph-attached Newton steps, for exact first and
       second derivatives of each stationary E(N);
     - a smooth or exponentially small rule for entering and leaving the
       active set.
3. **Quantum coupling.** The active configuration space is where
   H_{NN'} = E(N)δ + t enters. Averaging between spectators stays: for many
   molecules that fluctuate together, charge is band-like and the mean field
   is the appropriate model.

**Not proven here:** the bound for several partitions sharing c̄ (§12.6);
14.6 is per partition.

## 15. The exact canonical configuration ensemble — the default between molecules (`fragment_ensemble='exact'`) — NEW-TO-TRACE

§14 shows that averaging between molecules is forced only by cost; the exact
classical canonical ensemble is convex without any certificate (§14.2). This
section implements it, with averaging kept where it is physics: inside bonded
molecules (the atomic mean field) and, as an option, for band-like aggregates
(`fragment_ensemble='two-level'`, §12).

### 15.1 Definition (per partition 𝒫 of §12.6)

- **The operator.** W̃ = W − diag(r), with r the atom-level certified
  removal of §8a on the full W, so W̃ ⪰ 0. Φ_u carries +½ r_u n² (r = 0 for
  `sic='none'`).
- **Configuration energies**, one convex problem each:

      E(N) = min_{y : 1_Fᵀ y_F = N_F ∀F} Σ_u Φ_u(y_u) + bᵀy + ½ yᵀW̃y

- **Partition free energy:**

      F_𝒫 = −kT ln Σ_{N∈𝒞} w_N exp(−E(N)/kT)

- **Total energy and charges:**

      E = E_b0 + Σ_𝒫 π_𝒫 F_𝒫,     c = Σ_𝒫 π_𝒫 Σ_N P_𝒫(N) y_𝒫(N)

- **Properties:**
  - Per configuration, the inter-molecular Coulomb, the polarisation of every
    molecule (free and bound, through W) and the canonical constraint are
    exact. There is no block certificate and no θ.
  - For integer molecular charges on one-atom molecules the removal cancels
    exactly: −½ r n² in W̃ and +½ r n² in Φ.
  - A one-molecule partition (𝒞 = {Q}) is the atom-level model of §8a
    exactly.

### 15.2 One constraint per molecule (`free_solver.differentiable_constrained_solution`)

- **The dual.** D(z, μ) = −½zᵀW̃z + μᵀN + Σ_u Ω_u(ν_u) with
  ν = Gᵀμ − b − W̃z, where G is the 0/1 molecule membership.
- **Newton step**, by the Schur complement on μ:
  - S_μ = diag(Gx) − Bᵀ A⁻¹ B, with A = W̃ + W̃XW̃ and B = W̃XGᵀ.
  - S_μ is positive semidefinite.
- **Flat directions.**
  - A molecule deep in its dead zone has S_μ ≈ e^{−gap/kT} along μ_F. That
    direction is dropped (pseudo-inverse with the floor of §6).
  - Whenever that molecule's charge is off its target, μ_F is put back on its
    own constraint by an exact 1-D bisection.
- **Capacity bounds.** A molecule at a capacity bound (N_F = −|F| or
  Σ_u min(K, Z_u)) has every atom in its extreme state. It is fixed exactly
  and enters the others' problem only through its charges.
- **Derivatives.** One Newton step from the detached optimum s₀ makes E(N) a
  stationary value: s = s₀ − P g(s₀; θ), with the Newton operator
  P = (−Hess D)⁺ at s₀ **detached** and only the residual g carrying the
  graph. Since g(s₀; θ₀) = 0, dE/dθ = D_θ (envelope) and
  d²E/dθ² = D_θθ − g_θᵀ P g_θ, the exact Schur complement: every derivative of
  P would multiply g = 0. Its first and second derivatives are exact, as in §6.
  - Differentiating P instead (the code before 2026-10-08) runs `eigh`'s
    backward, whose 1/(λᵢ − λⱼ) is 0/0 at a degenerate spectrum: two identical
    molecules gave NaN gradients for a smooth energy.
  - The step starts from z = c, the primal charge. z is fixed only modulo
    null(W), and ∂D/∂W at z = c + n is ½ccᵀ − ½nnᵀ, against the primal ½ccᵀ.
    The shift moves ν by W(c − z₀), which is the residual.
- **Checked:**
  - against an autograd-Newton maximisation of the same dual: 1e-15–5e-14 eV,
    down to kT = 0.002 eV;
  - first derivatives against finite differences: 1.7e-11;
  - force-loss-type second derivatives against finite differences:
    1.6e-10 relative, with fractional, responsive charges.

### 15.3 Which configurations: canonical two-level screening, smooth inclusion

- **Screening.** The canonical two-level ensemble of §12 and §12.13, solved
  per partition (weight 1, π-weighted window test), gives
  log p₂(N) = Σ_F (−e_F(N_F)/kT) + A_𝒫/kT exactly.
- **𝒞** = {N : log p₂(N) ≥ ln 1e-12}. It is found by depth-first search,
  pruned exactly with the largest attainable remaining sum; the test checks
  the result against brute force.
- **Inclusion weight.** w_N is a C² quintic in log p₂, equal to 0 at 1e-12 and
  1 at 1e-10, so a configuration enters and leaves 𝒞 without a jump.
- **Error.** Configurations with p₂ below 1e-12 are dropped, and those in the
  band are down-weighted. The cost is of order kT × (their exact
  probability):
  - negligible unless the screening misjudges a configuration's energy by
    more than about 25 kT;
  - the screening error is bounded by §14.6, measured ≤ 0.13 eV.
- **Second derivatives.** They pass exactly through E(N) and the soft-min.
  Through w_N they are approximate only inside the band (weight ≤ 1e-10).
- **More than `fragment_max_configurations`** (512) selected configurations is
  refused, never truncated silently. In that regime many molecules fluctuate
  together, charge is band-like, and `'two-level'` is the physically
  appropriate model.

### 15.4 Partitions mixed linearly

- **The choice.** E = E_b0 + Σ_𝒫 π_𝒫 F_𝒫, each partition solved exactly and
  independently. §12.6 instead coupled partitions through a common mean
  charge, which is another average.
- **Why it is valid.** Both are C² (π is a polynomial in the C² bond
  switches), and both are exact where every s_b ∈ {0, 1}. They differ only
  inside a bond's switching shell. This is a modelling choice (NEW-TO-TRACE),
  stated as such.

### 15.5 Measured (untrained heads)

| system | exact ensemble − exact target | two-level − target | hole ⟨N_A⟩ (exact) |
|---|---|---|---|
| H₂O···H₂O⁺, kT 0.01, 4 Å | 9.2e-14 eV | −0.0486 | 0.873 |
| 6 Å | 9.4e-14 | −0.1133 | 0.917 |
| 10 Å | 8.7e-14 | −0.0307 | 0.669 |
| 30 Å | 8.9e-14 | −0.0035 | 0.518 |
| H₂O···H₂S⁺, 4–30 Å | ≤ 1.8e-14 | ≤ 1e-9 | ⟨N_H₂S⟩ = 1.000 |
| Na···Cl, Q = 0, 2.36–100 Å | ≤ 3.5e-14 (crossing 9.43 Å: 7e-17) | crossing −0.021 | q_Na = exact to 1e-9 |

- **Localisation.** The averaged ensemble put the hole half on each water
  (⟨N_A⟩ ≈ 0.52–0.54 at 4–10 Å). The exact ensemble localises it (0.87–0.92
  at 4–6 Å), and the enumeration confirms the marginals to 1e-8. This is the
  charge localisation that averaging erases.
- **One molecule:** exact ensemble − atom-level model = 8e-16 eV.
- **Au₂/MgO** (110 atoms, periodic, four training structures):
  - 1–3 configuration solves per structure;
  - one training step 3.5–4.1 s, against 3.4–4.6 s for `'two-level'` (2
    threads, beside a 6-thread training run);
  - energies equal to the two-level ones to 7e-9 eV, because with untrained
    heads Au₂'s charge fluctuates with probability 2e-10 and there is nothing
    for averaging to get wrong.

### 15.6 Tests (`tests/redox/test_redox_fragments.py`)

**What is checked.**
- The exact ensemble equals the independent enumeration in energy (≤ 1e-9 eV)
  and in hole marginals (≤ 1e-8): water at 4, 6 and 30 Å, and H₂S.
- Na···Cl at the crossing to round-off, energy and q_Na.
- One molecule equals the atom-level model.
- The multi-constraint solver against a reference, including the dead zone at
  kT = 0.002.
- A molecule at its capacity is fixed exactly.
- The configuration search is complete (against brute force; it prunes 20 of
  27).
- The inclusion weight is C².
- The mirror identity on both ensembles.
- Forces and force-loss gradients against finite differences: shared hole,
  breaking bond, periodic.
- **Values against an independent enumeration** (autograd-Newton dual per
  configuration; capacity-bound molecules fixed exactly, with independent
  code):
  - Two partitions of a breaking O–H bond (π = 0.527 / 0.473, default
    certified removal): 2e-13 eV. The split partition is a genuine mixture,
    OH⁺ + H⁰ 0.787 and OH⁰ + H⁺ 0.213. A first reference that skipped
    capacity-bound configurations missed the H⁺ one by 6.3 meV; the module was
    right.
  - Dications (max_charge = 2; H has no +2 state) for two waters at Q = +2:
    4e-15 eV, with +1 on each water.
- **Permutation invariance:** atoms relabelled inside and across molecules,
  molecule order swapped. E is unchanged to round-off and the forces permute.
- **The inclusion band at the training kT = 0.1 eV.** For H₂O···H₂S⁺ the
  water-hole configuration has screening probability 5.9e-11 at 4 Å,
  3.4e-11 at 6 Å and 1.4e-12 at 12 Å, so the band is exercised in ordinary
  geometries.
  - Inside it (6 Å), forces and the force-loss parameter gradient agree with
    finite differences.
  - At the separation R* where the probability equals 1e-12 (found by
    bisection), E has second differences ≤ 1e-12 eV and dE/dR changes by
    ≤ 1e-6 eV/Å over 2e-6 Å.

**Planted bugs, each caught:**
- (M50) configurations on W, not W̃;
- (M51) search pruning with the smallest remaining weight;
- (M52) a capacity-bound molecule at state 0;
- (M53) the Schur complement's sign flipped;
- (M54) configurations without the Boltzmann factor;
- (M55) the one-molecule partition on W;
- (M56) a fixed molecule without its own Coulomb;
- (M57) partitions summed without π;
- (M58) every free molecule in one constraint group.

### 15.7 Still outside this model

- **Coherent hopping** between configurations. The configuration set 𝒞 is
  the diabatic basis of H = E(N)δ + t (§14.8); t needs charge-transfer data,
  SPEC 22.
- **Spin.**
- **More than `fragment_max_configurations`** simultaneously fluctuating
  configurations: refused (or `'two-level'`).


## 16. Free-charge widths: what the mean-field sector allows (2026-10-09)

- The free sector's on-site curvature is the fraction h of the atomic hardness
  U_Z = IE − EA held as classical Hartree energy. A convex integer ladder
  (exact isolated-ion energies) needs h·U_Z < U_Z.
- Interactions between sites are geometry. Compact (ionic) pair charges in an
  ionic solid make the Gram operator indefinite for every allowed h: formal
  MgO has λ_min = −6.6 eV/e² at h = 1.
- So the mean-field free sector requires σ ≥ σ_H, the hardness rule's boundary.
  Localised ionic charges belong to integer configurations (audit step C2).
- `RedoxConfig(width='ionic')` is refused.
- The pieces for C2 are in place:
  - `heads.IonicFreeChargeWidth` (Shannon crystal radii, σ = √(2/5)·R);
  - `bound.eliminate_bound_sector(..., sigma_self=)` (pair and on-site widths
    separate);
  - `bound.periodic_free_charge_blocks_ewald`, through
    `trace_electrostatics.lattice.free_charge_lattice_blocks`.
- Tests: `tests/redox/test_redox_free_width.py`.


## 17. Coherent charge transfer: the diabatic Hamiltonian over configurations (audit step C2) — NEW-TO-TRACE

Status (2026-10-09): 17.1 is implemented (`diabatic.py`, tests/redox/test_redox_diabatic.py, 15 tests).
17.2 onward are in progress, test first.

### 17.1 Free energy of a diabatic Hamiltonian (`diabatic.canonical_free_energy`)

- **Definition.** Over configurations N with exact polarisable energies E(N) (§15),
  H = diag(E_N) + T and F = −kT ln Tr e^{−H/kT}.
- **Limits.**
  - T = 0 gives the exact ensemble of §15, F = −kT ln Σ_N e^{−E_N/kT}.
  - Two configurations give the Marcus–Hush Hamiltonian. Its Robin–Day class
    (II iff λ > 2|t|, for the symmetric pair) emerges from the learned E and t; it
    is not imposed.
- **First derivative.** dF/dH = ρ = e^{−H/kT}/Z, the Hellmann–Feynman forces.
- **Second derivative.** Daleckii–Krein with the normalisation, in the eigenbasis:
  dρ_ij = Γ_ij dH_ij + (1/kT) p_i δ_ij Σ_k p_k dH_kk, with
  Γ_ij = −(p_lo/kT) ψ(|λ_i − λ_j|/kT) and ψ(y) = −expm1(−y)/y. It is finite and
  exact at degenerate spectra. Third derivatives are refused.
- **Checks.**
  - Closed forms: t = 0, the two-state eigenvalues, the curvature 2λ − λ²/|t| at
    x = ½.
  - First and second derivatives against finite differences, including exact
    degeneracy.
  - Planted bugs caught: the normalisation term, the sign of Γ_ii, and the wrong
    weight in Γ.

### 17.2 Configurations as determinants (in progress)

- **Orbitals.** Unit u with states n ∈ {−1, …, +K_u} holds K_u + 1 local
  spin-less orbitals, one per charge step. Configuration n fills the lowest
  o_u = K_u − n_u of them.
- **Ordering.** Orbitals are ordered by (unit, local index).
- **One-electron moves.** A move takes an electron from unit a to unit b:
  n_a → n_a + 1 and n_b → n_b − 1, i.e. a's top occupied orbital to b's lowest
  empty orbital. The matrix element is ⟨n′| c†_q c_p |n⟩ τ_ab, with the
  Jordan–Wigner sign (−1)^{N_<(p)} (−1)^{N′_<(q)}.
  - N_<(p) counts the orbitals below p occupied in n.
  - N′_<(q) counts those below q after p is emptied.
  - This is the one-body Slater–Condon rule (ESTABLISHED). The sign matters as
    soon as two carriers exist: Au₂/MgO:Al has three donors (3 Al, measured on
    every doped structure).
- **Spin.** It is absent, as in §15, and stated as a limitation: no spin
  degeneracy factors.

### 17.3 Configuration energies and the search (`diabatic.configuration_energies`, `search_configurations`)

- **Configurations.** Integer charges on every atom: n_u ∈ {−1, …, K_u},
  Σ_u n_u = Q.
- **Energy.** E(n) = Σ_u ê_u(n_u) + bᵀn + ½ nᵀWn, the closed form of a fixed
  molecule. The on-site self term cancels for integer n, so W need not be
  positive definite.
- **Search.**
  - A greedy start, then single-move descent: a move is accepted only if the
    exact energy drops, so it terminates.
  - Then a flood fill through one-electron moves inside the window. Move
    validity is checked exactly.
  - It is exact when the window set is connected through in-window moves.
    Beyond the cap it refuses.
- **Checks.**
  - Against brute force over all 729 vectors of 6-unit problems: 48 instances,
    windows of 2.5 and 6 eV (14–28 configurations per 6 eV window).
  - Move energies against exact differences.
  - Termination under deliberately wrong move estimates. That once hung; it was
    found by a planted bug.

### 17.4 The module: `fragment_ensemble='diabatic'`

- **What it computes.** The canonical free energy of H = diag(E(n)) over the
  searched configurations. Charges are ρ's diagonal times n. There is no mean
  field anywhere.
- **Checks.** With couplings off it equals brute-force enumeration over every
  integer atomic charge vector, to 1e-9 relative (Q = 0, ±1); forces against
  finite differences.

**Probe, not a criterion (2026-10-09): the trained step-1 model at epoch 13 on
three doped Au₂/MgO:Al structures.**
- 333–367 configurations lie within 1 eV, almost all Al⁺ + O⁻: the electron sits
  on a neighbouring oxygen, 0.00 to −0.04 eV from all neutral.
- One electron Al → Au costs +4.59 to +5.51 eV, and two cost +11.8 to +12.2 eV.
  DFT puts the electrons on Au₂.
- **Diagnosis:** the atomic anchors (B1). In MgO, O is already O²⁻, a closed shell
  that cannot accept another electron. Al's donor electron is the one Al³⁺ has in
  excess over the Mg²⁺ it replaces, not a gas-phase electron (IE 5.99 eV).
- **Fix:** formal oxidation states with the closed-shell capacity (redoxSQE,
  ESTABLISHED; audit C4). These need compact ionic charges (B3), which need
  integer configurations: this section.
- **Couplings (§17.5) cannot help here.** Al–Au distances are 10.7–14.7 Å
  (measured), so direct couplings are negligible; the transfer is band-mediated.
  That corrects the audit's claim that C2 fixes the training signal (B6): it does
  so only for nearby units.

### 17.5 Formal oxidation states in the diabatic mode (`anchor='formal'`; audit C4) — in progress

Motivated by P5 (benchmarks README): integer charges with compact sizes, selected by energy,
reproduce formal ionic chemistry with no assignment.

- **Ladder** of unit u (element Z), states n ∈ [L_Z, K_Z]:

      ê_u(n) = E_Z(n) + Δ_u(n) − ½ S_0(σ_u) n²

  - **E_Z(n)**, the exact isolated-ion energy relative to the neutral atom (ESTABLISHED data):
    - Σ_{k≤n} IE_k for n > 0 (NIST successive IEs);
    - −EA₁ for n = −1;
    - −EA₁ + A₂ for n = −2, only where a second-electron cost A₂ is tabulated. O: A₂ = 8.75 eV,
      a Born–Haber estimate in the range 8.08–8.75 eV. No other −2 state, and no state beyond the
      data.
  - **Δ_u(n) = B·tanh(r_u(n)/B)**, B = `chi_bound`. These are learned environment shifts of each
    integer state: ChargeStateHead outputs r_u, zero at initialisation.
  - **S_0(σ)** = gram_self_energy(σ), the ISOLATED Gaussian self-energy. Only that is removed, so
    an isolated ion has E_Z(n) + Δ_u(n) exactly. The own-image interactions of a periodic cell are
    lattice energy and stay. (The mean-field anchors remove diag(J) including images; that is the
    mean-field convention and is not used here.)
- **Widths** σ_u: `IonicFreeChargeWidth` (Shannon crystal radii) for elements in the table. Other
  elements take the QEq width √2·K_E/(√π·(IE − EA)) as stated in P5.
- **Energy** E(n) = Σ_u ê_u(n_u) + bᵀn + ½ nᵀWn with W from the bound-sector elimination at σ.
  W need not be positive definite (§17.3), so neither the mean-field PD check nor the
  ladder-convexity refusal applies.
- **Free energy:** canonical, over the searched configurations.
- **Limitations, stated:**
  - no spin;
  - no −2 state without a tabulated A₂;
  - the on-site free–bound coupling uses σ_u (no separate self width: in configuration space the
    self term cancels exactly).

**17.5 status (2026-10-09): implemented** (`RedoxConfig(anchor='formal', fragment_ensemble='diabatic')`).

- **Checks** (tests/redox/test_redox_diabatic.py):
  - the ladder against NIST hand values (Al³⁺ 53.262 eV, O²⁻ −1.461 + 8.75, Mg⁻ +0.42);
  - an isolated ion through the whole module equals E_Z(n) to 1e-9 (Al³⁺, O²⁻, Cl⁻, Mg²⁺);
  - the module equals the enumeration over every configuration of an O/Mg/Na/Cl fragment;
  - forces against finite differences with random learned shifts;
  - `'formal'` is refused outside the diabatic ensemble.
- **Search limitation, measured.** On a 6-atom cluster with two atoms 0.29 Å apart (an
  unphysical geometry that the cluster helper produced), the search from all neutral stopped in a
  local minimum 12.8 eV above the true one. The flood fill does not cross barriers higher than
  the window.
- **On physical systems** (every P5 system, bare and screened), the searches from all neutral and
  from the formal guess found identical minima. A second, formal-guess start would need hand-
  assigned charges, so it is not used. Geometries with atoms far closer than bonding distances are
  outside the search's stated domain.

- **Cost (measured 2026-10-09, one training sub-step on the 110-atom Au₂/MgO cell, 2 threads):**
  the formal mode takes 2.21 s, against 2.26 s for the current mode.
  - The first version took 17.2 s: it routed the compact ionic widths through the plane-wave
    free-charge blocks (about 1.4·10⁵ k-vectors).
  - It now takes the Ewald route (`sigma_self = sigma_free`; the on-site correction is then exactly
    zero, so the physics is identical).

### 17.6 One-electron couplings τ_uv (`RedoxConfig(couplings=True)`) and the k-fold search

- **Coupling.** τ_uv = Σ_{edges u→v} [Σ_k c_k(x_u, x_v) g_k(r)] f(r/r_cut).
  - c comes from an MLP of the symmetric invariants (x_u + x_v, x_u ⊙ x_v), so τ_uv = τ_vu exactly.
  - g_k are Gaussians; f is the C² quintic switch.
  - Periodic images are summed (a Γ-point Bloch sum); self-image edges are excluded.
  - Form: the two-centre hopping integral (Slater–Koster; ESTABLISHED), learned from TRACE invariants
    (NEW-TO-TRACE).
  - Initialised small but non-zero: F is even in τ, so a zero coupling has zero gradient.
- **Hamiltonian:** H = diag(E) + Σ_moves sign·τ[donor, acceptor] (`diabatic.diabatic_hamiltonian`).
- **Checks:**
  - symmetry, invariance, and the C² cutoff;
  - zero couplings reproduce the uncoupled module exactly;
  - a symmetric Na₂⁺ gains exactly −kT ln cosh(τ/kT), with +½ on each Na;
  - forces against finite differences;
  - the signs of the Hamiltonian against the Fock-space construction with two carriers. A dropped
    sign is caught there; the one-carrier module tests could not see it.
- **k-fold moves in the search.** Moving k electrons a → b at once has the exact cost
  ΔE = [ê_a(n_a+k) − ê_a(n_a)] + [ê_b(n_b−k) − ê_b(n_b)] + k(f_a − f_b) + ½k²(W_aa + W_bb) − k² W_ab,
  with f = b + Wn.
  - This covers ion hops. Na₄²⁺ on a 3 Å square has four Na⁻ + 3 Na⁺ minima within 0.36 eV, connected
    only by two-electron moves; the single-move search found one.
  - On Au₂/MgO:Al with the step-2 model at epoch 1, the single-move search returned 15 configurations
    where the k-fold search finds 38 and 25 (2 of 3 doped structures).
  - Cost is unchanged within noise.

### 17.7 The electronic response of the formal model is the bound sector's (`RedoxElectrostatics.bound_state`)

**Identity (exact for this model; NEW-TO-TRACE as a training route).** An applied field E (open
boundaries) shifts each integer configuration n by its free dipole and by the bound response:

    E_n(E) = E_n(0) − d_n·E − ½ E·α_b·E,

with **the same α_b for every n**. W and b do not depend on E; the field enters the free sector only as
−Σ_u n_u r_u·E. The canonical free energy F = −kT ln Σ_n e^{−E_n/kT} then has

    −∂²F/∂E∂E = α_b + Var_p(d)/kT.

- The second term is weighted by the populations above the ground configuration, ~e^{−gap/kT}. For a
  gapped ionic crystal (formal MgO: excitations of several eV) it is far below round-off.
- So with `anchor='formal'` the model's electronic polarizability, and its ε∞ in a crystal, is the
  bound sector's. `bound_state` returns that sector as a `trace_electrostatics.ElectrostaticState`: the
  same input resolution as `forward`, every bound fragment neutral, the dense exact solve. Its energy
  is `energy_bound`.
- The mean-field anchors are not covered: their continuous free charges respond linearly too. Their
  refusal stays (`state()` raises; `train.py` refuses the loss).
- Under periodic boundaries an applied field is refused for the free charges (branch-dependent
  −Σ n_u r_u·E). The identity is therefore tested with open boundaries; ε∞ follows from the same α_b.

**Checks (`tests/redox/test_redox_bound_state.py`):**
- `bound_state` energy = `energy_bound`, open and periodic (1e-10 relative);
- central differences of the FULL formal free energy in a field (kT = 0.02 eV) against α_b, both
  routes: 2.4·10⁻¹¹ relative. The free charges are the formal integers (O −2, Mg +2, Na +1, Cl −1), so
  the field does couple to them (−∂F/∂E_x = −0.416 e·Å);
- the bound dielectric tensor is differentiable in the head parameters;
- flashace routes observables to `bound_state` for formal anchors only.

**Training (train.py):**
- `dielectric_weight` is allowed with `anchor='formal'`;
- `electrostatics.dielectric_prior: {file, weight}` gives ε∞ labels on 3-D periodic structures that
  carry no energies (bulk MgO for the Au₂/MgO data). They are evaluated once per optimizer step on a
  forward of their own, and enter no energy, force or validation loss.
- Tests: `tests/redox/test_redox_dielectric_train.py`. A dielectric-only run lowers its RMSE every
  epoch. With the prior at weight 1 the validation RMSE goes 24.9 → 20.7 over 4 epochs, against
  28.3 → 28.1 at weight 10⁻¹².

**Why it is needed (measured).** Every Au₂/MgO model so far over-screens: ε∞(MgO) = 7.5 (step 1), and
8.0 and 6.2 (steps 2 and 3 at epoch 2), against 2.95 measured (Sun, Allen, Stahnke, Jacobsen & Homes,
arXiv:0704.2267, Table I, 295 K; their Table II compiles 2.96). Over-screening alone produces Al → Au
"transfer" in the formal model at epoch 2; at ε∞ = 3 it is gone (experiments/benchmarks_4g/README.md,
amendment to steps 2 and 3).

**Amendment (2026-10-10): couplings, and the one-configuration condition.**
- With couplings (§17.6) the field also mixes configurations coherently. Then

      −∂²F/∂E∂E = α_b + Var_p(d)/kT + α_CT,   α_CT = 2 Σ_k |⟨0|d̂|k⟩|²/(E_k − E_0)   (kT → 0),

  a charge-transfer polarisability that `bound_state` does not contain. It is visible in a coupled Au₂:
  α∥ = 11.2 Å³ against α⊥ = 0.53 Å³ (step-5 model).
- So `bound_state` is the model's whole response **exactly when the window holds one configuration**:
  then the thermal and coherent terms vanish identically.
  - The response priors of `train.py` (`dielectric_prior`, `polarizability_prior`) check this per
    structure at every evaluation (`RedoxResult.iterations == 1` in the diabatic mode) and refuse
    anything else.
  - Bulk formal MgO and an isolated neutral atom have one configuration (measured on the step-5 model);
    Na₂⁺ has two and is refused, with or without couplings.
  - **[Superseded 2026-10-19 by §30:** with several configurations the priors now add the ensemble's
    response from its exact record and refuse only a structure whose record is not exact; Na₂⁺ trains (§30
    tests). The two bullets above are kept as written.**]**
- **`polarizability_prior: {file, weight}`:** open-boundary structures with a polarizability label
  (Å³). For example the Au atom, 36.68 ± 0.78 a.u. (relativistic CCSD(T), arXiv:2410.01493).
- Tests: `tests/redox/test_redox_polarizability_prior.py`.
  - The trained checkpoint's α of the prior atom from central differences of its TOTAL energy equals
    the bound-state α it trained on (to 1e-7).
  - Several configurations are refused.

### 17.8 Electronegativity enters once (`RedoxConfig(bound_electronegativity='none')`, formal anchors) — NEW-TO-TRACE

**Physics.**
- With `anchor='formal'` the exact isolated-ion ladders E_Z(n) (NIST ionisation energies and electron
  affinities) carry each element's electronegativity and hardness.
- The bound sector's own heads χ (and χ_Θ) add a second, independent drive: the bound solution y₀ with
  no free charge, its energy E_b0, and its potential b = A y₀ at every free site.
- In the mean-field model that drive was the charge physics. In the formal model it counts
  electronegativity twice, and no observable pins it.
- With 'none' the bound sector is linear response alone: polarisation and back-donation driven by the
  formal charges' field through W.

**Exact consequences** (χ, χ_Θ enter U_b only linearly):
- K, A, J and W are unchanged, and so is every response (polarisability, ε∞);
- with no applied field, y₀ = 0, hence E_b0 = 0 and b = 0;
- for every configuration n, E_learned(n) − E_none(n) = E_b0 + b·n;
- `bound_state` applies the same zeroing (its energy is `energy_bound`), through
  `trace_electrostatics.energy.state_from_assembly`. That is the second half of `electrostatic_state`,
  factored out bit for bit.

**Why (measured, Au₂/MgO:Al, 12 doped upright test structures).** E(Au⁻ on the near Au) − E(Au⁻ on the
far Au), split exactly into its four terms (eV):

| model | ΔE | δ | b | Coulomb | self |
|---|---|---|---|---|---|
| step 3 | +0.139 | −0.021 | +0.156 | +0.111 | −0.107 |
| step 5 | −0.431 | −0.054 | −0.389 | +0.205 | −0.194 |

- DFT puts the electron on the far Au (Hirshfeld +0.040 / −0.239).
- The unconstrained b flipped sign between the runs and decides the site in step 5.

**Tests:** `tests/redox/test_redox_bound_electronegativity.py` (8).
- E_b0 = b = 0 exactly; W and ê bitwise unchanged.
- The per-configuration identity to 1e-12.
- The same bound-state functional and polarisability.
- An applied field still polarises.
- Refused without formal anchors (including fragments='none').

**Status:** built and tested; not yet used in a run. Its run waits for step 6 to be scored (one change
per run).

### 17.9 Core + carrier free charges: no ionic collapse (`RedoxConfig(free_charge_model='core_carrier')`) — NEW-TO-TRACE

**The defect (exact).** Moving one electron between units u and v changes a configuration's energy by
a second difference

    Δ²E = E(n + d) + E(n − d) − 2E(n) = [U_u(n_u) − S₀,u] + [U_v(n_v) − S₀,v] + dᵀWd,   d = e_u − e_v

- U(n) = E_Z(n+1) + E_Z(n−1) − 2E_Z(n) is the exact ladder curvature.
- S₀ = √2·K_E/(√π σ) is the Gram self-energy subtracted for the free charge.
- dᵀWd ≥ 0, because W is a Gram Schur complement.
- With one compact (ionic) width per unit, S₀ exceeds the smallest curvature for every element:
  Au 12.03 > 6.92 eV; Al 26.9 > 5.55; Mg 21.1 > 7.39; O 14.4 > 10.2. So Δ²E can be negative
  (spontaneous disproportionation).
- **Measured consequences:**
  - an isolated Au₂ with untrained heads is Au^±0.76–0.86 at 1.9–2.5 Å, with Δ²E −3.8 to −4.9 eV;
  - the step-6 test outliers are compressed Au₂ with Au⁺Au⁻ cancelling the repulsive wall (Au force
    errors up to 17.5 eV/Å);
  - step 2's epoch-2 Al⁺ + Al³⁺.

**The construction.** Each unit's integer charge n = f + c is split into:
- **f, the frame:** the main-group formal ion (`constants.formal_frame`; groups 1, 2, 13 → +1, +2, +3;
  15, 16, 17 → −3, −2, −1; every other element 0; ESTABLISHED ionic-model oxidation states), on the
  compact width;
- **c, the carrier,** on a diffuse width set by the exact ladder:

      σ_car = √2 K_E / (√π U_min),   U_min = min over the unit's interior states of U(n)

  (`module.carrier_widths`; no parameter). Then S₀(σ_car) ≤ U(n) at every state the carrier passes
  through, and **Δ²E ≥ 0 for every one-electron move, at every distance, under any screening.**

**Exact reduction to the existing machinery** (`bound.eliminate_core_carrier`; shell model of
ion core and valence charge: Dick & Overhauser, Phys. Rev. 112, 90 (1958), ESTABLISHED).
- With x = [f; c] and the frames fixed, the free-sector interaction is

      b_n·n + ½ nᵀ W_cc n + C,   b_n = u0_c + W_fcᵀf − W_cc f,
      C = u0_f·f + ½fᵀW_ff f − (u0_c + W_fcᵀf)·f + ½fᵀW_cc f,

  so the configuration search, the couplings and the canonical free energy are unchanged.
- C is a free-sector constant (`BoundElimination.constant`); `energy_bound` stays the bound sector
  alone.
- ê_u(n) = E_Z(n) + δ_u(n) − S_u(n), with the exact two-Gaussian isolated-ion self-energy

      S_u(n) = ½f²S₀(σ_core) + f·c·k(√(σ_core² + σ_car²)) + ½c²S₀(σ_car).

**Exact identities, tested (`tests/redox/test_redox_core_carrier.py`, 15):**
- σ_car = σ_core reduces to the single-width model bit for bit, for any frames;
- the all-frame configuration (bulk formal MgO) has the single-width energy exactly, so the compact
  Madelung energy is untouched;
- isolated ions keep E_Z(n) + δ exactly;
- forces are −∇E;
- Δ²E ≥ 0 for compressed Au₂ (+1.19 to +1.65 eV at 1.9–2.5 Å), where the single width gives −3.8 to
  −4.9 eV;
- the carrier widths match the ladder (Au 1.661, Mg 1.555, O 1.125, Al 2.069 Å);
- H and C keep their hardness width;
- the ε∞ prior trains with nonzero frames.

**Status of 17.9 (2026-10-15, after step 7): do not use without an exclusion term.**
- Trained on Au₂/MgO:Al, the core + carrier model lost the electron transfer altogether: all Al stayed
  Al²⁺, T5 fell from −537 to −80 meV, and the forces were worse.
- **Hypothesis:** a diffuse carrier overlapping compact closed-shell anions gains a charge-penetration
  stabilisation of several eV, which the real system forbids by Pauli exclusion; TRACE has no such
  term.
- The exact identities and the Δ²E ≥ 0 theorem stand. The construction is incomplete physics, kept
  opt-in.

### 17.10 Free-charge widths by formal frame (`RedoxConfig(free_width_rule='frame')`) — NEW-TO-TRACE

**Rule.**
- Elements with a nonzero formal frame (`constants.formal_frame`: the ionic-model cations and anions)
  keep the compact ionic (Shannon) width with its learned factor (`IonicFreeChargeWidth`).
- Neutral-frame elements (metals and covalent elements: Au, Ag, C, H, ...) carry their free charge on
  the fixed ladder width σ_car = √2 K_E/(√π U_min) (17.9). For Au, Ag, C and H, U_min = IE − EA, so
  σ_car is the hardness width.

**Why.**
- *Neutral frames:* S₀ = U_min, so a one-electron move between two such units has Δ²E = dᵀWd ≥ 0. No
  ionic collapse (an isolated Au₂ with the ionic width is ±0.76–0.86 e at 1.9–2.5 Å, 17.9).
  - The learned factor in [½, 2] is dropped for them: below 1 it would break S₀ ≤ U. That includes
    H and C, whose 'ionic'-rule width is the QEq width times that factor.
- *Ionic frames:* compact widths keep the Madelung energy (formal MgO 99.35 % of point-charge).
  - Inside a closed-shell anion cage a diffuse carrier would penetrate the compact anion charges and
    gain a spurious energy that Pauli exclusion forbids.
- **Measured (step-6 weights, counterfactual, 6 doped Au₂/MgO:Al structures):**
  - only Au's carrier diffuse: the full transfer, −1.00 e on Au₂ in 6/6;
  - Al's carrier diffuse (σ 2.07 Å, in an O²⁻ cage): no transfer (0.00);
  - Mg's and O's carriers diffuse: more than 512 configurations within 2.5 eV (Mg⁺ artificially cheap).

**Tests:** `tests/redox/test_redox_frame_width.py` (7).
- The ionic frames' widths are untouched; neutral frames get the fixed ladder width (Au 1.661 Å).
- Δ²E ≥ 0 for Au₂ at 1.9–2.5 Å.
- Formal MgO's energy is identical under both rules.
- Forces are −∇E.
- Refused without formal anchors.

## 18. Exclusion (Pauli) of charge-state densities by closed shells (`RedoxConfig(exclusion=True)`) — NEW-TO-TRACE

Theory and evidence: docs/ELECTROSTATICS_THEORY_2026-10-15_exclusion.md. Module: `trace_redox/exclusion.py`.

    Pi_u(n) = K Σ_{v: f_v ≠ 0} N_v s(R_uv) [ N_u(n) S(ρ_u^(n), ρ_v; R_uv) − N_u(f_u) S(ρ_u^(f_u), ρ_v; R_uv) ],
    ê_u(n) → ê_u(n) + Pi_u(n)

- **Densities** (ESTABLISHED asymptotics, Katriel & Davidson 1980):
  ρ(r) = (r² + κ⁻²)^β e^(−2κr)/Z, κ = √(2I), β = (q + 1)/κ − 1. This is exact in the tail and finite at
  the nucleus.
  - Carrier unit u (frame f_u ≥ 0): I is the exact gas removal energy of state n, and N_u(n) the
    electrons beyond the closed core.
  - Closed-shell neighbour v (f_v ≠ 0):
    - an anion's valence shell (N = Z − f − core), with an in-crystal removal energy I_eff(Z) in
      (4, 16) eV (learned, initial 9 eV; gas-unbound);
    - a cation's outer core shell (the next ionisation energy).
- **Energy:** K·S (density-overlap model, ESTABLISHED), with K = 10 Ha × exp(ln 4 tanh k).
- **Zero at every frame**, so formal MgO is unchanged exactly.
- **An extra electron with no bound level in vacuum** (Mg⁻) is not a localised configuration. Its state
  is removed (+∞) everywhere.
  - The first version gave it zero exclusion. Measured on step-6 weights, one Mg⁻ then took all three
    donor electrons.
- **Overlap:** prolate-spheroidal quadrature with 24 Laguerre × (8 × 6) Legendre nodes. Worst relative
  error 6.4e-7 against an adaptive integral.
- **Tests** (`test_redox_exclusion_overlap.py`, 28; `test_redox_exclusion.py`, 10): the overlap against
  an independent integral; normalisation; symmetry; d/dR; zero at frames; isolated atoms; Al²⁺ in an O²⁻
  cage = the independent value (8.0888 eV at defaults); Au⁻ atop O pays more than far from it (≥ 0.4
  eV); a hole relieves; closed shells only; formal MgO unchanged; forces −∇E; smooth at the cutoff; the
  unbound rule.
- **Counterfactual (step-6 weights, untrained exclusion).** It moves the doped-upright electron to the
  far Au (near/far −0.061/−0.964; DFT +0.040/−0.239) and makes Al Al³⁺. It overshoots ionicity
  (undoped upright Au ±0.955; DFT ±0.12), consistent with the missing exchange polarisation (theory
  §4). On the direct quadrature the cost was 2.5× per evaluation; §18.3's tables remove it.

### 18.3 Tabulated overlaps (`Exclusion(tabulate=True)`, the default) — numerical method, same physics

S depends on two numbers only: R and, for an anion neighbour, κ_v = √(2 I_eff). Carrier states and cation
cores are fixed by the exact ladder.

- **Expansion:** log S as a Chebyshev series in R ∈ [0.2 Å, r_cut + 1 Å] (96 terms) and, for an anion
  neighbour, in κ_v over I_eff ∈ [4, 16] eV (20 terms).
  - Coefficients come from the quadrature's values at Chebyshev–Gauss nodes; S′ and S″ are the
    derivative series.
  - log S is analytic there (the quadrature's nodes scale with R), so the series converges spectrally.
- **Measured against the quadrature** (`test_redox_exclusion_table.py`):
  - S: ≤ 1e-7 relative (pre-registered; measured ~1e-12);
  - dS/dR, d²S/dR²: ≤ 1e-6 relative;
  - d ln S/d ln I_eff: ≤ 1e-6 absolute. (dS/dI_eff changes sign near 1.8 Å, so a relative criterion is
    undefined there.)
  - Exclusion energies: ≤ 1e-7 relative to `tabulate=False`.
- **Outside the interval,** values come from the quadrature directly and derivatives are refused.
- **Cost** (110-atom doped structure, 5260 edges): direct quadrature 8–11 s per structure (forward,
  forces and force-loss backward); tabulated 0.03 s, after a one-time build per process and pair type.
- **History.** A first table used bicubic Hermite interpolation, which is C¹ only. S″ then jumped at the
  nodes and the forces of §18.4 failed their finite-difference test. Step 9 (benchmarks_4g) trained on
  that version; it uses S alone, so its energy is C¹ and its values agree with the Chebyshev table to
  ≤ 1e-7.

### 18.4 Relaxed exclusion (`RedoxConfig(exclusion_relaxation=True)`) — NEW-TO-TRACE

The frozen §18 overestimates both the cost of an extra electron and the relief of a hole. Each state's
frontier density polarises away from the closed shells, and that relaxation already lowers the frame
state's exclusion. Measured for Au⁰ atop O²⁻ at 2.2 Å (K = 10 Ha, I_eff = 9 eV, α = 5.435 Å³): the frozen
1.35 eV relaxes by 0.50 eV at second order (0.60 eV in the full nonlinear minimum). So the relief of
ionising to Au⁺ is 0.85 eV, not 1.35 eV.

**Model.** Rigid-shell displacement d of state n's N_u(n) frontier electrons (ESTABLISHED: the shell model,
Dick & Overhauser, Phys. Rev. 112, 90 (1958); a Drude oscillator). The restoring energy is ½k|d|², with

    k_u(n) = N_u(n)² K_E / α_u

where α_u is the unit's polarisability (the bound sector's own α_u, Å³, the one the A1 prior pins), so the
shell relation α = Y²/k holds with Y = N_u(n)e. Then

    Φ_u(n) = min over d of ½k|d|² + Σ_v c_v S_v(|R_uv − d|),   c_v = K N_u(n) N_v s(R_uv)

taken to second order in d, with each pair's overlap Hessian projected onto its positive, longitudinal part:

    Φ_u(n) = Σ_v c_v S_v − ½ gᵀ (k I + Σ_v c_v max(S_v″, 0) R̂R̂ᵀ)⁻¹ g,   g = Σ_v c_v S_v′ R̂
    Π_u(n) = Φ_u(n) − Φ_u(f_u)

The shell moves by d = (kI + …)⁻¹ g, away from the closed shells. The dipole −N_u(n)e·d points at them.

**Why second order, and why the projection.**
- The transverse overlap curvature S′/R is negative. At large K it would make the exact second-order
  Hessian indefinite: the shell would slide sideways off an O²⁻, a spurious saddle.
- The projected Hessian is positive definite for every geometry, K and α. The result is closed-form, so
  there is no inner minimisation and the derivatives are exact through autograd.
- In a centrosymmetric cage g = 0, so the relaxation is exactly zero (the frozen §18 holds in bulk).

**Exact properties** (`test_redox_exclusion_relaxed.py`, 9):
- the frozen limit α → 0 gives §18;
- 0 ≤ relief ≤ frozen;
- no relief in a centrosymmetric cage;
- the single-neighbour closed form c²S′²/(2(k + cS″)), against S′ and S″ by autograd of the quadrature;
- the shell moves away from O²⁻;
- a hole relieves less once the neutral has relaxed;
- relief grows with α;
- forces are −∇E;
- smooth at the cutoff.

**Stated approximations.**
- **α is configuration-independent** (the frame's α_u for every state), as everywhere in TRACE. For a cation
  frame, α_u is the core's: a carrier there (Mg⁺, Al²⁺) is held stiffly and stays near the frozen §18.
- **The Pauli shift is not coupled to the bound sector's field response.** Their cross term −E·μ_P is
  omitted at this order. It belongs to a bound-sector dipole drive χ_μ (theory §4).

**Counterfactual** (step 6's weights, untrained, defaults; 400 test structures; benchmarks_4g README):
- the undoped flat − upright class residual moves from −314.5 meV to +771.0 (frozen §18: +1422.3);
- the doped-upright electron moves to the far Au (near/far −0.443/−0.957).

**Trained (step 10, benchmarks_4g, 2026-10-16): a negative result. Off by default; not recommended.**
- Doped Au₂ becomes Au₂²⁻; T5 −1788 meV against DFT −931.6; step 9 (frozen) gives −881.2.
- The mechanism is not established. Of the two stated approximations, the configuration-independent α
  makes Au⁻ stiffer, not softer, so it cannot by itself explain too much relief. The uncoupled Pauli
  dipole is untested.

### 18.5 Pauli exclusion between carrier units (`exclusion_pairs=True`) — NEW-TO-TRACE

SPEC 18 places Pauli repulsion only between a carrier and a closed-shell frame. Two open-shell-frame units
(Au–Au, Ag–Ag, H–H) never repel there, whatever their states. So a doped Au₂²⁻ (Au⁻Au⁻: two closed 6s²
shells, σ²σ*², bond order 0) keeps the neutral bond energy that the configuration-independent ACE gives it.
The general rule, of which SPEC 18 is the closed-frame case:

    Ξ_uv(n_u, n_v) = K s(R) [P(n_u, n_v) S(ρ_u^(n_u), ρ_v^(n_v)) − P(f_u, f_v) S(f_u, f_v)]
    P = N_u N_v  if closed(n_u) or closed(n_v)  (N ∈ {2, 8}),  else 0  (two open shells bond)
    E(n) = Σ ê + bᵀn + ½nᵀWn + Σ_pairs Ξ_p(n_i, n_j)

- **Values for Au₂:**
  - Au⁰Au⁰ (σ²): 0;
  - Au⁻Au⁰ (σ²σ*¹): 2KS;
  - Au⁻Au⁻ (σ²σ*²): 4KS, which is 0.45 eV and ~0.7 eV/Å at 2.47 Å for K = 7.74 Ha;
  - Au⁺Au⁻: 0 (empty 6s), so step 9's undoped mechanism is unchanged.
- **Bond strength now follows charge state**, which is what makes the charge state identifiable from
  forces.
- **Implementation.**
  - The two-body term in `configuration_energies`.
  - Exact move energies in the search: for a → b, ΔΞ = up[a] + down[b] + the (a, b) pair correction.
    Brute-force tested.
  - Pairs: neighbour pairs i < j within the cutoff whose frames are both 0.
  - Overlaps from the SPEC 18.3 tables.
- **Tests** (`test_redox_exclusion_pairs.py`, 26):
  - the closed form;
  - exact move energies;
  - search against brute force;
  - Au values against an independent quadrature;
  - zeros for open–open and for empty shells;
  - forces −∇E.
- **Measured, untrained:** in step 9 seed 43, Au₂²⁻ lies 1.7–2.4 eV below the best Au₂⁻ configuration,
  so the fixed-weight effect cannot flip it. Step 11 tests training.
- **Stated limitation:** closed(n) by electron count is exact for s-valence elements (alkali, coinage, H)
  and for s²/s²p⁶ closures. The d block is not treated.

### 18.6 In-situ anion densities (`exclusion_anions="in_situ"`) — NEW-TO-TRACE

An anion's frontier density exists only because of the Madelung field, so its extent is environmental. Each
anion-frame unit's density decays with the model's own in-situ removal energy: the configuration-energy
difference for one electron leaving it, all other units at frames, without Π:

    I_u(n) = ê_u(n+1) − ê_u(n) + (b + W f)_u + ½(2d+1) W_uu,   d = n − f_u
    Ĩ = smooth clamp to [1, 40] eV,   κ = √(2Ĩ),   β = 0

- It replaces the learned I_eff (no parameter). Cation and neutral frames keep the gas ladder: a carrier
  caged by anions has a near-nucleus density set by its own nucleus, and in situ a localised Mg⁺ is unbound
  (I ≈ −3 eV). That would make its density infinitely diffuse, which is the sink artifact of SPEC 18.
- **Measured on step 9's weights:**
  - O²⁻ 9.83 eV in the surface layers, 11.2–11.5 eV in the bulk layers (learned I_eff 9.29, the surface
    value);
  - Mg²⁺ core 54.7 eV (gas 80.1);
  - Al²⁺ 3.0 eV (gas 28.4).
- **Counterfactual** (step 9 seed 42, 20 test structures per class): in-situ frames give doped Au₂ −1.0 and
  undoped Au₂ 0, with class offsets +41 / −38 meV. The learned I_eff gave +49 / −27, so this is a
  physically derived drop-in.
- **Anion holes** (`exclusion_anion_holes`, off, not recommended): hole states on anion units (O⁻, Cl⁰)
  in the exclusion, with the correct sign in isolation (tests). On step 9's weights they drive spurious
  O → Au transfer in undoped cells (Au₂ −1.0 to −1.6 e; offsets ≈ −1660 meV). The hole releases frame–frame
  O–O overlap, which the ACE already holds. Not used until that double count is resolved.
- **Numerics:** the anion κ axis spans I ∈ [1, 40] eV (40 Chebyshev terms, ≤ 4e-10 against the quadrature),
  plus a 3-D (R, κ_u, κ_v) table for two in-situ anion densities.
- **Tests** (`test_redox_exclusion_insitu.py`, 10):
  - the configuration-energy identity;
  - an isolated Cl⁻ giving EA(Cl);
  - Madelung ordering corner < edge < face < centre;
  - the 3-D table;
  - hole relief for O⁻ and Cl⁰;
  - frames-only mode adding nothing to anions;
  - forces −∇E;
  - config refusals.

## 19. Free carriers and bound fluctuations (`RedoxConfig(carriers="covalent_units")`) — NEW-TO-TRACE

Theory: docs/ELECTROSTATICS_THEORY_2026-10-16_universal_charge.md.

**Measured motive.**
- The atom-resolved formal stack refuses the carbon chain: 1451 configurations within 2.5 eV for
  C₁₀H₃⁺, the lowest a charge-ordered C⁺/C⁻ chain.
- The configuration energies carry no covalent-bond energy loss, so the pair configurations inside a
  covalent network are spuriously low. The exact repair would be full CI, which is exponential in the
  unit size.

**19.1 The partition.**
- **Covalent units:** connected components of the chemical-bond graph over bonds joining two neutral-frame
  (open-shell) atoms (`fragments.covalent_units`). A closed-shell frame atom is its own unit.
- **Small units** (≤ `full_unit_atoms`, default 3) keep every intra-unit configuration: exact
  diagonalisation of TRACE's spinless charge-configuration model in that unit.
  - For a 2-site bond, the three charge configurations mixed by τ have the structure of the Hubbard dimer's
    singlet CI. The Hubbard-dimer singlet is reproduced exactly with coupling √2·t, which the learned τ
    absorbs.
  - The diagonal energies are TRACE's model energies (ladder, Coulomb, learned shifts), not the exact
    electronic ones, so this is not full CI of the true electronic problem.
  - Au₂/MgO is unchanged from step 9 (tested).
  - The threshold is an active-space size (ESTABLISHED practice: exact treatment within an affordable active
    space), not a fitted parameter.
- **Larger units** hold only free carriers: the deviations n_u − f_u are sign-definite inside the unit
  (`diabatic.admissible`, enforced exactly in all three phases of `search_configurations`; brute-force
  tested). A carrier may sit on any atom of its unit and hops through τ. By Theorem B one carrier is
  exact tight binding.
- Electron–hole pairs inside a large unit are bound-sector fluctuations, and the ACE holds the neutral
  bond energy.
- Between units every configuration is allowed (dissociation Na⁰Cl⁰, donor → acceptor; tested).
- **Counts:** C₁₀H₂ is 1 configuration; C₁₀H₃⁺ at most 13. The refused call now runs with 13, the hole
  spread over the chain.

**19.2 Exclusion on bound fluctuations** (`exclusion_bound=True`, needs exclusion). The Parr–Pearson finite
differences of Π at each frame enter the bound sector:

    χ_P = ½[Π(f+1) − Π(f−1)],   η_P = Π(f+1) + Π(f−1)

(ESTABLISHED closure: Parr & Pearson, J. Am. Chem. Soc. 105, 7512 (1983)).
- Applied through one helper in forward and bound_state, so energy_bound = bound_state energy (tested).
- Zero for isolated atoms and anions. η_P < 0 and unbound neighbour states are refused.
- **Counterfactual on step 9's weights, untrained** (2 structures per class; reported): it does not by
  itself restore the upright Au₂ stabilisation that the 2-site CI gives (undoped class offset −838 meV).
  It is not used for Au₂/MgO, where the dimer is solved exactly (19.1).

**Limitations (stated).**
- The unit partition changes where a bond between open-shell atoms forms or breaks. SPEC 12's percolation
  mixture is the smooth route and is not used here.
- Units just above the threshold switch treatment.
- Many delocalised carriers (doped slabs) need a carrier mean field, which is not built.

Tests: `test_redox_covalent_units.py` (57), `test_redox_exclusion_bound.py` (7).

## 20. A band unit for delocalised carriers (`RedoxConfig(carrier_band=True)`) — NEW-TO-TRACE

Theory: docs/ELECTROSTATICS_THEORY_2026-10-16_universal_charge.md §9–10. Module: `trace_redox/band.py`.

**Motive (measured).**
- Step 9's seeds split on where the doped cell's donor electrons go (Au₂²⁻ against Al²⁺). PBE
  delocalises them; integer site configurations cannot.
- An electron on any single Mg costs +3.9 to +4.3 eV, so a delocalised channel needs hopping, and the
  learned couplings never see Mg–Mg moves.

**Construction.**
- Band sites are the cation-frame atoms that can take an extra electron; their extra-electron states move
  to one band unit.
- One-carrier Hamiltonian h = diag(ε) − τ:
  - ε_v = E(f − e_v) − E(f), the exact configuration energy;
  - τ = V_ssσ(R)·exp(ln 20 tanh s)·switch, with V_ssσ = 1.40 ħ²/(mR²) (Harrison; ESTABLISHED) and s
    learned (initial factor `band_initial_factor`, default 1; range 1/20–20).
  - **The initial factor (2026-10-22).** The nearest-neighbour s band on the fcc cation lattice has
    m* = ħ²/(2 f V_ssσ a²); PBE's MgO conduction-band mass at the 4G lattice constant (0.322–0.330 mₑ) gives
    f = 0.55, Harrison's f = 1 gives 0.179 mₑ. The curvature is the anchor because the band carries the
    effective-mass physics of an extra electron near a perturbation; the band edge belongs to the site energies.
- The band holds m ≤ 2 electrons in φ₁:
  - E_b(1) = λ₁, exact one-carrier tight binding;
  - E_b(2) = 2λ₁ + J₁₁, with J₁₁ = Σ φ_u²φ_v² V_uv and V_vv the exact same-site second-electron cost.
- The band's charge is −m·φ₁² on the sites, exact for m ≤ 2. It enters through `omega` (b, W rows), so the
  bound response stays exact.
- Hopping to a unit u is Σ_v T_uv φ_v.
- **Derivatives:** a detached eigh is refined by two differentiable Newton steps on the bordered
  eigen-equation, so the derivatives are exact to third order at a non-degenerate λ₁ (Perron–Frobenius),
  whatever the degeneracy of higher states. A degenerate λ₁ is refused.

**Tests** (`test_redox_band.py`, 7):
- the eigenpair against eigh, Hellmann–Feynman and second-order perturbation theory with degenerate
  excited states;
- the eigenvector derivative against finite differences;
- no carriers, no change;
- one band electron = tight binding, which localises at weak hopping;
- charge conservation;
- forces −∇E.

**Counterfactual** (step 9 seed 42 weights, band untrained, 3 doped test structures).
- ε(Al) ≈ +2.9 eV and ε(Mg) ≈ +6.8–7.2 eV.
- At factor 1, λ₁ ≈ −2.1 to −2.5 eV and φ₁ is delocalised, with 36 % of its weight on Al. The lowest
  configuration is Au₂⁻ + 2 band electrons in all three structures.
- At factor ≤ 0.25, φ₁ localises on an Al and Au₂²⁻ reappears in one structure.

**Limitations (stated).**
- At most two band electrons, both in φ₁. Two electrons on two different deep sites need φ₂.
- The s–s anchor is universal, not MgO-specific; the learned factor carries the difference.
- The band sites' own hole states remain atom states.

## 21. Amplitude selection of the configuration space (`RedoxConfig(diabatic_selection='amplitude')`) — NEW-TO-TRACE

Module: `trace_redox/diabatic.py` (`coupling_amplitude`, `amplitude_switch`, `configuration_importance`,
`switched_hamiltonian`, `coupled_closure`, `select_configurations`). Probes:
`experiments/benchmarks_4g/probe_window_convergence.py`, `probe_window_deficit.py`,
`probe_amplitude_selection.py`.

**The defect (measured 2026-10-18, step 12 weights).**
- §17.3 keeps the configurations within `diabatic_window` of the lowest. P(R) is piecewise constant, so F(R)
  jumps whenever a configuration crosses the edge. With a coupling t to the low block the jump is the
  second-order shift ~ −t²/Δ, not the Boltzmann weight e^(−Δ/kT) of an uncoupled one. The search is
  detached, so the forces never see the jump.
- Au₂/MgO, τ(Au, Au) = 1.18 eV. On seed 43's undoped wetting test structure 4, Au⁺Au⁻ sits at 2.50 eV, on
  the edge. Window 2.5 → 3.5 eV changes E by **−834 meV** (1 → 3 configurations). On seed 42's undoped
  non-wetting structure 2 the change is −219 meV.
- The energy never converges in the window. Configurations at 12–16 eV (54 O²⁻ → band transfers,
  |T| = 0.27 eV) add another −200 to −260 meV. That is the lattice's electronic polarisation, which the bound
  sector already carries (ε∞), so counting it again would double count.
- Over all 400 test structures, the deficit d = E(2.5) − E(8 eV) carries the residual scatter: seed 43
  undoped wetting sd 307 meV, r(residual, d) = +0.99, sd(residual − d) = 41 meV; doped non-wetting 187, +0.98,
  40 meV.

**Construction.**
- Reference and excitations: E_ref = −kT ln Σ_c e^(−E_c/kT), the diagonal free energy of the set, and
  x_c = E_c − E_ref ≥ 0.
- Importance of a configuration: the leading Rayleigh–Schrödinger amplitude along its best path from the
  thermally weighted reference,
  r_c = max(√p_c, max over coupled i of r_i·g(T_ic, x_c)), with p_c = e^(−x_c/kT) and
  g(t, x) = |t|/√(x² + t²), the first-order amplitude |t|/x saturated smoothly at 1.
  - The criterion is that of selected CI (ESTABLISHED: Huron, Malrieu & Rancurel, J. Chem. Phys. 58, 5745
    (1973)). Its use to partition static mixing (explicit) from virtual excitation (the bound sector's linear
    response) in a learned potential is NEW-TO-TRACE.
  - Denominators are measured from the reference, as in perturbation theory. A first version used pairwise
    gaps |E_i − E_j|. That admitted chains of virtual hops between nearly degenerate high configurations
    (O-hole hopping at 10–19 eV) and exceeded the 512 cap on undoped non-wetting Au₂/MgO. That version was
    measured on frozen step-12 weights and discarded.
- Pair switch: e_ij = max(r_i·g(T, x_j), r_j·g(T, x_i)) and σ = smoothstep((ln e − ln a_off)/ln(a_on/a_off)),
  so σ = 0 for e ≤ a_off and σ = 1 for e ≥ a_on, C² in e.
- Hamiltonian: H̃ = diag(E) + Σ_moves sign·σ(e_ij)·T_ab, with the §17.2 signs.
- Set: P is the window set plus every configuration with r > a_off, found by a best-first (max-product
  Dijkstra) search through one-electron moves (`coupled_closure`). If it meets a configuration below the
  window's lowest, the search restarts there. More than `fragment_max_configurations` is refused, never
  truncated.
- Differentiation: the argmax tree is found detached, and r is rebuilt along it by gathers and products, so
  derivatives of any order exist. r has kinks only at exact ties of two paths.
- Defaults: a_off = 0.05 and a_on = 0.2. Measured on Au₂/MgO:
  - in: Au₂ ionic and covalent configurations, r ≥ 0.21; the O → Au transfer with a learned |T| = 2.0 eV
    at 10.5 eV, r = 0.19;
  - out: the O-hole hops that follow it, r ≈ 0.003; O → band, r = 0.019.

**What it guarantees.**
- g ≤ 1, so r_c ≥ e_ic for every neighbour i. Every pair that leaves P = {r > a_off} therefore has σ = 0: P
  is closed. A wider window adds only decoupled configurations. F(w′) = −kT ln(Z_P + Z_rest) up to the shift
  of E_ref, which is the added configurations' Boltzmann weight.
- F is continuous in R, to within the Boltzmann weight of decoupled configurations at the window edge. It is
  piecewise C² (kinks at path ties).
- As a_off, a_on → 0 it is the full diagonalisation over every connected configuration.
- Without couplings it is the §17.3 window selection exactly.
- A chain of k virtual excitations has r ~ Π g. Closure growth is governed by the amplitudes, not by the
  combinatorics of nearly degenerate high configurations: a ligand transfer followed by a hole ring gives
  |P| = 3 for n = 6 and 12 donor sites.

**Tests** (`test_redox_amplitude_selection.py`, 20):
- **The defect and its fix:** the window selection jumps by more than 0.3 eV at the edge, where the
  amplitude selection is continuous (≤ 2h) and equals the complete 3×3 value.
- **Closure:** on 6 random problems, the zero block between P(2.5) and the rest of P(6), and F(6) against
  −kT ln(Z_P + Z_rest) to 1e-10.
- **Importance is the path product:** a donor ring of 6 and 12 sites gives |P| = 3, with r = g(2, x) exactly.
- **Complete limit:** brute force over all 3⁵ vectors.
- **The switch and g:** C² at both ends, monotone, midpoint ½; g saturates, with finite gradients at x = 0
  and t = 0.
- **The switch region:** the closed-form 2×2 in that region (to 1e-12), with dF/dt against finite
  differences.
- **Bookkeeping:** refusal beyond the cap; no couplings means no change.
- **Module:** forces −∇E with couplings up to 1.16 eV (16 pairs in the switch region). The window
  dependence is ≤ 1e-6 eV, where the window selection on the same weights changes by 533 meV.
- **Validation** of the configuration.

**Limitations (stated).**
- Importance is a max over paths, not the coherent sum. A collective virtual excitation, such as 54 weak
  O → band transfers adding up coherently, stays with the bound sector by construction.
- In the switch region the approximation lives in σ: a pair there counts as σ²·t²/Δ to second order.
- r has kinks where two paths tie, so F is piecewise C², not C².
- a_off and a_on are model choices, not derived. The measured gap between static and virtual amplitudes
  on Au₂/MgO (0.24 against 0.019) brackets them.
- **The switch band leaks into responses (measured 2026-10-19, SPEC 30).** With a pair inside a_off < e < a_on,
  σ(e) depends on the configuration energies, and so on the field and the geometry. Its derivatives enter the
  field response and the forces. A random test model gives a whole-model α_zz of −3.99 against the fixed-H
  +2.36 e²Å²/eV. Trained step-13 Au₂ shows no pair in the band, and no leak to 0.01 Å³.

## 22. One degree of freedom, one sector: bond charge flow (`RedoxConfig(bond_partition=True)`) — NEW-TO-TRACE

Module: `RedoxElectrostatics._partition_bonds`, applied in `forward` and in `bound_state`. Probes:
`experiments/benchmarks_4g/probe_dimer_ct.py` and the bond-channel split of README "Over-polarisation".

**The double counting (measured 2026-10-18).**
- Charge flow along a bond i–j appears twice in the formal model. It is in the free sector's integer
  transfers (n_i, n_j) = (±1, ∓1), mixed by τ_ij: by Theorem A of the 2026-10-16 note, the exact two-state
  functional of the bond. It is also in the bound sector's split-charge variable t_b, with a learned
  hardness κ_b. The bound one screens the free ionic pair.
- Au₂ at 2.47 Å (isolated; step 12 weights), screening of Au⁺Au⁻ by the bound sector:

  | seed | full | bond channel alone | moments alone | κ_b |
  |---|---|---|---|---|
  | 42 | −3.49 eV | −3.41 eV | −0.92 eV | 13.3 eV |
  | 43 | −1.34 eV | −0.85 eV | −0.81 eV | 23.0 eV |

- Δ_ion = E(Au⁺Au⁻) − E(Au⁰Au⁰) is therefore +0.17 against +2.47 eV. With the bond channel removed on the
  same weights it is +2.74 against +3.00 eV: the seed spread falls from 2.30 to 0.26 eV. The data do not fix
  κ_b; the two seeds put the bond's charge transfer in different sectors.

**Construction.**
- The split-charge variable of bond b = (i, j) is kept only when i and j lie in one covalent unit of more
  than `full_unit_atoms` atoms (SPEC 19). Inside such a unit the free sector excludes electron–hole pairs
  (admissibility), so the bound variable is the unit's only intra-unit charge flow.
- Everywhere else, charge flow between carrier units is the free sector's. That covers the explicit transfer
  configurations and couplings (SPEC 17.6, 21), and the virtual ones that SPEC 21 leaves out. On those bonds
  f_b → 0.
- f_b is the dissociation envelope (SPEC 1.4.1), which scales the incidence columns, the seed and the
  periodic bond dipole, so every consumer of B sees it. The bond's own hardness stays, decoupled.
- With `carriers='atoms'` every atom is its own unit, and the bound sector is then the atom-centred moments
  alone. Their polarisabilities carry ε∞, as in the Clausius–Mossotti picture of ionic crystals
  (ESTABLISHED: Tessman, Kahn & Shockley, Phys. Rev. 92, 890 (1953)).

**Tests** (`test_redox_bond_partition.py`, 6):
- with atom units the module is the bound sector without bonds exactly (1e-12), and it differs from the
  unpartitioned model;
- a 5-carbon covalent unit keeps its C–C bonds, and its bonds to Na and Cl are zeroed;
- a dimer's ion pair is screened by the moments-only Schur complement (1e-10);
- forces −∇E;
- `bound_state` sees the same partition;
- off by default, and validated.

**Measured on frozen step-12 weights (counterfactual, not a criterion).**
- MgO ε∞ falls to 1.41 / 1.28 (from 3.16 / 3.11): these weights carried the lattice polarisation in bond
  charge flow. Training against the ε∞ prior has to move it into the moments.

**Limitations (stated).**
- The partition follows the SPEC 19 units. Those change where a bond's switch support begins, the same limit
  as SPEC 19.
- Virtual charge transfer between units that SPEC 21 leaves out is represented by no explicit term. The
  moments and the ACE carry its field and zero-field parts.

## 23. Charge penetration of charge-state densities (`RedoxConfig(carrier_penetration=True)`) — NEW-TO-TRACE

Module: `trace_redox/penetration.py` (runtime), `trace_redox/radial.py` (radial numerics); data
`trace_redox/data/ion_densities.npz` from `tools/electrostatics/build_ion_densities.py`; band corrections in
`trace_redox/band.py` (`_band_pair_corrections`). Probe: `experiments/benchmarks_4g/probe_penetration_counterfactual.py`.

**The defect (measured 2026-10-18).**
- Every free charge sits on a compact Gaussian (Au 1.11, O 0.81, Mg 0.56 Å in step 12). At contact the real
  charge distributions overlap: a diffuse carrier enters a neighbour's electron cloud and sees part of its
  bare nucleus.
- With X2C-PBE0/ANO-RCC densities (O²⁻ in a Watson sphere), an Au carrier at 2.2 Å from O²⁻ has exact
  electrostatics −11.40 eV (Au⁺) and +10.48 eV (Au⁻), against the model's −12.78 and +12.78. The model
  favours Au⁺ next to O²⁻ over Au⁻ by 3.7 eV too much (3.4–4.0 eV over Watson radii 1.0–1.4 Å), the direction
  of the over-polarised upright Au₂.
- For the Au⁺Au⁻ pair itself the two penetrations cancel (net −0.06 eV); for Ag⁺Ag⁻ the net is +0.66 eV.

**Construction.**
- The exact electrostatic energy of two spherical ions (nucleus Z plus density ρ):
  E_es(R) = Z_a Z_b/R − Z_a φ_b(R) − Z_b φ_a(R) + (ρ_a|ρ_b)(R).
  - φ is the electron potential of a spherical density.
  - (ρ_a|ρ_b) uses the shell average ⟨φ_b⟩ = [W(R + r) − W(|R − r|)]/(2rR), with W(x) = ∫₀ˣ φ_b(d) d dd.
- Pair term, for every pair within 8 Å (periodic images each once):
  Ξ_ab(n_a, n_b; R) = s(R){D_ab + (n_a n_b − f_a f_b)[K/R − J^G_ab(R)]}, with
  D_ab = [E_es(n_a, n_b) − E_es(f_a, f_b)] − (n_a n_b − f_a f_b)K/R.
  - J^G is the model's own free–free Gaussian kernel (learned widths, differentiable).
  - s is a C² switch over [6.5, 8] Å.
  - Ξ is zero at frames: the frame–frame part is state-independent and stays with the ACE.
- R·D is tabulated per (element, state) pair as a 96-term Chebyshev series on [0.5, 8] Å. All pairs of one
  element pair share one basis matrix, so the evaluation is one matmul and is cheap to differentiate twice.
- Band unit (SPEC 20): its diagonal ε_v is the exact one-electron configuration energy including the pair
  terms; V (for J₁₁) gets their second differences; and the band's φ₁²-weighted carrier gets a pair term with
  every other unit (bilinear in m).

**Densities.**
- Method: UKS PBE0 (ESTABLISHED: Adamo & Barone 1999); X2C spin-free for Z > 36 (ESTABLISHED); ANO-RCC plus two
  diffuse s and p shells (ratio 3.2). The lower of the two lowest spin states is taken, spherically averaged
  (302 directions) on a 12 000-point log grid and renormalised to Z − q.
- O: every state sits in one Watson sphere of +2 at 1.205 Å, where 2K/R_W equals the rock-salt MgO Madelung
  potential at O. The charge states of one site then share its confinement, and their differences are the
  carrier alone. A free O⁻ against a confined O²⁻ was tried first; it put the confinement of all nine
  electrons into the difference, and was replaced.
- Mg⁻ is unbound and left out: its pairs get no correction.
- **Functional, measured on Au⁻** (2026-10-18):

  | | carrier ⟨r⟩ | Ξ(Au⁻, O²⁻) at 2.2 Å | ΔSCF EA | −ε_HOMO(Au⁻) |
  |---|---|---|---|---|
  | PBE0 | 2.236 Å | −2.39 eV | 2.00 eV | |
  | ωB97X | 2.250 Å | −2.46 eV | 1.98 eV | 1.90 eV |
  | HF | 2.500 Å | −2.87 eV | 0.62 eV | 1.15 eV |
  | exact | | | 2.31 eV | 2.31 eV |

  - PBE0 and ωB97X agree to 3 % in the bulk that sets the penetration at contact. HF, without correlation,
    underbinds the anion and is too diffuse. PBE0 is kept.
- **No tail splice.** A C¹ match of the Katriel–Davidson tail was tried and is not robust: it found no
  crossing for most cations, and spurious ones inside the valence shell of Cl⁻, Na⁰ and Al⁰. The far tail
  contributes ~1 meV beyond 5 Å.
- **Numerics.** The pair tables use the cancellation-free difference route of `radial.penetration_difference`.
  - The total-energy route Z_aZ_b/R − … cancels ~1000 hartree for Au–Au and is sensitive to 10⁻⁶ relative
    electron counts.
  - The difference route is grid-converged to 1e-7 eV (2× and 4× refinement).
  - Chebyshev residuals: median worst 1e-4 eV·Å per type; worst 2.1e-3 eV·Å at 1.17 Å (Au³⁺–Au³⁺), which is
    2e-5 relative.
- **Stated uncertainties:** ~3 % from the functional and ±9 % from the Watson radius (1.0–1.4 Å: the Au⁺/Au⁻
  preference error is 3.37–3.97 eV).

**ESTABLISHED ingredients.**
- The electrostatics of overlapping spherical ions (Gordon & Kim, J. Chem. Phys. 56, 3122 (1972)).
- Charge-penetration corrections in force fields.
- Watson spheres (Phys. Rev. 111, 1108 (1958)).

**NEW-TO-TRACE:** state-resolved penetration of first-principles charge-state densities as a two-body term of
the integer-configuration energies, with the band's carriers included.

**Limitations (stated).**
- The bound sector's coupling to the carriers (their field at the moments) keeps the Gaussians.
- Densities are rigid free-ion densities: no in-crystal compression beyond the Watson sphere of O²⁻.
- The Pauli half of the overlap stays SPEC 18's, on its own Katriel–Davidson densities.

### 23.x Cells shorter than the cutoff, and exchange symmetry (2026-10-18 evening)

- **Small cells.** Before this, any atom meeting its own periodic image within R_CUT = 8 Å was refused. That
  covered every cell with a lattice vector below 8 Å: rock-salt MgO's 4.21 Å conventional cell and most bulk cells
  of foundation datasets.
  - An atom and its images are always in the same state, so their pair terms are one-body:
    ε_i(s) += Σ_{T in half space, 0 < |T| < R_CUT} Ξ_ii(|T|; s, s), half the sum over all T ≠ 0
    (`penetration_self_terms`, added to the ladder).
  - SPEC 26 needs nothing for them: the images at ±T exert opposite fields on the atom's moment, so they cancel
    exactly and are skipped.
- **Exchange symmetry.** Ξ between two atoms of one element is symmetric under their exchange, but the builder
  fits (Z, n, Z, m) and (Z, m, Z, n) separately.
  - Measured asymmetry: electrostatic ≤ 7.8e-6 eV·Å; Pauli ≤ 2.4e-2 eV·Å (exotic Na⁻/Na³⁺). For example,
    K(Mg⁺ in Mg²⁺) is fitted as 30.66071 and 30.66047.
  - The loaded library holds their mean, which is exact because the series is linear in its coefficients.
  - The new small-cell test found this: the cell and its supercell list the same pairs in opposite index orders,
    and Gordon–Kim differed by 3.2e-7 eV until the tables were symmetrised.
- **Tests** (`test_redox_penetration.py`, 3 new):
  - an 8-atom rock-salt cell equals its 2 × 2 × 2 supercell per cell, for a periodic configuration, to 1e-10, for
    both overlap models;
  - the same-element tables are bitwise exchange-symmetric;
  - the module on the small cell, with penetration, Gordon–Kim and induction, gives forces −∇E.

## 24. Exact isolated limits (`RedoxConfig(shift_gate=True)`; band states) — NEW-TO-TRACE

Theory: docs/ELECTROSTATICS_THEORY_2026-10-18_overlap.md §4. Module: `trace_redox.module.shift_gate`.

**The violation (measured 2026-10-18).**
- The learned charge-state shifts are an MLP of the atom's own invariants, so they need not vanish without an
  environment. Isolated atoms carry trained shifts: Au⁻ +1.18 eV, Au⁺ +1.25 eV (AuMgO step 12 seed 42; seed 43
  +1.05 / +1.20), Ag⁺ −1.36 eV (Ag₃ step 9), Na⁻ +0.19 eV, Cl⁻ −0.11 eV (NaCl step 9).
- So the model's free Au has EA 1.13 eV against the exact 2.31, and its free Ag has IE 6.2 eV against 7.58.
  Every dissociation into ions misses its exact limit.

**Construction.**
- shift_u(n) = g_u Δ_u(n), with g_u = 1 − Π_{edges into u}(1 − s(r)). Here s = 1 for r ≤ 0.8 r_c and the C²
  quintic down to 0 at r_c (r_c = attention_cutoff).
- g = 0 exactly for an isolated atom and g = 1 exactly with any neighbour inside 0.8 r_c. A periodic self-image
  counts as a neighbour.
- The product is evaluated as exp(Σ log(1 − s)), clamped at 1e-300. That is exactly 1.0 where s = 1, and s is
  flat there, so no derivative changes.
- Band (SPEC 20): the state vector reaches −2 whenever the band is on (the atoms' own ladders are +inf there),
  so the band holds its two electrons whatever elements are present. Before, an isolated Mg was refused
  ("capacity [1, 3]").

**Tests** (`test_redox_isolated_limits.py`, 12):
- isolated Au, Ag, Na, Cl and O have the exact ladder with ~1 eV test shifts, which shift them when the gate is
  off;
- one bonded neighbour leaves g = 1, and none beyond the cutoff gives g = 0;
- the Na + Cl ionic–covalent crossing falls at 9.462 Å, against the exact K/(IE_Na − EA_Cl) = 9.434 Å; the
  0.03 Å is the separating ions' induction. Without the gate there is no physical crossing (gap +2.1 eV);
- an Au + Au ion pair beyond every cutoff (10, 12 Å) has the exact classical gap IE − EA − K/R plus the full
  6×6 self-consistent induction of the two isolated atoms' 3×3 polarisabilities, to 256 ε, with SPEC 28 the
  states' own (ρ = 0.369, 3.389). The induction (~1e-3 eV) is checked to ten digits; the residual falls
  superexponentially (1.5e-8 eV at 6 Å, 2.3e-14 at 8 Å), as Gaussian damping must. A collinear 2×2 reference
  left an α_xz²/R⁷ term: the reference, not the model;
- forces −∇E through the gate's switching shell;
- off by default;
- an isolated Mg with the band is neutral, with free energy 3.6e-15 eV against the exact 0 (the neutral-atom
  reference).

**On the benchmark data.** Every atom of the AuMgO train and test sets has a neighbour within 2.94 Å, inside the
onset, so g ≡ 1 and the gate is an identity there. It is to be on in every run after step 15.

## 25. The Pauli half of the overlap from first principles (`RedoxConfig(carrier_overlap='gordon_kim')`) — NEW-TO-TRACE

Module: `trace_redox/radial.py` (`gordon_kim_difference`, `gordon_kim_table`, `density_overlap_parts`);
tables built by `tools/electrostatics/build_ion_densities.py`; runtime in `penetration_pairs(model='gordon_kim')`.

**Why (measured 2026-10-18).**
- SPEC 23's exact electrostatics alone over-corrects: a band electron on Mg drops ~7 eV for its six O²⁻
  neighbours.
- The Pauli repulsion of the same overlap was SPEC 18's fitted K·S, on other (Katriel–Davidson) densities. The
  electron-gas energy of the summed first-principles densities (Gordon & Kim, J. Chem. Phys. 56, 3122 (1972);
  ESTABLISHED) supplies it:
  G = ∫{f[ρ_a + ρ_b] − f[ρ_a] − f[ρ_b]} relative to the frames, with
  f = C_F ρ^(5/3) + (1/9)|∇ρ|²/(8ρ) − C_x ρ^(4/3) − 0.44ρ/(7.8 + r_s).
  - The terms are Thomas–Fermi, the second-order gradient term (Kirzhnits), LDA exchange and Wigner correlation.
  - It is evaluated pointwise as a difference on a prolate-spheroidal grid, so no core energy is subtracted.
  - Grid-converged to 3e-4 eV; symmetric in the two ions.

| pair | penetration | G | net |
|---|---|---|---|
| Au⁻–O²⁻ at 2.2 Å | −2.39 eV | +1.28 eV | −1.11 eV |
| Au⁺–O²⁻ at 2.2 Å | +1.38 eV | −1.27 eV | +0.11 eV |
| Mg⁺–O²⁻ at 2.1 Å | −1.59 eV | +1.23 eV | −0.36 eV |
| Au⁺Au⁻ at 2.47 Å | +0.05 eV | +0.57 eV | +0.62 eV |

**Construction: G at contact, the decay of the density overlap.**
- G is wrong in the tails. Beyond ~3.5 Å its gradient, exchange and correlation parts give a spurious attraction
  (Mg⁺–O²⁻ at 4 Å: −62 meV), the local functionals' known failure at low density.
- The density overlap decays correctly, and G/ΔS is nearly constant at contact: 4.5–5.3 Ha for Au and Mg against O
  over 2.0–3.0 Å.
- So, bilinearly: E_Pauli = K_{a←b}(δ_a|ρ_b^f) + K_{b←a}(ρ_a^f|δ_b) + K_cc(δ_a|δ_b).
  - The one-carrier K are fitted to G over the contact window [0.75, 1.25](r_cov,a + r_cov,b).
  - K_cc is fitted to G's non-additive two-carrier remainder.
  - The density-overlap model of exchange repulsion is ESTABLISHED (Wheatley & Price, Mol. Phys. 69, 507 (1990);
    Söderhjelm et al. 2006). K derived from the electron-gas functional at contact is NEW-TO-TRACE.
- **Measured fit to G at contact:**
  - median residual 5.6 % over 921 types;
  - the AuMgO one-carrier types: Au⁻–O 1.6 %, Au⁺–O 3.2 %, Mg⁺–O 3.1 %, Al²⁺–O 2.8 %, O⁻–Mg 7.3 %;
  - the two-carrier Au⁺Au⁻: 8.2 %, against 43 % with a single K;
  - one-carrier K: median 2.69 Ha;
  - the poor fits (90th percentile 27 %) are exotic high charge states, which are never populated.
- `carrier_overlap='gordon_kim'` adds R·K·ΔS to SPEC 23's R·D series; the band (SPEC 20) sees it through the
  pair terms.
- **`pauli_table` (2026-10-25).** `'gordon_kim'` (default): K as above. `'first_principles'`: the one-carrier K of the
  AuMgO element pairs fitted instead to the frame-relative frozen-density PBE interaction in the Watson-sphere
  convention (validated against the lattice-embedded values, theory note 2026-10-18 §4c) over the data's contact
  distances; an overlay (`data/pauli_fp.npz`, `tools/electrostatics/build_pauli_fp.py`) replacing only the Pauli
  series.
- It refuses `exclusion=True`, which would count the Pauli repulsion twice, and it needs `carrier_penetration`.

**Tests** (`test_redox_gordon_kim.py`, 9):
- G is zero at frames, symmetric and converged;
- K·ΔS reproduces G at contact within the measured residual, and keeps its sign and decays beyond 4.5 Å, where G
  does not;
- the tables reproduce K·ΔS (≤ 1e-6 eV·Å for one-carrier types, 1e-3 for the two-carrier Au⁺Au⁻);
- configuration validation;
- forces −∇E with the band on.

**Limitations (stated).**
- Thomas–Fermi + 1/9 von Weizsäcker is the electron-gas kinetic energy. Its error for overlap repulsion is tens of
  percent in the literature, and that error enters K.
- The densities are rigid free-ion densities (O in a Watson sphere).
- The carriers' field at the moments keeps the Gaussians (theory note §3).

## 26. Induction by the carriers' true fields (`RedoxConfig(carrier_induction=True)`) — NEW-TO-TRACE

Module: `trace_redox/induction.py` (`correction_columns`, `overlap_induction_terms`, `FieldFactorTable`); the
field-factor tables come from `tools/electrostatics/build_ion_densities.py` (`field_keys`, `field_coefficients`);
`radial.field_factor` is the direct integral. Theory note 2026-10-18 §3, Proposition 4.

**Why (measured 2026-10-18).**
- The bound sector's moments feel a free charge through the Gaussian kernel, A[i, n_b + 3j + c] =
  K λ₃ (r_i − r_j)_c / R³ ℓ_j, with λ₃ = P(3/2, R²/(σ_f,i² + w_j²)).
- A carrier (the charge Q = n − f that a state adds to its frame) has its true density δ = ρ(n) − ρ(f). By Gauss
  (ESTABLISHED), its field at a moment of width w is K Q λ_c(R, w)/R², with λ_c the fraction of δ, smeared by the
  moment's Gaussian, inside R.
- Measured at an O site (w = 0.84 Å) at 2.2 Å: λ_c = 0.475 (Au⁻), 0.668 (Au⁺), against λ₃ = 0.827. The model
  overstates the induction those carriers drive (∝ λ²) by 3× and 1.5×.

**Construction (exact in the configuration energies).**
- The coupling of atom i in state s becomes A_i n_i + Q_is v_is, with the correction column
  v_is[n_b + 3j + c] = K (λ_c,s(R_ij, w_j) − λ₃(R_ij)) (r_i − r_j)_c / R³ · ℓ_j · s(R).
  - The frame charge keeps the Gaussian; s is the C² switch of the SPEC 23 tables (6.5–8 Å).
  - v = 0 at the frame state and for states without a tabulated density.
  - Periodic images are summed by `penetration.pairs_within`.
- The bound energy of a configuration is quadratic in the coupling, so with M = K⁻¹Aᵀ, y0 = −K⁻¹g and
  ΔF(n) = Σ_u Q_u v_u(s_u):
  δE(n) = ΔFᵀ(y0 − M n) − ½ ΔFᵀ K⁻¹ ΔF   (no approximation).
- It splits exactly into one-body and two-body terms (c0 = Vᵀ(y0 − M f), N = K⁻¹V):
  - E1_u(s) = Q c0_us − Q² v_us·M_u − ½ Q² v_us·N_us, added to the ladder ê_u(s);
  - E2_uw(s, t) = −Q_us Q_wt (v_us·M_w + v_wt·M_u + v_us·N_wt), in the SPEC 18.5 pair slot for all u < w.
- Cost: λ_c is a 2-D Chebyshev table (64 × 16 in R ∈ [0.5, 8] Å, w ∈ [0.2, 2.5] Å) per carrier state
  (33 states).
- **Scalable route (2026-10-19; `induction.induction_terms`).** The correction columns have dipole rows only.
  - V_d is stored on 3n rows, against the n_b + 3n of the variational basis (n_b ≈ 40 n in bulk MgO).
  - N = K⁻¹V is never formed. C2 = V_dᵀ (K⁻¹)_dd V_d, C1 = V_dᵀ M_d, c0 = V_dᵀ (y0 − M f)_d.
  - (K⁻¹)_dd = S⁻¹ from the condensed operator's existing Schur factor: block elimination with a zero bond source
    gives x_p = S⁻¹ r_p (ESTABLISHED block inversion), so no bond-block work is done.
  - SPEC 28 takes N's rows at the softened dipoles as (K⁻¹)_dd[rows] V_d; the SPEC 30 record takes K⁻¹(V Q) by one
    solve per configuration (`InductionColumns`).
  - Measured at 512 atoms (bulk MgO, step 17's stack, 2 threads, under load): one energy + forces 119–122 → 66 s,
    SPEC 26's share ~14 → 5.2 s. The [m, n S] matrices (≈ 540 MB each at 512 atoms, growing as n²S) are gone.
    The cost stays cubic, as the dense elimination is.
  - The dense route stays as the reference (`induction.USE_DENSE_REFERENCE`, tests only).
- It needs `carrier_penetration` (the same density library).

**Measured on frozen step-12 seed-42 weights (doped cell, GK overlap on; `probe_induction_overlap.py`).**
- E1 median by state: Au⁻ +0.19 eV, Au⁺ +0.008 eV, Mg⁺ +0.16 eV, O⁻ +0.08 eV. That is the sign Proposition 4
  predicts: compact Gaussians overstate the induction by diffuse carriers.
- Pair terms with both |Q| ≤ 1: |E2| ≤ 0.27 eV, 99th percentile 0.06 eV. The largest entry, 2.26 eV, is an exotic
  pair (Al with Q = −4 against O with Q = +5) that is never populated.
- Total energy +0.41 eV; Au free charges [−0.792, −0.208] → [−0.796, −0.204]; ε(Mg) median +2.08 → +2.23 eV.
- Time per energy and forces: 2.56 s against 1.56 s with GK alone, best of 3 runs on 2 threads. The
  first version took 5.1 s, rebuilding the Chebyshev basis per state and element.

**Tests** (`test_redox_induction_overlap.py`, 16):
- the split reproduces the dense δE for every configuration of random problems (4 seeds, ≤ 1e-12 relative);
- the scalable route: (K⁻¹)_dd from both operators against inv(K) (≤ 1e-11); the dipole-row terms against the
  dense terms (3 seeds, ≤ 1e-12); and the module's energy, forces and SPEC 30 record (with SPEC 28 and the band,
  several configurations) against the dense route (≤ 1e-12 / 1e-10);
- E1 is zero at frames;
- the tables reproduce the direct integral (≤ 2e-5);
- a Gaussian carrier (table replaced by λ₃) gives no correction (≤ 1e-10 eV);
- forces are −∇E with the band on;
- configuration validation.

**Limitations (stated).**
- The densities are rigid and spherical. The carrier's own polarisation by the moments is in the bound sector's
  α, not in δ.
- The moments' Gaussian shape w is kept; only the carrier side is replaced.
- The band unit's carrier (SPEC 20) has no tabulated density and keeps the Gaussian coupling.

## 27. A physical bound on the learned couplings (`RedoxConfig(coupling_bound='harrison')`) — NEW-TO-TRACE

Module: `trace_redox/heads.py` (`harrison_bound`, `TransferCoupling(bound=...)`). Probes:
`experiments/benchmarks_4g/probe_closure_size.py`, `probe_coupling_bound.py`.

**The defect (measured 2026-10-18, step 13 seed 43, epoch 19, frozen weights).**
- Relaxing the undoped wetting T5 reference, the BFGS path goes steadily downhill. Forces equal −∇E to ~1e-6
  eV/Å against finite differences, so this is the model's surface, not a force bug.
- First stage: the flat Au₂ goes ionic. Free charges go ±0.03 → ±1.00 as Au108 approaches O
  (3.50 → 2.65 Å), and the free energy falls 1.27 eV in 20 steps (short-range +0.002, bound +0.025 eV).
- Second stage: Au108 is driven into a surface Mg (1.30 Å). The learned coupling T(Mg60, Au108) grows
  0.35 → 4 → 76 → 9200 eV, so the coherent energy −|T| has no lower bound, and the SPEC 21 closure exceeds
  its cap (1994–2041 configurations).
- The coupling head is an MLP of the invariants times a radial basis (§17.6): unbounded off the training
  distribution.

**Construction.**
- Harrison's universal tight-binding matrix elements are V = η ħ²/(m d²) with |η| ≤ 3.24 (V_ppσ; ESTABLISHED:
  Harrison, *Electronic Structure and the Properties of Solids*, 1980).
- Hence B(r) = 3.24 ħ²/(m(r² + r₀²)) with r₀ = 0.5 Å: 3.80 eV at 2.5 Å, 2.71 eV at 2.98 Å, 99 eV at r = 0.
  - Harrison's 1/d² diverges at coincidence. A hopping integral between orthogonalised orbitals does not: it is
    bounded by the orbital's energy spread.
  - Measured with the bare 1/r²: the seed-43 relaxation ran the Au onto the Mg nucleus (r = 0.003 Å), with the
    free energy −2.2·10⁶ eV = −B(r).
  - r₀ changes each ratio |τ|/B by at most the factor 1 + r₀²/r². A coupling at the measured 0.65 B crosses the
    knee only for r < 1.27 Å, which is shorter than any edge in the data.
- τ = f(τ_raw): f(x) = x for |x| ≤ a = 0.75 B, and sign(x)[a + (B − a) tanh((|x| − a)/(B − a))] beyond.
  - f is C² at the knee (tanh″(0) = 0), which force training needs. It stays strictly below B.
  - It is the bitwise identity below the knee.
- Why a knee and not B tanh(x/B): the trained step-12 seed-42 couplings reach 0.650 B on all 404 AuMgO structures
  (400 test structures plus the T5 references; 1.86 eV at 2.93 Å, bare-1/r² B). A plain tanh would change them by
  up to 12 %; the knee leaves every trained value exactly as it is.

**Tests** (`test_redox_coupling_bound.py`, 5):
- with huge weights the unbounded head exceeds B while the bounded one never does;
- bitwise identity below the knee, slope and curvature continuous at it;
- forces −∇E with couplings saturated (max |τ_raw|/B = 11);
- finite at coincidence;
- configuration validation.

**Limitations (stated).**
- B bounds the magnitude only. The shape of τ(r) inside the bound stays learned (§5 of the theory note: τ has
  no physical form).
- The bound stops the coupling's runaway, not the collapse. With the bound applied post hoc, the seed-43
  relaxation still drives the Au into the Mg, because nothing in the potential repels two nuclei at short range.
  - The short-range network changes by only −35 eV at 0.003 Å: no data there, and no ZBL-type core in `flashace`.
  - The universal screened nuclear repulsion (ESTABLISHED: Ziegler, Biersack & Littmark 1985), switched off below
    the data's shortest bond, is the missing piece. It is not built: it moves a benchmark number only if a T5
    relaxation collapses, and is pre-registered as that contingency (benchmarks README).
- The first stage, the drift to ionic Au⁺Au⁻, is a configuration-energy error: the Au⁺-next-to-O²⁻
  over-stabilisation of SPEC 23, corrected by SPEC 23/25/26.
- Not added to the pre-registered steps 14–16. It is the identity on the data, but it does not prevent the observed
  failure, so it would be a moving part without a benchmark motive.

## 28. Each charge state's own polarisability (`RedoxConfig(state_polarizability=True)`) — NEW-TO-TRACE

Module: `trace_redox/polarizability.py` (`ratio_table`, `state_polarizability_corrections`); data
`trace_redox/data/ion_polarizabilities.json` (`tools/electrostatics/ion_polarizabilities.py`); probe
`experiments/benchmarks_4g/probe_state_polarizability.py`.

**Why (measured 2026-10-18).**
- The bound sector gives atom j one dipole stiffness P_jj, its isolated polarisability ℓ²P_jj⁻¹, in every charge
  state. The model's Au is trained to the neutral atom (5.43 Å³ = 36.6 bohr³).
- First-principles α (the ion-density library's X2C-PBE0/ANO-RCC; finite field at F = 2e-3 and 1e-3 au,
  Richardson):
  - Au⁻ 123.7, Au⁰ 36.50 (exp 36 ± 3), Au⁺ 13.45 bohr³;
  - checks: Na⁰ 166.8 (exp 162.7), Na⁺ 1.03 (0.99), Mg⁰ 75.1 (71.2), Mg⁺ 35.3 (35.0), Mg²⁺ 0.50 (0.49),
    Cl⁰ 14.9 (14.7), Al⁺ 24.9 (24.1);
  - O in the Watson sphere: O²⁻ 16.4, O⁻ 7.3;
  - diffuse anions not converged in the field (Na⁻, Al⁻, Ag⁻; field-step change 20–60 %) are left out.

**Construction (exact on every selected configuration).**
- α_j(n) = ρ_j(n) α_j(f), with ρ = α(Z, n)/α(Z, f): ρ(Au⁻) = 3.39, ρ(Au⁺) = 0.37, ρ(O⁻) = 0.44.
- K(n) = K + Σ_j P_jᵀ Δ_j P_j with Δ_j = P_jj(1/ρ_j − 1) (as first written; in a periodic cell the correct
  factor is the atom's own stiffness A_μ,j, §28.6, the default since 2026-11-10). By Woodbury (ESTABLISHED),
  δE(n) = ½ y_S(n)ᵀ Δ_S (I + C_SS Δ_S)⁻¹ y_S(n), with y(n) = y0 − K⁻¹Aᵀq(n), C = K⁻¹ on the dipoles, and S the
  atoms away from their frame with ρ ≠ 1.
- One isolated atom: δE = −½(α(n) − α(f)) E_loc².
- With SPEC 26 on, the source of configuration n carries the carriers' field correction, so the moments are those
  of the full source: y(n) = y0 − K⁻¹Aᵀq − Σ_u Q_u(s_u) N[:, (u, s_u)], with N = K⁻¹V reused from SPEC 26.
  - A first version used y0 − K⁻¹Aᵀq only. That is exact for SPEC 28 alone but not with SPEC 26.
  - It was caught before step 17 (which runs both) by a brute-force test of the combination.
- It is many-body (the change is in K, not in the source), so there is no exact pair split. It is added to the
  energies of the selected configurations. The importance search uses the energies without it; the selection is
  detached and δE continuous, so F stays continuous as in SPEC 21.
- Cost: one solve with K for the 3a dipole columns of the a atoms that change, plus a 3|S| solve per
  configuration. Measured: 2.34 s per energy and forces with or without it.
- **K(n) must stay positive definite (added 2026-10-19).** A softened atom (ρ > 1: Au⁻ 3.39, Mg⁺ 70.7, Mg⁰ 150.5,
  Al²⁺ 53.9, Al⁺ 94.4, Na⁰ 161.8 against the Na⁺ frame) can make K(n) indefinite, a polarisation catastrophe.
  - Its bound sector then has no minimum, and Woodbury's value is that of a saddle: finite, smooth and wrong.
  - Exact criterion (Haynsworth inertia additivity on [[K, P_Sᵀ], [P_S, −Δ_S⁻¹]], K > 0):
    In(K(n)) = In(K) + In(−(Δ_S⁻¹ + C_SS)) − In(−Δ_S⁻¹). So K(n) > 0 iff Δ_S⁻¹ + C_SS is nonsingular with exactly
    as many negative eigenvalues as Δ_S (3 per softened atom). Cost O(|S|³) per configuration.
  - Such a configuration is refused with a ValueError naming its atoms and ratios, never given an energy.
    `stiffness_defects` reports them without raising.
  - Measured on trained AuMgO bound sectors (steps 13, 13F, 14; both seeds): every constructed Al²⁺O⁻ and Al⁺
    configuration of the doped slab and every Mg⁰ one of bulk MgO is a catastrophe, as is every Mg⁺O⁻ one except
    on 13F seed 43. No selected Au⁻ configuration (136) is.
  - `RedoxConfig(state_polarizability_elements=(79, 8))` restricts the ratios to listed elements, the others
    keeping their frame α (`RedoxElectrostatics._state_ratio_table`, used by the energy and by the SPEC 30 record).
    States with ρ ≤ 1 stiffen (Δ ≥ 0), so K(n) > 0 unconditionally.
  - Found by the test itself: the random problems of the first version of the tests (K = XXᵀ + mI) had indefinite
    K(n) for at least one configuration, [0, −1, 1, −1] (minimum eigenvalue −0.35 to −3.26 over four seeds), so the brute-force check compared two saddle
    values. The problems now have weaker couplings, and every test asserts K(n) > 0.

**Measured on frozen step-12 weights** (the four T5 geometries, unrelaxed):
- δE per structure: −321 / −405 / −35 / −116 meV for seed 42 and −265 / −128 / −69 / −230 meV for seed 43 (doped
  wet / doped upright / undoped wet / undoped upright).
- T5 −961.8 → −960.4 meV for seed 42 and −659.5 → −957.0 for seed 43.
- The module and the independent probe agree to the printed 0.1 meV.

**Tests** (`test_redox_state_polarizability.py`, 16):
- the correction equals the brute-force re-solve with the modified K (4 random problems with K(n) > 0 asserted,
  ≤ 1e-12 relative);
- the inertia criterion equals the sign of the spectrum of K(n) on 90 configurations, both outcomes occurring;
- a polarisation catastrophe is refused;
- the element restriction equals the full ratios with the other rows set to 1, and is validated;
- the same with the SPEC 26 source correction (3 problems);
- the isolated-dipole limit;
- all ratios 1 give a bitwise identical energy;
- forces −∇E with the band on, and with the full step-17 stack (penetration, Gordon–Kim, induction, band);
- validation.

**Limitations (stated).**
- The ratios are free-ion values, with O in the Watson sphere. They are an upper bound for an anion confined by
  its neighbours (Fowler–Madden), and the confinement is not modelled.
- The band unit's carrier keeps its stiffness.
- **The damping width is the frame state's (2026-10-19).** SPEC 12 makes K > 0 for every geometry by deriving each
  dipole's Gaussian width from its own stiffness (σ³ ∝ ‖A⁻²‖_F^{1/2}, the Gaussian analogue of Thole's α-dependent
  damping), which leaves an on-site residual ≥ (1 − a) λ_min(A).
  - SPEC 28 softens A_j → A_j/ρ but keeps σ_j. For an isotropic atom the residual becomes λ(1/ρ − a/3^{1/4}),
    so the guarantee survives only for ρ < 3^{1/4}/a = 2.63 at the default a = 0.5.
  - Au⁻ (3.39), Mg⁺, Mg⁰, Al²⁺, Al⁺ and Na⁰ (relative to Na⁺) are outside it. K(n) may still be positive
    definite, depending on the geometry; that is what the inertia check decides, exactly, per configuration.
  - The softened state is also under-damped at contact, since its smearing is the frame ion's.
  - The consistent construction would give each state its own width (SPEC 12 per configuration). The update then
    lies on the rows and columns of S's dipoles (rank ≤ 6|S|, so still exact by Woodbury), but needs the changed
    Gram rows, including their Ewald parts. Not built.
- The reported moments (`RedoxResult.mu`) are those of the state-independent K; the energy is exact, the dipole
  output is not corrected.

### 28.6 The correction acts on the atom's own stiffness (`RedoxConfig(state_polarizability_stiffness='onsite')`, 2026-11-10) — a correction to §28

**The statement above was wrong in a periodic cell.** "Δ_j = P_jj(1/ρ_j − 1)" took P_jj from the bound sector's
operator. Under periodic boundaries that block is P_jj = A_μ,j + T2_jj, where A_μ,j is the atom's own dipole
stiffness and T2_jj the dipole–dipole lattice sum of atom j with its own periodic images (zero in an open system,
cell dependent in a periodic one). A charge state changes the atom's polarisability, which is A_μ,j and nothing
else. Scaling T2_jj by 1/ρ_j rescales the interaction of the atom's dipole with its own images. That is not physics,
and it made the energy per cell depend on the cell chosen for the same crystal.

**Corrected statement.**

    K(n) = K + Σ_j P_jᵀ Δ_j P_j,     Δ_j = A_μ,j (1/ρ_j(n) − 1)

with everything else of §28 unchanged (Woodbury, the inertia criterion, the SPEC 26 source). ESTABLISHED: an ion's
polarisability depends on its charge state; Woodbury. The correction is to TRACE's own §28, not a new term.
- `'onsite'` is the default. `'block'` is the statement as first written, kept as legacy.
- A checkpoint written before the field exists trained with the block form (train.py stores every RedoxConfig
  field, so the field's absence identifies it): the loader restores `'block'` and says so
  (`LEGACY_FIELD_VALUES`). Every model up to step 31 is such a checkpoint.
- In an open system the two forms are identical.

**Measured** (step 31 s43 frozen weights; `probe_ext_terms.py`, `probe_ext_spec28.py`, `probe_ext_x1.py`).
- Au 108: A_μ eigenvalues (7.069, 9.388, 9.390) eV/Å² in the 1×1 and the 2×1 cell; T2_jj (−0.0426, −0.0426,
  +0.0447) at 1×1 and (−0.0454, +0.0072, +0.0179) at 2×1.
- The §28 term of the in-phase Au transfers, δE(2×1) − 2 δE(1×1): block +76.584, +19.005, −83.881, −84.355 meV;
  on-site at most 7.33e-07 meV in magnitude.
- The ladder, linear, quadratic and pair-table terms are extensive (their sum is 0.000 meV in the same rows; an
  atom's pair terms with its own images move between the one-body and the pair entries, §23.x).
- Table II on frozen weights trained with the block form (a size of the change, not a score): s43 wet_u +920.36 →
  +892.68, wet_d −75.73 → −60.70, T5 −996.09 → −953.38 meV; s42 +850.97 → +832.37, −177.44 → −164.68,
  −1028.42 → −997.04 meV. The checkpoint loaded without override reproduces its stored Table II.

**Tests** (`test_redox_extensivity.py`, 12): P_jj − A_μ,j is the own-image block, zero in an open system and
nonzero and cell dependent in a periodic one; the on-site term of a replicated cell is the number of copies times
the cell's (3-D and slab) and the block term is not; the on-site correction equals the brute-force
−½ sᵀK(n)⁻¹s + ½ sᵀK⁻¹s with only A_μ,j scaled; both forms agree in an open system; the field, its validation and
the loader's legacy rule. `test_the_closed_form_holds_in_a_small_periodic_cell` (`test_redox_master_equation.py`)
encoded the block statement and is now parametrized over both forms.

**Not done** (as of 2026-11-10). No model has been trained with `'onsite'`. **[Updated 2026-11-16:** steps 33 and
34b trained with `'onsite'` (the default, explicit in their configurations).**]**

## 29. Spin-pairing factors on the couplings (`RedoxConfig(spin_pairing=True)`) — NEW-TO-TRACE

Module: `trace_redox/diabatic.py` (`spin_pairing_factors`; the `electrons` argument of `diabatic_hamiltonian`,
`switched_hamiltonian`, `configuration_importance`, `coupled_closure`, `select_configurations`).

**The defect (exact).**
- The configurations are spinless: one local orbital per charge step (§17.2). For the two-site Hubbard model
  (ESTABLISHED), the lowest-spin matrix elements are:
  - 2 electrons: covalent singlet ↔ symmetric ionic 2t, so E₀ = U/2 − √(U²/4 + 4t²);
  - 1 or 3 electrons: the resonance t, so E₀ = −t (or U − t).
- The spinless model couples the neutral configuration to each ionic one with τ, which is √2 τ to their
  symmetric combination, and the resonances with τ. No single τ is right for both: the neutral needs τ = √2 t,
  the ions τ = t.
- The learned τ does not know the charge states it connects, so it must compromise between the undoped
  (ionic–covalent) and doped (resonance) physics that T5 combines.
- Measured on Au₂ (frozen step-13 weights, the full stack): the factor raises the neutral α∥ from 13.34 to
  14.03 Å³ (seed 42) and from 12.61 to 13.51 (seed 43), toward the reference 16.5. The ions are unchanged, as
  they must be.

**Construction.**
- A move from donor d to acceptor a flips both units' electron-count parities. The number of singly occupied
  frontier orbitals among the two then changes by ±2 iff N_d ≡ N_a (mod 2) before the move: pairing
  (1,1) → (0,2), or unpairing (2,0) → (1,1).
- Such moves carry f = √2 and every other f = 1, with H_ij = sign · σ(e_ij) · f · τ.
- N_u = Z_u − n_u for atom units; a band unit holds −n electrons.
- f enters the closure search, the pair importances and the Hamiltonian alike, so the closure lemma of §21 holds
  for the couplings actually used. f is symmetric under reversing a move.
- Exact for the two-site model at every electron count, with one τ = t.

**Tests** (`test_redox_spin_pairing.py`, 7):
- the spinful Fock-space Hubbard ground energies for N = 1, 2, 3 at four U/t, against the lowest eigenvalue of
  the spinless Hamiltonian with the factors (≤ 1e-12), and the neutral defect without them;
- f = √2 exactly on equal parities;
- forces −∇E with amplitude selection and the band;
- validation.

**Limitations (stated).**
- Many singly occupied sites: f takes each moving pair in its singlet. The exact factor depends on the global
  spin state, which a Heisenberg-coupled spin ensemble would supply.
- High-spin frontier states (O⁰ 2p⁴ triplet, open d shells) are outside the parity rule. It is exact for
  s-frontier elements (H, alkali, Cu/Ag/Au 6s/5s/4s) in their lowest-spin states.
- The triplet of two singly occupied sites, at the covalent energy with no coupling, is absent from the spinless
  ensemble. That is a thermal correction of order e^(−(4t²/U)/kT).

## 30. The whole model's electronic response with several configurations (response priors) — NEW-TO-TRACE

Module: `trace_redox/response.py` (`free_dipoles`, `ensemble_polarizability`); the record `RedoxResult.ensemble`
(`RedoxElectrostatics._ensemble_record`); train.py `_response_prior_loss`.

**The defect (measured 2026-10-19).**
- The ε∞ and polarisability priors trained the bound sector's response, ε = I + 4πK_E α_b/Ω. They refused any
  structure whose window held more than one configuration, because the thermal and coherent charge-transfer
  responses are not in `bound_state`.
- Step 13F (an Au₂ α prior that raised the couplings, which share one MLP across species pairs) crashed at epochs
  7 and 15: bulk MgO, the ε∞ prior's structure, reached 5 configurations.

**Construction (exact for a state-independent K).**
- Eliminating the bound sector at fixed configuration n in a uniform field F:
  E(n; F) = E(n; 0) − F·M(n) − ½ Fᵀα_b F, with α_b = D_varᵀ K⁻¹ D_var and
  M(n) = Σ_u q_u(n) r_u + D_varᵀ y*(n).
  - y*(n) = y0 − K⁻¹Aᵀq(n), minus K⁻¹ΔF(n) with SPEC 26.
  - D_var is `assembly.dipole_derivative()` on the elimination's variables, with bond-current rows under PBC.
- Hence α = α_b + α_ens, where α_ens = −∂²F_ens/∂F² and
  F_ens(F) = −kT ln Tr exp(−[H − Σ_k F_k diag M_k]/kT). This contains Var_p(M)/kT and the coherent
  charge-transfer response.
- α_ens is computed by central differences at h and h/2 (h = 1e-3 V/Å), Richardson-extrapolated, through the
  degeneracy-safe `canonical_free_energy`. First derivatives in the weights suffice.
- **Periodic cells:** Σ q r is defined only modulo lattice vectors. With ΔQ = q(n) − q(ref) (Σ ΔQ = 0),
  M_free(n) = Σ_u ΔQ_u p_u.
  - p_u is the minimal-image position around the first atom unit whose charge changed. A band unit sits at the
    φ²-weighted minimal-image centroid of its sites.
  - For one moved electron this is the minimal-image hop vector r_d − r_a.
- The priors add α_ens: ε = ε_b + 4πK_E α_ens/Ω, and α (Å³) = α_b + K_E α_ens.
- **With SPEC 28 (exact, 2026-10-19).** Each configuration has its own stiffness K(n) = K + P_Sᵀ Δ_S P_S, so
  E(n; F) = E(n; 0) − F·M(n) − ½ Fᵀ α_b(n) F with α_b(n) = D_varᵀ K(n)⁻¹ D_var. By Woodbury, with G = K⁻¹ D_var,
  C = K⁻¹ on the dipoles and W_S = Δ_S (I + C_SS Δ_S)⁻¹:
  - y(n) = K(n)⁻¹ K y_ref(n) = y_ref(n) − K⁻¹ P_Sᵀ W_S y_S(n), so M(n) gains dM(n) = −G_Sᵀ W_S y_S(n);
  - α_b(n) = α_b + dα(n), dα(n) = −G_Sᵀ W_S G_S (symmetric: W_S = (Δ_S⁻¹ + C_SS)⁻¹);
  - the field enters the diagonal as H(F) = H − diag[F·M(n) + ½ Fᵀ dα(n) F]; the common ½ Fᵀ α_b F gives α_b.
  - Consistency: SPEC 28's dE(n; F) = ½ y_S(F)ᵀ W_S y_S(F) with y_S(F) = y_S + G_S F expands to exactly these
    two terms, so the decomposition holds to all orders in F.
  - `polarizability.state_polarizability_response` returns (dM, dα); the record carries `alpha_shift = dα` and
    `ensemble_polarizability(..., quadratic=alpha_shift)` takes it.
- The record is marked inexact, and the prior refuses, only with the core + carrier elimination.

**Tests** (`test_redox_ensemble_response.py` 12; `test_redox_polarizability_prior.py` 2 replaced;
`test_redox_dielectric_train.py` 1):
- the two-level closed form (μ²/4) tanh(t/kT)/t to 1e-7, at three (t, kT);
- minimal-image hops across a periodic boundary;
- the decomposition equals the whole model's finite-field α of the total energy, all three components, to 1e-5,
  under window selection and under amplitude selection with saturated couplings;
- the switch-band leak (below);
- SPEC 28's (dM, dα) by Woodbury against brute-force inversion of K(n), three random problems, to 1e-11;
- with SPEC 28 on, Au₂ at 2.5 Å (3 configurations): the decomposition equals the whole model's finite-field α to
  1e-5. Measured: 0.393306, 0.430776, 1.051144 e²Å²/eV both ways; without dα, 0.226, 0.242, 0.983 (counterfactual
  asserted visible);
- the polarisability prior trains a coupled Na₂⁺ (two degenerate configurations, refused before) and equals the
  trained checkpoint's finite-field α to 1e-5;
- the ε∞ prior trains a periodic Na₂ cell with 2 configurations (α_ens along the bond 0.53 e²Å²/eV, measured).

**Limitation found with it (SPEC 21).**
- Inside the switch band (a_off < e < a_on), the coupling σ(e)τ depends on the configuration energies, which shift
  with the field. The model's whole −∂²E/∂F² then contains the switch's field derivatives, which are not physics.
- Measured on a random test model: −3.99 against the fixed-H +2.36 e²Å²/eV.
- On the trained step-13 models no Au₂ pair sat in the band, and whole model and fixed H agree to 0.01 Å³ in all
  six charge-state and seed cases (`probe_switch_leak.py`).
- The record, and so the priors, carry the fixed-H physics.

## 31. Harrison-anchored couplings (`RedoxConfig(coupling_anchor='harrison')`) — NEW-TO-TRACE

Module: `trace_redox/heads.py` (`frontier_shell`, `anchored_pairs`, `window_switch`, `TransferCoupling(anchor=...)`;
`open_frame`, the first rule, kept). Probes: `experiments/benchmarks_4g/probe_otop_channels.py`,
`probe_anchor_counterfactual.py`, `probe_t5_first_order.py`; first principles: `tools/electrostatics/almo_split.py
--direction`.

**The defects (measured 2026-10-27; theory note 2026-10-18 §4c, benchmarks README).**
- PBE's relaxation at Au₂'s O-top contact is −1.98 eV (28 ions). The trained model's is −0.22 / −0.14 eV (step 17,
  product-state split). 83 % of PBE's transfer there is O → Au.
- The model's O → Au configuration (Au⁻ + an O hole) is coupled by 0.1–1.0 eV against Harrison's V_spσ(2.26 Å) =
  2.75 eV. F is even in τ, so ∂F/∂τ ∝ τ p/ΔE, and a coupling to a configuration eV above the frame keeps its small
  initial value (scale 0.01): the vanishing-gradient trap of Theorem 3, here on τ.
- Couplings that no data constrain are a seed lottery. Hole configurations never enter the neutral training
  ensembles; step 20 seed 42 learned O–O 28.1 eV (ten times V_ppσ at 3.0 Å), seed 43 0.37 eV. Seed 42's clean-slab
  IP is −267.7 eV, its hole's coherent energy −275.4 eV.

**Construction.**
- τ_uv = η(l_u, l_v) ħ²/(m (r² + r₀²)) · exp(c tanh(x/c)) · w(r) on every anchored pair. Here x is the head's learned
  radial sum, c = ln 2 (factor in (½, 2)), r₀ = 0.5 Å (SPEC 27), and w the C² quintic on [r_cut − 1 Å, r_cut], equal
  to 1 below it.
- η: ss −1.40, sp +1.84 (σ), pp (3.24 + 2(−0.81))/3 = +0.54. These are Harrison's universal coefficients
  (ESTABLISHED: Froyen and Harrison, Phys. Rev. B 20, 2420 (1979); Harrison 1980).
  - s–p takes σ: a single partner couples to one combination of the p hole, the σ one.
  - p–p takes the trace average: a hole in a lattice cannot be σ to every neighbour. Measured: the anchored
    clean-slab IP is +5.14 / +3.90 eV against PBE's +4.80.
- The frontier shell l is s for H, He and groups 1, 2, 11, 12, and p for groups 13–18. d- and f-block partners of an
  anchored pair are refused.
- Anchored pairs: all but a cation and an anion of the formal frame (f_u f_v < 0).
  - Those transfers cross the ionic gap. Their second-order energy is the frame's covalency, smooth and
    state-independent, and it stays in the backbone and the moments (SPEC 22). They keep the learned value under
    SPEC 27's bound.
  - Every other transfer moves a carrier: a hole between anions, an electron between cations, an electron to or
    from a neutral frame (Au⁰).
- The state dependence follows from the configuration picture. An O → Au move needs an Au that can accept, so Au⁻
  (the doped dimer) cannot accept again (min charge −1), while Au⁰ can.

**Two first versions, corrected before any run (probe_anchor_counterfactual.py, frozen step-20 weights).**
- The learned head's full-range switch f(r/r_cut) is 0.38 at 2.26 Å and 0.26 at the Au–Au bond. It took 60–75 %
  off Harrison's values; replaced by the 1 Å window.
- Anchoring only the open-shell frames (the Au pairs) let seed 42's opened O → Au channel delocalise its hole
  through the 28 eV O–O couplings: wet_u +541 → +4990 meV. The cation-anion rule anchors O–O too: +1094 (Au–O and
  O–O), +989 (all).

**Tests** (`test_redox_coupling_anchor.py`, 8):
- the frontier shells and the anchored-pair rule;
- zero learned weights give Harrison's value on Au–O, Au–Au, Au–Mg, O–O, Mg–Mg (1e-12) and zero on O–Mg;
- the factor stays in (½, 2) under huge weights, and the cation-anion pairs are SPEC 27's bound of the raw value;
- symmetry, and the species required;
- d-block refusal;
- off by default, the species changing nothing when off, and validation;
- forces −∇E with the anchor on;
- the module routes the species (T(O, Au) equals the anchored value).

Old checkpoints load and give bitwise-identical energies (step 20 seed 42, the undoped upright reference:
−1476723.3224627066 eV with and without SPEC 31's code).

**Limitations (stated).**
- One orbital per unit: the p–p average and the s–p σ choice are approximations of the full Slater–Koster blocks.
- Harrison's coefficients are universal fits, about 30 % from specific materials, which the learned factor (½, 2)
  absorbs.
- The d block is refused, so transition-metal adsorbates need d-shell coefficients (Harrison's r_d³/d⁵ scaling)
  first.

## 32. The downfolded O 2p ↔ metal s bond (`RedoxConfig(downfold_transfer=True)`) — NEW-TO-TRACE

Module: `trace_redox/downfold.py` (`downfold_moves`, `dressing`, `block_ground`, `star_ground`,
`drop_folded_partners`, `DownfoldError`); the selection's `correction` / `bound` arguments in
`diabatic.search_configurations`, `coupled_closure`, `select_configurations`; the module's wiring in `forward`
(`_diabatic_solution(..., selection_correction, selection_bound, exclude)`). Probes:
`experiments/benchmarks_4g/probe_downfold_counterfactual.py`; first principles: `tools/electrostatics/almo_split.py
--direction`, the flat-height scans. Config: `downfold_transfer` (off), `downfold_cutoff` (4.5 Å).

**The defect (measured 2026-10-27/28; benchmarks README, theory note 2026-10-18 §4c).**
- PBE's relaxation at the O-top contact of Au₂ on MgO(100) is −1.96 eV (28 ions: polarisation −0.86, transfer
  −1.10), and 94 % of the transfer's wetting share is O → Au donation. The trained model's relaxation there is a
  seventh of it (−0.29 / −0.22 eV, step 17), with the explicit O → Au coupling at 0.1–1.0 eV against Harrison's
  2.6 eV, held small by the vanishing gradient of an even F(τ).
- Step 21 (SPEC 31, Harrison-sized couplings inside the explicit ensemble) failed: the selected ensemble is not
  converged in its threshold (the slab alone moves by eV), SPEC 21's switch adds 0.2–0.7 eV/Å of non-physical force,
  and short Au–O contacts overbind by 3 eV.

**Construction.** A strong local bond is a two-site problem, not a perturbative ensemble. The O–s channel is folded
into the configuration energies (Löwdin partitioning, ESTABLISHED: P.-O. Löwdin, J. Chem. Phys. 19, 1396 (1951)):
- the channel: every one-electron move between an O-class atom (a frame anion with a p frontier: O, the halides) and
  an s-class atom (a neutral frame with an odd electron count and an s frontier: Au, Ag, Cu, H), in either direction
  and in any ladder states. The module zeroes these pairs' explicit couplings (both orientations), the selection
  excludes a configuration whose folded manifold is not above it, and a selected configuration reachable from a
  lower selected one by a folded move is dropped (`drop_folded_partners`, to a fixed point);
- per explicit configuration n and per contact cluster (atoms joined by O–s edges within `downfold_cutoff` and by
  s–s pairs within it), the folded states q are the single-move configurations with Δ_q(n) = E(q) − E(n) exact
  (every other term of E, SPEC 28 included), coupled to n by

      v_q = f_spin √g_orb V(r),    V(r) = η_spσ ħ²/(m (r² + r₀²)) w(r),    η_spσ = 1.84,  r₀ = 0.5 Å,

  Harrison's universal V_spσ (ESTABLISHED: Froyen and Harrison, Phys. Rev. B 20, 2420 (1979)); r₀ is SPEC 27's
  regulariser and lowers V by 4.7 % at 2.26 Å (2.617 against 2.745 eV; stated, since the universal value is itself
  a ~30 % fit); w the C² quintic on [r_cut − 0.5 Å, r_cut]. Measured against the ion library's PBE0 monomers,
  |⟨Au 6s|O 2pσ⟩|(r)/S(2.26 Å) = 0.62 / 0.53 / 0.44 / 0.37 / 0.30 / 0.25 at 3.0 / 3.2 / 3.4 / 3.6 / 3.8 / 4.0 Å
  against 1/(r² + r₀²)'s 0.57 / 0.50 / 0.44 / 0.39 / 0.35 / 0.32: the form holds within 10 % to 3.6 Å, so the switch
  sits beyond it (a first version switched on [3, 4] Å, where the flat dimer's contacts are, and suppressed the flat
  transfer by ×0.2–0.3; measured 2026-10-28);
- f_spin is SPEC 29's factor from the two atoms' electron-count parities before the move (1 for O²⁻ → Au⁰, the
  three-electron resonance; √2 for O²⁻ → Au⁺, the unpairing move); g_orb the isotropic fraction of the σ p orbital
  that can donate (its occupation / 2: 1 for O²⁻, 5/6 for O⁻) or accept (its vacancy / 2: 1/6 for O⁻);
- a closed-shell O donor (O²⁻) is resolved into its three p components with v_m = f V l_m, l the direction cosines
  O → s (Slater–Koster, ESTABLISHED: Phys. Rev. 94, 1498 (1954)), Σ_m v_m² = f² V²; inside the folded manifold the
  moved electron hops between the cluster's s atoms with ±|T_ss'| f_hop (the model's own s–s coupling as a
  magnitude; the sign is `downfold_hop_ordering`: `bonding` (−, legacy: the bonding two-site state lowest) or
  `pauli` (+, 2026-10-29: the antibonding one lowest — the pair's own s electrons fill the bonding orbital, so the
  extra electron takes σ*). The sign is a gauge for a donor coupled through one path and physical for a donor
  coupled to both atoms of a pair, whose loop phase sign(v_g T v_g′) it fixes; the statement of 2026-10-28 that
  the sign is a gauge the explicit Hamiltonian never trained is replaced, CHANGELOG Round 28). This is the
  moved electron's resonance over Au₂: with the model's numbers (V 2.617, Δ 3.16–3.70, |T_AuAu| 2.11–2.37 eV) it
  adds 0.20–0.32 eV at the O-top contact against the uncoupled star (measured in review, 2026-10-28);
- λ(n, cluster) is the lowest eigenvalue of [[0, vᵀ], [v, M]], M = diag(Δ) + T_QQ, from the secular equation
  λ = vᵀ(λ − M)⁻¹v below μ_min (M's lowest eigenvalue; the left side minus the right is increasing there), by
  bracketed bisection on (min(0, μ_min) − ‖v‖ − 1, μ_min) and two Newton steps with the graph from the detached root
  (exact value, first and second derivatives; `torch.linalg.solve`, no eigenvector derivatives); E(n) → E(n) +
  Σ_clusters λ. The selection (the 2.5 eV window flood fill and the SPEC 21 closure) sees the dressed energies
  (`correction`), with a per-atom bound ‖v‖_cluster on how much one move can change the dressing as the move
  pre-screen's margin (a superset, so the window's stated exactness is kept).

**Properties (each with a test).** Size-consistent: one root per cluster, additive over clusters. Bounded:
λ ≥ min(0, μ_min) − ‖v‖. −Σ v²/Δ at weak coupling. State-dependent: an Au⁻ cannot accept (the doped dimer is
dressed less), an Au⁺ accepts with √2. Valid while μ_min > 0, the folded manifold above the explicit state (the P/Q
separation); otherwise the configuration is a folded state of something lower: the selection excludes it (+inf) and
the final solve refuses loudly (`DownfoldError`). The gradient ∂λ/∂Δ ≈ +0.23 at the contact is of order one, so the
O⁻ and Au⁻ level shifts, which set Δ, receive a gradient that an explicit coupling to a state 3 eV up never gives
(Theorem 3's trap, 2026-10-08); v is fixed, so Δ is identified by the data.

**The two-site states of a bonded pair (2026-10-29, the flat acceptor run; NEW-TO-TRACE).** Every s-class atom of a
contact cluster holds a folded state for each of the cluster's donors, with zero coupling where the donor does not
reach it (t = 0, l = 0: `downfold_moves`'s partner states), so a donor touching one atom of a bonded pair still sees
the pair's two-site states through the s–s hop (before: no such state, and a standing dimer's extra electron could
not delocalise; the on-axis test measured fold −1.516 against the dense −1.552 eV). With the Pauli ordering a donor
on the pair's bisector plane reaches the lower (antisymmetric, σ*) state only through its p component along the
bond axis and the upper (symmetric) one through its other two components: the measured suppression of the lying
dimer's contacts (0.41 per contact against a standing acceptor at the same site; its gained density 80 % p-type, its
σ* unshifted at −4.41 eV and 3.44 eV below the next virtual; theory note §4c). A leaf no chain of nonzero hops joins
to a coupled leaf is absent from the block (an unreachable state neither bounds the root nor enters the solve).
Tests: `test_the_pauli_ordering_of_a_bonded_pair`; the move-class test's partner state.

**The fold on the whole explicit block (`downfold_block: exact`; 2026-10-29; NEW-TO-TRACE, ESTABLISHED ingredient:
Löwdin partitioning with a multidimensional P, Löwdin 1951).** The folded states are shared between explicit
configurations (the frame reaches the ionic structure's folded state through the partner hop, the ionic structure
reaches it directly), so a self-energy per explicit state double-counts them whenever more than one explicit state
carries weight. With the explicit block H₀ = diag(E) + T (the configurations' energies with the inner corrections and
the model's couplings), one folded state per (hole configuration, O, p component or isotropic kind) reached from every
configuration that can move an electron into it (V [C, N_Q]), and the s–s hops between folded states that differ by
one electron between two s atoms of the cluster (±|T_ss'| f, the ordering above; H_QQ), each contact cluster k has
the fixed point

    E_k* = lowest eigenvalue of [[H₀, V_k], [V_kᵀ, H_Qk]],   Σ_k = V_k (E_k* − H_Qk)⁻¹ V_kᵀ,   H = H₀ + Σ_k Σ_k,

so that H₀ + Σ_k has E_k* as its lowest eigenvalue exactly (the Löwdin effective Hamiltonian at the block's own ground
energy); Σ's diagonal is each configuration's dressing and its off-diagonal the dressed couplings. E_k* is solved as
the fixed point E = λ_min(H₀ + Σ_k(E)) — bisection below the folded manifold, two Newton steps with the derivative
1 + ‖X w‖² (X = (E − H_Qk)⁻¹ V_kᵀ, w the C × C ground vector, detached) — not by an eigendecomposition of the full
block, whose backward divides by every eigenvalue difference and is infinite on the degenerate folded pairs in the
training's double backward (2026-10-29). One fixed point
per cluster keeps the dressing additive over separated clusters (the joint eigenvalue of two independent clusters is
not: the explicit-plus-folded space lacks their product states, the usual size-consistency failure of a truncated
configuration space), and with one explicit configuration Σ is the per-state secular root (tests
`test_the_exact_block_reduces_to_the_per_state_root_for_one_configuration`,
`test_the_exact_block_matches_the_dense_eigenvalue_with_shared_states`,
`test_the_exact_block_is_additive_over_separated_clusters`, `test_the_module_runs_the_exact_block`). A cluster whose
block ground state is not explicit-dominated (E_k* not below its folded manifold, or explicit weight ≤ ½) is refused:
the charge-transfer state must be explicit there. Folded states no chain of nonzero hops joins to a coupled one are
absent. **Promotion (2026-10-30; NEW-TO-TRACE).** A cluster whose block ground state is not explicit-dominated (E_k* not below
its folded manifold, or explicit weight ≤ ½) is the strong-coupling regime — a folded state below the explicit
block's ground energy, as at a close O contact — and the reduction above is the wrong description there. The fold
then promotes the block's dominant folded state (the largest |w_q|² of the ground vector) to an explicit
configuration: its charge configuration joins the explicit set with the folded state's energy on the diagonal and
the fold's couplings V_cq as its couplings to every explicit configuration; it leaves the folded space, its hops to
the remaining folded states become its couplings, a folded state equal to any other explicit configuration is
dropped (its coupling is explicit already), its component siblings stay folded, and **its own moves are never folded**: promotion re-partitions the block it
came from and never enlarges the folded space (a promoted configuration's two-electron states are outside the
fold's scope), so the promoted system's ground energy is the block's exactly. The request names the set of folded
states in descending weight until the explicit weight reaches 0.6 (at most 32 per pass), with their mutual hops
as explicit couplings among them; at most four passes per solve; recorded (`promoted`, `folded_states`). Tests:
`test_the_exact_block_requests_promotion_in_the_strong_coupling_regime`,
`test_promotion_is_a_repartition_of_the_same_block`, `test_the_module_promotes_and_records`.

Stated approximations of this mode: the selection's screen keeps the per-state dressing; the thermal
ensemble's excited eigenvalues of H₀ + Σ are the self-energy at E_k*, exact for the ground state only; the hop
structure is built per folded state in Python (N_Q × pairs lookups, fine for dimers; to be vectorised before metal
clusters). `per_state` (the default) is the 2026-10-28 path unchanged.

**The coupling's radial form (`downfold_decay`, `downfold_decay_anchor`; 2026-10-29; measured).** Harrison's
V_spσ(r) ∝ 1/(r² + r₀²) is the fold's default. Read against its own ALMO Δ, the vertical O-top series' effective
coupling √(|CT| Δ) is 1.79, 0.69, 0.44 eV at 2.26, 3.0, 3.5 Å — an e^(−1.1 r) decay, 3.9× over the range where
Harrison falls 2.33× (ESTABLISHED: a Slater-type overlap decay, V ∝ S). The option multiplies V by e^(−κ s(r − r₁)) with
s(x) = w softplus(x / w), w = 0.2 Å: Harrison's coupling at and below the anchor (the measurement covered 2.26–3.5 Å
only; the plain exponential raised the coupling below the anchor and accelerated a short-range collapse), the
measured e^(−κ (r − r₁)) above it, C¹ with slope ≤ κ, the anchor contact reduced by e^(−κ w ln 2) (6 %); κ ≥ 0 per Å
beyond Harrison (measured 0.45), r₁ the O-top distance 2.26 Å; zero is Harrison bit for bit. Test: `test_the_coupling_decays_beyond_harrison_when_asked`.

**Derivatives (2026-10-30).** The fold's energy, forces and the force loss's parameter gradient are exact: the
s–s hops carry their gradient (the coupling head's, and the positions'); a folded state shared by several
configurations takes its energy from one representative leaf; the exact block's fixed point uses a ground vector
exact to first order (reduced resolvent) inside a Rayleigh quotient, so its second derivatives include the
eigenvector response. Tests compare autograd with finite differences at first and second order, on a toy with
position-dependent energies and on the module with hop-connected folded states.

**Stated approximations.**
- The hops inside the folded manifold are the s–s ones only: a hole does not hop between O donors there (the learned
  O–O coupling is unconstrained by neutral data: 28 eV in step 20 s42). Isotropic leaves (O⁻ donating or accepting)
  carry no hop.
- A cluster that splits into disconnected parts in a given configuration (a bridging atom in a non-frame state) is
  still one root: mutually exclusive rather than additive, an under-estimate.
- The self-energy is evaluated at each explicit state's own energy E(n) + λ_n, exact when one explicit state
  dominates; with the undoped upright's frame (0.56) and Au⁺Au⁻ (0.44) mixed through 1.25 eV the second-order error is
  O((v²/Δ²) t_PP²/Δ) ~ tens of meV. Two explicit states that share a folded state (the frame and Au⁺Au⁻ through
  (O⁻, Au⁰, Au⁻) in the flat geometry) also have an effective coupling v v'/(E − E_q) that the per-state dressing
  omits.
- The SPEC 30 response record carries H's dressed diagonal but not ∂λ/∂F; it is marked inexact whenever a
  configuration is dressed (the priors refuse). The AuMgO priors (Au atom, MgO bulk) hold no O–s pair.
- The flood fill does not expand through a configuration whose folded manifold lies below it (it is excluded), so a
  valid state reachable only through such a configuration is not found; none is known.

**Tests** (`test_redox_downfold.py`, 13): the star root against the two-level formula and the dense eigenvalue
(degenerate leaves) to k·eps·‖H‖; the block root against the dense eigenvalue with hops, a padded block, a leaf
reachable only through a hop, degenerate leaves, and the Au–Au resonance (> 0.25 eV below the star); first and second
derivatives against central differences of the dense eigenvalue; the weak-coupling limit, the Weyl bound, the empty
block; a non-finite leaf absent, μ_min ≤ 0 refused (a single negative Δ; a resonance pulling μ below zero), an
uncoupled block 0; the class masks and the contact clusters (a far Au joins through the hop, edges and direction
cosines); the module's dressing against an independent dense rebuild on an upright test cluster (both directions,
orbital and spin factors, direction cosines, hops; the explicit O–Au couplings zero both ways, Au–Au learned); size
consistency on two contacts 30 Å apart for three explicit states; the selection sees the dressing (a state inside
the window only dressed is kept, the folded partner is not) and the partner drop; the Au⁺ pairing factor; off by
default, validated (needs the formal diabatic ensemble; a positive cutoff), forces −∇E with the fold on; the off path
leaves T and the record untouched; the response record marked inexact when dressed.

**The pair's measured constants (2026-10-29; `downfold_scale`, `downfold_anisotropy`, `downfold_crystal_field`,
`downfold_bond_cutoff`; defaults 1, 0, 0, 3.2 Å leave the plain term bit for bit).** Measured on the MgO(100) ten-ion
cluster with PBE (theory note 2026-10-18 §4c):
- the vertical O-top transfer follows Harrison's form to 3.5 Å at 0.66–0.99 of it (c ≈ 0.71 with the anisotropy
  below); the direct overlap of the Au 6s with the cluster's O 2p band orbitals is monotonic and within 10 % of
  Harrison's to 3.5 Å (a first construction from Löwdin natural orbitals was an artifact, recorded there);
- a single closed-shell Au⁺ acceptor transfers 0.65 per contact over a cation of what it transfers straight above an
  O at the same distance: the O 2p band's in-plane (cation-bonding) hole components lie 0.8–1.3 eV below the dangling
  p_z. **Crystal field:** each resolved p component of a hole on an O-class atom costs β per frame-cation neighbour
  along its axis (within 1.2 × the nearest-cation distance, |l·e_m| > cos 30°) beyond the least-coordinated axis:
  nothing in the bulk, β on a (100) surface O's in-plane components; β = 1.0 eV measured;
- the Au₂ dimer's flat contacts transfer a further 0.55 less per contact than the single acceptor: its acceptor is
  the sp-hybridised σ\* (PBE LUMO: 6s with p_z admixture, Mulliken κ ≈ 0.4), whose lobes point outward along the
  axis and whose node faces the inward side. **Anisotropy:** a bonded s-class acceptor (an s–s partner within
  `downfold_bond_cutoff`) couples with (1 + κ′ cos θ)/√(1 + κ′²), cos θ = ŵ·ê, ŵ the acceptor → donor direction, ê
  minus the sum of its bond unit vectors (no lobe at a symmetric site); κ′ = 0.22 from the vertical and inward-flat
  transfers (the LUMO's κ V_ppσ/V_spσ ≈ 0.7 overshoots: the O 2p – Au 6p element is well below Harrison's);
  κ′ = 0 for an acceptor without a partner, the Au⁺ case.
These are constants of the (O 2p, Au 6s) pair, fixed, tabulated like the Pauli constants, not learned. Tests: the
crystal-field counts on a surface and a bulk-like O and their entry into Δ; the lobe direction, the factor at cos θ =
+1 and on the inward side, and no lobe for a lone atom; the configuration (defaults, validation, bitwise return).

**The σ\* anisotropy falsified; the dangling-axis rule (2026-10-29, 06:35).** The 28-ion ALMO split gives
CT_cp(flat) = −0.444 eV against the σ\* prediction of < −0.80 (count-only ≈ −0.47): the outward O's are not
enhanced, inward and outward flat contacts transfer alike at ≈ 0.3–0.36 of a vertical contact at equal distance, and
κ′ = 0 (the code path stays, off). The Au⁺ angular test is withdrawn: Au⁺ over Mg²⁺ relieves a cation–cation
repulsion when it takes the electron, so its 0.65 mixes electrostatics with the angle. What the data support on the
donor side: with the Slater–Koster projection an acceptor at 45° has half its coupling² on the dangling p_z and half
on the cation-bonding p; killing the bonding components gives an angled/vertical ratio of 0.50 in λ, measured
0.30–0.36, so the bonding components are at least dead (a localised hole in an O–Mg bonding p costs the bond, not the
band's 0.8–1.3 eV splitting), and a residual ≈ 0.85 in v on the dangling channel at 45° is unexplained and stated.
**Rule (`downfold_free_axes`): a hole on an O-class donor is resolved onto its least-coordinated p axes only (the
axes with the fewest frame-cation neighbours within 1.2 × its nearest-cation distance, |l·e_m| > cos 30°); the
others do not couple.** A (100) surface O: p_z alone; a bulk O or a bare O: all three; an edge O: its free axes.
With it the vertical data give c = 0.85 and β is unused. Tests: the free axes of a surface, a bare and a
one-cation O (a cation untouched); the dressing under the rule equals the star with (V l_z)² against V² without it.

**A pair inside one covalent unit is never folded (2026-10-29).** The formal frame is per element, so water's O is
an O-class donor and its H an s-class acceptor; the fold would act across every O–H bond (V_spσ at 0.96 Å is 12 eV).
`downfold_moves(units=...)` skips a pair whose atoms share a covalent unit (SPEC 19/22: a bound-sector bond), as the
ensemble's admissibility skips the same electron–hole pair; with atom units (MgO, Au) nothing changes. Test: a water
molecule beside an O–Au contact folds the contact only.

**Review record (2026-10-28).** A first version (same day) took the folded states uncoupled, chose the explicit set
on undressed energies, let folded states re-enter the explicit set, masked the explicit O–Au couplings by species
while dressing the Au⁰ acceptor only, returned NaN for a non-finite or non-positive Δ, and switched the coupling on
[3, 4] Å; a two-pass review (code and physics) found each, and this section's construction is the corrected one.

## 33. The bond-oriented Pauli repulsion of a bonded s atom against a closed-shell anion (`RedoxConfig(bond_pauli=True)`) — NEW-TO-TRACE

Module: `trace_redox/bond_pauli.py` (`BondPauli`, `bond_pauli`); the module's wiring beside the frame wall in
`forward`. Config: `bond_pauli` (off), `bond_pauli_level` ('almo' | 'frz'). First principles:
`tools/electrostatics/frz_nonadditivity.py`, `frz_angular.py`, `almo_angular.py`, `bop_flat_check.py`; frozen
checks `experiments/benchmarks_4g/probe_bop_frozen.py`, `probe_tilt_path.py`; theory note 2026-10-18 §4c (2026-11-03/04).

**The defect (measured 2026-11-03/04).**
- Steps 21–24 all failed the same way: the flat (wetting) Au₂ of the T5 protocol, a PBE minimum behind a ≈ 0.25 eV
  barrier (25 and 28 quantum ions agree to 0.01 eV), rolled over into a tilted state with one Au 2.2 Å above an O. PBE
  puts that state 0.78–0.82 eV above the upright dimer on the same O; the models put it 0.24 eV below to 0.34 above.
- Along the rigid rotation of the dimer about its O-top contact (17 ions), PBE rises +0.36 / +0.64 / +0.75 eV at
  40 / 60 / 70°; the trained models (steps 20, 24; both seeds) stay within −0.22 … +0.54.
- Decomposed (ALMO, 6 ions, checks a1–a3): tilted − upright = FRZ +1.40 + POL −0.46 + CT −0.06 eV. Not charge transfer
  (the O → Au donation keeps 95 % at 70°, `almo_tilt.py`), not field polarisation (it favours the tilt). The frozen
  term is not pairwise in atomic densities: free-ion pair energies give +0.79 of the +1.40, and in the cluster's own
  densities E1(dimer) − ΣE1(atoms) is −0.76 eV upright against +0.19 tilted. The σ bond shields the contact atom when
  it points away from the anion.
- TRACE's overlap terms (§23, §25) are isotropic atom pairs relative to the frames; the backbone learned the shielded
  upright contact from the data and transferred it to every orientation.

**Construction (the law, measured).** For a bonded s-class atom a (frame 0, open frame, s frontier: Au, Ag, Cu, H)
with partner(s) b and a frame anion j at R = R_aj,

    E_bp(a, j) = λ_a s_c(R) [ D₁(R) cos θ + D₂(R) P₂(cos θ) ],   cos θ = r̂_aj · (−ê_a),
    D₁(R) = C₁ f(R)^p₁,   D₂(R) = C₂ f(R)^p₂,   f(R) = E1(R) / E1(2.26 Å),

- ê_a: the unit vector from a toward its partner(s); λ_a the resultant's length of the s_b(R_ab)-weighted partner unit
  vectors (s_b 1 at ≤ 2.9 Å, 0 at ≥ 3.4), capped at 1; a symmetric site has no lobe and no term.
- s_c: a C² switch, 1 at ≤ 3.8 Å, 0 at ≥ 4.3 Å.
- E1(R): the lone Au⁰'s frozen repulsion on the O-top site (1.4254 / 0.8454 / 0.3891 / 0.1045 eV at 2.26 / 2.4 / 2.6 /
  2.9 Å; linear in ln E1 between, the end slopes beyond). Tabulated pair: (Au, O). An untabulated bonded pair within
  the cutoff is refused.
- The constants are fitted jointly over the measured R and θ = 0–70° to the module's own two-atom form (the far atom's
  term is part of the fit, not absorbed into the contact's coefficients); 90° is excluded, since there the far Au sits
  2.26 Å over a Mg, whose non-additivity the law has no term for and the measured D(90°) holds.
- Levels (fitted on the extended table, `tools/electrostatics/fit_bond_pauli.py`): `frz`: C₁, C₂ = −0.072, −0.665 eV,
  p₁, p₂ = 2.29, 0.500 — the frozen densities, four R (2.26–2.9 Å), largest residual 3.7 meV; p₂ = ½ as exchange ∝ S²
  and the anisotropy ∝ S, the small cos part decaying as S^4.6. `almo` (default): the orbitals relaxed with charge
  transfer forbidden, the physical target, since the bound sector answers electric fields only and cannot supply the
  Pauli-driven relaxation (the lone atom's ALMO relaxation recovers 0.90 of 1.37 eV at 2.26 Å, 0.30 of 0.35 at 2.6,
  0.11 of 0.09 at 2.9): C₁, C₂ = −0.069, −0.497 eV, p₂ = 0.382, p₁ held at the frozen fit's earlier 1.877 (three R do
  not determine the small cos term's decay: freed, it runs to 0); three R (2.26, 2.6, 2.9 Å), largest residual 6.2 meV.
  The relaxed P₂ anisotropy decays more slowly than the frozen one (p₂ 0.38 against ½; 87 % kept at 2.6 Å against 64 %
  at 2.26); the frozen-level scaling pre-registered for it failed at 2.6 Å, and the two-R constants predicted the third
  R to 12 % (pre-registered PASS, 2.9 Å).
- E1(R) is tabulated at 1.8, 2.0, 2.1, 2.26, 2.4, 2.6, 2.9, 3.2, 3.5, 3.8 Å (7.73, 3.70, 2.56, 1.43, 0.85, 0.39, 0.10,
  0.013, −0.012, −0.017 eV; it turns attractive beyond 3.3 Å, penetration electrostatics over Pauli) and interpolated by
  monotone cubic Hermite forms (Fritsch–Carlson; ESTABLISHED): in ln E1 over the positive points for the law's f(R), in
  E1 for the wall with the positive knots' slopes from the ln form. C¹: the forces are continuous at the knots (tested),
  the second derivative may jump there.
- The isotropic part D₀(R) (+0.18 eV at 2.26 Å relaxed; ≤ 0.04 frozen) is left to the backbone: a pair function of R.
- Measured: per-R angular fits with residuals ≤ 8 meV (frozen) and 4 meV (relaxed); the frozen radial law within 9 %
  over a 14-fold range of E1 (pre-registered test at 2.4 and 2.9 Å, PASS; the joint fit then within 2 %); at the flat
  geometry PBE's non-additivity +0.109 eV against the frozen law's +0.055 (10 ions; PASS by the 0.1 eV criterion, at
  half the value).
- A frame term, configuration independent, trained with (as the frame wall, §31 Round 31).

**The wall part (`bond_pauli_wall`, 2026-11-05; needs `bond_pauli`).** E_wall(a, j) = s_c(R) I(R) for every s-class
atom a (bonded or not) and frame anion j: the lone Au⁰'s interaction with the O-top site of the cluster **at the law's
own level** — frozen (E1: penetration electrostatics + Pauli; the table of f(R)) for `frz`, relaxed with charge transfer
forbidden (I_ALMO: 3.814 / 1.637 / 1.039 / 0.472 / 0.047 / −0.028 / −0.035 / −0.027 / −0.019 eV at 1.8 / 2.0 / 2.1 / 2.26 /
2.6 / 2.9 / 3.2 / 3.5 / 3.8 Å: the lone atom's relaxation recovers 0.90 of 1.37 eV at 2.26 Å and 0.51 of 1.04 at 1.8, and a
shallow well of −0.035 eV sits near 3.2 Å) for `almo` — the quantity that level's law is measured relative to, so that wall + law is the dimer's
interaction at that level (relaxed: −0.030 / −0.255 / −0.207 eV upright at 2.26 / 2.6 / 2.9 Å, reproduced within 0.1)
and the backbone's residual is the charge transfer. A frozen wall under the relaxed law (step 26 as first launched,
2026-11-05 13:06) is neither level's interaction and was withdrawn after 14 minutes, before any result. Interpolated by
the C¹ monotone cubic Hermite form of the table (in E1 for the wall; the positive knots' slopes from the ln form).
- Why, measured (theory note §4c, 2026-11-05): §23 and §25 are frame-relative and the frame wall (Round 31) covers
  Mg²⁺–Au⁰ only, so the Au⁰–O²⁻ frozen interaction was the backbone's alone, learned at the data's 2.0–2.5 Å contacts:
  the frozen step-20 energy of the upright dimer falls by −0.5 / −1.4 / −296 eV at 2.03 / 1.83 / 1.63 Å (113
  configurations: the ensemble's collapse). The law's θ = 0 shielding (attractive relative to the lone atom, with a
  force of 1.9 eV/Å toward the O at 2.26 Å) softens such a backbone's wall by 0.6–0.7 eV and tips the frozen dimer over
  the cliff in five steps. With the wall explicit, E1 + D rises monotonically toward the anion at every angle (tested
  1.8–2.9 Å, θ = 0 / 40 / 70°, both levels) and the extrapolation below the data is the physics', not the backbone's.
- Exactness: at the frozen level, wall + law at θ = 0 reproduces the measured E1 of the upright dimer within 0.06 eV at
  2.26–2.9 Å (the omitted isotropic D₀ ≤ 0.04 and the fit's residual); at 2.9 Å that interaction is attractive
  (−0.10 eV: penetration electrostatics over Pauli), which the sum reproduces.
- Tests (`test_redox_bond_pauli_wall.py`, 5): each level's table reproduced at its points, zero beyond the cutoff and
  for Mg, the untabulated pair and an unknown level refused; isotropic (the dimer's wall is its atoms'); wall + law the
  dimer's interaction at each level (frozen within 0.06 eV at four R, relaxed within 0.10 at three) and the frozen sum's
  monotonic rise 1.8–2.9 Å at three angles; forces with periodic images, the double backward and force continuity across
  a knot; the module adds exactly the wall at the law's level, refuses it without the law.

**ESTABLISHED:** exchange repulsion ∝ S² (Murrell, Randić and Williams, Proc. R. Soc. A 284, 566 (1965)); anisotropic
atom–atom repulsion from the molecular density (Stone and Price, J. Phys. Chem. 92, 3325 (1988)); ALMO energy
decomposition (Khaliullin, Cobar, Lochan, Bell and Head-Gordon, J. Phys. Chem. A 111, 8753 (2007)).
**NEW-TO-TRACE:** the frame pair's repulsion modulated by the atom's own bond direction, measured as a law of (R, θ)
and entered as a term of the master equation.

**Frozen checks (step 20 weights, both seeds; `probe_bop_frozen.py`, 2026-11-04).** With the law referenced to the
data's own angular average (what a refit backbone absorbs to leading order): the flat dimer stays in place (2.89 →
3.01 / 2.91 Å; as trained it fell to 2.20 / 2.17 Å), the tilt path's barrier +0.36 / +0.36 against PBE +0.26 and its end
−0.01 / +0.10 against −0.20, T5 at the references unchanged (< 0.5 meV: a geometric term). How a refit backbone shares
the isotropic part is not decidable on frozen weights; step 25 trains it.

**Tests** (`test_redox_bond_pauli.py`, 7): the relaxed and frozen rotations at 2.26 Å within the fits' residuals and the
frozen radial test at 2.4 / 2.9 Å within 15 %; zero without a bond, an anion or within reach; forces against finite
differences (1e-7) and a finite double backward with periodic images; the periodic sum translation-invariant with
images counted; the untabulated pair and an unknown level refused, the default off; the module adds exactly the term.

**Limitations (stated).** One partner measured (Au₂); one anion (O²⁻ of MgO(100)); Mg²⁺ measured and left out: at the
flat contact (2.89 Å above a Mg) the bond's orientation changes the Au–Mg interaction by 0.010 eV (relaxed; 0.012 frozen),
against 0.70 eV over O at 2.26 Å, and the Au–Mg non-additivity itself is ≈ +0.1 eV, isotropic (2026-11-06); the radial law beyond 2.9 Å and below 2.26 Å is an extrapolation in ln E1; the
relaxed level's cos exponent is held at 1.877 (not determined by three R).

### 33.3 Per charge state (`RedoxConfig(bond_pauli_states=True)`; step 27, built 2026-11-08) — NEW-TO-TRACE

**Why (measured).** The law of §33 was measured on the neutral dimer. Over the same O-top O, Au₂⁻ turns from upright to 70°
for +0.200 eV relaxed (+0.325 frozen) at 2.26 Å against the neutral's +0.700 (+0.929): the σ* electron restores the axial
density the σ bond removed (`tools/electrostatics/almo_angular_anion.py`, five angles; 2.6 Å at 0 / 40 / 70°: +0.077
relaxed, +0.095 frozen). Applied to every configuration, the neutral law over-penalises a tilted Au₂⁻ by up to 0.5 eV:
the doped half of T5.

**Law.** The coefficients and exponents of §33 become functions of the s atom's charge state n:

    E_bp(a, j | n_a) = λ_a s_c(R_aj) [ C₁(n_a) f(R_aj)^{p₁(n_a)} cos t + C₂(n_a) f(R_aj)^{p₂(n_a)} P₂(cos t) ],

n = −1 the anion's measured law, every other state the neutral's (the cation and higher: unmeasured, stated).
`fit_bond_pauli.py --anion` (the neutral's form, free D₀ per R, p₁ held at the neutral's): relaxed C₁⁻, C₂⁻, p₂⁻ =
−0.1094, −0.0957 eV, 0.4561 (2.26 and 2.6 Å jointly; largest residual 6.3 meV; D⁻(70) − D⁻(0) model +0.198 / +0.076 against
measured +0.2005 / +0.0769); frozen: the joint fit runs to the p₂ bound (residual 18.1 meV), so the 2.26 Å five-angle
coefficients +0.1011, −0.2905 eV with the neutral's exponents — the adoption rule was written before the fit (README,
step 27 amendment).

**Entry in the master equation.** A one-body term of the configuration energies, on the formal ladder next to the
penetration self-terms (§23): ê_u(n) += T_u(n) − T_u(0), T_u(n) = C₁(n) A₁(u; p₁(n)) + C₂(n) A₂(u; p₂(n)), with the
geometric sums A₁ = Σ_j λ s_c f^{p₁} cos t and A₂ = Σ_j λ s_c f^{p₂} P₂(cos t) (`BondPauli.state_terms`,
`geometric_sums`). The frame path keeps E_bp at n = 0 for every s atom, so a configuration's energy carries its own
atoms' laws; the split is exact up to a constant per structure (it shifts every configuration's diagonal equally and the
ensemble's free energy by that constant only). The couplings are unchanged; the wall (§33.2) stays a frame term.

**Tests** (`test_redox_bond_pauli_state.py`, 4, written red 2026-11-06 and amended before the build): the state-0 column
sums to §33's energy to round-off, non-s rows zero, the anion column equals C₁⁻A₁(p₁⁻) + C₂⁻A₂(p₂⁻) and the cation
column the neutral's; the anion column reproduces the measured anion rotation at 2.26 Å within 20 meV (relaxed) / 30 meV
(frozen) and is below half the neutral's at 70°; forces against central differences (1e-7) with periodic images and a
finite double backward; the module: off by default, refused without the law or the formal anchor, every shared
configuration without an Au⁻ keeps its diagonal and couplings, one with an Au⁻ shifts by exactly that atom's
T(−1) − T(0). `BondPauli.energy` is unchanged by the refactor (bitwise, energies and forces, 76 geometries).

## 34. The overlap damping of the frame-charge → bound-dipole coupling (`RedoxConfig(frame_induction_damping=True)`) — NEW-TO-TRACE

Module: `trace_redox/frame_damping.py` (`LAWS`, `damping_factor`, `damping_vector`, `frame_induction_damping`); the law
measured by `tools/electrostatics/deformation_dipole.py`; probes `experiments/benchmarks_phonons/probe_overlap_damping.py`,
`probe_dipole_width.py`, `probe_sigma_scan.py`; tests `tests/redox/test_redox_frame_damping.py` (3). Theory note
2026-10-18 §4e.

**Why (measured 2026-11-06/07).** On step 20's bulk MgO the Born charge is Z*(Mg) = 2.43 against PBE 1.99 at the right
ε∞ (3.01 against 3.24): the whole excess is the induced-dipole channel, the Lorentz-type enhancement of a lattice of
Gaussian-smeared dipoles driven by the frame charges' Coulomb fields. Pre-registered frozen scans ruled out every other
lever: the dipole widths (SPEC 12's `gaussian_a`, σ_μ of O and Mg), the species split of α, the ensemble (bulk MgO holds
the frame alone at kT = 0.1 eV), and the short-range pair terms (the embedded Mg²⁺–O²⁻ pair's Pauli deformation +0.06,
polarisation −0.42, charge transfer −0.01 reproduce the model's budget). The same pair measured what is missing: the O²⁻
dipole's response to its own cation is damped by overlap to s(R) = 0.61 + 0.37 (R − 2.128 Å) of the point-field value
over 1.93–2.33 Å (ALMO polarisation dipole against α(O²⁻) E, α = 2.44 Å³ of `data/ion_polarizabilities.json`). The shell
model's short-range shell–shell coupling (Dick and Overhauser 1958; Sangster et al. 1970; ESTABLISHED) in
first-principles form.

**Construction (SPEC 26's on the frame; exact in the configuration energies).** The frame charge f_i drives the bound
moments through A + ΔA with, on the dipole rows of atom j,
ΔA[i, μ_j] = (s(R_ij) − 1) K λ₃(R_ij) (r_i − r_j)_c / R_ij³ ℓ_j, λ₃ = P(3/2, R²/(σ_f,i² + w_j²)), for the tabulated
cation–anion frame pairs (periodic images summed; `penetration.pairs_within`), s(R) clamped to [0.2, 1] and C²-switched
to 1 over [2.6, 3.0] Å. With v = ΔAᵀf: y0 → y0 − K⁻¹v, u0 → u0 − A K⁻¹v, constant → constant + v·y0 − ½ v·K⁻¹v (the
quadratic form of the drive). Untouched: K (so ε∞, the bound sector's own response), A (the carriers' couplings; SPEC 26
corrects those), the bound state (`bound_state`). The term vanishes at every centrosymmetric site (v = 0 in the perfect
crystal); it acts through the derivative of the drive. Applied in `forward` after the elimination
(`RedoxElectrostatics._damp_frame_induction`), and by `tools/electrostatics/mgo_phonons.py`'s Born-charge route.

**Measured (frozen step 20 s42, 64-atom MgO, a = 4.265 Å).** Z*(Mg) 2.429 → 1.946 (the probe's A-scaled form 1.927), ε∞
3.0094 unchanged; with α rescaled to ε∞ = 3.235: 1.911; with the physical polarizability split (O²⁻ 2.44 / Mg²⁺ 0.07 Å³)
1.928; PBE 1.992. Controls: no slope 2.68, half slope 2.30, double 1.17, s₀ = 0.8 → 1.80 — the slope is the mechanism.

**Checks (tests).** The law's values and window; v against an independent image-summed pair formula on a distorted
8-atom cell and the closed-form shifts of y0, u0, constant; s ≡ 1 bit-identical; the module flag changes energy and
forces (central differences, double backward) and leaves ε∞ bit-identical; refusals (needs `anchor='formal'`).

**Limitations.** One tabulated pair; untabulated frame pairs (Al³⁺–O²⁻ of the doped cells) are left alone. The slope
carries ±0.1 Å⁻¹ from the five-point ALMO fit against the Watson-sphere α (dZ*/ds′ ≈ −3.5 e per Å⁻¹): the law at the
model's own widths is the next measurement. The species split of α (trained 0.2 / 1.9 Å³, inverted against the table)
is a separate defect; this term is robust to it.

## 35. The frame-polarizability anchor (`RedoxConfig(frame_polarizability_anchor=True)`) — NEW-TO-TRACE

Module: `trace_redox/polarizability_anchor.py` (`frame_alpha`, `anchored_params`, `AnchoredHeads`); data
`data/ion_polarizabilities.json` (SPEC 28's table); tests `tests/redox/test_redox_polarizability_anchor.py` (2).

**Why (measured 2026-11-07).** Step 20's bound sector has α_O = 0.205 and α_Mg = 1.916 Å³: the ε∞ prior fixes their
sum through the crystal's response and nothing in the slab data fixes the split, so the model put the polarizability on
the cation. The first-principles table (X2C-PBE0 finite field; O²⁻ in the Watson sphere; ESTABLISHED as the free-ion
values) has O²⁻ 16.44 bohr³ = 2.44 Å³ and Mg²⁺ 0.50 bohr³ = 0.074 Å³. The inverted split leaves ε∞ and Z* alone but
turns SPEC 34 destructive on surfaces (the damped frame-driven Mg dipoles were what held the surface O²⁻: the hole states
fell ~1.3 eV and the switched couplings collapsed a test slab, |F| 455 eV/Å); with the physical split SPEC 34 is benign
on the same slabs and gives Z*(Mg) 1.93 at ε∞ 3.235.

**Construction.** For every frame ion with a reliably tabulated α_Z(f): A_μ,i = K_E ℓ_i² / (g α_Z(f_i)) I, with one
learned crystal factor g = exp(`log_alpha_scale`) (init 1; the in-crystal reduction of the free-ion value, set by the
ε∞ prior and the forces). Untabulated species (the d block, states the table marks unreliable) keep the heads' A_μ.
Applied at the heads' readout, before the assembly builds its operator (`AnchoredHeads`), in `forward` and
`bound_state` alike, so energy_bound and the bound state stay the same functional. χ, η, σ, ℓ, κ and the seed are
untouched.

**Checks (tests).** The table values per species (O²⁻, Mg²⁺, Au⁰, Al³⁺; Fe untabulated); the anchored A_μ against
K_E ℓ²/(g α) with g = 0.8, the other fields untouched; the module flag (default False; needs `anchor='formal'`), the
parameter, both routes anchored (forward's energy_bound equals the bound state's energy), ε∞ rises with g, the energy
differentiable in g and the positions.

**35.1 The factor's scope (2026-11-12; `RedoxConfig(frame_polarizability_scope='ions')`, the default) — NEW-TO-TRACE.**
g multiplies the **charged** frame ions only: A_μ,i = K_E ℓ_i² / (g_i α_Z(f_i)), g_i = g where f_i ≠ 0 and 1 where
f_i = 0. It is the in-crystal reduction of an ion's free polarizability (ESTABLISHED as physics: Fowler and Madden,
Phys. Rev. B 29, 1035 (1984), cited from memory, not looked up); a neutral frame atom such as Au⁰ is its own free
atom.
- **Replaced statement:** "one learned crystal factor g" on every tabulated species → on the charged ones.
  `'all'` keeps the earlier statement (legacy); a checkpoint without the field loads as `'all'` with a warning
  (`LEGACY_FIELD_VALUES`), so 28b and the PBEsol phonon models reproduce their trained energies.
- **Why (measured, `benchmarks_phonons/README.md`).** With one g everywhere, the polarizability prior on the isolated
  Au atom (weight 0.3) held g at 0.997 against the ε∞ prior (weight 0.05): the two gradients on log g were
  +0.046328 and −0.046347, a sum of −0.000019, and the forces' ±0.000056. The records' "g never learns" was a
  converged balance. Under `'ions'` the Au prior's gradient on g is exactly zero and α(Au atom) is the table's
  5.4085 Å³ at every g.
- **What it predicts (frozen copies, g set by hand, not trained):** at g = 0.90 the PBEsol phonon model gives
  Z* 1.967, ε∞ 3.258 and LO 693.4 cm⁻¹ (DFT 1.979, 3.242, 693.9) and passes 7/7; trained, it had 5/7.
- Tests: `test_redox_polarizability_anchor.py` (7; 5 new). On a crystal of charged ions only (MgO) the two scopes
  agree to 1e-12.

**Limitations.** The free-ion α of O²⁻ is the Watson-sphere value; the crystal factor g carries the difference, one
number for all species. A per-species crystal factor is not identifiable from ε∞ alone (the same non-identifiability
that produced the inverted split), so it stays one.

## 36. The cluster product ensemble (`RedoxConfig(diabatic_space='clusters')`, 2026-11-10) — NEW-TO-TRACE

Module: `trace_redox/clusters.py` (`linked_configurations`, `cluster_free_energy`); wired in
`RedoxElectrostatics._cluster_solution`. Probes: `experiments/benchmarks_4g/probe_ext_product.py`,
`probe_ext_clusters.py`. Records: `experiments/benchmarks_4g/README.md`, "The extensivity line" (R13, R14, R22).
`'global'` (the default) is the ensemble of §17 and §21 as shipped.

**Why (measured, step 31 s43 frozen weights, §28.6 on).** The global ensemble builds one Hamiltonian per structure
and measures every configuration's importance (§21) from the thermal reference of the whole structure. A
configuration with k copies of a cluster excited lies k excitations above that reference, its importance is the
product of k path factors, and §21 switches its couplings off. This is the size-consistency failure of an amplitude
threshold on a global configuration space. Undoped Au₂/MgO(001), E/cell − E(1×1): wetting 2×2 +220.84 meV,
non-wetting 2×2 +99.83 meV, forces off by 0.18–0.19 eV/Å; with every coupling at full strength on the product space
of the 1×1 configurations the same rows give +2.78 and −38.02 meV (that remainder is §37's).

### 36.1 Construction

1. **Reference** n⁰: the lowest configuration found (§17.3's placement and descent, re-centred on anything lower).
2. **Linked selection.** From n⁰ every admissible move is tried. From m ≠ n⁰ a move is tried only if its donor or
   its acceptor lies in supp(m) = {u ∉ R : m_u ≠ n⁰_u}, with R the reservoir units (the band unit): a reservoir
   never links. The window (§17.3) and the amplitude rule (§21) are measured from n⁰ (x = E − E(n⁰), r(n⁰) = 1), so
   a configuration's selection does not depend on what other clusters do.
3. **Clusters.** The connected components of the graph on non-reservoir units whose edges join the units of each
   selected configuration's support (with §36.7's exception for products). S_c holds n⁰ and the configurations of
   cluster c, every other cluster at n⁰, with the exact energies E(n) of the whole system (0.1).
4. **Cluster Hamiltonian.** H_c is §21's switched Hamiltonian on S_c, its importance measured from the cluster's own
   thermal reference; F_c = −kT ln Tr exp(−H_c / kT).
5. **Pair Hamiltonian** on the valid, admissible products of S_c × S_c′: diagonal E(n⁰ + δ_s + δ_s′), exact;
   off-diagonal O_c ⊗ 1 + 1 ⊗ O_c′ (each cluster's switched couplings); F_cc′ from it.
6. **Free energy.**

       F = E⁰ + Σ_c (F_c − E⁰) + Σ_{c<c′} (F_cc′ − F_c − F_c′ + E⁰),     E⁰ = E(n⁰)                    (36.1)

   Charges and populations are the same increments of the density matrices' diagonals, so charge = ∂F/∂b.

- ESTABLISHED: the method of increments, an energy as one-body plus two-body (plus higher) increments of local
  groups: H. Stoll, Phys. Rev. B 46, 6700 (1992); review B. Paulus, Phys. Rep. 428, 1 (2006) (both checked
  2026-11-11 from the publishers' records and abstracts; the full texts were not read).
- NEW-TO-TRACE: the linked selection, clusters from the supports of the selected configurations, the reservoir
  rule, and the increments applied to the free energy of a charge-configuration Hamiltonian.

### 36.2 Exact statements, each tested (`test_redox_cluster_product.py`, 33)

- P1: one cluster reproduces the global ensemble.
- P2: clusters that do not interact are additive, for any number of copies (measured ≤ 5.33e-15 eV up to 12
  copies). The global ensemble's failure of P2 at three and four copies is pinned by a test.
- P3: two clusters are exact: F equals that of the global unswitched Hamiltonian on their product space (12 cases,
  0.00e+00 eV).
- P4: with three clusters the error is of second order in the inter-cluster interaction (measured: it falls by
  4.16–4.03 per halving with polar dimers; third order, 6.66–7.82, without).
- P5: charge = ∂F/∂b; first and second derivatives pass gradcheck and gradgradcheck.

### 36.3 Measured on the benchmark (s43 frozen weights; E/cell − E(1×1), meV; max|ΔF| against the tiled forces)

| cell | global | clusters | max\|ΔF\| clusters (eV/Å) |
|---|---|---|---|
| undoped wetting 1×1 | 0 | 0.000e+00 eV from global | 2.2e-16 |
| undoped wetting 2×1 | +5.6358 | +5.6358 | 1.21e-02 |
| undoped wetting 2×2 | +220.8418 | +2.7852 | 1.25e-02 |
| undoped non-wetting 2×1 | −22.9030 | −22.9020 | 7.86e-03 |
| undoped non-wetting 2×2 | +99.8349 | −38.0254 | 1.23e-02 |

What is left is the 1×1 cell's own finite-cell term (§37), identical in the exact unswitched product space.

### 36.7 The product closure (`RedoxConfig(cluster_closure=True)`, the default in the clusters space; 2026-11-11) — NEW-TO-TRACE

**The defect of 36.1 as first built (measured).** The space is meant to hold every product of selected excitations
on disjoint supports. Steps 2–5 hold them only across clusters: the linked rule never builds, inside one cluster,
the product of two disjoint excitations. Two model dimers whose clusters merge at a coupling τ = 0.4335 eV jumped
by +0.790 eV; the benchmark's doped 2×1 cell, where two transfers through the band unit merged the two Au dimers,
was +1.243 eV from the global ensemble.

**Two rules.**
1. *Closure.* A cluster's space is the product closure of its linked set S: every n⁰ + Σ_{m∈I} (m − n⁰), I ⊂ S
   pairwise disjoint in support (reservoir units carry the sum), inside the bounds and admissible. Diagonal: the
   exact energy. Off-diagonal: the true one-electron elements of the closed set, each multiplied by a switch read
   from the linked parts: for the element between p and p′, the largest of §21's switches of the same move between
   linked rows (m, m′) with p = m ⊕ s, s in the closed set and supp(s) disjoint from supp(m) ∪ supp(m′); 0 when
   there is none. Elements between two linked rows keep §21's switch on S. A spectator excitation therefore never
   weakens a coupling.
2. *Products do not join clusters.* A selected configuration joins the units of its support unless it is a product
   m₁ ⊕ m₂ of selected configurations with disjoint supports and T vanishes between the two supports (a transfer
   between two clusters through a reservoir); it then lives in their pair space. If the pair space of the clusters
   it spans does not hold it, or it spans more than two, it joins them after all (a fixed point: the flags only
   grow).

`cluster_closure=False` keeps 36.1 as first built (legacy).

**Measured** (`test_redox_cluster_closure.py`, 7; README "C2b scored").
- The merge of two model dimers: F = −3.908721652 eV on both sides (first build +0.789756 eV).
- One merged, unswitched cluster: 13 linked rows → 19 closed (brute enumeration 19); F − exact = +4.4e-15 eV.
- Two dimers and a shared reservoir with T = 0 between them: two clusters [7, 7] (first build one of 21), pair space
  41 rows, F − exact = 0.0e+00 eV.
- Benchmark, doped wetting 2×1 with §38's band: two clusters of 4 rows; E(clusters) = E(global) at the printed
  1e-10 eV at nine scan points, forces at 1e-10 eV/Å.
- Against the complete space of a 6-atom random-weight module (141 neutral configurations; seeds 3, 4, 5), in eV:
  global +0.370, +0.159, +0.413; first build +0.538, +0.694, +0.486; closure +0.313, −0.016, +0.049. The closure's
  error splits exactly into its rows (+0.0218, 0, +0.0149; variational by Cauchy interlacing, asserted) and §21's
  switch (+0.2916, −0.0164, +0.0341).

### 36.8 Limits (stated; L1, L3 and L4 measured)

- **L1. Three- and four-cluster increments are dropped.** Measured as F of (36.1) minus F on the full product space:
  undoped 2×2 +0.006 (wetting) and −0.004 meV per cell (non-wetting); doped wetting 2×2 with §38's band −0.1605 meV
  per 2×2 cell (−0.040 meV per 1×1 cell; 246 valid product configurations); doped non-wetting 2×2 +3.4688 meV per
  2×2 cell, **+0.867 meV per 1×1 cell** (s43, 2026-11-11). On wet_d at a (2, 2, 1) mesh that is −0.907 meV at the
  reference geometries. s42, doped wetting 2×2 (clusters of 33, 6 and 6 rows; 1182 product configurations):
  −1.9e-06 eV.
- **L1′. The energy steps where n⁰ changes on three or more clusters** (found 2026-11-11, README R22). The dropped
  terms depend on the reference. Measured on the doped wetting 2×2 cell: displacing one Au by 0.0005 Å across the
  point where the lowest configuration moves each dimer's electron from one Au to the other (all four dimers at
  once), F steps by −5.7e-07 eV and the force on that atom by +3.5e-05 eV/Å; the dropped terms are −1.6046e-04 eV
  with one reference and −1.6103e-04 eV with the other, and F on the full product space is continuous. With one or
  two clusters (36.1) telescopes to the free energy of one Hamiltonian on the product space, and no step was found
  there (doped 2×1: clusters and global agree at 1e-10 eV at nine scan points). How the step grows with the number of clusters is not
  measured. Under §37 the 1×1 cell is evaluated as its supercell, so L1 and L1′ apply to it. Removing the step
  needs the three-cluster increments or a reference that is not a single configuration; neither is built.
- **L2. Cost of the closure.** Exponential in the disjoint parts of one cluster: 35 linked rows → 5184 closed on a
  12-atom strongly coupled charged cell. Beyond `fragment_max_configurations` (512) the structure is refused, never
  truncated. On the benchmark cells of s43 the clusters hold 2–5 rows. On s42's doped 2×2 supercell they do not
  (2026-11-11, README R23): the wetting cell has clusters of 33, 6 and 6 rows, and in the non-wetting cell 127
  linked rows close to 1218, which the cap refuses (evaluated at a cap of 2048 for the record only). A closure that
  is polynomial inside a merged cluster is not built.
- **L3. The band unit** was one two-electron orbital per structure when this section was built: §38.
- **L4. The finite-cell term** of a cluster in phase with its own images: §37.
- **L5.** Refused with §32 (`downfold_transfer`) and with §30's response priors on more than one cluster.
  **Replaced 2026-11-12:** the response record of several clusters is built (§36.10); the refusal with §32 stays.
- **§21's switch is not variational**, so neither is F: on seed 4 above the closed set is the complete space and the
  switched lowest eigenvalue lies 16.4 meV below the bare one. Not measured on the benchmark beyond the agreement
  of the two spaces.
- **Reservoir exchange.** The band unit is shared by the clusters; the exchange sign of two electrons placed in it
  from different clusters is dropped.

### 36.9 No percolation enumeration in the diabatic ensemble (2026-11-10; both spaces)

§12.6's 2^k bond-percolation patterns belong to the two-level and exact fragment ensembles. The diabatic ensemble
carries integer charges on every atom and never reads them (measured: dE = 0 exactly with the partitions emptied),
so with `fragment_ensemble='diabatic'` the module no longer enumerates them; the bond graph's coverage check stays.
ESTABLISHED: nothing new; a refusal without physical content is removed. Tests: `test_redox_percolation_diabatic.py`
(5). This removed one of the two refusals of the benchmark's doped 2×2 cell (12 > 10 uncertain connections); the
other was L3. The four 1×1 benchmark energies are unchanged at the 9 printed decimals.

### 36.10 The response record of several clusters (2026-11-12; lifts L5's second half) — NEW-TO-TRACE

Why: §30's response priors (ε∞, α) need the ensemble's exact field response whenever a structure has more than one
configuration, and §36 as first built returned no record on several clusters. A training in the clusters space would
stop the first time a prior's structure reached two clusters. Measured on frozen step 31 weights (README R26): s43's
two prior structures have one configuration in every setting; s42's bulk MgO prior cell has four clusters of two.

(0.3′) is linear in the free energies of its terms, and in a uniform field every configuration's energy is
E(n; F) = E(n; 0) − F·M(n) − ½ Fᵀ[α_b + δα(n)]F (§30, §28). So

    α = α_b + Σ_X w_X α_ens(H_X, M_X, δα_X),    w_0 = 1 − n_c + n_p,   w_c = 2 − n_c,   w_cc′ = 1,          (36.10)

X running over the reference configuration (a 1×1 Hamiltonian), the n_c clusters and the n_p pair spaces, α_ens of
§30 on each. Σ_X w_X = 1, so α_b enters once. Every row of every term gets its own exact M(n) and δα(n) from the
same routines as §30's record (no additivity of dipoles is assumed).

- `RedoxResult.ensemble` on several clusters: `'H'` is None and `'terms'` holds
  `{'weight', 'kind', 'H', 'rows'}` per term, `'rows'` indexing `'M'` and `'alpha_shift'`. One cluster keeps §30's
  record unchanged. Read both with `trace_redox.response.record_polarizability`; `train.py`'s priors do.
- At a mesh (§37) the record is the supercell's: `'alpha_bound'` is divided by the number of copies and
  `'scale' = 1 / copies` multiplies the ensemble part, so the prior sees the polarisability per cell.
- `ClusterEnsemble` keeps each pair space's Hamiltonian and configurations (`pair_index`, `pair_hamiltonians`,
  `pair_configs`); they were built already.

**Tested** (`test_redox_cluster_response.py`, 8; `test_redox_polarizability_prior.py`, 1 new):
- α from the record equals −∂²E/∂F² of the module's total energy in an applied field (open boundaries, Richardson
  central differences, all three diagonal components, 1e-5 relative) on two dimers that form two clusters: Na₂ in the
  window, Na₂ and Au₂ with one-electron couplings, with the amplitude selection saturated, and with §28. Leaving out
  the pair terms misses by more than that tolerance in every case.
- One cluster returns §30's record; the record carries the graph; at a mesh the record is the supercell's per cell
  (to the round-off of the finite-difference response, 16 ε |H| / (h/2)²).
- A polarizability prior on two neutral Na₂ trains in the clusters space (it raised before).

**Limits.** (i) Not tested under periodic boundaries against a finite field: the module has no uniform field there,
and a multi-move configuration's dipole takes §30's minimal-image convention around its first changed atom. On an
8-atom cell, where neighbours sit at exactly half a cell vector, that convention is ambiguous (it was already so
for §30's record in the global space). (ii) Inside §21's switch band the switch's field derivative is not in the
record (as in §30). (iii) Still refused with §32. (iv) The pair spaces' Hamiltonians are held until the next
evaluation.


## 37. Born–von Kármán sampling of the configuration fluctuations (`RedoxConfig(bvk_mesh=…)` or `bvk_length=…`, 2026-11-10) — NEW-TO-TRACE

Module: `trace_redox/bvk.py` (`mesh_for`, `tile`); `RedoxElectrostatics.forward` → `_forward_cell` / `_forward_bvk`;
`RedoxResult.bvk_mesh`. Probes: `experiments/benchmarks_4g/probe_ext_image_terms.py`, `probe_ext_bvk_table2.py`,
`probe_ext_bvk_module.py`. Records: README R15, R16, "SPEC 37 scored". The default (1, 1, 1) with `bvk_length = 0`
is the model as shipped, bit for bit.

**Why (measured, s43 frozen weights, §28.6 and §36 on).** With one cell at Γ a cluster's charge fluctuation is in
phase with its own periodic images, so the energy per cell depends on the cell chosen for the same crystal. It is a
finite-size error of the 1×1 calculation, not a size-consistency failure: it is identical in the exact unswitched
product space, so no selection or switch removes it. Undoped Au₂/MgO(001), E/cell − E(1×1): wetting +5.64 (2×1),
+2.79 (2×2) meV; non-wetting −22.90, −38.03 meV; max|ΔF| 1.2e-02 eV/Å. The reference configuration's energy E⁰
replicates exactly; the residual is the clusters' excitation energies (2×2: +1.48 and −39.29 meV) plus the pair
increments (+1.31, +1.26 meV).

**Statement.** For a cell C and a mesh M = (M₁, M₂, M₃), M_d = 1 along a non-periodic direction,

    E_M(C) := E_Γ(S_M[C]) / |M|,        |M| = M₁ M₂ M₃                                              (37.1)

- S_M[C] is the M-fold supercell (positions pos + R·cell, features and species repeated, total charge × |M|), and
  E_Γ is the model with §36's cluster ensemble. In S_M the |M| copies of a cluster fluctuate independently: an
  excitation meets its own images at M·a instead of a, and the other copies' clusters through §36's increments.
- `bvk_length = L` chooses the mesh per structure, M_d = ceil(L / w_d), with w_d the spacing of the lattice planes
  along d. Cells of one crystal with equal M_d·w_d then have the same energy per cell.
- The supercell is tiled from the cell's own graph (edge vectors repeated, positions and lattice built from the
  caller's), so forces and stress per cell follow from the same backward pass as at Γ. The record's per-atom fields
  are the home cell's.
- NEW-TO-TRACE: the integer-charge configuration fluctuations of a thermal diabatic ensemble sampled on a
  Born–von Kármán cell that is decoupled from the simulation cell, with E⁰ exact at every M.
- ESTABLISHED: cyclic boundary conditions over a supercell (M. Born and Th. von Kármán, "Über Schwingungen in
  Raumgittern", Phys. Z. 13, 297 (1912); bibliographic record checked 2026-11-11 from secondary sources, the
  article not read) and their equivalence to a uniform mesh of Bloch vectors (H. J. Monkhorst and J. D. Pack,
  Phys. Rev. B 13, 5188 (1976); record and abstract checked 2026-11-11, the paper not read).

**Exact statements, each tested** (`test_redox_bvk.py`, 29).
- B1: the module at a mesh M on C equals the module at (1, 1, 1) on the supercell built independently (its own
  neighbour list), divided by |M| (bulk and slab; measured ≤ 1.4e-14 eV).
- B2: commensurate replication, E_M(S_P[C]) = |P| · E_{M∘P}(C) (≤ 1.6e-13 eV).
- B3: extensivity at a fixed length L: E_L(S[C]) = 2 E_L(C).
- B4: forces −∂E_M/∂pos through the tiling equal central finite differences.
- B5: (1, 1, 1) with L = 0 returns the present energy bit for bit.
- B6: validation: positive integers; a mesh > 1 along a non-periodic direction is refused; the option needs
  `fragment_ensemble='diabatic'` and `diabatic_space='clusters'`; `bvk_mesh` and `bvk_length` exclude each other; a
  checkpoint without the fields loads as (1, 1, 1), L = 0.
- B7: the record's per-atom fields are the home cell's, and Σq is the cell's total charge.
- The charged 6-atom case runs with `cluster_closure=False` (its closure holds 5184 rows and is refused at the cap,
  §36.8 L2; a test asserts that refusal); at a cap of 8192 the identity holds with the closure too.

**Measured on the benchmark** (frozen step 31 weights trained at Γ with the block form: sizes, not scores).

| | unit mesh | `bvk_mesh = (2, 2, 1)` / `bvk_length = 18.0` Å |
|---|---|---|
| wet_u, s43 (DFT +928.9, 4G +975.4), meV | +892.68 | +935.11 |
| wet_u, s42, meV | +832.37 | +890.85 |
| undoped, E(2×2)/4 − E(1×1), wetting / non-wetting, meV | +2.785 / −38.025 | +1e-6 / < 1e-6 |
| undoped, max\|ΔF\|, eV/Å | 1.25e-2 / 1.23e-2 | 5.6e-9 / 3.4e-8 |

**Limits (stated).**
- Convergence in M is not shown: a 2×2 cell keeps an estimated 5–12 % of the 1×1 image term (R15), and a larger
  mesh is refused by the dense route.
  **Measured since (2026-11-11, README R24, M1, M1b, M1c; undoped, fixed reference geometries, frozen step 31 weights).**
  - The reference DFT sampled 3 × 3 × 1 k-points (Ko et al., Nat. Commun. 12, 398 (2021), Methods), so (3, 3, 1) is
    the mesh that corresponds to the labels. It is refused by the dense route (990 atoms, a 35793² operator,
    10.25 GB against the 4 GB limit).
    **Replaced 2026-11-12 (README R25, M1d):** that guard bounds a dense K which `bound_elimination='condensed'`
    never builds. With `max_operator_bytes` raised (3, 3, 1) runs: 127–131 s per evaluation, and
    E(wetting) − E(non-wetting) = +950.325 / +906.305 meV, a step of +2.388 / +3.364 meV from (2,2,1) against the
    fit's +2.230 / +3.329 written before the run. (2,2,1) is converged to 5 meV for the undoped pair.
  - E(wetting) − E(non-wetting), meV, s43 / s42: unit mesh +907.125 / +845.788; (2,2,1) +947.937 / +902.941;
    along x, (2,1,1) … (4,1,1): +935.664, +938.913, +939.696 / +888.448, +893.359, +894.665.
  - The image term is the tin-foil lattice sum of the excitation's dipole on the cell's lattice (R15 part 3), the
    vacuum period included: an upright (z) dipole's term falls as e^{−2π a M / L_z} along a line of cells, an
    in-plane dipole's term carries the Lorentz term (−4π/V_M in Λ along a line, −2π/V_M on a square mesh). M1c
    changed the vacuum period from 26.4589 to 40.0 Å and the upright dimer's decay ratios moved as that sum says
    (0.077, 0.099 → 0.129, 0.214 on s43; point dipole 0.124, 0.206); the sizes did not (partial).
  - Of the 1×1 image term a square mesh keeps 4.9 %, 0.4 %, 0.04 % (z dipole) and 14.3 %, 5.4 %, 2.9 % (in-plane
    dipole) at 2×2, 3×3, 4×4. The "5–12 %" above is replaced by these.
  - **Estimates, not measured** (one- and two-constant fits of that sum): (2,2,1) → (3,3,1) +2.2 / +3.3 meV;
    (2,2,1) → the lateral limit +2.7 / +4.2 meV; a lateral mesh keeps the chain of images through the vacuum,
    +1.6 / +2.2 meV.
  - A mean cell dipole M_z carries −2π K_E M_z²/V under tin-foil boundaries (+3.0 / +7.3 meV change for
    26.4589 → 40 Å on the upright cell). That is a term of the mean state, the same at every mesh, and no part
    of this section's defect.
- The mesh from `bvk_length` jumps when a width crosses L / integer, as a k-point spacing does; derivatives are
  taken at a fixed mesh.
- Response properties (`bound_state`, §30's record) stay those of the Γ cell.
- Cost: the supercell is evaluated whole (2×2 at 110 atoms: about 8.5× one cell).
  Measured (README S1, S2; M2 Pro CPU, 4 threads, energy and forces): one cell 1.53 s, (2,1,1) 3.34 s, (2,2,1)
  13.27 s with 5.4 GB peak memory; the electrostatics is 77 %, 89 %, 97 % of the whole. A tight-binding code
  (DFTB+ 25.1, GFN1-xTB) takes 11.7 s on the single cell and 34.7 s on 3 × 3 × 1 k-points: it pays 3.0× for the
  grid where this route pays 8.7× for (2,2,1). A route that uses the supercell's translation symmetry is not built.
- Wording: TRACE has no Bloch states. "Γ" means the single cell under periodic boundaries and a "mesh" the
  supercell of (0.4); the names are borrowed from band theory, where the two are equivalent.
- With §20's band (`band_form='phi1'`) a structure whose band unit holds more than 0.5 e is refused: that band takes
  two electrons per structure, so the supercell is not a replication of the cell. §38's band lifts the refusal.
- §36.8's L1 and L1′ apply to every cell evaluated at a mesh with three or more copies.
- No model has been trained at a mesh.


## 38. The Fermi band: a band unit that holds any number of electrons (`RedoxConfig(band_form='fermi')`, 2026-11-10 / 11) — NEW-TO-TRACE

Modules: `trace_redox/fermi.py` (`fermi_operator`, `band_free_energy`), `trace_redox/band.py`
(`augment_fermi`), the wiring in `trace_redox/module.py`. Probes: `experiments/benchmarks_4g/probe_ext_band_spectrum.py`,
`probe_ext_band_fermi.py`, `probe_ext_fermi_module.py`, `probe_ext_fermi_unit.py`, `probe_ext_fermi_table2.py`,
`probe_ext_fermi_split.py`, `probe_ext_fermi_filling.py`.
Records: README R17, R18, "C4" (pre-registration, stage 1 scored, the stage 2 mechanism, stage 2 scored), R19–R23.
The default stays §20's band, `band_form='phi1'`; a checkpoint without the field loads as 'phi1'.

**Why (measured, s43 frozen weights).** §20's band unit holds at most two electrons, both in the lowest orbital
φ₁, in one unit per structure. That is not a function of the crystal:
- The doped Au₂/MgO cell carries three excess electrons per 1×1 cell. Its 2×1 supercell carries six; the band takes
  two, every Au is forced to −1, and E/cell − E(1×1) = +5279.611 (wetting), +7047.327 meV (non-wetting). The 2×2
  cell (twelve) is refused: `total charge 0 is outside the capacity [2, 1320]`.
- The band is metallic-like under replication: J₁₁ halves with each doubling (0.8648 → 0.4348 eV), φ₁ spreads over
  the cell, and the folded levels come in near-degenerate groups (gaps of 0.3–5 meV). Neither one unit per orbital
  nor a zero-temperature closed shell is a usable form there.
- §37 therefore had to refuse every structure whose band holds more than 0.5 e, so wet_d and T5 existed at one cell
  size only.

### 38.1 The functional

For a band of B sites with the one-carrier Hamiltonian h = diag(ε) − τ, the interaction V (U_u = V_uu) of §20 and
the model's kT, the free energy at N electrons, 0 ≤ N ≤ 2B, is

    F_b(N) = A(N↑) + A(N↓) + ½ Σ_{u≠v} V_uv [ρ_u ρ_v − Σ_σ (P^σ_uv)²] + Σ_u U_u ρ↑_u ρ↓_u,            (38.1)
    A(n)   = μ n − kT tr ln(1 + e^{−(h − μ)/kT}),    P^σ = [1 + e^{(h − μ_σ)/kT}]⁻¹,    tr P^σ = N_σ,
    N↑ = ⌈N/2⌉,  N↓ = ⌊N/2⌋,    ρ^σ = diag P^σ,    ρ = ρ↑ + ρ↓.

The band's charge on site v at filling N is −ρ_N(v).
- NEW-TO-TRACE: an N-electron Fermi reservoir as one unit of the integer-charge configuration ensemble, each
  filling a state of the unit with its own charge shape ρ_N; the fixed-filling projectors P^σ as a polynomial in
  h, with μ(h) carried by two Newton steps on the same moments (derivatives exact to third order at any degeneracy,
  where differentiating an eigendecomposition is singular).
- ESTABLISHED: Fermi–Dirac occupations as the minimiser of the finite-temperature functional (N. D. Mermin,
  Phys. Rev. 137, A1441 (1965)); the first-order energy of a Hubbard / Pariser–Parr–Pople interaction in a frozen
  determinant (textbook; no single citation claimed); the Chebyshev expansion of the Fermi operator (S. Goedecker
  and L. Colombo, Phys. Rev. Lett. 73, 122 (1994)). For both papers the bibliographic record was checked; their
  equations were not read here.
- Reduction to §20. With a gap ≫ kT: F_b(0) = 0, F_b(1) = λ₁, F_b(2) = 2λ₁ + J₁₁ and ρ_m = m φ₁². At a finite gap
  the difference is of order exp(−gap/2kT) (grand-canonical occupations at a fixed mean filling): 3e-7 to 7e-7 eV
  on the doped 1×1 cells (gap 2.65 eV), 1–2 meV above a 1.15 eV gap. The form is therefore an option beside §20,
  not a silent replacement.

**Two constructions the derivatives need (NEW-TO-TRACE; found by the falsifying checks).**
1. A(n) takes its value from the detached eigenvalues and its gradient analytically, dA/dh = P
   (`_ValueWithGradient`), so no derivative of A's own series is taken; second derivatives flow through P's graph.
   Differentiating A's series twice had cost 6e-8 to 1e-7 at order ~1300, because the p-th derivative of T_k grows
   as k^{2p} at the ends of the interval and the coefficients carry round-off at every k.
2. The spectral interval is widened by 5 % (`SPECTRUM_MARGIN`), which keeps the spectrum away from those ends.
`torch.logaddexp` replaces `softplus`, whose linear branch above 20 drops e^{−20}.

**Exact statements, each tested** (`test_redox_fermi_band.py`, 21).
- F1: tr P = N_σ (measured 2.0e-13); P against the eigendecomposition (3.1e-14).
- F2: the 12 eV gap limits above (≤ 4.1e-14 eV; max|ρ₁ − φ₁²| 2.1e-14).
- F3: the full band, F_b(2B) = 2 tr h + 2 Σ_{u≠v} V_uv + Σ_u U_u (7.1e-15 on 61.79 eV).
- F4, F5: first and second derivatives with respect to h and V against central differences at an exactly
  degenerate spectrum with μ inside the degenerate shell (relative 1e-7 asserted).
- F6: two uncoupled copies at twice an even filling, F_b[h⊕h, V⊕V](2N) = 2 F_b[h, V](N) (≤ 1.4e-14 eV).
- F7: an infinite U_u is accepted iff N↓ = 0.
- F8: the saved s43 band arrays against an independent eigendecomposition (asserted 1e-9 eV; measured ≤ 3.6e-12 eV
  against `probe_ext_band_fermi.py`'s own function, 12 cells, N = 0..4 per copy).
- Mutations caught: μ held constant (F4); one Newton step instead of two (F5).

### 38.2 The unit

- **States (NEW-TO-TRACE: a searched reference filling as the origin of the unit's state label).** The band unit's
  state s on the common state axis means the filling N = N₀ − s; its charge is −N. Bounds:
  K_b = min(last state, N₀), L_b = max(first state, N₀ − N_max), with N_max = 2B when every U_u is finite and 1
  otherwise (F7). On the benchmark cells (state axis −2 … +3) the unit spans N₀ − 3 … N₀ + 2.
- **Energy of a configuration** (n the atoms' integer charges; band sites at or above their frame, as in §20):

      E(n, N) = E_at(n) + F_b(N) + Σ_v ρ_N(v) c_v(n),                                               (38.2)
      c_v(n)  = − Σ_u W_vu (n_u − f_u) + Σ_{u ∉ sites} δ_uv(n_u),

  with E_at the atoms' configuration energy (ladder, b, W, pair tables), F_b and ρ_N from (38.1) on §20's ε, V, τ,
  and δ_uv §20's pair-table change. With ρ_m = m φ₁² and F_b(m) = E_m this is §20's energy term by term.
  NEW-TO-TRACE: the filling-dependent shape ρ_N coupled to the atoms' exact configuration energies. ESTABLISHED:
  a charge density in the potential of the other charges, to first order (textbook; no citation claimed).
- **Assembly.** The band unit's entry of b and its row and column of W are zero. Its ladder row holds F_b(N(s))
  (+∞ outside [L_b, K_b]). One pair table per atom u with the band unit,
  Ξ_(u,band)[k, s] = −(k − f_u) Σ_v W_uv ρ_{N(s)}(v) + Σ_v δ_uv(k) ρ_{N(s)}(v). The search, §21's selection, §36's
  increments and the solver are unchanged: they already take a ladder and pair tables.
- **Reference filling.** Start at N₀ = clip(Σ_u f_u − Q, 0, N_max). Detached passes: assemble at N₀, find the
  lowest configuration with the solver's own search, read the band's state s_ref there, set N₀ ← N₀ − s_ref; stop
  at s_ref = 0. The reference energy cannot rise from one pass to the next, so the passes end; a revisited N₀ (an
  exact tie) ends them at the smaller N₀; more than 64 passes is refused. E(n, N) does not contain N₀, so the
  result is the same for every N₀ whose window holds the selected configurations (T2).
- **Charges.** A configuration's atom charges are n_v − ρ_{N(s)}(v) on band sites and n_u elsewhere; they enter
  §28's correction, §30's response and the free dipoles. The reported charges are the ensemble's expectation of
  the same expression, the band unit's reported charge is −⟨N⟩, and `RedoxResult.omega` is None.
- **Couplings.** The band unit's row and column of T are zero: it exchanges electrons with the other units
  thermally, without a coupling amplitude. Measured before the choice: with that row and column set to zero in the
  legacy form, ΔE = 0 exactly on six of the eight benchmark evaluations (two seeds, four cells) and +0.00097 /
  +0.00215 meV on the other two. Reasons of principle: a thermal band with a degenerate frontier has no single
  frontier orbital, and one amplitude per unit to a state normalised over M cells is of order 1/√M, which is not
  extensive.
- **Born–von Kármán sampling.** §37's refusal of a filled band is lifted for this form.

**Exact statements, each tested** (`test_redox_fermi_band_unit.py`, 19).
- T1: the assembled ladder and pair tables give E(n, N) equal to a reference built from the atoms'
  `configuration_energies` alone (ε_v, V_vw, U_v and c_v(n) as first and second differences of E_at), to round-off.
- T2: the same energies for two different N₀ on the configurations both windows hold.
- T3: a gapped band at N₀ = 0 reproduces the legacy unit's configuration energies (≤ 1e-9 eV at a gap ≥ 10 eV).
- T4: the atom charges of every configuration sum to Q and are n_v − ρ_N(v).
- T5: module level, a doped rock-salt cluster with three dopants: more than two band electrons; free charges sum
  to Q; s_ref = 0; forces equal −∇E by central differences; a double backward runs and is finite.
- T6: `_forward_bvk` evaluates a doped periodic cell for 'fermi' and still refuses it for 'phi1'.

### 38.3 Measured on the benchmark (frozen step 31 weights, trained at Γ with §28's block form and §20's band: sizes, not scores)

`state_polarizability_stiffness='onsite'` (§28.6), `diabatic_space='clusters'` with the closure (§36.7). README
"C4 stage 2 scored", R23.

| pre-registered | measured (s43) | |
|---|---|---|
| U1: doped 1×1 at Γ, 'fermi' against 'phi1', \|ΔE\| ≤ 1e-4 eV | +1.717e-06 (wetting), +1.641e-06 eV (non-wetting) | PASS |
| U2: the doped 2×1 and 2×2 cells evaluate; forces equal −∇E by central differences to 1e-6 relative on three components | all four cells evaluate; 1 of 12 components within 1e-6; three at 1e-2 | FAIL as written |
| U3: `bvk_length = 18.0` Å, E(2×2)/4 − E(1×1) ≤ 1e-3 meV and max\|ΔF\| ≤ 1e-6 eV/Å on both doped geometries | +3e-06 meV, 6.5e-08 eV/Å; +7e-06 meV, 6.2e-08 eV/Å (unit mesh: +1829.0995 and +1791.5162 meV, 1.0e-01 eV/Å) | PASS |
| U4: a double backward runs and matches central differences (no threshold) | relative 4.1e-07 (1×1) and 1.3e-06 (2×1) at the best step | runs |
| U5: both suites | `tests/redox` 944 passed; the rest 1045 passed, 4 skipped | PASS |

U2's failures are not in this section's unit: the 2.26 Å knot of §33's C¹ wall table inside the stencil (two
cells), §36.8's L1′ step (one cell), and the stencil's own resolution of about 1e-06 eV/Å on the other nine
components. The force is the exact derivative of the energy on each side of the knot and of the step.

| Table II, meV | wet_u | wet_d | T5 |
|---|---|---|---|
| DFT | +928.9 | −2.7 | −931.6 |
| 4G-HDNNP | +975.4 | −41.0 | −1016.4 |
| s43, 'phi1', Γ | +892.68 | −60.70 | −953.38 |
| s43, 'fermi', Γ | +892.6511 | −60.6996 | −953.3506 |
| s43, 'fermi', `bvk_mesh = (2, 2, 1)` | +935.1123 | −22.9474 | −958.0597 |
| s42, 'phi1', Γ | +832.37 | −164.68 | −997.04 |
| s42, 'fermi', Γ | +832.1547 | −164.6794 | −996.8341 |
| s42, 'fermi', `bvk_mesh = (2, 2, 1)` | +890.8461 | refused at the cap 512; +603.6879 at 2048 | −287.1582 at 2048 |

- **The doped cells now evaluate at any replication and at a mesh**, which §20's band could not do (the 2×2 cell
  was refused, the 2×1 cell wrong by 5.3 and 7.0 eV per cell).
- **The s43 shift of wet_d from Γ to the mesh, +37.58 meV at fixed geometry**, by parts (meV per cell): the band's
  own F_b −0.85, the rest of the reference configuration −2.62, §36's one-cluster increments +87.00, its pair
  increments −45.95. Each geometry's energy rises by 1.83 and 1.79 eV per cell; F_b at two electrons per cell
  carries +1855.2 and +1856.1 meV of it: the cell's two band electrons sit at the band bottom, the supercell's
  eight fill the folded levels 1.07 to 1.26 eV above it.
- **The second seed.** In the 2×2 supercell s42's band takes electrons from the dimers:

  | cell | band, e per cell | Au₂ per cell | clusters (rows) |
  |---|---|---|---|
  | s43, both geometries, 1×1 and 2×2 | 2.0000 | −1.0000 | [2] or [3]; [4, 4, 4, 4] |
  | s42, both geometries, 1×1 | 2.0000 | −1.0000 | [3]; [4] |
  | s42 wetting 2×2 | 2.5003 | −0.4997 (two dimers −1, two neutral) | [33, 6, 6] |
  | s42 non-wetting 2×2 | 2.8429 | −0.1571 | 127 linked → [1218] closed |

  wet_d at fixed geometry then shifts by +756.28 meV (F_b +553.10, the reference −120.33, one-cluster increments
  +323.53, pairs −0.03). The two seeds are 104 meV apart on wet_d at Γ and 627 meV apart at the mesh.

### 38.4 Limits (stated; the first three measured)

- **The filling of a supercell's band is not fixed by weights trained at Γ on one cell.** Such training sees two
  electrons in the band's lowest level; the band's width, its interaction above that filling and the position of
  a localised level against the folded band levels carry no label. Measured: the two seeds above. Measured on the
  s43 arrays: the band sites' second-electron cost U_v is negative on every band site (−4.2 to −2.4 eV), and the
  bare band's F_b in the 2×2 cell is lowest at 2.5 electrons per cell. A doped number at a mesh is therefore not
  claimed for these weights; a model trained at the mesh is the test and is not run.
- **The closure's cap is reached on a benchmark cell** (§36.8 L2): s42's doped non-wetting supercell is refused at
  `fragment_max_configurations = 512`.
- **Convergence in the mesh is not shown.** F_b per cell moved by +0.60 eV in the last doubling of the cell (R18);
  a 3×3 mesh is refused by the dense route.
  For the undoped cells the sequence is measured to four cells and understood term by term (§37, limits); for
  the doped cells, whose band fills, nothing beyond (2, 2, 1) is measured.
- (i) The orbitals are those of h, not self-consistent in the band's own interaction (as in §20). (ii) The lowest
  |S_z| is taken: an odd filling per cell is not the ferromagnetic state. (iii) The exchange sign of electrons
  placed in the band from different clusters is dropped (§36.8). (iv) Convergence in the mesh, as above. (v) No
  benchmark number until a model is trained with the form. **[Updated 2026-11-16:** steps 33 and 34b trained with
  `band_form='fermi'` at Γ on one cell; their scores are in the benchmarks README; none was trained at a mesh.**]**
- (vi) The hybridisation of a localised level with the band: the band unit's couplings are zero (38.2).
  (vii) Fillings outside N₀ − last … N₀ − first: a pair product that needs one is left out of its pair space.
  (viii) Continuity when the selection changes N₀: exact only under T2's condition. (ix) On the doped 1×1 cell
  'fermi' admits N = 3 and 4 and §20's band does not; U1 bounds their effect on the energy by 1.7e-06 eV.
- The reference configuration of a supercell can break its translation symmetry (s42 wetting 2×2: two of four
  equivalent dimers charged). The energy is still that of the supercell divided by its copies; the forces on the
  cell are then, by the chain rule, the mean over inequivalent copies. Not examined further.

## 39. The frame pair's own interaction, for every element pair (`RedoxConfig(frame_pair=True)`, 2026-11-13) — NEW-TO-TRACE

**Status: built, reviewed and tested; off by default; no model is trained with it** (2026-11-13). **[Updated
2026-11-16:** step 34b trained with it on both seeds (cold start, with §29 and §40); it FAILS as pre-registered on S2,
S5 and S8 and passes S1, S3, S4 and S7 (benchmarks README, "Step 34b scored").**]** The project's rule of 2026-11-13:
an update must be a general equation for all systems. `trace_redox/frame_pair.py`, `trace_redox/data/frame_pair.npz`
(`tools/electrostatics/build_frame_pair.py`), `tests/redox/test_redox_frame_pair.py` (45, and one strict xfail of an
inherited defect, §39.6 ix). Reviewed the same day by four independent readers and six verifiers
(`experiments/benchmarks_4g/runs/general_laws_2026-11-13/review_spec39.json`); five defects confirmed and corrected,
recorded in CHANGELOG Round 41.

**Update, 2026-11-13 evening (CHANGELOG Round 42).** Ten pairs now carry first-principles points; "the six" below
is the morning's state. Added, each computed after the table and compared with it out of sample before entering:
Na⁺–Cl⁻ (six points, 2.0–3.3 Å, ratio 1.017–1.107; c = 1 was 2–10 % low), Cl⁻–Cl⁻ (four, 2.6–3.8 Å, 0.527–1.097;
c = 1 was 31–90 % high in the tail), Ag⁰–Ag⁰ (six, 2.3–3.0 Å, 0.898–0.991; c = 1 left the whole interaction 37–39 %
high at 2.3–2.53 Å, falling to 4 % at 3.0 Å), H⁰–H⁰ (seven, 0.6–2.2 Å, 3.53–5.62; c = 1 was a factor 3.5–5.6 low:
at 0.74 Å the overlap terms are 2.596 against 10.009 eV, the totals 2.618 against 10.032). The electrostatic half agreed with first
principles at all 25 new points (H–H to 5e-6 eV, the others to 4e-4 eV). A point enters only where the electron-gas
term itself is above 5e-3 eV (§39.6 iii). The series and the six earlier pairs are bit-identical. The test file
holds 46 tests and no xfail (§39.6 ix is fixed). The other half of this term for two s¹ frames is §40.

**39.1 What was missing (measured).** §23 and §25 hold every carrier state *relative to the frame pair* and vanish
at it; (0.1) has no term for the frame pair itself. The backbone carried it where there are data. Below them the
charge module's only Au–Au repulsion was the learned coupling shrinking under compression, which crosses zero at
2.06 Å (step 31) and 2.13 Å (the single model); the dimer then collapses (bond force −12 … −23 eV/Å against DFT's
+15.2 at 2.089 Å; `experiments/benchmarks_4g/README.md`, 2026-11-12 / 13). The same disease is on record in the other
systems: τ(Cl–Cl, 1.99 Å) 24.9 eV against Harrison's 6.2, and Ag₂ as an ionic pair in the Ag₃ model.

**39.2 The term.** For every pair of atoms within 8 Å (periodic images included; an atom's own images once per ±T),

    Φ_ab(R) = s(R) { D_ff(R) + f_a f_b [K/R − J^G_ab(R)] + c_ab(R) P_ab(R) },                                 (39.1)

    D_ff = E_es(f_a, f_b) − f_a f_b K/R      the exact electrostatics of the two frozen frame densities beyond
                                             point charges (ESTABLISHED: Gordon and Kim, J. Chem. Phys. 56, 3122 (1972))
    J^G  = K erf(R/√(σ_a² + σ_b²))/R         the model's own Gaussian kernel of the frame charges (§23's; checked in
                                             the code for open and periodic systems: the same kernel, the same widths)
    P    = T_TF^nad + E_x^nad                the non-additive Thomas–Fermi kinetic and Dirac exchange energy of the
                                             frame densities: the electron-gas Pauli repulsion (ESTABLISHED, ibid.)
    c_ab(R) = 1, or the first-principles ratio E_fp(R_k)/P(R_k) of the pair's frozen-density PBE interaction at the
              computed distances, a monotone cubic between them; beyond the first and the last it keeps its end
              slope and saturates within w = 0.3 Å, c_end + m_end w tanh((R − R_end)/w)
              (NEW-TO-TRACE as a representation; the points are first principles)

s is §23's C² switch. E = (0.1) + Σ_pairs Φ: the term is the same in every configuration, moves no population and
enters no response record. **With §23 the electrostatics of any state pair becomes E_es(n_a, n_b) − n_a n_b J^G**
(Xi(n) + Xi_ff, identically): §23 made absolute; §25's carrier-minus-frame Pauli gets its frame. NEW-TO-TRACE: the
frame pair's whole frozen interaction as the baseline of the configuration ensemble, so that a bond is this repulsion
plus the ensemble's hopping energy (ESTABLISHED as a decomposition: the tight-binding total energy as band energy
plus a pair repulsion, Chadi, Phys. Rev. Lett. 41, 1062 (1978); its derivation from density-functional theory,
Foulkes and Haydock, Phys. Rev. B 39, 12520 (1989); both cited from memory).

**39.3 Why this form (measured on the six frame pairs with first-principles points, 2026-11-13).**
- The one-pair wall of Round 31 (`frame_wall`, K_TF T_TF) is not a law: K_TF is 0.618 (Au⁰–Au⁰), 0.55–0.59
  (Au⁰–O²⁻), 0.399 (O²⁻–O²⁻), 0.885 (O²⁻–Mg²⁺), 0.918 (Mg²⁺–Au⁰), 1.017 (O²⁻–Al³⁺), and falls with R inside each pair.
- Within a pair, first principles over P varies by 1.00–1.07× in R (1.22× for Au⁰–O²⁻ over 2.0–4.0 Å); over the
  density overlap by 1.03–1.25× (2.19×); over Thomas–Fermi alone by 1.07–1.53× (5.71×; 1.53× is Mg²⁺–Au⁰ over its
  16 points, 1.46× O²⁻–Mg²⁺). P has the best decay.
- **One constant per pair is not enough** (replaces the first build's "c_ab: 0.8495 (Au⁰–Au⁰) …"). Three Au⁰–Au⁰ points
  computed after the first build give the ratio 0.8957 at 2.00 Å, 0.8796 at 2.10 Å and 0.8664 at 2.20 Å against 0.843
  at 2.56–2.68 Å: 1 to 1.6 % per 0.1 Å. With the constant of 2.33–2.68 Å the whole interaction was 6.5 % low at 2.2 Å and 10.7 % low at
  2.1 Å; for Au⁰–O²⁻ one least-squares constant (pinned by the two shortest points) left it 13–65 % high at
  2.26–2.6 Å. Hence c_ab(R) through the points. Separate constants for the kinetic and the exchange part were
  tested and rejected: the two are nearly collinear and the pair of constants is not determined.
- The mean ratios, for the record: 0.882 (Au⁰–Au⁰, seven points at 2.00–2.68 Å), 0.962 (O²⁻–O²⁻), 0.828 (Au⁰–O²⁻), 1.208 (Mg²⁺–Au⁰),
  1.254 (O²⁻–Mg²⁺), 1.385 (O²⁻–Al³⁺). The other 39 pairs carry c = 1, the electron-gas value, which on the six would
  be 21 % high to 28 % low.
- For a neutral frame pair the penetration is as large as the Pauli term: Au⁰–Au⁰ at 2.331 Å, +8.77 − 6.30 = +2.47 eV.
  Φ(Au⁰–Au⁰) and its force: +12.34 eV, +46.8 eV/Å at 1.900 Å (below the points: the ratio's bounded continuation);
  +8.42, +32.4 at 2.000 Å; +5.99, +22.6 at 2.089 Å; +5.75, +21.6 at 2.100 Å; +3.96, +14.5 at 2.200 Å; +2.47, +8.7 at
  2.331 Å; +1.09, +3.5 at 2.570 Å. The force falls monotonically with distance (tested on 221 points, 1.9–3.0 Å).
- **The electrostatics is evaluated without cancellation:** D_ff = q_a ψ_b(R) + Z_b ψ_a(R) − (ρ_b | ψ_a)(R), ψ = N/d − φ
  (short-ranged), averaged over the two orientations, with ψ and its integral interpolated by cubic Hermite from
  their exact derivatives and the shell integral split at its kink. The direct form Z_a Z_b/R − Z_a φ_b − Z_b φ_a +
  (ρ_a | ρ_b) (`radial.pair_electrostatic_energy`) is wrong by up to 0.05 eV for two heavy neutral atoms (Au⁰–Au⁰ at
  2.331 Å: −6.3577 against PySCF's −6.3047; the form above: −6.3047). **Confirmed out of sample three times:**
  −16.4908, −12.3100 and −9.2061 eV at 2.0, 2.1 and 2.2 Å against PySCF's −16.4904, −12.3099 and −9.2062, each
  computed after the table. §23's own tables use
  differences and were not affected.

**39.4 Scope, as rules.** (i) Every pair of the library's nine elements (45 pairs), at every distance to the cutoff;
an element outside it is refused. (ii) **An in-situ measurement stands, and this term supplies the rest.** §33's wall
is the lone s atom's interaction with its *whole site* (the anion, its four in-plane cations and the cation below; it
is not a pair function), keyed by the height R above the anion. While it is on, the pair sum of that site at the
measurement's geometry, ref(R) = Φ(s, anion; R) + 4 Φ(s, cation; √(R² + d²)) + Φ(s, cation; R + d) (d = 2.1325 Å),
is taken out again over the wall's own pairs with the wall's own switch (`FramePair.site_reference`). On the measured
site the frame terms then add up to §33's table exactly; elsewhere, and beyond the wall's 4.3 Å, the pair sum acts.
(Replaces the first build's rule, "the pair is skipped": that counted the four Mg²⁺–Au⁰ pairs twice on the O-top
site, +0.34 eV at 2.26 Å and +0.94 eV at 1.8 Å, and dropped Au⁰–O²⁻ beyond 4.3 Å, −0.05 eV per adsorbed atom.)
(iii) The first-principles points of two open shells are their high-spin product, the state in which no electron
hops between the two. (iv) Needs `carrier_penetration` and `carrier_overlap='gordon_kim'` (it is their frame entry);
refused with `frame_wall` (the Mg–Au pair twice) and with `carriers='covalent_units'` (§39.6 v).

**39.5 Tests (45 and one strict xfail).** *Independent of the builder:* D_ff against PySCF's electrostatics of the
real monomer densities at ten points, three of them computed after the formula was fixed (5e-4 eV + 1e-3 relative);
D_ff against the direct four-term form for five light pairs (1e-3 eV); P against a quadrature written out in the
test for five pairs, three of them with c = 1 (5e-3 relative); (§23 + frame) against the direct form for five state
pairs. *The record:* c(R) P goes through every stored first-principles point of all six pairs (1e-5 relative), the
ratio is C¹ at its ends and bounded beyond them, and the whole Au⁰–Au⁰ interaction at 2.0, 2.1, 2.2 and 2.331 Å
equals PySCF's (2e-3 eV). *The pair sum:* two neutral atoms repel, ever more strongly on compression (above 15 eV/Å
at 2.089 Å, the force monotone in R); the Gaussian part of charged frames; isolated limits and additivity; an 8-atom rock-salt cell against
its 2×2×2 supercell; forces and the derivative under a homogeneous strain (where the self-image terms enter) against
central differences, and a finite double backward; refusal of an unknown element. *§33's wall:* on the measured site
the frame terms add up to the wall's table at four heights and both levels; beyond 4.3 Å the pair sum is whole.
*The module:* it adds exactly the term, with equal populations, in the global space and in the cluster space with
the Fermi band; per cell under periodic boundaries at the unit mesh and at a Born–von Kármán mesh, and with §33's wall in the
periodic cell; without the flag
the term's code is never reached; the refused combinations.

**39.6 Limits.** (i) Rigid free-ion densities; O²⁻ in the Watson sphere of rock-salt MgO (§23): for another crystal,
a surface or a molecule its confinement is not that one. (ii) The pair sum is not the whole frozen interaction of a
bonded atom (the bond shields; §33's law is the correction where it has been measured). (iii) 35 of 45 pairs carry
c = 1 (39 until 2026-11-13 evening; Na⁺–Cl⁻, Cl⁻–Cl⁻, Ag⁰–Ag⁰ and H⁰–H⁰ now have points). For a neutral pair Φ is the difference of two large terms, so an error of c is a several times larger relative
error of Φ (Au⁰–Au⁰ at 2.331 Å: c = 1 would give +3.97 eV instead of +2.47). **For two open-shell frames c = 1 is also
below the stated spin convention:** P is the spin-compensated electron gas of the spin-summed densities, the
spin-averaged value, where the high-spin product is higher (H⁰–H⁰ at 0.74 Å: 2.60 against 4.67 eV in the same
functional; the first-principles high-spin frozen product's overlap term is 10.01 eV, a factor 3.86 (the totals
are 2.62 and 10.03 eV), and H⁰–H⁰ now carries it).
C pairs and the mixed pairs of H, Ag and Au need their first-principles points before the term is used on them.
A point enters the table only where the electron-gas term itself is above 5e-3 eV (the ratio to a reference that
passes through zero has no meaning); for H⁰–H⁰ that leaves out 2.6 and 3.0 Å, where the table is then 0.045 eV low
(out of sample), of the wrong sign beyond 2.7 Å. (iv) Au⁰–Au⁰ has
points at 2.00–2.68 Å; below 2.00 Å the ratio continues with its end slope and saturates at 0.948 (0.896 at 2.0 Å):
an extrapolation, stated. Before the 2.0 Å point existed the table was 5.3 % low there (the ratio held constant),
and a zero end slope, the first choice, put a bump of several eV/Å into the force between the first two points. (v) **Covalent units (§19):** a bond inside a unit is held by the backbone with no hop to pair the wall
with (C–C at 1.2 Å: Φ = +8.7 eV per bond); the rule for pairs inside a unit is not decided and the combination is
refused. (vi) The series reproduce their integrals to 7e-6 eV wherever a structure can reach (worst: D_ff of
Au⁰–Au⁰) and 1e-6 eV beyond 4 Å; the energy is C¹ at the knots of c(R), so its second derivative jumps there.
(vii) **The hop law is the other half: built as §40 for pairs of two s¹ frames (2026-11-13 evening), not built for
any other pair.** Without it nothing keeps a learned τ from crossing zero (§31's
anchored form failed inside the selected ensemble, step 21, and overbound short contacts by 3 eV with no repulsion to
stop it; with this term the repulsion exists). (viii) Trained with, never judged frozen: on a backbone trained
without it the frame interaction is counted twice at the data distances. A warm start from such a backbone, or a
resumed run whose settings drop the key, is not flagged by the trainer. (ix) **Inherited, fixed on 2026-11-13 after step 33 was scored:** §23's pair
list scanned the lattice shifts around the raw difference of positions, so a pair was lost once two atoms were more
than about one cell apart in unwrapped coordinates. `penetration.pairs_within` now scans around the minimum-image
difference (the integer lattice vector detached). Checked on all 700 structures of the Au–MgO train and test
files: every structure has an atom outside the cell, none has a fractional span above 1 (largest 0.848), and the
fix changes no pair in any of them, so no trained model and no score is affected. The former strict xfail test
passes.

---

## 40. The two-site sector of two s¹ frames from first principles (`RedoxConfig(two_site=True)`, 2026-11-13; reviewed 2026-11-14) — NEW-TO-TRACE

**Status: built, tested and reviewed; off by default; nothing trained with it** (2026-11-15). **[Updated 2026-11-16:**
step 34b trained with it on both seeds; see §39's status.**]** The review of 2026-11-14 (four
independent readers, eight skeptics; `experiments/benchmarks_4g/runs/two_site_2026-11-14/review_spec40.json`;
README, "SPEC 40 reviewed") confirmed that §40.2–40.3 alone correct one of the pair's four levels: they give the
Au–Au bond its wall (§40.5) and move the dimer's electron affinity and ionisation energy the wrong way by about
1 eV (§40.6 iii). **§40.7 (the same day) fixes the other three levels by the same construction for Au–Au**: the
isolated pair is then first-principles as Au₂, Au₂⁻ and Au₂⁺. Whether a model is trained with it was left to a
separate decision (taken on 2026-11-15: steps 34 and 34b); the review's other findings (§40.6 vii, ix, x, xi) stand. Module `trace_redox/two_site.py`, data `trace_redox/data/two_site.json`, tools
`tools/electrostatics/pair_no_hop_curve.py` and `collect_two_site.py`, tests `tests/redox/test_redox_two_site.py`
(78 tests, no xfail, on 2026-11-15) and `tests/redox/test_redox_cluster_signs.py` (30 and two strict xfails); builders
`tools/electrostatics/pair_one_carrier_curve.py` and `build_two_site_entries.py` (§40.7).

**40.0 Why.** In every trained model the charge module's only Au–Au repulsion was the learned coupling τ shrinking
under compression; τ passes through zero (2.06 Å in step 31, 2.13–2.16 Å in the single model) and the dimer
collapses: on step 33 the five undoped test dimers below 2.18 Å carry 92 % (seed 42) and 94 % (seed 43) of the
undoped squared force error, and without them it is 0.054 eV/Å (criterion 0.10). §39 puts the frozen repulsion of the two frame atoms in; what is still missing is (a) a
hop that cannot change sign and (b) the level of the configuration the hop acts on: §39's frozen term lies 3.5 /
2.2 / 1.7 / 1.0 eV above the first-principles no-hop level of Au₂ at 2.0 / 2.3 / 2.47 / 2.8 Å.

**40.1 The mapping (ESTABLISHED: the two-electron, two-orbital model, as for H₂).** For two half-filled s frames
the neutral pair's sector is {a⁰b⁰, a⁺b⁻, a⁻b⁺}. With the pair's ground-state interaction energy E_gs(R) and its
vertical σσ* triplet and singlet excitations E_T(R), E_S(R),

    E_cov = E_gs + E_T                     the covalent configuration, the level the hop acts on        (40.1)
    U     = E_S − E_T                      the ionic gap,  W = √(U² + 16 t²)
    t     = √(E_T E_S) / 2                 the hop
    E_gs  = E_cov + (U − W) / 2            the identity that closes the three

(the triplet does not hop, so it sits at E_cov; the ungerade ionic singlet sits at E_cov + U; the ground state is
the lower root of the covalent–gerade-ionic 2×2 block, coupling 2t). These identities hold for the exact
eigenvalues of the model; they are applied here to TDA roots of one PBE determinant (§40.6 x).

**40.2 The two terms (NEW-TO-TRACE: each is the pair's own first-principles function; no form is assumed and no
constant is fitted).** For every pair of atoms of an APPLIED element pair (§40.3):

    covalent relaxation   O_ab(f_a, f_b; R) = − s_ρ(R) ρ_ab(R),    ρ_ab(R) = Φ_ab(R) − E_cov,ab(R) > 0      (40.2)
    hop                   τ_ab(R) = − t_ab(R) w(R)                                                          (40.3)

- (40.2) is one entry of the state-pair slot O_ij(n_i, n_j) of (0.2), the frame–frame entry that §23 and §25 leave
  at zero. Φ is §39's term. With it the pair's neutral configuration sits at E_cov; every other state pair keeps the
  frozen level plus the model's own induction, **which is measured not to be enough for the one-carrier pairs**
  (§40.6 iii). ln ρ is the monotone cubic (Fritsch–Carlson slopes with shape-preserving ends) through the points
  and linear beyond them; s_ρ is the C² quintic switch from 1 at 5 Å to 0 at 6 Å.
- (40.3) replaces the learned coupling of that pair (every periodic image within the coupling's cutoff adds, as in
  the learned head); ln t is the monotone cubic through the points and linear beyond them, so t > 0 and dt/dR < 0
  everywhere; w is the coupling's window, 1 − x³(10 − 15x + 6x²) with x = R − (coupling_cutoff − 1 Å) clamped to
  [0, 1]. It needs §29: the neutral pair's covalent–ionic element is then √2 t and a one-carrier resonance
  (a⁻b⁰ ↔ a⁰b⁻) t, one t for both, as first principles has it (Au₂ and Au₂⁻: one t to 0.07 eV).
- Why not Harrison's form with one constant: measured, t / (η ℏ²/m (R² + r₀²)⁻¹) runs 0.71–0.74 over 2.0–2.47 Å for
  Au₂, 0.68–0.88 for Ag₂ and 0.43–0.60 for H₂ (the points that pass the weight rule of §40.3); neither the constant
  nor the form is universal.

**40.3 The points (PBE / def2-TZVP, def2 ECP for Ag and Au; counterpoise-corrected E_gs; TDA roots picked by their
weight on the atoms' valence s combinations).** The rule pre-registered on 2026-11-13, "a root below 0.5 is flagged,
not used", holds for both roots: E_cov needs the triplet (weight ≥ 0.5 at every point); t and U are stored only where
the singlet root carries at least half of the σσ* weight too and continues the monotone fall of E_S. Until the
review the rule was applied to the triplet alone and three hop points rested on singlet roots of weight 0.47, 0.45
and 0.36; they are kept in the file under "flagged" and shown in brackets. Dropping them changes t by −0.2 % at
2.6 Å, −1.0 % at 2.8 Å and −2.1 % at 3.0 Å (Au–Au) and by +0.5 % at 0.6 Å (H–H).

| pair | R (Å) | t (eV) | U (eV) | E_cov (eV) | ρ (eV) |
|---|---|---|---|---|---|
| H–H (not applied) | 0.60 | — (6.458, singlet weight below 0.5; singlet weight 0.47) | — | +7.996 | 7.274 |
| H–H (not applied) | 0.74 | 5.738 | 2.684 | +5.681 | 4.351 |
| H–H (not applied) | 0.90 | 4.952 | 3.597 | +4.017 | 2.348 |
| H–H (not applied) | 1.10 | 4.048 | 4.710 | +2.653 | 1.029 |
| H–H (not applied) | 1.40 | 2.894 | 6.091 | +1.451 | 0.200 |
| Ag–Ag | 2.20 | 1.430 | 1.344 | +1.374 | 1.362 |
| Ag–Ag | 2.30 | 1.366 | 1.409 | +0.761 | 1.237 |
| Ag–Ag | 2.40 | 1.307 | 1.479 | +0.368 | 1.105 |
| Ag–Ag | 2.53 | 1.236 | 1.579 | +0.072 | 0.943 |
| Ag–Ag | 2.65 | 1.176 | 1.681 | -0.070 | 0.803 |
| Ag–Ag | 2.80 | 1.104 | 1.822 | -0.147 | 0.649 |
| Ag–Ag | 3.00 | 1.013 | 2.025 | -0.164 | 0.476 |
| Au–Au | 2.00 | 1.790 | 0.660 | +4.914 | 3.502 |
| Au–Au | 2.10 | 1.648 | 0.762 | +2.807 | 2.942 |
| Au–Au | 2.20 | 1.520 | 0.836 | +1.446 | 2.515 |
| Au–Au | 2.30 | 1.409 | 0.900 | +0.588 | 2.167 |
| Au–Au | 2.47 | 1.249 | 0.995 | -0.163 | 1.683 |
| Au–Au | 2.60 | — (1.146, singlet weight below 0.5; singlet weight 0.45) | — | -0.390 | 1.375 |
| Au–Au | 2.80 | — (1.009, singlet weight below 0.5; singlet weight 0.36) | — | -0.462 | 0.973 |
| Au–Au | 3.00 | — (1.214, root jump; singlet weight 0.46) | — | -0.392 | 0.648 |

E_cov and U in the file are recomputed from the logs' three-decimal roots and can differ from the logged values in
the last digit (Ag–Ag 2.30 Å: +0.761 here, +0.760 in the log). The unrestricted SCF triplet is NOT the no-hop state
for Au₂ below 2.6 Å (it is a 5d → σ* state, d population 9.38–9.45): the vertical TDA triplet of the σσ* pair is
used.

**Which pairs the module uses ("applied" in the file).** Au–Au. **Ag–Ag is tabulated but not applied since §40.7**
(its other three entries cannot be built yet: the polarizability table holds Ag⁻ only, so the pair's fixed terms
depend on the weights; with the neutral entry alone it would be in the state the review rejected). **H–H is
tabulated but not applied:** the
rule acts on element pairs (the formal frame of H, Ag and Au is the neutral atom in every environment), and in a
molecule the two hydrogens of a geminal pair are each bonded to a third atom and lie beyond H₂'s last point (the
continuation gives 2.5 eV at water's 1.51 Å and 1.9 eV at methane's 1.78 Å); the bare pair's hop is not that
pair's, and no first-principles point of such a pair exists. Pairs of two different s¹ atoms have no points.

**40.4 Requirements and refusals.** `two_site` must be True or False and needs `frame_pair` (ρ is defined against Φ),
`spin_pairing`, `couplings` and **`diabatic_selection='amplitude'`**: the 'window' selection does not see the
couplings, so a configuration reached by the first-principles hop would enter only inside the energy window, and
the energy would step where it does (measured on gas-phase Au₂ with untrained heads: steps of 1.6–2.0 eV near
1.95 Å and 0.24 eV at 3.3 Å; H₂ never hopped). The coupling's window must start at or beyond the last hop point
and, where the pair has the entries of §40.7, their last distance (Au–Au: 3.0 Å, so `coupling_cutoff` ≥ 4.0 Å),
else the forward pass is refused. A model whose options give other fixed terms than the ones a pair's entries
were built against is refused (§40.7).
Covalent units are refused through §39. An s¹ atom of an applied pair that meets its own periodic image within
6 Å is refused, also when it is the only such atom in the cell: the hop to an own image does not exist in the
cell's ensemble, so the relaxation of that pair would stand alone and the same crystal would depend on the cell
chosen (before the review a cell with one such atom was skipped silently: an Au chain at 2.9 Å was left alone
with one atom per cell, refused with two, and lowered by 0.80 eV per atom with three). A pair whose frozen term
lies below its covalent level (ρ ≤ 0 at a point) is refused; a malformed table (distances not increasing, a hop
that is not positive, fewer than two hop points, a pair of two different elements) is refused when it is read.

**40.5 Mechanism check on a trained model (frozen weights; pre-registered and scored in
`experiments/benchmarks_4g/README.md`, 2026-11-13; the final step 33 model of seed 43; test 167's rigid Au–Au
scan; the bond force of E_q alone, trained route).** C1 pass: repulsive from 2.00 to 2.40 Å and monotone (+18.1, +14.6, +12.5,
+9.9, +7.9, +6.3, +4.7, +2.3 eV/Å at 2.00, 2.05, 2.089, 2.15, 2.20, 2.25, 2.30, 2.40 Å; as trained −20.1, −19.8,
−18.8, −9.3, +5.7 …). C2 pass: +12.5 eV/Å at 2.089 Å against DFT's +15.25 (−18 %; gas-phase first principles
+14.5). C3 pass: the covalent–ionic element is √2 t(R), +2.53 → +1.65 eV over 2.00 → 2.57 Å, one sign; as trained
it passes through zero near 2.16 Å (the element and the gap are read in the 'global' / 'phi1' route, whose total
bond force differs from the trained route's by 0.3–0.7 eV/Å with the terms on). §39 alone, and §39 with §29,
leave a force that dips to +3.8 and −1.2 eV/Å at 2.089 Å. Repeated on 2026-11-14 with the table after the weight
rule of §40.3: the same to the printed digit from 2.00 to 2.30 Å, +2.34 at 2.40 Å and −0.13 (was −0.09) at 2.57 Å.
This is a check of the mechanism, not of accuracy: the backbone of a model trained without §39–40 holds the Au–Au
repulsion at the data's distances and its level shifts hold the missing relaxation, so both are counted twice here.

**40.6 Limits (as written after the review; (iii) describes §40.2–40.3 alone and is answered for Au–Au by §40.7; its numbers,
the hop-alone ones included, were measured while d-electron moves still carried the s–s hop: on the step-resolved code
§40.2–40.3 alone leave the ionisation energy 1.2–1.7 eV high, the electron affinity as stated; §40.7's table holds the
current values).**
(i) Gas-phase pair functions on adsorbed and embedded pairs; the check of §40.5 is the only
evidence so far that the adsorbed bond's force is the pair's own (to 18 %). (ii) ρ is pair-additive over the neutral
pairs of a configuration (three Ag atoms: the neutral configuration receives −3ρ, each one-carrier configuration
−ρ); not measured against first principles. (iii) **Only one of the pair's four levels is corrected, and the
other three are then worse (review, 2026-11-14; `probe_au2_ea_ip.py`).** Gas-phase Au₂ with §39 + 29 + 40 and no
learned shift (the package's fixed terms; identical on three checkpoints), at 2.30 / 2.47 / 2.60 / 2.80 Å:
the ionic gap is right (U to 0.01 eV at 2.47 Å; −0.46 / −0.70 / −0.95 eV at 2.2 / 2.1 / 2.0 Å), but the vertical
electron affinity is 0.17 / 0.77 / 1.13 / 1.58 eV against 1.85 / 2.03 / 2.17 / 2.37 (errors −1.68 / −1.26 / −1.04 /
−0.79) and the ionisation energy 10.54 / 10.38 / 10.22 / 9.93 eV against 9.43 / 9.22 / 9.09 / 8.91 (errors +1.11 /
+1.16 / +1.13 / +1.02); references PBE0, dimer minus atom on the exact atomic values, stated spread 0.2 eV. With
the same hop and no relaxation the errors are −0.68 / −0.39 / −0.28 / −0.19 and +0.11 / +0.29 / +0.36 / +0.42. The
learned shift is one-body and can move a level, but not the combination
I = E(a⁺b⁻) + E(a⁰b⁰) − E(a⁺b⁰) − E(a⁻b⁰), which (40.2) changes by exactly −ρ (−5.773 → −7.456 eV at 2.47 Å; first
principles, from the same references, −4.60): a shift that restores the electron affinity and keeps U leaves the
ionisation energy 1.85 eV high. Read against first principles at 2.47 Å, the fixed terms' levels above the
covalent one are off by +1.25 eV (a⁻b⁰), +1.60 eV (a⁺b⁰) and −0.01 eV (a⁺b⁻): every pair that holds a neutral open-
shell atom lacks a relaxation of the size of ρ, and the closed-shell ion pair does not. **The general form this
points to, not built: the one-carrier entries (−1, 0) and (+1, 0) of the pair slot carry their own first-principles
relaxation by the same construction (the three-electron and one-electron resonance doublets, E(a b^∓; R) + t(R)
against the separated ion and atom).** It bears on the doped Au–MgO class (Au₂⁻) and on Ag₃±. (iv) Homonuclear
pairs only. For two different s¹ atoms the singlet block has two ionic levels and
t² = −λ₀λ₁λ₂ / (2(λ₀ + λ₁ + λ₂)) from its three eigenvalues measured from the covalent level E_cov; the third is a
double excitation that TDA does not give. Not built. (v) Mixed levels of theory: Φ from the library densities
(PBE0, X2C, ANO; the frozen product in PBE), E_cov from PBE / def2-TZVP with an ECP; their difference is exact at
the points by construction, and ρ carries the inconsistency. (vi) Beyond the points both functions are
exponentials continued from the end slopes: Au–Au's hop below 2.0 Å and above 2.47 Å, its ρ above 3.0 Å; Ag–Ag
outside 2.2–3.0 Å; H–H outside 0.74–1.4 Å (the vertical triplet of stretched H₂ is not available beyond the
restricted reference's instability). (vii) **Signs on rings.** The hop has one sign, −t; what that gives beyond
two sites is decided by the configuration Hamiltonian's sign structure (§17.2: Jordan–Wigner signs of spin-less
local orbitals), which is exact for one carrier of either kind on any cluster and on two sites at any filling, and
not exact in general for two or more carriers on three or more sites (in t, at U = t / 4t, against the exact
lowest level of the lowest spin): two electrons, chain of three +0.51 / +0.40, triangle +1.06 / +1.12, square
+1.16 / +1.08, tetrahedron +2.07 / +2.17; two holes, chain +0.51 / +0.40, triangle −0.27 / −0.63 (the model sits
on the triplet, the true ground state there), tetrahedron exact; **at half filling, the neutral sector of three or
more s¹ atoms, the model lies BELOW the exact ground state of any spin** (chain −0.11 / −0.23, triangle −0.61 /
−0.73, square −0.50 / −0.48, tetrahedron −1.08 / −1.19): it over-binds. With Ag–Ag's own values at 2.65 Å (t 1.18,
U 1.68 eV) that is −0.77 eV for a neutral triangle and +1.26 eV for Ag₃⁺ (a reviewer's evaluation of the two-site
model, not a model energy). A learned coupling shares the limit.
The comparison of two sign rules on the Au–MgO slabs (0.00 meV on 32 structures) carries no evidence either way:
the selected configuration sets there are trees (the reference and single moves, no closed loop), whose spectrum
does not depend on any sign. (viii) Trained with, never judged frozen (as §39.6 viii). (ix) **[Cured on 2026-11-14 by §40.7's step-resolved hop; kept as the record of the defect.] One coupling per
atom pair (§17.2):** (40.3) replaced T[a, b] for every charge step of the pair. Of the 16 one-electron moves of an
Au or Ag pair with charges −1…+3, four stay within −1, 0, +1 (the s moves); the other twelve move a d electron and
carry the same −t w (times √2 on equal parities), a value not derived for them and not trainable. At frozen step
33 weights none of those configurations is selected on 80 test structures (they lie 8.5–10.4 eV up); it matters
for a cationic pair. The cure, a coupling resolved by charge step, is §40.7's;
a test (`test_the_tabulated_hop_is_given_to_s_moves_only`) holds it. (x) **The
mapping is exact for exact energies and is applied to TDA-PBE roots.** For a mean-field reference the same model
gives E_T E_S = 4t² − U²/4, so √(E_T E_S)/2 falls short of t as U grows (the mean-field reading (E_T + E_S)/4
exceeds the stored t by 0.4–4 % for Au, 3–12 % for Ag, 0.3–13 % for H); E_cov lies 0.05–0.18 eV below the SCF σσ*
triplet at 2.6–3.0 Å. Which reading TDA-PBE follows is not decided by any datum here; one independent number per
pair that does not pass through the singlet root (half the σ_g–σ_u splitting of the one-carrier ion) would decide
it. (xi) ln t and ln ρ are C¹ at the points, so the second derivative of the energy jumps there, and the points
sit on the dimers' equilibrium distances (a reviewer's evaluation before the weight rule: 13.4 against 11.3 eV/Å²
on the two sides of 2.47 Å for Au₂): a frequency or a Hessian is not to be read at a point.

**40.7 The pair's remaining entries: the isolated pair is first-principles in three charge states (2026-11-14;
NEW-TO-TRACE; built for Au–Au; corrected the same day after its review).** §40.6 (iii) as reviewed: (40.2) corrects
one of the pair's four levels. The other three are fixed the same way, each by one condition on the ISOLATED pair
through the package's own FIXED terms (no learned level shift and no learned coupling:
`module.fixed_terms_only`) and the tabulated hop (40.3):

    O_ab(∓1, 0) = O_ab(0, ∓1) = − s ρ_∓(R)      such that   E[X₂^∓](R) = E_gs(R) ∓ {EA*(R), −IP*(R)}          (40.4)
    O_ab(+1, −1) = O_ab(−1, +1) = + s λ(R)      such that   E[X₂](R)  = E_gs(R)                               (40.5)

E_gs is §40.3's ground-state interaction energy; EA*(R) and IP*(R) are the pair's first-principles vertical
electron affinity and ionisation energy on the exact atomic values (ESTABLISHED quantities; PBE, the cation in
def2-TZVP and the anion in def2-TZVPD, both resonance states of each ion pair as symmetry-counted self-consistent
states; `tools/electrostatics/pair_one_carrier_curve.py`; the tool reproduces H₂⁺ in aug-cc-pVTZ, E_int −2.788
against −2.793 eV and the σ_g–σ_u splitting 11.840 eV). With (40.2) the isolated pair's ground state in the model's
space is then the first-principles one in all three charge states at every tabulated distance, with ONE hop for
all sectors. (40.5) fixes the ionic gap GIVEN the hop: with E_cov = E_gs + E_T the condition is U = 4t²/E_T − E_T,
which is E_S − E_T at the hop's five points; at 2.6, 2.8 and 3.0 Å the hop is its continuation and U inherits it
(dU/dt = 8t/E_T, about 5).
- **The hop is the coupling of s moves only (the cure of §40.6 ix, needed here).** A move of the pair is an s move
  when an s electron leaves the donor (charge −1 or 0 before) and enters the acceptor's s shell (charge 0 or +1
  before); (40.3) is the coupling of those, and every other move of the pair (Au⁺Au⁰ ↔ Au²⁺Au⁻: a 5d electron
  leaves) keeps the learned coupling (`trace_redox.diabatic.StepCouplings`, threaded through the Hamiltonian, the
  amplitude selection and the module's cluster route; `clusters.cluster_ensemble`, a standalone function the module
  does not call, accepts the record too). NEW-TO-TRACE as a rule of the configuration model; that the s and d
  ionisations of a coinage-metal atom are different processes is ESTABLISHED atomic physics. As first built, the cation's condition was solved on a ground state
  reached through Au²⁺Au⁻ with the s–s hop (that configuration lies only 1–5 eV up at zero shift and lowered the
  cation by 0.88 / 0.33 / 0.06 eV at 2.0 / 2.47 / 3.0 Å), ρ₊ had absorbed it, and through the selection's switch on
  that coupling the entries depended on the selection thresholds (0.16 eV) and on kT. Found by the review; now the
  cation's fixed-term ground state is the two-level resonance exactly.
- **Why the conditions can be imposed by a table (measured).** The pair's fixed-term energies do not depend on any
  weight: the same on three trained checkpoints, after every one of 193 521 parameters of a trained model is
  perturbed by 5 % and by 30 %, and in freshly built modules (two seeds: 9e-16 eV). The pair's electrostatics and
  overlap are tables (§23, §25, §39) and the polarizability of Au⁰, Au⁻, Au⁺ is tabulated (§28, §35). A sweep of
  every field of RedoxConfig and ElectrostaticConfig (booleans flipped, numbers halved and doubled, the named
  alternatives; a test) finds five that move them: `bond_partition`, `pauli_table`, `carrier_induction`,
  `frame_polarizability_anchor` and the electrostatic `gaussian_a` (it sets the dipoles' widths; 0.24 eV). Every
  other one is refused as an illegal combination or leaves them within 1e-6 eV, the selection thresholds included;
  kT has its own thermal weight only (3e-5 eV at 0.2 eV).
- **The build** (`tools/electrostatics/build_two_site_entries.py`): each condition is one equation in one unknown
  at each distance, solved by the secant method on the package's own energy (largest residual 6e-11 eV), and
  written into `two_site.json` with the option set, `gaussian_a`, and the table values it was solved with (E_gs,
  E_T, t, EA*, IP* and ρ at the knots). **Refused:** a model whose options or `gaussian_a` differ
  (`TwoSite.check_options`); a frame table that gives another ρ at the knots; a file whose table has changed since
  the build; a pair marked applied without built entries; a coupling window that starts inside the entries'
  distances (so `coupling_cutoff` ≥ 4.0 Å for Au–Au).
- **Gold's entries (eV; the package's fixed terms without them, then the entries):**

| R (Å) | E_gs | package E₀ | EA* | package EA | IP* | package IP | ρ₋ | ρ₊ | λ |
|---|---|---|---|---|---|---|---|---|---|
| 2.00 | +1.648 | +1.188 | 1.355 | −1.631 | 9.937 | 11.682 | 2.526 | 1.285 | +0.945 |
| 2.10 | −0.129 | −0.457 | 1.485 | −0.873 | 9.718 | 11.509 | 2.029 | 1.463 | +0.701 |
| 2.20 | −1.205 | −1.411 | 1.604 | −0.288 | 9.538 | 11.336 | 1.685 | 1.591 | +0.459 |
| 2.30 | −1.815 | −1.927 | 1.715 | +0.173 | 9.390 | 11.153 | 1.431 | 1.651 | +0.257 |
| 2.47 | −2.212 | −2.215 | 1.893 | +0.772 | 9.192 | 10.822 | 1.117 | 1.627 | +0.008 |
| 2.60 | −2.211 | −2.162 | 2.022 | +1.130 | 9.073 | 10.559 | 0.942 | 1.535 | −0.130 |
| 2.80 | −1.980 | −1.875 | 2.211 | +1.581 | 8.922 | 10.153 | 0.735 | 1.336 | −0.301 |
| 3.00 | −1.649 | −1.509 | 2.389 | +1.943 | 8.793 | 9.778 | 0.586 | 1.125 | −0.444 |

  The one-extra-electron pair lacked a relaxation of 0.6–2.5 eV and the one-hole pair 1.1–1.65 eV (the neutral pair
  1.0–3.5 eV, §40.3); the ion pair's level was 0.9 eV too low at 2.0 Å and 0.4 eV too high at 3.0 Å. Each entry is
  the C¹ shape-preserving cubic through its knots, in the value (λ changes sign). Below 2.0 Å it continues on its
  end slope (on it ρ₊ reaches zero near 1.37 Å). **Beyond 3.0 Å every entry continues on its line and is switched
  off (C² quintic) over 1 Å**, ending at 4.0 Å where the coupling's window has closed the hop the entries complete:
  a NEW-TO-TRACE choice, not a measurement. The one-carrier conditions are met on the two-level resonance of a⁺b⁰ and
  a⁰b⁺ (ESTABLISHED, as for H₂⁺).
- **Which pairs.** Au–Au. **Ag–Ag is withheld from the module** (tabulated, with its carriers at seven distances):
  the polarizability table holds Ag⁻ only, so the neutral silver frame's polarizability is the learned head's, the
  pair's fixed terms depend on the weights (0.057 eV between two seeded modules) and the entries cannot be built.
  H–H is withheld as before. A pair without built entries cannot be applied (the file is refused).
- **First-principles facts found on the way.** The hop of the ion pair's own resonance is not the neutral table's:
  the cation's is 13–28 % larger (Au) and 9–27 % (Ag), the anion's within −8 to +10 % (Au) and 8–21 % smaller (Ag).
  With one hop per pair the conditions make each ground state right and leave the excited member of each doublet
  off by twice that difference. The no-hop levels from PBE lie 0.2–0.5 eV below PBE0's; EA* and IP*, which are
  what enters, differ by at most 0.14 eV between the two. PBE is used for all three, the level of the neutral's
  table. **Below about 2.4 Å the ground state of Au₂⁺ in PBE is not the σ_g state the cation's entry is fixed on
  but a d-hole state** (unconstrained UKS: 1.07 eV lower at 2.0 Å, 0.32 eV at 2.2 Å, the same state at 2.47 Å; spin
  0.91–0.95 on d functions): the model's Au⁺ is 5d¹⁰ and has no such state.
- **Frozen check, pre-registered (the final step 33 models; `probe_au2_ea_ip.py`, `probe_auau_two_site.py`,
  `probe_doped_pair_states.py`), on the corrected build.**
  Gas-phase Au₂ through the fixed terms, both seeds identical to the printed digit: EA 1.715 / 1.893 / 2.022 / 2.211 eV
  and IP 9.390 / 9.192 / 9.073 / 8.922 eV at 2.30 / 2.47 / 2.60 / 2.80 Å (references 1.85 / 2.03 / 2.17 / 2.37 and
  9.43 / 9.22 / 9.09 / 8.91): errors −0.14 to −0.16 and −0.04 to +0.01 eV, **pass** (criterion 0.2 eV; before
  §40.7: −1.68 to −0.79 and +1.11 to +1.02). The ionic gap at the hop's five distances is the table's U to the
  printed digit (0.660 … 0.995 eV). The wall check (test 167's scan, seed 43, the bond force of E_q alone, trained
  route): +19.65 / +15.82 / +13.51 / +10.68 / +8.48 / +6.77 / +5.13 / +2.64 / +0.05 eV/Å at 2.000 / 2.050 / 2.089 /
  2.150 / 2.200 / 2.250 / 2.300 / 2.400 / 2.570 Å: **C1 pass** (repulsive and monotone to 2.40 Å), **C2 pass**
  (+13.51 against DFT's +15.25, −11 %; +12.51 before the entries; my predicted window 13.5 to 15.5), **C3 pass**
  (the covalent–ionic element +2.532 → +1.651 eV). The wall check is the same to the printed digit before and
  after the hop was resolved by charge step (the undoped slab holds no one-hole state of the pair). As for §40.5,
  frozen weights: a check of the construction and of the mechanism, not a test force of a trained model.
- **Limits of 40.7.** (a) It makes the ISOLATED pair right in the model's space; on a surface the pair functions
  are transferred (limit i), and the pair's own induction, which is inside the conditions, is screened there by
  the neighbours'. (b) EA* and IP* carry the level-of-theory spread, up to 0.14 eV; EA* is formed in def2-TZVPD
  and set against the table's E_gs in def2-TZVP (the neutral's two basis sets differ by 12–16 meV). (c) The fixed
  terms depend on temperature through the thermal weight of each doublet's upper member only (3e-5 eV at
  kT = 0.2 eV); the entries were solved at 0.1 eV. (d) **The description ends at 3.0 Å.** Between 3.0 and 4.0 Å the
  hop's window and the entries' switch take it off: the isolated pair's binding goes from −1.65 eV (3.0 Å) to −0.20
  (3.6 Å) for Au₂, with a force of up to 3.09 eV/Å at 3.32 Å (1.73 at 3.0 Å), and likewise 2.59 eV/Å at 3.39 Å for
  Au₂⁻ and 3.36 eV/Å at 3.38 Å for Au₂⁺ (`experiments/benchmarks_4g/probe_two_site_beyond_points.py`, 0.01 Å grid;
  `runs/two_site_2026-11-15/probe_two_site_beyond_points.txt`). A dimer stretched beyond 3.0 Å is not described; the 4G sets hold one such structure (3.056 Å). The
  cure is first-principles points to larger distances and a coupling cutoff beyond them. Below 2.0 Å the entries
  are linear continuations. (e) The entries are tied to an option set, the electrostatic `gaussian_a`, the table
  values and the content of the four data files the pair's fixed terms read (ion densities, the Pauli overlay, the
  ion polarizabilities, the frame-pair table; sha256 recorded and compared on reading); each mismatch is refused, as
  is an entries block without that record and a hook call that does not pass the model's settings. NOT guarded: a
  change of the package's code or constants that moves the fixed terms (the atomic EA and IP, the formulas); the
  test that recomputes the three conditions at every distance, and `build_two_site_entries.py --check`, see it
  when they are run. (f) The excited member of
  each doublet, and with it any property that needs the ion pair's own hop, is not corrected. (g) The cation's
  entry is the s-hole state's; the d-hole ground state of Au₂⁺ at compressed distances is outside the model.
  (h) Pair states without an entry keep the frozen level: both atoms negative, and any charge beyond ±1. On the
  final step 33 model of seed 43 with the terms on, 40 + 40 test slabs: the classes with entries carry all of the
  weight on the undoped set (neutral 0.805, ion pair 0.192, one electron 0.003, one hole 0.000) and 0.997 on the
  doped set (one electron), Au⁻Au⁻ 0.003 there (at most 0.012 in a structure). (i) The first-principles states are
  converged, symmetry-counted SCF states; the stability analysis named in the pre-registration was not run for
  them (the unconstrained check of (g) is the only one).
