# TRACE-Electrostatics: methods

This document states the equations of TRACE-Electrostatics as the released models
use them, and the training methodology that produced those models. It is a reader's
guide to two contracts, which remain the authority:

- **E-SPEC n**: [`trace_electrostatics/SPEC.md`](../../trace_electrostatics/SPEC.md),
  section n. This is the bound-charge sector. Sections 17–26 are current; SPEC 2
  (the fixed-k solve) is legacy.
- **R-SPEC §n**: [`trace_redox/SPEC.md`](../../trace_redox/SPEC.md), section n.
  This is the free-charge (redox) sector, the TRACE-Redox package. Its §0 states the
  current model in three equations.

Every equation below is copied from the section cited next to it. Where this
document adds a sentence of its own, it says so. Changes to the physics are
recorded in [`trace_electrostatics/CHANGELOG.md`](../../trace_electrostatics/CHANGELOG.md).
Measured results and their pre-registrations are in
[`BENCHMARKS.md`](BENCHMARKS.md) and the two benchmark notebooks
(`experiments/benchmarks_4g/README.md`, `experiments/benchmarks_phonons/README.md`).
The checkpoints and their sha256 are listed in [`MODELS.md`](MODELS.md), and
the commands that rebuild them are in [`REPRODUCE.md`](REPRODUCE.md).

**Units.** Energy eV, length Å, charge e. `K_E = 14.399645` eV·Å/e²
(`trace_electrostatics/constants.py`, the single source of truth). Every
computation runs in float64; the package refuses float32 (E-SPEC 0).

**Marking.** Every term is marked **ESTABLISHED** (with the citation the SPEC gives) or
**TRACE-specific** (the SPECs write NEW-TO-TRACE).

---

## Contents

1. [What is claimed, and what is not](#1-what-is-claimed-and-what-is-not)
2. [The total energy of a released model](#2-the-total-energy-of-a-released-model)
3. [The bound sector (E-SPEC 1, 10–26)](#3-the-bound-sector-e-spec-1-1026)
4. [The free-charge sector (R-SPEC §0 and the sections it maps)](#4-the-free-charge-sector-r-spec-0-and-the-sections-it-maps)
5. [Which terms are on in which released model](#5-which-terms-are-on-in-which-released-model)
6. [Training methodology](#6-training-methodology)
7. [The phonon protocol (P0–P6) and the supercell question](#7-the-phonon-protocol-p0p6-and-the-supercell-question)
8. [Stated limits](#8-stated-limits)
9. [References](#9-references)

---

## 1. What is claimed, and what is not

A novelty audit of 2026-09-19 found every individual ingredient of the bound
sector already published. No single ingredient is claimed as new. The contribution
claimed is the TRACE-native combination:

- SQE bond variables with the dissociation envelope, whose bond hardness is read
  from TRACE's attention (E-SPEC 1.3, 1.4, 1.4.1, 17.4);
- the converged constrained minimum differentiated by implicit derivatives, to
  every order (E-SPEC 17);
- positive definiteness of the whole (q, μ, Θ) operator for every geometry, by the
  Gram construction (E-SPEC 12, 14);
- the bond-current polarization, which makes the itinerant polarization a state
  function of the configuration (E-SPEC 15);
- metals as zero-hardness spanning stars, with a circuit bond for electrodes
  (E-SPEC 20);
- empirical-valence-bond (EVB) charge states on top of the bound sector (E-SPEC 21).

The free-charge sector (R-SPEC) adds terms that its own contract marks NEW-TO-TRACE,
each built from cited ingredients. Sections 4 and 5 list each one with its marking.

Not every term in the contracts is in every released model. Section 5 says which
terms each released model trained with, read from the stored configurations of the
checkpoints.

---

## 2. The total energy of a released model

A released model is TRACE's local equivariant attention potential plus one
electrostatic sector:

    E_total(R) = Σ_i E_ref(Z_i) + E_local(R) + E_elec(R)

(This decomposition is this document's summary of `train.py` and
`flashace/model.py`; it is not an equation of either SPEC.)

- `E_ref(Z)` are per-element reference energies. They are solved from the
  training set (`solve_atomic_energies: true`). Step 26 instead fitted them on
  the initial model (`atomic_energies_from_model: true`). Every later warm start
  inherits them from its parent checkpoint.
- `E_local` is the TRACE backbone (`flashace/`): hidden dimension 64, `l_max = 2`,
  one layer, cutoff `r_max = 5.29` Å in every released electrostatic model.
- `E_elec` is one of two things:
  - **bound sector only** ("elec" and "elecq" models): the converged minimum
    of E-SPEC 1.5.1 / 17 (Section 3);
  - **TRACE-Redox** (steps 9, 26, 31, 33 and 34b): `E_short + F` of R-SPEC (0.3),
    where the bound sector is eliminated exactly inside every integer charge
    configuration (Section 4). In (0.3), `E_short` is the local energy above.

---

## 3. The bound sector (E-SPEC 1, 10–26)

### 3.1 Kernel and hardness floor (E-SPEC 1.1, 1.2, 17.3) — ESTABLISHED kernel, TRACE-specific floor

    J(r_ij) = K_E erf(r_ij / σ_ij) / r_ij ,   σ_ij = sqrt(σ_i² + σ_j²)
    J(0)    = 2 K_E / (σ sqrt(π))

Gaussian (erf) charges only. The Ohno/Klopman form produced an indefinite
charge block in audit and is forbidden (E-SPEC 1.1).

    η_i ≥ S_0(σ_i) = sqrt(2) K_E / (σ_i sqrt(π))        (7.659503 eV at σ = 1.5 Å)
    η   = softplus(raw) + S_0(σ) + margin,   margin = 1 eV/e²

With its true diagonal, `J` is the Coulomb Gram matrix of the Gaussian clouds,
so `diag(η) + J_off = Gram + diag(η − S_0)`. It is positive semi-definite for every
geometry iff `η_i ≥ S_0` (E-SPEC 1.2). The floor `S_0` replaced an older
`sqrt(2)`-larger one, which E-SPEC 1.2 retracts as a sufficient condition that had
been presented as a necessary one.

### 3.2 Charges on bonds, and per-fragment constraints (E-SPEC 1.3, 17.4)

    q_i = Σ_{j∈N(i)} t_ij + ρ_i ,   t_ij = −t_ji,   q = B diag(f) τ + ρ

- `B` is the bond incidence matrix, so `1ᵀB = 0` exactly.
- `ρ` carries the net charge. **It is a fixed constraint carrier, not variational.**
  With `ρ` variational the minimiser gave `|t| = 1.65e-07` against `3.57e-01`:
  `ρ` absorbs everything (E-SPEC 1.3).
- Constraints are per fragment, `Σ_{i∈F} q_i = Q_F`. A single global constraint is
  size-inconsistent: a +1 ion with distant Ar spectators kept 1.000, 0.242, 0.102,
  0.052, 0.016 e at 0, 5, 20, 50, 200 spectators, and per fragment gives 1.000000 at
  every count (E-SPEC 1.3).
- Every bond joining two different fragments is pinned, `τ_b = 0` (E-SPEC 17.4).
- The general principle (E-SPEC 1.3): any per-atom charge channel that bypasses the
  bond-hardness penalty absorbs all the charge and drives `t → 0`. There is
  therefore no geometry-dependent reference charge `q0_i(h_i)`.

### 3.3 Bond hardness from attention, and the dissociation envelope (E-SPEC 1.4, 1.4.1, 17.4) — TRACE-specific

    κ_ij = softplus( ½ (s_ij + s_ji) ) + κ_0 ,   κ_0 > 0

`s_ij` are attention logits. Attention is not symmetric, so κ is symmetrised; an
asymmetric κ would make the energy ill-defined because `t_ij = −t_ji` (E-SPEC 1.4).
The limit `κ → 0` must reproduce global QEq to 1e-15. That is a regression test,
and it is what keeps metals reachable.

The bond variable is `t_b = f(r_b) τ_b`, with the hardness `½ κ τ²`. Physically,
`½ κ τ² = ½ (κ/f²) t²`: the effective hardness diverges as the bond stretches
(the SQE dissociation limit, ESTABLISHED: Mathieu, J. Chem. Phys. 127, 224103 (2007)),
while the τ-space diagonal stays κ (E-SPEC 1.4.1). The envelope is the C² quintic
shell switch (E-SPEC 17.4):

    f = 1                         for r ≤ 0.75 r_c
    f = 1 − x³ (10 − 15x + 6x²)   on the shell 0.75 r_c ≤ r ≤ r_c

- Continuity at the cutoff is exact at any iteration count. The step across
  `r_c ± 1e-6` Å was measured as 3e-28 eV, where it had been 1.68e-3 eV with `t` as
  the variable (E-SPEC 1.4.1).
- The full-range cosine it replaced put a fixed-sign `2f′/f` into every Born charge
  (rock salt with a 3.5 Å cutoff: `Z*_Na = −0.90` for `q_Na = +0.24`). The bond
  cutoff must be at least about 1.4× the first-shell bond length (E-SPEC 17.4).

### 3.4 The master equation of the bound sector (E-SPEC 1.5.1, 12)

    E_LR = min_m U(m)  +  E_disp

    U(m) = ½ mᵀ K m + cᵀ m,     m = ( t, μ̂, Θ̂ ),   q = B diag(f) t + ρ
    μ̂_i = μ_i / (e ℓ_i),   Θ̂_i = Θ_i / (e ℓ_i²)

        [ BᵀJB + diag(κ)    BᵀG01        BᵀG02       ]
    K = [       .          A_μ + G11      G12         ]
        [       .              .        A_Θ + G22     ]

    c = ( Bᵀ(Jρ + χ),  G01ᵀρ,  G02ᵀρ + χ_Θ )

    G_ab(i,j) = the (a,b)-rank derivative of K_E erf(r_ij / s_ab,ij) / r_ij,
    s_ab,ij = sqrt(σ_a,i² + σ_b,j²)

U is quadratic only. A quartic term failed audit on eleven counts, including a
spurious hyperpolarizability 300× a real chemical β in a nominally centrosymmetric
system (E-SPEC 1.5).

**The Gram theorem (E-SPEC 12) — TRACE-specific use of an ESTABLISHED construction.**
Each site carries one Gaussian shape per rank. Every block of K is then a Coulomb
inner product, and the full (q, μ, Θ) matrix including its self blocks is a Gram
matrix:

    K = Mᵀ (Gram + D) M + diag(κ, 0, 0),   M = blockdiag(B diag(f), I, I)
    D = blockdiag(η − S_0,  A_μ − ℓ² S_1,  A_Θ − ℓ⁴ S_2)
    S_0 = sqrt(2) K_E/(sqrt(π) σ),  S_1 = K_E/(3 sqrt(π)/sqrt(2) σ³),  S_2 = K_E/(5 sqrt(2π) σ⁵)

K is positive definite whenever κ > 0 and D > 0. The dipole and quadrupole widths
are derived from the learned stiffness:

    σ_l^(2l+1) = K_E ℓ^(2l) ||A⁻²||_F^(1/2) / (a V_l),   0 < a < 1   (gaussian_a = 0.5)

This gives `ℓ^(2l) S_l ≤ a λ_min(A)`, with residual `≥ (1 − a) λ_min`. The damping
factors are the regularised incomplete gamma functions,
`λ_{2n+1}(r) = P(n + ½, r²/s²)`.

- ESTABLISHED: Gaussian charges make J a Gram matrix; Gaussian induced dipoles
  (Elking, Darden, Woods, J. Comput. Chem. 28, 1261 (2007)).
- TRACE-specific: the per-rank floors derived from the learned stiffnesses, which
  make the bond-SQE + dipole + quadrupole Hessian positive definite by construction.
- Why it is required: with Thole coupling, the (q, μ, Θ) Hessian at the heads'
  initial values was indefinite in 4 of 121 six-atom clusters, while every diagonal
  block and every pairwise sub-block was positive definite (E-SPEC 12). Thole
  remains available as an option; its `a < 1` guarantee holds for a pair only
  (E-SPEC 1.6).

### 3.5 The converged energy and its implicit derivatives (E-SPEC 17) — TRACE-specific

    E(θ) = min_{x∈V} U(x; θ),   U = ½ xᵀKx + cᵀx + u_const

- The solve is preconditioned projected CG to a relative residual `solve_rtol = 1e-10`
  (`solve_mode = "converged"`, the default in every released model).
- The energy's backward pass is the envelope theorem,
  `dE = x*ᵀ dc + ½ x*ᵀ dK x*`: forces need one extra operator application and no
  linear solve.
- The minimiser's backward pass is an adjoint solve `K s = g` on V, produced by the
  same function, so it is differentiable again. This gives the second derivatives
  that force training needs.
- Checked against a closed form: value, first, second and third derivatives of E
  to 1e-9 relative. The work of the forces around a closed loop of a 16-atom
  cluster is −1.2e-9 eV at `solve_rtol = 1e-10`, the quadrature floor (E-SPEC 17.2).
- Why the fixed-k energy (E-SPEC 2, legacy) was replaced: its response to anything
  the zero-field residual does not excite is arbitrary. Fixed-k α at k = 6 / 12
  against the exact minimiser: Td 0.315 / 0.315 vs 0.363; Oh 0.399 / 0.399 vs 0.730;
  D6h −2.25 / +3.82 vs 1.46 (E-SPEC 17.1).
- The converged CG raises `IndefiniteOperatorError` on the first direction with
  `pᵀKp ≤ 0`, and `SolverNotConvergedError` rather than return a truncated iterate
  (E-SPEC 17.5).
- Preconditioner: the Woodbury identity on the bond block (E-SPEC 17.6),

      (κ + BᵀCB)⁻¹ = κ⁻¹ − κ⁻¹Bᵀ(C⁻¹ + Bκ⁻¹Bᵀ)⁻¹Bκ⁻¹.

- The solver seed (E-SPEC 2.0.1) is only the starting point of the converged solve
  and changes the iteration count, never the energy. `train.py` turns its loss off
  for converged runs. The first-round cluster runs still had it on, at weight 1.0.

### 3.6 Periodic cells, the bond-current polarization and the enthalpy (E-SPEC 14, 15, 16, 18, 19) — TRACE-specific use

**The lattice Gram operator (E-SPEC 14).** With tinfoil boundaries and k = 0 removed,

    U = (K_E/2)(4π/Ω) Σ_{k≠0} |ρ(k)|² / k²,
    ρ(k) = Σ_j e^{−ik·r_j} Σ_a exp(−k² σ_{a,j}²/4) c_a(k)·m_{a,j},   c_0 = 1, c_1 = −ik, c_2 = −½ kk

This is a sum of non-negative terms, so the theorem of E-SPEC 12 holds for every
crystal with the same floors. Gaussian multipole Ewald is ESTABLISHED (pGM: Wei et
al., J. Chem. Phys. 153, 114116 (2020)).

**The bond-current polarization (E-SPEC 15).**

    Ω P = Σ_i ρ_i r_i + Σ_b f_b τ_b d_b + Σ_i μ_i ,   d_b = the image-resolved bond vector
    F(E) = U(m) − E · Ω P                          (Nunes–Gonze electric enthalpy)

- Under open boundaries `f_b d_b = (Bᵀr)_b`, so this is exactly `Σ_i (q_i r_i + μ_i)`.
- Under periodic boundaries it is single valued, bounded by the bond hardness in
  every direction, and it sees divergence-free currents that no atomic charge
  records. Self-image bonds are kept: without them ε∞ of CsCl lost 12.8 % and
  depended on the choice of cell (E-SPEC 15).
- ESTABLISHED: SQE with κ > 0 is a dielectric, never a metal (Nistor and Müser,
  Phys. Rev. B 79, 104303 (2009)); the electric enthalpy (Nunes and Gonze,
  Phys. Rev. B 63, 155107 (2001); E-SPEC 15 names the enthalpy, the reference is
  added here). TRACE-specific: the SQE bond
  transfer as the time-integrated bond current, which makes the itinerant
  polarization a state function, together with the self-image bonds that carry it.

**Response observables (E-SPEC 15).**
- `ε∞ = I + 4π K_E α / Ω`.
- Born charges are `−∂²F/∂E∂r` (`born_charges_from_energy`); the acoustic sum rule
  holds to 1.1e-19.
- `ε0 = ε∞ + 4π K_E Zᵀ Φ⁺ Z / Ω`.
- The non-analytic (LO–TO) force constants come from the model's own Z* and ε∞.

**Routes.**
- The dense route.
- The matrix-free lattice route (E-SPEC 16) evaluates the same functional with no
  n × n operator: `K v` equals the dense lattice operator to 1e-13 relative.
- The matrix-free open route (E-SPEC 18).
- Particle-mesh Ewald for the reciprocal half (E-SPEC 19), ESTABLISHED: Essmann et
  al., J. Chem. Phys. 103, 8577 (1995); multipoles after Sagui, Pedersen, Darden,
  J. Chem. Phys. 120, 73 (2004).
- `solver: auto` chooses the route per structure. Every route is the same
  functional, differentiated as in E-SPEC 17.

**Batching (E-SPEC 24).** Independent structures are solved as one block-diagonal
system. Every pair block is multiplied by `1[s(i) = s(j)]`, and the per-structure
energy partitions U term by term:

    U_s = Σ_{i∈s} [ χ_i q_i + ½ η_i q_i² + ½ μ̂_i·A_μ μ̂_i + ½ q_i (Jq)_i
                   + q_i (T01 μ̂)_i + ½ μ̂_i·(T2 μ̂)_i (+ the Θ terms) ]
          + Σ_{b∈s} ½ κ_b t_b²                                         (24.1)

It is exact by construction. Periodic structures still go one by one.

### 3.7 Charge-transfer contact law (E-SPEC 23) — TRACE-specific prior, ESTABLISHED ingredients

    κ_b ← κ_b · exp( ln(K) sigmoid(x) + β w softplus(x) ),
    x = (r_b − R_i − R_j − δ) / w                                  (23.2)

- `R` are covalent radii (Cordero et al., Dalton Trans. 2008, 2832).
- K = 1e3 and δ = 0.5 Å. `β` (`charge_transfer_decay`) enables the law. K and β
  are fixed, not learned: learned, energies alone would drive them flat again.
- Motivation: `κ = Δ³ / (2h²)` (23.1), from the perturbative charge susceptibility
  `χ = 2h²/Δ³`. Water models trained on energies and forces alone learned a flat κ,
  and the polarizability per molecule of a 4-molecule cluster came out at
  6.13–6.27 Å³ against PBE0's 1.44 (E-SPEC 23).
- **In the released models:** `train.py` switches the law on for new runs at
  β = 3.0 /Å. The stored configuration of every released electrostatic checkpoint
  checked (step 34b, step 31, first-round and protocol NaCl) carries
  `charge_transfer_decay: 3.0`. The protocol runs set `w = 0.3` Å; the others use
  the default `w = 0.1` Å.

### 3.8 Pauli screening of the split-charge channel and the two-level anchor (E-SPEC 26) — TRACE-specific use

**Theorem 1 (bond-compliance identity).** If every atom is an inversion centre of
the crystal and of its bond star,

    ε∞ = ε∞^dip + (4π K_E / Ω) Σ_b p_b ⊗ p_b / κ_b,   p_b = f_b d_b.

In a bulk crystal, nothing but κ_b bounds ε∞. A trained Au₂/MgO model kept
second-shell Mg–Mg bonds at κ = 0.134 eV/e², and 1281 of its ε∞ = 1316 came from
those bonds (E-SPEC 26).

**26.1 Screening** (`bond_screening = (C_min, C_max)`, used at (2.0, 2.8)). This is
the MEAM / environment-dependent tight-binding factor (ESTABLISHED: Baskes; Tang,
Wang, Chan and Ho, Phys. Rev. B 53, 979 (1996)):

    C_ikj = y² / (x(1 − x)),   S_ij = Π_k q((C_ikj − C_min)/(C_max − C_min)),   f_b → f_b S_b

It is exactly 0 on the rock-salt and fcc second shells and exactly 1 on the fcc
first shell.

**26.2 Two-level anchor** (`bond_hardness_law = "two_level"`, range R = 10):

    κ = G³/(4h²),   G = sqrt(Δ² + 4h²)                       (one occupied, one empty orbital)
    κ_anchor = (Δχ² + 4h²)^{3/2} / (4h²),   Δχ = χ_i − χ_j (Mulliken, NIST),   h = 1.42 ħ²/(m_e r²)
    κ = κ_anchor · exp(L tanh(s/L)) + κ_floor,   L = ln R

- The 1.42 is Froyen and Harrison's universal sp-σ coefficient. The certificate is
  `κ ≥ κ_anchor/R` for every learned value.
- On rock-salt MgO with untrained heads (three seeds): ε∞ = 454–596 with the free
  law and no screening, and 10.6–12.6 with the two-level law and screening (E-SPEC 26).

**In the released Au₂/MgO models (steps 26, 31, 33, 34b), both are configured but
neither enters the energy.** R-SPEC §22 (bond partition, on in these runs) zeroes
the envelope of every bond that is not inside one covalent unit of more than three
atoms. With `carriers='atoms'` (as stored in the checkpoints), every atom is its
own unit, so every bond is zeroed. The bound sector is then "the atom-centred
moments alone" (R-SPEC §22; tested to 1e-12, `test_redox_bond_partition.py`).
Sections 3.2, 3.3, 3.7 and 3.8 therefore act in the elec, elecq and step-9
cluster models, not in steps 26–34b. (This paragraph is this document's reading of
the stored configurations against R-SPEC §22 and the code of
`RedoxElectrostatics._partition_bonds`.)

### 3.9 Terms built and tested but off in every released model

| term | E-SPEC | marking |
|---|---|---|
| quadrupoles Θ with `A_Θ` (needs `A_Θ`, else unbounded; truncated at `l_max = 2`) | 10, 10.1 | TRACE-specific wiring |
| dispersion as the zero-point energy of the same moment block, `E_disp = (1/2π) ∫_0^∞ dω [ln det(K_yy + ω²M) − ln det(D_yy + ω²M)]`, `M_i = ℓ_i² / (N_i ħ²/m_e)` | 13, 13.1 | ESTABLISHED: QDOs (Jones et al., Phys. Rev. B 87, 144103 (2013)); MBD (Tkatchenko et al., Phys. Rev. Lett. 108, 236402 (2012); J. Chem. Phys. 138, 074106 (2013)). TRACE-specific: the master equation's own PD Hessian as the operator; subtractive range separation |
| metals as a spanning star of zero-hardness bonds (1e-8 eV/e²), electrodes joined by circuit bonds carrying `−(V_e − V_ref) τ` | 20 | ESTABLISHED constant-potential electrode (Siepmann and Sprik, J. Chem. Phys. 102, 511 (1995); Reed, Lanning, Madden, J. Chem. Phys. 126, 084704 (2007)); TRACE-specific: the spanning star and the circuit bond |
| EVB charge states, `E = (E_1+E_2)/2 − sqrt(((E_1−E_2)/2)² + V²)` for two states | 21 | ESTABLISHED (Warshel and Weiss, J. Am. Chem. Soc. 102, 6218 (1980)) |
| through-bond coupling `V_DA = w_Dᵀ [H_DA + H_DB (Δ_B − H_BB)⁻¹ H_BA] w_A` | 22 | ESTABLISHED superexchange and pathways (McConnell, J. Chem. Phys. 35, 508 (1961); Beratan and Onuchic, Science 252, 1285 (1991)) |
| spin channels (`0.5 J^x s²`, `J^x > 0`) | 9 | stated as fitted, not derived (E-SPEC 9.2) |

In the free-charge sector (Section 4), integer charges are localised by the
configuration ensemble of R-SPEC, not by E-SPEC 21's declared states.

---

## 4. The free-charge sector (R-SPEC §0 and the sections it maps)

### 4.1 Why it exists (E-SPEC 25.0)

E-SPEC 1 is a complete theory of bound charge. For any region A, the net charge is
`Q_A = 1_Aᵀρ + Σ_{b∈∂A} ± f_b τ_b`, and `K ⪰ κ₀ I ≻ 0` is a gapped dielectric
operator, so a charge-neutral local perturbation cannot place net charge on a
distant region. On Au₂/MgO:Al, DFT puts +0.2…0.4 e out to 13 Å of the dopants,
balanced on Au₂, while the bound-sector model returns to zero by 5.5 Å. Table II
came out at −5 meV against DFT's −932 meV (E-SPEC 25.0). What is missing is free
charge: an electron-number degree of freedom with its own energetics.

ESTABLISHED: the bound-plus-integer decomposition `Q_i = n_i e + Σ_j q_ij` is
redoxSQE's (Dapp and Müser, J. Chem. Phys. 139, 064106 (2013)).

### 4.2 The model in three equations (R-SPEC §0)

A unit u (an atom; or a band unit, R-SPEC §20) carries an integer charge `n_u`
(electrons removed). It has a frame `f_u` (its formal oxidation state) and a
carrier `Q_u = n_u − f_u`. A configuration n has the energy

    E(n) = Σ_u ε_u(n_u)  +  ½ q(n)ᵀ J q(n)  −  ½ s(n)ᵀ K(n)⁻¹ s(n)  +  Σ_{u<v} O_uv(n_u, n_v)        (0.1)
              ions             Coulomb             induction                  overlap

The configurations mix through one-electron hops:

    H_nn = E(n),     H_nm = ± σ_nm f_nm τ_uv     (m = n with one electron moved u → v)            (0.2)

    E_total = E_short(R) + F,     F = −kT ln Tr exp(−H / kT)                                      (0.3)

**The terms of (0.1)–(0.3), as R-SPEC §0 defines them.**

- **Ions.** `ε_u(n) = E_Z(n) + g_u Δ_u(n) − ½ S₀,u n²`.
  - `E_Z` is the exact isolated ion: NIST ionisation energies and electron
    affinities (R-SPEC §17.5, §5a).
  - `Δ_u` is a learned shift, `|Δ| ≤ χ` (`Δ_u(n) = B tanh(r_u(n)/B)`,
    `B = chi_bound = 2.0` eV). It is gated to zero without neighbours by `g_u`
    (R-SPEC §24).
  - The isolated Gaussian self-energy `S₀` is removed, so the remaining terms add
    only interactions. Under periodic boundaries, an ion's interaction with its own
    images is lattice energy and stays in J.
- **Coulomb.** `q(n)` are the charges on Gaussians of width `σ_f` (Shannon crystal
  radii for tabulated elements, R-SPEC §17.5), and J is their Coulomb matrix with
  S₀ on the diagonal.
- **Induction.** `s(n) = g + Aᵀq(n) + Σ_u Q_u v_u(n_u)` is the field the
  configuration exerts on the bound sector. A holds the Gaussian charge–moment
  couplings, and `v_u` corrects them to the carrier's true density (R-SPEC §26).
  `K(n)` is the bound sector's stiffness with each state's own polarisability
  (R-SPEC §28), and `K(n) ≻ 0` is checked exactly.
  - The bound sector is eliminated exactly. At `K(n) = K` and `v = 0`, this term
    and the Coulomb term are `E_b0 + bᵀn + ½ nᵀWn` with `W = J − A K⁻¹ Aᵀ`.
  - The default condensed elimination (R-SPEC §13) never forms the dense factor of
    K in the variational basis.
- **Overlap.** `O_uv` is the two charge-state densities' exact Coulomb energy minus
  the Gaussian one (penetration, R-SPEC §23), plus their Gordon–Kim Pauli energy
  (R-SPEC §25), tabulated from first-principles ion densities. An atom and its own
  periodic images enter as one-body terms (R-SPEC §23.x).
- **Hops.** `τ_uv` are learned one-electron couplings (R-SPEC §17.6), magnitude-bounded
  by Harrison's matrix elements where configured (R-SPEC §27). `σ` is the importance
  switch (R-SPEC §21). `f = √2` for a move that pairs or unpairs frontier electrons
  (R-SPEC §29).

In R-SPEC's words: exact ions, the classical physics of their true densities, and
learned hops between integer configurations.

- ESTABLISHED: the Gordon–Kim overlap model (Gordon and Kim, J. Chem. Phys. 56,
  3122 (1972)) and polarisable-ion models (Madden and Wilson, Chem. Soc. Rev. 25,
  339 (1996)).
- TRACE-specific: evaluating them per integer configuration with first-principles
  charge-state densities, inside a canonical ensemble mixed by learned hops.
- Why the overlap terms cannot be dropped: classical point or Gaussian ions put
  A⁺A⁻ below A⁰A⁰ at the bond length of most covalent bonds (Proposition 8 of
  `docs/ELECTROSTATICS_THEORY_2026-10-18_overlap.md`, §4c).

**Tested as one equation** (`tests/redox/test_redox_master_equation.py`). The module
never forms (0.1); it holds the ladder, b and W, the pair slot and the Woodbury
correction. The test evaluates (0.1) directly, with a dense `K(n)⁻¹` and a dense
`s(n)`, and checks that the module's diagonal equals it to 1e-10 in an open cluster
and in rock-salt MgO. Leaving out v, K(n) or O breaks it by 0.51, 0.62 and 2.26 eV
(R-SPEC §0).

### 4.3 The additions of 2026-11-07, 2026-11-11 and 2026-11-13/14 (R-SPEC §0)

**As of 2026-11-07** three terms enter (0.1) and its induction field:

    E(n) = Σ_u ε_u(n_u) + ½ qᵀJq − ½ s(n)ᵀ K(n)⁻¹ s(n) + Σ_{u<v} O_uv + Σ_C λ_C(n) + E_bp(R)        (0.1′)
                                                                     fold (§32)   bond-oriented Pauli (§33)

    s(n) = g + Aᵀ q(n) + ΔAᵀ f + Σ_u Q_u v_u(n_u),   ΔA[i, μ_j] = (s(R_ij) − 1) A[i, μ_j]  (§34)

- **§33**: `E_bp = Σ_a Σ_j λ_a s_c(R)[D₁(R) cos θ + D₂(R) P₂(cos θ)]`, plus the wall
  `Σ s_c I(R)`. It is configuration independent.
- **§34**: the frame charges drive their nearest-neighbour counter-ions' induced
  dipoles with the overlap-damped coupling `s(R) = 0.61 + 0.37 (R − 2.128 Å)`
  (Mg²⁺–O²⁻, measured on the embedded ALMO pair).
- **§35**: in `K(n)`, the frame ions' dipole stiffness is
  `A_μ,i = K_E ℓ_i² / (g α_Z(f_i))`, with the first-principles `α_Z` and one learned
  crystal factor g.
- **§32**: `λ_C`, the downfolded O–s block. It is off in every released model.

**As of 2026-11-11** (extensivity), four changes:

    K(n):   Δ_j = A_μ,j (1/ρ_j(n_j) − 1)                                                   §28.6 (default)
    F     = E⁰ + Σ_c (F_c − E⁰) + Σ_{c<c′} (F_cc′ − F_c − F_c′ + E⁰)                      (0.3′)  §36
    E_M(C) = E_Γ(S_M[C]) / |M|                                                            (0.4)   §37
    E(n, N) = E_at(n) + F_b(N) + Σ_v ρ_N(v) c_v(n)                                        (0.1″)  §38

As written in R-SPEC §0, with all four on (frozen step 31 s43 weights,
`bvk_length = 18.0` Å), `E(2×2)/4 − E(1×1)` is +1e-6 / < 1e-6 meV (undoped
wetting / non-wetting) and +3e-6 / +7e-6 meV (doped). Before them, the doped 2×1
cell was off by +5.28 / +7.05 eV per cell. **§37 (0.4) is off in every released
model** (unit mesh). Its stated limits are in Section 8.

**As of 2026-11-13/14** (R-SPEC §39, §40, §40.7): the frame pair's own interaction
`Σ_pairs Φ_ab(R)` (39.1). For Au–Au, a first-principles two-site sector with entries
of the pair slot `O_ab` and the hop `τ_ab` (40.2)–(40.5). Section 4.4 gives the
equations.

### 4.4 Every section mapped to its term

The table follows R-SPEC §0's "Where every section sits", extended to every
section a released model uses. "TRACE" means the SPEC marks the section
NEW-TO-TRACE. The ESTABLISHED column gives the ingredients it cites.

| R-SPEC | term in (0.1)–(0.3) | equation (copied from the section) | ESTABLISHED ingredients | marking |
|---|---|---|---|---|
| §17.1, §17.4 | F over integer configurations (`fragment_ensemble='diabatic'`) | `H = diag(E_N) + T`, `F = −kT ln Tr e^{−H/kT}`; `dF/dH = ρ = e^{−H/kT}/Z`; second derivative by Daleckii–Krein, finite at degenerate spectra | Marcus–Hush two-state limit (named in §17.1) | TRACE |
| §17.2 | signs of `H_nm` | `⟨n′\| c†_q c_p \|n⟩ τ_ab` with Jordan–Wigner sign `(−1)^{N_<(p)} (−1)^{N′_<(q)}` | one-body Slater–Condon rule | TRACE |
| §17.5, §5a | ε: the ladder (`anchor='formal'`, `max_charge`) | `ê_u(n) = E_Z(n) + Δ_u(n) − ½ S_0(σ_u) n²`; `E_Z = Σ_{k≤n} IE_k` (n > 0), `−EA₁` (n = −1), `−EA₁ + A₂` (n = −2, O only: A₂ = 8.75 eV, a Born–Haber estimate) | NIST IE / EA; formal oxidation states (redoxSQE) | TRACE |
| §24 | g_u (`shift_gate`) | `g_u = 1 − Π_{edges into u}(1 − s(r))`; s = 1 for r ≤ 0.8 r_c, C² quintic to 0 at r_c (r_c = 4.0 Å) | — | TRACE |
| §17.6 | τ: learned couplings (`couplings`) | `τ_uv = Σ_{edges u→v} [Σ_k c_k(x_u, x_v) g_k(r)] f(r/r_cut)`, c from an MLP of `(x_u + x_v, x_u ⊙ x_v)`, so `τ_uv = τ_vu`; `r_cut = 4.0` Å | two-centre hopping integral (Slater–Koster) | TRACE (learned from TRACE invariants) |
| §21 | σ and the configuration set (`diabatic_selection='amplitude'`) | `r_c = max(√p_c, max_i r_i g(T_ic, x_c))`, `g(t, x) = \|t\|/√(x² + t²)`; `σ = smoothstep((ln e − ln a_off)/ln(a_on/a_off))`, a_off = 0.05, a_on = 0.2; more than 512 configurations is refused, never truncated | selected CI (Huron, Malrieu, Rancurel, J. Chem. Phys. 58, 5745 (1973)) | TRACE (its use to separate static mixing from the bound sector's virtual response) |
| §27 | bound on τ (`coupling_bound='harrison'`) | `B(r) = 3.24 ħ²/(m(r² + r₀²))`, r₀ = 0.5 Å; `τ = x` for `\|x\| ≤ a = 0.75 B`, else `sign(x)[a + (B − a) tanh((\|x\| − a)/(B − a))]` | Harrison's universal matrix elements (Harrison, *Electronic Structure and the Properties of Solids*, 1980) | TRACE |
| §29 | f = √2 on pairing moves (`spin_pairing`) | `f = √2` iff `N_d ≡ N_a (mod 2)` before the move; `H_ij = sign · σ(e_ij) · f · τ` | two-site Hubbard model | TRACE |
| §20 | the band unit (`carrier_band`, `band_form='phi1'`) | `h = diag(ε) − τ`, `ε_v = E(f − e_v) − E(f)`, `τ = V_ssσ(R) exp(ln 20 tanh s) · switch`, `V_ssσ = 1.40 ħ²/(mR²)`; `E_b(1) = λ₁`, `E_b(2) = 2λ₁ + J₁₁` | V_ssσ (Harrison) | TRACE |
| §20 | `band_initial_factor` | initial factor f of the hopping; PBE's MgO conduction-band mass (0.322–0.330 m_e) gives f = 0.55 through `m* = ħ²/(2 f V_ssσ a²)`; Harrison's f = 1 gives 0.179 m_e | effective-mass theory | TRACE |
| §38 | the Fermi band (`band_form='fermi'`): (0.1″) | `F_b(N) = A(N↑) + A(N↓) + ½ Σ_{u≠v} V_uv [ρ_u ρ_v − Σ_σ (P^σ_uv)²] + Σ_u U_u ρ↑_u ρ↓_u`, `A(n) = μ n − kT tr ln(1 + e^{−(h − μ)/kT})`, `P^σ = [1 + e^{(h − μ_σ)/kT}]⁻¹`; `E(n, N) = E_at(n) + F_b(N) + Σ_v ρ_N(v) c_v(n)` | Mermin (Phys. Rev. 137, A1441 (1965)); Chebyshev Fermi operator (Goedecker and Colombo, Phys. Rev. Lett. 73, 122 (1994)) | TRACE |
| §22 | the structure of K and A (`bond_partition`) | f_b → 0 on every bond not inside one covalent unit of more than `full_unit_atoms` atoms; with atom units, every bond | Clausius–Mossotti picture of ionic crystals (Tessman, Kahn, Shockley, Phys. Rev. 92, 890 (1953)) | TRACE |
| §23 | O_uv: penetration (`carrier_penetration`) | `Ξ_ab(n_a, n_b; R) = s(R){D_ab + (n_a n_b − f_a f_b)[K/R − J^G_ab(R)]}`, `D_ab = [E_es(n_a, n_b) − E_es(f_a, f_b)] − (n_a n_b − f_a f_b)K/R`; s a C² switch over [6.5, 8] Å | electrostatics of overlapping spherical ions (Gordon and Kim); Watson spheres (Phys. Rev. 111, 1108 (1958)); PBE0 (Adamo and Barone 1999) | TRACE |
| §25 | O_uv: Pauli (`carrier_overlap='gordon_kim'`, `pauli_table='first_principles'`) | `E_Pauli = K_{a←b}(δ_a\|ρ_b^f) + K_{b←a}(ρ_a^f\|δ_b) + K_cc(δ_a\|δ_b)`, K fitted to the electron-gas G at contact; with `'first_principles'` the AuMgO one-carrier K are fitted to frame-relative frozen-density PBE interactions instead | Gordon–Kim functional; density-overlap repulsion (Wheatley and Price, Mol. Phys. 69, 507 (1990); Söderhjelm et al. 2006) | TRACE |
| §26 | v_u in s(n) (`carrier_induction`) | `δE(n) = ΔFᵀ(y0 − M n) − ½ ΔFᵀ K⁻¹ ΔF`, `ΔF(n) = Σ_u Q_u v_u(s_u)`, split exactly into one-body and two-body terms | Gauss's law for the carrier's field | TRACE |
| §28 | K(n) (`state_polarizability`, elements (79, 8)) | `α_j(n) = ρ_j(n) α_j(f)`; `δE(n) = ½ y_S(n)ᵀ Δ_S (I + C_SS Δ_S)⁻¹ y_S(n)` (Woodbury); `K(n) > 0` checked by Haynsworth inertia additivity, refused otherwise | Woodbury; charge-state dependence of ionic polarisability | TRACE |
| §28.6 | K(n): which stiffness (`state_polarizability_stiffness`) | `'onsite'`: `Δ_j = A_μ,j (1/ρ_j(n) − 1)`; `'block'` (legacy): `Δ_j = P_jj(1/ρ_j − 1)`, which in a periodic cell also rescales the atom's interaction with its own images | as §28 | correction to TRACE's own §28 |
| §30 | the response priors' α, ε∞ (training only) | `α = α_b + α_ens`, `α_ens = −∂²F_ens/∂F²`, `F_ens(F) = −kT ln Tr exp(−[H − Σ_k F_k diag M_k]/kT)` | — | TRACE |
| §33 | E_bp (`bond_pauli`, level `'almo'`, `bond_pauli_wall`) | `E_bp(a, j) = λ_a s_c(R)[D₁(R) cos θ + D₂(R) P₂(cos θ)]`, `D_i = C_i f(R)^{p_i}`, `f(R) = E1(R)/E1(2.26 Å)`; almo level C₁, C₂ = −0.069, −0.497 eV, p₂ = 0.382, p₁ = 1.877; wall `E_wall = s_c(R) I(R)` | exchange ∝ S² (Murrell, Randić, Williams, Proc. R. Soc. A 284, 566 (1965)); anisotropic repulsion (Stone and Price, J. Phys. Chem. 92, 3325 (1988)); ALMO-EDA (Khaliullin et al., J. Phys. Chem. A 111, 8753 (2007)); Fritsch–Carlson interpolation | TRACE |
| §34 | ΔA in s(n) (`frame_induction_damping`) | `ΔA[i, μ_j] = (s(R_ij) − 1) K λ₃(R_ij) (r_i − r_j)_c / R_ij³ ℓ_j`; with `v = ΔAᵀf`: `y0 → y0 − K⁻¹v`, `u0 → u0 − A K⁻¹v`, `constant → constant + v·y0 − ½ v·K⁻¹v` | shell-model short-range coupling (Dick and Overhauser 1958; Sangster et al. 1970) | TRACE |
| §35, §35.1 | A_μ of frame ions (`frame_polarizability_anchor`, scope `'ions'`) | `A_μ,i = K_E ℓ_i² / (g_i α_Z(f_i))`, `g_i = g = exp(log_alpha_scale)` where `f_i ≠ 0`, 1 where `f_i = 0` | in-crystal reduction of ionic polarisability (Fowler and Madden, Phys. Rev. B 29, 1035 (1984), cited from memory in R-SPEC §35.1) | TRACE |
| §36, §36.7 | how F is taken: (0.3′) (`diabatic_space='clusters'`, `cluster_closure`) | `F = E⁰ + Σ_c (F_c − E⁰) + Σ_{c<c′} (F_cc′ − F_c − F_c′ + E⁰)`, with the product closure inside each cluster | method of increments (Stoll, Phys. Rev. B 46, 6700 (1992); Paulus, Phys. Rep. 428, 1 (2006)) | TRACE |
| §36.10 | the response record of several clusters | `α = α_b + Σ_X w_X α_ens(H_X, M_X, δα_X)`, `w_0 = 1 − n_c + n_p`, `w_c = 2 − n_c`, `w_cc′ = 1` | — | TRACE |
| §39 | `Σ_pairs Φ_ab` (`frame_pair`) | `Φ_ab(R) = s(R){D_ff(R) + f_a f_b [K/R − J^G_ab(R)] + c_ab(R) P_ab(R)}`, `P = T_TF^nad + E_x^nad`, `c_ab` = 1 or first-principles ratios at computed distances | Gordon–Kim; tight-binding total energy as band energy plus pair repulsion (Chadi, Phys. Rev. Lett. 41, 1062 (1978); Foulkes and Haydock, Phys. Rev. B 39, 12520 (1989), both cited from memory in R-SPEC §39) | TRACE |
| §40, §40.7 | O_ab and τ_ab of two s¹ frames (`two_site`; applied to Au–Au only) | `O_ab(f_a, f_b; R) = − s_ρ(R) ρ_ab(R)`, `ρ_ab = Φ_ab − E_cov,ab` (40.2); `τ_ab(R) = − t_ab(R) w(R)` for s moves (40.3); `O_ab(∓1, 0) = − s ρ_∓(R)` (40.4); `O_ab(+1, −1) = + s λ(R)` (40.5); from `E_cov = E_gs + E_T`, `U = E_S − E_T`, `t = √(E_T E_S)/2` (40.1) | the two-electron, two-orbital model, as for H₂ | TRACE |

The sections above that no released model uses are:
- §1–16, the mean-field functional with continuous free charges;
- §17.8, `bound_electronegativity='none'`;
- §18, exclusion, used by the step-9 cluster models only (legacy);
- §19, covalent units, used by the step-9 carbon-chain model only;
- §31, Harrison-anchored couplings;
- §32, the downfolded O–s bond;
- §33.3, the bond-oriented Pauli term per charge state;
- §37, Born–von Kármán sampling.

R-SPEC §0 calls the legacy paths (§17.9, §18, §18.5) "kept only to reproduce
recorded runs".

**The minimal model** is (0.1)–(0.3) with v = 0, K(n) = K and f = 1. R-SPEC §0's
rule: each of §22, §26, §28 and §29 stays only if its pre-registered step moves a
benchmark number.

### 4.5 The electronic response of the formal model (R-SPEC §17.7, §30)

An applied field E shifts each integer configuration by its free dipole and by the
bound response, with the same α_b for every n:

    E_n(E) = E_n(0) − d_n·E − ½ E·α_b·E
    −∂²F/∂E∂E = α_b + Var_p(d)/kT                          (no couplings)
    −∂²F/∂E∂E = α_b + Var_p(d)/kT + α_CT,   α_CT = 2 Σ_k |⟨0|d̂|k⟩|²/(E_k − E_0)   (kT → 0)

Before R-SPEC §30 the response priors therefore refused any structure whose window
held more than one configuration. §30 (and §36.10 for several clusters) supplies the
ensemble's exact field response, with §28's per-configuration α_b(n), and the priors
use it.

---

## 5. Which terms are on in which released model

Read from the `config.yaml` of each run and from the configuration stored in the
checkpoints (`kT = 0.1` eV, `diabatic_window = 2.5` eV, `coupling_cutoff = 4.0` Å,
`attention_cutoff = 4.0` Å, `band_cutoff = 3.5` Å, `fragment_max_configurations = 512`,
`carriers = 'atoms'`, `bound_electronegativity = 'learned'`, the same in both).
"block" / "onsite" is §28.6; checkpoints written before the field existed load as
`'block'`.

| term | E-SPEC / R-SPEC | step 31 (`…_pfp_bopw_e1kw_bondlaw_r5_s{42,43}`) | step 34b (`AuMgO_twosite_bf055_[e1kw_]bondlaw_r5_s{42,43}`) |
|---|---|---|---|
| Gaussian kernel, Gram floors, converged implicit solve | E 12, 14, 17 | on | on |
| induced dipoles (no quadrupoles, no dispersion) | E 1.5.1 | on | on |
| SQE bond charge flow (envelope, contact law, screening, two-level law) | E 1.3–1.4.1, 23, 26 | configured; every bond zeroed by R §22 | configured; every bond zeroed by R §22 |
| diabatic ensemble, formal anchors, max_charge 3 | R §17.1–17.5, §5a | on | on |
| shift gate | R §24 | on | on |
| learned couplings | R §17.6 | on | on (Au–Au s moves replaced by §40.3) |
| amplitude selection | R §21 | on | on |
| coupling bound | R §27 | off | on (`'harrison'`) |
| spin pairing | R §29 | off | on |
| band unit, initial factor | R §20 | on, `'phi1'`, factor 1.0 | on |
| Fermi band | R §38 | off | on (`'fermi'`, initial factor 0.55) |
| bond partition | R §22 | on | on |
| penetration, Gordon–Kim with the first-principles Pauli table | R §23, §25 | on | on |
| carrier induction | R §26 | on | on |
| state polarisability, elements (79, 8) | R §28 | on, `'block'` (legacy) | on, `'onsite'` (§28.6) |
| bond-oriented Pauli, almo level, with wall | R §33 | on | on |
| frame-induction damping | R §34 | off | on |
| frame-polarizability anchor, scope `'ions'` | R §35, §35.1 | off | on |
| cluster product ensemble with closure | R §36, §36.7 | off (`'global'`) | on |
| frame pair's own interaction | R §39 | off | on |
| two-site sector of two s¹ frames (Au–Au) | R §40, §40.7 | off | on |
| Born–von Kármán sampling, downfold, anchored couplings, per-state bond Pauli, exclusion | R §37, §32, §31, §33.3, §18 | off | off |

- **Step 26** (step 31's cold-start parent) has step 31's terms. Step 31 differs
  from it in the training recipe only (Section 6.5).
- **Step 33** (`AuMgO_single_all_[e1kw_]bondlaw_r5_s{42,43}`) has step 34b's
  terms except §29, §39, §40 and the Fermi band's initial factor (default 1.0).
- **Step 34b** differs from step 34 (stopped in epoch 2 and not released as a
  model) in one key, `band_initial_factor: 0.55`.

**The cluster models** (4G Ag₃, NaCl, carbon chain):
- "base" / "local": no electrostatics.
- "elec": the bound sector of Section 3 with default heads plus the contact law.
- "elecq": "elec" plus a loss on Hirshfeld partial charges (weight 1.0).
- "step9": formal anchors, the diabatic ensemble, learned couplings and §18
  exclusion, with the bond channel active (no §22), screening and the two-level law.
- "step9cu": "step9" plus §19 covalent units.

[`MODELS.md`](MODELS.md) lists every released checkpoint.

---

## 6. Training methodology

### 6.1 Data

- **4G-HDNNP sets** (Ko, Finkler, Goedecker, Behler, Nat. Commun. 12, 398 (2021);
  Materials Cloud record 2020.137, DOI 10.24435/materialscloud:f3-yh, CC BY 4.0;
  FHI-aims, PBE, "light" settings).
  - The data are not redistributed. `experiments/benchmarks_4g/prepare_all.sh`
    rebuilds every split from the raw archive (`$TRACE_4G_RAW`) with `prepare.py`
    (seed 0), and `experiments/benchmarks_4g/verify_data.py` checks each result
    against `DATA_SHA256.txt`.
  - Splits: 1000 / 250 / 1000 for Ag_cluster, Carbon_chain and NaCl in the first
    round. Au₂/MgO is 300 / 76 / 400 (train / valid / test). The PFN-size protocol
    uses 8000 / 1000 for Ag₃ and the chain, and 4000 / 500 for NaCl and Au₂/MgO,
    with the remainder as test. This is the 4G data split at PFN's sizes, not PFN's
    own split.
  - Hirshfeld charges are used for comparison only, except by the "elecq" models.
- **Bulk MgO phonon cells** (A. Togo's phono3py PBE set for rock-salt MgO, NIMS MDR,
  CC BY 4.0; VASP PBE, a = 4.25566 Å). Of the 148 displaced supercells, the two
  512-atom fc2 cells are the reference's own force constants and are held out. The
  146 fc3 pair cells of 64 atoms are forces only (`has_energy = F`).
- **The training set of steps 33, 34 and 34b** (`AuMgO_bulkaugw_*`): the 300 / 76
  4G slabs plus 136 / 10 PBE cells (train / valid). Every PBE cell carries
  `force_weight = 122.4`, the measured ratio of the mean squared force components
  of the two sources (slabs 0.4185, bulk cells 0.0378 eV/Å RMS). At equal relative
  error, the bulk cells' share of the force loss goes from 0.37 % to 31 %. It is
  built by `experiments/benchmarks_phonons/make_bulk_augmented_data.py --force-weight 122.4`.
  Steps 26 and 31 train on the 4G slabs alone.

### 6.2 Losses and weights

Per structure s with N_s atoms (`train.py`, `energy_loss`, `force_loss`):

    L_E = Σ_s has_s ((E_s − t_s) / N_s)²
    L_F = Σ_s w_s |F_s − F_s^ref|² / (3 N_s)
    L   = w_E L_E + w_F L_F  [+ electrostatic observable terms]   per optimizer step, divided by
                                                                   (structures per batch × grad_accum_steps)

- **Electrostatic observables** (`_electrostatic_loss`). An extensive observable is
  divided by N_s before squaring, as the energy is. A per-atom or intensive one
  (Born charges, ε, partial charges) is averaged over its components. Only the
  "elecq" models use one: partial charges at weight 1.0.
- **Weights in the released Au₂/MgO models.**
  - Stage A / step 26: `energy_weight 10`.
  - Stage B / step 31: `energy_weight 1000`.
  - `forces_weight 1.0` throughout (the `train.py` default is 10).
- **Weights in the cluster models.** First round: `energy_weight 1.0`,
  `forces_weight 1.0`. PFN-size protocol: 10 and 1.

### 6.3 Response priors (`train.py` `_response_prior_loss`; R-SPEC §17.7, §30)

Structures with a response label and no energy or force labels. They are evaluated
once per optimizer step on a forward pass of their own, and they enter no energy,
force or validation loss:

    L_prior = mean_s mean_ij (R_ij − R_ref,ij)²,   added as weight × L_prior

Here R is ε∞ (dimensionless) or α (Å³), taken from the bound state plus, when the
structure has more than one configuration, the ensemble record of R-SPEC §30 / §36.10:
`ε = ε_b + 4π K_E α_ens/Ω`, `α = α_b + K_E α_ens`.

| prior | file | label | weight | used by |
|---|---|---|---|---|
| ε∞ of bulk MgO (8-atom cell, a = 4.264985 Å) | `experiments/benchmarks_4g/data/MgO_bulk_eps_prior.xyz` (`make_mgo_eps_prior.py`) | 2.95, experiment (Sun, Allen, Stahnke, Jacobsen and Homes, arXiv:0704.2267, Table I) | 0.05 | steps 26, 31 |
| ε∞ of bulk MgO (8-atom cell, a = 4.255657 Å, PBE lattice constant) | `experiments/benchmarks_phonons/data/MgO_bulk_eps_prior_pbe.xyz` (`make_mgo_eps_prior_dft.py`) | 3.235, PBE (Togo's set, LEPSILON) | 0.05 | steps 33, 34b |
| α of the isolated Au atom | `experiments/benchmarks_4g/data/Au_atom_alpha_prior.xyz` (`make_au_alpha_prior.py`) | 36.68 a.u. = 5.4354 Å³, relativistic CCSD(T) (arXiv:2410.01493) | 0.3 | steps 26, 31, 33, 34b |

With SPEC 35.1 on (steps 33, 34b), the Au atom is a neutral frame. Its α is the
table's 5.4085 Å³ at every g, and the Au prior's gradient on g is exactly zero
(R-SPEC §35.1). This document infers, and no record measures, that the Au prior then
contributes a constant to the loss in those runs.

The step-9 cluster models use NaCl ε∞ and Ag-atom α priors
(`make_nacl_eps_prior.py`, `make_ag_alpha_prior.py`).

### 6.4 Model and optimiser

These are the same in every released Au₂/MgO run. Each `config.yaml` is in
`experiments/benchmarks_4g/configs/`.

- **Model.** hidden_dim 64, l_max 2, num_layers 1, num_radial 12, r_max 5.29 Å,
  correlation order 4 with 16 channels, 2 attention heads, attention dropout 0.03.
- **Batches.** batch_size 1 × grad_accum_steps 8 (one optimizer step per 8
  structures), with batched graphs.
- **Optimiser.** muon, weight decay 1e-5, gradient clipping at norm 1.0, cosine
  learning rate.
- **Precision and runtime.** float64 throughout, `use_amp: false`. The
  `amp_dtype` and `sobolev_sigma` keys are inert: their switches are off.
- **Seeds.** 42 and 43 (`torch.manual_seed` and `np.random.seed` at the start of
  `train.py`), with `batch_seeded_rng: true`.
- **Released checkpoint.** `model.pt` is the best checkpoint by validation loss. The
  validation loss uses the same per-structure force weights as training.

### 6.5 The two-stage recipe

| stage | runs | start | epochs | learning rate (cosine) | energy weight | anneals |
|---|---|---|---|---|---|---|
| step 26 | `…_pfp_bopw_bondlaw_r5_s{42,43}` | cold | 40 | 1e-3 → 5e-5 | 10 | kT 0.5 → 0.1 eV over 20 epochs; attention temperature 1.3 → 1.0 over 15 |
| step 31 | `…_pfp_bopw_e1kw_bondlaw_r5_s{42,43}` | step 26's best weights, epoch 0, no optimizer state | 20 | 1e-4 → 5e-6 | 1000 | off |
| step 34b A | `AuMgO_twosite_bf055_bondlaw_r5_s{42,43}` | cold | 40 | 1e-3 → 5e-5 | 10 | off |
| step 34b B | `AuMgO_twosite_bf055_e1kw_bondlaw_r5_s{42,43}` | stage A's best weights | 20 | 1e-4 → 5e-6 | 1000 | off |

- **The kT anneal** (`train.py` `redox_kT`): `kT(e) = end (start/end)^(1 − e/epochs)`,
  exactly `start` at e = 0 and the deployed kT from `e = epochs` on. It is a
  continuation heuristic: a configuration Δ above the populated one weighs
  `exp(−Δ/kT)`, so training hotter lets its energy receive a gradient.
- **Why two stages.** Step 31 was pre-registered as the energy weight applied from a
  warm start. Step 30, the same weight from a cold start, was stopped at epoch 21 (s42) / 20 (s43)
  of 40 (decision of 2026-11-09), with force errors 3–6× those of step 26. Step 29's
  control (s43, energy weight 10, the same warm start, ten more epochs) moved the class
  offsets and wet_d / wet_u by 8–12 meV, so step 31's changes against step 26 are
  attributed to the weight.
- **Warm starts** are weights only, at epoch 0, with the same configuration
  (`experiments/benchmarks_4g/make_warm_start.py`).
  - Step 33's warm start from step 31 also reset `electrostatics.log_alpha_scale`
    to 0.0 (`--reset-log-alpha-scale`).
  - **Steps 26 and 31 trained R-SPEC §28's legacy `'block'` stiffness.** A
    retraining from their YAML needs `state_polarizability_stiffness: block`, since
    the current code resolves `'onsite'` when the key is absent. Evaluation is
    unaffected: the loader restores `'block'` from the checkpoint.
- **Compute.**
  - Step 26: the Mac CPU, 19.2 / 21.8 min per epoch (median, s42 / s43).
  - Step 31: the Mac CPU, 29.1 / 28.9 min per epoch.
  - Step 34b: one RTX 5060 Ti (16 GB) holding both seeds, stage A 19.8 / 19.9 and
    stage B 20.2 / 20.0 min per epoch.
  - Best epochs: step 31 s42 20 and s43 17; step 34b A 39 / 39 and B 15 / 18.

### 6.6 Pre-registration protocol

Every step in the notebooks follows the same order:

1. **The criteria are written before the run.** The entry gives the run names, the
   one change against the comparator, the benchmark numbers it is meant to move,
   numbered pass criteria for each seed, and the predictions, labelled as hypotheses.
   Example: step 31's Y1–Y6, written 2026-11-09 10:01 IST, launched 10:02:02.
2. **One change per run**, against a named comparator. A run that breaks this says
   so in its pre-registration. Step 34 switched on `frame_pair`, `spin_pairing`
   and `two_site` together, with a cold start and no control run (decision of
   2026-11-15). Its entry states that a failure would not say which of them failed.
3. **Both seeds.** PASS requires every criterion on both seeds.
   - Step 31 failed as pre-registered: s43 passed Y1–Y5, and s42 failed Y1 and Y3.
   - Step 34b failed as pre-registered: S2, S5 and S8 failed on both seeds.
4. **Pre-flight before an expensive launch.**
   - A one-epoch cold-start smoke run on CPU for each seed.
   - CPU against GPU energies and forces.
   - A few training steps on the GPU, with peak memory recorded.
   - An independent check of the configs (`check_run_config.py`).
5. **Scored as written.** A later correction is added below the original text,
   marked with its date, and the original text is kept. Nothing is rescored under
   criteria written afterwards. Example: step 34b's S4 is read on 2×2×2 as
   registered, and the 512-atom numbers are recorded beside it (Section 7).
6. **Artifacts are ruled out before a result is claimed.** Candidate artifacts are
   named and the falsifying check is run first. An unchecked claim is labelled a
   hypothesis.

### 6.7 What the pre-registered steps score

- **Step 31.**
  - Y1: training-set undoped class offset |·| ≤ 30 meV.
  - Y2: wet_u within ±100 meV of DFT's +928.9.
  - Y3: wet_d within ±150 meV of −2.7.
  - Y4: T5 within ±15 % of −931.6.
  - Y5: test force RMSE per subset not above step 26's by more than 10 %.
  - Y6: reported, not scored.
- **Step 34b** (step 34's criteria, unchanged).
  - S1: both stages finish.
  - S2: Table II by the T5 protocol.
  - S3: 4G test force RMSE ≤ 0.10 (undoped) and ≤ 0.13 eV/Å (doped).
  - S4: phonons against PBE. P0, P1, P2 and P4 pass, and at least 6 of 7.
  - S5: crystal factor g within 0.85–0.95, and the prior cell's ε∞ within 0.05 of
    3.235.
  - S7: the Au–Au bond force on test 167, repulsive and monotone over 2.00–2.40 Å,
    and within ±30 % of DFT's +15.25 eV/Å at 2.089 Å.
  - S8: undoped test force RMSE without the five dimers below 2.18 Å, at most
    0.060 eV/Å.
  - S6: reported.

The scored numbers are in [`BENCHMARKS.md`](BENCHMARKS.md).

---

## 7. The phonon protocol (P0–P6) and the supercell question

**Protocol** (`experiments/benchmarks_phonons/score_phonons.py`, calling
`tools/electrostatics/mgo_phonons.py`):
- The model is evaluated at the PBE lattice constant, a = 4.255656884758435 Å.
- Z* and ε∞ come from the bound sector on the 64-atom cell (the 2×2×2 conventional
  cell): the frame charge plus ∂P_bound/∂r.
- Finite displacements of 0.01 Å on the 2×2×2 conventional supercell.
- The NAC term uses the model's own Z* and ε∞, with q → 0 at |q| = 10⁻³.
- An ensemble guard records the frame population of every displaced supercell.
- The reference is Togo's PBE set: Z* = ±1.9923, ε∞ = 3.235, TO 373.3, LO 681.3 cm⁻¹.
  Its harmonic bands were recomputed with phonopy 2.17.2 along Γ–X–K–Γ–L, with and
  without NAC. `experiments/benchmarks_phonons/make_reference_bands.sh` rebuilds them.

**Criteria, written 2026-11-06 before any run on this line:**

| | criterion |
|---|---|
| P0 | ensemble in the frame on every displaced supercell (population ≥ 1 − 1e-4) |
| P1 | Z*(Mg) within 1.90–2.10 |
| P2 | ε∞ within 2.9–3.5 |
| P3 | TO within 3 % of 373.3 cm⁻¹ |
| P4 | LO within 660–700 cm⁻¹ |
| P5 | path RMS without NAC ≤ 15 cm⁻¹ |
| P6 | path RMS with NAC ≤ 20 cm⁻¹ |

P1, P2 and P4 depend on the electrostatic sector alone. P3 and P5 test the
short-range force constants, and P6 tests both.

**Why the supercell matters: 2×2×2 against 4×4×4.**
- Togo's fc2 supercell is 4×4×4 of the conventional 8-atom cell, 512 atoms. Off the
  commensurate points, a band without NAC is the supercell's own Fourier
  interpolation, so P5 on 2×2×2 compares two different interpolations.
- Hypothesis H, pre-registered on 2026-11-16 before the run: P5's 29–38 cm⁻¹ against
  PBE is this supercell mismatch. Validity condition C1: the model's no-NAC bands
  from 2×2×2 and 4×4×4 agree within 1 cm⁻¹ at Γ, X and L, where both are exact.
- Result on step 34b stage A checkpoints:

  | checkpoint | C1 | P5, 2×2×2 → 4×4×4 (cm⁻¹) |
  |---|---|---|
  | s42 ep21 (pre-registered) | 0.004 cm⁻¹ | 30.4 → 11.5 |
  | s42 ep30 | 0.005 cm⁻¹ | 29.4 → 8.2 |
  | s43 ep29 | 0.001 cm⁻¹ | 29.7 → 9.0 |

  H holds.
- On 4×4×4, P6 rose by 1.1–2.2 cm⁻¹ on the five step 34b checkpoints measured.
- **The rule since then**, for steps registered after 2026-11-16 10:31 IST: P5 is
  read on the 4×4×4 conventional supercell, with C1 as its validity condition, via
  `score_phonons.py MODEL --p5-supercell 4`. P1–P4 and P6 stay on 2×2×2.
- Step 34b was registered before the rule, so its S4 is read on 2×2×2 and its
  4×4×4 numbers are recorded beside it. On 4×4×4, stage B passes 7/7 on both seeds
  (P5 8.1 / 8.6, P6 8.5 / 8.8 cm⁻¹).
- Step 33 was not scored on 4×4×4.
- Cost: about 61–62 s and 6.5–6.9 GB peak memory per model on 4×4×4 (Mac CPU,
  4 threads). Under a minute on 2×2×2.

---

## 8. Stated limits

These are the limits that the SPECs state for the terms the released models use.
Each SPEC section lists its own in full.

- **Bound sector.**
  - Dispersion has no charge flow: the bond transfers are clamped in E_disp
    (E-SPEC 13).
  - The preconditioner's iteration count grows as L² with cell size (E-SPEC 17.6);
    the fix is open.
  - Open boundaries are O(n²), with no FMM (E-SPEC 18).
  - Periodic structures are not batched (E-SPEC 24).
- **Free-charge sector.**
  - No spin, beyond §29's pairing factor. §29 is exact for s-frontier elements in
    their lowest-spin states only.
  - Rigid free-ion densities, with O²⁻ in a Watson sphere (§23, §25, §28, §39).
  - The selection switch of §21 is not variational, F is piecewise C², and inside
    the switch band the switch's field derivative is outside the response record
    (§21, §30).
  - (0.3′) drops increments of three or more clusters. Where the reference changes
    on three or more clusters, the energy steps by about 1e-6 eV (§36.8 L1, L1′).
    The closure's cost is exponential inside a merged cluster, and beyond 512
    configurations the structure is refused (§36.8 L2).
  - The band unit's couplings to other units are zero (§38).
  - §40 / §40.7 are applied to Au–Au only. Ag–Ag and H–H are tabulated and
    withheld. The Au–Au entries are tied to the option set and `gaussian_a` they
    were built with, and another set is refused (§40.7).
- **Extensive is not the same as trained** (R-SPEC §0). Weights trained at Γ on one
  cell do not fix how a supercell's band fills. At a (2, 2, 1) mesh, the two step 31
  seeds give wet_d = −22.95 and +603.69 meV. No doped number at a mesh is claimed.
- **Retraining is not bitwise.**
  - `cholesky_solve` gives 1-ulp differences.
  - GPU sums vary in the last bits.
  - Transient NaN steps are recomputed by the training wrapper.
  - The released checkpoints are the reproduction path for the reported numbers.

---

## 9. References

These are the citations the SPECs give, with author surnames as the SPECs write them.
"Cited from memory" is marked where the SPEC marks it. One reference is added here
and marked so.

- Adamo and Barone (1999), PBE0 (R-SPEC §23).
- Baskes, MEAM screening; Tang, Wang, Chan and Ho, Phys. Rev. B 53, 979 (1996).
- Beratan and Onuchic, Science 252, 1285 (1991).
- Chadi, Phys. Rev. Lett. 41, 1062 (1978) (cited from memory).
- Cordero et al., Dalton Trans. 2008, 2832.
- Dapp and Müser, J. Chem. Phys. 139, 064106 (2013) (redoxSQE).
- Dick and Overhauser (1958); Sangster et al. (1970) (shell model).
- Elking, Darden and Woods, J. Comput. Chem. 28, 1261 (2007).
- Essmann, Perera, Berkowitz, Darden, Lee and Pedersen, J. Chem. Phys. 103, 8577 (1995).
- Foulkes and Haydock, Phys. Rev. B 39, 12520 (1989) (cited from memory).
- Fowler and Madden, Phys. Rev. B 29, 1035 (1984) (cited from memory).
- Froyen and Harrison, the universal sp-σ coefficient 1.42 (E-SPEC 26.2).
- Goedecker and Colombo, Phys. Rev. Lett. 73, 122 (1994).
- Gordon and Kim, J. Chem. Phys. 56, 3122 (1972).
- Harrison, *Electronic Structure and the Properties of Solids* (1980).
- Huron, Malrieu and Rancurel, J. Chem. Phys. 58, 5745 (1973).
- Jones, Crain, Sokhan, Whitfield and Martyna, Phys. Rev. B 87, 144103 (2013).
- Khaliullin, Cobar, Lochan, Bell and Head-Gordon, J. Phys. Chem. A 111, 8753 (2007).
- Ko, Finkler, Goedecker and Behler, Nat. Commun. 12, 398 (2021); data: Materials Cloud
  record 2020.137, DOI 10.24435/materialscloud:f3-yh, CC BY 4.0.
- Madden and Wilson, Chem. Soc. Rev. 25, 339 (1996).
- Mathieu, J. Chem. Phys. 127, 224103 (2007).
- McConnell, J. Chem. Phys. 35, 508 (1961).
- Mermin, Phys. Rev. 137, A1441 (1965).
- Murrell, Randić and Williams, Proc. R. Soc. A 284, 566 (1965).
- Nistor and Müser, Phys. Rev. B 79, 104303 (2009).
- Nunes and Gonze, Phys. Rev. B 63, 155107 (2001) (added here; E-SPEC 15 names the
  enthalpy only).
- Parr and Pearson, J. Am. Chem. Soc. 105, 7512 (1983) (R-SPEC §19.2; not used by a
  released model).
- Paulus, Phys. Rep. 428, 1 (2006).
- Perdew, Parr, Levy and Balduz, Phys. Rev. Lett. 49, 1691 (1982).
- Reed, Lanning and Madden, J. Chem. Phys. 126, 084704 (2007).
- Sagui, Pedersen and Darden, J. Chem. Phys. 120, 73 (2004).
- Siepmann and Sprik, J. Chem. Phys. 102, 511 (1995).
- Söderhjelm et al. (2006).
- Stone and Price, J. Phys. Chem. 92, 3325 (1988).
- Stoll, Phys. Rev. B 46, 6700 (1992).
- Sun, Allen, Stahnke, Jacobsen and Homes, arXiv:0704.2267 (ε∞ of MgO, Table I, 295 K).
- Tessman, Kahn and Shockley, Phys. Rev. 92, 890 (1953).
- Togo, phono3py PBE input data for rock-salt MgO, NIMS MDR
  (https://mdr.nims.go.jp/datasets/1fd84240-abc0-4d8f-b4d7-1db304d6166b), CC BY 4.0.
- Tkatchenko, DiStasio, Car and Scheffler, Phys. Rev. Lett. 108, 236402 (2012);
  Tkatchenko, Ambrosetti and DiStasio, J. Chem. Phys. 138, 074106 (2013).
- Warshel and Weiss, J. Am. Chem. Soc. 102, 6218 (1980).
- Watson, Phys. Rev. 111, 1108 (1958).
- Wei et al., J. Chem. Phys. 153, 114116 (2020) (pGM).
- Wheatley and Price, Mol. Phys. 69, 507 (1990).
- The Au atom's polarizability, relativistic CCSD(T): arXiv:2410.01493.
