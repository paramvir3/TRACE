# Electron transfer needs the band-structure term: derivation (2026-10-09)

Written after both Au₂/MgO redox arms failed their electron-transfer tests (T2,
T4, T5). Every statement is derived below or checked numerically, as stated.
**Nothing here is implemented yet.**

## 0. What the failures say, measured

1. **Au₂ and the slab are one molecule.** In both trained models, Au₂ is bonded
   to the slab by the redox bond rule (partition weight ≈ 1, one 110-atom
   molecule). So the free sector runs in its grand-canonical regime: free
   charge is shared fractionally at one Fermi level. No integer barrier blocks
   Au₂.
2. **The electron still stops on O** (T4: free Al +2.995, O −2.996, Au₂
   0.000), because O's free-electron acceptor level is the lowest. The ladder
   anchors O at the neutral atom (EA = 1.46 eV) and screening deepens it. In
   MgO, O²⁻ has a full 2p shell and is no acceptor at all: the extra electron
   belongs in the Mg-derived conduction band or on Au₂.
   - Neither TRACE sector can represent this. Shell filling is not a function
     of a partial charge: Hirshfeld O is −0.36 e while its 2p shell is full.
3. **The charge loss was absorbed by the bound sector** (Al bound −1.744 e
   against free +2.996). Only q_b + n is observed, so the split is not pinned
   by data.

## 1. The expansion of DFT that TRACE truncates (ESTABLISHED)

Expand the Kohn–Sham (Mermin) energy around the reference ρ⁰, the sum of
neutral-atom densities (Foulkes & Haydock, PRB 39, 12520 (1989); Elstner et
al., PRB 58, 7260 (1998)). In a minimal orthogonalised valence basis:

    E[ρ⁰ + δρ] = E_bs + E_rep + E_2 + O(δρ³)
    E_bs = Tr[H⁰ P] − kT S(P),   P = f(H) (Fermi–Dirac),   S(P) = −Tr[P ln P + (1 − P) ln(1 − P)]
    E_2  = ½ Σ_AB γ_AB Δq_A Δq_B,   Δq_A = Z_A^val − Σ_{μ∈A} P_μμ,   γ_AA ≈ IE_A − EA_A,   γ_AB → 1/R_AB

- E_bs, the **band-structure energy**, carries Pauli filling, hybridisation,
  delocalisation and the Fermi level.
- E_2 is the second-order Coulomb and hardness term.
- E_rep, the short-range repulsion, is what TRACE's ACE backbone learns.

## 2. TRACE's sectors are two limits of E_bs (NEW-TO-TRACE theorems)

**Theorem 5 (the redox ladder is the atomic limit).**
- Statement: let H⁰ have no inter-atomic hopping, with one frontier donor
  orbital ε_d and one acceptor orbital ε_a per atom. Then E_bs at integer
  filling gives Ê_A(±1) = ∓χ_A + G_A/2, with χ_A = −(ε_a + ε_d)/2 and
  G_A = ε_a − ε_d. That is exactly the TRACE-Redox ladder.
- At finite kT, the two-orbital Mermin grand potential differs from the
  three-state ladder by −kT ln(1 + AB/(1 + A + B)), with
  A = e^{−(ε_a+ν)/kT} and B = e^{(ε_d+ν)/kT}. Its magnitude is
  ≤ kT e^{−G/kT}.
- Checked numerically (G = 5, 2, 0.8 eV): −1.93e-23, −3.71e-12, −2.25e-07 eV,
  each within its bound.

**Theorem 6 (SQE is E_bs to second order in the hopping).**
- Statement: for a gapped reference with occupied orbitals on A and empty
  ones on B, coupled by hopping h, expanding E_bs to second order in h and in
  the transferred charge t gives ½ κ_b t², with

      κ_b⁻¹ = Σ_{i occ on A, j empty on B} 4|h_ij|² / (ε_j − ε_i)³   (+ A ↔ B).

- Proof: for one pair, N_B = 1 − Δ/G with G = √(Δ² + 4h²), so
  dN_B/dΔ = −4h²/G³.
- Checked: κ = 43.130849 by finite difference, against G³/(4h²) = 43.130849.
- Corollary (Pauli): with no occupied–empty pair across a bond (two full
  anions, two empty cations), κ_b = ∞.

**Consequence.**
- SQE is the harmonic expansion of E_bs about an insulating reference. By
  construction it cannot describe integer or long-range transfer.
- The ladder is E_bs with the hopping switched off, and with filling
  referenced to neutral atoms. It cannot know that O's shell is full in an
  oxide.
- **Electron transfer lives in E_bs itself, which TRACE does not have.**

**Theorem 7 (algebraic, not exponential, training signal).**
- Statement: in E_bs, an empty acceptor orbital coupled by t to the occupied
  manifold at an energy distance ΔE has occupation
  p = ½(1 − ΔE/√(ΔE² + 4t²)) = t²/ΔE² + O(t⁴). So every loss gradient
  through its level is O(t²/ΔE³).
- In the atomic-limit ensemble (Theorem 3) the same gradient is e^{−ΔE/kT}.
- Checked at t = 1.3 eV: p = 0.195 / 0.081 / 0.024 for ΔE = 2 / 4 / 8 eV,
  against Boltzmann factors of 2e-9 / 4e-18 / 2e-35.

**Theorem 8 (convexity: one SCC solution).** The functional

    F(P) = Tr[H⁰P] − kT S(P) + bᵀΔq(P) + ½ Δq(P)ᵀ W Δq(P)

is strictly convex on {0 ⪯ P ⪯ I, Tr P = N} for W ⪰ 0 and kT > 0. It is a
linear term, plus a strictly convex entropy term, plus a convex quadratic of
a linear map. So the self-consistent-charge solution is unique. Its concave
dual in the Hartree potential has exactly the structure of TRACE-Redox's
(z, μ) dual, with the diagonal susceptibility replaced by the full
non-interacting response matrix. (ESTABLISHED in essence: the Mermin
functional.)

**Theorem 9 (exact derivatives at any degeneracy).**

    dP = V[(Vᵀ dH V) ∘ F]Vᵀ,   F_ij = (f(λ_i) − f(λ_j))/(λ_i − λ_j),   F_ii = f′(λ_i)

This is finite at every degeneracy, unlike eigenvector derivatives: the
2026-10-08 NaN class. (ESTABLISHED: Daleckii & Krein (1965).)

## 3. What this predicts for Au₂/MgO (a hypothesis until built and tested)

Use a minimal valence basis (Mg 3s 3p, O 2s 2p, Al 3s 3p, Au 6s), with all
valence electrons and on-site energies anchored to atomic term values:
- the O 2p band is filled (Pauli);
- Al's third valence electron enters the lowest empty states: the Mg-derived
  conduction band, or Au₂'s σ* if it lies lower;
- Fermi–Dirac filling with self-consistent charges then gives fractional
  transfer, as PBE does.

This is the physics that 4G imitates with a metallic QEq. Here it comes with
the gap intact: ε∞ stays finite, and bond polarisation stays the Theorem 6
limit of the same E_bs.

## 4. Construction (proposed, test-driven)

**Stage 1 — the band-energy kernel.** Pure maths, no parameters:
- P = f(H) with Theorem 9 derivatives (a custom autograd function);
- E_bs, the Mulliken charges, and the SCC dual solved by the existing Newton
  scheme with matrix susceptibility.
- Tests, written first:
  - the two-level exact formulas;
  - the Hückel chain spectrum;
  - Theorem 5 to e^{−G/kT};
  - Theorem 6 (κ = G³/4h² by finite difference);
  - degenerate spectra giving finite, exact gradients;
  - forces = −dE/dR by finite difference;
  - charge conservation.

**Stage 2 — the parameters.**
- Minimal basis per element, with on-site energies from published
  atomic-orbital term values (the source to be agreed; no invented numbers).
- Slater–Koster hopping with Harrison's universal r⁻² scaling (Froyen &
  Harrison 1979) and an exponential nearsightedness envelope.
- Bounded, learned, environment-dependent corrections from TRACE's heads.

**Stage 3 — the energy.**
- E = E_ACE + E_bs + E_2, with E_2 from TRACE's existing Gaussian Coulomb
  kernel and screening, plus the induced dipoles (polarisation beyond the
  minimal basis).
- Trained on the four 4G/PFN benchmarks and scored by their unchanged tests,
  with T1–T6 for Au₂/MgO.

## 5. Honest risks

- **Parameters:** which tables, and whether energies and forces pin the
  hopping corrections. Anchoring is meant to give the right physics at
  initialisation.
- **Cost:** O(N_orb³) per structure. That is fine for the benchmarks (110
  atoms → about 400 orbitals); large systems need a linear-scaling Fermi
  operator.
- **Scope:** this is a new energy term, not a patch, and replaces the redox
  ladder for these systems.

## 6. The TRACE-compatible form: coherent free carriers inside the existing dual

The full E_bs (DFTB) solves all valence electrons, too slow for a foundation
model. What TRACE-Redox needs is narrower: let its **free carriers** hop.
- **What stays:** the bound sector (SQE + dipoles), its exact elimination
  (W, b) and the convex (z, μ) dual of trace_redox all stay as they are.
- **What changes:** only the per-unit grand potential Σ_u Ω_u(ν_u) is
  replaced.
- **Established ingredients:**
  - the excess-carrier tight-binding Hamiltonian of charge-transport theory:
    Holstein, Ann. Phys. 8, 325 (1959); Spencer, Gajdos & Blumberger, JCP 145,
    064102 (2016), FOB-SH;
  - the Mermin grand potential (Phys. Rev. 137, A1441 (1965));
  - Daleckii–Krein derivatives;
  - the Fermi-operator expansion for linear scaling (Goedecker, RMP 71, 1085
    (1999)).
- **NEW-TO-TRACE:** putting that Hamiltonian into a variational polarizable
  force field with exact bound-sector screening, together with properties
  (i)–(iv) below.

**Equations.**
- One acceptor orbital a_u and one donor orbital d_u per unit:

      ε^a_u = Ê_u(−1) = −χ_u + G_u/2,   ε^d_u = −Ê_u(+1) = −χ_u − G_u/2   (the existing ladder)
      H^a = diag(ε^a) + T^a,   H^d = diag(ε^d) + T^d     (T symmetric, zero diagonal: hopping)

- Grand potential:

      Ω(ν) = −kT Tr ln(1 + e^{−(H^a + ν̂)/kT}) − kT Tr ln(1 + e^{(H^d + ν̂)/kT})

- Free charge: c_u = −∂Ω/∂ν_u = [1 − f(H^d + ν̂)]_uu − [f(H^a + ν̂)]_uu.
- Susceptibility: X = −∂²Ω/∂ν∂ν ⪰ 0 (Daleckii–Krein form).
- The dual, unchanged:

      D(z, μ) = −½ zᵀWz + μᵀN + Ω(Gᵀμ − b − Wz)

  Its Newton step is the present one with diag(x) → X:
  A = W + WXW, B = WXGᵀ, S = GXGᵀ − BᵀA⁻¹B.

**Properties.**
- (i) **Concave dual, unique solution.** Ω is a minimum of functions affine in
  ν (Mermin), hence concave.
- (ii) **Exact reduction.** With T = 0, Ω is the current TRACE-Redox
  grand potential within kT e^{−G/kT} per unit (§2, Theorem 5, checked). So
  the present model is the special case "no hopping".
- (iii) **Algebraic training signal** (Theorem 7). An empty acceptor coupled
  by t has occupation t²/ΔE², so the loss reaches every level. This removes
  the measured 10⁻⁸ blockage (Theorem 3).
- (iv) **Exact derivatives at degeneracy** (Theorem 9).
- **Cost:** two U×U eigendecompositions per Newton step, U = number of units.
  That is trivial at 110 atoms; large systems use a Chebyshev Fermi-operator
  expansion, O(N).

**Hopping.** t_uv = −τ_uv · e^{−(r_uv − r⁰_uv)/λ} between bonded units, with
r⁰ the covalent contact distance. τ_uv ≥ 0 is learned (softplus,
environment-dependent through TRACE's heads). Its initialisation is to be
fixed by test P4.

**What it does and does not fix (measured causes, 2026-10-08/09).**

| measured cause | fixed by coherent free carriers? |
|---|---|
| Training cannot reach the Au₂ level (gradient ~10⁻⁸) | **yes**: algebraic signal |
| Delocalised carriers (C₁₀H₃⁺ hole, CB electron, hemibond within a molecule) | **yes** |
| The bound sector is local and cannot carry the shift (elec + Hirshfeld: +0.013 e, r = −0.42) | not needed: the free sector carries it at one Fermi level |
| No closed-shell rule (O accepts) | not directly. With the signal restored, the data can move the acceptor; formal states need ionic-radius free charges first (P3) |
| Free/bound local compensation (Al bound −1.74 e) | not directly |

**Pre-registered sequence.**
- **P4 (frozen, before any code in the module):** add a small uniform hopping
  between bonded units of the trained charge-fitting model. Prediction: the
  derivative of the Au₂ free charge with respect to Au's acceptor level rises
  from ~10⁻⁸ to ≥ 10⁻³ e/eV at t = 0.3 eV.
- **Unit tests, written first:**
  - T = 0 equals the current module to kT e^{−G/kT};
  - the dual identities (c = −∂Ω/∂ν; X ⪰ 0);
  - finite-difference derivatives;
  - finite gradients at a degenerate spectrum;
  - the two-site exact result.
- **Training:** the charge-fitting protocol plus coherent free carriers (one
  change), scored by T1–T6.

**Correction (2026-10-09, after P4).** The "~10⁻⁸" for the training signal
without hopping was an estimate. Measured (P4, three doped structures):
dq(Au₂)/dε^a_Au = 1.2e-10 to 3.5e-6 e/eV at t = 0; 2.8e-4 to 2.9e-2 at
t = 0.3 eV. The pre-registered P4 criterion (≥ 1e-3 in every structure)
failed in one of three.

## 7. Where the C2 Hamiltonian sits, and what a band of donor electrons needs (2026-10-09, late; nothing built)

**Identity.** With couplings, the C2 Hamiltonian of trace_redox (SPEC §17) is a Pariser–Parr–Pople
(PPP) Hamiltonian in second quantisation, restricted to the integer configurations inside the window:

    H = Σ_u ê_u(n̂_u) + Σ_u b_u n̂_u + ½ Σ_{u≠v} W_uv n̂_u n̂_v + Σ_{u≠v} τ_uv c†_v c_u

- **ESTABLISHED:** the form (Pariser & Parr, J. Chem. Phys. 21, 466 (1953); Pople, Trans. Faraday
  Soc. 49, 1375 (1953)); on-site correlation as in Hubbard, Proc. R. Soc. A 276, 238 (1963).
- **NEW-TO-TRACE:**
  - the on-site term is the exact ionisation ladder ê_u(n), not one orbital with a constant U;
  - W is screened exactly by the bound sector (and ε∞ is now trainable, SPEC 17.7);
  - b, the level shifts and τ are learned from TRACE's invariants.

**What the window cuts.** A donor electron in a band is a superposition of configurations with the
carrier on every cation site. Each such configuration (a localised Mg⁺) lies several eV above the
window minimum, so the window drops it, and with it the band. The Au₂/MgO:Al DFT data show this
physics: Al is Al³⁺-like (Hirshfeld +0.40 against Mg +0.35), Au₂ takes −0.27, and the rest spreads
(README, step-1 results).

**The hierarchy, exact to approximate (one choice to be made by measurement):**
1. **Full CI in the carrier subspace.** C(N_sites, N_carriers) configurations. Exact. 56 sites with 3
   carriers is about 2.8·10⁴: sparse, feasible by Lanczos, but large.
2. **One-carrier exact band.** The N_sites configurations of one carrier on a fixed framework, coupled
   by τ. Exact for one carrier and free of self-interaction.
   - Check: a ring gives E_k = ε − 2τ cos k exactly.
   - Check: Theorem 7 gives an algebraic training signal.
3. **Unrestricted Hartree–Fock of the PPP Hamiltonian.** One determinant per spin, O(N_sites³) per SCF
   iteration, self-interaction free for a single carrier. It can break symmetry (a polaron) or
   delocalise (a band), as τ against W decides.
   - Check: the two-site Hubbard dimer gives E_UHF ≥ E_exact, with equality at τ = 0.

**The decision rule (pre-registered here).** Build nothing until step 4 is scored.
- If step 4 passes F4 at ε∞ ≈ 3, the window CI suffices for the benchmark, and couplings come next.
- If F4 fails, build level 2 first: its exactness tests are cheap. Use it only if it lowers the
  transfer margin of `probe_transfer_margin.py` on the step-4 checkpoint by more than the margin itself.
