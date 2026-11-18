# Why TRACE fails Au₂/MgO:Al, and the equations that address it

Derivations, the measured checks behind them, and the hypotheses that the checks
falsified (2026-10-08). **None of this is implemented.** Every statement beyond
a measured number is a theorem proved here or a hypothesis labelled as such.
Units: eV, Å, e; K_E = 14.399645 eV·Å/e².

Measured with `experiments/benchmarks_4g/probe_bound_theory.py` (frozen),
on bulk MgO (rocksalt, a = 4.212 Å, 2×2×2 cubic cell, 64 atoms, 2560 bonds), from
the trained checkpoints `AuMgO_elec_r5_s42` (energies and forces) and
`AuMgO_elecq_s42` (plus the Hirshfeld charge loss).

| § | result | label |
|---|---|---|
| 1 | Bond-compliance identity: in a centrosymmetric crystal, ε∞ is a plain sum of bond compliances; the Coulomb interaction gives no protection (**verified in the suite**: `tests/electrostatics/test_periodic.py::test_bond_compliance_identity_for_centrosymmetric_crystals`, relative gap 9.2e-16, control 2.9e-3) | NEW-TO-TRACE identity (the SQE dielectric itself: Nistor & Müser 2009) |
| 2 | One atom, one curve: the bound–free cross-hardness is wrong in TRACE-Redox; the fix moves acceptors from anions to cations | NEW-TO-TRACE (principle: Iczkowski & Margrave 1961; PPLB 1982) |
| 3 | Vanishing occupancy gradient: an unoccupied acceptor receives gradient ∝ e^{−ΔE/kT} | NEW-TO-TRACE (envelope theorem; annealing: Rose 1998) |
| 4 | Identifiability: what energies, forces and charges can and cannot fix | NEW-TO-TRACE (extends the ½κt*² result, item 1 of §4) |
| 5 | Two-level bond law κ = G³/(4h²) and the Pauli selection rule | ESTABLISHED ingredients (second-order perturbation theory; Harrison 1980; Kohn 1996) |

## 0. Triage: which gaps are electrostatics

- **Au₂/MgO force RMSE** (TRACE + elec 0.119 eV/Å; local 0.152; NequIP 0.009 at
  6 layers × 4 Å and 4,000 structures, against our one layer, r_max 5.29 Å and
  300 structures): this gap is mostly the short-range backbone and the data.
  No electrostatic equation closes a factor of 13 in it. The like-for-like
  experiment (PFN's 4,000 structures, a deeper backbone) has not been run.
- **Au₂/MgO wetting difference** (DFT −931 meV; TRACE ≈ 2 meV) and the **Au₂
  charge shift** (DFT −0.266 e; TRACE +0.012 e): these are electron transfer.
  This note is about them.
- At PFN's training sizes TRACE + elec already has the lowest published pooled
  force error on Ag₃ (0.0159 eV/Å) and NaCl (0.0068 eV/Å), and the NaCl shift
  (0.104 Å against DFT 0.107 Å). See `experiments/benchmarks_4g/README.md`.

## 1. The bond-compliance identity

**Setup** (SPEC 1/15, periodic, tin-foil boundary, fixed macroscopic field E).
The variables are the bond currents t and the reduced dipoles μ; ρ is the fixed
neutral carrier. q = B t are the atomic charges, and p_b = f_b d_b is the
bond-current dipole (switch factor times image-resolved bond vector). Then

    U(t, μ) = ½ Σ_b κ_b t_b² + ½ qᵀ h q + qᵀ T₀₁ μ + ½ μᵀ (T₂ + A_μ) μ − E·(Σ_b t_b p_b + Σ_i ℓ_i μ_i),

with h = J_lattice + diag(η).

**Theorem 1.** Suppose every atom is an inversion centre of the crystal and of
its bond star, with κ_b and f_b invariant under that inversion. Then

    ε∞ = ε∞^dip + (4π K_E / Ω) Σ_b p_b ⊗ p_b / κ_b,

where ε∞^dip is the response of the dipole sector alone (t ≡ 0).

*Proof.* Take the trial point t_b = p_b·E/κ_b, μ = μ^dip(E).

1. q_i = Σ_b B_ib t_b = 0. The inversion at i pairs each bond b leaving i with a
   bond b′ leaving i that has p_b′ = −p_b and κ_b′ = κ_b, so t_b + t_b′ = 0.
2. The dipole-only solution is uniform on each sublattice. T₀₁ is odd and the
   lattice is inversion symmetric about i, so the potential φ_i = Σ_j T₀₁,ij μ_j
   vanishes.

So ∂U/∂t_b = κ_b t_b + (Bᵀ(hq + T₀₁μ))_b − p_b·E = 0. ∂U/∂μ reduces to the
dipole-only stationarity, because T₀₁ᵀq = 0. U is strictly convex on the
feasible set, so this point is the minimiser. ∎

**Consequence.**
- In a bulk crystal the charge Coulomb kernel gives no depolarisation of bond
  currents: a uniform bond current creates no charge.
- Only κ_b bounds ε∞, and one soft bond class makes the crystal a metal.
- Nistor & Müser (PRB 79, 104303 (2009)) showed that SQE is a dielectric with
  ε − 1 ∝ 1/κ. The identity is the exact TRACE form, with the dipole channel
  included.

**Measured** (`AuMgO_elec_r5_s42`):

| bond class | n | κ (median) | contribution to ε − 1 |
|---|---|---|---|
| Mg–Mg, 2.98 Å | 192 | 0.134 | **1281.44** |
| Mg–O, 2.11 Å | 192 | 2.575 | 33.37 |
| all other bonds | 2176 | ≥ 1.5·10³ | 0.16 |
| dipole sector alone (measured ε∞^dip = 1.99) | | | 0.99 |
| **sum** | | | **1315.96 → ε∞ = 1316.96** |
| exact minimiser | | | **ε∞ = 1316.955** |

With the Hirshfeld loss (`AuMgO_elecq_s42`): Mg–Mg κ = 0.126 contributes
1362.73, and ε∞ = 1378.62. The charge loss does not touch these bonds (§4).

**Without inversion symmetry.**
- The charge-sector Coulomb block Bᵀ h B is positive semidefinite: measured
  λ_min = −1.2·10⁻¹⁰ eV/e², which is round-off.
- So with the dipoles frozen, operator monotonicity of the inverse gives
  α_bond ⪯ Σ_b p_b p_bᵀ/κ_b in Loewner order.
- The **full** bound, with the dipoles included, is **false**: the dipole
  block has the Lorentz mode (λ_min = −2.37 eV/e²), the local-field enhancement
  of Clausius–Mossotti. The dipole channel must be treated exactly.

## 2. One atom, one curve

**Principle** (ESTABLISHED: Iczkowski & Margrave, JACS 83, 3547 (1961);
Perdew, Parr, Levy & Balduz, PRL 49, 1691 (1982)). The energy of an atom in a
molecule is a function of its **total** electron population. It does not
depend on how a model splits that population into "bound" and "free" parts.

**TRACE-Redox today.** These are the on-site terms of atom u, with bound charge
q_b (from SQE) and free integer charge n:

    e_u(q_b, n) = χ_u q_b + ½ η_u q_b² + J^fb_u q_b n + Ê_u(n) + ½(1 − λ) J_u n²

- η_u is the bound sector's total on-site curvature: h_qq = j_pair + diag(η).
- J^fb_u is the Coulomb self-integral of the free and bound Gaussians
  (`trace_redox/bound.py`, `charge_coulomb_block`, on-site element). The
  free–bound coupling A is pure Coulomb.

**Theorem 2.** e_u depends on (q_b, n) only through q_b + n, up to the integer
derivative discontinuity, only if ∂²e/∂q_b∂n = ∂²e/∂q_b². That is,
J^fb_u = η_u. TRACE-Redox has J^fb_u ≠ η_u, so it carries the defect

    δe_u = −Δ_u q_b n,     Δ_u = η_u − J^fb_u,

where Δ_u is the chemical (non-Coulomb) hardness of the bound charge. The
bound charge feels it, but the free charge on the same atom does not. ∎

**Measured** (bulk MgO; J^fb = J_self here, since σ_free = σ):

| model | atom | q_b | η | J_self | Δ = η − J | shift of the n = −1 level, −Δ q_b |
|---|---|---|---|---|---|---|
| elec | O | −0.762 | 10.93 | 9.45 | +1.48 | **+1.13 eV** (worse acceptor) |
| elec | Mg | +0.762 | 8.24 | 6.87 | +1.36 | **−1.04 eV** (better acceptor) |
| elecq | O | −0.334 | 13.20 | 10.73 | +2.47 | +0.82 eV |
| elecq | Mg | +0.334 | 10.28 | 7.20 | +3.08 | −1.03 eV |

**Consequence.**
- Restoring the consistent coupling raises the acceptor level of every anion
  and lowers that of every cation by Δ|q_b|, about 1 eV here. The donor
  levels move the other way.
- This is the band character of an ionic oxide: anion-derived valence band,
  cation-derived conduction band. The current coupling cannot produce it.
- In the epoch-2 redox model the donor electron sits on O (Ê_O(−1) = −4.58 eV)
  rather than on Au₂ (Ê_Au(−1) = −2.6 to −3.0 eV). The O–Au margin there,
  1.6–2.0 eV, has the size of these shifts.
- These shifts are **unscreened**. The bound sector relaxes around n and
  reduces them. That is test P1, and no claim is made before it.

**The consistent operator.** Let the chemical hardness act on the total charge,

    E_chem = ½ Σ_u Δ_u (q_b,u + n_u)²,

that is:
- A′ = A + diag(Δ) [B | 0] (the on-site charge row of the coupling);
- J′ = J + diag(Δ);
- Ê′_u(n) = Ê_u(n) − ½ Δ_u n².

An isolated atom has q_b ≡ 0, so its IE/EA anchors are unchanged exactly.
W′ = J′ − A′K⁻¹A′ᵀ is the Schur complement of a quadratic form that is the sum
of Coulomb(q_b + n), ½ΣΔ(q_b + n)² and the bond and dipole stiffnesses. That
form is positive definite, so W′ is too.

*The convexity condition, resolved: a physical bound.* TRACE-Redox keeps the
ladder Ê self-Coulomb-removed, with ½J_uu c² in W, so that an isolated atom has
E₊₁ = IE and E₋₁ = −EA exactly. One atom, one curve puts Δ_u in the same place,
as a self-term of the total charge. The anchor gap becomes
G′_u = U_Z − (1 − λ)J_Z − Δ_u, with U_Z = IE − EA (λ = 0 shown). Then
G′_u > 0 is the same condition as

    η_u < U_Z = IE − EA.

- **Why it holds.** By PPLB the integer second difference of the total atomic
  energy is U_Z, and the derivative discontinuity U_Z − η_u must be ≥ 0. So the
  smooth on-site hardness of the bound charge may not exceed the atom's
  fundamental gap.
- **Link to QEq.** This is Rappé & Goddard's QEq anchor (J. Phys. Chem. 95,
  3358 (1991), J⁰ = IP − EA) read as an upper bound.
- **Measured** (elec): η_O = 10.93 < U_O = 12.16, allowed;
  η_Mg = 8.24 > U_Mg = 8.07, a slight violation (8.066 − 6.87 − 1.36 = −0.16 in
  the Ê form).
- **Consequence.** It is not a defect of the theory. It is a bound the bound
  sector should obey: J_u < η_u < U_Z. How to impose it without changing the
  bound sector for non-redox models is a design decision for the
  implementation step.

## 3. Vanishing occupancy gradient

**Theorem 3.** Take the exact configuration ensemble
F = −kT ln Σ_N w_N e^{−E(N)/kT}, where Ê_u(n) enters each E(N) through the
populations.
1. ∂F/∂Ê_u(n) = ⟨p_u(n)⟩, the ensemble probability of unit u being in state n.
2. Every force derivative, |∂²F/∂Ê_u(n)∂R_a|, is at most a constant times
   ⟨p_u(n)⟩ (max_N |∂E(N)/∂R_a| / kT + 1).

*Proof.*
1. The envelope theorem in each E(N) gives ∂E(N)/∂Ê_u(n) = p_u(n | N).
   Averaging with the Boltzmann weights gives (1).
2. Write ∂⟨p⟩/∂R = −Cov_π(p, ∂E/∂R)/kT + ⟨∂p/∂R⟩. Since p ≥ 0,
   |Cov(p, X)| ≤ ⟨p⟩ max|X − ⟨X⟩|. Inside each configuration, ∂p/∂R ∝ p/kT
   by the same identity. ∎

**Consequence.**
- An acceptor ΔE above the occupied one has ⟨p⟩ ≤ e^{−ΔE/kT}. Measured in the
  epoch-2 redox model, Al → Au costs 1.2–2.6 eV more than Al → nearest O
  (+2.0 to +2.5 eV against −0.13 to +0.82 eV). At kT = 0.1 eV that gives
  ⟨p_Au⟩ ≤ 6·10⁻⁶ down to 5·10⁻¹².
- Measured: the free charge on Au₂ is 0.0000 e (epoch 2, 6 structures).
- Gradient descent therefore cannot promote an unoccupied acceptor. Whichever
  acceptor wins at initialisation keeps winning, whatever the data say about
  Au₂.

**Remedy** (ESTABLISHED technique: deterministic annealing, Rose, Proc. IEEE
86, 2210 (1998)).
- Use a kT homotopy: kT(epoch) runs from kT₀ ≈ ΔE/2 down to the production kT.
- Error control (Gibbs): 0 ≤ E_min − F_kT ≤ kT ln Ω_eff.
- This is a heuristic, not a guarantee. It finds the right basin only if that
  basin stays connected as kT falls.

## 4. Identifiability: what energies, forces and charges fix

1. **½κt*² (an earlier result of the project, restated here).** ∂E*/∂κ_b = ½ t_b*², so the force derivatives
   also scale with t_b*.
   - Measured median ½κt*²: Mg–Mg 6.1·10⁻⁶ eV against Mg–O 2.2·10⁻²
     (undoped).
   - The loss does not see κ_MgMg, so it stays at the floor (0.134 trained,
     0.126 with charges).
   - By Theorem 1 that floor value is ε∞.
2. **Locality absorption.** Write E_θ = E_loc + E_el, where E_loc is any sum of
   r_c-local terms (a universal local approximator).
   - Two electrostatic parameterisations are indistinguishable from E and F
     exactly when their difference is r_c-local on the data.
   - ε∞ is identifiable only through screening at separations beyond r_c.
3. **Far-field scaling.** The far field depends on Q/√ε∞ (and Z*/√ε∞) only:
   E_far = Σ Q_A Q_B/(ε∞ R_AB) + … is invariant under (Q, ε∞) → (sQ, s²ε∞).
   - Charge labels fix Q, but not the Mg–Mg null directions, whose t* ≈ 0.
   - Measured: the Hirshfeld-trained model still has ε∞ = 1379.

**For a foundation model,** ε∞ must come either from data that carry it, or
from a physical prior on exactly the unidentified directions (§5):
- MP DFPT ε∞: Petousis et al., Sci. Data 4, 160134 (2017);
- molecular polarizabilities.

By (1), such a prior costs the fit nothing to first order.

## 5. The two-level bond law and the Pauli rule (ESTABLISHED ingredients)

**The law.** Take one occupied orbital on A and one empty orbital on B, with
level gap Δ, hopping h and G = √(Δ² + 4h²).
- The bonding orbital puts N_B = 1 − Δ/G electrons on B, exactly.
- dN_B/dΔ = −4h²/G³.
- So the split-charge hardness is **κ = G³/(4h²)**, exact for two orbitals.

Consequences:
- **Pauli selection rule.** With no occupied–empty pair across a bond
  (cation–cation with empty valence shells, or anion–anion with full ones),
  there is no linear charge flow: κ = ∞. Mg²⁺–Mg²⁺ and O²⁻–O²⁻ in MgO.
- **Nearsightedness.** In an insulator h decays exponentially (Kohn, PRL 76,
  3168 (1996); He & Vanderbilt, PRL 86, 5341 (2001)). Then κ_b ≥ c e^{2r_b/ξ}
  and the sum in Theorem 1 converges as ∫ r⁴ e^{−2r/ξ} dr, independent of the
  bond cutoff.
  - The current heads have no such law. Measured, untrained, at cutoffs
    5.29 / 3.5 / 2.5 Å: ε∞ = 1517 / 275 / 56.
- **MgO estimate.**
  - Inputs: G ≈ 7.8 eV (the gap); h ≈ V_spσ = 1.42 ħ²/(m d²) = 2.44 eV at
    d = 2.106 Å (Froyen & Harrison, PRB 20, 2420 (1979)).
  - Result: κ_MgO ≈ 19.9 eV/e², against 2.575 and 5.904 trained.
  - What-if (frozen, nearest-neighbour Mg–O only, Mg–Mg and O–O removed by
    the Pauli rule): **ε∞ = 6.3** (elec dipoles) / **5.5** (elecq dipoles),
    against experiment ≈ 3.0 and the trained 1317 / 1379.
  - That is the right order of magnitude, about 2× high. The two-level model
    has no band dispersion.
  - The bond and dipole channels describe the same O 2p → Mg 3s response, and
    energies and forces cannot split it (§4). For comparison, Clausius–Mossotti
    with ionic dipoles alone needs α(O²⁻) + α(Mg²⁺) = 1.78 Å³ for ε∞ = 3.0.

## 6. Hypotheses falsified today

- **A full Loewner certificate on ε∞:** false (§1). The dipole block is
  indefinite (−2.37 eV/e²).
- **"The metallic bound sector is what makes O the acceptor":** mostly false.
  - From ε∞ = 1317 to 6.3, the on-site polarisation energy (J − W̃_OO)/2 of a
    free charge on O changes only from 1.94 to 1.44 eV. W̃_OO goes from 0.946
    to 1.942, with J = 4.827.
  - Local bonds screen an on-site charge in any dielectric.
  - The acceptor choice is set mainly by §2 and §3.
- **"The two-level law gives ε∞ ≈ 3":** no. It gives 6.3 / 5.5, the order of
  magnitude only.

## 7. Pre-registered sequence (one change at a time)

**Frozen probes first, no training:**

| id | test | prediction | if it fails |
|---|---|---|---|
| P1 | One-curve coupling (§2) applied to the redox checkpoint, after its run finishes: recompute the integer transfer energies Al → O, Al → Mg, Al → Au | O's screened acceptor level rises at least 0.5 eV relative to Au | §2 is dropped |
| P2 | ∂F/∂Ê_Au(−1) by autograd on the redox checkpoint at kT = 0.1, 0.5, 1.0 | equals ⟨p_Au(−1)⟩ to round-off (Theorem 3), and grows by orders of magnitude with kT | — |

**Unit tests, written before any code:**
- Theorem 1 on rocksalt and CsCl cells, to a tolerance derived from machine
  epsilon;
- κ = G³/(4h²) against exact diagonalisation of the two-orbital model;
- e_u depends on q_b + n only, under the consistent operator.

**Training,** each change alone, on the Au₂/MgO protocol, scored by the
unchanged T1–T6:
1. the bond law with the Pauli prior on κ;
2. plus the one-curve coupling;
3. plus the kT homotopy.

The NaN in `AuMgO_redoxq_r5_s42` is diagnosed first: a resumed reproduction is
running, and an exact rerun from scratch follows if it does not trigger.
Hypothesis H1 there is an `eigh` backward through a degenerate spectrum in
`trace_redox/free_solver.py:_constrained_step`.
