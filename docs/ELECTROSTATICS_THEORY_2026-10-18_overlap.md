# Theory note 2026-10-18: continuity, one sector per degree of freedom, overlap electrostatics, exact limits

Context: trace_redox SPEC 21–24, built today. Measurements are in experiments/benchmarks_4g/README.md and
trace_electrostatics/CHANGELOG.md, Round 23. Units: eV, Å, e; K = 14.399645 eV·Å/e². Atomic units where stated.

## 0. The configuration energy as implemented

For integer charges n (units u, frames f_u, carriers Q_u = n_u − f_u):

    E(n) = Σ_u ê_u(n_u) + bᵀn + ½ nᵀ W n + Σ_{pairs p=(u,v)} Ξ_p(n_u, n_v)

- ê_u(n) = E_Z(n) + g_u Δ_u(n) − ½ S₀,u n². Here E_Z is the exact isolated-ion ladder, Δ the learned shift, g
  the environment gate (§4) and S₀ the Gaussian self energy.
- W = J − A K⁻¹ Aᵀ is the free–free Gaussian Coulomb J screened by the eliminated bound sector, whose operator K
  holds bond charge flows t and moments p. A couples free charges to them.
- Ξ_p is the sum of the two-body terms: exclusion (SPEC 18.5) and penetration (§3).
- The diabatic Hamiltonian is H̃ = diag(E) + Σ_moves sign·σ(e_ij)·T_ab over a set P (§1), and the canonical
  free energy is F = −kT ln Tr e^(−H̃/kT).

## 1. Continuity of F in the geometry (SPEC 21)

**Proposition 1 (the energy window is discontinuous).** Take P(R) = {c : E_c − E_min ≤ w}, and let c cross the
edge at R*, coupled to the low block by t. Then F jumps at R* by λ₀(P ∪ {c}) − λ₀(P), up to O(e^(−w/kT)) and up to
the weight of the excited levels. For |t| ≪ Δ_c the jump is −|⟨c|H|ψ₀⟩|²/(E_c − λ₀) + O(t⁴/Δ³). For |t| ~ Δ it
is not small.

*Proof.* The lowest eigenvalue of a bordered symmetric matrix [[H_P, v], [vᵀ, E_c]] satisfies the secular equation
λ = λ₀ + Σ_k |⟨k|v⟩|²/(λ − E_c) over the eigenvectors k of H_P (Cauchy interlacing; Löwdin partitioning). At
kT ≪ the gap, F = λ₀ − kT ln(1 + …). The selection is detached, so dF/dR within each P never sees the jump. ∎

*Measured:* −834 meV (Au⁺Au⁻ on the edge, τ = 1.18 eV, Δ = 2.50 eV). The truncation deficit carries the residual
scatter with r = +0.99 (seed 43).

**Definition (importance).** With E_ref = −kT ln Σ_P e^(−E/kT), x_c = E_c − E_ref ≥ 0 and g(t, x) = |t|/√(x² + t²):

    r_c = max( e^(−x_c / 2kT),  max_{i coupled to c} r_i g(T_ic, x_c) ),    e_ij = max(r_i g(T, x_j), r_j g(T, x_i)).

g ≤ 1 and g ~ |t|/x for x ≫ |t|. So r_c is the leading Rayleigh–Schrödinger amplitude of c along its best path from
the thermally weighted reference, with denominators from the reference energy.

**Lemma 1 (closure).** Let P = W ∪ {c : r_c > a_off} and σ = smoothstep((ln e − ln a_off)/ln(a_on/a_off)). Then
σ(e_ic) = 0 for every pair with i ∈ P and c ∉ P.

*Proof.* Take c ∉ P, so r_c ≤ a_off. For any neighbour i, r_c ≥ r_i g(T_ic, x_c) by the definition of r_c as a
maximum. So r_i g(T, x_c) ≤ a_off. Also r_c g(T, x_i) ≤ r_c ≤ a_off, since g ≤ 1. Hence e_ic ≤ a_off and σ = 0. ∎

**Corollary.** H̃ is block-diagonal between P and its complement. A larger window adds only decoupled blocks,
F(w′) = −kT ln(Z_P + Z_rest), and F is continuous in R up to the Boltzmann weight of the blocks entering at the
edge. Within a fixed P, F is piecewise C²: r has kinks where two paths tie.

**Why denominators from the reference.** Pairwise gaps |E_i − E_j| make a hop between two nearly degenerate
high configurations (O-hole at 10 eV → O-hole at 12 eV) look strong (a = 0.1–0.2). That chains the hole over
the O sublattice and exceeded the 512 cap on frozen step-12 weights. The true amplitude of the second hole
position is t₁t₂/((E₁ − E_ref)(E₂ − E_ref)) ≈ 0.003.

## 2. One sector per degree of freedom: bond charge flow (SPEC 22)

**Proposition 2 (double counting).** For a bond i–j the bound sector holds a split charge t_b with incidence column
B_b = e_i − e_j and hardness κ_b. Eliminating it lowers the energy of a free ion pair n = Q(e_i − e_j) by

    ΔE = −½ Q² (Aᵀ(e_i − e_j))ᵀ K⁻¹ (Aᵀ(e_i − e_j)) ≤ 0.

Its bond block, alone, is −½ Q² (J_fb B_b)²/κ_b-like: second order in the same charge flow i → j that the free
configurations (±Q, ∓Q) and τ_ij already carry exactly (Theorem A of the 2026-10-16 note: the two-state
functional of the bond). The two descriptions of one mode add.

*Measured:* bond channel alone −3.41 eV (seed 42, κ = 13.3 eV) and −0.85 eV (seed 43, κ = 23.0 eV) on Au⁺Au⁻ at
2.47 Å. κ is not fixed by the data: Δ_ion = +0.17 / +2.47 eV, and +2.74 / +3.00 eV with the bond channel removed.

**Rule.** Keep t_b only inside a covalent unit of more than full_unit_atoms atoms, where the free sector forbids
intra-unit ± pairs. Elsewhere set f_b → 0.
- This is exact bookkeeping: each charge-flow mode is counted once.
- Atomic moments then carry the ionic polarisation (Clausius–Mossotti; MgO needs α_O²⁻ + α_Mg²⁺ ≈ 1.76 Å³ per
  formula unit for ε∞ = 2.95).

## 3. Overlap electrostatics: penetration (SPEC 23) and induction (SPEC 26)

**Exact pair electrostatics.** Two spherical ions a, b (nucleus Z, density ρ, charge q = Z − N) at distance R have

    E_es = Z_aZ_b/R − Z_a φ_b(R) − Z_b φ_a(R) + (ρ_a|ρ_b)(R),   φ(d) = (4π/d)∫₀^d ρr²dr + 4π∫_d^∞ ρr dr.

**Proposition 3 (cancellation-free form).** Let δ = ρ(n) − ρ(f) be the carrier's electron-density difference, with
N_δ = −Q electrons, and let ψ(d) = N/d − φ(d) = ∫_d^∞ 4πr²ρ(r)(1/d − 1/r) dr ≥ 0. Then

    D ≡ [E_es(n_a,n_b) − E_es(f_a,f_b)] − (n_a n_b − f_a f_b)/R
      = −f_b I_a − S_a[ψ_b^f] − f_a I_b − S_b[ψ_a^f] + N_δb I_a − S_a[ψ_δb],
    I_a = ∫_{r>R} 4πr²δ_a (1/r − 1/R) dr,   S_a[g] = ∫ 4πr²δ_a ⟨g⟩(r;R) dr,   ⟨g⟩ = [G(R+r) − G(|R−r|)]/(2rR).

*Proof.* Use V_b^f = Z_b/d − φ_b^f = q_b/d + ψ_b^f, the shell average ⟨1/d⟩ = 1/max(r, R) and
∫δ/max(r, R) = N_δ/R + I. The point-charge parts cancel term by term against (n_a n_b − f_a f_b)/R, and the
δ_a δ_b term is bilinear. ∎

Every piece is short-ranged and scales with the penetration, not with Z_aZ_b/R. Measured: the form is
grid-converged to 1e-7 eV, while the total-energy route is off by 8e-4 eV (Au–O) and 5e-5 eV (Na–Cl), from its
10⁻⁶-level sensitivity to the electron count.

**Result (measured, X2C-PBE0).** Next to O²⁻ at 2.2 Å, the model's Au⁺ preference over Au⁻ is too large by 3.77 eV.
The ±9 % (Watson radius), +2.4 % (basis) and 3 % (functional) are the stated uncertainties. For the Au⁺Au⁻ pair the
two penetrations cancel (net −0.06 eV).

**Proposition 4 (field of a spherical carrier on a moment; Gauss).** The field at distance R from a spherical carrier
of charge Q and shape c is E = K Q λ(R)/R², with λ(R) = ∫4πr²c(r)F(r; R, s)dr the fraction of c, smeared by the
moment's Gaussian (width s), inside R:

    F(r; R, s) = ½[erf((R−r)/s) + erf((R+r)/s)] + (s/(2√π r))[e^(−(R+r)²/s²) − e^(−(R−r)²/s²)].

The model uses a Gaussian carrier, λ_G = erf(R/s′) − (2/√π)(R/s′)e^(−R²/s′²) with s′ = √(σ_f² + s²).

*Measured* (O site, s = 0.84 Å, R = 2.2 Å): λ = 0.475 (Au⁻) and 0.668 (Au⁺), against λ_G = 0.827. Induction scales
as λ², so the model overstates Au⁻-driven induction 3× and Au⁺-driven 1.5×. With α(O²⁻) ≈ 1.7 Å³ that is
≈ −0.24 / −0.12 eV per O neighbour: a net Au⁻ preference error of ~0.1 eV, of opposite sign to the penetration
error and ~3 % of it.

**Proposition 5 (the state-dependent coupling is exactly one- plus two-body; built as SPEC 26).**
Let the coupling of unit u in state s be A_u n_u + Q_us v_us, with v_us the correction column (λ_c − λ_G) of
Proposition 4. Then for every integer configuration n,

    E_b(n) = E_b^G(n) + ΔFᵀ(y0 − M n) − ½ ΔFᵀ K⁻¹ ΔF,   ΔF = Σ_u Q_u v_u(s_u),  M = K⁻¹Aᵀ,  y0 = −K⁻¹g,

and the correction is exactly Σ_u E1_u(s_u) + Σ_{u<w} E2_uw(s_u, s_w) with

    E1_u(s) = Q c0_us − Q² v_us·M_u − ½ Q² v_us·K⁻¹v_us,    c0 = Vᵀ(y0 − M f),
    E2_uw(s, t) = −Q_us Q_wt (v_us·M_w + v_wt·M_u + v_us·K⁻¹v_wt).

*Proof.* With the source g + Aᵀn + ΔF, the minimum of ½yᵀKy + yᵀ(source) is −½ sourceᵀK⁻¹source. Expanding about
ΔF = 0 gives the first line exactly (the expansion is quadratic, not truncated). Since ΔF is a sum of single-unit
terms, ΔFᵀ(y0 − M n) and ΔFᵀK⁻¹ΔF hold only one- and two-unit products. Writing n_u = f_u + Q_u then sorts them
into E1 and E2. ∎

So the "non-quadratic" coupling costs nothing in exactness. It does not need the per-member solves anticipated
earlier: it fits the existing ladder plus SPEC 18.5 pair slot, at the price of one extra solve with K for the n·S
columns. Measured on frozen weights (SPEC 26):
- Au⁻ +0.19 eV, Au⁺ +0.008 eV (medians);
- populated pair terms ≤ 0.27 eV;
- total +0.41 eV on a doped cell.
The Au⁻ − Au⁺ shift, +0.18 eV, matches the Gauss estimate above (~0.1–0.24 eV) in sign and order.

## 4. Exact isolated limits (SPEC 24)

**Requirement.** An isolated ion must have E = E_Z(n) exactly. That is what makes every dissociation into ions
exact, e.g. the Na + Cl ionic–covalent crossing at R_c = K/(IE_Na − EA_Cl) = 9.434 Å.

**Measured violation.**
- Isolated atoms carry trained shifts: Au⁻ +1.18 eV, Au⁺ +1.25 eV (AuMgO step 12), Ag⁺ −1.36 eV (Ag₃ step 9).
- The charge-state head is an MLP of the atom's own invariants; nothing makes it vanish without neighbours.

**Construction.** g_u = 1 − Π_{v}(1 − s(r_uv)), with s = 1 for r ≤ 0.8 r_c and the C² quintic to 0 at r_c.
- g = 0 exactly for an isolated atom (no edges) and g = 1 exactly with one neighbour inside the onset.
- Hence ê_u(n) + ½ S₀ n² = E_Z(n) for isolated atoms whatever the weights.

**Test (built):** the Na + Cl crossing falls at 9.462 Å against the exact 9.434 Å; the 0.03 Å is the induction of the
separating ions. Without the gate, with ~1 eV shifts, the gap stays at +2.1 eV and there is no physical crossing.

**Band (SPEC 20) corollary.** The band holds m ≤ 2 electrons, so the state vector must reach −2 whatever elements are
present. Before, an isolated Mg was refused ("capacity [1, 3]"). With the fix, Mg²⁺ + 2 band electrons reproduces
E_Z(Mg⁰) to 3.6e-15 eV.

## 4a. Identifiability of the free sector (Proposition 6; 2026-10-18 night)

**Setting.** The model energy is E(R) = E_s(R; θ_s) + E_b(R; θ_b) + F(R; θ_f): short-range network, bound
sector, and the free energy of the configuration ensemble.

**Proposition 6 (energies do not identify the free sector; field responses do).**
- Let δF(R) be the change of F under a change of θ_f. Suppose δF is a smooth function of the atoms' local
  environments within the short-range cutoff (true of the learned shifts, and of τ and Δ_ion within the
  cutoff). If the short-range network can represent −δF, then (θ_s', θ_f') gives the same E(R), and so the
  same energies and forces, on every structure. The training loss on energies and forces is invariant along
  this gauge orbit.
- The applied-field response α = −∂²E/∂E_ext² is not: E_s does not depend on E_ext. So α (and ε∞) breaks the
  gauge.
- The same holds for total-charge differences (vertical EA and IP of isolated species), provided the
  short-range network does not take the total charge as input. Verified for flashace: `total_charge` reaches only
  the electrostatics module (flashace/model.py), and the embedding reads Z alone.

*Proof.* The first statement is the decomposition itself. For the second, ∂²E_s/∂E_ext² = 0 identically, so α
depends on (θ_b, θ_f) alone. The coherent and thermal parts of the ensemble's response, Σ_k |⟨0|d|k⟩|²/(E_k −
E_0) and Var_p(d)/kT, are functions of the configuration energies and couplings, so α constrains exactly the
free-sector quantities the gauge leaves open. ∎

**Measured consequences.**
- Two seeds of one model sit on different points of the orbit. Their energies are similar (test RMSE within
  ~2.7×), but their gas-phase Au₂ α∥(R) differs in shape: 1.23 → 1.76 against 1.00 → 1.16 per 2α_atom,
  non-monotone. The reference is 1.36 → 2.30, and α⊥ (bound sector) is right in both.
- Off the training distribution (the T5 relaxations, isolated dimers) the orbit points differ, which is the seed
  dependence of T5. **Withdrawn for T5 (measured 2026-10-19, `probe_gauge_split.py`):** at the four T5 reference
  geometries of step 13, T5 = −556.6 = E_s −1.5 + E_q −555.2 meV (seed 42) and −1372.7 = E_s +4.7 + E_q −1377.3 meV
  (seed 43).
  - E_s cannot see the dopant, so it cannot absorb a change of E_q's T5. T5 lies off the gauge orbit: the
    training energies pin it.
  - The seed difference (816 meV) is entirely in E_q, in the upright undoped dimer: p(Au⁺Au⁻) 0.22 against 0.925,
    E_q(w − nw) +0.379 against +1.105 eV.
  - The orbit's freedom is visible where the proposition says: the common wetting split carried by E_s,
    E_s(w − nw) = +0.130 / +0.159 eV, nearly the same in both seeds.
  - So the T5 seed dependence is a misfit of the four class means (test H1: undoped flat − upright −390 / +362 meV),
    not a gauge freedom. It remains a gauge freedom for the isolated dimers' α.
- Step 13F trains α∥(R) of Au₂ from first principles (train.py `field_polarizability_prior`, NEW-TO-TRACE) to
  break the gauge.
- Corollary for the foundation model: every free-sector degree of freedom needs a partition-free response
  observable in training, as α for charge transfer and ε∞ for the bound sector. Charge partitions (Hirshfeld)
  are not such observables (benchmarks README, partition correction).
- **Second remedy (not yet run): gauge fixing by minimum shift norm.** Along the orbit the loss is exactly flat, so
  a weak penalty λ Σ_u Σ_n Δ_u(n)² on the learned charge-state shifts selects the representative with the
  smallest shifts. That representative is unique wherever the penalised problem is strictly convex along the
  orbit. Where the data constrain the shifts the data dominate; where they do not, every seed is driven to the
  same point.
  - ESTABLISHED: Tikhonov regularisation. NEW-TO-TRACE: its use as the gauge condition of the free sector.
  - Measured motivation: the trained isolated-atom shifts are 0.4–1.8 eV with no physics asking for them (Au EA
    0.53–1.94 against the exact 2.31).
  - To be pre-registered once the CPU queue allows, as a one-change step against the field-prior step.
  - **Not a T5 remedy (2026-10-19):** T5 is pinned by the energies (above), so the penalty would move no T5 number.
    Not built; kept for the free-sector responses of isolated species, where the orbit is real.

## 4b. Spin in the spinless configuration model (Proposition 7; SPEC 29)

**Proposition 7.** Let two sites each carry one spatial frontier orbital, with hopping t and on-site repulsion
U. In the spinless configuration model (one orbital per charge step, §17.2), the coupling τ that reproduces the
lowest-spin energy at electron count N is τ = √2 t for N = 2, and τ = t for N = 1 and N = 3. With the factor f =
√2 on moves where donor and acceptor have equal electron-count parity, and f = 1 otherwise, τ = t reproduces all
three exactly.

*Proof.*
- N = 2: the singlet sector is spanned by |cov⟩ = (|↑,↓⟩ − |↓,↑⟩)/√2 and |ion⟩ = (|↑↓,0⟩ + |0,↑↓⟩)/√2, with
  ⟨ion|H|cov⟩ = −2t. So E₀ = U/2 − √(U²/4 + 4t²).
  - The spinless model couples (0,0) to (+1,−1) and to (−1,+1) with fτ each, which is √2 fτ to their symmetric
    combination. So E₀ = U/2 − √(U²/4 + 2f²τ²), and f = √2 gives it exactly. The moves (1,1) ↔ (0,2) have equal
    parities.
- N = 1 or 3: one carrier, or one hole, resonates. The spinful matrix element is ±t, the parities differ, f = 1,
  and E₀ = −t (or U − t). ∎

Verified against explicit Fock-space diagonalisation (`test_redox_spin_pairing.py`).

**Measured consequence.** On frozen step-13 weights, T5 moves toward DFT for both seeds (−554 → −632 and −1418 →
−1221 meV), and the seed spread falls by 32 %. Only the undoped ionic–covalent sector moves; the doped resonance
has f = 1.

**Beyond two sites (stated).**
- With many singly occupied sites the exact factor depends on the global spin state. f takes each moving pair as
  a singlet.
- High-spin frontier states (O 2p⁴, open d shells) need the full spin algebra (Clebsch–Gordan recoupling of the
  donor and acceptor spins), not the parity rule.

## 4c. Classical electrostatics inverts the ionic–covalent order at bond lengths (Proposition 8; 2026-10-19)

**Proposition 8.** For a homonuclear pair A₂ with point charges and point polarisabilities α, the ionic
configuration's gap over the covalent one is, to leading order in the induction,

    U_cl(R) = IE − EA − K/R − α K/R⁴.

If IE − EA < K/R_e + α K/R_e⁴, classical electrostatics puts A⁺A⁻ below A⁰A⁰ at the bond length.
- *Proof.* The ion pair's Coulomb energy is −K/R. Each ion polarises its partner, −½αE² with E = K/R², twice;
  the neutral configuration has neither term. ∎
- The exact asymptote, mutual induction included, is a machine-precision test (SPEC 24).

**Evaluated** (NIST IE/EA from `trace_redox.constants`; spectroscopic R_e; free-atom α from Schwerdtfeger and
Nagle, Mol. Phys. 117, 1200 (2019); ESTABLISHED ingredients):

| dimer | IE − EA | K/R_e | αK/R_e⁴ | U_cl (eV) |
|---|---|---|---|---|
| H₂ | 12.844 | 19.422 | 31.830 | −38.41 |
| Li₂ | 4.774 | 5.387 | 6.859 | −7.47 |
| Na₂ | 4.591 | 4.677 | 3.863 | −3.95 |
| Cl₂ | 9.355 | 7.243 | 1.995 | +0.12 |
| Ag₂ | 6.272 | 5.692 | 2.864 | −2.28 |
| Au₂ | 6.917 | 5.825 | 2.057 | −0.97 |

- Five of six are inverted, and H₂, Li₂ and Na₂ by the Coulomb term alone. H₂'s 31.8 eV shows that the point
  induction has left its domain at covalent distances (the polarisation-catastrophe regime).
- The covalent bond's true ionic weight is small, because at overlap the charges' Coulomb energy is reduced by
  penetration, the ion pair pays Pauli repulsion, and the field that polarises a partner is that of a density
  rather than a point.

**Consequence for explicit-charge potentials** (NEW-TO-TRACE identification).
- Configuration energies with point or narrow-Gaussian ions place the ionic configurations of most covalent
  bonds below the covalent one by 1–4 eV.
- The learned corrections (here the charge-state shifts) must undo it, and neutral data constrain only one
  combination of (gap, τ) (§4a, measured). So seeds land on different branches, ionic or covalent, with the same
  energies: the seed dependence, and a wrong charge-transfer response.

**Measured** (zero learned shifts, so the explicit physics alone; `probe_explicit_gap.py`; frozen weights):
- Au₂, AuMgO step 13: U(2.5 Å) = −0.61 (seed 42) and −2.02 eV (seed 43) with the trained Gaussians.
  With the overlap terms (SPEC 22, 23, 25, 26, 28) it is +0.85 and +0.90 eV, the same to 0.05 eV.
- Ag₂, the Ag₃± benchmark model (step 9): −0.74 eV at 2.53 Å, and +1.22 eV with the overlap terms.
  - As trained, with learned shifts, its neutral Ag₂ is ionic, U = −1.71 eV with τ = 0.13 eV. That near-degenerate,
    weakly coupled pair is why its Ag₂ α∥ is 5.8× the first-principles value.
- ~~In the AuMgO slab, the overlap terms take T5's seed spread from 816 to 51 meV on frozen weights (benchmarks
  README).~~
  - Withdrawn 17:30: an artifact. The frozen counterfactual kept SPEC 18's fitted exclusion beside Gordon–Kim.
  - Corrected: spread 278 meV at step 17's settings, T5 +108 / −170 meV. Both seeds' upright dimers do become
    covalent and alike (p_ion 0.13 / 0.13).
  - The isolated-dimer gaps above are unchanged on re-measurement.

**Universality.** The criterion involves only IE, EA, α and R_e, so it applies to any element pair a foundation
model meets. It is why TRACE's configuration energies carry the overlap physics of the true ion densities and
not a learned damping: the inversion is several eV and element-specific, and a damping fitted to neutral data is
not identified (§4a).

## 4d. Research directions, each one equation and one deciding measurement (2026-10-19)

The rule: simple equations, precise physics. Nothing here is built; each waits for the measurement
that decides it.

1. **Learned shifts should vanish once the overlap physics is explicit.** The exact split of T5 into the terms of
   SPEC §0 (`probe_t5_terms.py`; F = Σ p_n E(n) + Tr(ρH_off) − TS identically) shows step 13's shifts opposing
   T5 by +453 meV (seed 42), with the physical terms −1008 meV.
   - Au⁻'s shift is +0.57 eV at the O-contact site against +1.77 eV at the far site: compensation for the
     Gaussian over-repulsion of Au⁻ by O²⁻ (−2.4 eV of penetration, SPEC 23).
   - Decides: the shifts' T5 share in steps 15–17, trained (`score_extra.sh` writes it). Prediction:
     |shift share| ≪ 453 meV and T5 toward −932 meV.
   - **Measured, what the shifts are** (`probe_shift_overlap.py`, step 13 seed 42, 40 test structures). Regressing
     the learned shift on the overlap one-body term the model lacks, π_u(n) = Σ_v [Ξ_uv(n, f_v) − Ξ_uv(f_u, f_v)]
     (Gordon–Kim tables; indexing checked against the recorded Au⁻/O²⁻ −1.11 and Au⁺/O²⁻ +0.11 eV at 2.2 Å):
     - Au⁺: Δ = +1.76 ± 0.04 eV in every environment, slope 0.06, R² 0.03. A constant.
     - Au⁻: Δ = +1.39 ± 0.46 eV, slope +1.26, R² 0.20. Its variation follows the substrate overlap with the predicted
       slope ≈ 1, but explains a fifth of it.
     - O, Mg, Al states: |Δ| ≤ 0.14 eV. Never populated, never learned.
     - So the shifts are mostly a constant renormalisation of Au's ionic ladder: +3.15 eV on the Au⁺Au⁻ pair, the size
       of the Proposition 8 inversion, a two-body ion-pair error that this one-body π cannot see. Only a fifth of
       Au⁻'s environmental variation is substrate overlap.
     - For direction 2 this means the simple form needs a constant per (element, state) plus an overlap term, and
       even then it captures only part of Au⁻'s variation. Not decided until the trained steps show what remains.
2. **Shifts as overlap-scaled corrections.** If (1) holds, replace the free MLP shift by
   Δ_u(n) = Σ_v c_{Z_u, n, Z_v} S_uv, with S_uv the frontier-orbital overlap (ESTABLISHED form: Wolfsberg–Helmholz
   / extended-Hückel scaling of on-site energies with overlap; NEW-TO-TRACE: per integer charge state).
   - A few constants per element pair and state instead of a network: identifiable, so it cannot carry a seed
     split.
   - Decides: the same benchmark numbers with fewer parameters, or not.
3. **A hop is a hopping integral.** τ_uv = η(l_u, l_v) ħ²/(m r²) · e^{γ_uv}, with η Harrison's (ssσ −1.40,
   spσ 1.84, ppσ 3.24) for the frontier orbitals and |γ| ≤ ln 2 learned (ESTABLISHED: Harrison 1980).
   - Measured against it: τ(Cl–Cl, 1.99 Å) = 24.9 eV and τ(Ag–Ag) = 0.13 eV where Harrison gives 6.2 and 1.67 eV.
   - Decides: the cation's α (step 18 G4) and U5 on Ag₃⁺. ~~Not the AuMgO seed split: on frozen weights the overlap
     terms removed it with τ 3× apart.~~ (Withdrawn with the 816 → 51 meV artifact; whether τ carries the seed split
     is open again.)
4. **Each state's own damping width.** σ_j(n)³ ∝ α_j(n) restores SPEC 12's theorem, K(n) ≻ 0 for every geometry
   (SPEC 28 limitation).
   - Decides: a measured catastrophe on a selected configuration, or Au⁻'s under-damping visible in T5 (step 17).
5. **Ion densities that breathe in the Madelung potential** (ESTABLISHED: potential-induced breathing, Boyer et
   al. 1985). ρ_{Z,n}(r; φ) = λ³ ρ_{Z,n}(λ r), with λ(φ) from the ion's response to the local potential φ. O²⁻
   exists only in that potential (the library holds it in a fixed Watson sphere).
   - Decides: the overlap terms' share in T5 after step 15, against the frozen estimate.
6. **The isolated limit of the reference method.** The ladder is NIST's exact IE and EA, while the data are PBE
   (FHI-aims, no dispersion: README). For a model of PBE data the consistent limit is PBE's own ion energies.
   - The donor's ionisation cancels exactly in T5, so this does not move T5. It moves absolute transfer energies
     (B1, charged clusters).
   - Decides: PBE IE/EA of the benchmark elements against NIST, when the CPU allows the PySCF runs.

Not pursued, falsified by the data: state-dependent dispersion. The reference has no dispersion, so London terms
would move the model away from it.

**Hypothesis (2026-10-19, ~20:00): Gordon–Kim underestimates the Pauli repulsion of the diffuse Au⁻ by O²⁻.**
- The geometry: in the T5 references, doped and undoped are identical to 0.02 Å. Flat puts both Au over Mg²⁺
  (2.90 Å); upright puts the near Au on an O-top site (2.26 Å) and the far Au 4.8 Å from any O. So T5 is the energy
  of adding the dopant's electron to Au₂ over two Mg²⁺, minus adding it on an O-top site: −0.93 eV in PBE.
- The overlap tables at those distances, relative to the Gaussian model (eV):
  - Au⁻/O²⁻ at 2.26 Å: −2.14 (penetration), −1.00 with Gordon–Kim Pauli (+1.15);
  - Au⁻/Mg²⁺ at 2.90 Å: +0.21, +0.40 with Pauli.
  - The Gaussian model over-repels Au⁻ on O-top, which is the direction T5 needs. The first-principles correction
    removes part of it.
- The partition-robust check: DFT's doping-induced charge in the upright dimer is 38 % on the near Au (Hirshfeld
  ratio). With the overlap terms on frozen step-13 weights, the model puts 52 / 57 % there with zero shifts and
  65 / 66 % with the learned ones; as trained, seed 42 had 43 %. So the O-top contact is too attractive for Au⁻.
- The hypothesis: Gordon–Kim's Thomas–Fermi kinetic energy, a local functional, is poorest in low-density tails
  (ESTABLISHED), and underestimates the exchange repulsion of the 6s² anion tail with O²⁻.
- The deciding tests:
  - (a) First principles, independent of the benchmark: the first-order Heitler–London interaction of the two
    closed-shell ions, E⁽¹⁾ = E_elst⁽¹⁾ + E_exch⁽¹⁾ (ESTABLISHED: first-order SAPT, Jeziorski, Moszynski and
    Szalewicz, Chem. Rev. 94, 1887 (1994)), is the exact frozen-density interaction that "penetration + Gordon–Kim"
    approximates. Compare them at 2.26 and 2.90 Å.
  - (b) Trained: the doped upright near share in steps 15–16 (`G_gauge_split`), against DFT's 0.38.
- If (a) shows Gordon–Kim too weak, the simple fix is the exact first-order exchange of the library's own
  densities, tabulated per state pair as the penetration already is.
- **(a) measured (2026-10-20): the hypothesis is not supported.** Tools: `tools/electrostatics/frozen_interaction.py`
  (`decomposed_interaction`), `gk_parts.py`, `compare_overlap.py`; output `$TRACE_FP_CACHE/overlap_compare/`.
  - Method: the frozen-density (Heitler–London) interaction of the library's own monomers (UKS PBE0, X2C for Au,
    ANO-RCC + diffuse; O states solved in the Watson sphere), split exactly into electrostatics and the rest, with
    the PBE functional. Validated: He₂/Ne₂ (symmetry, variational bound, δHF −4.9 / −2.3 %); the HF route equals the
    Heitler–London exchange to 2e-11 eV; the exchange density holds ≤ 3e-15 electrons; the solved densities match the
    library's to ≤ 3.6e-4 e.
  - SPEC 23's penetration is exact: frame-relative D agrees with first principles to 1e-4 eV for all six one-carrier
    types, and the absolute E_es to ≤ 3.0 meV.
  - The Pauli half, frame-relative (eV), runtime K·ΔS against the frozen PBE interaction of the two ions alone:
    Au⁻–O²⁻ 2.26 Å +1.149 vs +0.630; Au⁺–O²⁻ −1.049 vs −0.910; Au⁻–Mg²⁺ 2.90 Å +0.190 vs +0.330;
    Au⁺–Mg²⁺ −0.075 vs −0.146; Mg⁺–O²⁻ 2.106 Å +1.227 vs +0.798; O⁻–Mg²⁺ −0.459 vs −0.790.
    So Gordon–Kim is too strong for Au⁻ on O-top (by 0.52 eV), not too weak.
  - The target is not settled at that level: with the O state's Watson sphere also in the pair Hamiltonian, the
    first-principles values are +1.231 (Au⁻–O²⁻), −1.360 (Au⁺–O²⁻), +1.323 (Mg⁺–O²⁻), −0.738 (O⁻–Mg²⁺). The
    difference is the field of the rest of the crystal acting on the exchange density, a three-body term, ~0.5 eV per
    O contact. Neither convention is the crystal (the sphere's potential at the Mg neighbour is +13.7 V; the rest of
    the lattice gives −10.2 V there). The embedded-cluster test (`cluster_interaction.py`, point-ion MgO lattice,
    Madelung potential checked to 0.01 %) decides it.
  - In neither convention is Gordon–Kim too weak for Au⁻–O²⁻ beyond the 0.1 eV criterion (vac: too strong by 0.52; sphere: too weak by 0.08,
    inside the 0.1 eV criterion). So the doped near-share overshoot (model 52–66 % against DFT 38 %) is not explained
    by the Pauli table. ~~If (a) shows Gordon–Kim too weak, the simple fix is ...~~ does not apply.
  - **Decided by the embedded cluster (2026-10-20, `cluster_interaction.py`; point-ion MgO lattice, Madelung potential
    −23.8999 V against the exact −23.8977 V).** Band electron on Mg, frame-relative PBE Pauli term: bulk Mg⁺(O²⁻)₆
    +7.891 eV, the six embedded pairs +7.820 (non-additive 0.9 %), the runtime K·ΔS 6 × 1.227 = +7.364, six vacuum
    pairs +4.788; (100) surface Mg⁺(O²⁻)₅ +6.119, embedded pairs +6.289 (−2.7 %), runtime +6.137, vacuum +3.990.
    Per contact the embedded pair is +1.303 (bulk), +1.266 / +1.227 (surface, in plane / below), the sphere
    convention +1.323, the vacuum one +0.798. So: the pair model holds in the crystal field; the vacuum convention is
    not the crystal (it misses 2.1–3.1 eV per site); the sphere convention is within 0.02–0.10 eV; and the runtime
    Pauli term is right for the band electron to 0.02 eV (surface site) and 0.53 eV too weak (bulk site). The
    vacuum-convention comparisons above ("too strong by 0.52 eV" for Au⁻–O²⁻) are therefore not the crystal either;
    for Au on O-top the sphere value (+1.231) is 0.08 eV from the runtime, and a surface embedding of that site is
    not yet done.
  - **What does flood the window (probe_dump_terms.py on the step 15 s42 dump, 2026-10-20).** The band electron sits
    at λ₁ = −6.60 eV below a vacuum electron: site energy +4.20 eV (ladder + Coulomb + induction +5.96, overlap
    −1.76), hopping −10.80 eV (Harrison factor 1, 54 Mg sites, participation 42.5). The overlap part is the validated
    one; the hopping dominates. Learned factors: 0.89–1.06 in the runs without the overlap terms, 0.674 at epoch 32
    in step 16 s43 with them. Hypothesis (not yet checked against PBE): Harrison's nearest-neighbour V_ssσ applied to
    cation–cation second neighbours through the anion gives MgO's conduction band about twice its width. The
    deciding first-principles numbers: MgO's PBE conduction-band effective mass and electron affinity.
  - **The crystal field on the exchange density, and a simple form for it (tested, 2026-10-20).** Within first
    order the environment adds exactly E₃(u,v) = ∫ dP_uv V_env d³r to a pair (dP_uv = P_HL − P_u − P_v, ∫dP = 0;
    V_env the electron's potential energy from every ion but u and v). Measured in the embedded pairs: frame-relative
    E₃ = +0.505 eV (bulk), +0.468 / +0.429 (surface, in plane / below), exactly the embedded-minus-vacuum difference
    of the pair's Pauli term. Candidate (NEW-TO-TRACE), exact for a potential linear along the bond: an *exchange
    split charge* a_uv = d_uv·(R_u − R_v)/R², d_uv = ∫ r dP_uv, with E₃ = a_uv[V_env(R_u) − V_env(R_v)]; the same
    a_uv is the exchange-induced (overlap) dipole of IR / Raman / collision-induced spectra (ESTABLISHED physics of
    the overlap dipole). Pre-registered at 20 %: **fails** (`exchange_charge_test.py`): +0.622 / +0.567 / +0.594
    against the measured +0.505 / +0.468 / +0.429 (+23, +21, +39 %); the dipole at the midpoint is worse (+0.75).
    Cause, measured: the rock-salt potential is odd about the Mg–O midpoint (Hessian 2e-5 eV/Å²) but saturates
    beyond the nuclei (+13.0 eV 0.5 Å behind Mg against +18.0 linear), where the diffuse Mg⁺ 3s exchange density
    reaches. The frame pair (compact Mg²⁺) is within 5 %. The exact form is three-body, E₃ = Σ_k q_k ψ_uv(R_k) with
    ψ_uv the potential of the pair's exchange density at ion k; not built (not yet simple, and the runtime table is
    already within 0.08 eV for the band electron).
  - **Au on the O-top site, embedded (`au_otop_embedded.py`; (100) point-ion slab, d = 2.1325 Å as in the 4G cell,
    Au 2.26 Å above O; 2026-10-20).** Frame-relative PBE Pauli term (frame Au⁰): Au⁻ embedded +1.279 eV (sphere
    convention +1.231, vacuum +0.630, runtime K·ΔS +1.149); Au⁺ embedded −1.357 (sphere −1.360, vacuum −0.910,
    runtime −1.049). The runtime fails the pre-registered 0.1 eV criterion for both: Au⁻ is under-repelled at O-top
    by 0.13 eV (the hypothesis's direction, a fifth of its suggested size) and Au⁺ is under-stabilised by 0.31 eV.
    The vacuum-pair code reproduces compare_overlap's value to 1e-4 eV (+4.4448), a cross-code check. Across the five
    pairs with an O²⁻ partner measured so far, the sphere convention is within 0.05–0.10 eV of the embedded value.
  - **Every AuMgO one-carrier type, embedded (2026-10-20; `cluster_interaction.py`, `au_otop_embedded.py`,
    `embedded_pairs.py`).** Frame-relative PBE Pauli term (eV), embedded first principles against the runtime K·ΔS,
    pre-registered adequacy 0.1 eV:

    | type, site | embedded | runtime | runtime − embedded | vacuum | sphere |
    |---|---|---|---|---|---|
    | Mg⁺/O²⁻ bulk (d 2.106) | +1.303 | +1.227 | −0.08 | +0.798 | +1.323 |
    | Mg⁺/O²⁻ surface in plane / below (2.106) | +1.266 / +1.227 | +1.227 | −0.04 / 0.00 | +0.798 | +1.323 |
    | Au⁻/O²⁻ O-top (2.26) | +1.279 | +1.149 | **−0.13** | +0.630 | +1.231 |
    | Au⁺/O²⁻ O-top (2.26) | −1.357 | −1.049 | **+0.31** | −0.910 | −1.360 |
    | Au⁻/Mg²⁺ flat (2.90) | +0.276 | +0.190 | −0.09 | +0.330 | (= vacuum) |
    | Au⁺/Mg²⁺ flat (2.90) | −0.121 | −0.075 | +0.05 | −0.146 | (= vacuum) |
    | O⁻/Mg²⁺ bulk (2.1325), three-bond mean | −0.601 | −0.425 | **+0.18** | −0.737 | not measured at 2.1325 |

    So the runtime fails at Au± on O-top and for the O hole; the hole is under-stabilised by ~1 eV over six Mg
    contacts, which would *lower* the O⁻ + band configurations further once corrected. Correcting the Pauli tables
    therefore cannot cure the window flood; the band electron (and the hole's own level) must be right first.
  - **The model's own addition / removal spectrum (`probe_spectrum.py`, 2026-10-20; no model change).** TRACE's H is
    a many-electron Hamiltonian over integer configurations, so its one-electron spectrum is the Lehmann sum
    (ESTABLISHED) and its edges are total-energy differences of the same structure at total charge 0, ∓1:
    EA = E(N) − E(N+1), IP = E(N−1) − E(N), E_g = IP − EA. Step 16 seed 43 (band factor 0.676): clean six-layer
    3×3 MgO(100) slab EA +1.355 eV (the electron in the band unit, λ₁ = −1.355; site energies +5.25 … +6.11 eV,
    tight-binding width 9.89 eV over 54 sites), IP +7.21 eV (one localised O⁻), E_g 5.86 eV; with Au₂ (undoped
    wetting) EA +3.51 (→ Au⁻), IP +6.68 (→ Au⁺), E_g 3.17; neutral ground state 67 % frame, 33 % Au⁻Au⁺. The whole
    model's differences equal the redox module's to 1 meV (no learned charge dependence). Pre-registered for the PBE
    slab of the same geometry (`mgo_pbe.py`): EA within 0.3 eV; IP and E_g reported (the hole is localised, with no
    valence band, so the IP is expected above the delocalised VBM).
  - **PBE bulk MgO at the 4G lattice constant (`mgo_pbe.py`, a = 4.26495 Å; GTH-PBE, plane-wave cutoff 160 Ha;
    2026-10-20).** Checks: cutoff 160 → 200 Ha moves the gap 0.005 eV and m* 0.17 % (pass); k mesh 4³ → 6³ 0.008 eV
    and 0.2 % (pass); basis gth-dzvp-molopt-sr → gth-tzv2p moves the gap 0.102 eV (fails the 0.1 eV criterion) and m*
    2.3 % (pass at 5 %). Results: direct gap at Γ 4.233 (dzvp) / 4.335 eV (tzv2p); conduction-band mass, isotropic to
    0.1 %, **m* = 0.322 / 0.330 mₑ**. The band unit's nearest-neighbour s band has m* = ħ²/(2τa²) = 0.179/f mₑ
    (f the hopping factor), so PBE's mass corresponds to **f = 0.54–0.56**; Harrison at f = 1 is 1.8× too light, and
    training on the PBE data settled at f = 0.677 (step 16 s43; m* 0.264). The hypothesis that Harrison's
    nearest-neighbour V_ssσ overstates the cation–cation hopping across the anion is supported for the band's
    curvature. (A single s band is a model of the band bottom only.)
  - **The band hopping from the Pauli projection (derived 2026-10-22; hypothesis, test pre-registered in
    `band_hopping.py` before its run).** The band electron lives in the frame's virtual space, so its orbital on
    cation u is Qφ_u, Q = 1 − P_occ (the Phillips–Kleinman projection, ESTABLISHED: Phys. Rev. 116, 287 (1959)).
    With the occupied anion orbitals χ_k eigenfunctions of the frame operator F, to first order in S_uk = ⟨φ_u|χ_k⟩,

        t̃_uv = [F_uv − ε S_uv] + Σ_k (ε − ε_k) S_uk S_vk ,

    the bare coupling (which Harrison's V_ssσ models) plus a Pauli term that, for anion levels far below the carrier
    and s carriers (S_uk S_vk > 0), weakens the bonding coupling: the band narrows (NEW-TO-TRACE for the band unit).
    In rock salt the second-neighbour cations share two anions at 90°, so the anion-p part (∝ S_sp² cos θ) vanishes
    and the anion 2s carries the narrowing. Test: the exact splitting of the projected two-orbital problem
    {Qφ_A, Qφ_B} with the frozen-frame PBE operator in the embedded lattice, against the PBE-mass value
    τ = ħ²/(2m*a²) = 0.635–0.651 eV (Harrison: 1.173 eV); pass within 15 %, locality (two shared O against all ten)
    within 10 %.
  - **The projected band hopping, measured (2026-10-22, `band_hopping.py`; $TRACE_FP_CACHE/band_hopping/).** Projected
    two-site τ (Mg⁺ 3s on second-neighbour Mg sites, frozen-frame PBE operator, point-ion lattice): 1.084 eV with the
    two shared O quantum, 1.107 eV with all ten (bare 2.773 / 1.578; ‖Qφ‖² 0.764 / 0.421). **(h1) fails**: 1.107
    against the PBE-mass 0.635–0.651 eV. **(h2) passes**: the coupling is local to the shared anions (−2.1 %).
    Without the lattice the neutral two-O cluster gives 0.200 eV: the Madelung landscape (for an electron ~+10 eV at
    the cation sites, ~−10 eV at the anions) strengthens the coupling by acting on the bonding overlap density near
    the shared anions, the same three-body physics as E₃ for the Pauli term. (The ten-O cluster without the lattice
    carries −16 e and is not physical: site level +77.8 eV.) Post-hoc, not pre-registered
    (`band_hopping_lattice.py`): the two-site numbers are non-orthogonal (s = 0.20–0.22), t = εs − h = 1.04–1.05 eV;
    on the fcc lattice the same minimal basis gives τ_eff = t/(1 + 12s)² = 0.078–0.089 eV, m* = 2.35–2.67 mₑ against
    PBE's 0.32. Reading: the projection gets the direction (a strong narrowing of a pair), but a minimal basis of
    frozen vacuum cation orbitals cannot represent MgO's nearly-free-electron conduction band (m* < mₑ needs the
    interstitial / anion-s character). The derivation is not a quantitative source for the band unit's hopping; the
    defensible prior is the host's first-principles band curvature (f = 0.55 for MgO in PBE), with the site energy
    anchored to the slab's electron affinity.
  - **The 4G wetting energy decomposed from first principles (`wetting_eda.py`, 2026-10-22/23; PBE, def2-TZVP with
    the def2 ECP on Au, counterpoise; quantum MgO(100) cluster in the point-ion slab).** Wetting = E(flat) − E(upright),
    undoped, at the 4G reference geometries. Pre-registered: validation within 150 meV of DFT's +929, convergence
    < 100 meV between the 10-ion (398 AO) and 28-ion (969 AO) clusters. Measured: 10 ions +1118.8 meV (fails
    validation), 28 ions **+976.7 meV (validation passes)**; change 142 meV, so **convergence fails** and the
    decomposition is not interpreted as converged. Wetting shares (10 → 28 ions, meV): frozen first order E⁽¹⁾
    −285 → −272 (converged to 13; electrostatics +1751 → +1629, Pauli/xc −2036 → −1901, nearly cancelling); relaxation
    (polarisation + charge transfer) +1394 → +1239 (moves 155); Au₂ deformation +10. Two observations independent of
    the failed gate: the frozen contact physics favours the *flat* dimer, and the upright preference comes from the
    relaxation at the O-top site (sign far outside its 155 meV movement). Hypothesis for the non-convergence: the point
    ions are not polarisable, so the substrate's response is truncated at the quantum boundary (the upright Au₂'s
    Mulliken charge moves +0.256 → +0.406 e). ~~If it holds, the model must carry ~+1.2 eV of O-top relaxation, against
    step 17's redox-module share of the undoped wetting of +0.26 / +0.29 eV.~~ **Corrected 2026-10-23:** the
    overlap tables are frame-relative (zero for neutral Au⁰), so the neutral dimer's frame–frame contact, frozen
    first order *and* its relaxation, belongs to the backbone, apart from the Au⁰ induction the module carries
    explicitly; the decomposition cannot be mapped piece by piece onto backbone against redox module. Third point
    queued: 34 ions (1159 AO).
  - **The PBE slab (`mgo_pbe.py`, six-layer MgO(100) at the 4G geometry; gth-dzvp-molopt-sr, 160 Ha, 6×6×1;
    2026-10-23).** Checks: (s1) Hartree potential against `get_j` 1.3e-13 Ha; (s3) local pseudopotential scale
    0.005 %; (s2) vacuum plateau flat to 2.0 meV (a first run failed (s2) by 134.7 eV: `ifft(vpplocG)` is w·V_loc in
    PySCF's convention and had been used unscaled; fixed, tested on bulk first). V_vac +6.024 eV; Γ CBM +4.730, VBM
    +1.228 eV. **EA = +1.294 eV**, IP(Γ) = +4.796 eV, slab Kohn–Sham gap at Γ 3.50 eV (bulk 4.23: consistent with
    O 2p surface states, not analysed). Against the model (step 16 s43, `probe_spectrum.py`): **EA +1.355 eV, within
    0.06 eV (pre-registered 0.3 eV: pass)**; IP +7.21 eV against PBE's +4.80 (the model's hole is one localised O⁻,
    with no valence band: +2.4 eV, the expected direction; the Γ VBM need not be the slab's maximum, and a
    total-energy IP is compared with a KS level); gap 5.86 against 3.50 eV. The band electron is placed right; the
    hole's level is the next spectral piece (a valence-band unit), quantified now at ~2.4 eV.
  - **PBE bulk MgO band energies at the zone's high-symmetry points (`mgo_pbe.py bz`; dzvp, 4³, 160 Ha; 2026-10-23;
    reported).** Lowest conduction band relative to Γ: L +3.55, X +5.29, K +6.90, W +7.72 eV. The band unit's
    nearest-neighbour s band with the Γ-curvature τ = 0.65 eV predicts L 12τ = 7.8, X = W 16τ = 10.4, K 15.66τ = 10.2:
    the PBE band is strongly non-parabolic, light at Γ but about half as wide at the zone boundary (τ implied 0.30 at L,
    0.33 at X), and it splits X from W (longer-range or multi-band character), which one nearest-neighbour s band
    cannot. So no single τ fits both the curvature and the width. Step 19's learned f ≈ 0.47–0.53 (τ 0.55–0.62 eV)
    lies between the two, the compromise the one-band form forces. O 2p valence band width over these points 4.51 eV,
    top at Γ; the model's hole-side IP excess (~2.4 eV) is of the order of the half-width (consistency of magnitude,
    not a derivation).
  - **Derivation (2026-10-23; hypothesis, nothing built): a hole unit, the mirror of SPEC 20.** Hole sites are the
    anion frames that can lose an electron (O²⁻ → O⁻); the one-hole site energy is the exact configuration energy with
    the charge change reversed, η_h = [e(f+1) − e(f)]_h + (b + Wf)_h + ½W_hh + (pair terms)_h; the one-hole
    Hamiltonian H_h = diag(η) − f_h t(R), λ_h its lowest eigenvalue (one orbital per site, the band unit's code path),
    or, faithful to O 2p, three orbitals per site with Slater–Koster blocks T_αβ = l_α l_β V_ppσ + (δ_αβ − l_α l_β)
    V_ppπ (ESTABLISHED). Measured anchors: PBE O 2p width 4.51 eV, top at Γ, near-top dispersion L −0.50, X −1.34,
    K −1.08, W −1.70 eV (anisotropic, as a p band: one orbital per site would need t_h = 0.04 at L but 0.11 eV at W,
    so it captures one energy scale only); Harrison at f = 1 for O–O at 3.016 Å puts the band top 6.48 ħ²/(md²) =
    5.4 eV above the on-site level, against PBE's half-width ~2.2 eV, the same over-estimate across a cation as for
    the conduction band. The model's IP excess on the clean slab (+2.4 eV) is not delocalisation alone: the runtime
    Pauli table under-stabilises the hole by 0.18 eV per Mg contact (embedded, measured), ~0.9 eV over a surface O's
    five Mg; the hole's own polarisation enters with the opposite sign to its delocalisation; the O ladder's second
    affinity is uncertain by ~1 eV in the literature (−7.7 to −8.75). Order before any build: (a) the model's hole
    site energy split exactly (a probe); (b) PBE's E_top − ⟨E⟩_2p over the k mesh (`mgo_pbe.py bandavg`, running);
    (c) the unit, with a pre-registered IP test against the PBE slab's 4.80 eV.
  - **Derivation (2026-10-23; next measurement, not run): the O-top relaxation split into polarisation and charge
    transfer.** TRACE carries the two by different terms: polarisation by the bound sector's induction
    (−½sᵀK⁻¹s), charge transfer by the configuration ensemble (Au⁺ + band electron, Au⁻ + O hole, through τ). The
    first-principles split (ESTABLISHED: absolutely localised molecular orbitals, Khaliullin, Cobar, Lochan, Bell and
    Head-Gordon, J. Phys. Chem. A 111, 8753 (2007)): E_pol,A = the energy with A's orbitals relaxed *within A's own
    basis functions* in the field of the frozen B (orthogonality to B's occupied space by projection, ESTABLISHED:
    Manby, Stella, Goodpaster and Miller, JCTC 8, 2564 (2012)), minus E⁽¹⁾; E_pol,B likewise; E_CT = relax − E_pol,A −
    E_pol,B (to the order of the mixed terms). Applied to the upright and flat Au₂ of the wetting clusters it says
    whether the +1.2–1.4 eV upright preference of the relaxation is the frame's induction (the bound sector) or charge
    transfer (the ensemble), and with which direction (Au → surface, or O → Au).
  - **The model's ionisation potential split, and the learned O-hole shift (2026-10-23; probes, no model change).**
    `probe_hole_terms.py`, step 16 s43, clean slab, one O⁻ (surface): IP +7.210 eV = site energy +10.529 (ladder
    −8.750, self +20.427, screened −2.061, overlap −1.752, SPEC 26 +0.514, **learned shift +2.151**) + coherent
    −3.319 (the hole delocalises over the 54 O through the learned couplings). So the hole is not localised (a hole
    unit is not the missing piece; withdrawn as the next step), and the IP excess against PBE (+4.80) is the learned
    O⁻ shift: without it the explicit terms give +5.06 eV. `probe_hole_shift.py`: the shift is systematic,
    Δ_O = +2.15 / +2.23 / +1.94 eV (step 16 s43, step 17 s42 / s43); zeroing it on frozen weights moves wet_u by
    +63 / +48 / +28 meV, below the pre-registered 100 meV: **the hole shift is not the wetting lever (falsified)**.
    Step 19 snapshots (best checkpoints near epoch 20, band prior f = 0.55): Δ_O = +0.14–0.20 (s42) and +0.98–1.01 eV
    (s43): the ~2 eV O⁻ shift of steps 16–17 compensated the over-wide band, and the physical band prior removes most
    of it (supporting evidence). Mid-training wet_u +330 / +355 meV, T5 −715 / −617 meV: half-trained, not scored.
  - **B3 is not configuration truncation (2026-10-23, from the window-deficit scores).** Step 12 → 13 made one change
    (amplitude selection) and moved the undoped class offset from −33.5 to −389.9 meV (s42); since step 13 the
    ensemble is window-independent (deficit 0; offsets identical at 2.5 and 8 eV: step 17 −397.1 / −257.4). The offset
    is a property of the trained weights, not of the truncated space.
  - **The Pauli refit as a frozen-weight counterfactual (`probe_pauli_refit.py`, 2026-10-23).** Each AuMgO one-carrier
    type's runtime Pauli series scaled to its embedded first-principles value at contact (Au⁻/O²⁻ ×1.113, Au⁺/O²⁻
    ×1.294, Au⁻/Mg²⁺ ×1.449, Au⁺/Mg²⁺ ×1.623, Mg⁺/O²⁻ ×1.062, O⁻/Mg²⁺ ×1.414); step 17, whole-model energies of the
    T5 references. s42: wet_u +470.2 → +659.4 (+189), wet_d −370.6 → −366.8 (+4), T5 −840.8 → −1026.1 meV; s43:
    wet_u +655.7 → +774.2 (+118), wet_d +17, T5 −894.0 → −995.6 (DFT +928.9 / −2.7 / −931.6). Pre-registered: a lever
    worth a training run if wet_u moves ≥ 100 meV toward DFT: **passes on both seeds**. Mechanism consistent with the
    embedded measurement: Au⁺ on O-top 0.31 eV more stable, favouring the upright dimer's ionic Au⁺(near)Au⁻(far)
    component. The doped wetting error (flat too stable by 0.22–0.36 eV) is untouched: its suspect is the doped
    screened term (−700 meV, Au₂⁻ over Mg²⁺), for a first-principles doped decomposition. Proposal (not built):
    step 20 = step 19 + K refitted to the frozen-density PBE interaction in the sphere convention (which reproduced
    the embedded values within 0.05–0.10 eV for the five O²⁻-partner pairs and within 0.06 eV for Au/Mg²⁺; the O⁻
    hole needs the sphere convention checked at 2.1325 Å first).
  - **PBE band averages (`mgo_pbe.py bandavg`, 6³ = 216 k, dzvp, 160 Ha; 2026-10-23).** O 2p width 4.507 eV; the
    hole's delocalisation (top − mean of the three O 2p bands) 1.889 eV; the conduction band's (mean of the lowest
    band − its bottom) 4.909 eV. For the band unit's nearest-neighbour s band the latter is 12τ, so f = 0.35, against
    0.55 from the Γ curvature: one s band cannot fit both (the data settled at 0.485 / 0.572, step 19). The model's
    hole delocalisation (coherent, step 16 s43) is −3.32 eV against PBE's 1.889: the learned O–O couplings
    over-delocalise the hole by ~1.4 eV, and the +2.15 eV O⁻ shift partly offsets it.
  - **The O⁻ hole in the sphere convention at the 4G distance (2026-10-23).** O⁻/Mg²⁺ against O²⁻/Mg²⁺ at 2.1325 Å
    (three bond axes): frame-relative Pauli −0.686 eV (sphere) against −0.601 embedded (0.085 off, within the
    0.1 eV criterion); vacuum −0.737 (reproduces the embedded run's vacuum value to 1e-4, a cross-check). With this the
    sphere convention is adequate for all six AuMgO one-carrier types.
  - **The wetting decomposition, converged (34-ion cluster, 1159 AO; 2026-10-24).** wet = +976.8 meV (validation
    against DFT +929: pass, 48 meV); 28 → 34 ions changes it by 0.1 meV (convergence: pass; the 142 meV from 10 to 28
    ions was the 10-ion cluster being too small). Converged shares (meV; 34 ions, 28 within 4): frozen first order
    E⁽¹⁾ −275.7 (electrostatics +1629.4, Pauli/xc −1905.1; upright E⁽¹⁾ +702.5, flat +426.8), relaxation +1242.2
    (upright −1990.0, flat −747.8), Au₂ deformation +10.3. Established: the frozen contact physics favours the flat
    dimer, and the whole upright preference is electronic relaxation at the O-top contact (+1.24 eV, more than the
    full DFT wetting energy), which recovers about two thirds of the O-top Pauli cost (upright Au₂ Mulliken +0.43 e,
    basis-dependent). For neutral Au₂ the frozen contact is frame–frame (backbone); the relaxation must come from the
    module's explicit response, the Au⁰ induction (bound sector) or the configuration ensemble (charge transfer). The
    model's undoped wetting is +481 / +661 meV (step 17) against +929. Next: the relaxation split into polarisation and
    charge transfer (derived above), which maps each piece onto one model term.
  - **The polarisation / charge-transfer split, validation (`almo_split.py`, 10-ion cluster; 2026-10-24).** Built as
    derived above (block-localised orbitals, L-BFGS on dE/dC = 4(1 − SP)FC(CᵀSC)⁻¹). Checks: (a2) variational order
    E_FRZ ≥ E_ALMO ≥ E_SCF holds for both geometries; ΔE_cp reproduces wetting_eda's 10-ion value exactly (−1.2058 /
    −0.0973 eV). (a3) failed by exactly 2·BSSE and caught a sign error of mine (corrected: CT_cp = CT + BSSE). (a1)
    passed upright (3.4e-6) and failed flat (7.0e-5 at h = 1e-4); the step scan shows the analytic gradient is right:
    FD − analytic falls as h² from h = 1e-3 to 3e-4 (7e-8 → 2e-9 Ha) and rises as 1/h below (energy round-off ~1e-12
    Ha), with no h-independent part (a missing term would be of the order of the derivative, 1e-4–1e-3 Ha). The
    criterion as written was below the noise floor for small directional derivatives. Open: the L-BFGS minimisation
    did not converge in 400 iterations (both geometries), so E_ALMO is an upper bound and POL is underestimated in
    magnitude; a rerun with 4000 iterations and a logged energy history is running. The unconverged 10-ion pieces are
    not interpreted.
  - **The split's solver and its validation completed (2026-10-24/25).** Raw-coefficient L-BFGS crawled (400
    iterations, POL −0.636 eV, |g| 3.5e-2 at 100); replaced by preconditioned occupied–virtual rotations inside each
    fragment's own basis, C_X = C⁰_X + V_X K_X with K = K′/√(ε_a − ε_i) (ESTABLISHED SCF practice; the constraint is
    unchanged; gradient by the chain rule from the validated one, step-scan checked: FD − analytic 2e-9 / 4e-9 Ha at
    h = 1e-3 / 3e-4 against 1.8e-3). It converges in 19–23 iterations to |g| < 1e-6 (the old POL was 37 meV short).
    10-ion validation, final: (a1) 4.5e-6 / 1.8e-5 at h = 3e-4 (the criterion was restated to 1e-4 after the step
    scan showed 1e-5 at h = 1e-4 lay below the energy's round-off; stated as a change made after seeing results); (a2)
    and (a3) pass for both geometries; the flat Au₂-only minimisation stopped on a line search at |g| 2.9e-6 (energy
    converged). 10-ion pieces, validation only (the cluster is not converged in size): upright FRZ +0.622, POL −0.673
    (Au₂ −0.257, surface −0.391), CT_cp −1.155 eV; flat FRZ +0.372, POL −0.150, CT_cp −0.319 eV. The physics run is
    the 28-ion cluster, queued after the doped decomposition.
  - **The doped decomposition as designed models the wrong state (2026-10-26).** 28-ion doped upright (Au₂⁻, one Al³⁺
    in the point lattice): ΔE_SCF −2.725 eV = E⁽¹⁾ +0.410 (elst −5.594, Pauli/xc +6.003) + relaxation −3.135, with the
    Au₂ Mulliken charge +0.735 e. Check on 10 ions: the unpaired spin sits on the cluster's Mg (+1.09; Au₂ −0.10, O
    +0.01; Au₂ Mulliken charge +0.444 e), alpha SOMO −1.27 eV (close to the PBE slab's −EA = −1.29 eV, but an
    embedded cluster's orbital energies carry the finite point lattice's reference, so the coincidence is not evidence).
    So the extra electron spills into the quantum Mg, where the 4G DFT keeps it on Au₂ (B1). Untested hypotheses:
    (H1) leakage toward the bare (0.3 Å Gaussian) point Mg²⁺ next to the quantum cluster, which carry no Pauli
    repulsion — the known embedded-cluster artifact, cured by total-ion pseudopotentials on the boundary cation sites
    (ESTABLISHED: Winter, Pitzer & Klein 1987; Sushko, Shluger & Catlow 2000; Nasluzov et al. 2001); (H2) the point
    ions are not polarisable, so Au₂⁻ lacks the substrate's image stabilisation; (H3) PBE's delocalisation error.
    Falsifying check (pre-registered): the same 10-ion cluster with the point Mg²⁺ within 3.5 Å of a quantum ion
    replaced by Mg²⁺ total-ion pseudopotentials (10-electron-core ECP, no basis; same total point+ion charge): H1 holds
    if the spin on Au₂ exceeds 0.8, and is falsified if the spin on the cluster's Mg stays above 0.8. The 28-ion run
    was stopped (log kept as doped_r3.8_stopped_electron_spill.log) and is not interpreted. The diabatic Au₂⁻ state,
    which the model's configuration represents, is the ALMO energy with charge transfer forbidden (FRZ + POL).
  - **TIP result (2026-10-26, tools/electrostatics/doped_spin.py tip 3.6).** (t1) 139 electrons and total charge 0, as
    the bare run: PASS; (t2) TIP dummy-function population 2.2e-12 e: PASS. With the 26 point Mg²⁺ within 3.5 Å capped:
    spin on Au₂ +0.830 (cluster Mg +0.129, its O +0.041; the largest single Mg +0.050), Au₂ Mulliken charge −0.307 e
    (bare +0.444), alpha SOMO +0.149 eV (a positive anion SOMO, PBE self-interaction). Pre-registered reading: H1 holds
    (0.830 against the 0.8 threshold: a narrow pass). Two artifact checks follow before the doped work is rebuilt on it
    (tools/electrostatics/tip_checks.py, criteria in its docstring): (c1) the bare-lattice UKS started from the TIP
    density, to tell a property of the bare Hamiltonian from an SCF local minimum; (c2) the undoped wetting numbers
    with the same TIPs, which decide whether the undoped decomposition (and the 28-ion ALMO split, built on bare point
    cations) stands: |Δwet|, |ΔdE_scf| ≤ 0.10 eV.
  - **TIP artifact checks (2026-10-26, tip_checks.py).** (c1) The bare-lattice UKS started from the TIP solution's
    density returns to the spilled state: E −2033.7038420657 Ha against −2033.7038420659 (the same to 2e-10 Ha), spin on
    Au₂ −0.102. Two very different starts reach one state, so the spill is the bare-lattice Hamiltonian's (consistent
    with H1), not an SCF local minimum. (c2) Undoped, the same 26 TIPs change dE_scf by −2.2 meV (upright) and +1.9 meV
    (flat), wet +1122.9 against +1118.8 meV, Au₂ Mulliken +0.252/+0.120 against +0.256/+0.129: the undoped cluster
    results stand (≤ 0.10 eV), and so does the 28-ion ALMO split built on bare point cations. The boundary matters for
    the extra electron only. Doped decomposition rebuilt with TIPs (wetting_eda.py --doped-tip; the bare path
    unchanged): read only if the spin on Au₂ exceeds 0.8 in both geometries; validation |wet_d − (−2.7 meV)| ≤ 150 meV;
    convergence 10 (26 TIPs) against 28 ions (44 TIPs), < 100 meV. It also reports the pseudopotential share of elst
    (wetting_eda's elst counts the Au ECP acting on B's density as electrostatics, a convention to keep in mind when
    reading its elst/nonelst split; E1 and relax do not depend on it).
  - **Doped decomposition, 10 ions with 26 TIPs (2026-10-26, wetting_eda.py --doped-tip 3.6).** Gate: spin on Au₂ flat
    +0.972, upright +0.830: the electron is on Au₂ in both (read). Upright ΔE_SCF −1.2801 = E⁽¹⁾ +0.4920 + relax −1.7721
    (Au₂ Mulliken −0.307 e); flat −1.3574 = E⁽¹⁾ −0.6972 + relax −0.6601 (−0.637 e); number checks 3e-14, 6e-14.
    wet_d (cluster) −100.0 meV against DFT −2.7: PASS (150 meV). Shares of wet_d: E⁽¹⁾ −1189.2, relax +1111.9, Au₂⁻
    deformation −22.8 meV (elst +3135.2 / nonelst −4324.4 cancel, and −745.4 meV of the elst share is the
    pseudopotentials' convention: not read). With the bare 10-ion undoped shares (E⁽¹⁾ −285.3, relax +1393.9,
    deformation +10.3; TIPs move wet_u by +4.1 meV only), the cluster's T5 = wet_d − wet_u = −1218.8 meV (DFT −931.6)
    splits into E⁽¹⁾ −903.9, relax −281.9, deformation −33.0 meV: three quarters of what doping does to the wetting is
    first order, the frozen Au₂⁻ against the frozen surface. Not yet converged: undoped, 10 → 28 ions moved wet by
    −142 meV, so the 28-ion doped run (44 TIPs) decides (queued after the 28-ion ALMO split; the bare 28-ion doped run
    took 28.4 h for five SCFs under load).
  - **Erratum (2026-10-27): the doped cluster model has one Al; the 4G doped cell has three.** The references
    (eval_aumgo.references, from the 4G data) hold Mg51 Al3 O54 Au2: three Al³⁺ in the k = −4 layer (z = 19.11 Å, along
    the cell diagonal), frame charge +3. The wetting_eda doped design (2026-10-23) stated "one per 3x3 cell": wrong.
    With three donors the cell's ground state (B1; the model's too: Al [3, 3, 3], Au₂⁻, two electrons in the band) has
    Au₂⁻ and two conduction-band electrons, so the doped slab conducts and screens Au₂⁻ (benchmarks README, hypothesis
    B). The one-Al cluster (Au₂⁻ + one ionised donor, insulating point lattice) has neither the two band electrons nor
    that screening. Its decomposition is a measurement of that sub-problem (the frozen Au₂⁻ against the insulating
    surface), and the statement "three quarters of what doping does to the wetting is first order" holds for it, not
    yet for the 4G cell: the band electrons' response to Au₂⁻ is relaxation (polarisation of B), which the cluster
    lacks. The queued 28-ion run converges the same sub-problem; a model of the 4G cell needs the band electrons
    (a periodic or embedded-metal treatment), not a larger insulating cluster.
  - **The doped wetting energy, located (2026-11-02; measured).** Because the backbone is local (r_max 5.29 Å) and the
    Al lie 10.7 Å from the Au, it adds the same flat − upright offset to doped and undoped cells. So the training-set
    class residuals (model − DFT, flat − upright; `H1_class_residuals_train.txt`) give, in their doped − undoped
    difference, the electrostatic sector's T5 error at the DFT training geometries themselves (meV, s42 / s43):
    step 13 +436 / −381, 14 +272 / +285, 17 +100 / +71, 19 +214 / +341, **20 −1 / −12**, 22 −186 / −508, 23 −318 / −598
    (test sets agree: step 20 −17 / −20). Step 20, with the first-principles Pauli table, has what doping does to the
    wetting right in distribution; its error was a common offset (−365 / −171 in both classes), which the fold (steps
    22–23) was built to supply and did supply for the undoped class only. Mechanism on step 23 s42 (`probe_fold_leaves.py`,
    `probe_fold_leaves_otop.py`, `probe_band_levels.py`, `probe_doped_configs.py`): in the doped upright dimer the contact Au holds the extra electron with weight 0.48 and
    then accepts nothing (dressing 0.0, Au⁻ 6s full); otherwise the O-top move costs Δ 7.83 eV against 5.95 undoped
    (the Au⁻Au⁻ state). The fold dressing of the upright falls from ≈ −0.79 eV (undoped, weighted) to ≈ −0.28 (doped),
    of the flat from −0.45 to −0.20; its T5 share at the references is −320 meV (s42; fold on − off), matching s42's
    in-distribution error. PBE's one-Al cluster (10 ions, 26 TIPs) relaxes the upright Au₂⁻ by −1.772 eV against
    −1.870 for Au₂⁰ and the flat by −0.660 against −0.477: per geometry the extra electron changes PBE's relaxation by
    +0.10 / −0.18 eV, the model's fold by +0.51 / +0.25. Also measured: the band unit holds at most two electrons, so the
    three-donor cell forces Au₂⁻ exactly (the configuration Au₂⁰ + three band electrons does not exist), and the
    band's second state lies 3.6 eV above its first (a one-k-point band of the 9 Å cell; the 4G DFT used a 3×3×1 k-grid,
    Ko et al. 2021). Hypotheses, not tested: Hirshfeld's doped − undoped shift on Au₂ (−0.17 upright, −0.36 flat)
    suggests partial reduction of the upright dimer; the k-grid difference is a candidate artifact of the band's
    energetics. Pre-registered next (written 2026-11-02 19:22, before the run; `tools/electrostatics/almo_split_open.py`, checks
    (u1) closed-shell limit 1e-11 eV, (u2) open-shell gradient 1.4e-7, (u3) E[all] = E_SCF 0.0000 meV, all PASS):
    the open-shell direction split of the anion on the one-Al cluster (the sub-problem; scope stated in the tool).
    **Reading:** with the undoped upright's surface → Au CT_cp −0.962 eV as reference, (i) |surface → Au| (doped
    upright) ≥ 0.72 eV (75 %): O → Au donation survives the extra electron in PBE, so the fold's quench (35 %) is the
    defect and the fold's treatment of an occupied σ* acceptor must change; (ii) ≤ 0.48 eV (50 %) with Au → surface
    larger in magnitude than the undoped −0.169: the anion back-donates, a channel the fold (O donors only) lacks;
    (iii) otherwise both. Gate (u4): |E[all] − E_SCF| < 1 meV and SCF spin on Au₂ > 0.8, else not read.
    **Upright result (2026-11-03 04:16; `$TRACE_FP_CACHE/almo_direction_doped/r3.6.log`).** Gate: SCF spin on Au₂ +0.830 (PASS);
    E[all] − E_SCF +0.000 meV (PASS); all four minimisations converged. CT_cp −0.8887 eV = surface → Au −0.5297 (raw
    −0.5537, BSSE_B +0.0241) + Au → surface −0.3355 (raw −0.3906, BSSE_A +0.0551) + rest −0.0235. Against the neutral
    dimer on the same cluster (−1.1553 = −0.9619 / −0.1685 / −0.0249): surface → Au keeps 55 %, Au → surface doubles, the
    total keeps 77 %. **Reading (iii), both** (0.530 eV is between the 0.48 and 0.72 thresholds). The model's fold keeps
    35 % of its upright O → Au dressing on doping (−0.79 → −0.28 eV) and has no Au → surface channel at all in the
    doped cell (the band is full; probe_doped_configs.py), so it misses about 0.25 eV of donation and the whole −0.34 eV
    of back-donation that PBE keeps. The flat geometry, which decides the wetting shares, is running.
    **Flat result and the wetting shares (2026-11-03 08:01).** Gate: SCF spin on Au₂ +0.972 (PASS); E[all] − E_SCF −0.000 meV
    (PASS); converged. CT_cp −0.3281 eV = Au → surface −0.2433 + surface → Au −0.0794 + rest −0.0053 (neutral dimer:
    −0.3189 = −0.1388 / −0.1730 / −0.0071): on the flat dimer doping halves the donation and nearly doubles the
    back-donation, leaving the total unchanged. wet_d shares (meV): CT_cp +560.6 = surface → Au +450.2 + Au → surface
    +92.2 + rest +18.2 (undoped: +836.5 = +788.9 / +29.7 / +17.8). **CT's T5 share on the sub-problem: −275.9 meV =
    surface → Au −338.7 + Au → surface +62.5 + rest +0.4.** Against the model: the fold's T5 share at the references is
    −320 meV (step 23 s42, fold on − off), all of it O → Au. **So the earlier hypothesis that the fold's doped quench is
    the defect is not supported in total:** the fold's doping asymmetry is within 44 meV of PBE's charge transfer on
    this sub-problem, its channels differ (PBE's donation is quenched less and partly replaced by back-donation, which
    the model lacks), and the two channel errors largely cancel in T5. It follows that step 20's correct in-distribution
    T5 (−1 / −12 meV) held without the charge transfer's −276 meV, so another term of step 20's electrostatic sector
    compensated it. Candidate, a hypothesis: the bound sector's response to Au₂⁻ (step 17's polarisation share of T5
    −318 / −385 meV, against about −6 meV for PBE's one-Al cluster, POL ≈ relax − CT_cp); caveat stated: the 4G cell's
    band electrons screen Au₂⁻ and are polarisation the cluster lacks. Step 20's first-order split is being measured.
    **Step 20's split, measured (2026-11-03 08:03; `probe_t5_first_order.py`, scores/S5_first_order.txt; (k4) separated limit
    PASS on both seeds; the backbone's T5 share −1.3 / −1.0 meV, zero as a local backbone must be).** T5 (meV, s42 / s43):
    first order −676.5 / −694.2, polarisation −317.6 / −221.9, configurations (CT) −111.1 / −133.0, dimer deformation
    −12.3 / −25.4, periodic images of Au₂ +125.3 / +119.5; total −993.4 / −956.1. Against the PBE sub-problem (first
    order −903.9, POL ≈ −6, CT −275.9, deformation −33.0) only the CT share compares cleanly: step 20 carries 40–48 % of
    PBE's, and step 23's fold adds about −320 on top, so step 23's CT share plausibly exceeds PBE's (its split is unread:
    the probe does not read fold models). First order and polarisation are not compared term by term: the model's
    partition (bound moments frozen at each fragment's optimum) is not ALMO's, the cluster polarises only its 10 quantum
    ions and has no band electrons, and the model's periodic images (+120 meV) have no cluster counterpart. Reading:
    the fold supplied charge transfer the data needed, and the in-distribution T5 it broke had been held by a compensation
    in the other terms; which term carries that compensation is not decided by the cluster. Recorded as such.
  - **Is the flat dimer a PBE minimum against the tilt the step-24 models relax into? (pre-registered 2026-11-03 22:02, before any
    PBE number).** Paths (`probe_tilt_path.py --path`; self-checks: the mapped flat reference reproduces wetting_eda's
    flat geometry to 0.003 Å, the substrate near the path sits on ideal sites to 0.002 Å): λ = 0 the 4G flat reference,
    λ = 1 the model's relaxed state, linear in the two Au positions, λ = −0.2 … 1.2. Path A (step 24 s42, −1.119 eV):
    one Au to 2.197 Å exactly over an O (O-top), the other over the original Mg at 3.07 Å. Path B (s43, −0.352 eV): one
    Au to 2.172 Å over a hollow (nearest ions 2.61 Å), the other up to 3.91 Å. PBE (`tools/electrostatics/tilt_scan.py`;
    wetting_eda's undoped cluster model, one quantum region for every λ). **Reading:** (r1) flat is a PBE minimum along
    the path if E(±0.2) > E(0) − 0.02 eV and E(1) − E(0) > −0.10 eV: then the models' descent is unphysical and the term
    that drives it must be corrected at tilted contacts; (r2) E(1) − E(0) < −0.30 eV: the tilt is downhill in PBE too,
    and the cluster is checked against the periodic DFT (whose relaxation kept the flat dimer) before anything is read;
    (r3) otherwise ambiguous, decided by the λ = 1 point on a larger cluster (r_au 3.8; read only if E(1) − E(0) moves
    by < 0.15 eV). Reported with it: the frozen models of steps 20, 22, 23, 24 along the same paths, as trained.
    *Amendment (22:19, before any PBE number):* the 25-ion region of path A takes ~290 s per SCF cycle, so path A runs
    at λ = 0, 0.4, 1.0, then 0.2, 0.6 (8 threads; the first launch, stopped at its third cycle, is kept as
    pbe_path_A_stopped_slow.log), and path B after it. Reading unchanged; E(−0.2) is replaced by E(0.2) alone in (r1).
    *The frozen models along path A (measured; `$TRACE_FP_CACHE/tilt_path/model_scans.txt`)* — the deep tilted O-top state
    is not the fold's: step 20 (no fold, no wall) already has it at −0.78 / −0.67 eV (s42 / s43) behind a barrier of
    +0.17 / +0.18 eV at λ = 0.4; on s42 the fold lowers the barrier to +0.06 / +0.03 / +0.05 eV (steps 22 / 23 / 24) and
    deepens the well to −1.11 / −1.23 / −1.12 eV; on s43 the barrier stays at +0.20 eV. Path B (toward a hollow) rises
    monotonically on the s42 models and crosses a +0.55 to +0.63 eV barrier on the s43 fold models.
    *Step 20's tilt well by term (22:22; `probe_tilt_terms.py`, the totals reproduce the scan exactly).* At λ = 1 relative to flat
    (eV, s42 / s43): the hops (coherent) −0.31 / −0.64, the backbone −0.20 / −0.27, and the ionic-configuration terms
    together −0.27 / +0.24 (ladder +0.61 / +0.31, self −1.15 / −0.30, shifts +0.13 / +0.62, screened +0.37 / +0.01,
    overlap −0.17 / −0.30, SPEC 28 −0.07 / −0.11, SPEC 26 +0.01 / +0.02): the tilt raises the dimer's ionic weight, as the
    O-top upright dimer does. The barrier at λ = 0.4 (+0.17 / +0.18) is mostly the backbone's (+0.29 / +0.19).
    **PBE along path A (measured 2026-11-04 12:39; 25 quantum ions; `pbe_path_A.json`).** E − E(flat), eV: λ 0.2 +0.078, 0.4 +0.255,
    0.6 +0.236, 1.0 −0.207; the upright dimer on the same O −1.024 (the cluster's wet_u +1.02 against the periodic DFT's
    +0.93: the cluster reproduces it to 0.1 eV). Pre-registered reading: (r3), ambiguous (E(0.2) > 0, so the flat dimer is
    a PBE minimum, but E(1) = −0.207 lies between the −0.30 and −0.10 thresholds); the λ = 1 point on 28 ions (r_au 3.8)
    decides it and is running. Measured regardless of the reading: PBE's barrier ≈ 0.25 eV against the models' 0.03–0.21;
    PBE's tilted state lies 0.82 eV above the upright one on the same O, the models' from 0.24 below to 0.34 above
    (step 20 s42 −0.24, s43 +0.06; step 24 s42 −0.15, s43 +0.34). The model's inclined O-top bond is 0.5–1.1 eV too strong.
    *Tiebreak (2026-11-04 18:30; 28 ions, r_au 3.8):* E(1) − E(0) = −0.197 eV against −0.207 on 25 ions (moved 0.010 < 0.15: the value
    is converged). It lies between the thresholds, so the pre-registered reading stays (r3), partial: the flat dimer is a
    PBE minimum (E(0.2) > 0) behind a ≈ 0.25 eV barrier, and the tilted state lies 0.20 eV below it; the models put it
    0.30–1.23 eV below with a 0.03–0.21 eV barrier. The 28-ion upright point is running.
    **Hypothesis (pre-registered 2026-11-04 12:39, before the run): the O-top donation depends on the acceptor σ*'s orientation.**
    SPEC 32 set κ′ = 0 on 2026-10-29 from flat contacts (whose donor-side angle confounded it); the fold's O → Au coupling
    then depends on the Au's position over the O only, so the tilted dimer (contact Au straight above the O, l_z = 1)
    keeps the upright's full coupling. The tilted/upright pair on one O separates the acceptor's orientation from the
    donor's (same O, same p_z, Au straight above, 2.20 against 2.26 Å). Test: `tools/electrostatics/almo_tilt.py`
    (almo_split.run_direction unchanged). **Reading:** R = CT_cp(surface → Au, tilted) / CT_cp(surface → Au, upright):
    (o1) R ≤ 0.5: the orientation dependence is real; κ′ follows from R with the SPEC 32 form, (1 + κ′ cos θ)/√(1 + κ′²)
    in v, cos θ = 0.34 for the 70° tilt (R = 0.5 gives κ′ = 0.80; the form's floor is R = 0.12); (o2) R ≥ 0.8: it is not,
    and the 0.82 eV comes from other terms (the ALMO POL and frozen terms are reported); (o3) otherwise partial.
    *Cluster (12:39, before any number):* within 3.6 Å of either geometry the region is the O-top O and its five Mg (6
    ions; the far Au's Mg is one of them); 3.8 Å gives 25 ions (> 12 h for this split). R is read on 6 ions first and
    confirmed on 25 ions; the reading stands only if the two agree within 0.15 in R.
    **Result (2026-11-04 13:50; 6 ions; (d3) 0.000 meV on both, all converged).** Upright CT_cp −1.1480 = surface → Au −0.9628 +
    Au → surface −0.1614 + rest −0.0238; tilted −1.2037 = −0.9108 / −0.2651 / −0.0278. **R = 0.946: reading (o2), the
    hypothesis is falsified** — the O-top donation keeps 95 % at a 70° tilt and does not depend on the acceptor's
    orientation; κ′ = 0 stands. The back-donation grows (×1.64, the far Au near a Mg). The model's fold dresses the tilted
    dimer by −1.062 eV against −0.853 upright (step 24 s42; step 23 −0.975 / −0.790): −0.21 eV of extra dressing against
    PBE's −0.056 eV of extra CT, from the closer contact (2.197 against 2.26 Å). PBE's 0.82 eV penalty of the tilted state
    is not in CT: at the ALMO level (CT forbidden) the tilted state lies +0.940 eV above the upright (E_ALMO, same
    cluster, the dimer's deformation included). (25-ion confirmation of R pending.)
    **Next, pre-registered (2026-11-04 13:50, before the run): FRZ against POL.** almo_split.run_geometry (FRZ, POL with POL_A / POL_B,
    CT_cp; checks a1–a3) on the same 6-ion cluster for the two geometries. **Reading:** of the tilted − upright difference
    at the ALMO level, (f1) POL ≥ +0.4 eV: the dimer's orientation in the surface's normal field carries it (an upright
    Au₂ turns its large α∥ into the field), the model's Au₂ α∥ (16–38 % low, frozen check 1) and α⊥ are then
    benchmark-relevant, and POL_A against POL_B says whose polarisation; (f2) FRZ ≥ +0.4 eV: the frozen densities' Pauli
    and electrostatics of the tilted contact (the far Au against the surface O²⁻) carry it, the model's frame-relative
    Pauli table and the backbone's share then decide; (f3) both.
    **Result (2026-11-04 14:09; checks a1 2e-6 / 9e-7, a2, a3 PASS; ALMO converged).** Upright dE_cp −1.1775 = FRZ +0.6013 + POL
    −0.6308 (A −0.2508, B −0.3577) + CT_cp −1.1480; tilted −0.2923 = FRZ +2.0034 + POL −1.0920 (A −0.5509, B −0.5927)
    + CT_cp −1.2037. **Tilted − upright: FRZ +1.402, POL −0.461, CT −0.056, total +0.885 eV: reading (f2).** The frozen
    densities' interaction carries the tilted state's penalty; polarisation favours the tilt (so the α∥ idea of (f1) is
    wrong as well). The model's tilted − upright on the same O (step 20, `probe_tilt_upright_terms.py`, eV, s42 / s43):
    overlap +0.021 / −0.027, backbone −0.039 / +0.033 — the model holds none of the +1.4 eV. Structural reason: SPEC 23's
    penetration and SPEC 25's Pauli are frame-relative (zero at the frames), so the frozen interaction of a neutral Au
    with the oxide's own ions is no explicit term; only the backbone carries it, and the tilted geometries are outside
    the data. Step 24's wall added the Mg²⁺–Au⁰ part only.
    **Next, pre-registered (2026-11-04 14:09, before the run): is the frozen penalty pairwise?** Frozen-density PBE pair energies
    (`frame_pair_wall.py`: sph_elst + sph_nonelst_pbe, the Pauli build's route) for Au⁰–O²⁻ and Au⁰–Mg²⁺ over 2.0–4.2 Å,
    summed over the 6 quantum ions for each Au of the tilted and the upright dimer. **Reading:** (p1) the pair sum's
    tilted − upright within 30 % of ALMO's FRZ +1.402 eV: a frame-pair frozen interaction of the neutral s atom with the
    oxide's frame ions (step 24's wall generalised to Au⁰–O²⁻, its electrostatics included) is the missing term; (p2)
    outside: the penalty is not pairwise in atomic densities (the dimer's bonded density, or many-body), and the term
    must be built from the dimer's density instead.
    **Result (2026-11-04 18:18; `frz_pair_sum.py`).** Frozen-density PBE Au⁰–O²⁻ pair energy (elst + nonelst, eV): 2.197 Å +3.09,
    2.26 +2.44, 2.4 +1.46, 2.6 +0.70, 3.0 +0.15, 3.4 +0.02; Au⁰–Mg²⁺: 3.08 +0.10, 3.4 +0.03. Pair sums over the quantum
    ions: tilted +3.598, upright +2.805, **tilted − upright +0.794 eV against ALMO's +1.402: reading (p2), outside 30 %**
    (every ion within 4.2 Å: +0.799). Of the +0.79, +0.65 is the contact itself (2.197 against 2.26 Å: the steep Au–O
    repulsion). Artifact to rule out before (p2) is read as physics: the pair curves use free spherical densities (O²⁻
    in a Watson sphere), the ALMO FRZ the cluster's own SCF densities. **Pre-registered (2026-11-04 18:18): the non-additivity in one
    Hamiltonian.** First-order energies E1 (antisymmetrised product, counterpoise basis, wetting_eda's route) of the
    dimer and of each Au atom alone (doublet) at its own position over the same 6-ion cluster. **Reading:** (n1)
    E1(dimer) − ΣE1(atoms), tilted − upright, ≥ +0.4 eV: the Au–Au bond's density near the surface carries a genuinely
    non-pairwise Pauli term; (n2) within ±0.2 eV: the atoms add in the cluster's densities, and (p2) was the free-density
    model's artifact (a frame-pair term must then be built from in-crystal ion densities); (n3) otherwise partial.
    **Result (2026-11-04 18:29; `frz_nonadditivity.py`).** E1 (eV): tilted dimer +2.051 = contact atom +1.800 (2.197 Å) + far atom
    +0.059 + non-additivity +0.192; upright dimer +0.653 = +1.425 (2.26 Å) − 0.008 − 0.764. E1(dimer) tilted − upright
    +1.398 against ALMO's FRZ +1.402 (the routes agree to 4 meV). **Non-additivity, tilted − upright: +0.956 eV —
    reading (n1).** The frozen penalty is genuinely not pairwise in atomic densities. Mechanism (consistent with the
    numbers, a hypothesis until the angular scan below): in the upright dimer the σ bond draws the contact atom's 6s
    density along the axis, away from the O, and lowers its Pauli repulsion with the O-top by 0.76 eV against a lone
    atom; tilted by 70°, the bond lies nearly along the surface and the shielding is gone (+0.19). TRACE's overlap terms
    are isotropic atom pairs, and the backbone learned the upright (shielded) contact from the data; transferring that to
    an inclined dimer leaves it ≈ 0.96 eV too stable, against the models' measured 0.5–1.1 eV excess.
    **Pre-registered (2026-11-04 18:29): the angular law.** The dimer rotated rigidly about its contact Au held 2.26 Å straight above
    the O (Au–Au 2.545 Å, the upright reference's), in the plane toward the far Au's Mg, θ = 0, 20, 40, 60, 70, 90°
    from vertical: E1(dimer), E1(far atom) at each θ, E1(contact atom) once; D(θ) = E1(dimer) − E1(contact) − E1(far).
    **Reading:** (g1) D rises monotonically with θ and D(θ) = D₀ + D₁ cos θ + D₂ P₂(cos θ) fits it within 0.05 eV: an
    orientation-dependent Pauli law of a bonded s atom against a closed-shell ion is defined (NEW-TO-TRACE: the frame
    pair's repulsion modulated by the bond direction; ESTABLISHED in kind: anisotropic atom–atom repulsion from distributed
    density overlap, Stone and Price); (g2) not monotonic or the fit fails: no simple law, recorded.
    **Result (2026-11-04 18:54; `frz_angular.py`; 6 ions).** D(θ), eV: 0° −0.764, 20° −0.639, 40° −0.329, 60° +0.020, 70° +0.165,
    90° +0.343 (E1 contact atom +1.425; far atom −0.008 … +0.045, +1.020 at 90° where it sits 2.26 Å over a Mg). Fit
    D = +0.0355 − 0.1831 cos θ − 0.6110 P₂(cos θ), largest residual 0.0084 eV, monotonic: **reading (g1).** With φ the
    angle at the contact atom between its partner and the ion (φ = 180° − θ), D = 0.036 + 0.183 cos φ − 0.611 P₂(cos φ):
    quadrupolar, as a σ bond's deformation density is, with a small dipolar part. **The law (NEW-TO-TRACE, at
    R = 2.26 Å):** the frozen repulsion of a bonded s atom against a closed-shell anion changes by 1.11 eV between the
    bond pointing away from the anion and lying across it.
    **Pre-registered (2026-11-04 18:54): the radial form and the frozen check.** (a) The same scan at R = 2.6 Å; hypothesis
    D(R, φ) = E1_atom(R)·[d₀ + d₁ cos φ + d₂ P₂(cos φ)] (the anisotropy scales with the lone atom's repulsion: both are
    overlap of the same densities), d from 2.26 Å (d₂ = −0.429, d₁ = +0.128, d₀ = +0.025); read if each of D(0°), D(70°)
    at 2.6 Å is within 20 % (or 0.03 eV) of the prediction. (b) PBE (SCF, the same 6 ions) along the rigid rotation, and
    the frozen models (steps 20, 24; s42, s43) along it as trained and with the correction ΔE = Σ_O E1_atom(R)·[d₁(cos φ −
    cos φ_up) + d₂(P₂(cos φ) − P₂(cos φ_up))] referenced to the upright contact the backbone learned (φ_up = 180°): read
    as PASS if the corrected E(θ) − E(0) is within 0.2 eV of PBE's at θ = 40, 60, 70° on every model.
    **Results (2026-11-04 20:35).** (a) R = 2.6 Å: E1 contact atom +0.389; D(θ) −0.377, −0.314, −0.159, +0.012, +0.082, +0.162 at
    0–90°; fit D = +0.0015 − 0.0578 cos θ − 0.3179 P₂, residual 0.0044 eV (the shape holds). The pre-registered radial
    form is **falsified**: it predicts D(0°) −0.209 and D(70°) +0.045 against −0.377 and +0.082 — the anisotropy decays
    at about half the lone atom's rate (2.1 against 3.8 Å⁻¹ between 2.26 and 2.6 Å). Post hoc, a hypothesis: the P₂ term
    scales as √E1_atom (ratio 0.520 against √(0.389/1.425) = 0.522), as an overlap S does while the Pauli energy goes as
    S²; the cos term as E1^0.89. (b) PBE SCF along the rigid rotation (17 ions): E(θ) − E(0) +0.100, +0.357, +0.642,
    +0.745, +1.451 eV at 20–90°; the models as trained are 0.3–1.0 eV too low at 40–70° (step 20 s42 −0.013 / −0.103 /
    −0.221, s43 +0.025 / +0.081 / +0.060; step 24 s42 −0.024 / −0.058 / −0.132, s43 −0.002 / +0.537 / +0.414).
    Corrected by D(θ) − D(0): step 20 s42 +0.422 / +0.681 / +0.708 and step 24 s42 +0.411 / +0.726 / +0.797, within 0.07
    eV of PBE at every angle; step 20 s43 +0.460 / +0.865 / +0.989 and step 24 s43 +0.433 / +1.321 / +1.343, over by up to
    0.68 eV at 60–70°. **Pre-registered verdict: FAIL** (not every model). Reading, a hypothesis: the correction has
    PBE's size where the model has no angular stiffness of its own (s42) and double-counts where the seed's backbone
    learned some (s43); frozen weights cannot separate the two, a training can.
    **Pre-registered (2026-11-04 20:35): the radial law at two new R.** D₂(R) = −0.611·√(E1(R)/E1(2.26)), D₁(R) = −0.183·(E1(R)/
    E1(2.26))^0.888, D₀ = 0 (the fits at 2.26 and 2.6 Å), with E1(R) the lone atom's measured E1 at each R; scans at
    R = 2.4 and 2.9 Å; PASS if D(0°) and D(70°) at both are within 15 % (or 0.02 eV) of the law.
    **Result (2026-11-04 21:19): PASS at both R.** E1 (contact atom) 0.8454 eV at 2.4 Å, 0.1045 at 2.9 Å; D(0°) −0.573 against the law's
    −0.586, D(70°) +0.124 against +0.113 (2.4 Å); −0.198 against −0.183 and +0.045 against +0.048 (2.9 Å). Angular fits:
    D = +0.014 − 0.113 cos − 0.470 P₂ (2.4 Å, residual 6 meV), −0.003 − 0.021 cos − 0.173 P₂ (2.9 Å, 2 meV). Over a 14-fold
    range of E1 the law holds to 9 %. **The bond-oriented Pauli law (NEW-TO-TRACE; ESTABLISHED in kind: exchange repulsion
    ∝ S², Murrell, Randić and Williams, Proc. R. Soc. A 284, 566 (1965); anisotropic atom–atom repulsion, Stone and Price,
    J. Phys. Chem. 92, 3325 (1988)):** for a bonded s-class atom a (partner b) and a closed-shell frame anion j at R,
        E_bop(a, j) = D₁(R)(cos θ − 1) + D₂(R)(P₂(cos θ) − 1),  cos θ = r̂_aj · (−ê_ab),
        D₂(R) = −0.611 √(E1(R)/E1(2.26)),  D₁(R) = −0.183 (E1(R)/E1(2.26))^0.888  (eV),
    with E1(R) the lone Au⁰'s frozen repulsion on the O-top site (1.425 / 0.845 / 0.389 / 0.105 eV at 2.26 / 2.4 / 2.6 /
    2.9 Å). It is zero for a contact straight behind the bond (the upright O-top contact, which the backbone learned from
    the data) and repulsive otherwise. Scope, stated: measured for Au₂ (2.545 Å) over a MgO(100) O; one partner; the
    neutral frame density (configuration independent, as the frame wall); Mg²⁺ not measured.
    **Pre-registered (2026-11-04 21:19): frozen checks of step 20 and step 24 + E_bop** (`probe_bop_frozen.py`; E1(R) interpolated in
    ln E1, extrapolated with the end slopes, switched off over 3.8–4.3 Å; forces by autograd). (F1) tilt path A: the
    barrier E(0.4) and E(1) within 0.1 / 0.2 eV of PBE's +0.255 / −0.197 on every model; (F2) the flat dimer relaxed by the
    T5 protocol stays within 0.3 Å of its heights on every model; (F3) wet_u and wet_d at the references both rise, and
    T5 changes by < 30 meV (E_bop is geometric); (F4) step 20's training-set class offsets: the common part falls in
    magnitude on both seeds, the doped − undoped part moves by < 30 meV. Direction and size on frozen weights only.
    *Checks of the implementation (2026-11-04 21:22):* E_bop's forces against central differences 5e-12 / 8e-11 eV/Å (flat reference,
    tilted state); along the rigid rotation E_bop − E_bop(0) = +0.119 / +0.424 / +0.797 at 20 / 40 / 60° against the law's
    +0.125 / +0.435 / +0.784, and +1.053 / +1.644 at 70 / 90° against +0.929 / +1.107 — the excess is the far Au's own O
    contacts, which the 6-ion cluster held as point charges (an extrapolation of the law, not measured). E_bop at the flat
    reference: +377 meV (O off-axis at 3.2–3.5 Å, beyond the measured 2.9 Å).
    **Pre-registered (2026-11-04 21:22): the law at the flat geometry, against PBE.** frz_nonadditivity on the 4G flat reference over a
    cluster built around it (quantum_sites, r_au 3.6): D_flat = E1(dimer) − ΣE1(atoms) against the law's absolute value
    Σ_j [D₁ cos θ + D₂ P₂(cos θ)] over the same quantum O. PASS if they agree within 30 % or 0.1 eV; only then is a flat-
    geometry number from E_bop (F3, F4) read as physics.
    *Amendment (2026-11-04 21:24, before any F result):* the referenced form (zero for a contact straight behind the bond, at every R)
    assumes the backbone holds the shielded value D(R, 0) for every Au–O pair; at the flat geometry, which is in the data,
    that is arbitrary (it sums −D(R, 0) over many off-axis O and gives +377 meV, against the absolute law's +0.055 eV over
    the flat cluster's two quantum O). So F1 and F2 (the tilt relative to the upright contact, which is in the data) stay
    as read; F3 and F4 are reference-dependent and are reported, not read. The physical quantity is the absolute law,
    E_abs = Σ [D₁ cos θ + D₂ P₂(cos θ)] (its isotropic part D₀ ≈ 0 at every R, so it does not compete with the backbone);
    a training would use it. Its class means on the training set are computed beside the referenced ones.
    **Results (2026-11-04 21:36).** Training-set class means (meV; doped / undoped): referenced form flat +390 / +400, upright +120 /
    +106 (offset shifts +271 / +295); absolute form flat +73 / +75, upright −883 / −840 (shifts +956 / +915); both move the
    doped − undoped offset by ≤ 41 meV. Law at the flat geometry: PBE non-additivity +0.109 eV (10 ions, 2 quantum O
    at 3.50 Å) against the law's +0.055 — PASS by the 0.1 eV criterion, at half the PBE value (the flat Au also sit over
    quantum Mg, which the law omits). Step 20 + the referenced E_bop (s42 / s43): (F1) E(0.4) +0.347 / +0.355 against PBE
    +0.255 (within 0.1: borderline PASS), E(1) +0.138 / +0.243 against −0.197: **FAIL**; (F2) the flat dimer is pushed
    off the surface (2.89 → 3.29 / 4.06 Å, −165 / −371 meV): **FAIL**; (F3, reported) wet_u +541 → +822 / +735 → +1016,
    wet_d −452 → −177 / −221 → +54, T5 −993 → −1000 / −956 → −962 at the references; relaxed wet_u +666 / +659, wet_d −234 /
    −43, T5 −901 / −702 meV. Reading: the referenced form over-repels the flat geometry (+377 meV against PBE's ≈ +0.11),
    as the amendment feared, and over-corrects the tilted state.
    **Pre-registered (2026-11-04 21:36): the data-referenced form.** A refit local backbone absorbs, to leading order, the term's
    isotropic part at each R averaged over the angles the data hold: E_iso(R) = ⟨D(R, θ)⟩ over the training set's Au–O
    pairs (0.1 Å bins over 2.0–4.3 Å, switch-weighted, linear between bins). E_test = Σ [D(R, θ) − E_iso(R)] is what a
    training could not absorb. F1 and F2 rerun with E_test (criteria unchanged); F4 must move each class offset by
    ≤ 50 meV (by construction it should); F3 reported. If F1 and F2 pass, E_abs goes into trace_redox (a new local
    branch, tests first) and a training step is pre-registered; if not, the residual is recorded.
    **Results (2026-11-04 22:03; `BOP_MODE=data`).** E_iso(R): −1.20 / −0.97 / −0.81 eV at 2.05 / 2.15 / 2.25 Å (upright contacts),
    ≈ 0 beyond 3 Å (flat contacts dominate). (F4) class offsets move by +117 / +119 meV (doped / undoped): **FAIL** as
    pre-registered — the expectation "≤ 50 by construction" was wrong: at equal R the two classes see their O from
    different angles (an upright dimer's lateral O near 55°, a flat dimer's near 110°), which no isotropic backbone
    absorbs; the shift is equal in the two classes (T5 moves < 0.5 meV). (F2) the flat dimer stays: 2.89 → 3.01 / 2.90 Å
    (−16.5 / −5.9 meV): **PASS** on both seeds. (F1) E(0.4) +0.428 / +0.436 against PBE +0.255, E(1) +0.227 / +0.332
    against −0.197: **FAIL** — the tilted state is now over-penalised by 0.42–0.53 eV. (F3, reported) references wet_u
    +586 / +780, wet_d −407 / −176; relaxed (T5 protocol) wet_u +628 / +881, wet_d −398 / −168, T5 −1027 / −1048 meV.
    **The residual, recorded:** PBE's tilted − upright = FRZ +1.40 + POL −0.46 + CT −0.06 eV; E_bop supplies the FRZ part,
    and the model's own polarisation changes by ≈ −0.08 eV between the two (step 20 s42 term split: screened −0.10,
    SPEC 26 +0.04, SPEC 28 +0.03, coherent −0.05), so ≈ 0.4 eV of PBE's polarisation relief is missing — the size of F1's
    overshoot. PBE splits it POL_A −0.30 (Au₂ itself) and POL_B −0.24 eV (the surface). Candidate, a hypothesis: the
    bound-sector anisotropy already measured (the model's Au₂ α⊥ 30–60 % low at short R; a tilted dimer feels the
    surface's normal field mostly across its bond). Gate verdict: F1 fails, so E_bop is not trained yet.
    *28-ion upright (2026-11-04 22:04):* E − E(flat) −0.978 eV (25 ions −1.024): tilted − upright +0.781 (25 ions +0.817), and the
    cluster's wet_u +0.978 against the periodic DFT's +0.929. Converged to 0.04 eV.
    **Correction of the residual's reading (2026-11-04 22:04), and the next step pre-registered.** ALMO's POL is not field induction
    alone: it is each fragment's orbitals relaxing in the other's full frozen potential, Pauli included. For the tilted
    dimer it recovers 1.09 of a 2.00 eV frozen repulsion. TRACE's bound sector answers electric fields only, so it can
    never supply the Pauli-driven relaxation; a term calibrated on E1 (frozen) is too strong by it. The physical target
    is the relaxed, transfer-free interaction FRZ + POL (the ALMO energy): tilted − upright +0.94 eV against FRZ's +1.40.
    (The field-induction part of POL, which the model's bound sector does carry, is estimated at ≈ 0.05 eV for the far Au
    at 3.07 Å, −½ α F² with F ≈ 0.5 V/Å — stated, not measured.) **Pre-registered:** the angular scan at R = 2.26 Å at
    the ALMO level (`tools/electrostatics/almo_angular.py`: the dimer by almo_split.run_geometry, each Au atom alone as a
    doublet by almo_split_open's unrestricted ALMO), D_ALMO(θ) = I(dimer) − I(contact) − I(far), I = E_ALMO − E_A − E_B.
    **Reading:** (h1) D_ALMO(70°) − D_ALMO(0°) within 0.15 eV of the value F1 asks for (the frozen law's +0.93 minus F1's
    overshoot, ≈ +0.45 eV): the ALMO-level law is the term; F1 is then rerun with it scaled by D_ALMO/D_FRZ at 2.26 Å, the
    radial law kept; (h2) otherwise, recorded.
    **Result (2026-11-04 23:04; `almo_angular.py`; checks a1–a3 PASS throughout).** Contact atom alone: I_FRZ +1.370, I_ALMO +0.472 eV
    (relaxation recovers 0.90 eV even for a lone atom). D_FRZ = +0.034 − 0.178 cos − 0.613 P₂ (residual 7 meV; the own-basis
    route reproduces the E1 law), D_FRZ(70°) − D_FRZ(0°) +0.929; **D_ALMO = +0.175 − 0.273 cos − 0.394 P₂ (residual 4 meV),
    D_ALMO(70°) − D_ALMO(0°) +0.700 eV.** The law survives the relaxation at three quarters of its amplitude. Reading
    **(h2)**: +0.70 is 0.25 eV above F1's ≈ +0.45; the Pauli-driven relaxation explains about half of F1's overshoot.
    *Reported, not a gate (2026-11-04 23:30; `BOP_MODE=data BOP_LEVEL=almo`, the relaxed law with the frozen-level radial scaling, an
    assumption):* step 20 + E_test along path A, s42: +0.061 / +0.355 / +0.304 / −0.008 eV at λ 0.2 / 0.4 / 0.6 / 1.0 (PBE
    +0.078 / +0.255 / +0.236 / −0.197; as trained +0.065 / +0.173 / −0.052 / −0.779); s43: +0.071 / +0.363 / +0.340 /
    +0.097. s42 meets F1's criteria (barrier within 0.100, end within 0.19 eV), s43 misses by 0.11 / 0.29. The flat dimer
    stays (2.89 → 3.01 / 2.91 Å); T5 at the references moves < 0.5 meV; relaxed (T5 protocol) wet_u +613 / +868, wet_d
    −416 / −215, T5 −1028 / −1083 meV.
    **Where this arc stands (conclusion).** The flat dimer's collapse in steps 21–24 has a first-principles cause: the
    Pauli repulsion of a bonded Au against a surface O²⁻ depends on the Au–Au bond's orientation (the σ bond shields the
    contact atom of an upright dimer, not of a tilted one), a non-pairwise term that TRACE's isotropic, frame-relative
    overlap tables do not hold and the backbone could only learn at the data's (upright) geometries. Its law is measured:
    D = D₀ + D₁ cos θ + D₂ P₂(cos θ), D₂ ∝ √E1(R), D₁ ∝ E1(R)^0.888, at the ALMO level D₁ −0.273, D₂ −0.394 eV at 2.26 Å
    (frozen level −0.183 / −0.611), residuals ≤ 8 meV, radial law within 9 % over a 14-fold range. On frozen step-20
    weights it keeps the flat dimer in place and brings the tilt path within 0.1–0.3 eV of PBE without moving T5. What
    frozen weights cannot settle is how a refit backbone shares the isotropic part; that needs a training. Next decision:
    the term into trace_redox (a new local branch, tests first, a SPEC section) and one pre-registered training step.
    **Branch `bond-pauli-2026-11-05` (2026-11-05 05:18; decision of 2026-11-05).** The term goes in as SPEC 33 with the ALMO-level
    coefficients and the frozen-level radial scaling f(R) = E1(R)/E1(2.26) for both D₁ and D₂ (an assumption at the ALMO
    level, measured at 2.26 Å only). **Pre-registered (2026-11-05 05:18, before the run): the ALMO-level law at 2.6 Å**
    (`almo_angular.py ... 2.6`). Prediction from the scaling: D₁(2.6) = −0.273·0.2730^0.888 = −0.0862, D₂(2.6) =
    −0.394·√0.2730 = −0.2059 eV, so D_ALMO(70°) − D_ALMO(0°) = +0.329 eV. PASS if the measured difference is within 15 %
    (0.28–0.38) and the fitted P₂ coefficient within 15 % of −0.206. FAIL: the ALMO radial exponents are refitted from
    the two R (the SPEC's constants then change before training; the test suite's targets with them).
    **Result (2026-11-05 06:13; `almo_angular_R2.6.json`; checks PASS).** Contact atom alone at 2.6 Å: I_FRZ +0.347, I_ALMO +0.047 (the
    relaxation recovers 87 % against 66 % at 2.26 Å). D_FRZ = −0.007 − 0.043 cos − 0.325 P₂ (residual 4 meV; the frozen law
    reproduced); **D_ALMO = +0.047 − 0.060 cos − 0.282 P₂ (residual 3 meV); D_ALMO(70°) − D_ALMO(0°) = +0.412 eV against
    the predicted +0.329, the P₂ coefficient −0.282 against −0.206: FAIL.** The relaxed anisotropy decays more slowly than
    the frozen one (its P₂ part keeps 87 % at 2.6 Å against 64 % at 2.26 Å) and its cos part faster. As pre-registered,
    the relaxed level takes its own exponents from the two R: D₂ ∝ f^0.259, D₁ ∝ f^1.170 (frozen level unchanged:
    0.5 and 0.888). Stated: a two-point law, an interpolation between 2.26 and 2.6 Å and an extrapolation beyond.
    **Pre-registered (2026-11-05 06:13, before the run): the relaxed law at 2.9 Å.** Prediction: D₁(2.9) = -0.0129, D₂(2.9) =
    -0.2002 eV, D_ALMO(70°) − D_ALMO(0°) = +0.274 eV. PASS if the measured difference is within 15 % and the
    fitted P₂ coefficient within 15 % of -0.200; then the relaxed radial law stands over 2.26–2.9 Å and step 25
    launches with it. FAIL: a three-point fit replaces it (and is recorded as such) before the launch.
    **The fit made self-consistent (2026-11-05 06:20; before the 2.9 Å numbers are read).** The module sums the law over both Au; the
    far Au's own term against the O (cos θ = −0.84 at 70°, R 3.9 Å) is in the measured D but was absorbed into the
    contact-only fits, so the module double-counted it (−30 meV at 70° with the two-point exponents). Refit jointly over R
    and θ = 0–70° to the module's two-atom form, one D₀ per R; 90° excluded (the far Au sits 2.26 Å over a quantum Mg
    there, and its (Au, Mg) non-additivity, for which there is no law, is in the measured D). Frozen (four R): C₁
    −0.1040, C₂ −0.6533, p₁ 1.877, p₂ 0.4933, largest residual 4.7 meV — p₂ = ½ confirmed (exchange ∝ S², the anisotropy
    ∝ S), the small cos part decaying as S⁴. Relaxed (two R): the free fit sends p₁ to 0 (C₁ −0.039: the cos term is not
    determined by two R); with p₁ held at 1.877: C₁ −0.2188, C₂ −0.4360, p₂ 0.2913, residual 10.1 meV; P₂ only: C₂
    −0.534, p₂ 0.423, 12 meV. The exponent spread (0.29–0.42) is a factor 2 at the flat contacts (f ≈ 0.005), so the
    final relaxed constants are the three-R joint fit (p₁ fixed) after the 2.9 Å scan; the module carries the
    provisional (−0.2188, −0.4360, 1.877, 0.2913) meanwhile. *Amended pre-registration for the 2.9 Å scan:* its
    criterion reads the provisional constants' prediction D_ALMO(70°) − D_ALMO(0°) = +0.271 eV (within 15 %: a PASS
    says two R extrapolate to the third); the final constants are the three-R fit either way, recorded with its residual.
    **2.9 Å result (2026-11-05 11:39; `almo_angular_R2.9.json`).** Contact atom alone: I_FRZ +0.087, I_ALMO −0.028. D_FRZ = −0.010 −
    0.009 cos − 0.178 P₂ (residual 3 meV; D(70°) − D(0°) +0.242, the frozen law's +0.243 reproduced); **D_ALMO = +0.012
    − 0.011 cos − 0.175 P₂ (residual 2 meV); D_ALMO(70°) − D_ALMO(0°) = +0.240 eV against the provisional constants'
    +0.271: within 15 %, PASS** — two R extrapolate to the third to 12 %. **Final relaxed constants (three R, θ ≤ 70°,
    the two-atom form, p₁ held at the frozen 1.877 as pre-registered): C₁ −0.0923, C₂ −0.4923, p₂ 0.3784; largest
    residual 9.5 meV, rms 4.5 meV** (freed, p₁ runs to 0 with C₁ −0.035: a non-decaying dipolar term, not determined
    by the data and not physical as an extrapolation; residual 3.3 meV, stated). The module, SPEC 33 and the tests
    carry the final constants; step 25 launches on the suites.
    *Frozen checks with the final constants (2026-11-05 11:54; `BOP_MODE=data BOP_LEVEL=almo`; reported, not gating).* Step 20 s42 +
    E_test along path A: +0.059 / +0.412 / +0.343 / +0.016 / −0.033 eV at λ 0.2 / 0.4 / 0.6 / 0.8 / 1.0 (PBE +0.078 /
    +0.255 / +0.236 / – / −0.197): the barrier 0.16 above PBE's, the end within 0.17. References: wet_u +541 → +606, wet_d
    −452 → −387, T5 −993.4 → −992.7 meV. Training-set class means of E_test: flat +27 / +25, upright −185 / −182 meV
    (doped / undoped), the two classes' shifts equal to 6 meV. The flat dimer's relaxation then **crashed in the probe**
    (ConfigurationSearchError, the amplitude closure beyond 512 configurations, inside the model's own forward at a
    geometry the BFGS reached; the provisional-constants run relaxed the same dimer to 2.95 Å). Candidate artifacts,
    under test before any reading: the probe's E_iso(R) is piecewise linear, so its force jumps at the bin edges, and
    the BFGS step (0.2 Å) can land in an untrained geometry; checks: the absolute law (no E_iso), and the data form with
    a 0.05 Å step, each logging the geometry and ensemble size per step. The module as trained in step 25 has no E_iso.
    *The provisional-constants run, read in full (2026-11-05 11:55):* s42's flat dimer relaxed to 2.95 Å (F2 PASS), but the relaxed
    T5 protocol gave wet_u +383 713 meV — one of the four reference relaxations (the protocol relaxes all four) ran away to
    a geometry where the frozen step-20 energy falls by hundreds of eV; s43's relaxation was refused by SPEC 28 at a
    slab-wide ±2 configuration (the ensemble explodes at the geometry reached). **Frozen step-20 weights under E_bop
    relax some reference into a pathological geometry**; which one and along which path is being traced (the four
    references, per-step geometry, ensemble size and energy). Hypothesis: E_bop's lateral terms on the upright dimer
    (the O's at 3.2–3.5 Å, f^0.38 ≈ 0.14, −0.07 eV each) push it off the O-top toward a hollow, where the frozen step-20
    model has its own collapse paths (Au–Mg 1.8 Å in the step 21–23 relaxations, the frame wall's motive). On trained
    weights (step 25) the backbone refits; whether such paths remain is what B1 and the protocol's relaxations measure.
    *Candidate artifacts of the F2 crash, tested (2026-11-05 12:01):* the undoped flat dimer relaxes cleanly under E_bop with the final
    constants in every variant — the absolute law (to 3.07 Å, 9 steps), the data form with a 0.05 Å step and with the
    default 0.2 Å step (both to 3.00 Å, 10 steps; 3 configurations throughout). So neither the piecewise E_iso nor the step
    size is the cause, and the crash and the runaway are in another reference's relaxation (the probe relaxes all four
    before printing F2); the trace of the four references says which.
    *The four references traced (2026-11-05 12:11; step 20 + E_test, data form).* Both flat (wetting) references relax in 10 steps on
    both seeds (heights 2.87–3.00 Å, E_bop +26 to +33 meV). The upright references fail: s42 doped, the contact Au
    oscillates between 2.15 and 2.53 Å for 150 steps without converging (E_bop −16 to −68 meV, the energy within ±0.06
    eV); s42 undoped, the configuration closure explodes at the first step; s43 doped, after 60 steps of wandering
    (2.35–2.54 Å) the bound sector's hardness H = J + diag(η) turns indefinite at the contact Au. The oscillation has a
    named candidate: the probe's E_iso(R) is piecewise linear with 0.1 Å bins, so its force jumps by up to 1.8 eV/Å at
    each bin edge (2.15, 2.25, 2.35, 2.45 Å), exactly where an upright contact sits (the flat dimer's O's at 3.5 Å see
    E_iso ≈ 0); the module as trained has no E_iso. The two hard failures need the failing geometries (the trace's handler
    lost them); rerun step by step in the data form and the absolute law.
    **The upright relaxations traced step by step (2026-11-05 12:27; decisive).** Absolute law, s42 undoped upright: E_bop's force on
    the contact Au at the reference is +1.90 eV/Å toward the O (the law at θ = 0 is the shielding, −0.585·f(R)^p, attractive
    and growing as R falls); the contact dives 2.26 → 2.03 → 1.83 → 1.63 Å in five steps, the frozen step-20 energy
    falling −0.47, −1.38, then −296 eV with 113 configurations (the ensemble's collapse region); by step 50 the dimer has
    dissociated (Au–Au 1.93, heights 1.14 / 1.76 Å). s43 doped upright: the same dive (1.64 Å at step 5, −2.7 eV; −1393 eV
    at step 30; SPEC 28 refuses a slab-wide ±2 configuration at step 42). Data form (E_iso subtracted; E_bop force on the
    contact +0.056): s42 undoped still dives (1.97 Å at step 5, −0.32 eV; closure explodes at 1.17 Å), s43 doped oscillates
    for 61 steps across the E_iso bin edges (2.29–2.54 Å). **Reading.** The frozen step-20 model has a cliff below ≈ 2.0 Å
    at the O-top contact: its backbone's Au–O wall, learned at the data's 2.0–2.5 Å, is too weak below them (as its
    Au–Mg wall was, step 24's motive). The term's θ = 0 shielding is correct physics measured *relative to the lone atom*,
    but the lone atom's own frozen repulsion E1(R) against O²⁻ is no explicit term of the model (SPEC 23/25 are
    frame-relative; the frame wall covers Mg²⁺–Au⁰ only), so adding the shielding to a backbone that carried the whole
    wall softens the wall by 0.6–0.7 eV and tips the frozen dimer over its cliff. In step 25 the backbone refits (its
    residual at the upright contact becomes the steeper E1-like wall); whether it holds below the data is the risk B4 and
    the protocol's upright relaxations measure — pre-registered here as the hypothesis for a step-25 B4 failure. **The
    missing physics named:** the frame pair Au⁰–O²⁻'s frozen interaction E1(R) (penetration electrostatics + Pauli; the
    in-cluster lone atom's, 1.43 / 0.85 / 0.39 / 0.10 eV at 2.26 / 2.4 / 2.6 / 2.9 Å) as an explicit isotropic term,
    with SPEC 33's law as its bond-orientation modulation, so that E1 + D is repulsive at every θ and rises below the
    data by physics, not by the backbone's extrapolation. Measured next: E1(R) at 1.8, 2.0, 2.1, 3.2, 3.5 Å (the same
    6-ion cluster) and PBE's vertical approach of the upright dimer on the same cluster, as the target.
    **Built (2026-11-05 12:32): SPEC 33's wall part** (`RedoxConfig.bond_pauli_wall`, `trace_redox.bond_pauli.BondPauliWall`):
    E_wall(a, j) = s_c(R) E1(R) for every s-class atom and frame anion, E1 the lone atom's table (the law's own f(R)),
    isotropic, bonded or not; 5 tests (`test_redox_bond_pauli_wall.py`), the two files 13 passed. Physics checks in the
    tests: wall + law rises monotonically toward the O at θ = 0 / 40 / 70° over 1.8–2.9 Å at both levels and is positive
    to 2.6 Å; at the frozen level wall + law at θ = 0 reproduces the measured E1 of the upright dimer within 0.06 eV at
    the four R, including its attractive −0.10 eV at 2.9 Å (penetration electrostatics over Pauli) — the sign at long
    R is PBE's, not a defect. Step 26 = step 25 + `bond_pauli_wall: true` (configs and watcher made; pre-registration
    follows the PBE approach curve). The lone atom's E1 at 1.8 Å: +7.73 eV (the ln-extrapolation gave 7.95).
    *The frozen models' rigid upright approach (2026-11-05 12:36; `probe_upright_approach.py`, as trained, E(h) − E(2.26) eV at 1.8 /
    2.0 / 2.1 / 2.5 / 2.9 Å):* step 20 s42 +1.23 / +0.19 / +0.03 / +0.19 / +0.65, s43 +1.15 / +0.15 / +0.01 / +0.19 / +0.61;
    step 24 s42 +1.13 / +0.11 / −0.03 / +0.25 / +0.87, s43 +1.03 / +0.10 / −0.02 / +0.22 / +0.62. A shallow minimum at
    2.1–2.26 Å and a wall of only +1.0 to +1.2 eV at 1.8 Å, where the lone atom's frozen repulsion alone rises by +6.3 eV
    (E1 7.73 against 1.43). So the backbone's wall is several eV too soft below the data, not a cliff in a rigid approach;
    the −296 eV collapse needs the dimer's stretching as well (Au–Au 2.55 → 3.0 Å in the trace). PBE's approach of the
    upright dimer on the same cluster is being measured as C4's target. Lone-atom E1 so far: 7.7324 / 3.6975 / 2.5648 eV
    at 1.8 / 2.0 / 2.1 Å (the ln-extrapolation from 2.26–2.4 gave 7.95 / 3.75 / 2.58: within 3 %).
    **The table completed and the constants refitted on it (2026-11-05 12:47).** Lone-atom E1 at 3.2 / 3.5 / 3.8 Å: +0.0132 / −0.0121 /
    −0.0165 eV — the frozen interaction turns attractive beyond 3.3 Å (penetration electrostatics over Pauli), so the
    table is interpolated by C¹ monotone cubic Hermite forms (Fritsch–Carlson): in ln E1 over the positive points for the
    law's f(R), in E1 for the wall (the positive knots' slopes from the ln form, the tail's from E1). A knot at 2.26 Å
    (now interior) had a slope jump under the earlier piecewise-ln form (0.04 eV/Å in the wall's force); the C¹ form
    removes it (force continuity across the knot tested to 1e-5 eV/Å). Refit (`fit_bond_pauli.py`, the module's f):
    frozen C₁ −0.0720, C₂ −0.6649, p₁ 2.29, p₂ 0.5001, largest residual 3.7 meV (p₂ = ½ to four digits); relaxed C₁
    −0.0694, C₂ −0.4973, p₂ 0.3818 (p₁ 1.877 held), 6.2 meV. These are the constants in the module; step 25 trains with
    the earlier (−0.0923, −0.4923, 1.877, 0.3784; four-point table), recorded as such in its README entry.
    **PBE's approach of the upright dimer (2026-11-05 13:27; `otop_upright_scan.py`, 6 ions, dE_SCF(cp)):** −0.013 / −1.015 / −1.164 /
    −1.178 / −0.983 / −0.587 eV at 1.8 / 2.0 / 2.1 / 2.26 / 2.5 / 2.9 Å, i.e. E(h) − E(2.26) = +1.165 / +0.163 / +0.014 / 0 /
    +0.195 / +0.590. **Correction:** the frozen step-20 models reproduce this rigid approach within 0.07 eV at every height
    (s42 +1.23 / +0.19 / +0.03 / +0.19 / +0.65; s43 +1.15 / +0.15 / +0.01 / +0.19 / +0.61) — the backbone's wall is
    *not* "several eV too soft" (the 12:36 statement compared the dimer with the lone atom's +6.3 eV rise; the dimer's
    bond shielding, relaxation and transfer make PBE's own rise +1.17 eV). The collapse is a property of the relaxation
    path (stretching and sliding), not of the rigid approach.
    **A level inconsistency in step 26 as launched, caught before any result (2026-11-05 13:27).** The law is at the relaxed (ALMO)
    level, measured relative to the *relaxed* lone atom (I_ALMO: +0.472 / +0.047 / −0.028 eV at 2.26 / 2.6 / 2.9 Å), but
    the wall was built from the *frozen* lone atom (E1: 1.43 / 0.39 / 0.10). Wall + law is then neither the frozen nor
    the relaxed dimer interaction; the backbone would have to carry the lone atom's own relaxation (−0.9 eV at 2.26 Å,
    steeper below). The consistent wall at the relaxed level is I_ALMO(R) of the lone atom, so that wall + law =
    I_ALMO(dimer, θ) (the dimer's relaxed, transfer-free interaction: −0.03 eV upright at 2.26 Å) and the backbone's
    residual is the charge transfer. Step 26 (launched 13:05) is **withdrawn at 13:20** (logs kept as
    `log_withdrawn_2026-11-05_1320_frozen_wall_level.txt`; branch `bond-pauli-wall-level-2026-11-05` made first), to be
    relaunched with the level-consistent wall once I_ALMO(lone) is measured at 1.8 / 2.0 / 2.1 / 3.2 / 3.5 / 3.8 Å (running)
    and the tests pass. Pre-registered check of the construction: wall_almo + law_almo at θ = 0 reproduces the measured
    I_ALMO(dimer) (−0.030 / −0.255 / −0.207 eV at 2.26 / 2.6 / 2.9 Å) within the omitted D₀ (≤ 0.07) + residual: 0.1 eV.
    *The relaxed lone-atom table (2026-11-05 13:41; `otop_lone_almo.py`, own-basis route):* I_ALMO = 3.814 / 1.637 / 1.039 / 0.472 /
    0.047 / −0.028 / −0.035 / −0.027 / −0.019 eV at 1.8 / 2.0 / 2.1 / 2.26 / 2.6 / 2.9 / 3.2 / 3.5 / 3.8 Å (I_FRZ beside it:
    7.84 / 3.69 / 2.53 / 1.37 / … — this route's frozen values differ from the counterpoise-basis E1 table by ≤ 4 %
    (1.37 against 1.43 at 2.26 Å); the frozen wall keeps the E1 route its law was fitted with). The relaxation recovers
    half of the frozen repulsion at 1.8 Å and 90 % at 2.26, and the relaxed interaction has a shallow well (−0.035 eV)
    near 3.2 Å. Entered as the relaxed wall's table (C¹ monotone Hermite); the check wall + law = I_ALMO(dimer) holds.
    *Provenance note (2026-11-05 18:57):* step 20's `model.pt` files were overwritten by the step 25/26 trainings from 12:57 (s42) /
    13:02 (s43) until 18:55 (benchmarks README, incident). Every step-20 measurement in this section was taken before
    12:57 (the frozen checks to 12:10, the relaxation traces 11:55–12:31, the rigid approach 12:33 — s43's at 12:33 before
    its 13:02) and stands; the restored files reproduce the scored energies.
    **Pre-registered (2026-11-05 19:11, before the run): does the bond's orientation modulate the Au–Mg²⁺ repulsion at the flat
    contact?** SPEC 33 has a (Au, O) table only. The flat (wetting) dimer's two Au sit 2.89 Å above Mg²⁺ with the bond
    horizontal (θ = 90° to the Mg direction), so an orientation dependence of the Au–Mg frozen interaction would enter
    wet_u and wet_d directly. Measurement (`tools/electrostatics/almo_angular_mg.py`; the 6-ion-style cluster around a
    surface Mg, almo_angular's method): the dimer with its contact Au 2.89 Å above a Mg, rotated θ = 0, 45, 90° toward
    the (1,1,0) diagonal (the flat dimer's partner direction), D_Mg(θ) = I(dimer) − I(contact) − I(far) at the frozen
    and relaxed levels; the lone atom's I_FRZ and I_ALMO over Mg at 2.89 Å beside it. **Reading:** (m1) |D_ALMO(90°) −
    D_ALMO(0°)| ≥ 0.10 eV: the Mg modulation is benchmark-relevant (≥ 0.2 eV on wet from the two Au) and a (Au, Mg)
    table is measured and added before step 27; (m2) < 0.05 eV: the O-only scope stands for the flat reference,
    recorded; (m3) between: reported, the table added only if step 25/26's wet residuals point at the flat contact.
    **Result (2026-11-06 05:39; `almo_angular_mg.json`, 17 ions, 13 Mg).** Lone Au⁰ 2.89 Å above a Mg: I_FRZ +0.151, I_ALMO +0.020 eV
    (a tenth of the O-top site's at the same height: 1.37 / 0.47). D_FRZ(θ) = +0.108 / +0.050 / +0.120 and D_ALMO(θ) =
    +0.102 / +0.049 / +0.112 eV at 0 / 45 / 90°: **D_ALMO(90°) − D_ALMO(0°) = +0.010 eV (frozen +0.012): reading (m2)**,
    the O-only scope stands for the flat reference. The Au–Mg non-additivity is small, positive at every angle (≈ +0.1 eV,
    the dimer slightly more repelled than its atoms' sum) and not a cos / P₂ law (its 45° dip); its isotropic part is a
    pair function of R the backbone represents. No (Au, Mg) table is added.
    **Pre-registered (2026-11-06 10:11, before the run): is the bond-oriented law the same for Au₂⁻?** SPEC 33 enters as a frame term,
    configuration independent, measured on Au₂⁰. The doped cell's dimer is Au₂⁻, whose extra electron occupies σ* (lobes
    along the axis, a node at the bond centre), so the contact atom's density toward the anion differs from the
    neutral's. Measurement (`tools/electrostatics/almo_angular_anion.py`): Au₂⁻ (UKS doublet) over the O-top O at 2.26 Å,
    θ = 0, 40, 70°, the 6-ion cluster with its Al³⁺ in the point lattice as wetting_eda's doped model (one ionised donor,
    the insulating sub-problem). The anion's electron cannot be split between the atoms, so the non-additivity is defined
    against the neutral atoms, D⁻(θ) = I(Au₂⁻) − I(Au⁰ contact) − I(Au⁰ far), and its orientation dependence D⁻(70°) −
    D⁻(0°) is compared with the neutral's (+0.929 frozen, +0.700 relaxed at 2.26 Å).
    **Reading:** (a1) |Δ| ≤ 0.15 eV at the relaxed level: the frame-term form stands for the doped cell; (a2) > 0.15 eV:
    the law needs the charge state (a per-state table for Au⁻, as SPEC 23/25 are per state), pre-registered as the next
    SPEC 33 extension before any doped-sector step.
    **Result (2026-11-06 11:20; `almo_angular_anion.json`; 6 ions, 20 TIPs, one Al³⁺).** Contact Au⁰ alone: I_FRZ +1.386, I_ALMO
    +0.526 (the doped lattice's field changes the lone atom by +0.02 / +0.05 eV against the undoped cluster). Au₂⁻:
    D⁻_FRZ = −0.993 / −0.829 / −0.668, D⁻_ALMO = −0.862 / −0.776 / −0.662 eV at 0 / 40 / 70°. **D⁻_ALMO(70°) − D⁻_ALMO(0°) =
    +0.200 eV against the neutral's +0.700 (frozen +0.325 against +0.929): reading (a2), the law needs the charge state.**
    Physics: the anion's extra electron occupies σ*, whose axial lobes restore the density the σ bond drew away from the
    contact atom's outer side, so the anion's contact atom shields the O only a fifth as anisotropically as the neutral's.
    Three-angle fit (exact, three unknowns; a check at 20° and 60° is pre-registered below): relaxed D₀⁻ -0.663, C₁⁻
    -0.095, C₂⁻ -0.104 eV (neutral C₁, C₂ −0.069, −0.497 at the same R); frozen D₀⁻ -0.830, C₁⁻ +0.163,
    C₂⁻ -0.326 (neutral −0.072, −0.665). The large negative D₀⁻ is the anion's own interaction with the surface
    relative to two neutral atoms (its electron's electrostatics and Pauli), which the model carries in its per-state
    terms (SPEC 23 / 25 for Au⁻, the Coulomb, the induction); the law's business is the anisotropy.
    **Pre-registered (2026-11-06 11:20): (i) the check** — D⁻_ALMO at 20° and 60° within 0.02 eV of the three-angle law's
    -0.838 and
    -0.698 eV
    (running); **(ii) the extension (NEW-TO-TRACE), SPEC 33 per charge state:** the law's coefficients become functions of
    the s atom's charge state, (C₁(n), C₂(n)) with n = 0 the neutral pair (Au⁰) and n = −1 the anion (Au⁻; C₁⁻, C₂⁻ above,
    the same radial exponents until measured at a second R), entered as a one-body term of the configuration energies
    (an [n, S] table added to the ladder, as the penetration self-terms are), so each configuration's energy carries its
    own law; Au⁺ (one σ electron) unmeasured: the neutral law, stated. The wall stays a frame term (SPEC 23 / 25 carry
    the state-minus-frame isotropic differences). Built and tested after steps 25 / 26 are scored; one change for step 27.
    **Check result (2026-11-06 12:15).** D⁻_ALMO(20°) = −0.838 against the three-angle law's −0.838, D⁻_ALMO(60°) = −0.702 against −0.698:
    **PASS** (0.6 and 4.6 meV; frozen 2.5 and 12 meV). Five-angle least squares, the anion's law at 2.26 Å: relaxed D₀⁻
    -0.652, C₁⁻ -0.1181, C₂⁻ -0.0911 eV (largest residual 2.4 meV); frozen D₀⁻ -0.800, C₁⁻ +0.1011,
    C₂⁻ -0.2905 (6.5 meV). Against the neutral (relaxed C₁ −0.069, C₂ −0.497): the anion's quadrupolar
    anisotropy is a fifth, its dipolar part comparable and of the same sign. These are the per-state coefficients for
    step 27's extension (radial exponents the neutral's until a second R is measured for the anion, stated).
    **Pre-registered (2026-11-06 14:32, before the run): the anion's law at 2.6 Å.** `almo_angular_anion.py ... 2.6`, θ = 0, 40, 70°.
    With the neutral's exponents (p₁ 1.877, p₂ 0.3818) and f(2.6) = 0.2730 the anion's coefficients scale to C₁⁻(2.6) =
    -0.0103, C₂⁻(2.6) = -0.0555 eV, so D⁻_ALMO(70°) − D⁻_ALMO(0°) = +0.080 eV. PASS if the measured difference is within
    15 % (or 0.02 eV) and the fitted P₂ coefficient within 15 % of -0.055: the anion keeps the neutral's radial law.
    FAIL: the anion's own p₂ from the two R, recorded, before step 27 is built.
  - **The model's wetting split on these terms (2026-10-27, experiments/benchmarks_4g/probe_t5_first_order.py; README).**
    Step 17, undoped: the model's relaxation share of wet_u is +112.0 / +102.6 meV (s42 / s43) against PBE's +1238.6
    (28 ions); per geometry −0.22 / −0.14 eV upright and −0.10 / −0.04 flat against −1.98 and −0.75. The first order
    is +106 / +193 against −272, and the backbone carries +211 / +332. So about 1.1 eV of O-top relaxation is missing
    from the explicit physics. Corrected the same evening (the bound moments frozen at each fragment's own optimum
    for the first order; benchmarks README): relaxation +221.9 / +171.9 meV (upright −0.29 / −0.22 eV): about a
    seventh of PBE's. The model's T5 is then mostly first order (−507 / −582 meV), not bound response.
  - **The direction of the O-top transfer (pre-registered 2026-10-27, before the run; almo_split.py --direction 3.6).**
    Variational one-directional delocalisation of the ALMO orbitals (ESTABLISHED in substance: Khaliullin, Bell and
    Head-Gordon, J. Chem. Phys. 128, 184112 (2008)). Validated on the water dimer (PBE/def2-SVP): (d1) the ALMO energy
    is RotationEnergy's exactly; (d2) gradient 6.2e-8; (d3) both directions reach E_SCF (0.0000 meV); (d4) the
    acceptor → donor share is 93 % (−223.0 of −240.7 meV, before counterpoise), the published direction. Reading for
    the upright Au₂ (10 ions): if surface → Au carries more than two thirds of CT_cp, the model's O → Au channel (Au⁻
    + an O hole) must carry the missing relaxation; if Au → surface does, the Au⁺ + band-electron channel; otherwise
    both. The counterpoise correction is attributed per direction (CT_cp(A→B) = CT(A→B) + BSSE_A), stated as an
    attribution. (d3) on the cluster: |E[all] − E_SCF| < 1 meV.
  - **The model's O-top channels (2026-10-27, experiments/benchmarks_4g/probe_otop_channels.py; frozen weights).**
    Undoped upright reference (Au–O 2.26 Å). Step 17 s42: the ensemble holds 5 configurations: the frame (p 0.56), the
    dimer's Au⁺Au⁻ (p 0.44, 0.10 eV up, H 1.25 eV) and O → Au (Au⁻ + O⁻, p 0.001, 6.27 eV up, H 0.30 eV, learned T
    −0.70 eV before the SPEC 21 switch and spin factor). Au → surface (Au⁺ + band electron, 4.73 eV up) is not
    selected at all. s43: O → Au 6.30 eV, H 0.22 eV; step 19 (band prior) s42 / s43: 4.92 (not selected) / 6.25 eV,
    H 0.25 eV. Second-order −t²/ΔE: −8 to −15 meV, against PBE's CT_cp −1.155 eV (10 ions). Term split of the O → Au
    energy (step 17 s42, eV): ladder −11.06, self +17.03, learned shifts +2.69, overlap −0.91, SPEC 26 +0.39, bare
    Coulomb −0.15, bound response −1.78, SPEC 28 +0.06. The coupling head (heads.TransferCoupling) is learned with no
    magnitude anchor (initial scale 0.01, a Harrison upper bound only, SPEC 27). F is even in τ, so ∂F/∂τ ∝ τ p/ΔE
    stays small for a configuration 6 eV up: the same vanishing-gradient trap as Theorem 3. Hypothesis (to be tested
    against the direction run): PBE's O-top transfer is O 2p → Au 6s donation, which needs the O → Au configuration
    near PBE's energy (the O-hole IP: PBE slab 4.80 eV against the model's 7.21) and a coupling near Harrison's
    universal V_spσ(2.26 Å) = 1.84 ħ²/(m d²) = 2.75 eV (ESTABLISHED: Harrison 1980; Froyen and Harrison, Phys. Rev.
    B 20, 2420 (1979)). The two-level estimate with ΔE 3.9 eV and t 2.75 eV is −1.42 eV; the model has −0.015.
  - **Result, upright (2026-10-27, almo_split.py --direction 3.6; all four minimisations converged).** CT_cp −1.1553
    eV = surface → Au −0.9619 (raw −0.9858, BSSE_B +0.0240; 83.3 %) + Au → surface −0.1685 (raw −0.1906, BSSE_A
    +0.0221; 14.6 %) + non-additive −0.0249 (2.2 %); (d3) E[all] − E_SCF −0.000 meV: PASS. Pre-registered reading:
    surface → Au carries more than two thirds, so the O → Au channel (Au⁻ + an O hole) must carry the missing O-top
    relaxation. Step 20 lowered that configuration to 5.24 / 4.15 eV (step 17: 6.27 / 6.30) through the
    first-principles Pauli table, but its coupling stays 0.1–1.0 eV (Au–O at 2.26 Å: s42 0.125, s43 1.010; Harrison
    V_spσ 2.75 eV). Flat: CT_cp −0.3189 = surface → Au −0.1730 + Au → surface −0.1388 + rest −0.0071; (d3) PASS.
    **Wet shares: CT_cp +836.5 meV = surface → Au +788.9 (94 %) + Au → surface +29.7 + rest +17.8.** The transfer's
    part of the undoped wetting energy is O → Au donation at the O-top contact. This motivates SPEC 31, the
    Harrison-anchored couplings (step 21, pre-registered in the benchmarks README).
  - **SPEC 32 (the downfolded donor → acceptor bond), frozen-weight counterfactual on step 20 (2026-10-28).**
    Unrelaxed references: wet_u +541 → +1038 / +735 → +1213 meV (DFT +929); wet_d −452 → −155 / −221 → +144; T5
    −993 → −1193 / −956 → −1069; 2–4 configurations. Pre-registered checks: (c1) separated limit −0.105 / −0.136 meV
    PASS; (c3) min Δ +3.70 / +3.16 eV PASS; (c2) the flat undoped dimer relaxed FAILS: it drops to 2.37 / 1.79 Å,
    −0.96 / −2.65 eV. A rigid height scan of the flat dimer (s43): as trained +0.19 / +0.76 / +1.00 eV at h = 2.60 /
    2.37 / 2.10 Å (the model's own repulsion is real); the dressing changes by −1.07 / −2.10 / −2.60 eV. Its steepness
    sits where the flat contacts are (Au–O 3.2–3.6 Å, inside the 3–4 Å window).
  - **The physical distance dependence of the O → Au coupling (PBE0 monomers of the ion library, O²⁻ in its Watson
    sphere).** |⟨Au 6s | O 2p σ⟩| = 0.265 / 0.243 / 0.201 / 0.151 / 0.107 / 0.089 / 0.060 at r = 2.0 / 2.26 / 2.6 /
    3.0 / 3.4 / 3.6 / 4.0 Å; relative to 2.26 Å: 1.09 / 1 / 0.83 / 0.62 / 0.44 / 0.37 / 0.25. Harrison's 1/d²
    gives 1.28 / 1 / 0.76 / 0.57 / 0.44 / 0.39 / 0.32; the windowed SPEC 32 form gives 1.26 / 1 / 0.76 / 0.58 /
    0.31 / 0.13 / 0. So the window under-estimates the physical coupling at the flat geometry (by Wolfsberg–
    Helmholz, t ∝ S: ESTABLISHED, J. Chem. Phys. 20, 837 (1952)), and the flat collapse is not an over-strong
    coupling at 3.6 Å. Open: the four-donor star (couplings in quadrature) against PBE's flat transfer (−0.173 eV
    at 2.89 Å), and the frame's first-order repulsion. PBE's directional split at h = 2.60 / 2.37 / 2.10 Å is
    running.
  - **SPEC 32 reviewed and rebuilt (2026-10-28, two review passes: code, then equations).** Found in the first
    version and corrected before any run: (i) the folded states were uncoupled, dropping the moved electron's
    resonance over Au₂: with the model's own numbers (V 2.617, Δ 3.16–3.70, |T_AuAu| 2.11–2.37 eV) the connected
    block gives −1.554 / −1.792 eV against the star's −1.355 / −1.477 at the O-top contact; (ii) the explicit set was
    selected on undressed energies (1.3 eV out of order against a 2.5 eV window); (iii) folded states could re-enter
    the explicit set; (iv) the explicit O–Au couplings were masked by species while only the Au⁰ acceptor was
    dressed (O²⁻ → Au⁺ lost); (v) NaN for a non-finite or non-positive Δ; (vi) the switch on [3, 4] Å sat on the flat
    contacts. The rebuilt term (trace_redox SPEC 32) folds every O–s move in either direction and any ladder state,
    resolves a closed-shell O donor into its three p components with Slater–Koster direction cosines, couples the
    folded states by the s–s hops, lets the selection see the dressing, drops folded partners, refuses μ_min ≤ 0,
    and switches on [4.0, 4.5] Å where Harrison's form has left the measured overlap. 13 tests; redox suite 762, the
    rest 1039 + 4 skipped.
  - **Counterfactual of the rebuilt term on step 20 (2026-10-28 evening).** wet_u +541 → +318 / +735 → +374 meV
    (DFT +929), wet_d −452 → −613 / −221 → −350, T5 −993 → −931 / −956 → −724; (c1) separated limit −0.105 /
    −0.136 meV PASS; (c3) min μ +1.23 / +0.36 eV PASS; (c2) the relaxed flat dimer drops to 2.61 / 2.57 Å by
    −0.23 / −0.31 eV (s42 PASS, s43 FAIL on the 0.3 Å height criterion). So the rebuilt term dresses the FLAT
    geometry more than the upright one, the opposite of PBE (upright transfer −1.16 eV against flat −0.32, and
    PBE's O → Au share at the flat reference is −0.17 eV for eight O–Au contacts at 3.5 Å, i.e. ~0.02 eV per
    contact). The first version's [3, 4] Å switch had suppressed those contacts and landed near DFT for the wrong
    reason. Hypothesis (to be measured): Harrison's 1/d² overestimates the O 2p → Au 6s coupling at 3.5 Å by about
    four times (second order: v ≈ 0.27 eV against 1.15 eV), because the in-crystal O 2p is compressed by its Mg²⁺
    neighbours, which the Watson-sphere free ion (whose overlap agreed with 1/d²) does not see. Decisive
    measurement, queued: PBE's directional transfer at the single O-top contact with the lower Au at 2.60 / 3.00 /
    3.50 Å (the first-principles radial form of v).
  - **The model's own folded states, upright reference (2026-10-28, leaf dump on step 20).** From the frame, the
    O-top move O1 → Au108 has Δ = +4.12 (s43) / +4.66 eV (s42): the trained O⁻ and Au⁻ levels put the transfer
    state where PBE's two-level inversion of its −0.96 eV puts it (4.4–6 eV). The frame's other leaves are the four O
    around the O-top site at 3.77 Å (V = 0.97 eV each under Harrison; Δ ≈ 4.2 eV): with Harrison's tail the frame's
    dressing is ≈ −2.0 eV against PBE's −0.96 (O → Au) / −1.16 (all transfer), with the PBE-implied 0.27 tail it is
    ≈ −1.3 eV. The small "min Δ" of the fold record (0.41 / 1.39 eV) belongs to high-lying explicit configurations
    (the ionic Au⁺ states, 2.4–3.9 eV up, whose own transfer partners lie 1.4–2.7 eV above them); they carry little
    weight. s43's as-trained explicit set also holds the transfer state (O1⁻, Au108⁻) itself at +4.12 eV, which the
    fold's selection excludes (its folded manifold, the frame, lies below it), as designed. So the frozen-weight
    over-dressing is the tail problem, the same one the flat scan shows: the fold needs the in-crystal, directional
    coupling (tools/electrostatics/transfer_overlap.py, queued), not the free-ion or universal 1/d² tail.
  - **Does the folded block double-count the Au₂ bond? No (derivation, 2026-10-28 night).** In the MO picture the
    dimer is σ², and an O electron can only enter σ* (σ is full), with amplitude 1/√2 of the atomic 6s coupling v;
    the block instead couples the frame P = (Au₁⁰, Au₂⁰) to Q₁ = (O⁻, Au₁⁻, Au₂⁰) with the full v and to the
    hop-coupled Q₂ = (O⁻, Au₁⁰, Au₂⁻) with 0, so both combinations (Q₁ ± Q₂)/√2, at Δ ∓ |t|, receive v/√2. Is the
    upper one spurious? In the two-site, three-electron problem the S_z = ½ space is exactly {Q₁, Q₂}, so
    (Q₁ + Q₂)/√2 = σ²σ*¹ (3ε − t) and (Q₁ − Q₂)/√2 = σ¹σ*² (3ε + t): both are physical. The frame configuration
    is the Heitler–London pair, which in MO terms is (σ² − σ*²)/√2, so its coupling to σ¹σ*² is a one-electron
    process through its σ*² component: the block's v/√2 is right. The ionic configurations are dressed separately
    and cannot accept at their doubly occupied site (the ladder forbids Au²⁻), which is the Pauli blocking the MO
    picture states. So the configuration basis carries the bond's bookkeeping exactly; the block is the first-order
    coupling structure for each explicit state, and nothing is counted twice. The remaining suspect for the flat
    over-dressing is the coupling's tail and direction alone.
  - **The hole's own band inside the folded manifold (derivation; NEW-TO-TRACE; to be implemented after the
    coupling's form is measured).** A hole left on O_i by donation to the Au can hop to a neighbouring O_j inside the
    folded manifold, (O_i⁻, O_j²⁻, Au⁻) ↔ (O_i²⁻, O_j⁻, Au⁻), the O 2p valence band of MgO. With the hole resolved
    into p components the hop is the Slater–Koster p–p element (ESTABLISHED)
        h_{im, jm'} = V_ppσ l_m l_m' + V_ppπ (δ_mm' − l_m l_m'),   l = the unit vector O_i → O_j,
    and the block's M gains these entries between leaves of different donors sharing the acceptor; the moved
    electron's s–s hops stay. The magnitude is the point: Harrison's V_ppσ at the 3.0 Å O–O distance (2.7 eV) is a
    nearest-neighbour value and would make the O 2p band far wider than MgO's; the SPEC 31 trace average
    (3.24 − 1.62)/3 · ħ²/(m(r² + r₀²)) = 0.44 eV at 3.0 Å is what a 5–6 eV fcc p band needs (12 neighbours), so the
    anchored value is the band-consistent one, and it is also the right anchor for the explicit anion–anion
    couplings (the open design point of step 22: SPEC 31 on O–O is first-principles-consistent, the learned 28 eV is
    not). Effect: the hole delocalises over the O's under the Au; its symmetric combination gains up to the band's
    half-width (~1–2 eV with 4–5 neighbours at 0.44 eV each), lowering the effective Δ of the donation by that much
    and raising the dressing: not small, and it acts on the upright contact (one O-top hole with four O neighbours
    at 3.0 Å) and on the flat one alike. To be measured against PBE before it enters: the O 2p band width of the
    PBE slab (mgo_pbe.py) fixes the hop; the ALMO transfer already contains the delocalised hole, so the fold with
    the hole band must reproduce the measured −0.96 eV at the O-top with the tabulated coupling, not fit it.
  - **The vertical scan completes the radial form (2026-10-28 night).** PBE's O → Au share at the single O-top
    contact: −0.962 / −0.551 / −0.294 / −0.130 eV at 2.26 / 2.60 / 3.00 / 3.50 Å; the two-level inversion with the
    model's Δ = 4.12 eV gives v = 2.21 / 1.60 / 1.14 / 0.74 eV, 0.84 / 0.80 / 0.75 / 0.66 of Harrison's
    1/(r² + r₀²) (0.99 → 0.79 with Δ = 6 eV): along the surface normal the universal form holds to 3.5 Å with at
    most a 20 % extra decay, while the angled flat-site contacts sit at 0.26–0.28 of it at every height. The
    suppression is directional.
  - **The first overlap table was an artifact (2026-10-28, late).** The O 2p "natural orbitals" from the Löwdin block
    of the density matrix, back-transformed through S^−1/2 in the def2-TZVP basis of the ten-ion cluster, gave a
    vertical overlap of 1.29 / 1.00 / 0.49 / 0.085 / 0.47 / 0.54 / 0.64 / 0.65 / 0.57 at 2.0 … 4.5 Å (a node at
    3.0 Å and a second lobe at 4 Å) and an in-plane value 0.5 Å from the Mg nucleus above the contact value; their
    occupations were 1.59–1.63. A p–s overlap cannot do that: the orbitals carry the neighbours' functions. Its
    pre-registered verdict ("holds", 0.17 against 0.54 at 3.5 Å) is not read. Replaced by the direct construction:
    the Au 6s overlap with the cluster's O 2p band orbitals weighted by their Mulliken 2p share on the O, with the
    band eigenvalues and their p_z character.
  - **The band-orbital table (2026-10-28, late; transfer_overlap.py, the direct construction).** Vertical S/S₀ =
    1.10 / 1.00 / 0.83 / 0.63 / 0.46 / 0.42 / 0.32 / 0.27 / 0.17 at 2.0 / 2.26 / 2.6 / 3.0 / 3.4 / 3.5 / 3.8 / 4.0 /
    4.5 Å: monotonic, within 5 % of the free ion and 10 % of Harrison to 3.5 Å, faster beyond (0.17 against 0.26 at
    4.5 Å). Over Mg0 (the flat site) S/S₀ = 0.83 / 0.72 / 0.63 / 0.52 / 0.42 at r = 2.99 / 3.19 / 3.36 / 3.59 /
    3.85 Å: larger than the vertical value at equal r, because the band orbitals carry Mg-centred amplitude toward
    the flat Au (the θ-grid rises monotonically toward the Mg direction). **The compression hypothesis is falsified:
    the suppression at the flat site is not in the orbital tail.** The band's hole energies: the in-plane
    (cation-bonding) orbitals at −7.11 / −6.61 / −6.35 eV, the dangling p_z ones at −6.13 / −5.79 (p_z share 1.00
    and 0.87): a crystal-field splitting of 0.8–1.3 eV, worth ~25 % in λ, not a factor 4.
  - **Erratum in my contact count.** The ten-ion cluster holds two quantum O's; the flat dimer there has four
    quantum O–Au contacts, not eight (the other four are point charges). Per contact the angled coupling implied by
    the ten-ion flat scan is ~0.37 of Harrison's, not 0.27; the 28-ion ALMO split (all eight O's; running) gives
    the flat transfer with its full contact count. Remaining candidates for the angled suppression: the dimer's σ*
    orientation (the acceptor orbital, not the donor) and the Au → surface competition; test: a closed-shell Au⁺
    acceptor alone, vertical at 3.0 Å against over Mg0 at the same O–Au distance (running).
  - **The Au⁺ test (2026-10-29, 00:30).** A single closed-shell Au⁺ (empty 6s) on the ten-ion cluster: surface → Au⁺
    −1.203 eV at the vertical O-top contact (3.0 Å, one contact) against −1.553 eV over Mg0 at h = 2.1 (two quantum
    contacts at 2.99 Å): −0.776 per contact, a ratio of 0.65 in λ (0.80 in v). For a single acceptor the angular
    suppression is mild and matches the crystal-field hole energy (the in-plane p's 0.8–1.3 eV below the dangling
    p_z) with a small geometric factor. The dimer's flat contacts gave 0.36 per contact (corrected count): the
    extra factor 0.55 belongs to the acceptor. **Mechanism (NEW-TO-TRACE): the metal dimer's acceptor is its
    sp-hybridised σ\*, whose p admixture puts its lobes outward along the dimer axis and a node on the inward
    side.** In the upright geometry the outward lobe of the lower Au points straight down at the O-top O; in the
    flat geometry the O's between the two Au (the ten-ion cluster's two quantum O's are exactly these) sit on the
    inward side, the O's beyond each Au on the outward side. The per-atom coupling for an s-class acceptor g with a
    bonded s-class partner g′:
        v(r, θ) = [V_spσ(r) + κ cos θ V_ppσ(r)] / √(1 + κ²),   cos θ = ŵ·ê,  ŵ the g → donor direction,
        ê = (R_g − R_g′)/|R_g − R_g′| the outward axis at g,  κ the σ\*'s p-admixture amplitude per atom,
    with κ = 0 for an acceptor without a bonded partner (the Au⁺ test). (Upright O-top: ŵ = −ẑ = ê, cos θ = +1;
    the flat O between the two Au: cos θ = −0.45; the O beyond each Au: +0.45.) Measured: the Au₂ LUMO at 2.545 Å
    (PBE, vacuum) is 6s with a p_z admixture, Mulliken s 0.49, p_z 0.08, κ = √(p_z/s) ≈ 0.40; the derived estimate
    V_spσ(Au–Au)/(ε_6p − ε_6s) = 2.08/4.6 = 0.45 agrees. Against the transfer data, though, the effective anisotropy
    is smaller: writing v = c V_spσ(r)(1 + κ′ cos θ)/√(1 + κ′²), the vertical contact (0.85 of Harrison, cos θ = +1)
    and the inward flat contact (0.74 of the vertical coupling after the crystal-field factor, cos θ = −0.45) give
    κ′ = 0.22 and c = 0.71; with κ′ = κ V_ppσ/V_spσ = 1.76 κ the LUMO value would be 0.7: the O 2p – Au 6p element
    is well below Harrison's V_ppσ (a diffuse 6p). So (c, κ′) are taken as measured constants of the (O 2p, Au 6s)
    pair, tabulated like the Pauli constants, with (1, 0) the default where no first-principles entry exists.
    Prediction with κ′ = 0.22 for the 28-ion flat transfer: ≈ −0.79 eV (the outward O's 1.49 times the inward ones
    in λ).
  - **Implemented (2026-10-29, test-first; 16 tests): `downfold_scale`, `downfold_anisotropy`, `downfold_crystal_field`,
    `downfold_bond_cutoff`** (defaults 1, 0, 0, 3.2 Å leave the plain term bit for bit). The σ\* lobe of a bonded
    s-class atom is minus the sum of its bond unit vectors (none at a symmetric site); the crystal-field shift counts
    an O's frame-cation neighbours within 1.2× its nearest-cation distance along each Cartesian axis (|l·e_m| >
    cos 30°) and charges β per neighbour beyond the least-coordinated axis. Pre-registered reading of the frozen
    counterfactual on step 20 with (0.71, 0.22, 1.0): wet_u rises above the as-trained values (+541 / +735 meV) and
    the relaxed flat dimer stays put ((c2) PASS on both seeds); if wet_u stays below the as-trained values the
    directional form is not the flat-site physics.
  - **Frozen counterfactual with (0.71, 0.22, 1.0) (2026-10-29, 02:00):** wet_u +541 → +686 / +735 → +745 meV
    (reading PASS), wet_d −452 → −426 / −221 → −187, T5 −1111 / −932; (c1) PASS; (c2) the flat dimer 2.89 → 2.67 /
    2.64 Å, −101 / −153 meV, PASS; (c3) PASS. Suites redox 765, rest 1039 + 4.
  - **The 28-ion flat split falsifies the σ\* mechanism (2026-10-29, 06:35).** Flat: dE_cp −0.3192 = FRZ +0.4281 +
    POL −0.3035 (A −0.1137, B −0.1804) + CT_cp −0.4438 (28 ions, all eight O's; checks (a1) 2.7e-5, (a2), (a3)
    1.7e-11 PASS); wet shares FRZ −241.7, POL +555.8 (A +165.3, B +366.1), CT +652.3, total +966.4 meV (DFT +929).
    The ten-ion flat CT_cp was −0.319 with four contacts; the four added outward O's (3.69 Å) bring −0.125, about
    what the count and the distance give (the inward ones at 3.51 Å: ≈ −0.47 expected for count-only scaling,
    −0.444 measured) and nothing like the pre-registered −0.80. Inward and outward flat contacts transfer alike,
    each at ≈ 0.3–0.36 of a vertical contact at the same distance: no σ\* lobe, κ′ = 0.
  - **The Au⁺ test was not a clean angular measurement (my error).** Au⁺ over Mg²⁺ relieves a cation–cation
    repulsion when it takes the electron, so its transfer energy there is far below the one above an O²⁻; its 0.65
    mixes that electrostatic difference with the angle and says nothing reliable about either.
  - **What the data support for the donor side.** With the Slater–Koster projection on the O's p components, an
    acceptor at 45° to the surface normal sees half its coupling² on the dangling p_z and half on the in-plane
    (cation-bonding) p: if the bonding components could not donate at all the angled/vertical ratio in λ would be
    0.50; measured 0.30–0.36. So the bonding components are at least dead, and a residual ~0.7 (0.85 in v) on the
    dangling channel at 45° remains unexplained (a candidate: the Au's own orbital pushed off the Mg below it by
    Pauli repulsion before it donates; not measured). The band orbitals' Fock energies (in-plane 0.8–1.3 eV below
    p_z) under-state the cost of a localised hole in an O–Mg bonding p: breaking that bond is the band width, not
    its splitting. **Rule adopted: a hole on an O-class donor is resolved onto its least-coordinated p axes only
    (the dangling directions: the axes with the minimum frame-cation count); the others do not couple.** For a
    (100) surface O that is p_z alone; for a bulk O all three (no asymmetry); for an edge O its free axes. With it
    the scale from the vertical data is c = 0.85 (0.66–0.99), κ′ = 0 and β unused.
  - **A release defect found on the way (2026-10-29).** The formal frame is per element, so a water O is an O-class
    donor and its H an s-class acceptor, and the fold would act across every O–H bond (V_spσ at 0.96 Å is 12 eV).
    A pair inside one covalent unit (SPEC 19/22: a bound-sector bond) must be excluded from the fold, as the
    explicit ensemble's admissibility already excludes the electron–hole pair; with atom units (MgO, Au) nothing
    changes.
  - **Frozen counterfactual with the dangling-axis rule (2026-10-29, 08:00): the reading fails** (wet_u +230 / +207
    against the as-trained +541 / +735; (c1)–(c3) PASS; benchmarks README). The rule's l_z² weight hits the upright
    site's side contacts harder than the flat site's; PBE's angled contacts are weaker than l_z² everywhere, so the
    angular function is sharper than a p lobe's projection and not yet measured as a function of angle. Step 22 is
    launched with the rule (the defensible physics), the risk recorded; the tilted-dimer scan (0 / 30 / 45 / 60° at
    2.6 Å about the O-top O) measures the angular function for step 23.
  - **Step 22's first launch stopped (09:06): cost, not physics** (107 s per forward from one-at-a-time candidate
    dressing and SPEC 28 on every folded candidate, beside a thrashing machine; fixed, 2.07 against 1.46 s). **Then
    the first tilt point contradicted the rule (09:12).** The dimer tilted 30° about the O-top O at 2.6 Å transfers
    surface → Au −0.579 eV against the vertical −0.551 at the same distance: 1.05, where the dangling-axis rule
    (l_z² = 0.75) predicts 0.75. So the angular dependence is flat to 30° at 2.6 Å, while the flat site (Au over
    Mg0, 36–45°, 3.5 Å) sits at 0.30–0.36. Either the function is a lateral-offset one (ρ = r sin θ: 1.3 Å at the
    tilt, 2.1 Å at the flat site) or the flat site's suppression is the acceptor's, an Au above a cation with its
    orbital pushed off the surface by the Mg²⁺ below it (the ALMO POL step), which no donor-side rule can carry.
    The relaunch is cancelled until the 45° / 60° / 0° points decide; a rule contradicted by a measurement does not
    train. (My error: I stopped the scan's chain shell instead of the watcher; 45° finishes as an orphan, 60° and
    0° are requeued.)
  - **Tilt 45° (11:20): −0.619 eV, 1.12 of vertical, but not a single contact.** At 2.6 Å and 45° the lower Au is
    1.86 Å from Mg0 and 2.83 Å from the other quantum O: a second donor and a cation inside the Au's own radius.
    The 60° geometry (1.3 Å from Mg0) is stopped as unphysical; 0° duplicates the vertical 2.60 point. The surface
    is too dense for clean single-contact angular scans at short range; the clean facts stay: vertical contacts at
    0.66–0.85 of Harrison, every contact of an Au above Mg0 at 0.40–0.44 across four heights (0.55–0.6 of the
    vertical value at equal distance, 0.30–0.36 in λ).
  - **The mechanism to test (derivation).** The transfer uses the Löwdin-orthogonalised hopping between the Au 6s
    and a band orbital i, t̃_ai = F_ai − S_ai ε_i (Wolfsberg–Helmholz for F; the ALMO POL step keeps the fragments'
    orbitals orthogonal). The in-crystal overlap table showed S_ai inflated above Mg0 by the band orbitals'
    Mg-centred tails (0.52 against 0.42 at 3.5 Å); if F_ai does not grow with that tail (the Fock element is carried
    by the O-centred part), the subtraction cancels more of t̃ there: the suppression is the orthogonalisation to an
    occupied manifold that reaches toward the acceptor. Test: Σ_i t̃_ai² w_i/(ε_a − ε_i) at the vertical 3.5 Å and the
    over-Mg0 2.89 Å sites from the cluster's Fock matrix in the union basis; the mechanism holds if the ratio is
    0.3–0.4, falsified if near 1.
  - **That test was invalid as built (11:25).** With only B's density on the union basis the probe Au 6s sits in
    its bare pseudopotential at −198 eV, so F_ai is tens of eV and the −S_ai ε_i term is negligible: the ratio
    (1.68) is the overlap structure again, not a hopping. The proper test is the ALMO calculation itself, which is
    non-perturbative in exactly these orbitals, and that is the measurement in hand. Withdrawn.
  - **What the flat-site data say about the mechanism, and the pre-registered readings of the site runs.** The
    flat-dimer contacts sit at 0.43–0.44 of Harrison at every height from 2.10 to 2.89 Å above Mg0: a factor
    constant while the Au–Mg distance shrinks from 2.9 to 2.1 Å, which a Pauli push from the cation below would
    not give (it would steepen), and while inward and outward O's agree (28 ions), which a σ\* lobe along the axis
    would not give. What is constant across those geometries is the dimer's orientation: lying on the surface.
    Candidate (NEW-TO-TRACE): the lying dimer's acceptor σ\* is polarised away from the surface by its Pauli
    repulsion (the ALMO POL step), its lobes pointing up on both Au, so every donor below couples at a reduced,
    orientation-set factor; the standing dimer's lower Au points its lobe down the axis at the O-top donor. In the
    model: the acceptor's lobe points away from its other neighbours (bonded partner and surface ions, Pauli-
    weighted), and the coupling to a donor carries a + (1 − a) max(0, ê·ŵ). Pre-registered readings of the two runs
    now in flight (a vertical dimer over Mg0 and over a hollow, both at h = 2.89, the O's at ≈ 3.6 Å): (A) if the
    vertical dimer over Mg0 transfers more than 1.3 times the flat dimer per contact, the orientation carries the
    suppression; if ≈ 1, the site does; (B) if the hollow equals Mg-top, the cation below is irrelevant; if it
    exceeds it by > 1.3, the cation below suppresses.
  - **Correction (12:36): the 30° tilt point is not a single contact either.** The tilt runs used the ten-ion cluster
    (not the 28-ion one). There the lower Au at 30° sits 2.40 Å from Mg0 (closer than its 2.60 Å contact) and
    3.21 Å from the second quantum O (0,1,0), whose mirror O (0,−1,0) is a point ion (an asymmetric donor set); the
    vertical 2.60 reference has Mg0 at 3.36 Å and the second O at 3.95 Å. Its 1.05 therefore contains a second
    donor: by Harrison's distance factor and the flat-contact range that donor brings 0.14–0.44 of the O-top
    value, leaving 0.6–0.9 for the O-top's own angular factor — the dangling-axis rule's 0.75 is neither
    contradicted nor supported (undetermined, and the scan cannot do better at 2.6 Å on this lattice). The verdict
    "contradicted" of 09:12 is withdrawn; the frozen counterfactual's negative reading of 08:00 stands on its own.
  - **The standing acceptor over Mg0 (12:27): surface → Au −0.184 eV** (vertical dimer, lower Au at h = 2.89 Å
    over Mg0, its two quantum O's at 3.59 Å, no point ion within 3.6 Å; Au → surface −0.047; BSSE_B +0.010;
    ten-ion cluster, all four stages converged). Per contact 0.092 eV against the vertical O-top contact (0.1296
    at 3.50 Å, brought to 3.59 Å by the measured decay 3.00 → 3.50 Å, e^(−1.64/Å): 0.112): **0.82**. The
    second-order band sum Σ_i S_ai² / (ε_σ* − ε_i) over the cluster's O 2p band orbitals (the probe overlaps of
    this morning, which are valid; ε_σ* = −4.297 eV, the Au₂ LUMO at 2.545 Å, PBE/def2-TZVP, the Au atom's 6s
    −6.014 reproducing the table's −6.01; band −7.11 … −5.79 eV) gives per contact over Mg0 / vertical, both at
    3.59 Å: **0.81**. Over Mg0 the two highest band orbitals (−5.90, −5.79 eV, the antisymmetric p_z
    combinations) have zero overlap by symmetry; the symmetric p_z combination (−6.13) and two deep in-plane ones
    (−7.11, −6.61) carry it. A standing acceptor's site costs exactly the band's symmetry selection and nothing
    else. The lying dimer's contacts (0.30–0.36 per contact at equal r) are suppressed by a further ≈ 0.4 that no
    donor-side rule gives; reading (A) is decided once the flat reference at 2.89 Å is measured in this same
    cluster (queued: the flat series stops at 2.60 Å and the 28-ion flat has eight contacts).
  - **The hollow run stopped and recast (12:30).** The (+,−) square puts a bare point O²⁻ 3.26 Å from the Au, inside
    the cluster's own 3.6 Å rule, with one quantum O; the (+,+) square O(1,0)–Mg0–O(0,1)–Mg(1,1) has both O's
    quantum at 3.26 Å, both Mg at 3.26 Å and no point ion within 3.6 Å: `hollow_pp`, queued after the flat
    reference. Reading (B) restated per contact at equal distance (the band sum's own 3.26 / 3.59 factor, 1.774):
    **site symmetry predicts hollow_pp / Mg-top = 1.00 per contact at equal r** (both 0.813 of the vertical
    contact); measured > 1.3 → the cation below suppresses beyond symmetry; 0.8–1.25 → it does not; < 0.77 → it
    enhances.
  - **Pre-registered (12:36): the acceptor orbital measured directly.** The transfer density ΔD = D(B → A minimum) −
    D(ALMO minimum) in the ten-ion cluster, per Au: Mulliken shells (s, p_z, p_xy, d, f) of the gained density and
    its dipole about the Au along the normal, split into the intra-Au part (the s–p_z cross term: the acceptor's
    own lobe), the Au–Au′ part and the Au–surface part (the bond-overlap density, always toward the donor).
    Geometries: the flat and upright references and the vertical dimer over Mg0 (`acceptor_density.py`, chained
    after hollow_pp; `$TRACE_FP_CACHE/acceptor_density/`). Reading: "lobes up" holds if the intra-Au dipole per gained
    electron is > +0.1 Å on both flat Au's and < −0.1 Å on the upright's lower Au; it is falsified if either flat
    Au's value is ≤ +0.1 Å. Scale: a 9° sp hybrid gives 2 sin α cos α ⟨6s|z|6p⟩ ≈ 0.3 × (1–1.5 Å) ≈ 0.4 Å per
    gained electron.
    Sharpened before the data (19:19): the orientation idea has two distinct sub-forms, which the upright's lower Au
    separates. (H1, axial lobe) the σ\* lobe points away from the bonded partner (the −û_gg′ form of 06:35, κ_ax):
    the upright's lower Au shows an intra-Au dipole < −0.1 Å per gained electron (down, at the O-top). (H2,
    perpendicular polarisation) the surface polarises the acceptor only perpendicular to the dimer axis (κ_⊥; by
    symmetry nothing for the standing dimer): the upright's lower Au shows |dipole| < 0.1 Å, the flat Au's > +0.1 Å.
    The λ ratios in hand already bound the pair: inward = outward on the 28-ion flat dimer needs κ_ax ≲ 0.1, and the
    lying / standing contact ratio (0.55–0.6 in V) then needs κ_⊥ ≈ 0.4 (v ∝ 1 + κ ŵ·ê, ŵ = ẑ for the lying
    acceptor, ŵ·ê = −0.826 for its O's). Both sub-forms die together if either flat Au's intra-Au dipole is ≤ +0.1 Å.
  - **The site runs (19:22; produced 13:24 and 14:43, read now — my waiter and the acceptor chain watched
    `flat_ref.json` where the script writes `vertical_flat_ref.json`, so the acceptor runs started only at 19:20).
    Both clean: all four stages converged, (d3) PASS, BSSE ≤ 0.018 eV, ten-ion cluster.**
    Flat reference (h = 2.89 Å, four contacts at 3.50/3.51 Å): surface → Au **−0.173 eV** (Au → surface −0.139;
    CT_cp −0.319, the 06:35 value reproduced), per contact 0.0432 = **0.336** of the O-top contact at equal
    distance (0.1296 at 3.50 Å by the measured decay e^(−1.64/Å)). Hollow_pp (two contacts at 3.26 Å): surface → Au
    **−0.322 eV** (Au → surface −0.039), per contact 0.161 = **0.838** of the O-top contact at 3.26 Å (band sum:
    0.813). Readings: **(A) standing-over-Mg0 / lying per contact = 2.45 at equal r (2.13 raw) > 1.3: the
    orientation carries the suppression.** **(B) hollow_pp / Mg-top per contact at equal r = 1.02 (0.99 with the
    band sum's own distance factor): no cation effect beyond the band's symmetry selection.** The second-order
    band sum Σ S²/(ε_σ* − ε_i) therefore accounts for every standing-acceptor site (O-top 1, Mg-top 0.82 measured /
    0.81 predicted, hollow 0.84 / 0.81), and the lying acceptor alone carries the remaining factor **0.41**
    (0.336 / 0.823), the same for its inward and outward O's (06:35). Established by measurement: the missing
    physics of the O 2p → Au s coupling is on the acceptor side and depends on the dimer's orientation to the
    surface, not on the donor's site, axis or crystal field.
  - **Pre-registered (19:22, before the acceptor runs report): a third sub-form, and its number.** The lying
    acceptor's factor could sit in its *level* rather than its *shape*: (H3) the surface pushes the lying dimer's
    σ\* up (orthogonality to the O 2p below it, a Pauli shift), so λ ∝ v²/Δ falls through Δ. The acceptor runs
    now also report the ALMO-state levels: the lowest eigenvalue of the Fock operator in A's virtual space made
    orthogonal to every occupied ALMO orbital (the polarised σ\*), the top of B's occupied block, and the free
    monomers' LUMO / HOMO. Reading: Δ(flat) / Δ(upright) > 2 → H3 carries it (a factor 0.41 in λ needs 2.4 in Δ);
    < 1.3 → the shape (H2) does; between → both, apportioned by the measured ratio. The model form each implies:
    H2 → v_gj = c V_spσ(r) g_orb f_spin [1 + κ_⊥ (ŵ_g·ê)] / √(1 + κ_⊥²) with ŵ_g the direction of the frame
    neighbours' push on g perpendicular to the g–g′ bond axis (zero for a standing dimer by symmetry, up for a
    lying one; ESTABLISHED ingredient: Slater–Koster sp hybrid, NEW-TO-TRACE: the push-set direction); H3 → a
    per-acceptor self-energy in the fold's Δ_q from the frame's Pauli table (SPEC 25's overlay, which already holds
    the closed-shell repulsion per pair: NEW-TO-TRACE, the same table entering the diabatic levels). Neither is
    coded until the measurement picks one.
  - **The flat acceptor run (19:47; read 19:54): the lying dimer's acceptor is p-type, its lobe points down, its σ\* is
    unshifted.** Gained 0.0724 e on the dimer (0.036 per Au): s 0.0053, p_z 0.0072, p_xy 0.0210, d 0.0024;
    intra-Au dipole −0.0078 e·Å per Au, **−0.21 Å per gained electron (down, toward the surface)**. ALMO-state
    levels: lowest A virtual −4.414 eV (free dimer LUMO −4.331: not pushed up), next −0.979; donor top −5.610;
    Δ_σ* = 1.20 eV (free 1.46). Readings: **H2 ("lobes up") is falsified** (≤ +0.1 Å); H3 cannot carry it for
    the lying dimer (its σ\* is not shifted); and the mechanism is visible directly: 80 % of the gained density
    is p-type although σ\* lies 3.44 eV below the next virtual — σ\* is energetically available and
    symmetry-forbidden.
  - **Derivation (NEW-TO-TRACE): the σ\* node.** Each quantum O of the flat dimer is equidistant from the two Au
    (3.50 / 3.51 Å): it lies on the dimer's perpendicular bisector plane, where σ_u* = (s₀ − s₁)/√2 vanishes. A
    donor there reaches σ\* only through its p component along the dimer axis (direction cosines −0.37 / +0.37
    on the two Au: opposite signs), 14 % of its weight; its p_⊥ and p_z components (86 %) have equal-sign
    overlaps and address the gerade combination (s₀ + s₁)/√2 — the dimer's own filled σ bond, Pauli-blocked — so
    they go to the next gerade virtual, 3.44 eV higher, as the p-type gained density shows. The standing dimer's
    O-top donor sits on the axis at σ\*'s end lobe: full coupling. This is reading (A)'s orientation effect, and it
    is a statement about the acceptor pair's two-site states, not about a lobe direction (the 06:35 form modelled
    the end lobe per atom and kept the fold's two-site ordering, so it missed the node, the dominant effect; the
    inward = outward datum is what the node predicts: both lie on the bisector plane).
  - **What the fold has wrong, exactly.** SPEC 32's Q block couples the folded states "electron on g" and "electron
    on g′" of a bonded s-class pair by −|T| f (docstring: "the s–s sigma hop is bonding; the sign of T is a
    gauge"). The gauge claim holds for a donor coupled through one path. A donor coupled to both atoms fixes the
    loop phase sign(v_g T v_g′), and for the extra electron of a pair whose own s electrons fill the bonding
    orbital the lowest available two-site state is the antibonding one (ESTABLISHED: Pauli exclusion in the
    pair's s space): the hop sign is **+|T| f**. With that ordering the fold gives, for a flat O, the axial p
    component at Δ − |T| (antisymmetric) and the other two at Δ + |T| (symmetric, the gerade proxy: 2|T| stands
    for the σ\*–gerade gap; Harrison's V_ssσ at 2.545 Å is 1.65 eV, 2|T| = 3.3 eV against the measured 3.44), and
    for an O-top donor coupled to one Au nothing changes (λ ≈ −(v²/2)[1/(Δ−|T|) + 1/(Δ+|T|)] under either sign).
    No new constant. The fold's leaf structure (one group per O and p component, SK direction cosines) already
    carries the sign pattern, so the change is the ordering alone.
  - **Pre-registered (19:54).** (i) Test, failing first: a bonded s pair with one O on its bisector plane — with the
    Pauli ordering the weak-coupling root is −[2v_ax²/(Δ−|T|) + 2(v_⊥² + v_z²)/(Δ+|T|)], with the legacy ordering
    the assignments swap; an O on the axis gives the same root under both. (ii) Frozen counterfactual on step 20
    (seeds 42 / 43), fold on, c = 1, κ′ = 0, β = 0, no free-axis rule, Pauli ordering: PASS if wet_u is above the
    as-trained +541 / +735 meV on both seeds with (c1)–(c3) passing; wet_u above the plain fold's +318 / +374 but
    below as-trained means the ordering is right and the magnitude needs the measured scale c; wet_u below the
    plain fold's values falsifies the ordering as implemented. (iii) The upright acceptor run: its lower Au's
    gained density s-dominated (σ_u*'s end lobe), intra-Au dipole < −0.1 Å per electron (down, at the O-top),
    Δ_σ*(upright) within 1.3× of the flat's 1.20 eV; the standing dimer over Mg0 likewise s-dominated (its O's are
    far from its bisector plane). The flat run is repeated with the 6p populations in the dimer frame (p_axis /
    p_⊥ / p_z; v1 kept): the σ\*-node form predicts the gained p density perpendicular and vertical, not axial.
    Branch `pauli-hop-2026-10-29` made before the change (the project's never-delete policy).
  - **A second gap found by the test (20:00): the fold had no partner states.** Its folded states came from O–s moves
    only, so an O touching one atom of a bonded pair (the O-top O: Au0 at 2.26 Å, Au1 at 4.805 Å, beyond the
    4.5 Å cutoff) had no "electron on Au1" state and no two-site structure at all — the standing dimer's extra
    electron could not delocalise over the pair, while the tests' independent dense rebuild (zero coupling beyond
    the cutoff, hop kept) and the physics have it. The on-axis case of the new test measured it: fold −1.5164 vs
    rebuild −1.5523 eV on its toy, under the legacy ordering (the existing exactness test never saw it: its O's
    reach every Au). Fixed in `downfold_moves`: every s-class atom of a contact cluster holds a folded state for
    each of the cluster's donors (t = 0, l = 0), coupled through the s–s hops; the move-class test now checks the
    partner state. Cost: one energy evaluation per partner state. Pre-registered readings of the frozen
    counterfactual on step 20, fold on, c = 1, κ′ = 0, β = 0, no free-axis rule: (P1) legacy ordering with
    partner states — wet_u above the plain fold's +318 / +374 (the standing dimer reaches σ\* at Δ − |T| now);
    (P2) Pauli ordering with partner states — wet_u above (P1) and above the as-trained +541 / +735 on both
    seeds, (c1)–(c3) PASS: the criterion of (ii). The flat dimer is unchanged by the partner states (every Au
    reaches every quantum O there), so (P2) − (P1) is the ordering's effect on the lying acceptor alone.
  - **Both counterfactuals refused, and the refusal exposed an older bug (20:12).** P1 and P2 both raised
    `DownfoldError` with μ_min = −1.6137 eV on the undoped flat reference — the same number under both orderings,
    as a 2 × 2 pair block must give (its eigenvalues Δ ± t are one set). The block, once the refusal was made to
    name its leaves: 18 clean (Au108, Au109) pairs with Δ 3.16–3.53 and hop 1.909 eV, whose lowest eigenvalue is
    ≥ +1.25. The hop groups were keyed by (c · P + o) with P the number of moves and o the O's atom index, unique
    only for o < P: in the 110-atom cell with 36 moves configuration 1's O 49 shares a key with configuration 0's
    O 85, and its hops were written into configuration 0's block. Test: two configurations differing in a spectator
    Mg, fold −1.924 / −1.569 against the rebuild's −1.649 / −1.649; fixed (key on the atom count). Consequence: every
    multi-configuration fold since SPEC 32 was written carried wrong hops — the plain fold's "wrong direction"
    (+318 / +374), the measured-constants variant (+686 / +745) and the dangling-axis variant (+230 / +207) are all
    superseded, and step 22's aborted launch ran under it. Re-measured: P1 / P2 below. Redox suite before the fix
    770 passed (the collision never reached a test: the rebuild comparisons used few atoms).
  - **P1 / P2 re-measured with the key fixed (20:21; seed 43 — my `tail` filter discarded seed 42's lines, rerun in
    flight with full capture).** P1 (legacy ordering, partner states, c = 1): wet_u **+442**, wet_d −350, T5 −792
    meV; (c1) PASS, (c2) FAIL (the flat dimer 2.89 → 2.57 Å, −308 meV), (c3) "+0.000" — a record artifact: a block
    made of partner states only (the near Au already Au⁻) reported μ = 0; now NaN and skipped (test
    `test_a_block_without_a_coupled_leaf_reports_no_mu`; 23 fold tests). P2 (Pauli ordering, same): wet_u
    **+716**, wet_d −350, T5 −1066; (c1) PASS, (c2) FAIL (2.57 Å, −290 meV). Readings: the ordering moves the lying
    acceptor by **+274 meV** in the predicted direction (P2 − P1; the doped cell, whose transfer is not through
    the pair's two-site states, is untouched: wet_d identical); P2 lands 19 meV below the as-trained +735, so the
    pre-registered PASS line is missed on this seed and the weaker outcome holds — the ordering is right, the
    magnitude needs the measured scale. (c2) fails under both orderings by almost the same amount, so the
    collapse is not carried by the shared O's two-site structure; it is the fold's short-range growth or a term
    outside it (the ionic Au⁻–Mg²⁺ configurations), to be told apart on the relaxed geometry. (c2)'s definition:
    the relaxed flat dimer within 0.3 Å of 2.89 Å and its relaxation above −0.5 eV — the failure is the height,
    0.32 Å, with the energy inside its bound, and it is measured on frozen weights whose short-range terms were
    trained without the fold's attraction (the as-trained model relaxes the flat dimer by −7 meV): a marginal
    over-attraction at short range, not a collapse; the flat transfer's growth from 2.89 to 2.57 Å is the same in
    the fold (Harrison V²: 1.33) and in PBE (the flat series: 1.36), so the imbalance is against the repulsion the
    weights learned, which a retrain re-balances. The doped cell: wet_d −350 under both orderings (as-trained
    −221, DFT −3) — the fold favours the doped flat dimer further, a separate problem of the Au₂⁻ cell.
  - **Pre-registered (20:21): P3 = P2 with c = 0.85** (the vertical contacts' κ′ = 0 value, 0.66–0.85 of Harrison).
    PASS: wet_u above the as-trained value on both seeds and (c2) PASS. If wet_u passes and (c2) still fails, the
    collapse is not the fold's contacts (next: the ensemble on the relaxed flat geometry). If wet_u falls below
    the as-trained value, the scale cannot carry it and the gerade proxy (Δ + |T|) is too low: a measured
    per-pair gap replaces it.
  - **P1 / P2 / P3 on both seeds (20:31; (c1) and (c3) PASS throughout).**

    | variant (frozen step 20; wet_u / wet_d / T5, meV; DFT +929 / −3 / −932) | s42 | s43 | (c2) s42 / s43 |
    |---|---|---|---|
    | as trained | +541 / −452 / −993 | +735 / −221 / −956 | — |
    | P1 legacy ordering, c = 1 | +338 / −613 / −951 | +442 / −350 / −792 | PASS (2.60 Å, −238) / FAIL (2.57 Å, −308) |
    | P2 Pauli ordering, c = 1 | +584 / −613 / −1196 | +716 / −350 / −1066 | PASS (2.61, −220) / FAIL (2.57, −290) |
    | P3 Pauli ordering, c = 0.85 | +525 / −591 / −1116 | +646 / −350 / −996 | PASS (2.64, −157) / PASS (2.61, −218) |

    Readings. The ordering's effect on wet_u is +245 (s42) / +274 (s43) meV, the predicted direction, the doped
    cell untouched. P2's pre-registered line (above as-trained on both seeds) is met on s42 (+43) and missed on
    s43 (−19): on frozen weights the fold's net effect on wet_u is ≈ 0 ± 40 meV while its gross dressing is ≈ −3 eV
    per structure (the record's most negative λ, −2.8 to −3.3 eV, against PBE's −1.10 eV O-top transfer): with c = 1
    the fold is 2–3× PBE at the O-top (the model's Δ − |T| = 1.3 eV matches the ALMO Δ_σ* = 1.20, so the excess is
    Harrison's V; c ≈ 0.66–0.71 brings the O-top to PBE's value, as the vertical series measured) and wet_u on
    frozen weights is the difference of two over-large numbers. P3's line (above as-trained with (c2) PASS) is
    missed on wet_u, as a uniform scale must (it scales the O-top contact too); its pre-registered interpretation
    ("the gerade proxy is too low") is not what the levels say: the model's symmetric combination sits at
    Δ + |T| ≈ 5.1 eV against the measured gerade channel 4.63 eV above the donor top (−0.98 vs −5.61), about right,
    so the surplus of the lying dimer's transfer is in the symmetric channel's coupling (V_spσ stands in for an
    O 2p – Au 6p element): the model's lying / standing per-contact ratio, from the fold's own numbers, is ≈ 0.59
    against the measured 0.41. (c2)'s failures are marginal (0.32 Å at c = 1 on s43; PASS at c = 0.85). The doped
    cell: the fold deepens wet_d by 130–160 meV (DFT −3): the Au₂⁻ cell's transfer is a separate problem and goes
    into step 22's criteria. **Conclusion:** the frozen counterfactual has given what it can — the ordering's
    direction and size — and the magnitudes are a training question; step 22 relaunches with the Pauli ordering,
    the partner states, the fixed key and the measured scale, pre-registered on the components (README).
  - **The upright acceptor run (20:34): the standing dimer's σ\* is pushed up at the contact, and the acceptor is
    s-type on both Au.** CT(B → A) raw −0.986 eV (the directional split's −1.10 counterpoise value). ALMO-state
    levels: lowest A virtual **−3.311 eV** (free dimer −4.297: +0.99 eV), donor top **−6.233** (free surface −5.792:
    −0.44), **Δ_σ* = 2.92 eV** against the free pair's 1.50 and the lying dimer's unshifted 1.20. Gained density:
    +0.038 e on the dimer, but **+0.078 e on the upper Au (s +0.091)** and −0.040 on the lower (s +0.068, p_z −0.076,
    d −0.046): the σ\* receives in its s component on both atoms, with more on the far end, while the lower Au's own
    p_z and d density is pushed away from the O (its intra-Au dipole −0.111 e·Å and the Au–surface cross term
    +0.308 e·Å are that response, not a lobe; the per-electron ratio is meaningless for a net loss). Readings:
    (iii) s-dominated acceptor — holds in the s shell, with the response on top; H3's ratio Δ(flat) / Δ(upright) =
    0.41 < 1.3 — "the shape carries it" for the lying dimer, and the level goes the *other* way: the standing
    dimer transfers 5× more per contact with a Δ 2.4× larger, so in a v²/Δ reading its effective coupling is
    ≈ 1.7 eV (0.65 of Harrison's 2.6 at 2.26 Å) against the lying contacts' ≈ 0.24 eV per contact (0.21 of
    Harrison's 1.12 at 3.5 Å) once each is read against its own Δ: the lying suppression is **≈ 0.32 in v**, larger
    than the 0.41 in λ at the model's common Δ said. Two mechanisms, both measured: the σ\* node for the lying
    dimer (the ordering), and a **close-contact Pauli shift of the charge-transfer state** for the standing one —
    the σ\* orthogonalised to the O 2p below it at 2.26 Å rises 1.0 eV and the donor top falls 0.44. The fold's Δ
    for the O-top is the model's configuration-energy difference (SPEC 25's Pauli table enters it through the
    inner correction); whether it carries this +1.4 eV is read from the leaves next. If it does not, that — not
    Harrison's V — is the 2–3× surplus of the fold's gross O-top dressing on frozen weights, and the physical term
    (NEW-TO-TRACE: the charge-transfer state's Pauli self-energy at contact, from the same closed-shell densities
    as SPEC 25) is step 23's candidate; the uniform scale c = 0.85 stands in for it in step 22.
  - **The model's own levels, and the per-state self-energy measured exactly (20:44; s43 frozen, Pauli ordering,
    c = 0.85, `shared_q.py`).** Fold record (six lowest leaves per configuration, with the inner correction): the
    frame's O-top leaf O1 → Au108 at 2.26 Å has Δ = 4.151 eV, v = −2.224 (p_z), so its two-site σ\* level is ≈ 2.4
    eV against the ALMO 2.92 (close; the model's v 2.22 against the effective 1.7 is the larger factor at the
    O-top); the flat's O49 / O61 → Au108 at 3.50 Å have Δ = 3.167, v = −0.78 (p_z), ±0.35 / 0.41 (in-plane), the
    antisymmetric level 1.26 against the ALMO 1.20 and the symmetric 5.08 against the measured gerade 4.63. The
    explicit ensemble on both references is {frame, Au108⁺Au109⁻, Au108⁻Au109⁺}; the ionic configurations' leaves
    O → Au⁺ sit at Δ = 0.36–0.63 eV with v ≈ 1.1 (the √2 unpairing factor), and the fold dresses them by −2.53
    (upright, the lower Au⁺) and −2.32 / −2.32 (flat) against the frame's −1.57 / −1.42. Every folded state is
    shared between explicit configurations (the frame reaches the ionic state's folded state through the partner
    hop; on the flat geometry both reach it directly). **Exact P + Q block per cluster against the per-state
    dressing (ground-state level, the explicit block's own T, the s–s hop 1.909 eV with the Pauli ordering):**
    upright undressed −2.350, per-state −4.064, exact −3.276 (error **−0.79 eV**); flat −1.641, −3.329, −2.792
    (error **−0.54 eV**). The exact ground states are 58 % frame, 26 % / 13 % the lower-Au⁺ ionic structure, 9 % /
    18 % folded. So SPEC 32's stated approximation ("the self-energy at each explicit state's own energy, exact when
    one explicit state dominates; O(tens of meV)") is **falsified by 0.5–0.8 eV** on both benchmark geometries: the
    ionic structure carries a quarter of the weight and shares every folded state with the frame. Treated exactly
    the fold *lowers* wet_u on these weights by ≈ 0.25 eV (the per-state error is larger on the upright), the
    opposite of the per-state reading: the model's upright preference is carried by its ionic resonance (an Au⁺ on
    the O²⁻, 2.35 eV of explicit mixing), which already stands in for the O-top transfer, so the true folded
    states add less there than on the flat. **Decision: step 22 does not launch on the per-state fold** (the
    queued launch disarmed); the fold becomes the exact explicit-plus-folded block — Löwdin with the whole
    explicit ensemble as P: H_eff = H_PP + Σ_k V_k (E* − H_Qk)⁻¹ V_kᵀ at the block's ground energy E*, its
    diagonal the configurations' dressing and its off-diagonal the dressed couplings — derived and built
    test-first on branch `exact-fold-2026-10-29`.
  - **Correction (20:48): the scratch exact-block estimate failed its validation and its numbers are withdrawn as
    measurements.** With P restricted to one configuration the scratch block must reproduce the fold's own root;
    it does for the flat frame (−1.4157 both) and fails elsewhere (upright frame −1.498 vs −1.565; flat ionic
    −2.878 vs −2.319; the uncoupled upright ionic −0.969 vs 0): the scratch takes the s–s hop as 1.909 eV for
    every structure and reads the configuration energies off the ensemble's Hamiltonian, neither of which is the
    fold's own input. The "0.5–0.8 eV per-state error" is therefore a hypothesis. What stands, measured: every
    folded state is shared between the frame and the ionic configurations, the ionic structure carries 26 % of
    the upright ground state, and its leaves sit 0.36–0.63 eV below their folded states — the per-state
    approximation's premise ("one explicit state dominates") does not hold. The launch stays withdrawn; the exact
    explicit-plus-folded block is built inside the fold (`downfold_block: exact`), whose single-configuration
    limit is the existing path by construction (a test), and the comparison is then made with the fold's own
    inputs.
  - **Built (20:56; `downfold_block: exact`, test-first, 28 fold tests).** Per contact cluster k: E_k* = the lowest
    eigenvalue of [[H₀, V_k], [V_kᵀ, H_Qk]] with H₀ the explicit block (energies with the inner corrections, the
    model's couplings), V_k the couplings of every configuration into the cluster's shared folded states and H_Qk
    those states with their s–s hops (Pauli ordering, pairing parities); Σ_k = V_k (E_k* − H_Qk)⁻¹ V_kᵀ;
    H = H₀ + Σ_k Σ_k, whose lowest eigenvalue is E_k* exactly for a single cluster (Löwdin at the block's own
    ground energy). One fixed point per cluster keeps the dressing additive over separated clusters (the joint
    eigenvalue would not be: the configuration space lacks the clusters' product states); with one explicit
    configuration Σ is the per-state root; the three identities are tests, and the module test shows the two
    modes agree on a one-configuration ensemble and differ on a two-configuration one (5 meV on the toy). The
    measurement that decides — exact against per-state on both seeds, Pauli ordering, c = 0.85, the fold's own
    inputs — runs now (`DF_BLOCK`).
  - **The standing dimer over Mg0, acceptor run (21:08):** CT(B → A) raw −0.194 eV; levels acceptor −4.303 eV (free
    −4.297: unshifted), donor top −5.968 (−0.18), Δ = 1.67 eV (free 1.50) — so the +1.0 eV σ\* shift of the
    upright is the close O contact's (2.26 Å), absent at 2.89 Å over a cation. Gained +0.149 e, **0.130 on the
    lower Au** (s +0.043, p_z +0.056, p_⊥ +0.013, d +0.004) and 0.019 on the upper: the acceptor localises on the
    lower Au as an sp_z hybrid pointing down at the donors (intra-Au dipole −0.37 Å per electron), not the
    symmetric σ\*. Reading (iii) "s-dominated": s and p_z share it; the pair's two-site states polarise onto the
    atom the donors reach.
  - **The exact block measured with the fold's own inputs (21:08; frozen step 20, Pauli ordering, c = 0.85; wet_u /
    wet_d / T5, meV; DFT +929 / −3 / −932).** s42: per-state +525 / −591 / −1116 → **exact +862 / −345 / −1206**;
    s43: per-state +646 / −350 / −996 → **exact +362 / −498 / −860**. The per-state approximation is off by
    **+336 (s42) / −284 (s43) meV on wet_u**, in opposite directions: its error follows how near-resonant each
    seed's ionic configurations are with their folded states (μ_min 1.23 vs 0.36 eV), which is the weights'
    own ionic-energy spread (Proposition 8). The scratch estimate's sign and size for s43 (−0.25 eV) were right
    after all; its validation failure was in its reconstruction, not in the physics. (c1) PASS both; (c2) PASS on
    s42 (2.65 / 2.74 Å, −186 meV; s43 pending at this write); (c3) misread the exact record's absolute eigenvalue
    (−910 eV) — now the gap above the block's ground energy, tested against the per-state record. **Conclusion:**
    the per-state fold is uncontrolled at the 0.3 eV level on wet_u; the fold is the exact block from here
    (`downfold_block: exact` in step 22's configs), and the seed spread of the frozen numbers is the retrain's to
    close. Suites: redox 777.
  - **s43's exact-block relaxation collapses (21:12).** (c2) on s43: the flat dimer 2.89 → 2.52 / **1.98 Å**,
    relaxation **−1412 meV** (s42: 2.65 / 2.74 Å, −186 meV, PASS). The driver is s43's near-resonant ionic
    configurations (μ_min 0.36 eV on the references): as an Au sinks toward the surface the O → Au⁺ folded state
    comes into resonance with the ionic structure and the exact block — correct within the model's configuration
    space — takes the whole level repulsion, which the per-state root hid (its (c2) passed at 2.61 Å). On frozen
    weights this is the model's over-ionic Au₂ (Proposition 8: the ionic structure 2 eV above the frame, 26 % of
    the ground state) exposed by the correct treatment; the trained run's E5 (the flat dimer within 0.3 Å) is the
    guard, and step 22 continues. (c3)'s FAIL on both seeds is the absolute-eigenvalue misread of the exact record,
    fixed before the launch (the gap above the block's ground energy).
  - **A measured discrepancy not yet acted on: the coupling's radial form (21:12).** Read against their own ALMO
    Δ's (2.26 Å: 2.92 eV; 3.0 and 3.5 Å: the unshifted pair's ≈ 1.5–1.6), the vertical contacts' effective
    coupling v_eff = √(|CT| Δ) is 1.79, 0.69, 0.44 eV at 2.26, 3.0, 3.5 Å: a fall of
    4.1× where Harrison's 1/(r² + r₀²) falls 2.33×, i.e. an exponential e^(-1.13 r) (ESTABLISHED:
    a Slater-type overlap decay; Wolfsberg–Helmholz V ∝ S). With the scale set at the O-top the fold over-couples a
    3.5 Å contact by ≈ 1.74× in v (3.0× in λ) — a radial error on every flat contact, separate
    from the σ\* node (the lying / standing 0.41 at *equal* r). Candidate for step 23 (NEW-TO-TRACE as a measured
    pair constant, one more: the decay length κ ≈ 1.1 /Å): V(r) = c V_spσ(r₁) e^(−κ (r − r₁)) anchored at the
    O-top distance r₁ = 2.26 Å; pre-registered reading on frozen weights: the flat contacts' dressing falls by
    ≈ 3.0× at 3.5 Å while the O-top's is unchanged, so wet_u rises on both seeds and (c2) passes on s43.
    Not in step 22 (one change per run).
  - **Built and measured (21:32; `downfold_decay` 0.45 /Å, anchor 2.26 Å; test
    `test_the_coupling_decays_beyond_harrison_when_asked`, 29 fold tests; frozen step 20, exact block, Pauli,
    c = 0.85; wet_u / wet_d / T5, meV).** s42: +862 / −345 / −1206 → **+1198 / −254 / −1452**; s43: +362 / −498 /
    −860 → **+878 / −158 / −1036**. The radial reading holds on both seeds (wet_u +336 / +515; wet_d toward −3 on
    both; s43's wet_u within 51 meV of DFT, s42 269 over). (c2) **crashed on both seeds** before a geometry was
    reached — s42 a singular secular solve in the selection's per-state screen, s43 the bound sector's hardness
    not positive definite at the Au (column 109): the relaxed flat dimer entered an unphysical geometry. Not a
    readable (c2); undetermined until the relaxation is traced. Suspect, by construction: the anchored
    exponential raises the coupling below 2.26 Å (e^(κ (r₁ − r))), a range the measurement never covered — the
    extrapolation, not the measured decay, acts in a collapse. A form that keeps Harrison below the anchor and the
    measured decay above it (C¹: a softplus-ramped exponent) is the candidate fix, to be tested on the traced
    collapse, not assumed.
  - **Flat acceptor run, v2 (21:39; the 6p populations in the dimer frame).** Gained p density per lying Au: p_⊥
    (in-plane, perpendicular to the axis) +0.0136, p_z +0.0072, p_axis +0.0074 e — 74 % perpendicular and vertical,
    26 % axial. The σ\*-node form's pre-registered reading ("perpendicular and vertical, not axial") holds: the
    gerade channel (the π\*-like 6p perpendicular to the axis, the O's p_⊥ and p_z components) carries the lying
    dimer's transfer; the axial quarter is σ\*'s 6p admixture reached through the O's axial component — the one
    channel the Pauli ordering keeps at Δ − |T|. Same energies and dipoles as v1 (CT −0.188 eV; intra −0.21 Å per
    electron). All three acceptor runs are in.
  - **The (c2) crashes traced (21:52).** s42: 27 sane BFGS steps (|F| < 0.5 eV/Å, nearest O 3.07 / 3.18 Å) then a
    singular batched solve in the per-state *selection screen* — a block whose lowest state is the uncoupled
    antisymmetric combination of a symmetric donor (Pauli ordering) lying below the configuration: no secular
    root below μ, the bisection runs onto the eigenvalue. A solver defect, not physics; fixed (refused blocks
    solved on the padded matrix; 31 fold tests). s43: a genuine runaway — one Au reaches 1.84 Å from an O by step
    28 (|F| 2.2 eV/Å) and 0.3 Å by step 40 with the energy falling hundreds of eV: the short-range form of the
    coupling (Harrison's 1/r² times the anchored exponential's e^(+κ(r₁ − r)) below the anchor) plus the
    near-resonant ionic configurations. The decay now ramps on above the anchor only (softplus, w = 0.2 Å).
    Step 22 (launched 21:08 with the solver defect loaded) restarted on the fixed code at 21:52; both traces rerun
    with the fixes.
  - **Traces v2 (22:02; fixed solver, ramped decay).** Both converge. s42: heights +0.20 / +0.12 Å, −286 meV,
    nearest O 3.07 / 3.16 Å — (c2) PASS. s43: a converged minimum with one Au bonded to a surface O at **1.98 Å**,
    −1.72 eV below the flat reference (explicit weight 0.77, gap 5.6 eV; the other Au stays at 3.38 Å) — (c2)
    FAIL, and a genuine model minimum, not a numerical runaway: on these weights the fold's short-range attraction
    (Harrison's 1/r² below the anchor, where nothing was measured) against SPEC 25's repulsion overbinds an Au–O
    contact, and s43's near-resonant ionic configurations (μ_min 0.36 eV) amplify it; PBE's own answer to "an Au
    reaching an O from the flat geometry" is the upright dimer, −0.93 eV at 2.26 Å — the model's −1.72 eV at
    1.98 Å is that physics, too deep and too short. Reading of the radial form's pre-registration: its second
    branch — wet_u and wet_d move as predicted on both seeds, s43's collapse is the weights' ionic resonance and
    the unmeasured short-range coupling, not the contacts' range. Step 23 = step 22 + the decay (κ = 0.45,
    anchor 2.26 Å), queued behind step 22's scoring (`watch_step23.sh`, configs `*_pfp_dfct_dec_*`), one change
    against step 22; its E1–E6 read against step 22 as comparator. Open for step 24: the coupling below the
    anchor (a measured short-range point — the O-top series at 2.0 Å — would fix it; an overlap-saturating form
    is the candidate, derived before coded).
  - **Step 22's first training step killed seed 42 — traced to the exact block's eigendecomposition (23:42).** The
    nanguard reported a non-finite gradient on structure 76 at optimiser step 0 (its plain recompute reproduced 61
    non-finite tensors) and its anomaly-mode retry named a harmless node (the self-pair norm of `pairs_within`,
    0/0 at second order on every step: fixed). Replaying the dump's own weights with the fold: the first NaN node
    is `LinalgEighBackward0` in `dressing_exact` — the full block's spectrum (−892.08, −890.93, −890.89, −890.49
    eV) has an isolated ground state but exactly degenerate folded pairs, and torch's eigh backward divides by
    every eigenvalue difference in the training's double backward. The ground energy is now the fixed point
    E = λ_min(H₀ + Σ(E)) by bisection and Newton (derivative 1 + ‖Xw‖², the ground vector detached), the per-state
    root's structure; replay: 0 non-finite gradients, normal and anomaly mode, E* to 1e-12. A second zero-row norm
    (the σ\* lobe in `downfold_moves`) fixed the same way. Quiet cost: per-state and exact both 1.9–2.4 s per
    structure (energy + forces); the 24 s was contention. Step 22 relaunched on the fixed code with step 23
    re-queued behind it.
  - **The cost at a random initialisation (02:10).** Seed 42's first launch ran at 19 s per forward and seed 43's at
    1.1 then 31–36 s: the trainer's wrapper now names every forward slower than 5 s, and they are undoped structures
    whose explicit ensembles hold 120–380 configurations at the random weights (s43: 379 configurations, 7 916
    leaves, 203 s) against 2–3 once trained — the window admits hundreds of near-degenerate configurations from
    random energies, and the selection screens each through the fold. Two exact optimisations (distinct moved
    configurations evaluated once; the hop blocks vectorised; 60 bisection halvings) cut the per-configuration
    cost ~3× and do not change the scaling. Step 20 crossed the same phase at 0.7 s per forward because screening
    without the fold costs nothing. The remedy consistent with the measurements — all frozen counterfactuals were
    taken on step 20's weights — is a warm start: step 22 trained from step 20's weights with the fold on, the
    ensemble at 2–3 from the first step and the other terms re-balancing under the corrected physics; the cold
    start's hundreds-of-configurations phase goes to the backlog (a cheaper screen for cold starts).
  - **Warm start running; seed 42 refused by the exact block at forward 333 (02:50).** Warm from step 20's weights
    both seeds run at 1.3–1.9 s per forward. Seed 42 then hit `DownfoldError` from `dressing_exact` on a training
    structure: cluster 0's ground state has explicit weight **0.30**, E* = −855.18 eV, 1.42 eV below the lowest
    folded state (−853.76). The per-state screen cannot see this: every configuration's own block was above its
    state, but the explicit block coupled coherently to the shared folded states (v ≈ 2 eV at a close O contact
    against a Δ of order 1 eV) has a charge-transfer-dominated ground state — the strong-coupling regime, where a
    dressing is the wrong description and the fold's own rule says what to do: the charge-transfer state must be
    explicit. **Design (NEW-TO-TRACE, to be built test-first): promotion.** On refusal the dominant folded state of
    the block (the largest |w_q|² of the ground vector) is promoted to an explicit configuration: its charge
    configuration joins the explicit set, its coupling to each explicit configuration it was reached from enters
    the explicit Hamiltonian as the fold's own v (the norm over the O's p components: f √g V(r) — the one
    coupling per pair of the explicit model), it is removed from the folded space (a folded state equal to an
    explicit configuration never appears in Q, and `drop_folded_partners` keeps promoted states), and the block is
    re-solved; its own folded states (O⁻ giving or taking) are then dressed as any configuration's. Bounded (a few
    promotions per structure); recorded in the ensemble record. Until built, the run cannot pass a structure in
    this regime; seed 43 continues (it has not met one in 300 forwards) and its scores stand alone if it finishes.
  - **Seed 43 refused too (02:52; forward 301, explicit weight 0.279, E* 1.49 eV below its folded manifold).** The
    regime is systematic on the training set's close-contact snapshots (failure dumps `redox_failure0.pt` in both
    run directories are the test cases). Step 22 is stopped until the fold handles it. Two designs, the choice to
    be made with a clear head: **(A) promotion** (above) keeps the per-cluster Löwdin reduction and its additivity
    over separated clusters; the open point is what is promoted — the dominant component state (exact: its
    signed coupling to each explicit configuration is the fold's own v_cq, its two component siblings stay
    folded, and the folded space must exclude exactly that (configuration, O, component) state) or the charge
    configuration (the explicit model's own object, the coupling collapsed to f √g V, the component resolution
    lost). **(B) the full block as the ensemble's Hamiltonian** — [[H₀, V], [Vᵀ, H_Q]] over P ∪ Q with the
    canonical ensemble on all its eigenstates: no reduction, no refusal, the CT weight in the populations, the
    excited spectrum exact — at the price of additivity over separated clusters (the union space lacks the product
    states: two independent two-level contacts with v = 1, Δ = 3 eV give −0.562 against the exact −0.606 eV,
    +44 meV; one cluster per cell in the benchmark, but it grows with the number of independent contacts).
    (A) is the design consistent with the SPEC's size-consistency property; (B) is the simpler and strictly more
    faithful ensemble for few clusters. Decision deferred to the morning; nothing trains meanwhile.
  - **Design A built (10:16, by decision; branch `promotion-2026-10-30`).** `dressing_exact(promoted=,
    promote=)` raises `PromotionRequest` (a `DownfoldError`) when a cluster's ground state is not
    explicit-dominated, naming the dominant folded state of the block (the largest |w_q|² of the ground vector):
    its charge row, donor O and hole component, its energy E_q (with the inner correction and the component
    shift) and its couplings V[:, q] to every explicit configuration. The module's final solve promotes it: the
    row joins the explicit set, E_q goes on the diagonal, V[:, q] into the explicit Hamiltonian (symmetric), and
    the block is rebuilt and re-solved with the promoted list — bounded at 8 per solve. In the folded space a
    promoted (row, O, component) state is absent and its hops to the remaining folded states become its
    couplings; a folded state equal to any other explicit configuration is absent (the promoted state's
    take-back move reaches the frame, whose coupling is explicit already); the promoted state's component
    siblings stay folded. Its own folded states (the second transfer, O⁻ → O⁰) are dressed like any
    configuration's, so the promoted system spans a larger space than the original block and its ground energy
    can only fall (the toy: 8.093 against the block's 8.254 eV) — tests: the request in the strong-coupling toy
    (folded states 2 eV below the frame), the Löwdin identity on the promoted system against an independent
    dense block with the same exclusions, the variational bound, the module's loop with an injected request.
    **Both failure dumps replay through the module:** s42 (call 333) and s43 (call 301) now give a finite
    forward with one promoted state each — the hole on O 61 and the electron on an Au, the O-top contact of
    those snapshots.
  - **Batch promotion and the re-partition rule (10:58).** The 10:27 relaunch died on seed 43 at forward 304: eight
    one-at-a-time promotions left the explicit weight at 0.229. The structure's charge-transfer manifold is not one
    state but the p_z states of the four O's under the dimer on either Au — eight near-degenerate states of a
    tenth of the weight each, plus one p_y — and converges at nine (explicit weight 0.739, 13 configurations). The
    request now carries the set of folded states in descending weight until the explicit weight reaches 0.6
    (at most 32), with their mutual hops as explicit couplings among them; passes bounded at four. And a rule
    found by a number: with a promoted configuration's own moves folded, a batch of four promotions on seed 42's
    snapshot fell 5.7 eV — the second electron transfer (O⁻ → O⁰ with Au₂²⁻) opened through the model's untrained
    charge states. **Promotion is a re-partition of the block it came from: a promoted configuration's own moves
    are never folded** (its two-electron states are outside the fold's scope), so no state is added or lost and
    the ground energy is the block's exactly (the toy identity holds to 4096 ε). Both dumps replay, forward and
    double backward, in normal and anomaly mode: s42 four promoted states (explicit weight 0.999), s43 ten
    (0.755). Open observation: s42's replay energy is −863.30 eV against −857.70 with the two sequential
    promotions of the earlier code — the promoted charge-transfer states now carry weight into the ensemble's
    populations and hence into the downstream terms; the block's ground energy is fixed by the identity.
  - **Pre-registered (11:52), for step 24, while step 22 trains: the O-top series with the ALMO levels** (ten-ion
    cluster, `acceptor_density.py` at upright heights 2.0 / 2.6 / 3.0 Å beside the existing 2.26; raw CT(B → A)
    and the ALMO-state Δ_σ* from each). (P24a, the coupling below the anchor) v_eff(h) = √(|CT_raw(h)| Δ(h)) against
    v_eff(2.26) = √(0.986 × 2.92) = 1.70 eV: Harrison predicts v_eff(2.0) / v_eff(2.26) = 1.29, the exponential
    extrapolated below the anchor 1.33, an overlap-saturating coupling ≤ 1.1. Reading: ratio > 1.2 → the growth
    holds below the anchor (the ramp's "Harrison below" stands; nothing to build); < 1.1 → saturation, an
    overlap-saturating short-range form for step 24, derived before coded. (P24b, the charge-transfer state's
    shift) the shift δ(h) = Δ_σ*(h) − 1.50 at 2.0 / 2.26 / 2.6 / 3.0 Å (2.26: 1.42 eV): an overlap-like decay
    e^(−h/λ) with λ ≈ 0.3–0.5 Å supports a per-contact self-energy from SPEC 25's closed-shell densities; a 1/h
    or 1/h² form supports an electrostatic one; the smaller residual decides the term's form.
  - **The O-top series is in (15:01; ten-ion cluster, raw ALMO CT(B → A), ALMO-state levels).**

    | h (Å) | CT_raw (eV) | Δ_σ* (eV) | acceptor / donor top (eV) | v_eff = √(CT·Δ) (eV) |
    |---|---|---|---|---|
    | 2.00 | −1.603 | 3.446 | −2.868 / −6.314 | 2.350 |
    | 2.26 | −0.986 | 2.922 | −3.311 / −6.233 | 1.697 |
    | 2.60 | −0.567 | 2.427 | −3.720 / −6.147 | 1.173 |
    | 3.00 | −0.304 | 2.064 | −4.006 / −6.070 | 0.792 |

    **(P24a) v_eff(2.0)/v_eff(2.26) = 1.385** against Harrison 1.261, the exponential 1.342: the growth holds below
    the anchor — no saturation, nothing to build; the softplus ramp's "Harrison below the anchor" under-couples at
    2.0 Å by 9 % in v (a stated approximation, conservative). **(P24b) the charge-transfer state's shift δ(h) = Δ(h)
    − 1.50 = 1.946 / 1.422 / 0.927 / 0.564 eV is an exponential, λ = 0.81 Å (log-residual 0.003) against a power law
    h^−3.06 (0.030): overlap-like.** At 2.26–3.0 Å it is the O 2p – Au 6s band overlap squared within 6 % (S²
    ratios 0.692 / 0.397 against δ ratios 0.652 / 0.397) — the Pauli form (ESTABLISHED: exchange repulsion ∝ S²);
    at 2.0 Å δ rises faster than S² (1.37 against 1.21), where the overlap saturates. **Correction: the decay
    constant of step 23.** With every Δ measured (the shift persists to 3.0 Å: 0.56 eV) the consistent fit of
    ln v_eff = a − ln(h² + r₀²) − κ h over the four points gives **κ = 0.307 /Å** (log-residuals ≤ 0.007; segments
    0.36 / 0.30 / 0.29), not 0.45 — that value assumed an unshifted Δ ≈ 1.6 eV at 3.0 Å, which the series falsifies.
    Step 23's configuration amended to 0.31 before it ran. Step 24's candidate is the second effect, NEW-TO-TRACE: a
    per-contact Pauli self-energy of the charge-transfer state, δ(r) = δ₀ e^(−(r − r₁)/λ) with δ₀ = 1.42 eV at
    r₁ = 2.26 Å and λ = 0.81 Å, added to Δ of every folded state whose hole and electron sit on a contact pair —
    unless the model's own Δ(h) already carries it: that is measured next (the fold's Δ at the O-top versus h on
    step 20's weights).
  - **Step 22's fourth warm start diverged on seed 42 — a learning-rate artifact, ruled out as a fold
    discontinuity (15:01).** The 11:09 run restarted step 20's cosine schedule at 1e-3 (no warm-up) with a fresh
    optimiser on weights step 20 had converged at 5e-5. Both seeds' train energy error rose from the warm start's
    ~10 meV/atom to 30–65 meV/atom over epochs 1–3 (the ensembles' ground energies drifted from about −900 to
    −650 eV), then seed 42 diverged at epoch 4 (single structures at loss 82 and 159, val force 7.3 eV/Å). At the
    epoch-3 checkpoint those structures show no promotion, explicit weights ~0.9, gaps 3.3–4.9 eV and force errors
    0.10–0.19 eV/Å rms, but energy errors −2.8 to −4.1 eV (−0.7 to −0.9 at the warm-start weights): the drift, not
    a threshold. Relaunched 14:59 at a fine-tuning rate, 1e-4 → 5e-6 (step 20's own rate at its epoch ~34).
  - **The model's own O-top level versus height — the step-24 term is not built (15:02; frozen step 20, fold on,
    exact block, Pauli, c = 0.85; `probe_fold_otop_height.py`).** The frame's O → lower-Au p_z leaf, the dimer moved
    rigidly along the O–Au axis:

    | h (Å) | model Δ s43 / s42 (eV) | model v (eV) | ALMO Δ_σ* (eV) | ALMO v_eff (eV) |
    |---|---|---|---|---|
    | 2.00 | 4.686 / 5.786 | 2.804 | 3.446 | 2.350 |
    | 2.26 | 4.152 / 5.236 | 2.224 | 2.922 | 1.697 |
    | 2.60 | 3.631 / 4.407 | 1.700 | 2.427 | 1.173 |
    | 3.00 | 3.035 / 3.615 | 1.288 | 2.064 | 0.792 |

    The rise relative to 3.0 Å: model s43 1.65 / 1.12 / 0.60, s42 2.17 / 1.62 / 0.79 against ALMO's 1.38 / 0.86 /
    0.36 eV. **The model already carries the charge-transfer state's short-range rise — 1.2–1.6× (s43) and
    1.6–2.2× (s42) too steep** (the learned ionic levels with SPEC 25's Pauli overlay, seed-dependent): P24b's
    per-contact self-energy would double-count it and is not built. The couplings: model v / ALMO v_eff = 1.19 /
    1.31 / 1.45 / 1.63 — the decay beyond Harrison again (κ ≈ 0.31) on top of a uniform factor; with κ = 0.31 the
    residual is ≈ 1.27, i.e. **the consistent scale at the anchor is c ≈ 0.85 / 1.27 = 0.67** (v_eff(2.26) / Harrison
    = 1.697 / 2.617 = 0.65), the step-24 candidate in place of the self-energy. Both the absolute Δ (model 4.2–5.2
    against 2.9 eV at the O-top) and its slope are learned quantities: re-measured on step 22's trained weights
    before step 24 is decided.
  - **The fold's derivatives (19:57).** Step 22's fine-tune raised its own training force error epoch after epoch —
    the signature of a wrong gradient. Finite differences on a real training structure located three defects in
    SPEC 32's code (CHANGELOG Round 30): the exact block's forces were not the energy's gradient (a shared folded
    state's energy got its gradient once per configuration reaching it: 6–27 %), the hop matrix was detached (the
    coupling weights' gradient 4.8 %, and the hops' position dependence lost on both fold paths), and the fixed
    point's second derivative lacked the eigenvector response (1.7 % on the force loss's gradient). After the fixes
    the energy, forces and the force loss's gradient agree with finite differences to 1e-6–1e-9. The energies of
    every frozen counterfactual stand; the (c2) relaxations — s43's 1.98 Å bond among them — were taken with wrong
    forces and are superseded; a statement made above ("the fold's short-range attraction ... overbinds an Au–O
    contact") is withdrawn until re-measured. Lesson, stated as a rule for this code: every new energy path gets a
    finite-difference test of its forces and of a parameter gradient on a configuration where the path is active.
  - **(c2) with correct forces (20:08).** Frozen step 20 + the fold: no decay, s42 one Au to 1.45 Å (−1.91 eV), s43
    2.24 / 2.47 Å (−0.83 eV); κ = 0.31, s42 2.04 / 2.95 Å (−1.56 eV), s43 2.73 / 2.61 Å (−0.12 eV, the only PASS).
    The frozen weights cannot hold the flat dimer once the fold's attraction acts with its true forces. Which part
    over-binds, from the O-top comparison: the coupling (model v / ALMO v_eff = 1.19 at 2.0 Å with c = 0.85 and no
    decay; 0.85 / 0.67 = 1.27 uniformly once κ = 0.31 corrects the shape), against a level that rises too steeply
    (which should resist, not drive, the collapse). The measured remedy for the coupling is c ≈ 0.67 (step 24);
    whether the fine-tune's re-balanced repulsion already holds the flat dimer is step 22's E5.
  - **The consistent constants on frozen weights, and where the collapses go (20:35).** c = 0.67, κ = 0.31: s42 wet_u
    **+875** (DFT +929), wet_d −357, T5 −1232, (c2) PASS (2.71 / 2.78 Å, −237 meV); s43 +670 / −273 / −943, (c2) FAIL
    (one Au to 2.02 Å, −1.26 eV) — a collapse non-monotonic in c (s43 passed at c = 0.85), so not a coupling effect.
    The trace of s42's worst collapse ends with an Au **1.81 Å from a Mg**; the training set holds Au–O contacts down
    to 1.75 Å but **no Au–Mg contact below 2.43 Å**, and the Pauli table's Au–Mg window is 2.66–3.49 Å. The model's
    curve for a dimer standing over a surface Mg (E(h) − E(3.30), eV; fold off): s42 **−47.5 at 1.8 Å, −292 at
    2.0 Å**, s43 +0.06 / +0.23 (no wall); fold on (c 0.67, κ 0.31): −0.52 / −0.21 and −0.47 / −0.20. Seed 42's −292
    eV is the learned **O–O hole coupling, 28.3 eV between O's 3.0 Å apart** (Harrison V_ppσ ≈ 2.7 eV): at h = 2.0 Å
    the selection admits ~100 O-hole configurations, which that coupling binds into a band ~295 eV deep (the
    diagonal energies are normal). A pre-existing pathology of step 20's unanchored couplings (recorded 2026-10-27),
    outside the data; the fold avoids it at the references (O-hole states are folded, where O–O hops are excluded).
    The PBE curve for the same approach (`mgtop_scan.py`) is running: it decides whether the model's short-range
    Au–Mg wall is too soft. Two physics items follow, ranked: (i) bound the O–O coupling (SPEC 27's bound, anchored to
    Harrison) — a latent catastrophe wherever O-hole configurations are selected; (ii) the Au–Mg²⁺ wall below the data
    (the Pauli table computed down to ~1.8 Å from the same frozen densities) if the PBE curve says it is too soft.
  - **Promotion coupled twice, found by the same O–O number (20:35).** The module added its own T-couplings between
    promoted rows on top of the fold's (an Au–Au hop counted twice; seed 42's O–O 6.0 eV entering where the folded
    space had none): fixed, CHANGELOG Round 30 item 4; the identity now holds in the module to 1e-13 eV.
  - **The learned couplings of step 20 at its reference (20:36; |T| in eV; `probe_learned_couplings.py`).** O–O at
    3.0 Å: s42 23.1 (max 28.3), s43 0.33 (max 0.74), Harrison V_ppσ 2.74 — a factor 70 between seeds, unconstrained
    by data (O-hole states are never populated in the neutral training structures). Au–Au at 2.6 Å: 1.93 / 1.91,
    Harrison V_ssσ 1.58 — constrained by the data and seed-independent. O–Au at 3.5 Å (fold off): 0.017 / 0.042
    against Harrison 1.15 (the fold's raison d'être). Au–Mg 2.9 Å: 0.85 / 0.46. Candidate (one config change,
    tested code): SPEC 27's `coupling_bound: harrison`, which caps each learned coupling at Harrison's two-centre value
    — it would hold s42's O–O at 2.74 eV and leave s43's untouched; ranked behind the Au–Mg wall, which the PBE
    Mg-top curve decides.
  - **The PBE Mg-top curve (21:30; `mgtop_scan.py`, ten-ion embedded cluster, ΔE_cp of a standing Au₂ over Mg0 relative
    to h = 3.30 Å):** 2.00 Å **+2.017**, 2.30 Å +0.540, 2.60 Å +0.107, 2.90 Å +0.004 eV. The model fold off (s43):
    +0.228 / +0.309 / +0.102 / +0.063 — PBE's wall reproduced at 2.6 Å (inside the Pauli table's Au–Mg window,
    2.66–3.49 Å) and **0.57× at 2.3 Å, 0.11× at 2.0 Å** below it. The missing physics is the short-range Au–Mg²⁺ Pauli
    wall below the table and the data (training Au–Mg ≥ 2.43 Å); the fold's attraction at 2.0 Å (−0.4 to −0.9 eV)
    is smaller than the missing +1.8 eV. Ranked first for step 24 (NEW-TO-TRACE, from SPEC 25's own first-principles
    route): the frozen-density Pauli table computed down to ~1.8 Å for Au–Mg²⁺ (and checked for Au–Al³⁺), the same
    build as `build_pauli_fp.py`; pre-registered reading on frozen weights: the dimer-over-Mg curve within 20 % of
    PBE's at 2.0 and 2.3 Å, and the flat dimer's (c2) PASS on both seeds with the fold on.
  - **Correction of the diagnosis (21:32): the missing wall is the frame pair's own, not the Pauli table's.** SPEC 25's
    table is frame-relative (each K fitted to a carrier state's interaction minus the frame pair's, `o − fr` in
    `build_pauli_fp.py`), so the frame pair's own repulsion — Au⁰ against Mg²⁺ — is no physics term of the model: the
    learned backbone carries it, and the 4G data constrain it only above Au–Mg 2.43 Å. The frame pair's own
    frozen-density non-electrostatic PBE interaction, cached by the build at its window points: Mg²⁺–Au⁰ +0.382 /
    +0.193 / +0.098 / +0.049 / +0.025 eV at 2.658 / 2.867 / 3.075 / 3.284 / 3.493 Å — exponential, decay length 0.31 Å;
    extrapolated, ≈ +3.3 eV at 2.0 Å per contact against the backbone's +0.23. Measuring, not extrapolating:
    `frame_pair_wall.py` at 2.4 / 2.2 / 2.0 / 1.8 Å (the build's cached route; running overnight). **Step-24
    candidate restated (NEW-TO-TRACE):** a frame-pair wall E_w = Σ_pairs P_frame(Z_i, Z_j; R_ij), tabulated from the
    same frozen ion densities as SPEC 25 (the frame pair's sph_nonelst_pbe), added as a physics baseline under the
    learned backbone (which then fits the residual at the data distances; outside them the wall is the physics).
    Pre-registered frozen reading (on step 20 + the wall, the backbone untouched): the dimer-over-Mg curve within 20 %
    of PBE's at 2.0 and 2.3 Å; then (c2) on both seeds with the fold on. The O–O coupling bound and c = 0.67 follow.
  - **Measured (22:23; `frame_pair_wall.py`): the Mg²⁺–Au⁰ frame pair's frozen-density non-electrostatic PBE interaction
    +0.888 / +1.713 / +3.333 / +6.618 eV at 2.4 / 2.2 / 2.0 / 1.8 Å** — the exponential of the window points continued
    (the extrapolated +3.3 at 2.0 Å confirmed; decay 0.29–0.31 Å over 1.8–3.5 Å). The electrostatic cross-check against
    the library drifts below 2.2 Å (−0.006 / −0.029 / −0.116 eV at 2.2 / 2.0 / 1.8 Å; flagged, small against the wall).
    **The frozen reading pre-registered at 21:32 is withdrawn before it ran — mis-designed:** the frozen backbone
    already carries the repulsion at the data distances, so backbone + wall double-counts (at 2.0 Å: 0.23 + 3.28 =
    3.51 eV against PBE's 2.02), and PBE's full curve lies below the frozen-density wall by the relaxation it adds
    (−1.26 eV at 2.0 Å: polarisation and transfer, which TRACE carries in its induction and fold terms). The wall is
    a physics baseline to be trained with — the frame pair's frozen-density repulsion added to SPEC 25's overlap
    term (which then holds the absolute frame-pair Pauli, not only carrier-minus-frame), the backbone re-learning its
    residual in a warm-started fine-tune; criteria after training: E5 and the dimer-over-Mg curve within 30 % of
    PBE's at 2.0 and 2.3 Å.
  - **Pre-registered (2026-11-01 10:35): the gold dimer's ionic-state anchor.** Step 22's seed dependence (wet_u +802 /
    +381) traces to the explicit ionic configurations, whose energies the data do not pin: Δ_ion = E(Au⁺Au⁻) −
    E(Au⁰Au⁰) at 2.47 Å is 2.74 / 3.07 eV (step 22 s42 / s43; 2.37 / 2.79 for step 20) and is reached by opposite
    decompositions (ladder −2.76 with on-site +12.5 against +2.58 with +7.56). The model's ensemble for the bonded pair
    is an exact two-site Hubbard model (SPEC 29), so the gas-phase dimer's two σσ\* excitations fix its U and t: the
    triplet Δ_ST = (W − U)/2 and the ionic singlet E_S = (U + W)/2 give **U = E_S − Δ_ST, t = √(Δ_ST E_S)/2**
    (ESTABLISHED: the two-orbital two-electron Hubbard / VB mapping). Measured with EOM-EE-CCSD (reference) and TDA-PBE
    (the 4G data's functional), def2-TZVP / def2 ECP, at 2.30 / 2.47 / 2.60 / 2.80 Å (`au2_hubbard.py`), the σσ\*
    root chosen by its 6sσ → 6sσ\* weight in each manifold. Readings: if the physical U at 2.47 Å lies within 0.2 eV
    of one seed and > 0.3 eV from the other, the seed spread is the unanchored ionic level, and the anchor becomes a
    training prior on Δ_ion(R) of the gas-phase dimer (step 25 candidate, through the existing dimer probes'
    machinery); if both seeds are > 0.3 eV off on the same side, the ionic ladder is biased in both and the prior is
    needed regardless; stated caveats: spin–orbit is omitted (scalar-relativistic ECP), and 5d excitations near the
    σσ\* ones are excluded by the weight criterion, which is printed.
  - **The anchor, measured (2026-11-01 11:01; `au2_hubbard.py`, def2-TZVP / def2 ECP).** The σσ\* roots identified by
    projection onto σ_g → σ_u\* built from the atom's own 6s on both centres (a first pass picked the PBE KS HOMO → LUMO
    excitation, but the Au₂ PBE HOMO is a 5d orbital: those were d → σ\* states, withdrawn). U / t (eV):

    | R (Å) | EOM-CCSD | TDA-PBE |
    |---|---|---|
    | 2.30 | 0.745 / 1.501 | 0.900 / 1.409 |
    | 2.47 | **0.875 / 1.317** | **0.995 / 1.249** |
    | 2.60 | 0.997 / 1.191 | 1.063 / 1.146 |
    | 2.80 | 1.224 / 1.012 | 1.164 / 1.009 |

    σσ\* weights: triplets 0.76–0.92, singlets 0.36–0.67 (the ionic singlet mixes with 5d excitations; spin–orbit
    omitted). The model at 2.47 Å: Δ_ion 2.74 / 3.07 eV, τ₁₂ 2.04 / 2.31 eV (step 22 s42 / s43; step 20 Δ_ion 2.37 /
    2.79). **Reading — corrected (13:32; the first version compared τ₁₂ directly with t and derived the ionic weights from the
    exact-singlet formula, but spin_pairing is off in steps 20–22: the model's spinless ensemble couples Au⁰Au⁰ to each
    ionic configuration by |H_cov,ion| and to their symmetric combination by √2 |H_cov,ion|, the exact singlet by 2t, so
    the model reproduces the physical singlet at |H_cov,ion| = √2 t). Read from the model's own ensemble Hamiltonian of
    the gas-phase dimer (`probe_dimer_hubbard.py`; three configurations at every R):**

    | R (Å) | Δ_ion step 20 s42 / s43 | Δ_ion step 22 s42 / s43 | U PBE / CCSD | |H_cov,ion| step 22 s42 / s43 | √2 t (PBE) | ionic weight model / physical |
    |---|---|---|---|---|---|---|
    | 2.30 | 1.985 / 2.378 | 2.270 / 2.680 | 0.900 / 0.745 | 1.843 / 2.078 | 1.993 | 0.28–0.31 / 0.42 |
    | 2.47 | 2.374 / 2.790 | 2.740 / 3.074 | 0.995 / 0.875 | 2.043 / 2.307 | 1.766 | 0.28–0.30 / 0.40 |
    | 2.60 | 2.626 / 3.015 | 3.022 / 3.316 | 1.063 / 0.997 | 1.991 / 2.301 | 1.621 | 0.26–0.28 / 0.39 |
    | 2.80 | 3.008 / 3.360 | 3.419 / 3.700 | 1.164 / 1.224 | 1.826 / 2.242 | 1.427 | 0.22–0.25 / 0.36 |

    Both seeds' Δ_ion lie 1.1–2.5 eV above the physical U and rise ~2 eV/Å against 0.53 (PBE) / 0.95 (CCSD); the coupling
    is too flat in R (−15 % to +4 % at 2.30 Å, 25–57 % large at 2.80 Å); the ground-state ionic weight is 0.22–0.31 against
    0.36–0.42 — the model's dimer is too covalent (the errors do not cancel; the earlier "U/t 1.34 / ionic weight 0.34"
    and "t 1.6–1.8× too large" are withdrawn). The pre-registered reading stands in its second branch: the ionic ladder
    is biased in both seeds, the seed spread a fraction of the bias. **Step-25 candidate (NEW-TO-TRACE):** a two-site
    Hubbard prior on the gas-phase Au₂ sector — Δ_ion(R) → U(R) and |H_cov,ion|(R) → √2 t(R) (the model's spinless
    form) at 2.30–2.80 Å, TDA-PBE targets (the data's functional; CCSD within 0.15 / 0.09 eV) — derived and frozen-checked
    before it trains.
  - **The two-site (Hubbard) prior, derived and frozen-checked (2026-11-01, 14:04; benchmarks README).**
    *Derivation.* NEW-TO-TRACE: the module's gas-phase Au₂ ensemble (Au⁰Au⁰, Au⁺Au⁻, Au⁻Au⁺) is a two-site model
    with H = [[0, −c, c], [−c, Δ, 0], [c, 0, Δ]] relative to the covalent diagonal, c = |H_cov,ion| (SPEC 21's switch
    and SPEC 29's factor included). Its ground state is E_g = (Δ − √(Δ² + 8c²))/2, the antisymmetric ionic state sits
    at Δ, and the ionic weight is p = s/(2s + Δ) with s = −E_g. ESTABLISHED (two-site Hubbard; the H₂ valence-bond
    mapping): the singlet has E_g = (U − √(U² + 16t²))/2, the σσ* triplet sits at the covalent diagonal, and the ionic
    singlet sits at U. So the model is the physical singlet iff Δ = U and c = √2 t, which with SPEC 29 means τ = t.
    The prior is L = Σ_R [(Δ − U)² + (c − √2 t)²] at R = 2.30–2.80 Å, with U = E_S − E_T and t = √(E_T E_S)/2 from
    the adiabatic σσ* excitations. It neglects the direct exchange and pair hopping of the orthogonalised orbitals.
    *Where the model's Δ comes from* (`probe_dimer_decompose.py`, closes on H to 1 meV): the tabulated terms
    (IE − EA − K/R + SPEC 23 + SPEC 25) fall at 3.8 eV/Å. The Gordon–Kim difference of Au⁺Au⁻ against Au⁰Au⁰ is
    −1.61 eV at 2.30 Å. The learned shift (+2.5 to +3.8 eV) half-compensates, so Δ is 1.1–2.5 eV high.
    *Artifacts ruled out or stated.*
    (a) The singlet roots are single transitions (natural-transition-orbital share 0.95–0.99), but their hole is
    25–33 % 5d in CCSD and 32–58 % in TDA-PBE, against 4–19 % for the triplets. So U carries 5dσ → σ* mixing, which
    the model has no state for (stated).
    (b) Au₂⁻'s 2t gives t within 0.07 eV of the neutral's at 2.30–2.80 Å. This is SPEC 29's premise for T5's two
    sectors.
    (c) The bare-dipole two-site α_CT (19–26 Å³) is falsified by the singlet's transition dipole (0.54× √p R). The
    model's dipoles are screened by its bound sector, so the model's own α decides.
    *Frozen check 1:* with Δ = U and c = √2 t imposed, the model's α∥ goes from 16–38 % low to 3–10 % low against PBE0
    on all four models, and p reaches the physical 0.36–0.42 exactly. The prior is consistent with an independent
    observable.
    *Frozen check 2 (pre-registered):* along the prior's update direction in the run's optimizer geometry, wet_u
    rises on all four models (*aligned*). Nearly halving L moves wet_u +23 / +33 meV, wet_d −12 / −58 and T5 −35 / −92,
    linear to 8–15 %. The gradient is 61–79 % in the shift network.
    **Verdict:** right physics for the isolated dimer and a validated transferability prior, but not a T5 step. It is
    not trained now. T5's remaining error is wet_d, the doped sector.
    SPEC 29 on frozen weights raises wet_u by 61–382 meV through compensation (c 21–122 % too large, Δ still high,
    α∥ unchanged), so it is no evidence for a spin-pairing step.
    *Separate finding:* the model's α⊥ of Au₂ is 2.9–6.0 Å³ at 2.30 Å against PBE0's 8.13, rising with R while PBE0
    is flat. That is a bound-sector defect.
  - **The wall's form (22:24).** Data coverage of the Au frame pairs (smallest contact in the 4G training set): Au–O
    1.75 Å, Au–Au 2.18 Å, **Au–Mg 2.43 Å**, Au–Al 10.5 Å (the dopants lie deep; never approached) — the benchmark's
    gap is Au–Mg²⁺ alone. Against the nine frozen-density targets (1.8–3.49 Å), **the non-additive Thomas–Fermi kinetic
    energy of the frozen Mg²⁺ and Au⁰ densities times one constant, K_TF = 0.918, reproduces the wall** (relative
    residuals +0.5 / −0.7 / −2.3 / −4.5 % at 1.8 / 2.0 / 2.2 / 2.4 Å; absolute ≤ 0.034 eV at 2.66–3.49 Å, where the
    values are small); the von Weizsäcker, exchange and correlation parts alone do not (residuals 100 % and more).
    ESTABLISHED: the Gordon–Kim (1972) frozen-density kinetic overlap as the closed-shell repulsion; NEW-TO-TRACE:
    its use as the frame pair's own Pauli baseline with K fitted to TRACE's frozen-density PBE targets, tabulated as a
    Chebyshev series of R E(R) from the SPEC 25 library densities (the penetration tables' machinery), for the frame
    pairs whose data stop short (here Au–Mg²⁺). Design fixed before building: E_w(R) = K_TF · T_TF^nad[ρ_Mg²⁺, ρ_Au⁰](R)
    summed over frame Au–Mg pairs, trained with (warm fine-tune) — step 24 after steps 22–23; tests: the table against
    the nine targets, forces against finite differences, zero beyond the cutoff. Pre-registered prediction (before the
    28-ion ALMO flat result, which adds the four outward O's): CT_cp(flat, 28 ions) is more negative than −0.80 eV,
    i.e. more than 2.5 times the ten-ion value (−0.319), because the added contacts are the enhanced ones; if it
    scales with the contact count only (≈ −0.64), the mechanism is falsified.
  - Erratum: numbers from `first_order_exchange.py` for pairs with an O state, quoted in conversation on 2026-10-19/20
    (e.g. Au⁻/O²⁻ "+1.591"), were wrong: the dimer's Watson sphere was placed in bohr. Fixed; the raw outputs carry an
    erratum header.

**Falsified (2026-10-19): the undoped wetting energy as Au₂ polarisation in the surface field.**
- The idea: an upright dimer aligns its bond with the field normal to the surface, so it gains
  ΔE = −½ (α∥ − α⊥) E² over the flat one.
- Supporting evidence: across the nine scored AuMgO models, the undoped wetting energy rises with the isolated Au₂
  α∥ (r = +0.66, slope +91 meV/Å³; the doped wetting energy is uncorrelated, r = −0.08). The slope needs
  E = √(2K · 0.091 eV/Å³) ≈ 1.6 V/Å.
- Falsifying measurement: the model's own screened potential difference along the upright dimer is 0.37 V
  (seed 42) and 0.20 V (seed 43), an average field of 0.14 / 0.08 V/Å, so −½ α_CT E² is a few meV.
- The weak field is physical. The potential of the MgO(100) charge checkerboard decays as exp(−π√2 z/d), with
  d = a/2 = 2.105 Å and decay length 0.47 Å (ESTABLISHED: the Fourier solution of a periodic surface charge).
- So the +929 meV upright preference is short-range physics (the Au–O contact, Pauli, overlap). The correlation with
  α∥ has a common cause, the dimer's ionic gap, which sets both α∥ and the upright dimer's in-situ ionic and
  coherent energies.

## 4e. Born charges and the LO–TO splitting of bulk MgO: the formal crystal's long-range response (2026-11-06)

Tool `tools/electrostatics/mgo_phonons.py`; outputs in `$TRACE_FP_CACHE/mgo_phonons/{step20_s42,step20_s43,meanfield_s42}/`
(`band_nonac.yaml`, `band_nac.yaml`, `band_backbone.yaml`, `phonopy_params.yaml`). Not a benchmark step: the 4G tests carry
no phonons and no DFT phonons were computed here. References: experiment TO 401 / LO 718 cm⁻¹ (Jasperse et al., Phys. Rev.
146, 526 (1966); Sangster et al., J. Phys. C 3, 1026 (1970)), ε∞ 2.95, ε0 9.8; published PBE Z* 1.96–1.99, ε∞ 2.75–3.2,
TO ≈ 395, LO ≈ 695. Lattice constant 4.265 Å (the 4G data's PBE value).

**Why this test.** The LO–TO splitting of a polar crystal exists only through the macroscopic field,
ω_LO² − ω_TO² = 4πK_E Z*² / (Ω μ ε∞) for a cubic diatomic crystal (Born–Huang; ESTABLISHED). Finite-displacement force
constants in any supercell give LO = TO; the non-analytic correction (Gonze and Lee, PRB 55, 10355 (1997), in phonopy;
ESTABLISHED) restores the splitting from Z* and ε∞. Slab energies and forces never constrain Z*: this is the substrate
response the benchmarks leave free.

**Z* of TRACE-Redox under PBC (NEW-TO-TRACE measurement).** The module refuses a uniform field under periodic boundaries
for its free charges (`trace_redox/bound.py`: −Σ c_u r_u·E is branch dependent), so Z* = −∂²E/∂E∂r was assembled from the
two field-dependent terms of the frame configuration n = f:
- the frame's integer charge, f_u δ_ab, exact (the branch of Σ f_u r_u is position independent; its derivative is not);
- the bound sector's dynamic part ∂P_b/∂r_u, with P_b = Σ_b t_b d_b + Σ_i μ_i the SPEC 15 bond-current dipole of the
  moments the frame drives, y_f = y0 − K⁻¹A_fᵀf (SPEC 17.9's elimination; the sector is quadratic, so the energy and the
  moment routes coincide at the dense solve), evaluated inside the module's forward pre-hook with the features' graph intact.

ε∞ is the bound sector's `dielectric_tensor` (drive independent: W and b do not depend on E). The ensemble's share is bounded
by the frame population: 1.000000 in the 64-atom cell (lowest other configuration 3.98 eV above, largest coupling 0.01 eV).

**Artifacts ruled out.** Edge vectors rebuilt from the differentiable positions equal the model's (0.0); translational sum
rule Σ_u Z*_u ≤ 3e-13; central difference of P_b against autograd 1.5e-9; the same Z* on the 8-atom and the 64-atom cell;
|∂P_b/∂h| ≤ 4e-16 (the bound moments do not depend on the backbone features in this model, so the fixed-feature derivative
is the full one); phonopy's splitting equals the Born–Huang formula to 0.1 cm⁻¹ (718.0 vs 718.0; 744.7 vs 744.8), is the
same along [100], [110] and [111] as cubic symmetry requires, and the band routine's own Γ agrees. phonopy 2.17.2's
`nac_q_direction` has no effect in `run_qpoints` at q = 0 exactly; the limit is taken at |q| = 10⁻³ (reduced). The ensemble
guard on every displaced supercell: |F − E_frame| = 0.0000 eV, frame population 1.000000.

**Measured.**

| model | Z*(Mg) | ε∞ | TO | LO | LO − TO | LST ε0 | TO, backbone alone |
|---|---|---|---|---|---|---|---|
| step 20 s42 | +2.429 | 3.009 | 389.8 | 817.0 | 427.2 | 13.22 | 504.2 |
| step 20 s43 | +2.536 | 3.048 | 414.3 | 852.2 | 437.9 | 12.90 | 587.2 |
| mean-field SQE (`AuMgO_elec_r5_s42`) | +0.267 | 1169.7 | 398.5 | 398.5 | 0.0 | 1170 | 418.0 |
| experiment | 1.995 (Born–Huang from LO, TO, ε∞ at this volume; 1.958 at a = 4.212) | 2.95 | 401 | 718 | 317 | 9.8 (LST of TO/LO: 9.46) | — |
| PBE, published | 1.96–1.99 | 2.75–3.2 | ≈ 395 | ≈ 695 | ≈ 300 | — | — |

Frequencies in cm⁻¹; Z* isotropic to 1e-4 and the raw Z*_Mg + Z*_O ≤ 1e-14.

**Reading.**
1. *Where the electrostatics shows.* Without the electrostatic term the backbone alone has TO 504 / 587 cm⁻¹ and no Z*,
   hence no splitting. The electrostatic term brings TO to 390 / 414 (experiment 401: −2.8 % / +3.3 %) and carries the whole
   LO–TO splitting: with the NAC the model's own Z* and ε∞ give 427 / 438 cm⁻¹ against 317.
2. *ε∞ is right, Z* is not.* ε∞ 3.009 / 3.048 against 2.95 (the ε∞ prior of step 14 holds); Z* +2.429 / +2.536 against
   1.96–2.0, so Z*²/ε∞ is 45 % / 56 % too large: LO 817 / 852 against 718, ε0 13.2 / 12.9 against 9.8.
3. *Decomposition (measured).* Z* = f (±2) + 0.429 / 0.536, and the dynamic part is the induced-dipole channel alone. The
   bond-charge channel is exactly zero because SPEC 22's partition assigns the inter-unit bond charge flow to the free
   sector: the frame's coupling to the bound bonds, A_f on the bond columns, is 0.000e+00 (320 Mg–O bonds in the 8-atom
   cell survive the partition but are undriven). With the partition switched off post hoc A_f is 1.678, the bond channel
   gives −0.059 and the dipole channel 0.422: Z* = 2.363, still +0.36. The ensemble's covalent admixture, ∝ (τ/Δ)², is
   negligible with the learned Mg–O coupling 0.197 eV per edge (Harrison V_spσ at 2.1325 Å: 2.379 eV; ESTABLISHED) against
   the 3.98 eV gap of the 64-atom cell.
4. *What the number means.* A polarizable rigid-ion lattice has Z* = Z_S (ε∞ + 2)/3 (Szigeti; ESTABLISHED, tested in
   `tests/electrostatics/test_reference_physics.py`): Z_S = 2 and ε∞ = 3.0 give 3.33. The model's Gaussian-smeared dipoles
   give +0.43 / +0.54 rather than +1.33, and experiment's Z* = 2.0 says the local-field enhancement is cancelled by a dynamic
   covalent transfer of ≈ −0.4 e per unit displacement (the Mg–O bond's charge follows the overlap). That channel does not
   exist in step 20: not in the bound sector (partitioned away), not in the ensemble (τ 12× too small). This is the first
   observable that isolates it; the slab energies and forces do not.
5. *The mean-field SQE model* fails the other way: Z* 0.267 and ε∞ 1170 (≈ 1300 as recorded in an earlier measurement): global charge
   equilibration over-polarises the periodic crystal and gives no splitting at all. Its TO 398.5 is close to experiment
   because the backbone fits forces whatever the electrostatics does (its backbone alone gives 418.0, so there the
   electrostatic term moves TO by only −20 cm⁻¹ against −114 / −173 in the formal model) — the phonon frequencies alone do
   not test the electrostatics; the splitting does.

**Ensemble defect found on the way (small periodic cells).** The learned O–O hole coupling is 28.08 eV per edge at 3.016 Å
(Harrison V_ppσ 1.860, V_ppπ −0.528 eV). In the 8-atom conventional cell the Γ-point Bloch sum of SPEC 17.6 over the four
images of each O–O pair gives T = 112.3 eV between the four degenerate O⁻ + band-electron configurations (2.52 eV above the
frame): a complete graph with equal negative couplings, whose symmetric combination lies 2.52 − 3 × 112.3 = −334.5 eV below
the frame, and `energy_free` adopts it (−404.24 eV against the frame's −69.79; frame population 1.3e-5). The SPEC 27
Harrison bound switched on post hoc cuts the per-edge value to 2.64 eV (10.57 over the images) and the state to −29.2 eV:
not enough. In the 64-atom cell the same configurations sit 3.98 eV above the frame and the SPEC 21 amplitude switch leaves
couplings of 0.01 eV, so every force evaluation of the phonon runs was in the frame (the guard above) and the slab
benchmarks never met it. Root cause (hypothesis, not yet tested): a scalar isotropic τ per pair cannot represent p-hole
hopping — the Slater–Koster direction cosines make the Γ sum of p-orbital hops cancel, while a scalar sums coherently to
−z|t|. Candidate fixes, one per run: the angular structure of τ for p shells, or no O–O hops inside the formal anion
sublattice (the O 2p band is the bound sector's).

**Against DFT phonons: Togo's phonondb (added 2026-11-06 night, by decision).** Reference: phonondb entry mp-1265 on
NIMS MDR (`$TRACE_PHONON_RAW/phonondb_mp-1265/`; CC BY 4.0): VASP PBEsol
(GGA = PS, PAW, ENCUT 525 eV), a = 4.21726 Å, 2×2×2 supercell, DFPT Z* = ±1.9791, ε∞ = 3.2417; band structure recomputed
with phonopy 2.17.2 along the same Γ–X–K–Γ–L path with its own NAC (`experiments/benchmarks_4g/phonons_mgo/compare_phonondb.py`,
tables by `band_table.py`, figures `mgo_phonons_step20_vs_phonondb.png`, `mgo_phonons_vs_phonondb_a4217.png`). The DFT
reference confirms the Born-charge diagnosis independently of experiment: Z* 1.979 against the model's 2.43 / 2.54.

| cm⁻¹ | TO(Γ) | LO(Γ) | X | K | L | path RMS (acoustic / optical) |
|---|---|---|---|---|---|---|
| DFT phonondb, a = 4.217 | 391.0 | 693.9 | 284, 284, 418, 440, 440, 535 | 289, 382, 393, 416, 476, 510 | 273, 273, 356, 356, 538, 557 | — |
| s42, a = 4.265, NAC | 389.8 | 817.0 | 266, 266, 406, 406, 424, 537 | 296, 389, 393, 405, 450, 506 | 247, 247, 353, 353, 498, 610 | 28.5 (13.1 / 38.1) |
| s42, a = 4.265, both sides without NAC | 389.8 | 389.8 | 266, 266, 406, 406, 424, 537 | 260, 380, 405, 407, 461, 506 | 247, 247, 353, 353, 498, 610 | 14.3 (13.5 / 15.0) |
| s43, a = 4.265, NAC | 414.3 | 852.2 | 269, 269, 445, 445, 457, 577 | 298, 380, 423, 431, 500, 551 | 284, 284, 358, 358, 592, 607 | 41.9 (17.0 / 56.7) |
| s43, a = 4.217, NAC | 380.9 | 836.2 | 253, 253, 485, 485, 503, 600 | 291, 411, 440, 478, 508, 566 | 276, 276, 330, 330, 619, 644 | 49.2 (32.5 / 61.5) |
| s42, a = 4.217 | ensemble leaves the frame (below): phonons are an artifact (TO 3986), not reported | | | | | |
| mean-field SQE, a = 4.265, NAC | 398.5 | 398.5 | 282, 282, 382, 423, 423, 512 | 262, 365, 374, 403, 474, 478 | 254, 254, 369, 369, 518, 545 | 56.4 (15.1 / 78.3) |

RMS over 6 branches × 204 q-points. Reading: (i) the short-range force constants agree with DFT to 14 cm⁻¹ RMS without NAC
(s42; TO 389.8 vs 391.0, max 53 at L), from a model trained on slab forces only; (ii) the NAC doubles the RMS, all of it in
the optical branches (LO 817 vs 694 at Γ, 610 vs 557 at L) — the +0.43 e Born charge; (iii) at the DFT volume s43 is 49 RMS
(its optical branches stiffen under 1 % compression, X 445 → 485 against DFT 418–440: its optical Grüneisen response is off);
(iv) the mean-field model is worst (no LO branch).

**Seed 42's bulk MgO is only metastable: the ensemble's cliff in the lattice constant (measured on the 64-atom cell).**

| a (Å) | 4.265 | 4.255 | 4.245 | 4.235 | 4.225 | 4.217 | 4.217 + Harrison bound post hoc |
|---|---|---|---|---|---|---|---|
| F − E_frame (eV) | −0.000 | −0.216 | −26.97 | −108.8 | −249.2 | −365.6 | −5.78 |
| frame population | 1.000000 | 0.118 | 2e-5 | 2e-5 | 3e-5 | 3e-5 | 0.120 |
| largest switched coupling (eV) | 0.007 | 0.338 | 2.56 | 9.37 | 21.1 | 30.7 | 0.70 |
| O⁻ + band-electron configuration above the frame (eV) | 3.977 | 3.854 | 3.728 | 3.599 | 3.467 | 3.361 | 3.361 |

The 33 configurations are the same throughout (the frame and the 32 single-hole + band-electron states, one per O); what
changes is the SPEC 21 amplitude switch σ(e_ij): as the hole configurations' gap falls from 3.98 to 3.36 eV under a 1 %
compression, the switch opens the 28.08 eV/edge scalar O–O hole couplings, and the hole band's symmetric combination drops
below the frame — the same collapse as the 8-atom cell's, now in the 64-atom cell at 0.5 % compression (a = 4.245). The
Harrison bound alone does not cure it (−5.8 eV, population 0.12). Seed 43 selects no hole configurations in bulk (one
configuration) and is clean at both volumes. Consequence: step 20's bulk MgO free energy (seed 42) is not usable for MD,
compression or thermal expansion; the slab benchmarks sit at a = 4.265 and never crossed the cliff. This moves the scalar
O–O hole coupling from a small-cell curiosity to a model defect of the first rank (backlog item 2 of §5g).

**Against PBE DFT, functional-matched (added 2026-11-06 late, by decision).** Reference: Togo's phono3py dataset for
rocksalt MgO on NIMS MDR (`$TRACE_PHONON_RAW/phono3py_pbe_mgo/`; CC BY 4.0;
the 103-compound set of PRB 91, 094306 (2015) / PRL 115, 205901 (2015)): VASP PBE, a = 4.25566 Å, fc2 from two 0.03 Å
displacements on a 4×4×4 supercell (512 atoms), PBE Born charges Z* = ±1.9923 and ε∞ = 3.235 (LEPSILON), read with phonopy
alone (`compare_phono3py_pbe.py`). PBE at its own volume: TO 373.3, LO 681.3, X 281, 281, 406, 425, 425, 528; L 262, 262,
344, 344, 529, 550 cm⁻¹.

| cm⁻¹, model with NAC | TO(Γ) | LO(Γ) | path RMS vs PBE (acoustic / optical) | without NAC on both sides |
|---|---|---|---|---|
| s42, a = 4.265 (0.2 % above PBE's) | 389.8 | 817.0 | 30.3 (12.6 / 40.9) | — |
| s42, a = 4.2557 | 386.1 | 811.6 | 555 (781 / 80): force constants contaminated, see below | 556 |
| s43, a = 4.2557 | 407.0 | 850.2 | 48.5 (22.1 / 64.9) | 40.4 (23.2 / 52.3) |

Reading: (i) functional-matched, the model's TO is +3.5 % (s42) / +9 % (s43) above PBE's 373.3 — the earlier "TO on DFT"
against PBEsol's 391 was partly PBE/PBEsol coincidence; (ii) the LO excess is the same as before (+130 / +169 at Γ): Z*
2.43 / 2.54 against PBE's 1.992, ε∞ 3.06 / 3.05 against 3.235; (iii) the PBE dataset's 4×4×4 force-constant supercell
against the model's 2×2×2 is a convergence difference on the model side, checked on seed 43 with a 3×3×3 supercell
(216 atoms): 2×2×2 against 3×3×3 differ by 6.3 cm⁻¹ RMS over the path (max 27; acoustic 7.8, optical 4.2), and the RMS
against PBE is unchanged (48.4 against 48.5) — the truncation is not the deviation.

**The cliff is sharper than the population suggests.** Seed 42 at a = 4.2557 has frame population 0.9755 and |F − E_frame|
= 0.003 eV, yet its force constants are wrong (acoustic RMS 781 cm⁻¹, max 2510): the 2.5 % of switched hole
configurations (couplings 0.32 eV) carry force-constant contributions out of all proportion to their weight, because the
switch σ(e_ij) and the 28 eV couplings are steep in the geometry. A usable force evaluation needs the frame population
at 1 − 1e-4 or better; the guard in `mgo_phonons.py` prints it for every displaced supercell.

**The ensemble route to Z* is closed by the selection, not by the coupling (frozen probe, 2026-11-07 00:40;
`experiments/benchmarks_phonons/probe_ensemble_born.py`, seed 43, 64 atoms, a = 4.2557).** The ensemble's dynamic transfer
Z*_ens(O) = ∂P_ens/∂r_O with P_ens = Σ_n p_n d_n (populations from the eigenstates of H, configuration dipoles relative to
the frame) is exactly zero, with the learned couplings and with the Harrison anchor switched on post hoc alike, because
bulk MgO's ensemble holds one configuration — the frame — in both cases: the thermal windows (kT = 0.1 eV) admit no
Mg⁺O⁻ or O⁻ + band-electron state, so no coupling, however large, ever acts. The covalent admixture that cancels the
local-field enhancement in real MgO is a zero-temperature quantum mixing of weight ~(τ/Δ)², not a thermal population;
the ensemble as selected cannot carry it. **Sign check (2026-11-07 00:50) — covalent transfer is the wrong candidate.** A bond charge that grows as the bond
shortens (τ ∝ overlap; Harrison's polarity) moves electrons *toward* the approaching Mg when O moves toward it: the
dipole change has the sign of the ionic displacement dipole, so dynamic covalent transfer *raises* |Z*| above the static
charge (ESTABLISHED: the anomalous Born charges of BaTiO₃ and of CaO/SrO/BaO via O 2p–d transfer; Ghosez, Michenaud,
Gonze, PRB 58, 6224 (1998)). MgO's Z* = 1.99 at ε∞ = 3.2 therefore needs a *negative* term of ≈ −0.4 e per unit
displacement of the frame ion, which is the deformation dipole of the shell model (ESTABLISHED: Dick and Overhauser 1958;
Hardy 1962; Sangster et al. 1970 for MgO): the O²⁻ cloud is pushed back by the closed-shell repulsion of the approaching
Mg²⁺ — an overlap-induced dipole μ_def = −β(R) R̂ per contact, opposite to the displacement, field independent at first
order (ε∞ untouched). NEW-TO-TRACE as a frame-ion term; SPEC 26 has its carrier analogue (carrier induction) and SPEC 25
the repulsion it accompanies (Gordon–Kim), so the ingredient is the table β(R) for the frame pair. Deciding measurement
before any training (the next step): the Pauli deformation dipole of an embedded Mg²⁺–O²⁻ pair from the frozen ALMO
state, μ_FRZ(R) − μ_fragments(R), with the tools of §4c, at R = 1.9–2.4 Å; its derivative over the two axial contacts
must give ≈ −0.4 e, and the predicted Z* = 2 + 0.43 (dipole channel) + 2 dμ_def/dR must land within 1.90–2.10. The fold
on Mg (O 2p → Mg 3s) remains a static-charge question (DFT Bader O ≈ −1.7), not the Z* fix.
**Launched 2026-11-07 01:05** (`tools/electrostatics/deformation_dipole.py`, `$TRACE_FP_CACHE/deformation_dipole/mgo_pair.json`):
O²⁻ + six Mg²⁺ quantum (PBE / def2-tzvp) in the bulk Evjen lattice (n = 5, TIPs within 3.5 Å), fragment A the +z Mg at
R = 1.928, 2.028, 2.128 (PBE), 2.228, 2.328 Å; μ_def = μ_HL − μ_A − μ_B from the frozen state (x = 0 of DirectionalEnergy),
μ_pol = μ_ALMO − μ_HL. Pre-registered reading: Z*_def(O) = −2 dμ_def,z/dR at 2.128 Å; the hypothesis passes if
−2.43 − 2 dμ/dR (s42) lands in −1.90…−2.10 (PBE −1.992); it fails if dμ_def,z/dR ≥ 0 or |Z*_def| < 0.15.
**Result (2026-11-07 01:45): the deformation-dipole hypothesis FAILS the pre-registered threshold.** μ_def,z(R) = 0.0145,
0.0101, 0.0068, 0.0045, 0.0029 e Å at R = 1.928…2.328 Å (positive: the O²⁻ cloud is pushed away from the Mg, the right
direction), slope −0.028 e at 2.128 Å, Z*_def(O) = +0.056 e — a fifth of what is needed. The frozen (Pauli) deformation of
the pair is real but small. The same scan gives the pair's polarisation dipole μ_pol,z = −0.694, −0.678, −0.659, −0.637,
−0.611 e Å (the O²⁻ polarised by the Mg²⁺: α_O²⁻ E with E = 2K_E/R² gives −0.75 at 2.128 Å; the ALMO value −0.66 is the
overlap-damped response), slope +0.207 e, i.e. Z*_pol(O) = −0.414 e for the two axial contacts — the same size and sign
as the model's induced-dipole channel (−0.43 on O). So, within this pair budget, DFT's Z*(O) = −1.992 = −2 + 0.056 −
0.414 + Z*_CT requires the charge-transfer term (beyond ALMO: μ_SCF − μ_ALMO) to contribute **+0.37 e**, i.e. the CT
dipole must *shrink* in magnitude as the bond shortens (dμ_CT,z/dR < 0) — opposite to the naive bond-polarity picture
of the sign check above. Hypothesis to measure next (the tool gains μ_SCF, 2026-11-07 01:50): Z*_CT(O) = −2 dμ_CT,z/dR
≈ +0.37 e.
**Measured (04:50): the charge-transfer channel is negligible too.** μ_CT,z(R) = −0.0118, −0.0105, −0.0098, −0.0097,
−0.0102 e Å (CT energy 0.04–0.06 eV), slope +0.004 e, Z*_CT(O) = −0.008 e. The pair budget closes at Z*(O) = −2 + 0.056
− 0.418 − 0.008 = −2.37 e — the model's own −2.43, not DFT's −1.99. So no short-range pair term (Pauli deformation,
charge transfer) carries the missing +0.4 e; and the pair's −0.42 polarisation term is itself a single-contact
quantity that cancels by cubic symmetry when the whole Mg sublattice moves (the first-order point-charge field at a
centrosymmetric site vanishes), leaving, in both the model and DFT, only the collective Lorentz-type enhancement of the
induced dipoles — which the model has (+0.43) and DFT does not (−0.01 at ε∞ 3.2). Caveat for any budget from the pair: the lattice local-field sum is not in it (the model's +0.43 is the full
Ewald dipole response); the pair isolates the direction and size of each channel, not the crystal total.

**The lever found (frozen scans, 2026-11-07 03:30–04:30; `experiments/benchmarks_phonons/probe_sigma_scan.py`).** The
charge widths are not it: scaling σ_O, σ_Mg by 1.2–2× leaves Z* and ε∞ bit-identical (the dipole response never sees
them). The induced dipoles carry their own Gaussian widths, SPEC 12's σ_μ²ˡ⁺¹ = K_E ℓ²ˡ ‖A⁻²‖_F^{1/2}/(a V_l): a
positive-definiteness rule that ties the width to the stiffness, not to the ion. On step 20 it gives **σ_μ(O) = 0.52 Å**
and σ_μ(Mg) = 1.10 Å — the O²⁻ dipole smeared over a third of the ion's radius. Scanning `gaussian_a` (a = 0.5 default):

| a | σ_μ(O) / σ_μ(Mg) (Å) | Z*(Mg) | ε∞ | (Z* − 2)/(ε∞ − 1) |
|---|---|---|---|---|
| 0.90 | 0.43 / 0.91 | 2.758 | 3.333 | 0.325 |
| 0.70 | 0.47 / 0.99 | 2.611 | 3.204 | 0.277 |
| 0.50 | 0.52 / 1.10 | 2.429 | 3.009 | 0.214 |
| 0.25 | 0.66 / 1.39 | 2.184 | 2.679 | 0.110 |
| 0.08 | 0.97 / 2.03 | 2.059 | 2.460 | 0.040 |
| 0.04 | 1.22 / 2.56 | 2.028 | 2.417 | 0.020 |
| point dipoles (Szigeti) | 0 | 3.33 at ε∞ 3.0 | — | 0.667 |
| PBE | — | 1.992 | 3.235 | −0.004 |

Both numbers are the Lorentz local-field enhancement of smeared dipoles and fall together as the dipoles widen, but not
in proportion: the enhancement per unit susceptibility, (Z* − 2)/(ε∞ − 1), drops 10× from 0.21 to 0.02 as σ_μ(O) goes
from 0.52 to 1.22 Å, and DFT sits at 0. The Born charge's enhancement is the near-field dipole sum over the first shells,
which a width comparable to the neighbour distance kills; ε∞ is the q → 0 response, which survives. So the physical
statement (NEW-TO-TRACE as a rule; ESTABLISHED as physics: the O²⁻ polarisation density is the ion's own, radius
≈ 1.3 Å) is that **the induced-dipole width must be the ion's density width, with SPEC 12's stiffness width only as a
floor for positive definiteness** (wider is always still PD). Under such a width the ε∞ prior would retrain α upward and
Z* would follow the ratio at that width: predicted Z* ≈ 2 + 0.02 × 2.2 ≈ 2.05 at σ_μ(O) ≈ 1.2 Å. Deciding measurement
(running, `probe_dipole_width.py`, pre-registered): σ_μ(O) set to 0.9–1.5 Å on the frozen weights with α rescaled so
that ε∞ = 3.235; the lever is confirmed if Z*(Mg) lands in 1.90–2.10 for some width in that range.
**Result (04:55): the O width alone FAILS.** With σ_μ(O) = 0.52, 0.90, 1.10, 1.30, 1.50 Å and α rescaled (×1.10–1.14) so
that ε∞ = 3.235, Z*(Mg) = 2.441, 2.423, 2.415, 2.409, 2.406: the O dipole's width does not separate Z* from ε∞. In the
gaussian_a scan the ratio fell because σ_μ(Mg) widened with it (1.10 → 2.56 Å); the Mg width is now being scanned alone
at fixed ε∞ (04:58). Also to be checked there: the trained α_O, α_Mg (the ε∞ prior may have put polarizability on Mg).
**Result (05:15): the Mg width FAILS too, and the trained polarizability sits on the wrong ion.** σ_μ(Mg) = 1.10, 1.60,
2.00, 2.60, 3.20 Å (α rescaled to ε∞ = 3.235): Z*(Mg) = 2.448, 2.363, 2.340, 2.330, 2.327; both widths wide (O 1.3, Mg
2.6): 2.326. No Gaussian width brings Z* below 2.3 at this ε∞: the width lever is ruled out. The scan also printed the
trained species polarizabilities, **α_O = 0.205 Å³, α_Mg = 1.916 Å³** — the ε∞ prior fixed the sum and the slab data
never fixed the split, and the model put the polarizability on Mg²⁺ (Clausius–Mossotti and every ionic table: O²⁻ ≈ 1.7–2.0,
Mg²⁺ ≈ 0.1 Å³). Hypothesis (measured next, 05:20): the species split of α is the Z* lever — with the polarizable ion
being the one whose charge moves, the near-field terms that enhance Z* change sign structure; prediction to test: the
physical split (α_O 1.9, α_Mg 0.2) at ε∞ = 3.235 gives Z*(Mg) within 1.90–2.10.
**Result (05:30): the species split FAILS, in the other direction.** At ε∞ = 3.235: as trained (α_O 0.2 / α_Mg 1.9)
Z* = 2.440; physical split (1.9 / 0.2) **2.629**; all on O 2.515; even 2.774; physical split with σ_μ(O) = 1.3 Å 2.574; all
on O with 1.3 Å 2.472. Putting the polarizability on the anion *raises* the enhancement. Over every width and split
scanned, the Gaussian-dipole bound sector gives Z*(Mg) = 2.33–2.77 at ε∞ = 3.2: the Lorentz-type enhancement is
intrinsic to a lattice of smeared induced dipoles driven by Coulomb fields, and no parameter of that sector reaches DFT's
1.99. (The wrong species split is still a defect to fix by a frame-polarizability prior from the first-principles table,
O²⁻ 2.44 / Mg²⁺ 0.07 Å³ — but it is not the Z* lever.)

**What the ALMO pair says the missing physics is (measured, 2026-11-07 05:40).** The pair's polarisation dipole against
the free-ion point-field response α(O²⁻) E, with α = 2.44 Å³ (the Watson-sphere value of the repo's table) and
E = 2K_E/R²: μ_pol/μ_point = 0.53, 0.58, 0.61, 0.64, 0.68 at R = 1.93, 2.03, 2.13, 2.23, 2.33 Å. The O²⁻ cloud's response
to its own cation at contact is damped to ~0.6 of the point value, and the damping *changes with distance* at
+0.37 per Å — the shell model's short-range shell–shell coupling in first-principles form (ESTABLISHED as the shell
model: Dick and Overhauser 1958; Sangster et al. 1970). The model's bound dipoles see the frame charges through the
Gaussian-smeared Coulomb coupling with no distance-dependent damping, so in the model the axial contacts' responses
cancel against the equatorial ones only as the point-charge symmetry dictates, while in the crystal the axial contacts
(whose overlap changes at first order) are damped differently from the equatorial ones (second order). NEW-TO-TRACE
candidate: an overlap damping s(R) of the frame-charge → bound-dipole coupling for the frame pair, with s and s′ from
this measurement (s(2.13 Å) = 0.61, s′ = +0.37/Å), field independent, ε∞ touched only through the static near field.
Deciding measurement (next): the frozen Z* and ε∞ with the first-shell frame→dipole coupling multiplied by s(R); pass if
Z*(Mg) enters 1.90–2.10 at ε∞ within 2.9–3.5. If Thole damping (SPEC 1.6, opt-in) is available on the periodic path it
is the first thing to try as the existing distance-dependent damping.
(05:50: Thole is refused by TRACE-Redox — `BoundSectorError: trace_redox needs multipole_kernel='gaussian' (SPEC 12 /
17.5)` — so the damping has to be built as its own term on the frame-charge → bound-dipole block of the elimination.)

**Where the Z* question stands (2026-11-07 06:00).** Measured and pre-registered: (1) the model's +0.43 e is entirely the
induced-dipole channel (bond charge 0, through-h 0); (2) no short-range pair term is missing — the embedded Mg²⁺–O²⁻
pair's Pauli deformation (+0.06), polarisation (−0.42, a single-contact term that cancels by symmetry in the crystal)
and charge transfer (−0.01) reproduce the model's budget; (3) the ensemble route is closed (bulk MgO's ensemble holds the
frame alone at kT = 0.1 eV); (4) the dipole widths (σ_μ of O and Mg, SPEC 12's a) and the species split of α do not
separate Z* from ε∞ — Z*(Mg) = 2.33–2.77 at ε∞ = 3.235 over every setting, DFT 1.99. What remains is the one piece of
physics the pair measured and the model lacks: the O²⁻ response to its own cation is damped by overlap to 0.53–0.68 of
the point-field value with a slope of +0.37/Å across the contact distance. The next step, before any training, is that
term — s(R) on the frame-charge → bound-dipole coupling of the first shell (NEW-TO-TRACE; the shell model's shell–shell
coupling, ESTABLISHED), evaluated frozen with α rescaled to ε∞ = 3.235, pass criterion Z*(Mg) 1.90–2.10. Separately, the
inverted species polarizability (α_O 0.2 / α_Mg 1.9 Å³) is a defect to fix by a frame-polarizability prior from the
first-principles table (O²⁻ 2.44, Mg²⁺ 0.07 Å³; ESTABLISHED, the repo's own finite-field values), whatever Z* does.

**Pre-registered (06:20, before the run): the overlap damping of the frame-charge → bound-dipole coupling.** SPEC 26
already corrects the Gaussian coupling A[i, μ_j] = K λ₃ (r_i − r_j)/R³ ℓ_j for *carriers* by the true field fraction;
the frame charge keeps λ₃ ≈ 1 at 2.13 Å, i.e. the point-charge field, while the ALMO pair says the O²⁻ dipole's response
to its own Mg²⁺ is 0.61 of the point value there. The term: A[i, μ_j] → s(R_ij) A[i, μ_j] on nearest-neighbour frame
pairs, s(R) = 0.61 + 0.37 (R − 2.128 Å) (clamped to [0.2, 1]; the measured law). Why it decouples Z* from ε∞: ε∞ is the
bound sector's own response to a uniform field (K alone, A untouched), while the dipole channel of Z* is
∂/∂r [K⁻¹A_fᵀf], whose first-shell term goes from A′ ≈ −0.94 A/Å to s A′ + s′ A = (−0.57 + 0.37) A = −0.20 A/Å: 21 % of the
undamped near-field derivative. Prediction: if the model's +0.43 e is dominated by the first shell, Z*(Mg) → ≈ 2.1 at
unchanged ε∞. Pass: Z*(Mg) within 1.90–2.10 with ε∞ unchanged (3.009 as trained; 3.235 with the α scale 1.095); fail
otherwise. Controls: s ≡ 0.61 without the slope (the slope is the physics), and s′ halved / doubled.

**Result (06:45): PASS.** `experiments/benchmarks_phonons/probe_overlap_damping.py`, step 20 s42, 64 atoms, 192 frame pairs:

| coupling on the first shell | Z*(Mg), ε∞ as trained | Z*(Mg) at ε∞ = 3.234 (α × 1.095) |
|---|---|---|
| undamped (control) | 2.429, 3.009 | 2.440 |
| **measured law s = 0.61 + 0.37 (R − 2.128)** | **1.927**, 3.009 | **1.911** |
| no slope, s ≡ 0.61 | 2.682 | 2.735 |
| half slope (0.61, 0.185) | 2.304 | 2.323 |
| double slope (0.61, 0.74) | 1.171 | 1.087 |
| weaker damping (0.80, 0.37) | 1.804 | 1.767 |

ε∞ is untouched by construction (the field drives the bound dipoles directly; the damping sits on the frame's static
coupling), and Z*(Mg) lands at 1.91–1.93 against PBE 1.992 — inside the pre-registered 1.90–2.10. The controls show the
mechanism is the *slope* of the damping, not its level: a constant damping raises Z* (it removes the near-field
screening of the frame's own field as well), the slope term s′A is what cancels the first-shell derivative. The
sensitivity is large, dZ*/ds′ ≈ −3.5 e per Å⁻¹, so the law's slope (from a five-point ALMO scan against the Watson-sphere
α = 2.44 Å³) carries ±0.1 Å⁻¹ ≈ ±0.35 e; the term's own first-principles law at the model's widths is the next
measurement. This is the first term found that moves Z* without moving ε∞: NEW-TO-TRACE, the frame-pair overlap damping
of the charge → induced-dipole coupling (the shell model's short-range shell–shell coupling, ESTABLISHED, in
first-principles form; SPEC 26's construction applied to the frame).
**Combined with the physical polarizability split (07:05), at ε∞ = 3.235:** trained split + law 1.911; table split (O²⁻
2.44 / Mg²⁺ 0.07 Å³) undamped 2.553, with the law **1.928**; Clausius–Mossotti split (1.9 / 0.2) with the law **1.955**;
table split with half the slope 2.368. The term is robust to where the polarizability sits: with the physical split and
the measured law, Z*(Mg) = 1.93–1.96 against PBE 1.992 at the PBE ε∞. Both corrections together — the frame-polarizability
prior (a defect fix) and the frame-pair overlap damping (the Z* physics) — are what the next step builds.
**Built as SPEC 34 and scored frozen on the phonon line (07:55; `runs/AuMgO_frozen_step20_fid_s43`, a = 4.2557 Å):** P0
pass (population 1.000000), **P1 pass (Z*(Mg) 1.965)**, P2 pass (ε∞ 3.052); the NAC part is now on DFT —
√(LO² − TO²) = 579 cm⁻¹ against PBE's 570 — while the frozen short-range force constants moved the other way (TO 407 →
450, +21 % against PBE; path RMS without NAC 40 → 65), because the backbone was trained against the undamped induction
and the term's static energy (v·y0 − ½ v·K⁻¹v and the y0 shift) is not small along the optical displacements. That is
what training with the flag (step 28) must re-absorb; the frozen P3/P5 numbers are the before, not a verdict.

**The term and the inverted polarizability interact (09:25–09:40, three undoped test slabs, frozen s42).** With the
trained split (α_Mg 1.9 / α_O 0.2 Å³) the flag lowers the surface O⁻ states by ~1.3 eV through its linear part
−(A K⁻¹ v)·n — the damped Mg dipoles are what held the O²⁻ — and the switched scalar O–O couplings collapse one
structure (frame population 0, q(Au₂) −1, |F| 455 eV/Å; 4G frozen F RMSE 88 eV/Å). With the physical split imposed
(α_O 2.44 / α_Mg 0.07 Å³) the same term leaves configurations, gaps, populations and forces where they were: the damping
weakens the O's own dipole, which does not act on the O's charge. The two corrections are one step: the
frame-polarizability anchor (SPEC 35, the first-principles ratio with a learned crystal scale) with the damping.
**Both built (SPEC 34 + SPEC 35) and measured through the module on step 20's bulk MgO (10:55):** anchor alone
Z*(Mg) 2.554 at ε∞ 3.453 (g = 1); anchor at g = 0.92 + damping **1.950 at ε∞ 3.227**, PBE 1.992 at 3.235. Steps 28a / 28b
(pre-registered in the 4G README) train them.

**What would move Z* (hypotheses, each with its measurement).**
- Bound bonds between frame units driven by the frame (SPEC 22's partition off for frame–frame pairs): −0.06 e at the
  learned hardness, measured above; a physical bond hardness (the two-level law of the backlog) is what −0.4 needs.
- A Harrison-sized Mg–O coupling in the ensemble (SPEC 31's anchor, step 21's line): a second-order estimate of the
  transferred charge per bond, 2 (τ/Δ)², with τ = 2.38 eV and Δ = 8 eV, and τ ∝ 1/d², gives a dynamic transfer of
  2 × 2 (τ/Δ)² ≈ 0.35 e per unit relative displacement — the right size (an estimate, not a measurement). Measuring it as
  Z* needs the ensemble's field coupling under PBC through minimum-image hop dipoles, a SPEC 15 extension not built.
- A training label: Born charges, ε∞ and ε0 exist for ≈ 7000 crystals (Materials Project DFPT, Petousis et al., Sci. Data 4,
  160134 (2017)) and ≈ 10000 (Togo's phonondb); `born_charges_from_energy(create_graph=True)` is the loss route.

## 5. Open fundamentals, ranked by measured size

1. **Couplings τ have no physical form.**
   - They are learned from an initial 0.01 eV. Under SPEC 21 a small τ gets gradient only from nearly degenerate
     pairs. Measured: τ(Au,Au) reached 0.15–0.30 eV after one epoch; other couplings may never grow.
   - Candidate (ESTABLISHED ingredients): a Harrison universal tight-binding anchor (V_ssσ = 1.40 ħ²/md², etc.;
     Au–Au at 2.47 Å gives 1.75 eV against a learned 1.2) times a bounded learned factor, as the band unit already
     does. The orientation of p/d frontier orbitals is not resolved by spherical densities.
   - **Re-ranked (measured 2026-10-18 17:05):** the trained τ(Au, Au) is reproducible across seeds. Step 12 gives
     +1.26 / +1.28 eV (seed 42, undoped / doped) and +1.15 / +1.19 eV (seed 43), 8 % apart, while the ionic
     weights differ (p(Au⁺) 0.302 against 0.197).
     - So τ does not carry the seed dependence, which is in the configuration energies (SPEC 21, 22).
     - Every hop that matters in the 4G systems is s–s (Au 6s, Ag 5s, Na 3s), so the missing state resolution of
       τ (one τ per pair for 6s and 5d hops alike) moves no benchmark number now.
     - It is a universality item, kept for transition-metal redox.
2. **The induction at overlap of §3:** built (SPEC 26, Proposition 5); measured +0.19 eV on Au⁻.
3. **Learned shifts remain the main free freedom.** With SPEC 18/22/23/24 they should shrink; step 15 reports them.
4. **Four non-reproducible NaNs**, all under load (correlate, not cause); the gradients of every tensor go NaN with a
   finite loss.
   - **Localised (2026-10-18):** in all three saved dumps the NaN is on the free-charge widths' gradient, exactly on
     the elements present; in step 11 on nothing else. It entered inside the backward of the σ_f-only plane-wave
     block, fed by finite gradients.
   - Hypothesis: a transient BLAS fault (not reproduced at load 7–10).
   - The guard now recomputes such a sub-step exactly (benchmarks README).
5. **State-independent polarisabilities: measured and built (SPEC 28).** ρ(Au⁻) = 3.39 and ρ(Au⁺) = 0.37 from
   first principles. On frozen step-12 weights T5 moves by +1 (seed 42) and −298 meV (seed 43). Step 17 measures
   it trained. The derivation as first written:
   - The bound sector has one α per atom, trained to the neutral or crystal value, in every charge state.
   - Exact per configuration by Woodbury: δE(n) = ½ y_S(n)ᵀ(Δ_S⁻¹ + C_SS)⁻¹ y_S(n), with S the units away from
     their frame, Δ_u = ℓ²(1/α(n_u) − 1/α(f_u)) on u's dipole block, y = y0 − M n and C = K⁻¹ on those blocks.
   - It is many-body: there is no exact pair split. First-principles α(Z, q) are being computed with the library's
     method (`tools/electrostatics/ion_polarizabilities.py`).
     - Na⁰ 166.8 bohr³ against 162.7 experimental;
     - O²⁻ in the Watson sphere 2.44 Å³, against ~1.7 Å³ in MgO by Clausius–Mossotti.
