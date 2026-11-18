# One charge model for all chemistry: carriers and fluctuations (2026-10-16)

Written after step 9 (exclusion; T5 95 % of DFT on Au₂/MgO:Al), step 10 (relaxed exclusion, negative) and the carbon chain's refusal. Every
claim is derived here, measured (named), or labelled a hypothesis. Nothing in §3–§6 is implemented yet.

## 0. What the measurements say

| system | TRACE mean-field (SQE bond variables + dipoles) | TRACE integer configurations (formal stack, steps 2–9) |
|---|---|---|
| carbon chain C₁₀H₂ / C₁₀H₃⁺ | best published forces: 0.0356 eV/Å (NequIP 0.0495, 4G 0.0757) | **refused**: > 512 configurations within 2.5 eV (also with frame widths) |
| Ag₃±, NaCl | best published forces (0.0159, 0.0068) | training (2026-10-16) |
| Au₂/MgO:Al transfer | wrong: the bound sector conducts across the slab (round 11) | step 9: T5 −881 meV (DFT −932, 4G −1016) |

- Neither treatment is universal.
- The integer stack fails exactly where the mean field succeeds (covalent, delocalised charge), and the
  reverse holds for long-range integer transfer.
- The question is not which treatment to choose but which charge fluctuations belong in which sector,
  decided by physics rather than by the modeller's choice.

## 1. The exact reference (ESTABLISHED)

In a minimal basis of one valence level per site, the electronic Hamiltonian is the Pariser–Parr–Pople /
Hubbard form (Pariser & Parr, J. Chem. Phys. 21, 466 (1953); Pople, Trans. Faraday Soc. 49, 1375 (1953);
Hubbard, Proc. R. Soc. A 276, 238 (1963)):

    H = Σ_u ε_u n_u − Σ_{u≠v,σ} t_uv c†_uσ c_vσ + Σ_u U_u n_u↑ n_u↓ + ½ Σ_{u≠v} γ_uv (n_u − Z_u)(n_v − Z_v)

TRACE's variables are this model's ingredients:
- ε_u and U_u come from the exact ionisation ladder (the formal anchors E_Z(n));
- γ_uv is the screened Coulomb W;
- t_uv is the coupling τ (configuration sector) and the hopping h (two-level bond hardness, SPEC 26.2).

The exact solution has two limits, which are TRACE's two sectors:
- **t ≪ Δ (charge-transfer gap):** integer charges, localised; the configuration ensemble is exact.
- **t ≳ Δ:** strongly mixed, covalent. The charge response is smooth, and the bound sector's quadratic
  functional is its controlled second-order representation.

## 2. Three exact results (derived here)

**Theorem A (the two-state charge functional).** A carrier on site a (state |0⟩, energy E₀) or site b (|1⟩,
E₁), coupling −t (t > 0), Δ = E₁ − E₀. The lowest energy over states with mean transferred charge
⟨n_b⟩ = ν is

    E(ν) = (1 − ν) E₀ + ν E₁ − 2 t √(ν (1 − ν)).

*Proof.* Any normalised state is ψ = c₀|0⟩ + c₁|1⟩ with |c₁|² = ν. ⟨H⟩ = (1−ν)E₀ + νE₁ − 2t Re(c₀* c₁),
minimal for c₀, c₁ real and of the same sign, where Re(c₀*c₁) = √(ν(1−ν)). Minimising over ν gives the
lower eigenvalue (E₀+E₁)/2 − ½√(Δ² + 4t²). ∎

Consequences:
- **Limits.**
  - t → 0: E(ν) is linear in ν, the derivative discontinuity of fractional charge (Perdew, Parr, Levy
    & Balduz, PRL 49, 1691 (1982)); integer charges are exact.
  - t ≫ Δ: E(ν) is smooth around ν = ½.
- **Curvature at the minimum.** With g = ν(1−ν), d²√g/dν² = −1/(4 g^{3/2}), so E″ = t/(2 g^{3/2}). At ν*,
  g* = t²/(Δ² + 4t²), so

      κ₁ = E″(ν*) = (Δ² + 4t²)^{3/2} / (2 t²)        (one carrier)

  This is 4t when t ≫ Δ and Δ³/(2t²) when Δ ≫ t.
- **Field response.** The field response of the exact eigenvalue is α = e²d²·2t²/G³ = e²d²/κ₁,
  G = √(Δ² + 4t²). So a quadratic bond variable with hardness κ₁ reproduces the exact polarisability
  at any t/Δ. The two-electron version, κ₂ = G³/(4h²), is TRACE's two-level anchor (SPEC 26.2).
- **The quadratic is not exact at large drive.** Driven by Δ → Δ − δ, the exact response saturates
  (ν ≤ 1); the quadratic one does not. The two agree to O((δ/G)²).

**Theorem B (one carrier on any number of sites).** For one carrier with site energies ε_u and couplings
t_uv ≥ 0 on a connected graph,

    E₀ = λ_min(diag ε − T) = min over ν in the simplex of  Σ_u ν_u ε_u − Σ_{u≠v} t_uv √(ν_u ν_v)

*Proof.* For any ψ, −t_uv ψ_u ψ_v ≥ −t_uv |ψ_u||ψ_v|, so ⟨ψ|H|ψ⟩ ≥ ⟨|ψ||H||ψ|⟩. The minimum is reached at
ψ = √ν ≥ 0 (Perron–Frobenius). ∎

This is the lattice von Weizsäcker functional. It requires t_uv ≥ 0 (stoquastic hopping). With signed
couplings on a non-bipartite graph the √ν form is only a bound. The statement that holds for any signs is
the next one: the integer-configuration ensemble with one carrier is exact diagonalisation of the one-carrier
tight-binding Hamiltonian, delocalised or not. (The band unit's Harrison hopping is positive, so the √ν form
holds there; TRACE's learned couplings τ are signed.) A hole spread over C₁₀H₃⁺ (13 atoms) is 13 configurations, not
a band-like explosion.

**Theorem C (why the chain explodes: counting).** Let a covalent unit have N_b bonds whose pair
excitation (u → +1, bonded v → −1) costs ε_p < W, the window. Every set of m disjoint such pairs with
m ε_p ≲ W lies in the window. So the count is at least Σ_{m ≤ W/ε_p} C(N_b, m) on a chain. In an ionic
crystal ε_p is a charge-transfer gap (MgO: Mg⁺O⁻ ≈ 5–8 eV > W), so the window holds the frame alone.

**Measured: why ε_p is small in the chain** (2026-10-16; the refused call replayed from its dump,
untrained heads, cap raised).
- The window holds **1451** configurations.
- The lowest configuration of C₁₀H₃⁺ is a charge-ordered chain, [0, 0, +1, 0, −1, +1, +1, −1, −1, +1, 0,
  0, 0]: 6 of its 12 bonds join charged atoms. It lies only 0.80–1.13 eV below every single-hole
  configuration on carbon.
- **Three artifacts ruled out:**
  1. the learned state shifts: they are exactly zero at initialisation;
  2. the ionic collapse of compact widths: carbon's width is already the ladder width, S₀ = IE − EA
     = 10.0 eV, under both width rules, so the frame-rule check tested nothing;
  3. over-screening: W_uu/S₀ = 0.75–0.82, and b ≈ 0.
- **The cause:**
  - The configuration energies have no configuration-dependent covalent energy. A bond between two
    neutral open-shell atoms carries the kinetic-exchange (resonance) energy of the PPP model, ~4t²/U.
    That is 2.3 eV with the textbook π hopping t ≈ 2.4 eV and U = IE − EA = 10 eV (ESTABLISHED
    magnitude, PPP/Hubbard second order).
  - A configuration that charges both atoms of a bond loses that energy.
  - TRACE gives every configuration the same bonding energy (the ACE), so charge-ordered configurations
    are about 6 × 2.3 ≈ 14 eV too low (estimate).
  - The window then fills with them.
- **The same mechanism, as a hypothesis, explains step 9's learned shifts.** Au⁻ +1.61 and Au⁺ +1.40 eV,
  at 80 % of the bound (step 8: +0.47 / +0.24): the model pushes the ionic Au⁺Au⁻ configuration back up
  by roughly the Au–Au covalent energy it does not otherwise lose.
- The explosion is pair excitations within covalent bonds that are missing their bond energy. It is not
  delocalised carriers. (The 2026-10-16 README entry first said "genuinely delocalised"; corrected.)

## 3. NEW-TO-TRACE: free carriers vs bound fluctuations

Semiconductor physics separates bound valence electrons (the dielectric response) from free carriers.
TRACE's sectors are named that way, but the formal stack let the free sector create electron–hole pairs
inside covalent bonds. Their energy there is wrong (the measured diagnosis above: no covalent-energy
loss), and they double count the screening the bound sector already holds.

There are two consistent fixes:
- (i) give every configuration the covalent energy it loses (the full PPP problem, exponential in a
  covalent network);
- (ii) keep pair fluctuations within covalent units out of configuration space and represent them, as
  fluctuations, in the bound sector. The ACE then holds the bonding energy of the neutral bond, which is
  correct because the reference configuration is the neutral one.

(ii) is universal and cheap. The partition:

1. **Covalent units.** These are the connected components of the chemical-bond graph restricted to bonds
   between two open-shell (neutral-frame) atoms: C–C, C–H, Au–Au, Ag–Ag. A closed-shell frame atom (Mg²⁺,
   O²⁻, Na⁺, Cl⁻, Al³⁺) is its own unit. Its contacts are closed-shell ionic contacts, where Theorem C's
   ε_p is a gap.
2. **Carriers.** Free carriers are integer deviations d_u = n_u − f_u that are sign-definite within each
   covalent unit: all d_u ≥ 0 (holes) or all ≤ 0 (electrons). A carrier may sit on any atom of its unit
   and hop between atoms (Theorem B: exact tight binding). Between different units every integer
   configuration is allowed, so real charge transfer between units is kept: ionic dissociation Na⁰Cl⁰,
   donor → acceptor Al → Au₂ across the slab, outer-sphere redox.
3. **Fluctuations.** Every electron–hole fluctuation within a covalent unit, and every fluctuation
   across an ionic contact, is bound-sector linear response with the two-level hardness (Theorem A).

**Counting.** A unit with N atoms and |D| carriers has at most C(N, |D|) placements, times the ladder
depth. C₁₀H₂ has one configuration and C₁₀H₃⁺ thirteen (one hole on any of 13 atoms). Au₂/MgO keeps its integer Al → Au₂ transfer, and
undoped Au₂ is its frame (Au⁰Au⁰). Its polarisation becomes a bound-sector bond response.

**What this removes from step 9 (stated before any test).** Step 9's undoped upright stabilisation came
from the Au⁺Au⁻ resonance in configuration space: free sector +811 meV of the +875 meV wetting energy,
p(Au⁺) = 0.116. Under §3 that pair is a fluctuation, so it must come from the bound sector. It can only
do so if the exclusion acts on bound charges too (§4). Without §4 the partition would lose step 9's
mechanism. §3 and §4 are one change.

## 4. NEW-TO-TRACE: exclusion on bound fluctuations (Parr–Pearson finite differences)

A bound fluctuation δ on atom u changes its electron count fractionally. The exact ensemble energy is
linear in δ between integers (PPLB), with a kink at the frame. The smooth closure the bound sector already
uses for the ladder itself is the finite-difference electronegativity and hardness (ESTABLISHED: Parr &
Pearson, J. Am. Chem. Soc. 105, 7512 (1983); Mulliken χ = (IE + EA)/2, η = IE − EA). Applied to the
exclusion energy Π_u(n) of SPEC 18:

    χ^P_u = ½ [Π_u(f_u + 1) − Π_u(f_u − 1)],        η^P_u = Π_u(f_u + 1) + Π_u(f_u − 1)  (Π_u(f_u) = 0)
    χ_u → χ_u + χ^P_u,   η_u → η_u + η^P_u          (bound sector)

These are the exclusion's shift of the in-situ IE and EA.

Measured at the defaults (K = 10 Ha, I_eff = 9 eV), Au⁰ atop O²⁻ at 2.2 Å: Π(Au⁺) = −1.345 eV and
Π(Au⁻) = +1.859 eV, so χ^P = −1.60 eV and η^P = +0.51 eV.
- The near Au becomes a donor by 1.6 eV relative to the far Au, so the bound Au–Au bond variable moves
  electron density from the near to the far Au. That is DFT's sign (Hirshfeld +0.11 / −0.13), and it is
  step 9's mechanism carried as linear response.
- η^P ≥ 0 is required for the bound sector to stay convex. It holds here; a negative value would be
  refused, never clipped silently.

Exact properties:
- zero for an isolated atom and wherever Π ≡ 0 (open-shell neighbours only);
- a uniform shift on equivalent sites;
- no change to any isolated-atom response prior (Au α, Ag α).

It does change the bulk ε∞ through η^P on cation sites (Π(Mg⁺) > 0). That is real physics, the Pauli
stiffening of ionic covalency; the ε∞ prior refits it.

## 5. NEW-TO-TRACE (next): in-situ removal energies for every density

The densities of SPEC 18 use gas-phase removal energies for carriers, and a learned I_eff for gas-unbound
anion frames. That is not consistent:
- For an anion carrier (a hole on O²⁻), the gas O⁻ removal energy (EA(O) = 1.46 eV) gives a hole state
  more diffuse than its in-crystal frame, so a hole would cost exclusion instead of relieving it. That is
  the wrong sign. It is why SPEC 18 excludes anion carriers, and why NaCl's Cl hole is outside it.
- The consistent rule is the in-situ removal energy: the model's own configuration-energy difference at
  the frame,

      I_u^env(n) = ê_u(n+1) − ê_u(n) + [b + W f]_u + ½ W_uu,       κ_u(n) = √(2 I_u^env(n))

  It contains the Madelung shift (+eφ on anion sites, −eφ on cation sites; ESTABLISHED as the point-ion
  shift of the ionic model) and the screened polarisation through W.
- It replaces the learned I_eff with a computed quantity and treats anion and cation carriers alike.
- Circularity is avoided by taking I^env without Π (zeroth order). It needs 3-D overlap tables (R, κ_u,
  κ_v), which are cheap as Chebyshev series.
- Built after §3–§4 have been measured.

## 6. Exact limits the build must pass (pre-stated tests)

1. An isolated atom: the ladder, unchanged.
2. Bulk formal MgO and NaCl: the frame configuration alone; energy unchanged by §3.
3. C₁₀H₂: one configuration. C₁₀H₃⁺: carriers on the unit's atoms only; the count equals brute-force
   enumeration of sign-definite configurations within the window.
4. H₂⁺ (one carrier, two sites): the configuration energy equals the two-state eigenvalue of Theorem A.
5. Na–Cl stretched to dissociation: Na⁰Cl⁰ is admissible, being between units.
6. χ^P and η^P equal the finite differences of Π. They vanish for an isolated atom. The bound sector stays
   positive definite.
7. Forces −∇E, and smooth wherever the unit partition is fixed.
   - Stated limitation: a bond forming between two open-shell atoms changes the partition. SPEC 12's
     percolation mixture over uncertain bonds is the smooth route, not built in the first version.

## 7. Benchmark predictions (to be pre-registered with the runs)

- **Carbon chain:** the stack runs; C₁₀H₂ is identical in structure to the round-5 mean-field model.
  Prediction (hypothesis): the pooled force error is at the round-5 level (0.0356 eV/Å). The cation now
  has a tight-binding carrier instead of a mean-field fraction, which may help its weak point (C₁₀H₃⁺
  0.047 eV/Å).
- **Ag₃±:** one carrier on a 3-atom unit; prediction: at the round-5 level.
- **Au₂/MgO** (step 11 = step 9 + §3 + §4):
  - the decisive test of whether the partition keeps step 9's physics;
  - criteria: T5 within ±15 % of DFT; undoped class offset ≤ 100 meV; doped electron on the far Au.
  - **Risk:** at large drive the quadratic response does not saturate (Theorem A). If step 11 overshoots,
    that names the missing physics: a non-quadratic bond response near resonance.
- **NaCl:** unchanged by §3 (all ionic). Exclusion of the Cl hole waits for §5.

## 8. Still open after this (stated)

- **Many delocalised carriers** (three Al donors in the slab, polarons in oxides). Theorem B makes one
  carrier exact. Many carriers in a band need a carrier mean field: fill the N_c lowest orbitals of the
  one-carrier tight-binding Hamiltonian, with Coulomb in mean field. That is cheaper than C(sites, N_c)
  configurations and exact for one carrier.
- **Frames for covalent molecules of electronegative elements** (O₂, organic O). formal_frame(O) = −2
  makes O a closed-shell unit. In a molecule like O₂ this is the formal ionic picture, carried by the bound
  sector. It is untested.
- **Spin** (−kT ln g_n) and **the Pauli dipole in the bound sector** (2026-10-15 note §4b).

## 9. Measured later the same day: the doped charge state, Pauli between carriers, the band question

**Step 9's two seeds split on the doped Au₂ charge state** (README results).
- Seed 42 holds mostly Au₂⁻; seed 43 holds Au₂²⁻ in every probed doped structure (T5 95 % against
  125 %).
- The undoped offset reproduces (−33 / −72 meV).
- K and I_eff are equal across seeds (7.74 / 7.84 Ha; 9.29 / 9.14 eV), and so are the mean learned shifts.
  The 2 eV difference in the Au₂²⁻ − Au₂⁻ gap lives in the environment terms.
- This is an identifiability failure. Nothing in the model ties Au₂'s bond strength to its charge state,
  because the ACE is configuration-independent.

**NEW-TO-TRACE: bond order follows charge state through Pauli repulsion between carrier units** (SPEC 18.5).
- Two frontier densities repel when either is a closed shell:
  - Au⁻Au⁻ (σ²σ*², bond order 0): 4KS;
  - Au⁻Au⁰ (σ²σ*¹): 2KS;
  - open–open: 0 (bonding);
  - an empty shell: 0.
- SPEC 18 is the case of a closed-shell frame neighbour.
- At K = 7.74 Ha and 2.47 Å the term is 0.45 eV for Au⁻Au⁻, with ~0.7 eV/Å on the bond.
- On step 9 seed 43's frozen weights Au₂²⁻ lies 1.7–2.4 eV below Au₂⁻, so the term cannot flip a trained
  model by its energy. Step 11 tests whether training with it, through the bond forces, avoids the Au₂²⁻
  basin.

**A core–core Pauli wall for the compressed dimer (estimated, not built).**
- The Au 5d¹⁰ cores (κ from IE₂ = 20.2 eV, N = 10, K = 7.74 Ha) give 3.5 eV and 10.8 eV/Å at 2.47 Å, and
  11.0 eV and 32 eV/Å at 2.09 Å.
- The force rise from 2.47 to 2.09 Å (21 eV/Å) is the order of DFT's (0 → 15.8 eV/Å). But the 3.5 eV at
  equilibrium shows that the density-overlap model overstates the repulsion of a 5d shell that takes part
  in s–d bonding.
- Not built: it would need its own calibration, and a wall the ACE must cancel at equilibrium risks worse
  extrapolation.

**The band question, measured.**
- In step 9's doped ground states, an electron on any single Mg (Mg⁺) costs +3.9 to +4.3 eV.
- A delocalised channel competes only if hopping lowers the band bottom by ≳ 4 eV, i.e. τ_MgMg ≳ 0.33 eV
  over 12 neighbours.
- Physical s–s hopping at 2.98 Å (Harrison's universal V_ssσ = 1.40 ħ²/(m d²) ≈ 1.2 eV; ESTABLISHED,
  Harrison 1980) would exceed that. PBE's delocalised donor electrons are consistent with it.
- The trained model cannot learn τ_MgMg, because Mg⁺ never enters its window. A band channel needs
  physically anchored couplings, τ = h_Harrison × a bounded learned factor, like SPEC 26.2's bond hardness.
  It also needs a many-carrier treatment that does not enumerate C(54, 3) site configurations.
- Next build if step 11 does not settle the charge state.

## 10. NEW-TO-TRACE: a band unit for delocalised carriers (design; built test-first, not yet trained)

**Sites and Hamiltonian.** Band sites are the atoms with cation frames (f > 0). Their extra-electron states
are removed from their own units and carried by one band unit. The one-carrier Hamiltonian over band sites:

    h_vv' = ε_v δ_vv' − τ_vv',      ε_v = E(f − e_v) − E(f)                (the exact configuration energy of
                                                                             one electron at v, all else at frame)
    τ_vv' = V_ssσ(R) · exp(ln 20 · tanh s) · s_cut(R),    V_ssσ = 1.40 ħ²/(m R²)  (Harrison; ESTABLISHED),
    s a learned scalar (one per model), so the factor lies in (1/20, 20).

**Band states.** The band unit holds m = 0, 1, 2 electrons, all in the lowest orbital φ₁.
- λ₁ is non-degenerate on a connected hopping graph with τ > 0 (Perron–Frobenius), so its first and
  second derivatives exist.
- Target energies relative to the frame:

      E_b(1) = λ₁                                  (exact one-carrier tight binding, Theorem B)
      E_b(2) = 2 λ₁ + J₁₁,   J₁₁ = Σ_uv φ_u² φ_v² V_uv,   V_vv = ê_v(f−2) − 2ê_v(f−1) + ê_v(f) + W_vv,  V_uv = W_uv

  E_b(2) is exact for a spin-paired pair in a frozen φ₁ (two-electron Hartree).

**Coupling to the rest.**
- The band's charge −m sits on the sites with weights φ₁². That is exact for m ≤ 2 because both electrons
  share φ₁.
- Its b, W rows are φ₁²-weighted; ê_b(m) is fixed so that the band alone reproduces E_b(m) exactly.
- Hopping between the band and a unit u is Σ_v T_uv φ_v (T the learned coupling).

**Exact limits (the tests).**
1. τ → 0: φ₁ localises on the lowest site, E_b(1) = min_v ε_v and E_b(2) = 2ε_v + V_vv = E(f − 2e_v) −
   E(f).
2. One band electron and no other carrier: E_b(1) equals the ground state of the atom-resolved
   configuration ensemble with couplings τ (Theorem B).
3. No carriers (bulk formal MgO): unchanged energy.
4. Forces −∇E.

**Stated limitations.**
- At most two band electrons, both in φ₁. Two electrons localised on two different deep sites (two Al²⁺)
  need a second orbital and are not represented.
- The hopping anchor is the universal s–s value. MgO's cation band is O-mediated, so the learned factor
  carries the difference.

## 11. NEW-TO-TRACE: in-situ removal energies for every exclusion density (design, 2026-10-17)

**The inconsistency.** SPEC 18 builds each carrier state's frontier density from the gas-phase removal energy
I_gas(n) = E_Z(n+1) − E_Z(n), and each anion frame's from a learned I_eff. Three consequences:
- the gas-unbound anion (O²⁻) needs a fitted, system-specific parameter;
- a hole on an anion (Cl⁰ in Na₈Cl₈⁺, O⁻ in an oxide) cannot be treated, because gas EA(O) = 1.46 eV makes
  the hole state more diffuse than its frame, which reverses the sign of the exclusion;
- a cation carrier in an anion cage (Al²⁺ in MgO) is treated as tightly bound (IE₃ = 28.4 eV), whereas the
  Madelung field of the cage weakens its binding by many eV.

**The rule.** Use the model's own in-situ removal energy, the configuration-energy difference for one electron
leaving unit u with all other units at their frames, taken without Π (zeroth order, no circularity):

    I_u^env(n) = E(f + (d+1) e_u) − E(f + d e_u)
               = ê_u(n+1) − ê_u(n) + (b + W f)_u + ½ (2d + 1) W_uu,        d = n − f_u

    κ_u(n) = √(2 Ĩ_u(n)),   Ĩ = I_floor ln(1 + exp(I^env / I_floor)),   I_floor = 0.5 eV  (smooth, positive)

Ĩ is used for the carrier states (any frame, anions included) and for every closed-shell neighbour's frame
density, replacing I_eff. β keeps its asymptotic form, (n + 1)/κ − 1 for carriers, and β = 0 for anion
frames as before.

**What it contains.** The ladder, the learned state shifts, the Madelung potential of the frames screened by
the bound sector, and the screened self-energy, all of it already in TRACE. No new parameter is added, and
I_eff is removed.

**Exact identity (test).** For an isolated atom, W_uu = S₀ and b_u = 0, so I^env(n) = I_gas(n) + δ(n+1) −
δ(n), which is the gas value when the shifts are zero.

**Stated approximation.** The decay rate of an in-crystal frontier density is modelled by the in-situ removal
energy measured from vacuum. A crystal's interstitial potential is not vacuum, so this is a model (in the
spirit of the compressible-ion picture, ESTABLISHED: Madden & Wilson), not a theorem.

**Numerics.** κ_u and κ_v now both vary, so S needs Chebyshev tables in (R, κ_u, κ_v) per pair of asymptotic
charges (β follows κ).

**§11, measured and refined (2026-10-17).** On step 9's weights the rule is sound for anion frames and wrong for
carriers in cages:
- O²⁻ comes out at 9.8 eV in the surface layers and 11.2–11.5 eV in the bulk layers. The single learned I_eff
  (9.29) had found the surface value.
- A localised Mg⁺ comes out unbound (−3 eV), which would make its density infinitely diffuse. A caged carrier's
  density near its neighbours is set by its own nucleus and the cage, not by its binding to the vacuum.
- **Built** (SPEC 18.6): in-situ densities for anion frames only. The counterfactual is a drop-in for I_eff.
- **Anion holes with in-situ densities:** the correct sign in isolation, but on the trained weights a hole
  releases frame–frame O–O overlap that the ACE already holds, and drives spurious O → Au transfer. Off until
  that double count is resolved.
