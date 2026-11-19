# Audit of the TRACE electrostatics and redox equations against established physics (2026-10-09)

Purpose: before any further code, check every equation, existing and proposed,
against established theory. That includes what a model must contain for
**Marcus electron-transfer theory** to emerge when it is simulated.

Verdicts:
- **correct**: exact, or the stated established approximation, inside its
  domain;
- **defect**: measured to be wrong for a target system;
- **limit**: correct, but restricted to a regime that must be stated.

## 1. What Marcus theory requires of a simulation model (ESTABLISHED)

| # | requirement | source |
|---|---|---|
| R1 | Diabatic states with integer, localised charge (D⁻A and DA⁻) and geometry-dependent energies E_D(R), E_A(R). Marcus's reaction coordinate is the energy gap ΔE(R) = E_A − E_D. | Marcus 1956; Lax and Warshel (energy-gap coordinate); constrained DFT builds such states by constraining fragment charges (Wu & Van Voorhis) |
| R2 | Fast electronic polarisation equilibrated to each diabatic state (it follows the electron adiabatically), with the correct optical dielectric response. Polarisable force fields lower λ_s by 30–45 % against fixed charges. | Pekar / Marcus; polarisable-FF studies |
| R3 | The spatial size of the transferring charge distributions. In the two-sphere model, λ_o = (Δe)²(1/2a_D + 1/2a_A − 1/R)(1/ε_op − 1/ε_s). | Marcus 1956 |
| R4 | Slow (nuclear) polarisation from the nuclear dynamics of the full model. | — |
| R5 | Electronic coupling H_DA(R) between the diabatic states. It decays roughly exponentially with distance. | Marcus–Hush; Hopfield |
| R6 | The two-state Hamiltonian [[E_D, H_DA], [H_DA, E_A]]: class II (double well, localised) if λ > 2H_DA, class III (single well, delocalised) if 2H_DA ≥ λ. | Robin–Day; Marcus–Hush |
| R7 | Linear-response consistency in MD: σ²(ΔE) = 2λkT, and λ = ½(⟨ΔE⟩_D − ⟨ΔE⟩_A). | fluctuation–dissipation; Warshel |

## 2. Audit of the existing equations

| id | equation | verdict | evidence |
|---|---|---|---|
| A1 | Bound sector: SQE + induced dipoles, exactly eliminated (W = J − AK⁻¹Aᵀ, b = Ay₀) | **correct** as the second-order expansion of the band energy about a gapped reference (Theorem 6), for small, local charge flow | κ = G³/(4h²) checked to 6 digits; ε∞ identity to 1e-16 (Theorem 1) |
| A2 | Bond hardness κ_b learned freely | **defect**: bonds with t* ≈ 0 (Mg–Mg) are unidentified (∂E/∂κ = ½t*²) and sit at the floor, giving ε∞ ≈ 1350 for MgO (experiment ≈ 3). This violates **R2**. | measured (T6: 1379, 1346) |
| A3 | Dipole sector (Gaussian/Thole, Lorentz local field) | **correct**: indefinite only through the physical Lorentz mode | λ_min −2.37 eV/e² on the dipole block only |
| B1 | Redox ladder Ê(±1) = ∓χ + G/2, anchored at neutral-atom IE/EA (exact PPLB integer energies for an isolated atom) | **limit**: correct relative to neutral atoms. **Defect** in ionic solids: no closed-shell rule (redoxSQE: "the acceptor cannot take one, e.g. Cl⁻"), so O²⁻ in MgO accepts. | measured (T4: Al → O) |
| B2 | Free–bound on-site coupling = Coulomb J_fb only | **defect** in principle (Theorem 2: the chemical hardness should act on q_b + n); the effect is small here | P1: +0.085 / +0.258 eV |
| B3 | Free-charge width σ_free from the hardness rule | **defect** for localised redox states, violating **R3**. σ_free = 1.8 (O) to 4.0 (Al) Å, so the free–free Mg–O Coulomb is 46 % of a point charge, and formal ionic MgO comes out +856 eV above neutral (Born–Haber: ≈ −960). Diffuse widths suit band carriers, not localised ions. | measured (P3) |
| B4 | Mean-field free sector within a molecule: Σ_u Φ_u(c_u) + ½ cᵀWc for fractional c | **limit**: each Φ_u is piecewise linear between integers (exact PPLB per unit), but the Hartree term between fractional charges is convex. That is the root of DFT's delocalisation error (Cohen, Mori-Sánchez & Yang). The certified SIC removes the on-site part only. So within one molecule a carrier is biased to spread (class III). This is right for imitating PBE data and **wrong for R1 / R6 (class II localisation)**. | structure of the functional; delocalisation-error literature |
| B5 | Exact canonical ensemble between molecules: F = −kT ln Σ_N e^{−E_N/kT}, each E_N a constrained-fragment-charge state | **correct**: E_N are diabatic states in the constrained-DFT sense (R1). **Limit**: no coupling (R5, R6), so this is the purely non-adiabatic, incoherent mixture; there is no avoided crossing and no class III. | — |
| B6 | Training signal through levels and configurations ∝ occupation | **defect for learning** (Theorem 3): ∝ e^{−ΔE/kT}, 10⁻⁸ for Au₂ | measured (P1 and T4 context) |

## 3. Audit of the proposed updates

| id | proposal | verdict |
|---|---|---|
| C1 | One-body "coherent free carriers" (§6 of the 2026-10-09 band note): H = diag(ε) + T, Hartree W, Mermin | **limit, corrected here.** It is the mean-field band regime. It inherits B4's delocalisation error, so it can imitate PBE (fractional transfer, metals) but **cannot produce Marcus class-II localisation**. It must be stated as the band/metallic regime, not as the electron-transfer model. |
| C2 | **The diabatic Hamiltonian over configurations**: H_NN = E_N (the existing constrained energies); H_NN′ = t_uv when N′ differs from N by one carrier moved from unit u to unit v (Slater–Condon one-body rule; fermionic sign for several carriers); H_NN′ = 0 otherwise. Electronic free energy F = −kT ln Tr e^{−H/kT}, or λ_min(H) as kT → 0. Forces −∇F = −Tr[ρ ∇H] (Hellmann–Feynman), with ρ = e^{−H/kT}/Z. | **correct, and Marcus-ready.** With t = 0 it is exactly B5 (Tr e^{−H/kT} = Σ_N e^{−E_N/kT}). Two configurations give the Marcus–Hush Hamiltonian, so R1, R5 and R6 hold, and class II/III emerges from λ against 2t. It has no self-interaction error, because each E_N carries integer charges with polarisation equilibrated to that configuration (R2, once A2 is fixed). For one carrier on U units it is a U×U matrix, the same cost as C1, without the delocalisation error. ESTABLISHED as EVB (Warshel & Weiss 1980) and CDFT-CI (Wu & Van Voorhis); NEW-TO-TRACE through diabatic energies from the exact polarisable elimination, learned couplings, and the canonical-ensemble selection already in trace_redox. |
| C3 | kT homotopy | heuristic for training only; no physics claim |
| C4 | Formal oxidation states with the closed-shell capacity | correct physics (redoxSQE, Born–Haber); **blocked by B3**, so it needs ionic-size free charges first |

The algebraic training signal follows for C2 as well: the population of a
configuration coupled by t at gap ΔE is t²/ΔE² (Theorem 7 applied in
configuration space). So C2 fixes B6 without any heuristic.

**Correction (2026-10-09, measured).** That holds only where t is not negligible, i.e. for
nearby units. In Au₂/MgO:Al the donors sit 10.7–14.7 Å from Au, so direct couplings vanish
and the transfer is band-mediated. The diabatic probe (trace_redox SPEC §17.4) points the
benchmark at B1/C4 instead.

## 4. What must change, in order, and why (each step names the requirement and the benchmark number it moves)

1. **A2, a physical bond hardness.** The two-level law κ = G³/(4h²), the
   Pauli rule κ = ∞ for closed-shell like pairs, and a nearsighted
   envelope; or ε∞ data where a dataset carries it.
   - Needed by R2 for every Marcus λ.
   - Benchmark: T6 (ε∞ ≈ 3) and the screening that sets T4/T5.
2. **B3, physical free-charge sizes.** σ for localised states from ionic and
   atomic radii; diffuse only for band carriers.
   - Needed by R3 (λ ∝ 1/a).
   - Benchmark: P3 (ionic Born–Haber energetics) and T5.
   - **Result (2026-10-09): B3 cannot be fixed inside the mean-field free sector.**
     - A convex ladder (exact isolated-ion energies) caps the on-site curvature at
       IE − EA. Compact ionic pair charges give formal MgO a Madelung field of
       11.95 eV/e².
     - λ_min(J) is −6.6 eV/e² even at the cap. The hardness rule σ ≥ σ_H is this
       consistency boundary.
     - So localised ionic charges need integer configurations: **B3 moves into
       step 3 (C2)**.
     - Built and tested for it: Shannon crystal radii with moment matching (formal
       MgO keeps 99.35 % of its Madelung energy, against 42 %); an Ewald-split
       free-charge route that stays cheap for compact widths; the on-site width
       split.
     - Details: trace_electrostatics CHANGELOG Round 15.
3. **C2, the diabatic Hamiltonian with couplings,** first between molecules
   (where B5 already enumerates configurations), then for a single carrier on
   units within a molecule.
   - Needed by R1, R5, R6, R7, and by B6 (training signal).
   - Benchmark: T2, T4, T5.
4. **B1/C4, the closed-shell rule** through formal states, once B3 allows
   ionic charges.
5. **C1, the band regime,** only where the reference data are metallic or
   delocalised (PBE-like), stated as such.

Every step is test-driven. The failing test comes first: closed forms (the
two-state Marcus–Hush eigenvalues; class II/III at λ = 2t; t = 0 → B5
exactly; Hellmann–Feynman forces against finite differences), then the
module, then one training run scored by the unchanged T1–T6.

## 5. Sources consulted (2026-10-09)

- Robin–Day classes and the Marcus–Hush two-state model: Chem. Soc. Rev.
  review (pubs.rsc.org c3cs60481k); arXiv:2410.13144.
- Energy-gap coordinate, σ² = 2λkT, Stokes and variance definitions:
  arXiv:cond-mat/0508109; arXiv:1810.02168; arXiv:0709.4282.
- Electronic polarisation and reorganisation energy (30–45 % reductions;
  Pekar separation): polarisable-FF studies indexed at tcbg.illinois.edu
  (Vladimirov, Ivanova & Rösch) and portal.fis.tum.de; Dinpajooh, Newton &
  Matyushov (2017).
- Constrained DFT diabatic states and couplings: dspace.mit.edu (Van Voorhis
  et al., CDFT review); arXiv:2005.08021 (PyCDFT); arXiv:1508.02735.
- Delocalisation and self-interaction error: Cohen, Mori-Sánchez & Yang
  (2008); stretched H₂⁺ example.
- redoxSQE (formal oxidation states, the closed-shell rule):
  arXiv:1308.3424; arXiv:1308.3573.

Primary papers (Marcus 1956, Warshel & Weiss 1980, Wu & Van Voorhis 2005/2007)
were cited through these secondary sources. Read the primary texts before any
claim in a manuscript.
