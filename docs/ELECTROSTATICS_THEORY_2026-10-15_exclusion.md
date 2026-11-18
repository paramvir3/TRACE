# What physics TRACE is missing: exclusion (Pauli) and the polarisation it drives (2026-10-15)

Written after steps 1–7 on Au₂/MgO:Al. Every claim below is either measured (named with its probe), derived
here, or labelled a hypothesis. §2–§3 are implemented as trace_redox SPEC 18 (step 9 trains on it), §4a as
SPEC 18.4 (not yet trained); §4b is not implemented.

## 0. The evidence that one piece of physics is missing

TRACE's energy is E = E_ACE(R) + E_b0 + F_free. Here F_free is the canonical free energy over integer
charge configurations n, with E(n) = Σ_u ê_u(n_u) + b·n + ½nᵀWn and ê_u(n) = E_Z(n) + δ_u(n) − S_u(n).
- The ACE part is configuration-independent.
- The electrostatic parts are Coulomb plus linear response.
- Nothing makes the energy of a charge state depend on how its electron density overlaps the
  closed shells around it, except the bounded learned shift δ_u(n) (±2 eV).

Measured failures that all point to that gap:

| failure (probe) | what exclusion would do |
|---|---|
| Doped electron kept as localised Al²⁺ (P5; steps 2–7); DFT's Al is Al³⁺-like (Hirshfeld) | Al²⁺'s 3s electron overlaps six O²⁻ closed shells: a large exclusion cost |
| Diffuse carriers penetrate the compact O²⁻ charges and gain several eV (step 7; counterfactuals: Al diffuse → no transfer, Mg/O diffuse → Mg⁺ "band-like") | exclusion forbids the penetration |
| Doped-upright electron on the near Au (site decomposition: ΔE −0.43 eV, DFT far) | an extra Au 6s electron atop O²⁻ pays exclusion; the far Au does not |
| Compressed Au₂ (2.09–2.20 Å): Au force errors 8–17 eV/Å, extrapolation (step 7 with neutral Au₂: 15.5) | a physical overlap wall extrapolates where data do not |
| Au under-bound toward MgO by ~0.04–0.09 eV/Å at fixed substrate (`probe_au_vertical_force.py`); literature: Au–O bonding is orbital hybridisation and polarisation (PRL 96, 146804) | exclusion pushes Au's valence density off the O²⁻, an exchange-driven dipole (§4) |

**Magnitude pre-check (K = 10 a.u.; I_eff(O²⁻) scanned 6–12 eV; unregularised densities r^(2β)e^(−2κr)). The package regularises them as (r² + κ⁻²)^β e^(−2κr), finite at the nucleus; for Al²⁺ that gives 8.09 eV instead of 8.95 at I_eff = 9 eV, a 9.6 % definitional difference:**
- Au⁻'s extra electron atop O²⁻ (2.2 Å) against the far Au (4.7 Å): **+0.68 to +1.18 eV** penalty
  (needed ≥ +0.4 eV to reverse the measured −0.43 eV);
- Al²⁺'s 3s electron against its six O²⁻: **+7.5 to +10.9 eV** (needed ≳ 1–2.5 eV).

## 1. Established ingredients (cited)

- **Density-overlap (Pauli) repulsion:** E_rep ≈ K·S, S = ∫ρ_A ρ_B d³r, between closed shells.
  - Exponent 0.97–0.99 (He, Ne, Ar dimers).
  - K ≈ 10 atomic units (Kim et al.; Ihm, Cole, Toigo & Klein; Wheatley & Price, Mol. Phys. 69, 507
    (1990); Söderhjelm, Karlström & Ryde, J. Chem. Phys. 124, 244101 (2006)).
  - Pauli repulsion is classical in its energetics: a depletion of density between the nuclei (Rackers
    & Ponder, J. Chem. Phys. 150, 084104 (2019)).
- **Exact asymptotics of the frontier orbital** (Katriel & Davidson, PNAS 77, 4403 (1980); Levy,
  Perdew & Sahni (1984); Morrell, Parr & Levy (1975)):

      φ(r) ~ r^β e^(−κr),   κ = √(2I),   β = (q + 1)/κ − 1      (atomic units)

  where I is the energy to remove that electron and q + 1 is the charge left behind.
- **Exchange polarisation:** Pauli repulsion also deforms the density (Rackers & Ponder, the
  anisotropic variable-polarisability model).
- **Shell model:** the repulsion acts on the valence shell, so it polarises (Dick & Overhauser,
  Phys. Rev. 112, 90 (1958)).

## 2. NEW-TO-TRACE: ladder-defined frontier densities for every charge state

TRACE already holds the exact ionisation ladder E_Z(n) of every element (NIST). For unit u in state n:

    I_u(n) = E_Z(n + 1) − E_Z(n)                      (removing its outermost electron; gas phase)
    κ_u(n) = √(2 I_u(n)),   β_u(n) = (n + 1)/κ_u(n) − 1
    ρ_u^(n)(r) = N_u(n) · r^(2β) e^(−2κr) / ∫ r^(2β) e^(−2κr) d³r

N_u(n) is the number of electrons in the open frontier shell beyond the closed core (Au: 6s → N(−1) = 2,
N(0) = 1, N(+1) = 0; Al: 3s3p beyond [Ne] → N(+3) = 0, N(+2) = 1, N(+1) = 2, ...).

So each state's size and softness follow from tabulated data, with no fitted parameter. A diffuse anion
(Au⁻: κ from EA = 2.31 eV) and a compact cation core (Al³⁺: none) come out of the same rule.

**The one exception: gas-unbound anions** (O²⁻, S²⁻, N³⁻). They exist only in a binding environment,
so their valence density decays with an in-crystal removal energy I_eff. That is one bounded
scalar per such element (O: initialised at 9 eV, range 4–16 eV).

## 3. NEW-TO-TRACE: state-dependent exclusion in the configuration energies

    Π_u(n) = K Σ_{v ∈ closed-shell frames, v ≠ u} [ S(ρ_u^(n), ρ_v^(f_v); R_uv) − S(ρ_u^(f_u), ρ_v^(f_v); R_uv) ] · s(R_uv)
    ê_u(n) → ê_u(n) + Π_u(n)

- **Closed-shell frames:** neighbours v whose formal frame f_v is a closed shell, i.e. the ionic-model
  ions (Mg²⁺, Al³⁺, O²⁻, Na⁺, Cl⁻, ...). Exclusion by a filled shell is rigorous. Open-shell and
  covalent partners (Au⁰, C, H) interact through the couplings τ and the ACE instead.
- **Zero at the frame:** Π_u(f_u) = 0. Frame–frame exclusion stays in the ACE (it is in the DFT
  energies the ACE fits), so bulk formal MgO is unchanged exactly.
- **Sign:** an extra electron (n < f) pays (Π > 0); a removed one relieves (Au⁺ loses its 6s overlap,
  Π < 0).
- **Locality:** s(R) is a C² switch at the bond cutoff. Overlaps beyond ~5 Å are below 1e-6 of contact.
- **One-body in n:** neighbours are taken at their frames, so the configuration search, couplings and
  canonical free energy are unchanged (k-fold moves include Π through ê). Carrier–carrier exclusion
  (two carriers adjacent) is second order and omitted, as stated.
- **Parameters:**
  - K, one universal positive scalar (initialised 10 a.u. = 272.1 eV);
  - I_eff for gas-unbound frame anions;
  - nothing per pair.

**Consequences that make the earlier constructions physical:**
- with Π, a diffuse carrier inside an anion cage pays for penetrating, so the core + carrier widths of
  SPEC 17.9 become admissible again;
- the localised Al²⁺ is raised;
- an electron on Au⁻ prefers the Au away from O²⁻.

## 4. NEW-TO-TRACE: exchange polarisation from the same overlap

Displacing unit u's valence density by d changes its overlaps. In the rigid-shell model (ESTABLISHED:
Dick & Overhauser 1958) the N_u(n) frontier electrons move together, held by the unit's own polarisability:
restoring energy ½k|d|², k = N_u(n)² K_E/α_u (shell relation α = Y²/k, Y = N_u(n)e). The exclusion then
acts on the shell as a force

    F^P_u = −∂/∂d Σ_v K N_u N_v S(|R_uv − d|) |_{d=0} = K N_u Σ_v N_v S′(R_uv) R̂_uv    (S′ < 0: away from v)

**Magnitudes** (Au⁰ atop O²⁻, K = 10 Ha, I_eff = 9 eV, α = 36.68 a.u.; scratch computation 2026-10-15):

| R (Å) | frozen exclusion | F^P/e as a field | Pauli stiffness ∥ / intrinsic 1/α | relief, full minimum | shell shift |
|---|---|---|---|---|---|
| 2.2 | 1.345 eV | 3.0 V/Å | 2.4× | 0.600 eV | 0.83 bohr |
| 2.5 | 0.679 eV | 1.6 V/Å | 1.3× | 0.224 eV | 0.57 bohr |
| 3.0 | 0.207 eV | 0.5 V/Å | 0.44× | 0.034 eV | 0.26 bohr |

- The drive is as strong as several V/Å, and the Pauli stiffness along the bond is comparable to the intrinsic
  one. Neither the drive alone (relief −½αF², about 7 eV here) nor a linear response without the stiffness
  is physical.
- **(4a, built: SPEC 18.4.)** The configuration energies use the relaxed exclusion
  Φ_u(n) = Σ c S − ½ gᵀ(kI + Σ c max(S″,0) R̂R̂ᵀ)⁻¹ g, which is second order in d and closed form.
  - The exact transverse curvature S′/R < 0 is dropped (positive projection). Otherwise, at large K, the
    shell slides off an O²⁻ through a spurious saddle.
  - Second order gives 0.50 eV for the 0.60 eV of the full minimum at 2.2 Å.
  - Consequence: the relief of ionising Au⁰ → Au⁺ atop O²⁻ drops from 1.35 eV (frozen) to 0.85 eV. This is
    the correction the step-9 prediction (over-ionisation) calls for.
- **(4b, not built.)** The same drive also belongs in the bound sector, as a dipole source χ_μ,u =
  F^P_u/(N_u e), the l = 1 analogue of electronegativity, with the projected Pauli stiffness added to A_μμ.
  - Then the Pauli dipole couples to every field. The field of the O²⁻ beneath Au points the same way, so
    the cross term −α E·χ_μ is stabilising.
  - It is configuration-dependent through N_u(n), which is linear in n, so it enters the elimination
    exactly. The stiffness at the frame densities keeps α configuration-independent.
  - Deferred until 4a has been measured.
- **Predicted effect (hypothesis):** Au's 6s density is pushed off the O²⁻ it sits on. The resulting
  dipole is aligned with the O²⁻ field and is stabilised linearly in that field. This is the
  orbital-polarisation bonding of the literature: on regular O²⁻ sites the metal is polarised and bound
  electrostatically, against Pauli repulsion (Inntam, Moskaleva, Neyman, Nasluzov & Rösch, Appl. Phys. A
  82, 181 (2006): embedded-cluster DFT of Cu, Ag and Au dimers and trimers on MgO(001)).

## 5. Ranked plan, each step a falsifiable test

1. **§2–3 exclusion Π** (built test-first). Pre-registered against step 6:
   - the doped-upright electron on the far Au (site ΔE > 0);
   - transfer robust at ε∞ = 3 (≥ 5/6);
   - T5 no worse;
   - compressed-dimer forces reported.
2. **§4a relaxed exclusion** (built, SPEC 18.4), if step 9 over-ionises the undoped Au₂ as predicted.
   **§4b** (the bound-sector dipole drive) if the Au–surface attraction (vertical-force probe) remains.
3. **Revisit core + carrier (17.9) with Π,** for the no-collapse theorem and admissible carrier widths.
4. **Delocalised carriers** (band note §7), if the doped slab's remaining electrons still need a band.
5. **Spin degeneracies** in the configuration ensemble (−kT ln g_n): universal correctness for
   radicals. At most 0.07 eV per unpaired electron at kT = 0.1, so not a T5 lever.

**Rejected on evidence:**
- an Au–O dative charge-transfer term (the literature: little charge transfer);
- re-weighting the loss (H1 failed);
- dispersion (absent from the PBE reference).
