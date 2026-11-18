# trace_electrostatics — changelog

## Round 46 (2026-11-15, branch `absolute-frame-2026-11-13`): the second review of SPEC 40.7: test gaps and record faults; no physics changed

The review of Round 45's corrections completed (`experiments/benchmarks_4g/runs/two_site_2026-11-14/review_spec40_7_second.json`;
README, "SPEC 40.7, second review"). No defect in the package's physics was found. **Still off by default; nothing
is trained with it; gold's entries are bit-identical.**

- **Tests added** (`test_redox_two_site.py` 72 → 78): the module's d moves keep the learned coupling with learned
  parts on and the two ensemble spaces agree (kills a Hamiltonian without the step record in either route); the
  band unit is outside the step record; the entries' line below the first point; the loader's new refusals.
  Refusal tests now probe each tolerance just above itself; the option sweep runs at three distances.
- **Code:** `TwoSite` compares the recorded table values at the entries' own distances and refuses a missing one;
  `two_site.FIXED_TERM_DATA`, `data_hashes()`: built entries record the sha256 of the four data files their fixed
  terms read and are refused when one has changed; `TwoSite(..., ignore_entries=True)` for the builder;
  `relaxation_pairs` refuses to apply built entries without the model's settings. The builder records the hashes;
  `--check` exits non-zero on a refused stored file.
- **Statements replaced (kept here under the project's never-delete policy):**
  1. SPEC 40.6 (iii) and two_site.py (iii) *"ionisation energy 1.0–1.2 eV too high"* for §40.2–40.3 alone: 1.2–1.7 eV
     on the step-resolved code (1.0–1.2 was measured with the s–s hop on d moves).
  2. SPEC 40.7 (d) *"likewise 2.6 and 3.3 eV/Å for Au₂⁻ and Au₂⁺"*: 2.59 and 3.36 eV/Å, with a saved probe.
  3. SPEC 40.7 (e) *"refused otherwise"*: the data files are now guarded; code and constants are not (stated).
  4. SPEC 40.7 *"threaded through … the cluster space"*: through the module's cluster route; `clusters.cluster_ensemble`
     is not on the module's path.
  5. two_site.py (ix) *"the s-s hop is also given to moves of a d electron"*: cured (Round 45).
- **Suites:** `tests/redox` 1118 passed, 2 xfailed (+6: `test_redox_two_site.py` 78 tests); the rest 1045 passed, 4 skipped, unchanged (Mac, 2026-11-15 07:58 IST).

## Round 45 (2026-11-14, branch `absolute-frame-2026-11-13`): SPEC 40.7 after its review: the hop resolved by charge step; the cation's entry rebuilt; the entries bounded and tied to their table

**Addendum, 2026-11-15 morning (after the mutation tests of an interrupted second review).** An earlier attempt of
the second review (cut off by the network and the usage limit before reporting) left 25 mutants of the package run
against `test_redox_two_site.py` (scratch only). Seven survived; each is now killed by a new test, checked by
running the tests against the same mutant copies:
- the selection closures and the cluster linkage judging a d move by the unresolved s–s hop (four mutants): the
  isolated cation's selected set is asserted to be {Au⁺Au⁰, Au⁰Au⁺} in both ensemble spaces;
- the pair importance of SPEC 21 computed with the unresolved coupling: asserted against its formula, move by move;
- the refusal of `two_site` with `downfold_transfer`: tested on a set the fold otherwise accepts (with the Fermi
  band SPEC 38 refused the fold first, so the earlier test did not reach the refusal);
- the stale-table check reduced to E_gs: each of E_gs, E_T, t, EA*, IP* changed after the build is refused.
**Two defects found while checking, fixed (tests first):** an entries block without its `solved_with` record, or
without the recorded `gaussian_a`, loaded and ran with no consistency check (measured: with `solved_with` removed and
E_gs changed by 0.5 eV the file loaded). `TwoSite` now refuses such a block, and `check_options` requires the
model's `gaussian_a`. `test_redox_two_site.py` 60 → 72 tests. Suites: `tests/redox` 1112 passed, 2 xfailed (+12); the rest 1045 passed, 4 skipped (Mac, 2026-11-15 07:2x IST).

A review of Round 44 (two of three readers reported before the usage limit; no skeptic ran;
`experiments/benchmarks_4g/runs/two_site_2026-11-14/review_spec40_7_partial.json`) found that the cation's entry
had been solved through configurations reached by a d-electron move carrying the s–s hop. Each finding acted on
here was reproduced first. **Still off by default; nothing is trained with it.**

- **New equation-level rule (the cure of SPEC 40.6 ix): couplings resolved by charge step.**
  `trace_redox.diabatic.StepCouplings(mask, other)`, `resolve_moves`, `resolve_matrix`; keyword `steps` on
  `diabatic_hamiltonian`, `configuration_importance`, `switched_hamiltonian`, `coupled_closure`,
  `select_configurations`, `clusters.linked_configurations`, `clusters.cluster_ensemble` and the module's two
  solution routes (default None: every existing path unchanged). For a masked atom pair the matrix entry is the
  coupling of its s moves (donor charge −1 or 0, acceptor charge 0 or +1 before the move); every other move takes
  `other` (the learned coupling). `TwoSite.resolve_hops` returns the hop matrix and the record.
- **`RedoxElectrostatics.fixed_terms_only`** is declared in `__init__` (False), switches every learned level shift
  AND every learned coupling off, and is refused in training mode.
- **Gold's entries rebuilt** on the corrected fixed terms: ρ₋, λ unchanged; ρ₊ 0.408 … 1.067 → 1.285 … 1.125 eV.
  The superseded entries (all eight distances) are in the kept SPEC text named below and in Round 44.
- **Entries' range:** every entry is switched off over `ENTRY_TAIL` = 1 Å beyond its last knot (was: exponentials
  to a 5–6 Å switch; ρ₊ was 0.40 eV at 5 Å).
- **Entries tied to what they were solved with:** `gaussian_a` of the electrostatic settings (moves the pair's
  fixed terms by 0.24 eV; it was not checked), ρ at the knots (the frame table), and the table values (E_gs, E_T,
  t, EA*, IP*); `TwoSite(path)` refuses a pair marked applied without built entries (`allow_unbuilt=True` is the
  builder's way in) and entries solved with another table. `collect_two_site.py` keeps the entries by default
  (`--drop-entries` to drop; a pair without entries is then not applied). `build_two_site_entries.py` builds on a
  private table object, restores the process's cache, and `--check` exits non-zero on a difference.
- `two_site` is refused with `downfold_transfer` (the fold's couplings are not step-resolved).
- **Statements replaced (kept here under the project's never-delete policy; SPEC 40.7 as first written is kept in
  `experiments/benchmarks_4g/runs/two_site_2026-11-14/SPEC_40_7_as_written_before_its_review.md`):**
  1. SPEC 40.7 and README *"Every pair that holds a neutral open-shell atom lacked a relaxation of 0.4–2.5 eV"*
     and the ρ₊ column 0.408 / 0.761 / 1.026 / 1.190 / 1.294 / 1.273 / 1.171 / 1.067: the one-hole pair lacks
     1.1–1.65 eV; the old column held the compensation of the Au²⁺Au⁻ channel.
  2. SPEC 40.7 *"Nor do they depend on the electrostatic settings … (0.0 eV); at kT = 0.05 eV … up to 1.3 meV"*:
     they depend on `gaussian_a` (0.24 eV); the 1.3 meV was the selection's switch on the d-move coupling, not
     temperature; after the cure kT enters by its thermal weight alone (3e-5 eV at 0.2 eV).
  3. SPEC 40.7 *"(40.5) needs no singlet root"*: it fixes U given the hop; at 2.6–3.0 Å the hop is a continuation.
  4. SPEC 40.7 *"Beyond 3.0 Å: ρ₋ and ρ₊ decay there and continue as exponentials … all three carry ρ's switch
     (5 to 6 Å)"*: one rule, the line switched off over 1 Å.
  5. SPEC 40.6 (ix) *"A strict xfail test records it; the cure is a coupling resolved by charge step"*: cured.
  6. Round 44 *"the gas-phase EA, IP and U are now right by construction"*: EA and IP at the eight distances and U
     at the hop's five; IP is the s-hole state's (a d-hole state of Au₂⁺ lies 0.32 / 1.07 eV lower at 2.2 / 2.0 Å
     in PBE and is outside the model).
  7. two_site.py *"a pair without built entries is withheld from the module for this reason"*, which no code did:
     the loader now refuses such a file.
- **Recorded, not fixed (SPEC 40.7 limits d, g, h):** the description ends at 3.0 Å (the pair's binding is taken
  off between 3.0 and 4.0 Å, force up to 3.1 eV/Å at 3.3 Å); the cation's d-hole ground state; pair states
  without an entry (Au⁻Au⁻: weight 0.003 on the doped test slabs at frozen weights).
- **Tests:** `test_redox_two_site.py` 42 → 60, no xfail left in it (the step-resolved hop; the cation as the s
  resonance; independence of thresholds and kT; refusal of each option that matters with its measured size; the
  completeness sweep over every option; `gaussian_a`, frame table, stale table and unbuilt pair refused; the
  entries' end; the switch declared and refused in training).
- **Suites:** `tests/redox` 1100 passed, 2 xfailed (the two sign limits; was 1082 passed, 3 xfailed: +18 in `test_redox_two_site.py`, now 60 tests and no xfail, the former d-move xfail being a passing test of the cure); the rest 1045 passed, 4 skipped, unchanged (Mac, 2026-11-14; the second run's wall time includes hours of sleep).

## Round 44 (2026-11-14, branch `absolute-frame-2026-11-13`): SPEC 40.7, the pair's remaining entries: the isolated gold pair is first-principles in three charge states

The step the review of Round 43 asked for. **Still off by default; nothing is trained with it.** Pre-registered and
scored in `experiments/benchmarks_4g/README.md` (2026-11-14).

- **First principles (new tool `tools/electrostatics/pair_one_carrier_curve.py`).** Au₂⁺, Au₂⁻ at eight distances and
  Ag₂⁺, Ag₂⁻ at seven, PBE and PBE0: the ion pair's ground state against the separated ion and atom, and its σ_g–σ_u
  splitting, both resonance states as symmetry-constrained self-consistent states (the first route, TDA, has no
  exact test: for a one-electron system PySCF's HF diagonalises h). The tool reproduces H₂⁺ in aug-cc-pVTZ (E_int
  −2.788 against −2.793 eV, splitting 11.840 eV). PBE: 30 of 30 points valid, no d-hole state. PBE0 EA and IP of
  Au₂ within 0.04 eV of the 2026-11-02 references. The ion pair's own hop differs from the neutral table's by up
  to 28 % (cation) and 21 % (anion, Ag). PBE and PBE0 differ by 0.2–0.5 eV in the no-hop level and by at most
  0.14 eV in EA and IP on the exact atom, which is what enters.
- **The equations (SPEC 40.7; NEW-TO-TRACE).** O_ab(∓1, 0) = −s ρ_∓(R) and O_ab(+1, −1) = +s λ(R), fixed by
  E[X₂^∓] = E_gs ∓ {EA*, −IP*} and E[X₂] = E_gs on the isolated pair through the package's own fixed terms and the
  tabulated hop. Solved once per distance by `tools/electrostatics/build_two_site_entries.py` (residual 6e-11 eV)
  and stored in `two_site.json` with the option set.
- **New in the package:** `TwoSite.carriers`, `entries`, `entry_values`, `check_options`, `ENTRY_NAMES`,
  `ENTRY_OPTIONS`, `ENTRY_TAIL`; `relaxation_pairs` writes the four classes of entries and takes the model's
  config; `RedoxElectrostatics.fixed_terms_only` (an attribute, default absent: the learned level shift is set to
  zero; for builders, probes and tests); `replace_hops` refuses a coupling window inside the entries' distances.
- **Measured, and what the entries rest on:** with no learned shift the configuration levels of gas-phase Au₂ do
  not depend on any weight (three checkpoints; every parameter of a trained model perturbed by 5 % and 30 %; fresh
  modules, 7e-15 eV), nor on the electrostatic settings, the heads' cutoff, the ensemble space, the band's form or
  the coupling bound; 1.3 meV on kT (0.05 against 0.1 eV).
- **Ag–Ag withheld from the module** (`"applied": false`, with the reason in the file): the polarizability table
  holds Ag⁻ only, so the neutral frame's polarizability is the learned head's and the pair's fixed terms depend on
  the weights (0.057 eV between two seeded modules); its entries cannot be built yet. Applied pairs: Au–Au.
- **Statements replaced (kept here under the project's never-delete policy):**
  1. SPEC 40 status *"NOT ready for training … the one-carrier levels need the same construction"*: built for
     Au–Au; the gas-phase EA, IP and U are now right by construction; a training proposal is a separate decision.
  2. SPEC 40.2 / two_site.py *"one t for both [sectors], as first principles has it (Au₂ and Au₂⁻: one t to
     0.07 eV)"*: true for Au's anion near 2.47 Å only; the cation's own hop is 13–28 % larger.
  3. SPEC 40.3 and Round 43 *"Which pairs the module uses: Au–Au and Ag–Ag"*: Au–Au.
  4. SPEC 40.6 (iii) as a limit of the whole term: now a limit of pairs without entries, and of the transfer of
     the isolated pair's conditions to a surface.
- **Tests:** `test_redox_two_site.py` 37 → 42 and one strict xfail, now on the reference option set (the three
  conditions at every distance to 1e-6 eV, with a counter-check that they fail by more than 1 eV without the
  entries; each class of configuration shifted by its own entry, gas phase and periodic; weight independence with
  a counter-check; refusal of other option sets; charged scans without steps; the pair alone in a periodic cell).
- **Suites:** `tests/redox` 1082 passed, 3 xfailed (was 1077 passed, 3 xfailed: +5 in `test_redox_two_site.py`, now 42 tests and one strict xfail); the rest 1045 passed, 4 skipped, unchanged (Mac, 2026-11-14 08:15 IST).

## Round 43 (2026-11-14, branch `absolute-frame-2026-11-13`): SPEC 40 after its review: not ready for training; six fixes, the rest recorded

An independent review (four readers, eight skeptics; `experiments/benchmarks_4g/runs/two_site_2026-11-14/review_spec40.json`;
README, "SPEC 40 reviewed") confirmed six of its eight leading findings and two in part. **Still off by default;
nothing is trained with it; and it is no longer proposed for training as built.**

- **The finding that decides (recorded, not fixed; SPEC 40.6 iii; `probe_au2_ea_ip.py`).** The term corrects one of
  the pair's four levels. With it and no learned shift, gas-phase Au₂ has its ionic gap right and its electron
  affinity 0.8–1.7 eV too low and ionisation energy 1.0–1.2 eV too high (with the hop alone: 0.2–0.7 and 0.1–0.4 eV);
  no one-body learned shift can repair I = E(a⁺b⁻) + E(a⁰b⁰) − E(a⁺b⁰) − E(a⁻b⁰), which the term moves by exactly −ρ.
  The one-carrier entries need the same first-principles construction first.
- **Fixed, failing tests first (8 red, then green):**
  1. `two_site` requires `diabatic_selection='amplitude'` (the 'window' selection does not see the hop: H₂ never
     hopped and Au₂'s energy stepped by 1.6–2.0 eV) and must be a boolean (`RedoxConfig.__post_init__`).
  2. An s¹ atom of an applied pair meeting its own periodic image within 6 Å is refused also when it is the only one
     in the cell (`TwoSite.relaxation_pairs`; it was skipped silently, and a test asserted the skip).
  3. H–H is tabulated but no longer applied by the module (`"applied"` in `two_site.json`, `TwoSite.applied`,
     `is_applied`): the rule acts on element pairs, and hydrogens inside a molecule are not a bare H₂.
  4. The pre-registered weight rule is applied to the singlet root too (`collect_two_site.py`): three hop points
     (H₂ 0.6 Å; Au₂ 2.6, 2.8 Å; singlet weights 0.47, 0.45, 0.36) are flagged, kept in the file, not used. The hop
     changes by at most 2.1 % (Au–Au at 3.0 Å). The superseded file is kept in
     `experiments/benchmarks_4g/runs/general_laws_2026-11-13/two_site_before_review_superseded.json`.
  5. `TwoSite.replace_hops` refuses a coupling window that starts inside an applied pair's hop points.
  6. `TwoSite` validates the table when it reads it, and its kept relaxation knots follow the frame table in use.
- **Tests:** `test_redox_two_site.py` 24 → 37 and one strict xfail (the window and the switch written out; the
  periodic cell entry by entry in both spaces; which pair gets which curve; the energy scan without steps; the
  refusals; the record's numbers; the d-move xfail); `test_redox_cluster_signs.py` 14 → 30 and two strict xfails
  (the measured errors at two electrons, half filling and two holes pinned to 1e-6 t; below every exact level at
  half filling); `test_redox_frame_pair.py` 46 → 47 (the slab's pair list against an independent enumeration; the
  hydrogen pair against its first-principles values written out; the point count 74). Ten mutants that survived
  the earlier tests are killed.
- **Recorded, not fixed (SPEC 40.6):** (ix) one coupling per atom pair, so the s–s hop is also given to the twelve
  d-electron moves of an Au or Ag pair (strict xfail; no such configuration is selected on 80 test structures at
  frozen weights); (x) the mapping is exact for exact energies and is applied to TDA-PBE roots; (xi) the curves are
  C¹ at the points; (vii) rewritten with its exceptions and with the half-filled sector, where the configuration
  model lies below the exact ground state.
- **Statements replaced (kept here under the project's never-delete policy; the whole of SPEC 40 as written on 2026-11-13 is kept in
  `experiments/benchmarks_4g/runs/two_site_2026-11-14/SPEC_40_as_written_2026-11-13.md`):**
  1. two_site.py's docstring *"the relaxation of two neutral atoms … belongs to the covalent configuration alone"*:
     the one-carrier pairs lack a relaxation of the same size (SPEC 40.6 iii).
  2. SPEC 40.6 (iii) *"The ionic gap is not anchored"* as the only stated limit on the levels: extended as above.
  3. SPEC 40.4 *"An atom of a tabulated element whose frame is not the neutral atom is refused"* as a condition:
     with formal frames it cannot occur (the frame of H, Ag, Au is the neutral atom everywhere); the condition is
     the element pair, and H–H is withheld for that reason.
  4. SPEC 40.6 (vii) and Round 42 *"not for two carriers on three or more sites (two electrons on a
     triangle: +1.06 t)"* and *"On the Au–MgO slabs the alternative singlet rule changes no energy (0.00 meV on 32
     structures)"* as evidence: the statement had exceptions (two holes on the triangle lie below the exact
     singlet, the tetrahedron's are exact) and left out half filling; the slab comparison could not have failed
     (the selected configuration sets are trees).
  5. SPEC 40.0 and the README *"the ten compressed dimers carry 93–95 %"*: the five undoped test dimers below 2.18 Å
     carry 92.2 % and 94.2 % (seeds 42, 43).
  6. SPEC 40.2 *"t / Harrison runs 0.71–0.77 … 0.37–0.60"*: 0.71–0.74 (Au₂) and 0.43–0.60 (H₂) over the points that
     pass the weight rule.
  7. Round 42 *"19"* and *"24"* tests of `test_redox_two_site.py`: 37 and one strict xfail.
  8. docs/electrostatics.md and the project notes, the recommendation that the training step with SPEC 39 + 40
     was a proposal awaiting a decision: withdrawn until the one-carrier entries exist.
- **Suites:** `tests/redox` 1077 passed, 3 xfailed (the d-move limit and the two sign limits; was 1047 passed, 2 xfailed: +13 `test_redox_two_site.py`, +16 `test_redox_cluster_signs.py`, +1 `test_redox_frame_pair.py`); the rest 1045 passed, 4 skipped, unchanged (Mac, 2026-11-14 02:59 IST).

## Round 42 (2026-11-13 evening, branch `absolute-frame-2026-11-13`): the two-site sector of two s¹ frames from first principles (trace_redox SPEC 40); ten pairs in SPEC 39's table; the pair-list fix

The other half of Round 41, under the same rule (general equations only). **Off by default; no model
is trained with it; a model without the key is unchanged** (the two-site table is not touched when the flag is off,
tested).

- **New:** `trace_redox/two_site.py` (`TwoSite`, `two_site()`), `trace_redox/data/two_site.json` (Au–Au, Ag–Ag, H–H),
  `RedoxConfig.two_site` (default False; needs `frame_pair`, `spin_pairing`, `couplings`), two hooks in the module's
  formal-anchor route (the hop after the learned coupling is evaluated, the relaxation in the state-pair slot),
  `tools/electrostatics/pair_no_hop_curve.py`, `pair_triplet_character.py`, `collect_two_site.py`,
  `tests/redox/test_redox_two_site.py` (24), `tests/redox/test_redox_cluster_signs.py` (14 and two strict xfails),
  `tools/electrostatics/hubbard_cluster_signs.py`, `hubbard_cluster_signs_package.py`,
  `experiments/benchmarks_4g/probe_auau_two_site.py`, `probe_sign_rule.py`.
- **The equations (SPEC 40).** With the pair's ground-state interaction E_gs and its vertical σσ* excitations
  E_T, E_S (PBE / def2-TZVP, TDA): E_cov = E_gs + E_T, U = E_S − E_T, t = √(E_T E_S)/2 (ESTABLISHED mapping).
  NEW-TO-TRACE: O_ab(f_a, f_b; R) = −s(R) ρ_ab(R) with ρ = Φ_ab (SPEC 39) − E_cov on the frame–frame entry of the
  state-pair slot, and τ_ab(R) = −t_ab(R) w(R) in place of the learned coupling of the pair; ln ρ and ln t are
  monotone cubics through the points, linear beyond. No form assumed, no constant fitted.
- **Measured before building (README, 2026-11-13):** the hop is not Harrison's form with one constant, and the
  constant is not universal (t / Harrison: Au₂ 0.71–0.77, Ag₂ 0.68–0.88, H₂ 0.37–0.60); the unrestricted SCF triplet
  of Au₂ is a d-hole state below 2.6 Å and is not the no-hop level.
- **Mechanism check, pre-registered, frozen on the final step 33 model of seed 43 (not a verdict on accuracy):**
  the charge module's Au–Au bond force on test 167's scan is repulsive and monotone from 2.00 to 2.40 Å with the
  three options on (+12.5 eV/Å at 2.089 Å; DFT +15.25; as trained −18.8); the covalent–ionic element keeps one sign.
  SPEC 39 alone leaves the learned coupling's zero in place (+3.8 eV/Å at 2.089 Å, non-monotone).
- **SPEC 39's table: six → ten pairs with first-principles points** (Na⁺–Cl⁻, Cl⁻–Cl⁻, Ag⁰–Ag⁰, H⁰–H⁰ added; the
  series and the six earlier pairs' ratios are bit-identical, checked array by array; the superseded files are kept
  in `experiments/benchmarks_4g/runs/general_laws_2026-11-13/`). **Builder rule added:** a point enters only where
  the electron-gas term itself is above 5e-3 eV (H⁰–H⁰: it passes through zero near 2.8 Å and the ratio there is 18
  and −6); no earlier pair is affected.
- **Fixed:** `penetration.pairs_within` scans the lattice shifts around the minimum-image difference, so unwrapped
  positions lose no pair (the review's inherited defect, SPEC 39.6 ix). The fix changes no pair in any of the 700
  Au–MgO train and test structures (checked against the old function), so no trained model and no score moves.
- **Statements replaced (kept here under the project's never-delete policy):**
  1. SPEC 39.6 (iii) *"39 of 45 pairs carry c = 1"* and *"H, C and Ag pairs need their first-principles points
     before the term is used on them"*: 35 of 45; H–H and Ag–Ag have them; C pairs and the mixed pairs do not.
  2. SPEC 39.6 (vii) *"The hop law is the other half and is not built"*: built for two s¹ frames as SPEC 40.
  3. SPEC 39.6 (ix) *"Inherited, not fixed"* and Round 41's *"A strict xfail test records it; the fix waits until
     step 33 is scored"*: fixed; the test passes (`test_redox_frame_pair.py`: 46 tests, no xfail).
  4. Round 41's *"six with first-principles points"* and *"45 and one strict xfail"*: ten; 46.
  5. An open point of the project notes *"0.70–0.74 of Harrison's value, falling as d^−1.7"*: superseded by the tabulated
     t(R); the ratio runs 0.71–0.77 once 2.0–2.2 Å are included.
  6. My statement that the 'global' space with the 'phi1' band is exactly the trained single model on an undoped
     cell: measured on one scan of the Mac model only; on the final seed-43 model the two routes differ by 35–67 meV
     (README, 2026-11-13 evening).
- **Measured, no rule proposed:** the configuration Hamiltonian's signs (Jordan–Wigner, spin-less local orbitals)
  are exact for one carrier of either kind on any cluster and on two sites at any filling, and not for two
  carriers on three or more sites (two electrons on a triangle: +1.06 t above the exact singlet). On the Au–MgO
  slabs the alternative "singlet rule" changes no energy (0.00 meV on 32 structures).
- **Not built:** the ionic gap's short-range anchor (U of gas-phase Au₂ missed by −0.46 to −0.95 eV at 2.2–2.0 Å);
  pairs of two different s¹ atoms (the three-level mapping needs a double excitation); the hop law for any pair
  that is not two s¹ frames; the hydrogen pair's overlap tail beyond 2.2 Å (0.045 eV low at 2.6 and 3.0 Å).
- **Suites (Mac, 2026-11-13 20:44 IST):** `tests/redox` 1047 passed, 2 xfailed (the two recorded sign limits; was 1008
  passed, 1 xfailed: +1 the former xfail, +24 `test_redox_two_site.py`, +14 `test_redox_cluster_signs.py`); the rest
  1045 passed, 4 skipped, unchanged.

## Round 41 (2026-11-13, branch `absolute-frame-2026-11-13`): the frame pair's own interaction for every element pair (trace_redox SPEC 39)

By the decision of 2026-11-13 (general equations only, no update aimed at one system). **Off by default; no model is
trained with it; nothing else changes** (a model without the key is bit-identical, tested).

- **New:** `trace_redox/frame_pair.py` (`FramePair`, `frame_pair()`, `IN_SITU_SITES`), `RedoxConfig.frame_pair`
  (default False; needs `carrier_penetration` and `carrier_overlap='gordon_kim'`; refused with `frame_wall` and with
  `carriers='covalent_units'`), the module's hook after the ensemble's free energy with the diagnostic
  `_frame_pair_last`, `trace_redox/data/frame_pair.npz` (45 element pairs; six with first-principles points, stored
  in the file) and its builder `tools/electrostatics/build_frame_pair.py` (numpy only),
  `tests/redox/test_redox_frame_pair.py` (45 and one strict xfail).
- **The equation:** Φ_ab(R) = s(R) { D_ff(R) + f_a f_b [K/R − J^G_ab(R)] + c_ab(R) P_ab(R) } over every pair of
  atoms: the exact electrostatics of the frozen frame densities beyond the model's Gaussian kernel, and their
  non-additive Thomas–Fermi kinetic + Dirac exchange energy with c_ab(R) = 1 or the first-principles ratio through
  the computed points. SPEC 23 and SPEC 25 made absolute.
- **Replaced statement (kept in `trace_redox/wall.py`, legacy):** the one-pair wall of Round 31, "K_TF T_TF^nad … one
  constant per pair", is a fit to Mg²⁺–Au⁰; measured on six pairs its constant spans 0.399–1.017 and it holds no
  electrostatics. `frame_wall` stays as built for the step 24 checkpoints.
- **Found on the way:** `trace_redox.radial.pair_electrostatic_energy` cancels four terms of order Z_a Z_b/R and is
  wrong by up to 0.05 eV for two heavy neutral atoms (Au⁰–Au⁰ at 2.331 Å: −6.3577 against PySCF's −6.3047). The
  builder uses a cancellation-free form (−6.3047), confirmed out of sample at 2.0, 2.1 and 2.2 Å to 4e-4 eV. The function
  is unchanged and is used by no table of the library (SPEC 23's use differences); the Pauli build's check
  `elst_vs_library` failed for Au–Au for this reason.
- **Corrected the same day, after two new first-principles points and a review by four readers and six verifiers**
  (`experiments/benchmarks_4g/runs/general_laws_2026-11-13/review_spec39.json`; 37 checks found sound, six findings
  verified, five confirmed). Statements of the first build that are replaced:
  1. *"c_ab = 1, or the least-squares ratio …" (one constant per pair).* The ratio drifts with distance (Au⁰–Au⁰ 0.843
     → 0.880 from 2.68 to 2.10 Å; Au⁰–O²⁻ 0.842 → 0.690 from 2.0 to 4.0 Å), which left the whole interaction 10.7 % low
     at 2.1 Å (Au⁰–Au⁰) and 13–65 % high at 2.26–2.6 Å (Au⁰–O²⁻). Now c_ab(R): a monotone cubic through the
     first-principles ratios, applied at run time (`FramePair._ratio`). Its ends were first held constant with zero
     slope; the Au⁰–Au⁰ point at 2.0 Å, computed afterwards, showed the table 5.3 % low there, and the zero slope put
     a bump of several eV/Å into the force between the first two points. Replaced by the shape-preserving end slope
     with a bounded continuation, c_end + m_end w tanh((R − R_end)/w), w = 0.3 Å; the Au⁰–Au⁰ force is now monotone
     (tested). Au⁰–Au⁰ has seven points, 2.00–2.68 Å.
  2. *"The pairs SPEC 33's wall holds are skipped while it is on."* The wall is the lone atom's interaction with its
     whole site, so the skip counted the four Mg²⁺–Au⁰ pairs twice on the O-top site (+0.34 eV at 2.26 Å, +0.94 eV at
     1.8 Å) and dropped Au⁰–O²⁻ beyond 4.3 Å. Now no pair is skipped and the site's pair sum is taken out with the
     wall's own switch (`FramePair.site_reference`); on the measured site the frame terms add up to SPEC 33's table.
  3. *"(iii) The spin state is … the high-spin product of two open shells."* True of the first-principles points
     only; for a pair without points P is the spin-averaged electron gas (H⁰–H⁰: 2.60 against 4.67 eV). Stated as a
     limit; H, C and Ag pairs need their points first.
  4. *"The table's series converges to 3e-4 eV for Au⁰–Au⁰ at 1.8–4 Å (2e-3 at 1.2–1.8 Å)."* The file's own check
     values gave 3.2e-3 at 1.21 Å and 8.7e-4 at 1.88 Å on a fine grid: the electrostatic integral was rough at
     4e-4 eV rms from linear interpolation and an unsplit kink. With cubic Hermite interpolation from exact
     derivatives and the integral split at the kink the worst error in reach is 7e-6 eV.
  5. *"16 % high to 28 % low"* for c = 1 on the measured pairs: 21 % high to 28 % low.
  6. Tests: the first-principles test left out Au⁰–O²⁻ and compared with fitted constants; three checks compared the
     builder with itself; the module was tested on one open cluster; rounding tolerances were hand-set. Rewritten
     (independent quadrature and direct-form checks, every stored point, periodic cell, mesh, cluster space with
     the Fermi band, the strain derivative, the gate, tolerances from machine epsilon and the operation count).
- **Inherited defect found by the review, not fixed here:** `penetration.pairs_within` loses a pair once two atoms
  are more than about one cell apart in unwrapped coordinates (wrapped inputs, which every benchmark structure is,
  are unaffected). A strict xfail test records it; the fix waits until step 33 is scored, because that scoring
  imports this tree.
- **Not built:** the hop law (a coupling that keeps its sign and grows on compression); the rule for pairs inside a
  covalent unit; first-principles points for 39 of the 45 pairs. Records: `experiments/benchmarks_4g/README.md`,
  "General physics only" and after.

## Round 40 (2026-11-12, branch `crystal-factor-2026-11-12`): the crystal factor acts on ions only (SPEC 35.1)

- **New field `RedoxConfig.frame_polarizability_scope`** (`'ions'`, **the default changes the statement**; `'all'` is
  SPEC 35 as first built, legacy, kept): the learned crystal factor g of the frame-polarizability anchor multiplies
  the charged frame ions only; a neutral frame atom (Au⁰) keeps its tabulated free-atom polarizability.
  `trace_redox/polarizability_anchor.py` (`anchored_params(..., scope)`, `AnchoredHeads(..., scope)`),
  `trace_redox/module.py` (`LEGACY_FIELD_VALUES`, the warning in `from_checkpoint`, `_heads_for`).
  **Replaced statement (SPEC 35):** "one learned crystal factor g" for every tabulated species → for the charged
  ones. **Older checkpoints keep the statement they trained** (`'all'`, with a warning when the anchor is on).
- **Why.** The records said "g never learns". Measured (`experiments/benchmarks_phonons/README.md`,
  `probe_alpha_scale_gradient.py`): g had converged at 0.997, where the ε∞ prior's gradient on log g (+0.046328)
  cancels the Au-atom polarizability prior's (−0.046347); the forces are 800 times weaker. Frozen copies with g
  set by hand to 0.88–0.95 pass 7/7 of the phonon criteria (5/7 as trained); at 0.90 LO is 693.4 against DFT's
  693.9 cm⁻¹.
- **Data, no physics:** `experiments/benchmarks_phonons/data/MgO_bulk_eps_prior_{pbesol,pbe}.xyz`
  (`make_mgo_eps_prior_dft.py`): the DFT ε∞ labels at the DFT lattice constants (decision of 2026-11-12); the
  measured-value prior of the slab runs is unchanged.
- **Cancelled by decision (branch made first):** the phonon retraining with the extensive forms, at
  epoch 13 of 40 on s43; it had matched the earlier run to six digits through epoch 11. Kept, not scored.
- **Tests:** `tests/redox/test_redox_polarizability_anchor.py` (7; 5 new, 6 of the 7 red before the code). Two of my
  new tests first assumed a graph and a response route that an isolated atom does not have; corrected in the test.

## Round 39 (2026-11-12, branch `extensivity-2026-11-10`): the operator guard follows the route (README R25)

- **`trace_redox.bound.assemble_neutral(..., elimination=None)`** (new keyword). `config.max_operator_bytes` bounds
  what the caller's elimination holds: with `elimination='condensed'` one [n, n_m] array (the incidence, the coupling
  A, K⁻¹Aᵀ; n_m = n_bonds + 3n, + 5n with quadrupoles) instead of the dense n_m × n_m operator K, which
  `_CondensedK` never forms. The quadrupole pair tensors keep the bound `trace_electrostatics.energy.assemble`
  states for them. `None` (every caller that does not name the route: probes, tools, older tests) and `'dense'`
  keep the legacy guard. `RedoxElectrostatics` passes its `bound_elimination`. `trace_electrostatics` is unchanged.
- **Why.** The 990-atom Au₂/MgO supercell (the (3, 3, 1) mesh, the reference DFT's sampling) was refused for a
  10.25 GB K that the route does not build; with the limit raised by hand it ran in 127–131 s (README M1d).
  **Replaced statement** (README R15 on, SPEC §37, Round 38): "(3, 3, 1) is refused by the dense
  route" → it runs; (2,2,1) → (3,3,1) moves the undoped E(wetting) − E(non-wetting) by +2.388 / +3.364 meV.
- **No energy changes** (a guard, not arithmetic). A structure refused before and accepted now needs memory that
  no guard of this route measures: the pair tensors grow linearly with the atoms (peak resident memory 5.7–8.4 GB
  at 990 atoms, lower bounds).
- **Tests:** `tests/redox/test_redox_operator_guard.py` (5; 3 red before the change). Suites, 2026-11-12:
  `tests/redox` 949 passed; `pytest tests --ignore=tests/redox` 1045 passed, 4 skipped.
  After SPEC 36.10 (below): `tests/redox` **958 passed**; the rest **1045 passed, 4 skipped** (2026-11-12, 10:2x IST).
- **The response record of the cluster product (SPEC 36.10, NEW-TO-TRACE; lifts the second half of SPEC 36's L5).**
  `RedoxResult.ensemble` on several clusters holds `'terms'`, one per free energy of
  F = E⁰ + Σ_c (F_c − E⁰) + Σ_{c<c′} (F_cc′ − F_c − F_c′ + E⁰), with the weights of that combination;
  `trace_redox.response.record_polarizability` (new) reads either form, and `train.py`'s response priors call it.
  At a mesh (SPEC 37) the record is per cell (`'scale'`). `ClusterEnsemble` keeps the pair spaces' Hamiltonians and
  configurations. **Replaced statement** (SPEC 36.8 L5): "Refused … with §30's response priors on more than one
  cluster" → built; the refusal with §32 stays. Why now: a retraining in the clusters space on step 31's protocol
  (two response priors) would have stopped whenever a prior structure reached two clusters, as s42's bulk MgO cell
  does (README R26). Tests: `tests/redox/test_redox_cluster_response.py` (8),
  `test_redox_polarizability_prior.py::test_two_clusters_train_on_the_whole_response_in_the_clusters_space`.
  Three of my test statements were wrong as first written and were corrected against measurement, not the code:
  a gradient through the couplings where no coupling enters the Hamiltonian; a bookkeeping assertion on which
  cases hold couplings; a 1e-9 tolerance not derived from the finite-difference operation.
- **Measured the same day, no package change (README R26; corrected the same day: the defect itself was recorded on
  2026-11-06 for step 20's s42 (theory note 2026-10-18 §4e), and R26 repeats it on step 31's weights):** step 31's s42 holds O–O couplings of up to +17.7 eV
  (6.5× SPEC 27's ceiling) on 576 edges of every benchmark cell; SPEC 27's bound was off. **Replaced statement**
  (SPEC §0 table): "§27 coupling bound: off: the identity on every benchmark structure" → the identity on s43,
  not on s42.

## Round 38 (2026-11-10 / 11, branch `extensivity-2026-11-10`): extensivity of the redox model (SPEC 28.6, 36, 36.7, 37, 38)

The defect (internal review, 2026-11-10): replicating a periodic cell changed the energy per cell and the forces of
the full redox model. Four terms were located by measurement and each got its own pre-registered step
(`experiments/benchmarks_4g/README.md`, "The extensivity line", R12–R23). Nothing was trained with any of them; all
numbers below are on step 31's frozen weights. Nothing is committed.

- **New field `RedoxConfig.state_polarizability_stiffness`** (`'onsite'`, **the default changes the statement**;
  `'block'` is the legacy form, kept). SPEC 28's charge-state scaling acts on the atom's own stiffness A_μ only, not
  on the whole diagonal block of K, which in a periodic cell also holds the atom's interaction with its own images
  (SPEC 28.6; `trace_redox/polarizability.py`). Measured, s43: the in-phase SPEC 28 term misses additivity by
  7.33e-07 meV at 2×1 (R12 located it; X1's criterion ≤ 1e-3 meV).
  **Replaced statement (SPEC 28):** "Δ_j = P_jj(1/ρ_j − 1)", with P_jj the bound operator's diagonal block
  (= A_μ,j + T2_jj under periodic boundaries) → Δ_j = A_μ,j (1/ρ_j(n) − 1). Identical in an open system.
  **Older checkpoints keep the statement they trained:** `RedoxConfig.from_checkpoint` with `LEGACY_FIELD_VALUES`
  restores `'block'` for a checkpoint that predates the field and warns (`flashace/checkpoint.py`); `train.py` warns
  on resume through `statement_changes` when the run's redox config states a different energy than the checkpoint's.
  s43 loaded without override returns its stored Table II (+920.36 / −75.73 / −996.09 meV).
- **New value `RedoxConfig.diabatic_space='clusters'`** (default `'global'`, unchanged; SPEC 36, NEW-TO-TRACE): the
  configuration ensemble as a product over clusters of linked units,
  F = E⁰ + Σ_c (F_c − E⁰) + Σ_{c<c′} (F_cc′ − F_c − F_c′ + E⁰), an increment expansion (ESTABLISHED as a form:
  Stoll, Phys. Rev. B 46, 6700 (1992); Paulus, Phys. Rep. 428, 1 (2006); records and abstracts only) of the
  ensemble's free energy. `trace_redox/clusters.py` (new). Measured, s43 undoped 2×2: E/cell − E(1×1) = +2.79 /
  −38.03 meV (wetting / non-wetting) against +220.84 / +99.83 in the global space.
- **New field `RedoxConfig.cluster_closure`** (True; `False` is SPEC 36 as first built, legacy, kept; SPEC 36.7): a
  merged cluster holds the products of its disjoint parts, and parts with no coupling between their supports are
  separate clusters. Without it a merge cost +0.790 eV on two model dimers and +1.243 eV on the doped 2×1 cell
  (R14). A closure beyond `fragment_max_configurations` is refused with the linked and the closed counts, never
  truncated. The closure is exponential in a cluster's disjoint parts.
- **New fields `RedoxConfig.bvk_mesh`** ((1, 1, 1)) **and `bvk_length`** (0.0; M_d = ceil(L / w_d)) (SPEC 37,
  NEW-TO-TRACE in this use; the boundary condition is Born and von Kármán's, the mesh Monkhorst and Pack's,
  Phys. Rev. B 13, 5188 (1976)): E_M(C) := E_Γ(S_M[C]) / |M|, the fluctuations sampled in the supercell the mesh
  names. `trace_redox/bvk.py` (new; `mesh_for`, `tile`), `RedoxResult.bvk_mesh`, `_forward_cell` / `_forward_bvk`.
  With `bvk_length = 18.0` Å, s43: E(2×2)/4 − E(1×1) = +1e-6 / < 1e-6 meV, max|ΔF| 5.6e-9 / 3.4e-8 eV/Å (undoped).
- **No 2^k percolation enumeration for `fragment_ensemble='diabatic'`** (C5; `_forward_cell`): the diabatic ensemble
  does not use the partitions; `chemical_bonds` and its coverage check still run, and `'exact'` / `'two-level'`
  still refuse. The 1×1 energies are unchanged at the 9 printed decimals.
- **New value `RedoxConfig.band_form='fermi'`** (default `'phi1'`, unchanged; SPEC 38, NEW-TO-TRACE): the band unit
  holds any number of electrons, E(n, N) = E_at(n) + F_b(N) + Σ_v ρ_N(v) c_v(n), with F_b the finite-temperature
  mean-field free energy of N electrons in the band's sites (Mermin, Phys. Rev. 137, A1441 (1965), for the
  functional; Goedecker and Colombo, Phys. Rev. Lett. 73, 122 (1994), for the Chebyshev expansion of the Fermi
  operator; records and abstracts only). `trace_redox/fermi.py` (new; `fermi_operator`, `band_free_energy`, the
  value from detached eigenvalues with an analytic gradient, two Newton steps for μ), `band.py` `augment_fermi` and
  `BandCharges` (the unit's charge mapped onto atoms, used by `response.py`), `FreeChargeSolution.expectation`
  (`free_solver.py`). `RedoxResult.omega` is None in this form.
- **Measured on the benchmark, frozen weights** (Table II: DFT wet_u +928.9, wet_d −2.7, T5 −931.6 meV; 4G +975.4,
  −41.0, −1016.4). C4 stage 2: U1 PASS (the two band forms agree to +1.717e-06 / +1.641e-06 eV on the doped 1×1
  cells at Γ); U2 FAIL as written (1 of 12 force components within 1e-6 relative of central differences; the causes are the 2.26 Å knot
  of SPEC 33's C¹ wall table, the reference step of R22 and the stencil's resolution); U3 PASS (doped cells,
  `bvk_length = 18.0` Å: E(2×2)/4 − E(1×1) = +0.000003 / +0.000007 meV, max|ΔF| 6.52e-08 / 6.19e-08 eV/Å,
  reproduced in a rerun); U4 runs; U5 PASS (suites below).

  | Table II, meV | wet_u | wet_d | T5 |
  |---|---|---|---|
  | s43 as trained (`'block'`, global, `'phi1'`, Γ) | +920.36 | −75.73 | −996.09 |
  | s43 `'onsite'`, clusters, `'phi1'`, Γ | +892.68 | −60.70 | −953.38 |
  | s43 `'fermi'`, Γ | +892.65 | −60.70 | −953.35 |
  | s43 `'fermi'`, mesh (2, 2, 1) | +935.11 | −22.95 | −958.06 |
  | s42 `'fermi'`, Γ | +832.15 | −164.68 | −996.83 |
  | s42 `'fermi'`, mesh (2, 2, 1) | +890.85 | refused at the cap of 512; +603.69 at 2048 | −287.16 (cap 2048) |

- **Limits, measured (SPEC 36.8, 38.4; README R22 addendum, R23).** (1) Extensive is not the same as trained: the
  doped number at a mesh is not fixed by weights trained at Γ on one cell; the two seeds differ by 104 meV at Γ and
  by 627 meV at the mesh, where s42's supercell band takes the dimers' electrons (2.50 / 2.84 e per cell against
  s43's 2.00). No doped number at a mesh is claimed. (2) The closure meets its cap on s42's non-wetting supercell
  (127 linked rows close to 1218); a closure that is polynomial inside a merged cluster is not built. (3) The
  dropped three- and four-cluster terms are +0.867 meV per 1×1 cell on s43's doped non-wetting 2×2 cell
  (−0.907 meV on wet_d at the mesh, fixed geometry). (4) SPEC 21's switch is not variational (16.4 meV below the
  full space on a 6-atom random module). (5) Convergence beyond a 2×2 mesh is not measured. (6) One control value
  of U3's first record (+1791.512941 against +1791.5162 meV) was not reproduced and its cause is not found.
- **Replaced statements.** SPEC 36.8 L2: "On the benchmark cells the clusters hold 3–5 rows." → 2–5 rows on s43;
  on s42's doped 2×2 supercell clusters of 33, 6 and 6 rows, and 127 → 1218 refused. README R22: "No benchmark
  number moves … below every criterion" → superseded by the R22 addendum (kept as written above it).
- **Tests.** New modules in `tests/redox`: `test_redox_extensivity.py` (12), `test_redox_cluster_product.py` (33),
  `test_redox_cluster_closure.py` (7), `test_redox_bvk.py` (29), `test_redox_percolation_diabatic.py` (5),
  `test_redox_fermi_band.py` (21), `test_redox_fermi_band_unit.py` (19). Changed, each with its record in the
  README: `test_the_closed_form_holds_in_a_small_periodic_cell` is parametrized over the stiffness (`'block'` keeps
  the legacy closed form verbatim; the default missed it by 1.9e-4 eV); the reservoir test counts 6 rows per
  cluster with the closure and 5 without; C2b's Q5 test was replaced by
  `test_the_six_atom_module_against_the_full_space[3, 4, 5]` after it failed as written; the charged bulk case of
  `test_redox_bvk.py` runs on `cluster_closure=False` and a new test asserts the refusal of its closure.
  Suites, 2026-11-11 09:38–09:53 IST: `tests/redox` 944 passed; `pytest tests --ignore=tests/redox` 1045 passed,
  4 skipped.
- **Defaults.** Only `state_polarizability_stiffness` changes what a new model states. `diabatic_space`, `bvk_mesh`,
  `bvk_length` and `band_form` keep their earlier values; `cluster_closure=True` acts only in the clusters space.
- Probes (no physics): `experiments/benchmarks_4g/probe_ext_*.py`; outputs under
  `runs/extensivity_2026-11-10/` (gitignored). An internal review of 2026-11-10 named the defect (benchmarks README, "Internal
  review, 2026-11-10: measurements").
- **Measurements after the round was written (2026-11-11, no package change; README R24, M1, M1b, M1c, S1, S2).**
  - R24: the reference DFT sampled 3 × 3 × 1 k-points (Ko et al., Nat. Commun. 12, 398 (2021), Methods).
  - **Replaced statement, limit (5):** "Convergence beyond a 2×2 mesh is not measured" → the undoped pair is
    measured to four cells along each axis on both seeds (`probe_ext_mesh_convergence.py`, new option `c=`);
    (3, 3, 1) is refused by the dense route. The image term follows R15's tin-foil lattice sum
    (`probe_ext_mesh_lattice_fit.py`, new): the upright dimer's (1,1,1) → (2,2,1) step is reproduced to 0.6 % /
    0.1 % with one constant; M1c (vacuum period 26.4589 → 40.0 Å) is scored partial. Estimates, not measured:
    (2,2,1) → (3,3,1) +2.2 / +3.3 meV on E(wetting) − E(non-wetting).
  - **Replaced statement, README R15 part 4:** a square mesh keeps "12.5 %, 3.7 %, 1.6 %" of the in-plane dipole's
    image term → 14.3 %, 5.4 %, 2.9 % (the remainder is the Lorentz term −2π/V_M, ∝ M⁻²).
  - `probe_ext_vacuum_dipole.py` (new): the mean cell dipole of the upright dimer's cell is +0.517 / +0.806 e·Å
    (s43 / s42); its tin-foil term changes by +3.0 / +7.3 meV between the two vacuum periods and is no part of
    the extensivity defect.
  - S1 (`probe_speed_profile.py`, new): one energy-and-forces evaluation of the 110-atom cell takes 1.32 s as
    trained and 1.53 s with the extensive forms (4 CPU threads); the electrostatics is 77–78 %; the dense solves
    are 4 % of its forward; (2,2,1) takes 13.27 s. S2 (`probe_speed_dftb.py`, new; DFTB+ 25.1, GFN1-xTB, in the
    separate conda environment `dftb-timing`): 11.7 s on the single cell from scratch, 7.1 s with a restart,
    34.7 s on 3 × 3 × 1.
  - My errors, recorded in the README: M1's first design asked for a mesh R15 had already recorded as refused;
    M1b's prediction used the wrong lattice sum and its score withdrew R15's numbers wrongly (annotated in place);
    two one-thread timings ran on four threads (kept, renamed `INVALID_*`).
  - **2026-11-12, R25 and M1d. Replaced statement:** "(3, 3, 1) is refused by the dense route" → the guard
    (`max_operator_bytes`, in `trace_electrostatics.energy.assemble`) bounds a dense n_m × n_m K that
    `bound_elimination='condensed'` never builds; with the limit raised in the probe (`opbytes=`) the 990-atom
    supercell runs in 127–131 s. Measured: D23 = +2.388 / +3.364 meV (s43 / s42; predicted +2.230 / +3.329 before
    the run); M1's reading (a) is met. My error: I repeated "refused by the dense route" from R15 on without
    reading what the route allocates.

## Round 37 (2026-11-08, a local bookkeeping branch): SPEC 33 per charge state (step 27)

- **New field `RedoxConfig.bond_pauli_states`** (False, unchanged; needs `bond_pauli=True` and `anchor='formal'`): the
  bond-oriented Pauli law's coefficients and exponents follow the s atom's charge state — Au₂⁻'s measured law for n = −1
  (relaxed C₁⁻, C₂⁻, p₂⁻ = −0.1094, −0.0957 eV, 0.4561, fitted at 2.26 and 2.6 Å; frozen +0.1011, −0.2905 with the
  neutral's exponents), the neutral's otherwise — entered as T(n) − T(0) on the formal ladder; the frame path keeps the
  neutral law (SPEC 33.3). `trace_redox.bond_pauli`: `ANION_LEVELS`, `state_coefficients`, `state_exponents`,
  `geometric_sums`, `state_terms`, a shared `_geometry`; `energy` bitwise unchanged.
- **Replaced statement (README step 27):** "the frame-term path is then not applied" → the frame path keeps the neutral
  law and the ladder carries T(n) − T(0) (equal up to a constant per structure). Test 4's first form asserted a frame
  configuration its geometry cannot have (total charge 0, frame sum −2); restated on the configurations it has.
- `tools/electrostatics/fit_bond_pauli.py --anion`; tests `test_redox_bond_pauli_state.py` (4) green, the strict-xfail
  marker removed.
- Benchmark tooling (no physics): `compare_protocol.py` merges its shared JSON instead of overwriting it and writes a
  per-run copy; `experiments/benchmarks_phonons/phonons_model.py` scores the phonons of a TRACE model without
  electrostatics with `mgo_phonons.py`'s phonopy settings (checked to 0.0006 cm⁻¹ on a TRACE model).

## Round 36 (2026-11-07, branch `bulkaug-cancelled-2026-11-07`): per-structure force weights in training

- **`train.py`: `atoms.info['force_weight']`** (default 1) multiplies a structure's force-loss term in the batched and the
  per-structure train and validation paths (`force_loss(per_struct_sq, n_at, force_weight)`); the dataset item and the
  collate carry it; the metrics stay unweighted. Why, measured: the 136 displaced PBE MgO cells (RMS force 0.038 eV/Å)
  carry 0.37 % of the force loss against the 300 slabs (0.42 eV/Å) at equal relative error, which capped the cancelled
  bulk-augmented run's optical branches; weight 122.4 (the measured ratio of mean squares) gives them 31 %. Tests
  (`tests/test_train_force_weight.py`, 2); the plain-pipeline contract test gained the key. Suites: 1045 passed, 4 skipped.

## Round 35 (2026-11-07, branch `overlap-damping-2026-11-07`): the frame-polarizability anchor (SPEC 35)

- **New field `RedoxConfig.frame_polarizability_anchor`** (False, unchanged; needs `anchor='formal'`): the dipole
  stiffness of every frame ion with a reliably tabulated free-ion polarizability becomes A_μ = K_E ℓ²/(g α_Z(f)) I
  (`data/ion_polarizabilities.json`: O²⁻ 2.44, Mg²⁺ 0.074, Au⁰ 5.41, Al³⁺ 0.039 Å³), one learned crystal factor
  g = exp(`log_alpha_scale`) (init 1), applied at the heads' readout before assembly in both routes
  (`trace_redox/polarizability_anchor.py`). Why, measured: step 20's split is inverted (α_O 0.2, α_Mg 1.9 Å³), which
  ε∞ cannot see and which turns SPEC 34 destructive on surfaces; with the physical split SPEC 34 is benign on the slabs
  and Z*(Mg) = 1.93 at ε∞ 3.235. Tests (`tests/redox/test_redox_polarizability_anchor.py`, 2).

## Round 34 (2026-11-07, branch `overlap-damping-2026-11-07`): the frame-pair overlap damping of induction (SPEC 34)

- **New field `RedoxConfig.frame_induction_damping`** (False, unchanged; needs `anchor='formal'`): the frame charge's
  drive on the bound dipoles of its tabulated nearest-neighbour counter-ions is multiplied by s(R) = 0.61 + 0.37 (R −
  2.128 Å) (Mg²⁺–O²⁻; clamped to [0.2, 1], C²-switched to 1 over 2.6–3.0 Å), built as SPEC 26's correction column on
  the frame: v = ΔAᵀf, y0 − K⁻¹v, u0 − A K⁻¹v, constant + v·y0 − ½ v·K⁻¹v (`trace_redox/frame_damping.py`). K, A, the
  bound state and ε∞ are untouched. Why, measured: step 20's bulk MgO Born charge 2.43 against PBE 1.99 is the Lorentz
  enhancement of the smeared induced dipoles, which no width, species split, short-range pair term or ensemble route
  moves (frozen scans, 2026-11-07); the embedded ALMO pair measured the O²⁻ response to its cation damped to 0.53–0.68
  of the point-field value with slope +0.37/Å — the shell model's shell–shell coupling in first-principles form. Frozen
  on step 20: Z*(Mg) 2.429 → 1.946 at ε∞ 3.009 unchanged (pre-registered 1.90–2.10: pass); the slope is the mechanism
  (no slope 2.68, double slope 1.17). Tests (`tests/redox/test_redox_frame_damping.py`, 3).
- Probes: `experiments/benchmarks_phonons/probe_{ensemble_born,sigma_scan,dipole_width,overlap_damping}.py`;
  `tools/electrostatics/deformation_dipole.py` (the embedded pair's deformation, polarisation and charge-transfer
  dipoles against R).

## Round 33 (2026-11-06, branch `phonon-benchmark-2026-11-06`): phonons against DFT, and forces-only structures in training

- **`train.py`: structures without an energy label train on forces alone.** `energy_target_from_atoms` returns the
  energy and `has_energy` (False when the structure carries no energy, or declares `has_energy=F` in its extxyz header);
  the dataset item and the collate carry the flag; `energy_loss(E, t, N, has_energy)` is Eq. (37)'s energy term over the
  labelled structures only, used by the batched and the per-structure train and validation paths; `MetricTracker`
  excludes unlabelled structures from the energy statistics (`n_atoms_e`) and keeps their forces; the composition
  baseline fit skips them. Why: Togo's displaced PBE MgO supercells (NIMS MDR, CC BY 4.0) carry VASP forces and no
  energies, and are the bulk data of the phonon line. Tests (`tests/test_train_energy_mask.py`, 3, written red first);
  the plain-pipeline contract test gained the key.
- **A phonon benchmark line** (`experiments/benchmarks_phonons/`, pre-registered P0–P6; `tools/electrostatics/
  mgo_phonons.py`): Z* of TRACE-Redox under PBC as the frame charge plus the bound sector's ∂P/∂r (the module refuses a
  field under PBC for free charges), ε∞, phonopy finite displacements with and without the NAC from the model's own Z*
  and ε∞, an ensemble guard on every displaced supercell. No package physics changed. Measured on step 20 (theory note
  §4e): ε∞ right, Z* +22–27 % (no dynamic covalent transfer), LO +20 %; the learned O–O hole coupling
  (28 eV/edge) collapses the 64-atom bulk ensemble under 0.5 % compression.

## Round 32 (2026-11-05, branch `bond-pauli-2026-11-05`): the bond-oriented Pauli repulsion (SPEC 33)

- **New fields `RedoxConfig.bond_pauli`** (False, unchanged) **and `bond_pauli_level`** ('almo'): adds
  E_bp = Σ_a Σ_j λ_a s_c(R) [D₁(R) cos θ + D₂(R) P₂(cos θ)] over the bonded s-class atoms a and the frame anions j within
  4.3 Å (`trace_redox/bond_pauli.py`), cos θ the angle between the atom's bond direction and the anion, D₁ = C₁ f^p₁,
  D₂ = C₂ f^p₂, f = E1(R)/E1(2.26 Å) with E1 the lone Au⁰'s frozen repulsion on the O-top site (tabulated 2.26–2.9 Å);
  the constants fitted jointly over R and θ = 0–70° to the module's two-atom form: frozen C₁, C₂ = −0.072, −0.665 eV,
  p₁, p₂ = 2.29, 0.500 (four R, 3.7 meV); relaxed (ALMO, the default) −0.069, −0.497 eV, p₂ = 0.382 (three R, 6.2 meV;
  p₁ held at 1.877); E1(R) tabulated at 1.8–3.8 Å, C¹ monotone cubic Hermite interpolants. Tabulated for (Au, O); an untabulated bonded pair
  within the cutoff is refused. A frame term, configuration independent, trained with.
- **New field `RedoxConfig.bond_pauli_wall`** (False; needs `bond_pauli`): the wall part of SPEC 33, E_wall = Σ s_c(R)
  E1(R) over every s-class atom and frame anion, I(R) the lone Au⁰'s interaction with the O-top site at the law's level
  (frozen E1 for 'frz', relaxed I_ALMO for 'almo': the law's reference at that level, so wall + law is the dimer's
  interaction there). Why: the Au⁰–O²⁻ frozen interaction was the backbone's alone (SPEC 23/25 frame-relative, the Round-31 wall
  Mg–Au only); on frozen step-20 weights the law's θ = 0 shielding tipped the upright dimer over the backbone's cliff
  below 2.0 Å. Tests (`test_redox_bond_pauli_wall.py`, 5).
- **Why, measured (theory note §4c, 2026-11-03/04):** the flat Au₂ of the T5 protocol is a PBE minimum that every
  model of steps 21–24 rolled out of, into a tilted O-top state PBE puts 0.8 eV above the upright dimer and the models
  0.5–1.1 eV too low. ALMO decomposition: the frozen repulsion carries +1.40 eV of it and is not pairwise in atomic
  densities (the σ bond shields the upright contact atom by 0.76 eV); the law of (R, θ) fitted to ≤ 8 meV at four R.
- Tests (`test_redox_bond_pauli.py`, 7): the measured rotations and the radial law, zero without a bond / anion / within
  reach, forces and double backward with images, periodic translation invariance, refusals and defaults, the module adds
  exactly the term.
- Probes and tools: `probe_tilt_path.py`, `probe_rotation_path.py`, `probe_tilt_terms.py`, `probe_tilt_upright_terms.py`,
  `probe_tilt_fold.py`, `probe_bop_frozen.py`; `tools/electrostatics/tilt_scan.py`, `almo_tilt.py`, `almo_tilt_frz.py`,
  `frz_pair_sum.py`, `frz_nonadditivity.py`, `frz_angular.py`, `almo_angular.py`, `bop_flat_check.py`.

## Round 31 (2026-10-31, branch `frame-wall-2026-10-31`): the frame pair's own Pauli wall

- **New field `RedoxConfig.frame_wall`** (False, unchanged): adds E_w = Σ K_TF T_TF^nad[ρ_i^f, ρ_j^f](R_ij) over the frame
  pairs of the tabulated element pairs (`trace_redox/wall.py`, `trace_redox/data/frame_wall.npz`, built by
  `tools/electrostatics/build_frame_wall.py`): the non-additive Thomas–Fermi kinetic energy of the frozen frame
  densities of the SPEC 25 library times one constant fitted to the frame pair's own frozen-density PBE
  non-electrostatic interaction (Mg²⁺–Au⁰: K_TF = 0.9176 over nine points 1.8–3.49 Å, residuals −0.032 to +0.040 eV),
  a Chebyshev series of R E_w with the penetration tables' range and switch, configuration-independent.
- **Why, measured:** SPEC 25's overlap term is frame-relative, so the frame pair's own repulsion was carried by the
  learned backbone alone, which the 4G data constrain only above Au–Mg 2.43 Å; PBE's wall for a dimer over a surface Mg
  is +2.02 eV at 2.0 Å where the backbone gives +0.23 (theory note §4c, benchmarks README step 24).
- Tests (`test_redox_frame_wall.py`, 4): the nine first-principles targets (5 % at R ≤ 2.45 Å, 0.04 eV beyond) and the
  table against its own direct values (1e-6); forces against finite differences and a finite double backward under
  anomaly detection (periodic images included); only the tabulated pairs contribute, additive, zero beyond the cutoff;
  the module adds exactly the wall and is bit for bit unchanged with the field off.

## Round 30 (2026-10-30, branch `exact-hessian-2026-10-30`): the fold's derivatives made exact (SPEC 32)

- **Why, measured.** Step 22's fine-tune (learning rate 1e-4) did not converge: over eight epochs both seeds'
  *training* force error rose (s43 0.126 → 0.143 eV/Å, s42 0.116 → 0.130), at a learning rate where a correct
  gradient descends. Checked against finite differences on the toy and on a real training structure (205, three
  explicit configurations; `probe_force_consistency.py`, `probe_mixed_second_derivative.py`), three defects:
  1. **The forces were not the energy's gradient with the exact block**: −dE/dx by finite differences against the
     model's forces, 6 %, 11 % and 27 % off (an O: −1.18 against −1.61 eV/Å; fold off and per-state both ≤ 7e-6).
     Cause: a folded state reached from several explicit configurations had its energy written by `index_put` from
     every duplicate leaf; the values agree, but the backward hands the full gradient to each duplicate, so ∂E_q/∂x
     was counted once per configuration. Fixed: the energy from one representative leaf
     (`test_the_exact_block_forces_are_the_energy_gradient`, failing first with −0.741 against −0.751).
  2. **The fold's s–s hop matrix was detached** (`T_fold = T_run.detach()`, SPEC 32's first version): the coupling
     weights received no gradient through the hops (dE/dθ 4.8 % off) and the hops' position dependence was dropped.
     Fixed (`test_the_module_forces_and_coupling_gradient_are_exact_with_the_fold[exact]`, at charge +1 with a 6 eV
     window so that hop-connected folded states exist; with the detach re-introduced it fails, 4.77 against −0.37).
  3. **The exact block's second derivative dropped the eigenvector response**: its fixed point took the C × C ground
     vector detached (Hellmann–Feynman, exact at first order), so d²E/dθ dx — the force loss's gradient — missed
     Σ_j |w_jᵀ ∂H w₀|² / (λ₀ − λ_j) (toy: −0.1897 against −0.1765, 7 %). Fixed: `_ground_vector` makes w₀ a function of
     H exact to first order through the reduced resolvent (constant R = Σ_{j>0} w_j w_jᵀ / (λ_j − λ₀)), and F is the
     Rayleigh quotient, exact to second order; only gaps from λ₀ enter (`test_the_exact_block_second_derivative_is_exact`).
  On structure 205 after the fixes: dE/dθ 3e-9, forces 1e-6 / 5e-6 (and 9e-5 on a 0.02 eV/Å component), the force
  loss's gradient dF/dθ 7e-8 relative to finite differences. Fold tests 40.
  4. **(same evening; branch `promoted-couplings-2026-10-30`) A promoted state coupled twice.** The module built the
     promoted rows' couplings from the request (the fold's V and the folded hops) *and* from its own T-coupling for the
     one-electron moves between their charge rows: an Au–Au hop counted twice (7.2 eV on seed 42's dump instead of
     the folded 3.76) and an O–O coupling the folded space never had (seed 42's learned 6.0 eV). The promoted rows'
     own T-couplings are now removed before the fold's are added (`test_a_promoted_state_couples_only_as_in_the_block`,
     failing first with 3.15 against 0.2). The re-partition identity then holds in the module on both dumps (the
     promoted ground equals the unpromoted block's to 1e-13 eV), and the forces through the promotion path agree with
     finite differences to 1e-8–1e-10. My toy identity test had assembled the promoted Hamiltonian by hand and so did
     not exercise the module's own couplings.
- **Consequences, stated.** Every frozen-counterfactual *energy* (wet_u, wet_d, T5) stands — evaluation needs no
  derivative. Every (c2) relaxation since SPEC 32 (the flat dimer's relaxed heights and energies, including s43's
  1.98 Å "collapse") used inconsistent forces and is superseded until re-measured. Every training with the fold
  (steps 22's launches of 2026-10-29/30) trained on a biased gradient: the per-state fold's position path through
  the detached hops, and with the exact block all three defects.

## Round 29 (2026-10-30, branch `promotion-2026-10-30`): promotion — the fold's strong-coupling regime (SPEC 32)

- **Why, measured.** Step 22's warm start died on both seeds (forwards 333 / 301) on the exact block's own rule: a
  close-contact training snapshot whose block ground state is 28–30 % explicit (E* 1.4–1.5 eV below the folded
  manifold) — the strong-coupling regime, where a Löwdin reduction to the explicit block is the wrong description
  and the charge-transfer state must be explicit. The per-state screen cannot see it (each configuration's own block
  passes); it needs a folded state below the explicit block's ground energy, which a single explicit state with
  every folded state above it can never produce (its weight is ½(1 + δ/√(δ² + 4v²)) > ½).
- **Built (design A, by decision).** `downfold.PromotionRequest` (a `DownfoldError`): `dressing_exact(promote=True)`
  raises it instead of refusing, naming the block's dominant folded state — charge row, donor O, hole component,
  energy (inner correction and component shift included), couplings to every explicit configuration, weight. The
  module's final solve (`_diabatic_solution`) promotes it: the row joins the explicit set with that energy on the
  diagonal and those couplings in the explicit Hamiltonian, and re-solves with `promoted=` (at most
  `MAX_PROMOTIONS` = 8). In the folded space a promoted (row, O, component) state is absent and its hops to the
  remaining folded states become its couplings; a folded state equal to any other explicit configuration is absent
  (its coupling is explicit already); the promoted state's component siblings stay folded; its own folded states
  (the second transfer) are dressed like any configuration's, so the promoted system spans a larger space and its
  ground energy can only fall. Records: `promoted`, `folded_states`; `_ensemble_last["promoted"]`; the record is
  complete whatever the outcome. Tests (fold tests 36): the request in a strong-coupling toy (folded states 2 eV
  below the frame), the Löwdin identity on the promoted system against an independent dense block with the same
  exclusions, the variational bound against the original block, the module's loop with an injected request. Both
  failure dumps (`redox_failure.pt`, s42 / s43) replay through the module with one promoted state each.
- **Batch promotion, and the re-partition rule (same day).** A relaunch died at eight one-at-a-time promotions
  (explicit weight 0.229: the manifold is nine near-degenerate states, the p_z states of four O's on either Au);
  the request now names the set of folded states in descending weight until the explicit weight reaches
  `PROMOTION_TARGET` (0.6; at most `PROMOTION_BATCH` 32), with their mutual hops (`mutual [n, n]`) as explicit
  couplings among them; `MAX_PROMOTIONS` is the number of passes (4). A promoted configuration's own moves are
  never folded (promotion re-partitions the block, it never enlarges the folded space): with them folded, four
  promotions fell 5.7 eV into the second electron transfer (O⁻ → O⁰ with Au₂²⁻), charge states the model never
  trained. The toy identity (the promoted system's ground energy = the block's) holds to 4096 ε; both dumps replay
  (s42 four promoted, s43 ten) with finite double backward. Fold tests 36.

## Round 28 (2026-10-29, branch `pauli-hop-2026-10-29`): the fold's two-site ordering and partner states (SPEC 32)

- **New field `RedoxConfig.downfold_hop_ordering`** (`bonding`, unchanged; `pauli`): the sign of the s–s hop inside
  the folded manifold (`dressing(hop_sign=)`). `pauli` puts the antibonding two-site state of a bonded s pair lowest,
  because the pair's own s electrons fill the bonding orbital. Replaced statement (SPEC 32, 2026-10-28): "the s–s σ
  hop is bonding, and the explicit Hamiltonian's sign of T is a gauge it never trained" — the sign is a gauge only
  for a donor coupled through one path; a donor coupled to both atoms of a pair fixes the loop phase.
- **Partner states** (`downfold_moves`): every s-class atom of a contact cluster holds a folded state for each of the
  cluster's donors (t = 0 where the donor does not reach it), coupled through the s–s hops. Before, an O touching one
  atom of a pair had no two-site structure at all (the standing dimer's O-top contact): the on-axis test measured fold
  −1.516 against the dense rebuild's −1.552 eV. `block_ground` drops leaves no chain of nonzero hops joins to a
  coupled leaf (an unreachable partner state made the secular solve singular). The default behaviour changes where a
  cluster's s atom lies beyond a donor's cutoff (the benchmark's upright dimer), nowhere else.
- **Why, measured (2026-10-29; theory note §4c, benchmarks README):** the site runs — a standing dimer over Mg0
  transfers 2.45× per contact what the lying dimer does at the same site, the hollow equals Mg-top (1.02) — and the
  flat acceptor run (gained density 80 % p-type, σ* unshifted) put the flat-site suppression in the acceptor pair's
  two-site states: the quantum O's lie on the dimer's bisector plane, σ*'s node. The donor-side rules of steps 21–22
  (anchoring, dangling axis, crystal field, σ* lobe) are closed as not the physics.
- **Bug fixed ({t}): the fold's hop groups collided across configurations.** The (configuration, O atom, component)
  key was `(c · P + o) · 3 + m` with P the number of moves, unique only while every O's atom index is below P — false
  in the benchmark cells (110 atoms, 24–36 moves), so hops of one configuration were written into another's block.
  Found by the partner states' refusal (a 2 × 2-structured block with Δ ≥ 3.16 and hop 1.91 reporting μ_min = −1.61);
  measured by `test_hop_groups_do_not_collide_across_configurations` (two configurations differing in a spectator
  Mg: fold −1.924 / −1.569 against the rebuild's −1.649 / −1.649); fixed by keying on the atom count U. Every
  multi-configuration fold since SPEC 32 (2026-10-28) ran under it: the frozen counterfactuals (+318 / +374,
  +686 / +745, +230 / +207) and the aborted step 22 are superseded and are re-measured below.
- **New field `RedoxConfig.downfold_block`** (`per_state`, unchanged; `exact`): the fold on the whole explicit block
  (`downfold.dressing_exact`): one folded state per (hole configuration, O, component) shared by every configuration
  that reaches it, the s–s hops between folded states, and per contact cluster the fixed point E_k* = lowest
  eigenvalue of [[H₀, V_k], [V_kᵀ, H_Qk]], Σ_k = V_k (E_k* − H_Qk)⁻¹ V_kᵀ, H = H₀ + Σ_k Σ_k (its diagonal the
  configurations' dressing, its off-diagonal the dressed couplings; additive over clusters; the per-state root when
  one configuration is explicit). Why: on both benchmark references the ionic Au⁺Au⁻ configurations carry a quarter
  of the ground state and share every folded state with the frame, so the per-state self-energy's premise fails
  (its size is measured with this mode, benchmarks README). `_diabatic_solution(block_correction=)` adds Σ to the
  ensemble Hamiltonian; the selection keeps the per-state screen. Leaf construction shared through
  `downfold._leaves`; the record carries `ground`, `explicit_weight`, `all_leaves_full`. Tests: the three
  `..._exact_block_...` tests and `test_the_module_runs_the_exact_block`; `test_redox_downfold.py` 28.
- **New fields `RedoxConfig.downfold_decay`** (0, unchanged: Harrison) **and `downfold_decay_anchor`** (2.26 Å): the
  coupling's radial form beyond Harrison, V(r) → V(r) e^(−κ (r − r₁)) (`downfold_moves(decay=, anchor=)`). Measured
  (theory note §4c, 2026-10-29): the O 2p – Au 6s effective coupling of the ALMO vertical series, read against its
  own Δ, falls as e^(−1.1 r) — 3.9× from 2.26 to 3.5 Å where Harrison's 1/(r² + r₀²) falls 2.33× — so κ ≈ 0.45 /Å
  beyond Harrison, anchored at the O-top distance; with the scale set at the O-top the plain form over-couples a
  3.5 Å contact 1.7× in v. Test: `test_the_coupling_decays_beyond_harrison_when_asked`; fold tests 29. A step-23
  candidate (not in step 22); its pre-registered frozen counterfactual is in the benchmarks README.
- **Solver fix (found by the (c2) relaxation trace):** `block_ground` ran its secular bisection and refinement on
  every coupled block, including one whose folded manifold is not above its configuration (μ ≤ 0) in the non-strict
  screen; when that block's lowest state is uncoupled (the antisymmetric combination of a symmetric donor under the
  Pauli ordering) the secular function has no root below μ, the bisection runs onto the eigenvalue and the batched
  solve is singular for every block at once. Such blocks are now solved on the padded matrix alone and reported
  +inf as before (`test_a_refused_block_does_not_break_the_batched_solve`). Step 22's first launch (21:08) ran with
  this defect loaded and was restarted on the fixed code.
- **The decay's form:** the anchored exponential raised the coupling below the anchor, a range the measurement never
  covered, and accelerated a short-range collapse in the frozen relaxation; `decay_factor` now ramps the exponent on
  with a softplus of width 0.2 Å (Harrison at and below the anchor, the measured decay above it, C¹, the anchor
  contact reduced by 6 %; `test_the_coupling_decay_keeps_harrison_below_the_anchor`). Fold tests 31.
- **Three double-backward defects found by step 22's first training step (seed 42, structure 76), all fixed
  test-first.** The training's force loss needs the second derivative of the energy, and three places were finite at
  first order but 0/0 or 1/0 at second: (i) `penetration.pairs_within` took the norm of the whole [n, n] image table
  including the zero-length self-pairs (torch's norm backward masks them at first order; its own backward is 0/0) —
  now the norm on the selected pairs only (`test_pairs_within_is_double_backward_safe_on_a_periodic_cell`); this
  harmless NaN tripped the nanguard's anomaly-mode retry on every step and blinded it to the real source; (ii)
  `downfold_moves` normalised the σ\* lobe with a norm over every atom's summed bond vectors, zero for every
  non-metal atom — now on the bonded atoms with a net lobe only; `star_ground` no longer takes √0
  (`test_downfold_moves_is_double_backward_safe`); (iii) the real source: `dressing_exact` took the block's ground
  energy from an eigendecomposition of the whole explicit-plus-folded matrix, whose backward divides by every
  eigenvalue difference — infinite on the exactly degenerate folded pairs (the symmetric O's p components) even
  with the ground state isolated by 1.15 eV. The ground energy is now the fixed point E = λ_min(H₀ + Σ(E)) found by
  bisection below the folded manifold and refined by two Newton steps with derivative 1 + ‖Xw‖² (the C × C ground
  vector detached: Hellmann–Feynman makes the first derivative exact), the same structure as the per-state root;
  nothing eigendecomposed is in the graph (`test_the_exact_block_is_double_backward_safe`). Replay of the dump:
  61 non-finite gradient tensors → 0, normal and anomaly mode, E* unchanged to 1e-12. Cost: per-state 1.9–2.4 s,
  exact 1.9–2.4 s per structure (energy + forces, quiet machine); the 24 s/forward in the first log was contention.
- **Cost at a random initialisation (2026-10-30).** Seed 42's initial weights admit 73–74 explicit configurations on
  the undoped structures (2–3 once trained; seed 43's 1.1 s per forward against 19 s): the selection screens its
  candidates through the per-state fold ~80 times per forward on ~1600 leaves. Two exact optimisations: `_leaves`
  evaluates each distinct moved configuration's energy once (the same folded state is reached from many explicit
  configurations), and `dressing` builds its hop blocks vectorised over the (configuration, O, component) groups
  (sorted keys, pairs of sorted neighbours) instead of a Python loop over them. Quiet machine, seed 42's
  initialisation, structure 205: 82 → 11.9 s per forward (fold parts 5.6 + 2.8 + 2.6 + 2.6 s); structure 76's
  fold parts ~10 s inside a 112 s first call whose remainder is not attributed (read from the training's timing
  lines). Fold tests 33 throughout (the dense-rebuild exactness tests cover the hops, pairing factors and sign).
- `BISECTION_STEPS` 120 → 60 (2026-10-30): sixty halvings of a bracket of a few eV reach 1e-18 eV, below double
  resolution, before the two Newton steps; the secular solves were 14 s of a 55 s forward at the 73-configuration
  initialisation. The exactness tests against dense eigenvalues (k · ε) pass unchanged (33 fold tests). The
  training's timing wrapper (`run_train_dump.py`) now names every forward slower than 5 s with its configuration
  count, leaves and the block's ground energies.
- **Record fix:** a block with no coupled leaf (partner states only, e.g. the near Au already Au⁻) reported μ = 0
  into the dressing record, so `mu_min` read 0 and the probe's (c3) failed; `block_ground` now returns NaN for such
  a block and the record skips it (`test_a_block_without_a_coupled_leaf_reports_no_mu`).
- Tests: `test_the_pauli_ordering_of_a_bonded_pair`, `test_hop_groups_do_not_collide_across_configurations` and
  `test_a_block_without_a_coupled_leaf_reports_no_mu` (new), the move-class test extended; `test_redox_downfold.py`
  23. Suites: redox 771 (before the record fix), the rest 1039 passed, 4 skipped. The full suites and the pre-registered frozen counterfactuals (P1 legacy / P2 Pauli, both with partner
  states) were running when this entry was written; their numbers go to the benchmarks README.

## Round 27 (2026-10-28, branch `downfold-ct-2026-10-28`): the downfolded O–s bond (SPEC 32, step 22)

- **New fields `RedoxConfig.downfold_transfer`** (False, unchanged) **and `downfold_cutoff`** (4.5 Å). On, every
  O–s one-electron move leaves the explicit ensemble and dresses each configuration's energy by the exact block root
  of its contact clusters with Harrison's fixed V_spσ, the three p components of a closed-shell O donor and the moved
  electron's s–s hops (`trace_redox/downfold.py`). The selection sees the dressed energies
  (`diabatic.search_configurations` / `coupled_closure` / `select_configurations(correction=, bound=)`), folded
  partners are dropped from the explicit set, a folded manifold at or below its state is refused. Tests:
  `test_redox_downfold.py` (13). Suites: redox 762 (749 + 13), the rest 1039 passed, 4 skipped.
- **Why, measured:** PBE's O-top relaxation is −1.96 eV (transfer −1.10, 94 % O → Au), the model's a seventh; step 21
  showed a physical coupling cannot live in the selected ensemble. Frozen-weight counterfactual on step 20 (first
  version): wet_u +541 → +1038 / +735 → +1213 meV; the second version's numbers in the benchmarks README.
- **Reviewed (2026-10-28, two passes):** the first version's uncoupled folded states (the Au₂ resonance, 0.2–0.3 eV),
  undressed selection, re-entering folded states, species-level masking against a state-level class, NaN at
  non-finite or non-positive Δ, and a switch on [3, 4] Å were each found and corrected before any run (SPEC 32,
  review record).
- `probe_downfold_counterfactual.py` reports min μ (the P/Q margin) on the references and at the relaxed flat
  geometry, and 'no folded pair' where none exists.
- **2026-10-29: the pair's measured constants** `downfold_scale` (c = 0.71), `downfold_anisotropy` (κ′ = 0.22, the
  dimer's sp-hybridised σ\* acceptor), `downfold_crystal_field` (β = 1.0 eV per extra cation neighbour of a hole's
  p component), `downfold_bond_cutoff` (3.2 Å); defaults (1, 0, 0) unchanged. From PBE on the MgO(100) cluster: the
  vertical O-top transfer against height, a single Au⁺ acceptor vertical against over-Mg, the band orbitals' hole
  energies, the Au₂ LUMO's composition (SPEC 32). Tests 16; suites redox 765, rest 1039 + 4 skipped (2026-10-29).
- **2026-10-29, 06:35: the σ\* anisotropy falsified** by the 28-ion flat split (CT_cp −0.444 against < −0.80;
  κ′ = 0, path kept off); **`downfold_free_axes`** (a hole couples through the O's dangling p axes only; measured
  0.30–0.36 angled/vertical, the rule's 0.50 bounds it) and **`downfold_moves(units=)`** (a pair inside one covalent
  unit is never folded: water's O–H). Tests 20; suites redox 769, rest 1039 + 4.
- **2026-10-29, 09:30: the fold's cost.** The first launch ran at 107 s per forward (step 20: 0.7 s): the window
  search dressed every candidate one at a time and SPEC 28 ran on every folded state of every candidate (166 / 281
  calls per forward), beside a thrashing machine. `search_configurations` now evaluates its candidates in one batch
  per frontier configuration, and the selection's dressing takes Δ without SPEC 28 (the solve keeps it; the
  selected set and the record were unchanged on the profiled structure). Fold forward 2.07 s against 1.46 s as
  trained.

## Round 26 (2026-10-27, branch `coupling-anchor-2026-10-27`): Harrison-anchored couplings (SPEC 31, step 21)

- **New field `RedoxConfig.coupling_anchor`** (None default, unchanged; `'harrison'`). Every pair but a frame cation-anion pair
  takes Harrison's universal two-centre value times a learned factor in (1/2, 2), with a 1 Å window switch; the
  cation-anion pairs keep the learned value under SPEC 27's bound. `heads.frontier_shell`, `anchored_pairs`,
  `window_switch`, `TransferCoupling(anchor=, forward(..., species=))`; the module passes the species. Tests:
  `test_redox_coupling_anchor.py` (8). Old checkpoints give bitwise-identical energies.
- **Why, measured:** the model's O-top relaxation is a tenth of PBE's, PBE's transfer there is 83 % O → Au, the
  learned O → Au coupling is 0.1–1.0 eV against Harrison's 2.75 eV, and unconstrained O–O couplings reached 28 eV
  (step 20 seed 42: clean-slab IP −267.7 eV; anchored: +5.14 eV, PBE +4.80). SPEC 31 states two first versions
  corrected on frozen weights before any run (the full-range switch; the open-frame-only rule).
- **Erratum recorded (theory note, wetting_eda.py):** the 4G doped cell holds three Al and two band electrons; the
  one-Al cluster runs measure a sub-problem.

## Round 25 (2026-10-25, branch `pauli-refit-2026-10-25`): SPEC 25's K from first principles (step 20)

- **New field `RedoxConfig.pauli_table`** (`'gordon_kim'` default, unchanged and bit-identical; `'first_principles'`
  reads the overlay `trace_redox/data/pauli_fp.npz`, which replaces only the Pauli series and must carry the library's
  pair keys in order; it needs `carrier_overlap='gordon_kim'`). `penetration.library(pauli)` holds one instance per
  choice. Tests: `test_redox_pauli_table.py` (5).
- **The overlay** (`tools/electrostatics/build_pauli_fp.py`): each one-carrier K of the AuMgO element pairs fitted to the
  frame-relative frozen-density PBE interaction in the Watson-sphere convention over the 4G data's contact distances
  (5th–95th percentile, 5 points); K_cc and the other pairs keep Gordon-Kim. Check (v1): fed the Gordon-Kim K, the
  reconstruction reproduces the library's series to 3.2e-16.
- **Why, measured:** theory note 2026-10-18 §4c (the runtime table against the lattice-embedded values; the sphere
  convention within 0.05–0.10 eV of them; the frozen-weight counterfactual wet_u +118 / +189 meV toward DFT).

## Round 24 (2026-10-22, branch `band-prior-2026-10-22`): the band unit's initial hopping factor (SPEC 20)

- **New field `RedoxConfig.band_initial_factor` (default 1, Harrison; every existing configuration unchanged,
  bit-identical).** The band unit's hopping τ = f V_ssσ starts at f = band_initial_factor (bounded to the open range
  (1/20, 20) of exp(ln 20 tanh s); refused otherwise). Tests: `test_redox_band.py`
  (`test_band_initial_factor_sets_the_hopping_factor_and_scales_tau`, `test_band_initial_factor_is_inside_the_bounded_range`).
- **Why, measured.** PBE's conduction-band mass of bulk MgO at the 4G lattice constant, 0.322 / 0.330 mₑ
  (`tools/electrostatics/mgo_pbe.py`), against the nearest-neighbour s band's m* = 0.179/f mₑ: f = 0.55. At f = 1 the
  step 15 dumped call had λ₁ = −6.60 eV (−10.80 eV of hopping) and 1338 configurations in the window (the refusal);
  at f = 0.55 the same call has λ₁ = −1.742 eV and 3 configurations, no refusal (`probe_dump_terms.py`,
  `replay_redox_failure.py` REPLAY_BAND_FACTOR).
- **Not adopted: the derived hopping.** The Pauli-projected two-site coupling (`tools/electrostatics/band_hopping.py`)
  failed its pre-registered test (1.107 eV against 0.635–0.651); a minimal basis of frozen ionic orbitals cannot
  represent MgO's nearly-free-electron conduction band (post-hoc: m* 2.35–2.67 mₑ). Theory note 2026-10-18 §4c.
- Pre-registered run: step 19 (benchmarks README).

## Round 23 (2026-10-18): the configuration window made F discontinuous; the amplitude selection (SPEC 21)

- **Step 12 seed 43's T5 deficit, diagnosed** (README "Diagnosis of step 12 seed 43").
  - **Correction:** it is not in the relaxation. The undoped wet − non-wet difference is already +612 meV
    (seed 42 +911) at the unrelaxed references, and relaxation moves it by < 40 meV.
  - **Not a rough surface:** the snapshot-to-reference paths are harmonic, with D(½)/D(0) = 0.24–0.28.
  - **The mechanism:** the §17.3 energy window. A coupled configuration crossing its edge moves F by ~−t²/Δ,
    and the detached search hides it from the forces.
  - **Measured:**
    - −834 meV on one test structure (seed 43; Au⁺Au⁻ at 2.50 eV, τ = 1.18 eV) and −219 meV on seed 42;
    - never converged (O → band transfers at 12–16 eV add −200 to −260 meV);
    - on all 400 test structures the truncation deficit carries seed 43's scatter: r = +0.99 and +0.98;
      residual sd 307 → 41 and 187 → 40 meV once it is removed.
- **Built, SPEC 21, test-first: the amplitude selection (`diabatic_selection='amplitude'`).**
  - Importance r_c is the leading Rayleigh–Schrödinger amplitude of a configuration along its best path from
    the thermal reference: r_c = max(√p_c, max_i r_i·|T|/√(x_c² + T²)).
  - Configurations with r > 0.05 are kept, and couplings are switched (C², log-scaled) to full at 0.2.
  - **Guarantees:** P is closed, so F is window-independent up to decoupled Boltzmann weight and continuous
    in R. The complete limit is full diagonalisation.
  - **First version discarded:** pairwise gaps chained O-hole hops and exceeded the 512 cap on frozen step-12
    weights.
  - **Frozen step-12 weights with the final version:** 2–7 configurations, unchanged cost, window dependence
    0.00 meV, seed 43 undoped-wetting scatter 309 → 32 meV.
  - **Tests:** 20 new; redox suite 608 passed.
- **Step 13** (step 12 + `diabatic_selection: amplitude`, seeds 42 and 43) was pre-registered and launched at
  08:33. After epoch 1, τ(Au, Au) had grown from 0.01 to 0.15–0.30 eV: gradient flows through the nearly
  degenerate doped structures.
- **Over-polarisation of covalent metal bonds, diagnosed** (README).
  - Isolated Ag₂ (step 9's Ag₃ model) has α∥ = 112 Å³ against GGA ≈ 22 (bound ≤ 41). Its Ag⁺Ag⁻ lies
    1.71 eV below Ag⁰Ag⁰: an ionic ground state.
  - **Cause 1, double counting.** The bound split-charge variable on the bond screens the free ionic pair
    (Au₂: bond channel −3.41 eV for seed 42, −0.85 eV for seed 43). Its learned hardness differs 1.7× between
    seeds and moves Au₂'s Δ_ion by 2.3 eV. With the bond channel removed the spread is 0.26 eV.
  - **Cause 2, compact widths.** The pair's attraction is ≈ K/d; Ohno and Mataga–Nishimoto give ~3–4.5 eV.
- **Built, SPEC 22, test-first: `bond_partition`.** The split charge is kept only inside covalent units of more
  than `full_unit_atoms` atoms; between units charge flow is the free sector's. It applies to `forward` and to
  `bound_state`.
  - Tests: 6 new; redox suite 614 passed.
  - Frozen counterfactual: MgO ε∞ 3.16 → 1.41, which the moments must carry after training.
  - New probes: `probe_dimer_ct.py`, `probe_eps_bound.py` (score_run.sh uses it for SPEC 22 models),
    `probe_eps_nacl.py`, `probe_ag_polarizability.py`.
- **Step 14** (step 13 + `bond_partition: true`) was pre-registered. The watcher `watch_step13_14.sh` launches
  it when step 13 ends and scores both.
- **U6 for the cluster runs:** NaCl ε∞ 2.330 and Ag α 7.552 Å³, both trained priors reproduced.
- **Cause 2 measured from first principles, and corrected.** Compact widths do not overbind Au⁺Au⁻: with
  X2C-PBE0 ion densities the net error is −0.06 eV (Ag⁺Ag⁻ +0.66 eV). The Ohno / Mataga–Nishimoto 3–4.5 eV
  quoted before was a semi-empirical estimate, not a measurement.
- **The width error that matters is carrier against substrate ion.** At 2.2 Å from O²⁻ the model favours Au⁺
  over Au⁻ by 3.8 eV too much (3.4–4.0 eV over Watson radii).
- **Built, SPEC 23, test-first: `carrier_penetration`.** The exact electrostatics of overlapping first-principles
  charge-state densities (nucleus + electrons) replaces the carriers' Gaussian Coulomb for every pair within
  8 Å, as a two-body term of the configuration energies.
  - The band unit's ε_v, J₁₁ and band–unit pairs see it too.
  - Data: `trace_redox/data/ion_densities.npz` from `tools/electrostatics/build_ion_densities.py`.
  - Densities: X2C-PBE0/ANO-RCC + diffuse; every O state in one Watson sphere matched to MgO's Madelung
    potential.
  - Pair tables: the cancellation-free difference route (grid-converged to 1e-7 eV), 96-term Chebyshev in R.
  - **Artifact checks:** the Watson radius (±9 %), the basis (diffuse augmentation +2.4 %) and the functional
    (PBE0 against ωB97X 3 %; HF is the outlier). A free O⁻ against a confined O²⁻ carried the confinement and was
    replaced; a C¹ tail splice was not robust and was removed.
  - **Frozen step-12 counterfactual:** undoped non-wetting free charges ±0.475 → ∓0.446 (seed 42) and
    ±0.463 → ±0.108 (seed 43).
  - Tests: 18 new; redox suite 632 passed.
- **Step 15** (step 14 + `carrier_penetration`) was pre-registered (Q1–Q5). `watch_step15.sh` launches it after
  step 14.
- **Step 13 seed 42:** a fourth non-reproducible NaN (epoch 8, load 26–30). Resumed from epoch 7. My CPU jobs are
  now run one at a time.

## Round 24 (2026-10-18, afternoon): reboot recovery; exact isolated limits (SPEC 24); theory note

- **The Mac rebooted at 13:56.** Step 13 (both seeds) and the carbon chain resumed from their last checkpoints
  (epochs 12, 13 and 272), at most one epoch lost. Watchers restarted with the new PIDs.
  - The /tmp scratch was wiped. The density work directory was rebuilt from the library in
    `$TRACE_FP_CACHE/ion_densities` (persistent), and Ag²⁺/Ag³⁺ are recomputing.
- **Isolated-ion anchor violated, measured:** trained shifts on isolated atoms of Au⁻ +1.18 eV, Au⁺ +1.25 eV and
  Ag⁺ −1.36 eV. Every dissociation into ions inherits them.
- **Built, SPEC 24, test-first: `shift_gate`** (an exact isolated limit, C²). Na + Cl crosses at 9.462 Å against
  the exact 9.434.
  - Band state range fixed: an isolated Mg was refused before. Mg²⁺ + 2 band electrons now reproduces the neutral
    atom to 3.6e-15 eV.
  - Identity on the AuMgO data (every atom has a neighbour within 2.94 Å), so it is on in runs after step 15.
- **Measured, not built:** the carrier field at a moment. By Gauss, λ = the enclosed fraction of the smeared
  carrier: 0.475 (Au⁻) and 0.668 (Au⁺) at O at 2.2 Å, against the model's 0.827. The induction is overstated 3×
  and 1.5×; about 0.1 eV of net preference error, opposite to penetration.
- **Theory note** docs/ELECTROSTATICS_THEORY_2026-10-18_overlap.md: the window discontinuity (Proposition 1), the
  closure lemma, the double counting (Proposition 2), the cancellation-free penetration (Proposition 3), the Gauss
  field factor (Proposition 4), the isolated limits, and the open problems ranked.
- **Built, SPEC 25, test-first: `carrier_overlap='gordon_kim'`**, the Pauli half of the overlap from the same
  first-principles densities.
  - Gordon–Kim G is reliable at contact; beyond ~3.5 Å its tails are spuriously attractive (measured).
  - G/ΔS is nearly constant at contact, so E_Pauli = K·ΔS (bilinear, three components), with K derived from G at
    contact. Median fit residual 5.6 %; AuMgO types 1.6–8.2 %.
  - It replaces SPEC 18's fitted exclusion and refuses to run with it.
  - Net overlap: Au⁻–O −1.11 eV, Au⁺–O +0.11, Mg⁺–O −0.36, Au⁺Au⁻ +0.62 (the Pauli cost of the ionic dimer).
- **Step 15 amended before its run** to SPEC 25 + SPEC 24 (`exclusion: false`). Q1–Q5 unchanged.
- **Ion-density library:** 983 pair types; Ag³⁺ still computing.

## Round 28 (2026-10-19): the ε∞-prior conflict fixed (SPEC 30); step 13 scored; a shutdown

- **Step 13 scored (A1–A5).**
  - A1 and A2 pass: window independence exact; residual sd 29–51 meV, against step 12 seed 43's 307.
  - A3 fails: T5 −544.1 (58 %) and −1368.7 meV (147 %).
  - B3 fails: undoped class offsets −390 and +362 meV, of opposite sign.
  - A5 fails: undoped E RMSE 1.84 / 2.18 meV/atom.
  - The seed dependence moved from scatter into the ionic admixture of the upright dimer (±0.135 against ±0.933),
    the gauge freedom of Proposition 6.
- **Step 13F crashed** (epochs 7 / 15). The bound-only ε∞ prior refused bulk MgO once it held 5 configurations.
- **Built, SPEC 30, test-first: the whole model's response for the priors.**
  - α = α_b + α_ens, exact for state-independent K, with minimal-image configuration dipoles in periodic cells.
  - The decomposition equals the whole model's finite-field α to 1e-5.
  - The priors now train structures with several configurations instead of refusing them.
  - The record is opt-in (only the priors build it).
  - Tests: 9 new; 2 refusal tests replaced by whole-response tests.
- **Measured:**
  - SPEC 21's switch band leaks into the response, −3.99 against +2.36 e²Å²/eV on a random model, but not in the
    trained step-13 models (agreement to 0.01 Å³).
  - Only one coupled pair sits in the band across the eight T5 references (10.5 meV). The switch is not T5's cause.
- **Shutdown at ~10:09:** step 14 resumed from epoch 10, step 13S from epoch 4. Watchers rebuilt; 13FC re-armed.
- Suites after the reboot: redox 704 passed; the rest 1039 passed, 4 skipped.
- **SPEC 30 made exact with SPEC 28 (test-first).** The record no longer refuses state-dependent polarisabilities,
  which would have crashed step 17 (ε∞ prior + SPEC 28) as 13F crashed.
  - Per configuration, by Woodbury on K(n) = K + P_Sᵀ Δ_S P_S: dM(n) = −G_Sᵀ W_S y_S(n) and
    dα(n) = −G_Sᵀ W_S G_S, W_S = Δ_S (I + C_SS Δ_S)⁻¹; the field enters H's diagonal as F·M(n) + ½ Fᵀ dα(n) F.
  - Au₂ at 2.5 Å, 3 configurations: decomposition = whole-model finite-field α (0.393306, 0.430776,
    1.051144 e²Å²/eV); without dα 0.226, 0.242, 0.983.
  - Tests: 4 new (Woodbury against brute force ×3; the SPEC 28 decomposition with its counterfactual), and the
    periodic ε∞-prior training test parametrised with SPEC 28 on (step 17's path).
- **Withdrawn: "the T5 seed dependence is the gauge freedom of Proposition 6"** (measured, `probe_gauge_split.py`).
  - At the four T5 references of step 13: T5 = E_s −1.5 + E_q −555.2 (seed 42), E_s +4.7 + E_q −1377.3 meV (seed 43).
  - The short-range network cannot see the dopant, so it cannot absorb a change of E_q's T5: T5 is pinned by the
    energies and lies off the gauge orbit.
  - The 816 meV seed difference sits in the upright undoped dimer: p(Au⁺Au⁻) 0.22 against 0.925, E_q(w − nw)
    +0.379 against +1.105 eV. Seed 42's Au shifts are +1.57 to +1.77 eV (0.79–0.89 of χ_bound), seed 43's
    +0.17 to +0.55.
  - So the minimum-shift gauge penalty would move no T5 number. Not built.
  - Energy weight as the cause: already falsified on 2026-10-15 (fine-tune at E weight 1000, class offset −411 → −303).
- **SPEC 28: K(n) must be positive definite (test-first).**
  - A softened state (Au⁻ 3.39; Mg⁺ 70.7, Al²⁺ 53.9 against their frames) can make K(n) indefinite. Woodbury
    then returns a saddle's value, finite and wrong.
  - Exact check by Haynsworth inertia additivity: K(n) > 0 iff Δ_S⁻¹ + C_SS has as many negative eigenvalues as
    Δ_S, at O(|S|³). Such a configuration is refused.
  - The check found that the first SPEC 28 tests' random problems had indefinite K(n) (minimum eigenvalue −0.35
    to −3.26): their brute-force comparison was between saddles. The problems now have weaker couplings, and every
    test asserts K(n) > 0.
  - Cause, stated in SPEC 28: SPEC 12's guarantee ties each dipole's damping width to its stiffness, and SPEC 28
    keeps the frame's width. The guarantee then holds only for ρ < 3^{1/4}/a = 2.63.
  - Measured on trained weights (steps 13, 13F, 14; brute-force spectrum agrees in all 17 MgO cases): every
    constructed Mg⁰ configuration of bulk MgO, every Mg⁺O⁻ one except on 13F seed 43 (step 14: min eig K(n)
    −0.362 / −0.403 against +2.42 for K) and every Al²⁺O⁻ / Al⁺ one of the doped slab is a catastrophe. No selected configuration is (186 over 45
    structures, both seeds of step 14).
  - `state_polarizability_elements` (opt-in) restricts the ratios to listed elements. Step 17 amended before its
    run to [79, 8]: on step 14's weights it changes no selected configuration, and it removes a predictable crash.
- **Isolated dimers: classical electrostatics inverts the ionic–covalent order (Proposition 8, theory note §4c).**
  - U_cl = IE − EA − K/R − αK/R⁴ is negative at R_e for H₂, Li₂, Na₂, Ag₂ and Au₂.
  - In the trained models with the shifts zeroed: Au₂ −0.61 / −2.02 eV (step 13, seeds 42 / 43), Ag₂ −0.74 eV
    (Ag₃ step 9). With SPEC 22/23/25/26/28: +0.85 / +0.90 and +1.22 eV.
  - As trained, seed 43's and the Ag₃ model's neutral dimers are ionic (gap −1.0 and −1.7 eV). That is the free
    sector's (gap, τ) degeneracy: same bond energy, opposite ionic weight.
  - ~~Frozen step-13 weights under each queued step's settings: T5 seed spread 816 → 51 meV at step 17's stack;
    magnitude 43 / 38 % of DFT.~~ Withdrawn (artifact, below).
  - New exact test: an ion pair beyond every cutoff is IE − EA − K/R plus the full 6×6 induction, to 256 ε, with
    SPEC 28's states.
- **13FC disarmed before launch** (confounded: α⊥ of Au₂± is bound-only and charge-independent without SPEC 28;
  measured +60 % / −47 %). Step 15 now launches in 13S's slot.
- **SPEC 26 scalable route** (exactly equivalent; 3 tests): V on the dipole rows only, (K⁻¹)_dd from the condensed
  Schur factor, no [m, n S] matrices. 512-atom MgO with step 17's stack: 119–122 → 66 s per energy + forces.
- **Proposition 8 on the other benchmarks** (frozen step-9 NaCl and Ag₃ models; `probe_dimer_ladder.py`).
  - Na₂ −1.41 and Ag₂ −1.71 eV ionic as trained; Cl₂ covalent (+1.40); NaCl ionic (−6.42), as each should by
    the criterion. The overlap terms make every homonuclear pair covalent.
  - Couplings extrapolate past the data: τ(Cl–Cl, 1.99 Å) 24.9 eV, τ(Na–Cl, 2.36 Å) 8.6 eV, against Harrison's
    6.2 and 2.5 eV (SPEC 27's bound would cap them; no benchmark number moves).
  - Ensembles (`probe_cluster_ensemble.py`): Ag₃⁺ multi-ion weight 0.984 (ground Ag⁺Ag⁺Ag⁻) → 0.018 with the
    overlap terms; Ag₃⁻ 1.000 → 0.17 with the overlap terms and zeroed shifts. That is the candidate for U5's
    undiagnosed Ag₃⁺ force failure.
  - Na₈Cl₈⁺: no multi-ion weight; the overlap terms localise its Cl⁰ hole (a risk against PBE data).
  - Pre-registered: the Ag₃± rerun with the overlap stack (V1–V4), to launch when the CPU allows.
- **Scoring:** `probe_dimer_ct.py` had crashed since step 13 (a fixed wrapper signature) and left out the pair
  terms; both fixed, and step 13 re-scored (Δ_ion +2.961 / −0.997 eV). Added `score_extra.sh`, `watch_extra.sh`
  and `read_criteria.py`.
- **Artifact found and fixed: counterfactuals that switched SPEC 18's exclusion off left it on.**
  - The module tested whether the exclusion submodule had been built (`self.exclusion is not None`), not the
    configuration. So `dataclasses.replace(redox, exclusion=False)` on exclusion-trained weights kept the fitted
    exclusion beside Gordon–Kim: Pauli was counted twice.
  - The couplings and the band unit had the same pattern. All three now follow the configuration, a no-op whenever
    it matches construction, as in every training run.
  - Test first (`test_a_configuration_switched_off_after_construction_is_off`): 0.139 eV before the fix, 1e-12
    after.
  - Re-measured: the frozen T5 under steps 15–17's settings is +399 / +158, +292 / +0.3 and +108 / −170 meV
    (spread 278, not 51). The "816 → 51 meV" claim is withdrawn.
  - Unchanged: the isolated Au₂ and Ag₂ gaps, the scorecards, the Ag₃ ensembles and Cl₂.
  - Corrected: Na₂ +1.68 → +1.16 eV and Na⁺Cl⁻ −5.60 → −4.14 eV with the stack (shifts zeroed).
- **Exact T5 decomposition** (`probe_t5_terms.py`; F = Σ p_n E(n) + Tr(ρH_off) − TS identically). Step 13 seed 42,
  as trained: screened electrostatics −318, SPEC 18 exclusion −413, ionic admixture −139, hops −151, learned shifts
  +453 meV. The shifts oppose T5; Au⁻'s shift is 1.2 eV lower at the O-contact site, compensation for the missing
  penetration.
- **Consolidation, no physics change: trace_redox SPEC §0, the model in three equations.**
  - E(n) = Σ ε_u + ½qᵀJq − ½sᵀK(n)⁻¹s + Σ O_uv (ions, Coulomb, induction, overlap); H_nm = ±σfτ; F = −kT ln Tr e^(−H/kT).
  - Every SPEC section is mapped to its term and status. Legacy paths are kept only for recorded runs (audit:
    §18 is used by the running trainings and the three benchmark models; nothing deleted while `trace_redox/` is
    uncommitted).
  - New test (4): the module's assembled diagonal equals (0.1) evaluated directly, to 1e-10, in an open cluster
    and in rock-salt MgO.
  - The reference first took S₀ = diag(J), which under PBC includes an ion's own lattice images (27.4 eV per
    Mg⁺O⁻ hop): the reference's error, not the model's.
- **Step 13's class offsets are on its training data** (300 training structures): undoped flat − upright
  −396.0 ± 6.9 / +353.8 ± 5.4 meV (test −389.9 / +361.8); doped +40.0 / −26.9. A representational limit of the
  form, not generalisation and not the loss weight. Steps 15–17 carry the state-resolved contact physics that
  could close it.

## Round 27 (2026-10-18, night): the Au₂ polarisability, a field-response prior, a fifth NaN

- **Au₂ polarisability from first principles** (`tools/electrostatics/dimer_polarizability.py`, PBE0/def2-TZVPPD).
  - α∥/(2α_atom) = 1.357, 1.515, 1.654, 1.894, 2.302 at 2.30–3.10 Å; α⊥ 0.79–0.87.
  - ωB97X agrees to ~1 % on α∥.
  - The library-method (X2C/ANO-RCC) Au₂ was stopped: one SCF ran past 38 min.
- **The models' Au₂ α∥ is the seed-dependent quantity.** α⊥ is right to 2–5 % in every model. α∥ is 9–23 %
  low (step 12, seed 42), non-monotone (step 12, seed 43), or too high and diverging at 3.1 Å (step 13, seed 43,
  epoch 23). This is a partition-free observable of the free sector.
- **Built, test-first: `electrostatics.field_polarizability_prior`** (train.py).
  - The whole model's static polarisability by finite fields of the total energy (h = 0.01 V/Å). It accepts
    structures with several configurations and reaches the couplings and the charge-state shifts.
  - 5 tests: it trains a coupled Na₂⁺; it equals the calculator's finite-field α to 1e-5; gradients reach the
    free sector; periodic structures and non-diagonal labels are refused.
  - Data: `experiments/benchmarks_4g/data/Au2_alpha_field_prior.xyz`.
- **Step 13F pre-registered** (step 13 + the field prior; F1–F5). It runs in parallel with step 14
  (`watch_step13f.sh`).
- Fifth NaN (Round 26) handled; step 13 seed 43 resumed from epoch 23.
- **Two transient faults recovered exactly by the guard.** The NaN probe bracketed one inside
  `free_charge_lattice_blocks`' σ-backward (elementwise kernels plus gather and scatter; no BLAS on that path).
  The probe now also wraps the lattice kernels from outside (3 tests).
- **Reference convergence:**
  - def2-TZVPPD is not converged for anions: PBE0 Au⁻ HOMO +0.26 eV and α 77 bohr³, against 122 with two more
    diffuse shells, where PBE0 and ωB97X agree. The library's 123.7 stands, and an inferred 2× confinement
    overshoot is withdrawn.
  - The neutral Au₂ targets of step 13F are converged to ≤ 1 %.
  - With SPEC 22–28 on frozen weights, α⊥ of Au₂⁺ / Au₂ / Au₂⁻ follows DFT (model/reference 1.23 / 1.06 / 1.10).
- **Second reboot at 22:03:** all runs resumed from their last epochs with `resume_run.py`.
- Suites: redox 688 passed.
- **Built, SPEC 29, test-first: `spin_pairing`.**
  - The spinless configuration couplings get the exact two-site spin factor: √2 on moves that pair or unpair
    frontier electrons (equal parities), 1 otherwise. The two-site Hubbard ground energies for N = 1, 2, 3 are
    reproduced to 1e-12.
  - Frozen step-13 weights: T5 moves toward DFT for both seeds and the spread falls 863 → 589 meV.
  - Step 13S pre-registered. 7 tests; redox 695 passed.
- **Au₂⁻ converged reference** (ωB97X, augmented basis): α∥ 43.0, α⊥ 16.9 Å³ at 2.47 Å.
  - The full stack gives 33.0 / 19.6. Two-state diagnosis: τ = 1.21 eV with μ_eff = 1.27 e·Å (48 % screened),
    so the response requires τ of order 0.3 eV.
  - Seed 43's isolated Au₂⁺ has a pathological α∥ (1440 Å³): its learned coupling is near zero there.
- **α scorecard over nine first-principles tensors (Au₂⁰/⁺/⁻):**
  - frozen step-13 seed 42: RMSE 8.59 → 4.19 Å³ with SPEC 22–28;
  - seed 43: 8.68 → 506 (an unidentified τ ≈ 0 in the cation environment).
  - Step 13FC (the field prior over all three charge states) pre-registered, with the Proposition 7 prediction
    that a spinless τ cannot meet all three. Queued after 13S.
- Third transient recovered: bracketed at the configuration energies, not the lattice. Two entry points, no
  single singular operation.

## Round 26 (2026-10-18, evening): a collapse of the potential, the coupling bound (SPEC 27), small cells, α(q)

- **Interim look at step 13 (not criteria):**
  - **Seed 42, epoch 16:** T5 −513 meV (55 %), upright charges much nearer DFT (+0.157/−0.191 against
    +0.110/−0.131).
  - **Seed 43, epoch 19:** the undoped wetting relaxation collapses.
    - Stage 1, exact forces: the flat Au₂ goes ionic, −1.27 eV, the Au⁺-near-O²⁻ error.
    - Stage 2: an Au is driven into a surface Mg, with the learned T(Mg, Au) 0.35 → 9200 eV.
- **Built, SPEC 27, test-first: `coupling_bound='harrison'`.**
  - B = 3.24 ħ²/(m(r² + r₀²)) with r₀ = 0.5 Å, applied as a C² soft clip that is the identity below 0.75 B.
  - It is the identity on all 404 AuMgO structures (max 0.650 B).
  - The first version with a bare 1/r² was measured insufficient: the Au still ran onto the nucleus, −2.2·10⁶ eV
    = −B(r). Hence r₀.
  - With the bound the collapse remains. The potential has no short-range nuclear repulsion (no ZBL in `flashace`;
    the network changes by −35 eV at 0.003 Å).
  - Pre-registered as a contingency (benchmarks README), not built. SPEC 27 is off by default and not added to
    steps 14–16.
  - Tests: 5.
- **Built, test-first: cells shorter than the penetration cutoff.**
  - Before, any atom meeting its own image within 8 Å was refused (rock-salt MgO's 4.21 Å cell; most bulk cells of
    foundation datasets).
  - Self-image pairs are now the one-body term Σ_half Ξ_ii(|T|; s, s). In SPEC 26 the self-image columns cancel
    exactly in ±T pairs.
  - The cell equals its 2 × 2 × 2 supercell per cell to 1e-10, and the module gives forces −∇E on the small cell.
- **Found by that test and fixed: the same-element tables were not exchange-symmetric.**
  - The builder fits (Z, n, Z, m) and (Z, m, Z, n) separately, with asymmetry up to 7.8e-6 eV·Å (electrostatic)
    and 2.4e-2 eV·Å (Pauli).
  - The library now holds their mean on load, which is exact and bitwise symmetric.
  - The Gordon–Kim table test now compares with the symmetric mean.
- **What carries T5 in step 12** (wetting decomposition): the undoped preference is the free sector's ionic
  admixture, +810.7 of +918.6 meV for seed 42. Hypothesis: part of seed 42's 103 % rests on the Au⁺-near-O²⁻
  error.
- **Ion-density library:** Ag³⁺ added, 1030 pair types; the old entries are bitwise unchanged.
- **State-dependent polarisabilities, being measured** (`tools/electrostatics/ion_polarizabilities.py`, the
  library's method).
  - Checks against reference values, in bohr³: Na⁰ 166.8 (exp 162.7), Na⁺ 1.03 (0.99), Mg⁰ 75.1 (71.2),
    Mg⁺ 35.3 (35.0), Mg²⁺ 0.50 (0.49), Cl⁰ 14.9 (14.7), Al⁺ 24.9 (24.1).
  - O in the Watson sphere: O²⁻ 2.44 Å³ (MgO crystal value ~1.7 by Clausius–Mossotti), O⁻ 1.08, O⁰ 0.55 Å³.
  - Diffuse anions are not converged in the field (Na⁻, Al⁻, Ag⁻: the field-step change is 20–60 %) and are
    excluded.
  - Exact correction per configuration by Woodbury, checked to 3.5e-14 against brute force.
  - Au: α(Au⁻) 123.7, α(Au⁰) 36.50 (exp 36 ± 3), α(Au⁺) 13.45 bohr³, so ρ = 3.39 and 0.37.
  - Frozen step-12 weights: δE 0.1–0.9 eV on the Au⁻ configurations. T5 −961.8 → −960.4 (seed 42) and
    −659.5 → −957.0 meV (seed 43).
- **Built, SPEC 28, test-first: `state_polarizability`.**
  - The exact per-configuration Woodbury correction on the selected set (`trace_redox/polarizability.py`;
    data `trace_redox/data/ion_polarizabilities.json`).
  - 8 tests: the brute-force re-solve to 1e-12; the isolated-dipole limit; unit ratios bitwise inert; forces;
    validation.
  - The module reproduces the independent probe exactly. Cost unmeasurable (2.34 s against 2.34 s).
- **Correction: the Au₂/MgO "over-polarisation" against Hirshfeld was a partition artifact.**
  - The falsifying check: the model's total charges against Hirshfeld for O, Mg and Al show the same 2.3–2.5
    fluctuation ratio (and ~5 in the means) as the Au polarisation (2.7–2.9).
  - The claim is withdrawn. The polarisation-magnitude criteria of steps 14–17 (P3, Q1, I1, S3) were amended to
    reported quantities before any of those runs.
  - The isolated Ag₂ polarisability, the Δ_ion spread and the penetration error are unaffected.
- **SPEC 28 made exact together with SPEC 26** (step 17 runs both): the moments include the carriers' field
  correction. A first version omitted it; a brute-force test of the combination caught it before any run.
  12 tests.
- **Fifth NaN** (step 13 seed 43, epoch 24, at load ~8.5):
  - the O and Mg width gradients and one head output are NaN; the rest of the electrostatics is finite;
  - resumed from epoch 23 with the new guard.
  - **Built:** `trace_redox/nanprobe.py`, gradient hooks that bracket the faulty operation in the failing pass.
    The guard enables them. 2 tests.
- Suites: redox 682 passed; the rest 1039 passed, 4 skipped.

## Round 25 (2026-10-18, late afternoon): induction by the carriers' true fields (SPEC 26)

- **Built, SPEC 26, test-first: `carrier_induction`.** The carriers' coupling to the bound moments uses the Gauss
  field factor of their first-principles densities instead of the Gaussian. The correction is exact in the
  configuration energies: one-body E1 into the ladder, two-body E2 into the pair slot.
  - Tests: 8 new (the split equals the dense formula for every configuration; the tables match the integral; a
    Gaussian carrier gives zero; forces −∇E).
  - Field-factor tables for 33 carrier states were added to the ion-density library.
- **Measured on frozen step-12 weights:**
  - Au⁻ E1 +0.19 eV (median), Au⁺ +0.008, Mg⁺ +0.16;
  - populated pair terms ≤ 0.27 eV. The 2.26 eV maximum is an exotic Al(Q = −4)/O(Q = +5) pair, checked before
    any claim.
- **Force training checked:** the force-loss gradient in the weights (a second derivative of E through the
  tables, gammainc and the solves) equals the Richardson central difference to 1e-7 relative (new test). A first
  version of the test failed at 1.8e-5 relative. That was O(h²) truncation along a direction of norm 247, measured
  over 5 step sizes and the same with the induction off; it was not a bug.
- **The non-reproducible NaNs, localised from the saved gradients:**
  - all three dumps have NaN on the free-charge widths' gradient, exactly on the elements present;
  - in step 11 nothing else is NaN;
  - so it entered inside the backward of the σ_f-only plane-wave block, fed by finite gradients.
  - Hypothesis: a transient Accelerate fault under oversubscription (not reproduced at load 7–10).
  - The guard now recomputes a non-finite sub-step exactly, from its recorded gradients and RNG state, and stops
    only if the fault reproduces (`experiments/benchmarks_4g/nanguard_retry.py`, 4 tests).
- Suites: redox 660 passed; the rest 1035 passed, 4 skipped.
- **Cost:** 5.1 s → 2.56 s per energy and forces, against 1.56 s without it. The Chebyshev bases are now built
  once per pair set (one matmul per table type), and a duplicated neighbour search was removed.

## Round 22 (2026-10-16): one charge model for all chemistry; step 9's seed split; Pauli between carriers

- **Theory** (docs/ELECTROSTATICS_THEORY_2026-10-16_universal_charge.md), with three exact results:
  - A: the two-state charge functional and its bond hardness;
  - B: one carrier is exact tight binding (lattice von Weizsäcker);
  - C: why covalent networks explode in configuration space.
- **Carbon chain, measured.** The refused call replays with 1451 configurations; the lowest is a
  charge-ordered C⁺/C⁻ chain. Ruled out: the learned shifts, the widths, over-screening. The cause: the
  configuration energies lack the covalent-bond energy loss.
- **Built, SPEC 19.**
  - `carriers='covalent_units'`: units of ≤ 3 atoms are solved exactly (full CI of the minimal basis);
    larger covalent units hold sign-definite carriers. The chain now trains.
  - `exclusion_bound`: Parr–Pearson anchors of Π in the bound sector.
- **Step 9's seed-43 replicate.** The undoped offset reproduces (−72 meV). Doped Au₂ flips to Au₂²⁻ (T5
  125 %). K and I_eff are equal across seeds, so this is an identifiability split.
- **Built, SPEC 18.5:** `exclusion_pairs`, Pauli repulsion between carrier units when either is a closed
  shell (Au⁻Au⁻ σ²σ*²). Step 11 (seeds 42 and 43) is running.
- **Probes:** probe_transfer_margin now passes pairs and units; new probe_carrier_counterfactual and
  probe_doped_au2_gap.
- **Built, SPEC 20:** `carrier_band`, a band unit for delocalised carriers.
  - Cation-site extra-electron states go to one unit: the lowest orbital of h = diag(ε) − τ, with Harrison
    s–s hopping × a bounded learned factor, holding ≤ 2 electrons.
  - Derivatives are exact to third order: two Newton steps on the bordered eigen-equation.
  - Counterfactual on step 9 seed 42: the band bottom lies ~5 eV below Al²⁺, and every probed doped cell
    holds Au₂⁻ + 2 band electrons.
  - Step 12 (seeds 42 and 43) launches after step 11.
- **Redox suite:** 577 passed.
- **Step 11 seed 43 (2026-10-17): negative.** The pair term does not remove Au₂²⁻ (trained gap 1.3–2.1 eV
  against a 0.45 eV term); T5 129 %. The undoped offset reproduces again (−45 meV). Seed 42 (resumed after
  a non-reproducible NaN) is still training. Step 12 (band unit, seeds 42 and 43) launched 03:33.
- **Step 11 seed 42:** Au₂⁻ (P1 pass), T5 −766 meV (82 %), undoped offset −157 meV (fail), per-atom r +0.909.
  Step 11 is negative for its purpose: each seed keeps step 9's basin.
- **Step 12** relaunched 03:50 after a band-unit bug (no band without cation sites; Au-atom prior). Test
  added; redox suite 578 passed.
- **Built, SPEC 18.6:** `exclusion_anions='in_situ'`. Anion densities come from the model's own in-situ
  removal energies, replacing the learned I_eff.
  - Measured: surface O²⁻ 9.8 eV against bulk 11.2–11.5 (learned 9.29); the counterfactual is a drop-in.
  - The anion-hole extension is behind its own flag, off: it releases O–O frame overlap and is spurious.
  - The anion κ table is widened to 1–40 eV.

## Round 21 (2026-10-16): steps 8 and 9 scored; exclusion closes the Au₂/MgO class offset

- **Step 8** (frame widths, SPEC 17.10):
  - undoped Au₂ is covalent (p(Au⁰) 0.594); five of six compressed dimers are fixed; the force RMSE falls
    by a third;
  - T5 −486.7 meV (52 %): F5′ and F1 fail. Part of step 6's wetting energy rested on an over-ionic Au₂.
- **Step 9** (exclusion, SPEC 18) passes every pre-registered criterion:
  - **T5 −881.2 meV (95 % of DFT −931.6; 4G-HDNNP −1016.4)**;
  - undoped class offset −33 ± 10.5 meV (step 6 −315);
  - test E RMSE 0.94 / 0.98 meV/atom (step 6 1.94 / 1.38);
  - doped-upright electron on the far Au, as DFT; per-atom r +0.874;
  - learned K 7.74 Ha, I_eff(O) 9.29 eV.
  - The predicted over-ionisation did not occur.
  - **Seed replicate (seed 43):**
    - the undoped offset reproduces (−72 meV);
    - the doped Au₂ charge state does not (Au₂²⁻; T5 −1168 meV, 125 %).
    - So the 95 % is one of two outcomes. Hypothesis: three integer donor electrons with nowhere
      delocalised to go.
  - Trained on 300 structures; the paper used 90 % of its data.
- `probe_transfer_margin.py` now reads the module's free-charge widths and separates the exclusion from
  the learned shifts. Its step-6 output is unchanged. New `probe_compressed_dimer.py`.

## Round 20 (2026-10-15): step 7 (negative), the energy-weight test, frame widths, exclusion by closed shells

**Measured.**
- **Step 7** (core + carrier): a negative result. The transfer vanished (Al stays Al²⁺) and T5 fell to
  −80 meV.
  - Reading: a diffuse carrier inside an O²⁻ cage gains charge-penetration energy with nothing to oppose
    it.
  - The compressed-Au₂ force errors (15.5 eV/Å with a neutral Au₂) are extrapolation of the repulsive
    wall, not the collapse.
- **The energy-weight test** (step 5, energy weight ×100): the undoped class offset stays at −303 meV.
  The loss weight is not the cause.

**Built, test-first.**
- `free_width_rule='frame'` (trace_redox SPEC 17.10): neutral-frame elements carry free charge on the
  fixed ladder width (step 8, running).
- **Exclusion by closed shells** (SPEC 18, `exclusion=True`, NEW-TO-TRACE): every charge state's frontier
  density comes from the exact ladder.
  - Asymptotics: Katriel & Davidson 1980.
  - Energy: density overlap K·S with the closed-shell frames; zero at every frame.
  - Unbound extra electrons are removed. Step 9 is running.
- **Overlap tables** (SPEC 18.3): a Chebyshev series of log S in R and κ_v.
  - It reproduces the quadrature to ~1e-12.
  - Cost: 8–11 s → 0.03 s per 110-atom structure. Step 9 was restarted on it at the same seed; its first
    gradient norms are identical.
- **Relaxed exclusion** (SPEC 18.4, `exclusion_relaxation=True`, NEW-TO-TRACE): each state's frontier
  density relaxes as a rigid shell held by the unit's own α_u, to second order in closed form, with the
  overlap Hessian projected onto its positive part.
  - Au⁰ atop O²⁻ (2.2 Å, defaults): the frozen 1.35 eV relaxes by 0.50 eV, so ionising to Au⁺ relieves
    0.85 eV, not 1.35 eV.
  - Zero in centrosymmetric cages.
- `probe_exclusion_counterfactual.py`: a checkpoint's weights with the exclusion switched on, untrained.
- Tests: 32 new (`test_redox_exclusion_table.py` 23, `test_redox_exclusion_relaxed.py` 9). Suite: 1515
  passed, 4 skipped.

## Round 19 (2026-10-15): step 6 scored; core + carrier free charges (no ionic collapse, SPEC 17.9)

**Step 6** (Au's measured polarisability, 36.68 a.u.):
- A1, E1, F2–F5 and G1 pass. The undoped wetting rose from +482 to +593 meV; the T5 difference is
  −537 meV (58 %).
- F1 fails. Six compressed-Au₂ test structures (Au–Au 2.09–2.20 Å) have Au force errors of 3–17.5
  eV/Å, against an unchanged median.
- Forces equal −dE/dR at the worst one. Its lowest configuration is Au⁺Au⁻, whose attraction cancels
  the repulsive wall.

**Root cause, exact:**
- the one-electron second difference is [U_u − S₀,u] + [U_v − S₀,v] + dᵀWd;
- with the compact ionic width S₀ > U for every element (Au: 12.03 > 6.92 eV), so the integer charges
  can collapse.
- An isolated Au₂ with untrained heads is ionic at 1.9–2.5 Å (Δ²E −3.8 to −4.9 eV).

**Built:** `free_charge_model='core_carrier'`.
- The frame (formal ion core) is compact. The carrier gets σ_car = √2 K_E/(√π U_min) from the exact
  ladder, so Δ²E ≥ 0 by theorem.
- It reduces exactly to the existing search and couplings (`bound.eliminate_core_carrier`); bulk
  formal MgO is unchanged exactly.
- `BoundElimination.constant` carries the frame terms in the free energy.
- Tests: 15. Suite: 1438 passed, 4 skipped.

## Round 18 (2026-10-10): steps 2–5 scored; couplings at physical screening; Au polarisability prior

**Scores** (experiments/benchmarks_4g/README.md):
- **Steps 2 and 3:** F2–F5 pass numerically. Step 3's wetting difference is −971 meV against DFT
  −931.6. But under the pre-registered rule these are over-screening artifacts: ε∞ 7.2 / 6.5, and
  the transfer is 0/6 at ε∞ = 3.
- **Step 4** (+ the measured ε∞): E1 and F2–F4 pass. F5 fails (46 %) and F1 fails.
- **Step 5** (+ couplings): **every pre-registered criterion passes at physical screening** (ε∞ 3.16).
  - T5: doped −23.2 meV (DFT −2.7, 4G −41.0), undoped +482.2 (DFT +928.9).
  - The transfer survives at ε∞ = 3 in 6/6. τ_AuAu = 1.6 eV turns the integer Au⁺Au⁻ into a
    covalent dimer.

**Diagnosis of the remaining undoped gap** (`probe_wetting_decomposition.py`):
- the short-range part gives only +64 to +87 meV in every formal model; upright Au₂ is stabilised by
  its polarisation on the O²⁻ site;
- the models' Au polarisability is 3.8–22× too small (step 5: 0.25 Å³ against 5.435 Å³).

**Built:**
- `polarizability_prior` (open structures, α labels) beside `dielectric_prior`. Both now share
  `_response_prior_items` / `_response_prior_loss`.
- The one-configuration guard (trace_redox SPEC 17.7 amendment): with several configurations the
  thermal and coherent charge-transfer responses are not in `bound_state`.
- Tests: `tests/redox/test_redox_polarizability_prior.py` (7). Suite: 1415 passed, 4 skipped.

**Step 6 pre-registered:** step 5 plus the Au atom's α at weight 0.3 (G1: the undoped wetting rises
by ≥ 100 meV).

## Round 17 (2026-10-09): step 1 scored; the over-screening artifact; ε∞ as a training target for formal models

**Step 1 scored** (`AuMgO_redoxq_bondlaw_r5_s42`, epoch 38; experiments/benchmarks_4g/README.md):
- S1 passes: ε∞(MgO) is 7.51, down from 1346.
- S2 passes: E 3.764 / 1.190 meV/atom, F 0.0882 / 0.0943 eV/Å.
- T2, T4, T5 and P1 fail, as predicted. The free charge leaves Al and lands on O, not on Au₂.

**The over-screening artifact (diagnostic).**
- `probe_transfer_margin.py` on the epoch-2 checkpoints of steps 2 and 3:
  - step 2 shows Au⁻ in 6/6 doped test structures (through Al⁺ + 2 Al³⁺) at its ε∞ = 8.0;
  - with the bound response scaled to ε∞ = 3, no structure transfers (+1.5 to +2.5 eV).
- Pre-registered before any final score: an F4 pass that vanishes at ε∞ = 3 is reported as an
  artifact.

**ε∞ in training for formal models** (trace_redox SPEC 17.7).
- `RedoxElectrostatics.bound_state`: with integer free charges, −∂²F/∂E² = α_b + Var_p(d)/kT. The model's
  response is therefore the bound sector's, up to e^{−gap/kT}.
- Checked against central differences of the full formal free energy: 2.4·10⁻¹¹ relative.
- flashace rebuilds observables from it for formal anchors only, checked against `energy_bound`.
- train.py:
  - `dielectric_weight` is allowed with `anchor='formal'`;
  - new `electrostatics.dielectric_prior: {file, weight}`: ε∞ labels on 3-D periodic structures
    with no energies, one forward per optimizer step, outside every energy, force and validation loss;
  - refused under data parallelism.
- Target for MgO: ε∞ = 2.95 (Sun et al., arXiv:0704.2267, Table I; 2.96 in their Table II).
- Tests: `test_redox_bound_state.py` (7), `test_redox_dielectric_train.py` (8),
  `test_redox_couplings_train.py` (1).
  - The last shows the coupling head trains, saves and reloads active. A detached head fails it.

## Round 16 (2026-10-09): integer charge configurations (audit C2, C4), probe P5, and a training kT schedule

All in trace_redox (SPEC §17), test first, each against an independent reference.

- **`diabatic.canonical_free_energy`**: F = −kT ln Tr e^{−H/kT} over configurations.
  - dF/dH = ρ; the second derivative is Daleckii–Krein with a cancellation-free divided
    difference, exact at degenerate spectra.
  - Checks: t = 0 gives the exact ensemble; the two-state closed form; the Robin–Day criterion
    (class II iff λ > 2|t|) emerges; derivatives against finite differences.
- **`one_electron_moves`**: Slater–Condon one-body elements with Jordan–Wigner signs, checked
  against explicit Fock-space operators, including per-unit charge ranges.
- **`configuration_energies` and `search_configurations`.**
  - Greedy start, exact-energy descent, flood fill inside a window, per-unit ranges [L_u, K_u].
  - Checked against brute force (14–28 configurations per 6 eV window).
  - Two flaws found by planted bugs and fixed: a non-terminating descent, and validity inferred
    from estimates.
  - Measured limitation: on an unphysical geometry (atoms 0.29 Å apart) a local minimum 12.8 eV
    high; on every physical P5 system both starts agree.
- **`RedoxConfig(fragment_ensemble='diabatic')`**: integer charges on every atom.
  - With couplings off it equals the enumeration over every atomic integer charge vector.
  - W need not be positive definite (the PD check is skipped).
- **`anchor='formal'`** (C4): exact isolated-ion ladders (NIST successive IEs, −EA₁, and A₂(O) =
  8.75 eV), learned bounded shifts per state, and the isolated self-energy removed (own images
  kept).
  - Widths are Shannon crystal radii with a QEq fallback.
  - Checks: an isolated ion equals E_Z(n) through the whole module; the enumeration is exact;
    forces against finite differences.
- **Probe P5** (pre-registered; benchmarks README):
  - Formal states are selected by energy in MgO and in the NaCl clusters (including the hole's
    site); the neutral chain stays neutral. Robust to A₂(O) and to the widths.
  - It fails for the covalent cation without polarization.
  - Au₂/MgO:Al transfers only under over-screening: none at ε∞ ≈ 3, with a margin of
    0.6–2.4 eV.
- **`train.redox_kT`, `redox_kT_start` / `redox_kT_epochs`** (audit C3; a training heuristic, no
  physics claim).
  - Geometric annealing to exactly the deployed kT before training ends.
  - Validation and checkpoints always use the deployed kT (a planted bug that validated hot is
    caught).
- **Fixed.** `LatticeWidthError` now advises the exact cutoff s·erfc⁻¹(10·accuracy). Its "4 s"
  advice failed at accuracy 1e-10, found by P5.

## Round 15 (2026-10-09): audit step 2 (B3, ionic free-charge sizes) — building blocks, and why the mean-field sector cannot use them

**Added: `lattice.free_charge_lattice_blocks`.** Ewald-split pair blocks between free Gaussian charges
and the bound sector: free–free, free–bound charge, free charge–bound dipole and free charge–bound
quadrupole.
- They share the R1–R3 machinery through helpers factored out of `gaussian_lattice_blocks`. That
  function is bitwise unchanged by the refactor (checked).
- The plane-wave Gram route of trace_redox needs k up to 12/s_min. For ionic widths on the Au₂/MgO
  cell that is 143 687 half-space vectors, against 1 654 for the hardness widths. The Ewald route's
  reciprocal sum depends on α alone.
- Against an independent plane-wave sum written from its formula, it agrees to 1.0–1.6e-15 relative
  (test tolerance 1e-10).

**Added in trace_redox:**
- `heads.IonicFreeChargeWidth`: σ_Z = √(2/5)·R_Z × a learned factor in [½, 2].
  - R_Z is Shannon's crystal radius, a published table, 11 ions: `constants.SHANNON_CRYSTAL_RADIUS`.
    Other elements are refused, not invented.
  - √(2/5) matches the second moment of a uniform sphere to the code's Gaussian, whose ⟨r²⟩ = 3σ²/2
    is checked by Poisson and quadrature.
  - Formal ±2 rocksalt MgO keeps 99.35 % of its point-charge Madelung energy, against 42 % with the
    hardness widths. The Madelung constant is 1.7475645946…; point charges reproduce it to 2.2e-16.
- `bound.periodic_free_charge_blocks_ewald`, `bound.on_site_self_width`, and
  `eliminate_bound_sector(..., sigma_self=)`.
  - The pair terms use one width; the same-image on-site terms use another, through an exact diagonal
    correction to J and to J^fb·B.
  - With equal widths this equals the plane-wave route to the accuracy.

**Finding (measured, with a proof sketch): compact ionic sizes are incompatible with the mean-field free
sector.**
- A convex integer ladder, which keeps isolated-ion energies exact, caps the on-site curvature at
  U_Z = IE − EA. Compact pair charges make the Madelung field of formal MgO α·K_E/d = 11.95 eV/e².
- So the alternating charge mode has negative curvature for every allowed on-site value:
  λ_min(J) = −10.8, −7.0 and −6.6 eV/e² at h = 0.5, 0.95 and 1. With the hardness widths everywhere,
  J is positive semi-definite (λ_min = 0).
- Even dilute geometries were indefinite once the bound sector screens the on-site curvature.
- The hardness rule (σ ≥ σ_H) sits exactly on this consistency boundary. That is why it is diffuse.
- **B3 therefore needs integer configurations (audit step C2), not fractional ones.**
- `RedoxConfig(width='ionic')` is refused with this reason. The building blocks above are kept for C2.

**Fixed.** When W − λD is not positive definite, a failing λ_max diagnostic (`eigh` on an
ill-conditioned W, met with compact widths) raised a raw `LinAlgError`. It now still raises the
`BoundSectorError`, reporting λ_max as NaN.

**Tests.**
- `tests/electrostatics/test_free_charge_lattice.py` (3).
- `tests/redox/test_redox_free_width.py` (13): the moment identity, the radii and their bounds, the
  refusal of untabulated elements, the Madelung energies, the Ewald route equal to the plane-wave route
  (periodic and slab, with and without quadrupoles), the on-site-only change, the proposition, and the
  refusal.
- One robustness test in `test_redox_module.py`.

## Round 14 (2026-10-09): periodic lattice blocks 2.8× faster (R1–R3)

**Why.** One training step on a 110-atom Au₂/MgO slab took 5.5 s and allocated
19.8 GiB in total (peak RSS 3.7 GiB). Profiling attributed about 3 s of it to
`gaussian_lattice_blocks`:
- the reciprocal `even`/`odd` contractions over 15 625 k-vectors, and their
  backward and second-order backward;
- the real-space kernels over all 907 500 pair-images of the image box.

**Changes.**
- **R1, the ±k half-space (exact).** `ewald._half_wave_vectors` builds one
  vector of each pair {k, −k}, with a static size. Every reciprocal block is
  w(k)F(k)·g(k·r_ij), with an even F under cos or an odd F under sin, so it is
  exactly twice the half-space sum (`lattice._reciprocal_terms`).
  - Measured change: 0.3–1.6e-15 relative to each block.
- **R2, the per-axis reciprocal grid (`EwaldParams.auto_tune`).** |n_d| ≤
  k_cut·|a_d|/2π holds per direction for any cell, so the per-axis box contains
  the whole k_cut sphere.
  - The longest vector is now used only along its own direction: (4, 4, 12) and
    2 025 vectors on the Au₂/MgO cell, against 12 everywhere and 15 625.
  - `k_max` is unchanged. A cell whose three counts agree keeps the scalar
    grid, bit for bit.
  - What changes is only the vectors with |k| > k_cut that the larger box
    happened to include.
  - Measured: forces move by 8e-12 to 1.8e-11 eV/Å on Au₂/MgO, with energies
    identical to the last digit. That is 10⁵ below the configured accuracy of
    1e-6.
- **R3, real space over a pair list within a rigorous round-off radius.**
  - The bound: every real-space factor is a difference of regularized upper
    incomplete gammas, |λ| ≤ Q(a, κ²r²) with κ = min(α, 1/s_max), and
    Q(a, x) ≤ x^{a−1}e^{−x}/(Γ(a)(1 − max(a−1, 0)/x)).
  - R₀ is the smallest radius at which every pair-image beyond it, together,
    adds at most 2⁻⁷⁰ K_E to any block entry: 21.1 Å for the Au₂/MgO widths, or
    23.8 Å with quadrupoles.
  - The selection is made on detached geometry. The kernels run on the
    [m]-length list (`lattice._near_pair_geometry`) and are scattered into the
    blocks.
  - Measured change: blocks bitwise identical to the whole box on the test
    systems. Forces change by 4e-15 to 1.6e-14 eV/Å (R1 + R3) on Au₂/MgO, the
    size of the Mac's own run-to-run noise.

**Measured on the Au₂/MgO training step** (2 threads, the same session):

| code | step | peak RSS |
|---|---|---|
| original | 5.94 s | 3.2–3.4 GiB |
| R1 + R2 + R3 | 2.14 s (**2.78×**) | 2.4–2.5 GiB |

R1 + R2 alone gave 1.56×, and R3 a further 1.72×.

**Tests.**
- `test_lattice_half_space.py` (9): the half-space holds one of each pair; the
  blocks and their position gradients equal the full grid within γ_n Σ|w F|.
- `test_ewald_per_axis.py` (9): the box holds the whole k_cut sphere,
  enumerated; the blocks move by at most the dropped tail.
- `test_lattice_pair_list.py` (12):
  - the tail bound against mpmath (an upper bound everywhere; within 6 % at
    x ≥ 10);
  - the radius meets its criterion and is minimal;
  - the blocks, forces and cell gradient against the whole box, within derived
    bounds, using the captured per-pair gradient terms;
  - the Au₂/MgO cell keeps fewer than a third of its pair-images.
- Mutation checks:
  - a factor 2(1 + 1e-9) on the half-space is caught;
  - `floor` instead of `ceil` in the per-axis grid is caught;
  - half the radius is caught;
  - a transposed scatter is caught by the electrostatics suite.
- Full suite: 1259 passed, 4 skipped.

## Round 13 (2026-10-09): exact closed-form damping for the periodic real-space sum (speed)

No change in any number beyond round-off.

### Changed
- **`special.half_integer_regularized_gamma`.** P(n + ½, x) in closed form:
  - for x < 3, the series for the top order (32 terms) and the downward
    recursion, which only adds positive terms;
  - for x ≥ 3, the upward recursion from erf(√x).
  - It is a custom autograd function. Its derivative is the analytic
    x^(a−1) e^(−x)/Γ(a), written in differentiable operations, so the double
    backward of force training is exact.
- **`lattice.real_space_damping`.**
  - It computes only the orders each block uses (it computed all four, and
    `t01` used one).
  - It computes only on the pair-images inside the real-space cutoff (`mask`).

### Measured
- **Accuracy against 50-digit mpmath:**
  - values within 32 eps over x ∈ [0, 1000];
  - first and second derivatives within 64 eps.
  - `torch.special.gammainc` itself is off by up to 1.8e-14 (≈ 80 eps).
- **Doped Au₂/MgO structure, energy and forces, 2 threads:**
  - |ΔE| = 0 and max |ΔF| = 1.2e-15 eV/Å against the `gammainc` path;
  - 2.66 s against 2.89–3.12 s (8–15 % faster end to end);
  - the damping is 2.4× faster (0.24 s against about 0.56 s).
- **A first version** that let autograd record the series doubled the
  backward pass (3.10 s against 1.47 s) and was slower overall. The custom
  backward fixed that.
- Tests: 17 new (`tests/electrostatics/test_half_integer_gamma.py`), with
  mutation checks. Full suite 1203 passed, 4 skipped.

## Round 12 (2026-10-09): a physical bond hardness (SPEC 26) — Pauli screening and the two-level anchor

Both are opt-in, and every earlier model is unchanged bit for bit. Defect A2
of `docs/ELECTROSTATICS_AUDIT_2026-10-09.md`.

### Added
- **`ElectrostaticConfig.bond_screening = (C_min, C_max)`** (`screening.py`).
  - The MEAM-type many-body Pauli screening of each split-charge channel,
    f_b → f_b S_b.
  - The rocksalt second shell, which carried 1281 of a trained MgO's
    ε∞ = 1316, is now exactly zero.
- **`ElectrostaticHeads(bond_hardness_law="two_level")`** (`bond_law.py`).
  - κ = κ_anchor · exp(L tanh(s/L)) + κ_floor, with
    κ_anchor = (Δχ² + 4h²)^{3/2}/(4h²) and h = 1.42 ħ²/(m r²).
  - Unidentified bonds keep a physical hardness.
  - It certifies an ε∞ bound for every learned value (Theorem 1).
- **Atomic IE/EA tables** moved into `trace_electrostatics.constants`.
  `trace_redox.constants` re-exports them, so there is one source.

### Measured
- Rocksalt MgO, untrained heads, three seeds:
  - free law: ε∞ 454–596;
  - plus screening: 100–117;
  - plus the two-level law: 10.6–12.6 (experiment ≈ 3).
- Tests: 10 for screening and 12 for the law, including a brute-force check
  over all periodic sites and a mutation check of the anchor coefficient.
  Full suite results are below.

### Not done
- A training run with these settings (pre-registered next).
- Pruning screened bonds from the solver, a speed-up for ionic solids.

## Round 11 (2026-10-08): TRACE-Redox — finite, exact derivatives at degenerate spectra

`trace_electrostatics` is unchanged. In `trace_redox/free_solver.py`, two
defects of the differentiable free-charge step, found by tests written first
(`tests/redox/test_redox_free_solver.py`, 5 new; red before, green after).

### Fixed
- **NaN gradients at a degenerate spectrum.**
  - Before: the graph-attached Newton step differentiated `eigh` (the Schur
    complement on μ in `_constrained_step`; the pseudo-inverse branch of
    `_newton_step` when W is singular).
  - At a degenerate spectrum (two identical, uncoupled molecules) the backward
    pass's 1/(λᵢ − λⱼ) is 0/0. The gradients of b, W and Ê were all NaN, while
    the energy is smooth (finite differences well defined).
  - Now the Newton operator is detached and only the residual carries the
    graph. This is exact to second order (SPEC §15, Derivatives). The residual
    and the operator columns are solved separately, so the operator never
    enters the graph.
- **∂E/∂W with singular W.**
  - Before: z drifts along null(W), and the dual gave ½ccᵀ − ½nnᵀ instead of
    the primal ½ccᵀ.
  - Now the step starts from z = c.
- **Checked:**
  - envelope identities dE/db = c, dE/dW = ½ccᵀ, dE/dÊ = p to 1e4·eps;
  - Hessian in b against the closed-form constrained linear response to 1e-9,
    at the degenerate spectrum;
  - the existing closed-form Hessian and dead-zone tests unchanged;
  - redox suite 228 passed.

### Not established
- Whether this caused the NaN that stopped `AuMgO_redoxq_r5_s42` at epoch 5.
- A resumed reproduction with the same batch order but different dropout masks
  did not trigger. The exact rerun from scratch, on the pre-fix code with a
  finiteness guard, is the test.

## Round 10 (2026-10-08): why average? — the exact canonical configuration ensemble between molecules

`trace_electrostatics` is unchanged. In `trace_redox`,
`fragment_ensemble='exact'` is the new default; `'two-level'` keeps §12.

### Derived (trace_redox/SPEC.md §14; NEW-TO-TRACE)
- **The exact problem is convex.** The exact classical canonical ensemble is
  convex in (p, x(N) = p y(N)) through the perspective construction, with no
  certificate. Averaging between molecules is forced only by its size, |𝒩|·U.
- **For two molecules, averaging restricts nothing.** The averaged
  (two-level) ensemble is the restriction to molecule-conditional charges, and
  for m = 2 that restriction is empty.
- **The error of averaging.** For the same ensemble, G_exact − G_2L =
  ½ tr(S̃ Cov(y)) exactly. Molecules whose charge does not fluctuate contribute
  zero.
- **Theorem (canonical only).** F_2L ≤ F_exact ≤ F_2L + ½ tr(S̃ Cov*).
  - Lower bound: Jensen's inequality and S̃ ⪰ 0.
  - Upper bound: the Gibbs variational principle.
  - It holds in every measured case.
- **Where averaging is physics:** inside bonded molecules (delocalised
  electrons). Between weakly coupled molecules it is not; there, the
  configuration set is the diabatic basis of a charge-transfer Hamiltonian.

### Added (trace_redox/SPEC.md §15)
- **The exact canonical configuration ensemble.**
  - Each molecular charge configuration is solved exactly, on the atom-level
    certified W̃, with one constraint per molecule.
  - A one-molecule partition is the atom-level model exactly.
  - Configurations are selected by the canonical two-level probabilities
    (≥ 1e-12, C²-smooth inclusion weight).
  - Partitions are mixed linearly.
- **`free_solver.differentiable_constrained_solution`.** One multiplier per
  molecule: the Schur complement on μ, dead-zone directions dropped with
  per-molecule bisection, and molecules at their capacity fixed exactly.
  First and second derivatives are exact (finite differences:
  1.7e-11 / 1.6e-10).

### Measured
- **Against the independent enumeration:**
  - H₂O···H₂O⁺ at 4 / 6 / 10 / 30 Å: 9e-14 eV (the two-level was off by
    0.049 / 0.113 / 0.031 / 0.0035);
  - H₂O···H₂S⁺: ≤ 2e-14;
  - Na···Cl, 2.36–100 Å including the crossing: ≤ 4e-14 (the two-level was
    21 meV off at the crossing).
- **The hole localises.** ⟨N_A⟩ = 0.87–0.92 at 4–6 Å, where the two-level
  had 0.52–0.54. The enumeration confirms the marginals to 1e-8.
- **Au₂/MgO cost:** 1–3 solves per structure, and the same cost as the
  two-level.
- **Tests:** 39 fragment tests; six planted bugs (M50–M55), each caught.
  Suites: redox 218 passed; full repository 1152 passed, 4 skipped (Mac).
- **Rigour pass**, five more tests against independent references; redox 223
  passed:
  - two partitions by value (2e-13 eV);
  - dications (4e-15 eV);
  - permutation invariance;
  - the inclusion band at kT = 0.1 (derivatives inside it, continuity at its
    edge);
  - planted bugs M56–M58, each caught.

### Fixed
- **Checkpoints now store the resolved `RedoxConfig`** (every field).
  - Before, the raw YAML block was stored (e.g. `{}`), so a model reloaded with
    whatever trace_redox's defaults were at load time. Two defaults changed this
    round (`bound_elimination`, `fragment_ensemble`).
  - Older checkpoints with a partial block still load, with the current
    defaults filling the gaps.
  - The training test asserts the stored config equals `RedoxConfig(kT=0.1)` in
    full.

## Round 9 (2026-10-08): TRACE-Redox 6× faster — the condensed bound-sector elimination

`trace_electrostatics` is unchanged. In `trace_redox`,
`bound_elimination='condensed'` is the new default; `'dense'` keeps the
previous route as the reference.

### Added (trace_redox/bound.py `_CondensedK`; contract trace_redox/SPEC.md §13)
- **Exact elimination without forming K** (NEW-TO-TRACE).
  - In the variational basis, K = blockdiag(diag κ, P) + Vᵀ[[H, T], [Tᵀ, 0]]V
    with V = blockdiag(B, I). The SQE bond block is a diagonal plus a rank-n
    term through the incidence B.
  - Woodbury on the bond block (F = I + G D⁻¹ Gᵀ, eigenvalues ≥ 1) and a
    Schur complement on the multipoles give K⁻¹ exactly.
  - Only [n, n] and [n_p, n_p] factorisations: O(n³ + n_p³ + n n_b k) against
    O((n_b + n_p)³). Plain autograd, at that cost, gives exact first and second
    derivatives.

### Measured
- **The diagnosis.** On Au₂/MgO (110 atoms, 66 neighbours per atom within
  5.29 Å) the bound sector has 4,079 unknowns and a dense K of 133 MB. The
  double backward through its Cholesky was 66 % `mm` and triangular solves.
- **One training step** (E, forces, d(ΣF² + 0.1E)/dθ):
  10.3–11.3 s → 1.73–1.95 s (Mac CPU, 6 threads). The elimination now takes
  0.03 s.
- **Agreement with the dense route:**
  - E_b0, u0, W, y0, forces and the force-loss gradient within
    3 cond(K)·eps on open, quadrupole, field, 3-D periodic and slab cases;
  - Au₂/MgO: |ΔE| ≤ 1.7e-14 eV, |ΔF| ≤ 3.6e-15 eV/Å.
- **Planted bugs:** five (M45–M49), each caught.
- **Suites:** redox 205 passed.

### Operations
- **The Au₂/MgO redox training was lost.**
  - It had reached epoch 2 on the dense route when the Mac rebooted on Oct 4.
  - It has been relaunched on the condensed route, with batch 8 (identical to
    `AuMgO_elec_r5_s42`; the per-structure memory is no longer 4 GB).
- **The GPU hosts** could not be reached on 2026-10-08.

## Round 8 (2026-10-03): TRACE-Redox — the two-level certified fragment ensemble

`trace_electrostatics` is unchanged. In `trace_redox`, `fragments='bonds'` is
the new default; `fragments='none'` is the Round 7 model; its tests pass unchanged. It is not
claimed bit-identical, because `free_solver` now stops on a potential-unit
residual. Suites:
191 redox tests; full repository 1125 passed, 4 skipped.

### Added (trace_redox/fragments.py; contract trace_redox/SPEC.md §12)
- **Molecules carry integer total charges** (NEW-TO-TRACE). Each molecule
  holds a canonical ensemble over its total charge N, with an intra-molecular
  mean field at fixed N. Molecules couple through the certified residual
  S = W − blockdiag(D).
- **Block convexity certificate** (NEW-TO-TRACE).
  - D_F = θ_F (M_FF)⁻¹ with M = W⁻¹, where θ_F = c/(1 + Σ_G ‖K_FG‖_F) and
    K_FG = M_FF^{−½} M_FG M_GG^{−½}.
  - Guarantees: S ⪰ (1−c)W and D_F ⪯ W_FF.
  - (M_FF)⁻¹ is the Schur complement W_FF − W_FR W_RR⁻¹ W_RF.
- **Molecules from a smooth bond graph** (NEW-TO-TRACE construction;
  ESTABLISHED radii).
  - Bond strength s_b is a C² switch between R_i + R_j + 0.25 Å and
    + 0.60 Å, with Cordero et al. 2008 covalent radii.
  - The energy is the π-weighted ensemble over percolation partitions:
    C² in the positions, exact where s ∈ {0, 1}.
  - Exact reductions: bonds inside certain components are dropped, and
    parallel bonds merge as 1 − Π(1 − s).
- **Solver.**
  - The convex dual, in the variables (z, μ, δ^P) with Σ π_P δ^P = 0.
  - The inner fixed-N problems reuse `free_solver`.
  - Analytic outer Hessian: the molecular susceptibility Σ_N p χ^N + Cov/kT.
  - One graph-attached Newton step gives exact first and second implicit
    derivatives.
- **Adaptive molecular charge windows.** A window is widened wherever an edge
  state is populated above 1e-8.

### Measured
- **Against the molecule-exact enumerated target** (untrained heads,
  kT = 0.01 eV, Q = +1; draft numbers, see the correction below for the
  canonical ones):
  - H₂O···H₂O: 0.056 / 0.120 / 0.038 / 0.010 eV at 4 / 6 / 10 / 30 Å; the
    mean field is 0.221 / 0.859 / 1.116 / 1.359 eV off.
  - H₂O···H₂S: within 1e-4 eV, with the hole an integer on H₂S.
- **Single molecule** (B2): 0 ≤ E_none − E_bonds ≤ ½(1−c) Σ r Var. On water
  the gap is 0, 3.3e-5 and 6.1e-6 eV for Q = 0, +1, −1, against a bound of
  1.5e-4 eV.
- **Derivatives and smoothness.**
  - Forces and force-loss parameter gradients match finite differences
    (1e-6 relative), including through a breaking bond.
  - At the switch ends: second differences ~3e-14 eV, and force changes
    ~5e-8 eV/Å over 2e-6 Å.
- **Robustness and cost.**
  - 60/60 real geometries converge.
  - Au₂/MgO (110 atoms, periodic) takes 2.48 s with forces, against 1.66 s
    for the mean field.
- **Mutation checks:** ten planted bugs, each caught (§12.11).

### Corrections found while implementing (SPEC §12.12)
- The multiplicative bond criterion needed neighbour lists past 4 Å, so it is
  additive now.
- A single-molecule shortcut jumped by 4.2e-5 eV at a switch onset, so it was
  removed.
- The ψ-form dual stalled by cancellation, so the solve was reparametrised
  to (z, μ, δ).
- The KKT residuals are now in potential units (eV), which fixed a
  near-singular W in a dense crystal. This also applies to `free_solver`.
- The certificate uses the Schur complement.
- The inner Newton step has a pseudo-inverse fallback.
- The coverage check uses the two largest radii, not twice the largest.

### Corrected the same day, found training on Au₂/MgO (trace_redox/SPEC.md §12.12–12.13)
- **The total charge is now canonical** (NEW-TO-TRACE).
  - The draft's product of molecular ensembles fixed only Σ⟨N_F⟩ = Q. Where
    the Au₂ bond was in its switch, it let the 108-atom MgO slab take ±1 with
    3% probability while Au₂ stayed neutral: charge from a reservoir that a
    closed system does not have.
  - Each partition is now the canonical ensemble at Σ_F N_F = Q, evaluated
    exactly by convolving tilted molecular distributions:
    A = μQ + Σ Ω_F − kT ln Pr_q(ΣN = Q), exact for every μ.
  - The Hessian gains the cross-molecule covariance.
  - Measured:
    - H₂O···H₂O⁺ is 6.5–7 meV closer to exact at every R (kT ln 2);
      30 Å: −0.010 → −0.0035 eV.
    - Na···Cl at the crossing: 28 → 21 meV; at 8 Å: 1.9e-4 → 1.1e-6 eV.
    - The slab's charge is now the exact mirror of Au₂'s.
- **Partitions of negligible weight.** A dumped training structure failed
  ("windows did not settle in 64 enlargements") after 64 re-solves:
  - one Au atom's merged absent probability underflowed to 0.0, giving an
    enumerated partition of weight exactly 0;
  - its unconverged potentials put population on the window edge;
  - the unweighted window test widened the window to N = −65.

  Fixed three ways:
  - a quantified floor (PARTITION_WEIGHT_FLOOR = 1e-30, moving E by
    < 1e-28 eV);
  - exact absent weights, a instead of 1 − (1 − a), which lost every
    a < 1e-16;
  - π-weighted widening.

  The structure now takes one solve (1.4 s). This was also why the first GPU
  attempts ran at > 13 s per call.
- **Tests:** 26 fragment tests and 17 planted bugs (M38–M44 new), each caught.
  Redox suite: 199 passed in 105 s (was 191 in 163 s); full repository 1133
  passed, 4 skipped.

### Remaining physics limitations (stated)
- No coherent inter-molecular hopping, so a hemibonded (H₂O)₂⁺ shares its
  charge thermally, not quantum-mechanically.
- No spin.
- `orbitals=True` is refused with fragments.

## Round 7 (2026-10-03): TRACE-Redox — certified SIC, periodic boundaries, training

`trace_electrostatics` itself is unchanged. `train.py`, `flashace/checkpoint.py`
and `flashace/calculator.py` each gain one branch that is inert without an
`electrostatics.redox` block. Full suite in one session: 1107 passed,
4 skipped.

### Added (trace_redox; contract trace_redox/SPEC.md §2a, §8a, §11)
- **Certified self-interaction removal, now the default** (NEW-TO-TRACE).
  - **Theorem:** W − diag(r) ⪰ 0 for r_u ≤ 1/Σ_v|(W⁻¹)_uv| (Collatz–Wielandt),
    so the largest convex-compatible removal is moved into the populations.
  - **Separated Na···Cl:** 0.000 e at 15–100 Å (v1: 0.30 e). The ionic bond
    stays (1.000 e at 2.36–6 Å) and the crossing falls at 9.43 Å
    (K_E/(IE − EA) = 9.434 Å).
  - **Against the exact enumerated ensemble at 60 Å:** within 5 meV for
    Na···Cl, Na···Na⁺ and H···H⁺. v1 was off by 0.30, 0.52 and 1.55 eV.
- **Periodic boundaries** (3-D and 2-D slab). The free blocks are built in the
  plane-wave Gram form, equal to the assembly's Ewald blocks to 1e-14. The
  large-cell limit converges as L⁻⁵, and the tinfoil surface term is verified
  with its sign.
- **Training:** an `electrostatics.redox` YAML block, checkpoint round trip,
  and the ASE calculator. Smoke-trained on 110-atom periodic Au₂/MgO.

### Measured
- **Ag₃ at PFN sizes** (held-out test of 2013 structures):
  - TRACE + elec has a pooled force RMSE of 0.0159 eV/Å, against PFN
    NequIP+QEq 0.0204 (digitised), NequIP+E_ele 0.0252 and 4G 0.031;
  - the local arm is at 1.892 eV/Å.
- **NaCl local:** 0.0553 eV/Å, against NequIP 0.0104 (a capacity result; the
  elec arms are training).

- **Charge states beyond +1** (`max_charge` K ≤ 4).
  - NIST cumulative-IE anchors, with gaps that are positive by construction
    (strict convexity).
  - Absent states (k > Z) are +∞, and the capacity is Σ min(K, Z).
  - Measured: Mg(H₂O)₄²⁺ puts +2.000 on Mg with K = 2, against +1 plus a
    ligand hole with K = 1.

### Remaining physics limitation (stated)
The per-unit certificate cannot make the transfer between distant
POLYATOMIC fragments exact, because that needs correlated ensembles at the
fragment level.

## Round 6 (2026-10-03): SPEC 25 implemented as a separate package, TRACE-Redox

Nothing in `trace_electrostatics` changed except `SPEC.md` §25 (status and
errata). The implementation is `trace_redox/` (contract `trace_redox/SPEC.md`,
70 tests in `tests/redox/`, 13 planted bugs all caught).

### SPEC 25 errata found while implementing
- **(25.1) sign:** the drafted supremum was +∞; corrected to the conjugate of
  Ω(ν) = −kT ln Σ e^{−(Ê − νn)/kT}. 25.8 A's softmax had the same sign error.
- **Convexity of the integer state energies:** the Corollary holds for the
  true E_n. Lemma 1 acts on Ê = E − ½J n², which needs
  IE − EA > (1−λ)J_uu. Otherwise a neutral unit disproportionates internally
  (measured with σ_free = 1.5 Å for Na, Li, Al, Si, Cu, Au). Fixed by a
  hardness-derived free-charge width (QEq relation, Rappé–Goddard 1991) and a
  gap parameterised positive.
- **λ_max ≈ 0 in molecules:** 4.6e-5 (benzene) to 0.135 (NaCl dimer). The
  "λ → λ_max for hybrids" note is not practical, so v1 uses λ = 0.
- **R_c(Na···Cl) = 9.434 Å**, not 9.44.

### Measured limitation (trace_redox/SPEC.md §8)
The mean-field free sector has the semilocal-DFT delocalization error. At
h = 0.5, kT = 0.01 eV, a separated Na···Cl keeps 0.30 e at R = 100 Å (the
closed-form two-site law gives 0.281 e at T → 0). A diffuse free charge
(h = 0.2) removes the error, but then the ionic bond at contact (0.001 e at
2.36 Å) is lost too.

## Round 5 (2026-10-02): competitive benchmarks, batched solve, charge states from files

### Proposed (design, not implemented)
- **SPEC 25: free charge, the derivative discontinuity and the Fermi level.**
  A convex free-charge sector, the ensemble (convex-envelope) energy of
  charge-state-resolved units, screened by the SPEC 1 dielectric.
  - The convexity–exactness theorem (25.2)–(25.3) with its λ_max bound.
  - Answers to identifiability, geometry dependence, spin (flat plane) and the
    solver.
  - Prior art checked: redoxSQE, MACE-POLAR-1, Baldwin et al., QEq-based ML
    potentials.
  - **Correction recorded:** "no spurious transfer" is exact only at λ = 1,
    outside the convex range.

### Added
- **SPEC 24: independent open structures in ONE block-diagonal solve.**
  - `assemble(..., structure_id)`, `electrostatic_energies`,
    `ElectrostaticModule.energies_by_structure`; `forward_batched` uses the
    route automatically.
  - Pair blocks are masked across structures. The same masked blocks go to
    the solver as overrides, so the minimised and reported functional are one
    operator. Per-structure energies come from the exact partition (24.1) at
    the implicitly differentiated minimiser.
  - Equal to the per-structure loop to 1e-10 in energy, 1e-9 in forces and
    1e-7 (relative) in parameter gradients of force losses. Each of five
    planted bugs fails the tests.
  - One training step, per structure:

    | | Ag3 | C10 chain | NaCl |
    |---|---|---|---|
    | GPU | 77 → 6.5 ms (1.9 ms at batch 128) | 93 → 9.5 | 87 → 14.1 |
    | CPU | 18.7 → 3.0 ms | 26.7 → 15.0 | 29.2 → 19.4 |

- **Charge states (SPEC 21) declared in extxyz files.**
  `arrays['charge_state_fragment']` holds the per-atom labels;
  `info['charge_state_charges']` holds an S × F integer matrix, row-major.
  - Validated: integer charges, every state summing to the net charge, labels
    0..F−1 all used, S ≥ 2, F ≥ 2. Both forms together are refused.
  - The calculator cache tracks both keys.
  - Equal bit for bit to the in-memory list form.
- **Training on reference partial charges** (`partial_charge_weight`,
  `partial_charge_key`; see the Round-5 entry under the observables below).
- `tools/electrostatics/profile_batch_step.py`: the cost of one real batched
  training step, the host synchronisations by source line (CUDA), and the
  top operators.

### Changed
- **`implicit._pcg` reads one scalar pair per CG iteration instead of three**
  (finite curvature, positive curvature, convergence). The checks and
  messages are identical; the update is kept only once the checks pass.
- `flashace.model`: the per-structure atom ranges come from one `bincount`
  (`_structure_atom_lists`), not one `nonzero` per structure. Periodicity
  flags are read to the host once per batch.

### Measured (benchmarks, see experiments/benchmarks_4g/README.md)
- The first-round 4G-benchmark runs were under-trained (40 epochs), and the
  loss was force-dominated.
- Ag3, 200 epochs at energy weight 10: test errors (E in meV/atom, F in
  meV/Å):

  | | before | after |
  |---|---|---|
  | Ag3+ | 3.55 / 8.4 | 0.48 / 3.8 |
  | Ag3− | 11.3 / 23.4 | 4.0 / 24.6 |

  The charge-state energy offset was under-training, not architecture.
- Au2/MgO:Al: TRACE's polarization is a dielectric and screens the dopant
  within 5.5 Å. DFT moves 0.17–0.36 e onto Au2 more than 13 Å away, a
  donor → acceptor redox. This is the SPEC 21 physics, not SPEC 1's.

## Round 4 (2026-10-01): new stack, GPU, learned coupling, training gaps

### Verified

- **The newer pinned stack.** Python 3.12, torch 2.11 + CUDA 12.8, e3nn 0.6.0,
  numpy 2.5, ase 3.29 (on the GPU host): the full electrostatics suite passes. Two
  tests had asserted rounding instead of physics and are now bounded by
  machine epsilon:
  - the Makov–Payne five-point stress budget (1.0e-11) sat **below** the
    stencil's own round-off floor, 1.5·ε·|E|/h = 1.63e-11, and passed on
    macOS by luck (9.2e-12 vs 1.09e-11 on Linux); the budget is now that floor
    at 4 ulp;
  - the Thole pair-block reconstruction asserted bitwise equality and differed
    by one ulp (8.9e-16 at 6.6).

  Outside the package, the LAMMPS padding test asserted bitwise invariance and
  differs by ≤ 8 ulp on torch 2.11; it is now bounded at 64 ulp.
- **CUDA parity, float64, every route** (RTX 5060 Ti). Dense open, matrix-free
  open, dense periodic, matrix-free Ewald and PME, slab, metal. Energy, forces,
  charges, α, ε∞ and Z* agree with the CPU to ≤ 1e-14 relative. The exception
  is the matrix-free open route at 3.6e-11: CG stopped at iteration 36 on one
  device and 37 on the other, both under the 1e-10 residual, so this is the
  solve tolerance, and the energy agrees to 6e-16.

### Diagnosed (not yet fixed)

- **Why the matrix-free iterations grow with size.** The preconditioner lacks
  the long-wavelength Coulomb. With the shipped preconditioner, the largest
  eigenvalue of the preconditioned operator grows as a + bL²: 2.8, 4.9, 7.7 and
  10.4 at 64, 216, 512 and 1000 atoms of rattled NaCl. That is the 1/k² law,
  giving iterations 32 → 54. An oracle with the exact periodic charge kernel
  in the Woodbury factor flattens the condition number at about 7 and the
  iterations at 24 → 29.
- **What this rules out.** No fixed-rank correction (deflating a few low-k
  modes) can stop the growth: the modes to correct are all |k| < k_c, and
  their number is proportional to n. The fix must be Fourier-diagonal
  (particle-mesh) and must keep the preconditioner SPD (SPEC 17.6).

### Practical: foundation-dataset compatibility and cost (OMat24 sample, CPU)

- **What large datasets actually carry** (official docs): OMol25's training
  files have energy, forces, total charge and spin multiplicity; OMat24 has
  energy, forces and stress. Neither has dipoles, α or Born charges. Design
  choices now assume exactly these labels.
- **OMol25 charge and spin are readable.** `spin_key: spin` reads
  `info['spin']` as 2S+1 (the fairchem convention; singlet = 1), checked by
  electron-count parity. The calculator's cache now sees spin changes.
  Open-shell structures given to closed-shell heads are counted and flagged.
- **`solver='auto'`** picks the route per structure: matrix-free for 3-D
  periodic cells and molecules over 128 atoms, dense otherwise (and for slabs
  and dispersion). It is bit-identical to the route it picks. Training step on
  CPU, dense vs matrix-free:
  - OMat24 cells of 15–80 atoms: 0.38–2.99 s vs 0.29–1.17 s;
  - molecules of 240/348 atoms: 5.6/15.3 s vs 2.0/2.8 s.
- **Defaults for NEW training runs:** `solver: auto`, the SPEC 23 contact
  law, and no solver-seed loss under the converged solve, all written into the
  checkpoint. The library defaults are unchanged.
  - Why no seed loss: on trained water models the trained seed saved 3–9 % of
    CG iterations (11.7 vs 12.0; 10.8 vs 11.9). Its separate assembly cost
    44 % of the electrostatic overhead per training step (C10H2/C10H3+,
    1 thread: 18.8 → 10.6 ms on top of a 32.1 ms local step).
  - The physics seed (Ohm's law) remains the starting point, and the converged
    energy does not depend on it.
- **Robustness on real periodic data.** 300 OMat24 cells (82 elements,
  10–80 atoms) all ran, with 23 CG iterations at the median (39 max) and
  0.15 s per cell. On the most triclinic cells (angles 15–23°), forces and
  stress agree with finite differences to 1e-7 eV/Å and 1e-8 eV/Å³. The
  residual is identical in the local model alone, so it is not the
  electrostatics.
- **Cost, still open.** Electrostatics add 2–3× (matrix-free, OMat24 cells)
  and 1.5–7× (molecules) to a CPU training step. The profile is dominated by
  building the pair tensors and differentiating them twice for force
  training, not by CG iterations; `igamma` is about 12 %.

### Found on real data, and answered (SPEC 23)

- **Charge-transfer polarizability catastrophe.** In water models trained on
  energies and forces only, the polarizability is wrong and grows with size:
  α per molecule is 2.5 Å³ (monomer) → 6.2 Å³ (tetramer), against PBE0's
  1.35–1.44. Charge transfer carries 68–99 % of it, because the learned bond
  hardness is flat in distance.
- **Answered by an opt-in contact law of the bond hardness**
  (`ElectrostaticHeads(charge_transfer_decay=3.0)`). The hardness jumps ×10³
  out of covalent/ionic contact (Cordero radii), then grows as e^{βr}. On
  water chains, α per molecule from 1 to 6 molecules grows ×1.20 (induced
  dipoles only) instead of ×3.5. Ionic and covalent contacts are untouched.
  The atomic numbers now reach the heads as `species`.
- **A smoothness kink removed in passing.** `F.softplus` switches branch at
  x = 20, a 2e-9 step in the exponent; the law uses `logaddexp(x, 0)` instead.

### Added

- **SPEC 22: through-bond electronic coupling of charge states**
  (`transfer_coupling.py`). `TransferCouplingHead`, `bridge_coupling`,
  `coupling_matrix`; model hook `charge_state_coupling="through_bond"`; YAML
  `electrostatics.transfer_coupling`; checkpoint rebuild. V is the Löwdin
  bridge Green's function over the model's neighbour graph. The gaps
  Δ_i = Σh + softplus(γ_i) + g_min make it SPD, convergent and smooth for every
  parameter value. It reproduces the uniform-bridge closed form
  V(n) = h sinh θ / sinh((n+1)θ) to 1e-13.
- **ε∞ training loss** (`dielectric_weight`, label
  `info['dielectric_tensor']`). Taken on 3-D periodic cells only, intensive,
  rotated with the structure.
- **Partial-charge training loss** (`partial_charge_weight`,
  `partial_charge_key`; label `arrays[partial_charge_key]`, e.g. Hirshfeld).
  Off by default: charges are a partitioning convention, not an observable.
  It exists for like-for-like benchmarks against models that train on charges
  (3G/4G-HDNNP, PFN's QEq NequIP). The loss is per atom and intensive,
  `mean_i (q_i - q_ref_i)^2`, on the solved `q = B diag(f) t + rho`. A label
  that does not sum to the net charge is refused.
  - A test checks the batch alignment atom by atom. A planted bug (a
    permuted q) was caught by it.
- **Fields, metals, electrodes and charge states in training**, both single and
  batched. The dataset reads the calculator's `electrostatic_environment`. The
  batched path gives every structure exactly its own environment, and a mixed
  batch equals the single-structure energies and forces to 1e-10.
- **`ElectrostaticModule.resolve_inputs` / `.state`.** The one place the batch
  dictionary is read; the energy and the observables are the same functional
  by construction.
- **`tests/test_electrostatics_guide.py`.** Runs every Python block of
  `docs/electrostatics.md` in order, against a checkpoint written by the real
  `train.main`.

### Fixed

- **Observable losses with quadrupole heads.** Every dipole, α, Born-charge or
  ε∞ loss on a model with quadrupole heads failed: the rebuilt state lacked
  the quadrupole sector, and its energy differed by 1e-5 eV. This was a
  regression of round 3, which switched the module's quadrupole default on but
  not the rebuild's.
- **Hopping parameterisation.** An unbounded `exp(MLP)` reached hoppings of
  1e-39 to 1e15 eV under randomised weights. The modulation is now bounded to
  two decades around a learned exponential distance law.
- **Stale user guide.** `docs/electrostatics.md` said `hardness_margin: legacy`
  is refused from YAML and that quadrupole heads train with the sector off.
  Both were fixed in round 3, but the guide still described the old behaviour.

## Round 3 (after the peer review of 2026-09-24)

Every change below answers a finding of the review, and every number was
measured in the package after the change. SPEC sections are in `SPEC.md`.

### Changed: what the energy IS (SPEC 17)

- **The energy is the converged constrained minimum**, solved to
  `solve_rtol = 1e-10` by preconditioned projected CG and differentiated
  implicitly (`implicit.py`). It used to be "k = 6 CG steps differentiated
  through" (SPEC 2), which is kept, bit for bit, as `solve_mode="fixed_k"`.
  - Why: the fixed-k response was arbitrary for anything the zero-field
    residual did not excite, and did not converge with k. At k = 6 / 12 vs the
    exact minimiser: Td molecule 0.315 / 0.315 vs 0.363; Oh 0.399 / 0.399 vs
    0.730; D6h ring −2.25 / +3.82 vs 1.46; ideal rocksalt ε∞ 2.780 vs 2.823;
    27-atom cube perturbed by 1e-10: −1720 (k = 6), +5.2e9 (k = 12). The
    fixed-k energy also depended on the pure gauge `ell` (0.11 eV on 12 atoms).
    All of these now return the minimiser's value.
  - Exact derivatives of every order: energy node by the envelope theorem (no
    solve for forces), moments by an adjoint solve that is itself
    differentiable. Checked to third order against a closed form.
  - Conservative: closed-loop work of the forces −1.2e-9 eV at the default
    tolerance, the same as the exactly conservative fixed-k energy
    (quadrature floor). Force vs finite difference ≤ 1.2e-11 eV/Å.
  - Robust: CG raises `IndefiniteOperatorError` on the first direction of
    non-positive curvature (the energy would be unbounded below),
    `SolverNotConvergedError` rather than return a truncated iterate, and
    `FloatingPointError` on NaN input.
  - Preconditioner: exact Woodbury inverse of the bond sector (SPEC 17.6),
    3–5x fewer iterations than before (216-atom rocksalt: 125 → 27 dense,
    147 → 43 matrix free).

- **Default multipole kernel is Gaussian** (positive definite by theorem).
  Thole's `a < 1` bound was a pair bound presented as an N-body one: a
  10-site chain at a = 0.95 is indefinite (SPEC 1.6, corrected).

- **Hardness floor corrected** (SPEC 1.2, 17.3). The tight Gram bound is
  `S_0 = sqrt(2) K_E/(sigma sqrt(pi))`; the head emits `softplus + S_0 + 1 eV`.
  The old floor was sqrt(2) larger and everything below it was wrongly
  called "unbounded". Thole needs `hardness_margin="legacy"` (measured
  indefinite at the tight floor).

- **Dissociation envelope moved into a shell** (SPEC 17.4): f = 1 below
  `0.75 r_max`, C2 quintic switch to `r_max`; `r_max` must be >= ~1.4x the
  first-shell bond length for the bias to vanish. The full-range cosine put a fixed
  −0.9/Å into d ln t/dr at 2 Å and drove Born charges to the wrong sign
  (rocksalt test case: q = +0.24, Z* = −0.90). `envelope_onset=None` restores
  it.

- **Explicit fragments exchange no charge** (SPEC 17.4): bonds joining two
  fragments are pinned (`crossing_bond_selector`). The old constraint let
  current pass through a fragment and carried a position-dependent projector.

### Added

- `open_operator.OpenOperator` and `solver="matrix_free"` for OPEN boundaries
  (SPEC 18): no n×n object; equals the dense route to solve tolerance.
- Particle-mesh Ewald (`reciprocal="pme"`, SPEC 19) for the matrix-free
  periodic route: matches the direct sum to the requested accuracy; 1728
  atoms energy + forces 3.8 s vs 23.3 s. With it the converged, exact solve
  costs about what the old truncated fixed-k solve did.
- Automatic real-space cutoff floor for the periodic Gaussian routes (no more
  `LatticeWidthError` at a user-chosen cutoff).
- `tests/electrostatics/test_reference_physics.py`: 67 external references —
  Madelung constants of five structures (≤ 4e-15), Clausius–Mossotti and its
  exact damped form, Szigeti / Lyddane–Sachs–Teller / LO–TO, three-body
  dispersion C9 — with mutation checks showing they detect injected errors.
- Training: `electrostatics:` YAML block in `train.py`, float64 enforcement,
  AMP refusal, per-structure charge state in batched training, dipole /
  polarizability / Born-charge losses, checkpoint round trip into the ASE
  calculator (`tests/test_train_electrostatics.py`).

- **Metals and constant-potential electrodes** (SPEC 20, `metals.py`):
  metallic regions as zero-hardness spanning stars (equipotential to 1e-9 V),
  electrodes held at potential differences by external-circuit bonds
  (`u_e − u_ref = ΔV` to 4e-9, Legendre `dE/dV = −τ` to 8 digits); the
  image-charge law of an ideal conductor is recovered as a plate grows (0.8 %
  at 87.5 Å).
- **Integer charge localisation by coupled diabatic charge states** (SPEC 21,
  `charge_states.py`): EVB lowest eigenvalue, smooth through crossings, integer
  charges away from them; forces exact.
- ASE calculator: `fragment_policy="initial"` (charged reactive MD no longer
  crashes when a molecule splits; energy continuous across the cutoff), and
  `info['external_field']`, `arrays['metallic']`, `arrays['electrode_potential']`,
  `info['charge_states']` reach the solve.
- `EnvelopeShellWarning` when an atom's nearest bonded neighbour lies inside
  the envelope's switching shell (the Born-charge bias of SPEC 17.4).
- Near-field Coulomb in the matrix-free preconditioner (n ≤ 2048): 13–16 %
  fewer iterations.

### Fixed

- `born_charges_minimiser` (and `born_charges`, `born_charges_from_energy`)
  freed the caller's graph; a second call on the same state crashed.
- LAMMPS export silently DROPPED the electrostatic energy of a model with
  electrostatics attached; it now refuses such a model.
- Contradictions in `SPEC.md` (quadrupoles "out of scope", periodic
  quadrupoles "NOT YET", Gaussian "open boundaries only", β "out of scope")
  and stale `config.py` docstrings.
- `fragment_spins` was documented as 2S; it is S (the code was right).
- Dead branch in `ElectrostaticConfig.ewald_params`.
- `max_direct_atoms` blocked the matrix-free open route; PME was silently
  ignored under open boundaries (now refused); quadrupole heads attached to the
  potential ran with the sector silently OFF (now on whenever the heads carry
  it); applied fields could not be passed through the model hook; YAML could
  not set `hardness_margin: legacy`.
- `check_hardness_floor=False` claimed to disable a check it cannot disable
  (now deprecated, warns).
- Stale docstrings and SPEC sections describing the fixed-k energy, the old
  floor, "open boundaries only", and a PME "NOT DONE"; SPEC 17.6 reordered.
- Test-suite noise: the e3nn TorchScript warning (~260 per run) is filtered by
  its exact message.

### Known limits (not done)

- No GPU benchmark; float64 is still required. Solve iterations still grow
  slowly with size (57 / 80 / 98 at 512 / 1728 / 4096 atoms).
- Open boundaries are O(n²) (no FMM / treecode).
- Dispersion and the dense-operator observables refuse on the matrix-free
  routes.
- Metals are spanning trees, so a periodic metal's in-plane polarizability
  stays finite; only electrode potential DIFFERENCES are imposed.
- The diabatic coupling of charge states is an input, not learned.
- Batched training does not yet carry fields, electrodes or charge states.
- Solve iterations still grow slowly with size (no multilevel preconditioner).
- No trained model and no comparison against DFT/CCSD reference data yet;
  the training pipeline for it now exists.
- The electrostatics cannot be exported through AOT / LAMMPS on torch 2.2
  (export refuses rather than drops it).
