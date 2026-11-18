> **About this file.** This is the dated lab notebook of the TRACE-Electrostatics work on the 4G-HDNNP / PFN
> charge-transfer benchmarks: every pre-registered step with its criteria, runs and scores, kept as it was written,
> failures, withdrawn runs and corrections included. The summary of results is
> [`docs/electrostatics/BENCHMARKS.md`](../../docs/electrostatics/BENCHMARKS.md) and the reproduction guide is
> [`docs/electrostatics/REPRODUCE.md`](../../docs/electrostatics/REPRODUCE.md).
> For the public release on 2026-11-16, private details were removed: machine names (now "GPU host 1" and
> "GPU host 2"), home-directory and remote paths (now `<gpu-host-workdir>`, `$TRACE_4G_RAW` and `$TRACE_FP_CACHE`),
> process IDs, an unrelated job's name, references to internal project notes and policy numbers (now pointing to the
> SPEC and the CHANGELOG, or naming the policy), and quotes of the project lead (now `decision of <date>`). No
> measured number was changed. Some queue, watcher and sync scripts named below drove the lab machines and are not
> shipped; `docs/electrostatics/REPRODUCE.md` gives the local route.

# 4G-HDNNP charge-transfer benchmarks

The four benchmark systems of Ko, Finkler, Goedecker and Behler, "A
fourth-generation high-dimensional neural network potential with accurate
electrostatics including non-local charge transfer", *Nat. Commun.* 12, 398
(2021), arXiv:2009.06484. Each is built so that a model which sees only local
geometry must fail. The same data underlie the pfnet-research
`charge_transfer_nnp` reimplementation.

## Data

- **Source.** Materials Cloud record 2020.137, `datasets.tar.gz` (37.9 MiB,
  CC-BY-4.0), unpacked anywhere (not redistributed here); `$TRACE_4G_RAW` points at its `datasets/`
  directory (default `<repo>/experiments/benchmarks_4g/raw/datasets`).
- **Reference method.** PBE/FHI-aims, "light" settings.
- **Format.** RuNNer, in atomic units.

`prepare.py` converts it to extxyz:
- units: eV and Å;
- `info['charge']` holds the total charge;
- `arrays['hirshfeld_charges']` holds Hirshfeld charges, kept for comparison
  and never trained on;
- every structure is checked so that its Hirshfeld charges sum to its total
  charge.

It writes fixed-seed, charge-state-stratified subsets to
`data/{NAME}_{train,valid,test}.xyz`.

| set | structures | systems | cutoff (paper) |
|---|---|---|---|
| Ag_cluster | 11,013 | Ag3+ (5,014), Ag3− (5,999) | 10 Bohr = 5.29 Å |
| Carbon_chain | 10,019 | C10H2 (5,008), C10H3+ (5,011) | 4.23 Å |
| NaCl | 5,000 | Na9Cl8+ (2,500), Na8Cl8+ (2,500) | 5.29 Å |
| AuMgO | 5,000 | Au2 on MgO(001), doped/undoped, 110 atoms, periodic | — |

## Protocol (compute-limited)

- **Subsets.** 1,000 train / 250 valid / 1,000 test per set, seed 0. The paper
  used 90 % of everything.
- **Model and run.** The water model (hidden 64, l_max 2, v2 attention),
  40 epochs, seed 42, with per-element reference energies.
- **The two arms:**
  - local TRACE;
  - TRACE + electrostatics with the new-run defaults (`solver: auto`, the
    SPEC 23 contact law, no seed loss). In these first runs the seed loss is
    still on; the default changed after they started.

`run_4g.sh` trains both arms on the three cluster sets, and
`eval_4g.py DATASET RUN...` evaluates them.

## The paper's tests, reproduced by `eval_4g.py`

1. **Test RMSE per charge state:** energy in meV/atom, forces in eV/Å.
2. **Carbon chain:** the mean force error per atom index (the paper's Fig. 4).
   The paper's errors at the DFT minima were 2G 0.095 / 0.127 eV/Å and 4G
   0.032 / 0.042 eV/Å (C10H2 / C10H3+).
3. **Ag3:** relax each charge state from its reference minimum (the
   lowest-force structure, converged below 0.0015 eV/Å) and measure the RMSD.
   The paper's numbers: 2G 0.420 / 1.291 Å, 4G 0.002 / 0.006 Å. At the
   optimised structures, the energy errors were 4G 1.166 / 0.320 meV/atom and
   2G 0.605 / 2.017 eV/atom.
4. **NaCl:** Na9Cl8+ at its mean data geometry, and Na8Cl8+ as the same
   geometry minus the distant Na (index 11), which is the paper's
   construction. Atom 0 is scanned along the line through atom 3. The DFT
   minima of the two clusters differ by more than 0.1 Å; a local model gives
   one minimum for both.

**AuMgO** is periodic. The paper's test is the energy of the wetting geometry
minus the non-wetting one, with the Au positions optimised and the substrate
fixed. `eval_aumgo.py` places both geometries on the mean substrate of each
doping.

The two mean substrates agree to 0.0018 Å within 10.58 Å of the Au. The
TRACE local model (r_max 5.29 Å, one layer) sees only atoms within 2 r_max of
the Au, and the nearest Al is 10.71 Å away, so its doped and undoped values
must coincide.

| | DFT | 2G | 4G | TRACE local | TRACE + elec |
|---|---|---|---|---|---|
| doped | −2.7 meV | 375.0 | −41.0 | +2.8 | +195.0 |
| undoped | 928.9 meV | 375.0 | 975.4 | +3.7 | +192.7 |
| non-wetting Au–O, doped | 2.332 Å | 2.256 | 2.342 | 2.273 | 2.261 |
| non-wetting Au–O, undoped | 2.190 Å | 2.256 | 2.177 | 2.273 | 2.261 |

Test RMSE (seed 42, **300** training structures, 25 epochs, 400 test), as E in meV/atom /
F in eV/Å:

| | doped | undoped |
|---|---|---|
| local | 1.29 / 0.171 | 5.35 / 0.131 |
| elec | 1.55 / 0.122 | 6.44 / 0.115 |

**TRACE + electrostatics fails this test, like 2G.** The doped/undoped
difference is 2 meV, against 931 meV in DFT.

**The cause, measured.** Average charges over 160 test structures:

| | Hirshfeld q(Au2) | TRACE q(Au2) |
|---|---|---|
| doped, non-wetting | −0.196 | −0.111 |
| undoped, non-wetting | −0.024 | −0.126 |
| doped, wetting | −0.178 | −0.270 |
| undoped, wetting | +0.186 | −0.286 |

In DFT, doping moves 0.17–0.36 e onto the Au2, more than 10 Å from the Al
donors. In TRACE the Au2 charge changes by 0.015 e. The Al charge (TRACE
+2.55 e for the three, Hirshfeld +1.2) is screened by the surrounding oxygens
and never reaches the gold.

The bond-hardness floor is not the cause: κ₀ = 0.1 eV/e² allows soft bonds.
The cause is the physics of SQE:
- SQE describes dielectric, local, fractional charge transfer. It suppresses
  long-range transfer through an insulator by construction, which is its
  purpose against QEq's spurious transfer (Nistor & Müser 2009).
- Au2/MgO:Al is a donor–acceptor redox across an insulator.
- 4G's global QEq is metallic and moves the charge.
- PFN's plain NequIP (6 layers × 4 Å = 24 Å of receptive field) sees the
  dopants directly, which explains its 9 meV/Å.

This is the open physics problem for TRACE on this benchmark; the steps below take it up,
and `docs/electrostatics/BENCHMARKS.md` summarises where it stands.

**Polarization or charge transfer? Measured** with `probe_aumgo_physics.py`:
20 test structures per class, the trained elec model, ASE's result cache reset
before every evaluation.

1. **Both polarization channels are active.**
   - Bond charge flow moves an RMS 0.89 e per atom; this is what makes Mg/O
     ionic.
   - Induced dipoles: RMS |μ| = Au 0.038, O 0.022, Mg 0.003 e·Å.
2. **Gauss's law around the dopants.** The table gives the cumulative net
   charge within R of the three dopant sites, doped minus undoped, in e.
   The undoped slab supplies the same sites as Mg.

   | R (Å) | 2.5 | 4.0 | 5.5 | 7.0 | 8.5 | 10.0 | 11.5 | 13.0 | all |
   |---|---|---|---|---|---|---|---|---|---|
   | TRACE, non-wetting | +1.05 | −0.35 | +0.04 | −0.01 | +0.29 | −0.01 | 0.00 | 0.00 | 0 |
   | DFT Hirshfeld, non-wetting | +0.71 | +0.22 | +0.20 | +0.17 | +0.32 | +0.18 | +0.12 | +0.11 | 0 |
   | TRACE, wetting | +1.02 | −0.40 | +0.02 | +0.14 | −0.13 | −0.03 | +0.01 | 0.00 | 0 |
   | DFT Hirshfeld, wetting | +0.77 | +0.34 | +0.33 | +0.34 | +0.24 | +0.38 | +0.27 | 0.00 | 0 |

   - TRACE screens the dopant like a dielectric: the net charge returns to
     zero within 5.5 Å.
   - DFT keeps +0.2 to +0.4 e around the dopants out to 10–13 Å. That charge
     is balanced only by the Au2, more than 13 Å away (−0.18 e non-wetting,
     −0.38 e wetting, matching the Au2 Hirshfeld differences).
   - That is electron transfer from donor to acceptor (redox), not
     polarization.
3. **The redox state in the trained model.** We imposed an integer transfer,
   substrate +1 / Au2 −1, as two fragments:

   | geometry | dE_CT doped | dE_CT undoped | doped − undoped |
   |---|---|---|---|
   | non-wetting | +1.394 eV | +1.419 eV | −0.025 eV |
   | wetting | +1.009 eV | +0.991 eV | +0.018 eV |

   The trained model gives the dopant no donor character. It never saw such
   a state, so it had no reason to learn one.

**Why polarization cannot do this, exactly.**
- TRACE's polarization is the linear response of bound charge:
  `P = Σ_b f_b t_b (r_j − r_i) + Σ_i μ_i`, with `ρ_bound = −∇·P`.
- By Gauss's law, the net charge of a region changes only by the flux of P
  through its boundary.
- In a dielectric (κ > 0, finite ε), a charge-neutral local change (Al for
  Mg) induces a polarization cloud that decays with distance. The induced net
  charge on a distant region therefore vanishes, and that is what the table
  shows.
- A donor electron is different: it occupies the lowest available state
  anywhere in the system, here the Au2 affinity level.
- Charge-equilibration models have two ways to produce this:
  - **Metallic flow (κ → 0, QEq).** This is what 4G does. It makes MgO a
    metal: infinite ε, the classic QEq polarizability failure that SQE was
    invented to fix.
  - **Discrete charge states (redox).** These keep MgO a correct dielectric.
    This is TRACE's SPEC 21 (EVB over integer fragment charges, Warshel and
    Weiss 1980) with the SPEC 22 through-bond coupling.
- The machinery exists and is tested, but needs the states declared
  (`info['charge_states']`). It was not switched on in these runs.

## Results, cluster benchmarks (seed 42, 1000 training structures, 40 epochs)

Test sets of 1000 structures. "local" is TRACE alone; "elec" is TRACE +
electrostatics. The elec runs still had the solver-seed loss on.

| benchmark | local: E (meV/atom) / F (eV/Å) | elec: E / F |
|---|---|---|
| Ag3+ | 751.3 / 0.463 | **3.55 / 0.0084** |
| Ag3− | 639.9 / 0.592 | **11.28 / 0.0234** |
| C10H2 | 7.10 / 0.145 | **2.19 / 0.055** |
| C10H3+ | 4.87 / 0.216 | **1.48 / 0.096** |
| Na8Cl8+ | 1.68 / 0.047 | **1.43 / 0.024** |
| Na9Cl8+ | 2.15 / 0.068 | **0.97 / 0.027** |

The paper's specific tests:

- **Carbon chain, mean |ΔF| per atom.** C10H3+: local 0.20–0.51 eV/Å along
  the chain (worst at atoms 4–10, the far part); elec 0.12–0.22. C10H2: local
  0.09–0.28; elec 0.03–0.12.
- **NaCl minimum of the 0–3 Na–Na distance** (the paper's construction).
  Local: 3.424 / 3.424 Å, identical by construction. Elec: 3.395 (Na9Cl8+) /
  3.450 (Na8Cl8+), a shift of 0.055 Å with the DFT ordering. The paper's DFT
  shift is 0.107 Å (extracted from its Fig. 7, see below), so about half is
  recovered.
- **Ag3 relaxed minima.** This test does NOT discriminate here: local
  0.007 / 0.008 Å, elec 0.000 / 0.007 Å. The local model learned one
  surface with both basins (triangle and linear), and each relaxation stays in
  its own basin. The paper's 2G instead found one minimum for both (0.42 /
  1.29 Å). The energy test is decisive instead: a local model cannot know the
  charge state (≈ 0.7 eV/atom error).

**Emergent charges vs DFT Hirshfeld.** No charge labels were used.
Hirshfeld is a partitioning convention, not the truth. 200 test structures
each:

| set | Pearson r | RMSE | model range | Hirshfeld range |
|---|---|---|---|---|
| Ag3± | 0.974 | 0.077 e | [−0.46, +0.37] | [−0.47, +0.40] |
| NaCl | 0.988 | 0.30 e | about ±0.9 | [−0.49, +0.74] |
| carbon chain | 0.14 | 0.15 e | [−0.39, +0.29] | [−0.07, +0.16] |

- **Ag3:** close agreement from energies, forces and the total charge alone.
- **NaCl:** the right trend, with larger magnitudes. Hirshfeld understates
  ionic charges, so this is not necessarily wrong.
- **Carbon chain:** the charges are NOT physical. In a covalent molecule,
  energies and forces do not pin the charge distribution (as with water's
  intramolecular α). The E/F gain there does not rest on correct charges.

Caveats:
- Single seed.
- About 1/9 of the paper's training data.
- Unlike 4G, no charge labels are used: only energies, forces and the total
  charge (a constraint).

## Comparison with 4G-HDNNP and PFN's NequIP

`compare_published.py` computes our numbers in each reference's own metric and
records the published values with their provenance (`comparison.json`).
`plot_comparison.py` draws `figures/fig1–4`.

**Where the published numbers come from.**
- **4G paper.** Read from the arXiv PDF: Table I, the Ag3 text, Table II and
  the Fig. 11 text. The NaCl curves of Figs. 7–8 were extracted exactly from
  the PDF's vector paths into `published/ko2021_nacl_fig7_fig8.json`: the
  tick-fit residual is < 3e-5 and the energy minima agree with the force
  zeros to 1e-3 Å. The paper's test RMSEs are in its SI, which we have not
  read.
- **PFN** (pfnet-research/charge_transfer_nnp).
  - The repo holds configs, no results:
    - n_train 8000 / n_val 1000 (Au2-MgO 4000 / 500), r_max 4 Å;
    - baseline trained on forces only; +E_ele and +QEq on forces (100) plus
      Hirshfeld charges (1).
  - Their only accuracy numbers are in the blog's Fig. 2: validation force
    RMSE vs epoch, with 2G/3G/4G drawn as reference lines.
  - Digitised from the 490×244 PNG (56 px per decade, 1 px = 4 %):
    - the lines by intensity-weighted row;
    - the NequIP curves as the colour band of the last 4 pixel columns,
      whose half-width is shown as an error bar.
  - A web-search summary had quoted "NequIP 34.32 / NequIP-LR 21.99 meV/Å"
    for Au2-MgO. The figure contradicts it (NequIP ends near 9 meV/Å there),
    so it is not used.

**Force RMSE, all charge states pooled (meV/Å)** (`fig1_force_rmse.png`):

| | TRACE + elec | TRACE local | 4G-HDNNP | NequIP | NequIP+E_ele | NequIP+QEq |
|---|---|---|---|---|---|---|
| training structures | 1,000 (Au2-MgO 300) | 1,000 (300) | ≈ 9,000 (4,500) | 8,000 (4,000) | 8,000 (4,000) | 8,000 (4,000) |
| C10H2/C10H3+ | 79 | 185 | 76 | **50** | 316 | 60 |
| NaCl | 25 | 59 | 33 | **10** | 16 | 12 |
| Ag3± | **18** | 537 | 31 | 97 | 25 | 20 (18.5–22.8) |
| Au2-MgO | 119 | 152 | 67 | **9** | 186 | 84 |

**Errors at the DFT minima**, the paper's Table I metric (`fig2_dft_minima.png`).
Energy in meV/atom, forces as the mean |ΔF| per atom in eV/Å:

| | 4G: E / F | TRACE + elec: E / F | in TRACE train set? |
|---|---|---|---|
| C10H2 | **0.463** / **0.032** | 4.08 / 0.036 | no |
| C10H3+ | **0.176** / 0.042 | 0.651 / **0.037** | yes |
| Ag3+ | **0.320** / — | 3.63 / 0.002 | yes |
| Ag3− | 1.166 / — | **0.466** / 0.013 | no |

The paper calls its force column "mean errors of the forces". We use the mean
per-atom norm, the |F_NN − F_DFT| of its Fig. 4. The mean absolute component
would be about 3× smaller for us: 0.012 / 0.013 eV/Å on the chain.

**The NaCl shift** (`fig3_nacl.png`). The Na8Cl8+ minimum minus the Na9Cl8+
minimum along the paper's scan:

| | shift |
|---|---|
| DFT | 0.107 Å |
| 4G | 0.109 Å |
| 3G | 0.017 Å |
| 2G | 0 |
| TRACE local | 0 |
| TRACE + elec | 0.055 Å |

**Verdict, stated without spin:**
- **Forces vs 4G.** At about 1/9 of 4G's data and 40 epochs, TRACE + elec is
  - better on Ag3± (18 vs 31) and NaCl (25 vs 33);
  - level on the chain (79 vs 76; 0.036 vs 0.032 at the minimum).
- **Forces vs PFN.**
  - Ag3±: level with NequIP+QEq (18 vs 20, inside their scatter band).
  - NaCl: 2.4× worse than plain NequIP.
  - Chain: 1.6× worse than plain NequIP.
  - PFN used 8× our data and 50–100× our epochs, so this is not a
    like-for-like data comparison in either direction.
- **Energies at the minima are our weak point.** We are 4–11× worse than 4G
  on three of the four structures, two of which were in our training set. We
  are better only on Ag3−.
  - Hypothesis, not tested: the charge-state energy offset lives entirely in
    the electrostatic E(Q), because the local energy never sees Q. With only
    40 epochs and the loss weighted toward forces, that offset is not yet
    resolved to sub-meV/atom.
- **The NaCl non-local test: 4G wins this first round.** It reproduces the
  DFT shift; TRACE + elec gets half of it and the right ordering. (Superseded
  at PFN's training sizes: TRACE + elec gets 0.104 Å against DFT 0.107 Å; see
  the protocol section below.)
- **Not a fair test of data efficiency.** That needs TRACE trained on PFN's
  8,000-structure splits. Ag3 (3 atoms) and NaCl (17 atoms) are cheap enough
  for that.

## Like-for-like protocol at PFN's training sizes (round 5, evaluated 2026-10-03)

**Setup** (`make_protocol.py`):
- PFN's split sizes (Ag₃ 8000 / 1000; NaCl 4000 / 500), with the rest of each
  data set held out as the test set (Ag₃ 2013, NaCl 500 structures);
- 300 epochs, batch 32, energy weight 10, seed 42, the first-round model.

**Scoring.**
- `EVAL_DATA_DIR=data_pfn python eval_4g.py ...` gives the per-charge-state
  RMSEs.
- `compare_protocol.py` gives the pooled force RMSE, PFN's Fig. 2 quantity.
  PFN report validation RMSEs; ours are on a test set never used for training
  or early stopping.

**Ag₃** (Ag₃⁺ / Ag₃⁻):

| run | E RMSE (meV/atom) | F RMSE (eV/Å) | pooled F RMSE (eV/Å) |
|---|---|---|---|
| TRACE local | 159.6 / 512.0 | 0.550 / 2.514 | 1.892 |
| TRACE + elec | 0.572 / 3.416 | 0.0022 / 0.0215 | **0.0159** |
| TRACE + elec + Hirshfeld loss | 0.569 / 3.462 | 0.0022 / 0.0212 | **0.0157** |
| PFN NequIP+QEq (digitised) | — | — | 0.0204 (0.0185–0.0228) |
| PFN NequIP+E_ele | — | — | 0.0252 |
| 4G-HDNNP (PFN's line) | — | — | 0.031 |

- TRACE + elec has the lowest pooled force error of every published model on
  Ag₃, and the local model fails, as it must: it cannot tell the charge states
  apart.
- **At the DFT minima:** 0.975 / 0.047 meV/atom (Ag₃⁻ / Ag₃⁺), against 4G's
  1.166 / 0.320. Both minimum structures are in our training set; the 4G paper
  trained on 90 % of the data.
- **Relaxation RMSD:** 0.007 / 0.000 Å (4G 0.006 / 0.002 Å).

**NaCl** (Na₈Cl₈⁺ / Na₉Cl₈⁺; evaluated 2026-10-03, all three arms on the same
300-epoch protocol, trained to completion):

| run | E RMSE (meV/atom) | F RMSE (eV/Å) | pooled F RMSE (eV/Å) | scan minima, Na₉Cl₈⁺ / Na₈Cl₈⁺ (Å) | shift |
|---|---|---|---|---|---|
| TRACE local | 1.56 / 1.96 | 0.0465 / 0.0624 | 0.0553 | 3.419 / 3.419 | 0 |
| TRACE + elec | 0.739 / 1.588 | 0.0057 / 0.0077 | **0.0068** | 3.367 / 3.471 | 0.104 |
| TRACE + elec + Hirshfeld loss | 0.177 / 0.432 | 0.0066 / 0.0089 | 0.0079 | 3.369 / 3.471 | 0.102 |
| DFT | — | — | — | 3.366 / 3.473 | 0.107 |
| PFN NequIP (digitised) | — | — | 0.0104 | — | — |
| PFN NequIP+QEq | — | — | 0.0117 | — | — |
| PFN NequIP+E_ele | — | — | 0.016 | — | — |
| 4G-HDNNP (PFN's line) | — | — | 0.0329 | — | 0.109 (4G paper) |

- **Forces.** TRACE + elec has the lowest pooled force error of every
  published model on NaCl: 0.0068 against NequIP's 0.0104.
  - The PFN numbers are validation RMSEs digitised from their Fig. 2; ours
    are on a held-out test set.
  - The 8× gain over our own local arm is physics against capacity: a
    one-layer, hidden-64 local model, against the same model with explicit
    Coulomb charge transfer.
- **Non-local shift.** Both elec arms reproduce the DFT minima to 0.002 Å and
  the shift (0.104 / 0.102 Å against DFT 0.107 Å; 4G 0.109 Å). The first-round
  run (40 epochs) had half of it.
- **The Hirshfeld charge loss** lowers the energy error 3–4×, at a 16 % higher
  force error.

**Carbon chain** (C₁₀H₂ / C₁₀H₃⁺).
- **Protocol.** PFN's split: 8000 training / 1000 validation structures, with
  the remaining 1019 held out as the test set (509 / 510). There is no
  overlap with the training set (checked by coordinates). 300 epochs, seed 42.
- **Runs.** Trained on GPU host 2. Scored on 2026-10-08 on the Mac by
  `compare_protocol.py` and `EVAL_DATA_DIR=data_pfn eval_4g.py`; logs are in
  `runs/protocol/{compare,eval}_Carbon_chain.log`.

| run | E RMSE (meV/atom) | F RMSE (eV/Å) | pooled F RMSE (eV/Å) |
|---|---|---|---|
| TRACE local | 0.791 / 1.577 | 0.0677 / 0.0907 | 0.0805 |
| TRACE + elec | 0.413 / 0.990 | 0.0145 / 0.0473 | **0.0356** |
| TRACE + elec + Hirshfeld loss | 0.284 / 1.030 | 0.0146 / 0.0493 | 0.0370 |
| PFN NequIP (digitised) | — | — | 0.0495 (0.0458–0.0518) |
| PFN NequIP+QEq | — | — | 0.0601 |
| PFN NequIP+E_ele | — | — | 0.316 |
| 4G-HDNNP (PFN's line) | — | — | 0.0757 |

At the DFT minima (the 4G paper's Table I), we take the lowest-max-force
structure of each charge state (max |F_DFT| = 0.0007 eV/Å). Both are in our
training set, as they are likely to be in 4G's, which trained on 90 % of the
data. Energy is in meV/atom; force is the mean per-atom |ΔF| in eV/Å, with the
component mean in parentheses:

| | 4G: E / F | TRACE + elec: E / F | TRACE + elec + Hirshfeld: E / F |
|---|---|---|---|
| C₁₀H₂ | 0.463 / 0.032 | **0.036** / **0.0034** (0.0011) | 0.207 / 0.0080 (0.0027) |
| C₁₀H₃⁺ | 0.176 / 0.042 | 0.190 / **0.0061** (0.0022) | **0.174** / 0.0167 (0.0057) |

- **Forces.** TRACE + elec has the lowest pooled force error of every
  published model on the chain: 0.0356 eV/Å, against NequIP's 0.0495 and 4G's
  0.0757. Ours is a held-out test set; PFN's numbers are digitised validation
  RMSEs.
- **At the DFT minima:**
  - forces are 6–10× below 4G's for both molecules, even in the conservative
    per-atom norm;
  - the energy is 13× below 4G's on C₁₀H₂;
  - on C₁₀H₃⁺ it is level with 4G (0.190 / 0.174 against 0.176 meV/atom).
- **The weak point is C₁₀H₃⁺.** Its force RMSE (0.047 eV/Å) is 3.3× that of
  C₁₀H₂. Per atom (mean |ΔF|), the cation's errors are 0.07–0.11 eV/Å at atom
  indices 1 and 3–9, against 0.01–0.03 on the neutral molecule. This is where
  the chain can still improve.
- **The Hirshfeld charge loss** lowers the C₁₀H₂ energy error (0.284 against
  0.413 meV/atom) and costs force accuracy, as on NaCl.
- Single seed.

## Step 9's physics on the other benchmarks (pre-registered 2026-10-16, before any result)

**Question.** Is Au₂/MgO step 9's physics universal, or tuned to one system and one DFT? The test applies
it unchanged to the other 4G/PFN benchmarks and compares with the round-5 like-for-like runs, 4G-HDNNP
and PFN's NequIP variants.

**Runs** (`make_step9_cluster_configs.py`; `runs/protocol/{system}_step9_s42`). Each is the round-5
config of that system with step 9's electrostatics:
- unchanged from round 5: PFN's split sizes, 300 epochs, batch 32, seed 42, the same data;
- from step 9: formal integer charge configurations, the diabatic ensemble with couplings, exclusion
  (SPEC 18), the SPEC 23 bond law, kT 0.5 → 0.1 over 20 epochs, model-solved reference energies.
- **Priors**, each the counterpart of a step-9 datum:
  - NaCl: ε∞ = 2.3295, the electronic limit of H. H. Li's Sellmeier fit (J. Phys. Chem. Ref. Data 5,
    329 (1976)); the phonon terms give ε(0) = 5.89 as a check;
  - Ag₃: Ag atom α = 50.97 a.u. = 7.553 Å³ (Cheng, arXiv:2410.01493, the source of the Au value);
  - carbon chain: none (no counterpart).

**What each run can test (stated before the results).**
- **Ag₃ and the carbon chain:** the exclusion term is identically zero, since Ag, C and H have neutral
  frames and no closed shells. These runs test the rest of the stack (integer configurations, couplings,
  kT schedule, bond law) and the no-harm property.
- **NaCl:** SPEC 18 acts on Na carriers (Na⁰) but not on the Cl hole of Na₈Cl₈⁺. Anion-frame carriers
  are outside SPEC 18. This run tests the stack; the anion-carrier extension (next build) is needed
  before NaCl tests exclusion itself.

| id | criterion (comparators: round-5 TRACE + elec on the same protocol; published best) |
|---|---|
| U1 | NaCl pooled F RMSE below PFN NequIP's 0.0104 eV/Å (round 5: 0.0068) |
| U2 | NaCl shift within 0.010 Å of DFT 0.107 Å; both scan minima within 0.01 Å of DFT (3.366 / 3.473) |
| U3 | Ag₃ pooled F RMSE below PFN NequIP+QEq's 0.0204 eV/Å (round 5: 0.0159) |
| U4 | carbon-chain pooled F RMSE below PFN NequIP's 0.0495 eV/Å (round 5: 0.0356) |
| U5 | no harm: each system's per-charge-state E and F RMSE no worse than round 5 by > 10 % |
| U6 | ε∞(NaCl) within 2.0–2.7; Ag atom α within 10 % of 7.553 Å³ |

**Reported:** the energies and forces at the DFT minima (4G Table I), the Na₈Cl₈⁺ hole site, and cost
per epoch.

**Reading.**
- U1–U5 pass: the stack transfers across ionic clusters, metal clusters and a conjugated molecule
  without harm. It still does not show exclusion transferring (above).
- U5 fails anywhere: the step-9 stack is not universal as built, and the failure names what is
  system-specific.

**Amendment to the Ag₃ run (2026-10-17 14:20).**
- It stopped at epoch 155 (09:41) inside PyTorch's autograd engine: `RuntimeError: dependency not found for
  MulBackward0`.
- That is an engine-internal error, not a non-finite value. It occurred while the Mac was under heavy
  concurrent load from my probes and suites. It appears in no other run's log; its cause is not identified.
- Resumed from the epoch-148 checkpoint with its optimiser, scheduler and reference energies (the step-11
  procedure).
- Log: `log_attempt1_autograd_error_epoch155.txt`.

### Results, step 9's stack on NaCl and Ag₃ (`runs/protocol/{NaCl,Ag_cluster}_step9_s42`; scored 2026-10-17/18)

| id | criterion | measured | verdict |
|---|---|---|---|
| U1 | NaCl pooled F RMSE < NequIP 0.0104 eV/Å | **0.0064** (round 5 0.0068; 4G 0.0329) | PASS |
| U2 | NaCl shift within 0.010 Å of DFT 0.107; minima within 0.01 Å of 3.366 / 3.473 | 0.102 Å; 3.369 / 3.471 Å | PASS |
| U3 | Ag₃ pooled F RMSE < NequIP+QEq 0.0204 eV/Å | **0.0160** (round 5 0.0159; 4G 0.031) | PASS |
| U5 | no state worse than round 5 by > 10 % (E and F) | E: Na₈Cl₈⁺ 0.181 (0.739), Na₉Cl₈⁺ 0.163 (1.588), Ag₃⁺ 0.553 (0.572), Ag₃⁻ 3.391 (3.416) meV/atom. F: Na₈Cl₈⁺ 0.0068 (0.0057, **+19 %**), Na₉Cl₈⁺ 0.0061 (0.0077), Ag₃⁺ 0.0027 (0.0022, **+23 %**), Ag₃⁻ 0.0215 (0.0215) eV/Å | **FAIL** (forces, two states) |
| U6 | ε∞(NaCl) 2.0–2.7; Ag α within 10 % | not measured yet | open |

**Reported.**
- At the DFT minima (4G Table I metric): Ag₃⁻ |ΔE| 0.815 meV/atom (4G 1.166, round 5 0.975); Ag₃⁺ 0.303
  (4G 0.32, round 5 0.047).
- The relaxed Ag₃± minima are within 0.007 Å of the reference.

**Reading.**
- The stack transfers to the ionic and metal clusters on the 4G/PFN force benchmarks. NaCl energies improve
  4–10× (0.18 / 0.16 against 0.74 / 1.59 meV/atom).
- U5 fails on two cation states' forces, by 0.0011 and 0.0005 eV/Å. Under the pre-registered reading the
  stack is not universal as built. The failures sit in the cations (Na₈Cl₈⁺, Ag₃⁺), where the configuration
  ensemble has fewest carriers; the cause is not diagnosed.
- Ag₃⁺ at its minimum is worse than round 5 (0.303 against 0.047 meV/atom), level with 4G.
- Single seed. The Ag₃ run includes a resume (amendment above).

### Where MgO's screening lives, and a written prediction for step 14 (2026-10-19 ~19:00, before step 14 is scored)

- **The channel.** On step 13's weights, bulk MgO's ε∞ is 3.157 / 3.089, and 1.401 / 1.296 once SPEC 22 removes the
  bond charge flow (`probe_eps_bound.py --partition`). The trained models carry ~80 % of MgO's electronic
  susceptibility as charge flow between Mg and O.
  - Physically it belongs to the ions' dipoles: Clausius–Mossotti with α(O²⁻) ≈ 1.7 Å³ gives ε∞ = 2.95
    (ESTABLISHED).
  - Step 14 retrains under the ε∞ prior, which must make that move.
- **The move alone** (`probe_substrate_channel.py`). Every O and Mg dipole α is scaled until ε∞ under SPEC 22 equals
  the checkpoint's own, with the widths following the stiffness (SPEC 12). T5 is then split exactly:

  | seed | ε∞ restored by | T5_q, SPEC 22 frozen α | T5_q, dipole channel | screened share |
  |---|---|---|---|---|
  | 42 | α × 4.20 | −276.0 | **−577.2** | −351.2 (as trained, charge flow: −318.1) |
  | 43 | α × 5.37 | −378.7 | **−949.1** | +793.9, against ladder +2967 and self −4503 |

  - The image response of the substrate survives the change of channel: seed 42's screened share −351 against −318
    meV.
  - Seed 43's −949 is a cancellation of eV-sized ionic terms (its upright dimer stays strongly ionic), not physics.
- **Prediction for step 14 (hypothesis, written before its scoring).**
  - Seed 42: T5 ≈ −580 meV, so P2 fails for it. SPEC 22 cures the free–bound double counting; it does not supply the
    missing ~0.35 eV.
  - Seed 43: its T5 depends on whether training keeps the ionic upright dimer.
  - The decisive readings are `G_t5_terms` (the screened, shift and ionic shares) and p_ion.

### What the learned shifts are (2026-10-19; frozen step-13 weights; `probe_shift_overlap.py`)

| state | learned shift Δ (mean ± sd), eV | slope on overlap π | R² |
|---|---|---|---|
| Au⁺ | +1.76 ± 0.04 | +0.06 | 0.03 |
| Au⁻ | +1.39 ± 0.46 | +1.26 | 0.20 |
| O, Mg, Al states | −0.14 … −0.03 | ≈ 0 | ≤ 0.4 |

- π is the overlap energy (penetration plus Gordon–Kim) of the atom's state next to its neighbours' frames, which
  step 13 lacks.
- The shifts are mostly a constant renormalisation of the Au ionic ladder (+3.15 eV on Au⁺Au⁻: the Proposition 8
  inversion). Only Au⁻'s environmental part follows the overlap, with slope ≈ 1 but a fifth of its variance.
- Prediction for the trained steps 15–17: the constant part falls, because the overlap terms carry the ion-pair
  physics. `G_t5_terms` reports the shifts' T5 share.

### Scoring repaired and extended (2026-10-19)

- **Bug: `probe_dimer_ct.py` had crashed on every model since step 13.** It wrapped `_diabatic_solution` with a fixed
  signature, and SPEC 28/29 added `diag_correction` and `electrons`. So step 13's `D_dimer_ct.txt` held a traceback,
  and step 14's P1 could not have been scored.
  - Fixed: keywords pass through.
  - A second defect in the same probe: its Δ_ion summed the ladder, b and W only, which misses the pair terms
    (SPEC 18.5/23/26) and SPEC 28's correction. It now uses `configuration_energies` with both, and prints the
    partial sum beside it.
  - Re-scored step 13: Δ_ion(2.47 Å) = +2.961 (seed 42) and −0.997 eV (seed 43). |difference| = 3.96 eV, against
    step 12's 2.30 and P1's 0.5. Of it, the ladder (learned shifts) is 3.80 eV and the screening 0.74 eV.
  - The other wrappers (`probe_site_decomposition`, `probe_transfer_margin`, `probe_anion_resonance`) already pass
    keywords through; no other step-13 score file holds a traceback.
- **`score_extra.sh RUN`** writes the 2026-10-19 probes into `scores/`:
  - `G_gauge_split`: T5 into E_s / E_q, the Au shifts and p(Au⁺Au⁻);
  - `G_t5_stack`: the later steps' settings on this run's frozen weights;
  - `G_explicit_gap`, `G_dimer_ladder`: the isolated Au₂ gap and τ, learned and zeroed shifts, with and without the
    overlap terms and SPEC 28;
  - `H1_class_residuals_train`: the class offsets on the training set.

  `watch_extra.sh` runs it when steps 14, 13S, 15, 16 and 17 are scored. It was validated on step 13 (no
  tracebacks).
- **`read_criteria.py STEP`** reads 13S (N1–N4) and step 14 (P1–P6) from the score files. Comparators come from
  their own score files: P6's step-12 seed-42 numbers read back as E 0.847 / 0.703 meV/atom and F 0.1052 / 0.0764
  eV/Å, undoped / doped. P4's "Au atom α within 10 %" is taken against the relativistic CCSD(T) 5.435 Å³ that
  `probe_au_polarizability.py` prints. On step 13 it reproduces every number above.

### Diagnosed: the Ag₃± and NaCl models' charge-state physics (2026-10-19; frozen step-9 weights)

`probe_dimer_ladder.py` evaluates chosen configurations exactly from the module's own inputs to the diabatic solve.
`probe_cluster_ensemble.py` reads the test-set ensembles. Values are E(ion pair) − E(neutral pair), eV.

| pair (model) | as trained | shifts zeroed | + overlap stack, shifts zeroed | learned \|τ\| |
|---|---|---|---|---|
| Na₂ 3.08 Å (NaCl) | −1.41 | −1.58 | ~~+1.68~~ **+1.16** | 0.12 eV |
| Cl₂ 1.99 Å (NaCl) | +1.40 | +1.88 | +2.61 | 24.9 eV |
| Na⁺Cl⁻ 2.36 Å (NaCl) | −6.42 | −6.33 | ~~−5.60~~ **−4.14** | 8.6 eV |
| Ag₂ 2.53 Å (Ag₃) | −1.71 | −0.74 | +1.22 | 0.13 eV |

- (Corrected 17:30: the NaCl-model stack values for Na₂ and Na⁺Cl⁻ had the same double-counted Pauli; Cl₂, Ag₂ and
  the Ag₃ ensembles are unchanged on re-measurement.)
- Proposition 8 (theory note §4c) holds in both models. Na₂ and Ag₂ are ionic as trained, as the classical
  criterion predicts (−3.95, −2.28 eV), and Cl₂ is not (criterion +0.12 eV). The overlap terms make every
  homonuclear pair covalent; NaCl stays ionic, as it should.
- The learned couplings extrapolate past the data: τ(Cl–Cl, 1.99 Å) = 24.9 eV against Harrison's ppσ
  3.24 ħ²/(m d²) = 6.2 eV, and τ(Na–Cl, 2.36 Å) = 8.6 eV against spσ 2.5 eV. SPEC 27's bound is the identity where
  the data are (τ(Na–Cl, 2.8 Å) = 1.2 eV against 3.3 eV) and caps this. It does not move a benchmark number here.
- SPEC 28 must not list Na: Na's frame is the cation, so Na⁰ is 161.8× softer, and neutral Na₂ is a polarisation
  catastrophe that is refused.
- **The test-set ensembles** (40 structures per charge):
  - Ag₃⁺ as trained has multi-ion weight 0.984. Its ground configuration is Ag⁺Ag⁺Ag⁻ (p 0.846), and the physical
    one-hole configurations lie 1.8–2.0 eV above. With the overlap terms (frozen, learned shifts) the weight is
    0.018: the hole is on all three sites, p 0.32–0.34 each.
  - Ag₃⁻ as trained has weight 1.000, ground Ag⁻Ag⁻Ag⁺. With the overlap terms it is 0.964 with the learned
    shifts and 0.17 (max 0.39) with them zeroed: the remaining ionic character is in shifts trained against the
    inverted physics.
  - Na₈Cl₈⁺ has no multi-ion weight either way. Its Cl⁰ hole is spread over ~6 Cl within 0.17 eV as trained and
    localised (p 0.999) with the overlap terms. PBE references delocalise such holes (self-interaction), so this is
    a risk for the NaCl rerun.
- This connects U5's undiagnosed Ag₃⁺ force failure (+23 %) to an unphysical ensemble. Hypothesis, tested below.

### Pre-registered: Ag₃± with the overlap stack (written 2026-10-19 ~15:10, before the run)

**Run.** `runs/protocol/Ag_cluster_overlap_s42` = `Ag_cluster_step9_s42` (same data, epochs, seed, priors) with the
overlap stack of AuMgO steps 14–16: `bond_partition`, `carrier_penetration`, `carrier_overlap: gordon_kim`,
`exclusion: false`, `shift_gate`, `carrier_induction`.
- This is a universality validation of an already ablated stack (as step 9's NaCl/Ag₃ runs were), not a
  one-change ablation. The AuMgO chain carries the ablations.
- SPEC 28 is not used: Ag has no converged ion polarisabilities (all ratios 1).
- It launches when the Mac has a free slot (300 epochs; step 9's took ~33–37 h). Comparator: step 9 and round 5.

| id | criterion |
|---|---|
| **V1** | U5 restored: no state worse than round 5 by > 10 %, Ag₃⁺ F ≤ 0.00242 and Ag₃⁻ F ≤ 0.0237 eV/Å; E ≤ 0.629 / 3.758 meV/atom |
| V2 | U3 kept: pooled F RMSE < NequIP+QEq 0.0204 eV/Å |
| V3 | physical ensembles: test-set mean multi-ion weight < 0.25 for Ag₃⁺ and Ag₃⁻ |
| V4 | isolated Ag₂ at 2.53 Å: covalent ground configuration (gap > 0), and α∥ within 30 % of PBE0's 19.20 Å³ (step 9: 111.9) |

**Reported:** the 4G Table I metric at the Ag₃± minima (4G 1.166 / 0.32 meV/atom); τ(Ag–Ag); the ensembles.

**Reading.** V1 and V3 pass: U5's cation failure was the inverted ionic gap (Proposition 8). V3 passes and V1 fails:
the ensemble is fixed and the force deficit lies elsewhere.

### Carbon chain with step 9's stack: refused at start (2026-10-16, measured)

- The run (`runs/protocol/Carbon_chain_step9_s42/log.txt`) stops on its fourth forward call:
  `ConfigurationSearchError: more than cap = 512 configurations within 2.5 eV: charge is band-like here`.
  The module refuses rather than truncate, by design.
- **Falsifying check of the obvious hypothesis.** "Compact ionic widths make C⁺C⁻ pairs cheap (the
  collapse)" would be cured by `free_width_rule: frame` (SPEC 17.10). With it, the same refusal occurs at
  the same call. The hypothesis is falsified.
- **Reading, corrected the same day after replaying the refused call** (theory note 2026-10-16, §2):
  - The window holds 1451 configurations, and the lowest is a charge-ordered C⁺/C⁻ chain with 6 of 12
    bonds between charged atoms, only 0.8–1.1 eV below the single-hole configurations.
  - Ruled out: the learned shifts (zero), the width collapse (carbon is on the ladder width in both
    rules, so the frame-rule check above tested nothing), and over-screening (W_uu/S₀ ≈ 0.78).
  - The cause: the configuration energies carry no covalent-bond energy loss for charging both atoms of
    a bond. The PPP estimate is ~2.3 eV per bond.
  - My first reading, "genuine delocalisation", was wrong.
  - So step 9's stack does **not** transfer to conjugated covalent molecules as built.
  - Its exclusion term is identically zero there in any case (no closed-shell frames).
  - TRACE's chain result remains the round-5 mean-field one (pooled F 0.0356 eV/Å, against PFN NequIP
    0.0495 and 4G 0.0757).
- **What a universal stack needs here:** localised integer charges where charge localises (ionic, redox)
  and the mean-field SQE bond variables where it is band-like, chosen by the physics rather than by hand.
  That is not built.

### Pre-registered: carbon chain with free carriers (written 2026-10-16, before the run)

**Run.** `runs/protocol/Carbon_chain_step9cu_s42` = the refused `Carbon_chain_step9_s42` config with one
change: `redox.carriers: covalent_units` (trace_redox SPEC 19).
- The chain is one covalent unit (12–13 atoms > `full_unit_atoms` = 3).
- C₁₀H₂ is its frame (1 configuration). C₁₀H₃⁺ holds one hole, which hops along the chain through τ.
- Exclusion is identically zero here (no closed shells).
- The refused call replays with 13 configurations.

| id | criterion |
|---|---|
| U4 | pooled F RMSE below PFN NequIP's 0.0495 eV/Å (round 5 mean field: 0.0356) |
| U5 | no harm against round 5: E 0.413 / 0.990 meV/atom and F 0.0145 / 0.0473 eV/Å (C₁₀H₂ / C₁₀H₃⁺), none worse by > 10 % |
| U7 | C₁₀H₃⁺ force RMSE (round 5's weak point, 0.0473) reported; prediction (hypothesis): a tight-binding hole improves it |

### Results, carbon chain with free carriers (`runs/protocol/Carbon_chain_step9cu_s42`; scored 2026-10-18 20:35, best checkpoint epoch 299 of 300)

The run was resumed after the 13:56 reboot from epoch 272; its path includes that resume.

| id | criterion | measured | verdict |
|---|---|---|---|
| U4 | pooled F RMSE < NequIP 0.0495 eV/Å | **0.0329** (round 5 mean field 0.0356; 4G 0.0757; NequIP+QEq 0.0601) | **PASS** |
| U5 | no state worse than round 5 by > 10 % | E 0.439 / 0.956 meV/atom (0.413 / 0.990: +6.3 % / −3.4 %); F 0.0145 / 0.0435 eV/Å (0.0145 / 0.0473: 0 / −8.0 %) | **PASS** |
| U7 | C₁₀H₃⁺ force RMSE reported | 0.0435 against 0.0473 (−8 %); largest per-atom errors at C2 and C10 (0.108, 0.100 eV/Å) | reported; prediction borne out in direction |

**Reading.**
- The covalent-unit carrier (SPEC 19) runs the conjugated chain that the formal stack refused. It beats NequIP's
  pooled force RMSE by 34 % and its own mean-field round 5 by 8 %, with no state harmed beyond 10 %.
- At the minima: |ΔE| 0.525 / 0.130 meV/atom (4G 0.463 / 0.176). The C₁₀H₂ minimum is slightly worse than round
  5's 0.036 meV/atom. That is within U5's pooled criterion, but reported.
- With NaCl (U1, U2) and Ag₃ (U3) from step 9, the free-sector stack now passes U1–U4. U5 fails only on step 9's
  two cation force entries (Na₈Cl₈⁺ +19 %, Ag₃⁺ +23 %).

## The SPEC 23 contact law on an ionic system (round 5)

One Na9Cl8+ test structure had a 2.7 eV/Å force error (DFT max 0.79 eV/Å).
Its Na12–Cl14 bond is 3.125 Å, the longest in the test set, at the law's
onset R_i + R_j + δ = 3.18 Å. Scanning that bond from 3.0 to 3.35 Å:
- U_elec rises 0.85 eV;
- the force peaks at −5.3 eV/Å at 3.20 Å;
- q(Cl) falls from −0.58 to −0.14 e.

The law switches the bond's charge transfer off within 0.3 Å. This is the
SPEC 23 defect for ionic bonds:
- the width w = 0.1 Å is sharper than the hopping decay allows,
  w ≥ ln K / (8β) ≈ 0.29 Å;
- the onset uses covalent radii.

The same training (1,000 structures, 200 epochs, energy weight 10) with the
law changed. Test set; E in meV/atom, F in meV/Å:

| law | Na8Cl8+ E / F | Na9Cl8+ E / F | worst structure F | NaCl shift |
|---|---|---|---|---|
| w = 0.1 Å (current default) | 0.40 / 4.6 | 0.82 / 25.8 | 545 | 0.106 Å |
| off | 0.48 / 5.7 | 1.05 / 10.2 | 71 | 0.105 Å |
| **w = 0.3 Å** | **0.22** / 6.7 | **0.64 / 9.6** | **31** | 0.102 Å |

The DFT shift is 0.107 Å. The widened law removes the defect and gives the
best energies; every variant keeps the long-range shift. The water validation
(the reason the law exists) is running: `r6_elec_ctw03_s{42,7}` in
experiments/electrostatics_water_dimers.

## Architectural point under test

In 4G-HDNNP the solved charges also enter the short-range energy. In TRACE the
local energy never sees the charge state: only the electrostatic energy
depends on the total charge.

For the energy as a function of total charge Q, the quadratic form
E(Q) = χQ + ½ηQ² reproduces the ionisation energy and the electron affinity
exactly (Mulliken / Parr–Pearson). Whether the GEOMETRY dependence of
different charge states, such as the triangle-vs-line minima of Ag3±, is
captured that way is what Ag_cluster tests.

## Long-range electron transfer in Au₂/MgO(001):Al (`eval_aumgo_transfer.py`, 2026-10-08)

**Data.** The test set (`data/AuMgO_test.xyz`) has 200 undoped and 200 doped
(three Al) structures. The nearest Al is 10.5–11.2 Å (minimum image) from Au₂.
Only energies, forces and the total charge are trained on; no charge labels.

**The quantity.** The doped − undoped shift of the Au₂ charge. Hirshfeld and
model charges are different partitions, so absolute values are not compared.

**Control.** The doped geometries are re-evaluated with Al → Mg at the same
positions. This splits the shift into a dopant (electronic) effect at fixed
geometry and a geometric effect.

**Elec-only baseline** (`AuMgO_elec_r5_s42`, 300 structures, 40 epochs;
mean ± standard error):

| | Au₂ charge shift (e) | dopant effect at fixed geometry | geometric effect | r with DFT (per structure) |
|---|---|---|---|---|
| DFT (Hirshfeld) | −0.2656 ± 0.0083 | — | — | — |
| TRACE + elec | +0.0118 ± 0.0083 | +0.0094 | +0.0024 | −0.397 |

- **No transfer.** The bound sector is a dielectric. Its shift is consistent
  with zero, and its dopant effect has the wrong sign.
- **Accuracy:** energy RMSE 3.97 / 2.28 meV/atom and force RMSE
  0.110 / 0.105 eV/Å (undoped / doped).
- **TRACE-Redox** (`AuMgO_redox_r5_s42`, identical protocol; condensed
  elimination, exact canonical ensemble) is training. It will be scored by the
  same script.

### Why both arms fail here: the bound sector is a metal (measured 2026-10-08)

**Where the redox model's free charges sit** (epoch-2 checkpoint, 6 test
structures each):

| | free on Al | free on O | free on Au₂ | total q(Au₂) |
|---|---|---|---|---|
| doped | +2.720 e | −2.725 e (most negative O −0.27) | 0.0000 | −0.0047 |
| undoped | — | −0.086 | 0.0000 | +0.0179 |

- The three Al donate, and their neighbouring O accept: a local Al⁺–O⁻ pair.
- DFT disagrees. Al is Hirshfeld +0.404, barely above Mg +0.352, against the
  model's +1.15; and Au₂ is −0.19 doped against +0.08 undoped.

**Donor → acceptor energies** in the trained model, integer, term by term:
- Al → nearest O: −0.13 to +0.82 eV.
- Al → farthest O (10.6 Å): +0.81 to +1.06 eV.
- Al → Au (11 Å): +2.0 to +2.5 eV.
- The on-site energies decide: Ê_O(−1) = −4.58 eV against Ê_Au(−1) =
  −2.6 to −3.0 eV. The pair attraction W̃_{Al,X} is only ≤ 0.47 eV, because
  the screened W̃ is tiny: W̃_OO = 0.70 against the bare J_OO = 4.90 eV/e².

**ε∞ of bulk MgO from the bound (SPEC 1) sector**, via `dielectric_tensor`;
experiment ≈ 3:

| model | ε∞ |
|---|---|
| elec-only, trained (`AuMgO_elec_r5_s42`) | 1317 |
| redox, epoch 2 | 1278 |
| untrained heads, bond cutoff 5.29 / 3.5 / 2.5 Å | 1517 / 275 / 56 |
| trained elec, induced dipoles only (bond transfer frozen) | 1.99 |
| trained elec, second-shell Mg–Mg bonds blocked | 35.5 |
| trained elec, nearest-neighbour Mg–O bonds only | 35.4 |

**The cause.** Training left every second-shell Mg–Mg "bond" (2.98 Å) at
κ = 0.134 eV/e², essentially the floor; O–O sits at 3278 and every longer
bond is stiff. The Mg²⁺ sublattice became a metal for bound charge.
- The bound sector then screens any charge within a few Å. This is the
  Gauss's-law result above.
- So the elec arm cannot transfer, and the redox free sector picks acceptors
  by on-site energy alone.
- Energies and forces do not determine ε∞ (trace_electrostatics SPEC §23; CHANGELOG Round 4). Electrical
  information is required, which is what 4G and PFN get from Hirshfeld
  charges.

**Next runs:**
- `AuMgO_redoxq_r5_s42`: the redox protocol plus the Hirshfeld charge loss
  (weight 1).
- If ε∞ stays unphysical, a direct dielectric constraint on bulk MgO.

### Pre-registered evaluation of the two redox runs (written 2026-10-08, before any result)

**Runs**, identical protocol: 300 structures, 40 epochs, seed 42, the
condensed elimination and the exact canonical ensemble.
- `AuMgO_redox_r5_s42`: energies and forces only.
- `AuMgO_redoxq_r5_s42`: plus the Hirshfeld charge loss (weight 1).

**Tests**, run on the best-validation `model.pt` after each run finishes,
unchanged. Fixed comparators: `AuMgO_elec_r5_s42` and DFT.

| id | script | quantity | pass criterion |
|---|---|---|---|
| T1 | `eval_aumgo_transfer.py` | test E / F RMSE per subsystem | no worse than elec-only (3.97 / 2.28 meV/atom; 0.110 / 0.105 eV/Å) by more than 10 % |
| T2 | `eval_aumgo_transfer.py` | Au₂ charge shift, doped − undoped, ± standard error | negative and below −0.10 e, more than 3 standard errors from 0 (DFT −0.266) |
| T3 | `eval_aumgo_transfer.py` | dopant effect at fixed geometry (doped − Al→Mg) | at least 80 % of the T2 shift (electronic, not geometric) |
| T4 | `probe_free_charges.py` | free charge on Au₂ in doped structures | below −0.10 e: the donor electron reaches Au₂ |
| T5 | `eval_aumgo.py` | E(wetting) − E(non-wetting), doped vs undoped | the doped − undoped difference has DFT's sign, and at least 50 % of DFT's −931 meV |
| T6 | `probe_eps_mgo.py` | ε∞ of bulk MgO, bound sector | diagnostic: reported against ≈ 3 |

**Predictions** (hypotheses, to be confirmed or refuted):
- `redox`: fails T2–T5. Its free charge compensates Al locally on O, as at
  epoch 2. ε∞ of order 10³.
- `redoxq`: T6 of order 10³, because the Hirshfeld-trained elec model
  already gives 1379. T2–T5 are open: the charge loss penalises Al at +1.15,
  but the bound sector remains metallic.

No new code is written and no run is started until these are measured.

**Amendment (2026-10-08, before any result).**
- **The crash.** `AuMgO_redoxq_r5_s42` stopped at epoch 5. Its parameters
  turned NaN after an optimiser step, and the step that caused it was not
  identified.
- **The replacement run.** The charge-fitting arm is scored instead on
  `AuMgO_redoxq_r5_s42_nanexact`:
  - the same config and seed, run from scratch to epoch 40;
  - under `run_train_nanguard.py`, which checks the loss and every gradient
    after each structure and every parameter after each step;
  - on the pre-fix `trace_redox` (a copy in `code_prefix/`).
- **Is it the same run?**
  - It matched the original to every printed digit at epoch 1, and differs
    from epoch 2 on (5th digit; CPU threading).
  - It passed the original crash window (epoch 5, step 30) with gradient
    norm 0.59.
- **The solver.** The fix of `trace_electrostatics/CHANGELOG.md` Round 11
  changes values only at degenerate spectra, where the guard would stop the
  run and report it.
- The tests and criteria above are unchanged.

**Results, `AuMgO_redox_r5_s42`** (energies and forces only; best checkpoint
epoch 39 of 40, validation loss 0.00530; scored 2026-10-08; logs
`runs/AuMgO_redox_r5_s42/T*.log`, `P1_*.log`):

| id | criterion | measured | verdict |
|---|---|---|---|
| T1 | E / F RMSE no worse than elec-only by > 10 % (elec: 3.97 / 2.28 meV/atom, 0.110 / 0.105 eV/Å, undoped / doped) | E 3.537 / **3.001** meV/atom; F 0.0851 / 0.0897 eV/Å | **FAIL**: doped energy 32 % worse; forces 23 % / 15 % better |
| T2 | Au₂ shift < −0.10 e, > 3 standard errors from 0 | −0.0468 ± 0.0049 e (DFT −0.2656) | **FAIL**: right sign, 18 % of DFT |
| T3 | dopant effect at fixed geometry ≥ 80 % of the shift | −0.0468 of −0.0468 (geometric +0.0001) | PASS: the small shift is electronic, carried by the bound sector |
| T4 | free charge on Au₂ (doped) < −0.10 e | −0.0000 e; the free charge is on Al +2.995, O −2.996 | **FAIL** |
| T5 | doped − undoped wetting difference: DFT sign, ≥ 50 % of −931 meV | +247.7 − 175.4 = **+72.3 meV** | **FAIL**: wrong sign |
| T6 | ε∞ of bulk MgO, diagnostic | **1379** (experiment ≈ 3) | as predicted (order 10³) |
| P1 | one-curve shift of [dE(Al→O) − dE(Al→Au)] ≥ +0.5 eV | +0.085 eV (−0.047 … +0.276), 18 Al sites | **FAIL** |

- **Predictions on record.**
  - Confirmed: "fails T2–T5; free charge on O; ε∞ of order 10³" held for T2,
    T4, T5 and T6.
  - Refuted: T3 passed. A pass is easy here, because the whole small shift is
    electronic.
- **Wetting details.**
  - Absolute wetting energies: +247.7 / +175.4 meV (DFT −2.7 / +928.9; 4G
    −41.0 / +975.4).
  - Non-wetting Au–O bond: 2.279 / 2.245 Å. That is the DFT ordering, at
    24 % of DFT's 0.142 Å split.
- **T6 needed an amendment.** The redox module refuses bulk MgO because its
  free-charge W is singular there (λ_max = −0.0000), the metallic screening
  of a bound sector with ε∞ ≈ 10³.
  - `probe_eps_mgo.py` now tolerates that refusal after capturing its
    inputs. ε∞ is a bound-sector property and its computation is unchanged.
  - The elec model, re-run alongside as a control, reproduces 1316.955.
- **P1 was first run with a wrong operator** (my error).
  - It set both on-site couplings to η, which deletes the on-site Coulomb
    self-energy of bound and free Gaussians of different width. W′ was then
    indefinite in all 6 structures, so that first "pass" (+1.14 eV) was
    invalid.
  - The operator of the theory note keeps the full Coulomb and puts only the
    chemical hardness D = η − J_bb on the total charge (D: Al 1.70, O 2.03,
    Mg 1.93, Au 1.50 eV/e²). With it, W′ is positive definite everywhere.
    The result above is from that operator.
  - **Correction, 2026-10-09.** The D values first written here (2.8 / 5.7 /
    5.9 / 11–14) were mean interatomic distances in Å. A loop variable in
    `probe_one_curve.py` reused the name of D. The energies and the shift
    were computed before the collision and are unchanged on re-run.
  - It lowers Mg most as an acceptor (−1.98 eV), as one atom, one curve
    predicts. But O stays the lowest acceptor, 1.7 eV below Au.
  - So, as a post-hoc correction to this trained model, Theorem 2 does not
    move the donor electron to Au₂.

**Results, `AuMgO_redoxq_r5_s42_nanexact`** (the charge-fitting arm:
energies, forces and Hirshfeld charges, weight 1). Best checkpoint epoch 40
of 40, validation loss 0.00607; no non-finite value in the whole run. Scored
2026-10-09.

| id | criterion | measured | verdict |
|---|---|---|---|
| T1 | E / F RMSE no worse than elec-only by > 10 % | E 3.917 / **1.537** meV/atom; F 0.0829 / 0.0906 eV/Å | PASS: doped energy 33 % better, forces 25 % / 14 % better |
| T2 | Au₂ shift < −0.10 e, > 3 standard errors | −0.0233 ± 0.0049 e (DFT −0.2656) | **FAIL**: 9 % of DFT |
| T3 | dopant effect ≥ 80 % of the shift | −0.0216 of −0.0233 (93 %) | PASS |
| T4 | free charge on Au₂ (doped) < −0.10 e | −0.0000 e; free Al +2.995 → O −2.998 | **FAIL** |
| T5 | wetting difference: DFT sign, ≥ 50 % of −931 meV | +176.2 − 87.5 = **+88.7 meV** | **FAIL**: wrong sign |
| T6 | ε∞ of bulk MgO | **1346** | as predicted (order 10³) |
| P1 | one-curve shift ≥ +0.5 eV | +0.258 eV (−0.045 … +0.507) | **FAIL** |

- **The Hirshfeld loss did not carry the Au₂ information.**
  - Per structure, model q(Au₂) correlates with Hirshfeld at r = +0.04.
  - Model q(Au₂) is −0.032 / −0.009 e doped / undoped, against Hirshfeld
    −0.189 / +0.076.
- **How it met the Al charge instead.** The model matches Al's Hirshfeld
  charge (3 Al: +1.25 e against Hirshfeld ≈ +1.2) by cancelling the free
  charge locally. Each Al gives its free electron to O (+2.996), and the
  bound sector takes 1.744 e back. That is a local compensation, with no
  transfer.
- **Why this fits Theorem 3.** The Au₂ charge error reaches the free sector's
  Au level only through the population of the Au state, ⟨p_Au⟩ ≈ 10⁻⁸ at
  kT = 0.1 eV. So the charge loss acts only on the bound sector, which
  screens locally (ε∞ ≈ 10³) and cannot move charge 11 Å.
- **Both redox arms therefore fail the transfer tests (T2, T4, T5).** The
  charge-fitting arm is the most accurate Au₂/MgO model so far on energies
  and forces (T1). Its pooled force RMSE, ≈ 0.087 eV/Å, is still behind 4G's
  0.067.

**The bound sector alone, with the Hirshfeld loss (2026-10-09).**
- `AuMgO_elecq_s42` (elec, Hirshfeld weight 1, 25 epochs), scored by
  `eval_aumgo_transfer.py`:
  - Au₂ shift **+0.0125 ± 0.0046 e**, the wrong sign (DFT −0.266);
  - per-structure r with Hirshfeld **−0.42**.
- So the local SQE sector cannot carry the shift even when it is supervised.
  The failure is the architecture of the charge model, not the training
  recipe.

**P3: formal oxidation states with the closed-shell capacity**
(`probe_formal_states.py`, pre-registered, `AuMgO_redoxq_r5_s42_nanexact`).
- Setup:
  - each atom carries its Born–Haber integer ladder: successive IEs, EA₁, and
    O⁻ + e → O²⁻ at +844 kJ/mol (chemguide; Born–Haber spread 780–844);
  - O ≥ −2 (O³⁻ forbidden), as in Dapp & Müser's redoxSQE rule that a
    closed-shell ion cannot accept;
  - b = 0, with W either the trained W or bare J.
- Result: **0 / 6, FAIL.** In both brackets, placing Al's electrons on Au₂ is
  12–47 eV above leaving them on Al.
- **Why, measured.**
  - TRACE-Redox's free charges are Gaussians σ_free = 1.78 (O), 3.16 (Mg),
    4.03 (Al), 3.36 (Au) Å, from `width='hardness'`.
  - Their nearest-neighbour Mg–O Coulomb is 3.14 eV/e², 46 % of the point
    charge (6.85).
  - The pair Coulomb of the formal ionic configuration is −769 eV, against
    ≈ −2580 eV for point charges. So the formal ionic base lies +856 eV above
    all-neutral, whereas Born–Haber puts it ≈ −960 eV below.
- **Reading.** Formal oxidation states cannot enter the present free sector;
  its widths are wider than ionic charges. P3 refutes the drop-in, not the
  closed-shell physics.

**P4: coherent hopping of the free carriers** (`probe_p4_hopping.py`,
pre-registered criterion |R(t = 0.3 eV)| ≥ 1e-3 e/eV in every structure;
`AuMgO_redoxq_r5_s42_nanexact`, 3 doped structures).
- **Setup.**
  - The model's own single-molecule free sector, with the per-unit ensemble
    replaced by acceptor and donor tight-binding orbitals.
  - Hopping −t·s_uv on its chemical bonds; Daleckii–Krein susceptibility.
  - Mechanics: with t = 0 it reproduces `differentiable_solution` to
    1e-14 e in all three structures.
- **R = d q_free(Au₂)/d ε^a_Au** (e/eV):

| structure | t = 0 | t = 0.3 eV | gain | free Au₂ at t = 0.3 |
|---|---|---|---|---|
| 0 | 2.5e-06 | 1.4e-02 | 5.6e3 | −0.012 e |
| 1 | 3.5e-06 | 2.9e-02 | 8.3e3 | −0.025 e |
| 2 | 1.2e-10 | **2.8e-04** | 2.3e6 | −0.0004 e |

- **Verdict: FAIL** (2 of 3 meet the threshold; structure 2 is 3.6× below).
- What the data do show: hopping turns the exponential signal into an
  algebraic one in every structure (gains 10³–10⁶), and Al's free charge
  starts to delocalise (2.995 → 2.73–2.75 e).
- What they do not show: the claimed strength at t = 0.3 eV for every
  geometry.
- The earlier "≈ 1e-8" for t = 0 was an estimate. The measured range is
  1.2e-10 to 3.5e-6.
- Not done, and only allowed as a new pre-registered test: R at other t.
  Theorem 7 predicts R ∝ t².



### Pre-registered: step 1, a physical bond hardness (written 2026-10-09, before the run)

**Run.** `AuMgO_redoxq_bondlaw_r5_s42`: identical to
`AuMgO_redoxq_r5_s42_nanexact` (seed 42, 300 structures, 40 epochs, Hirshfeld
weight 1) except two keys:
- `electrostatics.config.bond_screening: [2.0, 2.8]`;
- `electrostatics.heads.bond_hardness_law: two_level`.

That is SPEC 26 / CHANGELOG Round 12. It runs under `run_train_nanguard.py`.
The comparator is `AuMgO_redoxq_r5_s42_nanexact`.

| id | criterion (pass needed for step 1 to count) |
|---|---|
| S1 | T6: ε∞(MgO) ≤ 30 (comparator 1346; experiment ≈ 3). This is the defect step 1 targets. |
| S2 | T1: E / F RMSE no worse than the comparator by > 10 % (3.917 / 1.537 meV/atom; 0.0829 / 0.0906 eV/Å) |

**Reported, not criteria:** T2–T5 and P1, by the unchanged scripts.

**Predictions (hypotheses):**
- ε∞ between 5 and 30. The smoke model, after 2 epochs on 4 structures, gave
  17.5.
- T2, T4 and T5 still fail: step 1 does not change the acceptor ladder or the
  training signal.
- The bound compensation on Al falls: |q_bound(Al)| < 1.0 e, against 1.744.

## Training speed on the Mac (2026-10-09)

A redox epoch of `AuMgO_redoxq_*` (300 structures, batch 1, `grad_accum_steps: 8`)
took about 19 min at 4 threads. The goal is the same numbers, faster.

**Measured.** One energy-and-forces evaluation of a doped Au₂/MgO structure
with the step-1 smoke model. The 4-thread training run was occupying the same
cores at the time, so these are contended numbers.

| change | result |
|---|---|
| A + B: exact half-integer gamma; damping only where needed (CHANGELOG Round 13) | 2.66 s against 2.89–3.12 s (8–15 %); energy identical; forces within 1.2e-15 eV/Å |
| C: threads per process, best / median of 6 | 2 threads 2.44 / 3.01 s; 4 threads 2.25 / 2.60 s; 6 threads 2.64 / 2.86 s: overhead-bound, little gain |
| processes instead of threads | 1 process × 2 threads 2.99 s per evaluation; 3 concurrent processes × 2 threads 4.81 s each, i.e. 1.86× the throughput |
| E: GPU | not run: GPU host 1 at 99 % and GPU host 2 at 100 % with other users' jobs |

**Data parallelism (`train.py`, torchrun, gloo, CPU).** The batches of each
optimiser step are dealt out across processes:
- batch b of a window goes to rank (b mod `grad_accum_steps`) mod world;
- validation batch b goes to rank b mod world.

The run is the sequential one **bit for bit**, not merely up to round-off:
- Every batch's random numbers (dropout, rotation) come from a seed of
  (seed, epoch, batch index). This is `batch_seeded_rng`, forced on in
  parallel.
- Every rank sums all the window's batch gradients in batch order, the order
  in which a sequential `.grad` accumulates them.
- Metric updates are replayed in batch order.

`tests/test_train_data_parallel.py` shows it for 2 and 3 ranks, at 1 thread per
process: the same gradients at every step, the same parameters after each step,
the same history and checkpoint. Each rank evaluated only its share.

"Bit for bit" assumes deterministic kernels. On this Mac, Accelerate's LAPACK
is not deterministic for the 110-atom redox system (see the amendment below),
so there no run, sequential or parallel, reproduces bitwise.

The test was checked by breaking the code. Each of these fails it:
- summing the gradients in reverse order;
- dropping the per-batch seed;
- replaying the training metrics out of order;
- replaying the validation metrics out of order.

With one process and the flag off, the new `train.py` reproduces the old one
bit for bit on both the batched and the per-structure path. That covers 6
steps: gradients, parameters and history.

Caveat: BLAS results can depend on the thread count. P processes × T threads
equal one process × T threads with `batch_seeded_rng: true`, not the earlier
4-thread runs. Those also drew dropout masks from a different stream, so a new
parallel run differs from an old comparator at the level of seed variance.

**Launch.** Add `torch_num_threads: T` to the config, then run:

```bash
cd experiments/benchmarks_4g && GLOO_SOCKET_IFNAME=lo0 python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29533 run_train_nanguard.py runs/NAME/config.yaml
```

- Rank 0 alone prints and writes checkpoints. Rank r > 0 writes its output to
  `stdout_rank{r}.txt` next to the checkpoint.
- The NaN guard and the redox dump write one file per rank.
- `--standalone` hangs on macOS: its rendezvous resolves an IPv6 reverse name
  that does not exist. Use the static rendezvous above. On Linux, use
  `GLOO_SOCKET_IFNAME=lo`.

**Pre-registered speed measurement (written before it is run).**
- **When:** on the idle Mac, after `AuMgO_redoxq_bondlaw_r5_s42` finishes.
- **What:** the step-1 config on the first 32 training structures (4 optimiser
  steps) and 8 validation structures, 1 epoch, timing the wall clock per epoch.
- **Configurations:** 1 process × 4 threads (current), 1 × 2, 2 × 4, 3 × 2,
  4 × 2, 5 × 2 and 8 × 1. The Mac has 6 performance and 4 efficiency cores.
  8 × 1 was added before any run: with 8 batches per step, 8 ranks is the most
  parallel layout.
- **Criteria:**
  - **V1:** 4 × 2 must equal 1 × 2 with `batch_seeded_rng` bit for bit on this
    periodic redox system.
  - **V2:** the fastest configuration is adopted for the next runs if it is at
    least 1.5× faster than 1 × 4; otherwise nothing changes.

**Amendment, 2026-10-09, written before any timing run: V1 cannot hold as
written, because a sequential run on this Mac does not reproduce itself.**

Measured on the step-1 config, 4 structures and 1 optimiser step:
- Two 1 × 1 runs differ in 34 of 107 parameter tensors, by at most 6.4e-12.
- The batch-0 gradients at identical parameters differ by up to 8.6e-13
  relative.
- Three force evaluations in one process differ by up to 1.8e-15 eV/Å. The
  energy is identical.

Ruled out as causes:
- the Python hash seed (`PYTHONHASHSEED=0`);
- Accelerate's thread count (`VECLIB_MAXIMUM_THREADS=1`);
- TorchScript graph optimisation (switched off).

An op-level trace (every aten op, inputs hashed before it runs) located the
first difference in `torch.cholesky_solve`: bit-identical inputs, a different
output. Torch's BLAS here is Apple Accelerate (M2 Pro). The isolated call
differs by about 1 ulp, and only intermittently: 51 of 200 calls in one test,
0 of 1000 in another.
- **Hypothesis, untested:** the result depends on contention for the shared
  AMX unit.
- **Consequence:** the 2026-10-08 NaN rerun that diverged from epoch 2
  ("CPU threading") is better explained by this.

**V1 becomes V1′:**
- Define d(X, Y) = max over tensors of max|θ_X − θ_Y| / max|θ_Y|.
- Require d(4 × 2, 1 × 2) ≤ 10 · d(1 × 2, 1 × 2 repeat).
- The repeat run `1x2b` is added for this.
- On the idle Mac, also report whether the two 1 × 2 runs are bitwise equal.
  That tests the contention hypothesis.

**Result (run 16:28–16:49, 2026-10-09).** The Mac had no compute jobs; about
one core was busy with desktop apps. Script `bench_data_parallel.py`; output in
`runs/bench_data_parallel_2026-10-09/`. Epoch = 32 training + 8 validation
structures.

| config | epoch (s) | speed-up vs 1×4 | d to 1 × T at the same T |
|---|---|---|---|
| 1×4 (current) | 109.3 | 1.00 | — |
| 1×2 | 142.7 | 0.77 | — |
| 1×2 repeat | 136.4 | 0.80 | 1.2e-10 (not bitwise) |
| 2×4 | 131.6 | 0.83 | 2.6e-7 |
| 3×2 | 104.1 | 1.05 | 8.2e-11 |
| 4×2 | 105.7 | 1.03 | 6.3e-11 |
| 5×2 | 141.2 | 0.77 | 4.6e-8 |
| 8×1 | 305.1 | 0.36 | — |

- **V1′ PASS:** d(4×2, 1×2) = 6.3e-11 ≤ 10 · 1.2e-10.
- **V2 FAIL:** the fastest configuration, 3×2, is 1.05×. Nothing changes; runs
  stay at 1 process × 4 threads.
- **The contention hypothesis is not supported.** Two sequential runs on the
  idle Mac are still not bitwise equal.
- **Unexplained (hypothesis, not tested):** d for 2×4 and 5×2 is 400–2000× the
  single sequential-repeat sample. The run-to-run spread may be heavy-tailed,
  since one ulp flip is amplified through four optimiser steps. One repeat pair
  is not enough to tell.
- **Why processes do not scale is not yet measured.** 8×1 being 2.8× slower than
  1×4 points to a saturated shared resource: memory capacity (swapping) or
  memory bandwidth (the lattice sums build ~10⁶-element pair-image arrays,
  e.g. 907 390 entries in the op trace), or Accelerate's shared AMX units. This
  is diagnosed next.

### Amendment to step 1: stopped and resumed (2026-10-09, written before the resume)

- **Stopped:** `AuMgO_redoxq_bondlaw_r5_s42` was stopped at 16:16 during epoch
  9, by decision, to measure the speed-up above. Epoch 8 had
  validation loss 0.017167; that checkpoint is kept as `model_epoch8.pt`.
- **Resumed:** from `model_epoch8.pt` for epochs 9–40, with
  `resume_load_optimizer` and `resume_load_scheduler` true. Same physics keys,
  1 process × 4 threads as before, under `run_train_nanguard.py` with
  `NANGUARD_SKIP_EPOCHS=8`, which reproduces the loader's batch order for
  epochs 9–40 exactly.
- **What differs from an uninterrupted run:**
  - `batch_seeded_rng: true`: from epoch 9 the dropout masks come from
    per-batch seeds. That is a different random stream, a seed-level
    difference.
  - Accelerate's nondeterminism (above) is present in any run.
- **Best checkpoint:** a resumed run restarts "best validation loss" at +∞, so
  `model.pt` holds the best of epochs 9–40. If that best is not below
  0.017167, the evaluation uses `model_epoch8.pt`.
- **Criteria:** S1 and S2 and the reported tests are unchanged. The log is
  `log_resume8.txt`; epochs 1–8 are in `log.txt`.

### The lever that worked: cheaper lattice blocks (R1–R3, 2026-10-09)

The run-time profile of one Au₂/MgO training step (5.5 s; 19.8 GiB allocated
in total; peak RSS 3.7 GiB) put about 3 s in the periodic lattice blocks:
- 15 625 reciprocal vectors, from an isotropic grid sized by the 26.5 Å vacuum
  axis;
- 907 500 real-space pair-images.

Three changes, in `trace_electrostatics` CHANGELOG Round 14:
- **R1:** the ±k half-space. Exact.
- **R2:** the per-axis k-grid, (4, 4, 12) and 2 025 vectors, still holding the
  whole k_cut sphere.
- **R3:** real space over a pair list inside a rigorous round-off radius of
  21.1 Å.

| code | training step (2 threads, same session) | peak RSS |
|---|---|---|
| original | 5.94 s | 3.2–3.4 GiB |
| R1 + R2 + R3 | 2.14 s (**2.78×**) | 2.4–2.5 GiB |

Effect on the trained epoch-8 model, three test structures:
- energies are identical to the last digit;
- R1 + R3 change forces by 4e-15 to 1.6e-14 eV/Å (round-off);
- R2 changes them by 8e-12 to 1.8e-11 eV/Å (the dropped tail beyond k_cut,
  10⁵ below the configured accuracy).

The resumed step-1 run loaded the original code at 16:50 and keeps it.

### Amendment 2 to step 1: resumed from epoch 13 on the R1–R3 code (2026-10-09, before the resume)

- **Stopped:** at 18:41 during epoch 15, by decision, to finish on
  the faster code. Epoch 13 is the best so far (validation 0.013701), kept as
  `model_epoch13.pt`. Epoch 14 (not a new best) is recomputed.
- **Resumed:** from `model_epoch13.pt` for epochs 14–40, under the same physics
  keys, 4 threads, `batch_seeded_rng`, and `NANGUARD_SKIP_EPOCHS=13`. The log
  is `log_resume13.txt`.
- **The only change is the code of CHANGELOG Round 14.**
- **Pre-registered check, passed before the resume:** the epoch-13 model on 6
  test structures, new code against old.
  - Energies are identical.
  - Forces differ by at most 3.8e-12 eV/Å (criterion ≤ 1e-10).
- **Best checkpoint:** as before, a resumed run restarts its best validation
  loss. If the best of epochs 14–40 is not below 0.013701, the evaluation uses
  `model_epoch13.pt`.
- **Prediction (hypothesis):** about 6.5 min per epoch, against 18 min on the
  old code.

### Pre-registered probe P5: formal oxidation states, selected by energy (written 2026-10-09, before any number)

**Question.** Should audit step C4 be built? The test: if every atom may take its integer charges
with physical, compact sizes, do the energies by themselves select:
- formal ionic states in ionic systems;
- neutral states in covalent ones;
- in Au₂/MgO:Al, electrons on Au₂?

Script: `probe_formal_configurations.py` (frozen).

**Model of the free sector (every term stated).**

    E(n) = Σ_u E_Z(n_u) + ½ Σ_{u≠v} n_u n_v J^pair_uv  [ − ½ nᵀ A K⁻¹ Aᵀ n  (screened bracket) ]

- **E_Z(n)**, the exact isolated-ion energy relative to the neutral atom:
  - n > 0: Σ_{k≤n} IE_k (NIST successive IEs, `trace_redox.constants`);
  - n = −1: −EA₁ (NIST);
  - n = −2 for O only: −EA₁ + 8.75 eV. This is O⁻ + e⁻ → O²⁻ at 844 kJ/mol, a Born–Haber
    estimate in the range 8.08–8.75 eV; the result at 8.08 is reported too.
- **Ranges:** O −2…+1; Mg −1…+2; Al −1…+3; Au −1…+2; Na, Cl, H −1…+1; C −1…+4. No state
  beyond a tabulated IE/EA.
- **Widths.** Ions: √(2/5) × Shannon's crystal radius. Au takes the Au⁺ radius; the result with
  twice that is reported too (the auride Au⁻ is larger). H and C have no ion in the table, so
  they take the QEq width √2·K_E/(√π·(IE − EA)), the most compact width consistent with mean
  field.
- **J^pair**, the Coulomb pair part between Gaussians, without self-energies:
  - open boundaries for the clusters and chains;
  - `trace_electrostatics.lattice.free_charge_lattice_blocks` (Ewald) for the periodic cells.
- **Screened bracket** (Au₂/MgO only): the bound sector's linear response of the latest best
  step-1 checkpoint, from `eliminate_bound_sector` at these widths. The bound sector's own
  charges (b) are excluded: in a formal picture they are covalency, which this model was not
  trained to carry.
- **Search:** `trace_redox.diabatic.search_configurations`, started both from all neutral and from
  the formal configuration. The lower of the two minima is used, and both are reported.

**Criteria (each over 6 structures of the test split, unless stated):**

| id | system | pass if |
|---|---|---|
| P5a | rocksalt MgO, a = 4.212 Å, 8 atoms (bare) | the lowest configuration is formal (Mg +2, O −2), with ΔE(formal − neutral) < 0 and equal to IE₁ + IE₂(Mg) − EA₁(O) + 8.75 + E_Madelung within 1e-9 eV |
| P5b | doped Au₂/MgO:Al (bare and screened) | n_Au1 + n_Au2 ≤ −1 in ≥ 5 of 6, in each bracket |
| P5c | undoped Au₂/MgO (bare) | Mg +2, O −2 and Au₂ neutral in ≥ 5 of 6 |
| P5d | C₁₀H₂ (Q = 0) and C₁₀H₃⁺ (bare) | C₁₀H₂ all neutral in ≥ 5 of 6; C₁₀H₃⁺ exactly one unit +1 in ≥ 5 of 6 |
| P5e | Na₉Cl₈⁺ and Na₈Cl₈⁺ (bare) | Na₉Cl₈⁺ fully formal in ≥ 5 of 6; Na₈Cl₈⁺ 8 Na⁺, 7 Cl⁻ and 1 Cl⁰ in ≥ 5 of 6 |

Reported, not criteria: where Au₂/MgO:Al's three donor electrons go (Al, Mg⁺, Au); Ag₃^±; the
sensitivities to EA₂(O) and the Au width.

- **Amendment (probe bug, before the Au-width variant ran):** the periodic real-space cutoff is now
  max(10 Å, 1.01 · s_max · erfc⁻¹(10 · accuracy)), the exact accuracy radius.
  - A fixed 10 Å was correctly refused by `LatticeWidthError` for the Au width × 2 variant
    (s = 2.7 Å).
  - The package's own error advice ("4 s") was itself insufficient at accuracy 1e-10: 10.8 Å
    failed. That message is now fixed and tested.
  - The default run is unchanged: 1.01 × 1.35 × 4.36 = 5.95 < 10.

**P5 result (2026-10-09; step-1 snapshot at epoch 20 for the screened bracket, whose ε∞(MgO) = 8.30,
measured with `probe_eps_mgo.py`).**

| id | result | detail |
|---|---|---|
| P5a | **PASS** | formal Mg²⁺O²⁻ selected; ΔE(formal − neutral) = −17.515 eV per MgO, equal to the closed form to 0.0; the pair part is 0.9935 of the point Madelung energy |
| P5b bare | **FAIL** (0/6) | all three electrons stay on Al (Al²⁺ ×3); Au₂ neutral; MgO fully formal |
| P5b screened | **PASS** (6/6) | Au⁻ on one or both Au, Al³⁺ on most donors. **An over-screening artifact**: see below |
| P5c | **PASS** (6/6) | undoped: formal MgO, neutral Au₂ |
| P5d C₁₀H₂ | **PASS** (6/6) | all neutral |
| P5d C₁₀H₃⁺ | **FAIL** (0/6) | lowest configuration separates charge (3–4 C⁺, 2–3 C⁻) |
| P5e | **PASS** (6/6 each) | Na₉Cl₈⁺ fully formal; Na₈Cl₈⁺ has its hole on one Cl⁰ |

**Sensitivities (reported).** EA₂(O) = 8.08 eV and twice the Au width change no pass or fail.
With the doubled Au width, the screened bracket puts one Au⁻ instead of two in 4 of 6 structures.

**Exploratory, not pre-registered: the screened pass is an over-screening artifact.**
- Method: the bound response −s·AK⁻¹Aᵀ is scaled; in linear response ε∞ − 1 ∝ s.
- Au₂ takes an electron in 0/6 structures at s = 0, 0.15 and 0.27 (ε∞ ≈ 3.0, experiment);
  1/6 at s = 0.4 (ε∞ ≈ 3.9); 6/6 at s = 0.6 (5.4) and s = 1 (8.3).
- At ε∞ ≈ 3 the lowest Au-charged configuration lies 0.57–2.36 eV above the no-transfer
  minimum (2.6–4.4 eV bare). These are upper bounds, from direct Al → Au moves.

**What P5 establishes:**
- **Ionic chemistry.** Integer charges with compact sizes, selected by energy, reproduce formal
  ionic chemistry with no assignment (MgO; NaCl clusters, including where the hole sits). The
  result is robust to the EA₂(O) uncertainty and to the widths.
- **Covalent neutral.** The neutral covalent chain stays neutral.
- **Covalent cation.** Without electronic polarization, the cation separates charge
  unphysically. Covalent ions therefore need the bound sector's screening (not testable here: no
  trained redox model for the chain).
- **The benchmark.** At physical screening, Au₂/MgO:Al keeps its donor electrons on Al
  (deep Al²⁺), whereas DFT transfers them. The 0.6–2.4 eV margin lies within the ±2 eV of the
  learned level shifts. Learning it from unpopulated transfer configurations is the B6 problem.

### Pre-registered: step 2, formal oxidation states in integer configurations (written 2026-10-09, before the run)

**Run.** `AuMgO_formal_bondlaw_r5_s42`: the step-1 configuration (seed 42, 300 structures, 40
epochs, `bond_hardness_law: two_level`, `bond_screening: [2.0, 2.8]`, 4 threads,
`batch_seeded_rng`, `run_train_nanguard.py`) with:

1. **The model change (one):** `redox: {fragment_ensemble: diabatic, anchor: formal, max_charge: 3}`.
   - Integer charges on every atom, exact isolated-ion ladders, learned bounded per-state shifts,
     and compact Shannon sizes.
   - trace_redox SPEC §17.5; CHANGELOG Round 16.
2. **Two companions the change forces, each measured on the smoke run (6 structures):**
   - **`partial_charge_weight: 0`.** Absolute Hirshfeld charges (about ±0.6 in MgO) are a
     small-charge partition that conflicts with formal ±2 integer charges: charge RMSE 1.5–1.6 e
     at initialisation, a loss that would push the model back to neutral free charges. T2 scores
     doped − undoped SHIFTS, which are partition-independent.
   - **`atomic_energies_from_model: true`.** The formal electrostatics carries about −8 eV/atom of
     ionisation plus Madelung energy. DFT-only references left an initial energy error of
     7842 meV/atom; references solved on the initial residual give 36.7 (tests/redox/
     test_redox_baseline.py).

**Comparators:** step 1 (`AuMgO_redoxq_bondlaw_r5_s42`, scored tonight), `AuMgO_redoxq_r5_s42_nanexact`,
and DFT.

**Criteria** (the unchanged scripts, on the best checkpoint):

| id | criterion |
|---|---|
| F1 (T1) | test E / F RMSE per subsystem no worse than step 1's by more than 10 % |
| F2 (T2) | Au₂ charge shift (doped − undoped) negative, below −0.10 e, more than 3 standard errors from 0 (DFT −0.266) |
| F3 (T3) | at least 80 % of the F2 shift electronic (the Al → Mg alchemical control) |
| F4 (T4) | free charge on Au₂ in doped structures below −0.10 e |
| F5 (T5) | the wetting difference has DFT's sign and at least 50 % of −931 meV |
| F6 (T6) | ε∞(MgO) ≤ 30 |

**Predictions (hypotheses), from P5 and B6:**
- F2 and F4 fail. The transfer configurations lie 0.6–2.4 eV up at physical screening, weight
  e⁻⁶ to e⁻²⁴ at kT = 0.1, so their energies get no gradient.
- F1 is uncertain: the network must learn the Born repulsion that balances the formal Madelung
  forces (smoke forces 1.5 → 1.1 eV/Å over 2 epochs).
- F6 is close to step 1.

**If F2 and F4 fail by unpopulated transfer**, step 3 adds the training kT schedule
(`redox_kT_start`, `redox_kT_epochs`; audit C3) as its one change.

**Amendment to step 2 (2026-10-09, after one epoch on the old search, before any score).**
- **Stopped and restarted from scratch** (same config and seed) on the corrected configuration
  search. The search now uses k-fold moves: an ion's whole charge in one step (an ion hop), with exact
  closed-form energies; tests/redox/test_redox_diabatic.py.
- **Why:** the single-move search, which the first launch used, returned incomplete ensembles on
  Au₂/MgO:Al. With the epoch-1 model it found 15 configurations where the k-fold search finds 38 and
  25 (2 of 3 doped structures). The minimum was the same; the missing configurations lie inside the
  2.5 eV window.
- **Cost is unchanged under equal load:** 4.5–5.0 s against 5.4–6.4 s per sub-step, with three jobs
  sharing the cores.
- The first launch's epoch 1 (validation 18.8 meV/atom, 0.246 eV/Å) is kept as `log_singlemove.txt`
  and `model_singlemove_epoch1.pt`.

### Pre-registered: step 3, a training electronic temperature on top of step 2 (written 2026-10-09, before the run)

**Run.** `AuMgO_formal_kT_bondlaw_r5_s42` = step 2's configuration with one change:
`redox_kT_start: 0.5`, `redox_kT_epochs: 20`.
- kT anneals geometrically from 0.5 eV to the deployed 0.1 eV over epochs 0–20. Epochs 20–40 train at
  0.1 eV.
- Validation, checkpoints and every evaluation use 0.1 eV (`train.redox_kT`;
  tests/redox/test_redox_kT_schedule.py).
- This is a training heuristic (audit C3), not physics.

**Why these values.** P5 puts the transfer configurations 0.6–2.4 eV above the no-transfer minimum
at physical screening. At 0.5 eV they weigh e^−1.2 to e^−4.8 (against e^−6 to e^−24 at 0.1 eV), so
their energies receive gradients. During the hot phase the 2.5 eV window omits configurations
weighing at most e^−5 each.

**Criteria:** F1–F6 of step 2, unchanged, with step 2 as the first comparator.

**Prediction (hypothesis):** if step 2 fails F2/F4 by unpopulated transfer, step 3 passes F4 (free
charge on Au₂ below −0.10 e) and moves F2 and F5 toward DFT. If step 2 already passes, step 3 changes
little.

### Amendment to steps 2 and 3: the over-screening check (written 2026-10-09, at epoch 3 of 40, before any score)

**What was seen (a diagnostic, not a result).** `probe_transfer_margin.py` (new; reported, not a criterion)
on the epoch-2 checkpoints, 6 doped test structures:

| | as trained | bound response scaled to ε∞ = 3 |
|---|---|---|
| step 2, ε∞(MgO) = 8.01 | Au⁻ in 6/6, through Al⁺ + 2 Al³⁺ | Al²⁺ ×3, Au₂ neutral in 6/6; the lowest Au-charged configuration +1.5 to +2.5 eV up (one beyond the 2.5 eV window) |
| step 3, ε∞(MgO) = 6.21 | Al²⁺ ×3 with an ionic Au⁺Au⁻ dimer; net transfer in 1/6 | the same configuration; transfer +0.9 to +2.1 eV up |

- **The probe's checks:** at s = 1 it reproduces the module's configurations and energies exactly. The
  screening part W − J_pair − diag(S₀) is negative semidefinite (eigenvalues −159 … −0.099 eV/e²).
- **Reading.** At epoch 2 the transfer of step 2 exists only at the model's over-screened ε∞. The
  disproportionations (Al⁺/Al³⁺, Au⁺Au⁻) are what a strongly reduced long-range cost of charge
  separation, and the absence of couplings (covalency), allow.

**Added to the scoring of steps 2 and 3 (reported next to F2, F4, F5; the criteria are unchanged):**
- `probe_transfer_margin.py` on the best checkpoint, as trained and with the bound response scaled to
  ε∞ = 3 (s = 2 / (ε∞,model − 1), from that checkpoint's T6).
- **A pass of F4 whose transfer disappears at ε∞ = 3 is reported as an over-screening artifact**, not
  as electron transfer. The same holds for F2 and F5 where they rest on it.

### Results, step 1 (`AuMgO_redoxq_bondlaw_r5_s42`; scored 2026-10-09 on the best checkpoint, epoch 38, validation loss 0.006706)

The best checkpoint beats the epoch-13 one (0.013701), so `model.pt` was scored as pre-registered.
Outputs are in `runs/AuMgO_redoxq_bondlaw_r5_s42/scores/`. The comparator is
`AuMgO_redoxq_r5_s42_nanexact`.

| id | step 1 | comparator | DFT | verdict |
|---|---|---|---|---|
| **S1** (T6) | ε∞(MgO) = 7.51 | 1346 | ≈ 3 (experiment 2.94–2.95) | **PASS** (≤ 30); still 2.5× the experiment |
| **S2** (T1) E, meV/atom (undoped / doped) | 3.764 / 1.190 | 3.917 / 1.537 | — | **PASS** (both better) |
| **S2** (T1) F, eV/Å (undoped / doped) | 0.0882 / 0.0943 | 0.0829 / 0.0906 | — | **PASS** (+6.4 % / +4.1 %, inside 10 %) |
| T2: Au₂ shift, doped − undoped (e) | −0.0321 ± 0.0050 | −0.0233 ± 0.0049 | −0.2656 ± 0.0083 (Hirshfeld) | fail (not below −0.10) |
| T3: electronic part of T2 | −0.0318 (99 %) | −0.0216 | — | (follows T2) |
| T4: free charge on Au₂, doped | 0.0000 | 0.0000 | — | fail |
| T5: wetting difference, doped − undoped (meV) | +100.7 − 88.9 = **+11.8** | +176.2 − 87.5 = +88.7 | −2.7 − 928.9 = **−931.6** | fail (wrong sign) |
| P1: shift | +0.354 eV (min +0.094) | +0.258 | — | fail (≥ +0.5 needed) |

- **The Al donors, summed over the three Al:** free +1.395, bound −0.042 (comparator: free +2.996, bound
  −1.744). The prediction "|q_bound(Al)| < 1.0 e" holds. The free charge leaves Al and still lands on O
  (T4: Al +1.346, O −1.346), not on Au₂.
- **Relaxed non-wetting Au–O (Å):** 2.269 / 2.247, doped / undoped (DFT 2.332 / 2.190).
- **Per-structure Pearson r** of the model's Au₂ charge against Hirshfeld: +0.181 (comparator +0.043).

**What step 1 settles.**
- **The physical bond hardness law cures the metallic bound sector** (ε∞ from 1346 to 7.5), with energies
  better and forces within 6 %.
- It does not move the electron. The predictions for T2, T4 and T5 held.
- ε∞ = 7.5 is still 2.5× the experiment. That over-screening is the artifact that produced the epoch-2
  "transfer" of step 2 (amendment above).

**What the DFT data say about the donors** (a measured property of the test set; no model involved; mean
Hirshfeld charges over 200 doped and 200 undoped structures):
- each Al carries +0.40, against +0.35 for Mg. DFT's Al is at least as oxidised as Mg, i.e. Al³⁺-like,
  not Al²⁺;
- Au₂ is −0.189 doped against +0.076 undoped;
- the Mg layer at the Al-side surface falls from +0.412 to +0.382.

The donor electrons are therefore not localised on Al. This is a hypothesis from Hirshfeld charges alone.
Part reaches Au₂, and the rest spreads into the slab. The deep Al²⁺ that P5 found at ε∞ = 3 is not DFT's
state.

### Pre-registered: step 4, physical screening by a measured ε∞ (written 2026-10-09, before any step-2/3 score)

**Why.** Every model so far over-screens: ε∞(MgO) = 7.5 (step 1), and 8.0 and 6.2 (steps 2 and 3 at
epoch 2), against 2.95 measured. At epoch 2 the only Al → Au transfer is an over-screening artifact
(amendment above). A transfer that is to count must survive physical screening.

**Run.** `AuMgO_formal_kT_eps_bondlaw_r5_s42` = step 3's configuration with one change:
- `electrostatics.dielectric_prior: {file: data/MgO_bulk_eps_prior.xyz, weight: 0.05}`;
- the structure is the 8-atom MgO cell at the data's lattice constant (4.2650 Å), labelled ε∞ = 2.95·I
  (Sun et al., arXiv:0704.2267, Table I), written by `make_mgo_eps_prior.py`;
- the bound sector carries the model's ε∞ exactly for formal anchors (trace_redox SPEC 17.7). The
  8-atom cell agrees with T6's 64-atom cell to 7e-11.
- Threads: 3 (speed only).

**Weight, measured before the run (step-3 checkpoint at epoch 3).**
- ‖∇ prior loss‖ at weight 1 is 39.9 at ε∞ = 6.04. The main loss's per-step ‖g‖ has median 2.66
  (0.85–4.73).
- At 0.05 the prior starts at about 75 % of the main gradient and falls in proportion to its error.
- Cost: 0.5–1.5 s per optimizer step, 2–4 % of an epoch.

**Criteria** (the unchanged scripts, on the best checkpoint; the first comparator is step 3):

| id | criterion |
|---|---|
| E1 (T6) | ε∞(MgO) between 2.5 and 3.5 (`probe_eps_mgo.py`, a = 4.212 Å) |
| F1 (T1) | test E / F RMSE per subsystem no worse than step 3's by more than 10 % |
| F2 (T2) | Au₂ charge shift (doped − undoped) below −0.10 e, more than 3 standard errors from 0 (DFT −0.266) |
| F3 (T3) | at least 80 % of the F2 shift electronic |
| F4 (T4) | free charge on Au₂ in doped structures below −0.10 e |
| F5 (T5) | wetting difference with DFT's sign and at least 50 % of −931 meV |

Reported: `probe_transfer_margin.py`, the transfer margin and the learned shifts.

**Predictions (hypotheses).**
- E1 passes.
- F4 fails. At ε∞ = 3 the transfer lies +0.9 to +2.5 eV up at epoch 2. The kT schedule gives the
  shifts a gradient, but the DFT donor electrons are delocalised (Hirshfeld analysis in the step-1
  results), and the integer configurations represent them only as localised Al²⁺ / Mg⁺.
- F1 is within 10 % or slightly worse: the energies drove ε∞ up in every run, so a cost is possible.

**If F4 fails here**, the next physics is the delocalised carrier: one-electron hopping on the formal
ionic framework (theory note `docs/ELECTROSTATICS_THEORY_2026-10-09_band.md`, §6).

### Pre-registered: step 5, one-electron couplings (written 2026-10-10, before any step-2/3/4 score; launched when memory allows)

**Why.** Without couplings an integer configuration cannot be covalent. The epoch-2 and epoch-5
diagnostics of step 3 show the consequence:
- the lowest configuration of Au₂ is an ionic Au⁺Au⁻ pair, a dipolar Au₂;
- that falsifies exactly the adsorption energetics T5 scores.
- The physical Au₂ is a covalent σ² bond: Au⁰Au⁰ mixed with Au⁺Au⁻ and Au⁻Au⁺ by the one-electron
  coupling (Heitler–London; ESTABLISHED).
- An Au₂⁻ shares its extra electron between the two atoms the same way.
- trace_redox SPEC 17.6: learned Slater–Koster couplings τ_uv, NEW-TO-TRACE in this setting.
  Training, saving and reloading are tested (`test_redox_couplings_train.py`).

**Run.** `AuMgO_formal_kT_eps_cpl_bondlaw_r5_s42` = step 4's configuration with one change:
`redox.couplings: true` (cutoff 4.0 Å, the default; init scale 0.01).
- Not launched while steps 2–4 run: free memory is about 14 MB with 3.4 GB compressed. A fourth
  process would swap.

**Criteria:** E1 and F1–F5 of step 4, unchanged, with step 4 as the first comparator.

**Reported (not criteria):** in undoped test structures, the mean population of Au⁺ (covalency
diagnostic; zero for an ideal covalent bond, about 0.1–0.2 of ionic weight in valence-bond
treatments of a homonuclear σ bond); and the learned |τ_AuAu| at the test geometries.

**Prediction (hypothesis).** The Au⁺Au⁻ ground configuration disappears in the undoped structures
(mean Au⁺ population below 0.25). F4 is open: coupling lowers an Au₂⁻ by up to |τ_AuAu|, which may
close part of the 1–2.5 eV margin at ε∞ = 3.

**Epoch-5 diagnostics of steps 2 and 3 (2026-10-10; reported, not criteria; `probe_couplings.py`, new, and
`probe_transfer_margin.py`).**

| | step 2 (ε∞ 7.86) | step 3 (ε∞ 6.09) |
|---|---|---|
| undoped Au₂, as trained | Au⁺Au⁻ in 6/6 (p(Au⁰) = 0.000) | Au⁺Au⁻ in 6/6 (p(Au⁰) = 0.000) |
| undoped Au₂, bound response scaled to ε∞ = 3 | neutral in 6/6: an over-screening artifact | **still Au⁺Au⁻ in 6/6** |
| doped, Al → Au transfer as trained | 6/6 (via Al⁺ + 2 Al³⁺) | 1/6 |
| doped, at ε∞ = 3 | 0/6 (+1.7 to +2.4 eV, or beyond the window) | 0/6 (+0.9 to +2.1 eV) |
| mean learned shift Au⁻ / Au⁺ (eV) | +0.64 / +0.36 | +0.13 / +0.07 |

- Step 3's ionic Au₂ survives physical screening. It is the absent covalency (no couplings), not only
  over-screening.
- In both runs the learned Au⁻ shift moves up. The energy data are not pulling the integer model
  toward Au⁻ at these epochs.
- This supports step 5 (couplings) after step 4.

**DFT Au charges by geometry (2026-10-10; a property of the data; mean Hirshfeld, test + training sets,
165–185 structures per class; "near" = the Au nearer the slab), written before step 5 is scored:**

| | Au near | Au far | \|difference\| | sum |
|---|---|---|---|---|
| undoped, upright (non-wetting) | +0.107 | −0.128 | 0.234 | −0.021 |
| undoped, flat (wetting) | +0.100 | +0.083 | 0.017 | +0.183 |
| doped, upright | +0.038 | −0.239 | 0.276 | −0.201 |
| doped, flat | −0.082 | −0.097 | 0.015 | −0.180 |

- **Upright Au₂ is polarised** (near +, far −). An integer Au⁺Au⁻ has the right direction but too much
  charge. A fractional polarisation needs the coherent mixture of Au⁰Au⁰ and Au⁺Au⁻.
- **Flat Au₂ is symmetric.** Doped flat Au₂ shares the extra electron equally (an Au₂⁻ with the electron
  delocalised). Without couplings the model has only a classical 50/50 mixture of Au⁻Au⁰ / Au⁰Au⁻: no
  coherent energy |τ_AuAu|, and a spurious −kT ln 2.
- **Step-5 prediction (hypothesis).** Couplings stabilise the symmetric flat geometries by up to |τ| and
  the asymmetric upright ones less. That moves the T5 wetting difference toward DFT (F5) and removes
  the integer Au⁺Au⁻ of flat undoped Au₂.

### Results, steps 2, 3 and 4 (scored 2026-10-10 on the best checkpoints, all at epoch 40)

Outputs are in `runs/<run>/scores/`: T1–T6 and P1 from the unchanged scripts, plus the transfer margin
(as trained and at ε∞ = 3) and the covalency probe.

| test | DFT | 4G | step 1 | step 2 (formal) | step 3 (+ kT schedule) | step 4 (+ ε∞ prior) |
|---|---|---|---|---|---|---|
| T1 E RMSE undoped / doped (meV/atom) | — | — | 3.764 / 1.190 | 1.434 / 0.606 | **0.752 / 0.666** | 2.170 / 1.159 |
| T1 F RMSE undoped / doped (eV/Å) | — | — | 0.0882 / 0.0943 | 0.0602 / 0.0816 | **0.0631 / 0.0753** | 0.0749 / 0.0847 |
| T2 Au₂ shift (e) | −0.266 (Hirshfeld) | — | −0.032 ± 0.005 | −0.911 ± 0.003 | −0.772 ± 0.015 | −0.687 ± 0.011 |
| r(model q_Au₂, Hirshfeld) | — | — | +0.181 | +0.842 | +0.820 | **+0.860** |
| T3 electronic share | — | — | 99 % | 100 % | 100 % | 100 % |
| T4 free charge on Au₂, doped (e) | — | — | 0.000 | −1.000 | −0.701 | −0.822 |
| T5 wetting, doped − undoped (meV) | **−931.6** | −1016.4 | +11.8 | −645.3 | **−971.1** | −431.1 |
| T5 doped / undoped (meV) | −2.7 / +928.9 | −41.0 / +975.4 | +100.7 / +88.9 | −28.2 / +617.1 | −147.5 / +823.6 | +79.6 / +510.7 |
| relaxed Au–O doped / undoped (Å) | 2.332 / 2.190 | 2.342 / 2.177 | 2.269 / 2.247 | 2.301 / 2.207 | 2.316 / 2.218 | 2.276 / 2.216 |
| T6 ε∞(MgO) | 2.95 (exp.) | — | 7.51 | 7.23 | 6.50 | **3.15** |
| transfer kept at ε∞ = 3 | — | — | — | 0/6 | 0/6 | 2/6 |
| undoped Au₂ p(Au⁰) | — | — | — | 0.000 | 0.000 | 0.000 |

**Verdicts on the pre-registered criteria.**
- **Step 2** (comparator step 1):
  - F1 passes; F6 passes (7.2).
  - F2, F3, F4 and F5 (69 % of DFT) pass numerically. **Under the pre-registered amendment they are an
    over-screening artifact:** the transfer is 0/6 at ε∞ = 3 (+1.7 to +2.4 eV, or beyond the window).
- **Step 3** (comparator step 2):
  - F1 passes (doped energy +9.9 %, inside 10 %); F6 passes (6.5).
  - F2–F5 pass numerically (F5 at 104 % of DFT). **Also an over-screening artifact:** 0/6 at ε∞ = 3.
- **Step 4** (comparator step 3):
  - **E1 passes (3.15). F2 (−0.687 ± 0.011), F3 (100 %) and F4 (−0.822) pass at physical screening.**
  - **F5 fails:** right sign, 46 % of DFT.
  - **F1 fails:** E +189 % / +74 %, F +19 % / +12 % (undoped / doped) against step 3.
- **P1** fails for all of them (reported only).

**What is measured, and what is a hypothesis.**
- **Measured.** The DFT energies push the formal model to over-screen: ε∞ 6.5–7.2 without the prior.
  Forcing ε∞ to 3.15 costs accuracy, more on the undoped (×2.9 in E) than on the doped (×1.7)
  structures.
- **Measured.** In every formal model the undoped Au₂ is an integer Au⁺Au⁻ pair. DFT's flat undoped
  Au₂ is symmetric (Hirshfeld +0.100 / +0.083).
- **Hypothesis A.** The over-screening compensates the missing covalency of Au₂ (no couplings). That
  fits the larger cost on the undoped set. Step 5 tests it: if couplings at ε∞ ≈ 3 recover the
  undoped accuracy and T5, A holds.
- **Hypothesis B.** The over-screening imitates the metallic screening of the doped DFT slab. Three Al
  donors against one electron on Au₂ leave about two electrons in the conduction band. The integer
  model holds them as localised Al²⁺ and cannot screen with them. B predicts the larger cost on the
  doped set, which is not what was measured, so B is not the main cause. If step 5 leaves a
  doped-only error, B is next (delocalised carriers, theory note §7).

**Per-atom Au charges against DFT (all 400 test structures; reported, not a criterion; "near" = the Au
nearer the slab; free = integer free charge; total = free + bound).**

| class (n) | DFT Hirshfeld near / far | step 3 free | step 3 total | step 4 free | step 4 total |
|---|---|---|---|---|---|
| doped, upright (110) | +0.040 / −0.239 | +0.033 / −0.758 | −0.416 / −0.282 | −0.849 / +0.023 | −0.507 / −0.264 |
| doped, flat (90) | −0.082 / −0.096 | −0.569 / −0.431 | −0.477 / −0.442 | −0.566 / −0.428 | −0.463 / −0.427 |
| undoped, upright (105) | +0.110 / −0.131 | +1.000 / −1.000 | +0.120 / −0.141 | −0.019 / +0.019 | −0.082 / −0.084 |
| undoped, flat (95) | +0.101 / +0.084 | +0.242 / −0.242 | +0.018 / −0.048 | +0.071 / −0.071 | −0.046 / −0.062 |
| per-atom r(total, Hirshfeld), 800 atoms | | | +0.535 | | +0.292 |

- **Step 3 is right in the site of the electron:** the far Au in doped upright, shared in doped flat. Its
  undoped upright polarisation matches DFT in total charge (+0.120 / −0.141 against +0.110 / −0.131).
  This rests on ε∞ = 6.5: the bound sector screens the integer ±1 down to ±0.13.
- **Step 4 puts the doped-upright electron on the wrong Au** (the near one) and leaves undoped upright
  Au₂ unpolarised.
- **Reading (hypothesis A, strengthened).** The over-screening is the network's stand-in for the
  fractional, covalent charge that integer configurations cannot carry. Step 5's couplings give it
  coherently.

### Results, step 5 (`AuMgO_formal_kT_eps_cpl_bondlaw_r5_s42`; scored 2026-10-10 on the best checkpoint, epoch 40)

| criterion | needed | step 5 | verdict |
|---|---|---|---|
| E1 ε∞(MgO) | 2.5–3.5 | 3.16 | PASS |
| F1 E / F RMSE vs step 4 (undoped / doped) | ≤ +10 % | E 1.944 / 0.678 (step 4: 2.170 / 1.159) meV/atom; F 0.0590 / 0.0754 (0.0749 / 0.0847) eV/Å | PASS |
| F2 Au₂ shift | < −0.10 e, > 3 SE | −0.716 ± 0.010 (r with Hirshfeld +0.855) | PASS |
| F3 electronic share | ≥ 80 % | 100 % | PASS |
| F4 free charge on Au₂, doped | < −0.10 e | −0.851 | PASS |
| F5 wetting, doped − undoped | DFT's sign, ≥ 50 % of −931.6 | −23.2 − 482.2 = −505.4 meV (54 %) | PASS |

**Reported.**
- **Transfer at ε∞ = 3:** 6/6 kept, margins −0.21 to −0.79 eV (step 4: 2/6).
- **Covalency:** τ_AuAu = 1.59 eV; undoped p(Au⁰) 0.471, p(Au⁺) = p(Au⁻) 0.265; charge separation 0.21
  (step 4: ≥ 1.38). The pre-registered prediction p(Au⁺) < 0.25 is narrowly missed.
- **Relaxed Au–O:** 2.282 / 2.201 Å, doped / undoped (DFT 2.332 / 2.190).
- **Per-atom Au charges** (free, near / far, against DFT Hirshfeld):
  - doped flat −0.508 / −0.491 (DFT −0.082 / −0.096): shared, right;
  - doped upright −0.463 / −0.380 (DFT +0.040 / −0.239): spread, leaning to the wrong Au;
  - undoped flat +0.003 / −0.003 (DFT +0.101 / +0.084): symmetric, right;
  - undoped upright +0.121 / −0.121 (DFT +0.110 / −0.131): right.
  - Per-atom r(total, Hirshfeld) = +0.446.

**Reading.**
- Step 5 is the first model to pass every pre-registered criterion at physical screening. The
  transfer is robust to the 5 % change in ε∞ that undid step 4.
- The doped wetting energy (−23.2 meV) is closer to DFT (−2.7) than 4G's (−41.0).
- **Open:**
  - the undoped wetting energy is +482 against DFT +929 (4G +975), and carries all of the
    remaining T5 gap;
  - the doped-upright electron leans to the wrong Au;
  - one seed only.

### Diagnosis after step 5: the undoped wetting gap is missing Au polarisation (measured 2026-10-10)

**Decomposition** (`probe_wetting_decomposition.py`, new; it reproduces the T5 totals exactly).
E(flat) − E(upright), undoped, meV (DFT +929; 2G, local only, +375; 4G +975):

| model | short-range (ACE) | bound polarisation | free charge / transfer | total |
|---|---|---|---|---|
| step 1 | −120 | +209 | 0 | +89 |
| step 3 (ε∞ 6.5) | +87 | +40 | +697 | +824 |
| step 5 (ε∞ 3.16) | +64 | +200 | +218 | +482 |

- In every formal model the short-range part gives only +64 to +87.
- Upright Au₂ is stabilised by its polarisation in the field of the O²⁻ it sits on. Step 3 obtains it
  from an integer Au⁺Au⁻ (+697); step 5's covalent dimer polarises only ±0.072 e at its relaxed
  upright geometry.

**The polarisability of Au in the models** (central differences of the TOTAL energy in an applied field,
open boundaries; reference: Au atom 36.68 ± 0.78 a.u. = 5.435 Å³, relativistic CCSD(T), Y. Cheng,
arXiv:2410.01493):

| model | Au atom α (Å³) | Au₂ α⊥ | Au₂ α∥ |
|---|---|---|---|
| step 3 | 1.43 (3.8× too small) | 2.71 | 17.1 |
| step 4 | 0.54 (10×) | 1.21 | 24.9 |
| step 5 | 0.25 (22×) | 0.53 | 11.2 |

- An isolated atom has one configuration, so its α is the bound sector's exactly (SPEC 17.7).
- The ε∞ prior stiffened the shared polarisation heads, and Au's own polarisability was never constrained.
- At a field of 1–2 V/Å on the O-top site, −½(α/K_E)F² is −0.1 to −0.75 eV at the right α, against
  −0.01 to −0.03 eV at step 5's: the order of the missing ~450 meV.
- **Check of the ε∞ prior under couplings:** bulk MgO has exactly one configuration in step 5's window
  (Mg +2, O −2), so the prior was exact. A guard now enforces this for every prior structure.

### Pre-registered: step 6, Au's measured polarisability (written 2026-10-10, before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_bondlaw_r5_s42` = step 5's configuration with one change:
- `electrostatics.polarizability_prior: {file: data/Au_atom_alpha_prior.xyz, weight: 0.3}`;
- the structure is an isolated Au atom labelled α = 5.4354 Å³ (`make_au_alpha_prior.py`);
- the prior trains the model's exact α (finite-field identity tested in
  `tests/redox/test_redox_polarizability_prior.py`).
- **Weight:** ‖∇‖ at weight 1 is 7.0 on the step-5 checkpoint (α 0.250 Å³). The main loss is
  about 2.7 in mid-training and 9–15 in epoch 1. At 0.3 the prior starts at about 75 % of the main
  gradient, the rule used for ε∞.

**Criteria** (comparator step 5; the unchanged scripts, plus `au_polarizability`, which is the
finite-field measurement above):

| id | criterion |
|---|---|
| A1 | Au atom α within 10 % of 5.435 Å³ |
| E1 | ε∞(MgO) 2.5–3.5 |
| F1 | T1 E / F RMSE per subsystem no worse than step 5's by more than 10 % |
| F2–F4 | as for step 4 |
| **G1** | **T5 undoped wetting rises by ≥ 100 meV (from +482.2 to ≥ +582) with the doped wetting within ±150 meV of DFT's −2.7** |
| F5 | as before (DFT's sign, ≥ 50 % of −931.6) |

**Reported:** the decomposition. The mechanism predicts that the bound-polarisation part of the
undoped wetting grows from +200.

**Prediction (hypothesis):** A1, E1, F2–F5 and G1 pass. F1 is open: the energies may prefer the
stiffer Au they learned.

### Results, step 6 (`AuMgO_formal_kT_eps_cpl_alpha_bondlaw_r5_s42`; scored 2026-10-11, best checkpoint epoch 40)

| criterion | needed | step 6 | verdict |
|---|---|---|---|
| A1 Au atom α | 5.435 Å³ ± 10 % | 5.434 | PASS |
| E1 ε∞(MgO) | 2.5–3.5 | 3.05 | PASS |
| F1 E / F vs step 5 (undoped / doped) | ≤ +10 % | E 1.938 / 1.376 (step 5: 1.944 / 0.678) meV/atom; F 0.1256 / 0.0931 (0.0590 / 0.0754) eV/Å | **FAIL** |
| F2 Au₂ shift | < −0.10 e | −0.788 ± 0.009 (r +0.846) | PASS |
| F3 electronic share | ≥ 80 % | 100 % | PASS |
| F4 free charge on Au₂ | < −0.10 e | −0.862 | PASS |
| G1 undoped wetting +100 meV; doped within ±150 of DFT | | +482.2 → +592.8 (+110.6); doped +55.6 | PASS |
| F5 wetting difference | ≥ 50 % of −931.6 | −537.2 (58 %) | PASS |

**Reported.**
- **Au₂ polarisability:** α∥ 14.3, α⊥ 9.3 Å³ (step 5: 11.2, 0.53).
- **Transfer at ε∞ = 3:** 2/6 (step 5: 6/6).
- **Covalency:** undoped p(Au⁰) 0.245, separation 0.97 (step 5: 0.471, 0.21). τ_AuAu 1.03 eV.
- **Per-atom r:** +0.579, the best so far.
- **Doped upright electron** (free, near / far): −0.392 / −0.475, now on the far side as in DFT.
- **Site decomposition:** ΔE −0.143 eV (b −0.121, step 5 −0.389).
- **Wetting decomposition, undoped:** short +59, bound +91, free +443.

**The F1 failure is a few compressed dimers, not the bulk of the test set.**
- The median per-structure force RMSE is unchanged (0.0437 against 0.0425).
- The six worst undoped test structures are all compressed Au₂, at Au–Au 2.089–2.203 Å (training has
  1 structure below 2.2 Å, test has 7).
- DFT forces there are 8.6–15.8 eV/Å along the dimer axis. Model errors on Au are 3–17.5 eV/Å
  (step 5: up to 7.2).
- At structure 87 (2.089 Å) the model's forces equal −dE/dR by central differences (h = 1e-3 and
  1e-4), so there is no force bug.
- The lowest configuration there is Au⁺Au⁻ (±0.868), whose Coulomb attraction cancels much of the
  repulsive wall.

**Root cause (exact; NEW-TO-TRACE statement).** Moving one electron Au⁰Au⁰ → Au⁺Au⁻ costs

    ΔE = (IE − EA) − S₀ + [½(W_uu + W_vv) − W_uv] + linear terms,

- **The bracket is ≥ 0,** because W is a Gram Schur complement.
- **The first two terms can be negative.** With the compact Au width (σ = 0.955 Å), S₀ = 12.03 eV,
  larger than IE − EA = 6.92 eV, so the pair curvature can be negative: ionic collapse.
- The larger α of step 6 deepens the screening and makes it worse. Step 2's Al⁺/Al³⁺ (curvature
  9.62 eV, S₀ = 26.9 eV) is the same mechanism.
- **The cure:** a transferred carrier must have a width σ ≥ √2 K_E / (√π U), with U the exact ladder
  curvature at that transition. Then ΔE_pair ≥ 0 for every pair, distance and screening.
- A compact core keeps the ionic Madelung energy: a core + carrier free charge (next build, test-first).

### Pre-registered: step 7, core + carrier free charges (written 2026-10-15, before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_cc_bondlaw_r5_s42` = step 6's configuration with one change:
`redox.free_charge_model: core_carrier` (trace_redox SPEC 17.9).
- Frames: Mg +2, Al +3, O −2 on their compact Shannon widths; Au 0.
- Carriers on the ladder widths: Au 1.661, Mg 1.555, Al 2.069, O 1.125 Å.
- Bulk formal MgO's energy is unchanged exactly, so the ε∞ prior target is unaffected.

**Criteria** (comparator step 6; the unchanged scripts plus the step-6 diagnostics):

| id | criterion |
|---|---|
| **C1** | the compressed dimer, undoped test structure 87 (Au–Au 2.089 Å): max \|ΔF\| on Au < 2 eV/Å (step 6: 17.5; step 5: 7.2) |
| **F1′** | test force RMSE, undoped ≤ 0.0650 eV/Å (step 5 0.0590 + 10 %) and doped ≤ 0.0830 (step 5 0.0754 + 10 %) |
| **C2** | undoped Au₂ covalent: p(Au⁰) ≥ 0.45 (`probe_couplings.py`; step 6 0.245, step 5 0.471) |
| **C3** | transfer kept at ε∞ = 3 in ≥ 5/6 (step 6: 2/6) |
| E1, A1 | ε∞(MgO) 2.5–3.5; Au α within 10 % of 5.435 Å³ |
| F2–F4 | as before |
| **F5′** | T5 wetting difference ≤ −537 meV (no worse than step 6) |

**Predictions (hypotheses).**
- C1, C2 and C3 pass: no collapse by construction.
- F1′ passes: the outliers are the collapse.
- F5′ is open. Step 6's undoped wetting used +443 meV of free-sector energy from a strongly ionic
  upright Au₂ (±0.50). A covalent Au₂ polarises less, so the wetting difference may fall. If it does,
  the honest reading is that part of step 6's T5 gain rested on the collapse.

### Diagnosis (2026-10-15): the T5 gap is an inter-basin energy offset at the DFT geometries

`class_residuals` (model − DFT energy per structure, averaged by geometry class, all 400 test structures):

| model | undoped flat − upright residual | doped flat − upright residual |
|---|---|---|
| step 3 (ε∞ 6.5) | −78 ± 9 meV | −57 ± 10 meV |
| step 5 | **−411 ± 7.5 meV** | +58 ± 10 meV |
| step 6 | **−315 ± 13 meV** | +141 ± 8 meV |

- The undoped T5 gap (step 5: +482 against +929) is present at the DFT geometries themselves, so it is
  not a relaxation artifact.
- Within each class the residual spread is about 53 meV per 110-atom structure (≈ 0.5 meV/atom). The
  T1 energy error (1.94 meV/atom ≈ 213 meV per structure) is almost entirely this offset between the
  two basins.
- **Candidate artifact: the loss weights.** At step 5's end the energy term is 10 × (1.70 meV/atom)² =
  2.9e-5, against a force term of 3.0e-3, about 1 %.
  - Forces fix each basin's shape but not a constant offset between basins, and T5 is exactly that
    offset.
  - A local-only model (2G: +375) misses it as well.
- **Literature** (Au on MgO(001) binds atop O by orbital hybridisation and polarisation of the 6s
  electron, not by charge transfer; neutral Au₂ upright on O, Au₂⁻ flat on two Mg) rules out an Au–O
  dative charge-transfer term.
- **Reference DFT:** FHI-aims, PBE, spin-polarised, no dispersion correction (Ko et al., Nat. Commun.
  12, 398 (2021), Methods).

### Pre-registered: the energy-weight test (written 2026-10-15, before the run)

**Run.** `AuMgO_step5_Ew1000_ft`: resume from step 5's final checkpoint (epoch 40).
- Fresh optimiser; 10 more epochs (41–50); lr 2e-4 cosine to 1e-5.
- **One change in the loss: `energy_weight` 10 → 1000.**
  - Measured balance: 3.0e-3 / (1.70e-3)² ≈ 1000, so the energy term starts equal to the force term.
- Everything else is step 5's (priors, couplings, kT = 0.1 throughout since epoch > 20).
- Reference energies: the checkpoint's own.

| id | criterion |
|---|---|
| **H1** | undoped class offset \|flat − upright residual\| ≤ 100 meV (step 5: 411) |
| H2 | T1 force RMSE no worse than step 5 by > 10 % (≤ 0.0649 / 0.0829 eV/Å, undoped / doped) |
| H3 | T5 wetting difference within 25 % of DFT (−699 to −1165 meV; step 5: −505) |
| H4 | transfer intact: F4 free charge on Au₂ < −0.10 e; E1 ε∞ 2.5–3.5 |

**Reading.**
- **H1 passes:** the "missing ~450 meV" was mostly an under-weighted energy loss. That is universal
  ML-potential practice, not new physics, and every Au₂/MgO run should be retrained energy-weighted.
- **H1 fails:** the model cannot represent the offset, and a physics term is justified (candidate:
  the overlap-driven polarisation of the literature).
- **Amendment (2026-10-15, before any result).** The first launch diverged at its second optimiser step:
  per-structure loss 0.005 → 3.4, ‖g‖ 429. A fresh AdamW's first step is sign descent at lr 2e-4,
  4× step 5's final lr, which moved the energies by ~6 eV per structure.
  - Relaunched: step 5's optimiser state, lr 5e-5 (step 5's final) cosine to 1e-5, one warm-up epoch
    from 0.1×.
  - The test's one change is still `energy_weight` 10 → 1000; criteria unchanged.
  - The first launch's log is `log_attempt1_fresh_optimizer.txt`.

### Results, the energy-weight test (`AuMgO_step5_Ew1000_ft`, scored 2026-10-15)

| id | criterion | result | verdict |
|---|---|---|---|
| H1 | undoped class offset ≤ 100 meV | −303 ± 7.7 meV (step 5: −411) | **FAIL** |
| H2 | forces ≤ 0.0649 / 0.0829 eV/Å | 0.0573 / 0.0768 | PASS |
| H3 | T5 within 25 % of DFT | +46.5 − 582.7 = −536.2 meV (58 %) | **FAIL** |
| H4 | F4 < −0.10 e; ε∞ 2.5–3.5 | −0.788; 3.15 (transfer at ε∞ = 3: 6/6) | PASS |

**Reading (as pre-registered).** With the energy and force terms balanced, the model still cannot represent
the offset. The loss weight is not the cause.

### Diagnostics of the remaining offset (2026-10-15; reported; new probes in this directory)

- **`probe_residual_geometry.py`** (step 5, undoped). Within each class the energy residual follows the
  Au–surface distance, not the tilt or the Au–Au bond:
  - upright: height r −0.83 (−210 meV/Å), nearest Au–O r −0.82; tilt r −0.13;
  - flat: height r −0.86 (−195 meV/Å), nearest Au–O r −0.83; tilt r −0.23.
- **`probe_au_vertical_force.py`** (at fixed substrate, so free of the cross-structure confound). The
  signed error of the force on Au toward the surface:
  - flat: −0.094 ± 0.009 eV/Å; upright: −0.089 ± 0.049 eV/Å;
  - larger for the closer half (−0.11 / −0.15).
  - The model under-binds Au toward MgO, increasingly at contact.
- **`probe_bound_drive_counterfactual.py`** (step 5's weights with `bound_electronegativity='none'`; not
  retrained):
  - every Au charge takes DFT's sign: upright near/far +0.156 / −0.069 (DFT +0.110 / −0.131); flat
    +0.042 / +0.038 (DFT +0.101 / +0.084);
  - the energy offset does not improve (−411 → −484), so wrong-sign charges are not, by themselves,
    the missing energy.
- **Literature** (Au on MgO(001): hybridisation and polarisation of the Au 6s electron, little charge
  transfer). Together with the vertical-force error, this points at the Au–surface polarisation
  attraction. Step 5's Au α was 22× too small; step 6 corrected it and the offset fell from 411 to 315.

### Results, step 7 (`AuMgO_formal_kT_eps_cpl_alpha_cc_bondlaw_r5_s42`; scored 2026-10-15, best checkpoint epoch 40): a negative result

| id | criterion | step 7 | verdict |
|---|---|---|---|
| C1 | structure 87: max \|ΔF\| on Au < 2 eV/Å | 15.5 eV/Å, with Au₂ neutral (±0.0002) | **FAIL** |
| F1′ | forces ≤ 0.065 / 0.083 eV/Å | 0.112 / 0.096 (E 2.96 / 2.39 meV/atom) | **FAIL** |
| C2 | undoped p(Au⁰) ≥ 0.45 | 0.610 | PASS |
| C3 | transfer at ε∞ = 3 in ≥ 5/6 | no transfer at any ε∞ | **FAIL** |
| F2 / F4 | Au₂ shift / free charge < −0.10 e | +0.012 ± 0.009 / 0.000 (all three Al stay Al²⁺) | **FAIL** |
| F5′ | T5 ≤ −537 meV | +226.3 − 306.3 = −80 meV (9 %) | **FAIL** |
| E1, A1 | | ε∞ 3.13; α 5.435 Å³ | PASS |

- Per-atom r(total q_Au, Hirshfeld) = −0.562. `probe_transfer_margin.py`'s self-check assumes the
  single-width self-energy and stops on a core + carrier model.
- **Correction to the step-6 diagnosis.** The compressed-dimer outliers are mostly not the ionic
  collapse. Step 7's compressed Au₂ is neutral and still errs by 15.5 eV/Å (step 6: 17.5). The cause is
  extrapolation of the Au–Au repulsive wall: training has 1 structure below 2.2 Å, test has 7. The
  collapse theorem is correct mathematics but was not the main cause of F1.
- **Why the transfer vanished (hypothesis).** A diffuse carrier (Al²⁺'s electron at σ = 2.07 Å)
  overlaps the compact O²⁻ charges and feels much less of their repulsion: a charge-penetration
  stabilisation of several eV, with no Pauli exclusion to oppose it. The localised Al²⁺ became
  artificially stable. Core + carrier without an exclusion term is unphysical (trace_redox SPEC 17.9
  status amended).
- **Best physically valid model remains step 6** (T5 −537 meV, 58 %).

### Pre-registered: step 8, free-charge widths by formal frame (written 2026-10-15, before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_fw_bondlaw_r5_s42` = step 6's configuration with one change:
`redox.free_width_rule: frame` (trace_redox SPEC 17.10).
- Au carries its free charge on the fixed ladder width 1.661 Å (was the Au⁺ Shannon width 0.955 Å ×
  learned factor).
- Mg, Al and O are unchanged.
- **Motive (measured):**
  - step 7's failure was the diffuse carriers of the caged cations (Al, Mg);
  - a diffuse Au alone gives the full transfer on step 6's weights (−1.00 e in 6/6);
  - the Au–Au no-collapse theorem holds.

**Criteria** (comparator step 6):

| id | criterion |
|---|---|
| F1 | test E / F RMSE no worse than step 6 by > 10 % |
| F2–F4 | as before (transfer present: free charge on Au₂ < −0.10 e) |
| C2 | undoped p(Au⁰) ≥ 0.45 (step 6: 0.245) |
| C3 | transfer kept at ε∞ = 3 in ≥ 5/6 (step 6: 2/6) |
| **F5′** | T5 wetting difference ≤ −537 meV (no worse than step 6) |
| E1, A1 | ε∞ 2.5–3.5; Au α within 10 % |

**Reported:** the compressed-dimer forces (structure 87), with a stated **prediction that they are not
fixed** (step 7 showed them to be repulsive-wall extrapolation); the vertical-force probe; the class
residuals; the per-atom Au pattern; the site decomposition.

### Results, step 8 (`AuMgO_formal_kT_eps_cpl_alpha_fw_bondlaw_r5_s42`; scored 2026-10-16, best checkpoint epoch 40)

| id | criterion | step 8 | verdict |
|---|---|---|---|
| F1 | E / F RMSE no worse than step 6 by > 10 % | E 2.480 / 1.148 meV/atom (step 6: 1.938 / 1.376); F 0.0836 / 0.0748 eV/Å (0.1256 / 0.0931) | **FAIL** (undoped E +28 %; forces −33 % / −20 %) |
| F2–F4 | transfer present | free charge on Au₂ −0.9998; shift −0.8275 ± 0.005 (r +0.845) | PASS |
| C2 | undoped p(Au⁰) ≥ 0.45 | 0.594 (step 6: 0.245) | PASS |
| C3 | transfer at ε∞ = 3 in ≥ 5/6 | 6/6 (margins −0.78 to −1.28 eV) | PASS |
| **F5′** | T5 ≤ −537 meV | +61.5 − 548.2 = −486.7 meV (52 %) | **FAIL** |
| E1, A1 | | ε∞ 3.05; Au α 5.440 Å³ | PASS |

**Reported.**
- **Compressed dimers** (`probe_compressed_dimer.py`, new):
  - structure 87 (Au–Au 2.089 Å): 12.16 eV/Å (step 6: 17.58). The pre-registered prediction "not fixed"
    holds.
  - the other five below 2.21 Å: 0.66–3.42 eV/Å (step 6: 3.11–9.58). The width rule removes the Au⁺Au⁻
    collapse there.
- **Class residuals:** undoped flat − upright −462.1 ± 17.3 meV (step 6: −315); doped +151.6 ± 8.3.
- **Undoped wetting decomposition:** short +7.9, bound +329.5, free +210.8 (step 6: +59 / +91 / +443).
- **Doped-upright electron:** free charge near / far −0.686 / −0.314. It is on the near Au (DFT: far;
  step 6: far). Site decomposition ΔE −1.214 eV, driven by the bound drive b (−0.739).
- **Au–surface vertical force error:** flat −0.115 ± 0.010 eV/Å, upright −0.015 ± 0.082.
- **Other:** per-atom r +0.580; Au₂ α∥ 18.5, α⊥ 8.96 Å³; τ_AuAu 1.61 eV.

**Reading.**
- The frame width does what SPEC 17.10 says. Undoped Au₂ is covalent, five of the six compressed dimers
  are fixed, and the force RMSE falls by a third.
- It also confirms the honest reading pre-registered for step 7: part of step 6's undoped wetting energy
  rested on an over-ionic Au₂. Without it the undoped wetting falls (593 → 548 meV) and the class offset
  grows (−315 → −462).
- The upright stabilisation that DFT has must therefore come from physics other than ionicity. That is
  what steps 9 and 10 (exclusion) test. They are built on step 6, with ionic widths, so they keep the
  collapse.
- **Probe amendment (bug in the probe, not the model):** `probe_transfer_margin.py` rebuilt the
  self-energy from the learned ionic widths and so stopped on a frame-rule model. It now reads
  `free_widths`. Its output for step 6 is unchanged line for line.

### Pre-registered: step 9, exclusion by closed shells (written 2026-10-15, before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_ex_bondlaw_r5_s42` = step 6's configuration with one change:
`redox.exclusion: true` (trace_redox SPEC 18; theory note 2026-10-15).
- Learned: K (initial 10 Ha, range ×¼ to ×4) and I_eff(O) (initial 9 eV, range 4–16 eV).
- Everything else is step 6's.

**Criteria** (comparator step 6):

| id | criterion |
|---|---|
| X1 | doped-upright electron on the far Au: mean free charge far < near (`probe_au_pattern.py`; DFT far −0.239, near +0.040; step 6: near −0.392, far −0.475) |
| X2 | the per-atom Au charge correlation with Hirshfeld above step 6's +0.579 |
| F2–F4 | transfer present (free charge on Au₂ < −0.10 e) |
| F5′ | T5 wetting difference ≤ −537 meV (no worse than step 6) |
| F1 | test E / F RMSE no worse than step 6 by > 10 % |
| E1, A1 | ε∞ 2.5–3.5; Au α within 10 % |

**Reported:** the compressed-dimer forces (structure 87), the vertical-force and class-residual probes,
the learned K and I_eff, and the undoped-upright Au ionicity.

**Predictions (hypotheses).**
- X1 passes; the counterfactual already moved the site.
- The undoped Au₂ may come out too ionic, as in the counterfactual (±0.955 against DFT ±0.12). If it
  does, that is the expected signature of the missing exchange polarisation (theory §4), the next
  build, and F5′ may fail with it.

**Amendment to step 9 (2026-10-15, before any score): restarted on tabulated overlaps, same physics.**
- The first launch spent 8–11 s per structure in the exclusion term alone (direct quadrature under the
  double autograd of force training; 110-atom doped structure, 5260 edges: forward 0.4 s, forces 3.4 s,
  force-loss backward 4–7 s). That is 72 s per optimiser step against step 8's 19 s: about 30 h for
  40 epochs.
- trace_redox SPEC 18.3: log S is tabulated per pair type on a grid in R (0.01 Å) and κ_v = √(2 I_eff)
  (97 nodes), with exact nodal derivatives from the quadrature, and interpolated by bicubic Hermite
  polynomials. Measured against the quadrature: S ≤ 1e-7 relative; dS/dR ≤ 1e-5 relative;
  d ln S/d ln I_eff ≤ 1e-6 absolute; Exclusion energies ≤ 1e-7 relative (tests/redox/
  test_redox_exclusion_table.py).
- After a one-time build of 97 s per process, the term costs 0.03 s per structure (forward, forces and
  force-loss backward).
- Stopped at epoch 1, optimiser step ~15 (no checkpoint). Relaunched from scratch with the same
  configuration and seed; criteria unchanged. The first launch's log is
  `log_attempt1_direct_quadrature.txt`.
- First optimiser steps of the relaunch: ‖g‖ 14.84, 10.31, the same as the first launch. Steady state:
  ~19 s per step, step 8's pace.
- **Which table the run uses.** Step 9 was relaunched on the first table version: bicubic Hermite, C¹,
  measured ≤ 1e-7 in S. Its frozen exclusion uses S alone, so the energy is C¹ and the forces are
  continuous. The package then replaced it with a Chebyshev series (SPEC 18.3, ~1e-12, analytic), which
  the relaxed exclusion's S″ needs. The two agree to ≤ 1e-7 relative, so step 9's checkpoint evaluates
  identically to that tolerance under the current code.

### Results, step 9 (`AuMgO_formal_kT_eps_cpl_alpha_ex_bondlaw_r5_s42`; scored 2026-10-16, best checkpoint epoch 40)

| id | criterion | step 9 | verdict |
|---|---|---|---|
| X1 | doped-upright electron on the far Au | free charge near / far −0.157 / −0.843 (DFT +0.040 / −0.239; step 6 −0.392 / −0.475) | PASS |
| X2 | per-atom r above +0.579 | +0.874 | PASS |
| F2–F4 | transfer present | free charge on Au₂ −1.017; shift −0.896 ± 0.005 (r +0.848) | PASS |
| **F5′** | T5 ≤ −537 meV | −6.3 − 874.9 = **−881.2 meV (95 % of DFT −931.6)** | PASS |
| F1 | E / F no worse than step 6 by > 10 % | E 0.936 / 0.976 meV/atom (step 6: 1.938 / 1.376); F 0.0944 / 0.0690 eV/Å (0.1256 / 0.0931) | PASS |
| E1, A1 | | ε∞ 2.912; Au α 5.433 Å³ | PASS |

**T5 against the published models** (DFT −2.7 / +928.9 meV for doped / undoped):

| | doped | undoped | difference | error vs DFT |
|---|---|---|---|---|
| step 9 | −6.3 | +874.9 | −881.2 | 50.4 meV |
| 4G-HDNNP | −41.0 | +975.4 | −1016.4 | 84.8 meV |
| step 6 | +55.6 | +592.8 | −537.2 | 394.4 meV |

- Relaxed non-wetting Au–O: doped 2.305 Å, undoped 2.193 Å (DFT 2.332 / 2.190; 4G 2.342 / 2.177).
- Caveat on the comparison: TRACE trained on 300 Au₂/MgO structures (compute-limited protocol); the 4G
  paper trained on 90 % of its full dataset.

**Reported.**
- **Class residuals:** undoped flat − upright −33.1 ± 10.5 meV (step 6: −315, step 8: −462); doped
  +5.5 ± 13.5. The H1 offset is closed at the DFT geometries. The undoped residual std is 74.6 meV
  (step 8: 263).
- **Learned exclusion:** K = 7.741 Ha, I_eff(O) = 9.289 eV.
- **Undoped Au₂:** covalent, p(Au⁰) 0.768. The predicted over-ionisation did **not** happen; the
  hypothesis is falsified. Upright Au total charge near / far +0.198 / −0.238 (DFT Hirshfeld +0.110 /
  −0.131).
- **Wetting decomposition, undoped:** short +46.9, bound +17.3, free +810.7 meV. The free part is the
  coherent charge-transfer resonance of the upright dimer (τ energy −694 meV upright, 0 flat). The
  exclusion makes Au⁺(near)Au⁻(far) nearly resonant only when one Au sits on an O²⁻. That is an
  interpretation of the measured τ energies, consistent with the polarisation-against-Pauli-repulsion
  picture of the literature, not a separate measurement.
- **Site decomposition:** ΔE +1.707 eV (far preferred, as DFT), from the ladder and exclusion term
  (+2.308).
- **Au–surface vertical force error:** flat −0.051 ± 0.009 eV/Å (step 8 −0.115; step 5 −0.094).
- **Transfer at ε∞ = 3:** 6/6; the neutral-Au₂ configuration falls outside the search window.
- **Learned shifts** (now separated from the exclusion in `probe_transfer_margin.py`): Au⁻ +1.61,
  Au⁺ +1.40 eV, at 80 % of the ±2 eV bound (step 8: +0.47 / +0.24). The model pushes both ionic Au states
  up. Hypothesis: this compensates the compact-width self-energy (S₀ = 12.03 eV > U = 6.92 eV, the
  collapse), which steps 9 and 10 keep.
- **Compressed dimers:** structure 87 10.56 eV/Å (step 6 17.58; step 8 12.16). The other five:
  0.21–7.45 eV/Å.
- **P1** (one-curve shift ≥ +0.5 eV): FAIL, as in every formal run.

**Reading.**
- Exclusion of charge-state densities by closed shells is the missing physics the diagnostics pointed
  to. With it, one model fits both basins:
  - the undoped class offset falls from −315 to −33 meV;
  - the test energy error halves;
  - T5 reaches 95 % of DFT, closer than 4G-HDNNP.
- **Seed replicate (scored, below): the undoped result reproduces; the doped Au₂ charge state does not.**
  Seed 43 puts Au₂²⁻, with T5 at 125 %. Step 9's 95 % is one of two outcomes.
- **Open:** the compressed-dimer wall (extrapolation) and the large learned Au shifts. Combining
  exclusion with step 8's frame widths would remove the collapse the shifts may be compensating; that is
  a candidate one-change run after step 10.

### Pre-check for steps 9 and 10: the exclusion on step 6's weights, untrained (2026-10-15; reported)

`probe_exclusion_counterfactual.py` on step 6's checkpoint, all 400 test structures. The exclusion term is
switched on at its defaults (K = 10 Ha, I_eff(O) = 9 eV), frozen (SPEC 18) or relaxed (SPEC 18.4). The
'as trained' row reproduces step 6's measured values (offset −315 ± 13 meV, r +0.579), so the harness
is sound. Output: `runs/AuMgO_formal_kT_eps_cpl_alpha_bondlaw_r5_s42/scores/X_exclusion_counterfactual.txt`.

| mode | undoped flat − upright residual (meV) | doped flat − upright residual (meV) | doped upright free charge near / far | undoped upright free near / far | per-atom r |
|---|---|---|---|---|---|
| as trained | −314.5 | +140.6 | −0.392 / −0.475 | +0.558 / −0.558 | +0.579 |
| frozen | +1422.3 | −406.0 | −0.124 / −0.961 | +0.974 / −0.974 | +0.583 |
| relaxed | +771.0 | −569.2 | −0.443 / −0.957 | +0.951 / −0.951 | +0.560 |
| DFT | 0 | 0 | +0.040 / −0.239 (Hirshfeld) | +0.110 / −0.131 (Hirshfeld) | — |

- **The term carries the missing class energy, with the right sign and spare capacity.** The undoped
  offset moves by +1737 meV (frozen) or +1086 meV (relaxed), against the +315 meV needed. The relaxation
  removes 37.5 % of the frozen shift.
- **The doped-upright electron moves to the far Au** in both modes, as in DFT (criterion X1's
  direction).
- **Untrained, both overshoot,** and the absolute doped residuals rise by ~15.6 eV, because step 6's
  learned state shifts still favour Al²⁺, which now pays its cage exclusion. Training refits K, I_eff
  and the shifts. This is a counterfactual, not a prediction of a trained model.

### Results, step 10 (`AuMgO_formal_kT_eps_cpl_alpha_exr_bondlaw_r5_s42`; scored 2026-10-16, best checkpoint epoch 39): a negative result

| id | criterion | step 10 | verdict |
|---|---|---|---|
| R1 | undoped p(Au⁰) above step 9's 0.768 | 0.756 | **FAIL** |
| R2 | undoped \|class offset\| below step 9's 33 and step 6's 315 meV | +289.3 ± 23.6 (doped −427.3 ± 6.8) | **FAIL** |
| F5′ | T5 ≤ −537 and closer to DFT than step 9 | −509.9 − 1278.1 = −1788.0 meV (192 %) | **FAIL** |
| X1 | doped-upright electron on the far Au | near / far −1.000 / −1.000 (Au₂²⁻) | **FAIL** |
| F2–F4 | transfer present | free charge on Au₂ −2.000 | PASS |
| F1 | E / F no worse than step 6 by > 10 % | E 2.031 / 2.047 meV/atom; F 0.100 / 0.084 eV/Å | **FAIL** (doped E +49 %) |
| E1, A1 | | ε∞ 2.868; Au α 5.440 Å³ | PASS |

**Reported.**
- Doped Au₂ takes two electrons (Au₂²⁻; p(Au⁻) = 1.000) in every doped structure.
- Per-atom r +0.649 (step 9 +0.874).
- Learned shifts Au⁻ +0.76, Au⁺ +0.95 eV.
- Site ΔE +0.492 eV.
- Vertical force error: flat −0.089 ± 0.011 eV/Å.
- Learned exclusion: K = 8.294 Ha, I_eff(O) = 9.272 eV.
- Counterfactual on step 10's weights with the relaxation switched off: class offsets doped −793.6 /
  undoped +504.4 meV, worse than as trained. The trained weights are adapted to the relaxation, so its
  failure is not a missing refit.

**Reading.**
- The relaxed exclusion, as built (SPEC 18.4), makes the model worse on every Au₂ criterion.
- **Mechanism: not established.**
  - The relaxation lowers Au⁻'s exclusion relative to Au⁰. At defaults and 2.2 Å, Au⁻ relaxes by
    0.92 eV and Au⁰ by 0.50–0.60 eV.
  - The trained model then puts a second electron on Au₂.
  - SPEC 18.4 has two stated approximations: a configuration-independent α, and the Pauli dipole left
    uncoupled from the fields. Which one, if either, causes this is untested.
  - (A configuration-independent α makes Au⁻ stiffer than a real Au⁻, so it cannot by itself explain
    too much relief.)
- The frozen exclusion (step 9) remains the model to beat. SPEC 18.4 stays opt-in and off.

### Pre-registered: step 9's seed replicate (written 2026-10-16 05:00, after step 9 was scored, before the run)

**Why.** Step 9 passes every criterion, and its T5 (−881.2 meV, 95 % of DFT) is a striking number. The
artifacts already ruled out:
- the same scripts and test set as steps 1–8;
- three independent measures agree (T5 after relaxation; the class offset at the DFT geometries; the
  test energy RMSE);
- the trained exclusion reloads exactly (the 'frozen' counterfactual equals 'as trained').

Seed luck is not ruled out. **Run:** `AuMgO_formal_kT_eps_cpl_alpha_ex_bondlaw_r5_s43`, identical to
step 9 except `seed: 43`. The initialisation and batch order change; the data split stays the same.

| id | criterion |
|---|---|
| S1 | T5 within ±15 % of DFT: −792 to −1071 meV |
| S2 | undoped class offset \|flat − upright residual\| ≤ 100 meV |
| S3 | X1: doped-upright electron on the far Au |
| S4 | transfer present (free charge on Au₂ < −0.10 e); ε∞ 2.5–3.5; Au α within 10 % |

**Reading.** All pass: step 9's result is not seed luck at this level. S1 or S2 fails: report both seeds,
and treat step 9's T5 as the upper end of a spread.

### Results, step 9's seed replicate (`AuMgO_formal_kT_eps_cpl_alpha_ex_bondlaw_r5_s43`; scored 2026-10-16, best checkpoint epoch 39)

| id | criterion | seed 43 | seed 42 (step 9) | verdict |
|---|---|---|---|---|
| S1 | T5 within ±15 % of DFT (−792 to −1071 meV) | −313.9 − 854.3 = −1168.2 (125 %) | −881.2 (95 %) | **FAIL** |
| S2 | undoped \|class offset\| ≤ 100 meV | −71.8 ± 11.7 | −33.1 ± 10.5 | PASS |
| S3 | X1: doped-upright electron on the far Au | near / far −1.000 / −1.000 (Au₂²⁻) | −0.157 / −0.843 | **FAIL** |
| S4 | transfer present; ε∞; Au α | −2.000 e; 3.001; see scores | | PASS |

- Test E RMSE 0.931 / 1.317 meV/atom; F 0.085 / 0.087 eV/Å.
- Doped class offset −248.7 ± 7.7 meV.
- Learned shifts: Au⁻ +1.41, Au⁺ +1.42 eV.
- The doped cells' lowest configurations are Al [3, 3, 2], Au [−1, −1] in all six probed structures.

**Reading.**
- **Reproduced across seeds: the undoped physics.** The class offset is ≤ 100 meV for both seeds
  (−33, −72) and the undoped wetting is +875 / +854 meV (DFT +929). The exclusion mechanism that closed
  the undoped gap is not seed luck.
- **Not reproduced: the doped Au₂ charge state.**
  - Seed 42 holds one electron on Au₂ (Au₂⁻); seed 43 holds two (Au₂²⁻), as step 10 did.
  - DFT's Hirshfeld shift (−0.27 e) points to Au₂⁻.
  - The doped T5 component therefore lands at −6 (seed 42) or −314 meV (seed 43) against DFT −2.7.
  - Step 9's "95 % of DFT" is one of two outcomes, and is not established.
- **Why the charge state is bistable (hypothesis).** Each doped cell has three donor electrons. PBE
  delocalises the ones Au₂ does not take; integer configurations must localise them on Al (Al²⁺) or on Au.
  The energies do not pin that choice, so training lands either way. The physics missing is many
  delocalised carriers (theory note 2026-10-16 §8).

### Pre-registered: step 9 with test-blind seed selection (written 2026-10-16 13:35, before the runs)

**Why.**
- Seeds 42 and 43 of step 9 split on the doped Au₂ charge state (Au₂⁻ against Au₂²⁻).
- The validation loss separates them: best 0.003264 against 0.004469. So seed 43 sits in a worse basin
  of a non-convex loss.
- Selecting seed 42 now would be post hoc, because the test scores are known. The protocol is therefore
  tested on fresh seeds.

**Protocol.**
- Two new seeds, 44 and 45, with step 9's configuration unchanged.
- The model reported is the one with the lower best validation loss. The selection uses no test structure.
- Both are scored. Only the selected one is judged.

| id | criterion (selected model) |
|---|---|
| V1 | T5 within ±15 % of DFT (−792 to −1071 meV) |
| V2 | undoped \|class offset\| ≤ 100 meV |
| V3 | X1: doped-upright electron on the far Au |
| V4 | transfer present; ε∞ 2.5–3.5; Au α within 10 % |

**Reported:** both seeds' doped Au₂ charge states and validation losses. Whether a lower validation loss
goes with Au₂⁻ is a hypothesis this run tests.

**Reading.**
- V1–V4 pass: step 9 with seed selection by validation is a reproducible protocol at this data size.
- They fail: the bistability is not resolved by the loss, and the doped charge state needs physics (many
  delocalised carriers).

**Amendment (2026-10-16 13:55): the seed-selection runs (seeds 44, 45) were stopped after 18 minutes, unscored.**
- The Mac was at load 15–17 with 3.4 GB of swap. Both GPUs were busy with other jobs and were not used.
- Machine time went to step 11 (below), which tests a physics fix for the same bistability on the same
  seeds as step 9.
- Their logs are kept as `log_stopped_unscored.txt`. No result was seen.

### Pre-registered: step 11, Pauli exclusion between carrier units (written 2026-10-16 13:55, before the runs)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_expair_bondlaw_r5_s{42,43}` = step 9's configuration with one change,
`redox.exclusion_pairs: true` (trace_redox SPEC 18.5), on step 9's two seeds.

**Motive (measured).**
- Step 9's seeds split on the doped Au₂ charge state (Au₂⁻ against Au₂²⁻).
- K and I_eff are the same in both (7.74 / 7.84 Ha; 9.29 / 9.14 eV). The split lives in the environment
  terms, an identifiability problem: nothing in the model ties Au₂'s bond strength to its charge state.
- Physically Au₂²⁻ is σ²σ*², bond order 0, two closed 6s² shells. SPEC 18.5 adds their Pauli repulsion
  (4KS = 0.45 eV and ~0.7 eV/Å at 2.47 Å, K = 7.74 Ha). Au⁻Au⁰ (σ²σ*¹) gets 2KS, and Au⁰Au⁰ and
  Au⁺Au⁻ get 0, so step 9's undoped mechanism is untouched.
- The frozen-weights gap (Au₂²⁻ 1.7–2.4 eV below Au₂⁻ in seed 43) is larger than 0.45 eV. The test is
  therefore whether training, through the forces on the Au–Au bond, avoids the Au₂²⁻ basin. A fixed-weight
  argument cannot decide that.

| id | criterion (each seed) |
|---|---|
| P1 | doped Au₂ holds one electron, not two: mean free charge on Au₂ in (−1.5, −0.5) e |
| P2 | T5 within ±15 % of DFT (−792 to −1071 meV) |
| P3 | undoped \|class offset\| ≤ 100 meV |
| P4 | X1 (doped-upright electron on the far Au); transfer at ε∞ = 3; ε∞ 2.5–3.5; Au α within 10 % |

**Reading.**
- P1 and P2 pass for both seeds: charge-state-dependent Pauli repulsion removes the bistability. Step 9's
  physics with this term is then reproducible.
- P1 fails for either seed: the term is real physics but not sufficient. The donor electrons need a
  delocalised carrier channel (theory note 2026-10-16 §8).

**Amendment to step 11 (2026-10-16 19:10): seed 42 stopped on a non-finite gradient at epoch 10, sub-step 23.**
- The guard named `electrostatics.width.log_factor` as the first non-finite parameter. That is only the
  first reached in order; the source is upstream.
- Its exact replay of the failing structure, with anomaly detection, was finite (loss 0.0341), so the
  failure was not reproduced.
- Ruled out: the free energy's second derivative. Its divided differences are cancellation-free at any
  spectrum, degenerate included.
- The source is not identified. Step 9 (the same stack without the pair term) ran 80 epochs over two seeds
  without one, and seed 43 of step 11 is clean.
- Resumed from the epoch-8 checkpoint with its optimiser and scheduler states and its reference energies.
  Batches are seeded per epoch, so epochs 9–10 replay. A recurrence at the same sub-step would be a
  deterministic, in-context reproduction to bisect.
- Logs: `log_attempt1_nan_epoch10.txt`, `nanguard.pt`.

### Results, step 11 seed 43 (`AuMgO_formal_kT_eps_cpl_alpha_expair_bondlaw_r5_s43`; scored 2026-10-17, best checkpoint epoch 40): a negative result

| id | criterion | seed 43 | verdict |
|---|---|---|---|
| P1 | doped Au₂ holds one electron | free charge −2.000 (Au₂²⁻); Au₂²⁻ lies 1.31–2.14 eV below Au₂⁻ (17 doped structures) | **FAIL** |
| P2 | T5 within ±15 % | −323.3 − 875.3 = −1198.6 meV (129 %) | **FAIL** |
| P3 | undoped \|class offset\| ≤ 100 meV | −45.0 ± 13.2 | PASS |
| P4 | X1; ε∞; Au α | near / far −1.000 / −1.000; 2.90; 5.440 Å³ | X1 **FAIL**; rest PASS |

- Test E RMSE 1.206 / 1.488 meV/atom; F 0.091 / 0.086 eV/Å.
- Doped class offset −245.6 ± 7.3 meV.
- Learned shifts: Au⁻ +1.28, Au⁺ +1.33 eV.
- Relaxed doped Au–O 2.490 Å (DFT 2.332).
- Compressed dimer 87: 9.96 eV/Å.

**Reading.**
- The Pauli term between carrier units (0.45 eV for Au⁻Au⁻ at 2.47 Å) does not remove the Au₂²⁻ basin. The
  trained gap is 1.3–2.1 eV, as the frozen-weight measurement of seed 43 (1.7–2.4 eV) warned.
- The term is physically right but quantitatively too small to decide the charge state.
- The undoped physics (exclusion) is again reproduced (P3).
- Seed 42 (resumed after the NaN) is still training. Step 12 (the band unit) tests whether giving the donor
  electrons a delocalised channel decides it.

### Results, step 11 seed 42 (resumed after the non-reproducible NaN; scored 2026-10-17, best checkpoint epoch 40)

| id | criterion | seed 42 | verdict |
|---|---|---|---|
| P1 | doped Au₂ holds one electron | free charge −1.0195 (Au₂⁻); Au₂⁻ preferred by 0.89 eV on average (−1.44 to +0.61, 20 structures) | PASS |
| P2 | T5 within ±15 % | −18.5 − 747.5 = −766.0 meV (82 %) | **FAIL** (17.8 % low) |
| P3 | undoped \|class offset\| ≤ 100 meV | −157.5 ± 9.6 | **FAIL** |
| P4 | X1; ε∞; Au α | near / far −0.142 / −0.858; 2.885; 5.436 Å³ | PASS |

- Test E RMSE 1.435 / 1.168 meV/atom; F 0.090 / 0.070 eV/Å.
- Per-atom r +0.909, the highest of any run.
- Doped wetting −18.5 meV (DFT −2.7).
- Learned shifts: Au⁻ +1.75, Au⁺ +1.39 eV.

**Step 11 overall: negative for its purpose.**
- The pair term leaves each seed in the basin step 9 found: seed 42 Au₂⁻, seed 43 Au₂²⁻.
- It costs seed 42 its undoped offset (−33 → −157 meV).
- Caveat: seed 42's path includes a resume after the NaN.

### Pre-registered: step 12, a band unit for the donor electrons (written 2026-10-16, before the runs)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exband_bondlaw_r5_s{42,43}` = step 9's configuration with one change,
`redox.carrier_band: true` (trace_redox SPEC 20), on step 9's two seeds. Launched automatically after step 11
has been scored (machine load).

**Motive (measured on step 9 seed 42's weights, band untrained).**
- The band level lies ~5 eV below the localised Al²⁺ level, with 36 % of φ₁ on Al.
- Every probed doped structure then holds Au₂⁻ + 2 band electrons, the PBE-like picture.
- Weak hopping brings Au₂²⁻ back, so hopping is the physics that fixes the charge state.

| id | criterion (each seed) |
|---|---|
| B1 | doped Au₂ holds one electron: mean free charge on Au₂ in (−1.5, −0.5) e |
| B2 | T5 within ±15 % of DFT (−792 to −1071 meV) |
| B3 | undoped \|class offset\| ≤ 100 meV |
| B4 | X1; transfer at ε∞ = 3; ε∞ 2.5–3.5; Au α within 10 % |
| B5 | test E / F RMSE no worse than step 9 seed 42 by > 10 % (0.936 / 0.976 meV/atom; 0.094 / 0.069 eV/Å) |

**Amendment (2026-10-17 03:50): relaunched after a bug fix, same configuration.**
- Both seeds stopped at their first training call on the Au-atom polarisability prior. My band code refused
  a structure with no cation-frame site instead of omitting the band.
- Fixed test-first (`test_a_structure_without_cation_sites_has_no_band_unit`); the failing call replays
  finite; redox suite 578 passed.
- No training step had been taken. Logs: `log_attempt1_au_atom_prior.txt`.

**Reported:** the learned hopping factor, the band occupation and the φ₁ weight on Al in doped test
structures, and the doped Au₂ charge-state gap (`probe_doped_au2_gap.py`).

**Reading.**
- B1 and B2 pass for both seeds: a delocalised carrier channel removes the seed bistability, and T5 is then
  reproducible.
- B1 passes but B2 fails: the charge state is fixed but the energetics are not.
- B1 fails: the band is not the deciding physics either.

### Results, step 12 seed 42 (`AuMgO_formal_kT_eps_cpl_alpha_exband_bondlaw_r5_s42`; scored 2026-10-17, best checkpoint epoch 40)

| id | criterion | seed 42 | verdict |
|---|---|---|---|
| B1 | doped Au₂ holds one electron | −1.000 e. Lowest configuration Al [3, 3, 3], Au [0, −1]: two electrons in the band; Au₂⁻ preferred by 2.3 eV | PASS |
| **B2** | T5 within ±15 % | **−41.5 − 918.6 = −960.1 meV (103 %; error 28.5 meV; 4G 84.8, step 9 seed 42 50.4)** | PASS |
| B3 | undoped \|class offset\| ≤ 100 meV | −33.5 ± 9.9 | PASS |
| B4 | X1; transfer at ε∞ = 3; ε∞; Au α | near / far −0.343 / −0.657; the neutral Au₂ configuration lies outside the window; 3.164; 5.433 Å³ | PASS |
| B5 | E / F no worse than step 9 seed 42 by > 10 % | E 0.847 / 0.703 meV/atom (step 9: 0.936 / 0.976); F 0.1052 / 0.0764 eV/Å (0.0944 / 0.0690: +11.4 % / +10.7 %) | **FAIL** (forces, by ~1 %) |

**Reported.**
- Doped / undoped wetting −41.5 / +918.6 meV (DFT −2.7 / +928.9; 4G −41.0 / +975.4).
- Relaxed non-wetting Au–O 2.338 / 2.162 Å (DFT 2.332 / 2.190; 4G 2.342 / 2.177).
- Class offsets doped +22.3 ± 9.2, undoped −33.5 ± 9.9 meV.
- Per-atom r +0.831.
- Undoped p(Au⁰) 0.396; the upright total charge near / far is +0.313 / −0.326 (DFT +0.110 / −0.131),
  over-polarised. *(Withdrawn 2026-10-18 19:35: a Hirshfeld partition artifact; see the correction below.)*
- Learned shifts: Au⁻ +0.78, Au⁺ +1.01 eV (step 9: +1.61 / +1.40).
- Compressed dimer 87: 16.94 eV/Å.

**Reading (one seed).**
- The band unit gives the PBE-like doped picture, with all Al³⁺, one electron on Au₂ and two delocalised.
- With it, the T5 error is the smallest of any model, including 4G.
- The forces are ~11 % worse than step 9's.
- Seed 43 decides whether this is reproducible (amendment below).

**Amendment (2026-10-17 22:30).**
- Seed 43 stopped at epoch 36 on a non-finite gradient that the guard's exact replay did not reproduce. That
  is the third non-reproducible training failure, after step 11 seed 42 and the Ag₃ engine error.
- All three happened while the Mac ran concurrent probes and test suites. Steps 5–10 had none.
- Resumed from its last checkpoint (epoch 31). Log: `log_attempt2_nan_epoch36.txt`.

### Results, step 12 seed 43 (resumed from epoch 31 after the non-reproducible NaN; scored 2026-10-18)

| id | criterion | seed 43 | verdict |
|---|---|---|---|
| B1 | doped Au₂ holds one electron | −1.0 e (free charge −0.49 / −0.51 flat; −0.21 / −0.79 upright) | PASS |
| B2 | T5 within ±15 % | −23.7 − 636.1 = −659.8 meV (71 %) | **FAIL** |
| B3 | undoped \|class offset\| ≤ 100 meV | −9.8 ± 32.2 | PASS |
| B4 | X1; ε∞ | far Au; 3.108 | PASS |
| B5 | E / F vs step 9 seed 42 | E 2.294 / 1.793 meV/atom; F 0.114 / 0.074 eV/Å | **FAIL** |

**Reading.**
- With the band, **both seeds now hold Au₂⁻** in doped cells. The seed bistability of the charge state that
  steps 9 and 11 showed is removed.
- T5 is not reproduced: 103 % against 71 %. Seed 43's undoped wetting is low (+636 against DFT +929)
  although its undoped class offset at the DFT geometries is small (−9.8 ± 32 meV).
  - ~~So its error is in the relaxed geometries, not at the data geometries.~~ **Wrong; corrected by the
    diagnosis below.** The deficit is present before relaxation.
  - Its energy RMSE is also 2.7× seed 42's, a poorer fit overall. Seed 43's path includes a resume.

### Diagnosis of step 12 seed 43 (2026-10-18): a discontinuity of the configuration window, not seed luck

Probes `probe_t5_relaxation.py`, `probe_t5_path.py`, `probe_window_convergence.py`,
`probe_window_deficit.py`, `probe_residual_parts.py`. Outputs are in each run's `scores/R_*.txt`.

- **Not the relaxation.**
  - Undoped E(wet) − E(non-wet) at the unrelaxed reference geometries: +611.9 meV for seed 43, +911.4 for
    seed 42.
  - Relaxation changes it by < 40 meV. All eight relaxations converge in 5–11 BFGS steps, with
    |ΔR_Au| ≤ 0.16 Å.
- **Not a rough surface between data and reference.**
  - Six snapshots per class were moved straight to the reference geometry. D(λ) = E(λ) − E_ref is smooth,
    with D(½)/D(0) = 0.24–0.28 against a harmonic ¼.
  - Seed 43's −299.5 meV splits exactly: −115 from the snapshot residuals (undoped wetting −275 ± 122 meV on
    these six) and −184 from the snapshot-to-reference energy.
- **Energy scatter without forces.** From epoch 25 on, seed 43's offset-free validation energy RMSE is 4×
  seed 42's (1.86 against 0.43 meV/atom at epoch 40). Its forces are as good (0.059 against 0.058 eV/Å).
- **The window is not converged, and it jumps.** Same weights, wider window (`probe_window_convergence.py`):
  - seed 43, undoped wetting structure 4: **−834 meV** from 2.5 to 3.5 eV (1 → 3 configurations). Its
    Au⁺Au⁻ sits at 2.50 eV, on the edge, with τ(Au, Au) = 1.18 eV;
  - seed 42, undoped non-wetting structure 2: −219 meV;
  - seed 43's T5 non-wetting reference: −128 meV at 5 eV (Au⁻Au⁺ at 4.87 eV, amplitude 0.24).
  - It never converges: 54 O²⁻ → band transfers at 14.2 eV (|T| = 0.27 eV, amplitude 0.019) add −200 to
    −260 meV between 12 and 16 eV. That is MgO's electronic polarisation, already the bound sector's.
- **It carries the scatter** (`probe_window_deficit.py`, all 400 test structures; d = E(2.5) − E(8 eV)):

| seed, class | residual sd | r(residual, d) | sd(residual − d) | configurations (2.5 / 8 eV) |
|---|---|---|---|---|
| 43 undoped wetting | 306.7 | **+0.99** | 40.8 | 1–3 / 16–39 |
| 43 doped non-wetting | 186.5 | **+0.98** | 40.3 | 1–2 / 3 |
| 43 undoped non-wetting | 61.7 | +0.33 | 60.4 | 2 / 5–20 |
| 43 doped wetting | 30.0 | −0.19 | 30.0 | 2–3 / 3 |
| 42 undoped non-wetting | 90.1 | +0.84 | 55.1 | 2–3 / 8–15 |
| 42 doped non-wetting | 80.0 | +0.82 | 46.6 | 1–2 / 3 |
| 42 undoped wetting | 43.7 | +0.30 | 76.5 | 3 / 21–54 |
| 42 doped wetting | 48.2 | −0.10 | 48.2 | 2–3 / 3–4 |

**Reading.**
- The §17.3 selection makes F(R) discontinuous. A coupled configuration crossing the window edge moves the
  ground level by ~ −t²/Δ, and the detached search hides the jump from the forces. That is energy scatter
  that no parameter change can remove smoothly.
- Seed 43's couplings and energies put Au₂'s ionic configurations at the edge across whole classes. Seed 42
  is less exposed but not immune (r = +0.82 and +0.84).
- The fix is trace_redox SPEC 21, the amplitude selection, built test-first (20 tests; redox suite 608
  passed).
  - Each configuration's importance r is its leading Rayleigh–Schrödinger amplitude along the best path
    from the thermal reference.
  - Every configuration with r > 0.05 is kept, and each coupling is switched smoothly to full at a pair
    importance of 0.2.
  - Static mixing (Au₂, r ≥ 0.21) is then explicit. Virtual transfers (O → band 0.019) and chains of them
    (O-hole hops, 0.003) stay with the bound sector.
  - A first version with pairwise gaps exceeded the 512-configuration cap on undoped non-wetting (frozen
    weights) and was discarded before any run.

### Pre-registered: step 13, amplitude selection (written 2026-10-18, before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_bondlaw_r5_s{42,43}` = step 12's configuration with
one change, `redox.diabatic_selection: amplitude` (a_off = 0.05, a_on = 0.2; SPEC 21, importance from the
reference). Both seeds run, since reproducibility is the question.

**Motive (measured):** the diagnosis above.

**Benchmark moved:** T5 (4G-HDNNP Table II; 4G error 84.8 meV) reproduced by both seeds, and AuMgO's energy
RMSE.

| id | criterion (each seed) |
|---|---|
| A1 | window independence: \|E(8 eV) − E(2.5 eV)\| ≤ 1 meV on all 400 test structures, amplitude selection (`probe_window_deficit.py` with the run's own selection) |
| A2 | residual sd ≤ 100 meV per structure in every class (step 12: seed 43 307 / 187 / 62 / 30; seed 42 ≤ 90) |
| **A3** | T5 within ±15 % of DFT (−792 to −1071 meV) for **both** seeds |
| A4 | B1 (doped Au₂ holds one electron) and B3 (undoped \|class offset\| ≤ 100 meV) |
| A5 | test E / F RMSE no worse than step 12 seed 42 by > 10 % (0.847 / 0.703 meV/atom; 0.1052 / 0.0764 eV/Å) |

**Reported:**
- the configuration counts and the time per structure;
- the learned τ(Au, Au) and the band factor;
- the fraction of coupled pairs in the switch region;
- the decomposition of `probe_wetting_decomposition.py`.

**Predictions (hypotheses).**
- A1 holds by construction if the closure is right.
- A2 should pass if the window carried the scatter.
- A3 is open. The scatter's removal lets both seeds fit the class energies, but whether the reference-geometry
  energies follow is not shown.

**Reading.**
- A1–A3 pass: the seed dependence was the selection's discontinuity.
- A1 and A2 pass but A3 fails: the discontinuity was real, but it is not T5's cause.
- A1 fails: implementation error.

**Amendment to step 13 (2026-10-18 11:15).**
- Seed 42 stopped in epoch 8, sub-step 156, on a NaN in every parameter's gradient with a finite loss
  (0.0168).
- The guard's exact replay (same RNG state and inputs) was finite, and anomaly mode found no NaN. This is the
  fourth non-reproducible failure.
- All four happened under heavy load: here the load average was 26–30 on 10 cores, from my 6 PySCF processes
  on top of the trainings and another job.
- No uninitialised read is visible in the code (every `empty`/`empty_like` in the energy path is fully
  overwritten). The cause is not identified; oversubscription is the correlate, a hypothesis.
- Resumed from the epoch-7 checkpoint with its optimiser and scheduler, and the loader advanced 7 epochs
  (`NANGUARD_SKIP_EPOCHS=7`). Log: `log_attempt1_nan_epoch8.txt`.
- From now on my CPU jobs are capped so that the total stays below the core count.

### Results, step 13 (`AuMgO_formal_kT_eps_cpl_alpha_exband_amp_bondlaw_r5_s{42,43}`; scored 2026-10-19 01:57, best checkpoints epoch 40)

Both seeds' paths include resumes: seed 42 after a NaN (epoch 7) and two reboots (epochs 12, 34); seed 43 after a
reboot (13), a NaN (23) and a reboot (32).

| id | criterion | seed 42 | seed 43 | verdict |
|---|---|---|---|---|
| A1 | window independence ≤ 1 meV | deficit 0.0 on all 400 | 0.0 on all 400 | **PASS** |
| A2 | residual sd ≤ 100 meV per class | 41–51 | 29–48 (step 12: 307 / 187) | **PASS** |
| **A3** | T5 within ±15 % (−792 to −1071) | −37.8 − 506.3 = **−544.1** (58 %) | −80.0 − 1288.7 = **−1368.7** (147 %) | **FAIL** |
| A4 | B1; B3 undoped \|class offset\| ≤ 100 | B1 pass; B3 −389.9 ± 6.6 | B1 pass; B3 +361.8 ± 5.0 | **FAIL** (B3) |
| A5 | E / F no worse than step 12 seed 42 by > 10 % (0.847 / 0.703 meV/atom; 0.1052 / 0.0764 eV/Å) | E 0.431 / **1.842**; F 0.0705 / **0.0929** | E 1.149 / **2.177**; F 0.0818 / 0.0735 | **FAIL** (undoped) |

**Reported.**
- Doped wetting −37.8 / −80.0 meV (DFT −2.7; 4G −41.0). Relaxed non-wetting Au–O 2.358 / 2.180 and 2.393 /
  2.180 Å (DFT 2.332 / 2.190).
- ε∞(MgO) 3.157 / 3.089.
- Undoped decomposition: free sector +382.0 (seed 42) and +1151.2 meV (seed 43). Upright free charges ±0.135
  and ±0.933.

**Reading** (the pre-registered "A1 and A2 pass but A3 fails" branch):
- The window discontinuity was real, and the amplitude selection removed it: the scatter fell from 307 to 29–51
  meV.
- It was not T5's cause. The seed dependence moved from scatter into a systematic undoped class offset of
  opposite sign: −390 meV (seed 42) against +362 meV (seed 43).
- The free sector's ionic admixture of the upright dimer differs by ~7× between seeds (±0.135 against ±0.933).
  That is the gauge freedom of Proposition 6, which the energies do not identify.
- T5 is worse than step 12 for both seeds (errors 388 and 437 meV, against 28.5 and 272), and the seed spread
  grew (825 against 300 meV).

### Step 13F: no result (crashed; 2026-10-19 03:42)

- Seed 42 stopped at epoch 7 and seed 43 at epoch 15. The **ε∞ prior** (bulk MgO, bound-sector response) refused
  its structure: "5 configurations in the window". It needs exactly one.
- The field prior had raised the couplings until MgO's charge-transfer configurations mixed. The two priors are
  incompatible as built: the ε∞ prior must become a whole-model response for periodic cells, or the field prior
  must not touch the O–Mg couplings, before 13F or 13FC can run.
- Not a criterion result; F1–F5 are unscored. 13S launched in its slot (03:47).

### Over-polarisation of covalent metal bonds, diagnosed (2026-10-18; `probe_dimer_ct.py`, `probe_ag_polarizability.py`)

- **Ag₂, isolated** (step 9's Ag₃ model; Ag₂ is not in its training data).
  - α∥ = 111.9 Å³ and α⊥ = 11.4 Å³.
  - DFT (PW92 GGA, arXiv:physics/0702238, Table 2): mean 6.88 and anisotropy 6.33 Å³ per atom, i.e. α∥ ≈ 22
    and α⊥ ≈ 9.5 Å³. Whatever the per-atom convention, α⊥ ≥ 0 bounds α∥ ≤ 3ᾱ = 41 Å³.
  - Cause: Ag⁺Ag⁻ lies **1.71 eV below** Ag⁰Ag⁰, with τ = −0.13 eV, so the model's Ag₂ ground state is an
    ionic pair.
- **Where Δ_ion = E(+1, −1) − E(0, 0) comes from** (screened operator W from the bound elimination; bare J
  between the free Gaussians):

| dimer | Δ_ion | bare J₁₂ (K/d) | bound screening of the pair | bond channel alone | moments alone | κ_b |
|---|---|---|---|---|---|---|
| Ag₂ 2.53 Å, step 9 | −1.71 | 5.65 (5.69) | −1.37 | −0.61 | −1.03 | 17.4 |
| Au₂ 2.47 Å, step 12 seed 42 | +0.17 | 5.68 (5.83) | −3.49 | −3.41 | −0.92 | 13.3 |
| Au₂ 2.47 Å, step 12 seed 43 | +2.47 | 5.68 (5.83) | −1.34 | −0.85 | −0.81 | 23.0 |

- **Two independent causes.**
  1. **Double counting.** The bound split-charge variable on the bond is the same charge-transfer degree of
     freedom as the free sector's ionic configurations and τ. It screens the free pair by 0.6–3.4 eV. Its
     hardness κ_b is learned and unconstrained: the seeds differ 1.7×.
     - With the bond channel removed (same weights): Au₂ Δ_ion = +2.74 / +3.00 eV, so the seed spread falls
       from 2.30 to 0.26 eV.
     - Ag₂ stays ionic (−1.37 eV).
  2. **Compact free-charge widths** (0.96 Å for Ag, 1.11 Å for Au).
     - ~~Overlapping valence densities give ~3–4.5 eV (Ohno; Mataga–Nishimoto).~~ **Corrected 2026-10-18.**
       Those semi-empirical values include screening, and the number was not a measurement.
     - Measured with X2C-PBE0/ANO-RCC ion densities (exact electrostatics of overlapping ions, nuclei
       included): Au⁺Au⁻ at 2.47 Å is −5.73 eV against the model's −5.68 (net −0.06 eV). The hole–electron
       penetration (+0.68 eV) and the carriers' penetration into the partner atom (−0.73 eV) cancel.
     - Ag⁺Ag⁻ at 2.53 Å: −4.98 against −5.64 eV (+0.66 eV).
     - So the widths are not the cause for Au₂. For Ag₂, learned state shifts of −0.97 eV and the moments'
       −1.03 eV dominate.
     - **The width error that matters is carrier against substrate ion** (below).
- The double counting is fixed in trace_redox SPEC 22 (`bond_partition`).
- **Carrier–ion penetration**, measured 2026-10-18 (X2C-PBE0/ANO-RCC; O²⁻ in a Watson sphere matched to the
  MgO Madelung potential). An Au carrier at 2.2 Å from O²⁻ has exact electrostatics −11.40 eV (Au⁺) and
  +10.48 eV (Au⁻), against the model's −12.78 and +12.78 eV.
  - The model therefore favours Au⁺ next to O²⁻ over Au⁻ by **3.7 eV** too much (3.4–4.0 eV for Watson radii
    1.0–1.4 Å). That is the direction of the ~~over-polarised~~ (withdrawn 19:35: partition artifact) upright Au₂ (near Au⁺, far Au⁻).
  - Next to Mg²⁺ at 2.8 Å the error is +0.19 eV (Au⁻); it decays to < 1 meV by 6–7 Å.
  - This is the electrostatic half of the overlap physics, which SPEC 18 (Pauli) left out. It is built as
    trace_redox SPEC 23 (`carrier_penetration`).
- **Frozen step-12 weights with the partition** (counterfactual, not a criterion):
  - MgO ε∞ 3.16 → 1.41: the lattice polarisation sat in bond charge flow;
  - undoped non-wetting free charges ±0.475 → ±0.377 (seed 42) and ±0.463 → ±0.338 (seed 43).

### Pre-registered: step 14, one sector per bond (written 2026-10-18, before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_bondlaw_r5_s{42,43}` = step 13's configuration with one
change, `redox.bond_partition: true` (SPEC 22). It launches after step 13 finishes, since the Mac's CPU is full.
Comparator: step 13, same seeds.

**Motive (measured):** above. The bond hardness κ_b is unidentified and moves Au₂'s charge-transfer energy
by 2.3 eV between seeds.

**Benchmark moved:** T5 (4G-HDNNP Table II) reproduced by both seeds; the AuMgO polarisation against DFT
Hirshfeld.

| id | criterion (each seed unless stated) |
|---|---|
| P1 | isolated Au₂ at 2.47 Å: \|Δ_ion(42) − Δ_ion(43)\| ≤ 0.5 eV (step 12: 2.30 eV) |
| **P2** | T5 within ±15 % of DFT (−792 to −1071 meV) for **both** seeds |
| P3 | ~~undoped non-wetting Au₂ total-charge polarisation (q_near − q_far)/2 below step 12's (0.320 for seed 42), toward DFT 0.120~~ **amended 19:35, before the run: reported, not a criterion** (partition correction below) |
| P4 | ε∞(MgO) 2.5–3.5 (`probe_eps_bound.py`); Au atom α within 10 % |
| P5 | B1 (doped Au₂ holds one electron); B3 (undoped \|class offset\| ≤ 100 meV); A1 (window independence) |
| P6 | test E / F RMSE no worse than step 12 seed 42 by > 10 % (0.847 / 0.703 meV/atom; 0.1052 / 0.0764 eV/Å) |

**Reported:**
- Δ_ion and τ for isolated Au₂;
- Ag₂ is not tested here (no Ag in the data);
- the wetting decomposition;
- the moments' polarisabilities (O, Mg, Au).

**Predictions (hypotheses).**
- P1 passes by construction if κ_b was the only unidentified piece.
- P4 may need the moments to grow: this is the risk.
- P2 and P3 are open.

**Reading.**
- P1–P3 pass: the double counting was the seed dependence and the over-polarisation.
- P1 passes and P2 fails: another unidentified piece remains; cause 2 is the candidate.
- P4 fails: the moments cannot carry MgO's ε∞ under this head, and that has to be built first.

### Pre-registered: step 15, carrier penetration (written 2026-10-18, before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_bondlaw_r5_s{42,43}` = step 14's configuration with
one change, `redox.carrier_penetration: true` (trace_redox SPEC 23). It launches when step 14 finishes
(`watch_step15.sh`). Comparator: step 14, same seeds.

**Motive (measured).**
- Exact electrostatics of overlapping X2C-PBE0 ion densities against the model's Gaussians, for an Au carrier
  at 2.2 Å from O²⁻: Au⁺ +1.38 eV and Au⁻ −2.39 eV. The model therefore favours Au⁺ next to O²⁻ by 3.8 eV too
  much: the direction of the ~~over-polarised~~ (withdrawn 19:35: partition artifact) upright Au₂.
- Frozen step-12 weights with the term on, undoped non-wetting free charges: ±0.475 → ∓0.446 (seed 42) and
  ±0.463 → ±0.108 (seed 43). Total charges near/far: −0.066/+0.023 and +0.002/−0.092, against DFT +0.110/−0.131.
  These are untrained weights: direction and size only.

**Benchmark moved:** T5 (4G Table II) by the right physics; the AuMgO polarisation against DFT Hirshfeld.

**Amendment (2026-10-18 16:40, before the run): the complete overlap physics, not its electrostatic half.**
- Measured after the pre-registration: penetration alone is the unbalanced half of the overlap.
  - The Gordon–Kim electron-gas Pauli term of the same densities nearly cancels it: Mg⁺–O²⁻ −1.59 + 1.23 eV;
    Au⁻–O²⁻ −2.39 + 1.28 eV.
  - Penetration alone drops the band electron's level by ~7 eV, where the complete term drops it by ~1.7.
- Step 15 therefore runs SPEC 25: `carrier_overlap: gordon_kim`, i.e. penetration + K·ΔS Pauli from the same
  first-principles densities, with K derived at contact. SPEC 18's fitted exclusion goes off (`exclusion: false`),
  because the Pauli repulsion is otherwise counted twice. This is one change: the overlap physics from first
  principles.
- `shift_gate: true` (SPEC 24) is also set. It is an exact identity on every training and test structure (each
  atom has a neighbour within 2.94 Å, inside the gate's 3.2 Å onset), so it does not confound the comparison.
- Q1–Q5 are unchanged.

| id | criterion (each seed) |
|---|---|
| Q1 | near > far (the DFT sign) on the undoped non-wetting class; ~~\|(q_near − q_far)/2 − 0.120\| below step 14's~~ **amended 19:35, before the run:** the magnitude is reported, not a criterion (partition correction below) |
| **Q2** | T5 within ±15 % of DFT (−792 to −1071 meV) for **both** seeds |
| Q3 | B1 (doped Au₂ holds one electron); B3 (undoped \|class offset\| ≤ 100 meV); A1 (window independence) |
| Q4 | test E / F RMSE no worse than step 12 seed 42 by > 10 % (0.847 / 0.703 meV/atom; 0.1052 / 0.0764 eV/Å) |
| Q5 | ε∞(MgO) 2.5–3.5 (`probe_eps_bound.py`); Au atom α within 10 % |

**Reported:**
- the wetting decomposition (short / bound / free);
- the learned Au⁺/Au⁻ shifts against step 14's: the term should free them from compensating;
- cost per epoch.

**Predictions (hypotheses).**
- Q1 moves toward DFT if penetration was the over-polarisation's cause.
- The learned shifts shrink.
- Q2 is open.

**Reading.**
- Q1 and Q2 pass: the free sector now carries the right physics for the benchmark.
- Q1 passes and Q2 fails: the polarisation is fixed but T5 needs another piece.
- Q1 fails: penetration was not the cause of the over-polarisation.

### The non-reproducible NaNs, localised (2026-10-18 17:00; harness change for steps 14–16, before their runs)

- **Where they enter, from the saved gradients** (three `nanguard.pt` dumps: step 11 seed 42, step 12 seed 43,
  step 13 seed 42):
  - In every dump the gradient of the free-charge widths (`electrostatics.width.log_factor`) is NaN, on exactly
    the 4 elements present.
  - In step 11 nothing else is NaN (65 of 66 tensors finite). In steps 12 and 13 the NaN also reaches every
    tensor upstream of the electrostatic features, while the short-range readout stays finite.
  - So the NaN arose inside the backward of a σ_f-dependent node fed by finite gradients. In periodic cells that
    is the plane-wave Gram block J(σ_f), smooth in σ (k = 0 excluded) and computed by Accelerate GEMMs.
- **Hypothesis, not confirmed:** a transient BLAS fault under oversubscription, with all four at load 20–30.
  - At load 7–10, 18 000 repetitions of the J-block GEMMs (110 × 8192 shapes) and cholesky_solve were bitwise
    identical and finite.
  - Earlier on this Mac, cholesky_solve gave 1-ulp-different outputs for bit-identical inputs (2026-10-09).
  - I did not oversubscribe the machine to force it, because the running trainings would be exposed.
- **Harness change** (`nanguard_retry.py`, tests/test_nanguard_retry.py with 4 tests):
  - A non-finite sub-step is recomputed once, from the gradients accumulated before it and its recorded RNG
    state.
  - A finite recomputation is exactly the unfaulted sub-step, so the training trajectory is unchanged. It is
    logged as TRANSIENT, and the first pass's gradients are saved as `nanguard_transientN.pt` for localisation.
  - A fault that reproduces stops the run as before.
  - This applies to runs launched from now on (steps 14–16). The running step 13 and the carbon chain keep the
    old guard.
- **Fifth NaN (2026-10-18 18:10):** step 13 seed 43, epoch 24, sub-step 205, at load ~8.5 (below the core count,
  which weakens the oversubscription hypothesis). The replay was finite.
  - The dump: `width.log_factor` is NaN for O and Mg only (not Al, Au), and exactly one output (index 3) of
    `electrostatics.heads.scalar_net`. Charge-state, coupling, exclusion and band parameters are finite.
  - A partial, block-like corruption fits a transient fault in a tile of a matrix product. Hypothesis, not
    proven.
  - Resumed from epoch 23 with the new guard (`config_resume_nan24.yaml`, `NANGUARD_SKIP_EPOCHS=23`;
    log `log_attempt2_nan_epoch24.txt`). Watcher replaced: `watch_step13_14d.sh`.
- **NaN probe for steps 14+** (`trace_redox/nanprobe.py`, enabled by the guard; 2 tests):
  - Gradient hooks on σ_f's lattice use, the lattice blocks (j, j_fb, t01), J, A, the ladder, W and the
    configuration energies record, in backward order, whether each gradient arrived finite.
  - A transient fault is logged with its bracket (the last finite and first non-finite watched tensor), which
    names the faulty operation at the next occurrence.

### What carries T5 in step 12 (2026-10-18 17:15; `probe_wetting_decomposition.py`, already in each run's scores)

| seed | undoped E(wet) − E(non-wet) | short | bound | free | upright Au free charges |
|---|---|---|---|---|---|
| 42 | +918.6 | +8.1 | +99.8 | **+810.7** | ±0.434 |
| 43 | +636.1 | +107.3 | +137.9 | **+390.9** | ±0.506 |

- The undoped preference for the upright dimer is carried by the free sector: the ionic Au⁺(near O) Au⁻(far)
  admixture.
- DFT puts total charges of ±0.11–0.13 on the upright Au. The SPEC 23 measurement says the Gaussian free charges
  favour Au⁺ next to O²⁻ by 3.8 eV too much.
- **Hypothesis (to be tested by steps 15 and 16, not a result):** seed 42's T5 of 103 % rests in part on that
  spurious stabilisation.
  - With the overlap physics in, the free part should fall.
  - T5 then holds only if the short-range model learns the Au–O bonding that DFT attributes to the upright
    geometry.
  - Q1 (polarisation toward DFT) and Q2 (T5) are therefore expected to move against each other unless the short
    range picks it up. That is why both are criteria.

### Interim look at step 13 and a collapse of the potential (2026-10-18 17:20–18:10; reported, not criteria)

- **Interim T5** (best checkpoints at epochs 16 and 19, `probe_t5_relaxation.py`):
  - **Seed 42:** T5 = −37.9 − 474.9 = −512.8 meV (55 %). The upright Au free charges are ±0.20 against step
    12's ±0.43, and the total charges +0.157/−0.191 are much nearer DFT's +0.110/−0.131. The amplitude selection
    has removed much of the over-polarisation, and the free-sector part of T5 with it, as the decomposition above
    predicts.
  - **Seed 43:** the undoped wetting relaxation failed: the closure exceeded its cap.
- **Diagnosis of that failure** (`probe_closure_size.py`; scratch scripts):
  - Forces equal −∇E to ~1e-6 eV/Å against finite differences, so this is the model's surface.
  - **Stage 1:** the flat Au₂ goes ionic (free ±0.03 → ±1.00, free energy −1.27 eV in 20 steps) as one Au moves
    toward O. This is the Au⁺-next-to-O²⁻ over-stabilisation, which SPEC 23/25/26 (steps 15/16) correct.
  - **Stage 2:** the Au is driven into a surface Mg. The learned coupling T(Mg, Au) grows 0.35 → 9200 eV. With
    the Harrison bound (SPEC 27) applied post hoc, the Au still runs onto the Mg nucleus (0.003 Å): no term in the
    potential repels nuclei at short range, and the short-range network changes by only −35 eV.
- **Contingency, pre-registered now:**
  - If a T5 relaxation of steps 13–16 collapses (a heavy-atom pair below 1.5 Å), that seed's T5 is reported as
    unscoreable, not dropped.
  - The next step is then a ZBL-type core repulsion (ESTABLISHED: Ziegler, Biersack & Littmark 1985), switched off
    below the data's shortest bond so that it is the identity on the data. It is not built before that, since it
    moves no benchmark number otherwise.
- **SPEC 27 (coupling bound) is built but off.** It is the identity on all 404 AuMgO structures (max 0.650 B
  against a 0.75 knee). It does not prevent the collapse, so it is not added to steps 14–16.
- **The collapse is transient (18:45).** At epoch 23, seed 43's undoped wetting relaxation converges in ~5 steps
  with the dimer covalent (free charges ±0.013). The contingency is not triggered.
- **Second interim T5** (newest checkpoints, not criteria):

| | T5 | upright Au free charges | upright total charges (DFT +0.110/−0.131) |
|---|---|---|---|
| seed 42, epoch 24 | −18.7 − 496.5 = **−515.2 meV (55 %)** | ±0.155 | +0.171/−0.209 |
| seed 43, epoch 23 | −7.9 − 1521.2 = **−1529.1 meV (164 %)** | ±0.968 | +0.25/−0.34 |

- **Reading (interim):**
  - The seeds land on opposite sides of DFT, with the same lever: the energy of the upright dimer's ionic Au⁺Au⁻
    configuration. Seed 43 is fully ionic and seed 42 nearly covalent.
  - Amplitude selection alone (SPEC 21) has so far not removed the seed dependence. What fixes the ionic
    configuration energy is what steps 14 (bond channel κ_b, SPEC 22) and 15 (first-principles overlap, SPEC
    23/25) test.
  - A3 is decided at epoch 40.

### State-dependent polarisabilities, measured (2026-10-18 19:00; `tools/electrostatics/ion_polarizabilities.py`, `probe_state_polarizability.py`)

- **First-principles α(Z, q)** (the library's X2C-PBE0/ANO-RCC; finite field with Richardson extrapolation),
  in bohr³:
  - Au⁻ **123.7**, Au⁰ **36.50** (exp 36 ± 3; the model's trained Au is 5.43 Å³ = 36.6 bohr³), Au⁺ **13.45**.
  - Checks: Na⁰ 166.8 (exp 162.7), Mg⁰ 75.1 (71.2), Mg⁺ 35.3 (35.0), Mg²⁺ 0.50 (0.49), Cl⁰ 14.9 (14.7).
  - O in the Watson sphere: O²⁻ 16.4, O⁻ 7.3. Diffuse anions are not converged in the field (Na⁻, Al⁻, Ag⁻) and
    are left out.
- The model gives every Au state the neutral atom's α. The state's own α relative to it is ρ = 3.39 (Au⁻) and 0.37
  (Au⁺).
- **Exact correction per configuration** (Woodbury, checked to 3.5e-14 against brute force):
  δE(n) = ½ y_S(n)ᵀ Δ_S (I + C_SS Δ_S)⁻¹ y_S(n), with Δ_j = P_jj(1/ρ_j − 1).

| frozen step-12 weights, fixed T5 geometries | seed 42 | seed 43 |
|---|---|---|
| δE doped wet / doped upright / undoped wet / undoped upright (meV) | −321 / −405 / −35 / −116 | −265 / −128 / −69 / −230 |
| wetting doped: model → with α(q) | −50.4 → +33.0 | −47.6 → −184.2 |
| wetting undoped: model → with α(q) | +911.5 → +993.4 | +611.9 → +772.8 |
| **T5** (DFT −931.6) | −961.8 → **−960.4** | −659.5 → **−957.0** |

- **Reading:**
  - Configuration energies move by 0.1–0.9 eV, mostly the Au⁻ states. That is several times SPEC 26's
    induction correction.
  - T5 moves by +1 meV for seed 42 and −298 meV for seed 43. The two seeds' class energies shift differently.
- **Caveats (hypotheses until a run):**
  - The weights were trained without the term, and their learned shifts compensate part of whatever it corrects.
  - The free-ion ratio is an upper bound for an anion confined by its neighbours (Fowler–Madden).
  - The implied local field at Au⁻ (~1.4 V/Å) is plausible but not verified.
  - Only a retraining measures the benchmark effect. Built as trace_redox SPEC 28; a step after 16 is to be
    pre-registered.

### Correction (2026-10-18 19:35): the Au₂/MgO "over-polarisation" against Hirshfeld is a partition artifact

**Falsifying check, run before relying on the claim** (`probe_charge_partition.py`, `probe_au_charges.py`; step 12
seed 42).
- The model's total charges against DFT Hirshfeld for every element, over 40 test structures:

| element | Hirshfeld mean | model mean | ratio of means | ratio of sd (fluctuations) |
|---|---|---|---|---|
| O | −0.355 | −1.758 | 4.96 | 2.48 |
| Mg | +0.355 | +1.754 | 4.94 | 2.26 |
| Al | +0.404 | +2.208 | 5.47 | — |
| Au | −0.048 | −0.281 | — | 2.31 |

- Over all 400 test structures, the undoped upright Au₂ has p = (q_near − q_far)/2 = +0.319 (seed 42) and +0.347
  (seed 43), against Hirshfeld +0.120, with r = +0.87 and +0.90.
  - The ratio, 2.7–2.9, is the partition scale the substrate ions show (fluctuation ratio 2.3–2.5).
  - The doped Au₂ charge (model −0.91 e, Hirshfeld −0.20) has the ratio of the ionic means (~4.5–5).

**Reading.**
- Hirshfeld compresses ionic charges by a factor of 2–5 relative to the model's total charges (a known property of
  stockholder partitions). Comparing p against Hirshfeld's 0.120 therefore measures the partition, not the physics.
- The trend is right: r ≈ 0.9 on every class with spread.
- **Withdrawn:** "the upright Au₂ is over-polarised (+0.313/−0.326 against DFT +0.110/−0.131)" (step-12 results,
  step-14/15 motives, the interim notes). Not established.
- **Unaffected** (no Hirshfeld involved):
  - the isolated Ag₂ polarisability (α∥ 112 against GGA ~22 Å³);
  - the Au₂ Δ_ion seed spread (2.3 eV);
  - the first-principles penetration error (3.8 eV);
  - the window, NaN, coupling and α(q) findings.
- **Criteria amended before their runs:**
  - P3 (step 14), the magnitude part of Q1 (15), I1 (16) and S3 (17) become reported quantities.
  - The sign test (near > far) stays.
  - T5, E/F and the B criteria decide.
  - Reported with them: the calibrated p_cal = p_model/c, with c the model/Hirshfeld fluctuation ratio of the
    substrate ions on the same structures (2.37 for step 12 seed 42). This is an indication, not a measurement:
    the compression need not be the same for Au.
- The data hold no partition-independent polarisation observable, such as the DFT dipole or field. That is why no
  magnitude criterion is set.

### Pre-registered: step 16, induction by the carriers' true fields (written 2026-10-18 17:30, before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_bondlaw_r5_s{42,43}` = step 15's configuration with
one change, `redox.carrier_induction: true` (trace_redox SPEC 26). It launches when step 15 finishes
(`watch_step16.sh`). Comparator: step 15, same seeds.

**Motive (measured).**
- By Gauss, the field of a carrier at a moment is K Q λ_c/R², with λ_c the enclosed fraction of its true density
  smeared by the moment's Gaussian. At an O moment 2.2 Å from Au: λ_c = 0.475 (Au⁻), 0.668 (Au⁺), against the
  model's Gaussian 0.827.
- So the model overstates the induction driven by diffuse carriers. SPEC 26 makes the correction exact in the
  configuration energies (Proposition 5 of the theory note).
- Frozen step-12 seed-42 weights, doped cell (`probe_induction_overlap.py`):
  - one-body corrections: Au⁻ +0.19 eV, Au⁺ +0.008, Mg⁺ +0.16, O⁻ +0.08 (medians);
  - populated pair terms ≤ 0.27 eV;
  - total +0.41 eV.
  These are untrained weights: direction and size only.

**Benchmark moved:** T5 (4G Table II); the AuMgO polarisation against DFT Hirshfeld. The term removes a bias of
~0.18 eV in favour of Au⁻ next to O.

| id | criterion (each seed) |
|---|---|
| I1 | undoped non-wetting Au₂: near > far; ~~\|(q_near − q_far)/2 − 0.120\| no worse than step 15's by > 0.02 e~~ **amended 19:35: the magnitude is reported** |
| **I2** | T5 within ±15 % of DFT (−792 to −1071 meV) for **both** seeds |
| I3 | B1 (doped Au₂ holds one electron); B3 (undoped \|class offset\| ≤ 100 meV); A1 (window independence) |
| I4 | test E / F RMSE no worse than step 12 seed 42 by > 10 % (0.847 / 0.703 meV/atom; 0.1052 / 0.0764 eV/Å) |
| I5 | ε∞(MgO) 2.5–3.5 (`probe_eps_bound.py`); Au atom α within 10 % |

**Reported:**
- |T5(42) − T5(43)|, against step 15's (the seed dependence);
- the learned Au⁺/Au⁻ shifts against step 15's: the term should free them from compensating ~0.2 eV;
- cost per epoch, against step 15's (frozen weights: 2.56 s against 1.56 s per energy and forces).

**Predictions (hypotheses).**
- The correction is ~20 % of T5's scale, so I2 moves T5 by at most ~0.2 eV.
- The shifts change by about the E1 size.
- I1 is open.

**Reading.**
- I2 passes where step 15 failed: the induction at overlap was the missing piece.
- No change within the seed spread: it is a small, correct term and stays on for exactness, not for this benchmark.
- I4 fails: the term costs accuracy, which needs a cause found before it stays.

### The Au₂ polarisability: a partition-free observable of the free sector (2026-10-18 20:30; reported)

- **First principles** (`tools/electrostatics/dimer_polarizability.py`; PBE0/def2-TZVPPD with the def2 ECP;
  finite field with Richardson extrapolation), isolated Au₂. Values are per 2·α(atom) (def2 atom 5.080 Å³; X2C
  5.409; exp 36 ± 3 bohr³ = 5.3 ± 0.4 Å³):

| d (Å) | 2.30 | 2.47 | 2.60 | 2.80 | 3.10 |
|---|---|---|---|---|---|
| α∥, PBE0 | 1.357 | 1.515 | 1.654 | 1.894 | 2.302 |
| α⊥, PBE0 | 0.786 | 0.803 | 0.816 | 0.837 | 0.870 |
| α∥, ωB97X | — | 1.498 | — | 1.911 | — |
| α⊥, ωB97X | — | 0.827 | — | 0.870 | — |

- The functional uncertainty is ~1 % on α∥ and ~3–4 % on α⊥.
- **The models** (`probe_dimer_alpha.py`: finite field of the total energy, the whole ensemble included), α∥ per
  2·α(atom):

| d (Å) | 2.30 | 2.47 | 2.60 | 2.80 | 3.10 |
|---|---|---|---|---|---|
| step 12, seed 42 | 1.231 | 1.339 | 1.412 | 1.532 | 1.763 |
| step 12, seed 43 | 0.996 | 1.274 | 1.163 | 1.225 | 1.156 |
| step 13, seed 42 (epoch 24) | 1.226 | 1.317 | 1.386 | 1.496 | 1.629 |
| step 13, seed 43 (epoch 23) | 1.803 | 1.788 | 1.831 | 2.025 | 4.284 |

- α⊥ (the bound sector) is within 2–5 % of the reference in every model. α∥, which carries the ionic–covalent
  charge transfer, is where the seeds fail, each differently:
  - too low with the right shape (step 12, seed 42);
  - non-monotone (step 12, seed 43);
  - too high, diverging at 3.1 Å (step 13, seed 43).
- This is the identification the AuMgO energies cannot give: the dimer's response pins Δ_ion(R) and τ(R),
  which the energies leave free to trade against the short-range model.

**Au and Au₂ electron affinity and ionisation energy** (vertical; `probe_dimer_ip_ea.py`; reported, not a prior).
- DFT (def2-TZVPPD) underestimates the atomic EA: PBE0 1.873, ωB97X 1.864 eV against the exact 2.309.
- Dimer references are therefore the DFT dimer-minus-atom differences added to the exact atomic values. Au₂
  EA at 2.30 / 2.47 / 2.60 / 2.80 Å: PBE0 1.85 / 2.03 / 2.17 / 2.37 eV, ωB97X 1.63 / 1.83 / 1.98 / 2.22 eV
  (experiment, adiabatic: 1.94). IP (PBE0): 9.43 / 9.22 / 9.09 / 8.91 eV. The spread of ~0.2 eV is the
  reference uncertainty.

| model | Au atom EA / IP (exact 2.309 / 9.226) | Au₂ EA at 2.30 / 2.47 / 2.60 / 2.80 Å | Au₂ IP at the same |
|---|---|---|---|
| step 12, seed 42 | 1.128 / 10.472 | 1.81 / 1.79 / 1.85 / 1.98 | 9.34 / 9.21 / 9.17 / 9.14 |
| step 12, seed 43 | 1.258 / 10.424 | 2.42 / 2.00 / 2.94 / 3.06 | 9.23 / 9.66 / 8.74 / 8.62 |
| step 13, seed 42 (epoch 24) | 0.531 / 11.010 | 1.50 / 1.48 / 1.52 / 1.63 | 9.60 / 9.47 / 9.43 / 9.41 |
| step 13, seed 43 (epoch 23) | 1.936 / 9.613 | 1.75 / 1.74 / 1.80 / 1.91 | 9.53 / 9.46 / 9.39 / 9.32 |

- **Isolated atom:** errors of 0.4–1.8 eV. These are the trained shifts without a neighbour, which SPEC 24
  (`shift_gate`, on from step 15) removes exactly.
- **Dimer:** within 0.2–0.5 eV in most models (step 12 seed 43 is erratic in R). That is comparable to the
  reference's own uncertainty, so no energy prior. α∥ is the sharper constraint.

**Ag₂, for the Ag₃ revalidation** (PBE0/def2-TZVPPD; the Ag atom 7.343 Å³, against the prior's 7.553):
α∥ / α⊥ = 17.80 / 9.87 (2.40 Å), 19.20 / 10.10 (2.53 Å), 21.28 / 10.43 (2.70 Å), 24.05 / 10.87 (2.90 Å) Å³. The
step-9 Ag₃ model's α∥ = 111.9 Å³ at 2.53 Å is 5.8× too large. A field prior on this curve is the analogue of step
13F for the Ag₃± benchmark, to be pre-registered with the universality rerun.

**Reference convergence (2026-10-18 21:30; per the rule-out-artifacts policy).**
- def2-TZVPPD alone is not converged for anions. For Au⁻, PBE0's HOMO is +0.259 eV (unbound, held by the basis)
  and α = 77.0 bohr³.
- With two more even-tempered diffuse s, p, d shells (ratio 3), PBE0 gives 122.6 and ωB97X (HOMO −1.89 eV) 121.9
  bohr³. The library's X2C/ANO-RCC + 2 diffuse value, 123.7, was right.
- A "2× confinement overshoot" of SPEC 28, briefly inferred from the def2 anion dimer, was this artifact and is
  withdrawn.
- The neutral Au₂ references are converged: the augmented-basis ratios differ from def2-TZVPPD's by ≤ 0.6 % (α∥)
  and ≤ 1 % (α⊥). Step 13F's targets stand.
- Au₂⁺ is converged: α∥ / α⊥ 10.91 / 5.28, 11.33 / 5.35, 12.67 / 5.45 Å³ at 2.47 / 2.60 / 2.80 Å.
- Au₂⁻ with PBE0 in the augmented basis: α⊥ 16.77 / 16.37 / 16.05 Å³ (field-step change ≤ 8 %). α∥ is not
  converged in the field (step change of 663 bohr³ at 2.47 Å); ωB97X with weaker fields is running.

**The physics stack on these observables, frozen step-12 seed-42 weights** (Å³; references augmented-basis PBE0
scaled to the model's atom, 2.47 Å):

| | Au₂⁺ α∥ / α⊥ (ref 11.55 / 5.59) | Au₂ α∥ / α⊥ (ref 16.54 / 8.80) | Au₂⁻ α∥ / α⊥ (ref — / 17.76) |
|---|---|---|---|
| model as trained | 15.42 / 9.31 | 14.55 / 9.31 | 13.71 / 9.31 |
| + 23/25/26 | 15.73 / 9.31 | 15.43 / 9.31 | 14.98 / 9.31 |
| + 28 alone | 23.19 / 11.89 | 16.97 / 14.09 | 30.80 / 19.58 |
| + 22/23/25/26/28 | 14.26 / **6.90** | 11.16 / **9.31** | 32.97 / **19.58** |

- With the full stack, α⊥ follows the charge state as DFT does: model/reference 1.23, 1.06, 1.10. As trained,
  α⊥ is the same 9.31 for all three charges.
- SPEC 28 needs SPEC 22: without the bond partition the ensemble mixes in Au⁻-containing configurations, and
  SPEC 28 inflates even the cation (α⊥ 11.89).
- α∥ of the neutral dimer falls to 11.16 (−33 %) once the bond channel is removed. That is the charge-transfer
  response the energies leave free, and step 13F trains it.

**Second reboot (2026-10-18 22:03).** All trainings and watchers stopped. Resumed at 22:08 with
`resume_run.py`, a new tool: it archives the log, copies the checkpoint, writes `config_resume_reboot2.yaml`, and
launches with the loader advanced.
- step 13 seed 42 from epoch 34 and seed 43 from epoch 32;
- step 13F seeds 42 / 43 from epoch 3.
- In every run the checkpoint was the last completed epoch, so only the epoch in progress was lost.
- `atomic_energies_from_model` is dropped in a resume: train.py refuses it there, because the checkpoint keeps
  its references.
- Watchers: `watch_step13_14e.sh` (launches step 14), `watch_step13f_score.sh`, and `watch_step15/16/17.sh`
  restarted.
- All four runs now carry the recompute guard and the NaN probe, which also instruments the lattice kernels.
- Before the reboot the guard had recovered two transient faults exactly (step 13 seed 43, epoch 27; step 13F
  seed 43, epoch 3).
  - 13F's probe bracket: every watched gradient finite down to the lattice blocks' outputs, and non-finite at
    σ_f's lattice input.
  - So the NaN is made inside `free_charge_lattice_blocks`' σ-backward, from finite gradients, while the
    recomputed forward loss differed in the last digit (non-deterministic forward).
  - That path holds only elementwise special functions written against NaN traps, a gather (`index_select`)
    and a scatter (`index_add`). No BLAS product is on it, so the BLAS-fault hypothesis does not explain this
    occurrence.
  - The mechanism is not identified; the recompute handles it exactly. At the next occurrence the probe also
    brackets `coulomb_kernel`, `real_space_damping` and `_combined` (inputs and outputs), wrapped from outside
    `trace_electrostatics`.
- **Third recovered transient (step 13 seed 42, epoch 36, after the reboot; finer probe).**
  - Every lattice watch point was finite, kernel inputs and outputs included. The first non-finite watched
    gradient was the configuration energies `diabatic.E`, at the start of the weight-gradient pass.
  - So this NaN was made between the loss and E (the free energy's Daleckii–Krein backward, or the Hamiltonian
    assembly), not in the lattice.
  - Checked and ruled out:
    - e = 0 or t = 0 in the amplitude switch and coupling amplitude (sanitised inputs);
    - an overflow of d²(ln e)/de²: it needs e < 1e-154, while every selected configuration has r ≥ 4e-6 and every
      coupling is ≥ ~1e-47 or exactly 0.
  - Two occurrences, two different entry points, and a non-bitwise-reproducible forward (the recomputed loss
    differs in the last digit). No single singular operation explains them: this fits transient computational
    faults. The recompute keeps the trajectory exact.

**Au₂⁻, converged, and the resonance diagnosis (2026-10-18 22:50; `probe_anion_resonance.py`).**
- **Reference:** ωB97X in the augmented basis, weak fields (1e-3 / 5e-4 au; PBE0 is not converged in the field
  here). At 2.47 Å: α∥ 43.03 Å³, α⊥ 16.92 Å³ (field-step changes 2.1 and 0.18 bohr³); α∥(2.60 Å) 40.14 Å³.
  The ωB97X atom in the same basis is 5.289 Å³; scaled to the model's atom, 44.2 / 17.4 Å³.
  - This replaces the def2-TZVPPD anion values (30.6 / 13.0), which were basis-incomplete.
- **The model, full stack** (SPEC 22, 23, 25, 26, 28; frozen step-12 seed-42 weights; 2.47 Å):
  - two degenerate configurations, Au⁻Au⁰ and Au⁰Au⁻, coupled by τ = 1.21 eV;
  - the change of the total dipole between them is 1.27 e·Å against the bare 2.47: the bound moments induced
    by the moved charge screen 48 % of it, consistent with a Gauss field factor of ~0.67 at the partner
    (SPEC 26);
  - two-state α_CT = K_E μ_eff²/(4τ) = 4.8 Å³ (bare μ: 18.1);
  - finite field: α∥ − α⊥ = 13.4 Å³, against DFT's 26.1 Å³. α∥ 33.0 against 44.2 (−25 %), α⊥ 19.6 against
    17.4 (+13 %).
- **Reading (hypothesis until trained):**
  - Given the physical screening, DFT's charge-transfer response requires a resonance coupling of order 0.3 eV.
    Both seeds learned τ(Au, Au) ≈ 1.2 eV, which the energies alone do not test.
  - The neutral dimer's α∥ is also too low under the full stack (11.2 against 16.5).
  - One 6s–6s τ serves both the ionic–covalent and the anion resonance (spinless JW, §17.2), so a field prior
    over Au₂⁰, Au₂⁺ and Au₂⁻ over-determines it: a consistency test of the configuration model, not only a fit.

### Pre-registered: step 13F, the dimer's field response as a prior (written 2026-10-18 20:35, before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_fprior_bondlaw_r5_s{42,43}` = step 13's configuration with one
change: `electrostatics.field_polarizability_prior` on the Au₂ α tensor at 2.30, 2.47, 2.60 and 2.80 Å.
- The prior is new in train.py: the whole model's −∂²E/∂E² by finite fields, h = 0.01 V/Å, 5 tests. Weight 0.3,
  as the Au atom prior.
- Targets are the PBE0 ratios above times the atom prior's 5.4354 Å³ (`data/Au2_alpha_field_prior.xyz`). 3.10 Å
  is left out: it is the point most exposed to the functional's delocalisation error.
- Comparator: step 13, same seeds.
- Launch: `watch_step13f.sh`, 45 min after step 13 and the carbon chain end, in parallel with step 14
  (4 trainings on the Mac).
- The prior evaluated on the step 13 seed-42 epoch-24 checkpoint reproduces the probe exactly (14.301 Å³ at
  2.47 Å) in 1.24 s per evaluation.

**Benchmark moved:** T5 for both seeds (the seed dependence), via the free-sector parameters that the energies do
not identify.

**Note after launch (20:36; reported, criteria unchanged).**
- The first 15 optimiser steps have gradient norms of 1e4–4e5 before clipping (step 13: 10–15), so the prior's
  direction dominates the clipped steps at initialisation.
- Checked and ruled out: the training-time kT schedule (0.5 → 0.1 eV). On a trained step-13 checkpoint the
  prior's α and gradient are identical at kT 0.1 / 0.3 / 0.5 eV (|∇(0.3 L)| = 8.07), comparable with the data
  gradient.
- So the large norms come from the untrained model's dimer response. Epoch 1's E/F against step 13's epoch 1 is
  reported.
- **Epoch 1 (seed 42), 13F against step 13:** validation loss 0.525 / 0.108, F 0.722 / 0.318 eV/Å, E 19.08 /
  26.78 meV/atom, ε∞ 6.41 / 3.52; field-prior RMSE 9.67 Å³.
  - The gradient norm decays through the epoch: 5.7e4 → 656 (seed 42) and 4.4e4 → 591 (seed 43) by steps 36–40.
    That is a transient of the initial dimer response.
  - Epoch time 1771 s against 1104 s, partly from four concurrent trainings. F4 decides whether the early
    distortion is recovered.

| id | criterion (each seed unless stated) |
|---|---|
| **F1** | T5 within ±15 % of DFT (−792 to −1071 meV) for **both** seeds |
| F2 | \|T5(42) − T5(43)\| below step 13's |
| F3 | Au₂ α∥ within 5 % of the target at all four distances, and α∥(3.10 Å) monotone above α∥(2.80 Å) (not fitted) |
| F4 | test E / F RMSE no worse than step 13 (same seed) by > 10 % |
| F5 | B1 (doped Au₂ holds one electron), B3 (undoped \|class offset\| ≤ 100 meV), A1 (window independence) |

**Reported:** Δ_ion and τ of the isolated Au₂ (`probe_dimer_ct.py`) against step 13's; the decomposition of T5;
the time per epoch.

**Predictions (hypotheses).**
- F3 passes: the response is directly fitted.
- F2: the free-sector freedom the seeds used differently is now pinned in the gas phase; the adsorbed dimer's
  environment terms remain.

**Reading.**
- F1 and F2 pass: the seed dependence was the unidentified gas-phase charge-transfer physics.
- F3 passes and F2 fails: the environment terms (shifts, penetration) carry it, which is steps 14–17's ground.
- F4 fails: the prior conflicts with the energies. That would be a signal of missing physics in the free sector,
  not a reason to lower the weight.

### The α scorecard over three charge states (2026-10-18 23:40; `probe_alpha_scorecard.py`)

**Targets** (`data/Au2_charges_alpha_field_prior.xyz`, 9 tensors; Å³, scaled to the model's atom):

| | 2.30 Å | 2.47 Å | 2.60 Å | 2.80 Å |
|---|---|---|---|---|
| Au₂ (PBE0) | 14.84 / 8.61 | 16.54 / 8.80 | 18.04 / 8.95 | 20.63 / 9.18 |
| Au₂⁺ (PBE0) | — | 11.55 / 5.59 | 12.00 / 5.66 | 13.42 / 5.77 |
| Au₂⁻ (ωB97X) | — | 44.22 / 17.38 | 41.25 / 17.18 | — |

(α∥ / α⊥; def2-TZVPPD plus two diffuse s, p, d shells.)

| frozen weights | as trained | + SPEC 22–28 | + SPEC 22–29 |
|---|---|---|---|
| step 13 seed 42 (epoch 34) | RMSE 8.59 | **4.19** | 4.20 |
| step 13 seed 43 (epoch 32) | 8.68 | **506** (Au₂⁺ α∥ 1680) | 506 |

- For seed 42 the first-principles stack halves the error without retraining.
- For seed 43 it exposes a learned coupling that is ~0 in the cation-dimer environment. The two degenerate
  Au⁺Au⁰ configurations then respond thermally instead of coherently. Spin pairing cannot help, since it scales a
  τ that is ~0.
- So τ is unidentified wherever the data do not reach. Response data over every charge state is the principled
  identification, which is step 13FC.

**Third reboot (2026-10-19 ~10:09; shutdown).** Resumed at 10:15 with `resume_run.py`, tag `reboot3`, each from its
last completed epoch:
- step 14 seeds 42 / 43 from epoch 10;
- step 13S seeds 42 / 43 from epoch 4.

Watchers rebuilt:
- `watch_step14_score.sh` scores step 14, including the α scorecard;
- `watch_step15/16/17.sh`;
- `watch_step13s_score.sh`;
- `watch_step13fc.sh`, re-armed now that SPEC 30 has removed the ε∞-prior conflict.

Before the relaunch, the SPEC 30 ensemble record was made opt-in (`record_ensemble`, set only by the response priors
around their own forward; 1 new test). Ordinary training energies and forces therefore run exactly the code path the
interrupted runs started with.

### Pre-registered: step 13FC, the field prior over Au₂⁰, Au₂⁺ and Au₂⁻ (written 2026-10-18 23:45, before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_fpriorq_bondlaw_r5_s{42,43}` = step 13F's configuration with one
change: the prior file holds the nine tensors above instead of the four neutral ones. It launches when step 13S
has been trained and scored (`watch_step13fc.sh`). Comparator: step 13F, same seeds.

| id | criterion (each seed unless stated) |
|---|---|
| **C1** | T5 within ±15 % of DFT for **both** seeds |
| C2 | \|T5(42) − T5(43)\| below step 13F's |
| C3 | scorecard RMSE over the nine tensors ≤ 15 % of the mean target (2.6 Å³), with no tensor component off by > 30 % |
| C4 | test E / F RMSE no worse than step 13 (same seed) by > 10 % |
| C5 | B1, B3, A1 |

**Reported:** τ(Au, Au) of the isolated dimer in each charge state, against the anion's two-state estimate
(~0.3 eV at μ_eff 1.27 e·Å).

**Prediction (from Proposition 7, written before the run).**
- The coupling head reads the short-range invariants, which do not depend on the total charge. So at one geometry
  the three charge states share one τ.
- In the spinless model the neutral's ionic–covalent mixing then needs √2 times the τ the ion resonances need.
  The learned shifts can absorb part of this, but not the full α tensors at four distances.
- Prediction: C3 is not met for all three charge states at once. If it is met, Proposition 7's mismatch is
  absorbed elsewhere (shifts, screening), and that finding would itself be reported.

**Reading.**
- C3 fails while the fit converges: one spinless τ cannot serve all three charge states. That is the case for
  SPEC 29 to be combined (spin pairing makes them consistent at two sites) and for the next physics step.

**Disarmed before its launch (2026-10-19 13:10): 13FC is confounded and cannot test its own hypothesis.**
- Perpendicular to a homonuclear dimer's axis there is no charge-transfer response, so α⊥ is the bound sector's
  alone. Without SPEC 28 (and 22) that is the same for every charge state.
- Measured on step 13's weights (`probe_alpha_scorecard.py`, both seeds): α⊥ = 8.8–9.3 Å³ for Au₂⁰, Au₂⁺ and
  Au₂⁻, against targets 8.6–9.2, 5.6–5.8 and 17.2–17.4. The ions are off by +60 % and −47 % whatever τ does, so
  C3 (no component off by > 30 %) fails by construction.
  - That would have been read as "one spinless τ cannot serve three charge states", an unfair inference.
  - The anion's error is larger, so the net α⊥ gradient on the shared Au α_b would pull the neutral α⊥ (right
    today) away from its target.
- With SPEC 22–28 switched on (frozen, seed 42), α⊥ follows the charge state (7.5–7.8 / 10.4–11.1 / 19.4–19.5) and
  the RMSE falls 8.59 → 4.18 Å³. Seed 43's weights give a cation α∥ of 1407–1750 Å³ under the stack.
- The charged-dimer prior belongs on the stack that can represent α⊥(q): to be pre-registered as a one-change step
  on step 17's configuration, after step 17.
- Its slot (after 13S) now runs step 15, unchanged (`watch_step15_early.sh`), which shortens the 15 → 16 → 17 chain
  by one step-14 duration.

**Why the charged dimers matter: the free sector has its own (gap, τ) degeneracy** (measured 2026-10-19, isolated
Au₂, step 13 final weights).
- Neutral Au₂ at 2.3–3.0 Å:
  - seed 42: ionic gap E(Au⁺Au⁻) − E(Au⁰Au⁰) = +2.88 to +3.29 eV, τ 1.57–2.00 eV;
  - seed 43: gap −0.77 to −1.00 eV, an ionic ground configuration; τ 0.52–0.66 eV; then +2.90 eV at 3.4 Å.
- Au₂⁺: Au²⁺Au⁻ lies +4.57 eV (seed 42) and +0.39 eV (seed 43) above Au⁺Au⁰.
- The two-site bond energy relative to the covalent configuration is nearly the same,
  ½(gap − √(gap² + 8τ²)) = −1.71 and −1.54 eV, while the ionic weights are 0.27 and 0.74. Energies fix one
  combination of (gap, τ); the ionic character is free along it. That is the upright undoped dimer's
  p(Au⁺Au⁻) of 0.22 against 0.925, and so T5's seed split.
- The partition-free observables that break it are exact at two sites:
  - the cation's hole resonance, α_CT = μ_eff² tanh(τ/kT)/(4τ), fixes τ alone;
  - given τ, the neutral α∥ fixes the gap.
  This needs α_b(q) right first (above).

**T5 against the Au shifts, a representability scan** (`probe_t5_shift_scan.py`; frozen step-13 weights).
- With offsets δ₊, δ₋ on Au(+1), Au(−1), the doped wetting energy is exactly invariant (−176.4 meV, seed 42):
  every doped configuration holds one Au⁻, so a constant cancels.
- T5 depends only on δ₊ + δ₋, the ionic gap: (−1.5, +1.5) returns the unshifted −555.2 exactly.
- T5 moves only through the upright neutral dimer's ionic weight, −555 → −796 meV as p_ion 0.22 → 0.45. Reaching
  DFT this way needs a half-ionic upright Au₂: the right number for the wrong reason, not pursued.
- Seed 43: −1734 to −483 meV over δ₊ + δ₋ ∈ [−3, +3] eV. It is −871.7 meV at every point with δ₊ + δ₋ = +1.5
  (four points, identical), with p_ion 0.70. Same conclusion.

**The queued steps on frozen step-13 weights** (`probe_t5_stack.py`: each step's own `redox` block applied to the
frozen checkpoint; reference geometries; direction and size only, not a prediction of the trained steps).

| settings | T5 seed 42 | T5 seed 43 | spread | wet doped 42 / 43 | wet undoped 42 / 43 | p_ion 42 / 43 |
|---|---|---|---|---|---|---|
| step 13 as trained | −556.6 | −1372.7 | 816 | −46 / −109 | +510 / +1264 | 0.217 / 0.925 |
| step 13S (+ 29) | −636.5 | −1180.7 | 544 | −46 / −109 | +590 / +1072 | 0.238 / 0.864 |
| step 14 (+ 22) | −277.5 | −374.0 | 97 | −2 / −8 | +275 / +366 | 0.186 / 0.545 |
| step 15 (+ 23/24/25) | −131.6 | −40.1 | 92 | +150 / +136 | +282 / +176 | 0.165 / 0.311 |
| step 16 (+ 26) | −191.6 | −36.2 | 155 | +67 / +61 | +258 / +97 | 0.154 / 0.226 |
| step 17 (+ 28 Au, O) | −400.7 | −349.9 | **51** | −35 / −31 | +366 / +319 | 0.189 / 0.452 |

- ~~The stack removes the seed dependence on weights never trained with it: 816 → 51 meV, and seed 43's ionic upright
  dimer goes from p_ion 0.925 to 0.45.~~
- **Correction (2026-10-19 ~17:30): rows 15–17 of this table are an artifact, and the 816 → 51 meV claim is withdrawn.**
  - The module tested `self.exclusion is not None`, the submodule built at construction, not the configuration.
    So switching `exclusion: false` on these exclusion-trained weights left SPEC 18's fitted exclusion on beside
    Gordon–Kim: Pauli was counted twice.
  - Fixed: every optional term now follows the configuration (trace_redox `module.py`; new test
    `test_a_configuration_switched_off_after_construction_is_off`, which failed with 0.139 eV before the fix).
  - Re-measured, with step 14's row unchanged since its configuration keeps the exclusion:

  | settings | T5 seed 42 | T5 seed 43 | spread | p_ion 42 / 43 |
  |---|---|---|---|---|
  | step 14 (+ 22) | −277.5 | −374.0 | 97 | 0.186 / 0.545 |
  | step 15 (+ 23/24/25) | +398.7 | +158.1 | 241 | 0.121 / 0.096 |
  | step 16 (+ 26) | +292.0 | +0.3 | 292 | 0.110 / 0.076 |
  | step 17 (+ 28 Au, O) | +107.8 | −169.9 | 278 | 0.128 / 0.130 |

  - What stands: both seeds' upright dimers become covalent and alike (p_ion 0.13 / 0.13), and the isolated-dimer
    gaps below are unchanged, since an isolated dimer has no closed shell for the exclusion to act on.
  - What does not: frozen weights cannot say what replacing the fitted exclusion by Gordon–Kim does to T5. The
    shifts were trained with the exclusion, which carried −413 meV of T5 in seed 42 (`probe_t5_terms.py`), and
    dropping it unretrained flips T5's sign. Only the trained steps 15–17 measure it.
- That is the gas-phase finding in the slab: with the overlap terms, the isolated Au₂ ionic gap from the explicit
  physics (shifts zeroed, `probe_explicit_gap.py`) is +0.60 / +0.85 / +1.20 eV at 2.3 / 2.5 / 2.8 Å for seed 42,
  and +0.65 / +0.90 / +1.22 for seed 43: positive and the same to 0.05 eV.
  - Without the overlap terms it is −0.73 to −0.44 (seed 42) and −2.10 to −1.94 eV (seed 43). The classical
    ion-pair Coulomb and point polarisation at overlap make Au⁺Au⁻ the more stable configuration, and each seed's
    learned shifts repair that differently.
  - At 3.4 Å all four zero-shift values (2.03–2.07 eV) agree with the exact asymptote IE − EA − K/R −
    ½(α_A + α_B)K/R⁴ ≈ 2.10 eV. The asymptote itself is now a machine-precision test (trace_redox SPEC 24).
- The magnitude is not there on frozen weights: T5 43 % / 38 % of DFT. The doped wetting is near DFT (−35 / −31
  against −2.7) and the undoped one low (+366 / +319 against +929). Retraining refits the shifts; whether it
  restores the magnitude through physics or through ionic admixture is what steps 14–17 measure (p_ion is reported).
- ~~SPEC 28 carries the largest late increment of |T5|: +209 / +314 meV from step 16 to 17.~~ Corrected (same
  artifact): −184 / −170 meV from step 16 to 17.

### Pre-registered: step 13S, spin-pairing factors on the couplings (written 2026-10-18 23:20, before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_spin_bondlaw_r5_s{42,43}` = step 13's configuration with one
change, `redox.spin_pairing: true` (trace_redox SPEC 29). It launches when step 13F has been trained and scored
(`watch_step13s.sh`). Comparator: step 13, same seeds.

**Motive (exact, and measured).**
- The spinless configurations give the neutral ionic–covalent coupling √2 τ where the two-site singlet has 2t,
  and the ion resonances τ = t. No single τ is right for both.
- With the factor, the two-site Hubbard energies are exact for N = 1, 2, 3 (7 tests).
- Frozen step-13 weights (epochs 34 / 32), fixed T5 geometries: T5 −554.4 → −631.8 (seed 42) and −1417.6 →
  −1220.6 meV (seed 43). The seed spread falls 863 → 589 meV. The doped wetting energies are unchanged (the
  doped resonance has f = 1, as it must).
- Au₂ α∥ toward the reference: 13.34 → 14.03 and 12.61 → 13.51 Å³ (reference 16.5).

**Benchmark moved:** T5 for both seeds, and its seed spread.

| id | criterion (each seed unless stated) |
|---|---|
| **N1** | T5 within ±15 % of DFT (−792 to −1071 meV) for **both** seeds |
| N2 | \|T5(42) − T5(43)\| below step 13's |
| N3 | test E / F RMSE no worse than step 13 (same seed) by > 10 % |
| N4 | B1, B3, A1 as for step 13 |

**Reported:** the isolated Au₂ α∥(R) against the first-principles curve; τ(Au, Au); the T5 decomposition.

**Predictions (hypotheses).** The frozen-weight shift persists after retraining (N2), since the learned τ need no
longer compromise between the two sectors.

**Reading.**
- N1 and N2 pass: the spinless coupling was a source of the seed dependence.
- N2 fails: the shifts or Δ_ion carry it, which is 13F's and steps 14–17's ground.

### SPEC 32, the downfolded O 2p ↔ Au 6s bond: built, reviewed, measured; not yet pre-registered (2026-10-28)

**Built** (branch `downfold-ct-2026-10-28`, `trace_redox/downfold.py`, SPEC 32; 13 tests; redox 762, rest 1039 + 4):
the O–metal transfer folded out of the explicit ensemble into each configuration's energy as the exact block root of
its contact clusters, with Harrison's fixed V_spσ, the three p components of a closed-shell O donor, the moved
electron's Au–Au hops, the selection on dressed energies, folded partners dropped, μ_min > 0 enforced. A two-pass
review (code, equations) found 18 items in the first version; 17 fixed, one (the per-state self-energy) stated.

**Frozen-weight counterfactual on step 20** (`probe_downfold_counterfactual.py`; meV):

| | s42 | s43 |
|---|---|---|
| wet_u as trained → folded | +541 → **+318** | +735 → **+374** (DFT +929) |
| wet_d | −452 → −613 | −221 → −350 (DFT −2.7) |
| T5 | −993 → −931 | −956 → −724 (DFT −932) |
| separated limit (c1) | −0.105 meV PASS | −0.136 meV PASS |
| min μ on the references (c3) | +1.23 eV PASS | +0.36 eV PASS |
| relaxed flat dimer (c2) | 2.89 → 2.61 Å, −233 meV PASS | 2.89 → 2.57 Å, −307 meV FAIL (height) |

The fold dresses the flat geometry more than the upright one: the wrong direction. (The first version, with a
switch on [3, 4] Å, gave +1038 / +1213: near DFT for the wrong reason, the switch suppressing the flat contacts.)

**PBE's flat transfer against height** (`almo_split.run_direction`, 10 ions, both Au lowered rigidly; CT_cp in eV):

| h (Å) | Au–O (Å) | surface → Au | Au → surface | total |
|---|---|---|---|---|
| 2.89 (reference) | 3.51 / 3.69 | −0.173 | −0.139 | −0.319 |
| 2.60 | 3.27 / 3.47 | −0.260 | −0.243 | −0.515 |
| 2.37 | 3.09 / 3.30 | −0.333 | −0.346 | −0.694 |
| 2.10 | 2.89 / 3.11 | −0.419 | −0.454 | −0.890 |

Per contact (eight O–Au contacts, second order with the model's Δ = 3.4 eV) the implied coupling is 0.27 / 0.33 /
0.38 / 0.42 eV, a **constant 0.26–0.28 of Harrison's** 1/(r² + r₀²) at every height, while the vertical O-top
contact (2.26 Å, CT −0.962 eV) implies 2.05 eV, 0.78 of Harrison's. A radial decay alone cannot give a constant
ratio over 3.6 → 3.0 Å. Hypothesis: anisotropy of the O 2p tail, compressed toward the Mg²⁺ neighbours (the flat Au
sits over the neighbouring Mg) and free along the surface normal (the O-top Au). Decided by the O-top height scan
(2.60 / 3.00 / 3.50 Å, queued): a vertical ratio near 0.8 at 3.5 Å confirms it. Also measured: Au → surface is as
large as O → Au at the flat site (the Au 6s → Mg channel, in the model the Au⁺ + band configurations).

**PBE's total interaction against flat height** (`wetting_eda.run_geometry`, 10 ions; eV, relative to the reference
h = 2.89 Å where dE_SCF = −0.097 = E⁽¹⁾ +0.379 + relax −0.477):

| h (Å) | dE_SCF | first order E⁽¹⁾ | relaxation | model as trained (s43) |
|---|---|---|---|---|
| 2.60 | +0.235 | +0.629 | −0.394 | +0.189 |
| 2.37 | +0.868 | +1.753 | −0.885 | +0.756 |
| 2.10 | +2.958 | +4.731 | −1.772 | +0.998 (and +0.379 at 1.90) |

PBE's surface repels the lowered dimer: its first-order wall grows faster than its relaxation gains. The trained
model matches PBE to 2.37 Å and is too soft below it (its frame–frame repulsion is the backbone's, untrained there:
no ZBL-type core, SPEC 27's stated limitation). With the fold on, the model's relaxation grows faster than PBE's
−0.39 eV at 2.60 Å (the (c2) collapse to 2.6 Å): the same tail problem.

**PBE's transfer at the vertical O-top contact against height** (single contact; `almo_split.run_direction`, 10 ions):

| h = Au–O (Å) | surface → Au | Au → surface | v (Δ = 4.12 eV) | v (Δ = 6 eV) | Harrison | ratio |
|---|---|---|---|---|---|---|
| 2.26 | −0.962 | −0.169 | 2.21 | 2.59 | 2.62 | 0.84–0.99 |
| 2.60 | −0.551 | −0.069 | 1.60 | 1.90 | 2.00 | 0.80–0.95 |
| 3.00 | −0.294 | −0.029 | 1.14 | 1.36 | 1.52 | 0.75–0.90 |
| 3.50 | −0.130 | −0.012 | 0.74 | 0.89 | 1.12 | 0.66–0.79 |

(v from the two-level inversion v² = λ(Δ − λ) of the surface → Au share, with the model's frame Δ at the O-top and a
6 eV bound.) Along the surface normal the coupling follows Harrison's 1/(r² + r₀²) out to 3.5 Å, with at most a
20 % extra decay over the range; at the angled flat-site contacts it is 0.26–0.28 of it at every height. **The
suppression is directional, not radial.** The Au → surface share dies with height at the O-top (−0.17 → −0.01) but
stays as large as O → Au at the flat site (the Au 6s → Mg channel).

**The directional form, measured and implemented (2026-10-29; SPEC 32's constants).** The band-orbital overlap of
the Au 6s with the cluster's O 2p band is monotonic and within 10 % of Harrison's to 3.5 Å, and no smaller over Mg
than above an O: the suppression is not in the orbital tail (a first table from Löwdin natural orbitals was an
artifact, recorded in the theory note). Two measured effects carry it: (i) a single closed-shell Au⁺ acceptor
transfers 0.65 per contact over a cation of what it transfers above an O at the same distance (the in-plane hole
components lie 0.8–1.3 eV below the dangling p_z: a crystal-field shift β = 1.0 eV); (ii) the Au₂ dimer's flat
contacts transfer a further 0.55 less: its acceptor is the sp-hybridised σ\* (PBE LUMO, κ ≈ 0.4), outward lobes along
the axis, a node on the inward side where the flat O's sit; fitted to the vertical and inward-flat transfers,
κ′ = 0.22 with an overall scale c = 0.71 on V_spσ. Erratum in the earlier reading: the ten-ion cluster has two
quantum O's, so its flat dimer has four quantum contacts, not eight (0.37 of Harrison per contact, not 0.27).

**Frozen-weight counterfactual with the measured constants** (`DF_SCALE=0.71 DF_ANISO=0.22 DF_CF=1.0`; meV; the
pre-registered reading was "wet_u above the as-trained values"):

| | s42 | s43 |
|---|---|---|
| wet_u as trained → plain fold → measured constants | +541 → +318 → **+686** | +735 → +374 → **+745** (DFT +929) |
| wet_d | −452 → −613 → −426 | −221 → −350 → −187 (DFT −2.7) |
| T5 | −993 → −931 → −1111 | −956 → −724 → **−932** (DFT −932) |

Both seeds move toward DFT on wet_u (reading PASS). Rerun after a probe bug (a record key on the lone-Au₂ cell):
(c1) separated limit −0.105 / −0.136 meV PASS; (c2) the relaxed flat dimer 2.89 → 2.67 / 2.64 Å, −101 / −153 meV,
PASS on both seeds (the plain fold had collapsed it to 2.6 Å by −233 / −307 meV; as trained −22 / −7 meV); (c3)
min μ +1.23 / +0.36 eV PASS.

### Pre-registered: step 22 (branch `downfold-ct-2026-10-28`), the downfolded O 2p ↔ Au 6s bond (written 2026-10-29, before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_dfct_bondlaw_r5_s{42,43}` = step 20's
configuration and one change: `redox.downfold_transfer: true` with the pair's measured constants
`downfold_scale: 0.85`, `downfold_free_axes: true`, `downfold_anisotropy: 0.0`, `downfold_crystal_field: 0.0`
(SPEC 32, as amended below after the 28-ion flat split; `downfold_cutoff` 4.5 Å and `downfold_bond_cutoff` 3.2 Å at
their defaults). Comparator: step 20, same seeds. Nothing else changes: the
O–O couplings stay learned (the anion–anion anchor is step 23's candidate), so S6 is reported, not a criterion.

**Motive (measured; theory note 2026-10-18 §4c, this README above).** The model's O-top relaxation is a seventh of
PBE's; 94 % of PBE's transfer there is O → Au; a physical coupling inside the selected ensemble fails (step 21); the
fold is exact for its block, size-consistent, bounded, with order-one gradients on the charge-state levels; its
coupling's radial form (Harrison to 3.5 Å), crystal field (β) and the dimer's σ\* anisotropy (κ′) are measured. On
frozen step-20 weights it moves wet_u by +145 / +10 meV toward DFT and leaves the flat dimer in place.

**Benchmark moved:** the undoped wetting energy (B3) and T5 (4G Table II).

| id | criterion (each seed unless stated) |
|---|---|
| **D1** | wet_u closer to DFT (+928.9) than step 20's same seed by ≥ 100 meV (s42 +536.3, s43 +753.8) |
| **D2** | T5 within ±15 % of DFT (−792 to −1071 meV), both seeds |
| D3 | test E / F RMSE no worse than step 20's same seed by > 10 % |
| D4 | B1; ε∞(MgO) 2.5–3.5; Au atom α within 10 %; no configuration-cap refusal; no `DownfoldError` |
| D5 | the model's O-top relaxation (`probe_t5_first_order.py`, undoped upright, POL + CT) at least −0.6 eV (PBE −1.98; step 17 −0.29 / −0.22) |
| D6 | the relaxed flat dimer of the T5 protocol stays within 0.3 Å of the reference height |

**Reported:** wet_d (DFT −2.7); the first-order split of wet_u, wet_d, T5; the fold record on the references (min Δ,
min μ, λ); the clean-slab IP (S6 of step 21, PBE +4.80; exposed to the learned O–O couplings); the 28-ion flat
transfer against the σ\* prediction (below −0.80 eV).

**Predictions (hypotheses).** D1 passes for s42 (frozen +145) and is open for s43 (frozen +10; the backbone's +332
share must move). D2 passes (frozen 119 % / 100 %). D5 passes. D6 passes (frozen 0.21–0.24 Å).

**Amended before the launch (2026-10-29, 06:35; the 28-ion flat split).** CT_cp(flat, 28 ions) = −0.444 eV against
the pre-registered σ\* prediction of < −0.80 (count-only ≈ −0.47): the σ\* mechanism is falsified and
`downfold_anisotropy` goes to 0. The flat contacts, inward and outward alike, transfer ≈ 0.3–0.36 of a vertical
contact at equal distance; the Au⁺ angular test is withdrawn (contaminated by the ion-pair electrostatics). The
donor-side rule replaces it: a hole couples only through the O's least-coordinated (dangling) p axes
(`downfold_free_axes: true`), the cation-bonding components do not; `downfold_scale` 0.85 from the vertical data;
`downfold_crystal_field` unused (0). The frozen counterfactual and (c1)–(c3) are rerun with these before the run.
The 28-ion undoped shares (meV): FRZ −241.7, POL +555.8 (A +165.3, B +366.1), CT +652.3, total +966.4 (DFT +929):
the converged first-principles target for wet_u's relaxation is POL + CT = +1208.

**Frozen counterfactual with the amended rule** (`DF_SCALE=0.85 DF_FREE=1`, no anisotropy, no crystal field; the
pre-registered reading "wet_u above the as-trained values"): wet_u +541 → **+230** (s42), +735 → **+207** (s43);
wet_d −452 → −573, −221 → −407; T5 −804 / −614; (c1) −0.105 / −0.136 meV PASS; (c2) 2.89 → 2.66 / 2.63 Å, −105 /
−164 meV PASS; (c3) +1.23 / +0.36 eV PASS. **The reading fails.** Why: the rule weights every contact by l_z², which
cuts the upright site's four side O's (3.77 Å, 37°: 0.36) more than the flat site's (3.5–3.7 Å, 35–45°: 0.61–0.68),
and on frozen weights the model's upright dressing leans on those side contacts; PBE's angled contacts are weaker
than the rule's l_z² bound everywhere (0.30–0.36 against 0.50 at 45°; the upright side O's add nothing between 10 and
28 ions, a datum confounded by the POL/CT partition). The three frozen variants disagree (plain +318 / +374;
(0.71, 0.22, 1.0) +686 / +745; this +230 / +207), and the one that rose did so through an overall scale and a
mechanism the 28-ion split then falsified. The frozen indicator is therefore not decisive either way; D1 is read
after training, as pre-registered. Launched with the defensible physics; risk stated. Queued in parallel: the
angular transfer function of a single contact (the dimer tilted about the O-top O at 2.6 Å, 0 / 30 / 45 / 60°),
for step 23.

**Launched 2026-10-29 06:49 IST, stopped 09:08** (watch_step22.sh; suites before the launch: redox 769 passed, the
rest 1039 passed, 4 skipped): 107 s per forward against step 20's 0.7 s, no epoch in 2 h 15 m. Profiled on one
training structure (no grad): as trained 1.9 s, with the fold 6.0 s, from 166 dressing calls (one per window-search
candidate) and 281 SPEC 28 evaluations (every folded state of every candidate); the rest of the 107 s was the two
trainings swapped out beside a PySCF job. Fixed: the window search evaluates its candidates in one batch per frontier
configuration, and the selection's dressing takes Δ without SPEC 28 (meV-scale there; the solve keeps it). On the fast
path the frozen checks reproduce the slow path's numbers exactly (wet_u +230 / +207, (c1)–(c3) PASS) and the suites
pass (redox 769, rest 1039 + 4). **The relaunch is cancelled (09:12):** the first tilt point of the angular scan
(the dimer tilted 30° about the O-top O at 2.6 Å) transfers 1.05 of the vertical value where the dangling-axis rule
predicts 0.75; the rule is contradicted and does not train. Running: the 45° / 60° / 0° tilts, then a vertical dimer
over Mg0 at the flat height (the flat site with the acceptor vertical) and over a hollow at the same height (no
cation below), to separate a lateral-offset function from an acceptor-above-a-cation effect. Scored as step 20 plus `probe_t5_first_order.py` (D5), `probe_hole_terms.py` (S6, reported),
`probe_otop_channels.py`, `probe_downfold_counterfactual.py` (D6 and the fold record).

**Tilt scan in (11:20) and the hopping test withdrawn (11:25).** 45°: surface → Au −0.619 eV, 1.12 of the vertical
2.60 Å value, but not a single contact (the lower Au sits 1.86 Å from Mg0 and 2.83 Å from the other quantum O); 60°
(1.3 Å from Mg0) stopped as unphysical; 0° duplicates the vertical 2.60 Å point (−0.551 eV), not run. The one clean
angular datum is 30°: 1.05 of the vertical value, so the dangling-axis rule (0.75) stays contradicted and does not
train. The Löwdin-orthogonalised-hopping test (t̃ = F − Sε from the substrate's Fock matrix in the union basis) was
invalid as built: the probe Au 6s sits at −198 eV in its bare pseudopotential, so F dominates and the ratio over-Mg0 /
vertical (1.68) is the overlap structure again; withdrawn, the ALMO calculation itself is the non-perturbative
measurement (theory note §4c). The robust datum stays: every contact of an Au lying over Mg0 at 0.40–0.44 of
Harrison from 2.10 to 2.89 Å (0.55–0.6 of the vertical value at equal distance, 0.30–0.36 in λ), a factor constant
while the Au–Mg distance shrinks by 0.8 Å (not a cation Pauli push, which would steepen) and equal for inward and
outward O's (not a σ\* lobe along the axis). Candidate under test (NEW-TO-TRACE, theory note §4c): the lying
dimer's acceptor σ\* is polarised away from the surface by its Pauli repulsion (the ALMO POL step), its lobes
pointing up on both Au, so every donor below couples at an orientation-set factor; the standing dimer's lower Au
points its lobe down the axis at the O-top donor. Pre-registered readings of the two site runs in flight (a vertical
dimer over Mg0 and over a hollow, both at h = 2.89 Å, nearest O's at ≈ 3.6 Å; `$TRACE_FP_CACHE/site_direction/`):
(A) vertical over Mg0 transfers > 1.3 × the flat dimer per contact → the orientation carries the suppression; ≈ 1 →
the site does; (B) hollow ≈ Mg-top → the cation below is irrelevant; hollow > 1.3 × Mg-top → the cation below
suppresses. Step 22 stays cancelled until these are in.

**Correction and the first site result (12:36).** The 30° tilt point is not a single contact: in the ten-ion cluster
(the tilt runs' cluster) its lower Au sits 2.40 Å from Mg0 and 3.21 Å from a second quantum O, so its 1.05 contains a
second donor and the dangling-axis rule is *undetermined* by the scan, not contradicted (the sentence above is
superseded; the frozen counterfactual's negative reading stands on its own). Measured: the standing dimer over Mg0
(h = 2.89 Å) transfers surface → Au −0.184 eV, per contact 0.82 of the O-top contact at equal distance, where the
O 2p band's symmetry selection alone predicts 0.81 (the two highest band orbitals have zero overlap over Mg0): a
standing acceptor's site costs the band symmetry and nothing else. The hollow run was recast as `hollow_pp` (the
(+,+) square: both O's quantum, no point ion within 3.6 Å); site symmetry predicts hollow_pp = Mg-top per contact at
equal distance. Queued after it: the acceptor orbital measured directly from the B → Au transfer density
(pre-registered reading in the theory note §4c). Reading (A) waits for the flat reference at 2.89 Å in the same
cluster (running).

**Site runs in (19:22; produced 13:24 / 14:43, read late: my waiter watched the wrong filename).** Flat reference at
2.89 Å: surface → Au −0.173 eV (four contacts; per contact 0.336 of the O-top contact at equal distance). Hollow_pp:
−0.322 eV (two contacts at 3.26 Å; per contact 0.838 of the O-top; band symmetry predicted 0.813). **(A) = 2.45 > 1.3:
the orientation of the acceptor carries the flat-site suppression. (B) = 1.02: no cation effect beyond band
symmetry.** Every standing-acceptor site is reproduced by the O 2p band's symmetry selection; the lying acceptor
alone carries a factor 0.41. Step 22's coupling rule is therefore acceptor-side and orientation-dependent; its form
(the acceptor's shape, H2, or its level, H3) is decided by the acceptor-orbital runs now in flight (theory note §4c,
pre-registered readings). The donor-side rules of steps 21–22 (anchoring, dangling axis, crystal field, σ\* along
the axis) are closed as not the physics.

**The flat acceptor run, the fold's two-site ordering, partner states, and a bug (20:16; theory note §4c, CHANGELOG
Round 28; branch `pauli-hop-2026-10-29`).** The lying dimer's acceptor is p-type (gained density s 0.005, p_z 0.007,
p_xy 0.021 e per Au), its intra-Au dipole points down (−0.21 Å per electron: the "lobes up" form is falsified), its
σ* is unshifted (−4.41 eV; next virtual 3.44 eV higher). Mechanism: the quantum O's lie on the dimer's bisector
plane, σ*'s node, so only their axial p component reaches σ* and the rest must use the gerade virtual. Model:
`downfold_hop_ordering: pauli` (the antibonding two-site state lowest; one sign, no constant) and partner states (a
donor touching one atom of a pair now sees the pair's two-site states; the on-axis test measured the old fold at
−1.516 against the dense −1.552 eV). Then a bug: the fold's hop groups were keyed by (c · P + o), P the move count,
which collides between configurations in these cells (110 atoms, 24–36 moves) — one configuration's Au–Au hops were
written into another's block. Measured by a two-configuration test (fold −1.924 / −1.569 against the rebuild's −1.649
/ −1.649), fixed (key on the atom count). **Every frozen counterfactual of the fold above (+318 / +374, +686 / +745,
+230 / +207) and step 22's aborted launch ran under this bug and are superseded.** Suites after the changes: fold
tests 22, the rest 1039 passed, 4 skipped; the redox suite and the re-measured counterfactuals (P1 legacy ordering /
P2 Pauli ordering, both with partner states and the fix; pre-registered readings in the theory note) were running
when this was written.

**Re-measured counterfactuals and the amended pre-registration for step 22's relaunch (20:32; theory note §4c).**
On frozen step-20 weights with the partner states and the fixed key (wet_u / wet_d / T5, meV; DFT +929 / −3 / −932):
P1 legacy ordering, c = 1: s42 +338 / −613 / −951, s43 +442 / −350 / −792 ((c2) s43 FAIL, 2.57 Å); P2 Pauli ordering,
c = 1: s42 **+584** / −613 / −1196, s43 **+716** / −350 / −1066 ((c2) s43 FAIL by 0.02 Å); P3 Pauli, c = 0.85: s42 +525
/ −591 / −1116, s43 +646 / −350 / −996 ((c2) PASS both); (c1), (c3) PASS throughout. The ordering moves wet_u by
+245 / +274 meV (the doped cell untouched); P2's line (above as-trained +541 / +735 on both seeds) is met on s42 and
missed by 19 meV on s43; P3's is missed on wet_u, as a uniform scale must. The fold's gross dressing on frozen
weights is ≈ −3 eV per structure against PBE's −1.10 eV O-top transfer, so the frozen net wet_u is the difference of
two over-large numbers: the direction and size of the ordering are established, the magnitudes are a training
question. **D1–D6 above are superseded** (they were written for the dangling-axis rule under the key bug).

*Pre-registered, before the relaunch.* Run: `*_pfp_dfct_bondlaw_r5_s{42,43}` = step 20 + `downfold_transfer: true`,
`downfold_hop_ordering: pauli`, `downfold_scale: 0.85` (the vertical contact's κ′ = 0 value), `downfold_anisotropy`
0, `downfold_crystal_field` 0, `downfold_free_axes` false; partner states and the fixed hop-group key are code
(always on). Comparator: step 20, same seeds, at the reference geometries (wet_u +541 / +735, wet_d −452 / −221,
T5 −993 / −956 meV). Criteria: **E1** wet_u closer to +929 than step 20 on both seeds (primary); **E2** wet_d not
further from −3 than step 20 by more than 50 meV on either seed; **E3** the model's O-top relaxation
(`probe_t5_first_order.py`) above one third of PBE's (step 20: about one seventh); **E4** T5 within 90–110 % of
DFT on both seeds (step 20: 106 / 105 %); **E5** no `DownfoldError` in training, (c1)–(c3) PASS on the final model
and the relaxed flat dimer within 0.3 Å of 2.89 Å; **E6** energy and force RMSE within 10 % of step 20's. PASS = E1
and E5 on both seeds; E2–E4 and E6 reported. If E1 fails with E5 passing, the symmetric channel's coupling (V_spσ
standing in for an O 2p – Au 6p element; the model's lying / standing per-contact ratio 0.59 against the measured
0.41) is the next term, derived before it is coded. Launch after the acceptor-orbital runs release the CPU, with
`watch_step22.sh`.

**Launch withdrawn (20:44).** The fold's per-state self-energy, measured exactly against the full explicit-plus-folded
block on the two undoped references (s43 frozen, Pauli ordering, c = 0.85), came out at −0.79 eV (upright) and
−0.54 eV (flat) in a scratch assembly that then **failed its validation** (its single-configuration limit does not
reproduce the fold's roots; {t}): those numbers are a hypothesis, withdrawn as measurements. What is measured: the
ionic Au⁺Au⁻ configurations (26 % of the upright's ground state) share every folded state with the frame and their
leaves sit 0.36–0.63 eV below the folded states, so the per-state approximation's premise does not hold. Step 22
waits for the exact block built inside the fold (theory note §4c; branch `exact-fold-2026-10-29`); its E1–E6
pre-registration stands for the run that follows.

**The exact block, measured (21:08; theory note §4c).** On frozen step-20 weights (Pauli ordering, c = 0.85; wet_u /
wet_d / T5, meV): s42 per-state +525 / −591 / −1116 → exact **+862 / −345 / −1206**; s43 per-state +646 / −350 /
−996 → exact **+362 / −498 / −860**. The per-state fold's error on wet_u is +336 / −284 meV, opposite on the two
seeds (it tracks each seed's ionic-configuration resonance, μ_min 1.23 vs 0.36 eV). Step 22 relaunches with
`downfold_block: exact` added to its configuration (the one further field beside the ordering); E1–E6 unchanged.
Fold tests 28; redox suite 777.

**Launched 2026-10-29 21:08 IST** (`watch_step22.sh`; suites
before the launch: redox 777 passed, the rest 1039 passed, 4 skipped). Configuration = step 20 + `downfold_transfer:
true`, `downfold_hop_ordering: pauli`, `downfold_block: exact`, `downfold_scale: 0.85`, the anisotropy, crystal-field
and dangling-axis rules off; partner states and the fixed hop-group key in the code. Scored as pre-registered
(E1–E6). The flat-v2 acceptor run (the 6p populations in the dimer frame) runs beside it on one core.
The s43 exact-block relaxation on frozen weights collapses (2.52 / 1.98 Å, −1412 meV; s42 passes at 2.65 / 2.74
Å, −186 meV): the near-resonant ionic configurations of s43 take the exact block's full level repulsion as the Au
approaches the surface — the over-ionic Au₂ of the frozen weights exposed; E5 is the guard. Recorded for step 23: the
coupling's radial form (measured e^(−1.1 r) against Harrison's 1/r², a 3.0× over-coupling in λ at 3.5 Å
with the scale set at the O-top; theory note §4c).

**Pre-registered (21:16, before the result): the radial form's frozen counterfactual.** `downfold_decay: 0.45`
(anchor 2.26 Å) on top of the exact block, Pauli ordering, c = 0.85, frozen step 20, both seeds (`DF_DECAY`).
Comparator: the same without the decay (s42 wet_u +862, wet_d −345; s43 +362, −498; (c2) PASS / FAIL). Reading:
PASS if wet_u rises on both seeds (the 3.5 Å flat contacts fall ≈ 3× in λ, the O-top contact is unchanged) and (c2)
passes on s43; if wet_u rises but s43 still collapses, the collapse is the ionic resonance alone (the weights'), not
the contacts' range; if wet_u falls on either seed the radial reading is wrong and the form does not train.
Result (21:32): s42 +1198 / −254 / −1452, s43 +878 / −158 / −1036 (wet_u / wet_d / T5; without the decay +862 / −345
/ −1206 and +362 / −498 / −860). wet_u rises on both seeds and wet_d moves toward DFT on both; (c2) **crashed on both
seeds** (s42 a singular secular solve, s43 a non-positive-definite hardness at the Au) — an unphysical geometry
reached during the relaxation, so the pre-registered PASS is not met and the relaxation outcome is undetermined
until traced (theory note §4c).
Traced (21:52): s42's crash was a solver defect in the per-state selection screen (fixed, 31 fold tests); s43's was a
genuine short-range runaway (an Au into an O by step 40) under the plain exponential's extrapolation below the anchor
(now ramped off) and the ionic resonance. **Step 22 restarted at 21:52** on the fixed code (its 21:08 launch had the
solver defect loaded; logs kept as `log_aborted_2026-10-29_2108_solver.txt`); configuration unchanged (no decay).
Traces v2 (22:02): s42 converges to a modest rearrangement (+0.20 / +0.12 Å, −286 meV: (c2) PASS); s43 to a genuine
minimum with one Au on a surface O at 1.98 Å, −1.72 eV ((c2) FAIL): the fold's unmeasured short-range coupling
against the frozen repulsion, amplified by s43's ionic resonance. The radial form's pre-registration reads in its
second branch (wet_u / wet_d as predicted on both seeds; the s43 collapse is the weights', not the range).

**How E1–E6 are read (2026-10-31, before scoring; `read_criteria.py 22`).** Every comparator from its own score file:
E1 on the reference geometries as pre-registered (the fold probe's `scores/D6_fold.txt`; step 20's written with the same
probe on its own models, its "as trained" line, since step 20 never had one), and also on the T5 protocol's relaxed
geometries (`T5_wetting.txt`; step 20: +536.3 / +753.8 meV) — both printed, the reference-geometry one decides; E2 and
E4 from `T5_wetting.txt`; E3 from `S5_first_order.txt` (the wet_u row's POL + CT against the PBE cluster's); E5 from
the training log (DownfoldError count) and `D6_fold.txt` ((c1)–(c3) and the flat dimer's height change); E6 from the
`T5_wetting.txt` RMSEs. The parsers were checked against real outputs of the same formats before the run ended.

### Step 22 scored: seed 42 (2026-10-31 22:05; `read_criteria.py 22`); seed 43 crashed at epoch 31, resumed

**Seed 42 (wet_u / wet_d / T5 in meV; DFT +929 / −3 / −932; comparator step 20 s42).**

| | step 22 s42 | step 20 s42 | criterion |
|---|---|---|---|
| wet_u, reference geometries | **+802.1** | +541.3 | **E1 PASS** (pre-registered) |
| wet_u, T5 protocol (relaxed) | +685.9 | +536.3 | closer to DFT, PASS |
| wet_d, T5 protocol | −386.0 | −453.1 | E2 PASS (closer by 67) |
| T5, T5 protocol | −1071.9 (115 %) | −989.4 (106 %) | E4 FAIL (90–110 %) |
| T5, reference geometries | −1196.1 (128 %) | −993.4 (107 %) | — |
| E RMSE doped / undoped (meV/atom) | 1.548 / 0.823 | 2.728 / 2.493 | better |
| F RMSE doped / undoped (eV/Å) | 0.0838 / 0.0764 | 0.0886 / 0.0689 | E6 FAIL (undoped +10.9 %) |
| training refusals; (c1) / (c2) / (c3); flat dimer | 0; PASS / PASS / PASS; moved 0.21 Å (−154 meV) | — | **E5 PASS** |
| the model's wet_u relaxation share | — | — | E3 not read: `probe_t5_first_order.py` rebuilds the ensemble Hamiltonian without the fold's Σ and stops on its own (k2) check |

Also: ε∞(MgO) 3.33, α(Au) 5.434 Å³, the doped Au₂ free charge −1.000, Δ_ion +2.74 eV. **Seed 42 passes step 22's
primary criteria (E1 and E5).** The fold brings the undoped wetting energy from 58 % to 86 % of DFT at the reference
geometries (74 % on the relaxed protocol) and moves the doped one toward DFT too, so T5 = wet_d − wet_u overshoots
(115 %): the doped cell's transfer (wet_d −386 against −3) is the remaining error, as before.

**Seed 43 crashed at epoch 31** (training sub-step ~281) with `RuntimeError: dependency not found for ViewBackward0`
inside the force derivative of `forward_batched`, on the guard's first attempt at that step (not a retry), after ~14 000
clean forwards (val force 0.0846 eV/Å, val E 3.5 meV/atom at epoch 30). Not yet explained; the guard now saves a
reproducer (inputs, RNG and model state) on any exception in a training step. **Resumed (22:11) from its epoch-26
checkpoint** (`config_resume_crash31.yaml`; ~11 h; the crash may or may not recur — dropout masks are not
reproduced), scored on completion by `score_step22_s43.sh`, which then launches step 23. **The resume reproduced the
crashed run exactly through epoch 30** (val E 3.48 meV/atom, val force 0.08459 eV/Å, identical) **and passed epoch 31**
(val E 4.16, val force 0.0846; 2026-11-01 00:06): the autograd-engine error does not reproduce in the same deterministic
sequence — the crashed run shared the machine with seed 42 under heavy memory compression, the resume runs alone. Not
explained; recorded as a non-reproducible engine error, with the guard's reproducer dump armed should it recur. Step 23's own chain was
stopped before it could launch with step 22 half-scored.

**Seed 43 scored (2026-11-01 04:14; `read_criteria.py 22`) — step 22 FAILS its pre-registered PASS (E1 and E5 on both seeds).**

| | step 22 s42 | step 22 s43 | step 20 s42 / s43 |
|---|---|---|---|
| wet_u, reference geometries (E1, pre-registered) | **+802.1 PASS** | **+381.0 FAIL** | +541.3 / +735.3 |
| wet_u, T5 protocol (relaxed) | +685.9 | +220.7 | +536.3 / +753.8 |
| wet_d, T5 protocol (E2) | −386.0 PASS | −420.0 FAIL | −453.1 / −222.1 |
| T5, T5 protocol (E4) | −1071.9 (115 %) FAIL | −640.7 (69 %) FAIL | −989.4 / −975.9 |
| (c2): the flat dimer relaxed (E5) | 0.21 Å, −154 meV, PASS | **1.06 Å, −196 meV, FAIL** | — |
| E RMSE doped / undoped (meV/atom) | 1.548 / 0.823 | 1.785 / **4.354** | 2.728 / 2.493; 1.272 / 0.905 |
| F RMSE doped / undoped (eV/Å) (E6) | 0.0838 / 0.0764 FAIL | 0.1063 / 0.0976 FAIL | 0.0886 / 0.0689; 0.1091 / 0.0726 |

**Reading.** The trained wet_u tracks the frozen counterfactual of the same configuration within 60 meV on both seeds
(frozen: +862 / +362; trained: +802 / +381): the warm fine-tune at 1e-4 did not re-balance the model's wetting physics,
and the fold's effect is set by each seed's own explicit configurations — on s43 the ionic Au⁺Au⁻ configurations sit
near-resonant with their folded states in the flat geometry (μ_min 0.36 eV against 1.23 for s42), the exact block
dresses the flat structure more, and wet_u falls. Seed 43's undoped energy error (4.35 meV/atom against 0.91) says the
backbone could not absorb the fold in this protocol. Seed 43's flat dimer leaves its site (1.06 Å): the short-range
collapse step 24's wall addresses. Step 23 (κ = 0.31; frozen prediction +1053 / +713) launched 04:14 on both seeds as
queued; its comparator is step 22.

### Pre-registered: step 27 (branch `bond-pauli-wall-level-2026-11-05`), SPEC 33 per charge state (written 2026-11-06 21:04, before any score of steps 25 or 26 is read)

**Why, measured (theory note §4c, 2026-11-06).** SPEC 33's law is a frame term, measured on the neutral dimer. The doped
cell's dimer is Au₂⁻, whose σ* electron restores the axial density the σ bond removed: over the O-top O at 2.26 Å the
anion's orientation dependence is +0.200 eV (relaxed) against the neutral's +0.700 (the 20° / 60° check of the three-angle
law passed to 0.6 / 4.6 meV); five-angle coefficients C₁⁻, C₂⁻ = −0.1181, −0.0911 eV (neutral −0.0694, −0.4973). As a frame
term the law over-penalises tilted Au₂⁻ by up to 0.5 eV: the doped half of T5.

**Construction (NEW-TO-TRACE).** `RedoxConfig.bond_pauli_states` (needs `bond_pauli`): the law's coefficients become
functions of the s atom's charge state, (C₁(n), C₂(n)) with n = 0 the neutral's and n = −1 the anion's (the cation, one σ
electron, unmeasured: the neutral's, stated; the anion's radial exponents the neutral's until its 2.6 Å scan is read),
entered as a one-body term of the configuration energies (an [n, S] table added to the ladder beside the penetration
self-terms), so each configuration carries its own law; the frame-term path is then not applied; the wall stays a frame
term. Tests written red (`test_redox_bond_pauli_state.py`, 4, strict xfail until built).

**Run.** `..._pfp_bops_bondlaw_r5_s{42,43}` = step 26's configuration with one change, `redox.bond_pauli_states: true`;
cold start; comparator step 26, same seeds. Built and launched after steps 25 and 26 are scored (the suites and the
four red tests green first). Scored as step 26.

**Benchmark moved:** T5 through wet_d (the doped half) and its protocol.

| id | criterion (each seed unless stated) |
|---|---|
| **B1–B3, C4, C5** | as step 26 |
| **D1** | wet_d (references and protocol) closer to DFT (−2.7) than step 26's same seed by ≥ 50 meV, or within 50 meV of DFT |
| **D2** | training-set class offsets: doped − undoped within ±50 meV; the doped offset's magnitude not above step 26's by > 30 meV |
| D3 | wet_u not further from DFT than step 26's by > 50 meV; T5 reported; E / F RMSE within 10 % of step 26's |

**Amendment before the build (2026-11-08 16:15 IST; decision of 2026-11-08: resume the anion scan and implement step 27).**
(1) *The anion's radial exponents.* The 2.6 Å anion scan finished (0 / 40 / 70°; D⁻(70) − D⁻(0) = +0.0769 eV relaxed,
+0.0949 frozen; at 2.26 Å +0.2005 / +0.3245). Rule, written before the fit (a contact-atom-only two-equation estimate was
seen first and gave a non-decaying D2 at 2.6 Å, so the rule is the guard): `fit_bond_pauli.py --anion` fits C₁⁻, C₂⁻, p₂⁻
jointly at 2.26 Å (five angles) and 2.6 Å (three) with the neutral fit's form, bounds and free D0 per R, p₁⁻ held at the
neutral's value of that level (1.877 relaxed; 2.2925 frozen; two R cannot determine it); the fitted p₂⁻ is adopted only
if it lies strictly inside its bounds (0, 3) and the largest residual is ≤ 10 meV; otherwise the anion keeps the
neutral's (p₁, p₂) and the 2.26 Å five-angle coefficients (the pre-registered default), and the 2.6 Å misfit is recorded.
(2) *Bookkeeping, equivalent.* The frame path keeps the neutral law; the configuration energies carry each state's
difference from it, T(n) − T(0). This equals "the per-state law in the ladder, the frame path off" up to one constant
added to every configuration, which shifts the ensemble's free energy by that constant and nothing else; the red test
`test_the_module_applies_the_law_per_configuration_when_asked` pins this form (the frame configuration's diagonal
unchanged). The README sentence "the frame-term path is then not applied" is replaced by this.

PASS = B1 and B2 and D1 and D2. **Predictions (hypotheses).** The doped class offset moves toward zero by the anion
law's difference averaged over the doped class (the frozen estimate is computed when the term is built); wet_u unchanged
within 50 meV (the undoped cell has no anion). If D1 passes and D2 fails, the per-state law is right at the references
and wrong in the data's doped geometries (the cation or the radial law), recorded.

**Step 27 built (2026-11-08 16:20 IST; SPEC 33.3, CHANGELOG round 37).** The four red tests green (marker removed);
suites `tests/redox` 817 passed, the rest 1049 passed, 4 skipped; `BondPauli.energy` bitwise unchanged; the configs
differ from step 26's by `bond_pauli_states: true` only, `check_run_config.py` OK. **Not launched.**

**The frozen estimate (pre-registered; `probe_step27_frozen.py`, step 26's weights, the flag off → on; 300 training
structures; the references as single points; meV):**

| step 26 weights | doped offset (flat − upright) | undoped offset | doped − undoped | wet_u | wet_d | T5 |
|---|---|---|---|---|---|---|
| s42 off | −338.9 | −284.3 | −54.7 | +612.5 | −422.0 | −1034.5 |
| s42 on | −515.3 | −302.4 | −212.9 | +575.6 | −633.5 | −1209.2 |
| s43 off | −178.6 | −96.6 | −82.0 | +815.6 | −207.6 | −1023.2 |
| s43 on | −356.6 | −137.9 | −218.6 | +767.6 | −419.5 | −1187.2 |
| DFT | | | | +928.9 | −2.7 | −931.6 |

Class-mean residual change (on − off), both seeds: doped upright +149 / +151, doped flat −27 / −27, undoped upright
+10 / +33, undoped flat −8 / −8. **The prediction is contradicted on frozen weights:** the doped offset moves AWAY from
zero (by 176 / 178 meV) and wet_d away from DFT (by 212 / 212). Mechanism, from the law itself: the anion's anisotropy is
small, so relative to the neutral law it raises an upright Au⁻ (θ = 0: T(−1) − T(0) = +0.36 eV per contact atom at
2.26 Å) and lowers a tilted one (−0.14 eV at 70°). Artifacts ruled out: the sign is pinned by tests 1, 2 and 4; the flag
took effect (both seeds moved). **Reading:** step 26's doped error is flat-too-stable (wet_d −422 / −208 at the
references against −2.7), the opposite sign of what step 27's premise ("the neutral law over-penalises a tilted Au₂⁻")
would correct; frozen, D1 and D2 would fail. A cold-start retrain could redistribute the error, but the local backbone
cannot see the dopants (> 10 Å away), so the doped-only shift has no frame term to absorb it. Launch held for a decision.

**Incident (2026-11-05 18:57): step 20's checkpoints overwritten by steps 25 and 26, caught and restored.** The step 25 and 26
configs were derived from step 20's with `checkpoint_dir` replaced but `model_save_path` and `plot_dir` not, so the four
trainings saved their best models as step 20's `model.pt` (first at 12:57 for s42 and 13:02 for s43, last at 18:26 / 18:34)
and clobbered each other's saves. Found when a frozen probe on "step 20" showed a frame term; the file's redox block
carried `bond_pauli: true`. All four trainings and both watchers stopped by exact PID at 18:55. Restored: s42 from
`model_last.pt` (epoch 40, full checkpoint; reproduces the scored wet_u +541.3 exactly, so step 20's best was its last
epoch), s43 from `warm_from_step20_s43.pt` (the scored best's weights, epoch 0, no optimizer state — `model_last.pt`
gives +744.6, a different epoch); the overwritten files kept as `model_overwritten_by_steps25-26_2026-11-05.pt`. Every
probe that read step 20 today finished before 12:57 (the frozen checks, the relaxation traces, the rigid-approach curves
at 12:33), so their results stand; the 18:30–19:00 term-probe checks read the wrong models and are void (rerun on the
restored files). The four configs are fixed (every path key now its own directory; verified none points to step 20);
steps 25 and 26 relaunch from scratch (≈ 7 h and 5 h of training lost), their pre-registrations unchanged. **Verified 2026-11-05 18:59:** the restored files reproduce the scored wet_u / wet_d / T5 and the dimer's Δ_ion(2.30 Å) to every
printed digit (s42 +541.3 / −452.2 / −993.4 meV, +1.985 eV; s43 +735.3 / −220.7 / −956.1, +2.378); `probe_t5_terms.py`
reproduces step 20's stored split exactly and, with the law and relaxed wall switched on frozen, closes on the module's
energy with a frame row (+41.3 / +42.5 meV at wet_u / wet_d). **Relaunched 2026-11-05 18:59** (watchers `watch_step25.sh`,
`watch_step26.sh`; the stopped runs' logs kept as `log_stopped_2026-11-05_1855_wrong_save_path.txt`). Step 26's trainings paused 2026-11-06 15:13 (SIGSTOP, state intact) so that step 25 finishes first on the shared machine; they resume when step 25 is scored.

### Pre-registered: step 26 (branch `bond-pauli-2026-11-05`), the lone atom's frozen wall against O²⁻ (written 2026-11-05 12:33, before the run and before PBE's approach curve is read)

**Why, measured (theory note §4c, 2026-11-05; trace_redox SPEC 33, the wall part).** Step 25's term is the bond's
modulation of a bonded Au's frozen repulsion against O²⁻, measured relative to a lone Au. The lone Au's own frozen
interaction with O²⁻ (penetration electrostatics + Pauli; E1 = 7.73 / 3.70 / 1.43 / 0.85 / 0.39 / 0.10 eV at 1.8 / 2.0 /
2.26 / 2.4 / 2.6 / 2.9 Å on the 6-ion cluster) is no explicit term of the model: SPEC 23 / 25 are frame-relative and the
frame wall (step 24) covers Mg²⁺–Au⁰ only. The backbone carried it, learned at the data's 2.0–2.5 Å contacts, with a
cliff below: on frozen step-20 weights the upright dimer's energy falls −0.5 / −1.4 / −296 eV at 2.03 / 1.83 / 1.63 Å,
and the term's θ = 0 shielding (−0.6 eV, a force of 1.9 eV/Å toward the O) tips the frozen upright dimer over it in five
steps. Hypothesis for step 25: the refit backbone may hold the wall at the data and lose it below (a collapse of the
upright dimer in the protocol's relaxation would show as B4 / the relaxed wet numbers). With the wall explicit, wall +
law rises toward the anion at every angle (tested 1.8–2.9 Å) and reproduces the measured frozen interaction of the
upright dimer within 0.06 eV at the frozen level.

**Run.** `..._pfp_bopw_bondlaw_r5_s{42,43}` = step 25's configuration with one change, `redox.bond_pauli_wall: true`;
cold start as step 25; comparator step 25, same seeds. Launches when the suites pass on the extended table and the
lone atom's E1 at 2.1 / 3.2 / 3.5 Å is in the table (measured now). Scored as step 25 plus `probe_upright_approach.py`.

**Benchmark moved:** T5 through its protocol (both relaxed dimers) and wet_u.

| id | criterion (each seed unless stated) |
|---|---|
| **B1–B3** | as step 25 (the flat dimer stays; tilt path A within 0.1 / 0.2 eV of PBE; class offsets doped − undoped within 50 meV) |
| **C4** | the upright dimer's approach as trained, E(h) − E(2.26), within 0.2 eV of PBE's (6-ion cluster, dE_SCF(cp); `otop_upright_scan.py`, running) at h = 2.0 and 2.5 Å, both seeds; E(1.8) reported |
| C5 | the upright references relaxed by the protocol stay within 0.3 Å of their heights (no collapse), both seeds |
| B4, B5 | as step 25, against step 25 |

PASS = B1 and B2 and B3 and C4. **Reported:** C5; the rigid rotation; the term and wall class means.

**Launched 2026-11-05 13:06 and withdrawn 13:20, before any result** (both trainings stopped by exact PID
after 14 minutes; logs kept as `log_withdrawn_2026-11-05_1320_frozen_wall_level.txt`; branch
`bond-pauli-wall-level-2026-11-05` made first). Reason: the wall as launched was the *frozen* lone atom's E1(R) under
the *relaxed* law — wall + law was then neither level's dimer interaction (theory note §4c, 13:27). **Amendment (2026-11-05 13:29):**
the wall follows the law's level (`BondPauliWall(level)`): at the relaxed level it is the lone atom's I_ALMO(R)
(+0.472 / +0.047 / −0.028 eV at 2.26 / 2.6 / 2.9 Å; 1.8–2.1 and 3.2–3.8 Å being measured), so that wall + law = the
dimer's relaxed, transfer-free interaction (checked: −0.030 / −0.255 / −0.207 eV upright at 2.26 / 2.6 / 2.9 Å within
0.1) and the backbone's residual is the charge transfer. **C4's target, now measured** (PBE, 6 ions, dE_SCF(cp),
E(h) − E(2.26)): +1.165 / +0.163 / +0.014 / 0 / +0.195 / +0.590 eV at 1.8 / 2.0 / 2.1 / 2.26 / 2.5 / 2.9 Å; the frozen
step-20 models already match it within 0.07 eV (the collapse is a property of the relaxation path, not of the rigid
approach). Criteria and comparator unchanged; relaunch when the relaxed table is complete and the suites pass. **Relaunched 2026-11-05 13:55** (`watch_step26.sh`), concurrently with step 25, after the
relaxed table (I_ALMO at 1.8–3.8 Å) was entered and the suites passed on it (redox 808, the rest 1039 + 4). As trained:
the law's relaxed constants (−0.069, −0.497, 1.877, 0.382) and the relaxed wall; the configs are unchanged from the
withdrawn launch (the level is the code's, not the config's).

**Predictions (hypotheses).** C4 passes on both seeds (the wall is the physics of the approach; the backbone learns a small
residual); C5 passes; B1–B3 as step 25's. If step 25 passes B1–B3 but fails C5, the wall is what it lacked.

### Pre-registered: step 25 (branch `bond-pauli-2026-11-05`), the bond-oriented Pauli repulsion (written 2026-11-05 05:24, before the run)

**Why, measured (theory note §4c, 2026-11-03/04; trace_redox SPEC 33).** Steps 21–24 failed the same way: the flat
dimer of the T5 protocol rolled over into a tilted O-top state. PBE (wetting_eda's cluster, 25 and 28 quantum ions agree
to 0.01 eV) keeps the flat dimer as a minimum behind a ≈ 0.25 eV barrier and puts the tilted state 0.8 eV above the
upright dimer on the same O; the models put it 0.24 eV below to 0.34 above. ALMO decomposition: tilted − upright =
FRZ +1.40 + POL −0.46 + CT −0.06 eV, and the frozen term is not pairwise in atomic densities (the σ bond shields the
upright contact atom by 0.76 eV). Its law, D = D₀ + D₁ cos θ + D₂ P₂(cos θ), D₂ ∝ √E1(R), D₁ ∝ E1(R)^0.888, is fitted to
≤ 8 meV at four R and holds to 9 % over a 14-fold range; relaxed (ALMO) C₁, C₂ = −0.273, −0.394 eV at 2.26 Å. On frozen
step-20 weights the data-referenced form keeps the flat dimer in place on both seeds and brings the tilt path within
0.1–0.3 eV of PBE without moving T5 (< 0.5 meV). How a refit backbone shares the isotropic part is not decidable frozen.

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bop_bondlaw_r5_s{42,43}` = step 20's
configuration with one change, `redox.bond_pauli: true` (`bond_pauli_level: almo`); cold start (lr 1e-3 → 5e-5,
40 epochs), as step 20. No fold, no wall: one change; comparator step 20, same seeds. Launches when the suites pass and
the pre-registered ALMO-level radial check at 2.6 Å (theory note, 05:18) is read (a FAIL there changes the constants
first). Scored as step 24 plus `probe_tilt_path.py --scan` on paths A and the rigid rotation (`$TRACE_FP_CACHE/tilt_path/
path_step24_s42.json`, `rotation_path.json`) and `probe_class_residuals.py` on the training set.

**Benchmark moved:** T5 (4G Table II) through its protocol — the relaxed flat dimer, whose collapse set steps 21–24's
protocol values — and wet_u.

| id | criterion (each seed unless stated) |
|---|---|
| **B1** | the undoped flat dimer relaxed by the T5 protocol stays within 0.3 Å of its reference heights (2.89 Å), both seeds |
| **B2** | tilt path A as trained: the barrier E(0.4) within 0.1 eV of PBE's +0.255 and E(1) within 0.2 eV of −0.197, both seeds |
| **B3** | training-set class offsets: doped − undoped within ±50 meV (step 20: −1 / −12); the common offset reported |
| B4 | wet_u (references and protocol) closer to DFT than step 20's same seed or within 50 meV; wet_d not further from DFT by > 50 meV; T5 reported |
| B5 | test E / F RMSE within 10 % of step 20's same seed |

**Launched 2026-11-05 11:50** (`watch_step25.sh`) after the suites passed on the final constants
(redox 803, the rest 1039 + 4) and the pre-registered 2.9 Å ALMO check passed (the two-R constants predicted the third R
to 12 %). The constants as trained (SPEC 33, relaxed level): C₁, C₂ = −0.092, −0.492 eV, p₁, p₂ = 1.877, 0.378.

PASS = B1 and B2 and B3. **Reported:** the rigid rotation against PBE (+0.100 / +0.357 / +0.642 / +0.745 / +1.451 eV at
20 / 40 / 60 / 70 / 90°); the term's own class means; the gas-phase Au₂ sector (unchanged by construction: no anion).

**Predictions (hypotheses).** B1 and B2 pass on both seeds (the frozen check passed B1 and nearly B2 with the
data-referenced form; the training refits the isotropic part). B3 passes (the term is geometric and equal in the two
classes to 41 meV frozen). wet_u's protocol value rises toward its reference value on both seeds (no collapse to absorb).

**Reading.** B1–B3 pass: the missing physics of the flat dimer's collapse was this term, and the line resumes from step
25 with the fold and the wall as the next single changes. B1 or B2 fails: the term as entered (one partner, the relaxed
level's assumed radial scaling, no Mg) is insufficient or the backbone double-counts it; the trained rotation curve says
which. B3 fails: the term is not class-neutral after training, recorded.

### Step 24 scored (2026-11-03 14:04; `read_criteria.py 24`): FAIL on both seeds

| | s42 | s43 | PBE / DFT |
|---|---|---|---|
| **W1** dimer over Mg at 2.0 / 2.3 Å (eV, within 30 %) | +1.988 / +0.444 PASS | +2.773 / +0.665 FAIL (37 % high at 2.0 Å) | +2.017 / +0.540 |
| **W2** flat dimer relaxed (heights, Å; within 0.3 Å) | 2.89 / 2.89 → 2.20 / 3.07, −1118.7 meV **FAIL** | → 2.17 / 3.91, −352.4 meV FAIL | |
| wet_u, references / protocol (relaxed) | +968.9 / **−102.5** | +639.3 / +323.9 | +928.9 |
| wet_d, references / protocol | −357.6 / −349.9 | −317.2 / −315.3 | −2.7 |
| T5, references / protocol | −1326.5 / −247.4 | −956.5 / −639.2 | −931.6 |
| W4 (E6) E / F RMSE within 10 % of step 23 | FAIL (undoped F +13.2 %) | FAIL (doped E +14.6 %) | |

- **The wall works where it acts.** Over Mg it reproduces PBE's repulsion on s42 and overshoots on s43; the flat
  dimer no longer sinks onto Mg.
- **The flat dimer now tilts onto an O.** One Au drops to 2.17–2.20 Å above the surface, the Au–O bond length of the
  O-top site, while the other rises. With the fold off, s42's flat dimer instead moves up to 3.21 Å (−150 meV): the
  fold's O-top attraction drives the tilt. On s42 the tilted state lies 0.15 eV below the model's own upright dimer,
  which is why the relaxed wet_u falls to −102.5 meV. Whether PBE has the flat dimer as a minimum against this tilt is
  not measured.
- **A probe artifact, not a model result:** W1's "fold off" rows reach −332 eV at 2.0 Å on s42. Switching the fold off
  restores the module's own O–Au couplings, which steps 22–24 never trained (the fold zeroes them); the as-trained curve
  is normal.
- `G_t5_stack` and `G_t5_terms` failed as probes, as for step 23 (they do not read the current stack).

### Step 23 scored (2026-11-02 06:04; `read_criteria.py 23`): seed 42 PASS, seed 43 FAIL — step FAIL

| | s42 | s43 | DFT |
|---|---|---|---|
| wet_u, references (E1, against step 22) | +935.7 (802.1) PASS | +636.5 (381.0) PASS | +928.9 |
| wet_u, T5 protocol (relaxed) | +742.2 | +371.5 | |
| wet_d, references / protocol (E2) | −394.2 / −386.3 PASS | −287.9 / −282.0 PASS | −2.7 |
| T5, references / protocol (E4, 90–110 %) | −1329.8 / −1128.5 FAIL | −924.4 / −653.5 FAIL | −931.6 |
| E5: DownfoldError 0; (c1) (c2) (c3) | PASS PASS PASS (flat dimer 0.17 Å) | PASS **FAIL** PASS (flat dimer tilts: 3.48 / 2.19 Å) | |
| E6: E / F RMSE within 10 % of step 22 | PASS | PASS (undoped E 4.40 meV/atom, as step 22's 4.35) | |

- The frozen prediction (wet_u +1053 / +713 at the references) held in direction, not size: trained +936 / +637.
- Seed 42's wet_u at the references sits 7 meV from DFT. That is not a result: its wet_d is −394 against −2.7, so
  its T5 is 143 % of DFT at the references. The undoped wetting is near; the doped one is not.
- Seed 43's flat dimer tilts, with one Au to 2.19 Å above the surface. That is the short-range collapse that step 24's
  frame wall is pre-registered against (W2).
- E3 is unread: the first-order probe does not read fold models. `G_t5_stack` and `G_t5_terms` failed as probes:
  the first switches on the exclusion beside Gordon–Kim, which the module refuses, and the second misses the fold's
  diagonal. These are known probe limitations, not model results.
- Step 24 (step 23 + the frame wall) launched 06:04 on both seeds; comparator step 23.

### The Au₂ two-site sector: the Hubbard prior derived and frozen-checked (2026-11-01; written 13:54)

**The mapping (exact for the model's own sector).** On a gas-phase Au₂ the module's ensemble holds three
configurations, Au⁰Au⁰, Au⁺Au⁻, Au⁻Au⁺: a two-site model with the diabatic gap Δ_ion = H[ion, ion] − H[cov, cov] and
|H_cov,ion|. Its spinless form couples Au⁰Au⁰ to the symmetric ionic combination by √2 |H_cov,ion|, the exact singlet
by 2t (ESTABLISHED: two-site Hubbard). So with `spin_pairing` off (steps 20–24) the physical singlet is reproduced at
Δ_ion = U, |H_cov,ion| = √2 t; with SPEC 29 at τ = t. The adiabatic σσ* excitations fix U = E_S − E_T and
t = √(E_T E_S)/2 (`tools/electrostatics/au2_hubbard.py`; TDA-PBE U / t 0.900 / 1.409, 0.995 / 1.249, 1.063 / 1.146,
1.164 / 1.009 eV at 2.30 / 2.47 / 2.60 / 2.80 Å; EOM-CCSD within 0.16 / 0.09 eV).

**The model's sector (`probe_dimer_hubbard.py`, `probe_dimer_decompose.py`; the term split closes on H to 1 meV).**
Δ_ion is 1.1–2.5 eV above U on steps 20 and 22, both seeds, rising ~2 eV/Å against 0.53. The fixed terms
(IE − EA − K/R + SPEC 23 penetration + SPEC 25 Pauli) alone give −0.37 / +0.45 / +0.94 / +1.52 eV (3.8 eV/Å; the
Gordon–Kim Pauli difference of Au⁺Au⁻ against Au⁰Au⁰ is −1.61 eV at 2.30 Å, −0.16 at 2.80); the learned environment
shift adds +2.5 to +3.8 eV and the bound sector, SPEC 26 and SPEC 28 −0.8 to −1.1 eV. The coupling is −15 % to +4 % off at
2.30 Å and 25–57 % large at 2.80. The ground-state ionic weight is 0.22–0.31 against the physical 0.36–0.42.

**Artifact checks on the targets.**
- *State mixing.* The picked singlet roots carry 0.36–0.67 of the atomic-6s σσ* excitation. Natural transition
  orbitals (`au2_hubbard_nto.py`): each is one transition (leading pair 0.955–0.988 TDA; 0.954 CCSD at 2.47 Å) whose
  hole is 32–58 % 5d (TDA; 25 % CCSD) against the triplets' 16–19 % (the ground σg's own s–d hybrid). The singlet is
  mixed with 5dσ → σ*; the triplet is clean (pair 0.998–0.999). Stated as a limitation of U: it carries that mixing.
- *Charge-state transfer of t (SPEC 29's premise, `au2_ion_hopping.py`).* Au₂⁻'s σ → σ* excitation 2t gives t =
  1.335 / 1.232 / 1.142 / 1.016 eV against the neutral's 1.409 / 1.249 / 1.146 / 1.009: one t holds for the neutral and
  the anion (T5's two sectors) to ≤ 0.07 eV. The cation is unreliable (σ weights 0.07–0.60, a d-hole state at 2.47 Å).
- *The ionic dipole.* The singlet's TDA transition dipole is 0.54× the bare two-site √p R at every R
  (`au2_transition_dipole.py`): the bare-dipole α_CT (19–26 Å³) would exceed the dimer's whole α∥. The model's own
  configuration dipoles are screened by its bound sector (SPEC 30), so the check that decides is the model's α.

**Frozen check 1: the model's α with the prior's values imposed** (`probe_dimer_hubbard_alpha.py`; a constant δ on the
ionic energies and a constant coupling scale, fixed at zero field; run before a written criterion — designed as a
falsifier of the minimal-model reading after the dipole check, so reported, not scored):

| α∥ (Å³) at 2.30 / 2.47 / 2.60 / 2.80 Å | as trained | Δ → U and \|H\| → √2 t |
|---|---|---|
| step 20 s42 | 10.48 / 11.43 / 12.86 / 15.22 | 12.97 / 14.75 / 16.26 / 18.96 |
| step 20 s43 | 9.44 / 11.93 / 13.90 / 15.49 | 12.61 / 14.68 / 16.64 / 19.05 |
| step 22 s42 | 10.19 / 11.13 / 12.39 / 14.70 | 12.92 / 14.73 / 16.25 / 18.94 |
| step 22 s43 | 8.69 / 11.15 / 13.37 / 15.32 | 13.07 / 14.47 / 16.39 / 18.93 |
| PBE0 / def2-TZVPPD+2d | 14.01 / 15.62 / 17.03 / 19.49 | |

The prior's values move α∥ from 16–38 % low to 3–10 % low on all four models and collapse the seed spread; the
ground-state ionic weights become the physical 0.421 / 0.402 / 0.387 / 0.361 exactly. Δ → U carries it; the coupling
alone moves α∥ by < 0.6 Å³. The minimal-model values are consistent with the dimer's measured response. Separately:
the model's α⊥ is 2.9–6.0 Å³ at 2.30 Å as trained (PBE0 8.13; flat in R), a bound-sector defect the prior does not touch.

**Pre-registered (written 13:54, before the run): frozen check 2, the prior's direction against T5**
(`probe_prior_alignment.py`). For each of step 20 s42 / s43 and step 22 s42 / s43 as trained: the prior
L = Σ_R [(Δ_ion − U)² + (|H_cov,ion| − √2 t)²] (eV², the four R, TDA-PBE targets); its update direction in the run's
optimizer geometry (Muon's polar factor with the aspect scale on the hidden matrices, AdamW's g/(√v̂ + ε) with the
trained second moments, fresh momenta) and, secondarily, the Euclidean −∇L; the first-order changes of wet_u, wet_d
and T5 (four unrelaxed references) along it, scaled to the step that lowers L by 10 %; the parameter groups carrying
them. **Reading:** *aligned* if wet_u rises (toward +928.9) on all four models in the optimizer geometry;
*conflicting* if it falls on all four; *mixed* otherwise. wet_d is reported (DFT −2.7: up on every model). First order
on frozen weights: it says whether the prior's gradient and the wetting energies share weights in a helpful direction,
not what a training with both losses reaches. A step-25 pre-registration follows only on *aligned*, or on a stated
reason under *mixed*.

**Frozen check 2, scored (14:03; `probe_prior_alignment.py`; meV for the step that lowers L by 10 %).** The rebuilt
optimizer groups match the checkpoints (4 / 39 / 32 tensors; 10 never stepped). The probe's as-trained wet_u reproduces
the scored values (+541.3 / +735.3 / +802.1 / +381.0).

| model (spin_pairing off) | L (eV²) | optimizer: wet_u / wet_d / T5 | Euclidean: wet_u / wet_d / T5 |
|---|---|---|---|
| step 20 s42 | 9.24 | +4.2 / −2.1 / −6.3 | +6.4 / +1.2 / −5.3 |
| step 20 s43 | 14.46 | +13.2 / −3.9 / −17.1 | +25.2 / +15.1 / −10.2 |
| step 22 s42 | 14.24 | +6.8 / −4.9 / −11.6 | +11.4 / −0.4 / −11.7 |
| step 22 s43 | 20.42 | +7.1 / −13.7 / −20.8 | +28.3 / +16.8 / −11.5 |

- **Reading: *aligned*** (wet_u rises on all four in the optimizer geometry). But wet_d falls on all four and T5 moves
  away from DFT on three. The prior's gradient lives in the learned shift network (61–79 % of |∇L|²) and the backbone's
  center projection (13–30 %); the coupling head carries ≤ 7 %.
- **The first order is the regime** (finite steps along the same direction, `FINITE=1`): ×1 within 1–3 %; ×5, which
  nearly halves L (9.24 → 4.95, 20.42 → 11.41 eV²), within 8–15 %: wet_u +22.7 / +33.5, wet_d −12.4 / −58.2, T5 −35.0 /
  −91.7 meV (step 20 s42 / step 22 s43); the weights restored exactly.
- **With SPEC 29 on** (frozen; `SP=1`): wet_d is unchanged, as SPEC 29 requires; wet_u rises on all four (+541 → +646,
  +735 → +1019, +802 → +863, +381 → +763), so T5 overshoots (−1098 to −1257). The gas sector shows it is compensation:
  |H_cov,ion| becomes 2.40–3.26 eV against √2 t 1.43–1.99 while Δ_ion stays 1.1–2.5 eV high; p_ion rises to 0.29–0.36,
  and α∥ moves by < 0.7 Å³ (still 16–38 % low). The prior's direction there is *mixed* (wet_u +3.2 / −3.9 / +3.4 /
  −18.0) and small. With both prior values imposed, α∥ is identical with SPEC 29 on and off to every printed digit on
  all four models (the injection is exact; SPEC 29 acts on the gas dimer only through H_cov,ion).

**Decision (stated reason): the prior is not pre-registered as step 25.** It is the right physics for the gas-phase
sector: its targets survive the artifact checks with one stated limitation (the singlet's 5d mixing; CCSD hole 25–33 %
5d against the triplet's 4–7 %), and it brings an independent observable, α∥, to within 3–10 % of PBE0. But it is not
a T5 step: nearly halving its error moves wet_u up 23–33 meV and wet_d and T5 away from DFT, on frozen weights. T5's
remaining error is wet_d (−221 to −452 against −2.7), the doped sector, which the gas-phase prior does not reach. Kept
as a validated transferability prior. SPEC 29's frozen wet_u gain is compensation, not evidence; its premise (one t for
the neutral and the anion) is now measured to ≤ 0.07 eV.

### Pre-registered: step 24 (branch `frame-wall-2026-10-31`), the frame pair's own Pauli wall (written 2026-10-31 07:00, before the run)

**Why, measured.** The frozen model's flat-dimer relaxations fall to Au–Mg ≈ 1.8 Å; the 4G training set holds no Au–Mg
contact below 2.43 Å; SPEC 25's Pauli term is frame-relative, so the frame pair's own repulsion is carried by the
backbone alone. PBE (`mgtop_scan.py`, a dimer standing over a surface Mg, relative to 3.30 Å): +2.017 / +0.540 / +0.107
eV at 2.0 / 2.3 / 2.6 Å; the model (step 20 s43, fold off) +0.228 / +0.309 / +0.102 — the wall reproduced at 2.6 Å and
0.11× PBE's at 2.0 Å. The frame pair's frozen-density PBE repulsion (Mg²⁺–Au⁰): +0.888 / +1.713 / +3.333 / +6.618 eV at
2.4 / 2.2 / 2.0 / 1.8 Å, reproduced by K_TF = 0.918 times the non-additive Thomas–Fermi kinetic energy of the frozen
densities (residuals ≤ 0.04 eV; `build_frame_wall.py`, `trace_redox/wall.py`, tests `test_redox_frame_wall.py`).

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_dfct_dec_wall_bondlaw_r5_s{42,43}` = step 23's
configuration and one change: `frame_wall: true`; warm from step 20, fine-tune 1e-4 → 5e-6, as steps 22–23. Queued
behind step 23's scoring (`watch_step24.sh`). Comparator: step 23, same seeds. Criteria: **W1** the trained model's
dimer-over-Mg curve within 30 % of PBE's at 2.0 and 2.3 Å on both seeds (`probe_mgtop_curve.py`, as trained); **W2**
the flat dimer of the T5 protocol relaxed within 0.3 Å of its reference heights on both seeds (the (c2) of
`probe_downfold_counterfactual.py`); **W3** wet_u, wet_d, T5 reported against step 23 (E1–E4 as for step 22); **W4**
energy and force RMSE within 10 % of step 23's. PASS = W1 and W2. Frozen look (informational, not a criterion: the
frozen backbone double-counts at the data distances): s43 with the fold (c 0.85) and the wall +2.75 / +1.04 / +0.26 eV
at 2.0 / 2.3 / 2.6 Å against PBE's +2.02 / +0.54 / +0.11 — the wall restored, its excess for the training to absorb.

### Pre-registered: step 23 (branch `exact-fold-2026-10-29`), the coupling's measured radial form (written 22:02, before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_dfct_dec_bondlaw_r5_s{42,43}` = step 22's
configuration and one change: `downfold_decay: 0.45` (per Å beyond Harrison, softplus-ramped above
`downfold_decay_anchor: 2.26` Å; SPEC 32, the measured e^(−1.1 r) form of the O 2p – Au 6s coupling). Queued behind
step 22's scoring (`watch_step23.sh`). Comparator: step 22, same seeds. Criteria: E1–E6 as for step 22 with step 22 as
the comparator (E1: wet_u closer to +929 than step 22 on both seeds; E5: no refusal, (c1)–(c3), the flat dimer within
0.3 Å; E2–E4, E6 reported). Frozen prediction (counterfactual above): wet_u +336 / +515 meV above step 22's frozen
values, wet_d toward −3 on both seeds; the s43 flat relaxation is the risk (1.98 Å on frozen weights).

**Step 22 stopped (23:07).** Seed 43's log measured the exact block at **24.0 s per forward** (step 20: 0.7 s; the
per-state fold: 2 s) — not trainable; seed 42 died at 23:03 on a non-finite gradient the nanguard could not recompute
away (first seen in the carrier induction's pair geometry, `penetration.pairs_within`; dump `nanguard.pt`). Both
stopped (my PIDs), logs kept as `log_aborted_2026-10-29_2152_cost.txt`; step 23's queue chain stopped with it. The
exact block is being profiled and the dump replayed before any relaunch.
Profiled quietly (23:09; `cost_quiet.py`, step-20 s43 weights, energy + forces per structure, 2 threads): structures
0 / 76 / 120 / 205 — per-state 2.40 / 2.29 / 1.88 / 2.13 s, exact 2.36 / 2.14 / 1.90 / 2.07 s. The exact block costs
nothing beyond the per-state fold; the 24 s/forward was contention (two trainings, two probes, two suites, the
traces and a PySCF job on 10 cores). The cost objection is withdrawn; the reproducible non-finite gradient on
structure 76 is the one open blocker.
Traced and fixed (23:42; CHANGELOG Round 28): the exact block's eigendecomposition in the training's double backward
(degenerate folded pairs), plus two zero-row norms (`pairs_within`, the σ\* lobe). Replay of the dump: 61 → 0
non-finite gradients. **Step 22 relaunched 2026-10-29 23:51 IST** on the fixed code once the redox suite passed (783), step 23 re-queued
behind its scoring (logs of the 21:52 launch kept as `log_aborted_prev.txt` / `..._2152_cost.txt`). Its first timing
lines: s43 1.11 s per forward, **s42 19.1 s** — seed 42's random initialisation admits 73–74 explicit configurations
(2–3 once trained) and the selection screens them through the fold ~80× per forward. Two exact optimisations
(CHANGELOG Round 28: distinct moved configurations evaluated once; the hop blocks vectorised) bring structure 205 from
82 to 11.9 s at that initialisation; both seeds restarted 2026-10-30 00:34 IST (the 23:51 logs kept as `log_aborted_2026-10-29_2351_cost2.txt`). Its
timing lines: s42 12.45 s per forward as profiled, **s43 31.2 s** (1.11 s on the 23:51 code) — unexplained by the code
changes for a small ensemble; the trainer's wrapper now names every forward slower than 5 s, the bisection count is
halved (CHANGELOG), and **both seeds restarted again 2026-10-30 01:37 IST** (with the
step-23 chain; the 00:34 logs kept as `log_aborted_2026-10-30_0034_cost3.txt`). The slow-call log ({t}, first 50–100 forwards): the slow forwards are undoped structures whose explicit ensembles
at the random initialisation hold **120–380 configurations** (s42 up to 171 with 4 824 leaves at 61 s; s43 up to 379
with 7 916 leaves at 203 s; 2–3 once trained), and the fold's screening costs in proportion (s42 14.2 s, s43 35.6 s
per forward on average). Step 20 trained through the same phase at 0.7 s because screening without the fold is free.

**Protocol amendment ({t}): warm start.** Steps 22 and 23 train from step 20's weights (the same seed's `model.pt`,
copied as a weights-only checkpoint with `epoch: 0`; `resume_from` with the optimiser and schedule fresh,
`atomic_energies_from_model` dropped in favour of the checkpoint's references) instead of a random initialisation:
the ensemble starts at 2–3 configurations and the other terms re-balance under the fold, which is the question the
frozen counterfactuals (all on these weights) posed. E1–E6 unchanged (comparator step 20 for step 22, step 22 for
step 23); the cold start's hundreds-of-configurations phase is a backlog item (a cheaper screen for cold starts).
**Both seeds restarted warm 2026-10-30 02:12 IST** (with the step-23 chain; the
01:37 cold-start logs kept as `log_aborted_2026-10-30_0137_coldstart.txt`). First timing lines ({t}): **s42 1.91 s,
s43 1.17 s per forward** (s42 has six structures at 57 configurations, ~11 s each; s43 none above 5 s) — an epoch in
about an hour, against the cold start's 14–36 s per forward.
**Seed 42 refused at forward 333 (02:50):** the exact block's own rule — a training structure whose block ground state is
30 % explicit (E* 1.42 eV below the folded manifold), the strong-coupling regime at a close O contact, where the
charge-transfer state must be explicit. The handling is a build, not a patch: promotion of the dominant folded state
to an explicit configuration with the fold's coupling as its explicit coupling (theory note §4c, pre-registered
design). Seed 43 was refused the same way at forward 301 (explicit weight 0.279). **Step 22 is stopped** (watcher and the
step-23 chain killed, {t}); both failure dumps (`redox_failure0.pt`) are the test cases for the next build — the
promotion (design A) or the full-block ensemble (design B), theory note §4c; the choice is open.

**Promotion built (10:19; decision: design A; branch `promotion-2026-10-30`; CHANGELOG Round 29, SPEC 32).** On a
refusal the block's dominant folded state is promoted to an explicit configuration with its energy and the fold's
couplings, and the block re-solved (bounded at 8). Fold tests 36. Both failure dumps (`redox_failure.pt`) replay
through the module: forward and the training's double backward finite in normal and anomaly mode — s42 two promoted
states (explicit weight 0.974), s43 five (0.728). **Step 22 relaunched 2026-10-30 10:27 IST** once both suites passed (redox 786, rest 1039 + 4): warm from step 20's
weights, exact block with promotion, Pauli ordering, c = 0.85 (with the step-23
chain; the 02:12 logs kept as `log_aborted_2026-10-30_0212_refusal.txt`). E1–E6 unchanged; the promotion is
part of the fold under test. **Seed 43 died at forward 304 (10:50):** eight one-at-a-time promotions left the explicit
weight at 0.229 — the structure's charge-transfer manifold is nine near-degenerate states (the p_z states of four O's
on either Au). Built ({t}, CHANGELOG Round 29): **batch promotion** (the set of folded states by descending weight
until the explicit weight reaches 0.6, their mutual hops as explicit couplings; four passes at most) and the
**re-partition rule** (a promoted configuration's own moves are never folded — with them folded, four promotions
fell 5.7 eV into the second electron transfer, O⁻ → O⁰ with Au₂²⁻, untrained charge states). Both dumps replay with
finite double backward (s42 four promoted states, weight 0.999; s43 ten, 0.755). Seed 42 (then at forward 300+,
no refusal) stopped for code consistency (`log_aborted_2026-10-30_1027_promotion1.txt`); **Both seeds relaunched 2026-10-30 11:09 IST** once the suites passed (redox 786, rest 1039 + 4): warm from step 20,
exact block with batch promotion and the re-partition rule, Pauli ordering, c = 0.85 (with the
step-23 chain). At 350 forwards (11:46): no refusal; the structures that killed the earlier
launches pass in situ with the block's ground energies of the replays — s42 call 333 (6 configurations after
promotion, −863.30 eV), s43 call 304 (14, −849.48 eV); averages 1.23 / 1.10 s per forward, a promoted forward ~6 s.
Epoch 1 (11:54; warm start, so not comparable with step 20's cold epoch 1): s42 train loss 5.86, E 46 meV, force RMSE
2.41 eV/Å (MAE 0.25: a few outliers) — val loss 0.051, E 29.0 meV, force RMSE 0.206; s43 train 0.074 / 65 meV / 0.175
— val 0.204 / 11.9 meV / 0.450. Step 20's end state: val E 2.7 / 0.95 meV, force 0.071 / 0.085. No refusal. The s42 training outliers (structures
149 / 166 / 141 / 100 / 24, loss 4–7 at epoch 1) are flat-dimer MD snapshots on which the fold adds a systematic
−1.1 to −1.25 eV at the warm-start weights (fold off +0.03–0.09; |ΔF| 0.7 eV/Å on surface Mg; no promotion, explicit
weight 0.82–0.86, gap 4.6–4.9 eV; the upright-like 205 gets −0.54) — the offset the weights must re-absorb, not a
pathology (`outliers.py` → `probe_fold_outliers.py`).

**Seed 42 diverged at epoch 4 of the 11:09 run; protocol amended (15:01).** Per epoch (val E meV/atom / val force eV/Å):
s42 29.0 / 0.206, 16.0 / 0.215, 27.0 / 0.162, **307 / 7.33**, 202 / 4.71; s43 11.9 / 0.450, 20.2 / 0.159, 7.7 / 0.148,
18.5 / 0.278, 10.7 / 0.230. Cause, checked: the warm start restarted the cosine schedule at learning rate 1e-3 with a
fresh optimiser on converged weights (step 20 ended at 5e-5) — the train energy error rose from ~10 to 30–65
meV/atom over epochs 1–3 and seed 42's state energies drifted until single structures (203, 21) dominated a step;
at the epoch-3 checkpoint those structures have no promotion and explicit weights ~0.9 (not a fold threshold) but
energy errors of −2.8 to −4.1 eV. **Amendment: the warm start is a fine-tune, `learning_rate: 1.0e-04`,
`lr_min: 5.0e-06`** (cosine, 40 epochs; step 20's own rate near its epoch 34), for steps 22 and 23. **Step 22
relaunched 2026-10-30 14:59 IST** (with the step-23 chain; the 11:09 logs and
epoch-3 checkpoint kept as `*_aborted_2026-10-30_1109_lr1e-3*`). **Step 23's `downfold_decay` amended 0.45 → 0.31
before it ran:** the O-top series measured Δ at every height (theory note §4c) and the consistent fit gives
κ = 0.307 /Å; 0.45 assumed an unshifted Δ at 3.0 Å. The frozen counterfactual is restated with 0.31 (running).

**Frozen counterfactuals on the promotion code (15:17; step 20's weights, exact block, Pauli ordering, c = 0.85;
wet_u / wet_d / T5, meV; DFT +929 / −3 / −932; (c1)–(c3) as pre-registered).**

| configuration | s42 | s43 |
|---|---|---|
| as trained (step 20) | +541 / −452 / −993 | +735 / −221 / −956 |
| step 22's (no decay) | +862 / −345 / −1206; (c1)–(c3) PASS | +362 / −498 / −860; (c2) **FAIL** (one Au to 1.98 Å, −1413 meV) |
| step 23's (κ = 0.31) | +1053 / −295 / −1347; all PASS (2.69 / 2.75 Å, −266 meV) | +713 / −258 / −971; all PASS (2.73 / 2.59 Å, −116 meV) |

The references are unchanged by the promotion (no refusal there: step 22's numbers reproduce those of 2026-10-29).
The measured decay moves wet_u by +191 / +351 meV and wet_d toward DFT by +50 / +240 meV, and removes seed 43's
short-range collapse on frozen weights. **Restated frozen prediction for step 23 (against step 22):** wet_u higher on
both seeds, wet_d closer to −3 on both, (c2) PASS on both. On frozen weights s42 overshoots wet_u (+1053 against
+929) — the uniform over-coupling (c 0.85 against the consistent 0.67) the retrain or step 24 must absorb.

**The fine-tune's first epochs (16:35; train E meV/atom, train force eV/Å | val E, val force):** s42 6.65, 0.116 | 2.13,
0.112 → 3.50, 0.111 | 3.83, 0.113; s43 5.89, 0.126 | 6.55, 0.121 → 4.67, 0.122 | 3.36, 0.122. The energy error falls
from the warm start's level instead of rising (the 1e-3 run: 30–65 meV/atom by epoch 3): the learning-rate reading
holds. ~48 min per epoch (s42 1.9 s, s43 0.9 s per forward); 40 epochs end around 2026-11-01 00:00 IST. Step 20's end
state for reference: val E 2.7 / 0.95 meV/atom, val force 0.071 / 0.085 eV/Å.

**Stopped at epoch 8 — the fold's derivatives were wrong (19:57; CHANGELOG Round 30).** The training force error rose
every epoch at learning rate 1e-4 (s43 0.126 → 0.143, s42 0.116 → 0.130 eV/Å; two epoch metrics NaN). Checked against
finite differences on training structure 205 (three explicit configurations): with the exact block the **forces were
6–27 % off −dE/dx** (a shared folded state's energy received its gradient once per configuration), the coupling weights'
gradient 4.8 % off (the fold's hop matrix was detached), and the force loss's parameter gradient 1.7 % off (the
fixed point's ground vector was detached). All three fixed test-first (fold tests 40; each new test fails when its
defect is re-introduced); after the fixes 3e-9 / 1e-6 / 7e-8. **Superseded:** every (c2) relaxation since SPEC 32
(including s43's 1.98 Å "collapse" and the PASS / FAIL readings above), and every fold training so far. **Unchanged:**
the frozen-counterfactual energies (wet_u / wet_d / T5 need no derivative). **Step 22 relaunched again 2026-10-30 20:44 IST** after the promoted-couplings fix (CHANGELOG Round 30 item 4; suites
redox 791, rest 1039 + 4; with the step-23 chain; the 20:06 logs kept as
`log_aborted_2026-10-30_2006_promotedcouplings.txt`). First epochs (22:25; train E meV/atom, train force | val E, val force):
s42 7.19, 0.086 | 3.05, 0.084 → 2.27, 0.079 | 1.74, 0.078 → 1.95, 0.076 | 3.59, 0.078; s43 9.38, 0.104 | 6.62, 0.095 →
4.04, 0.094 | 3.61, 0.092 → 3.51, 0.093 | 3.62, 0.092. The force error falls with the corrected derivatives (the
same fine-tune on the defective code rose to 0.128 / 0.138); ~34 min per epoch, the 40 epochs end around 2026-10-31
19:00 IST, then scoring and step 23. Epoch 12 (2026-10-31 06:47): s42 val E 1.49 meV/atom, val force 0.0741 eV/Å; s43
3.47, 0.0889 (step 20 final: 2.72 / 0.0708 and 0.95 / 0.0848); 50 min per epoch measured, the run ends around
2026-11-01 06:00 IST. The 20:06 launch, superseded: **Step 22 relaunched 2026-10-30 20:06 IST** on the derivative-fixed code once the suites passed (redox 790, rest
1039 + 4): fine-tune 1e-4 → 5e-6, warm from step 20, exact block with promotion, Pauli ordering, c = 0.85 (with the
step-23 chain). The (c2) relaxations are re-measured on the fixed code beside it.

**(c2) re-measured with correct forces (20:08; frozen step 20, exact block, Pauli, c = 0.85; energies reproduce exactly).**
No decay (step 22's configuration): s42 one Au to **1.45 Å** above the surface, −1906 meV, FAIL; s43 2.24 / 2.47 Å,
−828 meV, FAIL. κ = 0.31 (step 23's): s42 2.04 / 2.95 Å, −1555 meV, FAIL; s43 2.73 / 2.61 Å, −118 meV, PASS. **The
PASS readings of the restated table above (and every earlier (c2) reading) were artefacts of the inconsistent forces
and are withdrawn.** On frozen step-20 weights the fold over-binds the flat dimer at short range: the weights' Pauli and
short-range repulsion were trained without the fold's attraction, and the fold at c = 0.85 couples 1.19× PBE's v_eff at
2.0 Å (the O-top series). The trained model's E5 (the flat dimer within 0.3 Å after relaxation) is the test that
matters: the fine-tune now receives correct forces, so whether the repulsion re-balances is measured, not assumed. Two
risks, stated before the result: the training set's coverage of Au–surface distances below ~2.5 Å (the relaxation can
leave the data), and c = 0.85 against the consistent 0.67 (the step-24 candidate).


### Pre-registered: step 21 (branch `coupling-anchor-2026-10-27`), Harrison-anchored couplings (written 2026-10-27, before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_anc_bondlaw_r5_s{42,43}` = step 20's
configuration and one change: `redox.coupling_anchor: harrison` (SPEC 31). Every pair but a frame cation-anion pair
(here Au–O, Au–Au, Au–Mg, O–O, Mg–Mg, Mg–Al, Al–Al) takes Harrison's universal two-centre value times a learned factor
in (½, 2); Mg–O and Al–O keep the learned value under SPEC 27's bound. Comparator: step 20, same seeds.

**Motive (measured, 2026-10-27).**
- The model's O-top relaxation is a tenth of PBE's: −0.22 / −0.14 eV against −1.98 (step 17,
  `probe_t5_first_order.py`). Its wet_u share is +112 / +103 meV against +1239.
- PBE's O-top transfer is 83 % O → Au (`almo_split.py --direction`).
- The model's O → Au coupling is 0.1–1.0 eV against Harrison's 2.75 eV, stuck small by the vanishing gradient of an
  even F(τ).
- Unconstrained O–O couplings reach 28 eV (step 20 s42: clean-slab IP −267.7 eV).
- Frozen-weight counterfactual on step 20 (`probe_anchor_counterfactual.py`, factor 1):

  | seed | wet_u | wet_d | T5 | IP |
  |---|---|---|---|---|
  | s42 | +541 → +989 | −452 → −224 | −993 → −1213 | −267.7 → +5.14 |
  | s43 | +735 → +746 | −221 → −204 | −956 → −950 | +4.89 → +3.90 |

  In s43, Au–O alone gives +1090; its trained Au–Au coupling (2.4 eV against Harrison's 1.75) takes it back.

**Benchmark moved:** the undoped wetting energy (B3, the previous R2) and T5 (4G Table II).

| id | criterion (each seed unless stated) |
|---|---|
| **S1** | wet_u closer to DFT (+928.9) than step 20's same seed by ≥ 100 meV (s42 +536.3, s43 +753.8) |
| **S2** | T5 within ±15 % of DFT (−792 to −1071 meV), both seeds |
| S3 | test E / F RMSE no worse than step 20's same seed by > 10 % |
| S4 | B1; ε∞(MgO) 2.5–3.5; Au atom α within 10 %; no configuration-cap refusal |
| S5 | the model's O-top relaxation (`probe_t5_first_order.py`, undoped upright: POL + CT) at least −0.5 eV (PBE −1.98) |
| S6 | clean-slab IP within 1.5 eV of PBE's +4.80 (`probe_hole_terms.py`) |

**Reported:** wet_d (DFT −2.7); the full first-order split of wet_u, wet_d and T5; the learned factors on Au–O,
Au–Au and O–O; the O → Au configuration energy (`probe_otop_channels.py`); configuration counts.

**Predictions (hypotheses).** S6 passes (frozen: 5.14 / 3.90). S5 passes. S1 passes for s42 (frozen +989) and is
open for s43 (frozen +746; the Au–Au compensation). S2 is open (frozen 130 % / 102 %).

**Result (scored 2026-10-28 10:04; best checkpoints at epochs 30 / 39). Step 21 fails.**

| id | s42 | s43 | verdict |
|---|---|---|---|
| S1 | T5 relaxation refused (ConfigurationSearchError: the amplitude closure exceeds 512) | wet_u −1718.5 meV (step 20 +753.8; DFT +928.9) | **FAIL** |
| S2 | missing | T5 +1922.9 meV (−206 %) | **FAIL** |
| S3 | E d/u 8.102 / 2.457 meV/atom (step 20 2.728 / 2.493); F 0.1791 / 0.1713 (0.0886 / 0.0689) | E 1.291 / 2.297 (1.272 / 0.905); F 0.1041 / 0.1068 (0.1091 / 0.0726) | **FAIL** |
| S4 | ε∞ 1.034; a cap refusal; B1 −1.100; Au α 5.508 | ε∞ 3.258; B1 −1.079; Au α 5.440; no refusal | **FAIL** (s42) |
| S5 | −1.62 eV, but the separated limit fails (F − F1 −541 meV, dE +1343 meV at 8 Å) | −1.74 eV; separated limit fails narrowly (−2.1 meV against 0.1) | not valid |
| S6 | IP +2.795 eV | IP −1.753 eV (the hole's coherent energy −9.25 eV) | **FAIL** |

**What training did with the anchor (measured).**
- The learned factors went up. Au–O at 2.26 Å is 2.85 / 3.90 eV (Harrison 2.62; s43 at 1.49 times). Au–Au is
  −2.60 / −2.47 eV (Harrison −1.64 at 2.5 Å; 1.5–1.6 times).
- s43's hole band is now too wide: the clean-slab IP is negative. At charge 0, the O → Au class (Au⁻ + O hole over
  10 O) is in the upright ensemble, so the O–O coupling now acts on neutral energies through the CT channel, and
  training uses it.
- s43's unrelaxed references give wet_u +1308.9 (CT share +1586.4), but the relaxed protocol gives −1718.5. The flat
  dimer relaxes into a state with 0.49 e on Au₂ and a coherent energy of −2.35 eV, overbound by the anchored O → Au
  channel.
- s42 went fully strong-mixing: Au⁺Au⁻ 1.79 eV below the neutral dimer at the upright contact, 222 configurations,
  ε∞ 1.03 (its polarisation moved out of the bound sector), and gradient spikes of 39–400 from epoch 11 on six
  undoped structures.
- Predictions: S6 "passes" was wrong (the factor moved after training); S1 for s42 was wrong; S2 failed.

**Reading (hypotheses).**
- Couplings at the Harrison scale (1–4 eV) take the configuration ensemble out of the weak-mixing regime that
  SPEC 21's selection was built for. The selection then truncates the product space: the separated limit fails, by
  541 meV in s42.
- A learned factor up to 2 lets training buy CT stabilisation that no data penalise. The relaxed flat dimer
  overbinds; the neutral training data hold no hole states and few distorted contacts.
- Magnitude alone is not the missing physics: PBE's O-top bond also limits the transfer, and the model has no such
  limit at these couplings.

**Diagnostics (2026-10-28, frozen step-21 weights; `probe_step21_diagnostics.py`, `probe_switch_forces.py`).**
- **D1, separated limit against the selection threshold** (a_off 0.05 / 0.02 / 0.005, a_on = 4 a_off):

  | seed | dE (meV) | B's configurations | E_B change from a_off 0.05 to 0.005 |
  |---|---|---|---|
  | s42 | +1343.0 / +356.8 / +508.5 | 55 → 955 | −3.04 eV |
  | s43 | +1.5 / +310.2 / +176.6 | 37 → 55 | −1.27 eV |

  Pre-registered reading (|dE| below 10 % at 0.005): **the truncation hypothesis is falsified as stated.**
  Measured instead: under the anchored couplings the selected ensemble's energy is not converged in the threshold at
  all. The slab alone moves by eV and its configuration set by 17×, so additivity cannot hold at any threshold tried.
- **D2, switch-band forces** (s42; spikes 205, 76, 163, 145, 202, 3 against controls 67, 251, 169, 123, 249):
  - Spike structures: 86–218 moves in the band, switch-derivative force max 0.22–0.69 eV/Å.
  - Controls with ~100 configurations: 0.047 and 0.58 eV/Å. Controls with 3 configurations: 0.
  - Pre-registered reading (spike min > 0.5 and ≥ 10× controls): **falsified as stated.**
  - Measured instead: wherever the ensemble is large, SPEC 21's switch derivative puts 0.2–0.7 eV/Å of non-physical
    force on atoms, spike or not.
- **D3, the relaxed flat dimer** (undoped wetting reference, Au relaxed):
  - s43: Au drops from 2.89 to 2.05 / 2.27 Å above the surface onto O (Au–O 2.38–2.51 Å), Au charges −0.23 /
    −0.26, ensemble mostly O → Au (Au⁻ + O hole 0.374, double transfer 0.058). Relaxation energy **−3.04 eV**.
  - s42: heights 2.09 / 2.05 Å, Au–Au stretched to 2.818 Å, charges −0.47 / −0.52. Relaxation energy **−3.57 eV**.
  - The 4G DFT keeps the flat dimer at 2.89 Å. The anchored O → Au channel overbinds at short Au–O contacts: the
    coupling grows as 1/d² and the transfer configuration's energy falls with the Au⁻–hole attraction. Nothing in
    the model's short-range repulsion stops it; the backbone was trained near equilibrium.

**Conclusion from step 21 (measured).** A physically sized coupling inside SPEC 21's selected ensemble does not work.
The ensemble leaves weak mixing, its energy depends on the threshold by eV, the switch adds 0.2–0.7 eV/Å of
non-physical force, and short contacts overbind by 3 eV.

### The model's wetting split as the first-principles decomposition splits it (2026-10-27; `probe_t5_first_order.py`, reported)

**Method.** For each T5 reference, A = Au₂ alone in the same cell (Q = 0 / −1), B = the slab (Q = 0 / +1). The
first-order state is ρ_A ⊗ ρ_B in AB's Hamiltonian, with the band orbital frozen at B's own (NEW-TO-TRACE; the
analogue of the Heitler-London product). Exact split: dE = backbone + e_b0 + [F1_AB − F_A − F_B] + [F_AB − F1_AB], each
bracket by term, the screened Coulomb split into bare and bound-response parts. Checks passed on both seeds: term sums
(1e-9), the module's H and F reproduced, b = Ωu0 and W = ΩWΩᵀ, totals (1e-8), and the separated limit (Au₂ at
mid-vacuum, 8 Å from either face: F − F1 = 0.0000 meV, dE −0.14 / −0.12 meV). A first version took A isolated: Au₂'s
interaction with its periodic images (+18.97 meV, measured) then failed the separated limit; A is now in the cell.

**Step 17, wet_u (meV): the model's relaxation is a tenth of PBE's.**

| share | s42 | s43 | PBE cluster (28 ions, converged) |
|---|---|---|---|
| first order (E1) | +106.0 | +192.6 | −272.2 |
| relaxation (POL + CT) | +112.0 | +102.6 | +1238.6 |
| backbone (learned) | +211.4 | +332.0 | — |
| dimer deformation (isolated) | +8.7 | +14.5 | +10.3 |
| periodic images of A | +32.0 | +14.0 | — |
| total | +470.2 | +655.7 | +976.7 (DFT periodic +928.9) |

Per geometry the model's relaxation is −0.216 / −0.141 eV (upright) and −0.104 / −0.038 eV (flat) against PBE's
−1.984 and −0.746: about a tenth in both. The model's first order has the opposite sign to PBE's, and the learned
backbone carries the upright preference instead (+211 / +332). The isolated dimer's deformation matches PBE (+8.7 /
+14.5 against +10.3; Au₂⁻ −23.3 / −10.7 against −22.8).

**wet_d and T5.** wet_d: E1 −120.7 / −111.7, POL −584.0 / −597.2, CT +5.6 / +12.4, backbone +208.6 / +336.6, images
+79 / +107. T5 −840.8 / −894.0 = E1 −226.7 / −304.3 + POL −598.3 / −663.2 + CT −92.2 / −24.3 + the rest. The model makes
T5 mostly from the bound sector's response to Au₂⁻ (first-order `response` −554.7 / −530.7, `alpha28` −131.2 / −185.4).
For comparison, a classical point-charge image estimate (ε∞ = 3, both Au at −0.5 e) gives −46 meV between flat and
upright. The doped first-principles side is not yet a model of the 4G cell: that cell holds three Al and two band
electrons, and the one-Al cluster has neither (erratum, theory note 2026-10-27). So T5's attribution is not compared
until it is.

**Reading (hypotheses, to be tested).** (1) The undoped wetting energy (R2, B3) lacks O-top relaxation in the explicit
physics: about 1.1 eV, which PBE's ALMO split (10 ions) puts 62 % in charge transfer and 38 % in polarisation. (2) The
model's doped wetting comes from a bound-sector response to Au₂⁻ more than ten times the classical image estimate.

**Correction (2026-10-27 evening): the bound sector's first order.** The split above counted the screened term's
whole W − J part, and e_b0, as polarisation. That includes each fragment's own induced moments acting on the other
(the slab's surface dipoles on Au₂⁻), which the first-principles side carries in its frozen densities. The probe now
freezes the bound moments at each fragment's own optimum (y_A* ⊕ y_B*) for the first order. Polarisation is the
moments' mutual relaxation −⟨U(y_A* ⊕ y_B*) − U(y_AB*)⟩ plus SPEC 26 and 28. Checks: U(y*) equals the closed form
e_b0 + u0·m + ½ mᵀWm (1e-9); the separated limit's frozen penalty is +0.163 meV, equal and opposite to the old
response term, so at 8 Å the response is all mutual polarisation. Step 17, corrected (meV, s42 / s43):

| share | wet_u | wet_d | T5 |
|---|---|---|---|
| first order | −3.9 / +123.3 | −510.9 / −458.8 | −507.0 / −582.2 |
| polarisation | +124.1 / +135.2 | −193.8 / −250.1 | −317.9 / −385.3 |
| configurations (CT) | +97.8 / +36.7 | +5.6 / +12.4 | −92.2 / −24.3 |
| relaxation (POL + CT) | +221.9 / +171.9 | −188.2 / −237.7 | −410.1 / −409.6 |

Per geometry, the undoped relaxation is −0.289 / −0.218 eV upright and −0.068 / −0.046 flat, against PBE's −1.984 and
−0.746. Reading (1) stands with these numbers: about a seventh of PBE's. **Reading (2) was wrong:** an artifact of the
attribution. Corrected, the model's T5 is mostly first order (−507 / −582 meV), as in the one-Al PBE cluster (−904),
with relaxation second (−410 against −282). S5 of step 21 is read with the corrected probe (step 17's value:
−0.29 / −0.22 eV).

### Pre-registered: step 20 (branch `pauli-refit-2026-10-25`), SPEC 25's K fitted to first principles (written 2026-10-25, before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bondlaw_r5_s{42,43}` = step 17's configuration
(with its amendment) and one change: `redox.pauli_table: first_principles` (new field; default `gordon_kim`,
unchanged for every other run). The overlay `trace_redox/data/pauli_fp.npz` (`tools/electrostatics/build_pauli_fp.py`)
replaces only the Pauli series: each one-carrier K of the AuMgO element pairs is fitted to the frame-relative
frozen-density PBE interaction in the Watson-sphere convention over the 4G data's contact distances; two-carrier K_cc
and the other pairs keep their Gordon-Kim values. Comparator: step 17, same seeds.

**Motive (measured, 2026-10-20 to 25; theory note 2026-10-18 §4c).**
- The runtime (Gordon-Kim) Pauli table fails against the lattice-embedded first-principles values at Au on O-top
  (Au⁻ 0.13 eV too weak, Au⁺ 0.31 eV) and for the O hole (0.18 eV per Mg contact); the sphere convention reproduces
  the embedded values within 0.05-0.10 eV for all six AuMgO one-carrier types.
- Frozen-weight counterfactual on step 17 (`probe_pauli_refit.py`): wet_u +470 → +659 (s42), +656 → +774 meV (s43)
  toward DFT's +929; wet_d +4 / +17; T5 −841 → −1026, −894 → −996 meV.
- The converged first-principles wetting decomposition (34 ions): the upright preference is O-top relaxation, not
  frozen contact physics, so this table, a frozen-contact quantity, fixes part of the error, not all of it.

**Benchmark moved:** the undoped wetting energy (B3) and T5 (4G Table II).

| id | criterion (each seed unless stated) |
|---|---|
| **R1** | T5 within ±15 % of DFT (−792 to −1071 meV) for both seeds |
| **R2** | wet_u closer to DFT (+928.9) than step 17's same seed by ≥ 100 meV (s42 +480.8, s43 +660.9) |
| R3 | test E / F RMSE no worse than step 17's same seed by > 10 % |
| R4 | B1; ε∞(MgO) 2.5–3.5; Au atom α within 10 %; no configuration-cap refusal |

**Reported:** wet_d (DFT −2.7), the doped near share (DFT 0.38), p_ion(upright), the learned O⁻ shift
(`probe_hole_shift.py`), B3's class offsets; the table's (v3) values at the six embedded contacts (filled in
below before the launch).

**The table, built and checked before the launch (2026-10-26; `build_pauli_fp.py`, 87 of 1030 types refitted).**
(v1) the reconstruction reproduces the Gordon-Kim series to 3.2e-16. One-carrier K (Ha bohr³, fit residual): O⁻/O²⁻
2.859 (1.3 %), O⁻/Mg²⁺ 4.021 (1.1 %), Mg⁺/O²⁻ 4.920 (6.2 %), O⁻/Al³⁺ 4.049 (1.0 %), Al²⁺/O²⁻ 7.916 (9.8 %),
Mg⁺/Mg²⁺ 4.145 (0.3 %), Mg⁺/Al³⁺ 4.200 (0.2 %), Al²⁺/Mg²⁺ 4.109 (2.8 %), O⁻/Au⁰ 1.228 (4.1 %), Au⁻/O²⁻ 4.241
(8.4 %), Au⁺/O²⁻ 5.528 (0.5 %), Mg⁺/Au⁰ 3.649 (13.7 %), Au⁻/Mg²⁺ 3.822 (0.8 %), Au⁺/Mg²⁺ 4.196 (1.0 %), Au⁻/Au⁰
1.137 (0.9 %), Au⁺/Au⁰ 2.493 (3.9 %). (v3) against the embedded values (eV; new / Gordon-Kim / embedded):
Au⁻/O²⁻ 2.26 Å +1.177 / +1.149 / +1.279 (**0.102 off: fails the 0.1 eV gate by 2 meV**); Au⁺/O²⁻ −1.362 / −1.049 /
−1.357; Au⁻/Mg²⁺ 2.90 Å +0.330 / +0.190 / +0.276; Au⁺/Mg²⁺ −0.146 / −0.075 / −0.121; Mg⁺/O²⁻ 2.106 Å +1.352 /
+1.227 / +1.303; O⁻/Mg²⁺ 2.1325 Å −0.685 / −0.425 / −0.601. The new table is closer than Gordon-Kim on all six.
**Deviations, recorded before the launch (decisions of 2026-10-26):** the Au-Au K is fitted from 3 of its 5
window points (2.331-2.563 A; the points took 2.5-4.8 h each); (v3) fails for Au⁻/O²⁻ by 2 meV and the run is launched
anyway, R1 and R2 deciding.

**Predictions (hypotheses).** R2 passes (the frozen counterfactual moved wet_u by +118 / +189 meV); wet_d stays wrong
(the counterfactual left it unchanged); R1 is open (frozen T5 rose to 107–110 % of DFT).

**Result (scored 2026-10-27 13:48; both seeds trained 40 epochs, best checkpoints at epochs 40 / 39).**

| id | s42 | s43 | verdict |
|---|---|---|---|
| R1 | T5 −989.4 meV (106.2 %) | −975.9 (104.8 %) | **PASS** (seed spread 13.5 meV) |
| R2 | wet_u +536.3 (step 17 +480.8: 55.5 closer) | +753.8 (+660.9: 92.9 closer) | **FAIL** (both closer, neither by 100) |
| R3 | E d/u 2.728 / 2.493 meV/atom (step 17 2.301 / 2.378: doped +18.6 %); F d/u 0.0886 / 0.0689 (0.0929 / 0.1547) | E 1.272 / 0.905 (1.341 / 1.287); F 0.1091 / 0.0726 (0.0900 / 0.0985: doped +21.2 %) | **FAIL** (s42 doped E, s43 doped F; the other six numbers improve, by up to 55 %) |
| R4 | B1 −1.000 e; ε∞ 3.371; Au α 5.433 Å³; no refusal | −1.005; 3.170; 5.437; none | **PASS** |

Reported: wet_d −453.1 / −222.1 (DFT −2.7); class offsets d/u −376.1 / −358.8 and −176.8 / −156.7; doped near share
0.55 / 0.51 (DFT 0.38); p_ion(upright) 0.271 / 0.228. Predictions: R2 "passes" was wrong (half the frozen
counterfactual's move); "wet_d stays wrong" was right; R1 passed. G_t5_stack.txt failed for both seeds: the probe's
stacked settings combine exclusion with carrier_overlap='gordon_kim', which the configuration refuses (a probe
incompatibility, not a model failure).

**What the refit did to the hole (measured).** The learned O-hole shift Δ_O falls from +2.2 eV (step 17) to a median
+0.47 eV (s42; `probe_hole_shift.py`): the shift had been compensating the Gordon-Kim Pauli term. The hole's overlap
term goes from −1.75 eV (step 16 s43) to −2.87 / −3.32 eV. s43's clean-slab IP is +4.887 eV against PBE's +4.796
(`probe_hole_terms.py`; step 16 s43: +7.21). The O → Au configuration at the O-top contact falls from 6.27 / 6.30 eV
(step 17) to 5.24 / 4.15 eV.

**A latent catastrophe in s42 (measured; the clean slab at charge +1).** IP = −267.7 eV: the hole's diagonal energy is
a sane +7.755 eV, but its coherent energy is −275.4 eV (s43 −2.49). The cause is s42's learned O–O coupling: 28.1 eV
(median, 3.0 Å), ten times Harrison's largest universal two-centre value there (ppσ 2.72 eV). s43 learned 0.37 eV and
step 17 s42 1.04 eV. Hole configurations never enter the neutral training ensembles, so no data constrains these
couplings: their value is a seed lottery. At charge 0 the amplitude selection never reaches the hole band (its first
hop from the frame, O → Au, has amplitude ~0.05). So s42's T5 is unaffected, but the exact ground state of its
Hamiltonian at charge 0 is not the selected one. SPEC 27's Harrison bound, built and tested, was off in steps 17–20.


**Reading.** R2 passes: the frozen-contact physics was part of the wetting error and first-principles tables are the
better prior. R2 fails: retraining absorbs the table change into other terms, and the remaining lever is the O-top
relaxation (the polarisation / charge-transfer split, running).

### Pre-registered: step 19 (branch `band-prior-2026-10-22`), the band unit's hopping anchored to PBE's band curvature (written 2026-10-22, before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_bprior_bondlaw_r5_s{42,43}` = step 17's
configuration (with its amendment, `state_polarizability_elements: [79, 8]`) and one change:
`redox.band_initial_factor: 0.55` (trace_redox SPEC 20; new field, default 1 = Harrison, unchanged for every other
run). Comparator: step 17, same seeds. Step 18 (step 17 + the field prior) runs beside it and is not its comparator.

**Motive (measured, 2026-10-20 to 22).**
- The step 15 refusals: on the dumped call the band electron sits at λ₁ = −6.60 eV below a vacuum electron, of
  which −10.80 eV is hopping at f = 1 (`probe_dump_terms.py`); the window holds 1338 configurations (Au⁺ + a band
  electron at −0.04 eV, O⁻ + band electron at +0.64 eV, against MgO's ~4.2–4.3 eV PBE gap).
- PBE bulk MgO at the 4G lattice constant (`tools/electrostatics/mgo_pbe.py`; cutoff, k-mesh checks pass): the
  conduction-band mass m* = 0.322 / 0.330 mₑ (dzvp / tzv2p). The band unit's nearest-neighbour s band has
  m* = 0.179/f mₑ, so f = 0.55. Harrison's f = 1 gives 0.179 mₑ (1.8× too light); step 16 s43 learned f = 0.677.
- The microscopic derivation (Pauli-projected two-site coupling, `band_hopping.py`) failed its pre-registered test
  (1.107 eV against 0.635–0.651), so the prior is the measured curvature, not the derivation.
- Same dumped call with f = 0.55, at the original cap 512: no refusal; 3 configurations, frame population 1.000;
  λ₁ = −1.742 eV (hopping −5.93 eV).
- Physics of the choice: the band unit carries the low-energy physics of an extra electron near a perturbation
  (dopant, Au₂, surface), which is effective-mass physics (hydrogenic donor binding ∝ m*/ε², radius ∝ ε/m*;
  ESTABLISHED); the band edge belongs to the site energies and their learned shifts.

**Benchmark moved:** robustness of the AuMgO training (the configuration cap), T5 (4G Table II) both seeds, the
test E / F RMSE (step 17's S4 failed), B3.

| id | criterion (each seed unless stated) |
|---|---|
| **Q1** | both seeds train to the end with no configuration-cap refusal |
| **Q2** | T5 within ±15 % of DFT (−792 to −1071 meV) for both seeds |
| Q3 | test E / F RMSE no worse than step 17's same seed by > 10 % (s42: 2.301 / 2.378 meV/atom, 0.0929 / 0.1547 eV/Å; s43: 1.341 / 1.287, 0.0900 / 0.0985) |
| Q4 | B3: undoped \|class offset\| ≤ 100 meV (failed in every step since 13) |
| Q5 | B1 (doped Au₂ holds one electron); ε∞(MgO) 2.5–3.5; Au atom α within 10 % |

**Reported:** the final learned f (against 0.55 and step 16's 0.677) and the model's m* = 0.179/f; λ₁ and the
clean-slab EA / IP / gap (`probe_spectrum.py`) against the PBE slab when it is in; the near share (DFT 0.38);
cost per epoch.

**Predictions (hypotheses).**
- Q1 passes (the dumped call already does).
- If the data prefer a lighter band than PBE's (step 16 drifted to 0.677 from 1), f drifts up from 0.55; if it
  stays near 0.55, the curvature anchor and the data agree.
- Q3 and Q4 are open: the hypothesis is that learned shifts compensating the over-wide band were part of the
  energy and offset errors.

**Reading.**
- Q1–Q3 pass: the physically anchored band is the better prior and stays.
- Q1 passes, Q3 fails: the anchor fixes robustness but the data's energies prefer a lighter band, so the band
  unit's single-s-band form, not its parameter, is the limit.

**Measured: step 19, trained and scored (2026-10-23 15:10; `read_criteria.py 19`).**

| id | seed 42 | seed 43 | verdict |
|---|---|---|---|
| **Q1** no refusal, to the end | epoch 40, 0 errors (3 NaN-guard transients) | epoch 39, 0 errors (2) | **PASS** |
| **Q2** T5 within ±15 % | −703.8 meV (75.5 %) | −611.2 meV (65.6 %) | **FAIL** (step 17: 90.5 / 97.0 %) |
| Q3 E / F RMSE vs step 17 (+10 %) | E 1.647 / 2.601, F 0.0867 / 0.0666 | E 1.165 / 2.735, F 0.0910 / 0.1263 | s42 pass; **s43 fails** (undoped E, F) |
| Q4 B3 undoped offset ≤ 100 meV | −542.4 | −530.8 | **FAIL** (worse than step 17) |
| Q5 B1 / ε∞ / α(Au) | −1.000 / **3.658** / 5.435 | −1.001 / 3.122 / 5.423 | s42 fails ε∞ |

- Reported: final f = 0.485 / 0.572 (m* 0.369 / 0.313 mₑ; PBE 0.322–0.330); wet_u +312.2 / +346.8 meV (step 17
  +480.8 / +660.9; DFT +928.9); doped upright near share 0.60 / 0.14 (DFT 0.38); p_ion(upright) 0.212 / 0.037;
  seed spread 92.6 meV. Learned O⁻ shift (probe_hole_shift, mid-training): +0.14–0.20 / +0.98–1.01 eV against
  ~+2 eV in steps 16–17.
- Reading, as pre-registered: Q1 passes and Q3 is split, so the anchor fixes robustness while the single-s-band
  form, not its parameter, limits the energies; PBE confirms it (band averages: CB mean − bottom 4.909 eV gives
  f = 0.35 for a nearest-neighbour s band, the Γ curvature gives 0.55; one s band cannot fit both). Steps 16–17's
  better T5 came partly from compensations the over-wide band allowed (the ~2 eV O⁻ shift). The band prior is not
  adopted on its own for T5; the measured wetting lever is the Pauli refit (frozen-weight counterfactual, theory note
  2026-10-18 §4c).

### Pre-registered: step 18, the charged-dimer field prior on the full stack (written 2026-10-19 ~14:00, before the run)

**Suspended 2026-10-22 14:31 (by decision), to resume after step 19 is scored.** The 16 GB Mac was over-committed
(about 22 GB of demand, 9.5 GB of swap), and seed 42 logged four NaN-guard transients that morning. Stopped at epochs
9 (s42) and 10 (s43); it resumes from its best checkpoints (epochs 7 and 8, with optimizer state) through
`resume_run.py` (tag `suspend1`; the loader replays the epochs, so the batch order is the uninterrupted run's), by
`watch_step18_resume.sh`. Its criteria and comparator are unchanged; nothing deleted.
- **Resumed 2026-10-23 15:14; seed 42 suspended a second time 2026-10-24 13:10 (by decision)**, at epoch 32
  (best checkpoint 31), because memory again starved the first-principles wetting run (8.3 GB swap). Seed 43 was in
  its final epoch (40) and finishes on its own. `watch_step18_resume2.sh` resumes seed 42 (tag `suspend2`) when the
  34-ion wetting run ends, scores both seeds, then starts the doped wetting decomposition.
- **Measured: step 18 seed 42 (scored 2026-10-25): T5 +98.3 meV (wrong sign), wet d/u +22.0 / −76.3 meV, test E
  RMSE d/u 2.138 / 4.457 meV/atom, F 0.207 / 0.187 eV/Å; G1 and G5 fail.** Seed 43's scoring is pending
  (`score_step18_rest.sh`). The suspensions are ruled out as the cause: the loss is continuous across each resume
  (s42 epoch 8: 0.0599 on both sides; epoch 31 → 32: 0.0270 → 0.0264; s43 epoch 9: 0.0749 / 0.0748), the damage
  precedes any suspension (force RMSE at epoch 8: 0.242 (s42), 0.268 (s43, epoch 9) against step 17's 0.133), and both
  seeds end alike (0.158 / 0.170 against step 17's 0.069). So the charged-dimer field prior at weight 0.3 competes
  with the energy and force fit from the start.
- **Step 18 scored in full (2026-10-25):** G1 fails (T5 +98.3 / −4.6 meV), G3 fails (scorecard RMSE 17.39 / 17.40 Å³
  against ≤ ~2.6), G5 fails (E d/u 2.14 / 4.46 and 2.51 / 5.43 meV/atom; F 0.207 / 0.187 and 0.225 / 0.174 eV/Å);
  G4 passes only formally (isolated Au₂ ionic gap +0.27 eV at 2.5 Å, but τ = 0.000 on both seeds). Reported: wet_u
  −76.3 / −310.4 meV, p_ion(upright) 0.001 / 0.000, doped near share 0.74 / 1.00 (DFT 0.38). Reading: the prior was
  to pin τ through the cation's hole resonance; instead τ collapsed to zero on both seeds, so the dimer has no
  charge-transfer mixing, cannot match the tensors it was trained on, and loses the upright ionic component.
  Unexplained: the explicit-gap probe returns NaN with the learned shifts (hypothesis: the 0/0 of tanh(τ/kT)/τ at
  τ = 0; not checked). Step 18 is not adopted.

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_fpriorq_bondlaw_r5_s{42,43}` = step 17's
configuration (with its amendment, `state_polarizability_elements: [79, 8]`) and one change: the field prior on the
nine first-principles tensors of Au₂⁰ (2.30–2.80 Å), Au₂⁺ (2.47–2.80) and Au₂⁻ (2.47, 2.60),
`data/Au2_charges_alpha_field_prior.xyz`, weight 0.3 (13FC's file and weight). It launches when step 17 has been
trained and scored (`watch_step18.sh`). Comparator: step 17, same seeds.

**Motive (measured 2026-10-19).**
- The free sector has its own (gap, τ) degeneracy: energies fix ½(gap − √(gap² + 8τ²)), not the ionic weight
  (isolated Au₂ of step 13: gap +2.97 / −1.00 eV, τ 2.00 / 0.65 eV, bond energies −1.71 / −1.54 eV).
- At two sites the cation's hole resonance fixes τ alone (α_CT = μ_eff² tanh(τ/kT)/(4τ)), and then the neutral's
  α∥ fixes the gap.
- 13FC, which applied this prior without SPEC 22/28, was disarmed as confounded: α⊥ of the ions is bound-only and
  needs α_b(q). With SPEC 22–28 on frozen step-13 weights, α⊥ follows the charge (scorecard RMSE 8.59 → 4.18 Å³,
  seed 42).

**Benchmark moved:** T5 (4G Table II) for both seeds and its seed spread, through the upright dimer's ionic weight.

| id | criterion (each seed unless stated) |
|---|---|
| **G1** | T5 within ±15 % of DFT (−792 to −1071 meV) for **both** seeds |
| G2 | \|T5(42) − T5(43)\| below step 17's |
| G3 | scorecard over the nine tensors: RMSE ≤ 15 % of the mean target (≈ 2.6 Å³), no component off by > 30 % |
| G4 | isolated Au₂ at 2.5 Å: ionic gap > 0 (covalent ground configuration) for both seeds, and τ within 25 % of each other |
| G5 | test E / F RMSE no worse than step 17 (same seed) by > 10 %; B1, B3, A1 as for step 13 |

**Reported:** the upright undoped dimer's p(Au⁺Au⁻) at the T5 references; τ and the gap of isolated Au₂ (2.3–3.4 Å,
`probe_explicit_gap.py`, learned and zero shifts); the stiffness-defect count (`probe_stiffness_defects.py`).

**Predictions (hypotheses, written before the run).**
- G4 passes: the cation resonance pins τ, and the stack's explicit gap is already positive.
- G2 passes if the remaining seed spread of step 17 is the (gap, τ) degeneracy; if step 17's spread is already
  small, G2 tests nothing new and is reported as such.
- G1 is the open question. If G3 and G4 pass and G1 fails, the T5 deficit lies in the doped/undoped contact physics,
  not in the dimer's character.

### Pre-registered: step 17, each charge state's own polarisability (written 2026-10-18 19:20, before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_bondlaw_r5_s{42,43}` = step 16's configuration
with one change, `redox.state_polarizability: true` (trace_redox SPEC 28). It launches when step 16 has been trained
and scored (`watch_step17.sh`). Comparator: step 16, same seeds.

**Motive (measured):** the section "State-dependent polarisabilities, measured" above.
- ρ(Au⁻) = 3.39 and ρ(Au⁺) = 0.37 from first principles.
- Frozen step-12 weights: δE of 0.1–0.9 eV on the Au⁻ configurations. T5 +1 meV (seed 42) and −298 meV
  (seed 43).

**Benchmark moved:** T5 (4G Table II) for both seeds, and its seed spread.

| id | criterion (each seed unless stated) |
|---|---|
| **S1** | T5 within ±15 % of DFT (−792 to −1071 meV) for **both** seeds |
| S2 | \|T5(42) − T5(43)\| below step 16's |
| S3 | undoped non-wetting Au₂: near > far; ~~\|(q_near − q_far)/2 − 0.120\| no worse than step 16's by > 0.02 e~~ **amended 19:35: the magnitude is reported** |
| S4 | test E / F RMSE no worse than step 12 seed 42 by > 10 % (0.847 / 0.703 meV/atom; 0.1052 / 0.0764 eV/Å) |
| S5 | B1, B3, A1 as before; ε∞(MgO) 2.5–3.5; Au atom α within 10 % (the frame α is unchanged by construction) |

**Reported:**
- the δE distribution on the test set;
- the learned Au⁻/Au⁺ shifts against step 16's: they should stop compensating the missing anion polarisability;
- cost per epoch.

**Predictions (hypotheses).**
- The frozen-weight counterfactual moves the seeds toward each other (seed 43 −298 meV, seed 42 +1 meV).
- Retraining absorbs part of this into the shifts, so S2 is the sharper test.

**Reading.**
- S1 and S2 pass: the state-independent α was part of the seed dependence.
- S2 fails: the term is right physics (measured) but not the seed-dependence lever.
- S4 fails: the free-ion upper bound overstates the in-situ anion α (Fowler–Madden confinement). The next step
  would be a confined-ion ratio.

**Amendment to step 17 (2026-10-19 ~11:40, before its run): `state_polarizability_elements: [79, 8]`.**
- Why, measured on frozen weights (`probe_stiffness_constructed.py`, `probe_stiffness_defects.py`; outputs in
  `$TRACE_FP_CACHE/probes/`):
  - With the frame's damping width, the free-ion ratios of reduced cations make the configuration's stiffness
    K(n) indefinite, a polarisation catastrophe (trace_redox SPEC 28, "K(n) must stay positive definite").
    - Trained bulk MgO (ε∞ prior structure), step 14: 16 of 17 constructed Mg⁺O⁻ / Mg⁰ configurations.
      Brute-force spectrum: min eig K(n) −0.362 (Mg⁺O⁻) and −0.403 (Mg⁰), against +2.42 for K.
    - Doped slab: 21 of 22 constructed Al²⁺O⁻ / Al⁺ configurations. The same holds on step 13 and 13F weights.
  - Such a configuration is now refused (ValueError), where before 2026-10-19 it silently got a saddle's energy.
    Unrestricted, step 17 would therefore crash if training selected a reduced cation, as 13F's MgO reached
    5 configurations.
- Effect on the pre-registered comparison: none measured.
  - On step 14's weights, the selected configurations of the 4 T5 references, bulk MgO and 40 seeded test
    structures are identical with and without the restriction: 113 configurations, 88 with a softened atom (all
    Au⁻), 0 defects.
  - Au⁻ (3.39) is outside SPEC 12's guarantee (ρ < 2.63) but positive definite in all 136 Au⁻ configurations
    checked on both seeds. Au⁺ and O⁻ stiffen, so they are always positive definite.
- The motive (Au⁻ 3.39, Au⁺ 0.37, O⁻ 0.44) is unchanged. Mg and Al keep their frame α; free-ion ratios for
  reduced cations confined by O²⁻ are not physical anyway (Fowler–Madden). S1–S4 are unchanged.

**Measured: step 17, trained and scored (2026-10-21 20:22; `read_criteria.py 17`).** Both seeds trained to the end
(epochs 40 / 39), with no configuration-cap refusal (step 15 refused at call 2 on both seeds, step 16 seed 42 at
epoch 4).

| id | seed 42 | seed 43 | verdict |
|---|---|---|---|
| **S1** T5 within ±15 % of DFT | −843.5 meV (90.5 %) | −904.1 meV (97.0 %) | **PASS** |
| S2 spread below step 16's | 60.6 meV | | not evaluable: step 16 has one seed (42 refused) |
| S4 test E / F RMSE (limits 0.932 / 0.773 meV/atom; 0.1157 / 0.0840 eV/Å) | E 2.301 / 2.378, F 0.0929 / 0.1547 | E 1.341 / 1.287, F 0.0900 / 0.0985 | **FAIL** (both seeds) |
| S5 B1 / B3 / ε∞ / α(Au) | −1.000 / offset −397 meV / 2.825 / 5.427 Å³ | −1.000 / −257 meV / 3.067 / 5.433 Å³ | B1, ε∞, α pass; **B3 fails** |

- Reported: doped upright near share 0.46 / 0.42 (DFT 0.38); p_ion(upright) 0.347 / 0.235; Δ_ion +0.231 / +0.597.
- Reading, as pre-registered: S1 passes, S4 fails, so by the step's own reading the free-ion state polarisability
  overstates the in-situ anion α (Fowler–Madden confinement); the next step would be a confined-ion ratio. T5 is
  not claimed solved while the energies are 1.5–2.7× worse than step 12 and B3 fails.

### Pre-registered: step 10, relaxed exclusion (written 2026-10-15 21:05, before step 9 is scored and before the run)

**Run.** `AuMgO_formal_kT_eps_cpl_alpha_exr_bondlaw_r5_s42` = step 9's configuration with one change:
`redox.exclusion_relaxation: true` (trace_redox SPEC 18.4).
- Each charge state's frontier density relaxes as a rigid shell held by the unit's own α_u, to second
  order, closed form. This is no new parameter.
- **Motive (measured):**
  - the frozen exclusion overstates both costs and reliefs: for Au⁰ atop O²⁻ at 2.2 Å, 0.50 of 1.35 eV
    relaxes;
  - on step 6's weights it overshoots the undoped class offset (−314.5 → +1422 meV), and the
    relaxation removes 37.5 % of that shift.
- **Launch:** automatically once step 8 has finished and been scored (memory: 16 GB, two runs at
  2.6 GB each plus compression). Cost: +0.35 s per structure over step 9 (measured).

**Criteria** (comparators step 9 and step 6, the same scripts):

| id | criterion |
|---|---|
| R1 | undoped Au₂ less ionic than step 9: undoped p(Au⁰) (`probe_couplings.py`) above step 9's |
| R2 | undoped class offset \|flat − upright residual\| (`probe_class_residuals.py`) below both step 9's and step 6's 315 meV |
| **F5′** | T5 wetting difference ≤ −537 meV (no worse than step 6), and closer to DFT −931.6 than step 9's |
| X1 | doped-upright electron on the far Au (free charge far < near) |
| F2–F4 | transfer present (free charge on Au₂ < −0.10 e) |
| F1 | test E / F RMSE no worse than step 6 by > 10 % |
| E1, A1 | ε∞ 2.5–3.5; Au α within 10 % |

**Reported:** learned K and I_eff, the vertical-force and residual-geometry probes, structure 87, and
the per-atom r.

**Predictions (hypotheses).**
- R1 passes if step 9 over-ionises.
- F5′ and R2 are open. The relaxation also lowers the Au⁰ frame's own exclusion, which the ACE absorbs.
  What remains is the state dependence, and its trained size is unknown.


## Validation outside the benchmark: phonons and Born charges of bulk MgO (2026-11-06; not a step)

`python tools/electrostatics/mgo_phonons.py runs/RUN/model.pt --cell 2` — Z*, ε∞, Γ-point phonons without and with the
non-analytic correction from the model's own Z* and ε∞ (phonopy), LST ε0, and the backbone alone. References: experiment
TO 401 / LO 718 cm⁻¹, ε∞ 2.95, ε0 9.8; PBE Z* 1.96–1.99. Step 20: Z* +2.429 / +2.536 (s42 / s43), ε∞ 3.009 / 3.048,
TO 389.8 / 414.3, LO 817.0 / 852.2 (splitting 427 / 438 against 317), ε0 13.2 / 12.9; backbone alone TO 504 / 587 and no
splitting; mean-field SQE: Z* 0.267, ε∞ 1170, no splitting. The whole splitting is the electrostatic term's; ε∞ is right and
Z* is 22–27 % too large because the model has no dynamic covalent transfer (theory note §4e; CHANGELOG Round 33). Proposed
criteria for any step that touches it (pre-registered 2026-11-06): V1 Z*(Mg) within 1.90–2.10; V2 LO within 690–740 cm⁻¹.
Against Togo's phonondb DFT (mp-1265, PBEsol, DFPT NAC; `phonons_mgo/compare_phonondb.py`, `band_table.py`): path RMS
28.5 cm⁻¹ (s42, a = 4.265; 14.3 without NAC on both sides), 41.9 (s43), 56.4 (mean-field). Seed 42's bulk ensemble leaves
the frame under 0.5 % compression (a ≤ 4.245 Å: F − E_frame −27 eV, −366 at the DFT volume) — the scalar O–O hole coupling
switched on by SPEC 21; its a = 4.217 phonons are an artifact (theory note §4e; CHANGELOG Round 33).
Functional-matched (Togo's phono3py PBE set, a = 4.2557, PBE Z* 1.992 / ε∞ 3.235; `phonons_mgo/compare_phono3py_pbe.py`):
PBE TO 373.3 / LO 681.3; model s42 (at 4.265) RMS 30.3, TO +4 %, LO +20 %; s43 (at 4.2557) RMS 48.5, TO +9 %. s42 at
4.2557 is at its cliff (frame population 0.976) with contaminated force constants (acoustic RMS 781).

## Pre-registered: step 28, the frame-pair overlap damping of induction (SPEC 34; written 2026-11-07 08:15, before the run)

Runs `runs/AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_fid_bondlaw_r5_s{42,43}`: step 20's config
with every path replaced (`check_run_config.py` OK) and one change, `frame_induction_damping: true` (SPEC 34: the frame
charge's drive on its nearest-neighbour counter-ions' bound dipoles damped by s(R) = 0.61 + 0.37 (R − 2.128 Å), the law
measured on the embedded Mg²⁺–O²⁻ pair; `trace_redox/frame_damping.py`, tests `tests/redox/test_redox_frame_damping.py`).
Cold start, 40 epochs, both seeds, `watch_step28.sh`; comparator **step 20**. Launch: on the Mac once step 25's two runs
finish (the GPU holds the two bulk-augmented seeds).

**Why:** the phonon line (benchmarks_phonons) measured step 20's bulk-MgO Born charge at 2.43 against PBE 1.99 at the
right ε∞, and the frozen term brings it to 1.95 with ε∞ unchanged (P1 pass on the frozen flagged s43: 1.965 at
ε∞ 3.052). The 4G tests never see Z*; the question for step 28 is whether the term costs anything there.

**Pass criteria (pre-registered):**
- **T1–T6 not worse than step 20** by more than the seed spread (the term touches the frame-driven induction of every
  O near Au, so the wetting split and the ion charges may move): in particular T6 ε∞ within 2.5–3.5 and T5's wet_u / wet_d
  within 50 meV of step 20's, or better.
- **Phonon line: P1 Z*(Mg) 1.90–2.10 at the PBE lattice constant** (the term's purpose), P0 on both seeds, P2 as before;
  P3 / P5 (the backbone's force constants, which the frozen term moved from +9 % to +21 % on TO) **must return to within
  step 20's values** (TO within 10 % of PBE, path RMS without NAC ≤ 45 cm⁻¹) — the backbone has to re-absorb the term.
- The frozen flagged s42 is scored on T1–T6 first (`runs/AuMgO_frozen_step20_fid_s42`, running 08:00): its deltas are
  the "before"; a frozen T5 move beyond 300 meV is a flag to look at the term's static energy before launching.
- **Frozen "before" (08:30, partial):** T6 ε∞ 3.371 unchanged (as designed). T4: the undoped slab's free energy falls by
  8.0 eV (E_free −913.4 → −921.4) and its ensemble puts −0.167 e on Au₂ (step 20: +0.0001): the term's linear part
  −(A K⁻¹ v)·n shifts the surface O's charge-state energies by eV-scale amounts on the frozen ladder, enough to populate
  an O⁻ + Au₂⁻ configuration at 1/6 in the undoped cell. Frozen, that is a ground-state change, not a small delta; it is
  what the cold start's ladder has to re-balance, and T1–T4 of step 28 are the test of whether it can. (The bulk Born
  charge is unaffected: there the term vanishes at the perfect crystal and acts through its derivative only.)
- **Frozen "before", complete (02:19 scoring; read 09:10):** T1 E RMSE 106 / 73 meV/atom (undoped / doped; step 20
  2.49 / 2.73) and **F RMSE 88 / 0.71 eV/Å** (step 20 0.069 / 0.089); T2 shift −0.961 e (−1.000); T5 wet_u +648 meV
  (+536), wet_d −458 (−453); T6 3.371 (unchanged). An 88 eV/Å force error on the undoped slabs is not an energy
  offset: it is the signature of the switched hole couplings opening (the 28 eV/edge scalar O–O coupling; the
  lattice-constant cliff of §5g) once the term lowers the surface O⁻ states on the frozen ladder. Being diagnosed on
  three undoped test structures (ensemble population, couplings, forces) before step 28 launches; if confirmed, step
  28 is amended before launch (the O–O hole coupling defect must be closed first, or bounded).
- **Diagnosis (09:25, a one-off script not kept in the repository, three undoped test structures, frozen s42):** flag off → on:
  configurations 3 → 72, 4 → 27, 3 → 67; lowest gap 2.08 → 0.81, 0.98 → 0.75, 0.86 → 0.66 eV; E_free − E_frame −1.9 →
  −9.0, −2.1 → −9.3, −1.8 → −34.8 eV; structure 2 collapses (frame population 0.0000, q(Au₂) −1.000, max |F| 455 eV/Å;
  the others 0.96 → 2.97 and 3.46 → 4.00 eV/Å). The term's linear part −(A K⁻¹ v)·n lowers the surface O⁻ states by
  ~1.3 eV on the frozen ladder — amplified by the inverted polarizability split (the frame-driven dipoles the damping
  weakens sit on Mg, α_Mg 1.9 Å³, and their potential at the O is what held the O²⁻) — and the switched scalar O–O hole
  couplings finish the collapse. **Step 28 is NOT launched as pre-registered** (the armed auto-launch is withdrawn);
  it is re-pre-registered once the physical split is tested with the term (running) and the O–O coupling defect is
  bounded or closed.
- **With the physical polarizability split imposed (09:40; α_O 2.44 / α_Mg 0.07 Å³, global ×0.92), the term is benign on
  the same three structures:** configurations 3 → 3, 3 → 4, 3 → 4; gaps 2.18 → 2.05, 0.99 → 1.24, 0.85 → 1.02 eV; frame
  populations 0.69 → 0.68, 0.64 → 0.65, 0.63 → 0.64; max |F| 1.5 → 2.1, 3.4 → 3.4, 6.1 → 5.9 eV/Å; q(Au₂) 0.000 throughout
  (E_free − E_frame −1.8 → −7.6 eV is the term's constant, added outside H). So the collapse was the inverted split's
  amplification of the term, not the term: with the polarizability on O²⁻ the damping weakens the O's own dipole, which
  does not act on the O's charge. **Step 28 therefore pairs the damping with a frame-polarizability anchor** (SPEC 35, to
  be built test-first: the frame ions' dipole stiffness A_μ = K_E ℓ²/(g α_Z) with the first-principles α_Z of
  `ion_polarizabilities.json` and one learned crystal factor g), as steps 28a (anchor alone, the defect fix; frozen it
  raises Z* to 2.6) and 28b (anchor + damping; frozen Z* 1.93 at ε∞ 3.235), both against step 20.


## Pre-registered: steps 28a and 28b (SPEC 35 anchor; SPEC 35 + SPEC 34 damping; written 2026-11-07 10:20, before the runs)

Supersedes the single "step 28" above (its `_fid_` configs stay as the record of the withdrawn launch). Runs
`runs/AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_fpa_bondlaw_r5_s{42,43}` (**28a**: step 20 +
`frame_polarizability_anchor`, SPEC 35 — the frame ions' dipole stiffness from the first-principles free-ion
polarizabilities with one learned crystal factor g, init 1) and `..._pfp_fpa_fid_bondlaw_r5_s{42,43}` (**28b**: 28a +
`frame_induction_damping`, SPEC 34). Every path replaced (`check_run_config.py` OK); cold start, 40 epochs, both seeds;
`watch_step28ab.sh` (four runs on the Mac, launched when step 25's two runs end); comparators: 28a → step 20, 28b → 28a.

**Why two runs:** the anchor alone is a defect fix (the trained split α_O 0.2 / α_Mg 1.9 Å³ is inverted; frozen it
raises Z* to 2.6 and must not change T1–T6 beyond the seed spread); the damping is the Z* physics and is only
well-defined on top of the physical split (with the trained split it collapsed the surface ensemble, |F| 455 eV/Å; with
the physical split it is benign on the same slabs: configurations 3 → 3/4, gaps and forces unchanged).

**Pass criteria.**
- 28a: T1–T6 within the seed spread of step 20; T6 ε∞ 2.5–3.5 (the crystal factor g has to find the prior's ε∞ with the
  polarizability on O: g ≈ 0.9 frozen); phonon line P2 pass, P1 expected to stay failed (frozen 2.55 at ε∞ 3.235).
- 28b: T1–T6 within the seed spread of 28a (no regression from the damping); **P1 Z*(Mg) 1.90–2.10** at the PBE lattice
  constant (frozen: 1.93); P0 on both seeds; P3/P5 back within step 20's (TO within 10 % of PBE, path RMS without NAC
  ≤ 45 cm⁻¹) — the backbone must re-absorb the term's static energy (frozen TO 407 → 450 on s43).
- Both: no ConfigurationSearchError / collapse in training (the guard of the phonon scorer on every displaced supercell;
  the training log's nanguard).
- **Frozen through the real module (10:40, s42, three undoped test slabs):** anchor alone at g = 0.92 (ε∞ 3.227):
  configurations 3/3/4, frame populations 0.68/0.60/0.59, gaps 2.01/0.56/0.42 eV (as trained 2.08/0.98/0.86: the
  O-centred polarizability lowers the surface hole states by ~0.4 eV), max |F| 2.0/3.3/6.1 eV/Å, q(Au₂) 0; anchor +
  damping: 3/4/3, 0.67/0.60/0.59, 1.89/0.65/0.47 eV, 1.6/3.2/6.1 eV/Å, q(Au₂) 0 — no collapse, forces unchanged. At
  g = 1 the table split gives ε∞ 3.45, at g = 0.92 3.23 (the ε∞ prior will set g).
- **Bulk MgO through the real module (10:55, s42, a = 4.265):** anchor alone Z*(Mg) 2.554 at ε∞ 3.453 (g = 1), 2.555
  at 3.227 (g = 0.92); **anchor + damping 1.950 at ε∞ 3.227** (PBE 1.992 / 3.235); at g = 0.75 1.996 at 2.749. The
  pre-registered combined result (1.93) is reproduced by the implementation; 28b's P1 target stands.

**Mac scheduling, 2026-11-07 07:56 (decision of 2026-11-07: train on the Mac one run at a time).** The six Mac trainings run one at a time:
`mac_queue.sh` holds the queued runs by SIGSTOP and releases them in order by SIGCONT — step 26 s42 (running), step 26
s43, 28a s42, 28b s42, 28a s43, 28b s43 (log `runs/mac_queue.log`). Each run keeps the 4 intra-op threads it started with.
Pauses show up as gaps in the logs' wall-clock stamps; no state is lost (the same mechanism held step 26 for a day on
2026-11-06 and its ε∞ prior then converged). The watchers still score each step when all its runs have exited.

**07-11 08:35: step 28b moves to GPU host 1** when the bulk-augmented runs there finish (`gpu_queue_28b.sh`, log
`runs/gpu_queue_28b.log`; the paused Mac copies are stopped at the GPU launch, logs kept; the Mac queue skips
them). 28a stays in the Mac queue. Same configs and pre-registration; only the machine changes.

**Step 26 scored (2026-11-07 22:44 IST; SPEC 33 law + wall; comparator step 25).** Seed 42: undoped / doped E 2.30 / 2.65
meV/atom, F 0.062 / 0.085 eV/Å (step 20: 2.49 / 2.73, 0.069 / 0.089 — the best slab forces of any step); the relaxed T5
protocol **crashed** (`cholesky_inverse` singular in `induction.dipole_block_inverse` during a wetting relaxation: the
bound operator lost positive definiteness at some relaxed geometry — a model defect to diagnose); at the reference
geometries wet_u 612.5, wet_d −422.0, T5 −1034.5 meV (111 %); tilt path B2 PASS (barrier 0.298, end −0.073), flat dimer
B1 PASS (0.14 Å), upright approach C4 PASS, class offsets B3 FAIL (−55 meV). Seed 43: E 1.40 / 2.27, F 0.077 / 0.106;
wet_u 810.6 (DFT 928.9), wet_d −198.2 (DFT −2.7; E2 FAIL against step 25's −79.3), **T5 −1008.8 meV (108.3 %, E4 PASS)**,
flat dimer 0.28 Å (B1 PASS), tilt path end +0.126 (B2 FAIL), C4 PASS, B3 FAIL, E6 FAIL (doped E 2.27 against step 25's
0.59). G_t5_stack diagnostics failed on both seeds (as on step 25).

### Why the doped flat dimer is too stable (diagnosis, 2026-11-08 ~17:00 IST; decision of 2026-11-08: hold step 27 and find the cause)

Step 27 is held (built, pre-registered, not launched). Measured, from the score files of every scored Au₂-MgO run:

1. **The doped wetting error is set in the training data.** Across 36 runs with both scores, wet_d − DFT follows the
   test set's doped class offset (model − DFT residual, flat minus upright, at the DFT geometries): Pearson r = +0.90,
   slope 0.89, intercept −73 meV, residual spread 132 meV of 303. The undoped wet_u error does not (r = +0.17; spread 531
   of 539): it is set outside the data (the relaxed protocol), not by the class offset.
2. **The class offsets are not fitted, and they are seed-dependent.** Step 26's training-set offsets: doped −339 (s42)
   / −179 (s43), undoped −284 / −97 meV; step 25's: doped +1246 / −46, undoped +329 / +152. The loss leaves them free:
   `train.py` adds 10 ((E − E_DFT)/N)² to the force MSE; at step 26's last epoch (E 1.31 meV/atom, F MSE 0.00631) the
   energy term is 1.7e-5, 0.3 % of the loss, and removing a 180 meV offset on 110 atoms would lower the loss by < 0.5 %.
   (4G-HDNNP: E RMSE 0.219 meV/atom on this set.)
3. **The offsets appeared with the ensemble terms, not with the loss.** With the same energy weight, the `exband` runs had
   doped / undoped test offsets +22 / −34 (s42) and −20 / −10 (s43); they grew to hundreds of meV with amplitude
   selection and the bond partition (SPEC 21 / 22: `exband_amp` −37 / +362, `exband_amp_part` −277 / −565 at s43) and
   stayed there. From `part` on, the electrostatic sector alone gives wet_d −485 to −671 meV at the relaxed references
   (step 26: screened −745, shifts −393, coherent +474, overlap +197), against a backbone share of ≈ +280 that is the same
   for both dopings (it cannot see the dopants).
4. **A doping-specific remainder.** Doped − undoped offset: step 26 −64 (s42) / −92 (s43), same sign both seeds; step 20
   −17 / −20. No local term can absorb it (the Al are > 10 Å away); T5 sits at 108–111 % of DFT.

**Reading (hypotheses, not yet falsified).** (a) The common part (−97 to −284 meV, seed-dependent) is a flat-versus-
upright energy the trainable local parts could correct but the 0.3 % energy term does not make them; (b) the doping-
specific part (−64 to −92 meV) is the model's non-local electron-transfer energetics favouring a flat Au₂⁻, a candidate
mechanism being the integer electron's screened Coulomb binding near the cations (the screened term; TRACE moves 1.00 e
where Hirshfeld moves 0.27 — different partitions, so not by itself evidence).

**Falsifying checks proposed (not run).** (A) One change: step 26 s43 warm-started with the energy weight raised so
the energy term is comparable to the force term; prediction under (a): the training-set class offsets fall toward zero,
wet_d and wet_u both move toward DFT, the doped − undoped remainder stays at −60 to −90. If the offsets do not fall, (a)
is wrong and the terms of point 3 cannot represent DFT's flat-upright energies. (B) Frozen: which configurations are
selected in the doped flat and upright references (the electron on the contact or the far Au; over which site) and their
screened Coulomb energies, against the undoped ones.

## Pre-registered: step 29, the energy-weight test (written 2026-11-08 17:02 IST, before the runs; decision of 2026-11-08: run the energy-weight test on the Mac)

**Question.** Is step 26's flat-versus-upright class offset (and with it wet_d, r = 0.90 across 36 runs) a fit the
model can make but the loss does not ask for (the energy term is 0.3 % of step 26's loss), or one its terms cannot make?

**Runs** (seed 43; on the Mac, the two concurrently, `nice 5`):
- **test** `..._pfp_bopw_ew1000_bondlaw_r5_s43`: `energy_weight: 1000` (from 10). At step 26's final state a 180 meV
  offset on 110 atoms then costs 1000 × (1.64e-3)² = 2.7e-3 in the loss, 43 % of the force term (6.3e-3).
- **control** `..._pfp_bopw_ew10_bondlaw_r5_s43`: `energy_weight: 10`, otherwise identical — it separates the weight from
  "ten more epochs at a small learning rate".
- Both: step 26 s43's final weights (`warm_from_step26_s43.pt`: weights only, epoch 0, no optimizer state), 10 epochs,
  lr 1e-4 cosine to 5e-6 (t_max 10; the warm-start learning rate of steps 22–24), the training-time kT anneal and the
  attention-temperature anneal off (the weights converged at the deployed values; a 10-epoch run cannot replay a
  20-epoch anneal); every other key step 26's (paths replaced in every string value, `check_run_config.py`).

**Benchmark moved:** wet_d and wet_u (4G Table II, the T5 protocol) through the training-set class offsets.

| id | criterion (test against its control) |
|---|---|
| **W1** | training-set class offsets (flat − upright, model − DFT): \|doped\| and \|undoped\| each ≤ 50 % of the control's |
| **W2** | wet_d (T5 protocol) closer to DFT (−2.7) than the control's by ≥ 50 meV |
| **W3** | test-set F RMSE, each subset, not above the control's by > 10 % |
| W4 | (prediction) the doped − undoped offset stays between −30 and −120 meV (the doping-specific part no weight removes) |

PASS = W1 and W2 and W3: the energy fit is the bottleneck, and the next step is the energy weight on the full protocol.
W1 fails: the model as built cannot represent DFT's flat-upright energies at this weight (hypothesis (a) falsified);
the terms of the diagnosis's point 3 are then the place to look. Reported: E RMSE per subset, wet_u, T5, the energy
term's share of the final loss, any relaxation failure.

**Step 29 launched 2026-11-08 17:02:56 IST** (`watch_step29.sh`; test and control trainings; scored by the
watcher, read with `read_step29.py`). Both inherit step 26's `clip_grad_norm: 1.0`: the test's pre-clip gradient norms
are 15–24 against the control's 0.2–0.7 in epoch 1, so the energy term sets the test's direction and the clip its size
(the optimiser is Muon, whose orthogonalised update has a size set by the learning rate, not by the gradient norm). About 32 min per epoch with the two together.

**Step 29 scored (2026-11-08 21:48 IST; `read_step29.py`).**

| seed 43 | test (ew 1000) | control (ew 10) | step 26 |
|---|---|---|---|
| train class offset doped / undoped (meV) | −85.9 / −9.1 | −170.0 / −88.5 | −178.6 / −96.6 |
| test class offset doped / undoped | −90.7 / −0.1 | −173.5 / −82.8 | −182.7 / −90.7 |
| wet_d / wet_u (T5 protocol; DFT −2.7 / +928.9) | −98.1 / +906.8 | −190.2 / +822.9 | −198.2 / +810.6 |
| T5 (DFT −931.6) | −1004.9 (107.9 %) | −1013.1 (108.7 %) | −1008.8 (108.3 %) |
| test E RMSE doped / undoped (meV/atom) | 0.665 / 0.424 | 1.598 / 1.952 | 2.274 / 1.399 |
| test F RMSE doped / undoped (eV/Å) | 0.1049 / 0.0778 | 0.1039 / 0.0762 | 0.1056 / 0.0768 |
| energy term's share of the final loss | 7.1 % | 0.1 % | 0.3 % |

**Verdict: FAIL as pre-registered** — W1 FAIL (undoped 10 % of the control, doped 50.5 % against ≤ 50 %), W2 PASS
(wet_d 92 meV closer), W3 PASS (forces +1 / +2 %); W4 as predicted (doped − undoped −76.8 meV).

**Reading.** The criterion was internally inconsistent: W1 asked the whole doped offset to halve while W4 predicted that
its doping-specific part (≈ −80 of the control's −170) survives; removing the common part entirely leaves ≈ 47 % —
on W1's edge. The verdict stands as written. Measured: the common (both-doping) part of the class offset is an
energy-fitting deficit — at a 7 % energy share the undoped offset falls from −88.5 to −9.1 meV and wet_u reaches +906.8
(the closest of any full-stack run; 4G +975.4) with energies 2.4–4.6× better and forces within 2 %; so the
pre-registered reading of a W1 failure ("cannot represent") is contradicted by the undoped half. The doping-specific
−77 meV remains, the non-local physics the local terms cannot absorb, and it carries most of the remaining wet_d error
(−98.1 against −2.7). One seed; a 10-epoch warm start from step 26.

## Pre-registered: step 30, the energy weight on the full protocol (written 2026-11-08 21:54 IST, before the 21:54:36 launch; decision of 2026-11-08: run the energy weight on the full protocol with both seeds)

**Runs.** `..._pfp_bopw_e1k_bondlaw_r5_s{42,43}` = step 26's configuration with one change, `energy_weight: 1000`
(from 10): cold start, 40 epochs, lr 1e-3 cosine, the training-time kT and attention-temperature anneals, every other key
step 26's (paths replaced in every string value; `check_run_config.py`). On the Mac, the two concurrently. Comparator:
step 26, same seeds (and step 29's test for s43). Basis: step 29 (an energy share of 7 % removed 90 % of the undoped
class offset at 1–2 % force cost, s43, warm start).

**Benchmark moved:** 4G Table II — wet_d, wet_u and T5, each per seed.

| id | criterion (each seed) |
|---|---|
| **X1** | training-set undoped class offset (flat − upright) \|·\| ≤ 30 meV (step 26: −284 / −97; step 29 test −9) |
| **X2** | wet_u (T5 protocol) within ±100 meV of DFT +928.9 (step 26: — / +810.6) |
| **X3** | wet_d (T5 protocol) within ±150 meV of DFT −2.7 — consistent with the predicted doping-specific remainder |
| **X4** | T5 within ±15 % of DFT −931.6 |
| **X5** | test F RMSE, each subset, not above step 26's same seed by > 10 % (s42 0.062 / 0.085; s43 0.077 / 0.106) |
| X6 | reported: test E RMSE; the doped − undoped training offset (prediction −30 to −120 meV); T2; any relaxation failure |

PASS = X1–X5 on both seeds. A T5-protocol relaxation that fails (step 26 s42's singular dipole block) leaves X2–X4
unscored for that seed: FAIL. **Predictions (hypotheses):** X1, X2, X5 pass on both seeds; wet_d −80 to −150 meV (the
doping-specific remainder plus the protocol's own −20 to −70); a cold start may balance energies and forces differently
from step 29's warm start, so the force cost could be larger than 2 % (X5 is the guard).

**Step 30 launched 2026-11-08 21:54:36 IST** (`watch_step30.sh`; trainings s42 / s43, cold start, 40 epochs;
scored by the watcher with `score_formal.sh` + `score_extra.sh`, read with `read_step30.py`). Expected ≈ 15–21 h of
training with the two together (step 29's 22–32 min per epoch), then the scoring.

**Step 30 cancelled (2026-11-09 10:01 IST, decision of 2026-11-09: cancel step 30 and run the warm start on both seeds; branch
`step30-cancelled-2026-11-09` made first).** Stopped by exact PID at epoch 21 (s42) / 20 (s43) of 40; runs kept,
unscored. Measured before the stop (validation F RMSE, eV/Å; step 26 at the same epoch): epoch 12 0.379 / 0.418 (0.104 /
0.118), epoch 18 0.328 / 0.345 (0.087 / 0.103), epoch 20 0.510 / 0.326 (0.083 / 0.097); s42 spiked to 2.364 at epoch 19
(training 1.29–1.81), at the end of the 20-epoch electronic-temperature anneal (the trigger is a hypothesis, unchecked).
Reading: from a cold start an energy share near 7 % lets the energies dominate the early descent and the forces lag by
3–6×; X5 (forces within 10 % of step 26) was out of reach. The risk was named in the pre-registration. The relaunch is
step 29's measured recipe, a warm start from converged weights, on both seeds (step 31).

## Pre-registered: step 31, the energy weight from a warm start, both seeds (written 2026-11-09 10:01 IST, before the runs)

**Runs.** `..._pfp_bopw_e1kw_bondlaw_r5_s{42,43}`: each seed's step 26 final weights (s42 epoch 40, s43 epoch 39;
`warm_from_step26_sXX.pt`: weights only, epoch 0, no optimizer state), `energy_weight: 1000`, **20 epochs**, lr 1e-4
cosine to 5e-6 (t_max 20), the kT and attention-temperature anneals off; every other key step 26's (paths replaced in
every string value; `check_run_config.py`). This is step 29's test recipe, twice as long, on both seeds. On the Mac, the
two concurrently. Comparator: step 26, same seeds. Step 29's control (s43, energy weight 10, same warm start) measured
what ten more epochs alone do: class offsets −178.6 / −96.6 → −170.0 / −88.5, wet_d / wet_u −198.2 / +810.6 →
−190.2 / +822.9 — small, so changes against step 26 are attributed to the weight.

**Benchmark moved:** 4G Table II, wet_d, wet_u and T5, each per seed.

| id | criterion (each seed) |
|---|---|
| **Y1** | training-set undoped class offset (flat − upright) \|·\| ≤ 30 meV |
| **Y2** | wet_u (T5 protocol) within ±100 meV of DFT +928.9 |
| **Y3** | wet_d (T5 protocol) within ±150 meV of DFT −2.7 |
| **Y4** | T5 within ±15 % of DFT −931.6 |
| **Y5** | test F RMSE, each subset, not above step 26's same seed by > 10 % (s42 0.0617 / 0.0853; s43 0.0768 / 0.1056) |
| Y6 | reported: test E RMSE; the doped − undoped training offset (prediction −30 to −120 meV); T2; relaxation failures |

PASS = Y1–Y5 on both seeds; a T5-protocol relaxation failure leaves Y2–Y4 unscored for that seed: FAIL. **Predictions
(hypotheses):** s43 repeats step 29's test (Y1–Y5 pass: undoped offset ≈ −9, wet_u ≈ +907, wet_d ≈ −98, T5 ≈ 108 %); s42
starts from larger offsets (−339 / −284) and its step 26 relaxation failed on a singular dipole block, so s42 carries the
risk (Y2–Y4).

**Step 31 launched 2026-11-09 10:02:02 IST** (`watch_step31.sh`; trainings s42 / s43, 20 epochs from step
26's weights; scored by the watcher, read with `read_step31.py`). Expected ≈ 8–10 h of training with the two together,
then about an hour of scoring.

**Step 31 scored (2026-11-09 21:10 IST, `watch_step31.sh`, read with `read_step31.py`): FAIL as pre-registered.** s43
passes Y1–Y5 and s42 fails Y1 and Y3. Best checkpoints: s42 epoch 20, s43 epoch 17.

| | Y1 undoped offset (≤ 30) | Y2 wet_u (DFT +928.9) | Y3 wet_d (DFT −2.7) | Y4 T5 (DFT −931.6) | Y5 test F undoped / doped (eV/Å) | test E undoped / doped (meV/atom) | doped offset | T2 q(Au₂) |
|---|---|---|---|---|---|---|---|---|
| step 26 s43 | −96.6 | +810.6 | −198.2 | −1008.8 | 0.0768 / 0.1056 | 1.399 / 2.274 | −178.6 | — |
| **step 31 s43** | **+7.4** | **+920.4** | **−75.7** | **−996.1 (106.9 %)** | 0.0776 / 0.1045 | **0.469 / 0.621** | −65.2 | −0.999 |
| step 26 s42 | −284.3 | relaxation failed | relaxation failed | — | 0.0617 / 0.0853 | — | −338.9 | — |
| **step 31 s42** | −73.3 FAIL | +851.0 | −177.4 FAIL (by 25) | −1028.4 (110.4 %) | 0.0614 / 0.0867 | 0.590 / 0.937 | −124.1 | −1.000 |
| 4G-HDNNP (paper) | — | +975.4 | −41.0 | −1016.4 (109.1 %) | — | — | — | — |

- **s43:** the predictions held (offset ≈ −9: +7.4; wet_u ≈ +907: +920.4; wet_d ≈ −98: −75.7; T5 ≈ 108 %: 106.9 %).
  - Against 4G, wet_u (8.5 meV off DFT against 4G's 46.5) and T5 (6.9 % against 9.1 %) are closer to DFT.
  - wet_d is not: 73.0 meV off against 4G's 38.3.
- **s42:** it starts from offsets three times larger. Twenty epochs remove three quarters of the undoped offset, but
  not all of it: −284.3 → −73.3, with the window at 30.
- **Y6 (doped − undoped training offset):** −50.8 on s42 and −72.6 on s43, inside the predicted −30 to −120.
  - On s43 this doping-specific part equals the remaining wet_d error (−73.0 against −72.6).
  - Hypothesis, not tested: what is left of wet_d on s43 is the doping-specific misfit, which the energy weight does
    not touch.
- **T2:** unchanged at −1.00 e of free charge on Au₂, against DFT's Hirshfeld −0.27.
- **Forces:** within 1.6 % of step 26 on both seeds (Y5).
- **T6 ε∞, computed afterwards** (2026-11-10, `probe_eps_bound.py`, as in `score_run.sh`): 3.284 on s42 and 2.934 on
  s43. Step 26 gave 3.304 and 2.962; experiment is 2.95 and the window is 2.5–3.5.

## Pre-registered: the wet_d diagnosis, D31 (written 2026-11-10 10:55 IST, before any of its probes is run on step 31; decision of 2026-11-10: start with the wet_d diagnosis)

Not a step: no training, frozen weights. **Benchmark number it serves:** 4G Table II wet_d, and through it T5. Step 31
s43 is 73.0 meV off DFT; 4G is 38.3 off.

**Question.** On step 31 s43 the wet_d error (−73.0 meV) equals the training set's doped − undoped class offset (−72.6;
−50.8 on s42). What carries this doping-specific part, and is it something a term of the model cannot represent?
Twenty epochs of the energy weight removed 95 % of the undoped offset on s43 (−96.6 → +7.4) but only 21 % of this part
(−91.9 → −72.6).

**Inputs.** Frozen weights of step 26 and step 31, both seeds; the 300 training structures at their DFT geometries; the
four T5 references.

**Measurements.**
- **M1** (`probe_class_terms.py`, new; built on `probe_t5_terms.terms()`, whose terms are asserted per structure to sum
  to the module's energy at 1e-9). For each training structure it records:
  - the residual (model − DFT);
  - the backbone energy (total − E_q) and E_q's terms;
  - the tilt θ, the two Au heights, and the contact site.

  It reports two things:
  - **(i)** the doped − undoped class offset split into each term's double difference,
    DD_term = [⟨E⟩_flat − ⟨E⟩_upright]_doped − [same]_undoped, and its change from step 26 to step 31;
  - **(ii)** the residual against θ within each doping (slope ± standard error), and against the contact Au's height.
- **M2** (`probe_t5_first_order.py`, step 31 s43 and s42): T5 split into first order and relaxation, against PBE's
  embedded-cluster decomposition. PBE gives E1 −903.9, relax −281.9 and deform −33.0 meV. That is a one-Al 10-ion
  cluster, so only the shares are compared, not the magnitudes: PBE's relax share is 23 %, and step 26 s43's is
  −360.7 of −1023.2, or 35 %.
- **M3** (frozen direction check of step 27): step 31's weights with `bond_pauli_states` switched on. It measures the
  change of the doped flat − upright training offset, and of wet_d at the T5 references (unrelaxed).
- **M4** (`probe_site_decomposition.py`, step 31): in doped upright structures, does the electron prefer the near or the
  far Au? Step 26 s43 preferred the near one by −0.664 eV; DFT's Hirshfeld charges put it on the far one.

**Artifacts named, and how each is excluded.**
- **Relaxation** (the error lives only in the relaxed energy surface): M1 is at the DFT geometries, where the doped
  offset is already measured (−72.6).
- **Class composition** (the doped and undoped sets sample different geometries): the backbone's double difference
  measures it. The backbone cannot see the Al (> 10 Å away), so its DD is composition, not doping physics.
- **Statistics:** the class offsets carry ±5–7 meV standard errors (n ≈ 75 per class), so a term DD below 15 meV is not
  read.
- **Energy zero:** per-structure constants cancel in flat − upright within one doping.

**Readings (hypotheses, each with what falsifies it).**
- **(a) Over-response.** The doping-specific part is the substrate's response to the anion (the screened, ind26 and
  alpha28 terms; M2's relax part).
  - Predicted: step 31's relax share of T5 stays above 30 % (PBE 23 %), and in M1 the response terms carry the largest
    doped DD and moved by less than 20 meV from step 26 to step 31.
  - Falsified if the relax share is 23 % or less, or if the response DD is under 15 meV.
- **(b) The anion's orientation energy** (step 27's mechanism).
  - Predicted: within the doped classes the residual depends on θ, with a slope that differs from the undoped one by
    more than three standard errors; and M3 moves the doped offset by more than 20 meV.
  - If M3 moves the offset more negative, step 27 as built would make wet_d worse at first order.
- **(c) The electron's site.** M4 is reported. If the near Au is still preferred by more than 0.3 eV, the upright
  anion's charge sits where DFT's does not put it; that is a candidate, not a reading of wet_d by itself.

**Decision.** The term that carries the doping-specific part, and that the energy weight did not move, is the target
of the next pre-registered step:
- if (b) holds with the right sign, that step is step 27;
- if (a) holds, a change on the response side;
- if neither, report and stop.

**Correction to D31's question (written 2026-11-10 11:17 IST, after M1 and before M2–M4 were read).** The question
says the energy weight removed "only 21 % of this part (−91.9 → −72.6)". That compares step 26's test-set value
(−182.7 − (−90.7) = −92.0) with step 31's training-set value.
- On the training set the doping-specific part moved −82.0 → −72.6 on s43 (11 %) and −54.7 → −50.8 on s42 (7 %).
- Over the same 20 epochs the undoped offset moved by +104.0 (s43) and +211.0 (s42).

**D31 read (2026-11-10 11:35 IST; frozen; all four M1 runs reproduce `probe_class_residuals.py`'s training offsets
exactly).**

| | step 26 s42 | step 26 s43 | step 31 s42 | step 31 s43 |
|---|---|---|---|---|
| doped − undoped training offset (meV) | −54.7 ± 7.5 | −82.0 ± 9.2 | −50.8 ± 9.9 | −72.6 ± 9.2 |
| θ slope, doped − undoped (meV/deg) | +2.68 ± 1.94 | +1.74 ± 2.38 | +3.81 ± 2.56 | +1.34 ± 2.39 |
| M2: relax share of T5 (PBE cluster 23.1 %) | — | 35.3 % | 42.2 % | 35.4 % |
| M2: E1 share of T5 (PBE 74.2 %) | — | — | 68.3 % | 73.5 % |
| M3: step 27's law on, change of the doped offset | — | — | −174.4 | −179.5 |
| M3: change of wet_d at the T5 references | — | — | −212.7 | −213.1 |
| M4: electron near − far Au, doped upright (eV) | — | −0.664 | −0.383 | −0.710 |

- **(b), the anion's orientation energy: falsified.**
  - No run has a doped − undoped θ slope above 1.5 standard errors.
  - M3 moves wet_d the wrong way, by −213 meV on both seeds: three times the error, with the wrong sign. Step 27 as
    built is not the next step; it stays held.
- **(a), over-response: its M2 half holds on both seeds.**
  - The relaxation part of T5 is 35 % (s43) and 42 % (s42) of T5, against PBE's 23 %. The first-order share is right on
    s43 (73.5 % against 74.2 %).
  - Its M1 half is not decidable as designed. Each term varies from structure to structure, so its class means carry
    ±30–117 meV, above the 15 meV read threshold; that is a flaw in the pre-registration. Only alpha28 (DD −292 to −317
    ± 7–9) and ind26 (+36 to +74 ± 9–15) are resolved.
  - Both resolved response terms moved by less than 6 meV under the energy weight, as (a) predicts.
- **(c), the electron's site: the candidate stands.** The near Au is still preferred, by −0.38 eV (s42) and −0.71 eV
  (s43); DFT's Hirshfeld charges put the electron on the far Au.
- **The doping-specific part is stiff.** Over 20 epochs of the energy weight it moved +3.9 (s42) and +9.4 (s43) meV,
  against +211 / +104 for the undoped offset.
- **Not pre-registered, reported:** in this training set the classes are also sites. Flat dimers lie over Mg²⁺ at a
  contact height of 2.84 Å; upright ones stand on O²⁻ at 2.26–2.30 Å. The undoped flat class (neutral Au₂ over Mg²⁺)
  has a residual slope in contact height of −148 to −213 meV/Å in all four runs (8–16 standard errors). The doped
  slopes change sign between runs. A first version of the geometry function measured heights from the slab's far
  surface; it was fixed and checked (contact Au–O 2.26–2.30 Å upright) before any height number was read.
- **Decision (by the pre-registered rule):** the response side. The frozen SPEC 34 / SPEC 35 test on the 4G slabs,
  pre-registered on 2026-11-07, was never run. Its counterfactual is the next measurement: each term alone on step 31's
  frozen weights, read for the relax share, the doped − undoped offset and wet_d, before any training.

**Correction to the D31 decision line (2026-11-10, found during the internal review of the same day).** The statement
"The frozen SPEC 34 / SPEC 35 test on the 4G slabs, pre-registered on 2026-11-07, was never run" is wrong. It was run
on 2026-11-07 23:07 (`probe_frozen_terms_4g.py`; output `runs/frozen_terms_4g.txt`) and never read into either README.
The statement stays above as written; this note replaces it. What the file holds (step 20's weights, frozen; every 4th
test structure, 47 undoped + 53 doped, not the 50 + 50 the pre-registration named):

| arm | undoped F RMSE s42 / s43 (eV/Å) | doped F RMSE s42 / s43 | doped mean energy offset s42 / s43 (meV/atom) |
|---|---|---|---|
| neither (step 20) | 0.0520 / 0.0576 | 0.0894 / 0.1083 | +2.1 / +0.8 |
| anchor, g = 1 | 4.6651 / 0.2413 | 0.4938 / 0.3647 | −36.4 / −1.0 |
| anchor, g = 0.92 | 3.5112 / 0.2455 | 0.4881 / 0.3864 | −33.1 / +2.3 |
| damping only | 49.9189 / 0.7583 | 0.7152 / 0.8286 | −73.1 / −83.4 |
| anchor g = 1 + damping | 0.4807 / 0.6526 | 0.5839 / 0.8405 | −86.7 / −53.4 |

- **Read by its pre-registered rule:** no arm leaves the undoped force RMSE within 10 % of step 20's; both terms cost
  the slabs, alone and together. The rule cannot rank them: every arm is 4 to 960 times step 20's error.
- **What it means for the measurement proposed above:** a frozen switch-on of either term is not a small perturbation
  of a backbone trained without it (class means move by up to 87 meV/atom). A frozen counterfactual on step 31's
  weights would therefore not give a readable wet_d or class offset. Only the term-level relax share could be read
  from it, and that reading is not yet justified. The proposal is withdrawn until it is redesigned.
- **Not checked (hypothesis):** seed 42's 4.67 and 49.9 eV/Å are large enough to suggest a near-singular bound response
  on a few structures rather than a uniform shift. The per-structure distribution has not been looked at.

## Internal review, 2026-11-10: measurements (step 31's frozen weights; no training, no code change)

Written for the review of the code, methods, novelty and results requested on 2026-11-10. Probes:
`probe_review_{spectrum,charges,hirshfeld,supercell,supercell_cfg,supercell_window,timing}.py`; outputs in
`runs/review_2026-11-10/`. Nothing here was pre-registered: these are review checks, each with the hypothesis it was
run to falsify.

**R1. At the T5 references the ensemble is a ground state of 2–5 configurations; kT plays no role.**
(`probe_review_spectrum.py`, both seeds.)
- Configurations: 2 (doped), 3 (undoped wetting), 4–5 (undoped non-wetting). Lowest eigen-gaps 3.56–5.30 eV.
- F − λ_min = 0.000 meV in all eight evaluations.
- wet_u / wet_d / T5 are identical at kT = 0.1 (trained), 0.05 and 0.025 eV: s43 +934.0 / −80.0 / −1014.0 meV, s42
  +864.0 / −175.0 / −1039.0 meV.
- Hypothesis falsified: the −69.05 meV "Gibbs F − F1" of `S5_first_order.txt` is not a kT ln 2 entropy (69.31 meV).

**R2. Correction to D31's reading (c), "the electron's site: the candidate stands".** That statement stays above as
written; this note replaces it.
- M4 is a difference of diabatic (diagonal) energies over 12 doped upright test structures. It is not the model's
  charge distribution. At the doped upright T5 reference the two configurations differ by 0.3231 (s43) / 0.2852 (s42)
  eV on the diagonal and are coupled by 2.6433 / 2.0723 eV, so the ground state shares the electron: weight on the Au
  nearer the slab 0.5305 (s43) / 0.5343 (s42). Doped wetting: 0.5002 / 0.4998 on both seeds.
- The reference data's own Hirshfeld charges, all 5000 frames (`probe_review_hirshfeld.py`):

  | class (1250 frames each) | q(near Au) | q(far Au) | q(Au₂) |
  |---|---|---|---|
  | doped wetting | −0.0809 ± 0.0266 | −0.0967 ± 0.0253 | −0.1775 |
  | doped non-wetting | +0.0345 ± 0.0595 | −0.2368 ± 0.0222 | −0.2022 |
  | undoped wetting | +0.0989 ± 0.0141 | +0.0832 ± 0.0133 | +0.1821 |
  | undoped non-wetting | +0.1071 ± 0.0346 | −0.1294 ± 0.0429 | −0.0223 |

  The far Au is already at −0.129 without doping. The doping shift of the upright dimer is −0.073 (near) / −0.107
  (far): 40 % / 60 % of −0.180 e. The model's ground state gives 53 % / 47 % of −1 e.
- Read: (c) as a site error is not supported. The gap between model and data is 13 points of a shared electron, in a
  population analysis that returns ±0.35–0.36 e for the slab's formal Mg²⁺ and O²⁻ (18 % of the formal charge).
- For T2: the doping shift of q(Au₂) is −0.360 e in the wetting class and −0.180 e in the non-wetting class (mean
  −0.270, the number T2 has used). A model charge of −1.00 e against a Hirshfeld −0.27 e is not a like-for-like test.

**R3. The full model is not extensive on replicated cells; the local model is.** (`probe_review_supercell.py`, s43;
the four T5 references tiled 2×1 and 2×2 in the surface plane; total charge 0; one energy + force call, 4 threads.)

| structure | cell | atoms | configurations | E/cell − E(1×1) (meV) | max force change (eV/Å) | time (s) |
|---|---|---|---|---|---|---|
| local model, all four | 2×1, 2×2 | 220, 440 | — | 0.0000 | below 1e-8 | 0.5–0.9, 1.0–1.2 |
| doped non-wetting | 1×1 | 110 | 2 | 0 | 0 | 1.37 |
| doped non-wetting | 2×1 | 220 | 1 | +7048.8 | 1.94 | 3.95 |
| doped non-wetting | 2×2 | 440 | refused | — | — | — |
| doped wetting | 2×1 | 220 | 1 | +5279.5 | 1.50 | 3.20 |
| doped wetting | 2×2 | 440 | refused | — | — | — |
| undoped non-wetting | 2×1 | 220 | 11 | −12.1 | 0.029 | 3.14 |
| undoped non-wetting | 2×2 | 440 | 65 | +116.5 | 0.186 | 13.60 |
| undoped wetting | 2×1 | 220 | 9 | −7.2 | 0.025 | 2.82 |
| undoped wetting | 2×2 | 440 | 33 | +211.9 | 0.192 | 12.86 |

- **Doped, cause identified** (`probe_review_supercell_cfg.py`, and `trace_redox/band.py`'s docstring): the band unit
  is one unit per structure holding at most 2 electrons. In the 110-atom cell the three Al³⁺ give 3 electrons: 2 in the
  band, 1 on Au₂. In the 2×1 cell they give 6: the band still takes 2, and the only configuration left has all four Au
  at −1. The refusal at 2×2 is `FragmentEnsembleError` (12 connections in their switching shell; the percolation
  ensemble enumerates 2^k patterns, k ≤ 10).
- **Undoped, cause partly identified.** The 2×1 ensemble holds 11 of the 25 products of the cell's 5 configurations;
  its weights are close to products of the cell's (0.4086 against 0.6407² = 0.4105; 0.1545 against 0.1553). A selected
  configuration space is not a product space, so it is not size-consistent.
- **Attempt to separate truncation from the periodic images** (`probe_review_supercell_window.py`; plain energy
  window in place of the amplitude selection): not completed. In the cell, a window of 4.5 / 8.0 eV holds 21 / 166
  (non-wetting) and 57 / 165 (wetting) configurations and moves the energy by −5.0 / −15.9 and −3.9 / −4.2 meV from
  the trained selection. In the 2×1 cell a window of 5.0 eV holds 288 / 445 configurations and is +359 / +384 meV per
  cell above; 9 eV and 16 eV are refused (`ConfigurationSearchError`, more than 512).
- **Hypothesis, not checked:** one Hamiltonian per cell puts every periodic image in the cell's configuration, so a
  state mixing configurations of different dipole carries the image interaction with ⟨μ²⟩, where a mean density (the
  Hartree term of the reference DFT) carries ⟨μ⟩². The in-plane image distance here is 9.047 Å.
- Artifact checks made: the replicas are exact (the local model reproduces energy to 1e-4 meV per cell and forces to
  1e-8 eV/Å); the total charge is 0 in every cell; the doped result follows from the band unit's stated capacity.

**R4. Cost and numerical health.**
- One energy + force call, 110 atoms, 4 threads, float64 on the Mac (`probe_review_timing.py`): local 0.30–0.58 s,
  full 1.50–1.52 s. Of the full model's time, 54 % is the backward pass and 41 % the redox forward.
- Training: 5.86 s per structure-step (step 31 s43, 4 threads, batch 1) against 1.12 s (local `AuMgO_base_s42`, 5
  threads, batch 8).
- Step 31 logged 2 transient non-finite gradients per seed in 6000 sub-steps (epochs 5 and 14 on s42; 7 and 19 on
  s43). Each recomputation was finite with a bitwise-equal loss; the probe brackets the first non-finite gradient
  between `lattice.combined.out` and `diabatic.E`. Cause not found. Checked and excluded: the two `torch.empty_like`
  allocations on that path (`special.py:40`, `downfold.py:522`) are fully written.
- `pytest tests/redox`: 817 passed in 192.6 s (2026-11-10, Mac).

### Internal review, additions (2026-11-10 afternoon; measured or derived from R1–R3; no training, no code change)

- **Re-run.** All seven probes were re-run into `runs/review_2026-11-10/` and reproduce R1–R3 to the printed digits.
  The timing of R4 varies between runs: local 0.27–0.52 s, full 1.26–1.29 s per energy + force call (earlier
  0.30–0.58 and 1.50–1.52 s); the backward pass is 48 % and the redox forward 45 % of the profiled call.
- **R5, the dopant's distance (receptive field a local model needs).** Over the 388 doped frames of our split, the
  nearest Au–Al distance (minimum image) is 10.49–11.98 Å, mean 11.19 Å; all Au–Al distances lie in 10.49–14.68 Å. The
  slab spans z = 10.58–21.25 Å with Au₂ below it (z 5.8–8.3 Å) and Al at z = 19.05–19.17 Å: the fifth layer from
  the Au side. One layer at r_max 5.29 Å cannot see the dopant; three layers (15.9 Å) can; PFN's NequIP has
  6 × 4 = 24 Å. No multi-layer local TRACE has been trained on Au₂/MgO.
- **R6, the cell dependence of Table II (derived from R3).** wet_u = E(wetting) − E(non-wetting), undoped: it changes
  by +4.8 meV at 2×1 (−7.2 − (−12.1)) and by +95.4 meV at 2×2 (+211.9 − (+116.5)) from its 1×1 value.
- **R7, wet_d as a difference of two large terms (exact 2 × 2 algebra on R1 / R2's numbers).** In the doped cells the
  ensemble is two configurations, the electron on one Au or the other, with diagonal difference Δ and coupling t; the
  ground state lies Δ/2 − ½ gap below the lower diabat.

  | seed | geometry | Δ (eV) | t (eV) | ground state − lower diabat (eV) |
  |---|---|---|---|---|
  | s42 | wetting | 0.0013 | 1.7809 | −1.7802 |
  | s42 | non-wetting | 0.2852 | 2.0723 | −1.9346 |
  | s43 | wetting | 0.0015 | 2.1237 | −2.1229 |
  | s43 | non-wetting | 0.3231 | 2.6433 | −2.4867 |

  The mixing energy contributes +154.4 meV (s42) and +363.8 meV (s43) to wet_d at the fixed references, where wet_d is
  −175.0 and −80.0 meV; everything else therefore contributes −329.4 and −443.8 meV. The two parts differ between the
  seeds by 209 and 114 meV, each more than wet_d's own seed spread (95 meV at the fixed references).
- **R8, T2 and electron counting (inference from R2's configuration lists and `band.py`; not separately tested).** In
  the doped cell three Al³⁺ give three electrons and the band unit holds at most two, so in every selected
  configuration the third sits on one Au. No selected configuration has a neutral dimer. The model's −1.00 e on Au₂ is
  therefore set largely by the band's capacity, not by a learned balance.
- **R9, unused reference data.** The source file holds 5000 frames; the split uses 776 (300 / 76 / 400). 4,224 frames
  have never been read by training, early stopping or any scoring.
- **R10, prior-art search (eight web queries, 2026-11-10; not a systematic review).** Found and not cited in
  `trace_redox/SPEC.md`: the Fragment Hamiltonian model (Valone and co-workers; fragments in integer charge states
  with inter-fragment hopping and on-fragment repulsion), Valone and Atlas (J. Chem. Phys. 120, 7262 (2004); Phys.
  Rev. Lett. 97, 256402 (2006)), Atlas (arXiv:2101.04786, J. Phys. Chem. A 2021; atoms as ensembles of neutral,
  excited and ionic states in a force field), Morales and Martínez (J. Phys. Chem. A 105, 2842 (2001); 108, 3076
  (2004)), Wu, Cheng and Van Voorhis (CDFT-CI, J. Chem. Phys. 2007), Kundu, Ye and Berkelbach (arXiv:2411.04807;
  diabatic charge-transfer states from constrained charge equilibration), Xie, Persson and Small (BpopNN, J. Chem.
  Theory Comput. 16, 4256 (2020)), and neural-network diabatic Hamiltonians (e.g. arXiv:2108.04879, 2203.03083). EVB
  and MS-EVB years are from memory and were not confirmed by the search.
- **R11, where the undoped non-extensivity sits (2026-11-10 evening; `probe_review_supercell_diag.py`, s43 weights,
  output in `runs/review_2026-11-10/supercell_diag.txt`; no training, no code change).** E(2×1) − 2 E(1×1) is −24.125
  (non-wetting) and −14.483 meV (wetting), i.e. R3's −12.06 / −7.24 meV per cell. Split: everything outside the ensemble
  (E − λ_min) +0.000 meV; the frame configuration's diagonal energy +0.000 meV; the mixing energy (λ_min − frame
  diagonal) carries all of it (per cell −2752.673 → −2764.735 meV non-wetting, −1949.382 → −1956.623 meV wetting). Two
  measured parts of that: (i) a transfer on one dimer with the neighbouring copy left in the frame costs 1.5258 eV at
  2×1 against 1.5417 eV in the cell (−15.9 meV), and the reverse transfer 3.3576 against 3.4319 eV (−74.3 meV): in the
  cell every periodic image is excited with it; (ii) the in-phase double transfer at 2×1, which is the same periodic
  charge state as the cell's single transfer, does not cost twice the cell's value: +76.6 and +19.0 meV (non-wetting,
  Au (+1, −1) and (−1, +1)), −83.9 and −84.4 meV (wetting). So the diagonal energy of a non-frame configuration is
  itself not extensive. **Cause of (ii) not identified**; no term-by-term decomposition of that element has been run.

## The extensivity line (branch `extensivity-2026-11-10`; decision of 2026-11-10: solve this defect)

The defect (R3, R6, R11): replicating the periodic cell changes the energy per cell and the forces, so Table II's
wet_u, wet_d and T5 depend on the cell they are computed in (R6: wet_u moves by +4.8 meV at 2×1 and +95.4 meV at 2×2).
**Benchmark numbers this line moves: Table II (wet_u, wet_d, T5) and the Au₂/MgO force RMSE, through their dependence
on the cell.** Nothing here is trained yet; every number below is from step 31's frozen weights (s43).

### R12, the non-extensive diagonal element located: SPEC 28 scales the atom's interaction with its own images (2026-11-10 16:00 IST; `probe_ext_terms.py`, `probe_ext_spec28.py`; output in `runs/review_2026-11-10/ext_terms.txt`, `ext_spec28.txt`; no training, no package code changed)

This replaces R11's last sentence ("Cause of (ii) not identified").

1. **Term by term.** For the frame configuration and the in-phase Au transfers, each diagonal term at 2×1 minus twice
   its 1×1 value (meV; excitations relative to the frame):

   | geometry | configuration | ladder | linear | quadratic | pair tables | SPEC 28 | total |
   |---|---|---|---|---|---|---|---|
   | non-wetting | frame | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
   | non-wetting | Au (+1, −1) | −7.272 | 0.000 | 0.000 | +7.272 | +76.584 | +76.584 |
   | non-wetting | Au (−1, +1) | +2.108 | 0.000 | 0.000 | −2.108 | +19.005 | +19.005 |
   | wetting | Au (+1, −1) | +51.605 | 0.000 | 0.000 | −51.605 | −83.881 | −83.881 |
   | wetting | Au (−1, +1) | +51.641 | 0.000 | 0.000 | −51.641 | −84.355 | −84.355 |

   The ladder and pair-table entries cancel exactly: an atom's pair terms with its own periodic images are one-body
   entries at 1×1 and pair entries at 2×1 (bookkeeping, not a defect). The linear and quadratic (W) terms are
   extensive. **All of the in-phase non-extensivity is the SPEC 28 term.**
2. **Cause.** SPEC 28 writes K(n) = K + Σ_j P_jᵀ Δ_j P_j with Δ_j = P_jj (1/ρ_j − 1), and
   `polarizability._dipole_block` takes P_jj from the periodic operator. There P_jj = A_μ,j + T2_jj, where T2_jj is the
   dipole–dipole lattice sum of atom j with its own periodic images: zero in an open system, cell dependent in a
   periodic one. Measured eigenvalues (eV/Å²): A_μ of Au 108 is (7.069, 9.388, 9.390) in both cells; T2_jj is
   (−0.0426, −0.0426, +0.0447) at 1×1 and (−0.0454, +0.0072, +0.0179) at 2×1 (Au 109, non-wetting: ±0.19–0.20 at 1×1).
   A charge state changes the atom's own polarisability, i.e. A_μ,j, and nothing else: scaling T2_jj by 1/ρ_j changes
   the interaction of the atom's dipole with its own images, which is not physics. **This is an error in SPEC 28's
   statement in a periodic cell; in an open system the two forms are identical.**
3. **Falsifying check, run before this record.** With Δ_j = A_μ,j (1/ρ_j − 1) (the block patched in the probe's process
   only), the same four excitations:

   | geometry | configuration | form | δE₂₈(1×1) meV | δE₂₈(2×1) meV | 2×1 − 2·(1×1) meV |
   |---|---|---|---|---|---|
   | non-wetting | Au (+1, −1) | block (as shipped) | −716.458 | −1356.332 | +76.584 |
   | non-wetting | Au (+1, −1) | on-site | −654.200 | −1308.400 | −0.000000 |
   | non-wetting | Au (−1, +1) | block | −672.139 | −1325.273 | +19.005 |
   | non-wetting | Au (−1, +1) | on-site | −656.414 | −1312.829 | −0.000001 |
   | wetting | Au (+1, −1) | block | −329.135 | −742.151 | −83.881 |
   | wetting | Au (+1, −1) | on-site | −362.995 | −725.989 | −0.000000 |
   | wetting | Au (−1, +1) | block | −328.038 | −740.430 | −84.355 |
   | wetting | Au (−1, +1) | on-site | −362.168 | −724.337 | +0.000000 |

   The on-site form is extensive to 1e-6 meV. It also changes the 1×1 values of these four configurations by +62.3,
   +15.7, −33.9 and −34.1 meV, so the trained models' 1×1 scores will move under it; by how much is not measured yet.
4. **What this does not explain.** The ensemble's own non-extensivity (R11 (i), the doped +7 eV of R3, the doped 2×2
   refusal) is separate; R11 (i)'s −15.9 and −74.3 meV were measured with the block form and are confounded by it.

### Pre-registered: X1, SPEC 28 on the atom's own stiffness (written 2026-11-10 16:10 IST, before the test and the code)

- **Change (one).** Δ_j = A_μ,j (1/ρ_j − 1) in `state_polarizability_corrections`, `state_polarizability_response` and
  `stiffness_defects`. ESTABLISHED: an ion's polarisability depends on its charge state; Woodbury. The correction is to
  TRACE's own SPEC 28 (NEW-TO-TRACE), not a new term. Config key `state_polarizability_stiffness`: `'onsite'`
  (default, the corrected statement) or `'block'` (legacy, kept; SPEC 28 as first written). A checkpoint written before
  the key exists stores every RedoxConfig field (train.py), so its absence identifies the block form: the loader gives
  such a checkpoint `'block'` and says so.
- **Tests, written red first** (`tests/redox/test_redox_extensivity.py`): (1) P_jj − A_μ,j is `t2_pair[j, j]`, zero in
  an open system and nonzero and cell dependent in a periodic one; (2) with the on-site form the in-phase SPEC 28 term
  of a replicated cell is the number of copies times the cell's, 3-D and slab, tolerance from the Ewald accuracy and
  machine epsilon; with the block form it is not; (3) the on-site correction equals the brute-force
  −½ sᵀK(n)⁻¹s + ½ sᵀK⁻¹s with only A_μ,j scaled in K(n), in a periodic cell; (4) on-site and block agree to round-off
  in an open system; (5) the config key, its validation and the loader's legacy rule.
- **Pass criteria.** (a) The new tests pass; `tests/redox` passes with no test changed in what it asserts (817 before);
  the rest stays at 1045 passed, 4 skipped. (b) On step 31 s43's frozen weights with the shipped `'onsite'` path, the
  four in-phase elements of R12.3 satisfy |δE(2×1) − 2 δE(1×1)| ≤ 1e-3 meV. (c) The same checkpoint loaded without any
  override reproduces its stored Table II (wet_u +920.4, wet_d −75.7, T5 −996.1 meV) to the printed digit.
- **Reported, not criteria.** Table II on the frozen weights of both seeds under `'onsite'` (models trained with the
  block form: this measures the size of the change, it is not a score); E(2×1) − 2 E(1×1) undoped under `'onsite'`;
  the image term of a single transfer (one copy excited at 2×1 and 2×2 against the cell's in-phase value), now
  unconfounded. These decide what the next change must be. A retraining under `'onsite'` is a training step of its own
  and needs its own pre-registration and a decision.

### X1 scored (2026-11-10)

- **(a) Tests: met, with one stated deviation.** 12 new tests pass (`tests/redox/test_redox_extensivity.py`);
  `tests/redox` 830 passed, the rest 1045 passed, 4 skipped. **Deviation from "no test changed in what it asserts":**
  `test_the_closed_form_holds_in_a_small_periodic_cell` (`test_redox_master_equation.py`) encoded the block statement
  (its closed form missed the on-site energy by 1.9e-4 eV). It is now parametrized: `'block'` keeps the legacy closed
  form verbatim, `'onsite'` asserts the on-site closed form.
- **(b) Met.** In-phase SPEC 28 term on s43's frozen weights, shipped `'onsite'` path: max |δE(2×1) − 2 δE(1×1)| =
  7.33e-07 meV (criterion ≤ 1e-3).
- **(c) Met.** s43's checkpoint loaded without override (the loader restores `'block'` and warns) gives wet_u +920.36,
  wet_d −75.73, T5 −996.1 meV, Au–O 2.306 / 2.168 Å: its stored Table II.
- **Reported, not criteria (frozen weights trained with the block form; a size of the change, not a score).**

  | Table II, meV | wet_u | wet_d | T5 | Au–O doped / undoped, Å |
  |---|---|---|---|---|
  | DFT | +928.9 | −2.7 | −931.6 | 2.332 / 2.190 |
  | 4G-HDNNP | +975.4 | −41.0 | −1016.4 | |
  | s43 `'block'` (as trained) | +920.36 | −75.73 | −996.09 | 2.306 / 2.168 |
  | s43 `'onsite'`, frozen | +892.68 | −60.70 | −953.38 | 2.306 / 2.169 |
  | s42 `'block'` (as trained) | +850.97 | −177.44 | −1028.42 | |
  | s42 `'onsite'`, frozen | +832.37 | −164.68 | −997.04 | 2.301 / 2.171 |

  Replicated cells, s43 `'onsite'`, E/cell − E(1×1) in meV (max |ΔF| against the tiled 1×1 forces, eV/Å):
  non-wetting 2×1 −22.90 (7.9e-03), 2×2 +99.83 (1.94e-01); wetting 2×1 +5.64 (1.21e-02), 2×2 +220.84 (1.84e-01);
  doped non-wetting 2×1 +7047.33 (1.93), doped wetting 2×1 +5279.61 (1.50); doped 2×2 refused (12 > 10 connections).
  One copy excited against the cell's in-phase value (the image term on a single transfer's excitation energy):
  non-wetting Au(+1,−1) −56.40 (2×1) / −95.42 (2×2) meV, Au(−1,+1) −84.24 / −137.55; wetting +15.48 / +5.05 and
  +15.36 / +4.88.
- **Reading.** X1 removes SPEC 28's own-image term to round-off. The model is still not extensive: R13 locates the rest.

### R13. Where the remaining non-extensivity is (measured 2026-11-10; `probe_ext_product.py`, s43, `'onsite'`)

The probe evaluates a replicated cell on three configuration sets: the shipped selection; the **product space** of the
1×1 cell's selected configurations (every copy in each of its 1×1 states); and the product space with every coupling
at full strength (SPEC 21's switch off). The shipped selection holds no configuration outside the product space in any
row. Non-wetting is evaluated on the four band-inactive configurations (the band unit is one unit for the whole cell,
R3, so its state has no product space; leaving it out moves the 1×1 energy by +0.0010 meV).

| E/cell − E(1×1), meV (max \|ΔF\|, eV/Å) | shipped | product space, switch on | product space, switch off |
|---|---|---|---|
| wetting 2×1 | +5.6358 (9 configurations; 1.21e-02) | +5.6358 (9) | +5.6358 (9; 1.21e-02) |
| wetting 2×2 | +220.8418 (33; 1.84e-01) | +220.7960 (81; 1.82e-01) | **+2.7793** (81; 1.25e-02) |
| non-wetting 2×1 | −22.9030 (11; 7.87e-03) | −22.9030 (16) | −22.8896 (16; 7.89e-03) |
| non-wetting 2×2 | +99.8340 (65; 1.94e-01) | +95.1139 (256; 3.28e-01) | **−38.0215** (256; 1.24e-02) |

(The switch-off column is taken against the 1×1 cell with the switch off, which differs from the shipped 1×1 by 0 and
−4.65 meV.)

1. **SPEC 21's importance is global, and that is the defect: +218.0 (wetting) and +133.1 (non-wetting) meV/cell at
   2×2, forces 0.18–0.33 eV/Å.** As coded, a configuration's excitation x_c is measured from the thermal reference of
   the whole structure and its importance multiplies along the path from that reference: r_j = r_i g(t, x_j),
   g(t, x) = |t| / √(x² + t²). With k copies excited x ≈ kΔ, so the factors are g(t, Δ), g(t, 2Δ), ... From the wetting
   1×1 numbers (t = 2.125 eV, Δ = 2.643 eV; arithmetic on the code's formula, copies taken as non-interacting) they are
   0.627, 0.373, 0.259, 0.197 and the importances of the singly to quadruply excited configurations 0.63, 0.23, 0.060,
   0.012: the doubles are above a_on = 0.2, the triples just above a_off = 0.05, the quadruples below it. So 2×1 is
   untouched and 2×2 loses the couplings into its triply and quadruply excited configurations. This is the
   size-consistency failure of an amplitude threshold on a global configuration space (truncated CI).
2. **The truncation of the set itself is small with the switch on:** 0.05 and 4.7 meV/cell at 2×2.
3. **What is left with every coupling on is a finite-cell term:** +5.6 / +2.8 (wetting) and −22.9 / −38.0 meV per cell
   (2×1 / 2×2), forces 1.2e-02 eV/Å. In the 1×1 cell each cluster's charge fluctuation is in phase with its periodic
   images (one cell at Γ); in the supercell the copies fluctuate independently. This is a finite-size error of the
   1×1 calculation, not a size-consistency failure: it converges with the supercell.
4. **The band unit (SPEC 20) is one unit per structure holding at most two electrons** (R3): the doped +5.3 / +7.0 eV
   per cell at 2×1. Separate change.
5. **The doped 2×2 refusal** (12 > 10 connections): cause not established. Hypothesis, unchecked: the formal-anchor
   branch computes `percolation_partitions` and uses only whether it is None.

### Pre-registered: SPEC 36, the cluster product ensemble (written 2026-11-10 16:58 IST, before the tests and the code)

**Benchmark number it moves.** E/cell − E(1×1) and max |ΔF| of the replicated Au₂/MgO Table II cells (R13's first
column), with Table II itself (wet_u, wet_d, T5 at 1×1) unchanged for the undoped pair.

**Change (one).** `RedoxConfig(diabatic_space='clusters')`; `'global'` (default) is the ensemble as shipped, kept.

- *Reference* n⁰: the lowest configuration found (SPEC 17.3's placement and descent, re-centred on anything lower).
- *Linked moves* — NEW-TO-TRACE. From n⁰ every admissible move is allowed. From a configuration m ≠ n⁰ a move is
  allowed only if its donor or its acceptor lies in supp(m) = {u ∉ R : m_u ≠ n⁰_u}. R holds the reservoir units (the
  band unit): a reservoir never links. The window rule (E − E(n⁰) ≤ window) and the amplitude rule (r·g > a_off, T ≠ 0)
  are SPEC 17.3's and SPEC 21's, with the selection's importance measured from n⁰ (x = E − E(n⁰), r(n⁰) = 1). The
  cluster's own importance below is never larger (its thermal reference lies at or below E(n⁰)), so the selection
  holds every configuration whose coupling the switch leaves on.
- *Clusters* — NEW-TO-TRACE. The connected components of the graph on non-reservoir units whose edges join the units
  of each selected configuration's support. By construction every selected configuration lies in one cluster. S_c is
  n⁰ and the configurations of cluster c.
- *Cluster Hamiltonian.* H_c is SPEC 21's switched Hamiltonian on S_c with the exact energies E(n) of the whole
  system (every other cluster at n⁰; SPEC 28 included), its importance measured from the cluster's own thermal
  reference. F_c = −kT ln Tr exp(−H_c / kT).
- *Pair Hamiltonian* on S_c × S_c′ (valid and admissible products): diagonal E(n⁰ + δ_s + δ_s′), exact; off-diagonal
  H_c ⊗ 1 + 1 ⊗ H_c′.
- *Free energy* — the method of increments (ESTABLISHED for correlation energies: Stoll 1992; the citation is checked
  before it enters the SPEC) applied to the free energy of TRACE's configuration Hamiltonian (NEW-TO-TRACE):
  F = E⁰ + Σ_c (F_c − E⁰) + Σ_{c<c′} (F_cc′ − F_c − F_c′ + E⁰), E⁰ = E(n⁰). Charges and populations by the same
  increments of ρ's diagonal, so charge = ∂F/∂b.
- *Claimed identities, each with a test.* P1: one cluster reproduces the shipped ensemble. P2: clusters that do not
  interact are additive to round-off, for any number of copies. P3: two clusters are exact: F equals that of the
  global Hamiltonian on their product space (unswitched, so that the comparison holds no importance). P4: with three
  clusters the error is of second order in the inter-cluster interaction, the chain c–c′–c″ (corrected 17:10 IST,
  before any test or code: first written as third order, which counts the populations too). P5: charge = ∂F/∂b; first and second
  derivatives pass gradcheck / gradgradcheck.
- *Stated limits.* L1: three-cluster terms are dropped. L2: a selected configuration that spans two clusters merges
  them, and the merged cluster's space lacks the products of its parts; the energy is not smooth where that happens.
  L3: the band unit is still one unit per structure (R13.4). L4: the 1×1 finite-cell term (R13.3) stays. L5: refused
  together with SPEC 32 (`downfold_transfer`) and with SPEC 30's response priors on more than one cluster. Cost: the
  selection is linear in the number of clusters; the pairs are quadratic.

**Tests, written red first** (`tests/redox/test_redox_cluster_product.py`): P1–P5 on small open systems built from
tensors; the shipped ensemble's failure of P2 at three and four copies (pinned, so the defect stays on record); the
config key and its refusals.

**Pass criteria** (s43 frozen weights, `'onsite'`, `diabatic_space='clusters'`).
(a) The new tests pass; `tests/redox` 830 plus the new ones with no test changed; the rest 1045 passed, 4 skipped.
(b) Undoped 1×1, both geometries: |E − E_global| ≤ 1e-9 eV and max |ΔF| ≤ 1e-8 eV/Å (Table II's undoped pair
    unchanged).
(c) Two copies are exact: wetting 2×1 E/cell − E(1×1) = +5.636 ± 0.010 meV; non-wetting 2×1 = −22.903 ± 0.010 meV.
(d) Four copies: wetting 2×2 = +2.78 ± 1.0 meV, non-wetting 2×2 = −38.02 ± 1.0 meV (the tolerance is my estimate of
    the three- and four-cluster terms, not a measurement), and max |ΔF| ≤ 2e-02 eV/Å in all four replicated rows.
**Reported, not criteria.** The doped cells (1×1 against global; 2×1, expected to fail on L3); cluster counts and
timings; the doped 2×2 refusal's falsifying check.
**Not in this step.** No training, no change of default, no change to the 1×1 functional. C2b (linking
configurations inside pair spaces, to remove L2), C4 (an extensive band, L3) and C3 (the cluster's own images out of
its fluctuation, L4) each need their own pre-registration.

### SPEC 36 scored (2026-11-10 17:45 IST; `probe_ext_clusters.py`, s43 frozen weights, `'onsite'`; output in `runs/extensivity_2026-11-10/clusters_onsite_s43.txt`, `spec36_redox_suite.txt`, `spec36_rest_suite.txt`): PASS on (a)–(d)

| criterion | pre-registered | measured | |
|---|---|---|---|
| (a) | new tests pass; `tests/redox` 830 + new, none changed; rest 1045 passed, 4 skipped | 33 new tests pass; `tests/redox` 863 passed (220 s); rest 1045 passed, 4 skipped (309 s) | PASS |
| (b) wetting 1×1 | \|E − E_global\| ≤ 1e-9 eV, max\|ΔF\| ≤ 1e-8 eV/Å | 0.000e+00 eV, 2.2e-16 eV/Å | PASS |
| (b) non-wetting 1×1 | the same | 0.000e+00 eV, 0.000e+00 eV/Å | PASS |
| (c) wetting 2×1 | +5.636 ± 0.010 meV | +5.6358 | PASS |
| (c) non-wetting 2×1 | −22.903 ± 0.010 meV | −22.9020 | PASS |
| (d) wetting 2×2 | +2.78 ± 1.0 meV | +2.7852 (global: +220.8418) | PASS |
| (d) non-wetting 2×2 | −38.02 ± 1.0 meV | −38.0254 (global: +99.8349) | PASS |
| (d) max\|ΔF\|, four rows | ≤ 2e-02 eV/Å | 1.21e-02, 1.25e-02, 7.86e-03, 1.23e-02 (global 2×2: 1.84e-01, 1.94e-01) | PASS |

- **Reported.** Wetting: one cluster of 3 configurations on units [108, 109] in the 1×1 cell; two clusters of 3 (pair
  space 9) at 2×1; four clusters of 3 (six pair spaces of 9) at 2×2. Non-wetting: one cluster of 5 on units
  [1, 108, 109, 110] at 1×1; two clusters of 4 (pair space 16) at 2×1; four of 4 (six pairs of 16) at 2×2: the
  configuration that moves an electron from unit 108 to the band unit is not selected in the replicated cells (the
  band unit is one per structure, L3; its 1×1 weight is +0.0010 meV, R13). Dropped three- and four-cluster terms,
  measured as clusters − exact unswitched product at 2×2: +0.006 meV per cell (wetting), −0.004 (non-wetting); the
  ± 1.0 meV of (d) was a loose estimate. Time: 1×1 1–2 s, 2×1 3 s, 2×2 11 s (global: the same).
- **Not run.** The doped cells; the doped 2×2 refusal's falsifying check; a 3×3 cell.
- **What is left of the replication error** (+5.6 / +2.8 meV per cell wetting, −22.9 / −38.0 non-wetting, max|ΔF|
  1.2e-02 eV/Å) is L4, the 1×1 cell's own finite-cell term: it is identical in the exact unswitched product space
  (R13), so no selection or switch can remove it.

**Record of test changes (per the project's test-driven policy).** Before any code: P4's order restated (third → second, 17:10 IST). After the code
existed, two guards of the new test file were changed, and are recorded here as such: P3's guard that the pair
increment is not vacuous, from `> 1e-3` eV to `> 1e7 ×` the test's tolerance, on a polar and a non-polar
parametrisation; P5's derivative checks moved to polar dimers. No assertion of an identity was loosened. Measured:
P2 additive to ≤ 5.33e-15 eV up to 12 copies; P3 F − exact = 0.00e+00 in 12 cases; P4 the error falls by 4.16–4.03
per halving of the interaction with polar dimers (second order), 6.66–7.82 without (third order).
**A limit the pre-registration did not state.** The band unit is a reservoir shared by the clusters; the exchange
sign of two clusters' electrons in it is dropped (reservoir units are left out of the supports).

### R14. A defect of SPEC 36 as built, found after the score: one merged cluster lacks the products of its parts (measured 2026-11-10; tensors and a 6-atom random-weight module; not on the benchmark)

L2 was stated as "not smooth at a merge". Measured, it is a jump of the whole disconnected-product energy, and it is
also a static error of any cluster that holds two disjoint excitations.
- *Merge.* Two dimers of the test file (`_dimers(2)`, x 2.64, t 2.12 eV) with a coupling τ between unit 1 and unit 2:
  τ = 0.4325 eV gives two clusters [3, 3] and F = −3.908722 eV; τ = 0.4335 eV gives one cluster of 7 and
  F = −3.118966 eV: a jump of +0.790 eV. The global ensemble stays at −3.908722 eV. Cause: the linked rule refuses,
  from an excited configuration m, every move that touches no unit of supp(m); inside one cluster the product of two
  disjoint excitations is therefore never built (configuration interaction with singles of the blocks only).
- *Static.* Six atoms (Z = 79, 79, 12, 8, 12, 8; random weights, couplings × 3), one cluster: clusters − global =
  +0.168 eV (seed 3, Q 0; 23 against 29 configurations, the 6 missing ones all products of two disjoint transfers,
  largest weight 2.6e-02), +0.535 (seed 4), +0.073 (seed 5); forces differ by up to 0.74 eV/Å.
- *Not affected.* Every pre-registered identity (P1's system has three units, so every configuration is linked) and
  the benchmark (one dimer per cell, no configuration spans two cells: (b)–(d) above).
- *Consequence.* `diabatic_space='clusters'` is correct for clusters that stay apart and wrong where two blocks share
  a cluster. It stays an option, not the default. The repair is the product closure of each cluster's linked set
  with the SPEC 21 switch read from the linked parts ("C2b"); it moves no benchmark number, so it is queued behind
  the 1×1 finite-cell term (L4), which does.

### R15. What L4 is: the excitation interacting in phase with its own periodic images (measured 2026-11-10 evening; `probe_ext_image_terms.py`, s43 frozen weights, `'onsite'`, clusters space; output `runs/extensivity_2026-11-10/image_terms_onsite_s43_v2.txt`)

Benchmark number addressed: E/cell − E(1×1) of the undoped Table II cells and, through it, the Table II pair.
The cell is 9.0474 × 9.0474 × 26.4589 Å, periodic in all three directions; the wetting dimer lies along x, the
non-wetting one along z.

**1. The residual is one cluster's excitation energies, not the pair terms** (meV per cell):

| cell | wetting: E/cell − E(1×1) | = d(F_c − E⁰) | + pair increments | non-wetting: E/cell − E(1×1) | = d(F_c − E⁰) | + pair increments |
|---|---|---|---|---|---|---|
| 2×1 | +5.6358 | +4.5934 | +1.0425 | −22.9020 | −23.4825 | +0.5805 |
| 1×2 | −1.2143 | −1.4033 | +0.1890 | −22.9009 | −23.4813 | +0.5805 |
| 3×1 | +7.1191 | +6.0076 | +1.1116 | −24.6675 | −25.3461 | +0.6786 |
| 1×3 | −1.2266 | −1.4222 | +0.1956 | −24.6663 | −25.3448 | +0.6785 |
| 4×1 | +7.7285 | +6.5964 | +1.1321 | −24.8414 | −25.5383 | +0.6969 |
| 1×4 | −1.2277 | −1.4241 | +0.1964 | −24.8401 | −25.5369 | +0.6968 |
| 2×2 | +2.7852 | +1.4799 | +1.3053 | −38.0254 | −39.2855 | +1.2601 |

The reference energy E⁰ replicates exactly (the two parts sum to the total to the printed digits).

**2. Which terms of an excitation energy x(m) = H_mm − H_00 move** (m1 = the dimer transfer 108 → 109; meV,
against the 1×1 cell; configurations matched by content; the split of x into its terms closes to 1e-12 eV):

| cell | wetting dx | ladder | linear | self | pairs | SPEC 28 | non-wetting dx | ladder | linear | self | pairs | SPEC 28 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2×1 | +15.479 | +25.803 | 0.000 | +64.135 | −3.371 | −71.087 | −56.404 | −3.636 | 0.000 | −129.231 | −3.429 | +79.892 |
| 1×2 | −4.668 | +8.131 | 0.000 | −3.930 | −9.542 | +0.672 | −56.401 | −3.638 | 0.000 | −129.224 | −3.427 | +79.889 |
| 3×1 | +20.248 | +34.262 | 0.000 | +75.364 | −8.951 | −80.428 | −60.475 | −1.512 | 0.000 | −139.889 | −6.209 | +87.135 |
| 4×1 | +22.233 | +38.472 | 0.000 | +79.810 | −12.077 | −83.972 | −60.868 | +0.001 | 0.000 | −140.995 | −7.798 | +87.924 |
| 2×2 | +5.045 | +24.995 | 0.000 | +49.255 | −6.744 | −62.461 | −95.423 | −7.279 | 0.000 | −212.173 | −4.137 | +128.166 |

`linear` (Δn·(b + W n⁰), the excitation in the field of the unexcited crystal) does not move: the reference crystal
is the same in every cell. Everything of second order in the excitation moves: `self` (½ ΔnᵀWΔn with the dressed,
lattice-summed W), SPEC 28, and the one-body and pair terms built from the same periodic operators (`ladder`,
`pairs`: a term on "the atom and its own images" in the 1×1 cell is a pair term with an unexcited copy in the
replicated cell). The one-electron couplings of the dimer transfer do not move (−2.124730 and −2.641342 eV in every
cell).

**3. It is the lattice sum of the excitation's dipole** (`probe_ext_dipole_lattice.py`: tin-foil Ewald sum Λ of
parallel unit point dipoles on the cell's lattice; two Ewald splittings agree to 2e-14 Å⁻³; Λ_zz(1×1) =
+6.3959e-3 Å⁻³ equals the planar sum 9.0336/a³ minus 4π/V, and Λ_zz(8×8) = −0.2596e-3 equals the chain along z,
−4ζ(3)/c³). One constant per term, fixed on one cell, then predicted for the others:

- non-wetting `self`, constant from 2×2 (33.533 meV per 1e-3 Å⁻³, a dipole of 2.158 e·Å against 2.545 e·Å bare):
  2×1 −128.60 predicted, −129.23 measured; 3×1 −139.12 / −139.89; 4×1 −140.30 / −141.00 (0.5 %);
- non-wetting dx(m1), constant from 2×2 (written before the 3×1, 4×1, 1×2 cells were read): 2×1 −57.84 / −56.40,
  3×1 −62.57 / −60.48, 4×1 −63.10 / −60.87 (3 %);
- wetting `self`, an x and a z constant from 2×1 and 1×2: 3×1 +75.70 / +75.36, 4×1 +80.39 / +79.81,
  2×2 +47.70 / +49.26 (0.4–3 %);
- wetting dx(m1), two constants from 2×1 and 2×2 (written before the other cells were read): 1×2 −8.52 predicted,
  −4.67 measured; 1×3 −9.12 / −4.73. **Failed**: the sum of five terms that cancel to a tenth is not a point
  dipole's lattice sum at a/L = 0.28.

So L4 is the energy of a lattice of in-phase excitations, and its leading part is a dipole lattice sum; a
point-dipole (Makov–Payne-type) correction would remove the bilinear term to about 1 % and the whole to no better
than a few meV. That is not a solution.

**4. How fast it converges with the cell.** From Λ: of the 1×1 image term a 2×2 cell keeps 4.9 % (z dipole) or
12.5 % (x dipole), a 3×3 cell 0.4 % or 3.7 %, a 4×4 cell 0.04 % or 1.6 %. A 3×3 cell could not be measured: the dense
route refuses 990 atoms (10.25 GB operator, limit 4 GB) in both spaces.

**5. What is right.** In the thermodynamic limit the cells fluctuate independently: an excitation in one cell sees
the other cells in their mean state. The 1×1 cell at Γ instead excites every image in phase, and the energy of that
is ½ Σ_m p_m δn_mᵀ W_img δn_m with W_img = Σ_{R≠0} W(R) (and the same for every second-order term). DFT has no such
term: its energy of a replicated cell is exactly the replicated energy. The model trained on 1×1 cells has absorbed
it into its weights.

**The undoped pair in the 2×2 cell, frozen weights, on the reference geometries** (clusters space, per cell):
E(wetting) − E(non-wetting) = +907.126 meV in the 1×1 cell and +907.126 + 2.785 + 38.025 = **+947.94 meV** in the
2×2 cell. This is not Table II: Table II relaxes the Au positions (+892.68 meV at 1×1 under `'onsite'`; DFT +928.9),
and the relaxed 2×2 pair is not measured here (measured afterwards: R16, +935.11 meV for s43; statement replaced 2026-11-10). What the number says is the size of the term: 41 meV on a benchmark number
whose error is 36 meV, absorbed by weights trained on 1×1 cells.

**The doped cells** (`probe_ext_clusters.py … doped`, output `clusters_onsite_s43_doped.txt`; SPEC 36 "reported, not
criteria"): 1×1 clusters = global to 0 eV and 2e-16 eV/Å in both geometries (2 configurations, one cluster on units
108, 109). 2×1: E/cell − E(1×1) = +5279.61 meV (wetting) and +7047.33 meV (non-wetting), max |ΔF| 1.50 and
1.93 eV/Å, one configuration and no cluster, the same in both spaces to the printed digits: L3, the band unit, as
expected. 2×2: refused in both spaces (12 > 10 connections in the percolation ensemble). **L3 is two orders of
magnitude larger than L4 and comes first.**

### R16. L4 on the benchmark number: Table II's undoped pair with the fluctuations in a 2×2 Born–von Kármán cell (2026-11-10)

`probe_ext_bvk_table2.py` (outputs `runs/extensivity_2026-11-10/table2_bvk_onsite_s43.txt`, `…_s42.txt`). The 1×1 cell
is evaluated as its own 2×2 supercell (energy per cell, forces of the first copy), frozen weights, `'onsite'`,
clusters space, and Table II's protocol is run unchanged (Au relaxed to 0.005 eV/Å, substrate fixed). Checks of the
script: its 1×1 rows repeat the recorded numbers (+892.68 and +832.37 meV); the copies' forces agree to ≤ 7.4e-08 eV/Å.

| wet_u = E(wetting) − E(non-wetting), undoped, meV | s43 | s42 | mean | error of the mean |
|---|---|---|---|---|
| DFT | +928.9 | | | |
| 4G-HDNNP | +975.4 | | | +46.5 |
| `'block'`, 1×1 (as trained) | +920.36 | +850.97 | +885.67 | −43.2 |
| `'onsite'`, 1×1, frozen | +892.68 | +832.37 | +862.53 | −66.4 |
| `'onsite'`, 2×2 Born–von Kármán, frozen | **+935.11** | **+890.85** | **+912.98** | **−15.9** |

The relaxed geometries barely move (s43: Au–Au 2.5366 → 2.5385 Å wetting, 2.5153 → 2.5136 Å non-wetting; Au–O
unchanged to 4e-4 Å). The shift is +42.4 meV (s43) and +58.5 meV (s42), the same sign and size in both seeds: each
seed's 1×1 number carries its own in-phase image term, of the size of the benchmark error. **What is not shown:** the
weights were trained with the 1×1 cell at Γ and the block form, so this is a size of the term on frozen weights, not a
score; a 2×2 cell keeps an estimated 5–12 % of the 1×1 image term (R15, part 4), which is not measured here (3×3 is
refused by the dense route); wet_d and T5 need the doped supercell, which L3 forbids.

**The doped cell's reference configuration** (`probe_ext_band.py`, output `band_onsite_s43.txt`; 3 Al per 1×1 cell, so
three electrons beyond the formal frames). 1×1: O²⁻ ×54, Mg²⁺ ×51, Al³⁺ ×3, Au⁻ + Au⁰, band unit at −2; one cluster
(the transfer between the two Au). 2×1: six electrons; all four Au at −1 and the band unit still at −2, no cluster.
The band holds two electrons per STRUCTURE (SPEC 20: "at most two band electrons, both in φ₁"), so the replicated
cell puts the other two on the gold: that is the +5.3 and +7.0 eV per cell above.

### Pre-registered: SPEC 37, Born–von Kármán sampling of the configuration fluctuations (written 2026-11-10 18:44 IST, before the tests and the code)

**What it is for.** L4 (R15): at Γ a cluster's excitation interacts in phase with its own periodic images, so the
energy per cell depends on the cell chosen for the same crystal. Benchmark number moved: Table II wet_u (DFT +928.9,
4G +975.4; R16: +892.68 → +935.11 meV for s43 and +832.37 → +890.85 meV for s42 on frozen weights).

**Statement (NEW-TO-TRACE).** For a cell C and a mesh M = (M₁, M₂, M₃), M_d = 1 along a non-periodic direction,

    E_M(C) := E_Γ(S_M[C]) / |M|,        |M| = M₁ M₂ M₃,

where S_M[C] is the M-fold supercell (positions pos + R·cell, features and species repeated, total charge × |M|) and
E_Γ is the model as it stands, with the SPEC 36 cluster ensemble. In S_M the |M| copies of a cluster fluctuate
independently: an excitation meets its own images at M·a instead of a, and the other copies' clusters through
SPEC 36's increments instead of in phase.
The reference configuration's energy E⁰ does not depend on M (R15, measured); only the fluctuation terms do. M = (1,1,1)
is the model as shipped. `bvk_length` L chooses the mesh per structure, M_d = ceil(L / w_d) with w_d the cell's
perpendicular width along d, so that cells of the same crystal with equal M_d·w_d have the same energy per cell.

- NEW-TO-TRACE: the integer-charge configuration fluctuations of a thermal diabatic ensemble sampled on a
  Born–von Kármán cell that is decoupled from the simulation cell, with E⁰ exact at every M.
- ESTABLISHED ingredients: cyclic boundary conditions over a supercell (Born and von Kármán, Phys. Z. 13, 297 (1912))
  and their equivalence to a uniform mesh of Bloch vectors (Monkhorst and Pack, Phys. Rev. B 13, 5188 (1976)).
  Both citations are from memory and are to be verified before they enter SPEC.md.

**Exact properties, each a test written before the code** (`tests/redox/test_redox_bvk.py`):

- B1, the definition: the module with `bvk_mesh = M` on C equals the module at (1,1,1) on the supercell built
  independently (its own neighbour list), divided by |M|, to round-off; in a bulk cell and in a slab.
- B2, commensurate replication: E_M(S_P[C]) = |P| · E_{M∘P}(C).
- B3, extensivity at fixed L: with `bvk_length` chosen so that C takes M = (2,1,1) and S_(2,1,1)[C] takes (1,1,1),
  E_L(S[C]) = 2 E_L(C) to round-off.
- B4, forces: −∂E_M/∂pos through the tiling equals central finite differences of E_M.
- B5, identity: (1,1,1) with L = 0 returns the present energy bit for bit.
- B6, validation: positive integers only; a mesh > 1 along a non-periodic direction is refused; the option needs
  `fragment_ensemble='diabatic'` and `diabatic_space='clusters'`; `bvk_mesh` and `bvk_length` exclude each other;
  a checkpoint without the fields loads as (1,1,1), L = 0.
- B7, the record: per-atom fields (q, μ, …) are the home cell's, and Σq equals the cell's total charge.

**Pass criteria on the benchmark (frozen step 31 weights, `'onsite'`, clusters space, Table II's protocol).**

- P1. `bvk_mesh = (2,2,1)` on the 1×1 cell, with no supercell built outside the module, gives wet_u = +935.11 ± 0.01
  meV (s43) and +890.85 ± 0.01 meV (s42), the numbers of R16.
- P2. `bvk_length = 18.0` Å at the two undoped reference geometries (s43): |E_L(2×2)/4 − E_L(1×1)| ≤ 1e-3 meV and
  max|ΔF| ≤ 1e-6 eV/Å. (At (1,1,1) the same differences are +2.79 and −38.03 meV; SPEC 36 score.)
- P3. Defaults unchanged: `tests/redox` 863 passed plus the new tests; the rest 1045 passed, 4 skipped.

**Not claimed.** (i) Convergence in M: a 2×2 cell keeps an estimated 5–12 % of the image term (R15), and a larger mesh
is refused by the dense route here. (ii) Doped cells: the band unit holds two electrons per structure (L3), so a
supercell is not a replication there; the option refuses a structure whose band unit is occupied until C4 is built.
(iii) R14 (a merged cluster lacks the products of its parts) is untouched. (iv) Response properties
(`bound_state`, the SPEC 30 record) stay those of the Γ cell. (v) The mesh from `bvk_length` jumps when a width
crosses L / integer, as a k-point spacing does; derivatives are taken at a fixed mesh. (vi) Cost: the supercell is
evaluated whole (2×2: about 8.5× one cell at 110 atoms); the weights in use were trained at Γ, so P1 is a
reproduction of R16 through the package, not a new score. Training at a mesh is a separate, pre-registered step.

### SPEC 37 scored (2026-11-10 20:57 IST; `probe_ext_bvk_module.py`, step 31 frozen weights, `'onsite'`, clusters space; output in `runs/extensivity_2026-11-10/spec37_module_s43.txt`, `spec37_module_s42.txt`)

**PASS on P1, P2 and P3.** Code: `trace_redox/bvk.py` (`mesh_for`, `tile`), `RedoxConfig.bvk_mesh` / `bvk_length`,
`RedoxResult.bvk_mesh`, `RedoxElectrostatics.forward` → `_forward_cell` / `_forward_bvk`. Tests:
`tests/redox/test_redox_bvk.py`, 28 (B1–B7 and the band refusal), written red first (27 failed before the code).

| criterion | pre-registered | measured |
|---|---|---|
| P1, s43: wet_u, Table II protocol, 1×1 cell, `bvk_mesh = (2,2,1)` | +935.11 ± 0.01 meV | **+935.1123 meV** |
| P1, s42 | +890.85 ± 0.01 meV | **+890.8461 meV** |
| P2, s43, `bvk_length = 18.0` Å: E_L(2×2)/4 − E_L(1×1), wetting / non-wetting | ≤ 1e-3 meV | **+1e-6 / < 1e-6 meV** |
| P2, s43: max\|ΔF\|, wetting / non-wetting | ≤ 1e-6 eV/Å | **5.6e-9 / 3.4e-8 eV/Å** |
| P3: `tests/redox` | 863 + the new tests | **891 passed** (863 + 28) |
| P3: `pytest tests --ignore=tests/redox` | 1045 passed, 4 skipped | **1045 passed, 4 skipped** |

- The probe's own check: at `bvk_mesh = (1,1,1)` it returns +892.6778 (s43) and +832.3650 meV (s42), eval_aumgo's
  numbers of R16's 1×1 rows. The mesh is read off the module's record (`RedoxResult.bvk_mesh`) at every call; the
  1×1 cell is the only structure handed to the calculator in P1.
- P2's control, same script: at the unit mesh the same pair of cells differs by **+2.785216** (wetting) and
  **−38.025424 meV** (non-wetting) per cell, max|ΔF| 1.25e-2 and 1.23e-2 eV/Å (the SPEC 36 score, reproduced).
  With `bvk_length = 18.0` Å the module chose (2,2,1) for the 1×1 cell and (1,1,1) for its 2×2 replication.
- Table II, undoped, relaxed, frozen weights (DFT +928.9; 4G +975.4): s43 +892.68 → +935.11, s42 +832.37 → +890.85;
  mean error −66.4 → −15.9 meV. As pre-registered this is R16 reproduced through the package, not a new score:
  the weights were trained at Γ.
- Round-off in the unit tests (random heads, 6-atom cells, energies 1–70 eV): module at a mesh against the
  independently built supercell ≤ 1.4e-14 eV; commensurate replication (B2) ≤ 1.6e-13 eV.
- Cost, measured in P1: a relaxation at (2,2,1) took 100–196 s against 11–18 s at the unit mesh (8–13 module calls
  each; the Mac was also running the test suites).

**Still open after this score** (none of it claimed above): convergence beyond 2×2; the doped cells (the band unit,
L3, and the 2^k percolation enumeration that refuses the doped 2×2 cell); R14; training at a mesh.

### R17. L3 measured: the band unit's spectrum, and the doped supercell is not representable (2026-11-10 ~21:00 IST; `probe_ext_band_spectrum.py`, s43 frozen weights, `'onsite'`, clusters space; output `runs/extensivity_2026-11-10/band_spectrum_s43.txt`, `band_spectrum_s43_2x2.txt`; no training, no package code changed)

SPEC 20's band unit holds at most two electrons, both in φ₁, in one unit per structure. Measured on the benchmark
cells (band factor 0.6665; levels are those of h = diag(ε) − τ above λ₁; E_det(2m) is the restricted frozen-orbital
determinant of the m lowest orbitals with the band's own interaction V, and m = 1 reproduces the module's
2λ₁ + J₁₁ to 6 decimals):

| cell | E/cell − E(1×1) (meV) | excess e⁻ | band charge | Au charges | λ₁ (eV) | J₁₁ (eV) | lowest levels above λ₁ (eV) |
|---|---|---|---|---|---|---|---|
| doped wetting 1×1 | 0 | 3 | −2.000 | −0.50 −0.50 | −3.4555 | 0.8648 | 0, 2.6547, 3.4646, 3.4689 |
| doped wetting 2×1 | **+5279.611** | 6 | −2.000 | −1.00 ×4 | −3.6011 | 0.4348 | 0, 1.1600, 1.1653, 2.6884 |
| doped non-wetting 1×1 | 0 | 3 | −2.000 | −0.53 −0.47 | −3.5406 | 0.8779 | 0, 2.6369, 3.4677, 3.4720 |
| doped non-wetting 2×1 | **+7047.327** | 6 | −2.000 | −1.00 ×4 | −3.6897 | 0.4412 | 0, 1.1609, 1.1664, 2.6693 |
| doped 2×2 (both) | **refused** | 12 | n/a | n/a | n/a | n/a | n/a |
| undoped wetting 1×1 | 0 | 0 | 0 | n/a | −0.2479 | 0.5218 | 0, 1.1518, 3.0181, 4.2087 |
| undoped wetting 2×2 | +2.785 | 0 | 0 | n/a | −0.4509 | 0.1273 | 0, 1.0671, 1.2539, 1.2577, 1.2602, 1.2640 |
| undoped non-wetting 2×2 | −38.025 | 0 | 0 | n/a | −0.5115 | 0.1332 | 0, 1.0702, 1.2550, 1.2564, 1.2620, 1.2634 |

1. **The doped 2×1 cell is wrong by 5.3 / 7.0 eV per cell.** Six excess electrons; the band takes two, so every Au
   is forced to −1 (Au₂²⁻ on both dimers). The band's own determinant energies per cell at 2×1 (wetting):
   E_det(2, 4, 6, 8)/2 = −3.3837, −4.8120, −5.2300, −4.1688 eV, i.e. the band would lower its energy by taking
   four to six electrons, which SPEC 20 does not allow.
2. **The doped 2×2 cell cannot be represented at all.** It is refused twice, independently: by the 2^k percolation
   enumeration (`FragmentEnsembleError`, 12 uncertain connections against the limit 10), and, with that enumeration
   bypassed in the probe's process, by the search: `ConfigurationSearchError: total charge 0 is outside the capacity
   [2, 1320]`. Twelve excess electrons, two in the band and at most one on each of the eight Au: ten can be placed.
   **A hypothesis of mine is falsified here**: I expected the cell to evaluate once the unused partitions were
   bypassed. It does not; the enumeration (C5) and the band's capacity (C4) are two separate blocks.
3. **The diabatic ensemble does not use the partitions.** With `percolation_partitions` replaced by an empty list
   the four 1×1 cells give dE = 0 exactly and max|ΔF| ≤ 4.4e-16 eV/Å.
4. **The band is metallic-like under replication.** λ₁ falls and J₁₁ halves with every doubling (0.8648 → 0.4348;
   0.5218 → 0.2567 → 0.1273), φ₁ spreads over the cell (participation 25.4 of 54, 50.3 of 108, 194.7 of 216), and
   the folded levels come in near-degenerate groups (5 meV pairs at 2×1; gaps of 0.32 and 0.38 meV at 2×2,
   undoped non-wetting). So neither one unit per orbital (singular orbital derivatives at the degeneracies) nor a
   zero-temperature closed-shell determinant (a partially filled near-degenerate shell at 2×2) is a usable form.

**What this says about the benchmark numbers.** wet_d (DFT −2.7 meV; step 31 s43 −75.7) and T5 (DFT −931.6; s43
−996.1) are evaluated on the doped 1×1 cell only, where the band holds exactly its cap of two and the Au pair holds
the third electron. They are numbers of one cell size, with no supercell or mesh to check them against: SPEC 37
refuses a band holding more than 0.5 e. Moving them needs an extensive band (C4 below).

### Pre-registered: C5, no 2^k percolation enumeration for the diabatic ensemble (written 2026-11-10 21:28 IST, before the test and the code)

**What it is for.** R17, point 2: one of the two refusals of the doped 2×2 cell. The benchmark number it serves is
wet_d / T5 at a supercell or mesh (not evaluable today); C5 alone moves no number, and C4 is still needed.

**The statement.** SPEC 12.6's bond-percolation partitions (2^k patterns of the k uncertain connections) belong to the
two-level and exact fragment ensembles. The diabatic ensemble (SPEC 17, the trained model's) carries integer charges
on every atom and never reads them: R17 point 3, dE = 0 exactly with the partitions emptied. With
`fragment_ensemble='diabatic'` the module therefore does not enumerate; the bond graph's coverage check
(`chemical_bonds`, "attention_cutoff") stays. ESTABLISHED: nothing new; a refusal with no physical content removed.

**Criteria** (tests in `tests/redox/test_redox_percolation_diabatic.py`, written red first):
- C5-a: a 6-atom system with 2 independent uncertain connections and `fragment_max_uncertain = 0` evaluates in the
  diabatic ensemble (refused today), with energy and forces bitwise equal to those at the default limit.
- C5-b: the 'exact' and 'two-level' ensembles still refuse the same system at the same limit.
- C5-c: the diabatic ensemble still refuses a bond graph the neighbour list does not cover.
- C5-d: `tests/redox` 891 + the new tests passed; `pytest tests --ignore=tests/redox` 1045 passed, 4 skipped.
- C5-e (benchmark cell): the doped 2×2 cell's refusal changes from `FragmentEnsembleError` to the capacity refusal
  `ConfigurationSearchError ... [2, 1320]` without any bypass in the probe; the four 1×1 energies are unchanged
  (|dE| ≤ 1e-9 eV).

**Not claimed.** The doped 2×2 cell does not evaluate after C5; that is C4.

### C5 scored (2026-11-10 21:39 IST; output in `runs/extensivity_2026-11-10/c5_redox_suite.txt`, `c5_rest_suite.txt`, `band_arrays_s43.txt`): PASS on C5-a to C5-e

Code: `trace_redox/module.py`, `_forward_cell` (the partitions are an empty list when `fragment_ensemble='diabatic'`;
`chemical_bonds` and its coverage check are still called). Tests: `tests/redox/test_redox_percolation_diabatic.py`,
5 (1 failed before the code: C5-a; the 4 controls passed before and after).

| criterion | pre-registered | measured |
|---|---|---|
| C5-a | evaluates at the limit 0, bitwise equal to the default limit | **pass** (refused before the fix) |
| C5-b | 'exact' and 'two-level' still refuse | **pass** |
| C5-c | the coverage check still refuses | **pass** |
| C5-d: `tests/redox` | 891 + the new tests | **896 passed** |
| C5-d: `pytest tests --ignore=tests/redox` | 1045 passed, 4 skipped | **1045 passed, 4 skipped** |
| C5-e: doped 2×2, no bypass (`probe_ext_band_arrays.py`) | `ConfigurationSearchError ... [2, 1320]` | **that refusal, both geometries** |
| C5-e: the 1×1 energies, s43 | \|dE\| ≤ 1e-9 eV | **0 at the 9 printed decimals** (doped −1480190.920517170 / −1480190.854570557; undoped −1476722.802529320 / −1476723.709655625 eV, against the values stored before the change) |

The doped 2×2 cell now reaches the configuration search (7 s) and is refused there by the band's capacity: C4.

### R18. C4-0, the candidate many-electron band energy on the frozen band arrays (2026-11-10 21:41 IST; `probe_ext_band_arrays.py` → `runs/extensivity_2026-11-10/band_arrays_s43.npz`; `probe_ext_band_fermi.py` → `band_fermi_s43.txt`; s43, kT = 0.1 eV; no package code used or changed by the second probe)

Candidate (a hypothesis until it is built and tested; pre-registration below): frozen orbitals of SPEC 20's
h = diag(ε) − τ, Fermi–Dirac occupations per spin channel at the model's kT with N↑ = ⌈N/2⌉, N↓ = ⌊N/2⌋, first order
in the band's own interaction V,

    F_b(N) = Σ_σ [Σ_k f_k^σ λ_k − kT S_σ] + ½ Σ_uv [ρ_u ρ_v − Σ_σ (P^σ_uv)²] V_uv.

| cell | F_b(1) − λ₁ (eV) | F_b(2) − (2λ₁ + J₁₁) (eV) | exp(−gap/2kT) | F_b/copy at 2 e⁻ per copy (eV) | lowest F_b at |
|---|---|---|---|---|---|
| doped wetting 1×1 | +5.9e-7 | +3.4e-7 | 1.7e-6 | −6.0461 | N = 2 |
| doped wetting 2×1 | +7.4e-4 | +1.5e-3 | 3.0e-3 | −4.7958 | N = 6 (3 per copy) |
| doped wetting 2×2 | −6.9e-5 | −1.2e-4 | 3.1e-3 | −4.1908 | N = 10 (2.5 per copy) |
| doped non-wetting 1×1 | +6.7e-7 | +4.1e-7 | 1.9e-6 | −6.2034 | N = 2 |
| doped non-wetting 2×1 | +7.7e-4 | +1.6e-3 | 3.0e-3 | −4.9513 | N = 6 |
| doped non-wetting 2×2 | −5.0e-5 | −8.6e-5 | 3.0e-3 | −4.3473 | N = 10 |
| undoped wetting 1×1 | −7.9e-4 | −2.3e-3 | 3.2e-3 | +0.0238 | N = 1 |
| undoped non-wetting 1×1 | −7.0e-4 | −2.1e-3 | 3.1e-3 | −0.0571 | N = 1 |

1. **Reduction to SPEC 20 is to exp(−gap/2kT), not to round-off.** With grand-canonical occupations at fixed mean
   filling the leak into the next level is exp(−gap/2kT): 3e-7 to 7e-7 eV on the doped 1×1 cells (gap 2.65 eV),
   1 to 2 meV on a band state above a 1.15 eV gap. (My earlier estimate exp(−gap/kT) was the canonical one and is
   withdrawn.) The new form is therefore an option beside SPEC 20, not a silent replacement.
2. **At Γ the doped band's energy is not converged in the cell size, at the eV scale.** At two band electrons per
   cell copy, F_b per copy is −6.0461 (1×1), −4.7958 (2×1), −4.1908 eV (2×2) for the wetting geometry: +1.86 eV per
   cell from 1×1 to 2×2, and the last step is still +0.60 eV. This is band filling: at 1×1 both electrons sit at
   the band bottom, in a supercell the folded levels 1.07 to 1.26 eV higher must be filled. Step 31's weights were
   trained with the 1×1 Γ value, so the doped benchmark numbers (wet_d, T5) rest on a one-point sampling of a
   band about 1.2 eV wide over the first folded shell. Not measured: 3×3 and the converged value.
3. **A Chebyshev expansion of the Fermi operator reaches round-off** against the eigendecomposition: max|ΔP| =
   1e-14 at order 800 for the doped cells (Gershgorin half-width 10.5 eV; 6e-9 at order 400), 7e-15 at order 400
   for the undoped (7.8 eV). It is a polynomial in h, so it has derivatives of every order at degenerate levels,
   which autograd through an eigendecomposition does not.

### Pre-registered: C4, the Fermi band (SPEC 38; written 2026-11-10 21:43 IST, before the tests and the code)

**What it is for.** L3 (R17, R18): SPEC 20's band holds two electrons in one orbital, so a doped supercell is wrong by
eV (2×1) or not representable (2×2), and SPEC 37 has to refuse a filled band. Benchmark numbers it serves: wet_d
(DFT −2.7 meV; 4G off by 38.3; step 31 s43 −75.7) and T5 (DFT −931.6; s43 −996.1), which today exist at one cell
size only. C4 makes them evaluable at a supercell or mesh; it does not by itself move them at 1×1 Γ (R18, point 1).

**The statement.** The band unit holds any number N of electrons, 0 ≤ N ≤ 2B (B band sites):

    F_b(N) = A(N↑) + A(N↓) + ½ Σ_{u≠v} V_uv [ρ_u ρ_v − Σ_σ (P^σ_uv)²] + Σ_u U_u ρ↑_u ρ↓_u,
    A(n) = μ n − kT tr ln(1 + e^{−(h − μ)/kT}),   P^σ = [1 + e^{(h − μ_σ)/kT}]⁻¹,   tr P^σ = N_σ,
    N↑ = ⌈N/2⌉, N↓ = ⌊N/2⌋,   ρ^σ = diag P^σ,   ρ = ρ↑ + ρ↓,

with h = diag(ε) − τ, V and U = diag V those of SPEC 20, and kT the model's. The band's charge on site v in the state
N is −ρ_N(v) (a state-dependent shape; SPEC 20's fixed φ₁² is its N ≤ 2 limit).
- NEW-TO-TRACE: an N-electron Fermi reservoir as one unit of the integer-charge configuration ensemble, every
  filling a state of that unit with its own charge shape ρ_N, coupled to the other units' exact polarisable
  configuration energies; the fixed-filling free energy and all projectors from one set of Chebyshev moments
  tr T_k(h), with μ(h) carried by two Newton steps on those moments (derivatives exact to third order at any
  degeneracy).
- ESTABLISHED: Fermi–Dirac occupations as the minimiser of the finite-temperature functional (Mermin, Phys. Rev.
  137, A1441 (1965)); the first-order (Hartree–Fock) energy of a Hubbard / Pariser–Parr–Pople interaction in a
  frozen determinant (textbook; no single citation claimed); the Chebyshev expansion of the Fermi operator
  (Goedecker and Colombo, Phys. Rev. Lett. 73, 122 (1994); bibliographic record verified, the letter's equations
  not read here).

**Stage 1 (this entry's code): the functional alone, `trace_redox/fermi.py`, nothing else changed.** Criteria
(tests in `tests/redox/test_redox_fermi_band.py`, written red first):
- F1: tr P = N_σ to round-off; P against the eigendecomposition to 64·order·ε_machine.
- F2: gap ≫ kT (≥ 10 eV at kT = 0.1): F_b(0) = 0, F_b(1) = λ₁, F_b(2) = 2λ₁ + J₁₁ (SPEC 20) to 1e-11 eV.
- F3: the full band (P = 1 in both channels): F_b(2B) = 2 tr h + 2 Σ_{u≠v} V_uv + Σ_u U_u to round-off (sums over
  ordered pairs, as in the statement: two electrons on each of two sites interact with 4 V_uv per unordered pair.
  This line was first written with a factor 1; corrected 21:48 IST, before any test or code existed).
- F4: first derivatives of F_b with respect to h and V against central differences at an exactly degenerate
  spectrum with the Fermi level inside the degenerate shell (relative 1e-7).
- F5: second derivatives (a gradient of a gradient) against central differences of the first, same system.
- F6: two uncoupled copies at twice an even filling: F_b[h⊕h, V⊕V](2N) = 2 F_b[h, V](N) to round-off.
- F7: an infinite U_u is accepted iff N↓ = 0 and refused otherwise.
- F8: on the saved s43 band arrays (R18) the module reproduces `probe_ext_band_fermi.py`'s F_b(N) to 1e-9 eV.
- F9: `tests/redox` 896 + the new tests; `pytest tests --ignore=tests/redox` 1045 passed, 4 skipped.

**Stage 2 (its mechanism is fixed in an addendum before its code): the unit.** `RedoxConfig.band_form = 'fermi'`
beside the legacy default `'phi1'`. Pre-registered criteria on the benchmark cells, s43 frozen weights:
- U1: doped 1×1 at Γ, 'fermi' against 'phi1': |ΔE| ≤ 1e-4 eV (R18: the leak is 3e-7 to 7e-7 eV per band state).
- U2: the doped 2×1 and 2×2 cells evaluate; forces equal −∇E by central differences (1e-6 relative on 3 components).
- U3: with `bvk_length = 18.0` Å, E(2×2)/4 − E(1×1) ≤ 1e-3 meV and max|ΔF| ≤ 1e-6 eV/Å on both doped geometries
  (SPEC 37's P2, now on the doped cells; SPEC 37's band refusal lifted for 'fermi').
- U4: a double backward (the force loss's gradient) runs on a doped cell and matches central differences.
- U5: both suites.
Reported, not criteria: wet_d and T5 at Γ and at the (2,2,1) mesh on the frozen weights. They were trained at Γ with
the two-electron band, so R18 point 2 predicts eV-scale shifts of the absolute energies; the differences are not
predicted here.

**Not claimed.** (i) Self-consistent orbitals: the orbitals are those of h (as in SPEC 20). (ii) Spin polarisation:
the lowest |S_z| is taken, so an odd filling per cell in a supercell is not the ferromagnetic state. (iii) The
exchange sign of electrons placed in the band from different clusters (SPEC 36's stated limit). (iv) Convergence
in the mesh (R18: +0.60 eV per cell in the last doubling). (v) Any benchmark number until a model is trained with
the form; a training run is a separate step and needs its own word.

### C4 stage 1 scored (2026-11-10 22:20 IST; `trace_redox/fermi.py`, `tests/redox/test_redox_fermi_band.py`; output in `runs/extensivity_2026-11-10/fermi_module_s43.txt`, `c4s1_redox_suite.txt`, `c4s1_rest_suite.txt`, `band_onsite_s43.txt`): PASS on F1–F9

Stage 1 is the functional alone. No model, no benchmark number is touched by it: the module does not import
`fermi.py` yet. The residuals below are measured (the tests assert the bounds; the numbers are from one run on
the Mac, torch 2.2.0).

| | criterion | measured | |
|---|---|---|---|
| F1 | tr P = n; P against the eigendecomposition to 64·order·ε (6.0e-12, 6.4e-12 at orders 421, 452) | polynomial route: max\|tr P − n\| 2.0e-13, max\|P − P_ref\| 3.1e-14, max\|A − A_ref\| 8.9e-15 eV; eigenvector route: 3.6e-15, 1.9e-15, 8.9e-15 | PASS |
| F2 | 12 eV gap: F_b(0) = 0, F_b(1) = λ₁, F_b(2) = 2λ₁ + J₁₁ to 1e-11 eV | 0 exactly; +1.8e-15; +1.6e-15 eV (eigenvector route −2.0e-14, −4.1e-14); max\|ρ₁ − φ₁²\| 2.1e-14 | PASS |
| F3 | full band: 2 tr h + 2 Σ_{u≠v} V_uv + Σ_u U_u | +7.1e-15 eV on 61.79 eV | PASS |
| F4 | first derivatives at a degenerate Fermi level, relative 1e-7 | six-ring, N = 3 and 4 (μ inside the doubly degenerate shell): ∂F/∂h −0.322563465 (difference −1.8e-11), +0.498476380 (−1.9e-12); ∂F/∂V −0.725191944 (−2.2e-13), −1.448406129 (+1.3e-13) | PASS |
| F5 | second derivatives, same system (the test asserts relative 1e-7) | ∂²F/∂h² +1.172244504 (+4.4e-11), +0.556711507 (+1.3e-9); ∂²F/∂h∂V +0.062499745 (+8.6e-13), +0.089632518 (−6.2e-11) | PASS |
| F6 | two uncoupled copies at twice an even filling | N = 2, 4, 6: 0, +7.1e-15, −1.8e-15 eV (eigenvector route ≤ 1.4e-14) | PASS |
| F7 | infinite U accepted iff N↓ = 0 | N = 0, 1 finite and bit-identical to the finite-U value; N = 2 raises `ValueError` | PASS |
| F8 | the saved s43 band arrays against `probe_ext_band_fermi.py` to 1e-9 eV | against the probe's own function, all 12 cells, N = 0..4 per copy: polynomial route ≤ 3.6e-12 eV, eigenvector route ≤ 6.0e-14 eV | PASS |
| F9 | suites | `tests/redox` 917 passed (896 + 21); `pytest tests --ignore=tests/redox` 1045 passed, 4 skipped | PASS |

**Changes to the tests after they were first written red** (all before the score; none loosens a pre-registered
bound). F1 and F8 were made to run both routes of the module (the polynomial in h and the eigenvectors). F4's
control assertion "μ equals the degenerate level" was first written to 1e-9 eV and was wrong: μ sits 9.08e-6 eV
above the level because the level below it holds a hole of e^{−10}; the control now asks 1e-3·kT. F4's first
`autograd.grad` needed `retain_graph=True`. One test was added that is not pre-registered
(`test_derivatives_of_a_gapped_band`: where the count does not respond to μ, μ is held constant). F8's test compares
with a numpy eigendecomposition written in the test file; the comparison with the probe's own function was run
separately (the row above).

**One error of the first implementation.** `torch.nn.functional.softplus` is linear above its threshold of 20 and
drops e^{−20} = 2e-9, which put up to 1.2e-10 eV into A. `torch.logaddexp` replaces it.

**What the falsifying checks found (per the rule-out-artifacts policy), and the two constructions that came out of it.**
- Mutations: with μ held constant the first derivatives are wrong by −6.2 and −19 (F4 catches it); with one Newton
  step instead of two the second derivatives are wrong by +4.1e-3 and +2.5e-2 (F5 catches it).
- The first version passed F1–F8, and its second derivative was still not right: the residual against differences
  (−3.5e-8, −4.8e-8) did not shrink with the difference step, grew with the expansion order (to 7e-7), and reached
  6e-7 on a 40-site band at order 1218. Cause, measured: the Chebyshev coefficients carry an absolute round-off of
  ~1e-17 (f) to ~1e-15 (the series of A) at every k, and the p-th derivative of T_k grows as k^{2p} at the ends of
  the interval. Differentiating A's own series twice cost about the order per derivative (2e-10 in the first,
  6e-8 to 1e-7 in the second at order ~1300).
- NEW-TO-TRACE, construction 1: A(n) takes its value from the detached eigenvalues and its gradient analytically,
  dA/dh = P (`_ValueWithGradient`), so no derivative of A's series is ever taken; the second derivative flows
  through P's own graph.
- NEW-TO-TRACE, construction 2: the spectral interval is widened by 5% (`SPECTRUM_MARGIN`; it was 0.1%), which
  keeps the spectrum away from the ends where T_k'' is largest. With margins ≥ 2% the second derivative agrees
  with differences to ≲ 1e-9.
- After both: the table's F5 row, and on the benchmark arrays below ≤ 1.0e-10.

**The module on the benchmark's own bands** (`probe_ext_fermi_module.py`, the saved s43 arrays, kT = 0.1 eV, N = 3
per copy for the derivatives; directional derivatives along fixed random symmetric directions):

| cell | sites | order | F_b, polynomial − reference (eV) | first derivative − differences | second − differences | energy + gradient / second backward (s) |
|---|---|---|---|---|---|---|
| doped wetting 1×1 | 54 | 769 | 5.2e-13 | +7.9e-12 | +9.6e-11 | 0.1 / 0.2 |
| doped wetting 2×1 | 108 | 771 | 3.5e-12 | +9.6e-12 | −1.0e-10 | 0.3 / 0.4 |
| doped wetting 2×2 | 216 | 809 | 1.1e-12 | +4.7e-12 | +3.2e-11 | 0.8 / 1.5 |
| doped non-wetting 2×2 | 216 | 806 | 2.6e-13 | +1.4e-11 | +8.9e-11 | 0.7 / 1.4 |
| undoped wetting 1×1 | 54 | 598 | 2.1e-12 | −4.7e-12 | −8.6e-11 | 0.2 / 0.3 |
| undoped wetting 2×2 | 216 | 612 | 4.6e-13 | +1.3e-11 | −2.3e-11 | 0.6 / 1.2 |

The cost is that of ~800 matrix products of the band's dimension; on a 2×2 cell it is small beside the rest of the
model's evaluation (R16: 93–145 s for a relaxation of 8–13 evaluations).

**A property of the trained arrays, measured while doing this (`probe_ext_band_onsite.py`, s43, 1×1; output
`band_onsite_s43.txt`).** The band sites' second-electron cost U_v (the diagonal of SPEC 20's V) is negative on every
band site of all four benchmark cells:

| cell | sites | ladder second difference e(f−2) − 2e(f−1) + e(f) | W_vv | pair tables | U_v (eV) |
|---|---|---|---|---|---|
| doped wetting | Mg (51) | −10.464 .. −9.632 | +11.286 .. +11.985 | −5.589 .. −4.548 | −4.214 .. −2.578 |
| doped wetting | Al (3) | −14.681 .. −14.680 | +18.128 .. +18.129 | −7.366 .. −7.362 | −3.918 .. −3.915 |
| doped non-wetting | Mg (51) | −10.463 .. −9.633 | +11.304 .. +12.053 | −5.592 .. −4.548 | −4.200 .. −2.522 |
| doped non-wetting | Al (3) | −14.682 .. −14.680 | +18.142 .. +18.143 | −7.366 .. −7.362 | −3.904 .. −3.901 |
| undoped wetting | Mg (54) | −10.036 .. −9.793 | +11.816 .. +12.259 | −5.590 .. −4.636 | −3.572 .. −2.414 |
| undoped non-wetting | Mg (54) | −10.207 .. −9.793 | +11.767 .. +12.268 | −5.593 .. −4.636 | −3.628 .. −2.405 |

Read strictly: at these weights two carriers on one cation cost 2.4 to 4.2 eV less than on two cations, before the
inter-site interaction. The ladder and W_vv nearly cancel (by the ranges above their sum lies between +0.8 and +3.5 eV); the pair
tables' second difference (−4.5 to −7.4 eV) makes the total negative. My first guess, that the bound sector over-screens through
W_vv, is refuted by the table: W_vv is +11 to +18 eV. Not done: which pair table carries the −5 eV, and whether
the training set constrains these states at all (the 1×1 band energy sees U only through J₁₁ = Σ ω_u ω_v V_uv,
which is +0.865 eV on the doped wetting cell). It is recorded here because the Fermi band's U_u ρ↑_u ρ↓_u term
uses these numbers at every filling above one electron per channel; nothing in stage 1 depends on their sign.

**Erratum, clock stamps.** Five stamps of this section were written from an estimate and ran ahead of the clock
(found when `date` gave 22:11 IST and the pre-registration claimed 22:40). They are corrected in place from the
session transcript's own timestamps (UTC + 5:30): C5 pre-registered 21:35 → 21:28; C5 scored 22:05 → 21:39; R18
22:20 → 21:41; C4 pre-registered 22:40 → 21:43; the F3 correction 22:50 → 21:48. The order claimed is the order in
the transcript: C4's pre-registration 21:43, the F3 correction 21:48:15, the first test file 21:48:38, the first
`fermi.py` 21:51. The earlier stamps of the day agree with the transcript to within 7 minutes.

**Not done.** Stage 2 (the unit): nothing of it exists. No benchmark number has moved.

### C4 stage 2, the mechanism (addendum, 2026-11-10 22:35 IST; written before any stage 2 test or code)

**Measured first** (frozen weights, 1×1 benchmark cells, 'onsite' stiffness, clusters space; the legacy band;
`probe_ext_band_coupling.py`, `probe_ext_band_pairs.py`; output `runs/extensivity_2026-11-10/band_coupling_s43.txt`,
`band_coupling_s42.txt`, `band_pairs_s43.txt`).

1. What the atom–band couplings carry. SPEC 20 gives the band unit the couplings T_band[u] = Σ_v T_uv φ₁(v). With
   that row and column of T set to zero in this process:

   | cell | seed | max\|T_band\| (eV) | ΔE (meV) | max\|ΔF\| (eV/Å) |
   |---|---|---|---|---|
   | doped wetting | s43 / s42 | 0.158 / 0.263 | 0 / 0 | 4e-16 / 2e-16 |
   | doped non-wetting | s43 / s42 | 0.158 / 0.263 | 0 / 0 | 0 / 0 |
   | undoped wetting | s43 / s42 | 0.081 / 0.186 | 0 / 0 | 0 / 0 |
   | undoped non-wetting | s43 / s42 | 0.396 / 0.230 | +0.00097 / +0.00215 | 4.8e-05 / 1.2e-04 |

   The zeros are exact, and the wrapper is effective (the last row responds). Why six of eight are exactly zero is
   not established: that SPEC 21's switch turns these couplings off is a hypothesis, not checked.
2. The state axis of these cells is −2 … +3 (six states: O's second-electron state to `max_charge`), not −3 … +3.
   The legacy band unit's bounds are L = −2, K = 0.
3. Every pair table vanishes at its frame pair: max|Ξ_p(f_i, f_j)| = 0 exactly on all four cells (16,335–16,361
   tables, 6×6). SPEC 20's d_ε uses this.

**The unit** (`RedoxConfig.band_form = 'fermi'`; the legacy `'phi1'` stays the default and is not touched).

- **States (NEW-TO-TRACE: a searched reference filling as the origin of the unit's state label).** The band unit's
  state s on the common state axis means the filling N = N₀ − s; its charge is −N. The solver is given the total
  Q + N₀. Bounds: K_b = min(last state, N₀), L_b = max(first state, N₀ − N_max), with N_max = 2B when every on-site
  U is finite and 1 otherwise (stage 1's F7). On these cells the unit therefore spans N₀ − 3 … N₀ + 2.
- **Energy of a configuration** (n the atoms' integer charges, band sites at or above their frame as in SPEC 20):

      E(n, N) = E_at(n) + F_b(N) + Σ_v ρ_N(v) c_v(n),
      c_v(n)  = − Σ_u W_vu (n_u − f_u) + Σ_{u ∉ sites} δ_uv(n_u),

  E_at the atoms' configuration energy (ladder, b, W, pair tables), F_b and ρ_N stage 1's functional with SPEC 20's
  ε, V, τ, and δ_uv SPEC 20's pair-table change (`_band_pair_corrections`). With ρ_m = m φ₁² and F_b(m) = E_m this
  is SPEC 20's energy term by term. NEW-TO-TRACE: the filling-dependent shape ρ_N coupled to the atoms' exact
  configuration energies. ESTABLISHED: a charge density in the potential of the other charges, to first order
  (textbook electrostatics; no citation claimed).
- **Assembly.** The band unit's entry of b and its row and column of W are zero. Its ladder row holds F_b(N(s))
  (+∞ outside [L_b, K_b]). One pair table per atom u with the band unit:
  Ξ_(u,band)[k, s] = −(k − f_u) Σ_v W_uv ρ_{N(s)}(v) + Σ_v δ_uv(k) ρ_{N(s)}(v) (zero in the columns outside
  [L_b, K_b]). The search, the linked selection, the cluster increments and the solver are not changed: they already
  take a ladder and pair tables.
- **Reference filling.** Start N₀ = clip(N_x, 0, N_max), N_x = Σ_u f_u − Q. Detached passes: assemble at N₀, find
  the lowest configuration with the solver's own search (window 0, same bounds, units and pair tables), read the
  band's state s_ref there, set N₀ ← N₀ − s_ref; stop at s_ref = 0. The reference energy cannot rise from one pass
  to the next (the new window contains the configuration just found), so the passes end; a revisited N₀ (an exact
  tie) ends them at the smaller N₀; more than 64 passes is refused.
  Identity (to be tested): E(n, N) does not contain N₀, so the result is the same for any N₀ whose window holds
  every selected configuration.
- **Charges.** A configuration's atom charges are n_v − ρ_{N(s)}(v) on band sites and n_u elsewhere. They replace
  `configs @ omega` in SPEC 28's correction, SPEC 30's response and the free dipoles. The reported atom charges are
  the ensemble's expectation of the same expression; the band's state populations are accumulated with the graph by
  the increments that give the unit charges (clusters) or from ρ's diagonal (global space). The band unit's reported
  charge is −⟨N⟩. `RedoxResult.omega` is None for this form.
- **Couplings.** The band unit's row and column of T are zero: it exchanges electrons with the other units in the
  ensemble (thermally), without a coupling amplitude. Reasons: (i) item 1 above; (ii) a thermal band with a
  degenerate frontier, which translation symmetry makes exact in every Born–von Kármán supercell, has no single
  frontier orbital, and Σ_v T_uv φ(v) is not a function of the projector; (iii) one amplitude per unit to a band
  state normalised over M cells is of order 1/√M, so the energy it gives is not extensive. The extensive form is a
  second-order self-energy over all band orbitals; it is its own step and is not built here.
- **Born–von Kármán sampling.** `_forward_bvk`'s refusal (more than 0.5 e in the band unit) is lifted for 'fermi'
  and kept for 'phi1'.

**Tests, written red before the code** (`tests/redox/test_redox_fermi_band_unit.py`):
- T1: on random arrays with pair tables, the assembled (ladder, pair tables) give E(n, N) equal to a reference built
  from atom-level `configuration_energies` alone: ε_v, V_vw, U_v and c_v(n) as first and second differences of
  E_at, F_b and ρ_N from `band_free_energy` on them (round-off).
- T2: the same energies for two different N₀ on the configurations both windows hold (round-off).
- T3: a gapped band at N₀ = 0: every configuration energy equals the legacy unit's (≤ 1e-9 eV at a gap ≥ 10 eV).
- T4: the atom charges of every configuration sum to Q, and are n_v − ρ_N(v).
- T5: module level on the doped rock-salt cluster of `test_redox_band.py` with three dopants: the model
  evaluates with more than two band electrons; free charges sum to Q; the reference has s_ref = 0; forces equal
  −∇E by central differences; a double backward runs and is finite.
- T6: `_forward_bvk` evaluates a doped periodic cell for 'fermi' and still refuses it for 'phi1'.

**Not claimed, in addition to stage 1's list.** (vi) The hybridisation of a localised level with the band (the
couplings above). (vii) Fillings outside N₀ − last … N₀ − first: a pair product that needs one is left out of its
pair space, as SPEC 36 already does at the legacy unit's bounds. (viii) Continuity when the selection changes N₀:
exact only under T2's condition. (ix) U1 compares the two forms on the doped 1×1 cell, where 'fermi'
admits N = 3 and 4 and the legacy unit (one orbital) does not. If those states carry more than 1e-4 eV, U1 fails as
written and the cause is reported; the criterion is not changed.

### R19. C4 stage 2, measured so far: U1 passes; U2 fails on the doped 2×1 cell, and the cause is R14, not the band (2026-11-11 07:30–07:45 IST; `probe_ext_fermi_unit.py`, s43 frozen weights, `'onsite'`, clusters space; output `runs/extensivity_2026-11-10/fermi_unit_U1_s43.txt`, `fermi_unit_U2_2x1_s43.txt`)

Benchmark numbers addressed: U2 / U3 on the doped cells, and through them wet_d (DFT −2.7 meV) and T5 (DFT −931.6
meV) at a Born–von Kármán mesh. No training; no package code changed after the stage 2 wiring.
- **Related test files after the wiring:** 130 passed (07:39 IST).
- **U1 PASS.** Doped 1×1 at Γ, 'fermi' − 'phi1': ΔE = +1.717e-06 eV (wetting), +1.641e-06 eV (non-wetting);
  max|ΔF| 9.9e-06 and 9.4e-06 eV/Å. The band holds 2.0000 e (N₀ = 2 in 2 passes); Au free charges −0.500 / −0.500
  (wetting), −0.531 / −0.469 (non-wetting).
- **U2, 2×1 only (the 2×2 not run).** Both doped 2×1 cells evaluate with 'fermi' (5 s; band 4.0000 e, N₀ = 4 in 2
  passes; 9 / 7 configurations). Forces against a 4th-order central difference of the energy (step 5e-3 Å, the
  atomic baseline left out of the differenced energy): second Au (atom 109, z) relative 1.8e-06 / 2.0e-06, first Al
  relative 2.0e-06 / 1.3e-06 (above the 1e-6 as written; whether that is the stencil's truncation is NOT verified);
  **first Au (atom 108, z): 0.127 eV/Å (wetting) and 2.6e-3 eV/Å (non-wetting). FAIL as written.**
- **Artifact check before the claim.** A scan of atom 108 along z (9 points, step 5e-3 Å): on each side of zero
  F = −dE/dR, and between 0 and +5e-3 Å the energy jumps by +1.34 meV while the Au charges flip between
  (−0.247, −0.753) and (−0.753, −0.247). The forces are right; the function is discontinuous.
- **Cause (measured).** In the clusters space the wetting 2×1 cell is ONE cluster of 9 configurations on Au108,
  Au109, Au218, Au219 and the band. Reference: Au109 and Au219 at −1. Above it: the two intra-dimer flips at
  +0.0672 eV, each coupled to the reference by −2.1237 eV; the two inter-dimer transfers Au109 → Au218 and
  Au219 → Au108 at +1.1979 eV, uncoupled (they enter by the 2.5 eV energy window and join the two dimers); Au → band
  (+1.4358 eV, two) and band → Au (+1.2378 eV, two). The product of the two flips is absent. Three states
  [[0, t, t], [t, 0, 0], [t, 0, 0]] give charges 0.25 / 0.75 and an error (2 − √2)·t = 1.243 eV at t = 2.12 eV.
  Measured: the global space on the same cell gives −1801.8091533 eV (baseline left out, 14 configurations, Au
  charges −0.4998 / −0.5002) against −1800.5659109 eV in the clusters space: **+1.243 eV**. This is R14 on the
  benchmark's doped supercell: R14's "not on the benchmark" held for the undoped cells only.
- **Consequence.** A Born–von Kármán mesh on the doped 1×1 cell builds this supercell, so U3 and wet_d / T5 at a
  mesh need the repair of R14 first. The global space is not a substitute (R13). U2's criterion is not changed.
- **Open, a hypothesis.** The global space on this cell shows a kink of about 1.3e-6 eV where the Au charges cross
  −0.5 (the lowest configuration changes). Cause not located; not checked on the 1×1 cell or with 'phi1'.
  (Falsified 2026-11-11: the kink is a knot of SPEC 33's wall table, R20.)

### Pre-registered: C2b, the product closure of the cluster ensemble (SPEC 36.7; written 2026-11-11 08:07 IST, before the tests and the code)

Benchmark numbers it moves: U2 and U3 of C4 stage 2 on the doped cells; wet_d and T5 at a mesh (reported after).
**Statement.** SPEC 36's model space is meant to hold every product of selected excitations on disjoint supports.
As built it holds them only across clusters. Two rules restore it:
1. **Closure (NEW-TO-TRACE).** A cluster's space is the product closure of its linked set S: every configuration
   n0 + Σ_{m∈I} (m − n0), I ⊂ S pairwise disjoint in support (reservoir units carry the sum), inside the bounds and
   admissible. Diagonal: the exact energy. Off-diagonal: the true one-electron elements of the closed set (signs
   and spin factors of the set itself), each multiplied by a switch read from the linked parts: for the move d → a
   between p and p′, σ = max of SPEC 21's switch of the same move between m and m′ in S, over all (m, m′) such that
   p = m ⊕ s with s in the closed set and supp(s) disjoint from supp(m) ∪ supp(m′); 0 when there is none. Elements
   between two members of S keep SPEC 21's switch on S. A spectator excitation therefore never weakens a coupling.
2. **Products do not join clusters (NEW-TO-TRACE).** A selected configuration joins the units of its support unless
   it is a product m₁ ⊕ m₂ of selected configurations with disjoint supports and T vanishes between the two
   supports; such a configuration is the product of two clusters' configurations (a transfer between clusters
   through a reservoir) and lives in their pair space. If the pair space of the clusters it spans does not hold it,
   or it spans more than two, it joins them after all (a fixed point: the flags only grow).
ESTABLISHED: the method of increments (citation not yet verified; none is added to the documents until it is);
size consistency of a product of disjoint excitations (the exponential ansatz of coupled-cluster theory, in words
only). Rejected: clusters from coupled moves alone (three degenerate uncoupled sites give −kT ln 4 against
−kT ln 3 with pair increments).
**Criteria** (tests red before the code; tolerances from machine epsilon unless stated):
- Q1: every identity of `test_redox_cluster_product.py` unchanged (P1–P5, the reservoir counts).
- Q2: R14's merge, `_dimers(2)` with τ between unit 1 and unit 2: F(τ = 0.4335) − F(τ = 0.4325) equals the global
  ensemble's difference to 1e-6 eV (as built: +0.790 eV).
- Q3: unswitched selection, one merged cluster of two dimers joined by an irreducible in-window transfer: F equals
  the global unswitched Hamiltonian on the closed set (round-off), and the closed set holds the 3 × 3 products.
- Q4: two dimers and a shared reservoir, with out / in exchanges and the inter-dimer transfer inside the window and
  T = 0 between the dimers: two clusters, and F equals the global unswitched Hamiltonian on the product space
  (round-off). As built: one cluster.
- Q5: R14's static case on the 6-atom random-weight module, seeds 3, 4, 5: |clusters − global| ≤ 1e-2 eV (as
  built +0.168, +0.535, +0.073 eV). The two spaces differ in their switches, so equality is not claimed.
- Q6: benchmark, doped wetting 2×1, 'fermi': two clusters; |E(clusters) − E(global)| ≤ 1e-3 eV (as built 1.243
  eV); the atom-108 scan has no jump above 1e-6 eV between neighbouring points beyond the force's own slope.
- Q7: `tests/redox` and the rest of the suite pass.
**Not claimed.** Scalability of rule 1 inside one large merged cluster (exponential in its disjoint parts; refused
beyond the cap, never truncated). Three-cluster products (dropped with the three-cluster increments, as before).

### C2b scored (2026-11-11 09:00 IST; `trace_redox/clusters.py`, `RedoxConfig.cluster_closure`; `tests/redox/test_redox_cluster_closure.py`, `test_redox_cluster_product.py`, `test_redox_bvk.py`; s43 frozen weights for Q6; output in `runs/extensivity_2026-11-10/c2b_scan108_wetting_2x1_s43.txt`, `c2b_scan108_wetting_2x1_s43_global.txt`, `c2b_scan109_wetting_2x1_s43.txt`): Q2, Q3, Q4, Q6 PASS; Q1, Q5, Q7 FAIL as written

No criterion was changed. Three test changes were made after the failures, each with its record in the test itself;
they are listed under Q1, Q5 and Q7 and are not counted as passes of the criteria as written.

| | criterion | measured | verdict |
|---|---|---|---|
| Q1 | every identity of `test_redox_cluster_product.py` unchanged | 32 of 33 unchanged; the reservoir test counts 6 rows per cluster, not 5 | **FAIL as written, on one count** |
| Q2 | R14's merge: F(τ = 0.4335) − F(τ = 0.4325) equals the global difference to 1e-6 eV | cluster sizes [3, 3] → [11] (first build [7]); F = −3.908721652 eV on both sides, jump +0.0e+00 eV; global +8.9e-16 eV; first build +0.789756 eV | PASS |
| Q3 | one merged cluster, unswitched: F equals the global Hamiltonian on the closed set; the 3 × 3 products are held | 13 linked rows → 19 closed (brute enumeration: 19); F − exact = +4.4e-15 eV; first build +0.789756 eV above | PASS |
| Q4 | two dimers and a shared reservoir, T = 0 between the dimers: two clusters, F exact on the product space | two clusters [7, 7] (first build: one cluster of 21); pair space 41 rows, the valid products; F − exact = 0.0e+00 eV; first build +0.789758 eV above | PASS |
| Q5 | 6-atom random-weight module, seeds 3, 4, 5: \|clusters − global\| ≤ 1e-2 eV | −0.056515, −0.175719, −0.363878 eV | **FAIL as written** |
| Q6 | doped wetting 2×1, 'fermi': two clusters; \|E(clusters) − E(global)\| ≤ 1e-3 eV; no jump above 1e-6 eV in the atom-108 scan | two clusters of 4 rows; E(clusters) = E(global) = −1801.8091533159 eV, equal at the printed 1e-10 eV at all nine scan points, forces equal at 1e-10 eV/Å (clusters 7 configurations, global 14); first build −1800.5659109 eV (+1.243 eV); largest trapezoid defect between neighbouring points 6.3e-8 eV (first build: a +1.34 meV jump) | PASS |
| Q7 | `tests/redox` and the rest of the suite pass | first run: `tests/redox` 3 failed, 940 passed; the rest 1045 passed, 4 skipped. After the test change below: `tests/redox` 944 passed; the rest 1045 passed, 4 skipped | **FAIL as written** (3 tests) |

**Q1, the one count.** `test_a_reservoir_unit_does_not_link_the_clusters` expected 5 rows per cluster. Rule 1 adds
a sixth, (a: +1, d: +1, R: −2): the product of the two single-unit exchanges with the reservoir, (a: +1, R: −1) and
(d: +1, R: −1), which SPEC 21's amplitudes had left out (amplitude 0.0104). F moves from −6.225526560 to
−6.225667026 eV (−1.405e-4 eV for the three clusters). The pre-registration overlooked that the closure acts inside
a cluster that was never merged. The test now asserts both builds: `closure=False` gives the 5 rows, the closure the
same 5 and the one product, and the free energy falls by less than 1e-3 eV.

**Q5, what was measured instead.** The criterion compared two approximate spaces. The 6-atom module's full space is
small enough to enumerate: 141 neutral configurations over charges (−1, 0, +1) on six units (a window of 25 eV and of
1000 eV give the same 141; a direct enumeration gives 141). Against it, in eV:

| seed | global (SPEC 21) | clusters, first build | clusters, closure | rows of the closed cluster |
|---|---|---|---|---|
| 3 | +0.369896 | +0.537829 | +0.313381 | 111 |
| 4 | +0.159338 | +0.693938 | −0.016381 | 141 |
| 5 | +0.412894 | +0.485813 | +0.049016 | 69 |

The closure is the nearest of the three to the full space on every seed, and the global space, which Q5 took as the
reference, is 0.16 to 0.41 eV from it. The closure's own error splits exactly into two parts:
- **rows** (the bare Hamiltonian on the closed rows against the full space): +0.021764, 0.000000, +0.014934 eV.
  This part is variational (Cauchy interlacing), and it is asserted;
- **SPEC 21's switch** (the switched Hamiltonian against the bare one on the same rows): +0.291618, −0.016381,
  +0.034082 eV. Of 644 / 980 / 304 coupled elements the switch is 1 on 53 / 427 / 110, 0 on 267 / 174 / 121 and
  between on 324 / 379 / 73.
**The switch is not variational**: on seed 4 the closed set is the full space, the diagonals are equal, and the
switched lowest eigenvalue is −5.965319 eV against −5.948938 eV for the bare matrix, 16.4 meV below. A principal
submatrix is bounded by interlacing; a matrix with some elements scaled by factors in [0, 1] is not. So C2b's error
on these strongly coupled random modules (couplings of about 1.2 eV at kT = 0.1 eV) is SPEC 21's switch, up to
0.29 eV, not the cluster product. On the benchmark cell the two spaces agree to 1e-10 eV (Q6). The Q5 test was
replaced by `test_the_six_atom_module_against_the_full_space[3, 4, 5]`, whose docstring records the failure; its
assertions were written after the measurement and are not a criterion.

**Q7, the three tests.** `test_redox_bvk.py`, case "bulk, charged cell" (two tests of B1 and the record test): the
2 × 1 × 1 supercell of the charged 6-atom cell is one cluster of 35 linked rows, and its closure holds 5184. The
default cap of 512 refuses it ("refused beyond the cap, never truncated", the limit stated under "Not claimed").
Measured at a cap of 8192: 47 s for the supercell (69 s under load), the energy per cell 6.0864583521 eV against 6.312848704 eV for
the first build, and the Born–von Kármán identity holds with the closure too (sampled − supercell per cell =
+0.00e+00 eV at a tolerance of 1.6e-9). The other three cases of that file run with the closure and pass. The change:
the charged case now runs on the first build (`cluster_closure=False`, a record in the file), since the identities
of that file do not depend on the closure, and a new test asserts the refusal on the supercell and on the mesh
(`test_the_closure_of_the_charged_cell_is_refused_not_truncated`). The refusal's message now gives the linked and
the closed counts and names `fragment_max_configurations`.

**A test that passed for the wrong reason, found and fixed before scoring.** The first red state was a TypeError
only (`closure` was not yet an argument). After `clusters.py` was written, 7 tests passed while `module.py` still
called the old signatures (`couplings=None`, which leaves rule 2's "T vanishes between the supports" vacuous, and
`bare=None`). That was caught by reading the call, not by a test; after the wiring Q5 failed. The first build is
asserted inside the tests through `closure=False` (one cluster, the +0.789756 eV jump), so the physics is red
without the closure and green with it in the same run.

**What the code now does.** `RedoxConfig.cluster_closure` (default True) applies both rules in the clusters space;
`False` keeps SPEC 36 as first built (legacy, kept: +0.790 eV at the merge of two model dimers, +1.243 eV on the
doped 2×1 cell). The default of `diabatic_space` is unchanged ('global'). A local branch marker
`c2b-closure-2026-11-11` was set at `c44f45b` before the change; it holds none of the uncommitted work.

**Limits, measured.**
1. Cost. The closure is exponential in the disjoint parts of one cluster: 35 → 5184 rows on a 12-atom strongly
   coupled cell. On the benchmark's doped 2×1 cell the clusters hold 4 rows each. A polynomial route (increments
   inside a merged cluster, with the linking rows in the pair spaces) is not designed; it is queued and would be its
   own pre-registered step if a benchmark cell is refused.
2. SPEC 21's switch is not variational (16.4 meV below the full space on seed 4) and costs up to 0.29 eV on the
   random modules. Not measured on the benchmark beyond Q6's agreement of the two spaces.
3. Not measured: the non-wetting doped cell; s42; any mesh. These are C4 stage 2's U2 to U4, next.

**Citation, verified 2026-11-11 from the publishers' records and abstracts (the full texts were not read).** The
method of increments: H. Stoll, Phys. Rev. B 46, 6700 (1992), correlation energy of diamond from increments of
localized bonds, pairs and triples of bonds; review: B. Paulus, Phys. Rep. 428, 1 (2006). ESTABLISHED there: the
expansion of an energy in one-body, two-body and higher increments of local groups. NEW-TO-TRACE here: its use for
the free energy of a charge-configuration ensemble, with the two rules above.

### R20. R19's kink located: the knots of SPEC 33's wall table, not the ensemble (2026-11-11 09:00 IST; s43 frozen weights, doped wetting 2×1, 'fermi', clusters space with the closure; output in `runs/extensivity_2026-11-10/c2b_scan108_wetting_2x1_s43.txt`, `c2b_scan109_wetting_2x1_s43.txt`, `r20_scan109_wetting_2x1_s43_wall_off.txt`)

R19 closed with a hypothesis: a kink of about 1.3e-6 eV where the Au charges cross −0.5. **That hypothesis is
false**, and the kink has nothing to do with the ensemble or with extensivity.
- **What the kink is.** Energy and force are continuous; the force's slope steps. Atom 108 along z, step 0.005 Å,
  dF/dz between neighbouring points: −0.4257, −0.4424, −0.4603, −0.4794, −0.4998, −0.5165, then −0.4322, −0.4617
  eV/Å². The step is +0.113 eV/Å², about 22 % of the local curvature. The clusters space and the global space give
  the same eight numbers.
- **Falsified, the charge crossing.** Atom 109 is the mirror: its Au charges cross −0.5 at −0.010 Å and the slope
  runs through smoothly there (−0.4289, −0.4454, −0.4632, −0.4822, −0.5025, −0.5243, −0.5474). Its step comes later,
  between +0.015 and +0.020 Å (−0.4170).
- **Rejected after reading the code, `band_cutoff` = 3.5 Å.** The band's hop switch is a C² quintic and hops join
  cation frames only; an Au–O pair is not a hop.
- **Located.** `BondPauliWall._e1` interpolates the Au–O table with a cubic Hermite form, which is C¹: its second
  derivative steps at every knot. Measured steps of d²E/dR² ('almo', Au–O; right minus left): −2.16584 at 2.9 Å,
  +0.14924 at 3.2 Å, +0.16915 at 3.5 Å, +0.00340 eV/Å² at 3.8 Å (the 'frz' table: +1.11406, −1.44566, −0.60899,
  +0.06881). The wall's own switch is 1 below 3.8 Å. Three checks:
  1. position: Au108–O61 passes 3.5 Å at dz = +0.0098 Å; Au109–O49 and Au109–O61 at +0.0166 and +0.0165 Å;
  2. size: 0.16915 × (dR/dz)² with dR/dz = −0.826 gives 0.115 eV/Å² for atom 108 (measured 0.113), and two
     crossings give 0.229 eV/Å² for atom 109, of which 69 % of the last interval lies beyond the crossing:
     predicted +0.138 for the wall's part of that interval's slope, measured +0.135;
  3. toggle: with the table multiplied by zero in a scratch process (the package unchanged) the atom-109 slope is
     smooth through the same interval (…, −0.5046, −0.5274, −0.5518) and the trapezoid defect of that interval
     falls from +5.5e-7 to −5.2e-8 eV, in line with its neighbours.
- **What it means.** The model's energy is C¹ in the Au–O distance, so a Hessian or a phonon involving an Au–O
  pair near 2.9, 3.2, 3.5 or 3.8 Å (and the inner knots, not measured here) carries a step of the sizes above.
  Relaxations and the Table II energies are not affected, and the MgO phonon benchmark has no Au–O pair. No
  benchmark number moves, so nothing is changed. Queued: a C² interpolant through the same knots. It changes the
  energy between the knots, so it would be a pre-registered step of its own with the SPEC 33 probes re-scored.
- **U2's residuals.** The central difference at step 0.005 Å differs from the force by 1.4e-5 to 2.0e-5 eV/Å in
  these scans (h² E‴ / 6). Whether that accounts for the 1.3e-6 to 2.0e-6 relative residuals of R19's U2 is not
  verified; U2 will be re-run with a second step.

### R21. U2's failure on the non-wetting cell: the 2.26 Å knot of SPEC 33's wall table, at the O-top bond length (2026-11-11 09:33 IST; s43 frozen weights, doped non-wetting 2×1, 'fermi', clusters space with the closure; output in `runs/extensivity_2026-11-10/fermi_unit_U2_2x1_s43_c2b.txt`, `r21_u2_nonwetting_2x1_s43.txt`)

With the closure the wetting 2×1 cell passes U2 on atom 108 (F − (−dE/dz) fell from +1.27e-01 to −1.75e-08 eV/Å,
R14 repaired). The non-wetting cell did not move: atom 108, F −0.2221193894, fourth-order stencil −0.2195342297,
difference **−2.59e-03 eV/Å** before and after the closure (second-order stencil: −4.82e-03). So it is not the
ensemble.
- **Located.** On the reference geometry Au108–O1 = 2.26078 Å (the O-top bond; dR/dz = −1.000). SPEC 33's wall
  table has a knot at 2.26 Å, 0.0008 Å away, and the interpolant is C¹ (R20). Step of d²E/dR² at that knot, right
  minus left: **+5.54474 eV/Å²** ('almo' wall).
- **A first check missed it.** It held the knots 2.9–3.8 Å only and predicted zero. Every knot of both
  interpolants was then measured (autograd at the knot ± 1e-7 Å), eV/Å²:

  | knot (Å) | 1.8 | 2.0 | 2.1 | 2.26 | 2.4 | 2.6 | 2.9 | 3.2 | 3.5 | 3.8 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | wall, 'almo' | −0.11659 | +6.05064 | +3.88545 | +5.54474 | no knot | −3.19919 | −2.16584 | +0.14924 | +0.16915 | +0.00340 |
  | law D1 (cos θ = 1), 'almo' | +0.53917 | −0.25640 | −0.28342 | −0.10432 | −0.10380 | −0.07202 | +0.00904 | +0.00020 | | |
  | law D2 (P2 = 1), 'almo' | +0.06269 | −0.08987 | −0.17164 | −0.15206 | −0.33042 | −0.73141 | +0.65538 | +0.32571 | | |
  | wall, 'frz' | −4.23138 | +2.85200 | +1.64018 | +1.43823 | +1.36362 | +1.60471 | +1.11406 | −1.44565 | −0.60899 | +0.06881 |
  | law D1, 'frz' | +1.37949 | −0.48275 | −0.45839 | −0.13219 | −0.10586 | −0.05320 | +0.00387 | +0.00004 | | |
  | law D2, 'frz' | +0.13411 | −0.17618 | −0.32223 | −0.26631 | −0.54398 | −1.09854 | +0.84256 | +0.32783 | | |

- **Three checks.**
  1. Size. A curvature step Δk at the stencil's centre gives the fourth-order stencil an error Δk h / 6 and the
     second-order one Δk h / 4; with the knot 0.0008 Å off centre and h = 0.005 Å the factors are 0.0958 and 0.178.
     From the wall's step alone: −2.65e-03 and −4.94e-03 eV/Å; with the neutral law's two steps (−0.256 eV/Å²):
     −2.53e-03 and −4.71e-03. Measured: −2.59e-03 and −4.82e-03.
  2. Toggles, in a scratch process (the package unchanged). Wall table × 0: the difference is +1.20e-04 eV/Å (the
     law's own steps). Wall × 0 and the law's distance dependence frozen (f = 1): −6.06e-07 eV/Å, the stencil's
     resolution (below).
  3. Shifted centre. The same stencil centred 0.030 Å to either side of the knot: −2.07e-08 and −1.04e-06 eV/Å.
- **The doped 2×2 cell** shows the same number on the same atom: −2.58e-03 eV/Å
  (`fermi_unit_U2_2x2_s43_c2b.txt`).
- **What it means.** Energy and forces are continuous. The Au–O stretch curvature of the model is 14.9 eV/Å² across
  the knot (from the forces at ±0.030 Å) and steps by about 5.4 eV/Å² at 2.26 Å (from the measured stencil error;
  the wall's step is +5.54, the neutral law's −0.26), which is the DFT O-top bond length the table was sampled at.
  A Hessian, a vibrational frequency or a phonon of Au on O-top is therefore not defined at the reference geometry
  to better than that step. Relaxations and the Table II energies are not affected.
- **Queued, not done.** A C² interpolant for both SPEC 33 tables (the wall and the law). It changes the energy
  between the knots, so it is its own pre-registered step with the SPEC 33 probes re-scored, and it needs the
  decision before any default changes.

### R22. The SPEC 36 energy steps where the reference n⁰ changes: the dropped three- and four-cluster terms (2026-11-11 09:33 IST; s43 frozen weights, doped wetting 2×2, 'fermi', clusters space with the closure; output in `runs/extensivity_2026-11-10/fermi_unit_U2_2x2_s43_c2b.txt`, `u2_scan_wetting_2x2_atom108_s43.txt`, `u2_fine_wetting_2x2_atom108_s43.txt`, `r22_reference_wetting_2x2_s43.txt`)

The doped 2×2 cells evaluate with the closure (13 configurations, no refusal; 21 s per energy and forces). U2 on
the wetting 2×2 cell, atom 108: F +0.0044100946, fourth-order stencil +0.0044843545, difference **−7.43e-05 eV/Å**.
That is above the stencil's resolution and absent on the 2×1 cell (−1.75e-08 on the same atom).
- **Candidates named before the scan:** a selection change inside the stencil; the reference-filling passes
  (N₀ = 8 in 3 passes); the 3.5 Å wall knot at dz = +0.0098 Å (R20).
- **Scan** (atom 108 along z, 15 points, 0.0005 Å apart). Between dz = +0.0020 and +0.0025 Å the reference
  configuration n⁰ changes: the electron of every dimer moves from its second Au to its first, in all four cells at
  once (`r22_which_units_s43.txt`; the lowest configuration is an ordered pattern with two nearly degenerate
  orientations). The four clusters keep 4 configurations each on the same units, the pair spaces 16, N₀ = 8 in 3
  passes throughout. At that point the energy steps by **−5.7e-07 eV** and the force on atom 108 by +3.5e-05 eV/Å.
  Everywhere else in the scan the second-order stencil agrees with the force to ≤ 1.1e-06 eV/Å. The step predicts
  −7.5e-05 eV/Å for U2's stencil (measured −7.43e-05). The 3.5 Å knot of R20 (dz = +0.0098 Å) is also seen in the
  coarse scan: dF/dz steps by about +0.10 eV/Å² after +0.010 Å (R20: +0.113).
- **Cause, checked.** At six points across the change the free energy was also built on the full product space of
  the four clusters (246 valid configurations; the same exact energies and the same switched couplings):

  | dz (Å) | n⁰ | F_pairs − F_full (eV) |
  |---|---|---|
  | +0.0010 | A | −1.6048e-04 |
  | +0.0015 | A | −1.6047e-04 |
  | +0.0020 | A | −1.6046e-04 |
  | +0.0025 | B | −1.6103e-04 |
  | +0.0030 | B | −1.6104e-04 |
  | +0.0035 | B | −1.6105e-04 |

  F_full runs through smoothly (second differences +1.75e-07, +1.77e-07, +1.77e-07, +1.78e-07 eV), and so does the
  rest of the energy (E − F_pairs: −6.4e-08, −6.3e-08, −6.4e-08, −6.5e-08). The step is entirely in
  F_pairs − F_full, the three- and four-cluster terms that limit L1 drops: their sum depends on the reference, by
  5.7e-07 eV here.
- **Why 2×1 is clean.** Two clusters are exact (P3), so nothing depends on the reference.
- **Two numbers this gives.** (i) Limit L1 on a doped cell: the dropped terms are −0.1605 meV on the doped wetting
  2×2 cell, −0.040 meV per 1×1 cell (the undoped cells: +0.006 and −0.004 meV per cell, SPEC 36's score).
  (ii) A limit that SPEC 36 did not state: **on three or more clusters the energy is discontinuous where the lowest
  configuration changes**, by the reference dependence of the dropped terms. Here all four clusters change
  together, 1.4e-07 eV per cell; how the step grows with the cell is not measured.
- **Not changed.** No benchmark number moves (Table II's cells hold one cluster each at 1×1, and −0.040 meV per
  cell is below every criterion). Removing the step needs three-cluster increments (cubic in the clusters) or a
  reference that is not a single configuration; neither is built. The limit goes into SPEC 36 as stated.

### U2's resolution: what a stencil on this energy can show (2026-11-11 09:33 IST; output in `runs/extensivity_2026-11-10/u2_second_step_2x1_s43.txt`, `u2_energy_noise_2x1_s43.txt`)

R20 left open whether U2's 1e-6-level residuals are force errors. They are not.
- **Three steps.** Fourth-order stencil, F − (−dE/dz) in eV/Å at h = 0.010 / 0.005 / 0.0025 Å, doped 2×1:
  wetting atom 109 +1.10e-05 / +5.44e-08 / −1.41e-07; wetting atom 8 +1.16e-05 / +9.48e-07 / +1.72e-06;
  non-wetting atom 109 −7.30e-08 / +2.24e-07 / −3.85e-07; non-wetting atom 8 +1.19e-05 / +6.35e-07 / +2.21e-07.
  The residuals do not fall with h and change sign; the second-order ones fall as h² (wetting atom 8: 1.14e-03,
  2.87e-04, 7.30e-05).
- **Noise of the energy.** Wetting 2×1, atom 8: a repeated evaluation is identical to the last digit. Displaced by
  ±1e-6 to ±1e-5 Å the energy leaves its Taylor form (F, and k = +11.058632 eV/Å²) by 1.66e-09 eV rms, 2.73e-09
  max. A fourth-order stencil at h = 0.005 Å then resolves the force to about 3.2e-07 eV/Å rms, with a truncation
  of up to 7e-07 eV/Å on top (the h = 0.010 value / 16).
- **So** U2's criterion (1e-6 relative) cannot be met by this stencil on forces below about 1 eV/Å. The criterion
  stays as written and is scored as written below. What sets the 2e-09 eV noise is not established (hypothesis,
  unchecked: the tolerance or conditioning of an iterative solve).

### C4 stage 2 scored (2026-11-11 10:40 IST; `trace_redox/band.py` `augment_fermi`, `RedoxConfig.band_form`; `tests/redox/test_redox_fermi_band_unit.py`; `probe_ext_fermi_unit.py`, `probe_ext_fermi_table2.py`, `probe_ext_fermi_split.py`, `probe_ext_full_product.py`, `probe_ext_fermi_filling.py`, `probe_ext_history.py`, `probe_ext_reference_margin.py`; frozen weights, `'onsite'`, clusters space with the closure; output in `runs/extensivity_2026-11-10/`: `fermi_unit_U1_s43_final.txt`, `fermi_unit_U2_2x1_s43_c2b.txt`, `fermi_unit_U2_2x2_s43_c2b.txt`, `fermi_unit_U3U4_s43_c2b.txt`, `fermi_unit_U3_s43_rerun.txt`, `fermi_unit_U4_s43_c2b.txt`, `fermi_unit_U4_steps_s43_c2b.txt`, `c4s2_suites.txt`, `fermi_table2_gamma_s43.txt`, `fermi_table2_mesh221_s43.txt`, `fermi_table2_s42.txt`, `fermi_table2_mesh221_doped_s42_cap2048.txt`, `fermi_split_s43.txt`, `l1_full_product_2x2_s43.txt`): U1, U3, U5 pass; U2 fails as written; U4 runs

Benchmark numbers addressed: wet_d (DFT −2.7 meV; 4G −41.0) and T5 (DFT −931.6; 4G −1016.4), which before this
step existed at one cell size only (R17). No training. The criteria are those of the pre-registration, unchanged.

| | criterion | measured | |
|---|---|---|---|
| U1 | doped 1×1 at Γ, 'fermi' against 'phi1': \|ΔE\| ≤ 1e-4 eV | wetting +1.717e-06 eV, non-wetting +1.641e-06 eV (max\|ΔF\| 9.9e-06, 9.4e-06 eV/Å); the band holds 2.0000 e in both forms, N₀ = 2 in 2 passes | PASS |
| U2 | the doped 2×1 and 2×2 cells evaluate; forces equal −∇E by central differences, 1e-6 relative on 3 components | all four cells evaluate (7, 11, 13, 13 configurations; the band holds 2.0000 e per cell copy). Relative differences, atoms 108 / 109 / 8 along z: wetting 2×1 6.3e-06 / 3.2e-05 / 2.0e-06; non-wetting 2×1 **1.2e-02** / 1.8e-06 / 1.3e-06; wetting 2×2 **1.7e-02** / 4.7e-05 / 2.0e-06; non-wetting 2×2 **1.2e-02** / 2.3e-07 / 2.0e-06 | **FAIL** (1 of 12 components within 1e-6) |
| U3 | `bvk_length = 18.0` Å: E(2×2)/4 − E(1×1) ≤ 1e-3 meV and max\|ΔF\| ≤ 1e-6 eV/Å, both doped geometries | wetting +0.000003 meV, 6.52e-08 eV/Å; non-wetting +0.000007 meV, 6.19e-08 eV/Å; the same in a rerun (control at the unit mesh: +1829.099515 meV, 1.03e-01 eV/Å; +1791.516223 meV, 1.00e-01 eV/Å, see the note below) | PASS |
| U4 | a double backward runs on a doped cell and matches central differences (no number pre-registered) | d Σ\|F\|²/d(`band.log_factor`): 1×1 autograd −2.3960163663e+01, differences at parameter steps 1e-4 / 1e-3 / 1e-2 off by 1.7e-05 / 1.6e-06 / 4.1e-07 relative; 2×1 autograd −2.6209126000e+01, 2.5e-05 / 1.3e-06 / 1.5e-06 (double backward 2–3 s and 6–12 s) | runs; agreement 4.1e-07 / 1.3e-06 at the best step |
| U5 | both suites | `tests/redox` 944 passed (492 s); `pytest tests --ignore=tests/redox` 1045 passed, 4 skipped (382 s); final code, 2026-11-11 09:38–09:53 IST | PASS |

**U1 was measured twice.** Its first record (`fermi_unit_U1_s43.txt`, 07:40) predates the closure code
(`module.py` 08:11, `clusters.py` 09:00). Rerun on the final code at 10:35 (`fermi_unit_U1_s43_final.txt`): the
same two numbers to the digits printed.

**U2, what the failures are** (each located by its own record; none is in the band unit).
- The three components at 1e-2 relative. Non-wetting, atom 108 (2×1 and 2×2, −2.59e-03 and −2.58e-03 eV/Å): the
  2.26 Å knot of SPEC 33's C¹ wall table sits inside the stencil at the O-top bond length (R21). Wetting 2×2, atom
  108 (−7.43e-05 eV/Å): the SPEC 36 energy steps by −5.7e-07 eV where the reference configuration changes on four
  clusters (R22). In both, the force is the exact derivative of the energy on each side; the stencil straddles a
  kink or a step of the energy itself.
- The nine other differences are 1.8e-08 to 1.0e-06 eV/Å. That is the stencil's resolution: about 3.2e-07 eV/Å of
  noise plus up to 7e-07 eV/Å of truncation ("U2's resolution" above), i.e. 2e-6 relative on a 0.5 eV/Å force and
  6e-4 on a 1.7e-3 eV/Å one.
- Before C2b the 2×1 cell failed for a third reason, R14's pair-space mismatch (R19); the closure removed it.
- The criterion is not changed and the score stays FAIL. A criterion that this energy could meet needs a C²
  interpolant for SPEC 33's tables and a continuous reference in SPEC 36; both are their own steps.

**U4's first run crashed in the probe**, not in the model: the probe evaluated under `no_grad` (the file
`fermi_unit_U3U4_s43_c2b.txt` keeps the traceback). Fixed in the probe; the numbers above are from the rerun.

**One value in U3's first record is not reproduced, and its cause is not found.** The control row for the
non-wetting cell at the unit mesh reads +1791.512941 meV in `fermi_unit_U3U4_s43_c2b.txt` (09:35). Every other
evaluation of the same two cells gives +1791.5162: U2's energies, the split below, and four checks made for this
(`history_nonwetting_s43.txt`, `history_u3_order_s43.txt`, `fermi_unit_U3_s43_rerun.txt`, `history_threads_s43.txt`).
- The 2×2 energy per copy is −1480189.063052692 eV in every position of every evaluation order tried (first,
  repeated, after the 1×1 cell, after the mesh calls, in U3's own order with forces), at 1, 2, 3, 4, 6 and 8
  threads, with the same reference and the same four clusters each time (SHA-1 tags). The 1×1 energy is
  −1480190.854568916 eV each time. The rerun of U3's block gives +1791.516223.
- The 09:35 value differs by 3.28e-03 meV per cell, 1.3e-05 eV on one evaluation of the 2×2 cell (the 1×1 energy
  in that record agrees to the six decimals it prints). No package file changed between the two runs (latest
  modification 09:00:37) and the probe's U3 block was not edited.
- Ruled out: evaluation order, thread count, a code change, a single configuration close to the reference (the
  next configuration of each cluster lies 0.197 eV above it, `reference_margin_2x2_s43.txt`). Not ruled out: a
  collective change of the reference as in R22 (hypothesis, unchecked; no trigger identified). No other
  evaluation of this session ran during that run (09:33:26 to 09:35:12).
- The criterion rows of U3 reproduce exactly, so the score does not depend on it. The record is kept as it is.

**Reported, not criteria: Table II on frozen weights with the Fermi band** (`probe_ext_fermi_table2.py`; Au relaxed
on the mean substrate with `eval_aumgo.relax`, fmax 0.005 eV/Å, the mesh read off the module's own record at every
call). The weights were trained at Γ with the block form of SPEC 28 and the two-electron band: these are sizes of
the change, not scores.

| Table II, meV | wet_u | wet_d | T5 |
|---|---|---|---|
| DFT | +928.9 | −2.7 | −931.6 |
| 4G-HDNNP | +975.4 | −41.0 | −1016.4 |
| s43 as stored (block, global, 'phi1', Γ) | +920.36 | −75.73 | −996.09 |
| s43 'onsite', clusters, 'phi1', Γ | +892.68 | −60.70 | −953.38 |
| s43 'onsite', clusters, 'fermi', Γ | +892.6511 | −60.6996 | −953.3506 |
| s43 'onsite', clusters, 'fermi', `bvk_mesh = (2, 2, 1)` | +935.1123 | **−22.9474** | −958.0597 |
| s42 'onsite', clusters, 'phi1', Γ | +832.37 | −164.68 | −997.04 |
| s42 'onsite', clusters, 'fermi', Γ | +832.1547 | −164.6794 | −996.8341 |
| s42 'onsite', clusters, 'fermi', `bvk_mesh = (2, 2, 1)` | +890.8461 | refused at the default cap (512); **+603.6879** at cap 2048 | −287.1582 at cap 2048 |

- At Γ the Fermi band reproduces the two-electron band on the doped cells (wet_d agrees to the two decimals
  recorded for 'phi1' on both seeds), as R18's point 1 predicted. wet_u differs by −0.027 (s43) and −0.22 meV
  (s42); that difference is not decomposed (candidates, unchecked: the band's thermal states above the undoped
  1.15 eV gap, R18's 1–2 meV leak; the zeroed couplings, +0.001 to +0.002 meV on the undoped non-wetting cell).
- wet_u at the mesh equals SPEC 37's value with the legacy band to the digits printed (+935.1123, s43). Consistent
  with SPEC 36's record that the configuration moving an electron from Au to the band unit is selected in the 1×1
  cell only, so that in the supercell the undoped band takes no part; not checked further.
- **wet_d at the mesh is the first doped Table II number at a second sampling**: +37.75 meV from Γ on s43. The
  falsifying check (per the rule-out-artifacts policy) for an artifact of the relaxation: at the fixed reference geometries
  E(2×2)/4 − E(1×1) = +1829.0995 (wetting) and +1791.5162 meV (non-wetting), a difference of +37.58 meV, against
  +37.75 meV between the two relaxed rows. The shift is therefore in the energy function, not in the relaxed
  geometries (Au–Au and Au–O move by ≤ 0.0005 Å between the two samplings).
- **The s43 shift decomposed** (`probe_ext_fermi_split.py`, `fermi_split_s43.txt`; fixed reference geometries;
  per 1×1 cell, E(2×2)/4 − E(1×1) in meV; F_b is the band's own free energy at the reference filling, E⁰ − F_b
  the rest of the reference configuration's energy, then SPEC 36's one-cluster and pair increments; everything
  outside the ensemble is identical per cell):

  | meV | total | F_b | E⁰ − F_b | Σ_c (F_c − E⁰) | pair increments |
  |---|---|---|---|---|---|
  | wetting | +1829.0995 | +1855.2126 | −38.7226 | +25.2247 | −12.6152 |
  | non-wetting | +1791.5162 | +1856.0576 | −36.1049 | −61.7747 | +33.3382 |
  | difference (the shift of wet_d) | +37.5833 | −0.8450 | −2.6177 | +86.9994 | −45.9534 |

  Each geometry's energy rises by 1.8 eV per cell from Γ to the supercell. That rise is the band's free energy at
  two electrons per cell (R18's point 2 measured +1855.3 and +1856.1 meV on the bare band; F_b here +1855.2 and
  +1856.1). The band carries −0.85 meV of the wet_d shift and the reference configuration −2.62. The shift is in
  the charge-transfer increments: +87.0 meV from the single clusters and −46.0 from the pairs. Read as physics
  (an interpretation, not tested separately): the energy that Au₂ gains by hybridising with the band changes when
  the dimer couples to the band of four cells, and the dimers' images then interact through it.
- **Limit L1 on these cells** (`probe_ext_full_product.py`, `l1_full_product_2x2_s43.txt`; the same exact energies
  and switched couplings on the full product space of the four clusters, 246 configurations). The shipped expansion
  minus the full space: +3.4688e-03 eV on the doped non-wetting 2×2 cell, **+0.867 meV per 1×1 cell**, and
  −1.6049e-04 eV on the wetting one (−0.040 meV per cell, R22's value). On wet_d at the mesh the dropped three- and
  four-cluster terms are therefore worth −0.907 meV (measured at the reference geometries; the table's −22.95 is
  relaxed, so a full-space value of about −22.0 meV is an estimate). That is 2.4 % of the shift, small against the
  seeds' spread and against 4G's 38 meV error on wet_d, and three orders above the 1e-3 meV to which U3 controls
  the energy (R22's addendum below).
- Whether the shift survives a larger mesh is not known: R18 measured +0.60 eV per cell in the last doubling of
  the band energy, and 3×3 is refused by the dense route. No convergence in the mesh is claimed.
- **The second seed does not give a doped number at the mesh** (R23 below): in the supercell its band takes the
  dimers' electrons, the default cap refuses the non-wetting cell, and at a raised cap wet_d is +603.69 meV.
- No claim against 4G follows from either mesh row: the weights were not trained at the mesh, and the two seeds
  differ there by 627 meV on wet_d (104 meV at Γ).

**Not done.** Convergence in the mesh; s42's U1–U4; the decomposition of wet_u's 'fermi' − 'phi1' difference; the
full-space check on s42's non-wetting cell (one cluster of 1218 configurations, so L1 has nothing to drop there);
any training with `band_form='fermi'`, the mesh, the clusters space or the on-site stiffness. Defaults are
unchanged: `band_form='phi1'`, `diabatic_space='global'`, the unit mesh, the cap 512.

### R23. The second seed at the mesh: the supercell's band takes the dimers' electrons, and the closure meets its cap (2026-11-11 10:40 IST; s42 and s43 frozen weights, doped cells, `'onsite'`, clusters space with the closure, 'fermi'; `probe_ext_fermi_filling.py`, `probe_ext_fermi_split.py`, `probe_ext_fermi_table2.py`, `probe_ext_full_product.py`, the last three with a measurement-only `cap=N` option; output in `runs/extensivity_2026-11-10/fermi_filling_s43.txt`, `fermi_filling_s42_cap2048.txt`, `fermi_split_s42.txt`, `fermi_split_s42_cap2048.txt`, `fermi_table2_s42.txt`, `fermi_table2_mesh221_doped_s42_cap2048.txt`, `l1_full_product_2x2_s42.txt`)

Benchmark numbers addressed: wet_d and T5 at a second sampling, second seed. No training, no package change, no
default changed (the `cap=N` option sets `fragment_max_configurations` inside the probe's process only).

1. **The default cap refuses s42's doped non-wetting supercell.** `ConfigurationSearchError`: 127 linked
   configurations in one cluster close to 1218 > 512. The Table II run at the mesh stopped there
   (`fermi_table2_s42.txt` keeps the rows before it and the traceback). The cap refuses and never truncates: with
   `cap=2048` the wetting rows are identical to those at 512 (E/cell −1480189.522170 eV; relaxed
   −1480189.548880 eV in both runs).
2. **Where the electrons are** (fixed reference geometries, Γ; the doped cell has three donor electrons per 1×1
   cell):

   | cell | band, e per cell | N₀ | Au free charges | Au₂ per cell | clusters |
   |---|---|---|---|---|---|
   | s43 wetting 1×1 | 2.0000 | 2 | −0.500 −0.500 | −1.0000 | [2] |
   | s43 wetting 2×2 | 2.0000 | 8 | −0.500 on all eight | −1.0000 | [4, 4, 4, 4] |
   | s43 non-wetting 1×1 | 2.0000 | 2 | −0.531 −0.469 | −1.0000 | [3] |
   | s43 non-wetting 2×2 | 2.0000 | 8 | −0.532 −0.468 on all four dimers | −1.0000 | [4, 4, 4, 4] |
   | s42 wetting 1×1 | 2.0000 | 2 | −0.500 −0.500 | −1.0000 | [3] |
   | s42 wetting 2×2 | **2.5003** | 10 | −0.500 −0.500 on two dimers, 0.000 on the other two | **−0.4997** | [33, 6, 6] |
   | s42 non-wetting 1×1 | 2.0000 | 2 | −0.535 −0.465 | −1.0000 | [4] |
   | s42 non-wetting 2×2 (cap 2048) | **2.8429** | 11 | +0.030 −0.286; +0.090 −0.215; +0.090 −0.215; +0.093 −0.215 | **−0.1571** | [1218] |

   On s43 every dimer keeps its electron in the supercell. On s42 the supercell's band takes half of the dimers'
   charge in the wetting cell (two of the four dimers are neutral: the reference configuration breaks the
   translation symmetry of the supercell) and 84 % of it in the non-wetting cell.
3. **What it does to the energy** (s42, cap 2048, fixed geometries; E(2×2)/4 − E(1×1), meV per cell):

   | meV | total | F_b | E⁰ − F_b | Σ_c (F_c − E⁰) | pair increments |
   |---|---|---|---|---|---|
   | wetting | +1538.3335 | +823.9162 | +652.6553 | +61.7915 | −0.0294 |
   | non-wetting | +782.0565 | +270.8120 | +772.9843 | −261.7398 | 0 (one cluster) |
   | difference (the shift of wet_d) | +756.2771 | +553.1042 | −120.3290 | +323.5313 | −0.0294 |

   Relaxed at `bvk_mesh = (2, 2, 1)` with cap 2048: doped wetting −1480189.548880 eV (Au–Au 2.5657, Au–O 3.6644 Å),
   non-wetting −1480190.152568 eV (2.5051, 2.1733 Å): **wet_d = +603.6879 meV** (Γ: −164.6794; DFT −2.7), and with
   wet_u = +890.8461 meV at the same mesh **T5 = −287.1582 meV** (DFT −931.6). The undoped cells are not affected:
   wet_u moves by +58.69 meV from Γ (s43: +42.46).
4. **The two seeds.** wet_d at Γ: −60.70 (s43), −164.68 (s42), 104 meV apart. At the mesh: −22.95 and +603.69,
   627 meV apart, and they differ in kind (Au₂⁻ on s43, a partly or mostly neutral Au₂ on s42).
5. **L1 on s42.** Wetting 2×2 (clusters [33, 6, 6], full product space 1182 configurations): shipped minus full
   −1.9269e-06 eV. Non-wetting: a single cluster, nothing dropped; the full-space run at the default cap is refused
   like the others.
6. **Reading (a hypothesis, not tested).** Training saw the doped cell at 1×1 and Γ only, where the band holds two
   electrons in its lowest level and the next level is 2.65 eV away (R18). That fixes the band's bottom and J₁₁. It
   does not fix the band's width, the interaction at higher filling (the band's own U_v came out negative, −4.2 to
   −2.4 eV, stage 1), or the position of the Au₂ level against the folded band levels 1.07 to 1.26 eV above the
   bottom, which is what decides whether the third electron sits on Au₂ or in the band of a supercell. R18's table
   already had the bare band of s43 lowest at 2.5 electrons per copy in the 2×2 cell. The two seeds then differ
   in quantities that no label constrained. The test of this reading is a model trained at the mesh; it is not
   run.
7. **Consequences.**
   - Extensivity at a fixed sampling length is a property of the equations: SPEC 37 evaluates the cell and its
     replication on the same supercell (U3: 3e-06 and 7e-06 meV on s43; s42's U3 is not run). A doped benchmark
     number at a mesh is a property of the weights, and the present weights do not determine it. No doped number at
     a mesh is claimed for either seed.
   - Training at a mesh is its own pre-registered step and needs a decision. Before it: which k-point sampling
     the reference DFT used for the 110-atom cell. I did not find it in this directory's records; it decides whether
     the DFT label belongs to Γ or to a converged mesh.
   - The closure of SPEC 36.7 is exponential inside a merged cluster, and here that is reached on a benchmark cell
     (127 → 1218). A polynomial closure, or increments inside the cluster, is needed before such cells can be
     relaxed or trained at the default cap. Not built.
   - A reference configuration that breaks the supercell's translation symmetry (s42 wetting) is a second face of
     R22's limit: the single-configuration reference. Not built: a reference that is not a single configuration.

### R22, addendum (2026-11-11 10:40 IST): the dropped terms are not below every criterion once a mesh is used

R22 closed with "No benchmark number moves (Table II's cells hold one cluster each at 1×1, and −0.040 meV per cell
is below every criterion)". That sentence holds at Γ only and is superseded here; it is kept above as written.
- Under a Born–von Kármán mesh the 1×1 cell is evaluated as its supercell (SPEC 37), which holds four clusters on
  the doped cells of s43. Limit L1 and R22's step then apply to the benchmark cell itself.
- Measured on the doped non-wetting 2×2 cell (s43; `l1_full_product_2x2_s43.txt`): shipped minus full product space
  +3.4688e-03 eV, **+0.867 meV per 1×1 cell**, about twenty times the wetting cell's −0.040. On wet_d at the mesh that is
  −0.907 meV at the fixed geometries.
- U3 is not affected: both sides of U3 evaluate the same supercell and drop the same terms.
- The step of R22 was measured on the wetting cell only (−5.7e-07 eV). Its size on the non-wetting cell is not
  measured.

### R24. The reference DFT's k-point sampling, found: 3 × 3 × 1 (2026-11-11 10:50 IST; answers R23's open question)

R23 left open "which k-point sampling the reference DFT used for the 110-atom cell". It is stated in the paper's
Methods (Ko, Finkler, Goedecker and Behler, Nat. Commun. 12, 398 (2021); arXiv:2009.06484): "For the Au2-MgO systems
stricter settings have been applied by multiplying each criterion by a factor 0.1 in combination with a 3 × 3 × 1
k-point grid." Read on 2026-11-11 from two copies of the text (the arXiv HTML rendering and the PubMed Central copy)
through a page-reading tool; the two agree. I did not read the PDF, and the paper does not say whether the grid
includes Γ. R23's sentence is kept as written; this record supersedes it.

**What it means for this line.**
1. ESTABLISHED: for one-electron states a k-mesh M on a cell is the Γ point of the M-fold supercell (Born–von
   Kármán; Monkhorst and Pack, Phys. Rev. B 13, 5188 (1976)). The model's analogue is SPEC 37,
   E_M(C) = E_Γ(S_M[C]) / |M|. So the benchmark's labels are those of a sampled calculation, 3 × 3 × 1 on the
   9.05 Å cell, and the statement of the model that corresponds to them is E_M at M = (3, 3, 1), not E_Γ. This is
   an analogy between two samplings, stated as such: the model's ensemble is not the DFT's wavefunction.
2. Every model so far, step 31 included, was trained with E_Γ on those labels. Its weights therefore carry the
   model's own finite-cell term (R15, R16: +42.4 meV on s43's wet_u between Γ and the (2, 2, 1) mesh). A
   frozen-weight number at a mesh counts that term twice in the sense that the weights were fitted with it in
   place; it is a size, not a prediction (as SPEC 37's score already said of wet_u).
3. A model trained at the mesh is the consistent one. Its cost is set by the mesh that is needed: (2, 2, 1) is
   4 cells per label, (3, 3, 1) is 9. Whether (2, 2, 1) is converged toward (3, 3, 1) is not measured (M1 below).

### Pre-registered: M1, the mesh convergence of Table II's undoped pair on frozen weights (written 2026-11-11 10:50 IST, before the probe and the run)

- **Benchmark number:** Table II's wet_u (DFT +928.9 meV, 4G +975.4), as a function of the sampling mesh; it sets
  the mesh, and so the cost, of the training step that R24 calls for.
- **Measurement.** `probe_ext_mesh_convergence.py` (new; no package code changes). Step 31 s43 and s42, frozen
  weights; `'onsite'`, clusters space with the closure, `'fermi'`; the undoped reference geometries, not relaxed
  (one evaluation per geometry and mesh); meshes (1, 1, 1), (2, 2, 1), (3, 3, 1). Reported: E per cell,
  E(wetting) − E(non-wetting), the two steps D12 = (2,2,1) − (1,1,1) and D23 = (3,3,1) − (2,2,1), the mesh read off
  the module's own record, cluster sizes, time per evaluation.
- **Reading, fixed now.** (a) |D23| ≤ 5 meV on both seeds: (2, 2, 1) is taken as converged to 5 meV for the
  undoped pair, and a training at a mesh may use it. (b) Otherwise it is not, no (2, 2, 1) number is quoted as the
  sampled value, and the next mesh is named as needed. (c) A refusal (cap) or a failure at (3, 3, 1) is recorded
  as such and decides nothing.
- **Prediction (hypothesis, written before the run).** If the image term of a transfer excitation falls as L⁻³
  (a dipole lattice in a slab plane), the residuals at the three meshes are c, c/8, c/27, so D12 = −(7/8) c,
  D23 = −(19/216) c and D23 / D12 = +0.1005: about +4 meV on s43 if D12 at fixed geometry is near the relaxed
  +42 meV.
- **Not measured by M1:** the doped cells (R23: their mesh numbers are not determined by the weights); relaxed
  geometries; a (4, 4, 1) mesh. Nothing is trained.

### M1, first result and a correction of its design (2026-11-11 10:52 IST; output `runs/extensivity_2026-11-10/mesh_convergence_undoped_s43.txt`)

s43, undoped, fixed reference geometries: E(wetting) − E(non-wetting) = **+907.1253 meV** at (1, 1, 1) and
**+947.9369 meV** at (2, 2, 1); D12 = **+40.8116 meV** (wetting +2.7852, non-wetting −38.0264 meV per cell, the
SPEC 36 score's 2×2 rows reproduced). At (3, 3, 1) both geometries are **refused**: "The solver would build a dense
35793x35793 operator K (10.25 GB; n_bonds=32823 for 990 atoms), above max_operator_bytes=4.00 GB". Reading (c): the
refusal decides nothing.

**My error in the design.** R15 (part 4) had already recorded that the dense route refuses 990 atoms. I
pre-registered a (3, 3, 1) leg without rereading it. M1 as written cannot decide (a) or (b) on this machine.

### Pre-registered: M1b, the meshes the dense route can hold (written 2026-11-11 10:52 IST, before the run)

- **Measurement.** The same probe and settings, both seeds, undoped, two one-dimensional sequences:
  (1,1,1), (2,1,1), (3,1,1), (4,1,1) and (1,1,1), (1,2,1), (1,3,1), (1,4,1) (at most 440 atoms). Reported: the steps
  of E(wetting) − E(non-wetting) and each later step over the first.
- **What it can and cannot show.** It measures how fast the image term falls with the sampling length along one
  axis. It does not measure (3, 3, 1): the term is second order in the couplings and is not a sum over image
  vectors. An additivity check is reported with it: D(2,2,1) against D(2,1,1) + D(1,2,1).
- **Prediction (hypothesis).** For images on a line with a pair term ∝ R⁻³ the residual falls as M⁻³ too: the
  second step is +0.1005 of the first and the third +0.0245.
- **Reading, fixed now.** If the second step is within [0.05, 0.20] of the first on both axes and both seeds, the
  M⁻³ law is supported and R15's estimate (a 2×2 cell keeps 5–12 % of the 1×1 image term, a 3×3 cell 0.4–3.7 %)
  stands as an estimate; otherwise it does not. Neither outcome replaces a (3, 3, 1) evaluation, which needs a route
  that does not build the supercell's dense operator (not built).

### M1 (s42) and M1b scored (2026-11-11 11:02 IST; both seeds, undoped, fixed reference geometries, `'onsite'`, clusters space with the closure, 'fermi'; `probe_ext_mesh_convergence.py`; output `runs/extensivity_2026-11-10/mesh_convergence_undoped_s42.txt`, `mesh_convergence_{x,y}_undoped_{s43,s42}.txt`)

**M1 on s42.** E(wetting) − E(non-wetting) = **+845.7884 meV** at (1, 1, 1) and **+902.9414 meV** at (2, 2, 1);
D12 = **+57.1530 meV** (wetting +9.2524, non-wetting −47.9006 meV per cell). (3, 3, 1) is refused as on s43.
Reading (c) on both seeds: M1 decides nothing about (3, 3, 1).

**M1b, the pre-registered quantity.** E(wetting) − E(non-wetting), meV:

| seed | axis | M = 1 | M = 2 | M = 3 | M = 4 | step 1 | step 2 (over step 1) | step 3 (over step 1) |
|---|---|---|---|---|---|---|---|---|
| s43 | x: (M, 1, 1) | +907.1253 | +935.6642 | +938.9129 | +939.6962 | +28.5388 | +3.2488 (0.1138) | +0.7832 (0.0274) |
| s43 | y: (1, M, 1) | +907.1253 | +928.8129 | +930.5660 | +930.7387 | +21.6876 | +1.7531 (0.0808) | +0.1727 (0.0080) |
| s42 | x: (M, 1, 1) | +845.7884 | +888.4481 | +893.3593 | +894.6654 | +42.6598 | +4.9112 (0.1151) | +1.3061 (0.0306) |
| s42 | y: (1, M, 1) | +845.7884 | +873.1848 | +875.3543 | +875.5542 | +27.3964 | +2.1695 (0.0792) | +0.1999 (0.0073) |

**Reading as fixed.** The second step is within [0.05, 0.20] of the first on both axes and both seeds (0.1138,
0.0808, 0.1151, 0.0792): the criterion is **met**.

**Additivity (reported as pre-stated).** D(2,1,1) + D(1,2,1) = +50.2264 meV against D(2,2,1) = +40.8116 meV on s43
(difference −9.41 meV); +70.0562 against +57.1530 meV on s42 (−12.90 meV). The term is not a sum over the two axes.

**A check that was not pre-registered: the two geometries separately.** The reference dimers: wetting, bond
vector (−2.578, −0.007, −0.007) Å, along x; non-wetting, (−0.005, +0.014, −2.545) Å, upright. Steps of the
energy per cell, meV, and the ratio of each step to the one before:

| seed | geometry | axis | step 1 | step 2 (ratio) | step 3 (ratio) |
|---|---|---|---|---|---|
| s43 | non-wetting | x | −22.9030 | −1.7655 (0.0771) | −0.1739 (0.0985) |
| s43 | non-wetting | y | −22.9018 | −1.7654 (0.0771) | −0.1739 (0.0985) |
| s43 | wetting | x (along the bond) | +5.6358 | +1.4833 (0.2632) | +0.6093 (0.4108) |
| s43 | wetting | y | −1.2143 | −0.0123 (0.0101) | −0.0011 (0.0918) |
| s42 | non-wetting | x | −29.0013 | −2.1825 (0.0753) | −0.2010 (0.0921) |
| s42 | non-wetting | y | −29.0024 | −2.1825 (0.0753) | −0.2010 (0.0921) |
| s42 | wetting | x (along the bond) | +13.6585 | +2.7287 (0.1998) | +1.1051 (0.4050) |
| s42 | wetting | y | −1.6060 | −0.0130 (0.0081) | −0.0010 (0.0798) |

A residual ∝ M⁻³ gives the ratios 0.1005 and 0.2434; ∝ M⁻² gives 0.1852 and 0.3500. What the table shows:

1. **No component follows M⁻³.** The upright dimer's term, the largest, falls by a factor 10–13 per added cell
   (ratios 0.075–0.099), faster than either power law and the same along x and y to 0.0012 meV. The flat dimer's
   term across the bond falls faster still. The flat dimer's term along the bond falls **slower** than M⁻² on s43
   (0.263, 0.411) and about as M⁻² on s42 (0.200, 0.405).
2. **The criterion was met by composition.** The second-step ratio of the difference landed inside [0.05, 0.20]
   because a fast term (non-wetting) and a slow one (wetting along the bond) add. I record the criterion as met
   and the hypothesis behind it, one pair term ∝ R⁻³, as **not supported**. R15's percentages rest on that
   hypothesis and are withdrawn as estimates of the remainder (the text of R15 is kept). **[Wrong, corrected in "M1c scored" below: R15's z numbers come from its tin-foil lattice sum, not from one R⁻³ pair term, and are not withdrawn.]**
3. **My prediction was also the wrong lattice sum.** In a sequence (M, 1, 1) the images across the axis stay at
   one cell: the row at distance i·M·a is a line of dipoles of spacing a, and a line contributes ∝ (i M a)⁻², so
   a dipole picture predicts M⁻² here (0.1852, 0.3500), not M⁻³. Neither law describes the upright dimer. **[Superseded: with the images along z, which the cell has, the tin-foil sum gives an exponential law for a z dipole; see M1c.]**
4. **The cause of the two decay laws is not established.** Hypothesis, untested: the fast term is the overlap of
   one copy's transfer states with the next copy's (bounded by `coupling_cutoff` 4.0 Å and the band's 3.5 Å, so
   it ends after the first neighbour shell of copies), and the slow one is the electrostatic interaction of the
   flat dimer's in-plane transfer dipole with its images along the bond. **[Withdrawn in "M1c scored": R15 part 3 had established the lattice sum, and M1c's vacuum test supports it.]**

**What is measured.** At four cells along one axis the last step is ≤ 0.21 meV for the upright dimer and for the
flat dimer across the bond, and +0.61 meV (s43), +1.11 meV (s42) for the flat dimer along the bond; in the
difference, +0.78 / +0.17 meV (s43, x / y) and +1.31 / +0.20 meV (s42).

**What is not measured.** (3, 3, 1), any mesh above four cells, relaxed geometries, doped cells. An **estimate**
of D23 = D(3,3,1) − D(2,2,1), with no law behind it: the sum of the two measured second steps, +5.00 meV (s43) and
+7.08 meV (s42), or the same scaled by the first step's measured non-additivity (0.8126, 0.8158), +4.06 and
+5.78 meV. By this estimate M1's reading (a), |D23| ≤ 5 meV on both seeds, would not hold on s42: **(2, 2, 1) is
not shown to be converged to 5 meV**, and the flat dimer's slow term continues beyond M = 3. **[Superseded by the lattice-sum estimate in the addendum to "M1c scored": +2.23 / +3.33 meV. Still an estimate.]**

**Cost (measured, Mac CPU, 4 threads, energy and forces of one cell).** (1,1,1) 2 s; two cells 3–4 s; three cells
7 s; four cells in a row 12–14 s; (2,2,1) 13–17 s. The cost of a training epoch at a mesh is not measured.

**Consequence for the benchmark.** The labels are sampled on 3 × 3 × 1 (R24). The mesh statement that corresponds
to them cannot be evaluated on the dense route; (2, 2, 1) is the nearest that can, and its distance from (3, 3, 1)
on Table II's undoped number is estimated at 4–7 meV on frozen weights **[superseded: 2.2–3.3 meV by the lattice-sum fit, an estimate]** (4G's error on that number: 46.5 meV; s43
at the mesh, relaxed: +935.11 against DFT +928.9 meV). A route that evaluates E_M without the supercell's dense
operator (one block per k-point of the mesh) is the step that would turn the estimate into a measurement and
bring a (3, 3, 1) training within reach. It is not built.

### Pre-registered: M1c, is the decay set by the periodic images through the vacuum? (written 2026-11-11 11:07 IST, before the probe's option and the run)

- **Hypothesis (replaces item 4 above, which stays as written).** The reference cells carry `pbc = (T, T, T)`
  (read off the four reference structures), so the electrostatics is 3-D Ewald with tinfoil boundaries
  (ESTABLISHED: de Leeuw, Perram and Smith 1980, not verified by me; `trace_electrostatics/ewald.py`) and the
  transfer excitation of one copy also sees its images at the vacuum period L_z = 26.4589 Å. For point dipoles on
  the lattice (M a, a, L_z) with k = 0 dropped, the M-dependent part of one dipole's energy is exact in closed
  form (planes normal to the sequence axis, 2-D reciprocal vectors G of the (y, z) lattice):
  a dipole in the planes, +(2π/A) Σ_{i≥1} Σ_{G≠0} G_u² e^{−|G| i M a} / |G|; a dipole normal to them,
  −(2π/A) Σ_{i≥1} Σ_{G≠0} |G| e^{−|G| i M a} − 2π/V_M. So a z dipole's term falls **exponentially**, at the rate
  e^{−2π a / L_z} = 0.1167 per cell, and an in-plane dipole's term along its own axis carries the tinfoil term
  −2π p² / V_M, which falls as **1/M**.
- **Computed before the run (point dipoles, a = 9.0474 Å; step-2 over step-1, step-3 over step-2):**

  | L_z | z dipole | in-plane dipole, along its axis | in-plane dipole, across |
  |---|---|---|---|
  | 26.4589 Å | 0.0818, 0.1121 | 0.1797, 0.4048 | 0.0013, 0.0015 |
  | 40.0 Å | 0.1240, 0.2057 | 0.1736, 0.3578 | 0.0013, 0.0015 |

  Measured at 26.4589 Å (table above): upright dimer 0.0771, 0.0985 (s43) and 0.0753, 0.0921 (s42); flat dimer
  along the bond 0.2632, 0.4108 and 0.1998, 0.4050. An agreement of ratios is not a test.
- **Test (one change).** The x sequence (1,1,1) … (4,1,1) again, both seeds, undoped, the same settings, with the
  cell's third vector lengthened to **40.0 Å** and no atom moved (the local model does not see the vacuum).
- **Criteria, fixed now.** (i) Upright dimer, both seeds: step-2 over step-1 rises into [0.10, 0.15] and step-3
  over step-2 into [0.16, 0.25]. (ii) Flat dimer, both seeds: step-3 over step-2 falls into [0.32, 0.39].
  (iii) Upright dimer: the second step grows by a factor in [1.5, 2.3] (point dipoles: 1.90) and the third by a
  factor in [2.6, 4.4] (3.49). If the upright dimer's two ratios stay within 10 % of their 26.4589 Å values the
  hypothesis is **refuted**. Any other outcome is reported as partial.
- **What it would mean.** The remainder of the mesh sequence would be two named terms of the tinfoil lattice sum,
  with known laws: e^{−2π a M / L_z} and 1/|M|. It would not make (3, 3, 1) measured.

### M1c scored, and two corrections of my M1b record (2026-11-11 11:15 IST; `probe_ext_mesh_convergence.py` with `c=40.0`, `probe_ext_mesh_lattice_fit.py`; output `runs/extensivity_2026-11-10/mesh_convergence_x_c40_undoped_{s43,s42}.txt`, `mesh_lattice_sum_fit.txt`)

**Measured, third cell vector 40.0 Å, no atom moved, x sequence (1,1,1) … (4,1,1), meV:**

| seed | E(wetting) − E(non-wetting) | flat dimer, steps (ratio to the step before) | upright dimer, steps (ratio) |
|---|---|---|---|
| s43 | +892.2729, +927.9816, +933.3109, +934.7123 | +5.6426, +1.4584 (0.2585), +0.5743 (0.3938) | −30.0661, −3.8709 (0.1287), −0.8270 (0.2136) |
| s42 | +823.0403, +875.5430, +883.2661, +885.3691 | +13.6434, +2.6750 (0.1961), +1.0265 (0.3837) | −38.8592, −5.0480 (0.1299), −1.0766 (0.2133) |

**Criteria as fixed.**

| | criterion | s43 | s42 |
|---|---|---|---|
| (i) | upright: step-2 ratio in [0.10, 0.15], step-3 ratio in [0.16, 0.25] (point dipole 0.1240, 0.2057) | 0.1287, 0.2136 **PASS** | 0.1299, 0.2133 **PASS** |
| (ii) | flat: step-3 ratio in [0.32, 0.39] (0.3578) | 0.3938 **FAIL** (by 0.004) | 0.3837 **PASS** |
| (iii) | upright: second step grows by [1.5, 2.3] (1.90), third by [2.6, 4.4] (3.49) | 2.193 **PASS**, 4.756 **FAIL** | 2.313 **FAIL**, 5.356 **FAIL** |

The refutation condition is not met: the upright dimer's ratios moved from 0.0771, 0.0985 to 0.1287, 0.2136 (s43) and
from 0.0753, 0.0921 to 0.1299, 0.2133 (s42). **Outcome: partial**, as pre-stated. The vacuum period sets the decay
of the upright dimer's term as the tin-foil sum of a z dipole says; the sizes grow more than a fixed point dipole
gives; the flat dimer's ratio moved in the predicted direction by about a third of the predicted amount.

**Also measured: the Γ value depends on the vacuum period.** E(wetting) − E(non-wetting) at (1,1,1) falls by
14.85 meV (s43) and 22.75 meV (s42) when the third cell vector goes from 26.4589 to 40.0 Å; at (4,1,1) by 4.98 and
9.30 meV. The frozen weights were trained at 26.4589 Å.

**Correction 1: the hypothesis was not new.** R15, part 3 had established it on real supercells (2×1, 3×1, 4×1,
1×2, 2×2): the image term is the tin-foil Ewald sum Λ of the excitation's dipole on the cell's lattice, with the
vacuum period in it (`probe_ext_dipole_lattice.py`). I wrote M1b's prediction, items 2–4 of its score and M1c's
"hypothesis" without rereading R15, for the second time today. What M1c adds to R15 is the change of the vacuum
period as an independent test. Item 4 of the M1b score (overlap of neighbouring copies) is withdrawn; the text stays.

**Correction 2: R15's percentages are not withdrawn.** Item 2 of the M1b score said they rest on one R⁻³ pair term.
They do not: the z numbers come from Λ_zz. The statement is replaced by the table below.

**The lattice sum against every measured step** (`probe_ext_mesh_lattice_fit.py`, written 11:24 IST; it reads the
probe outputs and R15's `Lam`, no model evaluation). Per geometry and seed the image term per cell is modelled as
c_x Λ_xx + c_z Λ_zz on the mesh's supercell. Upright dimer: c_z alone, from the first x step. Flat dimer: c_x and
c_z from the first x and the first y step. All constants at 26.4589 Å; every row below is a prediction
(predicted / measured meV, ratio in brackets).

| step | upright, s43 | upright, s42 | flat, s43 | flat, s42 |
|---|---|---|---|---|
| (2,1)→(3,1) | −1.873 / −1.766 (1.06) | −2.372 / −2.183 (1.09) | +1.097 / +1.483 (0.74) | +2.534 / +2.729 (0.93) |
| (3,1)→(4,1) | −0.210 / −0.174 (1.21) | −0.266 / −0.201 (1.32) | +0.465 / +0.609 (0.76) | +1.045 / +1.105 (0.95) |
| (1,2)→(1,3) | −1.873 / −1.765 (1.06) | −2.372 / −2.183 (1.09) | −0.071 / −0.012 (5.7) | −0.067 / −0.013 (5.2) |
| (1,3)→(1,4) | −0.210 / −0.174 (1.21) | −0.266 / −0.201 (1.32) | −0.008 / −0.001 (6.9) | −0.007 / −0.001 (7.2) |
| **(1,1)→(2,2)** | **−37.788 / −38.026 (0.994)** | **−47.849 / −47.901 (0.999)** | +3.450 / +2.785 (1.24) | +9.501 / +9.252 (1.03) |
| 40 Å: (1,1)→(2,1) | −28.736 / −30.066 (0.96) | −36.387 / −38.859 (0.94) | +5.413 / +5.643 (0.96) | +13.442 / +13.643 (0.99) |
| 40 Å: (2,1)→(3,1) | −3.563 / −3.871 (0.92) | −4.511 / −5.048 (0.89) | +0.993 / +1.458 (0.68) | +2.384 / +2.675 (0.89) |
| 40 Å: (3,1)→(4,1) | −0.733 / −0.827 (0.89) | −0.928 / −1.077 (0.86) | +0.376 / +0.574 (0.65) | +0.872 / +1.027 (0.85) |

Constants (meV per 1e-3 Å⁻³): upright c_z = +5.9721 (s43), +7.5623 (s42); flat c_x = +0.9107, c_z = +0.2233 (s43)
and +2.0293, +0.2108 (s42).

- **Upright dimer:** one constant gives the 2×2 step to 0.6 % (s43) and 0.1 % (s42), and the first step at 40 Å
  to 4 % and 6 %. The later, smaller steps are off by 6–32 %: too large at 26.4589 Å, too small at 40 Å. A fixed
  point dipole is therefore not the whole term (this is M1c's criterion (iii) again). Why is not measured.
- **Flat dimer:** two constants give the steps along the bond to 5–7 % (s42) and 24–26 % (s43), the 2×2 step to
  3 % and 24 %. The later steps across the bond (0.001–0.013 meV) are wrong by a factor 5–7. R15 had found the same
  failure of a point dipole for this geometry.

**The change with the vacuum period has two parts, both read off the model** (`probe_ext_vacuum_dipole.py`;
output `vacuum_dipole_undoped_{s43,s42}.txt`, `vacuum_dipole_m411_undoped_{s43,s42}.txt`). The mean cell dipole
of the upright dimer's cell is M_z = +0.5168 e·Å (s43) and +0.8060 e·Å (s42) at 26.4589 Å, +0.4125 and +0.6409 at
40 Å; at (4,1,1) it is +0.5135 and +0.8011. The flat dimer's cell has −0.1381 and −0.1159 e·Å. Under tin-foil
boundaries a mean dipole carries −2π K_E M_z²/V relative to the isolated slab (the slab correction of
`trace_electrostatics/ewald.py` with the opposite sign). That term belongs to the mean state, is nearly the same at
every mesh (+3.015 and +2.952 meV for s43, +7.305 and +7.168 meV for s42, at (1,1,1) and (4,1,1)), and is not part of
the extensivity defect.

| upright dimer, 26.4589 → 40 Å, meV per cell | measured | lattice sum (fit) | mean dipole's tin-foil term | sum / measured |
|---|---|---|---|---|
| s43, (1,1,1) | +15.360 | +11.731 | +3.015 | 0.96 |
| s43, (4,1,1) | +5.438 | +3.686 | +2.952 | 1.22 |
| s42, (1,1,1) | +23.171 | +14.854 | +7.305 | 0.96 |
| s42, (4,1,1) | +9.572 | +4.667 | +7.168 | 1.24 |

The flat dimer's cell changes by +0.51 / +0.42 meV at (1,1,1) and +0.45 / +0.28 meV at (4,1,1); its mean-dipole
term is +0.21 / +0.15 meV. Whether the reference DFT applied a dipole correction is **not verified**.

### Pre-registered: S1, where the time of one evaluation goes (written 2026-11-11 16:45 IST, before the probe and the run; decision of the same day: no change to the methods, measurements only)

The project's design rule (2026-11-11): TRACE stays a classical ML potential and must be much faster than DFT and DFTB;
it borrows from them only to solve its own problems. S1 measures the cost; it moves no Table II number.

- **Probe** `probe_speed_profile.py` (new, a probe only). Frozen step 31 s43, the undoped wetting reference cell
  (110 atoms), Mac M2 Pro (10 cores, 16 GB), `torch.set_num_threads` 4 and 1. One energy-and-forces evaluation per
  timing, each at a new geometry (all atoms displaced by a seeded 0.01 Å Gaussian, so no result is reused; the
  solver's seed is geometry-determined by design, `solver.py` refuses a previous-step warm start), after one
  untimed warm-up. Five timings; the median and the range are reported.
- **Split.** (a) the whole evaluation; (b) the electrostatics module's forward and its backward, timed by calling
  the module on the inputs a forward pre-hook captured (leaves with `requires_grad`); (c) the local model as
  (a) − (b), labelled as a difference; (d) inside (b), cProfile of the forward by function and the torch profiler
  of the backward by operator; (e) the size of the dense bound-sector operator n_m and the peak resident memory.
- **Settings.** As stored (`'block'`, `'global'`, `'phi1'`, Γ: the trained model) and the extensive forms
  (`'onsite'`, `'clusters'`, `'fermi'`) at the meshes (1,1,1), (2,1,1), (2,2,1).
- **Hypothesis (mine, stated in the discussion before this entry).** The dense bound-sector solve dominates.
  **Refuted at 110 atoms** if (b) is below half of (a) at (1,1,1) in either setting, or if, inside (b), the
  operations on the dense operator (its build, the CG matrix products and their backward) are below half of (b).
- **Not in S1.** A DFTB timing (no tight-binding code is installed on the Mac: `dftb+`, `xtb`, `tblite` absent;
  installing one is a download and needs a decision). GPU timings (the GPUs are for jobs, and float64 runs
  there at about 1/64 of float32).

### S1 scored (2026-11-11 16:56 IST; `probe_speed_profile.py`; output `runs/speed_2026-11-11/`)

**An error first.** The first two one-thread runs ran on 4 threads: `eval_aumgo.py` sets 4 threads on import, and
a later `torch.set_num_threads(1)` did not take effect (measured: the count stayed 4). The probe now sets the
count before the imports, skips `eval_aumgo`'s call and asserts the count. The two outputs are kept, renamed
`INVALID_ran_on_4_threads_*`. They are valid as repeats at 4 threads and give the spread between processes:
stored 1.324 and 1.157 s, extensive 1.528 and 1.320 s (about 15 %).

**Measured** (frozen s43, undoped wetting reference, M2 Pro CPU, float64; median of five evaluations at displaced
geometries, range in brackets; "local" is the whole minus the electrostatics module, a difference):

| setting | mesh | atoms in the evaluated cell | n_m (dense K) | threads | whole, s | electrostatics fw / bw, s | local, s | electrostatics share | peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| stored (block, global, phi1) | (1,1,1) | 110 | 3974 | 4 | 1.324 [1.268, 1.652] | 0.579 / 0.452 | 0.292 | 0.78 | 2.21 GB |
| stored | (1,1,1) | 110 | 3974 | 1 | 1.836 [1.725, 1.948] | 0.802 / 0.601 | 0.434 | 0.76 | 2.27 GB |
| extensive (onsite, clusters, fermi) | (1,1,1) | 110 | 3974 | 4 | 1.528 [1.426, 1.970] | 0.636 / 0.544 | 0.348 | 0.77 | 2.02 GB |
| extensive | (1,1,1) | 110 | 3974 | 1 | 1.844 [1.785, 2.197] | 0.864 / 0.603 | 0.377 | 0.80 | 2.42 GB |
| extensive | (2,1,1) | 220 | 7948 | 4 | 3.343 [3.234, 3.740] | 1.602 / 1.369 | 0.372 | 0.89 | 3.38 GB |
| extensive | (2,2,1) | 440 | 15896 | 4 | 13.265 [12.171, 15.268] | 6.626 / 6.231 | 0.408 | 0.97 | 5.40 GB |

- **n_m = n_bonds + 3 n_atoms**, with n_bonds = 3644 for 110 atoms: 33 bond variables per atom. They are every pair
  within the local model's cutoff r_max = 5.29 Å, not nearest-neighbour bonds, and they are 92 % of n_m.
- **Where the forward goes at 110 atoms** (stored, cProfile, 0.621 s): building K's blocks (`assemble`, with the
  Gaussian lattice sums) 0.239 s (38 %); the free-charge Ewald blocks 0.108 s (17 %); the carrier-induction
  columns (§26) 0.118 s (19 %); the diabatic solution 0.040 s and the selection 0.029 s; penetration 0.031 s; band
  0.014 s; bond Pauli 0.012 s; **the dense solves 0.022 s (4 %)**.
- **Where the backward goes** (torch profiler, 0.401 s of self CPU time): elementwise mul 16 % (8943 calls),
  index_put 13 %, MulBackward 11 %, mm 8 %, add_ 8 %, bmm 2 %. Dense products (mm, bmm) together 10 %.
- **At 440 atoms** (the (2,2,1) mesh, forward 6.319 s): the Fermi band 1.891 s (30 %; its pair corrections
  1.427 s), K's blocks 1.254 s (20 %), the bound elimination 1.010 s (16 %; dense solves 0.792 s), the clusters
  0.901 s (14 %), induction 0.850 s (13 %).
- **Scaling of the electrostatics (fw + bw) in the evaluated cell:** 1.180 → 2.971 → 12.857 s for 110 → 220 → 440
  atoms, local exponents 1.33 and 2.11. The local model stays at 0.35–0.41 s, because a mesh replicates only
  inside the electrostatics. Peak memory 2.0 → 3.4 → 5.4 GB.
- **Threads:** 4 threads are 1.2–1.6× faster than 1. That is consistent with many small operations rather than
  large dense ones at this size; the overhead per operation is not measured separately.

**Hypothesis scored.** (b) is 0.76–0.80 of (a) at (1,1,1), so the first refutation condition is not met. Inside
(b), the dense operator's build and solves are 42 % of the forward (38 % + 4 %), and its dense products 10 % of
the backward; the backward of the build is elementwise and this profiler cannot separate it from the other pair
kernels. **Refuted as stated at 110 atoms:** the dense solve is not the bottleneck. The time is spread over
building pair blocks (K, the free-charge blocks, the induction columns) in many small operations. At 440 atoms the
dense solves grow to 12.5 % of the forward and the electrostatics to 97 % of the whole.

**What S1 does not say.** It has no DFTB or DFT timing (S2 needs a decision for a download). It does not show
what a faster route would give; no method was changed.

### Pre-registered: S2, a DFTB-class timing on the same cell (written 2026-11-11 19:02 IST, before the download and the runs; the download was approved the same day)

- **Code.** DFTB+ 25.1 (conda-forge, `dftbplus-25.1-nompi`) in its own conda environment `dftb-timing`; the
  project's environment is not touched. Hamiltonian: GFN1-xTB through DFTB+'s tblite (ESTABLISHED: Grimme, Bannwarth
  and Shushkov, J. Chem. Theory Comput. 13, 1989 (2017), cited from memory). Reason: I found no Slater–Koster set that
  covers Au, Mg and O together (`auorg` has Au; `3ob-3-1` has Mg and O; sets must not be mixed), so a classic
  SCC-DFTB run is not possible without generating parameters.
- **Same as S1.** The undoped wetting reference cell (110 atoms, pbc (T,T,T), charge 0), the M2 Pro, energy and
  forces, 4 and 1 threads (`OMP_NUM_THREADS`), five geometries displaced by the same seeded 0.01 Å Gaussian.
- **Settings.** SCC from scratch at every geometry (TRACE has no warm start from a previous step), default SCC
  tolerance, at most 300 iterations, Fermi filling at 300 K; k-points Γ and 3 × 3 × 1 (Γ-centred; the reference
  DFT's grid, R24). Also, separately reported: the restart from the previous geometry's charges, which is what an
  MD code does.
- **Measured:** the wall time of the DFTB+ process and DFTB+'s own total timer, the SCC iteration count, the
  orbital count. A run that does not converge is reported as such and not timed.
- **Hypothesis (my guess, stated in the discussion before S1).** At Γ, GFN1-xTB energy and forces on this cell
  take no more than TRACE's 1.8 s (one thread, S1). Refuted if the one-thread Γ median exceeds 1.8 s.
- **What it cannot say.** Nothing about accuracy: GFN1-xTB is not fitted to this system and is not a benchmark
  reference. It is a cost reference only.

### S2 scored (2026-11-11 19:23 IST; `probe_speed_dftb.py`; output `runs/speed_2026-11-11/dftb/`)

**Two errors in the set-up, both caught before any number was used.** (1) DFTB+ 25.1 rejects
`Analysis { CalculateForces }` (it halted at parsing); the first attempt is kept as
`INVALID_k1_t4_forces_keyword_rejected`, and the probe uses `PrintForces = Yes`, which prints the total forces
(and the stress, which DFTB+ also computes). (2) My first batch command did not run (zsh did not split a command
held in a variable); it produced error lines only, overwritten by the rerun. One trial run (`try_forces`, geometry
g0 at Γ, 4 threads, 13.44 s) reproduces g0 of the set to 0.1 s and its energy to all printed digits.

**Measured** (DFTB+ 25.1, GFN1-xTB, 450 orbitals, 454 electrons; S1's cell and geometries; SCC converged in every
run; median wall time over g1–g5, range in brackets):

| case | threads | DFTB+ wall, s | SCC iterations | TRACE (S1), s | DFTB+ / TRACE |
|---|---|---|---|---|---|
| Γ, SCC from scratch | 1 | 14.315 [14.021, 16.304] | 52–64 | 1.836 (stored), 1.844 (extensive) | 7.8 |
| Γ, SCC from scratch | 4 | 11.676 [11.404, 13.545] | 52–64 | 1.324 (stored), 1.528 (extensive) | 8.8, 7.6 |
| Γ, restart from the previous charges (MD-like) | 4 | 7.091 [6.851, 7.680] | 27–30 | 1.324, 1.528 | 5.4, 4.6 |
| 3 × 3 × 1 (5 irreducible k-points), from scratch | 4 | 34.734 [33.634, 35.632] | 41–42 | no route; (2,2,1): 13.265 | (2.6 against (2,2,1)) |
| 3 × 3 × 1, from scratch | 1 | 44.512 [42.868, 45.641] | 41–42 | no route | — |

- **Hypothesis refuted.** I had guessed that DFTB at Γ takes no more than TRACE's 1.8 s; it takes 14.3 s on one
  thread.
- **Where DFTB+'s time goes** (its own timer, g1 at Γ, 4 threads): SCC 10.49 s (84.5 %), 0.18 s per iteration;
  pre-SCC 0.60 s; post-SCC (forces and stress) 1.32 s. Why an iteration costs 0.18 s with 450 orbitals is not
  measured.
- **Candidate artifacts, named.** The default mixer needs 52–64 iterations from scratch; a tuned mixer would need
  fewer. An estimate, not measured: at 15 iterations DFTB+ would take about 0.6 + 15 × 0.18 + 1.3 ≈ 4.6 s at Γ,
  still about 3× TRACE. DFTB+ also computes the stress; TRACE was timed without it. GFN1-xTB is a DFTB-class method,
  not SCC-DFTB with Slater–Koster tables, whose cost per iteration is not measured here.
- **The k-mesh is where the two differ in kind.** DFTB+ pays 3.0× its Γ time for 3 × 3 × 1 (Bloch's theorem: five
  cell-sized problems, and fewer SCC iterations). TRACE pays 8.7× its Γ time for (2,2,1), through a 440-atom
  supercell, and refuses (3,3,1).
- **Not a benchmark number.** GFN1-xTB is not fitted to this system; its energies are not compared with anything.
  (Its own Γ and 3 × 3 × 1 total energies differ by 0.2373 Hartree per cell; recorded, not used.)

**Against the project's design rule** (classical and much faster than DFTB): at Γ on this cell TRACE is 7.6–8.8× faster than
GFN1-xTB from scratch and 4.6–5.4× faster than an MD-style restart, on the same CPU and threads. That is faster, not
"much much faster". At the reference DFT's k-point sampling TRACE has no route at all.

### Addendum to "M1c scored": the correction to R15 part 4 and the estimates beyond (2, 2, 1) (written 2026-11-12 09:12 IST; numbers of 2026-11-11, `mesh_lattice_sum_fit.txt`)

This part of the M1c record was computed on 2026-11-11 and not written then; S1 and S2 came between.

**The exact limit of a lateral mesh.** A mesh (M, M, 1) leaves the chain of images along z: Λ_xx → +2ζ(3)/c³ =
+0.129790e-3 Å⁻³ and Λ_zz → −4ζ(3)/c³ = −0.259579e-3 Å⁻³ at c = 26.4589 Å. Checked at 12×12:
Λ_xx + 2π/V_M = +0.129790e-3, Λ_zz = −0.259579e-3.

**R15 part 4, corrected for the in-plane dipole.** Of Λ(1×1) − limit a square mesh keeps:

| c | component | 2×2 | 3×3 | 4×4 |
|---|---|---|---|---|
| 26.4589 Å | zz | 4.93 % | 0.42 % | 0.04 % |
| 26.4589 Å | xx | 14.28 % | 5.40 % | 2.93 % |
| 40.0 Å | zz | 7.59 % | 1.17 % | 0.22 % |
| 40.0 Å | xx | 13.04 % | 4.28 % | 2.11 % |

R15's z numbers (4.9 %, 0.4 %, 0.04 %) stand. Its x numbers (12.5 %, 3.7 %, 1.6 %) coincide with M⁻³ and are
replaced: the in-plane remainder is the two-dimensional Lorentz term −2π/V_M ∝ M⁻², which is 81.6 %, 95.8 % and
99.3 % of Λ_xx − limit at 2×2, 3×3, 4×4. How R15's x numbers were obtained I cannot reconstruct.

**Estimates, not measured** (the fit's constants; E(wetting) − E(non-wetting), undoped, fixed reference geometries):

| step | s43 | s42 |
|---|---|---|
| (2,2,1) → (3,3,1) | +2.230 meV | +3.329 meV |
| (3,3,1) → (4,4,1) | +0.285 meV | +0.497 meV |
| (2,2,1) → lateral limit | +2.696 meV | +4.217 meV |
| (3,3,1) → lateral limit | +0.467 meV | +0.888 meV |
| the chain through the vacuum that the lateral limit keeps | +1.610 meV | +2.172 meV |

The fit reproduces the upright dimer's measured (1,1,1) → (2,2,1) step to 0.6 % and 0.1 % and the flat dimer's to
24 % and 3 %, so the flat dimer's part of each estimate (about a third) carries that error. (3, 3, 1) is **not
measured**; relaxed geometries and doped cells are not measured at these meshes; nothing is trained.

**Wording, from here on.** TRACE has no Bloch states and samples no k-points. "Γ" in these records means the
single cell under periodic boundaries (every image in the same state), and "mesh (M₁, M₂, M₃)" means the
Born–von Kármán supercell of SPEC 37 evaluated per cell. The names are borrowed from band theory, where the two
are equivalent.

### R25 and pre-registered M1d: the refusal of (3, 3, 1) is a guard on a matrix this route does not build (written 2026-11-12 09:14 IST, before the run)

- **Reading of the code (R25).** The refusal comes from `trace_electrostatics.energy.assemble`: "the solver would
  build a dense n_m × n_m operator K" (35793² doubles, 10.25 GB, for 990 atoms). The redox route does not build
  that K. `eliminate_bound_sector(method='condensed')` (the default, and step 31's setting) inverts K through its
  SQE structure (`_CondensedK`: K = blockdiag(D, P) + Vᵀ C V), with arrays of at most n × n_b and 3n × 3n:
  0.26 GB and 0.07 GB at 990 atoms. The guard's premise holds for `trace_electrostatics`' own solver and for
  `method='dense'`, not for this route. Every record of this line that says "the dense route refuses 990 atoms"
  states the guard, not a measured need. **My error:** I repeated that statement from R15 on without reading what
  the route allocates (per the rule-out-artifacts policy: the operator's docstring says "never formed").
- **Candidate artifact of this reading, to be ruled out by the run:** some other allocation of the supercell
  evaluation may still grow as n_m². S1 measured the peak resident memory at 110, 220, 440 atoms as 2.0, 3.4,
  5.4 GB (close to linear, +2.0 GB for the last doubling). The Mac has 16 GB.
- **M1d (M1's (3, 3, 1) leg, as pre-registered on 2026-11-11 10:50 IST, with one change):**
  `max_operator_bytes` raised to 12e9 in the probe (the documented knob; no package change), both seeds, undoped,
  fixed reference geometries, `'onsite'`, clusters with the closure, `'fermi'`. M1's reading (a) / (b) / (c)
  stands as written: (a) |D23| ≤ 5 meV on both seeds.
- **Predictions, written on 2026-11-11 and in the addendum above before this run** (the lattice-sum fit, constants
  from the first x and y steps): D23 = **+2.230 meV (s43), +3.329 meV (s42)**; per geometry, upright −1.793 /
  −2.270 meV, flat +0.437 / +1.059 meV. Pass, fixed now: the upright dimer's step within 15 % of its prediction
  on both seeds (its (1,1,1) → (2,2,1) step was reproduced to 0.6 % and 0.1 %); the flat dimer's within a factor
  1.5 (its fit is off by up to 26 %). M1's own prediction (D23 / D12 = +0.1005) is also scored.
- **Abort condition.** The run is stopped by hand if its resident memory passes 13 GB; that is then recorded as
  the measured need.

### M1d scored: (3, 3, 1) is measured, and M1's reading (a) holds (2026-11-12 09:26 IST; output `runs/extensivity_2026-11-10/mesh_convergence_m331_undoped_{s43,s42}.txt`)

**R25 confirmed by the run.** With `max_operator_bytes = 12e9` the 990-atom supercell evaluates on both seeds:
127–131 s per energy-and-forces evaluation (4 CPU threads), peak resident memory 5.70 GB (s43) and 8.39 GB (s42).
(macOS's resident size leaves out compressed pages, so these are lower bounds; the 13 GB abort was not reached, but
my `ps` monitor matched no process and reported nothing, so the abort was not armed. The (2,2,1) values of both
seeds reproduce the 2026-11-11 records to the digits printed.)

| undoped, fixed reference geometries, meV | s43 | s42 |
|---|---|---|
| E(wetting) − E(non-wetting), (2,2,1) | +947.9369 | +902.9414 |
| E(wetting) − E(non-wetting), (3,3,1) | **+950.3246** | **+906.3054** |
| D23, measured | **+2.3876** | **+3.3640** |
| D23, predicted by the lattice-sum fit (written before the run) | +2.230 | +3.329 |
| upright dimer's step, measured / predicted | −1.7536 / −1.793 (2.2 %) | −2.1484 / −2.270 (5.4 %) |
| flat dimer's step, measured / predicted | +0.6341 / +0.437 (×1.45) | +1.2156 / +1.059 (×1.15) |
| D23 / D12 (M1's own prediction: +0.1005) | +0.0585 | +0.0589 |

- **M1d's criteria:** the upright dimer's step within 15 % on both seeds: **PASS** (2.2 %, 5.4 %). The flat dimer's
  within a factor 1.5: **PASS** (1.45, 1.15; s43 only just).
- **M1's reading (a): |D23| ≤ 5 meV on both seeds. Met.** (2, 2, 1) is converged to 5 meV for the undoped pair, and
  a training at a mesh may use it. By the same fit, (3,3,1) is +0.47 / +0.89 meV from the lateral limit (estimate).
- **M1's own prediction** (one R⁻³ pair term, D23 / D12 = +0.1005) is **not supported**: +0.0585 and +0.0589.
- **Cost, and the project's speed rule.** At the reference's sampling the supercell route takes 127 s, against 34.7 s
  for DFTB+ on 3 × 3 × 1 k-points (S2): **3.7× slower than the tight-binding code there**, where the single cell
  is 7.6–8.8× faster. (2,2,1) takes 12–16 s.
- **Replaced statements.** "(3, 3, 1) is refused by the dense route" / "cannot be evaluated" (R15 part 4, M1's first
  result, M1b, the M1c records, S1, S2, SPEC §0 / §37 / §38.4, the CHANGELOG): the refusal is the
  guard of R25; with the limit raised the mesh runs. S2's "TRACE has no route" for 3 × 3 × 1 is replaced by the
  127 s above.
- **Not measured:** relaxed geometries and the doped cells at (3,3,1); nothing is trained.

### The guard follows the route (R25 fixed in code, 2026-11-12 09:42 IST)

`trace_redox.bound.assemble_neutral` takes `elimination` (None: the legacy guard, for every caller that does not
name it; `'dense'`: the same; `'condensed'`: `max_operator_bytes` is compared with one [n, n_m] array, the largest
the route holds, and the quadrupole pair tensors keep their own bound). `RedoxElectrostatics` passes its
`bound_elimination`. Tests first (`tests/redox/test_redox_operator_guard.py`, 5; 3 failed before the change, as
they should), then green; suites: `tests/redox` **949 passed**, the rest **1045 passed, 4 skipped**. At the default
4 GB the 990-atom supercell (0.28 GB per array) is no longer refused. No energy changes: the guard is not arithmetic.

### R26: seed s42 learned an O–O hopping of +17.7 eV, and nothing in the training bounds it (2026-11-12 09:42 IST; `probe_ext_prior_records.py`, `probe_coupling_sizes.py`; output `runs/extensivity_2026-11-10/prior_records_{s43,s42}.txt`, `coupling_sizes_{s43,s42}.txt`)

**[Correction, 2026-11-12 10:52 IST: this defect was not new. See "R26 corrected" below: it was recorded on 2026-11-06 for step 20's s42 (CHANGELOG Round 33).]**

Found while checking whether step 31's response-prior structures would get an exact response record in the
clusters space (SPEC 36 L5), which a retraining with the extensive forms needs.

**Measured, frozen step 31 weights, settings as stored.**

| | s43 | s42 |
|---|---|---|
| O–O coupling at 3.015 Å, benchmark cells (1872 directed edges) | −1.32 … 0 eV | 0 … **+17.72 eV** |
| edges above Harrison's ceiling B(r) (SPEC 27; 2.64 eV at 3.015 Å) | 0 | **576 O–O edges in every benchmark cell** |
| largest \|τ\| / B(r) over all pairs | 0.69 (Au–Au, 2.64 eV at 2.545 Å) | **6.5** (O–O) |
| largest off-diagonal of H actually used, four benchmark cells | 2.12–2.64 eV (Au–Au) | 1.78–2.07 eV (Au–Au) |
| MgO bulk prior cell (8 atoms): configurations | 1 | 5 (reference + one O → band transfer per O, at +2.825 eV) |
| MgO bulk prior cell: largest off-diagonal of H | 0 | **68.72 eV** (= 4 images × 17.18 eV, O-hole hopping) |
| MgO bulk prior cell: lowest eigenvalue below the reference | 0 | **−203.33 eV** |
| MgO bulk prior cell: energy, as stored | −109286.956 eV | **−109489.844 eV** |

- On s42's benchmark cells the O–O coupling does not enter the single cell's Hamiltonian as stored: no selected
  configuration has a hole on an O. It enters wherever one is selected: on the 8-atom prior cell, where the four
  O-hole states couple through four periodic images each (the symmetric state lies at Δ − 12τ).
- **How it was learned (hypothesis, consistent with the facts, not tested by a training):** the dielectric prior
  is the only term of the loss in which an O-hole configuration appears; its gradient reaches τ(O–O); no energy
  or force label does. On s43 the prior cell has one configuration and τ(O–O) gets no gradient at all.
- **SPEC 27's bound was off** (`coupling_bound` None in step 31, as in every run since step 21). Its record in
  SPEC §0, "off: the identity on every benchmark structure", **is false for s42** and is replaced. The bound is
  the identity below 0.75 B: bitwise on s43's reference cells (largest ratio with r₀ = 0.5 Å: 0.72).
- Even at the ceiling an O–O hopping of +2.64 eV would put the prior cell's O-hole state 29 eV below the
  reference. The anchored value (SPEC 31, η_pp average) is 0.44 eV; s43's −0.35 eV has that size.
- The clusters space refuses the response record on s42's prior cell (four clusters of two; L5); with `'fermi'` it
  is one cluster of 15 configurations at −109504.97 eV. s43's prior cell has one configuration in every setting.

### Pre-registered: B1, does the unbounded coupling cause s42's supercell failure? (written 2026-11-12 09:42 IST, before the probe's options and the run)

- **Benchmark number:** Table II's wet_d on s42 at the (2,2,1) mesh: refused at the cap of 512, +603.69 meV at
  2048 (R23), against −22.95 meV on s43 and −2.7 meV in DFT.
- **Hypothesis H.** The +17.7 eV O–O couplings pass SPEC 21's amplitude selection in the supercell and link the
  large clusters R23 found on s42 (33, 6, 6 rows; 127 linked rows closing to 1218).
- **Test, frozen weights, no training, one change.** `mod.coupling.bound = 'harrison'` (SPEC 27, as built and
  tested in October) on the frozen head; `'onsite'`, clusters with the closure, `'fermi'`; fixed reference
  geometries; the doped pair at (1,1,1) and (2,2,1), default cap 512 and, if refused, 2048; with and without the
  bound. Controls: s43 with the bound must reproduce its own energies on all cells (the bound is the identity
  below the knee), and s42's undoped pair is reported too.
- **Reading, fixed now.** H is **supported** if with the bound s42's doped (2,2,1) supercells run at the default
  cap with every cluster ≤ 8 rows, and E(wetting) − E(non-wetting) at the mesh moves by more than 100 meV toward
  s43's. H is **refuted** if the cluster sizes and the refusal are unchanged. Anything else is reported as mixed.
- **What it would not show.** That s42 with the bound is a good model: it was trained without it, and its other
  weights have adapted to whatever the coupling did to the prior. Only a training with the bound tests that.

### B1 scored: H is refuted (2026-11-12 09:59 IST; output `runs/extensivity_2026-11-10/b1_{s43,s42}_{nobound,bound}.txt`)

- **Control, s43:** with SPEC 27's bound every energy, force maximum and cluster size is identical to the run
  without it, to the digits printed (undoped and doped, (1,1,1) and (2,2,1)). The bound is the identity on s43.
- **s42:** identical too. Undoped +845.7884 and +902.9414 meV; doped (1,1,1) −162.5843 meV; doped (2,2,1) wetting
  E/cell −1480189.522169584 eV with clusters [6, 6, 33]; doped (2,2,1) non-wetting **refused at the cap in both
  runs** (127 linked, 1218 closed).
- **Falsifying check that the bound was active** (per the rule-out-artifacts policy; one script, not kept as a probe): on s42 the largest
  \|τ\| on any edge falls from 17.7178 to 2.6448 eV and the prior cell's energy moves from −109489.843823 to
  −109315.395984 eV (still 28.44 eV below s43's: the ceiling itself is 6× the anchored size, R26). At cap 2048
  the doped non-wetting supercell gives E/cell −1480190.115862320 eV with one cluster of 1218 rows **with and
  without the bound**, and the largest off-diagonal of its Hamiltonian is 2.0723 eV (Au–Au) in both.
- **Reading as fixed:** cluster sizes and the refusal are unchanged, so **H is refuted**. The O–O couplings do not
  enter s42's supercell clusters. R23's reading stands: the supercell's band takes the dimers' electrons.
- **What remains of R26:** the unbounded coupling acts on the response-prior cell only. It is a defect of the
  training protocol for a seed like s42 (a prior evaluated on an 8-atom cell, with nothing bounding the one
  parameter the prior alone sees), not the cause of the supercell failure.

### SPEC 36.10 built: the response record of several clusters (2026-11-12 10:12 IST; lifts the second half of SPEC 36's L5)

- **Benchmark path it serves:** the retraining with the extensive forms (planned one change per run) on step 31's
  protocol, which carries two response priors. In the clusters space a prior structure with two clusters had no
  record, and `train.py` raises on that; s42's bulk MgO prior cell is such a structure (R26).
- **Statement.** α = α_b + Σ_X w_X α_ens(H_X, M_X, δα_X) over the reference, the clusters and the pair spaces, with
  the weights of F's own combination (they sum to one). Exact, because F is linear in its terms' free energies.
- **Tests first.** `tests/redox/test_redox_cluster_response.py` (8) and one training-level test in
  `test_redox_polarizability_prior.py`; all red before the code (ImportError / the single-Hamiltonian read in
  `train.py` / a factor 2 at a mesh). The record equals the module's finite-field −∂²E/∂F² to 1e-5 relative in
  five cases with two clusters; without the pair terms it does not.
- **Three statements of my own tests were wrong and were corrected against measurement, not against the code:**
  (1) a gradient through the couplings on Na₂ in the window, whose two configurations differ by two electrons (no
  coupling is in its Hamiltonian; measured: scaling the coupling weights changes nothing); (2) which cases hold
  couplings (the amplitude selection brings them in on Na₂); (3) a 1e-9 tolerance at a mesh, where the response's
  own finite differences amplify round-off to 1.9e-7 (now 16 ε |H| / (h/2)²).
- **On the real prior cell, frozen s42** (`prior_records_s42_after_3610.txt`): in the clusters space the record now
  has 11 terms (reference, four clusters, six pairs); α_b = 0.8507 and α_ens = +0.1662 e²Å²/eV per component. As
  stored (global space, the −203 eV state of R26): α_ens = −0.0028. These are sizes on a frozen model, not scores.
- **Limit, stated:** under periodic boundaries the record is not checked against a finite field (the module has
  none there), and a multi-move configuration's dipole uses SPEC 30's minimal-image convention, which is ambiguous
  when neighbours sit at exactly half a cell vector, as in the 8-atom prior cell.

**Suites after SPEC 36.10 (2026-11-12 10:25 IST):** `tests/redox` 958 passed (949 + 8 + 1); `pytest tests --ignore=tests/redox` 1045 passed, 4 skipped.

## Pre-registered: step 32, SPEC 27's coupling bound through the whole training (written 2026-11-12 10:35 IST, before the runs; decision of 2026-11-12: pre-register and launch it on GPU host 2)

**A fact found while preparing the run, and what it changes.** I had proposed "step 31's protocol with only the
bound switched on". Step 31 is a 20-epoch warm start from step 26's weights, and s42's O–O coupling is already
+17.31 eV in that warm start (step 31 left it at +17.18; s43: −0.30 → −0.35). The runaway happened in step 26's
cold start. A warm start with the bound would begin by clipping a learned 17 eV to 2.6 eV and could not show
whether the bound prevents the runaway. So step 32 repeats the **whole two-stage training that produced step 31's
models**, with the bound on from the first epoch. This is longer than what I first described; it is the
design that tests what I said the run would test.

**Runs** (GPU host 2, float64; both seeds of a stage together; `watch_step32.sh` on the Mac chains and scores):
- **32a** `…_pfp_bopw_cbh_bondlaw_r5_s{42,43}`: step 26's config (cold start, 40 epochs, lr 1e-3, energy weight 10).
- **32b** `…_pfp_bopw_cbh_e1kw_bondlaw_r5_s{42,43}`: step 31's config (32a's best weights only, epoch 0; energy
  weight 1000; 20 epochs; lr 1e-4 cosine to 5e-6).
- **The one change:** `redox.coupling_bound: harrison` (SPEC 27, built and tested in October: the identity below
  0.75 B(r), C² saturation to B(r) = 3.24 ħ²/(m (r² + r₀²)), r₀ = 0.5 Å). Every path is replaced in every string
  (`check_run_config.py`: OK on all four); the key-by-key difference to the source configs is the bound, the
  explicit `state_polarizability_stiffness: block` (the statement steps 26 and 31 trained; the default changed on
  2026-11-10) and, for 32b, the warm-start file.
- **Not the same as steps 26 / 31 besides the bound:** the machine (the GPU of GPU host 2 against the Mac's CPU) and the
  code as of today (the options added since step 26 are off; s43's stored Table II was reproduced on this code,
  X1 (c)). **s43 is the control for both:** on 24 training frames the bound changes 2 of 174,470 couplings on step
  26's and step 31's s43 weights, by at most 0.0001 eV; on step 26's s42 it changes 13,824 (7.9 %), by up to 17.37 eV.

**Benchmark moved:** 4G Table II (wet_u, wet_d, T5), each seed, at stage 32b; comparators step 31 and, for 32a,
step 26, same seeds.

| id | criterion |
|---|---|
| **Z1** | s42, end of 32a and of 32b: no state of the 8-atom ε∞ prior cell lies more than 5 eV below its reference configuration (step 31 s42: 203 eV) |
| **Z2** | reported, with its reading fixed now: s42's largest \|τ(O–O)\| / B at 3.015 Å on the benchmark cells. Below 0.75: the coupling stayed below the knee. Above 0.90: it sits at the ceiling, the prior still drives it, and the bound alone is not the fix |
| **Z3** | s43 (control), 32b: Y1–Y5 of step 31 pass, as step 31's s43 did |
| **Z4** | s42, 32b: Y1–Y5 of step 31 (step 31's s42 failed Y1, −73.3 meV against 30, and Y3, wet_d −177.4 against ±150 of −2.7) |
| Z5 | reported: each seed's Table II and its distance from step 31's; test E and F RMSE; ε∞ (T6); T2; relaxation failures; time per epoch on the GPU |

**Predictions (hypotheses).** (i) Z3 passes; s43's numbers differ from step 31's by the run-to-run noise of this
protocol, which is not known (no same-seed repeat exists) and which this control measures. (ii) Z2 on s42: I
expect the ceiling (above 0.90). Even at B the prior cell's O-hole state would lie about 29 eV below the
reference (B1's check), so **I expect Z1 to fail on s42**; the run then says that the prior needs a bound of the
anchored size (SPEC 31's 0.44 eV for O–O) or an evaluation where the images are not in phase. If instead Z1 and
Z2 pass, the runaway needed the unbounded range and the bound is the fix. (iii) Z4: no prediction of direction.
Whether s42's benchmark errors are tied to its prior cell at all is what the run tests.

**PASS** = Z1, Z3 and Z4. A crash, a NaN stop or a failed T5 relaxation is recorded and leaves the affected
criteria unscored (FAIL). **Expected time:** about 17 min per epoch for two runs on this GPU (step 28b's record on
GPU host 1): 32a about 11 h, 32b about 6 h, then the scoring on the Mac.

**Step 32 launched 2026-11-12 10:38 IST** (GPU host 2, free at launch: no compute process). 32a: s42 and s43
started 10:36 IST; GPU at 89 %, 2.1–2.2 GB each; no traceback at start; 1.01 s per forward with the two
together. The tree was synced add/update only; the training and validation files have the same md5 on both
machines. **A third difference to steps 26 / 31, found at launch:** GPU host 2's environment is torch 2.11.0+cu128
(the Mac runs 2.2.0); s43 is the control for it as well.
- `chain_step32.sh` (on GPU host 2, log `runs/chain_step32.log` there): when both 32a runs have finished
  cleanly it writes the weights-only warm starts and launches 32b on a GPU that runs no compute process.
- `watch_step32.sh` (on the Mac, log `runs/watch_step32.log`): copies each stage's `model.pt` and
  `log.txt` back and scores on the Mac (`scores/Z1_Z2_couplings.txt`, `T5_wetting.txt`; for 32b also the transfer
  set and the class residuals). ε∞ (T6) is not in the watcher and is computed afterwards, as for step 31.
- `probe_coupling_sizes.py` now prints Z1's quantity. Check on step 31's s42: the prior cell's lowest state lies
  203.3339 eV below its reference; the four benchmark cells' 1.65–2.17 eV.

### R26 corrected: the O–O hole coupling was already on record (2026-11-12 10:52 IST)

I presented R26 as a finding of 2026-11-12. It is the defect recorded (CHANGELOG Round 33) on
2026-11-06 for **step 20's s42** and that the phonon line met twice (`benchmarks_phonons/README.md`: criterion P0,
planned step 2; the step 20 control on the PBEsol cells, blocked at its start by this cliff):
- the learned O–O hole coupling was 28.08 eV per edge at 3.016 Å; in the 8-atom cell four images give 112.3 eV and
  a state 334.5 eV below the frame;
- **SPEC 27's bound post hoc: 2.64 eV per edge, −29.2 eV; "not enough"**; on the 64-atom cell at 4.217 Å: −5.8 eV
  and a frame population of 0.12; "not a cure";
- seed 43 selects no hole configurations in bulk and is clean;
- the recorded hypothesis: a scalar τ cannot represent p-hole hopping.

What R26 adds is the same measurement on **step 31's** weights (17.7 eV instead of 28.1; 203 eV instead of 334.5),
the statement that the coupling does not enter the slab benchmark cells' own Hamiltonians, B1 (it is not the cause
of s42's supercell failure) and the correction of SPEC §0's "identity on every benchmark structure". My value for
the bound, −28.44 eV, agrees with the record's −29.2 eV on step 20.

**Consequence for step 32.** Its prediction "Z1 fails on s42 even at the ceiling" is the recorded post-hoc result,
not a new one. What step 32 still tests, and what was not on record: whether the coupling is driven to the ceiling
when the bound is on **during** the training, and what that does to s42's Table II. A fix of the defect itself
(an anchored or angular O–O hop, or no O–O hops on the formal anion sublattice: the phonon line's planned step 2) is
not built.

**My error, for the fourth time in this line:** I did not reread the record (here §5g and the phonon README) before
writing a finding. R15 → M1's design, R15 → M1b's score, "refused by the dense route" → R25, and now §5g → R26.

**Step 32, two events on 2026-11-12 (recorded 2026-11-12 14:33 IST).**
- **32a s43 crashed at about 13:35 IST** in epoch 15 (sub-step 281, structure index 158):
  `torch.OutOfMemoryError: CUDA out of memory` (15.47 GB card; s42 held 3.36 GB, s43 had grown from 2.2 GB at the
  start to 6.4 GB at 12:43 and to the rest). Its log shows structures with up to 373 configurations in the global
  space (s42 logs none above the print threshold). Its last checkpoint is `model.pt`, epoch 14, with the optimizer
  and scheduler state; `nanguard_exception.pt` is kept. **32a s42 runs on** (epoch 20 at 14:28, 3.4 GB). Not
  restarted yet: whether to resume s43 (alone on the GPU, from epoch 14) or to finish step 32 on s42 only is the
  decision to make against the GPU time of the single model below.
- **The hand-over script had a bug that would have stopped 32b tonight:** `chain_step32.sh` wrote the warm-start
  files without the checkpoint's `config`, and `train.py` refuses such a file ("Cannot resume architecture v1
  weights in the v2 model"). I had misread step 31's warm file as holding two keys (my own listing had left
  `config` out). Found by the smoke test of the single model at 14:19; fixed at 14:24: the script on GPU host 2
  was stopped by exact PID and replaced (the first version kept as
  `chain_step32_v1_no_config_key.sh`). The trainings were not touched. As written it launches 32b only if both
  32a runs finished, so with s43 crashed it will report "not launched" unless it is replaced again.

## Pre-registered: step 33, the single model with every term on (written 2026-11-12 14:33 IST, before the launch; decision of 2026-11-12: run the single model with everything on; goal stated the same day: one equation, one model, one code for every benchmark)

**Many changes in one run, by decision.** The project's one-change-per-run policy is set aside here: if this run
fails, it does not say which term failed. The staged sequence (the extensive forms one change per run) stays the fallback.

**The model.** Step 31's redox block (formal anchors, couplings, the band unit, Gordon–Kim Pauli and penetration,
carrier induction, state polarizabilities, the bond-oriented Pauli law and wall of SPEC 33) **plus**
`frame_induction_damping` (SPEC 34), `frame_polarizability_anchor` with `frame_polarizability_scope: ions`
(SPEC 35, 35.1), `coupling_bound: harrison` (SPEC 27), `state_polarizability_stiffness: onsite` (SPEC 28.6),
`diabatic_space: clusters` (SPEC 36, 36.7, 36.10) and `band_form: fermi` (SPEC 38). **Not on in training:** the
supercell sampling of SPEC 37, which costs about 9× per step; the trained model is scored on the single cell and
on the 2×2 supercell.

**Data and labels.** `benchmarks_phonons/data/AuMgO_bulkaugw_{train,valid}.xyz`: the 300 / 76 4G slabs (PBE, energies
and forces) and 136 / 10 displaced PBE bulk MgO cells of 64 atoms (forces only, `force_weight` 122.4; the
reweighted set of 2026-11-07). ε∞ prior: `MgO_bulk_eps_prior_pbe.xyz` (PBE's DFPT 3.235 at 4.25566 Å; by decision,
the DFT value), weight 0.05. The Au-atom polarizability prior, weight 0.3, as before.

**Runs** (GPU host 2, both seeds). **A** `AuMgO_single_all_bondlaw_r5_s{42,43}`: warm start from step 31's same-seed
best weights (epoch 20 / 17; `log_alpha_scale` = 0 added), 40 epochs, lr 1e-3 cosine to 5e-5, energy weight 10.
**B** `AuMgO_single_all_e1kw_bondlaw_r5_s{42,43}`: A's best weights, step 31's recipe (20 epochs, lr 1e-4 to 5e-6,
energy weight 1000). The key-by-key difference to step 31's config is listed by the derivation and holds only
the terms, the data, the prior file and A's recipe; `check_run_config.py` OK on all four. A warm start, because
the slab-plus-bulk cold starts of 2026-11-07 died on the bulk cells at the configuration cap.

**Pre-flight, frozen step 31 weights with every term on** (`probe_single_preflight.py`;
`runs/extensivity_2026-11-10/single_preflight.txt`, `single_preflight_terms.txt`): no refusal on 6 slabs, 6 bulk cells
and the prior cell, either seed; bulk cells stay in the frame; 1.2–4 s per structure on the Mac. Slab force RMSE on
those 6 training slabs, eV/Å (s43 / s42): as trained 0.087 / 0.072; with the extensive forms and the bound
0.086 / 0.071 (they cost nothing); + the anchor 0.29 / 0.49; + the damping 0.84 / 0.70; + both 0.83 / 0.40. So the
run starts 5–10× off on the slab forces and has to relearn the induction; that is its risk. **Smoke test** (Mac,
the real `train.py` path, 3 optimizer steps, then stopped): the first attempt failed on the warm file's missing
`config` (above); with it the run starts, gradient norms 6.2, 4.1, 2.3, no refusal.

| id | criterion (each seed, after B) | comparators |
|---|---|---|
| **S1** | both stages finish: no refusal, no NaN stop, no memory failure | |
| **S2** | Table II by the T5 protocol, single cell: wet_u within ±100 meV of +928.9, wet_d within ±150 of −2.7, T5 within ±15 % of −931.6 (step 31's Y2–Y4) | step 31 s43 +920.4 / −75.7 / 106.9 %; s42 +851.0 / −177.4 / 110.4 %; 4G +975.4 / −41.0 / 109.1 % |
| **S3** | 4G test force RMSE ≤ 0.10 (undoped) and ≤ 0.13 eV/Å (doped) | step 31 0.078 / 0.105 (s43), 0.061 / 0.087 (s42); step 28b 0.155–0.163 undoped |
| **S4** | phonons against PBE (`score_phonons.py`, a = 4.2557 Å): P0, P1 (Z* 1.90–2.10), P2 (ε∞ 2.9–3.5), P4 (LO 660–700) pass, and at least 6 of 7 | step 31 family: Z* 2.4–2.5, LO above 800; the PBEsol phonon model at epoch 9 today: 7/7 against PBEsol |

**[Corrected 2026-11-16 16:00 IST; the row above is kept as written.]** "step 31 family" was step 20's measured phonons, not step 31's. Step 31 itself, scored for the first time: s42 Z* 2.402, LO 807.9, 3 / 7; s43 Z* 2.534, LO 851.5, 2 / 7 (benchmarks_phonons README, "Step 31's phonons, scored for the first time").
| **S5** | the crystal factor ends within 0.85–0.95 and the prior cell's ε∞ within 0.05 of 3.235 | |
| S6 | reported: the undoped class offset (Y1), Table II on the 2×2 supercell, T2, test E RMSE, the largest \|τ\|/B, minutes per epoch, peak GPU memory | |

**PASS** = S1–S5 on both seeds. **Predictions (hypotheses).** S1, S4, S5: I expect them to pass (the phonon physics
held in two independent trainings; the factor moved at once when freed). **S2 and S3 are the test, and I do not
predict them.** Step 28b had the same two induction terms on the slabs and lost both (force RMSE doubled, T5
66 % and a wrong sign); its record blamed ε∞ stuck at 3.56. If S2 and S3 pass here, that reading was right and one
model does both benchmarks. If they fail as 28b did, the damage is not the stuck factor, and the next step is the
frozen term-by-term test on the slabs, not more training.
**Expected time:** about 18–20 h on the GPU for the two stages, then the scoring on the Mac.

**Step 32 continues on seed 42 only, and step 33 is queued (2026-11-12 14:45 IST; decisions of that day).**
- **The decision for the GPU:** seed 42 only, then the single model: the crashed seed-43 leg of step 32 is dropped
  (branch `step32-s43-dropped-2026-11-12` made first, per the project's never-delete policy; its log, `nanguard_exception.pt` and epoch-14
  `model.pt` are kept on GPU host 2 and copied to the Mac by the watcher; not scored). **Consequence:** step 32 has
  no control seed, so Z3 is unscored and a change of s42's Table II cannot be separated from the machine and the
  run-to-run noise. Z1 and Z2 (the coupling and the prior cell on s42) do not need the control.
- GPU host 2 now: 32a s42 (epoch 20 of 40 at 14:28); `chain_step32.sh` third version:
  32b for s42 alone when 32a finishes; `queue_single.sh` (in the separate copy `<gpu-host-workdir>`,
  synced 14:43 with SPEC 35.1 and today's code; data md5 equal on both machines): step 33's stage A for both seeds
  when step 32 is over and the GPU runs nothing, then stage B. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  for step 33, after the out-of-memory failure.
- The Mac: `watch_step32.sh` second version (seed 42 only; the first kept as
  `watch_step32_v1_both_seeds.sh`); `watch_single.sh` copies and scores step 33 when its queue ends.
- **The single model's phonons first (decision of 2026-11-12: launch it on the Mac and compute the phonons first).**
  (1) Frozen, the starting weights with every term on, against PBE (`score_phonons.py`, a = 4.2557 Å):

  | seed | Z*(Mg) (PBE 1.992) | ε∞ (3.235) | TO (373.3) | LO (681.3) | RMS no NAC / NAC | passed |
  |---|---|---|---|---|---|---|
  | 43 | 1.927 | 3.471 | 443.4 (+18.8 %) | 692.5 | 65.9 / 61.8 | 4/7 (P0, P1, P2, P4) |
  | 42 | 1.930 | 3.471 | 417.7 (+11.9 %) | 677.1 | 51.7 / 44.2 | 4/7 (P0, P1, P2, P4) |

  The electrostatic sector is in place before any training (Z*, ε∞, LO inside their windows; step 31's own physics
  gave Z* 2.4–2.5 and LO above 800); the short-range force constants are not (TO 12–19 % stiff), which is what the
  bulk cells in the training set are for.
  (2) `runs/AuMgO_single_all_mac_s43`: stage A of step 33, seed 43, on the Mac's CPU since 14:38
  (`watch_single_mac_phonons.sh` scores the phonons at every new best checkpoint into
  `runs/watch_single_mac_phonons.log`). **An early look, not the pre-registered run** (that is the two-seed GPU run);
  it shares the Mac with the PBEsol phonon retraining.

### Step 32a, seed 42: finished and scored at its end (2026-11-12 17:30–17:42 IST; recorded 20:00 IST)

Stage 32a (step 26's recipe with `coupling_bound: harrison`, 40 epochs) ended on GPU host 2 at 17:30 IST: 24 503 s, about
10.2 min per epoch, no transient non-finite gradient, best = last (epoch 40, validation loss 0.004926). `chain_step32.sh`
then started 32b (epoch 16 of 20 at 19:58 IST). Scores by `watch_step32.sh` in
`runs/AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bopw_cbh_bondlaw_r5_s42/scores`:

- **Z2 (reading fixed at pre-registration): largest |τ(O–O)| / B = 0.973** at 3.016 Å (τ +2.64 eV, 1872 O–O edges per
  benchmark cell, none above B). Above 0.90: the coupling sits at the ceiling, the prior still drives it, and the bound
  alone is not the fix.
- **Z1 at the end of 32a: FAIL.** The 8-atom ε∞ prior cell's lowest state lies 29.05 eV below its reference
  configuration (limit 5 eV; step 31 s42: 203 eV; step 31 s42 with the bound applied post hoc: 28.44 eV). Largest
  off-diagonal of H there 10.56 eV.
- On the four benchmark cells the lowest state lies 1.65–2.00 eV below the reference; configurations 2, 3, 57, 60.
- Table II after 32a (the T5 protocol): undoped +606.7, doped −406.3 meV (DFT +928.9 / −2.7); test force RMSE 0.0600
  (undoped) / 0.0866 eV/Å (doped), energy RMSE 2.34 / 2.58 meV/atom; ε∞ of the prior cell in training 2.988. Table II
  is scored for the step at the end of 32b (Z4), not here.

Z1 at the end of 32b, Z4 and Z5 are scored when 32b ends. Z3 stays unscored (seed 43 dropped).

### Step 32 scored (seed 42 only; 32b ended 2026-11-12 20:43 IST, scores complete 21:53 IST; recorded 21:57 IST)

32b (step 31's recipe from 32a's weights, `coupling_bound: harrison` throughout): 20 epochs, 10 894 s (9.1 min per
epoch on the GPU), no transient non-finite gradient, best checkpoint epoch 19. Scores in
`runs/AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bopw_cbh_e1kw_bondlaw_r5_s42/scores`.

| | criterion | step 32, s42 | step 31, s42 | verdict |
|---|---|---|---|---|
| **Z1** | no state of the 8-atom ε∞ prior cell more than 5 eV below its reference | 29.05 eV (32a), **29.06 eV (32b)** | 203 eV | **FAIL** (as predicted) |
| **Z2** | largest \|τ(O–O)\| / B on the benchmark cells | **0.973** (32a and 32b; τ +2.64 eV at 3.016 Å, 1872 edges) | 6.5 (unbounded) | at the ceiling: the prior still drives it; the bound alone is not the fix |
| **Z3** | s43 control | not run (seed 43 dropped after its out-of-memory error) | | unscored |
| **Z4** / Y1 | training-set undoped class offset \|flat − upright\| ≤ 30 meV | −103.2 ± 7.0 | −73.3 | FAIL |
| Y2 | wet_u within ±100 meV of +928.9 | +828.1 (−100.8) | +851.0 | FAIL (by 0.8 meV) |
| Y3 | wet_d within ±150 meV of −2.7 | −186.8 (−184.1) | −177.4 | FAIL |
| Y4 | T5 within ±15 % of −931.6 | −1014.9 (108.9 %) | −1028.4 (110.4 %) | pass |
| Y5 | test F RMSE ≤ 1.1 × step 26's s42 (0.0679 / 0.0938) | 0.0600 / 0.0877 | 0.061 / 0.087 | pass |

Z5 (reported): Table II differs from step 31's s42 by −22.9 (wet_u), −9.4 (wet_d), +13.5 meV (T5); test E RMSE 0.635
(undoped) / 0.866 meV/atom (doped); test class offsets (flat − upright) −92.3 ± 5.3 (undoped), −140.6 ± 7.6 meV (doped);
T2: the doped − undoped shift of q(Au₂) −0.9997 e, per-structure Pearson r against Hirshfeld +0.848; relaxed non-wetting
Au–O 2.170 (undoped) / 2.302 Å (doped) against DFT 2.190 / 2.332; the training's own ε∞ of the prior cell 2.978.
NOT DONE: T6 (`probe_eps_bound.py`), the relaxation-failure count.

**Reading, as pre-registered.** Prediction (ii) held: with the bound on through both stages the O–O coupling goes to the
ceiling (0.973 B) and the prior cell still has a state 29 eV below its reference, the same 28–29 eV the bound gave when
applied post hoc to step 31's weights. Bounding the coupling does not remove seed 42's defect; it caps it. Table II on
this seed is no better than step 31's (Y1, Y2, Y3 fail; Y2 by 0.8 meV). With no control seed the run says nothing about
what the bound costs a healthy seed. What it leaves open is what the pre-registration named: a coupling prior of the
anchored size for O–O (SPEC 31's 0.44 eV), or an evaluation of the prior cell whose images are not in phase. Step 33
carries the same bound, so its seed 42 is expected to show the same ceiling.

### Single model, Mac early look (seed 43, stage A, epoch 14 of 40): Table II and the test forces (2026-11-12 22:22–22:38 IST; a mid-training read, not step 33's verdict)

Checkpoint copy `runs/AuMgO_single_all_mac_s43_epoch14` (g 0.9482; phonons against PBE 5/7, see the phonon README);
scores in its `scores/`, per-structure forces in `runs/single_mac_2026-11-12/force_outliers_epoch14_{valid,test}.{txt,npz}`
(`probe_force_outliers.py`, new).

| | single model, epoch 14 of stage A | step 31 s43 (after both stages) | DFT |
|---|---|---|---|
| wet_u / wet_d / T5 (meV) | +542.6 / −20.9 / −563.5 (60 % of DFT) | +920.4 / −75.7 / −996.1 | +928.9 / −2.7 / −931.6 |
| test F RMSE undoped / doped (eV/Å) | 0.1793 / 0.1329 | 0.078 / 0.105 | |
| test E RMSE undoped / doped (meV/atom) | 11.94 / 7.25 (the uniform offset of stage A) | | |
| relaxed non-wetting Au–O undoped / doped (Å) | 2.228 / 2.267 | | 2.190 / 2.332 |
| largest \|τ(O–O)\| / B | 0.793 (doped cells), 0.252 (undoped), 0.051 (prior cell; 1 configuration, no state below the reference) | | |

Stage A of this recipe is not where Table II is fitted (step 32a after its stage A: +606.7 / −406.3; after stage B:
+828.1 / −186.8), so wet_u = +543 is not a verdict on S2.

**The undoped test force RMSE of 0.179 is carried by a few compressed Au₂ dimers (measured).**
- The two routes agree: through the ASE calculator the validation slabs give 0.0765 (undoped) / 0.1211 (doped), the
  numbers of the trainer's own path in `probe_alpha_scale_terms.py`.
- Test, undoped (200): per-structure median 0.0765, 90th percentile 0.0895, largest 2.145; **the worst 10 carry 83 %
  of the squared error; without them the pooled RMSE is 0.0760**. Doped (200): median 0.1152, pooled 0.1329, the worst
  10 carry 26 %, without them 0.1172.
- The worst structures have Au–Au at 2.09–2.18 Å (median 2.57) and a DFT force of +9.5 to +15.3 eV/Å along the bond
  (pushing the atoms apart). The test set reaches 2.089 Å and has 6 structures with a bond force above 9 eV/Å; **the
  training set's shortest Au–Au is 2.185 Å and it has none above 9** (2 structures below 2.25 Å against 11 in the test set).

| test structure | Au–Au (Å) | DFT bond force | single model, epoch 14: force on the Au atom along it | step 31 s43 |
|---|---|---|---|---|
| 167 (undoped) | 2.089 | +15.25 | **−12.1 against +15.75 (wrong sign)**; structure F RMSE 2.145 | +6.66; 0.708 |
| 222 (undoped) | 2.134 | +11.82 | 2.1 against 11.7; 0.757 | 9.69 against 11.77; 0.165 |
| 245 (undoped) | 2.156 | +10.52 | 7.3 against 10.5; 0.253 | 9.66 against 10.49; 0.085 |
| 115 (undoped) | 2.175 | +9.50 | 7.2 against 9.5; 0.189 | 9.13 against 9.47; 0.057 |
| 17 (doped) | 2.142 | +12.02 | 5.8 against 12.0; 0.496 | 7.25 against 12.0; 0.378 |
| 28 (doped) | 2.157 | +11.09 | 7.1 against 11.4; 0.347 | 8.24 against 11.4; 0.254 |

**Reading.** On the bulk of the test set the single model at epoch 14 already has the undoped force error of the
validation set (0.076; step 31's pooled test value is 0.078, of which structure 167 alone is 41 % of the squared
error). Its Au–Au repulsive wall below about 2.2 Å is too soft and turns attractive at 2.09 Å, outside the training
range; step 31's wall is also soft there but stays repulsive. S3 (test F RMSE ≤ 0.10 undoped) therefore hangs on about
five structures. NOT MEASURED: which of the terms switched on (the band's `fermi` form, the clusters space, the on-site
stiffness, the bound, the damping, the anchor) softens the Au–Au wall, and whether it recovers by the end of stage A.

### Pre-registered: which part of the single model softens the Au–Au wall (written 2026-11-12 23:53 IST, before the run; frozen weights, a probe, no training and no code change)

**Benchmark number it serves:** S3 of step 33 (4G test force RMSE ≤ 0.10 eV/Å undoped), which the entry above shows
hangs on about five compressed Au₂ test structures. Step 33's own pre-registration names this as the next step if S3
is at risk ("the frozen term-by-term test on the slabs, not more training").

**What the records say first (reread before the design).**
- The compressed dimers are a known extrapolation: steps 6–10 erred by 10–17.6 eV/Å on undoped test structure 87
  (Au–Au 2.089 Å), read then as extrapolation of the Au–Au wall (training has 1–2 structures below 2.2 Å).
- Step 28b, the first model with SPEC 34 + SPEC 35 on the slabs, doubled the undoped test force RMSE (0.155 / 0.163
  against step 20's 0.073 / 0.069); whether that excess was the same few dimers was never looked at.
- A frozen switch of a term on a backbone trained the other way is not a small perturbation (frozen SPEC 34 / 35 on
  step 20's weights: every arm 4–960 × the force error; correction note of 2026-11-10). A pooled force RMSE under
  one-at-a-time reverts would therefore be unreadable. **The design below does not read a pooled error.**

**Design (`probe_auau_wall_terms.py`, new; checkpoint copy `runs/AuMgO_single_all_mac_s43_epoch14`; comparator
step 31 s43's final model).** The bond force is f = ½ (F_a − F_b)·û, û from Au b to Au a, positive = pushing apart;
moving both Au rigidly along û at a frozen substrate gives f = −dE/dd exactly.
1. **Split.** The model's energy is E = E_net + E_q with E_q the charge module's energy (its own record), and
   E_q = E_b0 + E_Pauli + E_ens: the bound sector's minimum without free charge (frame Coulomb + induction; where
   SPEC 34, 35 act), SPEC 33's frame terms, and the rest (the configuration ensemble's free energy). On test
   structure 167 (Au–Au 2.089 Å) the dimer is scanned over d = 2.00 … 2.57 Å; each part's bond force is its central
   difference in d (h = 0.005 Å), for the single model and for step 31 s43. Check per point: the sum of the parts'
   differences equals the analytic f (pass: within 1e-3 eV/Å + 1e-3 |f|).
2. **Reverts.** The network part does not see the redox settings, so a revert changes f only through E_q:
   f(v) − f(base) = f_q(v) − f_q(base), exactly. Variants on the single model's frozen weights, one at a time to step
   31's value: no damping (SPEC 34); no anchor (SPEC 35); anchor with scope 'all' (SPEC 35 as first built); no coupling
   bound (SPEC 27); 'block' stiffness (SPEC 28.6); 'global' space (SPEC 36); band 'phi1' (SPEC 38); and all six together
   (step 31's equations at these weights). At d = 2.089, 2.20 and 2.57 Å on 167, and at their own geometries on the
   other five (222, 245, 115 undoped; 17, 28 doped). A variant the module refuses is recorded as refused.
3. **Mechanism record (reported):** the induced dipoles on the two Au, their free charges, and each variant's E_b0.

**Readings (fixed now).** Let D = f(single) − f(step 31) at d = 2.089 on 167 (measured before on one atom: about −19 eV/Å).
- **(N) network:** the network part carries ≥ 2/3 of D. The softening is in the trained short-range weights; no
  term is to blame directly and reverts are reported only.
- **(Q) charge module:** E_q carries ≥ 2/3 of D. Then a variant **holds** it if its revert moves f by ≥ ½ |D_q| toward
  step 31 at 2.089 on 167, by the same sign on at least 4 of the 6 structures, and by less than ¼ of that at 2.57 Å
  (so it is a wall effect, not a uniform shift).
- **(M) mixed:** anything else, including a D_q that only the all-six revert removes (non-additive).
- **Limit, stated now:** this attributes the force at these weights. It does not say what a model trained without
  the term would give, and it is one seed at epoch 14 of stage A.

**Prediction (hypothesis, to be falsified).** (Q), held by the anchor. Reason: with SPEC 35.1 a neutral Au keeps the
free-atom polarizability whatever its surroundings, where step 31's heads could lower it; two polarizable points α
on a line have the in-phase stiffness 1/α − 2/d³ (point dipoles; ESTABLISHED: Applequist, Carl and Fung, J. Am. Chem.
Soc. 94, 2952 (1972); the damping that removes the divergence: Thole, Chem. Phys. 59, 341 (1981); both cited from
memory), which vanishes at d = (2α)^(1/3) = 2.2 Å for α ≈ 5.4 Å³, where the wall goes soft. The model's Gaussian dipoles
soften this, so the size is not predicted. **Falsified if** the anchor's revert moves f by less than ¼ |D_q| at 2.089,
or if the reading is (N).

### Scored: the Au–Au wall. The charge module carries the softening, no new setting holds it, and step 31 has the same collapse 0.05 Å further in (2026-11-13 00:06 IST)

Outputs `runs/auau_wall_2026-11-12/auau_wall_terms_epoch14.{txt,json}` (`probe_auau_wall_terms.py`, 154 evaluations,
313 s on 2 threads) and `auau_wall_onset_epoch14.txt` (`probe_auau_wall_onset.py`, a follow-up record, not
pre-registered). Single model: the Mac early look, seed 43, epoch 14 of stage A; comparator step 31 s43, final.

**Deviations from the pre-registration, and checks.**
- The difference step is h = 0.0005 Å, not 0.005. At 0.005 the check failed by truncation alone: the gap between the
  differenced and the analytic bond force at 2.089 Å was +3.1e-2, +4.9e-3, +1.2e-3, +3.1e-4 eV/Å at h = 0.005, 0.002,
  0.001, 0.0005 (single) and −1.5e-1, −2.4e-2, −5.9e-3, −1.5e-3 (step 31): a factor 4 per halving, so the forces are
  the exact derivative of the energy and the curve is very stiff there. With 0.0005 all 18 scan points pass.
- **The three-way split of E_q does not exist in these models:** E_b0 is identically zero (with formal anchors and no
  bound drive, the frame's Coulomb and induction sit inside the configuration energies). E_q = SPEC 33's frame terms
  + the ensemble's free energy; the pre-registered split reduces to that.
- The network part is blind to the redox settings: equal to the base's to 2.3e-10 eV under all eight variants.
- The 'global' space, the band form and the coupling bound change nothing on the undoped cell (0.000 eV/Å): one
  cluster, an empty band, and no coupling at its bound. On the doped cells the band form moves f by +0.20 / −0.37.

**1. The split at test 167 (Au–Au 2.089 Å; DFT f = +15.25 eV/Å).** f(single) = −12.30, f(step 31) = +6.18,
D = −18.47 eV/Å = network **+1.53 (−8 %)** + charge module **−20.00 (+108 %)**. SPEC 33's frame terms give +0.134 in
both; the whole difference is the ensemble's free energy (−14.74 against +5.26). **Reading (Q): the charge module.**
The network's own wall is small in both models (+2.3 and +0.8 eV/Å of DFT's +15.2).

**2. The reverts (frozen; change of f in eV/Å; the hold criterion needed ≥ 10.0 at 2.089 Å).**

| variant | 167 at 2.089 | at 2.20 | at 2.57 | t17 | t28 | t115 | t222 | t245 | share of D_q |
|---|---|---|---|---|---|---|---|---|---|
| no damping (SPEC 34) | −0.13 | +0.03 | −0.01 | +0.30 | −0.01 | −0.13 | −0.01 | +0.17 | −1 % |
| no anchor (SPEC 35) | **+4.85** | −1.10 | +0.07 | −3.65 | −0.11 | −0.09 | −0.02 | −1.19 | **24 %** |
| anchor, scope 'all' | +0.05 | −0.00 | −0.01 | −0.02 | +0.15 | −0.03 | +0.00 | +0.04 | 0 % |
| no coupling bound (27) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 % |
| 'block' stiffness (28.6) | −0.66 | +0.13 | −0.01 | +0.01 | −0.05 | −0.11 | +0.00 | +0.14 | −3 % |
| 'global' space (36) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 % |
| band 'phi1' (38) | 0.00 | 0.00 | 0.00 | +0.20 | −0.37 | 0.00 | 0.00 | 0.00 | 0 % |
| **all six (step 31's equations)** | **+4.86** | −1.16 | +0.11 | +0.21 | −0.85 | +0.05 | −0.01 | −1.66 | **24 %** |

**No variant holds the softening.** The anchor's revert moves f by 24.2 % of D_q at 167 and by the opposite sign on
the other five. **My prediction is falsified by its own line (< ¼ of D_q).** With all six settings back at step 31's
values the single model's weights still give f = −7.44 against step 31's +6.18: **76 % of the difference is in the
trained weights of the charge module**, not in any equation switched on. By the fixed readings this is (M): (Q) by
the split, with no holder.

**3. What the scan shows: an ionic collapse of the compressed dimer, in both models (measured; test 167's geometry,
the two Au moved along their bond).**

| d (Å) | single: f | E − E(2.57) (eV) | p(Au⁰) / p(Au⁺) on atom a | step 31: f | E − E(2.57) | p(Au⁰) / p(Au⁺) on atom a |
|---|---|---|---|---|---|---|
| 2.57 | +0.32 | 0 | 0.77 / 0.14 | −0.01 | 0 | 0.67 / 0.20 |
| 2.30 | +5.13 | +0.60 | 0.78 / 0.13 | +5.18 | +0.56 | 0.66 / 0.20 |
| 2.20 | +8.54 | +1.29 | 0.84 / 0.10 | +8.67 | +1.26 | 0.66 / 0.21 |
| 2.16 | +7.92 | +1.63 | 0.92 / 0.05 | +9.41 | +1.63 | 0.65 / 0.22 |
| 2.14 | +4.00 | +1.76 | 0.995 / 0.004 | +9.46 | +1.82 | 0.64 / 0.24 |
| 2.13 | +2.25 | +1.79 | 0.998 / 0.002 | +9.39 | +1.91 | 0.63 / 0.25 |
| 2.12 | −2.22 | **+1.80 (maximum)** | 0.975 / 0.024 | +9.29 | +2.00 | 0.62 / 0.27 |
| 2.10 | −9.67 | +1.67 | 0.78 / 0.17 | +10.69 | +2.19 | 0.58 / 0.32 |
| 2.089 (the test structure) | **−12.32** | +1.55 | 0.68 / 0.24 | **+6.17** | +2.27 | 0.51 / 0.41 |
| 2.08 | −13.80 | +1.43 | 0.62 / 0.29 | +5.18 | +2.32 | 0.38 / 0.57 |
| 2.07 | −15.06 | +1.29 | 0.56 / 0.32 | −5.56 | **+2.34 (maximum)** | 0.10 / 0.90 |
| 2.06 | −16.13 | +1.13 | 0.52 / 0.35 | −13.79 | +2.24 | 0.03 / 0.97 |
| 2.04 | −17.58 | +0.79 | 0.46 / 0.38 | −18.14 | +1.92 | 0.06 / 0.94 |
| 2.00 | −19.80 | +0.04 | 0.38 / 0.41 | −22.63 | +1.08 | 0.16 / 0.77 |

- **Step 31** falls into one Au⁺Au⁻ configuration (charges +0.97 / −1.00 at 2.06 Å) between 2.10 and 2.06 Å; its
  energy has a maximum at 2.07 Å and its bond force turns to −14 … −23 eV/Å below. The shortest test structure
  (2.089 Å) is 0.02 Å outside it. **This corrects the entry above:** step 31's wall "stays repulsive" only because the
  test set stops at 2.089 Å.
- **The single model** has the same turn 0.05 Å further out (energy maximum at 2.12 Å), with the two ionic
  configurations growing together (charges ±0.21, populations 0.41 / 0.20 at 2.00 Å), and E(2.00) is back at the
  equilibrium dimer's energy (+0.04 eV): past the 1.8 eV barrier the dimer would fuse.
- **DFT has no such turn** on the six test structures: +9.50 (2.175 Å), +10.52 (2.156), +11.09 and +12.02 (2.157 and
  2.142, doped), +11.82 (2.134), +15.25 (2.089). Both models are flat at +8.5 … +9.5 between 2.2 and 2.12 Å, where
  DFT rises by 2–3 eV/Å; that flattening is the collapse's precursor.
- The other compressed structures, DFT / single / step 31: t17 +12.02 / +5.84 / +7.28; t28 +11.09 / +6.82 / +8.04;
  t115 +9.50 / +7.20 / +9.19; t222 +11.82 / +2.14 / +9.80; t245 +10.52 / +7.38 / +9.70.
- Not explained: in the single model the ionic populations first vanish at 2.13–2.14 Å (p(Au⁰) 0.998) and then return.

**Reading.** The single model did not acquire a new defect from the terms switched on. It carries step 31's defect,
an ionic (Au⁺Au⁻) collapse of the compressed dimer that DFT does not have, with its onset moved out from 2.07 to
2.12 Å, three quarters of that by the weights and one quarter by the anchor (at these weights, on this structure).
S3 is therefore decided by where the onset sits against the test set's shortest dimers (2.089–2.16 Å), which training
can move either way. The no-collapse theorem of step 7–8 (carrier width ≥ √2 K_E / (√π U)) covers the Gaussian
self-energy only; the terms added since (penetration, Gordon–Kim overlap, carrier induction, state polarizability,
couplings) are outside its premise. **NOT MEASURED:** which term of the configuration energies lowers Au⁺Au⁻ under
compression (in either model); whether stage B or later epochs move the onset; seed 42. One seed, epoch 14.

### Pre-registered: which term of the configuration energy lowers Au⁺Au⁻ under compression (written 2026-11-13 00:07 IST, before the run; frozen weights, a probe)

**Benchmark number:** S3 (the same five structures), through the collapse measured above.
**Measurement (`probe_auau_ionic_gap.py`, new).** On test 167's scan, the gap G(d) = E(Au_a⁺Au_b⁻) − E(Au⁰Au⁰) between
two configurations of the ensemble (the substrate's units as in the most populated configuration), split exactly into
SPEC 0's terms by `probe_t5_terms.py`'s identity (the terms sum to H's diagonal to 1e-9, asserted per structure), and
the hop between the two. Step 31 s43; and the single model, with the three settings that are exactly inactive on this
cell ('global' space, band 'phi1', no bound: 0.000 eV/Å above) set so that the split's code path applies (checked: the
bond force must repeat −12.297 at 2.089 Å).
**Reading.** A term is the **driver** if it carries ≥ ½ of the fall of G between 2.20 and 2.06 Å; terms that rise
over that range oppose. If no term reaches ½, the two largest are named.
**Prediction (hypothesis).** The induction of the ion pair drives it: the bound response in `screened` (the W of
SPEC 0), SPEC 26 (`ind26`) and SPEC 28 (`alpha28`) together carry ≥ ½ of the fall, and the Gordon–Kim / penetration
term (`overlap`) opposes. Reason: an Au⁻ of large polarizability 2 Å from an Au⁺ gains −½ α E², which grows as
d⁻⁴, faster than a Gaussian-smeared Coulomb attraction (nearly flat at d ≈ σ). **Falsified if** the three induction
terms carry less than ¼ of the fall.

### Scored: the Au–Au wall is built from the learned Au–Au coupling, which crosses zero (2026-11-13 00:13 IST)

Outputs `runs/auau_wall_2026-11-12/auau_ionic_gap_{step31_s43,single_epoch14}.txt` (`probe_auau_ionic_gap.py`, new).
`probe_t5_terms.py` gained two things, nothing else changed in it: it keeps the per-configuration terms, and it names
the elimination's constant (`elim_const`; SPEC 34's damped frame drive is added outside the ensemble, and without the
name the identity E_q = Σ terms failed on the single model by −4.405 eV; with it the 1e-9 assertion holds). On the
single model the three settings that are inactive on this cell were set to the split's code path; the bond force
repeats the native −12.316 eV/Å at 2.089 Å.

**The pre-registered question (the gap G = E(Au_a⁺Au_b⁻) − E(Au⁰Au⁰), 2.20 → 2.06 Å).**

| | fall of G (eV) | overlap (SPEC 23 + 25) | screened | shifts | SPEC 28 | SPEC 26 | the three induction terms |
|---|---|---|---|---|---|---|---|
| step 31 s43 | −1.555 | −0.817 (**53 %**) | −0.462 (30 %) | −0.254 (16 %) | −0.153 (10 %) | +0.130 (−8 %) | 31 % |
| single, epoch 14 | −1.198 | −0.805 (**67 %**) | −0.245 (20 %) | −0.169 (14 %) | +0.006 (−1 %) | +0.016 (−1 %) | 19 % |

**The driver of the gap's fall is the carriers' overlap term, in both models. My prediction is falsified:** the
induction terms carry 31 % and 19 % (the line was ¼ on the single model), and the overlap term does not oppose, it
drives. G reaches zero at 2.10 Å (step 31) and 2.065 Å (single).

**The bond force by term (exact: F = −kT ln Tr e^(−H/kT) gives dF/dd = Tr(ρ dH/dd); the columns sum to the
differenced free energy to 1e-6 … 5e-3 eV/Å, 3e-2 and 1e-1 at step 31's two stiffest points).** This is step 1 of the
23:53 pre-registration at the level of the terms, which the E_b0 route could not give.

| d (Å) | step 31: τ (eV) | ensemble force | of which the hops | the configuration energies | single: τ | ensemble force | the hops | the configuration energies |
|---|---|---|---|---|---|---|---|---|
| 2.57 | −2.41 | | | | −1.48 | | | |
| 2.30 | −2.02 | +3.34 | **+5.06** | −1.72 | −0.99 | +2.60 | **+3.91** | −1.32 |
| 2.20 | −1.43 | +6.98 | **+10.29** | −3.31 | −0.51 | +5.33 | **+6.70** | −1.38 |
| 2.16 | −1.08 | +7.93 | +12.10 | −4.17 | −0.23 | +4.87 | +5.93 | −1.05 |
| 2.12 | −0.69 | +8.10 | +13.35 | −5.25 | **+0.08** | −4.97 | −4.20 | −0.77 |
| 2.10 | −0.47 | +9.67 | +15.72 | −6.05 | +0.26 | −12.23 | −9.56 | −2.67 |
| 2.089 | −0.35 | +5.25 | +12.45 | −7.20 | +0.35 | −14.77 | **−10.96** | −3.81 |
| 2.07 | −0.15 | −6.26 | +7.03 | −13.29 | +0.52 | −17.27 | −11.83 | −5.45 |
| 2.06 | **−0.01** | −14.38 | +0.21 | −14.59 | +0.60 | −18.19 | −12.01 | −6.18 |
| 2.04 | +0.18 | (list changes) | | | +0.78 | −19.32 | −11.76 | −7.56 |
| 2.00 | +0.59 | | | | +1.10 | | | |

τ is the matrix element of H between Au⁰Au⁰ and Au_a⁺Au_b⁻, the learned Au–Au coupling (SPEC 17.6).

**What is measured.**
- **In both models the repulsion the charge module gives between 2.3 and 2.15 Å is the hop term alone.** Every
  configuration-energy term pulls the atoms together (shifts, screened, overlap; SPEC 26 pushes by ≤ 1 eV/Å). The
  repulsion comes from |τ| shrinking under compression: 2.41 → 1.43 eV (step 31) and 1.48 → 0.51 eV (single) between
  2.57 and 2.20 Å, so that compressing the bond loses coupling energy.
- **τ then crosses zero: at 2.06 Å in step 31, at 2.13 Å in the single model.** At the zero the configurations do not
  mix (the single model's p(Au⁰) = 0.998 at 2.13 Å, the unexplained dip of the entry above). Below it |τ| grows again
  with the other sign and the same term pulls: −11 to −12 eV/Å of the single model's −15 to −19 at 2.09–2.04 Å.
- In step 31 the gap closes first (2.10 Å) and the dimer falls into Au⁺Au⁻ where τ ≈ 0; in the single model the gap
  is still open at 2.089 Å (+0.26 eV) and the wrong-signed force there is 74 % the hop's.
- **Why τ may do this (read in the code, `trace_redox/heads.py` `TransferCoupling`):** τ = Σ_k c_k(x_u, x_v) g_k(r)
  times a switch, Gaussians in r with learned coefficients of either sign. Nothing fixes its sign or its slope; the
  coupling bound (SPEC 27) caps |τ| only.

**Reading.** A hopping integral between two s-like orbitals keeps its sign and grows on compression (ESTABLISHED:
Harrison's universal V_ssσ ∝ 1/d², cited as in SPEC 27 / 31), so the term can only pull. The wall of a real Au₂ is the
closed-shell and nuclear repulsion of the two atoms, which is the same in every configuration and belongs to the
frame. The model has no such Au–Au term: SPEC 33's frame terms give +0.13 eV/Å along this bond, `frame_wall` is off,
and the network part gives +0.8 … +3.1 eV/Å of DFT's +9.5 … +15.2. The training therefore shaped τ(d) to stand in
for the wall, which holds only where there are data (Au–Au ≥ 2.185 Å) and reverses where τ runs out. This is one
defect in both models; the single model's τ is smaller (1.48 against 2.41 eV at 2.57 Å), so it runs out 0.07 Å
earlier, inside the test set's range. That is the 76 % "in the weights" of the entry above.

**Hypotheses, not tested.** (i) With a configuration-independent Au–Au repulsion in the frame the coupling would
no longer be asked to make the wall. The mechanism exists: `trace_redox/wall.py` (Round 31, `frame_wall`, off in
both models) states the same gap for the frame pair Au⁰–Mg²⁺ and fills it with the frozen-density repulsion of the
pair; its table would need the Au⁰–Au⁰ pair. The form is the tight-binding one (ESTABLISHED: a band energy from
hopping integrals plus a pair repulsion, as in Harrison's and the DFTB total energies, cited from memory); the
first-principles frozen-density table in place of a fitted repulsion is TRACE's. (ii) A sign-definite, monotonic Au–Au coupling (SPEC 31's anchored form, tried in step 21; its record has to be
reread before it is proposed again) would remove the zero crossing but, alone, also the only repulsion the module has.
**NOT DONE:** the same split on seed 42 and on the doped cells; the dependence on the training epoch; any fix.

### Step 33 on GPU host 2: status read (2026-11-13 06:55 IST; from the two runs' logs there, read remotely at 06:45 IST; not a score)

Both seeds are alive in stage A at **epoch 27 of 40** (started ~20:45 IST on 2026-11-12; about 22 min per epoch for
the pair; GPU at 89 %, 7.3 of 16.3 GB). Training / validation force RMSE at epoch 27: **seed 42 0.0693 / 0.0797**,
**seed 43 0.0749 / 0.0859 eV/Å** (the Mac's seed 43 at its epoch 25: 0.0766 training); validation energy 1.8 and
4.7 meV/atom. Stage B (20 epochs) follows; S1–S6 are scored after it. The local run directories hold only the
configs and warm-start files until `watch_single.sh` copies the results back.

### Single model, Mac early look (seed 43, stage A, epoch 25 of 40): Table II, the test forces and the Au–Au onset again (2026-11-13 07:12 IST; a mid-training read, not step 33's verdict)

Checkpoint copy `runs/AuMgO_single_all_mac_s43_epoch25` (the watcher's copy of the best checkpoint, g 0.9442; phonons
6/7, phonon README); `eval_aumgo.py`, `probe_force_outliers.py` and `probe_auau_wall_onset.py` on it; outputs in its
`scores/` and in `runs/single_mac_2026-11-13/`.

| | epoch 14 | epoch 25 | DFT | S2 / S3 line |
|---|---|---|---|---|
| wet_u (meV) | +542.6 | +564.1 | +928.9 | within ±100 |
| wet_d (meV) | −20.9 | +6.5 | −2.7 | within ±150 |
| T5 = wet_d − wet_u (meV) | −563.5 | −557.6 (60 %) | −931.6 | within ±15 % |
| test F RMSE, undoped (eV/Å) | 0.1793 | 0.1596 | | ≤ 0.10 |
| the same without the worst 10 structures | 0.0760 | **0.0536** | | |
| per-structure median, undoped | 0.0765 | 0.0536 | | |
| test F RMSE, doped | 0.1329 | 0.1213 (0.1114 without the worst 10) | | ≤ 0.13 |
| test E RMSE, undoped / doped (meV/atom) | 11.94 / 7.25 | 4.41 / 1.56 | | |
| relaxed non-wetting Au–O, undoped / doped (Å) | 2.228 / 2.267 | 2.212 / 2.272 | 2.190 / 2.332 | |

- **The bulk of the undoped test set keeps improving** (median 0.0765 → 0.0536 eV/Å) and the pooled number does not
  follow: the worst 10 now carry 89 % of the squared error (83 % at epoch 14), structure 167 alone has F RMSE 2.02
  (2.15) and 222 has 0.63 (0.76); 245 and 115 have come down to 0.13 and 0.10 (0.25, 0.19).
- **The Au–Au onset has not moved in 11 epochs.** Bond force on test 167's scan at 2.13 / 2.12 / 2.089 / 2.06 Å:
  +3.81 / −1.38 / −10.74 / −14.53 eV/Å (epoch 14: +2.25 / −2.22 / −12.32 / −16.13; DFT at 2.089: +15.25); the energy
  maximum stays at 2.12 Å (+1.78 eV; +1.80). The neutral state's population at 2.13–2.14 Å is 0.997–0.999 again: the
  coupling's zero is where it was.
- wet_u has barely moved (+542.6 → +564.1), as stage A's record for step 32 led one to expect; wet_d is at +6.5.
- **Against S3 on this seed and stage:** doped passes its line (0.1213 ≤ 0.13); undoped fails it (0.1596) by two
  structures. That answers the question left open on 2026-11-12 ("whether it recovers by the end of stage A") for
  epochs 14 to 25: it does not.

### General physics only (the project's rule of 2026-11-13): what a universal frame interaction needs, read from the first-principles cache (2026-11-13 12:10 IST; a reading of existing calculations, nothing new computed, nothing built)

**The rule:** every update must be a general equation applicable to all systems, not a fix for one. As worded,
"an Au–Au entry in the wall's table" was not. Survey of the code and records by seven read-only readers:
`runs/general_laws_2026-11-13/map_general_laws.json` (its gap check did not run). What follows tests whether the
existing wall, E_w = K_TF · T_TF^nad, is a law over pairs. Source: the frame-pair records already cached by the Pauli
build in `$TRACE_FP_CACHE/pauli_fp` (frozen-density PBE on the Heitler–London product of the library's monomers, the
high-spin product for two open shells; `fp.sph_nonelst_pbe`, `fp.sph_elst`, `gk_abs`).

| frame pair | R (Å) | non-electrostatic (eV) | electrostatic (eV) | ratio to Thomas–Fermi | ratio to Thomas–Fermi + exchange |
|---|---|---|---|---|---|
| Au⁰–Au⁰ | 2.331 / 2.447 / 2.563 / 2.680 | +8.77 / +6.16 / +4.34 / +3.08 | −6.30 / −4.51 / −3.23 / −2.32 | 0.635 … 0.575 | 0.854 / 0.846 / 0.843 / 0.843 |
| Au⁰–O²⁻ | 2.00 / 2.26 / 2.60 / 3.00 | +10.48 / +4.67 / +1.62 / +0.46 | −3.97 / −2.22 / −0.92 / −0.30 | 0.622 … 0.391 | 0.842 / 0.810 / 0.777 / 0.751 |
| O²⁻–O²⁻ | 2.970 … 3.063 | +0.275 … +0.202 | (sphere convention) | 0.411 … 0.384 | 0.961 … 0.963 |
| Mg²⁺–Au⁰ | 1.80 / 2.20 / 2.66 / 3.08 | +6.62 / +1.71 / +0.38 / +0.10 | −0.12 / −0.01 / 0.00 / 0.00 | 0.922 / 0.897 / 0.843 / 0.772 | 1.200 / 1.249 / 1.279 / 1.280 |
| O²⁻–Mg²⁺ | 2.098 / 2.506 / 2.914 | +1.34 / +0.33 / +0.08 | −0.01 / −0.01 / 0.00 | 0.889 / 0.830 / 0.738 | 1.252 / 1.274 / 1.266 |
| O²⁻–Al³⁺ | 2.099 / 2.507 / 2.916 | +1.25 / +0.31 / +0.07 | +0.17 / +0.03 / +0.01 | 1.020 / 0.976 / 0.907 | 1.384 / 1.412 / 1.419 |

Least-squares K_TF per pair: Au⁰–Au⁰ 0.618, Au⁰–O²⁻ 0.55–0.59, O²⁻–O²⁻ 0.399, O²⁻–Mg²⁺ 0.885, Mg²⁺–Au⁰ 0.918,
O²⁻–Al³⁺ 1.017 (Mg²⁺–Mg²⁺: no overlap at 3 Å).

**Measured, three statements.**
1. **K_TF is not a constant of nature.** It spans 0.40 to 1.02 over six pairs and falls with distance inside every
   pair (the wall's decay length is wrong). The shipped wall, 0.918 × Thomas–Fermi, is one pair's fit; copied to
   Au⁰–Au⁰ it would be 49 % too high.
2. **The non-additive kinetic plus exchange energy of the frozen densities has the right shape:** its ratio to first
   principles is flat in R inside a pair (Au⁰–Au⁰ 0.843–0.854 over 0.35 Å; O²⁻–O²⁻ 0.961–0.963; O²⁻–Mg²⁺ 1.25–1.27),
   and the constant follows the kind of pair: 0.84 for two diffuse frames, 0.96 for two anions, 1.25–1.28 with a
   Mg²⁺ core, 1.38–1.42 with Al³⁺. (ESTABLISHED: the electron-gas pair interaction, Gordon and Kim, J. Chem. Phys.
   56, 3122 (1972); that its exchange and kinetic parts need factors depending on the electron numbers: Waldman and
   Gordon, J. Chem. Phys. 71, 1325 (1979), cited from memory.) The full Gordon–Kim sum changes sign in the tails, as
   SPEC 25 already records.
3. **For a neutral frame pair the electrostatic penetration is as large as the Pauli term.** Au⁰–Au⁰ at 2.331 Å:
   +8.77 − 6.30 = **+2.47 eV** net (+1.64, +1.11, +0.76 at 2.447, 2.563, 2.680; decay length 0.29 Å, about +7 eV/Å
   at 2.39 Å). A wall holding the non-electrostatic part alone, as `frame_wall` does, would overstate the repulsion of
   two neutral atoms 3.6-fold. For the ionic pairs the penetration is ≤ 0.17 eV and the model's own Coulomb holds
   the rest.

**What a general law must therefore be (design constraints; NEW-TO-TRACE as the frame baseline of the ensemble).**
- The frame pair's **whole frozen interaction**, penetration electrostatics beyond the model's own Coulomb plus Pauli,
  in the spin state in which no electron can hop between the two (the only state of two closed shells; the high-spin
  product of two open shells). That is the pair repulsion of the tight-binding total energy, with the bond left to
  the hops (ESTABLISHED: band energy + pair repulsion; the two-site Hubbard model's singlet below its triplet by the
  hopping alone).
- One procedure for every pair in the density library (nine elements, 45 pairs, every frame state of the four 4G
  systems): the library's electrostatic tables, and c_ab × (kinetic + exchange non-additivity) with c_ab = 1 by
  default and a first-principles value where the pair has been computed, the same two-level scheme as SPEC 25's
  `pauli_table`. No constant fitted to a benchmark.
- **Precedence, to count nothing twice:** where a frame pair already has its interaction measured in situ (SPEC 33's
  wall part for an s-class atom against an anion), that one stands and the universal one is off for the pair; the
  carriers' terms (SPEC 23, 25) stay differences against the frame and are untouched.
- **The hop comes with it or not at all.** Step 21's record: Harrison-sized couplings without a short-range
  repulsion overbound short contacts by 3 eV and broke the selected ensemble's convergence; a sign-definite hop alone
  would also remove the module's only Au–Au repulsion. The other systems show the same disease, not an Au₂
  peculiarity: τ(Cl–Cl, 1.99 Å) 24.9 eV against Harrison's 6.2, τ(Na–Cl, 2.36 Å) 8.6 against 2.5, and Ag₂ as an ionic
  pair in the Ag₃ model (records of 2026-10-17/19).
- Code facts a general term meets: like-element pairs need the self-image terms the wall omits today; an untabulated
  pair is silently zero (must be refused); SPEC 31 refuses d-block partners and does not enforce growth on
  compression; covalent units (the carbon chain) hold their bonds in the network, with no hop to pair a wall with.

**NOT DONE / open before anything is built:** (a) whether c_ab separates into one number per species (six pairs
give a first look only: predicted from Au–Au and O–O, Au–O comes out 8–10 % high; from Mg–Au, O–Mg 5 % high);
(b) the frame penetration's convention for charged pairs against the model's Gaussian Coulomb; (c) the Au⁰–Au⁰ points
below 2.33 Å (the test set reaches 2.089 Å; each point cost 2.5–4.8 h); (d) the hop law's test on the cluster space,
which no run has combined with `coupling_anchor`; (e) the pre-registration of the training step.

### Built: SPEC 39, the frame pair's own interaction for every element pair (2026-11-13 12:47 IST; branch `absolute-frame-2026-11-13`; off by default, nothing trained with it)

**Benchmark number it is for:** S3 of step 33 (4G undoped test force RMSE ≤ 0.10 eV/Å), through the Au–Au collapse
measured above; and, being a law over pairs, the recorded coupling excesses of the NaCl and Ag₃ models. It moves no
number until a model is trained with it.

**What was decided on the way, by measurement (first-principles cache, six frame pairs).** Of three candidate forms
for the overlap part, first principles over the form varies inside a pair by 1.00–1.07× for kinetic + exchange
(1.22× for Au⁰–O²⁻ over 2.0–4.0 Å), 1.03–1.25× (2.19×) for the density overlap, and 1.07–1.46× (5.71×) for
Thomas–Fermi alone, the form of the one-pair wall. The term therefore uses kinetic + exchange.

**The term** (`trace_redox/frame_pair.py`, SPEC 39): Φ_ab(R) = s(R) { D_ff(R) + f_a f_b [K/R − J^G_ab(R)] + c_ab P_ab(R) }
over every pair of atoms of the nine library elements (45 pairs; `trace_redox/data/frame_pair.npz`, built in 259 s by
`tools/electrostatics/build_frame_pair.py`, numpy only). Six constants from first principles (Au⁰–Au⁰ 0.8495, O²⁻–O²⁻
0.9615, Au⁰–O²⁻ 0.8264, Mg²⁺–Au⁰ 1.2079, O²⁻–Mg²⁺ 1.2536, O²⁻–Al³⁺ 1.3853), 39 at the electron-gas value 1.

| pair | R (Å) | electrostatics D_ff (eV) | overlap c·P (eV) | Φ (eV) | force (eV/Å) |
|---|---|---|---|---|---|
| Au⁰–Au⁰ | 2.000 | −16.49 | +23.62 | +7.13 | +23.6 |
| | 2.089 | −12.71 | +18.03 | **+5.32** | **+17.4** (DFT's whole bond force there: +15.2; the models: −12.3, +6.2) |
| | 2.200 | −9.21 | +12.91 | +3.71 | +12.0 |
| | 2.331 | −6.31 | +8.73 | +2.43 (first principles +2.47) | +7.8 |
| | 2.570 | −3.17 | +4.29 | +1.12 | +3.6 |
| Ag⁰–Ag⁰ (c = 1, not measured) | 2.53 | −2.77 | +4.16 | +1.39 | +4.2 |
| C⁰–C⁰ (c = 1, not measured) | 1.20 | −9.62 | +18.36 | +8.75 | +29.0 |

**Checks that passed.** The electrostatics against PySCF's interaction of the real monomer densities: Au⁰–Au⁰ −6.3046
(−6.3047) and −3.2383 (−3.2337) eV at 2.331 and 2.563 Å; Mg²⁺–Au⁰ −0.1161 (−0.1161) and −0.0064 (−0.0064) at 1.8 and
2.2 Å; O²⁻–Mg²⁺ −0.0529 (−0.0529) at 2.098 Å. SPEC 23's tabulated difference plus the frame entry equals the exact
E_es − n_a n_b K/R of five state pairs evaluated from the densities alone (4e-3 eV). `tests/redox/test_redox_frame_pair.py`:
21 passed; `tests/redox`: **984 passed** (963 + 21; 625 s).

**Found on the way.** `trace_redox.radial.pair_electrostatic_energy` is wrong by up to 0.05 eV for two heavy neutral
atoms (four terms of order Z_a Z_b/R cancel): Au⁰–Au⁰ at 2.331 Å −6.3577, and its tail is noisy at 0.01 eV. That is
why the Pauli build's own check `elst_vs_library` failed for Au–Au (0.053 eV). The builder uses a cancellation-free
form. No table of the library uses the direct form (SPEC 23's are differences), so no trained model is affected.

**Code rules it follows.** Needs `carrier_penetration` and `carrier_overlap: gordon_kim` (it is their frame entry);
refused with `frame_wall` (legacy, now marked) and with `carriers: covalent_units`; skips the pairs SPEC 33's wall
holds while that is on; an element outside the library is refused; the module is bit-identical without the key.

**NOT DONE.** (a) The hop law (a coupling that keeps its sign and grows on compression): nothing yet stops a learned τ
from crossing zero. (b) First-principles constants for 39 pairs; Ag⁰–Ag⁰ before any use on Ag₃. (c) The Au⁰–Au⁰ points
at 2.0, 2.1, 2.2 Å (running on the Mac since 12:17; the constant rests on 2.33–2.68 Å until then). (d) The rule for
pairs inside a covalent unit. (e) Any training: the term is trained with, never judged frozen, and the step is
pre-registered only after step 33 is scored.

### SPEC 39 against two new first-principles points, Au⁰–Au⁰ at 2.1 and 2.2 Å (2026-11-13 13:06 IST; out of sample: the table was built before they existed)

`tools/electrostatics/frame_pair_wall.py 79 0 79 0 …` in the PySCF environment (about 45 min per point on 3 threads);
logs `runs/general_laws_2026-11-13/auau_frame_points_{A,B}.log`; the 2.0 Å point is still running.

| R (Å) | electrostatics, PySCF | table D_ff | overlap, first principles | table (c = 0.8495) | first principles / P | whole interaction, first principles | table |
|---|---|---|---|---|---|---|---|
| 2.100 (new) | −12.3099 | −12.3100 | +18.059 | +17.442 | **0.8796** | +5.749 | +5.132 |
| 2.200 (new) | −9.2062 | −9.2061 | +13.168 | +12.911 | **0.8664** | +3.961 | +3.705 |
| 2.331 | −6.3047 | −6.3047 | +8.771 | +8.730 | 0.8536 | +2.467 | +2.425 |
| 2.447 | −4.5129 | −4.5130 | +6.157 | +6.180 | 0.8463 | +1.644 | +1.667 |
| 2.563 | −3.2337 | −3.2337 | +4.344 | +4.378 | 0.8429 | +1.110 | +1.144 |
| 2.680 | −2.3203 | −2.3204 | +3.080 | +3.102 | 0.8433 | +0.759 | +0.782 |

- **The electrostatics is confirmed out of sample to 1e-4 eV** at both new distances (and at the four old ones).
- **The overlap constant is not constant below 2.33 Å:** first principles over P rises from 0.843 to 0.880 between
  2.68 and 2.10 Å, about 1 % per 0.1 Å of compression. With one constant fitted at 2.33–2.68 Å the whole interaction is
  6.5 % low at 2.2 Å and 10.7 % low at 2.1 Å, and its force near 2.15 Å about 20 % low (14.3 against 17.9 eV/Å), because
  the term is the difference of two large parts. The statement of SPEC 39.3, "c_ab: 0.8495 (Au⁰–Au⁰)", holds for
  2.33–2.68 Å only.
- **Tested and rejected:** separate constants for the kinetic and the exchange part. They fit Au–Au to 0.9 %, but the
  two functions are nearly collinear and the pair of constants is not determined (Au–Au 0.93 / 1.12 or 0.96 / 1.26,
  Mg–Au 1.25 / 1.24 or 1.06 / 0.59 by the weighting; O–Au no better than −6 … +4 %). A fit, not a law.
- **Decision (to build after the 2.0 Å point and the code review are in):** where a pair has first-principles points,
  the table carries their ratio as a function of distance, c_ab(R), interpolated through the points and held
  constant beyond the first and the last; pairs without points keep 1. The procedure stays one for every pair.

### SPEC 39 reviewed and corrected (2026-11-13 13:36 IST; four independent readers and six verifiers, read-only; `runs/general_laws_2026-11-13/review_spec39.json`)

**Found sound (37 checks; the ones the design rests on).** The cancellation-free electrostatics, rederived term by
term; SPEC 23 + the frame entry = E_es(n_a, n_b) − n_a n_b J^G, by algebra and in the code; **the kernel: the model's
frame charges interact through K erf(R/√(σ_a² + σ_b²))/R with the same widths the term uses, in open cells and in the
Ewald route**; units and constants; the hook is reached once on every route a model with the term can take, the
single model's included (clusters, Fermi band, mesh); the cell enters differentiably; the documented Au–Au numbers
reproduce from the file.

**Confirmed defects, all corrected today** (CHANGELOG Round 41 lists the replaced statements):

| # | defect | size | correction |
|---|---|---|---|
| 1 | With SPEC 33's wall on, the pair (8, 79) was skipped, but that wall is the lone atom's interaction with its whole site (O and five Mg): the four Mg²⁺–Au⁰ pairs were counted twice | +0.34 eV at 2.26 Å, +0.94 eV at 1.8 Å; a spurious 0.8 eV/Å away from the surface | no pair is skipped; the site's pair sum is taken out with the wall's own switch; on the measured site the frame terms now add up to SPEC 33's table (tested at four heights, both levels) |
| 2 | The same skip removed Au⁰–O²⁻ out to 8 Å while the wall ends at 4.3 Å | −0.05 eV per adsorbed atom | as above: beyond 4.3 Å the pair sum is whole |
| 3 | One constant per pair: pinned by the shortest points | Au⁰–O²⁻ whole interaction 13–65 % high at 2.26–2.6 Å; Au⁰–Au⁰ 10.7 % low at 2.1 Å (measured independently, entry above) | c_ab(R) through the first-principles points; every stored point of all six pairs is now reproduced (1e-5) |
| 4 | "High-spin product" stated for all open-shell pairs; the 39 pairs with c = 1 carry the spin-averaged electron gas | H⁰–H⁰ at 0.74 Å: 2.60 against 4.67 eV | stated as a limit; H, C and Ag pairs need their points first |
| 5 | Inherited from SPEC 23's pair list: a pair is lost once two atoms are more than about one cell apart in unwrapped coordinates | benchmark structures are wrapped: no recorded number is affected | **not fixed yet**: strict xfail test added; the fix in `pairs_within` waits until step 33 is scored, because that scoring imports this tree |

Refuted by its verifier: "a resumed run drops the flag silently" (a property of the trainer's resume path for every
setting, not of this term; noted in SPEC 39.6 viii). Also corrected: the series error statement (the electrostatic
integral was rough at 4e-4 eV rms; with cubic Hermite interpolation and the integral split at its kink the worst
table error in reach is **7e-6 eV**, from 1.6e-3); "16 % high" → 21 %; the tests (independent quadrature and
direct-form checks, a periodic cell, a mesh, the cluster space with the Fermi band, the strain derivative, the gate,
tolerances from machine epsilon).

**The term after the corrections** (Au⁰–Au⁰; the first-principles points at 2.1, 2.2 and 2.331 Å are met to 2e-3 eV):

| R (Å) | Φ (eV) | force (eV/Å) |
|---|---|---|
| 2.000 | +7.97 (ratio held at its 2.1 Å value; the 2.0 Å point is running) | +26.2 |
| 2.089 | +5.96 | +19.4 (DFT's whole bond force: +15.2) |
| 2.200 | +3.96 | +14.5 |
| 2.331 | +2.47 | +8.7 |
| 2.570 | +1.09 | +3.5 |

`tests/redox/test_redox_frame_pair.py`: 43 passed, 1 xfailed (strict).

### Single model, Mac early look finished (seed 43, stage A, best checkpoint epoch 39 of 40): Table II, the test forces and the Au–Au onset (2026-11-13 14:04 IST; one seed, stage A only, not step 33's verdict)

The run ended at 13:16 IST (40 epochs, no crash, one transient non-finite gradient at epoch 1; final epoch: training
/ validation force RMSE 0.0672 / 0.0801 eV/Å). Checkpoint copy `runs/AuMgO_single_all_mac_s43_epoch39` (g 0.9371;
phonons against PBE 6/7: Z* 1.943, ε∞ 3.293, TO 367.5 (−1.6 %), LO 662.3, RMS 31.0 without and 6.7 cm⁻¹ with NAC);
outputs in its `scores/` and in `runs/single_mac_2026-11-13/`.

| | epoch 14 | epoch 25 | epoch 39 | DFT | S2 / S3 line |
|---|---|---|---|---|---|
| wet_u (meV) | +542.6 | +564.1 | +569.1 | +928.9 | within ±100 |
| wet_d (meV) | −20.9 | +6.5 | +141.7 | −2.7 | within ±150 |
| T5 = wet_d − wet_u (meV) | −563.5 | −557.6 | −427.4 (46 %) | −931.6 | within ±15 % |
| test F RMSE, undoped (eV/Å) | 0.1793 | 0.1596 | 0.1696 | | ≤ 0.10 |
| the same without the worst 10 structures | 0.0760 | 0.0536 | **0.0491** | | |
| per-structure median, undoped | 0.0765 | 0.0536 | 0.0489 | | |
| test F RMSE, doped | 0.1329 | 0.1213 | 0.1345 (0.1059 without the worst 10) | | ≤ 0.13 |
| test E RMSE, undoped / doped (meV/atom) | 11.94 / 7.25 | 4.41 / 1.56 | 2.10 / 1.74 | | |
| relaxed non-wetting Au–O, undoped / doped (Å) | 2.228 / 2.267 | 2.212 / 2.272 | 2.217 / 2.291 | 2.190 / 2.332 | |
| Au–Au bond force on test 167's scan at 2.16 / 2.14 / 2.13 / 2.089 Å (eV/Å) | +7.9 / +4.0 / +2.3 / −12.3 | +9.5 / +4.4 / +3.8 / −10.7 | +4.7 / −1.0 / −2.7 / −10.7 | +15.25 at 2.089 | |

- **The bulk of the test set kept improving to the end** (undoped median 0.0765 → 0.0489 eV/Å; step 31 s43's pooled
  value is 0.078, of which one structure is 41 %).
- **The compressed dimers got worse, not better.** The bond force now changes sign between 2.14 and 2.16 Å (2.12–2.13 Å
  at epochs 14 and 25); the energy maximum is +1.56 eV at 2.14 Å, and E(2.00 Å) lies 0.05 eV *below* the equilibrium
  dimer. Structure F RMSE at epochs 25 → 39: test 222 0.63 → 0.95, 245 0.13 → 0.49, 115 0.10 → 0.20, doped 17
  0.44 → 0.98, doped 28 0.24 → 0.53; 167 unchanged at 2.02. The worst 10 carry 92 % of the undoped squared error and
  41 % of the doped. Both pooled values are outside S3's lines because of them.
- wet_d moved to +141.7 meV between two stage-A checkpoints (DFT −2.7); wet_u stays near +565. Stage A does not fit
  Table II; nothing is concluded from it.

**Reading.** Stage A of the single model on this seed ends with the best force field the line has had on ordinary
structures and with the Au–Au collapse 0.02 Å further out. That is the measurement SPEC 39 was built against: training
does not cure the collapse, it moves the coupling's zero, in either direction. Step 33's verdict is the GPU run's,
after stage B.

### SPEC 39 closed for today: the 2.0 Å point, the ratio's ends, the final suites (2026-11-13 14:05 IST)

- **Au⁰–Au⁰ at 2.000 Å, out of sample** (computed after the table): electrostatics −16.4904 eV (table −16.4908); overlap
  +24.906 (the table, with its ratio held at the 2.1 Å value, +24.458); whole +8.416 (+7.968, 5.3 % low). The table
  now carries seven Au⁰–Au⁰ points, 2.00–2.68 Å (ratio 0.843–0.896).
- **The ratio's ends:** a zero end slope put a bump of several eV/Å into the Au–Au force between the first two points.
  Replaced by the shape-preserving end slope with a bounded continuation (saturating within 0.3 Å). Φ(Au⁰–Au⁰) and its
  force are now +8.42 eV, +32.4 eV/Å at 2.000 Å; +5.99, +22.6 at 2.089 Å; +3.96, +14.5 at 2.200 Å; +2.47, +8.7 at
  2.331 Å; +1.09, +3.5 at 2.570 Å, monotone in R (tested on 221 points).
- **Suites on the final code:** `tests/redox` 1008 passed, 1 xfailed (the strict xfail of the inherited pair-list
  defect; 531 s); the rest 1045 passed, 4 skipped (run at 13:50, before the ratio's end change, which touches
  `trace_redox/frame_pair.py` and its data file only).
- **NOT DONE:** the fix of `penetration.pairs_within` (after step 33 is scored); the hop law; first-principles points
  for 39 pairs; the rule inside covalent units; the pre-registration of the
  training step (after step 33's verdict); the term-level split of the surface normal force (the oxygen factor).

### Measured: the no-hop state of Au₂ from first principles, against SPEC 39's frozen term (2026-11-13 17:15 IST; a measurement, no code change; the design decision that follows from it is under adversarial review)

**Why.** SPEC 39 as built takes, for two open-shell frame atoms, the frozen high-spin product of the two atoms. The
bond is then (frame term) + (the ensemble's hop). The partner of the ensemble's covalent configuration is the state
in which no electron hops, the pair's σσ* triplet (ESTABLISHED: the two-site Hubbard / valence-bond mapping). Its
energy had not been computed. `tools/electrostatics/pair_no_hop_curve.py` (new): PBE, def2-TZVP with the def2 ECP,
counterpoise-corrected interaction energies, the ground singlet (restricted), the relaxed triplet (unrestricted SCF,
⟨S²⟩ 2.002–2.003) and the TDA σσ* excitations with U = E_S − E_T, t = √(E_T E_S)/2 (the mapping of 2026-11-01, whose
values at 2.30–2.80 Å it reproduces). Log `runs/general_laws_2026-11-13/au2_no_hop_curve.log`.

| R (Å) | ground singlet (eV) | relaxed triplet = no hop (eV) | their difference | TDA triplet / singlet | U (eV) | t (eV) | SPEC 39 frozen Φ (eV) | Φ − triplet |
|---|---|---|---|---|---|---|---|---|
| 2.00 | +1.648 | **+3.581** | 1.933 | 3.266 / 3.926 | 0.660 | 1.791 | +8.416 | 4.84 |
| 2.10 | −0.129 | **+1.845** | 1.974 | 2.936 / 3.698 | 0.762 | 1.648 | +5.749 | 3.90 |
| 2.20 | −1.205 | **+0.800** | 2.005 | 2.651 / 3.487 | 0.837 | 1.520 | +3.961 | 3.16 |
| 2.30 | −1.815 | **+0.207** | 2.022 | 2.403 / 3.303 | 0.900 | 1.409 | +2.755 | 2.55 |
| 2.47 | −2.212 | **−0.191** | 2.021 | 2.049 / 3.044 | 0.995 | 1.249 | +1.520 | 1.71 |
| 2.60 | −2.211 | **−0.212** | 2.000 | | | | +0.985 | 1.20 |
| 2.80 | −1.980 | **−0.395** | 1.585 | | | | +0.511 | 0.91 |
| 3.00 | −1.649 | **−0.343** | 1.306 | | | | +0.256 | 0.60 |

**What is measured.**
1. **The frozen term lies 0.6 to 4.8 eV above the true no-hop state**, more the shorter the bond. Its slope between
   2.0 and 2.1 Å is 26.7 eV/Å; the relaxed triplet's is 17.4, and the ground state's 17.8 (the slab data: +15.25 at
   2.089 Å, +11.8 at 2.134 Å). A model trained with the frozen term would have to learn 1.7–4.8 eV of relaxation from
   the network and would be 5–9 eV/Å too repulsive below its data.
2. **Below 2.6 Å the ground state's wall is the no-hop state's wall.** The singlet–triplet difference is flat,
   1.93–2.02 eV from 2.0 to 2.6 Å. With the relaxed triplet as the frame term, the hop has to supply a constant 2.0 eV
   and the wall no longer depends on the shape of τ(R) at all.
3. **First principles has no ionic collapse and a hop that grows on compression:** U stays positive (0.995 → 0.660 eV
   from 2.47 to 2.00 Å) and t rises monotonically, 1.249 → 1.791 eV, as d^−1.7; it is 0.70–0.74 of Harrison's
   universal value at 2.30–2.80 Å. The models have Δ_ion crossing zero at 2.07–2.10 Å and τ shrinking to zero.
4. The vertical (TDA) triplet excitation exceeds the relaxed difference by 0.03 eV at 2.47 Å and 1.33 eV at 2.00 Å:
   the triplet's own orbital relaxation, which grows under compression. So "ground state + vertical excitation" and
   "relaxed triplet" are two different no-hop curves below 2.4 Å; which one is the ensemble's partner is part of the
   review.
5. PBE's relaxed triplet is bound by 0.2–0.4 eV at 2.5–3.0 Å.

**Not concluded here.** The decision (the frame term of two open shells at the relaxed no-hop level, and the form of
the hop law) is with three skeptics and a referee; nothing is changed in the code until it returns.

### SPEC 39 on two pairs it had never seen: Na⁺–Cl⁻ and Cl⁻–Cl⁻ (2026-11-13 17:15 IST; first-principles points computed after the table; `runs/general_laws_2026-11-13/nacl_frame_points.log`)

The pair tool was opened to all nine library elements (`compare_overlap.py`, `frame_pair_wall.py`: the element
tables; about 1 min per point for these pairs).

| pair | R (Å) | electrostatics beyond point charges: first principles / table (eV) | overlap: first principles / electron gas P (eV) | ratio |
|---|---|---|---|---|
| Na⁺–Cl⁻ | 2.0 | −0.6291 / −0.6291 | +3.640 / +3.289 | 1.107 |
| | 2.2 | −0.3158 / −0.3158 | +1.913 / +1.746 | 1.096 |
| | 2.4 | −0.1610 / −0.1610 | +1.020 / +0.937 | 1.089 |
| | 2.6 | −0.0836 / −0.0836 | +0.551 / +0.509 | 1.082 |
| | 2.9 | −0.0324 / −0.0324 | +0.223 / +0.209 | 1.065 |
| | 3.3 | −0.0096 / −0.0096 | +0.068 / +0.067 | 1.017 |
| Cl⁻–Cl⁻ | 2.6 | −2.3318 / −2.3318 | +2.948 / +2.688 | 1.097 |
| | 3.0 | −0.8909 / −0.8909 | +0.885 / +0.944 | 0.938 |
| | 3.4 | −0.3491 / −0.3491 | +0.245 / +0.320 | 0.766 |
| | 3.8 | −0.1404 / −0.1404 | +0.053 / +0.100 | 0.527 |

- **The electrostatic half holds to 1e-4 eV on both new pairs at all ten distances** (a free anion without a Watson
  sphere, an alkali cation): thirteen out-of-sample points in all today.
- **The parameter-free overlap (c = 1) is 2–10 % low for Na⁺–Cl⁻** over 2.0–3.3 Å, **and wrong in the Cl⁻–Cl⁻ tail**:
  10 % low at 2.6 Å, 31 % high at 3.4 Å, 90 % high at 3.8 Å, where the clusters' second neighbours sit (3.7–3.9 Å).
  The drift is that of Au⁰–O²⁻. These points enter the table at its next build (eight pairs with points).

### Correction to "Measured: the no-hop state of Au₂": the unrestricted triplet is a d-hole state below 2.6 Å, and the design review's outcome (2026-11-13 18:54 IST; nothing built)

The entry above stays as written; this note replaces its reading. Three skeptics reviewed the two decisions I had
drafted from it (`runs/general_laws_2026-11-13/critique_hop_and_level.json`; their referee did not run, so the rulings
below are mine, each checked by a measurement or by arithmetic on the table).

**1. The "relaxed triplet" column is the wrong state at short range (checked: `tools/electrostatics/pair_triplet_character.py`).**

| R (Å) | d population per atom (10 = full) | the two singly occupied orbitals | state |
|---|---|---|---|
| 2.00 | 9.38 | σ* 0.43–0.48, d character 0.32–0.63 | 5d → σ* triplet: a d hole, the 6s σ_g pair intact |
| 2.20 | 9.41 | one σ* (0.90), one d (0.97) | the same |
| 2.47 | 9.45 | one σ* (0.83), one d (0.78) | the same |
| 2.60 | 9.81 | mixed; the SCF is internally unstable | crossing region |
| 3.00 | 9.88 | one σ_g (0.81), one σ* (0.91) | the σσ* triplet, the no-hop state |

Below 2.6 Å the aufbau triplet keeps the 6s bond and removes a 5d electron. **Withdrawn from the entry above:** item 2
("below 2.6 Å the ground state's wall is the no-hop state's wall … the hop has to supply a constant 2.0 eV": that is
what a state that still holds the bond shows), the numbers of item 1 that use the triplet column (17.4 eV/Å; "1.7–4.8
eV of relaxation"), and item 5 as a statement about the no-hop state. Items 3 (U, t) and the frozen column stand.
I had named this candidate artifact nowhere before writing the reading; the skeptics did, as the rule-out-artifacts policy requires.

**2. The no-hop level that belongs to the hop is the vertical one, by identity.** U and t are defined from the vertical
σσ* excitations (U = E_S − E_T, t = √(E_T E_S)/2), so the two-site model's covalent configuration is
E_cov(R) = E_gs(R) + (W − U)/2 = E_gs(R) + E_T^vertical(R) (the triplet root carries 0.76–0.85 of the σσ* weight).

| R (Å) | 2.00 | 2.10 | 2.20 | 2.30 | 2.47 | 2.60 | 2.80 |
|---|---|---|---|---|---|---|---|
| E_cov (eV) | +4.915 | +2.808 | +1.445 | +0.589 | −0.162 | −0.390 | −0.462 |
| ρ = frozen Φ − E_cov (eV) | 3.50 | 2.94 | 2.52 | 2.17 | 1.68 | 1.38 | 0.97 |
| t / Harrison (r0 = 0.5 Å) | 0.714 | 0.720 | 0.725 | 0.732 | 0.744 | 0.753 | 0.765 |
| two-site model with **one constant**, t = 0.735 × Harrison, Δ_ion = U: E − E_gs (eV) | −0.107 | −0.069 | −0.041 | −0.013 | +0.028 | +0.053 | +0.076 |

Its force between successive points is 17.4 / 10.5 / 5.8 / 2.1 / −0.2 / −1.3 eV/Å against first principles' 17.8 /
10.8 / 6.1 / 2.3 / 0.0 / −1.2. **Three first-principles functions (E_cov, U, and a Harrison-shaped hop with one
constant) reproduce the Au₂ ground state to 0.11 eV and its force to 0.4 eV/Å from 2.0 to 2.8 Å.** The ground state's
ionic weight is 0.36–0.46: strong mixing, as the record of 2026-11-01 has it.

**3. Rulings on my two drafted decisions.**
- **D1 (take the relaxed high-spin state as the frame term): rejected.** (a) Wrong state, above. (b) Even with the
  right state, a relaxed no-hop level paired with first-principles U needs a hop that *shrinks* on compression (the
  required |H_cov,ion| would be 1.58 → 1.75 eV from 2.0 to 2.6 Å): the disease the term was built against. (c) A
  configuration-independent term cannot hold a relaxation that belongs to one configuration: Φ is the same for
  Au⁰Au⁰ and Au⁺Au⁻, so the ionic gap, whose fall drives the collapse (67 % from the carriers' overlap term), is
  untouched, and SPEC 23 / 25's differences, fitted against the *frozen* frame, would lose their meaning.
  (d) "High spin blocks the hop" is true of two half-filled s shells only (six of the 45 pairs).
- **D2 (a sign-fixed hop with a learned factor in (1/20, 2), initialised small): rejected as specified.**
  (a) Initialised at 1/20 the Au–Au coupling sits below the amplitude selection's off threshold: zero gradient.
  (b) Monotony is not delivered: the factor is read from invariants that move with the bond length, and it spans
  40× where the base changes 1.65×. (c) One factor cannot serve the neutral pair (√2 t) and the one-electron
  resonance of Au₂⁻ (t) with `spin_pairing` off. (d) Nothing physical keeps the couplings that broke step 21 small.
  What stands: Harrison's form with r0 = 0.5 Å fits first principles for Au–Au with one constant to ±3.5 %; a fixed
  sign removes the zero crossing.

**4. The design that survives (a plan; hypotheses marked; nothing is built).**
- **SPEC 39 stays as built:** frozen, additive, every pair; "SPEC 23 made absolute" is kept.
- **NEW-TO-TRACE, to be built as its own section:** the relaxation of two open-shell frames goes on the diagonal of
  the configuration that has it, one entry of the state-pair slot that is zero today:
  E(n) ∋ −ρ_ab(R) [n_a = f_a][n_b = f_b], ρ_ab = Φ_frozen − E_cov, E_cov from the pair's ground-state curve and
  its vertical σσ* triplet (first principles). The covalent level is then E_cov, and the ionic gap gains +ρ.
  **Hypothesis from the records, not measured:** the model's fixed terms give Δ_ion = −0.37 / +0.45 / +0.94 / +1.52 eV
  at 2.30 / 2.47 / 2.60 / 2.80 Å before a learned shift of +2.5 to +3.8 eV; with ρ added and the bound sector's
  −0.8 … −1.1 eV they come to about U (0.90–1.16) with no learned shift. If that holds, one missing term explains the
  missing wall, the gap's collapse and the size of the learned shifts.
- **The hop:** τ_ab(r) = η(l_a, l_b) ħ²/(m (r² + r0²)) × f_ab × w(r) with f_ab one first-principles constant per
  element pair (Au–Au 0.735) and `spin_pairing` on (SPEC 29), so that one t serves the neutral and the charged
  sector (first principles: equal within 0.07 eV). ESTABLISHED: Harrison's form; the two-site mapping.
- **Open before any of it is built:** the gap at zero learned shift (a frozen probe); the same three functions for
  Ag₂ (the tool extracts U, t for Au only); what the rule is for an open shell on a closed-shell ion (SPEC 33's
  in-situ level) and for p shells; how Harrison-sized couplings behave in the cluster space (step 21's failure was
  in the global space without a frame repulsion); the couplings of cation–anion pairs, which today keep the free form.

### Measured: the ionic gap of gas-phase Au₂ with no learned shift, plus the covalent relaxation ρ (2026-11-13 18:55 IST; frozen probe on the finished Mac model, epoch 39; the hypothesis of the entry above, tested)

`probe_dimer_decompose.py` (its R list extended to 2.0–2.2 Å in a scratch copy; the split closes on the module's own
Hamiltonian). "Fixed terms" = the gap minus the learned environment shift: IE − EA, the on-site and inter-site
screening, −K/R, SPEC 23, SPEC 25, SPEC 26, SPEC 28.

| R (Å) | gap as trained | learned shift | fixed terms | fixed terms + ρ | first-principles U | difference | coupling as trained | √2 t |
|---|---|---|---|---|---|---|---|---|
| 2.00 | −0.559 | +3.227 | −3.786 | −0.285 | 0.660 | **−0.945** | 1.085 | 2.533 |
| 2.10 | +0.754 | +3.634 | −2.880 | +0.061 | 0.762 | −0.701 | 0.358 | 2.331 |
| 2.20 | +1.573 | +3.711 | −2.138 | +0.378 | 0.837 | −0.459 | 0.326 | 2.150 |
| 2.30 | +2.155 | +3.680 | −1.525 | +0.641 | 0.900 | −0.259 | 0.658 | 1.993 |
| 2.47 | +2.749 | +3.446 | −0.697 | +0.985 | 0.995 | **−0.010** | 0.849 | 1.766 |
| 2.60 | +3.019 | +3.212 | −0.193 | +1.182 | 1.063 | +0.119 | 0.943 | 1.621 |
| 2.80 | +3.232 | +2.791 | +0.441 | +1.414 | 1.164 | +0.250 | 1.012 | 1.427 |

- **The hypothesis holds at bonding distances and fails below them.** With no learned shift at all, the model's fixed
  terms plus ρ give the first-principles gap to 0.01 eV at 2.47 Å and within 0.26 eV over 2.30–2.80 Å, where the
  trained model needs a learned shift of +2.8 to +3.7 eV and still sits 1.3–2.1 eV above U. **But the sum falls
  1.55 eV/Å where U falls 0.53:** at 2.0 Å it is −0.29 eV against +0.66, and it still crosses zero near 2.08 Å.
- So ρ explains the *size* of the gap and of the learned shifts where there are data; **it does not cure the gap's
  collapse under compression.** That sits in the ionic configuration's own terms, which fall 3.8 eV/Å (−K/R and the
  carriers' overlap difference; the two-carrier Pauli constant is the electron-gas value and the one-carrier ones
  were fitted at 2.33–2.56 Å only). The earlier line "one missing term explains the wall, the gap's collapse and the
  shifts" is therefore **two-thirds right: not the collapse.**
- The trained coupling is 0.33–1.09 eV against √2 t = 1.43–2.53: the model sits in weak mixing (large gap, small
  hop) where first principles has strong mixing (ionic weight 0.36–0.46).
- **Consequence for the design.** For a pair of s¹ frames the two-site sector is fixed by three first-principles
  functions, E_cov(R), U(R), t(R). The model's terms reproduce U at 2.47 Å unaided and miss it by −0.95 eV at 2.0 Å,
  so the ionic diabat needs its own short-range anchor (U(R) itself, or first-principles two-carrier overlap points
  below 2.33 Å) beside ρ. Nothing is built; one seed, one checkpoint, the gas-phase pair.

### SPEC 39 on a third pair it had never seen: Ag⁰–Ag⁰ (2026-11-13 19:03 IST; six first-principles points of the frozen high-spin product, computed after the table; `runs/general_laws_2026-11-13/agag_frame_points.log`)

| R (Å) | electrostatics: first principles / table (eV) | overlap: first principles / electron gas P | ratio | whole: first principles / table at c = 1 |
|---|---|---|---|---|
| 2.30 | −4.8477 / −4.8478 | +6.846 / +7.627 | 0.898 | +1.998 / +2.779 |
| 2.40 | −3.7936 / −3.7937 | +5.267 / +5.847 | 0.901 | +1.473 / +2.053 |
| 2.53 | −2.7685 / −2.7686 | +3.783 / +4.156 | 0.910 | +1.014 / +1.388 |
| 2.65 | −2.0771 / −2.0771 | +2.811 / +3.043 | 0.924 | +0.734 / +0.966 |
| 2.80 | −1.4565 / −1.4565 | +1.958 / +2.067 | 0.947 | +0.502 / +0.610 |
| 3.00 | −0.9123 / −0.9123 | +1.224 / +1.236 | 0.991 | +0.312 / +0.323 |

- **The electrostatic half holds to 1e-4 eV again** (nineteen out-of-sample points today: Au–Au 3, Na–Cl 6, Cl–Cl 4,
  Ag–Ag 6).
- **The default c = 1 overstates the whole Ag⁰–Ag⁰ interaction by 37–39 % at 2.3–2.53 Å** (a 9–10 % error of the
  overlap part, magnified because the term is a difference), as SPEC 39.6 (iii) warned. With these points the pair
  is at the frozen level like Au⁰–Au⁰; they enter the table at its next build (nine pairs with points).
- For Ag₃ the pair still lacks what the review of today asks of two s¹ frames: the vertical no-hop level and the
  hop (`pair_no_hop_curve.py` extracts U and t for Au only).

### Pre-registered: is the s–s hop one constant per pair, and is that constant universal? Ag₂ and H₂ (2026-11-13 19:56 IST, before the runs; first principles only, no model)

**Benchmark numbers it serves:** the Au–Au collapse (S3 of step 33) through the hop law, and the Ag₃± benchmark, whose
bonds are Ag⁰–Ag⁰. **Why:** the general-physics rule asks for general laws. Au₂ gave t(R) = 0.714–0.765 of Harrison's form
1.40 ħ²/(m (R² + r0²)), r0 = 0.5 Å (one constant to ±3.5 %). One pair is not a law.
**Measurement** (`tools/electrostatics/pair_no_hop_curve.py`, made generic in the valence s orbital): for Ag₂ over
2.2–3.0 Å and H₂ over 0.6–1.4 Å, PBE / def2-TZVP, the ground-state curve, the vertical σσ* triplet and singlet (TDA),
U = E_S − E_T, t = √(E_T E_S)/2, the vertical no-hop level E_cov = E_gs + E_T, and t over Harrison's form.
**Criteria.** (i) *Within a pair:* t over Harrison's form is constant to ±5 % across the range. (ii) *Across pairs:*
the constants of Au₂ (0.735), Ag₂ and H₂ lie within ±10 % of their mean; then one universal constant; otherwise one
first-principles constant per element pair, by the same procedure.
**Prediction (hypothesis).** (i) holds for Ag₂ and fails for H₂ (r0 = 0.5 Å removes 31 % of the base at 0.74 Å, a
regularisation chosen for heavy atoms). (ii) fails: the constant is a property of the pair, with Ag above Au (a more
diffuse 5s). The σσ* weights of the picked roots are reported; a root below 0.5 is flagged, not used.

### Scored: the s–s hop is not Harrison's form with one constant, and the constant is not universal (2026-11-13 19:58 IST; `runs/general_laws_2026-11-13/{ag2,h2}_two_site.log`)

First principles (PBE / def2-TZVP, TDA σσ* roots; triplet weights 0.69–0.98, singlet 0.47–0.74, the one below 0.5
is H₂ at 0.6 Å and is flagged).

| pair | R (Å) | t (eV) | t over Harrison's form (r0 = 0.5 Å) | U (eV) | no-hop level E_cov (eV) | ground state (eV) |
|---|---|---|---|---|---|---|
| Ag₂ | 2.20 / 2.30 / 2.40 / 2.53 / 2.65 / 2.80 / 3.00 | 1.430 / 1.366 / 1.307 / 1.236 / 1.176 / 1.104 / 1.013 | 0.682 / 0.709 / 0.736 / 0.771 / 0.802 / 0.838 / 0.879 | 1.34 / 1.41 / 1.48 / 1.58 / 1.68 / 1.82 / 2.03 | +1.374 / +0.760 / +0.369 / +0.072 / −0.070 / −0.147 / −0.164 | −0.891 / −1.356 / −1.609 / −1.734 / −1.727 / −1.625 / −1.417 |
| H₂ | 0.60 / 0.74 / 0.90 / 1.10 / 1.40 | 6.458 / 5.738 / 4.952 / 4.048 / 2.894 | 0.369 / 0.429 / 0.492 / 0.554 / 0.599 | 1.97 / 2.68 / 3.60 / 4.71 / 6.09 | +7.996 / +5.681 / +4.018 / +2.653 / +1.451 | −3.970 / −4.532 / −4.251 / −3.423 / −2.043 |
| Au₂ (above) | 2.00 … 2.80 | 1.791 … 1.009 | 0.714 … 0.765 | 0.66 … 1.16 | +4.915 … −0.462 | |

- **(i) Within a pair: FAIL for Ag₂ (my prediction was that it holds) and for H₂.** t over Harrison's form runs
  0.682–0.879 for Ag₂ (±12.6 %) and 0.369–0.599 for H₂ (±24 %); only Au₂ is constant (±3.5 %). t falls as d^−1.7
  (Au₂), d^−1.1 (Ag₂), d^−0.95 (H₂): Harrison's d^−2 fits one of the three.
- **(ii) Across pairs: FAIL, as predicted.** At each pair's own bond length the constant is 0.744 (Au₂), 0.771
  (Ag₂), 0.429 (H₂).
- **What does hold for all three:** t has one sign and falls monotonically with distance; ln t is linear in R to
  0.4 % (Ag₂), 2.4 % (Au₂), 3.6 % (H₂) at an interior point, with decay constants 0.43, 0.72 and 1.00 Å⁻¹. And U stays
  positive everywhere (no ionic collapse in first principles for any of the three).
- **Consequence for the hop law.** No universal constant and no universal radial form. The general statement is
  the procedure: for a pair of s¹ frames the hop is the first-principles t_ab(R) = √(E_T E_S)/2 of the pair,
  tabulated and interpolated monotonically (as the frame term's ratio is), with its sign fixed; the ensemble's
  spin pairing on, so that one t serves every charge state. Harrison's form with a learned factor (SPEC 31) is
  retired as the candidate. The same calculation gives the pair's no-hop level and U.
- For Ag₂, ρ = frozen Φ − E_cov is 1.24 / 1.10 / 0.94 / 0.80 / 0.65 / 0.48 eV at 2.30 / 2.40 / 2.53 / 2.65 / 2.80 /
  3.00 Å (Au₂: 2.17 at 2.30 Å, 1.68 at 2.47 Å).

### Step 33 scored: the single model with every term on FAILS as pre-registered (S2, S3 and S5 on both seeds; 2026-11-13 20:01 IST)

Trained on GPU host 2 (stage A 40 epochs from 20:45 IST on 2026-11-12, stage B 20 epochs, ended 18:26 IST on
2026-11-13; about 22 and 21 min per epoch for the two seeds together; 7.2–7.4 of 16.3 GB of GPU memory in the samples
taken); copied and scored on the Mac by `watch_single.sh` (ended 20:00). Best checkpoints of stage B: epoch 15 (s42),
epoch 20 (s43). Scores in `runs/AuMgO_single_all_e1kw_bondlaw_r5_s{42,43}/scores`.

| id | criterion | seed 42 | seed 43 | verdict |
|---|---|---|---|---|
| **S1** | both stages finish: no refusal, no NaN stop, no memory failure | finished; 0 events in both logs | finished; 0 events | **pass** |
| **S2** | wet_u within ±100 of +928.9; wet_d within ±150 of −2.7; T5 within ±15 % of −931.6 | +768.4 (−160.5) FAIL; −9.2 pass; −777.6 (83.5 %) FAIL by 1.5 points | +789.9 (−139.0) FAIL; +92.7 pass; −697.2 (74.8 %) FAIL | **FAIL** |
| **S3** | test F RMSE ≤ 0.10 (undoped), ≤ 0.13 (doped) eV/Å | 0.1930 FAIL / 0.1043 pass | 0.2211 FAIL / 0.1563 FAIL | **FAIL** |
| **S4** | P0, P1, P2, P4 pass and at least 6 of 7 | 6/7: Z* 1.948, ε∞ 3.291, TO 363.4 (−2.7 %), LO 661.2, RMS 32.1 / 8.4 with NAC | 6/7: 1.942, 3.308, 371.5 (−0.5 %), 663.2, 30.4 / 5.6 | **pass** |
| **S5** | g within 0.85–0.95 and ε∞ within 0.05 of 3.235 | g 0.9364; ε∞ 3.291 (0.056) FAIL by 0.006 | g 0.9423; ε∞ 3.308 (0.073) FAIL | **FAIL** |

**S6, reported.**
- Class offsets on the test set (flat − upright, meV): undoped −142.1 ± 14.1 (s42), −115.1 ± 18.6 (s43); doped
  +110.1 ± 7.9, +132.9 ± 9.2. Step 31: undoped −73.3 (s42), +7.4 (s43).
- T2: the doped − undoped shift of q(Au₂) −0.985 (s42), −0.978 (s43) against Hirshfeld −0.266; per-structure r +0.853.
- Test E RMSE (undoped / doped): 1.339 / 0.877 (s42), 1.396 / 0.880 meV/atom (s43).
- Largest |τ|/B: **0.973 (O–O) on seed 42**, with O–Mg 0.947 and Mg–Mg 0.966, the ceiling predicted from step 32;
  0.329 (Au–Au) on seed 43 (O–O 0.239). The 8-atom prior cell holds one configuration on both seeds: no state below
  its reference (step 32's seed 42 had one 29 eV below).
- Relaxed non-wetting Au–O (undoped / doped): 2.187 / 2.290 (s42), 2.182 / 2.367 Å (s43); DFT 2.190 / 2.332.
- Stage A alone, phonons: 4/7 on seed 42 (TO −3.3 %, LO 659.8: P3 and P4 fail), 6/7 on seed 43.
- NOT DONE: Table II on the 2×2 supercell; the per-structure force breakdown of these two models (running).

**Reading, by the pre-registration.** Its predictions: S1, S4, S5 pass; S2 and S3 "are the test, and I do not predict
them". S1 and S4 passed; **the prediction for S5 was wrong** (ε∞ ends 0.056 and 0.073 high; the factor is held by the
slab's surface force offset, measured on 2026-11-12 / 13). S2 and S3 failed on both seeds, as step 28b did with two
of these terms. By the pre-registered rule the damage is therefore not the stuck factor, and "the next step is the
frozen term-by-term test on the slabs, not more training". What today's measurements already say about each failure:
- **S3:** on the Mac copy of the same model (seed 43, stage A) the undoped test error without the ten worst
  structures was 0.049 eV/Å, the best of any model, and the ten compressed Au₂ dimers carried 92 % of the rest. The
  cause is the collapse of the Au–Au bond below 2.15 Å (the coupling's zero; no frame repulsion), in every model.
  Whether these two models show the same split is the check now running.
- **S2:** wet_u is 139–161 meV low on both seeds and the undoped class offset is −115 to −142 meV (the flat dimer too
  stable against the upright one): one error seen twice. Step 31 had it at +7 and −73 meV. **Not attributed.**
  wet_d is within its window on both seeds (−9.2 on seed 42; corrected 2026-11-15: this line said "the closest of
  any model to DFT's −2.7", which is false, step 9 s42 gave −6.3).
- **S5:** the crystal factor's pull is a static force offset on the surface ions (97.6 %), not a polarizability
  effect; which term carries it is not measured.
- **What the run does show:** the phonon physics survives the full model on both seeds (6/7; band error with NAC
  5.6–8.4 cm⁻¹, where the step 31 family has Z* 2.4–2.5 and LO above 800), with test energy errors of 0.9–1.4
  meV/atom. One model does phonons; it does not yet do Table II and the test forces.

### Built and pre-registered: SPEC 40, the two-site sector of two s¹ frames from first principles (2026-11-13 20:13 IST; branch `absolute-frame-2026-11-13`; off by default, nothing trained with it)

**Benchmark numbers it is built to move (per the project's benchmark-first policy).** S3 of step 33 (test forces: the ten compressed Au₂ dimers [corrected 2026-11-14: the five undoped test dimers below 2.18 Å carry 92.2 % and 94.2 % on the official seeds] carry
92 % of the undoped error on the Mac copy; DFT's bond force on test 167 at 2.089 Å is +15.25 eV/Å, every model has
−12 to −23) and, through the same Au–Au energy, the relaxed dimers of Table II. The Ag₃± set carries the same pair
class (Ag⁰–Ag⁰). Nothing is claimed for either before a model is trained with the terms.

**The equations (SPEC 40; `trace_redox/two_site.py`, option `two_site`, default off).** For two half-filled s frames
the ensemble is a two-site model. ESTABLISHED (the two-electron, two-orbital mapping, as for H₂): with the pair's
ground-state interaction E_gs and its vertical σσ* excitations E_T, E_S,

    E_cov = E_gs + E_T,   U = E_S − E_T,   t = √(E_T E_S) / 2,   E_gs = E_cov + (U − √(U² + 16 t²)) / 2.

NEW-TO-TRACE, two terms, each the pair's own first-principles function (PBE / def2-TZVP, TDA; no fitted constant, no
form assumed; the points of `trace_redox/data/two_site.json`):

    (40.1) covalent relaxation   O_ab(f_a, f_b; R) = −s(R) ρ_ab(R),   ρ_ab = Φ_ab (SPEC 39) − E_cov,ab > 0
           on the frame-frame entry of the state-pair slot only; ln ρ through the points, exponential beyond, off at 6 Å;
    (40.2) hop                   τ_ab(R) = −t_ab(R) w(R)   in place of the learned coupling of that pair; ln t through
           the points, exponential beyond; with SPEC 29 the neutral pair's element is √2 t and one carrier's is t.

Why not one constant times Harrison's form: scored above (t / Harrison runs 0.71–0.77 for Au₂, 0.68–0.88 for Ag₂,
0.37–0.60 for H₂). What a first-principles table guarantees that the learned head did not: one sign and a monotone
fall with distance, so the hop cannot make or unmake a wall.

**Coverage today.** Au–Au (7 points with t, 2.0–2.8 Å; 8 with E_cov), Ag–Ag (7, 2.2–3.0 Å), H–H (5, 0.6–1.4 Å). The
frozen term of H⁰–H⁰ needed its own first-principles points (electron-gas default 2.62 eV at 0.74 Å against the
high-spin frozen product's 10.03 eV, a factor 3.9; nine points at 0.6–3.0 Å computed today,
`runs/general_laws_2026-11-13/hh_frame_points.log`); with them ρ(H–H) = 7.27 / 4.35 / 2.35 / 1.03 / 0.20 eV at
0.6 / 0.74 / 0.9 / 1.1 / 1.4 Å. Mixed pairs (Ag–Au, H–Au, H–Ag) have no points: they keep the learned coupling and no
relaxation. The mapping above is the symmetric one; for two different atoms the singlet block has two ionic levels
and t² = −λ₀λ₁λ₂ / (2(λ₀ + λ₁ + λ₂)) from its three eigenvalues, the third of which is a double excitation that
TDA does not give: not built.

**Tests (written first, 10 of 17 failed before the module was wired; `tests/redox/test_redox_two_site.py`).** The
points obey the identities to the file's six decimals; t goes through its points, is positive and falls at every
distance from 0.6 R₁ to R_last + 1.5 Å, C¹ at the ends; Φ − ρ = E_cov at every point; in the module, gas-phase Au₂:
the Au⁰Au⁰ entry of the Hamiltonian falls by exactly ρ(R) and the Au⁺Au⁻, Au⁻Au⁺ entries do not move; the
covalent–ionic element is √2 t(R) and the Au⁻Au⁰ ↔ Au⁰Au⁻ element of Au₂⁻ is t(R); off by default and the table is
not touched when off; refused without `frame_pair` or `spin_pairing`; forces are the gradient.

**Pre-registered mechanism check, before running it (frozen weights; a probe, not a verdict on accuracy: the
backbone of a model trained without SPEC 39 / 40 already holds the Au–Au repulsion at the data's distances, so only
the charge module's own bond force is read).** On test 167's rigid Au–Au scan, the final step 33 model of seed 43,
the bond force of E_q alone (central differences of E_q, step 0.0005 Å), as trained and with `frame_pair`,
`spin_pairing` and `two_site` on:
- **C1 (the wall exists):** with the terms on, the charge module's bond force is repulsive at every scanned distance
  from 2.00 to 2.30 Å and rises monotonically under compression. As trained it is attractive below about 2.15 Å.
- **C2 (its size):** at 2.089 Å it lies within ±30 % of DFT's total bond force, +15.25 eV/Å (10.7 to 19.8). The
  gas-phase first-principles value from the same table is +15.1 eV/Å, so the adsorbed bond's force is expected to
  be the pair's own to that accuracy. My estimate of the error from the unanchored ionic gap (−0.46 / −0.70 / −0.95
  eV at 2.2 / 2.1 / 2.0 Å, measured 18:55): about −1.2 eV/Å.
- **C3 (no new wrong sign):** the element between the neutral and the Au⁺Au⁻ configuration keeps its sign over the
  whole scan, 2.00–2.57 Å.
- Prediction: C1 and C3 pass (they follow from the construction if the wiring is right); C2 I expect to pass, but
  the adsorbed dimer's populations differ from the gas phase and I have not measured by how much.
- If C1 fails: the wiring or the ensemble's selection is wrong and nothing is proposed for training until it is
  found. If C2 fails low: the ionic gap's short-range error is larger on the surface than in the gas phase, and the
  gap's anchor comes before any training proposal.

### Scored: SPEC 40's mechanism check passes C1, C2 and C3 (2026-11-13 20:22 IST; frozen weights, the final step 33 model of seed 43; `runs/two_site_2026-11-13/probe_auau_two_site_s43.txt`, `probe_auau_two_site.py`)

Test 167's rigid Au–Au scan; the bond force of E_q alone by central differences (step 0.0005 Å; the model's total
force by autograd and by differences agree to 0.001 eV/Å in the last column's variant). DFT's bond force at the
structure's own distance, 2.089 Å: **+15.25 eV/Å**. Gas-phase first principles (−dE_gs/dR, monotone cubic through
the table's points): +21.3 / +18.0 / +14.5 / +10.8 / +7.8 / +6.3 / +3.5 / +2.3 / −0.0 at the nine distances.

| d (Å) | as trained | + SPEC 39 | + SPEC 39, 29 | **+ SPEC 39, 29, 40** | backbone (f − f_q, the same in all) |
|---|---|---|---|---|---|
| 2.000 | −20.12 | +12.26 | +10.27 | **+18.07** | +0.10 |
| 2.050 | −19.80 | +6.66 | +2.63 | **+14.61** | +1.14 |
| 2.089 | −18.84 | +3.75 | −1.22 | **+12.51** | +1.79 |
| 2.150 | −9.29 | +8.45 | +3.38 | **+9.91** | +2.86 |
| 2.200 | +5.71 | +20.19 | +24.19 | **+7.89** | +3.20 |
| 2.250 | +3.79 | +15.77 | +18.10 | **+6.28** | +2.98 |
| 2.300 | +2.30 | +12.14 | +13.58 | **+4.73** | +2.64 |
| 2.400 | +0.77 | +7.51 | +8.11 | **+2.34** | +1.66 |
| 2.570 | −0.07 | +3.47 | +3.64 | **−0.09** | +0.09 |

- **C1 PASS.** With the three terms on, the charge module's bond force is repulsive at every distance from 2.00 to
  2.40 Å and rises monotonically under compression. As trained it turns attractive below 2.2 Å (−18.8 at 2.089).
- **C2 PASS.** +12.51 eV/Å at 2.089 Å, 18 % below DFT's +15.25 (window 10.7 to 19.8), 2.0 below the gas-phase
  +14.5 (the pre-registration's "+15.1" was my straight-line estimate between two points; the monotone cubic
  through all of them gives +14.5); my estimate of the shortfall from the unanchored ionic gap was 1.2. E_q − E_q(2.57 Å) is +3.61 / +2.27 eV
  at 2.000 / 2.089 Å; the gas-phase first-principles curve gives +3.86 / +2.26. [corrected 2026-11-14, see "SPEC 40 reviewed": +2.24, so the agreement at 2.089 Å is 0.03 eV]
- **C3 PASS.** The element between the neutral and the Au⁺Au⁻ configuration is +2.53 → +1.65 eV over 2.00 → 2.57 Å
  (√2 t(R) at each point: 2.531, 1.652[corrected 2026-11-14, see "SPEC 40 reviewed": 2.532; these two rows are readings of the 'global' / 'phi1' route]), one sign. As trained it runs +1.31 → −1.08 and passes through zero near
  2.16 Å.
- **SPEC 39 alone does not do it** (columns 2 and 3): the frame repulsion on top of the learned hop gives a force
  that falls from +20 at 2.2 Å to +4 (or −1 with spin pairing) at 2.089 Å and rises again: the learned coupling's
  zero is still there. The two terms are one change and are proposed for training together or not at all.
- **Read with the caveat stated beforehand:** frozen weights. The backbone column is what a model trained without
  these terms supplies; the sum (+14.30 at 2.089 Å) is within 6 % of DFT here, but at 2.3–2.4 Å the sum is 2.4–1.6
  above the as-trained total, the double count a retraining removes. With the terms on, the gap between the two
  configurations is +2.4 to +3.0 eV on the surface [corrected 2026-11-14, see "SPEC 40 reviewed": that is the range from 2.089 Å; over the whole scan +1.5 to +3.0, as trained −2.0 to +1.2] (as trained −0.6 to +1.2; first-principles U in the gas phase
  0.7 to 1.1): the trained level shifts had absorbed the missing relaxation, and with ρ added it is counted twice.
  Nothing here says what a trained model's test force is.

### Step 33, the per-structure force breakdown that was still running (2026-11-13 20:22 IST; `runs/step33_2026-11-13/force_outliers_s{42,43}.txt`)

| | undoped F RMSE | without the worst 10 | their share of the squared error | doped F RMSE | without the worst 10 | share |
|---|---|---|---|---|---|---|
| seed 42 | 0.1930 | **0.0527** | 93 % | 0.1043 | 0.0931 | 24 % |
| seed 43 | 0.2211 | **0.0531** | 95 % | 0.1563 | 0.1010 | 60 % |

The worst undoped structures are 167, 222, 207 (and 245 on seed 43), the compressed dimers, with the largest atomic
error on a gold atom every time (30.7 and 32.5 eV/Å on structure 167). The split measured on the Mac copy (0.049
without the ten) holds on both official models: S3's undoped failure is the Au–Au collapse and nothing else that
shows at this level. Doped (criterion 0.13): seed 42 passes (0.1043; its ten worst carry 24 %); seed 43 fails
(0.1563), its three worst are again compressed dimers (17, 28, 178: up to 18.2 eV/Å on a gold atom), the ten worst
carry 60 %, and without them it is 0.1010. So all three failed S3 entries of step 33 (undoped on both seeds, doped on seed 43) are the collapse. The doped median per
structure (0.090 and 0.098 eV/Å) is nearly twice the undoped one (0.051 and 0.052): a spread-out doped error that
is inside the criterion and **not attributed**.

### Measured: the sign structure of the configuration couplings against exact small clusters, and a pre-registered sensitivity check on the slabs (2026-11-13 20:25 IST; exact mathematics, no model; `tools/electrostatics/hubbard_cluster_signs.py`, `hubbard_cluster_signs_package.py`, `tests/redox/test_redox_cluster_signs.py`)

**Why it was looked at.** SPEC 40 gives every s–s hop one sign, −t. Before proposing it for the Ag₃± set I checked
what one sign means on a ring, against exact diagonalisation of the s-orbital Hubbard model (H = −t Σ c†c + U Σ n↑n↓;
dimer, chain of three, triangle, square, tetrahedron; every filling; U/t = 1 and 4; the level compared is the lowest
one of the lowest total spin, which is what one state per occupation pattern can stand for).

**What the package's Hamiltonian does (`diabatic_hamiltonian`: Jordan–Wigner signs of spin-less local orbitals,
SPEC 17.2, with SPEC 29's √2).** ESTABLISHED mathematics, measured here for the first time on this code:
- **Exact for one carrier of either kind on every cluster**, the one hole on an odd ring included (triangle: the
  electron gains 2t, the hole t; tetrahedron 3t and t): the particle–hole sign of a hop is already supplied by the
  string over the filled local orbitals. Exact on two sites at every filling. Independent of the units' order.
- **Not exact for two carriers on three or more sites.** Two electrons (lowest level, in t, U = t / U = 4t):

| cluster | exact singlet | package | error |
|---|---|---|---|
| chain of three | −2.508 / −2.000 | −2.000 / −1.604 | +0.51 / +0.40 |
| triangle | −3.702 / −3.123 | −2.646 / −2.000 | +1.06 / +1.12 |
| square | −3.785 / −3.419 | −2.626 / −2.340 | +1.16 / +1.08 |
| tetrahedron | −5.772 / −5.292 | −3.702 / −3.123 | +2.07 / +2.17 |

  Two holes on the chain of three: the same +0.51 / +0.40. The cause is one line of physics: two spin-less fermions
  exchange with a minus sign, the two electrons of a singlet with a plus, and a loop that exchanges two carriers
  exists as soon as there are three sites (on the open chain it runs through the doubly occupied middle site, so no
  choice of the couplings' signs removes it). The ground state of two electrons is a node-less singlet on any
  cluster (a theorem), so this is an error of the model's sign structure, not of the reference.
- A "singlet rule" (no string; every element −t·f, and +t·f where the donor holds an even and the acceptor an odd
  number of electrons, the move of a hole) is exact for one and two electrons and one and two holes on all five
  clusters, and worse than the package near half filling (square, four electrons: −1.11 / −0.71 against the
  package's −0.50 / −0.48; tetrahedron, four: −2.16 / −1.65 against −1.08 / −1.19). **Neither rule is exact for
  three or more carriers**: one state per occupation pattern cannot hold the spin multiplicity of three unpaired
  electrons. No rule is proposed here.

**What it could touch in the benchmarks (hypotheses, not checked).** Ag₃⁺ is two electrons on three sites and Ag₃⁻
two holes on three sites. A flat Au₂ on the oxide closes rings Au–Au–O that the upright one does not, and step 33's
undoped class offset (flat − upright) is −115 to −142 meV, unattributed.

**Pre-registered sensitivity check, before running it (frozen weights; it measures how much the convention matters
on the slab, not which is right).** The final step 33 model of seed 43, undoped structures only (the 'global' space
with the 'phi1' band is exactly the trained model there when no coupling is above the bound's knee; the per-
structure difference from the trained 'clusters' route is printed). The two undoped Table II references and the
first 15 flat and 15 upright undoped test structures, each evaluated with the package's signs and with the singlet
rule: the change of wet_u and of the class offset.
- If both change by less than 20 meV: the sign convention is not what makes wet_u 140–160 meV low; recorded and left.
- If either changes by 50 meV or more: the convention matters at the size of S2's error, and the two-carrier sign
  becomes a design question for the next step (which rule, and how it is trained).
- Between: recorded as inconclusive.
- Prediction: I do not know. The two rules differ only from fourth order in coupling over gap (two carriers at
  once), which argues for small; the flat dimer's rings argue for not negligible.

### Scored: the sign convention changes nothing on the slabs (0.00 meV on 32 structures); and the hydrogen pair's frozen term (2026-11-13 20:28 IST; `runs/two_site_2026-11-13/probe_sign_rule_s43.txt`, `probe_sign_rule.py`)

**The sensitivity check, as pre-registered.** The final step 33 model of seed 43, undoped, 'global' space:

| | package signs | singlet rule | change |
|---|---|---|---|
| wet_u on the two unrelaxed references (meV) | +769.2 | +769.2 | **+0.0** |
| flat, 15 test structures (mean energy change, meV) | | | **+0.00** (range 0.00 to 0.00) |
| upright, 15 test structures | | | **+0.00** (range 0.00 to 0.00) |
| class offset (flat − upright) | | | **+0.00** |

- 24 900 of the 48 148 couplings examined carry the opposite sign under the singlet rule, and no energy moves: on
  these ensembles (39 to 59 configurations per structure on the references) the two rules differ by a change of
  phase of the configurations, so the selected configurations close no loop on which two carriers exchange.
  **By the pre-registered rule the sign convention is not what makes wet_u 140–160 meV low; recorded and left.**
  The hypothesis about the flat dimer's rings is rejected for this model. [corrected 2026-11-14, see "SPEC 40 reviewed": WITHDRAWN. The selected configuration sets are trees (the reference and single moves, no closed loop), whose spectrum does not depend on any sign, so this check could not have failed and gives no evidence; and most of the 24 900 counted couplings are zero elements]
- It stays open for Ag₃± (two carriers on three sites by construction) and is to be measured there before the hop
  table is trained on that set.
- **A statement of mine that does not generalise, found by this probe:** "the 'global' space with the 'phi1' band is
  exactly the trained model on an undoped cell" was measured on test 167's scan of the Mac model only. On the final
  seed-43 model the 'global' route differs from the trained 'clusters' + 'fermi' route by −41.2 and −35.0 meV on the
  two references and by up to 67 meV on the test structures (wet_u +769.2 against +775.4 on the trained route).
  The sign comparison above is between two runs of the same route, so it stands; the earlier readings of the Au–Au
  scan that used the 'global' route on this model (the element and gap rows of the mechanism check) are of that
  route, and the bond forces in that table are of the trained route. Which of the three settings makes the 35–67
  meV: measured in the entry after this one (the band form, and with 'phi1' the space; not the bound).
- wet_u here is on the unrelaxed mean references of `eval_aumgo.references()`; the scored +789.9 is the scorer's
  own procedure and the 14.5 meV between them is not looked into.

**The hydrogen pair in SPEC 39's table (needed by SPEC 40, and a third kind of pair for the frozen term).** Nine
first-principles points of the high-spin frozen product at 0.6–3.0 Å (`runs/general_laws_2026-11-13/hh_frame_points.log`).
- The electron-gas default was wrong by a factor 3.5 to 5.6 over 0.6–2.2 Å (2.62 against 10.03 eV at 0.74 Å): two
  one-electron atoms are its worst case. The electrostatic half was right to 5e-6 eV at all nine points
  (out of sample).
- **A defect of the table's representation, found here and fixed in the builder:** the stored ratio c = (first
  principles) / (electron gas) has no meaning where the electron-gas term passes through zero. For H⁰–H⁰ it is
  +0.0041 eV at 2.6 Å and −0.0035 eV at 3.0 Å (first principles +0.0735 and +0.0213), ratios 18 and −6. A point
  now enters only where the electron-gas term itself is above the 5e-3 eV floor. The nine pairs already in the
  table have no such point (smallest 7.3e-3 eV) and are bit-identical after the rebuild (checked array by array).
- **What that leaves, measured out of sample at the two points left out:** the table's overlap term is +0.029 eV at
  2.6 Å and −0.025 eV at 3.0 Å against +0.074 and +0.021: 0.045 eV low in the tail, and of the wrong sign beyond
  2.7 Å. A limit of carrying the tail by a ratio to the electron gas; small against the wall the term is for
  (15.3 eV at 0.6 Å), not small against a hydrogen–hydrogen contact energy. Open.
- The table now holds ten pairs with points (H–H, O–O, O–Mg, O–Al, O–Au, Na–Cl, Mg–Au, Cl–Cl, Ag–Ag, Au–Au); the
  two superseded files are kept in `runs/general_laws_2026-11-13/`.

### Measured: what separates the 'global' / 'phi1' route from the trained route of the single model on undoped cells (2026-11-13 20:34 IST; frozen, final step 33 model of seed 43; `runs/two_site_2026-11-13/probe_route_difference_s43.txt`, `probe_route_difference.py`)

The two undoped Table II references, E − E(trained route: 'clusters' space, 'fermi' band, Harrison bound), in meV:

| one setting changed | wetting | non-wetting |
|---|---|---|
| no coupling bound | 0.000 | 0.000 |
| band 'phi1' | −63.207 | −63.389 |
| space 'global' (with the 'fermi' band) | 0.000 | 0.000 |
| 'global' + 'phi1' | −41.193 | −35.001 |

- The coupling bound is inactive on this model (largest |τ|/B 0.33, below the knee), as stated before.
- **The band form is not inactive:** 'phi1' lies 63.2–63.4 meV below 'fermi' on the undoped cell, the same for both
  geometries to 0.2 meV, so it cancels in wet_u but not in an absolute energy. My earlier "exactly the trained
  model" was measured on the Mac model's Au–Au scan only, where it read 0.000 eV/Å in a force.
- With the 'fermi' band the space setting changes nothing here (0.000; whether that is two routes agreeing or one
  route serving both settings is not looked into). With 'phi1' the 'global' space lies 22.0 and 28.4 meV above the
  'clusters' space: a geometry-dependent 6.4 meV, the size of the cluster closure's stated accuracy class, and the
  whole of the +769.2 against +775.4 in wet_u seen by the sign probe.
- Consequence for today's records: every bond force and E_q in the mechanism check is of the trained route; the
  sign comparison is between two runs of 'global' + 'phi1' and stands as a statement about that route.

### SPEC 40 reviewed: six findings confirmed, two partly; the term is NOT ready for training, and my recommendation to prepare a run is withdrawn (2026-11-14 02:59 IST; four independent readers, eight skeptics told to refute; `runs/two_site_2026-11-14/review_spec40.json`)

The first attempt (2026-11-13 20:45) was cut off before any reader reported; the second ran on 2026-11-14. 32
findings; the eight most severe were each given to a skeptic (six confirmed, two confirmed in part, none refuted);
the other 24 are minor or were not verified a second time and are acted on where I could check them myself.
No reader changed the tree (checked by modification times).

**1. The finding that changes the plan (partly confirmed; reproduced here, `probe_au2_ea_ip.py`,
`runs/two_site_2026-11-14/probe_au2_ea_ip_*.txt`).** SPEC 40 lowers the neutral pair's configuration by ρ and
leaves the other three levels of the pair alone. Gas-phase Au₂ through the package's fixed terms (no learned
shift; identical to the printed digit on the final step 33 models of both seeds and the Mac model):

| R (Å) | EA(Au₂), reference | no new terms | hop only | **hop + relaxation (SPEC 40)** | IP(Au₂), reference | no new terms | hop only | **hop + relaxation** |
|---|---|---|---|---|---|---|---|---|
| 2.30 | 1.85 | 2.12 | 1.17 | **0.17** | 9.43 | 8.94 | 9.54 | **10.54** |
| 2.47 | 2.03 | 2.47 | 1.64 | **0.77** | 9.22 | 8.82 | 9.51 | **10.38** |
| 2.60 | 2.17 | 2.62 | 1.89 | **1.13** | 9.09 | 8.79 | 9.45 | **10.22** |
| 2.80 | 2.37 | 2.78 | 2.18 | **1.58** | 8.91 | 8.75 | 9.33 | **9.93** |

(References: the README of 2026-11-02, PBE0 dimer minus atom on the exact atomic values, stated spread 0.2 eV;
"no new terms" is the trained model's own learned hop with spin pairing off.)
- With SPEC 40 the electron affinity is 0.8–1.7 eV too low and the ionisation energy 1.0–1.2 eV too high, where the
  same hop without the relaxation is within 0.2–0.7 and 0.1–0.4 eV. The ionic gap, the one number I had checked
  (0.01 eV at 2.47 Å), is right at the same time.
- A learned level shift is one-body: it can move a level, not the combination
  I = E(a⁺b⁻) + E(a⁰b⁰) − E(a⁺b⁰) − E(a⁻b⁰), which SPEC 40 moves by exactly −ρ: −5.773 → −7.456 eV at 2.47 Å on all
  three checkpoints. From the references with the table's own t and E_T (mixed levels of theory), I = −4.60 eV:
  the fixed terms were 1.18 eV off before and are 2.86 eV off with SPEC 40.
- Level by level at 2.47 Å, the fixed terms with ρ against the same references: a⁻b⁰ above the covalent level
  −1.576 (reference −2.830: +1.25), a⁺b⁰ +10.019 (+8.420: +1.60), a⁺b⁻ +0.987 (+0.995: −0.01). **Reading
  (a hypothesis until first principles has the charged dimers at the table's level): every pair that holds a
  neutral open-shell atom lacks a relaxation of the size of ρ (1.3–1.7 eV at 2.47 Å); the closed-shell ion pair
  lacks none.** My sentence that the relaxation "belongs to the covalent configuration alone" was wrong.
- Consequence: trained as built, the model would have to hide 1 eV errors of both charged dimers in its level
  shifts, on exactly the class (doped Au–MgO, Au₂⁻) where step 33 already over-transfers. **Not proposed for
  training.** The general form it points to: the one-carrier entries of the pair slot from first principles by
  the same construction. Not built; it needs E(Au₂⁻; R) and E(Au₂⁺; R) at the table's level first.

**2. Confirmed and fixed in the code (failing tests first: 8 red, then green).**
- *The 'window' selection was accepted* (the default, and what my module tests ran on): it does not see the
  couplings, so with the first-principles hop gas-phase H₂ never hopped and Au₂'s energy stepped by 1.6–2.0 eV
  near 1.95 Å (untrained heads). `two_site` now requires `diabatic_selection='amplitude'` (the trained route), the
  tests run on it, and a scan of Au₂ from 1.80 to 3.60 Å in 0.01 Å is tested to have no step.
- *One s¹ atom per cell meeting its own image was skipped, not refused* (and my test asserted the skip): an Au
  chain at 2.9 Å was left alone with one atom per cell, refused with two, and lowered by 0.80 eV per atom with
  three. Now refused in every case.
- *The hydrogen pair was applied by element alone:* two hydrogens of one water molecule (1.51 Å) would get H₂'s
  hop by continuation, 2.5 eV, and each is bonded to a third atom. H–H stays tabulated and is no longer applied
  by the module ("applied" in the data file; Au–Au and Ag–Ag are).
- *My pre-registered rule "a root below 0.5 is flagged, not used" had been applied to the triplet only:* three hop
  points rested on singlet roots of weight 0.47 (H₂, 0.6 Å), 0.45 and 0.36 (Au₂, 2.6 and 2.8 Å). Applied now; the
  three are kept in the file as flagged. The hop changes by −0.2 %, −1.0 % and −2.1 % at 2.6, 2.8 and 3.0 Å
  (Au–Au) and +0.5 % at 0.6 Å (H–H). t / Harrison over the points that pass: Au₂ 0.71–0.74, H₂ 0.43–0.60 (the
  verdict of that entry does not change).
- Also: the coupling's window may not start inside an applied pair's hop points (refused); a malformed table is
  refused when read; `two_site` must be a boolean; the kept relaxation knots follow the frame table in use.
- *Tests that could not fail* (ten surviving mutants: the hop's window, the switch of ρ, the module's wiring in
  periodic cells and in the cluster space, which pair gets which curve, the open direction of the pair list):
  each is now pinned by a test written out independently of the code. `test_redox_two_site.py` 24 → 37 tests and
  one strict xfail; `test_redox_cluster_signs.py` 14 → 30 and two; `test_redox_frame_pair.py` 46 → 47.

**3. Confirmed, recorded, not fixed.**
- *The s–s hop is given to every charge step of the pair* (one coupling per atom pair, SPEC 17.2): 12 of the 16
  one-electron moves of an Au pair move a d electron (Au⁺ → Au²⁺) and now carry the fixed t instead of a learned
  value. No such configuration is selected on 80 test structures at frozen weights (they lie 8.5–10.4 eV up); it
  matters for cationic pairs. Strict xfail test; the cure is a coupling resolved by charge step.
- *The mapping is exact for exact energies and is applied to TDA roots* (the mean-field reading of the same roots
  gives a hop 0.4–13 % larger); the curves are C¹ at the points, which sit on the equilibrium distances. Limits
  (x) and (xi) of SPEC 40.6.

**4. Corrections to what I wrote on 2026-11-13.**
- **The sign-sensitivity check could not have failed.** The selected configuration sets on the slabs are trees
  (the reference and single moves: 39 configurations and 38 non-zero couplings on the wetting reference, no closed
  loop), and a tree's spectrum does not depend on any sign. "24 900 of 48 148 couplings" counted ordered moves, of
  which about 94 % have a zero element; on the wetting reference 19 of 38 non-zero couplings change sign. The
  check gives no evidence on the sign structure, and "the hypothesis about the flat dimer's rings is rejected" is
  withdrawn: it is not testable on that ensemble.
- The sign table left out its exceptions: two holes on the triangle lie 0.27–0.63 t BELOW the exact singlet (the
  model sits on the triplet), two holes on the tetrahedron are exact, and **at half filling the model lies below
  the exact ground state of any spin** (chain of three −0.11 / −0.23 t, triangle −0.61 / −0.73, square −0.50 /
  −0.48, tetrahedron −1.08 / −1.19): it over-binds the neutral sector of three or more s¹ atoms. Now pinned in tests.
- "The ten compressed dimers" was loose: the ten worst structures are not all compressed. Measured from the
  stored per-structure errors: the five undoped test dimers below 2.18 Å (115, 167, 207, 222, 245) carry 92.2 %
  (seed 42) and 94.2 % (seed 43) of the undoped squared force error, and without them it is 0.0544 / 0.0540 eV/Å.
  Doped: the two below 2.18 Å (17, 28) carry 14 % and 53 %.
- Mechanism check, scoring entry: √2 t at 2.00 Å is 2.532 (not 2.531); "within 6 %" is 6.2 %; the gas-phase
  E_gs(2.089) − E_gs(2.57) by the same monotone cubic is +2.24 eV (not +2.26), so E_q's +2.27 agrees to 0.03 eV;
  the gap with the terms on runs +1.5 to +3.0 eV over the whole scan (I quoted +2.4 to +3.0, the range from
  2.089 Å) and as trained −2.0 to +1.2; those gap and element rows are readings of the 'global' / 'phi1' route,
  whose total bond force differs from the trained route's by 0.3–0.7 eV/Å with the terms on (2.1 at 2.15 Å as
  trained); the "backbone" column is the same in all variants to 0.014 eV/Å, not exactly.
- The pre-registration called the two terms (40.1) and (40.2); in SPEC 40 they are (40.2) and (40.3).
- The hydrogen pair's nine frozen points: only four were in the cited log; all nine records are now in
  `runs/general_laws_2026-11-13/hh_frame_points_cache/`. "A factor 3.9" is the mean ratio of the seven points in
  the table; at 0.74 Å the overlap terms are 10.009 against 2.596 eV (3.86).

**5. The mechanism check again, on the table after the weight rule (`runs/two_site_2026-11-14/probe_auau_two_site_s43_table_v2.txt`).**
With SPEC 39 + 29 + 40 on, the bond force of E_q alone is +18.07 / +14.61 / +12.51 / +9.91 / +7.89 / +6.28 / +4.73 /
+2.34 / −0.13 eV/Å at 2.000 / 2.050 / 2.089 / 2.150 / 2.200 / 2.250 / 2.300 / 2.400 / 2.570 Å: the same as on
2026-11-13 to the printed digit up to 2.30 Å, +2.340 for +2.342 at 2.40 Å and −0.129 for −0.089 at 2.57 Å (the hop
beyond 2.47 Å is now the continuation). C1, C2 and C3 pass unchanged (+12.51 against DFT's +15.25; the element
+2.532 → +1.651 eV, one sign). The model's configuration (amplitude selection, Au–Au) is one the fixed code accepts.

**6. Suites.** `tests/redox` 1077 passed, 3 xfailed (the d-move limit and the two sign limits; was 1047 passed, 2 xfailed: +13 `test_redox_two_site.py`, +16 `test_redox_cluster_signs.py`, +1 `test_redox_frame_pair.py`); the rest 1045 passed, 4 skipped, unchanged (Mac, 2026-11-14 02:59 IST).

**What stands and what does not.** The wall mechanism stands (C1–C3). SPEC 40 as a training candidate does not:
it is one of four entries. Next, in order:
- (a) First principles for Au₂⁻ and Au₂⁺ (and the Ag pair) at the table's distances: the doublet's interaction
  energy against the separated ion and atom, and half its σ_g–σ_u splitting as an independent t (which also
  settles limit (x)). To be pre-registered before it is run: a diffuse basis for the anion, the same basis for
  the neutral it is compared with; the character of each state checked (s weight of the singly occupied orbital,
  d population; the cation has a d-hole state near 2.47 Å in an earlier record); and more than one functional,
  because one- and three-electron bonds are where a semi-local functional's delocalisation error is largest (the
  2026-11-02 references differ by 0.2 eV between PBE0 and ωB97X).
- (b) The one-carrier entries (−1, 0) and (+1, 0) of the pair slot by the same construction as ρ, tests first,
  including which learned quantities enter the package's fixed terms for a charged pair (the three checkpoints
  agree to the printed digit at zero shift; why they do is not yet read from the code).
- (c) The pre-registered frozen check: gas-phase EA, IP and U of Au₂ at zero learned shift, each within the
  references' 0.2 eV spread, at 2.30–2.80 Å; and the wall check of 2026-11-13 again.
- (d) Only then a training proposal.

### Pre-registered: first principles for the one-carrier pair, Au₂⁻ / Au₂⁺ and Ag₂⁻ / Ag₂⁺ (2026-11-14 07:33 IST, before any run; no model code changes in this step; `tools/electrostatics/pair_one_carrier_curve.py`)

**Benchmark numbers it serves (per the project's benchmark-first policy).** SPEC 40.6 (iii): the gas-phase electron affinity and ionisation energy of
Au₂ with the new terms (0.8–1.7 and 1.0–1.2 eV off), which stand for the doped Au–MgO class (the charge on the
dimer: T2's shift, wet_d) and, through SPEC 40 as a whole, S3. Nothing is claimed for a benchmark from this step.

**What is computed.** ESTABLISHED: the one-electron and three-electron two-site resonance (as for H₂⁺): the two
configurations a^q b⁰ and a⁰ b^q at one level W_q(R), coupled by t, give the doublets at W_q ∓ t. So from the ion
pair's ground state and its vertical σ_g ↔ σ_u excitation ΔE,

    E_int^q(R) = E[X₂^q] − E[X^q] − E[X]        (counterpoise-corrected, against the separated ion and atom)
    W_q(R)     = E_int^q + ΔE / 2                the no-hop level of the one-carrier pair
    t_q(R)     = ΔE / 2                          an independent hop (it does not pass through the neutral's singlet root)

at the table's distances (Au: 2.0, 2.1, 2.2, 2.3, 2.47, 2.6, 2.8, 3.0 Å; Ag: 2.2, 2.3, 2.4, 2.53, 2.65, 2.8, 3.0 Å).
Unrestricted Kohn–Sham doublet; ΔE from TDA on it, the root picked by its weight on the σ_g → σ_u transition of
the right spin (α for the cation, β for the anion), σ_g and σ_u being the sum and difference of the atoms' own
singly occupied valence s orbitals. def2-TZVP with the def2 ECP for the cation; def2-TZVPD for the anion (and for
the atom, the anion atom and the neutral dimer it is compared with). PBE (the level of SPEC 40's table) and PBE0.

**The tool's own test, to pass before any other number is read.** H₂⁺ in Hartree–Fock (exact for one electron in
the basis; the excitation by CIS is exact too) at 1.0584 Å (2 a₀): exact E_int = −2.793 eV and ΔE = 11.840 eV
(1sσ_g −0.60263 Ha, 2pσ_u −0.16753 Ha). Pass: E_int within 0.03 eV and ΔE within 0.05 eV; the σ_g weight of the
singly occupied orbital and the root's weight both above 0.95.

**Validity of a point (each rule applies to every state it names; a point that fails is printed as flagged and is
not used).** SCF converged and stable; ⟨S²⟩ within 0.02 of 0.75; the spin population on the expected s combination
(σ_g for the cation, σ_u for the anion) at least 0.5; the picked root's weight at least 0.5. The d population per
atom is printed (an earlier record has a d-hole state for the cation near 2.47 Å).

**Cross-checks, with their criteria.**
- X1: PBE0 vertical EA and IP of Au₂ "on the exact atomic values" (EA_exp + EA_dimer^calc − EA_atom^calc, the same
  for IP) against the 2026-11-02 references (EA 1.85 / 2.03 / 2.17 / 2.37, IP 9.43 / 9.22 / 9.09 / 8.91 eV at 2.30 /
  2.47 / 2.60 / 2.80 Å): within 0.10 eV at every distance (same functional, another basis and code path).
- X2: t_q against the neutral table's t(R) at the same distance: reported; I predict within 10 %, the anion's
  below the cation's (a more diffuse carrier is not obviously larger or smaller; this is a guess and is labelled so).
- X3: PBE against PBE0 for W_q: reported as the level-of-theory spread. One- and three-electron bonds are where a
  semi-local functional's delocalisation error is largest; I predict PBE's W_q below PBE0's by 0.1–0.4 eV at
  2.47 Å, more at 3.0 Å. If the spread exceeds 0.3 eV at 2.47 Å, the level of theory of the whole table (the
  neutral's included) becomes a decision before anything is built on it.
- What the numbers are for: the relaxation the one-carrier entries lack, the package's fixed-term level minus
  W_q. From the review's numbers and the references I expect about 1.2–1.3 eV (anion) and 1.6 eV (cation) at
  2.47 Å; if the first-principles W_q put it below 0.3 eV for both, the reading "every pair with a neutral
  open-shell atom lacks a relaxation of the size of ρ" is wrong and SPEC 40.6 (iii) has to be re-read.

### Amendment to the pre-registration above, before any gold or silver number: both resonance states by constrained SCF, not by TDA (2026-11-14 07:36 IST)

The tool's own test failed as first written, for a reason in the method and not in H₂⁺: for a one-electron system
PySCF's Hartree–Fock object diagonalises the one-electron Hamiltonian, and TDA on that object adds the two-electron
kernel to orbital energies that carry none (H₂⁺ at 2 a₀: the root came out at 0.73 eV; the orbital-energy
difference itself was 11.807 eV in def2-TZVP and 11.840 eV in aug-cc-pVTZ, exact 11.840). So the TDA route had no
exact case to be tested on. **Changed, and better on its own terms:** the two states of the resonance are each
computed as a self-consistent state with the electrons counted per symmetry species (D2h on the bond axis: the
carrier in A_g for σ_g, in B1u for σ_u), ΔE is their difference, and the same counting applies to H₂⁺ (so the tool
is tested on the route it uses). Both states are then relaxed variational states, like E_gs itself. Everything
else of the pre-registration stands; the validity rule "the picked root's weight at least 0.5" becomes "the spin
population on the expected s combination at least 0.5 in each of the two states" (A_g also holds the d_z² and
d_x²−y² combinations, so the symmetry count alone does not make a state the s-carrier state).
- H₂⁺ criterion as written (def2-TZVP, E_int within 0.03 eV): **not met by the basis**, −2.748 against −2.793 eV
  (0.045 eV; aug-cc-pVTZ gives −2.784, 0.009 eV). Recorded as failed as written; the basis error of H₂⁺ in
  def2-TZVP is not a property of the tool. The tool's test is therefore read in aug-cc-pVTZ: E_int within 0.03 eV
  and ΔE within 0.05 eV of the exact values.

### Scored: the one-carrier pair from first principles (60 points); and pre-registered: the pair's remaining entries (2026-11-14 07:43 IST; `runs/general_laws_2026-11-14/{au2,ag2}_{cation,anion}_{pbe,pbe0}.log`, `one_carrier_table.txt`, `h2plus_tool_test.log`)

**The tool's test (H₂⁺, Hartree–Fock, both states by constrained SCF).** aug-cc-pVTZ: E_int −2.788 eV (exact −2.793),
ΔE 11.840 eV (exact 11.840): pass. The spin population on σ_g of the atoms' own 1s is 0.94, 0.01 below the 0.95 I
wrote (the molecule's σ_g is contracted against the free atom's 1s): failed as written, by 0.01. def2-TZVP: −2.753
and 11.807 eV, the basis error recorded in the amendment.

**Validity.** PBE: 30 of 30 points valid (Au₂⁺, Au₂⁻ at eight distances, Ag₂⁺, Ag₂⁻ at seven); d population 9.69–9.97
per atom in every state, spin population on the expected s combination 0.51–0.97: no d-hole state. PBE0: 23 of 30;
seven are flagged by my ⟨S²⟩ rule alone (0.770–0.774 against 0.75 ± 0.02; Au₂⁺ from 2.47 Å, Ag₂⁻ from 2.65 Å), their
characters being as good as the others'.

**Cross-checks.**
- **X1 PASS.** PBE0, on the exact atomic values, at 2.30 / 2.47 / 2.60 / 2.80 Å: EA(Au₂) 1.817 / 1.998 / 2.132 / 2.333
  (references 1.85 / 2.03 / 2.17 / 2.37: −0.03 to −0.04); IP(Au₂) 9.462 / 9.257 / 9.124 / 8.940 (9.43 / 9.22 / 9.09 /
  8.91: +0.03 to +0.04; the last three are ⟨S²⟩-flagged points). All within 0.10 eV.
- **X2: my prediction (within 10 %) fails for the cation.** Against the neutral table's t(R): the cation's
  t is 13–28 % larger (Au, 3.0 → 2.0 Å: 0.986 against 0.873, 2.298 against 1.790) and 9–27 % larger for Ag; the
  anion's is within −8 to +10 % for Au (1.254 against 1.249 at 2.47 Å) and 8–21 % smaller for Ag. The anion's is
  below the cation's everywhere, as guessed. **One hop per pair is therefore not one number for the three charge
  sectors**: the earlier "one t for the neutral and the anion to 0.07 eV" holds for Au near 2.47 Å and is not a
  general statement. It adds to SPEC 40.6 (ix): a hop resolved by charge step.
- **X3: the trigger I set is met, and it does not bite where the model takes its numbers.** The no-hop level W_q
  from PBE lies below PBE0's by 0.40–0.47 eV (Au₂⁺), 0.18–0.26 (Au₂⁻), 0.26–0.29 (Ag₂⁺), 0.17–0.19 (Ag₂⁻): above the
  0.3 eV I named for Au₂⁺. But W_q is not what the model needs. It needs the ion pair's ground state against the
  neutral's in one functional, i.e. the EA and IP on the exact atomic values, and there PBE and PBE0 differ by
  −0.14 to −0.10 eV (EA, Au), −0.07 to +0.02 (IP, Au), −0.11 to −0.09 (EA, Ag), +0.06 to +0.10 (IP, Ag): the
  functional's error is largely common to the three charge states. **Decision, mine, with its reason:** PBE for
  all three charge states, the level of SPEC 40's neutral table, so that the three ground states are one
  functional's; the spread of what enters is at most 0.14 eV and is carried as the level-of-theory uncertainty of
  the entries. (Experiment is not used to choose: I have no verified experimental numbers in the repository.)
- **What the numbers were for.** PBE, at 2.47 Å: EA 1.893, IP 9.192 eV. With the review's fixed-term levels the
  one-carrier entries lack about 1.1 eV (anion) and 1.6 eV (cation) against a level defined with the model's own
  hop; the reading of SPEC 40.6 (iii) stands (measured exactly in the build below).

**Also measured, needed by the build: the package's fixed-term levels of gas-phase Au₂ do not depend on any
weight.** With the learned shift at zero, the four configuration levels at 2.47 and 2.80 Å are the same to 1e-4 eV
(a) on three trained checkpoints, (b) after every one of the final seed-43 model's 193 521 parameters is perturbed
at random by 5 % and by 30 % of its tensor's mean size (the total energy moves by 0.42 eV, so the perturbation
acts), and (c) in a freshly built module with random heads and features (two seeds) and the single model's
options. They are a function of the option set alone, so the package can evaluate its own fixed terms for an
isolated pair without a checkpoint.

**Pre-registered: SPEC 40.7, the pair's remaining entries (before any code; benchmark numbers as in the entry
"Pre-registered: first principles for the one-carrier pair").**
NEW-TO-TRACE: for an applied pair, three more entries of the state-pair slot, each fixed by one condition on the
ISOLATED pair through the package's own fixed terms (no learned shift) and its tabulated hop (40.3):

    O_ab(∓1, 0) = O_ab(0, ∓1) = −s_ρ ρ_∓(R)     such that the ion pair's ground state is  E_gs(R) ∓ {EA*(R), −IP*(R)}
    O_ab(+1, −1) = O_ab(−1, +1) = +s_ρ λ(R)      such that the neutral pair's ground state is  E_gs(R)

with E_gs the table's ground-state interaction energy and EA*, IP* the first-principles vertical values on the
exact atomic values (PBE; ESTABLISHED quantities). With (40.2) the isolated pair is then first-principles in all
three charge states at every tabulated distance, with one hop for all sectors; the excited member of each doublet
is not (X2). The conditions are solved once per pair and distance by a builder that calls the package (the
entries are a property of the package's fixed terms under a recorded option set; a model whose options differ is
refused, and a test recomputes the model side so that a change of the fixed terms is seen).
- Tests written first, and what passes: gas-phase Au₂ through a freshly built module at zero shift has
  E(Au₂) − 2E(Au) = E_gs, EA = EA*, IP = IP* to 1e-6 eV at each of the eight distances; the same on the two final
  step 33 checkpoints to 1e-4 eV; no entry other than the named ones changes; a model with another option set is
  refused; forces are the gradient and a force loss differentiates; a scan of each charge state in 0.01 Å has no step.
- **(c) the frozen check, criteria fixed now.** On the final step 33 models at zero learned shift: EA and IP of
  gas-phase Au₂ within 0.2 eV of the 2026-11-02 references at 2.30 / 2.47 / 2.60 / 2.80 Å (by construction they
  will be PBE's 1.715 / 1.893 / 2.022 / 2.211 and 9.390 / 9.192 / 9.073 / 8.922: errors −0.14 to −0.16 and −0.04 to
  +0.01, so this passes unless the build is wrong); and the wall check again on test 167's scan, C1 (repulsive and
  monotone from 2.00 to 2.40 Å) and C2 (within ±30 % of +15.25 eV/Å at 2.089 Å). Prediction: the force at 2.089 Å
  rises from +12.5 toward the gas-phase +14.5 (13.5 to 15.5), because the ionic entry removes the error of the
  neutral's ground state at short range (−0.43 eV at 2.0 Å by my estimate). If it falls instead, the ionic
  entry's sign or the solve is wrong.
- Not in this step: Ag–Ag's entries need an option set with silver's state polarizabilities; until they are
  built, Ag–Ag will be withheld from the module like H–H (a pair with one of its four levels corrected is the
  state the review rejected).

### Built and scored: SPEC 40.7, the gold pair's remaining entries; the frozen checks pass (2026-11-14 08:02 IST; `tools/electrostatics/build_two_site_entries.py`; `runs/two_site_2026-11-14/probe_au2_ea_ip_entries_*.txt`, `probe_auau_two_site_s43_entries.txt`; off by default, nothing trained with it)

**What was built, as pre-registered.** For Au–Au, three more entries of the pair slot, each solved from one
condition on the isolated pair through the package's own fixed terms and the tabulated hop: ρ₋ on the
one-extra-electron configurations (so that E[Au₂⁻] = E_gs − EA*), ρ₊ on the one-hole configurations
(E[Au₂⁺] = E_gs + IP*), λ on the ion pair (E[Au₂] = E_gs). The solve meets each condition to 6e-11 eV and gives
the same entries with two differently seeded modules to 7e-15 eV.

| R (Å) | package EA without the entries | EA* | package IP without | IP* | ρ₋ | ρ₊ | λ |
|---|---|---|---|---|---|---|---|
| 2.00 | −1.631 | 1.355 | 10.720 | 9.937 | 2.526 | 0.408 | +0.945 |
| 2.20 | −0.288 | 1.604 | 10.617 | 9.538 | 1.685 | 1.026 | +0.459 |
| 2.47 | +0.772 | 1.893 | 10.376 | 9.192 | 1.117 | 1.294 [superseded 2026-11-14 afternoon, see "SPEC 40.7 reviewed"] | +0.008 |
| 2.80 | +1.581 | 2.211 | 9.927 | 8.922 | 0.735 | 1.171 | −0.301 |
| 3.00 | +1.943 | 2.389 | 9.656 | 8.793 | 0.586 | 1.067 | −0.444 |

(eV; all eight distances are in SPEC 40.7. The ρ₊ column and the package IP column of this table are superseded:
see "SPEC 40.7 reviewed".) The reading of the review stands and is now measured: every
configuration of the pair that holds a neutral open-shell atom lacked a relaxation of 0.4 to 2.5 eV; the ion pair
was 0.9 eV too low at 2.0 Å (the model's fixed terms put Au⁺Au⁻ 2 eV BELOW Au⁰Au⁰ there without SPEC 40) and
0.4 eV too high at 3.0 Å.

**Tests first.** Against the completed table 21 of the 37 earlier tests failed (the entries change which
configurations move; silver is withheld; a model with other options is refused); rewritten to the completed
design and extended, `test_redox_two_site.py` holds 42 tests and one strict xfail [72 on 2026-11-15, 78 after the second review]: the three conditions at each
of the eight distances to 1e-6 eV, with a counter-check that without the entries the same module misses EA by more
than 1 eV; each class of configuration shifted by its own entry, in the gas phase, in a crowded periodic cell and
for the pair alone in a periodic cell; independence of the weights, with a counter-check that the learned shift
does move the energies; refusal of another option set and of a coupling cutoff below 4.0 Å; scans of Au₂, Au₂⁻
and Au₂⁺ without steps.

**Silver: withheld, with a measured reason.** The same build for Ag–Ag met its conditions and then failed the
builder's own guard: two differently seeded modules differ by 0.057 eV, because the polarizability table holds Ag⁻
only and the neutral silver frame's polarizability is the learned head's. Its entries cannot be built until Ag⁰ and
Ag⁺ are tabulated, and with the neutral entry alone the pair would be in the state the review rejected, so Ag–Ag
is no longer applied by the module (its points and its carriers at seven distances stay in the file).

**(c) the frozen checks, as pre-registered.**
- **EA and IP of gas-phase Au₂ at zero learned shift, final step 33 models (both seeds identical): PASS.**

| R (Å) | EA, reference | before 40.7 | **with 40.7** | IP, reference | before 40.7 | **with 40.7** |
|---|---|---|---|---|---|---|
| 2.30 | 1.85 | 0.17 | **1.715** | 9.43 | 10.54 | **9.390** |
| 2.47 | 2.03 | 0.77 | **1.893** | 9.22 | 10.38 | **9.192** |
| 2.60 | 2.17 | 1.13 | **2.022** | 9.09 | 10.22 | **9.073** |
| 2.80 | 2.37 | 1.58 | **2.211** | 8.91 | 9.93 | **8.922** |

  Errors −0.14 to −0.16 eV (EA) and −0.04 to +0.01 eV (IP), inside 0.2 eV; they are PBE's own differences from the
  PBE0 references, as the construction says they must be. The ionic gap at the hop's five distances is the
  table's U to the printed digit.
- **The wall check again (test 167's scan, seed 43, bond force of E_q alone, trained route): C1, C2, C3 PASS.**
  +19.65 / +15.82 / +13.51 / +10.68 / +8.48 / +6.77 / +5.13 / +2.64 / +0.05 eV/Å at 2.000 / 2.050 / 2.089 / 2.150 /
  2.200 / 2.250 / 2.300 / 2.400 / 2.570 Å (before the entries +18.07 … +12.51 … −0.13). At 2.089 Å: +13.51 against
  DFT's +15.25 (−11 %); my prediction was 13.5 to 15.5, met at its lower edge. The covalent–ionic element is
  +2.532 → +1.651 eV, one sign.
- Read with the caveats of the earlier checks: frozen weights; the backbone and the level shifts of a model
  trained without these terms are counted on top (the "as trained shift" rows of the EA/IP probe show it: IP 0.75
  to 0.81 eV high). These checks say the construction is right for the isolated pair and that the wall is there on
  the surface; they are not a trained model's test force or Table II.

**Where this leaves SPEC 39 + 40 for gold.** The review's blocking finding is answered for Au–Au. Still recorded
and open: the pair functions are the isolated pair's, transferred to a surface; one hop per pair for three charge
sectors (the cation's own is 13–28 % larger); the s–s hop on d-electron moves (strict xfail) [cured the same afternoon: the hop is resolved by charge step]; the configuration
Hamiltonian's over-binding of three or more neutral s¹ atoms; the curves' C¹ knots at the equilibrium distances;
the level-of-theory spread of 0.14 eV. **A training run with it is now a proposal awaiting a decision, not
launched**: a cold start of the single model with `frame_pair`, `spin_pairing` and `two_site` on, both seeds, step
33's criteria; one GPU check of the new code path first (it has only run on CPU).

**Suites after SPEC 40.7.** `tests/redox` 1082 passed, 3 xfailed (was 1077 passed, 3 xfailed: +5 in `test_redox_two_site.py`, now 42 tests and one strict xfail); the rest 1045 passed, 4 skipped, unchanged (Mac, 2026-11-14 08:15 IST).

### SPEC 40.7 reviewed (two of three readers finished; no skeptic ran): the cation's entry was solved through a spurious channel; fixed at the root, the entries rebuilt (2026-11-14 16:00 IST; `runs/two_site_2026-11-14/review_spec40_7_partial.json`)

The review was cut off by the usage limit: the physics and the code reader reported (eight findings each); the
test-and-records reader and all five skeptics did not run. The two readers found the same central defect
independently, and I checked each finding below myself before acting on it. The cut-off parts are rerun on the
corrected code (next entry).

**1. The defect (both readers; reproduced).** The cation sector of the isolated pair is not {Au⁺Au⁰, Au⁰Au⁺}: at
zero shift the selection also holds Au²⁺Au⁻ and Au⁻Au²⁺, only 1.0 to 5.2 eV up, and the tabulated s–s hop sat on
the atom pair, so those d-electron moves carried it (the limit I had recorded as SPEC 40.6 ix and left for
later). They lowered the cation by 0.877 / 0.333 / 0.058 eV at 2.0 / 2.47 / 3.0 Å, and ρ₊ had been solved through
them: 0.41 … 1.07 eV where the s resonance alone needs 1.28 … 1.13. Through the amplitude switch on that coupling
the "fixed" terms also depended on the selection thresholds (0.165 eV at 2.8 Å) and on kT (the 1.3 meV I had put
down to temperature). My sentence "every pair that holds a neutral open-shell atom lacked a relaxation of 0.4 to
2.5 eV" was wrong for the cation.
- **Fixed at the root: the hop is resolved by charge step** (`trace_redox.diabatic.StepCouplings`, through the
  Hamiltonian, the amplitude selection and the cluster space; `TwoSite.resolve_hops`). (40.3) is the coupling of
  s moves only (an s electron leaves a donor of charge −1 or 0 and enters an acceptor of charge 0 or +1); every
  other move of the pair keeps the learned coupling. "Fixed terms" now means no learned shift AND no learned
  coupling (`fixed_terms_only`, declared in the module, refused in training mode).
- Tests first: the former strict xfail is now a passing test of the cure; the cation of the isolated pair is the
  two-level resonance to rounding at 2.0, 2.47, 3.0 Å; the pair's fixed terms are the same for other thresholds
  to 1e-6 eV and for another kT to its thermal weight (3e-5 eV at 0.2 eV).
- **Entries rebuilt.** ρ₋ and λ are unchanged to the printed digit; ρ₊ is 1.285 / 1.463 / 1.591 / 1.651 / 1.627 /
  1.535 / 1.336 / 1.125 eV at the eight distances (the physics reader's own numbers for the s resonance).

**2. Other findings, each checked, and what was done.**
- *The refusal list was neither necessary nor sufficient* (code reader). A sweep of every field of RedoxConfig
  and ElectrostaticConfig (now a test) finds five that move the pair's fixed terms: `bond_partition` (1.2 eV),
  `pauli_table` (2.2), `carrier_induction` (0.7), `frame_polarizability_anchor` (0.24) and the electrostatic
  `gaussian_a` (0.24), which was not checked and which my record said did not matter ("the electrostatic settings:
  0.0 eV": I had tried one setting). `gaussian_a` is now stored and checked; the refusal test asserts each of the
  four options with the size of what it moves.
- *ρ₊ beyond 3.0 Å* (both): an exponential of 2 Å decay length, 0.40 eV where a switch dimensioned for 0.008 eV
  starts, a force of 0.7 eV/Å at 5.5 Å made by that switch. **Every entry now ends 1 Å beyond its last point**
  (its line times the quintic switch), where the coupling's window has closed the hop it completes.
- *A plain re-run of the collector left Au–Au applied without entries*, the state the first review rejected, and
  nothing refused it (code reader). Now: a pair marked applied without built entries is refused when the file is
  read; the collector carries the entries over; the entries record the table values they were solved with and a
  changed table or frame table is refused.
- *The builder* mutated the process's cached table and its `--check` could not fail: it builds on a private
  object and restores the cache, and `--check` exits non-zero on a difference.
- *The cation's "ground state" was the lower of two symmetry counts only* (physics reader). Checked with an
  unconstrained calculation (`runs/general_laws_2026-11-14/au2_cation_unconstrained_check.log`): at 2.47 Å it is
  the same state; **at 2.2 and 2.0 Å a d-hole state lies 0.32 and 1.07 eV lower** (spin 0.91–0.95 on d functions).
  So IP* below about 2.4 Å is the s-hole state's, not the cation's ground state's, and the model (Au⁺ = 5d¹⁰) has
  no such state. Recorded as a limit; it does not touch the slabs (next item).
- *Pair states without an entry* (physics reader): Au⁻Au⁻ and charges beyond ±1. Measured on the final seed-43
  model with the terms on (`probe_doped_pair_states.py`, 40 + 40 test slabs): undoped, neutral 0.805, ion pair
  0.192, one electron 0.003, one hole 0.000; doped, one electron 0.997, Au⁻Au⁻ 0.003 (at most 0.012 in a
  structure). The classes with entries carry all the weight, the doped class is the one-electron class (the
  cleanest entry, an exact two-level resonance), and the cation's limits carry no weight on these slabs.
- *Beyond the last point* (physics reader): measured after the fix, the isolated pair's binding is taken off
  between 3.0 and 4.0 Å (Au₂: −1.65 eV at 3.0 Å, −0.20 at 3.6 Å; force up to 3.1 eV/Å at 3.3 Å against 1.7 at
  3.0 Å; Au₂⁻ 2.6, Au₂⁺ 3.3 eV/Å). The description ends at 3.0 Å; one 4G structure lies beyond (3.056 Å).
- Wording: "(40.5) needs no singlet root" (U is fixed given the hop; at 2.6–3.0 Å the hop is a continuation);
  the basis mix of EA* (12–16 meV); the stability analysis named in the pre-registration was not run for the
  symmetry-counted states. All three are in SPEC 40.7's text and limits now.

**3. The frozen checks again, on the corrected build.**
- Gas-phase Au₂ through the fixed terms (now: no learned shift and no learned coupling), both final step 33 models
  identical: EA 1.715 / 1.893 / 2.022 / 2.211 and IP 9.390 / 9.192 / 9.073 / 8.922 eV at 2.30 / 2.47 / 2.60 / 2.80 Å:
  the same as before the fix (the conditions are the same; what changed is the state they are met on), errors
  −0.14 to −0.16 and −0.04 to +0.01 eV against the references: **pass**. The one-hole level above the neutral one
  is now the two-level value (8.39 to 8.46 eV; 8.47 to 9.34 before).
  (`runs/two_site_2026-11-14/probe_au2_ea_ip_resolved_*.txt`)
- The wall check (test 167, seed 43, bond force of E_q, trained route): +19.65 / +15.82 / +13.51 / +10.68 / +8.48 /
  +6.77 / +5.13 / +2.64 / +0.05 eV/Å at 2.000 … 2.570 Å, the same to the printed digit as before the fix (no
  one-hole state on the undoped slab): **C1, C2, C3 pass** (+13.51 against DFT's +15.25).
  (`runs/two_site_2026-11-14/probe_auau_two_site_s43_resolved.txt`)

**4. Suites.** `tests/redox` 1100 passed, 2 xfailed (the two sign limits; was 1082 passed, 3 xfailed: +18 in `test_redox_two_site.py`, now 60 tests and no xfail, the former d-move xfail being a passing test of the cure); the rest 1045 passed, 4 skipped, unchanged (Mac, 2026-11-14; the second run's wall time includes hours of sleep).

### SPEC 40.7: what the interrupted second review's mutation tests showed, and what was fixed (2026-11-15 07:08 IST)

The second review was cut off three times (network loss during sleep twice, then the usage limit) before any
reader reported. Its mutation runs were complete on disk: 25 mutants of the package against
`test_redox_two_site.py`. Seven survived; new tests kill all seven (checked by running them against the same mutant
copies): the selection and the cluster linkage judging a d move by the unresolved hop (four), the pair importance
with the unresolved coupling, the untested downfold refusal (an earlier check refused the combination first), and
a stale-table check that only looked at E_gs. While checking, two record gaps turned up and were fixed with tests:
entries without their `solved_with` record, or without the recorded `gaussian_a`, escaped every consistency check
(with `solved_with` removed and E_gs changed by 0.5 eV the file loaded and ran). `test_redox_two_site.py` holds 72
tests. No package physics changed; the frozen checks are unaffected. The review itself was relaunched on the
current code.
Suites: `tests/redox` 1112 passed, 2 xfailed; the rest 1045 passed, 4 skipped.

### SPEC 40.7, second review (completed): no defect in the package's physics; five test gaps and four record faults, fixed (2026-11-15 07:44 IST; two readers, three skeptics; `runs/two_site_2026-11-14/review_spec40_7_second.json`)

The review of the corrected code finished on the fourth attempt. Its readers ran 13 new mutants on top of the 25 of
the interrupted attempt. Verified by skeptics: two confirmed, one in part, none refuted; nine minor findings, each
checked here before acting. **The production code was right in every case examined**: the review's own
measurements found the step-resolved couplings correct by enumeration, the default path unchanged, and the
three conditions met. What it found were places where a regression could come back unseen, and stale records.

**Test gaps, each now closed by a test that kills the mutant (run against the review's own mutant copies):**
- *The module's Hamiltonians* could drop the step record in either route and all 72 tests still passed: every
  module-level test of the cure ran on fixed terms, where the learned coupling is zero and no d-move configuration
  is selected. New test: the module with learned parts on and the learned Au–Au coupling made 0.7 eV larger, so
  that Au²⁺Au⁻ is selected. The d-move element is the learned coupling (times its amplitude switch: 1 at 2.0 Å,
  0.76 at 2.47 Å, measured), the s-move element is t(R), and the global and cluster spaces agree to 1e-9. The two
  mutants moved this energy by 0.27–0.64 eV.
- *The band unit's padding* in the step record: a mutant that masked the band unit passed (with the reference's
  Fermi band the band row is zero; with `phi1` it moved a cell by 0.16 eV). New unit test of the padding.
- *Below the first point* (2.0 Å): a constant in place of the end-slope line passed. New test of the line.
- *Tolerances*: the refusal tests perturbed 10³–10⁹ times above the checks' tolerances; now each is probed at ten
  to a hundred times its tolerance. The option sweep now runs at 2.0, 2.47 and 3.0 Å (was 2.47 Å only).
- One mutant that survives is equivalent and left: masking every atom pair in the step record changes nothing,
  because off the applied pairs the table and the learned couplings are the same matrix.

**Code gaps found by the review and fixed (tests first):**
- The stale-table check skipped a missing (null) value, and compared the carriers' EA*, IP* by position, so a
  table with E_gs missing at 3.0 Å, or with the carriers moved by 0.01 Å, loaded. Now every value is compared at
  the entries' own distances and a missing one is refused.
- A change of a data file behind the pair's fixed terms was not refused (alpha(Au⁺) × 1.5 moved the conditions by
  0.016–0.023 eV unseen). The entries now record the content hash of the four files (ion densities, Pauli overlay,
  ion polarizabilities, frame-pair table) and are refused when one differs. Changes of the package's code or
  constants are still not guarded at load (recorded in SPEC 40.7 (e)); the tests and `--check` see them.
- The pair slot's hook applied the entries unchecked when called without the model's settings. It now refuses.
- Entries rebuilt to carry the new record; the values are bit-identical (`--check`: 0.0 eV).

**Records corrected.** SPEC 40.6 (iii) and two_site.py's limit (iii) gave the ionisation-energy error of §40.2–40.3
alone as 1.0–1.2 eV, a number measured while d moves still carried the s–s hop; on the current code it is 1.2–1.7
eV. The beyond-3.0 Å forces now cite a saved probe (`probe_two_site_beyond_points.py`; Au₂⁺ 3.36 eV/Å at 3.38 Å,
which I had written as 3.3). The three unmarked ingredients of SPEC 40.7 are marked (the step rule NEW-TO-TRACE,
the two-level resonance ESTABLISHED, the 1 Å switch-off a NEW-TO-TRACE choice). Superseded statements in the
earlier entries above now carry markers at their places.

**The frozen checks are not affected** (no physics changed: the entries are bit-identical).
**Suites.** `tests/redox` 1118 passed, 2 xfailed (+6: `test_redox_two_site.py` 78 tests); the rest 1045 passed, 4 skipped, unchanged (Mac, 2026-11-15 07:58 IST).

## Pre-registered: step 34, the single model with the gold pair's first-principles sector (written 2026-11-15 08:36 IST, before any launch; decision of 2026-11-15: prepare it without the control run)

**What changes against step 33, and only this** (`derive_step34.py` prints the key-by-key difference; `check_run_config.py` OK
on all runs): `frame_pair` (SPEC 39, the frame pair's own interaction for every library pair), `spin_pairing` (SPEC
29) and `two_site` (SPEC 40 with 40.7, Au–Au's hop, covalent relaxation and the three entries of the pair slot), and
**stage A starts cold** (no warm start: weights trained without these terms hold the Au–Au repulsion in the backbone).
Data, priors, the two-stage recipe (A: 40 epochs, lr 1e-3 cosine to 5e-5, energy weight 10; B: 20 epochs from A's
best, lr 1e-4 to 5e-6, energy weight 1000), both seeds, every other option: step 33's.
**Two changes in one run, plus the cold start, by decision:** `spin_pairing` is switched on with the other
two because SPEC 40's hop needs it; no control run with spin pairing alone. If the run fails, it does not say which
of the three (terms, spin factor, cold start) failed. Against step 33 every difference is confounded with the start.

**Benchmark numbers it is for (per the project's benchmark-first policy).** S3 above all: in step 33 the five undoped test dimers below 2.18 Å (115,
167, 207, 222, 245) carried 92.2 % (s42) and 94.2 % (s43) of the undoped squared force error. Then S2 (Table II),
and S4 / S5 must not be lost (`frame_pair` adds the frozen O–O and Mg–O interactions to the bulk cells too).

| id | criterion (each seed, after B) | comparators |
|---|---|---|
| **S1** | both stages finish: no refusal, no NaN stop, no memory failure | step 33: both finished |
| **S2** | Table II by the T5 protocol, single cell: wet_u within ±100 meV of +928.9, wet_d within ±150 of −2.7, T5 within ±15 % of −931.6 | step 33 s42 +768.4 / −9.2 / 83.5 %; s43 +789.9 / +92.7 / 74.8 % |
| **S3** | 4G test force RMSE ≤ 0.10 (undoped) and ≤ 0.13 eV/Å (doped) | step 33 0.1930 / 0.1043 (s42), 0.2211 / 0.1563 (s43) |
| **S4** | phonons against PBE (`score_phonons.py`, a = 4.2557 Å): P0, P1, P2, P4 pass, and at least 6 of 7 | step 33 6/7 both seeds |
| **S5** | the crystal factor within 0.85–0.95 and the prior cell's ε∞ within 0.05 of 3.235 | step 33 FAIL (g 0.936 / 0.942, ε∞ 0.056 / 0.073 off) |
| **S7** (new) | the trained model on test 167's rigid Au–Au scan (`probe_auau_two_site.py`, "as trained"): total bond force repulsive and monotone from 2.00 to 2.40 Å, and within ±30 % of DFT's +15.25 eV/Å at 2.089 Å | step 33: −17.0 eV/Å at 2.089 Å (s43) |
| **S8** (new) | the undoped test force RMSE without the five dimers below 2.18 Å at most 0.060 eV/Å (the new terms must not cost the rest) | step 33 0.0544 / 0.0540 |
| S6 | reported: the undoped class offset, Table II on the 2×2 supercell, T2, test E RMSE, the largest \|τ\|/B, minutes per epoch, peak GPU memory, the gas-phase EA / IP / U of Au₂ at zero shift (construction check) | |

**PASS** = S1–S5, S7, S8 on both seeds. **Predictions (hypotheses).** S7: pass (the frozen check gave +13.5 eV/Å with a
backbone not trained for it). S3 undoped: pass, because the collapse was nearly all of the error and S8's rest was
already 0.054. S3 doped and S2: not predicted (step 33's wet_u was 140–160 meV low and is unattributed; SPEC 39 changes
the Au–O frozen interaction, which wetting rests on). S4, S5: I expect S4 to pass and do not predict S5 (it failed in
step 33). S1: the risk is the cold start; the cold starts with the bulk cells of 2026-11-07 died in epoch 1 at the
configuration cap, under another option set; the Mac smoke epoch below is the check.
**Pre-flight, before the launch:** (1) one cold-start epoch of the real training path on the Mac
(`runs/AuMgO_twosite_smoke_mac_s43`, stage A's config with `epochs: 1`): it must finish with no refusal; its epoch time
against step 33's Mac early look. (2) A GPU check on GPU host 2 in a working copy of its own (`<gpu-host-workdir>`;
the copies of steps 32 and 33 are not touched): the same structures' energies and forces on the GPU against the Mac.
(3) An independent check of these configs, the queue and this entry. The launch waits for a decision.

**Amendment to the step 34 pre-registration (2026-11-15 08:52 IST, before any launch; after an independent check of the launch kit,
`runs/step34_preflight/check_launch_kit.json`: no blocker, three findings verified).**
- **S7, stated exactly:** the trained model's TOTAL bond force on test 167's rigid scan at the probe's grid, 2.000,
  2.050, 2.089, 2.150, 2.200, 2.250, 2.300, 2.400 Å: positive at each, falling at each step outward, and within
  ±30 % of +15.25 eV/Å at 2.089 Å. My prediction quoted the wrong number: in the frozen check the TOTAL force was
  +15.30 eV/Å (E_q alone +13.51), both on step 33's trained backbone, not on a cold start.
- **Confounds, completed:** besides the three options, the spin factor and the cold start, the cold start re-fits
  the per-element energy references from the slabs (step 33 took them from step 31's checkpoint). The code tree is
  not a confound with the options off: today's tree reproduces step 33's stored test force errors to 7e-13
  (relative, six test structures; `runs/step34_preflight/code_tree_reproduces_step33.txt`).
- **Refusals:** a refusal ends that seed's stage (and the queue skips stage B); nothing skips a structure and nothing
  resumes on its own. The structure-dependent refusals of the new terms are ruled out for both seeds and every
  epoch by one complete epoch; the weight-dependent ones (the configuration cap, which killed the cold starts of
  2026-11-07) are checked for epoch 1 of BOTH seeds (a second Mac smoke, `runs/AuMgO_twosite_smoke_mac_s42`), not
  for later epochs.
- **The CUDA training path** (the force loss's second derivative through the new terms) has never run on a GPU:
  pre-flight (2b), `gpu_train_check_step34.sh`: stage A's config of both seeds together on GPU host 2 until optimizer
  step 3, peak GPU memory sampled; then stopped by exact PID. Pass: three steps each, no failure, peak below the
  card's 16 GB.
- **Scored by script now:** S5 (`score_step34_s7_s8.py`: the crystal factor from the checkpoint, the prior cell's
  ε∞ from the log, reproducing step 33's s43 values 0.9423 and 3.308). By hand, as in step 33: Table II on the 2×2
  supercell, peak GPU memory, minutes per epoch. The watcher refuses to score if the data files differ between the
  two machines.

**Pre-flight (1), seed 43, scored (2026-11-15 09:22 IST).** The cold-start epoch of the real training path finished: no refusal, no
non-finite gradient, 2657 s on the Mac (shared with seed 42's smoke from 08:49), validation force RMSE 0.404 eV/Å and
energy 203 meV/atom after one epoch (`runs/AuMgO_twosite_smoke_mac_s43/log.txt`). **A known risk, measured in this
project's own logs:** the prior cell's ε∞ after this epoch is 21.49 (target 3.235). Of 13 earlier runs whose ε∞ was
20–37 after epoch 1, ten ended at 2.96–3.03 and three, all seed 42, stayed at 19.0–19.1 to their end (the `_pfp_anc`,
`_pfp_bop` and `_pfp_bopw_e1k` runs). If that happens here, S4 and S5 fail on that seed. Early look, not a stopping
rule: the prior cell's ε∞ at epoch 10 of stage A is reported for each seed.

**Pre-flight (1), seed 42, scored (2026-11-15 09:38 IST).** The cold-start epoch finished: no refusal, no non-finite gradient, 2905 s
(sharing the CPU with seed 43's epoch for most of it), training force RMSE 1.37 and validation 0.435 eV/Å, validation
energy 108 meV/atom after one epoch; the prior cell's ε∞ [21.565] (`runs/AuMgO_twosite_smoke_mac_s42/log.txt`). Both seeds' epoch 1 pass the
hurdle that killed the cold starts of 2026-11-07. **Cost:** one epoch on the Mac 44–48 min with two runs sharing the
CPU; step 33's Mac early look took 34 min per epoch alongside another training. The new terms' own share is not
separated. Expected GPU time, from step 33's own logs: 20.9–21.3 min per epoch on GPU host 2 with both seeds on the card,
14.5–14.7 h for stage A and 6.9 h for stage B. For step 34 I estimate 22–30 min per epoch, about 22–30 h for both
stages (an estimate, to be replaced by pre-flight (2b)'s step times). (A first version of this line quoted 10 min per
epoch from memory: that was step 32a, with fewer terms.)

### Step 34's GPU pre-flight, scored (2026-11-15 10:44 IST; GPU host 2 in the new working copy `<gpu-host-workdir>`, synced with `sync_step34.sh`; the GPU was idle before each check and is idle after; outputs in `runs/step34_preflight/`)

- **Sync:** the 11 data files the run reads have the same md5 on both machines; `check_run_config.py` OK for the four
  runs there; the translated configs carry `device: cuda` and paths under `<gpu-host-workdir>` only.
- **(2) CPU against GPU, inference** (step 33's s43 model with the three terms on, six structures): energies agree to
  4e-12 – 1.1e-10 eV per atom; forces to 4e-10 – 1.6e-8 eV/Å on five structures. The doped slab's forces differ by
  1.3e-7 eV/Å, over the 1e-7 I had set (a tolerance chosen, not derived). Ruled out as the new code: with the three
  terms OFF the same slab differs by MORE, 1.5e-7 eV/Å, and in both cases the largest difference sits on the Al
  dopant; a repeat GPU run with the terms on gave 7.5e-8 (GPU sums are not reproducible to the last bits). It is the
  existing doped path at the 1e-7 eV/Å level, far below the labels' accuracy. Pass, read this way.
- **(2b) The training path on CUDA** (`gpu_train_check_step34.sh`; stage A's configs of both seeds together): three
  optimizer steps each, no failure, no non-finite gradient; peak memory 10 252 of 16 311 MiB for the two runs;
  both processes stopped by PID. Seed 43's gradient norms at steps 1 and 2 (167.6, 241.4) are those of its Mac smoke
  run: the GPU reproduces the CPU training path. **Pass.**
- **(2c) Cost per step** (`gpu_steptime_step34.sh`, both seeds together, steps 1–6 of the cold start): with the three
  terms ON 413.8 / 446.0 s to step 6 (s43 / s42), with them OFF 422.2 / 472.5 s. **The new terms cost nothing
  measurable.** The cold start's early steps take about 65 s each (steps 2–6), against 22.5 s on average for step 33,
  which started warm (21 min per epoch). If that pace held, stage A would take about 40 h (61 min per epoch); whether
  the steps get faster as the cold-started ensembles settle is not measured. Estimate: 45–60 h for both stages, not
  the 22–30 h written above (that assumed step 33's pace).

### Step 34 launched (2026-11-15 10:50 IST; decision of 2026-11-15: commit locally and launch step 34)

- **Local commit first:** `6b87ccd` on branch `two-site-training-2026-11-15` (142 files: the SPEC 39 / 40 / 40.7 code,
  tests, tools, data tables, probes, step 34's kit and the records; benchmark data `.xyz`, checkpoints and the
  unrelated untracked files left out; no upstream, not pushed). The 71 code and data files the run reads have the same
  md5 in the commit's tree and in `<gpu-host-workdir>` on GPU host 2.
- **GPU host 2:** `queue_step34.sh` found the GPU free twice and launched stage A at 10:49 IST for seeds 42 and 43
  (`check_run_config.py` OK at launch). Stage B follows from stage A's best checkpoints.
- **The Mac:** `watch_step34.sh` (copies and scores both stages when the queue ends: `runs/watch_step34.log`) and
  `early_look_step34.sh` (the per-epoch ε∞ of the prior cell for stage A, into `runs/early_look_step34.log`, until
  epoch 10 of both seeds).
- Expected: 45–60 h for both stages (pre-flight 2c). The early look at epoch 10 is about 10 h after the launch.

### Step 34 stage A stopped in epoch 2 on both seeds; diagnosed (2026-11-15, scored by 14:10 IST on the Mac; nothing relaunched)

**What happened (S1 FAILS on both seeds).** Seed 43 stopped at 12:21 IST (epoch 2, sub-step 420, train index 347),
seed 42 at 12:29 IST (epoch 2, sub-step 282, train index 404), both on `ConfigurationSearchError`: "the product closure
of one cluster holds more than cap = 512 configurations (63 linked, 529 closed ...)". The queue did not launch stage B.
Both structures are bulk MgO, the 64-atom fc3 cells (`togo_phono3py_PBE_MgO_fc3`, Mg₃₂O₃₂, forces only). Epoch 1 had
finished on both seeds (best checkpoints 11:45 / 11:55 IST; the prior cell's ε∞ 21.4765 / 21.5612). The run's own
outputs are in `runs/step34_failure_2026-11-15/` (logs, both failure dumps). The GPU of GPU host 2 now runs an unrelated
job (not touched).

**The cluster, measured** (`replay_step34_closure.py`, `replay_redox_failure.py` on the dumps; the refusal reproduces at
cap 512 on both). 529 = 1 + 32 + C(32, 2): every placement of zero, one or two O holes on the 32 oxygens, the band
taking the electrons. The band's reference filling (SPEC 38's descent) is N₀ = 2; the 62 linked moves are 60 hops of
a hole between oxygens and 2 refills from the band. At the raised cap the ensemble holds its whole population in the
two-hole configurations (1.000; frame 0.000); the lowest eigenvalue lies 1.6 eV (s42) / 3.6 eV (s43) below the lowest
diagonal entry, the hole hopping |T| up to 0.25 / 0.32 eV (`inspect_step34_cluster.py`, saved in
`runs/step34_failure_2026-11-15/cluster_spectrum.txt`). This is a neutral MgO cell holding two O 2p holes and two
band electrons, which is unphysical (an insulator with a PBE gap of 4.45 eV). The closure is not capped at 529: the
band holds up to 2 × 32 electrons, and N₀ = 3 would give 1 + 32 + 496 + 4960 = 5489 rows.

**Ruled out, with frozen weights (the dumps' own):**
- *The three new terms.* `two_site` off; `two_site` with `spin_pairing` also off; `two_site` with `frame_pair` also
  off; all three off: the same 63 linked, 529 closed on both seeds. (`spin_pairing` and `frame_pair` were not
  switched off alone: `two_site` needs both.) (`frame_pair` moves the cell's energy by 219 eV and leaves the cluster as it is.) Step 33's
  final model of seed 43 with the three terms on puts no hole in either cell and links nothing.
- *The replay.* It reproduces the refusal; `probe_step34_eps_band_factor.py` reproduces the log's epoch-1 ε∞ exactly
  (21.5612 / 21.4765).

**The lever, measured: the band's hopping factor f** (SPEC 20; τ = f V_ssσ; the default `band_initial_factor` is 1,
Harrison). It falls during training: 1.0 at the start, 0.960 / 0.941 after epoch 1, 0.928 / 0.897 in the dumps.

| check (frozen weights) | s42 | s43 |
|---|---|---|
| dump with only f changed (`scan_step34_band_factor.py`) | no hole up to f = 0.920; two holes and the 529-row closure from 0.928 (the dump) and at 0.95, 1.00 | no hole up to 0.880; two holes from 0.900 |
| epoch-1 model, 64-atom bulk cells (`probe_step34_bulk_holes.py`) | N₀ = 0, no hole (f 0.960) | N₀ = 0, no hole (f 0.941) |
| epoch-1 model, prior cell Mg₄O₄, as trained | ε∞ 21.561, N₀ = 2 | ε∞ 21.477, N₀ = 2 |
| same, f = 0.55 (SPEC 20's PBE-mass value) | **ε∞ 3.385, N₀ = 0** | **ε∞ 3.367, N₀ = 0** |
| same, f = 0.640 / 0.732 (step 33's learned values) | 3.385 / 19.94 | 3.367 / 15.56 |

So the hole state sets in sharply above a factor that depends on the rest of the model. The cold start begins at
f = 1, already above it for the prior cell, and by epoch 2 the threshold has moved below the falling factor for the
64-atom cells too.

**The project's own cold starts agree** (first and last ε∞ of the prior cell from each log):

| cold start | f at start | ε∞ after epoch 1 | ε∞ at the last epoch |
|---|---|---|---|
| step 19 (`_bprior`), s42 / s43 | 0.55 | 3.840 / 3.912 | 3.005 / 2.990 (epoch 40) |
| `_pfp`, s42 / s43 | 1 | 22.63 / 36.71 | 2.989 / 2.962 |
| `_pfp_anc`, s42 / s43 | 1 | 20.44 / 20.50 | **19.09** / 3.025 |
| `_pfp_bop`, s42 / s43 | 1 | 23.13 / 36.73 | **19.13** / 2.998 |
| `_pfp_bopw`, s42 / s43 | 1 | 22.43 / 36.73 | 2.988 / 2.971 |
| `_pfp_bopw_e1k`, s42 / s43 | 1 | 37.32 / 37.58 | **19.02** (epoch 21) / **19.55** (epoch 20) |
| `_pfp_fpa_fid`, s42 / s43 | 1 | 37.93 / 37.93 | 3.450 / 3.449 |
| `_pfp_bopw_cbh`, s42 / s43 | 1 | 22.46 / 36.73 | 2.988 / **18.90** (epoch 14) |
| `_pen_ind`, s42 / s43 | 1 | 20.23 / 36.67 | 20.32 (epoch 4) / 2.981 |
| `_pen_ind_spol`, s42 / s43 (**the clean control**: its configs differ from step 19's by `band_initial_factor` only) | 1 | **36.49 / 36.70** | 3.028 / 3.002 |
| `_pen_ind_spol_fpriorq`, s42 / s43 | 1 | 3.056 / 3.420 | |
| earlier band runs (`exband`, `exband_amp`, `_fprior`, `_part`, `_spin`), 10 seeds | 1 | 2.95–3.47 | |
| step 34 stage A, s42 / s43 | 1 | 21.56 / 21.48 | stopped in epoch 2 |

- From `_pen_ind` on (carrier penetration and induction), every cold start at f = 1 except `_fpriorq` began with
  the prior cell holding holes (ε∞ 20–38). The earlier band runs at f = 1 began at 2.95–3.47.
- So f = 1 is not enough by itself: the hole state needs the rest of the model too, as the frozen threshold above
  depends on it.
- The clean control isolates the factor: same config, f = 1 against 0.55, gives 36.49 / 36.70 against 3.84 / 3.91.
- Seeds near 19 at their last epoch: `_pfp_anc` s42, `_pfp_bop` s42, `_pfp_bopw_e1k` s42 and s43, `_pfp_bopw_cbh`
  s43; `_pen_ind` s42 was at 20.32 when it ended at epoch 4.
- The only cold start at f = 0.55 began at 3.8–3.9.
- **This corrects the "known risk" written in the launch kit above** ("three, all seed 42, stayed at 19.0–19.1"):
  `_pfp_bopw_e1k` s43 was also at 19.55 at its last epoch (20), and `_pfp_fpa_fid` ended at 3.45, not at 2.96–3.03.
- The cause of that risk is the starting band factor, now measured. Step 19's record had dropped the anchor ("not
  adopted on its own for T5"): its T5 was worse than step 17's at f = 1, which the record attributed partly to
  compensations the over-wide band allowed.
- Every cold start since then ran in the global configuration space, with no product closure. Step 34 is the first
  cold start in the cluster space (SPEC 36 / 36.7), where the two-hole state is refused; step 33 used the cluster
  space but started warm (step 31's checkpoints carry f = 0.834 / 0.667; step 33 ended at 0.732 / 0.640).

**Not examined:** the gradient spikes on train 63 (|g| 1782 on s43, 124 on s42) and 268 (715, s43), Au₂ slabs, and
on s43's per-structure gradients 578 on [247] and 469 on [223], in epoch 2.

**Proposed amendment (waits for a decision; nothing changed or launched):** relaunch stage A from scratch with one
change, `redox.band_initial_factor: 0.55`. It is SPEC 20's anchor from PBE's conduction-band mass of MgO, a general
rule (the s band's curvature from the material's own first-principles mass), not a fitted value. Criteria S1–S8
unchanged. Raising the cap is not proposed: it would admit an unphysical state whose closure grows as C(32, N₀).

## Amendment: step 34b, step 34 relaunched with the band's PBE-mass anchor (written 2026-11-15 14:20 IST, before the launch; decision of 2026-11-15: branch and relaunch with band_initial_factor 0.55)

- **Branch:** `step34-band-anchor-2026-11-15`, local, from `two-site-training-2026-11-15`. The run's code is commit
  `6b87ccd`'s: `git diff 6b87ccd` shows no change in `trace_redox`, `trace_electrostatics`, `flashace`, `train.py`
  or `transformers_ace`, and nothing is untracked there. The training data (`AuMgO_bulkaugw_*.xyz`) are untracked and
  in no commit; they are identified by md5, which `sync_step34b.sh` checks on both machines for the 11 files the run
  reads. The new probes and the records are uncommitted.
- **One change against step 34:** `electrostatics.redox.band_initial_factor: 0.55`, in all four configs.
  `derive_step34b.py` prints the key-by-key difference: that key only, besides the run names. Stage B starts warm
  from stage A, so the key changes nothing there; it is set so the four configs differ from step 34's by one key.
- **Runs:** `AuMgO_twosite_bf055_bondlaw_r5_s{42,43}` (stage A, cold) and
  `AuMgO_twosite_bf055_e1kw_bondlaw_r5_s{42,43}` (stage B). New names, so step 34's stopped runs stay as they are;
  `check_run_config.py` passes on all five, including the Mac smoke.
- **Kit:** `queue_step34b.sh`, `sync_step34b.sh`, `watch_step34b.sh`, `early_look_step34b.sh`, generated from step 34's
  scripts by name replacement (diffs checked). The sync also refuses while a step 34 process runs on GPU host 2. Same
  working copy `<gpu-host-workdir>`; `runs/` is not synced, so step 34's run directories there are not touched.
- **Criteria:** S1–S8 and PASS exactly as pre-registered for step 34. It is a different run, so step 34's S1 FAIL
  stays recorded.
- **Predictions (hypotheses):**
  - the prior cell's ε∞ after epoch 1 is 3–5, as step 19's 3.84 / 3.91 at f = 0.55 (the frozen probe of step 34's
    epoch-1 weights gave 3.39 / 3.37);
  - no closure refusal on the bulk cells;
  - S1 risk: the factor is learned and may drift up. Step 19's ended at 0.485 / 0.572; the frozen thresholds were
    0.92 / 0.88 at step 34's epoch-2 weights.
- **Early look (not a stopping rule):** the prior cell's ε∞ at epoch 10 of stage A, each seed
  (`early_look_step34b.sh`). The learned f is read by hand from the run's checkpoint at that time; the log does not
  print it.
- **Corrected after an independent check of this entry and the one before it (2026-11-15 14:30 IST; no blocker
  in the kit; the replaced statements):**
  - "singly and all together" (the three terms were never off singly);
  - "Every cold start at f = 1 began with the prior cell holding holes (ε∞ 20–38). Four of the twelve seeds stayed
    near 19 to their last epoch" (a selection: the earlier band runs and `_fpriorq` began near 3, and the table
    lacked `_pen_ind`, `_pen_ind_spol` and `_pfp_bopw_cbh`);
  - "(f 0.64 / 0.73 from step 31)" (those are step 33's final values);
  - "The run's code and data are those of commit `6b87ccd`" (the data are untracked);
  - "the learned f at epoch 10" as part of the early look (the script cannot read it);
  - the gradient spikes, two more added;
  - the eigenvalue and |T| numbers had no saved output: now saved.
- **Mac smoke:** `runs/AuMgO_twosite_bf055_smoke_mac_s43`, stage A of seed 43, one epoch, started 14:14 IST.
  Checks no refusal and the epoch-1 ε∞. It does not gate the GPU queue: the GPU of GPU host 2 runs an unrelated job
  now, and the queue starts only after it ends. If the smoke shows a defect before then, the queue is stopped by PID.

**Step 34b's Mac smoke, scored (2026-11-15 14:42 IST; stage A of seed 43, one epoch, `runs/AuMgO_twosite_bf055_smoke_mac_s43`).**
Finished, no refusal, no non-finite gradient. Against step 34's smoke (same seed, same epoch, f = 1):

| after epoch 1 | step 34b (f starts at 0.55) | step 34 (f starts at 1) |
|---|---|---|
| prior cell's ε∞ (PBE 3.235) | **3.389** | 21.49 |
| learned f | 0.503 | 0.941 |
| validation force RMSE (eV/Å) | 0.300 | 0.404 |
| validation energy RMSE (meV/atom) | 90.2 | 203.1 |
| validation loss | 0.165 | 0.530 |

- The first prediction holds: ε∞ is inside 3–5.
- The factor moved down, away from the frozen thresholds (0.88–0.93), not up.
- Epoch time: 1628 s alone on the Mac. Step 34's smoke took 2657 s while sharing the CPU, so the two are not
  comparable.
- This is one epoch: whether the bulk cells stay free of holes is what the GPU run tests.

**Step 34b running (2026-11-15).** The queue found the GPU free and launched stage A at 14:47 IST (seeds 42 and 43;
`check_run_config.py` OK). Read at 15:30 IST:

| | seed 42 | seed 43 |
|---|---|---|
| epoch time (both seeds on the card) | 1244 s, 1196 s | 1251 s, 1203 s |
| prior cell's ε∞ after epochs 1, 2 | 3.404, 3.336 | 3.389, 3.323 |
| learned f, best checkpoint (epoch 1) | 0.506 | 0.503 |
| validation force RMSE after epochs 1, 2 (eV/Å) | 0.378, 0.257 | 0.300, 0.251 |
| failures | none | none |

- Both seeds are past epoch 2, where step 34 stopped.
- Seed 43's epoch 1 on the GPU reproduces the Mac smoke: the same training loss, 27.1135, and the same ε∞, 3.3889.
- Each epoch takes about 20 min, against 61 min for step 34's cold start, which carried the two-hole ensembles.
- At this pace stage A takes about 13 h and stage B about 7 h, both seeds included. The 45–60 h estimate was made
  at the old pace.

**Step 34b's phonon early look (Mac; decision of 2026-11-15: run it locally).** Not a criterion: S4 is
scored by `watch_step34b.sh` on stage B's final models. `watch_step34b_phonons.sh` (started 15:35 IST) copies each
run's best checkpoint from GPU host 2 every 10 min. Each new checkpoint is scored against PBE with `score_phonons.py`,
which takes under a minute on the Mac, and the result goes to `runs/step34b_phonons_early.log`. Stage A's epoch-1
checkpoints, scored by hand at 15:34:

| epoch 1 of stage A | seed 42 | seed 43 | PBE |
|---|---|---|---|
| passes | 3 / 7 | 4 / 7 | |
| Z*(Mg) | 1.945 | 1.948 | 1.9923 |
| ε∞ | 3.404 | 3.389 | 3.235 |
| TO (cm⁻¹) | 319.3 (−14.5 %) | 398.5 (+6.8 %) | 373.3 |
| LO (cm⁻¹) | 629.3 | 674.8 | 681.3 |
| path RMS without / with NAC (cm⁻¹) | 156.3 / 151.0 | 115.9 / 112.2 | ≤ 15 / ≤ 20 |

- P0, the ensemble in the frame on every displaced supercell (population 1.000000), passes on both seeds. This is
  the sector that held two holes in step 34.
- The tool names its output folder after the checkpoint's directory. The first outputs landed in
  `benchmarks_phonons/runs/s42` and `s43`, both created at 15:34, so nothing was overwritten. They were moved to
  `AuMgO_twosite_bf055_bondlaw_r5_s{42,43}_ep1`, and the watcher uses distinctive names.

**Step 34b at epoch 10 of stage A (read 2026-11-15 18:20 IST).** Both seeds have finished epoch 10. There are no
failures, and the early look (`early_look_step34b.sh`) stopped as designed. An unrelated job left the card
between epochs 5 and 6, and epochs 6–10 took about 1155 s each. From the remote `log.txt`:

| | seed 42 | seed 43 |
|---|---|---|
| validation loss, epochs 3 → 10 | 0.048 → **0.0255** (best: epoch 10) | 0.056 → 0.0529 (best: epoch 9, 0.0212) |
| validation force RMSE, epochs 3 → 10 (eV/Å) | 0.215 → 0.146 | 0.221 → 0.130 |
| validation E_free, epoch 10 (meV/atom) | 3.78 | 2.52 |
| prior cell's ε∞ (eval), epochs 6–10 | 3.352, 3.351, 3.363, 3.368, 3.377 | 3.417, 3.413, 3.409, 3.396, 3.380 |

Phonon early look on each best checkpoint (`runs/step34b_phonons_early.log`). f is the band factor and g the crystal
factor, both read from the checkpoint:

| checkpoint | f | g | Z*(Mg) | ε∞ | TO vs PBE | LO (cm⁻¹) | RMS no NAC / NAC (cm⁻¹) | passes |
|---|---|---|---|---|---|---|---|---|
| s42 ep3 | 0.419 | 0.934 | 1.959 | 3.285 | +8.3 % | 687.6 | 43.2 / 34.8 | 4 / 7 |
| s42 ep8 | 0.369 | 0.962 | 1.951 | 3.363 | +7.0 % | 677.5 | 32.5 / 18.3 | 5 / 7 |
| s42 ep9 | 0.364 | 0.964 | 1.951 | 3.368 | +4.6 % | 671.9 | 33.9 / 21.1 | 4 / 7 |
| s42 ep10 | 0.365 | 0.967 | 1.950 | 3.377 | +7.5 % | 677.4 | 31.3 / 16.7 | 5 / 7 |
| s43 ep3 | 0.428 | 0.951 | 1.957 | 3.332 | −2.1 % | 661.7 | 33.3 / 19.1 | 6 / 7 |
| s43 ep5 | 0.389 | 0.974 | 1.948 | 3.396 | +4.4 % | 669.1 | 30.1 / 14.8 | 5 / 7 |
| s43 ep6 | 0.381 | 0.981 | 1.945 | 3.417 | +10.3 % | 680.2 | 30.8 / 19.9 | 5 / 7 |
| s43 ep7 | 0.374 | 0.979 | 1.946 | 3.413 | +4.3 % | 667.3 | 29.7 / 13.8 | 5 / 7 |
| s43 ep8 | 0.370 | 0.978 | 1.946 | 3.409 | +6.7 % | 672.8 | 30.2 / 16.3 | 5 / 7 |
| s43 ep9 | 0.371 | 0.974 | 1.947 | 3.396 | +13.6 % | 689.4 | 34.1 / 25.8 | 4 / 7 |

- **What improves:** validation forces fall every epoch, by 0.069 eV/Å (s42) and 0.091 eV/Å (s43) since epoch 3.
- **What does not:**
  - The phonons have stayed at 4–5 / 7 since epoch 5.
  - The path RMS without NAC sits at 30–34 cm⁻¹ (P5 needs 15).
  - TO is stiff on every checkpoint after s43's epoch 3, by +4 to +14 % (P3 needs 3 %).
  - P0, P1, P2 and P4 pass on every checkpoint, and none has an imaginary mode.
  - Better validation loss has not meant better phonons: s43's epoch 9, its best loss, has its worst TO.
- **S5, read as a trend, not scored** (S5 is scored on stage B's final models):
  - ε∞ sits 0.14 above PBE on both seeds at epoch 10; S5 needs it within 0.05.
  - g is 0.967 / 0.974, above S5's upper bound of 0.95.
- **Band factor:** it fell from 0.55 to 0.36–0.37 and has been nearly flat since epoch 8.
- **Not examined (hypothesis only):** why TO stiffens while LO stays on PBE. TO sets the short-range force constant at
  Γ, and LO adds the macroscopic field on top. No probe has separated the two.
- Figure: `benchmarks_phonons/step34b_A_s42ep10_s43ep9_nac_vs_pbe.png`.
- Expected stage A end: about 04:00 IST on 2026-11-16 at the present pace (30 epochs × 1155 s).

**Step 34b at epoch 15 of stage A (read 2026-11-15 19:51 IST).** No failures.
- **Best validation loss:** epoch 13 on both seeds, 0.0228 (s42) and 0.0159 (s43). Validation force RMSE at epoch 15:
  0.131 / 0.119 eV/Å; s43 reached 0.109 at epoch 14.
- **Prior cell's ε∞ (eval), epochs 11–15:**
  - s42: 3.374, 3.369, 3.372, 3.370, 3.367;
  - s43: 3.370, 3.359, 3.346, 3.342, 3.340.
  - s43 is coming back toward PBE (3.235); s42 is flat.
- **Phonon early look:**

| checkpoint | f | g | Z*(Mg) | ε∞ | TO vs PBE | LO (cm⁻¹) | RMS no NAC / NAC (cm⁻¹) | passes |
|---|---|---|---|---|---|---|---|---|
| s43 ep11 | 0.378 | 0.964 | 1.951 | 3.370 | +6.8 % | 676.6 | 29.1 / 14.6 | 5 / 7 |
| s42 ep13 | 0.375 | 0.965 | 1.950 | 3.372 | +6.5 % | 675.5 | 30.7 / 14.4 | 5 / 7 |
| s43 ep13 | 0.384 | 0.956 | 1.953 | 3.346 | **+0.5 %** | 665.2 | 31.6 / 13.9 | **6 / 7** |

- s43's epoch 13 fails only P5, the RMS without NAC, at 31.6 cm⁻¹ against a limit of 15.
- In the figure, its largest visible miss is the LO branch near X: about 490 against PBE's 528 cm⁻¹. Its acoustic
  branches lie on PBE.
- Figure: `benchmarks_phonons/step34b_A_s42ep13_s43ep13_nac_vs_pbe.png`.
- An unrelated job (1.8 GB) shares the card again since about 19:40. Epoch 15 took
  1242 / 1282 s.

**Step 34b at epoch 21 of stage A (read 2026-11-15 22:00 IST).** No failures. An unrelated job left the
card again, and epochs take about 1170 s.
- **Best validation loss:** 0.0147 (s42, epoch 21) and 0.0132 (s43, epoch 19). Validation force RMSE at epoch 21:
  0.117 / 0.109 eV/Å; s43 reached 0.098 at epoch 20.
- **Prior cell's ε∞ (eval), epochs 14–21:**
  - s42: 3.363 → 3.357, flat;
  - s43: 3.342 → 3.313, still falling toward PBE's 3.235.
- **Phonon early look (P5 against PBE carries the 2×2×2 against 4×4×4 supercell mismatch; benchmarks_phonons README,
  "P5 against PBE compares two different supercells"):**

| checkpoint | f | g | Z*(Mg) | ε∞ | TO vs PBE | LO (cm⁻¹) | RMS no NAC / NAC (cm⁻¹) | passes |
|---|---|---|---|---|---|---|---|---|
| s42 ep16 | 0.388 | 0.962 | 1.950 | 3.363 | +4.1 % | 671.1 | 29.2 / 11.6 | 5 / 7 |
| s43 ep16 | 0.391 | 0.951 | 1.955 | 3.333 | −4.0 % | 657.3 | 38.1 / 22.9 | 3 / 7 |
| s42 ep17 | 0.394 | 0.962 | 1.950 | 3.363 | +6.3 % | 675.8 | 29.6 / 13.8 | 5 / 7 |
| s43 ep19 | 0.403 | **0.946** | 1.957 | 3.317 | +7.3 % | 682.8 | 29.4 / 14.4 | 5 / 7 |
| s42 ep20 | 0.409 | 0.960 | 1.950 | 3.359 | +0.7 % | 663.9 | 29.5 / **8.0** | 6 / 7 |
| s42 ep21 | 0.416 | 0.960 | 1.950 | 3.357 | +2.7 % | 668.3 | 30.4 / 10.7 | 6 / 7 |

- Three of the last five best checkpoints show the S4 pattern (P0, P1, P2, P4 pass, and 6 / 7): s43 ep13, s42 ep20 and
  s42 ep21. In each, only P5 fails.
- Not stable yet: TO swings between −4 % and +7 % from one best checkpoint to the next.
- s43's crystal factor entered S5's window (0.946 at epoch 19); s42's sits at 0.960.
- The band factor turned around: after its low of 0.364 at epoch 9 it has risen to 0.403–0.416.
- Figure: `benchmarks_phonons/step34b_A_s42ep21_s43ep19_nac_vs_pbe.png`.

**P5 like-for-like (2026-11-16 01:20 IST; benchmarks_phonons README, "Pre-registered: P5 like-for-like on Togo's
512-atom supercell" and "Scored").** The model's force constants were computed on PBE's own 4×4×4 conventional
supercell (512 atoms). The consistency check passed: Γ, X and L agree with 2×2×2 within 0.004 cm⁻¹.
**[Corrected after an independent review, 2026-11-16 ~11:05 IST; the text above is kept as written.]** The values are 0.0042, 0.0047 and 0.0011, so the bound is 0.005.
- **P5 (path RMS without NAC):** 30.4 → 11.5 (s42 ep21, pre-registered), 29.4 → 8.2 (s42 ep30) and 29.7 → 9.0
  (s43 ep29). All three checkpoints score 7 / 7 on 4×4×4.
- **P5 > 15 against PBE was the supercell mismatch** for these three step 34b checkpoints. For earlier steps this is a
  hypothesis: they use the same 2×2×2 protocol but were not rerun.
- **P6 on 4×4×4:** 11.8 / 8.5 / 9.3. S4 stays scored as written, on 2×2×2.

**Step 34b's final scoring reports both supercells (written 2026-11-16 08:40 IST, while stage B runs; decision of
2026-11-16: report both in the final scoring).**
- **Unchanged:** S1–S8. S4 and S5 are scored as pre-registered: P0–P6 with the force constants on the 2×2×2
  conventional supercell (64 atoms), by `watch_step34b.sh` on stage B's final models.
- **Recorded alongside, not a criterion:**
  - P0–P6 with the force constants on 4×4×4 (512 atoms, PBE's own fc2 supercell);
  - the check C1: Γ, X and L agree with 2×2×2 within 1 cm⁻¹;
  - the same four final checkpoints (stages A and B, both seeds) and the same `model.pt` files as the 2×2×2 scoring
    (md5 checked before and after).
- **How:** `score_step34b_sc4.sh` (new; on the Mac) waits until `watch_step34b.sh` has ended. If the
  final scoring did not complete, it does not run. It writes:
  - `runs/step34b_sc4.log`;
  - `runs/<run>/scores/P_phonons_pbe_sc4.txt` and `P_phonons_sc2_vs_sc4.txt`;
  - the band yaml files in `benchmarks_phonons/runs/<run>_sc4/`;
  - the figures `benchmarks_phonons/step34b_{A,B}_final_*_{nonac_sc2_vs_sc4,nac_sc4_vs_pbe}.png`, model against DFT only.
- **Tested** at 08:31–08:34 IST on scratch copies of stage A's epoch-39 best checkpoints (2 min 11 s for both seeds;
  peak memory 6.6 and 6.9 GB). Stage A has ended on both seeds ("Training finished"). Its final `model.pt` files on
  GPU host 2 have the same md5 as these copies (3445a62c…, ae3c8743…), so these are stage A's final numbers. The real
  run recomputes them into the repository.

| stage A final | Z*(Mg) | ε∞ | TO vs PBE | LO (cm⁻¹) | P5 no NAC, 2×2×2 → 4×4×4 | P6 NAC, 2×2×2 → 4×4×4 | passes, 2×2×2 / 4×4×4 | C1 max (cm⁻¹) |
|---|---|---|---|---|---|---|---|---|
| s42 ep39 | 1.950 | 3.334 | +3.1 % | 670.7 | 29.3 → 8.4 | 7.5 → 8.6 | 5 / 7, 6 / 7 | 0.005 |
| s43 ep39 | 1.960 | 3.303 | −1.9 % | 664.9 | 30.6 → 9.8 | 8.1 → 10.1 | 6 / 7, **7 / 7** | 0.001 |

- **Not a verdict:** S4 and S5 are read on stage B's models.
- **Stage A, s42:** fails P3 by 0.1 % on both supercells (TO 384.7 against PBE's 373.3; the limit is 3 %).
  **[Corrected after an independent review, 2026-11-16 ~11:05 IST; the text above is kept as written.]** It is +3.05 %, 0.05 percentage points over the limit.
- **Stage A, ε∞:** 0.099 (s42) and 0.068 (s43) above PBE's 3.235. S5 asks for 0.05 or less.

**P5 is pre-registered on PBE's own 512-atom supercell for every step after 34b (2026-11-16 10:31 IST; decision of
2026-11-16: pre-register P5 on 512 atoms for future steps; full text in the benchmarks_phonons README, "Pre-registered:
P5 on PBE's own 512-atom supercell for every step after 34b").**
- Every phonon criterion against PBE of a step registered after this entry is scored with
  `score_phonons.py MODEL --p5-supercell 4`.
  - P5 is read with the model's force constants on 4×4×4 (512 atoms). The threshold is unchanged at ≤ 15.
  - P5 is FAIL unless C1 holds (the two supercells agree within 1 cm⁻¹ where both are exact).
  - P0 must hold on both supercells.
  - P1–P4 and P6 are unchanged, on 2×2×2.
- Step 34b and every earlier step keep their written protocol; the scorer's default stays theirs.
- Tests: `tests/test_score_phonons_protocol.py`: 6 at first; 21 after the review (benchmarks_phonons README,
  "Amendment after an independent review").
  - The flag is accepted only for the reference's own force-constant supercell: PBE 4, PBEsol 2 (already the default,
    so refused).
  - The no-flag output is byte-identical to HEAD's.

### Step 34b scored: FAILS as pre-registered (S2, S5 and S8 on both seeds; 2026-11-16 14:15 IST)

Both stages finished on both seeds:
- stage A ended at about 04:13 IST;
- stage B ended at 10:58 IST (`queue_step34b.log`: "step 34b trained").

`watch_step34b.sh` copied the four `model.pt` files at 11:08–11:09 IST, checked that the data files match GPU host 2's,
and scored them by 14:09 IST. `score_step34b_sc4.sh` added the 512-atom phonons by 14:15 IST, on the same `model.pt`
files (md5 unchanged). The models are the best checkpoints by validation loss: stage B s42 epoch 15, s43 epoch 18;
stage A epoch 39 on both seeds. Outputs are in `runs/AuMgO_twosite_bf055_*/scores/` and `runs/step34b_sc4.log`.

| criterion | target | s42 | s43 |
|---|---|---|---|
| **S1** | both stages finish | pass | pass |
| **S2** | wet_u +928.9 ± 100; wet_d −2.7 ± 150; T5 within ±15 % of −931.6 | **FAIL**: +610.4 (−318.5); +92.3 pass; −518.1 (55.6 %) | **FAIL**: +729.7 (−199.2); −27.7 pass; −757.4 (81.3 %) |
| **S3** | 4G test force RMSE ≤ 0.10 undoped, ≤ 0.13 doped (eV/Å) | pass: 0.0845 / 0.1242 | pass: 0.0616 / 0.1112 |
| **S4** | phonons (2×2×2): P0, P1, P2, P4 pass and ≥ 6 / 7 | pass: 6 / 7 (P5 29.5) | pass: 6 / 7 (P5 29.9) |
| **S5** | g within 0.85–0.95; the prior cell's ε∞ within 0.05 of 3.235 | **FAIL**: g 0.9515, ε∞ 3.333 (+0.098) | **FAIL**: g 0.9396 pass, ε∞ 3.300 (+0.065) |
| **S7** | Au–Au bond force on test 167: repulsive and monotone over 2.00–2.40 Å, within ±30 % of +15.25 at 2.089 Å | pass: +14.25 (−7 %) | pass: +15.82 (+4 %) |
| **S8** | undoped test force RMSE without the five dimers ≤ 0.060 eV/Å | **FAIL**: 0.0844 | **FAIL**: 0.0614 |

**Against step 33** (the same criteria):
- **S3 now passes.** Step 33 had 0.193 / 0.104 (s42) and 0.221 / 0.156 (s43). The five dimers below 2.18 Å now carry
  2.7 % (s42) and 3.1 % (s43) of the undoped squared force error, against 92.2 % and 94.2 % in step 33.
- **S7 now passes.** Step 33 had −17.0 eV/Å at 2.089 Å. **[Added after review, 2026-11-16:]** that value is s43's;
  the record gives no s42 value.
- **The prediction for S7 held. The one for S3 undoped held in S3 itself, but not in S8.**
- **S8, the rest of the undoped forces, got worse:** step 33 had 0.0544 / 0.0540; step 34b has 0.0844 / 0.0614.
- **S2's wet_u is further from DFT on both seeds:** step 33 had +768.4 / +789.9; step 34b has +610.4 / +729.7.
- **S5 failed in both steps.**
- Not attributed: why S8 and wet_u got worse. Candidates, none tested:
  - the cold start (step 33's stage A was not one);
  - SPEC 39's change to the frozen Au–O interaction;
  - seed spread, given the 0.023 eV/Å difference in S8 between the two seeds.

**[Note added after review, 2026-11-16]** S5's ε∞ is read, as the scorer does by design, at the training's last
evaluation (epoch 20). At the best checkpoints it is 3.3337 (s42, epoch 15) and 3.3001 (s43, epoch 18). The verdict
is unchanged.

**Recorded alongside (not criteria):**
- **Phonons on PBE's own 4×4×4 supercell (512 atoms):**
  - stage B: 7 / 7 on both seeds (P5 8.1 / 8.6, P6 8.5 / 8.8, C1 0.005 / 0.001 cm⁻¹);
  - stage A: s42 6 / 7 (P3 +3.05 %), s43 7 / 7;
  - figures: `benchmarks_phonons/step34b_{A,B}_final_*_{nonac_sc2_vs_sc4,nac_sc4_vs_pbe}.png`.
- **S6, Table II test energies:** RMSE 1.353 / 0.989 meV/atom (s42, undoped / doped) and 1.142 / 0.671 (s43).
- **S6, the wet_u error is a class offset.** On the test set, the undoped flat minus upright residual is
  −273.5 ± 8.1 meV (s42) and −95.3 ± 15.3 meV (s43). For the doped set it is +165.4 ± 7.9 and −46.1 ± 10.6 meV
  (`H1_class_residuals.txt`).
- **S6, gas-phase Au₂ at zero shift, as trained (2.30–2.80 Å):**
  - s42: EA +0.20 to +0.43 eV, IP +0.56 to +0.59 eV;
  - s43: EA −0.03 to 0.00 eV, IP +0.07 to +0.08 eV;
  - with the fixed terms only (identical on both seeds): EA −0.68 to −0.19, IP +0.65 to +0.74.
  - **[Corrected 2026-11-16 ~16:00 IST, found by an independent read; the line above is kept as written.]**
    - That line quotes the file's "SPEC 39 + 29 + the hop of 40, no relaxation; fixed terms only" block.
    - The full construction with the fixed terms only ("as trained; learned parts OFF") gives EA −0.159 to
      −0.135 eV and IP −0.040 to +0.012 eV, on both seeds. That matches SPEC 40.7's frozen check.
    - **[Corrected again 2026-11-16 ~17:50 IST, found by the methods report's checker; the two lines above are kept as
      written.]** Those numbers are the ERROR columns, model minus reference, not EA and IP.
      - The values: EA 1.715–2.211 eV and IP 8.922–9.390 eV over R = 2.30–2.80 Å.
      - The errors: EA −0.159 to −0.135 eV and IP −0.040 to +0.012 eV.
      - The same holds for the as-trained and fixed-terms lines above them: every signed number there is an error.
- **S6, relaxed non-wetting Au–O (not in the line above):** doped 2.297 / 2.430 Å and undoped 2.195 / 2.165 Å
  (s42 / s43), against DFT 2.332 / 2.190 and 4G 2.342 / 2.177. s43's doped value is 0.098 Å long.
- **S6, the transfer set:** q(Au₂) shift doped − undoped −1.000 (s42) and −0.997 (s43) e, against Hirshfeld −0.266.
  The per-structure r with Hirshfeld is 0.848 / 0.849.
- **S6, couplings:** the largest |τ|/B on the bulk MgO prior cell is 0.012 / 0.008. No edge is above B.
- **S6, cost:** minutes per epoch are about 20 in stage B (s42 20.2, s43 20.0); GPU memory in stage B was
  3.4 GB per run at each check (`nvidia-smi`), not a recorded peak.
- **NOT DONE:** Table II on the 2×2 supercell (S6).

**Verdict:** step 34b FAILS. PASS needed S1–S5, S7 and S8 on both seeds; S2, S5 and S8 fail on both. Nothing is
running. The next step awaits a decision. A diagnosis of S8 and wet_u (cold start against the new frozen terms)
comes before any new run.
