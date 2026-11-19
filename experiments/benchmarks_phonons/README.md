# Phonon benchmark against DFT (bulk MgO), a validation line separate from `benchmarks_4g` (opened 2026-11-06)

> **About this file.** This is the dated lab notebook of the MgO phonon benchmark: every pre-registered step with its
> criteria, runs and scores, kept as it was written, failures, cancelled runs and corrections included (a correction is
> added beside the entry it corrects, not in place of it). The summary of results is `docs/electrostatics/BENCHMARKS.md`,
> the methods (the equations and their SPEC sections) are `docs/electrostatics/METHODS.md`, the reproduction guide is
> `docs/electrostatics/REPRODUCE.md`, and the released checkpoints (GitHub Release assets, with sha256) are listed in
> `docs/electrostatics/MODELS.md`. The 4G-HDNNP / PFN benchmark has its own notebook,
> `experiments/benchmarks_4g/README.md` ("benchmarks_4g README" below).
> For the public release, private details were removed: machine names (now "the Mac", the local CPU machine, and
> "GPU host 1" / "GPU host 2", the two remote GPU machines, numbered as in the benchmarks_4g README), home-directory and
> remote paths (now `<gpu-host-workdir>`, `$TRACE_PHONON_RAW` for the raw downloads and `$TRACE_FP_CACHE` for the
> first-principles cache), process IDs, references to internal project notes and policy numbers (now naming the
> policy), and quotes of the project lead (now `decision of <date>`). No measured number was changed. Unmarked times
> are local time (IST, UTC+5:30); three entries of 2026-11-07 are marked GMT.
>
> - **Data:** `data/README.md` gives the sources, the licences, the script that rebuilds every file and its sha256.
>   Third-party data are not redistributed. The four DFT reference bands in `reference/` are this benchmark's only
>   third-party-derived files in the repository (A. Togo, NIMS Materials Data Repository, CC BY 4.0; attribution in
>   `reference/README.md`); `make_reference_bands.sh` rebuilds them from the downloads and checks them byte for byte.
> - **Run directories.** `runs/...` and `benchmarks_4g/runs/...` hold the logs, scores, band files and checkpoints of
>   the runs; they are not in git.
> - **Not in the release:** the job-queue scripts named below (`gpu_queue_28b.sh`, `gpu_queue_bulkaugw.sh`,
>   `gpu_queue_bulkonly.sh`, `mac_queue_pbesol.sh`, `mac_queue_pbesol2.sh`, `mac_queue_pbesol_ext.sh`,
>   `mac_queue_pbesol_ions.sh`), which chained the training and scoring commands on the lab machines (waiting for a
>   free GPU or for the previous job, and copying files to the GPU host), and the intermediate figures marked "not
>   released". The scorer, the probes, the plot scripts and the final figures are in this directory; the watcher and
>   scoring scripts named below under `benchmarks_4g/` are in that directory.
> - **Branches** named in the record (`bulkaug-cancelled-2026-11-07` and the others) are local git branches made before
>   each decision under the project's never-delete policy; they are not published.

**Purpose (2026-11-06):** benchmark phonons against DFT with the electrostatics code — the long-range response the
slab benchmarks never test. The 4G/PFN line (`benchmarks_4g`) stays like-for-like on its own data; this line may add bulk
training data and is scored on both (no regression on T1–T6, plus P0–P6 below).

## Reference (provenance)
- **PBE, functional-matched:** A. Togo's phono3py dataset for rocksalt MgO (NIMS MDR
  https://mdr.nims.go.jp/datasets/1fd84240-abc0-4d8f-b4d7-1db304d6166b, CC BY 4.0; the 103-compound set of PRB 91,
  094306 (2015) and PRL 115, 205901 (2015)): VASP PBE, PAW, a = 4.25566 Å, fc2 from two 0.03 Å displacements on a 4×4×4
  supercell, Born charges Z* = ±1.9923 and ε∞ = 3.235 (LEPSILON). Harmonic band structure recomputed with phonopy 2.17.2
  along Γ–X–K–Γ–L (51 points per segment), with and without NAC: `reference/MgO_PBE_togo_band_{nac,nonac}.yaml`
  (`benchmarks_4g/phonons_mgo/compare_phono3py_pbe.py`). PBE: TO 373.3, LO 681.3 cm⁻¹; X 281, 281, 406, 425, 425, 528;
  L 262, 262, 344, 344, 529, 550.
- **PBEsol (for the record):** phonondb mp-1265 (https://mdr.nims.go.jp/concern/datasets/bk128j14x), a = 4.21726 Å, Z*
  1.979, ε∞ 3.242, TO 391.0, LO 693.9: `reference/MgO_PBEsol_phonondb_band_nac.yaml`.
- Experiment at Γ: TO 401, LO 718 cm⁻¹, ε∞ 2.95, ε0 9.8 (citations in `data/README.md`).
- Raw downloads (checksums verified; `data/README.md`) in `$TRACE_PHONON_RAW/{phono3py_pbe_mgo,phonondb_mp-1265}/`. `data/*.extxyz` is
  gitignored (`*.extxyz`); regenerate it with `python -I extract_phono3py_cells.py <phono3py_params.yaml> data/MgO_PBE_togo_displaced.extxyz`.

## Protocol
`python experiments/benchmarks_phonons/score_phonons.py runs/RUN/model.pt` — the model at the PBE lattice constant,
Z* and ε∞ from the bound sector on the 64-atom cell (`tools/electrostatics/mgo_phonons.py`: frame charge + ∂P_bound/∂r),
finite displacements 0.01 Å on the 2×2×2 conventional supercell (3×3×3 changes the model's bands by 6.3 cm⁻¹ RMS, measured
on step 20 s43), NAC from the model's own Z* and ε∞, q → 0 at |q| = 10⁻³. The ensemble guard records the frame population
of every displaced supercell: forces are unusable below 1 − 10⁻⁴ (step 20 s42 at this lattice constant: population 0.976,
acoustic RMS 781 cm⁻¹).

## Pre-registered criteria (written 2026-11-06, before any run on this line)
| | criterion | step 20 s42 (at 4.265 Å; at 4.2557 it fails P0) | step 20 s43 |
|---|---|---|---|
| P0 | ensemble in the frame on every displaced supercell (population ≥ 1 − 1e-4) | FAIL at 4.2557 (0.976) | pass |
| P1 | Z*(Mg) within 1.90–2.10 | 2.43 FAIL | 2.54 FAIL |
| P2 | ε∞ within 2.9–3.5 | 3.06 pass | 3.05 pass |
| P3 | TO within 3 % of 373.3 | 389.8 (+4.4 %) FAIL | 407.0 (+9 %) FAIL |
| P4 | LO within 660–700 | 817 FAIL | 850 FAIL |
| P5 | path RMS without NAC ≤ 15 cm⁻¹ | — (contaminated at 4.2557; 14.3 vs PBEsol at 4.265) | 40.4 FAIL |
| P6 | path RMS with NAC ≤ 20 cm⁻¹ | 30.3 (at 4.265) FAIL | 48.5 FAIL |

P1, P2, P4 are backbone-independent (the electrostatic sector alone); P3, P5 are the short-range force constants; P6 both.

## Planned steps (one change per run; each names the criterion it moves)
1. **Physics, no training (P1, P4):** ruled out on 2026-11-07 (frozen, pre-registered): the ensemble route, the pair's
   short-range terms, the dipole widths and the species split of α (theory note
   `docs/ELECTROSTATICS_THEORY_2026-10-18_overlap.md` §4e). The overlap damping s(R) of the
   frame-charge → bound-dipole coupling measured on the ALMO pair (0.53–0.68, slope +0.37/Å) is **built as SPEC 34**
   (`frame_induction_damping`); frozen on step 20 it gives Z*(Mg) 1.95 at unchanged ε∞ (P1 pass). Frozen flagged copies
   `benchmarks_4g/runs/AuMgO_frozen_step20_fid_s{42,43}` are scored on both lines before training; then step 28. The
   frame-polarizability prior (trained α_O 0.2 / α_Mg 1.9 Å³, inverted against the table's 2.44 / 0.07) follows as its own step.
2. **Ensemble (P0):** the scalar O–O hole coupling (angular structure for p shells, or no O–O hops in the formal anion
   sublattice); criterion: 64-atom cell in the frame over a = 4.15–4.35 Å for both seeds.
3. **Data (P3, P5, P6): the bulk-augmented run family.** `data/MgO_PBE_togo_displaced.extxyz` holds the 148 displaced
   PBE supercells of the reference set (2 × 512 atoms, 146 × 64 atoms; forces only — `has_energy=F`; the fc2 cells are the
   reference's own force constants, so hold them out of training and keep the fc3 pair cells for training). Needs a
   per-structure energy mask in `train.py` (test first) or energies from OMat24's PBE MgO cells (VASP; per-source energy
   offset against the FHI-aims 4G energies). Scored on T1–T6 (no regression) and P0–P6.
4. **Label (P1):** Z* and ε∞ as training targets (`born_charges_from_energy(create_graph=True)` for the mean-field module;
   the frame + ∂P_bound/∂r route for TRACE-Redox), from the DFPT values above and the Materials Project DFPT set.

## Pre-registered: the bulk-augmented step (written 2026-11-06 23:50, before the launch)
- Runs `benchmarks_4g/runs/AuMgO_bulkaug_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bondlaw_r5_s{42,43}`:
  step 20's config with every path replaced (`check_run_config.py` OK) and one change — the data:
  `data/AuMgO_bulkaug_train.xyz` (300 4G slabs + 136 displaced PBE MgO cells, forces only) and
  `data/AuMgO_bulkaug_valid.xyz` (76 slabs + 10 held-out cells). Cold start, 40 epochs, both seeds, launched by
  `benchmarks_4g/watch_bulkaug.sh`, which scores T1–T6 and the probes as step 20 and then `score_phonons.py`.
- Needs the forces-only training path of `train.py` (Round 33 of the CHANGELOG; `tests/test_train_energy_mask.py`).
- Pass criteria: **P3** (TO within 3 % of 373.3) and **P5** (path RMS without NAC ≤ 15 cm⁻¹) — the force constants the
  bulk cells carry; **P0** must hold for both seeds at the PBE lattice constant; **T1–T6 must not regress** against step
  20 by more than the seed spread. P1, P4 (Z*, LO) are expected to stay failed: supercell forces carry no Born charge.
- Comparator: step 20 (same stack, same epochs, slab data only).
- **Launched 2026-11-07 00:12** (`nice 5`, alongside step 25's two runs; a first launch at 00:00 died in
  `compute_atomic_energies_from_dataset` on the forces-only cells — fixed, test extended, logs kept as
  `log_failed_start_2026-11-07_0005.txt`). The composition baseline is fitted on the 300 slabs alone and equals step 20's.
- **Amendment 2026-11-07 02:15 (before any score): warm start.** Both cold starts died in epoch 1 on the bulk cells —
  `ConfigurationSearchError: more than cap = 512 configurations within 2.5 eV` (random-init ladders put dozens of hole
  states of the 32 equivalent O's inside the window; the slabs never did this; the GPU copy died the same way). Relaunched
  from step 20's same-seed weights (`resume_from: warm_from_step20_sXX.pt`, epoch 0, weights only — the protocol of steps
  22/23), 40 epochs: the comparator stays step 20 and the one change stays the data, with "continued training" as the
  stated caveat. s43 on the Mac (`watch_bulkaug.sh`), s42 on GPU host 1 (`<gpu-host-workdir>`, `device: cuda`, paths
  translated, `check_run_config.py` OK there); the GPU copy is scored on the Mac once its `model.pt` is copied back.
  Failed logs kept as `log_failed_coldstart_*.txt`.
- 02:40: a first warm launch died on `atomic_energies_from_model: true` (refused with `resume_from`); as steps 22/23, the key
  is dropped and the checkpoint's atomic references are used (logs `log_failed_warmkey_*.txt`). Both seeds relaunched
  02:42 and load step 20's weights.
- 03:40: both seeds on GPU host 1 (two runs, 7.1 GB, 90 %): the GPU copy did 25 optimiser steps while the Mac copy
  did 6 (the Mac shares its CPU with step 25 and the PySCF scans), so the Mac copy of s43 was stopped (log kept as
  `log_mac_copy_stopped_*.txt`). ≈ 1.8 min per 8-structure step on the GPU → ≈ 100 min per epoch → ≈ 3 days for 40
  epochs; intermediate `model.pt` (best validation) can be scored earlier.

## Mid-training read of the bulk-augmented step (epoch 22 of 40, 2026-11-07 07:50; not the verdict)

Best checkpoints copied from GPU host 1 into `benchmarks_4g/runs/AuMgO_bulkaug_mid_e22_s{42,43}` and scored with
`score_phonons.py` (a = 4.2557 Å):

| | P0 | P1 Z*(Mg) | P2 ε∞ | P3 TO (PBE 373.3) | P4 LO (681.3) | P5 RMS no NAC | P6 RMS NAC | passed |
|---|---|---|---|---|---|---|---|---|
| step 20 s42 | FAIL (population 0.976) | — | — | — | — | contaminated | — | — |
| bulk-aug e22 s42 | **pass** (1.000000) | 2.417 | 3.022 | 384.5 (+3.0003 %, fails by 0.0003 %) | 812.0 | 29.2 | 27.6 | 2/7 |
| step 20 s43 | pass | 2.535 | 3.052 | 407.0 (+9.0 %) | 850.2 | 40.4 | 48.5 | 2/7 |
| bulk-aug e22 s43 | pass | 2.535 | 2.940 | 388.9 (+4.2 %) | 854.2 | 32.7 | 42.1 | 2/7 |

Reading: the bulk cells move the short-range force constants the right way (s43 TO +9.0 → +4.2 %, RMS 40 → 33) and
remove seed 42's lattice-constant cliff at the PBE volume (the cells sit at exactly that volume); P5's 15 cm⁻¹ is not
reached at epoch 22. Z* is unchanged, as predicted (supercell forces carry no Born charge).

**Epoch 25 (2026-11-07 08:20; `runs/AuMgO_bulkaug_mid_now_s*`, figure `bulkaug_e25_vs_pbe.png`, not released).** s42: P0, P2, **P3 pass**
(TO 381.7, +2.3 %), Z* 2.419, LO 810.5, RMS 29.1 / 27.7 (3/7); s43: TO 403.8 (+8.2 %; 388.9 at epoch 22 — the optical
stiffness fluctuates by ~15 cm⁻¹ between epochs), Z* 2.535, LO 863.5, RMS 35.9 / 46.3 (2/7). Force fit on the 10
held-out PBE cells (RMS force 36.5 meV/Å): step 20 s42 9119 meV/Å (the cells sit at a = 4.2557, where step 20 s42's
ensemble collapses), bulk-aug s42 **3.7 meV/Å (10 %)**; step 20 s43 8.3 (23 %), bulk-aug s43 6.3 (17 %). A 10–17 % force
error is a 5–8 % force-constant-weighted frequency error, ~20–30 cm⁻¹ on the optical branches — the size of the
remaining P5 miss. The LO branch (+130–180 cm⁻¹ at Γ) is the Born charge, untouched by this step.

**Bulk-augmented step cancelled, 2026-11-07 08:20 (decision of 2026-11-07: cancel it if the training cannot solve
the LO branch; branch `bulkaug-cancelled-2026-11-07` made first, under the never-delete policy).** Why it cannot solve the LO branch
(measured): (i) Z* stayed at 2.429 → 2.417 → 2.419 (s42) and 2.536 → 2.535 → 2.535 (s43) from step 20 through epochs 22 and
25 — supercell forces carry no Born charge; (ii) every parameter of the Gaussian-dipole bound sector scanned frozen at
ε∞ = 3.235 gives Z*(Mg) ≥ 2.33 (theory note §4e), i.e. √(LO² − TO²) ≥ 570 × 2.33/1.99 = 667 cm⁻¹ and LO ≥ 764 cm⁻¹
against P4's 660–700. And the rest of its purpose (P3, P5) was capped by the loss weighting: the 136 bulk cells carry
~0.1–0.3 % of the force loss once fitted to 10–17 % (an estimate from the test-set slab force errors), so the remaining
13 epochs could not move them much. Stopped at epochs 27 / 28 by exact PID on GPU host 1; best checkpoints (epoch 25 / 26)
copied to the Mac run dirs as `model.pt`, logs as `log_*_cancelled_2026-11-07_0820.txt`; last scored state: s42 P0
P2 P3 pass (3/7; TO +2.3 %, RMS 29.1), s43 2/7 (TO +8.2 %, RMS 35.9). Superseded by the reweighted bulk run (to be
pre-registered). Step 28b launched on GPU host 1 at 08:23 (`gpu_queue_28b.sh`).

## Pre-registered: the reweighted bulk run (written 2026-11-07 09:10, before the run)

Runs `benchmarks_4g/runs/AuMgO_bulkaugw_formal_..._r5_s{42,43}`: the cancelled bulk-augmented configs with every path
replaced (`check_run_config.py` OK) and one change — the 136 training and 10 validation PBE cells carry
`force_weight = 122.4`, the measured ratio of the mean squared force components of the training files (slabs 0.4185,
bulk cells 0.0378 eV/Å RMS), so each structure's force error counts relative to its own force scale: the bulk cells'
share of the force loss at equal relative error goes from 0.37 % to 31 % (`train.py` per-structure force weights;
`tests/test_train_force_weight.py`, written red first). Warm start from step 20 (as the cancelled run), 40 epochs, both
seeds; the validation loss (best-checkpoint selection) uses the same weights. Launched by `gpu_queue_bulkaugw.sh` on the
first free GPU (GPU host 2 when it runs nothing, or GPU host 1 when step 28b is done). Comparator: the cancelled
bulk-augmented run (epoch 25 / 26) and step 20.

**Pass criteria:** P5 path RMS without NAC ≤ 15 cm⁻¹ and P3 TO within 3 % on both seeds; P0 on both seeds; the
held-out PBE cells' force RMSE below 5 % of their RMS force (now 10 / 17 %); **4G T1–T6 within the seed spread of step
20** (the slab forces lose weight relative to the bulk cells; E/F RMSE on the 4G test set within 10 % of step 20's).
Not expected to move: P1, P4 (Z*, LO; step 28b's job).

## Step 28b mid-training (epoch 7 of 40, 2026-11-07 10:50; `runs/AuMgO_step28b_mid_s*`, figure `step28b_e7_vs_pbe.png` (not released))

The first trained model with the Born-charge physics (SPEC 35 anchor + SPEC 34 overlap damping), cold start:

| | P0 | P1 Z*(Mg) | P2 ε∞ | P3 TO | P4 LO | P5 / P6 RMS | LO − TO | LST ε0 | passed |
|---|---|---|---|---|---|---|---|---|---|
| s42 e7 | pass | **1.935** | 3.465 | 394.9 (+5.8 %) | **664.7 pass** | 46.4 / 36.9 | 269.8 | 9.82 | 4/7 |
| s43 e7 | pass | **1.934** | 3.471 | 352.8 (−5.5 %) | 640.1 | 51.9 / 43.8 | 287.3 | 11.42 | 3/7 |
| PBE | | 1.992 | 3.235 | 373.3 | 681.3 | | 308.0 | | |
| step 20 s43 | pass | 2.535 | 3.052 | 407.0 | 850.2 | 40.4 / 48.5 | 443.3 | | 2/7 |

Z* stays at 1.93 through training on both seeds (P1 pass) and the LO branch comes down from 850 to 640–665 cm⁻¹; √(LO² −
TO²) = 535 (s42) against PBE's 570, the 6 % deficit being Z*²/ε∞ (1.935²/3.465 against 1.992²/3.235): ε∞ sits at 3.47
because the anchor makes α = g α_Z and g has moved only 1.000 → 0.998 in 7 epochs. The short-range force constants are
not converged at epoch 7 of a cold start (RMS without NAC 46 / 52 against step 20's 40).
- **Caveat at epoch 7 (checked 11:00):** the acoustic branches are soft — TA(X) 204 (s42) / 168 (s43) against PBE 281
  (X is commensurate with the supercell, so this is not interpolation) — and s42 without NAC has an imaginary acoustic
  branch near Γ along K–Γ (min −21.6 cm⁻¹ at 16 of 204 path points; with NAC the path minimum is 0.0). Candidate
  artifact: Fourier interpolation from the 2×2×2 force constants near Γ; being checked with a 3×3×3 supercell. A cold
  start at epoch 7 has not converged its short-range force constants (step 20 at 40 epochs: TA(X) 266).

**3×3×3 check of the epoch-7 s42 (12:10):** the imaginary acoustic branch is real, not interpolation — min −31.7 cm⁻¹
without NAC (−32.1 with), 15 of 204 path points along K–Γ; TA(X) 204.0 at both supercell sizes.

**Step 28b best checkpoints at 12:30 (s42 epoch 9; s43 still epoch 7 — no validation improvement at epochs 8–9;
`runs/AuMgO_step28b_e10_s*`, figure `step28b_e9_vs_pbe.png` (not released), 28b and DFT only):** s42: P0, P1 (Z* 1.934), P2 (3.467),
**P3 (TO 383.0, +2.6 %)** pass; LO 657.5 (P4 misses 660 by 2.5), RMS 44.2 / 34.3 (4/7); no imaginary modes on the 2×2×2
path (16 points at epoch 7); TA(X) 198.3 (PBE 280.6) — the acoustic softening is the remaining defect. s43 unchanged (3/7).

**Step 28b, epoch 13 (2026-11-07 ~07:00 GMT; `runs/AuMgO_step28b_e13_s*`, figure `step28b_e13_vs_pbe.png` (not released)):** both seeds
4/7. s42: Z* 1.934, ε∞ 3.466, **TO 375.8 (+0.7 %)**, LO 653.2, RMS 38.8 / 28.0, TA(X) 202.8; s43: Z* 1.934, ε∞ 3.468,
**TO 373.4 (+0.0 %)**, LO 651.8, RMS 42.1 / 32.9 (was 51.9 / 43.8 at epoch 7), TA(X) 195.7. No imaginary modes on either
seed's path. LO sits 28–30 cm⁻¹ below PBE: √(LO² − TO²) = 534 / 534 against 570, i.e. Z*²/ε∞ is 12 % low (1.934²/3.466 against 1.992²/3.235) because
ε∞ = 3.47 (g = 0.998, still not learning) against 3.24. TA(X) 196–203 against 281 is unchanged since epoch 7.

**Step 28b, s42 epoch 24 / s43 epoch 22 (2026-11-07 ~09:00 GMT; `runs/AuMgO_step28b_e2x_s*`, figure
`step28b_e2x_vs_pbe.png`, not released):** path RMS with NAC **20.6 (s42) / 18.7 (s43, P6 pass — the first)**, without NAC 35.3 / 32.8;
acoustic RMS 21.1 / 14.0; TA(X) 233.5 / 249.9 (PBE 280.6; 203 / 196 at epoch 13); no imaginary modes. TO fell to
348.3 / 353.4 (−6.7 / −5.3 %; P3 fails, it passed at epoch 13) and LO to 637.3 / 640.6; √(LO² − TO²) = 534 on both, as
before (Z*²/ε∞ 12 % low, g = 0.999). Z* 1.932 / 1.934. s42 3/7, s43 4/7. The best checkpoint is selected by the 4G
validation loss, so TO moves by ±25 cm⁻¹ between selected epochs.

## Pre-registered: the bulk-only phonon model (written 2026-11-07 ~09:20 GMT, before the run; decided as a model for the phonon comparison only)

Runs `benchmarks_4g/runs/MgO_bulkonly_fpa_fid_r5_s{42,43}`: step 28b's final best weights (SPEC 35 anchor + SPEC 34
damping) fine-tuned on bulk MgO alone — `data/MgO_bulkonly_train.xyz` (the 136 displaced PBE cells; forces only) and
`MgO_bulkonly_valid.xyz` (the 10 held-out cells), 40 epochs, same config otherwise (paths replaced, `check_run_config.py`
OK; ε∞ and polarizability priors kept). Launched on GPU host 1 when 28b has finished (`gpu_queue_bulkonly.sh`), scored with
`score_phonons.py`.

**What it tests:** representability — whether TRACE's functional form, with the Born-charge physics, can carry PBE's bulk
force constants when given the bulk data. Not transfer: the training cells come from the same PBE calculation (Togo's
fc3 set, same volume and displacement amplitude) as the reference phonons (fc2 set; its two cells are excluded). Not a
4G model: it is never scored on T1–T6.

**Pass criteria:** all of P0–P6 on both seeds; the held-out cells' force RMSE below 5 % of their RMS force; Z* stays in
1.90–2.10 (fine-tuning must not undo the Born charge). If P5 ≤ 15 cm⁻¹ is missed even here, the backbone at this size
(r_max 5.29 Å, 40 epochs) cannot represent the PBE force constants, and the surface-trained deviation is not a data
question alone.

## Search for public PBE bulk MgO with energies and stresses (2026-11-07; parked by decision)

Formula-column scans of the ColabFit / Hugging Face mirrors (two ad hoc counting scripts, not released):
OMat24 (all 17 subsets, ~115 M structures) 478 MgO rows, rocksalt only the 49 AIMD-1000 K-NPT frames (24-atom (111)
cells), the rest other polymorphs or extreme rattles; MatPES-PBE-2025.2 76 MgO rows, **8 unique rocksalt** (VASP 6.4 PBE;
2-, 16-, 24-atom cells; V/f.u. 18.44–20.97 Å³, P +8.1 … −7.7 GPa; downloaded, SHA-256 verified, extracted to
`$TRACE_FP_CACHE/matpes_pbe_2025.2/extracted/MgO_matpes_pbe_rocksalt.extxyz`, stress flipped to ASE's sign from the data,
corr(V, −tr σ_stored/3) = +0.95); Alexandria PBE 3D 150 MgO rows, ≤ 12 rocksalt-density 2–4-atom relaxation frames;
MP-ALOE 414 MgO rows but r2SCAN. Public PBE rocksalt MgO with energies + forces + stresses totals ~70 frames. Not used
in any run yet; a computed strain + rattle set (QE PBE) is the alternative if needed.

## Step 28b final (2026-11-07 ~18:05 IST; figures `step28b_final_last_vs_pbe.png` (epoch 40, both seeds), `step28b_final_best_vs_pbe.png`)

| | P0 | Z*(Mg) | ε∞ | TO (373.3) | LO (681.3) | RMS no NAC / NAC | TA(X) (280.6) | passed |
|---|---|---|---|---|---|---|---|---|
| s42 best (epoch 35) | pass | 1.932 | 3.469 | 337.8 (−9.5 %) | 631.6 | 36.1 / 20.7 | 244.3 | 3/7 |
| s42 last (epoch 40) | pass | 1.932 | 3.468 | 345.4 (−7.5 %) | 635.7 | 34.2 / **18.6 (P6 pass)** | 243.4 | 4/7 |
| s43 best = last (epoch 40) | pass | 1.934 | 3.468 | 339.5 (−9.1 %) | 633.0 | 33.1 / **16.5 (P6 pass)** | 272.3 | 4/7 |
| step 20 s43 (comparator) | pass | 2.535 | 3.052 | 407.0 (+9.0 %) | 850.2 | 40.4 / 48.5 | 268.7 | 2/7 |

No imaginary modes on any path. √(LO² − TO²) = 534 cm⁻¹ on every model against PBE's 570: Z*²/ε∞ is 12 % low, from
ε∞ = 3.47 against 3.24 (the crystal factor g stayed at 0.999 through 40 epochs; the ε∞ prior's weight 0.05 did not move
it). TO is now 7.5–9.5 % soft (step 20 was 9 % stiff). Pre-registered for 28b: P1 PASS (both seeds); P0 PASS; P2 PASS;
P3, P4, P5 FAIL; P6 pass on s43 and s42-last, fail on s42-best (20.7). 4G T1–T6 scoring in progress.

**Bulk-only phonon model cancelled, 2026-11-07 ~18:10 IST (decision of 2026-11-07: cancel the run on GPU host 1; branch
`bulkonly-cancelled-2026-11-07` made first, under the never-delete policy).** Stopped by exact PID on GPU host 1 at epoch
4 / 3 of 40; its watcher stopped; logs kept as
`benchmarks_4g/runs/MgO_bulkonly_fpa_fid_r5_s*/log_*_cancelled_2026-11-07_1810.txt` and the best checkpoints as
`model_cancelled_best.pt`. Not scored.

**Reweighted surface + bulk run cancelled, 2026-11-07 ~18:15 IST (decision of 2026-11-07: cancel the run on GPU host 2
as well; branch `bulkaugw-cancelled-2026-11-07` made first, under the never-delete policy).** Stopped by exact PID on GPU
host 2 at epoch ~28 of 40; logs kept as
`benchmarks_4g/runs/AuMgO_bulkaugw_*/log_*_cancelled_2026-11-07_1815.txt`, best checkpoints as `model_cancelled_best.pt`.
Not scored.

## Pre-registered: the PBEsol bulk-only phonon model (written 2026-11-07 ~18:23 IST, before the run)

**Data:** Togo's PBEsol lattice-thermal-conductivity training set for MgO (NIMS MDR
https://mdr.nims.go.jp/datasets/59c5d436-339d-4b32-ba59-a3da5ea68bc7, CC BY 4.0; downloaded to
`$TRACE_PHONON_RAW/mdr_ltc2026_mgo/`): VASP PBEsol (GGA = PS, ENCUT 525.1 eV, PAW_PBE Mg_pv / O, 2×2×2 k), 200 rocksalt
supercells of 64 atoms at a = 4.21726 Å, every atom displaced 0.03 Å in a random direction; forces and supercell energies,
no stresses. Extracted by `tools/extract_ltc_pbesol.py` into `data/MgO_PBEsol_ltc_{train,valid}.xyz`
(180 / 20, numpy seed 0); **forces only** (`has_energy=F`; the VASP energy zero differs from the checkpoint's FHI-aims
references, and at one volume the energies carry 1.3 meV/atom). Born charge ±1.979, ε∞ 3.240 in the file.

**Runs:** `benchmarks_4g/runs/MgO_PBEsol_bulkonly_fpa_fid_r5_s{42,43}`, 28b's final best weights (s42 epoch 35, s43 epoch
40; SPEC 35 anchor + SPEC 34 damping) fine-tuned for 40 epochs on the PBEsol cells, same config otherwise (paths
replaced, `check_run_config.py` OK). On the Mac, one at a time after step 26 s43 (`mac_queue_pbesol.sh`; the queued 28a
runs were cancelled by decision, branch `mac-queue-28a-cancelled-2026-11-07`). Starting point: 28b s42 on 5
held-out PBEsol cells, force RMSE 45.7 meV/Å against an RMS force of 227 meV/Å (cos 0.981).

**Reference and criteria** (`score_phonons.py --ref pbesol`, a = 4.21726 Å): phonondb mp-1265 PBEsol (fc2 from 2
single displacements, the same author and settings; a fitting test, not a prediction test). P0 frame population;
P1 Z*(Mg) 1.90–2.10 (PBEsol 1.979); P2 ε∞ 2.9–3.5 (3.242); P3 TO within 3 % of 391.0; P4 LO 673–713 (693.9); P5 path
RMS without NAC ≤ 15 cm⁻¹; P6 with NAC ≤ 20 cm⁻¹; held-out force RMSE below 5 % of the RMS force.

**Step 28b on the 4G tests (scored 2026-11-07 19:13 IST; comparator step 20, since 28a was cancelled):**

| | undoped E / F | doped E / F | wet_d (DFT −2.7) | wet_u (DFT +928.9) | T5 = d − u (DFT −931.6) | ε∞ (T6, a = 4.212) |
|---|---|---|---|---|---|---|
| step 20 s42 | 2.49 / 0.069 | 2.73 / 0.089 | −453.1 | +536.3 | −989.4 | 3.371 |
| **28b s42** | 5.46 / **0.163** | 3.93 / 0.125 | +296.5 | +174.4 | **+122.1 (wrong sign)** | 3.558 |
| step 20 s43 | 0.91 / 0.073 | 1.27 / 0.109 | −222.1 | +753.8 | −975.9 | 3.170 |
| **28b s43** | 1.48 / **0.155** | 2.01 / 0.111 | +103.7 | +718.3 | −614.6 (66 %) | 3.557 |

E in meV/atom, F in eV/Å, wetting in meV. **28b fails its 4G criteria:** the undoped-slab force RMSE doubles on both
seeds (0.069 → 0.163, 0.073 → 0.155), T5 loses the transfer split (s42 wrong sign, s43 66 %), and T6 ε∞ 3.56 leaves the
2.5–3.5 window. The Au₂ charge shift stays at −1.00 e (DFT −0.27). The phonon line passes P0, P1, P2, P6; the 4G line
regresses. Reading (hypothesis, not yet tested): the two terms change the surface electrostatics the backbone was fitted
against — the anchor puts α on the O²⁻ (2.44 Å³ × g; the surface O is the most polarisable site), and ε∞ sits at 3.56
because g never moved — so the slab forces now carry an induction error the backbone cannot absorb in 40 epochs.

**Pre-registered: the frozen anchor / damping test on the 4G slabs (written 2026-11-07 ~20:00 IST, before the run;
`probe_frozen_terms_4g.py`).** Step 20's weights (both seeds), frozen, on every 4th structure of the 4G test set (50
undoped + 50 doped), with neither term, the anchor alone (g = 1 and 0.92), the damping alone, and both (28b's start).
Reading: the term whose frozen addition raises the undoped force RMSE toward 28b's 0.16 eV/Å is the one that costs the
slabs; a term that leaves it within 10 % of step 20's is not. Runs on the Mac after step 26 is trained and scored, before
the PBEsol bulk-only runs (`mac_queue_pbesol.sh`; one job at a time, by decision).

**PBEsol bulk-only, seed 42, epoch 6 of 40 (2026-11-08 ~00:05 IST; `benchmarks_4g/runs/MgO_PBEsol_s42_mid`, figure
`pbesol_bulkonly_s42_e6_vs_pbesol.png`, not released; `score_phonons.py --ref pbesol`, a = 4.21726 Å):** **5/7** — P0 pass, P1 Z* 1.942
pass, P2 ε∞ 3.541 FAIL (window 2.9–3.5; PBEsol 3.242), P3 TO 404.4 FAIL (+3.4 %; 391.0), P4 LO 673.1 pass (by 0.1; 693.9),
**P5 path RMS without NAC 8.1, P6 with NAC 9.0 cm⁻¹ pass** (acoustic 7.2, optical 10.5); X 265, 265, 412, 441, 441, 517
against 284, 284, 418, 440, 440, 535; L 276, 276, 368, 368, 534, 547 against 273, 273, 356, 356, 538, 557; no imaginary
modes; held-out force RMSE 0.0101 eV/Å (4.4 % of 0.227). √(LO² − TO²) = 538 against PBEsol's 573: Z*²/ε∞ is 12 % low
(1.942²/3.541 against 1.979²/3.242), from ε∞ 3.54 against 3.24 — g again not learning. Mid-training; the final checkpoint is scored by `mac_queue_pbesol.sh`.

## Pre-registered: the step 20 control on the PBEsol data (written 2026-11-08 ~00:14 IST, before the run)

Runs `benchmarks_4g/runs/MgO_PBEsol_bulkonly_step20_r5_s{42,43}`: step 20's weights (no SPEC 34 / 35) fine-tuned on the
same 180 PBEsol cells (20 held out), 40 epochs, same config as the 28b-physics PBEsol runs — the only differences are
`frame_polarizability_anchor` and `frame_induction_damping` (checked: no other redox or top-level key differs). On the
Mac, one job at a time (`mac_queue_pbesol2.sh`: 28b-PBEsol s42 → control s42 → 28b-PBEsol s43 → control s43), each scored
with `score_phonons.py --ref pbesol` and plotted without and with NAC.

**Prediction:** path RMS without NAC within a few cm⁻¹ of the 28b-physics PBEsol run (the force constants are the
data's); Z*(Mg) 2.3–2.6 (supercell forces carry no Born charge; step 20's local-field enhancement); LO above 800 cm⁻¹
(Born–Huang with Z* ≈ 2.43, ε∞ ≈ 3: splitting ≈ 1.33 × DFT's, LO ≈ 860 against 694); P1, P4, P6 FAIL. If it comes true,
the LO agreement of the 28b-physics run is the new physics' contribution, not the data's.

**28b-physics PBEsol bulk-only, seed 42, final (epoch 40; scored 2026-11-08 04:40 IST; figure
`MgO_PBEsol_bulkonly_fpa_fid_r5_s42_vs_pbesol.png`):** 5/7 — **path RMS without NAC 2.2, with NAC 5.9 cm⁻¹**, TO 392.9
(+0.5 %), Z* 1.942, P0 pass; FAIL P2 (ε∞ 3.541, window 2.9–3.5) and P4 (LO 666.3 against the 673–713 window; PBEsol 693.9):
the LO deficit is the ε∞ excess (g = exp(log_alpha_scale) never moved). Held-out force RMSE 0.0029 eV/Å (1.3 % of 0.227).

**Step 20 control, seed 42: blocked at start (04:40).** `ConfigurationSearchError: the amplitude closure holds more than
cap = 512 configurations` on a PBEsol cell in epoch 1: step 20 s42's ensemble collapses at a = 4.217 Å — the measured
lattice-constant cliff of the scalar O–O hole coupling (64-atom cell: F − E_frame −366 eV at this volume; benchmarks_4g README). Not a
code fault; raising the cap would train on a collapsed ensemble. The s42 control is therefore not runnable without first
fixing that defect; the s43 control (step 20 s43 selects no hole configurations in bulk and was clean at 4.217) runs
after the 28b-physics s43 run as queued.

**28b-physics PBEsol bulk-only, seed 43, final (scored 2026-11-08 09:35 IST; figure
`MgO_PBEsol_bulkonly_fpa_fid_r5_s43_vs_pbesol.png`):** 5/7 — path RMS without NAC 3.0, with NAC 5.8 cm⁻¹; TO 396.7
(+1.5 %); Z* 1.943; LO 668.7 (P4 FAIL, window 673–713); ε∞ 3.541 (P2 FAIL); held-out force RMSE 0.0025 eV/Å (1.1 %).
Both seeds agree: 2.2 / 3.0 cm⁻¹ without NAC, 5.9 / 5.8 with NAC; the shared failure is ε∞ (g never learns).

**Step 20 control, seed 43, final (epoch 40; scored 2026-11-08 14:25 IST; figure
`MgO_PBEsol_bulkonly_step20_r5_s43_vs_pbesol.png`; an interim score of the epoch-22 weights, 12:30, gave the same picture:
Z* 2.521, LO 871.1):** 3/7 — Z*(Mg) **2.520** (P1 FAIL), ε∞ 2.875 (P2 FAIL), TO 398.9 (+2.0 %), LO **871.6** (P4 FAIL),
path RMS without NAC **4.7** (P5 pass), with NAC **34.6** cm⁻¹ (P6 FAIL); the phonopy LO–TO splitting equals the
Born–Huang value from (Z*, ε∞) (776.2 = 776.2 cm⁻¹ at epoch 22).

| seed 43 | Z*(Mg) | ε∞ | TO | LO | RMS no NAC | RMS NAC | passed |
|---|---|---|---|---|---|---|---|
| step 20 control | 2.520 | 2.875 | 398.9 | 871.6 | 4.7 | 34.6 | 3/7 |
| 28b physics (SPEC 34 + 35) | 1.943 | 3.541 | 396.7 | 668.7 | 3.0 | 5.8 | 5/7 |
| PBEsol DFT | 1.979 | 3.242 | 391.0 | 693.9 | — | — | — |

**Prediction scored:** Z* in 2.3–2.6 (2.520), LO above 800 (871.6), P1 / P4 / P6 fail, RMS without NAC within a few
cm⁻¹ of the 28b run (4.7 against 3.0): all as written. Not predicted: P2 also fails (ε∞ 2.875, below the 2.9 window).
Reading: the force constants come from the data in both models; the LO branch is set by Z*²/ε∞ (2.21 here, 1.07 for
28b, 1.21 DFT), and the step from 2.21 to 1.07 is SPEC 34 + 35's. One seed for the control (s42 is blocked by the O–O
hole-coupling cliff).

## Pre-registered: TRACE without electrostatics on the PBEsol cells (written 2026-11-09 16:04 IST, before the runs; decision of 2026-11-09: train TRACE without electrostatics on the bulk MgO data and compare against DFT)

**Question.** How does TRACE's local backbone alone reproduce bulk MgO phonons against PBEsol DFT, and what exactly do the
electrostatics add?

**Runs** `benchmarks_4g/runs/MgO_PBEsol_bulkonly_local_r5_s{42,43}`: the bulk-only electrostatic runs' configuration
(`MgO_PBEsol_bulkonly_step20_r5_s43`) with the `electrostatics` block removed (TRACE local, as the 4G protocol's local
arms), the same 180 / 20 PBEsol cells (forces only), r_max 5.29, batch 1, lr 1e-3; cold start (the warm-start weights carry
electrostatic parameters), so **100 epochs** (cosine t_max 100) instead of 40; the training-time anneals off (no ensemble);
2 threads, `nice 10`, concurrently with step 31. Trial (2 epochs, `..._local_r5_s43_trial`): 1.8–1.9 min per epoch,
validation F RMSE 0.040 → 0.019 eV/Å.

**Scored by** `phonons_model.py` (TRACE's exact phonopy settings for a model without electrostatics; checked on a TRACE
model against `mgo_phonons.py` to 0.0006 cm⁻¹) at a = 4.21726 Å against phonondb mp-1265 PBEsol: the band without NAC; a
local model has no Born charge, so its NAC band is its no-NAC band; and, as a reference, its force constants with DFT's
own Z* 1.979 / ε∞ 3.2417 applied. Path RMS with frequencies sorted per q-point.

**Predictions (hypotheses), reported, not pass/fail — a control:**
- **L1** without NAC the local model is level with the electrostatic models (RMS within ±2 cm⁻¹ of 28b physics' 3.0): the
  2³ supercell's force constants reach at most ≈ 4.2 Å (minimum image), inside r_max 5.29, so a local model can carry them;
- **L2** with NAC it has no LO–TO splitting (LO = TO ≈ 391 against 693.9) and a NAC path RMS above 30 cm⁻¹ (28b: 5.8);
- **L3** with DFT's Z* / ε∞ supplied from outside, its NAC band is as good as its no-NAC band (NAC is analytic given
  Z*, ε∞): what the electrostatics contribute here is predicting Z* and ε∞, not the short-range force constants;
- held-out force RMSE reported against the electrostatic runs' 0.0025–0.0029 eV/Å.

**Interim (2026-11-09 ~17:45 IST, an interim read by decision; best checkpoints s42 epoch 39 / s43 epoch 38 of 100, copied before
scoring; `runs/MgO_PBEsol_bulkonly_local_r5_s{42,43}_interim_pbesol/{own,dft_born}`; figure
`MgO_PBEsol_bulkonly_local_r5_s42_interim_vs_pbesol.png`; the s43 one is not released).**

| a = 4.21726 Å | TO (391.0) | LO (693.9) | RMS no NAC | RMS NAC, own (no Z*) | RMS NAC, DFT's Z* / ε∞ supplied | val F (eV/Å) |
|---|---|---|---|---|---|---|
| local s42 | 392.8 | = TO | 2.7 | 53.7 | 2.7 (LO 694.9) | 0.0029 |
| local s43 | 393.2 | = TO | 2.1 | 53.7 | 2.1 (LO 695.1) | 0.0020 |
| 28b physics (electrostatic) s42 / s43 | 392.9 / 396.7 | 666.3 / 668.7 | 2.2 / 3.0 | 5.9 / 5.8 (own Z* 1.94, ε∞ 3.54) | — | 0.0029 / 0.0025 |
| step 20 control s43 | 398.9 | 871.6 | 4.7 | 34.6 (own Z* 2.52, ε∞ 2.88) | — | — |

All three predictions hold so far: L1 (no NAC level with the electrostatic models: 2.1–2.7 against 2.2–3.0), L2 (no
LO–TO splitting; NAC RMS 53.7 > 30), L3 (DFT's Z* / ε∞ supplied: the NAC band as good as the no-NAC one, LO within 1.2
cm⁻¹). Reading: in these 64-atom supercells the short-range force constants (including the supercell's own share of the
Coulomb interaction) are learnable without electrostatics; what the electrostatic model contributes to bulk MgO phonons is
predicting Z* and ε∞ itself (28b: 1.94 / 3.54 against DFT 1.98 / 3.24, NAC RMS 5.8), which a local model needs from an
external DFPT calculation (with them: 2.1–2.7). Interim; the final checkpoints are scored by the watcher.
[Release note: the watcher is `benchmarks_4g/watch_local_phonons.sh`. Per seed it runs `phonons_model.py` (band
without NAC, copied as the NAC band), `phonons_model.py --born-from` with DFT's Z* 1.9791 / ε∞ 3.2417,
`score_band_dir.py` on both outputs and `plot_pbesol_pair.py`.]

**Stopped by decision (2026-11-09 ~17:50 IST: the fit was judged good enough; branch
`local-phonons-stopped-2026-11-09` made first).** Both runs stopped by exact PID after epoch 40 of the pre-registered 100
(the watcher too); runs kept. Final scored checkpoints: s42 epoch 39 (the interim above), s43 epoch 40
(`runs/MgO_PBEsol_bulkonly_local_r5_s43_epoch40_pbesol`, figure `MgO_PBEsol_bulkonly_local_r5_s43_epoch40_vs_pbesol.png`):
TO 391.2 (DFT 391.0), path RMS without NAC **1.7**, with NAC (own, no Z*) 54.0, with DFT's Z* / ε∞ supplied 1.7 (LO
694.0 against 693.9); validation F RMSE 0.00175 eV/Å. L1, L2, L3 all as predicted, on both seeds. The local backbone
alone fits the 64-atom supercells' force constants to within 2–3 cm⁻¹; the LO branch needs Z* and ε∞, which only the
electrostatic model predicts (28b: NAC RMS 5.8 from its own Z* 1.94 / ε∞ 3.54) and a local model must borrow from DFPT.

## Pre-registered: the electrostatics phonon model retrained with the extensive forms (written 2026-11-12 10:51 IST, before the runs; decision of 2026-11-12: retrain the bulk MgO phonon model with electrostatics on the updated equations, and test it on the Mac)

**Frozen checks done first (today's code, the saved `MgO_PBEsol_bulkonly_fpa_fid_r5_s43`; `score_phonons.py --ref pbesol`, 15 s each):**

| setting at test time | Z*(Mg) | ε∞ | TO | LO | RMS no NAC | RMS NAC | passed |
|---|---|---|---|---|---|---|---|
| as stored, scored 2026-11-08 | 1.943 | 3.541 | 396.7 | 668.7 | 3.0 | 5.8 | 5/7 |
| as stored, today's code (`runs/recheck_2026-11-12_stored_fpa_fid_s43_pbesol`) | 1.943 | 3.541 | 396.7 | 668.7 | 3.0 | 5.8 | 5/7 |
| `'onsite'` + clusters + `'fermi'` (frozen copy `MgO_PBEsol_frozen_fpa_fid_ext_s43`) | 1.943 | 3.541 | 396.7 | 668.7 | 3.0 | 5.8 | 5/7 |
| the same + `bvk_mesh (2,2,2)` (`…_ext_m222_s43`) | not scored: `mgo_phonons.py` asserts on the supercell's record (513 units against the cell's 65) | | | | | | |

So (1) the code changes since 2026-11-08 leave the phonon result unchanged to the digits printed, and (2) on this
crystal the extensive forms change nothing: bulk MgO stays in its frame configuration (P0: population 1.000000), and
SPEC 28.6, 36 and 38 act only on configurations beyond it. The phonon tool is not supercell-aware (a limit of
`tools/electrostatics/mgo_phonons.py`, recorded; in the frame a supercell changes no energy, benchmarks_4g README R26).

**Runs** `benchmarks_4g/runs/MgO_PBEsol_bulkonly_fpa_fid_ext_r5_s{43,42}`: the config of
`MgO_PBEsol_bulkonly_fpa_fid_r5_s{43,42}` (28b's best weights, SPEC 34 + 35, 40 epochs on the 180 PBEsol cells, 20
held out) with `state_polarizability_stiffness: onsite`, `diabatic_space: clusters`, `band_form: fermi`. Key by
key nothing else differs (paths replaced; the warm start is the earlier run's own file; `check_run_config.py` OK).
Three keys at once is not one change of the trained function here: the frozen check shows all three are the
identity on this data. The run tests the **training path** of the new code (the clusters space with the two
response priors, SPEC 36.10; the Fermi band's gradients). On the Mac, one at a time, s43 first
(`mac_queue_pbesol_ext.sh`); each scored with `score_phonons.py --ref pbesol` and plotted against DFT only.

**Criteria (each seed; comparator: the same seed's earlier run).**
- **E1** the training finishes: no refusal (`ConfigurationSearchError`), no missing response record, no NaN stop.
- **E2** the same verdicts on P0–P6 as the earlier run (5/7: P2 and P4 fail), or better.
- **E3** against the earlier same-seed run: |ΔZ*| ≤ 0.01, |Δε∞| ≤ 0.02, |ΔTO| ≤ 5 cm⁻¹, |ΔLO| ≤ 5 cm⁻¹, both path RMS
  within 1.5 cm⁻¹, held-out force RMSE within 20 %. (The two earlier seeds differ by 3.8 in TO, 2.4 in LO, 0.8 and
  0.1 in the RMS; the trajectories are not expected to be bitwise equal, since the new path sums the same energy in
  a different order.)
- **E4** reported: minutes per epoch against the earlier run's 6.9 (s43).

**Prediction.** E1–E3 pass: the same model within its own noise. **This run is not expected to improve the phonons.**
The known failure of this model family, ε∞ 3.54 against PBEsol's 3.24 and with it LO 20–27 cm⁻¹ low, comes from the
frame-polarizability factor g that does not learn (records above); the extensive forms do not touch it. That needs
its own step (an ε∞ label at the PBEsol volume, or a reason why g gets no gradient).

**Launched 2026-11-12 10:59 IST** (`mac_queue_pbesol_ext.sh`, log `benchmarks_4g/runs/mac_queue_pbesol_ext.log`; s43 started
10:51:55, s42 follows). `train.py` warns at the warm start that the stiffness statement differs from the
checkpoint's ('block' → 'onsite'), as it should. **Epoch 1 of s43 equals the earlier run's epoch 1 to every printed
digit** (training force RMSE 0.048171, validation 0.029651 eV/Å); 447 s against 378 s for the epoch, with other
scoring runs on the Mac during it.

## Why the crystal factor g does not move: it is held by the Au-atom polarizability prior (2026-11-12 11:31 IST; measurements on frozen weights, no training, no package change; question of 2026-11-12: why the oxygen polarizability factor does not learn)

**What the records said:** "g stayed at 0.999", "g never learns". **What is measured** (`probe_alpha_scale_gradient.py`,
which calls `train.py`'s own prior-loss functions; output `runs/alpha_scale_2026-11-12/gradient_s{43,42}.txt`):

- g does move, and it has **converged**. `log_alpha_scale`: −0.00127 / −0.00101 after 28b (s43 / s42), −0.00297 /
  −0.00296 after the PBEsol fine-tune (g = 0.99704 on both seeds).
- It is in the optimizer (AdamW group `adamw_aux_decay`, lr 1e-3, weight decay 1e-5: 40 tensors with the anchor
  against 39 without).
- **One optimizer step's gradient on log g, by term (s43 / s42):**

  | term | model against its label | weight × ∂loss/∂log g |
  |---|---|---|
  | dielectric prior (weight 0.05) | ε 3.4445 against 2.95 | **+0.046328 / +0.046330** (wants g down) |
  | polarizability prior (weight 0.3) | α(Au atom) 5.3924 against 5.4354 Å³ | **−0.046347 / −0.046303** (wants g up) |
  | the two together | | −0.000019 / +0.000027 |
  | force loss, a window of 8 training cells | | −5.6e-05 / +5.6e-05 (std 1.8e-05) |

  The two priors cancel to 0.04 %. The forces are 800 times weaker than either.
- **Cause.** SPEC 35 has **one** factor g for every tabulated species, and Au⁰ is tabulated. So the same g scales
  the O²⁻ ion in the crystal and the **isolated Au atom** of the polarizability prior, whose label is its free-atom
  value. g = 1.005 satisfies the Au prior, g ≈ 0.82 the ε prior, and the Au spring is 20 times stiffer near g = 1.
  Frozen scan of g on the s43 weights (prior cell, a = 4.265 Å):

  | g | ε (prior cell) | α(Au atom), Å³ | weighted ε loss | weighted α loss | sum |
  |---|---|---|---|---|---|
  | 1.0000 | 3.4529 | 5.4085 | 0.004215 | 0.000073 | 0.004287 |
  | **0.9970 (trained)** | 3.4445 | 5.3924 | 0.004076 | 0.000185 | **0.004260 (the minimum)** |
  | 0.9500 | 3.3119 | 5.1380 | 0.002182 | 0.008843 | 0.011025 |
  | 0.9200 | 3.2272 | 4.9758 | 0.001281 | 0.021126 | 0.022406 |
  | 0.9000 | 3.1708 | 4.8676 | 0.000812 | 0.032239 | 0.033051 |
  | 0.8500 | 3.0298 | 4.5972 | 0.000106 | 0.070261 | 0.070367 |

**What the phonons would be if g were free** (frozen copies of the s43 PBEsol model with g set by hand,
`benchmarks_4g/runs/MgO_PBEsol_frozen_fpa_fid_g{095,092,090,088}_s43`; `score_phonons.py --ref pbesol`; bulk MgO holds
no Au, so this is what a factor on the crystal's ions alone would give; **not pre-registered, not trained**):

| g | Z*(Mg) | ε∞ | TO | LO | RMS no NAC | RMS NAC | passed |
|---|---|---|---|---|---|---|---|
| 0.997 (trained) | 1.943 | 3.541 | 396.7 | 668.7 | 3.0 | 5.8 | 5/7 |
| 0.95 | 1.955 | 3.404 | 396.8 | 680.3 | 4.3 | 4.8 | 7/7 |
| 0.92 | 1.962 | 3.316 | 397.0 | 688.0 | 5.7 | 5.5 | 7/7 |
| **0.90** | **1.967** | **3.258** | 397.0 | **693.4** | 6.7 | 6.5 | **7/7** |
| 0.88 | 1.973 | 3.200 | 397.1 | 698.9 | 7.9 | 7.7 | 7/7 |
| PBEsol DFT | 1.979 | 3.242 | 391.0 | 693.9 | — | — | — |

- g is fitted here by hand to one number (ε∞). **LO is not fitted and lands on DFT** (693.4 against 693.9 at
  g = 0.90), and Z* moves toward DFT as well. The path RMS without NAC rises from 3.0 to 6.7 cm⁻¹: the short-range
  part was trained against the electrostatics at g = 0.997 and has not been retrained.
- **Not a result yet.** A hand-set parameter on frozen weights; one seed; the criteria were not registered for it.

**The fix this points to (not built: SPEC 35's equation would change, which needs a separate decision, and the
package is not edited while the Mac queue trains from it).** The crystal factor belongs to ions inside a crystal, not to a
neutral free atom: apply g to the charged frame ions only (or to the anions only; ESTABLISHED as physics: the
in-crystal reduction of anion polarizabilities, Fowler and Madden, Phys. Rev. B 29, 1035 (1984), cited from memory,
not looked up), and leave Au⁰ at its tabulated free-atom value. Then the Au prior no longer holds g.
**A second question the scan raises:** the ε prior's label is the experimental 2.95, and the phonon references are
DFT (PBEsol 3.242, PBE 3.235). Freed, g would run to about 0.82 (ε 2.95 on the prior cell), past the g ≈ 0.90 that
matches DFT's ε∞ and LO. Which ε∞ the prior should carry is left to a decision.

**Cancelled 2026-11-12 12:30 IST by decision** (the latest equations changed nothing on this crystal; branch
`crystal-factor-2026-11-12` made first, under the never-delete policy). The queue script and the s43 training were
stopped by exact PID during epoch 13 of 40; s42 never started. Kept:
`log_cancelled_2026-11-12_1230.txt`, `model_cancelled_best.pt` (epoch 11). **Not scored as a result.** What it showed
before it stopped: the training force RMSE equals the earlier run's to all six printed digits through epoch 11
(0.048171, 0.012455, 0.005587 at epochs 1, 6, 11) and differs in the fifth digit at epoch 12 (0.006714 against
0.006703), as predicted (not bitwise); the epoch-11 checkpoint scores 5/7 with the same failures (ε∞ 3.541, LO
668.6; figure `MgO_PBEsol_ext_s43_epoch11_nac_vs_pbesol.png`, not released). E1–E4 are left unscored. Time per epoch 481 s
against 378–396 s, with probes running beside it.

## SPEC 35.1 built, and pre-registered: the phonon model with the crystal factor on the ions only and the DFT ε∞ label (written 2026-11-12 12:36 IST, before the runs; decision of 2026-11-12: build the fix, use the DFT value 3.24, and cancel the phonon training)

**The fix (SPEC 35.1; tests first, 6 of 7 red before the code).** `RedoxConfig.frame_polarizability_scope`:
`'ions'` (the new default) applies g to the charged frame ions only; a neutral frame atom (Au⁰) keeps its tabulated
free-atom polarizability. `'all'` is SPEC 35 as first built (legacy, kept); a checkpoint that predates the field
loads as `'all'` with a warning, so 28b and the earlier PBEsol models reproduce their trained energies.
**Check on the real weights** (the saved s43 PBEsol model, scope switched to `'ions'`;
`runs/alpha_scale_2026-11-12/gradient_s43_scope_ions.txt`): the polarizability prior's gradient on log g is
**+0.000000** (it was −0.046347), α(Au atom) is 5.4085 Å³ at every g, and the dielectric prior's +0.046328 is
unopposed.

**The label.** `data/MgO_bulk_eps_prior_pbesol.xyz` (new, `make_mgo_eps_prior_dft.py`): the 8-atom cell at the PBEsol
lattice constant 4.21725941 Å with phonondb's DFPT ε∞ = 3.2417. (A PBE file, 3.235 at 4.25566 Å, is written beside
it for the slab models; not used here.) The earlier prior (the measured 2.95 at 4.265 Å) is kept and unchanged.

**Runs** `benchmarks_4g/runs/MgO_PBEsol_bulkonly_fpa_fid_ions_r5_s{43,42}`: the config of
`MgO_PBEsol_bulkonly_fpa_fid_r5_s{43,42}` (28b's best weights, 40 epochs, the 180 PBEsol cells) with
`frame_polarizability_scope: ions`, the dielectric prior's file replaced by the PBEsol one, and
`state_polarizability_stiffness: onsite` written out (today's default; the identity on this crystal: the cancelled
run above matched the earlier one to six digits for 11 epochs with it). Two changes, both by the decision (the
scope and the label); key by key nothing else differs (`check_run_config.py` OK). On the Mac, one at a time, s43
first (`mac_queue_pbesol_ions.sh`). The same decision is read as the go-ahead for this retraining: it replaces the
cancelled one.

**Criteria (each seed; reference PBEsol, `score_phonons.py --ref pbesol`).**

| id | criterion | earlier run (s43 / s42) | frozen copy at g = 0.90 (s43) |
|---|---|---|---|
| **F1** | the factor moves: final g within 0.86–0.93 | 0.997 / 0.997 | set by hand |
| **F2** | ε∞ within 3.19–3.29 (PBEsol 3.2417 ± 0.05) | 3.541 / 3.541 | 3.258 |
| **F3** | LO within 684–704 cm⁻¹ (693.9 ± 10) | 668.7 / 666.3 | 693.4 |
| **F4** | 7/7 on P0–P6 | 5/7 / 5/7 | 7/7 |
| **F5** | path RMS ≤ 4.5 cm⁻¹ without NAC and ≤ 6.0 with NAC | 3.0, 5.8 / 2.2, 5.9 | 6.7, 6.5 |
| **F6** | held-out force RMSE within 20 % of the earlier run's | 0.0025 / 0.0029 eV/Å | — |

**Predictions.** F1–F4 pass: g needs to travel 0.11 in log, and the first epochs at lr 1e-3 allow about 0.02 per
epoch. F5 is the open one: it asks the retrained short-range part to win back what the frozen copy lost (6.7 and
6.5 at g = 0.90) while LO stays on DFT. **PASS** = F1–F6 on both seeds. **What a pass would not show:** anything
about the Au₂/MgO slabs. Whether a freed g also removes the slab damage of step 28b is the next test, on a slab
training.

**Launched 2026-11-12 12:50 IST** (`mac_queue_pbesol_ions.sh`, the s43 training first; log
`benchmarks_4g/runs/mac_queue_pbesol_ions.log`). Suites with SPEC 35.1 before the launch: `tests/redox` 963 passed;
`pytest tests --ignore=tests/redox` 1045 passed, 4 skipped. The log shows the PBEsol prior file and `train.py`'s two
statement warnings at the warm start (stiffness 'block' → 'onsite'; scope 'all' → 'ions'), as intended.

**Mid-training read, s43, epoch 9 of 40 (2026-11-12 14:01 IST; not the verdict; a copy of the best checkpoint in
`benchmarks_4g/runs/MgO_PBEsol_ions_s43_mid`, figure `MgO_PBEsol_ions_s43_epoch9_nac_vs_pbesol.png`, not released).** The factor: 0.9987 →
0.976 (epoch 1) → 0.955 (epoch 2) → **0.8943** (epoch 9; the frozen scan put ε∞ = 3.2417 at about 0.894). The prior
cell's ε is 3.2414 against the label 3.2417. `score_phonons.py --ref pbesol`: **7/7** — Z*(Mg) 1.969 (DFT 1.979),
ε∞ 3.241 (3.242), TO 395.0 (+1.0 %), **LO 693.8 (693.9)**, path RMS 3.0 cm⁻¹ without NAC and **3.0 with NAC** (the
earlier model: 3.0 and 5.8). Training force RMSE 0.006939 eV/Å at epoch 9 against the earlier run's 0.006379. ε∞ is
fitted (it is the prior's label); Z*, TO and LO carry no label.

**Cancelled 2026-11-12 15:08 IST by decision** (the run was not the single model; branch
`pbesol-ions-cancelled-2026-11-12` made first, under the never-delete policy). The queue script and the s43 training
were stopped by exact PID during epoch 18 of 40; s42 never started. Kept: `log_cancelled_*.txt`,
`model_cancelled_best.pt` (epoch 16). **F1–F6 are not scored as a verdict** (one seed, 16 of 40 epochs). The last
read, for the record (`benchmarks_4g/runs/MgO_PBEsol_ions_s43_cancelled_best`, `score_phonons.py --ref pbesol`): the
factor 0.8944; **7/7**: Z*(Mg) 1.969, ε∞ 3.242, TO 398.7 (+2.0 %), LO 695.9, path RMS 4.1 without NAC and 4.0 with
NAC; training force RMSE 0.00419 eV/Å at epoch 17. (At epoch 9 the same run read TO 395.0, LO 693.8, RMS 3.0 / 3.0:
the optical branches moved by a few cm⁻¹ between the two checkpoints, as earlier runs of this family did.) What the
run established before it stopped: freed by SPEC 35.1, the factor goes to the value the frozen scan predicted and
the phonon criteria that failed (ε∞, LO) pass. The single model (benchmarks_4g README, step 33) carries the same
fix and is where the work continues.

## Single model, Mac early look (seed 43): epoch 6 figure, and what P5 measures against the PBE reference (2026-11-12 17:55 IST)

Not the pre-registered run (that is step 33 on GPU host 2, both seeds); a mid-training read of
`benchmarks_4g/runs/AuMgO_single_all_mac_s43` (stage A, every term on), scored by `watch_single_mac_phonons.sh` at each
new best validation loss. Band files of the epoch 6 checkpoint are kept in `runs/AuMgO_single_all_mac_s43_epoch6`; figure
`AuMgO_single_all_mac_s43_epoch6_nac_vs_pbe.png`, not released (`plot_pbe_nac.py`, new: `plot_pbesol_nac.py` with the PBE reference;
model and DFT only).

| checkpoint | g | Z*(Mg) | ε∞ | TO (373.3) | LO (681.3) | RMS no NAC / NAC | passed |
|---|---|---|---|---|---|---|---|
| frozen start (step 31 weights, every term on) | 1.000 | 1.927 | 3.471 | 443.4 (+18.8 %) | 692.5 | 65.9 / 61.8 | 4/7 |
| epoch 1 | 0.9764 | 1.933 | 3.405 | 386.4 (+3.5 %) | 663.3 | 40.6 / 30.7 | 4/7 |
| epoch 4 | 0.9541 | 1.940 | 3.341 | 367.0 (−1.7 %) | 657.8 | 33.5 / 17.0 | 5/7 |
| epoch 6 | 0.9577 | 1.938 | 3.352 | 385.6 (+3.3 %) | 667.4 | 32.1 / 16.6 | 5/7 |

Epochs 2, 3 and 5 were not new bests and have no checkpoint (`train.py` saves the best and, at the end, the last).
Training / validation force RMSE (eV/Å, slabs and bulk cells together): 0.222 / 0.162, 0.132 / 0.140, 0.133 / 0.173,
0.128 / 0.132, 0.107 / 0.152, 0.124 / 0.111 at epochs 1–6. One transient non-finite gradient (epoch 1, structure 424, a
bulk cell), finite on recomputation; no refusal.

**Where the epoch 6 error sits (sorted branches, 204 path points).** With NAC, per branch: 19.4, 19.1, 10.0, 16.6, 16.9,
15.9 cm⁻¹. The largest single error is **TA(X) 237.4 against 280.6 (−43.2 cm⁻¹, −15 %)**; X is commensurate with the
supercell, so it is not interpolation. It is the acoustic softening recorded for step 28b (TA(X) 196–272). At X the model
has 237, 237, 406, 439, 439, 549 against 281, 281, 406, 425, 425, 528; at L 265, 265, 345, 345, 535, 567 against 262, 262,
344, 344, 529, 550.

**P5 against PBE compares two different supercells (measured today; a correction to how this file reads P5).** The PBE
reference's no-NAC band is built from fc2 on Togo's 4×4×4 supercell; the model's from the 2×2×2 conventional supercell
(protocol above). Without NAC the top branch between Γ and the first commensurate point is pure Fourier interpolation,
and it depends on the supercell. Top branch along Γ–X at points 0, 6, 13, 19, 25 of 51 (cm⁻¹; point 25 is commensurate
with both, and there NAC = no NAC on every row):

| | with NAC | without NAC |
|---|---|---|
| model, epoch 6 (2×2×2) | 667, 667, 662, 648, 622 | 386, 422, 516, 587, 622 |
| PBEsol reference (phonondb) | 694, 688, 667, 640, 607 | 391, 423, 506, 572, 607 |
| PBE reference (fc2 on 4×4×4) | 681, 676, 656, 630, 598 | 373, 494, 665, 675, 598 |

The model's no-NAC shape is the PBEsol reference's, not the PBE reference's. Against PBE the model's top branch alone has
RMS 63.5 without NAC and 15.9 with NAC; the three largest no-NAC deviations (−148.7, −136.3, −134.0 cm⁻¹) are all on
the top branch about a quarter of the way from Γ (Γ–X, K–Γ, Γ–L). The five lower branches have 20.7 without NAC and 16.8
with. Consequences, stated without changing any pre-registered criterion:
- every P5 number against PBE in this file (29–46 cm⁻¹: step 20, the bulk-augmented runs, step 28b, this run) contains
  this mismatch, and the sentence "P3, P5 are the short-range force constants" holds for P5 only where the two supercells
  match (the PBEsol reference: 1.7–8.1 cm⁻¹ in this file). No 2×2×2 model here has reached P5 against PBE.
- the with-NAC band (P6) is the like-for-like comparison against PBE, and the lower five branches without NAC are a
  second one;
  **[Superseded 2026-11-16; the bullet above is kept as written.]** P6 also moves with the supercell: +1.1 to +2.2 cm⁻¹
  from 2×2×2 to 4×4×4 on five step 34b checkpoints ("Scored (2026-11-16 ...)" and the amendment at the end). The
  like-for-like P6 is the 4×4×4 value.
- step 33's S4 ("P0, P1, P2, P4 pass, and at least 6 of 7") stays attainable only with P3 and P6 both passing; it is
  scored as written.
- NOT DONE: the model's force constants on the 4×4×4 primitive supercell (128 atoms), which would make P5 like-for-like;
  `mgo_phonons.py --supercell` takes conventional repeats only.
  **[Corrected 2026-11-15 22:10 IST; the sentence above is kept as written.]** Togo's fc2 supercell is 4×4×4 of the
  *conventional* 8-atom cell, 512 atoms. The evidence: `phonon_supercell_matrix` diag(4, 4, 4) in
  `$TRACE_PHONON_RAW/phono3py_pbe_mgo/phono3py_params.yaml`, against a primitive_matrix that is FCC relative to
  that cell; the reference's two fc2 cells hold 512 atoms each (item 3 above). So the like-for-like calculation is
  `score_phonons.py --supercell 4`, which `mgo_phonons.py`'s conventional repeats already allow. A 4×4×4 primitive
  supercell (128 atoms) would be a third supercell and not like-for-like. Still NOT DONE: its cost on 512 atoms
  (closures, Fermi band) is unmeasured.
  **[Superseded 2026-11-16; the sentence above is kept as written.]** Measured on step 34b's s42 ep21: 61 s and 6.5 GB
  peak, one configuration per displaced cell ("Scored (2026-11-16 ...)").

**Single model, Mac early look, epoch 9 (2026-11-12 18:51 IST; read at 20:00):** g 0.9527; **4/7** — Z*(Mg) 1.940,
ε∞ 3.337, TO 360.5 (−3.4 %, P3 fails), LO 654.5 (P4 fails by 5.5), path RMS 32.7 without NAC / **12.9 with NAC**.
Validation force RMSE 0.1055 eV/Å at epoch 9 (0.120, 0.138, 0.106, 0.111, 0.123 at epochs 7–11). **The crystal factor
and ε∞ have been flat since epoch 4:** g 0.9541, 0.9577, 0.9527 at epochs 4, 6, 9; the training's own ε∞ 3.341, 3.331,
3.352, 3.346, 3.344, 3.337, 3.337, 3.346 at epochs 4–11 (target 3.235). The bulk-only run with the same prior and weight
reached g 0.894 and ε∞ 3.241 by epoch 9. √(LO² − TO²) = 546 against PBE's 570: Z*²/ε∞ is 8 % low (1.940²/3.337 against
1.9923²/3.235). HYPOTHESIS, not measured: the slab energies and forces now pull on g against the ε∞ prior (weight 0.05);
the check is the gradient of each loss term on log g at this checkpoint (`probe_alpha_scale_gradient.py` has the prior
terms only).

## Pre-registered: which loss term holds the crystal factor in the single model (2026-11-12 21:14 IST; the measurement decided the same evening)

**Checkpoint:** the Mac early look's best at 21:02 IST, epoch 14 (copy `benchmarks_4g/runs/AuMgO_single_all_mac_s43_epoch14`):
g 0.9482; 5/7 against PBE (Z* 1.941, ε∞ 3.324, TO 366.2 (−1.9 %), LO 658.8 — P4 fails by 1.2 —, RMS 32.7 / 12.1);
validation force RMSE 0.0952. g at the scored checkpoints: 0.9764, 0.9541, 0.9577, 0.9527, 0.9557, 0.9482 (epochs 1, 4,
6, 9, 12, 14). No training, no package change; frozen weights.

**Probe:** `probe_alpha_scale_terms.py` (new): ∂/∂log g of every term of `train.py`'s optimizer step, with `train.py`'s
own `energy_loss`, `force_loss`, `_response_prior_loss` and normalisation: a step is (1/8) Σ over the window's 8
structures of [10 · energy term + force term (bulk cells × 122.4)] + 0.05 · dielectric prior + 0.3 · polarizability
prior, then the global norm is clipped to 1 and AdamW steps. So the step's expected data gradient is the mean over the
436 training structures of the per-structure terms, **D**; the window's noise is σ_w = std / √8. Sign: + lowers g.

**Seen before writing this** (a smoke test on three validation structures and the checkpoint's optimizer state):
- the probe's own check: autograd against a central difference in log g agrees to 6–7 digits on a slab, a doped slab
  and a bulk cell;
- dielectric prior at this checkpoint: **P_ε = +8.013e-03** (model 3.3243 against 3.2350); polarizability prior:
  exactly 0 (SPEC 35.1 holds: the Au atom no longer sees g);
- AdamW's own record of log g (group `adamw_aux_decay`, lr 7.41e-04, betas 0.9 / 0.95, step 770): m̂ = +9.8e-04,
  √v̂ = 1.26e-02, ratio +0.078; weight decay's step 4e-10 (negligible);
- three structures say nothing about D (+9.1e-03 ± 2.4e-02).

**Readings, fixed now** (D on the 436 training structures; P_ε as measured there):
- **(A) a data term holds g:** D ≤ −0.5 P_ε. Then the holder is named: the class (slabs without Al / slabs with Al /
  bulk cells) and the term (energy / force) with the largest negative share of D.
- **(B) nothing holds g; the prior's pull is buried in the windows' noise:** D > −0.5 P_ε and σ_w ≥ P_ε.
- **(C) neither:** D > −0.5 P_ε and σ_w < P_ε; the loss terms do not explain the stall (clipping, the schedule).
- Undecided between (A) and (B) if the standard error of D exceeds 0.25 P_ε.

**Prediction: (A), with D between −1.2 and −0.5 of P_ε.** Reason, from numbers already on record: log g fell by 0.0062
over epochs 4–14 (0.0006 per epoch), while the prior alone against the optimizer's recorded noise (after clipping,
roughly 5e-03 against 1.26e-02, ratio ~0.4) would move it by lr × 55 steps × 0.4 ≈ 0.018 per epoch, thirty times faster.
So the mean of the data terms must cancel most of the prior. Which class does it is not predicted.

**Check of the probe on the validation set (86 structures), written before it runs:** train.py's metrics from the probe
must reproduce the training log's validation line of epoch 14: E RMSE 9.88 meV/atom, offset-free 2.72, force RMSE
0.095216 eV/Å, each to the printed digits. If not, the probe's loss is not the trainer's and the reading is void.

Outputs: `runs/alpha_scale_2026-11-12/terms_single_mac_s43_epoch14_{valid,train}.{txt,npz}`.

## Scored: the crystal factor is held by the slab data, at a balance point of the loss (2026-11-12 22:08 IST)

`runs/alpha_scale_2026-11-12/terms_single_mac_s43_epoch14_{valid,train}.{txt,npz}`; frozen epoch-14 weights (g 0.9482).

**The probe's checks.** (i) Validation set, train.py's metrics from the probe: E RMSE 9.88 meV/atom, offset-free 2.72,
force RMSE 0.095216 eV/Å — the training log's epoch-14 validation line to every printed digit. (ii) Autograd against a
central difference in log g: 6–7 digits on a doped slab, an undoped slab and a bulk cell, in both sets.

**The training set (the pre-registered measurement; 436 structures; one optimizer step, before clipping; + lowers g):**

| term | share of the step's gradient on log g | sign over the structures |
|---|---|---|
| dielectric prior (weight 0.05; model 3.3243 against 3.2350) | **+8.013e-03** | |
| polarizability prior (weight 0.3) | 0 exactly | |
| undoped slabs (150), force term | **−6.909e-03 ± 0.19e-03** | negative in 150 of 150 |
| undoped slabs, energy term | −0.809e-03 ± 0.01e-03 | negative in 150 of 150 |
| doped slabs (150), energy term | **−4.210e-03 ± 0.05e-03** | negative in 150 of 150 |
| doped slabs, force term | +2.902e-03 ± 1.4e-03 | negative in 65 of 150 |
| bulk cells (136), force term (weight 122.4) | +0.772e-03 ± 0.05e-03 | positive in 115 of 136 |
| **data, D** | **−8.255e-03 ± 1.47e-03** | |
| **D + priors** | **−0.24e-03 = −0.03 ± 0.18 of the prior** | |

σ_w (the window's noise) 1.08e-02 = 1.35 of the prior. On the validation set the undoped slabs give the same per-structure
pull (force term −2.12e-02 against −2.01e-02 on the training set, negative in 38 of 38); its D is −2.04e-02 ± 0.50e-02,
the difference being the doped slabs' force term, which is noise-dominated in both sets.

**Verdict: reading (A), and the prediction held** (D / P_ε = −1.03 ± 0.18; predicted −1.2 to −0.5; the standard error
is below the 0.25 limit). g is not stuck in the optimizer: it sits where the data's pull cancels the prior's, to 3 % ±
18 %. By the rule fixed beforehand the holder is **the force term of the undoped slabs** (−6.9e-03, 86 % of the prior's
pull by itself, the same sign in all 188 undoped slabs of both sets).

**The second holder is the energy offset, not polarizability physics (measured).** The energy term carries 61 % of D
(−5.02e-03), and it acts through a uniform offset: the model's slab energies are +9.49 meV/atom high on average (E RMSE
9.85, offset-free 2.66), and a larger g lowers every slab's energy. With the mean offset removed from each structure's
error the energy term would be +1.15e-03 (opposite sign, small). The doped slabs feel it most (−1.22e-02 per structure
against −2.35e-03 undoped: their energy is about eight times more sensitive to g).

**What is not measured.** Why the undoped slabs' forces want more polarizable ions. Two candidates, not separated:
(i) ESTABLISHED physics that one global g cannot hold: an O²⁻ ion at the surface is less confined than in the bulk and
more polarizable (Fowler and Madden, in-crystal polarizabilities); (ii) compensation: the backbone at epoch 14 of the
first stage has not absorbed SPEC 34's damping, and g makes up for it. The check is the same gradient resolved by atom
(Au, surface layer, deeper layers); NOT DONE.

**Consequences (arithmetic on the numbers above; hypotheses until run).**
- To first order, at frozen weights, moving g from 0.948 to the bulk-only run's 0.894 raises the data loss per step by
  4.9e-04, 5.8 % of the epoch-14 training loss (0.0084): the slabs' preference is weak, and the prior at weight 0.05
  is weaker.
- With D unchanged, a dielectric-prior weight of about 0.23 would balance at ε∞ within 0.02 of the label.
- In the second stage the energy weight goes from 10 to 1000; the energy term's pull on g then depends on the residual
  energy errors and is not predicted here. Step 33's S5 (g within 0.85–0.95, ε∞ within 0.05 of 3.235) is at risk on
  the present balance (g 0.948, ε∞ 3.324).

### Pre-registered: which atoms of the undoped slabs pull the crystal factor up (written 2026-11-13 00:17 IST, before the run; frozen weights, a probe)

**Benchmark numbers it serves:** step 33's S5 (g within 0.85–0.95, the prior cell's ε∞ within 0.05 of 3.235) and P4
(LO 660–700 cm⁻¹; 659 at the last two reads because ε∞ sits at 3.33). The entry above measured that the undoped
slabs' force term cancels the ε∞ prior (−6.909e-3 of the step's gradient, 150 of 150 slabs with the same sign).
**Measurement (`probe_alpha_scale_atoms.py`, new; the epoch-14 copy, the 150 undoped training slabs).** The force term
of a structure is Σ_i w |F_i − F_i^DFT|² / (3N), a sum over atoms, so its derivative in log g splits by atom exactly:
c_i = 2 w (F_i − F_i^DFT)·(∂F_i/∂log g) / (3N), with ∂F/∂log g by a central difference in log g (h = 1e-4; the entry
above found the difference and autograd agree to 6–7 digits). Check per structure: Σ_i c_i against the stored
autograd value of that structure (`terms_single_mac_s43_epoch14_train.npz`), pass within 1e-4 relative. Atoms are
grouped as: the two Au; the layer facing the dimer (O, Mg); layers 2 to 5; the far, bare surface layer (O, Mg).
**Readings (fixed now), on the shares of the 150-slab sum.**
- **(S) the surface as such:** the two surface layers carry ≥ ½, and the bare far surface pulls per atom with the same
  sign and at least ⅓ of the facing layer's size. A surface ion wants a larger polarizability than a bulk ion
  (ESTABLISHED as physics: the in-crystal polarizability of O²⁻ follows its coordination, Fowler and Madden, Phys. Rev. B
  29, 1035 (1984), cited from memory); a surface-aware factor would be supported.
- **(C) the contact:** the Au atoms and the facing layer carry ≥ ½ and the far surface less than ⅓ of the facing
  layer per atom. The pull is the adsorbate's contact, not the surface; a surface-aware factor would be the wrong fix.
- **(B) the slab's interior:** layers 2 to 5 carry ≥ ½. The slabs as a whole disagree with the bulk label (their
  lattice constant is 4.265 Å against the label's 4.2557).
- Mixed otherwise.
**Prediction (hypothesis).** (C): the force errors of this set sit on the Au atoms and their contact, so that is where
g has leverage on the loss. **Falsified if** the far surface pulls with at least ⅓ of the facing layer per atom, or
if the interior carries ≥ ½. **Limit:** this resolves the force term only; the energy term of a structure is one
number and cannot be split by atom.

### Scored: the surface layers pull the crystal factor up, the bare face too; the Au atoms do not (2026-11-13 00:54 IST)

Output `runs/alpha_scale_2026-11-12/atoms_single_mac_s43_epoch14_train.{txt,npz}` (`probe_alpha_scale_atoms.py`; the
150 undoped training slabs, 1280 s on 2 threads). The sum over atoms reproduces the entry above: −2.0081e-2 per slab,
−6.9087e-3 of a 436-structure step (stored: −6.909e-3); 150 of 150 slabs negative.

**The check, by its letter: FAIL on 2 of 150 structures** (relative gap to the stored autograd value 2.1e-4 on
structure 76 and 1.9e-4 on structure 21, line 1e-4; median 1.5e-6, 90th percentile 9.1e-6; the set's sum agrees to
2.0e-6). Followed up before reading (h = 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
- structure 76: the difference is converged (−1.782587e-2 at every h down to 3e-5) and autograd gives −1.782966e-2.
  The gap is not truncation; autograd and the function's own derivative differ by 2.1e-4 there. Cause not found.
- structure 21: the difference scatters as h falls (gap 4.4e-5, 1.1e-4, 1.9e-4, 3.0e-4, 3.8e-4): noise in the
  forces of the size of the solver's tolerance. One of its evaluations took 217 s against 5 s.
- Neither can move a share below by more than 1e-5 of the total. The reading stands; the two structures are named.

| atoms | per slab | share of the pull | per atom | part along z | rms force error (eV/Å) | rms ∂F/∂log g (eV/Å) |
|---|---|---|---|---|---|---|
| the two Au | −2.3e-5 | **0.1 %** | −1.2e-5 | | 0.197 | 0.019 |
| facing layer, O (9) | −7.16e-3 | 35.7 % | −7.96e-4 | 102 % | 0.114 | 0.521 |
| facing layer, Mg (9) | −7.61e-3 | 37.9 % | −8.46e-4 | 101 % | 0.154 | 0.393 |
| layer 2 (18) | +3.9e-4 | −2.0 % | +2.2e-5 | | 0.061 | 0.128 |
| layer 3 (18) | +2.8e-4 | −1.4 % | +1.5e-5 | | 0.035 | 0.103 |
| layer 4 (18) | +4.6e-4 | −2.3 % | +2.6e-5 | | 0.032 | 0.101 |
| layer 5 (18) | +1.9e-4 | −1.0 % | +1.1e-5 | | 0.055 | 0.126 |
| bare far surface, O (9) | −3.37e-3 | 16.8 % | −3.75e-4 | 101 % | 0.057 | 0.516 |
| bare far surface, Mg (9) | −3.24e-3 | 16.1 % | −3.60e-4 | 100 % | 0.062 | 0.388 |

The two surface layers carry 106.5 %, Au + the facing layer 73.7 %, layers 2 to 5 −6.6 %. Per atom the bare far
surface pulls with **0.448** of the facing layer (O alone 0.47).

**Reading, by the fixed rules: (S), the surface as such. The prediction (C) is falsified** (its line: the far surface
below ⅓ of the facing layer per atom; measured 0.448).
- The crystal factor has no leverage through the Au atoms: their forces respond to it 27 times less than a surface
  oxygen's (0.019 against 0.52 eV/Å per unit log g) and they carry 0.1 % of the pull.
- The pull is the force normal to the surface on the outermost layer of ions, O and Mg alike, on both faces. Both
  faces respond to g equally (0.52 / 0.39 on each); the facing layer pulls twice as hard because its force error is
  twice as large (0.11–0.15 against 0.06), not because it responds more.
- The interior (72 atoms) pulls weakly the other way, the direction of the ε∞ prior.

So one number g is asked for two things that the data disagree on: the interior and the bulk label want it lower
(ε∞ 3.235 needs about 0.89), the surface layers want it higher, and it rests at 0.948 where the two cancel. That is
what a surface ion of larger polarizability than a bulk ion would look like (ESTABLISHED as physics: Fowler and
Madden, cited above). **It is not yet shown to be that:** a larger g is also the only handle the fit has on any
systematic error of the normal force on surface ions, whatever its origin.
**NOT MEASURED:** the sign and size of the mean normal force error on the surface ions (it would say whether the
model's surface layer sits too far in or out against DFT); the facing layer resolved by distance from the dimer;
the doped slabs (their force term pulls the other way, +2.9e-3) and the energy term, which cannot be split by atom.
**What it supports (hypothesis, nothing built):** a factor for the under-coordinated surface ions separate from the
bulk one (an equation change: SPEC 35.2, a branch, tests first), in preference to a stronger ε∞ prior, which would
buy ε∞ at the cost of the surface forces (first order: about 5.8 % of the data loss at frozen weights, entry above).

### Single model, Mac early look (seed 43, stage A): the reads at epochs 19 to 25 (recorded 2026-11-13 06:55 IST; from `benchmarks_4g/runs/watch_single_mac_phonons.log`; mid-training, one seed, not step 33's verdict)

| best checkpoint (epoch) | g | Z*(Mg) | ε∞ | TO (cm⁻¹) | LO | RMS without NAC / with NAC | passed |
|---|---|---|---|---|---|---|---|
| 19 | 0.9494 | 1.940 | 3.328 | 367.4 (−1.6 %) | 659.1 | 30.8 / 9.2 | 5/7 |
| 21 | 0.9456 | 1.941 | 3.317 | 368.8 (−1.2 %) | 660.9 | 31.2 / 8.5 | 6/7 |
| 22 | 0.9468 | 1.941 | 3.320 | 376.1 (+0.8 %) | 664.7 | 31.4 / 9.4 | 6/7 |
| 25 | 0.9442 | 1.942 | 3.313 | 372.9 (−0.1 %) | 663.5 | 30.4 / 6.5 | 6/7 |

PBE: Z* 1.9923, ε∞ 3.235, TO 373.3, LO 681.3. From epoch 21 on P0–P4 and P6 pass and only P5 fails (30–31 against
≤ 15; scored as written, see "what P5 measures" above: the like-for-like 4×4×4 calculation is NOT DONE, and at epoch
6 the five lower branches alone were at 20.7). **Margins against step 33's criteria, as they stand on this seed:**
P4 passes by 0.9 to 4.7 cm⁻¹ (LO 660.9–664.7 against the 660 edge; it failed at epochs 4, 9, 14, 19), so S4 rests on
that margin; **S5 is not met** (ε∞ 3.313 is 0.078 from 3.235, the line is 0.05; g 0.9442 is inside 0.85–0.95 by
0.006). Both follow from the balance measured above: g moves by about −0.001 per three epochs.

### Pre-registered: what the surface ions' pull on the crystal factor is made of, an offset or a stiffness (written 2026-11-13 09:29 IST, before the run; frozen weights, a probe)

**Why now.** The decision of 2026-11-13: every update must be general physics for all systems, not a fix for one.
A "surface factor" is not general. The general statement would be one law for an ion's polarizability as a function of
its confinement by its neighbours (free ion, surface ion, bulk ion of any crystal). Before any such law is designed,
this run tests whether the pull measured above is a response property at all. **Benchmark numbers:** S5 and P4, as
above.
**Measurement (`probe_alpha_scale_atoms.py` with per-atom output, then `analyse_surface_normal_force.py`, new; the
epoch-14 copy, the 150 undoped training slabs).** For every ion of the two surface layers, along the outward normal n:
the force error r = (F − F_DFT)·n, the response s = ∂F·n/∂log g, and the ion's normal displacement u from its own mean
position over the 150 slabs (the data are a fixed geometry plus small displacements). With least-squares fits
r = a_r + b_r u + ε and s = a_s + b_s u + η over each group, the group's pull Σ r s splits exactly into
**offset** N a_r a_s (a constant force error met by a constant response), **stiffness** b_r b_s Σ(u − ū)² (an error of
the normal force constant met by a change of it), and a **remainder** Σ ε η. The lateral components are reported.
**Readings (fixed now), on the four surface groups together.**
- **(offset)** ≥ ⅔ of the pull is the offset part. The fit uses g to move the surface layer's equilibrium height
  (relaxation and rumpling). That is a static force balance at the surface, which any missing surface term would show;
  it does not by itself support a law for the polarizability.
- **(stiffness)** ≥ ⅔ is the stiffness part. The surface ions' normal force constant is wrong and a larger
  polarizability corrects it: a response property, and support for a confinement law.
- Mixed otherwise; the remainder is reported and belongs to neither.
**Prediction (hypothesis).** (offset). Reason: at a surface site the field of the crystal does not vanish by symmetry,
so a change of α changes the mean normal force on every surface ion at once, which is why the response is five times
larger there than in the interior (0.52 against 0.10 eV/Å per unit log g) and equal on the two faces. **Falsified if**
the offset part is below ⅓.

### Scored: the pull on the crystal factor is a static force offset on the surface ions, not a stiffness (2026-11-13 12:07 IST)

Outputs `runs/alpha_scale_2026-11-12/per_atom_single_mac_s43_epoch14_train.{txt,npz}` and
`surface_normal_force_epoch14.txt` (`analyse_surface_normal_force.py`; 150 undoped training slabs, 16500 atom
records; the split is exact, asserted to 1e-9).

**Reading, by the fixed rules: (offset). The prediction held.** Of the four surface groups' pull along the normal
(−2.1595e-2 per slab), **97.6 % is the offset part**, 0.6 % the stiffness part, 1.8 % the remainder.

| surface ions | DFT mean normal force (eV/Å) | model − DFT | response ∂F·n/∂log g | DFT normal force constant (eV/Å²) | model − DFT |
|---|---|---|---|---|---|
| facing layer, O | +0.230 ± 0.014 | **+0.149** ± 0.003 | −0.883 | 7.75 | +0.78 ± 0.15 |
| facing layer, Mg | −0.231 ± 0.004 | **−0.204** ± 0.004 | +0.670 | 7.85 | +1.29 ± 0.19 |
| bare far layer, O | +0.334 ± 0.003 | **+0.069** ± 0.002 | −0.875 | 7.35 | +1.12 ± 0.08 |
| bare far layer, Mg | −0.307 ± 0.003 | **−0.086** ± 0.001 | +0.662 | 7.97 | +1.27 ± 0.07 |
| layers 3 and 4 (interior) | −0.006, −0.007 | −0.020, −0.015 | −0.030 | 11.27, 11.21 | −0.32, −0.36 |

(n is the outward normal; positive = pushing the ion out of the slab.)

- **What the data hold.** The slabs are bulk-terminated: DFT pushes the surface O outward and the surface Mg inward
  by 0.23–0.33 eV/Å (the rumpling of MgO(001), ESTABLISHED, cited from memory). **The model's rumpling force is too
  strong**: by 21 % (O) and 28 % (Mg) on the bare face and by 65 % and 88 % on the face under the dimer.
- **What g does about it.** A larger polarizability weakens the model's rumpling force (O −0.88, Mg +0.67 eV/Å per
  unit log g, the same on both faces). Removing the bare face's error would need g higher by 8 % (O) to 13 % (Mg),
  the facing layer's by 17 % to 30 %; the ε∞ label needs it about 6 % lower. That is the balance at 0.948.
- The stiffness errors exist and point the same way (the surface ions' normal force constant is 10–17 % too stiff,
  the interior's 3 % too soft; a larger g softens both, by 1.0–1.75 and 2.2 eV/Å² per unit log g), but they carry
  0.6 % of the pull.

**Consequence for the design (the project's rule: general physics only).** This does not support a law for the
polarizability of surface ions: the factor is being used as the only available handle on a **static force error at
under-coordinated ions**, which any term acting at a surface could cause. **No polarizability law is built on this
evidence.** NOT MEASURED, and the next step before any design: which term of the model carries the excess rumpling
force (the network part or the charge module, and within it the frame's Coulomb field on the induced dipoles, SPEC
34's damping, whose law was measured for bulk neighbours, or the overlap terms), by the split used for the Au–Au
bond, on the bare face.

## Pre-registered: P5 like-for-like on Togo's 512-atom supercell (written 2026-11-16 ~01:05 IST, before the run; decision of 2026-11-16: run the 512-atom check locally)

**[Corrected after an independent review, 2026-11-16 ~11:05 IST; the heading is kept as written.]**
- **When it was written:** 01:02:59 IST, 4.6 s before the pre-registered run started.
  - Evidence: the time stamp of the edit that wrote the entry, and `runs/AuMgO_twosite_bf055_bondlaw_r5_s42_ep21_sc4/timing.txt`
    (start 01:03:04, end 01:04:05).
  - "~01:05" was an estimate. The Scored heading's "01:04" start is likewise 01:03.

The checkpoint is step 34b's stage A, seed 42, epoch 21: `benchmarks_4g/runs/step34b_peek/AuMgO_twosite_bf055_bondlaw_r5_s42_ep21/model.pt`.
On its 2×2×2 force constants it scored 6 / 7, with P5 30.4 cm⁻¹ and P6 10.7. The run is
`score_phonons.py --supercell 4`, with output to `runs/AuMgO_twosite_bf055_bondlaw_r5_s42_ep21_sc4`.
- **Supercell:** 4×4×4 conventional, 512 atoms. This is Togo's fc2 supercell (`phonon_supercell_matrix`).
- **Unchanged:** Z* and ε∞ are still computed on the 64-atom cell, and every pre-registered criterion is unchanged. This
  is a diagnostic of what P5 measures, not a re-scoring of any step.

**Hypothesis H:** P5's 29–38 cm⁻¹ against PBE is the supercell mismatch (2×2×2 against 4×4×4).

Predictions, fixed before the run:
- **C1 (consistency, must hold or the run is not read):** the frequencies at Γ, X and L without NAC, which are
  commensurate with both supercells, agree with the 2×2×2 run within 1 cm⁻¹. The ensemble guard holds as well:
  population ≥ 1 − 10⁻⁴ and |F − E_frame| ≤ 1 meV.
- **H holds** if P5 on 512 atoms is ≤ 15 cm⁻¹, P5's own threshold.
- **H is refuted** if P5 stays > 20 cm⁻¹. In that case the no-NAC error lies in the model's force constants, not in the
  comparison.
- **Between 15 and 20:** partly.
- **Also recorded:** P6 on 512 atoms, and the RMS between the model's own 2×2×2 and 4×4×4 bands, without and with NAC.
- **Cost:** the time and the number of configurations per 512-atom supercell are recorded; both are unmeasured before
  the run.

**Scored (2026-11-16 01:04–01:20 IST): H holds; P5 against PBE is the supercell mismatch.**
`score_phonons.py --supercell 4` and `compare_sc2_sc4.py` (new; it reads the band yaml files only) were run on three
step 34b stage A checkpoints:
- s42 ep21 is the pre-registered one.
- s42 ep30 and s43 ep29 were added after it as the latest best checkpoints of each seed. They were not pre-registered,
  and all three are reported.

The outputs are in `runs/<run>_sc4/` (`P_phonons_pbe_sc4.txt`, `compare_sc2_sc4.txt`). Phonopy reloads the force
constants with 512 supercell atoms, shape (512, 512, 3, 3).

| checkpoint | C1: max \|2×2×2 − 4×4×4\| at Γ, X, L (cm⁻¹) | P5 no NAC, 2×2×2 → **4×4×4** | P6 NAC, 2×2×2 → 4×4×4 | top branch no NAC, 2×2×2 → 4×4×4 | passes on 4×4×4 |
|---|---|---|---|---|---|
| s42 ep21 (pre-registered) | 0.004 (0.0042) | 30.4 → **11.5** | 10.7 → 11.8 | 68.3 → 19.0 | **7 / 7** |
| s42 ep30 | 0.005 (0.0047; 12 / 12 pass) | 29.4 → **8.2** | 7.3 → 8.5 | 66.5 → 12.0 | **7 / 7** |
| s43 ep29 | 0.001 (0.0011; 12 / 12 pass) | 29.7 → **9.0** | 7.1 → 9.3 | 68.3 → 8.8 | **7 / 7** |

**[Corrected after an independent review, 2026-11-16 ~11:05 IST.]** The C1 cells of s42 ep30 and s43 ep29 read "0.004 or less". Recomputed from the band files, they
are 0.0047 and 0.0011.

- **C1 passes** on every commensurate point of every checkpoint. P0 holds on the 512-atom cells: one configuration,
  population 1.000000, |F − E_frame| 0.0002 eV.
- **Prediction met:** P5 is ≤ 15 on the pre-registered checkpoint, so H holds. On 4×4×4 the no-NAC and NAC RMS agree
  (11.5 / 11.8, 8.2 / 8.5, 9.0 / 9.3), as they should when both sides are built the same way.
- **The NAC band (P6) also moves with the supercell:** 4.9–5.7 cm⁻¹ RMS between the model's own 2×2×2 and 4×4×4 bands.
  - The like-for-like P6 is therefore the 4×4×4 value, 8.5–11.8, slightly above the 2×2×2 one.
  - Every P6 against PBE in this file was read on 2×2×2 and carries this 5 cm⁻¹ uncertainty.
    **[Corrected after an independent review, 2026-11-16 ~11:05 IST; the bullet above is kept as written.]**
    - The ~5 cm⁻¹ is the RMS between the model's own 2×2×2 and 4×4×4 bands.
    - P6 itself, against PBE, moved by +1.1 to +2.2 on the five step 34b checkpoints measured.
    - For earlier models the change is unmeasured (hypothesis).
- **What remains is the model's own error,** visible at commensurate points:
  - s42 ep21 at X: top branch 490.3 against 528.2 (−38 cm⁻¹), and 401.1 against 425.1 (−24).
  - The largest single deviation on the later checkpoints is about 29–30 cm⁻¹: s42 ep30 near the middle of Γ–L,
    s43 ep29 near the middle of K–Γ.
  - **[Corrected after an independent review, 2026-11-16 ~11:05 IST.]** The X values are at commensurate points. The 29–30 cm⁻¹ maxima are at interpolated points
    (path points 175 and 133), on the shared 4×4×4 supercell.
- **Cost:** 61 s for the whole s42 ep21 score on the Mac (4 threads, nice 5), with 6.5 GB peak resident memory.
- **Unchanged:** no pre-registered criterion. S4 of step 34 / 34b is scored as written, on 2×2×2. The 4×4×4 numbers are
  recorded alongside.
- **Figure** (model against DFT only, both without NAC; top row 2×2×2, bottom row 4×4×4):
  `step34b_A_s42ep30_s43ep29_nonac_sc2_vs_sc4.png`. The no-NAC peak near Γ along Γ–X (PBE 686 cm⁻¹) is the 4×4×4
  supercell's own interpolation, and the model reproduces it only on the same supercell.
- **Next (decision of 2026-11-16: report both in the final scoring):** step 34b's final checkpoints are scored on
  both supercells by `benchmarks_4g/score_step34b_sc4.sh`; record and stage A's final numbers in the benchmarks_4g
  README, "Step 34b's final scoring reports both supercells".

**Why P5 depends on the supercell: where the two supercells are exact (2026-11-16 ~09:05 IST; `compare_sc2_sc4.py`
now prints this breakdown, so the final 512-atom scoring records it).**
- **The checkpoints:** paired numbers are s42 ep30 / s43 ep39.
  - s43 ep39's 4×4×4 bands come from a test run of `score_step34b_sc4.sh` outside the repository. Step 34b's final
    512-atom scoring writes them to `runs/`; its checkpoint has the same md5.
- **Exact points:** a finite-displacement supercell gives the dynamical matrix exactly only at the q that fit it
  (commensurate q). Everywhere else the band is a Fourier interpolation of the folded force constants.
- **On the 204-point path:**
  - 7 points fit the 2×2×2 conventional supercell: Γ three times, the middle of Γ–X, X twice, and L.
  - 3 more fit only 4×4×4.
  - The other 194 fit neither.

| path points (no NAC) | n | max \|2×2×2 − 4×4×4\| (cm⁻¹), s42 ep30 / s43 ep39 | RMS vs PBE, 2×2×2 | RMS vs PBE, 4×4×4 |
|---|---|---|---|---|
| exact on both | 7 | 0.005 / 0.001 | 7.7 / 11.1 | 7.7 / 11.1 |
| exact on 4×4×4 only | 3 | 72.5 / 84.8 | 28.4 / 29.2 | 9.4 / 10.0 |
| interpolated on both | 194 | 152.1 / 161.7 | 29.9 / 31.1 | 8.2 / 9.8 |

- **Where the model is exact on both supercells,** its frequencies agree to 0.005 cm⁻¹ and its error against PBE is
  the same on both. All of the 2×2×2 excess is in the interpolation.
- **The mechanism, along Γ–X, on the top branch without NAC:**
  - **At Γ:** a periodic supercell has no macroscopic field, so the mode sits at TO (PBE 373.3; model 381.7 / 366.2).
  - **At the first commensurate q ≠ 0:** the field of the polarisation wave is fully in the supercell forces, so the
    mode is LO-like. The first such q is at t = 1/4 of Γ–X on 4×4×4 and at t = 1/2 on 2×2×2.
  - **The interpolation** ramps between the two:
    - PBE (4×4×4) peaks at 686.4, at t = 0.32;
    - the model on 4×4×4 peaks at 675.8 / 685.4, at t = 0.34;
    - the model on 2×2×2 peaks at 604.8 / 608.8, at t = 0.54.
  - **At t = 0.2:** PBE 608.5; the model 462.0 / 455.9 on 2×2×2 and 598.9 / 602.1 on 4×4×4.
  - **At t = 0.5 and at X (commensurate with both):** no NAC and NAC give the same value.
    - t = 0.5: PBE 597.9; model 602.1 / 606.4.
    - X: PBE 528.2; model 519.6 / 511.6.
- **With NAC** (Gonze–Lee), the dipole part is handled analytically and the ramp is gone. The supercell then changes the
  band by about 5 cm⁻¹ RMS (25 at most): the short-range force constants beyond the 64-atom cell are folded.
- **Hence** P6 read 7–11 on 2×2×2 throughout, while P5 stayed near 30.
- **[Corrected after an independent review, 2026-11-16 ~11:05 IST; the two bullets above are kept as written.]**
  - The maximum is 25.6 (s42 ep30).
  - The stated cause, folding, is a hypothesis: no probe separated it from other non-dipolar terms.
  - "7–11 throughout" holds only for the five late step 34b checkpoints analysed (7.1–10.7). Step 34b's earlier best
    checkpoints read 11.6–25.8 on 2×2×2 (benchmarks_4g README, early-look tables).

## Pre-registered: P5 on PBE's own 512-atom supercell for every step after 34b (written 2026-11-16 10:31 IST; decision of 2026-11-16: pre-register P5 on 512 atoms for future steps)

- **Scope:** every step pre-registered after this entry (step 35 on), scored against PBE.
  - Steps up to 34b keep the protocol they were registered with: step 34b's S4 is read on 2×2×2, and its 512-atom
    numbers are recorded alongside ("Step 34b's final scoring reports both supercells", benchmarks_4g README).
  - Nothing already scored is rescored or changes its verdict.
- **P5, as now read.**
  - The quantity: path RMS without NAC against PBE without NAC (Togo, fc2 on the 4×4×4 conventional supercell,
    512 atoms).
  - The model's force constants are on the same 4×4×4 conventional supercell.
  - The threshold is unchanged: ≤ 15 cm⁻¹.
- **C1, a validity condition of P5.**
  - The test: the model's bands without NAC from 2×2×2 and from 4×4×4 agree within 1 cm⁻¹ at every path point
    commensurate with 2×2×2. These are Γ (three times), the middle of Γ–X, X (twice) and L; both supercells are exact
    there.
  - If C1 fails, P5 is FAIL. Measured on step 34b: 0.001–0.005 cm⁻¹.
- **P0** must hold on the displaced cells of both supercells.
- **Unchanged:**
  - P1–P4 and P6 are read on 2×2×2 as before. Z* and ε∞ come from the 2×2×2 cell (`--cell 2`). TO and LO at Γ are the
    same on both supercells, since Γ is commensurate with both.
  - P6 on 4×4×4 and P5 on 2×2×2 are printed as records, not criteria.
  - P6 stays on 2×2×2 (the decision named P5 only). Its 4×4×4 value was 0.3–1.2 cm⁻¹ higher on step 34b's checkpoints;
    moving it too would be a separate decision.
- **Why:** this README, "P5 against PBE compares two different supercells" and "Why P5 depends on the supercell".
  - Off the commensurate points, the band without NAC is the supercell's own Fourier interpolation.
  - PBE's reference carries the 512-atom interpolation.
- **Command:** `score_phonons.py MODEL --p5-supercell 4`. It writes `<out>/` (2×2×2) and `<out>_sc4/` (4×4×4).
  - Without the flag the scorer is the protocol of steps up to 34b, line for line. Step 34b's pending final scoring and
    every older watcher call it that way, so the default stays until step 34b is scored.
  - A step registered after this entry must pass the flag explicitly. Its pre-registration says so.
- **Cost:** a second `mgo_phonons.py` run. Measured: 61 s and 6.5–6.9 GB peak on the Mac (4 threads); 98 s with the
  test suite running beside it.
- **PBEsol reference** (`--ref pbesol`, phonondb mp-1265): its fc2 supercell is not checked here (`fc2_supercell: None`
  in `REFS`). The flag is pre-registered for PBE only.
- **Tests:** `tests/test_score_phonons_protocol.py`, 6 tests, written red first. 5 failed on the old scorer, and the
  legacy-output test passed on it as designed.
  - The legacy output, line for line.
  - The commensurate points by name: 7 on 2×2×2, plus K twice and the middle of Γ–L on 4×4×4.
  - P5 read on 4×4×4 while P6 stays on 2×2×2.
  - C1 failing fails P5 even when the RMS passes.
  - P0 on both supercells.
  - A `--p5-supercell` that does not contain 2×2×2 is refused before any run.
  - The tests use the reference's own bands with known shifts, so every expected RMS is closed-form.
- **End-to-end check (2026-11-16 10:29–10:31 IST):** stage A's final s42 checkpoint (epoch 39), scored with the flag,
  reproduces the hand-run numbers. 6 / 7 (P3 +3.1 %); P5 8.4 with C1 0.005; P6 7.5; records P5 on 2×2×2 29.3 and P6 on
  4×4×4 8.6.

**Amendment after an independent review (2026-11-16 ~11:05 IST).** The review covered regression, physics, tests and
records. The entry above is kept as written; these corrections apply.
- **P6 on 4×4×4 is 1.1–2.2 cm⁻¹ higher, not "0.3–1.2".**
  - Per checkpoint: s42 ep21 10.7 → 11.8; s42 ep30 7.3 → 8.5; s43 ep29 7.1 → 9.3; s42 ep39 7.5 → 8.6;
    s43 ep39 8.1 → 10.1.
  - The 0.3 was the P5–P6 gap on 4×4×4.
  - P6 read on 2×2×2 is therefore optimistic by up to 2.2 cm⁻¹ against the like-for-like value.
  - P6 stays on 2×2×2; moving it is a separate decision.
- **The flag is accepted only for the reference's own supercell.**
  - PBEsol's reference (phonondb mp-1265) was checked: `supercell_matrix` diag(2, 2, 2) of the 8-atom cell, 64 atoms,
    with 0.01 Å displacements. The default protocol already matches it, so `--ref pbesol --p5-supercell 4` is refused.
  - `REFS` records `fc2_supercell` 4 for PBE and 2 for PBEsol.
- **One difference from Togo's PBE set remains: the displacement amplitude.**
  - The model uses 0.01 Å; PBE used 0.03 Å (`phono3py_params.yaml`). The effect on P5 is unmeasured.
  - Hypothesis: small, because the rocksalt site's inversion symmetry makes the one-sided displacement a central
    difference, with an O(u²) error.
  - Matched, as checked in the review: a, masses, phonopy 2.17.2, produce and symmetrise force constants,
    Gonze–Lee NAC.
- **TO and LO** are read at |q| = 10⁻³ next to Γ along [100], not at Γ itself (`mgo_phonons.py`). They agree on both
  supercells to the printed 0.1 cm⁻¹ (measured on s42 ep30 and s43 ep39).
- **The default is permanent:** without the flag the scorer stays the protocol of steps up to 34b (not "until step 34b
  is scored"); later steps pass the flag.
- **Cost:** the 98 s is the whole flagged run, measured by `time` with the test suite beside it. Of that, the 4×4×4 run
  took ~84 s, from the output files' times.
- **Code changes from the review.**
  - The flag must equal the reference's `fc2_supercell` and be a larger multiple of `--supercell` ≥ 1. Otherwise it is
    refused before any run.
  - `<out>_scN` is a sibling of `<out>` even when `--out` ends in a slash.
  - The q paths of both supercells' band files must equal the reference's; otherwise a ValueError.
  - `commensurate()` takes a from the primitive cell's volume, whatever the order of its vectors.
  - The legacy output stays byte-identical to HEAD's: 16 cases (`--ref` pbe and pbesol, `--supercell` 2 and 4, four log
    variants), 0 differences.
- **Tests:** 21 in `tests/test_score_phonons_protocol.py`, up from 6.
  - Why more: the review found 10 of 25 mutants surviving the 6. The new tests target each; they were not re-run
    against those mutants.
  - What they cover:
    - all nine legacy lines;
    - legacy PBEsol labels and the P3 threshold;
    - the full new-protocol output;
    - P0 on each side and for each guard;
    - C1 as a maximum;
    - P5 failing on its RMS when C1 holds;
    - P1–P4 read on the small supercell;
    - the trailing slash;
    - the q-path check;
    - eight refusals.
  - Written red first: 7 failed on the scorer before these fixes and 14 passed; all 21 pass after them.
- **Unchanged:** the criteria and thresholds of the entry above.
  - Re-scored with the fixed code, the five measured pairs give the same numbers: P5 11.5 (s42 ep21), 8.2 (s42 ep30),
    9.0 (s43 ep29), 8.4 (s42 ep39) and 9.8 (s43 ep39), with C1 0.0011–0.0050.

### Step 31's phonons, scored for the first time (2026-11-16 15:57–15:59 IST; asked the same day how step 31's phonons perform)

Step 31 is the best Table II model so far (benchmarks_4g README, "Step 34b scored").
- **Earlier record:** its phonons had never been scored.
  - The benchmarks_4g README's "step 31 family: Z* 2.4–2.5, LO above 800" (step 34's S4 row) was an inference. Its
    numbers are step 20's measured ones (Z* 2.43 / 2.54, LO 817 / 850). Step 31 descends from step 20's option set.
  - The only step 31 number in this file is a frozen probe with every newer term switched on (4 / 7), not the model as
    trained.
- **How it was scored:** `score_phonons.py --p5-supercell 4` on its best checkpoints, s42 epoch 20 and s43 epoch 17,
  loaded as trained. The loader restores SPEC 28's `'block'` stiffness, as for every checkpoint that predates the
  option.
- **Outputs:** `runs/step31_<run>_s{42,43}{,_sc4}/`.

| | Z*(Mg) | ε∞ | TO (cm⁻¹) | LO | P5 no NAC, 4×4×4 (2×2×2) | P6 NAC, 2×2×2 (4×4×4) | C1 | passes |
|---|---|---|---|---|---|---|---|---|
| PBE | 1.992 | 3.235 | 373.3 | 681.3 | | | | |
| step 31 s42 | **2.402** (+21 %) | 3.025 | 384.3 (+2.9 %) | **807.9** | 22.0 (29.6) | 28.9 (28.9) | 0.013 | **3 / 7** (P0, P2, P3) |
| step 31 s43 | **2.534** (+27 %) | 3.002 | 398.7 (+6.8 %) | **851.5** | 47.8 (45.5) | 53.9 (53.8) | 0.049 | **2 / 7** (P0, P2) |
| step 34b B s42, for comparison | 1.950 | 3.334 | 382.1 | 669.3 | 8.1 (29.5) | 7.1 (8.5) | 0.005 | 6 / 7; 7 / 7 on 4×4×4 |
| step 34b B s43 | 1.961 | 3.300 | 370.3 | 667.4 | 8.6 (29.9) | 6.8 (8.8) | 0.001 | 6 / 7; 7 / 7 on 4×4×4 |

- **The LO error is the Z* error.** ω_LO² − ω_TO² scales as Z*²/ε∞. From each seed's own Z*, ε∞ and TO, with PBE's
  splitting as the scale, the predicted LO is 807.8 and 851.6 cm⁻¹, against 807.9 and 851.5 scored.
- **The effective charge is 21–27 % too large.** Hypothesis, not tested: step 31's bulk is formal +2 / −2 ions with too
  little electronic screening. Step 31 trained on the slabs and an ε∞ prior cell, but no bulk force cells. Steps 33
  and 34b added the 136 bulk fc3 cells and SPEC 35.1's crystal factor.
- **The dispersion is also off on the like-for-like supercell:**
  - P5 on 4×4×4 is 22.0 (s42) and 47.8 (s43) cm⁻¹, against 8.1–8.6 for step 34b.
  - s43's top branch at X is 617.4 cm⁻¹, against PBE's 528.2 (from the band file); s42's is 536.2.
- **C1 holds on both seeds** (0.013, 0.049 ≤ 1), so the like-for-like reading is valid. P0 holds: population 1.000000
  on every displaced cell of both supercells.
- **Figure** (model against DFT only, with NAC, 4×4×4): `step31_s42_s43_nac_sc4_vs_pbe.png`.
- **What this means for the release question:** the model best on Table II fails the phonons (2–3 of 7). The model that
  passes the phonons (step 34b) fails Table II. No trained model does both yet.
