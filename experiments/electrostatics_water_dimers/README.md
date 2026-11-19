# Water clusters: TRACE alone vs TRACE + `trace_electrostatics`

> **The data of this experiment are not distributed, and the experiment cannot be rebuilt from public sources yet.**
> Its energies and forces come from `training_examples/water_ccsd/MB-pol-1250.xyz` (48141297 bytes, sha256
> `b1c73572e316f62f9fbf5abbb927941a6d987344906a1887b975aa00e602afb5`; gitignored). The source, authors and licence
> of that file are not documented in this repository, so nothing derived from it ships: neither the splits
> (`data/`) nor the same frames with the PBE0 dipole and polarizability labels added (`data_qm/`). Until the source
> and licence of the file are established, the scripts, the training histories in `results/` and the record below
> are published as they are, and the numbers cannot be reproduced from public data.
>
> With the file in place, the data are rebuilt as follows (commands from this directory):
>
> 1. `python prepare_data.py` writes `data/{train,valid,test}.xyz`.
> 2. The halves `data/train_a.xyz` and `data/train_b.xyz` are frames `[0::2]` and `[1::2]` of `data/train.xyz`. No
>    script writes them; `ase.io.write(..., format="extxyz")` of the two slices reproduces them.
> 3. `python qm_reference.py SPLIT [N_WORKERS] [THREADS]` adds the PBE0/aug-cc-pVDZ labels (PySCF). SPLIT is
>    `valid`, `test` or a file path (`data/train_a.xyz`, `data/train_b.xyz`); it writes `data_qm/<name>.xyz`.
> 4. `python merge_qm.py` joins the two labelled halves into `data_qm/train.xyz`.
>
> Steps 1–2 were rerun on 2026-11-18 (Python 3.8.16, ase 3.23.0, numpy 1.24.2) and gave these files byte for byte:
>
> ```
> 817a9f9ec7c3ac481b325e19fc9be13ac2e6f59d6653ee25a973e581de5de38e  data/train.xyz
> 79457ce25951798a1f414813ebff6253f9b90683a44e16b43017d15f88b2c861  data/valid.xyz
> 312105362a5714d521a6a0675602f9e15b369bd829af74eee34c9a45c2bcd657  data/test.xyz
> ec519ef263751fddebdf320007871120393c29169a2bc593d35e5fcad6428315  data/train_a.xyz
> 70a117e83d4f085d343c0c23884b1602a2738aece9344387fa4f266d8977523a  data/train_b.xyz
> ```
>
> Steps 3–4 were not rerun. The four labelling jobs took 4829 s (test), 1933 s (valid), 5353 s (train_a) and 6911 s
> (train_b) of wall time, from their logs. The labelled files that the wave-3 runs read (file dates 2026-10-01) have
> these sha256:
>
> ```
> aa194a344fe33f808ef3dc2cf2b25bce799273f84b44faaf719ea60bcc1c3be2  data_qm/train.xyz
> 8d9bfe620d9d10cf920c0ae0c3f0f84131b2e5f65e83e7bf3617d73cfd38a4f4  data_qm/valid.xyz
> 250e0c6383ef248dd550ceb0ac46e199bea85a24d012d56c7ad5b4b189f711e6  data_qm/test.xyz
> ```

First run of the electrostatics package on real reference data. **Status: all
four 80-epoch trainings finished on 2026-09-25; the held-out test-set evaluation
has NOT been run yet.** (This status line predates round 4; the test-set results
of round 4 are recorded below.) The checkpoints were written to a local scratch
directory and are not in git (`*.pt` is ignored), so a new machine must retrain
(about 30–50 min per run on 2 CPU threads).

## Data

`data/{train,valid,test}.xyz` (427 / 65 / 166 frames) are the 658 gas-phase
clusters (1–4 waters) of `training_examples/water_ccsd/MB-pol-1250.xyz`. That
48 MB source file is gitignored, and the split is not distributed either (see
the note at the top). It was
converted to open boundaries (`pbc=False`, no cell) with energy and forces in a
`SinglePointCalculator`. The source stores forces as the non-standard per-atom
array `force`. The 573 periodic 256-water boxes and the 19 slabs were dropped.

The split is stratified by cluster size and, for dimers, by the O–O distance
bin `[0, 2.5, 3, 3.5, 4.5, 6, inf)` Å. Fractions: 25 % test, 10 % valid, the
rest train. Seed: 20260925.

| split | 1 water | 2 | 3 | 4 |
|---|---|---|---|---|
| train | 27 | 136 | 56 | 208 |
| valid | 4 | 21 | 8 | 32 |
| test | 10 | 55 | 21 | 80 |

`prepare_data.py` regenerates the split when the MB-pol file is present.

## Protocol

Both arms use the same trunk, optimiser, schedule, seed (42), float64 and epoch
count. The only difference is the `electrostatics:` block (package defaults:
converged solve, Gaussian kernel, dense solver, heads `r_max` = model `r_max`).

`make_configs.py` copies `training_examples/water_ccsd/config.yaml` with three
changes:

- stress weights are 0;
- the removed temperature-force keys are dropped;
- early stopping is off.

Two cutoffs were run: `r_max` = 3 Å, where the local model cannot see most
intermolecular pairs, and `r_max` = 6 Å.

```bash
python make_configs.py r6_base 80 6.0 2
python make_configs.py r6_elec 80 6.0 2 elec
python run_train.py runs/r6_base/config.yaml   # float64 for both arms
python eval_models.py runs/r6_base runs/r6_elec          # -> test_errors_model.json
python compare.py model r6_base r6_elec r3_base r3_elec   # side-by-side + epoch times
python dissociation.py SRC_INDEX runs/r6_base runs/r6_elec  # rigid dimer pull-apart
```

`runs/` is gitignored.

## Results so far

These are validation-set numbers at the best-validation epoch, single seed,
copied from `results/*_training_history.csv`. The energy is `train.py`'s
atom-weighted per-atom RMSE in meV/atom; forces are in eV/Å. Treat them as
preliminary: n = 65 frames, one seed, and the test set is not yet evaluated.

| run | best epoch | val E RMSE | val E RMSE (offset-free) | val F RMSE |
|---|---|---|---|---|
| r3_base | 80 | 16.99 | 16.68 | 0.1172 |
| r3_elec | 72 | 12.71 | 11.34 | 0.1014 |
| r6_base | 79 | 5.94 | 5.80 | 0.0565 |
| r6_elec | 79 | 5.90 | 5.62 | 0.0579 |

Per-epoch wall time comes from dedicated 2-epoch timing runs (2 threads, one run
at a time): base 20.3 s, elec 34.8 s, about 1.7×. The four 80-epoch runs were
launched concurrently, so their wall times are not a timing comparison.

Reading, to be confirmed on the test set before anyone quotes it: at 3 Å the
electrostatics recover part of the missing long-range energy (−25 % energy,
−13 % force RMSE). At 6 Å the two arms are indistinguishable at this sample
size.

## Test-set results, seed 42 (round 4)

Retrained into `runs/` with `run_all.sh`. Seed 42 reproduced the saved
validation curves to the 4 printed digits, with the same best epochs.
`eval_models.py`, 166 held-out structures. **Energy RMSE is per STRUCTURE**
(meV), forces in meV/A. "std" is the error with the mean (bias) removed:

| run | E RMSE | E bias | E std | F RMSE | monomer E RMSE |
|---|---|---|---|---|---|
| r3_base | 150.0 | −54.6 | 139.8 | 124.6 | 62.0 |
| r3_elec | 124.0 | +21.5 | 122.1 | 107.4 | 20.9 |
| r6_base | 49.4 | +12.9 | 47.7 | 59.4 | 2.6 |
| r6_elec | 49.9 | −15.7 | 47.3 | 58.8 | 6.1 |

What this does and does not show:

- **3 A, overall.** Electrostatics reduce the bias-free energy error by 13 %
  (140 → 122) and the force RMSE by 14 %.
- **3 A, distant dimers.** These are the 13 dimers with R_OO >= r_max. The
  headline gain there, 112 → 58 meV RMSE, is almost entirely the monomer bias
  moving (−97 → −2 meV); the bias-free error is unchanged (54.7 vs 57.6 meV).
  So this is NOT evidence that the long-range interaction is learned better.
- **6 A.** The two arms are equivalent.
- **Locality probe.** For dimers with every intermolecular pair beyond r_max,
  the local model's E(AB) − E(A) − E(B) is exactly 0.0000 meV, as a local
  model's must be. The electrostatics model's is −14 to +26 meV, so it does
  carry long-range interaction. On this data, that is not what limits
  accuracy.
- **Dipoles are not determined by energies and forces.** The two
  electrostatics models predict the monomer dipole at the experimental
  geometry as 1.98 D (3 A) and 1.59 D (6 A), against the experimental 1.855 D.
  Hence the dipole-trained arms of wave 3.

Seed 7 (`run_all.sh`, second wave) is needed before any of this is quoted.

## Electrical properties vs PBE0, and the contact law (SPEC 23), seed 42

`eval_electrical.py` on `data_qm/test.xyz` (PBE0/aug-cc-pVDZ). All three
models were trained on energies and forces only:

| model | α/H2O monomer → tetramer (Å³) | dipole RMSE (D) | α RMSE (Å³) | E RMSE (meV/str.) | F RMSE (meV/Å) |
|---|---|---|---|---|---|
| r3_elec | 2.48 → 6.13 | 1.26 | 9.91 | 124.0 | 107.4 |
| r6_elec | 2.52 → 6.27 | 1.20 | 10.07 | 49.9 | 58.8 |
| r6_elec_ct (contact law) | 2.51 → 2.67 | 0.34 | 2.63 | 37.9 | 48.8 |
| PBE0 reference | 1.35 → 1.44 | — | — | — | — |

What the table shows:

- **The law removes the charge-transfer catastrophe.** α per molecule is now
  flat with cluster size, and the cluster dipole error falls 4×.
- **The absolute α is still +86 % too large, in the monomer already.** That
  is intramolecular charge flow along in-contact bonds, which the law leaves
  alone by design and which energies do not determine. Testing whether dipole
  data fixes it is the job of wave 3.
- **The E/F improvement (−24 % / −17 %) is a single seed.** It needs a seed-7
  contact-law run before it is quoted. The wall time of this run is not a
  timing comparison: it shared the CPU with the PySCF job.

## Results across seeds and wave 3 (round 4)

Test set, 166 structures. Energy errors are in meV per structure, force
errors in meV/Å, dipole RMSE in D. α/H2O is in Å³ for monomer → tetramer
(PBE0: 1.35 → 1.44).

| model | loss | E RMSE | F RMSE | dipole RMSE | α/H2O |
|---|---|---|---|---|---|
| r6_base s42 / s7 | E, F | 49.4 / 46.0 | 59.4 / 54.2 | — | — |
| r6_elec s42 / s7 | E, F | 49.9 / 47.6 | 58.8 / 54.7 | 1.20 / 1.30 | 2.52→6.27 / 3.07→7.29 |
| **r6_elec_ct s42 / s7** (SPEC 23) | E, F | **37.9 / 30.3** | **48.8 / 49.0** | 0.34 / 0.40 | **2.51→2.67 / 2.74→2.91** |
| r6_elec_mu s42 | E, F, μ | 40.0 | 59.3 | 0.36 | 2.73→6.05 |
| r6_elec_ct_mu s42 | E, F, μ | 30.8 | 49.1 | **0.17** | 3.07→3.18 |
| r6_elec_mualpha s42 | E, F, μ, α | 58.4 | 71.8 | 1.14 | **1.24→1.51** (α RMSE 0.54) |
| r3_base s42 / s7 | E, F | 150.0 / 152.4 | 124.6 / 125.0 | — | — |
| r3_elec s42 / s7 | E, F | 124.0 / 129.4 | 107.4 / 98.9 | 1.26 / 3.49 | 2.48→6.13 / 2.04→5.80 |
| r3_elec_disp s42 | E, F | 119.6 | 106.5 | 1.28 | 2.81→5.90 |

Findings, with how far each is established:

1. **The contact law is a robust win in both seeds.** α is size-extensive;
   at 6 Å the energy error falls 29 % and the force error 14 % against the
   local-only model, and each law seed beats each local-only seed.
2. **Energies and forces do not determine the electrical outputs** (both
   seeds). The r3_elec dipole RMSE is 1.26 D with one seed and 3.49 D with the
   other, and without the law α grows 2.4× from monomer to tetramer.
3. **Dipole labels alone do not fix α** (single seed). r6_elec_mu fits dipoles
   to 0.36 D, yet α still grows 2.73 → 6.05. Equilibrium dipoles constrain
   the charges, not the curvature of the energy that α measures. α has to be
   trained, or it has to be fixed by priors.
4. **The functional can represent the right α** (single seed). Training α
   gives α within −8 %…+5 % at every size. In that run, though, the dipoles
   (1.14 D) and E/F got worse, and charge transfer still carries 4 % → 47 %
   of α with size. The obvious next arm is the law + μ + α, which has not
   been run.
5. **At 3 Å, electrostatics cut the bias-free energy error by about 15 % in
   both seeds** (std 140 → 122, 147 → 118). Adding dispersion changes little.

## Wave 3 (`run_wave3.sh`): does the response physics transfer?

The training set is labelled with PBE0/aug-cc-pVDZ dipoles and finite-field
polarizabilities (`qm_reference.py`; the halves `train_a`/`train_b` were
computed on two machines and merged by `merge_qm.py`). The arms:

- `r6_elec_mu`: energies, forces and dipoles. Its α is never trained, so
  `eval_electrical.py` comparing that α with PBE0 is the test of whether the
  master equation's response is physical.
- `r6_elec_mualpha`: adds α, as the upper reference.
- `r{3,6}_elec_disp`: adds many-body dispersion from the same moment block
  (SPEC 13).

## Still to do

1. Run `eval_models.py` on the four best checkpoints. Report the test RMSE by
   cluster size and by dimer O–O bin, and the monomer dipole against the
   experimental 1.855 D as a sanity check. Use `compare.py` for the table.
2. Run `dissociation.py` on one test dimer: check the tail against the model's
   own dipole–dipole energy. There is no quantum-chemistry reference along
   that path.
3. Before claiming an effect, run a second seed. The difference at 6 Å is
   within seed noise until shown otherwise.
