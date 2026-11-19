# Reproducing TRACE-Electrostatics: from a clean clone to the published numbers

This page takes you from a clean clone of the `trace-electrostatics` branch to every benchmark number of the release.
There are two routes:

- **Check the released checkpoints (recommended).** Download them, then score them (steps 1–4 and 6). This is the
  reproduction path for the reported numbers. Every number in [BENCHMARKS.md](BENCHMARKS.md) was measured on these
  checkpoints, and scoring is deterministic up to rounding.
- **Retrain them (steps 1–5, then 6).** This rebuilds the models from the data. Retraining is **not bitwise
  reproducible** (section 5.5), so a retrained model will not repeat the published numbers to every digit.

All paths below are relative to the repository root, written `<repo>`, unless a step says `cd` first.

Every expected number on this page is copied from a score file of the published run. The file is named next to the
number as `runs/<run>/scores/<file>.txt`, relative to `experiments/benchmarks_4g/`. Those files are what
`score_model.sh` writes (section 6), so you can compare your output with them line by line.

---

## 1. Environment

```bash
git clone https://github.com/paramvir3/TRACE.git
cd TRACE
git checkout trace-electrostatics
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install --no-deps -e .
python -m pip install phonopy        # phonon benchmark only
```

**The environments of record.** Every reported number was measured in one of two environments:

| | CPU environment (training of steps 26 and 31, the cluster runs on the Mac, and every score) | GPU host (training of steps 33 and 34b, and the carbon-chain protocol runs) |
|---|---|---|
| hardware | Apple M2 Pro, 10 cores, 16 GB | one NVIDIA RTX 5060 Ti, 16 GB (float64 at about 1/64 of float32) |
| Python | 3.8.16 | 3.12 |
| torch | 2.2.0 | 2.11.0+cu128 |
| e3nn | 0.4.4 | 0.6.0 |
| ase | 3.23.0 | 3.29 |
| numpy / scipy | 1.24.2 / 1.10.0 | numpy 2.5 |
| phonopy / PyYAML | 2.17.2 / 6.0 | — |
| PySCF (table builders only) | 2.14.0 | — |

**`requirements.txt` is not the environment of record.** It pins torch 2.12.0, e3nn 0.6.0, ase 3.28.0 and
numpy 2.4.6, a set that no reported run or score used. The electrostatics suites have not been run on it. To match
the CPU environment, install the versions in the table, then `python -m pip install --no-deps -e .`. That recipe has
not itself been tested on a clean machine.

**float64 is mandatory.** `trace_electrostatics` and `trace_redox` refuse float32. `train.py` and
`TransformersACECalculator` set `torch.set_default_dtype(torch.float64)` themselves.

**Run the tests first** on a new machine; see [the main README](../../TRACE_ELECTROSTATICS.md#tests).

**Cross-version check of record.** The GPU-trained models were scored on the CPU environment (e3nn 0.4.4). Energies
agreed with the GPU to between 4e-12 and 1.1e-10 eV/atom on six structures. Forces agreed to between 4e-10 and
1.6e-8 eV/Å on five of them. The sixth, a doped slab, differed by 1.3e-7 eV/Å with the new terms on and by 1.5e-7 with
them off. Source: `experiments/benchmarks_4g/README.md`, "Step 34's GPU pre-flight, scored".

**Environment variables used by the scripts:**

| variable | holds | default |
|---|---|---|
| `TRACE_4G_RAW` | the unpacked 4G-HDNNP `datasets/` directory | `<repo>/experiments/benchmarks_4g/raw/datasets` |
| `TRACE_PHONON_RAW` | the three NIMS MDR downloads | `<repo>/experiments/benchmarks_phonons/raw` |
| `TRACE_FP_CACHE` | first-principles and phonon-driver work files | `./fp_cache` |

---

## 2. Download the raw data

No third-party raw data are in the repository. Download them yourself and check each file's sha256. Downloaded files
are untrusted input, so the scripts read them with `python -I` and take paths as arguments.

### 2.1 The 4G-HDNNP data sets (every 4G benchmark)

- **Record:** T. W. Ko, J. A. Finkler, S. Goedecker, J. Behler, Materials Cloud Archive 2020.137,
  https://archive.materialscloud.org/record/2020.137, DOI 10.24435/materialscloud:f3-yh. Licence CC BY 4.0.
- **File:** `datasets.tar.gz`, 39,726,732 bytes. The record lists its md5, `0aa11ea305b3af6a47bb3c577c2729bc`.
- **sha256:** `6194ee9e23b446f17c3e8b0b754fa9a9e9498b1b099231015ea10a6a313b9775`.
- **What it holds:** FHI-aims (fhi-aims.171221_1), PBE, "light" settings, according to the record's `readme.txt`.
  The data are RuNNer `input.data` files for `Ag_cluster`, `AuMgO`, `Carbon_chain` and `NaCl`.

```bash
mkdir -p experiments/benchmarks_4g/raw
# save datasets.tar.gz from the record page into experiments/benchmarks_4g/raw/
cd experiments/benchmarks_4g/raw
shasum -a 256 datasets.tar.gz     # must print 6194ee9e...b9775 (above)
tar xzf datasets.tar.gz           # creates datasets/{Ag_cluster,AuMgO,Carbon_chain,NaCl}/input.data
cd ../../..
export TRACE_4G_RAW="$PWD/experiments/benchmarks_4g/raw/datasets"   # or leave it unset: this is the default
```

sha256 of the unpacked files, measured on the machine of record:

| file | sha256 |
|---|---|
| `datasets/Ag_cluster/input.data` | `7eaede1efc2e4dbf4f3330961e92f7e838e1dcc023d64ee82bd77fc68ad6519b` |
| `datasets/AuMgO/input.data` | `a741c9b9a88c9bb593121d779a4dce3645ed4cbca3806d13c422025ffb24c9fc` |
| `datasets/Carbon_chain/input.data` | `8e8b445f12ca241c2b42176bc7ab7fd8d35d98edcb2d47974c45195ed727ba79` |
| `datasets/NaCl/input.data` | `174f9eaa1d537f91e996ccf89aa4420b5e2a4682b1bf3ddd348c3eccc2f81a22` |

`eval_aumgo.py` and `eval_4g.py` read `$TRACE_4G_RAW` directly at scoring time. Table II's mean substrates and the
Ag₃ / NaCl minima are built from the raw set, not from the splits, so keep it in place after step 3.

### 2.2 The three NIMS MDR records (MgO phonons; creator A. Togo; CC BY 4.0)

| record | file | sha256 of the download | sha256 of the unpacked `.yaml` | used for |
|---|---|---|---|---|
| Phono3py input data of rocksalt MgO (PBE), https://mdr.nims.go.jp/datasets/1fd84240-abc0-4d8f-b4d7-1db304d6166b | `phono3py_params.yaml.xz` (81,144 B) | `13b12ff2f0c16d5db00ae14ee6d32b667d4ba0e3932ddd03522bb83f81d7adce` | `d22ebbc2aec853c192204aac1474f2e586e187169b02d641254424d1306ec1df` | the PBE phonon reference; the bulk training cells of steps 33 and 34b |
| Ab-initio phonon calculation for MgO / Fm-3m (225) / materials id 1265 (phonondb), https://mdr.nims.go.jp/concern/datasets/bk128j14x | `phonopy_params.yaml.xz` (1,648 B) | `25b82cafd3efc18ba3f01c99e63e9f58510350978e90c8cb4100bc00637bddce` | `8bbb516cdd8dd701798897ce25d8477e7bf303d13dd64efa07d9f0ef0d92fceb` | the PBEsol phonon reference |
| First-principles lattice thermal conductivity calculation for MgO / Fm-3m (225) / materials id 1265, https://mdr.nims.go.jp/datasets/59c5d436-339d-4b32-ba59-a3da5ea68bc7 | `phonopy_training_dataset.yaml.xz` (456,252 B) | `49355412ced3ebff17be23dcf778cf964b1692b737faf6f2aee7ced22b5b39b0` | `9835a5f10ecdb8881874c8097ce335605c84779eb69fa69077702281cb29691b` | only the PBEsol bulk-only runs of the phonon lab notebook (no released model) |

Put them in this layout:

```
$TRACE_PHONON_RAW/phono3py_pbe_mgo/phono3py_params.yaml.xz
$TRACE_PHONON_RAW/phonondb_mp-1265/phonopy_params.yaml.xz
$TRACE_PHONON_RAW/mdr_ltc2026_mgo/phonopy_training_dataset.yaml.xz
```

`experiments/benchmarks_phonons/data/README.md` repeats these records and says which script makes which file.

**You need none of 2.2 to score phonons.** The four reference band files ship in
`experiments/benchmarks_phonons/reference/`. They are derived from the first two records under CC BY 4.0. You need the
PBE record only to retrain steps 33 and 34b (their bulk cells), or to rebuild the reference bands:

```bash
experiments/benchmarks_phonons/make_reference_bands.sh /tmp/bands_check   # rebuild into a scratch directory
```

The script checks each download's sha256, unpacks it, rebuilds the bands with phonopy, and compares them with the
shipped files byte for byte. The shipped files were written with phonopy 2.17.2. Another phonopy version may change
the last digits, and the script reports that.

---

## 3. Prepare the derived data

```bash
bash experiments/benchmarks_4g/prepare_all.sh
```

`prepare_all.sh` holds the exact `prepare.py` commands of the published runs (the seed is 0 throughout):

| directory | sets | `prepare.py` arguments `N_TRAIN N_VALID N_TEST` | used by |
|---|---|---|---|
| `experiments/benchmarks_4g/data/` | Ag_cluster, Carbon_chain, NaCl | `1000 250 1000` | the first-round cluster runs |
| `experiments/benchmarks_4g/data/` | AuMgO | `300 75 400` (written as 300 / 76 / 400 by per-stratum rounding) | steps 26 and 31; the slabs of steps 33 and 34b |
| `experiments/benchmarks_4g/data_pfn/` | Ag_cluster, Carbon_chain | `8000 1000 -1` | the PFN-size protocol |
| `experiments/benchmarks_4g/data_pfn/` | NaCl, AuMgO | `4000 500 -1` | the PFN-size protocol |
| `experiments/benchmarks_4g/data_cs/` | AuMgO | `300 75 400`, then `annotate_charge_states.py` | charge-state probes |

It also writes the response priors:
- `data/Au_atom_alpha_prior.xyz` (`make_au_alpha_prior.py`);
- `data/MgO_bulk_eps_prior.xyz` (`make_mgo_eps_prior.py`; ε∞ = 2.95, used by steps 26 and 31);
- `data_pfn/NaCl_bulk_eps_prior.xyz` and `data_pfn/Ag_atom_alpha_prior.xyz` (the step-9 cluster runs).

These splits were recomputed from the raw `input.data` with `prepare.py`'s own reader and split logic. The positions,
energies and forces of every split matched the files of the published runs to 1e-6.

**The phonon-side data** (only to retrain steps 33 and 34b). Run these from the repository root, after the 4G
`data/AuMgO_*` files exist:

```bash
cd experiments/benchmarks_phonons
xz -dk "$TRACE_PHONON_RAW/phono3py_pbe_mgo/phono3py_params.yaml.xz"        # check the sha256 of both files first (2.2)
python -I extract_phono3py_cells.py "$TRACE_PHONON_RAW/phono3py_pbe_mgo/phono3py_params.yaml" data/MgO_PBE_togo_displaced.extxyz
python make_bulk_augmented_data.py --force-weight 122.4    # data/AuMgO_bulkaugw_{train,valid}.xyz: 436 / 86 structures
python make_mgo_eps_prior_dft.py                            # data/MgO_bulk_eps_prior_{pbe,pbesol}.xyz
cd ../..
```

- `extract_phono3py_cells.py` writes the 148 displaced PBE supercells, forces only (`has_energy=F`).
- `make_bulk_augmented_data.py --force-weight 122.4` writes the training set of steps 33 and 34b: the 300 / 76 4G
  slabs, plus 136 / 10 of Togo's fc3 cells, with `info['force_weight'] = 122.4` on the bulk cells. The two fc2 cells
  are excluded. Without the flag it writes `AuMgO_bulkaug_*` instead; with `--bulk-only` it writes `MgO_bulkonly_*`.
- `make_mgo_eps_prior_dft.py` writes the PBE ε∞ prior of steps 33 and 34b (3.235 at a = 4.255656884758435 Å).

---

## 4. Verify the data

```bash
python experiments/benchmarks_4g/verify_data.py
```

`verify_data.py` checks every derived file against `experiments/benchmarks_4g/DATA_SHA256.txt`. It runs a sha256
check and a content check of positions, energies and forces. Those hashes are for files written by ase 3.23.0. Whether
another ase version writes byte-identical extxyz files has not been tested, so the content check is the one that
decides. Do not train or score until it passes.

---

## 5. Retrain the released models (optional)

Skip this section to check the released checkpoints instead (section 7).

### 5.1 How the runs are laid out

- Each released run has a sanitised training config at `experiments/benchmarks_4g/configs/<run>.yaml`. Every path in
  it is relative to `experiments/benchmarks_4g`. The file's header gives the exact commands, the warm-start parent,
  the device of the published run, and every difference from the run's recorded `config.yaml`.
- Training runs from `experiments/benchmarks_4g`:

```bash
cd experiments/benchmarks_4g
RUN=AuMgO_twosite_bf055_bondlaw_r5_s43                      # any released run name
mkdir -p runs/$RUN && cp configs/$RUN.yaml runs/$RUN/config.yaml
python run_train_nanguard.py runs/$RUN/config.yaml > runs/$RUN/log.txt 2>&1
```

- `run_train_nanguard.py` wraps `train.py`. When a training step produces a transient NaN, the guard recomputes it.
- Each run writes its best checkpoint by validation loss to `runs/<run>/model.pt`. That is the file every scorer
  reads.
- **Configs record `device: cpu`.** A run published from the GPU host was trained with the same file and two changes:
  its paths rewritten to absolute ones, and `device: cuda`. To train on a GPU, set `device: cuda` and export
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, as the published runs did.
- **Warm starts** are weights-only files: `{epoch 0, model_state_dict, config}` of the parent's best checkpoint. They
  hold no optimiser or scheduler state. `make_warm_start.py` writes them, and the config header gives the exact
  command. With torch 2.2.0 it rebuilds the published warm starts of steps 31 and 33A byte for byte.

### 5.2 Step 34b: the current-equation model (two stages, both seeds)

Chain: stage A, a cold start → warm start from stage A's best checkpoint → stage B.

| run | stage | warm-start parent | epochs | lr | energy weight | best epoch |
|---|---|---|---|---|---|---|
| `AuMgO_twosite_bf055_bondlaw_r5_s{42,43}` | A | none (cold start) | 40 | 1e-3 → 5e-5 | 10 | 39 / 39 |
| `AuMgO_twosite_bf055_e1kw_bondlaw_r5_s{42,43}` | B | stage A `model.pt`, same seed | 20 | 1e-4 → 5e-6 | 1000 | 15 / 18 |

```bash
cd experiments/benchmarks_4g
python check_run_config.py runs/AuMgO_twosite_bf055_bondlaw_r5_s42 runs/AuMgO_twosite_bf055_bondlaw_r5_s43   # after copying the configs
# stage A, each seed (the published runs trained both seeds at once on one GPU)
python run_train_nanguard.py runs/AuMgO_twosite_bf055_bondlaw_r5_s43/config.yaml > runs/AuMgO_twosite_bf055_bondlaw_r5_s43/log.txt 2>&1
# warm start, then stage B
python make_warm_start.py runs/AuMgO_twosite_bf055_bondlaw_r5_s43/model.pt \
    runs/AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43/warm_start.pt --archive-name warm_from_twosite_a_s43
python run_train_nanguard.py runs/AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43/config.yaml > runs/AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43/log.txt 2>&1
```

- **Training data:** `../benchmarks_phonons/data/AuMgO_bulkaugw_{train,valid}.xyz` (section 3).
- **Priors:** the PBE ε∞ prior `../benchmarks_phonons/data/MgO_bulk_eps_prior_pbe.xyz` (weight 0.05) and the Au α prior
  `data/Au_atom_alpha_prior.xyz` (weight 0.3).
- **The one change from step 34 is `redox.band_initial_factor: 0.55`.** That is SPEC 20's band anchor. Step 34's first
  launch, without it, stopped in epoch 2 on both seeds: the cluster closure refused 529 > 512 configurations on a
  cold-start bulk MgO state (`experiments/benchmarks_4g/README.md`, "Step 34 stage A stopped").
- **Stage-B warm start.** The published file was written on the GPU host (torch 2.11), and its sha256 was not
  recorded. The command above writes the same content: every tensor, the state-dict order, the stored config and
  epoch 0 were checked. With torch 2.2.0 on the Mac it gives sha256
  `819f8e699284e869416fef8681779daebe711a9e6717fbf76a11de3ecd570a03` for seed 43 (from the config's header).
- **Wall time (GPU host, one RTX 5060 Ti, torch 2.11.0+cu128, both seeds sharing the card):**
  - stage A: median 19.8 / 19.9 min per epoch, 48,169 / 48,317 s in all;
  - stage B: 20.2 / 20.0 min per epoch, 24,249 / 24,054 s in all.
- **Memory and NaN recomputations:** about 3.4 GB of GPU memory per run in stage B (sampled, not a recorded peak). The
  NaN guard recomputed nothing in this step.

### 5.3 Step 31: the closest Table II (step 26 → step 31, both seeds)

Chain: step 26, a cold start → warm start → step 31.

| run | step | warm-start parent | epochs | lr | energy weight | best epoch (s42 / s43) |
|---|---|---|---|---|---|---|
| `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bopw_bondlaw_r5_s{42,43}` | 26 | none (cold start) | 40 | 1e-3 → 5e-5 | 10 | 40 / 39 |
| `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bopw_e1kw_bondlaw_r5_s{42,43}` | 31 | step 26 `model.pt`, same seed | 20 | 1e-4 → 5e-6 | 1000 | 20 / 17 |

```bash
cd experiments/benchmarks_4g
P=AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bopw
python run_train_nanguard.py runs/${P}_bondlaw_r5_s43/config.yaml > runs/${P}_bondlaw_r5_s43/log.txt 2>&1
python make_warm_start.py runs/${P}_bondlaw_r5_s43/model.pt runs/${P}_e1kw_bondlaw_r5_s43/warm_start.pt \
    --key-order config-first --archive-name warm_from_step26_s43
python run_train_nanguard.py runs/${P}_e1kw_bondlaw_r5_s43/config.yaml > runs/${P}_e1kw_bondlaw_r5_s43/log.txt 2>&1
```

- **Data and priors:** `data/AuMgO_{train,valid}.xyz` with the measured ε∞ prior `data/MgO_bulk_eps_prior.xyz` (2.95)
  and the Au α prior.
- **Step 26 anneals:** kT 0.5 → 0.1 over 20 epochs; attention temperature 1.3 → 1 over 15 epochs. Its per-species
  reference energies are fitted on the initial model (`atomic_energies_from_model: true`).
- **Step 31 differs from step 26 only in** energy weight 1000, 20 epochs, lr 1e-4 → 5e-6, both anneals off, and the warm
  start.
- **The configs add `electrostatics.redox.state_polarizability_stiffness: block`.** These runs trained SPEC 28's
  `'block'` stiffness. The current code resolves an absent key to `'onsite'` (SPEC 28.6), a different energy. Scoring
  is unaffected, because the released checkpoints store their own RedoxConfig and load as `'block'`.
- **Warm-start check.** With torch 2.2.0, the warm-start command writes the published file byte for byte: seed 43
  sha256 `76813360c4a1a7606581fd75277767b4c9c67ede3030c0a067ad5650e98ce1a7`; seed 42
  `11bbcc91140072748fe2cd22b2504661a305c7c0a2c1123e1ad884454f5898c2`, with `--archive-name warm_from_step26_s42`.
- **Not checked:** whether the release code retrains these configs to the same models. The package changed after these
  runs. What was checked: a tree of 2026-11-10 reproduced step 31 s43's stored Table II from the checkpoint.
- **Wall time (CPU environment, M2 Pro, torch 2.2.0, 4 intra-op threads, both seeds at once):**
  - step 26: median 19.2 / 21.8 min per epoch;
  - step 31: 29.1 / 28.9 min per epoch, 34,909 / 35,148 s in all, under `nice -n 5`.
- **Step 26's total wall time** (151,593 / 176,024 s) includes paused periods.
- **NaN recomputations:** 6 / 15 in step 26, 2 / 2 in step 31.

### 5.4 Step 33: every term then built on (step 31 → stage A → stage B)

| run | stage | warm-start parent | best epoch (s42 / s43) |
|---|---|---|---|
| `AuMgO_single_all_bondlaw_r5_s{42,43}` | A | step 31 `model.pt`, same seed, with `--reset-log-alpha-scale` | 40 / 39 |
| `AuMgO_single_all_e1kw_bondlaw_r5_s{42,43}` | B | stage A `model.pt`, same seed | 15 / 20 |

```bash
cd experiments/benchmarks_4g
P=AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bopw
python make_warm_start.py runs/${P}_e1kw_bondlaw_r5_s42/model.pt runs/AuMgO_single_all_bondlaw_r5_s42/warm_start.pt \
    --reset-log-alpha-scale --archive-name warm_from_step31_s42
python run_train_nanguard.py runs/AuMgO_single_all_bondlaw_r5_s42/config.yaml > runs/AuMgO_single_all_bondlaw_r5_s42/log.txt 2>&1
python make_warm_start.py runs/AuMgO_single_all_bondlaw_r5_s42/model.pt runs/AuMgO_single_all_e1kw_bondlaw_r5_s42/warm_start.pt
python run_train_nanguard.py runs/AuMgO_single_all_e1kw_bondlaw_r5_s42/config.yaml > runs/AuMgO_single_all_e1kw_bondlaw_r5_s42/log.txt 2>&1
```

- **Why `--reset-log-alpha-scale`.** It adds SPEC 35's crystal factor `electrostatics.log_alpha_scale` at its initial
  value 0.0. Step 31's model does not have it.
- **Warm-start check.** With torch 2.2.0 the stage-A warm start comes out byte for byte: seed 42 sha256
  `253e63dff93baa992a0dec6cd05529f3d07dd99b216fdd92aa2de2cb017b255c`. The stage-A log warns that the warm start trained
  `'block'` and `'all'`.
- **Wall time (GPU host):** stage A 21.3 / 20.9 min per epoch; stage B 21.4 / 20.9 min per epoch.

### 5.5 What a retraining will and will not reproduce

Retraining is not bitwise reproducible, for three reasons:
- `cholesky_solve` gives differences of one unit in the last place;
- GPU reductions sum in a varying order;
- the NaN guard recomputes transient NaN steps.

The released checkpoints are the reproduction path for the reported numbers. A retraining reproduces the protocol; it
does not reproduce the digits.

### 5.6 The cluster benchmarks (Ag₃, NaCl, carbon chain)

- **Configs:** the PFN-size protocol and the step-9 cluster runs are in `experiments/benchmarks_4g/configs/protocol/`.
- **Training:** `python run_train.py runs/protocol/<tag>/config.yaml`, from `experiments/benchmarks_4g`, as each
  header says.
- **Scoring:** `EVAL_DATA_DIR=data_pfn python eval_4g.py DATASET RUN`, then `python compare_protocol.py DATASET RUN...`.
- **Median wall time per epoch** (local / elec / elecq arm, as recorded in the logs):

| system | local | elec | elecq | machine |
|---|---|---|---|---|
| Ag₃ | 21 s | 35 s | 36 s | CPU environment |
| NaCl | 52 s | 95 s | 90 s | CPU environment |
| carbon chain | 39 s | 106 s | 111 s | GPU host |

The expected numbers are in [BENCHMARKS.md](BENCHMARKS.md).

---

## 6. Score a model

```bash
cd experiments/benchmarks_4g
bash score_model.sh runs/<run>
```

`score_model.sh` scores one run on the local machine. It writes the files of `runs/<run>/scores/`:

| what | script it calls | score file |
|---|---|---|
| Table II, test E / F RMSE, relaxed Au–O | `eval_aumgo.py` | `T5_wetting.txt` |
| charge transfer T1–T3 | `eval_aumgo_transfer.py` | `T1_T3_transfer.txt` |
| class residuals (test and training) | `probe_class_residuals.py` | `H1_class_residuals{,_train}.txt` |
| S5, S7, S8 | `probe_auau_two_site.py`, `probe_force_outliers.py`, `score_step34_s7_s8.py` | `S7_S8.txt`, `S7_auau_scan.txt`, `S8_force_outliers.txt` |
| gas-phase Au₂ EA / IP | `probe_au2_ea_ip.py` at 2.30, 2.47, 2.60, 2.80 Å | `S6_au2_ea_ip.txt` |
| coupling sizes | `probe_coupling_sizes.py` | `Z1_Z2_couplings.txt` |
| MgO phonons against PBE | `../benchmarks_phonons/score_phonons.py` (2×2×2, then `--p5-supercell 4`) | `P_phonons_pbe.txt`, `P_phonons_pbe_sc4.txt` |

**Time.** Scoring the four step-34b models took about 3 h on the CPU environment. The 4×4×4 phonon score alone takes
about 61–62 s and 6.5–6.6 GB peak resident memory per model (`runs/<run>/scores/P_phonons_pbe_sc4_time.txt`).

**Phonons alone**, from the repository root:

```bash
python experiments/benchmarks_phonons/score_phonons.py experiments/benchmarks_4g/runs/<run>/model.pt --p5-supercell 4
```

- **Which protocol.** `--p5-supercell 4` is the protocol pre-registered on 2026-11-16 for every step after 34b. It
  reads P5 on PBE's own 512-atom supercell. Step 34b's own registered criterion S4 is the 2×2×2 reading, without the
  flag.
- **Where the output goes.** By default the band files go to `experiments/benchmarks_phonons/runs/<run>/` (and
  `<run>_sc4/`); `--out` writes them elsewhere. The phonon driver keeps its work files under
  `$TRACE_FP_CACHE/mgo_phonons`.

---

## 7. Check the released checkpoints instead of retraining

- **Where they are.** The checkpoints are GitHub release assets named `<run_name>.pt`, with a `SHA256SUMS` file. They
  are listed in [MODELS.md](MODELS.md).
- **What each is.** Each one is the best checkpoint by validation loss of its run.

```bash
cd experiments/benchmarks_4g
# download <run>.pt and SHA256SUMS from the release into a directory, e.g. ~/Downloads/trace-models, then:
( cd ~/Downloads/trace-models && shasum -a 256 -c SHA256SUMS --ignore-missing )
for RUN in AuMgO_twosite_bf055_e1kw_bondlaw_r5_s42 AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43; do
    mkdir -p runs/$RUN && cp ~/Downloads/trace-models/$RUN.pt runs/$RUN/model.pt
    bash score_model.sh runs/$RUN
done
```

sha256 of the main released checkpoints (the same values as `SHA256SUMS` and [MODELS.md](MODELS.md)):

| run | step, stage | best epoch | sha256 |
|---|---|---|---|
| `AuMgO_twosite_bf055_bondlaw_r5_s42` | 34b A | 39 | `681c36d227b6666e8e62bba177286f6fbfa06f4d005d73a3a7c71930132319b5` |
| `AuMgO_twosite_bf055_bondlaw_r5_s43` | 34b A | 39 | `29c908ed337a0ba7dfd622560898972f538a452aa30d78e5e73d4d42cdaf7b0b` |
| `AuMgO_twosite_bf055_e1kw_bondlaw_r5_s42` | 34b B | 15 | `332b2654d96e512433e43b750b09a19a927656f51a580847176b2e70fed7b5cf` |
| `AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43` | 34b B | 18 | `740ea30cca04950d153ebddc4301b9926ea2e97e89a8d6b09fa8f4e24a79c297` |
| `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bopw_e1kw_bondlaw_r5_s42` | 31 | 20 | `653c648f86019443fbdf385d440716829750b33341da5565a93165603a580d1f` |
| `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bopw_e1kw_bondlaw_r5_s43` | 31 | 17 | `3699c1f94b922247da24699a7a92b6e4b917d307d31ef48dc2d829f9b73e799e` |

[MODELS.md](MODELS.md) lists the other assets: steps 26 and 33, the warm starts, and the cluster runs.

**What a correct check looks like.**
- `T5_wetting.txt`, `T1_T3_transfer.txt`, `S7_S8.txt`, `S6_au2_ea_ip.txt` and the phonon files should repeat the
  values in section 8 to the printed digit.
- Expect agreement of order 1e-10 eV/atom in energy across machines, not bit equality (section 1).
- The `trace_redox/data/` tables are read at run time and are not stored in the checkpoint. With other tables, the
  model is a different model. `trace_redox/data/two_site.json` records the sha256 of the files it was solved with.

---

## 8. Expected numbers

- **What each value is.** Every value is copied from the named score file of the published run, under
  `experiments/benchmarks_4g/runs/<run>/scores/`. Each one is the model as trained, evaluated with frozen weights.
- **The DFT reference.** For the slabs it is PBE / FHI-aims (Ko et al. 2021); for the phonons it is Togo's PBE set
  (VASP).
- **What T5 is.** T5 = wet_d − wet_u. DFT's T5 is −931.6 meV, and "% of DFT" is T5 / −931.6.

### 8.1 Step 34b, stage B (`AuMgO_twosite_bf055_e1kw_bondlaw_r5_s{42,43}`)

| quantity | s42 | s43 | DFT | file |
|---|---|---|---|---|
| wet_u, undoped E(wetting) − E(non-wetting) (meV) | +610.4 | +729.7 | +928.9 | `T5_wetting.txt` |
| wet_d, doped (meV) | +92.3 | −27.7 | −2.7 | `T5_wetting.txt` |
| T5 (meV; % of DFT) | −518.1 (55.6 %) | −757.4 (81.3 %) | −931.6 | from the two lines above |
| test F RMSE undoped / doped (eV/Å) | 0.0845 / 0.1242 | 0.0616 / 0.1112 | | `T5_wetting.txt` |
| test E RMSE undoped / doped (meV/atom) | 1.353 / 0.989 | 1.142 / 0.671 | | `T5_wetting.txt` |
| relaxed non-wetting Au–O doped / undoped (Å) | 2.297 / 2.195 | 2.430 / 2.165 | 2.332 / 2.190 | `T5_wetting.txt` |
| q(Au₂) shift doped − undoped (e); Pearson r with Hirshfeld | −1.0000; 0.848 | −0.9966; 0.849 | −0.2656 (Hirshfeld) | `T1_T3_transfer.txt` |
| S7: Au–Au bond force on test 167 at 2.089 Å (eV/Å) | +14.25 (PASS) | +15.82 (PASS) | +15.25 | `S7_S8.txt` |
| S8: undoped test F RMSE without the five compressed dimers (eV/Å; ≤ 0.060) | 0.0844 (FAIL) | 0.0614 (FAIL) | | `S7_S8.txt` |
| S5: crystal factor g; ε∞ of the prior cell | 0.9515; 3.3333 (FAIL) | 0.9396; 3.3 (FAIL) | ε∞ 3.235 | `S7_S8.txt` |
| gas-phase Au₂ at 2.47 Å, as trained: EA / IP (eV) and errors | 2.393 / 9.786 (+0.363 / +0.566) | 2.014 / 9.294 (−0.016 / +0.074) | | `S6_au2_ea_ip.txt` |
| EA / IP error range, 2.30–2.80 Å (eV) | +0.197 to +0.431 / +0.562 to +0.593 | −0.034 to −0.001 / +0.070 to +0.081 | | `S6_au2_ea_ip.txt` |
| class residual flat − upright, test, undoped / doped (meV) | −273.5 ± 8.1 / +165.4 ± 7.9 | −95.3 ± 15.3 / −46.1 ± 10.6 | | `H1_class_residuals.txt` |
| MgO Z*(Mg) / ε∞ | 1.950 / 3.334 | 1.961 / 3.300 | 1.9923 / 3.235 | `P_phonons_pbe.txt` |
| MgO TO / LO at Γ (cm⁻¹) | 382.1 (+2.4 %) / 669.3 | 370.3 (−0.8 %) / 667.4 | 373.3 / 681.3 | `P_phonons_pbe.txt` |
| phonons on 2×2×2: P5 / P6 (cm⁻¹); passed | 29.5 / 7.1; 6 / 7 (P5 fails) | 29.9 / 6.8; 6 / 7 (P5 fails) | | `P_phonons_pbe.txt` |
| phonons, P5 on 4×4×4: P5 / P6 (cm⁻¹); passed | 8.1 / 8.5; 7 / 7 | 8.6 / 8.8; 7 / 7 | | `P_phonons_pbe_sc4.txt` |

**Verdict as pre-registered:** FAIL on S2 (Table II), S5 and S8 on both seeds; S1, S3, S4 and S7 pass on both
(`experiments/benchmarks_4g/README.md`, "Step 34b scored").

In `S6_au2_ea_ip.txt`, the PBE0 references are 2.03 eV (EA) and 9.22 eV (IP) at 2.47 Å; the errors are the differences
from them.

### 8.2 Step 34b, stage A (`AuMgO_twosite_bf055_bondlaw_r5_s{42,43}`): phonons only

| | s42 | s43 | file |
|---|---|---|---|
| 2×2×2: Z*(Mg), LO (cm⁻¹), P5 (cm⁻¹), passed | 1.950, 670.7, 29.3; 5 / 7 (P3 TO 384.7, +3.1 %; P5) | 1.960, 664.9, 30.6; 6 / 7 (P5) | `P_phonons_pbe.txt` |
| P5 on 4×4×4 (cm⁻¹), passed | 8.4; 6 / 7 (P3) | 9.8; 7 / 7 | `P_phonons_pbe_sc4.txt` |

### 8.3 Step 31 (`AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bopw_e1kw_bondlaw_r5_s{42,43}`)

| quantity | s42 | s43 | DFT | file |
|---|---|---|---|---|
| wet_u / wet_d (meV) | +851.0 / −177.4 | +920.4 / −75.7 | +928.9 / −2.7 | `T5_wetting.txt` |
| T5 (meV; % of DFT) | −1028.4 (110.4 %) | −996.1 (106.9 %) | −931.6 | from the line above |
| test F RMSE undoped / doped (eV/Å) | 0.0614 / 0.0867 | 0.0776 / 0.1045 | | `T5_wetting.txt` |
| test E RMSE undoped / doped (meV/atom) | 0.590 / 0.937 | 0.469 / 0.621 | | `T5_wetting.txt` |
| relaxed non-wetting Au–O doped / undoped (Å) | 2.301 / 2.171 | 2.306 / 2.168 | 2.332 / 2.190 | `T5_wetting.txt` |
| q(Au₂) shift doped − undoped (e) | −1.0000 | −0.9988 | −0.2656 (Hirshfeld) | `T1_T3_transfer.txt` |
| class residual flat − upright, test, undoped / doped (meV) | −64.5 ± 5.5 / −142.5 ± 7.4 | +15.3 ± 6.3 / −69.9 ± 8.1 | | `H1_class_residuals.txt` |
| T6: ε∞ of bulk MgO | 3.284 | 2.934 | | `T6_eps.txt` |

**Pre-registered verdict:** FAIL. s43 passes Y1–Y5; s42 fails Y1 and Y3 (`experiments/benchmarks_4g/README.md`, "Step
31 scored").

**Step 31's phonons** were scored later with `--p5-supercell 4`. The outputs are in
`experiments/benchmarks_phonons/runs/step31_<run>{,_sc4}/`, and the numbers are recorded in
`experiments/benchmarks_phonons/README.md`, "Step 31's phonons, scored for the first time":

| | Z*(Mg) | ε∞ | TO / LO (cm⁻¹) | P5 on 4×4×4 (2×2×2) | passed |
|---|---|---|---|---|---|
| s42 | 2.402 | 3.025 | 384.3 / 807.9 | 22.0 (29.6) | 3 / 7 |
| s43 | 2.534 | 3.002 | 398.7 / 851.5 | 47.8 (45.5) | 2 / 7 |

**Never measured for step 31:** S7 by the S7 protocol, and S8. The nearest number is a different probe
(`probe_auau_wall_terms.py`): the total bond force at 2.089 Å on test 167 is +6.18 eV/Å (s43). In the same scan the
compressed dimer turns ionic (Au⁺Au⁻): −5.56 eV/Å at 2.07 Å and −13.79 to −22.63 eV/Å from 2.06 to 2.00 Å
(`experiments/benchmarks_4g/README.md`, "Scored: the Au–Au wall").

### 8.4 Step 26 (step 31's warm-start parent)

| quantity | s42 | s43 | file |
|---|---|---|---|
| wet_u / wet_d (meV) | relaxation failed | +810.6 / −198.2 | `T5_wetting.txt` |
| test F RMSE undoped / doped (eV/Å) | 0.0617 / 0.0853 (lab notebook) | 0.0768 / 0.1056 | `T5_wetting.txt`; `experiments/benchmarks_4g/README.md`, "Step 31 scored" |

### 8.5 Step 33, stage B (`AuMgO_single_all_e1kw_bondlaw_r5_s{42,43}`)

| quantity | s42 | s43 | file |
|---|---|---|---|
| wet_u / wet_d (meV) | +768.4 / −9.2 | +789.9 / +92.7 | `T5_wetting.txt` |
| test F RMSE undoped / doped (eV/Å) | 0.1930 / 0.1043 | 0.2211 / 0.1563 | `T5_wetting.txt` |
| test E RMSE undoped / doped (meV/atom) | 1.339 / 0.877 | 1.396 / 0.880 | `T5_wetting.txt` |
| phonons, 2×2×2: Z*(Mg), ε∞, TO, LO, P5 / P6; passed | 1.948, 3.291, 363.4, 661.2, 32.1 / 8.4; 6 / 7 | 1.942, 3.308, 371.5, 663.2, 30.4 / 5.6; 6 / 7 | `P_phonons_pbe.txt` |

**Not scored for step 33:** P5 on 4×4×4.

### 8.6 MgO phonons from this release tree

`score_phonons.py --p5-supercell 4` was re-run on 2026-11-18 from this release tree, on the CPU environment, against
the step-34b stage-B s43 checkpoint (sha256 `740ea30c…c297`). Every printed number equalled the run's
`P_phonons_pbe.txt` and `P_phonons_pbe_sc4.txt`. The output is shown in
[the main README](../../TRACE_ELECTROSTATICS.md#4-score-mgo-phonons-against-pbe).

---

## 9. Not reproducible from this repository

- **Rebuilding the first-principles tables in `trace_redox/data/`** needs PySCF and caches of frozen-density points that
  are not distributed. Some Au–Au points took 2.5–4.8 h each. The builders are in `tools/electrostatics/` and read the
  cache from `$TRACE_FP_CACHE`. The shipped tables are the reproduction path.
- **`ion_polarizabilities.json`.** The step that converts the tool's output into the shipped schema is not in the
  repository.
- **The water-dimer data** (`experiments/electrostatics_water_dimers/`) are not distributed. Their source data set's
  licence is unknown.
- **Never measured:**
  - Table II on the 2×2 supercell for steps 33 and 34b;
  - P5 on 4×4×4 for step 33;
  - S7 and S8 for step 31;
  - whether the release code retrains steps 26 and 31, and the cluster runs, to their stored numbers.
