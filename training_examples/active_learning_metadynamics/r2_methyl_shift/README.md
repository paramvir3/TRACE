# R2 methyl-shift free energy (Figure 4)

This directory reproduces the Figure 4 protocol of Vitartas *et al.* with a
TRACE potential. The reaction is the gas-phase methyl rearrangement of
2,2-dimethylisoindene to 1,2-dimethylindene. It is a neutral singlet with 23
atoms (`C11H12`), and its reference labels are
PBE0-D3BJ/def2-SVP energies and forces.

The collective variable is

\[
s(\mathbf R)=r_1-r_2
=\lVert\mathbf r_{14}-\mathbf r_{11}\rVert
-\lVert\mathbf r_{14}-\mathbf r_{10}\rVert,
\]

where the subscripts are zero-based atom indices. The corresponding PLUMED
pairs are `15,12` and `15,11`. All PLUMED distance actions use `NOPBC` because
R2 is a finite gas-phase molecule.

## 1. Data provenance and separation

Regenerate the inputs from the publisher's version-4 Figshare archive:

```bash
cd /path/to/Transformers-ACE
python training_examples/active_learning_metadynamics/prepare_data.py
```

The resulting R2 sets are deliberately separate:

| Set | Frames | Role |
|---|---:|---|
| `data/wtmetad_all.xyz` | 192 | Primary Figure 4 training source |
| `data/downhill_all.xyz` | 131 | Optional training-source ablation |
| `data/combined_all.xyz` | 323 | Union of both AL sources for a higher-coverage TRACE model |
| `data/us_test.xyz` | 326 | Independent US/AIMD test set only |

`prepare_data.py` checks the source MD5 values, writes output SHA-256 values to
`data_manifest.json`, rejects train/test geometry overlap, and makes fixed
85:15 training/validation splits. Never include `us_test.xyz` in fitting,
early stopping, hyperparameter selection, or energy referencing.

Plot the Figure 4a coverage comparison:

```bash
cd training_examples/active_learning_metadynamics/r2_methyl_shift
python plot_figure4a.py
```

This writes `results/figure4a_trace.{png,pdf,json}`. The stationary-point CVs
obtained from the released coordinates are -0.96852 A (reactant), -0.10327 A
(transition state), and 1.02828 A (product).

## 2. Train and independently validate TRACE

The source paper used the MLIP trained by WTMetaD-IB AL for the free-energy
profiles in Figure 4b. Therefore `config.yaml`, and the default model path in
the sampling scripts, deliberately use only the WTMetaD-IB source. This is the
source-matched reproduction model, not an omission of the downhill data.

All three configurations use the same hybrid Muon optimizer. Muon updates the
hidden attention and scalar feed-forward matrices with a learning rate of
`5e-3`; auxiliary AdamW updates the ACE encoder, equivariant projections,
embeddings, normalization parameters, and energy readout at `5e-4`. This
partition avoids applying matrix orthogonalization to parameters for which its
geometry is not appropriate. The ten-epoch warmup and gradient clipping remain
enabled. For a controlled optimizer comparison, change only `optimizer` to
`adamw` and leave `learning_rate: 0.0005` unchanged.

Train the primary 192-frame model:

```bash
python ../../../train.py --config config.yaml
```

The checkpoint is written to `models/r2_trace_v2_wtmetad.pt`. The optional
downhill-data ablation is independent:

```bash
python ../../../train.py --config config_downhill.yaml
```

For a higher-coverage TRACE potential, train the union of the 192 WTMetaD-IB
and 131 downhill configurations:

```bash
python ../../../train.py --config config_combined.yaml
```

This writes `models/r2_trace_v2_combined.pt`. No duplicate geometries were
found between the two AL sources, so the combined pool contains 323 frames,
split reproducibly into 274 training and 49 validation frames. This model is a
useful sensitivity test and may be more robust in production dynamics, but its
results must be reported as a combined-training TRACE calculation rather than
as an exact reproduction of the paper's WTMetaD-IB training protocol.

Evaluate the primary checkpoint on the untouched 326-frame set:

```bash
python ../evaluate_checkpoint.py \
  --model models/r2_trace_v2_wtmetad.pt \
  --data data/us_test.xyz \
  --output evaluation/us \
  --transition-min -0.20 \
  --transition-max 0.00
```

For the current Muon-trained checkpoint, this evaluation gives an energy MAE
of 2.20 meV/atom and a force MAE of 111.50 meV/A over all 326 frames. In the
ten-frame transition region it gives 1.35 meV/atom and 152.88 meV/A. These
values were obtained from the independent set, not the validation split.

The evaluator reports global and transition-region energy and force errors.
For context, the source ACE fit reported test MAEs of 1.49 meV/atom and
161.76 meV/A globally, and 4.27 meV/atom and 290.24 meV/A in the interval
`-0.2 < s < 0.0 A`. These are comparison values, not acceptance thresholds
that should be enforced by tuning against the test set.

Evaluate the combined checkpoint against the same untouched test set with:

```bash
python ../evaluate_checkpoint.py \
  --model models/r2_trace_v2_combined.pt \
  --data data/us_test.xyz \
  --output evaluation/us_combined \
  --transition-min -0.20 \
  --transition-max 0.00
```

Model selection must use only the corresponding validation split. Do not use
the independent US/AIMD errors to choose between checkpoints and then report
those same errors as an unbiased test result.

The completed epoch-2000 combined checkpoint can be evaluated separately:

```bash
python ../evaluate_checkpoint.py \
  --model models/r2_trace_v2_combined_last.pt \
  --data data/us_test.xyz \
  --output evaluation/us_combined_last \
  --transition-min -0.20 \
  --transition-max 0.00
```

For the current run it gives energy/force RMSEs of 1.35 meV/atom and
0.113 eV/A over all 326 structures, and 0.62 meV/atom and 0.135 eV/A in the
ten-structure transition subset.

Do not start production sampling if the trajectory leaves the training
support, forms unintended bonds, or has unstable energy and forces. Begin with
a short pilot in a separate directory:

```bash
python run_wtmetad.py \
  --mode standard \
  --replica 0 \
  --duration-ps 1 \
  --output results/pilots/standard_1ps \
  --device cpu
```

Inspect `run_status.json`, `md.log`, `COLVAR`, and `trajectory.traj` before
increasing the duration. Replace `cpu` by `cuda` on a CUDA machine.

To run a pilot with the combined model, provide it explicitly:

```bash
python run_wtmetad.py \
  --model models/r2_trace_v2_combined.pt \
  --mode standard \
  --replica 0 \
  --duration-ps 1 \
  --output results/pilots/combined_standard_1ps \
  --device cpu
```

## 3. PLUMED environment

Dynamics needs both the PLUMED executable and its Python extension in the same
environment as TRACE:

```bash
source /path/to/work/transformers-ace-venv/bin/activate
source /path/to/plumed2/sourceme.sh
which python
which plumed
python -c "import plumed; print('PLUMED Python interface is available')"
```

If the executable is visible but `import plumed` fails, rebuild the PLUMED
Python extension for this exact Python interpreter as described in
`../r1_ch3cl_f/README.md`.

## 4. Reduced 12-hour screening protocol

Nine stable trajectories are not a physical requirement. The source study
attempted ten repetitions per method to estimate narrow confidence intervals;
nine WTMetaD and nine WTMetaD+IB trajectories remained stable. When the
available wall time is limited, three independent repetitions are the minimum
useful number for estimating between-run variability.

The workstation preset retains the published 30 umbrella centers, temperature,
0.5 fs timestep, metadynamics parameters, and umbrella force constant, but
uses 150 ps per standard WTMetaD replica, 100 ps per inherited-bias replica,
and 20 ps per umbrella window with the first 5 ps discarded. Its total sampling
is 2.55 ns rather than 19.5 ns. On the ten-core M2 Pro used to benchmark this
repository, six concurrent workers are expected to finish in roughly 8-12
hours, subject to thermal throttling.

```bash
python run_rapid12h.py --workers 6 --device cpu
```

Completed tasks are skipped when the launcher is run again. An incomplete,
nonempty task is left untouched unless `--overwrite` is explicitly supplied.
The protocol and task status are recorded under `results/rapid12h`.

After sampling, reconstruct all three replicas:

```bash
for replica in $(seq 0 2); do
  tag=$(printf "%02d" "$replica")
  python reconstruct_wtmetad.py \
    --replica-root "results/rapid12h/standard/replica_${tag}"
  python reconstruct_wtmetad.py \
    --replica-root "results/rapid12h/inherited/replica_${tag}"
  python analyze_umbrella.py \
    --replica-root "results/rapid12h/umbrella/replica_${tag}"
done

python plot_figure4b.py \
  --umbrella 'results/rapid12h/umbrella/replica_*/fes_wham.csv' \
  --standard 'results/rapid12h/standard/replica_*/fes_standard.csv' \
  --inherited 'results/rapid12h/inherited/replica_*/fes_inherited.csv' \
  --ci-method student-t \
  --output results/rapid12h/figure4b_trace_rapid.png
```

This reduced result is a screening calculation, not an exact reproduction of
the source confidence intervals. Report its three-replica Student-t interval,
inspect transition counts and convergence with trajectory length, and extend
the runs before making a publication-level quantitative claim.

If the screening profiles disagree strongly, place their maxima at the edge of
the transition region, or fail to visit both basins, rerun three replicas using
the source paper's full trajectory lengths. This preset writes to
`results/full3`, leaving `results/rapid12h` untouched:

```bash
caffeinate -dimsu python run_rapid12h.py \
  --preset full3 \
  --workers 6 \
  --device cpu
```

The `full3` preset uses 500 ps standard WTMetaD trajectories, 250 ps
WTMetaD+IB trajectories, and 40 ps in every umbrella window with 10 ps
discarded. It retains only three repetitions, so use Student-t intervals and
do not claim the narrow uncertainty of the source ten-repetition calculation.
On the benchmark M2 Pro it is expected to require approximately eight hours.

To refine only umbrella sampling, which is the most controlled estimator in
this workflow, launch only its 90 independent window calculations:

```bash
caffeinate -dimsu python run_rapid12h.py \
  --preset full3 \
  --methods umbrella \
  --workers 6 \
  --device cpu
```

This performs 3.6 ns in total and is expected to take approximately five
hours. It does not launch standard WTMetaD or WTMetaD+IB jobs.

## 5. Production sampling

The scripts refuse to overwrite a nonempty output directory unless
`--overwrite` is supplied. Failed replicas retain `run_status.json`; do not
silently remove them from the reported stability count.

### A. Standard well-tempered metadynamics

The source protocol uses ten independent 500 ps replicas at 365.6 K, a 0.5 fs
step, Langevin friction 0.02 in ASE time units, Gaussian width 0.07 A, height
0.0158 eV, one hill every 100 fs, and bias factor 50:

```bash
for replica in $(seq 0 9); do
  python run_wtmetad.py \
    --mode standard \
    --replica "$replica" \
    --device cpu
done
```

### B. Well-tempered metadynamics with inherited bias

Each of ten independent 250 ps replicas starts from the exact bias accumulated
through the 16th active-learning iteration. The released filename uses the
zero-based index 15, `bias_after_iter_15.dat`; its SHA-256 value is recorded in
every run. New hills use width 0.07 A, height 0.0158 eV, a 100 fs pace, and
bias factor 80:

```bash
for replica in $(seq 0 9); do
  python run_wtmetad.py \
    --mode inherited \
    --replica "$replica" \
    --device cpu
done
```

### C. Umbrella sampling

For each replica, run 30 windows between the exact reactant and product IRC
endpoints. Each window is 40 ps at 365.6 K with a 0.5 fs step and
`kappa = 20 eV/A^2`. Frames are saved every 10 fs; the first 10 ps are removed
during analysis.

```bash
for replica in $(seq 0 9); do
  for window in $(seq 0 29); do
    python run_umbrella_window.py \
      --replica "$replica" \
      --window "$window" \
      --device cpu
  done
done
```

The windows and replicas are independent jobs and may be distributed by a
scheduler. Do not launch all jobs concurrently on one CPU or one GPU without
controlling thread and memory oversubscription.

The complete source-matched calculation contains 5 ns of standard WTMetaD,
2.5 ns of inherited-bias WTMetaD, and 12 ns of umbrella sampling: 19.5 ns in
aggregate, or 39 million force evaluations at 0.5 fs. Run the complete protocol
only after the pilot and independent error checks pass.

## 6. Reconstruct and compare the free energies

Standard WTMetaD is reconstructed from the well-tempered hills. Inherited-bias
WTMetaD is reconstructed by final-bias reweighting with the paper's 0.02 A
continuous-kernel bandwidth. The reweighting calculation freezes the final
bias and does not deposit analysis-time hills.

```bash
for replica in $(seq 0 9); do
  tag=$(printf "%02d" "$replica")
  python reconstruct_wtmetad.py \
    --replica-root "results/standard/replica_${tag}"
  python reconstruct_wtmetad.py \
    --replica-root "results/inherited/replica_${tag}"
  python analyze_umbrella.py \
    --replica-root "results/umbrella/replica_${tag}"
done
```

Generate the Figure 4b comparison after all stable replicas have been
analyzed:

```bash
python plot_figure4b.py
```

The plot aligns every profile to its reactant-basin minimum, averages only the
explicitly supplied complete profiles, and reports a 95% normal-approximation
confidence interval across independent replicas. It also shows the published
PBE0-D3BJ/def2-SVP qRRHO barrier (26.6 kcal/mol) and experimental result
(29.2 +/- 1.1 kcal/mol at 365.6 K). The source sampling results were
28.2 +/- 0.1 kcal/mol for umbrella sampling, 28.1 +/- 0.3 kcal/mol for standard
WTMetaD, and 28.4 +/- 0.4 kcal/mol with inherited bias.

The experimental and qRRHO/DFT values are scalar activation barriers rather
than coordinate-resolved free-energy profiles. Their numerical values and
provenance are stored in `references/figure4b_barriers.csv`. The plotting
script reads that file and writes the final PNG, PDF, machine-readable JSON,
and aggregated TRACE profiles in `results/figure4b_trace.{png,pdf,json,csv}`.

`analyze_umbrella.py` records the minimum and mean adjacent-window histogram
overlap. Poor overlap, a drifting barrier with trajectory length, or agreement
obtained only after excluding unexplained replicas invalidates a quantitative
free-energy claim.

Before accepting any profile, run the chemistry and sampling audit:

```bash
python audit_umbrella.py \
  --umbrella-root results/full3/umbrella \
  --output results/full3/audit
```

The audit checks the orthogonal coordinate `r1 + r2`, persistent molecular
bonds, unintended close contacts, effective sample size, equilibration drift,
temperature, and adjacent-window overlap. A detailed response to the current
`full3` failure and a staged umbrella-only rerun protocol are given in
`UMBRELLA_RECOVERY.md`.

## Scope of the reproduction

This workflow reproduces the released data partition, thermodynamic state,
collective variable, sampling parameters, and free-energy estimators while
replacing the source ACE potential by TRACE. It does not claim to recreate the
electronic-structure oracle calls or the historical active-learning selection
sequence. Agreement therefore tests whether the trained TRACE potential
supports the same finite-temperature observable; it is not a bitwise replay of
the original calculation.

## Sources

- Vitartas *et al.*, *Digital Discovery* **5**, 108-122 (2026),
  <https://doi.org/10.1039/D5DD00261C>.
- Version-4 data archive, <https://doi.org/10.6084/m9.figshare.28631591.v4>.
- `mlp-train` source archive used to verify the simulation defaults,
  <https://doi.org/10.6084/m9.figshare.25816864.v2>.
