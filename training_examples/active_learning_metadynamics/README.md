# Reactive potentials from active-learning/metadynamics data

This example prepares TRACE v2 training and validation inputs from version 4 of
the data released with Vitartas *et al.*, *Digital Discovery* **5**, 108-122
(2026), DOI [10.1039/D5DD00261C](https://doi.org/10.1039/D5DD00261C). The
publisher's immutable data record is
[10.6084/m9.figshare.28631591.v4](https://doi.org/10.6084/m9.figshare.28631591.v4).
The redistributed numerical inputs retain the source record's CC BY 4.0
license and must be attributed to its authors.

The two prepared reactions must be treated as separate potentials:

| Folder | Chemical problem | Electronic-structure labels | Charge, multiplicity |
|---|---|---|---|
| `r1_ch3cl_f` | R1: F- + CH3Cl SN2 substitution in implicit water | CPCM(water)-PBE0-D3BJ/def2-SVP | -1, 1 |
| `r2_methyl_shift` | R2: 2,2-dimethylisoindene methyl shift in the gas phase | PBE0-D3BJ/def2-SVP | 0, 1 |

R2 is a methyl rearrangement, not an SN2 reaction. It is included because R2 is
the system for which the paper reports the reproducible one-dimensional free
energy benchmark. The paper validates R1 against independent IRC and
US/AIMD configurations but does not publish an R1 free-energy barrier. The R3
glycosylation calculation is a separate, substantially larger explicit-solvent
problem with mixed compositions and a two-dimensional free-energy surface; it
is not silently mixed into either compact model here.

## 1. Prepare and verify the data

From the repository root, activate an environment in which this project is
installed, then run:

```bash
python training_examples/active_learning_metadynamics/prepare_data.py
```

The script downloads the exact Figshare files, checks their publisher-supplied
MD5 values, converts the numeric labels to ExtXYZ, makes a deterministic 85:15
train/validation split within each sampling source, rejects train/test geometry
overlap, and writes a SHA-256 manifest for every output. Existing generated data
are reproducible from the same command.

The conversion deliberately does **not** copy the 100 A periodic cell in the
publisher's convenience XYZ files. R1 and R2 are finite systems; every prepared
frame has `pbc=False` and a zero cell. Energies and forces are in eV and eV/A.
Stress is undefined and is not trained.

The source electronic energies are approximately -10,000 eV. To retain meV
energy differences in the float32 model, one fixed reference is subtracted from
all labels of a reaction:

\[
E_{\mathrm{label}}(\mathbf R)=E_{\mathrm{DFT}}(\mathbf R)-E_{\mathrm{ref}}.
\]

This transformation leaves forces, relative energies, canonical probabilities,
and free-energy differences unchanged. The exact `E_ref` is recorded in every
frame and in `data_manifest.json`. Set `energy_shift_per_atom: 0.0` as in the
provided configurations; applying a second energy shift would be incorrect.

## 2. Train R1: aqueous SN2 potential

The paper's WTMetaD-IB training pool contains 78 configurations. The default
configuration uses only this source so that its independent US/AIMD and IRC
tests remain directly comparable with Table S4 of the paper.

```bash
cd training_examples/active_learning_metadynamics/r1_ch3cl_f
python ../../../train.py --config config.yaml
```

For an exploratory higher-coverage fit using the union of the 78 WTMetaD-IB and
45 downhill configurations:

```bash
python ../../../train.py --config config_combined.yaml
```

Evaluate the best WTMetaD checkpoint on both untouched tests:

```bash
python ../evaluate_checkpoint.py \
  --model models/r1_trace_v2_wtmetad.pt \
  --data data/us_test.xyz \
  --output evaluation/us

python ../evaluate_checkpoint.py \
  --model models/r1_trace_v2_wtmetad.pt \
  --data data/irc_test.xyz \
  --output evaluation/irc
```

The source ACE model reported US/AIMD MAEs of 1.83 meV/atom for energy and
93.29 meV/A for force, and an IRC energy MAE of 2.12 meV/atom. These numbers
are references, not values assumed for TRACE. Do not start reactive dynamics
until the independently measured TRACE errors and the error-versus-CV plot are
acceptable, especially near the transition state.

The combined-checkpoint ASE/PLUMED workflow for the Figure 2b and Figure 3a
comparisons is documented in
[`r1_ch3cl_f/README.md`](r1_ch3cl_f/README.md).

## 3. Train R2: published free-energy benchmark

```bash
cd training_examples/active_learning_metadynamics/r2_methyl_shift
python ../../../train.py --config config.yaml

python ../evaluate_checkpoint.py \
  --model models/r2_trace_v2_wtmetad.pt \
  --data data/us_test.xyz \
  --output evaluation/us \
  --transition-min -0.20 \
  --transition-max 0.00
```

The WTMetaD-IB source set has 192 configurations. On the independent US/AIMD
set, the source ACE model reported energy and force MAEs of 1.49 meV/atom and
161.76 meV/A. The transition-state region was harder (4.27 meV/atom and
290.24 meV/A), so a low overall error alone is not sufficient to approve a
free-energy run.

The provided v2 architecture retains the source study's maximum correlation
order of four and a 5 A local cutoff. It is not a bitwise reconstruction of the
source linear ACE: that model used a 4 A many-body cutoff plus a separate 5 A
pair term, whereas TRACE applies its learned density correlations and local
fixed-environment attention within one 5 A cutoff. The data and thermodynamic
protocol are reproduced; the fitted model class is intentionally different.

## 4. Reproduce Figure 4

The complete R2 workflow is documented in
[`r2_methyl_shift/README.md`](r2_methyl_shift/README.md). It provides:

- the Figure 4a training/test coverage plot;
- source-matched standard and inherited-bias WTMetaD drivers;
- 30-window umbrella sampling with exact IRC endpoint centers;
- final-bias reweighting and a log-space WHAM implementation;
- independent-replica aggregation and the Figure 4b barrier comparison.

The complete protocol is 19.5 ns across all replicas and windows. Run the
documented short pilot and independent transition-region validation before
launching production sampling. Report every failed or unstable replica rather
than silently deleting it.

## 5. Validation required before interpreting a free energy

1. Preserve `us_test.xyz` exclusively for global and transition-region tests.
2. Check unbiased reactant, transition-state, and product trajectories for
   energy drift and unintended chemistry at both 0.5 fs and a shorter step.
3. Inspect temperature, energy, all molecular distances, CV history, and HILLS
   continuity for every biased replica.
4. Demonstrate barrier convergence with trajectory length and adequate overlap
   between neighboring umbrella windows.
5. Average independently converged profiles only after aligning the reactant
   basin, and report the stability count and uncertainty estimator.

The default configurations use AdamW because these reactive sets are very small
and the repository's prior small-data comparisons found it more stable than
Muon. Optimizer choice should be made from independent validation, never from
the final US/AIMD test or the desired barrier value.
