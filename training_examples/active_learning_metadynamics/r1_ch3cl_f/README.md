# R1 SN2 validation with TRACE, ASE, and PLUMED

This directory evaluates the combined R1 TRACE checkpoint for the aqueous
`F- + CH3Cl -> CH3F + Cl-` substitution studied by Vitartas *et al.* The
released atom order is not assumed by the Python analysis: C, F, and Cl are
identified from their chemical symbols.

The two paper panels require different calculations:

- Figure 2b is a relaxed two-dimensional potential-energy surface in
  `(r_C-F, r_C-Cl)`, with the WTMetaD-IB configurations superimposed.
- Figure 3a is a pointwise energy comparison along the independent IRC.

The commands below use `models/r1_trace_v2_combined.pt`, trained on the union
of the 78 WTMetaD-IB and 45 downhill configurations.

## Figure 3a: IRC energies

No molecular dynamics or PLUMED installation is needed:

```bash
cd training_examples/active_learning_metadynamics/r1_ch3cl_f
python plot_figure3a.py \
  --model models/r1_trace_v2_combined.pt
```

The script independently references the electronic-structure and TRACE curves
to the reactant endpoint. It writes the figure, pointwise energies, and summary
errors to `results/figure3a_trace.{png,pdf,csv,json}`. The shaded band is
`+/- 1 kcal/mol` for the complete six-atom system, not per atom.

## Figure 2b: WTMetaD coverage and relaxed energy surface

First inspect the generated PLUMED input:

```bash
python run_wtmetad.py \
  --output results/wtmetad_input_check \
  --write-only
```

For dynamics, activate an environment containing TRACE, ASE, and a
Python-enabled PLUMED build. If PLUMED was installed from source:

```bash
source ../../../../work/transformers-ace-venv/bin/activate
source /path/to/plumed2/sourceme.sh
which python
python -c "import plumed; print('PLUMED Python interface is available')"
python run_wtmetad.py \
  --model models/r1_trace_v2_combined.pt \
  --duration-ps 20 \
  --output results/wtmetad
```

`which python` must report the Python executable inside
`work/transformers-ace-venv`, not `miniconda3/bin/python`. The local PLUMED
2.8.3 binding was compiled for Python 3.12 and therefore cannot be imported by
the older base Python 3.8 environment.

### Rebuilding the PLUMED Python interface

The PLUMED command-line executable and its Python interface are separate build
products. If `which plumed` succeeds but `import plumed` fails after recreating
the virtual environment, configure PLUMED while that environment is active:

```bash
source /path/to/work/transformers-ace-venv/bin/activate
python -m pip install "Cython==0.29.37"

cd /path/to/plumed2
PYTHON_BIN="$(which python)" ./configure \
  --enable-modules=all \
  --enable-python \
  --prefix="${PWD}"
make -j4
make install
source "${PWD}/sourceme.sh"

python -c "import plumed; p = plumed.Plumed(); p.finalize(); print('PLUMED OK')"
```

PLUMED 2.8.3 requires the Cython 0.29 series; its wrapper source does not
compile with Cython 3. Rebuilding against a different Python interpreter
requires rebuilding the extension because the resulting module is specific to
that Python ABI.

The published R1 sampling variables are used: 300 K Langevin dynamics, a
0.5 fs step, `s = r_C-Cl - r_C-F`, Gaussian width 0.05 A, height `5 k_B T`,
10 fs hill spacing, bias factor 100, and an upper wall at
`(r_C-F + r_C-Cl)/2 = 2.5 A` with `kappa = 1000 eV/A^2`. Increase
`--duration-ps` only after a short trajectory is stable and conserves molecular
connectivity away from the intended substitution coordinate.

### COLVAR, free energy, and trajectory diagnostics

Analyze a completed run with the same PLUMED executable used for sampling:

```bash
python analyze_wtmetad.py \
  --run results/wtmetad \
  --plumed "$(which plumed)"
```

The script writes `colvar.{png,pdf}`, `free_energy.{png,pdf}`, and
`trajectory.{png,pdf}` to `results/wtmetad/analysis`, together with the plotted
CSV data and `analysis_summary.json`. The trajectory plot audits every C-H
distance as well as the instantaneous temperature. If molecular connectivity
is lost, the affected interval is marked and the final-bias profile is labeled
as diagnostic rather than a reportable free energy. A smooth profile alone is
not evidence of convergence: production results require stable connectivity,
repeated transitions, agreement across independent replicas, and a profile
that no longer changes with added sampling.

Runs made with an earlier version of `run_wtmetad.py` may contain a numerically
rescaled PLUMED time column. The analysis reconstructs physical time from
`sample_interval_fs` in `run_metadata.json`; new runs write PLUMED time directly
in picoseconds.

Build the relaxed TRACE energy surface. The default 26 by 26 scan contains 676
constrained optimizations and can be resumed:

```bash
python run_relaxed_pes_scan.py \
  --model models/r1_trace_v2_combined.pt \
  --output results/pes_scan.csv

# Continue an interrupted scan:
python run_relaxed_pes_scan.py \
  --model models/r1_trace_v2_combined.pt \
  --output results/pes_scan.csv \
  --resume
```

Then combine the surface, the 78 released WTMetaD-IB structures, and the new
biased trajectory:

```bash
python plot_figure2b.py \
  --scan results/pes_scan.csv \
  --model models/r1_trace_v2_combined.pt \
  --trajectory results/wtmetad/trajectory.traj
```

Omit `--trajectory` for a direct analogue of the final panel based only on the
published 78 configurations. The color scale is referenced to the TRACE-relaxed
reactant obtained from `structures/ch3cl_f.xyz`; the histogram is referenced to
the electronic-structure reactant endpoint of the independent IRC.

## Interpretation

The original Figure 2b documents an iterative WTMetaD-informed active-learning
history. A final combined TRACE checkpoint cannot reconstruct the oracle
selection decisions or inherited bias at iterations 0, 2, and 33. The workflow
here therefore reproduces the final-panel observables and tests whether the
trained TRACE potential provides stable reactive sampling; it does not claim a
bitwise reconstruction of that active-learning history.

The relaxed scan extends beyond parts of the training support. Treat
low-energy features in sparsely sampled regions as extrapolative until they are
confirmed by electronic-structure calculations. Likewise, agreement along the
IRC is necessary but not sufficient for a converged finite-temperature free
energy.

## Sources

- Vitartas *et al.*, *Active learning meets metadynamics: automated workflow
  for reactive machine learning interatomic potentials*, Digital Discovery
  **5**, 108-122 (2026), <https://doi.org/10.1039/D5DD00261C>.
- Published data archive, <https://doi.org/10.6084/m9.figshare.28631591.v4>.
- `mlp-train` source archive used to verify the sampling defaults,
  <https://doi.org/10.6084/m9.figshare.25816864.v2>.
