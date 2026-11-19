# Transformers-ACE 

Transformers-ACE is a research implementation of local equivariant transformer
potentials built on Atomic Cluster Expansion (ACE) descriptors. It predicts a
single invariant total energy; forces and stresses are obtained by exact energy
derivatives so they remain conservative and suitable for molecular dynamics.

The model combines:

- periodic-correct local neighbor geometry;
- C2-cutoff radial functions, spherical harmonics, and learned body-order-two
  through body-order-four Clebsch-Gordan density contractions;
- strictly local neighbor-set attention with invariant weights and equivariant
  geometric values; updated hidden states are never sent between atom centers;
- energy-derived forces and symmetric stress;
- automatic train/validation plots for energy, force, stress, and total loss.

TRACE v3 is available as an experimental, separately versioned architecture. It
uses fixed ACE edge and body-order moment tokens with tensorial center-to-
environment cross-attention; it does not propagate updated atomic states. See
[docs/TRACE_V3.md](docs/TRACE_V3.md) and the supplied CsPbI3 and water
configurations under `configs/`.

> **Research status:** This is experimental research software. Validate a
> checkpoint against independent structures, equations of state, phonons, and
> molecular-dynamics stability before using it for scientific conclusions.

## Branches

| branch | contents | use it to |
|---|---|---|
| `CPU` | the published TRACE-v2 release, unchanged | reproduce the results in the manuscript exactly |
| `GPU` | CPU plus the CUDA/AOT deployment path and the corrected model | run new work, GPU MD, and current development |

`CPU` is preserved verbatim so every published number stays reproducible. `GPU`
carries corrections that **change the numbers**, so results from the two
branches are not interchangeable. In particular the layer-scale parameter was
tightened from one value per irrep *component* to one per irrep *copy*, which
makes a rotationally inconsistent state unrepresentable rather than merely
averaged away at inference; the checkpoints under `training/` on `GPU` are
retrained accordingly and are **not** the ones behind the published free-energy
results. Use `CPU` for reproduction, `GPU` for new science.

## Install

Python 3.10-3.13 and the stable e3nn 0.6 series are supported. Full download,
virtual-environment, Apple Silicon, and optional accelerator instructions are in
[docs/INSTALL.md](docs/INSTALL.md).

```bash
git clone https://github.com/paramvir3/TRACE.git
cd TRACE
./install.sh
source .venv/bin/activate
pytest tests/ -q
```

`install.sh` selects an interpreter, creates `.venv`, installs the pinned
dependency set and the package in editable mode, then verifies the result. It is
safe to re-run.

To install by hand instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .

python -c "import torch, e3nn; print(torch.__version__, e3nn.__version__)"
```

`--no-deps` on the final step keeps the pins in `requirements.txt` intact;
without it pip may resolve looser floors from `setup.py` and silently upgrade
`e3nn` or `torch`. A checkpoint trained under one e3nn series does not reproduce
the same energies under another.

The canonical Python API is `transformers_ace`. The original `flashace` import
is retained so existing scripts and `.pt` checkpoints continue to work.

## Train

Prepare an ASE-readable extended XYZ trajectory containing energies, forces,
periodic cells, and optionally stresses. Copy and edit the example configuration:

```bash
cp configs/cspbi3.yaml my_training.yaml
python train.py --config my_training.yaml
```

To train the v3 research architecture on the supplied local data:

```bash
python train.py --config configs/trace_v3_cspbi3.yaml
python train.py --config configs/trace_v3_water_ccsd.yaml
```

Training saves the configured `.pt` checkpoint. By default it also writes:

```text
plots/training_curves.png
plots/training_history.csv
```

`model.pt` is the best validation checkpoint after the stress-weight ramp;
`model_last.pt` records the final epoch. The supplied CsPbI3 configuration uses
a deterministic blocked trajectory split so adjacent frames are not scattered
between training and validation. Set `optimizer: "muon"` in the YAML to use
Muon on transformer hidden matrices with auxiliary AdamW for embeddings,
readout layers, radial/tensor-product support weights, biases, normalization
weights, and other non-hidden parameters.

Muon uses `muon_ns_dtype: "auto"` by default: Newton--Schulz arithmetic stays
at the model precision on CPU and uses BF16 on CUDA hardware that supports it,
matching the reference Muon throughput path. Set `muon_ns_dtype: "float32"`
when an exactly FP32 optimizer trajectory is required.

## ASE Calculator

```python
from ase.io import read
from transformers_ace import TransformersACECalculator

atoms = read("POSCAR")
atoms.calc = TransformersACECalculator("model.pt", device="cpu")

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```

## CsPbI3 Test

Five periodic CsPbI3 polymorph structures and scripts for phase energies and
geometry optimization are included under `tests/cspbi3`. See
[docs/CSPBI3_TEST.md](docs/CSPBI3_TEST.md) for the complete workflow.

```bash
python tests/cspbi3/evaluate_phases.py \
  --model /absolute/path/to/model.pt \
  --device cpu \
  --reference minimum
```

## LAMMPS and PLUMED

Native LAMMPS support is included for rare-event workflow testing. Export a
checkpoint to TorchScript, patch/build LAMMPS with `pair_style transformers_ace`,
and attach PLUMED as a normal LAMMPS fix. See [docs/LAMMPS.md](docs/LAMMPS.md).
The working CsPbI3 standalone LAMMPS smoke test is in
[`tests/run_lammps`](tests/run_lammps).

### GPU performance and scaling

Measured on one NVIDIA H100 80GB (driver 550.163.01, torch 2.12.1+cu129,
LAMMPS 30 Mar 2026, single GPU, no MPI). CsPbI3, NVE, 2 fs timestep, float32.
Both pair styles run the **same checkpoint**, so this compares runtimes and not
weights. `transformers_ace/aot` loads an ahead-of-time compiled `.pt2` package
whose force derivative is baked into the graph; `transformers_ace` is the
TorchScript path.

| atoms | `transformers_ace/aot` | `transformers_ace` | speed-up | us/atom/step | ns/day |
|---:|---:|---:|---:|---:|---:|
| 10,240 | 29.25 ms/step | 49.63 ms/step | 1.70x | 2.856 | 5.91 |
| 20,480 | 51.37 ms/step | 81.20 ms/step | 1.58x | 2.508 | 3.36 |
| 40,960 | 100.19 ms/step | 148.66 ms/step | 1.48x | 2.446 | 1.72 |
| 102,400 | 235.84 ms/step | 349.96 ms/step | 1.48x | 2.303 | 0.73 |

Scaling is sub-linear throughout: doubling the system costs 1.76x then 1.95x
the time, and 2.5x the atoms costs 2.35x. Cost per atom therefore *falls* with
size, from 2.856 to 2.303 us. The speed-up is largest at 10k atoms and settles
at 1.48x, which is the expected signature of removing kernel-launch overhead --
it dominates when kernels are small and matters less once real compute does.

Both styles agree on the physics. Over 200 independently integrated steps at
10,240 atoms the total energies differ by a constant 0.03 eV out of 325,917 eV
(9e-8 relative, so accumulation order rather than divergence), and temperature
agrees to six figures. Energy drift is at most 7.7e-8 relative across all sizes.

**Size ceiling.** 100k atoms exports and runs; 150k and 200k do not export on an
80 GB card. The limit is *export-time*, not run-time: inference costs a measured
7.55 GiB per 1M edges, so a 200k-atom system needs only ~32 GiB to run, but
export holds the eager reference, the traced graph and the Inductor compile at
once -- roughly 3x that. Moving the reference off the device, or compiling on a
larger card, should lift the ceiling without touching the model.

Full logs and method: [`tests/run_lammps/benchmark_aot/results_h100`](tests/run_lammps/benchmark_aot/results_h100).

### Building the AOT pair style

```bash
python -m transformers_ace.aot_deploy \
  --checkpoint training/h100_run/model_h100.pt \
  --output trace.pt2 --type-map Cs Pb I \
  --max-atoms 18432 --max-edges 217088 --device cuda

cd lammps/pair_style && ./patch_lammps.sh /path/to/lammps
```

The export verifies the compiled package against the eager model and refuses to
write one that disagrees. `patch_lammps.sh` prints the exact CMake invocation
for a CUDA host, including the three settings a stock image needs
(`CUDAARCHS`, `MKL_INCLUDE_DIR`, and a `libnvrtc.so` symlink the toolkit omits).

Shapes are fixed at compile time, so `--max-atoms` must cover local **plus
ghost** atoms and `--max-edges` the directed edges within `r_max`; the pair
style aborts with the required value rather than silently truncating. Oversizing
costs real compute, since padded edges are still evaluated.

For MPI and multi-GPU runs, use `newton on` in the LAMMPS input. The native
pair style evaluates local owned-atom energies with ghost atoms in the
neighborhood and lets LAMMPS reverse-communicate ghost force components. With
`pair_style transformers_ace device auto`, each MPI rank maps to
`local_rank % visible_gpu_count`, so a typical one-node four-GPU run is:

```bash
mpirun -np 4 /path/to/lammps/build/lmp -in in.transformers_ace
```

Install PLUMED first:

```bash
brew install pkg-config

git clone https://github.com/paramvir3/plumed2.git
cd plumed2
plumed_dir="${PWD}"

./configure --enable-modules=all --prefix="${PWD}"
make -j4
make install
source "${PWD}/sourceme.sh"
export PKG_CONFIG_PATH="${plumed_dir}/lib/pkgconfig:${PKG_CONFIG_PATH}"
```

Patch and configure LAMMPS:

```bash
git clone --depth=1 https://github.com/lammps/lammps
cd lammps

cd /path/to/Transformers-ACE/lammps/pair_style
bash patch_lammps.sh /path/to/lammps

cd /path/to/lammps
mkdir -p build
cd build

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DLAMMPS_EXCEPTIONS=yes \
  -DCMAKE_INSTALL_PREFIX="$(pwd)" \
  -DBUILD_MPI=ON \
  -DPKG_MANYBODY=yes \
  -DPKG_EXTRA-FIX=yes \
  -DPKG_EXTRA-PAIR=yes \
  -DPKG_EXTRA-DUMP=yes \
  -DPKG_MOLECULE=yes \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)');${plumed_dir}" \
  -DPKG_PLUMED=yes \
  -DPLUMED_MODE=shared \
  -DDOWNLOAD_PLUMED=no \
  ../cmake

make -j
```

```bash
python -m transformers_ace.deploy \
  --checkpoint training/model.pt \
  --output model.transformers_ace.pt \
  --type-map Cs Pb I \
  --example-structure tests/cspbi3/structures/cubic_alpha_phase.vasp
```

### PLUMED Rare-Event Example

The repository includes an explicit LAMMPS plus PLUMED biased-dynamics example
for a non-perovskite delta CsPbI3 to perovskite CsPbI3 transition:

```text
tests/run_lammps/test_plumed_cspbi3
```

It contains the LAMMPS input, PLUMED `DSFTHREE` structure-factor collective
variable, and the 640-atom CsPbI3 starting structure. The exact macOS
LAMMPS+PLUMED CMake build recipe is documented in
[docs/LAMMPS.md](docs/LAMMPS.md).

<!-- BEGIN TRACE-Electrostatics release block -->
## TRACE-Electrostatics

Long-range electrostatics, polarisation, charge transfer and integer charge states (redox) for TRACE. TRACE's
attention layers supply every per-atom and per-bond parameter. The electrostatic energy is minimised exactly, in
float64. Forces, stress, Born charges and polarisabilities are therefore exact derivatives of one energy.
Research release: [checkpoints (v0.1)](https://github.com/paramvir3/TRACE/releases/tag/trace-electrostatics-v0.1).

### Model

**Bound sector** (`trace_electrostatics`):

```
E_elec = min_m ½ mᵀ K m + cᵀ m,   m = (t, μ̂, Θ̂),   q = B diag(f) t + ρ
```

- t are split charges on bonds, B is the bond incidence matrix and f the dissociation envelope.
- μ̂ and Θ̂ are the induced dipoles and quadrupoles.
- K is the Gaussian-multipole Coulomb (Gram) matrix plus the hardness terms. It is positive definite under the
  Gaussian kernel.

**Free-charge sector** (`trace_redox`, SPEC §0). n is a configuration of integer unit charges:

```
E(n) = Σ_u ε_u(n_u) + ½ q(n)ᵀ J q(n) − ½ s(n)ᵀ K(n)⁻¹ s(n) + Σ_{u<v} O_uv(n_u, n_v)
H_nm = ± σ_nm f_nm τ_uv,      E_total = E_short + F,      F = −kT ln Tr exp(−H/kT)
```

- ε_u: the isolated-ion energies.
- O_uv: Gordon–Kim overlap and penetration, tabulated from first-principles charge-state densities.
- τ_uv: the one-electron hops.

### Benchmarks

The test sets are the 4G-HDNNP charge-transfer sets (Ko et al., *Nat. Commun.* **12**, 398 (2021); Materials Cloud
2020.137, CC BY 4.0) and MgO phonons against PBE (Togo, NIMS MDR, CC BY 4.0).
- **Errors** are RMSE on held-out test sets.
- **PFN NequIP values** were digitised from PFN's blog figure.
- **Seeds:** the Au₂/MgO models were trained with two seeds each.

**Cluster sets, PFN training sizes.** Pooled force RMSE (eV/Å) and the NaCl charge-transfer shift (Å). One model
per system, one seed.

| | Ag₃± | NaCl | C₁₀H₂/C₁₀H₃⁺ | NaCl shift |
|---|---|---|---|---|
| DFT | | | | 0.107 |
| 4G-HDNNP | 0.031 | 0.0329 | 0.0757 | 0.109 |
| best PFN NequIP variant | 0.0204 | 0.0104 | 0.0495 | |
| TRACE (local) | 1.892 | 0.0553 | 0.0805 | 0 |
| **TRACE-Electrostatics** | **0.0159** | **0.0068** | **0.0356** | **0.104** |

**Au₂ on MgO(001), Al-doped and undoped** (Table II of Ko et al.). The dopant moves an electron onto Au₂ and flips the
preferred adsorption geometry. wet = E(wetting) − E(non-wetting), in meV. T5 = wet_d − wet_u. Each TRACE cell gives
seed 42 / seed 43.

| | wet_u | wet_d | T5 (% DFT) | test F RMSE undoped / doped (eV/Å) | Au–Au force at 2.089 Å (eV/Å) |
|---|---|---|---|---|---|
| DFT (PBE) | +928.9 | −2.7 | 100 | | +15.25 |
| 4G-HDNNP | +975.4 | −41.0 | 109.1 | | |
| step 34b (current equations) | +610.4 / +729.7 | +92.3 / −27.7 | 55.6 / 81.3 | 0.0845 / 0.1242; 0.0616 / 0.1112 | +14.25 / +15.82 |
| step 31 | +851.0 / +920.4 | −177.4 / −75.7 | 110.4 / 106.9 | 0.0614 / 0.0867; 0.0776 / 0.1045 | — / +6.18 |

**Bulk MgO phonons against PBE.** The NAC correction uses each model's own Z* and ε∞. Force constants are on PBE's
own 4×4×4 supercell (512 atoms). There are seven pre-registered criteria (P0–P6).

| | Z*(Mg) | ε∞ | TO / LO at Γ (cm⁻¹) | path RMS, NAC (cm⁻¹) | criteria |
|---|---|---|---|---|---|
| PBE | 1.992 | 3.235 | 373.3 / 681.3 | | |
| step 34b s42 / s43 | 1.950 / 1.961 | 3.334 / 3.300 | 382.1 / 669.3; 370.3 / 667.4 | 8.5 / 8.8 | 7 / 7 both |
| step 31 s42 / s43 | 2.402 / 2.534 | 3.025 / 3.002 | 384.3 / 807.9; 398.7 / 851.5 | 28.9 / 53.9 | 3 / 7, 2 / 7 |

### Verdict and limitations

- **Cluster sets:** the lowest force error of all compared models on all three sets, and the DFT charge-transfer shift
  in NaCl. Caveats: one seed, separate models, an older equation stack.
- **Au₂/MgO:** no model passes every pre-registered criterion on both seeds.
  - Step 34b is right on the phonons, the Au–Au repulsion and gas-phase Au₂. But wet_u is 199–319 meV low.
  - Step 31 is right on Table II. Its phonons and its Au–Au wall fail.
  - Its pooled test force error, 0.075–0.106 eV/Å from 300 training structures, has not reached 4G-HDNNP's 0.067 from
    about 4,500.
- **Charge on Au₂:** the doped Au₂ carries a full electron (−1.00 e against Hirshfeld's −0.27; Hirshfeld is
  partition-dependent).
- **Not implemented:** ASE only (no LAMMPS export with electrostatics); trained at Γ.

Every result, failures included: [BENCHMARKS.md](docs/electrostatics/BENCHMARKS.md).

### Reproduce

```bash
git clone -b trace-electrostatics https://github.com/paramvir3/TRACE.git && cd TRACE
python -m pip install -r requirements.txt -r requirements-dev.txt && python -m pip install --no-deps -e .
python -m pytest tests -q -p no:cacheprovider --ignore=tests/run_lammps
```

- Data download and checksums, configs, training and scoring: [REPRODUCE.md](docs/electrostatics/REPRODUCE.md).
- Checkpoints: [MODELS.md](docs/electrostatics/MODELS.md).
- User guide: [TRACE_ELECTROSTATICS.md](TRACE_ELECTROSTATICS.md).
<!-- END TRACE-Electrostatics release block -->

## Compatibility

These imports are equivalent:

```python
from transformers_ace import TransformersACE, TransformersACECalculator
from flashace import FlashACE, FlashACECalculator
```

## Acknowledgements

Transformers-ACE was developed with assistance from OpenAI Codex.

## License

See [LICENSE](LICENSE).

## Citation

[![arXiv](https://img.shields.io/badge/arXiv-2607.25652-b31b1b.svg)](https://arxiv.org/abs/2607.25652)

```bibtex
@misc{ahlawat2026trace,
  title         = {Transformer Atomic Cluster Expansion: TRACE},
  author        = {Ahlawat, Paramvir},
  year          = {2026},
  eprint        = {2607.25652},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.mtrl-sci},
  doi           = {10.48550/arXiv.2607.25652},
  url           = {https://arxiv.org/abs/2607.25652}
}
@book{Varshalovich,
  author  = {Varshalovich, D. A. and Moskalev, A. N. and Khersonskii, V. K.},
  title   = {Quantum Theory of Angular Momentum: Irreducible Tensors, Spherical Harmonics, Vector Coupling Coefficients, 3nj Symbols}
}
```
