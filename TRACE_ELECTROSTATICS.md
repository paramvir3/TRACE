# TRACE-Electrostatics

TRACE-Electrostatics adds one master equation to the TRACE attention potential.
The equation covers long-range electrostatics, polarisation, charge transfer
along bonds, integer charge states (redox), quadrupoles and optional
dispersion. TRACE's equivariant attention layers supply the per-atom and
per-bond parameters. The package minimises the electrostatic energy exactly
and adds it to the TRACE energy. Forces, stress, polarisabilities and Born
charges are then exact derivatives of one energy.

This is a **research release**. Read the status paragraph below before you use
a checkpoint for science.

Contents: [Status](#status) · [What is in this release](#what-is-in-this-release) ·
[The equations](#the-equations) · [Installation](#installation) ·
[Quick start](#quick-start) · [Benchmark summary](#benchmark-summary) ·
[Data, licences and attribution](#data-licences-and-attribution) ·
[How to cite](#how-to-cite) · [Limitations](#limitations) ·
[Tests](#tests) · [Documentation map](#documentation-map)

---

## Status

**No released model passes all of its pre-registered criteria on both training
seeds.** The headline benchmark is the 4G-HDNNP Au₂/MgO(001) test of Ko et al.
(Nat. Commun. 12, 398 (2021)). On it, the two candidate models fail in
different ways:

- **Step 34b** is the model with the current equations (runs
  `AuMgO_twosite_bf055_e1kw_bondlaw_r5_s{42,43}`). It passes the MgO phonons
  against PBE: 6 / 7 on the registered 2×2×2 protocol. With P5 read on PBE's
  own 4×4×4 (512-atom) supercell, the like-for-like reading, it passes 7 / 7
  on both seeds. It passes the Au–Au wall of the compressed dimer: bond force
  +14.25 / +15.82 eV/Å at 2.089 Å, against DFT's +15.25. Gas-phase Au₂ is
  right on seed 43: the EA and IP errors are −0.03 to 0.00 eV and +0.07 to
  +0.08 eV over 2.30–2.80 Å. Seed 42 is not: its errors are +0.20 to +0.43 eV
  and +0.56 to +0.59 eV.
- **Step 34b's Table II is short of DFT.** Its T5 = E_wet(doped) −
  E_wet(undoped) is −757.4 meV (seed 43) and −518.1 meV (seed 42), against
  DFT's −931.6. That is **81 % and 56 % of DFT**. The run fails its
  pre-registration on S2 (Table II), S5 (the crystal factor and ε∞) and S8
  (the undoped forces without the five compressed dimers), on both seeds.
- **Step 31** comes closest to DFT on Table II. On seed 43: wet_u +920.4 meV
  (DFT +928.9), wet_d −75.7 meV (DFT −2.7), T5 106.9 % of DFT. Seed 42 fails
  two of its criteria (wet_u +851.0, wet_d −177.4, T5 110.4 %).
- **Step 31 has known defects.** Its MgO phonons pass 3 / 7 and 2 / 7: Z*(Mg)
  is 2.402 / 2.534 against PBE's 1.992, and LO is 807.9 / 851.5 cm⁻¹ against
  681.3. Its compressed Au₂ falls into an ionic Au⁺Au⁻ configuration between
  2.10 and 2.06 Å (seed 43). It was trained with the legacy, non-extensive forms of the
  equations.

Sources for these numbers:
- the score files `runs/<run>/scores/{T5_wetting,S7_S8,S6_au2_ea_ip,P_phonons_pbe,P_phonons_pbe_sc4}.txt`
  of the published runs, which [`experiments/benchmarks_4g/score_model.sh`](experiments/benchmarks_4g/score_model.sh)
  writes;
- the lab notebook [`experiments/benchmarks_4g/README.md`](experiments/benchmarks_4g/README.md),
  entries "Step 34b scored", "Step 31 scored" and "Scored: the Au–Au wall";
- for step 31's phonons, which were scored after the run and are not in its
  `scores/` directory: [`experiments/benchmarks_phonons/README.md`](experiments/benchmarks_phonons/README.md),
  entry "Step 31's phonons, scored for the first time" (outputs
  `experiments/benchmarks_phonons/runs/step31_<run>{,_sc4}/`).

The full tables are in
[`docs/electrostatics/BENCHMARKS.md`](docs/electrostatics/BENCHMARKS.md).

On the 4G cluster benchmarks (Ag₃, NaCl, the carbon chain, at PFN's split
sizes), the electrostatic arm has a lower pooled test force RMSE than every
published model we compare with. Those runs use an older option set; see
BENCHMARKS.md.

---

## What is in this release

| path | what it is |
|---|---|
| [`trace_electrostatics/`](trace_electrostatics/) | The bound sector. Charge equilibration with Gaussian charges on atoms and split charges on bonds; induced dipoles and quadrupoles; optional many-body dispersion. Open, slab and 3-D periodic boundaries; dense, matrix-free and PME routes; implicit derivatives of the converged minimum. Contract: [`SPEC.md`](trace_electrostatics/SPEC.md) (§17–21 current; §2 legacy). |
| [`trace_redox/`](trace_redox/) | TRACE-Redox, the free-charge (redox) sector, SPEC 25 of `trace_electrostatics`. Each atom or unit carries an integer charge; configurations mix through one-electron hops and are thermally averaged. It uses `trace_electrostatics` unchanged. It reads first-principles tables from `trace_redox/data/` at run time. Contract: [`SPEC.md`](trace_redox/SPEC.md); §0 states the model in three equations. |
| `flashace/model.py`, `flashace/checkpoint.py`, `flashace/charge_state.py`, `flashace/calculator.py` | The hooks into TRACE. `attach_electrostatics` / `trace_redox.attach_redox` put the module inside the model's `forward`, and its energy is added before the autograd call for forces and stress. Checkpoints carry the resolved electrostatics and redox configuration. `TransformersACECalculator` rebuilds it, sets float64, and reads the charge-state keys of an `Atoms` object (`info['charge']`, fragments, spins). |
| [`train.py`](train.py) | An `electrostatics:` block in the training YAML switches the package on; a `redox:` sub-block selects TRACE-Redox. The block also takes the observable losses and the response priors (ε∞ on bulk cells, α on isolated atoms). |
| `transformers_ace/` | The public API. The LAMMPS TorchScript and AOT exports **refuse** a model with electrostatics (`transformers_ace.deploy.refuse_electrostatics`). |
| `tests/electrostatics/`, `tests/redox/`, `tests/test_train_*.py`, `tests/test_score_phonons_protocol.py` | The test suites (see [Tests](#tests)). |
| `tools/electrostatics/` | The builders of the first-principles tables (PySCF) and the phonon driver `mgo_phonons.py`. |
| [`experiments/benchmarks_4g/`](experiments/benchmarks_4g/) | The 4G-HDNNP / PFN benchmark kit: data preparation, training configs of the released runs (`configs/`), scorers, probes and the lab notebook. |
| [`experiments/benchmarks_phonons/`](experiments/benchmarks_phonons/) | The MgO phonon benchmark against PBE: `score_phonons.py`, the four DFT reference bands (`reference/`), the bulk-data builders and the lab notebook. |

Checkpoints are not in git. They are attached to the GitHub release as
`<run_name>.pt` with a `SHA256SUMS` file; see
[`docs/electrostatics/MODELS.md`](docs/electrostatics/MODELS.md).

---

## The equations

The bound sector (`trace_electrostatics`, SPEC 1.5.1 and 17) is one convex
minimum:

```
E_elec = min_{m ∈ V}  ½ mᵀ K m + cᵀ m   (+ E_disp)

m = (t, μ̂, Θ̂)     split charges on bonds, reduced induced dipoles and quadrupoles
q = B diag(f) t + ρ    atomic charges: bond incidence B, dissociation envelope f, fixed carrier ρ
K                  Coulomb (Gram) matrix of the Gaussian multipoles + on-site and bond hardness;
                   positive definite by theorem under the Gaussian kernel (SPEC 12, 14)
```

The free-charge sector (`trace_redox`, SPEC §0) gives a configuration n of
integer unit charges this energy:

```
E(n) = Σ_u ε_u(n_u)  +  ½ q(n)ᵀ J q(n)  −  ½ s(n)ᵀ K(n)⁻¹ s(n)  +  Σ_{u<v} O_uv(n_u, n_v)     (0.1)
          ions             Coulomb             induction                 overlap

H_nn = E(n),   H_nm = ± σ_nm f_nm τ_uv   (m = n with one electron moved u → v)               (0.2)

E_total = E_short(R) + F,   F = −kT ln Tr exp(−H / kT)                                       (0.3)
```

- **Ions.** ε_u is the exact isolated ion (tabulated ionisation energies and
  electron affinities) plus a learned, gated shift.
- **Induction.** The bound sector of the first equation is eliminated exactly.
- **Overlap.** O_uv is tabulated from first-principles charge-state densities:
  their Coulomb penetration and their Gordon–Kim Pauli energy.
- **Hops.** τ_uv are one-electron couplings, learned or Harrison-anchored.
- **Later terms.** SPEC §0 lists the terms added after these three equations
  and maps every later section to its term.

Established and new, as the SPECs mark them:
- **NEW-TO-TRACE:** evaluating classical ion physics per integer
  configuration with first-principles charge-state densities, inside a
  canonical ensemble mixed by learned hops; the exact elimination of the bound
  sector per configuration; and the frame-pair and two-site sectors (§39, §40).
- **ESTABLISHED (cited ingredients):**
  - the Gordon–Kim overlap model (Gordon and Kim, J. Chem. Phys. 56, 3122 (1972));
  - polarisable-ion models (Madden and Wilson, Chem. Soc. Rev. 25, 339 (1996));
  - split-charge equilibration, and its redox form (Dapp and Müser, J. Chem.
    Phys. 139, 064106 (2013));
  - Harrison's tight-binding scaling of the couplings (Harrison, *Electronic
    Structure and the Properties of Solids*, 1980).

Where to read the physics:
- [`docs/electrostatics/METHODS.md`](docs/electrostatics/METHODS.md): the
  methods, written for a reader of the paper.
- [`trace_redox/SPEC.md`](trace_redox/SPEC.md) §0: the current model in three
  equations, and where every section sits.
- [`trace_electrostatics/SPEC.md`](trace_electrostatics/SPEC.md): the bound
  sector. §17–21 are current; §2 is legacy.
- [`docs/electrostatics.md`](docs/electrostatics.md): the user guide, with
  runnable examples and settings.
- [`docs/electrostatics_api.md`](docs/electrostatics_api.md): every public
  function and class, with its signature.
- [`trace_electrostatics/CHANGELOG.md`](trace_electrostatics/CHANGELOG.md):
  what changed, and when.

---

## Installation

```bash
git clone https://github.com/paramvir3/TRACE.git
cd TRACE
git checkout trace-electrostatics
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install --no-deps -e .
python -m pip install phonopy        # only for the phonon benchmark (released numbers: phonopy 2.17.2)
```

`setup.py` installs `trace_redox/data/*.npz` and `*.json` as package data.
`two_site.json` records the sha256 of four of those files, so they must stay
byte-identical.

**float64 is mandatory.** `trace_electrostatics` and `trace_redox` refuse
float32. The calculator and `train.py` set `torch.set_default_dtype(torch.float64)`
themselves. If you build a model in Python, set it before any e3nn module
exists; a later `.double()` is a silent no-op for e3nn (guide, section 2).
`train.py` refuses `use_amp: true` with electrostatics.

**The environments of record.** Every reported number was measured in one of
these two environments. Neither is the `requirements.txt` set.

| | CPU (training of steps 26 and 31, the cluster runs, and every score) | GPU host (training of steps 33 and 34b) |
|---|---|---|
| hardware | Apple M2 Pro, 10 cores, 16 GB | one NVIDIA RTX 5060 Ti, 16 GB (float64 at about 1/64 of float32) |
| Python | 3.8.16 | 3.12 |
| torch | 2.2.0 | 2.11.0+cu128 |
| e3nn | 0.4.4 | 0.6.0 |
| ase | 3.23.0 | 3.29 |
| numpy / scipy | 1.24.2 / 1.10.0 | numpy 2.5 |
| phonopy / PyYAML | 2.17.2 / 6.0 | — |

`requirements.txt` pins torch 2.12.0, e3nn 0.6.0, ase 3.28.0 and numpy 2.4.6.
No reported run or score used that set, and the electrostatics suites have
not been run on it. To match the CPU environment of record, install the
versions in the table, then `python -m pip install --no-deps -e .`. That
install recipe has not itself been run on a clean machine. `setup.py` asks for
`ase>=3.28`, which the CPU environment of record (ase 3.23.0) does not meet;
`--no-deps` leaves the installed ase alone. PySCF is needed only to rebuild the
first-principles tables (`tools/electrostatics/`); the shipped tables are the
reproduction path.

e3nn 0.4 and 0.6 are both accepted (`setup.py`: `e3nn>=0.4.4,<0.7.0`). The
GPU-trained models (e3nn 0.6.0) score on the CPU environment (e3nn 0.4.4):
energies agreed with the GPU to 4e-12 – 1.1e-10 eV/atom on six structures, and
forces to 4e-10 – 1.6e-8 eV/Å on five of them. The sixth, a doped slab, differed
by 1.3e-7 eV/Å with the new terms on and by 1.5e-7 with them off. Source:
`experiments/benchmarks_4g/README.md`, "Step 34's GPU pre-flight, scored".

Run the test suites first on any new machine ([Tests](#tests)).

---

## Quick start

All commands run from the repository root unless stated otherwise. Outputs
shown were produced on 2026-11-18 from this release tree, on the CPU
environment of record. torch 2.2 also prints TorchScript `UserWarning` lines at
import; they are harmless.

### 1. An ASE calculator on a released checkpoint

Download `AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43.pt` (step 34b, stage B,
seed 43) and `SHA256SUMS` from the release, then check the file
([`docs/electrostatics/MODELS.md`](docs/electrostatics/MODELS.md)). Its sha256
is `740ea30cca04950d153ebddc4301b9926ea2e97e89a8d6b09fa8f4e24a79c297`.

```python
import time

import numpy as np
from ase import Atoms
from ase.build import bulk
from transformers_ace import TransformersACECalculator

# A released checkpoint (docs/electrostatics/MODELS.md). The calculator sets float64 itself.
calc = TransformersACECalculator(model_path="AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43.pt", device="cpu")


def energy(atoms, charge):
    """Energy (eV) at a net charge (e); info["charge"] is read by the charge-state reader."""
    atoms = atoms.copy()
    atoms.info["charge"] = float(charge)
    atoms.calc = calc
    return atoms.get_potential_energy()


# 1. Gas-phase Au2, open boundaries, in three charge states.
au2 = Atoms("Au2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.47]], pbc=False)
e0, e_anion, e_cation = (energy(au2, q) for q in (0, -1, +1))
print(f"Au2 at 2.47 A: EA = {e0 - e_anion:.3f} eV, IP = {e_cation - e0:.3f} eV")

# 2. Bulk MgO at the PBE lattice constant: 64 atoms, periodic, every atom displaced by 0.01 A.
mgo = bulk("MgO", "rocksalt", a=4.255656884758435, cubic=True).repeat(2)
mgo.rattle(stdev=0.01, seed=0)
mgo.calc = calc
start = time.perf_counter()
e_mgo, f_mgo = mgo.get_potential_energy(), mgo.get_forces()
print(f"MgO, {len(mgo)} atoms: E = {e_mgo:.4f} eV, max |F| = {np.abs(f_mgo).max():.4f} eV/A, "
      f"net force {np.abs(f_mgo.sum(axis=0)).max():.1e} eV/A, {time.perf_counter() - start:.1f} s")
```

Output, re-run on 2026-11-18 (5.3 s wall for the whole script):

```
Loading Transformers-ACE from AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43.pt on cpu...
Checkpoint carries trace_electrostatics: default dtype set to float64.
Au2 at 2.47 A: EA = 2.014 eV, IP = 9.294 eV
MgO, 64 atoms: E = -874279.2322 eV, max |F| = 0.2655 eV/A, net force 2.0e-14 eV/A, 1.2 s
```

- **EA and IP.** These equal the "as trained" row at 2.47 Å of this run's
  `scores/S6_au2_ea_ip.txt` (EA +2.014, IP +9.294 eV). The scorer's PBE0
  references at 2.47 Å are 2.03 and 9.22 eV (`probe_au2_ea_ip.py`).
- **Energies are absolute,** on the FHI-aims scale of the 4G training data.
- **Timing.** The 1.2 s is one energy-and-forces call, timed once, with
  torch's default thread count. It is not a benchmark; for the recorded
  timings, see [Limitations](#limitations).
- **Charge states and fields.** The calculator also passes fragments, spins
  and external fields; see the user guide, section 7.1.

### 2. Train a small model from a config

This is a pipeline smoke test with **synthetic labels** (random numbers). It
checks that training with a TRACE-Redox block runs on your machine; the model
it writes means nothing physically.

```python
# Pipeline smoke test: a toy data set with SYNTHETIC labels (random numbers), three optimiser steps.
# It checks that the training path runs on your machine; the model it writes means nothing physically.
import os

import numpy as np
import yaml
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

os.makedirs("quickstart", exist_ok=True)
rng = np.random.default_rng(0)


def labelled(atoms, energy):
    atoms.calc = SinglePointCalculator(atoms, energy=energy + rng.normal(0.0, 0.05),
                                       forces=rng.normal(0.0, 0.1, size=(len(atoms), 3)))
    return atoms


def mgo(displacement):
    cell = bulk("MgO", "rocksalt", a=4.2557)
    cell.positions[1, 0] += displacement
    return labelled(cell, -10.0)


def au2(distance, charge):
    dimer = labelled(Atoms("Au2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, distance]]), -4.0)
    dimer.info["charge"] = charge                       # net charge in e, read by flashace.charge_state
    return dimer


write("quickstart/train.extxyz", [mgo(0.0), mgo(0.05), au2(2.47, 0), au2(2.60, -1)])
write("quickstart/valid.extxyz", [mgo(0.02), au2(2.50, 0)])
config = {
    "device": "cpu", "seed": 0, "architecture_version": 2,
    "train_file": "quickstart/train.extxyz", "valid_file": "quickstart/valid.extxyz",
    "r_max": 4.0, "l_max": 2, "num_radial": 4, "hidden_dim": 8, "num_layers": 1,
    "correlation_order": 2, "correlation_channels": 4, "attention_num_heads": 2,
    "batch_size": 4, "epochs": 3, "learning_rate": 1.0e-3, "energy_weight": 1.0, "forces_weight": 10.0,
    "num_workers": 0, "model_save_path": "quickstart/model.pt",
    "save_last_checkpoint": False, "plot_training_curves": False,
    "electrostatics": {
        "enabled": True,
        "heads": {"r_max": 3.5, "hidden_dim": 16, "key_dim": 8, "bond_radial_basis": 4},
        "redox": {"anchor": "formal", "max_charge": 1, "fragment_ensemble": "diabatic", "couplings": True},
    },
}
with open("quickstart/config.yaml", "w") as handle:
    yaml.safe_dump(config, handle)
```

```bash
python train.py --config quickstart/config.yaml
```

The last lines of the output (about 5 s):

```
    3 |     4.7001 |    2039.67 |    718.14 |     0.232340 |     0.053982 |     0.147321 |     0.0000 ||     3.7275 |    1728.01 |     433.37 |         0.272302 |         0.074148 |     0.000000
Saved checkpoint to quickstart/model.pt
New best validation loss at epoch 3: 3.727485
Training finished. Best checkpoint: quickstart/model.pt (epoch 3).
```

`quickstart/model.pt` loads in `TransformersACECalculator` like a released
checkpoint. Each released training config is in
[`experiments/benchmarks_4g/configs/`](experiments/benchmarks_4g/configs/),
with its exact command in the file's header. Those configs need the prepared
4G data and, for stage B, a warm start; see
[`docs/electrostatics/REPRODUCE.md`](docs/electrostatics/REPRODUCE.md).

### 3. Score a model on Table II

You need the 4G data ([REPRODUCE.md](docs/electrostatics/REPRODUCE.md), steps
2 and 3) and the checkpoint at `runs/<run>/model.pt`.

```bash
cd experiments/benchmarks_4g
mkdir -p runs/AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43
cp /path/to/AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43.pt runs/AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43/model.pt
python eval_aumgo.py runs/AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43             # Table II only
bash score_model.sh runs/AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43             # every benchmark number of the run
```

`eval_aumgo.py` reads the test split `data/AuMgO_test.xyz` and the raw 4G set
under `$TRACE_4G_RAW` (default `experiments/benchmarks_4g/raw/datasets`). The
raw set provides the mean substrates of Table II. The script relaxes the Au
positions on each fixed mean substrate. The published run's
`scores/T5_wetting.txt` reads:

```
=== AuMgO / AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43
  doped    n=200  E RMSE 0.671 meV/atom  F RMSE 0.1112 eV/A
  undoped  n=200  E RMSE 1.142 meV/atom  F RMSE 0.0616 eV/A
  E(wetting) - E(non-wetting): doped -27.7 meV, undoped +729.7 meV   (DFT -2.7 / +928.9; 2G 375.0 / 375.0; 4G -41.0 / +975.4)
  relaxed non-wetting Au-O: doped 2.430 A, undoped 2.165 A   (DFT 2.332 / 2.190; 4G 2.342 / 2.177; 2G 2.256 / 2.256)
```

### 4. Score MgO phonons against PBE

The reference bands ship in `experiments/benchmarks_phonons/reference/`, so
this needs only the checkpoint and phonopy.

```bash
python experiments/benchmarks_phonons/score_phonons.py \
    experiments/benchmarks_4g/runs/AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43/model.pt --p5-supercell 4
```

The output, run from this release tree on 2026-11-18 (`--out` pointed at a
scratch directory; without it the scorer writes to
`experiments/benchmarks_phonons/runs/<run>/` and `<run>_sc4/`). It took 82 s
wall and 4.8 GB peak resident memory (the published runs' records: 61–62 s and
6.5–6.6 GB, `scores/P_phonons_pbe_sc4_time.txt`):

```
=== phonon benchmark, AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43, reference pbe, a = 4.25566 A, force-constant supercell 2^3; P5 on 4^3 (the reference's own: 4^3)
  PASS  P0 ensemble in the frame on every displaced supercell of 2^3 and 4^3 (population >= 1 - 1e-4, |F - E_frame| <= 1 meV): population 1.000000, |F - E_frame| 0.0002 eV
  PASS  P1 Z*(Mg) within 1.90-2.10 (pbe 1.9923): 1.961
  PASS  P2 eps_inf within 2.9-3.5 (pbe 3.235): 3.300
  PASS  P3 TO within 3 % of pbe 373.3: 370.3 (-0.8 %)
  PASS  P4 LO within 660-700 (pbe 681.3): 667.4
  PASS  P5 path RMS without NAC <= 15 cm^-1, force constants on 4^3 (C1 <= 1 cm^-1): 8.6 (C1 0.001)
  PASS  P6 path RMS with NAC <= 20 cm^-1: 6.8
  7 / 7 passed; log and band yaml in <out> and <out>_sc4
  recorded, not criteria: P5 on 2^3 29.9; P6 on 4^3 8.8
```

Every number in it equals the published run's `scores/P_phonons_pbe.txt`
(2×2×2) and `scores/P_phonons_pbe_sc4.txt` (4×4×4).

**Which protocol applies.** `--p5-supercell 4` is the protocol pre-registered
on 2026-11-16 for every step after 34b. Step 34b's registered criterion S4 is
read without the flag, on 2×2×2. There it passes with 6 / 7, failing P5 with
29.9 cm⁻¹ (`experiments/benchmarks_phonons/README.md`, "Pre-registered: P5 on
PBE's own 512-atom supercell").

---

## Benchmark summary

4G-HDNNP Au₂/MgO(001), test set and Table II protocol (Ko et al. 2021), and
MgO phonons against Togo's PBE set. Each model is the best checkpoint by
validation loss. DFT is PBE: FHI-aims for the slabs, VASP for the phonons.

| quantity | DFT | 4G-HDNNP (paper) | step 34b s42 | step 34b s43 | step 31 s42 | step 31 s43 |
|---|---|---|---|---|---|---|
| wet_u, undoped E(wetting) − E(non-wetting) (meV) | +928.9 | +975.4 | +610.4 | +729.7 | +851.0 | +920.4 |
| wet_d, doped (meV) | −2.7 | −41.0 | +92.3 | −27.7 | −177.4 | −75.7 |
| T5 = wet_d − wet_u (meV; % of DFT) | −931.6 | −1016.4 (109.1 %) | −518.1 (55.6 %) | −757.4 (81.3 %) | −1028.4 (110.4 %) | −996.1 (106.9 %) |
| test force RMSE, undoped / doped (eV/Å) | | | 0.0845 / 0.1242 | 0.0616 / 0.1112 | 0.0614 / 0.0867 | 0.0776 / 0.1045 |
| test energy RMSE, undoped / doped (meV/atom) | | | 1.353 / 0.989 | 1.142 / 0.671 | 0.590 / 0.937 | 0.469 / 0.621 |
| Au–Au bond force on test 167 at 2.089 Å (eV/Å) | +15.25 | | +14.25 | +15.82 | not measured | +6.18 ᵃ |
| gas-phase Au₂ EA / IP error, 2.30–2.80 Å (eV) | | | +0.20 … +0.43 / +0.56 … +0.59 | −0.03 … 0.00 / +0.07 … +0.08 | not measured | not measured |
| q(Au₂) shift doped − undoped (e) | −0.266 (Hirshfeld) | | −1.000 | −0.997 | −1.000 | −0.999 |
| MgO Z*(Mg) / ε∞ | 1.9923 / 3.235 | | 1.950 / 3.334 | 1.961 / 3.300 | 2.402 / 3.025 | 2.534 / 3.002 |
| MgO TO / LO at Γ (cm⁻¹) | 373.3 / 681.3 | | 382.1 / 669.3 | 370.3 / 667.4 | 384.3 / 807.9 | 398.7 / 851.5 |
| MgO path RMS without NAC, 4×4×4 (cm⁻¹; criterion ≤ 15) | | | 8.1 | 8.6 | 22.0 | 47.8 |
| MgO phonon criteria passed, P5 on 4×4×4 | | | 7 / 7 | 7 / 7 | 3 / 7 | 2 / 7 |
| pre-registered verdict | | | FAIL (S2, S5, S8) | FAIL (S2, S5, S8) | FAIL (Y1, Y3) | pass (Y1–Y5) |

ᵃ From a different probe (`probe_auau_wall_terms.py`), not the S7 protocol.
Between 2.10 and 2.06 Å this model's compressed dimer turns ionic (Au⁺Au⁻); the
bond force is −5.56 eV/Å at 2.07 Å and −13.79 to −22.63 eV/Å from 2.06 to
2.00 Å (`experiments/benchmarks_4g/README.md`, "Scored: the Au–Au wall").

Sources:
- the published runs' `scores/T5_wetting.txt`, `T1_T3_transfer.txt`,
  `S7_S8.txt`, `S6_au2_ea_ip.txt`, `P_phonons_pbe.txt` and
  `P_phonons_pbe_sc4.txt`;
- `experiments/benchmarks_4g/README.md`, "Step 34b scored" and "Step 31
  scored";
- `experiments/benchmarks_phonons/README.md`, "Step 31's phonons, scored for
  the first time" (outputs `experiments/benchmarks_phonons/runs/step31_<run>{,_sc4}/`).
  Step 31 was scored only with `--p5-supercell 4`.

Step 31 passed its own pre-registration on seed 43 (criteria Y1–Y5). That
pre-registration did not include the Au–Au wall or the phonons.

Step 33 (`AuMgO_single_all_e1kw_bondlaw_r5_s{42,43}`) is also released. Its
Table II is +768.4 / −9.2 meV (s42) and +789.9 / +92.7 meV (s43); its undoped
test forces are 0.1930 / 0.2211 eV/Å (`scores/T5_wetting.txt`). The cluster
benchmarks (Ag₃, NaCl, carbon chain) and every other comparison with
4G-HDNNP and PFN are in
[`docs/electrostatics/BENCHMARKS.md`](docs/electrostatics/BENCHMARKS.md).

---

## Data, licences and attribution

No third-party raw data are redistributed. The repository ships download
instructions, derivation scripts and sha256 checks
([REPRODUCE.md](docs/electrostatics/REPRODUCE.md); the data README
[`experiments/benchmarks_phonons/data/README.md`](experiments/benchmarks_phonons/data/README.md)).

| source | used for | licence |
|---|---|---|
| 4G-HDNNP data sets: T. W. Ko, J. A. Finkler, S. Goedecker, J. Behler, Materials Cloud Archive 2020.137, https://archive.materialscloud.org/record/2020.137, DOI 10.24435/materialscloud:f3-yh (FHI-aims, PBE, "light" settings) | training and test data of every 4G benchmark (Au₂/MgO, Ag₃, NaCl, carbon chain) | CC BY 4.0 |
| Ko et al., Nat. Commun. 12, 398 (2021), DOI 10.1038/s41467-020-20427-2 (arXiv:2009.06484) | the published 4G-HDNNP numbers (Tables I and II, text) and the NaCl curves of Figs. 7 and 8 | article CC BY 4.0 |
| A. Togo, Phono3py input data of rocksalt MgO (PBE), NIMS Materials Data Repository, https://mdr.nims.go.jp/datasets/1fd84240-abc0-4d8f-b4d7-1db304d6166b | the PBE phonon reference and the bulk MgO training cells of steps 33 and 34b | CC BY 4.0 |
| A. Togo, Ab-initio phonon calculation for MgO / Fm-3m (225) / materials id 1265 (phonondb), NIMS MDR, https://mdr.nims.go.jp/concern/datasets/bk128j14x | the PBEsol phonon reference | CC BY 4.0 |
| A. Togo, First-principles lattice thermal conductivity calculation for MgO / Fm-3m (225) / materials id 1265, NIMS MDR, https://mdr.nims.go.jp/datasets/59c5d436-339d-4b32-ba59-a3da5ea68bc7 | the PBEsol bulk-only phonon runs (lab notebook) | CC BY 4.0 |
| PFN `charge_transfer_nnp` (Preferred Networks), https://github.com/pfnet-research/charge_transfer_nnp | the split sizes of the PFN-size protocol (8000 / 1000; Au₂/MgO and NaCl 4000 / 500) | MIT (code) |
| PFN blog post of 2022-04-08, https://preferred.jp/en/blog/tech/neural-network-potential-with-charge-transfer/ | PFN's force RMSEs, digitised from its Fig. 2 into `compare_published.py` (factual values only) | no open licence stated |

Two small derived references ship because the scorer and the tests read them:
- **`experiments/benchmarks_phonons/reference/*.yaml`.** The four DFT band
  files, computed from Togo's NIMS MDR data (CC BY 4.0, the first two NIMS
  records). Changes: the harmonic force constants were rebuilt with phonopy
  2.17.2 from the records' displacement data (PBE) or loaded from the record
  (PBEsol). The bands were evaluated along Γ–X–K–Γ–L, with and without the
  non-analytic term, using each record's own Born charges and ε∞.
  `experiments/benchmarks_phonons/make_reference_bands.sh` rebuilds them and
  checks them byte for byte.
- **`experiments/benchmarks_4g/published/ko2021_nacl_fig7_fig8.json`.** The
  NaCl curves of Figs. 7 and 8 of Ko et al. (2021), extracted from the
  figures' vector paths in the arXiv version (`compare_published.py`
  describes the extraction). The published article is CC BY 4.0.

The project's own first-principles tables in `trace_redox/data/` ship with the
package. Their methods and builders are recorded in each file and in
`tools/electrostatics/`.

Literature constants used as response priors or comparisons:
- MgO ε∞ = 2.95: Sun et al., Phys. Rev. B 77, 134303 (2008), Table I, 295 K.
- MgO TO 401 and LO 718 cm⁻¹: Jasperse et al., Phys. Rev. 146, 526 (1966);
  Sangster et al., J. Phys. C 3, 1026 (1970). Both are cited in
  `tools/electrostatics/mgo_phonons.py`.
- The atomic polarisabilities of Au and Ag: arXiv:2410.01493, cited in
  `make_au_alpha_prior.py` and `make_ag_alpha_prior.py`.
- The atomic ionisation energies and electron affinities: generated from
  pymatgen's element data, as recorded in `trace_electrostatics/constants.py`
  and `trace_redox/constants.py`.

---

## How to cite

- **TRACE itself:** the TRACE paper, arXiv:2607.25652. Its BibTeX is in
  [`README.md`](README.md#citation).
- **The 4G data and benchmark:** T. W. Ko, J. A. Finkler, S. Goedecker and J.
  Behler, Nat. Commun. 12, 398 (2021), together with the data set, Materials
  Cloud Archive 2020.137, DOI 10.24435/materialscloud:f3-yh.
- **The MgO phonon references:** the three NIMS MDR records above (creator A.
  Togo), and the phono3py set's papers, Phys. Rev. B 91, 094306 (2015) and
  Phys. Rev. Lett. 115, 205901 (2015). For the bands' software, cite phonopy
  as its documentation asks.
- **The PFN comparison:** the PFN repository and blog post above.

---

## Limitations

- **Research release.** No model passes all of its pre-registered criteria on
  both seeds ([Status](#status)). Treat every checkpoint as a record of a
  measured experiment, not as a production potential.
- **ASE only.** The LAMMPS TorchScript and AOT exports refuse a model with
  electrostatics, rather than drop its energy, forces and virial
  (`transformers_ace.deploy.refuse_electrostatics`). Use
  `TransformersACECalculator`.
- **float64 only.** On the GPU used for steps 33 and 34b, float64 runs at
  about 1/64 of float32.
- **Cost per evaluation, from the records.** The cell is step 31 seed 43,
  frozen, on the undoped wetting reference cell (110 atoms), on the M2 Pro CPU.
  Source: `experiments/benchmarks_4g/README.md`, "S1 scored" and "S2 scored".
  - **Time.** Energy and forces take 1.324 s on 4 threads and 1.836 s on 1
    thread, with the settings as stored.
  - **Where the time goes.** The electrostatics take 76–80 % of it. The dense
    bound-sector operator has n_m = 3974 rows (n_bonds + 3 n_atoms).
  - **Supercells.** The 440-atom (2, 2, 1) Born–von Kármán supercell takes
    13.265 s and 5.40 GB.
  - **Against DFTB.** GFN1-xTB (DFTB+ 25.1) on the same 110-atom cell, SCC from
    scratch, takes 11.676 s on 4 threads and 14.315 s on 1 thread.
  - **The reference DFT's 3 × 3 × 1 k-point sampling.** TRACE pays 8.7× its Γ
    time for (2, 2, 1). In that record (3, 3, 1) was refused; it has run since
    the route-aware operator guard of 2026-11-12.
  - **Training.** One epoch of step 34b (436 training structures) took about
    20 min on the GPU host. One epoch of step 31 (300 structures) took 29 min
    on the Mac ([REPRODUCE.md](docs/electrostatics/REPRODUCE.md)).
- **Integer charges.** The formal-charge models put −1.00 e of free charge on
  Au₂ on the doped slab; DFT's Hirshfeld charge shift is −0.27 e
  (`T1_T3_transfer.txt`). Integer charge localisation is a property of the
  model's charge states, and its comparison with a Hirshfeld partition is not
  like for like.
- **Extensivity.** Step 31 uses the legacy single-Hamiltonian forms, which are
  not extensive when a periodic cell is replicated (`trace_redox/SPEC.md` §0,
  "The equation as of 2026-11-11"). Steps 33 and 34b trained with §28.6, §36
  and §38, but at Γ on one cell. No model has been trained with the
  Born–von Kármán sampling (§37). Table II on the 2×2 supercell has not been
  scored for steps 33 and 34b.
- **Configuration caps.** The cluster closure of the ensemble refuses more than
  512 configurations, and so does the atom-resolved formal stack within its
  energy window. A structure outside the cap raises instead of returning an
  energy. Step 34's first launch died this way on a cold-start state
  (lab notebook, "Step 34 stage A stopped").
- **Element coverage of the two-site sector.** §40.7's first-principles
  entries are applied to Au–Au only. Ag–Ag and H–H are tabulated and withheld;
  `trace_redox/data/two_site.json` says why.
- **Run-time tables.** The checkpoint does not store the `trace_redox/data/`
  tables. A model evaluated with different tables is a different model.
  `two_site.json` records the sha256 of the files it was solved with.
- **Retraining is not bitwise reproducible.** The causes are 1-ulp differences
  in the solver, GPU reduction order, and transient NaN recomputations by the
  training guard. The released checkpoints are the reproduction path for the
  reported numbers.
- **Rebuilding the first-principles tables.** This needs PySCF caches that are
  not in the repository; some Au–Au points took hours each.

---

## Tests

The last recorded run was on 2026-11-16, on the CPU environment of record,
in the working tree this release was cut from. `pytest tests` in one session
gave **2184 passed, 4 skipped, 2 xfailed** in 17 minutes. That tree had the
derived benchmark data present. **The suite is to be re-run on this release
tree.** On a clean clone, the tests that read gitignored benchmark data
(`experiments/benchmarks_4g/data*/`) or gitignored checkpoints skip, so expect
more skips until `prepare_all.sh` has run.

```bash
python -m pytest tests/electrostatics tests/test_train_electrostatics.py -q -p no:cacheprovider
python -m pytest tests/redox -q -p no:cacheprovider                 # TRACE-Redox
python -m pytest tests --ignore=tests/electrostatics --ignore=tests/test_train_electrostatics.py \
    --ignore=tests/redox -q -p no:cacheprovider
```

Test modules in `tests/redox` carry a `test_redox_` prefix and share
`redox_helpers.py` (there is no `conftest.py` there), so that one combined run
has no duplicate basenames. The tests assert physics with tolerances derived
from machine epsilon and the operation, not one machine's rounding.

---

## Documentation map

| document | content |
|---|---|
| [`docs/electrostatics/METHODS.md`](docs/electrostatics/METHODS.md) | the methods: equations, training, protocols |
| [`docs/electrostatics/BENCHMARKS.md`](docs/electrostatics/BENCHMARKS.md) | every benchmark table, with its source file |
| [`docs/electrostatics/REPRODUCE.md`](docs/electrostatics/REPRODUCE.md) | from a clean clone to the published numbers |
| [`docs/electrostatics/MODELS.md`](docs/electrostatics/MODELS.md) | the released checkpoints (GitHub release assets) and their sha256 |
| [`docs/electrostatics.md`](docs/electrostatics.md), [`docs/electrostatics_api.md`](docs/electrostatics_api.md) | user guide and API reference of `trace_electrostatics` |
| [`trace_electrostatics/SPEC.md`](trace_electrostatics/SPEC.md), [`trace_redox/SPEC.md`](trace_redox/SPEC.md) | the physics contracts |
| [`trace_electrostatics/CHANGELOG.md`](trace_electrostatics/CHANGELOG.md) | the change record |
| [`experiments/benchmarks_4g/README.md`](experiments/benchmarks_4g/README.md), [`experiments/benchmarks_phonons/README.md`](experiments/benchmarks_phonons/README.md) | the lab notebooks: every pre-registration, run and score, failures included, in date order |

The code is under the MIT licence ([`LICENSE`](LICENSE)).
