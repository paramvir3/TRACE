# Released checkpoints

The trained models behind the reported TRACE-Electrostatics numbers are attached to the GitHub release of this
branch as release assets, not stored in git. Each asset is the run's `model.pt` (the best checkpoint by validation
loss), unmodified, renamed `<run_name>.pt`. The release also carries `SHA256SUMS`, which lists every asset. The
cluster runs of the PFN-size protocol live in `runs/protocol/<name>/` and are named `protocol_<name>.pt`, so they
cannot be confused with the first-round runs of the same name (which are not released).

Every number reported for these runs in [BENCHMARKS.md](BENCHMARKS.md) was measured on these files, not on a
retraining. Retraining is not bitwise reproducible (see [REPRODUCE.md](REPRODUCE.md)), so the checkpoints are the
reproduction path for the numbers.

## The checkpoints

`P` stands for `AuMgO_formal_kT_eps_cpl_alpha_exband_amp_part_pen_ind_spol_pfp_bopw`; the asset name is `P` with
the suffix shown. "Best epoch" is the epoch stored in the checkpoint, out of the run's total. "Trained on" is the
machine of the published run: the Mac CPU (torch 2.2.0) or the GPU host (one RTX 5060 Ti 16 GB, torch 2.11.0+cu128).
Each run's sanitised training config is `experiments/benchmarks_4g/configs/<run_name>.yaml` (cluster runs:
`configs/protocol/<name>.yaml`), whose header gives the warm-start parent and the training command. Sizes are in bytes.

### Au₂/MgO(001): the 4G-HDNNP benchmark and the MgO phonon benchmark

| asset (`<run_name>.pt`) | step | stage | seed | best epoch | trained on | what it is | sha256 | size |
|---|---|---|---|---|---|---|---|---|
| `P_bondlaw_r5_s42.pt` | 26 | cold start | 42 | 40 / 40 | Mac CPU | the first cold start of the chain; warm-start parent of step 31 | `daf31bfe85250278812a04c889d80c42cbd893d38563ae8cb8738b42514725e4` | 4322738 |
| `P_bondlaw_r5_s43.pt` | 26 | cold start | 43 | 39 / 40 | Mac CPU | as above, seed 43 | `005f01704613c5087136ae7fdabf1c461b726b2e48eea0fe81eca1074884f609` | 4322738 |
| `P_e1kw_bondlaw_r5_s42.pt` | 31 | warm start from 26 | 42 | 20 / 20 | Mac CPU | energy weight 1000; the step-31 Table II numbers; warm-start parent of step 33 | `653c648f86019443fbdf385d440716829750b33341da5565a93165603a580d1f` | 4322802 |
| `P_e1kw_bondlaw_r5_s43.pt` | 31 | warm start from 26 | 43 | 17 / 20 | Mac CPU | as above, seed 43 | `3699c1f94b922247da24699a7a92b6e4b917d307d31ef48dc2d829f9b73e799e` | 4322802 |
| `AuMgO_single_all_bondlaw_r5_s42.pt` | 33 | A | 42 | 40 / 40 | GPU host | the single model with every term on, on the slabs plus the PBE MgO cells; stage A | `e790ed4f0589d6a20aca19b65418e39e1a89f132240e3e8d2c4b9a7feaa1fb75` | 4327663 |
| `AuMgO_single_all_bondlaw_r5_s43.pt` | 33 | A | 43 | 39 / 40 | GPU host | as above, seed 43 | `16084f1bbfef90c95fec0f1565e5f58ccbd3716da93802d6ac65ea5f3207e641` | 4327663 |
| `AuMgO_single_all_e1kw_bondlaw_r5_s42.pt` | 33 | B | 42 | 15 / 20 | GPU host | step 33's scored model (energy weight 1000); scored FAIL on S2, S3, S5, pass on S1, S4 | `651ac161f2defaca8512629507b333c4d3ede77be39cb500473a61d2e4ffea4b` | 4327663 |
| `AuMgO_single_all_e1kw_bondlaw_r5_s43.pt` | 33 | B | 43 | 20 / 20 | GPU host | as above, seed 43 | `42d59b7607e12fc3dd647525cb110117b43e7bdbee2d6b7cc6a543aaada92892` | 4327663 |
| `AuMgO_twosite_bf055_bondlaw_r5_s42.pt` | 34b | A | 42 | 39 / 40 | GPU host | step 33's terms plus SPEC 39, 29 and 40 / 40.7, band starting factor 0.55; cold start, stage A | `681c36d227b6666e8e62bba177286f6fbfa06f4d005d73a3a7c71930132319b5` | 4327727 |
| `AuMgO_twosite_bf055_bondlaw_r5_s43.pt` | 34b | A | 43 | 39 / 40 | GPU host | as above, seed 43 | `29c908ed337a0ba7dfd622560898972f538a452aa30d78e5e73d4d42cdaf7b0b` | 4327727 |
| `AuMgO_twosite_bf055_e1kw_bondlaw_r5_s42.pt` | 34b | B | 42 | 15 / 20 | GPU host | step 34b's scored model (energy weight 1000); scored FAIL as pre-registered (S2, S5, S8), pass on S1, S3, S4, S7 | `332b2654d96e512433e43b750b09a19a927656f51a580847176b2e70fed7b5cf` | 4327727 |
| `AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43.pt` | 34b | B | 43 | 18 / 20 | GPU host | as above, seed 43; the checkpoint of the quick start in TRACE_ELECTROSTATICS.md | `740ea30cca04950d153ebddc4301b9926ea2e97e89a8d6b09fa8f4e24a79c297` | 4327727 |

### 4G-HDNNP clusters at the PFN split sizes (`runs/protocol/`)

All seed 42, 300 epochs, cold starts, on `data_pfn/`. `local`: no electrostatics; `elec`: TRACE electrostatics;
`elecq`: `elec` plus the Hirshfeld partial-charge loss (as 4G-HDNNP and PFN train on); `step9`: the step-9 redox
stack (learned couplings, formal anchor, kT anneal) with a response prior (NaCl ε∞, Ag α); `step9cu`: the same with
`carriers: covalent_units`.

| asset | system | arm | best epoch | trained on | sha256 | size |
|---|---|---|---|---|---|---|
| `protocol_Ag_cluster_local_s42.pt` | Ag₃⁺/Ag₃⁻ | local | 51 / 300 | Mac CPU | `1fe5c00a0c33b2e31fe5947292083c1e9031e8eb79e87e765c3897399e285d4b` | 2996454 |
| `protocol_Ag_cluster_elec_s42.pt` | Ag₃⁺/Ag₃⁻ | elec | 296 / 300 | Mac CPU | `b435874a90fbda8558dd768207de9c6bb0c621451fac6a9f93b10c57874b46b3` | 3471422 |
| `protocol_Ag_cluster_elecq_s42.pt` | Ag₃⁺/Ag₃⁻ | elecq | 296 / 300 | Mac CPU | `1967b0524cfd5e9828600e0d8e7da80ebe8606eea467680796af2abd21054294` | 3471422 |
| `protocol_Ag_cluster_step9_s42.pt` | Ag₃⁺/Ag₃⁻ | step9 | 295 / 300 | Mac CPU | `3a710dc7ca8a5892dfe97dfe226952aa7aeb65b8e51b6afb810f76e6790987c0` | 4322358 |
| `protocol_NaCl_local_s42.pt` | Na₉Cl₈⁺ / Na₈Cl₈⁺ | local | 300 / 300 | Mac CPU | `a60fce8cc957d12146e10019db90e1c9d9f04add8167b67d359f7ce660aa9913` | 2996454 |
| `protocol_NaCl_elec_s42.pt` | Na₉Cl₈⁺ / Na₈Cl₈⁺ | elec | 194 / 300 | Mac CPU | `569080271d9aa77980d3c374b440caa9e34a084a8f452f2d077fb336ed5c0fbf` | 3471358 |
| `protocol_NaCl_elecq_s42.pt` | Na₉Cl₈⁺ / Na₈Cl₈⁺ | elecq | 300 / 300 | Mac CPU | `55118f9cf2e648a67ec72d48aa2ccaefa9b1c55ba1a31e7aabfc05ab6dc8659d` | 3471358 |
| `protocol_NaCl_step9_s42.pt` | Na₉Cl₈⁺ / Na₈Cl₈⁺ | step9 | 300 / 300 | Mac CPU | `d804abc00fd160f24b27e3669a43b3ccc09a831a6831eeb2546fd694afd4f234` | 4324582 |
| `protocol_Carbon_chain_local_s42.pt` | C₁₀H₂ / C₁₀H₃⁺ | local | 299 / 300 | GPU host | `0db92a5fd77e2ebdfbf130431877333edd1ed6bdd11fcbbabf53ca82126ef626` | 2998703 |
| `protocol_Carbon_chain_elec_s42.pt` | C₁₀H₂ / C₁₀H₃⁺ | elec | 297 / 300 | GPU host | `bc7c361592d387152e9567312ce9232c5ddc447adf01087132c928e64230ad81` | 3474439 |
| `protocol_Carbon_chain_step9cu_s42.pt` | C₁₀H₂ / C₁₀H₃⁺ | step9cu | 299 / 300 | Mac CPU | `927bfa329233ff9ea99ba4a00811447bf3ca766a2b9be9d35340e20dca6632de` | 4322550 |

The step-9 runs of Ag₃ and of the chain were interrupted and resumed inside the run (Ag₃ from epoch 148, the chain
from epoch 272); the asset is the final best checkpoint of the resumed run.

### Not attached

- **`protocol_Carbon_chain_elecq_s42.pt`** (chain, `elecq`, best epoch 300 of 300 according to the run's log). The
  local copy was not available when the assets were packaged, so it is not attached and its sha256 is not recorded.
  Its config ships (`configs/protocol/Carbon_chain_elecq_s42.yaml`).
- **The warm-start files** (`warm_start.pt` of steps 31, 33 and 34b). They are not needed: each is rebuilt from a
  released parent by `experiments/benchmarks_4g/make_warm_start.py`, with the command in the run's config header. With
  torch 2.2.0 the step-31 and step-33-stage-A warm starts come out byte for byte identical to the files the published
  runs trained from (sha256 `11bbcc91…5898c2` and `76813360…8ce1a7` for step 31, `253e63df…7b255c` and
  `0f2976ad…348d0b` for step 33 stage A; the full values are in `make_warm_start.py`). The stage-B warm starts of steps
  33 and 34b were written on the GPU host and their sha256 was not recorded; the script's output was checked to hold
  the same content as the queue's recipe (every tensor equal, same stored config, epoch 0).

## Download and verify

From the release page (Assets), or with the GitHub CLI:

```bash
mkdir -p trace-models && cd trace-models
gh release download <release-tag> --repo <owner>/<repo> --pattern 'AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43.pt' --pattern SHA256SUMS
shasum -a 256 -c SHA256SUMS --ignore-missing      # Linux: sha256sum -c SHA256SUMS --ignore-missing
```

Every downloaded asset must print `OK`. To score a checkpoint or warm-start from it, place it where its config
expects it:

```bash
cd experiments/benchmarks_4g
RUN=AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43
mkdir -p runs/$RUN && cp /path/to/trace-models/$RUN.pt runs/$RUN/model.pt
./score_model.sh runs/$RUN                     # writes runs/$RUN/scores/*
# a cluster run: mkdir -p runs/protocol/NaCl_elec_s42 && cp .../protocol_NaCl_elec_s42.pt runs/protocol/NaCl_elec_s42/model.pt
```

`score_model.sh` needs the derived data first (`experiments/benchmarks_4g/prepare_all.sh`, which downloads nothing:
it derives the splits from the raw 4G and NIMS MDR files and checks them against `DATA_SHA256.txt`).

## Load

The checkpoint loads in the ASE calculator, as `experiments/benchmarks_4g/eval_aumgo.py` does:

```python
import torch
from ase.io import read
from flashace.calculator import TransformersACECalculator   # also exported as transformers_ace.TransformersACECalculator

torch.set_num_threads(4)
calc = TransformersACECalculator(model_path="AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43.pt", device="cpu")
atoms = read("experiments/benchmarks_4g/data/AuMgO_test.xyz", index=0)   # info["charge"] is the total charge (e)
atoms.calc = calc
print(atoms.get_potential_energy(), atoms.get_forces()[:2])
```

The calculator switches torch to float64 for checkpoints that carry `trace_electrostatics` (the package refuses
float32). Each file is a training checkpoint: a dict with `config` (the model and electrostatics configuration the run
resolved, including its `RedoxConfig`), `epoch`, `model_state_dict`, and the optimiser, scheduler and scaler states.

What the checkpoint does not carry:

- **The `trace_redox/data` tables.** Several terms (among them SPEC 23, 25, 39 and 40 / 40.7) read first-principles
  tables at run time. The step-34b numbers were computed with the tables of this release; rescoring
  `AuMgO_twosite_bf055_e1kw_bondlaw_r5_s43.pt` with them and `score_model.sh` on 2026-11-18 reproduced its recorded
  Table II values (−27.7 / +729.7 meV), test force RMSE (0.1112 / 0.0616 eV/Å) and phonon scores (P0–P6 on 64 and
  512 atoms) to the printed digits. The earlier runs were not rescored against these files.

  | file | sha256 |
  |---|---|
  | `trace_redox/data/frame_pair.npz` | `c6a96e6efb632b0c96c22b954e103cfd0435b4ba47feb485b583562a212eeba1` |
  | `trace_redox/data/frame_wall.npz` | `a1c27d631f05ddcbe89a107511b78a2a2d6b42dad0d7fa32b6736307a7b40a3a` |
  | `trace_redox/data/ion_densities.npz` | `6c95034f343fd5ab750490e97c660a198976f4d558e501a94837f2c9f0e6dc25` |
  | `trace_redox/data/ion_polarizabilities.json` | `8ab2f88115010f1de253ac96cdf61b209b4722199919ddf3f8ae0b7871de2bfa` |
  | `trace_redox/data/pauli_fp.npz` | `7fb1d586242481586d2fcec7b3428ea142e0766e5d284165303031aca12e8c5d` |
  | `trace_redox/data/two_site.json` | `5e8608f5aa7c681c655fa773cf8e83ed741a74b2b2522be9535c71d7361c84dd` |

  `two_site.json` also records the sha256 of the four tables its entries were solved with, and the loader refuses the
  entries when the tables on disk differ (`trace_redox/two_site.py`).
- **Options added after a run was trained.** Steps 26 and 31 predate SPEC 28.6 and SPEC 35.1. Their checkpoints store
  no `state_polarizability_stiffness` and no `frame_polarizability_scope`; the loader restores the forms they trained
  with (`'block'` and `'all'`), so evaluation is unaffected. The current default for a new run is `'onsite'` /
  `'ions'`, which is why the shipped configs of steps 26 and 31 state `state_polarizability_stiffness: block`
  explicitly (`frame_polarizability_scope` acts only with the frame-polarizability anchor, which these runs do not use).

Environment of record for every score: Python 3.8.16, torch 2.2.0, e3nn 0.4.4, ase 3.23.0, numpy 1.24.2, scipy 1.10.0,
phonopy 2.17.2 (Mac CPU). The GPU-trained checkpoints (torch 2.11.0+cu128, e3nn 0.6.0 on the GPU host) were scored in
that CPU environment.
