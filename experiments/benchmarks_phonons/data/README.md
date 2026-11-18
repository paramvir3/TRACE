# Data of the MgO phonon benchmark

Only this README is distributed from this directory (`../.gitignore` keeps the data files out of git). The data files
are rebuilt with the scripts named below and checked against the sha256 values at the end. All but two are derived
from third-party data, which this repository does not redistribute; the two dielectric-prior files are written by a
script from two numbers of the reference records (ε∞ and the lattice constant).

The only third-party-derived files of this benchmark that ship are the four DFT reference bands in
`../reference/`. They are small derived files under CC BY 4.0 (attribution below and in `../reference/README.md`),
and the scorer (`../score_phonons.py`) and `tests/test_score_phonons_protocol.py` read them.
`../make_reference_bands.sh` rebuilds them from the raw downloads and checks them byte for byte.

## Sources and licences

| record | creator | licence | file used | size | sha256 (download) | sha256 (unpacked `.yaml`) |
|---|---|---|---|---|---|---|
| Phono3py input data of rocksalt MgO (PBE), NIMS Materials Data Repository: https://mdr.nims.go.jp/datasets/1fd84240-abc0-4d8f-b4d7-1db304d6166b | A. Togo | CC BY 4.0 | `phono3py_params.yaml.xz` | 81144 B (unpacked 854761 B) | `13b12ff2f0c16d5db00ae14ee6d32b667d4ba0e3932ddd03522bb83f81d7adce` | `d22ebbc2aec853c192204aac1474f2e586e187169b02d641254424d1306ec1df` |
| Ab-initio phonon calculation for MgO / Fm-3m (225) / materials id 1265 (phonondb), NIMS Materials Data Repository: https://mdr.nims.go.jp/concern/datasets/bk128j14x | A. Togo | CC BY 4.0 | `phonopy_params.yaml.xz` | 1648 B (unpacked 21932 B) | `25b82cafd3efc18ba3f01c99e63e9f58510350978e90c8cb4100bc00637bddce` | `8bbb516cdd8dd701798897ce25d8477e7bf303d13dd64efa07d9f0ef0d92fceb` |
| First-principles lattice thermal conductivity calculation for MgO / Fm-3m (225) / materials id 1265, NIMS Materials Data Repository: https://mdr.nims.go.jp/datasets/59c5d436-339d-4b32-ba59-a3da5ea68bc7 | A. Togo | CC BY 4.0 | `phonopy_training_dataset.yaml.xz` | 456252 B (unpacked 2019435 B) | `49355412ced3ebff17be23dcf778cf964b1692b737faf6f2aee7ced22b5b39b0` | `9835a5f10ecdb8881874c8097ce335605c84779eb69fa69077702281cb29691b` |
| 4G-HDNNP data sets, Materials Cloud Archive 2020.137: https://archive.materialscloud.org/record/2020.137 (DOI 10.24435/materialscloud:f3-yh) | T. W. Ko, J. A. Finkler, S. Goedecker, J. Behler | CC BY 4.0 | `datasets.tar.gz` (the Au₂–MgO set) | see `experiments/benchmarks_4g` | `6194ee9e23b446f17c3e8b0b754fa9a9e9498b1b099231015ea10a6a313b9775` | — |

The PBE set belongs to the 103-compound phono3py set of Phys. Rev. B 91, 094306 (2015) and Phys. Rev. Lett. 115,
205901 (2015). The 4G-HDNNP data are those of T. W. Ko, J. A. Finkler, S. Goedecker and J. Behler, Nat. Commun. 12,
398 (2021).

**Where the downloads go.** Put the three NIMS files in `$TRACE_PHONON_RAW` (default
`experiments/benchmarks_phonons/raw`, which is not tracked) in this layout:

```
$TRACE_PHONON_RAW/phono3py_pbe_mgo/phono3py_params.yaml.xz
$TRACE_PHONON_RAW/phonondb_mp-1265/phonopy_params.yaml.xz
$TRACE_PHONON_RAW/mdr_ltc2026_mgo/phonopy_training_dataset.yaml.xz
```

The 4G data go in `$TRACE_4G_RAW` and are prepared by `experiments/benchmarks_4g/prepare_all.sh` (see that
directory). Downloaded files are untrusted input: read them with `python -I` and pass paths as arguments, as the
scripts below do.

## Which script makes which file

Commands from the repository root. `make_reference_bands.sh` checks and unpacks the first two downloads. Unpack the
third yourself (`xz -dk`, or Python's `lzma`) and check it against the table above.

| file | content | made by |
|---|---|---|
| `../reference/MgO_PBE_togo_band_{nac,nonac}.yaml` (ships) | PBE harmonic bands of Togo's phono3py set: fc2 from two 0.03 Å displacements on the 4×4×4 conventional supercell (512 atoms), a = 4.25566 Å, its own Z* = ±1.9923 and ε∞ = 3.235; Γ–X–K–Γ–L, 51 points per segment, with and without NAC | `experiments/benchmarks_phonons/make_reference_bands.sh`, which runs `experiments/benchmarks_4g/phonons_mgo/compare_phono3py_pbe.py` and renames its `band_dft_{nac,nonac}.yaml` |
| `../reference/MgO_PBEsol_phonondb_band_{nac,nonac}.yaml` (ships) | PBEsol bands of phonondb mp-1265: fc2 on the 2×2×2 conventional supercell (64 atoms), a = 4.21726 Å, Z* 1.979, ε∞ 3.242; same path | the same script, via `experiments/benchmarks_4g/phonons_mgo/compare_phonondb.py` |
| `MgO_PBE_togo_displaced.extxyz` | the 148 displaced PBE supercells of Togo's set with their VASP forces (2 fc2 cells of 512 atoms, 146 fc3 cells of 64 atoms); no energies exist in the set, so `has_energy=F` | `python -I experiments/benchmarks_phonons/extract_phono3py_cells.py $TRACE_PHONON_RAW/phono3py_pbe_mgo/phono3py_params.yaml experiments/benchmarks_phonons/data/MgO_PBE_togo_displaced.extxyz` |
| `MgO_PBEsol_ltc_{train,valid}.xyz` | 180 / 20 rocksalt MgO 2×2×2 supercells (64 atoms, a = 4.21726 Å), every atom displaced 0.03 Å, VASP PBEsol forces (energies kept in `info`, `has_energy=F`); split with numpy seed 0 | `python -I experiments/benchmarks_phonons/tools/extract_ltc_pbesol.py $TRACE_PHONON_RAW/mdr_ltc2026_mgo/phonopy_training_dataset.yaml experiments/benchmarks_phonons/data/MgO_PBEsol_ltc` |
| `AuMgO_bulkaug_{train,valid}.xyz` | the 4G Au₂–MgO slabs (300 train / 76 valid) plus the fc3 cells of `MgO_PBE_togo_displaced.extxyz` (136 train / 10 held out, numpy seed 0; the two fc2 cells, the reference's own force constants, are excluded) | `python experiments/benchmarks_phonons/make_bulk_augmented_data.py` (needs `experiments/benchmarks_4g/data/AuMgO_{train,valid}.xyz` from `experiments/benchmarks_4g/prepare_all.sh`) |
| `AuMgO_bulkaugw_{train,valid}.xyz` | the same, with `force_weight=122.4` on the 136 + 10 bulk cells (the ratio of the mean squared force components of the slabs and the bulk cells); training data of the reweighted bulk run and of steps 33 and 34b | `python experiments/benchmarks_phonons/make_bulk_augmented_data.py --force-weight 122.4` |
| `MgO_bulkonly_{train,valid}.xyz` | the 136 / 10 bulk cells alone | `python experiments/benchmarks_phonons/make_bulk_augmented_data.py --bulk-only` |
| `MgO_bulk_eps_prior_{pbe,pbesol}.xyz` | the 8-atom rocksalt cell with the reference's own DFPT ε∞ as a dielectric-tensor label: PBE 3.235 at a = 4.255656884758435 Å, PBEsol 3.2417 at a = 4.21725941 Å; no energies or forces. The PBE file is the dielectric prior of steps 33 and 34b | `python experiments/benchmarks_phonons/make_mgo_eps_prior_dft.py` (it refuses to overwrite an existing file) |

## sha256 of the derived files

Every file below was rebuilt from the downloads on 2026-11-16, and again on 2026-11-18 with the released scripts and
the commands above (`make_reference_bands.sh` starting from the two `.xz` files, then the two extractors, the three
modes of `make_bulk_augmented_data.py` and `make_mgo_eps_prior_dft.py`), and matched these values byte for byte both
times (Python 3.8.16, numpy 1.24.2, PyYAML 6.0, phonopy 2.17.2; ase 3.23.0b1 for the `python -I` runs and 3.23.0 for
the others). Another version of ase or phonopy may change the formatting of the files without changing their
numbers; compare the content, not only the hash, before concluding that a file differs.

```
725ae4bfba95efff1ef63ebff752707bc30a37f117cc78dde1b70dced6a71341  reference/MgO_PBE_togo_band_nac.yaml
35f65f2e93b8b30e68cf26abab0cd67bd566de3ebe85cdada9962fd5dd28efb7  reference/MgO_PBE_togo_band_nonac.yaml
2b1ef918a2651366a9f10f84f1b2732b2d0381883f0c07da4a532461b3c981e1  reference/MgO_PBEsol_phonondb_band_nac.yaml
12b34fc2306a9648df3ea9079f718cdb701b98a0eb787aae4b9fc64581899579  reference/MgO_PBEsol_phonondb_band_nonac.yaml
8b12f2c7f2d09c527cd209aa0a0f7e511bb84afdcc923ba6b77edf9edd45e7dd  data/MgO_PBE_togo_displaced.extxyz
834b812c601a2dec4328b312e9c325aba1be199792431c6f4429618644e9dc31  data/MgO_PBEsol_ltc_train.xyz
34cb9e69a00eeaf09ec5f2cc9f5a18c35cbcf9a3ef146016af881ad4018967ab  data/MgO_PBEsol_ltc_valid.xyz
b0dcb3a41704a045f849fdd504d458bb5c78a42f6121edb773c7405510cbb92a  data/AuMgO_bulkaug_train.xyz
53f3a8c6d9e695b0893aac2aab9c10bc370c39d6fc2213c6399297b10cb0f0fc  data/AuMgO_bulkaug_valid.xyz
5ed6237a9a0f4bf300a1c7081d7a1c2da61f6a7fe6b67d626d5310987d00d0a7  data/AuMgO_bulkaugw_train.xyz
e5b4a50a1b45e418978ec10f5c54b75edb55143cd63776e1c26591a80fc63779  data/AuMgO_bulkaugw_valid.xyz
fb2d08bed721cb0c889a854f707ff03f4379e8ea14f7ee695b0d09218c67fd79  data/MgO_bulkonly_train.xyz
a923e7515138ba82837c9f8f86dacd3bdca9884c072576b774cab9c7c3cf25b2  data/MgO_bulkonly_valid.xyz
abb3507dfff25d32d447db44532e2e3b63e0e899ead01c76634dbabb3c36dd83  data/MgO_bulk_eps_prior_pbe.xyz
7305cb8c72bef0cbf1942dd9c3ce78264589fd98e7e01a47c824ae72cfd3a690  data/MgO_bulk_eps_prior_pbesol.xyz
```

(Paths relative to `experiments/benchmarks_phonons/`.) The `AuMgO_bulkaug*` files also depend on
`experiments/benchmarks_4g/data/AuMgO_train.xyz` (sha256 `00dbbd41db6c7052573b4542caa3044d63a5c60e16c9a5d7d406c67cb220d9bb`)
and `AuMgO_valid.xyz` (`1f5770c075ed4816dcfca3191d1ab9b8b6bc7f4df6ad2f158aa6d76e1761a57b`).

## Attribution

- The reference bands in `../reference/` are computed from A. Togo's data in the NIMS Materials Data Repository
  (the first two records above), licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Changes: the
  harmonic force constants were built with phonopy 2.17.2 from each record's displacements and forces (PBE: the two
  fc2 displacements of the 512-atom supercell; PBEsol: the two displacements of the 64-atom supercell), symmetrised,
  and the band structure was evaluated along Γ–X–K–Γ–L with and without the non-analytic term, using each record's
  own Born charges and ε∞.
- The training files built here from the 4G-HDNNP data (Materials Cloud 2020.137, CC BY 4.0) and from Togo's records
  are not distributed.
- Experimental values quoted beside the references in `../README.md` (TO 401, LO 718 cm⁻¹, ε∞ 2.95, ε0 9.8) are
  literature values. TO and LO are cited in `tools/electrostatics/mgo_phonons.py` (Jasperse et al., Phys. Rev. 146,
  526 (1966); Sangster et al., J. Phys. C 3, 1026 (1970)); ε∞ 2.95 is Table I of Sun et al., Phys. Rev. B 77, 134303
  (2008) (arXiv:0704.2267), at 295 K, whose Table III also lists TO 401 as experiment. LO 718 and ε0 9.8 have not
  been checked against the source text here.
