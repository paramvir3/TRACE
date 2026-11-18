# DFT reference bands of the MgO phonon benchmark

Third-party-derived files, distributed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The scorer
(`../score_phonons.py`) and `tests/test_score_phonons_protocol.py` read them.

| file | computed from |
|---|---|
| `MgO_PBE_togo_band_{nac,nonac}.yaml` | A. Togo, "Phono3py input data of rocksalt MgO (PBE)", NIMS Materials Data Repository, https://mdr.nims.go.jp/datasets/1fd84240-abc0-4d8f-b4d7-1db304d6166b (CC BY 4.0) |
| `MgO_PBEsol_phonondb_band_{nac,nonac}.yaml` | A. Togo, "Ab-initio phonon calculation for MgO / Fm-3m (225) / materials id 1265" (phonondb), NIMS Materials Data Repository, https://mdr.nims.go.jp/concern/datasets/bk128j14x (CC BY 4.0) |

Changes made to the source data: the harmonic force constants were built with phonopy 2.17.2 from each record's
displacements and forces and symmetrised, and the phonon band structure was evaluated along Γ–X–K–Γ–L (51 points per
segment) with the non-analytic term (`*_nac.yaml`, each record's own Born charges and ε∞) and without it
(`*_nonac.yaml`).

`../make_reference_bands.sh` rebuilds the four files from the two downloads and compares them byte for byte with the
released ones; `../data/README.md` gives the download checksums and the sha256 of these files.
