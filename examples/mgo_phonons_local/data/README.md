# Data (not committed)

`MgO_PBEsol_ltc_{train,valid}.xyz` are made from A. Togo's PBEsol training set for MgO,
NIMS Materials Data Repository, https://mdr.nims.go.jp/datasets/59c5d436-339d-4b32-ba59-a3da5ea68bc7 (CC BY 4.0):
download `phonopy_training_dataset.yaml.xz`, decompress it, and run (repo root)

    python examples/mgo_phonons_local/extract_ltc_pbesol.py phonopy_training_dataset.yaml examples/mgo_phonons_local/data/MgO_PBEsol_ltc

(VASP PBEsol, 200 rocksalt 2×2×2 supercells of 64 atoms at a = 4.21726 Å, every atom displaced 0.03 Å; 180 for
training and 20 for validation, split with numpy seed 0).
