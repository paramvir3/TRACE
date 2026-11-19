#!/usr/bin/env bash
# Every derived data file the released runs train or score on, rebuilt from the third-party downloads, then checked
# against DATA_SHA256.txt by verify_data.py.  Nothing third-party is redistributed with the repository: download the
# raw files named below, and this script derives the rest.
#
# Raw downloads (sha256 and md5 in DATA_SHA256.txt, section "raw"):
#   4G-HDNNP data sets, Ko, Finkler, Goedecker & Behler, Nat. Commun. 12, 398 (2021); Materials Cloud record 2020.137,
#   DOI 10.24435/materialscloud:f3-yh, https://archive.materialscloud.org/record/2020.137 (CC BY 4.0):
#       datasets.tar.gz  ->  $TRACE_4G_RAW/../datasets.tar.gz, unpacked so that $TRACE_4G_RAW/<NAME>/input.data exist
#       ($TRACE_4G_RAW defaults to experiments/benchmarks_4g/raw/datasets; this script unpacks the tarball if it finds
#       it at experiments/benchmarks_4g/raw/datasets.tar.gz, or beside $TRACE_4G_RAW)
#   Togo's phono3py PBE MgO set, NIMS MDR, https://mdr.nims.go.jp/datasets/1fd84240-abc0-4d8f-b4d7-1db304d6166b
#   (CC BY 4.0):
#       phono3py_params.yaml.xz  ->  $TRACE_PHONON_RAW/phono3py_pbe_mgo/
#   Optional (no released run uses it): Togo's PBEsol lattice-thermal-conductivity set for MgO, NIMS MDR,
#   https://mdr.nims.go.jp/datasets/59c5d436-339d-4b32-ba59-a3da5ea68bc7 (CC BY 4.0):
#       phonopy_training_dataset.yaml.xz  ->  $TRACE_PHONON_RAW/mdr_ltc2026_mgo/
#   ($TRACE_PHONON_RAW defaults to experiments/benchmarks_phonons/raw.)  The two reference band files of the phonon
#   scorer are shipped; experiments/benchmarks_phonons/make_reference_bands.sh rebuilds them from the same downloads.
#
# What it writes (all gitignored), in this order:
#   experiments/benchmarks_4g/data/       prepare.py, seed 0: Ag_cluster, Carbon_chain, NaCl 1000/250/1000;
#                                         AuMgO 300/75/400 (written as 300/76/400: the split rounds per stratum)
#   experiments/benchmarks_4g/data_pfn/   prepare.py, seed 0, the PFN split sizes: Ag_cluster and Carbon_chain
#                                         8000/1000/rest, NaCl and AuMgO 4000/500/rest
#   experiments/benchmarks_4g/data_cs/    annotate_charge_states.py on data/AuMgO_* (SPEC 21 charge states)
#   the response priors                  make_au_alpha_prior.py, make_mgo_eps_prior.py (data/),
#                                         make_nacl_eps_prior.py, make_ag_alpha_prior.py (data_pfn/),
#                                         experiments/benchmarks_phonons/make_mgo_eps_prior_dft.py (PBE and PBEsol)
#   experiments/benchmarks_phonons/data/  extract_phono3py_cells.py -> MgO_PBE_togo_displaced.extxyz (148 cells);
#                                         make_bulk_augmented_data.py -> AuMgO_bulkaug_*, with --force-weight 122.4
#                                         -> AuMgO_bulkaugw_* (steps 33 and 34b), with --bulk-only -> MgO_bulkonly_*;
#                                         tools/extract_ltc_pbesol.py -> MgO_PBEsol_ltc_* (only if the optional
#                                         download is present)
# A step whose outputs all exist already is skipped (nothing is overwritten); delete a file to rebuild it.
#
# The recorded commands: the split sizes above were not written down when the data were first made; they were
# recovered by recomputing every split from the raw input.data with prepare.py's own reader and comparing positions,
# energies and forces (all equal to 1e-6; 2026-11-16).  The sha256 in DATA_SHA256.txt are those of the files written
# with ase 3.23.0 and numpy 1.24.2 (Python 3.8.16); another ase may write other bytes for the same numbers, which
# verify_data.py then checks by content.
#
# Usage (any directory):  experiments/benchmarks_4g/prepare_all.sh
#   PYTHON selects the interpreter (default python3); it needs numpy, ase and PyYAML.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
B4G="$REPO/experiments/benchmarks_4g"
PH="$REPO/experiments/benchmarks_phonons"
RAW4G="${TRACE_4G_RAW:-$B4G/raw/datasets}"
RAWPH="${TRACE_PHONON_RAW:-$PH/raw}"
PY="${PYTHON:-python3}"

sha256() { "$PY" -I -c 'import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
print(h.hexdigest())' "$1"; }

check_sha() {  # check_sha FILE EXPECTED
  local got; got="$(sha256 "$1")"
  if [ "$got" != "$2" ]; then echo "sha256 mismatch for $1: $got, expected $2" >&2; exit 1; fi
  echo "ok  sha256 $1"
}

all_exist() { local f; for f in "$@"; do [ -f "$f" ] || return 1; done; return 0; }

# xz_checked DIR NAME SHA_XZ SHA_YAML: verify a NIMS MDR download, unpack it beside itself, verify the yaml
xz_checked() {
  local dir="$1" name="$2"
  if [ ! -f "$dir/$name.yaml" ]; then
    [ -f "$dir/$name.yaml.xz" ] || { echo "missing $dir/$name.yaml.xz (see the top of this script)" >&2; return 1; }
    check_sha "$dir/$name.yaml.xz" "$3"
    "$PY" -I -c 'import lzma, sys; open(sys.argv[2], "wb").write(lzma.open(sys.argv[1]).read())' \
      "$dir/$name.yaml.xz" "$dir/$name.yaml"
  fi
  check_sha "$dir/$name.yaml" "$4"
}

# ---------------------------------------------------------------- 4G raw data
if [ ! -f "$RAW4G/AuMgO/input.data" ]; then
  TARBALL=""
  for t in "$(dirname "$RAW4G")/datasets.tar.gz" "$B4G/raw/datasets.tar.gz"; do [ -f "$t" ] && { TARBALL="$t"; break; }; done
  if [ -z "$TARBALL" ]; then
    echo "missing $RAW4G/<NAME>/input.data and datasets.tar.gz: download datasets.tar.gz from" >&2
    echo "https://archive.materialscloud.org/record/2020.137 into $B4G/raw/ (or set TRACE_4G_RAW)" >&2
    exit 1
  fi
  check_sha "$TARBALL" 6194ee9e23b446f17c3e8b0b754fa9a9e9498b1b099231015ea10a6a313b9775
  mkdir -p "$(dirname "$RAW4G")"
  # the archive holds datasets/<NAME>/input.data
  tar -xzf "$TARBALL" -C "$(dirname "$RAW4G")" datasets/Ag_cluster/input.data datasets/AuMgO/input.data \
    datasets/Carbon_chain/input.data datasets/NaCl/input.data
  if [ "$(basename "$RAW4G")" != "datasets" ]; then
    echo "unpacked into $(dirname "$RAW4G")/datasets; set TRACE_4G_RAW to that directory" >&2; exit 1
  fi
fi
check_sha "$RAW4G/Ag_cluster/input.data"   7eaede1efc2e4dbf4f3330961e92f7e838e1dcc023d64ee82bd77fc68ad6519b
check_sha "$RAW4G/AuMgO/input.data"        a741c9b9a88c9bb593121d779a4dce3645ed4cbca3806d13c422025ffb24c9fc
check_sha "$RAW4G/Carbon_chain/input.data" 8e8b445f12ca241c2b42176bc7ab7fd8d35d98edcb2d47974c45195ed727ba79
check_sha "$RAW4G/NaCl/input.data"         174f9eaa1d537f91e996ccf89aa4420b5e2a4682b1bf3ddd348c3eccc2f81a22

cd "$B4G"

# ---------------------------------------------------------------- data/ and data_pfn/ (prepare.py; seed 0)
# prepare.py DATASETS_DIR NAME N_TRAIN N_VALID N_TEST [SEED] [OUT_DIR]; the downloads are read with python -I
split() {  # split OUT_DIR NAME N_TRAIN N_VALID N_TEST
  local out="$1" name="$2"
  if all_exist "$out/${name}_train.xyz" "$out/${name}_valid.xyz" "$out/${name}_test.xyz"; then
    echo "kept $out/${name}_{train,valid,test}.xyz (exist)"; return
  fi
  mkdir -p "$out"
  "$PY" -I prepare.py "$RAW4G" "$name" "$3" "$4" "$5" 0 "$out"
}
split "$B4G/data" Ag_cluster   1000 250 1000
split "$B4G/data" Carbon_chain 1000 250 1000
split "$B4G/data" NaCl         1000 250 1000
split "$B4G/data" AuMgO        300  75  400
split "$B4G/data_pfn" Ag_cluster   8000 1000 -1
split "$B4G/data_pfn" Carbon_chain 8000 1000 -1
split "$B4G/data_pfn" NaCl         4000 500  -1
split "$B4G/data_pfn" AuMgO        4000 500  -1

# ---------------------------------------------------------------- data_cs/ (SPEC 21 charge states on data/AuMgO_*)
if all_exist data_cs/AuMgO_train.xyz data_cs/AuMgO_valid.xyz data_cs/AuMgO_test.xyz; then
  echo "kept data_cs/AuMgO_{train,valid,test}.xyz (exist)"
else
  "$PY" annotate_charge_states.py data data_cs
fi

# ---------------------------------------------------------------- response priors (our constructions from cited numbers)
# each writes a path relative to experiments/benchmarks_4g, so they run from here
prior() {  # prior OUTPUT SCRIPT
  if [ -f "$1" ]; then echo "kept $1 (exists)"; else "$PY" "$2"; fi
}
prior data/Au_atom_alpha_prior.xyz       make_au_alpha_prior.py    # alpha(Au) = 36.68 a.u., arXiv:2410.01493
prior data/MgO_bulk_eps_prior.xyz        make_mgo_eps_prior.py     # eps_inf(MgO) = 2.95, arXiv:0704.2267 (steps 26, 31)
prior data_pfn/NaCl_bulk_eps_prior.xyz   make_nacl_eps_prior.py    # eps_inf(NaCl) = 2.3295, Li 1976 (step 9, NaCl)
prior data_pfn/Ag_atom_alpha_prior.xyz   make_ag_alpha_prior.py    # alpha(Ag) = 50.97 a.u., arXiv:2410.01493 (step 9, Ag3)
if all_exist "$PH/data/MgO_bulk_eps_prior_pbe.xyz" "$PH/data/MgO_bulk_eps_prior_pbesol.xyz"; then
  echo "kept $PH/data/MgO_bulk_eps_prior_{pbe,pbesol}.xyz (exist)"
else
  # it refuses to overwrite either file, so it runs only when both are absent
  for f in "$PH/data/MgO_bulk_eps_prior_pbe.xyz" "$PH/data/MgO_bulk_eps_prior_pbesol.xyz"; do
    [ -f "$f" ] && { echo "$f exists but its partner does not; move it away and rerun" >&2; exit 1; }
  done
  mkdir -p "$PH/data"
  "$PY" "$PH/make_mgo_eps_prior_dft.py"    # PBE eps_inf 3.235 (steps 33, 34b), PBEsol 3.2417
fi

# ---------------------------------------------------------------- phonon-derived training sets
xz_checked "$RAWPH/phono3py_pbe_mgo" phono3py_params \
  13b12ff2f0c16d5db00ae14ee6d32b667d4ba0e3932ddd03522bb83f81d7adce d22ebbc2aec853c192204aac1474f2e586e187169b02d641254424d1306ec1df
mkdir -p "$PH/data"
if [ -f "$PH/data/MgO_PBE_togo_displaced.extxyz" ]; then
  echo "kept $PH/data/MgO_PBE_togo_displaced.extxyz (exists)"
else
  "$PY" -I "$PH/extract_phono3py_cells.py" "$RAWPH/phono3py_pbe_mgo/phono3py_params.yaml" "$PH/data/MgO_PBE_togo_displaced.extxyz"
fi
bulk() {  # bulk NAME [FLAGS...]
  local name="$1"; shift
  if all_exist "$PH/data/${name}_train.xyz" "$PH/data/${name}_valid.xyz"; then
    echo "kept $PH/data/${name}_{train,valid}.xyz (exist)"
  else
    "$PY" "$PH/make_bulk_augmented_data.py" "$@"
  fi
}
bulk AuMgO_bulkaug                         # 300 + 136 bulk cells / 76 + 10
bulk AuMgO_bulkaugw --force-weight 122.4   # the same, info force_weight = 122.4 on the bulk cells (steps 33, 34b)
bulk MgO_bulkonly --bulk-only              # the 136 / 10 bulk cells alone

LTC="$RAWPH/mdr_ltc2026_mgo"
if [ -f "$LTC/phonopy_training_dataset.yaml" ] || [ -f "$LTC/phonopy_training_dataset.yaml.xz" ]; then
  xz_checked "$LTC" phonopy_training_dataset \
    49355412ced3ebff17be23dcf778cf964b1692b737faf6f2aee7ced22b5b39b0 9835a5f10ecdb8881874c8097ce335605c84779eb69fa69077702281cb29691b
  if all_exist "$PH/data/MgO_PBEsol_ltc_train.xyz" "$PH/data/MgO_PBEsol_ltc_valid.xyz"; then
    echo "kept $PH/data/MgO_PBEsol_ltc_{train,valid}.xyz (exist)"
  else
    "$PY" -I "$PH/tools/extract_ltc_pbesol.py" "$LTC/phonopy_training_dataset.yaml" "$PH/data/MgO_PBEsol_ltc"
  fi
else
  echo "skipped MgO_PBEsol_ltc_* (optional download absent: $LTC/phonopy_training_dataset.yaml.xz)"
fi

# ---------------------------------------------------------------- check every file
"$PY" "$B4G/verify_data.py"
